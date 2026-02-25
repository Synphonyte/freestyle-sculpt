use glam::Vec3;
use hashbrown::{HashMap, HashSet};
use itertools::Itertools;
use mesh_graph::{MeshGraph, VertexId};
use parry3d::query::PointQueryWithLocation;
use tracing::{error, instrument};

use crate::deformation::{TopologyManager, cleanup_mesh};
use crate::selectors::WeightedSelection;
use crate::{ray::FaceIntersection, selectors::MeshSelector};

use crate::SculptParams;

/// Trait for deformation fields.
///
/// It describes how vertices should be moved based on factors like
/// pointer position, selection, pointer movement and strength.
pub trait DeformationField {
    /// Returns the movement vector for the given vertex.
    fn vertex_movement(&self, vertex: VertexId, position: Vec3, mesh_graph: &MeshGraph) -> Vec3;

    /// Called when the pointer is pressed.
    ///
    /// The parameter `face_intersection` is the intersection of the pointer with the mesh.
    fn on_pointer_down(&mut self, _face_intersection: FaceIntersection) {
        // by default, do nothing
    }

    /// Called when the pointer is moved.
    ///
    /// Parameters:
    /// - `pointer_translation` is the translation of the pointer in 3D space.
    /// - `face_intersection` is the intersection of the pointer with the mesh.
    ///
    /// It returns true if the deformation should be applied after this.
    fn on_pointer_move(
        &mut self,
        _mesh_graph: &MeshGraph,
        _pointer_translation: Vec3,
        _face_intersection: Option<FaceIntersection>,
    ) -> bool {
        // by default, do nothing
        true
    }

    /// Returns the latest face intersection.
    fn intersection(&self) -> Option<&FaceIntersection>;

    fn update_intersection(&mut self, mesh_graph: &MeshGraph);

    /// This computes the maximum vertex movement of all the affected vertices.
    /// Used to determine the number of steps needed to apply the deformation.
    #[instrument(skip_all)]
    fn max_movement_squared(
        &self,
        mesh_graph: &MeshGraph,
        selector: &dyn MeshSelector,
        strength: f32,
    ) -> f32 {
        let WeightedSelection { vertex_to_weight } =
            selector.select(mesh_graph, self.intersection().unwrap());

        let mut max_movement_squared: f32 = 0.0;

        for (&v_id, &weight) in &vertex_to_weight {
            let Some(&pos) = mesh_graph.positions.get(v_id) else {
                error!("Vertex position not found");
                continue;
            };

            let movement = self.vertex_movement(v_id, pos, mesh_graph) * weight * strength;

            max_movement_squared = max_movement_squared.max(movement.length_squared());
        }

        max_movement_squared
    }

    /// This is the main method of this trait. It applies the deformation to the mesh graph.
    ///
    /// This method should be called after `on_pointer_move` returns `true`.
    #[instrument(skip_all)]
    fn apply(
        &mut self,
        mesh_graph: &mut MeshGraph,
        selector: &dyn MeshSelector,
        strength: f32,
        params: SculptParams,
        topology_manager: &mut TopologyManager,
    ) {
        let max_movement_squared = self.max_movement_squared(mesh_graph, selector, strength);

        let steps = (max_movement_squared / params.max_move_dist_squared)
            .sqrt()
            .ceil()
            .max(1.0);

        let factor = 1.0 / steps;

        #[cfg(feature = "rerun")]
        {
            mesh_graph.log_rerun();
        }

        mesh_graph.collapse_until_edges_above_min_length(
            params.min_edge_length_squared,
            &mut topology_manager.protected_vertices,
        );

        mesh_graph.subdivide_until_edges_below_max_length(
            params.max_edge_length_squared,
            &mut topology_manager.protected_halfedges,
            &mut topology_manager.protected_vertices,
        );

        let mut vertex_to_weight = HashMap::new();

        for _ in 0..steps as usize {
            mesh_graph.optimize_bvh_incremental();

            self.update_intersection(mesh_graph);

            vertex_to_weight = selector
                .select(mesh_graph, self.intersection().unwrap())
                .vertex_to_weight;

            for (&v_id, &weight) in &vertex_to_weight {
                let Some(&pos) = mesh_graph.positions.get(v_id) else {
                    error!("Vertex position not found");
                    continue;
                };

                let movement = self.vertex_movement(v_id, pos, mesh_graph) * weight * strength;

                // already checked above
                mesh_graph.positions[v_id] = pos + movement * factor;
            }

            cleanup_mesh(mesh_graph, &params, topology_manager);
        }

        let mut affected_face_ids = HashSet::new();

        self.update_intersection(mesh_graph);

        vertex_to_weight = selector
            .select(mesh_graph, self.intersection().unwrap())
            .vertex_to_weight;

        for &v_id in vertex_to_weight.keys() {
            // just check above
            let v = mesh_graph.vertices[v_id];

            affected_face_ids.extend(v.faces(mesh_graph));

            mesh_graph
                .vertex_normals
                .as_mut()
                .unwrap()
                .insert(v_id, Vec3::ZERO);
        }

        for face_id in affected_face_ids {
            let Some(f) = mesh_graph.faces.get(face_id) else {
                error!("Face not found");
                continue;
            };

            if let Some(face_normal) = f.normal(mesh_graph) {
                for v_id in f.vertices(mesh_graph).collect_vec() {
                    if vertex_to_weight.contains_key(&v_id) {
                        mesh_graph.vertex_normals.as_mut().unwrap()[v_id] += face_normal;
                    }
                }
            } else {
                error!("Couldn't compute Face normal");
            };

            mesh_graph
                .bvh
                .insert_or_update_partially(f.aabb(mesh_graph), f.index, 0.0);
        }

        for &v_id in vertex_to_weight.keys() {
            mesh_graph.vertex_normals.as_mut().unwrap()[v_id] =
                mesh_graph.vertex_normals.as_ref().unwrap()[v_id].normalize()
        }

        mesh_graph.refit_bvh();
    }
}

/// Makes sure that the face of the intersection is up to date. If it has been deleted, the previous point is projected onto the
/// mesh_graph to find the closest face.
pub fn update_face_intersection(intersection: &mut FaceIntersection, mesh_graph: &MeshGraph) {
    if let Some(face) = mesh_graph.faces.get(intersection.face.id) {
        intersection.face = *face;
    } else if let Some((_, face)) = mesh_graph.project_local_point_and_get_location_with_max_dist(
        intersection.point,
        true,
        f32::MAX,
    ) {
        intersection.face = face;
    }
}

#[macro_export]
macro_rules! impl_deformation_field_boilerplate {
    () => {
        #[inline(always)]
        fn intersection(&self) -> Option<&FaceIntersection> {
            self.intersection.as_ref()
        }

        #[inline(always)]
        fn update_intersection(&mut self, mesh_graph: &MeshGraph) {
            if let Some(intersection) = self.intersection.as_mut() {
                update_face_intersection(intersection, mesh_graph);
            }
        }
    };
}
