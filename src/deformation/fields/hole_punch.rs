use glam::{Quat, Vec3, Vec3Swizzles};
use hashbrown::HashSet;
use mesh_graph::{MeshGraph, Polygon2};
use parry3d::bounding_volume::Aabb;
use tracing::{error, instrument};

use crate::{
    SculptParams,
    deformation::{TopologyManager, cleanup_mesh},
};

/// Punches a hole with the given shape in the direction of the given vector.
#[derive(Debug, Clone)]
pub struct HolePunch {
    hole_shape: Polygon2,
    punch_direction: Vec3,
}

impl HolePunch {
    pub fn new(hole_shape: Polygon2, punch_direction: Vec3) -> Self {
        Self {
            hole_shape,
            punch_direction: punch_direction.normalize(),
        }
    }

    #[instrument(skip(mesh_graph, topology_manager))]
    pub fn apply(
        &self,
        mesh_graph: &mut MeshGraph,
        params: &SculptParams,
        topology_manager: &mut TopologyManager,
    ) {
        let neg_max_move_dist = -params.max_move_dist_squared.sqrt();

        mesh_graph.collapse_until_edges_above_min_length(
            params.min_edge_length_squared,
            &mut topology_manager.protected_vertices,
        );

        mesh_graph.subdivide_until_edges_below_max_length(
            params.max_edge_length_squared,
            &mut topology_manager.protected_halfedges,
            &mut topology_manager.protected_vertices,
        );

        let quat = Quat::from_rotation_arc(self.punch_direction, Vec3::Z);
        mesh_graph.apply_quat(quat);
        mesh_graph.rebuild_bvh();

        let (min, max) = self.hole_shape.min_max();

        let aabb = Aabb::new(min.extend(f32::MIN), max.extend(f32::MAX));

        loop {
            let mut vertex_ids = HashSet::new();

            for face_idx in mesh_graph.bvh.intersect_aabb(&aabb) {
                let face_id = mesh_graph.index_to_face_id[&face_idx];

                let Some(face) = mesh_graph.faces.get(face_id) else {
                    error!("Face not found");
                    continue;
                };

                for v_id in face.vertices(mesh_graph) {
                    vertex_ids.insert(v_id);
                }
            }

            if vertex_ids.is_empty() {
                return;
            }

            for v_id in vertex_ids {
                let Some(pos) = mesh_graph.positions.get(v_id) else {
                    error!("Vertex position not found");
                    continue;
                };

                if self.hole_shape.contains_point(pos.xy()) {
                    let Some(&normal) = mesh_graph.vertex_normals.as_ref().unwrap().get(v_id)
                    else {
                        error!("Vertex normal not found");
                        continue;
                    };

                    let move_vec = normal * neg_max_move_dist;

                    // already checked for existence above
                    mesh_graph.positions[v_id] += move_vec;
                }
            }

            cleanup_mesh(mesh_graph, params, topology_manager);
        }
    }
}
