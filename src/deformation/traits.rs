use glam::Vec3;
use hashbrown::HashSet;
use mesh_graph::{HalfedgeId, MeshGraph, Selection, VertexId, error_none};
use tracing::{error, instrument};

use crate::{ray::FaceIntersection, selectors::MeshSelector};

use crate::SculptParams;

/// Trait for deformation fields.
///
/// It describes how vertices should be moved based on factors like
/// pointer position, selection, pointer movement and strength.
pub trait DeformationField {
    /// Returns the movement vector for the given vertex.
    fn vertex_movement(&self, vertex: VertexId, mesh_graph: &MeshGraph) -> Vec3;

    /// Called when the pointer is pressed.
    ///
    /// The parameter `face_intersection` is the intersection of the pointer with the mesh.
    fn on_pointer_down(
        &mut self,
        _mesh_graph: &MeshGraph,
        _selector: &dyn MeshSelector,
        _face_intersection: FaceIntersection,
    ) {
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
        _selector: &dyn MeshSelector,
        _pointer_translation: Vec3,
        _face_intersection: Option<FaceIntersection>,
    ) -> bool {
        // by default, do nothing
        true
    }

    /// Return the current selection usually updated by the selector given to `on_pointer_down` and `on_pointer_move`.
    fn selection(&self) -> &Selection;

    /// Same as `selection` but mutable.
    fn selection_mut(&mut self) -> &mut Selection;

    /// Return the current weight callback usually updated by the selector given to `on_pointer_down` and `on_pointer_move`.
    fn weight_callback(&self) -> &dyn Fn(Vec3) -> f32;

    /// This computes the maximum vertex movement of all the affected vertices.
    /// Used to determine the number of steps needed to apply the deformation.
    #[instrument(skip(self, mesh_graph))]
    fn max_movement_squared(&self, mesh_graph: &MeshGraph, strength: f32) -> f32 {
        let affected_vertices = self.selection().resolve_to_vertices(mesh_graph);
        let get_weight = self.weight_callback();

        let mut max_movement_squared: f32 = 0.0;

        for vertex in &affected_vertices {
            let weight = mesh_graph
                .positions
                .get(*vertex)
                .or_else(error_none!("Vertex position not found"))
                .copied()
                .map(get_weight)
                .unwrap_or_default();

            let movement = self.vertex_movement(*vertex, mesh_graph) * weight * strength;
            max_movement_squared = max_movement_squared.max(movement.length_squared());
        }

        max_movement_squared
    }

    /// This is the main method of this trait. It applies the deformation to the mesh graph.
    ///
    /// This method should be called after `on_pointer_move` returns `true`.
    #[instrument(skip(self, mesh_graph))]
    fn apply(
        &mut self,
        mesh_graph: &mut MeshGraph,
        strength: f32,
        params: SculptParams,
        protected_halfedges: &mut HashSet<HalfedgeId>,
        protected_vertices: &mut HashSet<VertexId>,
    ) {
        let max_movement_squared = self.max_movement_squared(mesh_graph, strength);

        let steps = (max_movement_squared / params.max_move_dist_squared)
            .sqrt()
            .ceil();

        let factor = 1.0 / steps;

        let selection = self.selection_mut();

        #[cfg(feature = "rerun")]
        {
            mesh_graph.log_rerun();
        }

        mesh_graph.collapse_until_edges_above_min_length(
            params.min_edge_length_squared,
            selection,
            protected_vertices,
        );

        mesh_graph.subdivide_until_edges_below_max_length(
            params.max_edge_length_squared,
            selection,
            protected_halfedges,
            protected_vertices,
        );

        let mut movements = Vec::new();

        for _ in 0..steps as usize {
            let affected_vertices = self.selection().resolve_to_vertices(mesh_graph);
            movements.clear();

            let get_weight = self.weight_callback();

            for vertex in &affected_vertices {
                let weight = mesh_graph
                    .positions
                    .get(*vertex)
                    .or_else(error_none!("Vertex position not found"))
                    .copied()
                    .map(get_weight)
                    .unwrap_or_default();

                let movement = self.vertex_movement(*vertex, mesh_graph) * weight * strength;
                movements.push(movement);
            }

            let selection = self.selection_mut();

            for (vertex, movement) in affected_vertices.iter().zip(movements.iter()) {
                if let Some(pos) = mesh_graph.positions.get_mut(*vertex) {
                    *pos += *movement * factor;
                } else {
                    error!("Position not found");
                }
            }

            #[cfg(feature = "rerun")]
            {
                mesh_graph.log_rerun();
            }

            mesh_graph.collapse_until_edges_above_min_length(
                params.min_edge_length_squared,
                selection,
                protected_vertices,
            );

            mesh_graph.subdivide_until_edges_below_max_length(
                params.max_edge_length_squared,
                selection,
                protected_halfedges,
                protected_vertices,
            );

            // TODO : merging and separation and cleanup
        }

        mesh_graph.refit_bvh();
    }
}
