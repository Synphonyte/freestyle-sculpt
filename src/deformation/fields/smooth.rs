use glam::Vec3;
use mesh_graph::{MeshGraph, VertexId, error_none};
use tracing::instrument;

use crate::{
    deformation::{DeformationField, update_face_intersection},
    impl_deformation_field_boilerplate,
    ray::FaceIntersection,
    selectors::MeshSelector,
};

/// Smoothing deformation field.
///
/// This deformation field applies a smoothing effect to the selected vertices.
/// It calculates the average position of the surrounding vertices of every selected vertex and moves it towards this average.
#[derive(Copy, Clone)]
pub struct SmoothDeformation {
    intersection: Option<FaceIntersection>,
    pub strength: f32,
}

impl Default for SmoothDeformation {
    fn default() -> Self {
        Self {
            intersection: None,
            strength: 0.1,
        }
    }
}

impl SmoothDeformation {
    pub fn new(strength: f32) -> Self {
        Self {
            intersection: None,
            strength,
        }
    }
}

impl DeformationField for SmoothDeformation {
    fn on_pointer_move(
        &mut self,
        _mesh_graph: &MeshGraph,
        _mouse_translation: Vec3,
        face_intersection: Option<FaceIntersection>,
    ) -> bool {
        self.intersection = face_intersection;
        self.intersection.is_some()
    }

    fn max_movement_squared(
        &self,
        _mesh_graph: &MeshGraph,
        _selector: &dyn MeshSelector,
        _strength: f32,
    ) -> f32 {
        // always return 0.0 because this operation can't lead to invalid topology so we don't need to subdivide it's movements
        0.0
    }

    #[instrument(skip(self, mesh_graph))]
    fn vertex_movement(&self, vertex: VertexId, position: Vec3, mesh_graph: &MeshGraph) -> Vec3 {
        let mut movement = Vec3::ZERO;

        let neighbours = mesh_graph
            .vertices
            .get(vertex)
            .map(|v| v.neighbours(mesh_graph).collect::<Vec<_>>())
            .or_else(error_none!("Vertex not found"))
            .unwrap_or_default();

        for neighbour in &neighbours {
            movement += mesh_graph
                .positions
                .get(*neighbour)
                .or_else(error_none!("Neighbour position not found"))
                .unwrap_or(&Vec3::ZERO);
        }

        if neighbours.is_empty() {
            return Vec3::ZERO;
        }

        movement /= neighbours.len() as f32;

        (movement - position) * self.strength
    }

    impl_deformation_field_boilerplate!();

    fn allow_topology_change(&self) -> bool {
        false
    }
}
