use glam::Vec3;
use mesh_graph::MeshGraph;

use crate::{
    deformation::{DeformationField, update_face_intersection},
    impl_deformation_field_boilerplate,
    ray::FaceIntersection,
    selectors::MeshSelector,
};

/// Erode/Dilate deformation field.
///
/// This deformation field moves every affected vertex along it's normal vector by the specified `amount`.
/// Positive `amount` values dilate the mesh, while negative values erode it.
#[derive(Copy, Clone)]
pub struct ErodeDilateDeformation {
    intersection: Option<FaceIntersection>,
    pub amount: f32,
}

impl ErodeDilateDeformation {
    pub fn new(amount: f32) -> Self {
        Self {
            intersection: None,
            amount,
        }
    }
}

impl DeformationField for ErodeDilateDeformation {
    fn on_pointer_move(
        &mut self,
        _mesh_graph: &MeshGraph,
        _mouse_translation: Vec3,
        face_intersection: Option<FaceIntersection>,
    ) -> bool {
        self.intersection = face_intersection;
        self.intersection.is_some()
    }

    #[inline(always)]
    fn max_movement_squared(
        &self,
        _mesh_graph: &MeshGraph,
        _selector: &dyn MeshSelector,
        _strength: f32,
        _intersection: &FaceIntersection,
    ) -> f32 {
        self.amount * self.amount
    }

    fn vertex_movement(
        &self,
        vertex: mesh_graph::VertexId,
        _position: glam::Vec3,
        mesh_graph: &mesh_graph::MeshGraph,
    ) -> Vec3 {
        if let Some(normals) = &mesh_graph.vertex_normals
            && let Some(normal) = normals.get(vertex)
        {
            normal * self.amount
        } else {
            Vec3::ZERO
        }
    }

    impl_deformation_field_boilerplate!();
}
