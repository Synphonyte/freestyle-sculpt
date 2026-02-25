use glam::Vec3;
use mesh_graph::MeshGraph;
use parry3d::query::PointQueryWithLocation;

use crate::{
    deformation::{DeformationField, update_face_intersection},
    impl_deformation_field_boilerplate,
    ray::FaceIntersection,
    selectors::MeshSelector,
};

/// Translation deformation field.
///
/// This deformation field translates vertices based on the pointer movement.
#[derive(Default)]
pub struct TranslateDeformation {
    translation: Vec3,
    point: Vec3,
    intersection: Option<FaceIntersection>,
}

impl TranslateDeformation {
    pub fn new() -> Self {
        Self::default()
    }
}

impl DeformationField for TranslateDeformation {
    #[inline(always)]
    fn on_pointer_down(&mut self, face_intersection: FaceIntersection) {
        self.point = face_intersection.point;
    }

    fn on_pointer_move(
        &mut self,
        mesh_graph: &MeshGraph,
        mouse_translation: Vec3,
        face_intersection: Option<FaceIntersection>,
    ) -> bool {
        self.translation = mouse_translation;

        self.point += mouse_translation;

        if let Some(face_intersection) = face_intersection {
            self.intersection = Some(face_intersection);
        } else if let Some((_, face)) = mesh_graph
            .project_local_point_and_get_location_with_max_dist(self.point, true, f32::MAX)
        {
            self.intersection = Some(FaceIntersection {
                face,
                point: self.point,
            });
        } else {
            self.intersection = None;

            return false;
        };

        #[cfg(feature = "rerun")]
        {
            mesh_graph::RR
                .log(
                    "translate/on_pointer_move/self_point",
                    &rerun::Points3D::new([mesh_graph::utils::vec3_array(
                        self.intersection.unwrap().point,
                    )]),
                )
                .unwrap();
        }

        true
    }

    #[inline(always)]
    fn max_movement_squared(
        &self,
        _mesh_graph: &MeshGraph,
        _selector: &dyn MeshSelector,
        strength: f32,
    ) -> f32 {
        self.translation.length_squared() * strength
    }

    #[inline(always)]
    fn vertex_movement(
        &self,
        _vertex: mesh_graph::VertexId,
        _pos: Vec3,
        _mesh_graph: &mesh_graph::MeshGraph,
    ) -> Vec3 {
        self.translation
    }

    impl_deformation_field_boilerplate!();
}
