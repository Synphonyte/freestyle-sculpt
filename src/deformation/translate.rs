use glam::Vec3;
use mesh_graph::{MeshGraph, Selection};
use parry3d::query::PointQueryWithLocation;

use crate::{
    ray::FaceIntersection,
    selectors::{MeshSelector, WeightedSelection},
};

use super::DeformationField;

/// Translation deformation field.
///
/// This deformation field translates vertices based on the pointer movement.
pub struct TranslateDeformation {
    selection: Selection,
    weight_callback: Box<dyn Fn(Vec3) -> f32>,
    translation: Vec3,
    point: Vec3,
}

impl Default for TranslateDeformation {
    fn default() -> Self {
        Self {
            selection: Selection::default(),
            weight_callback: Box::new(|_| 1.0),
            translation: Vec3::ZERO,
            point: Vec3::ZERO,
        }
    }
}

impl DeformationField for TranslateDeformation {
    fn on_pointer_down(
        &mut self,
        mesh_graph: &MeshGraph,
        selector: &dyn MeshSelector,
        face_intersection: FaceIntersection,
    ) {
        WeightedSelection {
            selection: self.selection,
            get_weight: self.weight_callback,
        } = selector.select(mesh_graph, face_intersection.point, face_intersection.face);

        #[cfg(feature = "rerun")]
        mesh_graph.log_selection_rerun("translate/on_pointer_down", &self.selection);

        self.point = face_intersection.point;

        #[cfg(feature = "rerun")]
        mesh_graph::RR
            .log(
                "translate/on_pointer_down/self_point",
                &rerun::Points3D::new([mesh_graph::utils::vec3_array(self.point)]),
            )
            .unwrap();
    }

    fn on_pointer_move(
        &mut self,
        mesh_graph: &MeshGraph,
        selector: &dyn MeshSelector,
        mouse_translation: Vec3,
        face_intersection: Option<FaceIntersection>,
    ) -> bool {
        self.translation = mouse_translation;

        self.point += mouse_translation;

        let (face, point) = if let Some(face_intersection) = face_intersection {
            (face_intersection.face, face_intersection.point)
        } else if let Some((_, face)) = mesh_graph
            .project_local_point_and_get_location_with_max_dist(self.point, true, f32::MAX)
        {
            (face, self.point)
        } else {
            return false;
        };

        WeightedSelection {
            selection: self.selection,
            get_weight: self.weight_callback,
        } = selector.select(mesh_graph, point, face);

        #[cfg(feature = "rerun")]
        {
            mesh_graph.log_selection_rerun("translate/on_pointer_move", &self.selection);

            mesh_graph::RR
                .log(
                    "translate/on_pointer_move/self_point",
                    &rerun::Points3D::new([mesh_graph::utils::vec3_array(point)]),
                )
                .unwrap();
        }

        true
    }

    #[inline(always)]
    fn max_movement_squared(&self, _mesh_graph: &MeshGraph, strength: f32) -> f32 {
        self.translation.length_squared() * strength
    }

    #[inline(always)]
    fn vertex_movement(
        &self,
        _vertex: mesh_graph::VertexId,
        _mesh_graph: &mesh_graph::MeshGraph,
    ) -> glam::Vec3 {
        self.translation
    }

    #[inline(always)]
    fn selection(&self) -> &Selection {
        &self.selection
    }

    #[inline(always)]
    fn selection_mut(&mut self) -> &mut Selection {
        &mut self.selection
    }

    #[inline(always)]
    fn weight_callback(&self) -> &dyn Fn(Vec3) -> f32 {
        self.weight_callback.as_ref()
    }
}
