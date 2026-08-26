use glam::Vec3;
use mesh_graph::{MeshGraph, VertexId};

use crate::{deformation::DeformationField, ray::FaceIntersection, selectors::MeshSelector};

/// Applies the sum of all contained deformation fields
pub struct CompoundDeformation<const C: usize> {
    fields: [Box<dyn DeformationField>; C],
}

impl<const C: usize> CompoundDeformation<C> {
    pub fn new(fields: [Box<dyn DeformationField>; C]) -> Self {
        const {
            assert!(
                C > 0,
                "CompoundDeformation requires at least one deformation field"
            )
        };
        CompoundDeformation { fields }
    }
}

impl<const C: usize> DeformationField for CompoundDeformation<C> {
    fn vertex_movement(&self, vertex: VertexId, position: Vec3, mesh_graph: &MeshGraph) -> Vec3 {
        let mut movement = Vec3::ZERO;

        for field in &self.fields {
            movement += field.vertex_movement(vertex, position, mesh_graph);
        }

        movement
    }

    #[inline(always)]
    fn intersection(&self) -> Option<&FaceIntersection> {
        // the other fields should yield the same intersections. For performance,
        // we only use the first field's intersection.
        self.fields[0].intersection()
    }

    fn update_intersection(&mut self, mesh_graph: &MeshGraph) {
        for field in &mut self.fields {
            field.update_intersection(mesh_graph);
        }
    }

    fn on_pointer_down(&mut self, face_intersection: FaceIntersection) {
        for field in &mut self.fields {
            field.on_pointer_down(face_intersection);
        }
    }

    fn on_pointer_move(
        &mut self,
        mesh_graph: &MeshGraph,
        pointer_translation: Vec3,
        face_intersection: Option<FaceIntersection>,
    ) -> bool {
        let mut compound_result = false;

        for field in &mut self.fields {
            let field_result =
                field.on_pointer_move(mesh_graph, pointer_translation, face_intersection);

            compound_result = compound_result || field_result;
        }

        compound_result
    }

    fn max_movement_squared(
        &self,
        mesh_graph: &MeshGraph,
        selector: &dyn MeshSelector,
        strength: f32,
        intersection: &FaceIntersection,
    ) -> f32 {
        let mut movement = 0.0;

        for field in &self.fields {
            movement += field
                .max_movement_squared(mesh_graph, selector, strength, intersection)
                .sqrt()
        }

        movement * movement
    }
}
