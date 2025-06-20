use glam::Vec3;

use mesh_graph::{Face, MeshGraph, Selection};

/// Trait for selecting a part of the mesh graph for deformation fields to be applied to.
pub trait MeshSelector {
    fn select(
        &self,
        mesh_graph: &MeshGraph,
        input_pos: Vec3,
        input_face: Face,
    ) -> WeightedSelection;
}

/// Returned by the `MeshSelector::select` method. Represents a mesh selection with associated weights per vertex.
pub struct WeightedSelection {
    pub selection: Selection,
    pub get_weight: Box<dyn Fn(Vec3) -> f32>,
}

/// Used to calculate the distance squared between two points to determine the inclusion and weight of a vertex on the selection.
pub trait DistanceCalculator {
    fn distance_squared(&self, a: Vec3, b: Vec3) -> f32;
}
