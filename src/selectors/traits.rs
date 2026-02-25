use hashbrown::HashMap;
use mesh_graph::{MeshGraph, VertexId};

use crate::ray::FaceIntersection;

/// Trait for selecting a part of the mesh graph for deformation fields to be applied to.
pub trait MeshSelector {
    fn select(
        &self,
        mesh_graph: &MeshGraph,
        face_intersection: &FaceIntersection,
    ) -> WeightedSelection;
}

/// Returned by the `MeshSelector::select` method. Represents a mesh selection with associated weights per vertex.
pub struct WeightedSelection {
    pub vertex_to_weight: HashMap<VertexId, f32>,
}
