use hashbrown::{HashMap, HashSet};

use mesh_graph::MeshGraph;
use tracing::{error, instrument};

use crate::ray::FaceIntersection;

use super::{FalloffFn, MeshSelector, WeightedSelection, sphere_with_falloff_weight};

/// Generates a selection on the surface of a mesh that is within a sphere with a falloff and that
/// is limited to be connected to the input face.
#[derive(Debug)]
pub struct GeodesicWithFalloff {
    /// The radius of the sphere.
    pub radius: f32,

    /// The falloff distance of the sphere. This means that the influence
    /// decreases from the radius to the radius + falloff.
    /// The way the influence decreases is controlled by `falloff_func`.
    pub falloff: f32,

    /// The falloff function used to calculate the weight of the selection.
    /// It receives values from 0.0 to 1.0 and has to return a value in the same range.
    /// Simply returning the input value is a linear falloff.
    pub falloff_func: FalloffFn,
}

impl GeodesicWithFalloff {
    #[inline]
    pub fn sphere(radius: f32, falloff: f32, falloff_func: FalloffFn) -> Self {
        Self {
            radius,
            falloff,
            falloff_func,
        }
    }
}

impl MeshSelector for GeodesicWithFalloff {
    #[instrument(skip(self, mesh_graph))]
    fn select(
        &self,
        mesh_graph: &MeshGraph,
        face_intersection: &FaceIntersection,
    ) -> WeightedSelection {
        let input_face = face_intersection.face;

        let max_dist = self.radius + self.falloff;

        let mut vertex_to_distance = HashMap::new();

        let mut new_vertices = HashSet::new();

        if let Some(he) = mesh_graph.halfedges.get(input_face.halfedge) {
            new_vertices.insert(he.end_vertex);
            vertex_to_distance.insert(he.end_vertex, 0.0);
        } else {
            error!("Halfedge not found");
        }

        while !new_vertices.is_empty() {
            let mut new_new_vertices = HashSet::new();

            for v_id in new_vertices {
                let vertex = match mesh_graph.vertices.get(v_id) {
                    Some(vertex) => vertex,
                    None => {
                        error!("Vertex not found");
                        continue;
                    }
                };

                let dist = vertex_to_distance[&v_id];

                for he_id in vertex.outgoing_halfedges(mesh_graph) {
                    let Some(he) = mesh_graph.halfedges.get(he_id) else {
                        error!("Halfedge not found");
                        continue;
                    };

                    if !vertex_to_distance.contains_key(&he.end_vertex) {
                        let new_dist = dist + he.length(mesh_graph);

                        if new_dist <= max_dist {
                            new_new_vertices.insert(he.end_vertex);
                            vertex_to_distance.insert(he.end_vertex, new_dist);
                        }
                    }
                }
            }

            new_vertices = new_new_vertices;
        }

        WeightedSelection {
            vertex_to_weight: HashMap::from_iter(vertex_to_distance.into_iter().map(
                |(v_id, dist)| {
                    (
                        v_id,
                        sphere_with_falloff_weight(
                            dist,
                            self.radius,
                            self.falloff,
                            self.falloff_func,
                        ),
                    )
                },
            )),
        }
    }
}
