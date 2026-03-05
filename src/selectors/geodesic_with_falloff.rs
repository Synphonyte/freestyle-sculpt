use std::{cmp::Reverse, collections::BinaryHeap};

use hashbrown::HashMap;

use mesh_graph::MeshGraph;
use tracing::{error, instrument};

use crate::ray::FaceIntersection;

use super::{FalloffFn, MeshSelector, WeightedSelection, sphere_with_falloff_weight};

#[derive(PartialEq)]
struct FloatOrd(f32);

impl Eq for FloatOrd {}

impl PartialOrd for FloatOrd {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FloatOrd {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

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

        let mut vertex_to_distance: HashMap<_, f32> = HashMap::new();
        let mut heap: BinaryHeap<Reverse<(FloatOrd, _)>> = BinaryHeap::new();

        for he_id in input_face.halfedges(mesh_graph) {
            let Some(he) = mesh_graph.halfedges.get(he_id) else {
                error!("Halfedge not found");
                continue;
            };
            let v_id = he.end_vertex;
            let Some(&pos) = mesh_graph.positions.get(v_id) else {
                error!("Vertex position not found");
                continue;
            };
            let dist = face_intersection.point.distance(pos);
            if dist <= max_dist {
                vertex_to_distance.insert(v_id, dist);
                heap.push(Reverse((FloatOrd(dist), v_id)));
            }
        }

        while let Some(Reverse((FloatOrd(dist), v_id))) = heap.pop() {
            if vertex_to_distance
                .get(&v_id)
                .copied()
                .unwrap_or(f32::INFINITY)
                < dist
            {
                continue;
            }

            let vertex = match mesh_graph.vertices.get(v_id) {
                Some(vertex) => vertex,
                None => {
                    error!("Vertex not found");
                    continue;
                }
            };

            for he_id in vertex.outgoing_halfedges(mesh_graph) {
                let Some(he) = mesh_graph.halfedges.get(he_id) else {
                    error!("Halfedge not found");
                    continue;
                };

                let new_dist = dist + he.length(mesh_graph);
                if new_dist <= max_dist {
                    let current = vertex_to_distance
                        .get(&he.end_vertex)
                        .copied()
                        .unwrap_or(f32::INFINITY);
                    if new_dist < current {
                        vertex_to_distance.insert(he.end_vertex, new_dist);
                        heap.push(Reverse((FloatOrd(new_dist), he.end_vertex)));
                    }
                }
            }
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
