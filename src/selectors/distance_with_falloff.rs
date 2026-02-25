use glam::Vec3;
use hashbrown::{HashMap, HashSet};

use mesh_graph::{MeshGraph, Selection, error_none};
use tracing::{error, instrument};

use crate::ray::FaceIntersection;

use super::{FalloffFn, MeshSelector, WeightedSelection, sphere_with_falloff_weight};

/// Generates a selection of a mesh that is within a sphere with a falloff
#[derive(Debug)]
pub struct DistanceWithFalloff {
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

impl DistanceWithFalloff {
    /// Creates a new `MetricWithFalloff` selector with a sphere metric (normal L2 distance).
    #[inline]
    pub fn sphere(radius: f32, falloff: f32, falloff_func: FalloffFn) -> Self {
        Self {
            radius,
            falloff,
            falloff_func,
        }
    }
}

impl MeshSelector for DistanceWithFalloff {
    #[instrument(skip(self, mesh_graph))]
    fn select(
        &self,
        mesh_graph: &MeshGraph,
        face_intersection: &FaceIntersection,
    ) -> WeightedSelection {
        let input_pos = face_intersection.point;

        let mut vertex_to_weight = HashMap::new();

        let aabb = parry3d::bounding_volume::Aabb::from_half_extents(
            input_pos,
            Vec3::splat(self.radius + self.falloff),
        );
        let potential_faces = mesh_graph.bvh.intersect_aabb(&aabb);

        let potential_selection = Selection {
            faces: HashSet::from_iter(potential_faces.filter_map(|idx| {
                mesh_graph
                    .index_to_face_id
                    .get(&idx)
                    .copied()
                    .or_else(error_none!("Face not found"))
            })),
            ..Default::default()
        };

        let sum = self.radius + self.falloff;
        let max_dist_sqr = sum * sum;

        for vertex_id in potential_selection.resolve_to_vertices(mesh_graph) {
            if let Some(pos) = mesh_graph.positions.get(vertex_id) {
                let dist_sqr = pos.distance_squared(input_pos);

                if dist_sqr <= max_dist_sqr {
                    vertex_to_weight.insert(
                        vertex_id,
                        sphere_with_falloff_weight(
                            dist_sqr.sqrt(),
                            self.radius,
                            self.falloff,
                            self.falloff_func,
                        ),
                    );
                }
            } else {
                error!("Position not found");
            }
        }

        WeightedSelection { vertex_to_weight }
    }
}
