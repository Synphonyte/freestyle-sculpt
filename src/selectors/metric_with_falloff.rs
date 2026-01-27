use glam::Vec3;
use hashbrown::HashSet;

use mesh_graph::{Face, MeshGraph, Selection, error_none};
use tracing::{error, instrument};

use super::{
    DistanceCalculator, FalloffFn, L2, MeshSelector, WeightedSelection, faces_incident_to_vertices,
    get_sphere_with_falloff_weight_callback,
};

/// Generates a selection of a mesh that is within a sphere with a falloff
#[derive(Debug)]
pub struct MetricWithFalloff<D: DistanceCalculator> {
    /// The radius of the sphere.
    pub radius: f32,

    /// The falloff distance of the sphere. This means that the influence
    /// decreases from the radius to the radius + falloff.
    /// The way the influence decreases is controlled by `falloff_func`.
    pub falloff: f32,

    /// The metric squared used to calculate the distance between the input position and the vertices.
    pub metric_squared: D,

    /// The falloff function used to calculate the weight of the selection.
    /// It receives values from 0.0 to 1.0 and has to return a value in the same range.
    /// Simply returning the input value is a linear falloff.
    pub falloff_func: FalloffFn,
}

impl MetricWithFalloff<L2> {
    /// Creates a new `MetricWithFalloff` selector with a sphere metric (normal L2 distance).
    #[inline]
    pub fn sphere(radius: f32, falloff: f32, falloff_func: FalloffFn) -> Self {
        Self {
            radius,
            falloff,
            metric_squared: L2,
            falloff_func,
        }
    }
}

impl<D: DistanceCalculator + Copy + 'static> MeshSelector for MetricWithFalloff<D> {
    #[instrument(skip(self, mesh_graph))]
    fn select(
        &self,
        mesh_graph: &MeshGraph,
        input_pos: Vec3,
        _input_face: Face,
    ) -> WeightedSelection {
        let mut vertices = HashSet::new();

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
                let distance = self.metric_squared.distance_squared(*pos, input_pos);

                if distance <= max_dist_sqr {
                    vertices.insert(vertex_id);
                }
            } else {
                error!("Position not found");
            }
        }

        WeightedSelection {
            selection: Selection {
                faces: faces_incident_to_vertices(vertices, mesh_graph),
                ..Default::default()
            },
            get_weight: get_sphere_with_falloff_weight_callback(
                input_pos,
                self.radius,
                self.falloff,
                self.falloff_func,
                self.metric_squared,
            ),
        }
    }
}
