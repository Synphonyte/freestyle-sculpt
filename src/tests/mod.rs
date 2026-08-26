#[cfg(all(feature = "gltf", feature = "serde"))]
mod logs;

use super::*;

#[test]
fn new_produces_consistent_params() {
    let params = SculptParams::new(2.0);

    assert_eq!(params.max_edge_length_squared, 4.0);
    assert!(params.max_move_dist_squared > 0.0);
    assert!(params.min_edge_length_squared > 0.0);
    assert!(params.min_edge_length_squared < params.max_edge_length_squared);
    assert!(params.max_thickness_squared > 0.0);
    assert!(params.max_thickness_half.is_finite());
}

/// A `MeshSelector` implementation that returns a stale/invalid vertex id
/// (e.g. from a previously deleted mesh) to simulate misbehaving user code.
struct StaleIdSelector;

impl crate::selectors::MeshSelector for StaleIdSelector {
    fn select(
        &self,
        _mesh_graph: &MeshGraph,
        _face_intersection: &crate::ray::FaceIntersection,
    ) -> crate::selectors::WeightedSelection {
        use hashbrown::HashMap;

        // The null key is never present in any `SlotMap`.
        crate::selectors::WeightedSelection {
            vertex_to_weight: HashMap::from_iter([(mesh_graph::VertexId::default(), 1.0)]),
        }
    }
}

#[test]
fn apply_ignores_stale_selector_vertex_ids() {
    use crate::deformation::{DeformationField, SmoothDeformation, TopologyManager};
    use crate::ray::FaceIntersection;
    use mesh_graph::primitives::IcoSphere;

    let mut mesh_graph = MeshGraph::from(IcoSphere {
        subdivisions: 1,
        radius: 1.0,
    });

    // Any face from the mesh is fine; the selector ignores it.
    let face = *mesh_graph.faces.values().next().unwrap();
    let intersection = FaceIntersection {
        point: glam::Vec3::ZERO,
        face,
        toi: 0.0,
    };

    let mut deformation = SmoothDeformation::new(0.1);
    deformation.on_pointer_move(&mesh_graph, glam::Vec3::ZERO, Some(intersection));

    let params = SculptParams::new(1.0);
    let mut topology_manager = TopologyManager::new(&mesh_graph, params);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        deformation.apply(
            &mut mesh_graph,
            &StaleIdSelector,
            1.0,
            params,
            &mut topology_manager,
        )
    }));

    assert!(
        result.is_ok(),
        "`apply` must not panic when a selector returns a stale vertex id"
    );
    assert!(mesh_graph.vertex_normals.is_some());
}
