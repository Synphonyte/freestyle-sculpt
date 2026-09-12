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

#[test]
fn from_mesh_graph_is_resolution_neutral() {
    use mesh_graph::primitives::IcoSphere;

    let mesh_graph = MeshGraph::from(IcoSphere {
        subdivisions: 3,
        radius: 1.0,
    });

    let mut edge_lengths = mesh_graph
        .halfedges
        .values()
        .map(|he| he.length(&mesh_graph))
        .collect_vec();
    let median_edge_length = parry3d::utils::median(&mut edge_lengths);

    let params = SculptParams::from_mesh_graph(&mesh_graph, 0.0);

    // The target is the *upper* bound of the band, so it must sit above the
    // median — otherwise half the edges are split on the first cleanup pass.
    assert!(
        params.max_edge_length_squared.sqrt() > median_edge_length,
        "target edge length must exceed the median, got {} for median {median_edge_length}",
        params.max_edge_length_squared.sqrt()
    );
    assert!(
        (params.max_edge_length_squared.sqrt()
            - median_edge_length * SculptParams::RESOLUTION_NEUTRAL_EDGE_LENGTH_FACTOR)
            .abs()
            < 1e-5
    );
}

#[test]
fn from_mesh_graph_with_factor_clamps_to_min_edge_length_after_scaling() {
    use mesh_graph::primitives::IcoSphere;

    let mesh_graph = MeshGraph::from(IcoSphere {
        subdivisions: 2,
        radius: 1.0,
    });

    // A floor far above anything the median could produce must win.
    let params = SculptParams::from_mesh_graph_with_factor(&mesh_graph, 100.0, 1.6);
    assert!((params.max_edge_length_squared.sqrt() - 100.0).abs() < 1e-3);

    // A larger factor must produce a coarser (longer) target.
    let fine = SculptParams::from_mesh_graph_with_factor(&mesh_graph, 0.0, 1.0);
    let coarse = SculptParams::from_mesh_graph_with_factor(&mesh_graph, 0.0, 2.0);
    assert!(coarse.max_edge_length_squared > fine.max_edge_length_squared);
}

#[test]
fn from_mesh_graph_handles_mesh_without_halfedges() {
    let mesh_graph = MeshGraph::default();
    let params = SculptParams::from_mesh_graph(&mesh_graph, 0.5);
    assert!((params.max_edge_length_squared.sqrt() - 0.5).abs() < 1e-6);
}
