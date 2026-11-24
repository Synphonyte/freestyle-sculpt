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
        use crate::hash::HashMap;

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

/// `max_movement_squared` decides the step count, so it must bound the real
/// movement `translation * weight * strength` for any strength, including
/// negative ones (which used to give a NaN step count and no movement).
#[test]
fn translate_max_movement_scales_with_strength_squared() {
    use crate::deformation::{DeformationField, TranslateDeformation};
    use crate::ray::{FaceIntersection, Ray};
    use crate::selectors::{DistanceWithFalloff, SMOOTH_FALLOFF};
    use glam::Vec3;
    use mesh_graph::primitives::IcoSphere;

    let mesh_graph = MeshGraph::from(IcoSphere {
        subdivisions: 2,
        radius: 1.0,
    });
    let hit = Ray {
        origin: Vec3::new(5.0, 0.0, 0.0),
        direction: Vec3::NEG_X,
    }
    .cast_ray_and_get_face_id(&mesh_graph)
    .unwrap();
    let intersection = FaceIntersection { toi: 0.0, ..hit };

    let mut translate = TranslateDeformation::new();
    translate.on_pointer_down(intersection);
    translate.on_pointer_move(&mesh_graph, Vec3::X * 0.2, Some(intersection));

    let selector = DistanceWithFalloff::sphere(0.3, 0.2, SMOOTH_FALLOFF);
    for strength in [1.0f32, 4.0, -1.0, -3.0] {
        let max_movement =
            translate.max_movement_squared(&mesh_graph, &selector, strength, &intersection);
        let expected = 0.2 * strength.abs();
        assert!(
            (max_movement.sqrt() - expected).abs() < 1e-5,
            "strength {strength}: got {}, expected {expected}",
            max_movement.sqrt()
        );
    }
}

/// Without topology changes the protected sets are never drained by a merge pass,
/// so the cleanup itself must clear them, or they grow across strokes.
#[test]
fn cleanup_without_topology_change_clears_protected_sets() {
    use crate::deformation::{TopologyManager, cleanup_mesh};
    use mesh_graph::primitives::IcoSphere;

    let mut mesh_graph = MeshGraph::from(IcoSphere {
        subdivisions: 2,
        radius: 1.0,
    });
    // A much finer target forces subdivisions, which mark their new vertices.
    let params = SculptParams::from_mesh_graph_with_factor(&mesh_graph, 0.0, 0.5);
    let mut topology_manager = TopologyManager::new(&mesh_graph, params);
    topology_manager
        .protected_halfedges
        .extend(mesh_graph.halfedges.keys());

    cleanup_mesh(&mut mesh_graph, &params, &mut topology_manager, false);

    assert!(topology_manager.protected_vertices.is_empty());
    assert!(topology_manager.protected_halfedges.is_empty());
}

/// `apply` fills in the normals the selection is missing before each step, so a
/// field that moves along normals moves vertices that have none yet (new ones from
/// the previous step's cleanup, or every vertex of a mesh without normals).
#[test]
fn erode_dilate_moves_vertices_without_normals() {
    use crate::deformation::{DeformationField, ErodeDilateDeformation, TopologyManager};
    use crate::ray::Ray;
    use crate::selectors::{DistanceWithFalloff, SMOOTH_FALLOFF};
    use glam::Vec3;
    use mesh_graph::primitives::IcoSphere;

    let mut mesh_graph = MeshGraph::from(IcoSphere {
        subdivisions: 3,
        radius: 1.0,
    });
    mesh_graph.vertex_normals = None;

    let hit = Ray {
        origin: Vec3::new(5.0, 0.0, 0.0),
        direction: Vec3::NEG_X,
    }
    .cast_ray_and_get_face_id(&mesh_graph)
    .unwrap();

    let params = SculptParams::from_mesh_graph(&mesh_graph, 0.0);
    let mut topology_manager = TopologyManager::new(&mesh_graph, params);

    let mut dilate = ErodeDilateDeformation::new(0.05);
    dilate.on_pointer_move(&mesh_graph, Vec3::ZERO, Some(hit));
    dilate.apply(
        &mut mesh_graph,
        &DistanceWithFalloff::sphere(0.2, 0.1, SMOOTH_FALLOFF),
        1.0,
        params,
        &mut topology_manager,
    );

    let max_x = mesh_graph
        .positions
        .values()
        .map(|p| p.x)
        .fold(f32::MIN, f32::max);
    assert!(
        max_x > 1.04,
        "the brush did not dilate the sphere: max x {max_x}"
    );
}

/// A closed surface of revolution around the z axis: `rings` rings of `k`
/// vertices from `z = 0` to `z = len`, radius `radius(z)`, closed by flat caps.
fn surface_of_revolution(
    k: usize,
    rings: usize,
    len: f32,
    radius: impl Fn(f32) -> f32,
) -> MeshGraph {
    use glam::Vec3;

    let ring = |j: usize| -> Vec<Vec3> {
        let z = len * j as f32 / (rings - 1) as f32;
        (0..k)
            .map(|i| {
                let a = std::f32::consts::TAU * i as f32 / k as f32;
                Vec3::new(radius(z) * a.cos(), radius(z) * a.sin(), z)
            })
            .collect()
    };

    let mut soup = Vec::new();
    for j in 0..rings - 1 {
        let (a, b) = (ring(j), ring(j + 1));
        for i in 0..k {
            let i2 = (i + 1) % k;
            soup.extend([a[i], a[i2], b[i2], a[i], b[i2], b[i]]);
        }
    }
    let (bottom, top) = (ring(0), ring(rings - 1));
    for i in 0..k {
        let i2 = (i + 1) % k;
        soup.extend([Vec3::ZERO, bottom[i2], bottom[i]]);
        soup.extend([Vec3::new(0.0, 0.0, len), top[i], top[i2]]);
    }

    MeshGraph::triangles(&soup).expect("surface of revolution must build a mesh graph")
}

/// The number of connected components, and the vertex count of each.
fn component_sizes(mesh_graph: &MeshGraph) -> Vec<usize> {
    let mut seen = crate::hash::HashSet::default();
    let mut sizes = Vec::new();

    for start in mesh_graph.vertices.keys() {
        if !seen.insert(start) {
            continue;
        }
        let mut size = 0;
        let mut stack = vec![start];
        while let Some(v_id) = stack.pop() {
            size += 1;
            for n in mesh_graph.vertices[v_id].neighbours(mesh_graph) {
                if seen.insert(n) {
                    stack.push(n);
                }
            }
        }
        sizes.push(size);
    }

    sizes
}

/// Eroding an hourglass must cut its neck: the opposite walls of a part thinner
/// than `max_thickness` are merged, splitting the mesh in two. A merge gate that
/// only accepts sheets facing each other (as a crease fix once tried) leaves the
/// neck as an ever thinner tube instead.
///
/// The bulbs are long and wide so that the open (erode, then dilate) leaves
/// plenty of surface on either side of the cut and the mesh cannot vanish.
#[test]
fn eroding_an_hourglass_splits_its_neck() {
    use crate::deformation::{TopologyManager, morphological_open_close};

    const LEN: f32 = 12.0;
    const BULB_RADIUS: f32 = 2.0;
    const NECK_RADIUS: f32 = 0.3;
    const NECK_HALF_LEN: f32 = 2.0;

    let mut mesh_graph = surface_of_revolution(40, 61, LEN, |z| {
        let t = ((z - LEN / 2.0).abs() / NECK_HALF_LEN).min(1.0);
        NECK_RADIUS + (BULB_RADIUS - NECK_RADIUS) * t * t
    });
    mesh_graph.compute_vertex_normals();
    assert_eq!(component_sizes(&mesh_graph).len(), 1);

    let params = SculptParams::new(0.35);
    let mut topology_manager = TopologyManager::new(&mesh_graph, params);

    // Open: erode by 0.3, then dilate by 0.3. The neck is thinner than the erode.
    morphological_open_close(&mut mesh_graph, &params, &mut topology_manager, 0.3, ());

    let sizes = component_sizes(&mesh_graph);
    let boundary = mesh_graph
        .halfedges
        .values()
        .filter(|he| he.face.is_none())
        .count();
    let euler = mesh_graph.vertices.len() as i64 - (mesh_graph.halfedges.len() / 2) as i64
        + mesh_graph.faces.len() as i64;

    assert_eq!(sizes.len(), 2, "the neck was not cut: components {sizes:?}");
    assert!(
        sizes.iter().all(|&size| size > 100),
        "a bulb lost most of its surface: components {sizes:?}"
    );
    assert_eq!(boundary, 0, "the cut left a hole");
    assert_eq!(euler, 4, "expected two closed spheres");
}
