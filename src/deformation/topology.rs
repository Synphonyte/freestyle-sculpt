use crate::hash::{HashMap, HashSet};
use mesh_graph::{FaceId, HalfedgeId, MergeVerticesOneRing, MeshGraph, VertexId};
use rapier3d::prelude::*;
use tracing::{error, instrument};

use crate::SculptParams;

pub struct TopologyManager {
    pub protected_halfedges: HashSet<HalfedgeId>,
    pub protected_vertices: HashSet<VertexId>,

    pub collision_pipeline: CollisionPipeline,
    pub island_manager: IslandManager,
    pub broad_phase: BroadPhaseBvh,
    pub narrow_phase: NarrowPhase,
    pub rigid_body_set: RigidBodySet,
    pub collider_set: ColliderSet,

    pub vertex_id_to_collider_handle: HashMap<VertexId, ColliderHandle>,
}

/// Debug hook for the open weld-hole defect: set `WELD_CAPTURE=1` to have the
/// merge loop watch the boundary-halfedge count and print, as a rebuildable
/// region, the neighborhood of the first merge that opens a watertight mesh.
/// Off by default (it snapshots a region before every merge).
const WELD_CAPTURE_MAX_RADIUS: usize = 4;

fn weld_capture_enabled() -> bool {
    std::env::var("WELD_CAPTURE").is_ok_and(|v| v != "0")
}

/// The weld-path collapse, wrapped in the collapse-phase capture: under
/// `WELD_CAPTURE=1` the pre-op mesh is cloned, and if the collapse opens a
/// previously-closed region (boundary gained with none before) or leaves a
/// wire edge (both halves faceless), the neighborhood of the newly-added
/// boundary vertices is cut from the clone, replayed standalone, minimised and
/// printed — the collapse analogue of the merge capture. Off by default.
pub(crate) fn collapse_until_edges_above_min_length_with_weld_capture(
    mesh_graph: &mut MeshGraph,
    min_len_sqr: f32,
    marked_vertices: &mut HashSet<VertexId>,
) -> mesh_graph::EdgeLengthCleanup {
    if !weld_capture_enabled() {
        return mesh_graph.collapse_until_edges_above_min_length(min_len_sqr, marked_vertices);
    }

    // Every weld call site passes `protected_vertices` as the marked set, so the
    // snapshot doubles as the replay's protected set.
    let protected = marked_vertices.clone();
    let pre = mesh_graph.clone();
    let before = boundary_edge_pairs(&pre);
    let wires_before = wire_halves(&pre);

    let outcome = mesh_graph.collapse_until_edges_above_min_length(min_len_sqr, marked_vertices);

    let after = boundary_edge_pairs(mesh_graph);
    let wires_after = wire_halves(mesh_graph);

    let added: Vec<VertexId> = after
        .difference(&before)
        .flat_map(|(a, b)| [*a, *b])
        .collect();
    let opened_closed = before.is_empty() && !after.is_empty();
    if (opened_closed || wires_after > wires_before) && !added.is_empty() {
        let mut seeds: Vec<VertexId> = Vec::new();
        for v in added {
            if !seeds.contains(&v) {
                seeds.push(v);
            }
        }
        capture_collapse_failure(&pre, &seeds, min_len_sqr, &protected);
    }

    outcome
}

/// Cut/Replay/Minimise/Print for a collapse-opened region, then abort the weld
/// exactly like the merge capture does.
fn capture_collapse_failure(
    pre: &MeshGraph,
    seeds: &[VertexId],
    min_len_sqr: f32,
    protected: &HashSet<VertexId>,
) -> ! {
    let regions: Vec<CapturedRegion> = (1..=WELD_CAPTURE_MAX_RADIUS)
        .map(|r| capture_region_seeded(pre, seeds, r))
        .collect();

    let found = regions.iter().enumerate().find_map(|(i, region)| {
        collapse_region_reproduces(region, min_len_sqr, protected)
            .map(|(opened, wire)| (i + 1, region, opened, wire))
    });

    match found {
        Some((radius, region, opened, wire)) => {
            let keep: Vec<VertexId> = seeds.to_vec();
            let minimal = minimise_collapse_region(region, min_len_sqr, protected, &keep);
            let still =
                collapse_region_reproduces(&minimal, min_len_sqr, protected).unwrap_or((0, 0));
            println!(
                "// reproduces collapse standalone at radius {radius}: +{opened} boundary / \
                 {wire} wire; minimised {} -> {} faces (still +{} / {})",
                region.1.len(),
                minimal.1.len(),
                still.0,
                still.1
            );
            print_collapse_region(&minimal, min_len_sqr, protected, seeds);
        }
        None => {
            println!(
                "// NO STANDALONE COLLAPSE REPRO up to radius \
                 {WELD_CAPTURE_MAX_RADIUS}; printing the widest cut"
            );
            print_collapse_region(regions.last().unwrap(), min_len_sqr, protected, seeds);
        }
    }

    panic!("WELD CAPTURE: collapse phase opened the mesh")
}

/// The merge-order knob of the weld soak: `None` keeps the shipping order,
/// `Some(seed)` shuffles it deterministically (seed 0 is the shipping order).
fn weld_order_seed() -> Option<u64> {
    use std::sync::OnceLock;

    static SEED: OnceLock<Option<u64>> = OnceLock::new();
    *SEED.get_or_init(|| match std::env::var("WELD_ORDER_SEED") {
        Ok(value) => match value.trim().parse::<u64>() {
            Ok(0) => None,
            Ok(seed) => Some(seed),
            Err(_) => {
                eprintln!("WELD_ORDER_SEED={value:?} is not a u64; keeping the shipping order");
                None
            }
        },
        Err(_) => None,
    })
}

/// `WELD_ORDER=reversed` reproduces the reversed merge order.
fn weld_order_reversed() -> bool {
    use std::sync::OnceLock;

    static REVERSED: OnceLock<bool> = OnceLock::new();
    *REVERSED.get_or_init(|| std::env::var("WELD_ORDER").is_ok_and(|v| v == "reversed"))
}

/// The splitmix64 step, used to turn a seed into the Fisher-Yates stream.
fn splitmix64(mut state: u64) -> u64 {
    state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

thread_local! {
    /// How many one-ring vertices the two merged vertices share, counted over
    /// every merge of a run: index = shared count.
    static SHARED_HISTOGRAM: std::cell::RefCell<Vec<usize>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// How many vertices the two one-rings have in common.
///
/// This is the number that says how well-posed the merge is. With no overlap the
/// two rings bound a clean annulus and "span a band between them" has an obvious
/// answer. Every shared vertex pinches that annulus, and every already-connected
/// stretch removes a piece of it.
fn shared_ring_vertices(mesh_graph: &MeshGraph, v1: VertexId, v2: VertexId) -> usize {
    let ring = |v: VertexId| -> Vec<VertexId> {
        mesh_graph
            .vertices
            .get(v)
            .map(|vert| {
                vert.one_ring(mesh_graph)
                    .filter_map(|he_id| mesh_graph.halfedges.get(he_id).map(|he| he.end_vertex))
                    .collect()
            })
            .unwrap_or_default()
    };
    let r1 = ring(v1);
    let r2 = ring(v2);
    r1.iter().filter(|v| r2.contains(v)).count()
}

/// Manifoldness of the whole mesh, as a list of problems (empty = manifold).
///
/// Purely combinatorial, from the public data:
/// * an undirected vertex pair joined by more than two halfedges is a
///   *non-manifold edge* (two edges between the same two vertices);
/// * for a vertex, `#incident edges == #outgoing halfedges`, and a manifold star
///   has `#faces == #edges` when closed or `#edges - 1` when it ends on a
///   boundary. Fewer faces than that means the star falls into more than one fan
///   - a pinched, non-manifold vertex.
fn manifold_problems(mesh_graph: &MeshGraph) -> Vec<String> {
    let mut problems = Vec::new();

    let mut per_edge: std::collections::BTreeMap<(VertexId, VertexId), usize> = Default::default();
    for (_, he) in mesh_graph.halfedges.iter() {
        let Some(start) = he.start_vertex(mesh_graph) else {
            continue;
        };
        let end = he.end_vertex;
        let key = if start < end {
            (start, end)
        } else {
            (end, start)
        };
        *per_edge.entry(key).or_default() += 1;
    }
    for ((a, b), count) in &per_edge {
        if *count > 2 {
            problems.push(format!("non-manifold edge {a:?}-{b:?}: {count} halfedges"));
        }
    }

    let mut faces_of: std::collections::BTreeMap<VertexId, usize> = Default::default();
    for (_, face) in mesh_graph.faces.iter() {
        for v in face.vertices(mesh_graph) {
            *faces_of.entry(v).or_default() += 1;
        }
    }
    for (v_id, out_hes) in mesh_graph.outgoing_halfedges.iter() {
        if !mesh_graph.vertices.contains_key(v_id) {
            continue;
        }
        let edges = out_hes.len();
        let faces = faces_of.get(&v_id).copied().unwrap_or(0);
        if edges > 0 && faces + 1 < edges {
            problems.push(format!(
                "pinched vertex {v_id:?}: {edges} edges but only {faces} faces (star is >1 fan)"
            ));
        }
    }

    problems
}

fn boundary_halfedges(mesh_graph: &MeshGraph) -> usize {
    mesh_graph
        .halfedges
        .values()
        .filter(|he| he.is_boundary())
        .count()
}

/// Boundary edges as unordered vertex pairs, plus the count of wire halves
/// (halfedges whose twin is also face-less): the vocabulary the capture replay
/// reasons about — a hole is boundary gained between interior vertices, a wire
/// is an edge with no face on either side.
fn boundary_edge_pairs(mesh_graph: &MeshGraph) -> std::collections::BTreeSet<(VertexId, VertexId)> {
    mesh_graph
        .halfedges
        .iter()
        .filter(|(_, he)| he.face.is_none())
        .filter_map(|(_, he)| {
            let start = he.start_vertex(mesh_graph)?;
            let end = he.end_vertex;
            Some(if start < end {
                (start, end)
            } else {
                (end, start)
            })
        })
        .collect()
}

fn wire_halves(mesh_graph: &MeshGraph) -> usize {
    mesh_graph
        .halfedges
        .iter()
        .filter(|(_, he)| {
            he.face.is_none()
                && he
                    .twin
                    .and_then(|t| mesh_graph.halfedges.get(t))
                    .is_some_and(|t| t.face.is_none())
        })
        .count()
}

/// A captured region: vertex positions, face triples, and — per vertex whose
/// whole star lies inside the cut — the order of its outgoing halfedges as
/// neighbor indices plus the neighbor its `outgoing_halfedge` seed points at.
///
/// That last part is not optional. `one_ring()` starts at the seed and the band
/// planner scans the ring from index 0, so it decides where the band is cut —
/// and a rebuild derives it from the order faces happen to be added, not from
/// the history the live mesh had. Without it, a cut-out is a different problem
/// from the one that failed.
type CapturedRegion = (
    Vec<(VertexId, [f32; 3])>,
    Vec<(FaceId, [usize; 3])>,
    Vec<(usize, Vec<usize>, usize)>,
);

/// Every face within `radius` rings of `v1`/`v2`.
fn capture_region(
    mesh_graph: &MeshGraph,
    v1: VertexId,
    v2: VertexId,
    radius: usize,
) -> CapturedRegion {
    capture_region_seeded(mesh_graph, &[v1, v2], radius)
}

/// [`capture_region`] with an arbitrary seed list — the collapse-phase capture
/// cannot know its failing vertices ahead of the op, so it seeds from the
/// boundary the op left behind.
fn capture_region_seeded(
    mesh_graph: &MeshGraph,
    seeds_in: &[VertexId],
    radius: usize,
) -> CapturedRegion {
    let mut seeds: Vec<VertexId> = seeds_in.to_vec();
    for _ in 0..radius {
        for v in seeds.clone() {
            if let Some(vert) = mesh_graph.vertices.get(v) {
                for he_id in vert.one_ring(mesh_graph) {
                    if let Some(he) = mesh_graph.halfedges.get(he_id)
                        && !seeds.contains(&he.end_vertex)
                    {
                        seeds.push(he.end_vertex);
                    }
                }
            }
        }
    }

    let mut face_ids: Vec<FaceId> = Vec::new();
    for v in seeds {
        if let Some(vert) = mesh_graph.vertices.get(v) {
            for f in vert.faces(mesh_graph) {
                if mesh_graph.faces.contains_key(f) && !face_ids.contains(&f) {
                    face_ids.push(f);
                }
            }
        }
    }
    face_ids.sort();

    let mut v_ids: Vec<VertexId> = Vec::new();
    for &f in &face_ids {
        for v in mesh_graph.faces[f].vertices(mesh_graph) {
            if !v_ids.contains(&v) {
                v_ids.push(v);
            }
        }
    }
    v_ids.sort();

    let positions = v_ids
        .iter()
        .map(|&v| {
            let p = mesh_graph.positions.get(v).copied().unwrap_or_default();
            (v, [p.x, p.y, p.z])
        })
        .collect();

    let faces = face_ids
        .iter()
        .map(|&f| {
            let tri: Vec<usize> = mesh_graph.faces[f]
                .vertices(mesh_graph)
                .map(|v| v_ids.iter().position(|x| *x == v).unwrap())
                .collect();
            (f, [tri[0], tri[1], tri[2]])
        })
        .collect();

    let mut outgoing = Vec::new();
    for (idx, &v) in v_ids.iter().enumerate() {
        let Some(out_hes) = mesh_graph.outgoing_halfedges.get(v) else {
            continue;
        };
        let ends: Option<Vec<usize>> = out_hes
            .iter()
            .map(|he_id| {
                let end = mesh_graph.halfedges.get(*he_id)?.end_vertex;
                v_ids.iter().position(|x| *x == end)
            })
            .collect();
        // Any neighbor outside the cut means the star is incomplete here; that
        // vertex's order cannot be restored and is not needed (it is on the rim).
        let Some(ends) = ends else { continue };

        let seed = mesh_graph
            .vertices
            .get(v)
            .and_then(|vert| vert.outgoing_halfedge)
            .and_then(|he_id| mesh_graph.halfedges.get(he_id))
            .and_then(|he| v_ids.iter().position(|x| *x == he.end_vertex));
        let Some(seed) = seed else { continue };

        outgoing.push((idx, ends, seed));
    }

    (positions, faces, outgoing)
}

/// Whether both merged vertices are *interior* to the region: no boundary
/// halfedge touches either of them, so their stars — and therefore their
/// one-rings — are complete.
///
/// A cut that fails this is not a faithful reproduction whatever it does on
/// replay: a merged vertex sitting on the cut has a truncated one-ring, so the
/// op sees a neighborhood the real mesh never had.
fn merge_pair_is_interior(mesh_graph: &MeshGraph, v1: VertexId, v2: VertexId) -> bool {
    [v1, v2].into_iter().all(|v| {
        mesh_graph.outgoing_halfedges.get(v).is_some_and(|out_hes| {
            !out_hes.is_empty()
                && out_hes.iter().all(|he_id| {
                    mesh_graph.halfedges.get(*he_id).is_some_and(|he| {
                        he.face.is_some()
                            && he
                                .twin
                                .and_then(|t| mesh_graph.halfedges.get(t))
                                .is_some_and(|t| t.face.is_some())
                    })
                })
        })
    })
}

/// Rebuilds a captured region as a standalone `MeshGraph`, in index order, so
/// the ids are a pure function of the region — that is what makes the extracted
/// test deterministic where the weld that produced it is not.
fn rebuild_region(region: &CapturedRegion) -> (MeshGraph, Vec<VertexId>) {
    let (positions, faces, outgoing) = region;

    let mut mesh_graph = MeshGraph::new();
    let vertex_ids: Vec<VertexId> = positions
        .iter()
        .map(|(_, p)| mesh_graph.add_vertex(glam::Vec3::new(p[0], p[1], p[2])))
        .collect();

    for (_, [a, b, c]) in faces {
        let _ = mesh_graph.add_face_from_vertices(vertex_ids[*a], vertex_ids[*b], vertex_ids[*c]);
    }

    restore_outgoing_order(&mut mesh_graph, &vertex_ids, outgoing);

    (mesh_graph, vertex_ids)
}

/// Puts each captured vertex's outgoing halfedges back in the order the live
/// mesh had them, and re-points its seed. See [`CapturedRegion`].
pub fn restore_outgoing_order(
    mesh_graph: &mut MeshGraph,
    vertex_ids: &[VertexId],
    outgoing: &[(usize, Vec<usize>, usize)],
) {
    for (idx, ends, seed) in outgoing {
        let v = vertex_ids[*idx];
        let Some(out_hes) = mesh_graph.outgoing_halfedges.get(v).cloned() else {
            continue;
        };

        // Rebuild the list in the captured neighbor order; anything the rebuild
        // has but the capture did not keeps its relative place at the end.
        let mut reordered = Vec::with_capacity(out_hes.len());
        let mut left = out_hes.clone();
        for end_idx in ends {
            let want = vertex_ids[*end_idx];
            if let Some(pos) = left.iter().position(|he_id| {
                mesh_graph
                    .halfedges
                    .get(*he_id)
                    .is_some_and(|he| he.end_vertex == want)
            }) {
                reordered.push(left.remove(pos));
            }
        }
        reordered.extend(left);

        let seed_v = vertex_ids[*seed];
        let seed_he = reordered.iter().copied().find(|he_id| {
            mesh_graph
                .halfedges
                .get(*he_id)
                .is_some_and(|he| he.end_vertex == seed_v)
        });

        mesh_graph.outgoing_halfedges.insert(v, reordered);
        if let Some(seed_he) = seed_he
            && let Some(vert) = mesh_graph.vertices.get_mut(v)
        {
            vert.outgoing_halfedge = Some(seed_he);
        }
    }
}

/// Replays the merge on a standalone rebuild of `region` and reports how many
/// boundary edges it gains. A region that gains none does not carry the defect:
/// the geometry that causes it lies outside the cut, and the region has to grow.
fn region_reproduces(
    region: &CapturedRegion,
    v1: VertexId,
    v2: VertexId,
    flip_threshold_sqr: f32,
    protected: &HashSet<VertexId>,
) -> Option<usize> {
    let (positions, _, _) = region;
    let index_of = |v: VertexId| positions.iter().position(|(x, _)| *x == v);

    let (idx1, idx2) = (index_of(v1)?, index_of(v2)?);
    let protected_idx: Vec<usize> = positions
        .iter()
        .enumerate()
        .filter(|(_, (v, _))| protected.contains(v))
        .map(|(i, _)| i)
        .collect();

    let (mut mesh_graph, vertex_ids) = rebuild_region(region);

    if !merge_pair_is_interior(&mesh_graph, vertex_ids[idx1], vertex_ids[idx2]) {
        return None;
    }

    // Boundary edges, as unordered vertex pairs.
    let boundary_edges = |mesh: &MeshGraph| -> std::collections::BTreeSet<(VertexId, VertexId)> {
        mesh.halfedges
            .iter()
            .filter(|(_, he)| he.face.is_none())
            .filter_map(|(_, he)| {
                let start = he.start_vertex(mesh)?;
                let end = he.end_vertex;
                Some(if start < end {
                    (start, end)
                } else {
                    (end, start)
                })
            })
            .collect()
    };

    let before = boundary_edges(&mesh_graph);
    // Vertices already on the cut. What happens to those is an artifact of the
    // cut (the ops treat boundary specially), so only edges between vertices
    // that were interior count as the defect - and comparing counts would let a
    // rim edge closing hide a hole opening.
    let rim: std::collections::BTreeSet<VertexId> =
        before.iter().flat_map(|(a, b)| [*a, *b]).collect();

    let mut marked_halfedges = HashSet::default();
    let mut marked_vertices = HashSet::default();
    for i in protected_idx {
        marked_vertices.insert(vertex_ids[i]);
    }

    mesh_graph.merge_vertices_one_rings(
        vertex_ids[idx1],
        vertex_ids[idx2],
        flip_threshold_sqr,
        &mut marked_halfedges,
        &mut marked_vertices,
    );

    let opened = boundary_edges(&mesh_graph)
        .difference(&before)
        .filter(|(a, b)| !rim.contains(a) && !rim.contains(b))
        .count();

    (opened > 0).then_some(opened)
}

/// The collapse-phase analogue of [`region_reproduces`]: rebuilds `region`,
/// runs `collapse_until_edges_above_min_length` on the cut with the weld's
/// min-edge threshold, and reports how many boundary edges strictly *between
/// interior vertices* it gains plus how many wire halves it leaves. Either
/// number nonzero means the cut carries the collapse defect.
fn collapse_region_reproduces(
    region: &CapturedRegion,
    min_len_sqr: f32,
    protected: &HashSet<VertexId>,
) -> Option<(usize, usize)> {
    let (positions, _, _) = region;
    let protected_idx: Vec<usize> = positions
        .iter()
        .enumerate()
        .filter(|(_, (v, _))| protected.contains(v))
        .map(|(i, _)| i)
        .collect();

    let (mut mesh_graph, vertex_ids) = rebuild_region(region);

    let boundary_set = |mesh: &MeshGraph| -> std::collections::BTreeSet<(VertexId, VertexId)> {
        mesh.halfedges
            .iter()
            .filter(|(_, he)| he.face.is_none())
            .filter_map(|(_, he)| {
                let start = he.start_vertex(mesh)?;
                let end = he.end_vertex;
                Some(if start < end {
                    (start, end)
                } else {
                    (end, start)
                })
            })
            .collect()
    };

    let before = boundary_set(&mesh_graph);
    let rim: std::collections::BTreeSet<VertexId> =
        before.iter().flat_map(|(a, b)| [*a, *b]).collect();

    let mut marked_vertices = HashSet::default();
    for i in protected_idx {
        marked_vertices.insert(vertex_ids[i]);
    }

    mesh_graph.collapse_until_edges_above_min_length(min_len_sqr, &mut marked_vertices);

    let after = boundary_set(&mesh_graph);
    let opened = after
        .difference(&before)
        .filter(|(a, b)| !rim.contains(a) && !rim.contains(b))
        .count();

    let wire = mesh_graph
        .halfedges
        .iter()
        .filter(|(_, he)| {
            he.face.is_none()
                && he
                    .twin
                    .and_then(|t| mesh_graph.halfedges.get(t))
                    .is_some_and(|t| t.face.is_none())
        })
        .count();

    if opened > 0 || wire > 0 {
        Some((opened, wire))
    } else {
        None
    }
}

/// Drops faces from a reproducing region for as long as it keeps reproducing.
///
/// Greedy and single-pass, which is enough to strip the ring of context the
/// radius search pulled in wholesale; the result is a region small enough to
/// paste into a test and read.
fn minimise_region(
    region: &CapturedRegion,
    v1: VertexId,
    v2: VertexId,
    flip_threshold_sqr: f32,
    protected: &HashSet<VertexId>,
    keep: &[VertexId],
) -> CapturedRegion {
    let mut faces = region.1.clone();

    // Faces that must not be dropped: anything touching the merge pair or their
    // one-rings. Those stars are the op's actual input - `remove_neighbor_faces`
    // deletes a ring vertex whose whole star it consumes, and the band planner
    // then plans over ids that no longer exist. Drop one of those faces and the
    // cut stops being the situation that failed; it becomes a more degenerate
    // one that fails for its own reasons.
    let keep_idx: Vec<usize> = keep
        .iter()
        .filter_map(|v| region.0.iter().position(|(x, _)| x == v))
        .collect();
    let pinned = |face: &(FaceId, [usize; 3])| face.1.iter().any(|idx| keep_idx.contains(idx));

    let mut i = 0;
    while i < faces.len() {
        if pinned(&faces[i]) {
            i += 1;
            continue;
        }
        let mut trial_faces = faces.clone();
        trial_faces.remove(i);

        // Re-index onto the vertices the trial faces still use.
        let mut kept_v: Vec<usize> = Vec::new();
        for (_, [a, b, c]) in &trial_faces {
            for idx in [a, b, c] {
                if !kept_v.contains(idx) {
                    kept_v.push(*idx);
                }
            }
        }
        kept_v.sort_unstable();

        let positions: Vec<(VertexId, [f32; 3])> =
            kept_v.iter().map(|idx| region.0[*idx]).collect();
        let outgoing_reindexed: Vec<(usize, Vec<usize>, usize)> = region
            .2
            .iter()
            .filter_map(|(idx, ends, seed)| {
                let new_idx = kept_v.iter().position(|x| x == idx)?;
                let new_ends: Option<Vec<usize>> = ends
                    .iter()
                    .map(|e| kept_v.iter().position(|x| x == e))
                    .collect();
                let new_seed = kept_v.iter().position(|x| x == seed)?;
                Some((new_idx, new_ends?, new_seed))
            })
            .collect();
        let faces_reindexed: Vec<(FaceId, [usize; 3])> = trial_faces
            .iter()
            .map(|(f, [a, b, c])| {
                (
                    *f,
                    [
                        kept_v.iter().position(|x| x == a).unwrap(),
                        kept_v.iter().position(|x| x == b).unwrap(),
                        kept_v.iter().position(|x| x == c).unwrap(),
                    ],
                )
            })
            .collect();

        let trial = (positions, faces_reindexed, outgoing_reindexed);
        if region_reproduces(&trial, v1, v2, flip_threshold_sqr, protected).is_some() {
            faces = trial_faces;
        } else {
            i += 1;
        }
    }

    // Final re-index of whatever survived.
    let mut kept_v: Vec<usize> = Vec::new();
    for (_, [a, b, c]) in &faces {
        for idx in [a, b, c] {
            if !kept_v.contains(idx) {
                kept_v.push(*idx);
            }
        }
    }
    kept_v.sort_unstable();

    let positions: Vec<(VertexId, [f32; 3])> = kept_v.iter().map(|idx| region.0[*idx]).collect();
    let outgoing_reindexed: Vec<(usize, Vec<usize>, usize)> = region
        .2
        .iter()
        .filter_map(|(idx, ends, seed)| {
            let new_idx = kept_v.iter().position(|x| x == idx)?;
            let new_ends: Option<Vec<usize>> = ends
                .iter()
                .map(|e| kept_v.iter().position(|x| x == e))
                .collect();
            let new_seed = kept_v.iter().position(|x| x == seed)?;
            Some((new_idx, new_ends?, new_seed))
        })
        .collect();
    let faces_reindexed: Vec<(FaceId, [usize; 3])> = faces
        .iter()
        .map(|(f, [a, b, c])| {
            (
                *f,
                [
                    kept_v.iter().position(|x| x == a).unwrap(),
                    kept_v.iter().position(|x| x == b).unwrap(),
                    kept_v.iter().position(|x| x == c).unwrap(),
                ],
            )
        })
        .collect();

    (positions, faces_reindexed, outgoing_reindexed)
}

fn print_region(
    region: &CapturedRegion,
    v1: VertexId,
    v2: VertexId,
    flip_threshold_sqr: f32,
    protected: &HashSet<VertexId>,
) {
    let (positions, faces, outgoing) = region;
    let idx_of = |v: VertexId| positions.iter().position(|(x, _)| *x == v);

    println!("// ---- WELD CAPTURE ----");
    println!(
        "// merge {v1:?} (index {:?}) with {v2:?} (index {:?}), flip_threshold_sqr = {flip_threshold_sqr}",
        idx_of(v1),
        idx_of(v2)
    );
    let protected_idx: Vec<usize> = positions
        .iter()
        .enumerate()
        .filter(|(_, (v, _))| protected.contains(v))
        .map(|(i, _)| i)
        .collect();
    println!("// protected vertices (indices): {protected_idx:?}");
    println!(
        "const REGION_VERTICES: [(f32, f32, f32); {}] = [",
        positions.len()
    );
    for (i, (v, p)) in positions.iter().enumerate() {
        println!("    ({:?}, {:?}, {:?}), // {v:?} -> {i}", p[0], p[1], p[2]);
    }
    println!("];");
    println!(
        "const REGION_FACES: [(usize, usize, usize); {}] = [",
        faces.len()
    );
    for (f, t) in faces {
        println!("    ({}, {}, {}), // {f:?}", t[0], t[1], t[2]);
    }
    println!("];");
    println!(
        "const REGION_OUTGOING: [(usize, &[usize], usize); {}] = [",
        outgoing.len()
    );
    for (idx, ends, seed) in outgoing {
        println!("    ({idx}, &{ends:?}, {seed}),");
    }
    println!("];");
    println!("// ---- END WELD CAPTURE ----");
}

/// Drops faces from a collapsing region for as long as it keeps reproducing,
/// with the same greedy single-pass strategy as [`minimise_region`]. The pinned
/// faces are those touching a `keep` vertex: they are the collapse's own input
/// (its flap removal happens right at those stars).
fn minimise_collapse_region(
    region: &CapturedRegion,
    min_len_sqr: f32,
    protected: &HashSet<VertexId>,
    keep: &[VertexId],
) -> CapturedRegion {
    let mut faces = region.1.clone();

    let keep_idx: Vec<usize> = keep
        .iter()
        .filter_map(|v| region.0.iter().position(|(x, _)| x == v))
        .collect();
    let pinned = |face: &(FaceId, [usize; 3])| face.1.iter().any(|idx| keep_idx.contains(idx));

    let mut i = 0;
    while i < faces.len() {
        if pinned(&faces[i]) {
            i += 1;
            continue;
        }
        let mut trial_faces = faces.clone();
        trial_faces.remove(i);

        // Re-index onto the vertices the trial faces still use.
        let mut kept_v: Vec<usize> = Vec::new();
        for (_, [a, b, c]) in &trial_faces {
            for idx in [a, b, c] {
                if !kept_v.contains(idx) {
                    kept_v.push(*idx);
                }
            }
        }
        kept_v.sort_unstable();

        let positions: Vec<(VertexId, [f32; 3])> =
            kept_v.iter().map(|idx| region.0[*idx]).collect();
        let outgoing_reindexed: Vec<(usize, Vec<usize>, usize)> = region
            .2
            .iter()
            .filter_map(|(idx, ends, seed)| {
                let new_idx = kept_v.iter().position(|x| x == idx)?;
                let new_ends: Option<Vec<usize>> = ends
                    .iter()
                    .map(|e| kept_v.iter().position(|x| x == e))
                    .collect();
                let new_seed = kept_v.iter().position(|x| x == seed)?;
                Some((new_idx, new_ends?, new_seed))
            })
            .collect();
        let faces_reindexed: Vec<(FaceId, [usize; 3])> = trial_faces
            .iter()
            .map(|(f, [a, b, c])| {
                (
                    *f,
                    [
                        kept_v.iter().position(|x| x == a).unwrap(),
                        kept_v.iter().position(|x| x == b).unwrap(),
                        kept_v.iter().position(|x| x == c).unwrap(),
                    ],
                )
            })
            .collect();

        let trial = (positions, faces_reindexed, outgoing_reindexed);
        if collapse_region_reproduces(&trial, min_len_sqr, protected).is_some() {
            faces = trial_faces;
        } else {
            i += 1;
        }
    }

    // Rebuild the region from whatever survived, with the same re-indexing as
    // the greedy loop's final pass.
    let mut kept_v: Vec<usize> = Vec::new();
    for (_, [a, b, c]) in &faces {
        for idx in [a, b, c] {
            if !kept_v.contains(idx) {
                kept_v.push(*idx);
            }
        }
    }
    kept_v.sort_unstable();

    let positions: Vec<(VertexId, [f32; 3])> = kept_v.iter().map(|idx| region.0[*idx]).collect();
    let outgoing_reindexed: Vec<(usize, Vec<usize>, usize)> = region
        .2
        .iter()
        .filter_map(|(idx, ends, seed)| {
            let new_idx = kept_v.iter().position(|x| x == idx)?;
            let new_ends: Option<Vec<usize>> = ends
                .iter()
                .map(|e| kept_v.iter().position(|x| x == e))
                .collect();
            let new_seed = kept_v.iter().position(|x| x == seed)?;
            Some((new_idx, new_ends?, new_seed))
        })
        .collect();
    let faces_reindexed: Vec<(FaceId, [usize; 3])> = faces
        .iter()
        .map(|(f, [a, b, c])| {
            (
                *f,
                [
                    kept_v.iter().position(|x| x == a).unwrap(),
                    kept_v.iter().position(|x| x == b).unwrap(),
                    kept_v.iter().position(|x| x == c).unwrap(),
                ],
            )
        })
        .collect();

    (positions, faces_reindexed, outgoing_reindexed)
}

/// Prints the rebuildable constants of a collapse-phase capture: the same
/// `REGION_*` blocks as the merge capture, plus the collapse's own threshold.
fn print_collapse_region(
    region: &CapturedRegion,
    min_len_sqr: f32,
    protected: &HashSet<VertexId>,
    seeds: &[VertexId],
) {
    let (positions, faces, outgoing) = region;
    println!("// ---- COLLAPSE CAPTURE ----");
    println!("// replay: `collapse_until_edges_above_min_length` with min_len_sqr = {min_len_sqr}");
    let seed_idx: Vec<usize> = seeds
        .iter()
        .filter_map(|v| positions.iter().position(|(x, _)| x == v))
        .collect();
    println!("// seeds (added-boundary vertices): {seed_idx:?}");
    let protected_idx: Vec<usize> = positions
        .iter()
        .enumerate()
        .filter(|(_, (v, _))| protected.contains(v))
        .map(|(i, _)| i)
        .collect();
    println!("// protected vertices (indices): {protected_idx:?}");
    println!(
        "const REGION_VERTICES: [(f32, f32, f32); {}] = [",
        positions.len()
    );
    for (i, (v, p)) in positions.iter().enumerate() {
        println!("    ({:?}, {:?}, {:?}), // {v:?} -> {i}", p[0], p[1], p[2]);
    }
    println!("];");
    println!(
        "const REGION_FACES: [(usize, usize, usize); {}] = [",
        faces.len()
    );
    for (f, t) in faces {
        println!("    ({}, {}, {}), // {f:?}", t[0], t[1], t[2]);
    }
    println!("];");
    println!(
        "const REGION_OUTGOING: [(usize, &[usize], usize); {}] = [",
        outgoing.len()
    );
    for (idx, ends, seed) in outgoing {
        println!("    ({idx}, &{ends:?}, {seed}),");
    }
    println!("];");
    println!("// ---- END COLLAPSE CAPTURE ----");
}

impl TopologyManager {
    #[cfg(feature = "rerun")]
    pub fn log_rerun(&self) {
        Self::log_colliders_rerun("all", self.collider_set.iter().map(|(_, col)| col));
    }

    #[cfg(feature = "rerun")]
    pub fn log_colliders_rerun<'a>(name: &str, colliders: impl IntoIterator<Item = &'a Collider>) {
        let (positions, radii): (Vec<_>, Vec<_>) = colliders
            .into_iter()
            .map(|col: &Collider| {
                (
                    mesh_graph::utils::vec3_array(col.translation()),
                    col.shape().as_ball().unwrap().radius,
                )
            })
            .unzip();

        mesh_graph::RR
            .log(
                format!("freestyle/colliders/{name}"),
                &rerun::Ellipsoids3D::from_centers_and_radii(positions, radii),
            )
            .unwrap();
    }

    pub fn create_collider(sculpt_params: &SculptParams, vertex_id: VertexId) -> ColliderBuilder {
        ColliderBuilder::ball(sculpt_params.max_thickness_half)
            .sensor(true)
            .active_events(ActiveEvents::COLLISION_EVENTS)
            .active_hooks(ActiveHooks::FILTER_INTERSECTION_PAIR)
            .active_collision_types(ActiveCollisionTypes::FIXED_FIXED)
            .user_data(vertex_id.into())
    }

    pub fn new(mesh_graph: &MeshGraph, sculpt_params: SculptParams) -> Self {
        let mut collider_set = ColliderSet::new();
        let mut vertex_id_to_collider_handle = HashMap::default();

        for (vertex_id, &pos) in &mesh_graph.positions {
            let collider = Self::create_collider(&sculpt_params, vertex_id).translation(pos);
            let collider_handle = collider_set.insert(collider);
            vertex_id_to_collider_handle.insert(vertex_id, collider_handle);
        }

        Self {
            protected_halfedges: HashSet::default(),
            protected_vertices: HashSet::default(),

            collision_pipeline: CollisionPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: BroadPhaseBvh::new(),
            narrow_phase: NarrowPhase::new(),
            rigid_body_set: RigidBodySet::new(),
            collider_set,

            vertex_id_to_collider_handle,
        }
    }

    #[instrument(skip_all)]
    pub fn update_collisions_and_merge(
        &mut self,
        mesh_graph: &mut MeshGraph,
        sculpt_params: &SculptParams,
    ) -> Option<MergeVerticesOneRing> {
        #[cfg(feature = "rerun")]
        self.log_rerun();

        self.collision_pipeline.step(
            0.002,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &TopologyPhysicsHooks {
                mesh_graph,
                protected_vertices: &self.protected_vertices,
            },
            &(),
        );

        // Take the colliding pairs in a fixed order rather than rapier's.
        //
        // `intersection_pairs()` yields the narrow phase's internal graph order,
        // which is not a function of the mesh: the same fixture gives a
        // different order on every run. Since the loop below merges the *first*
        // mergeable pair it finds and each merge rewrites the neighborhood the
        // later pairs live in, that order decides the whole sequence of merges.
        // Ordering by vertex id makes the sequence follow the mesh instead, the
        // same rule the mesh-graph ops follow.
        //
        // This ordering is not itself the fix - the rim-first band planner in
        // mesh-graph is - but it is what makes a failure reproducible: with
        // mesh-graph's fixed-seed hasher the fixture, each merge and the whole
        // sequence are a pure function of this order, so a failing order can be
        // replayed instead of struck by chance.
        //
        // The order is a knob, for that reason:
        //
        // - `WELD_ORDER_SEED=<u64>` applies a seeded Fisher-Yates shuffle on top
        //   of the shipping order, so other orders can be swept and replayed;
        // - `WELD_ORDER=reversed` reverses the shipping order, reproducing the
        //   sensitive instrument the weld hunt used to keep in a source edit.
        let mut candidates: Vec<(ColliderHandle, ColliderHandle)> = self
            .narrow_phase
            .intersection_pairs()
            .filter(|(_, _, intersecting)| *intersecting)
            .map(|(a, b, _)| (a, b))
            .collect();
        candidates.sort_by_key(|(a, b)| {
            let va = VertexId::from(self.collider_set[*a].user_data);
            let vb = VertexId::from(self.collider_set[*b].user_data);
            if va <= vb { (va, vb) } else { (vb, va) }
        });
        if weld_order_reversed() {
            candidates.reverse();
        }
        if let Some(seed) = weld_order_seed() {
            // Deterministic Fisher-Yates; seed 0 leaves the shipping order.
            let mut state = seed;
            for i in (1..candidates.len()).rev() {
                state = splitmix64(state);
                let j = (state % (i as u64 + 1)) as usize;
                candidates.swap(i, j);
            }
        }

        for (col_h1, col_h2) in candidates {
            let collider1 = &self.collider_set[col_h1];
            let collider2 = &self.collider_set[col_h2];

            #[cfg(feature = "rerun")]
            Self::log_colliders_rerun("collision", [collider1, collider2]);

            let v_id1 = VertexId::from(collider1.user_data);
            let v_id2 = VertexId::from(collider2.user_data);

            // with previous merges two vertices might have become connected or protected that weren't before
            if mesh_graph.vertices.contains_key(v_id1)
                && mesh_graph.vertices.contains_key(v_id2)
                && mesh_graph.halfedge_from_to(v_id1, v_id2).is_none()
                && !self.protected_vertices.contains(&v_id1)
                && !self.protected_vertices.contains(&v_id2)
            {
                #[cfg(feature = "rerun")]
                {
                    Self::log_colliders_rerun("merge", [collider1, collider2]);
                    mesh_graph.log_verts_rerun("merge", &[v_id1, v_id2]);
                }

                // Each individual merge is one journal step (hunt mode).
                #[cfg(feature = "instrumentation")]
                super::journal::record_step(
                    super::journal::JournalOp::MergeOneRing {
                        v1: v_id1,
                        v2: v_id2,
                        flip_threshold_sqr: sculpt_params.min_edge_length_squared,
                    },
                    &self.protected_vertices,
                    &self.protected_halfedges,
                );

                let capture = weld_capture_enabled();
                let boundary_before = if capture {
                    boundary_halfedges(mesh_graph)
                } else {
                    0
                };
                let manifold_before = if capture {
                    manifold_problems(mesh_graph)
                } else {
                    Vec::new()
                };
                let shared_before = if capture {
                    let n = shared_ring_vertices(mesh_graph, v_id1, v_id2);
                    SHARED_HISTOGRAM.with(|h| {
                        let mut h = h.borrow_mut();
                        if h.len() <= n {
                            h.resize(n + 1, 0);
                        }
                        h[n] += 1;
                    });
                    n
                } else {
                    0
                };
                // Snapshot wide enough to grow from: the cut is narrowed again
                // below, once we know which radius actually carries the defect.
                let region_before = if capture && boundary_before == 0 {
                    Some(
                        (1..=WELD_CAPTURE_MAX_RADIUS)
                            .map(|r| capture_region(mesh_graph, v_id1, v_id2, r))
                            .collect::<Vec<_>>(),
                    )
                } else {
                    None
                };
                let protected_before = if capture {
                    self.protected_vertices.clone()
                } else {
                    HashSet::default()
                };
                // The merge pair and their one-rings: the faces around these are
                // the op's input and minimisation must not touch them.
                let keep_before: Vec<VertexId> = if capture {
                    let mut keep = vec![v_id1, v_id2];
                    for v in [v_id1, v_id2] {
                        if let Some(vert) = mesh_graph.vertices.get(v) {
                            for he_id in vert.one_ring(mesh_graph) {
                                if let Some(he) = mesh_graph.halfedges.get(he_id)
                                    && !keep.contains(&he.end_vertex)
                                {
                                    keep.push(he.end_vertex);
                                }
                            }
                        }
                    }
                    keep
                } else {
                    Vec::new()
                };

                let merged = mesh_graph.merge_vertices_one_rings(
                    v_id1,
                    v_id2,
                    sculpt_params.min_edge_length_squared,
                    &mut self.protected_halfedges,
                    &mut self.protected_vertices,
                );

                if capture {
                    let manifold_after = manifold_problems(mesh_graph);
                    if manifold_before.is_empty() && !manifold_after.is_empty() {
                        println!(
                            "// MANIFOLD BROKE at merge {v_id1:?}+{v_id2:?} (boundary {boundary_before} -> {}); first problems:",
                            boundary_halfedges(mesh_graph)
                        );
                        for p in manifold_after.iter().take(6) {
                            println!("//   {p}");
                        }
                    } else if !manifold_before.is_empty() {
                        println!(
                            "// merge {v_id1:?}+{v_id2:?} ran on an ALREADY non-manifold mesh ({} problems)",
                            manifold_before.len()
                        );
                    }
                }

                if let Some(regions) = &region_before {
                    let boundary_after = boundary_halfedges(mesh_graph);
                    if boundary_after > 0 {
                        println!("// FAILING MERGE shared one-ring vertices: {shared_before}");
                        SHARED_HISTOGRAM.with(|h| {
                            println!(
                                "// shared-count histogram over all merges so far: {:?}",
                                h.borrow()
                            );
                        });
                        let flip = sculpt_params.min_edge_length_squared;

                        // The smallest cut that still opens when the merge is
                        // replayed on it alone. A one-ring cut often does not:
                        // the cleanup reaches further than one ring, so the
                        // geometry that decides the outcome can sit outside it.
                        let found = regions.iter().enumerate().find_map(|(i, region)| {
                            region_reproduces(region, v_id1, v_id2, flip, &protected_before)
                                .map(|opened| (i + 1, region, opened))
                        });

                        match found {
                            Some((radius, region, opened)) => {
                                let minimal = minimise_region(
                                    region,
                                    v_id1,
                                    v_id2,
                                    flip,
                                    &protected_before,
                                    &keep_before,
                                );
                                let still = region_reproduces(
                                    &minimal,
                                    v_id1,
                                    v_id2,
                                    flip,
                                    &protected_before,
                                )
                                .unwrap_or(0);
                                println!(
                                    "// reproduces standalone at radius {radius}: \
                                     +{opened} boundary halfedges; minimised from \
                                     {} to {} faces (+{still})",
                                    region.1.len(),
                                    minimal.1.len(),
                                );
                                print_region(&minimal, v_id1, v_id2, flip, &protected_before);
                            }
                            None => {
                                println!(
                                    "// NO STANDALONE REPRO up to radius \
                                     {WELD_CAPTURE_MAX_RADIUS}; printing the widest cut"
                                );
                                print_region(
                                    regions.last().unwrap(),
                                    v_id1,
                                    v_id2,
                                    flip,
                                    &protected_before,
                                );
                            }
                        }

                        panic!(
                            "WELD CAPTURE: merge {v_id1:?}+{v_id2:?} opened the mesh \
                             ({boundary_before} -> {boundary_after} boundary halfedges)"
                        );
                    }
                }

                #[cfg(feature = "rerun")]
                mesh_graph.log_rerun();

                // A failed merge (e.g. degenerate one-rings) leaves the mesh unchanged and
                // returns a default (all-empty) result.
                let merge_failed = merged.removed_vertices.is_empty()
                    && merged.removed_halfedges.is_empty()
                    && merged.removed_faces.is_empty()
                    && merged.added_vertices.is_empty()
                    && merged.added_halfedges.is_empty()
                    && merged.added_faces.is_empty();

                if merge_failed {
                    // Protect the pair so it isn't retried forever and keep looking for
                    // another mergeable pair.
                    error!("Merging {v_id1:?} and {v_id2:?} failed; protecting the pair");
                    self.protected_vertices.insert(v_id1);
                    self.protected_vertices.insert(v_id2);
                    continue;
                }

                return Some(merged);
            }
        }

        self.protected_halfedges.clear();
        self.protected_vertices.clear();

        None
    }

    pub fn sync_mesh_graph(&mut self, mesh_graph: &MeshGraph, sculpt_params: &SculptParams) {
        for (vertex_id, &position) in mesh_graph.positions.iter() {
            let collider_handle = *self
                .vertex_id_to_collider_handle
                .entry(vertex_id)
                .or_insert_with(|| {
                    self.collider_set
                        .insert(Self::create_collider(sculpt_params, vertex_id))
                });

            self.collider_set[collider_handle].set_translation(position);
        }

        let mut removed_vertices = Vec::new();
        for (&vertex_id, &collider_handle) in self.vertex_id_to_collider_handle.iter() {
            if !mesh_graph.vertices.contains_key(vertex_id) {
                self.collider_set.remove(
                    collider_handle,
                    &mut self.island_manager,
                    &mut self.rigid_body_set,
                    false,
                );

                removed_vertices.push(vertex_id);
            }
        }

        for removed_vertex in removed_vertices {
            self.vertex_id_to_collider_handle.remove(&removed_vertex);
        }
    }
}

pub struct TopologyPhysicsHooks<'a> {
    pub mesh_graph: &'a MeshGraph,
    pub protected_vertices: &'a HashSet<VertexId>,
}

impl<'a> PhysicsHooks for TopologyPhysicsHooks<'a> {
    #[instrument(skip_all)]
    fn filter_intersection_pair(&self, context: &PairFilterContext) -> bool {
        let collider1 = &context.colliders[context.collider1];
        let collider2 = &context.colliders[context.collider2];

        let v_id1 = VertexId::from(collider1.user_data);
        let v_id2 = VertexId::from(collider2.user_data);

        if self.protected_vertices.contains(&v_id1) || self.protected_vertices.contains(&v_id2) {
            #[cfg(feature = "rerun")]
            TopologyManager::log_colliders_rerun(
                "collision_filter/rejected_protected",
                [collider1, collider2],
            );

            return false;
        }

        let halfedge_from_to = self.mesh_graph.halfedge_from_to(v_id1, v_id2);

        if halfedge_from_to.is_some() {
            false
        } else {
            // A missing normal means merge directionality cannot be judged; skip the
            // pair. This is transient by design: vertices created mid-cleanup (e.g.
            // hole-punch subdivision) only get normals on the next full recompute.
            let Some(&norm_1) = self
                .mesh_graph
                .vertex_normals
                .as_ref()
                .and_then(|n| n.get(v_id1))
            else {
                return false;
            };
            let Some(&norm_2) = self
                .mesh_graph
                .vertex_normals
                .as_ref()
                .and_then(|n| n.get(v_id2))
            else {
                return false;
            };

            norm_1.dot(norm_2) <= 0.3
        }
    }
}
