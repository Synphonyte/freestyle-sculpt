//! Test-only observation points inside [`super::cleanup_mesh`].
//!
//! A weld's cleanup alternates one collapse/subdivide pair with one merge
//! (`update_collisions_and_merge` performs a single merge per call) until no
//! merge is left. Those are plain calls inside that loop, so a test has no seam
//! to measure at, and mesh-graph's merge probe only sees the merge op (see
//! `end_wire` in `mesh-graph/src/ops/merge_one_ring/rim_probe.rs`). A defect the
//! *cleanup* opens between two merges is therefore unattributable - and it is
//! invisible in the welded result as well whenever a later merge closes it again.
//!
//! [`record`] samples the mesh at the three points that separate those phases;
//! [`with_recording`] arms the sampler for one closure and returns the samples.
//! Nothing here is compiled outside `cfg(test)`, and the sampler is inert (one
//! thread-local flag read) until a test arms it, so no other test pays for it.
//!
//! The measurement is a policy-free count; which of these counters may not grow,
//! and in which phase, is the calling test's decision.

use std::cell::{Cell, RefCell};

use mesh_graph::{FaceId, MeshGraph, VertexId};

use crate::hash::HashMap;

/// Which point of a [`super::cleanup_mesh`] iteration a [`Sample`] was taken at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Phase {
    /// Top of an iteration, before the collapse: the state the cleanup phase
    /// starts from - the previous merge, or on the first iteration of a call the
    /// caller's own vertex movement and sheet zip.
    Entry,
    /// After the collapse/subdivide pair, before the merge: the state the
    /// cleanup phase hands to the merge.
    Cleanup,
    /// After the merge that `update_collisions_and_merge` performed.
    Merge,
}

/// The structural damage counters of one mesh state, measured in one pass.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Sample {
    pub phase: Phase,
    pub faces: usize,
    /// Halfedges with no face.
    pub boundary_halves: usize,
    /// Boundary halves whose twin is face-less too: an edge with no face on
    /// either side, which no hole filler can repair.
    pub wire_halves: usize,
    /// Undirected vertex pairs joined by more than two halfedges.
    pub non_manifold_edges: usize,
    /// Vertices whose incident faces fall short of one fan - the star check of
    /// the tests' `assert_manifold`, as a count.
    pub pinched_vertices: usize,
}

thread_local! {
    static RECORDING: Cell<bool> = const { Cell::new(false) };
    static SAMPLES: RefCell<Vec<Sample>> = const { RefCell::new(Vec::new()) };
}

/// Samples `mesh_graph` if a [`with_recording`] closure is running on this
/// thread, and does nothing otherwise.
pub(crate) fn record(phase: Phase, mesh_graph: &MeshGraph) {
    if !RECORDING.with(Cell::get) {
        return;
    }

    let sample = measure(phase, mesh_graph);
    SAMPLES.with(|samples| samples.borrow_mut().push(sample));
}

/// Runs `f` with the sampler armed and returns its value together with the
/// samples it produced, in call order. The sampler is disarmed again on the way
/// out, panics included.
pub(crate) fn with_recording<R>(f: impl FnOnce() -> R) -> (R, Vec<Sample>) {
    struct Disarm;

    impl Drop for Disarm {
        fn drop(&mut self) {
            RECORDING.with(|recording| recording.set(false));
        }
    }

    let _disarm = Disarm;
    SAMPLES.with(|samples| samples.borrow_mut().clear());
    RECORDING.with(|recording| recording.set(true));

    let value = f();

    let samples = SAMPLES.with(|samples| std::mem::take(&mut *samples.borrow_mut()));
    (value, samples)
}

fn measure(phase: Phase, mesh_graph: &MeshGraph) -> Sample {
    let mut boundary_halves = 0;
    let mut wire_halves = 0;
    let mut halves_per_edge: HashMap<(VertexId, VertexId), u8> = HashMap::default();

    for (_, he) in mesh_graph.halfedges.iter() {
        if he.face.is_none() {
            boundary_halves += 1;
            if he
                .twin
                .and_then(|twin| mesh_graph.halfedges.get(twin))
                .is_some_and(|twin| twin.face.is_none())
            {
                wire_halves += 1;
            }
        }

        let Some(start) = he.start_vertex(mesh_graph) else {
            continue;
        };
        let (a, b) = (start, he.end_vertex);
        *halves_per_edge
            .entry(if a < b { (a, b) } else { (b, a) })
            .or_default() += 1;
    }

    let non_manifold_edges = halves_per_edge
        .values()
        .filter(|halves| **halves > 2)
        .count();

    let mut pinched_vertices = 0;
    for (v_id, out_hes) in mesh_graph.outgoing_halfedges.iter() {
        if !mesh_graph.vertices.contains_key(v_id) {
            continue;
        }

        let mut faces: Vec<FaceId> = Vec::new();
        let mut edges = 0;
        for he_id in out_hes {
            let Some(he) = mesh_graph.halfedges.get(*he_id) else {
                continue;
            };
            edges += 1;
            if let Some(face_id) = he.face
                && !faces.contains(&face_id)
            {
                faces.push(face_id);
            }
        }

        // A manifold star has one face per edge, less one when it ends on a
        // boundary; fewer means the star falls into more than one fan.
        if edges > 0 && faces.len() + 1 < edges {
            pinched_vertices += 1;
        }
    }

    Sample {
        phase,
        faces: mesh_graph.faces.len(),
        boundary_halves,
        wire_halves,
        non_manifold_edges,
        pinched_vertices,
    }
}
