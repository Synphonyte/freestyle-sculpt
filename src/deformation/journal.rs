//! Operation journal for the layer-4 corruption hunt.
//!
//! When `MESH_GRAPH_DANGLING_CHECK=1` is set, every mesh-graph topology op the
//! deformation pipeline performs is recorded here as one *step*:
//!
//! - `collapse_until_edges_above_min_length`  → one step
//! - `subdivide_until_edges_below_max_length` → one step
//! - `merge_vertices_one_rings`               → one step (each individual merge)
//!
//! Each step also advances mesh-graph's replay position
//! (`mesh_graph::set_replay_position`), so the state-history ring snapshots map
//! 1:1 to journal steps. The journal itself is serialized and written next to the
//! dumped states by the replay harness (`tests::logs`), which also implements the
//! resume path (replaying the remaining steps on a dumped state).

use std::sync::Mutex;

use hashbrown::HashSet;
use mesh_graph::{HalfedgeId, VertexId};
use serde::{Deserialize, Serialize};

/// The mesh-graph topology op of one journal step and its parameters.
/// The ids stay valid in the context of the state the step is replayed on.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum JournalOp {
    Collapse {
        min_len_sqr: f32,
    },
    Subdivide {
        max_len_sqr: f32,
    },
    MergeOneRing {
        v1: VertexId,
        v2: VertexId,
        flip_threshold_sqr: f32,
    },
}

/// One journal step: the op plus the topology manager's protected sets as they
/// were right before the call (the ops mutate them in the original run; replaying
/// the snapshot restores the same pre-call state).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JournalEntry {
    pub op: JournalOp,
    pub marked_vertices: Vec<VertexId>,
    pub marked_halfedges: Vec<HalfedgeId>,
}

static JOURNAL: Mutex<Option<Vec<JournalEntry>>> = Mutex::new(None);

static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

fn enabled() -> bool {
    *ENABLED.get_or_init(|| std::env::var_os("MESH_GRAPH_DANGLING_CHECK").is_some())
}

/// True when the journal records steps (hunt mode).
pub fn journal_enabled() -> bool {
    enabled()
}

/// Records one step *before* the corresponding mesh op runs and advances mesh-graph's
/// replay position to this step's index, so the state-history ring snapshot that is
/// pushed at the op's (verified) end carries this step's index.
pub fn record_step(
    op: JournalOp,
    marked_vertices: &HashSet<VertexId>,
    marked_halfedges: &HashSet<HalfedgeId>,
) {
    if !enabled() {
        return;
    }

    let entry = JournalEntry {
        op,
        marked_vertices: marked_vertices.iter().copied().collect(),
        marked_halfedges: marked_halfedges.iter().copied().collect(),
    };

    let index = {
        let mut journal = JOURNAL.lock().unwrap();
        let journal = journal.get_or_insert_with(Vec::new);
        journal.push(entry);
        journal.len() - 1
    };

    mesh_graph::set_replay_position(index as u64);
}

/// The journal steps recorded so far (for the harness to serialize on demand).
pub fn journal_entries() -> Option<Vec<JournalEntry>> {
    JOURNAL.lock().unwrap().clone()
}
