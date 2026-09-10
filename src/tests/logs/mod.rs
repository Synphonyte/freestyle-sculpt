mod json;

#[cfg(feature = "instrumentation")]
use crate::deformation::journal::{self, JournalEntry, JournalOp};
use crate::{
    SculptParams,
    deformation::{
        DeformationField, ErodeDilateDeformation, SmoothDeformation, TopologyManager,
        TranslateDeformation, morphological_open_close, punch_hole,
    },
    ray::{FaceIntersection, Ray},
    selectors::{GeodesicWithFalloff, MeshSelector, SMOOTH_FALLOFF},
};
use glam::{Mat4, Vec3};
use hashbrown::HashMap;
use json::{InputLog, InputLogEntry, LoggedSculptState, Matrix4};
use mesh_graph::{MeshGraph, Polygon2, VertexId};
use parry3d::query::{PointProjection, PointQueryWithLocation};
use std::sync::Once;
use tracing::info;

static INIT_TRACING: Once = Once::new();

/// Initializes a tracing subscriber so `info!` (and higher) events are printed while tests run.
///
/// Respects `RUST_LOG` when set; otherwise defaults to `info`. Call once from any test that
/// emits tracing events.
fn init_tracing() {
    INIT_TRACING.call_once(|| {
        let filter = tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
        let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
    });
}

fn load_log(name: &str) -> (InputLog, MeshGraph) {
    let json_name = format!("{name}.json");
    let gltf_name = format!("{name}.glb");

    let input_log: InputLog = serde_json::from_slice(&std::fs::read(&json_name).unwrap()).unwrap();
    let mesh_graph = mesh_graph_from_gltf(&gltf_name);

    (input_log, mesh_graph)
}

/// Builds the starting mesh from a `.gltf` file using mesh-graph's gltf integration.
fn mesh_graph_from_gltf(path: &str) -> MeshGraph {
    let mut mesh_graph = mesh_graph::integrations::gltf::load(path)
        .unwrap_or_else(|e| panic!("failed to load gltf '{path}': {e}"));
    mesh_graph.compute_vertex_normals();
    mesh_graph.rebuild_bvh();

    mesh_graph
}

fn mat4_from_elements(m: &Matrix4) -> Mat4 {
    // three.js `Matrix4.elements` is column-major, which matches `glam::Mat4`.
    Mat4::from_cols_array(&m.elements)
}

/// Maps a recorded `deformation_type` code to the concrete deformation field to replay with.
fn deformation_field_for_type(code: u16) -> Box<dyn DeformationField> {
    match code {
        0 => Box::new(ErodeDilateDeformation::new(0.05)), // Inflate
        1 => Box::new(ErodeDilateDeformation::new(-0.05)), // Deflate
        2 => Box::new(TranslateDeformation::new()),
        3 => Box::new(SmoothDeformation::new(0.1)),
        // HolePunch / LineCut are not pointer-move deformations (they go through
        // `punch_hole`), so fall back to a benign translate field.
        _ => Box::new(TranslateDeformation::new()),
    }
}

fn deformation_name_for_type(code: u16) -> &'static str {
    match code {
        0 => "Inflate",
        1 => "Deflate",
        2 => "Translate",
        3 => "Smooth",
        _ => "Translate",
    }
}

/// Recreates the sculpting state from a recorded snapshot and initializes it for `mesh_graph`.
fn new_state(logged: &LoggedSculptState, mesh_graph: &MeshGraph) -> SculptState {
    let sculpt_params = SculptParams::new(logged.sculpt_params);
    let deformation_field = deformation_field_for_type(logged.deformation_type);

    // The recorded radius/falloff are the raw selector dimensions.
    let falloff = logged.radius * logged.falloff;
    let radius = logged.radius * (1.0 - logged.falloff);
    let selector = GeodesicWithFalloff::sphere(radius, falloff, SMOOTH_FALLOFF);

    SculptState {
        deformation_field,
        selector,
        topology_manager: TopologyManager::new(mesh_graph, sculpt_params),
        strength: logged.strength,
        active: false,
        prev_point: Vec3::ZERO,
        toi: 0.0,
        sculpt_params,
    }
}

/// Extracts the sculpt params from the first input-log entry (needed to rebuild
/// a topology manager when resuming from a dumped state).
#[cfg(feature = "instrumentation")]
fn logged_sculpt_params_from_log(input_log: &InputLog) -> Option<SculptParams> {
    let entry = input_log.0.first()?;
    match entry {
        InputLogEntry::PointerDownSlice { state, .. }
        | InputLogEntry::PointerDownPerspective { state, .. }
        | InputLogEntry::RemoveWithLasso { state, .. } => {
            Some(SculptParams::new(state.sculpt_params))
        }
        InputLogEntry::Weld { sculpt_params, .. } => sculpt_params.map(SculptParams::new),
        _ => None,
    }
}

/// Serializes the journal recorded so far into `dir/journal.json`, replacing any
/// previous contents.
#[cfg(feature = "instrumentation")]
fn write_journal(dir: &std::path::Path) {
    let Some(journal) = journal::journal_entries() else {
        return;
    };
    let path = dir.join("journal.json");
    match serde_json::to_vec_pretty(&journal) {
        Ok(bytes) => {
            if let Err(e) = std::fs::write(&path, bytes) {
                eprintln!("journal: could not write {}: {e:?}", path.display());
            } else {
                eprintln!(
                    "journal: wrote {} steps to {}",
                    journal.len(),
                    path.display()
                );
            }
        }
        Err(e) => eprintln!("journal: could not serialize: {e:?}"),
    }
}

/// Durability write: the first time a dump is seen, snapshot the journal next to it
/// so a run killed right after the dump can still be resumed.
///
/// Deliberately once-only. This runs after every replayed entry, and the journal is
/// megabytes of JSON, so rewriting it each time would cost quadratic I/O once a dump
/// exists. [`write_journal_final`] replaces this partial
/// snapshot with the complete journal when the run finishes normally.
#[cfg(feature = "instrumentation")]
fn write_journal_if_dumped() {
    static WRITTEN: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    if WRITTEN.get().is_some() {
        return;
    }
    if let Some(dir) = mesh_graph::state_dump_dir() {
        let _ = WRITTEN.set(());
        write_journal(&dir);
    }
}

/// Overwrites the durability snapshot with the complete journal, so the resume range
/// covers every step recorded *after* the dump - not just those up to the entry the
/// dump happened in.
#[cfg(feature = "instrumentation")]
fn write_journal_final() {
    if let Some(dir) = mesh_graph::state_dump_dir() {
        write_journal(&dir);
    }
}

/// Reads the journal from a dumped state directory.
#[cfg(feature = "instrumentation")]
fn load_journal(dir: &std::path::Path) -> Result<Vec<JournalEntry>, String> {
    let path = dir.join("journal.json");
    let bytes =
        std::fs::read(&path).map_err(|e| format!("could not read {}: {e}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("could not parse {}: {e}", path.display()))
}

/// Finds the newest snapshot in a dumped state-history directory: the `state_*.json`
/// file with the highest `pos_<n>` in its name — the ring's most recent entry, i.e.
/// the state closest to the corruption that triggered the dump. Returns the state
/// file path and its journal step index.
#[cfg(feature = "instrumentation")]
fn newest_snapshot(dump_dir: &std::path::Path) -> Option<(usize, std::path::PathBuf)> {
    let mut best: Option<(usize, std::path::PathBuf)> = None;
    for entry in std::fs::read_dir(dump_dir).ok()? {
        let Ok(entry) = entry else { continue };
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.starts_with("state_") || !name.ends_with(".json") {
            continue;
        }
        // A name without a parseable `pos_<n>` cannot be placed in the journal, and
        // guessing step 0 would replay the whole journal against a mid-run state.
        let Some(pos) = name.split("pos_").nth(1).and_then(|s| {
            s.chars()
                .take_while(|c| c.is_ascii_digit())
                .collect::<String>()
                .parse::<usize>()
                .ok()
        }) else {
            continue;
        };
        if best.as_ref().is_none_or(|(best_pos, _)| pos > *best_pos) {
            best = Some((pos, entry.path()));
        }
    }
    best
}

/// Applies the journal steps starting at `skip` (entries `0..skip` are assumed to
/// be already contained in `mesh_graph`) — the resume path. The topology manager
/// is rebuilt from the mesh and the journal's marked-set snapshots, mirroring the
/// original run's per-step state. Returns the mesh for post-replay inspection.
#[cfg(feature = "instrumentation")]
fn replay_journal(
    mut mesh_graph: MeshGraph,
    journal: Vec<JournalEntry>,
    params: SculptParams,
    skip: usize,
) -> Result<MeshGraph, String> {
    let mut topology_manager = TopologyManager::new(&mesh_graph, params);

    let mut count: usize = 0;
    for (i, entry) in journal.into_iter().enumerate().skip(skip) {
        count += 1;
        mesh_graph::set_replay_position(i as u64);
        topology_manager.protected_vertices = entry.marked_vertices.iter().copied().collect();
        topology_manager.protected_halfedges = entry.marked_halfedges.iter().copied().collect();

        match entry.op {
            JournalOp::Collapse { min_len_sqr } => {
                mesh_graph.collapse_until_edges_above_min_length(
                    min_len_sqr,
                    &mut topology_manager.protected_vertices,
                );
            }
            JournalOp::Subdivide { max_len_sqr } => {
                mesh_graph.subdivide_until_edges_below_max_length(
                    max_len_sqr,
                    &mut topology_manager.protected_halfedges,
                    &mut topology_manager.protected_vertices,
                );
            }
            JournalOp::MergeOneRing {
                v1,
                v2,
                flip_threshold_sqr,
            } => {
                mesh_graph.merge_vertices_one_rings(
                    v1,
                    v2,
                    flip_threshold_sqr,
                    &mut topology_manager.protected_halfedges,
                    &mut topology_manager.protected_vertices,
                );
            }
        }
    }

    eprintln!(
        "replayed {count} journal steps (from {skip}); final mesh has {} vertices",
        mesh_graph.vertices.len()
    );

    write_journal_final();

    Ok(mesh_graph)
}

/// Per-pointer-session sculpting state carried across PointerDown → PointerMove.
struct SculptState {
    deformation_field: Box<dyn DeformationField>,
    selector: GeodesicWithFalloff,
    topology_manager: TopologyManager,
    strength: f32,
    active: bool,
    prev_point: Vec3,
    toi: f32,
    sculpt_params: SculptParams,
}

fn sculpt_or_select(
    mesh_graph: &mut MeshGraph,
    state: &mut SculptState,
    current_point: Vec3,
    intersection: Option<FaceIntersection>,
) -> Option<HashMap<VertexId, f32>> {
    let pointer_translation = current_point - state.prev_point;

    let result = if state.active {
        if state
            .deformation_field
            .on_pointer_move(mesh_graph, pointer_translation, intersection)
        {
            let result = state.deformation_field.apply(
                mesh_graph,
                &state.selector,
                state.strength,
                state.sculpt_params,
                &mut state.topology_manager,
            );
            mesh_graph.optimize_bvh_incremental();
            Some(result)
        } else {
            None
        }
    } else {
        Some(
            state
                .selector
                .select(mesh_graph, &intersection?)
                .vertex_to_weight,
        )
    };

    state.prev_point = current_point;

    result
}

// -- perspective sculpting ----------------------------------------------------

fn sculpt_on_pointer_down_perspective(
    mesh_graph: &mut MeshGraph,
    state: &mut SculptState,
    local_ray: Ray,
) -> Option<()> {
    let intersection = local_ray.cast_ray_and_get_face_id(mesh_graph)?;

    state.deformation_field.on_pointer_down(intersection);
    state.prev_point = intersection.point;
    state.toi = intersection.toi;
    state.active = true;

    Some(())
}

fn sculpt_on_pointer_move_perspective(
    mesh_graph: &mut MeshGraph,
    state: &mut SculptState,
    local_ray: Ray,
) -> Option<HashMap<VertexId, f32>> {
    let intersection = local_ray.cast_ray_and_get_face_id(mesh_graph)?;
    let current_point = local_ray.point_at(state.toi);

    sculpt_or_select(mesh_graph, state, current_point, Some(intersection))
}

// -- slice sculpting -----------------------------------------------------------

fn sculpt_on_pointer_down_slice(
    mesh_graph: &MeshGraph,
    state: &mut SculptState,
    local_point: Vec3,
) {
    let Some((PointProjection { point, .. }, face)) =
        mesh_graph.project_local_point_and_get_location_with_max_dist(local_point, true, f32::MAX)
    else {
        return;
    };

    let intersection = FaceIntersection {
        point,
        face,
        toi: 0.0,
    };

    state.deformation_field.on_pointer_down(intersection);
    state.prev_point = local_point;
    state.active = true;
}

fn sculpt_on_pointer_move_slice(
    mesh_graph: &mut MeshGraph,
    state: &mut SculptState,
    local_point: Vec3,
) -> Option<HashMap<VertexId, f32>> {
    let intersection = mesh_graph
        .project_local_point_and_get_location_with_max_dist(local_point, true, f32::MAX)
        .map(|(PointProjection { point, .. }, face)| FaceIntersection {
            point,
            face,
            toi: 0.0,
        });

    sculpt_or_select(mesh_graph, state, local_point, intersection)
}

// -- lasso via hole_punch ------------------------------------------------------

/// Removes whatever the lasso polygon (NDC) encloses, using the local `punch_hole` operation.
fn sculpt_with_lasso(
    mesh_graph: &mut MeshGraph,
    state: &mut SculptState,
    lasso_points: Vec<Vec3>,
    object_to_camera: Mat4,
    projection: Mat4,
) {
    let hole_shape = Polygon2 {
        vertices: lasso_points.into_iter().map(|p| p.truncate()).collect(),
    };

    punch_hole(
        mesh_graph,
        hole_shape,
        object_to_camera,
        projection,
        &state.sculpt_params,
        &mut state.topology_manager,
    );
}

/// Replays the app's weld: a morphological close by `amount`.
///
/// Mirrors `voxel_wasm::sculpt::weld_meshes`, which is the shared core behind both the
/// `weldMeshes` wasm binding and that project's own log player. The mesh merging step
/// there is already done here - the log's `.glb` is the merged mesh - so this is the
/// close that follows it.
///
/// Growing every surface by `amount` bridges the gaps between disconnected sheets and
/// fuses them, then eroding by the same amount returns to the original silhouette. A
/// weld is always a close, and a close needs a negative amount, so the log's unsigned
/// tool value is forced negative exactly as `weld_meshes` does.
///
/// `sculpt_params` is the app's own recorded value, so this cannot drift from what
/// production used the way re-deriving it here would.
fn sculpt_weld(mesh_graph: &mut MeshGraph, amount: f32, sculpt_params: f32) {
    let sculpt_params = SculptParams::new(sculpt_params);
    let mut topology_manager = TopologyManager::new(mesh_graph, sculpt_params);

    morphological_open_close(
        mesh_graph,
        &sculpt_params,
        &mut topology_manager,
        -amount.abs(),
        (),
    );
}

// -- replay ---------------------------------------------------------------------

fn replay_log(mut mesh_graph: MeshGraph, input_log: InputLog) -> Result<(), String> {
    let mut active_state: Option<SculptState> = None;
    let entries = input_log.0;

    let total = entries
        .iter()
        .position(|e| matches!(e, InputLogEntry::PointerUp))
        .map(|i| i + 1)
        .unwrap_or(entries.len());

    #[cfg(feature = "rerun")]
    {
        mesh_graph::RR.set_time_sequence("replay_step", 0);
        mesh_graph.log_rerun();
    }

    for (i, entry) in entries.into_iter().enumerate() {
        match entry {
            InputLogEntry::PointerDownSlice { point, state } => {
                info!(
                    "[{i}] PointerDownSlice with deformation type {:?}",
                    deformation_name_for_type(state.deformation_type)
                );

                let mut state = new_state(&state, &mesh_graph);
                sculpt_on_pointer_down_slice(&mesh_graph, &mut state, point.into());
                active_state = Some(state);
            }
            InputLogEntry::PointerMoveSlice { point } => {
                info!("[{i}] PointerMoveSlice");

                if let Some(state) = active_state.as_mut() {
                    sculpt_on_pointer_move_slice(&mut mesh_graph, state, point.into());
                } else {
                    eprintln!(
                        "[{i}] PointerMoveSlice without a preceding PointerDownSlice; skipping"
                    );
                }
            }
            InputLogEntry::PointerDownPerspective { local_ray, state } => {
                info!(
                    "[{i}] PointerDownPerspective with deformation type {:?}",
                    deformation_name_for_type(state.deformation_type)
                );

                let mut state = new_state(&state, &mesh_graph);
                sculpt_on_pointer_down_perspective(&mut mesh_graph, &mut state, local_ray.into());
                active_state = Some(state);
            }
            InputLogEntry::PointerMovePerspective { local_ray } => {
                info!("[{i}] PointerMovePerspective");

                if let Some(state) = active_state.as_mut() {
                    sculpt_on_pointer_move_perspective(&mut mesh_graph, state, local_ray.into());
                } else {
                    eprintln!(
                        "[{i}] PointerMovePerspective without a preceding PointerDownPerspective; skipping"
                    );
                }
            }
            InputLogEntry::PointerUp => {
                info!("[{i}] PointerUp");

                break;
            }
            InputLogEntry::RemoveWithLasso {
                points,
                object_to_camera_matrix,
                projection_matrix,
                state,
            } => {
                info!("[{i}] RemoveWithLasso");

                let mut state = new_state(&state, &mesh_graph);
                let lasso_points = points
                    .iter()
                    .map(|p| Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32))
                    .collect();

                sculpt_with_lasso(
                    &mut mesh_graph,
                    &mut state,
                    lasso_points,
                    mat4_from_elements(&object_to_camera_matrix),
                    mat4_from_elements(&projection_matrix),
                );
            }
            InputLogEntry::Weld {
                amount,
                sculpt_params,
            } => {
                info!("[{i}] Weld amount={amount}");

                // Older exports (and exports whose preprocessing failed) carry no
                // params. Re-deriving them here would silently diverge from the app,
                // so refuse rather than guess.
                let sculpt_params = sculpt_params.ok_or_else(|| {
                    format!(
                        "[{i}] Weld entry has no `sculptParams`; re-export the log with a \
                         voxel-wasm that records them"
                    )
                })?;

                sculpt_weld(&mut mesh_graph, amount, sculpt_params);
                active_state = None;
            }
        }

        #[cfg(feature = "rerun")]
        {
            mesh_graph::RR.set_time_sequence("replay_step", i as i64 + 1);
            mesh_graph.log_rerun();
        }

        // Hunt mode: a corruption dump can happen inside this entry's ops; write
        // the operation journal next to it immediately so the run can be resumed
        // even when the process is killed right after the dump.
        #[cfg(feature = "instrumentation")]
        write_journal_if_dumped();
    }

    // Replace the mid-run durability snapshot with the full journal.
    #[cfg(feature = "instrumentation")]
    write_journal_final();

    eprintln!(
        "replayed {total} entries; final mesh has {} vertices",
        mesh_graph.vertices.len()
    );

    #[cfg(feature = "rerun")]
    {
        mesh_graph.log_rerun();
        mesh_graph::RR.flush_blocking().unwrap();
    }

    Ok(())
}

/// Generates one `#[test]` per recorded log (a `.glb`/`.json` pair), so each
/// log runs and reports as its own test case. Add new logs here as they are
/// recorded.
macro_rules! log_tests {
    ($($test_name:ident: $log_name:literal),+ $(,)?) => {
        $(
            #[test]
            fn $test_name() {
                init_tracing();

                // Keep the `src/tests/logs/` prefix so `load_log` can reconstruct
                // both the `.json` and `.glb` paths relative to the crate root.
                let name = concat!("src/tests/logs/", $log_name);

                let (input_log, mesh_graph_from_gltf) = load_log(name);

                // Resume support (hunt mode): `MESH_GRAPH_RESUME_STATE=<state file>`
                // plus `MESH_GRAPH_RESUME_INDEX=<step>` continues a run whose state
                // history was dumped. Each mesh-graph topology op call (collapse /
                // subdivide / individual merge_one_ring) is one journal step; the
                // step index of a dumped state is in the state file name and
                // `meta.txt`. The remaining journal steps are replayed on the
                // loaded state instead of the input log.
                #[cfg(feature = "instrumentation")]
                if let Some(state_path) = std::env::var_os("MESH_GRAPH_RESUME_STATE") {
                    let index = std::env::var("MESH_GRAPH_RESUME_INDEX")
                        .ok()
                        .and_then(|p| p.parse::<usize>().ok())
                        .unwrap_or(0);
                    let state = MeshGraph::load_state(&state_path)
                        .expect("failed to load resume state");
                    let dump_dir = std::path::Path::new(&state_path)
                        .parent()
                        .expect("resume state path has no parent")
                        .to_path_buf();
                    let journal = load_journal(&dump_dir)
                        .unwrap_or_else(|e| panic!("failed to load resume journal: {e}"));
                    let params = logged_sculpt_params_from_log(&input_log)
                        .expect("input log has no sculpt params for resume");
                    eprintln!(
                        "resuming from state '{}' at journal step {index}; replaying steps {}+..",
                        state_path.to_string_lossy(),
                        index + 1
                    );
                    if let Err(e) = replay_journal(state, journal, params, index + 1) {
                        panic!("resume failed: {e}");
                    }
                    return;
                }

                info!("replaying log '{name}'");

                replay_log(mesh_graph_from_gltf, input_log)
                    .unwrap_or_else(|e| panic!("failed to replay log '{name}': {e}"));
            }
        )+
    };
}

/// Replays the remaining operation journal on a dumped pre-corruption state and
/// asserts both that mesh-graph raised no integrity violation during the replay and
/// that the replayed mesh's outgoing halfedge lists still match the lists rebuilt
/// from its halfedges.
/// This is the regression test for the outgoing-list wipe (root cause of the
/// rare layer-4 corruption): `src/tests/trace_runs/dump_1/` is a state-history
/// dump from a pre-fix `log_002` run. Before the fix via
/// [`MeshGraph::rebuild_vertex_outgoing_list`], the steps right after the dump
/// (journal steps 877/878) reliably produced an "OUTGOING" corruption report;
/// with the fix, the entire remaining journal (~1300 steps) replays cleanly.
///
/// Any dump directory with a `journal.json` + `state_*.json` files can be used
/// in the same way (the newest snapshot is selected automatically).
#[test]
#[cfg(feature = "instrumentation")]
fn snapshot_replay_dump_1() {
    use mesh_graph::{integrity_violation_reported, reset_integrity_violation};

    init_tracing();

    // The flag is per-thread; clear it so this assertion covers only our own replay.
    reset_integrity_violation();

    let dump_dir = std::path::PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/tests/trace_runs/dump_1"
    ));
    let (pos, state_path) = newest_snapshot(&dump_dir).unwrap_or_else(|| {
        panic!(
            "no state_*.json in {} - the dump fixture is a committed part of this \
             regression test; restore it from git",
            dump_dir.display()
        )
    });
    eprintln!(
        "snapshot replay: state {} (journal step {pos}), replaying steps {}+..",
        state_path.display(),
        pos + 1
    );

    let state = MeshGraph::load_state(&state_path).expect("failed to load snapshot state");
    let journal = load_journal(&dump_dir).expect("failed to load snapshot journal");

    // The sculpt params come from the input log the snapshot was captured from
    // (log 002); the .glb mesh is not needed because the mesh is the snapshot.
    let json_bytes = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/tests/logs/002.json"
    ))
    .expect("failed to read src/tests/logs/002.json");
    let input_log: InputLog =
        serde_json::from_slice(&json_bytes).expect("failed to parse src/tests/logs/002.json");
    let params = logged_sculpt_params_from_log(&input_log)
        .expect("log 002 has no sculpt params for snapshot replay");

    // The snapshot was captured after journal step `pos`; replay the rest.
    let mut mesh = replay_journal(state, journal, params, pos + 1).expect("snapshot replay failed");

    // Catches the chain/twin probes, which report through mesh-graph's own inspectors.
    assert!(
        !integrity_violation_reported(),
        "mesh-graph reported an integrity violation during the snapshot replay"
    );

    // The outgoing-list wipe is checked directly on the mesh as well, so the specific
    // regression this fixture guards stays covered even if the probes are quiet.
    //
    // The wipe shows up as a *membership* difference against the lists rebuilt from the
    // halfedges. Order is explicitly arbitrary (`MeshGraph::outgoing_halfedges` is
    // documented as unordered, and mesh-graph reports order-only deviations as benign),
    // so the comparison is set-based.
    let stored: HashMap<VertexId, hashbrown::HashSet<mesh_graph::HalfedgeId>> = mesh
        .outgoing_halfedges
        .iter()
        .map(|(vertex, halfedges)| (vertex, halfedges.iter().copied().collect()))
        .collect();

    mesh.rebuild_outgoing_halfedges();

    let diverged = mesh
        .outgoing_halfedges
        .iter()
        .filter(|(vertex, rebuilt)| {
            let truth: hashbrown::HashSet<mesh_graph::HalfedgeId> =
                rebuilt.iter().copied().collect();
            stored.get(vertex).is_none_or(|kept| *kept != truth)
        })
        .map(|(vertex, _)| vertex)
        .collect::<Vec<_>>();

    assert!(
        diverged.is_empty(),
        "outgoing halfedge lists diverged from the rebuild ground truth at {} vertices \
         (first: {:?}); the outgoing-list wipe has regressed",
        diverged.len(),
        diverged.first()
    );
}

/// Regression test for the weld-hole defect: `merge_vertices_one_rings` on a
/// common-ring vertex pair (the rings share vertices) ended its op with
/// boundary halfedges (`face=None`) — holes in a mesh that must stay closed.
///
/// The fixture is `src/tests/trace_runs/weld_hole_dump/` — a state-history dump
/// from a `log_010_weld` run (journal step 1, ~4.5k vertices). The remaining
/// journal steps (2..) are replayed on the state; step 3 (merge 1487→1490)
/// deterministically leaves 7-12 boundary halfedges. Unlike the chain/twin
/// probes, the hole check is only active with `MESH_GRAPH_HOLE_CHECK=1`, so
/// this test verifies the defect directly on the replayed mesh instead of
/// relying on process-global instrumentation env vars.
#[test]
#[cfg(feature = "instrumentation")]
fn snapshot_replay_weld_hole() {
    init_tracing();

    let dump_dir = std::path::PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/tests/trace_runs/weld_hole_dump"
    ));
    let (pos, state_path) = newest_snapshot(&dump_dir).unwrap_or_else(|| {
        panic!(
            "no state_*.json in {} - the dump fixture is a committed part of this \
             regression test; restore it from git",
            dump_dir.display()
        )
    });
    eprintln!(
        "snapshot replay: state {} (journal step {pos}), replaying steps {}+..",
        state_path.display(),
        pos + 1
    );

    let state = MeshGraph::load_state(&state_path).expect("failed to load snapshot state");
    let journal = load_journal(&dump_dir).expect("failed to load snapshot journal");

    // A `Weld` entry carries no sculpt state, so the params are derived from the
    // snapshot mesh, the same way the app derives them (see `sculpt_weld`).
    let json_bytes = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/tests/logs/010_weld.json"
    ))
    .expect("failed to read src/tests/logs/010_weld.json");
    let input_log: InputLog =
        serde_json::from_slice(&json_bytes).expect("failed to parse src/tests/logs/010_weld.json");
    let params = logged_sculpt_params_from_log(&input_log)
        .expect("log 010_weld has no sculpt params for snapshot replay");

    let mesh = replay_journal(state, journal, params, pos + 1).expect("snapshot replay failed");

    let boundary = mesh
        .halfedges
        .values()
        .filter(|he| he.face.is_none())
        .count();
    assert_eq!(
        boundary, 0,
        "weld merge left {boundary} boundary halfedges (holes) after the replay"
    );
}

log_tests! {
    log_001: "001",
    log_002: "002",
    log_003: "003",
    log_004: "004",
    log_005: "005",
    log_006: "006",
    log_007: "007",
    log_008: "008",
    log_009: "009",
    log_010_weld: "010_weld",
}
