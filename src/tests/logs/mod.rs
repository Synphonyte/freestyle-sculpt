mod json;

use crate::{
    SculptParams,
    deformation::{
        DeformationField, ErodeDilateDeformation, SmoothDeformation, TopologyManager,
        TranslateDeformation, punch_hole,
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
        }

        #[cfg(feature = "rerun")]
        {
            mesh_graph::RR.set_time_sequence("replay_step", i as i64 + 1);
            mesh_graph.log_rerun();
        }
    }

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

                let (input_log, mesh_graph) = load_log(name);

                info!("replaying log '{name}'");

                replay_log(mesh_graph, input_log)
                    .unwrap_or_else(|e| panic!("failed to replay log '{name}': {e}"));
            }
        )+
    };
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
}
