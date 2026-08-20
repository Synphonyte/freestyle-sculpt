use std::fs::File;

use clap::Parser;
use examples::*;
use freestyle_sculpt::SculptParams;
use freestyle_sculpt::deformation::*;
use freestyle_sculpt::selectors::*;
use glam::Vec3;

#[derive(Parser)]
struct Args {
    /// Log file path
    log_file_path: std::path::PathBuf,
}

pub fn main() {
    let selectors: Vec<Box<dyn MeshSelector>> = vec![
        Box::new(GeodesicWithFalloff::sphere(1.5, 1.5, SMOOTH_FALLOFF)),
        Box::new(DistanceWithFalloff::sphere(1.5, 1.5, SMOOTH_FALLOFF)),
    ];
    let mut deformations: Vec<Box<dyn DeformationField>> = vec![
        Box::new(ErodeDilateDeformation::new(0.2)),
        Box::new(ErodeDilateDeformation::new(-1.0)),
        Box::new(TranslateDeformation::new()),
        Box::new(SmoothDeformation::new(0.1)),
    ];
    let params = SculptParams::new(1.0);

    let Args { log_file_path } = Args::parse();

    let log_file_path = match log_file_path.canonicalize() {
        Ok(path) => path,
        Err(err) => {
            eprintln!(
                "Failed to canonicalize log file path: {}\nCWD: {}\nFile path: {}",
                err,
                std::env::current_dir().unwrap().display(),
                log_file_path.display()
            );
            std::process::exit(1);
        }
    };

    println!("Log file path: {}", log_file_path.display());

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_line_number(true)
        .pretty()
        .init();

    let Log {
        mut mesh_graph,
        actions,
    } = serde_json::from_reader(File::open(log_file_path).unwrap()).unwrap();
    mesh_graph.compute_vertex_normals();

    let mut topology_manager = TopologyManager::new(&mesh_graph, params);

    let mut prev_point = Vec3::ZERO;

    for action in actions {
        mesh_graph.optimize_bvh_incremental();

        let intersection = action.ray.cast_ray_and_get_face_id(&mesh_graph);

        // The indices come from an untrusted log file that may have been recorded
        // with a different set of deformations/selectors, so bounds-check them.
        let Some(deformation) = deformations.get_mut(action.deformation) else {
            eprintln!(
                "Skipping action: deformation index {} is out of range ({})",
                action.deformation,
                deformations.len()
            );
            continue;
        };
        let Some(selector) = selectors.get(action.selector) else {
            eprintln!(
                "Skipping action: selector index {} is out of range ({})",
                action.selector,
                selectors.len()
            );
            continue;
        };

        match action.ty {
            EditActionType::MouseDown => {
                if let Some(intersection) = intersection {
                    deformation.on_pointer_down(intersection);
                    prev_point = intersection.point
                }
            }
            EditActionType::MouseUp => {
                // do nothing
            }
            EditActionType::MouseMove => {
                if ray.direction.z.abs() <= f32::EPSILON {
                    continue;
                }

                let cur_point = action
                    .ray
                    .point_at((prev_point.z - action.ray.origin.z) / action.ray.direction.z);

                if prev_point.distance_squared(cur_point) > 0.001 {
                    let mouse_translation = cur_point - prev_point;

                    if deformation.on_pointer_move(&mesh_graph, mouse_translation, intersection) {
                        let strength = if action.deformation == 0 { 1.0 } else { 0.01 };

                        deformation.apply(
                            &mut mesh_graph,
                            selector.as_ref(),
                            strength,
                            params,
                            &mut topology_manager,
                        );
                    }

                    prev_point = cur_point;
                }
            }
        }
    }

    mesh_graph::RR.flush_blocking().unwrap();
}
