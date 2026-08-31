mod resources;
mod systems;

use crate::resources::*;
use crate::systems::*;

use bevy::color::palettes::css::BLACK;
use bevy::input::common_conditions::{input_just_pressed, input_just_released, input_pressed};
use bevy::prelude::*;
// use bevy_inspector_egui::quick::WorldInspectorPlugin;
use bevy_panorbit_camera::PanOrbitCameraPlugin;
use freestyle_sculpt::SculptParams;
use freestyle_sculpt::deformation::*;
use freestyle_sculpt::selectors::*;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_line_number(true)
        .pretty()
        .init();

    App::new()
        .insert_resource(ClearColor(BLACK.into()))
        .init_resource::<GlobalAmbientLight>()
        .insert_resource(SculptParams::new(1.0))
        .insert_non_send(AvailableDeformations::new(vec![
            Box::new(ErodeDilateDeformation::new(0.2)),
            Box::new(ErodeDilateDeformation::new(-10.0)),
            Box::new(TranslateDeformation::new()),
            Box::new(SmoothDeformation::new(0.1)),
        ]))
        .init_resource::<CurrentDeformation>()
        .insert_non_send(AvailableSelections::new(vec![
            Box::new(GeodesicWithFalloff::sphere(1.5, 1.5, SMOOTH_FALLOFF)),
            Box::new(DistanceWithFalloff::sphere(1.5, 1.5, SMOOTH_FALLOFF)),
        ]))
        .init_resource::<CurrentSelection>()
        .add_plugins((
            DefaultPlugins,
            MeshPickingPlugin,
            PanOrbitCameraPlugin,
            // WorldInspectorPlugin::new(),
        ))
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                handle_mouse.run_if(
                    input_pressed(MouseButton::Left)
                        .or_else(input_just_released(MouseButton::Left)),
                ),
                cycle_deformation_mode.run_if(input_just_pressed(KeyCode::KeyD)),
                cycle_selection_mode.run_if(input_just_pressed(KeyCode::KeyS)),
                save_log.run_if(input_just_pressed(KeyCode::KeyL)),
                reset_log.run_if(input_just_pressed(KeyCode::KeyR)),
                handle_morph_open_close.run_if(
                    input_just_pressed(KeyCode::KeyO).or(input_just_pressed(KeyCode::KeyC)),
                ),
            ),
        )
        .run();
}
