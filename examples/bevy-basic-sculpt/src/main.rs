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
    App::new()
        .insert_resource(ClearColor(BLACK.into()))
        .init_resource::<GlobalAmbientLight>()
        .insert_resource(SculptParams::new(1.0))
        .insert_non_send_resource(AvailableDeformations::new(vec![
            Box::new(TranslateDeformation::default()),
            Box::new(SmoothDeformation::default()),
        ]))
        .init_resource::<CurrentDeformation>()
        .insert_non_send_resource(AvailableSelections::new(vec![
            Box::new(MetricWithFalloff::sphere(1.5, 1.5, SMOOTH_FALLOFF)),
            Box::new(SurfaceMetricWithFalloff::sphere(1.5, 1.5, SMOOTH_FALLOFF)),
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
                    input_pressed(MouseButton::Left).or(input_just_released(MouseButton::Left)),
                ),
                cycle_deformation_mode.run_if(input_just_pressed(KeyCode::KeyD)),
                cycle_selection_mode.run_if(input_just_pressed(KeyCode::KeyS)),
            ),
        )
        .run();
}
