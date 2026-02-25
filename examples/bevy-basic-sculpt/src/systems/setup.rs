use bevy::{color::palettes::css::SILVER, ecs::system::SystemState, prelude::*};
use bevy_panorbit_camera::PanOrbitCamera;
use freestyle_sculpt::deformation::TopologyManager;
use mesh_graph::{MeshGraph, primitives::IcoSphere};

use crate::resources::Log;

#[allow(clippy::type_complexity)]
pub fn setup(
    world: &mut World,
    params: &mut SystemState<(
        Commands,
        ResMut<Assets<Mesh>>,
        ResMut<Assets<StandardMaterial>>,
    )>,
) {
    let (mesh, mut mesh_graph) = init_icosphere();
    mesh_graph.compute_vertex_normals();

    world.insert_non_send_resource(TopologyManager::new(&mesh_graph, *world.resource()));
    world.insert_resource(Log::new(&mesh_graph));

    let (mut commands, mut meshes, mut materials) = params.get_mut(world);

    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(Color::from(SILVER))),
        mesh_graph,
        Name::new("Icosphere"),
        Transform::default(),
    ));

    commands.spawn((
        PointLight {
            intensity: 10_000_000.0,
            range: 100.0,
            ..default()
        },
        Transform::from_xyz(8.0, 16.0, 18.0),
    ));

    commands.spawn((
        Camera3d::default(),
        Msaa::Sample4,
        Transform::from_translation(Vec3::new(0.0, 0.0, 17.0)),
        PanOrbitCamera {
            button_orbit: MouseButton::Right,
            button_pan: MouseButton::Middle,
            ..default()
        },
    ));

    params.apply(world);
}

fn init_icosphere() -> (Mesh, MeshGraph) {
    let mesh_graph = MeshGraph::from(IcoSphere {
        subdivisions: 2,
        radius: 3.0,
    });

    (mesh_graph.clone().into(), mesh_graph)
}
