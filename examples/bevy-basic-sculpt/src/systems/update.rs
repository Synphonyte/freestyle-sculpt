use bevy::picking::backend::ray::RayMap;
use bevy::prelude::*;
use freestyle_sculpt::SculptParams;
use freestyle_sculpt::ray::Ray;
use hashbrown::HashSet;
use mesh_graph::{HalfedgeId, MeshGraph, VertexId};

use crate::resources::*;

pub fn cycle_deformation_mode(
    mut current_index: ResMut<CurrentDeformation>,
    available_deformations: NonSend<AvailableDeformations>,
) {
    **current_index += 1;
    if **current_index >= available_deformations.len() {
        **current_index = 0;
    }
}

pub fn cycle_selection_mode(
    mut current_index: ResMut<CurrentSelection>,
    available_selections: NonSend<AvailableSelections>,
) {
    **current_index += 1;
    if **current_index >= available_selections.len() {
        **current_index = 0;
    }
}

pub fn handle_mouse(
    ray_map: Res<RayMap>,
    buttons: Res<ButtonInput<MouseButton>>,
    sculpt_params: Res<SculptParams>,
    mut meshes: ResMut<Assets<Mesh>>,
    current_deformation: Res<CurrentDeformation>,
    mut available_deformations: NonSendMut<AvailableDeformations>,
    current_selection: Res<CurrentSelection>,
    available_selections: NonSend<AvailableSelections>,
    picking_cameras: Query<&Camera>,
    mut mesh_graphs: Query<(&mut MeshGraph, &Mesh3d)>,
    mut prev_point: Local<Vec3>,
    mut deformation_active: Local<bool>,
    mut protected_halfedges: Local<HashSet<HalfedgeId>>,
    mut protected_vertices: Local<HashSet<VertexId>>,
) -> Result {
    let (mut mesh_graph, mesh_handle) = mesh_graphs.single_mut()?;

    let mesh = meshes.get_mut(mesh_handle).unwrap();

    let mut available_deformations = available_deformations.get_mut(**current_deformation);
    let deformation_field = available_deformations.as_mut().unwrap();

    for (&ray_id, &ray) in ray_map.iter() {
        if !picking_cameras.contains(ray_id.camera) {
            continue;
        };

        let selector = available_selections.get(**current_selection).unwrap();

        let ray = Ray::from(ray);
        let intersection = ray.cast_ray_and_get_face_id(&mesh_graph);

        if buttons.just_pressed(MouseButton::Left) {
            // Mouse down
            if let Some(intersection) = intersection {
                deformation_field.on_pointer_down(&mesh_graph, selector.as_ref(), intersection);

                *deformation_active = true;
                *prev_point = intersection.point;
            } else {
                *deformation_active = false;
            }
        } else if buttons.just_released(MouseButton::Left) {
            // Mouse up
            if *deformation_active {
                mesh_graph.optimize_bvh_incremental();

                *deformation_active = false;
            }
        } else {
            // Mouse move
            if *deformation_active {
                // o.z + d.z * t = p.z
                // t = (p.z - o.z) / d.z
                let cur_point = ray.point_at((prev_point.z - ray.origin.z) / ray.direction.z);

                if prev_point.distance_squared(cur_point) > 0.001 {
                    let mouse_translation = cur_point - *prev_point;

                    if deformation_field.on_pointer_move(
                        &mesh_graph,
                        selector.as_ref(),
                        mouse_translation,
                        intersection,
                    ) {
                        let strength = if **current_deformation == 0 {
                            1.0
                        } else {
                            0.01
                        };

                        protected_halfedges.clear();
                        protected_vertices.clear();

                        deformation_field.apply(
                            &mut mesh_graph,
                            strength,
                            *sculpt_params,
                            &mut *protected_halfedges,
                            &mut *protected_vertices,
                        );

                        *mesh = mesh_graph.clone().into();
                    }

                    *prev_point = cur_point;
                }

                mesh_graph.optimize_bvh_incremental();
            }
        }
    }

    Ok(())
}
