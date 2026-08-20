use glam::{Mat4, Vec3Swizzles};
use hashbrown::HashSet;
use mesh_graph::{MeshGraph, Polygon2};
use parry3d::bounding_volume::Aabb;
use tracing::{error, instrument, warn};

use crate::{
    SculptParams,
    deformation::{TopologyManager, cleanup_mesh},
};

/// Punches a hole with the given shape projected from the camera.
///
/// The vertices of `hole_shape` are in the normalized device coordinate space.
/// In effect it is checked if a projected mesh vertex `v_p` is inside the hole shape like this.
///
/// ```text
/// v_p = projection * obj_to_camera_isometry * v
/// ```
///
/// If it is inside, the vertex is moved along it's negative normal direction until it is outside the hole shape
/// or removed by the cleanup process.
///
/// Make sure that `obj_to_camera_isometry` is an isometry, i.e. it only contains rotation and translation.
/// if you need to apply a scale, either manually apply it the mesh graph before calling this function
/// (and making sure that the `params` are still correct) or if it's a uniform scale you can apply it
/// to `projection` instead.
#[instrument(skip(mesh_graph, topology_manager))]
pub fn punch_hole(
    mesh_graph: &mut MeshGraph,
    hole_shape: Polygon2,
    obj_to_camera_isometry: Mat4,
    projection: Mat4,
    params: &SculptParams,
    topology_manager: &mut TopologyManager,
) {
    if hole_shape.vertices.is_empty() {
        warn!("Hole shape is empty => skipping punch_hole");
        return;
    }

    mesh_graph.apply_transform(obj_to_camera_isometry);
    mesh_graph.rebuild_bvh();

    #[cfg(feature = "rerun")]
    mesh_graph.log_rerun();

    let neg_max_move_dist = -params.max_move_dist_squared.sqrt();

    mesh_graph.collapse_until_edges_above_min_length(
        params.min_edge_length_squared,
        &mut topology_manager.protected_vertices,
    );

    mesh_graph.subdivide_until_edges_below_max_length(
        params.max_edge_length_squared,
        &mut topology_manager.protected_halfedges,
        &mut topology_manager.protected_vertices,
    );

    let proj_inv = projection.inverse();

    let aabb = Aabb::from_points(hole_shape.vertices.iter().flat_map(|v| {
        [
            proj_inv.project_point3(v.extend(-1.0)),
            proj_inv.project_point3(v.extend(1.0)),
        ]
    }));

    // Upper bound so the loop terminates even when moving along the negative normal never
    // changes the projected position (e.g. view direction tangent to the surface).
    const MAX_PUNCH_ITERATIONS: usize = 1000;

    let mut converged = false;

    for _ in 0..MAX_PUNCH_ITERATIONS {
        let mut vertex_ids = HashSet::new();

        for face_idx in mesh_graph.bvh.intersect_aabb(&aabb) {
            // The BVH can contain stale entries for faces removed during cleanup,
            // so look the face id up defensively instead of indexing.
            let Some(&face_id) = mesh_graph.index_to_face_id.get(&face_idx) else {
                error!("Face index not found");
                continue;
            };

            let Some(face) = mesh_graph.faces.get(face_id) else {
                error!("Face not found");
                continue;
            };

            for v_id in face.vertices(mesh_graph) {
                vertex_ids.insert(v_id);
            }
        }

        if vertex_ids.is_empty() {
            converged = true;
            break;
        }

        #[cfg(feature = "rerun")]
        mesh_graph.log_verts_rerun(
            "punch_hole/box",
            &vertex_ids.iter().copied().collect::<Vec<_>>(),
        );

        let mut changed = false;

        for v_id in vertex_ids {
            let Some(pos) = mesh_graph.positions.get(v_id) else {
                error!("Vertex position not found");
                continue;
            };

            if hole_shape.contains_point(projection.project_point3(*pos).xy()) {
                let Some(&normal) = mesh_graph.vertex_normals.as_ref().and_then(|n| n.get(v_id))
                else {
                    error!("Vertex normal not found");
                    continue;
                };

                #[cfg(feature = "rerun")]
                {
                    use mesh_graph::utils::vec3_array;

                    mesh_graph.log_vert_rerun("punch_hole/inside", v_id);
                    mesh_graph::RR
                        .log(
                            "punch_hole/normal",
                            &rerun::Arrows3D::from_vectors([vec3_array(normal)])
                                .with_origins([vec3_array(pos)]),
                        )
                        .unwrap();
                }
                let move_vec = normal * neg_max_move_dist;

                // already checked for existence above
                mesh_graph.positions[v_id] += move_vec;

                changed = true;
            }
        }

        if !changed {
            converged = true;
            break;
        }

        cleanup_mesh(mesh_graph, params, topology_manager, true);
        mesh_graph.compute_vertex_normals();

        #[cfg(feature = "rerun")]
        mesh_graph.log_rerun();
    }

    if !converged {
        error!("punch_hole did not converge after {MAX_PUNCH_ITERATIONS} iterations");
    }

    mesh_graph.apply_transform(obj_to_camera_isometry.inverse());
    mesh_graph.rebuild_bvh();
}
