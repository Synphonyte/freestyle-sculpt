mod fields;
#[cfg(feature = "instrumentation")]
pub(crate) mod journal;
mod topology;
mod traits;

pub use fields::*;
#[cfg(feature = "instrumentation")]
pub use journal::*;
use mesh_graph::MeshGraph;
pub use topology::*;
pub use traits::*;

use crate::SculptParams;

/// Upper bound on cleanup iterations to guarantee termination even if collapse, subdivision
/// and collision merging keep producing new work for each other.
const MAX_CLEANUP_ITERATIONS: usize = 1000;

pub fn cleanup_mesh(
    mesh_graph: &mut MeshGraph,
    params: &SculptParams,
    topology_manager: &mut TopologyManager,
    allow_topology_change: bool,
) {
    let mut protected_vertices_count = topology_manager.protected_vertices.len();

    for _ in 0..MAX_CLEANUP_ITERATIONS {
        #[cfg(feature = "rerun")]
        mesh_graph.log_rerun();

        // TODO : Optimize: After a collision merge only the affected halfedges should be considered
        #[cfg(feature = "instrumentation")]
        {
            journal::record_step(
                journal::JournalOp::Collapse {
                    min_len_sqr: params.min_edge_length_squared,
                },
                &topology_manager.protected_vertices,
                &topology_manager.protected_halfedges,
            );
        }
        mesh_graph.collapse_until_edges_above_min_length(
            params.min_edge_length_squared,
            &mut topology_manager.protected_vertices,
        );

        #[cfg(feature = "instrumentation")]
        {
            journal::record_step(
                journal::JournalOp::Subdivide {
                    max_len_sqr: params.max_edge_length_squared,
                },
                &topology_manager.protected_vertices,
                &topology_manager.protected_halfedges,
            );
        }
        mesh_graph.subdivide_until_edges_below_max_length(
            params.max_edge_length_squared,
            &mut topology_manager.protected_halfedges,
            &mut topology_manager.protected_vertices,
        );

        if allow_topology_change {
            topology_manager.sync_mesh_graph(mesh_graph, params);

            if topology_manager
                .update_collisions_and_merge(mesh_graph, params)
                .is_none()
            {
                return;
            }
        } else {
            let new_count = topology_manager.protected_vertices.len();
            if new_count == protected_vertices_count {
                break;
            }
            protected_vertices_count = new_count;
        }
    }

    tracing::error!("cleanup_mesh did not converge after {MAX_CLEANUP_ITERATIONS} iterations");
}
