mod fields;
/// Debug-only operation journal (`instrumentation` feature); crate-internal, not
/// part of the public API.
#[cfg(feature = "instrumentation")]
pub(crate) mod journal;
mod topology;
mod traits;

pub use fields::*;
use mesh_graph::MeshGraph;
pub use topology::*;
pub use traits::*;

use crate::SculptParams;
use mesh_graph::EdgeLengthCleanup;

/// Upper bound on cleanup iterations to guarantee termination even if collapse, subdivision
/// and collision merging keep producing new work for each other.
const MAX_CLEANUP_ITERATIONS: usize = 1000;

/// Brings the mesh back inside the edge-length band, merging colliding sheets on the
/// way when `allow_topology_change` is set.
///
/// Returns whether the mesh ended up in band. A [`EdgeLengthCleanup::Stalled`] result
/// means edges remain out of band and further passes will not fix it on their own, so
/// a caller that loops on geometric change (rather than on mesh quality) can stop
/// instead of spinning.
pub fn cleanup_mesh(
    mesh_graph: &mut MeshGraph,
    params: &SculptParams,
    topology_manager: &mut TopologyManager,
    allow_topology_change: bool,
) -> EdgeLengthCleanup {
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
        let collapse_outcome = mesh_graph.collapse_until_edges_above_min_length(
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
        let subdivide_outcome = mesh_graph.subdivide_until_edges_below_max_length(
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
                // No more sheets to merge. Report whether the last pass also left the
                // edge lengths in band, which is a separate question from merging.
                return combine(collapse_outcome, subdivide_outcome);
            }
        } else {
            if collapse_outcome.converged() && subdivide_outcome.converged() {
                // Every edge is within the length band, so there is nothing left to
                // iterate on. This used to be inferred from `protected_vertices` not
                // growing, which could not tell a clean mesh from a marked set that
                // happened to be the same size twice — and which fell through to the
                // error below on the way out, reporting a failure on every success.
                return EdgeLengthCleanup::Converged;
            }

            // Edges remain out of band. Keep going only while the marked set is still
            // growing: once it stops, the ops have stopped being able to make progress
            // and further iterations cannot change that.
            let new_count = topology_manager.protected_vertices.len();
            if new_count == protected_vertices_count {
                return EdgeLengthCleanup::Stalled;
            }
            protected_vertices_count = new_count;
        }
    }

    tracing::error!("cleanup_mesh did not converge after {MAX_CLEANUP_ITERATIONS} iterations");

    EdgeLengthCleanup::Stalled
}

/// Converged only if both halves of the cleanup converged.
fn combine(collapse: EdgeLengthCleanup, subdivide: EdgeLengthCleanup) -> EdgeLengthCleanup {
    if collapse.converged() && subdivide.converged() {
        EdgeLengthCleanup::Converged
    } else {
        EdgeLengthCleanup::Stalled
    }
}
