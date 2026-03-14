mod fields;
mod topology;
mod traits;

pub use fields::*;
use mesh_graph::MeshGraph;
pub use topology::*;
pub use traits::*;

use crate::SculptParams;

pub fn cleanup_mesh(
    mesh_graph: &mut MeshGraph,
    params: &SculptParams,
    topology_manager: &mut TopologyManager,
) {
    loop {
        #[cfg(feature = "rerun")]
        mesh_graph.log_rerun();

        // TODO : Optimize: After a collision merge only the affected halfedges should be considered
        mesh_graph.collapse_until_edges_above_min_length(
            params.min_edge_length_squared,
            &mut topology_manager.protected_vertices,
        );

        mesh_graph.subdivide_until_edges_below_max_length(
            params.max_edge_length_squared,
            &mut topology_manager.protected_halfedges,
            &mut topology_manager.protected_vertices,
        );

        topology_manager.sync_mesh_graph(mesh_graph, params);

        if topology_manager
            .update_collisions_and_merge(mesh_graph, params)
            .is_none()
        {
            break;
        }
    }
}
