use std::fmt::Debug;

use glam::Vec3;
use mesh_graph::MeshGraph;
use parry3d::query::PointQuery;
use tracing::{error, instrument};

use crate::{SculptParams, deformation::TopologyManager, deformation::cleanup_mesh};

/// Morphological open/close — global mesh operator.
///
/// Every vertex is moved along its normal. The sign of `amount` selects
/// the order of the two phases:
///
/// * `amount > 0` => **open**  = erode (`-normal * |amount|`) then dilate (`+normal * |amount|`).
///   Removes small protrusions and thin bridges.
/// * `amount < 0` => **close** = dilate (`+normal * |amount|`) then erode (`-normal * |amount|`).
///   Fills small gaps and fuses close sheets via the topology manager.
///
/// `amount` is the offset per phase. `0`, non-finite or near-zero
/// values are a no-op (warns). Topology merges run between and after the
/// two phases via [`cleanup_mesh`] with `allow_topology_change = true`,
/// so a close can fuse sheets separated by less than `params.max_thickness`.
///
/// To allow for non-uniform open/close amounts, you can provide a tuple of two values to `amount`.
/// The first value is used for the first phase, and the second value is used for the second phase.
/// For example, `(0.1, -0.2)` would dilate by `0.1` then erode by `0.2`.
/// If you want to effect a morphological **open**, the first value should be *negative* and the second value should be *positive*.
/// On the other hand, if you want to effect a morphological **close**, the first value should be *positive* and the second value should be *negative*.
///
/// `mask` is used to determine which points to affect. Everything that is inside the mask is affected,
/// and everything that is outside is left alone (apart from the mesh cleanup which is applied to the whole mesh).
/// You can provide anything that implements [`parry3d::query::PointQuery`] or `()` for no mask.
#[instrument(skip(mesh_graph, params, topology_manager, mask))]
pub fn morphological_open_close<M>(
    mesh_graph: &mut MeshGraph,
    params: &SculptParams,
    topology_manager: &mut TopologyManager,
    amount: impl Into<MorphOpenCloseAmount> + Debug,
    mask: impl ContainsPoint<M>,
) {
    let amount = amount.into();

    // Validate before touching the mesh: zero / non-finite amounts are a no-op.
    // This must run before pre-cleanup below, otherwise a no-op amount would
    // still collapse/subdivide the mesh and break the documented guarantee.
    if amount
        .get_amounts()
        .iter()
        .any(|signed| *signed == 0.0 || !signed.is_finite())
    {
        error!(
            "morphological_open_close: amount must be finite and non-zero, got {:?}. Aborting.",
            amount.get_amounts()
        );
        return;
    }

    // Cheap early-out: if mesh has no vertices, nothing to do.
    if mesh_graph.positions.is_empty() {
        return;
    }

    // Ensure normals exist — erode/dilate move along normals.
    if mesh_graph.vertex_normals.is_none() {
        mesh_graph.compute_vertex_normals();
    }

    // Pre-cleanup to enforce edge length invariants before moving.
    mesh_graph.collapse_until_edges_above_min_length(
        params.min_edge_length_squared,
        &mut topology_manager.protected_vertices,
    );
    mesh_graph.subdivide_until_edges_below_max_length(
        params.max_edge_length_squared,
        &mut topology_manager.protected_halfedges,
        &mut topology_manager.protected_vertices,
    );

    #[cfg(feature = "rerun")]
    mesh_graph.log_rerun();

    for signed in amount.get_amounts() {
        apply_phase(mesh_graph, params, topology_manager, signed, &mask);

        // Normals are invalidated by merges/subdivisions — recompute before
        // the next phase and for final output.
        mesh_graph.compute_vertex_normals();

        #[cfg(feature = "rerun")]
        mesh_graph.log_rerun();
    }

    mesh_graph.rebuild_bvh();
}

fn apply_phase<M>(
    mesh_graph: &mut MeshGraph,
    params: &SculptParams,
    topology_manager: &mut TopologyManager,
    signed_amount: f32,
    mask: &impl ContainsPoint<M>,
) {
    let mag = signed_amount.abs();
    if mag == 0.0 || !mag.is_finite() {
        return;
    }

    // Step count mirrors DeformationField::apply: ceil(sqrt(max_move² / max_move_dist²)).
    // Global operator: max movement == |amount| (weight 1.0 for all vertices).
    let steps = (mag * mag / params.max_move_dist_squared)
        .sqrt()
        .ceil()
        .clamp(1.0, 200.0) as usize;

    let factor = 1.0 / steps as f32;

    for _ in 0..steps {
        // The mesh folds as it is offset, so the normals from the previous step no
        // longer describe it: they drive the motion in the wrong direction and make
        // the merge gate in `TopologyPhysicsHooks` reject colliding sheets.
        mesh_graph.compute_vertex_normals();

        mesh_graph.optimize_bvh_incremental();

        // Move every vertex along its normal by signed_amount * factor.
        // Missing normal → skip (vertex created mid-cleanup before normals recomputed).
        if let Some(normals) = mesh_graph.vertex_normals.as_ref() {
            for (v_id, pos) in mesh_graph.positions.iter_mut() {
                if !mask.contains_point(*pos) {
                    continue;
                }

                if let Some(n) = normals.get(v_id) {
                    *pos += *n * signed_amount * factor;
                }
            }
        } else {
            // Should not happen — we compute normals before each phase — but handle gracefully.
            for (v_id, pos) in mesh_graph.positions.iter_mut() {
                if !mask.contains_point(*pos) {
                    continue;
                }

                if let Some(n) = mesh_graph.vertex_normals.as_ref().and_then(|m| m.get(v_id)) {
                    *pos += *n * signed_amount * factor;
                }
            }
        }

        // Sheets `2 * amount` apart now coincide; merging into that tears holes.
        zip_coincident_sheets(mesh_graph, topology_manager);

        cleanup_mesh(mesh_graph, params, topology_manager, true);
    }
}

/// [`MeshGraph::zip_coincident_patches`] over the whole mesh (the offset moved every
/// vertex), keeping the protected sets free of dead ids.
fn zip_coincident_sheets(mesh_graph: &mut MeshGraph, topology_manager: &mut TopologyManager) {
    let scope = mesh_graph.vertices.keys().collect::<Vec<_>>();
    let zip = mesh_graph.zip_coincident_patches(scope);

    for v_id in &zip.removed_vertices {
        topology_manager.protected_vertices.remove(v_id);
    }
    for he_id in &zip.removed_halfedges {
        topology_manager.protected_halfedges.remove(he_id);
    }
}

/// Amount of open/close deformation to apply to a mesh.
///
/// You don't need to use this enum directly. When calling [`morphological_open_close`], you can use a simple `f32` value for uniform open/close amounts, or a pair `(f32, f32)` for non-uniform amounts.
pub enum MorphOpenCloseAmount {
    /// Use the same amount for the two dilate/erode operations.
    Uniform(f32),
    /// Use a different amount for the two dilate/erode operations.
    /// Use the first value for the first operation and the second value for the second operation.
    /// Note that one should be positive and the other negative depending on if you want to do a morphological open or close operation.
    /// For an open operation, the first value should be negative and the second value should be positive.
    /// Consequently, for a close operation, the first value should be positive and the second value should be negative.
    NonUniform(f32, f32),
}

impl MorphOpenCloseAmount {
    pub fn get_amounts(&self) -> [f32; 2] {
        match self {
            Self::Uniform(amount) => [-*amount, *amount],
            Self::NonUniform(amount1, amount2) => [*amount1, *amount2],
        }
    }
}

impl From<f32> for MorphOpenCloseAmount {
    fn from(value: f32) -> Self {
        Self::Uniform(value)
    }
}

impl From<(f32, f32)> for MorphOpenCloseAmount {
    fn from(value: (f32, f32)) -> Self {
        Self::NonUniform(value.0, value.1)
    }
}

pub trait ContainsPoint<M> {
    fn contains_point(&self, point: Vec3) -> bool;
}

pub struct PointQueryMarker;

impl<PQ> ContainsPoint<PointQueryMarker> for PQ
where
    PQ: PointQuery,
{
    fn contains_point(&self, point: Vec3) -> bool {
        self.contains_local_point(point)
    }
}

pub struct UnitMarker;

impl ContainsPoint<UnitMarker> for () {
    fn contains_point(&self, _point: Vec3) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_graph::primitives::IcoSphere;

    /// Closed cube, `n` quads per side, wound outwards.
    fn tessellated_cube(origin: Vec3, size: f32, n: usize) -> (Vec<Vec3>, Vec<usize>) {
        let mut index: hashbrown::HashMap<[usize; 3], usize> = hashbrown::HashMap::new();
        let mut positions = Vec::new();
        let mut indices = Vec::new();
        let step = size / n as f32;

        for axis in 0..3 {
            let (u_axis, v_axis) = ((axis + 1) % 3, (axis + 2) % 3);
            for side in [0, n] {
                let mut outward = Vec3::ZERO;
                outward[axis] = if side == 0 { -1.0 } else { 1.0 };

                for i in 0..n {
                    for j in 0..n {
                        let quad = [(0, 0), (1, 0), (1, 1), (0, 1)].map(|(di, dj)| {
                            let mut cell = [0; 3];
                            cell[axis] = side;
                            cell[u_axis] = i + di;
                            cell[v_axis] = j + dj;
                            *index.entry(cell).or_insert_with(|| {
                                positions.push(origin + Vec3::from(cell.map(|c| c as f32)) * step);
                                positions.len() - 1
                            })
                        });
                        let [a, b, c] = [quad[0], quad[1], quad[2]].map(|i| positions[i]);
                        if (b - a).cross(c - a).dot(outward) > 0.0 {
                            indices.extend([quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]]);
                        } else {
                            indices.extend([quad[0], quad[2], quad[1], quad[0], quad[3], quad[2]]);
                        }
                    }
                }
            }
        }

        (positions, indices)
    }

    /// Two unit cubes `gap` apart along x.
    fn two_cubes(gap: f32, n: usize) -> MeshGraph {
        let (mut positions, mut indices) = tessellated_cube(Vec3::ZERO, 1.0, n);
        let (other_positions, other_indices) =
            tessellated_cube(Vec3::new(1.0 + gap, 0.0, 0.0), 1.0, n);
        let offset = positions.len();
        positions.extend(other_positions);
        indices.extend(other_indices.into_iter().map(|i| i + offset));

        let mut mesh = MeshGraph::indexed_triangles(&positions, &indices);
        mesh.compute_vertex_normals();
        mesh.rebuild_bvh();
        mesh
    }

    fn boundary_halfedges(mesh: &MeshGraph) -> usize {
        mesh.halfedges
            .values()
            .filter(|he| he.face.is_none())
            .count()
    }

    #[test]
    fn zip_coincident_sheets_joins_touching_cubes() {
        let n = 4;
        let mut mesh = two_cubes(0.0, n);
        let params = SculptParams::new(0.5);
        let mut tm = TopologyManager::new(&mesh, params);
        let (vertices, faces) = (mesh.vertices.len(), mesh.faces.len());

        // Protect the contact plane, where the zip removes.
        for (v_id, position) in &mesh.positions {
            if position.x == 1.0 {
                tm.protected_vertices.insert(v_id);
                for &he_id in &mesh.outgoing_halfedges[v_id] {
                    tm.protected_halfedges.insert(he_id);
                }
            }
        }

        zip_coincident_sheets(&mut mesh, &mut tm);

        // Both contact sides go; one side's vertices fuse, the other's inner ones die.
        assert_eq!(mesh.faces.len(), faces - 2 * 2 * n * n);
        assert_eq!(
            mesh.vertices.len(),
            vertices - (n + 1) * (n + 1) - (n - 1) * (n - 1)
        );
        assert_eq!(boundary_halfedges(&mesh), 0);
        assert!(
            tm.protected_vertices
                .iter()
                .all(|v| mesh.vertices.contains_key(*v))
        );
        assert!(
            tm.protected_halfedges
                .iter()
                .all(|he| mesh.halfedges.contains_key(*he))
        );
    }

    /// The gap=0.05 weld: closing by half the gap lands both sides on the midplane.
    #[test]
    fn close_joins_sheets_that_meet_on_the_midplane() {
        let mut mesh = two_cubes(0.05, 8);
        // `min_edge_length_squared` 0.0096, as in the weld.
        let params = SculptParams::new(0.2);
        let mut tm = TopologyManager::new(&mesh, params);

        morphological_open_close(&mut mesh, &params, &mut tm, -0.025, ());

        assert_eq!(boundary_halfedges(&mesh), 0);
        assert!(
            tm.protected_vertices
                .iter()
                .all(|v| mesh.vertices.contains_key(*v))
        );
    }

    fn setup_mesh() -> (MeshGraph, SculptParams, TopologyManager) {
        let mut mesh = MeshGraph::from(IcoSphere {
            subdivisions: 1,
            radius: 1.0,
        });
        mesh.compute_vertex_normals();
        mesh.rebuild_bvh();
        let params = SculptParams::new(0.5);
        let tm = TopologyManager::new(&mesh, params);
        (mesh, params, tm)
    }

    #[test]
    fn morphological_open_close_zero_and_non_finite_are_noop() {
        let (mut mesh, params, mut tm) = setup_mesh();
        let verts_before = mesh.vertices.len();
        let positions_before: Vec<_> = mesh.positions.values().copied().collect();

        morphological_open_close(&mut mesh, &params, &mut tm, 0.0, ());
        assert_eq!(mesh.vertices.len(), verts_before);

        morphological_open_close(&mut mesh, &params, &mut tm, f32::NAN, ());
        assert_eq!(mesh.vertices.len(), verts_before);

        morphological_open_close(&mut mesh, &params, &mut tm, f32::INFINITY, ());
        assert_eq!(mesh.vertices.len(), verts_before);

        // Positions unchanged for no-ops (up to floating point — should be exact no move)
        let positions_after: Vec<_> = mesh.positions.values().copied().collect();
        assert_eq!(positions_before, positions_after);
    }

    #[test]
    fn morphological_open_close_does_not_panic_on_small_mesh() {
        let (mut mesh, params, mut tm) = setup_mesh();
        // Small amount — exercises stepping with 1 step
        morphological_open_close(&mut mesh, &params, &mut tm, 0.01, ());
        assert!(!mesh.vertices.is_empty());
        assert!(mesh.vertex_normals.is_some());

        // Larger amount — exercises multi-step path
        morphological_open_close(&mut mesh, &params, &mut tm, -0.5, ());
        assert!(!mesh.vertices.is_empty());
    }

    #[test]
    fn morphological_open_and_close_are_not_noop_on_displacement() {
        // Check that open and close actually displace vertices (normals exist).
        // Use a slightly larger amount to ensure movement > edge thresholds isn't collapsed away entirely.
        let (mut mesh_open, params, mut tm_open) = setup_mesh();
        let pos_before_open: Vec<_> = mesh_open.positions.values().copied().collect();
        morphological_open_close(&mut mesh_open, &params, &mut tm_open, 0.2, ());
        let pos_after_open: Vec<_> = mesh_open.positions.values().copied().collect();
        assert_ne!(pos_before_open, pos_after_open);

        let (mut mesh_close, params2, mut tm_close) = setup_mesh();
        let pos_before_close: Vec<_> = mesh_close.positions.values().copied().collect();
        morphological_open_close(&mut mesh_close, &params2, &mut tm_close, -0.2, ());
        let pos_after_close: Vec<_> = mesh_close.positions.values().copied().collect();
        assert_ne!(pos_before_close, pos_after_close);
    }
}
