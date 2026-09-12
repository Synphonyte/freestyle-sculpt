//! This is a pure Rust implementation of Freestyle Sculpting, a real-time dynamic topology sculpting algorithm.
//!
//! It is based on the paper [Freestyle: Sculpting meshes with self-adaptive topology](https://inria.hal.science/inria-00606516/document) by Lucian Stanculescu, Raphaëlle Chaine, Marie-Paule Cani. This is the same algorithm that is used by the Dyntopo sculpting mode in Blender.
//!
//! ![Freestyle Sculpt Demo](https://raw.githubusercontent.com/Synphonyte/freestyle-sculpt/refs/heads/main/docs/freestyle-demo.webp)
//!
//! Please check out the [bevy-basic-sculpt example](https://github.com/Synphonyte/freestyle-sculpt/tree/main/examples/bevy-basic-sculpt) to see how it can be used in an interactive application.
//!
//! ## Optional Cargo features
//!
//! - `rerun`: Enables recording of the mesh graph and the different algorithms to [Rerun](https://rerun.io/) for visualization.
//! - `bevy`: Enables integration with the [Bevy](https://bevyengine.org/) game engine.
//! - `serde`: Derives [`serde`](https://serde.rs/) `Serialize`/`Deserialize` for [`SculptParams`] and the mesh graph.
//! - `gltf`: Enables mesh-graph's glTF loading (used by the log replay tests).
//! - `instrumentation`: Debug-only. Records every topology operation into an
//!   internal journal and enables mesh-graph's per-operation integrity validators
//!   so a mesh corruption can be dumped and replayed. It is very slow and is not
//!   part of the public API — never enable it for a release build.
//!
//! ## Customize sculpting
//!
//! To implement a custom deformation field, you can create a struct that implements the [`DeformationField`] trait. Have a look
//! at the existing deformation fields in the [`deformation`] module for inspiration.
//!
//! If you want to implement a custom selection strategy, you can create a struct that implements the [`MeshSelector`] trait. Have a look
//! at the existing selection strategies in the [`selectors`] module for inspiration.

use itertools::Itertools;
use mesh_graph::MeshGraph;
use parry3d::utils::median;
#[cfg(feature = "serde")]
use serde::Deserialize;

///Deformation fields to do the vertex manipulation
pub mod deformation;
mod integrations;
/// Ray casting onto mesh graphs
pub mod ray;
/// Selection strategies to decide which vertices to deform
pub mod selectors;

/// Defines all the necessary parameters for sculpting operations.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Resource))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(from = "SculptParamsSerde"))]
pub struct SculptParams {
    /// In the Freestyle paper referred to as `d_move`
    #[cfg_attr(feature = "serde", serde(skip))]
    pub max_move_dist_squared: f32,

    /// In the Freestyle paper referred to as `d`
    #[cfg_attr(feature = "serde", serde(skip))]
    pub min_edge_length_squared: f32,

    /// In the Freestyle paper referred to as `d_detail`
    pub max_edge_length_squared: f32,

    /// In the Freestyle paper referred to as `d_thickness`.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub max_thickness_half: f32,
    /// In the Freestyle paper referred to as `d_thickness`.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub max_thickness_squared: f32,
}

#[cfg(feature = "serde")]
#[derive(Deserialize)]
struct SculptParamsSerde {
    pub max_edge_length_squared: f32,
}
#[cfg(feature = "serde")]
impl From<SculptParamsSerde> for SculptParams {
    fn from(value: SculptParamsSerde) -> Self {
        Self::from_max_edge_length_squared(value.max_edge_length_squared)
    }
}

impl SculptParams {
    /// Factor applied to a mesh's median edge length by [`Self::from_mesh_graph`]
    /// to obtain `max_edge_length`.
    ///
    /// `max_edge_length` is the *upper* bound of the target band
    /// `[sqrt(0.24) * max, max]`, so feeding the median in directly would put half
    /// of all edges above the threshold and split every one of them. Subdivision
    /// also introduces shorter interior edges, which lands the post-cleanup median
    /// at roughly `0.62 * max_edge_length` rather than at the band's centre.
    /// Scaling by the inverse of that keeps the cleanup resolution-neutral:
    /// measured across six production scans it holds the triangle count within
    /// 0.93-1.15x and the median edge length within 0.99-1.06x of the input,
    /// where an unscaled median tripled the triangle count.
    pub const RESOLUTION_NEUTRAL_EDGE_LENGTH_FACTOR: f32 = 1.6;

    /// Creates a new instance of `SculptParams` with the specified maximum edge length.
    ///
    /// All other parameters are calculated based on the maximum edge length.
    pub fn new(max_edge_length: f32) -> Self {
        debug_assert!(
            max_edge_length > 0.0 && max_edge_length.is_finite(),
            "max_edge_length must be a finite, positive value, got {max_edge_length}"
        );

        let max_edge_length_squared = max_edge_length * max_edge_length;
        debug_assert!(
            max_edge_length_squared > 0.0 && max_edge_length_squared.is_finite(),
            "max_edge_length squared must be a finite, positive value, got {max_edge_length}"
        );

        Self::from_max_edge_length_squared(max_edge_length_squared)
    }

    fn from_max_edge_length_squared(max_edge_length_squared: f32) -> Self {
        debug_assert!(
            max_edge_length_squared > 0.0 && max_edge_length_squared.is_finite(),
            "max_edge_length_squared must be a finite, positive value, got {max_edge_length_squared}"
        );

        let max_move_dist_squared = max_edge_length_squared * 0.11;
        let max_thickness_squared = 4.0 * max_move_dist_squared + max_edge_length_squared * 0.35;

        Self {
            max_move_dist_squared,
            min_edge_length_squared: max_edge_length_squared * 0.24,
            max_edge_length_squared,
            max_thickness_squared,
            max_thickness_half: max_thickness_squared.sqrt() * 0.5,
        }
    }

    /// Derives the parameters from an existing mesh so that sculpting keeps the
    /// mesh at roughly the resolution it already has.
    ///
    /// Uses [`Self::RESOLUTION_NEUTRAL_EDGE_LENGTH_FACTOR`]; see
    /// [`Self::from_mesh_graph_with_factor`] to pick the detail level explicitly.
    pub fn from_mesh_graph(mesh_graph: &MeshGraph, min_edge_length: f32) -> Self {
        Self::from_mesh_graph_with_factor(
            mesh_graph,
            min_edge_length,
            Self::RESOLUTION_NEUTRAL_EDGE_LENGTH_FACTOR,
        )
    }

    /// Derives the parameters from an existing mesh, scaling the mesh's median
    /// edge length by `edge_length_factor` to obtain `max_edge_length`.
    ///
    /// A larger factor means longer target edges, i.e. a coarser and cheaper
    /// mesh; a smaller factor refines. Pass
    /// [`Self::RESOLUTION_NEUTRAL_EDGE_LENGTH_FACTOR`] to keep the current
    /// resolution.
    ///
    /// `min_edge_length` is a floor on the resulting target length, applied
    /// after scaling.
    pub fn from_mesh_graph_with_factor(
        mesh_graph: &MeshGraph,
        min_edge_length: f32,
        edge_length_factor: f32,
    ) -> Self {
        debug_assert!(
            edge_length_factor > 0.0 && edge_length_factor.is_finite(),
            "edge_length_factor must be a finite, positive value, got {edge_length_factor}"
        );

        let mut edge_lengths = mesh_graph
            .halfedges
            .values()
            .map(|he| he.length(mesh_graph))
            // parry's `median` panics on NaN values, so filter out non-finite lengths
            .filter(|l| l.is_finite())
            .collect_vec();

        // `median` panics on an empty slice (e.g. a mesh without halfedges)
        if edge_lengths.is_empty() {
            return Self::new(min_edge_length);
        }

        Self::new((median(&mut edge_lengths) * edge_length_factor).max(min_edge_length))
    }
}

#[cfg(test)]
mod tests;
