//! This is a pure Rust implementation of Freestyle Sculpting, a real-time dynamic topology sculpting algorithm.
//!
//! It is based on the paper [Freestyle: Sculpting meshes with self-adaptive topology](https://inria.hal.science/inria-00606516/document) by Lucian Stanculescu, Raphaëlle Chaine, Marie-Paule Cani. This is the same algorithm that is used by the Dyntopo sculpting mode in Blender.
//!
//! ![Freestyle Sculpt Demo](https://raw.githubusercontent.com/Synphonyte/freestyle-sculpt/refs/heads/main/docs/freestyle-demo.webp)
//!
//! Please check out the [bevy-basic-sculpt example](https://github.com/Synphonyte/freestyle-sculpt/tree/main/examples/bevy-basic-sculpt) to see how it can be used in an interactive application.
//!
//! ## Limitations
//!
//! At the moment it doesn't support topology genus changes, i.e. no splitting or merging of different parts of the mesh.
//!
//! ## Optional Cargo features
//!
//! - `rerun`: Enables recording of the mesh graph and the different algorithms to [Rerun](https://rerun.io/) for visualization.
//! - `bevy`: Enables integration with the [Bevy](https://bevyengine.org/) game engine.
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
pub struct SculptParams {
    pub max_move_dist_squared: f32,
    pub min_edge_length_squared: f32,
    pub max_edge_length_squared: f32,
}

impl SculptParams {
    /// Creates a new instance of `SculptParams` with the specified maximum edge length.
    ///
    /// All other parameters are calculated based on the maximum edge length.
    pub const fn new(max_edge_length: f32) -> Self {
        let max_edge_length_squared = max_edge_length * max_edge_length;

        Self {
            max_move_dist_squared: max_edge_length_squared * 0.11,
            min_edge_length_squared: max_edge_length_squared * 0.24,
            max_edge_length_squared,
        }
    }

    pub fn from_mesh_graph(mesh_graph: &MeshGraph, min_edge_length: f32) -> Self {
        let edge_length = median(
            &mut mesh_graph
                .halfedges
                .values()
                .map(|he| he.length(mesh_graph))
                .collect_vec(),
        );

        Self::new((edge_length * 1.5).max(min_edge_length))
    }
}
