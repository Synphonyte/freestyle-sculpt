# Freestyle Sculpt

[![Crates.io](https://img.shields.io/crates/v/freestyle-sculpt.svg)](https://crates.io/crates/freestyle-sculpt)
[![Docs](https://docs.rs/freestyle-sculpt/badge.svg)](https://docs.rs/freestyle-sculpt/)
[![MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/synphonyte/freestyle-sculpt#license)
[![Build Status](https://github.com/synphonyte/freestyle-sculpt/actions/workflows/cd.yml/badge.svg)](https://github.com/synphonyte/freestyle-sculpt/actions/workflows/cd.yml)

<!-- cargo-rdme start -->

This is a pure Rust implementation of Freestyle Sculpting, a real-time dynamic topology sculpting algorithm.

It is based on the paper [Freestyle: Sculpting meshes with self-adaptive topology](https://inria.hal.science/inria-00606516/document) by Lucian Stanculescu, Raphaëlle Chaine, Marie-Paule Cani. This is the same algorithm that is used by the Dyntopo sculpting mode in Blender.

![Freestyle Sculpt Demo](https://raw.githubusercontent.com/Synphonyte/freestyle-sculpt/refs/heads/main/docs/freestyle-demo.webp)

Please check out the [bevy-basic-sculpt example](https://github.com/Synphonyte/freestyle-sculpt/tree/main/examples/bevy-basic-sculpt) to see how it can be used in an interactive application.

### Optional Cargo features

- `rerun`: Enables recording of the mesh graph and the different algorithms to [Rerun](https://rerun.io/) for visualization.
- `bevy`: Enables integration with the [Bevy](https://bevyengine.org/) game engine.

### Customize sculpting

To implement a custom deformation field, you can create a struct that implements the [`DeformationField`] trait. Have a look
at the existing deformation fields in the [`deformation`] module for inspiration.

If you want to implement a custom selection strategy, you can create a struct that implements the [`MeshSelector`] trait. Have a look
at the existing selection strategies in the [`selectors`] module for inspiration.

<!-- cargo-rdme end -->
