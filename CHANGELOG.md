# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2026-03-14

- Refactored the API
- Added `punch_hole`
- Implemented topology genus change, i.e. you can now split and join surfaces
- Updated dependencies

## [0.5.0] - 2025-12-22

- `SculptParams::from_mesh_graph()` now takes a parameter `min_edge_length`.
- Updated dependencies `mesh-graph` and `rerun`.

## [0.4.1] - 2025-10-08

- Fixed BVH

## [0.4.0] (yanked) - 2025-10-07

- Hardened against panics
- Updated mesh-graph to version 0.3 and other dependencies to their latest versions

## [0.3.0] - 2025-06-01

- Made selector metric configurable
- Smooth deformation is 90% less sensitive

## [0.2.0] - 2025-05-29

- Made selector fields public
- Updated parry3d to version 0.21
- Updated mesh-graph to version 0.2
- Added `from_mesh_graph` constructor to `SculptParams`
- Added `serde` support
- Fixed selection handling while sculpting
- Fixed mouse handling for the translate deformation

## [0.1.1] - 2025-05-15

- Made `SculptParams::new` const

## [0.1.0] - 2025-05-15

- Sculpting without topology genus changes works
- Implemented deformations fields:
  - Smoothing
  - Translation
- Implemented two selectors
  - Sphere with falloff
  - Surface sphere with falloff
- Bevy example
- Rerun debug
