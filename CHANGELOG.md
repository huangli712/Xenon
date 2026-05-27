# Changelog

All notable changes to this project are documented in this file.

## [v0.0.10] — 2026-05-27

### Changed

- Replaced fully-qualified paths with direct imports throughout dimension module.
- Made axes/broadcast/dynamic/fixed/into submodules private; exposed via re-exports.
- Extracted Dimension/Reverse/RemoveAxis traits into dimension/types.rs.
- Moved MAX_DIMENSION and tests from mod.rs into types.rs.
- Reordered submodule declarations and re-exports in dimension/mod.rs.
- Colocated Reverse/RemoveAxis/Index/From impls with respective IxN structs.
- Moved into_dyn/try_from_dyn methods next to each IxN struct definition.
- Added RemoveAxis impls for Ix4, Ix5, Ix6.
- Added `///` doc comments to all Reverse/RemoveAxis impls.

### Removed

- Stale design-doc cross-references (5.x, 8.x) from dimension doc comments.
- Unused test helper `assert_dimension_bounds` in types.rs.

### Fixed

- Missing IntoDimension import in types.rs test module.

### Added

- Reverse and RemoveAxis tests for Ix4-Ix6 (9 new tests, 688→697).


## [v0.0.9] — 2026-05-27

### Changed

- Replaced fully-qualified paths with direct imports throughout error.rs, display.rs, ops.rs, and types.rs.
- Extracted `ComplexFloat` trait and `Complex` struct into a dedicated `types.rs` sub-module.
- Moved `ComplexFloat`/`Complex` unit tests from `mod.rs` into `types.rs`.
- Added `///` doc comments to all test functions across the complex module.
- Removed stale design-doc and task-tracking cross-references from doc comments.

## [v0.0.8] — 2026-05-26

### Removed

- `should_parallelize` function and its dedicated tests.

### Changed

- Refactored dispatch module into top-down dependency order.
- Removed unused `should_parallelize` function.
- Moved `dispatch_invalid_argument` from error.rs to dispatch.rs.
- Replaced fully-qualified paths with direct imports throughout dispatch.rs.
- Consolidated test imports: `use super::*` and direct imports throughout dispatch tests.
- Cleaned up stale design-doc cross-references and section labels in dispatch.rs.

## [v0.0.7] — 2026-05-26

### Removed

- Redundant CI workflows (`test.yml`, `docs.yml`).
- Obsolete baseline pin drift check from CI.

### Changed

- Consolidated CI into single `ci.yml` (merged `test.yml`, `docs.yml`).
- Pinned rust-toolchain to 1.95 across all workflows.
- Renamed `simd_comparison` benchmark to `simd`.
- Renamed `parallel_comparison` benchmark to `parallel`.
- Moved bench utility module to `benches/common/`.
- Replaced Python regression reporter with Rust binary (`benches/tool/bench-report`).
- Extended clippy config with `disallowed-methods` (transmute).
- Added type-complexity and too-many-arguments thresholds to clippy config.

### Fixed

- `.gitignore` patterns for `.sisyphus`, `.omo`, `__pycache__` now dir-specific.

## [v0.0.6] — 2026-05-25

### Added

- `Tensor::into_shared()` for zero-copy `ArcTensor` conversion.
- `Workspace::borrow()`: public error-documented borrow API.
- `Workspace::borrow_mut()`: public error-documented borrow API.
- Error and panic documentation sections across all public APIs.

### Removed

- `ParallelPool` struct.
- `par_iter` and `parallel/iter.rs`.
- `apply_binary` and `apply_compare` wrapper functions.
- `StorageSharedExt` trait with `ref_count` and `is_unique` methods.
- `broadcast_with` and `BroadcastPair` types.
- `SimdKernel` trait.
- f32 ULP helpers and `RealScalarBits::ulp` method.
- Tier 1 assertion helpers.
- `with_strategy` method and dead imports.

### Changed

- `ParallelGuard` now derives `Debug`; `TensorBase` derives `Debug`.
- Gated parallel threshold functions behind feature flags.
- Simplified `SimdElement` trait interface.

### Fixed

- Synchronized test version assertion and cleaned up inactive work-in-progress test placeholders.
