# Changelog

All notable changes to this project are documented in this file.


## [v0.0.7] — 2026-05-26

### Changed

- Consolidated CI into single `ci.yml` (merged `test.yml`, `docs.yml`).
- Pinned rust-toolchain to 1.95 across all workflows.
- Restructured bench infrastructure: benches renamed (`simd_comparison` → `simd`,
  `parallel_comparison` → `parallel`), utility module moved to `benches/common/`.
- Replaced Python regression reporter (`tools/bench/report.py`) with Rust binary
  (`benches/tool/bench-report`).
- Extended clippy config with `disallowed-methods` (transmute), type-complexity and
  too-many-arguments thresholds.

### Removed

- Redundant CI workflows (`test.yml`, `docs.yml`).
- Obsolete baseline pin drift check from CI.

### Fixed

- `.gitignore` patterns for `.sisyphus`, `.omo`, `__pycache__` now dir-specific.

## [v0.0.6] — 2026-05-25

### Added

- `Tensor::into_shared()` for zero-copy `ArcTensor` conversion.
- `Workspace::borrow()` and `Workspace::borrow_mut()`: public error-documented borrow APIs.
- Error and panic documentation sections across all public APIs.

### Changed

- `ParallelGuard` now derives `Debug`; `TensorBase` derives `Debug`.
- Gated parallel threshold functions behind feature flags.
- Simplified `SimdElement` trait interface.

### Fixed

- Synchronized test version assertion and cleaned up inactive work-in-progress test placeholders.

### Removed

- `ParallelPool`, `par_iter`, `apply_binary`, `apply_compare`, and `parallel/iter.rs`.
- `StorageSharedExt`, `ref_count`, `is_unique` methods.
- `broadcast_with` / `BroadcastPair` types.
- `SimdKernel` type.
- f32 ULP helpers and `RealScalarBits::ulp` method.
- Tier 1 assertion helpers.
- `with_strategy` and dead imports.
