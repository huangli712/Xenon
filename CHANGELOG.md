# Changelog

All notable changes to this project are documented in this file.

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
