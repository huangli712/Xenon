# Changelog

All notable changes to this project are documented in this file.

## [v0.0.26] — 2026-06-04

### Removed

- `src/ffi/blas.rs`, `src/ffi/offset.rs`, `src/ffi/private.rs` — content merged into `impls.rs` and `types.rs`.
- `TensorExport<'a, A>` / `TensorExportMut<'a, A>` intermediate generic descriptor types and their `From` impls.
- Module-level compile probe test from `ffi/mod.rs`.
- Stale `compile_fail` doctest (`into_raw_parts` on view) from `impls.rs`.
- Stale design-doc cross-references and task-tracking markers from all ffi module doc comments.
- Unicode `─` section separators replaced with ASCII `--` across ffi module.

### Changed

- Refactored FFI module from 5 files (`blas.rs`, `offset.rs`, `private.rs`, `ptr.rs`, `types.rs`) to 3 files (`impls.rs`, `types.rs`, `mod.rs`):
  - Merged `offset.rs` (index→offset helpers, BLAS layout queries) into `ptr.rs`.
  - Merged `blas.rs` (BLAS methods) into `offset.rs`, then both into `ptr.rs`.
  - Renamed `ptr.rs` → `impls.rs`.
  - Merged `private.rs` (generic `TensorExport` types, `From` impls, tests) into `types.rs`.
- Inlined `export_internal()`/`export_mut_internal()` — `export()`/`export_mut()` now build raw descriptors (`TensorExportRaw`/`TensorExportMutRaw`) directly.
- Re-export `TensorBase`/`OwnedRawParts` directly from `crate::tensor` (not transitively via `impls.rs`).
- Re-export `FfiBackend`/`FfiErrorCategory` directly from `crate::error` (not via `types.rs`).
- Reordered module declarations and re-exports alphabetically in `mod.rs`.
- Standardized `.expect()` placement (same line as closing `}`).
- Removed inline argument comments from unsafe `from_raw_parts` calls.
- Added blank-line separators between struct fields in `TensorExportRaw`/`TensorExportMutRaw`/`BlasInfo`.
- Moved inline `use` imports in test functions to module-level `use super::*`.
- Converted markdown error tables to bullet lists in doc comments.
- Formatted operator placement, multi-line `assert!`/`assert_eq!` macros, and section separators.
- Consolidated duplicate `use crate::error::{...}` imports.
- Removed trailing whitespace.

## [v0.0.25] — 2026-06-04

### Added

- SIMD `Complex<f32>` multiplication via `deinterleave_shfl` (SoA multiply, re-interleave).
- SIMD `Complex<f32>` and `Complex<f64>` conjugate dot product via `deinterleave_shfl` (SoA multiply-accumulate, horizontal reduction).
- Property-based tests for ComplexMulF32 tail/leftover-register coverage.
- Property-based tests for ComplexDotF32/ComplexDotF64 tail/leftover-register coverage.
- Complex admission tests (sub f32, add f64, neg f32/f64) in `binary.rs` and `unary.rs`.
- `use std::sync::OnceLock`, `use std::mem::size_of`, `use std::any::TypeId` imports across SIMD modules.
- `///` doc comments to all kernel structs (`DotF32Kernel`, `SumF32Kernel`, etc.) and dispatch helper functions.

### Changed

- Replaced `std::any::TypeId::of::<T>()` with imported `TypeId::of::<T>()`; `core::mem::size_of::<T>()` with `size_of::<T>()`; `std::sync::OnceLock` with `OnceLock`.
- Replaced `crate::simd::`/`simd::` path prefix with direct imported names in all SIMD test modules.
- Simplified `ComplexSumF32Kernel`/`ComplexSumF64Kernel` deinterleave: byte-level extraction replaced with direct `&[f32]`/`&[f64]` slice cast.
- Normalized `ELEMENTWISE_F32_F64_THRESHOLD` → `ELEMENTWISE_THRESHOLD`; removed duplicate test-local threshold constants.
- Reordered all kernel modules to `thresholds → kernel structs → dispatch helpers → tests`.
- Standardized section separator comments to fixed-width 80-column format across all SIMD modules.
- Unified `unsafe { slice::from_raw_parts }` block formatting and `// SAFETY:` comment placement.
- Consolidated `let handled = dispatch(...); if handled {` into `if dispatch(...) {`.
- Replaced `std::slice::` with `use std::slice;` throughout SIMD modules.
- Reordered imports (stdlib before crate, public before test).
- Added blank-line separators between struct field doc comments and constant declarations.
- Reordered `mod.rs` declarations (types → binary → unary → dot → sum → driver) and re-exports.

## [v0.0.24] — 2026-06-03

### Added

- Module-level doc comments to all dispatch submodules.
- Field doc comments to `ThresholdTestGuard` fields.
- Unit tests: `ThresholdTestGuard` construction, default, and threshold restoration.
- `Default` impl for `ThresholdTestGuard`.

### Removed

- `serial_test` dev-dependency (replaced by `ThresholdTestGuard` mutex).
- `#[serial]` annotations on all SIMD and parallel integration tests.
- Redundant `reset_*_threshold()` cleanup calls.

### Changed

- Split `dispatch.rs` into `dispatch/` directory module: `mod.rs`, `threshold.rs`, `types.rs`, `exec.rs`, `support.rs`.
- Replaced `#[allow(dead_code)]` with proper `#[cfg]` feature gates.
- Made `ParallelGuard` unconditionally re-exported (both feature-gated and placeholder variants).
- Consolidated imports and standardized section separator style across all dispatch submodules.
- Replaced fully-qualified paths with imported types in `exec.rs` and `types.rs`.
- Added blank line separators between `#[cfg]`-gated re-exports.

## [v0.0.23] — 2026-06-03

### Removed

- `serial_test` dev-dependency (replaced by process-global `ThresholdTestGuard`).
- `#[serial]` annotations on all SIMD and parallel integration tests.
- Redundant `reset_*_threshold()` calls (handled by `ThresholdTestGuard` drop).

### Changed

- Promoted `ThresholdTestGuard` from `pub(crate) mod test_support` to `pub` struct in `dispatch.rs`.
- Gated `ThresholdTestGuard` fields per-feature so it compiles under any feature combination.
- Exported `ThresholdTestGuard` through `crate::prelude` for external integration tests.
- Added `Default` impl for `ThresholdTestGuard`.

## [v0.0.22] — 2026-06-03

### Added

- `serial_test` dev-dependency for thread-safe threshold-mutating tests.
- Shared `ThresholdTestGuard` in `dispatch.rs` for parallel/SIMD threshold isolation.
- Integration test: SIMD-vs-serial consistency for `Complex<f64>` (`add`, `sum`, `dot`) at N=2048.
- `#[serial]` to all simd and parallel integration tests to eliminate threshold races.

### Changed

- Extracted `AddF32Kernel` from `simd/vector.rs` into `simd/binary.rs`.
- Replaced redundant `set_simd_threshold(1024)` calls with `#[serial]` ordering.
- All parallel test functions now acquire `ThresholdTestGuard` for sandboxed thresholds.
- Updated complex path test to use `Tensor1` API and proper reduction tolerances.

## [v0.0.21] — 2026-06-03

### Added

- Test: `AxisIterMut`/`IndexedIterMut`/`IterMut`/`StrideState` tests (write, empty, 3D, Ix0, size_hint).
- Field and method doc comments to all iter submodules (`Iter`, `IterMut`, `StrideState`, `AxisIter`, `IndexedIter`).
- Module-level docs to axis, impls, and indexed modules.

### Removed

- Section comment markers from all iter submodules.
- Stale design-doc cross-references from iter module.
- Redundant test imports and deduplicated test module structure.

### Changed

- Extracted `TensorBase` entry methods into `iter/impls.rs`.
- Extracted `StrideState` into `iter/types.rs`; renamed `iter/elements.rs` to `iter/primitives.rs`.
- Reordered module declarations and re-exports by dependency in `iter/mod.rs`.
- Replaced `crate::iter::` with `super::` sibling imports throughout iter submodules.
- Replaced fully-qualified paths with local imports in iter test code.
- Standardized unsafe block wrapping, doc comment line lengths, and import ordering.

## [v0.0.20] — 2026-06-02

### Added

- Test: overflow test for `checked_offset` in `ndindex.rs`.
- Test: `from_array` tests and `Range start==end` test for `SliceInfo`.
- Test: `get`/`get_mut` success and slice error-path tests in `impls.rs`.
- Inline step comments to `get()`, `get_mut()`, and `slice()` in `impls.rs`.
- Field doc comments to `SliceInfoIndices` and `SliceInfo` in `slice.rs`.

### Removed

- Empty compile anchor test from `index/mod.rs`.
- Isolated `mut_tests` module (merged into parent `tests`).
- Stale task-tracking comments and design-doc cross-references from index module.

### Changed

- Renamed `index/access.rs` to `index/impls.rs` for consistency.
- Moved `TensorBase::slice` from `slice.rs` to `impls.rs`.
- Consolidated `NdIndex` trait and `Sealed` impl ordering in `ndindex.rs`.
- Rewrote `index/mod.rs` docs with submodule table.
- Standardized section separator comments to `---` style across index module.
- Added blank lines between methods and enum variants.
- Replaced fully-qualified paths with local imports.
- Reformatted `expect`/`expect_err` calls and long function signatures.

## [v0.0.19] — 2026-06-02

### Added

- Test: overflow, zero-dim, and element index reporting for constructors and cast.
- Sealed supertrait to `CastElement`.
- Module-level doc comments to convert submodules.

### Removed

- `CastTo` intermediate trait and 44 forwarding `ConvertTo` impls (inlined into `CastTo`).
- Empty compile anchor tests from `construct` and `convert` modules.
- Stale Chinese comments and design-doc cross-references.

### Changed

- Refactored convert module: `types.rs` (`CastElement`), `cast.rs` (trait+impls), `impls.rs` (methods).
- Renamed `from.rs` to `impls.rs` in construct module, extracted `types.rs` for `CastElement`.
- Consolidated construct submodules (eye, init, scalar) into single `impls.rs`.
- Standardized section separator comments across `tensor/convert`.
- Replaced fully-qualified `crate::dimension::Ix1` with imported `Ix1` in tests.
- Unified imports ordering and formatting across construct and convert modules.

## [v0.0.18] — 2026-06-01

### Added

- Test: overflow and zero-dim cases for `zeros`, `ones`, `from_shape_vec`.
- Module-level doc comments to construct `impls.rs` and `types.rs`.

### Removed

- Empty compile anchor test from `construct/mod.rs`.
- Stale design-doc cross-references and task-tracking markers from construct module.

### Changed

- Consolidated construct submodules (eye, init, scalar) into a single `impls.rs`.
- Extracted `EyeElement` trait definitions into `construct/types.rs`.
- Restructured `construct/mod.rs` doc comment to table format.
- Restricted then restored `EyeElement` to pub (pub(crate) reverted).
- Replaced fully-qualified paths with direct imports; reformatted unsafe/expect calls.
- Reordered impl blocks for top-down logical reading.

## [v0.0.17] — 2026-06-01

### Added

- Doc comments and SAFETY comments to all unsafe blocks in tensor impls.
- Inline step comments for validation functions in `from_raw_parts_owned`.
- Section separator comments grouping related tests in `impls.rs`.
- Test: `from_raw_parts_owned` rejects capacity-below-len and invalid alignment.
- Test: `from_raw_parts` and `from_raw_parts_mut` valid-construction.

### Removed

- Stale intra-doc links and section separator comments (`// ── X ──`).
- Skeleton compile tests superseded by functional tests.

### Changed

- Consolidated multiple `impl` blocks into one per storage type (Owned, ViewRepr, ViewMutRepr).
- Unified section separator comments to `// ----------` style across tensor module.
- Reordered methods for logical grouping in query and Owned impl blocks.
- Replaced fully-qualified paths with local imports; reorganized `use` groups.
- Simplified tensor module doc comment to table format.
- Reformatted long function calls and unsafe blocks for consistency.

## [v0.0.16] — 2026-05-30

### Added

- `///` doc comments and SAFETY comments to all test functions in `transpose.rs`.
- Test: transpose slice offset preserved.
- Test: double-transpose identity for 3D (`transpose_high_dim`).
- Module-level doc for `shape/transpose.rs`.

### Removed

- Stale design-doc cross-references and task-tracking from shape module.
- Empty `test_shape_module_compiles` skeleton test.

### Changed

- Moved `transpose_impl` before `impl` block for top-down dependency order.
- Consolidated imports and removed redundant `.clone()` on `Copy` types.
- Moved inline `use` to top-level test imports in `transpose.rs`.
- Cleaned up shape module docs with concise descriptions.

## [v0.0.15] — 2026-05-30

### Added

- Doc comments to all `Display` impls in `error.rs`.
- Display output tests for all error enum variants.
- Test: `classify` for `LayoutFlags::EMPTY`.
- Test: derive traits for `Axis`.
- Additional `is_last` assertion for `Axis`.

### Removed

- Dead `AbiMismatchKind` enum and related `FfiErrorCategory` variants.
- Dead `StridesRankMismatch`/`UnsupportedStride`/`UnexpectedZeroStride` from `InvalidLayoutReason`.
- Dead `flags_for_f_layout` function and tests.
- Dead `ElementType::of::<A>()` const method.
- Dead `Axis::checked_next` method.

### Changed

- Updated `test_crate_metadata` error message to current version.

## [v0.0.14] — 2026-05-29

### Added

- Module-level doc and `///` doc comments to all test functions in `set/unique.rs`.
- Doc comments to all `UniqueElement` impls (`unique_eq`).
- Doc comments to `unique_impl` and `assert_set_eq_i32` helper.
- Test: `unique` on `Complex<f32>` values.
- Test: `unique` on tensor view.

### Removed

- Redundant `test_set_module_exports_unique_element` test.
- Stale section separator comments and task-tracking references from `unique.rs`.

### Changed

- Consolidated test imports in `unique.rs` using pre-imported short paths.
- Replaced fully-qualified `crate::dimension::IxDyn` with imported `IxDyn`.
- Converted `///` SAFETY doc comments to `//` in `owned.rs`.
- Escaped angle brackets in `unique.rs` module doc to prevent HTML parsing.

## [v0.0.13] — 2026-05-29

### Added

- Test coverage for `ArcRepr::zeros` and `ArcRepr::from_elem`.
- `ViewRepr` tests: from_raw_parts, sub-view, into_owned.
- `ViewMutRepr` tests: from_raw_parts_mut, sub-view, into_owned.
- `Owned::from_elem` and `Owned::zeros_non_float` tests.
- Trait method tests for `is_aligned_to`, `get_unchecked`, `get_mut`, `get_unchecked_mut` in `traits.rs`.
- Mock runtime tests for `StorageOwned` and `StorageShared` in `traits.rs`.

### Removed

- `RawStorageMut` trait (folded into `StorageMut`).
- `cause` field from `XenonError::Ffi` and `XenonError::Workspace` variants.
- `StorageConversionKind` enum.
- `LayoutMismatch` variant.
- `SplitCountInvariant` from `WorkspaceErrorCategory`.
- `RankExceedsStaticMax` from `InvalidShapeKind`.
- Stale section separator comments and redundant inherent `impl` blocks across storage module.

### Changed

- Consolidated all inherent `impl` blocks into one per struct across storage module.
- Reordered trait impls for logical grouping.
- Replaced fully-qualified paths with local imports throughout storage module.
- Converted `///` SAFETY doc comments to `//` comments on unsafe impl blocks in `owned.rs`.
- Reorganized `storage/mod.rs` declarations and re-exports.
- Broken intra-doc links for `pub(crate)` items across storage module.

## [v0.0.12] — 2026-05-28

### Added

- `aligned.rs` submodule for pointer alignment checks (`is_aligned`/`is_aligned_to`).
- Module-level doc with submodule table in `layout/mod.rs`.
- `///` doc comments to all layout test functions.

### Removed

- Stale design-doc cross-references from all layout submodules.
- Standalone `compute_f_strides` function (inlined into `Strides::f_contiguous`).

### Changed

- Colocated `flags_for_f_layout` into `LayoutFlags` impl block; `compute_layout_flags` and all tests into `flags.rs`.
- Moved `has_zero_stride`/`should_set_zero_stride_flag` into `Strides` methods.
- Moved `is_aligned`/`is_aligned_to` from `strides.rs` to `flags.rs`, then to dedicated `aligned.rs`.
- Switched all callers from `compute_f_strides` to `Strides::f_contiguous()`.
- Replaced fully-qualified paths with local imports throughout element module.
- Intra-doc links for `pub(crate)` items.

## [v0.0.11] — 2026-05-28

### Added

- `///` doc comments and section separators across all element submodules.

### Removed

- Unused `BoolElement` trait.
- `element_type_name_of` free function.

### Changed

- Extracted `checked.rs` from overload module; `CastTo`/`CastElement` into `convert/cast.rs`.
- Moved `Element` trait to `element/primitives.rs`; `OrderedCompareElement` to `element/order.rs`.
- Consolidated all `Sealed` impls into `src/private.rs`.
- Moved `Numeric`/`RealScalar`/`ComplexScalar` impls to respective submodules.
- FFI raw structs switched from `ElementType` to `u8` for C ABI compatibility.
- Restricted `ElementType` helpers (`name`, `of`, `element_type_of`) to `pub(crate)`.
- Replaced fully-qualified paths with local imports throughout element module.

## [v0.0.10] — 2026-05-27

### Added

- `Reverse` and `RemoveAxis` tests for `Ix4`-`Ix6` (9 new tests, 688→697).
- Added `RemoveAxis` impls for `Ix4`, `Ix5`, `Ix6`.

### Removed

- Stale design-doc cross-references (`5.x`, `8.x`) from `dimension` doc comments.
- Unused test helper `assert_dimension_bounds` in `types.rs`.

### Changed

- Replaced fully-qualified paths with direct imports throughout `dimension` module.
- Made `axes`/`broadcast`/`dynamic`/`fixed`/`into` submodules private; exposed via re-exports.
- Extracted `Dimension`/`Reverse`/`RemoveAxis` traits into `dimension/types.rs`.
- Moved `MAX_DIMENSION` and tests from `mod.rs` into `types.rs`.
- Reordered submodule declarations and re-exports in `dimension/mod.rs`.
- Colocated `Reverse`/`RemoveAxis`/`Index`/`From` impls with respective `IxN` structs.
- Moved `into_dyn`/`try_from_dyn` methods next to each `IxN` struct definition.
- Added `///` doc comments to all `Reverse`/`RemoveAxis` impls.
- Missing `IntoDimension` import in `types.rs` test module.


## [v0.0.9] — 2026-05-27

### Added

- `///` doc comments to all test functions across the complex module.

### Removed

- Stale design-doc and task-tracking cross-references from doc comments.

### Changed

- Replaced fully-qualified paths with direct imports throughout error.rs, display.rs, ops.rs, and types.rs.
- Extracted `ComplexFloat` trait and `Complex` struct into a dedicated `types.rs` sub-module.
- Moved `ComplexFloat`/`Complex` unit tests from `mod.rs` into `types.rs`.

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
- Synchronized test version assertion and cleaned up inactive work-in-progress test placeholders.
