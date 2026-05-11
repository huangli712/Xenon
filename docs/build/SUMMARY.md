# Xenon Implementation Plan — Wave & Task Summary

> Project: Xenon N-dimensional array library (Rust)
> Dependency Layers: L0 → L1 → L2 → L3 → L4 → L5 → L6 → Cross-cutting
> Task Granularity: Each task targets 1 function / 1 trait / 1 type, ~5–10 min, max 1 file

---

## Wave Overview

| Wave | Name | Layer | Task Count |  |
|------|------|-------|------------|-------------|
| W1 | Coding Standards & Project Setup | L0 | 6 | Cargo.toml, rustfmt.toml, lib.rs/prelude.rs skeleton with lint attrs, .clippy.toml, CI config |
| W2 | Error System | L0 | 5 | XenonError enum, Result alias, Display/Error impls, auxiliary enums, prelude exports |
| W3 | Dimension System | L1 | 21 | Static dims (Ix0–Ix6), IxDyn, Dimension/IntoDimension/RemoveAxis traits, Axis, Sealed |
| W4 | Element Type Hierarchy | L1 | 17 |  |
| W5 | Complex Type | L1 | 16 |  |
| W6 | Layout System | L2 | 10 | LayoutFlags bitflags, F-order stride computation, contiguity checks, alignment, zero-stride detection |
| W7 | Storage System | L2 | 19 | RawStorage/Storage/StorageMut/RawStorageMut/StorageOwned/StorageShared traits, marker traits, Owned/A, AlignedAlloc, ViewRepr, ViewMutRepr, ArcRepr |
| W8 | Tensor Core | L3 | 10 |  |
| W9 | Workspace | L2 | 7 | Workspace struct, borrow guards (WorkspaceBorrow/WorkspaceBorrowMut), split (SplitBorrowMut), expand (ensure_capacity/reallocate), docs |
| W10 | Dispatch | L4 | 6 | ExecPath enum, select_exec_path, thresholds, ParallelGuard (nested parallel protection), ParallelExecStrategy |
| W11 | Broadcasting | L4 | 10 | BroadcastDim trait, can_broadcast, broadcast_shape, broadcast_strides, broadcast_to, broadcast_with, error handling, integration tests |
| W12 | Iterators | L4 | 7 |  |
| W13 | FFI Helpers | L4 | 6 |  |
| W14 | SIMD Backend | L5 | 10 | SimdKernel trait, element-wise SIMD (add/sub/mul/div), SIMD sum (float/int/complex), SIMD dot, feature gates, property tests |
| W15 | Parallel Backend | L5 | 8 | ParIter, par_map, par_zip_map, par_sum, par_dot, ParallelPool, error/panic propagation, feature gates |
| W16 | Math Operations | L5 | 11 | Binary element-wise ops (add/sub/mul/div), unary ops (abs/neg/signum/square/sin/sqrt/exp/ln/floor/ceil/conj/modulus), comparison ops (eq/ne/lt/le/gt/ge), logical not, SIMD dispatch |
| W17 | Matrix Operations | L5 | 7 | dot product (serial + SIMD + parallel paths), rank/shape validation, complex dot, integration tests |
| W18 | Reduction Operations | L5 | 6 | sum (global), sum_axis, sum_axis_keepdims, SIMD/parallel dispatch gates, error convergence |
| W19 | Set Operations | L5 | 6 |  |
| W20 | Shape Operations | L5 | 4 | transpose (full-axis reversal), contiguity recomputation, integration tests |
| W21 | Indexing | L5 | 6 | index module root, NdIndex trait, try_at/get/get_unchecked, SliceInfo, try_at_mut/get_mut/get_unchecked_mut, slice shape/stride update |
| W22 | Tensor Construction | L5 | 9 |  |
| W23 | Operator Overloading | L6 | 9 | overload module root, Add/Sub/Mul/Div for owned/ref/mixed/scalar tensor combinations, integration tests |
| W24 | Utility Operations | L5 | 5 | util module root, fill, clip, to_contiguous/into_contiguous |
| W25 | Type Conversion | L5 | 7 |  |
| W26 | Output Formatting | L5 | 6 | FormatConfig, Display (Numpy-style), Debug (with metadata), pretty formatting helpers |
| W27 | Safety Audit | cross-cutting | 7 | Send/Sync impls for Owned/ViewRepr/ViewMutRepr/ArcRepr, parallel chunk safety, thread-safety integration tests |
| W28 | Benchmarks | cross-cutting | 12 | bench infrastructure (utils/generators), core benches (math/reduction/dot/set/broadcast), shape/construction benches, SIMD/parallel comparison, CI/report script |
| W29 | Integration Tests | cross-cutting | 25 |  |
| W30 | Documentation | cross-cutting | 52 | Crate-level docs, per-module docs, type/function-level docs + doctests, usage examples, README/LICENSE/CHANGELOG, docs CI |
| | **Total** | | **330** | |

---

## Detailed Task List

### W1: Coding Standards & Project Setup (L0)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W1T1 | `Cargo.toml` | Package manifest with crate metadata, edition/MSRV, feature gates, optional deps, profiles, docs.rs metadata | None | 01-architecture §4, 00-coding §10 |
| W1T2 | `rustfmt.toml` | Rustfmt configuration per §3.2 coding standard | W1T1 | 00-coding §12 |
| W1T3 | `src/lib.rs` | Crate root skeleton with lint declarations (missing_docs, unsafe_op_in_unsafe_fn, clippy::unwrap_used) | W1T1 | 00-coding §12, 01-architecture §3 |
| W1T4 | `src/prelude.rs` | Prelude module skeleton and initial public export surface placeholder | W1T3 | 01-architecture §3, §7 |
| W1T5 | `.clippy.toml` | Clippy configuration for numeric `as` casts lint, unwrap restrictions | W1T1 | 00-coding §12 |
| W1T6 | `.github/workflows/ci.yml` | CI config: fmt + clippy + test matrix across feature combinations | W1T3 | 00-coding §12 |

### W2: Error System (L0)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W2T1 | `src/error.rs` | XenonError enum definition with all structured variants | None | 26-error §5, §6, §7 |
| W2T2 | `src/error.rs` | Auxiliary enum types: FfiErrorCategory, WorkspaceErrorCategory, ConversionFailureReason + Result alias | W2T1 | 26-error §5, §6, §7 |
| W2T3 | `src/error.rs` | fmt::Display impl for XenonError with OrAny\<T\>, FmtShape helpers | W2T2 | 26-error §5, §6, §7 |
| W2T4 | `src/error.rs` | std::error::Error impl (source chaining for Ffi/Workspace, None for leaf variants) | W2T2 | 26-error §5, §6, §7 |
| W2T5 | `src/error.rs`, `src/lib.rs` | Public exports of XenonError, Result alias, auxiliary enums via prelude | W2T4 | 26-error §5, §6, §7 |

### W3: Dimension System (L1)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W3T1 | `src/dimension/mod.rs` | Module skeleton: sub-module declarations, public re-exports | None | 02-dimension §5, §6, §7 |
| W3T2 | `src/dimension/mod.rs` | Dimension trait definition (all method signatures, MAX_DIMENSION constant) | W3T1 | 02-dimension §5, §6, §7 |
| W3T3 | `src/dimension/static.rs` | Ix0 zero-dimensional scalar with Dimension impl | W3T2 | 02-dimension §5, §6, §7 |
| W3T4 | `src/dimension/static.rs` | Ix1 struct with Dimension impl + Index\<usize\> | W3T3 | 02-dimension §5, §6, §7 |
| W3T5 | `src/dimension/static.rs` | Ix2 struct with Dimension impl + Index\<usize\> | W3T4 | 02-dimension §5, §6, §7 |
| W3T6 | `src/dimension/static.rs` | Ix3 struct with Dimension impl + From\<(usize, usize, usize)\> | W3T5 | 02-dimension §5, §6, §7 |
| W3T7 | `src/dimension/static.rs` | Ix4 struct with Dimension impl + From\<tuple\> | W3T6 | 02-dimension §5, §6, §7 |
| W3T8 | `src/dimension/static.rs` | Ix5 struct with Dimension impl + From\<tuple\> | W3T7 | 02-dimension §5, §6, §7 |
| W3T9 | `src/dimension/static.rs` | Ix6 struct with Dimension impl + From\<tuple\> | W3T8 | 02-dimension §5, §6, §7 |
| W3T10 | `src/dimension/dynamic.rs` | IxDyn dynamic dimension with Dimension impl + constructors | W3T2 | 02-dimension §5, §6, §7 |
| W3T11 | `src/dimension/static.rs`, `dynamic.rs` | into_dyn() / try_from_dyn() conversion methods | W3T9, W3T10 | 02-dimension §5, §6, §7 |
| W3T12 | `src/error.rs` | XenonError::DimensionMismatch integration for dimension conversion failures | W2T1, W3T11 | 02-dimension §5, §6, §7 |
| W3T13 | `src/dimension/into.rs` | IntoDimension trait + tuple/array/slice/Vec impls | W3T12 | 02-dimension §5, §6, §7 |
| W3T14 | `src/dimension/axes.rs` | Axis newtype with new/index/checked_next/next/prev/is_first/is_last | W3T1 | 02-dimension §5, §6, §7 |
| W3T15 | `src/private.rs`, `src/dimension/mod.rs` | Sealed trait impl for all dimension types + public exports | W3T12, W3T13, W3T14 | 02-dimension §5, §6, §7 |
| W3T16 | All `src/dimension/` files | Doc comments on all pub items, cargo doc verification | W3T15 | 02-dimension §5, §6, §7 |
| W3T17 | `src/dimension/` (`#[cfg(test)] mod tests`) | Dimension in-module unit tests: Ix0, zero-length axis, large dim, overflow | W3T16 | 02-dimension §5, §6, §7 |
| W3T18 | `tests/test_tensor.rs` | Dimension cross-module tests for tensor interaction | W3T16 | 02-dimension §5, §6, §7 |
| W3T19 | `tests/test_shape.rs` | Dimension cross-module tests for shape operations | W3T16 | 02-dimension §5, §6, §7 |
| W3T20 | `tests/test_index.rs` | Dimension cross-module tests for indexing | W3T16 | 02-dimension §5, §6, §7 |
| W3T21 | `tests/property_tests.rs` | Dimension cross-module property tests | W3T16 | 02-dimension §5, §6, §7 |

### W4: Element Type Hierarchy (L1)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W4T1 | `src/element/mod.rs` | Module skeleton + import shared Sealed + Element trait definition | W3T15, W2T1 | 03-element §5, §6, §7 |
| W4T2 | `src/element/numeric.rs` | Numeric trait definition (arithmetic supertraits + conjugate) | W4T1 | 03-element §5, §6, §7 |
| W4T3 | `src/element/real.rs` | RealScalar trait: math functions (abs/sqrt/sin/exp/ln/floor/ceil) + NaN detection | W4T2 | 03-element §5, §6, §7 |
| W4T4 | `src/element/complex.rs` | ComplexScalar trait: associated type Real + complex methods (re/im/norm) | W4T2 | 03-element §5, §6, §7 |
| W4T5 | `src/element/primitives.rs` | Element + Numeric impl for i32 | W4T2 | 03-element §5, §6, §7 |
| W4T6 | `src/element/primitives.rs` | Element + Numeric impl for i64 | W4T5 | 03-element §5, §6, §7 |
| W4T7 | `src/element/primitives.rs` | Element + Numeric + RealScalar impls for f32, f64 | W4T3, W4T6 | 03-element §5, §6, §7 |
| W4T8 | `src/element/primitives.rs` | Element impl for bool (zero=false, one=true, no Numeric) | W4T1 | 03-element §5, §6, §7 |
| W4T9 | `src/element/mod.rs` | Documentation clarifying usize as index/shape metadata only, not an element type | W4T1 | 03-element §5, §6, §7 |
| W4T10 | `src/element/primitives.rs` | Element + Numeric + ComplexScalar impls for Complex\<f32\>/Complex\<f64\> | W4T4, W5T1 | 03-element §5, §6, §7 |
| W4T11 | `src/element/real.rs`, `src/element/mod.rs` | Calibrate math capability boundaries + document lossy CastTo error semantics | W4T7 | 03-element §5, §6, §7 |
| W4T12 | All `src/element/` files | Doc comments on all pub items, cargo doc verification | W4T10, W4T11 | 03-element §5, §6, §7 |
| W4T13 | `src/element/` (`#[cfg(test)] mod tests`) | Element in-module unit tests: verify trait impls for all 7 element types | W4T10 | 03-element §5, §6, §7 |
| W4T14 | `tests/test_tensor.rs` | Element cross-module tests for tensor interaction | W4T10 | 03-element §5, §6, §7 |
| W4T15 | `tests/test_math.rs` | Element cross-module tests for math operations | W4T10 | 03-element §5, §6, §7 |
| W4T16 | `tests/test_reduction.rs` | Element cross-module tests for reductions | W4T10 | 03-element §5, §6, §7 |
| W4T17 | `tests/test_conversion.rs` | Element cross-module tests for type conversion | W4T10 | 03-element §5, §6, §7 |

### W5: Complex Type (L1)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W5T1 | `src/complex/mod.rs` | Complex\<T\> struct (repr(C)) + new() constructor | W1T3 | 04-complex §5, §6, §7 |
| W5T2 | `src/complex/mod.rs` | ComplexFloat sealed trait + f32/f64 impls | W5T1, W3T15 | 04-complex §5, §6, §7 |
| W5T3 | `src/complex/mod.rs` | const size/align assertions for FFI layout guarantees | W5T1 | 04-complex §5, §6, §7 |
| W5T4 | `src/complex/mod.rs` | re() / im() accessors | W5T1 | 04-complex §5, §6, §7 |
| W5T5 | `src/complex/mod.rs` | from_imag(), conj(), and From\<T\> constructors | W5T1, W5T4 | 04-complex §5, §6, §7 |
| W5T6 | `src/complex/mod.rs` | is_real() / is_imaginary() predicates | W5T1 | 04-complex §5, §6, §7 |
| W5T7 | `src/complex/mod.rs` | PartialEq (NaN!=NaN) + Display (a+bj format) impls | W5T1 | 04-complex §5, §6, §7 |
| W5T8 | `src/complex/ops.rs` | Complex Add operator | W5T1 | 04-complex §5, §6, §7 |
| W5T9 | `src/complex/ops.rs` | Complex Sub operator | W5T8 | 04-complex §5, §6, §7 |
| W5T10 | `src/complex/ops.rs` | Complex Mul operator | W5T1 | 04-complex §5, §6, §7 |
| W5T11 | `src/complex/ops.rs` | Complex Div operator | W5T10 | 04-complex §5, §6, §7 |
| W5T12 | `src/complex/ops.rs` | Complex Neg operator | W5T1 | 04-complex §5, §6, §7 |
| W5T13 | `src/complex/ops.rs` | Tighten real-complex mixed arithmetic boundary (Complex op Complex only, explicit conversion) | W5T8–W5T12 | 04-complex §5, §6, §7 |
| W5T14 | `src/complex/mod.rs` | Math methods: norm (hypot), norm_sqr | W5T1 | 04-complex §5, §6, §7 |
| W5T15 | All `src/complex/` files | Doc comments on all pub complex items, cargo doc verification | W5T13, W5T14 | 04-complex §5, §6, §7 |
| W5T16 | `tests/test_complex.rs` | Integration and boundary tests for full complex type system | W5T15 | 04-complex §5, §6, §7 |

### W6: Layout System (L2)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W6T1 | `src/layout/mod.rs` | Module skeleton: sub-module declarations, public exports | None | 06-layout §5, §6, §7 |
| W6T2 | `src/layout/flags.rs` | Module skeleton: file placeholder, declarations | W6T1 | 06-layout §5, §6, §7 |
| W6T3 | `src/layout/strides.rs` | Module skeleton: file placeholder, declarations | W6T1 | 06-layout §5, §6, §7 |
| W6T4 | `src/layout/contiguous.rs` | Module skeleton: file placeholder, declarations | W6T1 | 06-layout §5, §6, §7 |
| W6T5 | `src/layout/flags.rs` | LayoutFlags(u8) bitflags: F_CONTIGUOUS, ALIGNED, HAS_ZERO_STRIDE + query/set methods | W6T2 | 06-layout §5, §6, §7 |
| W6T6 | `src/layout/strides.rs` | compute_f_strides\<D\>: F-order stride computation with overflow check returning Result | W6T3, W3T2 | 06-layout §5, §6, §7 |
| W6T7 | `src/layout/contiguous.rs` | is_f_contiguous\<D\>: F-order contiguity detection | W6T4, W3T2 | 06-layout §5, §6, §7 |
| W6T8 | `src/layout/strides.rs` | has_zero_stride: raw zero-stride detector (flag assignment checks product(shape) > 0) | W6T3, W3T2 | 06-layout §5, §6, §7 |
| W6T9 | `src/layout/strides.rs` | is_aligned_to / is_aligned alignment check functions | W6T3 | 06-layout §5, §6, §7 |
| W6T10 | `src/layout/` (`#[cfg(test)] mod tests`), `tests/test_tensor.rs`, `tests/test_shape.rs`, `tests/test_index.rs`, `tests/test_ffi.rs`, `tests/test_simd.rs` | Layout integration tests | W6T6–W6T9 | 06-layout §5, §6, §7 |

### W7: Storage System (L2)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W7T1 | `src/storage/mod.rs` | Module skeleton: sub-module declarations, public exports | None | 05-storage §5, §6, §7 |
| W7T2 | `src/storage/mod.rs` | unsafe trait RawStorage: as_ptr/len/is_empty/is_aligned_to/is_aligned | W7T1 | 05-storage §5, §6, §7 |
| W7T3 | `src/storage/mod.rs` | unsafe trait Storage: RawStorage + get/get_unchecked/as_slice | W7T2 | 05-storage §5, §6, §7 |
| W7T4 | `src/storage/mod.rs` | RawStorageMut and StorageMut trait definitions | W7T3 | 05-storage §5, §6, §7 |
| W7T5 | `src/storage/mod.rs` | StorageOwned and StorageShared trait definitions | W7T4 | 05-storage §5, §6, §7 |
| W7T6 | `src/storage/traits.rs` | Marker traits: IsOwned, IsView, IsViewMut, IsShared | W7T5 | 01-architecture §3, 05-storage §5, §6, §7 |
| W7T7 | `src/storage/alloc.rs` | AlignedAlloc struct: 64-byte aligned alloc/alloc_zeroed/dealloc | W7T1 | 05-storage §5, §6, §7 |
| W7T8 | `src/storage/owned.rs` | Owned\<A\> struct + new/with_capacity/from_vec/from_vec_aligned/zeros/from_elem constructors | W7T7 | 05-storage §5, §6, §7 |
| W7T9 | `src/storage/owned.rs` | Owned\<A\> RawStorage trait impl | W7T5, W7T8 | 05-storage §5, §6, §7 |
| W7T10 | `src/storage/owned.rs` | Owned\<A\> Storage trait impl | W7T9 | 05-storage §5, §6, §7 |
| W7T11 | `src/storage/owned.rs` | Owned\<A\> StorageMut + StorageOwned trait impls | W7T10 | 05-storage §5, §6, §7 |
| W7T12 | `src/storage/owned.rs` | Owned\<A\> into_shared + unsafe impl Send + unsafe impl Sync | W7T8, W7T16 | 05-storage §5, §6, §7 |
| W7T13 | `src/storage/owned.rs` | Owned\<A\> From + Default impls | W7T8 | 05-storage §5, §6, §7 |
| W7T14 | `src/storage/view.rs` | ViewRepr\<'a, A\>: struct, from_raw_parts/from_slice/view/slice, Clone/Copy, RawStorage/Storage impls | W7T5, W7T6 | 05-storage §5, §6, §7 |
| W7T15 | `src/storage/viewmut.rs` | ViewMutRepr\<'a, A\>: struct, from_raw_parts_mut/from_mut_slice/view_mut/view, no Clone, all trait impls | W7T5, W7T6 | 05-storage §5, §6, §7 |
| W7T16 | `src/storage/arc.rs` | ArcRepr\<A\>: struct, from_vec/from_vec_aligned/zeros/from_elem constructors, Clone (ref-count bump), all trait impls | W7T5 | 05-storage §5, §6, §7 |
| W7T17 | `src/storage/arc.rs` | ArcRepr\<A\> Send/Sync + Default + From impls | W7T16 | 05-storage §5, §6, §7 |
| W7T18 | `src/storage/mod.rs` | Module re-exports (all types + traits) + doc comments | W7T13, W7T14, W7T15, W7T17 | 05-storage §5, §6, §7 |
| W7T19 | `src/storage/` (`#[cfg(test)] mod tests`), `tests/test_tensor.rs`, `tests/test_index.rs`, `tests/test_ffi.rs`, `tests/test_iterator.rs` | Storage integration tests: cross-storage interaction, ZST behavior, empty arrays, alloc alignment, trait semantics via in-module unit tests + cross-module coverage through existing tensor/index/ffi/iterator test files; do NOT add standalone tests/test_storage.rs | W7T18 | 05-storage §5, §6, §7 |

### W8: Tensor Core (L3)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W8T1 | `src/tensor/mod.rs` | Module skeleton: sub-module declarations, public exports | None | 07-tensor §5, §6, §7 |
| W8T2 | `src/tensor/mod.rs` | TensorBase\<S, D\> struct: 6 fields (storage, shape, strides, offset, flags, derived_from_view_mut) | W8T1, W6T5 | 07-tensor §5, §6, §7 |
| W8T3 | `src/tensor/aliases.rs` | 4 primary type aliases (Tensor/TensorView/TensorViewMut/ArcTensor) + 4×8=32 dim convenience aliases | W8T2 | 07-tensor §5, §6, §7 |
| W8T4 | `src/tensor/impls.rs` | Shape & stride query methods: shape/strides/ndim/len/is_empty/offset/raw_dim/flags/storage_kind/access_semantics/data_location | W8T2, W3T2, W6T5 | 07-tensor §5, §6, §7 |
| W8T5 | `src/tensor/impls.rs` | Layout query delegation: layout_state/is_f_contiguous/is_aligned/has_zero_stride | W8T4 | 07-tensor §5, §6, §7 |
| W8T6 | `src/tensor/impls.rs` | Pointer access & slice: as_ptr/as_storage_ptr/as_mut_ptr/as_slice/as_mut_slice | W8T4 | 07-tensor §5, §6, §7 |
| W8T7 | `src/tensor/construct.rs` | from_raw_parts / from_raw_parts_mut with storage_len and validate_access_range | W8T2, W6T5, W3T2 | 07-tensor §5, §6, §7 |
| W8T8 | `src/tensor/construct.rs` | from_raw_vec_unchecked (pub(crate) unsafe internal constructor) | W8T5, W8T7 | 07-tensor §5, §6, §7 |
| W8T9 | `src/tensor/impls.rs` | View creation methods: view() / view_mut() | W8T6 | 07-tensor §5, §6, §7 |
| W8T10 | `tests/test_tensor.rs` | Integration tests: cross-module interaction, boundary tests, type alias compilation | W8T3, W8T9 | 07-tensor §5, §6, §7 |

### W9: Workspace (L2)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W9T1 | `src/error.rs` | WorkspaceErrorCategory definition, wired into XenonError::Workspace | W2T2 | 24-workspace §5, §6, §7 |
| W9T2 | `src/workspace/workspace.rs` | Workspace struct, constants, new(), with_default_capacity(), Drop | W9T1 | 24-workspace §5, §6, §7 |
| W9T3 | `src/workspace/mod.rs` | Module root: sub-module declarations, re-exports | W9T1 | 24-workspace §5, §6, §7 |
| W9T4 | `src/workspace/borrow.rs` | WorkspaceBorrow/WorkspaceBorrowMut guards + borrow/borrow_mut + MaybeUninit access methods + Drop | W9T2, W9T1 | 24-workspace §5, §6, §7 |
| W9T5 | `src/workspace/split.rs` | SplitBorrowMut guard, split_at_mut (top-level + recursive), Drop | W9T2, W9T1 | 24-workspace §5, §6, §7 |
| W9T6 | `src/workspace/expand.rs` | ensure_capacity() / reallocate() expansion strategy | W9T2, W9T1 | 24-workspace §5, §6, §7 |
| W9T7 | `src/workspace/mod.rs` + sub-modules | Complete public exports + doc comments + cargo doc verification | W9T4, W9T5, W9T6 | 24-workspace §5, §6, §7 |

### W10: Dispatch (L4)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W10T1 | `src/dispatch.rs` | Module skeleton: ExecPath enum, ParallelExecStrategy struct, module docs | None | 30-dispatch §5, §6, §7 |
| W10T2 | `src/dispatch.rs` | ParallelGuard type + try_acquire_guard(): thread_local Cell\<bool\>, Drop impl | W10T1 | 30-dispatch §5, §6, §7 |
| W10T3 | `src/dispatch.rs` | ParallelExecStrategy::new() validating constructor, auto() infallible default, field accessors | W10T1 | 30-dispatch §5, §6, §7 |
| W10T4 | `src/dispatch.rs` | select_exec_path() + should_parallelize(): three-way dispatch, threshold reading, feature gate branching | W10T1, W10T2 | 30-dispatch §5, §6, §7 |
| W10T5 | `src/dispatch.rs` | Threshold config: AtomicUsize storage, set/reset_parallel_threshold, Relaxed ordering | W10T4 | 30-dispatch §5, §6, §7 |
| W10T6 | `src/dispatch.rs` (#[cfg(test)]) | Full dispatch unit tests: path return, threshold boundaries, feature gate combos, nested guard, non-contiguous penalty | W10T4, W10T5, W10T2, W10T3 | 30-dispatch §5, §6, §7 |

### W11: Broadcasting (L4)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W11T1 | `src/broadcast/mod.rs` | Module skeleton: sub-module declarations, re-exports | None | 15-broadcast §5, §6, §7 |
| W11T2 | `src/broadcast/shape.rs` | Module skeleton: file placeholder, rule function stubs | W11T1 | 15-broadcast §5, §6, §7 |
| W11T3 | `src/broadcast/view.rs` | Module skeleton: file placeholder, view entry stubs | W11T1 | 15-broadcast §5, §6, §7 |
| W11T4 | `src/broadcast/shape.rs` | can_broadcast(): trailing-axis alignment compatibility check | W11T2 | 15-broadcast §5, §6, §7 |
| W11T5 | `src/broadcast/shape.rs` | broadcast_shape(): shared shape derivation with structured broadcast errors | W11T4 | 15-broadcast §5, §6, §7 |
| W11T6 | `src/broadcast/shape.rs` | broadcast_strides(): zero-stride insertion with input precondition validation | W11T5 | 15-broadcast §5, §6, §7 |
| W11T7 | `src/broadcast/view.rs` | broadcast_to() basic path: target shape validation + read-only view construction | W11T6 | 15-broadcast §5, §6, §7 |
| W11T8 | `src/broadcast/view.rs` | broadcast_to() error path + BroadcastView layout state update | W11T7 | 15-broadcast §5, §6, §7 |
| W11T9 | `src/broadcast/view.rs` | broadcast_with(): shared shape derivation, dual-input broadcast | W11T6, W11T7 | 15-broadcast §5, §6, §7 |
| W11T10 | `tests/test_broadcast.rs` | Integration tests | W11T8, W11T9 | 15-broadcast §5, §6, §7 |

### W12: Iterators (L4)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W12T1 | `src/iter/mod.rs` | Module skeleton: declarations, sub-module placeholders, public exports | None | 10-iterator §5, §6, §7 |
| W12T2 | `src/iter/elements.rs` | StrideState: F-order index increment state machine | W12T1 | 10-iterator §5, §6, §7 |
| W12T3 | `src/iter/elements.rs` | Iter: Iterator + ExactSizeIterator impls with fast/slow paths (contiguous vs non-contiguous) | W12T2 | 10-iterator §5, §6, §7 |
| W12T4 | `src/iter/elements.rs` | IterMut: Iterator + ExactSizeIterator impls | W12T3 | 10-iterator §5, §6, §7 |
| W12T5 | `src/iter/axis.rs` | AxisIter / AxisIterMut: iterate along one axis, yielding sub-views | W12T1 | 10-iterator §5, §6, §7 |
| W12T6 | `src/iter/indexed.rs` | IndexedIter / IndexedIterMut: index-wrapped iteration based on Iter | W12T4 | 10-iterator §5, §6, §7 |
| W12T7 | `src/iter/mod.rs` | TensorBase entry methods: iter(), iter_mut(), axis_iter(), indexed_iter() | W12T4, W12T5, W12T6 | 10-iterator §5, §6, §7 |

### W13: FFI Helpers (L4)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W13T1 | `src/ffi/mod.rs` | Module skeleton: sub-module declarations, re-exports | W8T7 | 23-ffi §5, §6, §7 |
| W13T2 | `src/ffi/types.rs` | Module skeleton + FfiErrorCategory + BlasInfo struct | W13T1 | 23-ffi §5, §6, §7 |
| W13T3 | `src/ffi/private.rs` | Internal generic descriptors: TensorExport and TensorExportMut | W13T2 | 23-ffi §5, §6, §7 |
| W13T4 | `src/ffi/ptr.rs` | Re-export as_ptr/as_mut_ptr/from_raw_parts/from_raw_parts_mut/into_raw_parts + FFI wrappers export()/export_mut() | W13T3 | 23-ffi §5, §6, §7 |
| W13T5 | `src/ffi/blas.rs` | is_blas_layout_compatible(), blas_info(), lda() | W13T2 | 23-ffi §5, §6, §7 |
| W13T6 | `src/ffi/offset.rs` | try_offset_of() / try_ptr_at() with checked arithmetic validation | W13T2 | 23-ffi §5, §6, §7 |

### W14: SIMD Backend (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W14T1 | `src/simd/mod.rs` | Module skeleton: SimdElement trait, SimdKernel trait, Arch cache, public vectorized entry points | None | 08-simd §5, §6, §7 |
| W14T2 | `src/simd/vector.rs` | Element-wise SIMD: VectorKernel\<A\>, Add/Sub/Mul/Div Kernel WithSimd impls for f32/f64 | W14T1 | 08-simd §5, §6, §7 |
| W14T3 | `src/simd/vector.rs` | Float sum SIMD: f32/f64 SumKernel with lane accumulation + documented merge | W14T2 | 08-simd §5, §6, §7 |
| W14T4 | `src/simd/vector.rs` | Integer sum/dot SIMD admission + fallback: i32/i64 checked semantics, ISA widening kernel gating | W14T3 | 08-simd §5, §6, §7 |
| W14T5 | `src/simd/vector.rs` | Complex sum SIMD: Complex\<f32\>/Complex\<f64\> AoS split real/imag accumulation | W14T3 | 08-simd §5, §6, §7 |
| W14T6 | `src/simd/vector.rs` | Float + complex dot SIMD: f32/f64/Complex\<f32\>/Complex\<f64\> dot kernel with conjugate contract | W14T3, W14T5 | 08-simd §5, §6, §7 |
| W14T7 | `src/simd/mod.rs`, `Cargo.toml` | Feature gate conditional compilation: #[cfg(feature = "simd")], public API export, dispatch integration | W14T1, W14T2 | 08-simd §5, §6, §7 |
| W14T8 | `src/simd/vector.rs` (#[cfg(test)]) | Element-wise consistency tests: SIMD vs serial baseline bitwise agreement | W14T7 | 08-simd §5, §6, §7 |
| W14T9 | `src/simd/vector.rs` (#[cfg(test)]) | Reduction/dot semantic + tolerance tests: float/complex/int SIMD entry condition constraints | W14T8 | 08-simd §5, §6, §7 |
| W14T10 | `tests/property/` | Randomized property tests: element-wise consistency + reduction/dot invariants under random input | W14T9 | 08-simd §5, §6, §7 |

### W15: Parallel Backend (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W15T1 | `src/parallel/iter.rs` | ParIter + TensorBase::par_iter(): single-input element-level parallel traversal entry | W10T4 | 09-parallel §5, §6, §7 |
| W15T2 | `src/parallel/map.rs` | par_map: pure parallel element-wise map entry, strategy from dispatch | W15T1 | 09-parallel §5, §6, §7 |
| W15T3 | `src/parallel/map.rs` | par_zip_map: dual-input broadcast element-wise parallel entry for math consumption | W15T1 | 09-parallel §5, §6, §7 |
| W15T4 | `src/parallel/reduce.rs` | par_reduce_impl + par_sum: parallel reduction, identity merge, semantic alignment with caller's serial baseline | W15T1 | 09-parallel §5, §6, §7 |
| W15T5 | `src/parallel/reduce.rs` | par_dot: ndim==1 check, length consistency, parallel inner product, error return + empty identity | W15T4 | 09-parallel §5, §6, §7 |
| W15T6 | `src/parallel/mod.rs` | ParallelPool: custom rayon::ThreadPool wrapper, preserving public API result semantics | W15T2, W15T4, W15T5 | 09-parallel §5, §6, §7 |
| W15T7 | `src/parallel/checked.rs` | Error + panic propagation: XenonError passthrough, panic not swallowed | W15T2, W15T4, W15T5 | 09-parallel §5, §6, §7 |
| W15T8 | `src/parallel/` all files, `tests/test_parallel.rs` | Feature gate + config matrix tests: default off, --features parallel build, single/multi-thread branch validation | W15T1–W15T7 | 09-parallel §5, §6, §7 |

### W16: Math Operations (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W16T1 | `src/math/mod.rs` | Module skeleton: declarations, public API re-exports | None | 11-math §5, §6, §7 |
| W16T2 | `src/math/binary.rs` | Shared binary element-wise execution skeleton with broadcast support | W16T1 | 11-math §5, §6, §7 |
| W16T3 | `src/math/unary.rs` | Unary element-wise ops: abs, neg, signum, square (with checked arithmetic for integers) | W16T1 | 11-math §5, §6, §7 |
| W16T4 | `src/math/unary.rs` | Math functions for RealScalar: sin, sqrt, exp, ln, floor, ceil | W16T1 | 11-math §5, §6, §7 |
| W16T5 | `src/math/unary.rs` | Complex ops: conjugate, modulus | W16T1 | 11-math §5, §6, §7 |
| W16T6 | `src/math/binary.rs` | Arithmetic ops: add, sub, mul, div (scalar variants) via shared binary skeleton | W16T2 | 11-math §5, §6, §7 |
| W16T7 | `src/math/unary.rs` | Logical not for bool tensors | W16T1 | 11-math §5, §6, §7 |
| W16T8 | `src/math/comparison.rs` | Equal + not_equal comparison ops (return bool tensors) | W16T2 | 11-math §5, §6, §7 |
| W16T9 | `src/math/comparison.rs` | Less + less_equal comparison ops (return bool tensors) | W16T8 | 11-math §5, §6, §7 |
| W16T10 | `src/math/comparison.rs` | Greater + greater_equal comparison ops (return bool tensors) | W16T8 | 11-math §5, §6, §7 |
| W16T11 | `src/math/binary.rs`, `unary.rs`, `comparison.rs`, `src/simd/vector.rs` | SIMD backend unified dispatch integration for math module | W16T6, W14 | 11-math §5, §6, §7 |

### W17: Matrix Operations (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W17T1 | `src/matrix/mod.rs` | Module skeleton: sub-module declarations, dot function signatures | None | 12-matrix §5, §6, §7 |
| W17T2 | `src/matrix/dot.rs` | Module skeleton: file placeholder, declarations | W17T1 | 12-matrix §5, §6, §7 |
| W17T3 | `src/matrix/dot.rs` | dot() base execution: rank/shape validation, scalar inner product (real + complex), dispatch skeleton | W17T2 | 12-matrix §5, §6, §7 |
| W17T4 | `src/matrix/dot.rs` | Scalar path consolidation: harden rank/shape validation + wire dispatch serial/parallel decision | W17T3 | 12-matrix §5, §6, §7 |
| W17T5 | `src/matrix/dot.rs`, `src/simd/mod.rs` | SIMD path integration | W17T4, W14 | 12-matrix §5, §6, §7 |
| W17T6 | `src/matrix/dot.rs`, `src/parallel/mod.rs` | Parallel path integration | W17T4, W15 | 12-matrix §5, §6, §7 |
| W17T7 | `tests/test_matrix.rs` | Integration tests | W17T3–W17T6 | 12-matrix §5, §6, §7 |

### W18: Reduction Operations (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W18T1 | `src/reduction/mod.rs` | Module skeleton: public API exports (sum family) | None | 13-reduction §5, §6, §7 |
| W18T2 | `src/reduction/sum.rs` | sum(): full traversal, integer checked arithmetic, empty array zero semantics | W18T1 | 13-reduction §5, §6, §7 |
| W18T3 | `src/reduction/sum.rs` | sum_axis(): axis validation, output shape reduction, per-axis slot accumulation | W18T2 | 13-reduction §5, §6, §7 |
| W18T4 | `src/reduction/sum.rs` | sum_axis_keepdims(): reuse per-axis logic, preserve reduced axis as length 1 | W18T3 | 13-reduction §5, §6, §7 |
| W18T5 | `src/reduction/sum.rs`, `src/simd/`, `src/parallel/` | SIMD / parallel dispatch guards: wire dispatch results, prevent routing semantically invalid inputs | W18T4, W14, W15 | 13-reduction §5, §6, §7 |
| W18T6 | `src/reduction/sum.rs`, `tests/test_reduction.rs` | Error semantic convergence: axis OOB → InvalidAxis, integer overflow → panic | W18T3–W18T5 | 13-reduction §5, §6, §7 |

### W19: Set Operations (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W19T1 | `src/set/mod.rs` | Module skeleton: declarations for unique and public re-exports | None | 01-architecture §3, 14-set §5, §6, §7 |
| W19T2 | `src/set/unique.rs` | Module skeleton: UniqueElement trait definition | W19T1 | 14-set §5, §6, §7 |
| W19T3 | `src/set/unique.rs` | unique(): element collection, equality-based deduplication, Tensor construction | W19T2 | 14-set §5, §6, §7 |
| W19T4 | `src/set/unique.rs` | Float NaN / ±0.0 equality handling: preserve each NaN, treat -0.0 and 0.0 as equal | W19T3 | 14-set §5, §6, §7 |
| W19T5 | `src/set/unique.rs` | Complex component-wise equality: real/imag parts follow respective real semantics, no ordering | W19T3 | 14-set §5, §6, §7 |
| W19T6 | `src/set/unique.rs` | unique() entry method on TensorBase | W19T3–W19T5 | 14-set §5, §6, §7 |

### W20: Shape Operations (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W20T1 | `src/shape/mod.rs` | Module skeleton: sub-module declarations, public exports | None | 16-shape §5, §6, §7 |
| W20T2 | `src/shape/transpose.rs` | Module skeleton: file placeholder, declarations | W20T1 | 16-shape §5, §6, §7 |
| W20T3 | `src/shape/transpose.rs` | transpose(): axis swap, O(1) shape/stride recomputation | W20T2 | 16-shape §5, §6, §7 |
| W20T4 | `tests/test_shape.rs` | Integration tests | W20T3 | 16-shape §5, §6, §7 |

### W21: Indexing (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W21T1 | `src/index/mod.rs` | Module skeleton: declarations for ndindex/access/slice and public re-exports | None | 01-architecture §3, 17-indexing §5, §6, §7 |
| W21T2 | `src/index/ndindex.rs` | NdIndex\<D\> trait + tuple/slice index legality check: rank match, per-axis bounds, offset calculation | W21T1 | 17-indexing §5, §6, §7 |
| W21T3 | `src/index/access.rs` | try_at / get / get_unchecked: unified safe + unsafe read access paths | W21T2 | 17-indexing §5, §6, §7 |
| W21T4 | `src/index/slice.rs` | SliceInfoElem + SliceInfoIndices: inline/dynamic slice descriptor representations | W21T2 | 17-indexing §5, §6, §7 |
| W21T5 | `src/index/access.rs` | try_at_mut / get_mut / get_unchecked_mut: mutable access, gated on StorageMut | W21T3 | 17-indexing §5, §6, §7 |
| W21T6 | `src/index/slice.rs` | slice() shape/stride update + layout recomputation: axis folding, Range→shape/stride, read-only view return | W21T4 | 17-indexing §5, §6, §7 |

### W22: Tensor Construction (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W22T1 | `src/construct/mod.rs`, `src/construct/init.rs` | Module skeleton: sub-module declarations, public exports (declarations only, no impls) | None | 18-construction §5, §6, §7 |
| W22T2 | `src/construct/init.rs` | zeros() constructor | W22T1 | 18-construction §5, §6, §7 |
| W22T3 | `src/construct/init.rs` | ones() constructor | W22T1 | 18-construction §5, §6, §7 |
| W22T4 | `src/construct/eye.rs` | eye(): identity matrix constructor | W22T1 | 18-construction §5, §6, §7 |
| W22T5 | `src/construct/from.rs` | from_shape_vec + from_vec: consume Vec into Owned path, 1D convenience | W22T1 | 18-construction §5, §6, §7 |
| W22T6 | `src/construct/from.rs` | from_shape_slice: copy from slice into Owned storage | W22T5 | 18-construction §5, §6, §7 |
| W22T7 | `src/construct/from.rs` | from_array: fixed-array construction | W22T6 | 18-construction §5, §6, §7 |
| W22T8 | `src/construct/scalar.rs` | from_scalar: zero-dim tensor constructor | W22T1 | 18-construction §5, §6, §7 |
| W22T9 | `tests/test_construction.rs` | Integration tests | W22T2–W22T8 | 18-construction §5, §6, §7 |

### W23: Operator Overloading (L6)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W23T1 | `src/overload/mod.rs` | Module skeleton: declarations for arithmetic and public operator re-exports | None | 01-architecture §3, 19-overload §5, §6, §7 |
| W23T2 | `src/overload/arithmetic.rs` | Module skeleton: declarations, imports | W23T1 | 19-overload §5, §6, §7 |
| W23T3 | `src/overload/arithmetic.rs` | Add\<Tensor, Tensor\> for owned: Tensor + Tensor impl | W23T2 | 19-overload §5, §6, §7 |
| W23T4 | `src/overload/arithmetic.rs` | Add for ref/mixed: &Tensor + &Tensor, Tensor + &Tensor, &Tensor + Tensor (4 combos) | W23T3 | 19-overload §5, §6, §7 |
| W23T5 | `src/overload/arithmetic.rs` | Add with scalar: Tensor + scalar, scalar + Tensor | W23T3 | 19-overload §5, §6, §7 |
| W23T6 | `src/overload/arithmetic.rs` | Sub operators for owned/ref/mixed/scalar tensor combinations | W23T4, W23T5 | 19-overload §5, §6, §7 |
| W23T7 | `src/overload/arithmetic.rs` | Mul operators for owned/ref/mixed/scalar tensor combinations | W23T4, W23T5 | 19-overload §5, §6, §7 |
| W23T8 | `src/overload/arithmetic.rs` | Div operators for owned/ref/mixed/scalar tensor combinations | W23T4, W23T5 | 19-overload §5, §6, §7 |
| W23T9 | `tests/test_overload.rs` | Integration tests: broadcast combos, scalar combos, type combos, deep-copy verification | W23T1–W23T8 | 19-overload §5, §6, §7 |

### W24: Utility Operations (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W24T1 | `src/util/mod.rs` | Module skeleton: declarations for fill/clip/contiguous and public re-exports | None | 20-utility §3, §5, §6, §7 |
| W24T2 | `src/util/fill.rs` | fill(): StorageMut-level fill helper + try_fill() dispatch for all tensor types | W24T1, W7, W8, W12, W2 | 20-utility §5, §6, §7 |
| W24T3 | `src/util/clip.rs` | clip(): element-wise clipping with NaN/min=max/NaN-bound/Integer error handling | W24T1, W4, W8, W12, W2 | 20-utility §5, §6, §7 |
| W24T4 | `src/util/contiguous.rs` | to_contiguous() + into_contiguous(): F-contiguous guarantee, reuse or repack | W24T1, W6, W8, W25 | 20-utility §5, §6, §7 |
| W24T5 | `tests/test_utility.rs` | Integration tests: boundary cases (empty, single-element, non-contiguous, zero-dim) for all utility ops | W24T2–W24T4 | 20-utility §5, §6, §7 |

### W25: Type Conversion (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W25T1 | `src/convert/mod.rs`, `src/lib.rs` | Module skeleton: sub-module declarations, pub use exports | None | 21-type §5, §6, §7 |
| W25T2 | `src/convert/cast.rs` | CastTo trait definition (lossy + dynamic) + ConvertTo shim (lossless) trait signatures only, no impls | W25T1 | 21-type §5, §6, §7 |
| W25T3 | `src/convert/cast.rs` | Tier-1 lossless CastTo impls: int↔int primitive conversions (11 cells) | W25T2 | 21-type §5, §6, §7 |
| W25T4 | `src/convert/cast.rs` | Tier-2 lossy CastTo impls: real↔int, complex↔int, complex↔real (14 cells) | W25T2 | 21-type §5, §6, §7 |
| W25T5 | `src/convert/cast.rs` | Tier-3 dynamic CastTo impls: remaining cross-type conversions (8 cells) | W25T2 | 21-type §5, §6, §7 |
| W25T6 | `src/convert/cast.rs` | cast\<B\>() method on TensorBase\<S, D\>: all readable storage inputs → owned result with error reporting | W25T3, W25T4, W25T5 | 21-type §5, §6, §7 |
| W25T7 | `src/convert/cast.rs` | to_owned() + into_owned(): clone and consume owned conversion methods | W25T1 | 21-type §5, §6, §7 |

### W26: Output Formatting (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W26T1 | `src/format/mod.rs` | Module skeleton: sub-module declarations, re-exports | None | 22-output §5, §6, §7 |
| W26T2 | `src/format/config.rs` | FormatConfig struct + Default impl | W26T1 | 22-output §5, §6, §7 |
| W26T3 | `src/format/pretty.rs` | Numpy-style formatting helpers: fmt_1d_display, fmt_1d_debug, fmt_nd_display, fmt_nd_debug with truncation | W26T2 | 22-output §5, §6, §7 |
| W26T4 | `src/format/display.rs` | core::fmt::Display for TensorBase\<S, D\> | W26T3 | 22-output §5, §6, §7 |
| W26T5 | `src/format/debug.rs` | core::fmt::Debug for TensorBase\<S, D\> with metadata | W26T3 | 22-output §5, §6, §7 |
| W26T6 | `src/format/mod.rs`, `display.rs` | Module docs + re-exports completion | W26T4, W26T5 | 22-output §5, §6, §7 |

### W27: Safety Audit (cross-cutting)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W27T1 | `src/storage/owned.rs` | Audit and strengthen SAFETY comments for Owned\<A\> Send + Sync impls | W7T12 | 25-safety §5, §6, §7 |
| W27T2 | `src/storage/view.rs` | Audit and strengthen SAFETY comments for ViewRepr\<'a, A\> Send + Sync impls | W7T14 | 25-safety §5, §6, §7 |
| W27T3 | `src/storage/viewmut.rs` | Audit and strengthen SAFETY comments for ViewMutRepr\<'a, A\> Send (no Sync) | W7T15 | 25-safety §5, §6, §7 |
| W27T4 | `src/storage/arc.rs` | Audit and strengthen SAFETY comments for ArcRepr\<A\> Send + Sync impls | W7T17 | 25-safety §5, §6, §7 |
| W27T5 | `src/parallel/iter.rs` | Parallel execution chunk safety: completeness, non-overlap, boundary tests | W27T1–W27T4 | 25-safety §5, §6, §7 |
| W27T6 | `tests/test_parallel.rs`, `tests/test_error.rs` | Thread-safety integration tests: cross-thread transfer, concurrent access | W27T1–W27T5 | 25-safety §5, §6, §7 |
| W27T7 | `src/storage/mod.rs` | Module-level thread-safety docs, Send/Sync matrix, cargo doc pass | W27T1–W27T4 | 25-safety §5, §6, §7 |

### W28: Benchmarks

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W28T1 | `Cargo.toml` | Add 9 [[bench]] entries, no new benchmark-specific third-party deps | None | 27-benchmark §5, §6, §7 |
| W28T2 | `benches/utils/mod.rs`, `benches/utils/generators.rs` | Shared constants (SIZES_1D, SIZES_2D) + data generation functions | W28T1 | 27-benchmark §5, §6, §7 |
| W28T3 | `benches/math.rs` | Element-wise math benchmarks: add/sub/mul/div/sin/exp/abs (f32/f64/Complex\<f64\>, contiguous + non-contiguous) | W28T2 | 27-benchmark §5, §6, §7 |
| W28T4 | `benches/reduction.rs` | Reduction benchmarks: sum_1d_f64, sum_2d_axis0, sum_2d_axis1, sum_sliced, sum_2d_keepdims | W28T2 | 27-benchmark §5, §6, §7 |
| W28T5 | `benches/dot.rs` | Dot benchmarks: dot_1d_f64, dot_1d_complex | W28T2 | 27-benchmark §5, §6, §7 |
| W28T6 | `benches/set.rs` | Set benchmarks: unique_1d (varying sizes, unique ratio) | W28T2 | 27-benchmark §5, §6, §7 |
| W28T7 | `benches/broadcast.rs` | Broadcast benchmarks: broadcast_scalar, broadcast_row, broadcast_col, broadcast_with | W28T2 | 27-benchmark §5, §6, §7 |
| W28T8 | `benches/shape.rs` | Shape benchmarks: transpose_2d | W28T2 | 27-benchmark §5, §6, §7 |
| W28T9 | `benches/construction.rs` | Construction benchmarks: zeros_1d, from_shape_vec_1d, eye_2d | W28T2 | 27-benchmark §5, §6, §7 |
| W28T10 | `benches/simd_comparison.rs` | SIMD comparison: add/sum/dot with --features simd on/off | W28T3, W28T4, W28T5, W14 | 27-benchmark §5, §6, §7 |
| W28T11 | `benches/parallel_comparison.rs` | Parallel comparison: sum/add/dot with --features parallel on/off | W28T3, W28T4, W28T5, W15 | 27-benchmark §5, §6, §7 |
| W28T12 | `.github/workflows/bench.yml`, `tools/bench/report.py` | CI bench smoke test + benchmark report script: quick-mode execution, regression annotation | W28T3–W28T11 | 27-benchmark §5, §6, §7 |

### W29: Integration Tests

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W29T1 | `tests/common/mod.rs`, `tests/common/assertions.rs`, `tests/common/generators.rs` | Test infrastructure: assert_tensor_exact_real/complex, real_bits_eq/real_ulp_eq, tolerance helpers, data generators | None | 28-tests §5, §6, §7 |
| W29T2 | `tests/test_tensor.rs` | Core tensor tests: shape/strides/view/to_owned/type_aliases/debug_display/arc | W29T1 | 28-tests §5, §6, §7 |
| W29T3 | `tests/test_math.rs` | Math tests: element-wise arithmetic/math/comparison/logical ops | W29T1 | 28-tests §5, §6, §7 |
| W29T4 | `tests/test_overload.rs` | Overload tests: Add/Sub/Mul/Div traits, broadcast dispatch, Result ownership, scalar operators | W29T1, W29T3 | 28-tests §5, §6, §7 |
| W29T5 | `tests/test_broadcast.rs` | Broadcast tests: scalar/row/col/incompatible/read-only broadcast | W29T1 | 28-tests §5, §6, §7 |
| W29T6 | `tests/test_index.rs` | Index tests: multi-dim indexing, OOB errors, slicing, structural SliceInfo validation | W29T1 | 28-tests §5, §6, §7 |
| W29T7 | `tests/test_construction.rs` | Construction tests: zeros/ones/eye/from_shape_vec/slice/from_scalar/from_array | W29T1 | 28-tests §5, §6, §7 |
| W29T8 | `tests/test_reduction.rs` | Reduction tests: sum/sum_axis/keepdims/empty/NaN/overflow | W29T1 | 28-tests §5, §6, §7 |
| W29T9 | `tests/test_iterator.rs` | Iterator tests: elements/axis/indexed iteration | W29T1 | 28-tests §5, §6, §7 |
| W29T10 | `tests/test_matrix.rs` | Matrix tests: dot/complex/shape mismatch | W29T1 | 28-tests §5, §6, §7 |
| W29T11 | `tests/test_set.rs` | Set tests: unique (int/complex/NaN/±0.0/multiset, output order unspecified) | W29T1 | 28-tests §5, §6, §7 |
| W29T12 | `tests/test_shape.rs` | Shape tests: transpose/high-dim | W29T1 | 28-tests §5, §6, §7 |
| W29T13 | `tests/test_conversion.rs` | Conversion tests: cast/to_owned/into_owned | W29T1 | 28-tests §5, §6, §7 |
| W29T14 | `tests/test_utility.rs` | Utility tests: fill/clip/to_contiguous | W29T1 | 28-tests §5, §6, §7 |
| W29T15 | `tests/test_output.rs` | Output tests: Display/Debug/truncation/complex (Numpy-style) | W29T1 | 28-tests §5, §6, §7 |
| W29T16 | `tests/test_error.rs` | Error tests: XenonError boundary + display output validation, Workspace structured fields | W29T1 | 28-tests §5, §6, §7 |
| W29T17 | `tests/test_workspace.rs` | Workspace tests: illegal alignment/borrow guard/split/expand/assume_init/!Send+!Sync | W29T1, W29T2 | 28-tests §5, §6, §7 |
| W29T18 | `tests/test_ffi.rs` | FFI tests: pointer/BLAS compatibility/export/export_mut/offset | W29T2 | 28-tests §5, §6, §7 |
| W29T19 | `tests/test_parallel.rs` | Parallel tests: sum/add behavioral consistency with parallel feature, concurrent read, nested prohibition | W29T3, W29T8 | 28-tests §5, §6, §7 |
| W29T20 | `tests/test_simd.rs` | SIMD tests: result consistency (add/sum/fallback) | W29T3, W29T8 | 28-tests §5, §6, §7 |
| W29T21 | `.github/workflows/test.yml` | CI test matrix: maintain std-environment lib/tests/doctest matrix | W29T2 | 28-tests §5, §6, §7 |
| W29T22 | `tests/property_tests.rs`, `property/tensor_props.rs`, `ops_props.rs`, `shape_props.rs` | Property tests: transpose involution, addition commutativity, unique no-duplicates, sum preserves identity, broadcast shape consistency | W29T3, W29T8, W29T12 | 28-tests §5, §6, §7 |
| W29T23 | `.github/workflows/test.yml` | CI test matrix full config: all feature combos and property tests | W29T1–W29T22 | 28-tests §5, §6, §7 |
| W29T24 | `tests/compile_fail_tests.rs` | Compile-fail harness using trybuild-style fixtures and assertion conventions | W29T1 | 28-tests §5, §6, §7 |
| W29T25 | `tests/compile-fail/*.rs` | Compile-fail fixtures: wrong dimension, missing element bound, storage mismatch, unsigned/bool/scalar rejection cases | W29T24 | 28-tests §5, §6, §7 |

### W30: Documentation

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W30T1 | `src/lib.rs` | Crate-level docs: project overview, Quick Start, Features table, element types table, memory layout | None | 29-documentation §5, §6, §7 |
| W30T2 | `src/lib.rs`, `Cargo.toml` | #![warn(missing_docs)] lint + docs.rs metadata: all-features = true | W30T1 | 29-documentation §5, §6, §7 |
| W30T3 | `README.md` | Project README: intro, features, Quick Start, install, doc links, license | W30T1 | 29-documentation §5, §6, §7 |
| W30T4 | `CHANGELOG.md` | Optional CHANGELOG.md in Keep a Changelog format | None | 29-documentation §5, §6, §7 |
| W30T5 | `src/dimension/mod.rs` | Dimension module-level docs: responsibilities, core concepts, Ix0–Ix6 usage, Dimension trait guide | W30T2 | 29-documentation §5, §6, §7 |
| W30T6 | `src/element/mod.rs` | Element module-level docs: trait hierarchy, 7 element types, sealed design rationale | W30T2 | 29-documentation §5, §6, §7 |
| W30T7 | `src/complex/mod.rs` | Complex module-level docs: repr(C) layout, FFI guarantees, arithmetic semantics | W30T2 | 29-documentation §5, §6, §7 |
| W30T8 | `src/storage/mod.rs` | Storage module-level docs: trait hierarchy, Owned/ViewRepr/ArcRepr, alignment guarantees | W30T2 | 29-documentation §5, §6, §7 |
| W30T9 | `src/layout/mod.rs` | Layout module-level docs: F-order conventions, LayoutFlags, stride computation | W30T2 | 29-documentation §5, §6, §7 |
| W30T10 | `src/tensor/mod.rs` | Tensor module-level docs: TensorBase, type aliases, storage kinds, memory model | W30T2 | 29-documentation §5, §6, §7 |
| W30T11 | `src/iter/mod.rs` | Iterator module-level docs: Iter/IterMut, AxisIter, IndexedIter, entry methods | W30T2 | 29-documentation §5, §6, §7 |
| W30T12 | `src/math/mod.rs` | Math module-level docs: binary/unary ops, comparison ops, SIMD dispatch | W30T2 | 29-documentation §5, §6, §7 |
| W30T13 | `src/overload/mod.rs` | Overload module-level docs: operator traits, broadcast dispatch, ownership rules | W30T2 | 29-documentation §5, §6, §7 |
| W30T14 | `src/broadcast/mod.rs` | Broadcast module-level docs: dimension rules, zero-stride semantics, read-only boundary | W30T2 | 29-documentation §5, §6, §7 |
| W30T15 | `src/reduction/mod.rs` | Reduction module-level docs: sum/sum_axis/keepdims, SIMD/parallel dispatch gates | W30T2 | 29-documentation §5, §6, §7 |
| W30T16 | `src/matrix/mod.rs` | Matrix module-level docs: dot product, rank validation, complex support | W30T2 | 29-documentation §5, §6, §7 |
| W30T17 | `src/shape/mod.rs` | Shape module-level docs: transpose, O(1) semantics, F-order contiguity | W30T2 | 29-documentation §5, §6, §7 |
| W30T18 | `src/index/mod.rs` | Index module-level docs: NdIndex trait, multi-dim access, slicing, bounds checking | W30T2 | 29-documentation §5, §6, §7 |
| W30T19 | `src/construct/mod.rs` | Construct module-level docs: initialization methods, ownership guarantees, error semantics | W30T2 | 29-documentation §5, §6, §7 |
| W30T20 | `src/set/mod.rs` | Set module-level docs: unique, NaN/±0.0 handling, complex equality | W30T2 | 29-documentation §5, §6, §7 |
| W30T21 | `src/ffi/mod.rs` | FFI module-level docs: pointer access, BLAS compatibility, C repr guarantees | W30T2 | 29-documentation §5, §6, §7 |
| W30T22 | `src/workspace/mod.rs` | Workspace module-level docs: allocation strategy, borrow guards, split semantics | W30T2 | 29-documentation §5, §6, §7 |
| W30T23 | `src/error.rs` | Error module-level docs: XenonError variants, Result alias, error categories | W30T2 | 29-documentation §5, §6, §7 |
| W30T24 | `src/convert/mod.rs` | Convert module-level docs: CastTo/ConvertTo traits, lossy/lossless semantics, tiered conversion | W30T2 | 29-documentation §5, §6, §7 |
| W30T25 | `src/format/mod.rs` | Format module-level docs: NumPy-style output, FormatConfig, truncation helpers | W30T2 | 29-documentation §5, §6, §7 |
| W30T26 | `src/util/mod.rs` | Util module-level docs: responsibility overview, utility function category notes | W30T2 | 29-documentation §5, §6, §7 |
| W30T27 | `src/simd/mod.rs` | SIMD module-level docs: SimdKernel trait, kernel wiring, feature gate notes | W30T2 | 29-documentation §5, §6, §7 |
| W30T28 | `src/parallel/mod.rs` | Parallel module-level docs: ParIter, map/reduce patterns, thread-safety, feature gate | W30T2 | 29-documentation §5, §6, §7 |
| W30T29 | `src/tensor/mod.rs` + related files | Tensor type-level docs + doctests: TensorBase, Tensor, TensorView, TensorViewMut, ArcTensor | W30T5–W30T28 | 29-documentation §5, §6, §7 |
| W30T30 | `src/dimension/mod.rs` | Dimension type-level docs + doctests: Ix0–Ix6, IxDyn, Dimension trait | W30T5–W30T28 | 29-documentation §5, §6, §7 |
| W30T31 | `src/element/mod.rs` | Element type-level docs + doctests: Element, Numeric, RealScalar, ComplexScalar traits | W30T5–W30T28 | 29-documentation §5, §6, §7 |
| W30T32 | `src/storage/mod.rs` | Storage type-level docs + doctests: Owned, ViewRepr, StorageMut trait | W30T5–W30T28 | 29-documentation §5, §6, §7 |
| W30T33 | `src/layout/mod.rs` | Layout type-level docs + doctests: LayoutFlags, compute_f_strides | W30T5–W30T28 | 29-documentation §5, §6, §7 |
| W30T34 | `src/math/` files | Math function docs + doctests: add, sub, mul, div, sin, sqrt, exp, ln, abs | W30T5–W30T28 | 29-documentation §5, §6, §7 |
| W30T35 | `src/reduction/` | Reduction docs + doctests: sum, sum_axis | W30T5–W30T28 | 29-documentation §5, §6, §7 |
| W30T36 | `src/matrix/` | Matrix docs + doctests: dot | W30T5–W30T28 | 29-documentation §5, §6, §7 |
| W30T37 | `src/broadcast/`, `src/shape/mod.rs` | Broadcast + shape docs + doctests: broadcast_shape, transpose | W30T5–W30T28 | 29-documentation §5, §6, §7 |
| W30T38 | `src/construct/mod.rs`, `src/set/mod.rs` | Construct + set docs + doctests: zeros, ones, eye, from_shape_vec, unique + error semantics | W30T5–W30T28 | 29-documentation §5, §6, §7 |
| W30T39 | `src/ffi/mod.rs`, `src/workspace/mod.rs`, `src/error.rs` | FFI + workspace + error docs + doctests | W30T5–W30T28 | 29-documentation §5, §6, §7 |
| W30T40 | `src/iter/mod.rs`, `src/convert/mod.rs`, `src/format/mod.rs`, `src/overload/mod.rs` | Iter + convert + format + overload module docs + doctests | W30T5–W30T28 | 29-documentation §5, §6, §7 |
| W30T41 | `src/index/mod.rs` | Index function-level docs + doctests | W30T5–W30T28 | 29-documentation §5, §6, §7 |
| W30T42 | `src/util/mod.rs` | Util function-level docs + doctests: clip, fill, try_fill, to_contiguous, into_contiguous | W30T26 | 29-documentation §5, §6, §7 |
| W30T43 | `examples/basic.rs` | Usage example: create, operate, reduce, print | W30T1 | 29-documentation §5, §6, §7 |
| W30T44 | `examples/complex.rs` | Usage example: complex construction, same-type arithmetic, explicit conversion ops | W30T1 | 29-documentation §5, §6, §7 |
| W30T45 | `examples/broadcasting.rs` | Usage example: broadcast rules, row/col/scalar broadcast | W30T1 | 29-documentation §5, §6, §7 |
| W30T46 | `examples/features.rs` | Feature-gated example: conditional compile with simd/parallel features | W30T1 | 29-documentation §5, §6, §7 |
| W30T47 | `examples/simd.rs` | Usage example: SIMD feature-gated execution and fallback behavior | W30T1 | 29-documentation §5, §6, §7 |
| W30T48 | `examples/ffi.rs` | Usage example: FFI export, pointer access, BLAS layout checks | W30T1 | 29-documentation §5, §6, §7 |
| W30T49 | `examples/workspace.rs` | Usage example: workspace allocation and borrow/split workflow | W30T1 | 29-documentation §5, §6, §7 |
| W30T50 | `src/lib.rs`, `README.md`, `examples/` | Audit that examples and crate docs only declare `std` environment; remove out-of-scope platform notes | W30T1, W30T3, W30T43–W30T49 | 29-documentation §5, §6, §7 |
| W30T51 | `LICENSE` | Project license file matching Cargo.toml package metadata | W1T1 | 01-architecture §3, §4 |
| W30T52 | `.github/workflows/docs.yml` | docs.rs CI integration, missing-docs check, doctest and example compilation | W30T40, W30T43–W30T49 | 29-documentation §5, §6, §7 |

---

## Dependency Graph (Simplified)

```
W1 (Coding/Setup) ──┬──→ W2 (Error)
                    ├──→ W3 (Dimension) ──→ W6 (Layout)
                    └──→ W5 (Complex)

W3 + W5 ──→ W4 (Element)
W6 + W3 ──→ W7 (Storage)
W2 ──→ W9 (Workspace, L2)
W3 + W4 + W6 + W7 ──→ W8 (Tensor Core)

W8 + W6 + W2 ─→ W10 (Dispatch)
W8 + W6 + W3 + W2 ─→ W11 (Broadcast)
W8 + W7 + W3 + W2 ─→ W12 (Iterators)
W8 + W6 + W3 + W4 + W2 ─→ W13 (FFI)
W8 + W6 + W4 + W5 ─→ W14 (SIMD)
W8 + W4 + W10 ─→ W15 (Parallel)

W8 + W4 + W11 + W12 + W14 + W15 ─→ W16 (Math)
W8 + W14 + W15 ─→ W17 (Matrix)
W8 + W14 + W15 ─→ W18 (Reduction)
W8 + W12 ─→ W19 (Set)
W8 + W6 ─→ W20 (Shape)
W8 + W3 ─→ W21 (Indexing)
W8 ─→ W22 (Construction)
W8 + W11 + W16 ─→ W23 (Overload)
W8 ─→ W24 (Utility)
W4 + W5 + W8 ─→ W25 (Type Conversion)
W8 ─→ W26 (Output Formatting)

W1–W26 ─→ W27 (Safety Audit)
W22 ─→ W28 (Benchmarks)
W22 ─→ W29 (Integration Tests)
W22 ─→ W30 (Documentation)
```

---

## Key Design Constraints (All Waves Must Follow)

1. **F-order only** — Column-major layout, no C-order support
2. **Single crate** — `xenon`, MSRV Rust 1.85+, edition 2024
3. **7 element types** — i32, i64, f32, f64, Complex\<f32\>, Complex\<f64\>, bool
4. **Sealed traits** — No external implementation of Element, Dimension, etc.
5. **Optional deps only** — rayon (parallel) and pulp (simd) via feature gates
6. **64-byte alignment** — Owned storage uses 64-byte aligned allocator
7. **Unified error model** — All recoverable errors via `XenonError` + `Result<T>`
8. **No negative strides** — Not supported in current version
9. **Zero-stride only on broadcast read-only views** — Broadcast zero-stride restricted to read-only/shared-read-only views
10. **ZST and empty arrays must not cause UB** — All operations on zero-size types and zero-element arrays must be safe
11. **NumPy-style output** — Display/Debug formatting mimics NumPy conventions with configurable truncation
12. **IEEE 754 compliance** — Floating-point special values (NaN, Inf) handled per standard semantics
