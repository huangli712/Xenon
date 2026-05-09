# Xenon Implementation Plan — Wave & Task Summary

> Project: Xenon N-dimensional array library (Rust)
> Dependency Layers: L0 → L1 → L2 → L4 → L5 → L6 → Cross-cutting
> Task Granularity: Each task targets 1 function / 1 trait / 1 type, ~5–10 min, max 1 file

---

## Wave Overview

| Wave | Name | Layer | Task Count | Description |
|------|------|-------|------------|-------------|
| W1 | Coding Standards & Project Setup | L0 | 4 | rustfmt.toml, lib.rs skeleton with lint attrs, .clippy.toml, CI config |
| W2 | Error System | L0 | 5 | XenonError enum, Result alias, Display/Error impls, auxiliary enums, prelude exports |
| W3 | Dimension System | L1 | 12 | Static dims (Ix0–Ix6), IxDyn, Dimension/IntoDimension/RemoveAxis traits, Axis, Sealed |
| W4 | Element Type Hierarchy | L1 | 12 | Element/Numeric/RealScalar/ComplexScalar traits, sealed, primitives impls, integration |
| W5 | Complex Type | L1 | 11 | Complex\<T\> struct, arithmetic ops (Add/Sub/Mul/Div), Display/Debug, math methods, FFI layout, convert |
| W6 | Layout System | L2 | 7 | LayoutFlags bitflags, F-order stride computation, contiguity checks, alignment, zero-stride detection |
| W7 | Storage System | L2 | 14 | RawStorage/Storage/StorageMut/RawStorageMut/StorageOwned/StorageShared traits, Owned/A, AlignedAlloc, ViewRepr, ViewMutRepr, ArcRepr |
| W8 | Tensor Core | L3 | 10 | TensorBase\<S,D\>, type aliases (Tensor/TensorView/TensorViewMut/ArcTensor + dim convensions), constructors, view methods, accessors, from_raw_parts |
| W9 | Workspace | L2 | 7 | Workspace struct, borrow guards (WorkspaceBorrow/WorkspaceBorrowMut), split (SplitBorrowMut), expand (ensure_capacity/reallocate), docs |
| W10 | Dispatch | L4 | 6 | ExecPath enum, select_exec_path, thresholds, ParallelGuard (nested parallel protection), ParallelExecStrategy |
| W11 | Broadcasting | L4 | 8 | can_broadcast, broadcast_shape, broadcast_strides, broadcast_to, broadcast_with, error handling, integration tests |
| W12 | Iterators | L4 | 6 | StrideState, Elements (flat Iter/IterMut), AxisIter/AxisIterMut, IndexedIter/IndexedIterMut, TensorBase entry methods |
| W13 | FFI Helpers | L4 | 4 | BlasInfo, ptr re-exports (as_ptr/as_mut_ptr/from_raw_parts), export/export_mut, is_blas_compatible/lda, try_offset_of/try_ptr_at |
| W14 | SIMD Backend | L5 | 10 | SimdKernel trait, element-wise SIMD (add/sub/mul/div), SIMD sum (float/int/complex), SIMD dot, feature gates, property tests |
| W15 | Parallel Backend | L5 | 8 | ParIter, par_map, par_zip_map, par_sum, par_dot, ParallelPool, error/panic propagation, feature gates |
| W16 | Math Operations | L5 | 8 | Binary element-wise ops (add/sub/mul/div), unary ops (abs/neg/signum/square/sin/sqrt/exp/ln/floor/ceil/conj), comparison ops (eq/ne/lt/le/gt/ge), SIMD dispatch |
| W17 | Matrix Operations | L5 | 6 | dot product (serial + SIMD + parallel paths), rank/shape validation, complex dot, integration tests |
| W18 | Reduction Operations | L5 | 6 | sum (global), sum_axis, sum_axis_keepdims, SIMD/parallel dispatch gates, error convergence |
| W19 | Set Operations | L5 | 5 | unique (real/complex/NaN/signed-zero handling), TensorBase entry method |
| W20 | Shape Operations | L5 | 3 | transpose, axis swap, contiguity recomputation, integration tests |
| W21 | Indexing | L5 | 5 | NdIndex trait, try_at/get/get_unchecked, SliceInfo, try_at_mut/get_mut/get_unchecked_mut, slice shape/stride update |
| W22 | Tensor Construction | L5 | 5 | zeros/ones, eye, from_shape_vec/from_shape_slice, from_array/from_scalar |
| W23 | Operator Overloading | L6 | 6 | Add/Sub/Mul/Div for owned/ref/mixed/scalar tensor combinations, integration tests |
| W24 | Utility Operations | L5 | 4 | fill, clip, to_contiguous/into_contiguous |
| W25 | Type Conversion | L5 | 5 | CastTo trait (lossy + dynamic tiers), ConvertTo (lossless), cast method, to_owned/into_owned |
| W26 | Output Formatting | L5 | 5 | FormatConfig, Display (Numpy-style), Debug (with metadata), pretty formatting helpers |
| W27 | Safety Audit | cross-cutting | 7 | Send/Sync impls for Owned/ViewRepr/ViewMutRepr/ArcRepr, parallel chunk safety, thread-safety integration tests |
| W28 | Benchmarks | — | 12 | bench infrastructure (utils/generators), core benches (math/reduction/dot/set/broadcast), shape/construction benches, SIMD/parallel comparison, CI |
| W29 | Integration Tests | — | 23 | tests/common utils, core test files (tensor/math/overload/broadcast/index/construction/reduction/iter/matrix/set/shape/conversion/utility/output/error), specialized tests (workspace/ffi/parallel/simd), property tests, CI matrix |
| W30 | Documentation | — | 26 | Crate-level docs, module-level docs, type/function-level docs, usage examples (basic/complex/broadcasting/features/simd/ffi/workspace), CI docs.rs integration |
| | **Total** | | **250** | |

---

## Detailed Task List

### W1: Coding Standards & Project Setup (L0)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W1T1 | `rustfmt.toml` | Rustfmt configuration per §3.2 coding standard | None | 00-coding §12 |
| W1T2 | `src/lib.rs` | Crate root skeleton with lint declarations (missing_docs, unsafe_op_in_unsafe_fn, clippy::unwrap_used) | None | 00-coding §12 |
| W1T3 | `.clippy.toml` | Clippy configuration for numeric `as` casts lint, unwrap restrictions | None | 00-coding §12 |
| W1T4 | `.github/workflows/ci.yml` | CI config: fmt + clippy + test matrix across feature combinations | W1T2 | 00-coding §12 |

### W2: Error System (L0)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W2T1 | `src/error.rs` | XenonError enum definition with all structured variants | None | 26-error §7 |
| W2T2 | `src/error.rs` | Auxiliary enum types: FfiErrorCategory, WorkspaceErrorCategory, ConversionFailureReason + Result alias | W2T1 | 26-error §7 |
| W2T3 | `src/error.rs` | fmt::Display impl for XenonError with OrAny\<T\>, FmtShape helpers | W2T2 | 26-error §7 |
| W2T4 | `src/error.rs` | std::error::Error impl (source chaining for Ffi/Workspace, None for leaf variants) | W2T2 | 26-error §7 |
| W2T5 | `src/error.rs`, `src/lib.rs` | Public exports of XenonError, Result alias, auxiliary enums via prelude | W2T4 | 26-error §7 |

### W3: Dimension System (L1)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W3T1 | `src/dimension/mod.rs` | Module skeleton: sub-module declarations, public re-exports | None | 02-dimension §7 |
| W3T2 | `src/dimension/mod.rs` | Dimension trait definition (all method signatures, MAX_DIMENSION constant) | W3T1 | 02-dimension §7 |
| W3T3 | `src/dimension/static.rs` | Ix0 zero-dimensional scalar with Dimension impl | W3T2 | 02-dimension §7 |
| W3T4 | `src/dimension/static.rs` | Ix1, Ix2 structs with Dimension impl + Index\<usize\> | W3T3 | 02-dimension §7 |
| W3T5 | `src/dimension/static.rs` | Ix3–Ix6 structs with Dimension impl + From\<tuple\> | W3T4 | 02-dimension §7 |
| W3T6 | `src/dimension/dynamic.rs` | IxDyn dynamic dimension with Dimension impl + constructors | W3T2 | 02-dimension §7 |
| W3T7 | `src/dimension/static.rs`, `dynamic.rs`, `src/error.rs` | into_dyn() / try_from_dyn() + XenonError::DimensionMismatch | W3T5, W3T6 | 02-dimension §7 |
| W3T8 | `src/dimension/into.rs` | IntoDimension trait + tuple/array/slice/Vec impls | W3T7 | 02-dimension §7 |
| W3T9 | `src/dimension/axes.rs` | Axis newtype with new/index/checked_next/next/prev/is_first/is_last | W3T1 | 02-dimension §7 |
| W3T10 | `src/private.rs`, `src/dimension/mod.rs` | Sealed trait impl for all dimension types + public exports | W3T7, W3T8, W3T9 | 02-dimension §7 |
| W3T11 | All `src/dimension/` files | Doc comments on all pub items, cargo doc verification | W3T10 | 02-dimension §7 |
| W3T12 | `tests/test_dimension.rs` | Integration and boundary tests for full dimension system | W3T10 | 02-dimension §7 |

### W4: Element Type Hierarchy (L1)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W4T1 | `src/element/mod.rs` | Module skeleton + import shared Sealed + Element trait definition | None | 03-element §7 |
| W4T2 | `src/element/numeric.rs` | Numeric trait definition (arithmetic supertraits + conjugate) | W4T1 | 03-element §7 |
| W4T3 | `src/element/real.rs` | RealScalar trait: math functions (abs/sqrt/sin/exp/ln/floor/ceil) + NaN detection | W4T2 | 03-element §7 |
| W4T4 | `src/element/complex.rs` | ComplexScalar trait: associated type Real + complex methods (re/im/norm) | W4T2 | 03-element §7 |
| W4T5 | `src/element/primitives.rs` | Element + Numeric impls for i32, i64 | W4T2 | 03-element §7 |
| W4T6 | `src/element/primitives.rs` | Element + Numeric + RealScalar impls for f32, f64 | W4T3, W4T5 | 03-element §7 |
| W4T7 | `src/element/primitives.rs` | Element impl for bool (zero=false, one=true, no Numeric) | W4T1 | 03-element §7 |
| W4T8 | `src/element/mod.rs` | Documentation clarifying usize as index/shape metadata only, not an element type | W4T1 | 03-element §7 |
| W4T9 | `src/element/primitives.rs` | Element + Numeric + ComplexScalar impls for Complex\<f32\>/Complex\<f64\> | W4T4, W4T5 | 03-element §7 |
| W4T10 | `src/element/real.rs`, `src/element/mod.rs` | Calibrate math capability boundaries + document lossy CastTo error semantics | W4T6 | 03-element §7 |
| W4T11 | All `src/element/` files | Doc comments on all pub items, cargo doc verification | W4T9 | 03-element §7 |
| W4T12 | Element unit/doctest, cross-module via tests/test_tensor/test_math/test_reduction/test_conversion | Integration tests for element trait system | W4T9 | 03-element §7 |

### W5: Complex Type (L1)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W5T1 | `src/complex/mod.rs` | Complex\<T\> struct (repr(C)), new(), ComplexFloat trait, f32/f64 impls | None | 04-complex §7 |
| W5T2 | `src/complex/mod.rs` | const size/align assertions for FFI layout guarantees | W5T1 | 04-complex §7 |
| W5T3 | `src/complex/mod.rs` | Basic accessors and constructors: re(), im(), from_imag(), conj(), is_real(), is_imaginary(), From\<T\> | W5T1 | 04-complex §7 |
| W5T4 | `src/complex/mod.rs` | PartialEq (NaN!=NaN) + Display (a+bj format) impls | W5T1 | 04-complex §7 |
| W5T5 | `src/complex/ops.rs` | Complex ± Complex operators (Add, Sub) | W5T1 | 04-complex §7 |
| W5T6 | `src/complex/ops.rs` | Complex × Complex, Complex ÷ Complex, Neg operators | W5T1 | 04-complex §7 |
| W5T7 | `src/complex/ops.rs` | Tighten real-complex mixed arithmetic boundary (Complex op Complex only, explicit conversion) | W5T5, W5T6 | 04-complex §7 |
| W5T8 | `src/complex/mod.rs` | Math methods: norm (hypot), norm_sqr | W5T1 | 04-complex §7 |
| W5T9 | `src/convert/cast.rs` | Complex type conversion impls (From f32/f64, widening, narrowing) | W5T1 | 04-complex §7 |
| W5T10 | All `src/complex/` files, `src/convert/cast.rs` | Doc comments on all pub items, cargo doc verification | W5T8, W5T9 | 04-complex §7 |
| W5T11 | `tests/test_complex.rs` | Integration and boundary tests for full complex type system | W5T10 | 04-complex §7 |

### W6: Layout System (L2)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W6T1 | `src/layout/mod.rs`, `flags.rs`, `strides.rs`, `contiguous.rs` | Module skeleton: declarations, sub-module placeholders, public exports | None | 06-layout §7 |
| W6T2 | `src/layout/flags.rs` | LayoutFlags(u8) bitflags: F_CONTIGUOUS, ALIGNED, HAS_ZERO_STRIDE + query/set methods | W6T1 | 06-layout §7 |
| W6T3 | `src/layout/strides.rs` | compute_f_strides\<D\>: F-order stride computation with overflow check returning Result | W6T1 | 06-layout §7 |
| W6T4 | `src/layout/contiguous.rs` | is_f_contiguous\<D\>: F-order contiguity detection | W6T1 | 06-layout §7 |
| W6T5 | `src/layout/strides.rs` | has_zero_stride: raw zero-stride detector (flag assignment checks product(shape) > 0) | W6T1 | 06-layout §7 |
| W6T6 | `src/layout/strides.rs` | is_aligned_to / is_aligned alignment check functions | W6T1 | 06-layout §7 |
| W6T7 | Layout unit/integration tests (coord with tensor/shape/index/ffi/simd tests) | Comprehensive verification: strides, contiguity, zero-stride, alignment | W6T3–W6T6 | 06-layout §7 |

### W7: Storage System (L2)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W7T1 | `src/storage/mod.rs` | Module skeleton: sub-module declarations, public exports | None | 05-storage §7 |
| W7T2 | `src/storage/mod.rs` | unsafe trait RawStorage: as_ptr/len/is_empty/is_aligned_to/is_aligned | W7T1 | 05-storage §7 |
| W7T3 | `src/storage/mod.rs` | unsafe trait Storage: RawStorage + get/get_unchecked/as_slice | W7T2 | 05-storage §7 |
| W7T4 | `src/storage/mod.rs` | RawStorageMut and StorageMut trait definitions | W7T3 | 05-storage §7 |
| W7T5 | `src/storage/mod.rs` | StorageOwned and StorageShared trait definitions | W7T4 | 05-storage §7 |
| W7T6 | `src/storage/alloc.rs` | AlignedAlloc struct: 64-byte aligned alloc/alloc_zeroed/dealloc | W7T1 | 05-storage §7 |
| W7T7 | `src/storage/owned.rs` | Owned\<A\> struct + new/with_capacity/from_vec/from_vec_aligned/zeros/from_elem constructors | W7T6 | 05-storage §7 |
| W7T8 | `src/storage/owned.rs` | Owned\<A\> all trait impls (RawStorage/Storage/StorageMut/StorageOwned) + into_shared/Send/Sync/From/Default | W7T5, W7T7 | 05-storage §7 |
| W7T9 | `src/storage/view.rs` | ViewRepr\<'a, A\>: struct, from_raw_parts/from_slice/view/slice, Clone/Copy, RawStorage/Storage impls | W7T5 | 05-storage §7 |
| W7T10 | `src/storage/viewmut.rs` | ViewMutRepr\<'a, A\>: struct, from_raw_parts_mut/from_mut_slice/view_mut/view, no Clone, all trait impls | W7T5 | 05-storage §7 |
| W7T11 | `src/storage/arc.rs` | ArcRepr\<A\>: struct, from_vec/from_vec_aligned/zeros/from_elem constructors, Clone (ref-count bump), all trait impls | W7T5 | 05-storage §7 |
| W7T12 | `src/storage/arc.rs` | ArcRepr\<A\> Send/Sync + Default + From impls | W7T11 | 05-storage §7 |
| W7T13 | `src/storage/mod.rs` | Module re-exports (all types + traits) + doc comments | W7T8, W7T9, W7T10, W7T12 | 05-storage §7 |
| W7T14 | `tests/test_storage.rs` | Integration tests: alloc alignment, owned/view/viewmut/arc trait semantics, Send/Sync | W7T13 | 05-storage §7 |

### W8: Tensor Core (L3)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W8T1 | `src/tensor/mod.rs` | Module skeleton: sub-module declarations, public exports | None | 07-tensor §7 |
| W8T2 | `src/tensor/mod.rs` | TensorBase\<S, D\> struct: 6 fields (storage, shape, strides, offset, flags, derived_from_view_mut) | W8T1 | 07-tensor §7 |
| W8T3 | `src/tensor/aliases.rs` | 4 primary type aliases (Tensor/TensorView/TensorViewMut/ArcTensor) + 4×8=32 dim convenience aliases | W8T2 | 07-tensor §7 |
| W8T4 | `src/tensor/impls.rs` | Shape & stride query methods: shape/strides/ndim/len/is_empty/offset/raw_dim/flags/storage_kind/access_semantics/data_location | W8T2 | 07-tensor §7 |
| W8T5 | `src/tensor/impls.rs` | Layout query delegation: layout_state/is_f_contiguous/is_aligned/has_zero_stride | W8T4 | 07-tensor §7 |
| W8T6 | `src/tensor/impls.rs` | Pointer access & slice: as_ptr/as_storage_ptr/as_mut_ptr/as_slice/as_mut_slice | W8T4 | 07-tensor §7 |
| W8T7 | `src/tensor/construct.rs` | from_raw_parts / from_raw_parts_mut with storage_len and validate_access_range | W8T2 | 07-tensor §7 |
| W8T8 | `src/tensor/construct.rs` | from_raw_vec_unchecked (pub(crate) unsafe internal constructor) | W8T5, W8T7 | 07-tensor §7 |
| W8T9 | `src/tensor/impls.rs` | View creation methods: view() / view_mut() | W8T6 | 07-tensor §7 |
| W8T10 | `tests/test_tensor.rs` | Integration tests: cross-module interaction, boundary tests, type alias compilation | W8T3, W8T9 | 07-tensor §7 |

### W9: Workspace (L2)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W9T1 | `src/error.rs` | WorkspaceErrorCategory definition, wired into XenonError::Workspace | None | 24-workspace §7 |
| W9T2 | `src/workspace/workspace.rs` | Workspace struct, constants, new(), with_default_capacity(), Drop | W9T1 | 24-workspace §7 |
| W9T3 | `src/workspace/mod.rs` | Module root: sub-module declarations, re-exports | W9T1 | 24-workspace §7 |
| W9T4 | `src/workspace/borrow.rs` | WorkspaceBorrow/WorkspaceBorrowMut guards + borrow/borrow_mut + MaybeUninit access methods + Drop | W9T2 | 24-workspace §7 |
| W9T5 | `src/workspace/split.rs` | SplitBorrowMut guard, split_at_mut (top-level + recursive), Drop | W9T2 | 24-workspace §7 |
| W9T6 | `src/workspace/expand.rs` | ensure_capacity() / reallocate() expansion strategy | W9T2 | 24-workspace §7 |
| W9T7 | `src/workspace/mod.rs` + sub-modules | Complete public exports + doc comments + cargo doc verification | W9T4, W9T5, W9T6 | 24-workspace §7 |

### W10: Dispatch (L4)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W10T1 | `src/dispatch.rs` | Module skeleton: ExecPath enum, ParallelExecStrategy struct, module docs | None | 30-dispatch §7 |
| W10T2 | `src/dispatch.rs` | ParallelGuard type + try_acquire_guard(): thread_local Cell\<bool\>, Drop impl | W10T1 | 30-dispatch §7 |
| W10T3 | `src/dispatch.rs` | ParallelExecStrategy::new() validating constructor, auto() infallible default, field accessors | W10T1 | 30-dispatch §7 |
| W10T4 | `src/dispatch.rs` | select_exec_path() + should_parallelize(): three-way dispatch, threshold reading, feature gate branching | W10T1, W10T2 | 30-dispatch §7 |
| W10T5 | `src/dispatch.rs` | Threshold config: AtomicUsize storage, set/reset_parallel_threshold, Relaxed ordering | W10T4 | 30-dispatch §7 |
| W10T6 | `src/dispatch.rs` (#[cfg(test)]) | Full dispatch unit tests: path return, threshold boundaries, feature gate combos, nested guard, non-contiguous penalty | W10T4, W10T5, W10T2, W10T3 | 30-dispatch §7 |

### W11: Broadcasting (L4)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W11T1 | `src/broadcast/mod.rs`, `shape.rs`, `view.rs` | Module skeleton: declarations, rule function stubs, view entry placeholders | None | 15-broadcast §7 |
| W11T2 | `src/broadcast/shape.rs` | can_broadcast(): trailing-axis alignment compatibility check | W11T1 | 15-broadcast §7 |
| W11T3 | `src/broadcast/shape.rs` | broadcast_shape(): shared shape derivation with structured broadcast errors | W11T2 | 15-broadcast §7 |
| W11T4 | `src/broadcast/shape.rs` | broadcast_strides(): zero-stride insertion with input precondition validation | W11T3 | 15-broadcast §7 |
| W11T5 | `src/broadcast/view.rs` | broadcast_to() basic path: target shape validation + read-only view construction | W11T4 | 15-broadcast §7 |
| W11T6 | `src/broadcast/view.rs` | broadcast_to() error path + BroadcastView layout state update | W11T5 | 15-broadcast §7 |
| W11T7 | `src/broadcast/view.rs` | broadcast_with(): shared shape derivation, dual-input broadcast, BroadcastDim output type alignment | W11T4, W11T5 | 15-broadcast §7 |
| W11T8 | `tests/test_broadcast.rs` | Unit + integration tests: compatibility rules, zero-stride semantics, shared read-only boundary, property tests | W11T6, W11T7 | 15-broadcast §7 |

### W12: Iterators (L4)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W12T1 | `src/iter/mod.rs` | Module skeleton: declarations, sub-module placeholders, public exports | None | 10-iterator §7 |
| W12T2 | `src/iter/elements.rs` | StrideState: F-order index increment state machine | W12T1 | 10-iterator §7 |
| W12T3 | `src/iter/elements.rs` | Iter / IterMut: Iterator + ExactSizeIterator impls with fast/slow paths (contiguous vs non-contiguous) | W12T2 | 10-iterator §7 |
| W12T4 | `src/iter/axis.rs` | AxisIter / AxisIterMut: iterate along one axis, yielding sub-views | W12T1 | 10-iterator §7 |
| W12T5 | `src/iter/indexed.rs` | IndexedIter / IndexedIterMut: index-wrapped iteration based on Iter | W12T3 | 10-iterator §7 |
| W12T6 | `src/iter/mod.rs` | TensorBase entry methods: iter(), iter_mut(), axis_iter(), indexed_iter() | W12T3, W12T4, W12T5 | 10-iterator §7 |

### W13: FFI Helpers (L4)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W13T1 | `src/ffi/mod.rs`, `types.rs` | Module skeleton: declarations, re-exports, FfiErrorCategory, BlasInfo struct | None | 23-ffi §7 |
| W13T2 | `src/ffi/ptr.rs` | Re-export as_ptr/as_mut_ptr/from_raw_parts/from_raw_parts_mut/into_raw_parts + FFI wrappers export()/export_mut() | W13T1 | 23-ffi §7 |
| W13T3 | `src/ffi/blas.rs` | is_blas_layout_compatible(), blas_info(), lda() | W13T1 | 23-ffi §7 |
| W13T4 | `src/ffi/offset.rs` | try_offset_of() / try_ptr_at() with checked arithmetic validation | W13T1 | 23-ffi §7 |

### W14: SIMD Backend (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W14T1 | `src/simd/mod.rs` | Module skeleton: SimdElement trait, SimdKernel trait, Arch cache, public vectorized entry points | None | 08-simd §7 |
| W14T2 | `src/simd/vector.rs` | Element-wise SIMD: VectorKernel\<A\>, Add/Sub/Mul/Div Kernel WithSimd impls for f32/f64 | W14T1 | 08-simd §7 |
| W14T3 | `src/simd/vector.rs` | Float sum SIMD: f32/f64 SumKernel with lane accumulation + documented merge | W14T2 | 08-simd §7 |
| W14T4 | `src/simd/vector.rs` | Integer sum/dot SIMD admission + fallback: i32/i64 checked semantics, ISA widening kernel gating | W14T3 | 08-simd §7 |
| W14T5 | `src/simd/vector.rs` | Complex sum SIMD: Complex\<f32\>/Complex\<f64\> AoS split real/imag accumulation | W14T3 | 08-simd §7 |
| W14T6 | `src/simd/vector.rs` | Float + complex dot SIMD: f32/f64/Complex\<f32\>/Complex\<f64\> dot kernel with conjugate contract | W14T3, W14T5 | 08-simd §7 |
| W14T7 | `src/simd/mod.rs`, `Cargo.toml` | Feature gate conditional compilation: #[cfg(feature = "simd")], public API export, dispatch integration | W14T1, W14T2 | 08-simd §7 |
| W14T8 | `src/simd/vector.rs` (#[cfg(test)]) | Element-wise consistency tests: SIMD vs serial baseline bitwise agreement | W14T7 | 08-simd §7 |
| W14T9 | `src/simd/vector.rs` (#[cfg(test)]) | Reduction/dot semantic + tolerance tests: float/complex/int SIMD entry condition constraints | W14T8 | 08-simd §7 |
| W14T10 | `tests/property/` | Randomized property tests: element-wise consistency + reduction/dot invariants under random input | W14T9 | 08-simd §7 |

### W15: Parallel Backend (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W15T1 | `src/parallel/iter.rs` | ParIter + TensorBase::par_iter(): single-input element-level parallel traversal entry | None | 09-parallel §7 |
| W15T2 | `src/parallel/map.rs` | par_map: pure parallel element-wise map entry, strategy from dispatch | W15T1 | 09-parallel §7 |
| W15T3 | `src/parallel/map.rs` | par_zip_map: dual-input broadcast element-wise parallel entry for math consumption | W15T1 | 09-parallel §7 |
| W15T4 | `src/parallel/reduce.rs` | par_reduce_impl + par_sum: parallel reduction, identity merge, semantic alignment with caller's serial baseline | W15T1 | 09-parallel §7 |
| W15T5 | `src/parallel/reduce.rs` | par_dot: ndim==1 check, length consistency, parallel inner product, error return + empty identity | W15T4 | 09-parallel §7 |
| W15T6 | `src/parallel/mod.rs` | ParallelPool: custom rayon::ThreadPool wrapper, preserving public API result semantics | W15T2, W15T4, W15T5 | 09-parallel §7 |
| W15T7 | `src/parallel/checked.rs` | Error + panic propagation: XenonError passthrough, panic not swallowed | W15T2, W15T4, W15T5 | 09-parallel §7 |
| W15T8 | `src/parallel/` all files, `tests/test_parallel.rs` | Feature gate + config matrix tests: default off, --features parallel build, single/multi-thread branch validation | W15T1–W15T7 | 09-parallel §7 |

### W16: Math Operations (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W16T1 | `src/math/mod.rs` | Module skeleton: declarations, public API re-exports | None | 11-math §7 |
| W16T2 | `src/math/binary.rs` | Shared binary element-wise execution skeleton with broadcast support | W16T1 | 11-math §7 |
| W16T3 | `src/math/unary.rs` | Unary element-wise ops: abs, neg, signum, square (with checked arithmetic for integers) | W16T1 | 11-math §7 |
| W16T4 | `src/math/unary.rs` | Math functions for RealScalar: sin, sqrt, exp, ln, floor, ceil | W16T1 | 11-math §7 |
| W16T5 | `src/math/unary.rs` | Complex ops: conjugate, modulus | W16T1 | 11-math §7 |
| W16T6 | `src/math/binary.rs` | Arithmetic ops: add, sub, mul, div (scalar variants) via shared binary skeleton | W16T2 | 11-math §7 |
| W16T7 | `src/math/unary.rs`, `comparison.rs` | Logical not (bool) + comparison ops: equal, not_equal, less, greater (return bool tensors) | W16T2 | 11-math §7 |
| W16T8 | `src/math/binary.rs`, `unary.rs`, `comparison.rs`, `src/simd/vector.rs` | SIMD backend unified dispatch integration for math module | W16T3 | 11-math §7 |

### W17: Matrix Operations (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W17T1 | `src/matrix/mod.rs`, `dot.rs` | Module skeleton: declarations, dot function signatures | None | 12-matrix §7 |
| W17T2 | `src/matrix/dot.rs` | dot() base execution: rank/shape validation, scalar inner product (real + complex), dispatch skeleton | W17T1 | 12-matrix §7 |
| W17T3 | `src/matrix/dot.rs` | Scalar path consolidation: harden rank/shape validation + wire dispatch serial/parallel decision | W17T2 | 12-matrix §7 |
| W17T4 | `src/matrix/dot.rs`, `src/simd/mod.rs` | SIMD path integration: SIMD kernel wiring, scalar fallback when conditions not met | W17T3 | 12-matrix §7 |
| W17T5 | `src/matrix/dot.rs`, `src/parallel/mod.rs` | Parallel path integration: dispatch-driven parallel decision, nested-parallel guard, per-worker local path selection | W17T3 | 12-matrix §7 |
| W17T6 | `tests/test_matrix.rs` | Integration tests: correctness, dimension mismatch, complex, feature-gate fallback | W17T2–W17T5 | 12-matrix §7 |

### W18: Reduction Operations (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W18T1 | `src/reduction/mod.rs` | Module skeleton: public API exports (sum family) | None | 13-reduction §7 |
| W18T2 | `src/reduction/sum.rs` | sum(): full traversal, integer checked arithmetic, empty array zero semantics | W18T1 | 13-reduction §7 |
| W18T3 | `src/reduction/sum.rs` | sum_axis(): axis validation, output shape reduction, per-axis slot accumulation | W18T2 | 13-reduction §7 |
| W18T4 | `src/reduction/sum.rs` | sum_axis_keepdims(): reuse per-axis logic, preserve reduced axis as length 1 | W18T3 | 13-reduction §7 |
| W18T5 | `src/reduction/sum.rs`, `src/simd/`, `src/parallel/` | SIMD / parallel dispatch guards: wire dispatch results, prevent routing semantically invalid inputs | W18T2–W18T4 | 13-reduction §7 |
| W18T6 | `src/reduction/sum.rs`, `tests/test_reduction.rs` | Error semantic convergence: axis OOB → InvalidAxis, integer overflow → panic | W18T3–W18T5 | 13-reduction §7 |

### W19: Set Operations (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W19T1 | `src/set/unique.rs` | Module skeleton: UniqueElement trait definition | None | 14-set §7 |
| W19T2 | `src/set/unique.rs` | unique(): element collection, equality-based deduplication, Tensor construction | W19T1 | 14-set §7 |
| W19T3 | `src/set/unique.rs` | Float NaN / ±0.0 equality handling: preserve each NaN, treat -0.0 and 0.0 as equal | W19T2 | 14-set §7 |
| W19T4 | `src/set/unique.rs` | Complex component-wise equality: real/imag parts follow respective real semantics, no ordering | W19T2 | 14-set §7 |
| W19T5 | `src/set/unique.rs` | unique() entry method on TensorBase | W19T2–W19T4 | 14-set §7 |

### W20: Shape Operations (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W20T1 | `src/shape/mod.rs`, `transpose.rs` | Module skeleton: declarations, transpose file placeholder, public exports | None | 16-shape §7 |
| W20T2 | `src/shape/transpose.rs` | transpose(): axis swap, O(1) shape/stride recomputation | W20T1 | 16-shape §7 |
| W20T3 | `tests/test_shape.rs` | Integration tests: transpose correctness, 0D/1D no-op semantics, large-array O(1) behavior | W20T2 | 16-shape §7 |

### W21: Indexing (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W21T1 | `src/index/ndindex.rs` | NdIndex\<D\> trait + tuple/slice index legality check: rank match, per-axis bounds, offset calculation | None | 17-indexing §7 |
| W21T2 | `src/index/access.rs` | try_at / get / get_unchecked: unified safe + unsafe read access paths | W21T1 | 17-indexing §7 |
| W21T3 | `src/index/slice.rs` | SliceInfoElem + SliceInfoIndices: inline/dynamic slice descriptor representations | W21T1 | 17-indexing §7 |
| W21T4 | `src/index/access.rs` | try_at_mut / get_mut / get_unchecked_mut: mutable access, gated on StorageMut | W21T2 | 17-indexing §7 |
| W21T5 | `src/index/slice.rs` | slice() shape/stride update + layout recomputation: axis folding, Range→shape/stride, read-only view return | W21T3 | 17-indexing §7 |

### W22: Tensor Construction (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W22T1 | `src/construct/mod.rs`, `init.rs` | Module skeleton + zeros() / ones() | None | 18-construction §7 |
| W22T2 | `src/construct/eye.rs` | eye(): identity matrix constructor | W22T1 | 18-construction §7 |
| W22T3 | `src/construct/from.rs` | from_shape_vec + from_shape_slice: consume Vec into shared owned path, copy from slice, 1D convenience | W22T1 | 18-construction §7 |
| W22T4 | `src/construct/from.rs`, `scalar.rs` | from_array + from_scalar: fixed-array construction, zero-dim tensor | W22T3 | 18-construction §7 |
| W22T5 | `tests/test_construction.rs` | Integration tests: all construction methods, boundary cases | W22T1–W22T4 | 18-construction §7 |

### W23: Operator Overloading (L6)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W23T1 | `src/overload/arithmetic.rs` | Module skeleton: declarations, imports | None | 19-overload §7 |
| W23T2 | `src/overload/arithmetic.rs` | Add\<Tensor, Tensor\> for owned: Tensor + Tensor impl | W23T1 | 19-overload §7 |
| W23T3 | `src/overload/arithmetic.rs` | Add for ref/mixed: &Tensor + &Tensor, Tensor + &Tensor, &Tensor + Tensor (4 combos) | W23T2 | 19-overload §7 |
| W23T4 | `src/overload/arithmetic.rs` | Add with scalar: Tensor + scalar, scalar + Tensor | W23T2 | 19-overload §7 |
| W23T5 | `src/overload/arithmetic.rs` | Sub / Mul / Div operators: replicate Add pattern for all combinations | W23T3, W23T4 | 19-overload §7 |
| W23T6 | `tests/test_overload.rs` | Integration tests: broadcast combos, scalar combos, type combos, deep-copy verification | W23T1–W23T5 | 19-overload §7 |

### W24: Utility Operations (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W24T1 | `src/util/fill.rs` | fill(): StorageMut-level fill helper + try_fill() dispatch for all tensor types | None | 20-utility §7 |
| W24T2 | `src/util/clip.rs` | clip(): element-wise clipping with NaN/min=max/NaN-bound/Integer error handling | None | 20-utility §7 |
| W24T3 | `src/util/contiguous.rs` | to_contiguous() + into_contiguous(): F-contiguous guarantee, reuse or repack | None | 20-utility §7 |
| W24T4 | `tests/test_utility.rs` | Integration tests: boundary cases (empty, single-element, non-contiguous, zero-dim) for all utility ops | W24T1–W24T3 | 20-utility §7 |

### W25: Type Conversion (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W25T1 | `src/convert/cast.rs` | CastTo trait: Tier-2 lossy (14 cells) + Tier-3 dynamic (8 cells); Tier-1 lossless (11 cells) via ConvertTo/From shims | None | 21-type §7 |
| W25T2 | `src/convert/mod.rs`, `src/lib.rs` | Module skeleton: sub-module declarations, pub use exports | W25T1 | 21-type §7 |
| W25T3 | `src/convert/cast.rs` | to_owned() / into_owned(): clone and consume owned conversion methods | W25T2 | 21-type §7 |
| W25T4 | `src/convert/cast.rs` | cast\<B\>() method: all readable storage inputs → owned result with error reporting | W25T2 | 21-type §7 |
| W25T5 | `src/convert/cast.rs` | Complete CastTo impls: int↔int, real↔complex, complex↔complex (all combos from require.md §23) | W25T1 | 21-type §7 |

### W26: Output Formatting (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W26T1 | `src/format/mod.rs`, `config.rs` | Module skeleton: declarations, re-exports, FormatConfig struct + Default | None | 22-output §7 |
| W26T2 | `src/format/pretty.rs` | Numpy-style formatting helpers: fmt_1d_display, fmt_1d_debug, fmt_nd_display, fmt_nd_debug with truncation | W26T1 | 22-output §7 |
| W26T3 | `src/format/display.rs` | core::fmt::Display for TensorBase\<S, D\>, delegating to pretty.rs | W26T2 | 22-output §7 |
| W26T4 | `src/format/debug.rs` | core::fmt::Debug for TensorBase\<S, D\> with shape/stride/type metadata, delegating to pretty.rs | W26T2 | 22-output §7 |
| W26T5 | `src/format/mod.rs`, `display.rs` | Module docs + re-exports completion | W26T3, W26T4 | 22-output §7 |

### W27: Safety Audit (cross-cutting)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W27T1 | `src/storage/owned.rs` | unsafe impl Send for Owned\<A\> + unsafe impl Sync for Owned\<A\> with full SAFETY comments | None | 25-safety §7 |
| W27T2 | `src/storage/view.rs` | unsafe impl Send for ViewRepr\<'a, A\> + unsafe impl Sync for ViewRepr\<'a, A\> with SAFETY comments | None | 25-safety §7 |
| W27T3 | `src/storage/viewmut.rs` | unsafe impl Send for ViewMutRepr\<'a, A\>, no Sync impl (documented), SAFETY comments | None | 25-safety §7 |
| W27T4 | `src/storage/arc.rs` | unsafe impl Send + Sync for ArcRepr\<A\> with SAFETY comments | None | 25-safety §7 |
| W27T5 | `src/parallel/iter.rs` | Parallel execution chunk safety: completeness, non-overlap, boundary tests | W27T1–W27T4 | 25-safety §7 |
| W27T6 | `tests/test_parallel.rs`, `tests/test_error.rs` | Thread-safety integration tests: cross-thread transfer, concurrent access | W27T1–W27T5 | 25-safety §7 |
| W27T7 | `src/storage/mod.rs` | Module-level thread-safety docs, Send/Sync matrix, cargo doc pass | W27T1–W27T4 | 25-safety §7 |

### W28: Benchmarks

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W28T1 | `Cargo.toml` | Add 9 [[bench]] entries, no new benchmark-specific third-party deps | None | 27-benchmark §7 |
| W28T2 | `benches/utils/mod.rs`, `generators.rs` | Shared constants (SIZES_1D, SIZES_2D) + data generation functions | W28T1 | 27-benchmark §7 |
| W28T3 | `benches/math.rs` | Element-wise math benchmarks: add/sub/mul/div/sin/exp/abs (f32/f64/Complex\<f64\>, contiguous + non-contiguous) | W28T2 | 27-benchmark §7 |
| W28T4 | `benches/reduction.rs` | Reduction benchmarks: sum_1d_f64, sum_2d_axis0, sum_2d_axis1, sum_sliced, sum_2d_keepdims | W28T2 | 27-benchmark §7 |
| W28T5 | `benches/dot.rs` | Dot benchmarks: dot_1d_f64, dot_1d_complex | W28T2 | 27-benchmark §7 |
| W28T6 | `benches/set.rs` | Set benchmarks: unique_1d (varying sizes, unique ratio) | W28T2 | 27-benchmark §7 |
| W28T7 | `benches/broadcast.rs` | Broadcast benchmarks: broadcast_scalar, broadcast_row, broadcast_col, broadcast_with | W28T2 | 27-benchmark §7 |
| W28T8 | `benches/shape.rs` | Shape benchmarks: transpose_2d | W28T2 | 27-benchmark §7 |
| W28T9 | `benches/construction.rs` | Construction benchmarks: zeros_1d, from_shape_vec_1d, eye_2d | W28T2 | 27-benchmark §7 |
| W28T10 | `benches/simd_comparison.rs` | SIMD comparison: add/sum/dot with --features simd on/off | W28T3, W28T4, W28T5 | 27-benchmark §7 |
| W28T11 | `benches/parallel_comparison.rs` | Parallel comparison: sum/add/dot with --features parallel on/off | W28T3, W28T4, W28T5 | 27-benchmark §7 |
| W28T12 | `.github/workflows/bench.yml` | CI bench smoke test: quick-mode execution, regression annotation | W28T3–W28T11 | 27-benchmark §7 |

### W29: Integration Tests

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W29T1 | `tests/common/mod.rs`, `assertions.rs`, `generators.rs` | Test infrastructure: assert_tensor_exact_real/complex, real_bits_eq/real_ulp_eq, tolerance helpers, data generators | None | 28-tests §7 |
| W29T2 | `tests/test_tensor.rs` | Core tensor tests: shape/strides/view/to_owned/type_aliases/debug_display/arc | W29T1 | 28-tests §7 |
| W29T3 | `tests/test_math.rs` | Math tests: element-wise arithmetic/math/comparison/logical ops | W29T1 | 28-tests §7 |
| W29T4 | `tests/test_overload.rs` | Overload tests: Add/Sub/Mul/Div traits, broadcast dispatch, Result ownership, scalar operators | W29T1, W29T3 | 28-tests §7 |
| W29T5 | `tests/test_broadcast.rs` | Broadcast tests: scalar/row/col/incompatible/read-only broadcast | W29T1 | 28-tests §7 |
| W29T6 | `tests/test_index.rs` | Index tests: multi-dim indexing, OOB errors, slicing, structural SliceInfo validation | W29T1 | 28-tests §7 |
| W29T7 | `tests/test_construction.rs` | Construction tests: zeros/ones/eye/from_shape_vec/slice/from_scalar/from_array | W29T1 | 28-tests §7 |
| W29T8 | `tests/test_reduction.rs` | Reduction tests: sum/sum_axis/keepdims/empty/NaN/overflow | W29T1 | 28-tests §7 |
| W29T9 | `tests/test_iterator.rs` | Iterator tests: elements/axis/indexed iteration | W29T1 | 28-tests §7 |
| W29T10 | `tests/test_matrix.rs` | Matrix tests: dot/complex/shape mismatch | W29T1 | 28-tests §7 |
| W29T11 | `tests/test_set.rs` | Set tests: unique (int/complex/NaN/±0.0/multiset, output order unspecified) | W29T1 | 28-tests §7 |
| W29T12 | `tests/test_shape.rs` | Shape tests: transpose/high-dim | W29T1 | 28-tests §7 |
| W29T13 | `tests/test_conversion.rs` | Conversion tests: cast/to_owned/into_owned | W29T1 | 28-tests §7 |
| W29T14 | `tests/test_utility.rs` | Utility tests: fill/clip/to_contiguous | W29T1 | 28-tests §7 |
| W29T15 | `tests/test_output.rs` | Output tests: Display/Debug/truncation/complex (Numpy-style) | W29T1 | 28-tests §7 |
| W29T16 | `tests/test_error.rs` | Error tests: XenonError boundary + display output validation, Workspace structured fields | W29T1 | 28-tests §7 |
| W29T17 | `tests/test_workspace.rs` | Workspace tests: illegal alignment/borrow guard/split/expand/assume_init/!Send+!Sync | W29T1, W29T2 | 28-tests §7 |
| W29T18 | `tests/test_ffi.rs` | FFI tests: pointer/BLAS compatibility/export/export_mut/offset | W29T2 | 28-tests §7 |
| W29T19 | `tests/test_parallel.rs` | Parallel tests: sum/add behavioral consistency with parallel feature, concurrent read, nested prohibition | W29T3, W29T8 | 28-tests §7 |
| W29T20 | `tests/test_simd.rs` | SIMD tests: result consistency (add/sum/fallback) | W29T3, W29T8 | 28-tests §7 |
| W29T21 | `.github/workflows/test.yml` | CI test matrix: maintain std-environment lib/tests/doctest matrix | W29T2 | 28-tests §7 |
| W29T22 | `tests/property_tests.rs`, `property/tensor_props.rs`, `ops_props.rs`, `shape_props.rs` | Property tests: transpose involution, addition commutativity, unique no-duplicates, sum preserves identity, broadcast shape consistency | W29T3, W29T8, W29T12 | 28-tests §7 |
| W29T23 | `.github/workflows/test.yml` | CI test matrix full config: all feature combos, compile-fail, property tests | W29T1–W29T22 | 28-tests §7 |

### W30: Documentation

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W30T1 | `src/lib.rs` | Crate-level docs: project overview, Quick Start, Features table, element types table, memory layout | None | 29-documentation §7 |
| W30T2 | `src/lib.rs`, `Cargo.toml` | #![warn(missing_docs)] lint + docs.rs metadata: all-features = true | W30T1 | 29-documentation §7 |
| W30T3 | `README.md` | Project README: intro, features, Quick Start, install, doc links, license | W30T1 | 29-documentation §7 |
| W30T4 | `CHANGELOG.md` | Optional CHANGELOG.md in Keep a Changelog format | None | 29-documentation §7 |
| W30T5 | Core module docs: `dimension/`, `element/`, `complex/`, `storage/`, `layout/` | Module-level docs: responsibilities, core concepts, usage examples, dependency graph, design decisions | W30T2 | 29-documentation §7 |
| W30T6 | Tensor + ops module docs: `tensor/`, `iter/`, `math/`, `overload/`, `broadcast/`, `reduction/`, `matrix/`, `shape/`, `index/`, `construct/`, `set/` | Module-level docs: responsibilities, core types, op categories, type constraint quick reference | W30T2 | 29-documentation §7 |
| W30T7 | Infrastructure module docs: `ffi/`, `workspace/`, `error/`, `convert/`, `format/`, `util/`, `simd/`, `parallel/` | Module-level docs: responsibilities, safety conventions, feature gate notes, conversion/output semantics | W30T2 | 29-documentation §7 |
| W30T8 | `src/util/mod.rs` | Util module-level docs: responsibility overview, utility function category notes (module-level only, no function-level doctests) | W30T2 | 29-documentation §7 |
| W30T9 | `src/tensor/mod.rs` + related files | Tensor type-level docs: TensorBase, Tensor, TensorView, TensorViewMut, ArcTensor | W30T5–W30T7 | 29-documentation §7 |
| W30T10 | `src/dimension/mod.rs` | Dimension docs: Ix0–Ix6, IxDyn, Dimension trait | W30T5–W30T7 | 29-documentation §7 |
| W30T11 | `src/element/mod.rs` | Element docs: Element, Numeric, RealScalar, ComplexScalar traits | W30T5–W30T7 | 29-documentation §7 |
| W30T12 | `src/storage/mod.rs` | Storage docs: Owned, ViewRepr, StorageMut trait | W30T5–W30T7 | 29-documentation §7 |
| W30T13 | `src/layout/mod.rs` | Layout docs: LayoutFlags, compute_f_strides | W30T5–W30T7 | 29-documentation §7 |
| W30T14 | `src/math/` files | Math function docs + doctests: add, sub, mul, div, sin, sqrt, exp, ln, abs | W30T5–W30T7 | 29-documentation §7 |
| W30T15 | `src/reduction/`, `src/matrix/` | Reduction + matrix docs + doctests: sum, sum_axis, dot | W30T5–W30T7 | 29-documentation §7 |
| W30T16 | `src/broadcast/`, `src/shape/mod.rs` | Broadcast + shape docs + doctests: broadcast_shape, transpose | W30T5–W30T7 | 29-documentation §7 |
| W30T17 | `src/construct/mod.rs`, `src/set/mod.rs` | Construct + set docs + doctests: zeros, ones, eye, from_shape_vec, unique + error semantics | W30T5–W30T7 | 29-documentation §7 |
| W30T18 | `src/ffi/mod.rs`, `src/workspace/mod.rs`, `src/error.rs` | FFI + workspace + error docs + doctests | W30T5–W30T7 | 29-documentation §7 |
| W30T19 | `src/iter/mod.rs`, `src/convert/mod.rs`, `src/format/mod.rs`, `src/overload/mod.rs` | Iter + convert + format + overload module docs + doctests | W30T5–W30T7 | 29-documentation §7 |
| W30T20 | `src/index/mod.rs` | Index function-level docs + doctests | W30T5–W30T7 | 29-documentation §7 |
| W30T21 | `src/util/mod.rs` | Util function-level docs + doctests: clip, fill, try_fill, to_contiguous, into_contiguous | W30T8 | 29-documentation §7 |
| W30T22 | `examples/basic.rs` | Usage example: create, operate, reduce, print | W30T1 | 29-documentation §7 |
| W30T23 | `examples/complex.rs` | Usage example: complex construction, same-type arithmetic, explicit conversion ops | W30T1 | 29-documentation §7 |
| W30T24 | `examples/broadcasting.rs` | Usage example: broadcast rules, row/col/scalar broadcast | W30T1 | 29-documentation §7 |
| W30T25 | `examples/features.rs` | Feature-gated example: conditional compile with simd/parallel features | W30T1 | 29-documentation §7 |
| W30T26 | CI docs.rs config | docs.rs CI integration + doc test verification | W30T19 | 29-documentation §7 |

---

## Dependency Graph (Simplified)

```
W1 (Coding/Setup)  ──┬──→ W3 (Dimension) ──→ W6 (Layout)
                      │
W2 (Error) ───────────┤
                      │
W3 → W4 (Element) ←── W5 (Complex) ──────────────┐
W2 → W5 (Complex) ────────────────────────────────┤
                      │
W6 + W3 ──→ W7 (Storage) ──┬──→ W8 (Tensor Core) ──→ W9 (Workspace, L2)
                            │
W8 + W6 + W2 ────→ W10 (Dispatch) ──┬──→ W11 (Broadcast)
                                     ├──→ W12 (Iterators)
                                     ├──→ W13 (FFI)
                                     ├──→ W14 (SIMD)
                                     └──→ W15 (Parallel)

W8 + W11 + W12 + W14 + W15 ─→ W16 (Math)
W8 + W14 + W15 ─→ W17 (Matrix)
W8 + W14 + W15 ─→ W18 (Reduction)
W8 + W12 ─→ W19 (Set)
W8 + W6 ─→ W20 (Shape)
W8 + W3 ─→ W21 (Indexing)
W8 ─→ W22 (Construction)
W8 + W11 + W16 ─→ W23 (Overload)
W8 ─→ W24 (Utility)
W4 + W8 ─→ W25 (Type Conversion)
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
