# Xenon Implementation Plan — Wave & Task Summary

> Project: Xenon N-dimensional array library (Rust)
> Dependency Layers: L0 → L1 → L2 → L3 → L4 → L5 → L6 → Cross-cutting
> Task Granularity: Each task targets 1 function / 1 trait / 1 type, ~5–10 min, max 1 file

---

## Wave Overview

| Wave | Name | Layer | Task Count | Description |
|------|------|-------|------------|-------------|
| W1 | Coding Standards & Project Setup | L0 | 6 | Cargo.toml, rustfmt.toml, lib.rs/prelude.rs skeleton with lint attrs, .clippy.toml, CI config |
| W2 | Error System | L0 | 5 | XenonError enum, Result alias, Display/Error impls, auxiliary enums, prelude exports |
| W3 | Dimension System | L1 | 22 | Static dims (Ix0–Ix6), IxDyn, Dimension/IntoDimension/RemoveAxis traits, Axis, Sealed, BroadcastDim trait + 57-item matrix |
| W4 | Element Type Hierarchy | L1 | 17 | Element/Numeric/RealScalar/ComplexScalar traits, sealed, primitives impls, integration |
| W5 | Complex Type | L1 | 16 | Complex\<T\> struct, arithmetic ops (Add/Sub/Mul/Div/Neg), Display/Debug, math methods, FFI layout, convert |
| W6 | Layout System | L2 | 11 | LayoutFlags bitflags, F-order stride computation, contiguity checks, alignment, zero-stride detection, compute_layout_flags central entry |
| W7 | Storage System | L2 | 19 | RawStorage/Storage/StorageMut/RawStorageMut/StorageOwned/StorageShared traits, marker traits, Owned/A, AlignedAlloc, ViewRepr, ViewMutRepr, ArcRepr |
| W8 | Tensor Core | L3 | 10 | TensorBase\<S,D\>, type aliases (Tensor/TensorView/TensorViewMut/ArcTensor + dimension convenience aliases), constructors, view methods, accessors, from_raw_parts |
| W9 | Workspace | L2 | 7 | Workspace struct, borrow guards (WorkspaceBorrow/WorkspaceBorrowMut), split (SplitBorrowMut), expand (ensure_capacity/reallocate), docs |
| W10 | Dispatch | L4 | 6 | ExecPath enum, select_exec_path, parallel/SIMD threshold config (set/reset for both), ParallelGuard (nested parallel protection), with_parallel_worker_context, ParallelExecStrategy |
| W11 | Broadcasting | L4 | 10 | can_broadcast, broadcast_shape, broadcast_strides, broadcast_to, broadcast_with, error handling, integration tests (BroadcastDim trait is delivered by W3T22) |
| W12 | Iterators | L4 | 7 | StrideState, Iter, IterMut, AxisIter/AxisIterMut, IndexedIter/IndexedIterMut, TensorBase entry methods |
| W13 | FFI Helpers | L4 | 6 | BlasInfo, TensorExport/TensorExportMut private descriptors, ptr re-exports, export/export_mut, is_blas_compatible/lda, try_offset_of/try_ptr_at (depends on W22T10 for OwnedRawParts) |
| W14 | SIMD Backend | L5 | 12 | pulp 0.18 API spike (W14T0), SimdKernel trait, element-wise SIMD (add/sub/mul/div/neg via dispatch_vector_binary_op + dispatch_vector_unary_op, for f32/f64 in W14T2 + Complex\<f32\>/Complex\<f64\> in W14T11), SIMD sum/dot (float/complex via `try_*` facades; integer i32 only with checked i64→i32 narrowing; i64 has no SIMD facade), feature gates, property tests |
| W15 | Parallel Backend | L5 | 8 | ParIter, par_map, par_zip_map, par_sum, par_dot, ParallelPool, error/panic propagation, feature gates |
| W16 | Math Operations | L5 | 11 | Binary element-wise ops (add/sub/mul/div), unary ops (abs/neg/signum/square/sin/sqrt/exp/ln/floor/ceil/conj/modulus), comparison ops (eq/ne/lt/le/gt/ge), logical not, SIMD dispatch |
| W17 | Matrix Operations | L5 | 7 | dot product (serial + SIMD + parallel paths), rank/shape validation, complex dot, integration tests |
| W18 | Reduction Operations | L5 | 6 | sum (global), sum_axis, sum_axis_keepdims, SIMD/parallel dispatch gates, error convergence |
| W19 | Set Operations | L5 | 6 | set module root, unique (real/complex/NaN/signed-zero handling), TensorBase entry method |
| W20 | Shape Operations | L5 | 4 | transpose (full-axis reversal), contiguity recomputation, integration tests |
| W21 | Indexing | L5 | 6 | index module root, NdIndex trait, try_at/get/get_unchecked, SliceInfo, try_at_mut/get_mut/get_unchecked_mut, slice shape/stride update |
| W22 | Tensor Construction | L5 | 10 | zeros, ones, eye, from_shape_vec, from_shape_slice, from_vec, from_array, from_scalar, OwnedRawParts + into_raw_parts/from_raw_parts_owned |
| W23 | Operator Overloading | L6 | 11 | overload module root, Add/Sub/Mul/Div for owned/ref/mixed/scalar tensor combinations, integration tests |
| W24 | Utility Operations | L5 | 5 | util module root, fill, clip, to_contiguous/into_contiguous |
| W25 | Type Conversion | L5 | 7 | CastTo trait (lossy + dynamic tiers), ConvertTo (lossless), cast method, to_owned/into_owned |
| W26 | Output Formatting | L5 | 6 | FormatConfig, Display (Numpy-style), Debug (with metadata), pretty formatting helpers |
| W27 | Safety Audit | cross-cutting | 7 | Send/Sync impls for Owned/ViewRepr/ViewMutRepr/ArcRepr, parallel chunk safety, thread-safety integration tests |
| W28 | Benchmarks | cross-cutting | 12 | bench infrastructure (utils/generators), core benches (math/reduction/dot/set/broadcast), shape/construction benches, SIMD/parallel comparison, CI/report script |
| W29 | Integration Tests | cross-cutting | 25 | tests/common utils, core test files (tensor/math/overload/broadcast/index/construction/reduction/iter/matrix/set/shape/conversion/utility/output/error), specialized tests (workspace/ffi/parallel/simd), compile-fail tests, property tests, CI matrix |
| W30 | Documentation | cross-cutting | 52 | Crate-level docs, per-module docs, type/function-level docs + doctests, usage examples, README/LICENSE/CHANGELOG, docs CI |
| | **Total** | | **337** | |

---

## Detailed Task List

### W1: Coding Standards & Project Setup (L0)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W1T1 | `Cargo.toml` | Package manifest with crate metadata, edition/MSRV, feature gates, optional deps, profiles, docs.rs metadata | None | 01-architecture §4, 00-coding §10 |
| W1T2 | `rustfmt.toml` | Rustfmt configuration per §3.2 coding standard | W1T1 | 00-coding §12 |
| W1T3 | `src/lib.rs` | Crate root skeleton with lint declarations (missing_docs, unsafe_op_in_unsafe_fn, clippy::unwrap_used) + lib.rs module declaration evolution protocol | W1T1 | 00-coding §12, 01-architecture §3, §8 |
| W1T4 | `src/prelude.rs` | Prelude module skeleton and initial public export surface placeholder | W1T3 | 01-architecture §3, §7 |
| W1T5 | `.clippy.toml` | Clippy configuration: `disallowed-methods` for `unwrap` restrictions; numeric `as` cast lint set documented but delegated to CI command (`.clippy.toml` cannot upgrade lint levels) | W1T1 | 00-coding §12 |
| W1T6 | `.github/workflows/ci.yml` | CI config: fmt + clippy (feature matrix) + MSRV 1.85 check + test matrix across feature combinations + docs smoke build | W1T2, W1T3, W1T5 | 00-coding §12 |

### W2: Error System (L0)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W2T1 | `src/error.rs` | Auxiliary enum types: FfiErrorCategory, WorkspaceErrorCategory, ConversionFailureReason, InvalidArgumentKind, InvalidLayoutReason, StorageKindTag, StorageConversionKind, InvalidShapeKind, AbiMismatchKind, FfiBackend, WorkspaceBorrowKind, WorkspaceBorrowState, TypedViewRejection | W1T3 | 26-error §5, §6, §7, §8 |
| W2T2 | `src/error.rs` | XenonError enum definition with all structured variants + Result type alias | W2T1 | 26-error §5, §6, §7, §8 |
| W2T3 | `src/error.rs` | fmt::Display impl for XenonError with OrAny\<T\>, FmtShape helpers | W2T2 | 26-error §5, §6, §7, §8 |
| W2T4 | `src/error.rs` | std::error::Error impl (source chaining for Ffi/Workspace, None for leaf variants) | W2T3 | 26-error §5, §6, §7, §8 |
| W2T5 | `src/prelude.rs`, `src/lib.rs` | Public exports of XenonError, Result alias, auxiliary enums via prelude + `pub use error::XenonError;` direct re-export at crate root | W2T4 | 26-error §5, §6, §7, §8, 01-architecture §7, §8 |

### W3: Dimension System (L1)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W3T1 | `src/dimension/mod.rs` | Module skeleton: sub-module declarations + crate-root wiring (public re-exports deferred to W3T15) | None | 02-dimension §5, §6, §7, §8 |
| W3T2 | `src/dimension/axes.rs` | Axis newtype with new/index/checked_next/next/prev/is_first/is_last | W3T1 | 02-dimension §5, §6, §7, §8 |
| W3T3 | `src/dimension/mod.rs` | Dimension trait definition (all method signatures, MAX_DIMENSION constant) | W3T1, W3T2 | 02-dimension §5, §6, §7, §8 |
| W3T4 | `src/dimension/static.rs` | Ix0 zero-dimensional scalar with Dimension impl | W3T3 | 02-dimension §5, §6, §7, §8 |
| W3T5 | `src/dimension/static.rs` | Ix1 struct with Dimension impl + Index\<usize\> | W3T4 | 02-dimension §5, §6, §7, §8 |
| W3T6 | `src/dimension/static.rs` | Ix2 struct with Dimension impl + Index\<usize\> | W3T5 | 02-dimension §5, §6, §7, §8 |
| W3T7 | `src/dimension/static.rs` | Ix3 struct with Dimension impl + From\<(usize, usize, usize)\> | W3T6 | 02-dimension §5, §6, §7, §8 |
| W3T8 | `src/dimension/static.rs` | Ix4 struct with Dimension impl + From\<tuple\> | W3T7 | 02-dimension §5, §6, §7, §8 |
| W3T9 | `src/dimension/static.rs` | Ix5 struct with Dimension impl + From\<tuple\> | W3T8 | 02-dimension §5, §6, §7, §8 |
| W3T10 | `src/dimension/static.rs` | Ix6 struct with Dimension impl + From\<tuple\> | W3T9 | 02-dimension §5, §6, §7, §8 |
| W3T11 | `src/dimension/dynamic.rs` | IxDyn dynamic dimension with Dimension impl + constructors | W3T3 | 02-dimension §5, §6, §7, §8 |
| W3T12 | `src/dimension/static.rs`, `dynamic.rs` | into_dyn() / try_from_dyn() conversion methods | W3T10, W3T11 | 02-dimension §5, §6, §7, §8 |
| W3T13 | `src/error.rs` | XenonError::DimensionMismatch integration for dimension conversion failures | W2T1, W3T12 | 02-dimension §5, §6, §7, §8 |
| W3T14 | `src/dimension/into.rs` | IntoDimension trait + tuple/array/slice/Vec impls | W3T13 | 02-dimension §5, §6, §7, §8 |
| W3T15 | `src/private.rs`, `src/dimension/mod.rs` | Sealed trait impl for all dimension types + public exports | W3T2, W3T12, W3T13, W3T14 | 02-dimension §5, §6, §7, §8 |
| W3T16 | All `src/dimension/` files | Doc comments on all pub items, cargo doc verification | W3T15 | 02-dimension §5, §6, §7, §8 |
| W3T17 | `src/dimension/` (`#[cfg(test)] mod tests`) | Dimension in-module unit tests: Ix0, zero-length axis, large dim, overflow | W3T16 | 02-dimension §5, §6, §7, §8 |
| W3T18 | `tests/test_tensor.rs` | Dimension cross-module tests for tensor interaction | W3T16 | 02-dimension §5, §6, §7, §8 |
| W3T19 | `tests/test_shape.rs` | Dimension cross-module tests for shape operations | W3T16 | 02-dimension §5, §6, §7, §8 |
| W3T20 | `tests/test_index.rs` | Dimension cross-module tests for indexing | W3T16 | 02-dimension §5, §6, §7, §8 |
| W3T21 | `tests/property_tests.rs` | Dimension cross-module property tests | W3T16 | 02-dimension §5, §6, §7, §8 |
| W3T22 | `src/dimension/broadcast_dim.rs`, `src/dimension/mod.rs` | BroadcastDim trait (public sealed) + complete impl matrix for Ix0..Ix6 / IxDyn + compile-time symmetry tests | W3T15, W3T16 | 02-dimension §5.10 |

### W4: Element Type Hierarchy (L1)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W4T1 | `src/element/mod.rs` | Module skeleton + import shared Sealed + Element trait + ElementType enum + element_type_of/element_type_name_of helpers + OrderedCompareElement trait (§5.5) + CastTo<T> trait (§5.9) definitions | W3T15, W2T1 | 03-element §5, §6, §7, §8 |
| W4T2 | `src/element/numeric.rs` | Numeric trait definition (arithmetic supertraits + conjugate) | W4T1 | 03-element §5, §6, §7, §8 |
| W4T3 | `src/element/real.rs` | RealScalar trait: math functions (abs/sqrt/sin/exp/ln/floor/ceil) + NaN detection | W4T2 | 03-element §5, §6, §7, §8 |
| W4T4 | `src/element/complex.rs` | ComplexScalar trait: associated type Real + complex methods (re/im/norm) | W4T2, W4T3 | 03-element §5, §6, §7, §8 |
| W4T5 | `src/element/primitives.rs` | Element + Numeric impl for i32 | W4T2 | 03-element §5, §6, §7, §8 |
| W4T6 | `src/element/primitives.rs` | Element + Numeric impl for i64 | W4T5 | 03-element §5, §6, §7, §8 |
| W4T7 | `src/element/primitives.rs` | Element + Numeric + RealScalar impls for f32, f64 | W4T3, W4T6 | 03-element §5, §6, §7, §8 |
| W4T8 | `src/element/primitives.rs` | Element impl for bool (zero=false, one=true, no Numeric) | W4T1 | 03-element §5, §6, §7, §8 |
| W4T9 | `src/element/mod.rs` | Documentation clarifying usize as index/shape metadata only, not an element type | W4T7 | 03-element §5, §6, §7, §8 |
| W4T10 | `src/element/primitives.rs` | Element + Numeric + ComplexScalar impls for Complex\<f32\>/Complex\<f64\> | W4T4, W5T1 | 03-element §5, §6, §7, §8 |
| W4T11 | `src/element/real.rs`, `src/element/mod.rs`, `src/element/primitives.rs` | Calibrate math boundaries + add CastElement/BoolElement/OrderedCompareElement impls + define §5.10 Checked Arithmetic Traits + document CastTo semantics | W4T7, W4T10 | 03-element §5, §6, §7, §8 |
| W4T12 | All `src/element/` files | Doc comments on all pub items, cargo doc verification | W4T10, W4T11 | 03-element §5, §6, §7, §8 |
| W4T13 | `src/element/primitives.rs` (`#[cfg(test)] mod tests`) | Element in-module unit tests: verify trait impls for all 7 element types | W4T10, W4T11 | 03-element §5, §6, §7, §8 |
| W4T14 | `tests/test_tensor.rs` | Element cross-module tests for tensor interaction | W4T10, W8 | 03-element §5, §6, §7, §8 |
| W4T15 | `tests/test_math.rs` | Element cross-module tests for math operations | W4T10 | 03-element §5, §6, §7, §8 |
| W4T16 | `tests/test_reduction.rs` | Element cross-module tests for reductions | W4T10, W8 | 03-element §5, §6, §7, §8 |
| W4T17 | `tests/test_conversion.rs` | Element cross-module tests for type conversion | W4T10, W8 | 03-element §5, §6, §7, §8 |

### W5: Complex Type (L1)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W5T1 | `src/complex/mod.rs`, `src/lib.rs` | Complex\<T\> struct (repr(C)) + new() + minimal ComplexFloat (Sealed+Copy+Default) + f32/f64 impls + `pub mod complex;` in lib.rs | W1T3 | 04-complex §5, §6, §7, §8 |
| W5T2 | `src/complex/mod.rs` | Extend ComplexFloat supertraits (Debug + PartialEq + PartialOrd + Add/Sub/Mul/Div/Neg) + sealed compile_fail doctest | W5T1, W3T15 | 04-complex §5, §6, §7, §8 |
| W5T3 | `src/complex/mod.rs` | const size/align assertions + field offset unit tests for FFI layout guarantees | W5T1 | 04-complex §5, §6, §7, §8 |
| W5T4 | `src/complex/mod.rs` | re() / im() accessors | W5T1 | 04-complex §5, §6, §7, §8 |
| W5T5 | `src/complex/mod.rs` | from_imag(), conj(), and From\<T\> constructors | W5T1, W5T2, W5T4, W5T7 | 04-complex §5, §6, §7, §8 |
| W5T6 | `src/complex/mod.rs` | is_real() / is_imaginary() predicates | W5T1, W5T2 | 04-complex §5, §6, §7, §8 |
| W5T7 | `src/complex/mod.rs` | PartialEq (NaN!=NaN) + Display (NaN-aware, -0.0-preserving, precision-aware) + crate-private PositiveZero | W5T1, W5T2 | 04-complex §5, §6, §7, §8 |
| W5T8 | `src/complex/ops.rs` | Complex Add operator | W5T1, W5T2, W5T7 | 04-complex §5, §6, §7, §8 |
| W5T9 | `src/complex/ops.rs` | Complex Sub operator | W5T8 | 04-complex §5, §6, §7, §8 |
| W5T10 | `src/complex/ops.rs` | Complex Mul operator | W5T1, W5T2, W5T7 | 04-complex §5, §6, §7, §8 |
| W5T11 | `src/complex/ops.rs` | Complex Div operator (independent Smith algorithm for f32 and f64) | W5T10 | 04-complex §5, §6, §7, §8 |
| W5T12 | `src/complex/ops.rs` | Complex Neg operator | W5T1, W5T2, W5T7 | 04-complex §5, §6, §7, §8 |
| W5T13 | `src/complex/ops.rs`, `src/complex/mod.rs` | Tighten real-complex mixed arithmetic boundary (static audit + 8 compile_fail doctests + explicit-conversion unit tests) | W5T8–W5T12 | 04-complex §5, §6, §7, §8 |
| W5T14 | `src/complex/mod.rs` | Math methods (norm via hypot, norm_sqr) + concrete predicates (is_nan, is_finite) for f32/f64 | W5T1 | 04-complex §5, §6, §7, §8 |
| W5T15 | All `src/complex/` files | Doc comments on all pub complex items, cargo doc verification | W5T13, W5T14 | 04-complex §5, §6, §7, §8 |
| W5T16 | `tests/test_complex.rs` | Integration and boundary tests for full complex type system | W5T15 | 04-complex §5, §6, §7, §8 |

### W6: Layout System (L2)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W6T1 | `src/layout/mod.rs` | Module skeleton: sub-module declarations, public exports | None | 06-layout §5, §6, §7, §8 |
| W6T2 | `src/layout/flags.rs` | Module skeleton: file placeholder, declarations | W6T1 | 06-layout §5, §6, §7, §8 |
| W6T3 | `src/layout/strides.rs` | Module skeleton: file placeholder, declarations | W6T1, W3T3 | 06-layout §5, §6, §7, §8 |
| W6T4 | `src/layout/contiguous.rs` | Module skeleton: file placeholder, declarations | W6T1, W6T3, W3T3 | 06-layout §5, §6, §7, §8 |
| W6T5 | `src/layout/flags.rs` | LayoutFlags(u8) bitflags: F_CONTIGUOUS, ALIGNED, HAS_ZERO_STRIDE + query/set methods + classify() (BroadcastView→FContiguous→NonContiguous) | W6T2 | 06-layout §5, §6, §7, §8 |
| W6T6 | `src/layout/strides.rs` | compute_f_strides\<D\> + Strides::from_slice/f_contiguous/try_stride/iter: F-order stride computation with overflow check returning Result (InvalidShape::ProductOverflow) | W6T3, W3T3 | 06-layout §5, §6, §7, §8 |
| W6T7 | `src/layout/contiguous.rs` | is_f_contiguous\<D\>: F-order contiguity detection | W6T4, W3T3 | 06-layout §5, §6, §7, §8 |
| W6T8 | `src/layout/strides.rs` | has_zero_stride: raw zero-stride detector + should_set_zero_stride_flag (checks product(shape) > 0) | W6T3, W3T3 | 06-layout §5, §6, §7, §8 |
| W6T9 | `src/layout/strides.rs` | is_aligned_to / is_aligned alignment check functions | W6T3 | 06-layout §5, §6, §7, §8 |
| W6T10 | `src/layout/mod.rs` (`#[cfg(test)] mod integration_tests`), `tests/test_layout.rs` | Layout integration tests (compute_f_strides + is_f_contiguous + compute_layout_flags cross-flow) | W6T6–W6T9, W6T11 | 06-layout §5, §6, §7, §8 |
| W6T11 | `src/layout/mod.rs` | compute_layout_flags\<A, D\> central entry + flags_for_f_layout (§5.2, §5.12, §6.1) | W6T5, W6T6, W6T7, W6T8, W6T9 | 06-layout §5, §6, §7, §8 |

### W7: Storage System (L2)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W7T1 | `src/storage/mod.rs` | Module skeleton: sub-module declarations, public exports | None | 05-storage §5, §6, §7, §8 |
| W7T2 | `src/storage/mod.rs` | unsafe trait RawStorage: as_ptr/len/is_empty/is_aligned_to/is_aligned | W7T1 | 05-storage §5, §6, §7, §8 |
| W7T3 | `src/storage/mod.rs` | unsafe trait Storage: RawStorage + get/get_unchecked/as_slice | W7T2 | 05-storage §5, §6, §7, §8 |
| W7T4 | `src/storage/mod.rs` | RawStorageMut and StorageMut trait definitions | W7T3 | 05-storage §5, §6, §7, §8 |
| W7T5 | `src/storage/mod.rs` | StorageOwned, StorageShared, StorageSharedExt, and StorageIntoOwned trait definitions | W7T4 | 05-storage §5.7, §5.8, §5.9, §6, §7, §8 |
| W7T6 | `src/storage/traits.rs` | Marker traits: IsOwned, IsView, IsViewMut, IsShared | W7T5 | 01-architecture §3, 05-storage §5, §6, §7, §8 |
| W7T7 | `src/storage/alloc.rs` | AlignedAlloc struct: 64-byte aligned alloc/alloc_zeroed/dealloc | W7T1 | 05-storage §5, §6, §7, §8 |
| W7T8 | `src/storage/owned.rs` | Owned\<A\> struct + new/with_capacity/from_vec/from_vec_aligned/zeros/from_elem constructors | W7T7 | 05-storage §5, §6, §7, §8 |
| W7T9 | `src/storage/owned.rs` | Owned\<A\> RawStorage trait impl | W7T5, W7T8 | 05-storage §5, §6, §7, §8 |
| W7T10 | `src/storage/owned.rs` | Owned\<A\> Storage trait impl | W7T9 | 05-storage §5, §6, §7, §8 |
| W7T11 | `src/storage/owned.rs` | Owned\<A\> StorageMut + StorageOwned trait impls + IsOwned marker impl | W7T10 | 05-storage §5, §6, §7, §8 |
| W7T12 | `src/storage/owned.rs` | Owned\<A\> into_shared + unsafe impl Send + unsafe impl Sync | W7T8, W7T16 | 05-storage §5, §6, §7, §8, 25-safety §5.5 |
| W7T13 | `src/storage/owned.rs` | Owned\<A\> TryFrom\<Vec\<A\>\> + Default impls | W7T8 | 05-storage §5, §6, §7, §8 |
| W7T14 | `src/storage/view.rs` | ViewRepr\<'a, A\>: struct, from_raw_parts/from_slice/view/slice, Clone/Copy, RawStorage/Storage impls | W7T5, W7T6 | 05-storage §5, §6, §7, §8 |
| W7T15 | `src/storage/viewmut.rs` | ViewMutRepr\<'a, A\>: struct, from_raw_parts_mut/from_mut_slice/view_mut/view, no Clone, all trait impls | W7T5, W7T6 | 05-storage §5, §6, §7, §8 |
| W7T16 | `src/storage/arc.rs` | ArcRepr\<A\>: struct, from_vec/from_vec_aligned/zeros/from_elem constructors, Clone (ref-count bump), all trait impls | W7T5 | 05-storage §5, §6, §7, §8 |
| W7T17 | `src/storage/arc.rs` | ArcRepr\<A\> Send/Sync + Default + TryFrom\<Vec\<A\>\> impls | W7T16 | 05-storage §5, §6, §7, §8 |
| W7T18 | `src/storage/mod.rs` | Module re-exports (all types + traits) + doc comments | W7T13, W7T14, W7T15, W7T17 | 05-storage §5, §6, §7, §8 |
| W7T19 | `src/storage/` (`#[cfg(test)] mod tests`), `tests/test_tensor.rs`, `tests/test_index.rs`, `tests/test_ffi.rs`, `tests/test_iterator.rs` | Storage integration tests: cross-storage interaction, ZST behavior, empty arrays, alloc alignment, trait semantics via in-module unit tests + cross-module coverage through existing tensor/index/ffi/iterator test files; do NOT add standalone tests/test_storage.rs | W7T18 | 05-storage §5, §6, §7, §8 |

### W8: Tensor Core (L3)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W8T1 | `src/tensor/mod.rs` | Module skeleton: sub-module declarations, public exports | None | 07-tensor §5, §6, §7, §8 |
| W8T2 | `src/tensor/mod.rs` | TensorBase\<S, D\> struct: 6 fields (storage, shape, strides, offset, flags, derived_from_view_mut) | W8T1, W6T5 | 07-tensor §5, §6, §7, §8 |
| W8T3 | `src/tensor/aliases.rs` | 4 primary type aliases (Tensor/TensorView/TensorViewMut/ArcTensor) + 4×8=32 dim convenience aliases | W8T2 | 07-tensor §5, §6, §7, §8 |
| W8T4 | `src/tensor/impls.rs` | Shape & stride query methods: shape/strides/ndim/len/is_empty/offset/raw_dim/flags/storage_kind/access_semantics/data_location | W8T2, W3T3, W6T5 | 07-tensor §5, §6, §7, §8 |
| W8T5 | `src/tensor/impls.rs` | Layout query delegation: layout_state/is_f_contiguous/is_aligned/has_zero_stride + AliasClass enum + alias_class() method | W8T4 | 07-tensor §5, §6, §7, §8 |
| W8T6 | `src/tensor/impls.rs` | Pointer access & slice: as_ptr/as_storage_ptr/storage_len/as_mut_ptr/as_storage_mut_ptr/as_slice/as_mut_slice | W8T4 | 07-tensor §5, §6, §7, §8 |
| W8T7 | `src/tensor/construct.rs` | from_raw_parts / from_raw_parts_mut with storage_len and validate_access_range | W8T2, W6T5, W3T3 | 07-tensor §5, §6, §7, §8 |
| W8T8 | `src/tensor/construct.rs` | from_raw_vec_unchecked (pub(crate) unsafe internal constructor) | W8T5, W8T7, W6T5 | 07-tensor §5, §6, §7, §8 |
| W8T9 | `src/tensor/impls.rs` | View creation methods: view() / view_mut() | W8T6 | 07-tensor §5, §6, §7, §8 |
| W8T10 | `tests/test_tensor.rs` | Integration tests: cross-module interaction, boundary tests, type alias compilation | W8T3, W8T9 | 07-tensor §5, §6, §7, §8 |

### W9: Workspace (L2)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W9T1 | `src/error.rs` | WorkspaceErrorCategory definition, wired into XenonError::Workspace | W2T2 | 24-workspace §5, §6, §7, §8 |
| W9T2 | `src/workspace/mod.rs` | Module root: sub-module declarations, re-exports | W9T1 | 24-workspace §3, §4, §7 T3 |
| W9T3 | `src/workspace/workspace.rs` | Workspace struct, constants, new(), with_default_capacity(), Drop | W9T1, W9T2 | 24-workspace §5.1–§5.4, §7 T2, §8 |
| W9T4 | `src/workspace/borrow.rs` | WorkspaceBorrow/WorkspaceBorrowMut guards + borrow/borrow_mut + MaybeUninit access methods + Drop | W9T3, W9T1 | 24-workspace §5.5–§5.6, §6, §7 T4, §8 |
| W9T5 | `src/workspace/split.rs` | SplitBorrowMut guard, split_at_mut (top-level + recursive), Drop | W9T3, W9T1, W9T4 | 24-workspace §5.7, §6.2–§6.3, §7 T5, §8 |
| W9T6 | `src/workspace/expand.rs` | ensure_capacity() / reallocate() expansion strategy | W9T3, W9T1, W9T4 | 24-workspace §5.8, §6.4, §7 T6, §8 |
| W9T7 | `src/workspace/mod.rs` + sub-modules | Complete public exports + doc comments + cargo doc verification | W9T4, W9T5, W9T6 | 24-workspace §5–§9, §7 T7 |

**编号说明**：W9T2/W9T3 编号与设计文档 `24-workspace.md §7` 的 T2/T3 **反向**（设计 T2=workspace.rs/T3=mod.rs；本 SUMMARY W9T2=mod.rs/W9T3=workspace.rs）。**理由**：批次2 中 mod.rs 拓扑应先于 workspace.rs，便于后续 `use super::workspace` 路径解析。

### W10: Dispatch (L4)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W10T1 | `src/dispatch.rs` | Module skeleton: ExecPath enum, ParallelExecStrategy struct, module docs | None | 30-dispatch §5, §6, §7, §8 |
| W10T2 | `src/dispatch.rs` | ParallelGuard type + try_acquire_guard() + with_parallel_worker_context(): thread_local Cell\<bool\>, Drop impl, panic-safe worker context helper | W10T1 | 30-dispatch §5, §6, §7, §8 |
| W10T3 | `src/dispatch.rs` | ParallelExecStrategy::new() validating constructor, auto() infallible default, field accessors | W10T1 | 30-dispatch §5, §6, §7, §8 |
| W10T4 | `src/dispatch.rs` | select_exec_path() + should_parallelize(): three-way dispatch, threshold storage + getters (DEFAULT_* consts, AtomicUsize), feature gate branching | W10T1, W10T2 | 30-dispatch §5, §6, §7, §8 |
| W10T5 | `src/dispatch.rs` | Threshold config: set/reset_parallel_threshold, set/reset_simd_threshold, Relaxed ordering | W10T4 | 30-dispatch §5, §6, §7, §8 |
| W10T6 | `src/dispatch.rs` (#[cfg(test)]) | Full dispatch unit tests: path return, threshold boundaries, feature gate combos, nested guard, non-contiguous penalty | W10T4, W10T5, W10T2, W10T3 | 30-dispatch §5, §6, §7, §8 |

### W11: Broadcasting (L4)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W11T1 | `src/broadcast/mod.rs` | Module skeleton: sub-module declarations, re-exports | None | 15-broadcast §5, §6, §7, §8 |
| W11T2 | `src/broadcast/shape.rs` | Module skeleton: file placeholder, rule function stubs | W11T1 | 15-broadcast §5, §6, §7, §8 |
| W11T3 | `src/broadcast/view.rs` | Module skeleton: file placeholder, view entry stubs | W11T1 | 15-broadcast §5, §6, §7, §8 |
| W11T4 | `src/broadcast/shape.rs` | can_broadcast(): trailing-axis alignment compatibility check | W11T2 | 15-broadcast §5, §6, §7, §8 |
| W11T5 | `src/broadcast/shape.rs` | broadcast_shape(): shared shape derivation with structured broadcast errors | W11T4 | 15-broadcast §5, §6, §7, §8 |
| W11T6 | `src/broadcast/shape.rs` | broadcast_strides(): zero-stride insertion with input precondition validation | W11T5 | 15-broadcast §5, §6, §7, §8 |
| W11T7 | `src/broadcast/view.rs` | broadcast_to() basic path: target shape validation + read-only view construction | W11T3, W11T6 | 15-broadcast §5, §6, §7, §8 |
| W11T8 | `src/broadcast/view.rs` | broadcast_to() error path + BroadcastView layout state update | W11T7 | 15-broadcast §5, §6, §7, §8 |
| W11T9 | `src/broadcast/view.rs` | broadcast_with(): shared shape derivation, dual-input broadcast | W11T6, W11T7 | 15-broadcast §5, §6, §7, §8 |
| W11T10 | `tests/test_broadcast.rs` | Integration tests | W11T8, W11T9, W12T*, W22T* | 15-broadcast §5, §6, §7, §8 |

### W12: Iterators (L4)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W12T1 | `src/iter/mod.rs` | Module skeleton: declarations, sub-module placeholders, public exports | None | 10-iterator §5, §6, §7, §8 |
| W12T2 | `src/iter/elements.rs` | StrideState: F-order index increment state machine | W12T1 | 10-iterator §5, §6, §7, §8 |
| W12T3 | `src/iter/elements.rs` | Iter: Iterator + ExactSizeIterator impls with fast/slow paths (contiguous vs non-contiguous) | W12T2 | 10-iterator §5, §6, §7, §8 |
| W12T4 | `src/iter/elements.rs` | IterMut: Iterator + ExactSizeIterator impls | W12T3 | 10-iterator §5, §6, §7, §8 |
| W12T5 | `src/iter/axis.rs` | AxisIter / AxisIterMut: iterate along one axis, yielding sub-views | W12T1 | 10-iterator §5, §6, §7, §8 |
| W12T6 | `src/iter/indexed.rs` | IndexedIter / IndexedIterMut: index-wrapped iteration based on Iter | W12T4 | 10-iterator §5, §6, §7, §8 |
| W12T7 | `src/iter/mod.rs` | TensorBase entry methods: iter(), iter_mut(), axis_iter(), indexed_iter() | W12T4, W12T5, W12T6 | 10-iterator §5, §6, §7, §8 |

### W13: FFI Helpers (L4)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W13T1 | `src/ffi/mod.rs` | Module skeleton: 5 sub-module declarations (types / private / ptr / blas / offset) + placeholder files; re-exports added by downstream tasks per W1T3 module-evolution protocol | W8T7 | 23-ffi §5, §6, §7, §8 |
| W13T2 | `src/ffi/types.rs` | C-visible raw descriptors (TensorExportRaw / TensorExportMutRaw) + BlasInfo struct + ElementType / FfiErrorCategory / FfiBackend re-exports | W13T1 | 23-ffi §5, §6, §7, §8 |
| W13T3 | `src/ffi/private.rs` | Internal generic descriptors: TensorExport and TensorExportMut | W13T2 | 23-ffi §5, §6, §7, §8 |
| W13T4 | `src/ffi/ptr.rs` | Inherent methods export() / export_mut() on TensorBase + re-export OwnedRawParts / TensorBase | W13T3, W22T10 | 23-ffi §5, §6, §7, §8 |
| W13T5 | `src/ffi/blas.rs` | is_blas_layout_compatible(), blas_info(), lda() | W13T2 | 23-ffi §5, §6, §7, §8 |
| W13T6 | `src/ffi/offset.rs` | try_offset_of() / try_ptr_at() with checked arithmetic validation | W13T2 | 23-ffi §5, §6, §7, §8 |

### W14: SIMD Backend (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W14T0 | `docs/spike/pulp-api-survey.md` | pulp 0.18 API capability spike: i32→i64 widening, complex AoS deinterleave/shuffle, FMA capability bit, unaligned load/store, lane width — report availability + fallback per API | None | 08-simd §5.1, §6.1, §7 |
| W14T1 | `src/simd/mod.rs`, `src/lib.rs` | Module skeleton: SimdElement sealed marker, SimdKernel trait, BinaryOp + UnaryOp enums, Arch cache, facade 统一表 (`dispatch_vector_binary_op`/`dispatch_vector_unary_op` → bool; `try_sum_*`/`try_dot_*` → Option<A>); register `pub(crate) mod simd;` in lib.rs | W14T0 | 08-simd §5, §6, §7, §8, §10 |
| W14T2 | `src/simd/vector.rs` | Element-wise SIMD: Add/Sub/Mul/Div Kernel WithSimd impls for f32/f64 + independent Neg unary Kernel; FMA forbidden in per-element main loop | W14T0, W14T1 | 08-simd §5.5, §5.6, §5.8, §6.1, §6.6, §7, §8.2, §10 |
| W14T3 | `src/simd/vector.rs` | Float sum SIMD: f32/f64 SumKernel with pairwise lane accumulation; tolerance per 13-reduction §6.3 | W14T2 | 08-simd §5.6, §5.8, §6.6, §7, §8.2, §10 |
| W14T4 | `src/simd/vector.rs` | Integer sum/dot SIMD admission + fallback: i32 widening primary path (per W14T0 spike); i64 default scalar fallback only (checked_add/checked_mul) | W14T0, W14T3 | 08-simd §5, §6, §7, §8, §10 |
| W14T5 | `src/simd/vector.rs` | Complex sum SIMD: Complex\<f32\>/Complex\<f64\> AoS→SoA deinterleave + split real/imag pairwise accumulation; **Complex sum threshold = 1024** (per PLAN.md W14 补充决策) | W14T0, W14T3 | 08-simd §5, §6, §7, §8, §10 |
| W14T6 | `src/simd/vector.rs` | Float + complex dot SIMD: f32/f64/Complex\<f32\>/Complex\<f64\> dot kernel; BLAS xdotc conjugate contract `sum(conj(lhs_i)*rhs_i)`; **Complex dot threshold = 512** (per PLAN.md W14 补充决策); reuses W14T5 split accumulator | W14T0, W14T3, W14T5 | 08-simd §5, §6, §7, §8, §10 |
| W14T7 | `src/simd/mod.rs`, `Cargo.toml` | Feature gate conditional compilation: #[cfg(feature = "simd")] guards all items including trait definitions; pub(crate) facade export (no lib.rs modification — W14T1 owns lib.rs registration); W10 ExecPath::Simd dispatch integration contract | W14T0, W14T1, W14T2 | 08-simd §5, §6, §7, §8, §10 |
| W14T8 | `src/simd/vector.rs` (#[cfg(all(test, feature = "simd"))]) | Element-wise consistency tests: SIMD vs serial bitwise agreement for Add/Sub/Mul/Div/Neg on f32+f64 + Complex\<f32\>/Complex\<f64\>; boundary lengths (0/1/below/at/tail/SIMD_WIDTH+k); NaN-aware comparison | W14T1, W14T2, W14T11, W14T7 | 08-simd §5, §6, §7, §8, §10 |
| W14T9 | `src/simd/vector.rs` (#[cfg(all(test, feature = "simd"))]) | Reduction/dot semantic + tolerance tests: entry threshold boundaries (1023/1024 etc.), ISA gating fallback, tolerance per 13-reduction §6.3 + 12-matrix dot bound, complex component-wise comparison, i32 admission tests (i64 has no SIMD facade — not in test scope) | W14T3, W14T4, W14T5, W14T6, W14T8 | 08-simd §5, §6, §7, §8, §10 |
| W14T10 | `tests/simd_property.rs` | Randomized property tests via **public Tensor API** (no pub(crate) imports); Cargo top-level discovery; deterministic splitmix64 PRNG (no proptest/quickcheck dep); element-wise consistency + sum/dot tolerance + complex dot conjugate + i32 no-panic + tail handling + ISA fallback; all of f32/f64/Complex\<f32\>/Complex\<f64\> | W14T2, W14T11, W14T6, W14T7, W14T9 | 08-simd §5, §6, §7, §8.4, §8.5, §10; 28-tests §6.4.2 |
| W14T11 | `src/simd/vector.rs` | Complex element-wise SIMD: Add/Sub/Mul/Div/Neg kernels for Complex\<f32\>/Complex\<f64\>; reuses W14T5 AoS deinterleave; threshold = 128 (08-simd §5.8 L451); FMA forbidden in element-wise main loop. | W14T0, W14T2, W14T5 | 08-simd §5.2, §5.5, §5.6, §5.8 L451, §6.1, §6.6, §8.2, §10 |

### W15: Parallel Backend (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W15T1 | `src/parallel/iter.rs`, `src/parallel/mod.rs` (skeleton + `compute_safe_chunks`), `src/lib.rs` (`pub(crate) mod parallel`) | Module skeleton + ParIter + TensorBase::par_iter() + compute_safe_chunks: parallel module entry, single-input element-level parallel traversal, and shared chunk policy | W10T4 | 09-parallel §3, §5.1, §5.6, §6.3, §6.7, §8.2 |
| W15T2 | `src/parallel/map.rs` | par_map: pure parallel element-wise map entry, strategy from dispatch | W15T1 | 09-parallel §5.4, §5.5, §6.2, §6.3, §6.7, §8.2 |
| W15T3 | `src/parallel/map.rs` | par_zip_map: dual-input broadcast element-wise parallel entry for math consumption | W15T2, W11T7, W21T3 | 09-parallel §5.5, §6.3, §6.7, §8.2, §10 |
| W15T4 | `src/parallel/reduce.rs` | par_reduce_impl + par_sum: parallel reduction, identity merge, semantic alignment with caller's serial baseline | W15T1 | 09-parallel §5.5, §6.3, §6.5, §6.7, §8.2 |
| W15T5 | `src/parallel/reduce.rs` | par_dot: ndim==1 check (InvalidArgument), length consistency (ShapeMismatch), parallel inner product, error return + empty identity | W15T4 | 09-parallel §5.5, §6.5, §6.7, §8.2, §8.3, §10 |
| W15T6 | `src/parallel/mod.rs` (increment over W15T1 skeleton) | ParallelPool: custom rayon::ThreadPool wrapper, preserving public API result semantics; nested-pool TLS guard | W15T2, W15T4, W15T5 | 09-parallel §5.1, §6.7, §8.2 |
| W15T7 | `src/parallel/checked.rs` | par_map_checked + error/panic propagation: XenonError passthrough via two-pass pattern, panic not swallowed | W15T2, W15T4, W15T5 | 09-parallel §5.5, §6.6, §6.7, §8.2, §10 |
| W15T8 | `src/parallel/` all files (feature gate audit) + `src/parallel/mod.rs` (#[cfg(test)] inline + compile_fail doctest) | Feature gate + config matrix tests: default off (no rayon dep), --features parallel build, single/multi-worker via `ParallelExecStrategy::new(None, Some(1))` vs `auto()`. External `tests/test_parallel.rs` deferred to W29 when public APIs (W16/W18) are ready. | W15T1–W15T7 | 09-parallel §5.1, §8.5, §8.6, §8.7 |

### W16: Math Operations (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W16T1 | `src/math/mod.rs`, `src/math/helpers.rs` | Module skeleton: declarations, shared element-wise helpers (apply_unary, apply_complex_to_real), public API re-exports | None | 11-math §5, §6, §7, §8 |
| W16T2 | `src/math/binary.rs` | Shared binary element-wise execution skeleton with broadcast support | W16T1 | 11-math §5, §6, §7, §8 |
| W16T3 | `src/math/unary.rs` | Unary element-wise ops: abs, neg, signum, square (with checked arithmetic for integers) | W16T1 | 11-math §5, §6, §7, §8 |
| W16T4 | `src/math/unary.rs` | Math functions for RealScalar: sin, sqrt, exp, ln, floor, ceil | W16T1 | 11-math §5, §6, §7, §8 |
| W16T5 | `src/math/unary.rs` | Complex ops: conjugate, modulus | W16T1 | 11-math §5, §6, §7, §8 |
| W16T6 | `src/math/binary.rs` | Arithmetic ops: add, sub, mul, div (tensor-tensor + scalar + `pub(crate)` left-scalar variants `sub_from_scalar` / `div_from_scalar` per 11-math §5.9) via shared binary skeleton | W16T2 | 11-math §5, §6, §7, §8 |
| W16T7 | `src/math/unary.rs` | Logical not for bool tensors | W16T1 | 11-math §5, §6, §7, §8 |
| W16T8 | `src/math/comparison.rs` | Equal + not_equal comparison ops (tensor-tensor + scalar variants, return bool tensors) | W16T2 | 11-math §5, §6, §7, §8 |
| W16T9 | `src/math/comparison.rs` | Less + less_equal comparison ops (tensor-tensor + scalar variants, return bool tensors) | W16T8 | 11-math §5, §6, §7, §8 |
| W16T10 | `src/math/comparison.rs` | Greater + greater_equal comparison ops (tensor-tensor + scalar variants, return bool tensors) | W16T8 | 11-math §5, §6, §7, §8 |
| W16T11 | `src/math/binary.rs`, `unary.rs`, `comparison.rs`, `src/simd/vector.rs` | SIMD/parallel dispatch integration for math module (unary + binary + comparison entries) | W16T3, W16T4, W16T5, W16T6, W16T7, W16T8, W16T9, W16T10, W14, W15 | 11-math §5, §6, §7, §8 |

### W17: Matrix Operations (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W17T1 | `src/matrix/mod.rs`, `src/matrix/dot.rs`, `src/lib.rs` | Module skeleton: matrix module + dot.rs compilable stub returning `Ok(A::zero())`, crate-root re-export | None | 12-matrix §5, §6, §7, §8 |
| W17T2 | `src/matrix/dot.rs` | Input validation: rank/length checks returning `InvalidArgument` / `ShapeMismatch` | W17T1 | 12-matrix §5, §6, §7, §8 |
| W17T3 | `src/matrix/dot.rs` | Scalar inner product via sealed `DotAccumulate` trait (`pub`, sealed via Sealed chain): float/complex via `acc + x.conjugate() * y`, integer via `CheckedMul`+`CheckedAdd` with typed panic diagnostics, `TensorBase::dot` method | W17T2, W12T7 | 12-matrix §5, §6, §7, §8 |
| W17T4 | `src/matrix/dot.rs` | Dispatch wiring: `alignment_ok(a, b)` private helper + `let (path, guard) = select_exec_path(...)` tuple destructure with all arms falling back to scalar | W17T3 | 12-matrix §5, §6, §7, §8 |
| W17T5 | `src/matrix/dot.rs` | SIMD path: `can_use_simd_dot` admission gate + per-type dispatch to `simd::try_dot_*` slice API, scalar fallback on `None` | W17T4, W14 | 12-matrix §5, §6, §7, §8 |
| W17T6 | `src/matrix/dot.rs` | Parallel path: route `ExecPath::Parallel` to `parallel::par_dot(lhs, rhs, &strategy, guard)` using `ParallelExecStrategy::auto()`, nested parallel fallback test; reuses `f64_dot_tolerance` helper defined by W17T5 | W17T4, W17T5, W15 | 12-matrix §5, §6, §7, §8 |
| W17T7 | `tests/test_matrix.rs` | Integration tests covering full §8.2 14-test matrix incl. tolerance + NaN cross-path | W17T3–W17T6 | 12-matrix §5, §6, §7, §8 |

### W18: Reduction Operations (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W18T1 | `src/reduction/mod.rs`, `src/reduction/sum.rs`, `src/lib.rs` | Module skeleton: empty sum.rs placeholder + public API exports (sum family) + crate-root wiring | None | 13-reduction §5, §6, §7, §8 |
| W18T2 | `src/reduction/sum.rs` | sum() + `checked_add_step` TypeId-dispatched helper: full traversal, integer checked arithmetic, empty array zero semantics, NaN / complex NaN propagation tests | W18T1 | 13-reduction §5, §6, §7, §8 |
| W18T3 | `src/reduction/sum.rs` | sum_axis() + `validate_axis` pub(crate) helper: axis validation, output shape reduction, per-axis slot accumulation via `try_at_mut`, zero-length reduced axis test | W18T2 | 13-reduction §5, §6, §7, §8 |
| W18T4 | `src/reduction/sum.rs` | sum_axis_keepdims() + local `dim_with_axis_set` helper: preserve rank with reduced axis length 1, reuses `validate_axis` / `checked_add_step`, zero-length keepdims test | W18T3 | 13-reduction §5, §6, §7, §8 |
| W18T5 | `src/reduction/sum.rs` | SIMD / parallel dispatch guards: integer type gate before `select_exec_path`, TypeId-dispatched W14 `try_sum_*` facades, `par_sum` integration, §6.3 tolerance-based consistency tests | W18T4, W14, W15 | 13-reduction §5, §6, §7, §8 |
| W18T6 | `tests/test_reduction.rs` | Integration tests: axis OOB → InvalidAxis on sum_axis/sum_axis_keepdims, integer overflow → panic on sum/sum_axis/sum_axis_keepdims, IEEE 754 Inf, high-rank IxDyn shape semantics, 10^7-class parallel-path §6.3 tolerance | W18T3–W18T5 | 13-reduction §5, §6, §7, §8 |

### W19: Set Operations (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W19T1 | `src/lib.rs, src/set/mod.rs` | Module skeleton: crate root wiring, unique module declaration, and forward re-exports following existing skeleton pattern | None | 01-architecture §3, 14-set §3, §7 |
| W19T2 | `src/set/unique.rs` | UniqueElement trait definition and real scalar impls (i32/i64/f32/f64) | W19T1 | 14-set §5, §6, §7, §8 |
| W19T3 | `src/set/unique.rs` | unique_impl(): logical F-order element collection, equality-based deduplication, Tensor construction | W19T2, W12T7 | 14-set §5, §6, §7, §8 |
| W19T4 | `src/set/unique.rs` | Float NaN / ±0.0 behavior tests (test-only): preserve each NaN, treat -0.0 and 0.0 as equal | W19T3 | 14-set §5, §6, §7, §8 |
| W19T5 | `src/set/unique.rs` | Complex component-wise equality using direct real/imag `==` per design §6.4, no ordering | W19T3 | 14-set §5, §6, §7, §8 |
| W19T6 | `src/set/unique.rs, src/prelude.rs` | unique() entry method on TensorBase, prelude re-export, remaining in-module unit tests | W19T3–W19T5 | 14-set §5, §6, §7, §8 |

### W20: Shape Operations (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W20T1 | `src/shape/mod.rs`, `src/lib.rs` | Module skeleton: declare transpose sub-module, wire `pub mod shape;` in lib.rs | None | 16-shape §5, §6, §7, §8 |
| W20T2 | `src/shape/transpose.rs` | Module skeleton: file placeholder, declarations | W20T1 | 16-shape §5, §6, §7, §8 |
| W20T3 | `src/shape/transpose.rs` | transpose(): axis swap, O(1) shape/stride recomputation | W20T2 | 16-shape §5, §6, §7, §8 |
| W20T4 | `tests/test_shape.rs`, `tests/property/test_shape.rs` | Integration tests (含大数组 O(1)、Ix6、IxDyn、ArcRepr 降级、view+index 组合) + property tests (§8.4 三项不变量) | W20T3 | 16-shape §5, §6, §7, §8 |

### W21: Indexing (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W21T1 | `src/index/mod.rs` | Module skeleton: declarations for ndindex/access/slice and public re-exports | None | 01-architecture §3, 17-indexing §5, §6, §7, §8 |
| W21T2 | `src/index/ndindex.rs` | NdIndex\<D\> trait + tuple/slice index legality check: rank match, per-axis bounds, offset calculation | W21T1 | 17-indexing §5, §6, §7, §8 |
| W21T3 | `src/index/access.rs` | try_at / get / get_unchecked: unified safe + unsafe read access paths | W21T2 | 17-indexing §5, §6, §7, §8 |
| W21T4 | `src/index/slice.rs` | SliceInfoElem + SliceInfoIndices: inline/dynamic slice descriptor representations | W21T2 | 17-indexing §5, §6, §7, §8 |
| W21T5 | `src/index/access.rs` | try_at_mut / get_mut / get_unchecked_mut: mutable access, gated on StorageMut | W21T3 | 17-indexing §5, §6, §7, §8 |
| W21T6 | `src/index/slice.rs` | slice() shape/stride update + layout recomputation: axis folding, Range→shape/stride, read-only view return | W21T4 | 17-indexing §5, §6, §7, §8 |

### W22: Tensor Construction (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W22T1 | `src/construct/mod.rs`, `src/construct/init.rs`, `src/construct/eye.rs`, `src/construct/from.rs`, `src/construct/scalar.rs`, `src/lib.rs` | Module skeleton: sub-module declarations + empty placeholder files (impl blocks for constructors are added by W22T2–W22T8; only `pub use eye::EyeElement;` re-export is deferred to W22T4 since all other constructors are `TensorBase` associated functions and auto-visible via impl blocks) | None | 18-construction §5, §6, §7, §8 |
| W22T2 | `src/construct/init.rs` | zeros() constructor | W22T1 | 18-construction §5, §6, §7, §8 |
| W22T3 | `src/construct/init.rs` | ones() constructor | W22T1 | 18-construction §5, §6, §7, §8 |
| W22T4 | `src/construct/eye.rs` | eye(): identity matrix constructor | W22T1, W22T2 | 18-construction §5, §6, §7, §8 |
| W22T5 | `src/construct/from.rs` | from_shape_vec + from_vec: consume Vec into Owned path, 1D convenience | W22T1 | 18-construction §5, §6, §7, §8 |
| W22T6 | `src/construct/from.rs` | from_shape_slice: copy from slice into Owned storage | W22T5 | 18-construction §5, §6, §7, §8 |
| W22T7 | `src/construct/from.rs` | from_array: fixed-array construction | W22T6 | 18-construction §5, §6, §7, §8 |
| W22T8 | `src/construct/scalar.rs` | from_scalar: zero-dim tensor constructor | W22T1 | 18-construction §5, §6, §7, §8 |
| W22T9 | `tests/test_construction.rs` | Integration tests | W22T2–W22T8 | 18-construction §5, §6, §7, §8 |
| W22T10 | `src/tensor/construct.rs`, `src/tensor/mod.rs` | OwnedRawParts struct + into_raw_parts() + from_raw_parts_owned() — Owned round-trip carrier (Rust-only, **non-C-ABI**); 5-gate validation (offset==0 / shape product / cap>=len / align valid / canonical F-order); routes to new_unchecked. Required by W13T4 for `pub use crate::tensor::OwnedRawParts`. | W8T7, W8T8；W22T5（测试构造器） | 07-tensor §5.7 (line 890-1086), §6.2, §6.3; 23-ffi §5.8 |

### W23: Operator Overloading (L6)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W23T1 | `src/overload/mod.rs` (+ placeholder `src/overload/arithmetic.rs`) | Module skeleton: `pub mod arithmetic;` declaration + compile-time anchor test; create empty `arithmetic.rs` so the module declaration resolves. `pub use arithmetic::Scalar;` is intentionally deferred to W23T2 (when `Scalar` is first declared) to keep this task's `cargo check` green | None | 01-architecture §3, 19-overload §5, §6, §7, §8 |
| W23T2 | `src/overload/arithmetic.rs` (+ `src/overload/mod.rs` backfill) | Imports for operators / math / broadcast / tensor / element / error / storage / complex; `Scalar<A>` declaration (no derives — strictly per §5.3 line 273); backfill `pub use arithmetic::Scalar;` into `mod.rs` and add the reexport unit test | W23T1 | 19-overload §5, §6, §7, §8 |
| W23T3 | `src/overload/arithmetic.rs` | `Add<TensorBase<Owned<A>,E>> for TensorBase<Owned<A>,D>` (owned + owned) with symmetric BroadcastDim where bound; delegates to inherent `TensorBase::add(&rhs)` | W23T2 | 19-overload §5, §6, §7, §8 |
| W23T4 | `src/overload/arithmetic.rs` | Add for ref/mixed owned combos: `&T+&T`, `T+&T`, `&T+T` (3 combos), all delegating to inherent `TensorBase::add` | W23T3 | 19-overload §5, §6, §7, §8 |
| W23T5 | `src/overload/arithmetic.rs` | Add scalar paths (owned `Tensor` only): `T+A`, `&T+A`, `Scalar<A>+T`, `Scalar<A>+&T`, native `T+Tensor<T,D>` and `T+&Tensor<T,D>` per-type for {f32,f64,i32,i64,Complex<f32>,Complex<f64>} | W23T3 | 19-overload §5, §6, §7, §8 |
| W23T6 | `src/overload/arithmetic.rs` | Sub operators for owned tensor: tensor×tensor (owned/ref/mixed) + scalar (right / `Scalar<A>` left / native left per-type); non-commutative left scalar delegates to `sub_from_scalar` | W23T4, W23T5 | 19-overload §5, §6, §7, §8 |
| W23T7 | `src/overload/arithmetic.rs` | Mul operators for owned tensor: tensor×tensor + scalar; commutative — left scalar reuses `mul_scalar` | W23T4, W23T5 | 19-overload §5, §6, §7, §8 |
| W23T8 | `src/overload/arithmetic.rs` | Div operators for owned tensor: tensor×tensor + scalar; non-commutative left scalar delegates to `div_from_scalar` | W23T4, W23T5 | 19-overload §5, §6, §7, §8 |
| W23T9 | `src/overload/arithmetic.rs` | `TensorView` tensor×tensor combos for Add/Sub/Mul/Div: `&View+&View`, `&View+&T`, `&T+&View` (3 combos × 4 operators = 12 impls); per §5.1 lines 123–125 | W23T6, W23T7, W23T8 | 19-overload §5, §6, §7, §8 |
| W23T10 | `src/overload/arithmetic.rs` | `TensorView` scalar paths for Add/Sub/Mul/Div: right scalar + `Scalar<A>` left + native left per-type (4 operators × {right2 + ScalarLeft2 + nativeLeft12} = 64 impls); per §5.1 lines 133–136 & §5.4 lines 391–467 | W23T9 | 19-overload §5, §6, §7, §8 |
| W23T11 | `tests/test_overload.rs` | Integration tests: broadcast combos, scalar combos (right/wrapper/native, owned+ref), non-commutative left-scalar (Sub/Div), type combos (f64/i32/Complex), deep-copy verification; integration-test-only (no `mod tests` wrapper, no `crate::` paths). Owned Tensor path only — TensorView coverage stays in W23T9/W23T10 unit tests | W23T1–W23T8 | 19-overload §5, §6, §7, §8 |

### W24: Utility Operations (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W24T1 | `src/util/mod.rs` | Module skeleton: submodule declarations (no `pub use`; algorithms exposed as inherent methods) | W1T3 | 20-utility §5, §6, §7, §8 |
| W24T2 | `src/util/fill.rs` | fill(): StorageMut-level fill helper + try_fill() sealed-marker dispatch for all tensor types | W24T1, W7T6, W7T18, W7T19, W8T4, W12T7, W2T1 | 20-utility §5, §6, §7, §8 |
| W24T3 | `src/util/clip.rs` | clip(): element-wise clipping with NaN/min=max/NaN-bound/Integer error handling | W24T1, W4, W8, W12T1, W2T1 | 20-utility §5, §6, §7, §8 |
| W24T4 | `src/util/contiguous.rs` | to_contiguous() + into_contiguous(): F-contiguous guarantee, reuse or repack | W24T1, W6T6, W6T7, W6T11, W7T18, W7T19, W8T4, W8T6, W8T7, W12T1, W25T7 | 20-utility §5, §6, §7, §8 |
| W24T5 | `tests/test_utility.rs` | Integration tests: boundary cases (empty, single-element, non-contiguous, zero-dim, large) for all utility ops | W24T2, W24T3, W24T4, W22T8 | 20-utility §5, §6, §7, §8 |

### W25: Type Conversion (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W25T1 | `src/convert/mod.rs`, `src/convert/cast.rs` (placeholder), `src/lib.rs` | Module skeleton: `mod cast;` declaration, placeholder `cast.rs`, `pub mod convert;` in `lib.rs`. `pub(crate) use cast::ConvertTo;` is deferred to W25T2 so W25T1 compiles standalone. | None | 21-type §5, §6, §7, §8 |
| W25T2 | `src/convert/cast.rs`, `src/convert/mod.rs` | ConvertTo sealed trait signature (`pub(crate) trait ConvertTo<B>: CastElement + Sealed`) + CastTo ownership doc + `pub(crate) use cast::ConvertTo;` in mod.rs. Trait signatures only, no impls. | W25T1 | 21-type §5, §6, §7, §8 |
| W25T3 | `src/convert/cast.rs` | Tier-1 lossless ConvertTo impls: 6 identity + 3 std From arithmetic widening + 4 real→complex zero-imag widening + 1 complex→complex widening (14 cells) | W25T2 | 21-type §5, §6, §7, §8 |
| W25T4 | `src/convert/cast.rs` | Tier-2 lossy CastTo + ConvertTo impls: float↔int, int↔int narrowing, int→narrow float, real→complex lossy, complex narrowing (14 cells with reason mapping table) | W25T2 | 21-type §5, §6, §7, §8 |
| W25T5 | `src/convert/cast.rs` | Tier-3 dynamic CastTo + ConvertTo impls: Complex\<f32/f64\> → {i32,i64,f32,f64} (8 cells); inner path branches per (src,tgt): identity / `From` widening / Tier-2 delegation | W25T2, W25T4 | 21-type §5, §6, §7, §8 |
| W25T6 | `src/convert/cast.rs` | cast\<B\>() method on TensorBase\<S, D\>: all readable storage inputs → owned result with error reporting | W25T3, W25T4, W25T5, W12T7 | 21-type §5, §6, §7, §8 |
| W25T7 | `src/convert/cast.rs` | to_owned() + into_owned(): clone and consume owned conversion methods; includes self-contained `pub(crate) fn into_owned_from_owned_storage` helper (not provided by 07-tensor.md / W8) | W25T1, W7T5, W12T7 | 21-type §5, §6, §7, §8 |

### W26: Output Formatting (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W26T1 | `src/format/mod.rs`, `src/lib.rs` | Module skeleton: crate-root declaration + format module root (no submodule declarations; submodules declared incrementally by W26T2-W26T5; re-exports finalized by W26T6) | None | 22-output §5, §6, §7, §8 |
| W26T2 | `src/format/config.rs`, `src/format/mod.rs` (append `mod config;`) | FormatConfig struct + Default impl | W26T1 | 22-output §5, §6, §7, §8 |
| W26T3 | `src/format/pretty.rs`, `src/format/mod.rs` (append `mod pretty;`) | Numpy-style formatting helpers: fmt_1d_display, fmt_1d_debug, fmt_nd_display, fmt_nd_debug with truncation + line_width soft-wrap (via module-private LineWriter column tracker) | W26T2 | 22-output §5, §6, §7, §8 |
| W26T4 | `src/format/display.rs`, `src/format/mod.rs` (append `mod display;`) | core::fmt::Display for TensorBase\<S, D\> + TensorDisplay\<'a, S, D, A\> wrapper + display_with(config) entry method | W26T3 | 22-output §5, §6, §7, §8 |
| W26T5 | `src/format/debug.rs`, `src/format/mod.rs` (append `mod debug;`) | core::fmt::Debug for TensorBase\<S, D\> with metadata | W26T3 | 22-output §5, §6, §7, §8 |
| W26T6 | `src/format/mod.rs` | Module docs + re-exports completion (`pub use config::FormatConfig;` + `pub use display::TensorDisplay;`) | W26T4, W26T5 | 22-output §5, §6, §7, §8 |

### W27: Safety Audit (cross-cutting)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W27T1 | `src/storage/owned.rs` | Audit and strengthen SAFETY comments for Owned\<A\> Send + Sync impls | W7T12 | 25-safety §5, §6, §7, §8 |
| W27T2 | `src/storage/view.rs` | Audit and strengthen SAFETY comments for ViewRepr\<'a, A\> Send + Sync impls | W7T14 | 25-safety §5, §6, §7, §8 |
| W27T3 | `src/storage/viewmut.rs` | Audit and strengthen SAFETY comments for ViewMutRepr\<'a, A\> Send (no Sync) | W7T15 | 25-safety §5, §6, §7, §8 |
| W27T4 | `src/storage/arc.rs` | Audit and strengthen SAFETY comments for ArcRepr\<A\> Send + Sync impls | W7T17, W7T12 (tests use into_shared) | 25-safety §5, §6, §7, §8 |
| W27T5 | `src/parallel/iter.rs` | Parallel execution chunk safety: completeness, non-overlap, boundary tests | W27T1–W27T4, W15T1 (compute_safe_chunks) | 25-safety §5, §6, §7, §8 |
| W27T6 | `tests/test_parallel.rs`, `tests/test_error.rs` | Thread-safety integration tests: cross-thread transfer, concurrent access | W27T1–W27T5 | 25-safety §5, §6, §7, §8 |
| W27T7 | `src/storage/mod.rs` | Module-level thread-safety docs, Send/Sync matrix, cargo doc pass | W27T1–W27T4 | 25-safety §5, §6, §7, §8 |

### W28: Benchmarks

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W28T1 | `Cargo.toml` | Add 9 [[bench]] entries, no new benchmark-specific third-party deps | None | 27-benchmark §5, §6, §7, §8 |
| W28T2 | `benches/utils/mod.rs`, `benches/utils/generators.rs` | Shared constants (SIZES_1D, SIZES_2D) + data generation functions | W28T1 | 27-benchmark §5, §6, §7, §8 |
| W28T3 | `benches/math.rs` | Element-wise math benchmarks: add/sub/mul/div/sin/exp/abs (f32/f64/Complex\<f64\>, contiguous + non-contiguous) | W28T2 | 27-benchmark §5, §6, §7, §8 |
| W28T4 | `benches/reduction.rs` | Reduction benchmarks: sum_1d_f64, sum_2d_axis0, sum_2d_axis1, sum_sliced, sum_2d_keepdims | W28T2 | 27-benchmark §5, §6, §7, §8 |
| W28T5 | `benches/dot.rs` | Dot benchmarks: dot_1d_f64, dot_1d_complex | W28T2 | 27-benchmark §5, §6, §7, §8 |
| W28T6 | `benches/set.rs` | Set benchmarks: unique_1d (varying sizes, unique ratio) | W28T2 | 27-benchmark §5, §6, §7, §8 |
| W28T7 | `benches/broadcast.rs` | Broadcast benchmarks: broadcast_scalar, broadcast_row, broadcast_col, broadcast_with | W28T2 | 27-benchmark §5, §6, §7, §8 |
| W28T8 | `benches/shape.rs` | Shape benchmarks: transpose_2d | W28T2 | 27-benchmark §5, §6, §7, §8 |
| W28T9 | `benches/construction.rs` | Construction benchmarks: zeros_1d, from_shape_vec_1d, eye_2d | W28T2 | 27-benchmark §5, §6, §7, §8 |
| W28T10 | `benches/simd_comparison.rs` | SIMD comparison: add/sum/dot with --features simd on/off | W28T3, W28T4, W28T5, W14 | 27-benchmark §5, §6, §7, §8 |
| W28T11 | `benches/parallel_comparison.rs` | Parallel comparison: sum/add/dot with --features parallel on/off | W28T3, W28T4, W28T5, W15 | 27-benchmark §5, §6, §7, §8 |
| W28T12 | `.github/workflows/bench.yml`, `tools/bench/report.py` | CI bench smoke test + benchmark report script: quick-mode execution, regression annotation | W28T3–W28T11 | 27-benchmark §5, §6, §7, §8 |

### W29: Integration Tests

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W29T1 | `tests/common/mod.rs`, `tests/common/assertions.rs`, `tests/common/generators.rs` | Test infrastructure: assert_tensor_exact_real/complex, real_bits_eq/real_ulp_eq, tolerance helpers, data generators | None | 28-tests §5, §6, §7, §8 |
| W29T2 | `tests/test_tensor.rs` | Core tensor tests: shape/strides/view/to_owned/type_aliases/debug_display/arc | W29T1 | 28-tests §5, §6, §7, §8 |
| W29T3 | `tests/test_math.rs` | Math tests: element-wise arithmetic/math/comparison/logical ops | W29T1 | 28-tests §5, §6, §7, §8 |
| W29T4 | `tests/test_overload.rs` | Overload tests: Add/Sub/Mul/Div traits, broadcast dispatch, Result ownership, scalar operators | W29T1, W29T3 | 28-tests §5, §6, §7, §8 |
| W29T5 | `tests/test_broadcast.rs` | Broadcast tests: scalar/row/col/incompatible/read-only broadcast | W29T1 | 28-tests §5, §6, §7, §8 |
| W29T6 | `tests/test_index.rs` | Index tests: multi-dim indexing, OOB errors, slicing, structural SliceInfo validation | W29T1 | 28-tests §5, §6, §7, §8 |
| W29T7 | `tests/test_construction.rs` | Construction tests: zeros/ones/eye/from_shape_vec/from_shape_slice/from_scalar/from_array | W29T1 | 28-tests §5, §6, §7, §8 |
| W29T8 | `tests/test_reduction.rs` | Reduction tests: sum/sum_axis/keepdims/empty/NaN/overflow | W29T1 | 28-tests §5, §6, §7, §8 |
| W29T9 | `tests/test_iterator.rs` | Iterator tests: elements/axis/indexed iteration | W29T1 | 28-tests §5, §6, §7, §8 |
| W29T10 | `tests/test_matrix.rs` | Matrix tests: dot/complex/shape mismatch | W29T1 | 28-tests §5, §6, §7, §8 |
| W29T11 | `tests/test_set.rs` | Set tests: unique (int/complex/NaN/±0.0/multiset, output order unspecified) | W29T1 | 28-tests §5, §6, §7, §8 |
| W29T12 | `tests/test_shape.rs` | Shape tests: transpose/high-dim | W29T1 | 28-tests §5, §6, §7, §8 |
| W29T13 | `tests/test_conversion.rs` | Conversion tests: cast/to_owned/into_owned | W29T1 | 28-tests §5, §6, §7, §8 |
| W29T14 | `tests/test_utility.rs` | Utility tests: fill/clip/to_contiguous | W29T1 | 28-tests §5, §6, §7, §8 |
| W29T15 | `tests/test_output.rs` | Output tests: Display/Debug/truncation/complex (Numpy-style) | W29T1 | 28-tests §5, §6, §7, §8 |
| W29T16 | `tests/test_error.rs` | Error tests: XenonError boundary + display output validation, Workspace structured fields | W29T1 | 28-tests §5, §6, §7, §8 |
| W29T17 | `tests/test_workspace.rs` | Workspace tests: illegal alignment/borrow guard/split/expand/assume_init/!Send+!Sync | W29T1, W29T2 | 28-tests §5, §6, §7, §8 |
| W29T18 | `tests/test_ffi.rs` | FFI tests: pointer/BLAS compatibility/export/export_mut/offset | W29T2 | 28-tests §5, §6, §7, §8 |
| W29T19 | `tests/test_parallel.rs` | Parallel tests: sum/add/dot behavioral consistency with parallel feature, concurrent read, nested prohibition, determinism | W29T3, W29T8 | 28-tests §5, §6, §7, §8 |
| W29T20 | `tests/test_simd.rs` | SIMD tests: result consistency (add/sum/dot/fallback/complex path) | W29T3, W29T8 | 28-tests §5, §6, §7, §8 |
| W29T21 | `.github/workflows/test.yml` | CI test matrix: maintain std-environment lib/tests/doctest matrix | W29T2 | 28-tests §5, §6, §7, §8 |
| W29T22 | `tests/property_tests.rs`, `property/tensor_props.rs`, `ops_props.rs`, `shape_props.rs` | Property tests: transpose involution, addition commutativity, unique no-duplicates, sum preserves identity, broadcast shape consistency | W29T3, W29T8, W29T12 | 28-tests §5, §6, §7, §8 |
| W29T23 | `.github/workflows/test.yml` | CI test matrix full config: all feature combos and property tests | W29T1–W29T22 | 28-tests §5, §6, §7, §8 |
| W29T24 | `tests/compile_fail_tests.rs` | Compile-fail harness using repository-local fixtures and assertion conventions (no trybuild dev-dep per 28-tests §4.3) | W29T1 | 28-tests §5, §6, §7, §8 |
| W29T25 | `tests/compile-fail/*.rs` | Compile-fail fixtures (9 files per design §3): wrong_dimension_type, missing_element_bound, mismatched_storage_type, unsigned_tensor_element_rejected, invalid_unsigned_element_rejected, ui_bool_sum_rejected, ui_bool_unique_rejected, ui_bool_arithmetic_rejected, blanket_scalar_add_rejected | W29T24 | 28-tests §5, §6, §7, §8 |

### W30: Documentation

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W30T1 | `src/lib.rs` | Crate-level docs: project overview, Quick Start, Features table, element types table, memory layout | None | 29-documentation §5, §6, §7, §8 |
| W30T2 | `src/lib.rs`, `Cargo.toml`, `src/prelude.rs` | #![warn(missing_docs)] lint + docs.rs metadata: all-features = true + Prelude `//!` module-level docs | W30T1 | 29-documentation §5, §6, §7, §8 |
| W30T3 | `README.md` | Project README: intro, features, Quick Start, install, doc links, license | W30T1 | 29-documentation §5, §6, §7, §8 |
| W30T4 | `CHANGELOG.md` | Optional CHANGELOG.md in Keep a Changelog format | None | 29-documentation §5, §6, §7, §8 |
| W30T5 | `src/dimension/mod.rs` | Dimension module-level docs: responsibilities, core concepts, Ix0–Ix6 usage, Dimension trait guide | W30T2 | 29-documentation §5, §6, §7, §8; 02-dimension.md |
| W30T6 | `src/element/mod.rs` | Element module-level docs: trait hierarchy, 7 element types, sealed design rationale | W30T2 | 29-documentation §5, §6, §7, §8; 03-element.md |
| W30T7 | `src/complex/mod.rs` | Complex module-level docs: repr(C) layout, FFI guarantees, arithmetic semantics | W30T2 | 29-documentation §5, §6, §7, §8; 04-complex.md |
| W30T8 | `src/storage/mod.rs` | Storage module-level docs: trait hierarchy, Owned/ViewRepr/ArcRepr, alignment guarantees | W30T2 | 29-documentation §5, §6, §7, §8; 05-storage.md |
| W30T9 | `src/layout/mod.rs` | Layout module-level docs: F-order conventions, LayoutFlags, stride computation | W30T2 | 29-documentation §5, §6, §7, §8; 06-layout.md |
| W30T10 | `src/tensor/mod.rs` | Tensor module-level docs: TensorBase, type aliases, storage kinds, memory model | W30T2 | 29-documentation §5, §6, §7, §8; 07-tensor.md |
| W30T11 | `src/iter/mod.rs` | Iterator module-level docs: Iter/IterMut, AxisIter, IndexedIter, entry methods | W30T2 | 29-documentation §5, §6, §7, §8; 10-iterator.md |
| W30T12 | `src/math/mod.rs` | Math module-level docs: binary/unary ops, comparison ops, SIMD dispatch | W30T2 | 29-documentation §5, §6, §7, §8; 11-math.md |
| W30T13 | `src/overload/mod.rs` | Overload module-level docs: operator traits, broadcast dispatch, ownership rules | W30T2 | 29-documentation §5, §6, §7, §8; 19-overload.md |
| W30T14 | `src/broadcast/mod.rs` | Broadcast module-level docs: dimension rules, zero-stride semantics, read-only boundary | W30T2 | 29-documentation §5, §6, §7, §8; 15-broadcast.md |
| W30T15 | `src/reduction/mod.rs` | Reduction module-level docs: sum/sum_axis/keepdims, SIMD/parallel dispatch gates | W30T2 | 29-documentation §5, §6, §7, §8; 13-reduction.md |
| W30T16 | `src/matrix/mod.rs` | Matrix module-level docs: dot product, rank validation, complex support | W30T2 | 29-documentation §5, §6, §7, §8; 12-matrix.md |
| W30T17 | `src/shape/mod.rs` | Shape module-level docs: transpose, O(1) semantics, F-order contiguity | W30T2 | 29-documentation §5, §6, §7, §8; 16-shape.md |
| W30T18 | `src/index/mod.rs` | Index module-level docs: NdIndex trait, multi-dim access, slicing, bounds checking | W30T2 | 29-documentation §5, §6, §7, §8; 17-indexing.md |
| W30T19 | `src/construct/mod.rs` | Construct module-level docs: initialization methods, ownership guarantees, error semantics | W30T2 | 29-documentation §5, §6, §7, §8; 18-construction.md |
| W30T20 | `src/set/mod.rs` | Set module-level docs: unique, NaN/±0.0 handling, complex equality | W30T2 | 29-documentation §5, §6, §7, §8; 14-set.md |
| W30T21 | `src/ffi/mod.rs` | FFI module-level docs: pointer access, BLAS compatibility, C repr guarantees | W30T2 | 29-documentation §5, §6, §7, §8; 23-ffi.md |
| W30T22 | `src/workspace/mod.rs` | Workspace module-level docs: allocation strategy, borrow guards, split semantics | W30T2 | 29-documentation §5, §6, §7, §8; 24-workspace.md |
| W30T23 | `src/error.rs` | Error module-level docs: XenonError variants, Result alias, error categories | W30T2 | 29-documentation §5, §6, §7, §8; 26-error.md |
| W30T24 | `src/convert/mod.rs` | Convert module-level docs: CastTo/ConvertTo traits, lossy/lossless semantics, tiered conversion | W30T2 | 29-documentation §5, §6, §7, §8; 21-type.md |
| W30T25 | `src/format/mod.rs` | Format module-level docs: numpy-style output, FormatConfig, truncation helpers | W30T2 | 29-documentation §5, §6, §7, §8; 22-output.md |
| W30T26 | `src/util/mod.rs` | Util module-level docs: responsibility overview, utility function category notes | W30T2 | 29-documentation §5, §6, §7, §8; 20-utility.md |
| W30T27 | `src/simd/mod.rs` | SIMD module-level docs: SimdKernel trait, kernel wiring, feature gate notes | W30T2 | 29-documentation §5, §6, §7, §8; 08-simd.md |
| W30T28 | `src/parallel/mod.rs` | Parallel module-level docs: ParIter, map/reduce patterns, thread-safety, feature gate | W30T2 | 29-documentation §5, §6, §7, §8; 09-parallel.md |
| W30T29 | `src/tensor/mod.rs` + related files | Tensor type-level docs + doctests: TensorBase, Tensor, TensorView, TensorViewMut, ArcTensor | W30T5–W30T28 | 29-documentation §5, §6, §7, §8; 07-tensor.md |
| W30T30 | `src/dimension/mod.rs` | Dimension type-level docs + doctests: Ix0–Ix6, IxDyn, Dimension trait | W30T5–W30T28 | 29-documentation §5, §6, §7, §8; 02-dimension.md |
| W30T31 | `src/element/mod.rs` | Element type-level docs + doctests: Element, Numeric, RealScalar, ComplexScalar traits | W30T5–W30T28 | 29-documentation §5, §6, §7, §8; 03-element.md |
| W30T32 | `src/storage/mod.rs` | Storage type-level docs + doctests: Owned, ViewRepr, StorageMut trait | W30T5–W30T28 | 29-documentation §5, §6, §7, §8; 05-storage.md |
| W30T33 | `src/layout/mod.rs` | Layout type-level docs + doctests: LayoutFlags, compute_f_strides | W30T5–W30T28 | 29-documentation §5, §6, §7, §8; 06-layout.md |
| W30T34 | `src/math/` files | Math function docs + doctests: add, sub, mul, div, sin, sqrt, exp, ln, abs | W30T5–W30T28 | 29-documentation §5, §6, §7, §8; 11-math.md |
| W30T35 | `src/reduction/` | Reduction docs + doctests: sum, sum_axis | W30T5–W30T28 | 29-documentation §5, §6, §7, §8; 13-reduction.md |
| W30T36 | `src/matrix/` | Matrix docs + doctests: dot | W30T5–W30T28 | 29-documentation §5, §6, §7, §8; 12-matrix.md |
| W30T37 | `src/broadcast/`, `src/shape/mod.rs` | Broadcast + shape docs + doctests: broadcast_shape, transpose | W30T5–W30T28 | 29-documentation §5, §6, §7, §8; 15-broadcast.md; 16-shape.md |
| W30T38 | `src/construct/mod.rs`, `src/set/mod.rs` | Construct + set docs + doctests: zeros, ones, eye, from_shape_vec, unique + error semantics | W30T5–W30T28 | 29-documentation §5, §6, §7, §8; 18-construction.md; 14-set.md |
| W30T39 | `src/ffi/mod.rs`, `src/workspace/mod.rs`, `src/error.rs` | FFI + workspace + error docs + doctests | W30T5–W30T28 | 29-documentation §5, §6, §7, §8; 23-ffi.md; 24-workspace.md; 26-error.md |
| W30T40 | `src/iter/mod.rs`, `src/convert/mod.rs`, `src/format/mod.rs`, `src/overload/mod.rs` | Iter + convert + format + overload module docs + doctests | W30T5–W30T28 | 29-documentation §5, §6, §7, §8; 10-iterator.md; 21-type.md; 22-output.md; 19-overload.md |
| W30T41 | `src/index/mod.rs` | Index function-level docs + doctests | W30T5–W30T28 | 29-documentation §5, §6, §7, §8; 17-indexing.md |
| W30T42 | `src/util/mod.rs` | Util function-level docs + doctests: clip, fill, try_fill, to_contiguous, into_contiguous | W30T26 | 29-documentation §5, §6, §7, §8; 20-utility.md |
| W30T43 | `examples/basic.rs` | Usage example: create, operate, reduce, print | W30T1 | 29-documentation §5, §6, §7, §8; 18-construction.md; 13-reduction.md; 22-output.md |
| W30T44 | `examples/complex.rs` | Usage example: complex construction, same-type arithmetic, explicit conversion ops | W30T1 | 29-documentation §5, §6, §7, §8; 04-complex.md; 21-type.md |
| W30T45 | `examples/broadcasting.rs` | Usage example: broadcast rules, row/col/scalar broadcast | W30T1 | 29-documentation §5, §6, §7, §8; 15-broadcast.md |
| W30T46 | `examples/features.rs` | Feature-gated example: conditional compile with simd/parallel features | W30T1 | 29-documentation §5, §6, §7, §8; 08-simd.md; 09-parallel.md; 01-architecture §4 |
| W30T47 | `examples/simd.rs` | Usage example: SIMD feature-gated execution and fallback behavior | W30T1 | 29-documentation §5, §6, §7, §8; 08-simd.md |
| W30T48 | `examples/ffi.rs` | Usage example: FFI export, pointer access, BLAS layout checks | W30T1 | 29-documentation §5, §6, §7, §8; 23-ffi.md |
| W30T49 | `examples/workspace.rs` | Usage example: workspace allocation and borrow/split workflow | W30T1 | 29-documentation §5, §6, §7, §8; 24-workspace.md |
| W30T50 | `src/lib.rs`, `README.md`, `examples/` | Audit that examples and crate docs only declare `std` environment; remove out-of-scope platform notes | W30T1, W30T3, W30T43–W30T49 | 29-documentation §5, §6, §7, §8 |
| W30T51 | `LICENSE` | Project license file matching Cargo.toml package metadata | W1T1 | 01-architecture §3, §4 |
| W30T52 | `.github/workflows/docs.yml` | docs.rs CI integration, missing-docs check, doctest and example compilation | W30T40, W30T43–W30T49 | 29-documentation §5, §6, §7, §8 |

---

## Dependency Graph (Simplified)

```
W1 (Coding/Setup) ──┬──→ W2 (Error)
                    ├──→ W3 (Dimension) ──→ W6 (Layout)
                    └──→ W5 (Complex)

W2 ──→ W3
W3 + W5 ──→ W4 (Element)
W6 + W3 ──→ W7 (Storage)
W2 ──→ W9 (Workspace, L2)
W3 + W6 + W7 ──→ W8 (Tensor Core)

W8 ─→ W10 (Dispatch)
W8 + W6 + W3 ─→ W11 (Broadcast)
W8 + W7 + W3 ─→ W12 (Iterators)
W8 + W6 + W3 ─→ W13 (FFI)
W4 + W5 ─→ W14 (SIMD)
W8 + W10 ─→ W15 (Parallel)

W4 + W8 + W11 + W12 + W14 + W15 ─→ W16 (Math)
W8 + W14 + W15 ─→ W17 (Matrix)
W8 + W14 + W15 ─→ W18 (Reduction)
W8 + W12 ─→ W19 (Set)
W8 + W6 ─→ W20 (Shape)
W8 + W3 ─→ W21 (Indexing)
W8 ─→ W22 (Construction)
W8 + W11 + W16 ─→ W23 (Overload)
W8 ─→ W24 (Utility)
W4 + W5 + W8 + W12 ─→ W25 (Type Conversion)  // W12T7 supplies TensorBase::iter() consumed by W25T6/W25T7
W8 ─→ W26 (Output Formatting)

W1–W26 ─→ W27 (Safety Audit)
W1–W26 ─→ W28 (Benchmarks)
W1–W26 ─→ W29 (Integration Tests)
W1–W26 ─→ W30 (Documentation)
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
11. **numpy-style output** — Display/Debug formatting mimics numpy conventions with configurable truncation
12. **IEEE 754 compliance** — Floating-point special values (NaN, Inf) handled per standard semantics
