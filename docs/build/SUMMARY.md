# Xenon Implementation Plan — Wave & Task Summary

> Project: Xenon N-dimensional array library (Rust)
> Dependency Layers: L0 → L1 → L2 → L3 → L4 → L5 → L6 → Tests/Benchmarks/Docs
> Task Granularity: Each task targets 1 function / 1 trait / 1 type, ~5–10 min, max 1 file

---

## Wave Overview

| Wave | Name | Layer | Task Count | Description |
|------|------|-------|------------|-------------|
| W1 | Project Setup & Error | L0 | 4 | Cargo.toml, lib.rs skeleton, XenonError, Sealed trait |
| W2 | Fixed Dimension Types | L1 | 8 | dimension/mod.rs + Ix0–Ix6 types |
| W3 | Dimension Traits | L1 | 6 | Dimension trait, IxDyn, IntoDimension, Axis, RemoveAxis, BroadcastDim |
| W4 | Element Trait Hierarchy | L1–L2 | 9 | Element/Numeric/RealScalar/ComplexScalar traits + sealed + impls |
| W5 | Complex Type | L1 | 6 | Complex\<T\> struct, arithmetic, Display, math methods, FFI layout |
| W6 | Layout System | L2 | 4 | LayoutFlags, Strides\<D\>, contiguity checks |
| W7 | Storage System | L2 | 7 | RawStorage/Storage/StorageMut + Owned/View/ViewMut/Arc reprs + allocator |
| W8 | Tensor Core | L3 | 6 | TensorBase\<S,D\>, type aliases, constructors, view methods, accessors |
| W9 | Dispatch | L4 | 3 | ExecPath, ParallelGuard, parallel thresholds |
| W10 | Broadcasting | L4 | 5 | can_broadcast, broadcast_shape, broadcast_to, broadcast_with |
| W11 | Iterators | L4 | 6 | StrideState, Elements, AxisIter, IndexedIter |
| W12 | FFI Helpers | L4 | 6 | BlasInfo, as_ptr/as_mut_ptr, lda, export/export_mut, from_raw_parts |
| W13 | SIMD Backend | L5 | 4 | SimdKernel trait, element ops, reduction/dot |
| W14 | Parallel Backend | L5 | 5 | ParallelBackend trait, par_map, par_zip_map, par_sum, par_dot |
| W15 | Math Operations | L5 | 5 | Binary/unary/comparison element-wise ops + SIMD |
| W16 | Matrix Operations | L5 | 4 | dot product (scalar + SIMD + parallel) |
| W17 | Reduction Operations | L5 | 5 | sum, sum_axis, sum_axis_keepdims + SIMD/parallel |
| W18 | Set Operations | L5 | 4 | unique, NaN/±0 handling, complex unique |
| W19 | Shape Operations | L5 | 3 | transpose |
| W20 | Indexing | L5 | 5 | NdIndex, try_at/get, SliceInfo, try_at_mut, slice |
| W21 | Tensor Construction | L5 | 5 | zeros/ones, eye, from_shape_vec, from_array/scalar |
| W22 | Operator Overloading | L6 | 6 | Add/Sub/Mul/Div for owned/ref/mixed/scalar |
| W23 | Utility Operations | L5 | 4 | fill, clip, to_contiguous |
| W24 | Type Conversion | L5 | 3 | CastTo trait, impls, tensor cast() |
| W25 | Output Formatting | L5 | 4 | FormatConfig, Display, Debug, pretty helpers |
| W26 | Workspace | L2 | 5 | Workspace struct, borrow guards, split, expand |
| W27 | Safety Audit | cross-cutting | 1 | Send/Sync bounds audit |
| W28 | Benchmarks | — | 5 | bench infrastructure + core/comparison benchmarks |
| W29 | Integration Tests | — | 5 | test infrastructure + core/special/property tests |
| W30 | Documentation | — | 5 | crate/module/type docs + examples + CI |
| | **Total** | | **138** | |

---

## Detailed Task List

### Wave 1: Project Setup & Error (L0)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W1T1 | `Cargo.toml` | Create package manifest with deps, features, bench targets | None | 01-architecture §4 |
| W1T2 | `src/lib.rs` | Crate root skeleton with module declarations + feature gates | W1T1 | 01-architecture §3 |
| W1T3 | `src/error.rs` | XenonError enum with all structured variants + Result alias | None | 26-error |
| W1T4 | `src/private.rs` | Sealed trait marker for sealed trait pattern | None | 01-architecture §5 |

### Wave 2: Fixed Dimension Types (L1)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W2T1 | `src/dimension/mod.rs` | Module skeleton with re-exports | W1T2 | 02-dimension |
| W2T2 | `src/dimension/static.rs` | Ix0 type definition + Dimension pre-impl stubs | W2T1 | 02-dimension §2 |
| W2T3 | `src/dimension/static.rs` | Ix1 type definition + Dimension pre-impl stubs | W2T2 | 02-dimension §2 |
| W2T4 | `src/dimension/static.rs` | Ix2 type definition + Dimension pre-impl stubs | W2T3 | 02-dimension §2 |
| W2T5 | `src/dimension/static.rs` | Ix3 type definition + Dimension pre-impl stubs | W2T4 | 02-dimension §2 |
| W2T6 | `src/dimension/static.rs` | Ix4 type definition + Dimension pre-impl stubs | W2T5 | 02-dimension §2 |
| W2T7 | `src/dimension/static.rs` | Ix5 type definition + Dimension pre-impl stubs | W2T6 | 02-dimension §2 |
| W2T8 | `src/dimension/static.rs` | Ix6 type definition + Dimension pre-impl stubs | W2T7 | 02-dimension §2 |

### Wave 3: Dimension Traits (L1)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W3T1 | `src/dimension/mod.rs` | Dimension trait definition (full) | W2T8 | 02-dimension §3 |
| W3T2 | `src/dimension/dynamic.rs` | IxDyn type + Dimension impl | W3T1 | 02-dimension §4 |
| W3T3 | `src/dimension/into.rs` | IntoDimension trait + all impls | W3T1 | 02-dimension §5 |
| W3T4 | `src/dimension/axes.rs` | Axis struct + axis operations | W3T1 | 02-dimension §6 |
| W3T5 | `src/dimension/axes.rs` | RemoveAxis trait + impls | W3T4 | 02-dimension §7 |
| W3T6 | `src/dimension/mod.rs` | BroadcastDim helper trait/func | W3T1 | 02-dimension §8 |

### Wave 4: Element Trait Hierarchy (L1–L2)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W4T1 | `src/element/mod.rs` | Module skeleton + Sealed sub-trait | W1T4, W1T3 | 03-element §3 |
| W4T2 | `src/element/mod.rs` | Element trait definition (sealed) | W4T1 | 03-element §4 |
| W4T3 | `src/element/primitives.rs` | Element impl for i32, i64 | W4T2 | 03-element §5 |
| W4T4 | `src/element/primitives.rs` | Element impl for f32, f64 | W4T2 | 03-element §5 |
| W4T5 | `src/element/primitives.rs` | Element impl for bool | W4T2 | 03-element §5 |
| W4T6 | `src/element/numeric.rs` | Numeric trait + impls for i32/i64/f32/f64 | W4T4 | 03-element §6 |
| W4T7 | `src/element/real.rs` | RealScalar trait + impls for f32/f64 | W4T6 | 03-element §7 |
| W4T8 | `src/element/complex.rs` | ComplexScalar trait + impl for Complex\<T\> | W4T6, W5 | 03-element §8 |
| W4T9 | `src/element/mod.rs` | Module re-exports + prelude integration | W4T1–W4T8 | 03-element |

> Note: W4T8 (ComplexScalar) depends on W5 (Complex type). In practice, W4T1–W4T7 can proceed in parallel with W5, and W4T8 is completed after W5.

### Wave 5: Complex Type (L1)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W5T1 | `src/complex/mod.rs` | Complex\<T\> struct definition, #[repr(C)], basic accessors | W1T4 | 04-complex §2 |
| W5T2 | `src/complex/ops.rs` | Complex Add + Sub operator impls | W5T1 | 04-complex §3 |
| W5T3 | `src/complex/ops.rs` | Complex Mul + Div operator impls | W5T2 | 04-complex §3 |
| W5T4 | `src/complex/mod.rs` | Complex Display + Debug impls | W5T1 | 04-complex §4 |
| W5T5 | `src/complex/mod.rs` | Complex math methods (norm, conj, arg, re, im, etc.) | W5T1 | 04-complex §5 |
| W5T6 | `src/complex/mod.rs` | Complex FFI layout guarantees + module re-exports | W5T1 | 04-complex §6 |

### Wave 6: Layout System (L2)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W6T1 | `src/layout/mod.rs` | Module skeleton | W3T1 | 06-layout |
| W6T2 | `src/layout/flags.rs` | LayoutFlags bitflags (F_CONTIGUOUS, ALIGNED, etc.) | W6T1 | 06-layout §2 |
| W6T3 | `src/layout/strides.rs` | Strides\<D\> struct + F-order stride calculation | W6T1, W3T1 | 06-layout §3 |
| W6T4 | `src/layout/contiguous.rs` | Contiguity check functions + module re-exports | W6T3 | 06-layout §4 |

### Wave 7: Storage System (L2)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W7T1 | `src/storage/mod.rs` | Module skeleton + RawStorage trait | W4T2 | 05-storage §3 |
| W7T2 | `src/storage/mod.rs` | Storage trait (extends RawStorage) | W7T1 | 05-storage §4 |
| W7T3 | `src/storage/mod.rs` | StorageMut trait (extends Storage) | W7T2 | 05-storage §5 |
| W7T4 | `src/storage/owned.rs` | Owned\<A\> repr + IsOwned marker | W7T3 | 05-storage §6 |
| W7T5 | `src/storage/view.rs` | ViewRepr\<'a, A\> + ViewMutRepr\<'a, A\> + IsView marker | W7T3 | 05-storage §7–8 |
| W7T6 | `src/storage/arc.rs` | ArcRepr\<A\> shared storage | W7T3 | 05-storage §9 |
| W7T7 | `src/storage/alloc.rs` | 64-byte aligned allocator + module re-exports | W7T4 | 05-storage §10 |

### Wave 8: Tensor Core (L3)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W8T1 | `src/tensor/mod.rs` | TensorBase\<S, D\> struct definition | W7T7, W3T1, W6T4, W4T2 | 07-tensor §2 |
| W8T2 | `src/tensor/aliases.rs` | Type aliases (Tensor, TensorView, TensorViewMut, ArcTensor) | W8T1 | 07-tensor §3 |
| W8T3 | `src/tensor/construct.rs` | Internal constructors (new, uninit, from_shape_vec) | W8T1 | 07-tensor §4 |
| W8T4 | `src/tensor/impls.rs` | View methods (view, view_mut, reshape, to_owned) | W8T1 | 07-tensor §5 |
| W8T5 | `src/tensor/impls.rs` | Accessor methods (shape, strides, len, rank, data_ptr, is_contiguous) | W8T1 | 07-tensor §6 |
| W8T6 | `src/tensor/mod.rs` | into_raw / from_raw_parts + module re-exports | W8T1 | 07-tensor §7 |

### Wave 9: Dispatch (L4)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W9T1 | `src/dispatch.rs` | ExecPath enum + dispatch selection logic | W8T1 | 01-architecture §5.5 |
| W9T2 | `src/dispatch.rs` | ParallelGuard (nested parallelism protection) | W9T1 | 01-architecture §5.5 |
| W9T3 | `src/dispatch.rs` | Parallel threshold constants + integration | W9T2 | 01-architecture §5.5 |

### Wave 10: Broadcasting (L4)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W10T1 | `src/broadcast/mod.rs` | Module skeleton | W8T1 | 15-broadcast |
| W10T2 | `src/broadcast/shape.rs` | can_broadcast function (shape compatibility check) | W10T1, W3T1 | 15-broadcast §2 |
| W10T3 | `src/broadcast/shape.rs` | broadcast_shape function (compute output shape) | W10T2 | 15-broadcast §3 |
| W10T4 | `src/broadcast/view.rs` | broadcast_to method (create broadcast view) | W10T3, W6T4 | 15-broadcast §4 |
| W10T5 | `src/broadcast/view.rs` | broadcast_with method + module re-exports + tests | W10T4 | 15-broadcast §5 |

### Wave 11: Iterators (L4)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W11T1 | `src/iter/mod.rs` | Module skeleton + StrideState struct | W8T1 | 10-iterator §2 |
| W11T2 | `src/iter/elements.rs` | Elements iterator (flat traversal) | W11T1 | 10-iterator §3 |
| W11T3 | `src/iter/axis.rs` | AxisIter (iteration along one axis) | W11T1 | 10-iterator §4 |
| W11T4 | `src/iter/indexed.rs` | IndexedIter (elements with indices) | W11T2 | 10-iterator §5 |
| W11T5 | `src/tensor/impls.rs` | Tensor entry methods for iteration (iter, iter_mut, axis_iter, indexed_iter) | W11T2–W11T4 | 10-iterator §6 |
| W11T6 | `src/iter/mod.rs` | Module re-exports + iter tests | W11T2–W11T4 | 10-iterator |

### Wave 12: FFI Helpers (L4)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W12T1 | `src/ffi/mod.rs` | Module skeleton | W8T1 | 23-ffi |
| W12T2 | `src/ffi/types.rs` | BlasInfo struct definition | W12T1 | 23-ffi §2 |
| W12T3 | `src/ffi/ptr.rs` | export() / export_mut() + into_raw_parts / from_raw_parts | W12T1 | 23-ffi §3 |
| W12T4 | `src/ffi/blas.rs` | is_blas_compatible + blas_info + lda() | W12T2 | 23-ffi §4 |
| W12T5 | `src/ffi/offset.rs` | try_offset_of / try_ptr_at (checked pointer arithmetic) | W12T1 | 23-ffi §5 |
| W12T6 | `src/ffi/mod.rs` | Module re-exports + FFI tests | W12T2–W12T5 | 23-ffi |

### Wave 13: SIMD Backend (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W13T1 | `src/simd/mod.rs` | Module skeleton + SimdKernel trait definition | W8T1 | 08-simd §2 |
| W13T2 | `src/simd/vector.rs` | Element-wise SIMD operations (add, sub, mul, div, abs, neg) | W13T1 | 08-simd §3 |
| W13T3 | `src/simd/vector.rs` | SIMD reduction (sum) + SIMD dot product | W13T2 | 08-simd §4 |
| W13T4 | `src/simd/mod.rs` | Runtime dispatch facade + module re-exports | W13T1 | 08-simd §5 |

### Wave 14: Parallel Backend (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W14T1 | `src/parallel/mod.rs` | Module skeleton + ParallelPool setup | W8T1, W9T3 | 09-parallel §2 |
| W14T2 | `src/parallel/map.rs` | par_map function | W14T1 | 09-parallel §3 |
| W14T3 | `src/parallel/map.rs` | par_zip_map function | W14T2 | 09-parallel §3 |
| W14T4 | `src/parallel/reduce.rs` | par_sum + par_dot functions | W14T1 | 09-parallel §4 |
| W14T5 | `src/parallel/iter.rs` + `checked.rs` | Parallel iteration helpers + error propagation + module re-exports | W14T1 | 09-parallel §5 |

### Wave 15: Math Operations (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W15T1 | `src/math/mod.rs` | Module skeleton + re-exports | W8T1, W10T5 | 11-math |
| W15T2 | `src/math/binary.rs` | Binary element-wise ops (add, sub, mul, div, add_scalar, sub_scalar, etc.) | W15T1 | 11-math §3 |
| W15T3 | `src/math/unary.rs` | Unary element-wise ops (abs, neg, signum, square, sin, modulus, conj) | W15T1 | 11-math §4 |
| W15T4 | `src/math/comparison.rs` | Comparison ops (eq, ne, lt, le, gt, ge element-wise) | W15T1 | 11-math §5 |
| W15T5 | `src/math/mod.rs` | SIMD-accelerated math dispatch + module integration tests | W15T2–W15T4, W13T4 | 11-math §6 |

### Wave 16: Matrix Operations (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W16T1 | `src/matrix/mod.rs` | Module skeleton | W8T1 | 12-matrix |
| W16T2 | `src/matrix/dot.rs` | dot() scalar implementation (1D vector inner product) | W16T1 | 12-matrix §2 |
| W16T3 | `src/matrix/dot.rs` | SIMD + parallel dot product integration | W16T2, W13T3, W14T4 | 12-matrix §3 |
| W16T4 | `src/matrix/mod.rs` | Module re-exports + matrix tests | W16T2 | 12-matrix |

### Wave 17: Reduction Operations (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W17T1 | `src/reduction/mod.rs` | Module skeleton + public API exports | W8T1 | 13-reduction |
| W17T2 | `src/reduction/sum.rs` | Global sum() function | W17T1 | 13-reduction §2 |
| W17T3 | `src/reduction/sum.rs` | sum_axis() function | W17T2 | 13-reduction §3 |
| W17T4 | `src/reduction/sum.rs` | sum_axis_keepdims() function | W17T3 | 13-reduction §4 |
| W17T5 | `src/reduction/mod.rs` | SIMD/parallel reduction dispatch + error handling + tests | W17T2, W13T3, W14T4 | 13-reduction §5 |

### Wave 18: Set Operations (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W18T1 | `src/set/mod.rs` | Module skeleton | W8T1 | 14-set |
| W18T2 | `src/set/unique.rs` | unique() function (real types) | W18T1, W11T2 | 14-set §2 |
| W18T3 | `src/set/unique.rs` | NaN/±0 handling + complex unique | W18T2 | 14-set §3 |
| W18T4 | `src/set/mod.rs` | Module re-exports + set tests | W18T2 | 14-set |

### Wave 19: Shape Operations (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W19T1 | `src/shape/mod.rs` | Module skeleton | W8T1 | 16-shape |
| W19T2 | `src/shape/transpose.rs` | transpose() implementation (axes swap) | W19T1, W6T3 | 16-shape §2 |
| W19T3 | `src/shape/mod.rs` | Module re-exports + shape tests | W19T2 | 16-shape |

### Wave 20: Indexing (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W20T1 | `src/index/mod.rs` + `ndindex.rs` | NdIndex trait definition + tuple/slice impls | W8T1, W3T1 | 17-indexing §2 |
| W20T2 | `src/index/access.rs` | try_at / get / get_unchecked methods | W20T1 | 17-indexing §3 |
| W20T3 | `src/index/slice.rs` | SliceInfo struct + slice_shape/stride computation | W20T1 | 17-indexing §4 |
| W20T4 | `src/index/access.rs` | try_at_mut / get_mut / get_unchecked_mut | W20T2 | 17-indexing §5 |
| W20T5 | `src/index/slice.rs` | slice update methods + module re-exports + index tests | W20T3 | 17-indexing §6 |

### Wave 21: Tensor Construction (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W21T1 | `src/construct/mod.rs` + `init.rs` | Module skeleton + zeros() + ones() | W8T1 | 18-construction §2 |
| W21T2 | `src/construct/eye.rs` | eye() constructor | W21T1 | 18-construction §3 |
| W21T3 | `src/construct/from.rs` | from_shape_vec + from_shape_slice | W21T1 | 18-construction §4 |
| W21T4 | `src/construct/from.rs` + `scalar.rs` | from_array + from_vec + from_scalar | W21T3 | 18-construction §5 |
| W21T5 | `src/construct/mod.rs` | Module re-exports + construction tests | W21T1–W21T4 | 18-construction |

### Wave 22: Operator Overloading (L6)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W22T1 | `src/overload/mod.rs` | Module skeleton | W8T1 | 19-overload |
| W22T2 | `src/overload/arithmetic.rs` | Add\<Tensor, Tensor\> for owned | W22T1, W10T5, W15T2 | 19-overload §2 |
| W22T3 | `src/overload/arithmetic.rs` | Add for ref/mixed (TensorView, &Tensor, etc.) | W22T2 | 19-overload §3 |
| W22T4 | `src/overload/arithmetic.rs` | Add with scalar (Tensor + f64, etc.) | W22T2 | 19-overload §4 |
| W22T5 | `src/overload/arithmetic.rs` | Sub/Mul/Div operators (owned, ref, mixed, scalar) | W22T2–W22T4 | 19-overload §5 |
| W22T6 | `src/overload/mod.rs` | Module re-exports + overload tests | W22T2–W22T5 | 19-overload |

### Wave 23: Utility Operations (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W23T1 | `src/util/mod.rs` + `fill.rs` | Module skeleton + fill() operation | W8T1 | 20-utility §2 |
| W23T2 | `src/util/clip.rs` | clip() operation | W23T1 | 20-utility §3 |
| W23T3 | `src/util/contiguous.rs` | to_contiguous() operation | W23T1, W6T4 | 20-utility §4 |
| W23T4 | `src/util/mod.rs` | Module re-exports + util tests | W23T1–W23T3 | 20-utility |

### Wave 24: Type Conversion (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W24T1 | `src/convert/mod.rs` + `cast.rs` | CastTo trait definition | W4T2, W8T1 | 21-type §2 |
| W24T2 | `src/convert/cast.rs` | CastTo impls for all supported type pairs | W24T1 | 21-type §3 |
| W24T3 | `src/convert/cast.rs` | tensor.cast() method + module re-exports + tests | W24T2 | 21-type §4 |

### Wave 25: Output Formatting (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W25T1 | `src/format/mod.rs` + `config.rs` | Module skeleton + FormatConfig struct | W8T1 | 22-output §2 |
| W25T2 | `src/format/display.rs` | Display impl for tensor (NumPy-style) | W25T1 | 22-output §3 |
| W25T3 | `src/format/debug.rs` | Debug impl for tensor | W25T1 | 22-output §4 |
| W25T4 | `src/format/pretty.rs` + `mod.rs` | Pretty formatting helpers + module re-exports + tests | W25T2 | 22-output §5 |

### Wave 26: Workspace (L2)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W26T1 | `src/workspace/mod.rs` + `workspace.rs` | Workspace struct + constants + construction/destruction | W1T3 | 24-workspace §2 |
| W26T2 | `src/workspace/borrow.rs` | WorkspaceBorrow + WorkspaceBorrowMut guards | W26T1 | 24-workspace §3 |
| W26T3 | `src/workspace/split.rs` | SplitBorrowMut guard | W26T1 | 24-workspace §4 |
| W26T4 | `src/workspace/expand.rs` | ensure_capacity + reallocate | W26T1 | 24-workspace §5 |
| W26T5 | `src/workspace/mod.rs` + `error.rs` | WorkspaceErrorCategory + module re-exports + tests | W26T1 | 24-workspace §6 |

### Wave 27: Safety Audit (cross-cutting)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W27T1 | All modules | Send/Sync bounds audit: verify all types have correct Send/Sync impls across all modules | W1–W26 complete | 25-safety |

### Wave 28: Benchmarks

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W28T1 | `benches/utils/` | Benchmark utilities (shared constants + test data generators) | W21 | 27-benchmark §2 |
| W28T2 | `benches/math.rs` + `construction.rs` | Element-wise + construction benchmarks | W28T1 | 27-benchmark §3 |
| W28T3 | `benches/reduction.rs` + `dot.rs` + `set.rs` | Reduction + dot + set benchmarks | W28T1 | 27-benchmark §3 |
| W28T4 | `benches/broadcast.rs` + `shape.rs` | Broadcast + shape benchmarks | W28T1 | 27-benchmark §3 |
| W28T5 | `benches/simd_comparison.rs` + `parallel_comparison.rs` | SIMD + parallel comparison benchmarks + CI config | W28T1 | 27-benchmark §4 |

### Wave 29: Integration Tests

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W29T1 | `tests/common/` | Test infrastructure (shared utilities, assertion helpers, generators) | W21 | 28-tests §2 |
| W29T2 | `tests/test_tensor.rs` + `test_error.rs` | Core tests (tensor, error) | W29T1 | 28-tests §3 |
| W29T3 | `tests/test_*.rs` (math, broadcast, reduction, etc.) | Specialized operation tests (all test files except core + property) | W29T1 | 28-tests §4 |
| W29T4 | `tests/property_tests.rs` + `tests/property/` + `tests/compile-fail/` | Property-based tests + compile-fail tests | W29T1 | 28-tests §5 |
| W29T5 | CI config | CI integration (cargo test / cargo bench / cargo doc) | W29T2–W29T4 | 28-tests §6 |

### Wave 30: Documentation

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W30T1 | `src/lib.rs` | Crate-level documentation (README-style, feature guide, examples) | W21 | 29-documentation §2 |
| W30T2 | `src/*/mod.rs` | Module-level documentation for all modules | W30T1 | 29-documentation §3 |
| W30T3 | All public items | Type/function-level documentation (all pub items) | W30T2 | 29-documentation §4 |
| W30T4 | `examples/` | Usage examples (basic, complex, broadcasting, features, simd, ffi, workspace) | W30T3 | 29-documentation §5 |
| W30T5 | CI config | docs.rs CI integration + doc test verification | W30T3 | 29-documentation §6 |

---

## Dependency Graph (Simplified)

```
W1 (Setup/Error)
 └→ W2 (Fixed Dims) → W3 (Dim Traits) ──┬→ W6 (Layout)
                                          │
W1 → W4 (Element) ←── W5 (Complex) ──────┤
                                          │
W1 → W5 (Complex) ────────────────────────┤
                                          │
W1 → W26 (Workspace, L2) ────────────────┤
                                          │
W6 + W7 + W3 + W4 ─→ W8 (Tensor Core) ──┬→ W9 (Dispatch)
                                          ├──→ W10 (Broadcast) ─┐
                                          ├──→ W11 (Iterators) ──┤
                                          └──→ W12 (FFI)         │
                                                                   │
W9 + W13 (SIMD) ←─────────────────────────────────────────────────┤
W9 + W14 (Parallel) ←─────────────────────────────────────────────┤
                                                                   │
W8 + W10 + W11 + W13 + W14 ─→ W15 (Math) ──────────────────────┐  │
W8 + W13 + W14 ─→ W16 (Matrix)                                  │  │
W8 + W13 + W14 ─→ W17 (Reduction)                               │  │
W8 + W11 ─→ W18 (Set)                                           │  │
W8 + W6 ─→ W19 (Shape)                                          │  │
W8 + W3 ─→ W20 (Indexing)                                       │  │
W8 ─→ W21 (Construction)                                        │  │
W8 + W10 + W15 ─→ W22 (Overload) ←──────────────────────────────┘  │
W8 ─→ W23 (Utility)                                                 │
W4 + W8 ─→ W24 (Type Conversion)                                    │
W8 ─→ W25 (Output Formatting)                                       │
                                                                     │
W1–W26 ─→ W27 (Safety Audit)                                        │
W21 ─→ W28 (Benchmarks)                                             │
W21 ─→ W29 (Integration Tests)                                      │
W21 ─→ W30 (Documentation)                                          │
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
8. **No negative strides** — Current version does not support negative stride layouts
9. **Zero-step only for broadcast** — Zero strides only appear on read-only broadcast views
10. **No UB with ZST/empty** — All operations with zero-size types or empty arrays must not cause UB
