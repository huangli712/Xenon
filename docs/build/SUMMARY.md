# Xenon Implementation Plan — Wave & Task Summary

- Project: Xenon N-dimensional array library (Rust)
- Dependency Layers: L0 → L1 → L2 → L3 → L4 → L5 → L6 → Tests/Benchmarks/Docs
- Task Granularity: Each task targets 1 function / 1 trait / 1 type, ~5–10 min, max 1 file

---

## Wave Overview

| Wave | Name | Layer | Task Count | Description |
|------|------|-------|------------|-------------|
| W01 | Project Setup & Error | L0 | 4 | Cargo.toml, lib.rs skeleton, XenonError, Sealed trait |
| W02 | Fixed Dimension Types | L1 | 8 | dimension/mod.rs + Ix0–Ix6 types |
| W03 | Dimension Traits | L1 | 6 | Dimension trait, IxDyn, IntoDimension, Axis, RemoveAxis, BroadcastDim |
| W04 | Element Trait Hierarchy | L1–L2 | 9 | Element/Numeric/RealScalar/ComplexScalar traits + sealed + impls |
| W05 | Complex Type | L1 | 6 | Complex<T> struct, arithmetic, Display, math methods, FFI layout |
| W06 | Layout System | L2 | 4 | LayoutFlags, Strides<D>, contiguity checks |
| W07 | Storage System | L2 | 7 | RawStorage/Storage/StorageMut + Owned/View/ViewMut/Arc reprs + allocator |
| W08 | Tensor Core | L3 | 6 | TensorBase<S,D>, type aliases, constructors, view methods, accessors |
| W09 | Dispatch | L4 | 3 | ExecPath, ParallelGuard, parallel thresholds |
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

### Wave 01: Project Setup & Error (L0)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W01T01 | `Cargo.toml` | Create package manifest with deps, features, bench targets | None | 01-architecture §4 |
| W01T02 | `src/lib.rs` | Crate root skeleton with module declarations + feature gates | W01T01 | 01-architecture §3 |
| W01T03 | `src/error.rs` | XenonError enum with all structured variants + Result alias | None | 26-error |
| W01T04 | `src/private.rs` | Sealed trait marker for sealed trait pattern | None | 01-architecture §5 |

### Wave 02: Fixed Dimension Types (L1)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W02T01 | `src/dimension/mod.rs` | Module skeleton with re-exports | W01T02 | 02-dimension |
| W02T02 | `src/dimension/static.rs` | Ix0 type definition + Dimension pre-impl stubs | W02T01 | 02-dimension §2 |
| W02T03 | `src/dimension/static.rs` | Ix1 type definition + Dimension pre-impl stubs | W02T02 | 02-dimension §2 |
| W02T04 | `src/dimension/static.rs` | Ix2 type definition + Dimension pre-impl stubs | W02T03 | 02-dimension §2 |
| W02T05 | `src/dimension/static.rs` | Ix3 type definition + Dimension pre-impl stubs | W02T04 | 02-dimension §2 |
| W02T06 | `src/dimension/static.rs` | Ix4 type definition + Dimension pre-impl stubs | W02T05 | 02-dimension §2 |
| W02T07 | `src/dimension/static.rs` | Ix5 type definition + Dimension pre-impl stubs | W02T06 | 02-dimension §2 |
| W02T08 | `src/dimension/static.rs` | Ix6 type definition + Dimension pre-impl stubs | W02T07 | 02-dimension §2 |

### Wave 03: Dimension Traits (L1)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W03T01 | `src/dimension/mod.rs` | Dimension trait definition (full) | W02T08 | 02-dimension §3 |
| W03T02 | `src/dimension/dynamic.rs` | IxDyn type + Dimension impl | W03T01 | 02-dimension §4 |
| W03T03 | `src/dimension/into.rs` | IntoDimension trait + all impls | W03T01 | 02-dimension §5 |
| W03T04 | `src/dimension/axes.rs` | Axis struct + axis operations | W03T01 | 02-dimension §6 |
| W03T05 | `src/dimension/axes.rs` | RemoveAxis trait + impls | W03T04 | 02-dimension §7 |
| W03T06 | `src/dimension/mod.rs` | BroadcastDim helper trait/func | W03T01 | 02-dimension §8 |

### Wave 04: Element Trait Hierarchy (L1–L2)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W04T01 | `src/element/mod.rs` | Module skeleton + Sealed sub-trait | W01T04, W01T03 | 03-element §3 |
| W04T02 | `src/element/mod.rs` | Element trait definition (sealed) | W04T01 | 03-element §4 |
| W04T03 | `src/element/primitives.rs` | Element impl for i32, i64 | W04T02 | 03-element §5 |
| W04T04 | `src/element/primitives.rs` | Element impl for f32, f64 | W04T02 | 03-element §5 |
| W04T05 | `src/element/primitives.rs` | Element impl for bool | W04T02 | 03-element §5 |
| W04T06 | `src/element/numeric.rs` | Numeric trait + impls for i32/i64/f32/f64 | W04T04 | 03-element §6 |
| W04T07 | `src/element/real.rs` | RealScalar trait + impls for f32/f64 | W04T06 | 03-element §7 |
| W04T08 | `src/element/complex.rs` | ComplexScalar trait + impl for Complex<T> | W04T06 | 03-element §8 |
| W04T09 | `src/element/mod.rs` | Module re-exports + prelude integration | W04T1–W04T08 | 03-element |

### Wave 05: Complex Type (L1)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W05T01 | `src/complex/mod.rs` | Complex<T> struct definition, #[repr(C)], basic accessors | W01T04 | 04-complex §2 |
| W05T02 | `src/complex/ops.rs` | Complex Add + Sub operator impls | W05T01 | 04-complex §3 |
| W05T03 | `src/complex/ops.rs` | Complex Mul + Div operator impls | W05T02 | 04-complex §3 |
| W05T04 | `src/complex/mod.rs` | Complex Display + Debug impls | W05T01 | 04-complex §4 |
| W05T05 | `src/complex/mod.rs` | Complex math methods (norm, conj, arg, re, im, etc.) | W05T01 | 04-complex §5 |
| W05T06 | `src/complex/mod.rs` | Complex FFI layout guarantees + module re-exports | W05T01 | 04-complex §6 |

### Wave 06: Layout System (L2)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W06T01 | `src/layout/mod.rs` | Module skeleton | W03T01 | 06-layout |
| W06T02 | `src/layout/flags.rs` | LayoutFlags bitflags (F_CONTIGUOUS, ALIGNED, etc.) | W06T01 | 06-layout §2 |
| W06T03 | `src/layout/strides.rs` | Strides<D> struct + F-order stride calculation | W06T01, W03T01 | 06-layout §3 |
| W06T04 | `src/layout/contiguous.rs` | Contiguity check functions + module re-exports | W06T03 | 06-layout §4 |

### Wave 07: Storage System (L2)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W07T01 | `src/storage/mod.rs` | Module skeleton + RawStorage trait | W04T02 | 05-storage §3 |
| W07T02 | `src/storage/mod.rs` | Storage trait (extends RawStorage) | W07T01 | 05-storage §4 |
| W07T03 | `src/storage/mod.rs` | StorageMut trait (extends Storage) | W07T02 | 05-storage §5 |
| W07T04 | `src/storage/owned.rs` | Owned<A> repr + IsOwned marker | W07T03 | 05-storage §6 |
| W07T05 | `src/storage/view.rs` | ViewRepr<'a, A> + ViewMutRepr<'a, A> + IsView marker | W07T03 | 05-storage §7–8 |
| W07T06 | `src/storage/arc.rs` | ArcRepr<A> shared storage | W07T03 | 05-storage §9 |
| W07T07 | `src/storage/alloc.rs` | 64-byte aligned allocator + module re-exports | W07T04 | 05-storage §10 |

### Wave 08: Tensor Core (L3)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W08T01 | `src/tensor/mod.rs` | TensorBase<S, D> struct definition | W07T07, W03T01, W06T04, W04T02 | 07-tensor §2 |
| W08T02 | `src/tensor/aliases.rs` | Type aliases (Tensor, TensorView, TensorViewMut, ArcTensor) | W08T01 | 07-tensor §3 |
| W08T03 | `src/tensor/construct.rs` | Internal constructors (new, uninit, from_shape_vec) | W08T01 | 07-tensor §4 |
| W08T04 | `src/tensor/impls.rs` | View methods (view, view_mut, reshape, to_owned) | W08T01 | 07-tensor §5 |
| W08T05 | `src/tensor/impls.rs` | Accessor methods (shape, strides, len, rank, data_ptr, is_contiguous) | W08T01 | 07-tensor §6 |
| W08T06 | `src/tensor/mod.rs` | into_raw / from_raw_parts + module re-exports | W08T01 | 07-tensor §7 |

### Wave 09: Dispatch (L4)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W09T01 | `src/dispatch.rs` | ExecPath enum + dispatch selection logic | W08T01 | 01-architecture §5.5 |
| W09T02 | `src/dispatch.rs` | ParallelGuard (nested parallelism protection) | W09T01 | 01-architecture §5.5 |
| W09T03 | `src/dispatch.rs` | Parallel threshold constants + integration | W09T02 | 01-architecture §5.5 |

### Wave 10: Broadcasting (L4)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W10T01 | `src/broadcast/mod.rs` | Module skeleton | W08T01 | 15-broadcast |
| W10T02 | `src/broadcast/shape.rs` | can_broadcast function (shape compatibility check) | W10T01, W03T01 | 15-broadcast §2 |
| W10T03 | `src/broadcast/shape.rs` | broadcast_shape function (compute output shape) | W10T02 | 15-broadcast §3 |
| W10T04 | `src/broadcast/view.rs` | broadcast_to method (create broadcast view) | W10T03, W06T04 | 15-broadcast §4 |
| W10T05 | `src/broadcast/view.rs` | broadcast_with method + module re-exports + tests | W10T04 | 15-broadcast §5 |

### Wave 11: Iterators (L4)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W11T01 | `src/iter/mod.rs` | Module skeleton + StrideState struct | W08T01 | 10-iterator §2 |
| W11T02 | `src/iter/elements.rs` | Elements iterator (flat traversal) | W11T01 | 10-iterator §3 |
| W11T03 | `src/iter/axis.rs` | AxisIter (iteration along one axis) | W11T01 | 10-iterator §4 |
| W11T04 | `src/iter/indexed.rs` | IndexedIter (elements with indices) | W11T02 | 10-iterator §5 |
| W11T05 | `src/tensor/impls.rs` | Tensor entry methods for iteration (iter, iter_mut, axis_iter, indexed_iter) | W11T02–W11T04 | 10-iterator §6 |
| W11T06 | `src/iter/mod.rs` | Module re-exports + iter tests | W11T02–W11T04 | 10-iterator |

### Wave 12: FFI Helpers (L4)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W12T01 | `src/ffi/mod.rs` | Module skeleton | W08T01 | 23-ffi |
| W12T02 | `src/ffi/types.rs` | BlasInfo struct definition | W12T01 | 23-ffi §2 |
| W12T03 | `src/ffi/ptr.rs` | export() / export_mut() + into_raw_parts / from_raw_parts | W12T01 | 23-ffi §3 |
| W12T04 | `src/ffi/blas.rs` | is_blas_compatible + blas_info + lda() | W12T02 | 23-ffi §4 |
| W12T05 | `src/ffi/offset.rs` | try_offset_of / try_ptr_at (checked pointer arithmetic) | W12T01 | 23-ffi §5 |
| W12T06 | `src/ffi/mod.rs` | Module re-exports + FFI tests | W12T02–W12T05 | 23-ffi |

### Wave 13: SIMD Backend (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W13T01 | `src/simd/mod.rs` | Module skeleton + SimdKernel trait definition | W08T01 | 08-simd §2 |
| W13T02 | `src/simd/vector.rs` | Element-wise SIMD operations (add, sub, mul, div, abs, neg) | W13T01 | 08-simd §3 |
| W13T03 | `src/simd/vector.rs` | SIMD reduction (sum) + SIMD dot product | W13T02 | 08-simd §4 |
| W13T04 | `src/simd/mod.rs` | Runtime dispatch facade + module re-exports | W13T01 | 08-simd §5 |

### Wave 14: Parallel Backend (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W14T01 | `src/parallel/mod.rs` | Module skeleton + ParallelPool setup | W08T01, W09T03 | 09-parallel §2 |
| W14T02 | `src/parallel/map.rs` | par_map function | W14T01 | 09-parallel §3 |
| W14T03 | `src/parallel/map.rs` | par_zip_map function | W14T02 | 09-parallel §3 |
| W14T04 | `src/parallel/reduce.rs` | par_sum + par_dot functions | W14T01 | 09-parallel §4 |
| W14T05 | `src/parallel/iter.rs` + `checked.rs` | Parallel iteration helpers + error propagation + module re-exports | W14T01 | 09-parallel §5 |

### Wave 15: Math Operations (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W15T01 | `src/math/mod.rs` | Module skeleton + re-exports | W08T01, W10T05 | 11-math |
| W15T02 | `src/math/binary.rs` | Binary element-wise ops (add, sub, mul, div, add_scalar, sub_scalar, etc.) | W15T01 | 11-math §3 |
| W15T03 | `src/math/unary.rs` | Unary element-wise ops (abs, neg, signum, square, sin, modulus, conj) | W15T01 | 11-math §4 |
| W15T04 | `src/math/comparison.rs` | Comparison ops (eq, ne, lt, le, gt, ge element-wise) | W15T01 | 11-math §5 |
| W15T05 | `src/math/mod.rs` | SIMD-accelerated math dispatch + module integration tests | W15T02–W15T04, W13T04 | 11-math §6 |

### Wave 16: Matrix Operations (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W16T01 | `src/matrix/mod.rs` | Module skeleton | W08T01 | 12-matrix |
| W16T02 | `src/matrix/dot.rs` | dot() scalar implementation (1D vector inner product) | W16T01 | 12-matrix §2 |
| W16T03 | `src/matrix/dot.rs` | SIMD + parallel dot product integration | W16T02, W13T03, W14T04 | 12-matrix §3 |
| W16T04 | `src/matrix/mod.rs` | Module re-exports + matrix tests | W16T02 | 12-matrix |

### Wave 17: Reduction Operations (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W17T01 | `src/reduction/mod.rs` | Module skeleton + public API exports | W08T01 | 13-reduction |
| W17T02 | `src/reduction/sum.rs` | Global sum() function | W17T01 | 13-reduction §2 |
| W17T03 | `src/reduction/sum.rs` | sum_axis() function | W17T02 | 13-reduction §3 |
| W17T04 | `src/reduction/sum.rs` | sum_axis_keepdims() function | W17T03 | 13-reduction §4 |
| W17T05 | `src/reduction/mod.rs` | SIMD/parallel reduction dispatch + error handling + tests | W17T02, W13T03, W14T04 | 13-reduction §5 |

### Wave 18: Set Operations (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W18T01 | `src/set/mod.rs` | Module skeleton | W08T01 | 14-set |
| W18T02 | `src/set/unique.rs` | unique() function (real types) | W18T01, W11T02 | 14-set §2 |
| W18T03 | `src/set/unique.rs` | NaN/±0 handling + complex unique | W18T02 | 14-set §3 |
| W18T04 | `src/set/mod.rs` | Module re-exports + set tests | W18T02 | 14-set |

### Wave 19: Shape Operations (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W19T01 | `src/shape/mod.rs` | Module skeleton | W08T01 | 16-shape |
| W19T02 | `src/shape/transpose.rs` | transpose() implementation (axes swap) | W19T01, W06T03 | 16-shape §2 |
| W19T03 | `src/shape/mod.rs` | Module re-exports + shape tests | W19T02 | 16-shape |

### Wave 20: Indexing (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W20T01 | `src/index/mod.rs` + `ndindex.rs` | NdIndex trait definition + tuple/slice impls | W08T01, W03T01 | 17-indexing §2 |
| W20T02 | `src/index/access.rs` | try_at / get / get_unchecked methods | W20T01 | 17-indexing §3 |
| W20T03 | `src/index/slice.rs` | SliceInfo struct + slice_shape/stride computation | W20T01 | 17-indexing §4 |
| W20T04 | `src/index/access.rs` | try_at_mut / get_mut / get_unchecked_mut | W20T02 | 17-indexing §5 |
| W20T05 | `src/index/slice.rs` | slice update methods + module re-exports + index tests | W20T03 | 17-indexing §6 |

### Wave 21: Tensor Construction (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W21T01 | `src/construct/mod.rs` + `init.rs` | Module skeleton + zeros() + ones() | W08T01 | 18-construction §2 |
| W21T02 | `src/construct/eye.rs` | eye() constructor | W21T01 | 18-construction §3 |
| W21T03 | `src/construct/from.rs` | from_shape_vec + from_shape_slice | W21T01 | 18-construction §4 |
| W21T04 | `src/construct/from.rs` + `scalar.rs` | from_array + from_vec + from_scalar | W21T03 | 18-construction §5 |
| W21T05 | `src/construct/mod.rs` | Module re-exports + construction tests | W21T01–W21T04 | 18-construction |

### Wave 22: Operator Overloading (L6)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W22T01 | `src/overload/mod.rs` | Module skeleton | W08T01 | 19-overload |
| W22T02 | `src/overload/arithmetic.rs` | Add\<Tensor, Tensor\> for owned | W22T01, W10T05, W15T02 | 19-overload §2 |
| W22T03 | `src/overload/arithmetic.rs` | Add for ref/mixed (TensorView, &Tensor, etc.) | W22T02 | 19-overload §3 |
| W22T04 | `src/overload/arithmetic.rs` | Add with scalar (Tensor + f64, etc.) | W22T02 | 19-overload §4 |
| W22T05 | `src/overload/arithmetic.rs` | Sub/Mul/Div operators (owned, ref, mixed, scalar) | W22T02–W22T04 | 19-overload §5 |
| W22T06 | `src/overload/mod.rs` | Module re-exports + overload tests | W22T02–W22T05 | 19-overload |

### Wave 23: Utility Operations (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W23T01 | `src/util/mod.rs` + `fill.rs` | Module skeleton + fill() operation | W08T01 | 20-utility §2 |
| W23T02 | `src/util/clip.rs` | clip() operation | W23T01 | 20-utility §3 |
| W23T03 | `src/util/contiguous.rs` | to_contiguous() operation | W23T01, W06T04 | 20-utility §4 |
| W23T04 | `src/util/mod.rs` | Module re-exports + util tests | W23T01–W23T03 | 20-utility |

### Wave 24: Type Conversion (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W24T01 | `src/convert/mod.rs` + `cast.rs` | CastTo trait definition | W04T02, W08T01 | 21-type §2 |
| W24T02 | `src/convert/cast.rs` | CastTo impls for all supported type pairs | W24T01 | 21-type §3 |
| W24T03 | `src/convert/cast.rs` | tensor.cast() method + module re-exports + tests | W24T02 | 21-type §4 |

### Wave 25: Output Formatting (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W25T01 | `src/format/mod.rs` + `config.rs` | Module skeleton + FormatConfig struct | W08T01 | 22-output §2 |
| W25T02 | `src/format/display.rs` | Display impl for tensor (NumPy-style) | W25T01 | 22-output §3 |
| W25T03 | `src/format/debug.rs` | Debug impl for tensor | W25T01 | 22-output §4 |
| W25T04 | `src/format/pretty.rs` + `mod.rs` | Pretty formatting helpers + module re-exports + tests | W25T02 | 22-output §5 |

### Wave 26: Workspace (L2)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W26T01 | `src/workspace/mod.rs` + `workspace.rs` | Workspace struct + constants + construction/destruction | W01T03 | 24-workspace §2 |
| W26T02 | `src/workspace/borrow.rs` | WorkspaceBorrow + WorkspaceBorrowMut guards | W26T01 | 24-workspace §3 |
| W26T03 | `src/workspace/split.rs` | SplitBorrowMut guard | W26T01 | 24-workspace §4 |
| W26T04 | `src/workspace/expand.rs` | ensure_capacity + reallocate | W26T01 | 24-workspace §5 |
| W26T05 | `src/workspace/mod.rs` + `error.rs` | WorkspaceErrorCategory + module re-exports + tests | W26T01 | 24-workspace §6 |

### Wave 27: Safety Audit (cross-cutting)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W27T01 | All modules | Send/Sync bounds audit: verify all types have correct Send/Sync impls across all modules | W01–W26 complete | 25-safety |

### Wave 28: Benchmarks

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W28T01 | `benches/utils/` | Benchmark utilities (shared constants + test data generators) | W21 | 27-benchmark §2 |
| W28T02 | `benches/math.rs` + `construction.rs` | Element-wise + construction benchmarks | W28T01 | 27-benchmark §3 |
| W28T03 | `benches/reduction.rs` + `dot.rs` + `set.rs` | Reduction + dot + set benchmarks | W28T01 | 27-benchmark §3 |
| W28T04 | `benches/broadcast.rs` + `shape.rs` | Broadcast + shape benchmarks | W28T01 | 27-benchmark §3 |
| W28T05 | `benches/simd_comparison.rs` + `parallel_comparison.rs` | SIMD + parallel comparison benchmarks + CI config | W28T01 | 27-benchmark §4 |

### Wave 29: Integration Tests

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W29T01 | `tests/common/` | Test infrastructure (shared utilities, assertion helpers, generators) | W21 | 28-tests §2 |
| W29T02 | `tests/test_tensor.rs` + `test_error.rs` | Core tests (tensor, error) | W29T01 | 28-tests §3 |
| W29T03 | `tests/test_*.rs` (math, broadcast, reduction, etc.) | Specialized operation tests (all test files except core + property) | W29T01 | 28-tests §4 |
| W29T04 | `tests/property_tests.rs` + `tests/property/` + `tests/compile-fail/` | Property-based tests + compile-fail tests | W29T01 | 28-tests §5 |
| W29T05 | CI config | CI integration (cargo test / cargo bench / cargo doc) | W29T02–W29T04 | 28-tests §6 |

### Wave 30: Documentation

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W30T01 | `src/lib.rs` | Crate-level documentation (README-style, feature guide, examples) | W21 | 29-documentation §2 |
| W30T02 | `src/*/mod.rs` | Module-level documentation for all modules | W30T01 | 29-documentation §3 |
| W30T03 | All public items | Type/function-level documentation (all pub items) | W30T02 | 29-documentation §4 |
| W30T04 | `examples/` | Usage examples (basic, complex, broadcasting, features, simd, ffi, workspace) | W30T03 | 29-documentation §5 |
| W30T05 | CI config | docs.rs CI integration + doc test verification | W30T03 | 29-documentation §6 |

---

## Key Design Constraints (All Waves Must Follow)

1. **F-order only** — Column-major layout, no C-order support
2. **Single crate** — `xenon`, MSRV Rust 1.85+, edition 2024
3. **7 element types** — i32, i64, f32, f64, Complex<f32>, Complex<f64>, bool
4. **Sealed traits** — No external implementation of Element, Dimension, etc.
5. **Optional deps only** — rayon (parallel) and pulp (simd) via feature gates
6. **64-byte alignment** — Owned storage uses 64-byte aligned allocator
7. **Unified error model** — All recoverable errors via `XenonError` + `Result<T>`
8. **No negative strides** — Current version does not support negative stride layouts
9. **Zero-step only for broadcast** — Zero strides only appear on read-only broadcast views
10. **No UB with ZST/empty** — All operations with zero-size types or empty arrays must not cause UB
