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
| W03 | Dimension Traits | L1 | 9 | Dimension trait, IxDyn struct+impl, into_dyn/try_from_dyn, IntoDimension, Axis, RemoveAxis, BroadcastDim |
| W04 | Element Trait Hierarchy | L2 | 9 | Element/Numeric/RealScalar/ComplexScalar traits + sealed + impls |
| W05 | Complex Type | L1 | 6 | Complex<T> struct, arithmetic, Display, math methods, FFI layout |
| W06 | Layout System | L2 | 4 | LayoutFlags, Strides<D>, contiguity checks |
| W07 | Storage System | L2 | 8 | RawStorage/Storage/StorageMut + Owned/View/ViewMut/Arc reprs + allocator |
| W08 | Tensor Core | L3 | 10 | TensorBase struct, type aliases, shape/layout/ptr queries, constructors, view, raw_parts, tests |
| W09 | Dispatch | L4 | 3 | ExecPath, ParallelGuard, parallel thresholds |
| W10 | Broadcasting | L4 | 5 | can_broadcast, broadcast_shape, broadcast_to, broadcast_with |
| W11 | Iterators | L4 | 6 | StrideState, Elements, AxisIter, IndexedIter |
| W12 | FFI Helpers | L4 | 6 | BlasInfo, as_ptr/as_mut_ptr, lda, export/export_mut, from_raw_parts |
| W13 | SIMD Backend | L5 | 4 | SimdKernel trait, element ops, reduction/dot |
| W14 | Parallel Backend | L5 | 9 | ParallelPool, par_map, par_zip_map, par_sum, par_dot, iter, checked, feature gate tests |
| W15 | Math Operations | L5 | 7 | Binary/unary/comparison element-wise ops + SIMD dispatch + integration tests |
| W16 | Matrix Operations | L5 | 5 | dot product (scalar + SIMD + parallel) |
| W17 | Reduction Operations | L5 | 6 | sum, sum_axis, sum_axis_keepdims + SIMD/parallel + error handling + tests |
| W18 | Set Operations | L5 | 5 | unique, NaN/±0 handling, complex unique |
| W19 | Shape Operations | L5 | 3 | transpose |
| W20 | Indexing | L5 | 5 | NdIndex, try_at/get, SliceInfo, try_at_mut, slice |
| W21 | Tensor Construction | L5 | 5 | zeros/ones, eye, from_shape_vec, from_array/scalar |
| W22 | Operator Overloading | L6 | 6 | Add/Sub/Mul/Div for owned/ref/mixed/scalar |
| W23 | Utility Operations | L5 | 4 | fill, clip, to_contiguous |
| W24 | Type Conversion | L5 | 5 | CastTo trait, module skeleton, to_owned/into_owned, cast(), extended CastTo impls |
| W25 | Output Formatting | L5 | 5 | FormatConfig, Display, Debug, pretty helpers, docs |
| W26 | Workspace | L2 | 7 | WorkspaceErrorCategory, Workspace struct, mod.rs, borrow guards, split, expand, docs |
| W27 | Safety Audit | cross-cutting | 7 | Send/Sync impls for storage reprs + parallel safety + tests |
| W28 | Benchmarks | — | 12 | Cargo.toml bench config, bench utils, math/reduction/dot/set/broadcast/shape/construction benches, SIMD/parallel comparison, CI |
| W29 | Integration Tests | — | 21 | test infra, 15 module test files, parallel/simd feature tests, std env check, property tests, CI matrix |
| W30 | Documentation | — | 26 | crate docs, missing_docs lint, README, CHANGELOG, core/tensor/infra module docs, type/function docs per module, 6 example files, CI docs |
| | **Total** | | **220** | |

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
| W03T02 | `src/dimension/dynamic.rs` | IxDyn struct + constructors (from_slice, from_vec, ones) | W03T01 | 02-dimension §5.5 |
| W03T03 | `src/dimension/dynamic.rs` | Dimension trait impl for IxDyn (incl. into_dyn/try_from_dyn trivial impls) | W03T02 | 02-dimension §5.5 |
| W03T04 | `src/dimension/static.rs` | into_dyn() impl for Ix0–Ix6 (static → dynamic) | W03T03, W02T08 | 02-dimension §5.5 |
| W03T05 | `src/dimension/static.rs` | try_from_dyn() impl for Ix0–Ix6 (dynamic → static) | W03T04 | 02-dimension §5.5 |
| W03T06 | `src/dimension/into.rs` | IntoDimension trait + all impls (tuple/array/slice/Vec) | W03T01 | 02-dimension §5 |
| W03T07 | `src/dimension/axes.rs` | Axis struct + axis operations | W03T01 | 02-dimension §6 |
| W03T08 | `src/dimension/axes.rs` | RemoveAxis trait + impls | W03T06 | 02-dimension §7 |
| W03T09 | `src/dimension/mod.rs` | BroadcastDim helper trait/func | W03T01 | 02-dimension §8 |

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
| W04T09 | `src/element/mod.rs` | Module re-exports + prelude integration | W04T01–W04T08 | 03-element |

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
| W07T05 | `src/storage/view.rs` | ViewRepr<'a, A> + IsView marker | W07T03 | 05-storage §7 |
| W07T06 | `src/storage/viewmut.rs` | ViewMutRepr<'a, A> | W07T05 | 05-storage §8 |
| W07T07 | `src/storage/arc.rs` | ArcRepr<A> shared storage | W07T03 | 05-storage §9 |
| W07T08 | `src/storage/alloc.rs` | 64-byte aligned allocator + module re-exports | W07T04 | 05-storage §10 |

### Wave 08: Tensor Core (L3)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W08T01 | `src/tensor/mod.rs` | Module skeleton (sub-module declarations, public exports) | W07T08, W03T01, W06T04, W04T02 | 07-tensor §7 (T1) |
| W08T02 | `src/tensor/mod.rs` | TensorBase<S, D> struct definition (5 fields: storage, shape, strides, offset, flags) | W08T01 | 07-tensor §7 (T2) |
| W08T03 | `src/tensor/aliases.rs` | Type aliases (Tensor, TensorView, TensorViewMut, ArcTensor + 32 dimension convenience aliases) | W08T02 | 07-tensor §7 (T3) |
| W08T04 | `src/tensor/impls.rs` | Shape & stride query methods: shape()/strides()/ndim()/len()/is_empty()/offset()/raw_dim()/flags()/storage_kind() | W08T02 | 07-tensor §7 (T4) |
| W08T05 | `src/tensor/impls.rs` | Layout query delegate methods: layout_state()/is_f_contiguous()/is_aligned()/has_zero_stride() | W08T04 | 07-tensor §7 (T5) |
| W08T06 | `src/tensor/impls.rs` | Pointer access methods: as_ptr()/as_storage_ptr()/as_mut_ptr() | W08T04 | 07-tensor §7 (T6) |
| W08T07 | `src/tensor/construct.rs` | from_raw_parts (immutable) + from_raw_parts_mut (mutable) with validate_access_range | W08T02 | 07-tensor §7 (T7) |
| W08T08 | `src/tensor/construct.rs` | Safe constructors: from_shape_vec + from_raw_vec_unchecked (internal) | W08T05, W08T07 | 07-tensor §7 (T8) |
| W08T09 | `src/tensor/impls.rs` | View creation methods: view() + view_mut() | W08T06 | 07-tensor §7 (T9) |
| W08T10 | `src/tensor/mod.rs` | Module re-exports + type alias compilation verification | W08T03, W08T09 | 07-tensor §7 (T10) |

### Wave 09: Dispatch (L4)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W09T01 | `src/dispatch.rs` | ExecPath enum + dispatch selection logic | W08T02 | 01-architecture §5.5 |
| W09T02 | `src/dispatch.rs` | ParallelGuard (nested parallelism protection) | W09T01 | 01-architecture §5.5 |
| W09T03 | `src/dispatch.rs` | Parallel threshold constants + integration | W09T02 | 01-architecture §5.5 |

### Wave 10: Broadcasting (L4)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W10T01 | `src/broadcast/mod.rs` | Module skeleton | W08T02 | 15-broadcast |
| W10T02 | `src/broadcast/shape.rs` | can_broadcast function (shape compatibility check) | W10T01, W03T01 | 15-broadcast §2 |
| W10T03 | `src/broadcast/shape.rs` | broadcast_shape function (compute output shape) | W10T02 | 15-broadcast §3 |
| W10T04 | `src/broadcast/view.rs` | broadcast_to method (create broadcast view) | W10T03, W06T04 | 15-broadcast §4 |
| W10T05 | `src/broadcast/view.rs` | broadcast_with method + module re-exports + tests | W10T04 | 15-broadcast §5 |

### Wave 11: Iterators (L4)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W11T01 | `src/iter/mod.rs` | Module skeleton + StrideState struct | W08T02 | 10-iterator §2 |
| W11T02 | `src/iter/elements.rs` | Elements iterator (flat traversal) | W11T01 | 10-iterator §3 |
| W11T03 | `src/iter/axis.rs` | AxisIter (iteration along one axis) | W11T01 | 10-iterator §4 |
| W11T04 | `src/iter/indexed.rs` | IndexedIter (elements with indices) | W11T02 | 10-iterator §5 |
| W11T05 | `src/tensor/impls.rs` | Tensor entry methods for iteration (iter, iter_mut, axis_iter, indexed_iter) | W11T02–W11T04 | 10-iterator §6 |
| W11T06 | `src/iter/mod.rs` | Module re-exports + iter tests | W11T02–W11T04 | 10-iterator |

### Wave 12: FFI Helpers (L4)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W12T01 | `src/ffi/mod.rs` | Module skeleton | W08T02 | 23-ffi |
| W12T02 | `src/ffi/types.rs` | BlasInfo struct definition | W12T01 | 23-ffi §2 |
| W12T03 | `src/ffi/ptr.rs` | export() / export_mut() + into_raw_parts / from_raw_parts | W12T01 | 23-ffi §3 |
| W12T04 | `src/ffi/blas.rs` | is_blas_compatible + blas_info + lda() | W12T02 | 23-ffi §4 |
| W12T05 | `src/ffi/offset.rs` | try_offset_of / try_ptr_at (checked pointer arithmetic) | W12T01 | 23-ffi §5 |
| W12T06 | `src/ffi/mod.rs` | Module re-exports + FFI tests | W12T02–W12T05 | 23-ffi |

### Wave 13: SIMD Backend (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W13T01 | `src/simd/mod.rs` | Module skeleton + SimdKernel trait definition | W08T02 | 08-simd §2 |
| W13T02 | `src/simd/vector.rs` | Element-wise SIMD operations (add, sub, mul, div, abs, neg) | W13T01 | 08-simd §3 |
| W13T03 | `src/simd/vector.rs` | SIMD reduction (sum) + SIMD dot product | W13T02 | 08-simd §4 |
| W13T04 | `src/simd/mod.rs` | Runtime dispatch facade + module re-exports | W13T01 | 08-simd §5 |

### Wave 14: Parallel Backend (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W14T01 | `src/parallel/mod.rs` | Module skeleton + re-exports + feature gate entry | W08T02, W09T03 | 09-parallel §3 |
| W14T02 | `src/parallel/iter.rs` | ParElements struct + TensorBase::par_iter() | W14T01 | 09-parallel §5 (T4) |
| W14T03 | `src/parallel/map.rs` | par_map: parallel element-wise mapping | W14T02 | 09-parallel §5 (T5) |
| W14T04 | `src/parallel/map.rs` | par_zip_map: binary broadcast parallel mapping | W14T03 | 09-parallel §5 (T5a) |
| W14T05 | `src/parallel/reduce.rs` | par_reduce_impl + par_sum | W14T02 | 09-parallel §5 (T6) |
| W14T06 | `src/parallel/reduce.rs` | par_dot: parallel inner product | W14T05 | 09-parallel §5 (T7) |
| W14T07 | `src/parallel/mod.rs` | ParallelPool: rayon ThreadPool wrapper | W14T03, W14T05, W14T06 | 09-parallel §5 (T8) |
| W14T08 | `src/parallel/checked.rs` | Error/panic propagation (XenonError passthrough, no panic swallow) | W14T03, W14T05, W14T06 | 09-parallel §5 (T9) |
| W14T09 | `src/parallel/` + `tests/` | Feature gate + config matrix tests (default off, --features parallel) | W14T02–W14T08 | 09-parallel §5 (T10) |

### Wave 15: Math Operations (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W15T01 | `src/math/mod.rs` | Module skeleton + re-exports | W08T02, W10T05 | 11-math §7 (T1) |
| W15T02 | `src/math/binary.rs` | Binary element-wise ops (add, sub, mul, div, add_scalar, sub_scalar, etc.) | W15T01 | 11-math §7 (T2) |
| W15T03 | `src/math/unary.rs` | Basic unary ops: abs, neg, signum, square | W15T01 | 11-math §7 (T3) |
| W15T04 | `src/math/unary.rs` | Math functions: sin, sqrt, exp, ln, floor, ceil | W15T03 | 11-math §7 (T4) |
| W15T05 | `src/math/unary.rs` | Complex ops: conjugate, modulus | W15T03 | 11-math §7 (T5) |
| W15T06 | `src/math/comparison.rs` | Comparison ops (eq, ne, lt, le, gt, ge element-wise) + not | W15T01 | 11-math §7 (T7) |
| W15T07 | `src/math/mod.rs` | SIMD-accelerated math dispatch + module integration tests | W15T02–W15T06, W13T04 | 11-math §7 (T8) |

### Wave 16: Matrix Operations (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W16T01 | `src/matrix/mod.rs` | Module skeleton | W08T02 | 12-matrix §7 (T1) |
| W16T02 | `src/matrix/dot.rs` | dot() scalar implementation (1D vector inner product) | W16T01 | 12-matrix §7 (T2) |
| W16T03 | `src/matrix/dot.rs` | SIMD dot product integration | W16T02, W13T03 | 12-matrix §7 (T3b) |
| W16T04 | `src/matrix/dot.rs` | Parallel dot product integration | W16T02, W14T04 | 12-matrix §7 (T3c) |
| W16T05 | `src/matrix/mod.rs` | Module re-exports + matrix tests | W16T02 | 12-matrix §7 (T4) |

### Wave 17: Reduction Operations (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W17T01 | `src/reduction/mod.rs` | Module skeleton + public API exports | W08T02 | 13-reduction §7 (T1) |
| W17T02 | `src/reduction/sum.rs` | Global sum() function | W17T01 | 13-reduction §7 (T2) |
| W17T03 | `src/reduction/sum.rs` | sum_axis() function | W17T02 | 13-reduction §7 (T3) |
| W17T04 | `src/reduction/sum.rs` | sum_axis_keepdims() function | W17T03 | 13-reduction §7 (T4) |
| W17T05 | `src/reduction/mod.rs` | SIMD/parallel reduction dispatch | W17T02, W13T03, W14T04 | 13-reduction §7 (T5) |
| W17T06 | `src/reduction/mod.rs` | Error handling + panic convergence + module re-exports + tests | W17T05 | 13-reduction §7 (T6) |

### Wave 18: Set Operations (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W18T01 | `src/set/mod.rs` | Module skeleton | W08T02 | 14-set §7 (T1) |
| W18T02 | `src/set/unique.rs` | unique() function (real types) | W18T01, W11T02 | 14-set §7 (T2) |
| W18T03 | `src/set/unique.rs` | NaN/±0 handling | W18T02 | 14-set §7 (T3) |
| W18T04 | `src/set/unique.rs` | Complex unique | W18T02 | 14-set §7 (T4) |
| W18T05 | `src/set/mod.rs` | TensorBase entry method + module re-exports + tests | W18T02 | 14-set §7 (T5) |

### Wave 19: Shape Operations (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W19T01 | `src/shape/mod.rs` | Module skeleton | W08T02 | 16-shape |
| W19T02 | `src/shape/transpose.rs` | transpose() implementation (axes swap) | W19T01, W06T03 | 16-shape §2 |
| W19T03 | `src/shape/mod.rs` | Module re-exports + shape tests | W19T02 | 16-shape |

### Wave 20: Indexing (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W20T01 | `src/index/mod.rs` + `ndindex.rs` | NdIndex trait definition + tuple/slice impls | W08T02, W03T01 | 17-indexing §2 |
| W20T02 | `src/index/access.rs` | try_at / get / get_unchecked methods | W20T01 | 17-indexing §3 |
| W20T03 | `src/index/slice.rs` | SliceInfo struct + slice_shape/stride computation | W20T01 | 17-indexing §4 |
| W20T04 | `src/index/access.rs` | try_at_mut / get_mut / get_unchecked_mut | W20T02 | 17-indexing §5 |
| W20T05 | `src/index/slice.rs` | slice update methods + module re-exports + index tests | W20T03 | 17-indexing §6 |

### Wave 21: Tensor Construction (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W21T01 | `src/construct/mod.rs` + `init.rs` | Module skeleton + zeros() + ones() | W08T02 | 18-construction §2 |
| W21T02 | `src/construct/eye.rs` | eye() constructor | W21T01 | 18-construction §3 |
| W21T03 | `src/construct/from.rs` | from_shape_vec + from_shape_slice | W21T01 | 18-construction §4 |
| W21T04 | `src/construct/from.rs` + `scalar.rs` | from_array + from_vec + from_scalar | W21T03 | 18-construction §5 |
| W21T05 | `src/construct/mod.rs` | Module re-exports + construction tests | W21T01–W21T04 | 18-construction |

### Wave 22: Operator Overloading (L6)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W22T01 | `src/overload/mod.rs` | Module skeleton | W08T02 | 19-overload |
| W22T02 | `src/overload/arithmetic.rs` | Add\<Tensor, Tensor\> for owned | W22T01, W10T05, W15T02 | 19-overload §2 |
| W22T03 | `src/overload/arithmetic.rs` | Add for ref/mixed (TensorView, &Tensor, etc.) | W22T02 | 19-overload §3 |
| W22T04 | `src/overload/arithmetic.rs` | Add with scalar (Tensor + f64, etc.) | W22T02 | 19-overload §4 |
| W22T05 | `src/overload/arithmetic.rs` | Sub/Mul/Div operators (owned, ref, mixed, scalar) | W22T02–W22T04 | 19-overload §5 |
| W22T06 | `src/overload/mod.rs` | Module re-exports + overload tests | W22T02–W22T05 | 19-overload |

### Wave 23: Utility Operations (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W23T01 | `src/util/mod.rs` + `fill.rs` | Module skeleton + fill() operation | W08T02 | 20-utility §2 |
| W23T02 | `src/util/clip.rs` | clip() operation | W23T01 | 20-utility §3 |
| W23T03 | `src/util/contiguous.rs` | to_contiguous() operation | W23T01, W06T04 | 20-utility §4 |
| W23T04 | `src/util/mod.rs` | Module re-exports + util tests | W23T01–W23T03 | 20-utility |

### Wave 24: Type Conversion (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W24T01 | `src/convert/cast.rs` | CastTo trait core conversion path (lossless + default error path) | W04T02, W08T02 | 21-type §7 (T1) |
| W24T02 | `src/convert/mod.rs` + `src/lib.rs` | Module skeleton + pub use re-exports | W24T01 | 21-type §7 (T2) |
| W24T03 | `src/convert/cast.rs` | to_owned() clone + into_owned() consume (view/arc → owned) | W24T02 | 21-type §7 (T3) |
| W24T04 | `src/convert/cast.rs` | cast<B>(&self) → Result<Tensor<B, D>, XenonError> method | W24T02 | 21-type §7 (T4) |
| W24T05 | `src/convert/cast.rs` | Extended CastTo impls: int↔int, real↔complex, complex↔complex (bool excluded) | W24T01 | 21-type §7 (T5) |

### Wave 25: Output Formatting (L5)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W25T01 | `src/format/mod.rs` + `config.rs` | Module skeleton + FormatConfig struct | W08T02 | 22-output §7 (T1) |
| W25T02 | `src/format/display.rs` | Display impl for tensor (NumPy-style) | W25T01 | 22-output §7 (T3) |
| W25T03 | `src/format/debug.rs` | Debug impl for tensor | W25T01 | 22-output §7 (T4) |
| W25T04 | `src/format/pretty.rs` + `mod.rs` | Pretty formatting helpers + module re-exports + tests | W25T02 | 22-output §7 (T2, T5) |
| W25T05 | `src/format/mod.rs` | Module-level documentation + usage examples | W25T02, W25T03 | 22-output §7 (T5) |

### Wave 26: Workspace (L2)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W26T01 | `src/workspace/error.rs` | WorkspaceErrorCategory + integrate into XenonError::Workspace | W01T03 | 24-workspace §7 (T1) |
| W26T02 | `src/workspace/workspace.rs` | Workspace struct + constants + new() + with_default_capacity() + Drop | W26T01 | 24-workspace §7 (T2) |
| W26T03 | `src/workspace/mod.rs` | Module skeleton + sub-module declarations + re-exports | W26T01 | 24-workspace §7 (T3) |
| W26T04 | `src/workspace/borrow.rs` | WorkspaceBorrow + WorkspaceBorrowMut guards + MaybeUninit access methods | W26T02 | 24-workspace §7 (T4) |
| W26T05 | `src/workspace/split.rs` | split_at_mut() + SplitBorrowMut + recursive split + Drop | W26T02 | 24-workspace §7 (T5) |
| W26T06 | `src/workspace/expand.rs` | ensure_capacity() + reallocate() | W26T02 | 24-workspace §7 (T6) |
| W26T07 | `src/workspace/mod.rs` + sub-modules | Complete public exports + doc comments | W26T04, W26T05, W26T06 | 24-workspace §7 (T7) |

### Wave 27: Safety Audit (cross-cutting)

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W27T01 | `src/storage/owned.rs` | `unsafe impl<A: Send> Send for Owned<A>` + `unsafe impl<A: Sync> Sync for Owned<A>` + full SAFETY comments | W07T04 | 25-safety §5.3 |
| W27T02 | `src/storage/view.rs` | `unsafe impl<'a, A: Sync> Send for ViewRepr<'a, A>` + `unsafe impl<'a, A: Sync> Sync for ViewRepr<'a, A>` + full SAFETY comments | W07T05 | 25-safety §5.4 |
| W27T03 | `src/storage/viewmut.rs` | `unsafe impl<'a, A: Send> Send for ViewMutRepr<'a, A>` + `!Sync` via PhantomData + SAFETY comments | W07T06 | 25-safety §5.5 |
| W27T04 | `src/storage/arc.rs` | `unsafe impl<A: Send + Sync> Send for ArcRepr<A>` + `unsafe impl<A: Send + Sync> Sync for ArcRepr<A>` + SAFETY comments | W07T07 | 25-safety §5.6 |
| W27T05 | `src/parallel/iter.rs` | Parallel chunk safety verification: chunk coverage + non-overlap tests | W27T01–W27T04, W14T05 | 25-safety §6.2 |
| W27T06 | `tests/test_parallel.rs` + `tests/test_error.rs` | Thread safety integration tests: cross-thread move, concurrent access | W27T05 | 25-safety §8.7 |
| W27T07 | `src/storage/mod.rs` | Module-level thread-safety docs + Send/Sync matrix | W27T01–W27T04 | 25-safety §5.1 |

### Wave 28: Benchmarks

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W28T01 | `Cargo.toml` | Add 9 [[bench]] entries, no new benchmark dependencies | None | 27-benchmark §7 (T1) |
| W28T02 | `benches/utils/mod.rs` + `generators.rs` | Shared constants (SIZES_1D/2D/3D) + data generation functions | W28T01 | 27-benchmark §7 (T2) |
| W28T03 | `benches/math.rs` | add/sub/mul/div/sin/exp/abs benches (f32/f64/Complex<f64> + non-contiguous) | W28T02 | 27-benchmark §7 (T3) |
| W28T04 | `benches/reduction.rs` | sum_1d/sum_2d_axis0/sum_2d_axis1/sum_sliced benches | W28T02 | 27-benchmark §7 (T4) |
| W28T05 | `benches/dot.rs` | dot_1d_f64/dot_1d_complex benches | W28T02 | 27-benchmark §7 (T5) |
| W28T06 | `benches/set.rs` | unique_1d benches (various sizes, uniqueness ratios) | W28T02 | 27-benchmark §7 (T6) |
| W28T07 | `benches/broadcast.rs` | broadcast_scalar/broadcast_row/broadcast_col benches | W28T02 | 27-benchmark §7 (T7) |
| W28T08 | `benches/shape.rs` | transpose_2d bench | W28T02 | 27-benchmark §7 (T8) |
| W28T09 | `benches/construction.rs` | zeros_1d/from_shape_vec_1d benches | W28T02 | 27-benchmark §7 (T9) |
| W28T10 | `benches/simd_comparison.rs` | add/sum/dot with --features simd on/off comparison | W28T03, W28T04 | 27-benchmark §7 (T10) |
| W28T11 | `benches/parallel_comparison.rs` | sum/add/dot with --features parallel on/off comparison | W28T03, W28T04 | 27-benchmark §7 (T11) |
| W28T12 | `.github/workflows/bench.yml` | Optional CI benchmark workflow (Smoke/Regression/Full tiers) | W28T03–W28T11 | 27-benchmark §7 (T12) |

### Wave 29: Integration Tests

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W29T01 | `tests/common/mod.rs` + `assertions.rs` + `generators.rs` | Test infra: assertion helpers, tolerance helpers, shape constants, data generators | None | 28-tests §7 (T1) |
| W29T02 | `tests/test_tensor.rs` | Tensor core tests (shape/strides/view/to_owned/type_aliases/debug_display/arc) | W29T01 | 28-tests §7 (T2) |
| W29T03 | `tests/test_math.rs` | Element-wise ops tests (arithmetic/math/comparison/logic) | W29T01 | 28-tests §7 (T3) |
| W29T04 | `tests/test_broadcast.rs` | Broadcast tests (scalar/row/col/incompatible/read-only) | W29T01 | 28-tests §7 (T4) |
| W29T05 | `tests/test_index.rs` | Indexing tests (multi-dim/OOB error/slicing/strides) | W29T01 | 28-tests §7 (T5) |
| W29T06 | `tests/test_construction.rs` | Construction tests (zeros/ones/eye/from_shape_vec/from_scalar/from_array) | W29T01 | 28-tests §7 (T6) |
| W29T07 | `tests/test_reduction.rs` | Reduction tests (sum/sum_axis/keepdims/empty/NaN/overflow) | W29T01 | 28-tests §7 (T7) |
| W29T08 | `tests/test_iterator.rs` | Iterator tests (elements/axis/indexed) | W29T01 | 28-tests §7 (T7a) |
| W29T09 | `tests/test_matrix.rs` | Matrix tests (dot/complex/shape mismatch) | W29T01 | 28-tests §7 (T7b) |
| W29T10 | `tests/test_set.rs` | Set tests (unique/integers/complex/NaN/±0.0) | W29T01 | 28-tests §7 (T7c) |
| W29T11 | `tests/test_shape.rs` | Shape tests (transpose/high-dim) | W29T01 | 28-tests §7 (T8) |
| W29T12 | `tests/test_conversion.rs` | Type conversion tests (cast/to_owned/into_owned) | W29T01 | 28-tests §7 (T9) |
| W29T13 | `tests/test_utility.rs` | Utility tests (fill/clip/to_contiguous) | W29T01 | 28-tests §7 (T9a) |
| W29T14 | `tests/test_output.rs` | Output formatting tests (Display/Debug/truncation/complex) | W29T01 | 28-tests §7 (T9b) |
| W29T15 | `tests/test_error.rs` | XenonError boundary + display output tests | W29T01 | 28-tests §7 (T10) |
| W29T16 | `tests/test_ffi.rs` | FFI tests (pointer/BLAS compat/export/export_mut/offset) | W29T02 | 28-tests §7 (T11) |
| W29T17 | `tests/test_parallel.rs` | Parallel feature tests (sum/add consistency, concurrent read, nested prevention) | W29T03, W29T07 | 28-tests §7 (T12) |
| W29T18 | `tests/test_simd.rs` | SIMD result consistency tests (add/sum/fallback) | W29T03, W29T07 | 28-tests §7 (T13) |
| W29T19 | `.github/workflows/test.yml` | Verify test matrix only covers std environment | W29T02 | 28-tests §7 (T14) |
| W29T20 | `tests/property_tests.rs` + `tests/property/` | Property-based tests (transpose involutive, addition commutative, unique no dupes) | W29T03, W29T07, W29T11 | 28-tests §7 (T15) |
| W29T21 | `.github/workflows/test.yml` | CI test matrix (default/parallel/simd/all-features) | W29T01–W29T20 | 28-tests §7 (T16) |

### Wave 30: Documentation

| Task | File | Goal | Dependencies | Design Docs |
|------|------|------|-------------|-------------|
| W30T01 | `src/lib.rs` | Crate-level docs (overview, Quick Start, Features table, element types, memory layout) | W21 | 29-documentation §7 (T1) |
| W30T02 | `src/lib.rs` + `Cargo.toml` | #![warn(missing_docs)] lint + docs.rs metadata + cfg_attr(docsrs, ...) | W30T01 | 29-documentation §7 (T2) |
| W30T03 | `README.md` | Project intro, Features, Quick Start, install, doc links, license | W30T01 | 29-documentation §7 (T3) |
| W30T04 | `CHANGELOG.md` | Optional Keep a Changelog format (engineering aid, not required deliverable) | None | 29-documentation §7 (T4) |
| W30T05 | Core module mod.rs files | Core module docs: dimension, element, complex, storage, layout | W30T02 | 29-documentation §7 (T5) |
| W30T06 | Tensor/ops module mod.rs files | Tensor & ops module docs: tensor, iter, math, overload, broadcast, reduction, matrix, shape, index, construct, set | W30T02 | 29-documentation §7 (T6) |
| W30T07 | Infra module mod.rs files | Infrastructure module docs: ffi, workspace, error, prelude, convert, format, simd/parallel internal | W30T02 | 29-documentation §7 (T7) |
| W30T08 | `src/tensor/` | Tensor module public API docs (TensorBase, Tensor, TensorView, TensorViewMut, ArcTensor) | W30T05–W30T07 | 29-documentation §7 (T8a) |
| W30T09 | `src/dimension/mod.rs` | Dimension module type docs (Ix0–Ix6, IxDyn, Dimension trait) | W30T05–W30T07 | 29-documentation §7 (T8b) |
| W30T10 | `src/element/mod.rs` | Element module trait docs (Element, Numeric, RealScalar, ComplexScalar) | W30T05–W30T07 | 29-documentation §7 (T8c) |
| W30T11 | `src/storage/mod.rs` | Storage module type docs (Owned, ViewRepr, StorageMut traits) | W30T05–W30T07 | 29-documentation §7 (T8d) |
| W30T12 | `src/layout/mod.rs` | Layout module docs (LayoutFlags, compute_f_strides) | W30T05–W30T07 | 29-documentation §7 (T8e) |
| W30T13 | `src/math/` | Math ops docs (add/sub/mul/div/sin/sqrt/exp/ln/abs) + doctests | W30T05–W30T07 | 29-documentation §7 (T9a) |
| W30T14 | `src/reduction/` + `src/matrix/` | Reduction & matrix docs (sum, sum_axis, dot) + doctests | W30T05–W30T07 | 29-documentation §7 (T9b) |
| W30T15 | `src/broadcast/` + `src/shape/` | Broadcast & shape docs (broadcast_shape, transpose) + doctests | W30T05–W30T07 | 29-documentation §7 (T9c) |
| W30T16 | `src/construct/` + `src/set/` | Construct & set docs (zeros, ones, eye, from_shape_vec, unique) + doctests | W30T05–W30T07 | 29-documentation §7 (T9d) |
| W30T17 | `src/ffi/` + `src/workspace/` + `src/error.rs` | FFI (incl. Safety sections), workspace, XenonError docs + doctests | W30T05–W30T07 | 29-documentation §7 (T9e) |
| W30T18 | `src/iter/` + `src/convert/` + `src/format/` + `src/overload/` | Iter, convert, format, overload module docs + doctests | W30T05–W30T07 | 29-documentation §7 (T9f) |
| W30T19 | `examples/basic.rs` | Basic example: create, compute, reduce, print | W30T01 | 29-documentation §7 (T10) |
| W30T20 | `examples/complex.rs` | Complex example: complex construction, arithmetic, cast | W30T01 | 29-documentation §7 (T11) |
| W30T21 | `examples/broadcasting.rs` | Broadcasting example: rules, row/col/scalar broadcast | W30T01 | 29-documentation §7 (T12) |
| W30T22 | `examples/features.rs` | Features example: optional features, parallel/simd execution paths | W30T01 | 29-documentation §7 (T13) |
| W30T23 | `examples/simd.rs` | SIMD example: acceleration, fallback strategy | W30T01 | 29-documentation §7 (T14) |
| W30T24 | `examples/ffi.rs` | FFI example: upstream C/BLAS integration helper API | W30T01 | 29-documentation §7 (T15) |
| W30T25 | `src/lib.rs` + `README.md` + `examples/` | Verify examples & crate docs only declare std environment | W30T01 | 29-documentation §7 (T16) |
| W30T26 | `.github/workflows/docs.yml` | CI docs workflow: missing docs check, doctest, example compilation | W30T01–W30T25 | 29-documentation §7 (T17) |

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
