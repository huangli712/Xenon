# Xenon 实施次序文件 (PLAN)

> 基于 `SUMMARY.md` 的任务依赖关系生成。  
> `∥` 标记表示可并行执行。  
> 同 Wave 内数字为相对执行批次（同批可并行）。

---

## 总体 Wave 执行次序

```
W1 ────┬──→ W2 ──→ W9
       ├──→ W3 ──→ W6 ──→ W7 ────┬──→ W8 ──→ W10 ──→ W15
       └──→ W5                   │            │
                                 │            ├──→ W11 ──→ W16 ──→ W23
                                 │            ├──→ W12 ──→ W19
                                 │            ├──→ W13
                                 │            ├──→ W20
                                 │            ├──→ W21
                                 │            ├──→ W22 ──→ W28
                                 │            ├──→ W24
                                 │            ├──→ W25
                                 │            └──→ W26
                                 │
                                 └──→ W4 ──────────────────→ W14 ──→ W16

W1–W26 全部完成 ──→ W27 (Safety Audit)
W22 完成 ──→ W28 (Benchmarks)
W22 完成 ──→ W29 (Integration Tests)
W22 完成 ──→ W30 (Documentation)
```

---

## Wave 级并行要点

| 可并行组 | Waves | 条件 |
|----------|-------|------|
| A | W3, W5 | 均在 W1 之后，互不依赖 |
| B | W6, W9 | W6 需 W3, W9 需 W2; 可并行 |
| C | W4, W7 | W4 需 W3+W5, W7 需 W3+W6; W6 先于 W7, W5 先于 W4 |
| D | W11,W12,W13,W14,W20,W21,W22,W24,W25,W26 | 全部在 W8 之后，互相独立 |
| E | W28, W29, W30 | 全部在 W22 之后，互相独立 |

---

## 各 Wave 内部 Task 执行次序

### W1: Coding Standards & Project Setup (L0)

```
批次1:
  W1T1 (Cargo.toml)
批次2 (并行):
  W1T2 (rustfmt.toml, 需 W1T1)
  W1T3 (lib.rs, 需 W1T1)
  W1T5 (.clippy.toml, 需 W1T1)
批次3 (并行):
  W1T4 (prelude.rs, 需 W1T3)
  W1T6 (CI, 需 W1T3)
```

---

### W2: Error System (L0)

```
批次1:
  W2T1 (XenonError enum)
批次2:
  W2T2 (aux enums+Result, 需 W2T1)
批次3 (并行):
  W2T3 (Display, 需 W2T2)
  W2T4 (Error impl, 需 W2T2)
批次4:
  W2T5 (prelude exports, 需 W2T4)
```

---

### W3: Dimension System (L1)

```
批次1:
  W3T1 (mod.rs skeleton)
批次2:
  W3T2 (Dimension trait, 需 W3T1)
批次3 (并行):
  W3T3 (Ix0, 需 W3T2)
  W3T10 (IxDyn, 需 W3T2)
  W3T14 (Axis, 需 W3T1)
批次4:
  W3T4 (Ix1, 需 W3T3)
批次5:
  W3T5 (Ix2, 需 W3T4)
批次6:
  W3T6 (Ix3, 需 W3T5)
批次7:
  W3T7 (Ix4, 需 W3T6)
批次8:
  W3T8 (Ix5, 需 W3T7)
批次9:
  W3T9 (Ix6, 需 W3T8)
批次10:
  W3T11 (into_dyn/try_from_dyn, 需 W3T9+W3T10)
批次11:
  W3T12 (error integration, 需 W2T1+W3T11)
批次12:
  W3T13 (IntoDimension, 需 W3T12)
批次13:
  W3T15 (Sealed+exports, 需 W3T12+W3T13+W3T14)
批次14:
  W3T16 (doc comments, 需 W3T15)
批次15 (全并行):
  W3T17 (in-module tests, 需 W3T16)
  W3T18 (test_tensor, 需 W3T16)
  W3T19 (test_shape, 需 W3T16)
  W3T20 (test_index, 需 W3T16)
  W3T21 (property tests, 需 W3T16)
```

> **注**: W3T3–W3T10 中，Ix0 和 IxDyn 可并行（均仅依赖 Dimension trait），
> 其余 Ix1–Ix6 必须串行（每个依赖前一个）。

---

### W4: Element Type Hierarchy (L1)

```
批次1:
  W4T1 (Element trait, 需 W3T15+W2T1)
批次2:
  W4T2 (Numeric trait, 需 W4T1)
批次3 (并行):
  W4T3 (RealScalar, 需 W4T2)
  W4T4 (ComplexScalar, 需 W4T2)
  W4T5 (i32 impl, 需 W4T2)
  W4T8 (bool impl, 需 W4T1)
  W4T9 (usize doc, 需 W4T1)
批次4:
  W4T6 (i64 impl, 需 W4T5)
批次5:
  W4T7 (f32+f64 impls, 需 W4T3+W4T6)
批次6:
  W4T10 (Complex impls, 需 W4T4+W5T1)
批次7:
  W4T11 (calibrate+doc, 需 W4T7)
批次8:
  W4T12 (all doc comments, 需 W4T10+W4T11)
批次9 (全并行):
  W4T13 (in-module tests, 需 W4T10)
  W4T14 (test_tensor, 需 W4T10)
  W4T15 (test_math, 需 W4T10)
  W4T16 (test_reduction, 需 W4T10)
  W4T17 (test_conversion, 需 W4T10)
```

> **注**: W4T10 依赖 W5T1（Complex 类型），需要 W5 先完成 W5T1。

---

### W5: Complex Type (L1)

```
批次1:
  W5T1 (Complex struct+new, 需 W1T3)
批次2 (并行):
  W5T2 (ComplexFloat sealed, 需 W5T1+W3T15)
  W5T3 (FFI assertions, 需 W5T1)
  W5T4 (re/im, 需 W5T1)
  W5T6 (is_real/is_imaginary, 需 W5T1)
  W5T7 (PartialEq+Display, 需 W5T1)
批次3:
  W5T5 (from_imag/conj/From, 需 W5T1+W5T4)
批次4 (并行=Add/Mul/Neg 无相互依赖):
  W5T8 (Add, 需 W5T1)
  W5T10 (Mul, 需 W5T1)
  W5T12 (Neg, 需 W5T1)
批次5:
  W5T9 (Sub, 需 W5T8)
批次6:
  W5T11 (Div, 需 W5T10)
批次7:
  W5T13 (tighten boundary, 需 W5T8–W5T12)
批次8:
  W5T14 (norm/norm_sqr, 需 W5T1)
批次9:
  W5T15 (doc comments, 需 W5T13+W5T14)
批次10:
  W5T16 (integration tests, 需 W5T15)
```

---

### W6: Layout System (L2)

```
批次1:
  W6T1 (mod.rs skeleton)
批次2 (全并行):
  W6T2 (flags.rs skeleton, 需 W6T1)
  W6T3 (strides.rs skeleton, 需 W6T1)
  W6T4 (contiguous.rs skeleton, 需 W6T1)
批次3:
  W6T5 (LayoutFlags, 需 W6T2)
批次4 (全并行):
  W6T6 (compute_f_strides, 需 W6T3+W3T2)
  W6T7 (is_f_contiguous, 需 W6T4+W3T2)
  W6T8 (has_zero_stride, 需 W6T3+W3T2)
  W6T9 (is_aligned, 需 W6T3)
批次5:
  W6T10 (integration tests, 需 W6T6–W6T9)
```

---

### W7: Storage System (L2)

```
批次1:
  W7T1 (mod.rs skeleton)
批次2:
  W7T2 (RawStorage trait, 需 W7T1)
批次3:
  W7T3 (Storage trait, 需 W7T2)
批次4:
  W7T4 (RawStorageMut+StorageMut, 需 W7T3)
批次5:
  W7T5 (StorageOwned+StorageShared, 需 W7T4)
批次6 (并行):
  W7T6 (marker traits, 需 W7T5)
  W7T7 (AlignedAlloc, 需 W7T1)
批次7:
  W7T8 (Owned struct, 需 W7T7)
批次8 (并行):
  W7T14 (ViewRepr, 需 W7T5+W7T6)
  W7T15 (ViewMutRepr, 需 W7T5+W7T6)
  W7T16 (ArcRepr, 需 W7T5)
批次9:
  W7T9 (Owned RawStorage impl, 需 W7T5+W7T8)
批次10:
  W7T10 (Owned Storage impl, 需 W7T9)
批次11:
  W7T11 (Owned StorageMut+StorageOwned, 需 W7T10)
批次12 (并行):
  W7T12 (into_shared+Send+Sync, 需 W7T8+W7T16)
  W7T13 (From+Default, 需 W7T8)
  W7T17 (ArcRepr Send/Sync+Default+From, 需 W7T16)
批次13:
  W7T18 (re-exports+doc, 需 W7T13+W7T14+W7T15+W7T17)
批次14:
  W7T19 (integration tests, 需 W7T18)
```

---

### W8: Tensor Core (L3)

```
批次1:
  W8T1 (mod.rs skeleton)
批次2:
  W8T2 (TensorBase struct, 需 W8T1+W6T5)
批次3 (并行):
  W8T3 (type aliases, 需 W8T2)
  W8T4 (query methods, 需 W8T2+W3T2+W6T5)
  W8T7 (from_raw_parts, 需 W8T2+W6T5+W3T2)
批次4 (并行):
  W8T5 (layout delegation, 需 W8T4)
  W8T6 (pointer access, 需 W8T4)
批次5:
  W8T8 (from_raw_vec_unchecked, 需 W8T5+W8T7)
批次6:
  W8T9 (view/view_mut, 需 W8T6)
批次7:
  W8T10 (integration tests, 需 W8T3+W8T9)
```

---

### W9: Workspace (L2)

```
批次1:
  W9T1 (WorkspaceErrorCategory, 需 W2T2)
批次2 (并行):
  W9T2 (Workspace struct, 需 W9T1)
  W9T3 (mod.rs, 需 W9T1)
批次3 (全并行):
  W9T4 (borrow guards, 需 W9T2+W9T1)
  W9T5 (split guard, 需 W9T2+W9T1)
  W9T6 (expand, 需 W9T2+W9T1)
批次4:
  W9T7 (exports+doc, 需 W9T4+W9T5+W9T6)
```

---

### W10: Dispatch (L4)

```
批次1:
  W10T1 (mod.rs skeleton)
批次2 (并行):
  W10T2 (ParallelGuard, 需 W10T1)
  W10T3 (ParallelExecStrategy, 需 W10T1)
批次3:
  W10T4 (select_exec_path, 需 W10T1+W10T2)
批次4:
  W10T5 (threshold config, 需 W10T4)
批次5:
  W10T6 (unit tests, 需 W10T2+W10T3+W10T4+W10T5)
```

---

### W11: Broadcasting (L4)

```
批次1:
  W11T1 (mod.rs skeleton)
批次2 (并行):
  W11T2 (shape.rs skeleton, 需 W11T1)
  W11T3 (view.rs skeleton, 需 W11T1)
批次3:
  W11T4 (can_broadcast, 需 W11T2)
批次4:
  W11T5 (broadcast_shape, 需 W11T4)
批次5:
  W11T6 (broadcast_strides, 需 W11T5)
批次6:
  W11T7 (broadcast_to basic, 需 W11T6)
批次7:
  W11T8 (broadcast_to error, 需 W11T7)
批次8:
  W11T9 (broadcast_with, 需 W11T6+W11T7)
批次9:
  W11T10 (tests, 需 W11T8+W11T9)
```

---

### W12: Iterators (L4)

```
批次1:
  W12T1 (mod.rs skeleton)
批次2:
  W12T2 (StrideState, 需 W12T1)
批次3:
  W12T5 (AxisIter/AxisIterMut, 需 W12T1)
批次4:
  W12T3 (Iter, 需 W12T2)
批次5:
  W12T4 (IterMut, 需 W12T3)
批次6:
  W12T6 (IndexedIter, 需 W12T4)
批次7:
  W12T7 (entry methods, 需 W12T4+W12T5+W12T6)
```

---

### W13: FFI Helpers (L4)

```
批次1:
  W13T1 (mod.rs, 需 W8T7)
批次2:
  W13T2 (types, 需 W13T1)
批次3:
  W13T3 (private descriptors, 需 W13T2)
批次4:
  W13T4 (ptr re-exports, 需 W13T3)
批次5 (并行):
  W13T5 (BLAS helpers, 需 W13T2)
  W13T6 (offset helpers, 需 W13T2)
```

---

### W14: SIMD Backend (L5)

```
批次1:
  W14T1 (mod.rs skeleton)
批次2:
  W14T2 (element-wise SIMD, 需 W14T1)
批次3:
  W14T3 (float sum SIMD, 需 W14T2)
批次4 (并行):
  W14T4 (integer sum/dot, 需 W14T3)
  W14T5 (complex sum, 需 W14T3)
批次5:
  W14T6 (float+complex dot, 需 W14T3+W14T5)
批次6:
  W14T7 (feature gate+exports, 需 W14T1+W14T2)
批次7:
  W14T8 (consistency tests, 需 W14T7)
批次8:
  W14T9 (reduction/dot tests, 需 W14T8)
批次9:
  W14T10 (property tests, 需 W14T9)
```

---

### W15: Parallel Backend (L5)

```
批次1:
  W15T1 (ParIter, 需 W10T4)
批次2 (并行):
  W15T2 (par_map, 需 W15T1)
  W15T3 (par_zip_map, 需 W15T1)
  W15T4 (par_reduce+par_sum, 需 W15T1)
批次3:
  W15T5 (par_dot, 需 W15T4)
批次4 (并行):
  W15T6 (ParallelPool, 需 W15T2+W15T4+W15T5)
  W15T7 (error propagation, 需 W15T2+W15T4+W15T5)
批次5:
  W15T8 (feature gate tests, 需 W15T1–W15T7)
```

---

### W16: Math Operations (L5)

```
批次1:
  W16T1 (mod.rs skeleton)
批次2:
  W16T2 (binary skeleton, 需 W16T1)
批次3 (全并行 — 均为独立函数):
  W16T3 (abs/neg/signum/square, 需 W16T1)
  W16T4 (sin/sqrt/exp/ln/floor/ceil, 需 W16T1)
  W16T5 (conjugate/modulus, 需 W16T1)
  W16T7 (logical not, 需 W16T1)
批次4:
  W16T6 (add/sub/mul/div, 需 W16T2)
批次5:
  W16T8 (equal/not_equal, 需 W16T2)
批次6 (并行):
  W16T9 (less/less_equal, 需 W16T8)
  W16T10 (greater/greater_equal, 需 W16T8)
批次7:
  W16T11 (SIMD dispatch integration, 需 W16T6+W14)
```

---

### W17: Matrix Operations (L5)

```
批次1:
  W17T1 (mod.rs)
批次2:
  W17T2 (dot.rs skeleton, 需 W17T1)
批次3:
  W17T3 (dot base, 需 W17T2)
批次4:
  W17T4 (scalar path, 需 W17T3)
批次5 (并行):
  W17T5 (SIMD path, 需 W17T4+W14)
  W17T6 (parallel path, 需 W17T4+W15)
批次6:
  W17T7 (tests, 需 W17T3–W17T6)
```

---

### W18: Reduction Operations (L5)

```
批次1:
  W18T1 (mod.rs)
批次2:
  W18T2 (sum, 需 W18T1)
批次3:
  W18T3 (sum_axis, 需 W18T2)
批次4:
  W18T4 (sum_axis_keepdims, 需 W18T3)
批次5:
  W18T5 (SIMD/parallel guards, 需 W18T4+W14+W15)
批次6:
  W18T6 (error convergence, 需 W18T3–W18T5)
```

---

### W19: Set Operations (L5)

```
批次1:
  W19T1 (mod.rs)
批次2:
  W19T2 (UniqueElement trait, 需 W19T1)
批次3:
  W19T3 (unique core, 需 W19T2)
批次4 (并行):
  W19T4 (float NaN/±0.0, 需 W19T3)
  W19T5 (complex equality, 需 W19T3)
批次5:
  W19T6 (TensorBase entry, 需 W19T3–W19T5)
```

---

### W20: Shape Operations (L5)

```
批次1:
  W20T1 (mod.rs)
批次2:
  W20T2 (transpose.rs skeleton, 需 W20T1)
批次3:
  W20T3 (transpose, 需 W20T2)
批次4:
  W20T4 (tests, 需 W20T3)
```

---

### W21: Indexing (L5)

```
批次1:
  W21T1 (mod.rs)
批次2:
  W21T2 (NdIndex trait, 需 W21T1)
批次3 (并行):
  W21T3 (try_at/get, 需 W21T2)
  W21T4 (SliceInfo, 需 W21T2)
批次4 (并行):
  W21T5 (try_at_mut/get_mut, 需 W21T3)
  W21T6 (slice shape update, 需 W21T4)
```

---

### W22: Tensor Construction (L5)

```
批次1:
  W22T1 (mod.rs+init.rs skeleton)
批次2 (全并行):
  W22T2 (zeros, 需 W22T1)
  W22T3 (ones, 需 W22T1)
  W22T4 (eye, 需 W22T1)
  W22T5 (from_shape_vec+from_vec, 需 W22T1)
  W22T8 (from_scalar, 需 W22T1)
批次3:
  W22T6 (from_shape_slice, 需 W22T5)
批次4:
  W22T7 (from_array, 需 W22T6)
批次5:
  W22T9 (tests, 需 W22T2–W22T8)
```

---

### W23: Operator Overloading (L6)

```
批次1:
  W23T1 (mod.rs)
批次2:
  W23T2 (arithmetic.rs skeleton, 需 W23T1)
批次3:
  W23T3 (Add owned, 需 W23T2)
批次4 (并行):
  W23T4 (Add ref/mixed, 需 W23T3)
  W23T5 (Add scalar, 需 W23T3)
批次5 (全并行 — 复用同一模式):
  W23T6 (Sub, 需 W23T4+W23T5)
  W23T7 (Mul, 需 W23T4+W23T5)
  W23T8 (Div, 需 W23T4+W23T5)
批次6:
  W23T9 (tests, 需 W23T1–W23T8)
```

---

### W24: Utility Operations (L5)

```
批次1:
  W24T1 (mod.rs)
批次2 (并行 — 独立文件):
  W24T2 (fill, 需 W24T1+W7+W8+W12+W2)
  W24T3 (clip, 需 W24T1+W4+W8+W12+W2)
  W24T4 (to/into_contiguous, 需 W24T1+W6+W8+W25)
批次3:
  W24T5 (tests, 需 W24T2–W24T4)
```

---

### W25: Type Conversion (L5)

```
批次1:
  W25T1 (mod.rs+lib.rs exports)
批次2:
  W25T2 (CastTo+ConvertTo trait defs, 需 W25T1)
批次3 (全并行 — 三个 tier 独立):
  W25T3 (Tier-1 lossless 11 cells, 需 W25T2)
  W25T4 (Tier-2 lossy 14 cells, 需 W25T2)
  W25T5 (Tier-3 dynamic 8 cells, 需 W25T2)
批次4 (并行):
  W25T6 (cast method, 需 W25T3+W25T4+W25T5)
  W25T7 (to_owned/into_owned, 需 W25T1)
```

---

### W26: Output Formatting (L5)

```
批次1:
  W26T1 (mod.rs)
批次2:
  W26T2 (FormatConfig, 需 W26T1)
批次3:
  W26T3 (pretty helpers, 需 W26T2)
批次4 (并行):
  W26T4 (Display impl, 需 W26T3)
  W26T5 (Debug impl, 需 W26T3)
批次5:
  W26T6 (docs+re-exports, 需 W26T4+W26T5)
```

---

### W27: Safety Audit (cross-cutting)

```
批次1 (全并行):
  W27T1 (Owned Send+Sync, 需 W7T12)
  W27T2 (ViewRepr Send+Sync, 需 W7T14)
  W27T3 (ViewMutRepr Send, 需 W7T15)
  W27T4 (ArcRepr Send+Sync, 需 W7T17)
批次2:
  W27T5 (parallel chunk safety, 需 W27T1–W27T4)
批次3 (并行):
  W27T6 (thread-safety tests, 需 W27T1–W27T5)
  W27T7 (Send/Sync docs, 需 W27T1–W27T4)
```

---

### W28: Benchmarks

```
批次1:
  W28T1 (Cargo.toml bench entries)
批次2:
  W28T2 (utils/generators, 需 W28T1)
批次3 (全并行 — 独立 bench 文件):
  W28T3 (math bench)
  W28T4 (reduction bench)
  W28T5 (dot bench)
  W28T6 (set bench)
  W28T7 (broadcast bench)
  W28T8 (shape bench)
  W28T9 (construction bench)
  (全部需 W28T2)
批次4 (并行):
  W28T10 (SIMD comparison, 需 W28T3+W28T4+W28T5+W14)
  W28T11 (parallel comparison, 需 W28T3+W28T4+W28T5+W15)
批次5:
  W28T12 (CI bench+report, 需 W28T3–W28T11)
```

---

### W29: Integration Tests

```
批次1:
  W29T1 (test infrastructure)
批次2 (全并行 — 独立 test 文件):
  W29T2 (test_tensor)
  W29T3 (test_math)
  W29T5 (test_broadcast)
  W29T6 (test_index)
  W29T7 (test_construction)
  W29T8 (test_reduction)
  W29T9 (test_iterator)
  W29T10 (test_matrix)
  W29T11 (test_set)
  W29T12 (test_shape)
  W29T13 (test_conversion)
  W29T14 (test_utility)
  W29T15 (test_output)
  W29T16 (test_error)
  (全部需 W29T1)
批次3 (并行):
  W29T4 (test_overload, 需 W29T1+W29T3)
  W29T17 (test_workspace, 需 W29T1+W29T2)
  W29T18 (test_ffi, 需 W29T2)
  W29T19 (test_parallel, 需 W29T3+W29T8)
  W29T20 (test_simd, 需 W29T3+W29T8)
  W29T22 (property tests, 需 W29T3+W29T8+W29T12)
  W29T24 (compile-fail harness, 需 W29T1)
批次4:
  W29T21 (CI test matrix, 需 W29T2)
批次5:
  W29T25 (compile-fail fixtures, 需 W29T24)
批次6:
  W29T23 (CI full matrix, 需 W29T1–W29T22)
```

---

### W30: Documentation

```
批次1 (全并行):
  W30T1 (crate-level docs)
  W30T4 (CHANGELOG)
批次2:
  W30T2 (missing_docs+doxcs.rs, 需 W30T1)
批次3:
  W30T3 (README, 需 W30T1)
批次4 (全并行 — 24 个模块级文档):
  W30T5–W30T28 (所有模块级文档, 需 W30T2)
批次5 (全并行 — 13 个类型/函数级文档+doctests):
  W30T29–W30T41 (需 W30T5–W30T28)
批次6:
  W30T42 (util function docs, 需 W30T26)
批次7 (全并行 — 7 个示例):
  W30T43–W30T49 (需 W30T1)
批次8:
  W30T50 (std env audit, 需 W30T1+W30T3+W30T43–W30T49)
批次9 (并行):
  W30T51 (LICENSE, 需 W1T1)
  W30T52 (docs CI, 需 W30T40+W30T43–W30T49)
```
