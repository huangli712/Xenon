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
| D | W11,W12,W13,W14,W20,W21,W22,W24,W25,W26 | 全部在 W8 之后；除 W25 与 W26 需要 W12T7（`TensorBase::iter()` 入口方法）外，组内互相独立。**例外：W13T4 依赖 W22T10 (OwnedRawParts)**，详见 W13 与 W22 节 |
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
  W2T1 (aux enums)
批次2:
  W2T2 (XenonError enum + Result, 需 W2T1)
批次3:
  W2T3 (Display, 需 W2T2)
批次4:
  W2T4 (Error impl, 需 W2T3)
批次5:
  W2T5 (prelude exports, 需 W2T4)
```

---

### W3: Dimension System (L1)

```
批次1:
  W3T1 (mod.rs skeleton)
批次2:
  W3T2 (Axis, 需 W3T1)
批次3:
  W3T3 (Dimension trait, 需 W3T1+W3T2)
批次4 (并行):
  W3T4 (Ix0, 需 W3T3)
  W3T11 (IxDyn, 需 W3T3)
批次5:
  W3T5 (Ix1, 需 W3T4)
批次6:
  W3T6 (Ix2, 需 W3T5)
批次7:
  W3T7 (Ix3, 需 W3T6)
批次8:
  W3T8 (Ix4, 需 W3T7)
批次9:
  W3T9 (Ix5, 需 W3T8)
批次10:
  W3T10 (Ix6, 需 W3T9)
批次11:
  W3T12 (into_dyn/try_from_dyn, 需 W3T10+W3T11)
批次12:
  W3T13 (DimensionMismatch verification, 需 W2T1+W3T12)
批次13:
  W3T14 (IntoDimension, 需 W3T13)
批次14:
  W3T15 (Sealed+exports, 需 W3T13+W3T14+W3T2)
批次15:
  W3T16 (doc comments, 需 W3T15)
批次16 (全并行):
  W3T17 (in-module tests, 需 W3T16)
  W3T18 (test_tensor, 需 W3T16)
  W3T19 (test_shape, 需 W3T16+W3T22)  # BroadcastDim placeholder 在 W3T22 完成后激活
  W3T20 (test_index, 需 W3T16)
  W3T21 (property tests, 需 W3T16)
  W3T22 (BroadcastDim trait + impl matrix + 对称性测试, 需 W3T15+W3T16)
```

> **注1**: W3T2 (Axis) 置于批次2，因为 W3T3 的 `Dimension::axis()` 签名依赖 `Axis` 类型。
> W3T2 仅需 W3T1 (模块骨架) 即可实现，不依赖任何维度类型。
> 
> **注2**: W3T4–W3T11 中，Ix0 和 IxDyn 可并行（均仅依赖 Dimension trait），
> 其余 Ix1–Ix6 必须串行（每个依赖前一个）。
> 
> **注3**: `DimensionMismatch` enum variant 已在 W2T1 中预先添加（见 `26-error.md`）。
> W3T13 改为校验型任务：核对 W2T1 已添加的 `DimensionMismatch` 字段（特别是 `operation: Cow<'static, str>`）符合设计文档，必要时补齐。因此 W3T13 在 W3T12 之后执行，验证 W3T12 实际使用 variant 的字段构造是否正确。
>
> **注4**: `into_dyn` 与 `try_from_dyn` 不在 `Dimension` trait 中定义；它们是各具体类型（`Ix0`-`Ix6`、`IxDyn`）的 inherent 方法。W3T12 集中实现这些 inherent 方法。详见 `02-dimension.md` §5.1 设计决策。
>
> **注5**: W3T22 在 Wave 11 审计修复（docs_fix 分支）中补充。原 SUMMARY.md `W11 ... BroadcastDim trait` 描述无对应 task，且 W3T19 placeholder 与 W11T3/W11T9 调用方都依赖该上游 trait。W3T22 归到 W3 batch 16（与 W3T17-W3T21 并行）是因为 `BroadcastDim` 是 dimension 模块的类型层逻辑（与 `Dimension`、`Sealed`、`IntoDimension` 同源），不属于 W11 广播运行时范畴。详见 `02-dimension.md §5.10`。

---

### W4: Element Type Hierarchy (L1)

```
批次1:
  W4T1 (Element trait, 需 W3T15+W2T1)
批次2:
  W4T2 (Numeric trait, 需 W4T1)
批次3 (并行):
  W4T3 (RealScalar, 需 W4T2)
  W4T5 (i32 impl, 需 W4T2)
  W4T8 (bool impl, 需 W4T1)
批次4 (并行):
  W4T4 (ComplexScalar, 需 W4T2+W4T3)
  W4T6 (i64 impl, 需 W4T5)
批次5:
  W4T7 (f32+f64 impls, 需 W4T3+W4T6)
批次6 (并行):
  W4T9 (usize doc, 需 W4T7)
  W4T10 (Complex impls, 需 W4T4+W5T1)
批次7:
  W4T11 (calibrate+doc+marker impls+§5.10 checked traits, 需 W4T7+W4T10)
批次8:
  W4T12 (all doc comments, 需 W4T10+W4T11)
批次9 (全并行):
  W4T13 (in-module tests, 需 W4T10+W4T11)
  W4T14 (test_tensor, 需 W4T10+W8)
  W4T15 (test_math, 需 W4T10)
  W4T16 (test_reduction, 需 W4T10+W8)
  W4T17 (test_conversion, 需 W4T10+W8)
```

> **注**: W4T10 依赖 W5T1（Complex 类型），需要 W5 先完成 W5T1。
> **注**: W4T14/W4T16/W4T17 跨 Wave 依赖 W8 (Tensor Core)，需在 Wave 4 实施时带 `#[ignore]` 标记，等 W8 完成后激活。

---

### W5: Complex Type (L1)

```
批次1:
  W5T1 (Complex struct+new+minimal ComplexFloat+lib.rs wiring, 需 W1T3)
批次2 (并行):
  W5T2 (Extend ComplexFloat supertraits + sealed compile_fail doctest, 需 W5T1+W3T15)
  W5T3 (FFI assertions + field offset tests, 需 W5T1)
  W5T4 (re/im, 需 W5T1)
批次3 (并行):
  W5T6 (is_real/is_imaginary, 需 W5T1+W5T2)
  W5T7 (PartialEq+Display+PositiveZero, 需 W5T1+W5T2)
批次4:
  W5T5 (from_imag/conj/From, 需 W5T1+W5T2+W5T4+W5T7)
批次5 (并行=Add/Mul/Neg 无相互依赖):
  W5T8 (Add, 需 W5T1+W5T2+W5T7)
  W5T10 (Mul, 需 W5T1+W5T2+W5T7)
  W5T12 (Neg, 需 W5T1+W5T2+W5T7)
批次6:
  W5T9 (Sub, 需 W5T8)
批次7:
  W5T11 (Div, 需 W5T10)
批次8:
  W5T13 (tighten boundary, 需 W5T8–W5T12)
批次9:
  W5T14 (norm/norm_sqr+is_nan/is_finite, 需 W5T1)
批次10:
  W5T15 (doc comments, 需 W5T13+W5T14)
批次11:
  W5T16 (integration tests, 需 W5T15)
```

---

### W6: Layout System (L2)

```
批次1:
  W6T1 (mod.rs skeleton)
批次2 (W6T2 ∥ W6T3 并行; W6T4 需 W6T3 先完成):
  W6T2 (flags.rs skeleton, 需 W6T1)
  W6T3 (strides.rs skeleton, 需 W6T1+W3T3)
  W6T4 (contiguous.rs skeleton, 需 W6T1+W6T3+W3T3)
批次3:
  W6T5 (LayoutFlags, 需 W6T2)
批次4:
  # strides.rs 同文件三任务必须串行执行（避免 git merge 冲突）；
  # 三者间无逻辑依赖，只是物理上共享文件。执行顺序任选（推荐 W6T6→W6T8→W6T9）。
  W6T6 (compute_f_strides, 需 W6T3+W3T3)         [strides.rs]
  W6T8 (has_zero_stride, 需 W6T3+W3T3)           [strides.rs，与 W6T6 串行]
  W6T9 (is_aligned, 需 W6T3)                     [strides.rs，与 W6T6/W6T8 串行]
  # W6T7 写入独立文件 contiguous.rs，可与 strides.rs 三任务并行执行
  W6T7 (is_f_contiguous, 需 W6T4+W3T3)           [contiguous.rs，独立并行]
批次5:
  W6T11 (compute_layout_flags + flags_for_f_layout, 需 W6T5+W6T6+W6T7+W6T8+W6T9)
批次6:
  W6T10 (integration tests, 需 W6T6–W6T9+W6T11)
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
  W7T16 (ArcRepr, 需 W7T5+W7T7)
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
  W8T4 (query methods, 需 W8T2+W3T3+W6T5)
  W8T7 (from_raw_parts, 需 W8T2+W6T5+W3T3)
批次4 (并行):
  W8T5 (layout delegation, 需 W8T4)
  W8T6 (pointer access, 需 W8T4)
批次5:
  W8T8 (from_raw_vec_unchecked, 需 W8T5+W8T7+W6T5)
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
  W9T2 (mod.rs, 需 W9T1)
  W9T3 (Workspace struct, 需 W9T1+W9T2)
批次3:
  W9T4 (borrow guards, 需 W9T3+W9T1)
批次4 (并行):
  W9T5 (split guard, 需 W9T3+W9T1+W9T4)
  W9T6 (expand, 需 W9T3+W9T1+W9T4)
批次5:
  W9T7 (exports+doc, 需 W9T4+W9T5+W9T6)
```

**编号说明**：本 PLAN 的 W9T2/W9T3 编号与设计文档 `24-workspace.md §7` 的 T2/T3 **反向**（设计 T2=workspace.rs/T3=mod.rs；本 PLAN W9T2=mod.rs/W9T3=workspace.rs）。**理由**：批次2 中 mod.rs 拓扑应先于 workspace.rs，便于后续 `use super::workspace` 路径解析；且 SUMMARY.md 与所有 W9T*.md 任务文件已采用此编号，保持一致性。

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
  W11T7 (broadcast_to basic, 需 W11T3+W11T6)
批次7:
  W11T8 (broadcast_to error, 需 W11T7)
批次8:
  W11T9 (broadcast_with, 需 W11T6+W11T7)
批次9:
  W11T10 (tests, 需 W11T8+W11T9+W12T*+W22T*)
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

> **Traceability**: `23-ffi.md` §7 把 Wave 1 的 mod.rs / types.rs / FfiErrorCategory /
> BlasInfo 合并为单个设计 task。本 PLAN 内部进一步拆分：W13T1 = mod.rs 骨架（仅
> 声明 5 个子模块，遵循 W1T3 模块演进协议），W13T2 = types.rs（C-visible raw
> 描述符 + BlasInfo + re-exports），W13T3 = private.rs（generic Rust-only 描述符）。
> 这一拆分纯属实施粒度细分，不改变设计 §7 的语义边界，便于评审与并行/串行依赖追踪。
>
> **重要 API 形式**：W13T4 / W13T5 / W13T6 全部以 `TensorBase` 的 inherent methods
> 形式提供（`tensor.export()` / `tensor.blas_info()?` / `tensor.try_offset_of(&[..])?`），
> 严格遵循 `23-ffi.md` §5.4 / §5.10-§5.13。`from_raw_parts*` 是 W8T7 实现的
> `TensorBase<ViewRepr,D>` / `TensorBase<ViewMutRepr,D>` 的 inherent methods，**无自
> 由路径符号可被 `pub use`**；`OwnedRawParts` / `into_raw_parts` /
> `from_raw_parts_owned` 是 Owned round-trip API，由 **W22T10** 负责实现
> （原本 W8T7 line 511 推迟声明处，docs_fix 分支中补为独立 task）。
> W13T4 仅 re-export `OwnedRawParts` 与 `TensorBase` 类型（`23-ffi.md` §5.8）。
>
> **W22T10 跨 wave 依赖**：W13T4 需要 `pub use crate::tensor::OwnedRawParts;`，
> 而 `OwnedRawParts` 由 W22T10 在 `src/tensor/construct.rs` 中定义。W13 与 W22
> 同处可并行组 D，本身互不隔离；为避免 W13T4 编译时未定义 OwnedRawParts，
> 本 PLAN 定义在组 D 内部额外约束：**W22T10 必须在 W13T4 之前完成**。

```
批次1:
  W13T1 (mod.rs 骨架, 需 W8T7)
批次2:
  W13T2 (types.rs, 需 W13T1)
批次3:
  W13T3 (private.rs generic descriptors, 需 W13T2)
批次4:
  W13T4 (ptr.rs export/export_mut + raw-parts type re-exports, 需 W13T3 + W22T10)
批次5 (并行):
  W13T5 (BLAS helpers, 需 W13T2)
  W13T6 (offset helpers, 需 W13T2)
```

> **§8.2 跨 wave 依赖测试推迟**：W13T2 / W13T4 / W13T5 / W13T6 的单元测试原使用 `Tensor::zeros`
> （W22T2）与 `Tensor::from_shape_vec`（W22T5）构造测试张量。W13 与 W22 同处可并行组 D，该些构造 API
> 在 W13 实施期不可用。为对齐 W19/W29T11 模式（详见本 PLAN §W19 说明），**W13 task 内部测试改用
> W8T7 `from_raw_parts*` 路径手工构造测试张量**，避免依赖 W22；需要「zeros」语义的完整集成测试
> （`test_lda_non_contiguous` 等需 `transpose()`《**W20T3**》或 `slice()`《**W21T6**》）一并推迟到 W29T18
> （`tests/test_ffi.rs`）。**以前补丁中“W17/W18 提供 slicing API”的描述为误记**，W17 为 Matrix Ops (dot)、
> W18 为 Reduction Ops (sum)，均不含 slice/transpose。

---

### W14: SIMD Backend (L5)

> **docs_fix 修订（W14 全面审计后）**：
> - 新增 W14T0（pulp 0.18 API capability spike）作为全 Wave 14 前置；T2/T4/T5/T6/T7 均依赖 W14T0 的能力清单。
> - W14T1 负责 `src/lib.rs` 模块注册（与 W14T7 解耦；W14T7 仅修改 `src/simd/mod.rs` + `Cargo.toml`）。
> - W14T8 依赖修正为 `W14T1, W14T2, W14T11, W14T7`（去除原 SUMMARY 中冗余的 T3/T5；补充 W14T11 覆盖 Complex element-wise）。
> - W14T10 依赖修正为 `W14T2, W14T11, W14T6, W14T7, W14T9`（补齐实际 API 提供者，含 Complex element-wise）。
> - W14T10 目标文件改为 `tests/simd_property.rs` 顶层文件，通过 Tensor 公有 API 测试，无需 Cargo.toml 显式声明。
> - **新增 W14T11**（docs_fix W14T2 audit FIND-1 修复）：Complex\<f32\>/Complex\<f64\> element-wise add/sub/mul/div/neg SIMD。原设计 08-simd §5.6 / §5.8 明示 Complex element-wise 「已实现」阈值=128，但原 W14T2 仅实现 f32/f64。W14T11 独立 task 避免拓扑循环（W14T2→W14T5→W14T3→W14T2），在批次6 W14T5 后完成，与 W14T6 可并行。
>
> **W14 docs_fix 补充决策 — Complex sum/dot 阈值**：
> design/08-simd.md §5.8 阈值表未列出 Complex 条目。本文件作为 build/ 实施层正式决策承载点，确定：
> - **Complex sum 阈值 = 1024**（派生自 §5.8 "f32/f64 sum=1024"，按 §5.2 "Complex SIMD 策略：AoS 输入 + 寄存器内重排按实部/虚部分离"推导；deinterleave 开销吸收在累加阶段，不改变 sum 量级）。
> - **Complex dot 阈值 = 512**（派生自 §5.8 "f32/f64 dot=512"，同 §5.2 推导；xdotc 共轭乘法仍归 dot 开销量级）。
> 本决策不修改 design/。W14T5/W14T6 实施中若 W14T0 spike 表明阈值不合理，经本节修订后可调整，设计文档不受影响。

```
批次1:
  W14T0 (pulp 0.18 API capability spike)
批次2:
  W14T1 (mod.rs + lib.rs skeleton, 需 W14T0)
批次3:
  W14T2 (element-wise SIMD incl. Neg unary for f32/f64 仅, 需 W14T0+W14T1)
批次4:
  W14T3 (float sum SIMD, 需 W14T2)
批次5 (并行):
  W14T4 (integer i32 widening + i64 scalar fallback, 需 W14T0+W14T3)
  W14T5 (complex sum threshold=1024, 需 W14T0+W14T3)
批次6 (并行，均依赖 W14T5 提供的 Complex AoS deinterleave 路径):
  W14T6 (float+complex dot threshold=512, 需 W14T0+W14T3+W14T5)
  W14T11 (Complex element-wise add/sub/mul/div/neg, threshold=128, 需 W14T0+W14T2+W14T5)
批次7:
  W14T7 (feature gate+exports+dispatch integration, 需 W14T0+W14T1+W14T2)
批次8:
  W14T8 (element-wise consistency tests, 需 W14T1+W14T2+W14T11+W14T7)
批次9:
  W14T9 (reduction/dot semantic+tolerance tests, 需 W14T3+W14T4+W14T5+W14T6+W14T8)
批次10:
  W14T10 (randomized property tests via public API, 需 W14T2+W14T11+W14T6+W14T7+W14T9)
```

---

### W15: Parallel Backend (L5)

```
批次1:
  W15T1 (module skeleton + ParIter + compute_safe_chunks, 需 W10T4)
        — 建立 src/parallel/{mod.rs, iter.rs}, 在 lib.rs 声明 parallel
        — 实现 compute_safe_chunks (W15T2-T5 共享的基础工具)
        — 实现 ParIter + ParIter::with_strategy + TensorBase::par_iter
批次2 (并行):
  W15T2 (par_map, 需 W15T1)
  W15T4 (par_reduce_impl + par_sum, 需 W15T1)
批次3 (并行):
  W15T3 (par_zip_map, 需 W15T2)
  W15T5 (par_dot, 需 W15T4)
批次4 (并行):
  W15T6 (ParallelPool 增量加到 mod.rs, 需 W15T2+W15T4+W15T5)
  W15T7 (par_map_checked + error/panic propagation, 需 W15T2+W15T4+W15T5)
批次5:
  W15T8 (feature gate + config matrix tests, 需 W15T1–W15T7)
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
  W16T11 (SIMD/parallel dispatch integration, 需 W16T3+W16T4+W16T5+W16T6+W16T7+W16T8+W16T9+W16T10+W14+W15)
```

---

### W17: Matrix Operations (L5)

```
批次1:
  W17T1 (module skeleton + Ok(A::zero()) stub)
批次2:
  W17T2 (input validation: rank/length checks, 需 W17T1)
批次3:
  W17T3 (scalar inner product via DotAccumulate trait + TensorBase::dot method, 需 W17T2)
批次4:
  W17T4 (dispatch wiring with alignment_ok helper, 需 W17T3)
批次5 (并行):
  W17T5 (SIMD path via simd::try_dot_*, 需 W17T4+W14)
  W17T6 (parallel path via parallel::par_dot, 需 W17T4+W15)
批次6:
  W17T7 (integration tests, 需 W17T3–W17T6)
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
  W19T1 (mod.rs + lib.rs wiring, forward re-exports; 不创建 src/set/unique.rs — 推迟到 W19T2)
批次2:
  W19T2 (UniqueElement trait + real scalar impls, 需 W19T1)
批次3:
  W19T3 (unique_impl core, 需 W19T2)
批次4 (并行):
  W19T4 (float NaN/±0.0 tests, test-only, 需 W19T3)
  W19T5 (complex equality impl + tests, 需 W19T3)
批次5:
  W19T6 (TensorBase entry + prelude re-export + remaining in-module unit tests, 需 W19T3–W19T5)
```

> W19T4 与 W19T5 并行成立的前提是 W19T4 改为 test-only（不修改 `unique_impl`），避免同文件写冲突。
>
> **§6.5 哈希路径延后**：14-set §6.5 规定"当输入规模导致线性扫描的 O(N²) 成本不可接受时，必须切换到哈希路径"。W19 范围内仅实现线性扫描（见 W19T3 关键设计决策的 scope note），哈希路径作为后续性能任务延后执行；相应地，§8.2 的 stress 级 `test_unique_large_tensor_high_dup`（10^7 元素）推迟到 W29T11，且需在哈希路径就绪后方可稳定通过。
>
> **§8.2 跨 wave 依赖测试推迟**：`test_unique_non_contiguous_view` 与 `test_unique_transposed_view` 依赖 W20 (`transpose()`) 与 W21 (`slice(SliceInfo)`) 公开 API，超出 SUMMARY.md 第 561 行 `W8 + W12 ─→ W19` 的依赖链，且 W19 与 W20/W21 同处可并行组 D；两项测试推迟到 W29T11。

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
  W22T1 (mod.rs skeleton + 4 个子模块占位文件 + lib.rs pub mod construct;)
批次2 (全并行):
  W22T2 (zeros, 需 W22T1)
  W22T3 (ones, 需 W22T1)
  W22T5 (from_shape_vec+from_vec, 需 W22T1)
  W22T8 (from_scalar, 需 W22T1)
批次3 (全并行):
  W22T4 (eye + EyeElement sealed trait, 需 W22T1, W22T2)
  W22T6 (from_shape_slice, 需 W22T5)
批次4:
  W22T7 (from_array, 需 W22T6)
批次5:
  W22T9 (tests, 需 W22T2–W22T8)
批次6 (独立，与 W22T9 可并行，但须在 W13T4 之前完成):
  W22T10 (OwnedRawParts + into_raw_parts + from_raw_parts_owned, 需 W8T7+W8T8；测试额外依赖 W22T5 提供的 from_shape_vec/from_vec)
```

---

### W23: Operator Overloading (L6)

```
批次1:
  W23T1 (mod.rs + placeholder arithmetic.rs)
批次2:
  W23T2 (arithmetic.rs imports + Scalar<A>, 需 W23T1)
批次3:
  W23T3 (Add owned×owned, 需 W23T2)
批次4 (并行):
  W23T4 (Add ref/mixed owned 3 combos, 需 W23T3)
  W23T5 (Add scalar owned 16 impls, 需 W23T3)
批次5 (全并行 — 复用 Add 模式):
  W23T6 (Sub owned full matrix, 需 W23T4+W23T5)
  W23T7 (Mul owned full matrix, 需 W23T4+W23T5)
  W23T8 (Div owned full matrix, 需 W23T4+W23T5)
批次6:
  W23T9 (TensorView tensor×tensor 12 impls for Add/Sub/Mul/Div, 需 W23T6+W23T7+W23T8)
批次7:
  W23T10 (TensorView scalar 64 impls for Add/Sub/Mul/Div, 需 W23T9)
批次8:
  W23T11 (integration tests, 需 W23T1–W23T8) — 仅覆盖 owned Tensor 路径；
    TensorView 的覆盖由 W23T9/W23T10 自身的 unit tests 提供，集成测试不再
    重复 view 路径，故依赖不含 W23T9/W23T10，可与其并行执行
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
  W25T2 (ConvertTo trait def + CastTo doc note, 需 W25T1)
批次3 (W25T3 ∥ W25T4，W25T5 需 W25T4 的内层 impl):
  W25T3 (Tier-1 lossless 14 cells, 需 W25T2)
  W25T4 (Tier-2 lossy 14 cells, 需 W25T2)
批次3.5 (需 W25T4):
  W25T5 (Tier-3 dynamic 8 cells, 需 W25T2+W25T4 — complex→real 委托到 W25T4 内层 CastTo impl)
批次4 (并行，均需 W12T7 提供 `TensorBase::iter()`):
  W25T6 (cast method, 需 W25T3+W25T4+W25T5+W12T7)
  W25T7 (to_owned/into_owned, 需 W25T1+W7T5+W12T7 — StorageIntoOwned trait dispatch + iter())
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
   W29T25 (compile-fail fixtures, 9 files, 需 W29T24)
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
