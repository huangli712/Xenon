# 广播模块设计

> 文档编号: 15
> 模块目录: src/broadcast/
> 任务阶段: Phase 3
> 前置文档: 02-dimension.md, 06-layout.md, 07-tensor.md, 26-error.md
> 需求参考: 需求说明书 §6、§7、§11、§16、§20、§27、§28
> 范围声明: 范围内

---

## 1. 模块定位/概述

### 1.1 职责边界

| 职责           | 包含                                                                    |
| -------------- | ----------------------------------------------------------------------- |
| 广播兼容性判定 | `can_broadcast()`、`broadcast_shape()`，按 Numpy 规则从尾轴开始逐轴比对 |
| 广播步长计算   | `broadcast_strides()` 生成目标视图步长；广播轴写入 `0`                  |
| 广播视图创建   | `broadcast_to()`、`broadcast_with()` 返回零拷贝共享底层数据的只读视图   |
| 类型层维度推导 | 通过 `D1: BroadcastDim<D2>` 这一 public sealed trait 在编译期确定输出维度类型 |
| 广播语义收敛   | 广播结果统一视为“共享底层数据且绝不暴露写权限”的只读广播语义            |

| 职责           | 不包含                                      |
| -------------- | ------------------------------------------- |
| 广播兼容性判定 | 自动触发广播、隐式改写其他模块的 shape 契约 |
| 广播步长计算   | 负步长、复制式 reshape、额外布局模式        |
| 广播视图创建   | 可写广播视图、共享可写广播结果              |
| 类型层维度推导 | 在类型层替代运行时 shape 兼容性检查         |
| 广播语义收敛   | 多输入同步迭代调度、多操作数广播编排        |

### 1.2 设计原则

| 原则             | 体现                                                                                     |
| ---------------- | ---------------------------------------------------------------------------------------- |
| Numpy 一致性     | 从尾轴开始比对；轴长度相同或一方为 `1` 时兼容，否则返回 `XenonError::BroadcastError`。   |
| 零拷贝优先       | 广播只改写 shape/stride/flags，不复制底层数据。                                          |
| 共享只读         | 广播结果始终降级为只读视图；任何可变访问都必须在类型层或运行时显式拒绝。                 |
| 步长显式化       | 广播轴使用 `usize` 零步长表达，与 `06-layout.md` 中的 `BroadcastView` 布局状态保持一致。 |
| 类型与运行时分层 | `BroadcastDim` 作为 public sealed trait 负责输出维度类型推导，`broadcast_shape()` 负责实际兼容性裁决。 |

---

## 2. 需求映射与范围约束

| 类型     | 内容                                                                                                                                    |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| 需求映射 | 需求说明书 §6、§7、§11、§16、§20、§27、§28。                                                                                            |
| 范围内   | `broadcast_shape()`、`can_broadcast()`、`broadcast_strides()`、`broadcast_to()`、`broadcast_with()`、零步长广播视图、共享只读广播结果。 |
| 范围外   | 可写广播视图、隐式广播、多操作数统一调度、负步长广播、复制式 expand/reshape。                                                           |
| 非目标   | 不在本文引入新的布局状态、存储模式、自动类型提升或任何额外第三方依赖。                                                                  |

---

## 3. 文件位置

```text
src/broadcast/
├── mod.rs             # module entry, re-export public functions and trait-bound-related entry points
├── shape.rs           # can_broadcast(), broadcast_shape(), broadcast_strides()
└── view.rs            # broadcast_to() and pub(crate) broadcast_with() internals
```

文件划分理由：广播模块天然分为“兼容性/步长规则”和“视图构造”两部分；前者只处理 shape 与 stride 元数据，后者负责把结果降级为只读广播视图。采用 `src/broadcast/` 目录结构能使规则函数与视图入口分离，同时保持当前版本只覆盖显式广播能力。

---

## 4. 依赖关系

### 4.1 依赖图（ASCII）

```
src/broadcast/
|
├── mod.rs
│   └── module entry, re-export public functions and trait-bound-related entry points
|
├── shape.rs
│   ├── crate::dimension  # Dimension, Ix0~Ix6, IxDyn
│   └── crate::error      # XenonError::BroadcastError, XenonError::InvalidArgument
|
└── view.rs
    ├── crate::tensor     # TensorBase<S, D>, TensorView<'a, A, D>, .shape(), .strides(), .offset()
    │                      # broadcast_to() 是在 broadcast 模块内为 TensorBase 添加的 inherent impl
    ├── crate::dimension  # Dimension, BroadcastDim<Other>
    ├── crate::layout     # Strides<D>, LayoutFlags, LayoutState::BroadcastView
    └── crate::error      # XenonError::BroadcastError, XenonError::InvalidArgument
```

### 4.2 类型级依赖

| 来源模块    | 使用的类型/trait                                                                                                                                    |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tensor`    | `TensorBase<S, D>`, `TensorView<'a, A, D>`, `.shape()`, `.strides()`, `.offset()`, 视图构造入口，以及从任意受支持存储模式降级到只读广播视图的入口。 |
| `dimension` | `Dimension`, `Ix0`~`Ix6`, `IxDyn`, `BroadcastDim<Other>`（public sealed trait，对外可命名；对称性见 02-dimension §5.10），`IntoDimension`（用于 `broadcast_to<E>` 接受目标 shape 的多种语法形式：`IxN` / 元组 / 数组 / `Vec<usize>` / `&[usize]`）。 |
| `layout`    | `Strides<D>`, `LayoutFlags`, `LayoutState::BroadcastView`（广播结果的目标布局状态），以及通过 `compute_layout_flags()` 间接关联的 `LayoutState::FContiguous` / `LayoutState::NonContiguous`。 |
| `error`     | `XenonError::BroadcastError`, `XenonError::InvalidArgument`（`InvalidArgumentKind::OperationSpecific` 用于 `broadcast_strides` 的 rank/长度前提失败），以及 `Cow<'static, str>` 用于 `operation` 字段（参见 26-error §5.1）。|

### 4.3 依赖合法性

| 项目           | 说明                                                                                          |
| -------------- | --------------------------------------------------------------------------------------------- |
| 新增第三方依赖 | 无                                                                                            |
| 合法性结论     | 合法；当前设计仅使用 Xenon 既有模块与标准库，符合本文前述需求映射以及最小依赖、单 crate 约束。|
| 替代方案       | 不适用；广播规则与只读视图构造可直接在现有模块边界内完成。                                    |

### 4.4 依赖方向声明

依赖方向：单向向上。`broadcast` 消费 `tensor`、`dimension`、`layout` 与 `error` 的既有能力，不反向定义这些核心模块的语义。

---

## 5. 公共 API 设计

### 5.1 公共接口草案与关键签名

```rust,ignore
pub fn broadcast_shape(shape_a: &[usize], shape_b: &[usize]) -> Result<IxDyn, XenonError>;

pub fn can_broadcast(shape_a: &[usize], shape_b: &[usize]) -> bool;

pub fn broadcast_strides(
    orig_shape: &[usize],
    orig_strides: &[usize],
    target_shape: &[usize],
) -> Result<Vec<usize>, XenonError>;

impl<S, A, D> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    pub fn broadcast_to<E>(&self, shape: E) -> Result<TensorView<'_, A, E::Dim>, XenonError>
    where
        E: IntoDimension;
}

pub(crate) fn broadcast_with<'a, A, S1, D, S2, E>(
    a: &'a TensorBase<S1, D>,
    b: &'a TensorBase<S2, E>,
) -> Result<
    (
        TensorView<'a, A, <D as BroadcastDim<E>>::Output>,
        TensorView<'a, A, <D as BroadcastDim<E>>::Output>,
    ),
    XenonError,
>
where
    S1: Storage<Elem = A>,
    S2: Storage<Elem = A>,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>;
```

**关于 `broadcast_with` 双向 `BroadcastDim` bound 的可满足性**：

`broadcast_with` 的 `where` 子句同时要求 `D: BroadcastDim<E>` 与 `E: BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>`。这条双向 bound 看似过强，但能完整覆盖封闭维度集合 `{Ix0..Ix6, IxDyn}` 的所有 `(D, E)` 组合（57 项），由 02-dimension §5.10 实现矩阵保证。57 项的计数口径是“同 rank 自广播 + 跨静态 rank 双向合并行 + 静态/IxDyn 双向合并行 + IxDyn 自广播”的文档矩阵行数，而不是底层 trait impl 条数：

- 同 rank自广播 7 项（`IxN BroadcastDim IxN → IxN`，自然对称）
- 跨静态 rank 双向合并 42 项（每个无序静态 rank 对在文档矩阵中列出两个方向：`IxM BroadcastDim IxN` 与 `IxN BroadcastDim IxM`，`Output` 都为较高 rank 的 `IxK`，`K = max(M, N)`）
- 静态 + IxDyn 双向合并 7 项（每项覆盖 `IxN BroadcastDim IxDyn` 与 `IxDyn BroadcastDim IxN`，`Output` 都为 `IxDyn`）
- `IxDyn BroadcastDim IxDyn → IxDyn` 1 项

公式：`7 + 42 + 7 + 1 = 57`。其中“静态 + IxDyn 双向合并 7 项”每项包含两个方向的对称实现；若按单个 trait impl 逐条计数，会得到不同数字，但 `broadcast_with` 只依赖 02-dimension §5.10 已声明的 57 项矩阵及其对称性测试。

02-dimension §5.10 通过显式 trait 实现对称性保证：对所有 `(D, E)`，`<D as BroadcastDim<E>>::Output == <E as BroadcastDim<D>>::Output`，并在 §5.10 末尾增加 compile-time 类型等价测试覆盖。因此 `broadcast_with` 的 bound 在所有合法组合上可满足，不会因为反向 trait 缺失而拒绝调用。

### 5.2 API 语义约束

| API                   | 语义                                                                                                                                                                     |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `can_broadcast()`     | 仅回答兼容性，不分配、不生成中间结果。                                                                                                                                   |
| `broadcast_shape()`   | 运行时计算公共 shape；不兼容时返回 `XenonError::BroadcastError`。                                                                                                        |
| `broadcast_strides()` | 对齐原 shape 与目标 shape，广播轴写入 `0` 步长；当 `orig_shape.len() != orig_strides.len()` 时返回 `XenonError::InvalidArgument { operation: Cow::Borrowed("broadcast_strides"), kind: InvalidArgumentKind::OperationSpecific { argument: Cow::Borrowed("orig_strides"), constraint: Cow::Borrowed("len must match orig_shape.len()") } }`（字段对齐 26-error §5.1）。 |
| `broadcast_to()`      | 显式广播入口；成功时返回共享底层数据的只读 `TensorView`。结果必须满足 `需求说明书 §6` 对“共享只读引用”的约束：可在多个张量实例之间共享同一底层数据，但不提供可写访问权。 |
| `broadcast_with()`    | 面向两个张量输入的 `pub(crate)` 助手：先计算共同 shape，再分别构造两个只读广播视图。它不承担通用 shape 工具职责；仅需 shape 判定时应使用 `can_broadcast()` / `broadcast_shape()`。 |

- **inherent 方法 vs 自由函数：** `broadcast_to()` 以 `&self` 为接收者，语义上属于"对该张量执行广播"，因此作为 `TensorBase` 的 inherent 方法定义在 `view.rs` 中。`broadcast_with()` 是双输入函数，没有自然接收者，因此定义为 `pub(crate)` 自由函数。`can_broadcast()`、`broadcast_shape()`、`broadcast_strides()` 仅操作 shape/stride 元数据，不涉及张量实例，因此也作为自由函数定义在 `shape.rs` 中。
- **同形状快捷路径**：当两个输入形状完全相同时，`broadcast_with()` 可直接返回两个原始视图而不执行步长重写，因为目标 shape 与输入 shape 一致。
- **目标秩语义**：`broadcast_to()` 的目标 shape 秩决定了输出视图的维度类型；标量广播到高维时，缺失前导轴按 `1` 补齐。
- **`IntoDimension` 说明：** `IntoDimension` 只决定目标 rank/type；逐轴长度兼容性完全由 `broadcast_shape()` / `broadcast_strides()` 在运行时检查。
- **类型设计说明：** `broadcast_to()` 是目标 shape 主导的 API，只需目标维度类型 `E: IntoDimension`。`broadcast_with()` 是双输入 shape 合流 API，需要双向 `BroadcastDim` 一致性以保证输出维度类型的静态可推导性。`BroadcastDim` 本身是 public sealed trait，因此在这些公开签名中对外可命名。
- **BroadcastError 字段映射：** 各 API 返回 `XenonError::BroadcastError` 时，结构化字段按以下规则填充（字段类型对齐 26-error §5.1：`operation: Cow<'static, str>`，`lhs_shape` / `rhs_shape` 总是 `Vec<usize>`，`attempted_target_shape: Option<Vec<usize>>`，`axis: Option<usize>`）：

  | API | `operation` | `lhs_shape` | `rhs_shape` | `attempted_target_shape` | `axis` |
  | --- | --- | --- | --- | --- | --- |
  | `broadcast_shape(a, b)` | `Cow::Borrowed("broadcast_shape")` | `a.to_vec()` | `b.to_vec()` | `None` | `Some(失败轴 index)` |
  | `broadcast_to(self, target)` | `Cow::Borrowed("broadcast_to")` | `self.shape().to_vec()` | `vec![]`（无右侧输入；用空 Vec 作占位） | `Some(target.shape().to_vec())` | `Some(失败轴 index)` |
  | `broadcast_with(a, b)` | `Cow::Borrowed("broadcast_with")` | `a.shape().to_vec()` | `b.shape().to_vec()` | `None` | `Some(失败轴 index)` |

  > 字段类型说明：`lhs_shape` 与 `rhs_shape` 不再是 `Option<Vec<usize>>`，而是 `Vec<usize>`。`broadcast_to` 这种"单输入 + 显式目标"的场景没有右侧输入，按约定用 `vec![]` 占位以满足结构体字段非 Option 的要求；调用方据此可识别"右侧输入不存在"。该占位只在 `operation == "broadcast_to"` 且 `attempted_target_shape.is_some()` 的语境下表示无右侧输入；标量 shape `[]` 仍需结合具体 operation / 字段位置解释。`attempted_target_shape` 仅 `broadcast_to` 填 `Some(..)`，其它 API 用 `None`。
- **返回类型与共享只读保证：** 当前版本复用 `TensorView` 作为返回类型，不引入单独的 `BroadcastView` 新类型。广播结果内部承载 `ViewRepr<'a, A>`（与 `05-storage.md` §5.11.1 "广播 / 转置 / 切片产生的只读视图统一使用 ViewRepr" 规则一致），`storage_kind()` 返回 `StorageKind::View`，`access_semantics()` 返回 `AccessSemantics::SharedReadOnly`。由于广播引入零步长布局，多个逻辑位置映射到同一物理元素，因此只读共享语义由以下机制共同保证：1) `LayoutFlags::HAS_ZERO_STRIDE` / `LayoutState::BroadcastView` 标识广播布局；2) 广播结果类型层缺失 `StorageMut` 能力且不提供 `into_mut()` 等 API；3) 广播结果的生命周期绑定源张量。

### 5.3 Good / Bad 对比

```rust,ignore
// Good - explicit broadcast with recoverable error handling
let a = lhs.view();
let b = rhs.view();
let (a2, b2) = broadcast_with(&a, &b)?;

// Bad - hide broadcast failure behind panic
let a = lhs.view();
let b = rhs.view();
let (a2, b2) = broadcast_with(&a, &b)
    .expect("broadcast must succeed for all shapes");
```

```rust,ignore
// Good - zero-copy broadcast result stays read-only
let view = tensor.view().broadcast_to([4, 3])?;
assert_eq!(view.strides()[0], 0);

// Bad - design a mutable broadcast API
// let mut_view = tensor.view_mut().broadcast_to_mut([4, 3])?;
// Forbidden: broadcast results must not expose mutable access.
```

---

## 6. 内部实现设计

### 6.1 广播不变式

- 广播必须是零拷贝；不得复制底层数据。
- 广播结果只能返回只读 `TensorView`，并按共享只读引用处理；这里的“共享只读引用”含义与 `需求说明书 §6` 一致：结果可在多个张量实例之间共享同一底层数据，但不提供可写访问权。
- 广播轴的 stride 必须写成 `0`，且 stride 类型保持为 `usize`。
- 若结果存在广播零步长轴**且** `product(shape) > 0`（即结果非空），布局状态必须标记为 `LayoutState::BroadcastView`。空数组退化情形（`product(shape) == 0`，例如 `1 → 0` 空轴广播）即使含 `stride == 0` 也**不**触发 `BroadcastView`——与 `06-layout.md §5.11` 的 `HAS_ZERO_STRIDE` 公式严格一致。详细分类口径见 §6.3。
- 广播不改变底层 storage、offset 与逻辑元素顺序语义。
- 所有 shape 兼容性裁决必须在创建结果视图前完成。

### 6.2 广播形状算法

```text
broadcast_shape(shape_a, shape_b):
    1. Align dimensions from right to left.
    2. Treat missing leading dimensions as 1.
    3. If two aligned dimensions differ and neither is 1, return BroadcastError.
    4. Otherwise result = max(a, b): if one is 1, take the other; if both equal (including both 1), take that value.
    5. Return the computed IxDyn shape.
```

Numpy 兼容性规则由 `broadcast_shape()` 和 `can_broadcast()` 共用：从尾轴开始逐轴比较，当轴长度相同或其中一方为 `1` 时兼容，否则不兼容。缺失的前导轴按 `1` 处理，因此标量与低维输入可广播到更高维目标。

### 6.3 广播步长算法

```text
broadcast_strides(orig_shape, orig_strides, target_shape):
    1. Validate rank compatibility.
    2. Right-align the original shape against the target shape.
    3. For each axis:
        - if original dimension == target dimension, keep the original stride;
        - if original dimension == 1 and target dimension != 1, write stride 0;
          (this includes empty-axis broadcasting `1 -> 0`; the result has no
          logical elements along that axis, and stride 0 preserves the
          broadcast-aliasing classification without accessing extra storage)
        - otherwise return BroadcastError.
    4. Return the computed stride vector.
```

对广播轴写入零步长意味着该轴被逻辑扩展（或在空轴广播中收缩为 0 长度目标），但所有有效索引都回落到同一底层元素；这与 `06-layout.md` §5.11 的零步长语义保持一致。`orig_dim == 1 && target_dim == 0` 是兼容的空轴广播，输出 stride 写为 `0`，且因为目标轴长度为 0 不会产生实际元素访问。布局分类口径以 **06-layout.md §5.11**（`HAS_ZERO_STRIDE` 权威定义）为准，由 `compute_layout_flags()`（06-layout.md §5.12）执行。非空广播视图进入 `BroadcastView`；空数组退化情形不触发广播分类——详见 06-layout.md §5.11 边界情形覆盖表。

- **再次广播规则：** 对已广播视图再次广播时，已有零步长轴保持为 `0`，新增广播轴也写入 `0` 步长；结果 `shape` 取"当前视图 shape"与"新目标 shape"的广播结果。 `broadcast_strides()` 的 `orig_shape` 参数始终传入当前视图的逻辑 shape（即 `.shape()` 返回值），而非某个"广播前的原始 shape"。

  **关于"已有零步长轴的识别"**：算法通过 `orig_strides[i] == 0` 直接识别零步长轴，而**不是**依赖 `orig_shape[i] == 1`。理由是：广播视图的当前逻辑 shape 中，原本被广播的轴长度可能已大于 1（例如 `[1] -broadcast→ [4]` 后再次广播，新视图 `orig_shape[axis] == 4` 但 `orig_strides[axis] == 0`）。算法的第 3 步分支"if original dimension == target dimension, keep the original stride"对此场景天然正确：保留原始 stride 即保留 0 步长，无需基于 shape 值判断。读者不应把 `orig_shape[i] == 1` 误认为零步长轴的识别条件——它只是"原始尚未广播的轴可被广播到更大目标"的条件，与"已存在的零步长轴"无关。
- **布局标志重算规则：** 布局标志必须通过 `compute_layout_flags()`（见 `06-layout.md`）重算，而非直接复制源标志。广播不改变逻辑首元素指针，因此重算后 `ALIGNED` 结果与源视图一致；`F_CONTIGUOUS` 仅在不存在零步长且结果 stride 仍满足 F-order 规则时保留。广播结果的 `HAS_ZERO_STRIDE` flag 与 `LayoutState` 分类按 **06-layout.md §5.11**（唯一权威）判定。非空广播视图归入 `LayoutState::BroadcastView`；空数组退化（`product(shape) == 0`）不触发，详见 06-layout.md §5.11 空张量退化规则。

### 6.4 共享只读视图构造

`broadcast_to()` 和 `broadcast_with()` 只负责在元数据层重建视图：保留原始 storage 与 offset，改写 shape、strides 与 flags。返回类型虽然仍是 `TensorView`，但其公开语义必须统一收敛到共享只读引用：广播结果内部承载 `ViewRepr<'a, A>`，`storage_kind()` 返回 `StorageKind::View`；只读共享语义由广播布局标志与访问控制 API 共同保证，任何试图从广播结果取得可变访问权的 API，都必须在类型层缺失或运行时返回错误。

**安全性论证（unchecked 视图构造）：** 若内部使用 `TensorView::new_unchecked()` 或等价未检查构造器，调用点必须先证明：1）目标 `shape` 与源 `shape` 广播兼容；2）新 `shape` / `stride` / `offset` 组合不会访问到底层 storage 可见边界之外；3）任何零步长元素都不会通过结果视图暴露为可变访问。

**布局状态判定（由视图构造方负责）：** 广播视图的 `LayoutFlags` 必须通过 `compute_layout_flags()`（见 `06-layout.md`）重算。广播结果的布局状态分类以 **06-layout.md §5.11**（`HAS_ZERO_STRIDE` 权威定义）为准。非空广播视图（`product(shape) > 0` 且存在广播零步长轴）归入 `LayoutState::BroadcastView`；空数组退化（`product(shape) == 0`，即使存在零步长写入）不触发广播分类，详见 06-layout.md §5.11 边界情形覆盖表。

### 6.5 `BroadcastDim` 的职责边界

`BroadcastDim<Other>` 是 public sealed trait，因此在公开 API 中可被外部稳定命名；它仅用于编译期计算输出维度类型：

- `IxM BroadcastDim IxN`（M > N）→ `IxM`（跨静态 rank，取较高 ndim；同 `02-dimension.md` §5.10）
- `IxN BroadcastDim IxN` → `IxN`（同 rank）
- `IxN BroadcastDim IxDyn` → `IxDyn`
- `IxDyn BroadcastDim IxN` → `IxDyn`
- `IxDyn BroadcastDim IxDyn` → `IxDyn`

它不负责判定具体 shape 值是否兼容；例如两个 `Ix2` 仍可能因 `[2, 3]` 与 `[4, 3]` 不兼容而在运行时失败。运行时裁决始终由 `broadcast_shape()` / `broadcast_with()` 完成。

### 6.6 与存储系统的对接

- **查询：** 广播结果内部使用 `ViewRepr<'a, A>`，因此 `storage_kind()` 返回 `StorageKind::View`，`access_semantics()` 返回 `AccessSemantics::SharedReadOnly`；是否为广播结果由 layout flags 中的 `LayoutFlags::HAS_ZERO_STRIDE` / `LayoutState::BroadcastView` 指示。
- **转换：** 广播结果可通过显式分配转成 `Owned` 连续张量（如 `to_owned()` / `to_contiguous()` 一类路径）；由于广播视图存在零步长别名，当前版本不允许把它转换为 `ViewMut`，也不提供 `into_mut()`。
- **线程：** 广播 `ViewRepr` 遵循标准借用规则；当 `A: Sync` 时可满足只读跨线程共享前提，`Send`/`Sync` 语义与普通只读视图一致，不因广播额外放宽。

---

## 7. 实现任务拆分

### Wave 1：规则函数与类型边界

- [ ] **T1**: 建立 `src/broadcast/mod.rs` 骨架与公共导出
  - 文件: `src/broadcast/mod.rs`, `src/broadcast/shape.rs`, `src/broadcast/view.rs`
  - 内容: 模块声明、规则函数声明、视图相关入口占位
  - 测试: 编译通过
  - 前置: `dimension`、`layout`、`tensor`、`error` 模块完成
  - 预计: 10 min

- [ ] **T2**: 实现 `can_broadcast()`
  - 文件: `src/broadcast/shape.rs`
  - 内容: 尾轴对齐兼容性判定
  - 测试: `test_can_broadcast_compatible`
  - 前置: T1
  - 预计: 5 min

- [ ] **T3**: 实现 `broadcast_shape()`
  - 文件: `src/broadcast/shape.rs`
  - 内容: 公共 shape 推导与结构化广播错误填充
  - 测试: `test_broadcast_shape_error`
  - 前置: T2
  - 预计: 5 min

- [ ] **T4**: 实现 `broadcast_strides()`
  - 文件: `src/broadcast/shape.rs`
  - 内容: 零步长写入与参数前提校验
  - 测试: `test_broadcast_strides_zero_stride`
  - 前置: T3
  - 预计: 10 min

### Wave 2: 视图构造基础

- [ ] **T5**: 实现 `broadcast_to()` 基本路径
  - 文件: `src/broadcast/view.rs`
  - 内容: 目标 shape 校验与只读视图构造
  - 测试: `test_broadcast_to_basic`
  - 前置: T4
  - 预计: 10 min

### Wave 3: 视图构造补全

- [ ] **T6**: 实现 `broadcast_to()` 错误路径与布局更新
  - 文件: `src/broadcast/view.rs`
  - 内容: 非法目标 shape 错误返回与 `BroadcastView` 布局状态更新
  - 测试: `test_broadcast_to_error`, `test_broadcast_read_only`
  - 前置: T5
  - 预计: 10 min

- [ ] **T7**: 实现 `broadcast_with()`
  - 文件: `src/broadcast/view.rs`
  - 内容: 公共 shape 推导、双输入广播、`BroadcastDim` 输出类型对齐
  - 测试: `test_broadcast_with_same_shape`, `test_broadcast_scalar_and_tensor`, `test_broadcast_with_incompatible_shapes`
  - 前置: T4, T5
  - 预计: 15 min

### Wave 4: 综合验证

- [ ] **T8**: 编写单元与集成测试
  - 文件: `tests/test_broadcast.rs`, `tests/property_tests.rs`, `tests/property/shape_props.rs`（路径需与项目实际测试目录结构一致，实现时以现有测试布局为准）
  - 内容: 兼容性规则、零步长语义、共享只读边界、属性测试
  - 测试: 覆盖范围内所有公开 API
  - 前置: T6, T7
  - 预计: 20 min

---

## 8. 测试计划

### 8.1 测试分类表

| 测试分类 | 位置                                                 | 说明                                                           |
| -------- | ---------------------------------------------------- | -------------------------------------------------------------- |
| 单元测试 | `src/broadcast/*` 的 `#[cfg(test)]`                  | 验证 shape 兼容性、零步长生成和错误结构。                      |
| 集成测试 | `tests/test_broadcast.rs`                            | 验证广播与 `tensor`、`layout`、`overload`、`iter` 的协同路径。 |
| 边界测试 | 同模块测试中标注                                     | 覆盖标量、空数组、再次广播、高维广播和 `10^7` 元素大张量广播。 |
| 属性测试 | `tests/property_tests.rs`, `tests/property/shape_props.rs` | 验证广播 shape/stride 不变量和零拷贝语义。               |

### 8.2 单元测试清单

| 测试函数                                 | 测试内容                               | 优先级 |
| ---------------------------------------- | -------------------------------------- | ------ |
| `test_can_broadcast_compatible`          | 兼容 shape 判定正确                    | 高     |
| `test_can_broadcast_incompatible`        | 不兼容 shape 判定正确                  | 高     |
| `test_broadcast_shape_basic`             | 公共 shape 推导正确                    | 高     |
| `test_broadcast_shape_error`             | 返回 `BroadcastError` 且字段完整       | 高     |
| `test_broadcast_strides_zero_stride`     | 广播轴步长为 `0`                       | 高     |
| `test_broadcast_strides_non_negative`    | 非广播轴保持 `usize` 步长              | 中     |
| `test_broadcast_to_basic`                | 只读广播视图创建正确                   | 高     |
| `test_broadcast_to_error`                | 非法目标 shape 返回结构化错误          | 高     |
| `test_broadcast_with_same_shape`         | 双输入公共 shape 正确                  | 中     |
| `test_broadcast_read_only`               | 广播视图不提供可写入口                 | 高     |
| `test_broadcast_high_rank_ixdyn`         | `IxDyn` 高 rank 广播形状与步长正确     | 中     |
| `test_broadcast_rebroadcast_zero_stride` | 再次广播时零步长继承与新增规则正确     | 中     |
| `test_broadcast_layout_flags_recomputed` | 广播后 flags 按零步长/F-order 规则重算 | 中     |
| `test_broadcast_large_tensor_zero_copy`  | `10^7` 元素级广播保持零拷贝与零步长语义 | 高    |

### 8.3 边界测试场景

| 场景                                | 预期行为                                           |
| ----------------------------------- | -------------------------------------------------- |
| `[0, 3]` 与 `[1, 3]`                | 允许空轴广播，结果 shape 为 `[0, 3]`；`1 -> 0` 轴输出 stride 为 `0`，不复制数据且不访问额外元素。 |
| 标量广播到高维                      | 缺失前导轴按 `1` 处理，广播结果为共享只读视图。    |
| 输入已是广播视图再次广播            | 允许继续广播，但结果仍保持只读且零步长语义一致。   |
| 高维输入 `[2,1,4]` → `[3,2,5,4]`    | 右对齐补 `1` 后逐轴校验，写入对应零步长。          |
| 靠近静态上限或 `IxDyn` 高 rank 广播 | 逐轴规则保持一致，输出维度与零步长位置正确。       |
| 标量或 `[1,3162,3162]` 广播到 `10^7` 量级目标 shape | 输出逻辑元素数约为 `10^7`，保持零拷贝、零步长与只读语义。 |

### 8.4 属性测试不变量

| 不变量                                                 | 测试方法                        |
| ------------------------------------------------------ | ------------------------------- |
| `can_broadcast(a, b) == broadcast_shape(a, b).is_ok()` | 随机 shape 对                   |
| 广播后逻辑元素数与目标 shape 一致                      | 随机目标 shape                  |
| 广播轴 stride 恒为 `0`                                 | 随机含 `1` 轴的 shape           |
| 广播结果共享源数据                                     | 比较 data pointer / offset 不变 |

### 8.5 集成测试

| 测试文件                  | 测试内容                                                                                                        |
| ------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `tests/test_broadcast.rs` | `broadcast_to()` / `broadcast_with()` / `broadcast_shape()` 与 `tensor`、`layout`、`overload`、`iter` 的协同路径 |

### 8.6 Feature gate / 配置测试

| 配置                          | 验证点                                                          |
| ----------------------------- | --------------------------------------------------------------- |
| 默认配置                      | 显式广播 API、零步长和共享只读语义保持一致。                    |
| `rayon` / SIMD/pulp 相关 feature（按 Cargo.toml 命名） | 广播模块本身不改变语义；不同执行路径不得改变 shape 与错误类别。 |
| 无额外 feature                | 当前模块不新增独立 feature gate。                               |

### 8.7 类型边界 / 编译期测试

| 场景                                                  | 测试方式                     |
| ----------------------------------------------------- | ---------------------------- |
| `BroadcastDim<Other>::Output` 对静态/动态维组合可编译 | compile-pass 测试            |
| 运行时不兼容 shape 仍可在方法级返回 `BroadcastError`  | 编译通过 + 运行时断言        |
| 广播结果无可变访问 API                                | compile-fail 或 API 缺失断言 |
| `broadcast_to()` 接受 `E: IntoDimension`              | 编译期签名检查               |

---

## 9. 模块交互设计

### 9.1 接口约定

| 方向                        | 对方模块     | 接口/类型                                       | 约定                                                     |
| --------------------------- | ------------ | ----------------------------------------------- | -------------------------------------------------------- |
| `broadcast → tensor`        | `tensor`     | `TensorBase`, `TensorView`                      | 读取 shape/stride/offset，并通过只读视图入口构造结果。   |
| `broadcast → dimension`     | `dimension`  | `Dimension`, `BroadcastDim`                     | 运行时 shape 计算与编译期输出维度类型推导分离。          |
| `broadcast → layout`        | `layout`     | `Strides<D>`, `LayoutState::BroadcastView`      | 非空（`product(shape) > 0`）且至少一轴 stride 为 0 的视图必须映射到 `BroadcastView` 布局状态；空数组退化的零步长不触发该状态（与 `06-layout.md §5.11` 严格一致）。 |
| `broadcast → error`         | `error`      | `XenonError::BroadcastError`, `InvalidArgument` | 广播不兼容与参数前提失败都必须返回结构化错误。           |
| `math ← broadcast`          | `math`       | `broadcast_with()`, `broadcast_shape()`         | 二元运算先广播再计算。`math` 模块内部统一调用 `broadcast_with()`（pub(crate) 唯一入口）完成双输入广播，不允许各模块私自重复定义广播规则。具体调用路径：`math` 通过 `dispatch::select_exec_path` 决定串行/SIMD/并行三路；并行路径下 `par_zip_map` 接收已广播好的 `output_dim`，所有广播裁决发生在调用 `parallel/` 后端**之前**（参见 11-math §5.2、09-parallel §6.3）。 |
| `iter ← broadcast`          | `iter`       | 只读广播视图                                    | 广播结果可被读取遍历，但不得提供可变迭代能力。           |

### 9.2 数据流描述

```text
User calls broadcast_to() or broadcast_with()
    │
    ├── broadcast_shape() checks Numpy compatibility from trailing axes
    ├── broadcast_strides() writes zero strides for expanded axes
    ├── tensor view constructor reuses original storage and offset
    └── result is exposed as a shared read-only broadcast view
```

---

## 10. 错误处理与语义边界

| 主题              | 内容                                                                                                                               |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Recoverable error | 广播不兼容时统一返回 `XenonError::BroadcastError`；`broadcast_to`、`broadcast_shape`、`broadcast_with` 都必须填充结构化字段（按 §5.2 表）。`operation` 用 `Cow::Borrowed(..)`，`lhs_shape` / `rhs_shape` 总是 `Vec<usize>`，`attempted_target_shape` / `axis` 是 `Option`。|
| 参数错误          | 当 `orig_shape.len() != orig_strides.len()` 等公开前提被破坏时，`broadcast_strides()` 返回 `XenonError::InvalidArgument { operation: Cow::Borrowed("broadcast_strides"), kind: InvalidArgumentKind::OperationSpecific { argument, constraint } }`（封闭枚举字段对齐 26-error §5.1）。|
| Panic             | 不允许把 shape 不兼容隐藏为 panic；公开 API 使用 `Result` 表达失败。                                                               |
| 语义边界          | 广播只负责显式元数据扩展，不改变元素值、不重排数据、不授予可写访问。                                                               |
| 路径一致性        | 默认路径、后续可能启用的 SIMD/并行消费路径都必须共享同一广播规则与错误类别；广播模块自身不分裂语义分支。                           |

---

## 11. 设计决策记录

### 决策 1：广播结果统一只读

| 属性     | 值                                                                                                  |
| -------- | --------------------------------------------------------------------------------------------------- |
| 决策     | 广播成功后的结果统一视为共享只读引用，不提供可写广播视图。                                          |
| 理由     | `需求说明书 §16` 明确要求结果共享底层数据且作为共享只读引用对待；零步长布局也无法安全支持可变别名。 |
| 替代方案 | 保留源张量的可写权限 —— 放弃，会破坏别名和独占性约束。                                              |

### 决策 2：显式广播而非隐式广播

| 属性     | 值                                                                 |
| -------- | ------------------------------------------------------------------ |
| 决策     | 通过 `broadcast_to()` / `broadcast_with()` 显式触发广播。          |
| 理由     | 保持 API 语义清晰，符合“避免隐式行为”的项目原则。                  |
| 替代方案 | 在遍历或运算路径中静默广播 —— 放弃，会让错误来源和布局变化不可见。 |

### 决策 3：类型层与运行时双层判定

| 属性     | 值                                                                    |
| -------- | --------------------------------------------------------------------- |
| 决策     | `BroadcastDim` 作为 public sealed trait 只负责输出维度类型推导，实际兼容性由运行时函数检查。|
| 理由     | 维度 rank 可在类型层表达，但具体轴长度仍需运行时输入决定。            |
| 替代方案 | 尝试完全在类型层判定广播成功 —— 放弃，不适用于动态 shape 与值级信息。 |

---

## 12. 性能描述

### 12.1 时间复杂度

| 操作                  | 复杂度                 | 说明                                 |
| --------------------- | ---------------------- | ------------------------------------ |
| `can_broadcast()`     | O(max(ndim_a, ndim_b)) | 仅做尾轴对齐比较                     |
| `broadcast_shape()`   | O(max(ndim_a, ndim_b)) | 线性扫描所有对齐后的轴               |
| `broadcast_strides()` | O(target_ndim)         | 对目标轴逐个生成步长                 |
| `broadcast_to()`      | O(target_ndim)         | shape/stride 校验 + 视图元数据构造   |
| `broadcast_with()`    | O(max(ndim_a, ndim_b)) | 先求公共 shape，再对两个输入分别广播 |

### 12.2 内存行为

| 操作                  | 内存分配             | 数据拷贝 |
| --------------------- | -------------------- | -------- |
| `broadcast_shape()`   | 仅输出 shape 元数据  | 无       |
| `broadcast_strides()` | 仅输出 stride 元数据 | 无       |
| `broadcast_to()`      | 仅构造视图元数据     | 无       |
| `broadcast_with()`    | 仅构造两个视图元数据 | 无       |

### 12.3 缓存行为

| 场景                 | 缓存友好性     | 说明                                               |
| -------------------- | -------------- | -------------------------------------------------- |
| 非广播轴访问         | 取决于原始布局 | 模块本身不改变非广播轴的 stride。                  |
| 广播轴重复读取       | 较好           | 零步长会重复访问同一底层元素，通常命中同一缓存行。 |
| 广播后下游逐元素运算 | 取决于消费方   | 广播模块保证零拷贝，但不承诺改善后续遍历顺序。     |

---

## 13. 平台与工程约束

| 约束       | 说明                                                            |
| ---------- | --------------------------------------------------------------- |
| `std` only | Xenon 当前版本仅支持 `std` 环境，广播设计不提供 `no_std` 分支。 |
| MSRV       | Rust 1.85+                                                      |
| 单 crate   | 广播模块保持在现有 crate 内，不拆分为独立 crate。               |
| SemVer     | 本设计只收敛现有广播语义与文档结构，不扩展超范围公开能力。      |
| 最小依赖   | 不新增任何第三方依赖，仅消费现有模块与标准库。                  |
| 负步长禁用 | 步长保持 `usize`，仅允许非负步长与零步长广播语义。              |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
