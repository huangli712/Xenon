# 广播模块设计

> 文档编号: 15 | 模块: `src/broadcast/` | 阶段: Phase 3
> 前置文档: `02-dimension.md`, `06-layout.md`, `07-tensor.md`, `26-error.md`
> 需求参考: `需求说明书 §6`, `需求说明书 §7`, `需求说明书 §11`, `需求说明书 §16`, `需求说明书 §20`, `需求说明书 §27`, `需求说明书 §28.2`, `需求说明书 §28.4`
> 范围声明: 范围内

---

## 1. 模块定位/概述

### 1.1 职责边界

| 职责           | 包含                                                                    | 不包含                                      |
| -------------- | ----------------------------------------------------------------------- | ------------------------------------------- |
| 广播兼容性判定 | `can_broadcast()`、`broadcast_shape()`，按 NumPy 规则从尾轴开始逐轴比对 | 自动触发广播、隐式改写其他模块的 shape 契约 |
| 广播步长计算   | `broadcast_strides()` 生成目标视图步长；广播轴写入 `0`                  | 负步长、复制式 reshape、额外布局模式        |
| 广播视图创建   | `broadcast_to()`、`broadcast_with()` 返回零拷贝共享底层数据的只读视图   | 可写广播视图、共享可写广播结果              |
| 类型层维度推导 | 通过 `D1: BroadcastDim<D2>` 这一 public sealed trait 在编译期确定输出维度类型                    | 在类型层替代运行时 shape 兼容性检查         |
| 广播语义收敛   | 广播结果统一视为“共享底层数据且绝不暴露写权限”的只读广播语义            | 多输入同步迭代调度、多操作数广播编排        |

### 1.2 设计原则

| 原则             | 体现                                                                                     |
| ---------------- | ---------------------------------------------------------------------------------------- |
| NumPy 一致性     | 从尾轴开始比对；轴长度相同或一方为 `1` 时兼容，否则返回 `XenonError::BroadcastError`。   |
| 零拷贝优先       | 广播只改写 shape/stride/flags，不复制底层数据。                                          |
| 共享只读         | 广播结果始终降级为只读视图；任何可变访问都必须在类型层或运行时显式拒绝。                 |
| 步长显式化       | 广播轴使用 `usize` 零步长表达，与 `06-layout.md` 中的 `BroadcastView` 布局状态保持一致。 |
| 类型与运行时分层 | `BroadcastDim` 作为 public sealed trait 负责输出维度类型推导，`broadcast_shape()` 负责实际兼容性裁决。            |

### 1.3 在架构中的位置

```text
Dependency layers:
L0: error, private
L1: dimension, element, complex
L2: layout
L3: storage
L4: tensor
L5: broadcast  ← current module
L6: shape, index, iter, math, overload
```

广播位于 `tensor` 之上、逐元素运算/形状操作之下：它消费张量元数据并产出只读广播视图，为运算符重载、逐元素运算和迭代路径提供统一的广播前置语义。

---

## 2. 需求映射与范围约束

| 类型     | 内容                                                                                                                                    |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| 需求映射 | `需求说明书 §16`；同时受 `需求说明书 §6` 存储模式、`需求说明书 §7` 内存布局、`需求说明书 §11` 迭代器、`需求说明书 §20` 运算符重载、`需求说明书 §27` 错误处理、`需求说明书 §28.2` 测试交付、`需求说明书 §28.4` 边界覆盖约束。                                                       |
| 范围内   | `broadcast_shape()`、`can_broadcast()`、`broadcast_strides()`、`broadcast_to()`、`broadcast_with()`、零步长广播视图、共享只读广播结果。 |
| 范围外   | 可写广播视图、隐式广播、多操作数统一调度、负步长广播、复制式 expand/reshape。                                                           |
| 非目标   | 不在本文引入新的布局状态、存储模式、自动类型提升或任何额外第三方依赖。                                                                  |

---

## 3. 文件位置

```text
src/broadcast/
├── mod.rs             # module entry, re-export public functions and trait-bound-related entry points
├── shape.rs           # can_broadcast(), broadcast_shape(), broadcast_strides()
└── view.rs            # TensorBase::broadcast_to(), broadcast_with()
```

文件划分理由：广播模块天然分为“兼容性/步长规则”和“视图构造”两部分；前者只处理 shape 与 stride 元数据，后者负责把结果降级为只读广播视图。采用 `src/broadcast/` 目录结构能使规则函数与视图入口分离，同时保持当前版本只覆盖显式广播能力。

---

## 4. 依赖关系

### 4.1 依赖图（ASCII）

```text
                    ┌──────────────┐
                    │    tensor    │
                    │ TensorBase   │
                    │ TensorView   │
                    └──────┬───────┘
                           │ uses
              ┌────────────┼───────────────┐
              │        broadcast            │
              │   shape.rs + view.rs        │
              └──┬──────────┬──────────┬────┘
                 │ uses     │ uses     │ uses
          ┌──────▼───┐ ┌────▼─────┐ ┌──▼──────────┐
          │ dimension│ │  layout  │ │    error    │
          │Dimension │ │Strides   │ │ XenonError  │
          │BroadcastDim││Flags/State││             │
          └──────────┘ └──────────┘ └─────────────┘
```

### 4.2 类型级依赖表

| 来源模块    | 使用的类型/trait                                                                                                                                    |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tensor`    | `TensorBase<S, D>`, `TensorView<'a, A, D>`, `.shape()`, `.strides()`, `.offset()`, 视图构造入口，以及从任意受支持存储模式降级到只读广播视图的入口。 |
| `dimension` | `Dimension`, `Ix0`~`Ix6`, `IxDyn`, `BroadcastDim<Other>`（public sealed trait，对外可命名）。                                                                                          |
| `layout`    | `Strides<D>`, `LayoutFlags`, `LayoutState::FContiguous`, `LayoutState::NonContiguous`, `LayoutState::BroadcastView`。                               |
| `error`     | `XenonError::BroadcastError`, `XenonError::InvalidArgument`。                                                                                       |

### 4.3 依赖方向声明

> **依赖方向：单向向上。** `broadcast/` 消费 `tensor`、`dimension`、`layout` 与 `error` 的既有能力，不反向定义这些核心模块的语义。

### 4.4 依赖合法性与新增依赖说明

| 项目           | 说明                                                                                         |
| -------------- | -------------------------------------------------------------------------------------------- |
| 新增第三方依赖 | 无                                                                                           |
| 合法性结论     | 合法；当前设计仅使用 Xenon 既有模块与标准库，符合本文前述需求映射以及最小依赖、单 crate 约束。 |
| 替代方案       | 不适用；广播规则与只读视图构造可直接在现有模块边界内完成。                                   |

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

pub fn broadcast_with<'a, A, S1, D, S2, E>(
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

### 5.2 API 语义约束

| API                   | 语义                                                                                                                                                                         |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `can_broadcast()`     | 仅回答兼容性，不分配、不生成中间结果。                                                                                                                                       |
| `broadcast_shape()`   | 运行时计算公共 shape；不兼容时返回 `XenonError::BroadcastError`。                                                                                                            |
| `broadcast_strides()` | 对齐原 shape 与目标 shape，广播轴写入 `0` 步长；输入长度非法时返回 `InvalidArgument`。                                                                                       |
| `broadcast_to()`      | 显式广播入口；成功时返回共享底层数据的只读 `TensorView`。结果必须满足 `需求说明书 §6` 对“共享只读引用”的约束：可在多个张量实例之间共享同一底层数据，但不提供可写访问权。     |
| `broadcast_with()`    | 面向两个张量输入的专用助手：先计算共同 shape，再分别构造两个只读广播视图。它不承担通用 shape 工具职责；仅需 shape 判定时应使用 `can_broadcast()` / `broadcast_shape()`。 |

> **同形状快捷路径**：当两个输入形状完全相同时，`broadcast_with()` 可直接返回两个原始视图而不执行步长重写，因为目标 shape 与输入 shape 一致。
> **目标秩语义**：`broadcast_to()` 的目标 shape 秩决定了输出视图的维度类型；标量广播到高维时，缺失前导轴按 `1` 补齐。
> **`IntoDimension` 说明：** `IntoDimension` 只决定目标 rank/type；逐轴长度兼容性完全由 `broadcast_shape()` / `broadcast_strides()` 在运行时检查。
> **类型设计说明：** `broadcast_to()` 是目标 shape 主导的 API，只需目标维度类型 `E: IntoDimension`。`broadcast_with()` 是双输入 shape 合流 API，需要双向 `BroadcastDim` 一致性以保证输出维度类型的静态可推导性。`BroadcastDim` 本身是 public sealed trait，因此在这些公开签名中对外可命名。
> **类型说明：** 当前版本继续复用 `TensorView` 作为返回类型，而不是引入单独的 `BroadcastView` 新类型；广播结果内部承载 `ViewRepr<'a, A>`，`storage_kind()` 返回 `StorageKind::View`。广播结果的只读共享语义通过 `LayoutFlags::HAS_ZERO_STRIDE` / `LayoutState::BroadcastView` 与访问控制 API 保证：任何试图取得其可变访问权的 API 在类型层或运行时拒绝。

> **共享只读强制说明：** `broadcast_to()` / `broadcast_with()` 返回 `TensorView`，其生命周期绑定源张量。由于广播结果引入零步长布局，可能导致多个逻辑位置映射到同一物理元素，因此：1) 广播结果在 API 层不提供可变访问入口；2) `storage_kind()` 仍返回 `StorageKind::View`；3) 只读共享语义由 `LayoutFlags::HAS_ZERO_STRIDE`、`LayoutState::BroadcastView`、缺失的 `StorageMut` 能力以及不提供 `into_mut()` 等 API 共同保证。

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

### 5.4 设计决策内联标注

> **设计决策：** 当前版本不承诺多输入同步迭代接口；广播模块只负责 shape 判定、零步长生成和共享只读视图构造。

---

## 6. 内部实现设计

### 6.1 广播不变式

- 广播必须是零拷贝；不得复制底层数据。
- 广播结果只能返回只读 `TensorView`，并按共享只读引用处理；这里的“共享只读引用”含义与 `需求说明书 §6` 一致：结果可在多个张量实例之间共享同一底层数据，但不提供可写访问权。
- 广播轴的 stride 必须写成 `0`，且 stride 类型保持为 `usize`。
- 若结果存在零步长轴，则布局状态必须标记为 `LayoutState::BroadcastView`。
- 广播不改变底层 storage、offset 与逻辑元素顺序语义。
- 所有 shape 兼容性裁决必须在创建结果视图前完成。

### 6.2 广播形状算法

```text
broadcast_shape(shape_a, shape_b):
    1. Align dimensions from right to left.
    2. Treat missing leading dimensions as 1.
    3. If two aligned dimensions differ and neither is 1, return BroadcastError.
    4. Otherwise choose the non-1 dimension, with 0 preserved when paired with 1.
    5. Return the computed IxDyn shape.
```

NumPy 兼容性规则由 `broadcast_shape()` 和 `can_broadcast()` 共用：从尾轴开始逐轴比较，当轴长度相同或其中一方为 `1` 时兼容，否则不兼容。缺失的前导轴按 `1` 处理，因此标量与低维输入可广播到更高维目标。

### 6.3 广播步长算法

```text
broadcast_strides(orig_shape, orig_strides, target_shape):
    1. Validate rank compatibility.
    2. Right-align the original shape against the target shape.
    3. For each axis:
        - if original dimension == target dimension, keep the original stride;
        - if original dimension == 1 and target dimension > 1, write stride 0;
        - otherwise return BroadcastError.
    4. Mark the result layout as BroadcastView when any stride is 0.
```

对广播轴写入零步长意味着该轴被逻辑扩展，但所有索引都回落到同一底层元素；这与 `06-layout.md` §5.7 的零步长语义保持一致。若任一轴出现 `0` 步长，结果即不再视为普通 `FContiguous` 或一般 `NonContiguous` 视图，而统一进入 `BroadcastView`。

> **再次广播规则：** 对已广播视图再次广播时，已有零步长轴保持为 `0`，新增广播轴也写入 `0` 步长；结果 `shape` 取“当前视图 shape”与“新目标 shape”的广播结果。

> **布局标志重算规则：** `ALIGNED` 继承源视图；若任一轴 stride 为 `0`，设置 `BroadcastView` flag。`F_CONTIGUOUS` 仅在不存在零步长且结果 stride 仍满足 F-order 规则时保留。

### 6.4 共享只读视图构造

`broadcast_to()` 和 `broadcast_with()` 只负责在元数据层重建视图：保留原始 storage 与 offset，改写 shape、strides 与 flags。返回类型虽然仍是 `TensorView`，但其公开语义必须统一收敛到共享只读引用：广播结果内部承载 `ViewRepr<'a, A>`，`storage_kind()` 返回 `StorageKind::View`；只读共享语义由广播布局标志与访问控制 API 共同保证，任何试图从广播结果取得可变访问权的 API，都必须在类型层缺失或运行时返回错误。

**安全性论证（unchecked 视图构造）：** 若内部使用 `TensorView::new_unchecked()` 或等价未检查构造器，调用点必须先证明：1）目标 `shape` 与源 `shape` 广播兼容；2）新 `shape` / `stride` / `offset` 组合不会访问到底层 storage 可见边界之外；3）任何零步长元素都不会通过结果视图暴露为可变访问。

### 6.5 `BroadcastDim` 的职责边界

`BroadcastDim<Other>` 是 public sealed trait，因此在公开 API 中可被外部稳定命名；它仅用于编译期计算输出维度类型：

- `IxN BroadcastDim IxN` → `IxN`
- `IxN BroadcastDim IxDyn` → `IxDyn`
- `IxDyn BroadcastDim IxN` → `IxDyn`
- `IxDyn BroadcastDim IxDyn` → `IxDyn`

它不负责判定具体 shape 值是否兼容；例如两个 `Ix2` 仍可能因 `[2, 3]` 与 `[4, 3]` 不兼容而在运行时失败。运行时裁决始终由 `broadcast_shape()` / `broadcast_with()` 完成。

### 6.6 与 §6 存储系统的对接

- **查询：** 广播结果内部使用 `ViewRepr<'a, A>`，因此 `storage_kind()` 返回 `StorageKind::View`；是否为广播结果由 layout flags 中的 `LayoutFlags::HAS_ZERO_STRIDE` / `LayoutState::BroadcastView` 指示。
- **转换：** 广播结果可通过显式分配转成 `Owned` 连续张量（如 `to_owned()` / `to_contiguous()` 一类路径）；由于广播视图存在零步长别名，当前版本不允许把它转换为 `ViewMut`，也不提供 `into_mut()`。
- **线程：** 广播 `ViewRepr` 遵循标准借用规则；当 `A: Sync` 时可满足只读跨线程共享前提，`Send`/`Sync` 语义与普通只读视图一致，不因广播额外放宽。

---

## 7. 实现任务拆分

### 基础前提：规则函数与类型边界

- [ ] **T1**: 建立 `src/broadcast/mod.rs` 骨架与公共导出
  - 文件: `src/broadcast/mod.rs`, `src/broadcast/shape.rs`, `src/broadcast/view.rs`
  - 内容: 模块声明、规则函数声明、视图相关入口占位
  - 测试: 编译通过
  - 前置: `dimension`、`layout`、`tensor`、`error` 模块完成
  - 预计: 10 min

- [ ] **T2a**: 实现 `can_broadcast()`
  - 文件: `src/broadcast/shape.rs`
  - 内容: 尾轴对齐兼容性判定
  - 测试: `test_can_broadcast_compatible`
  - 前置: T1
  - 预计: 5 min

- [ ] **T2b**: 实现 `broadcast_shape()`
  - 文件: `src/broadcast/shape.rs`
  - 内容: 公共 shape 推导与结构化广播错误填充
  - 测试: `test_broadcast_shape_error`
  - 前置: T2a
  - 预计: 5 min

- [ ] **T2c**: 实现 `broadcast_strides()`
  - 文件: `src/broadcast/shape.rs`
  - 内容: 零步长写入与参数前提校验
  - 测试: `test_broadcast_strides_zero_stride`
  - 前置: T2b
  - 预计: 10 min

### Wave 1: 视图构造基础

- [ ] **T3a**: 实现 `broadcast_to()` 基本路径
  - 文件: `src/broadcast/view.rs`
  - 内容: 目标 shape 校验与只读视图构造
  - 测试: `test_broadcast_to_basic`
  - 前置: T2c
  - 预计: 10 min

### Wave 2: 视图构造补全

- [ ] **T3b**: 实现 `broadcast_to()` 错误路径与布局更新
  - 文件: `src/broadcast/view.rs`
  - 内容: 非法目标 shape 错误返回与 `BroadcastView` 布局状态更新
  - 测试: `test_broadcast_to_error`, `test_broadcast_read_only`
  - 前置: T3a
  - 预计: 10 min

- [ ] **T4**: 实现 `broadcast_with()`
  - 文件: `src/broadcast/view.rs`
  - 内容: 公共 shape 推导、双输入广播、`BroadcastDim` 输出类型对齐
  - 测试: `test_broadcast_with_same_shape`, `test_broadcast_scalar_and_tensor`, `test_broadcast_with_incompatible_shapes`
  - 前置: T2c
  - 预计: 15 min

### Wave 3: 综合验证

- [ ] **T5**: 编写单元与集成测试
  - 文件: `tests/test_broadcast.rs`, `tests/property_tests.rs`, `tests/property/shape_props.rs`
  - 内容: 兼容性规则、零步长语义、共享只读边界、属性测试
  - 测试: 覆盖范围内所有公开 API
  - 前置: T3b, T4
  - 预计: 20 min

### 并行执行图

```text
基础前提: [T1] -> [T2a] -> [T2b] -> [T2c]
                                       │
Wave 1:                      [T3a]      [T4]
                                       │   │
Wave 2:                               [T3b]
                                       └─┬─┘
                                         │
Wave 3:                                 [T5]
```

---

## 8. 测试计划

### 8.1 测试分类表

| 测试分类 | 位置                                                 | 说明                                                           |
| -------- | ---------------------------------------------------- | -------------------------------------------------------------- |
| 单元测试 | `src/broadcast/*` 的 `#[cfg(test)]`                  | 验证 shape 兼容性、零步长生成和错误结构。                      |
| 集成测试 | `tests/test_broadcast.rs`                            | 验证广播与 `tensor`、`layout`、`overload`、`iter` 的协同路径。 |
| 边界测试 | 同模块测试中标注                                     | 覆盖标量、空数组、再次广播、高维广播和 `10^7` 元素大张量广播。 |
| 属性测试 | `tests/property_tests.rs`, `tests/property/shape_props.rs` | 验证广播 shape/stride 不变量和零拷贝语义。                     |

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
| `test_broadcast_large_tensor_zero_copy`  | `10^7` 元素级广播保持零拷贝与零步长语义 | 高     |

### 8.3 边界测试场景

| 场景                                | 预期行为                                           |
| ----------------------------------- | -------------------------------------------------- |
| `[0, 3]` 与 `[1, 3]`                | 允许空轴广播，结果 shape 为 `[0, 3]`，不复制数据。 |
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

### 8.5 Feature gate / 配置测试

| 配置                          | 验证点                                                          |
| ----------------------------- | --------------------------------------------------------------- |
| 默认配置                      | 显式广播 API、零步长和共享只读语义保持一致。                    |
| `rayon` / `simd` feature 开关 | 广播模块本身不改变语义；不同执行路径不得改变 shape 与错误类别。 |
| 无额外 feature                | 当前模块不新增独立 feature gate。                               |

### 8.6 类型边界 / 编译期测试

| 场景                                                  | 测试方式                     |
| ----------------------------------------------------- | ---------------------------- |
| `BroadcastDim<Other>::Output` 对静态/动态维组合可编译 | compile-pass 测试            |
| 运行时不兼容 shape 仍可在方法级返回 `BroadcastError`  | 编译通过 + 运行时断言        |
| 广播结果无可变访问 API                                | compile-fail 或 API 缺失断言 |
| `broadcast_to()` 接受 `E: IntoDimension`              | 编译期签名检查               |

---

## 9. 模块交互设计

### 9.1 接口约定

| 方向                        | 对方模块                       | 接口/类型                                       | 约定                                                     |
| --------------------------- | ------------------------------ | ----------------------------------------------- | -------------------------------------------------------- |
| `broadcast → tensor`        | `tensor`                       | `TensorBase`, `TensorView`                      | 读取 shape/stride/offset，并通过只读视图入口构造结果。   |
| `broadcast → dimension`     | `dimension`                    | `Dimension`, `BroadcastDim`                     | 运行时 shape 计算与编译期输出维度类型推导分离；`BroadcastDim` 为 public sealed trait，可在公开签名中稳定命名。          |
| `broadcast → layout`        | `layout`                       | `Strides<D>`, `LayoutState::BroadcastView`      | 零步长轴必须映射到广播视图布局状态。                     |
| `broadcast → error`         | `error`                        | `XenonError::BroadcastError`, `InvalidArgument` | 广播不兼容与参数前提失败都必须返回结构化错误。           |
| `overload/math ← broadcast` | `19-overload.md`, `11-math.md` | `broadcast_with()`, `broadcast_shape()`         | 二元运算先广播再计算，不允许各模块私自重复定义广播规则。 |
| `iter ← broadcast`          | `10-iterator.md`               | 只读广播视图                                    | 广播结果可被读取遍历，但不得提供可变迭代能力。           |

### 9.2 数据流描述

```text
User calls broadcast_to() or broadcast_with()
    │
    ├── broadcast_shape() checks NumPy compatibility from trailing axes
    ├── broadcast_strides() writes zero strides for expanded axes
    ├── tensor view constructor reuses original storage and offset
    └── result is exposed as a shared read-only broadcast view
```

---

## 10. 错误处理与语义边界

| 主题              | 内容                                                                                                                                                                              |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Recoverable error | 广播不兼容时统一返回 `XenonError::BroadcastError { operation, lhs_shape, rhs_shape, attempted_target_shape, axis }`；例如 `broadcast_to`、`broadcast_shape`、`broadcast_with` 都必须填充结构化字段。 |
| 参数错误          | 当 `orig_shape.len() != orig_strides.len()` 等公开前提被破坏时，`broadcast_strides()` 返回 `XenonError::InvalidArgument { operation, argument, expected, actual, axis, shape }`。 |
| Panic             | 不允许把 shape 不兼容隐藏为 panic；公开 API 使用 `Result` 表达失败。                                                                                                              |
| 语义边界          | 广播只负责显式元数据扩展，不改变元素值、不重排数据、不授予可写访问。                                                                                                              |
| 路径一致性        | 默认路径、后续可能启用的 SIMD/并行消费路径都必须共享同一广播规则与错误类别；广播模块自身不分裂语义分支。                                                                          |

### 10.1 错误示例

```rust,ignore
XenonError::BroadcastError {
    operation: "broadcast_to",
    lhs_shape: self.shape().to_vec(),
    rhs_shape: shape.slice().to_vec(),
    attempted_target_shape: Some(shape.slice().to_vec()),
    axis: None,
}

XenonError::BroadcastError {
    operation: "broadcast_shape",
    lhs_shape: shape_a.to_vec(),
    rhs_shape: shape_b.to_vec(),
    attempted_target_shape: None,
    axis: Some(axis_from_right),
}
```

```rust,ignore
XenonError::InvalidArgument {
    operation: "broadcast_strides".into(),
    argument: "orig_strides".into(),
    expected: "orig_shape.len() == orig_strides.len()".into(),
    actual: format!("shape={}, strides={}", orig_shape.len(), orig_strides.len()).into(),
    axis: None,
    axis_len: None,
    start: None,
    end: None,
    shape: Some(orig_shape.to_vec()),
}
```

文档中不得再使用 `shape_a`、`shape_b`、`from`、`to` 等旧字段名来描述广播错误结构。

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
| 决策     | `BroadcastDim` 作为 public sealed trait 只负责输出维度类型推导，实际兼容性由运行时函数检查。   |
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

## 版本历史

| 版本  | 日期       |
| ----- | ---------- |
| 1.0.0 | 2026-04-14 |
| 1.0.1 | 2026-04-15 |
| 1.0.2 | 2026-04-15 |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
