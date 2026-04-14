# 广播模块设计

> 文档编号: 15 | 模块: `src/broadcast.rs` | 阶段: Phase 3
> 前置文档: `02-dimension.md`, `06-memory.md`, `07-tensor.md`, `26-error.md`
> 需求参考: 需求说明书 §16
> 范围声明: 范围内

---

## §1 Overview（概述）

广播模块负责 Xenon 的零拷贝广播规则与只读广播视图创建。它严格遵循当前项目的约束：只支持 F-order 语义、布局步长使用 `usize`、广播结果只能是只读视图。

职责边界如下：

- 范围内：`broadcast_shape()`、`can_broadcast()`、`broadcast_strides()`、`broadcast_to()`、`broadcast_with()`
- 范围外：可写广播视图、多操作数调度、多输入同步迭代接口、负步长广播
- 关键目标：零拷贝、结构化错误、与 NumPy 广播规则兼容

## §2 Data Structures（数据结构）

### 2.1 核心元数据

- 输入 shape：`&[usize]`
- 输入 strides：`&[usize]`
- 目标 shape：`&[usize]`
- 输出广播步长：`Vec<usize>`，广播轴的步长为 `0`
- 布局状态：仅允许 `FContiguous`、`NonContiguous`、`BroadcastView`

### 2.2 相关类型

```rust
pub fn broadcast_shape(shape_a: &[usize], shape_b: &[usize]) -> Result<IxDyn, XenonError>;

pub fn can_broadcast(shape_a: &[usize], shape_b: &[usize]) -> bool;

pub fn broadcast_strides(
    orig_shape: &[usize],
    orig_strides: &[usize],
    target_shape: &[usize],
) -> Result<Vec<usize>, XenonError>;
```

广播模块不定义独立错误类型，必须统一返回 `XenonError`。

## §3 API（接口）

```rust
impl<'a, A, D> TensorView<'a, A, D>
where
    D: Dimension,
{
    pub fn broadcast_to<E>(&self, shape: E) -> Result<TensorView<'a, A, E>, XenonError>
    where
        E: Dimension;
}

pub fn broadcast_with<'a, A, D, E>(
    a: &TensorView<'a, A, D>,
    b: &TensorView<'a, A, E>,
) -> Result<
    (
        TensorView<'a, A, <D as BroadcastDim<E>>::Output>,
        TensorView<'a, A, <D as BroadcastDim<E>>::Output>,
    ),
    XenonError,
>
where
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>;
```

当前版本不再承诺任何多输入同步迭代集成接口；广播模块只负责 shape 与 view 元数据，不负责多数组同步迭代。

## §4 Algorithm（算法）

### §4.1 Invariants（不变式）

- 广播必须是零拷贝；不得复制底层数据
- 广播结果只能返回只读 `TensorView`
- 广播轴的步长必须写成 `0`，且步长类型为 `usize`
- 若结果存在零步长轴，则布局状态必须标记为 `BroadcastView`
- 广播不改变底层 storage、offset 与元素顺序语义
- 所有 shape 兼容性裁决都必须在创建结果视图前完成

### §4.2 Error Scenarios（错误场景）

广播相关错误必须对齐 `26-error.md` 的规范字段：

```rust
XenonError::BroadcastError {
    operation: "broadcast_to",
    input_shape: Cow::Owned(self.shape().to_vec()),
    target_shape: Cow::Owned(shape.slice().to_vec()),
    axis: None,
}

XenonError::BroadcastError {
    operation: "broadcast_shape",
    input_shape: Cow::Owned(shape_a.to_vec()),
    target_shape: Cow::Owned(shape_b.to_vec()),
    axis: Some(axis_from_right),
}
```

```rust
XenonError::InvalidArgument {
    operation: "broadcast_strides",
    argument: "orig_strides",
    expected: "orig_shape.len() == orig_strides.len()",
    actual: format!("shape={}, strides={}", orig_shape.len(), orig_strides.len()),
    axis: None,
    shape: Some(Cow::Borrowed(orig_shape)),
}
```

文档中不得再使用 `shape_a`、`shape_b`、`from`、`to` 等旧字段名来描述广播错误。

### 4.3 广播算法

```text
broadcast_shape(shape_a, shape_b):
    1. Align dimensions from right to left.
    2. Treat missing leading dimensions as 1.
    3. If two aligned dimensions differ and neither is 1, return BroadcastError.
    4. Otherwise choose the non-1 dimension, with 0 preserved when paired with 1.
    5. Return the computed IxDyn shape.
```

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

## §5 Testing（测试）

| 测试函数 | 目的 |
| --- | --- |
| `test_can_broadcast_compatible` | 兼容 shape 判定正确 |
| `test_can_broadcast_incompatible` | 不兼容 shape 判定正确 |
| `test_broadcast_shape_basic` | 公共 shape 推导正确 |
| `test_broadcast_shape_error` | 返回 `BroadcastError` 且字段完整 |
| `test_broadcast_strides_zero_stride` | 广播轴步长为 0 |
| `test_broadcast_strides_non_negative` | 非广播轴保持 `usize` 步长 |
| `test_broadcast_to_basic` | 只读广播视图创建正确 |
| `test_broadcast_to_error` | 非法目标 shape 返回结构化错误 |
| `test_broadcast_with_same_shape` | 双输入公共 shape 正确 |
| `test_broadcast_read_only` | 广播视图不提供可写入口 |

还必须覆盖以下边界：

- `[0, 3]` 与 `[1, 3]` 的空轴广播
- 标量广播到高维 shape
- 输入本身已是广播视图时的再次广播

## §6 References（参考）

- `02-dimension.md`
- `06-memory.md`
- `07-tensor.md`
- `26-error.md`
- 需求说明书 §16
