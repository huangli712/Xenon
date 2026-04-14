# 归约运算模块设计

> 文档编号: 13 | 模块: `src/reduction/` | 阶段: Phase 4
> 前置文档: `02-dimension.md`, `03-element.md`, `07-tensor.md`, `09-parallel.md`, `26-error.md`
> 需求参考: 需求说明书 §14
> 范围声明: 范围内

---

## §1 Overview（概述）

归约模块负责 Xenon 当前版本中唯一受支持的归约族：`sum`。它覆盖全局 `sum`、沿轴 `sum_axis` 与 `sum_axis_keepdims`，并保证以下语义：

- 整数路径使用 checked arithmetic，溢出为 panic
- 浮点路径遵循 IEEE 754 语义，`NaN` 自动传播
- 空数组 `sum` 返回加法单位元
- SIMD 与并行仅在可证明不改变串行结果时参与分派

当前版本明确不包含 `mean`、`var`、`prod`、`min`、`max`、`argmin`、`argmax` 或自定义归约器。

## §2 Data Structures（数据结构）

### 2.1 模块边界

```text
src/reduction/
├── mod.rs
└── sum.rs
```

`reduction/` 只保留最小语义层，后端优化分别委托给 `simd/` 与 `parallel/`；归约模块本身不拥有独立错误类型。

### 2.2 参与的核心类型

- `TensorBase<S, D>` / `Tensor<A, D>`：归约输入与输出载体
- `Axis` / `RemoveAxis`：按轴归约与输出维度推导
- `Numeric` / `CheckedAdd` / `ComplexScalar`：区分整数、浮点、复数语义
- `XenonError`：沿轴归约的可恢复错误返回类型

## §3 API（接口）

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Numeric,
{
    pub fn sum(&self) -> A;

    pub fn sum_axis(&self, axis: Axis) -> Result<Tensor<A, D::Smaller>, XenonError>
    where
        D: RemoveAxis;

    pub fn sum_axis_keepdims(&self, axis: Axis) -> Result<Tensor<A, D>, XenonError>
    where
        D: RemoveAxis;
}
```

沿轴归约的错误返回必须与 `26-error.md` 对齐：

```rust
XenonError::InvalidAxis {
    operation: "sum_axis",
    axis: axis.index(),
    ndim: self.ndim(),
    shape: Cow::Owned(self.shape().to_vec()),
}
```

若是 keepdims 版本，则 `operation` 必须改为 `"sum_axis_keepdims"`，其余字段保持同一结构。

## §4 Algorithm（算法）

### §4.1 Invariants（不变式）

- 公开归约 API 仅支持 `sum`
- `sum()` 对空输入返回 `A::zero()`；不得把空输入当作错误
- `sum_axis()` / `sum_axis_keepdims()` 必须先做 axis 边界检查，再进入具体归约逻辑
- 整数路径溢出必须 panic；不得转成 `XenonError`
- 并行与 SIMD 路径若无法证明与串行结果完全一致，必须回退串行
- 布局只支持 F-order 语义；归约算法不得引入 C-order 假设

### §4.2 Error Scenarios（错误场景）

```rust
XenonError::InvalidAxis {
    operation: "sum_axis",
    axis: axis.index(),
    ndim: self.ndim(),
    shape: Cow::Owned(self.shape().to_vec()),
}

XenonError::InvalidArgument {
    operation: "sum_axis_keepdims",
    argument: "axis",
    expected: "axis < ndim",
    actual: axis.index().to_string(),
    axis: Some(axis.index()),
    shape: Some(Cow::Owned(self.shape().to_vec())),
}
```

设计上优先使用 `InvalidAxis` 表达轴越界；`InvalidArgument` 仅保留给需要同时报告额外参数上下文的内部辅助入口。对外文档不得再使用缺少 `shape` 或 `ndim` 的旧字段形式。

### 4.3 归约路径

```rust
fn sum_int<I: Numeric + CheckedAdd>(iter: impl Iterator<Item = I>) -> I {
    iter.fold(I::zero(), |acc, x| {
        acc.checked_add(x)
            .expect("integer overflow in reduction (sum)")
    })
}

fn sum_float<F: Numeric + Copy>(iter: impl Iterator<Item = F>) -> F {
    iter.fold(F::zero(), |acc, x| acc + x)
}
```

```text
sum_axis_keepdims(tensor, axis):
    1. Validate axis against ndim.
    2. Clone the input shape.
    3. Set result_shape[axis] = 1.
    4. Allocate the output tensor with zeros.
    5. Accumulate each logical input element into the axis-collapsed slot.
    6. Return Tensor<A, D> with the reduced axis length preserved as 1.
```

## §5 Testing（测试）

| 测试函数 | 目的 |
| --- | --- |
| `test_sum_i32` | 整数全局求和正确 |
| `test_sum_overflow_panic` | 整数溢出触发 panic |
| `test_sum_nan` | 浮点 `NaN` 传播 |
| `test_sum_empty` | 空数组返回加法单位元 |
| `test_sum_axis_2d` | 二维沿轴归约正确 |
| `test_sum_axis_keepdims` | keepdims 保留长度 1 |
| `test_sum_axis_invalid_axis` | 非法轴返回 `InvalidAxis` |
| `test_sum_parallel_consistency` | 并行路径与串行一致 |
| `test_sum_simd_consistency` | SIMD 路径与标量一致 |

还必须验证：

- rank-0 输入调用 `sum_axis*` 时返回可恢复错误，而不是依赖 trait 隐式拒绝
- 非连续视图归约结果与连续输入一致
- `simd` feature 仍遵守 `simd = ["dep:pulp", "std"]`

## §6 References（参考）

- `02-dimension.md`
- `03-element.md`
- `07-tensor.md`
- `09-parallel.md`
- `26-error.md`
- 需求说明书 §14
