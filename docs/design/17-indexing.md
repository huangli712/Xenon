# 索引操作模块设计

> 文档编号: 17 | 模块: `src/index/` | 阶段: Phase 3
> 前置文档: `02-dimension.md`, `06-memory.md`, `07-tensor.md`, `26-error.md`
> 需求参考: 需求说明书 §18
> 范围声明: 范围内

---

## §1 Overview（概述）

索引模块负责 Xenon 的多维整数索引、只读切片与 `s![]` 宏。当前版本只支持 `usize` 索引与正步长切片，不支持负索引、负步长和高级索引。

职责边界如下：

- 范围内：`at` / `at_mut` / `get` / `get_unchecked*`、`slice` / `try_slice`、`s![]`
- 范围外：布尔掩码、整数数组高级索引、共享可写切片、负步长视图
- 关键约束：所有偏移量计算都遵循 F-order；安全接口必须返回规范化错误或 `Option`

## §2 Data Structures（数据结构）

### 2.1 多维索引约束

```rust
use crate::private::Sealed;

pub trait NdIndex<D: Dimension>: Sealed {
    fn index_checked(&self, dim: &D, strides: &Strides<D>) -> Option<usize>;

    unsafe fn index_unchecked(&self, strides: &Strides<D>) -> usize;
}
```

### 2.2 切片描述结构

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SliceInfoElem {
    Index(usize),
    Range {
        start: Option<usize>,
        end: Option<usize>,
        step: Option<usize>,
    },
}

#[derive(Debug, Clone)]
pub enum SliceInfoIndices {
    Inline {
        len: u8,
        elems: [SliceInfoElem; 6],
    },
    Dynamic(Vec<SliceInfoElem>),
}
```

索引与切片只使用 `usize`，从类型层面排除负索引与负步长。

## §3 API（接口）

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    pub fn at<I>(&self, index: I) -> Result<&A, XenonError>
    where
        I: NdIndex<D>;

    pub fn get(&self, index: &[usize]) -> Option<&A>;

    pub unsafe fn get_unchecked(&self, index: &[usize]) -> &A;

    pub fn slice<I>(&self, info: SliceInfo<I, D>) -> Result<TensorView<'_, A, I>, XenonError>
    where
        I: Dimension;

    pub fn try_slice<I>(&self, info: SliceInfo<I, D>) -> Result<TensorView<'_, A, I>, XenonError>
    where
        I: Dimension;
}

impl<S, D, A> TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    D: Dimension,
{
    pub fn at_mut<I>(&mut self, index: I) -> Result<&mut A, XenonError>
    where
        I: NdIndex<D>;

    pub fn get_mut(&mut self, index: &[usize]) -> Option<&mut A>;

    pub unsafe fn get_unchecked_mut(&mut self, index: &[usize]) -> &mut A;
}
```

`Index` / `IndexMut` 语法糖仍保留，但失败时是 panic；规范安全路径始终是 `at()` / `at_mut()`。

## §4 Algorithm（算法）

### §4.1 Invariants（不变式）

- 偏移量计算只能使用 F-order：`offset = sum(index[i] * strides[i])`
- `at()` / `at_mut()` 必须先验证 rank 与边界，再生成引用
- `get()` / `get_mut()` 只做轻量检查，失败返回 `None`
- `slice()` / `try_slice()` 只返回只读视图；不得新增共享可写切片 API
- `step == 0` 必须返回可恢复错误
- 切片后布局状态只能落在 `FContiguous`、`NonContiguous`、`BroadcastView` 三种之一

### §4.2 Error Scenarios（错误场景）

轴相关与参数相关错误必须与 `26-error.md` 对齐：

```rust
XenonError::InvalidAxis {
    operation: "slice",
    axis: axis,
    ndim: self.ndim(),
    shape: Cow::Owned(self.shape().to_vec()),
}

XenonError::InvalidArgument {
    operation: "slice",
    argument: "step",
    expected: "step > 0",
    actual: "0".to_string(),
    axis: Some(axis),
    shape: Some(Cow::Owned(self.shape().to_vec())),
}

XenonError::IndexError {
    operation: "at",
    attempted_index: index_component,
    axis,
    shape: Cow::Owned(self.shape().to_vec()),
}
```

```rust
XenonError::InvalidStorageMode {
    operation: "at_mut".into(),
    expected: "writable storage".into(),
    actual: storage_mode.into(),
    shape: Some(Cow::Owned(self.shape().to_vec())),
}
```

文档中不得再使用缺少 `ndim` / `shape` 的旧 `InvalidAxis` 形式，也不得把 `step == 0` 写成未命名的 `InvalidSliceStep` 私有公开错误。

### 4.3 偏移与切片计算

```rust
fn compute_offset_f<D: Dimension>(index: &[usize], strides: &Strides<D>) -> usize {
    let mut offset = 0usize;
    for (&idx, &stride) in index.iter().zip(strides.iter()) {
        offset += idx * stride;
    }
    offset
}
```

```text
compute_slice(shape, strides, offset, slices):
    1. Validate the slice descriptor rank against ndim.
    2. For Index(idx), check bounds and fold into the new offset.
    3. For Range { start, end, step }, reject step == 0.
    4. Compute the resulting axis length and stride multiplication.
    5. Recompute layout flags from the new shape and strides.
    6. Return a read-only TensorView.
```

## §5 Testing（测试）

| 测试函数 | 目的 |
| --- | --- |
| `test_at_2d` | `at()` 成功返回引用 |
| `test_at_out_of_bounds` | 越界返回 `IndexError` |
| `test_index_out_of_bounds` | `Index` 语法糖失败时 panic |
| `test_at_mut_invalid_storage_mode` | 只读存储返回 `InvalidStorageMode` |
| `test_get_returns_none` | `get()` 失败返回 `None` |
| `test_slice_basic` | 基本切片 shape 与数据正确 |
| `test_slice_with_step` | 正步长切片正确 |
| `test_try_slice_step_zero` | `step == 0` 返回 `InvalidArgument` |
| `test_slice_chain` | 视图的视图保持一致 |
| `test_slice_layout_recomputed` | 切片后重新计算布局状态 |

还必须覆盖以下边界：

- rank-0 张量索引
- 广播视图上的只读索引
- 非连续切片后的偏移量计算

## §6 References（参考）

- `02-dimension.md`
- `06-memory.md`
- `07-tensor.md`
- `26-error.md`
- 需求说明书 §18
