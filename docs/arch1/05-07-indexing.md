# 索引操作模块设计文档

> **模块路径**: `src/index/`
> **版本**: v18
> **日期**: 2026-03-28
> **前置文档**: 02-project-architecture.md, 03-06-tensor-core.md, 03-05-memory-layout.md, 05-01-iterator.md

---

## 1. 模块概述

### 1.1 定位

索引操作模块是 Xenon 张量库的数据访问核心，提供多维数组元素的读取、写入和选择能力。它实现了多种索引模式，从简单的多维索引到复杂的布尔掩码和条件选择。

### 1.2 核心职责

| 职责 | 说明 |
|------|------|
| 多维索引 | `[i, j, k]` 形式的元素访问 |
| 切片索引 | 范围选择，支持步长和负索引 |
| 切片宏 | `s![]` 宏的语法解析和展开 |
| 高级索引 | take/take_along_axis/mask/compress/put |
| 条件选择 | `where(condition, x, y)` 三元选择 |
| 索引查询 | argwhere/nonzero 返回非零元素位置 |

### 1.3 设计理念

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         索引系统架构                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Index Trait 层                                │   │
│  │  std::ops::Index  +  std::ops::IndexMut  +  TensorIndex        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                  │                                      │
│  ┌───────────────────────────────┼───────────────────────────────┐     │
│  │                               │                               │     │
│  ▼                               ▼                               ▼     │
│ ┌─────────────┐          ┌─────────────┐          ┌─────────────┐     │
│ │  MultiDim   │          │   Slice     │          │  Advanced   │     │
│ │  多维索引    │          │  切片索引    │          │  高级索引    │     │
│ └─────────────┘          └─────────────┘          └─────────────┘     │
│                                                                         │
│ ┌─────────────┐          ┌─────────────┐          ┌─────────────┐     │
│ │   take()    │          │   mask()    │          │   where()   │     │
│ │  按索引取    │          │  布尔掩码    │          │  条件选择    │     │
│ └─────────────┘          └─────────────┘          └─────────────┘     │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     基础设施层                                    │   │
│  │   SliceInfo  │  SliceInfoElem  │  边界检查  │  unchecked 变体    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.4 与 ndarray 的对比

| 特性 | ndarray | Xenon |
|------|---------|-------|
| 多维索引 | `arr[[i, j]]` | `arr[[i, j]]` |
| 切片宏 | `s![a..b, c..d]` | `s![a..b, c..d]` |
| take | `arr.select()` | `arr.take()` |
| take_along_axis | `arr.select_axis()` | `arr.take_along_axis()` |
| 布尔掩码 | `arr.mask()` | `arr.mask()` |
| 条件选择 | 无内置 | `where(cond, x, y)` |
| 边界检查 | checked + unsafe | checked + unchecked 变体 |
| 负索引 | 支持 | 支持 |

---

## 2. 文件结构

```
src/index/
├── mod.rs             # 索引 trait 定义、公开导出
├── multi_dim.rs       # 多维索引 [i, j, k]
├── slice_index.rs     # 切片索引、SliceInfo 类型
├── slice_macro.rs     # s![] 宏定义
├── advanced.rs        # 高级索引（take, take_along_axis, mask, compress, put）
├── where_.rs          # where 条件选择
└── nonzero.rs         # argwhere/nonzero
```

### 2.1 各文件职责

| 文件 | 职责 | 可见性 |
|------|------|--------|
| `mod.rs` | 模块导出、`TensorIndex` trait 定义 | pub |
| `multi_dim.rs` | 多维索引实现、Index/IndexMut trait | pub |
| `slice_index.rs` | SliceInfo 类型、切片索引实现 | pub |
| `slice_macro.rs` | `s![]` 宏定义 | pub |
| `advanced.rs` | take/take_along_axis/mask/compress/put | pub |
| `where_.rs` | where 条件选择实现 | pub |
| `nonzero.rs` | argwhere/nonzero 实现 | pub |

---

## 3. 多维索引设计

### 3.1 概述

多维索引使用 `[i, j, k, ...]` 形式访问张量元素，支持正索引和负索引（从末尾计数）。

### 3.2 Index/IndexMut trait 实现

```rust
use std::ops::{Index, IndexMut};

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// 多维索引访问元素（只读）。
    ///
    /// # 参数
    ///
    /// * `index` - 多维索引，如 `[i, j, k]`
    ///
    /// # 返回
    ///
    /// 元素的不可变引用。
    ///
    /// # Panics
    ///
    /// 若索引越界，panic。
    ///
    /// # 示例
    ///
    /// ```
    /// let tensor = Tensor2::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6])?;
    /// 
    /// assert_eq!(tensor[[0, 0]], 1);
    /// assert_eq!(tensor[[1, 2]], 6);
    /// 
    /// // 负索引
    /// assert_eq!(tensor[[-1, -1]], 6);  // 最后一行最后一列
    /// ```
    fn index(&self, index: [usize; N]) -> &A
    where
        D: Dimensionable<N>,
    {
        let normalized = self.normalize_index(&index);
        self.check_index_bounds(&normalized);
        
        // SAFETY: bounds checked above
        unsafe { self.get_unchecked(&normalized) }
    }
}

impl<S, D, A> TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    D: Dimension,
{
    /// 多维索引访问元素（可变）。
    ///
    /// # 示例
    ///
    /// ```
    /// let mut tensor = Tensor2::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6])?;
    /// 
    /// tensor[[0, 0]] = 10;
    /// assert_eq!(tensor[[0, 0]], 10);
    /// ```
    fn index_mut(&mut self, index: [usize; N]) -> &mut A
    where
        D: Dimensionable<N>,
    {
        let normalized = self.normalize_index(&index);
        self.check_index_bounds(&normalized);
        
        // SAFETY: bounds checked above
        unsafe { self.get_unchecked_mut(&normalized) }
    }
}
```

### 3.3 get/get_mut 方法

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// 获取元素引用，越界返回 None。
    ///
    /// # 参数
    ///
    /// * `index` - 多维索引切片
    ///
    /// # 返回
    ///
    /// `Some(&A)` 若索引合法，否则 `None`。
    ///
    /// # 示例
    ///
    /// ```
    /// let tensor = Tensor2::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6])?;
    /// 
    /// assert_eq!(tensor.get(&[0, 0]), Some(&1));
    /// assert_eq!(tensor.get(&[2, 0]), None);  // 越界
    /// ```
    pub fn get(&self, index: &[usize]) -> Option<&A> {
        let normalized = self.try_normalize_index(index)?;
        if self.is_index_valid(&normalized) {
            // SAFETY: bounds verified
            Some(unsafe { self.get_unchecked(&normalized) })
        } else {
            None
        }
    }
    
    /// 获取元素可变引用，越界返回 None。
    pub fn get_mut(&mut self, index: &[usize]) -> Option<&mut A>
    where
        S: StorageMut<Elem = A>,
    {
        let normalized = self.try_normalize_index(index)?;
        if self.is_index_valid(&normalized) {
            // SAFETY: bounds verified
            Some(unsafe { self.get_unchecked_mut(&normalized) })
        } else {
            None
        }
    }
}
```

### 3.4 unchecked 变体

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// 获取元素引用，不进行边界检查。
    ///
    /// # Safety
    ///
    /// 调用者必须保证 `index` 在有效范围内：
    /// - 每个维度 `0 <= index[i] < shape[i]`
    ///
    /// 若索引越界，行为未定义（可能访问非法内存）。
    ///
    /// # 示例
    ///
    /// ```
    /// let tensor = Tensor2::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6])?;
    /// 
    /// // SAFETY: [0, 0] is within bounds [2, 3]
    /// unsafe {
    ///     assert_eq!(*tensor.get_unchecked(&[0, 0]), 1);
    /// }
    /// ```
    pub unsafe fn get_unchecked(&self, index: &[usize]) -> &A {
        let offset = self.compute_offset(index);
        let ptr = self.as_ptr().add(offset);
        &*ptr
    }
    
    /// 获取元素可变引用，不进行边界检查。
    ///
    /// # Safety
    ///
    /// 调用者必须保证 `index` 在有效范围内。
    pub unsafe fn get_unchecked_mut(&mut self, index: &[usize]) -> &mut A
    where
        S: StorageMut<Elem = A>,
    {
        let offset = self.compute_offset(index);
        let ptr = self.as_mut_ptr().add(offset);
        &mut *ptr
    }
}
```

### 3.5 负索引处理

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// 将索引标准化（处理负索引）。
    ///
    /// # 负索引语义
    ///
    /// | 索引值 | 含义 |
    /// |--------|------|
    /// | `i >= 0` | 从开头计数（0-based） |
    /// | `i < 0` | 从末尾计数（-1 = 最后一个元素） |
    ///
    /// # Panics
    ///
    /// 若负索引超出范围（如 shape=5 时使用 -6），panic。
    fn normalize_index(&self, index: &[isize]) -> Vec<usize> {
        let shape = self.shape();
        if index.len() != shape.len() {
            panic!(
                "Index dimension mismatch: expected {}, got {}",
                shape.len(),
                index.len()
            );
        }
        
        index
            .iter()
            .zip(shape.iter())
            .map(|(&idx, &dim)| {
                if idx < 0 {
                    let normalized = (dim as isize + idx) as usize;
                    if normalized >= dim {
                        panic!(
                            "Negative index {} out of bounds for dimension {}",
                            idx, dim
                        );
                    }
                    normalized
                } else {
                    idx as usize
                }
            })
            .collect()
    }
    
    /// 尝试标准化索引，失败返回 None。
    fn try_normalize_index(&self, index: &[isize]) -> Option<Vec<usize>> {
        let shape = self.shape();
        if index.len() != shape.len() {
            return None;
        }
        
        let normalized: Option<Vec<_>> = index
            .iter()
            .zip(shape.iter())
            .map(|(&idx, &dim)| {
                if idx < 0 {
                    let normalized = (dim as isize + idx) as usize;
                    if normalized >= dim {
                        None
                    } else {
                        Some(normalized)
                    }
                } else if (idx as usize) < dim {
                    Some(idx as usize)
                } else {
                    None
                }
            })
            .collect();
        
        normalized
    }
}
```

### 3.6 边界检查 API 对称性

| 方法 | 返回类型 | 边界检查 | 行为 |
|------|----------|----------|------|
| `[[i, j]]` | `&A` | panic | 越界时 panic |
| `get(&[i, j])` | `Option<&A>` | 返回 None | 越界时返回 None |
| `get_unchecked(&[i, j])` | `&A` | 无（unsafe） | 越界时 UB |
| `[[i, j]] = v` | `&mut A` | panic | 越界时 panic |
| `get_mut(&[i, j])` | `Option<&mut A>` | 返回 None | 越界时返回 None |
| `get_unchecked_mut(&[i, j])` | `&mut A` | 无（unsafe） | 越界时 UB |

---

## 4. 切片索引设计

### 4.1 SliceInfoElem 类型

```rust
/// 切片元素类型，表示单个轴的切片描述。
#[derive(Debug, Clone, PartialEq)]
pub enum SliceInfoElem {
    /// 单个索引，如 `arr.slice(s![1, ..])` 中的 `1`。
    /// 
    /// 该轴将被移除，结果维度减少 1。
    Index(isize),
    
    /// 范围切片，如 `1..5` 或 `..3` 或 `2..`。
    /// 
    /// 该轴保留，长度为范围长度。
    Range {
        /// 起始索引（包含），None 表示 0
        start: Option<isize>,
        /// 结束索引（不包含），None 表示轴长度
        end: Option<isize>,
        /// 步长，None 表示 1
        step: Option<isize>,
    },
    
    /// 新轴，如 NumPy 的 `np.newaxis`。
    /// 
    /// 在该位置插入长度为 1 的新轴。
    NewAxis,
}
```

### 4.2 SliceInfo 类型

```rust
/// 切片信息，包含所有轴的切片描述。
///
/// # 类型参数
///
/// * `I` - 输出维度类型
/// * `D` - 输入维度类型
/// * `N` - 切片描述数量（通常等于输入维度数）
#[derive(Debug, Clone)]
pub struct SliceInfo<I, D, N>
where
    I: Dimension,
    D: Dimension,
    N: NdIndex,
{
    /// 各轴的切片描述。
    indices: Vec<SliceInfoElem>,
    
    /// 输出维度类型标记。
    _out_dim: PhantomData<I>,
    
    /// 输入维度类型标记。
    _in_dim: PhantomData<D>,
}

impl<I, D, N> SliceInfo<I, D, N>
where
    I: Dimension,
    D: Dimension,
    N: NdIndex,
{
    /// 创建切片信息。
    pub fn new(indices: Vec<SliceInfoElem>) -> Result<Self, ShapeError> {
        // Validate indices
        Self::validate(&indices)?;
        Ok(SliceInfo {
            indices,
            _out_dim: PhantomData,
            _in_dim: PhantomData,
        })
    }
    
    /// 获取各轴的切片描述。
    pub fn indices(&self) -> &[SliceInfoElem] {
        &self.indices
    }
    
    /// 计算切片后的输出形状。
    pub fn out_shape(&self, in_shape: &[usize]) -> Vec<usize> {
        self.indices
            .iter()
            .zip(in_shape.iter().chain(std::iter::repeat(&1)))
            .filter_map(|(elem, &dim)| match elem {
                SliceInfoElem::Range { start, end, step } => {
                    let (s, e, st) = self.normalize_range(*start, *end, *step, dim);
                    let len = if st > 0 {
                        (e.saturating_sub(s)).div_ceil(st as usize)
                    } else {
                        (s.saturating_sub(e)).div_ceil((-st) as usize)
                    };
                    Some(len)
                }
                SliceInfoElem::NewAxis => Some(1),
                SliceInfoElem::Index(_) => None, // Axis removed
            })
            .collect()
    }
}
```

### 4.3 s![] 宏完整语法规范

```rust
/// 切片宏，用于创建类型安全的切片描述。
///
/// # 语法
///
/// ```text
/// s![elem1, elem2, ..., elemN]
/// ```
///
/// 其中每个 `elem` 可以是：
///
/// | 语法 | 含义 | SliceInfoElem 变体 |
/// |------|------|-------------------|
/// | `..` | 全范围 | `Range { start: None, end: None, step: None }` |
/// | `a..b` | 范围 [a, b) | `Range { start: Some(a), end: Some(b), step: None }` |
/// | `a..` | 范围 [a, end] | `Range { start: Some(a), end: None, step: None }` |
/// | `..b` | 范围 [0, b) | `Range { start: None, end: Some(b), step: None }` |
/// | `a..b;c` | 带步长范围 | `Range { start: Some(a), end: Some(b), step: Some(c) }` |
/// | `..;c` | 全范围带步长 | `Range { start: None, end: None, step: Some(c) }` |
/// | `i` | 单索引 | `Index(i)` |
/// | `-i` | 负索引 | `Index(-i)` |
/// | `NewAxis` 或 `*` | 新轴 | `NewAxis` |
///
/// # 示例
///
/// ```
/// let tensor = Tensor3::<f64>::zeros([4, 5, 6]);
///
/// // 全范围
/// let view = tensor.slice(s![.., .., ..]);
///
/// // 范围切片
/// let view = tensor.slice(s![1..3, 2..5, ..]);
///
/// // 负索引
/// let view = tensor.slice(s![.., -2.., ..]);
///
/// // 步长
/// let view = tensor.slice(s![..;2, ..;3, ..]);  // 每 2/3 个取一个
///
/// // 混合
/// let view = tensor.slice(s![1, 2..4, ..]);  // 第一轴取索引 1，降维
///
/// // 新轴
/// let view = tensor.slice(s![.., .., NewAxis, ..]);  // 插入新轴
/// ```
#[macro_export]
macro_rules! s {
    // Empty slice
    () => {
        $crate::index::SliceInfo::new(vec![]).unwrap()
    };
    
    // Single element patterns
    (..) => {
        $crate::index::SliceInfoElem::Range {
            start: None,
            end: None,
            step: None,
        }
    };
    
    // Range with step: a..b;c
    ($start:expr .. $end:expr ; $step:expr) => {
        $crate::index::SliceInfoElem::Range {
            start: Some($start as isize),
            end: Some($end as isize),
            step: Some($step as isize),
        }
    };
    
    // Range with step: a..;c
    ($start:expr .. ; $step:expr) => {
        $crate::index::SliceInfoElem::Range {
            start: Some($start as isize),
            end: None,
            step: Some($step as isize),
        }
    };
    
    // Range with step: ..b;c
    (.. $end:expr ; $step:expr) => {
        $crate::index::SliceInfoElem::Range {
            start: None,
            end: Some($end as isize),
            step: Some($step as isize),
        }
    };
    
    // Full range with step: ..;c
    (.. ; $step:expr) => {
        $crate::index::SliceInfoElem::Range {
            start: None,
            end: None,
            step: Some($step as isize),
        }
    };
    
    // Range: a..b
    ($start:expr .. $end:expr) => {
        $crate::index::SliceInfoElem::Range {
            start: Some($start as isize),
            end: Some($end as isize),
            step: None,
        }
    };
    
    // Range from: a..
    ($start:expr ..) => {
        $crate::index::SliceInfoElem::Range {
            start: Some($start as isize),
            end: None,
            step: None,
        }
    };
    
    // Range to: ..b
    (.. $end:expr) => {
        $crate::index::SliceInfoElem::Range {
            start: None,
            end: Some($end as isize),
            step: None,
        }
    };
    
    // Index
    ($idx:expr) => {
        $crate::index::SliceInfoElem::Index($idx as isize)
    };
    
    // Multiple elements
    ($($elem:tt),+ $(,)?) => {
        $crate::index::SliceInfo::new(vec![
            $(s!(@elem $elem)),+
        ]).unwrap()
    };
    
    // Internal: element expansion
    (@elem ..) => {
        $crate::index::SliceInfoElem::Range {
            start: None,
            end: None,
            step: None,
        }
    };
    
    (@elem NewAxis) => {
        $crate::index::SliceInfoElem::NewAxis
    };
    
    (@elem *) => {
        $crate::index::SliceInfoElem::NewAxis
    };
    
    (@elem $start:expr .. $end:expr ; $step:expr) => {
        $crate::index::SliceInfoElem::Range {
            start: Some($start as isize),
            end: Some($end as isize),
            step: Some($step as isize),
        }
    };
    
    (@elem $start:expr ..) => {
        $crate::index::SliceInfoElem::Range {
            start: Some($start as isize),
            end: None,
            step: None,
        }
    };
    
    (@elem .. $end:expr) => {
        $crate::index::SliceInfoElem::Range {
            start: None,
            end: Some($end as isize),
            step: None,
        }
    };
    
    (@elem $idx:expr) => {
        $crate::index::SliceInfoElem::Index($idx as isize)
    };
}
```

### 4.4 负索引语义

```rust
impl SliceInfoElem {
    /// 标准化范围参数。
    ///
    /// # 负索引语义
    ///
    /// | 参数 | 正值 | 负值 |
    /// |------|------|------|
    /// | start | 从开头计数 | 从末尾计数 |
    /// | end | 从开头计数 | 从末尾计数 |
    /// | step | 正向遍历 | 反向遍历 |
    ///
    /// # 示例
    ///
    /// ```
    /// // shape = [10]
    /// // start=-3, end=None, step=None => [7, 8, 9]
    /// // start=2, end=-2, step=None => [2, 3, 4, 5, 6, 7]
    /// // start=-1, end=None, step=-1 => [9, 8, 7, ...]
    /// ```
    fn normalize_range(
        &self,
        start: Option<isize>,
        end: Option<isize>,
        step: Option<isize>,
        dim: usize,
    ) -> (usize, usize, isize) {
        let step = step.unwrap_or(1);
        
        let start = match start {
            Some(s) if s < 0 => (dim as isize + s).max(0) as usize,
            Some(s) => (s as usize).min(dim),
            None if step < 0 => dim,  // Reverse: start from end
            None => 0,
        };
        
        let end = match end {
            Some(e) if e < 0 => (dim as isize + e).max(0) as usize,
            Some(e) => (e as usize).min(dim),
            None if step < 0 => 0,  // Reverse: end at start
            None => dim,
        };
        
        (start, end, step)
    }
}
```

### 4.5 步长处理

```rust
impl SliceInfoElem {
    /// 计算切片后的子视图步长。
    ///
    /// # 步长计算
    ///
    /// 切片步长 = 原始步长 × 切片步长
    ///
    /// # 示例
    ///
    /// ```
    /// // 原始: strides = [1, 3], slice = s![..;2, ..]
    /// // 结果: strides = [2, 3]  (第一轴步长翻倍)
    /// ```
    fn compute_slice_stride(&self, orig_stride: isize) -> isize {
        match self {
            SliceInfoElem::Range { step, .. } => {
                orig_stride * step.unwrap_or(1)
            }
            SliceInfoElem::Index(_) | SliceInfoElem::NewAxis => orig_stride,
        }
    }
}
```

### 4.6 slice/slice_mut 方法

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// 创建切片视图。
    ///
    /// # 参数
    ///
    /// * `info` - 切片信息，通常由 `s![]` 宏创建
    ///
    /// # 返回
    ///
    /// 切片后的视图（零拷贝）。
    ///
    /// # Panics
    ///
    /// - 切片维度与张量维度不匹配
    /// - 索引越界
    ///
    /// # 示例
    ///
    /// ```
    /// let tensor = Tensor3::from_shape_vec([4, 5, 6], vec![...])?;
    /// 
    /// // 切片获取子数组
    /// let view = tensor.slice(s![1..3, 2..5, ..]);
    /// assert_eq!(view.shape(), &[2, 3, 6]);
    /// 
    /// // 单索引降维
    /// let view = tensor.slice(s![1, .., ..]);
    /// assert_eq!(view.shape(), &[5, 6]);
    /// ```
    pub fn slice<I, N>(&self, info: SliceInfo<I, D, N>) -> TensorView<'_, A, I>
    where
        I: Dimension,
        N: NdIndex,
    {
        let out_shape = info.out_shape(self.shape());
        let out_strides = info.out_strides(self.strides());
        let offset = info.compute_offset(self.shape());
        
        // Create view
        TensorView::from_raw_parts(
            unsafe { self.as_ptr().add(offset) },
            out_shape,
            out_strides,
        )
    }
    
    /// 创建可变切片视图。
    pub fn slice_mut<I, N>(&mut self, info: SliceInfo<I, D, N>) -> TensorViewMut<'_, A, I>
    where
        I: Dimension,
        N: NdIndex,
        S: StorageMut<Elem = A>,
    {
        let out_shape = info.out_shape(self.shape());
        let out_strides = info.out_strides(self.strides());
        let offset = info.compute_offset(self.shape());
        
        TensorViewMut::from_raw_parts_mut(
            unsafe { self.as_mut_ptr().add(offset) },
            out_shape,
            out_strides,
        )
    }
}
```

### 4.7 unchecked 切片变体

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// 创建切片视图，不进行边界检查。
    ///
    /// # Safety
    ///
    /// 调用者必须保证：
    /// - 切片信息中的所有索引在有效范围内
    /// - 步长不为零（除非是广播视图）
    ///
    /// # 示例
    ///
    /// ```
    /// let tensor = Tensor3::from_shape_vec([4, 5, 6], vec![...])?;
    /// 
    /// // SAFETY: slice bounds are valid
    /// unsafe {
    ///     let view = tensor.slice_unchecked(s![1..3, 2..5, ..]);
    /// }
    /// ```
    pub unsafe fn slice_unchecked<I, N>(&self, info: SliceInfo<I, D, N>) -> TensorView<'_, A, I>
    where
        I: Dimension,
        N: NdIndex,
    {
        let out_shape = info.out_shape(self.shape());
        let out_strides = info.out_strides(self.strides());
        let offset = info.compute_offset_unchecked(self.shape());
        
        TensorView::from_raw_parts(
            self.as_ptr().add(offset),
            out_shape,
            out_strides,
        )
    }
    
    /// 创建可变切片视图，不进行边界检查。
    ///
    /// # Safety
    ///
    /// 同 `slice_unchecked`。
    pub unsafe fn slice_unchecked_mut<I, N>(
        &mut self,
        info: SliceInfo<I, D, N>,
    ) -> TensorViewMut<'_, A, I>
    where
        I: Dimension,
        N: NdIndex,
        S: StorageMut<Elem = A>,
    {
        let out_shape = info.out_shape(self.shape());
        let out_strides = info.out_strides(self.strides());
        let offset = info.compute_offset_unchecked(self.shape());
        
        TensorViewMut::from_raw_parts_mut(
            self.as_mut_ptr().add(offset),
            out_shape,
            out_strides,
        )
    }
}
```

### 4.8 切片 API 对称性

| 方法 | 返回类型 | 边界检查 | 行为 |
|------|----------|----------|------|
| `slice(s![..])` | `TensorView` | panic | 越界时 panic |
| `slice_mut(s![..])` | `TensorViewMut` | panic | 越界时 panic |
| `slice_unchecked(s![..])` | `TensorView` | 无（unsafe） | 越界时 UB |
| `slice_unchecked_mut(s![..])` | `TensorViewMut` | 无（unsafe） | 越界时 UB |

---

## 5. take 设计

### 5.1 概述

`take` 按索引数组从张量中提取元素，沿指定轴（默认展平后取）。

### 5.2 函数签名

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Clone,
{
    /// 按索引数组提取元素。
    ///
    /// # 参数
    ///
    /// * `indices` - 索引数组
    /// * `axis` - 可选，指定沿哪个轴取；None 表示展平后取
    ///
    /// # 返回
    ///
    /// 新分配的 `Tensor`，包含索引位置的元素。
    ///
    /// # Panics
    ///
    /// - 索引越界
    /// - 指定的轴超出维度数
    ///
    /// # 示例
    ///
    /// ```
    /// let tensor = Tensor1::from_vec(vec![10, 20, 30, 40, 50]);
    /// let indices = Tensor1::from_vec(vec![0, 2, 4]);
    /// 
    /// let result = tensor.take(&indices, None);
    /// assert_eq!(result.as_slice(), &[10, 30, 50]);
    /// 
    /// // 沿轴取
    /// let tensor2d = Tensor2::from_shape_vec([3, 4], vec![...])?;
    /// let result = tensor2d.take(&indices, Some(Axis(0)));
    /// // result.shape() = [3, 4]  (3 个索引 × 4 列)
    /// ```
    pub fn take<DI>(
        &self,
        indices: &TensorBase<SI, DI>,
        axis: Option<Axis>,
    ) -> Tensor<A, IxDyn>
    where
        SI: Storage<Elem = usize>,
        DI: Dimension,
    {
        match axis {
            None => self.take_flat(indices),
            Some(axis) => self.take_along_axis_internal(indices, axis),
        }
    }
    
    /// 展平后按索引取元素。
    fn take_flat<SI, DI>(
        &self,
        indices: &TensorBase<SI, DI>,
    ) -> Tensor<A, IxDyn>
    where
        SI: Storage<Elem = usize>,
        DI: Dimension,
    {
        let flat_len = self.len();
        let mut result = Vec::with_capacity(indices.len());
        
        for &idx in indices.iter() {
            if idx >= flat_len {
                panic!(
                    "Index {} out of bounds for flattened tensor of length {}",
                    idx, flat_len
                );
            }
            result.push(self.get_flat(idx).clone());
        }
        
        Tensor::from_shape_vec(indices.shape().to_vec(), result).unwrap()
    }
    
    /// 获取展平后的元素。
    fn get_flat(&self, index: usize) -> &A {
        // Convert flat index to multi-dimensional index
        let md_index = self.flat_to_md_index(index);
        &self[&md_index]
    }
}
```

### 5.3 take 与切片的区别

| 特性 | take | 切片 (slice) |
|------|------|--------------|
| 返回类型 | Owned（拷贝） | View（零拷贝） |
| 索引来源 | 动态数组 | 静态范围 |
| 支持重复索引 | 是 | 否 |
| 支持任意顺序 | 是 | 仅连续/步长 |

---

## 6. take_along_axis 设计

### 6.1 概述

`take_along_axis` 沿指定轴按索引数组取元素，索引数组的形状需与源张量兼容。

### 6.2 函数签名

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Clone,
{
    /// 沿指定轴按索引取元素。
    ///
    /// # 参数
    ///
    /// * `indices` - 索引数组，形状须与源张量兼容
    /// * `axis` - 沿哪个轴取
    ///
    /// # 形状要求
    ///
    /// `indices` 的形状在除指定轴外的所有维度上须与源张量一致。
    /// 指定轴的长度可以不同。
    ///
    /// # 返回
    ///
    /// 新分配的 `Tensor`，形状与 `indices` 相同。
    ///
    /// # Panics
    ///
    /// - 索引越界
    /// - 形状不兼容
    ///
    /// # 示例
    ///
    /// ```
    /// // 源张量: [3, 4]
    /// let tensor = Tensor2::from_shape_vec(
    ///     [3, 4],
    ///     vec![10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33],
    /// )?;
    /// 
    /// // 索引: [3, 1]，沿轴 1 取
    /// // 索引数组形状须为 [3, 1]
    /// let indices = Tensor2::from_shape_vec([3, 1], vec![0, 2, 1])?;
    /// 
    /// let result = tensor.take_along_axis(&indices, Axis(1));
    /// // result = [[10], [22], [31]]
    /// assert_eq!(result.shape(), &[3, 1]);
    /// ```
    pub fn take_along_axis<SI, DI>(
        &self,
        indices: &TensorBase<SI, DI>,
        axis: Axis,
    ) -> Tensor<A, DI>
    where
        SI: Storage<Elem = usize>,
        DI: Dimension,
    {
        self.take_along_axis_internal(indices, axis)
    }
    
    fn take_along_axis_internal<SI, DI>(
        &self,
        indices: &TensorBase<SI, DI>,
        axis: Axis,
    ) -> Tensor<A, DI>
    where
        SI: Storage<Elem = usize>,
        DI: Dimension,
    {
        let axis_idx = axis.0;
        if axis_idx >= self.ndim() {
            panic!("Axis {} out of bounds for {}-D tensor", axis_idx, self.ndim());
        }
        
        // Check shape compatibility
        self.check_take_along_axis_shape(indices.shape(), axis_idx);
        
        let axis_len = self.shape()[axis_idx];
        let mut result = Tensor::uninitialized(indices.shape().to_vec());
        
        // Iterate over all positions and take along axis
        for (out_idx, &idx) in indices.indexed_iter() {
            if idx >= axis_len {
                panic!(
                    "Index {} out of bounds for axis {} of length {}",
                    idx, axis_idx, axis_len
                );
            }
            
            // Build source index
            let src_idx = self.build_source_index(out_idx, idx, axis_idx);
            result[out_idx] = self[&src_idx].clone();
        }
        
        result
    }
    
    /// 检查 take_along_axis 形状兼容性。
    fn check_take_along_axis_shape(&self, indices_shape: &[usize], axis: usize) {
        if indices_shape.len() != self.ndim() {
            panic!(
                "Index dimension mismatch: expected {}, got {}",
                self.ndim(),
                indices_shape.len()
            );
        }
        
        for (i, (&s1, &s2)) in self.shape().iter().zip(indices_shape.iter()).enumerate() {
            if i != axis && s1 != s2 {
                panic!(
                    "Shape mismatch at dimension {}: expected {}, got {}",
                    i, s1, s2
                );
            }
        }
    }
}
```

### 6.3 take_along_axis 与 take 的区别

| 特性 | take | take_along_axis |
|------|------|-----------------|
| 索引形状 | 任意 | 须与源张量兼容 |
| 结果形状 | 索引形状 | 索引形状 |
| 使用场景 | 批量提取 | 每行/列取不同位置 |

---

## 7. mask/compress 设计

### 7.1 概述

- `mask`: 按布尔掩码提取元素，返回 1D 数组
- `compress`: 按布尔掩码沿指定轴提取，保留维度

### 7.2 mask 函数签名

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Clone,
{
    /// 按布尔掩码提取元素。
    ///
    /// # 参数
    ///
    /// * `mask` - 布尔掩码，形状须与源张量一致
    ///
    /// # 返回
    ///
    /// 1D `Tensor`，包含所有 `mask[i] == true` 位置的元素。
    ///
    /// # Panics
    ///
    /// - 形状不匹配
    ///
    /// # 示例
    ///
    /// ```
    /// let tensor = Tensor1::from_vec(vec![1, 2, 3, 4, 5]);
    /// let mask = Tensor1::from_vec(vec![true, false, true, false, true]);
    /// 
    /// let result = tensor.mask(&mask);
    /// assert_eq!(result.as_slice(), &[1, 3, 5]);
    /// ```
    pub fn mask<SM, DM>(
        &self,
        mask: &TensorBase<SM, DM>,
    ) -> Tensor1<A>
    where
        SM: Storage<Elem = bool>,
        DM: Dimension,
    {
        if self.shape() != mask.shape() {
            panic!(
                "Mask shape mismatch: expected {:?}, got {:?}",
                self.shape(),
                mask.shape()
            );
        }
        
        let count = mask.iter().filter(|&&b| b).count();
        let mut result = Vec::with_capacity(count);
        
        for (&elem, &m) in self.iter().zip(mask.iter()) {
            if m {
                result.push(elem.clone());
            }
        }
        
        Tensor1::from_vec(result)
    }
}
```

### 7.3 compress 函数签名

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Clone,
{
    /// 按布尔掩码沿指定轴提取元素。
    ///
    /// # 参数
    ///
    /// * `mask` - 布尔掩码，长度须等于指定轴的长度
    /// * `axis` - 沿哪个轴提取
    ///
    /// # 返回
    ///
    /// `Tensor`，指定轴的长度为 `mask` 中 `true` 的数量。
    ///
    /// # Panics
    ///
    /// - 掩码长度与轴长度不匹配
    /// - 轴越界
    ///
    /// # 示例
    ///
    /// ```
    /// let tensor = Tensor2::from_shape_vec(
    ///     [3, 4],
    ///     vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    /// )?;
    /// 
    /// let mask = Tensor1::from_vec(vec![true, false, true]);  // 保留行 0 和 2
    /// 
    /// let result = tensor.compress(&mask, Axis(0));
    /// // result = [[1, 2, 3, 4], [9, 10, 11, 12]]
    /// assert_eq!(result.shape(), &[2, 4]);
    /// ```
    pub fn compress<SM>(
        &self,
        mask: &Tensor1<bool>,
        axis: Axis,
    ) -> Tensor<A, D>
    where
        SM: Storage<Elem = bool>,
    {
        let axis_idx = axis.0;
        if axis_idx >= self.ndim() {
            panic!("Axis {} out of bounds", axis_idx);
        }
        
        let axis_len = self.shape()[axis_idx];
        if mask.len() != axis_len {
            panic!(
                "Mask length {} does not match axis length {}",
                mask.len(),
                axis_len
            );
        }
        
        // Count true values
        let new_axis_len = mask.iter().filter(|&&b| b).count();
        
        // Build new shape
        let mut new_shape = self.shape().to_vec();
        new_shape[axis_idx] = new_axis_len;
        
        let mut result = Tensor::uninitialized(new_shape);
        
        // Copy elements
        let mut out_idx = 0;
        for (src_idx, &keep) in mask.iter().enumerate() {
            if keep {
                // Copy slice along axis
                let src_slice = self.slice_along_axis(axis, src_idx);
                let mut dst_slice = result.slice_along_axis_mut(axis, out_idx);
                dst_slice.assign(&src_slice);
                out_idx += 1;
            }
        }
        
        result
    }
}
```

### 7.4 mask 与 compress 的区别

| 特性 | mask | compress |
|------|------|----------|
| 返回维度 | 1D | 保持原维度 |
| 掩码形状 | 与源张量相同 | 等于轴长度 |
| 使用场景 | 提取满足条件的元素 | 按轴过滤行/列 |

---

## 8. put 设计

### 8.1 概述

`put` 按索引数组写入元素到张量的指定位置。

### 8.2 函数签名

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    D: Dimension,
    A: Clone,
{
    /// 按索引数组写入元素。
    ///
    /// # 参数
    ///
    /// * `indices` - 索引数组
    /// * `values` - 值数组，形状须与索引数组相同
    /// * `axis` - 可选，指定沿哪个轴；None 表示展平后写
    ///
    /// # Panics
    ///
    /// - 索引越界
    /// - 形状不匹配
    ///
    /// # 示例
    ///
    /// ```
    /// let mut tensor = Tensor1::zeros(5);
    /// let indices = Tensor1::from_vec(vec![0, 2, 4]);
    /// let values = Tensor1::from_vec(vec![1.0, 2.0, 3.0]);
    /// 
    /// tensor.put(&indices, &values, None);
    /// // tensor = [1.0, 0.0, 2.0, 0.0, 3.0]
    /// ```
    pub fn put<SI, DI, SV, DV>(
        &mut self,
        indices: &TensorBase<SI, DI>,
        values: &TensorBase<SV, DV>,
        axis: Option<Axis>,
    ) where
        SI: Storage<Elem = usize>,
        DI: Dimension,
        SV: Storage<Elem = A>,
        DV: Dimension,
    {
        if indices.shape() != values.shape() {
            panic!(
                "Indices and values shape mismatch: {:?} vs {:?}",
                indices.shape(),
                values.shape()
            );
        }
        
        match axis {
            None => self.put_flat(indices, values),
            Some(axis) => self.put_along_axis(indices, values, axis),
        }
    }
    
    fn put_flat<SI, DI, SV, DV>(
        &mut self,
        indices: &TensorBase<SI, DI>,
        values: &TensorBase<SV, DV>,
    ) where
        SI: Storage<Elem = usize>,
        DI: Dimension,
        SV: Storage<Elem = A>,
        DV: Dimension,
    {
        let flat_len = self.len();
        
        for (&idx, &val) in indices.iter().zip(values.iter()) {
            if idx >= flat_len {
                panic!(
                    "Index {} out of bounds for flattened tensor of length {}",
                    idx, flat_len
                );
            }
            *self.get_flat_mut(idx) = val.clone();
        }
    }
    
    fn put_along_axis<SI, DI, SV, DV>(
        &mut self,
        indices: &TensorBase<SI, DI>,
        values: &TensorBase<SV, DV>,
        axis: Axis,
    ) where
        SI: Storage<Elem = usize>,
        DI: Dimension,
        SV: Storage<Elem = A>,
        DV: Dimension,
    {
        let axis_idx = axis.0;
        let axis_len = self.shape()[axis_idx];
        
        for (idx, &src_idx) in indices.iter().enumerate() {
            if src_idx >= axis_len {
                panic!(
                    "Index {} out of bounds for axis {} of length {}",
                    src_idx, axis_idx, axis_len
                );
            }
            
            // Build destination index and assign
            let dst_idx = self.build_dest_index(idx, src_idx, axis_idx);
            self[&dst_idx] = values.flat_get(idx).clone();
        }
    }
}
```

### 8.3 put 与 take 的对称性

| 操作 | take | put |
|------|------|-----|
| 方向 | 读取 | 写入 |
| 参数 | indices | indices + values |
| 返回 | 新 Tensor | 无（原地修改） |
| 存储 | 只读 | 可变 |

---

## 9. argwhere/nonzero 设计

### 9.1 概述

- `argwhere`: 返回非零元素的多维索引
- `nonzero`: 返回各轴的非零索引（类似 NumPy）

### 9.2 argwhere 函数签名

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + PartialEq,
{
    /// 返回非零元素的多维索引。
    ///
    /// # 返回
    ///
    /// 2D `Tensor`，形状为 `[count, ndim]`，每行是一个多维索引。
    ///
    /// # 示例
    ///
    /// ```
    /// let tensor = Tensor2::from_shape_vec(
    ///     [3, 3],
    ///     vec![1, 0, 2, 0, 0, 3, 4, 0, 0],
    /// )?;
    /// 
    /// let indices = tensor.argwhere();
    /// // indices = [[0, 0], [0, 2], [1, 2], [2, 0]]
    /// assert_eq!(indices.shape(), &[4, 2]);
    /// ```
    pub fn argwhere(&self) -> Tensor2<usize> {
        let ndim = self.ndim();
        let mut indices = Vec::new();
        
        for (idx, &elem) in self.indexed_iter() {
            if elem != A::zero() {
                indices.extend_from_slice(idx.as_ref());
            }
        }
        
        let count = indices.len() / ndim;
        Tensor2::from_shape_vec([count, ndim], indices).unwrap()
    }
}
```

### 9.3 nonzero 函数签名

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + PartialEq,
{
    /// 返回各轴的非零索引。
    ///
    /// # 返回
    ///
    /// `Vec<Tensor1<usize>>`，长度为 `ndim`，每个元素是对应轴的非零索引。
    ///
    /// # 示例
    ///
    /// ```
    /// let tensor = Tensor2::from_shape_vec(
    ///     [3, 3],
    ///     vec![1, 0, 2, 0, 0, 3, 4, 0, 0],
    /// )?;
    /// 
    /// let indices = tensor.nonzero();
    /// // indices[0] = [0, 0, 1, 2]  // 非零元素的行索引
    /// // indices[1] = [0, 2, 2, 0]  // 非零元素的列索引
    /// ```
    pub fn nonzero(&self) -> Vec<Tensor1<usize>> {
        let ndim = self.ndim();
        let mut result: Vec<Vec<usize>> = (0..ndim).map(|_| Vec::new()).collect();
        
        for (idx, &elem) in self.indexed_iter() {
            if elem != A::zero() {
                for (i, &dim_idx) in idx.as_ref().iter().enumerate() {
                    result[i].push(dim_idx);
                }
            }
        }
        
        result.into_iter()
            .map(Tensor1::from_vec)
            .collect()
    }
}
```

### 9.4 argwhere 与 nonzero 的区别

| 特性 | argwhere | nonzero |
|------|----------|---------|
| 返回类型 | `Tensor2<usize>` | `Vec<Tensor1<usize>>` |
| 形状 | `[count, ndim]` | `ndim` 个 1D 数组 |
| 使用场景 | 获取完整坐标 | 按轴分离索引 |
| NumPy 等价 | `np.argwhere()` | `np.nonzero()` |

---

## 10. where 条件选择

### 10.1 概述

`where(condition, x, y)` 按条件逐元素选择，返回新数组。

### 10.2 函数签名

```rust
/// 按条件逐元素选择。
///
/// # 参数
///
/// * `condition` - 布尔条件数组
/// * `x` - condition 为 true 时选择的值
/// * `y` - condition 为 false 时选择的值
///
/// # 广播语义
///
/// `condition`、`x`、`y` 三者形状须一致或可广播到同一形状。
/// 结果形状为广播后的形状。
///
/// # 返回
///
/// 新分配的 `Tensor`，形状为广播后形状。
///
/// # Panics
///
/// - 形状无法广播
///
/// # 示例
///
/// ```
/// let cond = Tensor1::from_vec(vec![true, false, true, false]);
/// let x = Tensor1::from_vec(vec![1, 2, 3, 4]);
/// let y = Tensor1::from_vec(vec![10, 20, 30, 40]);
///
/// let result = where_(&cond, &x, &y);
/// // result = [1, 20, 3, 40]
/// ```
pub fn where_<SC, DC, SX, DX, SY, DY, A>(
    condition: &TensorBase<SC, DC>,
    x: &TensorBase<SX, DX>,
    y: &TensorBase<SY, DY>,
) -> Tensor<A, IxDyn>
where
    SC: Storage<Elem = bool>,
    DC: Dimension,
    SX: Storage<Elem = A>,
    DX: Dimension,
    SY: Storage<Elem = A>,
    DY: Dimension,
    A: Clone,
{
    // Compute broadcast shape
    let broadcast_shape = broadcast_shapes_3(
        condition.shape(),
        x.shape(),
        y.shape(),
    ).expect("Cannot broadcast condition, x, and y to same shape");
    
    let mut result = Tensor::uninitialized(broadcast_shape.clone());
    
    // Broadcast all inputs
    let cond_broadcast = condition.broadcast(&broadcast_shape).unwrap();
    let x_broadcast = x.broadcast(&broadcast_shape).unwrap();
    let y_broadcast = y.broadcast(&broadcast_shape).unwrap();
    
    // Iterate and select
    Zip::from(&mut result)
        .and(&cond_broadcast)
        .and(&x_broadcast)
        .and(&y_broadcast)
        .for_each(|out, &cond, &x, &y| {
            *out = if cond { x } else { y };
        });
    
    result
}
```

### 10.3 三操作数广播语义

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         where() 广播规则                                  │
│                                                                          │
│  输入形状:                                                               │
│    condition: [A1, A2, ..., An]                                         │
│    x:         [B1, B2, ..., Bm]                                         │
│    y:         [C1, C2, ..., Ck]                                         │
│                                                                          │
│  广播步骤:                                                               │
│  1. 右对齐所有形状                                                       │
│  2. 补 1 到左侧使维度数相同                                               │
│  3. 逐维比较，取最大值（兼容时）                                          │
│                                                                          │
│  示例:                                                                   │
│    condition: [2, 3]                                                     │
│    x:         [3]      → 广播为 [1, 3]                                   │
│    y:         [2, 1]                                                      │
│    结果:      [2, 3]                                                      │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  condition:  [2, 3]  [[T, F, T], [F, T, F]]                     │   │
│  │  x (broadcast):     [[1, 2, 3], [1, 2, 3]]  ← 从 [3] 广播       │   │
│  │  y (broadcast):     [[7, 7, 7], [8, 8, 8]]  ← 从 [2, 1] 广播    │   │
│  │  ───────────────────────────────────────────────────────────    │   │
│  │  result:     [[1, 7, 3], [8, 2, 8]]                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 10.4 广播形状计算

```rust
/// 计算三个形状的广播结果。
fn broadcast_shapes_3(
    shape1: &[usize],
    shape2: &[usize],
    shape3: &[usize],
) -> Option<Vec<usize>> {
    let max_ndim = shape1.len().max(shape2.len()).max(shape3.len());
    
    let mut result = vec![1usize; max_ndim];
    
    // Process shape1
    for (i, &dim) in shape1.iter().enumerate() {
        let idx = max_ndim - shape1.len() + i;
        result = broadcast_dim(result, idx, dim)?;
    }
    
    // Process shape2
    for (i, &dim) in shape2.iter().enumerate() {
        let idx = max_ndim - shape2.len() + i;
        result = broadcast_dim(result, idx, dim)?;
    }
    
    // Process shape3
    for (i, &dim) in shape3.iter().enumerate() {
        let idx = max_ndim - shape3.len() + i;
        result = broadcast_dim(result, idx, dim)?;
    }
    
    Some(result)
}

fn broadcast_dim(shape: Vec<usize>, idx: usize, dim: usize) -> Option<Vec<usize>> {
    match (shape[idx], dim) {
        (s, d) if s == d => Some(shape),
        (1, d) => {
            let mut new_shape = shape;
            new_shape[idx] = d;
            Some(new_shape)
        }
        (s, 1) => Some(shape),
        _ => None,
    }
}
```

### 10.5 标量广播

```rust
/// where_ 的标量重载。
///
/// # 示例
///
/// ```
/// let cond = Tensor1::from_vec(vec![true, false, true]);
/// let x = 1.0;  // 标量
/// let y = Tensor1::from_vec(vec![10, 20, 30]);
///
/// let result = where_scalar_x(&cond, x, &y);
/// // result = [1, 20, 1]
/// ```
pub fn where_scalar_x<SC, DC, SY, DY, A>(
    condition: &TensorBase<SC, DC>,
    x: A,
    y: &TensorBase<SY, DY>,
) -> Tensor<A, IxDyn>
where
    SC: Storage<Elem = bool>,
    DC: Dimension,
    SY: Storage<Elem = A>,
    DY: Dimension,
    A: Clone,
{
    let broadcast_shape = broadcast_shapes(condition.shape(), y.shape())
        .expect("Cannot broadcast condition and y");
    
    let mut result = Tensor::uninitialized(broadcast_shape.clone());
    
    let cond_broadcast = condition.broadcast(&broadcast_shape).unwrap();
    let y_broadcast = y.broadcast(&broadcast_shape).unwrap();
    
    Zip::from(&mut result)
        .and(&cond_broadcast)
        .and(&y_broadcast)
        .for_each(|out, &cond, &y| {
            *out = if cond { x.clone() } else { y };
        });
    
    result
}

/// where_ 的双标量重载。
pub fn where_scalar<A>(
    condition: &TensorBase<SC, DC>,
    x: A,
    y: A,
) -> Tensor<A, IxDyn>
where
    SC: Storage<Elem = bool>,
    DC: Dimension,
    A: Clone,
{
    let mut result = Tensor::uninitialized(condition.shape().to_vec());
    
    Zip::from(&mut result)
        .and(condition)
        .for_each(|out, &cond| {
            *out = if cond { x.clone() } else { y.clone() };
        });
    
    result
}
```

### 10.6 返回类型说明

| 输入维度 | 结果维度 | 返回类型 |
|----------|----------|----------|
| 全静态相同 | 保持 | `Tensor<A, D>` |
| 混合静态/动态 | 动态 | `Tensor<A, IxDyn>` |
| 含广播 | 动态 | `Tensor<A, IxDyn>` |

---

## 11. 与其他模块的交互

### 11.1 模块依赖图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           index 模块依赖                                  │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                         index                                    │   │
│  │  MultiDim, Slice, take, mask, where, nonzero                   │   │
│  └───────────────────────────────┬─────────────────────────────────┘   │
│                                  │                                      │
│         ┌────────────────────────┼────────────────────────┐            │
│         │                        │                        │            │
│         ▼                        ▼                        ▼            │
│  ┌─────────────┐          ┌─────────────┐          ┌─────────────┐    │
│  │   tensor    │          │   layout    │          │   iter      │   │
│  │ TensorBase  │          │ strides     │          │ Zip         │   │
│  │ TensorView  │          │ Order       │          │ indexed     │   │
│  └─────────────┘          └─────────────┘          └─────────────┘    │
│         │                        │                        │            │
│         └────────────────────────┼────────────────────────┘            │
│                                  │                                      │
│                                  ▼                                      │
│                          ┌─────────────┐                                │
│                          │  dimension  │                                │
│                          │  Ix0-Ix6    │                                │
│                          │  IxDyn      │                                │
│                          └─────────────┘                                │
└─────────────────────────────────────────────────────────────────────────┘
```

### 11.2 与 tensor 模块的接口

```rust
// tensor 模块提供的索引入口方法
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    // 多维索引
    pub fn get(&self, index: &[usize]) -> Option<&A>;
    pub unsafe fn get_unchecked(&self, index: &[usize]) -> &A;
    
    // 切片
    pub fn slice<I, N>(&self, info: SliceInfo<I, D, N>) -> TensorView<'_, A, I>;
    pub unsafe fn slice_unchecked<I, N>(&self, info: SliceInfo<I, D, N>) -> TensorView<'_, A, I>;
    
    // 高级索引
    pub fn take<SI, DI>(&self, indices: &TensorBase<SI, DI>, axis: Option<Axis>) -> Tensor<A, IxDyn>;
    pub fn take_along_axis<SI, DI>(&self, indices: &TensorBase<SI, DI>, axis: Axis) -> Tensor<A, DI>;
    pub fn mask<SM, DM>(&self, mask: &TensorBase<SM, DM>) -> Tensor1<A>;
    pub fn compress<SM>(&self, mask: &Tensor1<bool>, axis: Axis) -> Tensor<A, D>;
    pub fn argwhere(&self) -> Tensor2<usize>;
    pub fn nonzero(&self) -> Vec<Tensor1<usize>>;
}

impl<S, D, A> TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    D: Dimension,
{
    // 可变索引
    pub fn get_mut(&mut self, index: &[usize]) -> Option<&mut A>;
    pub unsafe fn get_unchecked_mut(&mut self, index: &[usize]) -> &mut A;
    
    // 可变切片
    pub fn slice_mut<I, N>(&mut self, info: SliceInfo<I, D, N>) -> TensorViewMut<'_, A, I>;
    pub unsafe fn slice_unchecked_mut<I, N>(&mut self, info: SliceInfo<I, D, N>) -> TensorViewMut<'_, A, I>;
    
    // 写入
    pub fn put<SI, DI, SV, DV>(&mut self, indices: &TensorBase<SI, DI>, values: &TensorBase<SV, DV>, axis: Option<Axis>);
}

// 全局函数
pub fn where_<SC, DC, SX, DX, SY, DY, A>(
    condition: &TensorBase<SC, DC>,
    x: &TensorBase<SX, DX>,
    y: &TensorBase<SY, DY>,
) -> Tensor<A, IxDyn>;
```

### 11.3 与 layout 模块的接口

```rust
// index 模块使用 layout 模块的步长计算
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// 计算索引对应的内存偏移量。
    fn compute_offset(&self, index: &[usize]) -> usize {
        let strides = self.strides();
        index
            .iter()
            .zip(strides.iter())
            .map(|(&idx, &stride)| (idx as isize * stride) as usize)
            .sum()
    }
}
```

### 11.4 与 iter 模块的接口

```rust
// index 模块使用 iter 模块的 Zip 进行广播迭代
use crate::iter::{Zip, NdProducer};

// where_ 函数使用 Zip
pub fn where_<...>(...) -> Tensor<A, IxDyn> {
    // ...
    Zip::from(&mut result)
        .and(&cond_broadcast)
        .and(&x_broadcast)
        .and(&y_broadcast)
        .for_each(|out, &cond, &x, &y| {
            *out = if cond { x } else { y };
        });
    // ...
}

// argwhere 使用 indexed_iter
impl<S, D, A> TensorBase<S, D> {
    pub fn argwhere(&self) -> Tensor2<usize> {
        for (idx, &elem) in self.indexed_iter() {
            // ...
        }
    }
}
```

---

## 12. 实现任务分解

### 任务清单

| # | 任务 | 预估时间 | 依赖 | 产出 |
|---|------|----------|------|------|
| 1 | 实现 `SliceInfoElem` 枚举和基础方法 | 10 min | 无 | `slice_index.rs` |
| 2 | 实现 `SliceInfo` 结构体和形状计算 | 10 min | T1 | `slice_index.rs` |
| 3 | 实现 `s![]` 宏定义 | 15 min | T2 | `slice_macro.rs` |
| 4 | 实现多维索引（Index/IndexMut trait） | 10 min | 无 | `multi_dim.rs` |
| 5 | 实现 get/get_mut/unchecked 变体 | 10 min | T4 | `multi_dim.rs` |
| 6 | 实现负索引标准化 | 10 min | T4 | `multi_dim.rs` |
| 7 | 实现 slice/slice_mut 方法 | 10 min | T2 | `slice_index.rs` |
| 8 | 实现 take/take_along_axis | 15 min | T4 | `advanced.rs` |
| 9 | 实现 mask/compress | 10 min | T4 | `advanced.rs` |
| 10 | 实现 put 方法 | 10 min | T4 | `advanced.rs` |
| 11 | 实现 where_ 函数（含广播） | 15 min | T4, iter | `where_.rs` |
| 12 | 实现 argwhere/nonzero | 10 min | T4, iter | `nonzero.rs` |

### 任务依赖图

```
T1 ──→ T2 ──→ T3 ──┐
        │          │
        └──→ T7 ───┼──→ 完成
                   │
T4 ──→ T5 ────────┤
  │                │
  └──→ T6 ────────┤
  │                │
  ├──→ T8 ────────┤
  │                │
  ├──→ T9 ────────┤
  │                │
  ├──→ T10 ───────┤
  │                │
  └──→ T11 ───────┤
       (需 iter)    │
  │                │
  └──→ T12 ───────┘
       (需 iter)
```

### 并行执行建议

- **Wave 1**: T1, T4（可独立开始）
- **Wave 2**: T2, T5, T6, T8, T9, T10（依赖 T1/T4，可并行）
- **Wave 3**: T3, T7（依赖 T2）
- **Wave 4**: T11, T12（依赖 T4 和 iter 模块）

---

## 13. 设计决策记录

### D1: 为什么 Index trait 使用数组而非元组？

**决策**: 使用 `[usize; N]` 而非 `(usize, usize, ...)` 作为索引类型。

**理由**:
1. **泛型支持**: 数组长度可泛型化，元组长度固定
2. **动态索引**: 支持运行时构造索引
3. **ndarray 兼容**: 与 ndarray 库的 `[[i, j]]` 语法一致

### D2: 为什么提供 unchecked 变体而非仅 unsafe 块？

**决策**: 为所有索引操作提供 `_unchecked` 后缀的安全包装。

**理由**:
1. **API 对称性**: checked 和 unchecked API 命名一致
2. **文档清晰**: unsafe 语义明确标注在函数签名
3. **调用便利**: 无需手写 unsafe 块

### D3: 为什么 where 返回 Owned 而非视图？

**决策**: `where_()` 始终返回新分配的 `Tensor`。

**理由**:
1. **广播语义**: 广播后的结果在原数组中不存在
2. **条件混合**: 结果值来自两个不同源，无法共享存储
3. **需求明确**: 需求文档指定返回 `Owned`

### D4: 为什么 s![] 宏不支持表达式作为索引？

**决策**: 宏参数需为字面量或简单标识符，不支持复杂表达式。

**理由**:
1. **宏复杂性**: 支持任意表达式大幅增加宏复杂度
2. **类型推断**: 简单参数便于类型推断
3. **性能**: 编译时展开更高效

### D5: 为什么 take_along_axis 要求索引形状兼容？

**决策**: 索引数组在除指定轴外的维度须与源张量一致。

**理由**:
1. **语义明确**: 每个位置沿轴取一个元素
2. **NumPy 兼容**: 与 NumPy 行为一致
3. **避免歧义**: 不支持额外广播简化实现

### D6: 为什么 mask 返回 1D 而非保持维度？

**决策**: `mask()` 返回展平的 1D 数组。

**理由**:
1. **非零元素不连续**: 无法表示为连续视图
2. **NumPy 兼容**: `np.extract()` 返回 1D
3. **compress 替代**: 需保持维度时使用 `compress()`

### D7: 为什么步长使用 isize 而非 usize？

**决策**: 步长类型为 `isize`（有符号）。

**理由**:
1. **负步长支持**: 切片 `..;-1` 需要负步长
2. **反转视图**: `flip()` 产生负步长视图
3. **偏移计算**: 负步长时偏移量可能为负

---

## 附录 A: s![] 宏语法快速参考

```
s![] 语法:

基本形式:
  s![elem1, elem2, ..., elemN]

元素类型:
  ┌─────────────────┬───────────────────────────────────┐
  │ 语法            │ 含义                              │
  ├─────────────────┼───────────────────────────────────┤
  │ ..              │ 全范围 [0, dim)                    │
  │ a..b            │ 范围 [a, b)                        │
  │ a..             │ 范围 [a, dim)                      │
  │ ..b             │ 范围 [0, b)                        │
  │ a..b;c          │ 范围 [a, b) 步长 c                 │
  │ ..;c            │ 全范围步长 c                       │
  │ a..;c           │ [a, dim) 步长 c                    │
  │ ..b;c           │ [0, b) 步长 c                      │
  │ i               │ 单索引 i（降维）                   │
  │ -i              │ 负索引（从末尾计数）               │
  │ NewAxis 或 *    │ 插入新轴                          │
  └─────────────────┴───────────────────────────────────┘

示例:
  s![.., .., ..]              # 全范围
  s![1..3, 2..5, ..]          # 范围切片
  s![1, .., ..]               # 单索引降维
  s![..;2, ..;3, ..]          # 带步长
  s![.., -2.., ..]            # 负索引
  s![.., NewAxis, ..]         # 插入新轴
  s![1..3;2, 2..5, -1]        # 混合
```

---

## 附录 B: 边界检查 vs Unchecked API 对照

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     边界检查 vs Unchecked API                            │
│                                                                          │
│  多维索引:                                                               │
│  ┌───────────────────────┬───────────────────────────────────────────┐  │
│  │ Checked (panic)       │ Unchecked (unsafe, UB on invalid)         │  │
│  ├───────────────────────┼───────────────────────────────────────────┤  │
│  │ arr[[i, j]]           │ N/A (使用 get_unchecked)                  │  │
│  │ arr.get(&[i, j])      │ arr.get_unchecked(&[i, j])                │  │
│  │ arr.get_mut(&[i, j])  │ arr.get_unchecked_mut(&[i, j])            │  │
│  └───────────────────────┴───────────────────────────────────────────┘  │
│                                                                          │
│  切片:                                                                   │
│  ┌───────────────────────┬───────────────────────────────────────────┐  │
│  │ Checked (panic)       │ Unchecked (unsafe, UB on invalid)         │  │
│  ├───────────────────────┼───────────────────────────────────────────┤  │
│  │ arr.slice(s![..])     │ arr.slice_unchecked(s![..])               │  │
│  │ arr.slice_mut(s![..]) │ arr.slice_unchecked_mut(s![..])           │  │
│  └───────────────────────┴───────────────────────────────────────────┘  │
│                                                                          │
│  使用建议:                                                               │
│  - 开发阶段: 始终使用 checked API                                        │
│  - 性能关键路径: 验证索引有效后使用 unchecked                           │
│  - 安全优先: 宁可 panic 也不要 UB                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 附录 C: where() 广播示例

```
示例 1: 形状完全一致
  condition: [2, 3]
  x:         [2, 3]
  y:         [2, 3]
  结果:      [2, 3]

示例 2: x 广播
  condition: [2, 3]
  x:         [3]      → 广播为 [1, 3] → [2, 3]
  y:         [2, 3]
  结果:      [2, 3]

示例 3: 三者都广播
  condition: [2, 1]    → 广播为 [2, 3]
  x:         [1, 3]    → 广播为 [2, 3]
  y:         [1]       → 广播为 [1, 1] → [2, 3]
  结果:      [2, 3]

示例 4: 标量混合
  condition: [2, 3]
  x:         标量 1.0  → 广播为 [2, 3]
  y:         [2, 3]
  结果:      [2, 3]

示例 5: 不兼容（报错）
  condition: [2, 3]
  x:         [2, 4]    ← 3 ≠ 4，无法广播
  y:         [2, 3]
  结果:      BroadcastError
```

---

*本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。*
