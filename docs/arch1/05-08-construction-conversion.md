# 构造与转换模块设计文档

> **模块路径**: `src/construct.rs`, `src/convert.rs`, `src/format.rs`
> **版本**: v18
> **日期**: 2026-03-28
> **需求来源**: require-v18.md §13

---

## 1. 模块概述

### 1.1 职责定义

构造与转换模块是 Xenon 张量库的入口层，负责：

| 职责 | 说明 |
|------|------|
| **张量构造** | 提供多种方式创建张量（零初始化、序列生成、从数据源构造等） |
| **运算符重载** | 实现算术、逻辑运算符及广播语义 |
| **类型转换** | 显式 `cast` 和隐式 `From` trait 转换 |
| **连续性保证** | 提供强制连续布局的转换方法 |
| **格式化输出** | Debug/Display 实现，支持大数组省略显示 |

### 1.2 在架构中的位置

```
依赖层级：
L0: error, private
L1: dimension, element, complex
L2: layout (依赖 dimension)
L3: storage (依赖 layout)
L4: tensor (依赖 storage, dimension)
L5: construct, convert, format  ← 当前模块
L6: ops, iter 等
```

构造与转换模块位于依赖层级 L5，直接依赖 `tensor` 核心模块，为上层用户 API 提供便捷的构造和转换功能。

### 1.3 文件职责

| 文件 | 职责 |
|------|------|
| `src/construct.rs` | 张量构造方法（zeros, ones, arange, from_vec 等） |
| `src/convert.rs` | 类型转换（cast, to_owned, into_owned, From trait） |
| `src/format.rs` | 格式化输出（Debug, Display, 大数组省略） |

---

## 2. 文件结构

```
src/
├── construct.rs       # 张量构造
│   ├── zeros/ones/full/empty
│   ├── eye/identity/diag
│   ├── from_vec/from_slice/from_fn
│   └── arange/linspace/logspace
│
├── convert.rs         # 类型转换
│   ├── cast 方法及精度行为
│   ├── to_owned/into_owned
│   ├── to_f_contiguous/to_c_contiguous/to_contiguous
│   ├── copy_to/fill
│   └── From trait 实现
│
└── format.rs          # 格式化输出
    ├── Debug trait 实现
    ├── Display trait 实现
    └── 大数组省略规则
```

---

## 3. 构造方法设计

### 3.1 zeros/ones/full/empty — 基础构造

#### 3.1.1 完整签名

```rust
// src/construct.rs

use crate::tensor::{Tensor, TensorBase};
use crate::dimension::{Dimension, IntoDimension};
use crate::element::Element;
use crate::layout::MemoryOrder;
use crate::storage::Owned;

impl<A, D> Tensor<A, D>
where
    A: Element,
    D: Dimension,
{
    /// 创建全零张量（默认 F-order）。
    ///
    /// # Arguments
    /// * `shape` - 形状，实现 `IntoDimension` 的类型
    ///
    /// # Examples
    /// ```
    /// let t = Tensor::<f64, _>::zeros([3, 4]);
    /// assert_eq!(t.shape(), &[3, 4]);
    /// ```
    pub fn zeros<Sh>(shape: Sh) -> Self
    where
        A: Default,
        Sh: IntoDimension<Dim = D>,
    {
        Self::zeros_order(shape, MemoryOrder::F)
    }
    
    /// 创建全零张量（指定内存布局）。
    ///
    /// # Arguments
    /// * `shape` - 形状
    /// * `order` - 内存布局（F-order 或 C-order）
    ///
    /// # Examples
    /// ```
    /// let t = Tensor::<f32, _>::zeros_order([2, 3], MemoryOrder::C);
    /// assert!(t.is_c_contiguous());
    /// ```
    pub fn zeros_order<Sh>(shape: Sh, order: MemoryOrder) -> Self
    where
        A: Default,
        Sh: IntoDimension<Dim = D>,
    {
        let dim = shape.into_dimension();
        let len = dim.size();
        let strides = match order {
            MemoryOrder::F => dim.strides_for_f_order(),
            MemoryOrder::C => dim.strides_for_c_order(),
        };
        
        // Allocate and zero-initialize
        let storage = Owned::zeros(len);
        
        TensorBase {
            storage,
            shape: dim,
            strides,
            offset: 0,
        }
    }
    
    /// 创建全一张量。
    ///
    /// # Examples
    /// ```
    /// let t = Tensor::<i32, _>::ones([2, 3]);
    /// assert_eq!(t[[0, 0]], 1);
    /// ```
    pub fn ones<Sh>(shape: Sh) -> Self
    where
        A: Element + Default + crate::element::One,
        Sh: IntoDimension<Dim = D>,
    {
        Self::full(shape, A::one())
    }
    
    /// 创建全一张量（指定内存布局）。
    pub fn ones_order<Sh>(shape: Sh, order: MemoryOrder) -> Self
    where
        A: Element + Default + crate::element::One,
        Sh: IntoDimension<Dim = D>,
    {
        Self::full_order(shape, A::one(), order)
    }
    
    /// 创建填充指定值的张量。
    ///
    /// # Examples
    /// ```
    /// let t = Tensor::<f64, _>::full([2, 2], 3.14);
    /// assert_eq!(t[[1, 1]], 3.14);
    /// ```
    pub fn full<Sh>(shape: Sh, value: A) -> Self
    where
        A: Clone,
        Sh: IntoDimension<Dim = D>,
    {
        Self::full_order(shape, value, MemoryOrder::F)
    }
    
    /// 创建填充指定值的张量（指定内存布局）。
    pub fn full_order<Sh>(shape: Sh, value: A, order: MemoryOrder) -> Self
    where
        A: Clone,
        Sh: IntoDimension<Dim = D>,
    {
        let dim = shape.into_dimension();
        let len = dim.size();
        let strides = match order {
            MemoryOrder::F => dim.strides_for_f_order(),
            MemoryOrder::C => dim.strides_for_c_order(),
        };
        
        let storage = Owned::from_elem(len, value);
        
        TensorBase {
            storage,
            shape: dim,
            strides,
            offset: 0,
        }
    }
    
    /// 创建未初始化张量。
    ///
    /// # Safety
    ///
    /// 返回的张量包含未初始化内存。调用者必须确保在使用前
    /// 正确初始化所有元素。
    ///
    /// # Examples
    /// ```
    /// let mut t = unsafe { Tensor::<f64, _>::empty([3, 4]) };
    /// t.fill(0.0);  // 立即初始化
    /// ```
    pub unsafe fn empty<Sh>(shape: Sh) -> Self
    where
        Sh: IntoDimension<Dim = D>,
    {
        Self::empty_order(shape, MemoryOrder::F)
    }
    
    /// 创建未初始化张量（指定内存布局）。
    pub unsafe fn empty_order<Sh>(shape: Sh, order: MemoryOrder) -> Self
    where
        Sh: IntoDimension<Dim = D>,
    {
        let dim = shape.into_dimension();
        let len = dim.size();
        let strides = match order {
            MemoryOrder::F => dim.strides_for_f_order(),
            MemoryOrder::C => dim.strides_for_c_order(),
        };
        
        let storage = Owned::allocate_aligned(len, Owned::<A>::DEFAULT_ALIGNMENT);
        
        TensorBase {
            storage,
            shape: dim,
            strides,
            offset: 0,
        }
    }
}
```

#### 3.1.2 语义总结

| 方法 | 初始化值 | 内存布局 | 适用场景 |
|------|----------|----------|----------|
| `zeros` | `A::default()` | F-order（默认） | 通用零初始化 |
| `zeros_order` | `A::default()` | 指定 | 需要特定布局 |
| `ones` | `A::one()` | F-order（默认） | 单位矩阵初始化等 |
| `ones_order` | `A::one()` | 指定 | 需要特定布局 |
| `full` | 指定值 | F-order（默认） | 常量填充 |
| `full_order` | 指定值 | 指定 | 需要特定布局 |
| `empty` | 未初始化 | F-order（默认） | 性能关键，手动初始化 |
| `empty_order` | 未初始化 | 指定 | 需要特定布局 |

---

### 3.2 eye/identity/diag — 矩阵构造

#### 3.2.1 完整签名

```rust
// src/construct.rs

impl<A> Tensor<A, Ix2>
where
    A: Element,
{
    /// 创建单位矩阵。
    ///
    /// 对角线元素为 1，其他元素为 0。
    ///
    /// # Arguments
    /// * `n` - 方阵大小（n × n）
    ///
    /// # Examples
    /// ```
    /// let e = Tensor::<f64, _>::eye(3);
    /// assert_eq!(e[[0, 0]], 1.0);
    /// assert_eq!(e[[0, 1]], 0.0);
    /// assert_eq!(e[[1, 1]], 1.0);
    /// ```
    pub fn eye(n: usize) -> Self
    where
        A: Default + crate::element::One,
    {
        Self::identity(n, n)
    }
    
    /// 创建单位矩阵（非方阵）。
    ///
    /// # Arguments
    /// * `rows` - 行数
    /// * `cols` - 列数
    ///
    /// # Examples
    /// ```
    /// let e = Tensor::<f64, _>::identity(3, 4);
    /// // [[1, 0, 0, 0],
    /// //  [0, 1, 0, 0],
    /// //  [0, 0, 1, 0]]
    /// ```
    pub fn identity(rows: usize, cols: usize) -> Self
    where
        A: Default + crate::element::One,
    {
        let mut result = Tensor::zeros([rows, cols]);
        let min_dim = rows.min(cols);
        for i in 0..min_dim {
            result[[i, i]] = A::one();
        }
        result
    }
    
    /// 从对角线元素创建对角矩阵。
    ///
    /// # Arguments
    /// * `diagonal` - 对角线元素的一维张量
    ///
    /// # Examples
    /// ```
    /// let d = Tensor1::from_vec(vec![1.0, 2.0, 3.0]);
    /// let m = Tensor2::diag(&d);
    /// // [[1, 0, 0],
    /// //  [0, 2, 0],
    /// //  [0, 0, 3]]
    /// ```
    pub fn diag(diagonal: &Tensor<A, Ix1>) -> Self
    where
        A: Default + Clone,
    {
        let n = diagonal.len();
        let mut result = Tensor::zeros([n, n]);
        for i in 0..n {
            result[[i, i]] = diagonal[i].clone();
        }
        result
    }
    
    /// 从对角线元素创建带状矩阵（可偏移）。
    ///
    /// # Arguments
    /// * `diagonal` - 对角线元素
    /// * `offset` - 对角线偏移（正数向上，负数向下）
    ///
    /// # Examples
    /// ```
    /// let d = Tensor1::from_vec(vec![1.0, 2.0, 3.0]);
    /// let m = Tensor2::diag_with_offset(&d, 1);  // 上对角线
    /// // [[0, 1, 0, 0],
    /// //  [0, 0, 2, 0],
    /// //  [0, 0, 0, 3],
    /// //  [0, 0, 0, 0]]
    /// ```
    pub fn diag_with_offset(diagonal: &Tensor<A, Ix1>, offset: isize) -> Self
    where
        A: Default + Clone,
    {
        let n = diagonal.len();
        let size = n + offset.unsigned_abs();
        let mut result = Tensor::zeros([size, size]);
        
        for (i, &val) in diagonal.iter().enumerate() {
            let row = if offset >= 0 { i } else { i + offset.unsigned_abs() };
            let col = if offset >= 0 { i + offset as usize } else { i };
            if row < size && col < size {
                result[[row, col]] = val.clone();
            }
        }
        result
    }
    
    /// 提取矩阵的对角线元素。
    ///
    /// # Arguments
    /// * `matrix` - 二维矩阵
    /// * `offset` - 对角线偏移（默认 0）
    ///
    /// # Returns
    /// 对角线元素组成的一维张量
    ///
    /// # Examples
    /// ```
    /// let m = Tensor2::eye(3);
    /// let d = Tensor2::extract_diag(&m, 0);
    /// assert_eq!(d.to_vec(), vec![1.0, 1.0, 1.0]);
    /// ```
    pub fn extract_diag(matrix: &Self, offset: isize) -> Tensor<A, Ix1>
    where
        A: Clone,
    {
        let (rows, cols) = (matrix.shape()[0], matrix.shape()[1]);
        let diag_len = if offset >= 0 {
            (cols - offset as usize).min(rows)
        } else {
            (rows - offset.unsigned_abs()).min(cols)
        };
        
        let mut result = Tensor::zeros([diag_len]);
        for i in 0..diag_len {
            let row = if offset >= 0 { i } else { i + offset.unsigned_abs() };
            let col = if offset >= 0 { i + offset as usize } else { i };
            result[i] = matrix[[row, col]].clone();
        }
        result
    }
}
```

---

### 3.3 from_vec/from_slice/from_fn — 从数据源构造

#### 3.3.1 完整签名

```rust
// src/construct.rs

impl<A, D> Tensor<A, D>
where
    A: Element,
    D: Dimension,
{
    /// 从 Vec 构造张量（推断形状）。
    ///
    /// 仅适用于一维张量。形状由 Vec 长度决定。
    ///
    /// # Examples
    /// ```
    /// let t = Tensor1::from_vec(vec![1.0, 2.0, 3.0]);
    /// assert_eq!(t.shape(), &[3]);
    /// ```
    pub fn from_vec(vec: Vec<A>) -> Tensor<A, Ix1> {
        let len = vec.len();
        let storage = Owned::from_vec_aligned(vec, Owned::<A>::DEFAULT_ALIGNMENT);
        TensorBase {
            storage,
            shape: Ix1(len),
            strides: Ix1(1),
            offset: 0,
        }
    }
    
    /// 从 Vec 构造张量（指定形状）。
    ///
    /// # Errors
    /// 如果 `shape.size() != data.len()`，返回 `ShapeError::IncompatibleSize`。
    ///
    /// # Examples
    /// ```
    /// let t = Tensor::<f64, _>::from_shape_vec([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    /// ```
    pub fn from_shape_vec<Sh>(shape: Sh, data: Vec<A>) -> Result<Self, ShapeError>
    where
        Sh: IntoDimension<Dim = D>,
    {
        Self::from_shape_vec_order(shape, data, MemoryOrder::F)
    }
    
    /// 从 Vec 构造张量（指定形状和布局）。
    pub fn from_shape_vec_order<Sh>(
        shape: Sh,
        data: Vec<A>,
        order: MemoryOrder,
    ) -> Result<Self, ShapeError>
    where
        Sh: IntoDimension<Dim = D>,
    {
        let dim = shape.into_dimension();
        let expected = dim.size();
        if data.len() != expected {
            return Err(ShapeError::IncompatibleSize {
                expected,
                actual: data.len(),
            });
        }
        
        let strides = match order {
            MemoryOrder::F => dim.strides_for_f_order(),
            MemoryOrder::C => dim.strides_for_c_order(),
        };
        
        let storage = Owned::from_vec_aligned(data, Owned::<A>::DEFAULT_ALIGNMENT);
        
        Ok(TensorBase {
            storage,
            shape: dim,
            strides,
            offset: 0,
        })
    }
    
    /// 从切片构造张量（复制数据）。
    ///
    /// # Examples
    /// ```
    /// let data = [1.0, 2.0, 3.0, 4.0];
    /// let t = Tensor1::from_slice(&data);
    /// ```
    pub fn from_slice(slice: &[A]) -> Tensor<A, Ix1>
    where
        A: Clone,
    {
        Self::from_vec(slice.to_vec())
    }
    
    /// 从切片构造张量（指定形状）。
    ///
    /// # Examples
    /// ```
    /// let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let t = Tensor::<f64, _>::from_shape_slice([2, 3], &data)?;
    /// ```
    pub fn from_shape_slice<Sh>(shape: Sh, slice: &[A]) -> Result<Self, ShapeError>
    where
        A: Clone,
        Sh: IntoDimension<Dim = D>,
    {
        let dim = shape.into_dimension();
        let expected = dim.size();
        if slice.len() != expected {
            return Err(ShapeError::IncompatibleSize {
                expected,
                actual: slice.len(),
            });
        }
        Self::from_shape_vec(dim, slice.to_vec())
    }
    
    /// 使用闭包构造张量。
    ///
    /// 闭包接收每个元素的索引（多维），返回元素值。
    ///
    /// # Arguments
    /// * `shape` - 形状
    /// * `f` - 闭包 `|idx: &[usize]| -> A`
    ///
    /// # Examples
    /// ```
    /// let t = Tensor::<i32, _>::from_fn([3, 3], |idx| {
    ///     (idx[0] * 3 + idx[1]) as i32
    /// });
    /// assert_eq!(t[[1, 2]], 5);
    /// ```
    pub fn from_fn<Sh, F>(shape: Sh, mut f: F) -> Self
    where
        Sh: IntoDimension<Dim = D>,
        F: FnMut(&[usize]) -> A,
    {
        Self::from_fn_order(shape, f, MemoryOrder::F)
    }
    
    /// 使用闭包构造张量（指定布局）。
    pub fn from_fn_order<Sh, F>(shape: Sh, mut f: F, order: MemoryOrder) -> Self
    where
        Sh: IntoDimension<Dim = D>,
        F: FnMut(&[usize]) -> A,
    {
        let dim = shape.into_dimension();
        let len = dim.size();
        let strides = match order {
            MemoryOrder::F => dim.strides_for_f_order(),
            MemoryOrder::C => dim.strides_for_c_order(),
        };
        
        // Allocate uninitialized storage
        let mut storage = unsafe { Owned::allocate_aligned(len, Owned::<A>::DEFAULT_ALIGNMENT) };
        
        // Fill with closure
        let mut idx = vec![0usize; dim.ndim()];
        for i in 0..len {
            // Convert linear index to multi-dimensional index
            let mut remaining = i;
            for (axis, &dim_size) in dim.slice().iter().enumerate().rev() {
                idx[axis] = remaining % dim_size;
                remaining /= dim_size;
            }
            
            let value = f(&idx);
            unsafe {
                *storage.as_mut_ptr().add(i) = value;
            }
        }
        
        TensorBase {
            storage,
            shape: dim,
            strides,
            offset: 0,
        }
    }
}
```

---

### 3.4 arange/linspace/logspace — 序列生成

#### 3.4.1 完整签名

```rust
// src/construct.rs

impl Tensor<f64, Ix1> {
    /// 生成等差序列（半开区间）。
    ///
    /// # Arguments
    /// * `start` - 起始值（包含）
    /// * `end` - 结束值（不包含）
    /// * `step` - 步长
    ///
    /// # Examples
    /// ```
    /// let t = Tensor1::arange(0.0, 5.0, 1.0);
    /// assert_eq!(t.to_vec(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn arange(start: f64, end: f64, step: f64) -> Self {
        assert!(step != 0.0, "step cannot be zero");
        
        let n = ((end - start) / step).max(0.0).ceil() as usize;
        let mut data = Vec::with_capacity(n);
        
        let mut val = start;
        if step > 0.0 {
            while val < end {
                data.push(val);
                val += step;
            }
        } else {
            while val > end {
                data.push(val);
                val += step;
            }
        }
        
        Tensor::from_vec(data)
    }
    
    /// 生成整数等差序列。
    ///
    /// # Examples
    /// ```
    /// let t = Tensor1::arange_int(0, 5);  // [0, 1, 2, 3, 4]
    /// ```
    pub fn arange_int(start: i64, end: i64) -> Tensor<i64, Ix1> {
        let step = if end > start { 1i64 } else { -1i64 };
        let n = (end - start).abs() as usize;
        let data: Vec<i64> = (0..n).map(|i| start + i as i64 * step).collect();
        Tensor::from_vec(data)
    }
    
    /// 生成等间隔序列（指定数量）。
    ///
    /// # Arguments
    /// * `start` - 起始值（包含）
    /// * `end` - 结束值（包含）
    /// * `num` - 元素数量
    ///
    /// # Examples
    /// ```
    /// let t = Tensor1::linspace(0.0, 1.0, 5);
    /// assert_eq!(t.to_vec(), vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    /// ```
    pub fn linspace(start: f64, end: f64, num: usize) -> Self {
        if num == 0 {
            return Tensor::from_vec(Vec::new());
        }
        if num == 1 {
            return Tensor::from_vec(vec![start]);
        }
        
        let step = (end - start) / (num - 1) as f64;
        let data: Vec<f64> = (0..num)
            .map(|i| start + i as f64 * step)
            .collect();
        
        Tensor::from_vec(data)
    }
    
    /// 生成对数间隔序列。
    ///
    /// # Arguments
    /// * `start` - 起始指数（base^start）
    /// * `end` - 结束指数（base^end）
    /// * `num` - 元素数量
    /// * `base` - 底数（默认 10.0）
    ///
    /// # Examples
    /// ```
    /// let t = Tensor1::logspace(0.0, 2.0, 3, 10.0);
    /// // [10^0, 10^1, 10^2] = [1.0, 10.0, 100.0]
    /// assert_eq!(t.to_vec(), vec![1.0, 10.0, 100.0]);
    /// ```
    pub fn logspace(start: f64, end: f64, num: usize, base: f64) -> Self {
        if num == 0 {
            return Tensor::from_vec(Vec::new());
        }
        if num == 1 {
            return Tensor::from_vec(vec![base.powf(start)]);
        }
        
        let step = (end - start) / (num - 1) as f64;
        let data: Vec<f64> = (0..num)
            .map(|i| base.powf(start + i as f64 * step))
            .collect();
        
        Tensor::from_vec(data)
    }
    
    /// 生成对数间隔序列（默认底数 10）。
    pub fn logspace_base10(start: f64, end: f64, num: usize) -> Self {
        Self::logspace(start, end, num, 10.0)
    }
    
    /// 生成对数间隔序列（自然对数底 e）。
    pub fn logspace_e(start: f64, end: f64, num: usize) -> Self {
        Self::logspace(start, end, num, core::f64::consts::E)
    }
}
```

#### 3.4.2 序列生成语义总结

| 方法 | 区间 | 元素 | 公式 |
|------|------|------|------|
| `arange(start, end, step)` | [start, end) | 取决于步长 | `val_i = start + i * step` |
| `arange_int(start, end)` | [start, end) | end - start | `val_i = start + i` |
| `linspace(start, end, num)` | [start, end] | num | `val_i = start + i * (end-start)/(num-1)` |
| `logspace(start, end, num, base)` | [base^start, base^end] | num | `val_i = base^(start + i * (end-start)/(num-1))` |

---

## 4. 运算符重载设计

### 4.1 二元运算符 impl 组合表

| LHS 类型 | RHS 类型 | 运算符 | Output | 广播 | impl 位置 |
|----------|----------|--------|--------|------|-----------|
| `Tensor<A, D>` | `Tensor<A, D>` | `+` | `Tensor<A, D>` | ✓ | `arithmetic.rs` |
| `Tensor<A, D>` | `Tensor<A, D>` | `-` | `Tensor<A, D>` | ✓ | `arithmetic.rs` |
| `Tensor<A, D>` | `Tensor<A, D>` | `*` | `Tensor<A, D>` | ✓ | `arithmetic.rs` |
| `Tensor<A, D>` | `Tensor<A, D>` | `/` | `Tensor<A, D>` | ✓ | `arithmetic.rs` |
| `Tensor<A, D>` | `Tensor<A, D>` | `%` | `Tensor<A, D>` | ✓ | `arithmetic.rs` |
| `Tensor<A, D>` | `A` | `+` | `Tensor<A, D>` | 标量广播 | `arithmetic.rs` |
| `Tensor<A, D>` | `A` | `-` | `Tensor<A, D>` | 标量广播 | `arithmetic.rs` |
| `Tensor<A, D>` | `A` | `*` | `Tensor<A, D>` | 标量广播 | `arithmetic.rs` |
| `Tensor<A, D>` | `A` | `/` | `Tensor<A, D>` | 标量广播 | `arithmetic.rs` |
| `A` | `Tensor<A, D>` | `+` | `Tensor<A, D>` | 标量广播 | `arithmetic.rs` |
| `A` | `Tensor<A, D>` | `-` | `Tensor<A, D>` | 标量广播 | `arithmetic.rs` |
| `A` | `Tensor<A, D>` | `*` | `Tensor<A, D>` | 标量广播 | `arithmetic.rs` |
| `A` | `Tensor<A, D>` | `/` | `Tensor<A, D>` | 标量广播 | `arithmetic.rs` |
| `&Tensor<A, D>` | `&Tensor<A, D>` | `+` | `Tensor<A, D>` | ✓ | `arithmetic.rs` |
| `&Tensor<A, D>` | `Tensor<A, D>` | `+` | `Tensor<A, D>` | ✓ | `arithmetic.rs` |
| `Tensor<A, D>` | `&Tensor<A, D>` | `+` | `Tensor<A, D>` | ✓ | `arithmetic.rs` |

**约束**: 所有算术运算符要求 `A: Numeric`（排除 `bool`）。

### 4.2 复合赋值运算符 impl 组合表

| LHS 类型 | RHS 类型 | 运算符 | 广播 | 约束 |
|----------|----------|--------|------|------|
| `Tensor<A, D>` | `Tensor<A, D>` | `+=` | ✓ (仅 RHS) | `A: Numeric` |
| `Tensor<A, D>` | `Tensor<A, D>` | `-=` | ✓ (仅 RHS) | `A: Numeric` |
| `Tensor<A, D>` | `Tensor<A, D>` | `*=` | ✓ (仅 RHS) | `A: Numeric` |
| `Tensor<A, D>` | `Tensor<A, D>` | `/=` | ✓ (仅 RHS) | `A: Numeric` |
| `Tensor<A, D>` | `Tensor<A, D>` | `%=` | ✓ (仅 RHS) | `A: Numeric` |
| `Tensor<A, D>` | `A` | `+=` | 标量广播 | `A: Numeric` |
| `Tensor<A, D>` | `A` | `-=` | 标量广播 | `A: Numeric` |
| `Tensor<A, D>` | `A` | `*=` | 标量广播 | `A: Numeric` |
| `Tensor<A, D>` | `A` | `/=` | 标量广播 | `A: Numeric` |
| `Tensor<A, D>` | `&Tensor<A, D>` | `+=` | ✓ (仅 RHS) | `A: Numeric` |

**关键语义**: 复合赋值仅支持 RHS 广播，LHS 必须拥有完整存储。

### 4.3 一元运算符 impl 组合表

| 运算符 | Trait | 类型约束 | 适用类型 | 说明 |
|--------|-------|----------|----------|------|
| `-` | `Neg` | `A: Numeric` | 整数、浮点、复数 | 算术取负 |
| `!` | `Not` | `A: Element + Not<Output=A>` | `bool`、整数 | 逻辑/位取反 |

**Neg 实现签名**:

```rust
// src/arithmetic.rs

impl<A, D> Neg for Tensor<A, D>
where
    A: Numeric + Neg<Output = A>,
    D: Dimension,
{
    type Output = Tensor<A, D>;
    
    fn neg(self) -> Self::Output {
        self.mapv(|x| -x)
    }
}

impl<A, D> Neg for &Tensor<A, D>
where
    A: Numeric + Neg<Output = A>,
    D: Dimension,
{
    type Output = Tensor<A, D>;
    
    fn neg(self) -> Self::Output {
        self.mapv(|x| -x)
    }
}
```

**Not 实现签名**:

```rust
impl<A, D> Not for Tensor<A, D>
where
    A: Element + Not<Output = A>,
    D: Dimension,
{
    type Output = Tensor<A, D>;
    
    fn not(self) -> Self::Output {
        self.mapv(|x| !x)
    }
}
```

### 4.4 PartialEq 实现

```rust
// src/convert.rs

impl<A, D> PartialEq for Tensor<A, D>
where
    A: Element + PartialEq,
    D: Dimension,
{
    fn eq(&self, other: &Self) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}
```

**注意**: 对于浮点类型，`PartialEq` 使用精确比较。近似比较应使用 `is_close()` 方法。

---

## 5. 实用操作设计

### 5.1 copy_to/fill — 批量操作

```rust
// src/convert.rs

impl<S, D, A> TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    D: Dimension,
    A: Element,
{
    /// 复制数据到目标张量。
    ///
    /// # Panics
    /// 如果形状不匹配，panic。
    ///
    /// # Examples
    /// ```
    /// let src = Tensor1::from_vec(vec![1.0, 2.0, 3.0]);
    /// let mut dst = Tensor1::zeros([3]);
    /// src.copy_to(&mut dst);
    /// assert_eq!(dst.to_vec(), vec![1.0, 2.0, 3.0]);
    /// ```
    pub fn copy_to<St>(&self, dest: &mut TensorBase<St, D>)
    where
        St: StorageMut<Elem = A>,
        A: Clone,
    {
        assert_eq!(self.shape(), dest.shape(), "shape mismatch in copy_to");
        for (src, dst) in self.iter().zip(dest.iter_mut()) {
            *dst = src.clone();
        }
    }
    
    /// 用指定值填充张量。
    ///
    /// # Examples
    /// ```
    /// let mut t = Tensor1::<f64>::zeros([5]);
    /// t.fill(3.14);
    /// assert!(t.iter().all(|&x| x == 3.14));
    /// ```
    pub fn fill(&mut self, value: A)
    where
        A: Clone,
    {
        for elem in self.iter_mut() {
            *elem = value.clone();
        }
    }
}
```

### 5.2 is_close/allclose — 近似比较

```rust
// src/convert.rs

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: RealScalar,
{
    /// 逐元素近似相等比较。
    ///
    /// 检查 `|self - other| <= atol + rtol * |other|`
    ///
    /// # Arguments
    /// * `other` - 比较目标
    /// * `rtol` - 相对容差
    /// * `atol` - 绝对容差
    ///
    /// # Returns
    /// 布尔张量，逐元素比较结果
    ///
    /// # Examples
    /// ```
    /// let a = Tensor1::from_vec(vec![1.0, 2.0, 3.0]);
    /// let b = Tensor1::from_vec(vec![1.00001, 2.0, 3.0001]);
    /// let close = a.is_close(&b, 1e-5, 1e-8);
    /// // [true, true, false]
    /// ```
    pub fn is_close<St, E>(
        &self,
        other: &TensorBase<St, E>,
        rtol: A,
        atol: A,
    ) -> Tensor<bool, <D as BroadcastDim<E>>::Output>
    where
        St: Storage<Elem = A>,
        E: Dimension,
        D: BroadcastDim<E>,
    {
        // 广播后逐元素比较
        let output_shape = self.shape().broadcast_shape(other.shape()).unwrap();
        let a_broadcast = self.broadcast(output_shape.clone()).unwrap();
        let b_broadcast = other.broadcast(output_shape.clone()).unwrap();
        
        let mut result = Tensor::zeros(output_shape);
        for (r, &a_val, &b_val) in izip!(result.iter_mut(), a_broadcast.iter(), b_broadcast.iter()) {
            let diff = (a_val - b_val).abs();
            let tol = atol + rtol * b_val.abs();
            *r = diff <= tol;
        }
        result
    }
    
    /// 全元素近似相等检查。
    ///
    /// # Examples
    /// ```
    /// let a = Tensor1::from_vec(vec![1.0, 2.0, 3.0]);
    /// let b = Tensor1::from_vec(vec![1.00001, 2.0, 3.0]);
    /// assert!(a.allclose(&b, 1e-4, 1e-8));
    /// ```
    pub fn allclose<St, E>(
        &self,
        other: &TensorBase<St, E>,
        rtol: A,
        atol: A,
    ) -> bool
    where
        St: Storage<Elem = A>,
        E: Dimension,
        D: BroadcastDim<E>,
    {
        self.is_close(other, rtol, atol).iter().all(|&x| x)
    }
}
```

### 5.3 clip — 裁剪操作

```rust
// src/convert.rs

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: RealScalar,
{
    /// 裁剪到指定范围。
    ///
    /// # Arguments
    /// * `min` - 最小值
    /// * `max` - 最大值
    ///
    /// # Returns
    /// 新张量，元素限制在 [min, max] 范围内
    ///
    /// # Examples
    /// ```
    /// let t = Tensor1::from_vec(vec![-1.0, 0.5, 1.0, 2.0, 3.0]);
    /// let clipped = t.clip(0.0, 2.0);
    /// assert_eq!(clipped.to_vec(), vec![0.0, 0.5, 1.0, 2.0, 2.0]);
    /// ```
    pub fn clip(&self, min: A, max: A) -> Tensor<A, D> {
        self.mapv(|x| x.max(min).min(max))
    }
    
    /// clip 的别名（与 std::clamp 一致）。
    pub fn clamp(&self, min: A, max: A) -> Tensor<A, D> {
        self.clip(min, max)
    }
    
    /// 原地裁剪。
    pub fn clip_inplace(&mut self, min: A, max: A)
    where
        S: StorageMut<Elem = A>,
    {
        self.mapv_inplace(|x| x.max(min).min(max));
    }
}
```

### 5.4 flip/flipud/fliplr — 翻转操作

```rust
// src/convert.rs

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element,
{
    /// 沿指定轴翻转。
    ///
    /// # Arguments
    /// * `axis` - 翻转的轴
    ///
    /// # Returns
    /// 视图（零拷贝）
    ///
    /// # Examples
    /// ```
    /// let t = Tensor1::from_vec(vec![1, 2, 3, 4, 5]);
    /// let flipped = t.flip(Axis(0));
    /// assert_eq!(flipped.to_vec(), vec![5, 4, 3, 2, 1]);
    /// ```
    pub fn flip(&self, axis: Axis) -> TensorView<'_, A, D> {
        let mut strides = self.strides().to_vec();
        strides[axis.0] = -strides[axis.0];
        
        // 构造翻转视图
        unsafe {
            TensorView::from_raw_parts(
                self.as_ptr(),
                self.shape().clone(),
                D::from_slice(&strides),
                self.offset(),
            )
        }
    }
    
    /// 沿所有轴翻转。
    pub fn flip_all(&self) -> TensorView<'_, A, D> {
        let mut strides = self.strides().to_vec();
        for s in &mut strides {
            *s = -*s;
        }
        
        unsafe {
            TensorView::from_raw_parts(
                self.as_ptr(),
                self.shape().clone(),
                D::from_slice(&strides),
                self.offset(),
            )
        }
    }
}

impl<S, A> TensorBase<S, Ix2>
where
    S: Storage<Elem = A>,
    A: Element,
{
    /// 上下翻转（沿轴 0 翻转）。
    ///
    /// # Examples
    /// ```
    /// let t = Tensor2::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6])?;
    /// let flipped = t.flipud();
    /// // [[4, 5, 6],
    /// //  [1, 2, 3]]
    /// ```
    pub fn flipud(&self) -> TensorView<'_, A, Ix2> {
        self.flip(Axis(0))
    }
    
    /// 左右翻转（沿轴 1 翻转）。
    ///
    /// # Examples
    /// ```
    /// let t = Tensor2::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6])?;
    /// let flipped = t.fliplr();
    /// // [[3, 2, 1],
    /// //  [6, 5, 4]]
    /// ```
    pub fn fliplr(&self) -> TensorView<'_, A, Ix2> {
        self.flip(Axis(1))
    }
}
```

### 5.5 to_owned/into_owned — 所有权转换

```rust
// src/convert.rs

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element,
{
    /// 克隆数据到新的拥有型张量。
    ///
    /// 总是分配新内存并复制数据，即使输入已是 Owned。
    ///
    /// # Examples
    /// ```
    /// let view: TensorView<f64, Ix1> = tensor.view();
    /// let owned = view.to_owned();
    /// ```
    pub fn to_owned(&self) -> Tensor<A, D>
    where
        A: Clone,
    {
        let mut result = Tensor::zeros_order(self.shape().clone(), self.memory_order());
        self.copy_to(&mut result);
        result
    }
}

impl<S, D, A> TensorBase<S, D>
where
    S: StorageIntoOwned<Elem = A>,
    D: Dimension,
    A: Element,
{
    /// 消费张量，转换为拥有型。
    ///
    /// - 对于 `Tensor`：直接返回，无复制
    /// - 对于 `TensorView`/`TensorViewMut`：复制数据到新分配
    /// - 对于 `ArcTensor`：若引用计数为 1 则直接返回，否则复制
    ///
    /// # Examples
    /// ```
    /// let view: TensorView<f64, Ix1> = tensor.view();
    /// let owned: Tensor<f64, Ix1> = view.into_owned();
    /// ```
    pub fn into_owned(self) -> Tensor<A, D> {
        // 由 StorageIntoOwned trait 实现
        // ...
    }
}
```

### 5.6 map/mapv/mapv_inplace — 映射操作

```rust
// src/convert.rs

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element,
{
    /// 逐元素映射（通过引用）。
    ///
    /// # Examples
    /// ```
    /// let a = Tensor1::from_vec(vec![1.0, 2.0, 3.0]);
    /// let b = a.map(|x| x.to_string());
    /// assert_eq!(b.to_vec(), vec!["1", "2", "3"]);
    /// ```
    pub fn map<B, F>(&self, f: F) -> Tensor<B, D>
    where
        B: Element,
        F: FnMut(&A) -> B,
    {
        let mut result = Tensor::zeros(self.shape().clone());
        for (src, dst) in self.iter().zip(result.iter_mut()) {
            *dst = f(src);
        }
        result
    }
    
    /// 逐元素映射（通过值）。
    ///
    /// # Examples
    /// ```
    /// let a = Tensor1::from_vec(vec![1.0, 2.0, 3.0]);
    /// let b = a.mapv(|x| x * 2.0);
    /// assert_eq!(b.to_vec(), vec![2.0, 4.0, 6.0]);
    /// ```
    pub fn mapv<B, F>(&self, mut f: F) -> Tensor<B, D>
    where
        B: Element,
        F: FnMut(A) -> B,
    {
        let mut result = Tensor::zeros(self.shape().clone());
        for (src, dst) in self.iter().zip(result.iter_mut()) {
            *dst = f(*src);
        }
        result
    }
}

impl<S, D, A> TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    D: Dimension,
    A: Element,
{
    /// 原地逐元素映射。
    ///
    /// # Examples
    /// ```
    /// let mut a = Tensor1::from_vec(vec![1.0, 2.0, 3.0]);
    /// a.mapv_inplace(|x| x * x);
    /// assert_eq!(a.to_vec(), vec![1.0, 4.0, 9.0]);
    /// ```
    pub fn mapv_inplace<F>(&mut self, mut f: F)
    where
        F: FnMut(A) -> A,
    {
        for elem in self.iter_mut() {
            *elem = f(*elem);
        }
    }
}
```

---

## 6. 连续性保证方法

### 6.1 完整签名

```rust
// src/convert.rs

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + Clone,
{
    /// 转换为 F-contiguous 布局的新张量。
    ///
    /// 如果输入已 F-contiguous，等价于 `to_owned()`。
    /// 否则，分配新内存并复制数据。
    ///
    /// # Returns
    /// 始终返回 `Tensor<A, D>`（Owned 类型）
    ///
    /// # Examples
    /// ```
    /// let t = Tensor2::from_shape_vec_order([2, 3], vec![1, 2, 3, 4, 5, 6], MemoryOrder::C)?;
    /// assert!(t.is_c_contiguous());
    /// assert!(!t.is_f_contiguous());
    /// 
    /// let f_contig = t.to_f_contiguous();
    /// assert!(f_contig.is_f_contiguous());
    /// ```
    pub fn to_f_contiguous(&self) -> Tensor<A, D> {
        if self.is_f_contiguous() {
            return self.to_owned();
        }
        
        let dim = self.shape().clone();
        let strides = dim.strides_for_f_order();
        
        let mut result = unsafe {
            TensorBase {
                storage: Owned::allocate_aligned(dim.size(), Owned::<A>::DEFAULT_ALIGNMENT),
                shape: dim,
                strides,
                offset: 0,
            }
        };
        
        // 复制数据（按 F-order 遍历）
        for idx in self.index_iter() {
            result[&idx] = self[&idx].clone();
        }
        
        result
    }
    
    /// 转换为 C-contiguous 布局的新张量。
    ///
    /// 如果输入已 C-contiguous，等价于 `to_owned()`。
    /// 否则，分配新内存并复制数据。
    ///
    /// # Examples
    /// ```
    /// let t = Tensor2::from_shape_vec_order([2, 3], vec![1, 2, 3, 4, 5, 6], MemoryOrder::F)?;
    /// assert!(t.is_f_contiguous());
    /// 
    /// let c_contig = t.to_c_contiguous();
    /// assert!(c_contig.is_c_contiguous());
    /// ```
    pub fn to_c_contiguous(&self) -> Tensor<A, D> {
        if self.is_c_contiguous() {
            return self.to_owned();
        }
        
        let dim = self.shape().clone();
        let strides = dim.strides_for_c_order();
        
        let mut result = unsafe {
            TensorBase {
                storage: Owned::allocate_aligned(dim.size(), Owned::<A>::DEFAULT_ALIGNMENT),
                shape: dim,
                strides,
                offset: 0,
            }
        };
        
        // 复制数据（按 C-order 遍历）
        for idx in self.index_iter_c_order() {
            result[&idx] = self[&idx].clone();
        }
        
        result
    }
    
    /// 转换为连续布局的新张量。
    ///
    /// 优先选择 F-contiguous。如果输入已 C-contiguous（但非 F-contiguous），
    /// 则保持 C-contiguous。
    ///
    /// # 规则
    /// - F-contiguous 输入 → F-contiguous 输出
    /// - C-contiguous 输入 → C-contiguous 输出
    /// - 非连续输入 → F-contiguous 输出
    ///
    /// # Examples
    /// ```
    /// let t = tensor.slice(s![.., ..;2]);  // 非连续切片
    /// let contig = t.to_contiguous();       // F-contiguous
    /// ```
    pub fn to_contiguous(&self) -> Tensor<A, D> {
        if self.is_f_contiguous() {
            self.to_owned()
        } else if self.is_c_contiguous() {
            self.to_c_contiguous()
        } else {
            self.to_f_contiguous()
        }
    }
}
```

### 6.2 设计理由

**为什么不使用 Cow 语义？**

| 考量 | Cow 语义 | 统一返回 Owned |
|------|----------|----------------|
| API 简洁性 | 复杂（返回类型不确定） | 简单（始终 `Tensor<A, D>`） |
| 生命周期 | 需要绑定到输入 | 无生命周期约束 |
| 性能优化 | 调用方难以预测 | 调用方可先检查 `is_contiguous()` |
| 与 ndarray 一致 | ndarray 使用 Owned | 一致 |

**调用方优化模式**：

```rust
// 优化模式：先检查再决定
if tensor.is_f_contiguous() {
    // 已连续，直接使用
    process(&tensor);
} else {
    // 需要连续，先转换
    let contiguous = tensor.to_f_contiguous();
    process(&contiguous);
}
```

---

## 7. cast 设计

### 7.1 完整签名

```rust
// src/convert.rs

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element,
{
    /// 显式类型转换。
    ///
    /// # Type Parameters
    /// * `B` - 目标元素类型
    ///
    /// # Returns
    /// 新张量，元素类型为 `B`
    ///
    /// # Examples
    /// ```
    /// let a = Tensor1::from_vec(vec![1.5, 2.7, 3.9]);
    /// let b: Tensor1<i32> = a.cast();
    /// assert_eq!(b.to_vec(), vec![1, 2, 3]);  // truncate
    /// ```
    pub fn cast<B>(&self) -> Tensor<B, D>
    where
        B: Element,
        A: CastTo<B>,
    {
        self.mapv(|x| x.cast_to())
    }
}

/// 类型转换 trait。
pub trait CastTo<T> {
    /// 执行类型转换。
    fn cast_to(self) -> T;
}
```

### 7.2 cast 精度行为规则表

| 源类型 | 目标类型 | 转换行为 | 示例 |
|--------|----------|----------|------|
| `f64` | `f32` | round-to-nearest-even | `1.23456789_f64 → 1.234568_f32` |
| `f32` | `f64` | 精确（无精度损失） | `1.5_f32 → 1.5_f64` |
| `f32/f64` | 整数 | truncate + saturating | `1.9 → 1`, `-0.1 → 0` |
| `f32/f64` (NaN) | 整数 | 返回 0 | `f64::NAN → 0i32` |
| `f32/f64` (+Inf) | 整数 | 饱和到 MAX | `f64::INFINITY → i32::MAX` |
| `f32/f64` (-Inf) | 整数 | 饱和到 MIN | `f64::NEG_INFINITY → i32::MIN` |
| 整数 | `f32/f64` | round-to-nearest-even | `123456789i32 → f32 可能不精确` |
| 整数 | 整数（窄化） | saturating | `300u16 → u8::MAX (255)` |
| 整数 | 整数（扩展） | 零扩展/符号扩展 | `u8 → u16` 零扩展 |
| `bool` | 数值 | `true → 1`, `false → 0` | `true → 1i32` |
| 数值 | `bool` | 非零 → `true`, 零 → `false` | `0 → false`, `1 → true` |
| `f32/f64` | `Complex<f32/f64>` | 虚部为 0 | `1.5 → Complex { re: 1.5, im: 0.0 }` |
| `Complex<T>` | `T` | **不允许**（编译错误） | 须显式取 `.re()` |

### 7.3 CastTo trait 实现

```rust
// src/convert.rs

// === 浮点 → 浮点 ===

impl CastTo<f32> for f64 {
    #[inline]
    fn cast_to(self) -> f32 {
        self as f32  // IEEE 754 round-to-nearest-even
    }
}

impl CastTo<f64> for f32 {
    #[inline]
    fn cast_to(self) -> f64 {
        self as f64  // 精确转换
    }
}

// === 浮点 → 整数（truncate + saturating）===

impl CastTo<i32> for f64 {
    #[inline]
    fn cast_to(self) -> i32 {
        if self.is_nan() {
            return 0;
        }
        if self >= i32::MAX as f64 {
            return i32::MAX;
        }
        if self <= i32::MIN as f64 {
            return i32::MIN;
        }
        self.trunc() as i32
    }
}

impl CastTo<u32> for f64 {
    #[inline]
    fn cast_to(self) -> u32 {
        if self.is_nan() || self <= 0.0 {
            return 0;
        }
        if self >= u32::MAX as f64 {
            return u32::MAX;
        }
        self.trunc() as u32
    }
}

// === 整数 → 浮点 ===

impl CastTo<f64> for i32 {
    #[inline]
    fn cast_to(self) -> f64 {
        self as f64  // i32 范围内精确
    }
}

impl CastTo<f32> for i64 {
    #[inline]
    fn cast_to(self) -> f32 {
        self as f32  // 可能损失精度（round-to-nearest-even）
    }
}

// === 整数 → 整数（saturating）===

impl CastTo<u8> for i32 {
    #[inline]
    fn cast_to(self) -> u8 {
        self.clamp(0, u8::MAX as i32) as u8
    }
}

impl CastTo<i8> for i32 {
    #[inline]
    fn cast_to(self) -> i8 {
        self.clamp(i8::MIN as i32, i8::MAX as i32) as i8
    }
}

// === bool ↔ 数值 ===

impl CastTo<i32> for bool {
    #[inline]
    fn cast_to(self) -> i32 {
        self as i32
    }
}

impl CastTo<bool> for i32 {
    #[inline]
    fn cast_to(self) -> bool {
        self != 0
    }
}

// === 实数 → 复数 ===

impl CastTo<Complex<f64>> for f64 {
    #[inline]
    fn cast_to(self) -> Complex<f64> {
        Complex { re: self, im: 0.0 }
    }
}

// === 复数 → 实数（不允许隐式）===
// 注意：不实现 CastTo<f64> for Complex<f64>
// 用户必须显式使用 .re() 或 .im()
```

### 7.4 NaN/Inf 转换行为详表

| 输入值 | 目标类型 | 输出值 | 说明 |
|--------|----------|--------|------|
| `f64::NAN` | `i32` | `0` | NaN → 整数 = 0 |
| `f64::NAN` | `f32` | `f32::NAN` | NaN 传播 |
| `f64::INFINITY` | `i32` | `i32::MAX` | +Inf 饱和到 MAX |
| `f64::NEG_INFINITY` | `i32` | `i32::MIN` | -Inf 饱和到 MIN |
| `f64::INFINITY` | `f32` | `f32::INFINITY` | Inf 保持 |
| `f32::NAN` | `bool` | `true` | 非零 = true |
| `0.0` | `bool` | `false` | 零 = false |

---

## 8. From trait 实现

### 8.1 Vec/切片/数组 → Tensor

```rust
// src/convert.rs

// === Vec → Tensor1 ===

impl<A> From<Vec<A>> for Tensor<A, Ix1>
where
    A: Element,
{
    fn from(vec: Vec<A>) -> Self {
        Self::from_vec(vec)
    }
}

// === &[A] → Tensor1 ===

impl<A> From<&[A]> for Tensor<A, Ix1>
where
    A: Element + Clone,
{
    fn from(slice: &[A]) -> Self {
        Self::from_slice(slice)
    }
}

// === [A; N] → Tensor1 ===

impl<A, const N: usize> From<[A; N]> for Tensor<A, Ix1>
where
    A: Element,
{
    fn from(arr: [A; N]) -> Self {
        Self::from_vec(arr.into_iter().collect())
    }
}

// === &Tensor → TensorView ===

impl<'a, A, D> From<&'a Tensor<A, D>> for TensorView<'a, A, D>
where
    A: Element,
    D: Dimension,
{
    fn from(tensor: &'a Tensor<A, D>) -> Self {
        tensor.view()
    }
}

// === &mut Tensor → TensorViewMut ===

impl<'a, A, D> From<&'a mut Tensor<A, D>> for TensorViewMut<'a, A, D>
where
    A: Element,
    D: Dimension,
{
    fn from(tensor: &'a mut Tensor<A, D>) -> Self {
        tensor.view_mut()
    }
}

// === IxN → IxDyn（静态维度转动态维度）===

impl<D> From<D> for IxDyn
where
    D: Dimension,
{
    fn from(dim: D) -> Self {
        dim.into_dyn()
    }
}
```

### 8.2 From 实现总结表

| 源类型 | 目标类型 | 行为 | 分配 |
|--------|----------|------|------|
| `Vec<A>` | `Tensor<A, Ix1>` | 从 Vec 构造 | 转移所有权 |
| `&[A]` | `Tensor<A, Ix1>` | 从切片构造 | 复制 |
| `[A; N]` | `Tensor<A, Ix1>` | 从数组构造 | 复制 |
| `&Tensor<A, D>` | `TensorView<'a, A, D>` | 创建视图 | 无（借用） |
| `&mut Tensor<A, D>` | `TensorViewMut<'a, A, D>` | 创建可变视图 | 无（借用） |
| `Ix0-Ix6` | `IxDyn` | 静态→动态 | 可能（IxDyn 使用 Vec） |
| `Tensor<A, IxN>` | `Tensor<A, IxDyn>` | 维度转换 | 无（元数据转换） |

---

## 9. 格式化输出设计

### 9.1 Debug/Display 实现签名

```rust
// src/format.rs

use core::fmt::{self, Debug, Display, Formatter};

impl<S, D, A> Debug for TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Debug + Element,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.ndim() == 0 {
            // 标量
            write!(f, "Tensor({:?})", self[&[]])
        } else if self.ndim() == 1 {
            // 一维
            fmt_1d(f, self, "Tensor1")
        } else {
            // 多维
            fmt_nd(f, self, "Tensor")
        }
    }
}

impl<S, D, A> Display for TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Display + Element,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.ndim() == 0 {
            write!(f, "{}", self[&[]])
        } else if self.ndim() == 1 {
            fmt_1d_display(f, self)
        } else {
            fmt_nd_display(f, self)
        }
    }
}
```

### 9.2 大数组省略规则

```rust
// src/format.rs

/// 格式化输出配置。
pub struct FormatConfig {
    /// 边缘元素数量（每边）。
    /// 超过此数量的数组将在中间省略。
    pub edge_items: usize,
    
    /// 触发省略的最小元素总数。
    /// 元素数少于此值时，显示全部元素。
    pub threshold: usize,
    
    /// 行省略阈值。
    /// 行数超过此值时省略中间行。
    pub line_width: usize,
}

impl Default for FormatConfig {
    fn default() -> Self {
        Self {
            edge_items: 3,       // 每边显示 3 个元素
            threshold: 1000,     // 超过 1000 个元素触发省略
            line_width: 80,      // 行宽 80 字符
        }
    }
}

/// NumPy 风格的格式化输出。
fn fmt_nd<S, D, A>(f: &mut Formatter<'_>, tensor: &TensorBase<S, D>, name: &str) -> fmt::Result
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Debug + Element,
{
    let config = FormatConfig::default();
    let total = tensor.len();
    
    writeln!(f, "{}(shape={:?}, dtype={})", name, tensor.shape(), type_name::<A>())?;
    
    if total > config.threshold {
        // 大数组：省略显示
        writeln!(f, "[")?;
        fmt_large_array(f, tensor, &config, 0)?;
        write!(f, "]")
    } else {
        // 小数组：完整显示
        writeln!(f, "[")?;
        fmt_full_array(f, tensor, 0)?;
        write!(f, "]")
    }
}

/// 大数组省略格式化。
fn fmt_large_array<S, D, A>(
    f: &mut Formatter<'_>,
    tensor: &TensorBase<S, D>,
    config: &FormatConfig,
    indent: usize,
) -> fmt::Result
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Debug + Element,
{
    let shape = tensor.shape();
    let ndim = tensor.ndim();
    
    if ndim == 1 {
        // 一维：显示边缘元素
        let len = shape[0];
        let edge = config.edge_items.min(len / 2);
        
        write!(f, "{}", " ".repeat(indent))?;
        
        // 前边缘
        for i in 0..edge {
            write!(f, "{:?}, ", tensor[[i]])?;
        }
        
        if len > 2 * edge {
            write!(f, "..., ")?;
        }
        
        // 后边缘
        for i in (len - edge)..len {
            write!(f, "{:?}", tensor[[i]])?;
            if i < len - 1 {
                write!(f, ", ")?;
            }
        }
        
        writeln!(f)?;
    } else {
        // 多维：递归处理
        let first_dim = shape[0];
        let edge = config.edge_items.min(first_dim / 2);
        
        // 前边缘块
        for i in 0..edge {
            write!(f, "{}", " ".repeat(indent))?;
            writeln!(f, "[")?;
            let sub = tensor.index_axis(Axis(0), i);
            fmt_large_array(f, &sub, config, indent + 2)?;
            write!(f, "{}", " ".repeat(indent))?;
            writeln!(f, "],")?;
        }
        
        if first_dim > 2 * edge {
            write!(f, "{}", " ".repeat(indent))?;
            writeln!(f, "...")?;
        }
        
        // 后边缘块
        for i in (first_dim - edge)..first_dim {
            write!(f, "{}", " ".repeat(indent))?;
            writeln!(f, "[")?;
            let sub = tensor.index_axis(Axis(0), i);
            fmt_large_array(f, &sub, config, indent + 2)?;
            write!(f, "{}", " ".repeat(indent))?;
            if i < first_dim - 1 {
                writeln!(f, "],")?;
            } else {
                writeln!(f, "]")?;
            }
        }
    }
    
    Ok(())
}
```

### 9.3 格式化示例

**小数组（完整显示）**:

```
Tensor(shape=[2, 3], dtype=f64):
[[1.0, 2.0, 3.0],
 [4.0, 5.0, 6.0]]
```

**大数组（省略显示）**:

```
Tensor(shape=[100, 100], dtype=f64):
[[1.0, 2.0, 3.0, ..., 98.0, 99.0, 100.0],
 [101.0, 102.0, 103.0, ..., 198.0, 199.0, 200.0],
 [201.0, 202.0, 203.0, ..., 298.0, 299.0, 300.0],
 ...,
 [9701.0, 9702.0, 9703.0, ..., 9798.0, 9799.0, 9800.0],
 [9801.0, 9802.0, 9803.0, ..., 9898.0, 9899.0, 9900.0],
 [9901.0, 9902.0, 9903.0, ..., 9998.0, 9999.0, 10000.0]]
```

### 9.4 省略规则阈值表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `edge_items` | 3 | 每边显示的元素/行数 |
| `threshold` | 1000 | 触发省略的最小元素数 |
| `line_width` | 80 | 一行最大字符数（用于换行） |

---

## 10. 与其他模块的交互

### 10.1 模块依赖图

```
┌─────────────────────────────────────────────────────────────┐
│                    construct / convert / format             │
│  (张量构造、类型转换、格式化输出)                             │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│    tensor     │   │   element     │   │    layout     │
│ TensorBase    │   │ Element trait │   │ MemoryOrder   │
│ Tensor alias  │   │ Numeric trait │   │ is_contiguous │
└───────────────┘   └───────────────┘   └───────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   storage     │   │   dimension   │   │    error      │
│ Owned<A>      │   │ Ix0-Ix6       │   │ ShapeError    │
│ Storage trait │   │ IxDyn         │   │ CastError     │
└───────────────┘   └───────────────┘   └───────────────┘
```

### 10.2 与 tensor 模块的交互

| 交互点 | 说明 |
|--------|------|
| `Tensor::zeros()` | 创建 `Tensor<A, D>` 类型 |
| `Tensor::from_vec()` | 从 `Vec` 构造 |
| `tensor.cast()` | 返回新 `Tensor<B, D>` |
| `tensor.to_owned()` | 从视图创建 Owned |

### 10.3 与 element 模块的交互

| 交互点 | 使用的 trait |
|--------|-------------|
| `zeros()` | `A: Default` |
| `ones()` | `A: One` |
| `full()` | `A: Clone` |
| `cast()` | `A: CastTo<B>` |
| 算术运算 | `A: Numeric` |
| `is_close()` | `A: RealScalar` |

### 10.4 与 layout 模块的交互

| 交互点 | 说明 |
|--------|------|
| `zeros_order(shape, order)` | 指定 F/C 布局 |
| `to_f_contiguous()` | 强制 F-contiguous |
| `to_c_contiguous()` | 强制 C-contiguous |
| `is_contiguous()` | 检查连续性 |

### 10.5 与 error 模块的交互

| 错误类型 | 触发场景 |
|----------|----------|
| `ShapeError::IncompatibleSize` | `from_shape_vec` 形状不匹配 |
| `CastError` | 类型转换失败（预留） |
| `DimensionMismatch` | 维度转换失败 |

---

## 11. 实现任务分解

### 任务 1：创建模块文件结构

**名称**：创建 construct.rs, convert.rs, format.rs

**涉及文件**：
- `src/construct.rs`
- `src/convert.rs`
- `src/format.rs`

**前置依赖**：tensor, storage, dimension, element, layout

**验收标准**：
- [ ] 文件创建完成
- [ ] 模块导出到 `src/lib.rs`
- [ ] `cargo check` 通过

---

### 任务 2：实现 zeros/ones/full/empty

**名称**：实现基础构造方法

**涉及文件**：
- `src/construct.rs`

**前置依赖**：任务 1

**验收标准**：
- [ ] `zeros()` / `zeros_order()` 实现
- [ ] `ones()` / `ones_order()` 实现
- [ ] `full()` / `full_order()` 实现
- [ ] `empty()` / `empty_order()` 实现（unsafe）
- [ ] 单元测试通过

---

### 任务 3：实现 eye/identity/diag

**名称**：实现矩阵构造方法

**涉及文件**：
- `src/construct.rs`

**前置依赖**：任务 2

**验收标准**：
- [ ] `eye(n)` 实现
- [ ] `identity(rows, cols)` 实现
- [ ] `diag(diagonal)` 实现
- [ ] `diag_with_offset()` 实现
- [ ] `extract_diag()` 实现
- [ ] 单元测试通过

---

### 任务 4：实现 from_vec/from_slice/from_fn

**名称**：实现从数据源构造

**涉及文件**：
- `src/construct.rs`

**前置依赖**：任务 3

**验收标准**：
- [ ] `from_vec()` 实现
- [ ] `from_shape_vec()` 实现
- [ ] `from_slice()` 实现
- [ ] `from_shape_slice()` 实现
- [ ] `from_fn()` 实现
- [ ] 单元测试通过

---

### 任务 5：实现 arange/linspace/logspace

**名称**：实现序列生成

**涉及文件**：
- `src/construct.rs`

**前置依赖**：任务 4

**验收标准**：
- [ ] `arange()` 实现
- [ ] `arange_int()` 实现
- [ ] `linspace()` 实现
- [ ] `logspace()` 实现
- [ ] `logspace_base10()` 实现
- [ ] 单元测试通过

---

### 任务 6：实现运算符重载

**名称**：实现算术运算符

**涉及文件**：
- `src/arithmetic.rs`（或在 convert.rs 中）

**前置依赖**：任务 5

**验收标准**：
- [ ] `Add/Sub/Mul/Div/Rem` for Tensor
- [ ] `Add/Sub/Mul/Div/Rem` for Tensor op Scalar
- [ ] `AddAssign/SubAssign/MulAssign/DivAssign/RemAssign`
- [ ] `Neg` / `Not` 一元运算
- [ ] `PartialEq` 实现
- [ ] 广播语义正确
- [ ] 单元测试通过

---

### 任务 7：实现 copy_to/fill/clip/flip

**名称**：实现实用操作

**涉及文件**：
- `src/convert.rs`

**前置依赖**：任务 6

**验收标准**：
- [ ] `copy_to()` 实现
- [ ] `fill()` 实现
- [ ] `clip()` / `clamp()` 实现
- [ ] `flip()` / `flipud()` / `fliplr()` 实现
- [ ] 单元测试通过

---

### 任务 8：实现 is_close/allclose

**名称**：实现近似比较

**涉及文件**：
- `src/convert.rs`

**前置依赖**：任务 7

**验收标准**：
- [ ] `is_close()` 实现
- [ ] `allclose()` 实现
- [ ] NaN 处理正确
- [ ] 单元测试通过

---

### 任务 9：实现 to_owned/into_owned

**名称**：实现所有权转换

**涉及文件**：
- `src/convert.rs`

**前置依赖**：任务 8

**验收标准**：
- [ ] `to_owned()` 实现
- [ ] `into_owned()` 实现
- [ ] `StorageIntoOwned` trait 定义
- [ ] 单元测试通过

---

### 任务 10：实现连续性保证方法

**名称**：实现 to_f_contiguous/to_c_contiguous/to_contiguous

**涉及文件**：
- `src/convert.rs`

**前置依赖**：任务 9

**验收标准**：
- [ ] `to_f_contiguous()` 实现
- [ ] `to_c_contiguous()` 实现
- [ ] `to_contiguous()` 实现
- [ ] 布局标志正确设置
- [ ] 单元测试通过

---

### 任务 11：实现 cast

**名称**：实现显式类型转换

**涉及文件**：
- `src/convert.rs`

**前置依赖**：任务 10

**验收标准**：
- [ ] `cast()` 方法实现
- [ ] `CastTo<T>` trait 定义
- [ ] 浮点→浮点转换
- [ ] 浮点→整数转换（truncate + saturating）
- [ ] 整数→浮点转换
- [ ] bool↔数值转换
- [ ] NaN/Inf 处理正确
- [ ] 单元测试通过

---

### 任务 12：实现 From trait

**名称**：实现标准类型转换

**涉及文件**：
- `src/convert.rs`

**前置依赖**：任务 11

**验收标准**：
- [ ] `From<Vec<A>>` 实现
- [ ] `From<&[A]>` 实现
- [ ] `From<[A; N]>` 实现
- [ ] `From<&Tensor> for TensorView` 实现
- [ ] `From<&mut Tensor> for TensorViewMut` 实现
- [ ] 单元测试通过

---

### 任务 13：实现 map/mapv/mapv_inplace

**名称**：实现映射操作

**涉及文件**：
- `src/convert.rs`

**前置依赖**：任务 12

**验收标准**：
- [ ] `map()` 实现
- [ ] `mapv()` 实现
- [ ] `mapv_inplace()` 实现
- [ ] 单元测试通过

---

### 任务 14：实现格式化输出

**名称**：实现 Debug/Display

**涉及文件**：
- `src/format.rs`

**前置依赖**：任务 13

**验收标准**：
- [ ] `Debug` trait 实现
- [ ] `Display` trait 实现
- [ ] 大数组省略规则实现
- [ ] NumPy 风格输出
- [ ] 单元测试通过

---

### 任务 15：集成测试和文档

**名称**：编写集成测试和文档

**涉及文件**：
- `tests/construct_tests.rs`
- `tests/convert_tests.rs`
- `tests/format_tests.rs`

**前置依赖**：任务 14

**验收标准**：
- [ ] 边界用例覆盖（空数组、单元素、大数组）
- [ ] NaN/Inf 边界测试
- [ ] 文档注释完整
- [ ] `cargo doc` 无警告

---

## 12. 设计决策记录

### 决策 1：zeros/ones/full 默认使用 F-order

**背景**：需要确定默认内存布局。

**选项**：
1. F-order（列优先）
2. C-order（行优先）

**选择**：F-order

**理由**：
- 与 BLAS/LAPACK 兼容（科学计算导向）
- 与需求文档一致（§1.2 核心原则）
- 提供 `*_order` 变体支持 C-order

---

### 决策 2：to_f_contiguous/to_c_contiguous 始终返回 Owned

**背景**：连续性转换是否使用 Cow 语义。

**选项**：
1. 返回 `Cow<Tensor<A, D>>`（借用或拥有）
2. 始终返回 `Tensor<A, D>`（Owned）

**选择**：始终返回 Owned

**理由**：
- API 简洁，无生命周期复杂度
- 调用方可通过 `is_contiguous()` 预检查
- 与 ndarray 行为一致
- 避免 Cow 引入的间接开销

---

### 决策 3：cast 的 NaN→整数 = 0

**背景**：NaN 转换为整数的行为。

**选项**：
1. 返回 0
2. panic
3. 返回 Result

**选择**：返回 0

**理由**：
- 与 Rust `as` 转换行为一致
- 与 NumPy `astype` 行为一致
- 避免 panic 影响性能
- 提供 `checked_cast` 变体用于检测（预留）

---

### 决策 4：复数不允许隐式转换为实数

**背景**：`Complex<T>` → `T` 是否自动实现。

**选项**：
1. 实现 `CastTo<T> for Complex<T>`（取实部）
2. 不实现，要求显式 `.re()` 调用

**选择**：不实现

**理由**：
- 复数到实数的转换有歧义（实部？模？）
- 显式调用 `.re()` 或 `.im()` 意图清晰
- 避免意外数据丢失
- 符合 Rust 显式优于隐式的哲学

---

### 决策 5：大数组省略阈值设为 1000

**背景**：格式化输出何时触发省略。

**选项**：
1. 阈值 100（NumPy 默认）
2. 阈值 1000
3. 可配置

**选择**：阈值 1000，可配置

**理由**：
- 1000 元素足以显示大多数调试信息
- 可通过 `FormatConfig` 自定义
- 与 NumPy `np.set_printoptions` 设计一致

---

### 决策 6：flip 返回视图而非 Owned

**背景**：翻转操作是否复制数据。

**选项**：
1. 返回 `Tensor<A, D>`（复制数据）
2. 返回 `TensorView<'a, A, D>`（零拷贝）

**选择**：返回视图

**理由**：
- 翻转通过负步长实现，零拷贝
- 用户可通过 `.to_owned()` 转换为 Owned
- 与切片、转置操作一致
- 性能最优

---

### 决策 7：运算符重载支持引用组合

**背景**：`&Tensor + &Tensor` vs `Tensor + Tensor`。

**选项**：
1. 仅支持 `Tensor + Tensor`
2. 支持所有引用组合

**选择**：支持所有引用组合

**理由**：
- 避免不必要的所有权转移
- 灵活性高
- 与 ndarray 行为一致
- 实现成本低（泛型）

---

## 附录 A：API 速查表

### A.1 构造方法

| 方法 | 签名 | 返回 |
|------|------|------|
| `zeros` | `<Sh: IntoDimension>(shape: Sh) -> Self` | `Tensor<A, D>` |
| `zeros_order` | `<Sh: IntoDimension>(shape: Sh, order: MemoryOrder) -> Self` | `Tensor<A, D>` |
| `ones` | `<Sh: IntoDimension>(shape: Sh) -> Self` | `Tensor<A, D>` |
| `full` | `<Sh: IntoDimension>(shape: Sh, value: A) -> Self` | `Tensor<A, D>` |
| `empty` | `<Sh: IntoDimension>(shape: Sh) -> Self` | `Tensor<A, D>` (unsafe) |
| `eye` | `(n: usize) -> Self` | `Tensor<A, Ix2>` |
| `identity` | `(rows: usize, cols: usize) -> Self` | `Tensor<A, Ix2>` |
| `diag` | `(diagonal: &Tensor<A, Ix1>) -> Self` | `Tensor<A, Ix2>` |
| `from_vec` | `(vec: Vec<A>) -> Tensor<A, Ix1>` | `Tensor<A, Ix1>` |
| `from_shape_vec` | `<Sh: IntoDimension>(shape: Sh, data: Vec<A>) -> Result<Self, ShapeError>` | `Tensor<A, D>` |
| `from_fn` | `<Sh: IntoDimension, F: FnMut(&[usize]) -> A>(shape: Sh, f: F) -> Self` | `Tensor<A, D>` |
| `arange` | `(start: f64, end: f64, step: f64) -> Self` | `Tensor<f64, Ix1>` |
| `linspace` | `(start: f64, end: f64, num: usize) -> Self` | `Tensor<f64, Ix1>` |
| `logspace` | `(start: f64, end: f64, num: usize, base: f64) -> Self` | `Tensor<f64, Ix1>` |

### A.2 转换方法

| 方法 | 签名 | 返回 |
|------|------|------|
| `to_owned` | `(&self) -> Tensor<A, D>` | `Tensor<A, D>` |
| `into_owned` | `(self) -> Tensor<A, D>` | `Tensor<A, D>` |
| `to_f_contiguous` | `(&self) -> Tensor<A, D>` | `Tensor<A, D>` |
| `to_c_contiguous` | `(&self) -> Tensor<A, D>` | `Tensor<A, D>` |
| `to_contiguous` | `(&self) -> Tensor<A, D>` | `Tensor<A, D>` |
| `cast` | `(&self) -> Tensor<B, D>` | `Tensor<B, D>` |
| `copy_to` | `(&self, dest: &mut TensorBase<St, D>)` | `()` |
| `fill` | `(&mut self, value: A)` | `()` |
| `clip` | `(&self, min: A, max: A) -> Tensor<A, D>` | `Tensor<A, D>` |
| `flip` | `(&self, axis: Axis) -> TensorView<'_, A, D>` | `TensorView<'_, A, D>` |

### A.3 映射方法

| 方法 | 签名 | 说明 |
|------|------|------|
| `map` | `<B, F: FnMut(&A) -> B>(&self, f: F) -> Tensor<B, D>` | 引用映射 |
| `mapv` | `<B, F: FnMut(A) -> B>(&self, f: F) -> Tensor<B, D>` | 值映射 |
| `mapv_inplace` | `<F: FnMut(A) -> A>(&mut self, f: F)` | 原地映射 |

---

## 附录 B：类型转换规则完整表

| 源类型 | 目标类型 | 行为 | 实现位置 |
|--------|----------|------|----------|
| `f64` | `f32` | round-to-nearest-even | `CastTo<f32> for f64` |
| `f32` | `f64` | 精确 | `CastTo<f64> for f32` |
| `f64` | `i32` | truncate + saturating | `CastTo<i32> for f64` |
| `f64` | `u32` | truncate + saturating | `CastTo<u32> for f64` |
| `f64` | `i64` | truncate + saturating | `CastTo<i64> for f64` |
| `f64` | `u64` | truncate + saturating | `CastTo<u64> for f64` |
| `f32` | `i32` | truncate + saturating | `CastTo<i32> for f32` |
| `i32` | `f64` | 精确 | `CastTo<f64> for i32` |
| `i64` | `f64` | 精确 | `CastTo<f64> for i64` |
| `i64` | `f32` | 可能不精确 | `CastTo<f32> for i64` |
| `i32` | `i8` | saturating | `CastTo<i8> for i32` |
| `i32` | `u8` | saturating | `CastTo<u8> for i32` |
| `u32` | `i32` | 可能溢出 | `CastTo<i32> for u32` |
| `bool` | `i32` | `true→1, false→0` | `CastTo<i32> for bool` |
| `bool` | `f64` | `true→1.0, false→0.0` | `CastTo<f64> for bool` |
| `i32` | `bool` | `非零→true` | `CastTo<bool> for i32` |
| `f64` | `Complex<f64>` | `虚部=0` | `CastTo<Complex<f64>> for f64` |
| `f32` | `Complex<f32>` | `虚部=0` | `CastTo<Complex<f32>> for f32` |

---

## 附录 C：格式化输出阈值配置

```rust
/// 全局格式化配置。
pub static FORMAT_CONFIG: AtomicUsize = AtomicUsize::new(1000);

/// 设置格式化阈值。
pub fn set_printoptions(threshold: usize) {
    FORMAT_CONFIG.store(threshold, Ordering::Relaxed);
}

/// 获取格式化阈值。
pub fn get_printoptions() -> usize {
    FORMAT_CONFIG.load(Ordering::Relaxed)
}
```

---

*文档结束*
