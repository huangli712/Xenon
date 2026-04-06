# Senon 形状操作模块设计文档

> **文档版本**: v1.0  
> **最后更新**: 2026-03-28  
> **模块路径**: `src/shape_ops/`  
> **需求来源**: require-v18.md §11

---

## 1. 模块概述

### 1.1 设计哲学

形状操作（Shape Operations）是 Senon 多维数组库的核心能力之一，用于在不复制数据的情况下改变数组的逻辑视图，或在必要时创建新数组。设计遵循以下原则：

| 原则 | 说明 |
|------|------|
| **零拷贝优先** | 凡是可以通过调整步长和偏移量实现的操作，均返回视图而非复制数据 |
| **类型安全** | 维度变化在编译时可追踪（静态维度）或运行时检查（动态维度） |
| **NumPy 兼容** | API 语义与 NumPy 保持一致，降低用户迁移成本 |
| **BLAS 友好** | 保持 F-order 布局的连续性，确保与 BLAS 的互操作性 |
| **显式语义** | 零拷贝与需拷贝操作分类明确，避免隐式性能陷阱 |

**核心设计决策**：

1. **零拷贝操作**：通过调整 `shape`、`strides` 和 `offset` 实现逻辑变形，返回 `TensorView`
2. **需拷贝操作**：当物理内存布局无法满足目标形状时，创建新的 `Owned` 数组
3. **条件零拷贝**：根据输入布局决定是否零拷贝（如 `flatten`）

### 1.2 在架构中的位置

```
┌─────────────────────────────────────────────────────────────┐
│                    用户层 API                                │
│    tensor.reshape(), tensor.transpose(), tensor.slice()     │
└─────────────────────┬───────────────────────────────────────┘
                      │ 方法调用
┌─────────────────────▼───────────────────────────────────────┐
│              形状操作模块 (本模块)                            │
│  reshape.rs: reshape, into_shape                            │
│  transpose.rs: transpose, permute, swapaxes, moveaxis       │
│  slice.rs: slice, index_axis, s![] 宏                       │
│  squeeze.rs: squeeze, unsqueeze, flatten                    │
│  split.rs: split, chunk, unstack                            │
│  stack.rs: cat, stack, vstack, hstack                       │
│  pad.rs: pad (Constant/Edge/Reflect)                        │
│  repeat.rs: repeat, tile                                    │
└─────────────────────┬───────────────────────────────────────┘
                      │ 依赖
┌─────────────────────▼───────────────────────────────────────┐
│            底层模块                                          │
│  tensor (核心结构) | layout (布局) | dimension (维度)        │
│  element (类型约束) | error (错误处理) | broadcast (广播)    │
└─────────────────────────────────────────────────────────────┘
```

**依赖层级**：

```
L5: shape_ops ←── tensor, layout, dimension, error, broadcast
         │
         ├──→ reshape.rs    (依赖 layout 连续性检查)
         ├──→ transpose.rs  (依赖 dimension 轴操作)
         ├──→ slice.rs      (依赖 index 模块)
         ├──→ squeeze.rs    (依赖 dimension 大小为 1 的检测)
         ├──→ split.rs      (依赖 slice)
         ├──→ stack.rs      (依赖 layout 连续性、broadcast)
         ├──→ pad.rs        (依赖 element)
         └──→ repeat.rs     (依赖 element)
```

### 1.3 操作分类

| 类型 | 操作 | 是否零拷贝 | 返回类型 |
|------|------|:----------:|----------|
| **形状变形** | reshape | ✓ (连续时) | TensorView / Result<Tensor, LayoutMismatch> |
| | flatten | 视情况 | TensorView / Tensor |
| | squeeze | ✓ | TensorView |
| | unsqueeze | ✓ | TensorView |
| **轴操作** | transpose | ✓ | TensorView |
| | permute | ✓ | TensorView |
| | swapaxes | ✓ | TensorView |
| | moveaxis | ✓ | TensorView |
| **切片与索引** | slice (s![]) | ✓ | TensorView |
| | index_axis | ✓ | TensorView |
| **分割** | split | ✓ | Vec<TensorView> |
| | chunk | ✓ | Vec<TensorView> |
| | unstack | ✓ | Vec<TensorView> |
| **拼接** | cat | ✗ | Tensor |
| | stack | ✗ | Tensor |
| **填充与重复** | pad | ✗ | Tensor |
| | repeat / tile | ✗ | Tensor |

---

## 2. 文件结构

```
src/shape_ops/
├── mod.rs             # 模块入口，re-export 公开 trait 和函数
├── reshape.rs         # reshape, into_shape, flatten
├── transpose.rs       # transpose, permute_axes, swap_axes, moveaxis
├── slice.rs           # slice, slice_mut, index_axis, s![] 宏
├── squeeze.rs         # squeeze, unsqueeze, expand_dims
├── pad.rs             # pad (Constant/Edge/Reflect 模式)
├── repeat.rs          # repeat, tile
├── split.rs           # split, chunk, unstack, hsplit, vsplit, dsplit
└── stack.rs           # cat, concatenate, stack, vstack, hstack, dstack
```

### 2.1 各文件职责

| 文件 | 职责 | 核心内容 |
|------|------|----------|
| `mod.rs` | 模块组织与导出 | re-export 形状操作函数、定义 `ShapeOps` trait |
| `reshape.rs` | 形状变形 | `reshape`, `into_shape`, `flatten`, `into_shape_with_order` |
| `transpose.rs` | 轴顺序调整 | `transpose`, `permute_axes`, `swap_axes`, `moveaxis`, `t()` |
| `slice.rs` | 切片操作 | `slice`, `slice_mut`, `slice_collapse`, `index_axis`, `s![]` 宏 |
| `squeeze.rs` | 维度压缩与扩展 | `squeeze`, `unsqueeze`, `expand_dims` |
| `pad.rs` | 边界填充 | `pad`, `PadMode` 枚举 (Constant/Edge/Reflect) |
| `repeat.rs` | 元素重复 | `repeat`, `tile` |
| `split.rs` | 数组分割 | `split`, `chunk`, `unstack`, `hsplit`, `vsplit`, `dsplit` |
| `stack.rs` | 数组拼接 | `cat`, `concatenate`, `stack`, `vstack`, `hstack`, `dstack` |

### 2.2 模块依赖

| 依赖模块 | 用途 |
|---------|------|
| `crate::tensor` | `TensorBase<S, D>` 核心类型 |
| `crate::dimension` | `Dimension`, `Ix0`-`Ix6`, `IxDyn`, `Axis`, `RemoveAxis` |
| `crate::layout` | `LayoutFlags`, 连续性检查, 步长计算 |
| `crate::element` | `Element`, `Numeric` trait 约束 |
| `crate::error` | `InvalidShape`, `LayoutMismatch`, `InvalidAxis`, `ShapeMismatch` |
| `crate::broadcast` | `can_broadcast`, 广播形状检查 |
| `crate::iter` | 轴迭代器, 元素迭代器 |

---

## 3. 零拷贝操作详细设计

零拷贝操作通过调整视图的 `shape`、`strides` 和 `offset` 实现逻辑变形，不复制底层数据。

### 3.1 reshape

#### 3.1.1 方法签名

```rust
// reshape.rs

impl<S, A, D> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// 重塑数组形状
    ///
    /// 在不复制数据的情况下改变数组的逻辑形状。
    /// 仅当数组内存连续时可零拷贝执行，否则返回错误。
    ///
    /// # 参数
    ///
    /// * `shape` - 目标形状，元素总数须与原数组相同
    ///
    /// # 返回值
    ///
    /// * `Ok(TensorView<'_, A, E>)` - 重塑后的视图（零拷贝）
    /// * `Err(InvalidShape)` - 目标元素总数与源不一致
    /// * `Err(LayoutMismatch)` - 源数组非连续，无法零拷贝重塑
    ///
    /// # 约束
    ///
    /// - 目标形状元素总数 = 源数组元素总数
    /// - 源数组须连续（F-contiguous 或 C-contiguous）
    /// - 目标维度数受 `Dimension` trait 约束
    ///
    /// # 示例
    ///
    /// ```
    /// use Senon::{Tensor, Ix2, Ix3};
    ///
    /// let a = Tensor::<f64, Ix3>::zeros([2, 3, 4]);
    /// let reshaped = a.reshape([6, 4]).unwrap();
    /// assert_eq!(reshaped.shape(), &[6, 4]);
    ///
    /// // 元素总数不匹配
    /// let result = a.reshape([5, 4]);
    /// assert!(result.is_err());
    /// ```
    pub fn reshape<E>(&self, shape: E) -> Result<TensorView<'_, A, E>, ShapeError>
    where
        E: Dimension,
    {
        // 1. 检查元素总数匹配
        let new_len: usize = shape.slice().iter().product();
        if new_len != self.len() {
            return Err(ShapeError::InvalidShape {
                expected: self.len(),
                actual: new_len,
                reason: "element count mismatch",
            });
        }
        
        // 2. 检查连续性
        if !self.is_contiguous() {
            return Err(ShapeError::LayoutMismatch {
                reason: "source array is not contiguous",
            });
        }
        
        // 3. 计算新步长
        let order = if self.is_f_contiguous() {
            MemoryOrder::F
        } else {
            MemoryOrder::C
        };
        let new_strides = Strides::from_shape(&shape, order);
        
        // 4. 创建视图
        Ok(TensorView {
            storage: ViewRepr::from(&self.storage),
            shape,
            strides: new_strides,
            offset: self.offset,
            layout: LayoutFlags::from_order(order),
        })
    }
    
    /// 消费数组并重塑形状
    ///
    /// 与 `reshape` 类似，但消费原数组并返回拥有数据的 Tensor。
    /// 即使源数组非连续，也可通过拷贝实现重塑。
    ///
    /// # 参数
    ///
    /// * `shape` - 目标形状
    ///
    /// # 返回值
    ///
    /// * `Ok(Tensor<A, E>)` - 重塑后的拥有型数组
    /// * `Err(InvalidShape)` - 目标元素总数与源不一致
    ///
    /// # 示例
    ///
    /// ```
    /// use Senon::{Tensor, Ix2};
    ///
    /// let a = Tensor::<f64, Ix2>::zeros([4, 6]);
    /// let reshaped = a.into_shape([2, 12]).unwrap();
    /// assert_eq!(reshaped.shape(), &[2, 12]);
    /// ```
    pub fn into_shape<E>(self, shape: E) -> Result<Tensor<A, E>, ShapeError>
    where
        E: Dimension,
        S: IntoOwned<A>,
    {
        let new_len: usize = shape.slice().iter().product();
        if new_len != self.len() {
            return Err(ShapeError::InvalidShape {
                expected: self.len(),
                actual: new_len,
                reason: "element count mismatch",
            });
        }
        
        // 如果连续，直接重塑
        if self.is_contiguous() {
            let order = if self.is_f_contiguous() { MemoryOrder::F } else { MemoryOrder::C };
            let new_strides = Strides::from_shape(&shape, order);
            
            return Ok(Tensor {
                storage: self.storage.into_owned(),
                shape,
                strides: new_strides,
                offset: 0,
                layout: LayoutFlags::from_order(order),
            });
        }
        
        // 非连续，拷贝后重塑
        let mut owned = self.to_f_contiguous();
        let new_strides = Strides::from_shape(&shape, MemoryOrder::F);
        owned.shape = shape;
        owned.strides = new_strides;
        Ok(owned)
    }
}
```

#### 3.1.2 语义表

| 属性 | 行为 |
|------|------|
| 元素总数 | 必须保持不变 |
| 连续性要求 | 零拷贝 reshape 要求源数组 F-contiguous 或 C-contiguous |
| 非连续数组 | `reshape()` 返回 `Err(LayoutMismatch)`，`into_shape()` 自动拷贝 |
| 布局保持 | F-contiguous 输入默认输出 F-contiguous |
| -1 维度 | 支持（如 NumPy），-1 自动推断为剩余元素数 |
| 空数组 | 允许 reshape 到任意元素总数为 0 的形状 |
| 单元素数组 | 可 reshape 到任意形状（所有维度均为 1 或 0） |

#### 3.1.3 连续性约束详解

```
连续性检查逻辑:

is_contiguous() = is_f_contiguous() || is_c_contiguous()

F-contiguous 条件:
  strides[i] = product(shape[0..i])  (列优先)
  例: shape=[2,3,4], strides=[1,2,6]

C-contiguous 条件:
  strides[i] = product(shape[i+1..ndim])  (行优先)
  例: shape=[2,3,4], strides=[12,4,1]

非连续示例 (reshape 失败):
  原始: shape=[4,6], strides=[6,1] (C-order)
  切片后: shape=[2,6], strides=[12,1] (非连续)
  → reshape([12]) 失败，因为 strides[0]=12 ≠ 1
```

#### 3.1.4 示例代码

```rust
use Senon::{Tensor, TensorView, Ix2, Ix3, s};

// 基本 reshape
let a = Tensor::<f64, Ix3>::zeros([2, 3, 4]);  // 24 elements
let b = a.reshape([6, 4]).unwrap();             // shape: [6, 4]
let c = a.reshape([24]).unwrap();               // shape: [24]
let d = a.reshape([2, 12]).unwrap();            // shape: [2, 12]

// 使用 -1 自动推断
let e = a.reshape([2, -1]).unwrap();            // shape: [2, 12]
let f = a.reshape([-1, 8]).unwrap();            // shape: [3, 8]

// 非连续数组 reshape 失败
let g = Tensor::<f64, Ix2>::zeros([4, 6]);
let slice = g.slice(s![1..3, ..]);              // shape: [2, 6], 非连续
assert!(slice.reshape([12]).is_err());

// into_shape 自动处理非连续
let h = slice.into_shape([12]).unwrap();        // OK, 自动拷贝

// 空数组 reshape
let empty: Tensor<f64, Ix1> = Tensor::zeros([0]);
let reshaped_empty = empty.reshape([0, 5]).unwrap();  // shape: [0, 5]
```

---

### 3.2 transpose

#### 3.2.1 方法签名

```rust
// transpose.rs

impl<S, A, D> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// 转置数组
    ///
    /// 返回轴顺序反转的视图，零拷贝操作。
    ///
    /// # 返回值
    ///
    /// 转置后的 `TensorView`，形状和步长均反转
    ///
    /// # 二维数组
    ///
    /// 对于 2D 数组，等价于矩阵转置（行变列，列变行）
    ///
    /// # 高维数组
    ///
    /// 对于 N 维数组，轴顺序完全反转 (0→N-1, 1→N-2, ...)
    ///
    /// # 示例
    ///
    /// ```
    /// use Senon::{Tensor2, Ix2};
    ///
    /// let a = Tensor2::<f64>::from_shape_vec([2, 3], vec![1,2,3,4,5,6]);
    /// let b = a.transpose();
    /// assert_eq!(b.shape(), &[3, 2]);
    /// // a: [[1,2,3], [4,5,6]]
    /// // b: [[1,4], [2,5], [3,6]]
    /// ```
    pub fn transpose(&self) -> TensorView<'_, A, D>
    where
        D: Reverse,
    {
        let new_shape = self.shape().reverse();
        let new_strides = self.strides().reverse();
        
        TensorView {
            storage: ViewRepr::from(&self.storage),
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            layout: self.layout.update_for_transpose(),
        }
    }
    
    /// 简写转置（仅 2D）
    ///
    /// 等价于 `.transpose()`，提供类似 ndarray 的简洁语法
    ///
    /// # 示例
    ///
    /// ```
    /// let a = Tensor2::<f64>::zeros([3, 4]);
    /// let b = a.t();  // shape: [4, 3]
    /// ```
    pub fn t(&self) -> TensorView<'_, A, D>
    where
        D: Dimension + Reverse,
    {
        self.transpose()
    }
}
```

#### 3.2.2 语义表

| 属性 | 行为 |
|------|------|
| 零拷贝 | 始终零拷贝，仅调整步长 |
| 形状变化 | `shape[i]` → `shape[ndim-1-i]` |
| 步长变化 | `strides[i]` → `strides[ndim-1-i]` |
| 连续性 | F-contiguous ↔ C-contiguous 互换 |
| 偏移量 | 保持不变 |
| 1D 数组 | 转置后形状不变（1D 无轴顺序概念） |
| 0D 标量 | 转置后不变 |

#### 3.2.3 示例代码

```rust
use Senon::{Tensor2, Tensor3, Ix2, Ix3};

// 2D 转置
let a = Tensor2::<f64>::from_shape_vec([2, 3], vec![1,2,3,4,5,6]);
let b = a.t();
assert_eq!(b.shape(), &[3, 2]);

// 3D 转置（轴反转）
let c = Tensor3::<f64>::zeros([2, 3, 4]);  // shape: [2, 3, 4]
let d = c.transpose();                      // shape: [4, 3, 2]

// 连续性变化
let e = Tensor2::<f64>::zeros([3, 4]);  // F-contiguous
let f = e.t();                           // C-contiguous
assert!(e.is_f_contiguous());
assert!(f.is_c_contiguous());
```

---

### 3.3 permute

#### 3.3.1 方法签名

```rust
// transpose.rs

impl<S, A, D> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// 重新排列轴顺序
    ///
    /// 根据指定的轴顺序返回重排后的视图，零拷贝操作。
    ///
    /// # 参数
    ///
    /// * `axes` - 新轴顺序，长度须等于 ndim，包含 0..ndim 的所有整数
    ///
    /// # 返回值
    ///
    /// * `Ok(TensorView)` - 重排后的视图
    /// * `Err(InvalidAxis)` - 轴索引无效或重复
    ///
    /// # 示例
    ///
    /// ```
    /// use Senon::{Tensor3, Ix3};
    ///
    /// let a = Tensor3::<f64>::zeros([2, 3, 4]);
    /// // 原轴顺序: (0, 1, 2) -> shape [2, 3, 4]
    /// // 新轴顺序: (2, 0, 1) -> shape [4, 2, 3]
    /// let b = a.permute([2, 0, 1]).unwrap();
    /// assert_eq!(b.shape(), &[4, 2, 3]);
    /// ```
    pub fn permute<E>(&self, axes: E) -> Result<TensorView<'_, A, D>, InvalidAxis>
    where
        E: AsRef<[usize]>,
    {
        let axes = axes.as_ref();
        let ndim = self.ndim();
        
        // 验证轴顺序
        if axes.len() != ndim {
            return Err(InvalidAxis {
                axis: ndim,  // 使用 ndim 表示长度错误
                ndim,
                reason: "axes length must equal ndim",
            });
        }
        
        // 检查轴索引有效性和唯一性
        let mut seen = vec![false; ndim];
        for &ax in axes {
            if ax >= ndim {
                return Err(InvalidAxis {
                    axis: ax,
                    ndim,
                    reason: "axis index out of bounds",
                });
            }
            if seen[ax] {
                return Err(InvalidAxis {
                    axis: ax,
                    ndim,
                    reason: "duplicate axis index",
                });
            }
            seen[ax] = true;
        }
        
        // 计算新形状和步长
        let new_shape: SmallVec<[usize; 6]> = axes.iter()
            .map(|&ax| self.shape()[ax])
            .collect();
        let new_strides: SmallVec<[isize; 6]> = axes.iter()
            .map(|&ax| self.strides()[ax])
            .collect();
        
        Ok(TensorView {
            storage: ViewRepr::from(&self.storage),
            shape: D::from_slice(&new_shape),
            strides: D::from_isize_slice(&new_strides),
            offset: self.offset,
            layout: self.layout.update_for_permute(&axes),
        })
    }
    
    /// permute 的别名，与 PyTorch API 兼容
    pub fn permute_axes<E>(&self, axes: E) -> Result<TensorView<'_, A, D>, InvalidAxis>
    where
        E: AsRef<[usize]>,
    {
        self.permute(axes)
    }
}
```

#### 3.3.2 语义表

| 属性 | 行为 |
|------|------|
| 零拷贝 | 始终零拷贝 |
| 轴顺序 | 任意排列，须包含所有轴索引 |
| 形状变化 | `shape[i]` → `shape[axes[i]]` |
| 步长变化 | `strides[i]` → `strides[axes[i]]` |
| 连续性 | 通常变为非连续，除非排列是恒等或反转 |
| 偏移量 | 保持不变 |

#### 3.3.3 示例代码

```rust
use Senon::{Tensor3, Tensor4, Ix3, Ix4};

// 基本排列
let a = Tensor3::<f64>::zeros([2, 3, 4]);
let b = a.permute([2, 0, 1]).unwrap();
assert_eq!(b.shape(), &[4, 2, 3]);

// 高维排列
let c = Tensor4::<f64>::zeros([2, 3, 4, 5]);
let d = c.permute([3, 1, 0, 2]).unwrap();
assert_eq!(d.shape(), &[5, 3, 2, 4]);

// 错误示例
let e = Tensor3::<f64>::zeros([2, 3, 4]);
assert!(e.permute([0, 1]).is_err());      // 长度不匹配
assert!(e.permute([0, 1, 3]).is_err());   // 轴索引越界
assert!(e.permute([0, 1, 1]).is_err());   // 重复轴
```

---

### 3.4 swapaxes / moveaxis

#### 3.4.1 方法签名

```rust
// transpose.rs

impl<S, A, D> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// 交换两个轴
    ///
    /// 零拷贝操作，交换指定两个轴的位置。
    ///
    /// # 参数
    ///
    /// * `ax1` - 第一个轴索引
    /// * `ax2` - 第二个轴索引
    ///
    /// # 返回值
    ///
    /// * `Ok(TensorView)` - 交换后的视图
    /// * `Err(InvalidAxis)` - 轴索引越界
    ///
    /// # 示例
    ///
    /// ```
    /// use Senon::{Tensor3, Ix3};
    ///
    /// let a = Tensor3::<f64>::zeros([2, 3, 4]);
    /// let b = a.swapaxes(0, 2).unwrap();
    /// assert_eq!(b.shape(), &[4, 3, 2]);
    /// ```
    pub fn swapaxes(&self, ax1: usize, ax2: usize) -> Result<TensorView<'_, A, D>, InvalidAxis> {
        let ndim = self.ndim();
        
        if ax1 >= ndim || ax2 >= ndim {
            return Err(InvalidAxis {
                axis: if ax1 >= ndim { ax1 } else { ax2 },
                ndim,
                reason: "axis index out of bounds",
            });
        }
        
        if ax1 == ax2 {
            // 无需交换
            return Ok(self.view());
        }
        
        // 构建排列
        let mut axes: Vec<usize> = (0..ndim).collect();
        axes.swap(ax1, ax2);
        
        self.permute(&axes)
    }
    
    /// 移动轴到新位置
    ///
    /// 将指定轴移动到目标位置，其他轴保持相对顺序。
    ///
    /// # 参数
    ///
    /// * `source` - 源轴索引
    /// * `destination` - 目标位置索引
    ///
    /// # 返回值
    ///
    /// * `Ok(TensorView)` - 移动后的视图
    /// * `Err(InvalidAxis)` - 轴索引越界
    ///
    /// # 示例
    ///
    /// ```
    /// use Senon::{Tensor3, Ix3};
    ///
    /// let a = Tensor3::<f64>::zeros([2, 3, 4]);  // shape: [2, 3, 4]
    /// let b = a.moveaxis(0, 2).unwrap();          // shape: [3, 4, 2]
    /// // 轴 0 移动到位置 2，原轴 1,2 左移
    /// ```
    pub fn moveaxis(
        &self,
        source: usize,
        destination: usize,
    ) -> Result<TensorView<'_, A, D>, InvalidAxis> {
        let ndim = self.ndim();
        
        if source >= ndim || destination >= ndim {
            return Err(InvalidAxis {
                axis: if source >= ndim { source } else { destination },
                ndim,
                reason: "axis index out of bounds",
            });
        }
        
        if source == destination {
            return Ok(self.view());
        }
        
        // 构建排列
        let mut axes: Vec<usize> = (0..ndim).collect();
        let src_val = axes.remove(source);
        axes.insert(destination, src_val);
        
        self.permute(&axes)
    }
}
```

#### 3.4.2 语义表

| 操作 | 输入形状 | 参数 | 输出形状 |
|------|----------|------|----------|
| `swapaxes(0, 2)` | [2, 3, 4] | ax1=0, ax2=2 | [4, 3, 2] |
| `swapaxes(1, 2)` | [2, 3, 4] | ax1=1, ax2=2 | [2, 4, 3] |
| `moveaxis(0, 2)` | [2, 3, 4] | src=0, dst=2 | [3, 4, 2] |
| `moveaxis(2, 0)` | [2, 3, 4] | src=2, dst=0 | [4, 2, 3] |

---

### 3.5 squeeze / unsqueeze

#### 3.5.1 方法签名

```rust
// squeeze.rs

impl<S, A, D> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// 移除大小为 1 的轴
    ///
    /// 零拷贝操作，压缩指定或所有大小为 1 的维度。
    ///
    /// # 参数
    ///
    /// * `axis` - 可选，指定要压缩的轴；若为 None，压缩所有大小为 1 的轴
    ///
    /// # 返回值
    ///
    /// * `Ok(TensorView)` - 压缩后的视图
    /// * `Err(InvalidAxis)` - 轴索引越界，或指定轴大小不为 1
    ///
    /// # 维度变化
    ///
    /// 返回类型为 `TensorView<'_, A, D::Smaller>`（指定轴）或动态维度
    ///
    /// # 示例
    ///
    /// ```
    /// use Senon::{Tensor3, Ix2};
    ///
    /// let a = Tensor3::<f64>::zeros([2, 1, 4]);
    /// let b = a.squeeze(None).unwrap();  // shape: [2, 4]
    /// let c = a.squeeze(Some(1)).unwrap();  // shape: [2, 4]
    /// ```
    pub fn squeeze<E>(&self, axis: Option<usize>) -> Result<TensorView<'_, A, E>, InvalidAxis>
    where
        E: Dimension,
    {
        match axis {
            Some(ax) => {
                // 压缩指定轴
                if ax >= self.ndim() {
                    return Err(InvalidAxis {
                        axis: ax,
                        ndim: self.ndim(),
                        reason: "axis index out of bounds",
                    });
                }
                if self.shape()[ax] != 1 {
                    return Err(InvalidAxis {
                        axis: ax,
                        ndim: self.ndim(),
                        reason: "cannot squeeze axis with size != 1",
                    });
                }
                
                // 构建新形状和步长
                let new_shape: SmallVec<[usize; 6]> = self.shape()
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != ax)
                    .map(|(_, &s)| s)
                    .collect();
                let new_strides: SmallVec<[isize; 6]> = self.strides()
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != ax)
                    .map(|(_, &s)| s)
                    .collect();
                
                Ok(TensorView {
                    storage: ViewRepr::from(&self.storage),
                    shape: E::from_slice(&new_shape),
                    strides: E::from_isize_slice(&new_strides),
                    offset: self.offset,
                    layout: self.layout.update_for_squeeze(ax),
                })
            }
            None => {
                // 压缩所有大小为 1 的轴
                let new_shape: SmallVec<[usize; 6]> = self.shape()
                    .iter()
                    .filter(|&&s| s != 1)
                    .copied()
                    .collect();
                let new_strides: SmallVec<[isize; 6]> = self.strides()
                    .iter()
                    .zip(self.shape().iter())
                    .filter(|(_, &s)| s != 1)
                    .map(|(&st, _)| st)
                    .collect();
                
                Ok(TensorView {
                    storage: ViewRepr::from(&self.storage),
                    shape: E::from_slice(&new_shape),
                    strides: E::from_isize_slice(&new_strides),
                    offset: self.offset,
                    layout: self.layout.update_for_squeeze_all(),
                })
            }
        }
    }
    
    /// 在指定位置插入大小为 1 的轴
    ///
    /// 零拷贝操作，扩展数组维度。
    ///
    /// # 参数
    ///
    /// * `axis` - 新轴插入位置（0 到 ndim 之间）
    ///
    /// # 返回值
    ///
    /// 扩展后的 `TensorView`，新维度大小为 1
    ///
    /// # 示例
    ///
    /// ```
    /// use Senon::{Tensor1, Ix2};
    ///
    /// let a = Tensor1::<f64>::zeros([5]);  // shape: [5]
    /// let b = a.unsqueeze(0);  // shape: [1, 5]
    /// let c = a.unsqueeze(1);  // shape: [5, 1]
    /// ```
    pub fn unsqueeze<E>(&self, axis: usize) -> TensorView<'_, A, E>
    where
        E: Dimension,
    {
        let ndim = self.ndim();
        let actual_axis = axis.min(ndim);
        
        // 构建新形状和步长
        let mut new_shape: SmallVec<[usize; 6]> = SmallVec::with_capacity(ndim + 1);
        let mut new_strides: SmallVec<[isize; 6]> = SmallVec::with_capacity(ndim + 1);
        
        for (i, (&s, &st)) in self.shape().iter()
            .zip(self.strides().iter())
            .enumerate()
        {
            if i == actual_axis {
                new_shape.push(1);
                // 计算新轴步长：如果插入位置在开头，步长为元素总数；否则为下一轴步长
                let new_stride = if actual_axis == 0 {
                    self.len() as isize
                } else {
                    new_strides[actual_axis - 1] * new_shape[actual_axis - 1] as isize
                };
                new_strides.push(new_stride);
            }
            new_shape.push(s);
            new_strides.push(st);
        }
        
        if actual_axis >= ndim {
            new_shape.push(1);
            new_strides.push(1);
        }
        
        TensorView {
            storage: ViewRepr::from(&self.storage),
            shape: E::from_slice(&new_shape),
            strides: E::from_isize_slice(&new_strides),
            offset: self.offset,
            layout: self.layout.update_for_unsqueeze(actual_axis),
        }
    }
    
    /// unsqueeze 的别名，与 PyTorch API 兼容
    pub fn expand_dims<E>(&self, axis: usize) -> TensorView<'_, A, E>
    where
        E: Dimension,
    {
        self.unsqueeze(axis)
    }
}
```

#### 3.5.2 语义表

| 操作 | 输入形状 | 参数 | 输出形状 | 说明 |
|------|----------|------|----------|------|
| `squeeze(None)` | [2, 1, 4] | - | [2, 4] | 压缩所有 size-1 轴 |
| `squeeze(None)` | [1, 1, 4] | - | [4] | 多个 size-1 轴全部压缩 |
| `squeeze(Some(1))` | [2, 1, 4] | ax=1 | [2, 4] | 压缩指定轴 |
| `squeeze(Some(0))` | [2, 3, 4] | ax=0 | Error | 轴大小不为 1 |
| `unsqueeze(0)` | [5] | ax=0 | [1, 5] | 前置新轴 |
| `unsqueeze(1)` | [5] | ax=1 | [5, 1] | 后置新轴 |
| `unsqueeze(2)` | [5] | ax=2 | [5, 1] | 等同于 ax=1 |

---

### 3.6 flatten

#### 3.6.1 方法签名

```rust
// reshape.rs

impl<S, A, D> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// 展平数组
    ///
    /// 将多维数组展平为一维数组。
    /// 当数组连续时零拷贝，否则需要拷贝。
    ///
    /// # 参数
    ///
    /// * `order` - 可选，指定展平顺序（F 或 C）
    ///
    /// # 返回值
    ///
    /// 展平后的 `Tensor1<A>` 或 `TensorView<'_, A, Ix1>`
    ///
    /// # 示例
    ///
    /// ```
    /// use Senon::{Tensor2, Ix1};
    ///
    /// let a = Tensor2::<f64>::zeros([2, 3]);
    /// let b = a.flatten(None);  // shape: [6]
    /// ```
    pub fn flatten(&self, order: Option<MemoryOrder>) -> CowTensor<'_, A, Ix1>
    where
        A: Clone,
    {
        let target_order = order.unwrap_or_else(|| {
            if self.is_f_contiguous() {
                MemoryOrder::F
            } else if self.is_c_contiguous() {
                MemoryOrder::C
            } else {
                MemoryOrder::F  // 默认
            }
        });
        
        // 检查是否可以零拷贝
        let can_zero_copy = match target_order {
            MemoryOrder::F => self.is_f_contiguous(),
            MemoryOrder::C => self.is_c_contiguous(),
        };
        
        if can_zero_copy {
            // 零拷贝：仅调整形状
            CowTensor::View(TensorView {
                storage: ViewRepr::from(&self.storage),
                shape: Ix1(self.len()),
                strides: Ix1(1),
                offset: self.offset,
                layout: LayoutFlags::F_CONTIGUOUS | LayoutFlags::C_CONTIGUOUS,
            })
        } else {
            // 需要拷贝
            let mut data = vec![A::zero(); self.len()];
            
            match target_order {
                MemoryOrder::F => {
                    // F-order 遍历
                    for (i, &val) in self.iter().enumerate() {
                        data[i] = val;
                    }
                }
                MemoryOrder::C => {
                    // C-order 遍历
                    for (idx, &val) in self.indexed_iter() {
                        let flat_idx = self.shape().iter()
                            .rev()
                            .zip(idx.iter().rev())
                            .fold(0usize, |acc, (&s, &i)| acc * s + i);
                        data[flat_idx] = val;
                    }
                }
            }
            
            CowTensor::Owned(Tensor {
                storage: Owned::from_vec(data),
                shape: Ix1(self.len()),
                strides: Ix1(1),
                offset: 0,
                layout: LayoutFlags::F_CONTIGUOUS | LayoutFlags::C_CONTIGUOUS,
            })
        }
    }
    
    /// 展平指定范围内的轴
    ///
    /// 将 `start_axis` 到 `end_axis`（含）之间的所有轴展平为单个轴。
    ///
    /// # 参数
    ///
    /// * `start_axis` - 起始轴（含）
    /// * `end_axis` - 结束轴（含）
    ///
    /// # 示例
    ///
    /// ```
    /// use Senon::{Tensor3, Ix2};
    ///
    /// let a = Tensor3::<f64>::zeros([2, 3, 4]);  // shape: [2, 3, 4]
    /// let b = a.flatten_axes(1, 2).unwrap();      // shape: [2, 12]
    /// ```
    pub fn flatten_axes(&self, start_axis: usize, end_axis: usize) -> Result<TensorView<'_, A, IxDyn>, InvalidAxis> {
        let ndim = self.ndim();
        
        if start_axis >= ndim || end_axis >= ndim || start_axis > end_axis {
            return Err(InvalidAxis {
                axis: start_axis.max(end_axis),
                ndim,
                reason: "invalid axis range",
            });
        }
        
        // 计算展平后的维度大小
        let flattened_size: usize = self.shape()[start_axis..=end_axis].iter().product();
        
        // 构建新形状
        let mut new_shape: SmallVec<[usize; 6]> = SmallVec::new();
        for &s in &self.shape()[..start_axis] {
            new_shape.push(s);
        }
        new_shape.push(flattened_size);
        for &s in &self.shape()[end_axis + 1..] {
            new_shape.push(s);
        }
        
        // 零拷贝视图（假设展平区域在内存中连续）
        // 实际实现需要更复杂的步长计算
        // ...
        
        unimplemented!()
    }
}
```

#### 3.6.2 语义表

| 属性 | 行为 |
|------|------|
| 连续输入 | 零拷贝，返回视图 |
| 非连续输入 | 需拷贝，返回拥有型数组 |
| 默认顺序 | F-order（保持与 BLAS 兼容） |
| F-contiguous 输入 | `flatten(F)` 零拷贝 |
| C-contiguous 输入 | `flatten(C)` 零拷贝 |
| 空数组 | 返回空 1D 数组 |

---

### 3.7 slice

#### 3.7.1 方法签名

```rust
// slice.rs

/// 切片描述符
///
/// 用于描述单个维度的切片范围。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceInfo {
    /// 起始索引（含）
    pub start: usize,
    /// 结束索引（不含）
    pub end: usize,
    /// 步长
    pub step: usize,
}

impl SliceInfo {
    /// 创建切片描述符
    pub fn new(start: usize, end: usize, step: usize) -> Self {
        Self { start, end, step }
    }
    
    /// 计算切片后的长度
    pub fn len(&self) -> usize {
        if self.start >= self.end {
            return 0;
        }
        (self.end - self.start + self.step - 1) / self.step
    }
}

impl<S, A, D> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// 切片操作
    ///
    /// 使用 `s![]` 宏创建切片描述符，返回零拷贝视图。
    ///
    /// # 参数
    ///
    /// * `info` - 切片描述符，通过 `s![]` 宏创建
    ///
    /// # 返回值
    ///
    /// 切片后的 `TensorView`
    ///
    /// # 示例
    ///
    /// ```
    /// use Senon::{Tensor2, s};
    ///
    /// let a = Tensor2::<f64>::zeros([6, 8]);
    ///
    /// // 基本切片
    /// let b = a.slice(s![1..4, 2..6]);       // shape: [3, 4]
    ///
    /// // 步长切片
    /// let c = a.slice(s![..;2, ..;3]);       // shape: [3, 3]
    ///
    /// // 混合切片
    /// let d = a.slice(s![1..5;2, 2..7]);     // shape: [2, 5]
    ///
    /// // 开放范围
    /// let e = a.slice(s![.., ..]);           // shape: [6, 8] (全范围)
    /// let f = a.slice(s![..3, 2..]);         // shape: [3, 6]
    /// ```
    pub fn slice<I>(&self, info: I) -> TensorView<'_, A, I::OutputDim>
    where
        I: SliceArg<D>,
    {
        let slices = info.as_slice();
        let ndim = self.ndim();
        
        // 计算新形状、步长和偏移
        let mut new_shape: SmallVec<[usize; 6]> = SmallVec::with_capacity(ndim);
        let mut new_strides: SmallVec<[isize; 6]> = SmallVec::with_capacity(ndim);
        let mut new_offset = self.offset;
        
        for (i, slice_info) in slices.iter().enumerate() {
            let dim_size = self.shape()[i];
            let stride = self.strides()[i];
            
            // 调整范围
            let start = slice_info.start.min(dim_size);
            let end = slice_info.end.min(dim_size);
            let step = slice_info.step.max(1);
            
            // 计算新维度大小
            let new_dim_size = if start >= end { 0 } else { (end - start + step - 1) / step };
            new_shape.push(new_dim_size);
            
            // 计算新步长
            new_strides.push(stride * step as isize);
            
            // 更新偏移
            new_offset += start * stride as usize;
        }
        
        TensorView {
            storage: ViewRepr::from(&self.storage),
            shape: I::OutputDim::from_slice(&new_shape),
            strides: I::OutputDim::from_isize_slice(&new_strides),
            offset: new_offset,
            layout: self.layout.update_for_slice(&slices),
        }
    }
    
    /// 可变切片操作
    ///
    /// 返回可写视图，语义同 `slice`。
    pub fn slice_mut<I>(&mut self, info: I) -> TensorViewMut<'_, A, I::OutputDim>
    where
        I: SliceArg<D>,
        S: StorageMut<Elem = A>,
    {
        // 类似 slice，但返回 TensorViewMut
        // ...
    }
}

/// 切片宏
///
/// 用于创建切片描述符的便捷宏。
///
/// # 语法
///
/// - `..` : 全范围
/// - `a..b` : 范围 [a, b)
/// - `a..` : 从 a 到末尾
/// - `..b` : 从开头到 b
/// - `a..b;c` : 范围 [a, b)，步长 c
/// - `..;c` : 全范围，步长 c
///
/// # 示例
///
/// ```
/// use Senon::s;
///
/// let info = s![1..4, 2..6];       // 两维切片
/// let info = s![.., 0..5;2];       // 混合切片
/// let info = s![0..10;2, ..;3];    // 步长切片
/// ```
#[macro_export]
macro_rules! s {
    [$($t:tt),* $(,)?] => {
        // 宏展开逻辑
        // ...
    };
}
```

#### 3.7.2 语义表

| 切片语法 | 含义 | 等价 Python |
|----------|------|-------------|
| `..` | 全范围 | `:` |
| `a..b` | 范围 [a, b) | `a:b` |
| `a..` | 从 a 到末尾 | `a:` |
| `..b` | 从开头到 b | `:b` |
| `a..b;c` | 范围 [a, b)，步长 c | `a:b:c` |
| `..;c` | 全范围，步长 c | `::c` |
| `a..b;-c` | 反向范围（步长为负） | `a:b:-c` |

#### 3.7.3 边界情况处理

| 情况 | 行为 |
|------|------|
| 空切片（start >= end） | 返回该维度大小为 0 的视图 |
| 越界起始索引 | 自动截断到末尾 |
| 越界结束索引 | 自动截断到末尾 |
| 步长为 0 | panic（无效步长） |
| 负步长 | 支持，产生反转视图 |

---

### 3.8 index_axis

#### 3.8.1 方法签名

```rust
// slice.rs

impl<S, A, D> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// 沿轴取单个切片
    ///
    /// 沿指定轴取单个索引，返回降维视图（ndim - 1）。
    /// 零拷贝操作，共享底层存储。
    ///
    /// # 参数
    ///
    /// * `axis` - 轴索引
    /// * `index` - 沿该轴的索引
    ///
    /// # 返回值
    ///
    /// * `Ok(TensorView)` - 降维后的视图
    /// * `Err(InvalidAxis)` - 轴索引越界
    /// * `Err(IndexOutOfBounds)` - 索引越界
    ///
    /// # 降维语义
    ///
    /// - 输入: shape [D0, D1, ..., Dn]
    /// - 输出: shape [D0, ..., D(axis-1), D(axis+1), ..., Dn]
    /// - ndim 减少 1
    ///
    /// # BLAS 兼容性
    ///
    /// 从 3D batch tensor 沿最外层 batch 轴索引取出的 2D 视图，
    /// 若源数组 F-contiguous 则保持 F-contiguous 和原 LDA。
    ///
    /// # 示例
    ///
    /// ```
    /// use Senon::{Tensor3, Ix2};
    ///
    /// let a = Tensor3::<f64>::zeros([3, 4, 5]);
    /// let b = a.index_axis(0, 1).unwrap();  // shape: [4, 5]
    /// let c = a.index_axis(2, 2).unwrap();  // shape: [3, 4]
    /// ```
    pub fn index_axis<E>(&self, axis: usize, index: usize) -> Result<TensorView<'_, A, E>, ShapeError>
    where
        E: Dimension,
        D: RemoveAxis,
    {
        let ndim = self.ndim();
        
        // 轴检查
        if axis >= ndim {
            return Err(ShapeError::InvalidAxis(InvalidAxis {
                axis,
                ndim,
                reason: "axis index out of bounds",
            }));
        }
        
        // 索引检查
        if index >= self.shape()[axis] {
            return Err(ShapeError::IndexOutOfBounds(IndexOutOfBounds {
                index,
                size: self.shape()[axis],
                axis: Some(axis),
            }));
        }
        
        // 计算新偏移
        let offset_delta = index * self.strides()[axis] as usize;
        let new_offset = self.offset + offset_delta;
        
        // 构建新形状和步长（移除指定轴）
        let new_shape: SmallVec<[usize; 6]> = self.shape()
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != axis)
            .map(|(_, &s)| s)
            .collect();
        let new_strides: SmallVec<[isize; 6]> = self.strides()
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != axis)
            .map(|(_, &s)| s)
            .collect();
        
        // 更新布局标志
        let new_layout = self.layout.update_for_index_axis(axis, self.shape(), self.strides());
        
        Ok(TensorView {
            storage: ViewRepr::from(&self.storage),
            shape: E::from_slice(&new_shape),
            strides: E::from_isize_slice(&new_strides),
            offset: new_offset,
            layout: new_layout,
        })
    }
    
    /// 沿轴取单个切片（无检查版本）
    ///
    /// # Safety
    ///
    /// 调用者须保证 `axis < ndim` 且 `index < shape[axis]`
    pub unsafe fn index_axis_unchecked<E>(&self, axis: usize, index: usize) -> TensorView<'_, A, E>
    where
        E: Dimension,
    {
        let offset_delta = index * self.strides()[axis] as usize;
        let new_offset = self.offset + offset_delta;
        
        let new_shape: SmallVec<[usize; 6]> = self.shape()
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != axis)
            .map(|(_, &s)| s)
            .collect();
        let new_strides: SmallVec<[isize; 6]> = self.strides()
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != axis)
            .map(|(_, &s)| s)
            .collect();
        
        TensorView {
            storage: ViewRepr::from(&self.storage),
            shape: E::from_slice(&new_shape),
            strides: E::from_isize_slice(&new_strides),
            offset: new_offset,
            layout: self.layout.update_for_index_axis(axis, self.shape(), self.strides()),
        }
    }
}
```

#### 3.8.2 语义表

| 属性 | 行为 |
|------|------|
| 零拷贝 | 始终零拷贝 |
| 维度变化 | ndim 减少 1 |
| 连续性保持 | 沿最外层轴索引时保持连续性 |
| F-contiguous 保持 | 从 F-contiguous 数组沿最外层轴索引后仍 F-contiguous |
| LDA 保持 | BLAS leading dimension 保持不变 |

#### 3.8.3 连续性保持规则

```
假设输入 shape = [B, M, N], F-contiguous, strides = [1, B, B*M]

沿轴 0 索引 (batch 轴):
  - 输出 shape = [M, N], strides = [B, B*M]
  - 保持 F-contiguous: strides[0] = 1? 否，变为非连续
  - 但 LDA = B 保持不变

沿轴 2 索引 (最内层轴):
  - 输出 shape = [B, M], strides = [1, B]
  - 保持 F-contiguous: strides[0] = 1 ✓
  - LDA = B 保持不变

结论：沿 F-order 最内层轴索引保持连续性
```

---

### 3.9 unstack

#### 3.9.1 方法签名

```rust
// split.rs

impl<S, A, D> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// 沿轴拆分为多个降维视图
    ///
    /// 将数组沿指定轴拆分为 n 个视图，n 为该轴长度。
    /// 每个视图的 ndim 比原数组少 1。
    ///
    /// # 参数
    ///
    /// * `axis` - 拆分轴
    ///
    /// # 返回值
    ///
    /// * `Ok(Vec<TensorView>)` - 拆分后的视图列表
    /// * `Err(InvalidAxis)` - 轴索引越界
    ///
    /// # 与 index_axis 的关系
    ///
    /// `unstack(axis)[i]` 等价于 `index_axis(axis, i)`
    ///
    /// # 空轴
    ///
    /// 若轴长度为 0，返回空 Vec
    ///
    /// # 示例
    ///
    /// ```
    /// use Senon::{Tensor3, Ix2};
    ///
    /// let a = Tensor3::<f64>::zeros([3, 4, 5]);
    /// let views = a.unstack(0).unwrap();  // 3 个 [4, 5] 视图
    /// assert_eq!(views.len(), 3);
    /// assert_eq!(views[0].shape(), &[4, 5]);
    /// ```
    pub fn unstack(&self, axis: usize) -> Result<Vec<TensorView<'_, A, <D as RemoveAxis>::Smaller>>, InvalidAxis>
    where
        D: RemoveAxis,
    {
        let ndim = self.ndim();
        
        if axis >= ndim {
            return Err(InvalidAxis {
                axis,
                ndim,
                reason: "axis index out of bounds",
            });
        }
        
        let axis_len = self.shape()[axis];
        let mut views = Vec::with_capacity(axis_len);
        
        for i in 0..axis_len {
            // 使用 index_axis 实现每个切片
            views.push(unsafe {
                self.index_axis_unchecked::<<D as RemoveAxis>::Smaller>(axis, i)
            });
        }
        
        Ok(views)
    }
}
```

#### 3.9.2 语义表

| 属性 | 行为 |
|------|------|
| 零拷贝 | 始终零拷贝 |
| 返回数量 | 等于指定轴的长度 |
| 视图形状 | 原形状移除指定轴 |
| 空轴 | 返回空 Vec |
| 与 index_axis 关系 | `unstack(axis)[i] == index_axis(axis, i)` |

---

### 3.10 split / chunk

#### 3.10.1 方法签名

```rust
// split.rs

impl<S, A, D> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// 按索引列表分割数组
    ///
    /// 沿指定轴按索引列表分割，返回零拷贝视图列表。
    ///
    /// # 参数
    ///
    /// * `axis` - 分割轴
    /// * `indices` - 分割点索引列表
    ///
    /// # 返回值
    ///
    /// `Vec<TensorView>`，长度为 `indices.len() + 1`
    ///
    /// # 示例
    ///
    /// ```
    /// use Senon::{Tensor2, Ix2};
    ///
    /// let a = Tensor2::<f64>::zeros([6, 4]);
    /// let parts = a.split(0, &[2, 4]).unwrap();
    /// // parts[0]: [2, 4], parts[1]: [2, 4], parts[2]: [2, 4]
    /// assert_eq!(parts.len(), 3);
    /// ```
    pub fn split(&self, axis: usize, indices: &[usize]) -> Result<Vec<TensorView<'_, A, D>>, InvalidAxis> {
        let ndim = self.ndim();
        
        if axis >= ndim {
            return Err(InvalidAxis {
                axis,
                ndim,
                reason: "axis index out of bounds",
            });
        }
        
        let axis_len = self.shape()[axis];
        let mut views = Vec::with_capacity(indices.len() + 1);
        
        let mut prev = 0;
        for &idx in indices {
            if idx > axis_len {
                return Err(InvalidAxis {
                    axis: idx,
                    ndim: axis_len,
                    reason: "split index exceeds axis length",
                });
            }
            
            let slice_info = SliceInfo::new(prev, idx, 1);
            views.push(self.slice_axis(axis, slice_info));
            prev = idx;
        }
        
        // 最后一段
        let slice_info = SliceInfo::new(prev, axis_len, 1);
        views.push(self.slice_axis(axis, slice_info));
        
        Ok(views)
    }
    
    /// 均匀分割数组
    ///
    /// 沿指定轴均匀分割为 n 块。
    /// 若轴长度不能整除 n，前面的块各多 1 个元素。
    ///
    /// # 参数
    ///
    /// * `axis` - 分割轴
    /// * `n_chunks` - 分割块数
    ///
    /// # 返回值
    ///
    /// `Vec<TensorView>`
    ///
    /// # 特殊情况
    ///
    /// - `n_chunks = 0`: 返回空 Vec
    /// - `n_chunks > 轴长度`: 返回轴长度个大小为 1 的块
    /// - 空轴: 返回 n_chunks 个空视图
    ///
    /// # 示例
    ///
    /// ```
    /// use Senon::{Tensor1, Ix1};
    ///
    /// let a = Tensor1::<f64>::from_vec(vec![1, 2, 3, 4, 5, 6, 7]);
    ///
    /// // 7 个元素分 3 块: [3, 2, 2]
    /// let chunks = a.chunk(0, 3).unwrap();
    /// assert_eq!(chunks[0].len(), 3);  // [1, 2, 3]
    /// assert_eq!(chunks[1].len(), 2);  // [4, 5]
    /// assert_eq!(chunks[2].len(), 2);  // [6, 7]
    /// ```
    pub fn chunk(&self, axis: usize, n_chunks: usize) -> Result<Vec<TensorView<'_, A, D>>, InvalidAxis> {
        let ndim = self.ndim();
        
        if axis >= ndim {
            return Err(InvalidAxis {
                axis,
                ndim,
                reason: "axis index out of bounds",
            });
        }
        
        if n_chunks == 0 {
            return Ok(Vec::new());
        }
        
        let axis_len = self.shape()[axis];
        
        if n_chunks > axis_len {
            // 每块大小为 1
            let mut views = Vec::with_capacity(axis_len);
            for i in 0..axis_len {
                let slice_info = SliceInfo::new(i, i + 1, 1);
                views.push(self.slice_axis(axis, slice_info));
            }
            return Ok(views);
        }
        
        // 计算分割点
        let base_size = axis_len / n_chunks;
        let remainder = axis_len % n_chunks;
        
        let mut views = Vec::with_capacity(n_chunks);
        let mut start = 0;
        
        for i in 0..n_chunks {
            let chunk_size = base_size + if i < remainder { 1 } else { 0 };
            let end = start + chunk_size;
            
            let slice_info = SliceInfo::new(start, end, 1);
            views.push(self.slice_axis(axis, slice_info));
            
            start = end;
        }
        
        Ok(views)
    }
    
    /// 沿指定轴切片（内部辅助方法）
    fn slice_axis(&self, axis: usize, info: SliceInfo) -> TensorView<'_, A, D> {
        // 构建切片描述符
        let mut slices: Vec<SliceInfo> = self.shape().iter()
            .map(|&s| SliceInfo::new(0, s, 1))
            .collect();
        slices[axis] = info;
        
        self.slice(SliceArgImpl::from_slices(&slices))
    }
}

/// 便捷分割函数

impl<S, A, D> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// 水平分割（沿轴 1）
    pub fn hsplit(&self, indices: &[usize]) -> Result<Vec<TensorView<'_, A, D>>, InvalidAxis> {
        self.split(1, indices)
    }
    
    /// 垂直分割（沿轴 0）
    pub fn vsplit(&self, indices: &[usize]) -> Result<Vec<TensorView<'_, A, D>>, InvalidAxis> {
        self.split(0, indices)
    }
    
    /// 深度分割（沿轴 2，仅 3D+）
    pub fn dsplit(&self, indices: &[usize]) -> Result<Vec<TensorView<'_, A, D>>, InvalidAxis> {
        self.split(2, indices)
    }
}
```

#### 3.10.2 语义表

| 操作 | 输入 | 参数 | 输出 |
|------|------|------|------|
| `split(0, [2,4])` | [6, 4] | ax=0, idx=[2,4] | [[2,4], [2,4], [2,4]] |
| `chunk(0, 3)` | [7] | ax=0, n=3 | [[3], [2], [2]] |
| `chunk(0, 10)` | [5] | ax=0, n=10 | [[1], [1], [1], [1], [1]] |
| `chunk(0, 0)` | [5] | ax=0, n=0 | [] |

---

## 4. 需拷贝操作详细设计

需拷贝操作由于物理内存布局无法满足目标形状，必须创建新的拥有型数组。

### 4.1 cat

#### 4.1.1 方法签名

```rust
// stack.rs

/// 沿轴拼接数组
///
/// 将多个数组沿指定轴拼接为单个数组。需拷贝操作。
///
/// # 参数
///
/// * `tensors` - 要拼接的数组切片
/// * `axis` - 拼接轴
///
/// # 返回值
///
/// * `Ok(Tensor)` - 拼接后的数组
/// * `Err(ShapeMismatch)` - 形状不兼容
/// * `Err(InvalidAxis)` - 轴索引越界
///
/// # 形状约束
///
/// - 所有数组除拼接轴外，其他轴大小须相同
/// - 所有数组维度数须相同
///
/// # 示例
///
/// ```
/// use Senon::{Tensor1, Tensor2, cat};
///
/// let a = Tensor1::from_vec(vec![1, 2, 3]);
/// let b = Tensor1::from_vec(vec![4, 5, 6]);
/// let c = cat(&[a.view(), b.view()], 0).unwrap();
/// // c: [1, 2, 3, 4, 5, 6]
///
/// let d = Tensor2::<f64>::zeros([2, 3]);
/// let e = Tensor2::<f64>::zeros([2, 4]);
/// let f = cat(&[d.view(), e.view()], 1).unwrap();
/// // f: shape [2, 7]
/// ```
pub fn cat<A, D>(tensors: &[TensorView<'_, A, D>], axis: usize) -> Result<Tensor<A, D>, ShapeError>
where
    A: Element + Clone,
    D: Dimension,
{
    if tensors.is_empty() {
        return Err(ShapeError::EmptyInput);
    }
    
    let first = &tensors[0];
    let ndim = first.ndim();
    
    // 轴检查
    if axis >= ndim {
        return Err(ShapeError::InvalidAxis(InvalidAxis {
            axis,
            ndim,
            reason: "axis index out of bounds",
        }));
    }
    
    // 形状检查
    for (i, t) in tensors.iter().enumerate().skip(1) {
        if t.ndim() != ndim {
            return Err(ShapeError::DimensionMismatch {
                expected: ndim,
                actual: t.ndim(),
                context: format!("tensor {}", i),
            });
        }
        
        for (ax, (&s1, &s2)) in first.shape().iter().zip(t.shape().iter()).enumerate() {
            if ax != axis && s1 != s2 {
                return Err(ShapeError::ShapeMismatch {
                    expected: first.shape().to_vec(),
                    actual: t.shape().to_vec(),
                    axis: Some(ax),
                    reason: "shapes must match except in concatenation axis",
                });
            }
        }
    }
    
    // 计算输出形状
    let mut out_shape: SmallVec<[usize; 6]> = first.shape().into();
    out_shape[axis] = tensors.iter().map(|t| t.shape()[axis]).sum();
    
    // 分配输出
    let mut result = Tensor::<A, D>::zeros(D::from_slice(&out_shape));
    
    // 拷贝数据
    let mut offset = 0;
    for t in tensors {
        let len = t.shape()[axis];
        let slice = result.slice_axis_mut(axis, offset..offset + len);
        slice.assign(t);
        offset += len;
    }
    
    Ok(result)
}

/// concatenate 的别名
pub fn concatenate<A, D>(tensors: &[TensorView<'_, A, D>], axis: usize) -> Result<Tensor<A, D>, ShapeError>
where
    A: Element + Clone,
    D: Dimension,
{
    cat(tensors, axis)
}

impl<S, A, D> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// 垂直拼接（沿轴 0）
    pub fn vstack(tensors: &[TensorView<'_, A, D>]) -> Result<Tensor<A, D>, ShapeError>
    where
        A: Element + Clone,
    {
        cat(tensors, 0)
    }
    
    /// 水平拼接（沿轴 1）
    pub fn hstack(tensors: &[TensorView<'_, A, D>]) -> Result<Tensor<A, D>, ShapeError>
    where
        A: Element + Clone,
    {
        cat(tensors, 1)
    }
    
    /// 深度拼接（沿轴 2）
    pub fn dstack(tensors: &[TensorView<'_, A, D>]) -> Result<Tensor<A, D>, ShapeError>
    where
        A: Element + Clone,
    {
        cat(tensors, 2)
    }
}
```

#### 4.1.2 语义表

| 属性 | 行为 |
|------|------|
| 内存分配 | 创建新数组，拷贝所有输入数据 |
| 形状约束 | 除拼接轴外，其他轴须相同 |
| 维度约束 | 所有输入维度数须相同 |
| 空输入 | 返回 `EmptyInput` 错误 |
| 单输入 | 等价于拷贝 |
| 顺序 | 按输入顺序拼接 |

---

### 4.2 stack

#### 4.2.1 方法签名

```rust
// stack.rs

/// 沿新轴堆叠数组
///
/// 将多个数组沿新轴堆叠，维度数增加 1。需拷贝操作。
///
/// # 参数
///
/// * `tensors` - 要堆叠的数组切片
/// * `axis` - 新轴插入位置
///
/// # 返回值
///
/// * `Ok(Tensor)` - 堆叠后的数组（ndim + 1）
/// * `Err(ShapeMismatch)` - 形状不兼容
///
/// # 形状约束
///
/// - 所有数组形状须完全相同
///
/// # 示例
///
/// ```
/// use Senon::{Tensor1, Tensor2, stack};
///
/// let a = Tensor1::from_vec(vec![1, 2, 3]);
/// let b = Tensor1::from_vec(vec![4, 5, 6]);
/// let c = stack(&[a.view(), b.view()], 0).unwrap();
/// // c: shape [2, 3], [[1,2,3], [4,5,6]]
///
/// let d = stack(&[a.view(), b.view()], 1).unwrap();
/// // d: shape [3, 2], [[1,4], [2,5], [3,6]]
/// ```
pub fn stack<A, D>(tensors: &[TensorView<'_, A, D>], axis: usize) -> Result<Tensor<A, D::Larger>, ShapeError>
where
    A: Element + Clone,
    D: Dimension,
    D::Larger: Dimension,
{
    if tensors.is_empty() {
        return Err(ShapeError::EmptyInput);
    }
    
    let first = &tensors[0];
    let ndim = first.ndim();
    
    // 轴检查（允许 axis == ndim，表示追加到末尾）
    if axis > ndim {
        return Err(ShapeError::InvalidAxis(InvalidAxis {
            axis,
            ndim: ndim + 1,  // 新维度数
            reason: "axis index out of bounds",
        }));
    }
    
    // 形状检查
    for (i, t) in tensors.iter().enumerate().skip(1) {
        if t.shape() != first.shape() {
            return Err(ShapeError::ShapeMismatch {
                expected: first.shape().to_vec(),
                actual: t.shape().to_vec(),
                axis: None,
                reason: "all arrays must have the same shape for stacking",
            });
        }
    }
    
    // 计算输出形状
    let mut out_shape: SmallVec<[usize; 7]> = SmallVec::with_capacity(ndim + 1);
    for (i, &s) in first.shape().iter().enumerate() {
        if i == axis {
            out_shape.push(tensors.len());
        }
        out_shape.push(s);
    }
    if axis == ndim {
        out_shape.push(tensors.len());
    }
    
    // 分配输出
    let mut result = Tensor::<A, D::Larger>::zeros(D::Larger::from_slice(&out_shape));
    
    // 拷贝数据
    for (i, t) in tensors.iter().enumerate() {
        let mut slice = result.index_axis_mut(axis, i);
        slice.assign(t);
    }
    
    Ok(result)
}
```

#### 4.2.2 语义表

| 操作 | 输入形状 | axis | 输出形状 |
|------|----------|------|----------|
| `stack(&[a,b], 0)` | [3], [3] | 0 | [2, 3] |
| `stack(&[a,b], 1)` | [3], [3] | 1 | [3, 2] |
| `stack(&[a,b], 2)` | [3], [3] | 2 | [3, 2]（等同于 1） |
| `stack(&[a,b,c], 0)` | [4, 5], [4, 5], [4, 5] | 0 | [3, 4, 5] |

---

### 4.3 pad

#### 4.3.1 方法签名

```rust
// pad.rs

/// 填充模式
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadMode<A> {
    /// 常量填充
    ///
    /// 使用指定常量值填充边界
    Constant(A),
    
    /// 边缘填充
    ///
    /// 使用边缘元素值填充
    Edge,
    
    /// 镜像反射填充
    ///
    /// 镜像反射边界元素（不含边缘元素本身）
    Reflect,
}

/// 填充宽度配置
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PadWidth {
    /// 前填充宽度
    pub before: usize,
    /// 后填充宽度
    pub after: usize,
}

impl PadWidth {
    pub fn new(before: usize, after: usize) -> Self {
        Self { before, after }
    }
    
    pub fn symmetric(width: usize) -> Self {
        Self { before: width, after: width }
    }
}

impl<S, A, D> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + Clone,
{
    /// 边界填充
    ///
    /// 沿各轴两侧填充，返回新的拥有型数组。需拷贝操作。
    ///
    /// # 参数
    ///
    /// * `widths` - 每轴的填充宽度，长度可小于 ndim（剩余轴不填充）
    /// * `mode` - 填充模式
    ///
    /// # 返回值
    ///
    /// 填充后的 `Tensor<A, D>`
    ///
    /// # 示例
    ///
    /// ```
    /// use Senon::{Tensor1, PadMode, PadWidth};
    ///
    /// let a = Tensor1::from_vec(vec![1, 2, 3]);
    ///
    /// // 常量填充
    /// let b = a.pad(&[PadWidth::symmetric(1)], PadMode::Constant(0));
    /// // b: [0, 1, 2, 3, 0]
    ///
    /// // 边缘填充
    /// let c = a.pad(&[PadWidth::symmetric(1)], PadMode::Edge);
    /// // c: [1, 1, 2, 3, 3]
    ///
    /// // 反射填充
    /// let d = a.pad(&[PadWidth::symmetric(1)], PadMode::Reflect);
    /// // d: [2, 1, 2, 3, 2]
    /// ```
    pub fn pad(&self, widths: &[PadWidth], mode: PadMode<A>) -> Tensor<A, D>
    where
        A: Clone,
    {
        let ndim = self.ndim();
        
        // 计算输出形状
        let mut out_shape: SmallVec<[usize; 6]> = SmallVec::with_capacity(ndim);
        for (i, &s) in self.shape().iter().enumerate() {
            let w = widths.get(i).copied().unwrap_or(PadWidth::symmetric(0));
            out_shape.push(s + w.before + w.after);
        }
        
        // 分配输出
        let mut result = match &mode {
            PadMode::Constant(val) => Tensor::full(D::from_slice(&out_shape), val.clone()),
            _ => Tensor::zeros(D::from_slice(&out_shape)),
        };
        
        // 计算数据区域的起始偏移
        let mut data_offset: SmallVec<[usize; 6]> = SmallVec::with_capacity(ndim);
        for (i, _) in self.shape().iter().enumerate() {
            let w = widths.get(i).copied().unwrap_or(PadWidth::symmetric(0));
            data_offset.push(w.before);
        }
        
        // 拷贝原始数据到中心区域
        {
            let mut center = result.slice_at_offsets(&data_offset, self.shape());
            center.assign(self);
        }
        
        // 填充边界
        match mode {
            PadMode::Constant(_) => {
                // 已通过 full 初始化，无需额外操作
            }
            PadMode::Edge => {
                self.pad_edge(&mut result, widths);
            }
            PadMode::Reflect => {
                self.pad_reflect(&mut result, widths);
            }
        }
        
        result
    }
    
    /// 边缘填充实现
    fn pad_edge(&self, result: &mut Tensor<A, D>, widths: &[PadWidth]) {
        let ndim = self.ndim();
        
        // 沿每个轴填充
        for axis in 0..ndim {
            let w = widths.get(axis).copied().unwrap_or(PadWidth::symmetric(0));
            if w.before == 0 && w.after == 0 {
                continue;
            }
            
            // 前填充：复制边缘行
            for i in 0..w.before {
                let src_idx = 0;  // 边缘索引
                let dst_idx = w.before - 1 - i;
                self.copy_along_axis(result, axis, src_idx, dst_idx);
            }
            
            // 后填充：复制边缘行
            for i in 0..w.after {
                let src_idx = self.shape()[axis] - 1;  // 边缘索引
                let dst_idx = w.before + self.shape()[axis] + i;
                self.copy_along_axis(result, axis, src_idx, dst_idx);
            }
        }
    }
    
    /// 反射填充实现
    fn pad_reflect(&self, result: &mut Tensor<A, D>, widths: &[PadWidth]) {
        let ndim = self.ndim();
        
        for axis in 0..ndim {
            let w = widths.get(axis).copied().unwrap_or(PadWidth::symmetric(0));
            if w.before == 0 && w.after == 0 {
                continue;
            }
            
            // 前填充：镜像反射（不含边缘）
            for i in 0..w.before {
                let src_idx = w.before - i;  // 从边缘下一个开始反射
                let dst_idx = w.before - 1 - i;
                self.copy_along_axis(result, axis, src_idx, dst_idx);
            }
            
            // 后填充：镜像反射（不含边缘）
            for i in 0..w.after {
                let src_idx = w.before + self.shape()[axis] - 2 - i;
                let dst_idx = w.before + self.shape()[axis] + i;
                self.copy_along_axis(result, axis, src_idx, dst_idx);
            }
        }
    }
    
    /// 沿轴复制切片
    fn copy_along_axis(&self, result: &mut Tensor<A, D>, axis: usize, src_idx: usize, dst_idx: usize) {
        // 实现细节...
    }
}
```

#### 4.3.2 语义表

| 模式 | 边界行为 | 示例（输入 [1,2,3], 填充宽度 1） |
|------|----------|--------------------------------|
| `Constant(0)` | 用 0 填充 | [0, 1, 2, 3, 0] |
| `Constant(-1)` | 用 -1 填充 | [-1, 1, 2, 3, -1] |
| `Edge` | 复制边缘值 | [1, 1, 2, 3, 3] |
| `Reflect` | 镜像反射（不含边缘） | [2, 1, 2, 3, 2] |

#### 4.3.3 零宽度填充

```rust
// 零宽度填充等价于拷贝
let a = Tensor1::from_vec(vec![1, 2, 3]);
let b = a.pad(&[PadWidth::symmetric(0)], PadMode::Constant(0));
// b: [1, 2, 3] (与 a 相同，但是新分配)
```

---

### 4.4 repeat / tile

#### 4.4.1 方法签名

```rust
// repeat.rs

impl<S, A, D> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// 沿各轴重复元素
    ///
    /// 返回新数组，每个元素沿对应轴重复指定次数。需拷贝操作。
    ///
    /// # 参数
    ///
    /// * `reps` - 每轴重复次数，长度可小于 ndim（左侧补 1）
    ///
    /// # 返回值
    ///
    /// 重复后的 `Tensor<A, D>`
    ///
    /// # 特殊情况
    ///
    /// - `reps` 长度 < ndim: 左侧补 1（与 NumPy np.tile 一致）
    /// - `reps` 含 0: 对应轴长度变为 0，结果为空数组
    /// - `reps` 全为 1: 等价于拷贝
    ///
    /// # 示例
    ///
    /// ```
    /// use Senon::{Tensor1, Tensor2};
    ///
    /// let a = Tensor1::from_vec(vec![1, 2, 3]);
    ///
    /// let b = a.repeat(&[2]);  // [1, 1, 2, 2, 3, 3]
    ///
    /// let c = Tensor2::from_shape_vec([2, 2], vec![1, 2, 3, 4]);
    /// let d = c.repeat(&[2, 3]);
    /// // shape: [4, 6]
    /// // [[1,1,1,2,2,2],
    /// //  [1,1,1,2,2,2],
    /// //  [3,3,3,4,4,4],
    /// //  [3,3,3,4,4,4]]
    /// ```
    pub fn repeat(&self, reps: &[usize]) -> Tensor<A, D>
    where
        A: Clone,
    {
        let ndim = self.ndim();
        
        // 左侧补 1
        let full_reps: SmallVec<[usize; 6]> = if reps.len() < ndim {
            let padding = ndim - reps.len();
            (0..padding).map(|_| 1)
                .chain(reps.iter().copied())
                .collect()
        } else {
            reps[..ndim].iter().copied().collect()
        };
        
        // 计算输出形状
        let out_shape: SmallVec<[usize; 6]> = self.shape().iter()
            .zip(full_reps.iter())
            .map(|(&s, &r)| s * r)
            .collect();
        
        // 检查空数组
        if out_shape.iter().any(|&s| s == 0) {
            return Tensor::zeros(D::from_slice(&out_shape));
        }
        
        // 分配输出
        let mut result = Tensor::zeros(D::from_slice(&out_shape));
        
        // 填充数据
        for (out_idx, elem) in result.indexed_iter_mut() {
            // 计算对应的输入索引
            let in_idx: SmallVec<[usize; 6]> = out_shape.iter()
                .zip(self.shape().iter())
                .zip(out_idx.iter())
                .map(|((&os, &is), &oi)| oi % is)
                .collect();
            
            *elem = self[&in_idx].clone();
        }
        
        result
    }
    
    /// tile 的别名，与 NumPy np.tile 一致
    pub fn tile(&self, reps: &[usize]) -> Tensor<A, D>
    where
        A: Clone,
    {
        self.repeat(reps)
    }
}
```

#### 4.4.2 语义表

| 输入形状 | reps | 输出形状 | 说明 |
|----------|------|----------|------|
| [3] | [2] | [6] | 每元素重复 2 次 |
| [2, 2] | [2, 3] | [4, 6] | 整块重复 |
| [2, 3] | [1] | [2, 3] | 左侧补 1，无变化 |
| [2, 3] | [2] | [2, 6] | 左侧补 1 |
| [2, 3] | [0, 2] | [0, 6] | 空数组 |
| [2, 3] | [1, 1] | [2, 3] | 等价于拷贝 |

---

## 5. 与其他模块的交互

### 5.1 模块依赖图

```
                        ┌─────────────────────────────────────────┐
                        │            shape_ops 模块                │
                        │  reshape | transpose | slice | squeeze  │
                        │  split | stack | pad | repeat           │
                        └──────────────────┬──────────────────────┘
                                           │
          ┌────────────────┬───────────────┼───────────────┬────────────────┐
          │                │               │               │                │
          ▼                ▼               ▼               ▼                ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │  tensor  │    │  layout  │    │dimension │    │  error   │    │broadcast │
    │TensorBase│    │LayoutFlags│   │Dimension │    │ShapeError│    │can_      │
    │Storage   │    │Strides   │    │Ix0-Ix6   │    │InvalidAxis│   │broadcast │
    └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘
         │               │               │               │               │
         │               │               │               │               │
         └───────────────┴───────────────┴───────────────┴───────────────┘
                                           │
                                           ▼
                                    ┌──────────┐
                                    │  element │
                                    │Element   │
                                    │Numeric   │
                                    └──────────┘
```

### 5.2 与 tensor 模块的接口

| 接口 | 用途 |
|------|------|
| `TensorBase::shape()` | 获取当前形状 |
| `TensorBase::strides()` | 获取当前步长 |
| `TensorBase::offset()` | 获取数据偏移 |
| `TensorBase::storage` | 访问底层存储 |
| `TensorBase::layout` | 获取布局标志 |
| `TensorBase::len()` | 获取元素总数 |
| `TensorBase::ndim()` | 获取维度数 |

### 5.3 与 layout 模块的接口

| 接口 | 用途 |
|------|------|
| `LayoutFlags::is_f_contiguous()` | 检查 F-order 连续性 |
| `LayoutFlags::is_c_contiguous()` | 检查 C-order 连续性 |
| `LayoutFlags::is_contiguous()` | 检查任一方向连续性 |
| `LayoutFlags::update_for_transpose()` | 转置后更新标志 |
| `LayoutFlags::update_for_slice()` | 切片后更新标志 |
| `Strides::from_shape()` | 从形状计算步长 |

### 5.4 与 dimension 模块的接口

| 接口 | 用途 |
|------|------|
| `Dimension::slice()` | 获取形状切片 |
| `Dimension::from_slice()` | 从切片构造维度 |
| `RemoveAxis::Smaller` | 移除轴后的维度类型 |
| `Dimension::Larger` | 增加轴后的维度类型 |
| `Dimension::reverse()` | 反转维度顺序 |

### 5.5 与 error 模块的接口

| 错误类型 | 触发场景 |
|----------|----------|
| `InvalidShape` | reshape 目标元素总数不匹配 |
| `LayoutMismatch` | reshape 非连续数组 |
| `InvalidAxis` | 轴索引越界 |
| `IndexOutOfBounds` | 索引越界 |
| `ShapeMismatch` | cat/stack 形状不兼容 |
| `EmptyInput` | cat/stack 空输入 |

### 5.6 与 broadcast 模块的接口

| 接口 | 用途 |
|------|------|
| `can_broadcast()` | 检查形状是否可广播 |
| `broadcast_shape()` | 计算广播后形状 |

---

## 6. 实现任务分解

### 6.1 任务清单

| # | 任务 | 文件 | 预估时间 | 依赖 |
|---|------|------|----------|------|
| 1 | 定义 `ShapeError` 错误类型 | `error.rs` | 15 min | 无 |
| 2 | 实现 `SliceInfo` 和 `s![]` 宏 | `slice.rs` | 30 min | T1 |
| 3 | 实现 `reshape` 和 `into_shape` | `reshape.rs` | 30 min | T1, layout |
| 4 | 实现 `flatten` | `reshape.rs` | 15 min | T3 |
| 5 | 实现 `transpose` 和 `t()` | `transpose.rs` | 20 min | T1, dimension |
| 6 | 实现 `permute` | `transpose.rs` | 20 min | T5 |
| 7 | 实现 `swapaxes` 和 `moveaxis` | `transpose.rs` | 15 min | T6 |
| 8 | 实现 `slice` 和 `slice_mut` | `slice.rs` | 30 min | T2 |
| 9 | 实现 `index_axis` | `slice.rs` | 20 min | T8 |
| 10 | 实现 `squeeze` 和 `unsqueeze` | `squeeze.rs` | 25 min | T1, dimension |
| 11 | 实现 `split` 和 `chunk` | `split.rs` | 25 min | T8 |
| 12 | 实现 `unstack` | `split.rs` | 15 min | T9 |
| 13 | 实现 `cat` 和 `concatenate` | `stack.rs` | 25 min | T1, T8 |
| 14 | 实现 `stack` | `stack.rs` | 20 min | T13 |
| 15 | 实现 `PadMode` 和 `pad` | `pad.rs` | 35 min | T1, element |
| 16 | 实现 `repeat` 和 `tile` | `repeat.rs` | 20 min | T1, element |
| 17 | 实现便捷函数 (vstack, hstack 等) | `stack.rs` | 10 min | T13, T14 |
| 18 | 编写单元测试 | `tests/shape_ops.rs` | 60 min | T1-T17 |
| 19 | 编写文档注释 | 所有文件 | 30 min | T1-T17 |

**总预估时间**: 约 7 小时

### 6.2 任务依赖图

```
T1 (ShapeError) ─────┬────────────────────────────────────────────────────┐
                      │                                                    │
    ┌─────────────────┼─────────────────┬─────────────────┐              │
    │                 │                 │                 │              │
    ▼                 ▼                 ▼                 ▼              │
T2 (SliceInfo)   T3 (reshape)      T5 (transpose)    T10 (squeeze)   T15 (pad)
    │                 │                 │                 │              │
    │                 │                 │                 │              │
    ▼                 ▼                 ▼                 ▼              │
T8 (slice) ◄───── T4 (flatten)    T6 (permute)          │              │
    │                                   │                 │              │
    │                                   │                 │              │
    ├───────────────┬───────────────────┤                 │              │
    │               │                   │                 │              │
    ▼               ▼                   ▼                 │              │
T9 (index_axis) T11 (split)       T7 (swap/move)         │              │
    │               │                                       │              │
    │               │                                       │              │
    ▼               │                                       │              │
T12 (unstack) ◄────┘                                       │              │
                                                            │              │
T13 (cat) ◄─────────────────────────────────────────────────┘              │
    │                                                                       │
    │                                                                       │
    ├──────────────────┐                                                    │
    │                  │                                                    │
    ▼                  ▼                                                    │
T14 (stack)       T17 (便捷函数)                                            │
                                                                            │
T16 (repeat) ◄─────────────────────────────────────────────────────────────┘
                                                                            │
                    ┌───────────────────────────────────────────────────────┘
                    │
                    ▼
              T18 (测试) ──→ T19 (文档)
```

### 6.3 并行执行建议

- **Wave 1**: T1（可独立开始）
- **Wave 2**: T2, T3, T5, T10, T15, T16（依赖 T1，可并行）
- **Wave 3**: T4, T6, T8（依赖 Wave 2）
- **Wave 4**: T7, T9, T11（依赖 Wave 3）
- **Wave 5**: T12, T13（依赖 Wave 4）
- **Wave 6**: T14, T17（依赖 Wave 5）
- **Wave 7**: T18（依赖所有前置任务）
- **Wave 8**: T19（依赖 T18）

**预估总工期**: 
- 单线程：约 7 小时
- 4 线程并行：约 3 小时

---

## 7. 设计决策记录

### D1: 为什么 reshape 要求连续性？

**决策**: `reshape()` 仅对连续数组零拷贝执行，非连续数组返回 `Err(LayoutMismatch)`。

**理由**:
1. **语义正确性**: 非连续数组的元素在内存中非顺序排列，直接调整步长无法保证逻辑顺序正确
2. **NumPy 一致**: NumPy 的 `reshape` 同样要求连续性
3. **性能可预测**: 显式错误比隐式拷贝更安全，用户可选择 `into_shape()` 自动处理

**替代方案**: 非连续数组自动拷贝。
**拒绝原因**: 隐藏性能陷阱，用户可能无意中触发昂贵的拷贝操作。

### D2: 为什么 transpose 总是零拷贝？

**决策**: `transpose()` 对任何数组都零拷贝执行。

**理由**:
1. **数学正确性**: 转置仅交换步长，数学上总是正确
2. **性能优势**: 无需任何数据移动
3. **链式操作**: 允许多次转置而不产生多次拷贝

**替代方案**: 转置后检查连续性，若需连续则拷贝。
**拒绝原因**: `transpose` 语义是"逻辑转置"而非"物理转置"，后者由 `to_f_contiguous()` 等方法处理。

### D3: 为什么 flatten 是条件零拷贝？

**决策**: `flatten()` 在数组连续时零拷贝，否则拷贝。

**理由**:
1. **语义要求**: flatten 输出必须是一维连续数组
2. **用户期望**: 用户通常期望 flatten 后可直接用于需要连续内存的 API
3. **灵活性**: 提供 `order` 参数允许选择 F/C 顺序

**替代方案**: 始终拷贝。
**拒绝原因**: 连续输入时拷贝是浪费。

### D4: 为什么 cat/stack 需要拷贝？

**决策**: `cat()` 和 `stack()` 始终创建新数组。

**理由**:
1. **物理布局**: 多个数组无法共享同一连续内存块
2. **所有权清晰**: 输出拥有数据，避免生命周期复杂性
3. **BLAS 兼容**: 输出保证连续，可安全传递给 BLAS

**替代方案**: 使用引用计数共享输入数据。
**拒绝原因**: 输出将是非连续的，且生命周期管理复杂。

### D5: 为什么 slice 使用 s![] 宏？

**决策**: 切片操作通过 `s![]` 宏创建切片描述符。

**理由**:
1. **语法简洁**: `s![1..4, 2..6]` 比构建结构体更直观
2. **编译时检查**: 宏可在编译时验证语法
3. **与 ndarray 兼容**: 用户熟悉的 API 风格

**替代方案**: 使用 builder 模式或函数调用。
**拒绝原因**: 语法冗长，不符合 Rust 惯例。

### D6: 为什么 pad 有三种模式？

**决策**: 支持 `Constant`、`Edge`、`Reflect` 三种填充模式。

**理由**:
1. **Constant**: 最常用，适用于一般边界条件
2. **Edge**: 信号处理常用，避免边界突变
3. **Reflect**: 图像处理常用，保持边界连续性

**替代方案**: 仅支持 Constant。
**拒绝原因**: 无法满足科学计算和图像处理的常见需求。

### D7: 为什么 repeat 支持左侧补 1？

**决策**: `repeat(&[2])` 对 shape=[2,3] 的数组等价于 `repeat(&[1, 2])`。

**理由**:
1. **NumPy 兼容**: `np.tile` 行为相同
2. **便捷性**: 常见场景只需指定后几轴的重复次数
3. **一致性**: 与广播规则的对齐方式一致

**替代方案**: 要求 reps 长度等于 ndim。
**拒绝原因**: API 冗长，用户体验差。

### D8: 为什么 index_axis 返回视图而非拷贝？

**决策**: `index_axis()` 返回 `TensorView`，零拷贝。

**理由**:
1. **性能**: 避免不必要的拷贝
2. **一致性**: 与 `slice` 语义一致
3. **BLAS 兼容**: 从 F-contiguous 数组沿最外层轴索引后，2D 视图仍可与 BLAS 互操作（通过 LDA）

**替代方案**: 返回拥有型数组。
**拒绝原因**: 性能损失，且视图语义更通用。

---

## 附录 A: API 速查表

### A.1 零拷贝操作

| 方法 | 签名 | 说明 |
|------|------|------|
| `reshape` | `fn reshape<E>(&self, shape: E) -> Result<TensorView<'_, A, E>, ShapeError>` | 重塑形状 |
| `into_shape` | `fn into_shape<E>(self, shape: E) -> Result<Tensor<A, E>, ShapeError>` | 消费并重塑 |
| `transpose` | `fn transpose(&self) -> TensorView<'_, A, D>` | 转置 |
| `t` | `fn t(&self) -> TensorView<'_, A, D>` | 简写转置 |
| `permute` | `fn permute<E>(&self, axes: E) -> Result<TensorView<'_, A, D>, InvalidAxis>` | 轴重排 |
| `swapaxes` | `fn swapaxes(&self, ax1: usize, ax2: usize) -> Result<TensorView<'_, A, D>, InvalidAxis>` | 交换轴 |
| `moveaxis` | `fn moveaxis(&self, source: usize, dest: usize) -> Result<TensorView<'_, A, D>, InvalidAxis>` | 移动轴 |
| `slice` | `fn slice<I>(&self, info: I) -> TensorView<'_, A, I::OutputDim>` | 切片 |
| `slice_mut` | `fn slice_mut<I>(&mut self, info: I) -> TensorViewMut<'_, A, I::OutputDim>` | 可变切片 |
| `squeeze` | `fn squeeze<E>(&self, axis: Option<usize>) -> Result<TensorView<'_, A, E>, InvalidAxis>` | 压缩轴 |
| `unsqueeze` | `fn unsqueeze<E>(&self, axis: usize) -> TensorView<'_, A, E>` | 扩展轴 |
| `index_axis` | `fn index_axis<E>(&self, axis: usize, index: usize) -> Result<TensorView<'_, A, E>, ShapeError>` | 沿轴索引 |
| `split` | `fn split(&self, axis: usize, indices: &[usize]) -> Result<Vec<TensorView<'_, A, D>>, InvalidAxis>` | 分割 |
| `chunk` | `fn chunk(&self, axis: usize, n: usize) -> Result<Vec<TensorView<'_, A, D>>, InvalidAxis>` | 均匀分割 |
| `unstack` | `fn unstack(&self, axis: usize) -> Result<Vec<TensorView<'_, A, D::Smaller>>, InvalidAxis>` | 拆分 |

### A.2 需拷贝操作

| 方法 | 签名 | 说明 |
|------|------|------|
| `flatten` | `fn flatten(&self, order: Option<MemoryOrder>) -> CowTensor<'_, A, Ix1>` | 展平 |
| `cat` | `fn cat<A, D>(tensors: &[TensorView<'_, A, D>], axis: usize) -> Result<Tensor<A, D>, ShapeError>` | 拼接 |
| `stack` | `fn stack<A, D>(tensors: &[TensorView<'_, A, D>], axis: usize) -> Result<Tensor<A, D::Larger>, ShapeError>` | 堆叠 |
| `pad` | `fn pad(&self, widths: &[PadWidth], mode: PadMode<A>) -> Tensor<A, D>` | 填充 |
| `repeat` | `fn repeat(&self, reps: &[usize]) -> Tensor<A, D>` | 重复 |

---

## 附录 B: 错误类型表

| 错误类型 | 字段 | 触发场景 | 示例消息 |
|----------|------|----------|----------|
| `InvalidShape` | `expected: usize, actual: usize, reason: String` | reshape 元素数不匹配 | `"reshape: element count mismatch: expected 24, actual 20"` |
| `LayoutMismatch` | `reason: String` | reshape 非连续数组 | `"reshape requires contiguous array"` |
| `InvalidAxis` | `axis: usize, ndim: usize, reason: String` | 轴索引越界 | `"axis 5 out of bounds for 3-dimensional array"` |
| `IndexOutOfBounds` | `index: usize, size: usize, axis: Option<usize>` | 索引越界 | `"index 10 out of bounds for axis 1 with size 5"` |
| `ShapeMismatch` | `expected: Vec<usize>, actual: Vec<usize>, axis: Option<usize>, reason: String` | cat/stack 形状不兼容 | `"shapes [2,3] and [2,4] incompatible for concatenation at axis 1"` |
| `EmptyInput` | - | cat/stack 空输入 | `"cannot concatenate empty array list"` |
| `DimensionMismatch` | `expected: usize, actual: usize` | 维度数不匹配 | `"dimension count mismatch: expected 3, actual 2"` |

---

## 附录 C: 性能建议

### C.1 零拷贝优先

| 操作 | 零拷贝条件 | 非零拷贝替代 |
|------|-----------|-------------|
| `reshape` | 数组连续 | `into_shape()` |
| `flatten` | 数组连续 | 自动拷贝 |
| `transpose` | 总是 | N/A |
| `slice` | 总是 | N/A |
| `squeeze/unsqueeze` | 总是 | N/A |

### C.2 避免不必要的连续化

```rust
// 不推荐：不必要的连续化
let a = tensor.to_f_contiguous().reshape([10, 20]);

// 推荐：直接 reshape（若原数组连续）
if tensor.is_contiguous() {
    let a = tensor.reshape([10, 20])?;
}

// 或使用 into_shape 自动处理
let a = tensor.into_shape([10, 20])?;
```

### C.3 批量操作优于循环

```rust
// 不推荐：循环拼接
let mut result = tensors[0].clone();
for t in &tensors[1..] {
    result = cat(&[result.view(), t.view()], 0)?;
}

// 推荐：批量拼接
let views: Vec<_> = tensors.iter().map(|t| t.view()).collect();
let result = cat(&views, 0)?;
```

### C.4 切片链式操作

```rust
// 推荐：链式切片（零拷贝）
let view = tensor.slice(s![1..10, 2..8]).transpose().slice(s![.., 0..3]);

// 注意：链式操作可能产生非连续视图
// 需要连续时显式调用 to_f_contiguous()
```

---

*本文档由 Senon 项目维护。如有问题请提交 Issue 或 PR。*

---

## 6. Squeeze / Unsqueeze 设计

### 6.1 函数签名

```rust
impl<S, D> TensorBase<S, D>
where
    S: Storage,
    D: Dimension,
{
    /// Remove all size-1 dimensions.
    /// Returns a view with reduced ndim. Output dimension type is IxDyn.
    pub fn squeeze(&self) -> TensorView<'_, S::Elem, IxDyn> { ... }

    /// Remove a specific size-1 dimension.
    /// Returns `InvalidAxis` if axis >= ndim.
    /// Returns `InvalidShape` if shape[axis] != 1.
    pub fn squeeze_axis(&self, axis: usize) -> Result<TensorView<'_, S::Elem, IxDyn>> { ... }

    /// Insert a size-1 dimension at the specified axis position.
    /// Returns `InvalidAxis` if axis > ndim (note: axis == ndim is valid).
    pub fn unsqueeze(&self, axis: usize) -> Result<TensorView<'_, S::Elem, IxDyn>> { ... }
}
```

### 6.2 算法

```
function squeeze(shape, strides) -> (new_shape, new_strides):
    new_shape = []
    new_strides = []
    for i in 0..ndim:
        if shape[i] != 1:
            new_shape.push(shape[i])
            new_strides.push(strides[i])
    // If all dimensions are 1, result is 0-dim (scalar)
    return (new_shape, new_strides)

function unsqueeze(shape, strides, axis) -> (new_shape, new_strides):
    assert axis <= ndim
    new_shape = shape.insert(axis, 1)
    // Stride for size-1 dim doesn't matter for data access,
    // but set to a reasonable value for contiguity calculation.
    // Use the stride that would make it contiguous in F-order context.
    if axis == 0:
        new_stride = 1
    else:
        new_stride = strides[axis - 1] * shape[axis - 1]
    new_strides = strides.insert(axis, new_stride)
    return (new_shape, new_strides)
```

### 6.3 布局标志更新

| 操作 | F_CONTIGUOUS | C_CONTIGUOUS | ALIGNED | HAS_ZERO_STRIDE | HAS_NEG_STRIDE |
|------|-------------|-------------|---------|-----------------|----------------|
| squeeze | 重新计算 | 重新计算 | 继承 | 重新计算（移除的轴可能有零步长） | 重新计算 |
| unsqueeze | 重新计算 | 重新计算 | 继承 | 继承 | 继承 |

---

## 7. Index_axis / Unstack 设计

### 7.1 函数签名

```rust
impl<S, D> TensorBase<S, D>
where
    S: Storage,
    D: Dimension,
{
    /// Select a single slice along the specified axis, reducing ndim by 1.
    ///
    /// # BLAS Compatibility
    /// For a 3D F-contiguous batch tensor, indexing along the outermost
    /// batch axis (axis=2 in F-order) returns a 2D F-contiguous view
    /// with the original LDA preserved.
    ///
    /// # Errors
    /// - `InvalidAxis` if axis >= ndim
    ///
    /// # Panics
    /// - Panics if index >= shape[axis]
    pub fn index_axis(&self, axis: usize, index: usize) -> Result<TensorView<'_, S::Elem, D::Smaller>> { ... }

    /// Mutable version of index_axis.
    pub fn index_axis_mut(&mut self, axis: usize, index: usize) -> Result<TensorViewMut<'_, S::Elem, D::Smaller>>
    where
        S: StorageMut,
    { ... }

    /// Split the tensor along the specified axis into individual slices.
    /// Returns Vec of views, each with ndim - 1.
    /// Equivalent to calling index_axis for each index along the axis.
    ///
    /// # Errors
    /// - `InvalidAxis` if axis >= ndim
    pub fn unstack(&self, axis: usize) -> Result<Vec<TensorView<'_, S::Elem, D::Smaller>>> { ... }
}
```

### 7.2 算法

```
function index_axis(shape, strides, offset, axis, index) -> (new_shape, new_strides, new_offset):
    assert axis < ndim
    assert index < shape[axis]

    // Update offset: jump to the selected index along the axis
    new_offset = offset + index * strides[axis]

    // Remove the indexed axis from shape and strides
    new_shape = shape.remove(axis)
    new_strides = strides.remove(axis)

    return (new_shape, new_strides, new_offset)

function unstack(shape, strides, offset, axis) -> Vec<(new_shape, new_strides, new_offset)>:
    assert axis < ndim
    result = []
    for i in 0..shape[axis]:
        result.push(index_axis(shape, strides, offset, axis, i))
    return result
```

### 7.3 BLAS 兼容性分析

对于 3D F-contiguous 张量 `shape=[M, N, B]`, `strides=[1, M, M*N]`：

```
index_axis(axis=2, index=k):
    new_offset = offset + k * M * N
    new_shape = [M, N]
    new_strides = [1, M]
    → F-contiguous 2D view, LDA = M ✓

index_axis(axis=0, index=k):
    new_offset = offset + k * 1
    new_shape = [N, B]
    new_strides = [M, M*N]
    → NOT contiguous (stride[0] = M ≠ 1)
```

### 7.4 布局标志更新

| 操作 | F_CONTIGUOUS | C_CONTIGUOUS | ALIGNED | HAS_ZERO_STRIDE | HAS_NEG_STRIDE |
|------|-------------|-------------|---------|-----------------|----------------|
| index_axis (最外层轴, F-order) | 继承 | 重新计算 | 检查新 offset | 重新计算 | 重新计算 |
| index_axis (内层轴) | 重新计算 | 重新计算 | 检查新 offset | 重新计算 | 重新计算 |
| unstack | 同 index_axis | 同 index_axis | 同 index_axis | 同 index_axis | 同 index_axis |

---

## 8. Split / Chunk 设计

### 8.1 函数签名

```rust
impl<S, D> TensorBase<S, D>
where
    S: Storage,
    D: Dimension,
{
    /// Split the tensor along the specified axis at the given indices.
    /// Returns Vec of views (zero-copy).
    ///
    /// `indices` specifies split points: [2, 5] splits into [0..2], [2..5], [5..end].
    ///
    /// # Errors
    /// - `InvalidAxis` if axis >= ndim
    pub fn split(&self, axis: usize, indices: &[usize]) -> Result<Vec<TensorView<'_, S::Elem, D>>> { ... }

    /// Split the tensor into n approximately equal chunks along the specified axis.
    ///
    /// If the axis length is not evenly divisible, the first (len % n_chunks)
    /// chunks each have one extra element.
    ///
    /// # Edge cases
    /// - n_chunks = 0: returns empty Vec
    /// - n_chunks > axis length: returns axis_length chunks of size 1
    /// - axis length = 0: returns n_chunks empty views
    ///
    /// # Errors
    /// - `InvalidAxis` if axis >= ndim
    pub fn chunk(&self, axis: usize, n_chunks: usize) -> Result<Vec<TensorView<'_, S::Elem, D>>> { ... }
}
```

### 8.2 算法

```
function split(shape, strides, offset, axis, indices) -> Vec<view>:
    assert axis < ndim
    // Build split ranges
    points = [0] + sorted(indices) + [shape[axis]]
    result = []
    for i in 0..points.len()-1:
        start = points[i]
        end = points[i+1]
        if start >= end: continue  // skip empty ranges
        // Create view for this segment
        seg_offset = offset + start * strides[axis]
        seg_shape = shape.clone()
        seg_shape[axis] = end - start
        result.push((seg_shape, strides.clone(), seg_offset))
    return result

function chunk(shape, strides, offset, axis, n_chunks) -> Vec<view>:
    assert axis < ndim
    if n_chunks == 0: return []
    axis_len = shape[axis]
    if axis_len == 0:
        return [empty_view] * n_chunks

    actual_chunks = min(n_chunks, axis_len)
    base_size = axis_len / actual_chunks
    remainder = axis_len % actual_chunks

    result = []
    pos = 0
    for i in 0..actual_chunks:
        chunk_size = base_size + (1 if i < remainder else 0)
        seg_offset = offset + pos * strides[axis]
        seg_shape = shape.clone()
        seg_shape[axis] = chunk_size
        result.push((seg_shape, strides.clone(), seg_offset))
        pos += chunk_size
    return result
```

### 8.3 布局标志更新

| 操作 | F_CONTIGUOUS | C_CONTIGUOUS | ALIGNED | HAS_ZERO_STRIDE | HAS_NEG_STRIDE |
|------|-------------|-------------|---------|-----------------|----------------|
| split/chunk | 重新计算 | 重新计算 | 检查每段 offset | 继承 | 继承 |

---

## 9. Cat / Stack 设计

### 9.1 函数签名

```rust
/// Concatenate tensors along an existing axis.
/// All tensors must have the same shape except along the concatenation axis.
/// Returns a new Owned tensor (data copy).
///
/// # Errors
/// - `ShapeMismatch` if shapes differ on non-concatenation axes
/// - `InvalidAxis` if axis >= ndim
pub fn cat<A, D>(tensors: &[TensorView<'_, A, D>], axis: usize) -> Result<Tensor<A, D>>
where
    A: Element,
    D: Dimension,
{ ... }

/// Stack tensors along a new axis.
/// All tensors must have the same shape.
/// Returns a new Owned tensor with ndim + 1.
///
/// # Errors
/// - `ShapeMismatch` if any tensor has a different shape
/// - `InvalidAxis` if axis > ndim (note: axis == ndim is valid)
pub fn stack<A, D>(tensors: &[TensorView<'_, A, D>], axis: usize) -> Result<Tensor<A, D::Larger>>
where
    A: Element,
    D: Dimension,
{ ... }
```

### 9.2 形状验证算法

```
function validate_cat(tensors, axis) -> Result<output_shape>:
    if tensors.is_empty(): return Error(InvalidShape)
    ref_shape = tensors[0].shape()
    total_axis_len = 0
    for t in tensors:
        for i in 0..ndim:
            if i == axis:
                total_axis_len += t.shape()[i]
            else:
                if t.shape()[i] != ref_shape[i]:
                    return Error(ShapeMismatch)
    output_shape = ref_shape.clone()
    output_shape[axis] = total_axis_len
    return Ok(output_shape)

function validate_stack(tensors, axis) -> Result<output_shape>:
    if tensors.is_empty(): return Error(InvalidShape)
    ref_shape = tensors[0].shape()
    for t in tensors[1..]:
        if t.shape() != ref_shape:
            return Error(ShapeMismatch)
    // Insert new axis
    output_shape = ref_shape.insert(axis, tensors.len())
    return Ok(output_shape)
```

### 9.3 数据拷贝策略

```
function cat_copy(tensors, axis, output) -> ():
    // Copy each tensor's data into the output at the correct offset
    offset_along_axis = 0
    for t in tensors:
        // Create a mutable slice of the output
        dst = output.slice_mut(axis, offset_along_axis..offset_along_axis + t.shape()[axis])
        copy_elements(t, dst)
        offset_along_axis += t.shape()[axis]
```

### 9.4 布局标志

| 操作 | F_CONTIGUOUS | C_CONTIGUOUS | ALIGNED | HAS_ZERO_STRIDE | HAS_NEG_STRIDE |
|------|-------------|-------------|---------|-----------------|----------------|
| cat | 根据输出 Order 设置 | 根据输出 Order 设置 | true (新分配) | false | false |
| stack | 根据输出 Order 设置 | 根据输出 Order 设置 | true (新分配) | false | false |

---

## 10. Pad 设计

### 10.1 函数签名

```rust
/// Padding mode specification.
pub enum PadMode<A> {
    /// Fill with a constant value.
    Constant(A),
    /// Replicate edge values.
    Edge,
    /// Reflect values (excluding edge).
    Reflect,
}

impl<S, D> TensorBase<S, D>
where
    S: Storage,
    D: Dimension,
{
    /// Pad the tensor along each axis.
    ///
    /// `widths` specifies (before, after) padding for each axis.
    /// Returns a new Owned tensor.
    ///
    /// # Errors
    /// - `InvalidShape` if widths.len() != ndim
    /// - `InvalidShape` if Reflect mode and padding width >= axis length
    pub fn pad(&self, widths: &[(usize, usize)], mode: PadMode<S::Elem>) -> Result<Tensor<S::Elem, D>>
    where
        S::Elem: Element,
    { ... }
}
```

### 10.2 各模式算法

```
function pad_constant(src, widths, value) -> dst:
    dst_shape = [src.shape[i] + widths[i].0 + widths[i].1 for i in 0..ndim]
    dst = Tensor::full(dst_shape, value)
    // Copy source data into the center region
    center_slices = [widths[i].0 .. widths[i].0 + src.shape[i] for i in 0..ndim]
    dst[center_slices] = src
    return dst

function pad_edge(src, widths) -> dst:
    dst = allocate(dst_shape)
    // Copy center
    dst[center] = src
    // For each axis, extend edge values
    for axis in 0..ndim:
        // Before padding: replicate first element along axis
        for j in 0..widths[axis].0:
            dst.index_axis(axis, j) = src.index_axis(axis, 0)
        // After padding: replicate last element along axis
        for j in 0..widths[axis].1:
            dst.index_axis(axis, center_end + j) = src.index_axis(axis, src.shape[axis] - 1)

function pad_reflect(src, widths) -> dst:
    // Reflect without including edge
    // Before: src[width-1, width-2, ..., 1] (not src[0])
    // After: src[n-2, n-3, ..., n-width-1] (not src[n-1])
    assert widths[i].0 < src.shape[i] for all i
    assert widths[i].1 < src.shape[i] for all i
    dst = allocate(dst_shape)
    dst[center] = src
    for axis in 0..ndim:
        for j in 0..widths[axis].0:
            src_idx = widths[axis].0 - j  // 1, 2, 3, ...
            dst.index_axis(axis, j) = src.index_axis(axis, src_idx)
        for j in 0..widths[axis].1:
            src_idx = src.shape[axis] - 2 - j  // n-2, n-3, ...
            dst.index_axis(axis, center_end + j) = src.index_axis(axis, src_idx)
```

### 10.3 布局标志

新分配的 Owned 数组，标志位根据分配 Order 设置（默认 F-order）。

---

## 11. Repeat / Tile 设计

### 11.1 函数签名

```rust
impl<S, D> TensorBase<S, D>
where
    S: Storage,
    D: Dimension,
{
    /// Repeat the tensor along each axis.
    ///
    /// `reps` specifies repetition count per axis.
    /// If reps.len() < ndim, left-pad with 1s (NumPy np.tile convention).
    /// If any rep is 0, the corresponding axis has length 0.
    ///
    /// Returns a new Owned tensor (data copy).
    pub fn repeat(&self, reps: &[usize]) -> Tensor<S::Elem, D>
    where
        S::Elem: Element,
    { ... }
}
```

### 11.2 算法

```
function repeat(src, reps) -> dst:
    // Normalize reps: left-pad with 1s if shorter than ndim
    full_reps = [1] * (ndim - reps.len()) + reps

    dst_shape = [src.shape[i] * full_reps[i] for i in 0..ndim]
    dst = allocate(dst_shape)

    // Tile by copying src into each block
    for each block_index in cartesian_product(full_reps):
        dst_offset = [block_index[i] * src.shape[i] for i in 0..ndim]
        copy src into dst at dst_offset
    return dst
```

### 11.3 布局标志

新分配的 Owned 数组，标志位根据分配 Order 设置。

---

## 12. Flatten 设计

### 12.1 函数签名

```rust
impl<S, D> TensorBase<S, D>
where
    S: Storage,
    D: Dimension,
{
    /// Flatten the tensor to 1D.
    ///
    /// If the tensor is contiguous, returns a zero-copy view.
    /// If not contiguous, returns a new Owned tensor with copied data.
    pub fn flatten(&self) -> TensorBase<impl Storage<Elem = S::Elem>, Ix1> { ... }

    /// Flatten to 1D, always returning an Owned tensor.
    pub fn flatten_to_owned(&self) -> Tensor<S::Elem, Ix1>
    where
        S::Elem: Element,
    { ... }
}
```

### 12.2 判断逻辑

```
function flatten(tensor):
    total_elements = product(shape)
    if is_f_contiguous(shape, strides):
        // Zero-copy: reshape to [total_elements] with stride [1]
        return view(shape=[total_elements], strides=[1], offset=tensor.offset)
    elif is_c_contiguous(shape, strides):
        // Zero-copy: reshape to [total_elements] with stride [1]
        return view(shape=[total_elements], strides=[1], offset=tensor.offset)
    else:
        // Must copy: iterate in logical order, write to contiguous buffer
        dst = allocate([total_elements])
        for (i, elem) in tensor.iter().enumerate():
            dst[i] = elem
        return dst
```

### 12.3 布局标志

| 情况 | F_CONTIGUOUS | C_CONTIGUOUS | ALIGNED | HAS_ZERO_STRIDE | HAS_NEG_STRIDE |
|------|-------------|-------------|---------|-----------------|----------------|
| 零拷贝 (1D) | true | true | 继承 | false | false |
| 需拷贝 (新分配) | true | true | true | false | false |


---

## 13. 布局标志更新总览

### 13.1 全操作标志更新矩阵

| 操作 | F_CONTIG | C_CONTIG | ALIGNED | ZERO_STRIDE | NEG_STRIDE | 说明 |
|------|----------|----------|---------|-------------|------------|------|
| reshape | 保持/计算 | 保持/计算 | 继承 | false | false | 输入必须连续 |
| transpose | F↔C 互换 | F↔C 互换 | 继承 | 继承 | 继承 | 步长反转 |
| permute | 重新计算 | 重新计算 | 继承 | 继承 | 继承 | 步长重排 |
| swapaxes | 重新计算 | 重新计算 | 继承 | 继承 | 继承 | 两轴交换 |
| moveaxis | 重新计算 | 重新计算 | 继承 | 继承 | 继承 | 等价于 permute |
| slice (step=1) | 重新计算 | 重新计算 | 检查 offset | 继承 | 继承 | 子区域 |
| slice (step>1) | 通常 false | 通常 false | 检查 offset | 继承 | 继承 | 跳步 |
| slice (step<0) | 重新计算 | 重新计算 | 检查 offset | 继承 | true | 反向 |
| squeeze | 重新计算 | 重新计算 | 继承 | 重新计算 | 重新计算 | 移除轴 |
| unsqueeze | 重新计算 | 重新计算 | 继承 | 继承 | 继承 | 添加轴 |
| index_axis | 重新计算 | 重新计算 | 检查 offset | 重新计算 | 重新计算 | 降维 |
| unstack | 同 index_axis | 同上 | 同上 | 同上 | 同上 | 批量 index_axis |
| split/chunk | 重新计算 | 重新计算 | 检查 offset | 继承 | 继承 | 子区域 |
| cat | 按 Order | 按 Order | true | false | false | 新分配 |
| stack | 按 Order | 按 Order | true | false | false | 新分配 |
| pad | 按 Order | 按 Order | true | false | false | 新分配 |
| repeat | 按 Order | 按 Order | true | false | false | 新分配 |
| flatten (零拷贝) | true | true | 继承 | false | false | 1D 连续 |
| flatten (拷贝) | true | true | true | false | false | 新分配 |

### 13.2 标志更新实现策略

```rust
/// Recompute all layout flags from shape and strides.
/// Called after any shape/stride modification.
fn recompute_flags(shape: &[usize], strides: &[isize], ptr_offset: usize, align: usize) -> LayoutFlags {
    let mut flags = LayoutFlags::empty();

    if is_f_contiguous(shape, strides) {
        flags |= LayoutFlags::F_CONTIGUOUS;
    }
    if is_c_contiguous(shape, strides) {
        flags |= LayoutFlags::C_CONTIGUOUS;
    }
    if (ptr_offset * elem_size) % align == 0 {
        flags |= LayoutFlags::ALIGNED;
    }
    if strides.iter().any(|&s| s == 0) {
        flags |= LayoutFlags::HAS_ZERO_STRIDE;
    }
    if strides.iter().any(|&s| s < 0) {
        flags |= LayoutFlags::HAS_NEG_STRIDE;
    }

    flags
}
```

---

## 14. 与其他模块的交互

### 14.1 依赖关系

```
05-06 shape_ops
├── 依赖 dimension (03.01)
│   ├── Dimension trait: ndim(), shape 操作
│   ├── D::Smaller: index_axis/unstack 降维
│   ├── D::Larger: stack 升维
│   └── IntoDimension: reshape 参数转换
│
├── 依赖 layout (03.05)
│   ├── LayoutFlags: 标志位更新
│   ├── is_f_contiguous/is_c_contiguous: 连续性检查
│   └── Order: cat/stack/pad 输出布局
│
├── 依赖 storage (03.04)
│   ├── Storage trait: 数据访问
│   ├── StorageMut: 可变操作
│   └── StorageOwned: into_shape
│
├── 依赖 tensor (03.06)
│   ├── TensorBase: 方法实现目标
│   ├── TensorView/TensorViewMut: 返回类型
│   └── Tensor: 需拷贝操作的返回类型
│
├── 依赖 error (03.07)
│   ├── InvalidShape: reshape 元素数不匹配
│   ├── LayoutMismatch: 非连续 reshape
│   ├── InvalidAxis: 轴越界
│   └── ShapeMismatch: cat/stack 形状不匹配
│
├── 被 ops (05.02) 依赖
│   └── 逐元素运算可能需要 reshape/broadcast
│
├── 被 iter (05.01) 依赖
│   └── 轴迭代使用 index_axis
│
└── 被 index (05.07) 依赖
    └── 高级索引使用 slice
```

---

## 15. 实现任务分解

| 任务 | 描述 | 预计时间 | 依赖 |
|------|------|----------|------|
| T1 | 创建 `src/shape_ops/mod.rs` 模块入口 | 5 min | 无 |
| T2 | 实现 `reshape.rs`: reshape, reshape_into, into_shape | 10 min | T1, 03.05 |
| T3 | 实现 `reshape.rs`: flatten, flatten_to_owned | 10 min | T2 |
| T4 | 实现 `transpose.rs`: t(), t_mut() | 10 min | T1 |
| T5 | 实现 `transpose.rs`: permute, swapaxes, moveaxis | 10 min | T4 |
| T6 | 实现 `slice.rs`: slice 核心计算（不含宏） | 10 min | T1 |
| T7 | 实现 `slice.rs`: s![] 宏 | 10 min | T6 |
| T8 | 实现 `squeeze.rs`: squeeze, squeeze_axis, unsqueeze | 10 min | T1 |
| T9 | 实现 `split.rs`: index_axis, index_axis_mut | 10 min | T1 |
| T10 | 实现 `split.rs`: unstack | 5 min | T9 |
| T11 | 实现 `split.rs`: split, chunk | 10 min | T1 |
| T12 | 实现 `stack.rs`: cat | 10 min | T1, 05.01 |
| T13 | 实现 `stack.rs`: stack | 10 min | T12 |
| T14 | 实现 `pad.rs`: PadMode 枚举 + pad_constant | 10 min | T1 |
| T15 | 实现 `pad.rs`: pad_edge, pad_reflect | 10 min | T14 |
| T16 | 实现 `repeat.rs`: repeat/tile | 10 min | T1 |
| T17 | 布局标志更新: recompute_flags 统一实现 | 10 min | T2-T16 |

### 15.1 并行执行分组

```
Wave 1 (无依赖):
  T1

Wave 2 (依赖 T1，可并行):
  T2, T4, T6, T8, T9, T11, T14, T16

Wave 3 (依赖 Wave 2，可并行):
  T3, T5, T7, T10, T12, T15

Wave 4 (依赖 Wave 3):
  T13

Wave 5 (依赖所有):
  T17
```

---

## 16. 设计决策记录

### 16.1 决策：reshape 返回 Result 而非 panic

| 属性 | 值 |
|------|-----|
| 决策 | 非连续数组 reshape 返回 `LayoutMismatch` 错误 |
| 理由 | 连续性是运行时属性，调用方无法在编译时保证；返回 Result 允许优雅处理 |
| 替代方案 | panic — 放弃，不符合"错误而非 panic"原则 |
| 替代方案 | 自动拷贝 — 放弃，隐式拷贝违反"无隐式行为"原则 |

### 16.2 决策：squeeze 返回 IxDyn

| 属性 | 值 |
|------|-----|
| 决策 | squeeze 返回 `TensorView<'_, A, IxDyn>` 而非保持静态维度 |
| 理由 | squeeze 后的维度数在编译时未知（取决于哪些轴是 size-1） |
| 替代方案 | 泛型返回 — 放弃，Rust 类型系统无法表达"编译时未知的维度数" |

### 16.3 决策：flatten 返回 impl Storage

| 属性 | 值 |
|------|-----|
| 决策 | flatten 返回 `TensorBase<impl Storage, Ix1>`，连续时零拷贝，否则拷贝 |
| 理由 | 调用方通常不关心底层是视图还是拥有，只需要 1D 连续数据 |
| 替代方案 | 总是拷贝 — 放弃，连续数组的拷贝是不必要的开销 |
| 替代方案 | 返回 Cow — 放弃，增加 API 复杂度 |

### 16.4 决策：cat/stack 为自由函数

| 属性 | 值 |
|------|-----|
| 决策 | cat 和 stack 为模块级自由函数而非方法 |
| 理由 | 操作多个张量，不属于单个张量的方法；与 NumPy `np.concatenate` / `np.stack` 一致 |
| 替代方案 | 方法 `a.cat_with(&b, axis)` — 放弃，仅支持两个张量，不够通用 |

### 16.5 决策：s![] 宏语法参考 ndarray

| 属性 | 值 |
|------|-----|
| 决策 | s![] 宏语法与 ndarray 的 s![] 宏保持一致 |
| 理由 | Rust 科学计算社区已熟悉此语法；降低迁移成本 |
| 替代方案 | 自定义语法 — 放弃，增加学习成本 |

---

*本文档由 Senon 项目维护。如有问题请提交 Issue 或 PR。*
