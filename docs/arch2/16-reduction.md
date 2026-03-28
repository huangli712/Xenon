# 归约操作模块设计

> 文档编号: 16 | 模块: `src/ops/reduction.rs`, `src/ops/accumulate.rs` | 阶段: Phase 3
> 前置文档: `00-rust-standards.md`, `01-architecture-overview.md`, `07-tensor-core.md`
> 需求参考: 需求说明书 §10.3

---

## 1. 模块定位

归约操作模块是 Xenon 运算体系的核心组件之一，负责将张量沿指定维度（或全局）聚合为标量或低维张量。涵盖三大类操作：

| 类别 | 操作 | 典型用途 |
|------|------|----------|
| 全局归约 | sum, prod, min, max, mean, var, std, argmin, argmax, all, any | 损失计算、统计摘要、条件判断 |
| 沿轴归约 | 以上所有运算的 `_axis` 变体 | 批量统计、特征聚合 |
| 累积归约 | cumsum, cumprod | 前缀和/积、累积收益率 |

**本模块职责边界：**

| 职责 | 包含 | 不包含 |
|------|------|--------|
| 归约计算逻辑 | 标量/SIMD/并行归约内核 | 逐元素运算（`element_wise.rs`） |
| 方差/标准差 | Welford 算法实现 | 排序（`set_ops.rs`） |
| 累积运算 | cumsum/cumprod 的前缀扫描 | 滑动窗口（`iter/windows.rs`） |
| 公共 trait | `Reduce`, `Accumulate` | 迭代器 trait（`iter/mod.rs`） |

**设计目标：**

| 目标 | 实现方式 |
|------|----------|
| 数值稳定性 | 方差/标准差使用 Welford 在线算法 |
| 性能分层 | 标量 → SIMD → 并行三级 dispatch |
| 类型安全 | 通过 trait bound 区分数值/实数/布尔归约 |
| 零开销 | 泛型单态化，trait 方法内联 |

---

## 2. 文件位置

```
src/ops/
├── mod.rs              # pub mod reduction, accumulate; re-export traits
├── reduction.rs        # Reduce trait + 全局/沿轴归约实现
└── accumulate.rs       # Accumulate trait + cumsum/cumprod 实现
```

**双文件拆分理由：** 归约（`reduction.rs`）与累积（`accumulate.rs`）的计算模式不同——前者是多对一的聚合操作，后者是保持形状的前缀扫描。拆分后每个文件职责单一，预计 `reduction.rs` ~800 行，`accumulate.rs` ~300 行。

---

## 3. 依赖关系

```
ops/reduction.rs
├── crate::tensor       # TensorBase<S, D>, Tensor
├── crate::dimension    # Dimension, RemoveAxis (D::Smaller), Ix0~Ix6, IxDyn
├── crate::element      # Element, Numeric, RealScalar
├── crate::storage      # RawStorage, Storage, Owned
├── crate::layout       # LayoutFlags (contiguity checks)
├── crate::error        # TensorError::EmptyArray, TensorError::InvalidAxis
├── crate::construction # zeros() for output allocation
├── crate::backend
│   ├── scalar.rs       # 标量归约回退路径
│   ├── simd.rs         # SIMD 归约路径 (feature = "simd")
│   └── parallel.rs     # 并行归约路径 (feature = "parallel")
└── crate::private
    └── math.rs         # 数学辅助 (is_nan 等)

ops/accumulate.rs
├── crate::tensor       # TensorBase<S, D>, Tensor
├── crate::dimension    # Dimension
├── crate::element      # Element, Numeric
├── crate::storage      # RawStorage, Owned
├── crate::error        # TensorError::InvalidAxis
└── crate::construction # zeros() / full() for output allocation
```

### 3.1 依赖的具体类型

| 来源模块 | 使用的类型/trait |
|----------|-----------------|
| `tensor` | `TensorBase<S, D>`, `Tensor<A, D>`, `TensorView` |
| `dimension` | `Dimension`, `RemoveAxis`, `Ix0`~`Ix6`, `IxDyn` |
| `element` | `Element`, `Numeric`, `RealScalar` |
| `storage` | `RawStorage`, `Storage`, `Owned<A>` |
| `layout` | `LayoutFlags` (contiguity checks for SIMD path selection) |
| `error` | `TensorError::EmptyArray`, `TensorError::InvalidAxis` |
| `construction` | `zeros()`, `full()` (output tensor allocation) |

### 3.2 RemoveAxis trait 依赖

沿轴归约需要维度系统提供 `RemoveAxis` trait（已在 `docs/12-indexing.md` 中设计）：

```rust
/// Trait for dimensions that support removing an axis.
pub trait RemoveAxis: Dimension {
    /// The dimension type after removing one axis.
    type Smaller: Dimension;

    /// Removes the axis at the given position.
    fn remove_axis(&self, axis: usize) -> Self::Smaller;
}
```

实现映射：`Ix1 → Ix0`, `Ix2 → Ix1`, ..., `Ix6 → Ix5`, `IxDyn → IxDyn`。`Ix0` 不实现此 trait。

---

## 4. 公共 API 设计

### 4.1 设计决策：扩展 trait 模式

采用**扩展 trait** 模式（extension trait），在 `ops/reduction.rs` 中定义 `Reduce` trait 并为 `TensorBase<S, D>` 提供实现。通过 `pub use crate::ops::*` 在 crate 根 re-export，用户只需 `use xenon::Reduce;` 即可在任意 `TensorBase` 上调用归约方法。

**选择 trait 而非直接 inherent impl 的理由：**

1. 归约方法跨越多个元素类型约束（`Numeric`, `RealScalar`, `bool`），单一 impl 块无法表达；trait 方法级别 `where` 子句天然支持这种分约束模式
2. trait 可被第三方 crate 扩展（为自定义存储类型实现归约）
3. 与 `Accumulate` trait 并列，形成清晰的 API 边界

**trait 方法级别 `where` 子句说明：** Rust 允许在 trait 方法的返回类型中使用 `where` 子句引入的约束关联类型（如 `<D as RemoveAxis>::Smaller`）。调用时编译器在调用点检查额外约束，不满足则编译失败。

### 4.2 Reduce trait 定义

```rust
/// Reduction operations for N-dimensional arrays.
///
/// This trait provides global and along-axis reduction methods.
/// Import this trait to enable `.sum()`, `.mean()`, etc. on tensors.
///
/// # Type Bounds
///
/// The trait itself requires only `Element` (the base layer).
/// Individual methods add stricter bounds (e.g., `Numeric` for `sum`,
/// `RealScalar` for `var`).
///
/// # Examples
///
/// ```
/// use xenon::{Tensor, Reduce, zeros};
/// let a: Tensor<f64, Ix2> = zeros([3, 4]);
/// assert_eq!(a.sum(), 0.0);
/// ```
pub trait Reduce<S, A, D>
where
    S: RawStorage<Elem = A>,
    A: Element,
    D: Dimension,
{
    // ── Global Numeric Reductions ─────────────────────────────────

    /// Computes the sum of all elements.
    ///
    /// Uses `A::zero()` as the initial accumulator. For integer types,
    /// overflow panics (checked arithmetic in all build modes).
    ///
    /// # NaN Behavior
    ///
    /// If any element is NaN, the result is NaN (IEEE 754 propagation).
    ///
    /// # Examples
    ///
    /// ```
    /// use xenon::{Tensor, Reduce};
    /// let a = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], [3]);
    /// assert_eq!(a.sum(), 6.0);
    /// ```
    fn sum(&self) -> A
    where
        A: Numeric;

    /// Computes the product of all elements.
    ///
    /// Uses `A::one()` as the initial accumulator. For integer types,
    /// overflow panics (checked arithmetic in all build modes).
    ///
    /// # NaN Behavior
    ///
    /// If any element is NaN, the result is NaN (IEEE 754 propagation).
    fn prod(&self) -> A
    where
        A: Numeric;

    // ── Global Ordered Reductions ─────────────────────────────────

    /// Returns the minimum element.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::EmptyArray` if the tensor contains zero elements.
    ///
    /// # NaN Behavior
    ///
    /// If any element is NaN, returns NaN (NaN propagation semantics).
    fn min(&self) -> Result<A>
    where
        A: PartialOrd;

    /// Returns the maximum element.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::EmptyArray` if the tensor contains zero elements.
    ///
    /// # NaN Behavior
    ///
    /// If any element is NaN, returns NaN (NaN propagation semantics).
    fn max(&self) -> Result<A>
    where
        A: PartialOrd;

    /// Returns the index of the minimum element (flat index).
    ///
    /// When multiple elements share the minimum value, returns the index
    /// of the first occurrence (in memory layout order).
    ///
    /// # Errors
    ///
    /// Returns `TensorError::EmptyArray` if the tensor contains zero elements.
    fn argmin(&self) -> Result<usize>
    where
        A: PartialOrd;

    /// Returns the index of the maximum element (flat index).
    ///
    /// When multiple elements share the maximum value, returns the index
    /// of the first occurrence (in memory layout order).
    ///
    /// # Errors
    ///
    /// Returns `TensorError::EmptyArray` if the tensor contains zero elements.
    fn argmax(&self) -> Result<usize>
    where
        A: PartialOrd;

    // ── Global Statistical Reductions ─────────────────────────────

    /// Computes the arithmetic mean of all elements.
    ///
    /// Equivalent to `self.sum() / A::from(self.len())`.
    fn mean(&self) -> A
    where
        A: RealScalar;

    /// Computes the population variance (ddof = 0).
    ///
    /// Uses Welford's online algorithm for numerical stability.
    /// Equivalent to `self.var_with_ddof(0)`.
    fn var(&self) -> A
    where
        A: RealScalar;

    /// Computes the variance with the given delta degrees of freedom.
    ///
    /// The variance is computed as `sum((x - mean)^2) / (N - ddof)`.
    /// Uses Welford's online algorithm for numerical stability.
    ///
    /// # Panics
    ///
    /// Panics if `ddof >= self.len()` (division by zero).
    fn var_with_ddof(&self, ddof: usize) -> A
    where
        A: RealScalar;

    /// Computes the population standard deviation (ddof = 0).
    ///
    /// Equivalent to `self.std_with_ddof(0)`.
    fn std(&self) -> A
    where
        A: RealScalar;

    /// Computes the standard deviation with the given ddof.
    ///
    /// Equivalent to `self.var_with_ddof(ddof).sqrt()`.
    ///
    /// # Panics
    ///
    /// Panics if `ddof >= self.len()` (division by zero).
    fn std_with_ddof(&self, ddof: usize) -> A
    where
        A: RealScalar;

    // ── Global Boolean Reductions ─────────────────────────────────

    /// Returns `true` if all elements are `true`.
    ///
    /// For an empty tensor, returns `true` (vacuous truth).
    fn all(&self) -> bool
    where
        S: RawStorage<Elem = bool>;

    /// Returns `true` if any element is `true`.
    ///
    /// For an empty tensor, returns `false`.
    fn any(&self) -> bool
    where
        S: RawStorage<Elem = bool>;

    // ── Along-Axis Numeric Reductions ─────────────────────────────

    /// Sums elements along the specified axis.
    ///
    /// The reduced axis is removed from the output shape.
    ///
    /// # Panics
    ///
    /// Panics if `axis >= self.ndim()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use xenon::{Tensor, Reduce};
    /// // Input shape: [2, 3]
    /// let a = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    /// // sum_axis(0) -> shape [3]: [5.0, 7.0, 9.0]
    /// // sum_axis(1) -> shape [2]: [6.0, 15.0]
    /// let s = a.sum_axis(1);
    /// assert_eq!(s.shape(), &[2]);
    /// ```
    fn sum_axis(&self, axis: usize) -> Tensor<A, <D as RemoveAxis>::Smaller>
    where
        A: Numeric,
        D: RemoveAxis;

    /// Computes the product along the specified axis.
    fn prod_axis(&self, axis: usize) -> Tensor<A, <D as RemoveAxis>::Smaller>
    where
        A: Numeric,
        D: RemoveAxis;

    // ── Along-Axis Ordered Reductions ─────────────────────────────

    /// Returns the minimum along the specified axis.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::InvalidAxis` if `axis >= self.ndim()`.
    /// Returns `TensorError::EmptyArray` if the specified axis has length 0.
    fn min_axis(&self, axis: usize) -> Result<Tensor<A, <D as RemoveAxis>::Smaller>>
    where
        A: PartialOrd,
        D: RemoveAxis;

    /// Returns the maximum along the specified axis.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::InvalidAxis` if `axis >= self.ndim()`.
    /// Returns `TensorError::EmptyArray` if the specified axis has length 0.
    fn max_axis(&self, axis: usize) -> Result<Tensor<A, <D as RemoveAxis>::Smaller>>
    where
        A: PartialOrd,
        D: RemoveAxis;

    /// Returns the argmin along the specified axis.
    ///
    /// The returned tensor contains indices (usize) along the specified axis.
    /// When multiple elements share the minimum, returns the first index.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::InvalidAxis` if `axis >= self.ndim()`.
    /// Returns `TensorError::EmptyArray` if the specified axis has length 0.
    fn argmin_axis(&self, axis: usize) -> Result<Tensor<usize, <D as RemoveAxis>::Smaller>>
    where
        A: PartialOrd,
        D: RemoveAxis;

    /// Returns the argmax along the specified axis.
    ///
    /// The returned tensor contains indices (usize) along the specified axis.
    /// When multiple elements share the maximum, returns the first index.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::InvalidAxis` if `axis >= self.ndim()`.
    /// Returns `TensorError::EmptyArray` if the specified axis has length 0.
    fn argmax_axis(&self, axis: usize) -> Result<Tensor<usize, <D as RemoveAxis>::Smaller>>
    where
        A: PartialOrd,
        D: RemoveAxis;

    // ── Along-Axis Statistical Reductions ─────────────────────────

    /// Computes the mean along the specified axis.
    ///
    /// # Panics
    ///
    /// Panics if `axis >= self.ndim()`.
    fn mean_axis(&self, axis: usize) -> Tensor<A, <D as RemoveAxis>::Smaller>
    where
        A: RealScalar,
        D: RemoveAxis;

    /// Computes the population variance along the specified axis (ddof = 0).
    fn var_axis(&self, axis: usize) -> Tensor<A, <D as RemoveAxis>::Smaller>
    where
        A: RealScalar,
        D: RemoveAxis;

    /// Computes the variance along the specified axis with the given ddof.
    ///
    /// # Panics
    ///
    /// Panics if `axis >= self.ndim()`.
    /// Panics if `ddof >= axis_length`.
    fn var_axis_with_ddof(
        &self,
        axis: usize,
        ddof: usize,
    ) -> Tensor<A, <D as RemoveAxis>::Smaller>
    where
        A: RealScalar,
        D: RemoveAxis;

    /// Computes the population standard deviation along the specified axis (ddof = 0).
    fn std_axis(&self, axis: usize) -> Tensor<A, <D as RemoveAxis>::Smaller>
    where
        A: RealScalar,
        D: RemoveAxis;

    /// Computes the standard deviation along the specified axis with the given ddof.
    ///
    /// # Panics
    ///
    /// Panics if `axis >= self.ndim()`.
    /// Panics if `ddof >= axis_length`.
    fn std_axis_with_ddof(
        &self,
        axis: usize,
        ddof: usize,
    ) -> Tensor<A, <D as RemoveAxis>::Smaller>
    where
        A: RealScalar,
        D: RemoveAxis;

    // ── Along-Axis Boolean Reductions ─────────────────────────────

    /// Returns `true` along the axis where all elements are `true`.
    ///
    /// # Panics
    ///
    /// Panics if `axis >= self.ndim()`.
    fn all_axis(&self, axis: usize) -> Tensor<bool, <D as RemoveAxis>::Smaller>
    where
        S: RawStorage<Elem = bool>,
        D: RemoveAxis;

    /// Returns `true` along the axis where any element is `true`.
    ///
    /// # Panics
    ///
    /// Panics if `axis >= self.ndim()`.
    fn any_axis(&self, axis: usize) -> Tensor<bool, <D as RemoveAxis>::Smaller>
    where
        S: RawStorage<Elem = bool>,
        D: RemoveAxis;

    // ── Keepdim Variants ──────────────────────────────────────────

    /// Sums elements along the specified axis, keeping the reduced axis
    /// as a size-1 dimension.
    ///
    /// Unlike `sum_axis`, the output has the same number of dimensions as
    /// the input, with `shape[axis] == 1`.
    fn sum_axis_keepdim(&self, axis: usize) -> Tensor<A, D>
    where
        A: Numeric,
        D: RemoveAxis + Clone;

    /// Computes the mean along the specified axis, keeping the reduced axis.
    fn mean_axis_keepdim(&self, axis: usize) -> Tensor<A, D>
    where
        A: RealScalar,
        D: RemoveAxis + Clone;

    /// Computes the population variance along the axis, keeping the reduced axis.
    fn var_axis_keepdim(&self, axis: usize) -> Tensor<A, D>
    where
        A: RealScalar,
        D: RemoveAxis + Clone;

    /// Computes the population standard deviation along the axis, keeping the reduced axis.
    fn std_axis_keepdim(&self, axis: usize) -> Tensor<A, D>
    where
        A: RealScalar,
        D: RemoveAxis + Clone;

    /// Returns the minimum along the axis, keeping the reduced axis.
    fn min_axis_keepdim(&self, axis: usize) -> Result<Tensor<A, D>>
    where
        A: PartialOrd,
        D: RemoveAxis + Clone;

    /// Returns the maximum along the axis, keeping the reduced axis.
    fn max_axis_keepdim(&self, axis: usize) -> Result<Tensor<A, D>>
    where
        A: PartialOrd,
        D: RemoveAxis + Clone;
}
```

### 4.3 Reduce trait 实现

```rust
impl<S, A, D> Reduce<S, A, D> for TensorBase<S, D>
where
    S: RawStorage<Elem = A>,
    A: Element,
    D: Dimension,
{
    fn sum(&self) -> A
    where
        A: Numeric,
    {
        reduce_sum(self)
    }

    fn prod(&self) -> A
    where
        A: Numeric,
    {
        reduce_prod(self)
    }

    fn min(&self) -> Result<A>
    where
        A: PartialOrd,
    {
        if self.is_empty() {
            return Err(TensorError::EmptyArray {
                operation: "min",
            });
        }
        Ok(reduce_min(self))
    }

    fn max(&self) -> Result<A>
    where
        A: PartialOrd,
    {
        if self.is_empty() {
            return Err(TensorError::EmptyArray {
                operation: "max",
            });
        }
        Ok(reduce_max(self))
    }

    fn argmin(&self) -> Result<usize>
    where
        A: PartialOrd,
    {
        if self.is_empty() {
            return Err(TensorError::EmptyArray {
                operation: "argmin",
            });
        }
        Ok(reduce_argmin(self))
    }

    fn argmax(&self) -> Result<usize>
    where
        A: PartialOrd,
    {
        if self.is_empty() {
            return Err(TensorError::EmptyArray {
                operation: "argmax",
            });
        }
        Ok(reduce_argmax(self))
    }

    fn mean(&self) -> A
    where
        A: RealScalar,
    {
        let sum = reduce_sum(self);
        let n = A::from(self.len() as _);
        sum / n
    }

    fn var(&self) -> A
    where
        A: RealScalar,
    {
        self.var_with_ddof(0)
    }

    fn var_with_ddof(&self, ddof: usize) -> A
    where
        A: RealScalar,
    {
        assert!(
            ddof < self.len(),
            "var: ddof ({}) must be less than element count ({})",
            ddof,
            self.len()
        );
        reduce_var_welford(self, ddof)
    }

    fn std(&self) -> A
    where
        A: RealScalar,
    {
        self.var().sqrt()
    }

    fn std_with_ddof(&self, ddof: usize) -> A
    where
        A: RealScalar,
    {
        self.var_with_ddof(ddof).sqrt()
    }

    fn all(&self) -> bool
    where
        S: RawStorage<Elem = bool>,
    {
        // Every element must be true; vacuous truth for empty.
        for elem in self.iter() {
            if !elem {
                return false;
            }
        }
        true
    }

    fn any(&self) -> bool
    where
        S: RawStorage<Elem = bool>,
    {
        // At least one element must be true; false for empty.
        for elem in self.iter() {
            if *elem {
                return true;
            }
        }
        false
    }

    // ── Axis reduction implementations delegate to internal functions ──

    fn sum_axis(&self, axis: usize) -> Tensor<A, <D as RemoveAxis>::Smaller>
    where
        A: Numeric,
        D: RemoveAxis,
    {
        validate_axis(self.ndim(), axis);
        reduce_sum_axis(self, axis)
    }

    fn prod_axis(&self, axis: usize) -> Tensor<A, <D as RemoveAxis>::Smaller>
    where
        A: Numeric,
        D: RemoveAxis,
    {
        validate_axis(self.ndim(), axis);
        reduce_prod_axis(self, axis)
    }

    fn min_axis(&self, axis: usize) -> Result<Tensor<A, <D as RemoveAxis>::Smaller>>
    where
        A: PartialOrd,
        D: RemoveAxis,
    {
        validate_axis(self.ndim(), axis);
        if self.shape()[axis] == 0 {
            return Err(TensorError::EmptyArray {
                operation: "min_axis",
            });
        }
        Ok(reduce_min_axis(self, axis))
    }

    fn max_axis(&self, axis: usize) -> Result<Tensor<A, <D as RemoveAxis>::Smaller>>
    where
        A: PartialOrd,
        D: RemoveAxis,
    {
        validate_axis(self.ndim(), axis);
        if self.shape()[axis] == 0 {
            return Err(TensorError::EmptyArray {
                operation: "max_axis",
            });
        }
        Ok(reduce_max_axis(self, axis))
    }

    fn argmin_axis(&self, axis: usize) -> Result<Tensor<usize, <D as RemoveAxis>::Smaller>>
    where
        A: PartialOrd,
        D: RemoveAxis,
    {
        validate_axis(self.ndim(), axis);
        if self.shape()[axis] == 0 {
            return Err(TensorError::EmptyArray {
                operation: "argmin_axis",
            });
        }
        Ok(reduce_argmin_axis(self, axis))
    }

    fn argmax_axis(&self, axis: usize) -> Result<Tensor<usize, <D as RemoveAxis>::Smaller>>
    where
        A: PartialOrd,
        D: RemoveAxis,
    {
        validate_axis(self.ndim(), axis);
        if self.shape()[axis] == 0 {
            return Err(TensorError::EmptyArray {
                operation: "argmax_axis",
            });
        }
        Ok(reduce_argmax_axis(self, axis))
    }

    fn mean_axis(&self, axis: usize) -> Tensor<A, <D as RemoveAxis>::Smaller>
    where
        A: RealScalar,
        D: RemoveAxis,
    {
        validate_axis(self.ndim(), axis);
        reduce_mean_axis(self, axis, 0)
    }

    fn var_axis(&self, axis: usize) -> Tensor<A, <D as RemoveAxis>::Smaller>
    where
        A: RealScalar,
        D: RemoveAxis,
    {
        validate_axis(self.ndim(), axis);
        reduce_var_axis(self, axis, 0)
    }

    fn var_axis_with_ddof(
        &self,
        axis: usize,
        ddof: usize,
    ) -> Tensor<A, <D as RemoveAxis>::Smaller>
    where
        A: RealScalar,
        D: RemoveAxis,
    {
        validate_axis(self.ndim(), axis);
        assert!(
            ddof < self.shape()[axis],
            "var_axis: ddof ({}) must be less than axis length ({})",
            ddof,
            self.shape()[axis]
        );
        reduce_var_axis(self, axis, ddof)
    }

    fn std_axis(&self, axis: usize) -> Tensor<A, <D as RemoveAxis>::Smaller>
    where
        A: RealScalar,
        D: RemoveAxis,
    {
        self.var_axis(axis).mapv(|x| x.sqrt())
    }

    fn std_axis_with_ddof(
        &self,
        axis: usize,
        ddof: usize,
    ) -> Tensor<A, <D as RemoveAxis>::Smaller>
    where
        A: RealScalar,
        D: RemoveAxis,
    {
        self.var_axis_with_ddof(axis, ddof).mapv(|x| x.sqrt())
    }

    fn all_axis(&self, axis: usize) -> Tensor<bool, <D as RemoveAxis>::Smaller>
    where
        S: RawStorage<Elem = bool>,
        D: RemoveAxis,
    {
        validate_axis(self.ndim(), axis);
        reduce_all_axis(self, axis)
    }

    fn any_axis(&self, axis: usize) -> Tensor<bool, <D as RemoveAxis>::Smaller>
    where
        S: RawStorage<Elem = bool>,
        D: RemoveAxis,
    {
        validate_axis(self.ndim(), axis);
        reduce_any_axis(self, axis)
    }

    // ── Keepdim variants ──────────────────────────────────────────

    fn sum_axis_keepdim(&self, axis: usize) -> Tensor<A, D>
    where
        A: Numeric,
        D: RemoveAxis + Clone,
    {
        validate_axis(self.ndim(), axis);
        let reduced = reduce_sum_axis(self, axis);
        expand_keepdim(reduced, self.shape(), axis)
    }

    fn mean_axis_keepdim(&self, axis: usize) -> Tensor<A, D>
    where
        A: RealScalar,
        D: RemoveAxis + Clone,
    {
        validate_axis(self.ndim(), axis);
        let reduced = reduce_mean_axis(self, axis, 0);
        expand_keepdim(reduced, self.shape(), axis)
    }

    fn var_axis_keepdim(&self, axis: usize) -> Tensor<A, D>
    where
        A: RealScalar,
        D: RemoveAxis + Clone,
    {
        validate_axis(self.ndim(), axis);
        let reduced = reduce_var_axis(self, axis, 0);
        expand_keepdim(reduced, self.shape(), axis)
    }

    fn std_axis_keepdim(&self, axis: usize) -> Tensor<A, D>
    where
        A: RealScalar,
        D: RemoveAxis + Clone,
    {
        validate_axis(self.ndim(), axis);
        let reduced = reduce_var_axis(self, axis, 0);
        let stddev = reduced.mapv(|x| x.sqrt());
        expand_keepdim(stddev, self.shape(), axis)
    }

    fn min_axis_keepdim(&self, axis: usize) -> Result<Tensor<A, D>>
    where
        A: PartialOrd,
        D: RemoveAxis + Clone,
    {
        validate_axis(self.ndim(), axis);
        if self.shape()[axis] == 0 {
            return Err(TensorError::EmptyArray {
                operation: "min_axis_keepdim",
            });
        }
        let reduced = reduce_min_axis(self, axis);
        Ok(expand_keepdim(reduced, self.shape(), axis))
    }

    fn max_axis_keepdim(&self, axis: usize) -> Result<Tensor<A, D>>
    where
        A: PartialOrd,
        D: RemoveAxis + Clone,
    {
        validate_axis(self.ndim(), axis);
        if self.shape()[axis] == 0 {
            return Err(TensorError::EmptyArray {
                operation: "max_axis_keepdim",
            });
        }
        let reduced = reduce_max_axis(self, axis);
        Ok(expand_keepdim(reduced, self.shape(), axis))
    }
}
```

### 4.4 Accumulate trait 定义

```rust
/// Cumulative reduction operations for N-dimensional arrays.
///
/// Import this trait to enable `.cumsum()` and `.cumprod()` on tensors.
pub trait Accumulate<S, A, D>
where
    S: RawStorage<Elem = A>,
    A: Element,
    D: Dimension,
{
    /// Computes the cumulative sum along the specified axis.
    ///
    /// The output has the same shape as the input. Each element `out[i]`
    /// equals the sum of `input[0..=i]` along the specified axis.
    ///
    /// # NaN Behavior
    ///
    /// If any element along the axis is NaN, all subsequent elements
    /// in the cumulative result are also NaN (NaN propagation).
    ///
    /// # Panics
    ///
    /// Panics if `axis >= self.ndim()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use xenon::{Tensor, Accumulate};
    /// let a = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], [3]);
    /// let cs = a.cumsum(0);
    /// assert_eq!(cs.to_vec(), vec![1.0, 3.0, 6.0]);
    /// ```
    fn cumsum(&self, axis: usize) -> Tensor<A, D>
    where
        A: Numeric;

    /// Computes the cumulative product along the specified axis.
    ///
    /// The output has the same shape as the input. Each element `out[i]`
    /// equals the product of `input[0..=i]` along the specified axis.
    ///
    /// # NaN Behavior
    ///
    /// If any element along the axis is NaN, all subsequent elements
    /// in the cumulative result are also NaN (NaN propagation).
    ///
    /// # Panics
    ///
    /// Panics if `axis >= self.ndim()`.
    fn cumprod(&self, axis: usize) -> Tensor<A, D>
    where
        A: Numeric;
}
```

### 4.5 Accumulate trait 实现

```rust
impl<S, A, D> Accumulate<S, A, D> for TensorBase<S, D>
where
    S: RawStorage<Elem = A>,
    A: Element,
    D: Dimension,
{
    fn cumsum(&self, axis: usize) -> Tensor<A, D>
    where
        A: Numeric,
    {
        validate_axis(self.ndim(), axis);
        accumulate_scan(self, axis, A::zero(), |acc, x| *acc = *acc + *x)
    }

    fn cumprod(&self, axis: usize) -> Tensor<A, D>
    where
        A: Numeric,
    {
        validate_axis(self.ndim(), axis);
        accumulate_scan(self, axis, A::one(), |acc, x| *acc = *acc * *x)
    }
}
```

### 4.6 输出维度规则汇总

| 操作 | 输入维度 | 输出维度 | 形状变化示例 |
|------|---------|---------|-------------|
| 全局归约（sum 等） | `D` | 标量 `A` | `[2,3,4] → A` |
| 沿轴归约（sum_axis） | `D` | `<D as RemoveAxis>::Smaller` | `[2,3,4] axis=1 → [2,4]` (Ix3→Ix2) |
| 沿轴归约 keepdim | `D` | `D` | `[2,3,4] axis=1 → [2,1,4]` |
| argmin/argmax 全局 | `D` | `usize` | `[2,3,4] → usize` |
| argmin/argmax 沿轴 | `D` | `<D as RemoveAxis>::Smaller`（元素类型为 `usize`） | `[2,3,4] axis=1 → [2,4]` (usize) |
| cumsum/cumprod | `D` | `D`（相同形状） | `[2,3,4] → [2,3,4]` |
| all/any 全局 | `D` | `bool` | `[2,3,4] → bool` |
| all/any 沿轴 | `D` | `<D as RemoveAxis>::Smaller`（元素类型为 `bool`） | `[2,3,4] axis=1 → [2,4]` (bool) |

### 4.7 ops/mod.rs 集成

```rust
// src/ops/mod.rs

pub mod element_wise;
pub mod matrix;
pub mod reduction;
pub mod accumulate;
pub mod set_ops;

// Re-export traits for convenience
pub use reduction::Reduce;
pub use accumulate::Accumulate;
```

### 4.8 lib.rs re-export

```rust
// src/lib.rs (追加到现有 re-export 块)

// Reduction and accumulate traits
pub use crate::ops::{Reduce, Accumulate};
```

---

## 5. 内部实现设计

### 5.1 核心架构

归约操作内部采用**三层分离**架构：

```
┌──────────────────────────────────────────────┐
│  Reduce trait methods (public API)           │
│  输入校验 → 调度 → 返回结果                    │
├──────────────────────────────────────────────┤
│  Dispatch layer                              │
│  根据 LayoutFlags + 数据量 + feature 选择路径  │
├──────────────────────────────────────────────┤
│  Backend kernels                             │
│  scalar_reduce  │  simd_reduce  │  par_reduce│
└──────────────────────────────────────────────┘
```

### 5.2 调度策略

```rust
/// Selects the execution path for reduction operations.
fn dispatch_reduce<S, A, D>(
    tensor: &TensorBase<S, D>,
) -> ReducePath
where
    S: RawStorage<Elem = A>,
    A: Element,
    D: Dimension,
{
    let len = tensor.len();
    let contiguous = tensor.is_contiguous();

    // Non-contiguous data: always scalar (strided access)
    if !contiguous {
        return ReducePath::Scalar;
    }

    // Small data: scalar is faster (avoids SIMD setup overhead)
    if len < SIMD_WIDTH {
        return ReducePath::Scalar;
    }

    #[cfg(feature = "simd")]
    {
        #[cfg(feature = "parallel")]
        {
            if len >= PARALLEL_THRESHOLD {
                return ReducePath::SimdParallel;
            }
        }
        return ReducePath::Simd;
    }

    #[cfg(not(feature = "simd"))]
    {
        #[cfg(feature = "parallel")]
        {
            if len >= PARALLEL_THRESHOLD {
                return ReducePath::ScalarParallel;
            }
        }
    }

    ReducePath::Scalar
}

enum ReducePath {
    Scalar,
    Simd,
    SimdParallel,
    ScalarParallel,
}
```

**调度规则汇总：**

| 条件 | 执行路径 | 备注 |
|------|---------|------|
| 非连续内存 | 标量 | 步长跳跃，SIMD 无法高效处理 |
| 元素数 < SIMD 宽度 | 标量 | SIMD 启动开销不划算 |
| 连续 + simd feature + 元素数 ≥ SIMD 宽度 | SIMD | pulp 向量化 |
| 连续 + simd + parallel + 元素数 ≥ 64K | SIMD + 并行 | 每线程 SIMD |
| 连续 + parallel（无 simd）+ 元素数 ≥ 64K | 标量 + 并行 | 无 SIMD 的并行回退 |

### 5.3 全局归约标量内核

```rust
/// Scalar reduction kernel: sum all elements.
///
/// Iterates in memory layout order for cache efficiency.
#[inline]
fn reduce_sum<S, A, D>(tensor: &TensorBase<S, D>) -> A
where
    S: RawStorage<Elem = A>,
    A: Numeric,
    D: Dimension,
{
    let mut acc = A::zero();
    for elem in tensor.iter() {
        acc = acc + *elem;
    }
    acc
}

/// Scalar reduction kernel: product of all elements.
#[inline]
fn reduce_prod<S, A, D>(tensor: &TensorBase<S, D>) -> A
where
    S: RawStorage<Elem = A>,
    A: Numeric,
    D: Dimension,
{
    let mut acc = A::one();
    for elem in tensor.iter() {
        acc = acc * *elem;
    }
    acc
}

/// Scalar reduction kernel: minimum element.
///
/// # Safety
/// Caller must ensure tensor is non-empty.
#[inline]
fn reduce_min<S, A, D>(tensor: &TensorBase<S, D>) -> A
where
    S: RawStorage<Elem = A>,
    A: PartialOrd,
    D: Dimension,
{
    let mut iter = tensor.iter();
    let mut min_val = *iter.next().unwrap(); // SAFETY: caller ensures non-empty
    for elem in iter {
        // NaN propagation: if either is NaN, PartialOrd returns false.
        // We treat this as "the new element is smaller" to propagate NaN.
        if !(min_val <= *elem) {
            min_val = *elem;
        }
    }
    min_val
}

/// Scalar reduction kernel: maximum element.
#[inline]
fn reduce_max<S, A, D>(tensor: &TensorBase<S, D>) -> A
where
    S: RawStorage<Elem = A>,
    A: PartialOrd,
    D: Dimension,
{
    let mut iter = tensor.iter();
    let mut max_val = *iter.next().unwrap();
    for elem in iter {
        if !(max_val >= *elem) {
            max_val = *elem;
        }
    }
    max_val
}

/// Scalar reduction kernel: index of minimum element.
#[inline]
fn reduce_argmin<S, A, D>(tensor: &TensorBase<S, D>) -> usize
where
    S: RawStorage<Elem = A>,
    A: PartialOrd,
    D: Dimension,
{
    let mut iter = tensor.iter().enumerate();
    let (_, mut min_val) = iter.next().unwrap();
    let mut min_idx = 0usize;
    for (idx, elem) in iter {
        if !(min_val <= *elem) {
            min_val = *elem;
            min_idx = idx;
        }
    }
    min_idx
}

/// Scalar reduction kernel: index of maximum element.
#[inline]
fn reduce_argmax<S, A, D>(tensor: &TensorBase<S, D>) -> usize
where
    S: RawStorage<Elem = A>,
    A: PartialOrd,
    D: Dimension,
{
    let mut iter = tensor.iter().enumerate();
    let (_, mut max_val) = iter.next().unwrap();
    let mut max_idx = 0usize;
    for (idx, elem) in iter {
        if !(max_val >= *elem) {
            max_val = *elem;
            max_idx = idx;
        }
    }
    max_idx
}
```

### 5.4 NaN 传播语义

`min`/`max`/`argmin`/`argmax` 使用 `!(a <= b)` / `!(a >= b)` 替代 `a > b` / `a < b` 进行比较。这是因为 IEEE 754 下 `NaN` 与任何值的 `<=`/`>=` 均返回 `false`，使得 `!(NaN <= x)` 为 `true`，从而确保一旦遇到 `NaN`，它会被保留在结果中。

```
输入序列: [1.0, NaN, 3.0]
min 逻辑: min_val = 1.0
          !(1.0 <= NaN) → !(false) → true → min_val = NaN
          !(NaN <= 3.0) → !(false) → true → min_val = NaN (unchanged)
结果: NaN ✓
```

### 5.5 Welford 在线算法（var/std）

```rust
/// Computes variance using Welford's online algorithm.
///
/// Numerically stable for large datasets with small variance.
/// Avoids catastrophic cancellation in the naive two-pass formula:
///   var = E[X^2] - (E[X])^2  (unstable)
///
/// Welford's recurrence:
///   M_k = M_{k-1} + (x_k - M_{k-1}) / k
///   S_k = S_{k-1} + (x_k - M_{k-1}) * (x_k - M_k)
///   var = S_n / (n - ddof)
fn reduce_var_welford<S, A, D>(tensor: &TensorBase<S, D>, ddof: usize) -> A
where
    S: RawStorage<Elem = A>,
    A: RealScalar,
    D: Dimension,
{
    let n = tensor.len();
    assert!(ddof < n, "ddof must be less than element count");

    let mut mean = A::zero();
    let mut m2 = A::zero();

    for (k, elem) in tensor.iter().enumerate() {
        let k_plus_1 = A::from((k + 1) as _);
        let delta = *elem - mean;
        mean = mean + delta / k_plus_1;
        let delta2 = *elem - mean;
        m2 = m2 + delta * delta2;
    }

    let count = A::from((n - ddof) as _);
    m2 / count
}
```

### 5.6 沿轴归约内核

沿轴归约的核心思路是将输入张量视为 "外层 × 轴长 × 内层" 的三段式结构，对外层每个位置沿轴遍历并累积：

```rust
/// Generic axis reduction kernel.
///
/// Conceptually, the tensor is viewed as:
///   - outer_dims: all dimensions before `axis`
///   - axis_len: the dimension being reduced
///   - inner_dims: all dimensions after `axis`
///
/// For each position (outer_idx, inner_idx), the kernel iterates
/// along `axis` and accumulates values.
fn reduce_sum_axis<S, A, D>(tensor: &TensorBase<S, D>, axis: usize) -> Tensor<A, D::Smaller>
where
    S: RawStorage<Elem = A>,
    A: Numeric,
    D: RemoveAxis,
{
    let out_shape = tensor.shape().remove_axis(axis);
    let axis_len = tensor.shape()[axis];
    let mut output = zeros::<A, D::Smaller>(out_shape.clone());

    // Compute iteration strides for the three segments
    let outer_count: usize = tensor.shape()[..axis].iter().product();
    let inner_count: usize = tensor.shape()[axis + 1..].iter().product();

    // For F-contiguous data: inner dims are contiguous, iterate efficiently
    // For general layout: use element-wise indexing

    for outer in 0..outer_count {
        for inner in 0..inner_count {
            let mut acc = A::zero();
            for k in 0..axis_len {
                // Compute flat source index for (outer, k, inner)
                let src_idx = flatten_index(tensor.shape(), axis, outer, k, inner);
                // SAFETY: index is computed from valid shape/axis
                let val = unsafe { tensor.get_unchecked(&src_idx) };
                acc = acc + *val;
            }
            // Write to output at (outer, inner)
            let dst_idx = flatten_output_index(&out_shape, outer, inner);
            unsafe {
                output.get_unchecked_mut(&dst_idx).write(acc);
            }
        }
    }

    output
}
```

**沿轴归约的优化路径：**

| 输入布局 | 轴位置 | 优化策略 |
|---------|--------|---------|
| F-contiguous | axis = 0 | 轴方向连续内存，可 SIMD 内轴 |
| F-contiguous | axis = last | 外层连续，内轴步进大 → 标量 |
| C-contiguous | axis = last | 轴方向连续内存，可 SIMD 内轴 |
| C-contiguous | axis = 0 | 外层步进大 → 标量 |
| 非连续 | 任意 | 标量循环 |

**轴连续性判断**（用于 SIMD 路径选择）：

```rust
/// Returns true if the specified axis has stride 1 (contiguous along axis).
#[inline]
fn is_axis_contiguous<S, D>(tensor: &TensorBase<S, D>, axis: usize) -> bool
where
    S: RawStorage,
    D: Dimension,
{
    tensor.strides()[axis] == 1
}
```

### 5.7 SIMD 归约路径（feature-gated）

```rust
#[cfg(feature = "simd")]
mod simd_reduce {
    use pulp::Arch;

    /// SIMD-accelerated sum for contiguous f64 data.
    pub(super) fn sum_f64_simd(data: &[f64]) -> f64 {
        let arch = Arch::new();
        arch.dispatch(|| {
            // Pseudocode: load SIMD lanes, accumulate horizontal sum
            // Use pulp's SIMD intrinsics for AVX2/AVX-512/SSE4.1/NEON
            // Tail elements processed by scalar loop
            let mut sum = 0.0_f64;
            // ... SIMD lanes ...
            sum
        })
    }

    /// SIMD-accelerated sum for contiguous f32 data.
    pub(super) fn sum_f32_simd(data: &[f32]) -> f32 {
        // Similar to f64 but with f32 SIMD lanes
        todo!()
    }

    /// SIMD-accelerated min/max for contiguous data.
    /// Uses platform-specific SIMD min/max instructions.
    pub(super) fn min_f64_simd(data: &[f64]) -> f64 { todo!() }
    pub(super) fn max_f64_simd(data: &[f64]) -> f64 { todo!() }
    pub(super) fn min_f32_simd(data: &[f32]) -> f32 { todo!() }
    pub(super) fn max_f32_simd(data: &[f32]) -> f32 { todo!() }
}
```

**SIMD 归约策略：**

1. 将数据按 SIMD 宽度分块（f64: 4/8/16 lanes, f32: 8/16/32 lanes）
2. 每个块使用 SIMD 加载、累加/比较
3. 最后一个不完整块用标量处理（tail loop）
4. 所有 SIMD lanes 水平归约（horizontal reduction）得到最终结果

### 5.8 并行归约路径（feature-gated）

```rust
#[cfg(feature = "parallel")]
mod parallel_reduce {
    use rayon::prelude::*;

    const PARALLEL_THRESHOLD: usize = 64 * 1024; // 64K elements
    const MIN_CHUNK_SIZE: usize = 4 * 1024;       // 4K elements per thread

    /// Parallel sum using rayon.
    ///
    /// Splits data into chunks, each thread computes a partial sum,
    /// then reduces partial sums sequentially.
    pub(super) fn sum_par<A>(data: &[A]) -> A
    where
        A: Numeric + Sync + Send,
    {
        data.par_chunks(MIN_CHUNK_SIZE)
            .map(|chunk| {
                chunk.iter().fold(A::zero(), |acc, &x| acc + x)
            })
            .reduce(|| A::zero(), |a, b| a + b)
    }

    /// Parallel min/max using rayon.
    pub(super) fn min_par<A>(data: &[A]) -> A
    where
        A: PartialOrd + Sync + Send + Copy,
    {
        data.par_chunks(MIN_CHUNK_SIZE)
            .map(|chunk| {
                chunk.iter().copied().reduce(|a, b| {
                    if !(a <= b) { b } else { a }
                }).unwrap()
            })
            .reduce(|a, b| {
                if !(a <= b) { b } else { a }
            })
            .unwrap()
    }

    /// Parallel axis reduction for large tensors.
    ///
    /// Parallelizes over the outer dimensions (not the axis being reduced).
    /// Each thread handles a subset of outer positions.
    pub(super) fn sum_axis_par<S, A, D>(
        tensor: &TensorBase<S, D>,
        axis: usize,
    ) -> Tensor<A, <D as RemoveAxis>::Smaller>
    where
        S: RawStorage<Elem = A>,
        A: Numeric + Sync + Send,
        D: RemoveAxis,
    {
        // Split outer dimension indices across threads
        // Each thread writes to non-overlapping output positions
        todo!()
    }
}
```

### 5.9 整数溢出检查

需求规定 `sum/prod` 作用于整数时溢出 **panic**（所有 build mode）。实现方式：

```rust
/// Wrapping addition that panics on overflow for integer types.
/// For float types, delegates to normal addition (IEEE 754 handles Inf).
#[inline]
fn checked_add<A: Numeric>(a: A, b: A) -> A {
    // For integer types, use checked_add and panic on overflow.
    // For float types, normal addition (Inf propagates naturally).
    // This is handled at the type-specialization level:
    //
    // Integer path:
    //   a.checked_add(&b).expect("sum overflow")
    //
    // Float path:
    //   a + b
    //
    // Implementation uses a trait or specialization.
    a + b // Simplified; actual impl uses checked arithmetic for integers
}
```

**具体实现策略：** 通过 `Numeric` trait 的关联方法 `checked_add` 表达：

```rust
pub trait Numeric: Element + Add<Output=Self> + Sub<Output=Self>
    + Mul<Output=Self> + Div<Output=Self> + Neg<Output=Self>
{
    /// Checked addition. Returns None on overflow.
    /// For float types, always returns Some(self + other).
    fn checked_numeric_add(&self, other: &Self) -> Option<Self>;

    /// Checked multiplication. Returns None on overflow.
    fn checked_numeric_mul(&self, other: &Self) -> Option<Self>;
}

// Integer impls return None on overflow
impl Numeric for i32 { /* checked_add via intrinsic */ }
impl Numeric for f64 { /* always Some */ }

// In reduction kernel:
fn reduce_sum<S, A, D>(tensor: &TensorBase<S, D>) -> A where A: Numeric {
    let mut acc = A::zero();
    for elem in tensor.iter() {
        acc = acc.checked_numeric_add(elem).expect("overflow in sum");
    }
    acc
}
```

### 5.10 Keepdim 辅助函数

```rust
/// Expands a reduced tensor by re-inserting the axis with size 1.
///
/// Input:  Tensor<A, D::Smaller> with shape e.g. [2, 4] (axis 1 removed)
/// Output: Tensor<A, D> with shape e.g. [2, 1, 4] (axis 1 re-inserted)
fn expand_keepdim<A, D>(
    reduced: Tensor<A, <D as RemoveAxis>::Smaller>,
    original_shape: &[usize],
    axis: usize,
) -> Tensor<A, D>
where
    A: Element,
    D: RemoveAxis + Clone,
{
    // Build new shape: insert 1 at `axis` position
    let mut new_shape: Vec<usize> = original_shape.to_vec();
    new_shape[axis] = 1;
    let new_dim = D::from_slice(&new_shape); // Requires D construction from slice

    // Reshape the reduced tensor to include the size-1 axis
    reduced.reshape_into(new_dim).expect("keepdim reshape should always succeed")
}
```

### 5.11 累积归约内核

```rust
/// Generic prefix scan for cumulative operations.
///
/// Scans along the specified axis, applying the accumulation function.
fn accumulate_scan<S, A, D, F>(
    tensor: &TensorBase<S, D>,
    axis: usize,
    init: A,
    accum_fn: F,
) -> Tensor<A, D>
where
    S: RawStorage<Elem = A>,
    A: Numeric,
    D: Dimension,
    F: Fn(&mut A, &A),
{
    let mut output = tensor.to_owned(); // Same shape, clone data

    let outer_count: usize = tensor.shape()[..axis].iter().product();
    let axis_len = tensor.shape()[axis];
    let inner_count: usize = tensor.shape()[axis + 1..].iter().product();

    for outer in 0..outer_count {
        for inner in 0..inner_count {
            let mut acc = init;
            for k in 0..axis_len {
                let idx = flatten_index(tensor.shape(), axis, outer, k, inner);
                let val = *unsafe { tensor.get_unchecked(&idx) };
                accum_fn(&mut acc, &val);
                unsafe {
                    *output.get_unchecked_mut(&idx) = acc;
                }
            }
        }
    }

    output
}
```

### 5.12 辅助函数

```rust
/// Validates that the axis index is within bounds.
///
/// # Panics
///
/// Panics if `axis >= ndim`.
#[inline]
fn validate_axis(ndim: usize, axis: usize) {
    assert!(axis < ndim, "axis {} is out of bounds for ndim {}", axis, ndim);
}

/// Flattens a (outer, k, inner) tuple into a multi-dimensional index.
///
/// Given a tensor with shape [d0, d1, ..., dn] and an axis to reduce,
/// maps (outer_idx, axis_idx, inner_idx) to a full index vector.
#[inline]
fn flatten_index(
    shape: &[usize],
    axis: usize,
    outer: usize,
    k: usize,
    inner: usize,
) -> Vec<usize> {
    let mut index = Vec::with_capacity(shape.len());
    // Decompose outer into indices for dims before axis
    let mut rem = outer;
    for i in (0..axis).rev() {
        index[i] = rem % shape[i];
        rem /= shape[i];
    }
    index[axis] = k;
    // Decompose inner into indices for dims after axis
    let mut rem = inner;
    for i in (axis + 1..shape.len()).rev() {
        index[i] = rem % shape[i];
        rem /= shape[i];
    }
    index
}

/// Flattens (outer, inner) into an output index (axis already removed).
#[inline]
fn flatten_output_index(
    out_shape: &[usize],
    outer: usize,
    inner: usize,
) -> Vec<usize> {
    // Similar to flatten_index but without the axis dimension
    todo!()
}
```

---

## 6. 实现任务拆分

> 每个任务约 10 分钟，可独立验证和提交。

### 6.1 基础设施

- [ ] **T1: 模块骨架 + Reduce trait 定义**
  - 文件: `src/ops/reduction.rs:1-80`
  - 内容: `Reduce` trait 定义（仅方法签名，不含实现）
  - 测试: 编译通过
  - 前置: tensor, element, dimension, error 模块完成
  - 预计: 10 min

- [ ] **T2: Accumulate trait 定义 + 模块骨架**
  - 文件: `src/ops/accumulate.rs:1-60`
  - 内容: `Accumulate` trait 定义、`src/ops/mod.rs` 添加 `pub mod`
  - 测试: 编译通过
  - 前置: T1
  - 预计: 10 min

- [ ] **T3: 辅助函数（validate_axis, flatten_index, flatten_output_index）**
  - 文件: `src/ops/reduction.rs`
  - 内容: `validate_axis()`, `flatten_index()`, `flatten_output_index()` 内部函数
  - 测试: 单元测试验证索引计算正确性（2D、3D 场景）
  - 前置: T1
  - 预计: 10 min

### 6.2 全局标量归约

- [ ] **T4: 全局 sum/prod 标量内核 + trait impl**
  - 文件: `src/ops/reduction.rs`
  - 内容: `reduce_sum()`, `reduce_prod()` 内核函数；`Reduce` trait 的 `sum()`, `prod()` 方法实现
  - 测试: `test_sum_1d`, `test_sum_2d`, `test_prod`, `test_sum_with_nan`
  - 前置: T1
  - 预计: 10 min

- [ ] **T5: 全局 min/max 标量内核 + trait impl**
  - 文件: `src/ops/reduction.rs`
  - 内容: `reduce_min()`, `reduce_max()` 内核；`Reduce` 的 `min()`, `max()` 方法（含空数组 Result）
  - 测试: `test_min_basic`, `test_max_basic`, `test_min_empty_returns_error`, `test_min_nan_propagation`
  - 前置: T4
  - 预计: 10 min

- [ ] **T6: 全局 argmin/argmax 标量内核 + trait impl**
  - 文件: `src/ops/reduction.rs`
  - 内容: `reduce_argmin()`, `reduce_argmax()` 内核；`Reduce` 的 `argmin()`, `argmax()` 方法
  - 测试: `test_argmin_basic`, `test_argmax_basic`, `test_argmin_empty`, `test_argmin_tie_returns_first`
  - 前置: T5
  - 预计: 10 min

- [ ] **T7: 全局 mean/var/std（Welford 算法）**
  - 文件: `src/ops/reduction.rs`
  - 内容: `reduce_var_welford()` 内核；`Reduce` 的 `mean()`, `var()`, `var_with_ddof()`, `std()`, `std_with_ddof()` 方法
  - 测试: `test_mean_basic`, `test_var_population`, `test_var_sample`, `test_std_basic`, `test_var_numerical_stability`
  - 前置: T4
  - 预计: 10 min

- [ ] **T8: 全局 all/any 布尔归约**
  - 文件: `src/ops/reduction.rs`
  - 内容: `Reduce` 的 `all()`, `any()` 方法
  - 测试: `test_all_true`, `test_all_false`, `test_all_empty_returns_true`, `test_any_empty_returns_false`
  - 前置: T1
  - 预计: 10 min

### 6.3 沿轴归约

- [ ] **T9: 沿轴 sum_axis/prod_axis**
  - 文件: `src/ops/reduction.rs`
  - 内容: `reduce_sum_axis()`, `reduce_prod_axis()` 内核；trait 方法实现
  - 测试: `test_sum_axis_0`, `test_sum_axis_1`, `test_sum_axis_2d`, `test_sum_axis_3d`
  - 前置: T3, T4
  - 预计: 10 min

- [ ] **T10: 沿轴 min_axis/max_axis**
  - 文件: `src/ops/reduction.rs`
  - 内容: `reduce_min_axis()`, `reduce_max_axis()` 内核；trait 方法实现
  - 测试: `test_min_axis_basic`, `test_max_axis_basic`, `test_min_axis_empty_returns_error`
  - 前置: T9
  - 预计: 10 min

- [ ] **T11: 沿轴 argmin_axis/argmax_axis**
  - 文件: `src/ops/reduction.rs`
  - 内容: `reduce_argmin_axis()`, `reduce_argmax_axis()` 内核；trait 方法实现
  - 测试: `test_argmin_axis_basic`, `test_argmax_axis_tie`
  - 前置: T9
  - 预计: 10 min

- [ ] **T12: 沿轴 mean_axis/var_axis/std_axis**
  - 文件: `src/ops/reduction.rs`
  - 内容: `reduce_mean_axis()`, `reduce_var_axis()` 内核；trait 方法实现
  - 测试: `test_mean_axis`, `test_var_axis`, `test_std_axis`, `test_var_axis_ddof`
  - 前置: T9
  - 预计: 10 min

- [ ] **T13: 沿轴 all_axis/any_axis**
  - 文件: `src/ops/reduction.rs`
  - 内容: `reduce_all_axis()`, `reduce_any_axis()` 内核；trait 方法实现
  - 测试: `test_all_axis`, `test_any_axis`
  - 前置: T9
  - 预计: 10 min

- [ ] **T14: Keepdim 变体（sum_axis_keepdim, mean_axis_keepdim 等）**
  - 文件: `src/ops/reduction.rs`
  - 内容: `expand_keepdim()` 辅助函数；所有 keepdim trait 方法实现
  - 测试: `test_sum_axis_keepdim_shape`, `test_mean_axis_keepdim_values`
  - 前置: T9, T10, T12
  - 预计: 10 min

### 6.4 累积归约

- [ ] **T15: cumsum 内核 + Accumulate trait impl**
  - 文件: `src/ops/accumulate.rs`
  - 内容: `accumulate_scan()` 通用内核；`Accumulate` 的 `cumsum()` 实现
  - 测试: `test_cumsum_1d`, `test_cumsum_2d_axis0`, `test_cumsum_2d_axis1`, `test_cumsum_nan_propagation`
  - 前置: T2
  - 预计: 10 min

- [ ] **T16: cumprod 内核 + Accumulate trait impl**
  - 文件: `src/ops/accumulate.rs`
  - 内容: `Accumulate` 的 `cumprod()` 实现（复用 `accumulate_scan`）
  - 测试: `test_cumprod_basic`, `test_cumprod_with_zero`, `test_cumprod_empty`
  - 前置: T15
  - 预计: 10 min

### 6.5 性能优化（feature-gated）

- [ ] **T17: 整数溢出检查（checked 算术）**
  - 文件: `src/ops/reduction.rs`, `src/element.rs`
  - 内容: `Numeric` trait 添加 `checked_numeric_add/mul` 关联方法；sum/prod 内核使用 checked 算术
  - 测试: `test_sum_i32_overflow_panics`, `test_prod_u8_overflow_panics`, `test_sum_f64_no_panic`
  - 前置: T4
  - 预计: 10 min

- [ ] **T18: SIMD 全局归约路径（f64 sum/min/max）**
  - 文件: `src/ops/reduction.rs` `#[cfg(feature = "simd")]` 块
  - 内容: pulp-based SIMD 内核 for f64 sum, min, max；dispatch 逻辑集成
  - 测试: `test_sum_simd_matches_scalar`, `test_min_simd_matches_scalar`
  - 前置: T4, T5
  - 预计: 10 min

- [ ] **T19: SIMD 全局归约路径（f32 + 扩展）**
  - 文件: `src/ops/reduction.rs`
  - 内容: pulp-based SIMD 内核 for f32；prod 的 SIMD 路径
  - 测试: `test_sum_f32_simd_matches_scalar`
  - 前置: T18
  - 预计: 10 min

- [ ] **T20: 并行全局归约路径**
  - 文件: `src/ops/reduction.rs` `#[cfg(feature = "parallel")]` 块
  - 内容: rayon-based 并行 sum, prod, min, max；dispatch 集成
  - 测试: `test_sum_par_matches_scalar`, `test_min_par_matches_scalar`
  - 前置: T4, T5
  - 预计: 10 min

- [ ] **T21: 并行沿轴归约路径**
  - 文件: `src/ops/reduction.rs`
  - 内容: rayon-based `sum_axis_par()`；沿轴并行 dispatch
  - 测试: `test_sum_axis_par_matches_scalar`
  - 前置: T9, T20
  - 预计: 10 min

### 6.6 集成与导出

- [ ] **T22: ops/mod.rs + lib.rs re-export**
  - 文件: `src/ops/mod.rs`, `src/lib.rs`
  - 内容: `pub mod reduction; pub mod accumulate;` 和 `pub use Reduce, Accumulate`
  - 测试: `use xenon::{Reduce, Accumulate}` 编译通过
  - 前置: T1-T16
  - 预计: 5 min

---

## 7. 测试计划

### 7.1 单元测试

位于 `src/ops/reduction.rs` 和 `src/ops/accumulate.rs` 中的 `#[cfg(test)] mod tests`：

| 分类 | 测试项 | 关键断言 |
|------|--------|----------|
| **全局 sum** | `test_sum_1d` | `[1,2,3] → 6` |
| | `test_sum_2d_f_contiguous` | `[[1,2],[3,4]] → 10` |
| | `test_sum_with_nan` | `[1.0, NaN, 3.0] → NaN` |
| | `test_sum_single_element` | `[42.0] → 42.0` |
| | `test_sum_i32_no_overflow` | `[1,2,3] → 6` |
| | `test_sum_i32_overflow_panics` | `[i32::MAX, 1] → panic` |
| **全局 prod** | `test_prod_basic` | `[2,3,4] → 24` |
| | `test_prod_with_zero` | `[1,0,3] → 0` |
| | `test_prod_with_nan` | `[2.0, NaN, 3.0] → NaN` |
| **全局 min/max** | `test_min_basic` | `[3,1,2] → 1` |
| | `test_max_basic` | `[3,1,2] → 3` |
| | `test_min_empty_returns_error` | `[] → Err(EmptyArray)` |
| | `test_min_nan_propagation` | `[1.0, NaN, 3.0] → NaN` |
| | `test_min_single_element` | `[42.0] → 42.0` |
| **全局 argmin/argmax** | `test_argmin_basic` | `[3,1,2] → 1` |
| | `test_argmax_basic` | `[3,1,2] → 0` |
| | `test_argmin_empty_returns_error` | `[] → Err(EmptyArray)` |
| | `test_argmin_tie_returns_first` | `[1,2,1] → 0` |
| **全局 mean** | `test_mean_basic` | `[2.0, 4.0] → 3.0` |
| | `test_mean_single` | `[5.0] → 5.0` |
| **全局 var/std** | `test_var_population` | `[2,4,4,4,5,5,7,9] → 4.0` |
| | `test_var_sample_ddof1` | ddof=1 验证 |
| | `test_std_basic` | `std = sqrt(var)` |
| | `test_var_numerical_stability` | 大 N + 小方差，Welford 精度 |
| | `test_var_ddof_panics` | `ddof >= len` panic |
| **全局 all/any** | `test_all_true` | `[true, true] → true` |
| | `test_all_false` | `[true, false] → false` |
| | `test_all_empty_true` | `[] → true` |
| | `test_any_empty_false` | `[] → false` |
| **沿轴 sum** | `test_sum_axis_0` | `[[1,2],[3,4]] axis=0 → [4,6]` |
| | `test_sum_axis_1` | `[[1,2],[3,4]] axis=1 → [3,7]` |
| | `test_sum_axis_3d` | `[2,3,4] axis=1 → [2,4]` |
| | `test_sum_axis_invalid_panics` | axis 越界 → panic |
| **沿轴 min/max** | `test_min_axis_basic` | 值正确 + 形状正确 |
| | `test_min_axis_empty_returns_error` | 轴长 0 → Err(EmptyArray) |
| **沿轴 argmin/argmax** | `test_argmin_axis_basic` | 返回正确索引张量 |
| | `test_argmin_axis_tie` | 多值相同时返回首个 |
| **沿轴 mean/var/std** | `test_mean_axis` | 沿轴均值正确 |
| | `test_var_axis_ddof` | ddof 参数正确应用 |
| **沿轴 all/any** | `test_all_axis` | 布尔张量沿轴 all 正确 |
| **keepdim** | `test_sum_axis_keepdim_shape` | 输出维度不变，axis 处为 1 |
| | `test_sum_axis_keepdim_values` | 值与 sum_axis 一致 |
| | `test_min_axis_keepdim_shape` | 形状包含 size-1 轴 |
| **cumsum** | `test_cumsum_1d` | `[1,2,3] → [1,3,6]` |
| | `test_cumsum_2d_axis0` | 沿轴 0 累积 |
| | `test_cumsum_2d_axis1` | 沿轴 1 累积 |
| | `test_cumsum_nan_propagation` | NaN 后续全为 NaN |
| | `test_cumsum_empty` | 空数组 → 空数组 |
| | `test_cumsum_single_element` | `[5.0] → [5.0]` |
| **cumprod** | `test_cumprod_basic` | `[1,2,3] → [1,2,6]` |
| | `test_cumprod_with_zero` | `[2,0,3] → [2,0,0]` |
| | `test_cumprod_empty` | 空数组 → 空数组 |

### 7.2 集成测试

位于 `tests/reduction.rs`：

| 分类 | 测试项 | 关键断言 |
|------|--------|----------|
| **跨存储类型** | `test_sum_owned` | Owned tensor 归约 |
| | `test_sum_view` | View tensor 归约 |
| | `test_sum_view_mut` | ViewMut 归约 |
| | `test_sum_arc_tensor` | ArcTensor 归约 |
| | `test_sum_axis_view_non_contiguous` | 非连续视图沿轴归约 |
| **非连续布局** | `test_sum_transposed` | 转置后归约正确 |
| | `test_sum_sliced` | 切片后归约正确 |
| | `test_sum_broadcast_view` | 广播视图归约 |
| **数值精度** | `test_sum_large_f64_precision` | 10^6 个 1.0 → 10^6（浮点精度验证） |
| | `test_var_large_numerical_stability` | Welford 算法精度对比 |
| **高维** | `test_sum_4d` | 4 维张量全局和沿轴归约 |
| | `test_sum_ixdyn` | 动态维度归约 |
| **边界情况** | `test_reduce_scalar_tensor` | 0 维张量（标量）归约 |
| | `test_reduce_large_tensor` | 大张量（触发并行路径） |
| | `test_argmin_flat_index_correctness` | argmin 返回的 flat index 可反查元素 |
| **特征组合** | `test_sum_then_mean` | 归约链式调用 |
| | `test_var_then_std` | var → sqrt 一致性 |
| | `test_cumsum_then_sum` | cumsum 最后元素 == sum |

### 7.3 边界测试

位于 `tests/edge_cases.rs`，重点关注归约相关边界：

| 测试项 | 场景 | 预期行为 |
|--------|------|---------|
| `test_reduce_empty_tensor` | shape=[0, 3] | sum→0, min→Err, all→true |
| `test_reduce_single_element` | shape=[1, 1] | sum→该元素, argmin→0 |
| `test_reduce_nan_all_elements` | 全 NaN | sum→NaN, min→NaN |
| `test_reduce_inf` | 含 ±Inf | sum→Inf, prod→Inf |
| `test_reduce_subnormal` | 含非正规浮点数 | 正确归约不丢失 |
| `test_reduce_i32_max_overflow` | sum 溢出 i32::MAX | panic |
| `test_reduce_u8_prod_overflow` | prod 溢出 u8 | panic |
| `test_reduce_axis_len_1` | 轴长为 1 | sum_axis→恒等变换 |
| `test_cumsum_nan_in_middle` | 中间出现 NaN | NaN 后续全为 NaN |
| `test_cumprod_all_zeros` | 全零 | 全零 |
| `test_sum_non_contiguous_stride` | 非连续步长张量 | 正确求和 |
| `test_sum_negative_stride` | 含负步长（翻转） | 正确求和 |
| `test_sum_zero_stride_broadcast` | 含零步长（广播） | 正确求和 |
| `test_var_ddof_equals_len_minus_1` | ddof = N-1 | 样本方差 |
| `test_var_ddof_out_of_bounds` | ddof >= N | panic |

### 7.4 属性测试

位于 `tests/property/`：

| 测试项 | 不变量 |
|--------|--------|
| `prop_sum_commutative` | `a.sum() == a.to_f_contiguous().sum()`（布局无关） |
| `prop_sum_addition_distributes` | `(a + b).sum() == a.sum() + b.sum()`（形状一致时） |
| `prop_min_le_max` | `a.min() <= a.max()` |
| `prop_argmin_index_valid` | `a[argmin] == a.min()` |
| `prop_argmax_index_valid` | `a[argmax] == a.max()` |
| `prop_cumsum_last_equals_sum` | `a.cumsum(0).last() == a.sum()` |
| `prop_var_non_negative` | `a.var() >= 0.0` |
| `prop_std_non_negative` | `a.std() >= 0.0` |
| `prop_mean_from_sum` | `a.mean() == a.sum() / a.len()` |
| `prop_all_from_any` | `a.all() → !a.any() == false` (对全 true) |

### 7.5 基准测试

位于 `benches/reduction.rs`：

| 基准 | 数据规模 | 变体 |
|------|---------|------|
| `bench_sum_global` | 10K, 100K, 1M | f32, f64 |
| `bench_sum_axis` | [1000, 1000] | axis=0, axis=1 |
| `bench_min_global` | 10K, 100K | f64 |
| `bench_var_global` | 10K, 100K | f64 (Welford) |
| `bench_cumsum_1d` | 1M | f64 |
| `bench_sum_non_contiguous` | 10K (stride=2) | f64 |
| `bench_sum_simd_vs_scalar` | 100K | contiguous |
| `bench_sum_parallel_vs_single` | 1M | with/without rayon |

---

## 附录 A：NaN 传播行为汇总

| 操作 | 全 NaN 输入 | 含 NaN 输入 | 无 NaN 输入 |
|------|------------|------------|------------|
| sum | NaN | NaN | 正常求和 |
| prod | NaN | NaN | 正常求积 |
| min | NaN | NaN | 最小值 |
| max | NaN | NaN | 最大值 |
| mean | NaN | NaN | 均值 |
| var | NaN | NaN | 方差 |
| std | NaN | NaN | 标准差 |
| argmin | 首个 NaN 的索引 | 首个 NaN 或最小值的索引 | 最小值索引 |
| argmax | 首个 NaN 的索引 | 首个 NaN 或最大值的索引 | 最大值索引 |
| cumsum | 全 NaN | NaN 首次出现后全 NaN | 正常累积 |
| cumprod | 全 NaN | NaN 首次出现后全 NaN | 正常累积 |
| all | N/A (bool only) | N/A | N/A |
| any | N/A (bool only) | N/A | N/A |

## 附录 B：元素类型支持矩阵

| 操作 | f32 | f64 | i8~i64 | u8~u64 | bool | Complex |
|------|-----|-----|--------|--------|------|---------|
| sum | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| prod | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| min | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| max | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| mean | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| var | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| std | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| argmin | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| argmax | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| all | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| any | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| cumsum | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| cumprod | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |

## 附录 C：与 NumPy 行为对比

| 行为 | NumPy | Xenon | 说明 |
|------|-------|-------|------|
| 空数组 min/max | ValueError (raise) | `Err(EmptyArray)` | Result 替代异常 |
| 空数组 sum | 0 (identity) | `A::zero()` | 一致 |
| 空数组 all | True | true | 一致（空真） |
| 空数组 any | False | false | 一致 |
| var 默认 ddof | 0 | 0 | 一致 |
| argmin/argmax 平局 | 首个索引 | 首个索引 | 一致 |
| 整数 sum 溢出 | 静默溢出（C 行为） | panic | Xenon 更安全 |
| cumsum 遇 NaN | NaN 传播 | NaN 传播 | 一致 |
| Complex min/max | 不支持 | 不支持 | 一致 |
| bool sum | 转为 int | ❌ 不支持 | Xenon 需显式 cast |
