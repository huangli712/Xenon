# 归约运算模块设计文档

> **文档版本**: v1.0  
> **最后更新**: 2026-03-28  
> **模块路径**: `src/ops/`  
> **需求来源**: require-v18.md §10.3

---

## 1. 模块概述

### 1.1 设计哲学

归约运算（Reduction Operations）是 Xenon 数值计算的核心能力之一，用于将多维数组聚合为标量或低维数组。设计遵循以下原则：

- **类型安全边界**: 数值归约仅支持 `Numeric` 类型，布尔归约仅支持 `bool`
- **空数组安全**: 明确定义空数组行为，`min/max/argmin/argmax` 返回错误，其他返回单位元
- **NaN 传播明确**: 每种归约有明确定义的 NaN 行为
- **整数溢出检查**: 整数 `sum/prod` 使用 checked arithmetic，溢出时 panic
- **SIMD 友好**: 核心归约（sum/prod/min/max）支持向量化实现

### 1.2 在架构中的位置

```
┌─────────────────────────────────────────────────────────┐
│                    用户层 API                            │
│    tensor.sum(), tensor.mean(), tensor.argmax()         │
└─────────────────────┬───────────────────────────────────┘
                      │ 方法调用
┌─────────────────────▼───────────────────────────────────┐
│              归约运算模块 (本模块)                        │
│  reduction.rs: sum, prod, mean, var, std, min, max      │
│                argmin, argmax, all, any                 │
│  accumulate.rs: cumsum, cumprod                         │
│  set_ops.rs: unique, bincount, histogram                │
└─────────────────────┬───────────────────────────────────┘
                      │ 依赖
┌─────────────────────▼───────────────────────────────────┐
│            底层模块                                      │
│  iter (迭代器) | layout (布局) | simd (向量化)           │
│  element (类型约束) | tensor (核心结构) | error          │
└─────────────────────────────────────────────────────────┘
```

### 1.3 运算分类

| 类别 | 操作 | 返回类型 | 文件 |
|------|------|----------|------|
| 全局归约 | sum, prod, mean, var, std | 标量 | `reduction.rs` |
| 极值归约 | min, max, argmin, argmax | 标量 / 索引 | `reduction.rs` |
| 布尔归约 | all, any | `bool` | `reduction.rs` |
| 沿轴归约 | 以上所有 + `axis` 参数 | 降维 Tensor | `reduction.rs` |
| 累积运算 | cumsum, cumprod | 同形状 Tensor | `accumulate.rs` |
| 集合操作 | unique, unique_counts, unique_inverse | Tensor1 | `set_ops.rs` |
| 计数操作 | bincount, histogram, histogram_bin_edges | Tensor1 | `set_ops.rs` |

---

## 2. 文件结构

```
src/ops/
├── mod.rs             # 模块入口，re-export 公开 trait 和函数
├── reduction.rs       # 归约运算（sum/prod/mean/var/std/min/max/argmin/argmax/all/any）
├── accumulate.rs      # 累积运算（cumsum/cumprod）
└── set_ops.rs         # 集合操作（unique/bincount/histogram）
```

### 2.1 各文件职责

| 文件 | 职责 | 核心内容 |
|------|------|----------|
| `mod.rs` | 模块组织与导出 | re-export 归约函数、累积函数、集合操作 |
| `reduction.rs` | 核心归约 | `sum`, `prod`, `mean`, `var`, `std`, `min`, `max`, `argmin`, `argmax`, `all`, `any` |
| `accumulate.rs` | 累积运算 | `cumsum`, `cumprod` |
| `set_ops.rs` | 集合与计数 | `unique`, `unique_counts`, `unique_inverse`, `bincount`, `histogram`, `histogram_bin_edges` |

### 2.2 模块依赖

| 依赖模块 | 用途 |
|---------|------|
| `crate::tensor` | `TensorBase<S, D>` 核心类型 |
| `crate::element` | `Element`, `Numeric`, `RealScalar` trait 约束 |
| `crate::dimension` | `Dimension`, `Ix0`, `Ix1`, `Ix2`, etc. |
| `crate::layout` | `LayoutFlags`, 连续性检查 |
| `crate::error` | `EmptyArray`, `InvalidAxis` 错误类型 |
| `crate::simd` (可选) | SIMD 归约路径 |
| `crate::parallel` (可选) | 并行归约路径 |

---

## 3. 全局归约设计

全局归约将整个数组归约为单个标量值。

### 3.1 sum 设计

#### 3.1.1 方法签名

```rust
// reduction.rs

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Numeric,
{
    /// 全局求和
    ///
    /// 计算数组中所有元素的总和。
    ///
    /// # 返回值
    ///
    /// 所有元素的和（类型 `A`）
    ///
    /// # 空数组行为
    ///
    /// 空数组返回 `A::zero()`
    ///
    /// # 整数溢出
    ///
    /// 整数类型在溢出时 panic（debug 和 release 模式均如此）
    ///
    /// # NaN 行为
    ///
    /// 浮点类型：任一元素为 NaN 则返回 NaN
    ///
    /// # 示例
    ///
    /// ```
    /// use xenon::Tensor1;
    ///
    /// let a = Tensor1::from_vec(vec![1.0, 2.0, 3.0]);
    /// assert_eq!(a.sum(), 6.0);
    ///
    /// // 空数组
    /// let empty: Tensor1<f64> = Tensor1::zeros([0]);
    /// assert_eq!(empty.sum(), 0.0);
    /// ```
    pub fn sum(&self) -> A
    where
        A: Numeric,
    {
        // 实现...
    }
}
```

#### 3.1.2 整数溢出策略

```rust
// 整数类型使用 checked_add
impl<S, D> TensorBase<S, D>
where
    S: Storage<Elem = i32>,
    D: Dimension,
{
    pub fn sum(&self) -> i32 {
        let mut acc: i32 = 0;
        for &x in self.iter() {
            acc = acc.checked_add(x).expect("integer overflow in sum");
        }
        acc
    }
}

// 浮点类型使用常规加法（NaN 传播）
impl<S, D> TensorBase<S, D>
where
    S: Storage<Elem = f64>,
    D: Dimension,
{
    pub fn sum(&self) -> f64 {
        self.iter().fold(0.0, |acc, &x| acc + x)
    }
}
```

#### 3.1.3 类型支持

| 类型 | Trait 约束 | 溢出行为 | NaN 行为 |
|------|-----------|----------|----------|
| i8/i16/i32/i64/u8/u16/u32/u64 | `Numeric` | panic | N/A |
| f32/f64 | `Numeric` | Inf | NaN 传播 |
| Complex<f32/f64> | `Numeric` | Inf | NaN 传播 |

---

### 3.2 prod 设计

#### 3.2.1 方法签名

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Numeric,
{
    /// 全局求积
    ///
    /// 计算数组中所有元素的乘积。
    ///
    /// # 返回值
    ///
    /// 所有元素的积（类型 `A`）
    ///
    /// # 空数组行为
    ///
    /// 空数组返回 `A::one()`
    ///
    /// # 整数溢出
    ///
    /// 整数类型在溢出时 panic
    ///
    /// # NaN 行为
    ///
    /// 浮点类型：任一元素为 NaN 则返回 NaN
    ///
    /// # 示例
    ///
    /// ```
    /// use xenon::Tensor1;
    ///
    /// let a = Tensor1::from_vec(vec![2.0, 3.0, 4.0]);
    /// assert_eq!(a.prod(), 24.0);
    ///
    /// // 空数组
    /// let empty: Tensor1<f64> = Tensor1::zeros([0]);
    /// assert_eq!(empty.prod(), 1.0);
    /// ```
    pub fn prod(&self) -> A {
        // 实现...
    }
}
```

---

### 3.3 mean 设计

#### 3.3.1 方法签名

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: RealScalar,
{
    /// 全局均值
    ///
    /// 计算数组中所有元素的算术平均值。
    ///
    /// # 返回值
    ///
    /// 所有元素的均值（类型 `A`）
    ///
    /// # 空数组行为
    ///
    /// 空数组返回 NaN
    ///
    /// # NaN 行为
    ///
    /// 任一元素为 NaN 则返回 NaN
    ///
    /// # 约束
    ///
    /// 仅支持 `RealScalar`（f32/f64），不支持整数或复数
    ///
    /// # 示例
    ///
    /// ```
    /// use xenon::Tensor1;
    ///
    /// let a = Tensor1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    /// assert_eq!(a.mean(), 2.5);
    ///
    /// // 空数组
    /// let empty: Tensor1<f64> = Tensor1::zeros([0]);
    /// assert!(empty.mean().is_nan());
    /// ```
    pub fn mean(&self) -> A {
        let n = self.len();
        if n == 0 {
            return A::nan();
        }
        self.sum() / A::from(n).unwrap()
    }
}
```

**设计决策**: `mean` 仅支持 `RealScalar`，因为：
1. 整数除法会截断，结果不准确
2. 复数均值虽可定义，但需求文档未提及
3. 与 NumPy/PyTorch 行为一致

---

### 3.4 var/std 设计

#### 3.4.1 方法签名

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: RealScalar,
{
    /// 全局方差
    ///
    /// 计算数组中所有元素的方差。
    ///
    /// # 参数
    ///
    /// * `ddof` - Delta Degrees of Freedom（自由度调整值）
    ///   - `ddof=0`（默认）: 有偏估计，除以 N
    ///   - `ddof=1`: 无偏估计，除以 N-1（样本方差）
    ///
    /// # 返回值
    ///
    /// 方差值（类型 `A`）
    ///
    /// # 空数组行为
    ///
    /// - `ddof=0`: 单元素数组返回 0.0，空数组返回 NaN
    /// - `ddof=1`: 单元素数组返回 NaN（除以 0），空数组返回 NaN
    ///
    /// # NaN 行为
    ///
    /// 任一元素为 NaN 则返回 NaN
    ///
    /// # 公式
    ///
    /// ```text
    /// var = sum((x - mean)^2) / (N - ddof)
    /// ```
    ///
    /// # 示例
    ///
    /// ```
    /// use xenon::Tensor1;
    ///
    /// let a = Tensor1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    /// 
    /// // 有偏估计（默认，与 NumPy 一致）
    /// assert_eq!(a.var(0), 2.0);  // 10/5 = 2.0
    ///
    /// // 无偏估计（样本方差）
    /// assert_eq!(a.var(1), 2.5);  // 10/4 = 2.5
    /// ```
    pub fn var(&self, ddof: usize) -> A {
        let n = self.len();
        if n == 0 {
            return A::nan();
        }
        
        let mean = self.mean();
        let sum_sq: A = self.iter()
            .map(|&x| (x - mean) * (x - mean))
            .fold(A::zero(), |acc, x| acc + x);
        
        let divisor = A::from(n - ddof).unwrap_or(A::nan());
        sum_sq / divisor
    }
    
    /// 全局标准差
    ///
    /// 计算数组中所有元素的标准差（方差的平方根）。
    ///
    /// # 参数
    ///
    /// * `ddof` - Delta Degrees of Freedom，同 `var()`
    ///
    /// # 返回值
    ///
    /// 标准差值（类型 `A`）
    ///
    /// # 示例
    ///
    /// ```
    /// use xenon::Tensor1;
    ///
    /// let a = Tensor1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    /// assert_eq!(a.std(0), 2.0_f64.sqrt());
    /// ```
    pub fn std(&self, ddof: usize) -> A {
        self.var(ddof).sqrt()
    }
}
```

#### 3.4.2 ddof 参数说明

| ddof | 名称 | 公式 | 用途 |
|------|------|------|------|
| 0 | 有偏估计（总体方差） | `sum((x-μ)²) / N` | 已知总体，与 NumPy 默认一致 |
| 1 | 无偏估计（样本方差） | `sum((x-μ)²) / (N-1)` | 样本推断总体 |

---

### 3.5 min/max 设计

#### 3.5.1 方法签名

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + PartialOrd,
{
    /// 全局最小值
    ///
    /// 返回数组中的最小元素。
    ///
    /// # 返回值
    ///
    /// `Result<A, EmptyArray>` - 最小值或空数组错误
    ///
    /// # 空数组行为
    ///
    /// 空数组返回 `Err(EmptyArray)`
    ///
    /// # NaN 行为
    ///
    /// 浮点类型：任一元素为 NaN 则返回 NaN（NaN 传播）
    ///
    /// # 多值行为
    ///
    /// 存在多个相同最小值时，返回第一个（按遍历顺序）
    ///
    /// # 示例
    ///
    /// ```
    /// use xenon::Tensor1;
    ///
    /// let a = Tensor1::from_vec(vec![3.0, 1.0, 2.0, 1.0]);
    /// assert_eq!(a.min().unwrap(), 1.0);  // 第一个 1.0
    ///
    /// // 空数组
    /// let empty: Tensor1<f64> = Tensor1::zeros([0]);
    /// assert!(empty.min().is_err());
    /// ```
    pub fn min(&self) -> Result<A, EmptyArray>
    where
        A: PartialOrd,
    {
        if self.len() == 0 {
            return Err(EmptyArray);
        }
        
        let mut min_val = None;
        for &x in self.iter() {
            min_val = Some(match min_val {
                None => x,
                Some(m) => {
                    if x < m { x } else { m }
                }
            });
        }
        Ok(min_val.unwrap())
    }
    
    /// 全局最大值
    ///
    /// 返回数组中的最大元素。
    ///
    /// # 返回值
    ///
    /// `Result<A, EmptyArray>` - 最大值或空数组错误
    ///
    /// # 空数组行为
    ///
    /// 空数组返回 `Err(EmptyArray)`
    ///
    /// # NaN 行为
    ///
    /// 浮点类型：任一元素为 NaN 则返回 NaN（NaN 传播）
    ///
    /// # 多值行为
    ///
    /// 存在多个相同最大值时，返回第一个（按遍历顺序）
    ///
    /// # 示例
    ///
    /// ```
    /// use xenon::Tensor1;
    ///
    /// let a = Tensor1::from_vec(vec![1.0, 3.0, 2.0, 3.0]);
    /// assert_eq!(a.max().unwrap(), 3.0);  // 第一个 3.0
    /// ```
    pub fn max(&self) -> Result<A, EmptyArray>
    where
        A: PartialOrd,
    {
        // 类似 min 实现
    }
}
```

#### 3.5.2 NaN 传播实现

```rust
// NaN 传播语义：任一参数为 NaN 则返回 NaN
fn min_with_nan<A: RealScalar>(a: A, b: A) -> A {
    if a.is_nan() || b.is_nan() {
        A::nan()
    } else if a < b {
        a
    } else {
        b
    }
}
```

---

### 3.6 argmin/argmax 设计

#### 3.6.1 方法签名

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + PartialOrd,
{
    /// 全局最小值索引
    ///
    /// 返回最小元素的扁平索引。
    ///
    /// # 返回值
    ///
    /// `Result<usize, EmptyArray>` - 最小值索引或空数组错误
    ///
    /// # 空数组行为
    ///
    /// 空数组返回 `Err(EmptyArray)`
    ///
    /// # NaN 行为
    ///
    /// 浮点类型：存在 NaN 时，返回第一个 NaN 的索引
    ///
    /// # 多值行为
    ///
    /// 存在多个相同最小值时，返回第一个（按遍历顺序）
    ///
    /// # 示例
    ///
    /// ```
    /// use xenon::Tensor1;
    ///
    /// let a = Tensor1::from_vec(vec![3.0, 1.0, 2.0, 1.0]);
    /// assert_eq!(a.argmin().unwrap(), 1);  // 第一个 1.0 的索引
    /// ```
    pub fn argmin(&self) -> Result<usize, EmptyArray>
    where
        A: PartialOrd,
    {
        if self.len() == 0 {
            return Err(EmptyArray);
        }
        
        let mut min_idx = 0;
        let mut min_val = unsafe { *self.as_ptr() };
        
        for (i, &x) in self.iter().enumerate() {
            if x < min_val {
                min_val = x;
                min_idx = i;
            }
        }
        Ok(min_idx)
    }
    
    /// 全局最大值索引
    ///
    /// 返回最大元素的扁平索引。
    ///
    /// # 返回值
    ///
    /// `Result<usize, EmptyArray>` - 最大值索引或空数组错误
    ///
    /// # 空数组行为
    ///
    /// 空数组返回 `Err(EmptyArray)`
    ///
    /// # NaN 行为
    ///
    /// 浮点类型：存在 NaN 时，返回第一个 NaN 的索引
    ///
    /// # 多值行为
    ///
    /// 存在多个相同最大值时，返回第一个（按遍历顺序）
    ///
    /// # 示例
    ///
    /// ```
    /// use xenon::Tensor1;
    ///
    /// let a = Tensor1::from_vec(vec![1.0, 3.0, 2.0, 3.0]);
    /// assert_eq!(a.argmax().unwrap(), 1);  // 第一个 3.0 的索引
    /// ```
    pub fn argmax(&self) -> Result<usize, EmptyArray>
    where
        A: PartialOrd,
    {
        // 类似 argmin 实现
    }
}
```

---

### 3.7 all/any 设计（布尔归约）

#### 3.7.1 方法签名

```rust
impl<S, D> TensorBase<S, D>
where
    S: Storage<Elem = bool>,
    D: Dimension,
{
    /// 全局逻辑与
    ///
    /// 检查所有元素是否为 `true`。
    ///
    /// # 返回值
    ///
    /// `bool` - 所有元素为 `true` 时返回 `true`
    ///
    /// # 空数组行为
    ///
    /// 空数组返回 `true`（空真）
    ///
    /// # 示例
    ///
    /// ```
    /// use xenon::Tensor1;
    ///
    /// let a = Tensor1::from_vec(vec![true, true, true]);
    /// assert_eq!(a.all(), true);
    ///
    /// let b = Tensor1::from_vec(vec![true, false, true]);
    /// assert_eq!(b.all(), false);
    ///
    /// // 空数组
    /// let empty: Tensor1<bool> = Tensor1::zeros([0]);
    /// assert_eq!(empty.all(), true);  // 空真
    /// ```
    pub fn all(&self) -> bool {
        self.iter().all(|&x| x)
    }
    
    /// 全局逻辑或
    ///
    /// 检查是否存在任意元素为 `true`。
    ///
    /// # 返回值
    ///
    /// `bool` - 任一元素为 `true` 时返回 `true`
    ///
    /// # 空数组行为
    ///
    /// 空数组返回 `false`（空假）
    ///
    /// # 示例
    ///
    /// ```
    /// use xenon::Tensor1;
    ///
    /// let a = Tensor1::from_vec(vec![false, true, false]);
    /// assert_eq!(a.any(), true);
    ///
    /// let b = Tensor1::from_vec(vec![false, false, false]);
    /// assert_eq!(b.any(), false);
    ///
    /// // 空数组
    /// let empty: Tensor1<bool> = Tensor1::zeros([0]);
    /// assert_eq!(empty.any(), false);  // 空假
    /// ```
    pub fn any(&self) -> bool {
        self.iter().any(|&x| x)
    }
}
```

---

### 3.8 全局归约语义总结

| 操作 | 返回类型 | 空数组行为 | NaN 行为 | 整数溢出 |
|------|----------|-----------|----------|----------|
| `sum` | `A` | `A::zero()` | NaN 传播 | panic |
| `prod` | `A` | `A::one()` | NaN 传播 | panic |
| `mean` | `A` | `NaN` | NaN 传播 | N/A |
| `var` | `A` | `NaN` | NaN 传播 | N/A |
| `std` | `A` | `NaN` | NaN 传播 | N/A |
| `min` | `Result<A, EmptyArray>` | `Err(EmptyArray)` | NaN 传播 | N/A |
| `max` | `Result<A, EmptyArray>` | `Err(EmptyArray)` | NaN 传播 | N/A |
| `argmin` | `Result<usize, EmptyArray>` | `Err(EmptyArray)` | 返回 NaN 索引 | N/A |
| `argmax` | `Result<usize, EmptyArray>` | `Err(EmptyArray)` | 返回 NaN 索引 | N/A |
| `all` | `bool` | `true`（空真） | N/A | N/A |
| `any` | `bool` | `false`（空假） | N/A | N/A |

---

## 4. 沿轴归约设计

沿轴归约将多维数组沿指定轴归约，返回降维后的数组。

### 4.1 通用框架

#### 4.1.1 方法签名模式

```rust
// reduction.rs

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Numeric,
{
    /// 沿轴求和
    ///
    /// # 参数
    ///
    /// * `axis` - 归约轴（0-indexed）
    ///
    /// # 返回值
    ///
    /// 降维后的 `Tensor<A, D::Smaller>`
    ///
    /// # 错误
    ///
    /// * `InvalidAxis` - 轴索引超出范围
    ///
    /// # 示例
    ///
    /// ```
    /// use xenon::Tensor2;
    ///
    /// let a = Tensor2::from_shape_vec([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// 
    /// // 沿轴 0 归约（行方向）
    /// let row_sum = a.sum_axis(0).unwrap();  // shape: [3]
    /// // [1+4, 2+5, 3+6] = [5.0, 7.0, 9.0]
    ///
    /// // 沿轴 1 归约（列方向）
    /// let col_sum = a.sum_axis(1).unwrap();  // shape: [2]
    /// // [1+2+3, 4+5+6] = [6.0, 15.0]
    /// ```
    pub fn sum_axis(&self, axis: usize) -> Result<Tensor<A, D::Smaller>, InvalidAxis>
    where
        D: RemoveAxis,
    {
        // 轴检查
        if axis >= self.ndim() {
            return Err(InvalidAxis { 
                axis, 
                ndim: self.ndim() 
            });
        }
        
        // 计算输出形状
        let out_shape = self.shape().remove_axis(axis);
        let mut result = Tensor::zeros(out_shape);
        
        // 沿轴归约
        // ...
        
        Ok(result)
    }
}
```

### 4.2 返回类型（降维）

| 输入维度 | 沿轴归约后 | 类型变化 |
|----------|-----------|----------|
| `Tensor<A, Ix1>` | 标量 | `Tensor<A, Ix0>` |
| `Tensor<A, Ix2>` | 1D | `Tensor<A, Ix1>` |
| `Tensor<A, Ix3>` | 2D | `Tensor<A, Ix2>` |
| `Tensor<A, IxDyn>` | 动态 | `Tensor<A, IxDyn>` |

### 4.3 沿轴归约方法列表

```rust
// 所有全局归约都有对应的沿轴版本

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    // 数值归约
    pub fn sum_axis(&self, axis: usize) -> Result<Tensor<A, D::Smaller>, InvalidAxis>
        where A: Numeric, D: RemoveAxis;
    
    pub fn prod_axis(&self, axis: usize) -> Result<Tensor<A, D::Smaller>, InvalidAxis>
        where A: Numeric, D: RemoveAxis;
    
    pub fn mean_axis(&self, axis: usize) -> Result<Tensor<A, D::Smaller>, InvalidAxis>
        where A: RealScalar, D: RemoveAxis;
    
    pub fn var_axis(&self, axis: usize, ddof: usize) -> Result<Tensor<A, D::Smaller>, InvalidAxis>
        where A: RealScalar, D: RemoveAxis;
    
    pub fn std_axis(&self, axis: usize, ddof: usize) -> Result<Tensor<A, D::Smaller>, InvalidAxis>
        where A: RealScalar, D: RemoveAxis;
    
    // 极值归约
    pub fn min_axis(&self, axis: usize) -> Result<Tensor<A, D::Smaller>, InvalidAxis>
        where A: Element + PartialOrd, D: RemoveAxis;
    
    pub fn max_axis(&self, axis: usize) -> Result<Tensor<A, D::Smaller>, InvalidAxis>
        where A: Element + PartialOrd, D: RemoveAxis;
    
    // 索引归约
    pub fn argmin_axis(&self, axis: usize) -> Result<Tensor<usize, D::Smaller>, InvalidAxis>
        where A: Element + PartialOrd, D: RemoveAxis;
    
    pub fn argmax_axis(&self, axis: usize) -> Result<Tensor<usize, D::Smaller>, InvalidAxis>
        where A: Element + PartialOrd, D: RemoveAxis;
    
    // 布尔归约
    pub fn all_axis(&self, axis: usize) -> Result<Tensor<bool, D::Smaller>, InvalidAxis>
        where S: Storage<Elem = bool>, D: RemoveAxis;
    
    pub fn any_axis(&self, axis: usize) -> Result<Tensor<bool, D::Smaller>, InvalidAxis>
        where S: Storage<Elem = bool>, D: RemoveAxis;
}
```

### 4.4 沿轴归约实现策略

```rust
// 沿轴归约的通用实现
fn reduce_axis<A, S, D, F>(
    tensor: &TensorBase<S, D>,
    axis: usize,
    init: A,
    f: F,
) -> Result<Tensor<A, D::Smaller>, InvalidAxis>
where
    S: Storage<Elem = A>,
    D: RemoveAxis,
    F: Fn(A, A) -> A,
{
    if axis >= tensor.ndim() {
        return Err(InvalidAxis { axis, ndim: tensor.ndim() });
    }
    
    let out_shape = tensor.shape().remove_axis(axis);
    let mut result = Tensor::zeros(out_shape);
    
    // 使用 axis_iter 沿轴迭代
    for (i, lane) in tensor.axis_iter(axis).enumerate() {
        // 对每个 lane 执行归约
        let reduced = lane.iter().fold(init, &f);
        // 将结果写入输出
        // ...
    }
    
    Ok(result)
}
```

---

## 5. 累积运算设计

累积运算沿指定轴执行累积操作，返回与输入相同形状的数组。

### 5.1 cumsum 设计

#### 5.1.1 方法签名

```rust
// accumulate.rs

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Numeric,
{
    /// 累积求和
    ///
    /// 沿指定轴计算累积和。输出形状与输入相同。
    ///
    /// # 参数
    ///
    /// * `axis` - 累积轴（0-indexed）
    ///
    /// # 返回值
    ///
    /// 与输入形状相同的 `Tensor<A, D>`
    ///
    /// # 空数组行为
    ///
    /// 空数组返回同形状空数组
    ///
    /// # NaN 传播
    ///
    /// 遇到 NaN 时，该位置及后续所有位置均为 NaN
    ///
    /// # 整数溢出
    ///
    /// 整数类型在溢出时 panic
    ///
    /// # 示例
    ///
    /// ```
    /// use xenon::Tensor1;
    ///
    /// let a = Tensor1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    /// let cum = a.cumsum(0);
    /// assert_eq!(cum, Tensor1::from_vec(vec![1.0, 3.0, 6.0, 10.0]));
    ///
    /// // NaN 传播
    /// let b = Tensor1::from_vec(vec![1.0, f64::NAN, 3.0, 4.0]);
    /// let cum_b = b.cumsum(0);
    /// // [1.0, NaN, NaN, NaN]
    /// ```
    pub fn cumsum(&self, axis: usize) -> Tensor<A, D> {
        let mut result = self.to_owned();
        
        // 沿轴累积
        let axis_len = self.shape()[axis];
        
        if axis_len == 0 {
            return result;  // 空数组返回空
        }
        
        // 使用 axis_iter_mut 累积
        for mut lane in result.axis_iter_mut(axis) {
            let mut acc = A::zero();
            for elem in lane.iter_mut() {
                acc = acc + *elem;
                *elem = acc;
            }
        }
        
        result
    }
}
```

### 5.2 cumprod 设计

#### 5.2.1 方法签名

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Numeric,
{
    /// 累积求积
    ///
    /// 沿指定轴计算累积积。输出形状与输入相同。
    ///
    /// # 参数
    ///
    /// * `axis` - 累积轴（0-indexed）
    ///
    /// # 返回值
    ///
    /// 与输入形状相同的 `Tensor<A, D>`
    ///
    /// # 空数组行为
    ///
    /// 空数组返回同形状空数组
    ///
    /// # NaN 传播
    ///
    /// 遇到 NaN 时，该位置及后续所有位置均为 NaN
    ///
    /// # 整数溢出
    ///
    /// 整数类型在溢出时 panic
    ///
    /// # 示例
    ///
    /// ```
    /// use xenon::Tensor1;
    ///
    /// let a = Tensor1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    /// let cum = a.cumprod(0);
    /// assert_eq!(cum, Tensor1::from_vec(vec![1.0, 2.0, 6.0, 24.0]));
    /// ```
    pub fn cumprod(&self, axis: usize) -> Tensor<A, D> {
        let mut result = self.to_owned();
        
        let axis_len = self.shape()[axis];
        
        if axis_len == 0 {
            return result;
        }
        
        for mut lane in result.axis_iter_mut(axis) {
            let mut acc = A::one();
            for elem in lane.iter_mut() {
                acc = acc * *elem;
                *elem = acc;
            }
        }
        
        result
    }
}
```

### 5.3 NaN 传播详细说明

```rust
// cumsum 的 NaN 传播实现
fn cumsum_with_nan_propagation<A: RealScalar>(data: &mut [A]) {
    let mut acc = A::zero();
    let mut has_nan = false;
    
    for elem in data.iter_mut() {
        if has_nan || elem.is_nan() {
            has_nan = true;
            *elem = A::nan();
        } else {
            acc = acc + *elem;
            *elem = acc;
        }
    }
}

// 示例
// 输入: [1.0, 2.0, NaN, 4.0, 5.0]
// 输出: [1.0, 3.0, NaN, NaN, NaN]
```

### 5.4 累积运算语义总结

| 操作 | 返回类型 | 空数组行为 | NaN 行为 | 整数溢出 |
|------|----------|-----------|----------|----------|
| `cumsum` | `Tensor<A, D>` | 同形状空数组 | 传播（后续全 NaN） | panic |
| `cumprod` | `Tensor<A, D>` | 同形状空数组 | 传播（后续全 NaN） | panic |

---

## 6. unique 系列设计

### 6.1 unique 设计

#### 6.1.1 方法签名

```rust
// set_ops.rs

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element,
{
    /// 唯一值
    ///
    /// 返回数组中所有唯一值，按升序排列。
    ///
    /// # 返回值
    ///
    /// `Tensor1<A>` - 排序后的唯一值
    ///
    /// # 空数组行为
    ///
    /// 空数组返回空 `Tensor1`
    ///
    /// # 复杂度
    ///
    /// O(N log N)，其中 N 为元素数量
    ///
    /// # 示例
    ///
    /// ```
    /// use xenon::Tensor1;
    ///
    /// let a = Tensor1::from_vec(vec![3, 1, 2, 1, 3, 2]);
    /// let unique = a.unique();
    /// assert_eq!(unique, Tensor1::from_vec(vec![1, 2, 3]));
    ///
    /// // 空数组
    /// let empty: Tensor1<i32> = Tensor1::zeros([0]);
    /// assert_eq!(empty.unique().len(), 0);
    /// ```
    pub fn unique(&self) -> Tensor1<A>
    where
        A: Ord,
    {
        // 收集所有元素
        let mut values: Vec<A> = self.iter().copied().collect();
        
        // 排序
        values.sort();
        
        // 去重
        values.dedup();
        
        Tensor1::from_vec(values)
    }
}
```

### 6.2 unique_counts 设计

#### 6.2.1 方法签名

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element,
{
    /// 唯一值及计数
    ///
    /// 返回唯一值及每个值的出现次数。
    ///
    /// # 返回值
    ///
    /// `(Tensor1<A>, Tensor1<usize>)` - (values, counts)
    /// - `values`: 排序后的唯一值
    /// - `counts`: 对应的出现次数
    ///
    /// # 约束
    ///
    /// `values.len() == counts.len()`
    ///
    /// # 示例
    ///
    /// ```
    /// use xenon::Tensor1;
    ///
    /// let a = Tensor1::from_vec(vec![1, 2, 1, 3, 2, 1]);
    /// let (values, counts) = a.unique_counts();
    /// 
    /// assert_eq!(values, Tensor1::from_vec(vec![1, 2, 3]));
    /// assert_eq!(counts, Tensor1::from_vec(vec![3, 2, 1]));
    /// ```
    pub fn unique_counts(&self) -> (Tensor1<A>, Tensor1<usize>)
    where
        A: Ord + Copy,
    {
        // 收集并排序
        let mut sorted: Vec<A> = self.iter().copied().collect();
        sorted.sort();
        
        // 计算唯一值和计数
        let mut values = Vec::new();
        let mut counts = Vec::new();
        
        let mut current = None;
        let mut count = 0usize;
        
        for &x in &sorted {
            match current {
                None => {
                    current = Some(x);
                    count = 1;
                }
                Some(c) if c == x => {
                    count += 1;
                }
                Some(_) => {
                    values.push(current.unwrap());
                    counts.push(count);
                    current = Some(x);
                    count = 1;
                }
            }
        }
        
        if count > 0 {
            values.push(current.unwrap());
            counts.push(count);
        }
        
        (Tensor1::from_vec(values), Tensor1::from_vec(counts))
    }
}
```

### 6.3 unique_inverse 设计

#### 6.3.1 方法签名

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element,
{
    /// 唯一值及逆索引
    ///
    /// 返回唯一值及原数组每个元素在唯一值中的索引。
    ///
    /// # 返回值
    ///
    /// `(Tensor1<A>, Tensor1<usize>)` - (values, inverse)
    /// - `values`: 排序后的唯一值
    /// - `inverse`: 原数组每个元素在 values 中的索引
    ///
    /// # 约束
    ///
    /// - `inverse.len() == input.len()`
    /// - `inverse[i] ∈ [0, values.len())`
    /// - `values[inverse[i]] == input[i]`（可重建原数组）
    ///
    /// # 示例
    ///
    /// ```
    /// use xenon::Tensor1;
    ///
    /// let a = Tensor1::from_vec(vec![1, 2, 1, 3, 2, 1]);
    /// let (values, inverse) = a.unique_inverse();
    /// 
    /// assert_eq!(values, Tensor1::from_vec(vec![1, 2, 3]));
    /// // inverse: [0, 1, 0, 2, 1, 0]
    /// // values[0]=1, values[1]=2, values[2]=3
    /// 
    /// // 重建验证
    /// for (i, &inv) in inverse.iter().enumerate() {
    ///     assert_eq!(values[inv], a[i]);
    /// }
    /// ```
    pub fn unique_inverse(&self) -> (Tensor1<A>, Tensor1<usize>)
    where
        A: Ord + Copy + std::hash::Hash,
    {
        use std::collections::HashMap;
        
        // 收集唯一值并排序
        let mut unique_set: Vec<A> = self.iter().copied().collect();
        unique_set.sort();
        unique_set.dedup();
        
        // 构建值到索引的映射
        let mut value_to_idx: HashMap<A, usize> = HashMap::new();
        for (i, &v) in unique_set.iter().enumerate() {
            value_to_idx.insert(v, i);
        }
        
        // 构建逆索引
        let inverse: Vec<usize> = self.iter()
            .map(|&x| value_to_idx[&x])
            .collect();
        
        (Tensor1::from_vec(unique_set), Tensor1::from_vec(inverse))
    }
}
```

---

## 7. bincount 设计

### 7.1 方法签名

```rust
// set_ops.rs

impl<S, D> TensorBase<S, D>
where
    S: Storage<Elem = i32>,  // 支持所有整数类型
    D: Dimension,
{
    /// 非负整数计数
    ///
    /// 统计每个非负整数的出现次数。
    ///
    /// # 参数
    ///
    /// * `minlength` - 输出最小长度（默认 0）
    ///
    /// # 返回值
    ///
    /// `Tensor1<usize>` - 长度为 `max(max(input) + 1, minlength)`
    /// - `output[i] = count of i in input`
    ///
    /// # 空数组行为
    ///
    /// 空数组返回长度为 `minlength` 的全零数组
    ///
    /// # Panic
    ///
    /// * 输入包含负值时 panic
    ///
    /// # 示例
    ///
    /// ```
    /// use xenon::Tensor1;
    ///
    /// let a = Tensor1::from_vec(vec![0, 1, 1, 2, 2, 2]);
    /// let counts = a.bincount(0);
    /// // [1, 2, 3]  // 0 出现 1 次，1 出现 2 次，2 出现 3 次
    ///
    /// // 使用 minlength
    /// let b = Tensor1::from_vec(vec![1, 2]);
    /// let counts = b.bincount(5);
    /// // [0, 1, 1, 0, 0]  // 长度至少为 5
    /// ```
    pub fn bincount(&self, minlength: usize) -> Tensor1<usize> {
        // 检查负值
        for &x in self.iter() {
            if x < 0 {
                panic!("bincount: input contains negative value {}", x);
            }
        }
        
        // 计算输出长度
        let max_val: usize = self.iter()
            .map(|&x| x as usize)
            .max()
            .unwrap_or(0);
        let out_len = max_val.max(minlength) + 1;
        
        // 计数
        let mut counts = vec![0usize; out_len];
        for &x in self.iter() {
            counts[x as usize] += 1;
        }
        
        Tensor1::from_vec(counts)
    }
}
```

### 7.2 带权重的 bincount

```rust
impl<S, D> TensorBase<S, D>
where
    S: Storage<Elem = i32>,
    D: Dimension,
{
    /// 带权重的非负整数计数
    ///
    /// 统计每个非负整数的加权和。
    ///
    /// # 参数
    ///
    /// * `weights` - 权重数组，长度须与输入相同
    /// * `minlength` - 输出最小长度
    ///
    /// # 返回值
    ///
    /// `Tensor1<A>` - `output[i] = sum of weights[j] where input[j] == i`
    ///
    /// # Panic
    ///
    /// * 输入包含负值
    /// * `weights.len() != input.len()`
    ///
    /// # 示例
    ///
    /// ```
    /// use xenon::Tensor1;
    ///
    /// let a = Tensor1::from_vec(vec![0, 1, 1, 2]);
    /// let weights = Tensor1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    /// let weighted = a.bincount_weighted(&weights, 0);
    /// // [1.0, 5.0, 4.0]  // 0: 1.0, 1: 2.0+3.0=5.0, 2: 4.0
    /// ```
    pub fn bincount_weighted<A, W>(
        &self,
        weights: &Tensor1<A>,
        minlength: usize,
    ) -> Tensor1<A>
    where
        A: Numeric,
    {
        if weights.len() != self.len() {
            panic!(
                "bincount: weights length {} != input length {}",
                weights.len(),
                self.len()
            );
        }
        
        // 检查负值
        for &x in self.iter() {
            if x < 0 {
                panic!("bincount: input contains negative value {}", x);
            }
        }
        
        // 计算输出长度
        let max_val: usize = self.iter()
            .map(|&x| x as usize)
            .max()
            .unwrap_or(0);
        let out_len = max_val.max(minlength) + 1;
        
        // 加权计数
        let mut counts = vec![A::zero(); out_len];
        for (&idx, &w) in self.iter().zip(weights.iter()) {
            counts[idx as usize] = counts[idx as usize] + w;
        }
        
        Tensor1::from_vec(counts)
    }
}
```

### 7.3 bincount 语义总结

| 属性 | 行为 |
|------|------|
| 输入类型 | 整数类型（i8/i16/i32/i64/u8/u16/u32/u64） |
| 输入约束 | 所有值须 ≥ 0 |
| 负值输入 | panic |
| 输出长度 | `max(max(input) + 1, minlength)` |
| 空数组 | 返回长度为 `minlength` 的全零数组 |
| 权重参数 | 可选，`output[i] = sum of weights[j] where input[j] == i` |

---

## 8. histogram 设计

### 8.1 方法签名

```rust
// set_ops.rs

impl<S, D> TensorBase<S, D>
where
    S: Storage<Elem = f64>,  // RealScalar
    D: Dimension,
{
    /// 直方图统计
    ///
    /// 统计落入各 bin 的元素数量。
    ///
    /// # 参数
    ///
    /// * `bins` - bin 定义：
    ///   - 整数 `n`: 等宽分割为 n 个 bin
    ///   - `Tensor1<A>`: 自定义 bin 边界（单调递增）
    /// * `range` - 可选范围 `(min, max)`，超出范围的值不计入
    ///
    /// # 返回值
    ///
    /// `Tensor1<usize>` - 各 bin 的计数，长度等于 bin 数
    ///
    /// # Bin 边界规则
    ///
    /// - 左闭右开 `[left, right)`
    /// - 最后一个 bin 闭区间 `[left, right]`
    /// - 例如 bins=[0, 1, 2, 3]:
    ///   - bin 0: [0, 1)
    ///   - bin 1: [1, 2)
    ///   - bin 2: [2, 3]
    ///
    /// # 空数组行为
    ///
    /// 空数组返回全零数组
    ///
    /// # 示例
    ///
    /// ```
    /// use xenon::Tensor1;
    ///
    /// let a = Tensor1::from_vec(vec![0.5, 1.5, 2.5, 1.0, 2.0, 3.0]);
    ///
    /// // 等宽 bin
    /// let hist = a.histogram(3, Some((0.0, 3.0)));
    /// // bins: [0, 1, 2, 3]
    /// // [0.5, 1.0) -> 1 (0.5)
    /// // [1.0, 2.0) -> 2 (1.5, 1.0)
    /// // [2.0, 3.0] -> 3 (2.5, 2.0, 3.0)
    /// // 结果: [1, 2, 3]
    ///
    /// // 自定义 bin 边界
    /// let edges = Tensor1::from_vec(vec![0.0, 1.0, 2.5, 4.0]);
    /// let hist = a.histogram_with_edges(&edges, None);
    /// // [0.0, 1.0) -> 1 (0.5)
    /// // [1.0, 2.5) -> 2 (1.5, 2.0)
    /// // [2.5, 4.0] -> 2 (2.5, 3.0)
    /// // 结果: [1, 2, 2]
    /// ```
    pub fn histogram(&self, bins: usize, range: Option<(A, A)>) -> Tensor1<usize>
    where
        A: RealScalar,
    {
        // 确定范围
        let (min_val, max_val) = range.unwrap_or_else(|| {
            let min = self.min().unwrap_or(A::zero());
            let max = self.max().unwrap_or(A::zero());
            (min, max)
        });
        
        // 生成等宽 bin 边界
        let bin_width = (max_val - min_val) / A::from(bins).unwrap();
        let edges: Vec<A> = (0..=bins)
            .map(|i| min_val + A::from(i).unwrap() * bin_width)
            .collect();
        
        // 计数
        let mut counts = vec![0usize; bins];
        
        for &x in self.iter() {
            // 跳过超出范围的值
            if x < min_val || x > max_val {
                continue;
            }
            
            // 找到所属 bin
            let bin_idx = if x == max_val {
                bins - 1  // 最后一个 bin 闭区间
            } else {
                ((x - min_val) / bin_width).floor().to_usize().unwrap_or(0)
            };
            
            if bin_idx < bins {
                counts[bin_idx] += 1;
            }
        }
        
        Tensor1::from_vec(counts)
    }
    
    /// 使用自定义 bin 边界的直方图
    pub fn histogram_with_edges(&self, edges: &Tensor1<A>, range: Option<(A, A)>) -> Tensor1<usize>
    where
        A: RealScalar,
    {
        let n_bins = edges.len() - 1;
        let mut counts = vec![0usize; n_bins];
        
        for &x in self.iter() {
            // 检查范围
            if let Some((min_val, max_val)) = range {
                if x < min_val || x > max_val {
                    continue;
                }
            }
            
            // 二分查找确定 bin
            let bin_idx = Self::find_bin(x, edges);
            if bin_idx < n_bins {
                counts[bin_idx] += 1;
            }
        }
        
        Tensor1::from_vec(counts)
    }
    
    /// 二分查找确定元素所属 bin
    fn find_bin(x: A, edges: &Tensor1<A>) -> usize
    where
        A: RealScalar,
    {
        let n = edges.len();
        
        // 边界情况
        if x < edges[0] || x > edges[n - 1] {
            return n;  // 超出范围
        }
        if x == edges[n - 1] {
            return n - 2;  // 最后一个 bin 闭区间
        }
        
        // 二分查找
        let mut left = 0;
        let mut right = n - 1;
        
        while left < right - 1 {
            let mid = (left + right) / 2;
            if x < edges[mid] {
                right = mid;
            } else {
                left = mid;
            }
        }
        
        left
    }
}
```

### 8.2 histogram_bin_edges 设计

```rust
impl<S, D> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: RealScalar,
{
    /// 计算 bin 边界
    ///
    /// 返回等宽 bin 的边界值。
    ///
    /// # 参数
    ///
    /// * `bins` - bin 数量
    /// * `range` - 可选范围 `(min, max)`
    ///
    /// # 返回值
    ///
    /// `Tensor1<A>` - 长度为 `bins + 1` 的边界值
    ///
    /// # 示例
    ///
    /// ```
    /// use xenon::Tensor1;
    ///
    /// let a = Tensor1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    /// let edges = a.histogram_bin_edges(4, Some((0.0, 4.0)));
    /// // [0.0, 1.0, 2.0, 3.0, 4.0]
    /// ```
    pub fn histogram_bin_edges(&self, bins: usize, range: Option<(A, A)>) -> Tensor1<A> {
        let (min_val, max_val) = range.unwrap_or_else(|| {
            let min = self.min().unwrap_or(A::zero());
            let max = self.max().unwrap_or(A::zero());
            (min, max)
        });
        
        let bin_width = (max_val - min_val) / A::from(bins).unwrap();
        let edges: Vec<A> = (0..=bins)
            .map(|i| min_val + A::from(i).unwrap() * bin_width)
            .collect();
        
        Tensor1::from_vec(edges)
    }
}
```

### 8.3 Bin 边界规则详解

| Bin | 范围 | 说明 |
|-----|------|------|
| 第 1 个 | `[edges[0], edges[1])` | 左闭右开 |
| 第 i 个 | `[edges[i], edges[i+1])` | 左闭右开 |
| 最后一个 | `[edges[n-2], edges[n-1]]` | **闭区间** |

**示例**: `edges = [0, 10, 20, 30]`

| 值 | 所属 Bin | 原因 |
|----|----------|------|
| 0 | 0 | `[0, 10)` |
| 5 | 0 | `[0, 10)` |
| 10 | 1 | `[10, 20)` |
| 19.9 | 1 | `[10, 20)` |
| 20 | 2 | `[20, 30]` (最后一个 bin 闭区间) |
| 30 | 2 | `[20, 30]` (闭区间) |

---

## 9. NaN/Inf 处理

### 9.1 各归约操作的 NaN 行为

| 操作 | NaN 输入行为 | 说明 |
|------|-------------|------|
| `sum` | 返回 NaN | 任何元素为 NaN 则结果为 NaN |
| `prod` | 返回 NaN | 任何元素为 NaN 则结果为 NaN |
| `mean` | 返回 NaN | 依赖 sum，NaN 传播 |
| `var` | 返回 NaN | 依赖 mean，NaN 传播 |
| `std` | 返回 NaN | 依赖 var，NaN 传播 |
| `min` | 返回 NaN | NaN 传播语义 |
| `max` | 返回 NaN | NaN 传播语义 |
| `argmin` | 返回 NaN 索引 | 返回第一个 NaN 的索引 |
| `argmax` | 返回 NaN 索引 | 返回第一个 NaN 的索引 |
| `all` | N/A | bool 类型无 NaN |
| `any` | N/A | bool 类型无 NaN |
| `cumsum` | 传播 NaN | 遇 NaN 后所有值均为 NaN |
| `cumprod` | 传播 NaN | 遇 NaN 后所有值均为 NaN |
| `bincount` | N/A | 整数输入无 NaN |
| `histogram` | 跳过或计入边缘 | 取决于实现 |

### 9.2 NaN 检测与处理

```rust
// NaN 传播辅助函数
fn propagate_nan<A: RealScalar>(a: A, b: A) -> A {
    if a.is_nan() || b.is_nan() {
        A::nan()
    } else {
        // 正常计算
    }
}

// argmin/argmax 的 NaN 处理
fn argmin_with_nan<A: RealScalar>(data: &[A]) -> usize {
    let mut min_idx = 0;
    let mut min_val = data[0];
    
    for (i, &x) in data.iter().enumerate() {
        // NaN 被视为最小（返回第一个 NaN 索引）
        if x.is_nan() {
            return i;
        }
        if x < min_val {
            min_val = x;
            min_idx = i;
        }
    }
    min_idx
}
```

### 9.3 Inf 处理

| 操作 | +Inf 输入 | -Inf 输入 |
|------|----------|----------|
| `sum` | +Inf | -Inf |
| `prod` | 取决于其他元素 | 取决于其他元素 |
| `min` | 取决于其他元素 | -Inf 为最小 |
| `max` | +Inf 为最大 | 取决于其他元素 |

---

## 10. 整数溢出处理

### 10.1 Checked Arithmetic 策略

```rust
// 整数类型的 checked 归约
trait CheckedReduce<A> {
    fn checked_sum(iter: impl Iterator<Item = A>) -> A;
    fn checked_prod(iter: impl Iterator<Item = A>) -> A;
}

impl CheckedReduce<i32> for i32 {
    fn checked_sum(iter: impl Iterator<Item = i32>) -> i32 {
        iter.fold(0i32, |acc, x| {
            acc.checked_add(x).expect("integer overflow in sum")
        })
    }
    
    fn checked_prod(iter: impl Iterator<Item = i32>) -> i32 {
        iter.fold(1i32, |acc, x| {
            acc.checked_mul(x).expect("integer overflow in prod")
        })
    }
}

// 为所有整数类型实现
impl CheckedReduce<i8> for i8 { /* ... */ }
impl CheckedReduce<i16> for i16 { /* ... */ }
impl CheckedReduce<i64> for i64 { /* ... */ }
impl CheckedReduce<u8> for u8 { /* ... */ }
impl CheckedReduce<u16> for u16 { /* ... */ }
impl CheckedReduce<u32> for u32 { /* ... */ }
impl CheckedReduce<u64> for u64 { /* ... */ }
```

### 10.2 溢出处理总结

| 类型 | 操作 | 溢出行为 |
|------|------|----------|
| 整数 | `sum` | panic |
| 整数 | `prod` | panic |
| 整数 | `cumsum` | panic |
| 整数 | `cumprod` | panic |
| 浮点 | `sum` | Inf |
| 浮点 | `prod` | Inf |
| 浮点 | `cumsum` | Inf |
| 浮点 | `cumprod` | Inf |

### 10.3 Panic 消息示例

```
thread 'main' panicked at 'integer overflow in sum: 2147483647 + 1', src/ops/reduction.rs:42:5
```

---

## 11. SIMD 加速

### 11.1 SIMD 适用操作

| 操作 | SIMD 支持 | 条件 |
|------|----------|------|
| `sum` | ✓ | 连续内存 + `simd` feature |
| `prod` | ✓ | 连续内存 + `simd` feature |
| `min` | ✓ | 连续内存 + `simd` feature |
| `max` | ✓ | 连续内存 + `simd` feature |
| `mean` | ✓ | 依赖 SIMD sum |
| `var/std` | 部分 | 依赖 SIMD sum 和平方 |
| `argmin/argmax` | 部分 | 需要 SIMD min/max + 索引追踪 |
| `all/any` | ✓ | SIMD 位运算 |

### 11.2 SIMD 归约策略

```rust
#[cfg(feature = "simd")]
mod simd {
    use pulp::{Arch, Simd};
    
    /// SIMD 向量和
    pub fn sum_f32(data: &[f32]) -> f32 {
        let arch = Arch::new();
        let n = data.len();
        
        if n < 8 {
            return data.iter().sum();
        }
        
        arch.dispatch(|| {
            let chunks = n / 8;
            let mut acc = pulp::f32x8::splat(0.0);
            
            for i in 0..chunks {
                let v = pulp::f32x8::from_slice_unaligned(&data[i * 8..]);
                acc = acc + v;
            }
            
            // 水平求和
            let mut sum = acc.reduce_add();
            
            // 尾部标量处理
            for i in (chunks * 8)..n {
                sum += data[i];
            }
            
            sum
        })
    }
    
    /// SIMD 向量最小值
    pub fn min_f32(data: &[f32]) -> Option<f32> {
        if data.is_empty() {
            return None;
        }
        
        let arch = Arch::new();
        let n = data.len();
        let mut min_val = data[0];
        
        if n < 8 {
            for &x in data {
                if x < min_val {
                    min_val = x;
                }
            }
            return Some(min_val);
        }
        
        arch.dispatch(|| {
            let chunks = n / 8;
            let mut acc = pulp::f32x8::splat(f32::INFINITY);
            
            for i in 0..chunks {
                let v = pulp::f32x8::from_slice_unaligned(&data[i * 8..]);
                acc = acc.min(v);
            }
            
            // 水平最小值
            min_val = acc.reduce_min();
            
            // 尾部标量处理
            for i in (chunks * 8)..n {
                if data[i] < min_val {
                    min_val = data[i];
                }
            }
        });
        
        Some(min_val)
    }
}
```

### 11.3 SIMD 路径选择

```
                    ┌─────────────────┐
                    │   归约请求       │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ simd feature 启用?│
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │ No                          │ Yes
              ▼                             ▼
     ┌────────────────┐           ┌────────────────┐
     │ 标量路径        │           │ 连续 + 对齐?    │
     └────────────────┘           └────────┬───────┘
                                           │
                            ┌──────────────┴──────────────┐
                            │ No                          │ Yes
                            ▼                             ▼
                   ┌────────────────┐           ┌────────────────┐
                   │ 非对齐 SIMD     │           │ 对齐 SIMD       │
                   └────────────────┘           └────────────────┘
```

---

## 12. 与其他模块的交互

### 12.1 模块依赖图

```
┌─────────────────────────────────────────────────────────────┐
│                    ops/reduction.rs                          │
│         sum, prod, mean, var, std, min, max, argmin, argmax │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│    tensor     │   │   element     │   │    error      │
│ TensorBase    │   │ Numeric       │   │ EmptyArray    │
│ shape/strides │   │ RealScalar    │   │ InvalidAxis   │
└───────────────┘   └───────────────┘   └───────────────┘
        │                   │
        └───────────────────┼───────────────────┐
                            │                   │
                            ▼                   ▼
                    ┌───────────────┐   ┌───────────────┐
                    │    layout     │   │     simd      │
                    │ is_contiguous │   │ pulp Arch     │
                    │ is_aligned    │   │ dispatch      │
                    └───────────────┘   └───────────────┘
```

### 12.2 与 tensor 模块的接口

```rust
// 依赖的 tensor 方法
// - shape(): &[usize]
// - len(): usize
// - iter(): Elements<A>
// - iter_mut(): ElementsMut<A>
// - axis_iter(axis): AxisIter
// - axis_iter_mut(axis): AxisIterMut
// - is_contiguous(): bool
// - is_aligned(): bool
```

### 12.3 与 element 模块的接口

```rust
// 依赖的 element trait
// - Element: Copy, Clone, PartialEq, Debug
// - Numeric: Element + Add + Sub + Mul + Div + Neg
// - RealScalar: Numeric + 数学函数 + NaN/Inf 检测
```

### 12.4 与 error 模块的接口

```rust
// 错误类型
pub struct EmptyArray;
pub struct InvalidAxis {
    pub axis: usize,
    pub ndim: usize,
}

// 错误显示
impl Display for EmptyArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cannot compute min/max/argmin/argmax on empty array")
    }
}

impl Display for InvalidAxis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "axis {} is invalid for {}-dimensional array", self.axis, self.ndim)
    }
}
```

---

## 13. 实现任务分解

### 任务清单

| # | 任务 | 文件 | 预估时间 | 依赖 |
|---|------|------|----------|------|
| 1 | 创建 `reduction.rs`，实现全局 `sum`/`prod` | `reduction.rs` | 10 min | tensor, element |
| 2 | 实现全局 `mean`/`var`/`std`（含 ddof） | `reduction.rs` | 10 min | #1 |
| 3 | 实现全局 `min`/`max`（含 EmptyArray 错误） | `reduction.rs` | 10 min | #1, error |
| 4 | 实现全局 `argmin`/`argmax` | `reduction.rs` | 10 min | #3 |
| 5 | 实现全局 `all`/`any`（布尔归约） | `reduction.rs` | 10 min | #1 |
| 6 | 实现沿轴归约框架（`sum_axis`/`prod_axis`） | `reduction.rs` | 15 min | #1 |
| 7 | 实现沿轴 `min_axis`/`max_axis`/`argmin_axis`/`argmax_axis` | `reduction.rs` | 10 min | #3, #4, #6 |
| 8 | 创建 `accumulate.rs`，实现 `cumsum`/`cumprod` | `accumulate.rs` | 10 min | tensor |
| 9 | 创建 `set_ops.rs`，实现 `unique` | `set_ops.rs` | 10 min | tensor |
| 10 | 实现 `unique_counts`/`unique_inverse` | `set_ops.rs` | 10 min | #9 |
| 11 | 实现 `bincount`（含 minlength、weights、负值检查） | `set_ops.rs` | 10 min | #9 |
| 12 | 实现 `histogram`（等宽 bin） | `set_ops.rs` | 10 min | #9 |
| 13 | 实现 `histogram_bin_edges` 和自定义边界 histogram | `set_ops.rs` | 10 min | #12 |
| 14 | 实现 SIMD 加速（sum/prod/min/max） | `reduction.rs` | 15 min | #1-#5, simd |
| 15 | 编写单元测试和文档测试 | `tests/` | 15 min | #1-#14 |

**总预估时间**: 约 165 分钟（2.75 小时）

### 任务依赖图

```
#1 (sum/prod) ──→ #2 (mean/var/std)
      │
      ├──→ #3 (min/max) ──→ #4 (argmin/argmax)
      │                              │
      └──→ #5 (all/any)              │
             │                       │
             └──→ #6 (axis framework) ┴─→ #7 (axis min/max/arg*)
                        │
                        └──→ #8 (cumsum/cumprod)

#9 (unique) ──→ #10 (unique_counts/inverse) ──→ #11 (bincount)
      │
      └──→ #12 (histogram) ──→ #13 (histogram_bin_edges)

#1-#13 ──→ #14 (SIMD) ──→ #15 (tests)
```

---

## 14. 设计决策记录

### 14.1 为什么 var/std 默认 ddof=0？

**决策**: `var` 和 `std` 默认使用 `ddof=0`（有偏估计）。

**理由**:
1. **NumPy 兼容**: `np.var()` 默认 `ddof=0`
2. **数学定义**: 总体方差除以 N 是标准定义
3. **避免混淆**: 显式参数让用户明确使用哪种估计

**替代方案**: 默认 `ddof=1`。
**拒绝原因**: 与 NumPy/PyTorch 行为不一致。

---

### 14.2 为什么整数 sum/prod 溢出时 panic？

**决策**: 整数 `sum` 和 `prod` 在溢出时 panic（debug 和 release 模式均如此）。

**理由**:
1. **Rust 哲学**: 溢出是编程错误，应显式处理
2. **正确性优先**: 静默 wrap-around 可能导致严重 bug
3. **与 Rust 默认一致**: debug 模式下整数溢出 panic

**替代方案**: 使用 wrapping 或 saturating 算术。
**拒绝原因**: 与需求文档明确要求的 "溢出将 panic" 不一致。

---

### 14.3 为什么 min/max 对空数组返回 Result？

**决策**: `min` 和 `max` 返回 `Result<A, EmptyArray>` 而非 panic。

**理由**:
1. **可恢复错误**: 空数组是运行时条件，非编程错误
2. **与其他库一致**: ndarray 返回 `Option`
3. **用户友好**: 允许优雅处理空输入

**替代方案**: 空数组返回 `A::max_value()` 或 panic。
**拒绝原因**: 隐藏错误，不符合 Rust 错误处理哲学。

---

### 14.4 为什么 cumsum/cumprod 遇 NaN 后全为 NaN？

**决策**: 累积运算遇到 NaN 后，后续所有值均为 NaN。

**理由**:
1. **NaN 传播**: 保持 IEEE 754 NaN 传播语义
2. **不可恢复**: 一旦累积链中出现 NaN，后续结果无意义
3. **与 NumPy 一致**: `np.cumsum([1, np.nan, 3])` 产生 `[1, nan, nan]`

---

### 14.5 为什么 histogram 最后一个 bin 是闭区间？

**决策**: 最后一个 bin 使用闭区间 `[left, right]`，其他 bin 左闭右开 `[left, right)`。

**理由**:
1. **NumPy 兼容**: `np.histogram` 使用相同规则
2. **覆盖最大值**: 确保最大值被计入
3. **避免遗漏**: 否则 `max_val` 可能不被任何 bin 包含

---

### 14.6 为什么 argmin/argmax 遇 NaN 返回 NaN 索引？

**决策**: 当数组包含 NaN 时，`argmin`/`argmax` 返回第一个 NaN 的索引。

**理由**:
1. **NaN 传播语义**: 与 `min`/`max` 的 NaN 传播一致
2. **可预测性**: 用户可以通过检查 `arr[idx].is_nan()` 判断
3. **与 NumPy 一致**: `np.argmin([1, np.nan, 3])` 返回 1

---

### 14.7 为什么 bincount 不支持负值？

**决策**: `bincount` 对负值输入 panic。

**理由**:
1. **语义约束**: bincount 统计非负整数的出现次数
2. **数组索引**: 负值无法作为数组索引
3. **与 NumPy 一致**: `np.bincount([-1])` 抛出 ValueError

**替代方案**: 忽略负值。
**拒绝原因**: 隐藏错误，可能导致难以发现的 bug。

---

### 14.8 为什么 mean 仅支持 RealScalar？

**决策**: `mean` 仅支持 `f32` 和 `f64`，不支持整数或复数。

**理由**:
1. **整数除法截断**: `mean([1, 2])` 整数除法结果为 1 而非 1.5
2. **精度问题**: 整数均值无意义
3. **复数均值**: 需求文档未提及

**替代方案**: 整数 mean 返回浮点。
**拒绝原因**: 隐式类型转换与 Rust 哲学冲突。

---

### 14.9 为什么 all 对空数组返回 true？

**决策**: `all()` 对空数组返回 `true`（空真）。

**理由**:
1. **逻辑定义**: ∀x ∈ ∅, P(x) 为真（因为没有反例）
2. **与标准库一致**: `iter().all()` 对空迭代器返回 `true`
3. **数学一致性**: 空集的全称量词为真

---

### 14.10 为什么 any 对空数组返回 false？

**决策**: `any()` 对空数组返回 `false`（空假）。

**理由**:
1. **逻辑定义**: ∃x ∈ ∅, P(x) 为假（因为没有元素）
2. **与标准库一致**: `iter().any()` 对空迭代器返回 `false`
3. **数学一致性**: 空集的存在量词为假

---

## 附录 A: API 速查表

### A.1 全局归约

| 方法 | 签名 | 返回类型 |
|------|------|----------|
| `sum` | `(&self) -> A` | `A` |
| `prod` | `(&self) -> A` | `A` |
| `mean` | `(&self) -> A` | `A` (RealScalar) |
| `var` | `(&self, ddof: usize) -> A` | `A` (RealScalar) |
| `std` | `(&self, ddof: usize) -> A` | `A` (RealScalar) |
| `min` | `(&self) -> Result<A, EmptyArray>` | `Result<A, EmptyArray>` |
| `max` | `(&self) -> Result<A, EmptyArray>` | `Result<A, EmptyArray>` |
| `argmin` | `(&self) -> Result<usize, EmptyArray>` | `Result<usize, EmptyArray>` |
| `argmax` | `(&self) -> Result<usize, EmptyArray>` | `Result<usize, EmptyArray>` |
| `all` | `(&self) -> bool` | `bool` |
| `any` | `(&self) -> bool` | `bool` |

### A.2 沿轴归约

| 方法 | 签名 | 返回类型 |
|------|------|----------|
| `sum_axis` | `(&self, axis: usize) -> Result<Tensor<A, D::Smaller>, InvalidAxis>` | 降维 Tensor |
| `prod_axis` | `(&self, axis: usize) -> Result<Tensor<A, D::Smaller>, InvalidAxis>` | 降维 Tensor |
| `mean_axis` | `(&self, axis: usize) -> Result<Tensor<A, D::Smaller>, InvalidAxis>` | 降维 Tensor |
| `var_axis` | `(&self, axis: usize, ddof: usize) -> Result<Tensor<A, D::Smaller>, InvalidAxis>` | 降维 Tensor |
| `std_axis` | `(&self, axis: usize, ddof: usize) -> Result<Tensor<A, D::Smaller>, InvalidAxis>` | 降维 Tensor |
| `min_axis` | `(&self, axis: usize) -> Result<Tensor<A, D::Smaller>, InvalidAxis>` | 降维 Tensor |
| `max_axis` | `(&self, axis: usize) -> Result<Tensor<A, D::Smaller>, InvalidAxis>` | 降维 Tensor |
| `argmin_axis` | `(&self, axis: usize) -> Result<Tensor<usize, D::Smaller>, InvalidAxis>` | 降维 Tensor |
| `argmax_axis` | `(&self, axis: usize) -> Result<Tensor<usize, D::Smaller>, InvalidAxis>` | 降维 Tensor |
| `all_axis` | `(&self, axis: usize) -> Result<Tensor<bool, D::Smaller>, InvalidAxis>` | 降维 Tensor |
| `any_axis` | `(&self, axis: usize) -> Result<Tensor<bool, D::Smaller>, InvalidAxis>` | 降维 Tensor |

### A.3 累积运算

| 方法 | 签名 | 返回类型 |
|------|------|----------|
| `cumsum` | `(&self, axis: usize) -> Tensor<A, D>` | 同形状 Tensor |
| `cumprod` | `(&self, axis: usize) -> Tensor<A, D>` | 同形状 Tensor |

### A.4 集合操作

| 方法 | 签名 | 返回类型 |
|------|------|----------|
| `unique` | `(&self) -> Tensor1<A>` | 唯一值 |
| `unique_counts` | `(&self) -> (Tensor1<A>, Tensor1<usize>)` | (值, 计数) |
| `unique_inverse` | `(&self) -> (Tensor1<A>, Tensor1<usize>)` | (值, 逆索引) |

### A.5 计数操作

| 方法 | 签名 | 返回类型 |
|------|------|----------|
| `bincount` | `(&self, minlength: usize) -> Tensor1<usize>` | 计数 |
| `bincount_weighted` | `(&self, weights: &Tensor1<A>, minlength: usize) -> Tensor1<A>` | 加权计数 |
| `histogram` | `(&self, bins: usize, range: Option<(A, A)>) -> Tensor1<usize>` | 直方图 |
| `histogram_with_edges` | `(&self, edges: &Tensor1<A>, range: Option<(A, A)>) -> Tensor1<usize>` | 自定义 bin |
| `histogram_bin_edges` | `(&self, bins: usize, range: Option<(A, A)>) -> Tensor1<A>` | bin 边界 |

---

## 附录 B: 错误类型

| 错误 | 触发场景 | 处理方式 |
|------|----------|----------|
| `EmptyArray` | 对空数组调用 min/max/argmin/argmax | `Result` |
| `InvalidAxis` | 沿轴归约时轴索引超出范围 | `Result` |
| panic | 整数 sum/prod 溢出 | panic |
| panic | bincount 输入含负值 | panic |

---

## 附录 C: 性能建议

1. **优先使用连续数组**: 非连续数组回退到标量路径
2. **对齐内存**: 使用 Xenon 默认的 64 字节对齐以启用 SIMD
3. **沿轴归约选择**: 对于 F-order 数组，沿轴 0 归约更高效
4. **大数组并行**: 元素数 ≥ 64K 时启用并行（需 `parallel` feature）
5. **预分配输出**: 批量操作时预分配输出数组

---

*本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。*
