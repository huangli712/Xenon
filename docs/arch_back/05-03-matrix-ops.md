# 矩阵运算模块设计文档

> **模块路径**: `src/ops/matrix.rs`
> **版本**: v18
> **日期**: 2026-03-28
> **前置文档**: 02-project-architecture.md, 03-06-tensor-core.md, 03-05-memory-layout.md

---

## 1. 模块概述

### 1.1 定位

矩阵运算模块 (`src/ops/matrix.rs`) 提供 Senon 张量库的基础线性代数运算。本模块专注于**矩阵-向量乘法**和**向量运算**，不包含完整的矩阵-矩阵乘法（GEMM）。

### 1.2 支持的操作

| 操作 | 描述 | 输入形状 | 输出形状 |
|------|------|----------|----------|
| `matvec` | 矩阵-向量乘法 | `(M, N) × (N,)` | `(M,)` |
| `dot` / `inner` | 向量内积 | `(N,) × (N,)` | 标量 |
| `outer` | 向量外积 | `(M,) × (N,)` | `(M, N)` |
| `batch_matvec` | 批量矩阵-向量乘法 | `(..., M, N) × (..., N)` | `(..., M)` |
| `batch_dot` | 批量向量内积 | `(..., N) × (..., N)` | `(...,)` |
| `batch_add` | 批量逐元素加法 | `(..., M, N) × (..., M, N)` | `(..., M, N)` |
| `batch_scale` | 批量标量缩放 | `(..., M, N) × scalar` | `(..., M, N)` |

### 1.3 范围外操作

| 操作 | 原因 |
|------|------|
| **GEMM（矩阵-矩阵乘法）** | 复杂度高，应由上游库通过 FFI 调用优化 BLAS 实现 |
| 矩阵分解（LU, QR, SVD） | 超出基础张量库范围 |
| 特征值计算 | 超出基础张量库范围 |
| 线性方程求解 | 超出基础张量库范围 |

### 1.4 设计目标

| 目标 | 说明 |
|------|------|
| **SIMD 友好** | matvec 和 dot 使用向量化实现 |
| **BLAS 兼容** | 内存布局支持 BLAS 调用约定 |
| **批量高效** | batch 操作利用连续内存优化 |
| **类型安全** | 泛型约束确保编译期维度正确性 |

---

## 2. 文件结构

```
src/ops/
└── matrix.rs          # 矩阵运算（本模块）
```

### 2.1 模块内部组织

```rust
// matrix.rs 内部结构

// 公开 API
pub fn matvec<A, S1, S2>(matrix: &TensorBase<S1, Ix2>, vec: &TensorBase<S2, Ix1>) -> Tensor1<A>
where ...;

pub fn dot<A, S1, S2>(a: &TensorBase<S1, Ix1>, b: &TensorBase<S2, Ix1>) -> A
where ...;

pub fn outer<A, S1, S2>(a: &TensorBase<S1, Ix1>, b: &TensorBase<S2, Ix1>) -> Tensor2<A>
where ...;

pub fn batch_matvec<A, S1, S2, D>(matrix: &TensorBase<S1, D>, vec: &TensorBase<S2, D>) -> Tensor<A, D>
where ...;

// 内部实现
mod scalar {
    // 标量回退实现
}

#[cfg(feature = "simd")]
mod simd {
    // SIMD 优化实现
}
```

---

## 3. matvec 设计

### 3.1 函数签名

```rust
/// 矩阵-向量乘法：y = A × x
///
/// # 参数
///
/// * `matrix` - 形状为 `(M, N)` 的矩阵
/// * `vec` - 形状为 `(N,)` 的向量
///
/// # 返回
///
/// 形状为 `(M,)` 的结果向量
///
/// # 约束
///
/// - 矩阵列数必须等于向量长度
/// - 元素类型必须实现 `Numeric`
///
/// # 示例
///
/// ```
/// use Senon::{Tensor2, Tensor1, matvec};
///
/// let a = Tensor2::<f64>::zeros([3, 4]);
/// let x = Tensor1::<f64>::zeros([4]);
/// let y = matvec(&a, &x);  // 形状 [3]
/// ```
pub fn matvec<A, S1, S2>(
    matrix: &TensorBase<S1, Ix2>,
    vec: &TensorBase<S2, Ix1>,
) -> Tensor1<A>
where
    A: Numeric + Copy,
    S1: Storage<Elem = A>,
    S2: Storage<Elem = A>,
{
    // 维度检查
    let (m, n) = (matrix.shape()[0], matrix.shape()[1]);
    assert_eq!(vec.shape()[0], n, "Matrix columns must equal vector length");
    
    // 选择实现路径
    #[cfg(feature = "simd")]
    {
        if matrix.is_contiguous() && vec.is_contiguous() {
            return simd::matvec_impl(matrix, vec);
        }
    }
    
    scalar::matvec_impl(matrix, vec)
}
```

### 3.2 维度约束

| 约束 | 检查 | 错误 |
|------|------|------|
| 矩阵维度 | `matrix.ndim() == 2` | panic |
| 向量维度 | `vec.ndim() == 1` | panic |
| 形状兼容 | `matrix.shape()[1] == vec.shape()[0]` | panic |

### 3.3 F-order 和 C-order 处理

**核心差异**：内存布局影响遍历模式和 SIMD 效率。

#### F-contiguous（列优先）

```
矩阵形状: (M, N) = (3, 4)
内存布局: 列优先存储
步长: [1, M]

访问模式 for matvec:
- 输出 y[i] = sum(A[i,j] * x[j]) for j in 0..N
- 内层循环沿 j（列方向）遍历
- F-order 下 A[i,j] 步长为 M（跨列跳跃）

优化策略:
- 转置访问或使用 BLAS dgemv with trans='T'
```

#### C-contiguous（行优先）

```
矩阵形状: (M, N) = (3, 4)
内存布局: 行优先存储
步长: [N, 1]

访问模式 for matvec:
- 输出 y[i] = sum(A[i,j] * x[j]) for j in 0..N
- 内层循环沿 j 遍历
- C-order 下 A[i,j] 步长为 1（连续）

优化策略:
- 直接 SIMD 向量化内层循环
- 每行与向量 x 做点积
```

#### 处理策略伪代码

```
function matvec_dispatch(matrix, vec):
    m = matrix.shape[0]
    n = matrix.shape[1]
    
    // 分配输出
    y = zeros([m])
    
    if matrix.is_f_contiguous():
        // F-order: 使用列优先遍历
        // y[i] = sum_j A[i,j] * x[j]
        // A 在内存中按列存储，跨行访问 stride=1
        for j in 0..n:
            x_j = vec[j]
            for i in 0..m:
                y[i] += matrix[[i, j]] * x_j
        // 这种模式对缓存更友好（按列顺序访问矩阵）
        
    else if matrix.is_c_contiguous():
        // C-order: 使用行优先遍历（适合 SIMD）
        for i in 0..m:
            row_i = matrix.row(i)  // 连续内存
            y[i] = dot_simd(row_i, vec)  // SIMD 点积
            
    else:
        // 非连续: 使用通用标量实现
        for i in 0..m:
            for j in 0..n:
                y[i] += matrix[[i, j]] * vec[j]
    
    return y
```

### 3.4 SIMD 优化策略

#### 条件判断

```rust
fn can_use_simd<A, S1, S2>(matrix: &TensorBase<S1, Ix2>, vec: &TensorBase<S2, Ix1>) -> bool
where
    S1: Storage<Elem = A>,
    S2: Storage<Elem = A>,
{
    // 条件 1: 连续内存
    if !matrix.is_contiguous() || !vec.is_contiguous() {
        return false;
    }
    
    // 条件 2: 对齐（可选，非对齐也可用非对齐加载）
    let aligned = matrix.is_aligned() && vec.is_aligned();
    
    // 条件 3: 足够长的向量（避免 SIMD 开销）
    let n = matrix.shape()[1];
    let min_len = std::mem::size_of::<A>() * 8 / 256; // AVX 宽度
    
    n >= min_len
}
```

#### C-order SIMD 实现

```rust
#[cfg(feature = "simd")]
fn matvec_simd_c_order<A>(
    matrix: &Tensor2<A>,
    vec: &Tensor1<A>,
    out: &mut [A],
) where
    A: RealScalar,
{
    use pulp::Simd;
    
    let arch = pulp::Arch::new();
    let (m, n) = (matrix.shape()[0], matrix.shape()[1]);
    let vec_ptr = vec.as_ptr();
    
    arch.dispatch(|| {
        for i in 0..m {
            let row_ptr = unsafe { matrix.as_ptr().add(i * n) };
            out[i] = simd_dot(row_ptr, vec_ptr, n);
        }
    });
}

#[cfg(feature = "simd")]
fn simd_dot<A: RealScalar>(a: *const A, b: *const A, n: usize) -> A {
    // 使用 pulp 的 SIMD 抽象
    // AVX-512: 16 x f32 或 8 x f64
    // AVX2: 8 x f32 或 4 x f64
    // ...
}
```

---

## 4. dot/inner 设计

### 4.1 函数签名

```rust
/// 向量内积：result = a · b = sum(a[i] * b[i])
///
/// # 参数
///
/// * `a` - 形状为 `(N,)` 的向量
/// * `b` - 形状为 `(N,)` 的向量
///
/// # 返回
///
/// 标量值（类型 `A`）
///
/// # 约束
///
/// - 两个向量长度必须相等
/// - 元素类型必须实现 `Numeric`
///
/// # 数值精度
///
/// 对于浮点类型，当 N 较大时可能产生精度损失。
/// 当前实现使用直接求和，未来可考虑 Kahan 补偿求和。
///
/// # 示例
///
/// ```
/// use Senon::{Tensor1, dot};
///
/// let a = Tensor1::from_vec(vec![1.0, 2.0, 3.0]);
/// let b = Tensor1::from_vec(vec![4.0, 5.0, 6.0]);
/// let result = dot(&a, &b);  // 1*4 + 2*5 + 3*6 = 32.0
/// ```
pub fn dot<A, S1, S2>(
    a: &TensorBase<S1, Ix1>,
    b: &TensorBase<S2, Ix1>,
) -> A
where
    A: Numeric + Copy,
    S1: Storage<Elem = A>,
    S2: Storage<Elem = A>,
{
    assert_eq!(a.shape()[0], b.shape()[0], "Vector lengths must be equal");
    
    #[cfg(feature = "simd")]
    {
        if a.is_contiguous() && b.is_contiguous() {
            return simd::dot_impl(a, b);
        }
    }
    
    scalar::dot_impl(a, b)
}

/// `inner` 是 `dot` 的别名
pub use dot as inner;
```

### 4.2 维度约束

| 约束 | 检查 | 错误 |
|------|------|------|
| a 维度 | `a.ndim() == 1` | panic |
| b 维度 | `b.ndim() == 1` | panic |
| 长度相等 | `a.shape()[0] == b.shape()[0]` | panic |

### 4.3 数值精度考虑

#### 直接求和（当前实现）

```
result = sum(a[i] * b[i]) for i in 0..N
```

**问题**：当 N 很大时，累加可能导致精度损失。

#### Kahan 补偿求和（未来可选）

```
function kahan_dot(a, b):
    sum = 0
    c = 0  // 补偿项
    
    for i in 0..N:
        product = a[i] * b[i]
        y = product - c
        t = sum + y
        c = (t - sum) - y
        sum = t
    
    return sum
```

**决策**：当前版本使用直接求和，原因：
1. SIMD 实现更简单
2. 大多数场景精度足够
3. 可通过 `batch_dot` 拆分大向量

### 4.4 SIMD 实现

```rust
#[cfg(feature = "simd")]
mod simd {
    use pulp::{Simd, f32x16, f64x8};
    
    pub fn dot_impl<A>(a: &Tensor1<A>, b: &Tensor1<A>) -> A
    where
        A: RealScalar,
    {
        let n = a.shape()[0];
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        
        let arch = pulp::Arch::new();
        
        arch.dispatch(|| {
            // SIMD 循环
            let mut sum = A::zero();
            let simd_width = <A::Simd>::LANES;
            let chunks = n / simd_width;
            let remainder = n % simd_width;
            
            // 向量化累加
            let mut acc = <A::Simd>::splat(A::zero());
            for i in 0..chunks {
                let va = <A::Simd>::from_slice_unaligned(unsafe {
                    std::slice::from_raw_parts(a_ptr.add(i * simd_width), simd_width)
                });
                let vb = <A::Simd>::from_slice_unaligned(unsafe {
                    std::slice::from_raw_parts(b_ptr.add(i * simd_width), simd_width)
                });
                acc = acc + va * vb;
            }
            sum = acc.reduce_add();
            
            // 尾部标量处理
            for i in (chunks * simd_width)..n {
                sum = sum + unsafe { *a_ptr.add(i) } * unsafe { *b_ptr.add(i) };
            }
            
            sum
        })
    }
}
```

---

## 5. outer 设计

### 5.1 函数签名

```rust
/// 向量外积：result = a ⊗ b，其中 result[i,j] = a[i] * b[j]
///
/// # 参数
///
/// * `a` - 形状为 `(M,)` 的向量
/// * `b` - 形状为 `(N,)` 的向量
///
/// # 返回
///
/// 形状为 `(M, N)` 的矩阵
///
/// # 约束
///
/// - 元素类型必须实现 `Numeric`
///
/// # 示例
///
/// ```
/// use Senon::{Tensor1, outer};
///
/// let a = Tensor1::from_vec(vec![1.0, 2.0]);      // [2]
/// let b = Tensor1::from_vec(vec![3.0, 4.0, 5.0]); // [3]
/// let result = outer(&a, &b);  // 形状 [2, 3]
/// // result[[0, 0]] = 1.0 * 3.0 = 3.0
/// // result[[0, 1]] = 1.0 * 4.0 = 4.0
/// // ...
/// ```
pub fn outer<A, S1, S2>(
    a: &TensorBase<S1, Ix1>,
    b: &TensorBase<S2, Ix1>,
) -> Tensor2<A>
where
    A: Numeric + Copy,
    S1: Storage<Elem = A>,
    S2: Storage<Elem = A>,
{
    let m = a.shape()[0];
    let n = b.shape()[0];
    
    // 默认 F-order 输出
    let mut result = Tensor2::<A>::zeros([m, n]);
    
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let result_ptr = result.as_mut_ptr();
    
    // F-order 填充：先遍历列（b），再遍历行（a）
    // 内存中 result[i, j] = a[i] * b[j]
    // F-order 步长: [1, m]
    for j in 0..n {
        let b_j = unsafe { *b_ptr.add(j) };
        for i in 0..m {
            let a_i = unsafe { *a_ptr.add(i) };
            unsafe {
                *result_ptr.add(j * m + i) = a_i * b_j;
            }
        }
    }
    
    result
}
```

### 5.2 维度推导

```
输入:
  a: shape [M]
  b: shape [N]

输出:
  result: shape [M, N]

计算:
  result[i, j] = a[i] * b[j]
```

### 5.3 内存布局选择

| 选项 | 步长 | 优势 |
|------|------|------|
| **F-order（默认）** | `[1, M]` | 与 BLAS 兼容，后续 matvec 高效 |
| C-order | `[N, 1]` | 与某些 C 库兼容 |

---

## 6. batch_matvec 设计

### 6.1 函数签名

```rust
/// 批量矩阵-向量乘法：y[..., i] = sum_j A[..., i, j] * x[..., j]
///
/// # 参数
///
/// * `matrix` - 形状为 `(..., M, N)` 的张量
/// * `vec` - 形状为 `(..., N)` 的张量
///
/// # 返回
///
/// 形状为 `(..., M)` 的张量
///
/// # Batch 维度约定
///
/// - batch 轴在最前面（轴 0, 1, ..., ndim-3）
/// - 最后 2 维为矩阵（matrix）或最后 1 维为向量（vec）
/// - batch 维度须形状一致或可广播
///
/// # 示例
///
/// ```
/// use Senon::{Tensor3, Tensor2, batch_matvec};
///
/// // 单 batch
/// let a = Tensor3::<f64>::zeros([2, 3, 4]);  // 2 个 (3, 4) 矩阵
/// let x = Tensor2::<f64>::zeros([2, 4]);     // 2 个长度为 4 的向量
/// let y = batch_matvec(&a, &x);              // 形状 [2, 3]
///
/// // 多 batch
/// let a = TensorD::<f64>::zeros(&[5, 3, 2, 4]);  // 5x3 个 (2, 4) 矩阵
/// let x = TensorD::<f64>::zeros(&[5, 3, 4]);     // 5x3 个长度为 4 的向量
/// let y = batch_matvec(&a, &x);                  // 形状 [5, 3, 2]
/// ```
pub fn batch_matvec<A, S1, S2, D>(
    matrix: &TensorBase<S1, D>,
    vec: &TensorBase<S2, D>,
) -> Tensor<A, D>
where
    A: Numeric + Copy,
    S1: Storage<Elem = A>,
    S2: Storage<Elem = A>,
    D: Dimension,
{
    // 维度检查和广播推导
    let (batch_shape, m, n) = infer_batch_matvec_dims(matrix.shape(), vec.shape());
    
    // 分配输出
    let mut out_shape = batch_shape.clone();
    out_shape.push(m);
    let mut result = Tensor::<A, D>::zeros(D::from_slice(&out_shape));
    
    // 执行批量计算
    // ...
    
    result
}
```

### 6.2 维度推导算法

```
function infer_batch_matvec_dims(matrix_shape, vec_shape):
    // matrix_shape: [..., M, N]
    // vec_shape:    [..., N]
    
    matrix_ndim = len(matrix_shape)
    vec_ndim = len(vec_shape)
    
    // 检查最小维度
    if matrix_ndim < 2:
        error("matrix must have at least 2 dimensions")
    if vec_ndim < 1:
        error("vec must have at least 1 dimension")
    
    // 提取矩阵维度
    m = matrix_shape[matrix_ndim - 2]  // 倒数第二维
    n = matrix_shape[matrix_ndim - 1]  // 最后一维
    
    // 提取向量维度
    vec_n = vec_shape[vec_ndim - 1]    // 最后一维
    if vec_n != n:
        error("matrix columns must equal vector length")
    
    // 提取 batch 形状
    matrix_batch = matrix_shape[0:matrix_ndim-2]
    vec_batch = vec_shape[0:vec_ndim-1]
    
    // 广播 batch 维度
    batch_shape = broadcast_shapes(matrix_batch, vec_batch)
    if batch_shape is None:
        error("batch dimensions are not broadcastable")
    
    return (batch_shape, m, n)


function broadcast_shapes(shape1, shape2):
    // 遵循 NumPy 广播规则
    // 从右向左对齐，size-1 维度可扩展
    
    result = []
    i = len(shape1) - 1
    j = len(shape2) - 1
    
    while i >= 0 or j >= 0:
        d1 = shape1[i] if i >= 0 else 1
        d2 = shape2[j] if j >= 0 else 1
        
        if d1 == d2:
            result.insert(0, d1)
        elif d1 == 1:
            result.insert(0, d2)
        elif d2 == 1:
            result.insert(0, d1)
        else:
            return None  // 不可广播
        
        i -= 1
        j -= 1
    
    return result
```

### 6.3 Batch 维度广播示例

| matrix 形状 | vec 形状 | 输出形状 | 说明 |
|-------------|----------|----------|------|
| `(3, 4)` | `(4,)` | `(3,)` | 无 batch，基础 matvec |
| `(2, 3, 4)` | `(2, 4)` | `(2, 3)` | 单 batch 维度 |
| `(5, 2, 3, 4)` | `(5, 2, 4)` | `(5, 2, 3)` | 双 batch 维度 |
| `(3, 4)` | `(1, 4)` | `(1, 3)` | vec batch 广播 |
| `(2, 3, 4)` | `(4,)` | `(2, 3)` | vec 无 batch，广播到每个矩阵 |
| `(2, 1, 3, 4)` | `(1, 5, 4)` | `(2, 5, 3)` | 双向广播 |

### 6.4 实现策略

```rust
fn batch_matvec_impl<A, D>(
    matrix: &Tensor<A, D>,
    vec: &Tensor<A, D>,
    result: &mut Tensor<A, D>,
) where
    A: Numeric + Copy,
    D: Dimension,
{
    let matrix_shape = matrix.shape();
    let vec_shape = vec.shape();
    
    let (batch_shape, m, n) = infer_batch_matvec_dims(matrix_shape, vec_shape);
    let batch_size: usize = batch_shape.iter().product();
    
    // 策略 1: 如果 batch 维度相同且连续，使用批量处理
    if is_batch_aligned(matrix, vec, &batch_shape) {
        batch_matvec_contiguous(matrix, vec, result, batch_size, m, n);
    }
    // 策略 2: 否则逐元素处理
    else {
        batch_matvec_strided(matrix, vec, result, &batch_shape, m, n);
    }
}

fn batch_matvec_contiguous<A>(
    matrix: &Tensor<A, IxDyn>,
    vec: &Tensor<A, IxDyn>,
    result: &mut Tensor<A, IxDyn>,
    batch_size: usize,
    m: usize,
    n: usize,
) {
    // 对每个 batch 元素调用 matvec
    for b in 0..batch_size {
        let mat_view = matrix.index_axis(0, b);  // (M, N)
        let vec_view = vec.index_axis(0, b);     // (N,)
        let mut out_view = result.index_axis_mut(0, b);  // (M,)
        
        // 调用标量或 SIMD matvec
        matvec_into(&mat_view, &vec_view, &mut out_view);
    }
}
```

---

## 7. batch_dot 设计

### 7.1 函数签名

```rust
/// 批量向量内积：result[..., ] = sum(a[..., i] * b[..., i])
///
/// # 参数
///
/// * `a` - 形状为 `(..., N)` 的张量
/// * `b` - 形状为 `(..., N)` 的张量
///
/// # 返回
///
/// 形状为 `(...,)` 的张量（去掉最后维度）
///
/// # 示例
///
/// ```
/// use Senon::{Tensor2, batch_dot};
///
/// let a = Tensor2::<f64>::zeros([3, 4]);  // 3 个长度为 4 的向量
/// let b = Tensor2::<f64>::zeros([3, 4]);
/// let y = batch_dot(&a, &b);  // 形状 [3]
/// ```
pub fn batch_dot<A, S1, S2, D>(
    a: &TensorBase<S1, D>,
    b: &TensorBase<S2, D>,
) -> Tensor<A, D::Smaller>
where
    A: Numeric + Copy,
    S1: Storage<Elem = A>,
    S2: Storage<Elem = A>,
    D: Dimension,
{
    // 维度检查
    assert_eq!(a.ndim(), b.ndim(), "Tensors must have same ndim");
    assert_eq!(a.shape()[a.ndim()-1], b.shape()[b.ndim()-1], "Last dim must match");
    
    // 计算输出形状（去掉最后维度）
    let out_shape = &a.shape()[0..a.ndim()-1];
    let mut result = Tensor::<A, D::Smaller>::zeros(D::Smaller::from_slice(out_shape));
    
    // 执行批量计算
    // ...
    
    result
}
```

### 7.2 维度推导算法

```
function infer_batch_dot_dims(a_shape, b_shape):
    // a_shape: [..., N]
    // b_shape: [..., N]
    
    a_ndim = len(a_shape)
    b_ndim = len(b_shape)
    
    // 检查维度数相同
    if a_ndim != b_ndim:
        error("tensors must have same number of dimensions")
    
    // 检查最后维度相同
    n = a_shape[a_ndim - 1]
    if b_shape[b_ndim - 1] != n:
        error("last dimensions must be equal")
    
    // 检查 batch 维度广播
    a_batch = a_shape[0:a_ndim-1]
    b_batch = b_shape[0:b_ndim-1]
    batch_shape = broadcast_shapes(a_batch, b_batch)
    
    if batch_shape is None:
        error("batch dimensions are not broadcastable")
    
    return (batch_shape, n)
```

---

## 8. batch_add / batch_scale 设计

### 8.1 batch_add 签名

```rust
/// 批量逐元素加法：result = a + b
///
/// # 参数
///
/// * `a` - 任意形状的张量
/// * `b` - 与 `a` 形状兼容的张量
///
/// # 返回
///
/// 形状为广播后形状的张量
///
/// # 广播语义
///
/// 遵循 NumPy 广播规则：
/// - 维度从右向左对齐
/// - size-1 维度可扩展
/// - 缺失维度视为 size-1
///
/// # 示例
///
/// ```
/// use Senon::{Tensor2, batch_add};
///
/// let a = Tensor2::<f64>::zeros([3, 4]);
/// let b = Tensor2::<f64>::zeros([3, 4]);
/// let c = batch_add(&a, &b);  // 形状 [3, 4]
///
/// // 广播示例
/// let a = Tensor2::<f64>::zeros([3, 4]);
/// let b = Tensor1::<f64>::zeros([4]);       // 广播为 (1, 4) -> (3, 4)
/// let c = batch_add(&a, &b);  // 形状 [3, 4]
/// ```
pub fn batch_add<A, S1, S2, D1, D2>(
    a: &TensorBase<S1, D1>,
    b: &TensorBase<S2, D2>,
) -> Tensor<A, IxDyn>
where
    A: Numeric + Copy,
    S1: Storage<Elem = A>,
    S2: Storage<Elem = A>,
    D1: Dimension,
    D2: Dimension,
{
    // 计算广播后的形状
    let out_shape = broadcast::broadcast_shapes(a.shape(), b.shape())
        .expect("shapes are not broadcastable");
    
    let mut result = Tensor::<A, IxDyn>::zeros(IxDyn::from(&out_shape[..]));
    
    // 广播并相加
    let a_broadcast = a.broadcast(&out_shape).unwrap();
    let b_broadcast = b.broadcast(&out_shape).unwrap();
    
    for (r, (av, bv)) in result.iter_mut().zip(a_broadcast.iter().zip(b_broadcast.iter())) {
        *r = av + bv;
    }
    
    result
}
```

### 8.2 batch_scale 签名

```rust
/// 批量标量缩放：result = a * scalar
///
/// # 参数
///
/// * `a` - 任意形状的张量
/// * `scalar` - 标量值
///
/// # 返回
///
/// 与 `a` 形状相同的张量
///
/// # 示例
///
/// ```
/// use Senon::{Tensor2, batch_scale};
///
/// let a = Tensor2::<f64>::zeros([3, 4]);
/// let scaled = batch_scale(&a, 2.0);  // 形状 [3, 4]
/// ```
pub fn batch_scale<A, S, D>(
    a: &TensorBase<S, D>,
    scalar: A,
) -> Tensor<A, D>
where
    A: Numeric + Copy,
    S: Storage<Elem = A>,
    D: Dimension,
{
    let mut result = a.to_owned();
    
    for r in result.iter_mut() {
        *r = *r * scalar;
    }
    
    result
}
```

### 8.3 广播语义详细说明

```
广播规则（NumPy 兼容）：

1. 维度对齐（从右向左）：
   a: [3, 4]
   b: [4]
   对齐后:
   a: [3, 4]
   b: [1, 4]

2. size-1 扩展：
   b: [1, 4] -> [3, 4]  （复制 3 次）

3. 缺失维度补 1：
   a: [3, 4]
   b: [] （标量）
   对齐后:
   a: [3, 4]
   b: [1, 1] -> [3, 4]

4. 不兼容情况：
   a: [3, 4]
   b: [3, 5]
   最后一维 4 != 5 且都不是 1，广播失败
```

---

## 9. BLAS 兼容性

### 9.1 FFI 指针 API 集成

Senon 提供 FFI 指针 API，允许上游库直接调用 BLAS：

```rust
// 上游库使用示例

fn blas_matvec<A: RealScalar>(
    matrix: &Tensor2<A>,
    vec: &Tensor1<A>,
) -> Tensor1<A> {
    let (m, n) = (matrix.shape()[0], matrix.shape()[1]);
    let mut result = Tensor1::<A>::zeros([m]);
    
    // 检查 BLAS 兼容性
    if !matrix.is_blas_compatible() || !vec.is_contiguous() {
        // 回退到 Senon 内部实现
        return Senon::matvec(matrix, vec);
    }
    
    unsafe {
        // 获取 BLAS 参数
        let trans = if matrix.is_f_contiguous() { b'N' } else { b'T' };
        let lda = matrix.lda().unwrap() as i32;
        let m_i32 = m as i32;
        let n_i32 = n as i32;
        let alpha = A::one();
        let beta = A::zero();
        let incx = 1i32;
        let incy = 1i32;
        
        // 调用 BLAS dgemv/sagemv
        blas::dgemv(
            &trans,
            &m_i32,
            &n_i32,
            &alpha,
            matrix.as_ptr(),
            &lda,
            vec.as_ptr(),
            &incx,
            &beta,
            result.as_mut_ptr(),
            &incy,
        );
    }
    
    result
}
```

### 9.2 BLAS 兼容性检查

```rust
impl<S, D> TensorBase<S, D> {
    /// 检查是否可直接传递给 BLAS
    ///
    /// BLAS 要求：
    /// - 连续内存（F 或 C）
    /// - 正步长
    /// - 无零步长
    pub fn is_blas_compatible(&self) -> bool {
        self.is_contiguous() 
            && !self.has_zero_stride()
            && !self.has_neg_stride()
    }
    
    /// 返回 BLAS 布局标识
    pub fn blas_layout(&self) -> Option<BlasLayout> {
        if !self.is_blas_compatible() {
            return None;
        }
        if self.is_f_contiguous() {
            Some(BlasLayout::ColumnMajor)
        } else {
            Some(BlasLayout::RowMajor)
        }
    }
    
    /// 返回 leading dimension（LDA）
    ///
    /// F-order: LDA = stride[0]（第一轴步长）
    /// C-order: LDA = stride[1]（第二轴步长）
    pub fn lda(&self) -> Option<isize>
    where
        D: Dimension,
    {
        if self.ndim() != 2 || !self.is_blas_compatible() {
            return None;
        }
        let strides = self.strides();
        if self.is_f_contiguous() {
            Some(strides[0])
        } else {
            Some(strides[1])
        }
    }
}
```

### 9.3 GEMM 排除理由

| 考量 | 说明 |
|------|------|
| **实现复杂度** | 高性能 GEMM 需要分块、缓存优化、多线程，超出本库范围 |
| **BLAS 成熟度** | OpenBLAS、MKL、BLIS 等已高度优化，重复造轮子无意义 |
| **FFI 集成** | 通过指针 API，上游库可轻松调用 BLAS GEMM |
| **维护成本** | GEMM 优化需要针对不同架构持续调优 |

**推荐做法**：
- 使用 Senon 进行张量管理和基础运算
- 需要矩阵乘法时，通过 FFI 调用 BLAS

---

## 10. SIMD 加速

### 10.1 支持的操作

| 操作 | SIMD 加速 | 条件 |
|------|-----------|------|
| `matvec` | ✓ | 连续内存 |
| `dot` | ✓ | 连续内存 |
| `outer` | 部分 | 输出填充可向量化 |
| `batch_matvec` | ✓ | 每个 batch 元素满足 SIMD 条件 |
| `batch_dot` | ✓ | 每个 batch 元素满足 SIMD 条件 |
| `batch_add` | ✓ | 连续内存 |
| `batch_scale` | ✓ | 连续内存 |

### 10.2 SIMD 路径选择

```
function select_simd_path(tensor):
    if not cfg(feature = "simd"):
        return Scalar
    
    if not tensor.is_contiguous():
        return Scalar
    
    if tensor.len() < SIMD_WIDTH:
        return Scalar
    
    if tensor.is_aligned():
        return SimdAligned
    else:
        return SimdUnaligned
```

### 10.3 matvec SIMD 实现

```rust
#[cfg(feature = "simd")]
mod simd {
    use pulp::{Arch, Simd};
    
    pub fn matvec_impl<A>(matrix: &Tensor2<A>, vec: &Tensor1<A>) -> Tensor1<A>
    where
        A: RealScalar,
    {
        let (m, n) = (matrix.shape()[0], matrix.shape()[1]);
        let mut result = Tensor1::<A>::zeros([m]);
        
        let arch = Arch::new();
        let vec_ptr = vec.as_ptr();
        
        // C-order 优化：每行连续，适合 SIMD
        if matrix.is_c_contiguous() {
            let matrix_ptr = matrix.as_ptr();
            
            arch.dispatch(|| {
                for i in 0..m {
                    let row_ptr = unsafe { matrix_ptr.add(i * n) };
                    result[i] = simd_dot_row(row_ptr, vec_ptr, n);
                }
            });
        }
        // F-order 或其他布局：使用通用实现
        else {
            scalar::matvec_impl(matrix, vec, &mut result);
        }
        
        result
    }
    
    fn simd_dot_row<A: RealScalar>(row: *const A, vec: *const A, n: usize) -> A {
        // 使用 pulp 的 SIMD 抽象
        // 实现 SIMD 点积
    }
}
```

### 10.4 指令集优先级

| 指令集 | 架构 | 优先级 | pulp 支持 |
|--------|------|--------|-----------|
| AVX-512 | x86_64 | 最高 | ✓ |
| AVX2 + FMA | x86_64 | 高 | ✓ |
| SSE4.1 | x86_64 | 中 | ✓ |
| NEON | aarch64 | 高 | ✓ |
| 标量 | 所有 | 最低 | - |

---

## 11. 与其他模块的交互

### 11.1 模块依赖图

```
┌─────────────────────────────────────────────────────────────┐
│                      ops/matrix.rs                          │
│  matvec, dot, outer, batch_matvec, batch_dot, batch_*      │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│    tensor     │   │   storage     │   │   layout      │
│ TensorBase    │   │ Storage trait │   │ is_contiguous │
│ shape/strides │   │ as_ptr()      │   │ is_aligned    │
└───────────────┘   └───────────────┘   └───────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   broadcast   │   │     ffi       │   │     simd      │
│ broadcast_    │   │ is_blas_      │   │ pulp Arch     │
│ shapes        │   │ compatible    │   │ dispatch      │
└───────────────┘   └───────────────┘   └───────────────┘
```

### 11.2 与 tensor 模块的接口

```rust
// 使用 tensor 模块的类型和方法
use crate::tensor::{TensorBase, Tensor, Tensor1, Tensor2, TensorView};
use crate::dimension::{Dimension, Ix1, Ix2, IxDyn};
use crate::storage::Storage;

// 依赖的方法
// - shape(): &[usize]
// - strides(): &[isize]
// - as_ptr(): *const A
// - as_mut_ptr(): *mut A
// - is_contiguous(): bool
// - is_f_contiguous(): bool
// - is_c_contiguous(): bool
// - is_aligned(): bool
```

### 11.3 与 layout 模块的接口

```rust
// 使用 layout 模块的布局检查
use crate::layout::LayoutFlags;

// 依赖的布局信息
// - is_f_contiguous(): bool
// - is_c_contiguous(): bool
// - is_contiguous(): bool
// - is_aligned(): bool
// - has_zero_stride(): bool
// - has_neg_stride(): bool
```

### 11.4 与 broadcast 模块的接口

```rust
// 使用 broadcast 模块的广播规则
use crate::broadcast::{broadcast_shapes, BroadcastError};

// batch 操作依赖
fn infer_batch_dims(a_shape: &[usize], b_shape: &[usize]) -> Result<Vec<usize>, BroadcastError> {
    broadcast_shapes(a_shape, b_shape)
}
```

### 11.5 与 ffi 模块的接口

```rust
// 使用 ffi 模块的 BLAS 兼容性检查
use crate::ffi::{is_blas_compatible, blas_layout, lda};

// 上游库可通过这些接口调用 BLAS
```

### 11.6 与 simd 模块的接口

```rust
// 使用 simd 模块的向量化
#[cfg(feature = "simd")]
use crate::simd::{SimdOps, ArchExt};

// SIMD 路径选择
#[cfg(feature = "simd")]
fn simd_path<A: RealScalar>() -> bool { true }

#[cfg(not(feature = "simd"))]
fn simd_path<A: RealScalar>() -> bool { false }
```

---

## 12. 实现任务分解

### 任务 1：定义公开 API 签名

**预计时间**: 10 分钟

**任务内容**:
- 定义 `matvec`, `dot`, `outer` 函数签名
- 定义 `batch_matvec`, `batch_dot`, `batch_add`, `batch_scale` 函数签名
- 添加完整文档注释

**验收标准**:
- 所有函数签名编译通过
- 泛型约束正确

---

### 任务 2：实现维度检查逻辑

**预计时间**: 10 分钟

**任务内容**:
- 实现 matvec 维度检查（矩阵 (M,N) × 向量 (N,)）
- 实现 dot 维度检查（向量长度相等）
- 实现 outer 维度检查（任意长度）

**验收标准**:
- 错误消息清晰
- panic 条件正确

---

### 任务 3：实现标量 matvec

**预计时间**: 10 分钟

**任务内容**:
- 实现 `scalar::matvec_impl`
- 处理 F-order 和 C-order 布局
- 正确计算索引偏移

**验收标准**:
- 结果正确
- 非连续输入正确处理

---

### 任务 4：实现标量 dot/inner

**预计时间**: 10 分钟

**任务内容**:
- 实现 `scalar::dot_impl`
- 实现简单的循环累加

**验收标准**:
- 结果正确
- 边界情况（空向量、单元素）正确

---

### 任务 5：实现 outer

**预计时间**: 10 分钟

**任务内容**:
- 实现 `outer` 函数
- 生成 F-order 输出矩阵
- 双重循环填充

**验收标准**:
- 输出形状正确
- 内存布局正确

---

### 任务 6：实现 batch 维度推导

**预计时间**: 10 分钟

**任务内容**:
- 实现 `infer_batch_matvec_dims`
- 实现 `infer_batch_dot_dims`
- 实现 `broadcast_shapes` 辅助函数

**验收标准**:
- 广播规则正确
- 错误情况处理正确

---

### 任务 7：实现 batch_matvec

**预计时间**: 10 分钟

**任务内容**:
- 实现 batch_matvec 主体逻辑
- 调用 matvec 处理每个 batch 元素
- 处理广播情况

**验收标准**:
- batch 形状正确
- 广播正确

---

### 任务 8：实现 batch_dot 和 batch_add/scale

**预计时间**: 10 分钟

**任务内容**:
- 实现 `batch_dot`
- 实现 `batch_add`
- 实现 `batch_scale`

**验收标准**:
- 所有函数工作正确
- 广播语义正确

---

### 任务 9：实现 SIMD matvec 和 dot

**预计时间**: 15 分钟

**任务内容**:
- 实现 `simd::matvec_impl`
- 实现 `simd::dot_impl`
- 使用 pulp 的 SIMD 抽象
- 处理尾部元素

**验收标准**:
- SIMD 路径正确
- 标量回退正确

---

### 任务 10：编写单元测试

**预计时间**: 15 分钟

**任务内容**:
- matvec 测试（F/C order）
- dot 测试（各种长度）
- outer 测试
- batch 操作测试
- 边界情况测试

**验收标准**:
- 覆盖率 ≥ 80%
- 所有测试通过

---

## 13. 设计决策记录

### D1: 为什么不包含 GEMM？

**决策**: 矩阵运算模块不实现 GEMM（矩阵-矩阵乘法）。

**理由**:
1. **实现复杂度高**: 高性能 GEMM 需要：
   - 分块（blocking/tiling）优化缓存
   - 多线程并行
   - 针对不同 CPU 架构的微优化
   - 持续的性能调优

2. **BLAS 已足够成熟**: OpenBLAS、Intel MKL、BLIS 等库已提供高度优化的 GEMM 实现，其性能难以超越。

3. **FFI 集成更灵活**: 通过 Senon 的 FFI 指针 API，上游库可以：
   - 使用 Senon 进行张量管理
   - 需要矩阵乘法时直接调用 BLAS
   - 选择最适合的 BLAS 实现

4. **维护成本**: 维护高性能 GEMM 需要持续投入，超出本库资源范围。

**替代方案**: 用户需要 GEMM 时，使用 BLAS 绑定库（如 `blas-src`）配合 Senon 的 FFI API。

---

### D2: 为什么默认输出 F-order？

**决策**: `outer` 等操作默认生成 F-order（列优先）布局的输出。

**理由**:
1. **BLAS 兼容**: BLAS/LAPACK 使用列优先布局
2. **后续操作高效**: 生成的矩阵可用于后续 matvec 等 BLAS 操作
3. **项目惯例**: Senon 默认 F-order，保持一致性

---

### D3: 为什么 batch 维度在最前面？

**决策**: batch 轴位于张量的最前面（轴 0, 1, ..., ndim-3）。

**理由**:
1. **NumPy/PyTorch 惯例**: 与主流框架一致
2. **索引友好**: `tensor[i]` 获取第 i 个 batch 元素
3. **内存局部性**: batch 维度在最外层，遍历时缓存友好

---

### D4: 为什么使用 panic 而非 Result？

**决策**: 维度不匹配等错误使用 panic 而非返回 `Result`。

**理由**:
1. **编程错误**: 维度不匹配通常是编程错误，不可恢复
2. **API 简洁**: 避免到处 `?` 和 `unwrap()`
3. **性能**: 无错误处理开销
4. **与 ndarray 一致**: ndarray 也使用 panic

**可选方案**: 未来可提供 `_checked` 变体返回 `Result`。

---

### D5: 为什么 dot 不使用 Kahan 补偿求和？

**决策**: 当前版本 `dot` 使用直接求和，不实现 Kahan 补偿。

**理由**:
1. **SIMD 兼容**: Kahan 求和难以向量化
2. **常见场景精度足够**: 对于大多数科学计算场景，直接求和精度足够
3. **性能优先**: 保持 SIMD 路径的高效性
4. **替代方案**: 需要高精度时，用户可拆分向量或使用专门库

---

## 附录 A: API 快速参考

```rust
// 基础操作
pub fn matvec<A, S1, S2>(matrix: &TensorBase<S1, Ix2>, vec: &TensorBase<S2, Ix1>) -> Tensor1<A>
where A: Numeric + Copy, S1: Storage<Elem = A>, S2: Storage<Elem = A>;

pub fn dot<A, S1, S2>(a: &TensorBase<S1, Ix1>, b: &TensorBase<S2, Ix1>) -> A
where A: Numeric + Copy, S1: Storage<Elem = A>, S2: Storage<Elem = A>;

pub fn inner<A, S1, S2>(a: &TensorBase<S1, Ix1>, b: &TensorBase<S2, Ix1>) -> A
where A: Numeric + Copy, S1: Storage<Elem = A>, S2: Storage<Elem = A>;  // dot 的别名

pub fn outer<A, S1, S2>(a: &TensorBase<S1, Ix1>, b: &TensorBase<S2, Ix1>) -> Tensor2<A>
where A: Numeric + Copy, S1: Storage<Elem = A>, S2: Storage<Elem = A>;

// 批量操作
pub fn batch_matvec<A, S1, S2, D>(matrix: &TensorBase<S1, D>, vec: &TensorBase<S2, D>) -> Tensor<A, D>
where A: Numeric + Copy, S1: Storage<Elem = A>, S2: Storage<Elem = A>, D: Dimension;

pub fn batch_dot<A, S1, S2, D>(a: &TensorBase<S1, D>, b: &TensorBase<S2, D>) -> Tensor<A, D::Smaller>
where A: Numeric + Copy, S1: Storage<Elem = A>, S2: Storage<Elem = A>, D: Dimension;

pub fn batch_add<A, S1, S2, D1, D2>(a: &TensorBase<S1, D1>, b: &TensorBase<S2, D2>) -> Tensor<A, IxDyn>
where A: Numeric + Copy, S1: Storage<Elem = A>, S2: Storage<Elem = A>, D1: Dimension, D2: Dimension;

pub fn batch_scale<A, S, D>(a: &TensorBase<S, D>, scalar: A) -> Tensor<A, D>
where A: Numeric + Copy, S: Storage<Elem = A>, D: Dimension;
```

---

## 附录 B: 形状推导示例

### matvec

```
输入: matrix (M, N), vec (N,)
输出: (M,)

示例:
  matrix: (3, 4), vec: (4,) -> output: (3,)
```

### batch_matvec

```
输入: matrix (..., M, N), vec (..., N)
输出: (..., M)

示例:
  matrix: (2, 3, 4), vec: (2, 4) -> output: (2, 3)
  matrix: (5, 2, 3, 4), vec: (5, 2, 4) -> output: (5, 2, 3)
  matrix: (2, 3, 4), vec: (4,) -> output: (2, 3)  // vec 广播
```

### batch_dot

```
输入: a (..., N), b (..., N)
输出: (...,)

示例:
  a: (3, 4), b: (3, 4) -> output: (3,)
  a: (5, 3, 4), b: (5, 3, 4) -> output: (5, 3)
```

---

## 附录 C: 性能建议

1. **优先使用连续数组**: 非连续数组回退到标量路径
2. **对齐内存**: 使用 Senon 默认的 64 字节对齐以启用 SIMD
3. **批量操作**: 对于多个小矩阵，使用 batch 操作减少函数调用开销
4. **BLAS 集成**: 对于大型矩阵乘法，通过 FFI 调用 BLAS

---

*本文档由 Senon 项目维护。如有问题请提交 Issue 或 PR。*
