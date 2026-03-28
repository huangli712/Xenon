# 矩阵运算模块设计

> 文档编号: 15 | 模块: `src/ops/matrix.rs` | 阶段: Phase 3
> 前置文档: `00-rust-standards.md`, `01-architecture-overview.md`, `07-tensor-core.md`, `03-element-types.md`

---

## 1. 模块定位

`ops::matrix` 模块是 Xenon 的线性代数原语层，提供矩阵-向量乘法、向量内积/外积、矩阵乘法、张量缩并、Kronecker 积、Einstein 求和及其批量变体。所有函数均为**自由函数**（free functions），操作对象为 `TensorBase<S, D>` 的引用（通过 `RawStorage` trait bound 同时接受 Owned / View / ArcRepr），返回新分配的 `Tensor<A, D>` 类型结果。

**核心设计目标：**

| 目标 | 体现 |
|------|------|
| 类型安全 | 专用函数（matvec / matmul / outer）使用静态维度（Ix1 / Ix2），编译时保证维度数正确 |
| 泛型灵活 | 通用函数（dot / tensordot / einsum）使用 IxDyn，支持任意维度输入 |
| F-order 优先 | 内层循环针对列优先布局优化，数据访问模式对缓存友好 |
| BLAS 兼容 | 提供 FFI 辅助函数，上游库可通过指针 API 将数据直接传递给 BLAS |
| 渐进加速 | 标量回退 → SIMD → BLAS FFI（外部），三层 dispatch 链 |

**本模块职责边界：**

| 职责 | 包含 | 不包含 |
|------|------|--------|
| 向量运算 | `dot`（内积）、`outer`（外积） | 逐元素运算（由 `element_wise` 提供） |
| 矩阵-向量 | `matvec`、`batch_matvec` | 解线性方程组 |
| 矩阵-矩阵 | `matmul`、`batch_matmul` | GEMM 完整接口（αAB + βC） |
| 通用缩并 | `tensordot`、`einsum` | 自动微分 |
| 特殊积 | `kron`（Kronecker） | 张量积（tensor product 通用形式） |
| 批量运算 | `batch_dot`、`batch_add`、`batch_scale` | 分布式批量 |

**重要约束（需求 §10.2）：**

- **不包含 GEMM**（全功能 `C = α·op(A)·op(B) + β·C`）。本模块的 `matmul` 仅为 `C = A·B` 的简化接口。完整 GEMM 由上游 BLAS 库通过 FFI 指针 API 实现。
- 矩阵乘法的并行不由 Xenon 管理（需求 §7.2.2：矩阵乘法由 BLAS 内部管理线程）。

---

## 2. 文件位置

```
src/ops/
    mod.rs              # pub mod matrix; + re-export 公共函数
    matrix.rs           # 本模块：所有矩阵运算函数 + 内部 dispatch
```

`src/lib.rs` 中的集成：

```rust
pub mod ops;
pub use crate::ops::*;
```

`src/ops/mod.rs` 中的集成：

```rust
pub mod matrix;
pub use crate::ops::matrix::{
    dot, inner, matvec, matmul, outer,
    tensordot, einsum, kron,
    batch_matmul, batch_matvec, batch_dot,
    batch_add, batch_scale,
    AxesArg, EinsumError,
};
```

单文件设计理由：矩阵运算函数共享大量内部逻辑（dispatch 决策、BLAS 兼容性检查、workspace 分配），拆分到多文件会增加模块间调用开销。预计文件行数 ~2000 行，在可维护范围内。

---

## 3. 依赖关系

```
src/ops/matrix.rs
    │
    ├── crate::tensor          # TensorBase, Tensor, TensorView, type aliases
    ├── crate::dimension       # Dimension, Ix0~Ix6, IxDyn
    ├── crate::element         # Element, Numeric, RealScalar, ComplexScalar
    ├── crate::layout          # LayoutFlags, Order
    ├── crate::error           # TensorError, Result
    ├── crate::storage         # RawStorage, Storage, Owned, ViewRepr, ViewMutRepr, ArcRepr
    │
    │   (可选)
    ├── crate::workspace       # Workspace（workspace buffer 分配）
    │
    │   (feature-gated)
    ├── crate::backend::simd   # pulp SIMD 路径（feature = "simd"）
    └── crate::broadcast       # 广播维度（batch 操作的 batch 维广播）
```

### 3.1 依赖的具体类型

| 来源模块 | 使用的类型 / trait |
|----------|-------------------|
| `tensor` | `TensorBase<S, D>`, `Tensor<A, D>`, `Tensor0<A>`, `Tensor1<A>`, `Tensor2<A>`, `TensorD<A>` |
| `dimension` | `Dimension`, `Ix0`, `Ix1`, `Ix2`, `IxDyn` |
| `element` | `Element`, `Numeric`, `RealScalar` |
| `layout` | `LayoutFlags`, `Order` |
| `error` | `TensorError`, `Result` |
| `storage` | `RawStorage`（只读访问 bound） |
| `workspace` | `Workspace`（中间缓冲区分配） |
| `broadcast` | 广播形状计算（batch 维度广播） |

### 3.2 依赖方向

```
matrix.rs → tensor, dimension, element, layout, error, storage
matrix.rs → workspace（可选，用于中间缓冲区）
matrix.rs → broadcast（batch 广播计算）
matrix.rs → backend::simd（feature-gated，SIMD 加速）
```

`matrix.rs` 不被任何 Phase 2 核心模块依赖，属于 Phase 3 API 层，与 `element_wise.rs`、`reduction.rs` 等同级模块无依赖关系。

---

## 4. 公共 API 设计

### 4.1 辅助类型

#### 4.1.1 AxesArg — tensordot 轴参数

```rust
/// Specifies which axes to contract in `tensordot`.
///
/// # Examples
///
/// Contract last 1 axis of `a` with first 1 axis of `b` (matrix multiply):
/// ```
/// use xenon::ops::AxesArg;
/// let axes = AxesArg::Count(1);
/// ```
///
/// Contract specific axes:
/// ```
/// use xenon::ops::AxesArg;
/// let axes = AxesArg::Pair(vec![0, 2], vec![1, 3]);
/// ```
#[derive(Debug, Clone)]
pub enum AxesArg {
    /// Contract the last `n` axes of `a` with the first `n` axes of `b`.
    Count(usize),

    /// Contract the specified axes of `a` with the specified axes of `b`.
    ///
    /// The two vectors must have the same length. Each axis in `a_axes`
    /// is paired with the corresponding axis in `b_axes`, and the paired
    /// axes must have matching lengths.
    Pair(Vec<usize>, Vec<usize>),
}
```

#### 4.1.2 EinsumError — einsum 专用错误

```rust
/// Errors that can occur during Einstein summation parsing and execution.
#[derive(Debug, Clone, thiserror::Error)]
pub enum EinsumError {
    /// The subscript string is malformed.
    #[error("invalid einsum subscript: {reason}")]
    InvalidSubscript {
        reason: String,
    },

    /// The number of operands does not match the subscript.
    #[error("operand count mismatch: expected {expected}, got {actual}")]
    OperandCountMismatch {
        expected: usize,
        actual: usize,
    },

    /// Shape mismatch for a shared index label.
    #[error("shape mismatch for index '{label}': operand {op_a} has {size_a}, operand {op_b} has {size_b}")]
    ShapeMismatch {
        label: char,
        op_a: usize,
        size_a: usize,
        op_b: usize,
        size_b: usize,
    },

    /// An index label appears in more than two input operands but not in output.
    #[error("ambiguous index '{label}': appears in more than two operands without output specification")]
    AmbiguousIndex {
        label: char,
    },

    /// The operation would require more memory than available.
    #[error("workspace allocation failed: requested {requested} bytes")]
    WorkspaceOverflow {
        requested: usize,
    },
}
```

### 4.2 dot — 通用点积

`dot` 根据输入维度数自动分派到正确的语义：

| 输入维度 | 语义 | 结果形状 |
|----------|------|----------|
| 1D × 1D | 向量内积 | 标量 (Ix0) |
| 2D × 1D | 矩阵-向量乘 | 向量 (Ix1) |
| 1D × 2D | 向量-矩阵乘 | 向量 (Ix1) |
| 2D × 2D | 矩阵-矩阵乘 | 矩阵 (Ix2) |

```rust
/// Computes the dot product of two tensors.
///
/// The behavior depends on the dimensionality of the inputs:
///
/// | `a` dims | `b` dims | Operation | Result shape |
/// |----------|----------|-----------|-------------|
/// | 1D       | 1D       | Inner product | Scalar (Ix0) |
/// | 2D       | 1D       | Matrix-vector multiply | 1D |
/// | 1D       | 2D       | Vector-matrix multiply | 1D |
/// | 2D       | 2D       | Matrix-matrix multiply | 2D |
///
/// For all other dimension combinations, use [`tensordot`] instead.
///
/// # Arguments
///
/// * `a` - Left operand (view or owned).
/// * `b` - Right operand (view or owned).
///
/// # Errors
///
/// Returns `TensorError::ShapeMismatch` if:
/// - 1D × 1D: lengths differ.
/// - 2D × 1D: `a.shape[1] != b.shape[0]`.
/// - 1D × 2D: `a.shape[0] != b.shape[0]`.
/// - 2D × 2D: `a.shape[1] != b.shape[0]`.
///
/// Returns `TensorError::InvalidOperation` if either operand has
/// dimensionality other than 1 or 2.
///
/// # Examples
///
/// ```
/// use xenon::{Tensor1, Tensor2, Ix1, Ix2};
/// use xenon::ops::dot;
///
/// // Inner product of two vectors
/// let a: Tensor1<f64> = /* ... */;
/// let b: Tensor1<f64> = /* ... */;
/// let result = dot(&a, &b)?;
/// assert_eq!(result.ndim(), 0);
///
/// // Matrix-vector multiply
/// let mat: Tensor2<f64> = /* ... */; // shape [3, 4]
/// let vec: Tensor1<f64> = /* ... */; // shape [4]
/// let result = dot(&mat, &vec)?;      // shape [3]
/// ```
pub fn dot<A, S1, S2>(
    a: &TensorBase<S1, IxDyn>,
    b: &TensorBase<S2, IxDyn>,
) -> Result<TensorD<A>>
where
    A: Numeric,
    S1: RawStorage<Elem = A>,
    S2: RawStorage<Elem = A>,
{
    match (a.ndim(), b.ndim()) {
        (1, 1) => { /* inner product → Ix0 */ }
        (2, 1) => { /* matvec → Ix1 */ }
        (1, 2) => { /* vecmat → Ix1 */ }
        (2, 2) => { /* matmul → Ix2 */ }
        _ => Err(TensorError::InvalidOperation {
            reason: format!(
                "dot requires 1D or 2D operands, got {}D × {}D; use tensordot instead",
                a.ndim(), b.ndim()
            ),
        }),
    }
}
```

**设计决策：** `dot` 返回 `TensorD<A>`（动态维度），因为返回的维度数取决于输入维度组合。对于需要静态类型安全的场景，使用专用函数 `inner`、`matvec`、`matmul`。

### 4.3 inner — 向量内积

```rust
/// Computes the inner product of two 1-D tensors.
///
/// Returns a scalar tensor (0-D) equal to `Σ a[i] * b[i]`.
///
/// # Arguments
///
/// * `a` - First vector (length N).
/// * `b` - Second vector (length N).
///
/// # Errors
///
/// Returns `TensorError::ShapeMismatch` if the lengths differ.
///
/// # Examples
///
/// ```
/// use xenon::ops::inner;
/// let a = xenon::array(&[1.0_f64, 2.0, 3.0]);
/// let b = xenon::array(&[4.0, 5.0, 6.0]);
/// let result = inner(&a, &b)?; // 1*4 + 2*5 + 3*6 = 32.0
/// ```
pub fn inner<A, S1, S2>(
    a: &TensorBase<S1, Ix1>,
    b: &TensorBase<S2, Ix1>,
) -> Result<Tensor0<A>>
where
    A: Numeric,
    S1: RawStorage<Elem = A>,
    S2: RawStorage<Elem = A>,
{
    let n = a.len();
    if b.len() != n {
        return Err(TensorError::ShapeMismatch {
            expected: vec![n],
            actual: vec![b.len()],
        });
    }
    // Scalar fallback: accumulate a[i] * b[i]
    let mut sum = A::zero();
    for i in 0..n {
        // SAFETY: i < n = a.len() = b.len()
        unsafe {
            let av = *a.get_unchecked(&[i]);
            let bv = *b.get_unchecked(&[i]);
            sum = sum + av * bv;
        }
    }
    Ok(Tensor0::from_scalar(sum))
}
```

### 4.4 matvec — 矩阵-向量乘法

```rust
/// Computes the matrix-vector product `y = A · x`.
///
/// Given a matrix `A` of shape `[M, N]` and a vector `x` of shape `[N]`,
/// returns a vector `y` of shape `[M]` where `y[i] = Σ_j A[i, j] * x[j]`.
///
/// # Arguments
///
/// * `mat` - Matrix of shape `[M, N]`.
/// * `vec` - Vector of shape `[N]`.
///
/// # Errors
///
/// Returns `TensorError::ShapeMismatch` if `mat.shape[1] != vec.shape[0]`.
///
/// # Performance
///
/// For F-contiguous matrices (the Xenon default), the inner loop iterates
/// over the rows of `mat` for each element of `vec`, which is cache-friendly.
/// For non-contiguous inputs, a scalar fallback is used.
///
/// # Examples
///
/// ```
/// use xenon::ops::matvec;
/// // A: 2×3, x: 3 → y: 2
/// let a = xenon::array(&[[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]]);
/// let x = xenon::array(&[1.0, 1.0, 1.0]);
/// let y = matvec(&a, &x)?;
/// assert_eq!(y.shape(), &[2]);
/// ```
pub fn matvec<A, S1, S2>(
    mat: &TensorBase<S1, Ix2>,
    vec: &TensorBase<S2, Ix1>,
) -> Result<Tensor1<A>>
where
    A: Numeric,
    S1: RawStorage<Elem = A>,
    S2: RawStorage<Elem = A>,
{
    let (m, n) = (mat.shape()[0], mat.shape()[1]);
    if vec.shape()[0] != n {
        return Err(TensorError::ShapeMismatch {
            expected: vec![n],
            actual: vec![vec.shape()[0]],
        });
    }

    let mut result = Tensor1::<A>::uninitialized(m);
    // Dispatch: scalar / SIMD based on layout and features
    dispatch_matvec(mat, vec, &mut result, m, n);
    Ok(result)
}
```

### 4.5 matmul — 矩阵-矩阵乘法

```rust
/// Computes the matrix-matrix product `C = A · B`.
///
/// Given `A` of shape `[M, K]` and `B` of shape `[K, N]`, returns `C` of
/// shape `[M, N]` where `C[i, j] = Σ_k A[i, k] * B[k, j]`.
///
/// This is a simplified matmul interface (`C = A·B`, no α/β scalars).
/// For full GEMM semantics (`C = α·op(A)·op(B) + β·C`), use an external
/// BLAS library via the FFI pointer API.
///
/// # Arguments
///
/// * `a` - Left matrix of shape `[M, K]`.
/// * `b` - Right matrix of shape `[K, N]`.
///
/// # Errors
///
/// Returns `TensorError::ShapeMismatch` if `a.shape[1] != b.shape[0]`.
///
/// # Performance
///
/// The implementation uses a tiled loop order optimized for F-contiguous
/// (column-major) data layout. For `f32` and `f64`, SIMD vectorization
/// is applied to the inner dot product when the `simd` feature is enabled.
///
/// # Examples
///
/// ```
/// use xenon::ops::matmul;
/// let a = xenon::array(&[[1.0_f64, 2.0], [3.0, 4.0]]);
/// let b = xenon::array(&[[5.0, 6.0], [7.0, 8.0]]);
/// let c = matmul(&a, &b)?;
/// ```
pub fn matmul<A, S1, S2>(
    a: &TensorBase<S1, Ix2>,
    b: &TensorBase<S2, Ix2>,
) -> Result<Tensor2<A>>
where
    A: Numeric,
    S1: RawStorage<Elem = A>,
    S2: RawStorage<Elem = A>,
{
    let (m, k1) = (a.shape()[0], a.shape()[1]);
    let (k2, n) = (b.shape()[0], b.shape()[1]);
    if k1 != k2 {
        return Err(TensorError::ShapeMismatch {
            expected: vec![m, k2],
            actual: vec![m, k1],
        });
    }

    let mut result = Tensor2::<A>::uninitialized([m, n]);
    dispatch_matmul(a, b, &mut result, m, n, k1);
    Ok(result)
}
```

### 4.6 outer — 向量外积

```rust
/// Computes the outer product of two 1-D tensors.
///
/// Given vectors `a` of length `N` and `b` of length `M`, returns a
/// matrix of shape `[N, M]` where `result[i, j] = a[i] * b[j]`.
///
/// # Arguments
///
/// * `a` - First vector of length `N`.
/// * `b` - Second vector of length `M`.
///
/// # Examples
///
/// ```
/// use xenon::ops::outer;
/// let a = xenon::array(&[1.0_f64, 2.0, 3.0]);
/// let b = xenon::array(&[4.0, 5.0]);
/// let c = outer(&a, &b)?;
/// // c.shape() == [3, 2]
/// // c == [[4, 5], [8, 10], [12, 15]]
/// ```
pub fn outer<A, S1, S2>(
    a: &TensorBase<S1, Ix1>,
    b: &TensorBase<S2, Ix1>,
) -> Result<Tensor2<A>>
where
    A: Numeric,
    S1: RawStorage<Elem = A>,
    S2: RawStorage<Elem = A>,
{
    let n = a.len();
    let m = b.len();
    let mut result = Tensor2::<A>::zeros([n, m]);
    for i in 0..n {
        let ai = unsafe { *a.get_unchecked(&[i]) };
        for j in 0..m {
            let bj = unsafe { *b.get_unchecked(&[j]) };
            unsafe {
                result.get_unchecked_mut(&[i, j]).write(ai * bj);
            }
        }
    }
    Ok(result)
}
```

### 4.7 tensordot — 张量缩并

```rust
/// Computes the tensor contraction of two tensors over specified axes.
///
/// This generalizes dot, inner, outer, and matmul to arbitrary dimensions.
/// The contracted axes are summed over, and the remaining axes form the
/// output tensor.
///
/// # Arguments
///
/// * `a` - First tensor.
/// * `b` - Second tensor.
/// * `axes` - Which axes to contract. See [`AxesArg`].
///
/// # Errors
///
/// Returns `TensorError::ShapeMismatch` if:
/// - `AxesArg::Count(n)` and `a.ndim() < n` or `b.ndim() < n`.
/// - `AxesArg::Pair` and the specified axes have mismatched lengths.
/// - `AxesArg::Pair` and the two axis lists have different lengths.
///
/// # Output Shape
///
/// The output shape is the concatenation of:
/// 1. Non-contracted axes of `a` (in order).
/// 2. Non-contracted axes of `b` (in order).
///
/// # Examples
///
/// ```
/// use xenon::ops::{tensordot, AxesArg};
///
/// // Matrix multiply: contract last axis of a with first axis of b
/// let a = /* shape [3, 4] */;
/// let b = /* shape [4, 5] */;
/// let c = tensordot(&a, &b, AxesArg::Count(1))?;
/// // c.shape() == [3, 5]
///
/// // Contract specific axes
/// let a = /* shape [3, 4, 5] */;
/// let b = /* shape [4, 3, 2] */;
/// let c = tensordot(&a, &b, AxesArg::Pair(vec![0, 1], vec![1, 0]))?;
/// // Contracts a.axes[0] with b.axes[1] (both size 3)
/// // and a.axes[1] with b.axes[0] (both size 4)
/// // c.shape() == [5, 2]
/// ```
pub fn tensordot<A, S1, S2, D1, D2>(
    a: &TensorBase<S1, D1>,
    b: &TensorBase<S2, D2>,
    axes: AxesArg,
) -> Result<TensorD<A>>
where
    A: Numeric,
    S1: RawStorage<Elem = A>,
    S2: RawStorage<Elem = A>,
    D1: Dimension,
    D2: Dimension,
{
    // 1. Parse axes → (a_axes, b_axes) pairs
    // 2. Validate paired axis lengths match
    // 3. Compute output shape = non-contracted a axes + non-contracted b axes
    // 4. Transpose a and b so contracted axes are innermost
    // 5. Reshape to 2D × 2D → matmul
    // 6. Reshape result to output shape
    ...
}
```

### 4.8 einsum — Einstein 求和

```rust
/// Evaluates the Einstein summation convention on the operands.
///
/// The subscript string defines the contraction pattern using single
/// character labels. The syntax follows NumPy's `np.einsum`:
///
/// - Input subscripts for each operand, separated by commas.
/// - `->` followed by output subscripts (optional).
/// - Repeated labels in inputs are summed over (trace, diagonal).
/// - Labels present in inputs but absent from output are summed over.
/// - `...` (ellipsis) enables broadcasting for unspecified dimensions.
///
/// # Arguments
///
/// * `subscripts` - Einstein summation notation string (e.g., `"ij,jk->ik"`).
/// * `operands` - Slice of tensors to contract.
///
/// # Errors
///
/// Returns [`EinsumError`] for:
/// - Malformed subscript strings.
/// - Operand count mismatch.
/// - Shape mismatch for shared labels.
///
/// # Examples
///
/// ```
/// use xenon::ops::einsum;
///
/// // Matrix multiply
/// let a = /* shape [3, 4] */;
/// let b = /* shape [4, 5] */;
/// let c = einsum::<f64>("ij,jk->ik", &[&a, &b])?;
///
/// // Trace
/// let m = /* shape [3, 3] */;
/// let trace = einsum::<f64>("ii->", &[&m])?;
///
/// // Outer product
/// let x = /* shape [3] */;
/// let y = /* shape [4] */;
/// let z = einsum::<f64>("i,j->ij", &[&x, &y])?;
///
/// // Batch matrix multiply
/// let a = /* shape [2, 3, 4] */;
/// let b = /* shape [2, 4, 5] */;
/// let c = einsum::<f64>("bij,bjk->bik", &[&a, &b])?;
/// ```
pub fn einsum<A>(
    subscripts: &str,
    operands: &[&TensorD<A>],
) -> Result<TensorD<A>, EinsumError>
where
    A: Numeric,
{
    // 1. Parse subscript string → EinsumSpec
    // 2. Validate operand count
    // 3. Resolve label → size mapping (validate shape consistency)
    // 4. Determine output shape (explicit `-> ...` or implicit)
    // 5. Optimize contraction order (minimize intermediate size)
    // 6. Execute contractions via tensordot / diagonal / transpose
    ...
}
```

**设计决策：**

- `einsum` 接受 `&[&TensorD<A>]` 而非泛型多维参数：Rust 不支持可变泛型参数（varargs），且 einsum 本身处理任意维度数，动态维度是最自然的选择。
- 返回 `Result<_, EinsumError>` 而非 `Result<_, TensorError>`：einsum 有大量特有的错误模式（下标解析、歧义标签等），独立错误类型更清晰。调用方可通过 `EinsumError` → `TensorError` 转换统一处理。

### 4.9 kron — Kronecker 积

```rust
/// Computes the Kronecker product of two tensors.
///
/// Given `a` of shape `[n1, n2, ...]` and `b` of shape `[m1, m2, ...]`,
/// the result has shape `[n1*m1, n2*m2, ...]` where:
///
/// ```text
/// result[i1*m1 + j1, i2*m2 + j2, ...] = a[i1, i2, ...] * b[j1, j2, ...]
/// ```
///
/// For 1-D inputs, this is equivalent to `outer(a, b).flatten()`.
/// For 2-D inputs, this produces the block matrix `a[i,j] * b`.
///
/// # Arguments
///
/// * `a` - First tensor.
/// * `b` - Second tensor.
///
/// # Errors
///
/// Returns `TensorError::InvalidOperation` if the tensors have
/// different numbers of dimensions.
///
/// # Examples
///
/// ```
/// use xenon::ops::kron;
///
/// // 2×2 Kronecker product
/// let a = xenon::array(&[[1.0_f64, 2.0], [3.0, 4.0]]);
/// let b = xenon::array(&[[0.0, 5.0], [6.0, 7.0]]);
/// let k = kron(&a, &b)?;
/// // k.shape() == [4, 4]
/// ```
pub fn kron<A, S1, S2, D1, D2>(
    a: &TensorBase<S1, D1>,
    b: &TensorBase<S2, D2>,
) -> Result<TensorD<A>>
where
    A: Numeric,
    S1: RawStorage<Elem = A>,
    S2: RawStorage<Elem = A>,
    D1: Dimension,
    D2: Dimension,
{
    let ndim = a.ndim();
    if b.ndim() != ndim {
        return Err(TensorError::InvalidOperation {
            reason: format!(
                "kron requires operands with the same ndim, got {}D and {}D",
                ndim, b.ndim()
            ),
        });
    }
    // Compute output shape: [a.shape[i] * b.shape[i] for i in 0..ndim]
    // Allocate output and fill: result[a_idx * b_shape + b_idx] = a[a_idx] * b[b_idx]
    ...
}
```

### 4.10 batch_matmul — 批量矩阵-矩阵乘法

```rust
/// Computes batched matrix-matrix multiplication.
///
/// Given `a` of shape `[..., M, K]` and `b` of shape `[..., K, N]`,
/// returns a tensor of shape `[..., M, N]` where `...` denotes batch
/// dimensions. The batch dimensions of `a` and `b` must be identical
/// or broadcastable following NumPy rules.
///
/// # Batch Dimension Convention
///
/// - The last 2 axes are matrix dimensions (M×K and K×N).
/// - All preceding axes (axis 0 through ndim-3) are batch axes.
/// - Batch axes are broadcast using NumPy rules.
///
/// # Arguments
///
/// * `a` - Batch of matrices, shape `[..., M, K]`.
/// * `b` - Batch of matrices, shape `[..., K, N]`.
///
/// # Errors
///
/// Returns `TensorError::ShapeMismatch` if:
/// - `a.shape[-1] != b.shape[-2]` (inner dimension mismatch).
/// - Batch dimensions are incompatible for broadcasting.
///
/// # Examples
///
/// ```
/// use xenon::ops::batch_matmul;
///
/// // Batch of 4 matrices, each 3×4 × 4×5 → 4×3×5
/// let a = /* shape [4, 3, 4] */;
/// let b = /* shape [4, 4, 5] */;
/// let c = batch_matmul(&a, &b)?;
///
/// // Broadcasting: [2, 1, 3, 4] × [1, 5, 4, 5] → [2, 5, 3, 5]
/// ```
pub fn batch_matmul<A>(
    a: &TensorD<A>,
    b: &TensorD<A>,
) -> Result<TensorD<A>>
where
    A: Numeric,
{
    let ndim_a = a.ndim();
    let ndim_b = b.ndim();
    if ndim_a < 2 || ndim_b < 2 {
        return Err(TensorError::InvalidOperation {
            reason: format!(
                "batch_matmul requires at least 2D operands, got {}D and {}D",
                ndim_a, ndim_b
            ),
        });
    }

    let m = a.shape()[ndim_a - 2];
    let k_a = a.shape()[ndim_a - 1];
    let k_b = b.shape()[ndim_b - 2];
    let n = b.shape()[ndim_b - 1];

    if k_a != k_b {
        return Err(TensorError::ShapeMismatch {
            expected: vec![k_b],
            actual: vec![k_a],
        });
    }

    // Compute broadcast batch shape and total batch size
    // Iterate over broadcast batch indices
    // For each batch index, extract 2D views and call dispatch_matmul
    ...
}
```

### 4.11 batch_matvec — 批量矩阵-向量乘法

```rust
/// Computes batched matrix-vector multiplication.
///
/// Given `mat` of shape `[..., M, N]` and `vec` of shape `[..., N]`,
/// returns a tensor of shape `[..., M]`.
///
/// # Batch Dimension Convention
///
/// - For `mat`, the last 2 axes are matrix dimensions (M×N).
/// - For `vec`, the last axis is the vector dimension (N).
/// - All preceding axes are batch axes and must be broadcastable.
///
/// # Arguments
///
/// * `mat` - Batch of matrices, shape `[..., M, N]`.
/// * `vec` - Batch of vectors, shape `[..., N]`.
///
/// # Errors
///
/// Returns `TensorError::ShapeMismatch` if:
/// - `mat.shape[-1] != vec.shape[-1]` (inner dimension mismatch).
/// - Batch dimensions are incompatible for broadcasting.
///
/// # Examples
///
/// ```
/// use xenon::ops::batch_matvec;
///
/// // Batch of 8 matrices, each 3×4, and 8 vectors of length 4
/// let mat = /* shape [8, 3, 4] */;
/// let vec = /* shape [8, 4] */;
/// let result = batch_matvec(&mat, &vec)?;
/// // result.shape() == [8, 3]
/// ```
pub fn batch_matvec<A>(
    mat: &TensorD<A>,
    vec: &TensorD<A>,
) -> Result<TensorD<A>>
where
    A: Numeric,
{
    let ndim_mat = mat.ndim();
    let ndim_vec = vec.ndim();
    if ndim_mat < 2 {
        return Err(TensorError::InvalidOperation {
            reason: format!(
                "batch_matvec requires mat to be at least 2D, got {}D",
                ndim_mat
            ),
        });
    }
    if ndim_vec < 1 {
        return Err(TensorError::InvalidOperation {
            reason: format!(
                "batch_matvec requires vec to be at least 1D, got {}D",
                ndim_vec
            ),
        });
    }

    let m = mat.shape()[ndim_mat - 2];
    let n_mat = mat.shape()[ndim_mat - 1];
    let n_vec = vec.shape()[ndim_vec - 1];

    if n_mat != n_vec {
        return Err(TensorError::ShapeMismatch {
            expected: vec![n_mat],
            actual: vec![n_vec],
        });
    }

    // Compute broadcast batch shape from mat[..ndim_mat-2] and vec[..ndim_vec-1]
    // Iterate over broadcast batch indices
    // For each batch index, extract 2D mat view and 1D vec view, call dispatch_matvec
    ...
}
```

### 4.12 batch_dot — 批量向量内积

```rust
/// Computes batched dot product along the last axis.
///
/// Given `a` of shape `[..., N]` and `b` of shape `[..., N]`,
/// returns a tensor of shape `[..., ]` (batch dimensions only).
///
/// Batch dimensions must be identical or broadcastable.
///
/// # Arguments
///
/// * `a` - Batch of vectors, shape `[..., N]`.
/// * `b` - Batch of vectors, shape `[..., N]`.
///
/// # Errors
///
/// Returns `TensorError::ShapeMismatch` if:
/// - Last dimensions differ (`a.shape[-1] != b.shape[-1]`).
/// - Batch dimensions are incompatible for broadcasting.
pub fn batch_dot<A>(
    a: &TensorD<A>,
    b: &TensorD<A>,
) -> Result<TensorD<A>>
where
    A: Numeric,
{
    let ndim_a = a.ndim();
    let ndim_b = b.ndim();
    if ndim_a == 0 || ndim_b == 0 {
        return Err(TensorError::InvalidOperation {
            reason: "batch_dot requires at least 1D operands".into(),
        });
    }

    let n_a = a.shape()[ndim_a - 1];
    let n_b = b.shape()[ndim_b - 1];
    if n_a != n_b {
        return Err(TensorError::ShapeMismatch {
            expected: vec![n_a],
            actual: vec![n_b],
        });
    }

    // Compute broadcast batch shape from a[..ndim_a-1] and b[..ndim_b-1]
    // For each batch index, compute inner product of the last axis
    ...
}
```

### 4.13 batch_add — 批量逐元素加法

```rust
/// Computes batched element-wise addition with broadcasting.
///
/// Input shapes must be identical or broadcastable.
///
/// # Arguments
///
/// * `a` - First operand.
/// * `b` - Second operand.
///
/// # Errors
///
/// Returns `TensorError::BroadcastError` if shapes are incompatible.
pub fn batch_add<A>(
    a: &TensorD<A>,
    b: &TensorD<A>,
) -> Result<TensorD<A>>
where
    A: Numeric,
{
    // Delegate to broadcast + element-wise add
    ...
}
```

### 4.14 batch_scale — 批量标量乘法

```rust
/// Computes batched element-wise scaling: `result = scalar * a`.
///
/// Multiplies every element of `a` by the scalar `scalar`.
///
/// # Arguments
///
/// * `a` - Input tensor.
/// * `scalar` - Scalar multiplier.
///
/// # Examples
///
/// ```
/// use xenon::ops::batch_scale;
/// let a = xenon::array(&[1.0_f64, 2.0, 3.0]);
/// let result = batch_scale(&a, 2.0)?;
/// // result == [2.0, 4.0, 6.0]
/// ```
pub fn batch_scale<A>(
    a: &TensorD<A>,
    scalar: A,
) -> Result<TensorD<A>>
where
    A: Numeric,
{
    // Map each element: elem * scalar
    ...
}
```

### 4.15 FFI 辅助函数（BLAS 互操作）

```rust
/// Extracts BLAS-compatible parameters for a matrix-vector operation.
///
/// Returns `(data_ptr, lda, trans)` for the given 2D tensor, suitable
/// for passing to BLAS `dgemv`/`sgemv`.
///
/// # Arguments
///
/// * `mat` - A 2-D tensor (matrix).
///
/// # Returns
///
/// A tuple `(ptr, rows, cols, lda, layout, trans)`:
/// - `ptr`: raw pointer to matrix data.
/// - `rows`, `cols`: logical matrix dimensions.
/// - `lda`: leading dimension (stride between columns in F-order).
/// - `layout`: `Order::F` or `Order::C`.
/// - `trans`: whether the matrix needs to be logically transposed for BLAS.
///
/// Returns `None` if the matrix layout is not BLAS-compatible
/// (e.g., non-contiguous or has zero/negative strides).
pub fn blas_mat_params<A, S>(
    mat: &TensorBase<S, Ix2>,
) -> Option<(*const A, usize, usize, usize, Order, bool)>
where
    A: Element,
    S: RawStorage<Elem = A>,
{
    if !mat.is_contiguous() || mat.has_zero_stride() || mat.has_neg_stride() {
        return None;
    }
    let rows = mat.shape()[0];
    let cols = mat.shape()[1];
    let ptr = mat.as_ptr();
    if mat.is_f_contiguous() {
        Some((ptr, rows, cols, rows, Order::F, false))
    } else {
        // C-contiguous: BLAS sees it as transposed F-order
        Some((ptr, cols, rows, cols, Order::F, true))
    }
}
```

---

## 5. 内部实现设计

### 5.1 Dispatch 分层策略

所有运算函数通过内部 dispatch 函数选择执行路径。dispatch 决策基于以下条件：

```
                    ┌──────────────────────┐
                    │ 输入形状 / 布局检查  │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  布局是否连续？       │
                    │  (F-contig/C-contig) │
                    └──┬──────────────┬────┘
                Yes    │              │  No
                       ▼              ▼
            ┌──────────────┐  ┌──────────────────┐
            │ 元素类型？    │  │ 标量回退路径      │
            │ f32 / f64？  │  │ (嵌套循环，       │
            └──┬───────┬───┘  │  逐元素访问)     │
        Yes    │       │ No   └──────────────────┘
               ▼       ▼
    ┌──────────────┐  ┌────────────┐
    │ SIMD 路径    │  │ 标量路径   │
    │ (pulp)      │  │ (通用      │
    │ f32/f64 only│  │  Numeric)  │
    └──────────────┘  └────────────┘
```

**Dispatch 决策表：**

| 条件 | 执行路径 | 说明 |
|------|----------|------|
| 非连续输入 | 标量回退 | 逐元素通过 strides 寻址 |
| 连续输入 + 整数类型 | 标量优化路径 | 连续内存遍历，无 SIMD |
| 连续输入 + f32/f64 + `simd` feature | SIMD 路径 | pulp 向量化内积 |
| 连续输入 + f32/f64 + 无 `simd` | 标量优化路径 | 编译器自动向量化 |

**注意：** 本模块**不直接调用 BLAS**。BLAS 调用由上游库通过 `blas_mat_params()` 等 FFI 辅助函数获取参数后自行完成。

### 5.2 标量回退实现 — matvec

以 `matvec` 为例说明标量回退路径的 F-order 优化：

```rust
/// Internal dispatch for matrix-vector multiplication.
///
/// Dispatches to SIMD or scalar path based on layout and element type.
fn dispatch_matvec<A, S1, S2>(
    mat: &TensorBase<S1, Ix2>,
    vec: &TensorBase<S2, Ix1>,
    result: &mut Tensor1<A>,
    m: usize,
    n: usize,
) where
    A: Numeric,
    S1: RawStorage<Elem = A>,
    S2: RawStorage<Elem = A>,
{
    // Fill result with zeros first
    result.fill(A::zero());

    if mat.is_f_contiguous() {
        // F-contiguous: iterate column-by-column
        // result[i] += mat[i, j] * vec[j]
        // mat column j = &mat_data[j * m .. (j+1) * m] (contiguous!)
        for j in 0..n {
            let vj = unsafe { *vec.get_unchecked(&[j]) };
            for i in 0..m {
                let mij = unsafe { *mat.get_unchecked(&[i, j]) };
                unsafe {
                    let ri = result.get_unchecked_mut(&[i]);
                    ri.write(ri.read() + mij * vj);
                }
            }
        }
    } else {
        // Non-contiguous: generic element access via strides
        for i in 0..m {
            let mut sum = A::zero();
            for j in 0..n {
                let mij = unsafe { *mat.get_unchecked(&[i, j]) };
                let vj = unsafe { *vec.get_unchecked(&[j]) };
                sum = sum + mij * vj;
            }
            unsafe {
                result.get_unchecked_mut(&[i]).write(sum);
            }
        }
    }
}
```

**F-order 优化要点：**

- F-contiguous 矩阵的每一列在内存中连续，因此 `mat[i, j]` 与 `mat[i+1, j]` 相邻。
- 外层循环遍历 `j`（列），内层循环遍历 `i`（行），使得内层循环的内存访问是连续的。
- `vec[j]` 在外层循环中只读取一次，适合寄存器缓存。

### 5.3 标量回退实现 — matmul

```rust
/// Internal dispatch for matrix-matrix multiplication.
fn dispatch_matmul<A, S1, S2>(
    a: &TensorBase<S1, Ix2>,
    b: &TensorBase<S2, Ix2>,
    result: &mut Tensor2<A>,
    m: usize,
    n: usize,
    k: usize,
) where
    A: Numeric,
    S1: RawStorage<Elem = A>,
    S2: RawStorage<Elem = A>,
{
    // Initialize result to zero
    result.fill(A::zero());

    if a.is_f_contiguous() && b.is_f_contiguous() {
        // Both F-contiguous: jki loop order
        // C[:, j] += A[:, k] * B[k, j]
        // A[:, k] is contiguous (stride = 1)
        // B[:, j] is contiguous (stride = 1)
        // C[:, j] is contiguous (stride = 1)
        for j in 0..n {
            for ki in 0..k {
                let bkj = unsafe { *b.get_unchecked(&[ki, j]) };
                for i in 0..m {
                    unsafe {
                        let cij = result.get_unchecked_mut(&[i, j]);
                        cij.write(cij.read() + *a.get_unchecked(&[i, ki]) * bkj);
                    }
                }
            }
        }
    } else if a.is_c_contiguous() && b.is_c_contiguous() {
        // Both C-contiguous: ikj loop order
        for i in 0..m {
            for ki in 0..k {
                let aik = unsafe { *a.get_unchecked(&[i, ki]) };
                for j in 0..n {
                    unsafe {
                        let cij = result.get_unchecked_mut(&[i, j]);
                        cij.write(cij.read() + aik * *b.get_unchecked(&[ki, j]));
                    }
                }
            }
        }
    } else {
        // Generic fallback: ijk loop order
        for i in 0..m {
            for j in 0..n {
                let mut sum = A::zero();
                for ki in 0..k {
                    let aik = unsafe { *a.get_unchecked(&[i, ki]) };
                    let bkj = unsafe { *b.get_unchecked(&[ki, j]) };
                    sum = sum + aik * bkj;
                }
                unsafe {
                    result.get_unchecked_mut(&[i, j]).write(sum);
                }
            }
        }
    }
}
```

**循环顺序选择原理：**

| 输入布局 | 循环顺序 | 最内层访问模式 |
|----------|----------|---------------|
| A: F, B: F | j → k → i | `A[i,k]` 连续（沿 i）、`B[k,j]` 标量、`C[i,j]` 连续（沿 i） |
| A: C, B: C | i → k → j | `A[i,k]` 标量、`B[k,j]` 连续（沿 j）、`C[i,j]` 连续（沿 j） |
| 混合 | i → j → k | 通用回退，可能缓存不友好 |

### 5.4 SIMD 路径（feature = "simd"）

当 `simd` feature 启用时，为 `f32`/`f64` 的内积和矩阵乘法提供 SIMD 加速路径：

```rust
#[cfg(feature = "simd")]
mod simd_dispatch {
    use crate::backend::simd::Arch;
    use crate::element::RealScalar;

    /// SIMD-accelerated inner product for f32/f64.
    ///
    /// Processes elements in SIMD-width batches, falls back to scalar
    /// for the tail remainder.
    pub fn inner_simd<A>(a_ptr: *const A, b_ptr: *const A, n: usize) -> A
    where
        A: RealScalar,
    {
        let arch = Arch::new();
        arch.dispatch(|| {
            // SIMD accumulation loop (width-dependent)
            // Scalar tail loop for remainder
            ...
        })
    }

    /// SIMD-accelerated matvec for F-contiguous matrices.
    pub fn matvec_f_order_simd<A>(
        mat_ptr: *const A,   // F-contiguous, shape [m, n], lda = m
        vec_ptr: *const A,   // contiguous, length n
        result_ptr: *mut A,  // output, length m
        m: usize,
        n: usize,
    ) where
        A: RealScalar,
    {
        // For each column j of mat:
        //   result[i] += mat[i + j*m] * vec[j]  (i = 0..m)
        // SIMD: accumulate using SIMD registers for the i loop
        ...
    }
}
```

**SIMD 策略：**

| 操作 | SIMD 宽度 (f64) | SIMD 宽度 (f32) | 尾部处理 |
|------|-----------------|-----------------|----------|
| 内积 | 4 (AVX2) / 8 (AVX-512) | 8 (AVX2) / 16 (AVX-512) | 标量循环 |
| matvec 行 | 同上 | 同上 | 标量循环 |
| matmul 内层 | 同上 | 同上 | 标量循环 |

### 5.5 Workspace 缓冲区管理

对于 `tensordot` 和 `batch_*` 操作，可能需要临时缓冲区用于：

1. **转置缓冲区**：将非连续维度重排到连续位置
2. **Batch 中间结果**：批量操作中每个 batch slice 的独立结果
3. **Einsum 中间张量**：多步缩并的中间结果

```rust
/// Internal workspace allocation for matrix operations.
///
/// Uses the `workspace` module for aligned allocation.
/// Falls back to heap allocation if no workspace is provided.
struct MatrixWorkspace {
    /// Buffer for transposition / reshaping intermediates.
    transpose_buf: Option<Vec<u8>>,
}

impl MatrixWorkspace {
    /// Allocates a workspace buffer of the given capacity (in elements).
    fn allocate<A: Element>(&mut self, capacity: usize) -> &mut [A] {
        let byte_size = capacity * core::mem::size_of::<A>();
        // Use 64-byte aligned allocation
        ...
    }
}
```

**Workspace 使用场景：**

| 操作 | 需要的缓冲区大小 | 用途 |
|------|-----------------|------|
| `tensordot` (非连续) | `max(a.len(), b.len()) * sizeof(A)` | 转置输入到连续布局 |
| `batch_matmul` | `batch_size * m * n * sizeof(A)` | 临时存储每个 batch 的结果 |
| `einsum` (多步) | 取决于缩并顺序 | 中间张量 |

### 5.6 Batch 处理策略

批量运算的核心策略：

```
batch_matmul(a: [B1, B2, M, K], b: [B1, B2, K, N])
    │
    ├── 1. 计算广播后的 batch 形状 (broadcast_batch_shape)
    │       a_batch = [B1, B2], b_batch = [B1, B2]
    │       result_batch = [B1, B2]
    │
    ├── 2. 计算总 batch 数 = B1 * B2
    │
    ├── 3. 遍历所有 broadcast flat 索引 (0..total_batch)
    │       for flat_idx in 0..total_batch {
    │           // 将 flat_idx 解码为多维 batch 索引
    │           let batch_idx = unravel(flat_idx, broadcast_batch_shape);
    │
    │           // 从 a 中提取对应的 2D slice
    │           // broadcast 意味着 size=1 的 batch 维会被重复
    │           let a_batch_idx = broadcast_index(batch_idx, a_batch_shape);
    │           let b_batch_idx = broadcast_index(batch_idx, b_batch_shape);
    │
    │           let a_2d = a.index_axis_iter(...)  // 2D view
    │           let b_2d = b.index_axis_iter(...)  // 2D view
    │           let mut c_2d = result.index_axis_mut(...) // 2D mutable view
    │
    │           dispatch_matmul(&a_2d, &b_2d, &mut c_2d, m, n, k);
    │       }
    │
    └── 4. 返回结果 Tensor
```

**Batch 维度广播实现：**

```rust
/// Computes the broadcast shape of two batch dimension tuples.
///
/// Follows NumPy broadcasting rules:
/// - Dimensions are aligned from the right.
/// - Dimensions must be equal, or one of them must be 1.
fn broadcast_batch_shape(
    a_batch: &[usize],
    b_batch: &[usize],
) -> Result<Vec<usize>> {
    use core::cmp::Ordering;
    let max_len = a_batch.len().max(b_batch.len());
    let mut result = Vec::with_capacity(max_len);

    // Pad shorter shape with 1s on the left
    let (a_pad, b_pad) = pad_broadcast(a_batch, b_batch, max_len);

    for (da, db) in a_pad.iter().zip(b_pad.iter()) {
        match (*da, *db) {
            (da, db) if da == db => result.push(da),
            (1, db) => result.push(db),
            (da, 1) => result.push(da),
            (da, db) => {
                return Err(TensorError::BroadcastError {
                    reason: format!(
                        "cannot broadcast batch dimensions: {} vs {}",
                        da, db
                    ),
                });
            }
        }
    }
    Ok(result)
}
```

### 5.7 F-order (列优先) 优化策略

Xenon 默认使用 F-order（列优先），所有矩阵运算的内层循环都针对此布局优化：

| 操作 | F-order 最优循环顺序 | 内存访问模式 |
|------|---------------------|-------------|
| `inner(a, b)` | 线性遍历 i = 0..N | `a[i]` 和 `b[i]` 均连续 |
| `matvec(A, x)` | 外层 j，内层 i | `A[:, j]` 连续（列向量） |
| `matmul(A, B)` | j → k → i | `A[:, k]` 连续、`C[:, j]` 连续 |
| `tensordot` | 转置后降级为 matmul | 同 matmul |

**结果张量布局：** 所有运算结果默认分配为 F-contiguous（列优先），与 Xenon 的默认布局一致。

### 5.8 tensordot 实现策略

`tensordot` 的核心思路是将通用缩并降级为矩阵乘法：

```
tensordot(a, b, axes)
    │
    ├── 1. 解析 axes → (a_contract_axes, b_contract_axes)
    │
    ├── 2. 分离自由轴和缩并轴
    │       a_free = a 的非缩并轴
    │       b_free = b 的非缩并轴
    │
    ├── 3. 重排 a 和 b 的维度顺序
    │       a' = transpose(a, [a_free..., a_contract...])
    │       b' = transpose(b, [b_contract..., b_free...])
    │
    ├── 4. Reshape 为 2D
    │       a_2d = a'.reshape([free_prod_a, contract_prod])
    │       b_2d = b'.reshape([contract_prod, free_prod_b])
    │
    ├── 5. 矩阵乘法
    │       c_2d = matmul(a_2d, b_2d)  // [free_prod_a, free_prod_b]
    │
    └── 6. Reshape 结果
            result = c_2d.reshape([a_free_shape..., b_free_shape...])
```

### 5.9 einsum 实现策略

einsum 解析和执行的完整流程：

```
einsum("ij,jk->ik", &[a, b])
    │
    ├── 1. 解析 subscript 字符串
    │       ├── 按 ',' 分割输入子标签 → ["ij", "jk"]
    │       ├── 解析 '->' 后的输出标签 → "ik"
    │       └── 验证标签合法性（单字符、无重复输出标签）
    │
    ├── 2. 构建 label → size 映射
    │       ├── 遍历每个操作数的子标签
    │       ├── 将 label 关联到对应轴长度
    │       └── 验证同一 label 在不同操作数中长度一致
    │
    ├── 3. 确定输出形状
    │       ├── 显式 (有 "->")：按输出标签顺序收集轴长度
    │       └── 隐式 (无 "->")：按字母序收集仅出现一次的标签
    │
    ├── 4. 优化缩并路径
    │       ├── 对于 2 操作数：直接 tensordot
    │       ├── 对于 3+ 操作数：贪心选择最小中间结果的缩并对
    │       └── 缩并顺序不影响结果正确性，仅影响性能
    │
    └── 5. 执行缩并
            ├── 对角线操作（重复标签在同一输入中，如 "ii->i"）
            ├── 两两 tensordot（逐步缩并）
            └── 最终转置到目标输出顺序
```

**EinsumSpec 内部结构：**

```rust
/// Parsed representation of an einsum subscript.
struct EinsumSpec {
    /// Subscript labels for each input operand.
    input_labels: Vec<Vec<char>>,

    /// Subscript labels for the output.
    output_labels: Vec<char>,

    /// Map from label character to its size (validated across all operands).
    label_sizes: Vec<(char, usize)>,
}
```

---

## 6. 实现任务拆分

> 每个任务约 10 分钟，可独立验证和提交。

### 6.1 基础设施

- [ ] **T1: 模块骨架 + 辅助类型定义**
  - 文件: `src/ops/matrix.rs:1-80`
  - 内容: `AxesArg` enum、`EinsumError` enum、模块 import、`MatrixWorkspace` struct 骨架
  - 测试: 编译通过
  - 前置: tensor, element, layout, error 模块完成
  - 预计: 10 min

- [ ] **T2: broadcast_batch_shape 辅助函数**
  - 文件: `src/ops/matrix.rs`
  - 内容: `broadcast_batch_shape()` 和 `pad_broadcast()` 内部函数
  - 测试: `test_broadcast_batch_equal`, `test_broadcast_batch_padding`, `test_broadcast_batch_incompatible`
  - 前置: T1
  - 预计: 10 min

### 6.2 向量运算

- [ ] **T3: inner — 向量内积**
  - 文件: `src/ops/matrix.rs`
  - 内容: `inner()` 函数完整实现（标量路径）
  - 测试: `test_inner_basic`, `test_inner_shape_mismatch`, `test_inner_zero_length`, `test_inner_float_precision`
  - 前置: T1
  - 预计: 10 min

- [ ] **T4: outer — 向量外积**
  - 文件: `src/ops/matrix.rs`
  - 内容: `outer()` 函数完整实现
  - 测试: `test_outer_2x3`, `test_outer_1x1`, `test_outer_zero_length`
  - 前置: T1
  - 预计: 10 min

### 6.3 矩阵运算

- [ ] **T5: dispatch_matvec 内部函数 + matvec 公共 API**
  - 文件: `src/ops/matrix.rs`
  - 内容: `dispatch_matvec()` 内部 dispatch 函数（F-contig / C-contig / generic 三条路径）+ `matvec()` 公共函数
  - 测试: `test_matvec_f_contiguous`, `test_matvec_c_contiguous`, `test_matvec_non_contiguous`, `test_matvec_shape_mismatch`
  - 前置: T1
  - 预计: 10 min

- [ ] **T6: dispatch_matmul 内部函数 + matmul 公共 API**
  - 文件: `src/ops/matrix.rs`
  - 内容: `dispatch_matmul()` 内部 dispatch 函数（F/F、C/C、混合三条路径）+ `matmul()` 公共函数
  - 测试: `test_matmul_f_contiguous`, `test_matmul_c_contiguous`, `test_matmul_mixed_layout`, `test_matmul_shape_mismatch`, `test_matmul_identity`
  - 前置: T1
  - 预计: 10 min

- [ ] **T7: dot — 通用点积分派**
  - 文件: `src/ops/matrix.rs`
  - 内容: `dot()` 函数，根据 ndim 分派到 inner / matvec / vecmat / matmul
  - 测试: `test_dot_1d_1d`, `test_dot_2d_1d`, `test_dot_1d_2d`, `test_dot_2d_2d`, `test_dot_invalid_dims`
  - 前置: T3, T5, T6
  - 预计: 10 min

### 6.4 通用缩并

- [ ] **T8: tensordot — 解析与形状计算**
  - 文件: `src/ops/matrix.rs`
  - 内容: `AxesArg` 解析逻辑、自由轴 / 缩并轴分离、输出形状计算
  - 测试: `test_axes_count_parsing`, `test_axes_pair_parsing`, `test_tensordot_output_shape`, `test_tensordot_shape_mismatch`
  - 前置: T1
  - 预计: 10 min

- [ ] **T9: tensordot — 执行（transpose + matmul 降级）**
  - 文件: `src/ops/matrix.rs`
  - 内容: `tensordot()` 完整实现：转置 → reshape → matmul → reshape
  - 测试: `test_tensordot_matrix_multiply`, `test_tensordot_3d_contract`, `test_tensordot_pair_axes`
  - 前置: T6, T8
  - 预计: 10 min

- [ ] **T10: einsum — subscript 解析器**
  - 文件: `src/ops/matrix.rs`
  - 内容: `EinsumSpec` struct、subscript 字符串解析、label → size 验证、输出形状推导
  - 测试: `test_einsum_parse_basic`, `test_einsum_parse_implicit_output`, `test_einsum_parse_invalid`, `test_einsum_parse_shape_mismatch`
  - 前置: T1
  - 预计: 10 min

- [ ] **T11: einsum — 执行引擎**
  - 文件: `src/ops/matrix.rs`
  - 内容: `einsum()` 完整实现：解析 → 优化路径 → 逐步 tensordot / diagonal → 输出
  - 测试: `test_einsum_matmul`, `test_einsum_trace`, `test_einsum_outer`, `test_einsum_batch_matmul`
  - 前置: T9, T10
  - 预计: 15 min

- [ ] **T12: kron — Kronecker 积**
  - 文件: `src/ops/matrix.rs`
  - 内容: `kron()` 完整实现
  - 测试: `test_kron_2x2`, `test_kron_1d`, `test_kron_3d`, `test_kron_ndim_mismatch`
  - 前置: T1
  - 预计: 10 min

### 6.5 批量运算

- [ ] **T13: batch_matmul**
  - 文件: `src/ops/matrix.rs`
  - 内容: `batch_matmul()` 实现，batch 维广播 + 循环调用 `dispatch_matmul`
  - 测试: `test_batch_matmul_basic`, `test_batch_matmul_broadcast`, `test_batch_matmul_shape_mismatch`
  - 前置: T2, T6
  - 预计: 10 min

- [ ] **T14: batch_matvec**
  - 文件: `src/ops/matrix.rs`
  - 内容: `batch_matvec()` 实现，类似 batch_matmul 的 batch 循环
  - 测试: `test_batch_matvec_basic`, `test_batch_matvec_broadcast`, `test_batch_matvec_shape_mismatch`
  - 前置: T2, T5
  - 预计: 10 min

- [ ] **T15: batch_dot + batch_add + batch_scale**
  - 文件: `src/ops/matrix.rs`
  - 内容: `batch_dot()`、`batch_add()`、`batch_scale()` 实现
  - 测试: `test_batch_dot_basic`, `test_batch_add_broadcast`, `test_batch_scale_basic`
  - 前置: T2, T3
  - 预计: 10 min

### 6.6 FFI 辅助与 SIMD

- [ ] **T16: blas_mat_params FFI 辅助函数**
  - 文件: `src/ops/matrix.rs`
  - 内容: `blas_mat_params()` 和 `blas_vec_params()` FFI 辅助函数
  - 测试: `test_blas_mat_params_f_contiguous`, `test_blas_mat_params_c_contiguous`, `test_blas_mat_params_non_contiguous`
  - 前置: T1
  - 预计: 10 min

- [ ] **T17: SIMD 路径 — inner 加速** (feature-gated)
  - 文件: `src/ops/matrix.rs`（`#[cfg(feature = "simd")]` mod）
  - 内容: `inner_simd()` f32/f64 SIMD 实现
  - 测试: `test_inner_simd_f64`, `test_inner_simd_f32`, `test_inner_simd_vs_scalar`
  - 前置: T3, `backend::simd` 模块完成
  - 预计: 10 min

- [ ] **T18: SIMD 路径 — matvec / matmul 加速** (feature-gated)
  - 文件: `src/ops/matrix.rs`
  - 内容: `matvec_f_order_simd()` 和 matmul 内层循环 SIMD 化
  - 测试: `test_matvec_simd_vs_scalar`, `test_matmul_simd_vs_scalar`
  - 前置: T5, T6, T17
  - 预计: 10 min

### 6.7 集成

- [ ] **T19: ops/mod.rs re-export + lib.rs 集成**
  - 文件: `src/ops/mod.rs`, `src/lib.rs`
  - 内容: `pub mod matrix;` 及 re-export 所有公共函数
  - 测试: `test_public_api_import_matrix_ops`
  - 前置: T1-T18
  - 预计: 5 min

---

## 7. 测试计划

### 7.1 单元测试

位于 `src/ops/matrix.rs` 中的 `#[cfg(test)] mod tests`：

#### inner（向量内积）

| 测试函数 | 验证内容 |
|---------|---------|
| `test_inner_basic` | `[1, 2, 3] · [4, 5, 6] = 32` |
| `test_inner_unit_vectors` | `[1, 0, 0] · [0, 1, 0] = 0` |
| `test_inner_single_element` | `[5] · [3] = 15` |
| `test_inner_zero_length` | 空向量内积 = 0（若允许）或返回错误 |
| `test_inner_shape_mismatch` | 长度不同时返回 `ShapeMismatch` |
| `test_inner_float_precision` | f64 精度验证（`assert_close!`，tol = 1e-15） |
| `test_inner_integer` | `[1, 2, 3] · [4, 5, 6] = 32`（i32） |

#### outer（外积）

| 测试函数 | 验证内容 |
|---------|---------|
| `test_outer_2x3` | `[1, 2] × [3, 4, 5]` 的每个元素正确 |
| `test_outer_1x1` | 单元素外积 |
| `test_outer_result_shape` | 输出形状为 `[a.len(), b.len()]` |
| `test_outer_f_contiguous` | 结果为 F-contiguous |

#### matvec（矩阵-向量乘法）

| 测试函数 | 验证内容 |
|---------|---------|
| `test_matvec_f_contiguous` | F-contiguous 矩阵 × 向量结果正确 |
| `test_matvec_c_contiguous` | C-contiguous 矩阵 × 向量结果正确 |
| `test_matvec_non_contiguous` | 非连续矩阵（转置视图）× 向量结果正确 |
| `test_matvec_identity` | 单位矩阵 × 向量 = 原向量 |
| `test_matvec_zero_matrix` | 零矩阵 × 向量 = 零向量 |
| `test_matvec_shape_mismatch` | 列数 ≠ 向量长度时返回错误 |
| `test_matvec_1x1` | 标量矩阵 × 单元素向量 |
| `test_matvec_result_layout` | 输出为 F-contiguous 1D tensor |

#### matmul（矩阵-矩阵乘法）

| 测试函数 | 验证内容 |
|---------|---------|
| `test_matmul_f_contiguous` | F×F 结果正确 |
| `test_matmul_c_contiguous` | C×C 结果正确 |
| `test_matmul_mixed_layout` | F×C 和 C×F 结果正确 |
| `test_matmul_identity` | A × I = A, I × A = A |
| `test_matmul_associative` | `(AB)C ≈ A(BC)`（浮点近似） |
| `test_matmul_shape_mismatch` | K 维不匹配时返回错误 |
| `test_matmul_1x1` | 1×1 矩阵乘法 |
| `test_matmul_result_layout` | 输出为 F-contiguous 2D tensor |

#### dot（通用点积）

| 测试函数 | 验证内容 |
|---------|---------|
| `test_dot_1d_1d` | 等价于 `inner` |
| `test_dot_2d_1d` | 等价于 `matvec` |
| `test_dot_1d_2d` | 向量-矩阵乘结果正确 |
| `test_dot_2d_2d` | 等价于 `matmul` |
| `test_dot_invalid_dims` | 3D × 2D 返回 `InvalidOperation` |

#### tensordot

| 测试函数 | 验证内容 |
|---------|---------|
| `test_tensordot_matrix_multiply` | 等价于 matmul |
| `test_tensordot_inner_product` | `AxesArg::Count(1)` 对 1D × 1D |
| `test_tensordot_3d_2d` | `[3,4,5] × [5,2]` → `[3,4,2]` |
| `test_tensordot_pair_axes` | `AxesArg::Pair` 指定任意轴 |
| `test_tensordot_all_axes` | 缩并所有轴 → 标量 |
| `test_tensordot_shape_mismatch` | 缩并轴长度不匹配 |

#### einsum

| 测试函数 | 验证内容 |
|---------|---------|
| `test_einsum_matmul` | `"ij,jk->ik"` 等价于 matmul |
| `test_einsum_inner` | `"i,i->"` 等价于 inner |
| `test_einsum_outer` | `"i,j->ij"` 等价于 outer |
| `test_einsum_trace` | `"ii->"` 返回迹 |
| `test_einsum_diagonal` | `"ii->i"` 提取对角线 |
| `test_einsum_batch_matmul` | `"bij,bjk->bik"` 批量矩阵乘 |
| `test_einsum_transpose` | `"ij->ji"` 转置 |
| `test_einsum_implicit_output` | 无 `->` 时隐式输出 |
| `test_einsum_invalid_subscript` | 非法下标返回 `InvalidSubscript` |
| `test_einsum_operand_mismatch` | 操作数数量不匹配 |
| `test_einsum_shape_mismatch` | 同标签尺寸不同 |

#### kron（Kronecker 积）

| 测试函数 | 验证内容 |
|---------|---------|
| `test_kron_2x2` | 标准块矩阵结果 |
| `test_kron_1d` | 1D Kronecker = flatten(outer) |
| `test_kron_identity` | 与单位矩阵的 Kronecker 积 |
| `test_kron_3d` | 3D 张量的 Kronecker 积 |
| `test_kron_ndim_mismatch` | 维度数不同返回错误 |

#### batch 运算

| 测试函数 | 验证内容 |
|---------|---------|
| `test_batch_matmul_basic` | `[4, 3, 4] × [4, 4, 5] → [4, 3, 5]` |
| `test_batch_matmul_broadcast` | `[2, 1, 3, 4] × [1, 5, 4, 5] → [2, 5, 3, 5]` |
| `test_batch_matvec_basic` | `[4, 3, 4] × [4, 4] → [4, 3]` |
| `test_batch_matvec_broadcast` | 广播 batch 维 |
| `test_batch_dot_basic` | `[4, 3] × [4, 3] → [4]` |
| `test_batch_add_broadcast` | 广播加法结果正确 |
| `test_batch_scale_basic` | 标量乘法结果正确 |

### 7.2 边界测试

位于 `tests/edge_cases.rs` 或 `src/ops/matrix.rs` 测试模块中：

| 边界场景 | 测试项 |
|----------|--------|
| **空张量** | `test_matvec_empty_matrix`, `test_inner_empty_vectors` |
| **单元素** | `test_matmul_1x1`, `test_inner_single_element` |
| **大张量** | `test_matmul_large_1024x1024`（验证不溢出） |
| **NaN 传播** | `test_matmul_nan_propagation`（输入含 NaN → 输出含 NaN） |
| **Inf 处理** | `test_matmul_inf_overflow`（大数乘法溢出到 Inf） |
| **非连续输入** | `test_matmul_transposed_view`, `test_matvec_sliced_matrix` |
| **高维** | `test_tensordot_5d`, `test_einsum_4d` |
| **零矩阵** | `test_matmul_zero_matrix`, `test_matvec_zero_matrix` |

### 7.3 属性测试

位于 `tests/property/`：

| 不变量 | 测试 |
|--------|------|
| `inner` 对称性 | `inner(a, b) == inner(b, a)` |
| `inner` 线性性 | `inner(s*a, b) == s * inner(a, b)` |
| `matmul` 结合律 | `(AB)C ≈ A(BC)`（浮点容差内） |
| `matvec` 一致性 | `matvec(A, x) == dot(A, x)` |
| `outer` 展平 | `outer(a, b).flatten() == kron(a, b)` |
| `tensordot(n=1) ≡ dot` | 对于 2D 输入，`tensordot(a, b, 1) ≈ dot(a, b)` |
| `einsum ≡ explicit` | `"ij,jk->ik"` 与 `matmul` 结果一致 |
| `kron` 双线性 | `kron(s*a, b) == s * kron(a, b)` |

### 7.4 性能基准测试

位于 `benches/matrix_ops.rs`：

| 基准 | 操作 | 规模 | 对比对象 |
|------|------|------|----------|
| `bench_inner_f64` | 向量内积 | 1024, 1M | ndarray |
| `bench_matvec_f64_f_order` | 矩阵-向量 | 1024×1024 | ndarray |
| `bench_matvec_f64_c_order` | 矩阵-向量 (C) | 1024×1024 | ndarray |
| `bench_matmul_f64_f_order` | 矩阵-矩阵 | 256×256, 1024×1024 | ndarray |
| `bench_matmul_f64_c_order` | 矩阵-矩阵 (C) | 256×256 | ndarray |
| `bench_matmul_f64_non_contiguous` | 转置视图 | 256×256 | ndarray |
| `bench_tensordot` | 3D 缩并 | 64×64×64 | ndarray |
| `bench_batch_matmul` | 批量矩阵乘 | 32×128×128 | ndarray |
| `bench_einsum_simple` | `"ij,jk->ik"` | 256×256 | ndarray |

### 7.5 测试覆盖率目标

| 函数类别 | 目标行覆盖率 |
|----------|-------------|
| `inner`, `outer` | ≥ 95% |
| `matvec`, `matmul` | ≥ 95%（含三条 dispatch 路径） |
| `dot` | ≥ 90%（所有维度组合分派） |
| `tensordot` | ≥ 85% |
| `einsum` | ≥ 80%（解析器 + 执行引擎） |
| `kron` | ≥ 90% |
| `batch_*` | ≥ 85%（含广播路径） |
| SIMD 路径 | ≥ 70%（硬件依赖） |

---

## 附录 A：API 速查表

### 自由函数

| 函数 | 输入维度 | 输出维度 | 元素约束 | 语义 |
|------|----------|----------|----------|------|
| `inner` | Ix1 × Ix1 | Ix0 | Numeric | 向量内积 |
| `outer` | Ix1 × Ix1 | Ix2 | Numeric | 向量外积 |
| `matvec` | Ix2 × Ix1 | Ix1 | Numeric | 矩阵-向量乘 |
| `matmul` | Ix2 × Ix2 | Ix2 | Numeric | 矩阵-矩阵乘 |
| `dot` | IxDyn × IxDyn | IxDyn | Numeric | 通用（分派） |
| `tensordot` | D1 × D2 | IxDyn | Numeric | 通用缩并 |
| `einsum` | &[TensorD] | IxDyn | Numeric | Einstein 求和 |
| `kron` | D1 × D2 | IxDyn | Numeric | Kronecker 积 |
| `batch_matmul` | TensorD × TensorD | IxDyn | Numeric | 批量矩阵乘 |
| `batch_matvec` | TensorD × TensorD | IxDyn | Numeric | 批量矩阵-向量 |
| `batch_dot` | TensorD × TensorD | IxDyn | Numeric | 批量内积 |
| `batch_add` | TensorD × TensorD | IxDyn | Numeric | 批量加法 |
| `batch_scale` | TensorD × A | IxDyn | Numeric | 批量标量乘 |

### FFI 辅助

| 函数 | 输入 | 返回 | 说明 |
|------|------|------|------|
| `blas_mat_params` | &TensorBase<Ix2> | `Option<(*const A, usize, usize, usize, Order, bool)>` | BLAS 矩阵参数提取 |

## 附录 B：Dispatch 路径汇总

| 操作 | 标量路径 | SIMD 路径 | BLAS FFI |
|------|----------|-----------|----------|
| `inner` | ✅ 所有 Numeric | ✅ f32/f64 (simd) | 通过 `blas_mat_params` |
| `outer` | ✅ 所有 Numeric | — | — |
| `matvec` | ✅ F/C/generic | ✅ f32/f64 F-contig (simd) | 通过 `blas_mat_params` |
| `matmul` | ✅ F-F / C-C / mixed | ✅ f32/f64 (simd) | 通过 `blas_mat_params` |
| `tensordot` | ✅ (降级 matmul) | ✅ 继承 matmul | 继承 matmul |
| `einsum` | ✅ (降级 tensordot) | ✅ 继承 tensordot | 继承 tensordot |
| `kron` | ✅ 所有 Numeric | — | — |
| `batch_*` | ✅ (循环 per-batch) | ✅ 继承 per-slice | 继承 per-slice |

## 附录 C：与需求规格的映射

| 需求 §10.2 操作 | 本模块函数 | 备注 |
|-----------------|-----------|------|
| matvec | `matvec`, `batch_matvec` | 需求明确要求 |
| dot / inner | `dot`, `inner` | dot 为通用，inner 为专用 |
| outer | `outer` | — |
| batch_matvec | `batch_matvec` | 需求明确要求 |
| batch_dot | `batch_dot` | 需求明确要求 |
| batch_add | `batch_add` | 需求明确要求 |
| batch_scale | `batch_scale` | 需求明确要求 |
| GEMM | 不包含 | 需求明确排除，由上游 BLAS 通过 FFI 提供 |
| matmul | `matmul`, `batch_matmul` | 简化矩阵乘（无 α/β），非完整 GEMM |
| tensordot | `tensordot` | 通用张量缩并 |
| einsum | `einsum` | Einstein 求和约定 |
| kron | `kron` | Kronecker 积 |
