# SIMD 后端模块设计

> 文档编号: 08 | 模块: `src/simd/` | 阶段: Phase 5
> 前置文档: `03-element.md`, `06-memory.md`, `07-tensor.md`
> 需求参考: 需求说明书 §9.1
> 范围声明: 范围内

---

## 1. 模块定位

SIMD 后端模块是 Xenon 张量库的可选性能加速层，通过 `pulp` crate 提供跨平台 SIMD 抽象。当前版本 SIMD 仅覆盖逐元素算术/一元运算和整数归约；数学函数、浮点归约与内积统一回退标量。该模块默认关闭，通过 `features = ["simd"]` 启用。

### 1.1 职责边界

| 职责       | 包含                                    | 不包含                                           |
| ---------- | --------------------------------------- | ------------------------------------------------ |
| SIMD 抽象  | 通过 pulp 统一抽象 x86/ARM 指令集       | GPU 加速 (CUDA/OpenCL)                           |
| 逐元素运算 | 加减乘除、abs、neg 等已覆盖操作的向量化 | `sqrt` 等数学函数、复杂线性代数 (矩阵分解)       |
| 归约运算   | 整数 sum 的 SIMD 求和与合并             | 浮点/复数归约的 SIMD kernel                      |
| 内积运算   | 内积统一经由本模块分发并在当前版本回退标量 | 外积、矩阵乘法                                 |
| 标量回退   | 所有操作的纯标量基准实现                | —                                                |
| 运行时分发 | Arch 检测缓存、自动最优路径选择         | 编译期静态分发                                   |

### 1.2 设计原则

| 原则       | 体现                                                 |
| ---------- | ---------------------------------------------------- |
| 结果一致性 | SIMD 路径与标量路径结果须与标量实现保持一致          |
| 自动回退   | 不满足 SIMD 条件时自动回退到标量，用户无需感知       |
| 零成本抽象 | 未启用 `simd` feature 时无任何运行时开销             |
| 跨平台     | pulp 统一 x86_64 (AVX/AVX2/AVX512) 和 ARM (Neon/SVE) |

### 1.3 在架构中的位置

```
Dependency levels:
L0: error, private
L1: dimension, element, complex
L2: layout (depends on dimension)
L3: storage (depends only on core/alloc, not layout)
L4: tensor (depends on storage, dimension)
L5: simd  <- current module (optional, feature = "simd")
```

### 1.4 性能分层中的角色

```
┌─────────────────────────────────────────────────────────────────┐
│                    Call layer (overload/iter)                   │
│                 math, reduction, matrix                         │
│   See 11-math.md §5, 13-reduction.md §5, 12-matrix.md §5        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│               Performance dispatch decision                     │
│    Choose path by len/contiguity/alignment/feature             │
└──────────┬──────────────┬──────────────┬────────────────────────┘
           │              │              │
           ▼              ▼              ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
     │ SIMD path│   │ Par path │   │Scalar path│
     │(this mod)│   │ (rayon)  │   │(fallback) │
    └────┬─────┘   └────┬─────┘   └────┬─────┘
         │              │              │
         └──────────────┴──────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │ Hardware exec   │
              │ AVX-512/AVX2/   │
              │ SSE4.1/NEON     │
              └─────────────────┘
```

---

## 2. 需求映射与范围约束

| 类型     | 内容                                                                                 |
| -------- | ------------------------------------------------------------------------------------ |
| 需求映射 | `需求说明书 §9.1`, `§9.3`                                                            |
| 范围内   | Current version SIMD covers only element-wise arithmetic/unary ops and integer reduction; auto-fallback |
| 范围外   | SIMD for math functions (sin/sqrt/exp/ln/floor/ceil) and float/complex reduction or any dot kernel in current version |
| 非目标   | No new third-party dependencies beyond pulp                                          |

---

## 3. 文件位置

```
src/simd/
├── mod.rs             # pulp 集成、Arch 缓存、公开 API、SimdKernel trait
├── scalar.rs          # 标量回退实现（所有操作的基准实现）
└── vector.rs          # 向量化实现（基于 pulp WithSimd）
```

单文件职责划分：`mod.rs` 负责分发和 trait 定义，`scalar.rs` 提供无条件可用的标量基准，`vector.rs` 封装所有 pulp SIMD 逻辑。

---

## 4. 依赖关系

### 4.1 依赖图

```
src/simd/
├── pulp (optional)           # External dependency, feature = "simd"
├── crate::tensor             # TensorBase<S, D>, type aliases
├── crate::storage            # Storage trait, raw slice access
├── crate::tensor             # Contiguity/layout queries via TensorBase
└── crate::element            # Element trait, SimdElement
```

### 4.2 依赖精确到类型级

| 来源模块  | 使用的类型/trait                                                                   |
| --------- | ---------------------------------------------------------------------------------- |
| `pulp`    | `Arch`, `Simd`, `WithSimd`                                                         |
| `tensor`  | `TensorBase<S, D>`, `.as_ptr()`, `.as_slice()`（参见 `07-tensor.md` §5.4 / §5.4a） |
| `storage` | `RawStorage`, `Storage`, `.len()`（参见 `05-storage.md` §5）                       |
| `tensor`  | `.is_f_contiguous()`, 布局标志查询（参见 `07-tensor.md` §5）                       |
| `element` | `Element`（参见 `03-element.md` §5.1）                                             |
| `simd`    | `SimdElement`（本模块定义，见 §5.2）                                               |

### 4.3 依赖方向声明

> **依赖方向：单向向上。** `simd/` 仅消费 `tensor`、`storage`、`element` 等核心模块，不被它们依赖。布局/连续性判断经由 `TensorBase` 暴露的查询接口完成；`simd/` 模块在未启用 feature 时完全不存在。

### 4.4 依赖合法性与新增依赖说明

| 项目           | 说明                               |
| -------------- | ---------------------------------- |
| 新增第三方依赖 | `pulp`                             |
| 合法性结论     | 符合需求说明书最小依赖限制         |
| 替代方案       | 不适用                             |

---

## 5. 公共 API 设计

### 5.1 Xenon SIMD 约束

```toml
# Cargo.toml
[features]
default = ["std"]
std = []
simd = ["dep:pulp", "std"]

[dependencies]
pulp = { version = "0.18", optional = true }
```

- 默认关闭，通过 `features = ["simd"]` 显式启用
- 启用后 pulp 自动引入，提供跨平台 SIMD 抽象

### 5.2 SimdElement Trait

```rust
// src/simd/mod.rs

/// Marker trait for SIMD kernel element types.
///
/// Distinguishes different element types at compile time for SIMD dispatch.
/// Only implemented for numeric types that support SIMD operations.
pub trait SimdElement: Copy + Clone + Send + Sync + 'static {
    /// Element size in bytes.
    const SIZE: usize;

    /// Natural alignment of the element.
    const ALIGN: usize;
}

impl SimdElement for f32 {
    const SIZE: usize = 4;
    const ALIGN: usize = 4;
}

impl SimdElement for f64 {
    const SIZE: usize = 8;
    const ALIGN: usize = 8;
}

impl SimdElement for i32 {
    const SIZE: usize = 4;
    const ALIGN: usize = 4;
}

impl SimdElement for i64 {
    const SIZE: usize = 8;
    const ALIGN: usize = 8;
}
```

> **不实现 SimdElement 的类型**：以下元素类型不支持 SIMD 加速，始终回退到标量路径：
>
> | 类型         | 原因                                       |
> | ------------ | ------------------------------------------ |
> | `bool`       | 逻辑运算无 SIMD 意义                       |
> | `usize`      | 指针宽度依赖平台，SIMD 语义不稳定          |
> | `Complex<T>` | 实虚部交叉操作需要特殊排列，当前版本不支持 |
>
> **复数 SIMD 策略：** `Complex<f32>` 和 `Complex<f64>` 当前使用标量回退路径。
> 未来优化可考虑将实部和虚部分离为独立数组进行 SIMD 处理（structure-of-arrays），
> 或使用专门的复数乘法/加法指令序列。这需要额外的布局转换开销评估。

### 5.3 SimdKernel Trait

```rust
// src/simd/mod.rs

/// SIMD kernel trait.
///
/// Defines the unified interface for all SIMD operations. Both scalar fallback
/// and vectorized implementations implement this trait; callers use it generically.
///
/// # Type Parameters
///
/// * `A` - Element type, must implement `SimdElement`
pub trait SimdKernel<A: Copy + Send + Sync + 'static>: Send + Sync {
    // ========================================
    // Metadata
    // ========================================

    /// Kernel name (for debugging and logging).
    fn name() -> &'static str where Self: Sized;

    /// Returns the SIMD width (number of elements) for this kernel.
    ///
    /// Scalar kernels return 1; vectorized kernels return the actual SIMD width.
    fn width() -> usize where Self: Sized;

    // ========================================
    // Element-wise binary operations
    // ========================================

    /// Element-wise addition.
    ///
    /// # Panics
    ///
    /// Panics if the three slices have different lengths.
    fn add(&self, lhs: &[A], rhs: &[A], dst: &mut [A]);

    /// Element-wise subtraction.
    fn sub(&self, lhs: &[A], rhs: &[A], dst: &mut [A]);

    /// Element-wise multiplication.
    fn mul(&self, lhs: &[A], rhs: &[A], dst: &mut [A]);

    /// Element-wise division.
    fn div(&self, lhs: &[A], rhs: &[A], dst: &mut [A]);

    // ========================================
    // Element-wise unary operations
    // ========================================

    /// Element-wise negation.
    fn neg(&self, src: &[A], dst: &mut [A]);

    /// Element-wise absolute value.
    fn abs(&self, src: &[A], dst: &mut [A]);

    // ========================================
    // Reduction operations
    // ========================================

    /// Sum reduction.
    fn sum(&self, data: &[A]) -> A;

    // ========================================
    // Dot product operations
    // ========================================

    /// Vector dot product.
    ///
    /// # Panics
    ///
    /// Panics if the two slices have different lengths.
    fn dot(&self, lhs: &[A], rhs: &[A]) -> A;

    // ========================================
    // Fill operations
    // ========================================

    /// Fill with the specified value.
    fn fill(&self, dst: &mut [A], value: A);

    /// Fill with zero.
    fn fill_zero(&self, dst: &mut [A]) where A: Default;
}
```

### 5.4 pulp 集成与 Arch 缓存

```rust
// src/simd/mod.rs

#[cfg(feature = "simd")]
use pulp::Arch;

/// Returns the `Arch` instance for SIMD dispatch.
///
/// Cached via `OnceLock` for zero-cost reuse after first call.
#[cfg(feature = "simd")]
#[inline]
pub fn get_arch() -> Arch {
    static ARCH: std::sync::OnceLock<Arch> = std::sync::OnceLock::new();
    *ARCH.get_or_init(Arch::new)
}

/// Placeholder when SIMD is disabled.
#[cfg(not(feature = "simd"))]
pub fn get_arch() -> () {
    ()
}

/// Stable slice-based dispatch facade used by upper semantic modules.
///
/// `math/`, `matrix/`, and `reduction/` must call this facade only after
/// extracting compatible contiguous slices, rather than passing tensor/view
/// objects directly. The facade may choose a vector kernel or fall back to
/// `ScalarKernel` based on contiguity, alignment, and the operation-specific
/// consistency contract.
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

pub fn dispatch_binary_op<A>(op: BinaryOp, lhs: &[A], rhs: &[A], dst: &mut [A])
where
    A: SimdElement + Numeric + Copy;
```

### 5.4a 当前版本的 SIMD 覆盖范围

当前版本 SIMD 仅覆盖逐元素算术/一元运算和整数归约；数学函数（如 sin/cos/exp/log）和浮点归约/内积均回退标量实现。**当前版本不提供 SIMD 加速的 dot kernel，也不提供浮点/复数 reduction kernel。** 以下需求说明书 §12 / §13 中的操作在当前版本**不提供专门的 SIMD kernel**，即使启用了 `simd` feature，也统一回退到标量路径：

| 类别      | 标量回退操作                                      |
| --------- | ------------------------------------------------- |
| 一元/比较 | `square`, `signum`, `eq`, `ne`, `lt`, `gt`, `not` |
| 复数      | `conj`, `norm`                                    |
| 数学函数  | `sin`, `sqrt`, `exp`, `ln`, `floor`, `ceil`       |

该约束不改变公开 API 的可用性；SIMD 仅影响执行路径选择，不改变结果语义。

### 5.5 SIMD 加速路径设计

#### 逐元素运算（加减乘除）

```
Element-wise operation flow

┌─────────────────────────────────────────────────────────────────┐
│ Input: lhs[0..N], rhs[0..N], dst[0..N]                          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ Check SIMD conditions                                           │
│ • F-order contiguous memory?                                    │
│ • Supported element type? (f32/f64/i32/i64)                     │
│ • 64-byte aligned?                                              │
│ • feature = "simd" enabled?                                    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼ YES                   ▼ NO
     ┌─────────────────┐     ┌─────────────────┐
      │ SIMD main loop   │     │ Scalar loop     │
     │ width = S::len() │     │ for i in 0..N   │
     │ for chunk:       │     │   dst[i] = op   │
     │   load + op +    │     │     (lhs[i],    │
     │   store          │     │      rhs[i])    │
      ├─────────────────┤     └─────────────────┘
      │ Scalar tail      │
     │ [chunks*W..N)    │
     └─────────────────┘
```

#### 归约运算（sum）

```
SIMD sum reduction flow

┌─────────────────────────────────────────────────────────────────┐
│ Input: data[0..N]                                               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ SIMD accumulation                                               │
│ acc = splat(0)                                                  │
│ for chunk in data.chunks(W):                                    │
│     vec = load(chunk)                                           │
│     acc = add(acc, vec)                                         │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ Horizontal sum (merge SIMD lanes)                               │
│ store acc → arr[0..W]                                           │
│ sum += arr[0] + arr[1] + ... + arr[W-1]                         │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ Scalar tail accumulation [chunks*W..N)                          │
└─────────────────────────────────────────────────────────────────┘
```

#### 向量内积（dot product，当前版本回退标量）

```
Current-version dot dispatch flow

┌─────────────────────────────────────────────────────────────────┐
│ Input: lhs[0..N], rhs[0..N]                                     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ Check current-version coverage                                  │
│ dot kernel is not enabled in current version                    │
│ dispatch to scalar dot implementation                           │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ Scalar accumulation over all elements                           │
│ result = scalar_dot(lhs, rhs)                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 5.6 条件检查与自动回退

```rust
// src/simd/mod.rs

/// Alignment checking functions are defined in `06-memory.md §5.5`.
/// SIMD paths use `layout::is_aligned()` from the `layout` module.
use crate::layout::is_aligned;

/// Checks whether SIMD conditions are met.
///
/// Returns `true` when conditions are satisfied; otherwise automatically
/// falls back to the scalar path.
///
/// # SIMD Conditions
///
/// 1. `simd` feature is enabled
/// 2. Memory is contiguous (F-order)
/// 3. Element type supports SIMD (f32/f64/i32/i64)
/// 4. Data is 64-byte aligned
#[cfg(feature = "simd")]
pub fn can_use_simd<A: SimdElement>(
    lhs_ptr: *const A,
    rhs_ptr: *const A,
    dst_ptr: *const A,
    len: usize,
    is_contiguous: bool,
) -> bool {
    if !is_contiguous || len < simd_width() {
        return false;
    }
    is_aligned(lhs_ptr as *const u8)
        && is_aligned(rhs_ptr as *const u8)
        && is_aligned(dst_ptr as *const u8)
}

#[cfg(not(feature = "simd"))]
pub fn can_use_simd<A: SimdElement>(
    _lhs_ptr: *const A,
    _rhs_ptr: *const A,
    _dst_ptr: *const A,
    _len: usize,
    _is_contiguous: bool,
) -> bool {
    false  // simd feature not enabled, always fall back
}
```

### 5.7 Good/Bad 对比示例

```rust
// Good - automatic fallback, user does not need to be aware of SIMD
use xenon::simd::SimdKernel;

fn add_arrays<A: SimdElement>(
    lhs: &[A], rhs: &[A], dst: &mut [A],
    is_contiguous: bool,
) {
    if can_use_simd(
        lhs.as_ptr(),
        rhs.as_ptr(),
        dst.as_ptr(),
        lhs.len(),
        is_contiguous,
    ) {
        #[cfg(feature = "simd")]
        {
            dispatch_binary_op(BinaryOp::Add, lhs, rhs, dst);
            return;
        }
    }
    // Fallback to scalar automatically
    let kernel = ScalarKernel::<A>::new();
    kernel.add(lhs, rhs, dst);
}

// Bad - hardcoded SIMD path with no fallback
fn add_arrays_bad<A: SimdElement>(lhs: &[A], rhs: &[A], dst: &mut [A]) {
    // FORBIDDEN: using SIMD without checking conditions
    // May produce incorrect results on non-contiguous or unaligned data
    dispatch_binary_op(BinaryOp::Add, lhs, rhs, dst);
}
```

---

## 6. 内部实现设计

### 6.1 pulp WithSimd 使用模式

以 f32 加法为例，展示完整的 pulp API 使用方式：

```rust
// src/simd/vector.rs

#[cfg(feature = "simd")]
use pulp::{Simd, WithSimd};

/// f32 addition SIMD kernel.
#[cfg(feature = "simd")]
pub struct AddF32Kernel<'a> {
    pub lhs: &'a [f32],
    pub rhs: &'a [f32],
    pub dst: &'a mut [f32],
}

#[cfg(feature = "simd")]
impl WithSimd for AddF32Kernel<'_> {
    type Output = ();

    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let len = self.lhs.len();
        debug_assert_eq!(self.rhs.len(), len);
        debug_assert_eq!(self.dst.len(), len);

        // NOTE: The following method names have been verified against pulp 0.18.x API.
        // If upgrading pulp, verify these names remain stable.
        let width = S::f32s_len();
        let chunks = len / width;

        // SIMD main loop
        for i in 0..chunks {
            let offset = i * width;
            unsafe {
                // SAFETY: offset + width <= chunks * width <= len
                // Aligned load: already verified 64-byte alignment via can_use_simd()
                let lhs_vec = simd.f32s_load(
                    self.lhs.as_ptr().add(offset)
                );
                let rhs_vec = simd.f32s_load(
                    self.rhs.as_ptr().add(offset)
                );
                let result = simd.f32s_add(lhs_vec, rhs_vec);
                simd.f32s_store(
                    self.dst.as_mut_ptr().add(offset),
                    result,
                );
            }
        }

        // Scalar tail handling
        for i in (chunks * width)..len {
            self.dst[i] = self.lhs[i] + self.rhs[i];
        }
    }
}

/// Dispatches f32 addition to the optimal SIMD path.
#[cfg(feature = "simd")]
pub fn add_f32_simd(lhs: &[f32], rhs: &[f32], dst: &mut [f32]) {
    let arch = crate::simd::get_arch();
    arch.dispatch(AddF32Kernel { lhs, rhs, dst });
}
```

### 6.2 标量回退实现

```rust
// src/simd/scalar.rs

use crate::simd::SimdKernel;

/// Scalar kernel implementation.
///
/// All operations use pure scalar loops without SIMD instructions.
/// Serves as the reference implementation and fallback path.
pub struct ScalarKernel<A: Copy + Send + Sync + 'static> {
    _marker: core::marker::PhantomData<A>,
}

impl<A: Copy + Send + Sync + 'static> ScalarKernel<A> {
    #[inline]
    pub const fn new() -> Self {
        Self { _marker: core::marker::PhantomData }
    }
}

impl<A: Copy + Send + Sync + 'static> Default for ScalarKernel<A> {
    fn default() -> Self { Self::new() }
}

impl SimdKernel<f64> for ScalarKernel<f64> {
    fn name() -> &'static str { "scalar_f64" }
    fn width() -> usize { 1 }

    fn add(&self, lhs: &[f64], rhs: &[f64], dst: &mut [f64]) {
        assert_eq!(lhs.len(), rhs.len());
        assert_eq!(lhs.len(), dst.len());
        for i in 0..lhs.len() {
            dst[i] = lhs[i] + rhs[i];
        }
    }

    fn sub(&self, lhs: &[f64], rhs: &[f64], dst: &mut [f64]) {
        assert_eq!(lhs.len(), rhs.len());
        assert_eq!(lhs.len(), dst.len());
        for i in 0..lhs.len() {
            dst[i] = lhs[i] - rhs[i];
        }
    }

    fn mul(&self, lhs: &[f64], rhs: &[f64], dst: &mut [f64]) {
        assert_eq!(lhs.len(), rhs.len());
        assert_eq!(lhs.len(), dst.len());
        for i in 0..lhs.len() {
            dst[i] = lhs[i] * rhs[i];
        }
    }

    fn div(&self, lhs: &[f64], rhs: &[f64], dst: &mut [f64]) {
        assert_eq!(lhs.len(), rhs.len());
        assert_eq!(lhs.len(), dst.len());
        for i in 0..lhs.len() {
            dst[i] = lhs[i] / rhs[i];
        }
    }

    fn neg(&self, src: &[f64], dst: &mut [f64]) {
        assert_eq!(src.len(), dst.len());
        for i in 0..src.len() {
            dst[i] = -src[i];
        }
    }

    fn abs(&self, src: &[f64], dst: &mut [f64]) {
        assert_eq!(src.len(), dst.len());
        for i in 0..src.len() {
            dst[i] = src[i].abs();
        }
    }

    fn sum(&self, data: &[f64]) -> f64 {
        let mut acc = 0.0;
        for &x in data {
            acc += x;
        }
        acc
    }

    fn dot(&self, lhs: &[f64], rhs: &[f64]) -> f64 {
        assert_eq!(lhs.len(), rhs.len());
        let mut acc = 0.0;
        for i in 0..lhs.len() {
            acc += lhs[i] * rhs[i];
        }
        acc
    }

    fn fill(&self, dst: &mut [f64], value: f64) {
        for elem in dst.iter_mut() {
            *elem = value;
        }
    }

    fn fill_zero(&self, dst: &mut [f64]) {
        for elem in dst.iter_mut() {
            *elem = 0.0;
        }
    }
}

// f32, i32, i64 implementations are similar, omitted here
impl SimdKernel<f32> for ScalarKernel<f32> {
    fn name() -> &'static str { "scalar_f32" }
    fn width() -> usize { 1 }
    // ... other methods follow the same pattern as f64
}
```

> **整数标量内核差异：** `i32`/`i64` 的标量内核与 `f64` 的关键差异在于：
> (1) 不提供 `hypot` 等浮点专用运算；(2) 除法使用整数除法（截断向零），
> 溢出语义不同（如 `i32::MIN / -1` 会 panic）；(3) 归约操作对整数类型保证
> 逐位一致（无不结合律问题）。
>
> **整数归约溢出处理：** 需求说明书 §14 明确规定"整数归约溢出时视为不可恢复错误"。
> 整数 `sum` 归约使用 `checked_add` 逐元素累加，溢出时立即 panic（不可恢复错误）。
> 这与 Rust 的 debug 模式行为一致，但即使在 release 模式下也保证检测。
>
> ```rust
> // Integer sum with overflow detection (scalar path)
> impl SimdKernel<i32> for ScalarKernel<i32> {
>     // ... other methods follow the same pattern as f64 ...
>
>     fn sum(&self, data: &[i32]) -> i32 {
>         data.iter().copied().try_fold(0i32, |acc, x| {
>             acc.checked_add(x)
>         }).expect("integer overflow in sum reduction")
>     }
> }
> ```
>
> **SIMD 路径的整数归约溢出：** 向量化路径中，整数元素在 SIMD 寄存器中并行累加，
> 无法在每次加法后检查溢出。采用以下策略：对 `i32` 类型使用 `i64` 宽寄存器累加
> （`i64` 对 `i32` 求和有 2^32 倍的安全余量），水平求和后再检查是否溢出 `i32` 范围；
> `i64` 类型使用 `i128` 累加。若最终结果溢出目标类型范围，panic。尾部标量处理仍使用
> `checked_add`。这一策略在大多数实际场景下足够安全，同时保持 SIMD 性能优势。

### 6.3 dispatch 流程

```
Dispatch call flow

┌─────────────────────────────────────────────────────────────────┐
│               arch.dispatch(kernel)                              │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│         Arch runtime detection (cached)                         │
│   Check order: AVX-512 -> AVX2+FMA -> SSE4.1 -> NEON -> scalar │
└─────────────────────────┬───────────────────────────────────────┘
                          │
           ┌──────────────┼──────────────┐
           │              │              │
           ▼              ▼              ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │AVX-512   │   │AVX2+FMA  │   │SSE4.1/   │
    │Avx512    │   │Avx2      │   │NEON      │
    └────┬─────┘   └────┬─────┘   └────┬─────┘
         │              │              │
         └──────────────┴──────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│         kernel.with_simd(simd: S)                               │
│         S is the best ISA type selected by pulp                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.4 计算结果一致性保证

```
Consistency guarantee strategy

┌─────────────────────────────────────────────────────────────────┐
│ SIMD path                        Scalar path                    │
│                                                                 │
│ add(a[0:W], b[0:W])             a[0]+b[0], a[1]+b[1], ...     │
│ add(a[W:2W], b[W:2W])           a[W]+b[W], ...                 │
│ ...                             ...                             │
│ tail: a[n]+b[n] (scalar)         ...                             │
│                                                                 │
│ Guarantees:                                                     │
│ • Element-wise ops: SIMD and scalar are bitwise identical       │
│ • Reduction ops: use SIMD only when matching scalar semantics   │
│ • Dot ops: use SIMD only when matching scalar semantics         │
└─────────────────────────────────────────────────────────────────┘
```

> **设计决策：** 对于逐元素运算，SIMD 与标量结果须**逐位一致**。对于归约和内积，只有当某个具体 SIMD kernel 能证明与标量路径一致时才可启用；否则必须自动回退到标量路径。当前版本对 `Complex<f32>` / `Complex<f64>` 统一回退标量，对浮点 `sum` 与所有 `dot` 也默认保守回退，直到对应内核具备可验证的一致性证明。
>
> **一致性说明：** 对于逐元素操作（add、mul 等），SIMD 和标量路径产生逐位一致的结果。
> 对于归约/内积操作，Xenon 不接受“近似一致”作为默认语义；若某个 SIMD 内核无法证明与标量路径一致，则不走 SIMD 路径。
>
> **覆盖说明：** 当前版本 SIMD 仅覆盖逐元素算术/一元运算和整数归约；数学函数与浮点归约/内积维持公开 API 不变并自动回退标量路径。

---

## 7. 实现任务拆分

### Wave 1: 基础设施

- [ ] **T1**: 创建 `src/simd/mod.rs` 骨架
  - 文件: `src/simd/mod.rs`
  - 内容: 模块声明、`SimdElement` trait、`SimdKernel` trait 定义、Arch 缓存、`can_use_simd()`
  - 测试: 编译通过
  - 前置: 无
  - 预计: 10 min

- [ ] **T2**: 创建 `src/simd/scalar.rs` 标量回退
  - 文件: `src/simd/scalar.rs`
  - 内容: `ScalarKernel<A>` 结构体及 `SimdKernel<f32>`、`SimdKernel<f64>` 实现
  - 测试: `test_scalar_add_f64`、`test_scalar_sum_f64`
  - 前置: T1
  - 预计: 10 min

### Wave 2: 向量化实现

- [ ] **T3**: 创建 `src/simd/vector.rs` 逐元素操作
  - 文件: `src/simd/vector.rs`
  - 内容: `VectorKernel<A>` 结构体、f32/f64 的 `AddKernel`/`SubKernel`/`MulKernel`/`DivKernel` WithSimd 实现
  - 测试: `test_vector_add_f32`、`test_vector_add_f64`
  - 前置: T1
  - 预计: 10 min

- [ ] **T4**: 实现受限的归约 SIMD 路径与内积标量回退分发
  - 文件: `src/simd/vector.rs`
  - 内容: 仅为已具备一致性证明的整数类型启用 `SumKernel`；当前版本不提供 `DotKernel`，内积统一 dispatch 到 `ScalarKernel`
  - 测试: `test_sum_dispatch_fallback`、`test_dot_dispatch_fallback`
  - 前置: T3
  - 预计: 10 min

### Wave 3: 集成与条件编译

- [ ] **T5**: 实现 feature gate 条件编译
  - 文件: `src/simd/mod.rs`, `Cargo.toml`
  - 内容: `#[cfg(feature = "simd")]` 条件编译、公开 API 导出、与 tensor 模块的集成接口
  - 测试: 不启用 simd 时编译通过且无 pulp 依赖
  - 前置: T2, T3
  - 预计: 10 min

### Wave 4: 测试与验证

- [ ] **T6**: 编写一致性测试
  - 文件: `src/simd/scalar.rs` (#[cfg(test)])
  - 内容: SIMD 与标量结果一致性测试、属性测试（随机数据）
  - 测试: `test_simd_scalar_consistency`
  - 前置: T5
  - 预计: 10 min

```
Wave 1: [T1]
           │
           ▼
Wave 2: [T2] [T3]
           │    │
           │    ▼
           │  [T4]
           │    │
           ▼    ▼
Wave 3:   [T5]
           │
           ▼
Wave 4:   [T6]
```

---

## 8. 测试计划

### 8.1 测试分类表

| 类型       | 位置                     | 目的                   |
| ---------- | ------------------------ | ---------------------- |
| 单元测试   | `#[cfg(test)] mod tests` | 验证单个 kernel 正确性 |
| 一致性测试 | `src/simd/tests.rs`      | SIMD 与标量结果一致性  |
| 边界测试   | 集成测试中标注           | 空数组、单元素、非对齐 |
| 属性测试   | `tests/property/`        | 随机数据验证不变量     |

### 8.2 单元测试清单

| 测试函数                       | 测试内容                              | 优先级 |
| ------------------------------ | ------------------------------------- | ------ |
| `test_scalar_add_f64`          | 标量加法基本正确性                    | 高     |
| `test_scalar_sum_f64`          | 标量求和基本正确性                    | 高     |
| `test_scalar_dot_f64`          | 标量内积基本正确性                    | 高     |
| `test_vector_add_f32`          | SIMD f32 加法正确性                   | 高     |
| `test_vector_add_f64`          | SIMD f64 加法正确性                   | 高     |
| `test_sum_dispatch_fallback`   | 浮点 sum 在未证明一致性时自动回退标量 | 高     |
| `test_dot_dispatch_fallback`   | 浮点 dot 在未证明一致性时自动回退标量 | 高     |
| `test_simd_scalar_consistency` | SIMD 与标量结果一致                   | 高     |
| `test_tail_handling`           | 非宽度整数倍数组尾部处理              | 中     |
| `test_empty_array`             | 空数组不 panic                        | 中     |
| `test_single_element`          | 单元素数组正确处理                    | 中     |
| `test_misaligned_ptr`          | 非对齐数据回退到标量                  | 中     |

### 8.3 边界测试场景

| 场景                      | 预期行为                    |
| ------------------------- | --------------------------- |
| 空数组 `len=0`            | 立即返回，不 panic          |
| 单元素 `len=1`            | 回退到标量路径              |
| 短数组 `len < SIMD_WIDTH` | 回退到标量路径              |
| 非对齐数据                | 回退到标量路径              |
| 非 F-order 连续           | 回退到标量路径              |
| `len = SIMD_WIDTH`        | 恰好一个 SIMD 块，无尾部    |
| `len = SIMD_WIDTH + 1`    | 一个 SIMD 块 + 1 个标量尾部 |

### 8.4 属性测试不变量

| 不变量                                                               | 测试方法                              |
| -------------------------------------------------------------------- | ------------------------------------- |
| SIMD add(a, b) == scalar add(a, b) 逐元素一致                        | 随机 `[f64; 0..1024]`                 |
| SIMD sum(a) == scalar sum(a)（若无法保证则 dispatch 回退标量）       | 随机 `[f64; 1..8192]`                 |
| SIMD dot(a, b) == scalar dot(a, b)（若无法保证则 dispatch 回退标量） | 随机 `[f64; 1..8192]`                 |
| tail 处理正确                                                        | `len = n * width + k`, k ∈ [0, width) |

### 8.5 集成测试

| 测试文件             | 测试内容                                                                                  |
| -------------------- | ----------------------------------------------------------------------------------------- |
| `tests/test_simd.rs` | SIMD dispatch 与 `math`、`reduction`、`matrix`、`layout`、`parallel` 组合路径的端到端验证 |

### 8.6 Feature gate / 配置测试

| 场景                    | 配置方式                         | 预期行为                               |
| ----------------------- | -------------------------------- | -------------------------------------- |
| 默认关闭                | 默认 feature 集                  | 不编译 SIMD 路径，且无 `pulp` 运行时依赖 |
| 显式启用                | `--features simd`                | SIMD API 可用，满足条件时进入 SIMD 路径 |
| 无硬件能力自动回退      | `--features simd` + 非目标 ISA 环境 | 成功编译运行，并自动回退到标量路径     |

### 8.7 类型边界 / 编译期测试

| 测试类型     | 覆盖内容                                                        | 预期结果                         |
| ------------ | --------------------------------------------------------------- | -------------------------------- |
| 类型边界     | `f32`/`f64`/`i32`/`i64` 实现 `SimdElement`                      | 可编译并进入对应 dispatch 分支   |
| 类型边界     | `bool`/`usize`/`Complex<T>` 不实现 `SimdElement`                | 编译期拒绝直接使用 SIMD kernel   |
| 编译期测试   | `#[cfg(feature = "simd")]` / `#[cfg(not(feature = "simd"))]` 两侧 API | 两种 feature 组合均可编译        |
| 编译期测试   | 未覆盖数学函数与浮点/复数 reduction/dot kernel                  | 保持公开 API 可用且落入标量路径  |

---

## 9. 与其他模块的交互

### 9.1 接口约定

`math/` 模块在执行逐元素运算时，调用 `simd::can_use_simd(lhs_ptr, rhs_ptr, dst_ptr, len, is_contiguous)` 检查条件，满足时使用 `VectorKernel`，否则使用 `ScalarKernel`（参见 `11-math.md §5.3`）。

> **分派策略**: `math` 模块提供公共 `sqrt()` API，但在当前版本中 `sqrt()` 不接入 SIMD kernel，而是保持标量路径。用户仅调用 `math::sqrt()`，无需关心底层是否存在加速能力。

### 9.2 数据流描述

```
math/reduction/matrix call acceleration entry
    │
    ├── simd::can_use_simd(lhs_ptr, rhs_ptr, dst_ptr, len, is_contiguous)
    │       ├── Check feature = simd
    │       ├── Check F-order contiguity
    │       ├── Check whether element type implements SimdElement
    │       └── Check 64-byte alignment
    │
    ├── YES -> get_arch().dispatch(VectorKernel)
    │
    └── NO  -> ScalarKernel fallback for element-wise/reduction/dot
```

### 9.3 与 parallel 模块

并行路径的每个工作线程内部可以使用 SIMD。组合使用时：先按并行阈值分块到各线程，线程内部再检查 SIMD 条件执行向量化（参见 `09-parallel.md §9.1`）。

### 9.4 与 storage/layout 模块

SIMD 模块依赖 layout 提供的连续性和对齐信息来判断是否可以使用 SIMD 路径（参见 `06-memory.md` §5.5）。

---

## 10. 错误处理与语义边界

| 类型                 | 说明                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------ |
| Recoverable error    | 无专属 recoverable error；SIMD 不可用、未对齐、类型不支持或一致性无法证明时均自动回退标量路径 |
| Panic                | 切片长度不一致；整数 `sum` / `dot` 溢出；违反 kernel 前置条件的内部 bug               |
| 路径一致性           | SIMD 结果与标量路径保持一致；若某类 kernel 无法证明一致，则不得启用 SIMD               |
| 容差边界             | integer: exact; float: same numerical semantics or documented tolerance per require §28.3 |

---

## 11. 设计决策记录（ADR）

### 决策 1：选择 pulp 作为 SIMD 抽象层

| 属性     | 值                                                                                                    |
| -------- | ----------------------------------------------------------------------------------------------------- |
| 决策     | 使用 pulp crate 作为 SIMD 抽象层                                                                      |
| 理由     | 跨平台统一抽象（x86/ARM）、运行时自动检测（`Arch::new()`）、`WithSimd` trait 模式代码简洁、维护成本低 |
| 替代方案 | `std::arch` 直接使用 — 放弃，需手动处理平台差异和大量 `cfg` 条件编译                                  |
| 替代方案 | `core::simd` (portable SIMD) — 放弃，仍为 nightly 特性，不稳定                                        |
| 替代方案 | `packed_simd2` — 放弃，项目不再活跃维护                                                               |

### 决策 2：SIMD 与标量结果一致性策略

| 属性     | 值                                                                                                                                             |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| 决策     | 逐元素运算须逐位一致；当前版本仅为整数归约在可证明与标量路径一致时启用 SIMD，所有内积与浮点归约均回退标量                                     |
| 理由     | 逐元素运算的 SIMD 实现与标量在数学上等价，应完全一致。当前整数归约可维持该性质；浮点归约和内积在当前版本尚无可验证的一致性证明，因此必须回退标量路径。 |
| 替代方案 | 所有归约/内积一律强行使用 SIMD — 放弃，无法保证与标量路径一致时会破坏 Xenon 的结果语义                                                         |
| 替代方案 | 允许任意差异 — 放弃，无法保证正确性                                                                                                            |

### 决策 3：默认关闭 SIMD feature

| 属性     | 值                                                              |
| -------- | --------------------------------------------------------------- |
| 决策     | SIMD 默认关闭，通过 `features = ["simd"]` 启用                  |
| 理由     | 最小依赖原则；pulp 在某些平台上可能引入编译问题；用户可按需启用 |
| 替代方案 | 默认启用 — 放弃，违反最小依赖原则                               |

---

## 12. 性能考量

### 12.1 支持的指令集与性能特征

| 指令集     | 架构    | 寄存器宽度 | f32 元素数 | f64 元素数 | 优先级   |
| ---------- | ------- | ---------- | ---------- | ---------- | -------- |
| AVX-512    | x86_64  | 512 bit    | 16         | 8          | 最高     |
| AVX2 + FMA | x86_64  | 256 bit    | 8          | 4          | 高       |
| SSE4.1     | x86_64  | 128 bit    | 4          | 2          | 中       |
| NEON       | aarch64 | 128 bit    | 4          | 2          | 高 (ARM) |

### 12.2 性能数据（目标值，非当前默认路径）

> 说明：下表中的 `sum` / `dot` SIMD 数据仅表示“在未来具备一致性证明的内核落地后”的目标值；按照 §11 ADR-2，当前默认实现中浮点 `sum` / `dot` 仍保守回退到标量路径。

| 操作          | 标量 | AVX2 (f64) | 加速比 | 标量 | AVX2 (f32) | 加速比 |
| ------------- | ---- | ---------- | ------ | ---- | ---------- | ------ |
| add (1M 元素) | ~2ms | ~0.5ms     | ~4x    | ~2ms | ~0.25ms    | ~8x    |
| sum (1M 元素) | ~1ms | ~0.3ms     | ~3.3x  | ~1ms | ~0.15ms    | ~6.7x  |
| dot (1M 元素) | ~2ms | ~0.5ms     | ~4x    | ~2ms | ~0.3ms     | ~6.7x  |

### 12.3 性能影响因素

| 方面       | 设计决策                                     |
| ---------- | -------------------------------------------- |
| 向量化宽度 | pulp 运行时自动选择最优宽度，无需编译期配置  |
| 内存对齐   | 64 字节对齐时启用 SIMD，否则回退到标量路径   |
| 尾部处理   | 标量循环处理尾部，简单安全                   |
| 循环展开   | pulp 内部处理，无需手动展开                  |
| FMA 利用   | pulp 自动使用 FMA 指令（如可用）进行乘加融合 |

---

## 13. 平台与工程约束

| 约束       | 说明                      |
| ---------- | ------------------------- |
| `std` only | SIMD 路径依赖 `std` 环境  |
| 单 crate   | 保持单 crate 边界         |
| SemVer     | SIMD API 变更遵循 SemVer  |
| 最小依赖   | 可选依赖 `pulp`，默认关闭 |

---

## 版本历史

| 版本  | 日期       |
| ----- | ---------- |
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-07 |
| 1.0.2 | 2026-04-08 |
| 1.1.0 | 2026-04-08 |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
