# SIMD 后端模块设计

> 文档编号: 08 | 模块: `src/simd/` | 阶段: Phase 5
> 前置文档: `03-element-types.md`, `06-memory.md`, `07-tensor.md`
> 需求参考: 需求说明书 §9.1

---

## 1. 模块定位

SIMD 后端模块是 Xenon 张量库的可选性能加速层，通过 `pulp` crate 提供跨平台 SIMD 抽象，为逐元素运算、归约和内积操作提供硬件向量化加速。该模块默认关闭，通过 `features = ["simd"]` 启用。

### 1.1 职责边界

| 职责 | 包含 | 不包含 |
|------|------|--------|
| SIMD 抽象 | 通过 pulp 统一抽象 x86/ARM 指令集 | GPU 加速 (CUDA/OpenCL) |
| 逐元素运算 | 加减乘除、abs、neg 等向量化 | sqrt（通过 pulp 内置方法直接提供，不通过 SimdKernel trait 暴露）、复杂线性代数 (矩阵分解) |
| 归约运算 | sum 的 SIMD 求和与合并 | BLAS 绑定 |
| 内积运算 | dot product 分块 SIMD 计算 | 外积、矩阵乘法 |
| 标量回退 | 所有操作的纯标量基准实现 | — |
| 运行时分发 | Arch 检测缓存、自动最优路径选择 | 编译期静态分发 |

### 1.2 设计原则

| 原则 | 体现 |
|------|------|
| 结果一致性 | SIMD 路径与标量路径数值结果须一致 |
| 自动回退 | 不满足 SIMD 条件时自动回退到标量，用户无需感知 |
| 零成本抽象 | 未启用 `simd` feature 时无任何运行时开销 |
| 跨平台 | pulp 统一 x86_64 (AVX/AVX2/AVX512) 和 ARM (Neon/SVE) |

### 1.3 在架构中的位置

```
依赖层级：
L0: error, private
L1: dimension, element, complex
L2: layout (依赖 dimension)
L3: storage (依赖 layout)
L4: tensor (依赖 storage, dimension)
L5: simd  ← 当前模块（可选，feature = "simd"）
```

### 1.4 性能分层中的角色

```
┌─────────────────────────────────────────────────────────────────┐
│                       调用层 (ops/iter)                          │
│            math, reduction, dot product                         │
│            参见 11-math.md §5, 13-reduction.md §4                │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  性能分层决策 (dispatch)                          │
│    根据 元素数/连续性/对齐/feature 决定执行路径                      │
└──────────┬──────────────┬──────────────┬────────────────────────┘
           │              │              │
           ▼              ▼              ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ SIMD路径  │   │ 并行路径  │   │ 标量路径   │
    │(本模块)   │   │ (rayon)  │   │  (回退)   │
    └────┬─────┘   └────┬─────┘   └────┬─────┘
         │              │              │
         └──────────────┴──────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │   硬件执行       │
              │ AVX-512/AVX2/   │
              │ SSE4.1/NEON     │
              └─────────────────┘
```

---

## 2. 文件位置

```
src/simd/
├── mod.rs             # pulp 集成、Arch 缓存、公开 API、SimdKernel trait
├── scalar.rs          # 标量回退实现（所有操作的基准实现）
└── vector.rs          # 向量化实现（基于 pulp WithSimd）
```

单文件职责划分：`mod.rs` 负责分发和 trait 定义，`scalar.rs` 提供无条件可用的标量基准，`vector.rs` 封装所有 pulp SIMD 逻辑。

---

## 3. 依赖关系

### 3.1 依赖图

```
src/simd/
├── pulp (可选)                # 外部依赖，feature = "simd"
├── crate::tensor             # TensorBase<S, D>, 类型别名
├── crate::storage            # Storage trait, 获取原始切片
├── crate::layout             # LayoutFlags, Order, 对齐检查
└── crate::element            # Element trait, SimdElement
```

### 3.2 依赖精确到类型级

| 来源模块 | 使用的类型/trait |
|----------|-----------------|
| `pulp` | `Arch`, `Simd`, `WithSimd` |
| `tensor` | `TensorBase<S, D>`, `.as_ptr()`, `.as_slice()`（参见 `07-tensor.md §4`） |
| `storage` | `RawStorage`, `Storage`, `.len()`（参见 `05-storage.md §4`） |
| `layout` | `LayoutFlags`, `is_f_contiguous()`, 对齐查询（参见 `06-memory.md §4`） |
| `element` | `SimdElement`, `Element`（参见 `03-element-types.md §4`） |

### 3.3 依赖方向声明

> **依赖方向：单向向上。** `simd/` 仅消费 `tensor`、`storage`、`layout`、`element` 等核心模块，不被它们依赖。`simd/` 模块在未启用 feature 时完全不存在。

---

## 4. 公共 API 设计

### 4.1 Xenon SIMD 约束

```toml
# Cargo.toml
[features]
default = ["std"]
std = []
simd = ["dep:pulp"]

[dependencies]
pulp = { version = "0.18", optional = true }
```

- 默认关闭，通过 `features = ["simd"]` 显式启用
- 启用后 pulp 自动引入，提供跨平台 SIMD 抽象

### 4.2 SimdElement Trait

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
> | 类型 | 原因 |
> |------|------|
> | `bool` | 逻辑运算无 SIMD 意义 |
> | `usize` | 指针宽度依赖平台，SIMD 语义不稳定 |
> | `Complex<T>` | 实虚部交叉操作需要特殊排列，当前版本不支持 |
>
> **复数 SIMD 策略：** `Complex<f32>` 和 `Complex<f64>` 当前使用标量回退路径。
> 未来优化可考虑将实部和虚部分离为独立数组进行 SIMD 处理（structure-of-arrays），
> 或使用专门的复数乘法/加法指令序列。这需要额外的布局转换开销评估。

### 4.3 SimdKernel Trait

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
pub trait SimdKernel<A: SimdElement>: Send + Sync {
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

### 4.4 pulp 集成与 Arch 缓存

```rust
// src/simd/mod.rs

#[cfg(feature = "simd")]
use pulp::Arch;

/// Returns the `Arch` instance for SIMD dispatch.
///
/// - With `std`: cached via `OnceLock` for zero-cost reuse after first call.
/// - Without `std` (no_std): calls `Arch::new()` each time.
///   `Arch` is `Copy` and lightweight, so no caching is needed.
///   `OnceLock` is unavailable in `core` (Rust 1.85).
#[cfg(all(feature = "simd", feature = "std"))]
#[inline]
pub fn get_arch() -> Arch {
    static ARCH: std::sync::OnceLock<Arch> = std::sync::OnceLock::new();
    *ARCH.get_or_init(Arch::new)
}

/// Returns the `Arch` instance (no_std path, no caching).
#[cfg(all(feature = "simd", not(feature = "std")))]
#[inline]
pub fn get_arch() -> Arch {
    Arch::new()
}

/// Placeholder when SIMD is disabled.
#[cfg(not(feature = "simd"))]
pub fn get_arch() -> () {
    ()
}
```

### 4.5 SIMD 加速路径设计

#### 逐元素运算（加减乘除）

```
逐元素运算流程

┌─────────────────────────────────────────────────────────────────┐
│ 输入: lhs[0..N], rhs[0..N], dst[0..N]                           │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 检查 SIMD 条件                                                  │
│ • 内存连续 F-order?                                             │
│ • 元素类型支持? (f32/f64/i32/i64)                               │
│ • 64 字节对齐?                                                  │
│ • feature = "simd" 启用?                                        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼ YES                   ▼ NO
     ┌─────────────────┐     ┌─────────────────┐
     │ SIMD 主循环      │     │ 标量循环        │
     │ width = S::len() │     │ for i in 0..N   │
     │ for chunk:       │     │   dst[i] = op   │
     │   load + op +    │     │     (lhs[i],    │
     │   store          │     │      rhs[i])    │
     ├─────────────────┤     └─────────────────┘
     │ 尾部标量处理      │
     │ [chunks*W..N)    │
     └─────────────────┘
```

#### 归约运算（sum）

```
SIMD sum 归约流程

┌─────────────────────────────────────────────────────────────────┐
│ 输入: data[0..N]                                                │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ SIMD 累加                                                       │
│ acc = splat(0)                                                  │
│ for chunk in data.chunks(W):                                    │
│     vec = load(chunk)                                           │
│     acc = add(acc, vec)                                         │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 水平求和（合并 SIMD 寄存器）                                     │
│ store acc → arr[0..W]                                           │
│ sum += arr[0] + arr[1] + ... + arr[W-1]                         │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 尾部标量累加 [chunks*W..N)                                       │
└─────────────────────────────────────────────────────────────────┘
```

#### 向量内积（dot product）

```
SIMD dot product 流程

┌─────────────────────────────────────────────────────────────────┐
│ 输入: lhs[0..N], rhs[0..N]                                      │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ SIMD 分块累加                                                   │
│ acc = splat(0)                                                  │
│ for i in 0..chunks:                                             │
│     l = load(lhs[i*W..(i+1)*W])                                 │
│     r = load(rhs[i*W..(i+1)*W])                                 │
│     acc = add(acc, mul(l, r))  // FMA: fused multiply-add      │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 水平求和 + 尾部标量处理                                          │
│ result = horizontal_sum(acc) + scalar_tail                      │
└─────────────────────────────────────────────────────────────────┘
```

### 4.6 条件检查与自动回退

```rust
// src/simd/mod.rs

/// Alignment checking functions are defined in `06-memory.md §4.5`.
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
    ptr: *const A,
    len: usize,
    is_contiguous: bool,
) -> bool {
    if !is_contiguous || len < 4 {
        return false;
    }
    is_aligned(ptr as *const u8)
}

#[cfg(not(feature = "simd"))]
pub fn can_use_simd<A: SimdElement>(
    _ptr: *const A,
    _len: usize,
    _is_contiguous: bool,
) -> bool {
    false  // simd feature not enabled, always fall back
}
```

### 4.7 Good/Bad 对比示例

```rust
// Good - automatic fallback, user does not need to be aware of SIMD
use xenon::simd::SimdKernel;

fn add_arrays<A: SimdElement>(
    lhs: &[A], rhs: &[A], dst: &mut [A],
    is_contiguous: bool,
) {
    if can_use_simd(lhs.as_ptr(), lhs.len(), is_contiguous) {
        #[cfg(feature = "simd")]
        {
            let arch = get_arch();
            arch.dispatch(AddKernel { lhs, rhs, dst });
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
    let arch = get_arch();
    arch.dispatch(AddKernel { lhs, rhs, dst });
}
```

---

## 5. 内部实现设计

### 5.1 pulp WithSimd 使用模式

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

### 5.2 标量回退实现

```rust
// src/simd/scalar.rs

use crate::simd::{SimdElement, SimdKernel};

/// Scalar kernel implementation.
///
/// All operations use pure scalar loops without SIMD instructions.
/// Serves as the reference implementation and fallback path.
pub struct ScalarKernel<A: SimdElement> {
    _marker: core::marker::PhantomData<A>,
}

impl<A: SimdElement> ScalarKernel<A> {
    #[inline]
    pub const fn new() -> Self {
        Self { _marker: core::marker::PhantomData }
    }
}

impl<A: SimdElement> Default for ScalarKernel<A> {
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
```

### 5.3 dispatch 流程

```
dispatch 调用流程

┌─────────────────────────────────────────────────────────────────┐
│               arch.dispatch(kernel)                              │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│         Arch 内部运行时检测（已缓存）                             │
│   检查顺序：AVX-512 → AVX2+FMA → SSE4.1 → NEON → 标量          │
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
│         kernel.with_simd(simd: S)                                │
│         S 为 pulp 自动选择的最优指令集类型                        │
└─────────────────────────────────────────────────────────────────┘
```

### 5.4 计算结果一致性保证

```
一致性保证策略

┌─────────────────────────────────────────────────────────────────┐
│ SIMD 路径                        标量路径                       │
│                                                                 │
│ add(a[0:W], b[0:W])             a[0]+b[0], a[1]+b[1], ...     │
│ add(a[W:2W], b[W:2W])           a[W]+b[W], ...                 │
│ ...                             ...                             │
│ tail: a[n]+b[n] (标量)           ...                             │
│                                                                 │
│ 保证：                                                          │
│ • 逐元素操作：SIMD 与标量逐位一致                                │
│ • 归约操作：浮点求和顺序可能不同，允许 ULP 级差异                │
│ • 内积操作：同归约，允许 ULP 级差异                              │
└─────────────────────────────────────────────────────────────────┘
```

> **设计决策：** 对于逐元素运算，SIMD 与标量结果须**逐位一致**。对于归约和内积，由于浮点累加顺序不同，允许 **1-2 ULP** 差异，但必须通过属性测试验证偏差在可接受范围内。
>
> **一致性说明：** 对于逐元素操作（add、mul 等），SIMD 和标量路径产生逐位一致的结果。
> 对于归约/内积操作，浮点结合律不成立，不同求和顺序允许 ≤2 ULP 差异。
> 这是对需求 §28.5"SIMD 结果须与标量路径一致"的公认解释——对于非结合操作，
> "一致"意味着数值误差在可接受的 ULP 范围内。

---

## 6. 实现任务拆分

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
  - 预计: 15 min

- [ ] **T4**: 实现归约与内积 SIMD 路径
  - 文件: `src/simd/vector.rs`
  - 内容: `SumKernel`/`DotKernel` WithSimd 实现（含水平求和）
  - 测试: `test_vector_sum_f64`、`test_vector_dot_f64`
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

## 7. 测试计划

### 7.1 测试分类表

| 类型 | 位置 | 目的 |
|------|------|------|
| 单元测试 | `#[cfg(test)] mod tests` | 验证单个 kernel 正确性 |
| 一致性测试 | `src/simd/tests.rs` | SIMD 与标量结果一致性 |
| 边界测试 | 集成测试中标注 | 空数组、单元素、非对齐 |
| 属性测试 | `tests/property/` | 随机数据验证不变量 |

### 7.2 单元测试清单

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_scalar_add_f64` | 标量加法基本正确性 | 高 |
| `test_scalar_sum_f64` | 标量求和基本正确性 | 高 |
| `test_scalar_dot_f64` | 标量内积基本正确性 | 高 |
| `test_vector_add_f32` | SIMD f32 加法正确性 | 高 |
| `test_vector_add_f64` | SIMD f64 加法正确性 | 高 |
| `test_vector_sum_f64` | SIMD 归约求和正确性 | 高 |
| `test_vector_dot_f64` | SIMD 内积正确性 | 高 |
| `test_simd_scalar_consistency` | SIMD 与标量结果一致 | 高 |
| `test_tail_handling` | 非宽度整数倍数组尾部处理 | 中 |
| `test_empty_array` | 空数组不 panic | 中 |
| `test_single_element` | 单元素数组正确处理 | 中 |
| `test_misaligned_ptr` | 非对齐数据回退到标量 | 中 |

### 7.3 边界测试场景表

| 场景 | 预期行为 |
|------|----------|
| 空数组 `len=0` | 立即返回，不 panic |
| 单元素 `len=1` | 回退到标量路径 |
| 短数组 `len < SIMD_WIDTH` | 回退到标量路径 |
| 非对齐数据 | 回退到标量或非对齐加载路径 |
| 非 F-order 连续 | 回退到标量路径 |
| `len = SIMD_WIDTH` | 恰好一个 SIMD 块，无尾部 |
| `len = SIMD_WIDTH + 1` | 一个 SIMD 块 + 1 个标量尾部 |

### 7.4 属性测试不变量

| 不变量 | 测试方法 |
|--------|----------|
| SIMD add(a, b) == scalar add(a, b) 逐元素一致 | 随机 `[f64; 0..1024]` |
| SIMD sum(a) ≈ scalar sum(a)（ULP ≤ 2） | 随机 `[f64; 1..8192]` |
| SIMD dot(a, b) ≈ scalar dot(a, b)（ULP ≤ 2） | 随机 `[f64; 1..8192]` |
| tail 处理正确 | `len = n * width + k`, k ∈ [0, width) |

---

## 8. 与其他模块的交互

### 8.1 与 ops 模块

`ops/` 模块在执行逐元素运算时，调用 `simd::can_use_simd()` 检查条件，满足时使用 `VectorKernel`，否则使用 `ScalarKernel`（参见 `11-math.md §5.3`）。

### 8.2 与 parallel 模块

并行路径的每个工作线程内部可以使用 SIMD。组合使用时：先按并行阈值分块到各线程，线程内部再检查 SIMD 条件执行向量化（参见 `09-parallel.md §8.1`）。

### 8.3 与 storage/layout 模块

SIMD 模块依赖 layout 提供的连续性和对齐信息来判断是否可以使用 SIMD 路径（参见 `06-memory.md §4.5`）。

---

## 9. 设计决策记录（ADR）

### 决策 1：选择 pulp 作为 SIMD 抽象层

| 属性 | 值 |
|------|-----|
| 决策 | 使用 pulp crate 作为 SIMD 抽象层 |
| 理由 | 跨平台统一抽象（x86/ARM）、运行时自动检测（`Arch::new()`）、`WithSimd` trait 模式代码简洁、`no_std` 兼容、维护成本低 |
| 替代方案 | `std::arch` 直接使用 — 放弃，需手动处理平台差异和大量 `cfg` 条件编译 |
| 替代方案 | `core::simd` (portable SIMD) — 放弃，仍为 nightly 特性，不稳定 |
| 替代方案 | `packed_simd2` — 放弃，项目不再活跃维护 |

### 决策 2：SIMD 与标量结果一致性策略

| 属性 | 值 |
|------|-----|
| 决策 | 逐元素运算须逐位一致；归约/内积允许 ULP 级差异 |
| 理由 | 逐元素运算的 SIMD 实现与标量在数学上等价，应完全一致。归约操作因浮点累加顺序不同，不可避免产生舍入差异，控制在 ULP 级别即可 |
| 替代方案 | 所有操作要求逐位一致 — 放弃，归约操作无法保证（浮点非结合律） |
| 替代方案 | 允许任意差异 — 放弃，无法保证正确性 |

### 决策 3：默认关闭 SIMD feature

| 属性 | 值 |
|------|-----|
| 决策 | SIMD 默认关闭，通过 `features = ["simd"]` 启用 |
| 理由 | 最小依赖原则；pulp 在某些平台上可能引入编译问题；用户可按需启用 |
| 替代方案 | 默认启用 — 放弃，违反最小依赖原则 |

---

## 10. 性能考量

### 10.1 支持的指令集与性能特征

| 指令集 | 架构 | 寄存器宽度 | f32 元素数 | f64 元素数 | 优先级 |
|--------|------|-----------|-----------|-----------|--------|
| AVX-512 | x86_64 | 512 bit | 16 | 8 | 最高 |
| AVX2 + FMA | x86_64 | 256 bit | 8 | 4 | 高 |
| SSE4.1 | x86_64 | 128 bit | 4 | 2 | 中 |
| NEON | aarch64 | 128 bit | 4 | 2 | 高 (ARM) |

### 10.2 性能数据（预期）

| 操作 | 标量 | AVX2 (f64) | 加速比 | 标量 | AVX2 (f32) | 加速比 |
|------|------|-----------|--------|------|-----------|--------|
| add (1M 元素) | ~2ms | ~0.5ms | ~4x | ~2ms | ~0.25ms | ~8x |
| sum (1M 元素) | ~1ms | ~0.3ms | ~3.3x | ~1ms | ~0.15ms | ~6.7x |
| dot (1M 元素) | ~2ms | ~0.5ms | ~4x | ~2ms | ~0.3ms | ~6.7x |

### 10.3 性能影响因素

| 方面 | 设计决策 |
|------|----------|
| 向量化宽度 | pulp 运行时自动选择最优宽度，无需编译期配置 |
| 内存对齐 | 64 字节对齐时使用对齐加载，否则使用非对齐加载 |
| 尾部处理 | 标量循环处理尾部，简单安全 |
| 循环展开 | pulp 内部处理，无需手动展开 |
| FMA 利用 | pulp 自动使用 FMA 指令（如可用）进行乘加融合 |

---

## 11. no_std 兼容性

pulp crate 支持 `no_std` 环境。在 `no_std` 环境下：

| 环境 | pulp 支持 | 说明 |
|------|----------|------|
| `std` | ✅ 完整支持 | 默认 |
| `no_std` + `alloc` | ✅ 支持 | pulp 在 no_std 下自动适配，Xenon 不引入额外依赖 |
| `no_std` 无 `alloc` | ⚠️ 部分支持 | Arch 检测可能受限 |

条件编译处理：

```rust
#[cfg(all(feature = "simd", not(feature = "std")))]
// In no_std environments:
// - pulp handles math functions internally without Xenon adding libm as a dependency
// - Arch is not cached (OnceLock unavailable in core),
//   Arch::new() is called per-use (Arch is Copy, negligible cost)
```

---

## 版本历史

| 版本 | 日期 |
|------|------|
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-07 |
| 1.0.2 | 2026-04-08 |
| 1.1.0 | 2026-04-08 |

---

*本文档由 Xenon 维护。如有问题请提交 Issue 或 PR。*
