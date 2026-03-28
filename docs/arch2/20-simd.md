# SIMD 后端模块设计

> 文档编号: 20 | 模块: `src/backend/simd.rs` | 阶段: Phase 4（后端模块）
> 前置文档: `00-rust-standards.md`, `01-architecture-overview.md`, `03-element-types.md`, `06-layout.md`

---

## 1. 模块定位

SIMD 后端模块是 Xenon 计算加速层的核心，通过 pulp crate 提供跨平台 SIMD 指令集抽象，在运行时自动检测并选择最优指令集（AVX-512 > AVX2 > SSE2 > NEON > 标量回退），为上层运算模块（逐元素运算、归约、矩阵运算等）提供零开销的向量计算能力。

### 核心设计目标

| 目标 | 体现 |
|------|------|
| 运行时分派 | 使用 pulp `Arch::new().dispatch()` 检测最优指令集，单态化后无虚函数调用 |
| 跨平台抽象 | 通过 pulp `Simd` trait 统一 x86（SSE2/AVX2/AVX-512）和 ARM（NEON） |
| 对齐感知 | 64 字节对齐时使用对齐加载，否则使用非对齐加载；非连续内存回退标量 |
| 尾部安全 | `partial_load` / `partial_store` 或标量尾部循环处理余数元素 |
| 零开销 | 所有分派通过泛型单态化 + `#[inline(always)]` 实现，运行时开销仅一次 `is_a_feature_supported!` |
| Feature gate | 全模块受 `#[cfg(feature = "simd")]` 保护，不启用时零编译依赖 |

### 职责边界

| 职责 | 包含 | 不包含 |
|------|------|--------|
| SIMD 操作封装 | 封装 pulp 的向量运算方法 | 不直接使用 `std::arch` intrinsics |
| 运行时分派 | `Arch` 检测 + `WithSimd` 模式 | 不硬编码特定指令集 |
| 对齐/尾部处理 | `as_simd_*` 切片分割 + `partial_*` 操作 | 不改变调用方内存布局 |
| 连续内存判定 | 提供 `should_use_simd()` 判定函数 | 张量布局分析（由上层 `apply_*` 函数负责） |
| 性能阈值 | SIMD 路径阈值常量 | 并行阈值（由 `backend/parallel.rs` 负责） |

---

## 2. 文件位置

```
src/backend/
├── mod.rs              # 后端抽象层入口，re-export + 条件编译
├── scalar.rs           # 标量回退路径
├── simd.rs             # 本模块：SIMD 加速路径（~800 行）
└── parallel.rs         # 并行路径（rayon）
```

在 `src/backend/mod.rs` 中的声明：

```rust
// src/backend/mod.rs

pub mod scalar;

#[cfg(feature = "simd")]
pub mod simd;

#[cfg(feature = "parallel")]
pub mod parallel;

// Re-export: always provide scalar fallback
pub use scalar::*;

// Re-export: SIMD types when feature enabled
#[cfg(feature = "simd")]
pub use simd::*;
```

在 `src/lib.rs` 中的声明：

```rust
pub mod backend;

// Internal use only — backend is consumed by ops/ modules
// No public re-export needed; users never interact with backend directly.
```

---

## 3. 依赖关系

### 3.1 本模块的依赖（上游）

| 依赖 | 来源 | 用途 |
|------|------|------|
| `pulp::{Arch, Simd, WithSimd, NullaryFnOnce}` | `pulp` crate | SIMD 抽象、运行时分派 |
| `pulp::{f32x4, f32x8, f32x16, f64x2, f64x4, f64x8, ...}` | `pulp` crate | 架构相关向量类型 |
| `pulp::{cast, as_arrays}` | `pulp` crate | 安全的位转换和数组分组 |
| `Element`, `Numeric`, `RealScalar` | `crate::element` | 元素类型约束 |
| `Complex` | `crate::complex` | 复数 SIMD 运算 |
| `LayoutFlags` | `crate::layout` | 连续性/对齐标志查询 |
| `core::mem` | `core` | `size_of`, `align_of` |

### 3.2 依赖本模块的下游模块

| 模块 | 使用方式 |
|------|----------|
| `ops/element_wise.rs` | 逐元素运算的 SIMD 路径分派 |
| `ops/reduction.rs` | 归约运算（sum/prod/min/max）的 SIMD 路径 |
| `ops/matrix.rs` | 矩阵-向量乘法、点积的 SIMD 路径 |
| `backend/scalar.rs` | SIMD 不可用时的标量回退 |

### 3.3 依赖关系图

```
pulp crate (Arch, Simd, WithSimd)
    │
    ▼
┌──────────────────────┐
│  backend/simd.rs     │  ← 本模块（Phase 4）
│  (feature = "simd")  │
└──────────┬───────────┘
           │
    ┌──────┼──────────┐
    ▼      ▼          ▼
element  layout    complex
(traits) (flags)   (types)
```

---

## 4. 公共 API 设计

### 4.1 SimdContext — SIMD 执行上下文

SIMD 操作需要一个执行上下文来封装 pulp 的 `Arch` 检测结果。该上下文为 `Copy` 类型，可在库初始化时创建一次并缓存复用。

```rust
/// SIMD execution context wrapping pulp's runtime feature detection.
///
/// `SimdContext` caches the result of `pulp::Arch::new()` so that
/// runtime feature detection (CPUID on x86, HWCAP on ARM) is only
/// performed once. It is `Copy` and cheap to pass by value.
///
/// # Feature Gate
///
/// This type is only available when the `simd` feature is enabled.
///
/// # Example
///
/// ```ignore
/// use xenon::backend::simd::SimdContext;
///
/// let ctx = SimdContext::new();
/// ctx.dispatch_unary_f64(src, dst, |simd, a| simd.neg_f64s(a));
/// ```
#[cfg(feature = "simd")]
#[derive(Debug, Clone, Copy)]
pub struct SimdContext {
    arch: pulp::Arch,
}

#[cfg(feature = "simd")]
impl SimdContext {
    /// Creates a new SIMD context by detecting the best available instruction set.
    ///
    /// Detection order (x86_64): AVX-512 → AVX2 → SSE2 → Scalar
    /// Detection order (aarch64): NEON → Scalar
    ///
    /// # Performance
    ///
    /// This calls CPUID (x86) or reads HWCAP (ARM) once.
    /// Prefer caching the result rather than calling this in hot loops.
    #[inline]
    pub fn new() -> Self {
        Self {
            arch: pulp::Arch::new(),
        }
    }

    /// Returns the detected architecture variant name for diagnostics.
    ///
    /// Useful for logging and benchmark annotations.
    #[inline]
    pub fn arch_name(&self) -> &'static str {
        #[cfg(target_arch = "x86_64")]
        {
            match self.arch {
                pulp::Arch::Scalar => "scalar",
                pulp::Arch::V3(_) => "avx2",
                #[cfg(feature = "nightly")]
                pulp::Arch::V4(_) => "avx512",
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            match self.arch {
                pulp::Arch::Scalar => "scalar",
                pulp::Arch::Neon(_) => "neon",
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            "scalar"
        }
    }
}

#[cfg(feature = "simd")]
impl Default for SimdContext {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}
```

### 4.2 对齐与判定辅助函数

```rust
/// Returns `true` if SIMD acceleration should be used for the given data layout.
///
/// # Conditions
///
/// - Data length must be at least `MIN_SIMD_ELEMENTS` (4 for f64, 8 for f32)
/// - Data must be contiguous (stride == 1, checked via `LayoutFlags::is_contiguous()`)
/// - Data pointer should be aligned (recommended but not required; unaligned loads work)
///
/// # Arguments
///
/// * `len` - Total number of elements.
/// * `flags` - Layout flags from the tensor.
/// * `element_size` - Size of each element in bytes (`size_of::<A>()`).
#[cfg(feature = "simd")]
#[inline]
pub fn should_use_simd(len: usize, flags: LayoutFlags, element_size: usize) -> bool {
    if !flags.is_contiguous() {
        return false;
    }
    // Minimum elements to benefit from SIMD: at least one full vector register.
    // For f32 with 128-bit SSE: 4 elements; for f64 with AVX2: 4 elements.
    // Using a conservative threshold of 4 elements for any type.
    len >= MIN_SIMD_ELEMENTS
}

/// Returns the SIMD lane count for the given element type.
///
/// This is the number of elements that fit in one SIMD vector register
/// at the best available instruction set level.
///
/// # Examples
///
/// | Type | SSE2 (128-bit) | AVX2 (256-bit) | AVX-512 (512-bit) |
/// |------|----------------|----------------|--------------------|
/// | `f32` | 4 | 8 | 16 |
/// | `f64` | 2 | 4 | 8 |
/// | `i32` | 4 | 8 | 16 |
#[cfg(feature = "simd")]
#[inline]
pub const fn simd_lane_count<A>() -> usize
where
    A: Element,
{
    // Conservative: use the minimum lane count across supported instruction sets.
    // The actual dispatch via pulp handles wider registers automatically.
    core::mem::size_of::<A>()
}

/// Minimum number of elements required to justify SIMD dispatch overhead.
///
/// Below this threshold, scalar code is faster due to SIMD setup cost.
#[cfg(feature = "simd")]
pub const MIN_SIMD_ELEMENTS: usize = 4;
```

### 4.3 向量运算 trait — SimdOps

`SimdOps` 是本模块的核心 trait，将 pulp 的 `Simd` trait 方法封装为类型安全的泛型接口，使上层代码不需要直接依赖 pulp 的具体类型。

```rust
/// Trait for SIMD vectorized operations on contiguous element arrays.
///
/// This trait abstracts over element types (f32, f64, Complex<f32>, Complex<f64>)
/// and provides a unified interface for unary, binary, and reduction operations
/// using pulp's SIMD abstractions.
///
/// All methods operate on raw slices (`*const A` / `*mut A`) rather than
/// tensor types, keeping this module decoupled from the tensor layer.
///
/// # Safety
///
/// Methods in this trait require:
/// - Pointers must be valid for the specified number of elements
/// - For aligned operations, pointers must satisfy the alignment requirement
/// - Source and destination must not overlap (unless explicitly allowed)
///
/// # Feature Gate
///
/// Only available with `feature = "simd"`.
#[cfg(feature = "simd")]
pub trait SimdOps<A: Element>: Sized {
    /// The pulp SIMD vector type for element type `A`.
    ///
    /// Examples: `f32x4` (SSE2), `f32x8` (AVX2), `f64x2` (SSE2), `f64x4` (AVX2).
    type Vec: Copy;

    // ── Splat (broadcast scalar to all lanes) ────────────────

    /// Broadcasts a scalar value to all SIMD lanes.
    fn splat(simd: &impl pulp::Simd, value: A) -> Self::Vec;

    // ── Arithmetic operations ────────────────────────────────

    /// Vectorized addition: `a + b` for each lane.
    fn add(simd: &impl pulp::Simd, a: Self::Vec, b: Self::Vec) -> Self::Vec;

    /// Vectorized subtraction: `a - b` for each lane.
    fn sub(simd: &impl pulp::Simd, a: Self::Vec, b: Self::Vec) -> Self::Vec;

    /// Vectorized multiplication: `a * b` for each lane.
    fn mul(simd: &impl pulp::Simd, a: Self::Vec, b: Self::Vec) -> Self::Vec;

    /// Vectorized division: `a / b` for each lane.
    fn div(simd: &impl pulp::Simd, a: Self::Vec, b: Self::Vec) -> Self::Vec;

    /// Vectorized negation: `-a` for each lane.
    fn neg(simd: &impl pulp::Simd, a: Self::Vec) -> Self::Vec;

    // ── Fused multiply-add ───────────────────────────────────

    /// Fused multiply-add: `a * b + c` for each lane.
    ///
    /// Uses hardware FMA when available (AVX2+FMA, AVX-512).
    /// Falls back to `mul + add` on SSE2/NEON.
    fn mul_add(simd: &impl pulp::Simd, a: Self::Vec, b: Self::Vec, c: Self::Vec) -> Self::Vec;

    // ── Math functions (RealScalar only) ─────────────────────

    /// Vectorized absolute value: `|a|` for each lane.
    fn abs(simd: &impl pulp::Simd, a: Self::Vec) -> Self::Vec;

    /// Vectorized square root for each lane.
    fn sqrt(simd: &impl pulp::Simd, a: Self::Vec) -> Self::Vec;

    // ── Load / Store ─────────────────────────────────────────

    /// Loads a SIMD vector from an aligned pointer.
    ///
    /// # Safety
    ///
    /// `ptr` must be aligned to `align_of::<Self::Vec>()` and valid for
    /// `size_of::<Self::Vec>()` bytes.
    unsafe fn load_aligned(ptr: *const A) -> Self::Vec;

    /// Loads a SIMD vector from a potentially unaligned pointer.
    ///
    /// # Safety
    ///
    /// `ptr` must be valid for `size_of::<Self::Vec>()` bytes.
    unsafe fn load_unaligned(ptr: *const A) -> Self::Vec;

    /// Stores a SIMD vector to an aligned pointer.
    ///
    /// # Safety
    ///
    /// `ptr` must be aligned to `align_of::<Self::Vec>()` and valid for
    /// `size_of::<Self::Vec>()` bytes.
    unsafe fn store_aligned(ptr: *mut A, vec: Self::Vec);

    /// Stores a SIMD vector to a potentially unaligned pointer.
    ///
    /// # Safety
    ///
    /// `ptr` must be valid for `size_of::<Self::Vec>()` bytes.
    unsafe fn store_unaligned(ptr: *mut A, vec: Self::Vec);

    // ── Partial load / store (tail handling) ─────────────────

    /// Loads a partial SIMD vector from a slice shorter than the SIMD width.
    ///
    /// Elements beyond `slice.len()` are initialized to zero.
    /// This handles the tail elements when `len % SIMD_WIDTH != 0`.
    fn partial_load(simd: &impl pulp::Simd, slice: &[A]) -> Self::Vec;

    /// Stores a partial SIMD vector to a slice shorter than the SIMD width.
    ///
    /// Only the first `slice.len()` lanes are written.
    fn partial_store(simd: &impl pulp::Simd, slice: &mut [A], vec: Self::Vec);

    // ── Slice splitting ──────────────────────────────────────

    /// Splits a slice into SIMD-aligned chunks and a scalar tail.
    ///
    /// Returns `(&[Self::Vec], &[A])` where the first element contains
    /// complete SIMD vectors and the second contains remaining elements.
    fn as_simd(slice: &[A]) -> (&[Self::Vec], &[A]);

    /// Splits a mutable slice into SIMD-aligned chunks and a scalar tail.
    ///
    /// Returns `(&mut [Self::Vec], &mut [A])`.
    fn as_simd_mut(slice: &mut [A]) -> (&mut [Self::Vec], &mut [A]);

    // ── Reduction ────────────────────────────────────────────

    /// Reduces a SIMD vector to a single scalar by summing all lanes.
    fn reduce_sum(simd: &impl pulp::Simd, vec: Self::Vec) -> A;

    /// Reduces a SIMD vector to a single scalar by multiplying all lanes.
    fn reduce_product(simd: &impl pulp::Simd, vec: Self::Vec) -> A;

    // ── Comparison ───────────────────────────────────────────

    /// Element-wise less-than comparison.
    ///
    /// Returns a mask vector where each lane is all-ones (true) or all-zeros (false).
    fn cmp_lt(simd: &impl pulp::Simd, a: Self::Vec, b: Self::Vec) -> Self::Vec;

    /// Element-wise greater-than comparison.
    fn cmp_gt(simd: &impl pulp::Simd, a: Self::Vec, b: Self::Vec) -> Self::Vec;

    /// Element-wise equality comparison.
    fn cmp_eq(simd: &impl pulp::Simd, a: Self::Vec, b: Self::Vec) -> Self::Vec;
}
```

### 4.4 SimdOps 实现：f32

```rust
#[cfg(feature = "simd")]
impl SimdOps<f32> for f32 {
    type Vec = pulp::f32s;

    #[inline(always)]
    fn splat(simd: &impl pulp::Simd, value: f32) -> Self::Vec {
        simd.splat_f32s(value)
    }

    #[inline(always)]
    fn add(simd: &impl pulp::Simd, a: Self::Vec, b: Self::Vec) -> Self::Vec {
        simd.add_f32s(a, b)
    }

    #[inline(always)]
    fn sub(simd: &impl pulp::Simd, a: Self::Vec, b: Self::Vec) -> Self::Vec {
        simd.sub_f32s(a, b)
    }

    #[inline(always)]
    fn mul(simd: &impl pulp::Simd, a: Self::Vec, b: Self::Vec) -> Self::Vec {
        simd.mul_f32s(a, b)
    }

    #[inline(always)]
    fn div(simd: &impl pulp::Simd, a: Self::Vec, b: Self::Vec) -> Self::Vec {
        simd.div_f32s(a, b)
    }

    #[inline(always)]
    fn neg(simd: &impl pulp::Simd, a: Self::Vec) -> Self::Vec {
        // Negate via XOR with sign bit mask, or use splat(-0.0) + sub
        simd.sub_f32s(simd.splat_f32s(0.0), a)
    }

    #[inline(always)]
    fn mul_add(simd: &impl pulp::Simd, a: Self::Vec, b: Self::Vec, c: Self::Vec) -> Self::Vec {
        simd.mul_add_f32s(a, b, c)
    }

    #[inline(always)]
    fn abs(simd: &impl pulp::Simd, a: Self::Vec) -> Self::Vec {
        simd.abs_f32s(a)
    }

    #[inline(always)]
    fn sqrt(simd: &impl pulp::Simd, a: Self::Vec) -> Self::Vec {
        simd.sqrt_f32s(a)
    }

    #[inline(always)]
    unsafe fn load_aligned(ptr: *const f32) -> Self::Vec {
        // SAFETY: Caller guarantees alignment and validity.
        unsafe { pulp::f32s::from_ptr_aligned(ptr) }
    }

    #[inline(always)]
    unsafe fn load_unaligned(ptr: *const f32) -> Self::Vec {
        // SAFETY: Caller guarantees validity.
        unsafe { pulp::f32s::from_ptr(ptr) }
    }

    #[inline(always)]
    unsafe fn store_aligned(ptr: *mut f32, vec: Self::Vec) {
        // SAFETY: Caller guarantees alignment and validity.
        unsafe { vec.to_ptr_aligned(ptr) }
    }

    #[inline(always)]
    unsafe fn store_unaligned(ptr: *mut f32, vec: Self::Vec) {
        // SAFETY: Caller guarantees validity.
        unsafe { vec.to_ptr(ptr) }
    }

    #[inline(always)]
    fn partial_load(simd: &impl pulp::Simd, slice: &[f32]) -> Self::Vec {
        simd.partial_load_f32s(slice)
    }

    #[inline(always)]
    fn partial_store(simd: &impl pulp::Simd, slice: &mut [f32], vec: Self::Vec) {
        simd.partial_store_f32s(slice, vec)
    }

    #[inline(always)]
    fn as_simd(slice: &[f32]) -> (&[Self::Vec], &[f32]) {
        pulp::Simd::as_simd_f32s(slice)
    }

    #[inline(always)]
    fn as_simd_mut(slice: &mut [f32]) -> (&mut [Self::Vec], &mut [f32]) {
        pulp::Simd::as_mut_simd_f32s(slice)
    }

    #[inline(always)]
    fn reduce_sum(simd: &impl pulp::Simd, vec: Self::Vec) -> f32 {
        simd.reduce_sum_f32s(vec)
    }

    #[inline(always)]
    fn reduce_product(simd: &impl pulp::Simd, vec: Self::Vec) -> f32 {
        // Product reduction: multiply-accumulate lanes pairwise
        let mut acc = vec;
        let lanes = core::mem::size_of::<Self::Vec>() / core::mem::size_of::<f32>();
        let mut stride = lanes / 2;
        while stride > 0 {
            // SAFETY: this is a bitmask-based lane shuffle; pulp provides
            // no direct reduce_product, so we emulate with pairwise mul.
            // For small fixed-width types this unrolls completely.
            let hi = simd.extract_f32s(acc, stride);
            let lo = simd.extract_f32s(acc, 0);
            // Fallback: scalar reduction for portability
            stride = 0;
        }
        simd.reduce_sum_f32s(vec) // Simplified; real impl uses pairwise mul
    }

    #[inline(always)]
    fn cmp_lt(simd: &impl pulp::Simd, a: Self::Vec, b: Self::Vec) -> Self::Vec {
        pulp::cast!(simd.cmp_lt_f32s(a, b))
    }

    #[inline(always)]
    fn cmp_gt(simd: &impl pulp::Simd, a: Self::Vec, b: Self::Vec) -> Self::Vec {
        pulp::cast!(simd.cmp_gt_f32s(a, b))
    }

    #[inline(always)]
    fn cmp_eq(simd: &impl pulp::Simd, a: Self::Vec, b: Self::Vec) -> Self::Vec {
        pulp::cast!(simd.cmp_eq_f32s(a, b))
    }
}
```

### 4.5 SimdOps 实现：f64

```rust
#[cfg(feature = "simd")]
impl SimdOps<f64> for f64 {
    type Vec = pulp::f64s;

    #[inline(always)]
    fn splat(simd: &impl pulp::Simd, value: f64) -> Self::Vec {
        simd.splat_f64s(value)
    }

    #[inline(always)]
    fn add(simd: &impl pulp::Simd, a: Self::Vec, b: Self::Vec) -> Self::Vec {
        simd.add_f64s(a, b)
    }

    #[inline(always)]
    fn sub(simd: &impl pulp::Simd, a: Self::Vec, b: Self::Vec) -> Self::Vec {
        simd.sub_f64s(a, b)
    }

    #[inline(always)]
    fn mul(simd: &impl pulp::Simd, a: Self::Vec, b: Self::Vec) -> Self::Vec {
        simd.mul_f64s(a, b)
    }

    #[inline(always)]
    fn div(simd: &impl pulp::Simd, a: Self::Vec, b: Self::Vec) -> Self::Vec {
        simd.div_f64s(a, b)
    }

    #[inline(always)]
    fn neg(simd: &impl pulp::Simd, a: Self::Vec) -> Self::Vec {
        simd.sub_f64s(simd.splat_f64s(0.0), a)
    }

    #[inline(always)]
    fn mul_add(simd: &impl pulp::Simd, a: Self::Vec, b: Self::Vec, c: Self::Vec) -> Self::Vec {
        simd.mul_add_f64s(a, b, c)
    }

    #[inline(always)]
    fn abs(simd: &impl pulp::Simd, a: Self::Vec) -> Self::Vec {
        simd.abs_f64s(a)
    }

    #[inline(always)]
    fn sqrt(simd: &impl pulp::Simd, a: Self::Vec) -> Self::Vec {
        simd.sqrt_f64s(a)
    }

    #[inline(always)]
    unsafe fn load_aligned(ptr: *const f64) -> Self::Vec {
        unsafe { pulp::f64s::from_ptr_aligned(ptr) }
    }

    #[inline(always)]
    unsafe fn load_unaligned(ptr: *const f64) -> Self::Vec {
        unsafe { pulp::f64s::from_ptr(ptr) }
    }

    #[inline(always)]
    unsafe fn store_aligned(ptr: *mut f64, vec: Self::Vec) {
        unsafe { vec.to_ptr_aligned(ptr) }
    }

    #[inline(always)]
    unsafe fn store_unaligned(ptr: *mut f64, vec: Self::Vec) {
        unsafe { vec.to_ptr(ptr) }
    }

    #[inline(always)]
    fn partial_load(simd: &impl pulp::Simd, slice: &[f64]) -> Self::Vec {
        simd.partial_load_f64s(slice)
    }

    #[inline(always)]
    fn partial_store(simd: &impl pulp::Simd, slice: &mut [f64], vec: Self::Vec) {
        simd.partial_store_f64s(slice, vec)
    }

    #[inline(always)]
    fn as_simd(slice: &[f64]) -> (&[Self::Vec], &[f64]) {
        pulp::Simd::as_simd_f64s(slice)
    }

    #[inline(always)]
    fn as_simd_mut(slice: &mut [f64]) -> (&mut [Self::Vec], &mut [f64]) {
        pulp::Simd::as_mut_simd_f64s(slice)
    }

    #[inline(always)]
    fn reduce_sum(simd: &impl pulp::Simd, vec: Self::Vec) -> f64 {
        simd.reduce_sum_f64s(vec)
    }

    #[inline(always)]
    fn reduce_product(simd: &impl pulp::Simd, vec: Self::Vec) -> f64 {
        // Same pairwise-mul strategy as f32; simplified here.
        simd.reduce_sum_f64s(vec) // Placeholder; real impl uses pairwise mul
    }

    #[inline(always)]
    fn cmp_lt(simd: &impl pulp::Simd, a: Self::Vec, b: Self::Vec) -> Self::Vec {
        pulp::cast!(simd.cmp_lt_f64s(a, b))
    }

    #[inline(always)]
    fn cmp_gt(simd: &impl pulp::Simd, a: Self::Vec, b: Self::Vec) -> Self::Vec {
        pulp::cast!(simd.cmp_gt_f64s(a, b))
    }

    #[inline(always)]
    fn cmp_eq(simd: &impl pulp::Simd, a: Self::Vec, b: Self::Vec) -> Self::Vec {
        pulp::cast!(simd.cmp_eq_f64s(a, b))
    }
}
```

### 4.6 高层向量运算函数

这些函数是上层 `ops/` 模块调用的入口，封装了完整的 SIMD 循环逻辑（切片分割 → SIMD 循环 → 尾部处理）。

```rust
// ── Unary operations ────────────────────────────────────────

/// Applies a unary SIMD operation to contiguous `f32` data.
///
/// Reads `len` elements from `src`, applies `op`, writes to `dst`.
/// Handles SIMD chunking and scalar tail automatically.
///
/// # Safety
///
/// - `src` must be valid for `len * size_of::<f32>()` bytes.
/// - `dst` must be valid for `len * size_of::<f32>()` bytes.
/// - `src` and `dst` must not overlap.
#[cfg(feature = "simd")]
#[inline(always)]
pub unsafe fn simd_unary_f32<F>(
    simd: &impl pulp::Simd,
    src: *const f32,
    dst: *mut f32,
    len: usize,
    op: F,
)
where
    F: Fn(&impl pulp::Simd, pulp::f32s) -> pulp::f32s + Copy,
{
    // SAFETY: Caller guarantees validity.
    let src_slice = unsafe { core::slice::from_raw_parts(src, len) };
    let dst_slice = unsafe { core::slice::from_raw_parts_mut(dst, len) };

    let (src_vecs, src_tail) = pulp::Simd::as_simd_f32s(src_slice);
    let (dst_vecs, dst_tail) = pulp::Simd::as_mut_simd_f32s(dst_slice);

    for (src_v, dst_v) in src_vecs.iter().zip(dst_vecs.iter_mut()) {
        *dst_v = op(simd, *src_v);
    }

    if !src_tail.is_empty() {
        let partial = simd.partial_load_f32s(src_tail);
        let result = op(simd, partial);
        simd.partial_store_f32s(dst_tail, result);
    }
}

/// Applies a unary SIMD operation to contiguous `f64` data.
///
/// Same semantics as `simd_unary_f32` but for `f64` elements.
#[cfg(feature = "simd")]
#[inline(always)]
pub unsafe fn simd_unary_f64<F>(
    simd: &impl pulp::Simd,
    src: *const f64,
    dst: *mut f64,
    len: usize,
    op: F,
)
where
    F: Fn(&impl pulp::Simd, pulp::f64s) -> pulp::f64s + Copy,
{
    let src_slice = unsafe { core::slice::from_raw_parts(src, len) };
    let dst_slice = unsafe { core::slice::from_raw_parts_mut(dst, len) };

    let (src_vecs, src_tail) = pulp::Simd::as_simd_f64s(src_slice);
    let (dst_vecs, dst_tail) = pulp::Simd::as_mut_simd_f64s(dst_slice);

    for (src_v, dst_v) in src_vecs.iter().zip(dst_vecs.iter_mut()) {
        *dst_v = op(simd, *src_v);
    }

    if !src_tail.is_empty() {
        let partial = simd.partial_load_f64s(src_tail);
        let result = op(simd, partial);
        simd.partial_store_f64s(dst_tail, result);
    }
}

// ── Binary operations ────────────────────────────────────────

/// Applies a binary SIMD operation to two contiguous `f32` arrays.
///
/// Reads `len` elements from `lhs` and `rhs`, applies `op(lane_a, lane_b)`,
/// writes results to `dst`.
///
/// # Safety
///
/// - `lhs`, `rhs` must be valid for `len * size_of::<f32>()` bytes.
/// - `dst` must be valid for `len * size_of::<f32>()` bytes.
/// - `dst` must not overlap with `lhs` or `rhs`.
#[cfg(feature = "simd")]
#[inline(always)]
pub unsafe fn simd_binary_f32<F>(
    simd: &impl pulp::Simd,
    lhs: *const f32,
    rhs: *const f32,
    dst: *mut f32,
    len: usize,
    op: F,
)
where
    F: Fn(&impl pulp::Simd, pulp::f32s, pulp::f32s) -> pulp::f32s + Copy,
{
    let lhs_slice = unsafe { core::slice::from_raw_parts(lhs, len) };
    let rhs_slice = unsafe { core::slice::from_raw_parts(rhs, len) };
    let dst_slice = unsafe { core::slice::from_raw_parts_mut(dst, len) };

    let (lhs_vecs, lhs_tail) = pulp::Simd::as_simd_f32s(lhs_slice);
    let (rhs_vecs, rhs_tail) = pulp::Simd::as_simd_f32s(rhs_slice);
    let (dst_vecs, dst_tail) = pulp::Simd::as_mut_simd_f32s(dst_slice);

    for ((l, r), d) in lhs_vecs.iter().zip(rhs_vecs.iter()).zip(dst_vecs.iter_mut()) {
        *d = op(simd, *l, *r);
    }

    if !lhs_tail.is_empty() {
        let l_partial = simd.partial_load_f32s(lhs_tail);
        let r_partial = simd.partial_load_f32s(rhs_tail);
        let result = op(simd, l_partial, r_partial);
        simd.partial_store_f32s(dst_tail, result);
    }
}

/// Applies a binary SIMD operation to two contiguous `f64` arrays.
///
/// Same semantics as `simd_binary_f32` but for `f64` elements.
#[cfg(feature = "simd")]
#[inline(always)]
pub unsafe fn simd_binary_f64<F>(
    simd: &impl pulp::Simd,
    lhs: *const f64,
    rhs: *const f64,
    dst: *mut f64,
    len: usize,
    op: F,
)
where
    F: Fn(&impl pulp::Simd, pulp::f64s, pulp::f64s) -> pulp::f64s + Copy,
{
    let lhs_slice = unsafe { core::slice::from_raw_parts(lhs, len) };
    let rhs_slice = unsafe { core::slice::from_raw_parts(rhs, len) };
    let dst_slice = unsafe { core::slice::from_raw_parts_mut(dst, len) };

    let (lhs_vecs, lhs_tail) = pulp::Simd::as_simd_f64s(lhs_slice);
    let (rhs_vecs, rhs_tail) = pulp::Simd::as_simd_f64s(rhs_slice);
    let (dst_vecs, dst_tail) = pulp::Simd::as_mut_simd_f64s(dst_slice);

    for ((l, r), d) in lhs_vecs.iter().zip(rhs_vecs.iter()).zip(dst_vecs.iter_mut()) {
        *d = op(simd, *l, *r);
    }

    if !lhs_tail.is_empty() {
        let l_partial = simd.partial_load_f64s(lhs_tail);
        let r_partial = simd.partial_load_f64s(rhs_tail);
        let result = op(simd, l_partial, r_partial);
        simd.partial_store_f64s(dst_tail, result);
    }
}

// ── Reduction operations ─────────────────────────────────────

/// SIMD-accelerated sum reduction for contiguous `f32` data.
///
/// Uses 4-way ILP accumulator pattern for maximum throughput,
/// with FMA (fused multiply-add) when available.
///
/// # Safety
///
/// `data` must be valid for `len * size_of::<f32>()` bytes.
#[cfg(feature = "simd")]
#[inline(always)]
pub unsafe fn simd_sum_f32(simd: &impl pulp::Simd, data: *const f32, len: usize) -> f32 {
    let slice = unsafe { core::slice::from_raw_parts(data, len) };
    let (vecs, tail) = pulp::Simd::as_simd_f32s(slice);

    // 4-way ILP accumulators
    let mut acc0 = simd.splat_f32s(0.0);
    let mut acc1 = simd.splat_f32s(0.0);
    let mut acc2 = simd.splat_f32s(0.0);
    let mut acc3 = simd.splat_f32s(0.0);

    let (vecs4, vecs1) = pulp::as_arrays::<4, _>(vecs);

    for [v0, v1, v2, v3] in vecs4 {
        acc0 = simd.mul_add_f32s(*v0, simd.splat_f32s(1.0), acc0);
        acc1 = simd.mul_add_f32s(*v1, simd.splat_f32s(1.0), acc1);
        acc2 = simd.mul_add_f32s(*v2, simd.splat_f32s(1.0), acc2);
        acc3 = simd.mul_add_f32s(*v3, simd.splat_f32s(1.0), acc3);
    }

    for v in vecs1 {
        acc0 = simd.add_f32s(acc0, *v);
    }

    // Combine accumulators
    acc0 = simd.add_f32s(acc0, acc1);
    acc2 = simd.add_f32s(acc2, acc3);
    acc0 = simd.add_f32s(acc0, acc2);

    // Handle tail
    if !tail.is_empty() {
        let tail_vec = simd.partial_load_f32s(tail);
        acc0 = simd.add_f32s(acc0, tail_vec);
    }

    simd.reduce_sum_f32s(acc0)
}

/// SIMD-accelerated sum reduction for contiguous `f64` data.
///
/// # Safety
///
/// `data` must be valid for `len * size_of::<f64>()` bytes.
#[cfg(feature = "simd")]
#[inline(always)]
pub unsafe fn simd_sum_f64(simd: &impl pulp::Simd, data: *const f64, len: usize) -> f64 {
    let slice = unsafe { core::slice::from_raw_parts(data, len) };
    let (vecs, tail) = pulp::Simd::as_simd_f64s(slice);

    let mut acc0 = simd.splat_f64s(0.0);
    let mut acc1 = simd.splat_f64s(0.0);
    let mut acc2 = simd.splat_f64s(0.0);
    let mut acc3 = simd.splat_f64s(0.0);

    let (vecs4, vecs1) = pulp::as_arrays::<4, _>(vecs);

    for [v0, v1, v2, v3] in vecs4 {
        acc0 = simd.add_f64s(acc0, *v0);
        acc1 = simd.add_f64s(acc1, *v1);
        acc2 = simd.add_f64s(acc2, *v2);
        acc3 = simd.add_f64s(acc3, *v3);
    }

    for v in vecs1 {
        acc0 = simd.add_f64s(acc0, *v);
    }

    acc0 = simd.add_f64s(acc0, acc1);
    acc2 = simd.add_f64s(acc2, acc3);
    acc0 = simd.add_f64s(acc0, acc2);

    if !tail.is_empty() {
        let tail_vec = simd.partial_load_f64s(tail);
        acc0 = simd.add_f64s(acc0, tail_vec);
    }

    simd.reduce_sum_f64s(acc0)
}

// ── Dot product ──────────────────────────────────────────────

/// SIMD-accelerated dot product for contiguous `f32` data.
///
/// Computes `sum(lhs[i] * rhs[i])` for `i in 0..len`.
/// Uses 4-way ILP with FMA for maximum throughput.
///
/// # Safety
///
/// `lhs` and `rhs` must each be valid for `len * size_of::<f32>()` bytes.
#[cfg(feature = "simd")]
#[inline(always)]
pub unsafe fn simd_dot_f32(
    simd: &impl pulp::Simd,
    lhs: *const f32,
    rhs: *const f32,
    len: usize,
) -> f32 {
    let lhs_slice = unsafe { core::slice::from_raw_parts(lhs, len) };
    let rhs_slice = unsafe { core::slice::from_raw_parts(rhs, len) };

    let (lhs_vecs, lhs_tail) = pulp::Simd::as_simd_f32s(lhs_slice);
    let (rhs_vecs, rhs_tail) = pulp::Simd::as_simd_f32s(rhs_slice);

    let mut acc0 = simd.splat_f32s(0.0);
    let mut acc1 = simd.splat_f32s(0.0);
    let mut acc2 = simd.splat_f32s(0.0);
    let mut acc3 = simd.splat_f32s(0.0);

    let (lhs4, lhs1) = pulp::as_arrays::<4, _>(lhs_vecs);
    let (rhs4, rhs1) = pulp::as_arrays::<4, _>(rhs_vecs);

    for ([l0, l1, l2, l3], [r0, r1, r2, r3]) in lhs4.iter().zip(rhs4.iter()) {
        acc0 = simd.mul_add_f32s(*l0, *r0, acc0);
        acc1 = simd.mul_add_f32s(*l1, *r1, acc1);
        acc2 = simd.mul_add_f32s(*l2, *r2, acc2);
        acc3 = simd.mul_add_f32s(*l3, *r3, acc3);
    }

    for (l, r) in lhs1.iter().zip(rhs1.iter()) {
        acc0 = simd.mul_add_f32s(*l, *r, acc0);
    }

    acc0 = simd.add_f32s(acc0, acc1);
    acc2 = simd.add_f32s(acc2, acc3);
    acc0 = simd.add_f32s(acc0, acc2);

    if !lhs_tail.is_empty() {
        let l_partial = simd.partial_load_f32s(lhs_tail);
        let r_partial = simd.partial_load_f32s(rhs_tail);
        acc0 = simd.mul_add_f32s(l_partial, r_partial, acc0);
    }

    simd.reduce_sum_f32s(acc0)
}

/// SIMD-accelerated dot product for contiguous `f64` data.
///
/// # Safety
///
/// `lhs` and `rhs` must each be valid for `len * size_of::<f64>()` bytes.
#[cfg(feature = "simd")]
#[inline(always)]
pub unsafe fn simd_dot_f64(
    simd: &impl pulp::Simd,
    lhs: *const f64,
    rhs: *const f64,
    len: usize,
) -> f64 {
    let lhs_slice = unsafe { core::slice::from_raw_parts(lhs, len) };
    let rhs_slice = unsafe { core::slice::from_raw_parts(rhs, len) };

    let (lhs_vecs, lhs_tail) = pulp::Simd::as_simd_f64s(lhs_slice);
    let (rhs_vecs, rhs_tail) = pulp::Simd::as_simd_f64s(rhs_slice);

    let mut acc0 = simd.splat_f64s(0.0);
    let mut acc1 = simd.splat_f64s(0.0);
    let mut acc2 = simd.splat_f64s(0.0);
    let mut acc3 = simd.splat_f64s(0.0);

    let (lhs4, lhs1) = pulp::as_arrays::<4, _>(lhs_vecs);
    let (rhs4, rhs1) = pulp::as_arrays::<4, _>(rhs_vecs);

    for ([l0, l1, l2, l3], [r0, r1, r2, r3]) in lhs4.iter().zip(rhs4.iter()) {
        acc0 = simd.mul_add_f64s(*l0, *r0, acc0);
        acc1 = simd.mul_add_f64s(*l1, *r1, acc1);
        acc2 = simd.mul_add_f64s(*l2, *r2, acc2);
        acc3 = simd.mul_add_f64s(*l3, *r3, acc3);
    }

    for (l, r) in lhs1.iter().zip(rhs1.iter()) {
        acc0 = simd.mul_add_f64s(*l, *r, acc0);
    }

    acc0 = simd.add_f64s(acc0, acc1);
    acc2 = simd.add_f64s(acc2, acc3);
    acc0 = simd.add_f64s(acc0, acc2);

    if !lhs_tail.is_empty() {
        let l_partial = simd.partial_load_f64s(lhs_tail);
        let r_partial = simd.partial_load_f64s(rhs_tail);
        acc0 = simd.mul_add_f64s(l_partial, r_partial, acc0);
    }

    simd.reduce_sum_f64s(acc0)
}

// ── Scalar-tensor operations ─────────────────────────────────

/// Multiplies every element of a contiguous `f32` array by a scalar.
///
/// # Safety
///
/// `data` must be valid for `len * size_of::<f32>()` bytes.
#[cfg(feature = "simd")]
#[inline(always)]
pub unsafe fn simd_scale_f32(
    simd: &impl pulp::Simd,
    data: *mut f32,
    len: usize,
    scalar: f32,
) {
    let slice = unsafe { core::slice::from_raw_parts_mut(data, len) };
    let (vecs, tail) = pulp::Simd::as_mut_simd_f32s(slice);

    let s = simd.splat_f32s(scalar);
    for v in vecs {
        *v = simd.mul_f32s(*v, s);
    }

    if !tail.is_empty() {
        let partial = simd.partial_load_f32s(tail);
        simd.partial_store_f32s(tail, simd.mul_f32s(partial, s));
    }
}

/// Multiplies every element of a contiguous `f64` array by a scalar.
///
/// # Safety
///
/// `data` must be valid for `len * size_of::<f64>()` bytes.
#[cfg(feature = "simd")]
#[inline(always)]
pub unsafe fn simd_scale_f64(
    simd: &impl pulp::Simd,
    data: *mut f64,
    len: usize,
    scalar: f64,
) {
    let slice = unsafe { core::slice::from_raw_parts_mut(data, len) };
    let (vecs, tail) = pulp::Simd::as_mut_simd_f64s(slice);

    let s = simd.splat_f64s(scalar);
    for v in vecs {
        *v = simd.mul_f64s(*v, s);
    }

    if !tail.is_empty() {
        let partial = simd.partial_load_f64s(tail);
        simd.partial_store_f64s(tail, simd.mul_f64s(partial, s));
    }
}
```

### 4.7 WithSimd 分派模式

上层模块通过 `SimdContext` 的 `dispatch` 方法调用 SIMD 操作。使用 pulp 的 `WithSimd` trait + struct 模式（而非闭包）以确保 `#[inline(always)]` 生效。

```rust
/// Dispatches a SIMD operation using the WithSimd pattern.
///
/// This is the primary entry point for SIMD operations from upper-layer modules.
/// It performs runtime feature detection once and dispatches to the optimal
/// SIMD implementation.
///
/// # Usage Pattern
///
/// ```ignore
/// use xenon::backend::simd::{SimdContext, simd_dispatch};
/// use pulp::{Simd, WithSimd};
///
/// struct AddOp<'a> {
///     lhs: &'a [f32],
///     rhs: &'a [f32],
///     dst: &'a mut [f32],
/// }
///
/// impl WithSimd for AddOp<'_> {
///     type Output = ();
///
///     #[inline(always)]
///     fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
///         let s = &simd;
///         let (lhs_v, lhs_t) = S::as_simd_f32s(self.lhs);
///         let (rhs_v, rhs_t) = S::as_simd_f32s(self.rhs);
///         let (dst_v, dst_t) = S::as_mut_simd_f32s(self.dst);
///
///         for ((l, r), d) in lhs_v.iter().zip(rhs_v.iter()).zip(dst_v.iter_mut()) {
///             *d = simd.add_f32s(*l, *r);
///         }
///
///         if !lhs_t.is_empty() {
///             let lp = simd.partial_load_f32s(lhs_t);
///             let rp = simd.partial_load_f32s(rhs_t);
///             simd.partial_store_f32s(dst_t, simd.add_f32s(lp, rp));
///         }
///     }
/// }
///
/// let ctx = SimdContext::new();
/// ctx.dispatch(AddOp { lhs, rhs, dst });
/// ```
#[cfg(feature = "simd")]
impl SimdContext {
    /// Dispatches an operation to the best available SIMD implementation.
    ///
    /// Uses pulp's `Arch::dispatch()` internally, which performs a match
    /// on the detected CPU features and calls `with_simd()` with the
    /// appropriate SIMD type (V4/V3/V2/Neon/Scalar).
    #[inline(always)]
    pub fn dispatch<Op>(&self, op: Op) -> Op::Output
    where
        Op: pulp::WithSimd,
    {
        self.arch.dispatch(op)
    }
}
```

---

## 5. 内部实现设计

### 5.1 pulp 的分派机制

pulp 使用 **运行时特性检测 + 编译时单态化** 的混合策略：

```
Arch::new()                        // Runtime: CPUID / getauxval
    │
    ├── Arch::V4(V4)               // AVX-512 (nightly feature)
    │       └── Simd::vectorize(v4, op)
    │
    ├── Arch::V3(V3)               // AVX2 + FMA
    │       └── Simd::vectorize(v3, op)
    │
    ├── Arch::V2(V2)               // SSE2 (on x86, always available)
    │       └── Simd::vectorize(v2, op)
    │
    ├── Arch::Neon(Neon)           // NEON (aarch64)
    │       └── Simd::vectorize(neon, op)
    │
    └── Arch::Scalar               // Fallback (no SIMD)
            └── Simd::vectorize(Scalar, op)
```

`Simd::vectorize(simd, op)` 调用 `op.with_simd::<S>(simd)`，其中 `S` 是具体的 SIMD 类型。由于泛型单态化，编译器为每个 `S` 生成独立的代码路径，**零虚函数调用**。

### 5.2 Lane 宽度选择

pulp 的 `Simd` trait 为每种类型定义了"自然"向量宽度：

| 指令集 | 寄存器位宽 | `f32` lanes | `f64` lanes | pulp 类型 |
|--------|-----------|-------------|-------------|-----------|
| SSE2 (V2) | 128 | 4 (`f32x4`) | 2 (`f64x2`) | `pulp::x86::V2` |
| AVX2 (V3) | 256 | 8 (`f32x8`) | 4 (`f64x4`) | `pulp::x86::V3` |
| AVX-512 (V4) | 512 | 16 (`f32x16`) | 8 (`f64x8`) | `pulp::x86::V4` |
| NEON | 128 | 4 (`f32x4`) | 2 (`f64x2`) | `pulp::aarch64::Neon` |

pulp 通过关联类型 `Simd::f32s` 和 `Simd::f64s` 自动选择正确的向量宽度，上层代码无需手动管理。

### 5.3 对齐处理策略

```
数据指针 64-byte 对齐？
    │
    ├── 是 → 使用 aligned load/store（更快的内存访问）
    │         pulp 的 as_simd_* 自动处理
    │
    └── 否 → 使用 unaligned load/store（正确但稍慢）
              pulp 的 as_simd_* 内部使用 unaligned 路径
```

pulp 的 `Simd::as_simd_f32s()` 和 `as_simd_f64s()` 内部已处理对齐：
- 将 slice 分割为完整的 SIMD vectors 和尾部标量
- 对于已对齐的内存，编译器可能自动使用对齐加载
- Xenon 的 64 字节默认对齐保证了大多数场景下使用对齐路径

关键点：Xenon 默认 64 字节对齐分配（参见 `docs/06-layout.md` 5.4 节），新创建的 `Owned` 存储天然满足 SIMD 对齐要求。视图（slice/transpose 后）的起始地址可能偏移，此时 `LayoutFlags::ALIGNED` 标志降级，SIMD 路径仍然安全工作（使用非对齐加载）。

### 5.4 尾部元素处理

```
数据: [x0 x1 x2 ... x95 | x96 x97 x98 x99 x100]
       ├─ SIMD chunks ──┤  ├── tail ──┘

SIMD 宽度 = 8 (f32, AVX2)
chunks = 12 个 f32x8 向量
tail   = 5 个 f32 标量
```

两种策略（均为 pulp 原生支持）：

| 策略 | pulp 方法 | 适用场景 |
|------|-----------|---------|
| **标量尾部循环** | `as_simd_f32s` → `(&[f32x8], &[f32])` | 所有操作，简单可靠 |
| **Partial load/store** | `simd.partial_load_f32s(tail)` | 归约和 dot product，避免分支 |

设计选择：逐元素运算使用标量尾部循环（`for x in tail { *x = op(*x) }`），归约和 dot product 使用 partial load/store 以在尾部也利用 SIMD。

### 5.5 标量回退

当 `feature = "simd"` 未启用，或运行时检测到标量模式时，所有运算回退到 `backend/scalar.rs`：

```rust
// In ops/element_wise.rs dispatch logic:

#[cfg(feature = "simd")]
if should_use_simd(len, flags, core::mem::size_of::<A>()) {
    // SIMD path via backend::simd
} else {
    // Scalar fallback via backend::scalar
}

#[cfg(not(feature = "simd"))]
{
    // Always scalar path
}
```

### 5.6 复数 SIMD 运算

pulp 原生支持复数 SIMD 运算，通过 `Simd::c32s` 和 `Simd::c64s` 关联类型：

```rust
// pulp provides complex SIMD operations:
// simd.mul_e_c32s(a, b)           - complex multiplication
// simd.conj_mul_e_c32s(a, b)      - conjugate(a) * b
// simd.mul_add_e_c32s(a, b, c)    - a * b + c (complex FMA)
// simd.conj_mul_add_e_c32s(a, b, c) - conj(a) * b + c
```

复数 SIMD 的实现将作为后续扩展（参见任务拆分 T10），初始版本仅支持 `f32` 和 `f64`。

### 5.7 no_std 兼容性

pulp 自身支持 `no_std`，因此本模块在 `no_std` 环境下同样可用：

```rust
// pulp uses core::arch internally, no std dependency
#[cfg(feature = "simd")]
use pulp::{Arch, Simd, WithSimd};
```

---

## 6. 实现任务拆分

每个任务约 10 分钟，单一职责，可独立验证。

### T1: SimdContext 结构体与基本方法
- [ ] **文件**: `src/backend/simd.rs:1-80`
- **内容**: 定义 `SimdContext` 结构体（`arch: pulp::Arch`），实现 `new()`、`default()`、`arch_name()`、`dispatch()` 方法
- **测试**: `test_simd_context_new`, `test_simd_context_default`, `test_simd_context_arch_name`
- **前置**: Cargo.toml 添加 `pulp` 可选依赖
- **预计**: 10 min

### T2: should_use_simd 判定函数与常量
- [ ] **文件**: `src/backend/simd.rs:81-120`
- **内容**: 实现 `should_use_simd()`、`MIN_SIMD_ELEMENTS` 常量、`simd_lane_count<A>()` 函数
- **测试**: `test_should_use_simd_contiguous`, `test_should_use_simd_strided`, `test_should_use_simd_small`
- **前置**: T1
- **预计**: 10 min

### T3: SimdOps trait 定义
- [ ] **文件**: `src/backend/simd.rs:121-210`
- **内容**: 定义 `SimdOps<A>` trait 完整签名（splat、算术、FMA、数学、load/store、partial、slice、reduction、comparison）
- **测试**: 编译检查（trait 定义无运行时测试）
- **前置**: T1
- **预计**: 10 min

### T4: SimdOps 实现 — f32
- [ ] **文件**: `src/backend/simd.rs:211-330`
- **内容**: `impl SimdOps<f32> for f32`，所有方法委托 pulp 的 `Simd` trait 方法（splat_f32s、add_f32s、mul_f32s、partial_load_f32s 等）
- **测试**: `test_simd_ops_f32_add`, `test_simd_ops_f32_mul`, `test_simd_ops_f32_partial_load_store`
- **前置**: T3
- **预计**: 15 min

### T5: SimdOps 实现 — f64
- [ ] **文件**: `src/backend/simd.rs:331-450`
- **内容**: `impl SimdOps<f64> for f64`，结构同 f32 impl
- **测试**: `test_simd_ops_f64_add`, `test_simd_ops_f64_mul`, `test_simd_ops_f64_partial_load_store`
- **前置**: T3
- **预计**: 15 min

### T6: SIMD 一元运算函数
- [ ] **文件**: `src/backend/simd.rs:451-530`
- **内容**: 实现 `simd_unary_f32()` 和 `simd_unary_f64()` — 切片分割 + SIMD 循环 + 标量尾部处理
- **测试**: `test_simd_unary_f32_neg`, `test_simd_unary_f32_abs`, `test_simd_unary_f64_neg`, `test_simd_unary_tail_handling`
- **前置**: T4, T5
- **预计**: 10 min

### T7: SIMD 二元运算函数
- [ ] **文件**: `src/backend/simd.rs:531-620`
- **内容**: 实现 `simd_binary_f32()` 和 `simd_binary_f64()` — 双输入切片分割 + SIMD 循环 + 标量尾部
- **测试**: `test_simd_binary_f32_add`, `test_simd_binary_f32_mul_div`, `test_simd_binary_f64_sub`, `test_simd_binary_tail_handling`
- **前置**: T6
- **预计**: 10 min

### T8: SIMD 归约运算函数
- [ ] **文件**: `src/backend/simd.rs:621-720`
- **内容**: 实现 `simd_sum_f32()`、`simd_sum_f64()` — 4-way ILP 累加器 + partial load 尾部
- **测试**: `test_simd_sum_f32_small`, `test_simd_sum_f32_large`, `test_simd_sum_f64`, `test_simd_sum_f32_tail`
- **前置**: T4, T5
- **预计**: 15 min

### T9: SIMD 点积函数
- [ ] **文件**: `src/backend/simd.rs:721-820`
- **内容**: 实现 `simd_dot_f32()` 和 `simd_dot_f64()` — 4-way ILP FMA + partial load 尾部
- **测试**: `test_simd_dot_f32_unit_vectors`, `test_simd_dot_f32_orthogonal`, `test_simd_dot_f64`, `test_simd_dot_tail`
- **前置**: T8
- **预计**: 10 min

### T10: SIMD 标量乘法函数
- [ ] **文件**: `src/backend/simd.rs:821-880`
- **内容**: 实现 `simd_scale_f32()` 和 `simd_scale_f64()`
- **测试**: `test_simd_scale_f32`, `test_simd_scale_f64`, `test_simd_scale_tail`
- **前置**: T6
- **预计**: 5 min

### T11: backend/mod.rs 集成
- [ ] **文件**: `src/backend/mod.rs`
- **内容**: 添加 `#[cfg(feature = "simd")] pub mod simd;` 和 `pub use simd::*;`
- **测试**: `cargo build --features simd` 编译通过
- **前置**: T1-T10
- **预计**: 5 min

### T12: 文档注释与内联标注
- [ ] **文件**: `src/backend/simd.rs` 全文件
- **内容**: 为所有 pub struct/trait/fn 添加 doc comment，添加 `#[inline]` / `#[inline(always)]` 标注
- **测试**: `cargo doc --features simd --no-deps` 无警告
- **前置**: T1-T10
- **预计**: 10 min

---

## 7. 测试计划

### 7.1 单元测试（`src/backend/simd.rs` 内 `#[cfg(test)] mod tests`）

所有测试均受 `#[cfg(feature = "simd")]` 保护。

#### SimdContext 测试

| 测试函数 | 验证内容 |
|---------|---------|
| `test_simd_context_new` | `SimdContext::new()` 不 panic，`arch_name()` 返回非空字符串 |
| `test_simd_context_default` | `Default::default()` 等价于 `new()` |
| `test_simd_context_copy` | `Copy` 语义：clone 后两者独立可用 |

#### 判定函数测试

| 测试函数 | 验证内容 |
|---------|---------|
| `test_should_use_simd_contiguous_large` | 连续 + len ≥ 4 → true |
| `test_should_use_simd_contiguous_small` | 连续 + len < 4 → false |
| `test_should_use_simd_strided` | 非连续 → false |
| `test_should_use_simd_zero_len` | len == 0 → false |

#### SimdOps 测试

| 测试函数 | 验证内容 |
|---------|---------|
| `test_simd_ops_f32_splat` | `splat(3.14)` 所有 lane == 3.14 |
| `test_simd_ops_f32_add` | `[1,2,3,4] + [5,6,7,8] == [6,8,10,12]` |
| `test_simd_ops_f32_sub` | `[5,6,7,8] - [1,2,3,4] == [4,4,4,4]` |
| `test_simd_ops_f32_mul` | `[2,3,4,5] * [1,2,3,4] == [2,6,12,20]` |
| `test_simd_ops_f32_div` | `[6.0,8.0] / [2.0,4.0] == [3.0,2.0]` |
| `test_simd_ops_f32_neg` | `neg([1,-2,3,-4]) == [-1,2,-3,4]` |
| `test_simd_ops_f32_mul_add` | `2*3+4 == 10` (FMA) |
| `test_simd_ops_f32_abs` | `abs([-1,2,-3,4]) == [1,2,3,4]` |
| `test_simd_ops_f32_sqrt` | `sqrt([4.0,9.0,16.0]) ≈ [2.0,3.0,4.0]` |
| `test_simd_ops_f64_add` | f64 版加法 |
| `test_simd_ops_f64_mul` | f64 版乘法 |
| `test_simd_ops_f64_abs` | f64 版绝对值 |

#### 一元运算测试

| 测试函数 | 验证内容 |
|---------|---------|
| `test_simd_unary_f32_neg` | 全量取反正确 |
| `test_simd_unary_f32_abs` | 全量绝对值正确 |
| `test_simd_unary_f64_neg` | f64 版取反 |
| `test_simd_unary_exact_multiple` | len 恰好是 SIMD 宽度整数倍 |
| `test_simd_unary_tail_handling` | len 非 SIMD 宽度整数倍时尾部正确 |
| `test_simd_unary_single_element` | len == 1 |
| `test_simd_unary_empty` | len == 0 |

#### 二元运算测试

| 测试函数 | 验证内容 |
|---------|---------|
| `test_simd_binary_f32_add` | 向量加法 |
| `test_simd_binary_f32_sub` | 向量减法 |
| `test_simd_binary_f32_mul` | 向量乘法 |
| `test_simd_binary_f32_div` | 向量除法 |
| `test_simd_binary_f64_add` | f64 版加法 |
| `test_simd_binary_tail_handling` | 尾部正确处理 |
| `test_simd_binary_large` | 大数组（>4096 元素）正确性 |

#### 归约测试

| 测试函数 | 验证内容 |
|---------|---------|
| `test_simd_sum_f32_empty` | 空数组返回 0.0 |
| `test_simd_sum_f32_single` | 单元素 |
| `test_simd_sum_f32_small` | 7 个元素（< 1 个 SIMD 宽度 × 4-way ILP） |
| `test_simd_sum_f32_exact_multiple` | 恰好是 SIMD 宽度整数倍 |
| `test_simd_sum_f32_large` | 10000 个元素 |
| `test_simd_sum_f32_all_ones` | N 个 1.0 → sum == N |
| `test_simd_sum_f64_large` | f64 版大数组 |

#### 点积测试

| 测试函数 | 验证内容 |
|---------|---------|
| `test_simd_dot_f32_unit_vectors` | `[1,0,0] · [0,1,0] == 0` |
| `test_simd_dot_f32_parallel` | `[1,1,1] · [2,2,2] == 6` |
| `test_simd_dot_f32_orthogonal` | 正交向量点积为 0 |
| `test_simd_dot_f32_large` | 大数组与标量实现结果一致 |
| `test_simd_dot_f64_large` | f64 版大数组 |
| `test_simd_dot_f32_tail` | 非 SIMD 整数倍长度 |

#### 标量乘法测试

| 测试函数 | 验证内容 |
|---------|---------|
| `test_simd_scale_f32_zero` | 乘以 0 全部为 0 |
| `test_simd_scale_f32_one` | 乘以 1 不变 |
| `test_simd_scale_f32_two` | 乘以 2 每元素翻倍 |
| `test_simd_scale_f64` | f64 版 |
| `test_simd_scale_f32_tail` | 尾部正确处理 |

### 7.2 集成测试

SIMD 后端的集成测试在 `tests/` 目录下由上层运算模块间接覆盖：

| 文件 | 覆盖场景 |
|------|----------|
| `tests/arithmetic.rs` | 逐元素运算使用 SIMD 路径时的结果正确性 |
| `tests/reduction.rs` | 归约运算使用 SIMD 路径时的结果正确性 |
| `tests/matrix_ops.rs` | 矩阵-向量乘法使用 SIMD 点积时的结果正确性 |
| `tests/edge_cases.rs` | SIMD 路径的边界条件（空数组、单元素、NaN、非对齐） |

### 7.3 基准测试

| 文件 | 测试内容 |
|------|----------|
| `benches/element_ops.rs` | 标量 vs SIMD 逐元素运算性能对比 |
| `benches/reduction.rs` | 标量 vs SIMD 归约性能对比 |
| `benches/matrix_ops.rs` | 标量 vs SIMD 矩阵运算性能对比 |

基准测试须验证 SIMD 路径的加速比：

| 操作 | 预期加速（AVX2 vs 标量） |
|------|--------------------------|
| 逐元素 add (f32) | ≥ 4x |
| 逐元素 add (f64) | ≥ 2x |
| sum 归约 (f32) | ≥ 6x（含 ILP） |
| dot product (f32) | ≥ 6x（含 FMA + ILP） |

### 7.4 测试辅助宏

```rust
/// Assert that two f32 slices are element-wise approximately equal.
#[cfg(feature = "simd")]
macro_rules! assert_slice_close {
    ($left:expr, $right:expr, $tol:expr) => {
        let l: &[f32] = $left;
        let r: &[f32] = $right;
        assert_eq!(l.len(), r.len(), "slice length mismatch");
        for (i, (a, b)) in l.iter().zip(r.iter()).enumerate() {
            assert!(
                (a - b).abs() <= $tol,
                "assertion failed at index {i}: |{a} - {b}| > {}",
                $tol
            );
        }
    };
}

/// Assert that two f64 slices are element-wise approximately equal.
#[cfg(feature = "simd")]
macro_rules! assert_slice_close_f64 {
    ($left:expr, $right:expr, $tol:expr) => {
        let l: &[f64] = $left;
        let r: &[f64] = $right;
        assert_eq!(l.len(), r.len(), "slice length mismatch");
        for (i, (a, b)) in l.iter().zip(r.iter()).enumerate() {
            assert!(
                (a - b).abs() <= $tol,
                "assertion failed at index {i}: |{a} - {b}| > {}",
                $tol
            );
        }
    };
}
```

---

## 附录 A：pulp API 映射表

pulp 的 `Simd` trait 方法名采用 `{op}_{type}s` 命名（`s` 后缀表示"vector"，如 `f32s` = f32 SIMD vector）：

| Xenon 操作 | pulp f32 方法 | pulp f64 方法 | 说明 |
|-----------|---------------|---------------|------|
| splat | `simd.splat_f32s(v)` | `simd.splat_f64s(v)` | 广播标量到所有 lane |
| add | `simd.add_f32s(a, b)` | `simd.add_f64s(a, b)` | 逐 lane 加 |
| sub | `simd.sub_f32s(a, b)` | `simd.sub_f64s(a, b)` | 逐 lane 减 |
| mul | `simd.mul_f32s(a, b)` | `simd.mul_f64s(a, b)` | 逐 lane 乘 |
| div | `simd.div_f32s(a, b)` | `simd.div_f64s(a, b)` | 逐 lane 除 |
| FMA | `simd.mul_add_f32s(a, b, c)` | `simd.mul_add_f64s(a, b, c)` | a*b+c |
| abs | `simd.abs_f32s(a)` | `simd.abs_f64s(a)` | 绝对值 |
| sqrt | `simd.sqrt_f32s(a)` | `simd.sqrt_f64s(a)` | 平方根 |
| reduce_sum | `simd.reduce_sum_f32s(v)` | `simd.reduce_sum_f64s(v)` | 水平求和 |
| partial_load | `simd.partial_load_f32s(s)` | `simd.partial_load_f64s(s)` | 不完整向量加载 |
| partial_store | `simd.partial_store_f32s(s, v)` | `simd.partial_store_f64s(s, v)` | 不完整向量存储 |
| as_simd | `S::as_simd_f32s(slice)` | `S::as_simd_f64s(slice)` | 切片分割 |
| as_mut_simd | `S::as_mut_simd_f32s(slice)` | `S::as_mut_simd_f64s(slice)` | 可变切片分割 |
| cmp_lt | `simd.cmp_lt_f32s(a, b)` | `simd.cmp_lt_f64s(a, b)` | 逐 lane 小于比较 |
| cmp_eq | `simd.cmp_eq_f32s(a, b)` | `simd.cmp_eq_f64s(a, b)` | 逐 lane 等于比较 |
| is_nan | `simd.is_nan_f32x4(a)` | `simd.is_nan_f64x2(a)` | NaN 检测 |

## 附录 B：性能分层决策流程

```
上层 ops 模块调用 apply_binary(lhs, rhs, |a, b| a + b)
    │
    ├── 元素数 < MIN_SIMD_ELEMENTS (4)?
    │   └── 是 → backend::scalar 路径
    │
    ├── LayoutFlags::is_contiguous() == false?
    │   └── 是 → backend::scalar 路径（非连续内存不适合 SIMD）
    │
    ├── #[cfg(feature = "simd")]
    │   ├── SimdContext::dispatch() → pulp 自动选择最优指令集
    │   │   ├── AVX-512 (V4): 512-bit 寄存器，16 f32 / 8 f64 per op
    │   │   ├── AVX2 (V3): 256-bit 寄存器，8 f32 / 4 f64 per op
    │   │   ├── SSE2 (V2): 128-bit 寄存器，4 f32 / 2 f64 per op
    │   │   ├── NEON: 128-bit 寄存器，4 f32 / 2 f64 per op
    │   │   └── Scalar: 标量回退
    │   │
    │   └── 运算内部: as_simd → SIMD 循环 → partial/tail 处理
    │
    ├── #[cfg(feature = "parallel")]
    │   └── 元素数 ≥ PARALLEL_THRESHOLD (64K)?
    │       └── 是 → 分块 + 每块内使用 SIMD
    │
    └── #[cfg(not(feature = "simd"))]
        └── backend::scalar 路径
```

## 附录 C：Cargo.toml 配置

```toml
[features]
default = ["std"]
std = []
parallel = ["dep:rayon"]
simd = ["dep:pulp"]
full = ["std", "parallel", "simd"]

[dependencies]
rayon = { version = "1.10", optional = true }
pulp = { version = "0.18", optional = true }
```

pulp 的 feature flags（在 pulp 的 Cargo.toml 中控制编译时指令集支持）：

| pulp feature | 启用的指令集 | 说明 |
|--------------|-------------|------|
| `x86-v3` (默认) | SSE2 + AVX2 | 稳定 Rust，x86_64 最常用 |
| `nightly-x86-v4` | + AVX-512 | 需要 nightly Rust |
| (无特殊 feature) | Scalar | 通用回退 |

Xenon 默认使用 pulp 的 `x86-v3` feature（AVX2），不在 Xenon 层面暴露 pulp 的 nightly feature。
