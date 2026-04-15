# SIMD 后端模块设计

> 文档编号: 08 | 模块: `src/simd/` | 阶段: Phase 5
> 前置文档: `03-element.md`, `06-layout.md`, `07-tensor.md`
> 需求参考: 需求说明书 §9.1, §9.3, §27, §28.3, §28.5
> 范围声明: 范围内

---

## 1. 模块定位

SIMD 后端模块是 Xenon 张量库的可选性能加速层，通过 `pulp` crate 提供跨平台 SIMD 抽象。根据需求说明书 §9.1，当前版本以算术运算为优先，逐步覆盖比较、`conjugate`、`sum` 与 **vector dot** 等已实现子集；数学函数属于后续增强目标，完整覆盖计划见 §5.4a。该模块默认关闭，通过 `features = ["simd"]` 启用。

### 1.1 职责边界

| 职责       | 包含                                                                 | 不包含                                     |
| ---------- | -------------------------------------------------------------------- | ------------------------------------------ |
| SIMD 抽象  | 通过 pulp 统一抽象 x86/ARM 指令集                                    | GPU 加速 (CUDA/OpenCL)                     |
| 逐元素运算 | 仅覆盖本文明确列出的已实现子集：二元算术、部分一元/比较、复数 `conjugate`，以及可向量化时的 `bool not` | 未列出的逐元素运算与数学函数专用 kernel |
| 归约运算   | `sum` 的 SIMD 加速与自动回退策略                                     | `max`/`argmax` 等其他归约                  |
| 内积运算   | `dot` 的 SIMD 加速与自动回退策略                                     | 外积、矩阵乘法                             |
| 标量回退   | 所有操作的纯标量基准实现                                             | —                                          |
| 运行时分发 | Arch 检测缓存、自动最优路径选择                                      | 编译期静态分发                             |

### 1.2 设计原则

| 原则         | 体现                                                                                     |
| ------------ | ---------------------------------------------------------------------------------------- |
| 结果一致性   | 逐元素路径要求保持公开语义一致；整数 `sum` / `dot` 必须等价于逐步 checked arithmetic，浮点/复数 `sum` / `dot` 允许在已文档化容差内偏离标量累加顺序结果      |
| 自动回退     | 不满足 SIMD 条件时自动回退到标量，用户无需感知                                           |
| 零成本抽象   | 未启用 `simd` feature 时无任何运行时开销                                                 |
| 跨平台       | pulp 统一 x86_64 (SSE4.1/AVX2/AVX-512) 和 ARM (NEON)；SVE 属于后续扩展，不纳入当前设计                                     |
| 精度优先约束 | 元素级 `mul`/`add` 不使用 FMA 以保持逐位一致；仅在已记录容差的 reduction merge 内部可使用 |

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
│              math, reduction, matrix::dot                       │
│  See 11-math.md §5, 13-reduction.md §5, 12-matrix.md §5.1       │
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
| 需求映射 | `需求说明书 §9.1`、`§9.3`、`§27`、`§28.3`、`§28.5` |
| 范围内   | 本文覆盖矩阵中列出的 SIMD 子集：`add`/`sub`/`mul`/`div`、`abs`/`neg`、`eq`/`ne`/`lt`/`gt`、复数 `conjugate`、可选 `bool not`、以及 `sum` / `dot`；未列出能力统一回退标量 |
| 范围外   | 矩阵乘法、非 `sum` 归约、归约模块负责的 `norm` 等专用 SIMD kernel，以及 SVE 当前版本实现 |
| 标量回退 | 任一操作在当前类型、ISA、对齐或语义约束下无法高效且正确实现 SIMD 时，自动回退标量 |
| 非目标   | No new third-party dependencies beyond pulp                                          |

---

## 3. 文件位置

```
src/simd/
├── mod.rs             # pulp integration, Arch cache, public API, SimdKernel trait
├── scalar.rs          # scalar fallback implementation (reference path for all ops)
└── vector.rs          # vectorized implementation based on pulp WithSimd
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
├── crate::layout             # Alignment helpers
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
simd = ["dep:pulp"]

[dependencies]
pulp = { version = "0.18", optional = true }
```

- 默认关闭，通过 `features = ["simd"]` 显式启用
- 启用后 pulp 自动引入，提供跨平台 SIMD 抽象

### 5.2 SimdElement Trait

```rust,ignore
// src/simd/mod.rs

use crate::complex::Complex;

/// Marker trait for SIMD kernel element types.
///
/// Distinguishes different element types at compile time for SIMD dispatch.
/// Only implemented for numeric types that support SIMD operations.
pub trait SimdElement: Copy + Clone + Send + Sync + 'static {
    /// Element size in bytes.
    const SIZE: usize;

    /// Natural alignment of the element type.
    ///
    /// This constant is only for kernel-internal type metadata and must not be
    /// used as the public SIMD fast-path admission rule. External alignment
    /// checks always use Xenon's unified threshold via `layout::is_aligned()`.
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

impl SimdElement for Complex<f32> {
    const SIZE: usize = 8;
    const ALIGN: usize = 4;
}

impl SimdElement for Complex<f64> {
    const SIZE: usize = 16;
    const ALIGN: usize = 8;
}
```

> **不实现 SimdElement 的类型**：以下元素类型不通过本 trait 直接进入数值 SIMD kernel：
>
> | 类型         | 原因                                       |
> | ------------ | ------------------------------------------ |
> | `bool`       | 若需要 `not` / `eq` / `ne` 的向量化，使用独立 mask/byte-lane kernel；不复用数值 `SimdElement` |
> | `usize`      | 指针宽度依赖平台，SIMD 语义不稳定          |
> | 其他 `Complex<T>` | 仅 `Complex<f32>` / `Complex<f64>` 在当前设计中进入专用 kernel |
>
> **复数 SIMD 策略：** `Complex<f32>` 和 `Complex<f64>` 在当前设计中优先覆盖 `sum()` / `dot()` 与 `conjugate()`。实现上采用 AoS 输入 + 寄存器内重排，按实部/虚部分离后执行
> 向量累加与复数乘加，避免引入额外的布局转换 API。

### 5.3 SimdKernel Trait

```rust,ignore
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

    // NOTE: Unary operations are part of the SIMD design target.
    // Concrete kernels may still fall back to scalar per type/ISA.

    /// Element-wise negation.
    fn neg(&self, src: &[A], dst: &mut [A]);

    /// Element-wise absolute value.
    fn abs(&self, src: &[A], dst: &mut [A]);

    // ========================================
    // Reduction operations
    // ========================================

    /// Sum reduction.
    ///
    /// Covers integer, floating-point, and supported complex element types.
    /// Integer kernels must be equivalent to scalar checked arithmetic at every
    /// observable accumulation step; if this cannot be guaranteed efficiently,
    /// they must fall back to the scalar path. Floating-point/complex kernels may
    /// use a different accumulation order, but per `require.md` §9.3 / §28.3 any
    /// tolerance must be defined and documented from the final algorithm and test baseline.
    fn sum(&self, data: &[A]) -> A;

    // ========================================
    // Vector dot operations
    // ========================================

    /// Vector dot product.
    ///
    /// Floating-point and complex kernels may use SIMD lane-local accumulation and a
    /// documented merge step, with the final result staying within the documented
    /// tolerance bound. Integer kernels must be equivalent to scalar checked
    /// multiply-then-add semantics at each step; if SIMD cannot guarantee that,
    /// this operation must fall back to scalar.
    ///
    /// # Panics
    ///
    /// Panics if the two slices have different lengths.
    fn dot(&self, lhs: &[A], rhs: &[A]) -> A;

}
```

### 5.4 pulp 集成与 Arch 缓存

```rust,ignore
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
/// `math/`, `matrix::dot`, and `reduction/` must call this facade only after
/// extracting compatible contiguous slices, rather than passing tensor/view
/// objects directly. The facade may choose a vector kernel or fall back to
/// `ScalarKernel` based on the current version's contiguous unified-alignment fast path
/// and the operation-specific
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

当前版本 SIMD 只覆盖下表列出的子集；未列出的操作统一走标量路径。是否实际进入 SIMD 还取决于元素类型、ISA 能力、统一对齐快路径与语义约束。当前版本在 `matrix` 相关范围内仅承载 **vector dot**，不展开矩阵乘法或其他 matrix 范围能力。以下表格给出当前设计的覆盖矩阵：

> **覆盖范围说明：** 当前版本 SIMD 路径按运算和类型逐步启用。对于尚未实现 SIMD kernel 的运算，系统自动回退到标量路径，不影响公开 API 语义。

| 类别 | 操作 | 类型 | 当前设计目标 |
| ---- | ---- | ---- | ------------ |
| 二元算术 | `add` / `sub` / `mul` / `div` | `i32` / `i64` / `f32` / `f64` | 提供 SIMD path |
| 一元 | `abs` / `neg` | `i32` / `i64` / `f32` / `f64` | 提供 SIMD path，不满足约束时回退标量 |
| 一元/数学函数 | `signum` / `square` / `sin` / `sqrt` / `exp` / `ln` / `floor` / `ceil` | 支持类型见需求说明书 §12 | 当前版本以算术运算为优先；`sin` / `sqrt` / `exp` / `ln` / `floor` / `ceil` 作为后续增强目标，`signum` / `square` 因无 ISA 原生支持走标量路径 |
| 比较 | `eq` / `ne` / `lt` / `gt` | `i32` / `i64` / `f32` / `f64` | 提供 SIMD path |
| 逻辑 | `not` | `bool` | 若可用 mask/byte-lane 向量化则走 SIMD，否则回退标量 |
| 复数 | `conjugate` | `Complex<f32>` / `Complex<f64>` | 提供 SIMD path |
| 归约 | `sum` | `i32` / `i64` / `f32` / `f64` / `Complex<f32>` / `Complex<f64>` | 提供 SIMD path；整数无法保证 checked 语义时回退标量 |
| 内积 | `dot` | `i32` / `i64` / `f32` / `f64` / `Complex<f32>` / `Complex<f64>` | 提供 SIMD path；整数无法保证 checked 语义时回退标量 |

| Public API | SIMD 状态 |
| ---------- | --------- |
| `math::add` / `sub` / `mul` / `div` | implemented |
| `math::abs` / `neg` | implemented |
| `math::eq` / `ne` / `lt` / `gt` | implemented |
| `math::not` (`bool`) | fallback（仅在存在独立 mask/byte-lane kernel 时启用 SIMD） |
| `math::signum` / `square` / `sin` / `sqrt` / `exp` / `ln` / `floor` / `ceil` | fallback |
| `math::conjugate` | implemented |
| `reduction::sum` | implemented |
| `matrix::dot` / `dot` | implemented |

> **`SimdKernel::dot()` 说明：** `dot()` 为当前版本正式能力。`f32` / `f64` / `Complex<f32>` / `Complex<f64>` 可以使用 SIMD dot kernel；`i32` / `i64` 仅在可证明与标量逐步 checked arithmetic 等价时使用 SIMD，否则自动回退标量实现。

该覆盖目标不改变公开 API 的可用性；SIMD 仅影响执行路径选择，不改变公开语义契约。

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
│ • Supported element type / op?                                  │
│ • Current unified-alignment fast path enabled?                  │
│   (contiguous + aligned under the current version contract)     │
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
│ int: use SIMD only when checked semantics can be preserved      │
│      at each effective accumulation step; otherwise scalar      │
│ float: lane_acc = pairwise_or_kahan_add(...)                    │
│ complex: split real/imag lanes, then accumulate separately      │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ Horizontal merge                                                │
│ int: scalar checked merge only after a semantics-safe SIMD      │
│      prefix; if not possible, do not enter SIMD path            │
│ float: pairwise scalar merge, allowing documented tolerance     │
│ complex: pairwise merge on real/imag parts, rebuild Complex     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ Scalar tail accumulation [chunks*W..N)                          │
└─────────────────────────────────────────────────────────────────┘
```

#### 轴向归约 SIMD 策略

- 轴向 `sum(axis)` / `sum_axis(axis)` / `sum_keepdims(axis)` 的 SIMD 路径仅在被归约维对应连续内层维度时启用。
- 若输入布局非连续，或目标轴无法映射为连续内层归约区间，则自动回退标量路径。
- 当目标轴是连续内层维度时，实现可在每个逻辑归约段内使用 SIMD 加速全局归约，再按轴向输出形状写回结果；`keepdims` 仅影响输出 shape，不改变 SIMD/标量裁决规则。

#### 向量内积（dot product）

```
SIMD dot dispatch flow

┌─────────────────────────────────────────────────────────────────┐
│ Input: lhs[0..N], rhs[0..N]                                     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ Check SIMD conditions + type-specific contract                  │
│ • f32/f64: SIMD mul + lane-local accumulation                   │
│ • Complex: `conj(lhs)`-multiply-add on split real/imag lanes    │
│ • Integer: only use SIMD when checked multiply/add semantics    │
│   can be preserved; otherwise scalar fallback                   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ SIMD main loop + scalar tail + documented merge                 │
│ if contract cannot be met on this ISA -> scalar fallback        │
└─────────────────────────────────────────────────────────────────┘
```

### 5.6 条件检查与自动回退

```rust,ignore
// src/simd/mod.rs

/// Alignment checking functions are defined in `06-layout.md §5.5`.
/// SIMD paths use `layout::is_aligned()` from the `layout` module.
use crate::layout::is_aligned;
use pulp::{Arch, Simd, WithSimd};

/// Checks whether the current unified-alignment SIMD fast path can be used.
///
/// Returns `true` only for the current version's contiguous, aligned fast path;
/// all other cases automatically fall back to the scalar path.
///
/// # SIMD Conditions
///
/// 1. `simd` feature is enabled
/// 2. Memory is contiguous (F-order)
/// 3. Element type supports SIMD (`f32`/`f64`/`i32`/`i64` plus supported complex reduction/dot types)
/// 4. The current version's unified-alignment fast path is satisfied
///
/// `SimdElement::ALIGN` is not part of this admission rule; it remains an
/// internal kernel-side description of element natural alignment. Public path
/// selection only consults Xenon's unified alignment threshold.
///
/// `simd_width()` returns the runtime-selected lane width exposed by the best
/// ISA available on the current machine for the requested element type.
/// It is derived from `get_arch()` + the corresponding `pulp::Simd` lane count,
/// rather than from the scalar kernel. When the active ISA has no vector support
/// for `A`, this function returns `1` and callers naturally fall back to scalar.
#[cfg(feature = "simd")]
#[inline]
fn simd_width<A: SimdElement>() -> usize {
    detect_vector_width::<A>(get_arch()).max(1)
}

#[cfg(feature = "simd")]
fn detect_vector_width<A: SimdElement>(arch: Arch) -> usize {
    arch.dispatch(DetectWidth::<A>::new())
}

#[cfg(feature = "simd")]
struct DetectWidth<A>(core::marker::PhantomData<A>);

#[cfg(feature = "simd")]
impl<A> DetectWidth<A> {
    fn new() -> Self {
        Self(core::marker::PhantomData)
    }
}

#[cfg(feature = "simd")]
impl<A: SimdElement> WithSimd for DetectWidth<A> {
    type Output = usize;

    fn with_simd<S: Simd>(self, _: S) -> usize {
        vector_width_for::<A, S>()
    }
}

#[cfg(feature = "simd")]
pub fn can_use_simd<A: SimdElement>(
    lhs_ptr: *const A,
    rhs_ptr: *const A,
    dst_ptr: *const A,
    len: usize,
    is_contiguous: bool,
) -> bool {
    if !is_contiguous || len < simd_width::<A>() {
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

上例中的 `vector_width_for::<A, S>()` 是类型级辅助函数：对 `f32` / `f64` / `i32` / `i64` 直接返回对应 lane 数；对 `Complex<f32>` / `Complex<f64>` 返回“每次迭代可处理的复数元素数”（即底层实数 lane 数的一半）；若该 ISA 不支持目标类型则返回 `1`。因此 `simd_width()` 反映的是**当前机器当前类型**的实际可用宽度，而不是固定常量或标量占位值。

### 5.7 Good/Bad 对比示例

```rust,ignore
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
// May produce incorrect results on non-contiguous or data failing
// Xenon's unified alignment threshold
    dispatch_binary_op(BinaryOp::Add, lhs, rhs, dst);
}
```

---

## 6. 内部实现设计

### 6.1 pulp WithSimd 使用模式

以 f32 加法为例，展示完整的 pulp API 使用方式：

```rust,ignore
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
                // Xenon's unified alignment fast-path precondition has already
                // been checked before dispatch.
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

```rust,ignore
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

}

// f32, i32, i64, Complex<f32>, and Complex<f64> implementations are similar,
// omitted here for brevity.
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
> ```rust,ignore
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
> **SIMD 路径的整数归约溢出：** `require.md` 要求整数归约/内积在每一步都满足 checked arithmetic 语义，而不仅是最终结果不越界。单纯 widening accumulation 不能捕获“中间溢出后又回到范围内”的情况，因此**不能**单独作为正确性依据。当前设计改为：仅当某个整数 SIMD kernel 能明确证明与标量逐步 `checked_add` / `checked_mul` + `checked_add` 等价时才允许进入 SIMD；否则 `sum` / `dot` 必须直接回退标量路径。允许的 SIMD 优化范围仅限安全的批量装载、前缀过滤或不会改变可观察 checked 语义的局部步骤。

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
│ • Reduction ops: integer exact; float/complex allow documented  │
│   tolerance due to accumulation order                           │
│ • Dot ops: integer exact; float/complex allow documented        │
│   tolerance due to accumulation order                           │
└─────────────────────────────────────────────────────────────────┘
```

> **设计决策：** 对于逐元素运算，SIMD 与标量路径必须保持相同公开语义；对 `add` / `sub` / `mul` / `div`、比较、`conjugate`、以及可向量化的一元/逻辑操作，若对应 ISA kernel 已实现，则结果应与标量逐元素求值一致。对于归约和内积，当前版本必须实现 SIMD 设计路径，并遵循 `require.md` §9.1 / §9.3 / §28.3 的数值约束：整数结果必须与标量逐步 checked arithmetic 精确一致；`f32` / `f64` / `Complex<f32>` / `Complex<f64>` 若允许因结合顺序变化产生差异，则每个实数分量与标量路径结果之差不得超过 `max(1 ULP, 2^-23 * |scalar_result|)`（`f32`）或 `max(1 ULP, 2^-52 * |scalar_result|)`（`f64`）。
>
> **一致性说明：** 对于逐元素操作（add、mul 等），SIMD 和标量路径产生逐位一致的结果。
> 对于归约/内积操作，Xenon 不接受未记录的“近似一致”；若某个 SIMD 内核无法满足文档定义的数值语义与容差边界，则不走 SIMD 路径。
>
> **覆盖说明：** 当前版本只覆盖本文表格中列出的 SIMD 子集；其中 `bool not` 是否实际进入 SIMD，取决于具体 ISA kernel 是否存在且能满足本文语义约束。数学函数作为后续增强目标逐步补齐；这不改变 `math` 模块对这些逐元素运算的公开 API 承诺。完整覆盖计划见 §5.4a。

### 6.5 SIMD reduction / dot 内部策略

| 类型 | `sum()` SIMD 策略 | `dot()` SIMD 策略 | 精度/溢出约束 |
| ---- | ----------------- | ----------------- | ------------- |
| `i32` | 仅在可证明与标量逐步 `checked_add` 等价时进入 SIMD；否则标量回退 | 仅在可证明与标量逐步 `checked_mul` + `checked_add` 等价时进入 SIMD；否则标量回退 | 与标量整数语义精确一致，任一步溢出立即 panic |
| `i64` | 同上；不以 widening accumulation 代替 checked 语义 | 同上 | 与标量整数语义精确一致，任一步溢出立即 panic |
| `f32` | lane 内 pairwise/Kahan-style 累加，水平合并允许不同顺序 | `mul` 后进入同一累加流程 | 每个实数分量与标量路径结果之差不超过 `max(1 ULP, 2^-23 * |scalar_result|)` |
| `f64` | lane 内 pairwise/Kahan-style 累加，水平合并允许不同顺序 | `mul` 后进入同一累加流程 | 每个实数分量与标量路径结果之差不超过 `max(1 ULP, 2^-52 * |scalar_result|)` |
| `Complex<f32>` | 将 AoS 数据重排为实/虚 lane，分别累加后重组 | 先执行共轭乘法：`conj(lhs_i) * rhs_i`，再分别累加实部与虚部 | 实部/虚部分别满足 `f32` 容差契约 |
| `Complex<f64>` | 将 AoS 数据重排为实/虚 lane，分别累加后重组 | 先执行共轭乘法：`conj(lhs_i) * rhs_i`，再分别累加实部与虚部 | 实部/虚部分别满足 `f64` 容差契约 |

> **复数内积方向说明：** 根据 `require.md` §13，复数 `dot` 的定义必须是 `sum(conj(x_i) * y_i)`，即对左操作数（lhs）取共轭，而不是对右操作数取共轭。SIMD 复数 dot kernel 必须保持这一共轭线性方向。

> **整数归约/内积补充约束：** 本文不再声明 widening accumulation 与 checked 语义等价。对 `i32` / `i64` 的 `sum()` / `dot()`，只有在某个 SIMD kernel 能证明与标量逐步 `checked_add` / `checked_mul` + `checked_add` 完全等价时才允许进入 SIMD；否则必须回退标量路径。

> **FMA 使用约束：** 元素级 `mul()` / `add()` / `dot()` 主循环中的乘法和加法必须按标量表达式顺序分开执行，不得在这些公开语义上隐式启用 FMA，以保持逐元素路径与标量路径的逐位一致。仅在 `sum()` / `dot()` 的内部 reduction merge 已显式声明“允许末位 ULP 差异”的位置，才能在特定 ISA 上使用 FMA 作为局部优化；启用时必须满足本节容差约束。

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
  - 内容: `ScalarKernel<A>` 结构体及整数/浮点/复数 `SimdKernel` 参考实现
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

- [ ] **T4a**: 实现浮点 `sum` SIMD 路径
  - 文件: `src/simd/vector.rs`
  - 内容: 为 `f32`/`f64` 实现 lane accumulation 与 documented merge 的 `SumKernel`
  - 测试: `test_sum_dispatch_simd_float`
  - 前置: T3
  - 预计: 10 min

- [ ] **T4b**: 实现整数 `sum` 与 `dot` 的 checked 语义 SIMD 路径
  - 文件: `src/simd/vector.rs`
  - 内容: 为 `i32`/`i64` 实现仅在可证明等价于标量逐步 checked arithmetic 时启用的 SIMD 前缀/合并路径，否则直接标量回退，并补齐 overflow validation
  - 测试: `test_sum_dispatch_simd_int`、`test_dot_dispatch_simd_int`
  - 前置: T4a
  - 预计: 10 min

- [ ] **T4c**: 实现复数 `sum` SIMD 路径
  - 文件: `src/simd/vector.rs`
  - 内容: 为 `Complex<f32>`/`Complex<f64>` 实现 AoS split real/imag accumulation
  - 测试: `test_sum_dispatch_simd_complex`
  - 前置: T4a
  - 预计: 10 min

- [ ] **T4d**: 实现浮点与复数 `dot` SIMD 路径
  - 文件: `src/simd/vector.rs`
  - 内容: 为 `f32`/`f64`/`Complex<f32>`/`Complex<f64>` 实现 dot kernel 与 conjugate contract
  - 测试: `test_dot_dispatch_simd_float`、`test_dot_dispatch_simd_complex`
  - 前置: T4a, T4c
  - 预计: 10 min

### Wave 3: 集成与条件编译

- [ ] **T5**: 实现 feature gate 条件编译
  - 文件: `src/simd/mod.rs`, `Cargo.toml`
  - 内容: `#[cfg(feature = "simd")]` 条件编译、公开 API 导出、与 tensor 模块的集成接口
  - 测试: 不启用 simd 时编译通过且无 pulp 依赖
  - 前置: T2, T3
  - 预计: 10 min

### Wave 4: 测试与验证

- [ ] **T6a**: 编写逐元素一致性测试
  - 文件: `src/simd/scalar.rs` (#[cfg(test)])
  - 内容: 验证 `add`/`sub`/`mul`/`div` 的 SIMD 与 scalar 逐位一致
  - 测试: `test_simd_scalar_consistency_elementwise`
  - 前置: T5
  - 预计: 10 min

- [ ] **T6b**: 编写 reduction / dot 语义与容差测试
  - 文件: `src/simd/scalar.rs` (#[cfg(test)])
  - 内容: 覆盖浮点、复数与整数回退条件的语义约束及文档化容差边界
  - 测试: `test_simd_scalar_consistency_reduction_dot`
  - 前置: T6a
  - 预计: 10 min

- [ ] **T6c**: 编写随机属性测试
  - 文件: `tests/property/`
  - 内容: 随机输入下验证逐元素一致性与 reduction/dot 不变量
  - 测试: `test_simd_property_consistency`
  - 前置: T6b
  - 预计: 10 min

```
Wave 1: [T1]
           │
           ▼
Wave 2: [T2] [T3]
           │    │
           │    ▼
           │  [T4a]
           │   ├── [T4b]
           │   ├── [T4c]
           │   └── [T4d]
           │
           ▼
Wave 3:   [T5]
           │
           ▼
Wave 4: [T6a] -> [T6b] -> [T6c]
```

---

## 8. 测试计划

### 8.1 测试分类表

| 类型       | 位置                     | 目的                   |
| ---------- | ------------------------ | ---------------------- |
| 单元测试   | `#[cfg(test)] mod tests` | 验证单个 kernel 正确性 |
| 一致性测试 | `src/simd/tests.rs`      | 验证逐元素精确一致与 reduction/dot 容差边界 |
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
| `test_sum_dispatch_simd`       | 浮点/复数 sum 满足条件时进入 SIMD 路径 | 高     |
| `test_dot_dispatch_simd`       | dot 满足条件时进入 SIMD 路径           | 高     |
| `test_simd_scalar_consistency` | 逐元素精确一致，reduction/dot 满足文档容差 | 高     |
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
| SIMD sum(a) 与 scalar sum(a) 满足 §28.3 规定的数值语义/容差         | 随机 `[f64; 1..8192]`、`Complex<f64>` |
| SIMD dot(a, b) 与 scalar dot(a, b) 满足 §28.3 规定的数值语义/容差   | 随机 `[f64; 1..8192]`、`Complex<f64>` |
| tail 处理正确                                                        | `len = n * width + k`, k ∈ [0, width) |

### 8.5 集成测试

| 测试文件             | 测试内容                                                                                  |
| -------------------- | ----------------------------------------------------------------------------------------- |
| `tests/test_simd.rs` | SIMD dispatch 与 `math`、`reduction`、`dot`、`layout`、`parallel` 组合路径的端到端验证 |

### 8.6 Feature gate / 配置测试

| 场景                    | 配置方式                         | 预期行为                               |
| ----------------------- | -------------------------------- | -------------------------------------- |
| 默认关闭                | 默认 feature 集                  | 不编译 SIMD 路径，且无 `pulp` 运行时依赖 |
| 显式启用                | `--features simd`                | SIMD API 可用，满足条件时进入 SIMD 路径 |
| 无硬件能力自动回退      | `--features simd` + 非目标 ISA 环境 | 成功编译运行，并自动回退到标量路径     |

### 8.7 类型边界 / 编译期测试

| 测试类型     | 覆盖内容                                                        | 预期结果                         |
| ------------ | --------------------------------------------------------------- | -------------------------------- |
| 类型边界     | `f32`/`f64`/`i32`/`i64`/`Complex<f32>`/`Complex<f64>` 实现 `SimdElement` | 可编译并进入对应 dispatch 分支   |
| 类型边界     | `bool`/`usize`/其他 `Complex<T>` 不实现 `SimdElement`                   | 编译期拒绝直接使用 SIMD kernel   |
| 编译期测试   | `#[cfg(feature = "simd")]` / `#[cfg(not(feature = "simd"))]` 两侧 API | 两种 feature 组合均可编译        |
| 编译期测试   | 未覆盖数学函数与非 sum reduction kernel                         | 保持公开 API 可用且落入标量路径  |

---

## 9. 与其他模块的交互

### 9.1 接口约定

| 方向 | 对方模块 | 接口/类型 | 契约 |
| ---- | -------- | --------- | ---- |
| 消费（输入） | `tensor` | `TensorBase<S, D>::as_slice()` | 只消费满足 `07-tensor.md` 连续切片契约的输入；非连续或广播视图回退标量 |
| 消费（输入） | `math` / `reduction` / `dot` | `dispatch_binary_op()` / `SimdKernel::sum()` / `SimdKernel::dot()` | 上层语义模块先完成形状和类型裁决，再传入兼容切片 |
| 消费（输入） | `layout` | 对齐/连续性元数据 | 当前版本仅对统一对齐快路径启用 SIMD，其余情况自动回退标量 |
| 产出（输出） | 上层语义模块 | 标量结果或写回目标切片 | 不改变公开 API 形状、错误类别和数值语义边界 |

`math/` 模块在执行逐元素运算时，只在输入已经提取为兼容连续切片且满足当前统一对齐快路径时进入 SIMD，否则使用 `ScalarKernel`（参见 `11-math.md §5.3`）。

> **分派策略**: `math` 模块提供公共 `sqrt()` API，但在当前版本中 `sqrt()` 不接入 SIMD kernel，而是保持标量路径。用户仅调用 `math::sqrt()`，无需关心底层是否存在加速能力。

### 9.2 数据流描述

```
math/reduction/dot call acceleration entry
    │
    ├── check feature + contiguous slice contract + unified alignment fast path
    │       ├── Check feature = simd
    │       ├── Check F-order contiguity via tensor contract
    │       ├── Check whether element type implements SimdElement
    │       └── Check aligned fast-path preconditions
    │
    ├── YES -> get_arch().dispatch(VectorKernel)
    │
    └── NO  -> ScalarKernel fallback for element-wise/reduction/dot
```

### 9.3 与 parallel 模块

并行路径的每个工作线程内部可以使用 SIMD。组合使用时：先按并行阈值分块到各线程，线程内部再检查 SIMD 条件执行向量化（参见 `09-parallel.md §6.1`）。

### 9.4 与 storage/layout 模块

SIMD 模块依赖 layout 提供的连续性和对齐信息来判断是否可以使用 SIMD 路径（参见 `06-layout.md` §5.5）。

---

## 10. 错误处理与语义边界

| 类型                 | 说明                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------ |
| Recoverable error    | 无专属 recoverable error；SIMD 不可用、类型不支持、未满足统一对齐快路径或当前 ISA 无法满足语义约束时均自动回退标量路径 |
| Panic                | 切片长度不一致；整数 `sum` / `dot` 溢出；违反 kernel 前置条件的内部 bug |
| 路径一致性           | 逐元素 SIMD 路径保持公开语义一致；整数 `sum` / `dot` 与标量 checked arithmetic 精确一致；浮点/复数归约与内积允许已文档化容差 |
| 容差边界             | integer: exact; for float/complex reduction and dot, each real-valued component may differ from the scalar result by at most `max(1 ULP, 2^-23 * |scalar_result|)` for `f32` or `max(1 ULP, 2^-52 * |scalar_result|)` for `f64`, per `require.md` §9.3 / §28.3 |

### 10.1 非有限值容差规则

- `NaN`：按 IEEE 754 语义检查（`NaN !=` 任何值），不使用数值容差。
- `±Inf`：必须同号同类。
- `+0.0` / `-0.0`：符号必须一致。
- 容差规则仅适用于有限值结果。

### 10.2 线程安全

SIMD 后端不改变 `TensorBase<S, D>` 的 `Send` / `Sync` 判定。线程安全性仍由元素类型与存储模式共同决定（参见 `25-safety.md`）。

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
| 决策     | SIMD 只覆盖本文列出的子集以及 `sum` / `dot`；整数 `sum` / `dot` 只有在可证明保持 checked arithmetic 语义时才使用 SIMD，浮点/复数 reduction / dot 允许文档化容差 |
| 理由     | `require.md` §9.1 / §9.3 / §28.3 明确要求公开能力可由 SIMD 加速实现，但不能破坏公开语义；因此整数路径只在可证明保持 checked arithmetic 语义时使用 SIMD，否则直接回退标量；浮点/复数路径则采用受控的累加顺序变化与显式容差记录。 |
| 替代方案 | 所有归约/内积一律回退标量 — 放弃，与需求文档冲突                                                                                           |
| 替代方案 | 允许任意差异 — 放弃，无法保证正确性                                                                                                          |

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

> 说明：下表中的 `sum` / `dot` SIMD 数据代表当前版本目标实现。是否进入 SIMD 仍取决于连续性、对齐、ISA 能力与本设计文档定义的语义约束。

| 操作          | 标量 | AVX2 (f64) | 加速比 | 标量 | AVX2 (f32) | 加速比 |
| ------------- | ---- | ---------- | ------ | ---- | ---------- | ------ |
| add (1M 元素) | ~2ms | ~0.5ms     | ~4x    | ~2ms | ~0.25ms    | ~8x    |
| sum (1M 元素) | ~1ms | ~0.3ms     | ~3.3x  | ~1ms | ~0.15ms    | ~6.7x  |
| dot (1M 元素) | ~2ms | ~0.5ms     | ~4x    | ~2ms | ~0.3ms     | ~6.7x  |

### 12.3 性能影响因素

| 方面       | 设计决策                                     |
| ---------- | -------------------------------------------- |
| 向量化宽度 | pulp 运行时自动选择最优宽度，无需编译期配置  |
| 内存对齐   | 当前版本仅对统一对齐快路径启用 SIMD；未满足该条件时回退标量路径 |
| 尾部处理   | 标量循环处理尾部，简单安全                   |
| 循环展开   | pulp 内部处理，无需手动展开                  |
| FMA 利用   | 不用于逐元素 `mul`/`add`；仅可在已文档化容差的 reduction merge 中受控使用 |

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
| 1.2.0 | 2026-04-14 |
| 1.2.1 | 2026-04-15 |
| 1.2.2 | 2026-04-15 |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
