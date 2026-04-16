# SIMD 后端模块设计

> 文档编号: 08 | 模块: `src/simd/` | 阶段: Phase 5
> 前置文档: `03-element.md`, `06-layout.md`, `07-tensor.md`
> 需求参考: `需求说明书 §9.1`, `需求说明书 §9.3`, `需求说明书 §13`, `需求说明书 §27`, `需求说明书 §28.3`, `需求说明书 §28.4`, `需求说明书 §28.5`
> 范围声明: 范围内

---

## 1. 模块定位

SIMD 后端模块是 Xenon 张量库的可选性能加速层，通过 `pulp` crate 提供跨平台 SIMD 抽象。根据 `需求说明书 §9.1`，当前版本以算术运算、`neg`、`sum` 与 **vector dot** 为优先，并对复数逐元素算术提供 SIMD 支持；比较、`abs`、`square` 与其他数学函数按阶段推进，完整覆盖计划见 §5.4a。该模块默认关闭，通过 `features = ["simd"]` 启用。

### 1.1 职责边界

| 职责       | 包含                                                                | 不包含                                              |
| ---------- | ------------------------------------------------------------------- | --------------------------------------------------- |
| SIMD 抽象  | 通过 pulp 统一抽象 x86/ARM 指令集                                   | GPU 加速 (CUDA/OpenCL)                              |
| 逐元素运算 | `SimdKernel` 直接覆盖二元算术与 `neg`；比较、`abs`、`square`、`not` 等由独立专用 kernel 路径承载 | 未列出的逐元素运算与其它数学函数专用 kernel |
| 归约运算   | `sum` 的 SIMD 加速与向量化可用性约束；整数路径仅在已验证 ISA widening 实现存在时启用 | `max`/`argmax` 等其他归约                           |
| 内积运算   | `dot` 的 SIMD 加速与向量化可用性约束；整数路径仅在已验证 ISA widening 实现存在时启用 | 外积、矩阵乘法                                      |
| 运行时分发 | Arch 检测缓存、自动最优路径选择                                     | 编译期静态分发                                      |

> 标量回退不再由 `simd/` 模块承担；各语义模块在未进入 SIMD/并行路径时，使用各自的串行实现作为公开语义基线。

### 1.2 设计原则

| 原则         | 体现                                                                                                                                                   |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 结果一致性   | 逐元素路径要求保持公开语义一致；整数 `sum` / `dot` 必须等价于逐步 checked arithmetic，浮点/复数 `sum` / `dot` 允许在已文档化容差内偏离标量累加顺序结果 |
| 自动分层     | 路径选择由内部 `dispatch.rs` 负责；未满足 SIMD 条件时回到各语义模块的串行实现，用户无需感知                                                            |
| 零成本抽象   | 未启用 `simd` feature 时无任何运行时开销                                                                                                               |
| 跨平台       | pulp 统一 x86_64 (SSE4.1/AVX2/AVX-512) 和 ARM (NEON)；SVE 属于后续扩展，不纳入当前设计                                                                 |
| 精度优先约束 | 元素级 `mul`/`add` 不使用 FMA 以保持逐位一致；仅在已记录容差的 reduction merge 内部可使用                                                              |

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
│             dispatch.rs path selection layer                    │
│   Choose serial / parallel / SIMD by len/contiguity/alignment   │
└──────────┬──────────────┬───────────────────────────────────────┘
           │              │
           ▼              ▼
    ┌──────────┐   ┌──────────┐
    │ SIMD path│   │ Other    │
    │(this mod)│   │ backends │
    └────┬─────┘   └──────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                Pure vectorized hardware exec                    │
│                AVX-512 / AVX2 / SSE4.1 / NEON                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 需求映射与范围约束

| 类型     | 内容                                                                                                                                    |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| 需求映射 | `需求说明书 §9.1`、`需求说明书 §9.3`、`需求说明书 §13`、`需求说明书 §27`、`需求说明书 §28.3`、`需求说明书 §28.4`、`需求说明书 §28.5` |
| 范围内   | 本文覆盖矩阵中列出的 SIMD 子集：`add`/`sub`/`mul`/`div`、`neg`、`sum` / `dot`，以及复数逐元素算术；未列出能力由各语义模块的串行实现承担 |

> **SIMD 覆盖范围**：当前版本正式承诺的 SIMD 覆盖以逐元素算术、归约（sum）和内积（dot）为优先。其他运算（一元、比较、逻辑、复数、数学函数）的 SIMD 路径作为实现优化项，不构成当前版本的稳定交付承诺。

| 范围外   | 矩阵乘法、非 `sum` 归约、归约模块负责的 `norm` 等专用 SIMD kernel，以及 SVE 当前版本实现                                                |
| 标量回退 | `simd/` 模块内不再提供标量回退；当 SIMD 不可用时，由 `dispatch.rs` 回到各语义模块的串行实现                                             |
| 非目标   | No new third-party dependencies beyond pulp；在本模块内提供标量回退路径——该职责由各语义模块的串行实现承担                               |

---

## 3. 文件位置

```
src/simd/
├── mod.rs             # pulp integration, Arch cache, internal dispatch facade, internal kernel traits
└── vector.rs          # vectorized implementation based on pulp WithSimd
```

单文件职责划分：`mod.rs` 负责 pulp 集成、Arch 缓存与内部 trait 定义，`vector.rs` 封装所有基于 `pulp::WithSimd` 的纯向量化逻辑。

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

| 项目           | 说明                       |
| -------------- | -------------------------- |
| 新增第三方依赖 | `pulp`                     |
| 合法性结论     | 符合需求说明书最小依赖限制 |
| 替代方案       | 不适用                     |

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

> **内部边界说明：** `simd/` 中出现的 trait、enum 与函数签名仅用于说明后端实现分层；所有 SIMD 类型与入口都属于内部实现细节，不构成稳定公开 API，对外导出层级统一按 `pub(crate)` 理解。

### 5.2 SimdElement Trait

```rust,ignore
// src/simd/mod.rs

use crate::complex::Complex;

/// Marker trait for SIMD kernel element types.
///
/// Distinguishes different element types at compile time for SIMD dispatch.
/// Only implemented for numeric types that support SIMD operations.
pub(crate) trait SimdElement: Copy + Clone + Send + Sync + 'static {
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
> | 类型              | 原因                                                                                                           |
> | ----------------- | -------------------------------------------------------------------------------------------------------------- |
> | `bool`            | 当前版本不进入数值 SIMD kernel；`not` 通过独立 bool / mask SIMD kernel 覆盖，因此不复用本 trait 的数值类型边界 |
> | `usize`           | 指针宽度依赖平台，SIMD 语义不稳定                                                                              |
> | 其他 `Complex<T>` | 仅 `Complex<f32>` / `Complex<f64>` 在当前设计中进入专用 kernel                                                 |
>
> **复数 SIMD 策略：** `Complex<f32>` 和 `Complex<f64>` 在当前设计中优先覆盖逐元素 `add` / `sub` / `mul` / `div`、`neg()`、`sum()` 与 `dot()`。实现上采用 AoS 输入 + 寄存器内重排，按实部/虚部分离后执行
> 向量累加与复数乘加，避免引入额外的布局转换 API。

### 5.3 SimdKernel Trait

```rust,ignore
// src/simd/mod.rs

/// SIMD kernel trait.
///
/// Defines the unified interface for SIMD kernels; callers use it generically.
///
/// # Type Parameters
///
/// * `A` - Element type, must implement `SimdElement`
pub(crate) trait SimdKernel<A: Copy + Send + Sync + 'static>: Send + Sync {
    // ========================================
    // Metadata
    // ========================================

    /// Kernel name (for debugging and logging).
    fn name() -> &'static str where Self: Sized;

    /// Returns the SIMD width (number of elements) for this kernel.
    ///
    /// Vectorized kernels return the actual SIMD width.
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

> 方案 D 中保留 `SimdKernel` trait 作为向量化 kernel 的统一接口；当前仅 `VectorKernel` 系列实现该 trait，`ScalarKernel` 已移除。

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
pub(crate) fn get_arch() -> Arch {
    static ARCH: std::sync::OnceLock<Arch> = std::sync::OnceLock::new();
    *ARCH.get_or_init(Arch::new)
}

/// Placeholder when SIMD is disabled.
#[cfg(not(feature = "simd"))]
pub(crate) fn get_arch() -> () {
    ()
}

/// Stable vectorized entry used after internal dispatch has selected SIMD.
///
/// `math/`, `matrix::dot`, and `reduction/` first go through their internal
/// `dispatch.rs` path selection, extract compatible contiguous slices, and only
/// then call into the pure vectorized backend exposed by `simd/`.
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

pub(crate) fn dispatch_vector_binary_op<A>(op: BinaryOp, lhs: &[A], rhs: &[A], dst: &mut [A])
where
    A: SimdElement + Numeric + Copy;
```

> `can_use_simd` 已迁移至内部 `dispatch.rs` 模块（`can_use_simd`）。`simd/` 模块只提供纯向量化执行能力。
>
> **职责边界说明：** SIMD 内核不自行裁决回退。`dispatch.rs` 根据 target feature、元素类型、操作种类、连续性、对齐与输入特征决定是否调用 SIMD 路径。SIMD 内核的前提条件由 `dispatch.rs` 保证。

### 5.4a 当前版本的 SIMD 覆盖范围

根据 `require.md` §9.1，SIMD 路径当前已覆盖逐元素运算、归约与内积三个大类。是否实际进入 SIMD 仍取决于元素类型、ISA 能力、统一对齐快路径与语义约束；当前版本在 `matrix` 相关范围内仅承载 **vector dot**，不展开矩阵乘法或其他 matrix 范围能力。

> **SIMD 覆盖范围**：当前版本正式承诺的 SIMD 覆盖以逐元素算术、归约（sum）和内积（dot）为优先。其他运算（一元、比较、逻辑、复数、数学函数）的 SIMD 路径作为实现优化项，不构成当前版本的稳定交付承诺。

> **透明回退说明：** 对于下表中尚未提供 SIMD kernel 的操作，或运行时不满足 SIMD 入口条件的输入，`dispatch.rs` 会透明回退到对应语义模块的标量/串行路径；公开 API 与结果语义保持不变。

#### SIMD 操作覆盖状态表

| 操作                                             | 类型                                                            | 状态（已实现/规划中/标量回退）                            | 说明                                                                                                                                           |
| ------------------------------------------------ | --------------------------------------------------------------- | --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `add` / `sub` / `mul` / `div`                    | `f32` / `f64` / `Complex<f32>` / `Complex<f64>`                 | 已实现                                                    | 当前版本提供 SIMD kernel                                                                                                                       |
| `neg`                                            | `f32` / `f64` / `Complex<f32>` / `Complex<f64>`                 | 已实现                                                    | 当前版本提供 SIMD kernel                                                                                                                       |
| `sign` / `signum`                                | `f32` / `f64`                                                   | 本版覆盖（通过 SIMD 比较 + blend）                        | 当前版本仅为实数浮点提供 SIMD 路径；其他类型保持标量/串行路径                                                                                  |
| `sum`                                            | `f32` / `f64` / `Complex<f32>` / `Complex<f64>`                 | 已实现                                                    | 浮点/复数遵循本文档记录的归约容差                                                                                                              |
| `sum`                                            | `i32` / `i64`                                                   | 条件实现，默认标量回退                                    | 整数 `sum` 默认走串行 checked 路径；仅在存在已验证的 ISA 专用 widening SIMD 实现时启用（例如 `i32 -> i64`），且必须与标量 checked 语义完全等价 |
| `dot`                                            | `f32` / `f64` / `Complex<f32>` / `Complex<f64>`                 | 已实现                                                    | 复数 `dot` 采用 `sum(conj(lhs_i) * rhs_i)`                                                                                                     |
| `dot`                                            | `i32` / `i64`                                                   | 条件实现，默认标量回退                                    | 整数 `dot` 默认走串行 checked 路径；仅在存在已验证的 ISA 专用 widening SIMD 实现时启用（例如 `i32 -> i64`），且必须与标量 checked 语义完全等价 |
| `abs`                                            | `f32` / `f64`                                                   | 本版覆盖（通过 SIMD 算术路径或专用 kernel）               | 通过专用一元 kernel 路径或共享装载/收尾框架实现，不经 `SimdKernel` 二元算术 trait                                                              |
| `square`                                         | `i32` / `i64` / `f32` / `f64` / `Complex<f32>` / `Complex<f64>` | 本版覆盖（通过 SIMD 算术路径或专用 kernel）               | 通过专用一元 kernel 路径或复用乘法/分量级 kernel，不经 `SimdKernel` 二元算术 trait                                                             |
| `eq` / `ne` / `lt` / `gt`                        | 适用的整数 / 浮点类型                                           | 本版覆盖（通过专用比较 kernel 路径）                      | 比较 kernel 将布尔结果写入 `bool` 目标缓冲区，不经 `SimdKernel` 二元算术 trait                                                                 |
| `sin` / `sqrt` / `exp` / `ln` / `floor` / `ceil` | `f32` / `f64`                                                   | 标量回退 + 可选 SIMD 加速                                 | 数学函数的 SIMD 实现依赖平台 libm 或手动实现，精度约束见 §5.5                                                                                  |
| `not`                                            | `bool`                                                          | 本版覆盖（单指令 SIMD not）                               | 通过独立 bool / mask kernel 实现，不经 `SimdKernel` 二元算术 trait                                                                             |
| `complex_abs` / `conjugate`                      | `Complex<f32>` / `Complex<f64>`                                 | 分量级 SIMD 加速（算术路径覆盖），模/共轭通过分量组合实现 | 通过专用一元/复数 kernel 路径实现；复数 AoS 输入在寄存器内重排后执行，不经 `SimdKernel` 二元算术 trait                                         |

> **`SimdKernel::dot()` 说明：** `dot()` 为当前版本正式能力；`SimdKernel` 直接覆盖 `f32` / `f64` / `Complex<f32>` / `Complex<f64>`。整数 `dot` 仅在存在已验证的 ISA 专用 widening 实现时才进入 SIMD，否则保持语义模块串行路径。
>
> **注意**：`i64` 类型的 SIMD 归约和内积路径在当前版本默认不提供，除非存在已验证的 widening ISA 实现。不存在时回退到标量路径。

该覆盖目标不改变公开 API 的可用性；SIMD 仅影响执行路径选择，不改变公开语义契约。

### 5.4b SIMD Path Selection Thresholds

`dispatch.rs` 在尝试进入 SIMD 前，至少按以下规则做统一裁决：

1. **Minimum length**：输入长度必须达到对应操作的最小向量化阈值，避免短切片因装载/收尾成本高于收益而误入 SIMD。
2. **Alignment requirement**：参与运算的切片必须满足 Xenon 统一对齐快路径要求（通过 `layout::is_aligned()` 一类检查判定）；不满足时直接回退标量/串行路径。
3. **ISA width check**：运行时需确认当前 `pulp::Arch` 上存在可用 ISA 且 lane 宽度大于 1；若目标类型或当前 ISA 无法提供有效向量宽度，则不进入 SIMD。

| 操作类型                  | 元素类型                        | SIMD 最小长度 | 说明                                                         |
| ------------------------- | ------------------------------- | ------------- | ------------------------------------------------------------ |
| 逐元素算术                | `f32` / `f64`                   | 64            | 对齐后向量宽度                                               |
| 逐元素算术                | `i32` / `i64`                   | 64            | 同上                                                         |
| 逐元素算术                | `Complex<f32>` / `Complex<f64>` | 128           | AoS 输入需寄存器内重排，默认阈值高于实数路径                 |
| 比较                      | 适用的整数 / 浮点类型           | 64            | 与逐元素算术共享向量装载/收尾框架                            |
| `abs` / `sign` / `signum` | `f32` / `f64`                   | 64            | 一元实数路径，通常复用比较/位运算或算术框架                  |
| `square`                  | `i32` / `i64` / `f32` / `f64`   | 64            | 复用逐元素乘法 kernel                                        |
| `square`                  | `Complex<f32>` / `Complex<f64>` | 128           | 复用 complex 算术路径与寄存器重排框架                        |
| `bool` (`not`)            | `bool`                          | N/A           | 由独立 bool / mask kernel 决定，不在统一数值阈值表内单独承诺 |
| 归约 `sum`                | `f32` / `f64`                   | 1024          | 归约开销较高，需更大输入                                     |
| 归约 `sum`                | `i32` / `i64`                   | 512           | widening accumulator                                         |
| 内积 `dot`                | `f32` / `f64`                   | 512           | 同归约                                                       |
| 内积 `dot`                | `i32` / `i64`                   | 256           | 同上                                                         |

以上阈值为默认值，可通过 dispatch 配置调整。当输入长度低于阈值时，自动回退到标量路径。

这些阈值只影响执行路径选择，不改变 API 结果；未通过任一条件时，系统透明回退到非 SIMD 路径。

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
│ dispatch.rs already selected SIMD path                          │
│ • F-order contiguous memory                                     │
│ • Supported element type / op                                   │
│ • Current unified-alignment fast path enabled                   │
│ • feature = "simd" enabled                                     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ SIMD main loop  │
                 │ width = S::len()│
                 │ for chunk:      │
                 │   load + op +   │
                 │   store         │
                 ├─────────────────┤
                 │ Scalar tail     │
                 │ [chunks*W..N)   │
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

- 轴向 `sum_axis(axis)` / `sum_axis_keepdims(axis)` 的 SIMD 路径仅在被归约维对应连续内层维度时启用。
- 若输入布局非连续，或目标轴无法映射为连续内层归约区间，则由对应语义模块保持串行路径。
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
│ dispatch.rs already selected SIMD path + type-specific contract │
│ • f32/f64: SIMD mul + lane-local accumulation                   │
│ • Complex: `conj(lhs)`-multiply-add on split real/imag lanes    │
│ • Integer: only enter SIMD when checked multiply/add semantics  │
│   can be preserved                                              │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ SIMD main loop + scalar tail + documented merge                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.6 条件检查与自动回退

```rust,ignore
// internal dispatch.rs

// `can_use_simd` has moved to the internal dispatch layer.
// The decision still checks feature gate, contiguous layout, supported type,
// unified alignment fast path, and runtime vector width before entering
// `simd/`.
```

SIMD 条件判断已迁移至内部 `dispatch.rs` 模块（`can_use_simd`）。`simd/` 模块只提供纯向量化执行能力。上层仍按统一规则检查 feature、连续性、对齐、元素类型与运行时可用 lane 宽度后，才进入 `simd/`。SIMD 内核本身不再重复执行回退裁决。

### 5.6a `dispatch.rs` 能力查询接口

```rust,ignore
// src/dispatch.rs -> src/simd/mod.rs

pub(crate) enum SimdOp {
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Sum,
    Dot,
    Abs,
    Square,
    Eq,
    Ne,
    Lt,
    Gt,
    Not,
    ComplexAbs,
    Conjugate,
}

pub(crate) enum SimdElementKind {
    Bool,
    F32,
    F64,
    I32,
    I64,
    ComplexF32,
    ComplexF64,
}

pub(crate) fn is_supported(element_type: SimdElementKind, op: SimdOp) -> bool;
```

- `dispatch.rs` 通过该接口查询“某元素类型 + 某操作”是否存在 SIMD 实现资格。
- 资格查询只回答“当前版本是否存在可进入的 SIMD 能力”，不替代连续性、对齐、长度阈值和 ISA 检查。
- 对状态为“规划中”或“标量回退”的条目，`is_supported(...)` 必须返回 `false`。

### 5.7 Good/Bad 对比示例

```rust,ignore
// Good - use the public semantic entry; dispatch selects serial / parallel / SIMD.
let out = lhs.add(&rhs)?;

// Bad - calling the SIMD backend directly bypasses dispatch.rs path selection.
// FORBIDDEN: internal kernels are not a public integration surface.
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
        assert_eq!(self.rhs.len(), len);
        assert_eq!(self.dst.len(), len);

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
pub(crate) fn add_f32_simd(lhs: &[f32], rhs: &[f32], dst: &mut [f32]) {
    let arch = crate::simd::get_arch();
    arch.dispatch(AddF32Kernel { lhs, rhs, dst });
}
```

### 6.2 标量回退实现

ScalarKernel（标量回退）已在方案 D 中移除。标量路径由各语义模块的串行实现承担。

### 6.2a `unsafe` 健全性边界

SIMD 内核中的 `unsafe` 只允许用于底层 load/store 与寄存器装载；其健全性前提必须由 `dispatch.rs` 和调用点共同保证：

| 主题            | 必须满足的前提                                                                                                                                                       |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 对齐要求        | 若内核调用对齐 load/store 变体，输入与输出指针必须满足 Xenon 统一对齐快路径要求；若只满足非对齐访问语义，则必须改用相应的 unaligned load/store 变体或直接不进入 SIMD |
| load/store 安全 | 对任一主循环迭代，`offset + width <= len`；`lhs`、`rhs`、`dst` 的切片长度已经过调用侧验证；`dst` 可写区间与本次 store 范围完全重合且不越界                           |
| 尾部处理不变量  | 向量主循环仅覆盖 `[0, chunks * width)`；标量尾部只覆盖 `[chunks * width, len)`；两段区间不重叠且并集恰好等于完整输入区间                                             |

若上述任一条件无法证明成立，则该实现不得进入 `unsafe` SIMD 主循环。

### 6.3 dispatch 流程

```
Dispatch call flow

┌─────────────────────────────────────────────────────────────────┐
│        dispatch.rs has already admitted the SIMD execution       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│         Arch runtime detection (cached)                         │
│   Check order: AVX-512 -> AVX2 -> SSE4.1 -> NEON                │
└─────────────────────────┬───────────────────────────────────────┘
                          │
           ┌──────────────┼──────────────┐
           │              │              │
           ▼              ▼              ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │AVX-512   │   │AVX2      │   │SSE4.1/   │
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

> **设计决策：** 对于逐元素运算，SIMD 与标量路径必须保持相同公开语义；当前版本仅对 §5.4a 表中“已实现”或“本版覆盖”的条目提供对应 SIMD kernel 路径，其中比较、一元与 bool 操作走独立专用 kernel，不经 `SimdKernel` 二元算术 trait。对于归约和内积，当前版本仅对已验证的 SIMD 覆盖子集启用向量化：`f32` / `f64` / `Complex<f32>` / `Complex<f64>` 为正式 SIMD 覆盖，`i32` / `i64` 仅在存在已验证的 ISA widening 实现时才进入 SIMD，否则默认回到串行 checked 基线，并继续遵循 `require.md` 中对应数值约束。容差与比较规则统一遵循 `00-coding.md §7.4` 的定义。同执行路径基础算术/比较默认精确一致；仅跨路径比较和数学函数比较允许使用文档化容差。
>
> **一致性说明：** 对于逐元素操作（add、mul 等），SIMD 和标量路径产生逐位一致的结果。
> 对于归约/内积操作，Xenon 不接受未记录的“近似一致”；若某个 SIMD 内核无法满足文档定义的数值语义与容差边界，则不走 SIMD 路径。
>
> **覆盖说明：** 当前版本只覆盖 §5.4a 表中列出的“已实现”“本版覆盖”与“条件实现，默认标量回退”子集；其中整数 `sum` / `dot` 默认仍回到串行 checked 路径，只有在存在已验证 ISA widening 实现时才实际进入 SIMD。其余条目由 `dispatch.rs` 回到对应语义模块的标量/串行路径。数学函数作为后续增强目标逐步补齐；这不改变 `math` 模块对这些逐元素运算的公开 API 承诺。完整覆盖计划见 §5.4a。

### 6.5 SIMD reduction / dot 内部策略

| 类型           | `sum()` SIMD 策略                                                                                        | `dot()` SIMD 策略                                                                                                        | 精度/溢出约束                                                                                 |
| -------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- |
| `i32`          | 默认走串行 `checked_add`；仅在存在已验证的 ISA 专用 widening SIMD 实现时，才可使用 `i32 -> i64` 中间累加 | 默认走串行 `checked_mul` + `checked_add`；仅在存在已验证的 ISA 专用 widening SIMD 实现时，才可使用 `i32 -> i64` 中间累加 | 与标量整数语义精确一致；无法维持等价时不得进入 SIMD                                           |
| `i64`          | 默认走串行 `checked_add`；当前版本不泛化承诺 widening SIMD                                               | 默认走串行 `checked_mul` + `checked_add`；当前版本不泛化承诺 widening SIMD                                               | 与标量整数语义精确一致；无法维持等价时不得进入 SIMD                                           |
| `f32`          | lane 内 pairwise/Kahan-style 累加，水平合并允许不同顺序                                                  | `mul` 后进入同一累加流程                                                                                                 | 容差与比较规则统一遵循 `00-coding.md §7.4` 的定义；同执行路径基础算术/比较默认精确一致；仅跨路径比较和数学函数比较允许使用文档化容差 |
| `f64`          | lane 内 pairwise/Kahan-style 累加，水平合并允许不同顺序                                                  | `mul` 后进入同一累加流程                                                                                                 | 容差与比较规则统一遵循 `00-coding.md §7.4` 的定义；同执行路径基础算术/比较默认精确一致；仅跨路径比较和数学函数比较允许使用文档化容差 |
| `Complex<f32>` | 将 AoS 数据重排为实/虚 lane，分别累加后重组                                                              | 先执行共轭乘法：`conj(lhs_i) * rhs_i`，再分别累加实部与虚部                                                              | 容差与比较规则统一遵循 `00-coding.md §7.4` 的定义；同执行路径基础算术/比较默认精确一致；仅跨路径比较和数学函数比较允许使用文档化容差 |
| `Complex<f64>` | 将 AoS 数据重排为实/虚 lane，分别累加后重组                                                              | 先执行共轭乘法：`conj(lhs_i) * rhs_i`，再分别累加实部与虚部                                                              | 容差与比较规则统一遵循 `00-coding.md §7.4` 的定义；同执行路径基础算术/比较默认精确一致；仅跨路径比较和数学函数比较允许使用文档化容差 |

> **容差说明：** 容差与比较规则统一遵循 `00-coding.md §7.4` 的定义。同执行路径基础算术/比较默认精确一致；仅跨路径比较和数学函数比较允许使用文档化容差。

> **复数内积方向说明：** 根据 `require.md` §13，复数 `dot` 的定义必须是 `sum(conj(x_i) * y_i)`，即对左操作数（lhs）取共轭，而不是对右操作数取共轭。SIMD 复数 dot kernel 必须保持这一共轭线性方向。

> **整数归约/内积补充约束：** 对 `i32` / `i64` 的 `sum()` / `dot()`，当前默认路径是串行 checked arithmetic。仅当某个 ISA 专用 widening SIMD 实现（例如 `i32 -> i64`）已被验证与标量逐步 `checked_add` / `checked_mul` + `checked_add` 完全等价时，才允许启用 SIMD；`i64` 不做泛化 widening 承诺。
>
> **注意**：`i64` 类型的 SIMD 归约和内积路径在当前版本默认不提供，除非存在已验证的 widening ISA 实现。不存在时回退到标量路径。

> **FMA 使用约束：** 元素级 `mul()` / `add()` / `dot()` 主循环中的乘法和加法必须按标量表达式顺序分开执行，不得在这些公开语义上隐式启用 FMA，以保持逐元素路径与标量路径的逐位一致。仅在 `sum()` / `dot()` 的内部 reduction merge 已显式声明“允许末位 ULP 差异”的位置，才能在特定 ISA 上使用 FMA 作为局部优化；启用时必须满足本节容差约束。

> **ISA 检测补充：** `dispatch.rs` 与 `pulp::Arch` 的主分支检测顺序按 `AVX-512 -> AVX2 -> SSE4.1 -> NEON` 组织。FMA 可用性若需要利用，必须作为独立能力位单独检测，不得把 `AVX2` 与 `FMA` 隐式绑定成同一准入条件。

---

## 7. 实现任务拆分

### Wave 1: 基础设施

- [ ] **T1**: 创建 `src/simd/mod.rs` 骨架
  - 文件: `src/simd/mod.rs`
  - 内容: 模块声明、`SimdElement` trait、`SimdKernel` trait 定义、Arch 缓存、纯向量化公开入口
  - 测试: 编译通过
  - 前置: 无
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
  - 内容: 为 `i32`/`i64` 实现仅在可证明等价于标量逐步 checked arithmetic 时启用的 SIMD 前缀/合并路径；无法满足条件时由语义模块保持串行路径，并补齐 overflow validation
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
  - 内容: `#[cfg(feature = "simd")]` 条件编译、公开 API 导出、与内部 `dispatch.rs` 的集成接口
  - 测试: 不启用 simd 时编译通过且无 pulp 依赖
  - 前置: T1, T3
  - 预计: 10 min

### Wave 4: 测试与验证

- [ ] **T6a**: 编写逐元素一致性测试
  - 文件: `src/simd/vector.rs` (#[cfg(test)])
  - 内容: 验证 `add`/`sub`/`mul`/`div` 的 SIMD kernel 与语义模块串行基线逐位一致
  - 测试: `test_simd_vector_consistency_elementwise`
  - 前置: T5
  - 预计: 10 min

- [ ] **T6b**: 编写 reduction / dot 语义与容差测试
  - 文件: `src/simd/vector.rs` (#[cfg(test)])
  - 内容: 覆盖浮点、复数与整数 SIMD 入口条件的语义约束及文档化容差边界
  - 测试: `test_simd_vector_semantics_reduction_dot`
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
Wave 2: [T3]
           │
           ▼
         [T4a]
          ├── [T4b]
          ├── [T4c]
          └── [T4d]
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

| 类型       | 位置                     | 目的                                        |
| ---------- | ------------------------ | ------------------------------------------- |
| 单元测试   | `#[cfg(test)] mod tests` | 验证单个 kernel 正确性                      |
| 一致性测试 | `src/simd/tests.rs`      | 验证逐元素精确一致与 reduction/dot 容差边界 |
| 边界测试   | 集成测试中标注           | 空数组、单元素、非对齐                      |
| 属性测试   | `tests/property/`        | 随机数据验证不变量                          |

### 8.2 单元测试清单

| 测试函数                 | 测试内容                               | 优先级 |
| ------------------------ | -------------------------------------- | ------ |
| `test_vector_add_f32`    | SIMD f32 加法正确性                    | 高     |
| `test_vector_add_f64`    | SIMD f64 加法正确性                    | 高     |
| `test_sum_dispatch_simd` | 浮点/复数 sum 满足条件时进入 SIMD 路径 | 高     |
| `test_dot_dispatch_simd` | dot 满足条件时进入 SIMD 路径           | 高     |
| `test_tail_handling`     | 非宽度整数倍数组尾部处理               | 中     |
| `test_empty_array`       | 空数组不 panic                         | 中     |
| `test_single_element`    | 单元素数组正确处理                     | 中     |
| `test_misaligned_ptr`    | 非对齐数据回退到标量                   | 中     |

> SIMD 路径与各语义模块串行实现的一致性测试，由各语义模块的测试计划覆盖。

### 8.3 边界测试场景

| 场景                      | 预期行为                          |
| ------------------------- | --------------------------------- |
| 空数组 `len=0`            | 立即返回，不 panic                |
| 单元素 `len=1`            | 由 `dispatch.rs` 保持非 SIMD 路径 |
| 短数组 `len < SIMD_WIDTH` | 由 `dispatch.rs` 保持非 SIMD 路径 |
| 非对齐数据                | 由 `dispatch.rs` 保持非 SIMD 路径 |
| 非 F-order 连续           | 由 `dispatch.rs` 保持非 SIMD 路径 |
| `len = SIMD_WIDTH`        | 恰好一个 SIMD 块，无尾部          |
| `len = SIMD_WIDTH + 1`    | 一个 SIMD 块 + 1 个标量尾部       |

### 8.3a `需求说明书 §28.4` 边界占位

- [ ] 补充 `min_positive` / `max_finite` / `subnormal` 输入的 SIMD 与串行一致性边界测试。
- [ ] 补充复数实部/虚部跨 `±0.0`、`±Inf`、`NaN` 组合时的归约与内积边界测试。
- [ ] 补充尾部长度与 lane 宽度交错组合下的 load/store 安全性回归测试。

### 8.4 属性测试不变量

| 不变量                                                              | 测试方法                              |
| ------------------------------------------------------------------- | ------------------------------------- |
| SIMD add(a, b) 与对应语义模块串行实现逐元素一致                     | 随机 `[f64; 0..1024]`                 |
| SIMD sum(a) 与对应语义模块串行实现满足 §28.3 规定的数值语义/容差    | 随机 `[f64; 1..8192]`、`Complex<f64>` |
| SIMD dot(a, b) 与对应语义模块串行实现满足 §28.3 规定的数值语义/容差 | 随机 `[f64; 1..8192]`、`Complex<f64>` |
| tail 处理正确                                                       | `len = n * width + k`, k ∈ [0, width) |

### 8.5 集成测试

| 测试文件             | 测试内容                                                                               |
| -------------------- | -------------------------------------------------------------------------------------- |
| `tests/test_simd.rs` | SIMD dispatch 与 `math`、`reduction`、`dot`、`layout`、`parallel` 组合路径的端到端验证 |

### 8.6 Feature gate / 配置测试

| 场景               | 配置方式                            | 预期行为                                          |
| ------------------ | ----------------------------------- | ------------------------------------------------- |
| 默认关闭           | 默认 feature 集                     | 不编译 SIMD 路径，且无 `pulp` 运行时依赖          |
| 显式启用           | `--features simd`                   | SIMD API 可用，满足条件时进入 SIMD 路径           |
| 无硬件能力自动分层 | `--features simd` + 非目标 ISA 环境 | 成功编译运行，并由 `dispatch.rs` 回到非 SIMD 路径 |

### 8.7 类型边界 / 编译期测试

| 测试类型   | 覆盖内容                                                                 | 预期结果                                |
| ---------- | ------------------------------------------------------------------------ | --------------------------------------- |
| 类型边界   | `f32`/`f64`/`i32`/`i64`/`Complex<f32>`/`Complex<f64>` 实现 `SimdElement` | 可编译并进入对应 dispatch 分支          |
| 类型边界   | `bool`/`usize`/其他 `Complex<T>` 不实现 `SimdElement`                    | 编译期拒绝直接使用 SIMD kernel          |
| 编译期测试 | `#[cfg(feature = "simd")]` / `#[cfg(not(feature = "simd"))]` 两侧 API    | 两种 feature 组合均可编译               |
| 编译期测试 | 未覆盖数学函数与非 sum reduction kernel                                  | 保持公开 API 可用且落入语义模块串行路径 |

---

## 9. 与其他模块的交互

### 9.1 接口约定

| 方向         | 对方模块                     | 接口/类型                                                                                                    | 契约                                                                                            |
| ------------ | ---------------------------- | ------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------- |
| 消费（输入） | `tensor`                     | `TensorBase<S, D>::as_slice()`                                                                               | 只消费满足 `07-tensor.md` 连续切片契约的输入；非连续或广播视图保持语义模块串行路径              |
| 消费（输入） | `math` / `reduction` / `dot` | `dispatch.rs` / `dispatch_vector_binary_op()` / `SimdKernel::sum()` / `SimdKernel::dot()` / `is_supported()` | 上层语义模块先完成形状和类型裁决，再在 `dispatch.rs` 中做路径选择，只有 SIMD 分支才传入兼容切片 |
| 消费（输入） | `layout`                     | 对齐/连续性元数据                                                                                            | 当前版本仅对统一对齐快路径启用 SIMD，其余情况由 `dispatch.rs` 回到非 SIMD 路径                  |
| 产出（输出） | 上层语义模块                 | 标量结果或写回目标切片                                                                                       | 不改变公开 API 形状、错误类别和数值语义边界                                                     |

`math/` 模块在执行逐元素运算时，只在输入已经提取为兼容连续切片且满足当前统一对齐快路径时经 `dispatch.rs` 进入 SIMD，否则保持其串行实现（参见 `11-math.md §5.3`）。

> **分派策略**: `math` 模块提供公共 `sqrt()` API，但在当前版本中 `sqrt()` 不接入 SIMD kernel，而是保持语义模块串行路径。用户仅调用 `math::sqrt()`，无需关心底层是否存在加速能力。

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
    └── NO  -> stay in semantic-module serial/other backend path
```

### 9.3 与 parallel 模块

SIMD 与并行的组合策略：由 `dispatch.rs` 统一裁决。`dispatch` 先决定是否启用并行（基于元素数和并行阈值），再在每个执行线程（或串行路径）内决定是否启用 SIMD（基于元素数、对齐和 ISA 支持）。并行路径的 worker 线程内调用 SIMD helper 时，不会再次触发并行分派。

### 9.4 与 storage/layout 模块

SIMD 模块依赖 layout 提供的连续性和对齐信息来判断是否可以使用 SIMD 路径（参见 `06-layout.md` §5.5）。

---

## 10. 错误处理与语义边界

| 类型              | 说明                                                                                                                                              |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Recoverable error | 无专属 recoverable error；SIMD 不可用、类型不支持、未满足统一对齐快路径或当前 ISA 无法满足语义约束时均由 `dispatch.rs` 回到非 SIMD 路径           |
| Panic             | 切片长度不一致；整数 `sum` / `dot` 溢出；违反 kernel 前置条件的内部 bug                                                                           |
| 路径一致性        | 逐元素 SIMD 路径保持公开语义一致；整数 `sum` / `dot` 与标量 checked arithmetic 精确一致；浮点/复数归约与内积允许已文档化容差                      |
| 容差边界          | integer: exact; for float/complex reduction and dot, tolerance and comparison rules follow `00-coding.md §7.4`.                                  |

> **容差说明：** 容差与比较规则统一遵循 `00-coding.md §7.4` 的定义。同执行路径基础算术/比较默认精确一致；仅跨路径比较和数学函数比较允许使用文档化容差。

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

| 属性     | 值                                                                                                                                                                                                                              |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 决策     | SIMD 只覆盖本文列出的子集以及 `sum` / `dot`；整数 `sum` / `dot` 只有在可证明保持 checked arithmetic 语义时才使用 SIMD，浮点/复数 reduction / dot 允许文档化容差                                                                 |
| 理由     | `require.md` §9.1 / §9.3 / §28.3 明确要求公开能力可由 SIMD 加速实现，但不能破坏公开语义；因此整数路径只在可证明保持 checked arithmetic 语义时使用 SIMD，否则直接回退标量；浮点/复数路径则采用受控的累加顺序变化与显式容差记录。 |
| 替代方案 | 所有归约/内积一律回退标量 — 放弃，与需求文档冲突                                                                                                                                                                                |
| 替代方案 | 允许任意差异 — 放弃，无法保证正确性                                                                                                                                                                                             |

### 决策 3：默认关闭 SIMD feature

| 属性     | 值                                                              |
| -------- | --------------------------------------------------------------- |
| 决策     | SIMD 默认关闭，通过 `features = ["simd"]` 启用                  |
| 理由     | 最小依赖原则；pulp 在某些平台上可能引入编译问题；用户可按需启用 |
| 替代方案 | 默认启用 — 放弃，违反最小依赖原则                               |

### 决策 4：移除 ScalarKernel

| 属性     | 值                                                                                       |
| -------- | ---------------------------------------------------------------------------------------- |
| 决策     | 移除 `simd` 模块中的 `ScalarKernel`（`scalar.rs`），标量路径由各语义模块的串行实现承担   |
| 理由     | `ScalarKernel` 与各语义模块的串行实现功能重复；`simd` 模块定位为纯向量化后端更清晰       |
| 替代方案 | 保留 `ScalarKernel` 作为 `SimdKernel` trait 的参考实现 — 放弃，违反方案 D 的职责分离原则 |

---

## 12. 性能考量

### 12.1 支持的指令集与性能特征

| 指令集  | 架构    | 寄存器宽度 | f32 元素数 | f64 元素数 | 优先级   |
| ------- | ------- | ---------- | ---------- | ---------- | -------- |
| AVX-512 | x86_64  | 512 bit    | 16         | 8          | 最高     |
| AVX2    | x86_64  | 256 bit    | 8          | 4          | 高       |
| SSE4.1  | x86_64  | 128 bit    | 4          | 2          | 中       |
| NEON    | aarch64 | 128 bit    | 4          | 2          | 高 (ARM) |

### 12.2 性能数据（目标值，非当前默认路径）

> 说明：下表中的 `sum` / `dot` SIMD 数据代表当前版本目标实现。是否进入 SIMD 仍取决于连续性、对齐、ISA 能力与本设计文档定义的语义约束。FMA 若被用作 reduction merge 的局部优化，需单独检测，不隐含在 `AVX2` 标签内。

| 操作          | 标量 | AVX2 (f64) | 加速比 | 标量 | AVX2 (f32) | 加速比 |
| ------------- | ---- | ---------- | ------ | ---- | ---------- | ------ |
| add (1M 元素) | ~2ms | ~0.5ms     | ~4x    | ~2ms | ~0.25ms    | ~8x    |
| sum (1M 元素) | ~1ms | ~0.3ms     | ~3.3x  | ~1ms | ~0.15ms    | ~6.7x  |
| dot (1M 元素) | ~2ms | ~0.5ms     | ~4x    | ~2ms | ~0.3ms     | ~6.7x  |

### 12.3 性能影响因素

| 方面       | 设计决策                                                                             |
| ---------- | ------------------------------------------------------------------------------------ |
| 向量化宽度 | pulp 运行时自动选择最优宽度，无需编译期配置                                          |
| 内存对齐   | 当前版本仅对统一对齐快路径启用 SIMD；未满足该条件时由 `dispatch.rs` 保持非 SIMD 路径 |
| 尾部处理   | 标量循环处理尾部，简单安全                                                           |
| 循环展开   | pulp 内部处理，无需手动展开                                                          |
| FMA 利用   | 不用于逐元素 `mul`/`add`；仅可在已文档化容差的 reduction merge 中受控使用            |

---

## 13. 平台与工程约束

| 约束       | 说明                      |
| ---------- | ------------------------- |
| `std` only | SIMD 路径依赖 `std` 环境  |
| MSRV       | Rust 1.85+                |
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
| 1.2.3 | 2026-04-15 |
| 1.2.4 | 2026-04-15 |
| 1.2.5 | 2026-04-16 |
| 1.2.6 | 2026-04-16 |
| 1.2.7 | 2026-04-16 |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
