# 元素类型体系模块设计

> 文档编号: 03
> 模块目录: src/element/
> 任务阶段: Phase 1
> 前置文档: 00-coding.md, 01-architecture.md
> 需求参考: 需求说明书 §4、§5、§12 - §15、§23
> 范围声明: 范围内

---

## 1. 模块定位

### 1.1 职责边界

| 职责                | 包含                                                              |
| ------------------- | ----------------------------------------------------------------- |
| Element trait       | 基础约束（Copy+Clone+...）+ zero()/one()                          |
| Numeric trait       | Element + Add+Sub+Mul+Div+Neg + conjugate（通用数值运算能力标记） |
| RealScalar trait    | Numeric + PartialOrd + abs/sqrt/sin/exp/ln/floor/ceil + NaN 检测  |
| ComplexScalar trait | Numeric + re/im/norm（复数专用只读能力）                          |
| 基础类型实现        | 为 i32/i64/f32/f64/Complex<f32>/Complex<f64>/bool 实现上述 trait  |
| Sealed trait        | 封闭集合，禁止外部 crate 实现                                     |

| 职责                | 不包含                                                          |
| ------------------- | --------------------------------------------------------------- |
| Element trait       | -                                                               |
| Numeric trait       | 运算实现本身（委托给 core::ops）                                |
| RealScalar trait    | 复数运算                                                        |
| ComplexScalar trait | 复数类型定义（在 `src/complex/` 模块，参见 `04-complex.md` §5） |
| 基础类型实现        | 类型转换逻辑（在 `src/convert/` 模块）                          |
| Sealed trait        | 开放扩展                                                        |

### 1.2 设计原则

| 原则          | 体现                                         |
| ------------- | -------------------------------------------- |
| 能力最小化    | 每层 trait 仅声明必要约束，避免过度限制泛型  |
| 正交性        | 数值运算、实数函数、复数运算职责分离         |
| 零运行时开销  | 所有约束为编译期静态分派                     |
| 封闭集合      | Sealed trait 阻止下游 crate 扩展类型集       |
| IEEE 754 兼容 | 浮点特殊值（NaN、Inf）处理遵循标准语义       |

---

## 2. 需求映射与范围约束

| 项目     | 内容                                                              |
| -------- | ----------------------------------------------------------------- |
| 需求映射 | 需求说明书 §4、§5、§12 - §15、§23                                 |
| 范围内   | Element/Numeric/RealScalar/ComplexScalar trait 与封闭元素类型集合 |
| 范围外   | 张量存储、自动类型提升、开放外部元素扩展、具体类型转换执行逻辑    |
| 非目标   | 引入新的基础数值类型集合、运行时类型擦除或动态分派元素系统        |

---

## 3. 文件位置

```
src/element/
├── mod.rs             # Element trait definitions and module re-exports
├── numeric.rs         # Numeric trait definitions (arithmetic bounds)
├── real.rs            # RealScalar trait definitions (real math functions)
├── complex.rs         # ComplexScalar trait definitions (complex operations)
└── primitives.rs      # Trait implementations for primitive element types
```

模块内聚设计：四层 trait 按文件分离，基础类型实现集中在 `primitives.rs`。

---

## 4. 依赖关系

### 4.1 依赖图（ASCII）

```
src/element/
├── crate::error      # XenonError only (consumed for fallible APIs).
                      # NOTE (v1.4.0): `ElementType` is defined HERE in the
                      # element module (see §5.1.1), NOT in error.
├── crate::complex    # Complex<T> type definition
├── crate::private    # Sealed trait infrastructure
├── core::ops         # Add/Sub/Mul/Div/Neg operator traits
├── core::fmt         # Debug/Display formatting
└── core::cmp         # PartialEq/PartialOrd comparisons
```

### 4.2 类型级依赖

| 来源模块         | 使用的类型/trait                                           |
| ---------------- | ---------------------------------------------------------- |
| `crate::error`   | `XenonError`（显式类型转换失败时返回）；本模块**不**反向依赖 error。注意 `ElementType` 枚举在 v1.4.0 起重新由本模块拥有（详见 §5.1.1），`error` 模块用 `&'static str` 记录类型诊断信息（来自 `Element::ELEMENT_TYPE_NAME`），不再持有 `ElementType` 字段 |
| `crate::complex` | `Complex<f32>`, `Complex<f64>`（元素类型实现目标）         |
| `crate::private` | `Sealed`（封闭 trait 实现边界）                            |
| `core::ops`      | `Add`, `Sub`, `Mul`, `Div`, `Neg`（Numeric supertrait）    |
| `core::fmt`      | `Debug`, `Display`（Element supertrait）                   |
| `core::cmp`      | `PartialEq`, `PartialOrd`（Element/RealScalar supertrait） |

### 4.3 依赖合法性与新增依赖说明

| 项目           | 结论                       |
| -------------- | -------------------------- |
| 新增第三方依赖 | 无                         |
| 合法性结论     | 符合需求说明书最小依赖限制 |
| 替代方案       | 不适用                     |

### 4.4 依赖方向声明

**依赖方向**：单向向上。 `element/` 消费 `complex` 的类型定义（即 `element` 依赖 `complex`），`complex` 不反向依赖 `element`。

---

## 5. 公共 API 设计

**sealed 约束说明**：以下所有公开 trait 均通过 `private::Sealed` 实现 sealed trait 模式，禁止下游 crate 为自定义类型实现这些 trait。元素类型集合为封闭集合，不支持外部扩展。

### 5.1 Element trait

```rust,ignore
/// Base trait for all tensor element types.
///
/// Sealed: cannot be implemented outside this crate.
/// All tensor elements must be Copy, thread-safe, and have zero/one identities.
pub trait Element:
    Copy
    + Clone
    + PartialEq
    + core::fmt::Debug
    + core::fmt::Display
    + Send
    + Sync
    + Sealed
{
    /// Additive identity (zero).
    fn zero() -> Self;

    /// Multiplicative identity (one).
    fn one() -> Self;

    /// Element type discriminant for FFI consumers.
    ///
    /// Maps Rust element types to a C-compatible enum discriminant at
    /// compile time. `ElementType` is **owned by this module**
    /// (authoritative definition below in §5.1.1) so that `error` (L0)
    /// stays free of any element-trait dependency. `ffi` (L4) re-exports
    /// `crate::element::ElementType` for C consumers — the C ABI surface
    /// path is `crate::ffi::ElementType` (which equals
    /// `crate::element::ElementType`).
    const ELEMENT_TYPE: ElementType;

    /// Human-readable static name of this element type, e.g. "f32",
    /// "Complex<f64>".
    ///
    /// This is the string form used in **error** diagnostics
    /// (`XenonError::TypeConversion::source_type` /  `target_type`,
    /// `AbiMismatchKind::ElementTypeMismatch::expected` / `actual`,
    /// see `26-error.md §5.1`). Storing the static `&'static str` in
    /// the `Element` trait — rather than carrying an `ElementType`
    /// value into the error module — keeps `error` (L0) free of any
    /// dependency on `element` (L2): error fields just hold
    /// `&'static str` and Display them directly. The string values
    /// here are the canonical Xenon names and **must** stay in sync
    /// with `ElementType::name()` below.
    const ELEMENT_TYPE_NAME: &'static str;
}
```

| Supertrait  | 作用                                   |
| ----------- | -------------------------------------- |
| `Copy`      | 值语义，可按位复制，避免所有权转移开销 |
| `Clone`     | 显式克隆能力                           |
| `PartialEq` | 相等比较，用于断言和测试               |
| `Debug`     | 调试格式输出 `{:?}`                    |
| `Display`   | 用户友好格式输出 `{}`                  |
| `Send`      | 可跨线程移动（并行迭代必需）           |
| `Sync`      | 可跨线程共享引用（并行只读访问必需）   |
| `Sealed`    | 防止外部类型实现                       |

### 5.1.1 `ElementType` 枚举（权威定义；element 模块拥有）

**v1.4.0 起，`ElementType` 的权威定义重新搬回 `crate::element`**（之前 v1.3.x 曾下沉到 `crate::error`，详见下方决策回滚说明）。`error` 模块**不再**直接持有 `ElementType` —— 它使用 `&'static str`（来自 `Element::ELEMENT_TYPE_NAME`）记录类型诊断信息，从而严格保持 `error`（L0）→ 不依赖任何上层模块。`ffi` 模块（L4）通过 `pub use crate::element::ElementType` re-export 暴露给 C 消费者。

```rust,ignore
// src/element/mod.rs (authoritative definition)

/// Compile-time discriminant of every supported element type.
///
/// `#[non_exhaustive]` so adding a new supported element type in a future
/// version does not constitute a breaking change for downstream `match`
/// expressions. Downstream code MUST include a `_ => ...` arm when matching
/// exhaustively.
///
/// `#[repr(u8)]` keeps the enum cheap to copy/hash and gives the FFI layer
/// (see `23-ffi.md`) a stable single-byte tag. **Discriminant values ARE
/// part of the public C ABI contract** (v1.4.0+): they are SemVer-pinned
/// for `crate::ffi::ElementType` C consumers. Reordering existing variants
/// or reusing existing values is a breaking change and requires a major
/// version bump. Adding a new variant gets a new value and is non-breaking
/// under `#[non_exhaustive]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
#[non_exhaustive]
pub enum ElementType {
    Bool      = 0,
    I32       = 1,
    I64       = 2,
    F32       = 3,
    F64       = 4,
    Complex32 = 5,
    Complex64 = 6,
}

impl ElementType {
    /// Canonical Xenon name for this element type, as a `&'static str`.
    ///
    /// **This must stay in sync with `Element::ELEMENT_TYPE_NAME`** for
    /// every concrete `impl Element` in this module — every Element impl
    /// MUST set `ELEMENT_TYPE_NAME` to the exact string returned here for
    /// its corresponding `ELEMENT_TYPE`. The crate-internal test
    /// `tests/element_type_name_consistency.rs` enforces this.
    pub const fn name(self) -> &'static str {
        // No wildcard arm: this `match` is in the defining crate and is
        // exhaustive over current variants. `#[non_exhaustive]` only affects
        // out-of-crate matches; in-crate, adding a new variant must update
        // this `match` (the compiler will fail with E0004, which is the
        // intended behavior — silent fallthrough to "<unknown>" would be
        // worse than a hard compile error inside the crate).
        match self {
            ElementType::Bool       => "bool",
            ElementType::I32        => "i32",
            ElementType::I64        => "i64",
            ElementType::F32        => "f32",
            ElementType::F64        => "f64",
            ElementType::Complex32  => "Complex<f32>",
            ElementType::Complex64  => "Complex<f64>",
        }
    }

    /// Inherent constructor returning the `ElementType` discriminant for `A`.
    ///
    /// Equivalent to the free function `element_type_of::<A>()`. Provided
    /// for ergonomic call sites that prefer `ElementType::of::<f32>()`
    /// over the free-function form. v1.4.0 onward.
    pub const fn of<A: Element>() -> Self {
        A::ELEMENT_TYPE
    }
}

impl core::fmt::Display for ElementType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

/// Free function returning the `ElementType` discriminant for `A`.
///
/// `pub const fn` here (rather than an inherent method on `ElementType`)
/// keeps the `A: Element` bound on the `element` side; it is also the
/// natural form for ergonomic call sites like `element_type_of::<f32>()`.
pub const fn element_type_of<A: Element>() -> ElementType {
    A::ELEMENT_TYPE
}

/// Free function returning the canonical name for `A`.
///
/// Equivalent to `<A as Element>::ELEMENT_TYPE_NAME`. Provided for symmetry
/// with `element_type_of::<A>()` and for use in error construction sites
/// that want to avoid spelling out the trait method.
pub const fn element_type_name_of<A: Element>() -> &'static str {
    A::ELEMENT_TYPE_NAME
}
```

**Element impl 必须同时设置 `ELEMENT_TYPE` 与 `ELEMENT_TYPE_NAME`，且后者等于前者的 `.name()`：**

```rust,ignore
impl Element for f32 {
    const ELEMENT_TYPE: ElementType = ElementType::F32;
    const ELEMENT_TYPE_NAME: &'static str = "f32"; // == ElementType::F32.name()
    fn zero() -> Self { 0.0 }
    fn one() -> Self  { 1.0 }
}

// Compile-time consistency check (recommended pattern; or do it as a
// regular runtime test in tests/element_type_name_consistency.rs):
const _: () = {
    assert!(matches!(<f32 as Element>::ELEMENT_TYPE, ElementType::F32));
    // const_eq for &'static str isn't stable as of MSRV 1.85, so the
    // string check happens in the runtime consistency test instead.
};
```

**设计决策（v1.4.0：ElementType owner 回到 element）：**

| 项 | v1.3.x（已废弃） | v1.4.0（当前） |
|:--|:--|:--|
| `ElementType` 定义位置 | `crate::error` | `crate::element` |
| `error` 字段类型 | `source_type: ElementType` | `source_type: &'static str` |
| `error` 是否依赖 `element` | 不依赖（用 enum 自带 Display） | 不依赖（用 `&'static str` 自带 Display） |
| FFI 字段 | `pub use crate::error::ElementType` | `pub use crate::element::ElementType` |
| Element 关联常量 | `ELEMENT_TYPE` | `ELEMENT_TYPE` + 新增 `ELEMENT_TYPE_NAME` |

回滚理由：v1.3.x 把 `ElementType` 下沉到 L0 是为了让 error 模块能直接 Display 类型名；但这同时让"元素类型枚举"在概念归属上离开了 element 模块，命名与所在位置出现张力（`Element::ELEMENT_TYPE` 关联常量类型却定义在 error）。v1.4.0 通过引入 `Element::ELEMENT_TYPE_NAME: &'static str` 让 error 直接持有用于 Display 的字符串，**两个目标同时达成**：error 不依赖 element + element 重新拥有 `ElementType` 类型。`element_type_of::<A>()` 仍是自由函数（不是 inherent impl），保留是因为它是 `pub const fn` 形态的便利入口；inherent impl 也可以现在加上（`impl ElementType { pub const fn of<A: Element>() -> Self { A::ELEMENT_TYPE } }`）因为类型定义已回 element 模块、Rust E0116 不再触发——v1.4.0 同时新增此 inherent 形式作为推荐入口（自由函数保留作为完全等价的别名供调用点选择）。

### 5.2 Numeric trait

```rust,ignore
/// Numeric element trait.
///
/// Adds arithmetic operations on top of Element.
/// Xenon's generic numeric core currently covers signed integers, real scalars,
/// and complex scalars: `i32`, `i64`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`.
/// `bool` is explicitly excluded. `usize` is reserved for index/shape metadata
/// and is not part of the tensor element set.
///
/// The native operator supertraits describe syntax availability only.
/// Overflow-sensitive integer paths must additionally follow Xenon's checked
/// arithmetic contracts in operation modules so that recoverable vs panic
/// behavior remains consistent with the requirements specification.
///
/// Note: `Sealed` is not listed as a separate supertrait here because
/// `Element` already inherits `Sealed`.
pub trait Numeric:
    Element
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Div<Output = Self>
    + core::ops::Neg<Output = Self>
{
    /// Returns the canonical conjugate of the value.
    ///
    /// Real-valued types return `self` unchanged (identity).
    /// Complex-valued types return the mathematical conjugate.
    fn conjugate(self) -> Self;
}
```

| 必需项                | 语义说明                                                    |
| --------------------- | ----------------------------------------------------------- |
| `Add/Sub/Mul/Div/Neg` | 提供通用数值四则运算与取负能力                              |
| `conjugate(self)`     | 统一提供共轭语义：实数类型返回 `self`，复数类型返回数学共轭 |

**设计决策：** `Numeric` 在保留通用算术分层的同时，通过 `Numeric::conjugate()` 统一提供共轭语义：实数类型为恒等操作，复数类型执行数学共轭。`ComplexScalar` 保留复数专用能力（`re`/`im`/`norm`），不再单独承担 `conjugate` 的唯一 trait 入口角色。这与 `01-architecture.md`、`11-math.md`、`12-matrix.md` 的泛型约定保持一致。

**`conjugate()` 语义说明：** `conjugate()` 为泛型算法统一入口；对实数类型返回恒等值（自身），对复数类型返回共轭。此方法不代表所有 `Numeric` 类型均具备复数运算能力。

**整数算术契约**：`Add/Sub/Mul/Div/Neg` 只表达运算符可用性，不单独定义 Xenon 的溢出语义。凡需求文档要求“溢出/除零/结果不可表示即 panic”的整数运算路径，具体模块必须通过 checked 标量原语或等价显式检查落实，不得仅凭原生运算符 trait 假定语义成立。

### 5.3 RealScalar trait

```rust,ignore
/// Real-valued scalar trait.
///
/// Provides stable real-valued math functions and NaN detection.
/// Only f32 and f64 implement this trait.
pub trait RealScalar: Numeric + PartialOrd + Sealed {
    // Sealed is already inherited via Element (which Numeric extends),
    // but listed here for defensive clarity — makes the sealed intent explicit
    // at each trait level.
    // ========== Math functions ==========
    fn abs(self) -> Self;
    /// Returns the standard-library sign of the value.
    ///
    /// Finite non-NaN inputs return `1.0` or `-1.0`; specifically,
    /// `signum(+0.0) == 1.0`, `signum(-0.0) == -1.0`,
    /// `signum(+∞) == 1.0`, `signum(-∞) == -1.0`, and
    /// `signum(NaN) == NaN`.
    fn signum(self) -> Self;
    fn sqrt(self) -> Self;
    fn sin(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;

    fn is_nan(self) -> bool;

    /// Returns `true` if `self` is positive or negative infinity.
    /// Forwards to `f32::is_infinite` / `f64::is_infinite`.
    fn is_infinite(self) -> bool;

    /// Returns `true` if `self` is neither NaN nor infinity.
    /// Forwards to `f32::is_finite` / `f64::is_finite`.
    fn is_finite(self) -> bool;
}
```

- 公开 `RealScalar` trait 仅保留当前版本可稳定承诺的实数运算能力。
- `is_nan` / `is_infinite` / `is_finite` 三个谓词作为 IEEE 754 浮点类别检测的最小公开集合（与标准库 `f32`/`f64` 同名方法语义一致），供 `11-math` 与 `13-reduction` 等模块以及测试代码统一使用。
- 其余常量访问器与 NaN/无穷辅助逻辑降为 crate 内部扩展 trait，避免把实现便利误暴露为公开契约。
- `RealScalar::signum()` 明确跟随标准库 `f32::signum()` / `f64::signum()` 语义。有限非 NaN 输入返回 `1.0` 或 `-1.0`，其中 `signum(+0.0) == 1.0`、`signum(-0.0) == -1.0`，`NaN` 传播为 `NaN`。`11-math.md` 中张量级 `signum()` 的浮点语义以此 trait 契约为权威基线。

### 5.4 ComplexScalar trait

```rust,ignore
/// Complex scalar trait.
///
/// Provides the minimal complex-specific operations required by the current
/// tensor API surface. Only Complex<f32> and Complex<f64> implement this.
pub trait ComplexScalar: Numeric + Sealed {
    // Sealed is already inherited via Element (which Numeric extends),
    // but listed here for defensive clarity — makes the sealed intent explicit
    // at each trait level.
    /// Real part type (must be RealScalar).
    type Real: RealScalar;

    fn re(self) -> Self::Real;
    fn im(self) -> Self::Real;
    fn norm(self) -> Self::Real;
}
```

- `Numeric::conjugate()` 是全体数值元素唯一的统一共轭入口：实数路径返回恒等，复数路径返回数学共轭。
- `ComplexScalar` 只保留 `re` / `im` / `norm` 这类复数专用能力，不再重复声明 `conjugate()`。

### 5.5 OrderedCompareElement trait

```rust,ignore
/// Ordered comparison element trait.
///
/// Publicly exposed for the `lt` / `gt` comparison API in the math module,
/// while remaining sealed so only Xenon's supported ordered element types can
/// implement it.
pub trait OrderedCompareElement: Element + PartialOrd + Sealed {}

impl OrderedCompareElement for i32 {}
impl OrderedCompareElement for i64 {}
impl OrderedCompareElement for f32 {}
impl OrderedCompareElement for f64 {}
```

- `OrderedCompareElement` 需要作为公开 sealed trait 暴露，因为 `11-math` 的公开比较 API（`less` / `greater`，详见 `11-math.md §5` 与 `01-architecture.md §10.1`）直接使用它作为元素类型约束；但其实现集合仍限制为 Xenon 当前支持的有序比较元素类型。
- `OrderedCompareElement` 用于把有序比较能力显式收敛到 `i32`、`i64`、`f32`、`f64`。该 trait 虽然为配合 `11-math` 的公开 `less` / `greater` 比较 API 而公开暴露，但仍通过 `Sealed` 保持 sealed，只允许 Xenon 为这四种类型提供实现。

### 5.6 BoolElement（`pub(crate)` sealed）

`BoolElement` 是仅在 element 模块内部使用的辅助 trait，标记 `bool` 类型以区分布尔运算的可用性（例如 `not()`）。

```rust,ignore
/// Internal marker for the bool element type.
///
/// Used by `11-math.md` `not()` to constrain its impl to bool tensors only.
/// Not part of the public API; sealed via `crate::private::Sealed`.
pub(crate) trait BoolElement: Element + Sealed {}

impl BoolElement for bool {}
```

- **用途**：`11-math.md` 的 `not()` 方法 trait bound 使用 `A: BoolElement`，阻止其他元素类型偶然实现该 trait。
- **`pub(crate)` 与公开 API 边界**：`BoolElement` 本身不出现在 `not()` 等方法的 *公开* 签名上；公开 API 仅通过 `impl<S, D> TensorBase<S, D> where S: Storage<Elem = bool>` 之类的具体类型约束暴露 `not()`，避免出现私有 trait 出现在公开 bound 上的可见性冲突。详见 `11-math.md §5.7`。

### 5.7 支持的类型与 trait 矩阵

| 类型           | Element | Numeric | RealScalar | ComplexScalar |
| -------------- | :-----: | :-----: | :--------: | :-----------: |
| `i32`          |    ✓    |    ✓    |     ✗      |       ✗       |
| `i64`          |    ✓    |    ✓    |     ✗      |       ✗       |
| `f32`          |    ✓    |    ✓    |     ✓      |       ✗       |
| `f64`          |    ✓    |    ✓    |     ✓      |       ✗       |
| `Complex<f32>` |    ✓    |    ✓    |     ✗      |       ✓       |
| `Complex<f64>` |    ✓    |    ✓    |     ✗      |       ✓       |
| `bool`         |    ✓    |    ✗    |     ✗      |       ✗       |

- 仅支持上表列出的 7 种元素类型。
- 不支持 `usize`、u8/u16/u32/i8/i16 等其他整数类型。
- `usize` 仅作为索引和形状元数据使用。

**按运算的元素类型可用矩阵（跨模块快速参考，非规范性索引）：** 以下表格汇总各运算模块支持的元素类型；个别运算因数学语义限制（如有序比较对复数无定义）仅支持子集。**权威定义仍以各运算模块文档为准**，本表仅作为单点查询与 sealed trait 实现一致性核对的参考；每行的“权威文档”列必须保留对应 owner 链接。

| 运算 / 模块 | i32 | i64 | f32 | f64 | Complex<f32> | Complex<f64> | bool | 权威文档 |
| ----------- | :-: | :-: | :-: | :-: | :----------: | :----------: | :--: | -------- |
| 算术 add/sub/mul/div（11-math） | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | `11-math.md §5.3` |
| neg / abs / square（11-math） | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | `11-math.md §5.4` |
| 内积 dot（12-matrix） | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | `12-matrix.md §5.1` |
| sum 归约（13-reduction） | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | `13-reduction.md §5`（当前版本仅 sum；mean/min/max 不在范围） |
| unique 集合运算（14-set） | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | `14-set.md §6.1`（哈希查重 + F-order 顺序输出） |
| eye 单位矩阵构造（18-construction） | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | `18-construction.md` |
| clip（20-utility） | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | `20-utility.md`（无序比较不适用） |
| cast 类型转换（21-type） | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | `21-type.md` |
| 有序比较 less/greater（OrderedCompareElement） | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | `03-element.md §5.5`（复数无序；命名权威见 `11-math.md §5`，仅 less/greater，不含 less_equal/greater_equal） |
| Checked 整数原语（CheckedAdd/Sub/Mul/Neg/Div） | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | `03-element.md §5.10` |

### 5.8 Sealed trait 策略

`Element`、`Numeric`、`RealScalar`、`ComplexScalar`、`CastTo<T>`、`CastElement`、`OrderedCompareElement` 全部通过共享的 `private::Sealed` 基础设施实现 sealed trait 模式。下游 crate 只能使用 Xenon 已声明的元素类型，不能为自定义类型补充这些 trait 实现。`CastTo<T>` 与 `CastElement` 都通过 `Self: Element`（间接 `Sealed`）封闭实现范围。

```rust,ignore
// src/element/mod.rs
// Uses the shared Sealed trait from crate::private
// (see src/private.rs, referenced in 01-architecture.md §3)
use crate::private::Sealed;

// Sealed implementations in primitives.rs
impl Sealed for i32 {}
impl Sealed for i64 {}
impl Sealed for f32 {}
impl Sealed for f64 {}
impl Sealed for Complex<f32> {}
impl Sealed for Complex<f64> {}
impl Sealed for bool {}
```

### 5.9 CastTo<T> trait

`CastTo<T>` 的 trait 定义位于 `src/element/mod.rs`，具体 impl 统一放在 `src/convert/cast.rs`；`convert/` 负责消费该 trait 并承载受支持转换矩阵的实现（参见 `21-type.md §5.1`），不在其他模块重复定义或分散实现。

```rust,ignore
// src/element/mod.rs

use crate::error::XenonError;

/// Element-wise type conversion trait.
///
/// Defines explicit conversion from `Self` to `T`.
/// Lossless conversions return `Ok(T)`.
/// Lossy conversions default to recoverable
/// `XenonError::TypeConversion { operation, source_type, target_type, reason, element_index }`
/// unless a documented success precondition is satisfied (see `21-type.md §5.3`).
///
/// This trait is implemented only inside Xenon for the supported source/target pairs.
/// External crates cannot extend the conversion matrix.
///
/// The target type `T` is itself constrained to `Element`: this prevents
/// instantiating `CastTo<T>` for types outside Xenon's closed element set
/// at the trait-level, complementing the sealed `Self: Element` bound.
pub trait CastTo<T: Element>: Element {
    /// Performs the type conversion.
    fn cast_to(self) -> Result<T, XenonError>;
}
```

- 类型转换错误载荷的完整定义见 `26-error.md §5.1`，`CastTo<T>` 的转换矩阵与实现约束见 `21-type.md §5.2`、`§6.1`。本节仅保留元素层 trait 骨架。
- `CastTo<T>` 直接返回 `XenonError::TypeConversion`。
- 类型参数 `T` 必须满足 `T: Element`，结合 `Self: Element` 的 sealed 边界，从 trait 层面把转换关系限制在 Xenon 封闭元素集合内。
- `bool` 不出现为任何 `CastTo<T>` 的 *源类型*，也不出现为任何 `CastTo<T>` 的 *目标类型*；这两个方向都通过"不提供 impl"在编译期阻断（与 §6.1 决策一致）。
- `Complex<T> -> Real` 的条件成功语义、受支持矩阵与 `XenonError::TypeConversion` 字段约束，统一以 `21-type.md §5.3`、`§6.1` 以及 `26-error.md §5.6` 为准。

#### 5.9.1 CastElement marker trait

`CastElement` 是公开 sealed marker trait，标记"可作为 `cast()` 操作源/目标元素类型集合"。`21-type.md §5.1` 的 `cast()` 公开方法签名 `where A: CastElement, T: CastElement` 通过此 trait 在编译期排除 `bool`，并把元素集合统一收敛到 6 个数值类型。

```rust,ignore
// src/element/mod.rs

/// Marker trait for element types that participate in the `cast()` operation.
///
/// Sealed via `Element` (which inherits `Sealed`); only Xenon's six numeric
/// element types implement it. `bool` is intentionally excluded — `bool` is
/// neither a valid source nor a valid target of `cast()`, matching the
/// "no impl" exclusion of `bool` from `CastTo<T>` (see `§5.9` and `§6.1`).
///
/// This trait is consumed by `21-type.md §5.1` `cast()` to bound both the
/// receiver element type `A` and the target element type `T`. It does **not**
/// itself define a `cast` method — conversion logic still goes through
/// `CastTo<T>` and `convert/cast.rs`.
pub trait CastElement: Element {}

impl CastElement for i32 {}
impl CastElement for i64 {}
impl CastElement for f32 {}
impl CastElement for f64 {}
impl CastElement for Complex<f32> {}
impl CastElement for Complex<f64> {}
// bool: NOT implemented — bool is excluded from cast() per §6.1.
```

- **与 `CastTo<T>` 的关系**：`CastElement` 是"哪些元素类型属于 cast 矩阵"的封闭集合标记；`CastTo<T>` 是"具体的 (源, 目标) 对存在合法转换"的关系。两者同属 §5.9 类型转换主题，但语义槽位不同：`CastElement` 让 `cast()` 公开签名能用单一 trait 边界排除 `bool`；`CastTo<T>` 表达具体可转换关系并实现转换逻辑。
- **sealed 论证**：`CastElement: Element` 间接 sealed（`Element: Sealed`），下游 crate 无法为新类型实现。
- **owner**：`CastElement` 定义在 `src/element/mod.rs`；`21-type.md §5.1` 仅消费、不重新定义。

### 5.10 Checked arithmetic traits

`CheckedAdd` 为整数类型提供 checked 加法，供 `sum` 归约操作在整数溢出时 panic（参见 `13-reduction.md §5.1`）。

```rust,ignore
// src/element/mod.rs

/// Checked addition for types that support it.
///
/// Returns `None` on overflow instead of wrapping.
/// Only implemented for integer types (`i32`, `i64`).
/// Float types use ordinary `+` (NaN propagation handles the semantics).
///
/// Used by integer `sum()` reduction to guarantee overflow is detected
/// in both debug and release builds (per requirement §14).
pub(crate) trait CheckedAdd: Numeric + Sealed {
    /// Returns `Some(self + rhs)` if no overflow, `None` otherwise.
    fn checked_add(self, rhs: Self) -> Option<Self>;
}

impl CheckedAdd for i32 {
    #[inline]
    fn checked_add(self, rhs: Self) -> Option<Self> {
        i32::checked_add(self, rhs)
    }
}

impl CheckedAdd for i64 {
    #[inline]
    fn checked_add(self, rhs: Self) -> Option<Self> {
        i64::checked_add(self, rhs)
    }
}
// f32, f64, Complex: NOT implemented — overflow is handled by IEEE 754 semantics.
// bool: NOT implemented — not Numeric.

/// Checked subtraction for integer-only overflow-sensitive paths.
pub(crate) trait CheckedSub: Numeric + Sealed {
    fn checked_sub(self, rhs: Self) -> Option<Self>;
}

impl CheckedSub for i32 {
    #[inline]
    fn checked_sub(self, rhs: Self) -> Option<Self> { i32::checked_sub(self, rhs) }
}

impl CheckedSub for i64 {
    #[inline]
    fn checked_sub(self, rhs: Self) -> Option<Self> { i64::checked_sub(self, rhs) }
}

/// Checked multiplication for integer-only overflow-sensitive paths.
pub(crate) trait CheckedMul: Numeric + Sealed {
    fn checked_mul(self, rhs: Self) -> Option<Self>;
}

impl CheckedMul for i32 {
    #[inline]
    fn checked_mul(self, rhs: Self) -> Option<Self> { i32::checked_mul(self, rhs) }
}

impl CheckedMul for i64 {
    #[inline]
    fn checked_mul(self, rhs: Self) -> Option<Self> { i64::checked_mul(self, rhs) }
}

/// Checked negation for integer-only overflow-sensitive paths.
pub(crate) trait CheckedNeg: Numeric + Sealed {
    fn checked_neg(self) -> Option<Self>;
}

impl CheckedNeg for i32 {
    #[inline]
    fn checked_neg(self) -> Option<Self> { i32::checked_neg(self) }
}

impl CheckedNeg for i64 {
    #[inline]
    fn checked_neg(self) -> Option<Self> { i64::checked_neg(self) }
}

/// Checked division for integer-only overflow-sensitive paths.
///
/// Returns `None` for divisor zero or for the `MIN / -1` overflow case;
/// callers translate `None` to a panic per the project-wide integer
/// overflow policy (see `26-error.md §6`).
pub(crate) trait CheckedDiv: Numeric + Sealed {
    fn checked_div(self, rhs: Self) -> Option<Self>;
}

impl CheckedDiv for i32 {
    #[inline]
    fn checked_div(self, rhs: Self) -> Option<Self> { i32::checked_div(self, rhs) }
}

impl CheckedDiv for i64 {
    #[inline]
    fn checked_div(self, rhs: Self) -> Option<Self> { i64::checked_div(self, rhs) }
}
```

- 此 trait 为内部实现辅助，不纳入稳定公开 API 面。具体可见性由实现决定。
- `CheckedAdd` 仅覆盖整数加法（`i32`/`i64`），用于归约等必须精确检测溢出的路径。
- 当前元素层统一提供 `CheckedAdd` / `CheckedSub` / `CheckedMul` / `CheckedNeg` / `CheckedDiv` 五类整数 checked 原语，作为整数溢出检测的**唯一权威定义点**，供 `math`、`matrix`、`reduction` 等所有上层模块复用。上层模块**不应**重新定义同语义 trait；余数与更高阶组合检查仍由具体运算模块在实现层基于这些原语组合完成。

### 5.11 Good / Bad 对比示例

```rust,ignore
// Good - Numeric constraint automatically excludes bool and non-Numeric types
fn sum<'a, A, D>(tensor: &TensorView<'a, A, D>) -> A
where
    A: Numeric,
    D: Dimension,
{
    tensor.iter().fold(A::zero(), |acc, &x| acc + x)
}
// sum(&bool_tensor);   // Compile error: bool does not satisfy Numeric

// Bad - Element constraint cannot exclude non-arithmetic element types
fn sum_bad<'a, A, D>(tensor: &TensorView<'a, A, D>) -> A
where
    A: Element,
    D: Dimension,
{
    // Cannot use + operator, Element has no Add bound
    todo!()
}
```

```rust,ignore
// Good - explicit type conversion, no automatic promotion
let a: Tensor<f64, Ix2> = Tensor::zeros((3, 4));
let b: Tensor<i32, Ix2> = Tensor::zeros((3, 4));
let b64 = b.cast::<f64>()?;
let c = &a + &b64;

// Bad - expecting automatic type promotion (not supported in Xenon)
// let c = &a + &b;  // Compile error: no matching impl for f64 + i32
```

---

## 6. 内部实现设计

### 6.1 bool 排除策略

`bool` 仅实现 `Element`，不实现 `Numeric`：

```rust,ignore
// primitives.rs
impl Element for bool {
    fn zero() -> Self { false }
    fn one() -> Self { true }

    // Both `ELEMENT_TYPE` and `ELEMENT_TYPE_NAME` are mandatory for *every*
    // `Element` impl (see §5.1 trait definition + §5.1.1 ElementType).
    // For `bool`, the type-tag pair maps to (Bool, "bool"). FFI consumers
    // identify `Tensor<bool, _>` via `ELEMENT_TYPE`; error diagnostics
    // (e.g. `XenonError::TypeConversion::source_type`) use the
    // `ELEMENT_TYPE_NAME` string. Both constants are required even though
    // `bool` is excluded from `Numeric`.
    const ELEMENT_TYPE: ElementType = ElementType::Bool;
    const ELEMENT_TYPE_NAME: &'static str = "bool"; // == ElementType::Bool.name()
}

// Equivalent constant pairs for the other six element types:
//   impl Element for i32          { ... ELEMENT_TYPE = ElementType::I32;       NAME = "i32";          }
//   impl Element for i64          { ... ELEMENT_TYPE = ElementType::I64;       NAME = "i64";          }
//   impl Element for f32          { ... ELEMENT_TYPE = ElementType::F32;       NAME = "f32";          }
//   impl Element for f64          { ... ELEMENT_TYPE = ElementType::F64;       NAME = "f64";          }
//   impl Element for Complex<f32> { ... ELEMENT_TYPE = ElementType::Complex32; NAME = "Complex<f32>"; }
//   impl Element for Complex<f64> { ... ELEMENT_TYPE = ElementType::Complex64; NAME = "Complex<f64>"; }
//
// Each NAME literal MUST equal `ElementType::<discriminant>.name()` exactly;
// this is enforced by the runtime test
// `tests/element_type_name_consistency.rs` (see §5.1.1).
```

编译时阻止无效泛型实例化：`fn sum<A: Numeric>` 无法接受 `bool` 张量；需要布尔专用逐元素逻辑非时，使用 `!`。此外，`bool` 不实现任何 `CastTo<T>`；`bool_tensor.cast::<f32>()` 必须在编译期失败，而不是返回运行时类型转换错误。

### 6.2 usize 语义边界

`usize` 不属于 Xenon 的张量元素集合，仅作为索引、轴和形状元数据类型使用。所有元素 trait（`Element`/`Numeric`/`RealScalar`/`ComplexScalar`）都不为 `usize` 提供实现，也不再为其预留算术扩展路径。

### 6.3 类型提升规则

Xenon 不支持自动类型提升。 所有跨类型运算须显式转换：

```rust,ignore
// Implicit conversion not supported
// let a: f64 = 1.0;
// let b: i32 = 2;
// let c = a + b;  // Compile error

// Must convert explicitly through Xenon's cast contract
let c = a + b.cast_to()?;
```

### 6.4 NaN/Inf 处理语义

| 方法                  | NaN 输入 | Inf 输入 |
| --------------------- | -------- | -------- |
| `abs(NaN)`            | NaN      | Inf      |
| `sqrt(-1.0)`          | NaN      | —        |
| `ln(0.0)`             | —        | -Inf     |
| `exp(Inf)`            | —        | Inf      |
| `min(a, b)` 含 NaN    | NaN      | 正常比较 |
| `partial_cmp(NaN, _)` | None     | 正常比较 |

### 6.5 RealScalar 实现

以 f64 为例：

```rust,ignore
impl Numeric for f64 {
    #[inline]
    fn conjugate(self) -> Self { self }
}

// RealScalar math functions are implemented in the `std` environment required by Xenon.
impl RealScalar for f64 {
    fn abs(self) -> Self { self.abs() }
    fn signum(self) -> Self { self.signum() }
    fn sqrt(self) -> Self { self.sqrt() }
    fn sin(self) -> Self { self.sin() }
    fn exp(self) -> Self { self.exp() }
    fn ln(self) -> Self { self.ln() }
    fn floor(self) -> Self { self.floor() }
    fn ceil(self) -> Self { self.ceil() }
    fn is_nan(self) -> bool { self.is_nan() }
    // ...
}

// Same pattern applies to f32.
```

- `i32`/`i64`/`f32`/`f64` 作为实数路径上的 `Numeric` 实现，`conjugate()` 一律为恒等操作
- `Complex<f32>`/`Complex<f64>` 的数学共轭也统一通过 `Numeric::conjugate()` 暴露，`ComplexScalar` 实现只补充复数特有的 `re`/`im`/`norm`。

---

## 7. 实现任务拆分

### Wave 1: 基础 trait 定义

- [ ] **T1**: 创建 `mod.rs`，导入共享 Sealed trait 并定义 Element trait
  - 文件: `src/element/mod.rs`
  - 内容: 从 `crate::private` 导入 `Sealed`，定义 `Element` trait、模块 re-export
  - 测试: 编译通过
  - 前置: 无
  - 预计: 10 min

- [ ] **T2**: 创建 `numeric.rs`，定义 Numeric trait 及其核心方法契约
  - 文件: `src/element/numeric.rs`
  - 内容: `Numeric` trait 定义（四则运算 supertrait + 统一 `conjugate()` 语义）
  - 测试: 编译通过
  - 前置: T1
  - 预计: 5 min

### Wave 2: 扩展 trait 定义

- [ ] **T3**: 创建 `real.rs`，定义 RealScalar trait
  - 文件: `src/element/real.rs`
  - 内容: `RealScalar` 仅含公开数学函数与 `is_nan()` / `is_infinite()` / `is_finite()`
  - 测试: 编译通过
  - 前置: T2
  - 预计: 10 min

- [ ] **T4**: 创建 `complex.rs`，定义 ComplexScalar trait
  - 文件: `src/element/complex.rs`
  - 内容: `ComplexScalar` trait 定义（关联类型 Real + 复数方法）
  - 测试: 编译通过
  - 前置: T2
  - 预计: 10 min

### Wave 3: 基础类型实现

- [ ] **T5**: 为 i32/i64 实现 Element + Numeric
  - 文件: `src/element/primitives.rs`
  - 内容: `Sealed` impl、`Element` impl、`Numeric` impl
  - 测试: `test_i32_zero_one`, `test_i64_arithmetic`
  - 前置: T2
  - 预计: 10 min

- [ ] **T6**: 为 f32/f64 实现 Element + Numeric + RealScalar
  - 文件: `src/element/primitives.rs`
  - 内容: 三层 trait 实现 + 数学函数委托
  - 测试: `test_f64_sqrt`, `test_f32_nan_detection`
  - 前置: T3, T5
  - 预计: 10 min

- [ ] **T7**: 为 bool 实现 Element（仅此）
  - 文件: `src/element/primitives.rs`
  - 内容: `Element` impl（`zero()=false`, `one()=true`，不实现 Numeric）
  - 测试: `test_bool_element_only`
  - 前置: T1
  - 预计: 5 min

- [ ] **T8**: 补充索引/形状侧对 `usize` 的边界说明
  - 文件: `src/element/mod.rs`
  - 内容: 文档中明确 `usize` 仅作为索引和形状元数据使用，不属于元素 trait 实现集合
  - 测试: 编译通过
  - 前置: T1
  - 预计: 5 min

### Wave 4: 复数类型实现

- [ ] **T9**: 为 Complex<f32>/Complex<f64> 实现 Element + Numeric + ComplexScalar
  - 文件: `src/element/primitives.rs`
  - 内容: 三层 trait 实现，关联类型 `Real = f32`/`f64`
  - 测试: `test_complex_f64_conj`, `test_complex_f32_norm`
  - 前置: T4, T5
  - 预计: 10 min

### Wave 5: 集成完善

- [ ] **T10**: 校准数学能力与转换错误语义文档
  - 文件: `src/element/real.rs`, `src/element/mod.rs`
  - 内容: 保持 `std` 环境下的数学接口边界，并将有损 CastTo 默认语义标注为可恢复错误
  - 测试: 编译通过
  - 前置: T6
  - 预计: 10 min

- [ ] **T11**: 文档注释与 cargo doc 验证
  - 文件: 所有 `src/element/` 文件
  - 内容: 所有 pub 项添加文档注释
  - 测试: `cargo doc` 无警告
  - 前置: T9
  - 预计: 10 min

- [ ] **T12**: 集成测试（跨模块交互验证）
  - 文件: element 内部单元测试 + doctest，跨模块协同覆盖经由 `tests/test_tensor.rs` / `tests/test_math.rs` / `tests/test_reduction.rs` / `tests/test_conversion.rs` 等已存在的集成测试间接验证（与 `28-tests.md §9.2` 覆盖映射一致；**不**新增独立 `tests/test_element.rs`）
  - 内容: 各类型各层 trait 的完整性验证
  - 测试: 见测试计划 §8
  - 前置: T10, T11
  - 预计: 10 min

---

## 8. 测试计划

### 8.1 测试分类表

| 测试分类 | 位置                      | 说明                                       |
| -------- | ------------------------- | ------------------------------------------ |
| 单元测试 | `#[cfg(test)] mod tests`  | 验证各 trait 和基础类型实现                |
| 集成测试 | `tests/test_tensor.rs` / `tests/test_math.rs` / `tests/test_reduction.rs` / `tests/test_conversion.rs` | 通过张量/数学/归约/转换层间接验证 element 协同路径（**不**新增独立 `tests/test_element.rs`，与 `28-tests.md §9.2` 一致） |
| 边界测试 | 同模块测试中标注          | 覆盖 NaN/Inf、bool 限制与 sealed 行为      |
| 属性测试 | 同模块单元测试 / `tests/property_tests.rs` | 验证零元、单位元与数学函数不变量（不依赖独立 `test_element.rs`） |

### 8.2 单元测试清单

| 测试函数                        | 测试内容                                                   | 优先级 |
| ------------------------------- | ---------------------------------------------------------- | ------ |
| `test_i32_zero_one`             | `i32::zero()==0`, `i32::one()==1`                          | 高     |
| `test_i64_zero_one`             | `i64::zero()==0`, `i64::one()==1`                          | 高     |
| `test_i32_arithmetic`           | `i32` 的 Add/Sub/Mul/Div/Neg                               | 高     |
| `test_f32_zero_one`             | `f32::zero()==0.0`, `f32::one()==1.0`                      | 高     |
| `test_f64_zero_one`             | `f64::zero()==0.0`, `f64::one()==1.0`                      | 高     |
| `test_f64_sqrt`                 | `f64::sqrt(4.0)==2.0`                                      | 高     |
| `test_f64_sin`                  | `sin(0)==0`                                                | 高     |
| `test_f64_exp_ln_inverse`       | 对 `x > 0` 且有限输入使用容差断言验证 `exp(ln(x)) ≈ x`     | 高     |
| `test_f32_nan_detection`        | `NaN.is_nan()`, `Inf.is_infinite()`                        | 高     |
| `test_bool_element_only`        | `bool::zero()==false`, `bool::one()==true`                 | 高     |
| `test_bool_not_numeric`         | bool 不满足 Numeric（编译测试）                            | 高     |
| `test_bool_cast_to_f32_fails`   | `bool` 张量 `.cast::<f32>()` 不可编译（compile-fail）      | 高     |
| `test_usize_not_element`        | `usize` 不属于 Element（编译测试）                         | 中     |
| `test_complex_f64_zero_one`     | `Complex<f64>::zero()`, `Complex<f64>::one()`              | 高     |
| `test_complex_f64_conj`         | `Complex::new(3.0, 4.0).conj() == Complex::new(3.0, -4.0)` | 高     |
| `test_complex_f32_norm`         | `Complex::new(3.0f32, 4.0f32).norm() == 5.0`               | 高     |
| `test_sealed_prevents_external` | 外部类型无法实现 Element（编译测试）                       | 中     |

### 8.3 边界测试场景

| 场景                                          | 预期行为                         |
| --------------------------------------------- | -------------------------------- |
| `f64::nan().is_nan()`                         | 返回 `true`                      |
| `f64::infinity().is_finite()`                 | 返回 `false`                     |
| `f64::sqrt(-1.0).is_nan()`                    | 返回 `true`                      |
| `f64::ln(0.0)`                                | 返回 `-Inf`                      |
| `Complex::new(f64::NAN, 0.0).norm().is_nan()` | 返回 `true`                      |
| `bool` 张量调用 `sum()`                       | 编译错误（Numeric 约束不满足）   |
| `bool` 张量调用 `.cast::<f32>()`              | 编译错误（未实现 `CastTo<f32>`） |

### 8.4 属性测试不变量

| 不变量                                    | 测试方法                  |
| ----------------------------------------- | ------------------------- |
| `A::zero() + a == a`                      | 所有 Numeric 类型，随机 a |
| `A::one() * a == a`                       | 所有 Numeric 类型，随机 a |
| `let y = a.sqrt(); (y * y) ≈ a`（容差内） | f32/f64，随机非负数 a     |
| `a.exp().ln() ≈ a`                        | f32/f64，随机有限 a       |
| `x.ln().exp() ≈ x`                        | f32/f64，随机正且有限 x   |

### 8.5 集成测试

> **不**新增独立 `tests/test_element.rs`（与 `28-tests.md §9.2` 一致）。element 模块的端到端协同路径通过下列已存在的集成测试间接覆盖：

| 集成测试文件             | 覆盖的 element 协同路径                                                |
| ------------------------ | ---------------------------------------------------------------------- |
| `tests/test_tensor.rs`   | `Element` / `Numeric` bound 在张量构造、`storage_kind`、`access_semantics` 等路径上 |
| `tests/test_math.rs`     | `RealScalar` / `ComplexScalar` 数学函数语义                            |
| `tests/test_reduction.rs`| 元素类型与归约结果类型协同（`Numeric` bound 在 sum 路径；当前 reduction 仅 sum，dot 在 matrix 测试覆盖；matrix-matrix multiplication / matmul 在 `01-architecture.md §2.2` 范围外） |
| `tests/test_matrix.rs`   | dot 中 `Numeric` bound 协同路径（matrix-matrix multiplication / matmul 在 `01-architecture.md §2.2` 范围外）                               |
| `tests/test_conversion.rs` | `CastElement` 在 cast 路径上的 6×6 矩阵覆盖（文件名与 `01-architecture.md §3` ./tests 目录树严格一致） |

### 8.6 Feature gate / 配置测试

| 配置项         | 覆盖方式              | 说明                                        |
| -------------- | --------------------- | ------------------------------------------- |
| 默认配置       | 常规单元/集成测试路径 | 本模块无独立 feature gate，默认配置即主路径 |
| 非默认 feature | 不适用                | 本模块未定义 feature gate，故无额外配置矩阵 |

### 8.7 类型边界 / 编译期测试

| 测试类型       | 覆盖方式                                | 说明                                         |
| -------------- | --------------------------------------- | -------------------------------------------- |
| sealed 边界    | compile-fail 测试外部类型实现 `Element` | 验证封闭元素集合不会被外部 crate 扩展        |
| 元素能力边界   | compile-fail 测试 `bool.cast::<f32>()`  | 验证布尔元数据类型不会进入算术元素层         |
| 元素能力边界   | compile-fail 测试 `usize: Element`      | 验证索引元数据类型不会进入算术元素层         |
| trait 分层边界 | 编译期验证 `RealScalar`/`ComplexScalar` 仅覆盖规定类型 | 验证 trait 能力分层不被误扩展 |

---

## 9. 与其他模块的交互

### 9.1 接口约定

| 模块         | 使用的 trait                | 用途                   |
| ------------ | --------------------------- | ---------------------- |
| `overload`   | `Numeric`                   | 逐元素运算泛型约束     |
| `reduction`  | `Numeric`                   | 归约运算泛型约束       |
| `tensor`     | `Element`                   | Tensor<A, D> 的 A 约束 |
| `matrix`     | `Numeric` / `ComplexScalar` | 内积运算               |
| `convert`    | `Element`                   | 类型转换               |
| `math`       | 全部 Traits                 | 数学运算               |

各模块的详细接口约定参见对应设计文档（`11-math.md` §4、`13-reduction.md` §4、`21-type.md` §4）。

### 9.2 数据流描述

```text
Upstream modules declare element bounds
    │
    ├── tensor accepts the sealed element set via `Element` (excluding `usize`)
    ├── math / matrix / reduction select capabilities via `Numeric`, `RealScalar`, and `ComplexScalar`
    ├── convert / set / format continue consuming type-level capabilities or formatting semantics
    └── unsupported element types are rejected by compile-time trait bounds
```

---

## 10. 错误处理与语义边界

| 项目              | 内容                                                                                                                           |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Recoverable error | 有损 `CastTo` 默认返回可恢复错误；`Complex<T> -> Real` 在虚部非零时也返回可恢复错误；对外统一使用 `XenonError::TypeConversion` |
| Panic             | 本模块 trait 方法本身不以 panic 作为常规错误语义；若底层标准库数学实现遇到其自身前置条件，遵循标准库行为                       |
| 路径一致性        | scalar 路径必须与普通标量实现一致；SIMD：不适用；parallel：不适用                                                              |
| 容差边界          | 浮点相关比较遵循 IEEE 754 与各测试中显式容差；整数与布尔类型不适用                                                             |

---

## 11. 设计决策记录

### 决策 1：封闭集合，不支持下游扩展

| 属性     | 值                                                                           |
| -------- | ---------------------------------------------------------------------------- |
| 决策     | 所有 trait 继承 Sealed，仅允许 crate 内类型实现                              |
| 理由     | API 稳定性（可添加新方法不破坏外部）；所有实现类型行为经过验证；版本控制能力 |
| 替代方案 | 开放实现 — 放弃，失去版本控制能力，可能导致不一致行为                        |

### 决策 2：仅支持 7 种元素类型

| 属性     | 值                                                                                                    |
| -------- | ----------------------------------------------------------------------------------------------------- |
| 决策     | 仅支持 i32/i64/f32/f64/Complex<f32>/Complex<f64>/bool 作为张量元素类型；`usize` 仅作为索引/形状元数据 |
| 理由     | 科学计算元素类型需要稳定且平台无关的数值语义；`usize` 作为平台相关的无符号宽度，不适合作为数值元素    |
| 替代方案 | 支持全部整数类型（u8/u16/u32/i8/i16）— 放弃，增加矩阵复杂度                                           |

### 决策 3：bool 排除 Numeric

| 属性     | 值                                                                         |
| -------- | -------------------------------------------------------------------------- |
| 决策     | `bool` 仅实现 `Element`，不实现 `Numeric`                                  |
| 理由     | 布尔四则运算无数学意义；防止 `sum([true, false])` 等无意义操作；编译时阻止 |
| 替代方案 | bool 实现 Numeric（true=1, false=0）— 放弃，语义不清晰                     |

### 决策 4：usize 不属于元素 trait 集合

| 属性     | 值                                                                                                                          |
| -------- | --------------------------------------------------------------------------------------------------------------------------- |
| 决策     | `usize` 不实现 `Element`/`Numeric`/`RealScalar`/`ComplexScalar`，仅用于索引和形状                                           |
| 理由     | `usize` 在 Xenon 中承担索引和形状元数据语义，而不是数值计算语义；其平台相关位宽会引入跨平台差异，不适合作为科学计算元素类型 |
| 替代方案 | 让 `usize` 作为元素类型存在但排除在 `Numeric` 之外 — 放弃，仍会混淆元素集合与索引语义                                       |

### 决策 5：不支持自动类型提升

| 属性     | 值                                                                     |
| -------- | ---------------------------------------------------------------------- |
| 决策     | 类型转换须显式，不支持隐式提升                                         |
| 理由     | 显式优于隐式，避免精度损失；性能可预测；与 Rust 哲学一致               |
| 替代方案 | 类似 C++ 的类型提升规则 — 放弃，增加复杂度，可能导致难以调试的精度问题 |

### 决策 6：RealScalar 和 ComplexScalar 平行继承 Numeric

| 属性     | 值                                                                                  |
| -------- | ----------------------------------------------------------------------------------- |
| 决策     | 两者都继承 Numeric，无交叉继承                                                      |
| 理由     | 提供正交的数学函数集；复数无自然全序（不应实现 PartialOrd）；未来可扩展其他标量类型 |
| 替代方案 | ComplexScalar 继承 RealScalar — 放弃，语义不正确                                    |

---

## 12. 性能考量

| 方面         | 设计决策                                           |
| ------------ | -------------------------------------------------- |
| 零运行时开销 | 所有 trait 约束为编译期静态分派，无虚调用          |
| 内联         | RealScalar 数学方法标注 `#[inline]`                |
| 单态化       | `Tensor<A, D>` 中 A 的 trait 约束在编译期单态化    |
| Sealed       | 封闭集合允许编译器做更激进的优化（已知完整类型集） |

---

## 13. 平台与工程约束

| 约束       | 说明                                          |
| ---------- | --------------------------------------------- |
| `std` only | 本模块依赖 `std` 环境，不讨论 `no_std`        |
| MSRV       | Rust 1.85+                                    |
| 单 crate   | 保持单 crate 边界                             |
| SemVer     | 公开 trait、类型约束与转换语义变更遵循 SemVer |
| 最小依赖   | 无新增第三方依赖                              |

---

## 版本历史

| 版本  | 日期       |
| ----- | ---------- |
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-07 |
| 1.0.2 | 2026-04-08 |
| 1.0.3 | 2026-04-08 |
| 1.1.0 | 2026-04-08 |
| 1.2.0 | 2026-04-08 |
| 1.2.1 | 2026-04-10 |
| 1.2.2 | 2026-04-14 |
| 1.2.3 | 2026-04-14 |
| 1.2.4 | 2026-04-14 |
| 1.2.5 | 2026-04-15 |
| 1.2.6 | 2026-04-15 |
| 1.2.7 | 2026-04-15 |
| 1.2.8 | 2026-04-16 |
| 1.2.9 | 2026-05-03 |
| 1.3.0 | 2026-05-03 |
| 1.3.1 | 2026-05-03 |
| 1.4.0 | 2026-05-03 |

### v1.4.0 (2026-05-03) — ElementType 回归 element + Element 新增 ELEMENT_TYPE_NAME

> 本版本与 `26-error.md v3.2.0` 协同。语义破坏性变更：`ElementType` 类型定义所在模块从 `crate::error` 改回 `crate::element`，同时 `Element` trait 新增 `ELEMENT_TYPE_NAME: &'static str` 关联常量。下游编译影响：(1) 路径 `crate::error::ElementType` 不再存在（`crate::element::ElementType` 是唯一权威路径，`crate::ffi::ElementType` 是 ffi re-export）；(2) 每个 `impl Element for X` 必须新增 `const ELEMENT_TYPE_NAME: &'static str = "..."` 行，值与 `ElementType::name()` 严格一致；(3) `XenonError::TypeConversion::source_type` / `target_type` 与 `AbiMismatchKind::ElementTypeMismatch::expected` / `actual` 字段类型从 `ElementType` 改为 `&'static str`。

**契约更新：**

- §5.1 + §5.1.1：`ElementType` 枚举回到本模块；`Element` trait 新增 `ELEMENT_TYPE_NAME: &'static str`；新增 `ElementType::name() -> &'static str` 与同名自由函数 `element_type_name_of::<A>()`。
- §4.1：本模块**不**反向依赖 `error`；error 改用 `&'static str` 字段，详见 `26-error.md v3.2.0`。
- §5.9.1（无变化）`CastElement` 保持 v1.3.0 形态。
- §5.7（无变化）越界词汇仍删除。

**与 v1.3.x 的取舍：** v1.3.x 把 `ElementType` 下沉到 L0 是为了让 error 自带枚举 Display；v1.4.0 改用"error 持有 `&'static str`，element 拥有枚举"的双层方案，两边职责更清晰，error 仍能直接 Display 类型名（写字符串即可），且不再有 v1.3.1 中跨模块 inherent impl E0116 的限制——`impl ElementType { fn of::<A: Element>() }` 可以正常存在于 element 模块。

### v1.3.1 (2026-05-03) — element_type_of 自由函数

> v1.3.0 计划中作为 `element` inherent impl 的 `ElementType::of::<A>()` 因 Rust E0116（跨模块 inherent impl 禁止）改为自由函数 `element_type_of::<A>()`。该决策在 v1.4.0 因 `ElementType` 类型回归 `element` 模块而失效，v1.4.0 同时支持 inherent 与自由函数两种入口。

### v1.3.0 (2026-05-03) — ElementType 下沉到 L0 + 新增 CastElement + 删除越界词汇

> 本版本是 P0 修复任务（C2 + C1 + C14）的落地，与 `26-error.md v3.1.0` 协同。

**契约更新（破坏性内部变更，公开 trait 名称兼容）**：

- §4.1 / §4.2 / §5.1 / §5.1.1：v1.4.0 起 `ElementType` 枚举的权威定义**重新回到 `crate::element`**（不再下沉到 `error`），同时 `error` 模块改用 `&'static str` 字段（值来自 `Element::ELEMENT_TYPE_NAME` 关联常量）记录类型诊断信息——L0 单向依赖仍然成立，因为 `error` 持有的是字面量字符串而非枚举类型。`Element::ELEMENT_TYPE` 关联常量类型与定义都回到本模块；新增 `Element::ELEMENT_TYPE_NAME: &'static str`，由每个 Element impl 设置（与 `ElementType::name()` 严格一致）。`element_type_of::<A>()` 自由函数保留作为便利入口；同时新增 `impl ElementType { pub const fn of<A: Element>() -> Self }` inherent 形式（v1.3.x 因跨模块 E0116 不可用，v1.4.0 类型回 element 后已可用）。详见 §5.1.1 决策表与 v1.4.0 changelog 段。
- §5.9.1：新增 `CastElement` 公开 sealed marker trait——标记"可作为 `cast()` 操作源/目标元素类型集合"。封闭实现集合：i32 / i64 / f32 / f64 / Complex<f32> / Complex<f64>（排除 bool）。`21-type.md v2.1.0 §5.1` 通过 `use crate::element::CastElement;` 消费。这填补了 v1.x 中 `21-type.md` 引用 `CastElement` 但 `03-element.md` 未定义的 owner 缺口。
- §5.8：sealed trait 列表新增 `CastElement`。

**协同与一致性更新**：

- §5.7 跨模块运算矩阵（非规范性索引）：删除 `mean`、`min` / `max` 行，与 `require.md §14`（当前版本仅 `sum`）一致；保留 `sum 归约` 一行并标注 `mean/min/max 不在范围`。

### v1.2.9 (2026-05-03) — Medium/Low 文档修复

- 将跨模块运算矩阵标注为非规范性索引，并为 `unique` 复数支持补充 `14-set.md §6.1` 的哈希查重与 F-order 输出引用。
- 澄清 `CastTo<T>` 通过 `Self: Element` 间接 sealed，并同步 `RealScalar` 实现任务的方法范围。
- 移除 element 单元测试中属于 math/reduction 语义的 `min(NaN, 1.0)` 测试项。
- 修正文档表格格式、`全部 Traits` 术语空格，并为最近版本补充变更摘要。

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
