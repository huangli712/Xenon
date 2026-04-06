# Senon 文档与示例设计文档

> **文档版本**: v1.0  
> **最后更新**: 2026-03-28  
> **模块路径**: `docs/`  
> **需求来源**: require-v18.md §17

---

## 1. 模块概述

### 1.1 文档在科学计算库中的重要性

高质量的文档是科学计算库成功的关键因素。对于 Senon 这样的 Rust N 维张量库，文档不仅是 API 参考，更是用户理解设计哲学、正确使用功能、避免常见陷阱的指南。文档体系的重要性体现在：

| 优势 | 说明 |
|------|------|
| **降低学习曲线** | 清晰的快速入门和示例帮助新用户快速上手 |
| **减少支持成本** | 完善的文档减少重复性问题，降低维护负担 |
| **提升代码质量** | 文档测试（doctest）确保示例代码始终正确 |
| **促进社区贡献** | 良好的文档结构便于贡献者理解项目架构 |
| **建立信任** | 专业、完整的文档传达项目成熟度和可靠性 |

**文档设计原则**：

```rust
// Documentation should be:
// 1. Accurate - reflect actual behavior
// 2. Complete - cover all public APIs
// 3. Clear - understandable to target audience
// 4. Consistent - uniform style and terminology
// 5. Maintainable - easy to update with code changes
```

### 1.2 设计目标

1. **全覆盖**: 所有公开 API 必须有文档注释（require-v18.md §17.1）
2. **可测试**: 关键 API 的使用示例通过 `cargo test --doc` 验证
3. **安全性透明**: 所有 unsafe 函数必须有 Safety 文档节
4. **层次分明**: Crate 级、模块级、类型级、函数级文档各司其职
5. **用户友好**: 提供丰富的示例代码和常见用例

### 1.3 在架构中的位置

```
文档层次结构：

L0: Crate-level (lib.rs)
    └── 项目概述、快速入门、特性列表
    
L1: Module-level (mod.rs)
    └── 模块职责、核心概念、类型关系
    
L2: Submodule-level (submodule/mod.rs)
    └── 子模块职责、实现细节
    
L3: Type/Function-level (doc comments)
    └── API 文档、参数说明、使用示例

L4: Examples (examples/)
    └── 完整示例程序、教程代码

文档覆盖所有模块，与代码同步演进，通过 CI 自动验证。
```

### 1.4 文档分类

| 层级 | 位置 | 内容 | 目标读者 |
|------|------|------|----------|
| L0 | `lib.rs` 顶层注释 | 项目介绍、快速入门、特性概览 | 新用户 |
| L1 | 各 `mod.rs` | 模块概述、设计决策、类型关系 | 库开发者 |
| L2 | 子模块 `mod.rs` | 子模块职责、实现策略 | 贡献者 |
| L3 | struct/trait/fn 注释 | API 详细说明、参数、返回值、示例 | 所有用户 |
| L4 | `examples/` | 完整可运行示例 | 新用户、集成者 |

### 1.5 需求来源

来自 `require-v18.md` 第 17.1 节：

- doc comment：所有公开 API
- Safety 文档节：所有 unsafe 函数
- 使用示例：关键 API

---

## 2. 文件结构

### 2.1 docs/ 目录组织

```
docs/
├── architecture/           # 架构设计文档
│   ├── 01-overview.md      # 项目概述
│   ├── 02-dimension.md     # 维度系统设计
│   ├── 03-storage.md       # 存储系统设计
│   ├── 04-layout.md        # 内存布局设计
│   ├── 05-tensor.md        # 张量核心设计
│   ├── 05-05-broadcast.md  # 广播机制设计
│   ├── 06-ops.md           # 运算操作设计
│   ├── 07-iterators.md     # 迭代器设计
│   └── 08-documentation.md # 本文档
│
├── src/                    # 文档源码（用于 docs.rs）
│   └── lib.rs              # 自动生成的文档入口
│
└── README.md               # 项目 README
```

### 2.2 examples/ 目录组织

```
examples/
├── basic.rs                # 基础张量创建与操作
├── matrix_ops.rs           # 矩阵运算示例
├── no_std.rs               # no_std 环境示例
├── parallel.rs             # 并行计算示例
├── broadcasting.rs         # 广播机制示例
├── slicing.rs              # 切片与视图示例
├── performance.rs          # 性能优化技巧
└── ffi_integration.rs      # FFI 集成示例
```

### 2.3 模块职责

| 组件 | 职责 |
|------|------|
| `docs/architecture/` | 架构设计文档，供开发者和贡献者参考 |
| `docs/src/` | docs.rs 自动生成的 API 文档源码 |
| `examples/` | 完整可运行的示例程序 |
| doc comments | 内联 API 文档，通过 `cargo doc` 生成 |

### 2.4 文档层次结构

```
文档层次
├── L0: Crate 级文档
│   ├── README.md              — 项目介绍、安装、快速入门
│   ├── lib.rs 顶层文档        — crate 概述、feature 说明、架构导览
│   └── CHANGELOG.md           — 版本变更记录
│
├── L1: 模块级文档
│   ├── 每个 mod.rs 顶部       — 模块职责、核心类型、使用指南
│   └── 模块间关系说明          — 依赖方向、数据流
│
├── L2: 子模块级文档
│   └── 子模块 mod.rs          — 子模块职责、实现策略
│
├── L3: 类型/函数级文档
│   ├── struct/enum/trait       — 类型说明、泛型参数、生命周期
│   ├── 方法                    — 参数、返回值、错误条件、示例
│   └── unsafe 函数             — Safety 前提条件
│
└── L4: 示例与教程
    ├── examples/ 目录          — 独立可运行示例
    ├── doctest                 — 内嵌在文档中的代码片段
    └── tests/doc_examples.rs   — 文档示例的集成验证
```

### 2.5 各层覆盖要求

| 层次 | 覆盖率要求 | 验证方式 |
|------|-----------|----------|
| L0 | 必须存在 | CI 检查文件存在性 |
| L1 | 每个 pub mod 必须有模块文档 | `#![warn(missing_docs)]` |
| L2 | 子模块有文档 | `#![warn(missing_docs)]` |
| L3 | 每个 pub 项必须有 doc comment | `#![warn(missing_docs)]` |
| L4 | 关键 API 必须有至少一个示例 | `cargo test --doc` |

---

## 3. Crate 级文档设计（L0）

### 3.1 lib.rs 顶层文档结构

Crate 级文档是用户接触项目的第一入口，需要清晰传达项目定位和快速入门指南。

```rust
//! # Senon — N-dimensional Tensor Library for Rust
//!
//! Senon is a high-performance N-dimensional array (tensor) library for Rust,
//! designed as numerical infrastructure for scientific computing — similar to
//! NumPy in the Python ecosystem.
//!
//! ## Design Goals
//!
//! - **Correctness first**: type safety, memory safety, numerical precision
//! - **Clear abstractions**: explicit API semantics, no implicit behavior
//! - **Scientific computing oriented**: column-major default, SIMD-friendly, BLAS-compatible memory layout
//! - **Minimal dependencies**: only rayon (optional) and pulp (optional)
//!
//! ## Quick Start
//!
//! ### Creating Tensors
//!
//! ```rust
//! use Senon::{Tensor, Tensor1, Tensor2, Ix2};
//!
//! // Create a 1D tensor (vector)
//! let v: Tensor1<f64> = Tensor::zeros(5);
//!
//! // Create a 2D tensor (matrix)
//! let m: Tensor2<f64> = Tensor::zeros([3, 4]);
//!
//! // Create from data
//! let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0])?;
//! let matrix = data.into_shape([2, 2])?;
//! # Ok::<(), Senon::Error>(())
//! ```
//!
//! ### Basic Operations
//!
//! ```rust
//! use Senon::{Tensor, Tensor2};
//!
//! let a: Tensor2<f64> = Tensor::from_shape_vec([2, 2], vec![1.0, 2.0, 3.0, 4.0])?;
//! let b: Tensor2<f64> = Tensor::from_shape_vec([2, 2], vec![5.0, 6.0, 7.0, 8.0])?;
//!
//! // Element-wise operations (support broadcasting)
//! let sum = &a + &b;
//! let prod = &a * &b;
//!
//! // Matrix-vector multiplication
//! let v = Tensor::from_vec(vec![1.0, 2.0])?;
//! let result = a.matvec(&v)?;
//! # Ok::<(), Senon::Error>(())
//! ```
//!
//! ### Broadcasting
//!
//! ```rust
//! use Senon::{Tensor, Tensor2};
//!
//! let matrix: Tensor2<f64> = Tensor::zeros([3, 4]);
//! let row: Tensor2<f64> = Tensor::from_shape_vec([1, 4], vec![1.0, 2.0, 3.0, 4.0])?;
//!
//! // Row broadcasts to match matrix shape
//! let result = &matrix + &row;
//! assert_eq!(result.shape(), &[3, 4]);
//! # Ok::<(), Senon::Error>(())
//! ```
//!
//! ## Features
//!
//! | Feature | Default | Description |
//! |---------|:-------:|-------------|
//! | `std` | ✓ | Standard library support (enables heap allocation) |
//! | `parallel` | ✗ | Data parallelism via rayon |
//! | `simd` | ✗ | SIMD acceleration via pulp |
//!
//! ### Enabling Features
//!
//! ```toml
//! [dependencies]
//! Senon = { version = "0.1", features = ["parallel", "simd"] }
//! ```
//!
//! ## Supported Dimension Types
//!
//! | Type | Meaning | Usage |
//! |------|---------|-------|
//! | `Ix0` | 0-dimensional (scalar) | Scalar computation |
//! | `Ix1` | 1-dimensional (vector) | Vector operations |
//! | `Ix2` | 2-dimensional (matrix) | Matrix operations |
//! | `Ix3`-`Ix6` | 3-6 dimensional | Higher-order tensors |
//! | `IxDyn` | Dynamic dimension | Runtime-determined rank |
//!
//! ## Supported Element Types
//!
//! | Level | Types | Trait Bound |
//! |-------|-------|-------------|
//! | Base | integers, floats, complex, bool | `Element` |
//! | Numeric | integers, floats, complex | `Numeric: Element` |
//! | Real | f32, f64 | `RealScalar: Numeric` |
//! | Complex | Complex<f32>, Complex<f64> | `ComplexScalar: Numeric` |
//!
//! ## Memory Layout
//!
//! Default layout is **F-order (column-major)**, compatible with BLAS/LAPACK:
//!
//! ```rust
//! use Senon::{Tensor, Order, Tensor2};
//!
//! // Create F-order tensor (default)
//! let f_order: Tensor2<f64> = Tensor::zeros([3, 4]);
//!
//! // Create C-order tensor
//! let c_order: Tensor2<f64> = Tensor::zeros_with_order([3, 4], Order::C);
//! ```
//!
//! ## no_std Support
//!
//! Senon supports `no_std` environments (requires `alloc` crate):
//!
//! ```rust
//! // In Cargo.toml:
//! // [dependencies]
//! // Senon = { version = "0.1", default-features = false }
//! ```
//!
//! ## License
//!
//! MIT 或 Apache-2.0

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(rustdoc::missing_crate_level_docs)]
```

### 3.2 特性概览表格

在 Crate 级文档中，使用表格清晰展示特性：

| 特性名称 | 默认启用 | 依赖 | 描述 |
|----------|:--------:|------|------|
| `std` | ✓ | - | 标准库支持，启用堆分配 |
| `parallel` | ✗ | `rayon`, `std` | 数据并行，使用 rayon |
| `simd` | ✗ | `pulp` | SIMD 加速 |

### 3.3 快速入门示例

快速入门应覆盖最常用的 80% 用例：

1. **创建张量**: `zeros`, `ones`, `from_vec`
2. **基本运算**: 算术运算符、广播
3. **形状操作**: `reshape`, `slice`
4. **归约**: `sum`, `mean`, `max`

### 3.4 README.md 结构

```markdown
# Senon

Rust scientific computing N-dimensional array library.

## Features

- N-dimensional arrays with static (0-6D) and dynamic dimensions
- Column-major default, BLAS-compatible memory layout
- Custom FFI-friendly complex number type
- Optional SIMD (pulp) and parallel (rayon) acceleration
- no_std support (requires alloc)

## Installation

## Quick Start

## Documentation

## License
```


---

## 4. 模块级文档设计（L1/L2）

### 4.1 模块文档结构

每个 `mod.rs` 应包含以下内容：

```rust
//! # [模块名称]
//!
//! [模块简介，1-2 句话描述模块职责]
//!
//! ## 概述
//!
//! [详细描述模块的功能和设计目标]
//!
//! ## 核心概念
//!
//! [介绍模块中的核心类型和它们之间的关系]
//!
//! ## 使用示例
//!
//! ```rust
//! // 展示模块主要用法的示例
//! use Senon::module_name::SomeType;
//!
//! let example = SomeType::new();
//! ```
//!
//! ## 与其他模块的关系
//!
//! [描述依赖关系和交互点]
//!
//! ## 设计决策
//!
//! [重要的设计决策和理由]

mod submodule;
```

### 4.2 storage 模块示例

```rust
//! # 存储系统
//!
//! 提供张量的底层存储抽象，支持多种所有权模式。
//!
//! ## 概述
//!
//! 存储系统是 Senon 的基础层，定义了数据如何被拥有、借用和访问。
//! 通过泛型参数 `S`，`TensorBase<S, D>` 可以使用不同的存储模式，
//! 实现所有权语义的编译时保证。
//!
//! ## 核心概念
//!
//! ### 存储模式
//!
//! | 存储模式 | 拥有数据 | 可读 | 可写 | 典型用途 |
//! |----------|:--------:|:----:|:----:|----------|
//! | `Owned<A>` | ✓ | ✓ | ✓ | 数组创建、运算结果 |
//! | `ViewRepr<&A>` | ✗ | ✓ | ✗ | 只读视图、切片 |
//! | `ViewMutRepr<&mut A>` | ✗ | ✓ | ✓ | 原地修改 |
//! | `ArcRepr<A>` | 共享 | ✓ | CoW | 跨线程共享 |
//!
//! ### Storage Trait
//!
//! 所有存储模式实现 `Storage` trait：
//!
//! ```rust
//! use Senon::storage::Storage;
//!
//! /// Storage trait defines common operations for all storage modes
//! pub trait Storage {
//!     /// Element type
//!     type Elem;
//!     /// Get a slice of the underlying data
//!     fn as_slice(&self) -> &[Self::Elem];
//! }
//! ```
//!
//! ## 使用示例
//!
//! ### Owned 存储
//!
//! ```rust
//! use Senon::{Tensor, Tensor2};
//!
//! // Create owned tensor
//! let mut tensor: Tensor2<f64> = Tensor::zeros([3, 4]);
//! tensor[[0, 0]] = 1.0;
//! ```
//!
//! ### View 存储
//!
//! ```rust
//! use Senon::{Tensor, Tensor2, TensorView};
//!
//! let tensor: Tensor2<f64> = Tensor::zeros([3, 4]);
//! let view: TensorView<f64, _> = tensor.view();
//! // view is read-only
//! ```
//!
//! ### ViewMut 存储
//!
//! ```rust
//! use Senon::{Tensor, Tensor2, TensorViewMut};
//!
//! let mut tensor: Tensor2<f64> = Tensor::zeros([3, 4]);
//! let view_mut: TensorViewMut<f64, _> = tensor.view_mut();
//! view_mut[[0, 0]] = 1.0;
//! ```
//!
//! ### ArcRepr 存储
//!
//! ```rust
//! use Senon::{ArcTensor, Tensor2};
//! use std::sync::Arc;
//!
//! let tensor: ArcTensor<f64, _> = ArcTensor::zeros([3, 4]);
//! let cloned = tensor.clone(); // Cheap reference count increment
//! ```
//!
//! ## 与其他模块的关系
//!
//! ```
//! storage ←── error
//!     ↓
//! layout ←── storage
//!     ↓
//! tensor ←── storage, layout, dimension
//! ```
//!
//! ## 设计决策
//!
//! ### 为什么使用泛型存储而非枚举？
//!
//! 泛型存储允许编译器在编译时确定访问权限，避免运行时检查。
//! 例如，`TensorView` 类型保证只读，编译器会阻止写操作。

pub mod owned;
pub mod view;
pub mod view_mut;
pub mod arc;
```

### 4.3 ops 模块示例

```rust
//! # 运算操作
//!
//! 提供张量的数学运算，包括逐元素运算、归约、矩阵运算等。
//!
//! ## 概述
//!
//! ops 模块实现了 Senon 的计算能力，通过运算符重载和命名函数
//! 提供直观的数学表达式支持。所有运算都支持广播机制。
//!
//! ## 核心功能
//!
//! ### 逐元素运算
//!
//! 支持整数、浮点、复数类型（不含 bool）：
//!
//! - 算术: `add`, `sub`, `mul`, `div`
//! - 三角: `sin`, `cos`, `tan`
//! - 指数/对数: `exp`, `ln`, `log2`, `log10`
//! - 数值: `abs`, `sign`, `floor`, `ceil`, `round`
//!
//! ### 归约运算
//!
//! - 全局: `sum`, `prod`, `mean`, `var`, `std`, `min`, `max`
//! - 沿轴: 所有全局归约均支持 `axis` 参数
//! - 累积: `cumsum`, `cumprod`
//!
//! ### 矩阵运算
//!
//! - `matvec`: 矩阵-向量乘法
//! - `dot`/`inner`: 内积
//! - `outer`: 外积
//!
//! ## 使用示例
//!
//! ```rust
//! use Senon::{Tensor, Tensor2};
//!
//! let a: Tensor2<f64> = Tensor::zeros([3, 4]);
//! let b: Tensor2<f64> = Tensor::zeros([3, 4]);
//!
//! // Arithmetic with broadcasting
//! let sum = &a + &b;
//! let diff = &a - &b;
//! let prod = &a * &b;
//!
//! // Reduction
//! let total = a.sum();
//! let row_sums = a.sum_axis(1);
//! ```
//!
//! ## 运算符重载
//!
//! | 运算符 | 含义 | 示例 |
//! |--------|------|------|
//! | `+` | 加法 | `a + b` |
//! | `-` | 减法 | `a - b` |
//! | `*` | 乘法 | `a * b` |
//! | `/` | 除法 | `a / b` |
//! | `-` (一元) | 负号 | `-a` |
//! | `+=` | 原地加 | `a += b` |
//! | `-=` | 原地减 | `a -= b` |
//! | `*=` | 原地乘 | `a *= b` |
//! | `/=` | 原地除 | `a /= b` |
//!
//! ## 约束
//!
//! - 逐元素算术运算仅适用于数值类型（`Numeric` trait）
//! - bool 类型仅支持逻辑运算
//! - 原地运算的 RHS 可广播，LHS 必须拥有完整存储

mod arithmetic;
mod reduction;
mod linalg;
mod math;
```

### 4.4 各模块文档要点

| 模块 | 必须说明的要点 |
|------|---------------|
| `dimension/` | Ix0-Ix6 与 IxDyn 的选择指南、互转规则、Ix0 标量语义 |
| `element/` | 四层 trait 体系图、各层支持的操作、bool 的特殊地位 |
| `complex/` | repr(C) 布局保证、与 C99 _Complex 的兼容性、hypot 算法 |
| `storage/` | 四种存储模式对比表、所有权与借用关系、ArcRepr CoW 语义 |
| `layout/` | F-order vs C-order 选择指南、布局标志位含义、对齐策略 |
| `tensor/` | TensorBase 泛型参数说明、类型别名速查表、视图创建方式 |
| `iter/` | 迭代器类型选择指南、遍历顺序说明、zip 广播行为 |
| `ops/` | 运算分类（逐元素/矩阵/归约）、类型约束速查、广播规则 |
| `shape_ops/` | 零拷贝 vs 需拷贝分类、各操作的布局影响 |
| `index/` | 索引类型对比、s![] 宏语法速查、高级索引使用场景 |
| `construct/` | 构造方法速查表、Order 参数说明、cast 精度行为 |
| `ffi/` | 指针 API 安全约定、BLAS 兼容性检查流程、from_raw_parts 前提条件 |
| `workspace/` | 工作空间生命周期、分割与嵌套、scratch 查询模式 |
| `simd/` | 启用方式、支持的指令集、回退策略、性能分层 |
| `parallel/` | 启用方式、并行阈值、嵌套并行禁止规则 |
| `error/` | 错误分类（Result vs panic）、错误类型速查 |

### 4.5 模块文档规范

| 要素 | 要求 | 示例 |
|------|------|------|
| 模块简介 | 1-2 句话 | "提供张量的底层存储抽象" |
| 核心概念 | 列出关键类型 | 存储模式表格 |
| 使用示例 | 覆盖主要用例 | Owned/View/ViewMut/Arc |
| 模块关系 | ASCII 依赖图 | `storage → layout → tensor` |
| 设计决策 | 解释关键选择 | "为什么用泛型而非枚举" |

---

## 5. 类型与函数级文档设计（L3）

### 5.1 struct 文档规范

所有公开 struct 必须有完整的文档注释：

```rust
/// N 维张量的核心数据结构
///
/// `TensorBase<S, D>` 是 Senon 的核心类型，通过泛型参数支持
/// 不同的存储模式和维度类型。
///
/// # 类型参数
///
/// * `S` - 存储模式，实现 [`Storage`] trait
/// * `D` - 维度类型，实现 [`Dimension`] trait
///
/// # 类型别名
///
/// 推荐使用便捷类型别名：
///
/// | 别名 | 展开 | 用途 |
/// |------|------|------|
/// | `Tensor<A, D>` | `TensorBase<Owned<A>, D>` | 拥有数据的数组 |
/// | `TensorView<'a, A, D>` | `TensorBase<ViewRepr<&'a A>, D>` | 只读视图 |
/// | `TensorViewMut<'a, A, D>` | `TensorBase<ViewMutRepr<&'a mut A>, D>` | 可变视图 |
/// | `ArcTensor<A, D>` | `TensorBase<ArcRepr<A>, D>` | 共享所有权 |
///
/// # 示例
///
/// ## 创建张量
///
/// ```rust
/// use Senon::{Tensor, Tensor2, Ix2};
///
/// // Create a 2D tensor
/// let tensor: Tensor2<f64> = Tensor::zeros([3, 4]);
/// assert_eq!(tensor.shape(), &[3, 4]);
/// ```
///
/// ## 访问元素
///
/// ```rust
/// use Senon::{Tensor, Tensor2};
///
/// let mut tensor: Tensor2<f64> = Tensor::zeros([3, 4]);
/// tensor[[0, 0]] = 1.0;
/// assert_eq!(tensor[[0, 0]], 1.0);
/// ```
///
/// ## 视图与切片
///
/// ```rust
/// use Senon::{Tensor, Tensor2, s};
///
/// let tensor: Tensor2<f64> = Tensor::zeros([3, 4]);
/// let view = tensor.slice(s![0..2, 1..3]);
/// assert_eq!(view.shape(), &[2, 2]);
/// ```
///
/// # 内存布局
///
/// 默认使用 F-order（列优先）布局。可以通过 `layout()` 方法
/// 查询当前布局属性。
///
/// # 另见
///
/// * [`Storage`] - 存储模式 trait
/// * [`Dimension`] - 维度类型 trait
/// * [`Layout`] - 内存布局信息
pub struct TensorBase<S, D>
where
    S: Storage,
    D: Dimension,
{
    // Internal fields...
}
```

### 5.2 trait 文档规范

```rust
/// 维度类型 trait
///
/// 定义张量维度数的表示和操作。Senon 支持静态维度（0-6 维）
/// 和动态维度（运行时确定维度数）。
///
/// # 实现类型
///
/// | 类型 | ndim | 内部表示 | 用途 |
/// |------|------|----------|------|
/// | `Ix0` | 0 | `()` | 标量 |
/// | `Ix1` | 1 | `[usize; 1]` | 向量 |
/// | `Ix2` | 2 | `[usize; 2]` | 矩阵 |
/// | `Ix3`-`Ix6` | 3-6 | `[usize; N]` | 高维张量 |
/// | `IxDyn` | 动态 | `Vec<usize>` | 运行时维度 |
///
/// # 示例
///
/// ```rust
/// use Senon::{Dimension, Ix2, IxDyn};
///
/// // Static dimension
/// let dim: Ix2 = Ix2::from_slice(&[3, 4]);
/// assert_eq!(dim.ndim(), 2);
///
/// // Dynamic dimension
/// let dyn_dim: IxDyn = IxDyn::from_slice(&[3, 4, 5]);
/// assert_eq!(dyn_dim.ndim(), 3);
/// ```
///
/// # 维度转换
///
/// 静态维度可以无损转换为动态维度：
///
/// ```rust
/// use Senon::{Ix2, IxDyn, Dimension};
///
/// let static_dim = Ix2::from_slice(&[3, 4]);
/// let dyn_dim: IxDyn = static_dim.into_dimension();
/// ```
///
/// 动态维度转换为静态维度时，维度数必须匹配：
///
/// ```rust
/// use Senon::{Ix2, IxDyn, Dimension};
///
/// let dyn_dim = IxDyn::from_slice(&[3, 4]);
/// let static_dim: Ix2 = dyn_dim.into_dimension()?; // ndim must be 2
/// # Ok::<(), Senon::Error>(())
/// ```
///
/// # Sealed
///
/// This trait is sealed and cannot be implemented outside of this crate.
pub trait Dimension: Clone + Debug {
    /// 返回维度数
    fn ndim(&self) -> usize;
    
    /// 返回形状切片
    fn slice(&self) -> &[usize];
    
    /// 从切片创建维度
    fn from_slice(slice: &[usize]) -> Self;
    
    /// 转换为目标维度类型
    fn into_dimension<D>(self) -> Result<D, DimensionMismatch>
    where
        D: Dimension;
}
```

### 5.3 函数文档规范

#### 普通函数

```rust
/// 计算两个形状广播后的结果形状
///
/// 广播遵循 NumPy 规则：从最右维度开始对齐，维度数不足的数组
/// 在左侧补 1，对应维度相等或其中一个为 1 则兼容。
///
/// # 参数
///
/// * `shape_a` - 第一个形状
/// * `shape_b` - 第二个形状
///
/// # 返回值
///
/// * `Ok(result_shape)` - 广播后的形状
/// * `Err(BroadcastError)` - 形状不兼容
///
/// # 示例
///
/// ```rust
/// use Senon::broadcast::broadcast_shape;
///
/// let a = &[3, 1, 4];
/// let b = &[4, 1];
/// let result = broadcast_shape(a, b)?;
/// assert_eq!(result.as_slice(), &[3, 4, 4]);
/// # Ok::<(), Senon::BroadcastError>(())
/// ```
///
/// # 错误
///
/// 当两个形状的对应维度都不为 1 且不相等时，返回 `BroadcastError`：
///
/// ```rust
/// use Senon::broadcast::broadcast_shape;
///
/// let a = &[3, 2];
/// let b = &[3, 4];
/// let result = broadcast_shape(a, b);
/// assert!(result.is_err());
/// ```
///
/// # 另见
///
/// * [`can_broadcast`] - 检查形状兼容性
/// * [`broadcast_strides`] - 计算广播后的步长
pub fn broadcast_shape(
    shape_a: &[usize],
    shape_b: &[usize],
) -> Result<SmallVec<[usize; 6]>, BroadcastError> {
    // Implementation...
}
```

#### unsafe 函数

所有 unsafe 函数必须有 `# Safety` 文档节：

```rust
/// 从原始部件构造张量视图
///
/// 允许从原始指针、形状、步长和偏移量构造 `TensorView`，
/// 用于与 C 库或其他低级代码集成。
///
/// # 参数
///
/// * `ptr` - 数据起始指针，须非空且对齐
/// * `shape` - 各轴长度
/// * `strides` - 各轴步长（元素单位，有符号）
/// * `offset` - 数据起始偏移量（元素单位）
///
/// # 返回值
///
/// 构造的 `TensorView` 实例
///
/// # Safety
///
/// 调用方必须保证以下条件，否则导致未定义行为：
///
/// 1. **指针有效性**: `ptr` 必须非空、非悬垂，且对齐到 `align_of::<A>()`
/// 2. **内存范围**: `ptr` 起始的内存范围必须足够大，能覆盖所有可访问元素
/// 3. **生命周期**: 内存必须在返回的视图生命周期内保持有效
/// 4. **别名规则**: 内存可被共享读取，但不可被写入
/// 5. **布局一致性**: `shape` 与 `strides` 长度必须一致
/// 6. **边界安全**: 任意合法索引计算出的偏移量不得越界
/// 7. **元素初始化**: 所有可访问元素必须已正确初始化
///
/// # 示例
///
/// ```rust
/// use Senon::{TensorView, Ix2, Dimension};
///
/// // Safe wrapper around unsafe construction
/// fn from_slice_safe(data: &[f64], shape: [usize; 2]) -> TensorView<f64, Ix2> {
///     assert!(!data.is_empty());
///     assert!(data.as_ptr().aligned_to(8));
///     
///     let strides = [1, shape[0] as isize]; // F-order
///     
///     // SAFETY: We've verified pointer validity and alignment above.
///     // The slice owns the data and outlives the view.
///     unsafe {
///         TensorView::from_raw_parts(
///             data.as_ptr(),
///             Ix2::from_slice(&shape),
///             Ix2::from_isize_slice(&strides),
///             0,
///         )
///     }
/// }
/// ```
///
/// # 另见
///
/// * [`as_ptr`] - 获取数据指针
/// * [`into_raw_parts`] - 解构为原始部件
pub unsafe fn from_raw_parts<'a, A, D>(
    ptr: *const A,
    shape: D,
    strides: D,
    offset: usize,
) -> TensorView<'a, A, D>
where
    A: Element,
    D: Dimension,
{
    // Implementation...
}
```

### 5.4 方法文档规范

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element,
{
    /// 返回张量的形状
    ///
    /// # 返回值
    ///
    /// 各轴长度的切片引用
    ///
    /// # 示例
    ///
    /// ```rust
    /// use Senon::{Tensor, Tensor2};
    ///
    /// let tensor: Tensor2<f64> = Tensor::zeros([3, 4]);
    /// assert_eq!(tensor.shape(), &[3, 4]);
    /// ```
    ///
    /// # 时间复杂度
    ///
    /// O(1)
    pub fn shape(&self) -> &[usize] {
        self.shape.slice()
    }
    
    /// 返回张量的步长
    ///
    /// 步长表示沿每个轴移动一个位置所需的元素数。
    /// 有符号类型支持负步长（反转维度）。
    ///
    /// # 返回值
    ///
    /// 各轴步长的切片引用（元素单位，有符号）
    ///
    /// # 示例
    ///
    /// ```rust
    /// use Senon::{Tensor, Tensor2, Order};
    ///
    /// // F-order (column-major)
    /// let f_tensor: Tensor2<f64> = Tensor::zeros([3, 4]);
    /// assert_eq!(f_tensor.strides(), &[1, 3]);
    ///
    /// // C-order (row-major)
    /// let c_tensor: Tensor2<f64> = Tensor::zeros_with_order([3, 4], Order::C);
    /// assert_eq!(c_tensor.strides(), &[4, 1]);
    /// ```
    ///
    /// # 另见
    ///
    /// * [`strides_bytes`] - 字节单位的步长
    /// * [`is_f_contiguous`] - 检查 F-order 连续性
    /// * [`is_c_contiguous`] - 检查 C-order 连续性
    pub fn strides(&self) -> &[isize] {
        self.strides.slice()
    }
}
```

### 5.5 文档节使用规则

| 文档节 | 何时必须 | 说明 |
|--------|---------|------|
| `# Arguments` / `# 参数` | 方法有 2+ 参数时 | 描述每个参数的含义和约束 |
| `# Returns` / `# 返回值` | 返回值语义非显而易见时 | 描述返回值属性 |
| `# Errors` / `# 错误` | 函数返回 Result 时 | 列出所有可能的错误变体 |
| `# Panics` / `# Panic` | 函数可能 panic 时 | 列出所有 panic 条件 |
| `# Safety` | unsafe 函数 | 列出所有安全前提条件（**必须**） |
| `# Examples` / `# 示例` | 所有关键 API | 至少一个可编译运行的示例 |
| `# Performance` / `# 性能` | 复杂度非 O(1) 或有性能注意事项时 | 时间/空间复杂度 |
| `# See Also` / `# 另见` | 有相关 API 时 | 交叉引用相关类型/方法 |

### 5.6 文档注释检查清单

| 检查项 | 要求 | 示例 |
|--------|------|------|
| 简短描述 | 第一行简明扼要 | "计算两个形状广播后的结果形状" |
| 详细说明 | 解释行为和约束 | 广播规则、兼容条件 |
| 参数说明 | 每个参数的用途 | `shape_a` - 第一个形状 |
| 返回值 | 返回类型和含义 | `Ok(result_shape)` / `Err(BroadcastError)` |
| 示例 | 至少一个可运行示例 | `assert_eq!(result, ...)` |
| 错误 | 可能的错误情况 | `BroadcastError` 触发条件 |
| Safety | unsafe 函数必须有 | 7 条前提条件 |
| 另见 | 相关 API 链接 | [`can_broadcast`], [`broadcast_strides`] |


---

## 6. 示例设计 (L4)

### 6.1 examples/ 目录结构

```
examples/
├── quickstart.rs           — 最小入门示例（创建、运算、打印）
├── matrix_operations.rs    — 矩阵运算示例（matvec, dot, outer）
├── broadcasting.rs         — 广播机制示例
├── slicing_indexing.rs     — 切片与索引示例（s![] 宏、高级索引）
├── shape_manipulation.rs   — 形状操作示例（reshape, transpose, cat, stack）
├── complex_numbers.rs      — 复数运算示例
├── parallel_compute.rs     — 并行计算示例（需 parallel feature）
├── simd_acceleration.rs    — SIMD 加速示例（需 simd feature）
├── ffi_interop.rs          — FFI 互操作示例（与 C/BLAS 交互）
├── custom_reduction.rs     — 自定义归约示例
└── no_std_usage.rs         — no_std 环境使用示例
```

### 6.2 示例分级

| 级别 | 目标用户 | 示例 | 复杂度 |
|------|---------|------|--------|
| 入门 | 首次使用者 | quickstart.rs | < 30 行 |
| 基础 | 日常使用 | matrix_operations, broadcasting, slicing_indexing | 30-80 行 |
| 进阶 | 性能优化 | parallel_compute, simd_acceleration | 50-100 行 |
| 专家 | FFI/底层 | ffi_interop, no_std_usage | 50-100 行 |

### 6.3 示例编写规范

| 规范 | 说明 |
|------|------|
| 自包含 | 每个示例独立可运行，不依赖其他示例 |
| 有注释 | 关键步骤有行内注释说明意图 |
| 有输出 | 使用 `println!` 展示结果，方便用户验证 |
| 有 feature gate | 需要可选 feature 的示例在文件顶部注明 |
| 无 unwrap | 使用 `?` 操作符，main 返回 `Result` |

### 6.4 示例模板

```rust
//! Example: Brief description
//!
//! Demonstrates: feature A, feature B
//!
//! Run with: `cargo run --example example_name`
//! (Add `--features parallel` if needed)

use Senon::prelude::*;

fn main() -> Senon::Result<()> {
    // Step 1: Create tensors
    let a = Tensor2::<f64>::zeros([3, 4]);
    println!("Created 3x4 zero matrix: shape={:?}", a.shape());

    // Step 2: Perform operation
    let b = a.t();
    println!("Transposed: shape={:?}", b.shape());

    Ok(())
}
```

### 6.5 quickstart.rs 代码骨架

```rust
//! Example: Quick Start
//!
//! Demonstrates: tensor creation, basic arithmetic, indexing, printing
//!
//! Run with: `cargo run --example quickstart`

use Senon::prelude::*;

fn main() -> Senon::Result<()> {
    // --- Creating tensors ---

    // From explicit values
    let a = Tensor1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    println!("1D tensor: {a}");

    // Zeros / ones
    let zeros = Tensor2::<f64>::zeros([3, 4]);
    let ones = Tensor2::<f64>::ones([2, 3]);
    println!("Zeros shape: {:?}", zeros.shape());
    println!("Ones shape: {:?}", ones.shape());

    // From a flat vec with shape
    let m = Tensor2::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    println!("2D tensor:\n{m}");

    // --- Basic arithmetic ---

    let x = Tensor1::from_vec(vec![1.0, 2.0, 3.0]);
    let y = Tensor1::from_vec(vec![4.0, 5.0, 6.0]);

    let sum = &x + &y;
    let diff = &x - &y;
    let prod = &x * &y;
    println!("x + y = {sum}");
    println!("x - y = {diff}");
    println!("x * y = {prod}");

    // Scalar operations
    let scaled = &x * 2.0;
    println!("x * 2 = {scaled}");

    // --- Reductions ---

    println!("sum(x) = {}", x.sum());
    println!("mean(x) = {}", x.mean());
    println!("max(x) = {}", x.max());

    // --- Indexing ---

    let mat = Tensor2::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [2, 3],
    );
    println!("mat[0, 1] = {}", mat[[0, 1]]);

    // Slicing
    let row = mat.slice(s![0, ..]);
    println!("First row: {row}");

    Ok(())
}
```

### 6.6 broadcasting.rs 代码骨架

```rust
//! Example: Broadcasting
//!
//! Demonstrates: broadcasting rules, shape compatibility, common patterns
//!
//! Run with: `cargo run --example broadcasting`

use Senon::prelude::*;

fn main() -> Senon::Result<()> {
    // --- Rule 1: Scalar broadcasts to any shape ---

    let a = Tensor2::<f64>::ones([3, 4]);
    let result = &a + 10.0;
    println!("ones(3,4) + 10.0:\n{result}");

    // --- Rule 2: Dimensions are compatible when equal or one of them is 1 ---

    // [3, 4] + [1, 4] => [3, 4]
    let matrix = Tensor2::from_vec(
        vec![1.0, 2.0, 3.0, 4.0,
             5.0, 6.0, 7.0, 8.0,
             9.0, 10.0, 11.0, 12.0],
        [3, 4],
    );
    let row_vec = Tensor2::from_vec(vec![10.0, 20.0, 30.0, 40.0], [1, 4]);
    let added = &matrix + &row_vec;
    println!("matrix + row_vec (broadcast [1,4] to [3,4]):\n{added}");

    // [3, 4] + [3, 1] => [3, 4]
    let col_vec = Tensor2::from_vec(vec![100.0, 200.0, 300.0], [3, 1]);
    let added = &matrix + &col_vec;
    println!("matrix + col_vec (broadcast [3,1] to [3,4]):\n{added}");

    // --- Rule 3: Prepend 1s to shorter shape ---

    // [3, 4] + [4] => [3, 4] + [1, 4] => [3, 4]
    let vec_1d = Tensor1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let added = &matrix + &vec_1d;
    println!("matrix + 1d_vec (broadcast [4] to [3,4]):\n{added}");

    // --- Outer product via broadcasting ---

    // [3, 1] * [1, 4] => [3, 4]
    let a = Tensor2::from_vec(vec![1.0, 2.0, 3.0], [3, 1]);
    let b = Tensor2::from_vec(vec![10.0, 20.0, 30.0, 40.0], [1, 4]);
    let outer = &a * &b;
    println!("Outer product via broadcasting:\n{outer}");

    // --- In-place broadcasting ---

    let mut target = Tensor2::<f64>::zeros([3, 4]);
    let bias = Tensor2::from_vec(vec![1.0, 2.0, 3.0], [3, 1]);
    target += &bias; // broadcast [3,1] into [3,4] in-place
    println!("In-place broadcast add:\n{target}");

    Ok(())
}
```

---

## 7. Doctest 设计

### 7.1 Doctest 规范

| 规范 | 说明 |
|------|------|
| 可编译运行 | 所有 doctest 必须通过 `cargo test --doc` |
| 使用 `?` | 使用 `?` 而非 `unwrap()`（遵循 C-QUESTION-MARK 约定） |
| 隐藏样板 | 用 `# ` 隐藏 use 语句和 main 函数包装 |
| 最小化 | 只展示当前 API 的用法，不引入无关概念 |
| 有断言 | 用 `assert_eq!` 或 `assert!` 验证结果 |

### 7.2 Doctest 模板

```rust
/// Compute the sum of all elements.
///
/// # Examples
///
/// ```
/// use Senon::prelude::*;
///
/// let t = Tensor1::from_vec(vec![1.0, 2.0, 3.0]);
/// assert_eq!(t.sum(), 6.0);
/// ```
///
/// Sum along an axis:
///
/// ```
/// # use Senon::prelude::*;
/// let m = Tensor2::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
/// let col_sums = m.sum_axis(0)?;
/// assert_eq!(col_sums[[0]], 3.0); // 1 + 2
/// assert_eq!(col_sums[[1]], 7.0); // 3 + 4
/// # Ok::<(), Senon::SenonError>(())
/// ```
pub fn sum(&self) -> A { ... }
```

### 7.3 Feature-gated Doctest

```rust
/// Parallel sum using rayon.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "parallel")]
/// # {
/// use Senon::prelude::*;
///
/// let t = Tensor1::<f64>::ones([1_000_000]);
/// let s = t.par_sum();
/// assert_eq!(s, 1_000_000.0);
/// # }
/// ```
#[cfg(feature = "parallel")]
pub fn par_sum(&self) -> A { ... }
```

### 7.4 no_std Doctest

```rust
/// Works in no_std environments.
///
/// ```no_std
/// # extern crate alloc;
/// use Senon::prelude::*;
///
/// let t = Tensor1::from_vec(alloc::vec![1.0, 2.0, 3.0]);
/// assert_eq!(t.len(), 3);
/// ```
```

---

## 8. 文档风格指南

### 8.1 语言规范

| 规范 | 说明 | 示例 |
|------|------|------|
| 英文文档 | 所有 doc comment 使用英文 | `/// Compute the sum.` |
| 第三人称 | 方法描述用第三人称动词开头 | `/// Returns ...` / `/// Computes ...` |
| 简洁首行 | 第一行为完整句子，不超过 80 字符 | `/// Reshape the tensor to a new shape.` |
| 空行分隔 | 首行与详细描述之间空一行 | 见模板 |
| 链接类型 | 引用其他类型使用 `[`TypeName`]` 语法 | `/// See [`TensorView`].` |

### 8.2 术语一致性

| 术语 | 使用场景 | 不使用 |
|------|---------|--------|
| tensor | 泛指 N 维数组 | array, ndarray, matrix (除非特指 2D) |
| element | 数组中的单个值 | item, value, entry |
| axis | 维度方向 | dimension (避免与 Dimension trait 混淆) |
| shape | 各轴长度 | size (除非指总元素数) |
| view | 借用视图 | reference, borrow |
| owned | 拥有数据的张量 | allocated, heap |
| contiguous | 内存连续 | dense, packed |
| stride | 步长 | step |

### 8.3 代码风格

| 规范 | 说明 |
|------|------|
| 类型标注 | 示例中显式标注类型，帮助读者理解 |
| 变量命名 | 使用有意义的名称（`matrix` 而非 `m`） |
| 注释 | 关键步骤添加行内注释 |
| 错误处理 | 使用 `?` 而非 `unwrap()` |

---

## 9. CI 文档验证

### 9.1 验证项目

| 检查项 | 命令 | 失败条件 |
|--------|------|----------|
| 缺失文档 | `RUSTDOCFLAGS="-D warnings" cargo doc --all-features` | 任何 pub 项缺少 doc comment |
| Doctest | `cargo test --doc --all-features` | 任何 doctest 编译或运行失败 |
| 示例编译 | `cargo build --examples --all-features` | 任何示例编译失败 |
| 链接检查 | `cargo doc --all-features` + 检查死链 | 存在无效的文档内链接 |
| README 同步 | 自定义脚本检查 README 与 lib.rs 一致性 | 版本号或 feature 列表不一致 |

### 9.2 CI 配置

```yaml
# .github/workflows/docs.yml (conceptual)
docs:
  steps:
    - name: Check missing docs
      run: RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps

    - name: Run doctests
      run: cargo test --doc --all-features

    - name: Build examples
      run: cargo build --examples --all-features

    - name: Check doc links
      run: |
        cargo doc --all-features --no-deps 2>&1 | grep -i "unresolved" && exit 1 || true
```

### 9.3 docsrs 配置

```toml
# Cargo.toml
[package.metadata.docs.rs]
all-features = true
rustdoc-args = ["--cfg", "docsrs"]
```

```rust
// lib.rs
#![cfg_attr(docsrs, feature(doc_cfg))]
```

---

## 10. Prelude 设计

### 10.1 prelude 内容

```rust
// src/prelude.rs

//! Convenience re-exports for common usage.
//!
//! ```
//! use Senon::prelude::*;
//! ```

// Core types
pub use crate::tensor::{Tensor, TensorView, TensorViewMut, ArcTensor, TensorBase};

// Type aliases
pub use crate::tensor::{Tensor0, Tensor1, Tensor2, Tensor3, TensorD};

// Dimension types
pub use crate::dimension::{Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn, Dimension};

// Element traits
pub use crate::element::{Element, Numeric, RealScalar, ComplexScalar};

// Complex type
pub use crate::complex::Complex;

// Error types
pub use crate::error::{SenonError, Result};

// Slice macro
pub use crate::s;

// Construction helpers
pub use crate::construct::{arange, linspace, logspace};

// Free functions
pub use crate::shape_ops::{cat, stack};
```

### 10.2 Prelude 设计原则

| 原则 | 说明 |
|------|------|
| 最小化 | 只包含日常使用必需的类型 |
| 无冲突 | 避免与 std prelude 或常见 crate 冲突 |
| 可发现 | 用户通过 prelude 能发现核心 API |
| 不含实现细节 | 不导出内部 trait 或辅助类型 |

---

## 11. 格式化输出设计

### 11.1 Display 格式

```rust
// 0D (scalar)
// 42.0

// 1D
// [1.0, 2.0, 3.0]

// 2D (matrix)
// [[1.0, 2.0, 3.0],
//  [4.0, 5.0, 6.0]]

// 3D
// [[[1.0, 2.0],
//   [3.0, 4.0]],
//
//  [[5.0, 6.0],
//   [7.0, 8.0]]]
```

### 11.2 Debug 格式

```rust
// TensorBase { shape: [3, 4], strides: [1, 3], layout: F_CONTIGUOUS | ALIGNED, data: [...] }
```

### 11.3 大数组截断

| 条件 | 行为 |
|------|------|
| 轴长度 ≤ 6 | 完整显示 |
| 轴长度 > 6 | 显示前 3 个和后 3 个，中间用 `...` |
| 总元素 > 1000 | 强制截断 |

```rust
// Large 1D array
// [1.0, 2.0, 3.0, ..., 998.0, 999.0, 1000.0]

// Large 2D matrix
// [[1.0, 2.0, 3.0, ..., 98.0, 99.0, 100.0],
//  [101.0, 102.0, 103.0, ..., 198.0, 199.0, 200.0],
//  [201.0, 202.0, 203.0, ..., 298.0, 299.0, 300.0],
//  ...,
//  [801.0, 802.0, 803.0, ..., 898.0, 899.0, 900.0],
//  [901.0, 902.0, 903.0, ..., 998.0, 999.0, 1000.0]]
```

### 11.4 Display trait 实现骨架

```rust
use core::fmt;

/// Threshold for axis truncation.
const AXIS_TRUNCATE_THRESHOLD: usize = 6;
/// Number of elements to show at each end when truncating.
const AXIS_TRUNCATE_EDGE: usize = 3;
/// Maximum total elements before forced truncation.
const TOTAL_ELEMENT_LIMIT: usize = 1000;

impl<T, D> fmt::Display for TensorBase<T, D>
where
    T: Element + fmt::Display,
    D: Dimension,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let shape = self.shape();
        let ndim = shape.len();

        match ndim {
            0 => write!(f, "{}", self.scalar_value()),
            1 => format_1d(f, self, shape[0]),
            _ => format_nd(f, self, shape, 0, &mut vec![0; ndim]),
        }
    }
}

/// Format a 1D slice with optional truncation.
fn format_1d<T: fmt::Display>(
    f: &mut fmt::Formatter<'_>,
    data: &[T],
    len: usize,
) -> fmt::Result {
    write!(f, "[")?;
    if len <= AXIS_TRUNCATE_THRESHOLD {
        for (i, val) in data.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{val}")?;
        }
    } else {
        // Show first EDGE elements
        for i in 0..AXIS_TRUNCATE_EDGE {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{}", data[i])?;
        }
        write!(f, ", ...")?;
        // Show last EDGE elements
        for i in (len - AXIS_TRUNCATE_EDGE)..len {
            write!(f, ", {}", data[i])?;
        }
    }
    write!(f, "]")
}

/// Recursively format N-dimensional tensor.
fn format_nd<T, D>(
    f: &mut fmt::Formatter<'_>,
    tensor: &TensorBase<T, D>,
    shape: &[usize],
    depth: usize,
    indices: &mut Vec<usize>,
) -> fmt::Result
where
    T: Element + fmt::Display,
    D: Dimension,
{
    let axis_len = shape[depth];
    let is_last_axis = depth == shape.len() - 1;
    let indent = " ".repeat(depth + 1);

    write!(f, "[")?;
    let truncate = axis_len > AXIS_TRUNCATE_THRESHOLD;
    let ranges: Vec<core::ops::Range<usize>> = if truncate {
        vec![0..AXIS_TRUNCATE_EDGE, (axis_len - AXIS_TRUNCATE_EDGE)..axis_len]
    } else {
        vec![0..axis_len]
    };

    let mut first = true;
    for (ri, range) in ranges.iter().enumerate() {
        if ri == 1 {
            // Insert ellipsis between ranges
            write!(f, ",\n{indent}...")?;
        }
        for i in range.clone() {
            if !first {
                write!(f, ",")?;
                if is_last_axis {
                    write!(f, " ")?;
                } else {
                    write!(f, "\n{indent}")?;
                }
            }
            indices[depth] = i;
            if is_last_axis {
                write!(f, "{}", tensor.get(indices))?;
            } else {
                format_nd(f, tensor, shape, depth + 1, indices)?;
            }
            first = false;
        }
    }
    write!(f, "]")
}
```

### 11.5 Debug trait 实现骨架

```rust
impl<T, D> fmt::Debug for TensorBase<T, D>
where
    T: Element + fmt::Debug,
    D: Dimension,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorBase")
            .field("shape", &self.shape())
            .field("strides", &self.strides())
            .field("layout", &self.layout_flags())
            .field("data", &DebugDataAbbrev(self))
            .finish()
    }
}

/// Helper to abbreviate data in Debug output.
struct DebugDataAbbrev<'a, T, D>(&'a TensorBase<T, D>);

impl<T, D> fmt::Debug for DebugDataAbbrev<'_, T, D>
where
    T: Element + fmt::Debug,
    D: Dimension,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let total = self.0.len();
        if total <= AXIS_TRUNCATE_THRESHOLD {
            // Show all elements
            write!(f, "{:?}", self.0.as_slice())
        } else {
            // Show abbreviated form
            write!(f, "[{total} elements]")
        }
    }
}
```

---

## 12. CHANGELOG 设计

### 12.1 格式

遵循 [Keep a Changelog](https://keepachangelog.com/) 格式：

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Initial implementation of N-dimensional tensor type
- Dimension system (Ix0-Ix6, IxDyn)
- Element type hierarchy (Element, Numeric, RealScalar, ComplexScalar)
- Custom complex number type with repr(C)
- Storage system (Owned, View, ViewMut, Arc)
- Memory layout with F-order default and 64-byte alignment

### Changed

### Deprecated

### Removed

### Fixed

### Security
```

### 12.2 版本号规则

| 变更类型 | 版本号影响 | 示例 |
|----------|-----------|------|
| 新增 API | minor | 0.1.0 → 0.2.0 |
| 破坏性变更 | major (1.0 后) | 1.0.0 → 2.0.0 |
| Bug 修复 | patch | 0.1.0 → 0.1.1 |
| 性能优化（无 API 变更） | patch | 0.1.0 → 0.1.1 |
| 0.x 阶段 | minor 可含破坏性变更 | 0.1.0 → 0.2.0 |

---

## 13. 与其他模块的交互

### 13.1 依赖关系总览

```
08-documentation
├── 依赖所有模块设计文档（01-07）
│   └── 每个模块的文档内容基于其设计文档
├── 依赖 01 编码规范
│   └── 文档风格遵循编码规范中的文档节
├── 依赖 03.07 错误处理
│   └── Errors 文档节引用错误类型
├── 被 07 集成测试依赖
│   └── doctest 也是测试的一部分
└── 被 06 benchmark 依赖
    └── benchmark 文档引用性能相关 API 文档
```

### 13.2 详细交互矩阵

| 模块 | 方向 | 交互内容 | 具体影响 |
|------|------|----------|----------|
| 01 编码规范 | 08 依赖 01 | 文档风格规范、注释语言要求 | §8 文档风格指南的规则来源；doc comment 必须英文的依据 |
| 02 项目架构 | 08 依赖 02 | 模块划分、crate 结构 | §3 lib.rs 文档的模块列表和架构描述；§4 模块文档的组织结构 |
| 03 核心类型 | 08 依赖 03 | 类型定义、trait 签名、错误类型 | §5 类型/函数文档的内容来源；Errors 节引用 03.07 错误类型；§10 prelude 内容取决于 03.06 |
| 04 优化模块 | 08 依赖 04 | SIMD/并行 API 签名 | §5 unsafe 函数文档模板（Safety 节）；examples/ 中 simd/parallel 示例依赖 04 的 API |
| 05 运算模块 | 08 依赖 05 | 运算 API、广播规则、索引语义 | §6 示例代码的核心内容来源；doctest 中的运算示例 |
| 06 基准测试 | 06 依赖 08 | 性能相关 API 文档 | benchmark 文档引用 API doc 中的性能说明和复杂度标注 |
| 07 集成测试 | 07 依赖 08 | doctest 作为测试的一部分 | §7 doctest 纳入 CI 测试流水线；doctest 覆盖率计入测试指标 |
| 01 编码规范 | 08 约束 01 | CI 文档验证配置 | §9 CI 配置需与 01 中 CI 清单保持一致 |

### 13.3 关键接口约定

| 约定 | 说明 |
|------|------|
| 错误类型引用 | doc comment 中 `# Errors` 节必须引用 03.07 定义的具体错误枚举变体，使用 intra-doc link |
| Safety 契约同步 | `# Safety` 节的内容必须与 04 中 unsafe 函数的实际前置条件完全一致 |
| 示例代码可编译 | §6 examples/ 和 §7 doctest 中的代码必须基于 03-05 的实际 API 签名，不得使用虚构 API |
| prelude 同步 | §10 prelude 导出列表变更时，§3 lib.rs 文档中的 prelude 说明需同步更新 |
| CI 配置同步 | §9 CI 文档验证步骤需与 01 附录 D CI 检查清单中的文档相关项保持一致 |

---

## 14. 实现任务分解

| 任务 | 描述 | 预计时间 | 依赖 |
|------|------|----------|------|
| T1 | 编写 lib.rs 顶层 crate 文档 | 10 min | 02 完成 |
| T2 | 编写 README.md | 10 min | T1 |
| T3 | 编写 prelude.rs 及其文档 | 10 min | 03.06 完成 |
| T4 | 编写各模块 mod.rs 模块级文档（dimension, element, complex） | 10 min | 03.01-03.03 完成 |
| T5 | 编写各模块 mod.rs 模块级文档（storage, layout, tensor） | 10 min | 03.04-03.06 完成 |
| T6 | 编写各模块 mod.rs 模块级文档（ops, shape_ops, index, construct） | 10 min | 05.01-05.08 完成 |
| T7 | 编写各模块 mod.rs 模块级文档（ffi, workspace, simd, parallel, error） | 10 min | 其余模块完成 |
| T8 | 实现 Display/Debug 格式化（含截断逻辑） | 10 min | 03.06 完成 |
| T9 | 编写 examples/quickstart.rs | 5 min | 核心模块完成 |
| T10 | 编写 examples/matrix_operations.rs | 10 min | 05.03 完成 |
| T11 | 编写 examples/broadcasting.rs | 10 min | 05.05 完成 |
| T12 | 编写 examples/slicing_indexing.rs | 10 min | 05.07 完成 |
| T13 | 编写 examples/shape_manipulation.rs | 10 min | 05.06 完成 |
| T14 | 编写 examples/complex_numbers.rs | 10 min | 03.03 完成 |
| T15 | 编写 examples/parallel_compute.rs | 10 min | 04.02 完成 |
| T16 | 编写 examples/simd_acceleration.rs | 10 min | 04.01 完成 |
| T17 | 编写 examples/ffi_interop.rs | 10 min | 05.09 完成 |
| T18 | 编写 examples/no_std_usage.rs | 10 min | 核心模块完成 |
| T19 | 创建 CHANGELOG.md | 5 min | 无 |
| T20 | 配置 CI 文档验证工作流 | 10 min | T1-T18 |
| T21 | 配置 docsrs 元数据 | 5 min | T1 |

### 14.1 并行执行分组

```
Wave 1 (无依赖):
  T1, T19

Wave 2 (依赖 T1):
  T2, T3, T21

Wave 3 (依赖各模块完成，可并行):
  T4, T5, T6, T7, T8

Wave 4 (依赖 Wave 3，可并行):
  T9, T10, T11, T12, T13, T14, T15, T16, T17, T18

Wave 5 (依赖 Wave 4):
  T20
```

---

## 15. 设计决策记录

### 15.1 决策：英文文档

| 属性 | 值 |
|------|-----|
| 决策 | 所有 doc comment 和 README 使用英文 |
| 理由 | Rust 生态惯例；docs.rs 面向全球开发者；代码注释规范要求英文 |
| 替代方案 | 中文文档 — 放弃，不符合 Rust 社区惯例 |

### 15.2 决策：prelude 最小化

| 属性 | 值 |
|------|-----|
| 决策 | prelude 只包含日常使用必需的类型，不含内部 trait |
| 理由 | 避免命名空间污染；用户可按需 use 具体模块 |
| 替代方案 | 全量导出 — 放弃，容易与用户代码冲突 |

### 15.3 决策：Display 截断阈值

| 属性 | 值 |
|------|-----|
| 决策 | 轴长度 > 6 时截断，显示前 3 + 后 3 |
| 理由 | 与 NumPy 默认行为一致；6 个元素足以展示数据模式 |
| 替代方案 | 可配置阈值 — 未来可添加，初始版本使用固定值 |

### 15.4 决策：doctest 使用 ? 而非 unwrap

| 属性 | 值 |
|------|-----|
| 决策 | 遵循 Rust API Guidelines C-QUESTION-MARK |
| 理由 | 示例代码应展示惯用错误处理；unwrap 给用户错误示范 |
| 替代方案 | unwrap — 放弃，违反 Rust API Guidelines |

---

## 16. 文档质量度量

### 16.1 覆盖率指标

| 指标 | 目标 | 度量方式 | 阈值 |
|------|------|---------|------|
| pub 项文档覆盖率 | 100% | `#![deny(missing_docs)]` 编译通过 | 100%（硬性） |
| unsafe 函数 Safety 节覆盖率 | 100% | CI 脚本扫描 `/// # Safety` | 100%（硬性） |
| 核心类型示例覆盖率 | ≥ 90% | `cargo test --doc` 通过 + 人工审查 | 90% |
| 模块级文档覆盖率 | 100% | 每个 `mod.rs` 有 `//!` 文档 | 100%（硬性） |
| 交叉引用有效率 | 100% | `cargo doc 2>&1 \| grep warning` 为空 | 0 warnings |
| examples/ 可编译率 | 100% | `cargo build --examples` | 100%（硬性） |

### 16.2 Lint 规则配置

```rust
// lib.rs — 文档质量 lint
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(rustdoc::private_intra_doc_links)]
#![warn(rustdoc::missing_crate_level_docs)]
#![warn(rustdoc::invalid_html_tags)]
#![warn(rustdoc::bare_urls)]
```

**Clippy 文档相关 lint**：

```toml
# clippy.toml（文档相关）
# 以下 lint 在 CI 中启用
```

```rust
// CI clippy 参数
#![warn(clippy::missing_errors_doc)]      // 返回 Result 的函数需 Errors 节
#![warn(clippy::missing_panics_doc)]      // 可能 panic 的函数需 Panics 节
#![warn(clippy::missing_safety_doc)]      // unsafe 函数需 Safety 节
```

### 16.3 CI 度量脚本

```bash
#!/bin/bash
# scripts/doc-metrics.sh — 文档质量度量报告

set -euo pipefail

echo "=== Senon Documentation Metrics ==="

# 1. missing_docs check
echo -n "pub item coverage: "
if cargo doc 2>&1 | grep -q "missing documentation"; then
    echo "FAIL"
    cargo doc 2>&1 | grep "missing documentation" | wc -l
    echo " items missing docs"
else
    echo "PASS (100%)"
fi

# 2. Broken links
echo -n "Intra-doc links: "
WARNINGS=$(cargo doc 2>&1 | grep -c "warning:" || true)
echo "${WARNINGS} warnings"

# 3. Doctest pass rate
echo -n "Doctest: "
if cargo test --doc 2>&1 | tail -1 | grep -q "test result: ok"; then
    echo "PASS"
else
    echo "FAIL"
fi

# 4. Examples compilation
echo -n "Examples: "
if cargo build --examples 2>/dev/null; then
    echo "PASS"
else
    echo "FAIL"
fi

# 5. Safety section coverage
echo -n "Safety docs: "
UNSAFE_FNS=$(grep -rn "pub unsafe fn" src/ | wc -l || true)
SAFETY_DOCS=$(grep -rn "/// # Safety" src/ | wc -l || true)
echo "${SAFETY_DOCS}/${UNSAFE_FNS} unsafe fns documented"

echo "=== End Report ==="
```

### 16.4 质量门禁

以下条件全部满足方可合并 PR：

| 门禁 | 条件 | 阻断级别 |
|------|------|---------|
| 编译文档 | `cargo doc --no-deps` 零 warning | 🔴 阻断 |
| Doctest | `cargo test --doc` 全部通过 | 🔴 阻断 |
| 示例编译 | `cargo build --examples` 成功 | 🔴 阻断 |
| missing_docs | `#![deny(missing_docs)]` 通过 | 🔴 阻断 |
| Safety 覆盖 | 所有 `pub unsafe fn` 有 `# Safety` | 🔴 阻断 |
| Clippy 文档 lint | `missing_errors_doc` / `missing_panics_doc` 通过 | 🟡 警告 |
| 示例代码风格 | 使用 `?` 而非 `unwrap` | 🟡 警告 |

---

## 附录 A：文档检查清单速查

| 检查项 | 适用范围 | 验证方式 |
|--------|---------|----------|
| Crate 级文档存在 | `lib.rs` | `#![warn(rustdoc::missing_crate_level_docs)]` |
| 所有 pub 项有 doc comment | 全局 | `#![deny(missing_docs)]` |
| unsafe 函数有 Safety 节 | unsafe fn | Code review |
| 关键 API 有示例 | 核心类型/方法 | `cargo test --doc` |
| 示例使用 `?` 而非 `unwrap` | 所有 doctest | Code review |
| Feature-gated API 有 `doc_cfg` | 可选 feature | `cargo doc --all-features` |
| 类型参数有说明 | 泛型 struct/fn | Code review |
| 错误条件有文档 | 返回 Result 的函数 | Code review |
| 交叉引用有效 | 所有 `[`Type`]` 链接 | `cargo doc` 无警告 |
| 模块文档有依赖图 | 每个 mod.rs | Code review |

## 附录 B：文档模板速查

### B.1 模块文档模板

```rust
//! # [模块名称]
//!
//! [1-2 句话简介]
//!
//! ## 概述
//! [详细描述]
//!
//! ## 核心概念
//! [类型关系、设计决策]
//!
//! ## 使用示例
//! ```rust
//! // 示例代码
//! ```
//!
//! ## 与其他模块的关系
//! [依赖图]
```

### B.2 函数文档模板

```rust
/// [简短描述，第三人称动词开头]
///
/// [详细说明]
///
/// # Arguments
///
/// * `param` - [描述]
///
/// # Returns
///
/// [返回值描述]
///
/// # Errors
///
/// * [`ErrorType`] - [触发条件]
///
/// # Examples
///
/// ```
/// # use Senon::prelude::*;
/// // 示例代码
/// # Ok::<(), Senon::SenonError>(())
/// ```
///
/// # See Also
///
/// * [`related_fn`]
```

### B.3 unsafe 函数文档模板

```rust
/// [简短描述]
///
/// # Safety
///
/// 调用方必须保证：
///
/// 1. [前提条件 1]
/// 2. [前提条件 2]
/// ...
///
/// # Examples
///
/// ```
/// // SAFETY: [解释为何满足前提条件]
/// unsafe { ... }
/// ```
```

## 附录 C：术语表

| 英文术语 | 中文对应 | 在文档中的用法 |
|----------|---------|---------------|
| tensor | 张量 | 泛指 N 维数组 |
| element | 元素 | 数组中的单个值 |
| axis | 轴 | 维度方向（避免与 Dimension trait 混淆） |
| shape | 形状 | 各轴长度的元组/数组 |
| stride | 步长 | 沿轴移动一个位置的元素偏移量 |
| view | 视图 | 借用的张量切片 |
| owned | 拥有的 | 拥有底层数据的张量 |
| contiguous | 连续的 | 内存中元素无间隔排列 |
| broadcast | 广播 | 自动扩展形状以匹配运算 |
| F-order | 列优先 | 列连续存储（Fortran 风格） |
| C-order | 行优先 | 行连续存储（C 风格） |
| scalar | 标量 | 0 维张量 |
| rank | 秩/维度数 | 张量的维度数量（ndim） |
| slice | 切片 | 沿轴选取子区间 |
| reduction | 归约 | 沿轴或全局聚合运算 |

---

## 版本历史

| 版本 | 日期 | 作者 | 变更内容 |
|------|------|------|---------|
| v1.0 | 2026-03-28 | Senon Team | 初始版本：完整文档体系设计 |

---

*Senon v18 — 08 文档与示例设计*
