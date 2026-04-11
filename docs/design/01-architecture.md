# 项目总体架构

> 文档编号: 01 | 模块: 全局 | 阶段: Phase 0
> 前置文档: `00-coding.md`
> 需求参考: 需求说明书 §1, §2, §7, §9, §10, §23, §26, §27, §28

> **格式豁免声明**：本文档为全局架构规范，按 `design.md` §3 豁免标准章节结构。

---

## 1. 项目概览

### 1.1 定位

Xenon 是一个纯 Rust 实现的 N 维数组（张量）库，定位为科学计算的数值基础设施。设计理念与 NumPy ndarray 层相似，但针对 Rust 生态系统进行了深度优化：类型安全、内存高效、零成本抽象、F-order 单一布局。

### 1.2 目标用户

| 用户类型 | 核心诉求 |
|----------|----------|
| 库开发者 | 稳定 API、高性能互操作 |
| 系统开发者 | no_std 支持、底层内存控制、确定性数值行为、最小依赖 |
| 间接用户 | 性能、正确性、与 Python 经验的直觉一致性 |

### 1.3 核心设计原则

| 原则 | 描述 |
|------|------|
| **正确性优先** | 类型安全、内存安全、数值精度满足 IEEE 754 |
| **零成本抽象** | 所有高级操作在编译后内联为基础指针运算 |
| **内存可控** | 支持 `no_std`，显式控制分配和对齐 |
| **渐进增强** | 核心功能无依赖，并行/SIMD 通过独立 backend 模块和 feature gate 按需启用 |
| **错误语义集中裁决** | 方法型 API 统一返回 `Result` 报告可恢复错误；索引语法和运算符语法的 panic 属显式例外，并统一记录于 `26-error.md` |
| **FFI 友好** | 提供 C 兼容的 F-order 内存布局，便于与 BLAS/LAPACK 互操作 |

### 1.4 工程约束

| 约束 | 要求 |
|------|------|
| Crate 结构 | 单 crate，遵循 SemVer |
| MSRV | Rust 1.85+ |
| License | MIT |
| no_std | 支持 `no_std`（依赖 alloc），`std` 为默认 feature |
| 默认内存序 | Fortran-order（列优先），不支持 C-order |
| 内存对齐 | 64 字节（缓存行对齐，AVX-512 友好） |
| 外部依赖 | 仅 rayon（可选并行）+ pulp（可选 SIMD）；benchmark 与扩展验证工具不属于 crate 合约的一部分 |
| no_std 数学函数 | `no_std + alloc` 环境下仍提供 `RealScalar` / `ComplexScalar` 的数学函数能力；Xenon 通过 crate 内部可移植数学实现保持 API 可用性与语义一致性 |

---

## 2. 范围

### 2.1 范围内

- N 维数组的存储、构造
- N 维数组的索引操作（多维整数索引、范围切片）、形状操作（转置、reshape）
- 归约操作（仅 sum）、集合操作（仅 unique）
- 广播操作、逐元素运算
- 显式类型转换
- 向量内积（dot）
- 原始指针 API（FFI）
- 自定义复数类型（Complex\<T\>）
- 临时工作空间

### 2.2 范围外

- 矩阵-矩阵乘法、矩阵分解、对角化等高级线性代数
- 快速傅里叶变换、稀疏矩阵、自动微分、随机数
- BLAS/LAPACK 绑定（由上游库通过指针 API 集成）
- GPU 后端
- serde 序列化
- arena 分配器
- 栈分配小数组

---

## 3. 目录结构

```
xenon/
├── Cargo.toml                 # 包配置和 feature 定义
├── rustfmt.toml               # 代码格式配置
├── README.md
├── LICENSE                    # MIT
├── CHANGELOG.md
│
├── src/
│   ├── lib.rs                 # crate root: feature gates, re-exports, 文档
│   ├── prelude.rs             # 常用类型的 pub use 集合
│   ├── private.rs             # sealed trait 基础设施（防止外部实现）
│   ├── error.rs               # XenonError 枚举、Result 类型别名
│   │
│   ├── dimension/             # 维度类型系统
│   │   ├── mod.rs             # Dimension trait 定义
│   │   ├── static_dims.rs     # Ix0, Ix1, ..., Ix6 静态维度
│   │   ├── dynamic.rs         # IxDyn 动态维度
│   │   ├── into_dimension.rs  # IntoDimension trait
│   │   └── axes.rs            # Axis 标记和轴操作
│   │
│   ├── element/               # 元素类型体系
│   │   ├── mod.rs             # Element trait 定义
│   │   ├── numeric.rs         # Numeric trait（数值运算）
│   │   ├── real.rs            # RealScalar trait（实数）
│   │   ├── complex.rs         # ComplexScalar trait（复数）
│   │   └── primitives.rs      # 基础类型 impl（f32, f64, i32, i64, bool, usize）
│   │
│   ├── complex/               # 自定义复数类型
│   │   ├── mod.rs             # Complex<T> 定义，#[repr(C)]
│   │   ├── ops.rs             # 算术运算实现
│   │   └── cast.rs            # 类型转换
│   │
│   ├── storage/               # 存储系统（独立于 layout，仅管理底层 buffer 与所有权）
│   │   ├── mod.rs             # Storage trait 和 RawStorage trait
│   │   ├── owned.rs           # Owned<A> 拥有型存储
│   │   ├── view.rs            # ViewRepr<'a, A> 不可变视图
│   │   ├── view_mut.rs        # ViewMutRepr<'a, A> 可变视图
│   │   ├── arc.rs             # ArcRepr<A> 原子引用计数存储
│   │   ├── alloc.rs           # 64 字节对齐分配器
│   │   └── traits.rs          # IsOwned, IsView 等 marker traits
│   │
│   ├── layout/                # 内存布局（仅 F-order）
│   │   ├── mod.rs             # LayoutFlags、Strides<D> 和公开辅助函数
│   │   ├── flags.rs           # 布局标志位（F_CONTIGUOUS, ALIGNED 等）
│   │   ├── strides.rs         # F-order 步长计算和验证
│   │   └── contiguous.rs      # 连续性检查
│   │
│   ├── tensor/                # TensorBase 核心
│   │   ├── mod.rs             # TensorBase<S, D> 结构体
│   │   ├── impls.rs           # 核心方法（shape, strides, data_ptr）
│   │   ├── aliases.rs         # 类型别名（Tensor, TensorView, TensorViewMut, ArcTensor）
│   │   └── construct.rs       # 内部构造方法
│   │
│   ├── iter/                  # 迭代器系统
│   │   ├── mod.rs             # 迭代器 trait 定义
│   │   ├── elements.rs        # Elements 迭代器（扁平遍历）
│   │   ├── axis.rs            # AxisIter 沿轴迭代
│   │   ├── windows.rs         # Windows 窗口迭代
│   │   ├── indexed.rs         # IndexedIter 带索引迭代
│   │   ├── zip.rs             # Zip 多张量同步迭代
│   │
│   ├── simd/                  # SIMD 后端（独立性能层，feature = "simd"）
│   │   ├── mod.rs             # pulp 集成、公开 API、SimdKernel trait
│   │   ├── scalar.rs          # 标量回退实现
│   │   └── vector.rs          # 向量化实现
│   │
│   ├── parallel/              # 并行后端（独立性能层，feature = "parallel"）
│   │   ├── mod.rs             # 模块入口、全局阈值配置、模块导出
│   │   ├── par_iter.rs        # 并行迭代器（ParElements, ParZip）
│   │   └── par_ops.rs         # 并行运算（par_map, par_reduce, par_zip_with）
│   │
│   ├── math/                  # 逐元素数学运算
│   │   ├── mod.rs             # 模块入口，re-exports
│   │   ├── map.rs             # 逐元素映射（map, mapv, mapv_inplace）
│   │   ├── zip.rs             # 二元逐元素（zip_with，含广播）
│   │   ├── unary.rs           # 一元运算（abs, neg, signum, square, sin, sqrt, exp, ln, floor, ceil, norm, conj, not）
│   │   ├── binary.rs          # 二元算术方法（add, sub, mul, div, add_scalar, sub_scalar, mul_scalar, div_scalar）
│   │   └── comparison.rs      # 比较运算（eq, ne, lt, gt）
│   │
│   ├── overload/              # 运算符重载
│   │   ├── mod.rs             # 运算 trait 导出
│   │   └── arithmetic.rs      # 运算符重载（Add, Sub, Mul, Div）
│   │
│   ├── util/                  # 实用操作
│   │   ├── mod.rs             # 模块根，re-exports
│   │   ├── clip.rs            # clip / clip_inplace（范围裁剪）
│   │   ├── fill.rs            # fill（原地填充）
│   │   └── contiguous.rs      # to_contiguous（连续性保证）
│   │
│   ├── set/                   # 集合操作
│   │   ├── mod.rs             # 集合操作 trait 导出
│   │   └── unique.rs          # unique（去重）
│   │
│   ├── matrix/                # 矩阵运算
│   │   ├── mod.rs             # 模块入口，re-exports，dot() 公共 API
│   │   └── dot.rs             # 标量向量内积实现，必要时委托 `simd/`
│   │
│   ├── broadcast.rs           # 广播规则实现
│   │
│   ├── reduction/             # 归约操作
│   │   ├── mod.rs             # 模块根，re-exports
│   │   └── sum.rs             # 全局 sum 和沿轴 sum_axis，必要时委托 `simd/` 或 `parallel/`
│   │
│   ├── shape/                 # 形状操作
│   │   ├── mod.rs             # 形状操作 trait
│   │   ├── reshape.rs         # reshape
│   │   └── transpose.rs       # transpose
│   │
│   ├── index/                 # 索引系统
│   │   ├── mod.rs             # 索引 trait 定义
│   │   ├── multi_dim.rs       # 多维整数索引 [i, j, k]
│   │   └── slice_index.rs     # 范围切片索引
│   │
│   ├── construct/             # 张量构造
│   │   ├── mod.rs             # 模块根，re-exports
│   │   ├── fill.rs            # zeros, ones, full（填充构造）
│   │   ├── eye.rs             # eye（单位矩阵）
│   │   ├── from_data.rs       # from_shape_vec, from_shape_slice, from_array（从数据源构造）
│   │   └── from_fn.rs         # from_fn, from_scalar（从闭包/标量构造）
│   │
│   ├── convert/               # 类型转换
│   │   ├── mod.rs             # 模块根，re-exports
│   │   ├── cast.rs            # CastTo trait、cast() 方法、类型转换路径
│   │   ├── owned.rs           # to_owned、into_owned、存储模式互转
│   │   ├── from_impl.rs       # From/TryFrom trait 实现
│   │   └── contiguous.rs      # to_contiguous 连续化转换
│   │
│   ├── format/                # 格式化输出
│   │   ├── mod.rs             # 模块根，re-exports，cfg gates
│   │   ├── config.rs          # FormatConfig 配置结构体
│   │   ├── display.rs         # Display trait 实现
│   │   ├── debug.rs           # Debug trait 实现
│   │   └── pretty.rs          # NumPy 风格格式化辅助函数（fmt_1d, fmt_nd, 截断）
│   │
│   ├── ffi/                   # FFI 接口
│   │   ├── mod.rs             # 模块根，re-exports
│   │   ├── types.rs           # BlasLayout, BlasTrans, BlasInfo 类型定义
│   │   ├── ptr.rs             # 原始指针 API（as_ptr, as_mut_ptr, from_raw_parts, into_raw_parts）
│   │   ├── blas.rs            # BLAS 兼容性检查（is_blas_compatible, blas_info, lda）
│   │   └── offset.rs          # 多维索引到指针偏移（offset_of, ptr_at）
│   │
│   ├── workspace/             # 临时工作空间
│       ├── mod.rs             # 模块根，re-exports
│       ├── error.rs           # WorkspaceError 枚举
│       ├── workspace.rs       # Workspace 结构体、常量、构造、析构
│       ├── borrow.rs          # WorkspaceBorrow、WorkspaceBorrowMut 借用守卫
│       ├── split.rs           # SplitBorrowMut 分割守卫
│       └── expand.rs          # ensure_capacity、reallocate 扩容
│
├── tests/                     # 集成测试
│   ├── common/
│   │   └── mod.rs             # 共享测试工具
│   ├── test_tensor.rs         # 张量基础测试
│   ├── test_math.rs            # 运算测试
│   ├── test_broadcast.rs      # 广播测试
│   └── test_index.rs          # 索引测试
│
├── benches/                   # 性能基准测试
│   ├── math.rs                # 逐元素操作
│   ├── reduction.rs           # 归约操作
│   ├── dot_product.rs         # 向量内积
│   ├── set.rs                 # 集合操作
│   ├── broadcast.rs           # 广播操作
│   ├── shape.rs               # 形状操作
│   ├── simd_comparison.rs     # SIMD 比较
│   ├── parallel_comparison.rs # 并行比较
│   └── construction.rs        # 张量构造
│
└── examples/                  # 使用示例
    ├── basic.rs               # 基础用法
    └── no_std.rs              # 嵌入式示例
```

### 模块职责速览

| 模块 | 职责 |
|------|------|
| `error.rs` | `XenonError` 统一错误枚举，`Result<T>` 类型别名 |
| `dimension/` | `Dimension` trait 和静态/动态维度类型（Ix0-Ix6, IxDyn） |
| `element/` | 元素类型 trait 层次（Element → Numeric → RealScalar/ComplexScalar） |
| `complex/` | 自定义 `Complex<T>` 类型，`#[repr(C)]` 兼容 C FFI |
| `storage/` | 四种存储模式（Owned/ViewRepr/ViewMutRepr/ArcRepr） |
| `layout/` | F-order 布局标志位、步长计算、连续性检查 |
| `tensor/` | 核心 `TensorBase<S, D>` 结构体及类型别名 |
| `iter/` | 元素/轴/窗口/索引/Zip 迭代器 |
| `simd/` | SIMD 后端：向量化 kernel、标量回退、运行时分发 |
| `parallel/` | 并行后端：并行迭代器、并行 map/reduce/zip、阈值与 guard |
| `math/` | 逐元素数学运算（映射、一元、二元算术、比较），按需委托 `simd/` / `parallel/` |
| `overload` | 运算符重载（Add, Sub, Mul, Div trait 实现） |
| `util/` | 实用操作（clip 裁剪、fill 填充、to_contiguous 连续化） |
| `set/` | 集合操作（unique 去重） |
| `broadcast.rs` | NumPy 广播规则 |
| `matrix/` | 向量内积 dot，必要时委托 `simd/` |
| `reduction/` | 归约操作（sum），必要时委托 `simd/` / `parallel/` |
| `shape/` | reshape、transpose |
| `index/` | 多维整数索引、范围切片索引 |
| `construct/` | 张量构造 |
| `convert/` | 类型转换（cast、存储模式互转、From trait、连续化） |
| `format/` | NumPy 风格格式化输出 |
| `ffi/` | 原始指针 API、BLAS 兼容性检查、多维索引偏移（types/ptr/blas/offset） |
| `workspace/` | 临时工作空间（对齐分配、借用守卫、分割、扩容） |

---

## 4. Cargo.toml 设计

```toml
[package]
name = "xenon"
version = "0.1.0"
edition = "2024"
rust-version = "1.85"
license = "MIT"
description = "A Rust N-dimensional array library for scientific computing"
keywords = ["tensor", "array", "numpy", "scientific", "ndarray"]
categories = ["science", "mathematics", "no-std", "data-structures"]

[features]
default = ["std"]

# Standard library support
std = []

# Parallel computing (depends on rayon + std)
parallel = ["dep:rayon", "std"]

# SIMD acceleration (depends on pulp)
simd = ["dep:pulp"]

[dependencies]
rayon = { version = "1.10", optional = true }
pulp = { version = "0.18", optional = true }

# No mandatory dev-dependencies in the base crate contract.
# Optional local tooling for benchmarks or extended verification is maintained
# outside the SemVer-stable crate surface.

[[bench]]
name = "math"
harness = false

[[bench]]
name = "reduction"
harness = false

[[bench]]
name = "dot_product"
harness = false

[[bench]]
name = "set"
harness = false

[[bench]]
name = "broadcast"
harness = false

[[bench]]
name = "shape"
harness = false

[[bench]]
name = "simd_comparison"
harness = false

[[bench]]
name = "parallel_comparison"
harness = false

[[bench]]
name = "construction"
harness = false

[profile.release]
lto = "thin"
codegen-units = 1
opt-level = 3

[profile.bench]
lto = "thin"
codegen-units = 1

[package.metadata.docs.rs]
all-features = true
rustdoc-args = ["--cfg", "docsrs"]
```

### Feature 组合说明

| 组合 | 命令 | 适用场景 |
|------|------|------|
| 最小化 | `--no-default-features` | 嵌入式、WASM、内核模块 |
| 标准 | （默认） | 桌面应用、CLI 工具 |
| 高性能 | `--features parallel,simd` | 数据科学、机器学习 |
| 仅 SIMD | `--features simd` | 需 SIMD 但无需并行的场景 |

---

## 5. 模块依赖关系

### 5.1 依赖层级

各模块的详细设计参见对应编号文档。层级关系如下：

| 层级 | 模块 | 依赖 | 参见 |
|------|------|------|------|
| **L0** | error, private | 无 | `26-error.md` |
| **L1** | dimension, element, complex | error（element 额外依赖 complex） | `02-dimension.md`、`03-element.md`、`04-complex.md` |
| **L2** | layout | error, dimension | `06-memory.md` |
| **L2** | workspace | core, alloc（独立于核心类型系统，可被上游库直接使用） | `24-workspace.md` |
| **L3** | storage | core, alloc（布局信息由 `tensor` 持有并消费 `layout` 结果） | `05-storage.md` |
| **L4** | tensor | storage, dimension, layout, element | `07-tensor.md` |
| **L5** | broadcast, iter, ffi, simd, parallel | tensor（parallel 额外依赖 iter/broadcast；simd 额外依赖 layout/element） | `15-broadcast.md`、`10-iterator.md`、`23-ffi.md`、`08-simd.md`、`09-parallel.md` |
| **L6** | math, overload, set, matrix, reduction, shape, index, util | tensor, broadcast，以及按需调用独立 backend 模块 | `11-math.md`、`12-matrix.md`、`13-reduction.md`、`14-set.md`、`16-shape.md`、`17-indexing.md`、`19-overload.md`、`20-utility.md` |
| **L7** | construct, convert, format | tensor, shape | `18-construction.md`、`21-type.md`、`22-output.md` |

### 5.2 依赖图（ASCII）

```
L0:  error, private
      │
L1:  dimension, element, complex
      │
L2:  layout        workspace
      │
L3:  storage
      │
L4:  tensor
      │
L5:  iter    broadcast    ffi    simd    parallel
      │         │           │      │         │
      └─────────┴───────────┴──────┴─────────┘
                           │
L6:        math   overload   set   matrix   reduction   shape   index   util
                           │
L7:                 construct   convert   format
```

> **L5/L6 模块说明**：`simd` 与 `parallel` 保持为独立 backend 模块，不再内嵌到 `math/`、`matrix/`、`reduction/` 目录中。上层业务模块只定义语义与公共 API，性能后端统一由独立 backend 模块承载与分发。

---

## 6. Feature Gate 矩阵

| 功能 | 默认 (std) | no_std | +parallel | +simd | +parallel+simd |
|------|:----------:|:------:|:---------:|:-----:|:--------------:|
| 基础张量操作 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 视图/视图可变 | ✅ | ✅ | ✅ | ✅ | ✅ |
| Arc 存储 | ✅ | ✅³ | ✅ | ✅ | ✅ |
| 迭代器 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 逐元素运算 | ✅ | ✅ | ✅ | ✅ | ✅ |
| sum 归约 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 内积 (dot) | ✅ | ✅ | ✅ | ✅ | ✅ |
| reshape / transpose | ✅ | ✅ | ✅ | ✅ | ✅ |
| 整数索引 / 切片 | ✅ | ✅ | ✅ | ✅ | ✅ |
| Display 格式化 | ✅ | ✅¹ | ✅ | ✅ | ✅ |
| Vec 分配 | ✅ | ✅² | ✅ | ✅ | ✅ |
| 并行迭代器 | ❌ | ❌ | ✅ | ❌ | ✅ |
| 并行归约 | ❌ | ❌ | ✅ | ❌ | ✅ |
| SIMD 向量化 | ❌ | ❌ | ❌ | ✅ | ✅ |
| BLAS 兼容 API | ✅ | ✅ | ✅ | ✅ | ✅ |

> ¹ Display 格式化仅使用 `core::fmt`，不依赖 `std`。
> ² Vec 分配需要 `alloc` crate（`no_std` 环境下通过 `extern crate alloc` 提供）。
> ³ Arc 存储在 `no_std` 下要求目标平台提供原子能力并启用 `alloc`；若目标不支持原子指令，则该能力不可用。

---

## 7. prelude.rs 导出清单

```rust
// src/prelude.rs

// Core tensor types
pub use crate::tensor::{
    TensorBase,
    Tensor,           // TensorBase<Owned<A>, D>
    TensorView,       // TensorBase<ViewRepr<'a, A>, D>
    TensorViewMut,    // TensorBase<ViewMutRepr<'a, A>, D>
    ArcTensor,        // TensorBase<ArcRepr<A>, D>
};

// Dimension types
pub use crate::dimension::{
    Dimension,
    IntoDimension,
    Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6,
    IxDyn,
    Axis,
};

// Layout
pub use crate::layout::LayoutFlags;

// Element traits
pub use crate::element::{
    Element,
    Numeric,
    RealScalar,
    ComplexScalar,
};
pub use crate::complex::Complex;

// Error types
pub use crate::error::{XenonError, Result};

// Slice macro
pub use crate::index::s;

// Construction helpers
pub use crate::construct::{
    zeros, ones, eye,
    full, from_shape_vec, from_fn,
};
```

---

## 8. lib.rs 模块结构

```rust
// src/lib.rs

//! # Xenon
//!
//! A Rust N-dimensional array library for scientific computing.

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
#![warn(rust_2024_compatibility)]
#![warn(unsafe_op_in_unsafe_fn)]

#[cfg(not(feature = "std"))]
extern crate alloc;

// Internal modules
mod private;

// Public modules
pub mod error;
pub mod dimension;
pub mod element;
pub mod complex;
pub mod storage;
pub mod layout;
pub mod tensor;
pub mod iter;
pub mod math;
pub mod overload;
pub mod matrix;
pub mod util;
pub mod set;
pub mod broadcast;
pub mod reduction;
pub mod shape;
pub mod index;
pub mod construct;
pub mod convert;
pub mod format;
pub mod ffi;
pub mod workspace;

// Conditional modules
#[cfg(feature = "simd")]
#[cfg_attr(docsrs, doc(cfg(feature = "simd")))]
pub mod simd;

#[cfg(feature = "parallel")]
#[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
pub mod parallel;

// Prelude
pub mod prelude;

// Convenience re-exports
pub use prelude::*;
pub use error::XenonError;
```

---

## 9. API 稳定性说明

| 层级 | 稳定性 | 说明 |
|------|--------|------|
| `prelude::*` | **稳定** | 主版本号内保持兼容 |
| 公开 trait 方法 | **稳定** | 只增不减 |
| 内部模块 (`mod private`) | **不稳定** | 随时可能变更 |
| `#[doc(hidden)]` | **不稳定** | 仅供内部使用 |
| `simd`/`parallel` 模块 | **稳定** | 作为公开模块纳入 SemVer；feature gate 仅控制可用性，不降低兼容性承诺 |

---

## 10. 核心类型速查

各类型的详细设计参见对应模块文档（`02-dimension.md`、`03-element.md`、`05-storage.md`、`06-memory.md`）。

```rust
// Tensor core types
TensorBase<S, D>              // Generic base type
Tensor<A, D>                  // = TensorBase<Owned<A>, D>
TensorView<'a, A, D>          // = TensorBase<ViewRepr<'a, A>, D>
TensorViewMut<'a, A, D>       // = TensorBase<ViewMutRepr<'a, A>, D>
ArcTensor<A, D>               // = TensorBase<ArcRepr<A>, D>

// Dimension types
Ix0, Ix1, Ix2, ..., Ix6       // Static dimensions (0-6 dimensions)
IxDyn                         // Dynamic dimension

// Layout flags (F-order only)
LayoutFlags: u8
├── F_CONTIGUOUS    (0b00001) // Fortran contiguous
├── ALIGNED         (0b00100) // 64-byte aligned
├── HAS_ZERO_STRIDE (0b01000) // Contains zero stride (broadcast)
└── HAS_NEG_STRIDE  (0b10000) // Contains negative stride (flip view)

// Element trait hierarchy
Element                        // Base: Copy + PartialEq + Debug + Display + Send + Sync
└── Numeric                    // Numeric: Add + Sub + Mul + Div + Neg (i32/i64/f32/f64/Complex only)
    ├── RealScalar             // Real: sqrt, sin, cos, etc.
    └── ComplexScalar          // Complex: conjugate, modulus, etc.
```

---

## 11. 实现任务分解

### Wave 1: 基础设施（可完全并行）

| 任务 | 依赖 | 预估复杂度 | 产出 |
|------|------|------------|------|
| W1.1 error types | 无 | 低 | `XenonError`, `Result<T>` |
| W1.2 private module | 无 | 低 | `Sealed` trait |
| W1.3 dimension traits | W1.1 | 中 | `Dimension`, `IntoDimension` |
| W1.4 static dimensions | W1.3 | 中 | `Ix0`-`Ix6` |
| W1.5 dynamic dimension | W1.3 | 中 | `IxDyn` |
| W1.6 element traits | W1.1 | 中 | `Element`, `Numeric`, `RealScalar` |
| W1.7 Complex\<T\> | W1.6 | 高 | 自定义复数类型 |
| W1.8 layout flags | W1.1 | 低 | `LayoutFlags` |
| W1.9 F-order strides | W1.1, W1.3 | 中 | F-order 步长计算 |

### Wave 2: 核心（依赖 Wave 1）

| 任务 | 依赖 | 预估复杂度 | 产出 |
|------|------|------------|------|
| W2.1 Storage trait | 无 | 高 | `Storage`, `RawStorage`（仅依赖 core/alloc） |
| W2.2 Owned storage | W2.1 | 中 | `Owned<A>` + 64 字节对齐分配 |
| W2.3 View storage | W2.1 | 中 | `ViewRepr<'a, A>` |
| W2.4 ViewMut storage | W2.1 | 中 | `ViewMutRepr<'a, A>` |
| W2.5 Arc storage | W2.1 | 高 | `ArcRepr<A>` |
| W2.6 TensorBase | W2.1-W2.5, W1.3-W1.5 | 高 | 核心结构体 |
| W2.7 Type aliases | W2.6 | 低 | `Tensor`, `TensorView` 等 |

### Wave 3: 操作（依赖 Wave 2）

| 任务 | 依赖 | 预估复杂度 | 产出 |
|------|------|------------|------|
| W3.1 Elements iterator | W2.6 | 中 | 扁平元素迭代 |
| W3.2 Axis iterator | W2.6 | 中 | 沿轴迭代 |
| W3.3 Window iterator | W2.6 | 高 | 窗口迭代 |
| W3.4 Zip iterator | W2.6, W3.1 | 高 | 多张量同步迭代 |
| W3.5 Math | W3.1 | 中 | map, zip_with |
| W3.6 Arithmetic | W3.5 | 中 | Add, Sub, Mul, Div |
| W3.7 Reduction (sum) | W3.1 | 中 | sum, sum_axis |
| W3.8 Dot (inner product) | W2.6 | 中 | 向量内积 |
| W3.9 Broadcast | W2.6 | 高 | 广播规则 |
| W3.10 Reshape | W2.6 | 中 | reshape |
| W3.11 Transpose | W2.6 | 中 | transpose |
| W3.12 Multi-dim index | W2.6 | 中 | [i, j, k] 索引 |
| W3.13 Slice index | W2.6 | 高 | 范围切片 |

### Wave 4: 集成（依赖 Wave 3）

| 任务 | 依赖 | 预估复杂度 | 产出 |
|------|------|------------|------|
| W4.1 construct | W2.6, W3.10 | 中 | zeros, ones, eye, from_vec |
| W4.2 convert | W2.6 | 中 | cast, to_owned |
| W4.3 format | W2.6 | 低 | Display/Debug |
| W4.4 ffi | W2.6 | 中 | 原始指针 API |
| W4.5 workspace | 无 | 中 | 临时缓冲区 |
| W4.6 comparison | W3.5 | 低 | equal, not_equal, less, greater |

### Wave 5: 性能（依赖 Wave 4）

| 任务 | 依赖 | 预估复杂度 | 产出 |
|------|------|------------|------|
| W5.1 par_iter | W3.1-W3.4 | 高 | 并行迭代器 |
| W5.2 par_reduction | W3.7, W5.1 | 高 | 并行 sum |
| W5.3 simd math | W3.5 | 高 | SIMD 逐元素 |
| W5.4 simd reduction | W3.7 | 高 | SIMD sum |

### 并行执行分组图

```
Wave 1: [W1.1] [W1.2] [W1.3] [W1.6] [W1.8]
           │       │       │       │       │
           └───────┴───────┴───────┴───────┘
                           │
                           ▼
Wave 2: [W2.1] [W2.2] [W2.3] [W2.4] [W2.5]
           │       │       │       │       │
           └───────┴───────┴───────┴───────┘
                           │
                           ▼
        [W2.6] ──▶ [W2.7]
                           │
                           ▼
Wave 3: [W3.1] [W3.2] [W3.3] [W3.9] [W3.10] [W3.11] [W3.12] [W3.13]
           │       │       │       │       │       │       │       │
           └───────┴───────┴───────┴───────┴───────┴───────┴───────┘
                           │
                           ▼
        [W3.4] [W3.5] [W3.7] [W3.8] [W3.6]
                           │
                           ▼
Wave 4: [W4.1] [W4.2] [W4.3] [W4.4] [W4.5] [W4.6]
                           │
                           ▼
Wave 5: [W5.1] [W5.2] [W5.3] [W5.4]
```

---

## 12. 设计决策记录

### 决策 1：单 Crate 设计

| 属性 | 值 |
|------|-----|
| 决策 | 使用单 crate（`xenon`）而非多 crate workspace |
| 理由 | 降低发布复杂度；避免版本协调问题；简化依赖管理 |
| 替代方案 | workspace 多 crate（xenon-core, xenon-math, ...） — 放弃，对当前规模过度工程化 |

### 决策 2：F-order 单一布局

| 属性 | 值 |
|------|-----|
| 决策 | 仅支持列优先（F-order）布局 |
| 理由 | 与 BLAS/LAPACK 兼容；减少布局组合爆炸；简化步长计算（参见 `06-memory.md` §1） |
| 替代方案 | 同时支持 F-order 和 C-order — 放弃，超出范围且增加复杂度 |

### 决策 3：功能最小化原则

| 属性 | 值 |
|------|-----|
| 决策 | 归约仅 sum、集合仅 unique、形状仅 transpose+reshape、索引仅整数+切片、矩阵仅内积 |
| 理由 | 先做精再做广；每个功能确保正确性和性能后再扩展 |
| 替代方案 | 一开始支持所有 ndarray 功能 — 放弃，范围失控风险高 |

### 决策 4：依赖层级严格单向

| 属性 | 值 |
|------|-----|
| 决策 | 模块依赖严格按 L0→L7 层级单向，禁止循环依赖 |
| 理由 | 确保编译时间可预测；依赖关系清晰可维护 |
| 替代方案 | 允许跨层引用 — 放弃，维护成本高 |

### 决策 5：独立 backend 模块

| 属性 | 值 |
|------|-----|
| 决策 | `simd/` 与 `parallel/` 保持为独立顶级 backend 模块，由 `math` / `matrix` / `reduction` 按需调用 |
| 理由 | 性能后端是横切关注点，独立模块更便于统一 feature gate、共享分发逻辑、集中测试与文档维护 |
| 替代方案 | 将 backend 内嵌到 `math/`、`reduction/`、`matrix/` — 放弃，会让性能实现与语义 API 耦合，扩大重复实现 |

### 决策 6：错误语义集中裁决

| 属性 | 值 |
|------|-----|
| 决策 | 方法型 API（如 `reshape`、`broadcast_to`、`sum_axis`、`cast`）统一返回 `Result` 报告可恢复错误；索引语法 `tensor[[...]]` 与四则运算符语法保留 panic，作为语言接口约束下的显式例外 |
| 理由 | 保持可恢复错误的一致出口，同时兼顾 Rust `Index` / `Add` 等 trait 的惯用语义与签名限制 |
| 替代方案 | 所有接口统一 panic — 放弃，不利于库集成和诊断；所有语法接口也返回 `Result` — 放弃，与 Rust 标准 trait 签名冲突 |

---

## 版本历史

| 版本 | 日期 |
|------|------|
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-08 |
| 1.0.2 | 2026-04-08 |
| 1.1.0 | 2026-04-08 |
| 1.2.0 | 2026-04-08 |
| 1.2.1 | 2026-04-10 |

---

*本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。*
