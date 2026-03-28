# Xenon 项目架构概览

> 版本: 0.1.0 | 最后更新: 2026-03-28

---

## 1. 项目概览

### 1.1 定位

Xenon 是一个纯 Rust 实现的 N 维数组（张量）计算库，设计理念与 NumPy 相似但针对 Rust 生态系统进行了深度优化。它提供类型安全、内存高效、零成本抽象的张量操作，同时支持 `no_std` 环境和可选的并行/SIMD 加速。

### 1.2 目标用户

- **科学计算研究人员**：需要高性能数值计算能力
- **机器学习工程师**：构建自定义模型或底层框架
- **嵌入式开发者**：在资源受限环境中进行信号处理
- **Rust 库作者**：需要可靠的张量基础组件

### 1.3 核心设计原则

| 原则 | 描述 |
|------|------|
| **类型安全** | 通过泛型参数 `TensorBase<S, D>` 在编译期捕获维度和存储错误 |
| **零成本抽象** | 所有高级操作在编译后内联为基础指针运算 |
| **内存可控** | 支持 `no_std`，显式控制分配和对齐 |
| **渐进增强** | 核心功能无依赖，并行/SIMD 通过 feature gate 按需启用 |
| **FFI 友好** | 提供 C 兼容的内存布局，便于与 BLAS/LAPACK 互操作 |

### 1.4 工程约束

- **MSRV**: Rust 1.85+
- **License**: MIT
- **版本策略**: 严格遵循 SemVer
- **内存对齐**: 64 字节（缓存行对齐，AVX-512 友好）
- **默认内存序**: Fortran-order（列优先），兼容 BLAS
- **no_std 支持**: 依赖 `alloc` crate，`std` 为默认 feature

---

## 2. 目录结构

```
xenon/
├── Cargo.toml                 # 包配置和 feature 定义
├── README.md                  # 项目说明
├── LICENSE                    # MIT 许可证
├── CHANGELOG.md               # 变更日志
│
├── src/
│   ├── lib.rs                 # crate root：feature gates、re-exports、文档
│   ├── prelude.rs             # 常用类型的 pub use 集合
│   ├── private.rs             # sealed trait 基础设施（防止外部实现）
│   ├── error.rs               # 错误类型：ShapeError, IndexError, StrideError 等
│   │
│   ├── dimension/             # 维度类型系统
│   │   ├── mod.rs             # Dimension trait 定义
│   │   ├── static.rs          # Ix0, Ix1, ..., Ix6 静态维度
│   │   ├── dynamic.rs         # IxDyn 动态维度
│   │   ├── into_dimension.rs  # IntoDimension trait
│   │   └── axes.rs            # Axis 标记和轴操作
│   │
│   ├── element/               # 元素类型体系
│   │   ├── mod.rs             # Element trait 定义
│   │   ├── numeric.rs         # Numeric trait（数值运算）
│   │   ├── real.rs            # RealScalar trait（实数）
│   │   ├── complex.rs         # ComplexScalar trait（复数）
│   │   └── primitives.rs      # 基础类型 impl（f32, f64, i32, etc.）
│   │
│   ├── complex/               # 自定义复数类型
│   │   ├── mod.rs             # Complex<T> 定义
│   │   ├── ops.rs             # 算术运算实现
│   │   └── cast.rs            # 类型转换
│   │
│   ├── storage/               # 存储系统
│   │   ├── mod.rs             # Storage trait 和 RawStorage trait
│   │   ├── owned.rs           # Owned<T> 拥有型存储
│   │   ├── view.rs            # ViewRepr<T> 不可变视图
│   │   ├── view_mut.rs        # ViewMutRepr<T> 可变视图
│   │   ├── arc.rs             # ArcRepr<T> 原子引用计数存储
│   │   ├── alloc.rs           # 64 字节对齐分配器
│   │   └── traits.rs          # IsOwned, IsView 等 marker traits
│   │
│   ├── layout/                # 内存布局
│   │   ├── mod.rs             # Layout 类型定义
│   │   ├── flags.rs           # 布局标志位（F/C contiguous 等）
│   │   ├── strides.rs         # Strides 计算和验证
│   │   └── contiguous.rs      # 连续性检查
│   │
│   ├── tensor/                # TensorBase 核心
│   │   ├── mod.rs             # TensorBase<S, D> 结构体
│   │   ├── impls.rs           # 核心方法（shape, strides, data_ptr）
│   │   ├── aliases.rs         # 类型别名（Tensor, Array, TensorView 等）
│   │   └── construct.rs       # 内部构造方法
│   │
│   ├── iter/                  # 迭代器系统
│   │   ├── mod.rs             # 迭代器 trait 定义
│   │   ├── elements.rs        # Elements 迭代器（扁平遍历）
│   │   ├── axis.rs            # AxisIter 沿轴迭代
│   │   ├── windows.rs         # Windows 窗口迭代
│   │   ├── indexed.rs         # IndexedIter 带索引迭代
│   │   ├── zip.rs             # Zip 多张量同步迭代
│   │   └── lanes.rs           # LaneIter 行/列迭代
│   │
│   ├── ops/                   # 数学运算
│   │   ├── mod.rs             # 运算 trait 导出
│   │   ├── elementwise.rs     # 逐元素运算（map, zip_with, apply）
│   │   ├── arithmetic.rs      # 运算符重载（Add, Sub, Mul, Div, Rem）
│   │   ├── matrix.rs          # 矩阵运算（matvec, dot, outer, batch_matmul）
│   │   ├── reduction.rs       # 归约（sum, prod, mean, var, min, max, argmin, argmax）
│   │   ├── accumulate.rs      # 累积运算（cumsum, cumprod）
│   │   ├── set_ops.rs         # 集合操作（unique, bincount, histogram）
│   │   └── comparison.rs      # 比较运算（is_close, allclose, clip, clamp）
│   │
│   ├── broadcast.rs           # 广播规则实现
│   │
│   ├── shape_ops/             # 形状操作
│   │   ├── mod.rs             # 形状操作 trait
│   │   ├── reshape.rs         # reshape, into_shape
│   │   ├── transpose.rs       # transpose, permute_axes, swap_axes
│   │   ├── slice.rs           # slice, slice_mut, slice_collapse
│   │   ├── squeeze.rs         # squeeze, expand_dims
│   │   ├── pad.rs             # pad（常数、边缘、反射填充）
│   │   ├── repeat.rs          # repeat, tile
│   │   ├── split.rs           # split, hsplit, vsplit, dsplit
│   │   └── stack.rs           # stack, concatenate, vstack, hstack
│   │
│   ├── index/                 # 索引系统
│   │   ├── mod.rs             # 索引 trait 定义
│   │   ├── multi_dim.rs       # 多维索引 [i, j, k]
│   │   ├── slice_index.rs     # 切片索引 s![.., 0..10, ..;2]
│   │   ├── advanced.rs        # 高级索引（布尔掩码、整数数组）
│   │   └── where_.rs          # where 条件选择
│   │
│   ├── construct.rs           # 张量构造（zeros, ones, eye, arange, linspace, from_vec, from_fn）
│   │
│   ├── convert.rs             # 类型转换（cast, From impls, to_owned, to_contiguous）
│   │
│   ├── format.rs              # Display/Debug 格式化（矩阵打印、截断规则）
│   │
│   ├── ffi.rs                 # FFI API（as_ptr, as_mut_ptr, BLAS 兼容布局检查）
│   │
│   ├── workspace.rs           # 临时工作空间（避免重复分配）
│   │
│   ├── parallel/              # 并行后端（仅 parallel feature）
│   │   ├── mod.rs             # rayon 集成
│   │   ├── par_iter.rs        # 并行迭代器
│   │   └── par_ops.rs         # 并行运算
│   │
│   └── simd/                  # SIMD 后端（仅 simd feature）
│       ├── mod.rs             # pulp 集成
│       ├── scalar.rs          # 标量回退实现
│       └── vector.rs          # 向量化实现
│
├── tests/                     # 集成测试
│   ├── test_tensor.rs         # 张量基础测试
│   ├── test_ops.rs            # 运算测试
│   ├── test_broadcast.rs      # 广播测试
│   ├── test_index.rs          # 索引测试
│   └── test_no_std.rs         # no_std 兼容性测试
│
├── benches/                   # 性能基准测试
│   ├── bench_matmul.rs        # 矩阵乘法
│   ├── bench_reduction.rs     # 归约操作
│   └── bench_broadcast.rs     # 广播开销
│
└── examples/                  # 使用示例
    ├── basic.rs               # 基础用法
    ├── matrix_ops.rs          # 矩阵运算
    ├── no_std.rs              # 嵌入式示例
    └── parallel.rs            # 并行计算示例
```

### 模块职责速览

| 模块 | 职责 |
|------|------|
| `dimension/` | 定义 `Dimension` trait 和静态/动态维度类型，处理形状的编译期和运行时表示 |
| `element/` | 定义元素类型的 trait 层次结构，约束哪些类型可以作为张量元素 |
| `complex/` | 提供自定义 `Complex<T>` 类型，保证 `#[repr(C)]` 布局以兼容 C FFI |
| `storage/` | 定义存储后端 trait 和四种存储模式（Owned/View/ViewMut/Arc），管理内存分配 |
| `layout/` | 管理内存布局标志位（连续性、对齐、步长方向），支持 F-order 和 C-order |
| `tensor/` | 核心 `TensorBase<S, D>` 结构体及其类型别名，是整个库的中心类型 |
| `iter/` | 提供各种遍历张量的迭代器，支持元素遍历、轴遍历、窗口遍历、多张量同步遍历 |
| `ops/` | 实现所有数学运算，包括逐元素运算、矩阵运算、归约、累积、比较等 |
| `shape_ops/` | 提供改变张量形状但不改变数据的操作（reshape、transpose、slice 等） |
| `index/` | 实现多维索引、切片索引和高级索引（掩码、整数数组） |
| `broadcast/` | 实现广播规则，自动扩展不同形状的张量以进行逐元素运算 |
| `construct/` | 提供创建张量的便捷函数（zeros、ones、arange、linspace 等） |
| `convert/` | 处理类型转换和存储模式转换 |
| `format/` | 实现张量的可读打印，支持截断和对齐 |
| `ffi/` | 提供与 C/BLAS 互操作的原始指针 API |
| `parallel/` | 基于 rayon 的并行后端（可选） |
| `simd/` | 基于 pulp 的 SIMD 后端（可选） |

---

## 3. Cargo.toml 设计

```toml
[package]
name = "xenon"
version = "0.1.0"
edition = "2024"
rust-version = "1.85"
authors = ["Xenon Contributors"]
license = "MIT"
description = "A Rust N-dimensional array library for scientific computing"
repository = "https://github.com/xenon-rs/xenon"
documentation = "https://docs.rs/xenon"
readme = "README.md"
keywords = ["tensor", "array", "numpy", "scientific", "ndarray"]
categories = ["science", "mathematics", "no-std", "data-structures"]
exclude = ["/.github", "/benches", "/examples", "/tests"]

[features]
default = ["std"]

# 标准库支持（启用更多功能如 Vec、Box、String）
std = []

# 并行计算支持（依赖 rayon）
parallel = ["dep:rayon", "std"]

# SIMD 加速支持（依赖 pulp）
simd = ["dep:pulp"]

# BLAS 集成（未来扩展）
# blas = ["dep:blas-src", "std"]

[dependencies]
# 可选依赖
rayon = { version = "1.10", optional = true }
pulp = { version = "0.18", optional = true }

# no_std 环境所需
[target.'cfg(not(feature = "std"))'.dependencies]
alloc = { version = "0.2", default-features = false, features = ["unstable"] }

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
proptest = "1.4"
approx = "0.5"
quickcheck = "1.0"
rand = "0.8"

[[bench]]
name = "matmul"
harness = false

[[bench]]
name = "reduction"
harness = false

[[bench]]
name = "broadcast"
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
|------|------|----------|
| 最小化 | `--no-default-features` | 嵌入式、WASM、内核模块 |
| 标准 | （默认） | 桌面应用、CLI 工具 |
| 高性能 | `--features parallel,simd` | 数据科学、机器学习训练 |
| 兼容性 | `--features std,simd` | 需要 SIMD 但无需并行的场景 |

---

## 4. 模块依赖关系

### 4.1 依赖图（ASCII）

```
                          ┌─────────────┐
                          │   lib.rs    │
                          │  prelude    │
                          └──────┬──────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
              ▼                  ▼                  ▼
        ┌─────────┐        ┌─────────┐        ┌─────────┐
        │  error  │        │ private │        │ format  │
        └────┬────┘        └────┬────┘        └────┬────┘
             │                  │                  │
    ┌────────┴────────┐         │         ┌────────┴────────┐
    │                 │         │         │                 │
    ▼                 ▼         │         ▼                 ▼
┌─────────┐     ┌─────────┐     │    ┌─────────┐     ┌─────────┐
│dimension│     │ element │     │    │construct│     │ convert │
└────┬────┘     └────┬────┘     │    └────┬────┘     └────┬────┘
     │               │          │         │               │
     │          ┌────┴────┐     │         │               │
     │          │ complex │     │         │               │
     │          └────┬────┘     │         │               │
     │               │          │         │               │
     ▼               ▼          │         │               │
┌─────────┐     ┌─────────┐     │         │               │
│  layout │◄────│ storage │◄────┘         │               │
└────┬────┘     └────┬────┘               │               │
     │               │                    │               │
     └───────┬───────┘                    │               │
             │                            │               │
             ▼                            │               │
        ╔═════════╗                       │               │
        ║ tensor  ║◄──────────────────────┴───────────────┘
        ╚════╤════╝
             │
     ┌───────┼───────┬───────────┬───────────┐
     │       │       │           │           │
     ▼       ▼       ▼           ▼           ▼
┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐
│  iter  ││broadcast││  ffi   ││workspace││ index  │
└───┬────┘└───┬────┘└───┬────┘└───┬────┘└───┬────┘
    │         │         │         │         │
    ▼         │         │         │         ▼
┌────────┐    │         │         │    ┌────────┐
│  ops   │◄───┘         │         │    │shape_ops│
└───┬────┘              │         │    └───┬────┘
    │                   │         │        │
    └───────────────────┴─────────┴────────┘
            
             ═══════════════════════════════
             ║     横切关注点（可选）      ║
             ═══════════════════════════════
             
    ┌─────────────┐              ┌─────────────┐
    │   parallel  │              │    simd     │
    │   (rayon)   │              │   (pulp)    │
    └──────┬──────┘              └──────┬──────┘
           │                            │
           └────────────┬───────────────┘
                        │
                        ▼
                   ┌─────────┐
                   │  iter   │
                   │  ops    │
                   └─────────┘
```

### 4.2 依赖层级

| 层级 | 模块 | 依赖 |
|------|------|------|
| **L0** | error, private | 无 |
| **L1** | dimension, element, complex | error |
| **L2** | layout | error, dimension |
| **L3** | storage | error, layout, element |
| **L4** | tensor | storage, dimension, layout, element |
| **L5** | iter, broadcast, ffi, workspace | tensor |
| **L6** | ops | tensor, iter, broadcast |
| **L7** | shape_ops, index | tensor, ops |
| **L8** | construct, convert, format | tensor, shape_ops |
| **横切** | parallel, simd | iter, ops（通过 feature gate） |

---

## 5. Feature Gate 矩阵

| 功能 | 默认 (std) | no_std | +parallel | +simd | +parallel+simd |
|------|:----------:|:------:|:---------:|:-----:|:--------------:|
| 基础张量操作 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 视图/视图可变 | ✅ | ✅ | ✅ | ✅ | ✅ |
| Arc 存储 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 迭代器 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 逐元素运算 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 归约运算 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 形状操作 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 索引/切片 | ✅ | ✅ | ✅ | ✅ | ✅ |
| Display 格式化 | ✅ | ❌ | ✅ | ✅ | ✅ |
| Box<Vec> 分配 | ✅ | ❌ | ✅ | ✅ | ✅ |
| 并行迭代器 | ❌ | ❌ | ✅ | ❌ | ✅ |
| 并行归约 | ❌ | ❌ | ✅ | ❌ | ✅ |
| SIMD 向量化 | ❌ | ❌ | ❌ | ✅ | ✅ |
| BLAS 兼容 API | ✅ | ✅ | ✅ | ✅ | ✅ |

### 5.1 Feature Gate 实现模式

```rust
// lib.rs 中的 feature gate 示例

#[cfg(feature = "std")]
pub use std::vec::Vec;

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
pub use alloc::vec::Vec;

// 并行模块
#[cfg(feature = "parallel")]
pub mod parallel;

// SIMD 加速的运算
#[cfg(feature = "simd")]
pub mod simd;

// 条件编译的方法实现
impl<S, D> TensorBase<S, D> {
    #[cfg(feature = "parallel")]
    pub fn par_map<F>(&self, f: F) -> Self
    where
        F: Fn(&T) -> T + Sync,
        T: Send,
    {
        // rayon 并行实现
    }

    #[cfg(not(feature = "parallel"))]
    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(&T) -> T,
    {
        // 串行实现
    }
}
```

---

## 6. 公开 API 层次

### 6.1 prelude.rs 导出

```rust
// src/prelude.rs

// 核心类型
pub use crate::tensor::{
    TensorBase,
    Tensor,           // TensorBase<Owned<T>, D>
    TensorView,       // TensorBase<ViewRepr<&T>, D>
    TensorViewMut,    // TensorBase<ViewMutRepr<&mut T>, D>
    ArcTensor,        // TensorBase<ArcRepr<T>, D>
};

// 维度类型
pub use crate::dimension::{
    Dimension,
    IntoDimension,
    Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6,
    IxDyn,
    Axis,
};

// 布局和形状
pub use crate::layout::{Layout, Strides};
pub use crate::shape_ops::ShapeError;

// 切片宏
pub use crate::index::s;

// 常用 trait
pub use crate::element::{
    Element,
    Numeric,
    RealScalar,
    ComplexScalar,
};
pub use crate::complex::Complex;

// 构造函数（便捷导入）
pub use crate::construct::{
    zeros, ones, eye,
    arange, linspace, logspace,
    from_vec, from_fn,
};

// 运算 trait
pub use crate::ops::{
    Elementwise,
    Reduction,
    MatrixOps,
};
```

### 6.2 lib.rs 模块结构

```rust
// src/lib.rs

//! # Xenon
//!
//! 一个用于科学计算的 Rust N 维数组库。

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(not(feature = "std"))]
extern crate alloc;

// 内部模块
mod private;

// 公开模块
pub mod error;
pub mod dimension;
pub mod element;
pub mod complex;
pub mod storage;
pub mod layout;
pub mod tensor;
pub mod iter;
pub mod ops;
pub mod broadcast;
pub mod shape_ops;
pub mod index;
pub mod construct;
pub mod convert;
pub mod format;
pub mod ffi;
pub mod workspace;

// 条件编译模块
#[cfg(feature = "parallel")]
#[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
pub mod parallel;

#[cfg(feature = "simd")]
#[cfg_attr(docsrs, doc(cfg(feature = "simd")))]
pub mod simd;

// Prelude
pub mod prelude;

// 便捷 re-exports
pub use prelude::*;
pub use error::Error;
```

### 6.3 API 稳定性保证

| 层级 | 稳定性 | 说明 |
|------|--------|------|
| `prelude::*` | **稳定** | 主版本号内保持兼容 |
| 公开 trait 方法 | **稳定** | 只增不减 |
| 内部模块 (`mod private`) | **不稳定** | 随时可能变更 |
| `#[doc(hidden)]` | **不稳定** | 仅供内部使用 |
| `simd`/`parallel` 模块 | **实验性** | 可能有破坏性变更 |

---

## 7. 实现任务分解

### Wave 1: 基础设施（可完全并行）

| 任务 | 依赖 | 预估复杂度 | 产出 |
|------|------|------------|------|
| W1.1 error types | 无 | 低 | `Error`, `ShapeError`, `IndexError` |
| W1.2 private module | 无 | 低 | `Sealed` trait |
| W1.3 dimension traits | W1.1 | 中 | `Dimension`, `IntoDimension` |
| W1.4 static dimensions | W1.3 | 中 | `Ix0`-`Ix6` |
| W1.5 dynamic dimension | W1.3 | 中 | `IxDyn` |
| W1.6 element traits | W1.1 | 中 | `Element`, `Numeric`, `RealScalar` |
| W1.7 Complex<T> | W1.6 | 高 | 自定义复数类型 |
| W1.8 layout flags | W1.1 | 低 | `LayoutFlags` (5-bit u8) |
| W1.9 strides | W1.1, W1.3 | 中 | `Strides` 计算逻辑 |

### Wave 2: 核心（依赖 Wave 1）

| 任务 | 依赖 | 预估复杂度 | 产出 |
|------|------|------------|------|
| W2.1 Storage trait | W1.6, W1.8, W1.9 | 高 | `Storage`, `RawStorage` |
| W2.2 Owned storage | W2.1 | 中 | `Owned<T>` + 64 字节对齐分配 |
| W2.3 View storage | W2.1 | 中 | `ViewRepr<&T>` |
| W2.4 ViewMut storage | W2.1 | 中 | `ViewMutRepr<&mut T>` |
| W2.5 Arc storage | W2.1 | 高 | `ArcRepr<T>` |
| W2.6 TensorBase | W2.1-W2.5, W1.3-W1.5 | 高 | 核心结构体 |
| W2.7 Type aliases | W2.6 | 低 | `Tensor`, `TensorView` 等 |

### Wave 3: 操作（依赖 Wave 2）

| 任务 | 依赖 | 预估复杂度 | 产出 |
|------|------|------------|------|
| W3.1 Elements iterator | W2.6 | 中 | 扁平元素迭代 |
| W3.2 Axis iterator | W2.6 | 中 | 沿轴迭代 |
| W3.3 Window iterator | W2.6 | 高 | 窗口迭代 |
| W3.4 Zip iterator | W2.6, W3.1 | 高 | 多张量同步迭代 |
| W3.5 Elementwise ops | W3.1 | 中 | map, zip_with |
| W3.6 Arithmetic ops | W3.5 | 中 | Add, Sub, Mul, Div |
| W3.7 Reduction | W3.1 | 高 | sum, mean, var 等 |
| W3.8 Matrix ops | W2.6 | 高 | dot, matvec, outer |
| W3.9 Broadcast | W2.6 | 高 | 广播规则 |
| W3.10 Reshape | W2.6 | 中 | reshape, into_shape |
| W3.11 Transpose | W2.6 | 中 | transpose, permute_axes |
| W3.12 Slice | W2.6 | 高 | 切片操作 |
| W3.13 Multi-dim index | W2.6 | 中 | [i, j, k] 索引 |
| W3.14 Slice index | W3.12 | 中 | s![.., 0..10] |
| W3.15 Advanced index | W3.13 | 高 | 布尔掩码、整数数组 |

### Wave 4: 集成（依赖 Wave 3）

| 任务 | 依赖 | 预估复杂度 | 产出 |
|------|------|------------|------|
| W4.1 construct | W2.6, W3.10 | 中 | zeros, ones, arange 等 |
| W4.2 convert | W2.6 | 中 | cast, to_owned |
| W4.3 format | W2.6 | 低 | Display/Debug |
| W4.4 ffi | W2.6 | 中 | 原始指针 API |
| W4.5 workspace | W2.6 | 中 | 临时缓冲区 |
| W4.6 accumulate | W3.7 | 中 | cumsum, cumprod |
| W4.7 set_ops | W3.7 | 中 | unique, histogram |
| W4.8 comparison | W3.5 | 低 | is_close, clip |

### Wave 5: 性能（依赖 Wave 4）

| 任务 | 依赖 | 预估复杂度 | 产出 |
|------|------|------------|------|
| W5.1 par_iter | W3.1-W3.4 | 高 | 并行迭代器 |
| W5.2 par_reduction | W3.7, W5.1 | 高 | 并行归约 |
| W5.3 simd elementwise | W3.5 | 高 | SIMD 逐元素 |
| W5.4 simd reduction | W3.7 | 高 | SIMD 归约 |
| W5.5 simd matrix | W3.8 | 非常高 | SIMD 矩阵运算 |

### Wave 6: 质量（依赖 Wave 5）

| 任务 | 依赖 | 预估复杂度 | 产出 |
|------|------|------------|------|
| W6.1 unit tests | W1-W5 | 中 | 各模块单元测试 |
| W6.2 integration tests | W1-W5 | 高 | 端到端测试 |
| W6.3 proptest | W1-W5 | 中 | 属性测试 |
| W6.4 benchmarks | W1-W5 | 高 | criterion 基准 |
| W6.5 docs | W1-W5 | 高 | 文档注释 |
| W6.6 examples | W1-W5 | 中 | 使用示例 |

### 并行执行策略

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
        [W3.4] [W3.5] [W3.7] [W3.8] [W3.14] [W3.15]
                           │
                           ▼
Wave 4: [W4.1] [W4.2] [W4.3] [W4.4] [W4.5] [W4.6] [W4.7] [W4.8]
                           │
                           ▼
Wave 5: [W5.1] [W5.2] [W5.3] [W5.4] [W5.5]
                           │
                           ▼
Wave 6: [W6.1] [W6.2] [W6.3] [W6.4] [W6.5] [W6.6]
```

---

## 附录 A: 核心类型速查

```rust
// 张量核心类型
TensorBase<S, D>              // 泛型基础类型
Tensor<T, D>                  // = TensorBase<Owned<T>, D>
TensorView<'a, T, D>          // = TensorBase<ViewRepr<&'a T>, D>
TensorViewMut<'a, T, D>       // = TensorBase<ViewMutRepr<&'a mut T>, D>
ArcTensor<T, D>               // = TensorBase<ArcRepr<T>, D>

// 维度类型
Ix0, Ix1, Ix2, ..., Ix6       // 静态维度 (0-6 维)
IxDyn                         // 动态维度

// 布局标志
LayoutFlags: u8
├── F_CONTIGUOUS    (0b00001) // Fortran 连续
├── C_CONTIGUOUS    (0b00010) // C 连续
├── ALIGNED         (0b00100) // 64 字节对齐
├── HAS_ZERO_STRIDE (0b01000) // 包含零步长（广播）
└── HAS_NEG_STRIDE  (0b10000) // 包含负步长（翻转）

// 元素 trait 层次
Element                        // 基础：可复制、可默认初始化
└── Numeric                    // 数值：支持加减乘除
    ├── RealScalar             // 实数：支持 sqrt, sin, cos 等
    └── ComplexScalar          // 复数：支持共轭、模长等
```

## 附录 B: 内存布局约定

| 属性 | 值 | 原因 |
|------|-----|------|
| 默认内存序 | Fortran (列优先) | BLAS/LAPACK 兼容 |
| 对齐 | 64 字节 | AVX-512 缓存行对齐 |
| 步长类型 | `isize` | 支持负步长（翻转视图） |
| 形状类型 | `usize` | 仅非负 |

## 附录 C: 命名约定

| 模式 | 示例 | 说明 |
|------|------|------|
| 类型 | `Tensor`, `TensorView` | 大驼峰 |
| 函数 | `zeros`, `into_shape` | 蛇形 |
| 宏 | `s![]`, `azip!` | 蛇形 + 感叹号 |
| 常量 | `F_CONTIGUOUS` | 大写蛇形 |
| 静态方法 | `Tensor::zeros` | 关联函数 |
| trait | `Dimension`, `Numeric` | 大驼峰 |

---

*本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。*
