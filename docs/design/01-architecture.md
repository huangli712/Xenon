# 项目总体架构

> 文档编号: 01 | 适用范围: 项目总体架构与跨模块边界 | 阶段: Phase 0
> 前置文档: `00-coding.md`
> 需求参考: 需求说明书 §1, §2, §3, §4, §5, §6, §7, §8, §9, §10, §11, §12, §13, §14, §15, §16, §17, §18, §19, §20, §21, §22, §23, §24, §25, §26, §27, §28
> 范围声明: 范围内

> **格式豁免声明**：本文档为全局架构规范，按 `design.md` §3 豁免标准章节结构。

---

## 1. 项目概览

### 1.1 定位

Xenon 是一个纯 Rust 实现的 N 维数组（张量）库，定位为科学计算的数值基础设施。设计理念与 NumPy ndarray 层相似，但针对 Rust 生态系统进行了深度优化：类型安全、内存高效、零成本抽象、F-order 单一布局。

### 1.2 目标用户

| 用户类型   | 核心诉求                                             |
| ---------- | ---------------------------------------------------- |
| 库开发者   | 稳定 API、高性能互操作                               |
| 系统开发者 | `std` 环境下的底层内存控制、确定性数值行为、最小依赖 |
| 间接用户   | 性能、正确性、与 Python 经验的直觉一致性             |

### 1.3 核心设计原则

| 原则                 | 描述                                                                                                                               |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **正确性优先**       | 类型安全、内存安全、数值精度满足 IEEE 754                                                                                          |
| **零成本抽象**       | 视图、索引、元数据访问等核心路径追求零额外抽象成本；分配、并行调度、格式化等高层操作不做此承诺                                     |
| **内存可控**         | 基于 `std` 显式控制分配和对齐                                                                                                      |
| **渐进增强**         | 核心功能无依赖，并行/SIMD 通过独立 backend 模块和 feature gate 按需启用                                                            |
| **错误语义集中裁决** | 可恢复错误统一通过 `Result` 报告；公开安全索引入口使用 `try_at` / `try_at_mut` 等可恢复错误模型；`[]` 语法糖仅限已验证索引的快捷路径 |
| **FFI 友好**         | 提供 C 兼容的 F-order 内存布局，便于与 BLAS/LAPACK 互操作                                                                          |

### 1.4 工程约束

| 约束       | 要求                                                                                       |
| ---------- | ------------------------------------------------------------------------------------------ |
| Crate 结构 | 单 crate，遵循 SemVer                                                                      |
| MSRV       | Rust 1.85+                                                                                 |
| License    | MIT                                                                                        |
| 平台支持   | 当前版本仅支持 `std` 环境                                                                  |
| 默认内存序 | Fortran-order（列优先），不支持 C-order                                                    |
| 内存对齐   | 64 字节（缓存行对齐，AVX-512 友好）                                                        |
| 外部依赖   | 仅 rayon（可选并行）+ pulp（可选 SIMD）；benchmark 与扩展验证工具不属于 crate 合约的一部分 |
| 工程边界   | 保持单 crate；外部依赖仅限可选 `rayon` / `pulp`                                            |

---

## 2. 范围

### 2.1 范围内

- N 维数组的存储、构造
- N 维数组的索引操作（多维整数索引、范围切片）、形状操作（仅转置）
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

### 2.3 需求映射与范围约束

| 类型     | 内容                                                                 |
| -------- | -------------------------------------------------------------------- |
| 需求映射 | `require.md §1`, `§2`, `§3`, `§4`, `§5`, `§6`, `§7`, `§8`, `§9`, `§10`, `§11`, `§12`, `§13`, `§14`, `§15`, `§16`, `§17`, `§18`, `§19`, `§20`, `§21`, `§22`, `§23`, `§24`, `§25`, `§26`, `§27`, `§28` |
| 范围内   | 总体分层、模块边界、feature gate、依赖约束、错误语义与质量边界       |
| 范围外   | 单模块内部算法、实现细节、额外平台与超范围能力扩展                   |
| 非目标   | 通过总体架构文档新增需求未授权的 API、依赖或多 crate 拆分            |

---

## 3. 目录结构

```
xenon/
├── Cargo.toml                 # Package manifest and feature definitions
├── rustfmt.toml               # Rust formatting configuration
├── README.md
├── LICENSE                    # MIT
├── CHANGELOG.md
│
├── src/
│   ├── lib.rs                 # Crate root: feature gates, re-exports, docs
│   ├── prelude.rs             # Common pub-use exports
│   ├── private.rs             # Sealed-trait infrastructure
│   ├── dispatch.rs            # Internal dispatch helper (ExecPath, ParallelGuard, thresholds, SIMD checks)
│   ├── error.rs               # XenonError enum and Result alias
│   │
│   ├── dimension/             # Dimension type system
│   │   ├── mod.rs             # Dimension trait definition
│   │   ├── static_dims.rs     # Ix0, Ix1, ..., Ix6 static dimensions
│   │   ├── dynamic.rs         # IxDyn dynamic dimension
│   │   ├── into_dimension.rs  # IntoDimension trait
│   │   └── axes.rs            # Axis marker and axis operations
│   │
│   ├── element/               # Element trait hierarchy
│   │   ├── mod.rs             # Element trait definition
│   │   ├── numeric.rs         # Numeric trait for arithmetic
│   │   ├── real.rs            # RealScalar trait for real numbers
│   │   ├── complex.rs         # ComplexScalar trait for complex numbers
│   │   └── primitives.rs      # Primitive impls (f32, f64, i32, i64, bool)
│   │
│   ├── complex/               # Custom complex type
│   │   ├── mod.rs             # Complex<T> definition, #[repr(C)]
│   │   └── ops.rs             # Arithmetic implementations
│   │
│   ├── storage/               # Storage system (buffer and ownership only)
│   │   ├── mod.rs             # Storage and RawStorage traits
│   │   ├── owned.rs           # Owned<A> owned storage
│   │   ├── view.rs            # ViewRepr<'a, A> immutable view
│   │   ├── view_mut.rs        # ViewMutRepr<'a, A> mutable view
│   │   ├── arc.rs             # ArcRepr<A> atomic reference-counted storage
│   │   ├── alloc.rs           # 64-byte aligned allocator
│   │   └── traits.rs          # Marker traits such as IsOwned and IsView
│   │
│   ├── layout/                # Memory layout (F-order only)
│   │   ├── mod.rs             # Module-level layout helpers and validation entry points
│   │   ├── flags.rs           # Layout flags (F_CONTIGUOUS, ALIGNED, ...)
│   │   ├── strides.rs         # F-order stride calculation and validation
│   │   └── contiguous.rs      # Contiguity checks
│   │
│   ├── tensor/                # TensorBase core
│   │   ├── mod.rs             # TensorBase<S, D> struct
│   │   ├── impls.rs           # Core methods (shape, strides, data_ptr)
│   │   ├── aliases.rs         # Type aliases (Tensor, TensorView, ...)
│   │   └── construct.rs       # Internal constructors
│   │
│   ├── iter/                  # Iterator system
│   │   ├── mod.rs             # Iterator trait definitions
│   │   ├── elements.rs        # Elements iterator (flat traversal)
│   │   ├── axis.rs            # AxisIter over one axis
│   │   └── indexed.rs         # IndexedIter with indices
│   │
│   ├── simd/                  # SIMD backend (feature = "simd")
│   │   ├── mod.rs             # pulp integration, dispatch facade, public kernel trait
│   │   └── vector.rs          # Vectorized implementation
│   │
│   ├── parallel/              # Parallel backend (feature = "parallel")
│   │   ├── mod.rs             # Module entry, re-exports, ParallelPool
│   │   ├── par_iter.rs        # ParElements and TensorBase::par_iter()
│   │   ├── map.rs             # par_map, par_map_with_threshold, par_zip_map
│   │   ├── reduce.rs          # par_reduce_impl, par_sum, par_dot
│   │   └── checked.rs         # par_map_checked and error/panic propagation
│   │
│   ├── broadcast/             # Broadcast rules and read-only views
│   │   ├── mod.rs             # Module entry and re-exports
│   │   ├── shape.rs           # Compatibility and stride rules
│   │   └── view.rs            # broadcast_to() and broadcast_with()
│   │
│   ├── math/                  # Element-wise math
│   │   ├── mod.rs             # Module entry and re-exports
│   │   ├── unary.rs           # Unary ops (abs, neg, signum, square, sin, sqrt, exp, ln, floor, ceil, modulus, conj, not)
│   │   ├── binary.rs          # Binary arithmetic methods (add, sub, mul, div, add_scalar, sub_scalar, mul_scalar, div_scalar)
│   │   └── comparison.rs      # Comparison ops (eq, ne, lt, gt)
│   │
│   ├── overload/              # Operator overloading
│   │   ├── mod.rs             # Operator trait exports
│   │   └── arithmetic.rs      # Add, Sub, Mul, Div implementations
│   │
│   ├── util/                  # Utility operations
│   │   ├── mod.rs             # Module root and re-exports
│   │   ├── clip.rs            # clip
│   │   ├── fill.rs            # fill
│   │   └── contiguous.rs      # to_contiguous public entry point
│   │
│   ├── set/                   # Set operations
│   │   ├── mod.rs             # Set operation exports
│   │   └── unique.rs          # unique
│   │
│   ├── matrix/                # Matrix operations
│   │   ├── mod.rs             # Module entry, re-exports, dot() API
│   │   └── dot.rs             # Vector inner product, may delegate to `simd/`
│   │
│   ├── reduction/             # Reduction operations
│   │   ├── mod.rs             # Module root and re-exports
│   │   └── sum.rs             # Global sum and sum_axis, may delegate to `simd/` or `parallel/`
│   │
│   ├── shape/                 # Shape operations
│   │   ├── mod.rs             # Shape operation trait
│   │   └── transpose.rs       # transpose
│   │
│   ├── index/                 # Indexing system
│   │   ├── mod.rs             # Index trait definitions
│   │   ├── multi_dim.rs       # Multi-dimensional integer indexing [i, j, k]
│   │   └── slice_index.rs     # Range-slice indexing
│   │
│   ├── construct/             # Tensor construction
│   │   ├── mod.rs             # Module root and re-exports
│   │   ├── init.rs            # zeros, ones
│   │   ├── eye.rs             # eye
│   │   ├── from_data.rs       # from_shape_vec, from_shape_slice, from_array, from_vec (non-normative convenience)
│   │   └── scalar.rs          # from_scalar
│   │
│   ├── convert/               # Type conversion
│   │   ├── mod.rs             # Module root and re-exports
│   │   ├── cast.rs            # Consumes element::CastTo and hosts conversion implementations
│   │   ├── owned.rs           # to_owned, into_owned, storage-mode conversions
│   │   ├── from_impl.rs       # From/TryFrom implementations
│   │   └── contiguous.rs      # Internal contiguous helper for convert/util
│   │
│   ├── format/                # Formatting output
│   │   ├── mod.rs             # Module root, re-exports, cfg gates
│   │   ├── config.rs          # FormatConfig
│   │   ├── display.rs         # Display implementation
│   │   ├── debug.rs           # Debug implementation
│   │   └── pretty.rs          # NumPy-style formatting helpers
│   │
│   ├── ffi/                   # FFI interface
│   │   ├── mod.rs             # Module root and re-exports
│   │   ├── types.rs           # BlasLayout, BlasTrans, BlasInfo definitions
│   │   ├── ptr.rs             # Raw pointer API (export/export_mut, from_raw_parts, into_raw_parts)
│   │   ├── blas.rs            # BLAS compatibility checks (is_blas_compatible, blas_info, lda)
│   │   └── offset.rs          # Index-to-pointer offset (export/export_mut helpers rely on checked arithmetic)
│   │
│   ├── workspace/             # Temporary workspace
│       ├── mod.rs             # Module root and re-exports
│       ├── error.rs           # Internal workspace error type
│       ├── workspace.rs       # Workspace struct, constants, construction, destruction
│       ├── borrow.rs          # WorkspaceBorrow, WorkspaceBorrowMut guards
│       ├── split.rs           # SplitBorrowMut guard
│       └── expand.rs          # ensure_capacity and reallocate
│
├── tests/                     # Integration tests
│   ├── common/
│   │   └── mod.rs             # Shared test helpers
│   ├── test_tensor.rs         # Tensor core tests
│   ├── test_math.rs           # Math operation tests
│   ├── test_broadcast.rs      # Broadcast tests
│   └── test_index.rs          # Indexing tests
│
├── benches/                   # Performance benchmarks
│   ├── math.rs                # Element-wise operations
│   ├── reduction.rs           # Reduction operations
│   ├── dot_product.rs         # Vector inner product
│   ├── set.rs                 # Set operations
│   ├── broadcast.rs           # Broadcast operations
│   ├── shape.rs               # Shape operations
│   ├── simd_comparison.rs     # SIMD comparison
│   ├── parallel_comparison.rs # Parallel comparison
│   └── construction.rs        # Tensor construction
│
└── examples/                  # Usage examples
    └── basic.rs               # Basic usage
```

### 模块职责速览

| 模块           | 职责                                                                                                               |
| -------------- | ------------------------------------------------------------------------------------------------------------------ |
| `error.rs`     | `XenonError` 统一错误枚举，`Result<T>` 类型别名                                                                    |
| `dispatch.rs`  | 私有内部执行路径裁决层（`ExecPath`、`ParallelGuard`、阈值、SIMD 条件判断），不作为公开模块导出 |
| `dimension/`   | `Dimension` trait 和静态/动态维度类型（Ix0-Ix6, IxDyn）                                                            |
| `element/`     | 元素类型 trait 层次（Element → Numeric → RealScalar/ComplexScalar；`usize` 仅作为索引/形状类型，不属于元素算术层） |
| `complex/`     | 自定义 `Complex<T>` 类型，`#[repr(C)]` 兼容 C FFI                                                                  |
| `storage/`     | 四种存储模式（Owned/ViewRepr/ViewMutRepr/ArcRepr）                                                                 |
| `layout/`      | F-order 布局函数、步长计算、连续性检查与验证入口                                                                   |
| `tensor/`      | 核心 `TensorBase<S, D>` 结构体及类型别名                                                                           |
| `iter/`        | 元素/轴/索引迭代器                                                                                                 |
| `simd/`        | SIMD 后端：向量化 kernel（pulp）、运行时分发，不含标量回退                                         |
| `parallel/`    | 并行后端：多文件模块，承载纯并行执行入口（par_map/par_zip_map/par_sum/par_dot）和 ParElements 迭代器，不含串行回退         |
| `math/`        | 逐元素数学运算（一元、二元算术、比较），按需委托 `simd/` / `parallel/`                                            |
| `overload`     | 运算符重载（Add, Sub, Mul, Div trait 实现）                                                                        |
| `util/`        | 实用操作（clip 裁剪、fill 填充、to_contiguous 连续性保证的公共入口）                                               |
| `set/`         | 集合操作（unique 去重）                                                                                            |
| `broadcast/`   | NumPy 广播规则与只读广播视图构造                                                                                   |
| `matrix/`      | 向量内积 dot，必要时委托 `simd/`                                                                                   |
| `reduction/`   | 归约操作（sum），必要时委托 `simd/` / `parallel/`                                                                  |
| `shape/`       | transpose                                                                                                          |
| `index/`       | 多维整数索引、范围切片索引                                                                                         |
| `construct/`   | 张量构造（`zeros`、`ones`、`eye`、from-data、`from_scalar`）                                                       |
| `convert/`     | 类型转换（cast、存储模式互转、From trait；必要时复用内部 contiguous helper）                                       |
| `format/`      | NumPy 风格格式化输出                                                                                               |
| `ffi/`         | 原始指针 API、BLAS 兼容性检查、多维索引偏移；公开入口含 `export()` / `export_mut()` 与 `try_offset_of()` / `try_ptr_at()` |
| `workspace/`   | 临时工作空间（对齐分配、借用守卫、分割、扩容；错误类型仅内部使用）                                                 |

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
categories = ["science", "mathematics", "data-structures"]

[features]

# Parallel computing (depends on rayon)
parallel = ["dep:rayon"]

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

Xenon 仅支持 `std` 环境；`simd` 与 `parallel` 都建立在该无条件前提之上，不再把 `std` 建模为独立 feature。

### Feature 组合说明

| 组合    | 命令                       | 适用场景                 |
| ------- | -------------------------- | ------------------------ |
| 标准    | （默认）                   | 桌面应用、CLI 工具       |
| 高性能  | `--features parallel,simd` | 数据科学、机器学习       |
| 仅 SIMD | `--features simd`          | 需 SIMD 但无需并行的场景 |

---

## 5. 模块依赖关系

### 5.1 依赖层级

各模块的详细设计参见对应编号文档。层级关系如下：

| 层级   | 模块                                                       | 依赖                                                                     | 参见                                                                                                                             |
| ------ | ---------------------------------------------------------- | ------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| **L0** | error, private                                             | 无                                                                       | `26-error.md`                                                                                                                    |
| **L1** | dimension, element, complex                                | error（element 额外依赖 complex）                                        | `02-dimension.md`、`03-element.md`、`04-complex.md`                                                                              |
| **L2** | layout                                                     | error, dimension（提供模块级布局函数与判定规则，不承诺额外公共布局类型） | `06-layout.md`                                                                                                                   |
| **L2** | workspace                                                  | std（独立于核心类型系统，可被上游库直接使用）                            | `24-workspace.md`                                                                                                                |
| **L3** | storage                                                    | core, alloc, std::sync::Arc, crate::error（只持有底层连续缓冲区，不消费 `dimension` 或 `layout`） | `05-storage.md`                                                                                                                  |
| **L4** | tensor                                                     | storage, dimension, layout, element                                      | `07-tensor.md`                                                                                                                   |
| **L5** | broadcast, iter, ffi, simd, dispatch                       | tensor（simd 额外依赖 layout/element）                                    | `15-broadcast.md`、`10-iterator.md`、`23-ffi.md`、`08-simd.md`、`09-parallel.md`、`01-architecture.md`                           |
| **L6** | math, overload, set, matrix, reduction, shape, index, util, parallel | tensor, broadcast，以及按需调用 `dispatch` 做路径选择并委托后端或模块内串行实现 | `11-math.md`、`12-matrix.md`、`13-reduction.md`、`14-set.md`、`16-shape.md`、`17-indexing.md`、`19-overload.md`、`20-utility.md`、`09-parallel.md` |
| **L7** | construct, convert, format                                 | tensor, shape, element, complex, storage                                 | `18-construction.md`、`21-type.md`、`22-output.md`                                                                               |

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
L5:  iter    broadcast    ffi    simd    dispatch
      │         │           │      │        │
      └─────────┴───────────┴──────┴────────┘
                           │
L6:        math   overload   set   matrix   reduction   shape   index   util   parallel
                           │
L7:                 construct   convert   format
```

> **L5/L6 模块说明**：`simd` 与 `parallel` 保持为独立后端模块，各自只提供纯执行能力（向量化/并行），不含串行回退。`dispatch.rs` 是内部执行路径裁决层，统一承载阈值判断、嵌套并行防护和 SIMD 条件检查；各 L6 语义模块通过 `dispatch::select_exec_path()` 选择串行、SIMD 或并行路径，再委托到对应后端或使用本模块串行实现。`dispatch.rs` 不属于稳定公开模块面。

### 5.3 依赖合法性与新增依赖说明

| 项目           | 说明                                                            |
| -------------- | --------------------------------------------------------------- |
| 新增第三方依赖 | 仅 `rayon` 与 `pulp` 作为可选依赖；架构文档本身不新增其他第三方依赖 |
| 合法性结论     | 符合最小依赖限制                                                |
| 替代方案       | 其他第三方 crate 不适用；核心能力优先使用标准库与 crate 内部模块 |

---

## 6. Feature Gate 矩阵

| 功能             | 默认 | +parallel | +simd | +parallel+simd |
| ---------------- | :--: | :-------: | :---: | :------------: |
| 基础张量操作     |  ✅  |    ✅     |  ✅   |       ✅       |
| 视图/视图可变    |  ✅  |    ✅     |  ✅   |       ✅       |
| Arc 存储         |  ✅  |    ✅     |  ✅   |       ✅       |
| 迭代器           |  ✅  |    ✅     |  ✅   |       ✅       |
| 逐元素非数学运算 |  ✅  |    ✅     |  ✅   |       ✅       |
| 逐元素数学函数   |  ✅  |    ✅     |  ✅   |       ✅       |
| sum 归约         |  ✅  |    ✅     |  ✅   |       ✅       |
| 内积 (dot)       |  ✅  |    ✅     |  ✅   |       ✅       |
| transpose        |  ✅  |    ✅     |  ✅   |       ✅       |
| 整数索引 / 切片  |  ✅  |    ✅     |  ✅   |       ✅       |
| Display 格式化   |  ✅  |    ✅     |  ✅   |       ✅       |
| 并行执行后端     |  ❌  |    ✅     |  ❌   |       ✅       |
| SIMD 向量化      |  ❌  |    ❌     |  ✅   |       ✅       |
| BLAS 兼容 API    |  ✅  |    ✅     |  ✅   |       ✅       |

> **说明**：`parallel` 只表示内部执行后端可用，不引入额外的公开并行迭代 API。公开 API 的并行加速为透明内部实现。

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

// Layout helpers stay module-scoped and are not re-exported by prelude.

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

// Construction helpers
pub use crate::construct::{
    zeros, ones, eye,
    from_shape_vec,
};
```

> **注**：当前版本 `prelude` 仅导出 `require.md` 已授权的稳定公开项；`s!` 之类便捷宏不纳入默认导出面。

---

## 8. lib.rs 模块结构

```rust
// src/lib.rs

//! # Xenon
//!
//! A Rust N-dimensional array library for scientific computing.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
#![warn(rust_2024_compatibility)]
#![warn(unsafe_op_in_unsafe_fn)]

// Internal modules
mod private;
mod dispatch;

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

// `simd` public surface is limited to the dispatch facade and public kernel
// traits. Concrete kernels and ISA detection details remain `pub(crate)` or
// `#[doc(hidden)]` implementation details.

#[cfg(feature = "parallel")]
#[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
pub(crate) mod parallel;

// Prelude
pub mod prelude;

// Convenience re-exports
pub use prelude::*;
pub use error::XenonError;
```

---

## 9. API 稳定性说明

| 层级                     | 稳定性     | 说明                                                                 |
| ------------------------ | ---------- | -------------------------------------------------------------------- |
| `prelude::*`             | **稳定**   | 主版本号内保持兼容                                                   |
| 公开 trait 方法          | **稳定**   | 只增不减                                                             |
| 内部模块 (`mod private`, `mod dispatch`) | **不稳定** | 随时可能变更                                             |
| `#[doc(hidden)]`         | **不稳定** | 仅供内部使用                                                         |
| `simd` 模块             | **稳定**   | 作为公开模块纳入 SemVer；稳定公开面仅含 dispatch facade 与公共 kernel trait；feature gate 仅控制可用性，不降低兼容性承诺 |
| 内部后端 (`parallel`)   | **不稳定** | `pub(crate)` 内部实现，仅影响自动并行加速，不构成公开模块 API         |

---

## 10. 核心类型速查

各类型的详细设计参见对应模块文档（`02-dimension.md`、`03-element.md`、`05-storage.md`、`06-layout.md`）。

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

// Layout helpers (F-order only)
compute_f_strides(shape)       // Compute canonical F-order strides
validate_layout(shape, strides, offset, storage_len)
classify_layout(shape, strides) // Internal classification helper

// Element trait hierarchy
Element                        // Base: Copy + PartialEq + Debug + Display + Send + Sync
└── Numeric                    // Numeric: arithmetic syntax + checked integer contract + conjugate semantics
    ├── RealScalar             // Real: sqrt, sin, exp, ln, floor, ceil
    └── ComplexScalar          // Complex: complex-specific conj/modulus/re/im helpers (`Complex<f32> -> f32`, `Complex<f64> -> f64`)
```

其中 `Numeric` 不仅表示 `Add + Sub + Mul + Div + Neg` 语法可用，还要求：

- 对整数路径，具体运算模块必须落实 checked overflow / divide-by-zero / unrepresentable-result contract；
- 对实数类型，`conjugate(self)` 为恒等；对复数类型，`conjugate(self)` 执行数学共轭；
- 统一错误入口与结构化字段约束遵循 `00-coding.md` §4.2，不得在架构层引入第二套公开错误模型。

---

## 11. 实现任务分解

### Wave 1: 基础设施（可完全并行）

| 任务                   | 依赖       | 预估复杂度 | 产出                               |
| ---------------------- | ---------- | ---------- | ---------------------------------- |
| W1.1 error types       | 无         | 低         | `XenonError`, `Result<T>`          |
| W1.2 private module    | 无         | 低         | `Sealed` trait                     |
| W1.3 dimension traits  | W1.1       | 中         | `Dimension`, `IntoDimension`       |
| W1.4 static dimensions | W1.3       | 中         | `Ix0`-`Ix6`                        |
| W1.5 dynamic dimension | W1.3       | 中         | `IxDyn`                            |
| W1.6 element traits    | W1.1       | 中         | `Element`, `Numeric`, `RealScalar` |
| W1.7 Complex\<T\>      | W1.6       | 高         | 自定义复数类型                     |
| W1.8 layout helpers    | W1.1       | 低         | 模块级布局函数与判定入口          |
| W1.9 F-order strides   | W1.1, W1.3 | 中         | F-order 步长计算                   |

### Wave 2: 核心（依赖 Wave 1）

| 任务                 | 依赖                 | 预估复杂度 | 产出                                         |
| -------------------- | -------------------- | ---------- | -------------------------------------------- |
| W2.1 Storage trait   | 无                   | 高         | `Storage`, `RawStorage`（仅依赖 core/alloc） |
| W2.2 Owned storage   | W2.1                 | 中         | `Owned<A>` + 64 字节对齐分配                 |
| W2.3 View storage    | W2.1                 | 中         | `ViewRepr<'a, A>`                            |
| W2.4 ViewMut storage | W2.1                 | 中         | `ViewMutRepr<'a, A>`                         |
| W2.5 Arc storage     | W2.1                 | 高         | `ArcRepr<A>`                                 |
| W2.6 TensorBase      | W2.1-W2.5, W1.3-W1.5 | 高         | 核心结构体                                   |
| W2.7 Type aliases    | W2.6                 | 低         | `Tensor`, `TensorView` 等                    |

### Wave 3: 操作（依赖 Wave 2）

| 任务                     | 依赖       | 预估复杂度 | 产出               |
| ------------------------ | ---------- | ---------- | ------------------ |
| W3.1 Elements iterator   | W2.6       | 中         | 扁平元素迭代       |
| W3.2 Axis iterator       | W2.6       | 中         | 沿轴迭代           |
| W3.3 Math                | W3.1       | 中         | unary, binary, comparison |
| W3.4 Arithmetic          | W3.3       | 中         | Add, Sub, Mul, Div |
| W3.5 Reduction (sum)     | W3.1       | 中         | sum, sum_axis      |
| W3.6 Dot (inner product) | W2.6       | 中         | 向量内积           |
| W3.7 Broadcast           | W2.6       | 高         | 广播规则           |
| W3.8 Transpose           | W2.6       | 中         | transpose          |
| W3.9 Multi-dim index     | W2.6       | 中         | [i, j, k] 索引     |
| W3.10 Slice index        | W2.6       | 高         | 范围切片           |

### Wave 4: 集成（依赖 Wave 3）

| 任务            | 依赖        | 预估复杂度 | 产出                            |
| --------------- | ----------- | ---------- | ------------------------------- |
| W4.1 construct  | W2.6, W3.10 | 中         | zeros, ones, eye, from_vec（非规范便捷层） |
| W4.2 convert    | W2.6        | 中         | cast, to_owned                  |
| W4.3 format     | W2.6        | 低         | Display/Debug                   |
| W4.4 ffi        | W2.6        | 中         | 原始指针 API                    |
| W4.5 workspace  | 无          | 中         | 临时缓冲区                      |
| W4.6 comparison integration/tests | W3.3        | 低         | comparison integration/tests |

### Wave 5: 性能（依赖 Wave 4）

| 任务                | 依赖       | 预估复杂度 | 产出        |
| ------------------- | ---------- | ---------- | ----------- |
| W5.1 parallel dispatch | W3.1-W3.2  | 高         | 纯并行执行后端（不含串行回退） |
| W5.2 par_reduction  | W3.5, W5.1 | 高         | 并行 sum    |
| W5.3 simd math      | W3.3       | 高         | 纯向量化逐元素（不含标量回退） |
| W5.4 simd reduction | W3.5       | 高         | 纯向量化 sum |

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
Wave 3: [W3.1] [W3.2] [W3.7] [W3.8] [W3.9] [W3.10]
            │       │       │       │       │       │       │       │
            └───────┴───────┴───────┴───────┴───────┴───────┴───────┘
                           │
                           ▼
        [W3.3] [W3.4] [W3.5] [W3.6]
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

| 属性     | 值                                                                             |
| -------- | ------------------------------------------------------------------------------ |
| 决策     | 使用单 crate（`xenon`）而非多 crate workspace                                  |
| 理由     | 降低发布复杂度；避免版本协调问题；简化依赖管理                                 |
| 替代方案 | workspace 多 crate（xenon-core, xenon-math, ...） — 放弃，对当前规模过度工程化 |

### 决策 2：F-order 单一布局

| 属性     | 值                                                                            |
| -------- | ----------------------------------------------------------------------------- |
| 决策     | 仅支持列优先（F-order）布局                                                   |
| 理由     | 与 BLAS/LAPACK 兼容；减少布局组合爆炸；简化步长计算（参见 `06-layout.md` §1） |
| 替代方案 | 同时支持 F-order 和 C-order — 放弃，超出范围且增加复杂度                      |

### 决策 3：功能最小化原则

| 属性     | 值                                                                       |
| -------- | ------------------------------------------------------------------------ |
| 决策     | 归约仅 sum、集合仅 unique、形状仅 transpose、索引仅整数+切片、矩阵仅内积 |
| 理由     | 先做精再做广；每个功能确保正确性和性能后再扩展                           |
| 替代方案 | 一开始支持所有 ndarray 功能 — 放弃，范围失控风险高                       |

### 决策 4：依赖层级严格单向

| 属性     | 值                                          |
| -------- | ------------------------------------------- |
| 决策     | 模块依赖严格按 L0→L7 层级单向，禁止循环依赖 |
| 理由     | 确保编译时间可预测；依赖关系清晰可维护      |
| 替代方案 | 允许跨层引用 — 放弃，维护成本高             |

### 决策 5：独立 backend 模块

| 属性     | 值                                                                                                   |
| -------- | ---------------------------------------------------------------------------------------------------- |
| 决策     | `simd/`、`parallel/` 保持为独立顶级后端模块，只提供纯执行能力；执行路径裁决由内部 `dispatch.rs` 统一承担 |
| 理由     | 性能后端是横切关注点，独立模块便于统一 feature gate 与共享分发逻辑；`dispatch.rs` 集中阈值判断、嵌套并行防护和 SIMD 条件检查，避免各语义模块重复实现分支树 |
| 替代方案 | 将后端内嵌到各语义模块 — 放弃，会让性能实现与语义 API 耦合，扩大重复实现；使用独立 kernel 模块承载串行基线 — 改为 `dispatch.rs` + 各模块自含串行实现，减少冗余 |

### 决策 6：dispatch.rs 统一执行路径裁决

| 属性     | 值                                                                                                   |
| -------- | ---------------------------------------------------------------------------------------------------- |
| 决策     | 新增 `dispatch.rs` 内部 helper，统一承载执行路径裁决（`ExecPath`）、嵌套并行防护（`ParallelGuard`）和 SIMD 条件判断 |
| 理由     | 判断归 dispatch，执行归各模块；避免 `parallel/` 和 `simd/` 各自携带串行回退，也避免各语义模块重复实现阈值/SIMD 分支树 |
| 替代方案 | 各语义模块各自实现判断逻辑 — 放弃，会导致阈值行为不一致和代码重复 |

### 决策 7：错误语义集中裁决

| 属性     | 值                                                                                                                               |
| -------- | -------------------------------------------------------------------------------------------------------------------------------- |
| 决策     | 可恢复错误统一通过 `Result` 暴露；公开安全索引入口收敛为 `try_at()` / `try_at_mut()`；`[]` 单独作为受限 panic sugar 说明；FFI 公开入口包含结构化导出 `export()` / `export_mut()` 与 checked 查询 `try_offset_of()` / `try_ptr_at()`，并统一以 checked arithmetic 计算偏移与指针 |
| 理由     | 保持与 `require.md` §18 的安全接口契约一致，避免相同失败条件在公开方法与运算符之间分裂成两套模型，同时让 FFI 结构化导出与偏移/指针查询都遵循相同的可恢复错误约束 |
| 替代方案 | 所有接口统一 panic — 放弃，不利于库集成和诊断；把 `[]` 语法糖当作稳定公开安全 API — 放弃，会与索引失败的可恢复错误契约冲突；为 FFI 额外提供 `offset_of()` / `ptr_at()` 这类 panic-sugar 包装 — 放弃，会破坏公开错误入口的一致性                 |

### 决策 8：FfiError / WorkspaceError 仅限模块内部使用

| 属性     | 值                                                                                                        |
| -------- | --------------------------------------------------------------------------------------------------------- |
| 决策     | `FfiError` 与 `WorkspaceError` 仅作为各自模块内部错误类型；所有面向 Xenon 用户的公开 API 统一暴露 `XenonError` |
| 理由     | 保持公开错误模型单一，同时保留模块内部诊断语义，避免调用方在跨模块组合时处理多套错误语义                        |
| 替代方案 | 将 `FfiError` 或 `WorkspaceError` 直接暴露给公开 API — 放弃，会破坏错误入口的一致性                             |

---

## 错误处理与语义边界

本文档不直接定义错误类型，但要求所有架构层级、模块边界与执行路径统一遵循单一 `XenonError` 公开错误模型；架构层只裁决错误入口应单一、路径语义应一致，不在此重复定义完整错误枚举。`FfiError`、`WorkspaceError` 等模块局部错误只允许在模块内部保留语义，跨公开 API 边界时必须映射为 `00-coding.md` §4.2 定义的结构化 `XenonError` 字段。对于 FFI 场景，公开 Rust 入口包含结构化导出 `export()` / `export_mut()` 与 checked 查询 `try_offset_of()` / `try_ptr_at()`，并统一通过 checked arithmetic 计算偏移与指针。

> **FFI 补充说明**：`extern "C"` 边界不得返回 `Result`，也不得依赖 panic-sugar helper；公开 Rust API 层提供结构化导出 `export()` / `export_mut()` 与 checked 查询 `try_offset_of()` / `try_ptr_at()`，不额外承诺 `offset_of()` / `ptr_at()` 这类 panic 包装，参见 `00-coding.md` §4.2。

---

## 测试计划补充

### Feature gate / 配置测试

| 配置             | 验证点                                           |
| ---------------- | ------------------------------------------------ |
| 默认配置         | 架构分层与模块导出在默认配置下保持成立           |
| `parallel`       | 并行 backend 仅作为可选上层能力接入，不破坏层级  |
| `simd`           | SIMD backend 仅作为可选上层能力接入，不破坏层级  |
| `parallel + simd` | 两类 backend 可组合启用且不引入循环依赖          |

### 类型边界 / 编译期测试

| 场景                       | 测试方式                                   |
| -------------------------- | ------------------------------------------ |
| feature gate 导出边界       | `cargo check` / `cargo test` 配置矩阵验证   |
| 公开模块分层不发生反向依赖 | 编译期模块依赖审查与架构评审检查            |
| 错误类型统一入口            | 通过 `test_error.rs` 与对应 doctest 验证    |

---

## 版本历史

| 版本  | 日期       |
| ----- | ---------- |
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-08 |
| 1.0.2 | 2026-04-08 |
| 1.1.0 | 2026-04-08 |
| 1.2.0 | 2026-04-08 |
| 1.2.1 | 2026-04-10 |
| 1.2.2 | 2026-04-14 |
| 1.2.3 | 2026-04-15 |
| 1.2.4 | 2026-04-15 |
| 1.3.0 | 2026-04-15 |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
