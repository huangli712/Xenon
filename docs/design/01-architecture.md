# 项目总体架构

> 文档编号: 01
> 适用范围: 项目总体架构与跨模块边界
> 任务阶段: Phase 0
> 前置文档: 00-coding.md
> 需求参考: 需求说明书 §1 - §28
> 范围声明: 范围内

---

## 1. 项目概览

### 1.1 定位

Xenon 是一个纯 Rust 实现的 N 维数组（张量）库，定位为科学计算的数值基础设施。设计理念与 Numpy ndarray 层相似，但针对 Rust 生态系统进行了深度优化：类型安全、内存高效、零成本抽象、F-order 单一布局。

### 1.2 目标用户

| 用户类型   | 核心诉求                                             |
| ---------- | ---------------------------------------------------- |
| 库开发者   | 稳定 API、高性能互操作                               |
| 系统开发者 | `std` 环境下的底层内存控制、确定性数值行为、最小依赖 |
| 间接用户   | 性能、正确性、与 Python 经验的直觉一致性             |

### 1.3 核心设计原则

| 原则             | 描述                                                      |
| ---------------- | --------------------------------------------------------- |
| 正确性优先       | 类型安全、内存安全、数值精度满足 IEEE 754                 |
| 零成本抽象       | 视图、索引、元数据访问等核心路径追求零额外抽象成本        |
| 内存可控         | 基于 `std` 显式控制分配和对齐                             |
| 渐进增强         | 核心功能无依赖，并行/SIMD 通过独立 feature gate 按需启用  |
| 错误语义集中裁决 | 可恢复错误统一通过 `Result` 报告                          |
| FFI 友好         | 提供 C 兼容的 F-order 内存布局，便于与 BLAS/LAPACK 互操作 |

### 1.4 工程约束

| 约束       | 要求                                   |
| ---------- | ---------------------------------------|
| Crate 结构 | 单 crate，遵循 SemVer                  |
| MSRV       | Rust 1.85+                             |
| License    | MIT                                    |
| 平台支持   | 当前版本仅支持 `std` 环境              |
| 默认内存序 | F-order（列优先），不支持 C-order      |
| 内存对齐   | 默认建议 64 字节                       |
| 外部依赖   | 仅 rayon（可选并行）+ pulp（可选 SIMD）|

### 1.5 协同基线

本文档作为架构总览，以下游设计文档的已修版本为协同基线；若本文档提及类型、trait、字段名或执行边界，须与这些文档保持一致：

- `26-error.md` v3.0.0：`XenonError` 结构化变体、`ElementType` 类型转换字段、`FfiBackend` 与 workspace/FFI 错误分类。
- `02-dimension.md` v1.x、`03-element.md` v1.x、`04-complex.md` v2.0.0：维度、元素封闭实现集与复数显式构造/运算边界。
- `05-storage.md` v2.0.0、`06-layout.md` v1.3、`07-tensor.md` v2.0.0：存储模式、F-order 布局状态与张量核心类型。
- `08-simd.md` v2.0.0、`09-parallel.md` v2.0.0、`30-dispatch.md` v1.1.1：执行路径、`ParallelGuard`、worker 内 SIMD 与阈值语义。
- `11-math.md` v2.0.0、`12-matrix.md` v2.0.0、`13-reduction.md` v2.0.0、`14-set.md` v2.0.0：数学、矩阵、归约与集合操作命名及 F-order 顺序契约。
- `17-indexing.md`、`18-construction.md`、`19-overload.md`、`21-type.md`、`23-ffi.md`、`24-workspace.md`、`25-safety.md` v2.0.0：索引、构造、运算符、类型转换、FFI、workspace 与线程安全边界。

### 1.6 全局布局不变量

以下布局规则为跨模块统一不变量，所有涉及 shape/stride/layout 的模块设计须遵守：

| 不变量 | 说明 |
|--------|------|
| 拥有型连续存储仅支持 F-order | 列优先布局为唯一合法的拥有型连续存储顺序 |
| 不支持负步长 | 当前版本不支持负步长布局 |
| 零步长仅用于广播只读视图 | 广播产生的零步长布局仅允许出现在只读/共享只读视图上 |
| 转置/切片可产生非连续合法视图 | 转置和切片操作可产生步长非连续的合法视图，该视图共享底层数据 |
| 涉及 ZST 和空数组的操作不得引发 UB | 零大小类型和零元素数组的所有操作不得引发未定义行为 |

---

## 2. 范围

### 2.1 范围内

- N 维数组的存储、构造
- N 维数组的索引操作（多维整数索引、范围切片）
- 形状操作（仅转置）
- 归约操作（仅 sum）
- 集合操作（仅 unique）
- 向量内积（dot）
- 广播操作
- 逐元素运算
- 显式类型转换
- 原始指针 API（FFI）
- 自定义复数类型（Complex<T>）
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

| 类型     | 内容                                                           |
| -------- | -------------------------------------------------------------- |
| 需求映射 | 需求说明书 §1 - §28                                            |
| 范围内   | 总体分层、模块边界、feature gate、依赖约束、错误语义与质量边界 |
| 范围外   | 单模块内部算法、实现细节、额外平台与超范围能力扩展             |
| 非目标   | 通过总体架构文档新增需求未授权的 API、依赖或多 crate 拆分      |

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
│   ├── error.rs               # XenonError enum and Result alias
│   ├── dispatch.rs            # Internal dispatch helper (ExecPath, ParallelExecStrategy, ParallelGuard, ParallelContext, parallel thresholds)
│   │
│   ├── dimension/             # Dimension type system
│   │   ├── mod.rs             # Dimension trait definition
│   │   ├── static.rs          # Ix0, Ix1, ..., Ix6 static dimensions
│   │   ├── dynamic.rs         # IxDyn dynamic dimension
│   │   ├── into.rs            # IntoDimension trait
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
│   │   ├── viewmut.rs         # ViewMutRepr<'a, A> mutable view
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
│   │   ├── mod.rs             # pulp integration, dispatch facade, crate-internal kernel trait
│   │   └── vector.rs          # Vectorized implementation
│   │
│   ├── parallel/              # Parallel backend (feature = "parallel")
│   │   ├── mod.rs             # Module entry, re-exports, pub(crate) ParallelPool internals
│   │   ├── iter.rs            # Internal parallel iteration helpers (pub(crate))
│   │   ├── map.rs             # par_map, par_zip_map (threshold selection is handled by dispatch.rs)
│   │   ├── reduce.rs          # par_reduce_impl, par_sum, par_dot
│   │   └── checked.rs         # par_map_checked and error/panic propagation
│   │
│   ├── broadcast/             # Broadcast rules and read-only views
│   │   ├── mod.rs             # Module entry and re-exports
│   │   ├── shape.rs           # Compatibility and stride rules
│   │   └── view.rs            # broadcast_to() and pub(crate) broadcast_with() internals
│   │
│   ├── math/                  # Element-wise math
│   │   ├── mod.rs             # Module entry and re-exports
│   │   ├── unary.rs           # Unary ops (abs, neg, signum, square, sin, modulus, conj, etc)
│   │   ├── binary.rs          # Binary arithmetic methods (add, sub, mul, div, add_scalar, etc)
│   │   └── comparison.rs      # Comparison ops (equal, not_equal, less, greater)
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
│   │   └── dot.rs             # Vector inner product, may delegate to `simd/` or `parallel/`
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
│   │   ├── ndindex.rs         # NdIndex trait and tuple/slice index implementations
│   │   ├── access.rs          # try_at/try_at_mut and unchecked internals
│   │   └── slice.rs           # SliceInfo, slice, shape/stride updates
│   │
│   ├── construct/             # Tensor construction
│   │   ├── mod.rs             # Module root and re-exports
│   │   ├── init.rs            # zeros, ones
│   │   ├── eye.rs             # eye
│   │   ├── from.rs            # from_shape_vec, from_shape_slice, from_array, from_vec
│   │   └── scalar.rs          # from_scalar
│   │
│   ├── convert/               # Type conversion
│   │   ├── mod.rs             # Module root and re-exports
│   │   └── cast.rs            # Consumes element::CastTo and hosts cast-related implementations
│   │
│   ├── format/                # Formatting output
│   │   ├── mod.rs             # Module root, re-exports, cfg gates
│   │   ├── config.rs          # FormatConfig
│   │   ├── display.rs         # Display implementation
│   │   ├── debug.rs           # Debug implementation
│   │   └── pretty.rs          # Numpy-style formatting helpers
│   │
│   ├── ffi/                   # FFI interface
│   │   ├── mod.rs             # Module root and re-exports
│   │   ├── types.rs           # BlasInfo; re-exports ElementType (from element), FfiErrorCategory (from error)
│   │   ├── ptr.rs             # Raw pointer API (export/export_mut, from_raw_parts, into_raw_parts)
│   │   ├── blas.rs            # BLAS compatibility checks (is_blas_layout_compatible, blas_info, lda)
│   │   └── offset.rs          # Index-to-pointer offset
│   │
│   ├── workspace/             # Temporary workspace
│       ├── mod.rs             # Module root and re-exports
│       ├── workspace.rs       # Workspace struct, constants, construction, destruction
│       ├── borrow.rs          # WorkspaceBorrow, WorkspaceBorrowMut guards
│       ├── split.rs           # SplitBorrowMut guard
│       └── expand.rs          # ensure_capacity and reallocate
│
├── tests/                     # Integration tests
│   ├── common/
│   │   ├── mod.rs             # Shared utility exports
│   │   ├── assertions.rs      # Custom assertion helpers and macros
│   │   └── generators.rs      # Test data generators
│   │
│   ├── test_tensor.rs         # Tensor core functionality (creation/query/type aliases)
│   ├── test_math.rs           # Element-wise operations (arithmetic/math/comparison/logic)
│   ├── test_overload.rs       # Operator overloading (Add/Sub/Mul/Div trait implementations)
│   ├── test_broadcast.rs      # Broadcasting (scalar/vector/matrix broadcasting)
│   ├── test_index.rs          # Indexing operations (multi-dimensional indexing/range slicing)
│   ├── test_construction.rs   # Constructors (zeros/ones/eye/from_shape_vec/...)
│   ├── test_iterator.rs       # Iterators (elements/by-axis/by-index)
│   ├── test_reduction.rs      # Reduction operations (sum/sum along axis)
│   ├── test_matrix.rs         # Vector dot product (dot)
│   ├── test_set.rs            # Set operations (unique)
│   ├── test_shape.rs          # Shape operations (transpose)
│   ├── test_conversion.rs     # Type conversion (cast)
│   ├── test_utility.rs        # Utility operations (fill/clip/to_contiguous)
│   ├── test_output.rs         # NumPy-style formatted output (Display/Debug/truncation)
│   ├── test_ffi.rs            # FFI integration (raw pointers/BLAS compatibility)
│   ├── test_workspace.rs      # Workspace-specific errors and borrow/split/growth
│   ├── test_parallel.rs       # Parallel computation (consistency/data races)
│   ├── test_simd.rs           # SIMD computation (result consistency)
│   ├── test_error.rs          # Error handling (all error types)
│   │
│   ├── compile_fail_tests.rs  # Repository-local compile-fail harness
│   ├── compile-fail/
│   │   ├── wrong_dimension_type.rs
│   │   ├── missing_element_bound.rs
│   │   ├── mismatched_storage_type.rs
│   │   ├── unsigned_tensor_element_rejected.rs
│   │   ├── invalid_unsigned_element_rejected.rs
│   │   ├── ui_bool_sum_rejected.rs
│   │   ├── ui_bool_unique_rejected.rs
│   │   └── ui_bool_arithmetic_rejected.rs
│   │
│   ├── property_tests.rs      # Property-test entry point (integration test target)
│   └── property/
│       ├── tensor_props.rs    # Tensor invariants (transpose involution, unique boundaries, etc.)
│       ├── ops_props.rs       # Operation invariants (commutativity/associativity, etc.)
│       └── shape_props.rs     # Shape invariants (transpose involution, etc.)
│
├── benches/                   # Performance benchmarks
│   ├── utils/
│   │   ├── mod.rs             # Shared constants and utility exports
│   │   └── generators.rs      # Test data generators
│   ├── math.rs                # Element-wise operations
│   ├── reduction.rs           # Reduction operations
│   ├── dot.rs                 # Vector inner product
│   ├── set.rs                 # Set operations
│   ├── broadcast.rs           # Broadcast operations
│   ├── shape.rs               # Shape operations
│   ├── simd_comparison.rs     # SIMD comparison
│   ├── parallel_comparison.rs # Parallel comparison
│   └── construction.rs        # Tensor construction
│
└── examples/                  # Usage examples
    ├── basic.rs               # Basic-operations example
    ├── complex.rs             # Complex-number operations example
    ├── broadcasting.rs        # Broadcasting example
    ├── features.rs            # Optional-feature behavior example (simd/parallel effects)
    ├── simd.rs                # SIMD-acceleration example (requires simd feature)
    ├── ffi.rs                 # FFI integration example
    └── workspace.rs           # Workspace borrow/split/growth example
```

说明：测试文件仅列代表性子集。

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

[[bench]]
name = "math"
harness = false

[[bench]]
name = "reduction"
harness = false

[[bench]]
name = "dot"
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

Xenon 仅支持 `std` 环境；`simd` 与 `parallel` 都建立在该无条件前提之上。

---

## 5. 模块设计

### 5.1 模块速览

| 模块           | 职责                                                                |
| -------------- | ------------------------------------------------------------------- |
| `error.rs`     | `XenonError` 统一错误枚举，`Result<T>` 类型别名                     |
| `dispatch.rs`  | 私有内部执行路径裁决层，不作为公开模块导出 |
| `dimension/`   | `Dimension` trait 和静态/动态维度类型（Ix0-Ix6, IxDyn）             |
| `element/`     | 元素类型 trait 层次（Element → Numeric → RealScalar/ComplexScalar） |
| `complex/`     | 自定义 `Complex<T>` 类型，`#[repr(C)]` 兼容 C FFI                   |
| `storage/`     | 四种存储模式（Owned/ViewRepr/ViewMutRepr/ArcRepr）                  |
| `layout/`      | F-order 布局函数、步长计算、连续性检查与验证入口                    |
| `tensor/`      | 核心 `TensorBase<S, D>` 结构体及类型别名                            |
| `iter/`        | 元素/轴/索引迭代器                                                  |
| `simd/`        | SIMD 后端：向量化 kernel（pulp）、运行时分发，不含标量回退          |
| `parallel/`    | 并行后端：承载纯并行执行入口与内部并行迭代 helper，不含串行回退     |
| `math/`        | 逐元素数学运算（一元、二元算术、`equal`/`not_equal`/`less`/`greater` 比较等） |
| `overload/`     | 运算符重载（Add, Sub, Mul, Div trait 实现）                         |
| `util/`        | 实用操作（clip 裁剪、fill 填充、to_contiguous 连续性保证的公共入口）|
| `set/`         | 集合操作（unique 去重）                                             |
| `broadcast/`   | NumPy 广播规则与只读广播视图构造                                    |
| `matrix/`      | 向量内积 dot，必要时委托 `simd/` 或 `parallel`                      |
| `reduction/`   | 归约操作（sum）                                                     |
| `shape/`       | 转置操作（transpose）                                               |
| `index/`       | 多维整数索引、范围切片索引；公开安全入口为 `try_at` / `try_at_mut`  |
| `construct/`   | 张量构造（`zeros`、`ones`、`eye` 等）                               |
| `convert/`     | 类型转换                                                            |
| `format/`      | Numpy 风格格式化输出                                                |
| `ffi/`         | 原始指针 API 导出、BLAS 兼容性检查、多维索引偏移                    |
| `workspace/`   | 临时工作空间（对齐分配、借用守卫、分割、扩容）                      |

### 5.2 依赖层级

各模块的详细设计参见对应编号文档。层级关系如下：

| 层级   | 模块      | 依赖                                | 参见                |
| ------ | --------- | ----------------------------------- | ------------------- |
| L0     | error     | 无                                  | 26-error.md         |
| L0     | private   | 无                                  |                     |
| L1     | dimension | error, private                      | 02-dimension.md     |
| L1     | complex   | private                             | 04-complex.md       |
| L2     | element   | error, complex                      | 03-element.md       |
| L2     | layout    | error, dimension                    | 06-layout.md        |
| L2     | workspace | std, error                          | 24-workspace.md     |
| L2     | storage   | core, alloc, std, error             | 05-storage.md       |
| L3     | tensor    | storage, dimension, layout, element, error | 07-tensor.md        |
| L4     | broadcast | tensor, dimension, layout, error    | 15-broadcast.md     |
| L4     | iter      | tensor, storage, dimension, error   | 10-iterator.md      |
| L4     | ffi       | tensor, layout, storage, dimension, element, error | 23-ffi.md           |
| L4     | dispatch  | tensor                              | 01-architecture.md  |
| L5     | parallel  | tensor, dimension, element, error, dispatch  | 09-parallel.md      |
| L5     | simd      | tensor, layout, element             | 08-simd.md          |
| L5     | math      | tensor, broadcast, element, iter    | 11-math.md          |
| L5     | set       | tensor, element, complex, iter      | 14-set.md           |
| L5     | matrix    | tensor, element                     | 12-matrix.md        |
| L5     | reduction | tensor, dimension, element, error   | 13-reduction.md     |
| L5     | shape     | tensor, dimension, layout           | 16-shape.md         |
| L5     | index     | tensor, dimension, layout, error    | 17-indexing.md      |
| L5     | util      | tensor, dimension, storage, layout, iter | 20-utility.md  |
| L5     | construct | tensor, storage, layout, dimension, element | 18-construction.md |
| L5     | format    | tensor, storage, dimension, element | 22-output.md        |
| L5     | convert   | tensor, element                     | 21-type.md          |
| L6     | overload  | tensor, broadcast, math, dimension, element | 19-overload.md      |

### 5.2a 内部 Helper 函数清单

以下为跨模块引用的 `pub(crate)` 内部 helper 函数，统一在此记录其位置与职责：

| 函数                                | 定义位置               | 职责                                  | 引用模块                |
| ----------------------------------- | ---------------------- | ------------------------------------- | ----------------------- |
| `validate_access_range()`           | `src/tensor/` 内部      | 校验 raw parts 构造的存储边界         | 07-tensor, 23-ffi       |
| `validate_non_overlapping_layout()` | `src/tensor/` 内部      | 保守验证非重叠可变布局               | 07-tensor, 23-ffi       |
| `compute_safe_chunks()`             | `src/parallel/mod.rs`   | 将 [0,total) 划分为非重叠区间         | 25-safety               |
| `util_internal_to_f_contiguous()`   | `src/util/` 内部     | 将张量连续化为 canonical F-order     | 20-utility              |
| `fill_storage_mut()`                | `src/util/` 内部     | 通过 StorageMut 填充后备缓冲区       | 20-utility              |
| `fill_try_dispatch()`               | `src/util/` 内部     | try_fill 的错误感知分派              | 20-utility              |

### 5.3 依赖图（ASCII）

```
L0  ┌───────────┐  ┌───────────┐
    │   error   │  │  private  │
    └───────────┘  └───────────┘
          │
          ▼
L1  ┌───────────┐  ┌───────────┐
    │ dimension │  │  complex  │
    └───────────┘  └───────────┘
          │
          ▼
L2  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐
    │  element  │  │   layout  │  │ workspace │  │  storage  │
    └───────────┘  └───────────┘  └───────────┘  └───────────┘
          │
          ▼
L3  ┌───────────────────────────────────────────┐
    │                  tensor                   │
    └───────────────────────────────────────────┘
          │
          ▼
L4  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐
    │ broadcast │  │   iter    │  │    ffi    │  │  dispatch │
    └───────────┘  └───────────┘  └───────────┘  └───────────┘
          │
          ▼
L5  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ parallel │  │   simd   │  │   math   │  │   set    │  │  matrix  │  │ reduction│
    ├──────────┤  ├──────────┤  ├──────────┤  ├──────────┤  ├──────────┤  ├──────────┤
    │  shape   │  │  index   │  │   util   │  │construct │  │  format  │  │ convert  │
    └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘
          │
          ▼
L6  ┌───────────┐
    │ overload  │
    └───────────┘
```

### 5.4 新增依赖说明

仅 `rayon` 与 `pulp` 作为可选依赖, 核心能力优先使用标准库与 crate 内部模块。

---

## 6. Feature Gate 矩阵

| 组合    | 命令                       | 适用场景                 |
| ------- | -------------------------- | ------------------------ |
| 标准    | （默认）                   | 桌面应用、CLI 工具       |
| 仅并行  | `--features parallel`      | 仅并行加速               |
| 高性能  | `--features parallel,simd` | 数据科学、机器学习       |
| 仅 SIMD | `--features simd`          | 需 SIMD 但无需并行的场景 |

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

---

## 7. prelude.rs 导出清单

以下为当前实现组织建议，不属于 `需求说明书 §27` 的稳定 API 承诺。

```rust,ignore
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

// Construction convenience helpers delegating to TensorBase inherent methods
pub use crate::construct::{
    zeros, ones, eye,
    from_shape_vec,
};
```

---

## 8. lib.rs 模块结构

```rust,ignore
// src/lib.rs

//! # Xenon
//!
//! A Rust N-dimensional array library for scientific computing.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
#![warn(rust_2024_compatibility)]
#![warn(unsafe_op_in_unsafe_fn)]
#![warn(clippy::unwrap_used)]

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
pub(crate) mod simd;

// `simd` remains a feature-gated internal backend module; concrete SIMD
// traits, kernels, and ISA detection details stay `pub(crate)` or
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

本片段须与 `00-coding.md` 编码规范中的 lint 配置保持一致。

---

## 9. API 稳定性说明

| 层级             | 稳定性 | 说明                                             |
| ---------------- | ------ | ------------------------------------------------ |
| `prelude::*`     | 稳定   | 主版本号内保持兼容                               |
| 公开 trait 方法  | 稳定   | 只增不减                                         |
| `private`        | 不稳定 | 内部模块，随时可能变更                           |
| `dispatch`       | 不稳定 | 内部模块，随时可能变更                           |
| `#[doc(hidden)]` | 不稳定 | 仅供内部使用                                     |
| `simd`           | 不稳定 | SIMD 加速对用户透明，但是`simd` 非公开模块       |
| `parallel`       | 不稳定 | 并行加速对用户透明，但是`parallel`非公开模块 API |

---

## 10. 重点 API 暴露方式

Xenon 的公开 API 以 `TensorBase` 固有方法为主。以下按类别列出所有公开方法及对应模块文档。

### 10.1 逐元素数学（参见 `11-math.md`）

| 方法 | 暴露方式 | 说明 |
| ---- | -------- | ---- |
| `add` | `TensorBase` 固有方法 | 元素级加法（另可通过 `+` 运算符；参见 `19-overload.md`） |
| `sub` | `TensorBase` 固有方法 | 元素级减法（另可通过 `-` 运算符） |
| `mul` | `TensorBase` 固有方法 | 元素级乘法（另可通过 `*` 运算符） |
| `div` | `TensorBase` 固有方法 | 元素级除法（另可通过 `/` 运算符） |
| `neg` | `TensorBase` 固有方法 | 逐元素取反（另可通过 `-` 一元运算符） |
| `abs` | `TensorBase` 固有方法 | 逐元素绝对值 |
| `square` | `TensorBase` 固有方法 | 逐元素平方 |
| `signum` | `TensorBase` 固有方法 | 逐元素符号 |
| `sin` | `TensorBase` 固有方法 | 逐元素正弦 |
| `sqrt` | `TensorBase` 固有方法 | 逐元素平方根 |
| `exp` | `TensorBase` 固有方法 | 逐元素指数 |
| `ln` | `TensorBase` 固有方法 | 逐元素自然对数 |
| `floor` | `TensorBase` 固有方法 | 逐元素向下取整 |
| `ceil` | `TensorBase` 固有方法 | 逐元素向上取整 |
| `modulus` | `TensorBase` 固有方法 | 逐元素取模 |
| `conjugate` | `TensorBase` 固有方法 | 逐元素共轭（实数类型为恒等操作；参见 `04-complex.md`） |

### 10.2 逐元素比较（参见 `11-math.md`）

| 方法 | 暴露方式 | 说明 |
| ---- | -------- | ---- |
| `equal` | `TensorBase` 固有方法 | 逐元素等于比较（返回 `Tensor<bool, D>`） |
| `not_equal` | `TensorBase` 固有方法 | 逐元素不等于比较 |
| `less` | `TensorBase` 固有方法 | 逐元素小于比较 |
| `greater` | `TensorBase` 固有方法 | 逐元素大于比较 |

### 10.3 布尔操作（参见 `11-math.md`）

| 方法 | 暴露方式 | 说明 |
| ---- | -------- | ---- |
| `not` | `TensorBase` 固有方法 | 仅 `bool` 张量可用，逐元素逻辑非 |

### 10.4 标量算术（参见 `11-math.md`）

| 方法 | 暴露方式 | 说明 |
| ---- | -------- | ---- |
| `add_scalar` | `TensorBase` 固有方法 | 张量 + 标量；运算符路径使用 `Scalar<A>` 包装类型 |
| `sub_scalar` | `TensorBase` 固有方法 | 张量 - 标量 |
| `mul_scalar` | `TensorBase` 固有方法 | 张量 * 标量 |
| `div_scalar` | `TensorBase` 固有方法 | 张量 / 标量 |

### 10.5 标量比较（参见 `11-math.md`）

| 方法 | 暴露方式 | 说明 |
| ---- | -------- | ---- |
| `equal_scalar` | `TensorBase` 固有方法 | 逐元素与标量比较相等 |
| `not_equal_scalar` | `TensorBase` 固有方法 | 逐元素与标量比较不等 |
| `less_scalar` | `TensorBase` 固有方法 | 逐元素与标量比较小于 |
| `greater_scalar` | `TensorBase` 固有方法 | 逐元素与标量比较大于 |

### 10.6 归约（参见 `13-reduction.md`）

| 方法 | 暴露方式 | 说明 |
| ---- | -------- | ---- |
| `sum` | `TensorBase` 固有方法 | 归约语义由张量实例直接触发 |
| `sum_axis` | `TensorBase` 固有方法 | 沿指定轴归约并移除该轴（要求 `D: RemoveAxis`） |
| `sum_axis_keepdims` | `TensorBase` 固有方法 | 沿指定轴归约并保留长度为 1 的轴 |

### 10.7 形状变换（参见 `16-shape.md`）

| 方法 | 暴露方式 | 说明 |
| ---- | -------- | ---- |
| `transpose` | `TensorBase` 固有方法 | 形状变换直接挂载在张量实例上 |
| `broadcast_to` | `TensorBase` 固有方法 | 广播视图构造由张量实例发起 |

### 10.8 逐元素实用操作

| 方法 | 暴露方式 | 说明 |
| ---- | -------- | ---- |
| `clip` | `TensorBase` 固有方法 | 逐元素裁剪作为张量实用操作暴露 |
| `fill` | `TensorBase` 固有方法 | 仅 `TensorViewMut` / 可变存储路径可调用 |
| `try_fill` | `TensorBase` 固有方法 | 带错误检查的填充操作（参见 `20-utility.md`） |

### 10.9 类型转换（参见 `21-type.md`）

| 方法 | 暴露方式 | 说明 |
| ---- | -------- | ---- |
| `cast` | `TensorBase` 固有方法 | 类型转换保持实例方法风格 |
| `to_owned` | `TensorBase` 固有方法 | 从视图/引用创建拥有所有权的副本 |
| `into_owned` | `TensorBase` 固有方法 | 消费自身并返回拥有所有权的副本 |

### 10.10 集合与迭代

| 方法 | 暴露方式 | 说明 |
| ---- | -------- | ---- |
| `unique` | `TensorBase` 固有方法 | 集合操作直接从张量实例触发 |
| `iter` / `axis_iter` | `TensorBase` 固有方法 | 迭代器入口保持实例方法风格 |

### 10.11 输出（参见 `22-output.md`）

| 方法 | 暴露方式 | 说明 |
| ---- | -------- | ---- |
| `display_with` | `TensorBase` 固有方法 | 带格式化选项的张量显示（精度、宽度等） |

### 10.12 FFI（参见 `23-ffi.md`）

| 方法 | 暴露方式 | 说明 |
| ---- | -------- | ---- |
| `export` | `TensorBase` 固有方法 | 导出不可变底层缓冲区指针 |
| `export_mut` | `TensorBase` 固有方法 | 导出可变底层缓冲区指针 |
| `blas_info` | `TensorBase` 固有方法 | 返回 BLAS 兼容的布局元数据 |
| `lda` | `TensorBase` 固有方法 | 返回行主维数（leading dimension） |
| `try_offset_of` | `TensorBase` 固有方法 | 安全计算多维索引的线性偏移 |
| `try_ptr_at` | `TensorBase` 固有方法 | 获取指定位置元素的原始指针 |

### 10.13 连续性与内存管理（参见 `20-utility.md`）

| 方法 | 暴露方式 | 说明 |
| ---- | -------- | ---- |
| `to_contiguous` | `TensorBase` 固有方法 | 创建连续布局的副本（不消费自身） |
| `into_contiguous` | `TensorBase` 固有方法 | 消费自身并确保连续布局 |

### 10.14 双入口 API

| API | 暴露方式 | 说明 |
| --- | -------- | ---- |
| `dot` | 双入口 | 自由函数 + `TensorBase` 固有方法；位于 `matrix` 模块 |

### 10.15 补充说明

- `parallel` / `simd` 仅影响这些公开 API 的内部执行路径，不额外暴露稳定的并行或 SIMD 用户侧入口。
- `dispatch.rs`、`prelude` 中的重导出布局，属于当前实现组织建议。
- `construct` 模块自由函数/便捷包装的存在方式，属于当前实现组织建议。
- 构造操作统一通过 `TensorBase` 固有方法暴露，`prelude` 仅重导出少量委托到这些固有方法的便捷自由函数。
- 面向最终用户的规范化构造入口以 `TensorBase`/类型别名上的固有方法为主。
- `construct` 模块中的自由函数只作为薄包装或预导出便捷层，不单独扩展第二套构造语义。

---

## 11. 核心类型速查

各类型的详细设计参见对应模块文档（`02-dimension.md`、`03-element.md`、`05-storage.md`、`06-layout.md`）。

```rust,ignore
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
compute_f_strides(shape)             // Compute canonical F-order strides
compute_layout_flags(shape, strides, ptr) // Central function for all layout flags
LayoutState                          // FContiguous / NonContiguous / BroadcastView

// Tensor semantic query enums
StorageKind                          // Owned / View / ViewMut / Arc
AccessSemantics                      // ReadOnly / SharedReadOnly / Writable / Owned
DataLocation                         // Cpu (current version only supports CPU)

// Element trait hierarchy
Element                        // Base: Copy + Sealed with const ELEMENT_TYPE: ElementType
└── Numeric                    // Numeric: arithmetic syntax + checked integer contract + conjugate semantics
    ├── RealScalar             // Real: sqrt, sin, exp, ln, floor, ceil
    └── ComplexScalar          // Complex: complex-specific modulus/re/im helpers; conjugation is unified by `Numeric::conjugate()`
```

| 名称                    | 分类               | 稳定性说明                       |
| ----------------------- | ------------------ | -------------------------------- |
| `BroadcastDim`          | 公开 sealed trait  | 允许命名但禁止外部实现           |
| `PermuteAxes`           | 模块内部辅助 trait | 非稳定公开面；供转置实现内部辅助 |
| `BoolElement`           | 模块内部辅助 trait | 非稳定公开面；布尔专用 helper    |
| `CheckedAdd`            | 模块内部辅助 trait | 非稳定公开面；整数 checked 原语（element 层权威定义） |
| `CheckedSub`            | 模块内部辅助 trait | 非稳定公开面；整数 checked 原语（element 层权威定义） |
| `CheckedMul`            | 模块内部辅助 trait | 非稳定公开面；整数 checked 原语（element 层权威定义） |
| `CheckedNeg`            | 模块内部辅助 trait | 非稳定公开面；整数 checked 原语（element 层权威定义） |
| `CheckedDiv`            | 模块内部辅助 trait | 非稳定公开面；整数 checked 原语（element 层权威定义） |
| `CastTo<T>`             | 公开 trait         | 受 `convert/` 模块消费的显式转换契约 |
| `OrderedCompareElement` | 公开 sealed trait  | 出现在有序比较相关的公开签名中；允许命名但禁止外部实现 |

上述公开元素能力 trait（`Element`、`Numeric`、`RealScalar`、`ComplexScalar`、`CastTo`）均通过 `private::Sealed` 实现 sealed trait 模式，禁止下游 crate 自行实现。其中 `Numeric` 不仅表示 `Add + Sub + Mul + Div + Neg` 语法可用，还要求：

- 对整数路径，具体运算模块必须落实 checked overflow / divide-by-zero / unrepresentable-result contract；
- `ElementType` 的公开取值为 `I32`、`I64`、`F32`、`F64`、`Complex32`、`Complex64`、`Bool`；封闭实现集为 `i32`、`i64`、`f32`、`f64`、`Complex<f32>`、`Complex<f64>`、`bool`；
- 对实数类型，`conjugate(self)` 为恒等；对复数类型，`conjugate(self)` 执行数学共轭；`Complex<T>` 的显式实数构造路径仅为 `From<T> for Complex<T>`，不提供 `Complex<T> op T` 便捷运算符；
- 统一错误入口与结构化字段约束遵循 `26-error.md`，不得在架构层引入第二套公开错误模型。

---

## 12. 实现任务分解

### Wave 1: 基础设施（可完全并行）

| 任务                   | 依赖       | 预估复杂度 | 产出                               |
| ---------------------- | ---------- | ---------- | ---------------------------------- |
| W1.1 error types       | 无         | 低         | `XenonError`, `Result<T>`          |
| W1.2 private module    | 无         | 低         | `Sealed` trait                     |
| W1.3 dimension traits  | W1.1       | 中         | `Dimension`, `IntoDimension`       |
| W1.4 static dimensions | W1.3       | 中         | `Ix0`-`Ix6`                        |
| W1.5 dynamic dimension | W1.3       | 中         | `IxDyn`                            |
| W1.6 Complex<T>        | W1.2       | 高         | 自定义复数类型                     |
| W1.7 element traits    | W1.6, W1.1 | 中         | `Element`, `Numeric`, `RealScalar` |
| W1.8 layout helpers    | W1.1, W1.3 | 低         | 模块级布局函数与判定入口           |
| W1.9 F-order strides   | W1.1, W1.3 | 中         | F-order 步长计算                   |

### Wave 2: 核心（依赖 Wave 1）

| 任务                 | 依赖                 | 预估复杂度 | 产出                          |
| -------------------- | -------------------- | ---------- | ----------------------------- |
| W2.1 Storage trait   | W1.1                 | 高         | `Storage`, `RawStorage`       |
| W2.2 Owned storage   | W2.1                 | 中         | `Owned<A>` + 64 字节对齐分配  |
| W2.3 View storage    | W2.1                 | 中         | `ViewRepr<'a, A>`             |
| W2.4 ViewMut storage | W2.1                 | 中         | `ViewMutRepr<'a, A>`          |
| W2.5 Arc storage     | W2.1                 | 高         | `ArcRepr<A>`                  |
| W2.6 TensorBase      | W2.1-W2.5, W1.3-W1.9 | 高         | 核心结构体                    |
| W2.7 Type aliases    | W2.6                 | 低         | `Tensor`, `TensorView` 等     |
| W2.8 Workspace       | W1.1                 | 中         | 临时缓冲区                    |

### Wave 3: 操作（依赖 Wave 2）

| 任务                  | 依赖       | 预估复杂度 | 产出               |
| --------------------- | ---------- | ---------- | ------------------ |
| W3.1 Iterator         | W2.6       | 中         | 扁平元素迭代       |
| W3.2 Axis iterator    | W2.6       | 中         | 沿轴迭代           |
| W3.3 Broadcast        | W2.6       | 高         | 广播规则           |
| W3.4 Math             | W3.1, W1.7, W3.3 | 中   | unary, binary, comparison |
| W3.5 Arithmetic       | W3.3, W3.4 | 中         | Add, Sub, Mul, Div |
| W3.6 Reduction (sum)  | W3.1       | 中         | sum                |
| W3.7 Dot              | W2.6       | 中         | 向量内积           |
| W3.8 Transpose        | W2.6       | 中         | transpose          |
| W3.9 Multi-dim index  | W2.6       | 中         | [i, j, k] 索引     |
| W3.10 Slice index     | W2.6       | 高         | 范围切片           |
| W3.11 Set             | W2.6, W3.1 | 高         | unique             |
| W3.12 Util            | W2.6, W3.1 | 高         | clip, fill         |

### Wave 4: 集成（依赖 Wave 3）

| 任务            | 依赖  | 预估复杂度 | 产出                        |
| --------------- | ----- | ---------- | --------------------------- |
| W4.1 Construct  | W2.6  | 中         | zeros, ones, eye, from_vec  |
| W4.2 Convert    | W2.6  | 中         | cast, to_owned              |
| W4.3 Format     | W2.6  | 低         | Display/Debug               |
| W4.4 Ffi        | W2.6  | 中         | 原始指针 API                |
| W4.5 Comparison integration/tests | W3.4   | 低 | comparison integration/tests |

### Wave 5: 性能（依赖 Wave 4）

| 任务              | 依赖       | 预估复杂度 | 产出           |
| ------------------| ---------- | ---------- | -------------- |
| W5.1 Dispatch     | W2.6       | 高         | 纯并行执行后端 |
| W5.2 Parallel     | W3.1, W3.2 | 高         | 纯并行执行后端 |
| W5.3 parallel sum | W3.6, W5.2 | 高         | 并行 sum       |
| W5.4 SIMD math    | W3.4       | 高         | 纯向量化逐元素 |
| W5.5 SIMD sum     | W3.6       | 高         | 纯向量化 sum   |

---

## 13. 设计决策记录

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
| 决策     | 模块依赖严格按 L0→L6 层级单向，禁止循环依赖 |
| 理由     | 确保编译时间可预测；依赖关系清晰可维护      |
| 替代方案 | 允许跨层引用 — 放弃，维护成本高             |

### 决策 5：独立 backend 模块

| 属性     | 值                                                                                                   |
| -------- | ---------------------------------------------------------------------------------------------------- |
| 决策     | `simd/`、`parallel/` 保持为独立顶级后端模块，只提供纯执行能力；执行路径裁决由内部 `dispatch.rs` 统一承担 |
| 理由     | 性能后端是横切关注点，独立模块便于统一 feature gate 与共享分发逻辑；`dispatch.rs` 集中并行阈值判断与嵌套并行防护，避免各语义模块重复实现并行分支树，而 SIMD 细节保持在 `simd/` 内部 |
| 替代方案 | 将后端内嵌到各语义模块 — 放弃，会让性能实现与语义 API 耦合，扩大重复实现；使用独立 kernel 模块承载串行基线 — 改为 `dispatch.rs` + 各模块自含串行实现，减少冗余 |

### 决策 6：dispatch.rs 三路 ExecPath 裁决模型

| 属性     | 值                                                                                                   |
| -------- | ---------------------------------------------------------------------------------------------------- |
| 决策     | dispatch.rs 通过 `select_exec_path` 返回 `(ExecPath, Option<ParallelGuard>)`，统一指示执行路径：`Serial` / `Simd` / `Parallel` |
| 理由     | 集中式三路裁决避免消费者模块（math/matrix/reduction）各自重复实现路径选择树；同时保持 simd/ 后端对其内部细化（ISA、lane 宽度、对齐细节）的最终准入权，并通过 guard 显式传递并行进入状态 |
| 替代方案 | 二元 ExecPath（Serial/Parallel）+ SIMD 在 Serial 分支内部隐式裁决 — 放弃，导致消费者代码中需要嵌套裁决，且 dispatch 无法在 Simd 与 Serial 之间做相同精度的阈值差异化 |

#### 三路裁决模型

| 路径 | 触发条件 | 后端 |
|------|----------|------|
| `ExecPath::Serial` | 默认回退；len 低于所有阈值，或 feature 禁用，或 `ParallelGuard::enter()` 失败 | 消费者模块自身的串行实现 |
| `ExecPath::Simd` | feature = "simd" 启用 + len ≥ simd_threshold + 连续 + 对齐前提满足 + 不进入并行路径 | `simd/` 后端（`dispatch_vector_binary_op` 以 `bool` 报告是否接管执行；失败时由 dispatch 消费方回退） |
| `ExecPath::Parallel` | feature = "parallel" 启用 + len ≥ parallel_threshold + `ParallelGuard::enter()` 成功 | `parallel/` 后端；并行 worker 内部可在 `_guard: ParallelGuard` 保护下调用 SIMD |

#### 职责边界

- **dispatch.rs**: 仅做 ExecPath 三路裁决；阈值计算使用 `saturating_mul`，threshold = 0 为禁用 sentinel；不参与 ISA 检测、不参与 SIMD lane 选择、不参与对齐细节判断
- **simd/**: 在被 dispatch 选中（ExecPath::Simd 返回）后，内部决定是否最终启用 SIMD（ISA 可用性、lane 宽度、对齐 fast path）；`SimdElement` 仍为 sealed trait，整数仲裁失败回退归 dispatch 消费方处理
- **parallel/**: 在被 dispatch 选中（ExecPath::Parallel 返回）后执行；worker 内部可结合 SIMD 形成 thread × SIMD 双层加速
- **pulp::Arch**: ISA 检测（AVX-512 -> AVX2 -> SSE4.1 -> NEON）的唯一权威，仅在 simd/ 内部使用

详见 30-dispatch.md（dispatch 模块设计文档）和 08-simd.md（SIMD 后端设计文档）。

### 决策 7：错误语义集中裁决

| 属性     | 值                                                                                                                               |
| -------- | -------------------------------------------------------------------------------------------------------------------------------- |
| 决策     | 可恢复错误统一通过 `Result` 暴露；公开安全索引入口收敛为 `try_at()` / `try_at_mut()`，不提供方括号索引 trait 实现；FFI 公开入口包含结构化导出 `export()` / `export_mut()` 与 checked 查询 `try_offset_of()` / `try_ptr_at()`，并统一以 checked arithmetic 计算偏移与指针 |
| 理由     | 保持与 `需求说明书 §18` 的安全接口契约一致，避免相同失败条件在公开方法与运算符之间分裂成两套模型，同时让 FFI 结构化导出与偏移/指针查询都遵循相同的可恢复错误约束 |
| 替代方案 | 所有接口统一 panic — 放弃，不利于库集成和诊断；实现 `[]` 语法糖 — 放弃，会与索引失败的可恢复错误契约冲突；为 FFI 额外提供 `offset_of()` / `ptr_at()` 这类 panic-sugar 包装 — 放弃，会破坏公开错误入口的一致性                 |

### 决策 8：所有可恢复错误统一使用 XenonError 结构化变体

| 属性     | 值                                                                                                        |
| -------- | --------------------------------------------------------------------------------------------------------- |
| 决策     | 所有可恢复错误直接以 `XenonError` 结构化变体返回，不再定义或保留模块内部错误类型作为公开边界前的中间层 |
| 理由     | 消除模块内部错误类型，所有错误直接以 `XenonError` 结构化变体返回，满足需求说明书 §27 对公开结构化诊断的要求 |
| 替代方案 | 保留模块内部专用错误类型作为额外中间层 — 放弃，引入不必要的映射层且与结构化诊断要求存在张力 |

---

## 14. 错误处理与语义边界

- 本文档不直接定义错误类型，但要求所有架构层级、模块边界与执行路径统一遵循单一 `XenonError` 公开错误模型。
- 架构层只裁决错误入口应单一、路径语义应一致，不在此重复定义完整错误枚举。
- 所有可恢复错误直接以 `XenonError` 结构化变体返回，不使用模块内部错误类型。
- 规范错误模型的 canonical source 以 `26-error.md` 为准。
- 所有 `operation` 字段使用 `Cow<'static, str>`；类型转换错误使用 `source_type: ElementType` / `target_type: ElementType` 与 `ConversionFailureReason`，不使用运行时类型标识。
- Workspace 错误使用 `WorkspaceErrorCategory` 七子变体的结构化负载；借用冲突为 `BorrowConflict { requested, current }`。
- FFI 错误使用 `Ffi { operation, category, backend, cause }` 四字段模型；`FfiErrorCategory` 含八个子变体，`FfiBackend` 为 `RawParts` / `Blas`。
- 对于 FFI 场景，公开 Rust 入口包含结构化导出 `export()` / `export_mut()` 与 checked 查询 `try_offset_of()` / `try_ptr_at()`，并统一通过 checked arithmetic 计算偏移与指针。

---

## 15. 验证与落地方式

本节汇总架构层的验证入口、配置矩阵与落地检查方式，作为跨模块一致性的统一核对清单。其中发布前配置矩阵属于非规范性工程治理清单，不替代 `需求说明书 §28` 的强制要求。

### 15.1 Feature gate / 配置测试

| 配置              | 验证点                                           |
| ----------------- | ------------------------------------------------ |
| 默认配置          | 架构分层与模块导出在默认配置下保持成立           |
| `parallel`        | 并行 backend 仅作为 L5 可选后端接入，不破坏层级  |
| `simd`            | SIMD backend 仅作为可选上层能力接入，不破坏层级  |
| `parallel + simd` | 两类 backend 可组合启用且不引入循环依赖          |

### 15.2 类型边界 / 编译期测试

| 场景                       | 测试方式                                  |
| -------------------------- | ----------------------------------------- |
| feature gate 导出边界      | `cargo check` / `cargo test` 配置矩阵验证 |
| 公开模块分层不发生反向依赖 | 编译期模块依赖审查与架构评审检查          |
| 错误类型统一入口           | 通过 `test_error.rs` 与对应 doctest 验证  |

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
| 1.3.1 | 2026-04-16 |
| 2.0.0 | 2026-05-03 |

### v2.0.0

改动清单：

- 新增 §1 协同基线，明确本文档与已修下游设计文档版本的对齐关系。
- 对齐 `26-error.md` v3.0.0：架构层错误边界补充 `Cow<'static, str>`、`ElementType`、`WorkspaceErrorCategory`、`FfiErrorCategory` 与 `FfiBackend` 约束。
- 对齐 `17-indexing.md` v2.0.0：公开安全索引入口收敛为 `try_at` / `try_at_mut`，不提供方括号索引 trait 实现。
- 对齐 `11-math.md` 与 `19-overload.md`：比较方法命名改为 `equal` / `not_equal` / `less` / `greater`，标量运算符路径引用 `Scalar<A>` 包装类型。
- 对齐 `03-element.md`、`04-complex.md`、`05-storage.md`：核心类型速查更新 `Element`、`ElementType`、封闭实现集、`Complex<T>` 实数构造边界与 `StorageKind::Arc` 命名。
- 对齐 `08-simd.md`、`09-parallel.md`、`30-dispatch.md`：dispatch 决策补充 `(ExecPath, Option<ParallelGuard>)`、threshold = 0 sentinel、`saturating_mul` 与 worker 内 SIMD 的双层加速边界。

未变更项：

- 单 crate 架构、11 模块设计边界、L0-L6 分层和依赖图保持不变。
- F-order 单一布局、广播零步长只读视图、`LayoutState` 三状态边界保持不变。
- Feature gate 矩阵、公开模块导出策略、prelude 组织建议保持不变。
- 归约仅 sum、集合仅 unique、形状仅 transpose、矩阵仅 dot 的范围决策保持不变。
- `simd/`、`parallel/` 作为独立可选后端，`dispatch.rs` 作为内部裁决层的架构边界保持不变。

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
