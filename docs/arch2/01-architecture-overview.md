# Renon 项目整体架构设计

> 本文档定义 Renon 的整体架构，包括目录结构、模块层次、依赖关系和实施阶段。

---

## 1. 项目概览

Renon 是一个 Rust 多维数组（张量）库，定位为科学计算的数值基础设施。核心抽象为双参数泛型 `TensorBase<S, D>`，其中 S 决定存储模式（所有权/视图），D 决定维度类型（静态/动态）。

### 设计原则

| 原则 | 体现 |
|------|------|
| 正确性优先 | 类型系统区分只读/可写/拥有，编译时保证访问安全 |
| 清晰抽象 | 双参数泛型，存储与维度正交 |
| 零开销 | 泛型单态化，视图零拷贝，SIMD/并行可选 |
| 科学计算导向 | 列优先默认，64 字节对齐，BLAS 兼容布局 |
| 最小依赖 | 仅 rayon（可选）和 pulp（可选） |
| 可扩展性 | Storage trait 预留 Device 关联类型 |

---

## 2. 目录结构

```
Renon/
├── Cargo.toml                  # crate 配置
├── rustfmt.toml                # 格式化配置
├── clippy.toml                 # lint 配置
├── README.md
├── CHANGELOG.md
├── LICENSE                     # MIT
├── docs/                       # 设计文档
│   ├── 00-rust-standards.md
│   ├── 01-architecture-overview.md
│   ├── 02-dimension.md
│   ├── ...
│   └── 24-documentation.md
├── src/
│   ├── lib.rs                  # crate 入口，re-export 公共 API
│   │
│   ├── # === 核心模块（Phase 2）===
│   ├── dimension.rs            # Ix0~Ix6, IxDyn, Dimension trait
│   ├── element.rs              # Element, Numeric, RealScalar, ComplexScalar trait
│   ├── complex.rs              # Complex<T> 类型
│   ├── storage/                # 存储系统
│   │   ├── mod.rs              # Storage trait, RawStorage trait
│   │   ├── owned.rs            # Owned<A>
│   │   ├── view.rs             # ViewRepr<&'a A>
│   │   ├── view_mut.rs         # ViewMutRepr<&'a mut A>
│   │   └── arc.rs              # ArcRepr<A>
│   ├── layout.rs               # 步长计算, LayoutFlags, 对齐策略
│   ├── tensor.rs               # TensorBase<S, D> 核心, 类型别名
│   ├── error.rs                # TensorError, Result 别名
│   │
│   ├── # === API 模块（Phase 3，可并行）===
│   ├── construction.rs         # zeros/ones/full/empty/eye/identity/diag/arange/linspace/logspace
│   ├── conversion.rs           # cast, From impls, to_owned, 运算符重载
│   ├── iter/                   # 迭代器
│   │   ├── mod.rs              # 公共 trait + re-export
│   │   ├── elements.rs         # 元素迭代器
│   │   ├── axis.rs             # 轴迭代器
│   │   ├── windows.rs          # 窗口迭代器
│   │   ├── indexed.rs          # 索引迭代器
│   │   └── zip.rs              # 多数组同步迭代
│   ├── broadcast.rs            # 广播机制
│   ├── indexing.rs             # 索引操作, s![] 宏, take/put/where
│   ├── shape/                  # 形状操作
│   │   ├── mod.rs
│   │   ├── reshape.rs
│   │   ├── transpose.rs
│   │   ├── slice.rs
│   │   ├── squeeze.rs
│   │   ├── split.rs            # split, chunk, unstack
│   │   ├── pad.rs
│   │   └── repeat.rs           # repeat, tile
│   ├── ops/                    # 运算操作
│   │   ├── mod.rs              # 公共 trait + dispatch
│   │   ├── element_wise.rs     # 逐元素运算 (add/sub/mul/div, 三角, 指数/对数)
│   │   ├── matrix.rs           # 矩阵运算 (matvec, dot, outer, batch)
│   │   ├── reduction.rs        # 归约 (sum/prod/min/max/mean/var/std/argmin/argmax)
│   │   ├── accumulate.rs       # 累积归约 (cumsum/cumprod)
│   │   └── set_ops.rs          # 集合操作 (unique/bincount/histogram)
│   ├── ffi.rs                  # 指针 API, BLAS 兼容性
│   ├── workspace.rs            # 临时工作空间, ScratchNeed
│   │
│   ├── # === 后端模块（Phase 4）===
│   ├── backend/
│   │   ├── mod.rs              # 后端抽象, 性能分层 dispatch
│   │   ├── scalar.rs           # 标量回退路径
│   │   ├── simd.rs             # SIMD 路径 (pulp)
│   │   └── parallel.rs         # 并行路径 (rayon)
│   │
│   ├── # === 辅助模块 ===
│   ├── private/                # 内部工具 #[doc(hidden)]
│   │   ├── mod.rs
│   │   ├── alloc.rs            # 对齐分配器
│   │   └── math.rs             # 数学辅助函数
│   └── macros.rs               # s![] 宏等
│
├── tests/                      # 集成测试
│   ├── construction.rs
│   ├── arithmetic.rs
│   ├── broadcasting.rs
│   ├── indexing.rs
│   ├── shape_ops.rs
│   ├── reduction.rs
│   ├── matrix_ops.rs
│   ├── set_ops.rs
│   ├── ffi.rs
│   ├── workspace.rs
│   ├── edge_cases.rs           # 空张量、单元素、NaN/Inf、非连续
│   └── property/               # 属性测试
│       ├── mod.rs
│       ├── associativity.rs
│       └── broadcast_rules.rs
│
├── benches/                    # 基准测试
│   ├── reduction.rs
│   ├── element_ops.rs
│   ├── matrix_ops.rs
│   ├── construction.rs
│   └── iterator.rs
│
└── examples/                   # 示例
    ├── basics.rs
    ├── linalg.rs
    └── ffi_integration.rs
```

---

## 3. 模块依赖图

```
                    ┌──────────┐
                    │  error   │  ← 全局错误类型
                    └────┬─────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
    ┌─────┴─────┐  ┌────┴─────┐  ┌─────┴──────┐
    │ dimension │  │ element  │  │   layout    │
    │ (Ix0~D)  │  │ (traits) │  │ (flags/str) │
    └─────┬─────┘  └────┬─────┘  └─────┬──────┘
          │              │              │
          │        ┌─────┴─────┐       │
          │        │ complex   │       │
          │        └─────┬─────┘       │
          │              │             │
          └──────┬───────┼─────────────┘
                 │       │
           ┌─────┴───────┴─────┐
           │     storage       │  ← Storage trait + 4 种实现
           └─────────┬─────────┘
                     │
           ┌─────────┴─────────┐
           │     tensor        │  ← TensorBase<S, D> + 类型别名
           └─────────┬─────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
  ┌─────┴─────┐ ┌───┴────┐ ┌─────┴──────┐
  │broadcast  │ │ indexing│ │  shape/    │
  │           │ │         │ │  (ops)     │
  └─────┬─────┘ └───┬─────┘ └─────┬──────┘
        │           │             │
        └─────┬─────┴──────┬─────┘
              │            │
        ┌─────┴─────┐ ┌───┴──────────┐
        │   iter/   │ │ construction │
        └─────┬─────┘ └──────┬───────┘
              │               │
        ┌─────┴───────────────┴─────┐
        │         ops/               │
        │  (element_wise, matrix,   │
        │   reduction, accumulate,  │
        │   set_ops)                │
        └──────────┬────────────────┘
                   │
          ┌────────┼────────┐
          │        │        │
    ┌─────┴──┐ ┌──┴───┐ ┌──┴──────────┐
    │  ffi   │ │ws/sp │ │ conversion  │
    └────────┘ └──────┘ └─────────────┘

    # 后端模块（横切关注点）
    ┌─────────────────────────────────┐
    │         backend/                │
    │  (scalar, simd, parallel)       │
    └─────────────────────────────────┘
```

---

## 4. Feature Flags 与条件编译

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

[dev-dependencies]
# 测试/基准用依赖可额外添加
```

### Feature 影响矩阵

| Feature | 影响模块 | 说明 |
|---------|----------|------|
| `std`（默认） | 全局 | 启用 `std::sync::Arc`, `std::vec::Vec` 等 |
| `!std` | 全局 | 使用 `alloc` crate 替代 |
| `parallel` | `backend/parallel.rs`, `iter/` | rayon 并行迭代与归约 |
| `simd` | `backend/simd.rs`, `ops/` | pulp SIMD 加速 |
| `full` | 以上全部 | 便利 feature |

---

## 5. 实施阶段

### Phase 1: 项目脚手架

| 任务 | 产出 | 预计 |
|------|------|------|
| 初始化 Cargo.toml | 项目配置 | 5 min |
| 创建 .rustfmt.toml | 格式化配置 | 3 min |
| 创建 clippy.toml | lint 配置 | 3 min |
| 创建 src/lib.rs 骨架 | 模块声明 + re-export | 10 min |

### Phase 2: 核心模块（顺序依赖）

```
dimension ──→ storage ──→ tensor ──→ (API 模块)
element   ──→ storage
complex   ──→ element
layout    ──→ storage
error     ──→ tensor
```

实施顺序：

| 波次 | 模块 | 可并行 | 依赖 |
|------|------|--------|------|
| W1 | dimension, element, error, layout | ✅ 全部可并行 | 无 |
| W2 | complex | ❌ | element |
| W3 | storage | ❌ | dimension, element, layout |
| W4 | tensor | ❌ | dimension, storage, layout, error |

### Phase 3: API 模块（可并行）

所有 API 模块仅依赖 Phase 2 的核心模块，彼此无依赖，可全部并行：

| 模块 | 独立性 | 关键依赖 |
|------|--------|----------|
| construction | 独立 | tensor, element |
| conversion | 独立 | tensor, element, complex |
| iter/ | 独立 | tensor, dimension |
| broadcast | 独立 | tensor, dimension, layout |
| indexing | 独立 | tensor, dimension, layout |
| shape/ | 独立 | tensor, dimension, layout |
| ops/element_wise | 独立 | tensor, element, backend |
| ops/matrix | 独立 | tensor, element, layout |
| ops/reduction | 独立 | tensor, element, backend |
| ops/accumulate | 独立 | tensor, element |
| ops/set_ops | 独立 | tensor, element |
| ffi | 独立 | tensor, storage, layout |
| workspace | 独立 | 无（独立工具模块） |

### Phase 4: 后端模块

| 模块 | 依赖 |
|------|------|
| backend/scalar | 无（标量回退） |
| backend/simd | pulp, element |
| backend/parallel | rayon, tensor |

### Phase 5: 测试与文档

| 模块 | 依赖 |
|------|------|
| 集成测试 | 所有 API 模块 |
| 基准测试 | 所有 API 模块 |
| 文档 | 所有公开 API |

---

## 6. 核心数据流

### 6.1 数据创建流程

```
用户调用 zeros::<f64, Ix2>([3, 4])
    │
    ├── Dimension::ndim()         → 2
    ├── Dimension::slice()        → [3, 4]
    ├── 计算总元素数               → 12
    ├── 计算步长 (F-order)         → [1, 3]
    ├── 对齐分配 12 * 8 = 96 字节  → 64 字节对齐
    ├── 初始化为零值               → [0.0; 12]
    ├── 计算 LayoutFlags           → F_CONTIGUOUS | ALIGNED
    └── 返回 TensorBase<Owned<f64>, Ix2>
```

### 6.2 运算执行流程

```
a + b (逐元素加法)
    │
    ├── 检查形状兼容性
    │   ├── 兼容 → 继续
    │   └── 不兼容 → 尝试广播
    │       ├── 广播成功 → 继续使用广播视图
    │       └── 广播失败 → 返回 BroadcastError
    │
    ├── 选择执行路径
    │   ├── 元素数 < SIMD宽度 → 标量
    │   ├── 连续 + simd feature → SIMD
    │   ├── 大数组 + parallel feature → SIMD + 并行
    │   └── 非连续 → 标量
    │
    ├── 执行运算
    │   └── 分配新 Owned 存储（64 字节对齐）
    │
    └── 返回 Tensor<A, D>
```

### 6.3 视图创建流程

```
tensor.slice(s![0..2, ..])
    │
    ├── 计算切片偏移量             → offset += 0 * strides[0]
    ├── 计算新 shape              → [2, original_shape[1]]
    ├── 继承/调整 strides          → 不变
    ├── 重新计算 LayoutFlags       → 可能降级
    └── 返回 TensorView<'a, A, Ix2>（零拷贝）
```

---

## 7. 类型系统全景

### 7.1 存储 × 维度 组合

|  | Ix0 | Ix1 | Ix2 | Ix3 | IxDyn |
|--|-----|-----|-----|-----|-------|
| **Owned\<A\>** | Tensor0\<A\> | Tensor1\<A\> | Tensor2\<A\> | Tensor3\<A\> | TensorD\<A\> |
| **ViewRepr\<&A\>** | View0\<A\> | View1\<A\> | View2\<A\> | View3\<A\> | ViewD\<A\> |
| **ViewMutRepr\<&mut A\>** | ViewMut0\<A\> | ViewMut1\<A\> | ViewMut2\<A\> | ViewMut3\<A\> | ViewMutD\<A\> |
| **ArcRepr\<A\>** | ArcTensor0\<A\> | ArcTensor1\<A\> | ArcTensor2\<A\> | ArcTensor3\<A\> | ArcTensorD\<A\> |

### 7.2 元素类型层次

```
Element (基础层: Copy, Clone, PartialEq, Debug, Display, Send, Sync, zero(), one())
├── Numeric (数值层: Add, Sub, Mul, Div, Neg)
│   ├── RealScalar (实数层: f32, f64 — 数学函数, 常量)
│   └── ComplexScalar (复数层: Complex<f32>, Complex<f64> — 复数运算)
└── bool (仅基础层, 不支持四则运算)
```

---

## 8. 配置常量

```rust
// 默认对齐（AVX-512 缓存行）
pub const DEFAULT_ALIGNMENT: usize = 64;

// 并行阈值
pub const PARALLEL_THRESHOLD: usize = 64 * 1024;  // 64K 元素

// 并行分块最小大小
pub const PARALLEL_MIN_CHUNK: usize = 4 * 1024;   // 4K 元素

// Debug 显示截断阈值
pub const DISPLAY_MAX_ELEMENTS: usize = 1000;

// 布局标志位
pub const F_CONTIGUOUS: u8 = 0b0000_0001;
pub const C_CONTIGUOUS: u8 = 0b0000_0010;
pub const ALIGNED:       u8 = 0b0000_0100;
pub const HAS_ZERO_STRIDE: u8 = 0b0000_1000;
pub const HAS_NEG_STRIDE: u8 = 0b0001_0000;
```

---

## 9. 公共 API 导出策略

`src/lib.rs` 负责 re-export 所有公共类型，用户只需 `use Renon::...`：

```rust
// 核心类型
pub use crate::tensor::{TensorBase, Tensor, TensorView, TensorViewMut, ArcTensor};
pub use crate::tensor::{Tensor0, Tensor1, Tensor2, Tensor3, TensorD};

// 维度类型
pub use crate::dimension::{Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn, Dimension};

// 元素 trait
pub use crate::element::{Element, Numeric, RealScalar, ComplexScalar};

// 复数
pub use crate::complex::Complex;

// 存储
pub use crate::storage::{Owned, ViewRepr, ViewMutRepr, ArcRepr, Storage};

// 布局
pub use crate::layout::{LayoutFlags, Order};

// 错误
pub use crate::error::{TensorError, Result};

// 构造函数
pub use crate::construction::{zeros, ones, full, empty, eye, identity, diag, arange, linspace, logspace};

// 运算符 re-export
pub use crate::ops::*;

// 宏
pub use crate::macros::s;
```
