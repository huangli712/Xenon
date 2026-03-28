# 文档方案设计

> 文档编号: 24 | 范围: API 文档、用户指南、贡献者指南、CI 集成
> 前置文档: `00-rust-standards.md`, `01-architecture-overview.md`, 需求说明书 §17.1

---

## 1. 模块定位

Xenon 文档体系由四个互补层级组成，覆盖从 API 参考到架构决策的完整知识谱：

| 层级 | 载体 | 目标读者 | 维护方式 |
|------|------|----------|----------|
| API 参考 | 源码 doc comments（rustdoc 生成） | 库开发者、系统开发者 | 随代码同步维护 |
| 用户指南 | `book/` 目录（mdBook 格式） | 所有用户 | 独立 markdown 文件 |
| 贡献者指南 | `CONTRIBUTING.md` | 项目贡献者 | 手动维护 |
| 设计文档 | `docs/` 目录（00–21 已完成） | 维护者、贡献者 | 设计阶段产出，后续按需更新 |

### 文档目标

1. **API 可发现性** — 通过 rustdoc 让用户快速找到所需函数/trait，理解签名、约束和行为
2. **学习曲线平滑** — 用户指南从快速入门到高级主题递进，配合可运行的代码示例
3. **正确性保证** — doc test 作为测试矩阵的一部分，确保文档示例始终可编译且行为正确
4. **贡献效率** — 贡献者指南明确约定，减少 review 来回

---

## 2. 文档分类与位置

### 2.1 整体目录结构

```
xenon/
├── src/                        # API 文档：doc comments 写在源码中
│   ├── lib.rs                  # crate-level doc comment (//!)
│   ├── dimension.rs
│   ├── ...
│   └── backend/
│       └── ...
│
├── book/                       # 用户指南（mdBook）
│   ├── book.toml               # mdBook 配置
│   ├── src/
│   │   ├── SUMMARY.md          # 目录结构
│   │   ├── introduction.md     # 简介
│   │   ├── quickstart.md       # 快速入门
│   │   ├── tensor-basics.md    # 张量基础
│   │   ├── indexing.md         # 索引与切片
│   │   ├── operations.md       # 运算操作
│   │   ├── broadcasting.md     # 广播机制
│   │   ├── performance.md      # 性能指南
│   │   ├── ffi.md              # FFI 集成
│   │   ├── no-std.md           # no_std 使用
│   │   └── migration.md        # 从 ndarray 迁移指南
│   └── theme/                  # 可选：自定义 mdBook 主题
│       └── index.hbs
│
├── docs/                       # 设计文档（已完成 00–21）
│   ├── 00-rust-standards.md
│   ├── ...
│   └── 21-parallel.md
│
├── examples/                   # 可运行的示例程序
│   ├── basics.rs
│   ├── linalg.rs
│   └── ffi_integration.rs
│
├── CONTRIBUTING.md             # 贡献者指南
└── README.md                   # 项目简介（含 badges）
```

### 2.2 各载体详细说明

#### API 文档（源码 doc comments）

- **位置**：所有 `src/**/*.rs` 文件中的 `///` 和 `//!` 注释
- **生成**：`cargo doc --all-features --no-deps`
- **输出**：`target/doc/xenon/`
- **原则**：doc comment 与代码同生命周期，修改 API 时必须同步更新文档

#### 用户指南（mdBook）

- **位置**：`book/` 目录
- **构建**：`mdbook build book/` → `book/book/`（静态 HTML）
- **部署**：GitHub Pages（CI 自动构建与发布）
- **内容来源**：独立于源码，可包含概念解释、教程、最佳实践

#### 贡献者指南

- **位置**：项目根目录 `CONTRIBUTING.md`
- **内容**：开发环境设置、代码规范引用、PR 流程、测试要求

#### 设计文档

- **位置**：`docs/` 目录（00–21 已存在）
- **用途**：实现阶段的参考蓝图，不面向最终用户

---

## 3. API 文档规范

### 3.1 Doc Comment 格式

#### Crate-level 文档（`//!`）

`src/lib.rs` 顶部使用 `//!` 撰写 crate 级文档：

```rust
//! # Xenon
//!
//! A Rust multidimensional array (tensor) library for scientific computing.
//!
//! ## Quick Start
//!
//! ```
//! use xenon::{Tensor, zeros, Ix2};
//!
//! let a: Tensor<f64, Ix2> = zeros([3, 4]);
//! assert_eq!(a.shape(), &[3, 4]);
//! ```
//!
//! ## Feature Flags
//!
//! - `std` (default) — Standard library support
//! - `parallel` — Rayon-based data parallelism
//! - `simd` — SIMD acceleration via pulp
//!
//! ## Memory Layout
//!
//! By default, Xenon uses **F-order (column-major)** layout for compatibility
//! with BLAS/LAPACK conventions. C-order (row-major) is opt-in via [`Order`].
```

#### Item-level 文档（`///`）

所有 `pub` 项使用 `///` 撰写文档。每条 doc comment 遵循以下结构：

```
/// 一句话摘要（第三人称动词开头）。
///
/// # Arguments  （如有参数）
///
/// # Returns    （如返回值需要说明）
///
/// # Panics     （如可能 panic）
///
/// # Errors     （如返回 Result）
///
/// # Safety     （仅 unsafe 函数）
///
/// # Examples   （关键 API）
```

**示例**：

```rust
/// Creates a new tensor filled with zeros.
///
/// # Arguments
///
/// * `shape` - The shape of the tensor. Accepts any type that implements
///   [`Dimension`], such as `[usize; 2]` for a 2D tensor or [`IxDyn`]
///   for dynamic rank.
/// * `order` - Memory layout order. Defaults to F-order if not specified.
///
/// # Panics
///
/// Panics if the total number of elements overflows `usize`.
///
/// # Examples
///
/// ```
/// use xenon::{Tensor, zeros, Ix2};
///
/// let a: Tensor<f64, Ix2> = zeros([3, 4]);
/// assert_eq!(a.shape(), &[3, 4]);
/// assert_eq!(a[[0, 0]], 0.0);
/// ```
pub fn zeros<A, D>(shape: D) -> Tensor<A, D>
where
    A: Element,
    D: Dimension,
{
    // ...
}
```

### 3.2 各模块 doc comment 覆盖要求

| 模块 | 覆盖范围 | 优先级 |
|------|----------|--------|
| `tensor.rs` | `TensorBase` 及所有公开方法、类型别名 | P0 |
| `dimension.rs` | `Dimension` trait、`Ix0`–`Ix6`、`IxDyn` | P0 |
| `element.rs` | `Element`、`Numeric`、`RealScalar`、`ComplexScalar` | P0 |
| `complex.rs` | `Complex<T>` 及所有公开方法 | P0 |
| `storage/` | `Storage` trait、`Owned`/`ViewRepr`/`ViewMutRepr`/`ArcRepr` | P0 |
| `layout.rs` | `LayoutFlags`、`Order` | P0 |
| `error.rs` | `TensorError` 各变体、`Result` 别名 | P0 |
| `construction.rs` | 所有构造函数 | P0 |
| `conversion.rs` | `cast`、运算符重载 | P1 |
| `iter/` | 迭代器 trait 与类型 | P1 |
| `broadcast.rs` | 广播函数 | P1 |
| `indexing.rs` | 索引操作、`s![]` 宏 | P1 |
| `shape/` | 所有形状操作 | P1 |
| `ops/` | 所有运算操作 | P1 |
| `ffi.rs` | 指针 API、BLAS 兼容方法 | P1 |
| `workspace.rs` | `Workspace`、`ScratchNeed` | P1 |
| `backend/` | 计算后端（内部模块，最低覆盖） | P2 |

### 3.3 Safety 文档节

所有 `unsafe fn` 和 `unsafe impl` **必须**包含 `# Safety` 节，逐条列出调用方须满足的前提条件：

```rust
/// Constructs a tensor view from raw parts.
///
/// # Safety
///
/// The caller must ensure that:
///
/// - `ptr` is non-null, non-dangling, and aligned to `align_of::<A>()`.
/// - The memory range starting from `ptr` covers all accessible elements
///   (considering `offset`, `shape`, and `strides`).
/// - The memory remains valid for the lifetime `'a`.
/// - No mutable references to the same memory exist for the returned
///   view's lifetime (aliasing rules).
/// - All accessible elements are properly initialized.
///
/// # Examples
///
/// ```
/// use xenon::{TensorView, from_raw_parts, Ix2};
///
/// let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let view: TensorView<f64, Ix2> = unsafe {
///     from_raw_parts(data.as_ptr(), [2, 3], [1, 2], 0)
/// };
/// assert_eq!(view.shape(), &[2, 3]);
/// ```
pub unsafe fn from_raw_parts<'a, A, D>(
    ptr: *const A,
    shape: D,
    strides: D,
    offset: usize,
) -> TensorView<'a, A, D>
where
    D: Dimension,
{
    // ...
}
```

### 3.4 Panic 文档节

任何可能 panic 的公开函数**必须**包含 `# Panics` 节，精确描述触发条件：

```rust
/// Returns the element at the given index.
///
/// # Panics
///
/// Panics if any dimension of `index` is out of bounds.
/// Use [`get`](TensorBase::get) for a checked variant that returns `Result`.
pub fn index(&self, index: &[usize]) -> &A {
    // ...
}
```

### 3.5 Feature-Gate 文档

当 API 受 feature gate 控制时，doc comment 中须注明启用条件：

```rust
/// Computes the sum of all elements using parallel iteration.
///
/// **Requires** `feature = "parallel"`.
///
/// # Examples
///
/// ```ignore
/// use xenon::{Tensor, zeros, Ix2};
///
/// let a: Tensor<f64, Ix2> = zeros([1000, 1000]);
/// let sum = a.par_sum();
/// ```
#[cfg(feature = "parallel")]
pub fn par_sum(&self) -> A
where
    A: Numeric + Send + Sync,
{
    // ...
}
```

### 3.6 交叉引用

使用 rustdoc 的链接语法引用其他类型，提升 API 可导航性：

| 语法 | 用途 | 示例 |
|------|------|------|
| `[`Type`]` | 引用同 crate 类型 | `[`Tensor`]`, `[`Dimension`]` |
| `[`Type::method`]` | 引用方法 | `[`TensorBase::shape`]` |
| `[`module::Type`]` | 引用其他模块类型 | `[`layout::Order`]` |
| `[`Trait`]` | 引用 trait | `[`Element`]`, `[`Numeric`]` |

**原则**：在 doc comment 中首次提及某个 Xenon 类型时，使用 intra-doc link；后续提及不需要重复链接。

### 3.7 Module-level 文档

每个公开模块（`pub mod`）须有模块级 doc comment（`//!`），说明模块职责、核心类型和典型用法：

```rust
//! Dimension types for tensor shape representation.
//!
//! This module provides static dimension types ([`Ix0`]–[`Ix6`]) and a
//! dynamic dimension type ([`IxDyn`]). All dimension types implement the
//! [`Dimension`] trait.
//!
//! # Static vs Dynamic
//!
//! Static dimensions (`IxN`) use fixed-size arrays for shape/stride storage,
//! enabling stack allocation and compile-time rank checking. Dynamic dimensions
//! (`IxDyn`) use heap-allocated `Vec<usize>` for runtime-determined rank.
//!
//! ```
//! use xenon::{Ix2, IxDyn, Dimension};
//!
//! let static_dim = Ix2([3, 4]);
//! assert_eq!(static_dim.ndim(), 2);
//!
//! let dyn_dim: IxDyn = static_dim.into_dimension();
//! assert_eq!(dyn_dim.ndim(), 2);
//! ```
```

---

## 4. mdBook 用户指南结构

### 4.1 book.toml 配置

```toml
# book/book.toml
[book]
title = "Xenon Book"
authors = ["Xenon Contributors"]
language = "en"
multilingual = false
src = "src"

[build]
build-dir = "book"

[output.html]
default-theme = "navy"
git-repository-url = "https://github.com/user/xenon"
edit-url-template = "https://github.com/user/xenon/edit/main/book/{path}"

[output.html.search]
enable = true
limit-results = 30
```

### 4.2 SUMMARY.md（目录结构）

```markdown
# Summary

- [Introduction](introduction.md)
- [Quick Start](quickstart.md)
- [Tensor Basics](tensor-basics.md)
  - [Creating Tensors](tensor-basics.md#creating-tensors)
  - [Shape & Strides](tensor-basics.md#shape--strides)
  - [Memory Layout](tensor-basics.md#memory-layout)
  - [Element Types](tensor-basics.md#element-types)
- [Indexing & Slicing](indexing.md)
  - [Basic Indexing](indexing.md#basic-indexing)
  - [The `s![]` Macro](indexing.md#the-s-macro)
  - [Advanced Indexing](indexing.md#advanced-indexing)
- [Operations](operations.md)
  - [Element-wise Operations](operations.md#element-wise-operations)
  - [Reductions](operations.md#reductions)
  - [Matrix Operations](operations.md#matrix-operations)
  - [Set Operations](operations.md#set-operations)
- [Broadcasting](broadcasting.md)
- [Performance Guide](performance.md)
  - [SIMD](performance.md#simd)
  - [Parallelism](performance.md#parallelism)
  - [Memory Alignment](performance.md#memory-alignment)
  - [Benchmarking Tips](performance.md#benchmarking-tips)
- [FFI Integration](ffi.md)
  - [Pointer API](ffi.md#pointer-api)
  - [BLAS Compatibility](ffi.md#blas-compatibility)
  - [Integrating with C](ffi.md#integrating-with-c)
- [no_std Usage](no-std.md)
- [Migration from ndarray](migration.md)
```

### 4.3 各章节内容规划

#### introduction.md

- Xenon 的定位与设计哲学
- 与 ndarray、nalgebra 的区别
- 功能范围说明（范围内/范围外）

#### quickstart.md

- 添加依赖（Cargo.toml 配置）
- 创建第一个张量
- 基本运算
- 索引与切片入门

```rust
// quickstart.md 示例代码
use xenon::{Tensor, zeros, ones, Ix2, Ix1};

// Create tensors
let a: Tensor<f64, Ix2> = zeros([3, 4]);
let b: Tensor<f64, Ix2> = ones([3, 4]);

// Arithmetic
let c = &a + &b;

// Indexing
let first_row = c.index_axis(0, 0);
```

#### tensor-basics.md

- 四种存储模式（Owned / View / ViewMut / Arc）
- 静态维度与动态维度
- F-order vs C-order 布局
- 元素类型层次（Element → Numeric → RealScalar / ComplexScalar）
- Complex 类型

#### indexing.md

- 多维索引语法 `[i, j, k]`
- 范围索引与切片
- `s![]` 宏用法
- 高级索引（take、put、mask、compress）
- 条件选择（where）

#### operations.md

- 逐元素运算（算术、三角、指数/对数、数值函数）
- 归约（sum、prod、mean、var、std、min、max、argmin、argmax）
- 累积归约（cumsum、cumprod）
- 矩阵运算（matvec、dot、outer、batch 操作）
- 集合操作（unique、bincount、histogram）
- 运算符重载（`+`、`-`、`*`、`/`、`+=` 等）

#### broadcasting.md

- NumPy 广播规则详解
- 广播视图与零步长
- 广播与运算符的结合
- 常见广播模式

#### performance.md

- SIMD 配置与支持的操作
- 并行阈值与调优
- 内存对齐对性能的影响
- 连续 vs 非连续布局的性能差异
- 何时使用 `to_f_contiguous()` / `to_c_contiguous()`

#### ffi.md

- 原始指针 API
- BLAS 兼容性检查方法
- 与 C 库集成的完整示例
- 临时工作空间使用

#### no-std.md

- Feature gate 配置
- `alloc` crate 依赖
- 可用的功能子集
- 嵌入式场景注意事项

#### migration.md

- ndarray 用户迁移对照表
- API 差异说明
- 常见迁移问题与解决方案

### 4.4 mdBook 中的代码示例约定

- 所有代码示例使用 Rust 语法高亮（默认 ` ```rust `）
- 优先使用可编译运行的完整代码片段
- 输出结果以注释形式附在代码后方：

```rust
let a: Tensor<f64, Ix2> = zeros([2, 3]);
println!("{}", a.shape());
// Output: [2, 3]
```

- 不适用 doc test 的复杂示例使用 ` ```ignore ` 标记
- 依赖 feature gate 的示例使用 ` ```ignore ` 并在文中说明

---

## 5. Doc tests 策略

### 5.1 Doc tests 分类

| 类别 | 标记 | 执行策略 | 适用场景 |
|------|------|----------|----------|
| 标准 doc test | ` ``` ` | `cargo test --doc` 自动执行 | 大多数公开 API 示例 |
| 忽略执行 | ` ```ignore ` | 不执行 | 需要 feature gate、外部依赖或复杂 setup 的示例 |
| 编译检查 | ` ```compile_fail ` | 确保编译失败 | 演示错误用法（如形状不匹配） |
| 无运行 | ` ```no_run ` | 编译但不运行 | 需要文件 I/O 或长时间运行的示例 |

### 5.2 标准 Doc test 模式

#### 基本模式（无额外 import）

大多数 doc test 遵循以下模式：从 `xenon` crate 导入所需类型后操作：

```rust
/// # Examples
///
/// ```
/// use xenon::{Tensor, zeros, Ix2};
///
/// let a: Tensor<f64, Ix2> = zeros([3, 4]);
/// assert_eq!(a.shape(), &[3, 4]);
/// ```
```

#### 泛型 API 的 Doc test

当 API 涉及泛型参数时，显式标注类型以确保可编译：

```rust
/// # Examples
///
/// ```
/// use xenon::{Tensor, ones, Ix1, Tensor1};
///
/// let v: Tensor1<f64> = ones(5);
/// assert_eq!(v.shape(), &[5]);
/// ```
```

#### Result 返回的 Doc test

函数返回 `Result` 时，doc test 使用 `?` 并标注返回类型：

```rust
/// # Examples
///
/// ```
/// use xenon::{Tensor, zeros, Ix2};
///
/// let a: Tensor<f64, Ix2> = zeros([3, 4]);
/// let b: Tensor<f64, Ix2> = zeros([3, 4]);
/// let c = a.add(&b)?;  // Result<Tensor, TensorError>
/// # Ok::<(), xenon::TensorError>(())
/// ```
```

### 5.3 Feature-Gated Doc test 处理

Feature-gated API 的 doc test 使用 ` ```ignore ` 标记，因为 `cargo test --doc` 默认仅使用 `default` features：

```rust
/// Computes the sum using parallel iteration.
///
/// # Examples
///
/// ```ignore
/// use xenon::{Tensor, ones, Ix2};
///
/// let a: Tensor<f64, Ix2> = ones([1000, 1000]);
/// let sum = a.par_sum();
/// assert_eq!(sum, 1_000_000.0);
/// ```
#[cfg(feature = "parallel")]
pub fn par_sum(&self) -> A { ... }
```

**CI 中额外运行**：`cargo test --doc --all-features` 确保所有 feature 组合下的 doc test 通过。

### 5.4 unsafe API 的 Doc test

unsafe 函数的 doc test 须使用 `unsafe` 块并解释安全性假设：

```rust
/// # Examples
///
/// ```
/// use xenon::{from_raw_parts, Ix1};
///
/// let data = vec![1.0_f64, 2.0, 3.0];
///
/// // SAFETY: data is valid, non-null, properly aligned, and the shape/strides
/// // describe a valid 3-element contiguous view.
/// let view = unsafe {
///     from_raw_parts(data.as_ptr(), Ix1(3), Ix1(1), 0)
/// };
/// assert_eq!(view.shape(), &[3]);
/// ```
```

### 5.5 Doc test 不适用的场景

以下场景使用 ` ```ignore ` 而非 doc test：

| 场景 | 原因 | 替代方案 |
|------|------|----------|
| 需要 `feature = "parallel"` 或 `"simd"` | 默认 feature 不包含 | CI 中 `--all-features` 验证 |
| 需要 `no_std` 环境 | doc test 在 std 环境运行 | 独立集成测试 |
| 演示 panic 行为 | `should_panic` 不适用于 doc test | 单元测试中覆盖 |
| 大规模数据操作 | doc test 不应耗时过长 | `examples/` 目录 |
| FFI 集成示例 | 需要外部 C 库 | `examples/ffi_integration.rs` |

### 5.6 公共 import 模式

所有 doc test 中的 import 应使用 crate 级 re-export 路径（`xenon::XXX`），不引用内部模块路径：

```rust
// 正确
use xenon::{Tensor, zeros, Ix2, Dimension};

// 错误（内部模块路径，不属于公共 API）
use xenon::tensor::TensorBase;
use xenon::dimension::Ix2;
```

---

## 6. CI 集成

### 6.1 CI Pipeline 中的文档任务

```yaml
# .github/workflows/docs.yml
name: Documentation

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  api-docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable

      # Build API docs with all features
      - name: Build rustdoc
        run: cargo doc --all-features --no-deps

      # Check for doc warnings (treat warnings as errors)
      - name: Check doc warnings
        run: cargo doc --all-features --no-deps -- -D warnings

      # Run doc tests with all features
      - name: Run doc tests
        run: cargo test --doc --all-features

  book:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install mdBook
        run: cargo install mdbook

      # Build the user guide
      - name: Build mdBook
        run: mdbook build book

      # Check internal links
      - name: Link check
        uses: lycheeverse/lychee-action@v1
        with:
          args: book/book/**/*.html

      # Deploy to GitHub Pages (main branch only)
      - name: Deploy to GitHub Pages
        if: github.ref == 'refs/heads/main'
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./book/book

  linkcheck-rustdoc:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - name: Build and check intra-doc links
        run: |
          cargo doc --all-features --no-deps
          # Verify no broken intra-doc links (RUSTDOCFLAGS enables checking)
          RUSTDOCFLAGS="--cfg docsrs" cargo doc --all-features --no-deps
```

### 6.2 CI 检查项清单

| 检查项 | 命令 | 失败条件 |
|--------|------|----------|
| rustdoc 构建成功 | `cargo doc --all-features --no-deps` | 任何 doc comment 语法错误 |
| rustdoc 无警告 | `cargo doc --all-features --no-deps -- -D warnings` | 缺少 doc comment、broken link |
| Doc tests 通过 | `cargo test --doc --all-features` | 任何 doc test 编译失败或断言失败 |
| mdBook 构建成功 | `mdbook build book` | markdown 语法或配置错误 |
| 链接有效 | lychee link checker | 任何 broken 链接 |
| 页面部署 | GitHub Pages | 仅 main 分支触发 |

### 6.3 rustdoc 配置

在 `Cargo.toml` 中配置文档元数据：

```toml
[package]
name = "xenon"
version = "0.1.0"
description = "A Rust multidimensional array library for scientific computing"
documentation = "https://docs.rs/xenon"
repository = "https://github.com/user/xenon"
license = "MIT"
keywords = ["tensor", "array", "numpy", "scientific", "numerical"]
categories = ["science", "mathematics", "no-std"]

[package.metadata.docs.rs]
all-features = true
rustdoc-args = ["--cfg", "docsrs"]
```

### 6.4 docs.rs 发布配置

通过 `docs.rs` 元数据确保发布文档包含所有 feature：

- `all-features = true`：构建所有 feature 组合下的 API
- `rustdoc-args = ["--cfg", "docsrs"]`：允许 `#[cfg(docsrs)]` 条件编译

源码中使用 `#[cfg(docsrs)]` 标注需要特殊处理的文档：

```rust
#[cfg(docsrs)]
#[doc(cfg(feature = "parallel"))]
impl<A, D> TensorBase<A, D> {
    /// Computes the sum using parallel iteration.
    #[cfg(feature = "parallel")]
    pub fn par_sum(&self) -> A { ... }
}
```

---

## 7. CONTRIBUTING.md 要点

贡献者指南须包含以下关键内容（文档正文使用中文）：

### 7.1 必含章节

| 章节 | 内容 |
|------|------|
| 开发环境 | Rust 1.85+、cargo、mdBook 安装 |
| 代码规范引用 | 指向 `docs/00-rust-standards.md` |
| 测试要求 | `cargo test --all-features`、覆盖率 ≥ 80% |
| 文档要求 | 所有 pub 项须有 doc comment、doc test 须通过 |
| PR 流程 | fork → branch → commit → push → PR |
| Commit 规范 | 英文、简洁、描述"做了什么"而非"怎么做的" |
| CI 检查 | 列出所有 CI 检查项，本地预检命令 |

### 7.2 本地预检命令

贡献者提交 PR 前应运行：

```bash
# 格式化
cargo fmt --check

# Lint
cargo clippy --all-features -- -D warnings

# 测试
cargo test --all-features

# Doc tests
cargo test --doc --all-features

# 文档构建
cargo doc --all-features --no-deps -- -D warnings

# mdBook 构建
mdbook build book
```

---

## 8. 实现任务拆分

以下每个任务约 10 分钟，按依赖顺序排列。

### Phase 1: 基础设施（无代码依赖）

- [ ] **Task 1**: 创建 `book/book.toml` 和 `book/src/SUMMARY.md`
  - 文件: `book/book.toml`, `book/src/SUMMARY.md`
  - 产出: mdBook 配置和目录骨架
  - 前置: 无
  - 预计: 10 min

- [ ] **Task 2**: 创建 `book/src/introduction.md`
  - 文件: `book/src/introduction.md`
  - 产出: Xenon 简介、定位、功能范围
  - 前置: Task 1
  - 预计: 10 min

- [ ] **Task 3**: 在 `Cargo.toml` 中添加 docs.rs 元数据
  - 文件: `Cargo.toml`
  - 产出: `[package.metadata.docs.rs]` 配置节
  - 前置: 无
  - 预计: 5 min

- [ ] **Task 4**: 创建 `CONTRIBUTING.md` 基础框架
  - 文件: `CONTRIBUTING.md`
  - 产出: 开发环境、代码规范引用、PR 流程、本地预检命令
  - 前置: 无
  - 预计: 10 min

- [ ] **Task 5**: 创建 `.github/workflows/docs.yml`
  - 文件: `.github/workflows/docs.yml`
  - 产出: API docs 构建、doc tests、mdBook 构建/部署、link check
  - 前置: Task 1, Task 3
  - 预计: 10 min

### Phase 2: Crate-level 文档

- [ ] **Task 6**: 撰写 `src/lib.rs` crate-level doc comment
  - 文件: `src/lib.rs`
  - 产出: `//!` 格式的 crate 概览、feature flags 说明、快速入门示例
  - 前置: lib.rs 存在
  - 预计: 10 min

### Phase 3: 模块级 doc comments（与代码实现并行）

- [ ] **Task 7**: 撰写 `dimension` 模块 doc comment
  - 文件: `src/dimension.rs`
  - 产出: `//!` 模块级文档，静态 vs 动态维度说明
  - 前置: dimension.rs 存在
  - 预计: 10 min

- [ ] **Task 8**: 撰写 `element` 模块 doc comment
  - 文件: `src/element.rs`
  - 产出: `//!` 模块级文档，元素类型层次说明
  - 前置: element.rs 存在
  - 预计: 10 min

- [ ] **Task 9**: 撰写 `storage` 模块 doc comment
  - 文件: `src/storage/mod.rs`
  - 产出: `//!` 模块级文档，四种存储模式说明
  - 前置: storage/mod.rs 存在
  - 预计: 10 min

- [ ] **Task 10**: 撰写 `layout` 模块 doc comment
  - 文件: `src/layout.rs`
  - 产出: `//!` 模块级文档，布局标志与对齐策略说明
  - 前置: layout.rs 存在
  - 预计: 10 min

- [ ] **Task 11**: 撰写 `tensor` 模块 doc comment
  - 文件: `src/tensor.rs`
  - 产出: `//!` 模块级文档，TensorBase 核心抽象说明
  - 前置: tensor.rs 存在
  - 预计: 10 min

- [ ] **Task 12**: 撰写 `error` 模块 doc comment
  - 文件: `src/error.rs`
  - 产出: `//!` 模块级文档，错误类型与处理策略说明
  - 前置: error.rs 存在
  - 预计: 10 min

- [ ] **Task 13**: 撰写 `complex` 模块 doc comment
  - 文件: `src/complex.rs`
  - 产出: `//!` 模块级文档，Complex 类型与实数互操作说明
  - 前置: complex.rs 存在
  - 预计: 10 min

### Phase 4: 核心 API doc comments（与代码实现并行）

- [ ] **Task 14**: 为 `TensorBase` 公开方法添加 doc comments
  - 文件: `src/tensor.rs`
  - 产出: shape/strides/布局查询/类型转换等方法的完整文档 + doc tests
  - 前置: tensor.rs 方法签名完成
  - 预计: 10 min × N（按方法分批）

- [ ] **Task 15**: 为 `Dimension` trait 及实现添加 doc comments
  - 文件: `src/dimension.rs`
  - 产出: Ix0–Ix6、IxDyn、Dimension trait 方法文档
  - 前置: dimension.rs 实现
  - 预计: 10 min

- [ ] **Task 16**: 为 `Element`/`Numeric`/`RealScalar`/`ComplexScalar` 添加 doc comments
  - 文件: `src/element.rs`
  - 产出: 四层 trait 的方法文档 + 类型约束说明
  - 前置: element.rs 实现
  - 预计: 10 min

- [ ] **Task 17**: 为构造函数（zeros/ones/full/eye 等）添加 doc comments
  - 文件: `src/construction.rs`
  - 产出: 所有构造函数的完整文档 + doc tests
  - 前置: construction.rs 实现
  - 预计: 10 min × N（按函数分批）

- [ ] **Task 18**: 为 FFI API（from_raw_parts/as_ptr/blas_*）添加 doc comments + Safety 节
  - 文件: `src/ffi.rs`
  - 产出: 所有 unsafe 函数的 Safety 文档 + BLAS 兼容方法文档
  - 前置: ffi.rs 实现
  - 预计: 10 min

- [ ] **Task 19**: 为 ops 模块公开 API 添加 doc comments
  - 文件: `src/ops/*.rs`
  - 产出: 逐元素运算/归约/矩阵运算/集合操作的文档
  - 前置: ops/ 实现
  - 预计: 10 min × N（按文件分批）

### Phase 5: 用户指南章节

- [ ] **Task 20**: 撰写 `book/src/quickstart.md`
  - 文件: `book/src/quickstart.md`
  - 产出: 依赖配置、第一个张量、基本运算、索引入门
  - 前置: Task 1, 核心 API 文档完成
  - 预计: 10 min

- [ ] **Task 21**: 撰写 `book/src/tensor-basics.md`
  - 文件: `book/src/tensor-basics.md`
  - 产出: 存储模式、维度系统、内存布局、元素类型
  - 前置: Task 1, 核心 API 文档完成
  - 预计: 10 min

- [ ] **Task 22**: 撰写 `book/src/indexing.md`
  - 文件: `book/src/indexing.md`
  - 产出: 索引语法、切片宏、高级索引
  - 前置: Task 1, indexing 文档完成
  - 预计: 10 min

- [ ] **Task 23**: 撰写 `book/src/operations.md`
  - 文件: `book/src/operations.md`
  - 产出: 逐元素运算、归约、矩阵运算、集合操作
  - 前置: Task 1, ops 文档完成
  - 预计: 10 min

- [ ] **Task 24**: 撰写 `book/src/broadcasting.md`
  - 文件: `book/src/broadcasting.md`
  - 产出: 广播规则详解、常见模式
  - 前置: Task 1, broadcast 文档完成
  - 预计: 10 min

- [ ] **Task 25**: 撰写 `book/src/performance.md`
  - 文件: `book/src/performance.md`
  - 产出: SIMD/并行/对齐/连续性性能指南
  - 前置: Task 1, backend 文档完成
  - 预计: 10 min

- [ ] **Task 26**: 撰写 `book/src/ffi.md`
  - 文件: `book/src/ffi.md`
  - 产出: 指针 API、BLAS 兼容、C 集成示例
  - 前置: Task 1, ffi 文档完成
  - 预计: 10 min

- [ ] **Task 27**: 撰写 `book/src/no-std.md`
  - 文件: `book/src/no-std.md`
  - 产出: no_std 配置、可用功能子集
  - 前置: Task 1
  - 预计: 10 min

- [ ] **Task 28**: 撰写 `book/src/migration.md`
  - 文件: `book/src/migration.md`
  - 产出: ndarray → Xenon 迁移对照表
  - 前置: Task 1, 全部 API 文档完成
  - 预计: 10 min

### Phase 6: 验证与部署

- [ ] **Task 29**: 验证所有 doc tests 通过
  - 命令: `cargo test --doc --all-features`
  - 产出: 所有 doc test 绿色通过
  - 前置: 所有 doc comments 完成
  - 预计: 10 min

- [ ] **Task 30**: 验证 rustdoc 构建无警告
  - 命令: `cargo doc --all-features --no-deps -- -D warnings`
  - 产出: 零 warning 的 API 文档
  - 前置: 所有 doc comments 完成
  - 预计: 10 min

- [ ] **Task 31**: 验证 mdBook 构建并检查链接
  - 命令: `mdbook build book && mdbook test book`
  - 产出: 可发布 HTML，无 broken links
  - 前置: 所有 book 章节完成
  - 预计: 10 min

- [ ] **Task 32**: 首次部署文档到 GitHub Pages
  - 操作: 合并到 main 分支，触发 CI 部署
  - 产出: API 文档 + 用户指南在线可访问
  - 前置: Task 5, Task 29–31
  - 预计: 10 min

---

## 9. 文档质量指标

| 指标 | 目标 | 验证方式 |
|------|------|----------|
| 公开 API 文档覆盖率 | 100% | `cargo doc -- -D warnings`（缺文档即 warning） |
| Doc test 通过率 | 100% | `cargo test --doc --all-features` |
| Broken intra-doc links | 0 | `cargo doc` 警告检测 |
| 用户指南章节完整性 | SUMMARY.md 中所有章节有内容 | mdBook 构建成功 + link check |
| CONTRIBUTING.md 准确性 | 本地预检命令与 CI 一致 | 人工 review |
| docs.rs 构建成功 | `all-features` 下无错误 | docs.rs 发布后验证 |

---

*Xenon 文档方案设计 — docs/24-documentation.md*
