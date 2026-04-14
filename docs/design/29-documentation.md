# 文档模块设计

> 文档编号: 29 | 影响范围: `src/**` pub API 文档、`README.md`、`examples/` 与 docs CI | 阶段: Phase 6
> 前置文档: 所有前置文档（`00-coding.md` ~ `28-tests.md`）
> 需求参考: 需求说明书 §28.1
> 范围声明: 范围内

---

## 1. 模块定位

### 1.1 职责边界

| 职责         | 包含                                                           | 不包含                 |
| ------------ | -------------------------------------------------------------- | ---------------------- |
| API 文档     | 所有 pub 类型和函数的 doc comment                              | 内部实现注释（非 pub） |
| 使用示例     | 关键 API 的可运行代码示例（doctest）                           | 完整教程、视频教程     |
| Safety 说明  | 所有 unsafe 函数的 `# Safety` 文档节（参见 `00-coding.md §5`） | 安全函数的 Safety 节   |
| Crate 级文档 | lib.rs 顶层文档、README、CHANGELOG                             | 第三方博客文章         |
| 模块级文档   | 各 mod.rs 的 `//!` 模块概述                                    | 内部实现文档           |
| examples/    | 独立可运行示例程序                                             | 交互式 notebook        |
| docs.rs 配置 | metadata、feature gate 标注                                    | 自定义文档主题         |

### 1.2 设计原则

| 原则       | 体现                                                      |
| ---------- | --------------------------------------------------------- |
| 全覆盖     | 所有 pub API 必须有 doc comment（参见 `00-coding.md §6`） |
| 可测试     | 关键 API 的示例通过 doctest 或独立 examples 验证          |
| 安全性透明 | 所有 unsafe 函数有 `# Safety` 节                          |
| 惯用法     | 遵循 Rust API Guidelines                                  |
| 英文文档   | 所有 doc comment 使用英文                                 |

### 1.3 在架构中的位置

```
Dependency layers:
L0: error, private
L1: dimension, element, complex
L2: layout (depends on dimension)
L3: storage (independent of layout; owned by tensor and consumes layout results)
L4: tensor (depends on storage, dimension)
L5: overload/, iter/, index/, shape/, broadcast.rs, construct/, ffi/, convert/, format/

Cross-cutting concern (global):
┌─────────────────────────────────────────────────────────────────┐
│  Documentation (doc comments, README, examples/)  <- current document (global) │
│  - Spans pub API docs across all L0-L5 modules                               │
└─────────────────────────────────────────────────────────────────┘
```

## 2. 需求映射与范围约束

| 类型     | 内容                                                           |
| -------- | -------------------------------------------------------------- |
| 需求映射 | `require.md §28.1`                                             |
| 范围内   | pub API 文档、doctest、examples、docs.rs 配置、README/CHANGELOG |
| 范围外   | 第三方教程平台、自定义文档主题、交互式 notebook 或站点系统     |
| 非目标   | 通过文档规范扩展产品能力、引入额外文档构建依赖或改变平台边界   |

---

## 3. 文件位置

### 3.1 文档源码分布

```
src/
├── lib.rs                    # Crate-level docs (L0)
├── dimension/
│   └── mod.rs                # Dimension module docs (L1)
├── element/
│   └── mod.rs                # Element-type module docs (L1)
├── complex/
│   └── mod.rs                # Complex-number module docs (L1)
├── storage/
│   └── mod.rs                # Storage module docs (L1)
├── layout/
│   └── mod.rs                # Layout module docs (L1)
├── tensor/
│   └── mod.rs                # Tensor module docs (L1)
├── iter/
│   └── mod.rs                # Iterator module docs (L1)
├── math/
│   └── mod.rs                # Element-wise operation module docs (L1)
├── overload/
│   └── mod.rs                # Operator-overload module docs (L1)
├── matrix/
│   └── mod.rs                # Vector dot-product module docs (L1)
├── reduction/
│   └── mod.rs                # Reduction module docs (L1)
├── broadcast.rs              # Broadcast module docs (L1, single-file module)
├── shape/
│   └── mod.rs                # Shape-operation module docs (L1)
├── index/
│   └── mod.rs                # Indexing module docs (L1)
├── construct/
│   └── mod.rs                # Constructor module docs (L1)
├── convert/
│   └── mod.rs                # Type-conversion module docs (L1)
├── set/
│   └── mod.rs                # Set-operation module docs (L1)
├── format/
│   └── mod.rs                # Output-formatting module docs (L1)
├── ffi/
│   └── mod.rs                # FFI module docs (L1)
├── workspace/
│   └── mod.rs                # Workspace module docs (L1)
├── simd/
│   └── mod.rs                # SIMD module docs (L1)
├── parallel/
│   └── mod.rs                # Parallel module docs (L1)
├── error.rs                  # Error module docs (L1)
└── prelude.rs                # Prelude docs (L1)

examples/
├── basic.rs                  # Basic-operations example
├── complex_numbers.rs        # Complex-number operations example
├── broadcasting.rs           # Broadcasting example
├── parallel.rs               # Parallel-computation example (requires `parallel` feature)
├── simd.rs                   # SIMD-acceleration example (requires `simd` feature)
└── ffi.rs                    # FFI integration example

README.md                     # Project README
CHANGELOG.md                  # Version change log
```

### 3.2 划分理由

文档与代码共存：doc comment 在源码中，CI 自动验证一致性。examples/ 独立运行。

---

## 4. 依赖关系

### 4.1 依赖图

```
29-documentation
├── depends on all design docs (00-28)
│   └── each module's docs are derived from its design doc
├── depends on `00-coding.md`
│   └── documentation style follows the coding conventions (see `00-coding.md §6`)
├── depends on `28-tests.md`
│   └── doctest / examples / docs CI validation must stay aligned
└── depends on `27-benchmark.md`
    └── performance examples, feature-gate notes, and benchmark references must stay aligned
```

### 4.2 依赖精确到类型级

| 来源             | 使用的内容                         |
| ---------------- | ---------------------------------- |
| 所有 `src/` 模块 | pub API 签名、类型定义、trait 定义 |
| `Cargo.toml`     | feature 列表、依赖列表、metadata   |
| 需求说明书       | API 行为规范、精度要求、边界定义   |

### 4.3 依赖方向声明

> **依赖方向：文档跟随代码。** 文档内容基于源码 API 签名和设计文档，不被代码依赖。

### 4.4 依赖合法性与新增依赖说明

| 项目           | 说明                                                     |
| -------------- | -------------------------------------------------------- |
| 新增第三方依赖 | 无新增依赖                                               |
| 合法性结论     | 符合最小依赖限制                                         |
| 替代方案       | 不适用；文档生成依赖 rustdoc 与现有工程配置              |

---

## 5. 文档组织结构

### 5.1 文档层次

```
L0: Crate level (lib.rs)
    └── project overview, quick start, feature list

L1: Module level (each mod.rs)
    └── module responsibilities, core concepts, type relationships

L2: Type/function level (doc comments)
    └── API docs, parameter notes, usage examples

L3: Examples (examples/)
    └── complete runnable example programs
```

### 5.2 各层覆盖要求

| 层次 | 覆盖率要求                    | 验证方式                                             |
| ---- | ----------------------------- | ---------------------------------------------------- |
| L0   | 必须存在                      | CI 检查                                              |
| L1   | 每个 pub mod 必须有模块文档   | `#![warn(missing_docs)]`                             |
| L2   | 每个 pub 项必须有 doc comment | `#![warn(missing_docs)]`（参见 `00-coding.md §6`）   |
| L3   | 关键 API 至少一个示例         | `cargo build --examples` / `cargo run --example ...` |

### 5.3 关键 API 示例覆盖矩阵

| API 族 | 必须有示例 | 对应设计文档 |
| ------ | ---------- | ------------ |
| 构造 (`zeros`/`ones`/`eye`/`from_*`) | ✅ | `18-construction` |
| 索引/切片 | ✅ | `17-indexing` |
| 转置 | ✅ | `16-shape` |
| 广播 | ✅ | `15-broadcast` |
| 逐元素运算 | ✅ | `11-math` |
| 归约 (`sum`) | ✅ | `13-reduction` |
| 内积 (`dot`) | ✅ | `12-matrix` |
| 类型转换 (`cast`) | ✅ | `21-type` |
| FFI unsafe API | ✅ | `23-ffi` |
| 运算符重载 | ✅ | `19-overload` |
| `clip`/`fill` | ✅ | `20-utility` |
| 工作空间 | ✅ | `24-workspace` |
| 格式化输出 | ✅ | `22-output` |

---

## 6. 公共 API 设计

### 6.1 lib.rs 顶层文档结构

````rust
//! # Xenon — N-dimensional Tensor Library for Rust
//!
//! Xenon is a high-performance N-dimensional array (tensor) library for Rust,
//! designed as numerical infrastructure for scientific computing.
//!
//! ## Quick Start
//!
//! ```rust
//! use xenon::prelude::*;
//!
//! // Create tensors (see 18-construction.md §5.1 for constructor signatures)
//! let a = Tensor1::<f64>::zeros(5.into());
//! let b = Tensor2::<f64>::zeros([3, 4].into());
//!
//! // Element-wise operations with broadcasting
//! let sum = (&a + &a).unwrap();
//!
//! // Reduction
//! let total = b.sum();
//! ```
//!
//! ## Features
//!
//! | Feature | Default | Description |
//! |---------|:-------:|-------------|
//! | `std` | ✓ | Standard library support |
//! | `parallel` | ✗ | Data parallelism via rayon |
//! | `simd` | ✗ | SIMD acceleration via pulp |
//!
//! ## Supported Element Types
//!
//! | Level | Types | Trait Bound |
//! |-------|-------|-------------|
//! | Base | i32, i64, f32, f64, Complex, bool | `Element` |
//! | Numeric | i32, i64, f32, f64, Complex | `Numeric: Element` |
//! | Real | f32, f64 | `RealScalar: Numeric` |
//! | Complex | Complex<f32>, Complex<f64> | `ComplexScalar: Numeric` |
//!
//! `usize` is reserved for shape and index metadata, not as a tensor element type.
//!
//! ## Memory Layout
//!
//! Default layout is **F-order (column-major)**, compatible with BLAS/LAPACK.
//!

// During development: warn level allows gradual documentation
// In CI: RUSTDOCFLAGS="-D warnings" enforces deny-level documentation
#![warn(missing_docs)]
#![warn(unsafe_op_in_unsafe_fn)]
#![warn(rustdoc::missing_crate_level_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
````

### 6.2 文档节使用规则

| 文档节        | 何时必须           | 说明                         |
| ------------- | ------------------ | ---------------------------- |
| `# Arguments` | 方法有 2+ 参数时   | 描述每个参数                 |
| `# Returns`   | 返回值非显而易见时 | 描述返回值属性               |
| `# Errors`    | 返回 Result 时     | 列出所有错误变体             |
| `# Panics`    | 可能 panic 时      | 列出所有 panic 条件          |
| `# Safety`    | unsafe 函数        | 列出安全前提条件（**必须**） |
| `# Examples`  | 所有关键 API       | 至少一个可运行示例           |
| `# See Also`  | 有相关 API 时      | 交叉引用                     |

---

## 7. #![warn(missing_docs)] 配置

### 7.1 Lint 规则

> **开发提示**：在开发期间可将 deny 改为 warn（`#![warn(missing_docs)]`），CI 中通过 `RUSTDOCFLAGS="-D warnings" cargo doc` 来强制执行文档完整性检查（参见 §13.1 CI checks）。

```rust
// lib.rs
// During development: warn level allows gradual documentation
// In CI: RUSTDOCFLAGS="-D warnings" cargo doc enforces deny-level
#![warn(missing_docs)]                        // all pub items should have documentation
#![deny(rustdoc::broken_intra_doc_links)]     // doc links must be valid
#![deny(rustdoc::private_intra_doc_links)]    // private item links are invalid
#![warn(rustdoc::missing_crate_level_docs)]   // crate-level docs must exist
#![warn(unsafe_op_in_unsafe_fn)]              // unsafe ops in unsafe fn should be documented
#![cfg_attr(docsrs, feature(doc_cfg))]        // docs.rs feature annotation
```

### 7.2 Clippy 文档 lint

```rust
// Enabled in CI
#![warn(clippy::missing_errors_doc)]      // Result functions need Errors section
#![warn(clippy::missing_panics_doc)]      // Panicking functions need Panics section
#![warn(clippy::missing_safety_doc)]      // Unsafe functions need Safety section
```

---

## 8. Doctest 规范

### 8.1 规则

| 规范       | 说明                                                                              |
| ---------- | --------------------------------------------------------------------------------- |
| 可编译运行 | 所有 doctest 通过 `cargo test --doc`；独立 examples 通过 `cargo build --examples` |
| 使用 `?`   | doctest 天然返回 `Result` 时必须优先使用 `?`；避免在文档示例中使用 `unwrap()`     |
| 隐藏样板   | 用 `# ` 隐藏 use 语句                                                             |
| 最小化     | 只展示当前 API 用法                                                               |
| 有断言     | 用 `assert_eq!` 验证结果                                                          |

### 8.2 Doctest 模板

````rust
/// Compute the sum of all elements.
///
/// # Examples
///
/// ```
/// use xenon::prelude::*;
///
/// # fn demo() -> xenon::Result<()> {
/// let t = Tensor1::from_shape_vec([3], vec![1.0, 2.0, 3.0])?;
/// assert_eq!(t.sum(), 6.0);
/// # Ok(())
/// # }
/// ```
pub fn sum(&self) -> A { ... }
````

### 8.3 Feature-gated Doctest

````rust
/// Parallel sum using rayon.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "parallel")]
/// # {
/// use xenon::prelude::*;
///
/// let t = Tensor1::<f64>::ones([1_000_000]);
/// let s = t.par_sum();
/// assert_eq!(s, 1_000_000.0);
/// # }
/// ```
#[cfg(feature = "parallel")]
pub fn par_sum(&self) -> A { ... }
````

---

## 9. examples/ 目录规划

### 9.1 示例清单

| 文件                 | 内容                                       | Feature    | 目标用户                             |
| -------------------- | ------------------------------------------ | ---------- | ------------------------------------ |
| `basic.rs`           | 创建、运算、归约、打印                     | 默认       | 新用户                               |
| `complex_numbers.rs` | 复数构造、同类型复数运算、显式转换后的运算 | 默认       | 科学计算                             |
| `broadcasting.rs`    | 广播规则、行/列/标量广播                   | 默认       | 日常使用                             |
| `parallel.rs`        | 并行计算、阈值配置                         | `parallel` | 性能优化（参见 `09-parallel.md §5`） |
| `simd.rs`            | SIMD 加速、回退策略                        | `simd`     | 性能优化（参见 `08-simd.md §5`）     |
| `ffi.rs`             | 与 C/BLAS 交互                             | 默认       | 库开发者                             |

### 9.2 示例模板

```rust
//! Example: Brief description
//!
//! Run with: `cargo run --example basic`

use xenon::prelude::*;

fn main() -> xenon::Result<()> {
    // Step 1: Create tensors
    let a = Tensor2::<f64>::zeros([3, 4]);
    println!("Created 3x4 zero matrix: shape={:?}", a.shape());

    // Step 2: Perform operation
    let b = a.t();
    println!("Transposed: shape={:?}", b.shape());

    Ok(())
}
```

### 9.3 示例编写规范

| 规范         | 说明                            |
| ------------ | ------------------------------- |
| 自包含       | 独立可运行，不依赖其他示例      |
| 有注释       | 关键步骤有行内注释              |
| 有输出       | 使用 `println!` 展示结果        |
| Feature gate | 需可选 feature 的在文件顶部注明 |
| 无 unwrap    | 使用 `?`，main 返回 `Result`    |

---

## 10. README.md 内容规划

### 10.1 结构

````markdown
# Xenon

Rust N-dimensional tensor library for scientific computing.

## Features

- N-dimensional arrays with static (0-6D) and dynamic dimensions (`IxDyn` for runtime-rank tensors)
- Column-major (F-order) default, BLAS-compatible memory layout
- Custom FFI-friendly complex number type
- Optional SIMD (pulp) and parallel (rayon) acceleration

## Quick Start

[code example]

## Installation

```toml
[dependencies]
xenon = "x.y.z"
```

## Documentation

[docs.rs link]

## License

MIT
````

---

## 11. CHANGELOG.md 规范

### 11.1 格式

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
- Element type hierarchy
- Custom complex number type with repr(C)
- Storage system (Owned, View, ViewMut, Arc)
- F-order memory layout with 64-byte alignment

### Changed

### Fixed
```

### 11.2 版本号规则

| 变更类型   | 版本号影响           |
| ---------- | -------------------- |
| 新增 API   | minor                |
| 破坏性变更 | major (1.0 后)       |
| Bug 修复   | patch                |
| 性能优化   | patch                |
| 0.x 阶段   | minor 可含破坏性变更 |

---

## 12. docs.rs 配置

### 12.1 Cargo.toml metadata

```toml
[package.metadata.docs.rs]
all-features = true
rustdoc-args = ["--cfg", "docsrs"]
```

### 12.2 Feature gate 标注

```rust
// lib.rs
#![cfg_attr(docsrs, feature(doc_cfg))]

// Each feature-gated pub item
#[cfg(feature = "parallel")]
#[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
pub fn par_sum(&self) -> A { ... }
```

---

## 13. 文档 CI 检查

### 13.1 验证项目

| 检查项   | 命令                                                            | 失败条件     |
| -------- | --------------------------------------------------------------- | ------------ |
| 缺失文档 | `RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps` | 任何 warning |
| Doctest  | `cargo test --doc --all-features`                               | 任何失败     |
| 示例编译 | `cargo build --examples --all-features`                         | 任何失败     |
| 链接检查 | `cargo doc` 无 broken links 警告                                | 无效链接     |

### 13.2 CI 配置

```yaml
# .github/workflows/docs.yml
docs:
  steps:
    - name: Check missing docs
      run: RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps

    - name: Run doctests
      run: cargo test --doc --all-features

    - name: Build examples
      run: cargo build --examples --all-features
```

---

## 14. Good / Bad 文档注释对比

### 14.1 Good — 完整的函数文档

````rust
/// Compute the sum of all elements in the tensor.
///
/// Returns the additive identity (zero) for empty tensors.
/// For floating-point types, NaN values propagate to the result.
///
/// # Examples
///
/// ```
/// use xenon::prelude::*;
///
/// # fn demo() -> xenon::Result<()> {
/// let t = Tensor1::from_shape_vec([3], vec![1.0, 2.0, 3.0])?;
/// assert_eq!(t.sum(), 6.0);
///
/// let empty = Tensor1::<f64>::zeros([0]);
/// assert_eq!(empty.sum(), 0.0);
/// # Ok(())
/// # }
/// ```
///
/// # Performance
///
/// O(n) time complexity. With `simd` feature enabled, uses SIMD
/// acceleration for contiguous data.
///
/// # See Also
///
/// * [`sum_axis`](Self::sum_axis) — sum along a specific axis
pub fn sum(&self) -> A { ... }
````

### 14.2 Bad — 不完整的函数文档

````rust
// Bad: no documentation, no examples, no description
pub fn sum(&self) -> A { ... }

// Bad: documentation too brief, missing key information
/// Sums the tensor.
pub fn sum(&self) -> A { ... }

// Bad: the example is incomplete — it omits the surrounding API description,
// return-value semantics, and edge-case notes even though the doctest itself compiles.
/// ```
/// use xenon::prelude::*;
/// # fn demo() -> xenon::Result<()> {
/// let t = Tensor1::from_shape_vec([3], vec![1.0, 2.0, 3.0])?;
/// assert_eq!(t.sum(), 6.0);
/// # Ok(())
/// # }
/// ```
pub fn sum(&self) -> A { ... }
````

### 14.3 Good — unsafe 函数文档

````rust
/// Create a tensor view from raw parts.
///
/// # Safety
///
/// The caller must ensure:
///
/// 1. `ptr` is non-null, non-dangling, and aligned to `align_of::<A>()`
/// 2. `storage_len` matches the number of elements in the backing storage reachable from `ptr`
/// 3. The memory is valid for the lifetime `'a`
/// 4. No mutable references to the memory exist
/// 5. `shape` and `strides` have the same length
/// 6. All index-calculated offsets are in bounds
/// 7. All accessible elements are properly initialized
///
/// # Examples
///
/// ```rust
/// use xenon::prelude::*;
///
/// # fn demo() -> xenon::Result<()> {
/// let data = vec![1.0f64, 2.0, 3.0, 4.0];
///
/// // SAFETY: data is non-empty, properly aligned, and outlives the view.
/// let view = unsafe {
///     TensorView::from_raw_parts::<f64, Ix2>(
///         data.as_ptr(),
///         data.len(),
///         Ix2::from_slice(&[2, 2]),
///         Strides::from_slice(&[1, 2]),
///         0,
///     )?
/// };
/// assert_eq!(view.shape(), &[2, 2]);
/// # Ok(())
/// # }
/// ```
pub unsafe fn from_raw_parts<'a, A, D>(
    ptr: *const A,
    storage_len: usize,
    shape: D,
    strides: Strides<D>,
    offset: usize,
) -> Result<TensorView<'a, A, D>, XenonError>
where
    A: Element,
    D: Dimension
````

### 14.4 Bad — 缺少 Safety 节

```rust
// Bad: unsafe function has no Safety documentation
/// Create a tensor view from raw parts.
pub unsafe fn from_raw_parts<'a, A, D>(...) -> TensorView<'a, A, D>
```

---

## 15. 内部实现设计

### 15.1 文档生成流程

````
Doc comments in the source code
    │
    ├── cargo doc → rustdoc → HTML docs
    │       ├── parse Markdown
    │       ├── validate intra-doc links
    │       └── generate docs.rs-compatible output
    │
    └── cargo test --doc → rustdoc --test
            ├── extract ```rust ``` code blocks
            ├── compile them as standalone executables
            └── run them and verify assertions
````

### 15.2 文档覆盖率计算

```bash
# Check for missing docs at deny level
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps

# Count undocumented pub items (manual audit)
# 1. cargo doc --no-deps 2>&1 | grep "missing documentation"
# 2. Ensure zero warnings
```

### 15.3 doc comment 编写工作流

| 步骤                | 操作                        | 验证                              |
| ------------------- | --------------------------- | --------------------------------- |
| 1. 编写 API         | 实现函数/类型               | `cargo check`                     |
| 2. 添加 doc comment | 描述、参数、返回值、示例    | `cargo doc` 无 warning            |
| 3. 添加 doctest     | `# Examples` 节             | `cargo test --doc` 通过           |
| 4. Safety 文档      | unsafe 函数的 `# Safety` 节 | `clippy::missing_safety_doc` 通过 |

---

## 16. 测试计划

### 16.1 测试分类表

| 类型         | 命令                                                            | 目的                                              |
| ------------ | --------------------------------------------------------------- | ------------------------------------------------- |
| 单元检查     | `cargo test --doc --all-features`                               | 验证单个 API 文档示例可编译运行                   |
| 集成检查     | `cargo doc --all-features --no-deps` + examples 构建            | 验证模块文档、README、examples 与源码接口协同一致 |
| 边界检查     | feature-gated/unsafe doctest 逐项编译                           | 验证条件编译、unsafe 说明和 `std` 环境边界        |
| 属性检查     | 文档覆盖率目标 + broken links / missing docs 不变量             | 验证“公开 API 均有文档、关键入口均可追踪”         |
| Doctest      | `cargo test --doc --all-features`                               | 验证文档中的代码示例可编译运行                    |
| 缺失文档检查 | `RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps` | 确保所有 pub API 有文档                           |
| 示例编译     | `cargo build --examples --all-features`                         | 验证 examples/ 下程序可编译运行                   |
| 链接检查     | `cargo doc` 无 broken links 警告                                | 确保文档内交叉引用有效                            |
| CI 门禁      | `missing_docs` lint deny 级别                                   | 阻止无文档代码合入                                |

### 16.2 Doctest 覆盖率目标

| 模块                                             | 目标覆盖率 | 说明                                                              |
| ------------------------------------------------ | ---------- | ----------------------------------------------------------------- |
| 核心类型（tensor, dimension, storage）           | ≥80%       | 核心入口和高频查询方法必须有 doctest，其余 pub 方法至少有文档说明 |
| 运算模块（overload, math, broadcast, reduction） | ≥80%       | 核心入口、广播/错误路径和代表性运算必须有 doctest                 |
| 工具模块（ffi, workspace, simd, parallel）       | ≥80%       | 关键 API 有 doctest                                               |
| 辅助模块（convert, format, error）               | ≥60%       | 至少构造和基本使用有 doctest                                      |
| 迭代与归约模块（iter, reduction, matrix）        | ≥80%       | 核心入口、边界行为和错误路径有 doctest                            |

### 16.3 边界测试场景表

| 场景              | 预期行为                                                              |
| ----------------- | --------------------------------------------------------------------- |
| feature-gated API | 在未启用 feature 时不会出现在文档中，启用后 doctest 通过              |
| `std` 平台边界    | 文档示例默认以 `std` 环境为前提；feature-gated API 需显式标注启用条件 |
| unsafe API 文档   | 必须包含 `# Safety` 且示例不省略关键前置条件                          |
| 大型数组输出示例  | 截断格式与 `22-output.md` 保持一致                                    |

### 16.4 属性测试不变量

| 不变量                               | 验证方式                          |
| ------------------------------------ | --------------------------------- |
| 所有公开 API 都能在 docs.rs 中被发现 | `missing_docs` + docs.rs 构建检查 |
| 所有关键模块都有至少一个可运行示例   | doctest / examples 构建联合验证   |
| 文档中的路径与模块名和架构文档一致   | broken links 检查 + 人工审阅      |

### 16.5 CI 配置

```yaml
# .github/workflows/docs.yml
docs:
  steps:
    - name: Check missing docs
      run: RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps

    - name: Run doctests
      run: cargo test --doc --all-features

    - name: Build examples
      run: cargo build --examples --all-features
```

### 16.6 Feature gate / 配置测试

| 配置        | 验证点                                                        |
| ----------- | ------------------------------------------------------------- |
| 默认配置    | 默认 `std` 文档、README 与 examples 描述一致                  |
| 启用并行    | `parallel` API 的 `doc(cfg)`、doctest 与示例说明保持一致      |
| 启用 SIMD   | `simd` API 的 `doc(cfg)`、doctest 与示例说明保持一致          |
| 全 feature  | docs.rs 构建、doctest 与 examples 在组合配置下均通过          |

### 16.7 类型边界 / 编译期测试

| 场景                         | 测试方式                                            |
| ---------------------------- | --------------------------------------------------- |
| `unsafe fn` 的 `# Safety` 节 | rustdoc/clippy 文档 lint + `cargo doc` 校验         |
| feature-gated API 可见性     | docs.rs 构建与 `doc(cfg)` 检查                      |
| 公共 API 文档覆盖边界         | `missing_docs` / broken link 检查                   |

---

## 错误处理与语义边界

本文档不直接定义错误类型，但要求所有文档示例、`# Errors` 节、panic 说明与 feature-gated 文档行为统一遵循 `26-error.md` 的错误语义边界；文档层负责准确转述，不重新定义公开错误模型。

---

## 17. 与其他模块的交互

### 17.1 文档对被文档模块的依赖

| 文档任务            | 依赖的模块                                                | 说明                   |
| ------------------- | --------------------------------------------------------- | ---------------------- |
| T1 (lib.rs 文档)    | 全部                                                      | 需要了解所有模块的概览 |
| T5 (核心模块文档)   | dimension, element, complex, storage, layout              | 基于 §1 章节编写       |
| T6 (张量与运算文档) | tensor, overload, broadcast, shape, index, construct, set | 基于 §1 章节编写       |
| T7 (基础设施文档)   | ffi, workspace, simd, parallel, error, prelude            | 基于 §1 章节编写       |
| T8 (类型级文档)     | 全部                                                      | 逐类型添加 doc comment |
| T9 (函数级文档)     | 全部                                                      | 逐函数添加 doc comment |

### 17.2 数据流

````
Design docs (00-28)
    │
    ├── extract module responsibilities, core concepts, and API signatures
    │       │
    │       └── write module docs into each mod.rs (`//!`)
    │
    └── extract type definitions and method signatures
            │
            ├── write doc comments (`///`)
            └── write doctests (```rust ```)
                    │
                    └── validate with `cargo test --doc`
````

---

## 18. 实现任务拆分

### Wave 1: Crate 级文档

- [ ] **T1**: 编写 lib.rs 顶层 crate 文档
  - 文件: `src/lib.rs`
  - 内容: 项目概述、Quick Start、Features 表、元素类型表、内存布局说明
  - 测试: `cargo doc --no-deps` 无 warning
  - 前置: 无
  - 预计: 10 min

- [ ] **T2**: 配置 `#![warn(missing_docs)]` 和 docs.rs metadata
  - 文件: `src/lib.rs`, `Cargo.toml`
  - 内容: lint 规则、`[package.metadata.docs.rs]`、`cfg_attr(docsrs, ...)`
  - 测试: 编译通过
  - 前置: T1
  - 预计: 5 min

- [ ] **T3**: 编写 README.md
  - 文件: `README.md`
  - 内容: 项目介绍、Features、Quick Start、安装、文档链接、许可证
  - 测试: 内容完整
  - 前置: T1
  - 预计: 10 min

- [ ] **T4**: 创建 CHANGELOG.md
  - 文件: `CHANGELOG.md`
  - 内容: Keep a Changelog 格式，初始版本条目
  - 测试: 格式正确
  - 前置: 无
  - 预计: 5 min

### Wave 2: 模块级文档（可并行）

- [ ] **T5**: 编写核心模块文档（dimension, element, complex, storage, layout）
  - 文件: 各 `mod.rs`
- 内容: 模块职责、核心概念、使用示例、依赖图、设计决策（参见 `02-dimension.md §1`、`03-element.md §1`、`04-complex.md §1`、`05-storage.md §1`、`06-layout.md §1`）
  - 测试: `cargo doc --no-deps` 无 warning
  - 前置: T2
  - 预计: 10 min

- [ ] **T6**: 编写张量与运算模块文档（tensor, iter, math, overload, broadcast, reduction, matrix, shape, index, construct, set）
  - 文件: 各 `mod.rs`
  - 内容: 模块职责、核心类型、运算分类、类型约束速查（参见 `07-tensor.md §1`、`10-iterator.md §1`、`11-math.md §1`、`12-matrix.md §1`、`13-reduction.md §1`、`15-broadcast.md §1`、`16-shape.md §1`）
  - 测试: `cargo doc --no-deps` 无 warning
  - 前置: T2
  - 预计: 10 min

- [ ] **T7**: 编写基础设施模块文档（ffi, workspace, simd, parallel, error, prelude, convert, format）
  - 文件: 各 `mod.rs`
  - 内容: 模块职责、Safety 约定、feature gate 说明、转换与输出语义（参见 `23-ffi.md §1`、`24-workspace.md §1`、`08-simd.md §1`、`09-parallel.md §1`、`21-type.md §1`、`22-output.md §1`、`26-error.md §1`）
  - 测试: `cargo doc --no-deps` 无 warning
  - 前置: T2
  - 预计: 10 min

### Wave 3: 类型/函数级文档（可并行）

- [ ] **T8a**: tensor 模块公共 API 文档
  - 文件: `src/tensor/mod.rs` 及相关文件
  - 内容: TensorBase, Tensor, TensorView, TensorViewMut, ArcTensor 类型文档
  - 测试: `cargo doc --no-deps` 无 warning
  - 前置: T5, T6, T7
  - 预计: 10 min

- [ ] **T8b**: dimension 模块文档
  - 文件: `src/dimension/mod.rs`
  - 内容: Ix0~Ix6, IxDyn, Dimension trait 文档
  - 测试: `cargo doc --no-deps` 无 warning
  - 前置: T5, T6, T7
  - 预计: 10 min

- [ ] **T8c**: element 模块文档
  - 文件: `src/element/mod.rs`
  - 内容: Element, Numeric, RealScalar, ComplexScalar trait 文档
  - 测试: `cargo doc --no-deps` 无 warning
  - 前置: T5, T6, T7
  - 预计: 10 min

- [ ] **T8d**: storage 模块文档
  - 文件: `src/storage/mod.rs`
  - 内容: Owned, ViewRepr, StorageMut trait 文档
  - 测试: `cargo doc --no-deps` 无 warning
  - 前置: T5, T6, T7
  - 预计: 10 min

- [ ] **T8e**: layout 模块文档
  - 文件: `src/layout/mod.rs`
  - 内容: LayoutFlags, compute_f_strides 文档
  - 测试: `cargo doc --no-deps` 无 warning
  - 前置: T5, T6, T7
  - 预计: 10 min

- [ ] **T9a**: math 模块逐元素运算文档
  - 文件: `src/math/` 下相关文件
  - 内容: add, sub, mul, div, sin, sqrt, exp, ln, abs 等逐元素运算函数文档和 doctest
  - 测试: `cargo test --doc --all-features`
  - 前置: T5, T6, T7
  - 预计: 10 min

- [ ] **T9b**: reduction 与 matrix 模块文档
  - 文件: `src/reduction/`, `src/matrix/` 下相关文件
  - 内容: sum, sum_axis, dot 等归约/内积函数文档和 doctest
  - 测试: `cargo test --doc --all-features`
  - 前置: T5, T6, T7
  - 预计: 10 min

- [ ] **T9c**: broadcast 和 shape 模块文档
  - 文件: `src/broadcast.rs`, `src/shape/mod.rs`
  - 内容: broadcast_shape, transpose 函数文档和 doctest
  - 测试: `cargo test --doc --all-features`
  - 前置: T5, T6, T7
  - 预计: 10 min

- [ ] **T9d**: construct 和 set 模块文档
  - 文件: `src/construct/mod.rs`, `src/set/mod.rs`
- 内容: zeros, ones, eye, from_shape_vec, unique 函数文档和 doctest（`full` 当前版本未提供）
  - 测试: `cargo test --doc --all-features`
  - 前置: T5, T6, T7
  - 预计: 10 min

- [ ] **T9e**: ffi, workspace, error 模块文档
  - 文件: `src/ffi/mod.rs`, `src/workspace/mod.rs`, `src/error.rs`
  - 内容: FFI 函数（含 Safety 节）、Workspace、XenonError 文档和 doctest
  - 测试: `cargo test --doc --all-features`
  - 前置: T5, T6, T7
  - 预计: 10 min

- [ ] **T9f**: iter, convert, format, overload 模块文档
  - 文件: `src/iter/mod.rs`, `src/convert/mod.rs`, `src/format/mod.rs`, `src/overload/mod.rs`
  - 内容: 迭代器入口、类型转换、输出格式化、运算符语法边界的模块文档和 doctest
  - 测试: `cargo test --doc --all-features`
  - 前置: T5, T6, T7
  - 预计: 10 min

### Wave 4: 示例程序（可并行）

- [ ] **T10**: 编写 examples/basic.rs
  - 文件: `examples/basic.rs`
  - 内容: 创建、运算、归约、打印
  - 测试: `cargo run --example basic`
  - 前置: T1
  - 预计: 10 min

- [ ] **T11**: 编写 examples/complex_numbers.rs
  - 文件: `examples/complex_numbers.rs`
  - 内容: 复数构造、同类型复数算术、显式转换后的运算
  - 测试: `cargo run --example complex_numbers`
  - 前置: T1
  - 预计: 10 min

- [ ] **T12**: 编写 examples/broadcasting.rs
  - 文件: `examples/broadcasting.rs`
  - 内容: 广播规则、行/列/标量广播
  - 测试: `cargo run --example broadcasting`
  - 前置: T1
  - 预计: 10 min

- [ ] **T13**: 编写 examples/parallel.rs
  - 文件: `examples/parallel.rs`
  - 内容: 并行计算、阈值配置
  - 测试: `cargo run --example parallel --features parallel`
  - 前置: T1
  - 预计: 10 min

- [ ] **T14**: 编写 examples/simd.rs
  - 文件: `examples/simd.rs`
  - 内容: SIMD 加速、回退策略
  - 测试: `cargo run --example simd --features simd`
  - 前置: T1
  - 预计: 10 min

- [ ] **T15**: 编写 examples/ffi.rs
  - 文件: `examples/ffi.rs`
  - 内容: 与 C/BLAS 交互
  - 测试: `cargo run --example ffi`
  - 前置: T1
  - 预计: 10 min

- [ ] **T16**: 校验示例与 crate 文档仅声明 `std` 环境
  - 文件: `src/lib.rs`, `README.md`, `examples/`
  - 内容: 清理超范围的平台说明，确保示例与文档默认面向 `std` 环境
  - 测试: `cargo doc --no-deps` 与 `cargo build --examples --all-features`
  - 前置: T1
  - 预计: 10 min

### Wave 5: CI 集成

- [ ] **T17**: 配置 CI 文档验证工作流
  - 文件: `.github/workflows/docs.yml`
  - 内容: missing docs 检查、doctest、示例编译
  - 测试: CI 触发运行
  - 前置: T1-T16
  - 预计: 10 min

### 并行执行分组图

```
Wave 1: [T1] [T4]
            │
Wave 2: [T2] [T3]
            │
Wave 3: [T5] [T6] [T7]
            │
Wave 4: [T8a] [T8b] [T8c] [T8d] [T8e] [T9a] [T9b] [T9c] [T9d] [T9e] [T9f]
            │
Wave 5: [T10] [T11] [T12] [T13] [T14] [T15] [T16]
            │
Wave 6: [T17]
```

---

## 19. 设计决策记录

### 决策 1：英文文档

| 属性     | 值                                                              |
| -------- | --------------------------------------------------------------- |
| 决策     | 所有 doc comment 和 README 使用英文                             |
| 理由     | Rust 生态惯例；docs.rs 面向全球开发者（参见 `00-coding.md §6`） |
| 替代方案 | 中文文档 — 放弃，不符合 Rust 社区惯例                           |

### 决策 2：doctest 统一使用 `?`

| 属性     | 值                                                                                 |
| -------- | ---------------------------------------------------------------------------------- |
| 决策     | doctest 在示例天然返回 `Result` 时统一使用 `?`；不再为最小示例保留 `unwrap()` 例外 |
| 理由     | 同时遵循 Rust API Guidelines C-QUESTION-MARK，并避免为纯展示型示例引入多余样板     |
| 替代方案 | 完全禁止 unwrap — 放弃，对最小示例过于僵硬                                         |

### 决策 3：开发期间 `#![warn(missing_docs)]`，CI 中 deny

| 属性     | 值                                                                                                        |
| -------- | --------------------------------------------------------------------------------------------------------- |
| 决策     | 开发期间使用 `warn` 级别，CI 中通过 `RUSTDOCFLAGS="-D warnings"` 强制 deny 级别                           |
| 理由     | 需求说明书 §28.1 要求所有公开 API 有文档；开发期间 warn 允许渐进式补全文档，CI 中 deny 阻止无文档代码合入 |
| 替代方案 | 始终 deny 级别 — 放弃，开发期间过于严格，阻碍快速迭代                                                     |

### 决策 4：按模块组织模块级文档

| 属性     | 值                                                    |
| -------- | ----------------------------------------------------- |
| 决策     | 每个模块的 mod.rs 包含完整的模块概述                  |
| 理由     | 用户从 docs.rs 进入模块时能快速理解模块定位和核心类型 |
| 替代方案 | 仅函数级文档 — 放弃，缺乏模块整体视图                 |

### 决策 5：examples 按场景而非按模块

| 属性     | 值                                                         |
| -------- | ---------------------------------------------------------- |
| 决策     | examples/ 按使用场景（basic/broadcasting/parallel 等）组织 |
| 理由     | 用户按需求查找示例，而非按源码模块                         |
| 替代方案 | 按源码模块组织 — 放弃，不便于用户理解实际用法              |

---

## 20. 平台与工程约束

| 约束项     | 约束内容                                                  |
| ---------- | --------------------------------------------------------- |
| 平台支持   | 文档、doctest 与 examples 默认面向 `std` 环境             |
| crate 结构 | 文档产物围绕当前单 crate 组织，不维护额外平台模板工程     |
| 依赖约束   | 仅文档化现有 feature 与依赖，不扩展超出需求范围的工程契约 |

---

## 版本历史

| 版本  | 日期       |
| ----- | ---------- |
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-08 |
| 1.0.2 | 2026-04-08 |
| 1.0.3 | 2026-04-08 |
| 1.1.0 | 2026-04-08 |
| 1.1.1 | 2026-04-08 |
| 1.1.2 | 2026-04-10 |
| 1.1.3 | 2026-04-10 |
| 1.1.4 | 2026-04-14 |
| 1.1.5 | 2026-04-15 |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
