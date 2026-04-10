# 文档模块设计

> 文档编号: 29 | 模块: 全局 | 阶段: Phase 6
> 前置文档: 所有前置文档（`00-coding.md` ~ `28-tests.md`）
> 需求参考: 需求说明书 §28.1

---

## 1. 模块定位

### 1.1 职责边界

| 职责 | 包含 | 不包含 |
|------|------|--------|
| API 文档 | 所有 pub 类型和函数的 doc comment | 内部实现注释（非 pub） |
| 使用示例 | 关键 API 的可运行代码示例（doctest） | 完整教程、视频教程 |
| Safety 说明 | 所有 unsafe 函数的 `# Safety` 文档节（参见 `00-coding.md §5`） | 安全函数的 Safety 节 |
| Crate 级文档 | lib.rs 顶层文档、README、CHANGELOG | 第三方博客文章 |
| 模块级文档 | 各 mod.rs 的 `//!` 模块概述 | 内部实现文档 |
| examples/ | 独立可运行示例程序 | 交互式 notebook |
| docs.rs 配置 | metadata、feature gate 标注 | 自定义文档主题 |

### 1.2 设计原则

| 原则 | 体现 |
|------|------|
| 全覆盖 | 所有 pub API 必须有 doc comment（参见 `00-coding.md §6`） |
| 可测试 | 关键 API 的示例通过 doctest 或独立 examples 验证 |
| 安全性透明 | 所有 unsafe 函数有 `# Safety` 节 |
| 惯用法 | 遵循 Rust API Guidelines |
| 英文文档 | 所有 doc comment 使用英文 |

### 1.3 在架构中的位置

```
依赖层级：
L0: error, private
L1: dimension, element, complex
L2: layout (依赖 dimension)
L3: storage (独立于 layout，由 tensor 持有并消费 layout 结果)
L4: tensor (依赖 storage, dimension)
L5: overload/, iter/, index/, shape/, broadcast.rs, construct/, ffi/, convert/, format/

横切关注点（全局）：
┌─────────────────────────────────────────────────────────────────┐
│  文档 (doc comments, README, examples/)  ← 当前文档（全局）         │
│  ─ 横贯所有 L0-L5 模块的 pub API 文档                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 文件位置

### 2.1 文档源码分布

```
src/
├── lib.rs                    # Crate 级文档（L0）
├── dimension/
│   └── mod.rs                # 维度模块文档（L1）
├── element/
│   └── mod.rs                # 元素类型模块文档（L1）
├── complex/
│   └── mod.rs                # 复数模块文档（L1）
├── storage/
│   └── mod.rs                # 存储模块文档（L1）
├── layout/
│   └── mod.rs                # 布局模块文档（L1）
├── tensor/
│   └── mod.rs                # 张量模块文档（L1）
├── overload/
│   └── mod.rs                # 运算符重载模块文档（L1）
├── broadcast.rs
│   └── mod.rs                # 广播模块文档（L1）
├── shape/
│   └── mod.rs                # 形状操作模块文档（L1）
├── index/
│   └── mod.rs                # 索引模块文档（L1）
├── construct/
│   └── mod.rs                # 构造模块文档（L1）
├── set/
│   └── mod.rs                # 集合操作模块文档（L1）
├── ffi/
│   └── mod.rs                # FFI 模块文档（L1）
├── workspace/
│   └── mod.rs                # 工作空间模块文档（L1）
├── simd/
│   └── mod.rs                # SIMD 模块文档（L1）
├── parallel/
│   └── mod.rs                # 并行模块文档（L1）
├── error.rs                  # 错误模块文档（L1）
└── prelude.rs                # Prelude 文档（L1）

examples/
├── basic.rs                  # 基础操作示例
├── complex_numbers.rs        # 复数运算示例
├── broadcasting.rs           # 广播机制示例
├── parallel.rs               # 并行计算示例（需 parallel feature）
├── simd.rs                   # SIMD 加速示例（需 simd feature）
├── templates/
│   └── no_std-app/           # no_std + alloc 示例模板工程
└── ffi.rs                    # FFI 集成示例

README.md                     # 项目 README
CHANGELOG.md                  # 版本变更记录
```

### 2.2 划分理由

文档与代码共存：doc comment 在源码中，CI 自动验证一致性。examples/ 独立运行。

---

## 3. 依赖关系

### 3.1 依赖图

```
29-documentation
├── 依赖所有模块设计文档（00-28）
│   └── 每个模块的文档内容基于其设计文档
├── 依赖 `00-coding.md`
│   └── 文档风格遵循编码规范（参见 `00-coding.md §6`）
├── 被 28-integration-tests 依赖
│   └── doctest 也是测试的一部分（参见 `28-tests.md §11`）
└── 被 27-benchmark 依赖
    └── benchmark 文档引用性能相关 API 文档（参见 `27-benchmark.md §10` 与 `§12`）
```

### 3.2 依赖精确到类型级

| 来源 | 使用的内容 |
|------|-----------|
| 所有 `src/` 模块 | pub API 签名、类型定义、trait 定义 |
| `Cargo.toml` | feature 列表、依赖列表、metadata |
| 需求说明书 | API 行为规范、精度要求、边界定义 |

### 3.3 依赖方向声明

> **依赖方向：文档跟随代码。** 文档内容基于源码 API 签名和设计文档，不被代码依赖。

---

## 4. 文档组织结构

### 4.1 文档层次

```
L0: Crate 级 (lib.rs)
    └── 项目概述、快速入门、feature 列表

L1: 模块级 (各 mod.rs)
    └── 模块职责、核心概念、类型关系

L2: 类型/函数级 (doc comments)
    └── API 文档、参数说明、使用示例

L3: 示例 (examples/)
    └── 完整可运行示例程序
```

### 4.2 各层覆盖要求

| 层次 | 覆盖率要求 | 验证方式 |
|------|-----------|----------|
| L0 | 必须存在 | CI 检查 |
| L1 | 每个 pub mod 必须有模块文档 | `#![warn(missing_docs)]` |
| L2 | 每个 pub 项必须有 doc comment | `#![warn(missing_docs)]`（参见 `00-coding.md §6`） |
| L3 | 关键 API 至少一个示例 | `cargo build --examples` / `cargo run --example ...` |

---

## 5. 公共 API 设计

### 5.1 lib.rs 顶层文档结构

```rust
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
//! // Create tensors (see 18-construction.md §4.1 for constructor signatures)
//! let a = Tensor1::<f64>::zeros(5.into());
//! let b = Tensor2::<f64>::zeros([3, 4].into());
//!
//! // Element-wise operations with broadcasting
//! let sum = &a + &a;
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
//! | Base | i32, i64, f32, f64, Complex, bool, usize | `Element` |
//! | Numeric | i32, i64, f32, f64, Complex | `Numeric: Element` |
//! | Real | f32, f64 | `RealScalar: Numeric` |
//! | Complex | Complex<f32>, Complex<f64> | `ComplexScalar: Numeric` |
//!
//! ## Memory Layout
//!
//! Default layout is **F-order (column-major)**, compatible with BLAS/LAPACK.
//!
//! ## no_std Support
//!
//! Xenon supports `no_std` environments (requires `alloc` crate).

// During development: warn level allows gradual documentation
// In CI: RUSTDOCFLAGS="-D warnings" enforces deny-level documentation
#![warn(missing_docs)]
#![warn(unsafe_op_in_unsafe_fn)]
#![warn(rustdoc::missing_crate_level_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
```

### 5.2 文档节使用规则

| 文档节 | 何时必须 | 说明 |
|--------|---------|------|
| `# Arguments` | 方法有 2+ 参数时 | 描述每个参数 |
| `# Returns` | 返回值非显而易见时 | 描述返回值属性 |
| `# Errors` | 返回 Result 时 | 列出所有错误变体 |
| `# Panics` | 可能 panic 时 | 列出所有 panic 条件 |
| `# Safety` | unsafe 函数 | 列出安全前提条件（**必须**） |
| `# Examples` | 所有关键 API | 至少一个可运行示例 |
| `# See Also` | 有相关 API 时 | 交叉引用 |

---

## 6. #![warn(missing_docs)] 配置

### 6.1 Lint 规则

> **开发提示**：在开发期间可将 deny 改为 warn（`#![warn(missing_docs)]`），CI 中通过 `RUSTDOCFLAGS="-D warnings" cargo doc` 来强制执行文档完整性检查（参见 §12.1 CI checks）。

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

### 6.2 Clippy 文档 lint

```rust
// Enabled in CI
#![warn(clippy::missing_errors_doc)]      // Result functions need Errors section
#![warn(clippy::missing_panics_doc)]      // Panicking functions need Panics section
#![warn(clippy::missing_safety_doc)]      // Unsafe functions need Safety section
```

---

## 7. Doctest 规范

### 7.1 规则

| 规范 | 说明 |
|------|------|
| 可编译运行 | 所有 doctest 通过 `cargo test --doc`；独立 examples 通过 `cargo build --examples` |
| 使用 `?` | 使用 `?` 而非 `unwrap()`（C-QUESTION-MARK） |
| 隐藏样板 | 用 `# ` 隐藏 use 语句 |
| 最小化 | 只展示当前 API 用法 |
| 有断言 | 用 `assert_eq!` 验证结果 |

### 7.2 Doctest 模板

```rust
/// Compute the sum of all elements.
///
/// # Examples
///
/// ```
/// use xenon::prelude::*;
///
/// let t = Tensor1::from_shape_vec([3], vec![1.0, 2.0, 3.0]).unwrap();
/// assert_eq!(t.sum(), 6.0);
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
/// use xenon::prelude::*;
///
/// let t = Tensor1::<f64>::ones([1_000_000]);
/// let s = t.par_sum();
/// assert_eq!(s, 1_000_000.0);
/// # }
/// ```
#[cfg(feature = "parallel")]
pub fn par_sum(&self) -> A { ... }
```

---

## 8. examples/ 目录规划

### 8.1 示例清单

| 文件 | 内容 | Feature | 目标用户 |
|------|------|---------|---------|
| `basic.rs` | 创建、运算、归约、打印 | 默认 | 新用户 |
| `complex_numbers.rs` | 复数构造、运算、混合运算 | 默认 | 科学计算 |
| `broadcasting.rs` | 广播规则、行/列/标量广播 | 默认 | 日常使用 |
| `parallel.rs` | 并行计算、阈值配置 | `parallel` | 性能优化（参见 `09-parallel.md §4`） |
| `simd.rs` | SIMD 加速、回退策略 | `simd` | 性能优化（参见 `08-simd.md §4`） |
| `templates/no_std-app/` | `no_std + alloc` 使用模式模板工程 | 默认关闭 `std` | 系统/嵌入式开发者 |
| `ffi.rs` | 与 C/BLAS 交互 | 默认 | 库开发者 |

### 8.2 示例模板

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

### 8.3 示例编写规范

| 规范 | 说明 |
|------|------|
| 自包含 | 独立可运行，不依赖其他示例 |
| 有注释 | 关键步骤有行内注释 |
| 有输出 | 使用 `println!` 展示结果 |
| Feature gate | 需可选 feature 的在文件顶部注明 |
| 无 unwrap | 使用 `?`，main 返回 `Result` |

---

## 9. README.md 内容规划

### 9.1 结构

```markdown
# Xenon

Rust N-dimensional tensor library for scientific computing.

## Features
- N-dimensional arrays with static (0-6D) and dynamic dimensions
- Column-major (F-order) default, BLAS-compatible memory layout
- Custom FFI-friendly complex number type
- Optional SIMD (pulp) and parallel (rayon) acceleration
- no_std support (requires alloc)

## Quick Start
[代码示例]

## Installation
```toml
[dependencies]
xenon = "0.1"
```

## Documentation
[docs.rs 链接]

## License
MIT
```

---

## 10. CHANGELOG.md 规范

### 10.1 格式

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

### 10.2 版本号规则

| 变更类型 | 版本号影响 |
|----------|-----------|
| 新增 API | minor |
| 破坏性变更 | major (1.0 后) |
| Bug 修复 | patch |
| 性能优化 | patch |
| 0.x 阶段 | minor 可含破坏性变更 |

---

## 11. docs.rs 配置

### 11.1 Cargo.toml metadata

```toml
[package.metadata.docs.rs]
all-features = true
rustdoc-args = ["--cfg", "docsrs"]
```

### 11.2 Feature gate 标注

```rust
// lib.rs
#![cfg_attr(docsrs, feature(doc_cfg))]

// Each feature-gated pub item
#[cfg(feature = "parallel")]
#[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
pub fn par_sum(&self) -> A { ... }
```

---

## 12. 文档 CI 检查

### 12.1 验证项目

| 检查项 | 命令 | 失败条件 |
|--------|------|----------|
| 缺失文档 | `RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps` | 任何 warning |
| Doctest | `cargo test --doc --all-features` | 任何失败 |
| 示例编译 | `cargo build --examples --all-features` | 任何失败 |
| 链接检查 | `cargo doc` 无 broken links 警告 | 无效链接 |

### 12.2 CI 配置

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

## 13. Good / Bad 文档注释对比

### 13.1 Good — 完整的函数文档

```rust
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
/// let t = Tensor1::from_shape_vec([3], vec![1.0, 2.0, 3.0]).unwrap();
/// assert_eq!(t.sum(), 6.0);
///
/// let empty = Tensor1::<f64>::zeros([0]);
/// assert_eq!(empty.sum(), 0.0);
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
```

### 13.2 Bad — 不完整的函数文档

```rust
// Bad: no documentation, no examples, no description
pub fn sum(&self) -> A { ... }

// Bad: documentation too brief, missing key information
/// Sums the tensor.
pub fn sum(&self) -> A { ... }

// Bad: example includes unnecessary Ok::<(), xenon::XenonError>(()) return type annotation,
// which is only needed when the doctest uses the ? operator.
// Since sum() does not return Result, the annotation is misleading.
/// ```
/// use xenon::prelude::*;
/// let t = Tensor1::from_shape_vec([3], vec![1.0, 2.0, 3.0]).unwrap();
/// assert_eq!(t.sum(), 6.0);
/// # Ok::<(), xenon::XenonError>(())  // Unnecessary: no ? operator used
/// ```
pub fn sum(&self) -> A { ... }
```

### 13.3 Good — unsafe 函数文档

```rust
/// Create a tensor view from raw parts.
///
/// # Safety
///
/// The caller must ensure:
///
/// 1. `ptr` is non-null, non-dangling, and aligned to `align_of::<A>()`
/// 2. The memory range covers all accessible elements
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
/// let data = vec![1.0f64, 2.0, 3.0, 4.0];
///
/// // SAFETY: data is non-empty, properly aligned, and outlives the view.
/// let view = unsafe {
///     TensorView::from_raw_parts(
///         data.as_ptr(),
///         Ix2::from_slice(&[2, 2]),
///         Ix2::from_isize_slice(&[1, 2]),
///         0,
///     )
/// };
/// assert_eq!(view.shape(), &[2, 2]);
/// ```
pub unsafe fn from_raw_parts<'a, A, D>(...) -> TensorView<'a, A, D>
```

### 13.4 Bad — 缺少 Safety 节

```rust
// Bad: unsafe function has no Safety documentation
/// Create a tensor view from raw parts.
pub unsafe fn from_raw_parts<'a, A, D>(...) -> TensorView<'a, A, D>
```

---

## 14. 内部实现

### 14.1 文档生成流程

```
源码中的 doc comments
    │
    ├── cargo doc → rustdoc → HTML 文档
    │       ├── 解析 markdown
    │       ├── 验证 intra-doc links
    │       └── 生成 docs.rs 兼容输出
    │
    └── cargo test --doc → rustdoc --test
            ├── 提取 ```rust ``` 代码块
            ├── 编译为独立可执行文件
            └── 运行并验证断言
```

### 14.2 文档覆盖率计算

```bash
# Check for missing docs at deny level
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps

# Count undocumented pub items (manual audit)
# 1. cargo doc --no-deps 2>&1 | grep "missing documentation"
# 2. Ensure zero warnings
```

### 14.3 doc comment 编写工作流

| 步骤 | 操作 | 验证 |
|------|------|------|
| 1. 编写 API | 实现函数/类型 | `cargo check` |
| 2. 添加 doc comment | 描述、参数、返回值、示例 | `cargo doc` 无 warning |
| 3. 添加 doctest | `# Examples` 节 | `cargo test --doc` 通过 |
| 4. Safety 文档 | unsafe 函数的 `# Safety` 节 | `clippy::missing_safety_doc` 通过 |

---

## 15. 测试计划

### 15.1 测试分类

| 类型 | 命令 | 目的 |
|------|------|------|
| Doctest | `cargo test --doc --all-features` | 验证文档中的代码示例可编译运行 |
| 缺失文档检查 | `RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps` | 确保所有 pub API 有文档 |
| 示例编译 | `cargo build --examples --all-features` | 验证 examples/ 下程序可编译运行 |
| 链接检查 | `cargo doc` 无 broken links 警告 | 确保文档内交叉引用有效 |
| CI 门禁 | `missing_docs` lint deny 级别 | 阻止无文档代码合入 |

### 15.2 Doctest 覆盖率目标

| 模块 | 目标覆盖率 | 说明 |
|------|-----------|------|
| 核心类型（tensor, dimension, storage） | 100% | 所有 pub 方法有 doctest |
| 运算模块（overload, math, broadcast, reduction） | 100% | 所有 pub 方法有 doctest |
| 工具模块（ffi, workspace, simd, parallel） | ≥80% | 关键 API 有 doctest |
| 辅助模块（convert, format, error） | ≥60% | 至少构造和基本使用有 doctest |
| 迭代与归约模块（iter, reduction, matrix） | ≥80% | 核心入口、边界行为和错误路径有 doctest |

### 15.3 CI 配置

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

## 16. 与其他模块的交互

### 16.1 文档对被文档模块的依赖

| 文档任务 | 依赖的模块 | 说明 |
|----------|-----------|------|
| T1 (lib.rs 文档) | 全部 | 需要了解所有模块的概览 |
| T5 (核心模块文档) | dimension, element, complex, storage, layout | 基于 §1 章节编写 |
| T6 (张量与运算文档) | tensor, overload, broadcast, shape, index, construct, set | 基于 §1 章节编写 |
| T7 (基础设施文档) | ffi, workspace, simd, parallel, error, prelude | 基于 §1 章节编写 |
| T8 (类型级文档) | 全部 | 逐类型添加 doc comment |
| T9 (函数级文档) | 全部 | 逐函数添加 doc comment |

### 16.2 数据流

```
设计文档 (00-28)
    │
    ├── 提取模块职责、核心概念、API 签名
    │       │
    │       └── 写入各 mod.rs 的 //! 模块文档
    │
    └── 提取类型定义、方法签名
            │
            ├── 写入 doc comment (///)
            └── 写入 doctest (```rust ```)
                    │
                    └── cargo test --doc 验证
```

---

## 17. 实现任务拆分

### Wave 1: Crate 级文档

- [ ] **T1**: 编写 lib.rs 顶层 crate 文档
  - 文件: `src/lib.rs`
  - 内容: 项目概述、Quick Start、Features 表、元素类型表、内存布局说明、no_std 说明
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
  - 内容: 模块职责、核心概念、使用示例、依赖图、设计决策（参见 `02-dimension.md §1`、`03-element.md §1`、`04-complex.md §1`、`05-storage.md §1`、`06-memory.md §1`）
  - 测试: `cargo doc --no-deps` 无 warning
  - 前置: T2
  - 预计: 10 min

- [ ] **T6**: 编写张量与运算模块文档（tensor, overload, broadcast, shape, index, construct, set）
  - 文件: 各 `mod.rs`
  - 内容: 模块职责、核心类型、运算分类、类型约束速查（参见 `07-tensor.md §1`、`11-math.md §1`、`15-broadcast.md §1`、`16-shape.md §1`）
  - 测试: `cargo doc --no-deps` 无 warning
  - 前置: T2
  - 预计: 10 min

- [ ] **T7**: 编写基础设施模块文档（ffi, workspace, simd, parallel, error, prelude）
  - 文件: 各 `mod.rs`
  - 内容: 模块职责、Safety 约定、feature gate 说明（参见 `23-ffi.md §1`、`24-workspace.md §1`、`08-simd.md §1`、`09-parallel.md §1`、`26-error.md §1`）
  - 测试: `cargo doc --no-deps` 无 warning
  - 前置: T2
  - 预计: 10 min

### Wave 3: 类型/函数级文档（可并行）

- [ ] **T8a**: tensor 模块公共 API 文档
  - 文件: `src/tensor/mod.rs` 及相关文件
  - 内容: TensorBase, Tensor, TensorView, TensorViewMut, ArcTensor 类型文档
  - 测试: `cargo doc --no-deps` 无 warning
  - 前置: T5, T6, T7
  - 预计: 15 min

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

- [ ] **T9a**: overload 模块逐元素运算文档
  - 文件: `src/overload/` 下相关文件
  - 内容: add, sub, mul, div, sin, cos, exp, abs 等逐元素运算函数文档和 doctest
  - 测试: `cargo test --doc --all-features`
  - 前置: T5, T6, T7
  - 预计: 15 min

- [ ] **T9b**: overload 模块归约运算文档
  - 文件: `src/overload/` 下相关文件
  - 内容: sum, sum_axis, dot 等归约函数文档和 doctest
  - 测试: `cargo test --doc --all-features`
  - 前置: T5, T6, T7
  - 预计: 10 min

- [ ] **T9c**: broadcast 和 shape 模块文档
  - 文件: `src/broadcast.rs`, `src/shape/mod.rs`
  - 内容: broadcast_shape, transpose, reshape 函数文档和 doctest
  - 测试: `cargo test --doc --all-features`
  - 前置: T5, T6, T7
  - 预计: 10 min

- [ ] **T9d**: construct 和 set 模块文档
  - 文件: `src/construct/mod.rs`, `src/set/mod.rs`
  - 内容: zeros, ones, full, eye, from_shape_vec, from_fn, unique 函数文档和 doctest
  - 测试: `cargo test --doc --all-features`
  - 前置: T5, T6, T7
  - 预计: 10 min

- [ ] **T9e**: ffi, workspace, error 模块文档
  - 文件: `src/ffi/mod.rs`, `src/workspace/mod.rs`, `src/error.rs`
  - 内容: FFI 函数（含 Safety 节）、Workspace、XenonError 文档和 doctest
  - 测试: `cargo test --doc --all-features`
  - 前置: T5, T6, T7
  - 预计: 15 min

### Wave 4: 示例程序（可并行）

- [ ] **T10**: 编写 examples/basic.rs
  - 文件: `examples/basic.rs`
  - 内容: 创建、运算、归约、打印
  - 测试: `cargo run --example basic`
  - 前置: T1
  - 预计: 10 min

- [ ] **T11**: 编写 examples/complex_numbers.rs
  - 文件: `examples/complex_numbers.rs`
  - 内容: 复数构造、算术、混合运算
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

- [ ] **T16**: 编写 `templates/no_std-app/` 最小工程模板
  - 文件: `templates/no_std-app/`
  - 内容: `no_std + alloc` 环境下的最小依赖工程（不是普通 `examples/` 可执行程序）
  - 测试: `cargo check --manifest-path templates/no_std-app/Cargo.toml --no-default-features`
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
Wave 4: [T8a] [T8b] [T8c] [T8d] [T8e] [T9a] [T9b] [T9c] [T9d] [T9e]
            │
Wave 5: [T10] [T11] [T12] [T13] [T14] [T15] [T16]
            │
Wave 6: [T17]
```

---

## 18. ADR 决策记录

### 决策 1：英文文档

| 属性 | 值 |
|------|-----|
| 决策 | 所有 doc comment 和 README 使用英文 |
| 理由 | Rust 生态惯例；docs.rs 面向全球开发者（参见 `00-coding.md §6`） |
| 替代方案 | 中文文档 — 放弃，不符合 Rust 社区惯例 |

### 决策 2：doctest 使用 `?` 而非 `unwrap()`

| 属性 | 值 |
|------|-----|
| 决策 | doctest 使用 `?` 而非 `unwrap()` |
| 理由 | 遵循 Rust API Guidelines C-QUESTION-MARK；展示惯用错误处理 |
| 替代方案 | unwrap — 放弃，给用户错误示范 |

### 决策 3：开发期间 `#![warn(missing_docs)]`，CI 中 deny

| 属性 | 值 |
|------|-----|
| 决策 | 开发期间使用 `warn` 级别，CI 中通过 `RUSTDOCFLAGS="-D warnings"` 强制 deny 级别 |
| 理由 | 需求说明书 §28.1 要求所有公开 API 有文档；开发期间 warn 允许渐进式补全文档，CI 中 deny 阻止无文档代码合入 |
| 替代方案 | 始终 deny 级别 — 放弃，开发期间过于严格，阻碍快速迭代 |

### 决策 4：按模块组织模块级文档

| 属性 | 值 |
|------|-----|
| 决策 | 每个模块的 mod.rs 包含完整的模块概述 |
| 理由 | 用户从 docs.rs 进入模块时能快速理解模块定位和核心类型 |
| 替代方案 | 仅函数级文档 — 放弃，缺乏模块整体视图 |

### 决策 5：examples 按场景而非按模块

| 属性 | 值 |
|------|-----|
| 决策 | examples/ 按使用场景（basic/broadcasting/parallel 等）组织 |
| 理由 | 用户按需求查找示例，而非按源码模块 |
| 替代方案 | 按源码模块组织 — 放弃，不便于用户理解实际用法 |

---

## 版本历史

| 版本 | 日期 |
|------|------|
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-08 |
| 1.0.2 | 2026-04-08 |
| 1.0.3 | 2026-04-08 |
| 1.1.0 | 2026-04-08 |
| 1.1.1 | 2026-04-08 |
| 1.1.2 | 2026-04-10 |
| 1.1.3 | 2026-04-10 |

---

*本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。*
