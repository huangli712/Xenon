# 文档交付规范

> 文档编号: 29
> 适用目录: src/** pub API 文档、README.md、examples
> 任务阶段: Phase 6
> 前置文档: 所有前置文档（00-coding.md ~ 28-tests.md）

---

## 1. 模块定位

### 1.1 职责边界

| 职责         | 包含                                   | 不包含                             |
| ------------ | -------------------------------------- | ---------------------------------- |
| API 文档     | 所有 pub 类型和函数的 doc comment      | 内部实现注释（非 pub）             |
| 使用示例     | 关键 API 的可运行代码示例（doctest）   | 完整教程、视频教程                 |
| Safety 说明  | 所有 unsafe 函数的 `# Safety` 文档节   | 安全函数的 Safety 节               |
| Crate 级文档 | lib.rs 顶层文档、README                | 第三方博客文章、CHANGELOG 工程产物 |
| 模块级文档   | 各 mod.rs 的 `//!` 模块概述            | 内部实现文档                       |
| examples/    | 独立可运行示例程序                     | 交互式 notebook                    |
| docs.rs 配置 | metadata、feature gate 标注            | 自定义文档主题                     |

### 1.2 设计原则

| 原则       | 体现                                               |
| ---------- | -------------------------------------------------- |
| 全覆盖     | 所有 pub API 必须有 doc comment                    |
| 可测试     | 关键 API 的示例通过 doctest 或独立 examples 验证   |
| 安全性透明 | 所有 unsafe 函数有 `# Safety` 节                   |
| 惯用法     | 遵循 Rust API Guidelines                           |
| 英文文档   | 所有 doc comment 使用英文                          |

---

## 2. 需求映射与范围约束

| 类型     | 内容                                                         |
| -------- | ------------------------------------------------------------ |
| 需求映射 | 需求说明书 §28                                               |
| 范围内   | pub API 文档、doctest、examples、docs.rs 配置、README        |
| 范围外   | 第三方教程平台、自定义文档主题、交互式 notebook 或站点系统   |
| 非目标   | 通过文档规范扩展产品能力、引入额外文档构建依赖或改变平台边界 |

---

## 3. 文件位置

```
src/
├── lib.rs                    # Crate-level docs (L0)
├── prelude.rs                # Prelude docs (L1)
├── private.rs                # Sealed-trait infrastructure (internal, no pub docs)
├── error.rs                  # Error module docs (L1)
├── dispatch.rs               # Internal dispatch helper (internal, no pub docs)
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
├── simd/                     # (pub(crate), feature-gated, no externally-visible docs)
├── parallel/                 # (pub(crate), feature-gated, no externally-visible docs)
├── broadcast/
│   └── mod.rs                # Broadcast module docs (L1)
├── math/
│   └── mod.rs                # Element-wise operation module docs (L1)
├── overload/
│   └── mod.rs                # Operator-overload module docs (L1)
├── util/
│   └── mod.rs                # Utility module docs (L1) — clip, fill, to_contiguous
├── set/
│   └── mod.rs                # Set-operation module docs (L1)
├── matrix/
│   └── mod.rs                # Vector dot-product module docs (L1)
├── reduction/
│   └── mod.rs                # Reduction module docs (L1)
├── shape/
│   └── mod.rs                # Shape-operation module docs (L1)
├── index/
│   └── mod.rs                # Indexing module docs (L1)
├── construct/
│   └── mod.rs                # Constructor module docs (L1)
├── convert/
│   └── mod.rs                # Type-conversion module docs (L1)
├── format/
│   └── mod.rs                # Output-formatting module docs (L1)
├── ffi/
│   └── mod.rs                # FFI module docs (L1)
├── workspace/
│   └── mod.rs                # Workspace module docs (L1)

examples/
├── basic.rs                  # Basic-operations example
├── complex.rs                # Complex-number operations example
├── broadcasting.rs           # Broadcasting example
├── features.rs               # Optional-feature behavior example
├── simd.rs                   # SIMD-acceleration example
├── ffi.rs                    # FFI integration example
└── workspace.rs              # Workspace borrow/split/growth example

README.md                     # Project README
CHANGELOG.md                  # Optional engineering changelog artifact (non-required deliverable)
```

---

## 4. 依赖关系

### 4.1 依赖图（ASCII）

```
29-documentation
├── depends on all design docs (00-28)
│   └── each module's docs are derived from its design doc
├── depends on `00-coding.md`
│   └── documentation style follows the coding conventions (see `00-coding.md §7`)
├── depends on `01-architecture.md`
│   └── module layout, feature list, and dependency layer graph inform docs structure
├── depends on `25-safety.md`
│   └── thread safety (Send/Sync) documentation aligned with safety invariant definitions
├── depends on `00-coding.md §6` + `23-ffi.md`
│   └── unsafe API documentation checklist aligned with safety invariant definitions
├── depends on `28-tests.md`
│   └── doctest / examples / docs CI validation must stay aligned
└── may reference `27-benchmark.md`
    └── If a benchmark documentation template is needed, refer to `27-benchmark.md`
```

### 4.2 类型级依赖

| 来源             | 使用的内容                         |
| ---------------- | ---------------------------------- |
| 所有 `src/` 模块 | pub API 签名、类型定义、trait 定义 |
| `Cargo.toml`     | feature 列表、依赖列表、metadata   |
| 需求说明书       | API 行为规范、精度要求、边界定义   |

### 4.3 依赖合法性

| 项目           | 说明                                        |
| -------------- | ------------------------------------------- |
| 新增第三方依赖 | 无新增依赖                                  |
| 合法性结论     | 符合最小依赖限制                            |
| 替代方案       | 不适用；文档生成依赖 rustdoc 与现有工程配置 |

### 4.4 依赖方向声明

依赖方向：文档跟随代码。文档内容基于源码 API 签名和设计文档，不被代码依赖。

### 4.5 数据流

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

## 5. 公共 API 设计

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
| L2   | 每个 pub 项必须有 doc comment | `#![warn(missing_docs)]`                             |
| L3   | 关键 API 至少一个示例         | `cargo build --examples` / `cargo run --example ...` |

### 5.3 关键 API 示例覆盖矩阵

| API 族                               | 必须有示例 | 示例载体  | 对应设计文档         |
| ------------------------------------ | ---------- | --------- | -------------------- |
| 构造 (`zeros`/`ones`/`eye`/`from_*`) | ✅         | example   | `18-construction.md` |
| 索引/切片                            | ✅         | doctest   | `17-indexing.md`     |
| 转置                                 | ✅         | doctest   | `16-shape.md`        |
| 广播                                 | ✅         | example   | `15-broadcast.md`    |
| 逐元素运算                           | ✅         | doctest   | `11-math.md`         |
| 归约 (`sum`)                         | ✅         | doctest   | `13-reduction.md`    |
| 内积 (`dot`)                         | ✅         | doctest   | `12-matrix.md`       |
| 类型转换 (`cast`)                    | ✅         | doctest   | `21-type.md`         |
| FFI unsafe API                       | ✅         | example   | `23-ffi.md`          |
| 运算符重载                           | ✅         | doctest   | `19-overload.md`     |
| `clip/fill/try_fill/to_contiguous/into_contiguous` | ✅ | doctest | `20-utility.md`  |
| 集合操作 (`unique`)                  | ✅         | doctest   | `14-set.md`          |
| 工作空间                             | ✅         | example   | `24-workspace.md`    |
| 格式化输出                           | ✅         | doctest   | `22-output.md`       |

上表是示例覆盖矩阵的理想目标。"示例载体"列标明该 API 族主要通过独立 example 程序还是 doctest 满足覆盖要求。

### 5.4 核心文档模板

#### 5.4.1 lib.rs 顶层文档结构

````rust,ignore
//! # Xenon — N-dimensional Tensor Library for Rust
//!
//! Xenon is a high-performance N-dimensional array (tensor) library for Rust,
//! designed as numerical infrastructure for scientific computing.
//!
//! ## Quick Start
//!
//! ```rust
//! # use xenon::prelude::*;
//!
//! # fn demo() -> xenon::Result<()> {
//! // Create tensors
//! let a = Tensor::<f64, _>::zeros([5])?;
//! let b = Tensor::<f64, _>::zeros([3, 4])?;
//!
//! // Element-wise operations with broadcasting
//! let added = (&a + &a)?;
//!
//! // Reduction
//! let total = b.sum();
//! assert_eq!(added.len(), 5);
//! assert_eq!(total, 0.0);
//! # Ok(())
//! # }
//! ```
//!
//! ## Runtime Environment
//!
//! Xenon supports only the `std` environment.
//! It does not need or provide a `std` feature toggle.
//! All documentation assumes a `std` environment.
//!
//! ## Optional Features
//!
//! | Feature | Default | Description |
//! |---------|:-------:|-------------|
//! | `parallel` | ✗ | Data parallelism via rayon |
//! | `simd` | ✗ | SIMD acceleration via pulp |
//!
//! ## Supported Element Types
//!
//! | Level | Types | Trait Bound |
//! |-------|-------|-------------|
//! | Base | i32, i64, f32, f64, Complex<f32>, Complex<f64>, bool | `Element` |
//! | Numeric | i32, i64, f32, f64, Complex<f32>, Complex<f64> | `Numeric: Element` |
//! | Real | f32, f64 | `RealScalar: Numeric` |
//! | Complex | Complex<f32>, Complex<f64> | `ComplexScalar: Numeric` |
//!
//! `usize` is reserved for shape and index metadata, not as a tensor element type.
//!
//! ## Memory Layout
//!
//! Default layout is **F-order (column-major)**.
//! Xenon provides helper APIs that make upstream BLAS/LAPACK integration easier,
//! but not every legal layout is natively BLAS/LAPACK-compatible.
//!

// lint configuration — see §5.5.1 Lint Rules
````

#### 5.4.2 文档节使用规则

| 文档节        | 何时必须           | 说明                 |
| ------------- | ------------------ | -------------------- |
| `# Arguments` | 方法有 2+ 参数时   | 描述每个参数         |
| `# Returns`   | 返回值非显而易见时 | 描述返回值属性       |
| `# Errors`    | 返回 Result 时     | 列出所有错误变体     |
| `# Panics`    | 可能 panic 时      | 列出所有 panic 条件  |
| `# Safety`    | unsafe 函数        | 列出安全前提条件     |
| `# Examples`  | 所有关键 API       | 至少一个可运行示例   |
| `# See Also`  | 有相关 API 时      | 交叉引用             |

容差体系、错误模型和 panic 语义的具体内容由对应技术规范（`26-error.md`、`28-tests.md`）定义。公共文档需引用 `需求说明书 §28.3` 的容差语义。文档层仅要求引用这些规范，不重复定义。

对运算符重载入口（如 `Add` / `Sub` / `Mul` / `Div` 的实现文档），即使签名经由 trait 间接暴露，也应补齐与对应方法型 API 一致的 `# Errors` / `# Panics` 模板，并引用 `19-overload.md` 中定义的对应技术规范，避免仅留下语法糖示例而缺少失败条件说明。

所有 doc comment、doctest、README、CHANGELOG 与 examples 中的示例代码必须执行协同审查：

- 错误构造必须匹配 `26-error.md §5.1`。
- 索引示例必须使用 `try_at` / `try_at_mut` 或 `get(&[...])` / `get_mut(&[...])`，不得展示方括号索引语法。
- 构造示例必须保持 `Tensor::from_shape_vec` 的失败语义为 `InvalidShapeKind::ElementCountMismatch`。
- `zeros` / `ones` 的实现说明必须使用 `<Owned<A> as StorageOwned>::from_elem(len, value)` 的完全限定调用。
- 运算符示例必须体现 `Output = Result<Tensor, XenonError>`：同形状可写 `(&a + &b)?`，异形状优先写显式方法 `a.add(&b)?`；不得新增或引用额外的 try 前缀算术方法；左标量示例使用 `Scalar<A>` 包装类型，不展示原生左标量加张量语法。
- 类型转换示例必须遵循静态无损 `From`、静态有损与动态条件性 `CastTo<T>` 三层结构；复数实数构造只展示 `From<T> for Complex<T>`，整数到复数转换走 `CastTo`。
- Workspace 示例必须展示 `borrow_mut(&mut self)`、`Workspace::split_at_mut(&mut self)` 与消费式 `SplitBorrowMut::split_at_mut(self)`。
- 线程安全说明以 `25-safety.md` 为准。

#### 5.4.3 Sealed trait 公开 doc 约定

公开但 sealed 的 trait（外部不可实现，但可在 `where` 子句、trait 对象、关联类型中命名）必须在其 `pub trait` 的 doc comment 中显式声明 sealed 状态。这是 docs.rs 公开 API 一致性约定，不是实现细节披露。

**适用范围**：

- `05-storage.md`：`RawStorage` / `RawStorageMut` / `Storage` / `StorageMut` / `StorageOwned` / `StorageShared` 6 个 storage trait + `IsOwned` / `IsView` / `IsViewMut` / `IsShared` 4 个 marker trait
- `02-dimension.md`：`Dimension` 及其超 trait `Reverse`
- `03-element.md`：`Element` / `Numeric` / `RealScalar` / `ComplexScalar` / `CastElement`（封闭元素集成员关系）
- `21-type.md`：`CastTo<T>`（封闭转换矩阵）

**强制 doc comment 格式**：

```rust,ignore
/// ... existing trait description ...
///
/// # Sealed
///
/// This trait is sealed and cannot be implemented outside of `Xenon`.
/// External crates may name it in `where` clauses or trait bounds, but
/// adding new implementations is intentionally not supported. The
/// implementor set is closed: see the owner design document for the
/// complete list and the rationale.
pub trait MyPublicSealedTrait: crate::private::Sealed { /* ... */ }
```

**禁止**：

- 在公开 doc 中暴露 `private.rs` 模块路径或 `Sealed` super-trait 的实现细节（仅说明"sealed and cannot be implemented outside this crate"即可，不写"via `crate::private::Sealed`"）。
- 让 sealed 状态仅通过文件树注释（如 `private.rs # Sealed-trait infrastructure`）传达；必须在每个 sealed 公开 trait 自身的 doc comment 中显式声明。
- 对 `pub(crate)` 内部 trait（如 `ConvertTo<B>`、`PermuteAxes`）施加同样要求——它们不在公开 API 面，无须公开 sealed 声明。

文档审查（`§5.5` Lint 与文档门禁）必须包含一项 grep 检查：所有 `pub trait` 声明位于 sealed 范围内时，doc comment 必须出现 `Sealed` 段落。

### 5.5 Lint 与文档门禁

#### 5.5.1 Lint 规则

本节列出文档专项 lint。完整 `lib.rs` 项目级 lint 基线（含 `missing_docs`、`unsafe_op_in_unsafe_fn`、`missing_debug_implementations` 等）见 `00-coding.md §7.1`。

**开发提示**：在开发期间可将 deny 改为 warn（`#![warn(missing_docs)]`），CI 中通过 `RUSTDOCFLAGS="-D warnings" cargo doc` 来强制执行文档完整性检查（参见 §5.11.1 CI checks）。

**门禁说明**：`#![warn(missing_docs)]` 为最小存在性检查（确保公开 item 有文档文本），不保证文档质量、完整性或示例覆盖。完整文档质量由评审流程保障。

**执行矩阵说明**：doctest 与测试 CI 矩阵由 `28-tests.md` 统一定义。本文档仅规定“需要哪些文档验证”，不维护 CI 执行矩阵。

```rust,ignore
// lib.rs — documentation-specific lint additions only.
// The complete lib.rs lint baseline is in `00-coding.md §7.1`; the entries
// below are supplementary rustdoc / clippy rules not already covered there.
#![deny(rustdoc::broken_intra_doc_links)]     // doc links must be valid
#![deny(rustdoc::private_intra_doc_links)]    // private item links are invalid
#![warn(rustdoc::missing_crate_level_docs)]   // crate-level docs must exist
// NOTE: NO `#![cfg_attr(docsrs, feature(doc_cfg))]` — that gate is nightly-only
// and would break MSRV 1.85 stable builds. See `00-coding.md §10.3`.
```

#### 5.5.2 Clippy 文档 lint

```rust,ignore
// Enabled in CI
#![warn(clippy::missing_errors_doc)]      // Result functions need Errors section
#![warn(clippy::missing_panics_doc)]      // Panicking functions need Panics section
#![warn(clippy::missing_safety_doc)]      // Unsafe functions need Safety section
```

以上 lint 对应 §5.4.2 中 `# Errors`、`# Panics`、`# Safety` 三节的必须规则，仅在 CI 中作为自动化门禁补充；完整文档节定义仍以 §5.4.2 为准。完整 `lib.rs` 基线（含 `missing_docs`、`unsafe_op_in_unsafe_fn`、`missing_debug_implementations` 等）见 `00-coding.md §7.1`。

#### 5.5.3 Sealed-trait doc grep 检查

针对 §5.4.3 列出的所有公开 sealed trait，CI 必须执行一项 grep / 脚本检查：

- 在源码中扫描所有满足 `pub trait <Name>(<...>): ... crate::private::Sealed` 或在 §5.4.3 列表内的 `pub trait` 声明；
- 对每个匹配的 trait，断言其紧邻前置 doc comment（`///` 连续块）必须出现 `# Sealed` 段落；
- 不出现 `# Sealed` 段落 → CI 失败。

最小实现示例（CI shell hook，伪代码）。**关键**：检查的是 `pub trait` 行**之前**的连续 `///` 块，**不是**之后的函数体内容：

```bash
# Fail if any sealed pub trait lacks the `# Sealed` doc section in its
# IMMEDIATELY PRECEDING contiguous `///` doc-comment block.
#
# Candidate discovery uses TWO sources unioned:
#   (1) Direct grep match: `pub trait <Name>...crate::private::Sealed` —
#       catches traits whose declaration line contains the Sealed bound.
#   (2) §5.4.3 whitelist: explicit list of sealed pub trait names — catches
#       traits that are sealed via super-trait (e.g., `Reverse: Dimension`
#       inherits Dimension's Sealed bound) or whose `pub trait` line does
#       NOT literally contain "Sealed". The whitelist MUST be kept in sync
#       with §5.4.3.
set -euo pipefail

# §5.4.3 whitelist:
WHITELIST='RawStorage|RawStorageMut|Storage|StorageMut|StorageOwned|StorageShared|IsOwned|IsView|IsViewMut|IsShared|Dimension|Reverse|Element|Numeric|RealScalar|ComplexScalar|CastElement|CastTo'

# Union of (1) direct Sealed match and (2) whitelist match.
matches_direct=$(grep -REn '^pub (unsafe )?trait [A-Z][A-Za-z0-9_]*.*Sealed' src/ || true)
matches_whitelist=$(grep -REn "^pub (unsafe )?trait (${WHITELIST})\\b" src/ || true)
matches=$(printf '%s\n%s\n' "$matches_direct" "$matches_whitelist" | sort -u | grep -v '^$' || true)

exit_code=0
while IFS= read -r match; do
    [ -z "$match" ] && continue
    file=$(echo "$match" | cut -d: -f1)
    line=$(echo "$match" | cut -d: -f2)
    name=$(echo "$match" | sed -E 's/.*pub (unsafe )?trait ([A-Z][A-Za-z0-9_]*).*/\2/')
    # Walk upward from (line - 1) collecting contiguous `///` lines until
    # the first non-/// line; assert `# Sealed` appears in that block.
    block=$(awk -v target="$line" '
        NR < target {
            if ($0 ~ /^[[:space:]]*\/\/\//) { buf = buf "\n" $0 } else { buf = "" }
        }
        NR == target { print buf; exit }
    ' "$file")
    if ! echo "$block" | grep -q '^[[:space:]]*///[[:space:]]*# Sealed'; then
        echo "MISSING # Sealed doc section: $name in $file:$line" >&2
        exit_code=1
    fi
done <<< "$matches"
exit $exit_code
```

该门禁与 §5.4.3 的"强制 doc comment 格式"形成闭环：§5.4.3 要求作者写、§5.5.3 在 CI 校验。脚本逻辑必须扫描 `pub trait` 之前的 doc comment 块（非之后），否则即使作者按 §5.4.3 正确书写也会被误报为缺失。

### 5.6 Doctest 规范

#### 5.6.1 规则

| 规范       | 说明                                                                                                                   |
| ---------- | ---------------------------------------------------------------------------------------------------------------------- |
| 可编译运行 | 所有 doctest 通过 `cargo test --doc`；关键示例至少通过 `cargo run --example ...` 实际运行，其余 examples 至少编译通过  |
| 使用 `?`   | doctest 天然返回 `Result` 时必须优先使用 `?`；避免在文档示例中使用 `unwrap()`                                          |
| 隐藏样板   | 用 `# ` 隐藏 use 语句                                                                                                  |
| 最小化     | 只展示当前 API 用法                                                                                                    |
| 有断言     | 用 `assert_eq!` 验证结果                                                                                               |

API 形态与错误字段审查必须执行 §5.4.2 的协同审查清单；特别是索引示例不得使用标准库索引 trait 语法，错误示例不得使用旧字段或运行时类型 ID。

**关键示例定义**：§5.11.1 Gate 4 列出的示例（当前为 `basic` / `broadcasting` / `workspace`）为关键默认示例，必须在 CI 中实际运行。§5.3 示例覆盖矩阵列出的全部 API 族示例均须编译通过；其中被 Gate 4 命名的还须运行通过。

#### 5.6.2 Doctest 模板

````rust,ignore
/// Compute the sum of all elements.
///
/// # Examples
///
/// ```
/// # use xenon::prelude::*;
///
/// # fn demo() -> xenon::Result<()> {
/// let t = Tensor::<f64, _>::from_shape_vec([3], vec![1.0, 2.0, 3.0])?;
/// assert_eq!(t.sum(), 6.0);
/// # Ok(())
/// # }
/// ```
pub fn sum(&self) -> A { ... }
````

#### 5.6.3 Feature-gated Doctest

本模板演示的是 feature-gated doctest 的编写模式。`sum()` 本身是始终可用的 API（见 §5.10.2），此处仅因为示例中验证的行为（并行路径的正确性）依赖 `parallel` feature，因此用 `#[cfg(feature = "parallel")]` 包裹。`sum()` 的基础 doctest 见 §5.6.2。

````rust,ignore
/// Compute the sum of all elements.
///
/// With the `parallel` feature enabled, the implementation may choose an
/// internal parallel execution path while preserving the documented public
/// `sum()` semantics.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "parallel")]
/// # {
/// # use xenon::prelude::*;
///
/// # fn demo() -> xenon::Result<()> {
/// let t = Tensor::<f64, _>::ones([1_000_000])?;
/// let s = t.sum();
/// assert_eq!(s, 1_000_000.0);
/// # Ok(())
/// # }
/// # }
/// ```
pub fn sum(&self) -> A { ... }
````

### 5.7 examples/ 目录规划

#### 5.7.1 示例清单

| 文件                 | 内容                                               | Feature            | 目标用户                    |
| -------------------- | -------------------------------------------------- | ------------------ | --------------------------- |
| `basic.rs`           | 创建、运算、归约、打印                             | 默认               | 新用户                      |
| `complex.rs`         | 复数构造、同类型复数运算、显式转换后的运算         | 默认               | 科学计算                    |
| `broadcasting.rs`    | 广播规则、行/列/标量广播                           | 默认               | 日常使用                    |
| `features.rs`        | 可选 feature 对公开 API 语义/性能路径的影响        | `parallel`, `simd` | 性能优化                    |
| `simd.rs`            | `simd` feature 对公开运算路径的影响与回退策略      | `simd`             | 性能优化                    |
| `ffi.rs`             | 为上游 C/BLAS-LAPACK 集成提供辅助 API 与兼容性判断 | 默认               | 库开发者                    |
| `workspace.rs`       | 工作空间借用、split 与扩容语义示例                 | 默认               | 上游 scratch-buffer 使用者  |

#### 5.7.2 示例模板

```rust,ignore
//! Example: Brief description
//!
//! Run with: `cargo run --example basic`

use xenon::prelude::*;

fn main() -> xenon::Result<()> {
    // Step 1: Create tensors
    let a = Tensor::<f64, _>::zeros([3, 4])?;
    println!("Created 3x4 zero matrix: shape={:?}", a.shape());

    // Step 2: Perform operation
    let b = a.transpose();
    println!("Transposed: shape={:?}", b.shape());

    Ok(())
}
```

#### 5.7.3 示例编写规范

| 规范         | 说明                            |
| ------------ | ------------------------------- |
| 自包含       | 独立可运行，不依赖其他示例      |
| 有注释       | 关键步骤有行内注释              |
| 有输出       | 使用 `println!` 展示结果        |
| Feature gate | 需可选 feature 的在文件顶部注明 |
| 无 unwrap    | 使用 `?`，main 返回 `Result`    |

### 5.8 README.md 内容规划

README 使用英文的来源与 crate 内 doc comment 一致：遵循 `00-coding.md §7` 的英文文档约束，并面向 docs.rs / crates.io 的 Rust 生态读者。

````markdown
# Xenon

Rust N-dimensional tensor library for scientific computing.

## Features

- N-dimensional arrays with static (0-6D) and dynamic dimensions (`IxDyn` for runtime-rank tensors)
- Column-major (F-order) default, with helper APIs and compatibility checks for upstream BLAS/LAPACK integration when the layout preconditions are satisfied
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

### 5.9 CHANGELOG.md

**设计扩展说明：** `CHANGELOG.md` 为可选工程辅助产物，不属于 `需求说明书 §28.1` 的文档要求范围。若维护，建议遵循 [Keep a Changelog](https://keepachangelog.com/) 格式。

### 5.10 docs.rs 配置

#### 5.10.1 Cargo.toml metadata

```toml
[package.metadata.docs.rs]
all-features = true
# NOTE: do NOT add `rustdoc-args = ["--cfg", "docsrs"]` (see `00-coding.md §10.3`).
```

#### 5.10.2 Feature gate 标注

文档中必须显式区分以下两类情况：

1. **API gated by feature**：API 本身只在特定 feature 启用时出现，此时仅使用 `#[cfg(feature = "...")]` 条件编译；**不**使用 `#[doc(cfg(...))]`（nightly-only，详见 `00-coding.md §10.3`）。
2. **API always present but behavior varies by feature**：API 始终存在，只是启用 feature 后内部执行路径或性能特征变化；此时不得把该 API 误写成“仅在 feature 下可用”，而应在正文中说明行为差异。

```rust,ignore
// lib.rs — NO `#![cfg_attr(docsrs, feature(doc_cfg))]` (nightly-only,
// breaks MSRV 1.85 stable; see `00-coding.md §10.3`).

// Public APIs whose behavior is affected by an optional feature should document
// the behavior change directly instead of using doc(cfg) when the API itself is
// always available.
// Documentation note example:
// Enabled with the `parallel` feature, the internal execution path may use
// parallel acceleration while the public API semantics remain unchanged.
pub fn sum(&self) -> A { ... }
```

### 5.11 文档 CI 检查

#### 5.11.1 验证项目

| 检查项                                   | 命令                                                                                                                       | 失败条件                                                          |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| Gate 1：rustdoc 文档门禁                 | `RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps`                                                            | 任何 missing docs / broken intra-doc links / 其他 rustdoc warning |
| Gate 2：文档节完整性门禁（补充文档 lint） | `cargo clippy --all-features -- -D clippy::missing_errors_doc -D clippy::missing_panics_doc -D clippy::missing_safety_doc` | 缺少 `# Errors` / `# Panics` / `# Safety` 文档节                  |
| Gate 3：Doctest                          | `cargo test --doc --all-features`                                                                                          | 任何失败                                                          |
| Gate 4：示例验证                         | `cargo build --examples --all-features` + 关键示例运行命令（见 §5.6.1 定义；当前为 `basic` / `broadcasting` / `workspace`）| 任何失败                                                          |
| Gate 5：Clippy 完整门禁                  | `cargo clippy --all-features -- -D warnings`                                                                               | 任何 clippy warning                                               |
| Gate 6：编译警告门禁                     | `RUSTFLAGS="-D warnings" cargo check --all-features`                                                                       | 任何编译器 warning                                                |

Gate 4 当前涵盖 `basic` / `broadcasting` / `workspace` 三个核心示例；完整示例覆盖清单见 §5.3。随着项目成熟，Gate 4 范围可逐步扩展至 §5.3 表中的所有 14 个 API 族。

Gate 1、Gate 5、Gate 6 为 `00-coding.md §7.1` 规定的三项 CI 硬门禁。Gate 5（完整 clippy）覆盖所有 clippy lint，远不止 Gate 2 的三条文档专项 lint；Gate 2 作为补充文档节完整性校验继续保留，但不替代完整 clippy 扫描。Gate 6 确保常规编译警告在所有 feature 组合下均提升为错误。

#### 5.11.2 CI 配置与 Feature 维度验证矩阵

**说明**：§5.11.1 定义了文档交付需要的验证项。权威的 doctest / examples CI 执行矩阵（含 Feature 维度验证矩阵）由 `28-tests.md` 统一维护，本文档不再重复。

### 5.12 Good / Bad 文档注释对比

#### 5.12.1 Good — 完整的函数文档

````rust,ignore
/// Compute the sum of all elements in the tensor.
///
/// Returns the additive identity (zero) for empty tensors.
/// For floating-point types, NaN values propagate to the result.
///
/// # Examples
///
/// ```
/// # use xenon::prelude::*;
///
/// # fn demo() -> xenon::Result<()> {
/// let t = Tensor::<f64, _>::from_shape_vec([3], vec![1.0, 2.0, 3.0])?;
/// assert_eq!(t.sum(), 6.0);
///
/// let empty = Tensor::<f64, _>::zeros([0])?;
/// assert_eq!(empty.sum(), 0.0);
/// # Ok(())
/// # }
/// ```
///
/// # Performance
///
/// O(n) time complexity. With `simd` feature enabled, the implementation may
/// choose an internal SIMD path for contiguous data while preserving the same
/// public API semantics.
///
/// # See Also
///
/// * [`sum_axis`](Self::sum_axis) — sum along a specific axis
pub fn sum(&self) -> A { ... }
````

#### 5.12.2 Bad — 不完整的函数文档

````rust,ignore
// Bad: no documentation, no examples, no description
pub fn sum(&self) -> A { ... }

// Bad: the example is incomplete — it omits the surrounding API description,
// return-value semantics, and edge-case notes even though the doctest itself compiles.
/// ```
/// # use xenon::prelude::*;
/// # fn demo() -> xenon::Result<()> {
/// let t = Tensor::<f64, _>::from_shape_vec([3], vec![1.0, 2.0, 3.0])?;
/// assert_eq!(t.sum(), 6.0);
/// # Ok(())
/// # }
/// ```
pub fn sum(&self) -> A { ... }
````

#### 5.12.3 Bad — Safety 文档不完整的 unsafe 函数

```rust,ignore
// Bad: safety contract is incomplete for a still-supported raw-parts constructor.
/// Create a tensor view from raw parts.
///
/// # Safety
///
/// Caller guarantees ptr/shape/strides/offset are valid.
///
/// // Missing: aliasing, lifetime provenance, initialization,
/// // bounds, and overflow/layout preconditions required by 23-ffi.md.
pub unsafe fn from_raw_parts<'a, A, D>(...) -> TensorView<'a, A, D>
```

---

## 6. 内部实现设计

### 6.1 文档生成流程

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

### 6.2 文档覆盖率计算

```bash
# Check for missing docs at deny level
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps

# Count undocumented pub items (manual audit)
# 1. cargo doc --no-deps 2>&1 | grep "missing documentation"
# 2. Ensure zero warnings
```

### 6.3 doc comment 编写工作流

| 步骤                | 操作                        | 验证                              |
| ------------------- | --------------------------- | --------------------------------- |
| 1. 编写 API         | 实现函数/类型               | `cargo check`                     |
| 2. 添加 doc comment | 描述、参数、返回值、示例    | `cargo doc` 无 warning            |
| 3. 添加 doctest     | `# Examples` 节             | `cargo test --doc` 通过           |
| 4. Safety 文档      | unsafe 函数的 `# Safety` 节 | `clippy::missing_safety_doc` 通过 |

---

## 7. 实现任务拆分

### Wave 1: Crate 级文档

- [ ] **T1**: 编写 lib.rs 顶层 crate 文档
  - 文件: `src/lib.rs`
  - 内容: 项目概述、Quick Start、Features 表、元素类型表、内存布局说明
  - 测试: `cargo doc --no-deps` 无 warning
  - 前置: 无
  - 预计: 10 min

- [ ] **T2**: 配置 `#![warn(missing_docs)]` 和 docs.rs metadata
  - 文件: `src/lib.rs`, `Cargo.toml`
  - 内容: lint 规则、`[package.metadata.docs.rs] all-features = true`（**不**使用 `cfg_attr(docsrs, ...)` 与 `--cfg docsrs`，详见 `00-coding.md §10.3`）
  - 测试: 编译通过
  - 前置: T1
  - 预计: 5 min

- [ ] **T3**: 编写 README.md
  - 文件: `README.md`
  - 内容: 项目介绍、Features、Quick Start、安装、文档链接、许可证；README 英文说明需明确引用 `00-coding.md §7` 与 Rust 生态受众
  - 测试: 内容完整
  - 前置: T1
  - 预计: 10 min

- [ ] **T4**: 可选维护 CHANGELOG.md
  - 文件: `CHANGELOG.md`
  - 内容: Keep a Changelog 格式；仅作为可选工程整理项，不属于 `需求说明书 §28.1` 的默认交付物
  - 测试: 格式正确
  - 前置: 无
  - 预计: 5 min

### Wave 2: 模块级文档

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

- [ ] **T7**: 编写基础设施模块文档（ffi, workspace, error, prelude, convert, format，以及 simd/parallel 内部后端说明）
  - 文件: 对外模块各 `mod.rs`；`simd` / `parallel` 仅补充内部架构说明与 feature 影响说明，不视为独立公开模块文档交付
  - 内容: 模块职责、Safety 约定、feature gate 说明、转换与输出语义；`simd` / `parallel` 作为内部执行后端，仅文档化内部架构说明及其对公开 API feature 行为/执行路径的影响，不定义独立公开 API surface 文档（参见 `23-ffi.md §1`、`24-workspace.md §1`、`08-simd.md §1`、`09-parallel.md §1`、`21-type.md §1`、`22-output.md §1`、`26-error.md §1`）
  - 测试: `cargo doc --no-deps` 无 warning
  - 前置: T2
  - 预计: 10 min

- [ ] **T8**: 编写 util 模块级文档
  - 文件: `src/util/mod.rs`
  - 内容: 模块职责概述、utility 函数分类说明（参见 `20-utility.md §1`）；仅模块级文档（`//!`），不含函数级 doctest
  - 测试: `cargo doc --no-deps` 无 warning
  - 前置: T2
  - 预计: 5 min

### Wave 3: 类型/函数级文档

- [ ] **T9**: tensor 模块公共 API 文档
  - 文件: `src/tensor/mod.rs` 及相关文件
  - 内容: TensorBase, Tensor, TensorView, TensorViewMut, ArcTensor 类型文档
  - 测试: `cargo doc --no-deps` 无 warning
  - 前置: T5, T6, T7
  - 预计: 10 min

- [ ] **T10**: dimension 模块文档
  - 文件: `src/dimension/mod.rs`
  - 内容: Ix0~Ix6, IxDyn, Dimension trait 文档
  - 测试: `cargo doc --no-deps` 无 warning
  - 前置: T5, T6, T7
  - 预计: 10 min

- [ ] **T11**: element 模块文档
  - 文件: `src/element/mod.rs`
  - 内容: Element, Numeric, RealScalar, ComplexScalar trait 文档
  - 测试: `cargo doc --no-deps` 无 warning
  - 前置: T5, T6, T7
  - 预计: 10 min

- [ ] **T12**: storage 模块文档
  - 文件: `src/storage/mod.rs`
  - 内容: Owned, ViewRepr, StorageMut trait 文档
  - 测试: `cargo doc --no-deps` 无 warning
  - 前置: T5, T6, T7
  - 预计: 10 min

- [ ] **T13**: layout 模块文档
  - 文件: `src/layout/mod.rs`
  - 内容: LayoutFlags, compute_f_strides 文档
  - 测试: `cargo doc --no-deps` 无 warning
  - 前置: T5, T6, T7
  - 预计: 10 min

- [ ] **T14**: math 模块逐元素运算文档
  - 文件: `src/math/` 下相关文件
  - 内容: add, sub, mul, div, sin, sqrt, exp, ln, abs 等逐元素运算函数文档和 doctest
  - 测试: `cargo test --doc --all-features`
  - 前置: T5, T6, T7
  - 预计: 10 min

- [ ] **T15**: reduction 与 matrix 模块文档
  - 文件: `src/reduction/`, `src/matrix/` 下相关文件
  - 内容: sum, sum_axis, dot 等归约/内积函数文档和 doctest
  - 测试: `cargo test --doc --all-features`
  - 前置: T5, T6, T7
  - 预计: 10 min

- [ ] **T16**: broadcast 和 shape 模块文档
  - 文件: `src/broadcast/`, `src/shape/mod.rs`
  - 内容: broadcast_shape, transpose 函数文档和 doctest
  - 测试: `cargo test --doc --all-features`
  - 前置: T5, T6, T7
  - 预计: 10 min

- [ ] **T17**: construct 和 set 模块文档
  - 文件: `src/construct/mod.rs`, `src/set/mod.rs`
  - 内容: zeros, ones, eye, from_shape_vec, unique 函数文档和 doctest（`full` 当前版本未提供）；构造错误语义与 `<Owned<A> as StorageOwned>::from_elem` 调用形态须与 `18-construction.md v3.0.1` 一致
  - 测试: `cargo test --doc --all-features`
  - 前置: T5, T6, T7
  - 预计: 10 min

- [ ] **T18**: ffi, workspace, error 模块文档
  - 文件: `src/ffi/mod.rs`, `src/workspace/mod.rs`, `src/error.rs`
  - 内容: FFI 函数（含 Safety 节）、Workspace、XenonError 文档和 doctest；错误字段须对齐 `26-error.md v3.2.0 §5.1`（`TypeConversion` 等使用 `&'static str`），workspace 借用示例须使用 `24-workspace.md v3.0.1` 的 `&mut self` / 消费式 split 形态
  - 测试: `cargo test --doc --all-features`
  - 前置: T5, T6, T7
  - 预计: 10 min

- [ ] **T19**: iter, convert, format, overload 模块文档
  - 文件: `src/iter/mod.rs`, `src/convert/mod.rs`, `src/format/mod.rs`, `src/overload/mod.rs`
  - 内容: 迭代器入口、类型转换、输出格式化、运算符语法边界的模块文档和 doctest；类型转换示例不得使用运行时类型 ID，运算符示例必须体现 `Output = Result<..., XenonError>` 且不得新增额外的 try 前缀算术方法
  - 测试: `cargo test --doc --all-features`
  - 前置: T5, T6, T7
  - 预计: 10 min

- [ ] **T20**: index 模块函数级文档和 doctest
  - 文件: `src/index/mod.rs`
  - 内容: 索引/切片相关函数的文档和 doctest（参见 `17-indexing.md §1`）；示例只使用 `try_at` / `try_at_mut` / `get` / `get_mut`，不得展示方括号索引语法
  - 测试: `cargo test --doc --all-features`
  - 前置: T5, T6, T7
  - 预计: 10 min

- [ ] **T21**: util 模块函数级文档和 doctest
  - 文件: `src/util/mod.rs`
  - 内容: clip / fill / try_fill / to_contiguous / into_contiguous 等 utility 函数的函数级文档和 doctest（参见 `20-utility.md §1`）
  - 测试: `cargo test --doc --all-features`
  - 前置: T8
  - 预计: 10 min

### Wave 4: 示例程序

- [ ] **T22**: 编写 examples/basic.rs
  - 文件: `examples/basic.rs`
  - 内容: 创建、运算、归约、打印
  - 测试: `cargo run --example basic`
  - 前置: T1
  - 预计: 10 min

- [ ] **T23**: 编写 examples/complex.rs
  - 文件: `examples/complex.rs`
  - 内容: 复数构造、同类型复数算术、显式转换后的运算
  - 测试: `cargo run --example complex`
  - 前置: T1
  - 预计: 10 min

- [ ] **T24**: 编写 examples/broadcasting.rs
  - 文件: `examples/broadcasting.rs`
  - 内容: 广播规则、行/列/标量广播
  - 测试: `cargo run --example broadcasting`
  - 前置: T1
  - 预计: 10 min

- [ ] **T25**: 编写 examples/features.rs
  - 文件: `examples/features.rs`
  - 内容: 可选 feature 的启用方式，以及 `parallel` / `simd` 对公开 API **语义可见性与执行路径**的横向对比（如同一 API 在不同 feature 组合下的行为差异）；不深入单个 feature 的内部实现细节
  - 测试: `cargo run --example features --features parallel,simd`
  - 前置: T1
  - 预计: 10 min

- [ ] **T26**: 编写 examples/simd.rs
  - 文件: `examples/simd.rs`
  - 内容: `simd` feature 专属的**内部加速路径、数据布局前提与回退策略**纵向深入示例；聚焦 SIMD 实现细节而非 feature 横向对比
  - 测试: `cargo run --example simd --features simd`
  - 前置: T1
  - 预计: 10 min

- [ ] **T27**: 编写 examples/ffi.rs
  - 文件: `examples/ffi.rs`
  - 内容: 为上游 C/BLAS 集成展示辅助 API 与兼容性判断
  - 测试: `cargo run --example ffi`
  - 前置: T1
  - 预计: 10 min

- [ ] **T28**: 编写 examples/workspace.rs
  - 文件: `examples/workspace.rs`
  - 内容: 工作空间借用、split 与扩容语义示例
  - 测试: `cargo run --example workspace`
  - 前置: T1
  - 预计: 10 min

- [ ] **T29**: 校验示例与 crate 文档仅声明 `std` 环境
  - 文件: `src/lib.rs`, `README.md`, `examples/`
  - 内容: 清理超范围的平台说明，确保示例与文档默认面向 `std` 环境
  - 测试: `cargo doc --no-deps` 与 `cargo build --examples --all-features`
  - 前置: T1, T3, T22-T28
  - 预计: 10 min

### Wave 5: CI 集成

- [ ] **T30**: 配置 CI 文档验证工作流
  - 文件: `.github/workflows/docs.yml`
  - 内容: missing docs 检查、doctest、示例编译
  - 测试: CI 触发运行
  - 前置: T1-T29
  - 预计: 10 min

### unsafe API 执行清单

须在实现阶段维护一份全项目 unsafe 公开函数清单，并对清单中的**每一个 unsafe 函数**逐项执行以下检查项：

- [ ] Aliasing: 无重叠访问保证
- [ ] Lifetime/Provenance: 指针来源可追溯
- [ ] Initialization: 内存已初始化
- [ ] Bounds: 访问范围在合法边界内
- [ ] Overflow/Layout: 布局前置条件已满足

最低基线至少覆盖 `23-ffi.md` 中的 `from_raw_parts()` / `from_raw_parts_mut()` / `from_raw_parts_owned()` 系列，以及 `24-workspace.md` 中的 `assume_init_*` 系列高风险函数。

### 关键 API 示例矩阵

须在实现阶段维护一份关键 API 清单，每个条目标注是否已有使用示例（doctest 或 `example`）。`需求说明书 §28.1` 要求关键 API 提供使用示例，本矩阵作为验收基线。

---

## 8. 测试计划

### 8.1 验证入口

文档验证的完整 Gate 定义和 CI 执行矩阵参见 §5.11.1 和 `28-tests.md`。本节仅补充文档特有的覆盖要求和边界场景。

### 8.2 Doctest 覆盖要求

| 模块类别                                         | 定性要求                                                                                                      |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| 核心类型（tensor, dimension, storage）           | 核心入口和高频查询方法必须有 doctest                                                                          |
| 运算模块（overload, math, broadcast, reduction） | 代表性运算、广播与错误路径必须有 doctest                                                                      |
| 工具模块（ffi, workspace）                       | 关键 API、feature gate 与 Safety 边界必须有 doctest；`simd` / `parallel` 仅文档化其对公开 API 的 feature 影响 |
| 辅助模块（convert, format, error）               | 至少覆盖构造、基本使用与错误语义                                                                              |
| 迭代与归约模块（iter, reduction, matrix）        | 核心入口、边界行为和错误路径必须可追踪                                                                        |

### 8.3 边界测试场景表

| 场景              | 预期行为                                                              |
| ----------------- | --------------------------------------------------------------------- |
| feature-gated API | 在未启用 feature 时不会出现在文档中，启用后 doctest 通过              |
| `std` 平台边界    | 文档示例默认以 `std` 环境为前提；feature-gated API 需显式标注启用条件 |
| unsafe API 文档   | 必须包含 `# Safety` 且示例不省略关键前置条件                          |
| 大型数组输出示例  | 截断格式与 `22-output.md` 保持一致                                    |

### 8.4 属性测试不变量

| 不变量                               | 验证方式                          |
| ------------------------------------ | --------------------------------- |
| 所有公开 API 都能在 docs.rs 中被发现 | `missing_docs` + docs.rs 构建检查 |
| 所有关键模块都有至少一个可运行示例   | doctest / examples 构建联合验证   |
| 文档中的路径与模块名和架构文档一致   | broken links 检查 + 人工审阅      |

### 8.5 类型边界 / 编译期测试

| 场景                         | 测试方式                                         |
| ---------------------------- | ------------------------------------------------ |
| `unsafe fn` 的 `# Safety` 节 | Gate 1 + Gate 2 联合校验，其中 Gate 2 为权威门禁 |
| feature-gated API 可见性     | docs.rs 构建与条件编译可见性检查                 |
| 公共 API 文档覆盖边界        | Gate 1（missing docs / broken intra-doc links）  |

---

## 9. 错误处理与语义边界

本文档不直接定义错误类型，但要求所有文档示例、`# Errors` 节、panic 说明与 feature-gated 文档行为统一遵循 `26-error.md` 的错误语义边界；文档层负责准确转述，不重新定义公开错误模型。

---

## 10. 设计决策记录

### 决策 1：英文文档

| 属性     | 值                                                              |
| -------- | --------------------------------------------------------------- |
| 决策     | 所有 doc comment 和 README 使用英文                             |
| 理由     | Rust 生态惯例；docs.rs 面向全球开发者（参见 `00-coding.md §7`） |
| 替代方案 | 中文文档 — 放弃，不符合 Rust 社区惯例                           |

### 决策 2：doctest 统一使用 `?`

| 属性     | 值                                                                                 |
| -------- | ---------------------------------------------------------------------------------- |
| 决策     | doctest 在示例天然返回 `Result` 时统一使用 `?`；不再为最小示例保留 `unwrap()` 例外 |
| 理由     | 同时遵循 Rust API Guidelines C-QUESTION-MARK，并避免为纯展示型示例引入多余样板     |
| 替代方案 | 允许最小示例使用 `unwrap()` 以减少样板 — 放弃，与 `00-coding.md §7.4` 要求冲突且增加了审查负担 |

### 决策 3：开发期间 `#![warn(missing_docs)]`，CI 中 deny

| 属性     | 值                                                                                                          |
| -------- | ----------------------------------------------------------------------------------------------------------- |
| 决策     | 开发期间使用 `warn` 级别，CI 中通过 `RUSTDOCFLAGS="-D warnings"` 强制 deny 级别                             |
| 理由     | `需求说明书 §28.1` 要求所有公开 API 有文档；开发期间 warn 允许渐进式补全文档，CI 中 deny 阻止无文档代码合入 |
| 替代方案 | 始终 deny 级别 — 放弃，开发期间过于严格，阻碍快速迭代                                                       |

### 决策 4：按模块组织模块级文档

| 属性     | 值                                                    |
| -------- | ----------------------------------------------------- |
| 决策     | 每个模块的 mod.rs 包含完整的模块概述                  |
| 理由     | 用户从 docs.rs 进入模块时能快速理解模块定位和核心类型 |
| 替代方案 | 仅函数级文档 — 放弃，缺乏模块整体视图                 |

### 决策 5：examples 按场景而非按模块

| 属性     | 值                                                              |
| -------- | --------------------------------------------------------------- |
| 决策     | examples/ 按使用场景（basic/broadcasting/features 等）组织      |
| 理由     | 用户按需求查找示例，而非按源码模块                              |
| 替代方案 | 按源码模块组织 — 放弃，不便于用户理解实际用法                   |

---

## 11. 性能描述

| 方面     | 说明                                                                                                |
| -------- | --------------------------------------------------------------------------------------------------- |
| 构建成本 | 文档方案主要关心 `cargo doc`、`cargo test --doc` 与 examples 构建成本，避免引入额外文档站点生成链路 |
| 运行门禁 | 当前版本以文档完整性与可运行示例为主，不把文档构建耗时定义为正式性能门禁                            |
| 工程增强 | 若后续需要统计 docs CI 时间、broken-link 密度或 missing-docs 趋势，可作为工程增强单独演进           |

---

## 12. 平台与工程约束

| 约束项     | 约束内容                                                          |
| ---------- | ----------------------------------------------------------------- |
| 平台支持   | 文档、doctest 与 examples 默认面向 `std` 环境                     |
| MSRV       | Rust 1.85+                                                        |
| crate 结构 | 文档产物围绕当前单 crate 组织，不维护额外平台模板工程             |
| SemVer     | 无影响；文档组织、doctest 与 examples 策略不单独扩展稳定 API 合约 |
| 最小依赖   | 仅文档化现有 feature 与依赖，不扩展超出需求范围的工程契约         |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
