# 格式化输出模块设计

> 文档编号: 22 | 模块: `src/format/` | 阶段: Phase 4
> 前置文档: `07-tensor.md`
> 需求参考: 需求说明书 §24

---

## 1. 模块定位

### 1.1 职责边界

| 职责 | 包含 | 不包含 |
|------|------|--------|
| Display 实现 | 面向用户的简洁可读输出 | serde 序列化 |
| Debug 实现 | 面向开发的形状/步长/类型信息 | 文件 I/O（读写文件） |
| NumPy 风格输出 | 嵌套括号、矩阵形式、按逻辑索引顺序展示 | HTML 渲染 |
| 截断规则 | 超过阈值触发 `...` 省略 | 自定义格式化器注册 |

### 1.2 设计原则

| 原则 | 体现 |
|------|------|
| NumPy 对齐 | 输出格式与 NumPy `np.array_repr` 尽可能一致 |
| 可配置截断 | 阈值/边缘元素数通过常量或 `FormatConfig` 配置 |
| no_std 兼容 | `Display` 和 `Debug` 均通过 `core::fmt` 在 `no_std` 下可用 |
| 零拷贝 | 格式化过程不修改原始数据 |

### 1.3 在架构中的位置

```
依赖层级：
L0: error, private
L1: dimension, element, complex
L2: layout (依赖 dimension)
L3: storage (独立于 layout，由 tensor 持有并消费 layout 结果)
L4: tensor (依赖 storage, dimension)
L5: broadcast, iter, ffi
L6: math, matrix, reduction, shape, index, util
L7: format  ← 当前模块
```

---

## 2. 文件位置

```
src/
└── format/
    ├── mod.rs         # 模块根，re-exports，cfg gates
    ├── config.rs      # FormatConfig 配置结构体及 Default 实现
    ├── display.rs     # Display trait 实现（基于 core::fmt，无 std gate）
    ├── debug.rs       # Debug trait 实现
    └── pretty.rs      # NumPy 风格格式化辅助函数（fmt_1d, fmt_nd, 截断规则）
```

多文件设计：将格式化输出按职责拆分为多个文件，便于后期拓展和维护。

| 文件 | 职责 |
|------|------|
| `mod.rs` | 模块入口，导出公共 API，cfg 门控 |
| `config.rs` | `FormatConfig` 结构体及 `Default` 实现 |
| `display.rs` | `Display` trait 实现（1D/ND 格式化入口） |
| `debug.rs` | `Debug` trait 实现（含元信息） |
| `pretty.rs` | NumPy 风格格式化辅助函数（`fmt_1d_display`, `fmt_nd_display`, 截断逻辑） |

---

## 3. 依赖关系

### 3.1 依赖图

```
src/format/
|
├── mod.rs
│   └── re-exports from config, display, debug, pretty
|
├── config.rs
│   └── (无外部依赖，仅 core/alloc)
|
├── display.rs
│   ├── crate::tensor        # TensorBase<S, D>, shape(), ndim()
│   ├── crate::dimension     # Dimension trait, ndim()
│   ├── crate::storage       # Storage trait
│   ├── crate::element       # Element trait
│   └── super::pretty        # fmt_1d_display, fmt_nd_display
|
├── debug.rs
│   ├── crate::tensor        # TensorBase<S, D>, shape(), strides()
│   ├── crate::storage       # Storage trait
│   ├── crate::element       # Element trait, type_name()
│   └── super::pretty        # fmt_1d_display, fmt_nd_display
|
└── pretty.rs
    ├── crate::tensor        # TensorBase<S, D>
    ├── crate::dimension     # Dimension trait
    ├── crate::storage       # Storage trait
    ├── crate::element       # Element trait
    └── super::config        # FormatConfig
```

### 3.2 类型级依赖

| 来源模块 | 使用的类型/trait | 使用者 |
|----------|-----------------|--------|
| `tensor` | `TensorBase<S, D>`, `.shape()`, `.ndim()`, `.len()`（参见 `07-tensor.md` §4） | `display.rs`, `debug.rs`, `pretty.rs` |
| `dimension` | `Dimension`（参见 `02-dimension.md` §4） | `display.rs`, `pretty.rs` |
| `storage` | `Storage<Elem=A>`（参见 `05-storage.md` §4） | `display.rs`, `debug.rs`, `pretty.rs` |
| `element` | `Element`, `type_name::<A>()`（参见 `03-element.md` §3） | `display.rs`, `debug.rs` |

### 3.3 依赖方向声明

> **依赖方向：单向向上。** `format` 仅消费 `tensor` 等核心模块，不被它们依赖。

---

## 4. 公共 API 设计

### 4.1 FormatConfig 配置结构体

```rust
/// Formatting output configuration.
///
/// Controls truncation behavior and display parameters for large arrays.
pub struct FormatConfig {
    /// Number of edge items (elements/rows shown on each side).
    ///
    /// Defaults to 3, showing the first 3 and last 3 elements.
    pub edge_items: usize,

    /// Minimum total elements to trigger truncation.
    ///
    /// Truncation is enabled when element count exceeds this value. Defaults to 1000.
    pub threshold: usize,

    /// Floating point precision (decimal places).
    ///
    /// Defaults to None (uses the type's default formatting).
    pub precision: Option<usize>,

    /// Line width (characters), used for line-break decisions.
    ///
    /// Defaults to 80.
    pub line_width: usize,
}

impl Default for FormatConfig {
    fn default() -> Self {
        Self {
            edge_items: 3,
            threshold: 1000,
            precision: None,
            line_width: 80,
        }
    }
}
```

### 4.1b TensorDisplay 包装结构

```rust
/// A wrapper for formatting a tensor with a specific config.
pub struct TensorDisplay<'a, S, D, A>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    tensor: &'a TensorBase<S, D>,
    config: FormatConfig,
}

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element,
{
    /// Returns a display wrapper that formats this tensor with the given config.
    ///
    /// # Examples
    ///
    /// ```
    /// let tensor = Tensor1::from_vec(vec![1.0, 2.0, 3.0]);
    /// let config = FormatConfig { precision: Some(2), ..Default::default() };
    /// println!("{}", tensor.display_with(config));
    /// ```
    pub fn display_with(&self, config: FormatConfig) -> TensorDisplay<'_, S, D, A> {
        TensorDisplay { tensor: self, config }
    }
}

impl<S, D, A> core::fmt::Display for TensorDisplay<'_, S, D, A>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: core::fmt::Display + Element,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Delegate to the same formatting logic, passing self.config
        // for precision/edge_items/threshold configuration.
        fmt_with_config(f, self.tensor, &self.config)
    }
}
```

### 4.2 Display 实现

> **注意**：`core::fmt::Display` 在 Rust 1.85 中对 f32/f64 无需 `std` 即可使用，因此此实现不加 `#[cfg(feature = "std")]` 门控。

```rust
// Display is available in no_std via core::fmt
impl<S, D, A> core::fmt::Display for TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: core::fmt::Display + Element,
{
    /// User-facing concise readable output.
    ///
    /// Follows NumPy style:
    /// - 1D: `[1, 2, 3, 4]`
/// - 2D: matrix form, displayed by logical row/column structure while preserving Xenon's F-order storage model internally
    /// - ND: nested brackets
    ///
    /// Large arrays are automatically truncated (see §4.4).
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.ndim() == 0 {
            // 0-dim tensor: output scalar value directly
            // Zero-dimensional tensor element access via NdIndex<Ix0>,
            // tensor[&[]] corresponds to Index<[usize; 0]> trait (see 17-indexing.md §4.0).
            write!(f, "{}", self[&[]])
        } else if self.ndim() == 1 {
            fmt_1d_display(f, self)
        } else {
            fmt_nd_display(f, self)
        }
    }
}
```

### 4.3 Debug 实现

```rust
// Debug is available in no_std via core::fmt; Debug reuses Display for the data section,
// so A must satisfy both Debug and Display.
impl<S, D, A> core::fmt::Debug for TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: core::fmt::Debug + core::fmt::Display + Element,
{
    /// Developer-facing debug output.
    ///
    /// Includes metadata such as shape, strides, and type, then delegates
    /// data formatting to Display.
    ///
    /// # Output Format
    ///
    /// ```text
    /// Tensor(shape=[3, 4], strides=[1, 3], dtype=f64, f-contiguous)
    /// [[1.0, 2.0, 3.0, 4.0],
    ///  [5.0, 6.0, 7.0, 8.0],
    ///  [9.0, 10.0, 11.0, 12.0]]
    /// ```
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Tensor(shape={:?}, strides={:?}, dtype={}, ",
            self.shape(),
            self.strides(),
            core::any::type_name::<A>()
        )?;
        // Layout information
        if self.is_f_contiguous() {
            write!(f, "f-contiguous")?;
        } else {
            write!(f, "non-contiguous")?;
        }
        write!(f, ")\n")?;
        // Data section (reuses Display formatting logic)
        core::fmt::Display::fmt(self, f)
    }
}
```

> **设计决策：** Debug 输出包含完整的元信息（形状/步长/类型/布局），方便开发调试。Display 只输出数据，面向最终用户。Debug 和 Display 统一使用 `core::fmt`，均在 `no_std` 下可用。

### 4.4 NumPy 风格输出示例

**1D（完整）**:

```
[1, 2, 3, 4, 5]
```

**2D（完整）**:

```
[[1, 2, 3],
 [4, 5, 6],
 [7, 8, 9]]
```

**3D（完整）**:

```
[[[1, 2],
  [3, 4]],
 [[5, 6],
  [7, 8]]]
```

**大数组（截断，默认 edge_items=3）**:

```
[[1, 2, 3, ..., 98, 99, 100],
 [101, 102, 103, ..., 198, 199, 200],
 [201, 202, 203, ..., 298, 299, 300],
 ...,
 [9701, 9702, 9703, ..., 9798, 9799, 9800],
 [9801, 9802, 9803, ..., 9898, 9899, 9900],
 [9901, 9902, 9903, ..., 9998, 9999, 10000]]
```

**Complex<f64> 类型**:

```
[[1.0+2.0j, 3.0+4.0j],
 [5.0+6.0j, 7.0+8.0j]]
```

### 4.5 截断规则

```
truncation_rule(tensor, config):
    total = tensor.len()
    if total <= config.threshold:
        display all elements
    else:
        for each axis:
            show first config.edge_items elements
            show "..."
            show last config.edge_items elements
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `edge_items` | 3 | 每边显示的元素/行/列数 |
| `threshold` | 1000 | 触发截断的最小元素总数 |
| `precision` | `None` | 浮点精度（None = 类型默认） |
| `line_width` | 80 | 每行最大字符数（用于换行） |

### 4.6 Good/Bad 对比

```rust
// Good - Use Display for readable output
let tensor = Tensor2::<f64>::zeros([3, 4]);
println!("{}", tensor);  // NumPy style output

// Bad - Manual string concatenation
let tensor = Tensor2::<f64>::zeros([3, 4]);
for i in 0..3 {
    for j in 0..4 {
        print!("{} ", tensor[[i, j]]);  // unreadable, no truncation
    }
    println!();
}
```

```rust
// Good - Use Debug for debug information
let tensor = Tensor2::<f64>::zeros([3, 4]);
println!("{:?}", tensor);
// Tensor(shape=[3, 4], strides=[1, 3], dtype=f64, f-contiguous)
// [[0.0, 0.0, 0.0, 0.0],
//  [0.0, 0.0, 0.0, 0.0],
//  [0.0, 0.0, 0.0, 0.0]]

// Bad - Print each field manually
println!("shape: {:?}", tensor.shape());
println!("strides: {:?}", tensor.strides());
// ... verbose and incomplete
```

---

## 5. 内部实现设计

### 5.1 格式化算法

**精度控制**：如果 `FormatConfig::precision` 为 `Some(p)`，浮点数格式化使用 `write!(f, "{:.prec$}", value, prec = p)`；为 `None` 时使用默认精度（即 `write!(f, "{}", value)`）。

```
fmt_1d(tensor, f):
    len = tensor.shape()[0]
    if len > 2 * edge_items and total > threshold:
        write "["
        for i in 0..edge_items:
            write tensor[[i]], ", "
        write "..., "
        for i in (len - edge_items)..len:
            write tensor[[i]]
            if i < len - 1: write ", "
        write "]"
    else:
        write "["
        for i in 0..len:
            write tensor[[i]]
            if i < len - 1: write ", "
        write "]"

fmt_nd(tensor, f, depth):
    if depth == ndim - 1:
        fmt_1d(current_row, f)
    else:
        write "[" * (ndim - depth)
        for each row in tensor:
            if should_truncate:
                show first edge_items rows
                write "..."
                show last edge_items rows
            else:
                fmt_nd(sub_tensor, f, depth + 1)
        write "]" * (ndim - depth)
```

---

## 6. 实现任务拆分

### Wave 1: 基础设施

- [ ] **T1**: 创建 `src/format/` 模块骨架
  - 文件: `src/format/mod.rs`, `src/format/config.rs`
  - 内容: 模块声明、re-exports、`FormatConfig` 结构体及 Default 实现
  - 测试: 编译通过
  - 前置: tensor 模块完成
  - 预计: 5 min

- [ ] **T2**: 实现 NumPy 风格格式化辅助函数
  - 文件: `src/format/pretty.rs`
  - 内容: `fmt_1d_display`, `fmt_1d_debug`, `fmt_nd_display`, `fmt_nd_debug`，一维/多维完整输出和截断输出
  - 测试: `test_fmt_1d_full`, `test_fmt_1d_truncated`
  - 前置: T1
  - 预计: 15 min

### Wave 2: trait 实现

- [ ] **T3**: 实现 `Display` trait
  - 文件: `src/format/display.rs`
  - 内容: `core::fmt::Display` for `TensorBase<S, D>`，委托调用 `pretty.rs`
  - 测试: `test_display_tensor`
  - 前置: T2
  - 预计: 5 min

- [ ] **T4**: 实现 `Debug` trait（含元信息）
  - 文件: `src/format/debug.rs`
  - 内容: `core::fmt::Debug` for `TensorBase<S, D>`，包含形状/步长/类型，委托调用 `pretty.rs`
  - 测试: `test_debug_tensor`
  - 前置: T2
  - 预计: 10 min

### Wave 3: 收尾

- [ ] **T5**: 添加模块文档和 re-exports 完善
  - 文件: `src/format/mod.rs`, `src/format/display.rs`
  - 内容: 模块文档、re-exports 完善（Display/Debug 均无需 std 门控）
  - 测试: `test_no_std_compile`
  - 前置: T3, T4
  - 预计: 5 min

### 并行执行图

```
Wave 1: [T1] → [T2]
                 │
Wave 2:    ┌─────┴─────┐
           │           │
          [T3]        [T4]   (可并行)
           │           │
           └─────┬─────┘
                 │
Wave 3:        [T5]
```

---

## 7. 测试计划

### 7.0 测试分类表

| 测试分类 | 位置 | 说明 |
|----------|------|------|
| 单元测试 | `#[cfg(test)] mod tests` | 验证 `Display`、`Debug` 与截断格式化语义 |
| 集成测试 | `tests/` | 验证 `output` 与 `tensor`、`iter`、`element` 的协同路径 |
| 边界测试 | 同模块测试中标注 | 覆盖空数组、零维张量、阈值截断和 NaN/Inf 输出 |

### 7.1 单元测试清单

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_fmt_1d_full` | 1D 小数组完整输出 `[1, 2, 3]` | 高 |
| `test_fmt_1d_truncated` | 1D 大数组截断 `[1, 2, 3, ..., 98, 99, 100]` | 高 |
| `test_fmt_1d_empty` | 1D 空数组 `[]` | 中 |
| `test_fmt_1d_single` | 1D 单元素 `[42]` | 中 |
| `test_fmt_2d` | 2D 矩阵形式输出 | 高 |
| `test_fmt_3d` | 3D 嵌套括号输出 | 中 |
| `test_fmt_float_precision` | 浮点精度格式化 | 中 |
| `test_fmt_i32` | 整数类型格式化 | 中 |
| `test_fmt_bool` | bool 类型格式化 `[true, false]` | 低 |
| `test_display_tensor` | Display trait 完整流程 | 高 |
| `test_debug_tensor` | Debug trait 含元信息 | 高 |
| `test_fmt_zero_dim` | 零维张量输出标量 | 中 |
| `test_fmt_large_2d_truncated` | 大 2D 数组行列截断 | 高 |

### 7.2 边界测试场景

| 场景 | 预期行为 |
|------|----------|
| 空数组 `shape=[0]` | 输出 `[]` |
| 单元素 `shape=[1]` | 输出 `[42]` |
| 零维张量 | 输出标量值 |
| 1001 元素 1D | 触发截断 |
| 999 元素 1D | 不截断 |
| NaN/Inf | 输出 `NaN`/`inf` |

### 7.3 属性测试不变量

| 不变量 | 测试方法 |
|--------|----------|
| `debug(tensor)` 包含 shape / strides / dtype 元信息 | 随机形状 |
| 截断输出包含 `...` | 大数组 |

### 7.4 集成测试

| 测试文件 | 测试内容 |
|----------|----------|
| `tests/output.rs` | `Display` / `Debug` 与 `tensor` 元数据查询、`iter` 遍历、复数与浮点格式化路径的端到端集成 |

---

## 8. 与其他模块的交互

### 8.1 接口约定

| 方向 | 对方模块 | 接口/类型 | 约定 |
|------|----------|-----------|------|
| `format → tensor` | `tensor` | `.shape()` / `.ndim()` / `.len()` | `Display` 路径读取基础张量元数据，参见 `07-tensor.md` §4 |
| `format → tensor` | `tensor` | `.strides()` / `is_f_contiguous()` | `Debug` 额外输出布局相关元数据，参见 `06-memory.md` §4 |
| `format → storage` | `storage` | `iter()` | 通过迭代器按逻辑顺序遍历元素，参见 `05-storage.md` §4 |
| `format → element` | `element` | `core::any::type_name::<A>()` | 输出 dtype 与元素类型信息，参见 `03-element.md` §3 |

### 8.2 数据流描述

```text
用户调用 `format!("{}", tensor)` / `format!("{:?}", tensor)`
    │
    ├── output 模块先查询 tensor 的 shape / strides / flags / dtype
    ├── 再通过 iter 路径按逻辑顺序读取需要展示的元素
    ├── 若元素总数超过 threshold，则按截断规则挑选 edge items
    └── 最终直接写入 Formatter，不分配额外堆内存
```

---

## 9. 设计决策记录（ADR）

### 决策 1：截断阈值选择

| 属性 | 值 |
|------|-----|
| 决策 | 默认阈值 1000，默认边缘 3 |
| 理由 | 与 NumPy 默认行为一致（`np.set_printoptions(threshold=1000, edgeitems=3)`） |
| 替代方案 | 更小的阈值（如 100） — 放弃，对中等数组也触发截断 |
| 替代方案 | 可配置阈值通过全局变量 — 放弃，全局可变状态不利于并发测试 |

### 决策 2：输出格式与 NumPy 对齐程度

| 属性 | 值 |
|------|-----|
| 决策 | 尽可能对齐 NumPy 风格，但不追求 100% 一致 |
| 理由 | Rust 的 `fmt::Display` 约定与 Python 不同；追求语义一致而非字符级一致 |
| 替代方案 | 100% 复制 NumPy 格式 — 放弃，Rust 类型信息有价值，不应完全省略 |
| 替代方案 | 完全自定义格式 — 放弃，与用户 Python 经验的直觉一致性是目标 |

### 决策 3：Display 和 Debug 在 no_std 下均可使用

| 属性 | 值 |
|------|-----|
| 决策 | `Display` 和 `Debug` 均无条件实现，不加 `#[cfg(feature = "std")]` 门控 |
| 理由 | Rust 1.85+ 中 f32/f64 的 `Display` 在 `core::fmt` 中可用，无需 `std`；`Debug` 始终在 `core::fmt` 中可用 |
| 替代方案 | Display 加 std 门控 — 放弃，Rust 1.85 已支持 no_std 浮点 Display |

---

## 10. 性能考量

| 方面 | 设计决策 |
|------|----------|
| 格式化开销 | O(n)，不可避免（须遍历每个元素） |
| 大数组截断 | 截断后仅格式化 O(edge_items * 2 * ndim) 个元素，非 O(n) |
| 零拷贝 | 格式化过程不修改原始数据 |
| 临时分配 | 格式化过程无堆分配（直接写入 `Formatter`） |

---

## 11. no_std 兼容性

```rust
// Display is available in no_std via core::fmt (Rust 1.85+)
// f32/f64 Display is in core, not gated by std
impl<S, D, A> core::fmt::Display for TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: core::fmt::Display + Element,
{
    // ...
}

// Debug is also available under no_std via core::fmt
impl<S, D, A> core::fmt::Debug for TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
A: core::fmt::Debug + core::fmt::Display + Element,
{
    // ...
}
```

| 特性 | std | no_std |
|------|-----|--------|
| `Display` | ✅ | ✅（通过 `core::fmt`，Rust 1.85+） |
| `Debug` | ✅ | ✅（通过 `core::fmt`） |
| 浮点精度控制 | ✅ | ✅（`core::fmt` 支持） |
| 截断规则 | ✅ | ✅ |

> **与 Feature 矩阵一致**：`01-architecture.md §6` Feature 矩阵中，no_std 列下 `Display 格式化` 应更新为 ✅，与此处定义对齐。
>
> **关于 Rust 1.85 浮点格式化的说明：** 自 Rust 1.85 起，`f32` / `f64` 的 `Display` 已可通过 `core::fmt` 在无 `std` 情况下使用，因此这里不再为 `Display` 添加 `#[cfg(feature = "std")]` 门控。

---

## 版本历史

| 版本 | 日期 |
|------|------|
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-08 |
| 1.0.2 | 2026-04-08 |
| 1.0.3 | 2026-04-08 |
| 1.0.4 | 2026-04-08 |
| 1.1.0 | 2026-04-08 |
| 1.1.1 | 2026-04-08 |
| 1.1.2 | 2026-04-10 |

---

*本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。*
