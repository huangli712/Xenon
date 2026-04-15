# 格式化输出模块设计

> 文档编号: 22 | 模块: `src/format/` | 阶段: Phase 4
> 前置文档: `05-storage.md`, `06-layout.md`, `07-tensor.md`
> 需求参考: 需求说明书 §4, §5, §8, §24, §28.1
> 范围声明: 范围内

---

## 1. 模块定位

### 1.1 职责边界

| 职责           | 包含                                   | 不包含               |
| -------------- | -------------------------------------- | -------------------- |
| Display 实现   | 面向用户的简洁可读输出                 | serde 序列化         |
| Debug 实现     | 面向开发的形状/步长/类型信息           | 文件 I/O（读写文件） |
| NumPy 风格输出 | 嵌套括号、矩阵形式、按逻辑索引顺序展示 | HTML 渲染            |
| 截断规则       | 超过阈值触发 `... (N elements omitted)  shape=[...]` 后缀 | 自定义格式化器注册   |

### 1.2 设计原则

| 原则       | 体现                                                     |
| ---------- | -------------------------------------------------------- |
| NumPy 对齐 | 输出格式与 NumPy `np.array_repr` 尽可能一致              |
| 可配置截断 | 阈值/边缘元素数通过常量或 `FormatConfig` 配置            |
| 平台一致性 | `Display` 和 `Debug` 在当前 `std` 环境下保持一致格式语义 |
| 零拷贝     | 格式化过程不修改原始数据                                 |

### 1.3 在架构中的位置

```
Dependency layers:
L0: error, private
L1: dimension, element, complex
L2: layout (depends on dimension)
L3: storage (independent of layout; owned by tensor and consumes layout results)
L4: tensor (depends on storage, dimension)
L5: broadcast, iter, ffi
L6: math, matrix, reduction, shape, index, util
L7: format  <- current module
```

---

## 2. 需求映射与范围约束

| 类型     | 内容 |
| -------- | ---- |
| 需求映射 | 需求说明书 §4, §5, §8, §24, §28.1 |
| 范围内   | `Display` / `Debug`、`FormatConfig`、NumPy 风格多维输出、截断规则与零维张量显式标记。 |
| 范围外   | 二进制序列化、JSON 输出、自定义 formatter 注册与 HTML / 富文本渲染。 |
| 非目标   | 不把格式化模块扩展为序列化层，不新增第三方格式化依赖，也不改变 tensor 数据本身。 |

| 需求条款 | 本文承接方式 |
| -------- | ------------ |
| §4 元素类型 | `Display` / `Debug` 覆盖全部受支持元素类型，包括 `bool` 与复数。 |
| §5 复数类型 | 复数文本表示采用稳定、可读且可区分实部/虚部的输出格式。 |
| §8 张量类型 | 输出可读取 shape / strides / layout / dtype 等元数据，并明确区分标量与零维张量。 |
| §24 格式化输出 | 采用稳定的 NumPy 风格文本表示与统一截断规则。 |
| §28.1 文档 | 关键格式化 API 提供使用示例；示例代码若非完整可编译上下文则标记为 `ignore`。 |

---

## 3. 文件位置

```
src/
└── format/
    ├── mod.rs         # Module root, re-exports, cfg gates
    ├── config.rs      # FormatConfig and Default implementation
    ├── display.rs     # Display trait implementation (based on core::fmt, no std gate)
    ├── debug.rs       # Debug trait implementation
    └── pretty.rs      # NumPy-style formatting helpers (fmt_1d, fmt_nd, truncation rules)
```

多文件设计：将格式化输出按职责拆分为多个文件，便于后期拓展和维护。

| 文件         | 职责                                                                     |
| ------------ | ------------------------------------------------------------------------ |
| `mod.rs`     | 模块入口，导出公共 API，cfg 门控                                         |
| `config.rs`  | `FormatConfig` 结构体及 `Default` 实现                                   |
| `display.rs` | `Display` trait 实现（1D/ND 格式化入口）                                 |
| `debug.rs`   | `Debug` trait 实现（含元信息）                                           |
| `pretty.rs`  | NumPy 风格格式化辅助函数（`fmt_1d_display`, `fmt_nd_display`, 截断逻辑） |

---

## 4. 依赖关系

### 4.1 依赖图

```
src/format/
|
├── mod.rs
│   └── re-exports from config, display, debug, pretty
|
├── config.rs
│   └── (no external dependency, only core/alloc)
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

### 4.2 类型级依赖

| 来源模块    | 使用的类型/trait                                                              | 使用者                                |
| ----------- | ----------------------------------------------------------------------------- | ------------------------------------- |
| `tensor`    | `TensorBase<S, D>`, `.shape()`, `.ndim()`, `.len()`（参见 `07-tensor.md` §5） | `display.rs`, `debug.rs`, `pretty.rs` |
| `dimension` | `Dimension`（参见 `02-dimension.md` §5）                                      | `display.rs`, `pretty.rs`             |
| `storage`   | `Storage<Elem=A>`（参见 `05-storage.md` §5）                                  | `display.rs`, `debug.rs`, `pretty.rs` |
| `element`   | `Element`, `type_name::<A>()`（参见 `03-element.md` §5.1）                    | `display.rs`, `debug.rs`              |

### 4.3 依赖方向声明

> **依赖方向：单向向上。** `format` 仅消费 `tensor` 等核心模块，不被它们依赖。

### 4.4 依赖合法性与替代方案

| 项目           | 说明 |
| -------------- | ---- |
| 新增第三方依赖 | 无 |
| 合法性结论     | 合法；当前设计仅复用 Xenon 既有模块、标准库以及文档中已声明的项目内可选能力。 |
| 替代方案       | 不适用；当前范围内无需额外第三方依赖。 |

---

## 5. 公共 API 设计

### 5.1 FormatConfig 配置结构体

```rust,ignore
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
    /// When a rendered line would exceed this width, insert line breaks
    /// between elements, preferring axis boundaries when possible.
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

### 5.1b TensorDisplay 包装结构

````rust,ignore
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
    /// let tensor = Tensor::from_shape_vec([3], vec![1.0, 2.0, 3.0])
    ///     .expect("shape and data length should match");
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
````

### 5.2 Display 实现

> **读取顺序约定**：格式化输出按**逻辑多维索引顺序**读取元素，而不是按底层物理内存顺序线性扫描。格式化层不得把 `iter()` 的顺序当作公共契约前提；若内部复用 `iter()`，那只应视为私有实现细节，必要时应改为显式逻辑索引或递归子视图遍历。

> **内部访问说明**：内部实现可使用 `read_at(indices)` 之类的辅助函数访问逻辑位置元素；这只是实现细节，**不扩展 `require.md` §18 的公开索引契约**。

> **注意**：`core::fmt::Display` 在 Rust 1.85 中对 f32/f64 无需 `std` 即可使用，因此此实现不加 `#[cfg(feature = "std")]` 门控。

> **默认配置绑定**：`Display for TensorBase` 默认使用 `FormatConfig::default()` 配置；如需自定义，请使用 `display_with(config)` 方法。

```rust,ignore
// Display uses core::fmt and stays available in the std-only baseline.
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
    /// Large arrays are automatically truncated (see §5.5), and any
    /// truncated output appends the full tensor shape after the closing bracket.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.ndim() == 0 {
            // 0-dim tensor: render with an explicit marker to distinguish it
            // from a plain scalar value in textual output.
            // Zero-dimensional tensor element access goes through an internal
            // logical-index helper such as read_at([]). This is an implementation
            // detail and does not extend the public indexing contract.
            write!(f, "Tensor0({})", self.read_at([]))
        } else if self.ndim() == 1 {
            fmt_1d_display(f, self)
        } else {
            fmt_nd_display(f, self)
        }
    }
}
```

### 5.3 Debug 实现

> **Display / Debug 分工约定：** `Display` 只负责数据文本；当发生截断时，它在最外层右括号后追加 `shape=[...]`，用于满足 `require.md §24` 的“可识别 shape”要求。`Debug` 已在头部输出 `shape=`、`strides=`、`dtype=` 和 `layout=`，因此其数据段复用相同截断选点规则，但不再重复追加 `shape=[...]` 后缀。

````rust,ignore
// Debug reuses Display for the data section,
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
    /// When the data section is truncated, Debug keeps the same visible head/tail
    /// selection as Display but omits the trailing `shape=[...]` suffix because
    /// shape metadata is already present in the header.
    ///
    /// # Output Format
    ///
    /// ```text
    /// Tensor(shape=[3, 4], strides=[1, 3], dtype=f64, layout=f-contiguous)
    /// [[1.0, 4.0, 7.0, 10.0],
    ///  [2.0, 5.0, 8.0, 11.0],
    ///  [3.0, 6.0, 9.0, 12.0]]
    /// ```
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Tensor(shape={:?}, strides={:?}, dtype={}, layout=",
            self.shape(),
            self.strides(),
            dtype_name::<A>()
        )?;
        // Layout information
        if self.strides().iter().any(|&stride| stride == 0) {
            write!(f, "broadcast")?;
        } else if self.is_f_contiguous() {
            write!(f, "f-contiguous")?;
        } else {
            write!(f, "non-contiguous")?;
        }
        write!(f, ")\n")?;
        // Data section (shares the same rendering rules as Display, but does
        // not append a second shape suffix when truncation happens).
        fmt_debug_data(f, self)
    }
}
````

> **设计决策：** Debug 输出包含完整的元信息（形状/步长/类型/布局），方便开发调试。Display 只输出数据，面向最终用户；其中零维张量使用显式标记，避免与裸标量文本混淆。

> **布局分类约定：** Debug 至少区分三类布局：`layout=f-contiguous`、`layout=broadcast`（存在零步长）、`layout=non-contiguous`（如转置、切片等非广播非连续布局）。

### 5.4 NumPy 风格输出示例

**1D（完整）**:

```
[1, 2, 3, 4, 5]
```

**2D（完整，shape=[3, 3]，底层数据按 F-order 为 `[1, 2, 3, 4, 5, 6, 7, 8, 9]`）**:

```
[[1, 4, 7],
 [2, 5, 8],
 [3, 6, 9]]
```

**Debug 元信息示例（连续 / broadcast / 非连续）**:

```text
Tensor(shape=[3, 4], strides=[1, 3], dtype=f64, layout=f-contiguous)
Tensor(shape=[3, 4], strides=[1, 0], dtype=f64, layout=broadcast)
Tensor(shape=[3, 4], strides=[4, 1], dtype=f64, layout=non-contiguous)
```

**3D（完整，shape=[2, 2, 2]，底层数据按 F-order 为 `[1, 2, 3, 4, 5, 6, 7, 8]`）**:

```
[[[1, 5],
  [3, 7]],
 [[2, 6],
  [4, 8]]]
```

**1D 大数组（截断，默认 `edge_items=3`，`threshold=1000`，元素数 `> 1000` 时触发）**:

```
[1, 2, 3, ..., 999, 1000, 1001] ... (995 elements omitted)  shape=[1001]
```

**2D 大数组（截断，默认 `edge_items=3`，shape=[100, 100]，底层数据按 F-order 为 `1..=10000`）**:

```
[[1, 101, 201, ..., 9701, 9801, 9901],
 [2, 102, 202, ..., 9702, 9802, 9902],
 [3, 103, 203, ..., 9703, 9803, 9903],
 ...,
 [98, 198, 298, ..., 9798, 9898, 9998],
 [99, 199, 299, ..., 9799, 9899, 9999],
 [100, 200, 300, ..., 9800, 9900, 10000]] ... (9964 elements omitted)  shape=[100, 100]
```

> 当任意维度触发截断时，输出主体仍保持 NumPy 风格的局部预览，但必须在最外层右括号后追加 `shape=[...]`，以暴露完整维度信息。

**Complex<f64> 类型**:

```
[[1.0+2.0j, 5.0+6.0j],
 [3.0+4.0j, 7.0+8.0j]]
```

**零维张量**:

```
Tensor0(42)
```

### 5.5 截断规则

#### 5.5.1 形式化截断算法

1. 截断决策仅由 `threshold` 决定：当 `tensor.len() <= threshold` 时，输出全部逻辑元素；当 `tensor.len() > threshold` 时，进入截断模式。
2. 截断模式下，每一层轴只使用一个局部规则：
   - 若当前轴长度 `axis_len <= 2 * edge_items`，该轴完整显示；
   - 若 `axis_len > 2 * edge_items`，该轴仅显示前 `edge_items` 项、一个 `...` 标记、以及后 `edge_items` 项。
3. `visible_elements` 定义为最终真实打印出的逻辑元素数量，不包含任何 `...` 标记、逗号、括号、换行或 `shape=[...]` 后缀。
4. `omitted = tensor.len() - visible_elements`。只有在 `tensor.len() > threshold` 且至少一个轴满足 `axis_len > 2 * edge_items` 时，`omitted` 才会大于 `0`。
5. `Display` 与 `Debug` 共享完全相同的元素选点规则与 `visible_elements` / `omitted` 计算规则；差异仅在元信息呈现：
   - `Display` 只输出数据文本；若 `omitted > 0`，则在最外层右括号后追加 ` ... (N elements omitted)  shape=[...]`；
   - `Debug` 先输出 `shape=` / `strides=` / `dtype=` / `layout=` 头部；若 `omitted > 0`，则只在数据段末尾追加 ` ... (N elements omitted)`，不重复追加 `shape=[...]`。
6. `line_width` 仅影响换行位置；它不得改变是否截断、每层保留的头尾项数量、`visible_elements` 或 `omitted`。

> **单一执行规范：** 当前版本不定义 `max_display_elements`、`max_rows` 或其他额外截断阈值；所有截断行为都只由 `threshold` 与 `edge_items` 共同决定。

```
truncation_rule(tensor, config, mode):
    truncated = tensor.len() > config.threshold
    rendered, visible_elements = render_axis(tensor, config, axis = 0, prefix = [], truncated)
    omitted = tensor.len() - visible_elements

    if omitted == 0:
        return rendered
    if mode == Display:
        return rendered + " ... (" + omitted + " elements omitted)  shape=" + tensor.shape()
    return rendered + " ... (" + omitted + " elements omitted)"

render_axis(tensor, config, axis, prefix, truncated):
    if axis == tensor.ndim():
        return render_scalar(prefix), 1

    axis_len = tensor.shape()[axis]
    if !truncated || axis_len <= 2 * config.edge_items:
        entries = [0, 1, ..., axis_len - 1]
    else:
        k = config.edge_items
        entries = [0, 1, ..., k - 1, Ellipsis, axis_len - k, ..., axis_len - 1]

    rendered = "["
    visible = 0
    for entry in entries:
        if entry == Ellipsis:
            rendered += "..."
        else:
            child_rendered, child_visible = render_axis(
                tensor,
                config,
                axis + 1,
                prefix + [entry],
                truncated,
            )
            rendered += child_rendered
            visible += child_visible
        rendered += separator_for_axis(axis, entry)
    rendered += "]"
    return rendered, visible
```

在该规则下，若 rank 为 `n` 且截断模式下第 `i` 个轴实际显示 `shown_i` 个索引位置（`shown_i = axis_len_i` 或 `2 * edge_items`），则：

`visible_elements = Π shown_i`（对所有轴求乘积）。

因此 `shape=[100, 100]`、`edge_items=3`、`threshold=1000` 时：

- 每个轴都显示 `6` 个索引位置；
- `visible_elements = 6 × 6 = 36`；
- `omitted = 10000 - 36 = 9964`。

`line_width` 行为：当一行输出超过 `line_width` 字符时，在元素之间插入换行，优先在轴边界处折行。

| 参数         | 默认值 | 说明                        |
| ------------ | ------ | --------------------------- |
| `edge_items` | 3      | 每层轴在截断时保留的头/尾项数 |
| `threshold`  | 1000   | 元素总数严格大于该值时触发截断 |
| `precision`  | `None` | 浮点精度（None = 类型默认） |
| `line_width` | 80     | 每行最大字符数（用于换行）  |

### 5.6 Good/Bad 对比

```rust,ignore
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

```rust,ignore
// Good - Use Debug for debug information
let tensor = Tensor2::<f64>::zeros([3, 4]);
println!("{:?}", tensor);
// Tensor(shape=[3, 4], strides=[1, 3], dtype=f64, layout=f-contiguous)
// [[0.0, 0.0, 0.0, 0.0],
//  [0.0, 0.0, 0.0, 0.0],
//  [0.0, 0.0, 0.0, 0.0]]

// Bad - Print each field manually
println!("shape: {:?}", tensor.shape());
println!("strides: {:?}", tensor.strides());
// ... verbose and incomplete
```

---

## 6. 内部实现设计

### 6.1 格式化算法

**精度控制**：如果 `FormatConfig::precision` 为 `Some(p)`，浮点数格式化使用 `write!(f, "{:.prec$}", value, prec = p)`；为 `None` 时使用默认精度（即 `write!(f, "{}", value)`）。

```
fmt_1d(tensor, f):
    total = tensor.len()
    len = tensor.shape()[0]
    if len > 2 * edge_items and total > threshold:
        write "["
        for i in 0..edge_items:
            write read_at([i]), ", "
        write "..., "
        for i in (len - edge_items)..len:
            write read_at([i])
            if i < len - 1: write ", "
        omitted = total - 2 * edge_items
        write "] ... (" + omitted + " elements omitted)  shape=" + tensor.shape()
    else:
        write "["
        for i in 0..len:
            write read_at([i])
            if i < len - 1: write ", "
        write "]"

fmt_nd(tensor, f, prefix):
    total = tensor.len()
    axis = prefix.len()
    if axis == tensor.ndim():
        write read_at(prefix)
        return

    write "["
    indices = logical_indices_for_axis(tensor.shape()[axis], edge_items, total > threshold)
    for (pos, entry) in indices.enumerate():
        if entry == Ellipsis:
            write "..."
        else:
            next_prefix = prefix + [entry]
            // Outermost dimension is axis 0. For shape [M, N], this emits M rows,
            // and each displayed element at [i, j] is tensor[[i, j]] in logical F-order.
            fmt_nd(tensor, f, next_prefix)
        if pos < indices.len() - 1:
            write separator_for_axis(axis)

    if axis == 0 and total > threshold:
        omitted = total - count_displayed_elements(tensor.shape(), edge_items)
        write "] ... (" + omitted + " elements omitted)  shape=" + tensor.shape()
    else:
        write "]"
```

> 对 F-order 张量，格式化必须按**逻辑索引**而不是物理线性内存顺序展开。以 `shape=[3, 3]` 为例，显示位置 `[i, j]` 对应逻辑索引 `[i, j]`，其线性位置为 `i + j * 3`；因此输出为 `[[1, 4, 7], [2, 5, 8], [3, 6, 9]]`，而不是按物理连续内存直接切成 `[[1, 2, 3], [4, 5, 6], [7, 8, 9]]`。内部若使用 `read_at(indices)` 等辅助函数，仅表示实现通过逻辑坐标取值，不构成新的公开索引承诺。

### 6.2 dtype 名称映射

```rust,ignore
fn dtype_name<A: Element + 'static>() -> &'static str {
    match core::any::TypeId::of::<A>() {
        id if id == core::any::TypeId::of::<f32>() => "f32",
        id if id == core::any::TypeId::of::<f64>() => "f64",
        id if id == core::any::TypeId::of::<i32>() => "i32",
        id if id == core::any::TypeId::of::<i64>() => "i64",
        id if id == core::any::TypeId::of::<Complex<f32>>() => "Complex<f32>",
        id if id == core::any::TypeId::of::<Complex<f64>>() => "Complex<f64>",
        id if id == core::any::TypeId::of::<bool>() => "bool",
        _ => core::any::type_name::<A>(),
    }
}
```

> Debug 输出的 `dtype=` 字段应通过内部 `dtype_name()` 映射获得稳定、紧凑的展示名，而不是直接暴露编译器的完整类型路径。

---

## 7. 实现任务拆分

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
  - 预计: 10 min

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
  - 测试: `test_display_compile`
  - 前置: T3, T4
  - 预计: 5 min

### 并行执行图

```
Wave 1: [T1] → [T2]
                 │
Wave 2:    ┌─────┴─────┐
           │           │
          [T3]        [T4]   (can run in parallel)
           │           │
           └─────┬─────┘
                 │
Wave 3:        [T5]
```

---

## 8. 测试计划

### 8.1 测试分类表

| 测试分类 | 位置                     | 说明                                                    |
| -------- | ------------------------ | ------------------------------------------------------- |
| 单元测试 | `#[cfg(test)] mod tests` | 验证 `Display`、`Debug` 与截断格式化语义                |
| 集成测试 | `tests/`                 | 验证 `output` 与 `tensor`、`iter`、`element` 的协同路径 |
| 边界测试 | 同模块测试中标注         | 覆盖空数组、零维张量、阈值截断和 NaN/Inf 输出           |
| 属性测试 | `tests/property/`        | 验证截断阈值、逻辑顺序与格式配置不变量                  |

### 8.2 单元测试清单

| 测试函数                      | 测试内容                                    | 优先级 |
| ----------------------------- | ------------------------------------------- | ------ |
| `test_fmt_1d_full`            | 1D 小数组完整输出 `[1, 2, 3]`               | 高     |
| `test_fmt_1d_truncated`       | 1D 大数组截断，并追加统一后缀 `... (N elements omitted)  shape=[...]` | 高  |
| `test_fmt_1d_empty`           | 1D 空数组 `[]`                              | 中     |
| `test_fmt_1d_single`          | 1D 单元素 `[42]`                            | 中     |
| `test_fmt_2d`                 | 2D 矩阵形式输出                             | 高     |
| `test_fmt_3d`                 | 3D 嵌套括号输出                             | 中     |
| `test_fmt_float_precision`    | 浮点精度格式化                              | 中     |
| `test_display_complex_f64`    | 复数格式化输出（如 `1.0+2.0j`）             | 中     |
| `test_display_complex_negative_imag` | 复数负虚部格式化                       | 中     |
| `test_fmt_i32`                | 整数类型格式化                              | 中     |
| `test_fmt_bool`               | bool 类型格式化 `[true, false]`             | 低     |
| `test_display_tensor`         | Display trait 完整流程                      | 高     |
| `test_debug_tensor`           | Debug trait 含元信息，且数据段不重复输出 `shape=[...]` | 高     |
| `test_fmt_zero_dim`           | 零维张量输出带区分标记                      | 中     |
| `test_fmt_large_2d_truncated` | 大 2D 数组行列截断                          | 高     |
| `test_line_width_wrapping`    | 多行输出遵守配置的 `line_width`             | 中     |
| `test_line_width_narrow`      | 窄行宽配置触发更早换行                      | 中     |

### 8.3 边界测试场景

| 场景               | 预期行为                            |
| ------------------ | ----------------------------------- |
| 空数组 `shape=[0]` | 输出 `[]`                           |
| 单元素 `shape=[1]` | 输出 `[42]`                         |
| 零维张量           | 输出 `Tensor0(...)`，与裸标量可区分 |
| 1001 元素 1D       | 触发截断，并在尾部输出 `... (N elements omitted)  shape=[1001]` |
| 1000 元素 1D       | 不截断                              |
| NaN/Inf            | 输出 `NaN`/`inf`                    |

### 8.4 属性测试不变量

| 不变量                                              | 测试方法 |
| --------------------------------------------------- | -------- |
| `debug(tensor)` 包含 shape / strides / dtype 元信息 | 随机形状 |
| 截断输出包含统一后缀 `... (N elements omitted)  shape=[...]` | 大数组 |

### 8.5 集成测试

| 测试文件               | 测试内容                                                                                  |
| ---------------------- | ----------------------------------------------------------------------------------------- |
| `tests/test_output.rs` | `Display` / `Debug` 与 `tensor` 元数据查询、`iter` 遍历、复数与浮点格式化路径的端到端集成 |

### 8.6 Feature gate / 配置测试

| 配置 | 验证点 |
| ---- | ---- |
| 默认配置 | `Display` / `Debug` / `FormatConfig::default()` 输出满足 NumPy 风格与零维区分契约。 |
| 其他 feature 组合 | 不适用；当前模块无额外 feature gate。 |

### 8.7 类型边界 / 编译期测试

| 场景 | 测试方式 |
| ---- | ---- |
| `Display` / `Debug` 仅对满足对应 fmt trait 的元素类型开放 | 编译期测试。 |
| `TensorDisplay` 包装器保留原张量维度与借用生命周期 | 编译期测试。 |
| binary / JSON / custom formatter APIs 不属于当前范围 | API 缺失断言。 |

---

## 9. 与其他模块的交互

### 9.1 接口约定

| 方向                    | 对方模块          | 接口/类型                          | 约定                                                         |
| ----------------------- | ----------------- | ---------------------------------- | ------------------------------------------------------------ |
| `format → tensor`       | `tensor`          | `.shape()` / `.ndim()` / `.len()`  | `Display` 路径读取基础张量元数据，参见 `07-tensor.md` §5     |
| `format → tensor`       | `tensor`          | `.strides()` / `is_f_contiguous()` | `Debug` 额外输出布局相关元数据，参见 `06-layout.md` §5       |
| `format → tensor/index` | `tensor`, `index` | `shape()`, 内部 `read_at(indices)` | 按逻辑行/列结构读取元素；不依赖 `iter()` 的 F-order 内存顺序，且不扩展公开索引契约 |
| `format → element`      | `element`         | 内部 `dtype_name::<A>()`           | 输出稳定 dtype 名称与元素类型信息，参见 `03-element.md` §5.1 |

### 9.2 数据流描述

```text
User calls format!("{}", tensor) / format!("{:?}", tensor)
    │
    ├── format queries tensor shape / strides / flags / dtype metadata
    ├── pretty recursively reads the required logical elements
    ├── large arrays are truncated according to threshold and edge-items rules
    ├── truncated output appends omitted-element count and full shape metadata
    └── the module writes directly into Formatter without heap allocation
```

---

## 10. 错误处理与语义边界

| 主题 | 内容 |
| ---- | ---- |
| Recoverable error | 不适用；按 `26-error.md` 的边界约定，当前格式化 API 只通过 `fmt::Result` 与格式化器交互，不额外引入 `XenonError` 变体。 |
| Panic | 不适用；公开格式化路径不引入新的 panic 语义。 |
| 路径一致性 | `Display`、`Debug` 与 `display_with(config)` 必须共享同一逻辑索引读取与截断契约；无 SIMD / 并行分支。 |
| 容差边界 | 不适用；`precision` 仅影响文本呈现，不构成数值误差容差语义。 |

---

## 11. 设计决策记录（ADR）

### 决策 1：截断阈值选择

| 属性     | 值                                                                          |
| -------- | --------------------------------------------------------------------------- |
| 决策     | 默认阈值 1000，默认边缘 3                                                   |
| 理由     | 与 NumPy 默认行为一致（`np.set_printoptions(threshold=1000, edgeitems=3)`） |
| 替代方案 | 更小的阈值（如 100） — 放弃，对中等数组也触发截断                           |
| 替代方案 | 可配置阈值通过全局变量 — 放弃，全局可变状态不利于并发测试                   |

### 决策 2：输出格式与 NumPy 对齐程度

| 属性     | 值                                                                    |
| -------- | --------------------------------------------------------------------- |
| 决策     | 尽可能对齐 NumPy 风格，但不追求 100% 一致                             |
| 理由     | Rust 的 `fmt::Display` 约定与 Python 不同；追求语义一致而非字符级一致 |
| 替代方案 | 100% 复制 NumPy 格式 — 放弃，Rust 类型信息有价值，不应完全省略        |
| 替代方案 | 完全自定义格式 — 放弃，与用户 Python 经验的直觉一致性是目标           |

### 决策 3：零维张量使用显式标记

| 属性     | 值                                                             |
| -------- | -------------------------------------------------------------- |
| 决策     | 零维张量输出采用 `Tensor0(...)` 形式，而不是直接打印裸元素     |
| 理由     | 满足 `require.md §24` 中“以可区分方式显示标量与零维张量”的要求 |
| 替代方案 | 直接输出裸标量 — 放弃，会与普通标量文本混淆                    |

---

## 12. 性能考量

| 方面       | 设计决策                                                 |
| ---------- | -------------------------------------------------------- |
| 格式化开销 | 非截断输出 O(n)                                          |
| 大数组截断 | 截断输出 O(visible_elements + overhead)                  |
| 零拷贝     | 格式化过程不修改原始数据                                 |
| 临时分配   | 格式化过程尽量避免中间字符串构造，直接写入 `Formatter`；动态维度可能需要少量索引缓冲，但不产生格式化结果字符串本身的堆分配 |

---

## 13. 平台与工程约束

| 约束       | 说明                                                  |
| ---------- | ----------------------------------------------------- |
| `std` only | 当前版本仅讨论 `std` 环境下的格式化输出行为           |
| 单 crate   | 格式化逻辑保持在 `src/format/` 内，不拆出独立 crate   |
| SemVer     | 0D 文本表示属于公开输出契约，后续变更需视为兼容性事项 |
| 最小依赖   | 不引入额外第三方格式化依赖                            |

---

## 版本历史

| 版本  | 日期       |
| ----- | ---------- |
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-08 |
| 1.0.2 | 2026-04-08 |
| 1.0.3 | 2026-04-08 |
| 1.0.4 | 2026-04-08 |
| 1.1.0 | 2026-04-08 |
| 1.1.1 | 2026-04-08 |
| 1.1.2 | 2026-04-10 |
| 1.1.3 | 2026-04-14 |
| 1.1.4 | 2026-04-14 |
| 1.1.5 | 2026-04-15 |
| 1.1.6 | 2026-04-15 |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
