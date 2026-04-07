# 格式化输出模块设计

> 文档编号: 22 | 模块: `src/format.rs` | 阶段: Phase 4
> 前置文档: `07-tensor.md`
> 需求参考: 需求说明书 §24

---

## 1. 模块定位

### 1.1 职责边界

| 职责 | 包含 | 不包含 |
|------|------|--------|
| Display 实现 | 面向用户的简洁可读输出 | serde 序列化 |
| Debug 实现 | 面向开发的形状/步长/类型信息 | 文件 I/O（读写文件） |
| NumPy 风格输出 | 嵌套括号、矩阵形式、F-order 排列 | HTML 渲染 |
| 截断规则 | 超过阈值触发 `...` 省略 | 自定义格式化器注册 |

### 1.2 设计原则

| 原则 | 体现 |
|------|------|
| NumPy 对齐 | 输出格式与 NumPy `np.array_repr` 尽可能一致 |
| 可配置截断 | 阈值/边缘元素数通过常量或 `FormatConfig` 配置 |
| std feature 依赖 | `Display` 需要 `std`，`no_std` 环境可只实现 `Debug`（`core::fmt`） |
| 零拷贝 | 格式化过程不修改原始数据 |

### 1.3 在架构中的位置

```
依赖层级：
L0: error, private
L1: dimension, element, complex
L2: layout (依赖 dimension)
L3: storage (依赖 layout)
L4: tensor (依赖 storage, dimension)
L5: format  ← 当前模块
```

---

## 2. 文件位置

```
src/
└── format.rs    # Display/Debug 实现、NumPy 风格输出、截断规则
```

单文件设计：格式化输出逻辑内聚性高，代码量适中（~300 行），无需拆分。

---

## 3. 依赖关系

### 3.1 依赖图

```
src/format.rs
├── crate::tensor        # TensorBase<S, D>, shape(), ndim()
├── crate::dimension     # Dimension trait, ndim()
├── crate::storage       # Storage trait
└── crate::element       # Element trait, type_name()
```

### 3.2 类型级依赖

| 来源模块 | 使用的类型/trait |
|----------|-----------------|
| `tensor` | `TensorBase<S, D>`, `.shape()`, `.ndim()`, `.len()` |
| `dimension` | `Dimension` |
| `storage` | `Storage<Elem=A>` |
| `element` | `Element`, `type_name::<A>()` |

### 3.3 依赖方向声明

> **依赖方向：单向向上。** `format` 仅消费 `tensor` 等核心模块，不被它们依赖。

---

## 4. 公共 API 设计

### 4.1 FormatConfig 配置结构体

```rust
/// 格式化输出配置。
///
/// 控制大数组截断行为和显示参数。
pub struct FormatConfig {
    /// 边缘元素数量（每边显示的元素/行数）。
    ///
    /// 默认 3，即显示前 3 个和后 3 个元素。
    pub edge_items: usize,

    /// 触发截断的最小元素总数。
    ///
    /// 元素数超过此值时启用截断。默认 1000。
    pub threshold: usize,

    /// 浮点精度（小数位数）。
    ///
    /// 默认 None（使用类型的默认格式化）。
    pub precision: Option<usize>,

    /// 行宽（字符数），用于换行决策。
    ///
    /// 默认 80。
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

### 4.2 Display 实现

```rust
impl<S, D, A> core::fmt::Display for TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: core::fmt::Display + Element,
{
    /// 面向用户的简洁可读输出。
    ///
    /// 遵循 NumPy 风格：
    /// - 1D: `[1, 2, 3, 4]`
    /// - 2D: 矩阵形式，F-order 排列
    /// - ND: 嵌套括号
    ///
    /// 大数组自动截断（参见 §4.4）。
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.ndim() == 0 {
            // 零维张量：直接输出标量值
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
impl<S, D, A> core::fmt::Debug for TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: core::fmt::Debug + Element,
{
    /// 面向开发的调试输出。
    ///
    /// 包含形状、步长、类型等元信息。
    ///
    /// # 输出格式
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
        // 布局信息
        if self.is_f_contiguous() {
            write!(f, "f-contiguous")?;
        } else if self.is_c_contiguous() {
            write!(f, "c-contiguous")?;
        } else {
            write!(f, "non-contiguous")?;
        }
        write!(f, ")\n")?;
        // 数据部分（复用 Display 的格式化逻辑）
        core::fmt::Display::fmt(self, f)
    }
}
```

> **设计决策：** Debug 输出包含完整的元信息（形状/步长/类型/布局），方便开发调试。Display 只输出数据，面向最终用户。

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
// Good - 使用 Display 获取可读输出
let tensor = Tensor2::<f64>::zeros([3, 4]);
println!("{}", tensor);  // NumPy style output

// Bad - 手动拼接输出
let tensor = Tensor2::<f64>::zeros([3, 4]);
for i in 0..3 {
    for j in 0..4 {
        print!("{} ", tensor[[i, j]]);  // 不可读，无截断
    }
    println!();
}
```

```rust
// Good - 使用 Debug 获取调试信息
let tensor = Tensor2::<f64>::zeros([3, 4]);
println!("{:?}", tensor);
// Tensor(shape=[3, 4], strides=[1, 3], dtype=f64, f-contiguous)
// [[0.0, 0.0, 0.0, 0.0],
//  [0.0, 0.0, 0.0, 0.0],
//  [0.0, 0.0, 0.0, 0.0]]

// Bad - 逐个字段手动打印
println!("shape: {:?}", tensor.shape());
println!("strides: {:?}", tensor.strides());
// ... 冗余且不完整
```

---

## 5. 内部实现设计

### 5.1 格式化算法

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

### 5.2 no_std 兼容性

```rust
// Display 需要 std（core::fmt::Display 可用，但格式化浮点精度需要 std）
// 在 no_std 环境下只实现 Debug

#[cfg(feature = "std")]
impl<S, D, A> core::fmt::Display for TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: core::fmt::Display + Element,
{
    // ...
}

// Debug 在 no_std 下也可用
impl<S, D, A> core::fmt::Debug for TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: core::fmt::Debug + Element,
{
    // ...
}
```

| 特性 | std | no_std |
|------|-----|--------|
| `Display` | ✅ | ❌ |
| `Debug` | ✅ | ✅（通过 `core::fmt`） |
| 浮点精度控制 | ✅ | ❌ |
| 截断规则 | ✅ | ✅ |

---

## 6. 实现任务拆分

### Wave 1: 基础设施

- [ ] **T1**: 创建 `src/format.rs` 骨架和 `FormatConfig`
  - 文件: `src/format.rs`
  - 内容: `FormatConfig` 结构体及 Default 实现
  - 测试: 编译通过
  - 前置: tensor 模块完成
  - 预计: 5 min

- [ ] **T2**: 实现 1D 格式化（`fmt_1d_display`, `fmt_1d_debug`）
  - 文件: `src/format.rs`
  - 内容: 一维数组完整输出和截断输出
  - 测试: `test_fmt_1d_full`, `test_fmt_1d_truncated`
  - 前置: T1
  - 预计: 10 min

### Wave 2: 多维格式化

- [ ] **T3**: 实现 ND 格式化（`fmt_nd_display`, `fmt_nd_debug`）
  - 文件: `src/format.rs`
  - 内容: 多维递归格式化，嵌套括号
  - 测试: `test_fmt_2d`, `test_fmt_3d`
  - 前置: T2
  - 预计: 15 min

- [ ] **T4**: 实现 `Display` trait
  - 文件: `src/format.rs`
  - 内容: `core::fmt::Display` for `TensorBase<S, D>`
  - 测试: `test_display_tensor`
  - 前置: T3
  - 预计: 5 min

### Wave 3: Debug 和收尾

- [ ] **T5**: 实现 `Debug` trait（含元信息）
  - 文件: `src/format.rs`
  - 内容: `core::fmt::Debug` for `TensorBase<S, D>`，包含形状/步长/类型
  - 测试: `test_debug_tensor`
  - 前置: T4
  - 预计: 10 min

- [ ] **T6**: 添加 `#[cfg(feature = "std")]` 门控和文档
  - 文件: `src/format.rs`
  - 内容: Display 的 std 门控、模块文档
  - 测试: `test_no_std_compile`
  - 前置: T5
  - 预计: 5 min

### 并行执行图

```
Wave 1: [T1] → [T2]
                │
Wave 2: [T3] → [T4]
                │
Wave 3: [T5] → [T6]
```

---

## 7. 测试计划

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
| `format(tensor).contains(shape)` | 随机形状 |
| 截断输出包含 `...` | 大数组 |

---

## 8. 与其他模块的交互

| 交互点 | 方向 | 说明 |
|--------|------|------|
| `Display` → `tensor` | format → tensor | 读取 `.shape()`, `.ndim()`, `.len()` |
| `Debug` → `tensor` | format → tensor | 额外读取 `.strides()`, `is_f_contiguous()` |
| `Display` → `storage` | format → storage | 通过 `iter()` 遍历元素 |
| `Display` → `element` | format → element | 使用 `core::any::type_name::<A>()` |

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

### 决策 3：Display 依赖 std

| 属性 | 值 |
|------|-----|
| 决策 | `Display` 实现加 `#[cfg(feature = "std")]`，`Debug` 无条件可用 |
| 理由 | `no_std` 环境下浮点格式化受限；`Debug` 通过 `core::fmt` 可用 |
| 替代方案 | 全部无条件实现 — 放弃，`no_std` 浮点 `Display` 可能不可用 |

---

## 10. 性能考量

| 方面 | 设计决策 |
|------|----------|
| 格式化开销 | O(n)，不可避免（须遍历每个元素） |
| 大数组截断 | 截断后仅格式化 O(edge_items * 2 * ndim) 个元素，非 O(n) |
| 零拷贝 | 格式化过程不修改原始数据 |
| 临时分配 | 格式化过程无堆分配（直接写入 `Formatter`） |

---

## 版本历史

| 版本 | 日期 |
|------|------|
| 1.0.0 | 2026-04-07 |

---

*本文档由 Xenon 维护。如有问题请提交 Issue 或 PR。*
