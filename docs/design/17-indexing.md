# 索引操作模块设计

> 文档编号: 17 | 模块: `src/index/` | 阶段: Phase 3
> 前置文档: `02-dimension.md`, `06-layout.md`, `07-tensor.md`, `26-error.md`
> 需求参考: 需求说明书 §18
> 范围声明: 范围内

---

## 1. 模块定位/概述

### 1.1 职责边界表

| 职责 | 包含 | 不包含 |
| --- | --- | --- |
| 多维整数索引 | `usize` 多维索引、`try_at` / `try_at_mut` / `get` / `get_mut` / `get_unchecked*` | 负索引、布尔掩码、整数数组高级索引 |
| 范围索引（切片） | 以范围描述符表达的只读切片、`slice`、`s![]` 宏 | 负步长切片、共享可写切片、隐式复制切片 |
| 元数据更新 | 按 F-order 规则更新 offset、shape、stride 与布局标记 | 改变逻辑元素顺序、引入与源张量无关的新存储 |
| 错误边界 | 安全接口返回可恢复错误，unsafe 接口由调用方保证前提 | 将越界默认为 panic 的安全主路径 |

### 1.2 设计原则

| 原则 | 体现 |
| --- | --- |
| `usize` 元数据角色 | `usize` 仅用于索引、轴、shape 与切片边界，不属于张量元素类型 |
| F-order 一致性 | 所有偏移量推导都遵循 `offset = sum(index[i] * strides[i])` |
| 零拷贝视图 | 切片优先共享底层数据，仅返回只读或共享只读结果 |
| 安全/不安全分层 | 安全接口显式验证 rank 与边界；unsafe 路径仅跳过检查，不改变语义 |

### 1.3 在架构中的位置

```text
Dependency layers:
L0: error, private
L1: dimension, layout
L2: storage
L3: tensor
L4: index  <- current module
```

索引模块位于 `tensor` 之上，消费张量的 shape、stride、storage mode 与只读/可写访问能力；它不定义新的存储模式，也不反向影响 `dimension`、`layout`、`storage` 的基础语义。

---

## 2. 需求映射与范围约束

| 类型 | 内容 |
| --- | --- |
| 需求映射 | 需求说明书 §18；并受 §4（`usize` 元数据角色）与 §6（只读/共享只读存储约束）补充约束 |
| 范围内 | `usize` 多维索引、范围索引（切片）、rank 一致性检查、越界 recoverable error、unsafe 未检查变体、切片后 shape/stride 更新 |
| 范围外 | 负索引、负步长、布尔/整数数组高级索引、共享可写视图、额外索引语法 |
| 非目标 | 不新增索引能力，不改变现有 panic 语法糖定位，不引入新的存储模式或复制语义 |

> **范围决策：** 当前版本只承诺“`usize` 多维索引 + 范围切片”两类能力；`s![]` 宏仅作为现有范围描述的语法包装，不扩展新语义。

---

## 3. 文件位置

```text
src/
└── index/
    ├── mod.rs           # module root and public re-exports
    ├── ndindex.rs       # NdIndex trait and tuple/slice index implementations
    ├── access.rs        # try_at/get/get_unchecked and mutable variants
    ├── slice.rs         # SliceInfo, slice, shape/stride updates
    └── macros.rs        # s![] macro and descriptor helpers
```

按能力拆分为 `ndindex`、`access`、`slice`、`macros`，可把“索引地址计算”和“切片元数据变换”分开维护；`mod.rs` 负责统一导出，保持对外仍是单一 `src/index/` 模块边界。

---

## 4. 依赖关系

### 4.1 依赖图（ASCII）

```text
                 ┌────────────┐
                 │   error    │
                 └─────┬──────┘
                       │
      ┌────────────────┼────────────────┐
      │                │                │
┌─────▼─────┐    ┌─────▼─────┐    ┌─────▼─────┐
│ dimension │    │  layout   │    │  storage  │
└─────┬─────┘    └─────┬─────┘    └─────┬─────┘
      └────────────┬───┴───────────────┘
                   │
              ┌────▼────┐
              │ tensor  │
              └────┬────┘
                   │
              ┌────▼────┐
              │ index   │
              └─────────┘
```

### 4.2 类型级依赖表

| 来源模块 | 使用的类型/trait |
| --- | --- |
| `tensor` | `TensorBase<S, D>`, `TensorView<'a, A, I>`, `.shape()`, `.strides()`, `.ndim()`, storage mode query |
| `dimension` | `Dimension`, `Ix0`~`Ix6`, `IxDyn`, rank / axis metadata |
| `layout` | `Strides<D>`, layout flags, F-order offset interpretation |
| `storage` | `Storage`, `StorageMut`, read-only / writable storage capability |
| `error` | `XenonError::InvalidAxis`, `InvalidArgument`, `IndexOutOfBounds`, `DimensionMismatch` |
| `private` | `Sealed`，用于封闭 `NdIndex` 的外部实现面 |

### 4.3 依赖方向声明

> **依赖方向：单向向上。** `index` 仅消费 `tensor`、`dimension`、`layout`、`storage`、`error` 的既有能力，不被这些底层模块反向依赖。

### 4.4 依赖合法性与新增依赖说明

| 项目 | 说明 |
| --- | --- |
| 新增第三方依赖 | 无 |
| 合法性结论 | 合法；当前设计仅复用项目内既有模块与标准库 |
| 替代方案 | 不适用；索引能力无需额外 crate，也不应因文档重写扩展依赖面 |

---

## 5. 公共 API 设计

### 5.1 核心接口草案

```rust
use crate::private::Sealed;

pub trait NdIndex<D: Dimension>: Sealed {
    fn index_checked(&self, dim: &D, strides: &Strides<D>) -> Option<usize>;

    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `index` length matches the dimension rank
    /// - Each index component is within bounds for the corresponding axis
    /// - The resulting offset does not overflow `usize`
    unsafe fn index_unchecked(&self, strides: &Strides<D>) -> usize;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SliceInfoElem {
    Index(usize),
    Range {
        start: Option<usize>,
        end: Option<usize>,
        step: Option<usize>,
    },
}

#[derive(Debug, Clone)]
pub enum SliceInfoIndices {
    Inline {
        len: u8,
        elems: [SliceInfoElem; 6],
    },
    Dynamic(Vec<SliceInfoElem>),
}

pub struct SliceInfo<I, D>
where
    I: Dimension,
    D: Dimension,
{
    indices: SliceInfoIndices,
    in_dim: D,
    out_dim: I,
}

impl<I, D> SliceInfo<I, D>
where
    I: Dimension,
    D: Dimension,
{
    pub fn new(indices: SliceInfoIndices, in_dim: D, out_dim: I) -> Result<Self, XenonError>;

    pub fn indices(&self) -> &SliceInfoIndices;

    pub fn input_dim(&self) -> &D;

    pub fn output_dim(&self) -> &I;
}
```

`SliceInfo<I, D>` 是切片描述符的公开包装类型：`D` 表示输入维度，`I` 表示切片后的输出维度；其内部字段保持私有，必须通过带校验的构造器建立，以避免手工拼出“索引长度、输入维度、输出维度彼此矛盾”的无效状态。

### 5.2 张量访问与切片 API

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    pub fn try_at<I>(&self, index: I) -> Result<&A, XenonError>
    where
        I: NdIndex<D>;

    pub fn get(&self, index: &[usize]) -> Result<&A, XenonError>;

    /// # Safety
    ///
    /// Caller must ensure index is valid: len == ndim and each component < shape[i].
    pub unsafe fn get_unchecked(&self, index: &[usize]) -> &A;

    pub fn slice<I>(&self, info: SliceInfo<I, D>) -> Result<TensorView<'_, A, I>, XenonError>
    where
        I: Dimension;

}

impl<S, D, A> TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    D: Dimension,
{
    pub fn try_at_mut<I>(&mut self, index: I) -> Result<&mut A, XenonError>
    where
        I: NdIndex<D>;

    pub fn get_mut(&mut self, index: &[usize]) -> Result<&mut A, XenonError>;

    /// # Safety
    ///
    /// Caller must ensure index is valid: len == ndim and each component < shape[i].
    /// Caller must also have exclusive mutable access to the referenced element.
    pub unsafe fn get_unchecked_mut(&mut self, index: &[usize]) -> &mut A;
}
```

> **设计决策：** 当前版本不把 `Index` / `IndexMut` 运算符重载纳入公共 API 契约，也不为其提供规范承诺。对外规范安全主路径始终是 `try_at()` / `get()` / `try_at_mut()` / `get_mut()` 与 `slice()`；若实现中保留 `Index` / `IndexMut`，其也只属于内部便捷层 / 非规范行为，且必须显式文档化“失败即 panic”。此前草案中的 `try_slice()` 与当前 `slice()` 具有相同签名且同样返回 `Result`，没有额外语义，因此移除以避免冗余接口。

### 5.3 Good / Bad 对比

```rust
// Good - checked indexing keeps recoverable errors on the main path.
let value = tensor.try_at((2, 1))?;
let value2 = tensor.get(&[2, 1])?;

// Bad - panic-based sugar is not part of the normative public contract.
let value = &tensor[(2, 1)];
```

```rust
// Good - slice validation rejects invalid step values explicitly.
let view = tensor.slice(s![1..4;2, ..])?;

// Bad - invalid slice input must not be forced through as if it were valid.
let view = tensor.slice(s![1..4;0, ..]).unwrap();
```

```rust
// Good - unsafe path is only used when the caller already proved validity.
let value = unsafe { tensor.get_unchecked(&[1, 2, 0]) };

// Bad - using unchecked access as a substitute for normal validation.
let value = unsafe { tensor.get_unchecked(user_index.as_slice()) };
```

---

## 6. 内部实现设计

### 6.1 核心数据结构

`NdIndex<D>` 负责把多维索引转换为偏移量；`SliceInfoElem` / `SliceInfoIndices` 负责表达“定点索引”和“范围索引”的组合。两类描述都只接受 `usize`，从类型层面排除负索引与负步长。`SliceInfo<I, D>` 额外负责把这些索引描述与输入/输出维度绑定，但其内部字段不对外公开，只能通过构造器统一校验。

### 6.2 偏移量计算

```rust
fn compute_offset_f<D: Dimension>(index: &[usize], strides: &Strides<D>) -> usize {
    let mut offset = 0usize;
    for (&idx, &stride) in index.iter().zip(strides.iter()) {
        offset += idx * stride;
    }
    offset
}
```

内部不变式：

- 索引元组长度必须与张量 rank 一致。
- 每个索引分量必须落在对应轴的有效范围内。
- 所有偏移量计算必须使用已有 stride，不得假设连续布局。
- 任何安全接口在生成引用前都必须先完成上述验证。

### 6.3 切片元数据更新

```text
compute_slice(shape, strides, offset, slices):
    1. Validate the slice descriptor rank against ndim.
    2. For Index(idx), check bounds and fold into the new offset.
    3. For Range { start, end, step }, reject step == 0.
    4. Compute the resulting axis length and stride multiplication.
    5. Recompute layout flags from the new shape and strides.
    6. Return a read-only TensorView.
```

切片后的语义约束如下：

- 结果须保持原有逻辑元素顺序。
- `Index(usize)` 会折叠对应轴并累加 offset。
- `Range` 会按起止边界与步长更新 shape 和 stride。
- 当前版本仅允许正步长；`step == 0` 返回可恢复错误。
- 切片结果与源张量共享底层数据时，仅可落在只读或共享只读范围内，不提供共享可写视图。
- 布局状态只能重新落在 `FContiguous`、`NonContiguous`、`BroadcastView` 三种之一。
- `SliceInfo::new(...)` 必须校验索引描述长度、`in_dim` 与 `out_dim` 的对应关系，拒绝构造内部自相矛盾的描述符。

### 6.4 安全性论证

| `unsafe` 点 | 安全前提 | 为什么仍然需要 |
| --- | --- | --- |
| `NdIndex::index_unchecked` | 调用方已保证 rank 匹配且每个分量有效 | 为内部已验证路径消除重复检查 |
| `get_unchecked` / `get_unchecked_mut` | 调用方已保证索引合法且可写性前提成立 | 为热点访问路径提供零额外分支的能力 |

**安全性论证：** unsafe 变体只省略检查，不改变偏移量公式、shape/stride 解释或引用别名规则。若输入索引非法，责任由调用方承担；若输入合法，unsafe 与安全路径的结果必须一致。

### 6.5 内部性能考量

| 方面 | 设计决策 |
| --- | --- |
| 偏移计算 | 对 rank 做线性遍历，时间复杂度 O(rank) |
| 切片创建 | 仅更新元数据并返回视图，正常路径 O(rank) 且不分配底层数据 |
| 布局重算 | 依据新 shape/stride 重新标记布局，不复制元素 |

---

## 7. 实现任务拆分

### Wave 1: 索引地址计算基础

- [ ] **T1**: 定义 `NdIndex<D>` 与 tuple / slice index 的合法性检查
    - 文件: `src/index/ndindex.rs`
    - 内容: rank 匹配、逐轴边界检查、checked / unchecked offset 计算
    - 测试: `test_try_at_2d`, `test_try_at_out_of_bounds`
    - 前置: `dimension`、`tensor` 基础能力已可用
    - 预计: 10 min

- [ ] **T2**: 实现 `try_at` / `get` / `get_unchecked`
    - 文件: `src/index/access.rs`
    - 内容: 统一安全与 unsafe 访问路径，保证错误边界一致
    - 测试: `test_get_returns_index_out_of_bounds`
    - 前置: T1
    - 预计: 10 min

### Wave 2: 可写访问与 trait 约束

- [ ] **T3**: 实现 `try_at_mut` / `get_mut` / `get_unchecked_mut`
    - 文件: `src/index/access.rs`
    - 内容: 仅在 `StorageMut` trait 前提成立时暴露可写访问
    - 测试: `test_try_at_mut_requires_storage_mut`
    - 前置: T2
    - 预计: 10 min

### Wave 3: 切片描述与视图构造

- [ ] **T4**: 定义 `SliceInfoElem` 与 `SliceInfoIndices`
    - 文件: `src/index/slice.rs`
    - 内容: inline / dynamic 切片描述表示
    - 测试: `test_slice_basic`
    - 前置: T1
    - 预计: 10 min

- [ ] **T5**: 实现 `slice` 的 shape/stride 更新与布局重算
    - 文件: `src/index/slice.rs`
    - 内容: `Index` 折轴、`Range` 更新 shape/stride、只读视图返回
    - 测试: `test_slice_with_step`, `test_slice_layout_recomputed`
    - 前置: T4
    - 预计: 10 min

- [ ] **T6**: 集成 `s![]` 宏到切片描述符
    - 文件: `src/index/macros.rs`
    - 内容: 将现有语法映射到范围内切片能力，不新增语义
    - 测试: `test_slice_chain`
    - 前置: T5
    - 预计: 10 min

### Wave 依赖图

```text
Wave 1: [T1] -> [T2]
                  │
                  ├──────────────┐
                  ▼              ▼
Wave 2:         [T3]         Wave 3: [T4] -> [T5] -> [T6]
```

---

## 8. 测试计划

### 8.1 测试分类表

| 类型 | 位置 | 目的 |
| --- | --- | --- |
| 单元测试 | `src/index/` 对应测试模块 | 验证单个访问/切片函数的语义与错误边界 |
| 集成测试 | 索引与张量交互测试 | 验证 `tensor` + `index` + `error` 的组合行为 |
| 边界测试 | 与单元/集成测试配套组织 | 覆盖 rank-0、广播视图、非连续切片、越界 |
| 属性测试（按需） | 索引模块测试目录 | 验证 offset 计算与 shape/stride 更新不变量 |
| Feature gate / 配置测试 | 不适用 | 当前模块不涉及 SIMD、并行或可选 feature |
| 类型边界 / 编译期测试 | trait 约束测试或编译期失败测试 | 验证 `usize` 仅用于元数据角色，索引 trait 不放宽到负数或元素类型 |

### 8.2 单元测试清单

| 测试函数 | 测试内容 | 优先级 |
| --- | --- | --- |
| `test_try_at_2d` | `try_at()` 成功返回二维张量元素引用 | 高 |
| `test_try_at_out_of_bounds` | 越界返回 `IndexOutOfBounds` | 高 |
| `test_try_at_mut_requires_storage_mut` | 只在 `StorageMut` 前提成立时才存在可写访问入口 | 高 |
| `test_get_returns_index_out_of_bounds` | `get()` 失败返回 `IndexOutOfBounds` | 高 |
| `test_slice_basic` | 基本切片结果的 shape 与数据正确 | 高 |
| `test_slice_with_step` | 正步长切片结果正确 | 高 |
| `test_slice_step_zero` | `step == 0` 返回 `InvalidArgument` | 高 |
| `test_slice_chain` | 视图的视图保持一致的共享数据语义 | 中 |
| `test_slice_layout_recomputed` | 切片后布局状态被重新计算 | 高 |

### 8.3 边界测试场景表

| 场景 | 预期行为 |
| --- | --- |
| rank-0 张量索引 | 仅接受零维合法索引形式，偏移为 0 |
| 广播视图上的只读索引 | 索引成功但结果仍遵循只读/共享只读语义 |
| 非连续切片后的访问 | 偏移量计算继续基于 stride，不假设连续 |
| 任一轴越界 | 安全接口返回 recoverable error；非规范语法糖若存在可 panic |
| 切片 `step == 0` | 返回 `InvalidArgument`，不得构造结果视图 |

### 8.4 属性测试不变量（按需）

| 不变量 | 测试方法 |
| --- | --- |
| `checked_offset == unchecked_offset`（在合法输入上） | 随机生成合法 shape / strides / index 并比较两条路径 |
| `slice.len()` 与更新后的 shape 一致 | 随机合法范围输入，验证逻辑元素数量守恒 |
| 切片保持逻辑顺序 | 对合法正步长切片比较视图遍历序列与参考实现 |

### 8.5 Feature gate / 配置测试

| 配置 | 验证点 |
| --- | --- |
| 默认配置 | 索引模块语义完整可用 |
| SIMD | 不适用 |
| 并行 | 不适用 |

### 8.6 类型边界与编译期测试

| 场景 | 测试方式 |
| --- | --- |
| `usize` 不作为元素类型扩展索引语义 | 文档审查 + 类型约束测试 |
| 非 `usize` 负数索引不被接受 | 编译期失败测试或 trait 约束验证 |
| 非法 `NdIndex` 外部实现 | `Sealed` 约束测试 |

---

## 9. 模块交互设计

### 9.1 接口约定

| 方向 | 对方模块 | 接口/类型 | 约定 |
| --- | --- | --- | --- |
| 消费（输入） | `tensor` | `TensorBase<S, D>` | 索引前读取 shape、stride、offset 与存储模式 |
| 消费（输入） | `dimension` | `Dimension` | 用于 rank 与轴边界验证 |
| 消费（输入） | `layout` | `Strides<D>`, layout flags | 偏移量解释与切片后布局重算 |
| 产出（输出） | `tensor` | `&A`, `&mut A`, `TensorView<'a, A, I>` | 返回值生命周期绑定到源张量；切片结果共享底层数据 |
| 产出（输出） | `error` | `XenonError` | 安全路径对外暴露统一错误类型 |

### 9.2 数据流描述

```text
User calls tensor.try_at(index)
    │
    ├── index/ validates rank and bounds
    ├── index/ computes offset from shape + strides
    └── tensor/storage returns shared or mutable reference

User calls tensor.slice(info)
    │
    ├── index/ validates each SliceInfoElem
    ├── index/ updates offset, shape, and strides
    ├── index/ recomputes layout flags
    └── tensor returns read-only TensorView sharing source data
```

### 9.3 生命周期与所有权约定

> **约定：** `try_at()`、`get()` 与 `slice()` 返回的借用结果都绑定源张量生命周期；切片不会转移所有权，也不会复制底层数据。当前版本不存在共享可写视图，因此任何共享结果都必须失去可写访问权。

---

## 10. 错误处理与语义边界

| 主题 | 说明 |
| --- | --- |
| Recoverable error | `try_at()` / `get()` / `slice()` 在 rank 不匹配、轴非法、越界、`step == 0` 时返回 `XenonError`；其中索引长度与张量 `ndim` 不匹配时，错误类型固定为 `XenonError::DimensionMismatch { expected, actual }` |
| Trait-bound 边界 | `try_at_mut()` / `get_mut()` / `get_unchecked_mut()` 仅在 `S: StorageMut` 前提成立时存在；不再为“只读存储上的可写索引”设计运行时 `InvalidStorageMode` 分支 |
| Panic | 若实现保留 `Index` / `IndexMut` 作为内部便捷层 / 非规范 API 契约，其失败时可 panic；这不是规范安全主路径，也不属于公开承诺 |
| 路径一致性 | 对同一合法输入，checked 与 unchecked 路径必须给出同一偏移和同一逻辑结果；unsafe 只省略检查 |
| 容差边界 | 不适用；本模块不涉及浮点容差、SIMD 误差或并行归约差异 |

错误实例须与 `26-error.md` 对齐：

```rust
XenonError::InvalidAxis {
    operation: "slice".into(),
    axis,
    ndim: self.ndim(),
    shape: self.shape().to_vec(),
}

XenonError::InvalidArgument {
    operation: "slice".into(),
    argument: "step".into(),
    expected: "step > 0".into(),
    actual: "0".into(),
    axis: Some(axis),
    shape: Some(self.shape().to_vec()),
}

XenonError::IndexOutOfBounds {
    operation: "try_at".into(),
    attempted_index: index_component,
    axis,
    shape: self.shape().to_vec(),
}

```

显式边界：

- 不再使用缺少 `ndim` / `shape` 的旧 `InvalidAxis` 形式。
- 不引入单独公开的 `InvalidSliceStep` 私有错误名。
- `get()` / `get_mut()` 与 `try_at()` / `try_at_mut()` 在各自存在的前提下都返回结构化可恢复错误；越界时使用 `XenonError::IndexOutOfBounds`。
- 可写访问能力由 `StorageMut` trait-bound 决定，而不是在运行时回退为 `InvalidStorageMode`。

---

## 11. 设计决策记录

### ADR-1: 安全主路径使用 recoverable error

| 属性 | 值 |
| --- | --- |
| 决策 | `try_at()` / `get()` / `try_at_mut()` / `get_mut()` / `slice()` 作为规范安全接口，失败返回可恢复错误 |
| 理由 | 符合 `require.md` §18 对安全接口的要求，并与 `26-error.md` 的统一诊断模型对齐 |
| 替代方案 | 全部使用 `Index` / `IndexMut` panic 语法糖 — 放弃，错误恢复与上游组合能力不足 |
| 替代方案 | 统一返回 `Option` — 放弃，无法承载轴、shape、索引等诊断信息 |

### ADR-2: 切片结果保持共享只读语义

| 属性 | 值 |
| --- | --- |
| 决策 | 范围索引返回共享底层数据的只读或共享只读视图，不提供共享可写视图 |
| 理由 | 符合 `require.md` §18 与 §6，对共享数据结果收敛到可验证的只读语义 |
| 替代方案 | 允许共享可写切片 — 放弃，超出当前版本范围且引入别名写入风险 |
| 替代方案 | 切片总是复制生成独立张量 — 放弃，会破坏零拷贝视图语义并扩大成本 |

### ADR-3: 当前版本只支持正步长切片

| 属性 | 值 |
| --- | --- |
| 决策 | `Range.step` 仅支持正整数；`step == 0` 报错，负步长不纳入能力集合 |
| 理由 | 当前需求只要求范围索引与 shape/stride 更新，不要求反向遍历或负步长语法 |
| 替代方案 | 支持负步长并反转逻辑顺序 — 放弃，属于新能力扩展 |
| 替代方案 | 把 `step == 0` 视为 panic — 放弃，不符合安全接口 recoverable error 要求 |

---

## 12. 性能描述

### 12.1 复杂度

| 操作 | 时间复杂度 | 空间复杂度 |
| --- | --- | --- |
| `try_at` / `get` / `try_at_mut` | O(rank) | O(1) |
| `get_unchecked*` | O(rank) | O(1) |
| `slice` | O(rank) | O(1)（仅视图元数据） |

### 12.2 内存与缓存行为

| 场景 | 行为 |
| --- | --- |
| 连续 F-order 张量索引 | 偏移量计算后访问目标元素，缓存局部性由原布局保证 |
| 非连续视图索引 | 仍可正确访问，但缓存友好性取决于 stride 跳跃模式 |
| 范围切片 | 仅重建视图元数据并共享源数据，不复制元素 |

### 12.3 性能边界说明

- 偏移量计算成本与 rank 成正比，而非与元素总数成正比。
- 切片为元数据级操作；性能关键点在后续对视图的消费，而非视图创建本身。
- unsafe 变体的价值仅在于消除重复检查，不意味着不同的逻辑结果。

---

## 13. 平台与工程约束

| 约束 | 说明 |
| --- | --- |
| `std` only | 当前项目基线为 `std` 环境；本文不扩展 `no_std` 路径 |
| 单 crate | 索引设计保持在现有 crate 内，不引入额外 crate 或拆分子包 |
| SemVer | 文档仅把旧结构重写为标准模板，不新增索引能力或改变已承诺语义 |
| 最小依赖 | 不新增第三方依赖；继续复用仓库既有模块 |

---

## 版本历史

| 版本 | 日期 |
| --- | --- |
| 1.0.0 | 2026-04-14 |
| 1.0.1 | 2026-04-14 |
| 1.0.2 | 2026-04-15 |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
