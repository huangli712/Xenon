# 布局模块设计

> 文档编号: 06
> 模块目录: src/layout/
> 任务阶段: Phase 2
> 前置文档: 01-architecture.md, 02-dimension.md
> 需求参考: 需求说明书 §7、§8、§16 - §19, §22、§25 - §28
> 范围声明: 范围内

---

## 1. 模块定位

### 1.1 职责边界

| 职责         | 包含                                             |
| ------------ | ------------------------------------------------ |
| 布局标志位   | `LayoutFlags: u8` 位域定义和操作                 |
| 步长计算     | F-order 步长公式、合法性验证                     |
| 连续性检查   | F-连续检测算法                                   |
| 对齐检查     | 指针对齐状态查询                                 |
| 步长范围约束 | 当前版本仅讨论非负步长与零步长布局               |
| 零步长语义   | 广播维度的零步长标记                             |

| 职责         | 不包含                                                         |
| ------------ | -------------------------------------------------------------- |
| 布局标志位   | 数据分配（由 `storage/` 提供）                                 |
| 步长计算     | 元素访问（由 `tensor/` 提供）                                  |
| 连续性检查   | 运算逻辑（由 `math/` 提供）                                    |
| 对齐检查     | 实际对齐分配（由 `storage/alloc.rs` 提供）                     |
| 步长范围约束 | 超出 F-order、转置与广播所产生合法布局的 stride 组合           |
| 零步长语义   | 广播规则实现（由 `broadcast/` 提供，参见 `15-broadcast.md §3`）|

### 1.2 设计原则

| 原则          | 体现                                                                       |
| ------------- | -------------------------------------------------------------------------- |
| F-order only  | Xenon 仅支持列优先布局，不支持 C-order                                     |
| O(1) 布局查询 | 通过缓存布局标志位，将高频查询优化为常数时间                               |
| 步长显式建模  | `Strides<D>` 使用显式 `usize` 存储，当前版本仅接受非负步长与零步长（广播） |
| 零成本抽象    | 布局信息为纯元数据，不引入运行时开销                                       |

---

## 2. 需求映射与范围约束

| 项目     | 内容                                                                                              |
| -------- | ------------------------------------------------------------------------------------------------- |
| 需求映射 | 需求说明书 §7、§8、§16 - §19、§22、§25 - §28                                                      |
| 范围内   | `LayoutFlags`、`Strides<D>`、F-order 步长/连续性/对齐/零步长语义、转置/广播相关合法布局校验       |
| 范围外   | 存储分配、元素访问、C-order、负步长布局支持、reshape、into_shape、布局顺序转换                    |
| 非目标   | 引入第三方 bitflags 依赖、多布局系统、运行时可插拔布局后端，或在当前版本引入 reshape/顺序转换语义 |

当前版本不支持 reshape 或布局顺序转换。布局接口保持纯函数式，不额外承诺顺序枚举类型。

---

## 3. 文件位置

```
src/layout/
├── mod.rs             # LayoutFlags, Strides<D>, and public API
├── flags.rs           # LayoutFlags: u8 bitfield definitions and operations
├── strides.rs         # stride computation, validation, and zero-stride handling
└── contiguous.rs      # F-contiguity detection algorithm
```

文件划分理由：步长计算、标志位操作、连续性检测各自独立且职责清晰，拆分后便于独立测试和维护。

---

## 4. 依赖关系

### 4.1 依赖图（ASCII）

```
src/layout/
├── crate::error         # XenonError for checked stride computation / validation
└── crate::dimension     # Dimension trait, Ix0~Ix6, IxDyn
```

### 4.2 依赖精确到类型级

| 来源模块    | 使用的类型/trait                                               |
| ----------- | -------------------------------------------------------------- |
| `error`     | `XenonError`（步长计算和布局校验失败）                         |
| `dimension` | `Dimension`, `Ix0`~`Ix6`, `IxDyn`（参见 `02-dimension.md §5`） |
| `core`      | `usize`                                                        |

### 4.3 依赖合法性

| 项目           | 结论                       |
| -------------- | -------------------------- |
| 新增第三方依赖 | 无                         |
| 合法性结论     | 符合需求说明书最小依赖限制 |
| 替代方案       | 不适用                     |

### 4.4 依赖方向声明

依赖方向：单向向上。 `layout/` 仅消费 `dimension` 的 trait 和类型，不被其依赖。`tensor/`、`math/`、`simd/` 等上层模块消费 layout 的类型和函数。对齐检查（`is_aligned`）接受原始指针 `*const u8`，无需依赖 `storage` 模块。

`Dimension` 只拥有 shape/rank 语义；所有 stride 计算与保存（包括其它文档中曾出现的 `strides_for_f_order` 一类能力）统一收敛到 `layout/` 的 `Strides<D>` 与 `compute_f_strides()`，避免跨模块重复承载 stride 语义。

---

## 5. 公共 API 设计

### 5.1 LayoutFlags（u8 bitflags）

使用 `u8` 类型存储布局标志位，占用 1 字节：

```
LayoutFlags (u8):
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ 7 │ 6 │ 5 │ 4 │ 3 │ 2 │ 1 │ 0 │
├───┼───┼───┼───┼───┼───┼───┼───┤
│ - │ - │ - │ - │ZER│ALG│ - │ F │
└───┴───┴───┴───┴───┴───┴───┴───┘

F   = F_CONTIGUOUS    (0b00001)  Fortran contiguous
ALG = ALIGNED         (0b00100)  64-byte aligned
ZER = HAS_ZERO_STRIDE (0b01000)  contains broadcast zero stride
-   = reserved bits
```

> **注意**：Xenon 不需要 `C_CONTIGUOUS` 标志位（不支持 C-order）。1D 和 0D 数组天然 F-连续。

```rust,ignore
/// A set of layout flags.
///
/// Uses a bitfield to store multiple layout properties with O(1) queries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LayoutFlags(u8);

impl LayoutFlags {
    /// Empty flags
    pub const EMPTY: Self = Self(0b0000_0000);

    /// F-order contiguity flag
    pub const F_CONTIGUOUS: Self = Self(0b0000_0001);  // 0x01

    /// SIMD alignment flag (64-byte)
    pub const ALIGNED: Self = Self(0b0000_0100);        // 0x04

    /// Zero stride flag (broadcast dimension)
    pub const HAS_ZERO_STRIDE: Self = Self(0b0000_1000); // 0x08

    // === Query methods ===

    /// Returns true if F-order contiguous.
    #[inline]
    pub const fn is_f_contiguous(self) -> bool {
        (self.0 & Self::F_CONTIGUOUS.0) != 0
    }

    /// Returns true if 64-byte aligned.
    #[inline]
    pub const fn is_aligned(self) -> bool {
        (self.0 & Self::ALIGNED.0) != 0
    }

    /// Returns true if any broadcast zero stride exists.
    #[inline]
    pub const fn has_zero_stride(self) -> bool {
        (self.0 & Self::HAS_ZERO_STRIDE.0) != 0
    }

    // === Setter methods ===

    /// Sets the F-order contiguity flag.
    #[inline]
    pub const fn set_f_contiguous(self, val: bool) -> Self {
        if val { Self(self.0 | Self::F_CONTIGUOUS.0) }
        else { Self(self.0 & !Self::F_CONTIGUOUS.0) }
    }

    /// Sets the alignment flag.
    #[inline]
    pub const fn set_aligned(self, val: bool) -> Self {
        if val { Self(self.0 | Self::ALIGNED.0) }
        else { Self(self.0 & !Self::ALIGNED.0) }
    }

    /// Sets the zero stride flag.
    #[inline]
    pub const fn set_has_zero_stride(self, val: bool) -> Self {
        if val { Self(self.0 | Self::HAS_ZERO_STRIDE.0) }
        else { Self(self.0 & !Self::HAS_ZERO_STRIDE.0) }
    }

}
```

### 5.1b 纯函数式布局接口

当前版本仅支持 F-order，因此布局接口保持纯函数式：由 `shape`、`strides`、对齐信息等输入直接计算标志与分类结果，不额外承诺任何顺序枚举或基于顺序值的公开构造约定。

```rust
/// Computes the canonical flags for a validated F-order layout.
#[inline]
pub(crate) const fn flags_for_f_layout(aligned: bool, has_zero_stride: bool) -> LayoutFlags {
    LayoutFlags::EMPTY
        .set_f_contiguous(!has_zero_stride)
        .set_aligned(aligned)
        .set_has_zero_stride(has_zero_stride)
}
```

### 5.1c LayoutState 布局分类 API

除底层 `LayoutFlags` 外，张量层还需要一个更直接的布局分类结果，便于上层 API、诊断输出和跨模块分支选择表达“当前这块逻辑数据在内存中的连续性状态”。

> **说明：** Xenon 的原生构造路径仍以 F-order 为准；`LayoutState` 是一个**分类结果**，用于描述某个 `shape + strides` 组合所呈现的布局状态，而不是放宽当前版本的 F-order only 设计边界。

> **权威定义声明**：`LayoutState` 由本模块（`layout/`）定义并持有。`07-tensor.md` 中的 `TensorBase::layout_state()` 方法返回本模块定义的 `LayoutState`，不再重复定义该枚举。若后续版本需扩展布局状态分类，须在本模块中修改并通过模块间接口暴露。

> **权威计算入口声明**：`LayoutFlags` 的唯一权威计算入口为 `compute_layout_flags(shape, strides, ptr)`。其他模块查询布局状态时须引用本模块的结果，不得各自复算或绕过本模块自行裁定 `F_CONTIGUOUS` / `ALIGNED` / `HAS_ZERO_STRIDE`。

```rust,ignore
/// Classification of tensor memory layout contiguity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutState {
    /// Fortran-contiguous (column-major): first stride = 1 and follows F-order progression.
    FContiguous,
    /// Arbitrary non-broadcast view that is not F-contiguous.
    NonContiguous,
    /// Broadcast view: at least one axis has zero stride.
    BroadcastView,
}
```

分类语义约定如下：

- `LayoutState::FContiguous`：满足当前文档 §5.4 的 F-order 连续性判定；广播引入的零步长轴不得归入该类，但空数组在退化表示下出现的零步长可继续保持 `FContiguous`。
- `LayoutState::NonContiguous`：表示任意其它非广播 stride 模式，例如转置后布局或切片后非连续布局。
- `LayoutState::BroadcastView`：表示至少包含一个零步长轴的广播视图，用于与一般非连续视图区分。

`LayoutFlags -> LayoutState` 的确定性映射规则如下：

1. 若 `HAS_ZERO_STRIDE == true`，返回 `LayoutState::BroadcastView`
2. 否则若 `F_CONTIGUOUS == true`，返回 `LayoutState::FContiguous`
3. 否则返回 `LayoutState::NonContiguous`

`layout_state()` 与 `is_f_contiguous()` 等张量层公开方法定义见 `07-tensor.md`；本模块只定义布局分类与判定规则，不承载 `TensorBase` 的方法声明。

### 5.2 步长类型：usize

步长使用 `usize` 存储；当前版本仅接受非负步长与零步长：

```rust,ignore
/// Each element in the stride array represents the memory offset change
/// (in elements) when moving one index along that axis.
///
/// - Positive stride: forward traversal
/// - Zero stride: broadcast dimension (repeats the same element)
```

### 5.2a `Strides<D>` 正式定义

`Strides<D>` 是 layout 模块拥有的正式步长类型，用于把“每个轴前进一步需要跨过多少个元素”从 `Dimension` 的形状语义中独立出来。

```rust,ignore
/// Strides describe the element-offset along each dimension.
/// For a tensor with shape [n0, n1, ..., nk], stride[i] gives the number of
/// elements to skip to advance one position along dimension i.
pub struct Strides<D: Dimension> {
    /// Stride values, one per dimension.
    strides: D,
}

impl<D: Dimension> Strides<D> {
    /// Construct strides from a dimension value.
    /// Zero stride is allowed and represents a broadcast dimension.
    /// This constructor only wraps the stride carrier and does not perform full layout validation.
    pub fn new(strides: D) -> Self;

    /// Construct strides from raw slice data.
    /// Returns `XenonError` if the slice length does not match `D`.
    pub fn from_slice(slice: &[usize]) -> Result<Self, XenonError>;

    /// Compute default F-contiguous strides for the given shape.
    pub fn f_contiguous(shape: &D) -> Result<Self, XenonError>;

    /// Returns the stride for dimension `axis`.
    /// Returns `Err(XenonError)` if axis is out of bounds.
    pub fn try_stride(&self, axis: usize) -> Result<usize, XenonError>;

    /// Returns an iterator over stride values.
    pub fn iter(&self) -> impl Iterator<Item = &usize>;

    /// Borrows the stride storage as a slice.
    pub fn as_slice(&self) -> &[usize];
}
```

不变量：

- 所有 stride 值必须为非负；零表示广播维度。
- `Strides<D>` 的维度数必须与对应 `shape: D` 完全一致。
- 对于 F-contiguous 布局：`stride[0] = 1`，且 `stride[i] = stride[i-1] * shape[i-1]`。

拥有型存储的 stride 必须满足 F-order 连续条件（即 `strides[i] = product of shape[0..i]`）；若传入非 F-order stride，构造须返回 `XenonError::InvalidLayout`。

> **需求说明书 §7 收紧解释：** 本设计对需求说明书 §7 中“仅用于对齐或实现目的的填充区域”做收紧解释：当前版本仅允许 tail padding，不允许 inter-axis padding（例如 padded leading dimension）。这一收紧不影响需求兼容性，因为逻辑元素值及其访问结果与不存在填充区域时保持一致。

补充说明：`Strides::new()` 仅构造承载对象，不执行完整合法性验证。布局合法性由构造器/validator 入口统一负责。

与 `Dimension`/`TensorBase` 的职责边界如下：

- `02-dimension.md` 中的 `Dimension` 只描述 shape/rank，不保存 stride 语义。
- `Strides<D>` 负责保存与 shape 同 rank 的步长元数据。
- `07-tensor.md` 中的 `TensorBase` 通过 `strides: Strides<D>` 持有这部分元数据，并交由 layout 层推导连续性与布局标志。

负步长布局不在当前版本范围内（参见 `需求说明书 §7`）。当前文档仅讨论由 F-order、转置、切片派生的正步长非连续视图与广播产生的合法布局。

### 5.3 F-order 步长计算

> **Padding 说明**：`compute_f_strides()` 产出的仍是规范化 packed F-order stride（`stride[i] = product(shape[0..i])`）。`需求说明书 §7` 提到的“padding”在当前版本不进入逻辑布局元数据；同时需满足 `需求说明书 §11` 关于“带填充区域的数组迭代须仅遍历逻辑元素，不得暴露为对齐或实现目的引入的填充区域”的要求。

**算法**：

```
function compute_f_strides(shape: [usize; N]) -> Result<[usize; N], XenonError>:
    strides = array of size N
    cumulative = 1

    for i from 0 to N-1:
        strides[i] = cumulative
        cumulative = checked_mul(cumulative, shape[i])
            或在整数溢出时返回可恢复错误（`InvalidLayout` 或等效错误类别）

    return Ok(strides)
```

**示例**：

```
shape = [3, 4, 5]
i=0: stride[0] = 1,    cumulative = 1 * 3 = 3
i=1: stride[1] = 3,    cumulative = 3 * 4 = 12
i=2: stride[2] = 12,   cumulative = 12 * 5 = 60

Result: Ok(strides = [1, 3, 12])
```

**API**：

```rust,ignore
/// Computes strides for an F-order contiguous layout from the given shape.
///
/// # Arguments
/// * `shape` - Length of each axis
///
/// # Returns
/// `Result<Strides<D>, XenonError>` with the same rank as `shape`, storing
/// explicit stride metadata.
pub fn compute_f_strides<D: Dimension>(shape: &D) -> Result<Strides<D>, XenonError>;
```

### 5.4 连续性检查算法

**F-连续条件**：

```
function is_f_contiguous(shape: [usize; N], strides: [usize; N]) -> bool:
    // Empty arrays or single elements are always contiguous
    if product(shape) <= 1:
        return true

    expected_stride = 1
    for i from 0 to N-1:
        // Skip size=1 axes (their stride may be any value)
        if shape[i] != 1 and strides[i] != expected_stride:
            return false
        expected_stride = expected_stride * shape[i]

    return true
```

**示例**：

```
shape = [2, 3], strides = [1, 2]
i=0: shape[0]=2, stride[0]=1, expected=1 ✓, expected := 1*2=2
i=1: shape[1]=3, stride[1]=2, expected=2 ✓
Result: true (F-contiguous)

shape = [2, 3], strides = [3, 1]
i=0: shape[0]=2, stride[0]=3, expected=1 ✗
Result: false (not F-contiguous)
```

### 5.4a 布局合法性与校验规则

当前版本支持的 stride/layout 组合按以下闭合规则判定：

#### 合法 stride 布局族

- **F-order contiguous**：对所有轴满足 `strides[i] == product(shape[0..i])`；对 `shape[i] == 1` 的轴，可按 §5.4 的连续性规则放宽判定；若零步长来自广播语义，则不得归类为 `F_CONTIGUOUS`，但空数组在退化 metadata 表示下出现的零步长不自动破坏 `F_CONTIGUOUS`。
- **转置视图（non-contiguous）**：`strides` 是对应 F-order contiguous stride 集合的轴置换结果，且所有 stride 都为正。
- **切片派生的正步长非连续视图**：仅指由 Xenon 内部张量切片 API 产生的布局；所有 stride 都为正，但不满足 F-order 连续条件；例如在已验证父布局上继续切片后得到的正步长布局仍属合法。
- **广播视图**：广播轴允许 stride 为 `0`；是否为广播轴由广播语义决定（源维度为 1 且目标维度 > 1），而非由结果张量的 `shape[i] == 1` 判定。其余非广播轴必须保持 F-order 或转置后的正 stride 模式。
- **单元素或 0-D**：可放宽连续性判定，但 stride 仍须落在当前版本支持的布局族内；其中零步长只允许来自广播语义或空数组退化 metadata，不能把“任意零步长”视为一般合法布局。

> **校验口径说明：** 上述“合法 stride 布局族”同时定义当前版本 safe 构造可接受的布局边界。safe 构造只接受能够**仅凭 metadata** 验证正确性的布局：`shape + strides + offset + storage_len` 必须足以证明访问范围不越界，且布局须落在当前版本支持的布局族内。这里的“切片派生”只指 Xenon 内部张量切片 API 产出的布局，不接受外部 raw-parts 输入仅凭“它看起来像切片结果”就走 safe 路径。任何 raw-parts 构造即使 metadata 恰好匹配切片结果，也只能走 unsafe 构造路径并由调用方承担额外正确性责任。该口径与 `需求说明书` §8 保持一致。

#### 非法 stride 组合

- **负步长**：当前版本不支持。
- **可写上下文中的重叠访问**：若多个逻辑索引会写入同一物理位置，则该布局非法；广播视图因此只能作为只读/共享只读视图暴露。

#### 校验规则

- `total_elements = product(shape)`，且计算过程不得溢出 `usize`。
- 访问范围校验必须显式纳入 `offset`。当 `total_elements == 0` 时，要求 `offset <= storage_len`，且不得计算 `shape[i] - 1`；当 `total_elements > 0` 时，
  `max_accessed_offset = offset + sum(stride[i] * (shape[i] - 1) for all i where shape[i] > 0)`（逐轴累加且使用 checked arithmetic）必须满足 `max_accessed_offset < storage_len`。
- 每个 `stride[i]` 都必须可表示为 `isize`，不得发生表示溢出。

> **当前版本的具体校验口径：**
>
> 合法 stride 族：
>
> 1. F-order 连续：`strides[i] = product(shape[0..i])`
> 2. 转置衍生：对 F-order 连续布局的轴置换结果，`stride[i]` 仍为正且与原始轴的 stride 对应
> 3. 广播衍生：部分轴 `stride = 0`
> 4. 切片衍生：正步长子范围，`stride` 不变，`offset` 调整
>
> 验证规则：
>
> - 所有 `stride[i] >= 0`（非负）
> - 所有 `stride[i] <= isize::MAX`
> - 当 `total_elements == 0` 时，仅要求 `offset <= storage_len`
> - 当 `total_elements > 0` 时，`max_accessed_offset = offset + sum((shape[i] - 1) * stride[i]) < storage_len`
> - 广播视图：广播轴的 `stride[i]` 可为 `0`；是否为广播轴由广播语义决定（源维度为 1 且目标维度 > 1），而非由结果张量的 `shape[i] == 1` 判定
> - 单元素轴 `shape[i] == 1` 时 `stride[i]` 不受连续性约束

#### safe vs unsafe 构造的责任分工

- **Safe constructors**：只接受可由 metadata 单独证明正确的布局；必须检查以上全部规则，且 `shape + strides + offset + storage_len` 足以证明所有逻辑访问都在 backing storage 范围内；任一条件不满足时返回 `Result::Err`。
- **Unsafe constructors**：至少检查可验证的 metadata 约束（如 shape/stride 一致性、元素总数与访问范围公式）；指针有效性、生命周期与实际可访问内存范围由调用方保证。即使外部传入的 raw-parts metadata 与某个“切片派生”布局一致，也不得因此提升为 safe。

### 5.5 对齐检查

```rust,ignore
/// Checks whether the logical-first pointer satisfies the alignment requirement.
///
/// # Preconditions
///
/// `align` must be greater than 0 and a power of 2.
#[inline]
pub fn is_aligned_to(ptr: *const u8, align: usize) -> bool {
    (ptr as usize) % align == 0
}

/// Checks whether the logical first-element pointer is 64-byte aligned.
#[inline]
pub fn is_aligned(ptr: *const u8) -> bool {
    is_aligned_to(ptr, 64)
}
```

> **空张量对齐规则：** 空张量（元素数为 0）的 `ALIGNED` flag 统一设为 `true`，不依赖逻辑首指针是否可观测地满足 64 字节对齐。`compute_layout_flags()` 在空张量分支上应直接写入该约定，以保持 `需求说明书` 的空布局查询稳定语义。

### 5.6 对齐与数据一致性

> **数据一致性保证：** 对齐布局（64 字节对齐）与非对齐布局必须产生相同的元素值。对齐仅影响 SIMD 访问性能，不改变数据语义。当前设计中 `Owned::from_vec(data)` 统一委托到对齐分配路径，因此与 `Owned::from_vec_aligned(data)` 在逻辑语义上完全等价；`is_aligned()` 标志仅用于指导 SIMD 路径选择，而不是区分两套用户可见的构造语义。

> **Strides 归属约定：** `Strides<D>` 由 layout 模块定义并拥有；`dimension` 只提供 `checked_size()` 和无符号 F-order 形状推导，绝不保存 stride 或 logical-first pointer 语义。`tensor` 持有 `Strides<D>` 实例并把它交给 layout 计算标志位。

### 5.7 零步长语义

零步长需要区分两类来源：

- **广播零步长**：由广播语义引入；所有索引访问同一物理元素，布局分类为 `BroadcastView`。
- **空数组退化零步长**：仅因 `shape` 含零轴而出现；它不表示广播，也不必取消 `FContiguous`。

广播零步长示例：

```
shape = [3, 4], strides = [1, 0]  // axis 1 is broadcast

Indices [0, 0], [0, 1], [0, 2], and [0, 3] access the same physical element
```

### 5.8 当前任务边界

> **任务收缩：** 当前版本的 layout 设计不再单独引入 `Layout` 结构体。`TensorBase` 直接缓存 `LayoutFlags`，layout 模块对外只提供 `LayoutFlags`、`LayoutState`、`Strides<D>` 与相关计算/校验函数。若后续版本确需额外布局描述对象，须以新需求为前提单独设计。

### 5.9 compute_layout_flags 内部函数

```rust,ignore
/// Computes layout flags from shape, strides, and data pointer.
///
/// This is the central function that determines all layout properties
/// at tensor construction time. The result is cached in TensorBase.flags
/// for O(1) queries thereafter.
///
/// # Arguments
///
/// * `shape` - Dimension lengths
/// * `strides` - Strides in element units (`Strides<D>` with explicit `usize` storage)
/// * `ptr` - Raw pointer to the logical first element (`TensorBase::as_ptr()`)
///
/// # Returns
///
/// A `LayoutFlags` instance with all relevant flags set.
pub(crate) fn compute_layout_flags<A, D: Dimension>(
    shape: &D,
    strides: &Strides<D>,
    ptr: *const A,
) -> LayoutFlags
```

> **命名约定：** 当前文档统一使用 `compute_layout_flags` 表示“从 `shape + strides + ptr` 计算 `LayoutFlags`”的主函数；若实现中存在更细粒度辅助函数，应在文档中明确其仅为内部步骤。

### 5.10 Good/Bad 对比

```rust,ignore
// Good - checked F-order stride computation
let strides = compute_f_strides(&shape)?;  // [1, 3, 12] for [3,4,5]

// Bad - hardcoded strides (not general-purpose, error-prone)
let strides = [1, 3, 12];  // only valid for [3,4,5]
```

```rust,ignore
// Good - Query using LayoutFlags
if tensor.is_f_contiguous() && tensor.is_aligned() {
    // Use SIMD accelerated path
}

// Bad - Recomputing contiguity every time
let contig = check_contiguity(tensor.shape(), tensor.strides());  // O(n) repeated computation
```

---

## 6. 内部实现设计

### 6.1 标志位计算算法

```
function compute_flags(shape, strides, ptr):
    flags = LayoutFlags::EMPTY

    // 1. Stride properties
    has_zero_stride = any(stride == 0 for stride in strides) and product(shape) > 0
    flags = flags.set_has_zero_stride(has_zero_stride)

    // At this point, HAS_ZERO_STRIDE already excludes empty-array degenerate metadata.
    is_broadcast_zero_stride = has_zero_stride

    // 2. Contiguity
    flags = flags.set_f_contiguous(!is_broadcast_zero_stride && is_f_contiguous(shape, strides))

    // 3. Alignment (based on the logical-first-element pointer, not the backing allocation base)
    flags = flags.set_aligned(is_aligned(ptr))

    return flags
```

### 6.2 特殊情况处理

| 情况            | F-contiguous | 说明                                                               |
| --------------- | :----------: | ------------------------------------------------------------------ |
| 空数组 (size=0) |     true     | 无元素，视为连续；退化零步长 metadata 不强制降级为 `BroadcastView` |
| 标量 (ndim=0)   |     true     | 单元素                                                             |
| 1D 数组         |     true     | 一维情况天然 F-连续                                                |
| 含 size=1 轴    |  检查其他轴  | size=1 轴的步长不影响连续性                                        |
| 广播零步长轴    |    false     | 广播视图不得带 `F_CONTIGUOUS`                                      |

### 6.2a 内部 stride 校验规则

当前版本内部实现与 safe 构造的接受边界保持一致：safe 构造只接受 packed F-order、其轴置换、广播零步长以及正步长切片派生这四类 stride 布局族；校验不仅要求 stride 非负、可表示且访问范围不越界，还要求该布局能够被判定落在这四类受支持族内，并且这些结论能由 metadata 单独机械验证。超出这些布局族的更宽正 stride 组合，即使在某些情况下满足基本内存安全充分条件，也不属于当前版本 safe API 的接受范围，必须走 unsafe 构造路径。

- F-order 连续布局：`stride[i] = product(shape[0..i])`
- 转置派生布局：stride 必须是某个 F-order 连续 stride 集合的轴置换，且全部为正
- 广播派生布局：仅广播轴允许 `stride = 0`；是否为广播轴由广播语义决定（源维度为 1 且目标维度 > 1），而非由结果张量的 `shape[i] == 1` 判定
- 切片派生布局：仅允许 Xenon 内部张量切片 API 产出的正步长子范围；其判定必须满足以下可检查条件：父布局已验证合法、切片不改写非广播轴 stride、`offset` 仅按切片起点单调增加、结果 `shape` 与新 `offset` 仍满足访问范围不越界；外部 raw-parts 输入即使满足同样 metadata 形状，也只能走 unsafe 路径

### 6.3 标志位更新规则

> 所有 flags 更新规则统一通过 `compute_layout_flags()` 入口执行（参见 §5.9）。

| 操作     | 标志位更新方式                                                         |
| -------- | ---------------------------------------------------------------------- |
| 创建     | 调用 `compute_layout_flags()` 统一重新计算                             |
| 切片     | 调用 `compute_layout_flags()` 统一重新计算（对齐基于切片后逻辑首元素） |
| 转置     | 调用 `compute_layout_flags()` 统一重新计算                             |
| 视图创建 | 调用 `compute_layout_flags()` 统一重新计算                             |
| 广播     | 调用 `compute_layout_flags()` 统一重新计算                             |
| reshape  | **不在当前版本范围内**（`需求说明书 §17` 当前仅允许 transpose）        |

### 6.4 安全性论证

Layout 模块不涉及 `unsafe` 操作。标志位计算基于 shape/strides 的只读查询，结果缓存在 `LayoutFlags` 中。

### 6.5 与 Dimension 模块的接口

步长不再存储在 `D` 中。`layout` 模块通过 `Strides<D>` 保持与 shape 同维度数量的显式 `usize` 元数据，直接表达当前版本允许的非负步长与零步长广播。

安全性来自单独的 stride 表示：shape 只描述轴长度，stride 只描述步幅与零步长广播特征，两者的职责边界清晰且可独立验证。

---

## 7. 实现任务拆分

### Wave 1: 基础设施

- [ ] **T1**: 创建 `src/layout/mod.rs` 骨架
  - 文件: `src/layout/mod.rs`, `flags.rs`, `strides.rs`, `contiguous.rs`
  - 内容: 模块声明、子模块文件占位、公共导出声明
  - 测试: 编译通过
  - 前置: dimension 模块完成
  - 预计: 5 min

- [ ] **T2**: 实现 `LayoutFlags` 位定义和基础方法
  - 文件: `src/layout/flags.rs`
  - 内容: `LayoutFlags(u8)` 定义，常量（F_CONTIGUOUS/ALIGNED/HAS_ZERO_STRIDE），查询/设置方法
  - 测试: `test_flags_set_clear`, `test_flags_default_empty`
  - 前置: T1
  - 预计: 10 min

### Wave 2: 核心算法

- [ ] **T3**: 实现带错误返回的 F-order 步长计算算法
  - 文件: `src/layout/strides.rs`
  - 内容: `compute_f_strides<D: Dimension>(shape: &D) -> Result<Strides<D>, XenonError>`
  - 测试: `test_f_strides_2d`, `test_f_strides_3d`, `test_f_strides_scalar`, `test_f_strides_overflow`
  - 前置: T1
  - 预计: 10 min

- [ ] **T4**: 实现 F-连续性检测算法
  - 文件: `src/layout/contiguous.rs`
  - 内容: `is_f_contiguous<D: Dimension>(shape: &D, strides: &Strides<D>) -> bool`
  - 测试: `test_f_contig_true`, `test_f_contig_false`, `test_f_contig_empty`, `test_f_contig_scalar`
  - 前置: T1
  - 预计: 10 min

- [ ] **T5**: 实现步长特性检测
  - 文件: `src/layout/strides.rs`
  - 内容: `has_zero_stride<D: Dimension>(strides: &Strides<D>) -> bool`
  - 测试: `test_zero_stride_detect`
  - 前置: T1
  - 预计: 10 min

### Wave 3: 集成

- [ ] **T7**: 实现对齐检查
  - 文件: `src/layout/strides.rs`
  - 内容: `is_aligned_to`/`is_aligned` 函数
  - 测试: `test_alignment_check`
  - 前置: T1
  - 预计: 10 min

### Wave 4: 测试和文档

- [ ] **T8**: 集成测试和文档完善
  - 文件: `tests/test_layout.rs`
  - 内容: 综合测试套件：步长计算、连续性检查、零步长、对齐检查
  - 测试: 完整集成测试
  - 前置: T3, T4, T5, T7
  - 预计: 10 min

### 并行执行图

```
Wave 1: [T1] → [T2]
              ↓
Wave 2: [T3] [T4] [T5] [T7]   (parallelizable)
              ↓
Wave 3:       [T8]
```

---

## 8. 测试计划

### 8.1 测试分类表

| 类型     | 位置                            | 目的                             |
| -------- | ------------------------------- | -------------------------------- |
| 单元测试 | `#[cfg(test)] mod tests`        | 验证单个算法                     |
| 集成测试 | `tests/`                        | 验证跨维度类型交互               |
| 边界测试 | `tests/test_layout_boundary.rs` | 空数组、标量、高维等边界情况     |
| 属性测试 | `tests/property/`               | 验证步长、连续性与对齐相关不变量 |

### 8.2 单元测试清单

| 测试函数                   | 测试内容                          | 优先级 |
| -------------------------- | --------------------------------- | ------ |
| `test_f_strides_1d`        | 1D: `[5]` → strides `[1]`         | 高     |
| `test_f_strides_2d`        | 2D: `[3,4]` → strides `[1,3]`     | 高     |
| `test_f_strides_3d`        | 3D: `[2,3,4]` → strides `[1,2,6]` | 高     |
| `test_f_strides_scalar`    | 0D: `()` → strides `()`           | 高     |
| `test_f_strides_overflow`  | 乘积溢出时返回 `XenonError`       | 高     |
| `test_f_contig_true`       | F-连续数组判定                    | 高     |
| `test_f_contig_false`      | 非连续数组判定                    | 高     |
| `test_f_contig_empty`      | 空数组判定为连续                  | 中     |
| `test_f_contig_size1_axis` | 含 size=1 轴的连续性判定          | 中     |
| `test_zero_stride_detect`  | 区分广播零步长与空数组退化零步长  | 高     |
| `test_alignment_aligned`   | 对齐指针对齐检查                  | 高     |
| `test_alignment_unaligned` | 非对齐指针对齐检查                | 高     |
| `test_flags_all_set`       | 所有标志位同时设置                | 中     |
| `test_flags_default_empty` | 默认值为空                        | 高     |

### 8.3 边界测试场景

| 场景                       | 预期行为                                        |
| -------------------------- | ----------------------------------------------- |
| 空数组 `shape=[0, 3]`      | F-连续为 true；退化零步长仍可保持 `FContiguous` |
| 标量 `shape=()`            | F-连续为 true，步长为空                         |
| 1D 数组 `shape=[5]`        | F-连续为 true                                   |
| 高维 `shape=[2,2,2,2,2,2]` | F-连续，步长 `[1,2,4,8,16,32]`                  |

### 8.3a `需求说明书 §28.4` 边界测试占位

| 占位项         | 说明                                                                                  |
| -------------- | ------------------------------------------------------------------------------------- |
| 空张量布局边界 | 占位：覆盖 `shape.checked_size() == 0` 时 `F_CONTIGUOUS == true` 且 `ALIGNED == true` |
| 大张量布局边界 | 占位：覆盖大 shape 的步长乘法/访问范围 checked arithmetic                             |
| 高维布局边界   | 占位：覆盖高维 `Strides<D>`、转置后正步长和广播零步长组合                             |

### 8.4 属性测试不变量

| 不变量                                                 | 测试方法                                                               |
| ------------------------------------------------------ | ---------------------------------------------------------------------- |
| F-步长乘积 == 总元素数                                 | `product(shape[i]) == total`                                           |
| 空数组/标量始终 F-连续                                 | 随机 0D/空 shape                                                       |
| `compute_f_strides` 成功后 `is_f_contiguous` 返回 true | 随机 shape                                                             |
| `Owned::from_vec(v)` 走规范对齐分配路径                | 验证返回存储/布局的对齐标志与指针对齐状态一致，且满足 64-byte 对齐契约 |

### 8.5 集成测试

| 测试文件               | 测试内容                                                                                                    |
| ---------------------- | ----------------------------------------------------------------------------------------------------------- |
| `tests/test_layout.rs` | `compute_f_strides` / `compute_flags` / `is_aligned` 与 `tensor`、`storage`、`simd`、`ffi` 的端到端协同路径 |

### 8.6 Feature gate / 配置测试

| 配置项         | 覆盖方式              | 说明                                        |
| -------------- | --------------------- | ------------------------------------------- |
| 默认配置       | 常规单元/集成测试路径 | 本模块无独立 feature gate，默认配置即主路径 |
| 非默认 feature | 不适用                | 本模块未定义 feature gate，故无额外配置矩阵 |

### 8.7 类型边界 / 编译期测试

| 测试类型     | 覆盖方式                                            | 说明                                          |
| ------------ | --------------------------------------------------- | --------------------------------------------- |
| 布局顺序边界 | 编译期与文档测试验证仅保留纯函数式 F-order 布局接口 | 验证当前版本不会误暴露顺序枚举或 C-order 契约 |
| stride 边界  | raw-parts / layout 相关编译期与运行时测试结合覆盖   | 验证 `Strides<D>` 与 `D` 的维度数保持一致     |
| 标志位边界   | 编译期验证不依赖外部 bitflags crate                 | 验证布局标志保持最小依赖实现                  |

---

## 9. 与其他模块的交互

### 9.1 接口约定

| 方向                      | 对方模块             | 接口/类型              | 约定                                                                                                       |
| ------------------------- | -------------------- | ---------------------- | ---------------------------------------------------------------------------------------------------------- |
| `tensor/storage → layout` | `tensor` / `storage` | `layout::is_aligned()` | `TensorBase` 构造时将逻辑首元素指针传入 layout 计算对齐标志；layout 只操作原始指针，不依赖 Storage trait。 |

### 9.2 数据流描述

```text
Upper layers create or transform tensor metadata
    │
    ├── layout computes flags from shape + strides + logical-first pointer
    ├── tensor caches the F-contiguous / aligned / zero-stride results
    ├── simd / ffi / shape / index consume these flags for path selection
    └── repeated hot-path contiguity and alignment recomputation is avoided
```

### 9.3 与 Tensor 模块

| 方向              | 对方模块 | 接口/类型                     | 约定                                                                                                                               |
| ----------------- | -------- | ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `layout ← tensor` | `tensor` | `LayoutFlags`                 | `TensorBase` 直接内联 `LayoutFlags` 作为计算字段，并结合 `LayoutState` / `Strides<D>` 表达布局元数据（参见 `07-tensor.md` §5.1）。 |
| `tensor → layout` | `tensor` | 切片后的 flags 更新           | 切片时统一调用 `compute_layout_flags()` 更新连续性与对齐标志（参见 §5.9、`17-indexing.md` §5）                                     |
| `tensor → layout` | `tensor` | transpose 后的步长/flags 重算 | transpose 后统一调用 `compute_layout_flags()` 重算 layout state 与 flags（参见 §5.9、`16-shape.md` §5.1）                          |

### 9.4 与 SIMD 模块

| 方向            | 对方模块 | 接口/类型                            | 约定                                                    |
| --------------- | -------- | ------------------------------------ | ------------------------------------------------------- |
| `simd ← layout` | `simd`   | `is_aligned()` / `is_f_contiguous()` | simd 用这些查询结果做路径选择（参见 `08-simd.md` §5.6） |
| `simd ← layout` | `simd`   | 步长检查                             | simd 继续检查步长是否为 1，以确认连续访问路径           |

### 9.5 与 FFI 模块

| 方向           | 对方模块 | 接口/类型            | 约定                                                                  |
| -------------- | -------- | -------------------- | --------------------------------------------------------------------- |
| `ffi ← layout` | `ffi`    | BLAS 兼容检查        | FFI 路径依赖连续、正步长、无零步长等布局前提（参见 `23-ffi.md` §5.5） |
| `ffi ← layout` | `ffi`    | `lda()` 相关步长信息 | FFI 从 layout 步长推导 leading dimension                              |

---

## 10. 错误处理与语义边界

| 项目              | 内容                                                                                                                                                                                                  |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Recoverable error | 对外布局校验失败返回 `XenonError`（由上层构造路径传播），上下文字段应包含 shape、strides、offset 或操作名等布局元数据                                                                                 |
| Panic             | 对外公开语义中，safe 布局构造/校验失败必须返回 `XenonError`，不以 panic 作为常规错误通道；仅内部 bug（例如“已验证快捷路径”仍触发不可能的整数溢出）或文档明确标注的前置条件违背，才可视为 panic 级缺陷 |
| 路径一致性        | scalar 布局、SIMD 路径选择与 parallel 上游消费必须共享同一 `LayoutFlags` 语义，不允许因路径差异改变结果                                                                                               |
| 容差边界          | 不适用                                                                                                                                                                                                |

---

## 11. 设计决策记录

### 决策 1：F-order only

| 属性     | 值                                                                         |
| -------- | -------------------------------------------------------------------------- |
| 决策     | Xenon 仅支持 F-order（列优先），不支持 C-order                             |
| 理由     | BLAS/LAPACK 兼容；科学计算惯例；简化设计减少分支；`需求说明书 §7` 明确要求 |
| 替代方案 | 同时支持 F/C order — 放弃，增加复杂度且当前版本不需要                      |
| 替代方案 | 仅支持 C-order — 放弃，与 BLAS 不兼容                                      |

### 决策 2：步长使用 usize

| 属性     | 值                                                                                  |
| -------- | ----------------------------------------------------------------------------------- |
| 决策     | 步长类型为 `usize`（无符号）                                                        |
| 理由     | `需求说明书 §7` 明确当前版本不支持负步长；`Strides<D>` 只需表达非负步长与零步长广播 |
| 替代方案 | `isize` — 放弃，会暗示当前版本支持负步长                                            |
| 替代方案 | `i64` — 放弃，与平台 `usize` 不一致                                                 |

### 决策 3：64 字节对齐选择

| 属性     | 值                                                                         |
| -------- | -------------------------------------------------------------------------- |
| 决策     | `ALIGNED` 标志基于 64 字节对齐                                             |
| 理由     | AVX-512 = 512-bit = 64 字节；现代 CPU 缓存行 64 字节；满足所有 SIMD 指令集 |
| 替代方案 | 16 字节 — 放弃，AVX-512 未对齐                                             |
| 替代方案 | 动态对齐 — 放弃，增加运行时开销                                            |

### 决策 4：使用裸 u8 而非 bitflags crate

| 属性     | 值                                                            |
| -------- | ------------------------------------------------------------- |
| 决策     | 使用裸 `u8` 包装类型而非 `bitflags` crate                     |
| 理由     | 零依赖（项目最小依赖原则）；仅 3 个标志位，手写位操作足够清晰 |
| 替代方案 | bitflags crate — 放弃，引入不必要依赖                         |

### 决策 5：Owned Layout Padding Policy

| 属性     | 值                                                                                                                                                                                                                                                                                                              |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 决策     | 当前版本中，`需求说明书` §7 提到的“padding”仅指分配层面的尾部容量/对齐冗余（例如分配器为满足对齐返回多于请求值的字节数）；拥有型张量的逻辑布局元数据（`shape`、`strides`）仍保持规范化 packed F-order，即 `stride[i] = product(shape[0..i])`。当前版本**不支持**轴间 padding（例如 padded leading dimension）。 |
| 决策补充 | `需求说明书` §7 所指的“填充区域”在本设计中解释为仅用于分配层对齐目的的尾部冗余空间（tail padding），不包含轴间填充（inter-axis padding）或 padded leading dimension。该收紧解释不改变逻辑元素值及其访问结果，因此保持与需求兼容。                                                                               |
| 理由     | 规范化 packed F-order 可简化布局校验与 FFI 导出；轴间 padding 将引入 `leading_dim` 一类概念，显著增加类型系统复杂度；分配级对齐由 `storage` 层的 `AlignedBuf` 处理，而非 `layout` 层。                                                                                                                          |
| 替代方案 | 在 `layout` 元数据中显式支持轴间 padding / padded leading dimension — 放弃，会扩大布局状态空间并增加验证、类型表达与 FFI 映射复杂度。                                                                                                                                                                           |

---

## 12. 性能考量

| 方面       | 设计决策                                                   |
| ---------- | ---------------------------------------------------------- |
| 布局查询   | O(1)，直接读取缓存的 `LayoutFlags`                         |
| 步长计算   | O(ndim)，仅在创建、转置视图重算或广播视图校验时计算        |
| 连续性检查 | O(ndim)，仅在切片/转置后重算                               |
| 缓存友好性 | F-order 列优先访问与内存布局一致，顺序遍历时缓存命中率最优 |

**性能数据（参考）**：

| 操作                        | 复杂度  | 说明              |
| --------------------------- | ------- | ----------------- |
| `is_f_contiguous()`         | O(1)    | 读取缓存标志      |
| `is_aligned()`              | O(1)    | 读取缓存标志      |
| `compute_f_strides()`       | O(ndim) | ndim ≤ 6 时可忽略 |
| `is_f_contiguous()`（计算） | O(ndim) | ndim ≤ 6 时可忽略 |

**缓存友好性分析**：

| 遍历模式           | 缓存命中率 | 说明                     |
| ------------------ | ---------- | ------------------------ |
| F-order 顺序遍历   | ~95%       | 顺序访问内存，预取器友好 |
| 非连续遍历（切片） | ~60%       | 跳跃访问，缓存行利用率低 |
| 广播遍历（零步长） | ~95%       | 重复访问同一缓存行       |

---

## 13. 平台与工程约束

| 约束       | 说明                                                                                                             |
| ---------- | ---------------------------------------------------------------------------------------------------------------- |
| `std` only | 本模块依赖 `std` 环境，不讨论 `no_std`                                                                           |
| 单 crate   | 保持单 crate 边界                                                                                                |
| SemVer     | 布局类型和 stride 计算变更遵循 SemVer                                                                            |
| 最小依赖   | 无新增第三方依赖                                                                                                 |
| 线程安全   | `Strides<D>` 本身是 immutable 的值类型；构造后不可修改，因此无需同步原语即可跨线程共享（`Send + Sync` 自动推导） |
| MSRV       | Rust 1.85+                                                                                                       |

---

## 版本历史

| 版本  | 日期       |
| ----- | ---------- |
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-07 |
| 1.0.2 | 2026-04-08 |
| 1.0.3 | 2026-04-08 |
| 1.0.4 | 2026-04-08 |
| 1.0.5 | 2026-04-10 |
| 1.1.0 | 2026-04-08 |
| 1.2.0 | 2026-04-08 |
| 1.2.1 | 2026-04-09 |
| 1.2.2 | 2026-04-14 |
| 1.2.3 | 2026-04-15 |
| 1.2.4 | 2026-04-15 |
| 1.2.5 | 2026-04-16 |
| 1.2.6 | 2026-04-16 |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
