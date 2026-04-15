# 布局模块设计

> 文档编号: 06 | 模块: `src/layout/` | 阶段: Phase 2
> 前置文档: `02-dimension.md`
> 需求参考: 需求说明书 §7, §8, §16, §17, §19, §22, §25, §26, §27, §28
> 范围声明: 范围内

---

## 1. 模块定位

### 1.1 职责边界

| 职责         | 包含                               | 不包含                                                          |
| ------------ | ---------------------------------- | --------------------------------------------------------------- |
| 布局标志位   | `LayoutFlags: u8` 位域定义和操作   | 数据分配（由 `storage/` 提供）                                  |
| 步长计算     | F-order 步长公式、合法性验证       | 元素访问（由 `tensor/` 提供）                                   |
| 连续性检查   | F-连续检测算法                     | 运算逻辑（由 `math/` 提供）                                     |
| 对齐检查     | 指针对齐状态查询                   | 实际对齐分配（由 `storage/alloc.rs` 提供）                      |
| 步长范围约束 | 当前版本仅讨论非负步长与零步长布局 | 超出 F-order、转置与广播所产生合法布局的 stride 组合            |
| 零步长语义   | 广播维度的零步长标记               | 广播规则实现（由 `broadcast/` 提供，参见 `15-broadcast.md §3`） |

### 1.2 设计原则

| 原则          | 体现                                                                       |
| ------------- | -------------------------------------------------------------------------- |
| F-order only  | Xenon 仅支持列优先布局，不支持 C-order                                     |
| O(1) 布局查询 | 通过缓存布局标志位，将高频查询优化为常数时间                               |
| 步长显式建模  | `Strides<D>` 使用显式 `usize` 存储，当前版本仅接受非负步长与零步长（广播） |
| 零成本抽象    | 布局信息为纯元数据，不引入运行时开销                                       |

### 1.3 在架构中的位置

```
Dependency layers:
L0: error, private
L1: dimension, element, complex
L2: layout (depends on dimension)  ← current module
L3: storage (independent from layout, only provides the backing buffer and ownership)
L4: tensor (depends on storage, dimension)
L5: math/, iter/, index/, shape/, broadcast/, construct/, ffi/, convert/, format/
```

### 1.4 F-order 设计决策

> **重要**：Xenon 仅支持 F-order（列优先），不支持 C-order。
>
> **理由**：
>
> 1. **BLAS 兼容**：BLAS/LAPACK 使用列优先布局
> 2. **科学计算惯例**：Fortran 在科学计算领域历史悠久
> 3. **简化设计**：去除 C-order 减少分支和复杂度
> 4. **需求明确**：需求说明书 §7 明确"只支持列优先（F-order）布局"

---

## 2. 需求映射与范围约束

| 项目     | 内容                                                                                                 |
| -------- | ---------------------------------------------------------------------------------------------------- |
| 需求映射 | 需求说明书 §7、§8、§16、§17、§19、§22、§25、§26、§27、§28                                            |
| 范围内   | `LayoutFlags`、`Strides<D>`、F-order 步长/连续性/对齐/零步长语义、转置/广播相关合法布局校验         |
| 范围外   | 存储分配、元素访问、C-order、负步长布局支持、reshape、into_shape、布局顺序转换                      |
| 非目标   | 引入第三方 bitflags 依赖、多布局系统、运行时可插拔布局后端，或在当前版本引入 reshape/顺序转换语义   |

> **范围声明**：当前版本不支持 reshape 或布局顺序转换。`Order` 枚举仅用于标识 F-order 连续布局状态。

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

### 4.2 类型级依赖

| 来源模块    | 使用的类型/trait                                               |
| ----------- | -------------------------------------------------------------- |
| `error`     | `XenonError`（步长计算和布局校验失败）                         |
| `dimension` | `Dimension`, `Ix0`~`Ix6`, `IxDyn`（参见 `02-dimension.md §5`） |
| `core`      | `usize`                                                        |

### 4.2a 依赖合法性

| 项目           | 结论                           |
| -------------- | ------------------------------ |
| 新增第三方依赖 | 无                             |
| 合法性结论     | 符合需求说明书最小依赖限制     |
| 替代方案       | 不适用                         |

### 4.3 依赖方向声明

> **依赖方向：单向向上。** `layout/` 仅消费 `dimension` 的 trait 和类型，不被其依赖。`tensor/`、`math/`、`simd/` 等上层模块消费 layout 的类型和函数。对齐检查（`is_aligned`）接受原始指针 `*const u8`，无需依赖 `storage` 模块。

> **职责统一声明：** `Dimension` 只拥有 shape/rank 语义；所有 stride 计算与保存（包括其它文档中曾出现的 `strides_for_f_order` 一类能力）统一收敛到 `layout/` 的 `Strides<D>` 与 `compute_f_strides()`，避免跨模块重复承载 stride 语义。

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
ZER = HAS_ZERO_STRIDE (0b01000)  contains zero stride (broadcast)
-   = reserved bits
```

> **注意**：Xenon 不需要 `C_CONTIGUOUS` 标志位（不支持 C-order）。1D 和 0D 数组天然 F-连续。

```rust
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

    /// Returns true if any zero stride exists.
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

### 5.1b 内存顺序枚举 Order

`Order` 枚举仅用于标识 “该布局是 F-order 连续布局”。当前版本不支持 reshape、`into_shape` 或布局顺序转换；转置只更新 shape/stride 元数据，不引入新的目标顺序语义。

```rust
/// Memory layout order.
///
/// Xenon only supports F-order (column-major) as its native layout.
/// In the current version, this enum is only used to identify
/// the F-order contiguous layout state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    /// Fortran-order (column-major): first index varies fastest.
    ///
    /// Strides satisfy `strides[i] = product(shape[0..i])`.
    /// This is the only order natively supported by Xenon.
    F,
}
```

`LayoutFlags` 提供从 `Order` 构造标志的便捷方法。该能力仅用于把 F-order 连续状态映射为 `LayoutFlags`；当前版本的布局元数据只使用 `LayoutFlags`、`LayoutState` 与 `Strides<D>`，也不代表支持顺序转换：

````rust
impl LayoutFlags {
    /// Creates a LayoutFlags with F_CONTIGUOUS set (and no other flags).
    ///
    /// Used internally when metadata construction needs to stamp
    /// an F-order contiguous layout state.
    ///
    /// # Examples
    /// ```ignore
    /// let flags = LayoutFlags::from_order(Order::F);
    /// assert!(flags.is_f_contiguous());
    /// ```
    #[inline]
    pub(crate) const fn from_order(order: Order) -> Self {
        match order {
            Order::F => Self(Self::F_CONTIGUOUS.0),
        }
    }
}
````

### 5.1c LayoutState 布局分类 API

除底层 `LayoutFlags` 外，张量层还需要一个更直接的布局分类结果，便于上层 API、诊断输出和跨模块分支选择表达“当前这块逻辑数据在内存中的连续性状态”。

> **说明：** Xenon 的原生构造路径仍以 F-order 为准；`LayoutState` 是一个**分类结果**，用于描述某个 `shape + strides` 组合所呈现的布局状态，而不是放宽当前版本的 F-order only 设计边界。

> **权威定义声明**：`LayoutState` 由本模块（`layout/`）定义并持有。`07-tensor.md` 中的 `TensorBase::layout_state()` 方法返回本模块定义的 `LayoutState`，不再重复定义该枚举。若后续版本需扩展布局状态分类，须在本模块中修改并通过模块间接口暴露。

```rust
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

`layout_state()` 与 `is_f_contiguous()` 等张量层公开方法定义见 `07-tensor.md`；本模块只定义布局分类与判定规则，不承载 `TensorBase` 的方法声明。

### 5.2 步长类型：usize

步长使用 `usize` 存储；当前版本仅接受非负步长与零步长：

```rust
/// Each element in the stride array represents the memory offset change
/// (in elements) when moving one index along that axis.
///
/// - Positive stride: forward traversal
/// - Zero stride: broadcast dimension (repeats the same element)
```

### 5.2a `Strides<D>` 正式定义

`Strides<D>` 是 layout 模块拥有的正式步长类型，用于把“每个轴前进一步需要跨过多少个元素”从 `Dimension` 的形状语义中独立出来。

```rust
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

与 `Dimension`/`TensorBase` 的职责边界如下：

- `02-dimension.md` 中的 `Dimension` 只描述 shape/rank，不保存 stride 语义。
- `Strides<D>` 负责保存与 shape 同 rank 的步长元数据。
- `07-tensor.md` 中的 `TensorBase` 通过 `strides: Strides<D>` 持有这部分元数据，并交由 layout 层推导连续性与布局标志。

负步长布局不在当前版本范围内（参见 `require.md §7`）。当前文档仅讨论由 F-order、转置与广播产生的合法布局。

### 5.3 F-order 步长计算

**算法**：

```
function compute_f_strides(shape: [usize; N]) -> Result<[usize; N], XenonError>:
    strides = array of size N
    cumulative = 1

    for i from 0 to N-1:
        strides[i] = cumulative
        cumulative = checked_mul(cumulative, shape[i])
            or return IntegerOverflow / InvalidLayout

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

```rust
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
- **广播视图**：广播轴允许 stride 为 `0`；其余非广播轴必须保持 F-order 或转置后的正 stride 模式。
- **单元素或 0-D**：只要 metadata 可表示且访问范围合法，任意有效 stride 都视为合法。

#### 非法 stride 组合

- **负步长**：当前版本不支持。
- **可写上下文中的重叠访问**：若多个逻辑索引会写入同一物理位置，则该布局非法；广播视图因此只能作为只读/共享只读视图暴露。

#### 校验规则

- `total_elements = product(shape)`，且计算过程不得溢出 `usize`。
- 访问范围校验必须显式纳入 `offset`。当 `total_elements == 0` 时，要求 `offset <= storage_len`，且不得计算 `shape[i] - 1`；当 `total_elements > 0` 时，
  `max_accessed_offset = offset + sum(stride[i] * (shape[i] - 1) for all i where shape[i] > 0)`（逐轴累加且使用 checked arithmetic）必须满足 `max_accessed_offset < storage_len`。
- 每个 `stride[i]` 都必须可表示为 `isize`，不得发生表示溢出。

#### safe vs unsafe 构造的责任分工

- **Safe constructors**：必须检查以上全部规则；任一条件不满足时返回 `Result::Err`。
- **Unsafe constructors**：至少检查可验证的 metadata 约束（如 shape/stride 一致性、元素总数与访问范围公式）；指针有效性、生命周期与实际可访问内存范围由调用方保证。

### 5.5 对齐检查

```rust
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

```rust
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
    has_zero_stride = any(stride == 0 for stride in strides)
    flags = flags.set_has_zero_stride(has_zero_stride)

    // Distinguish broadcast-introduced zero strides from empty-array degenerate metadata.
    is_broadcast_zero_stride = has_zero_stride and product(shape) > 0

    // 2. Contiguity
    flags = flags.set_f_contiguous(!is_broadcast_zero_stride && is_f_contiguous(shape, strides))

    // 3. Alignment (based on the logical-first-element pointer, not the backing allocation base)
    flags = flags.set_aligned(is_aligned(ptr))

    return flags
```

### 6.2 特殊情况处理

| 情况            | F-contiguous | 说明                        |
| --------------- | :----------: | --------------------------- |
| 空数组 (size=0) |     true     | 无元素，视为连续；退化零步长 metadata 不强制降级为 `BroadcastView` |
| 标量 (ndim=0)   |     true     | 单元素                      |
| 1D 数组         |     true     | 一维情况天然 F-连续         |
| 含 size=1 轴    |  检查其他轴  | size=1 轴的步长不影响连续性 |
| 广播零步长轴    |    false     | 广播视图不得带 `F_CONTIGUOUS` |

### 6.3 标志位更新规则

> 所有 flags 更新规则统一通过 `compute_layout_flags()` 入口执行（参见 §5.9）。

| 操作     | 标志位更新方式                                                                 |
| -------- | ------------------------------------------------------------------------------ |
| 创建     | 调用 `compute_layout_flags()` 统一重新计算                                     |
| 切片     | 调用 `compute_layout_flags()` 统一重新计算（对齐基于切片后逻辑首元素）         |
| 转置     | 调用 `compute_layout_flags()` 统一重新计算                                     |
| 视图创建 | 调用 `compute_layout_flags()` 统一重新计算                                     |
| 广播     | 调用 `compute_layout_flags()` 统一重新计算                                     |
| reshape  | **不在当前版本范围内**（`require.md §17` 当前仅允许 transpose）                |

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

| 场景                       | 预期行为                       |
| -------------------------- | ------------------------------ |
| 空数组 `shape=[0, 3]`      | F-连续为 true；退化零步长仍可保持 `FContiguous` |
| 标量 `shape=()`            | F-连续为 true，步长为空        |
| 1D 数组 `shape=[5]`        | F-连续为 true                  |
| 高维 `shape=[2,2,2,2,2,2]` | F-连续，步长 `[1,2,4,8,16,32]` |

### 8.4 属性测试不变量

| 不变量                                             | 测试方法                                          |
| -------------------------------------------------- | ------------------------------------------------- |
| F-步长乘积 == 总元素数                             | `product(shape[i]) == total`                      |
| 空数组/标量始终 F-连续                             | 随机 0D/空 shape                                  |
| `compute_f_strides` 成功后 `is_f_contiguous` 返回 true | 随机 shape                                     |
| 对齐与非对齐数据元素值一致                         | `from_vec_aligned(v)` 与 `from_vec(v)` 逐元素比较 |

### 8.5 集成测试

| 测试文件               | 测试内容                                                                                                    |
| ---------------------- | ----------------------------------------------------------------------------------------------------------- |
| `tests/test_layout.rs` | `compute_f_strides` / `compute_flags` / `is_aligned` 与 `tensor`、`storage`、`simd`、`ffi` 的端到端协同路径 |

### 8.6 Feature gate / 配置测试

| 配置项 | 覆盖方式                             | 说明                                         |
| ------ | ------------------------------------ | -------------------------------------------- |
| 默认配置 | 常规单元/集成测试路径                 | 本模块无独立 feature gate，默认配置即主路径  |
| 非默认 feature | 不适用                             | 本模块未定义 feature gate，故无额外配置矩阵 |

### 8.7 类型边界 / 编译期测试

| 测试类型 | 覆盖方式                                        | 说明                                                      |
| -------- | ----------------------------------------------- | --------------------------------------------------------- |
| 布局顺序边界 | 编译期验证 `Order` 仅暴露 `F` 变体                | 验证当前版本不会误暴露 C-order                            |
| stride 边界 | raw-parts / layout 相关编译期与运行时测试结合覆盖 | 验证 `Strides<D>` 与 `D` 的维度数保持一致                 |
| 标志位边界 | 编译期验证不依赖外部 bitflags crate               | 验证布局标志保持最小依赖实现                               |

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

| 方向              | 对方模块 | 接口/类型           | 约定                                                                                                        |
| ----------------- | -------- | ------------------- | ----------------------------------------------------------------------------------------------------------- |
| `layout ← tensor` | `tensor` | `LayoutFlags`       | `TensorBase` 直接内联 `LayoutFlags` 作为计算字段，并结合 `LayoutState` / `Strides<D>` 表达布局元数据（参见 `07-tensor.md` §5.1）。 |
| `tensor → layout` | `tensor` | 切片后的 flags 更新 | 切片时统一调用 `compute_layout_flags()` 更新连续性与对齐标志（参见 §5.9、`17-indexing.md` §5）            |
| `tensor → layout` | `tensor` | transpose 后的步长/flags 重算 | transpose 后统一调用 `compute_layout_flags()` 重算 layout state 与 flags（参见 §5.9、`16-shape.md` §5.1） |

### 9.4 与 SIMD 模块

| 方向            | 对方模块 | 接口/类型                            | 约定                                                    |
| --------------- | -------- | ------------------------------------ | ------------------------------------------------------- |
| `simd ← layout` | `simd`   | `is_aligned()` / `is_f_contiguous()` | simd 用这些查询结果做路径选择（参见 `08-simd.md` §5.6） |
| `simd ← layout` | `simd`   | 步长检查                             | simd 继续检查步长是否为 1，以确认连续访问路径           |

### 9.5 与 FFI 模块

| 方向           | 对方模块 | 接口/类型            | 约定                                                                |
| -------------- | -------- | -------------------- | ------------------------------------------------------------------- |
| `ffi ← layout` | `ffi`    | BLAS 兼容检查        | FFI 路径依赖连续、正步长、无零步长等布局前提（参见 `23-ffi.md` §5.5） |
| `ffi ← layout` | `ffi`    | `lda()` 相关步长信息 | FFI 从 layout 步长推导 leading dimension                            |

---

## 10. 错误处理与语义边界

| 项目           | 内容 |
| -------------- | ---- |
| Recoverable error | 对外布局校验失败返回 `XenonError`（由上层构造路径传播），上下文字段应包含 shape、strides、offset 或操作名等布局元数据 |
| Panic | 纯布局查询函数不以 panic 作为常规错误语义；若内部辅助计算在已验证快捷路径上发生整数溢出，可按契约 panic |
| 路径一致性 | scalar 布局、SIMD 路径选择与 parallel 上游消费必须共享同一 `LayoutFlags` 语义，不允许因路径差异改变结果 |
| 容差边界 | 不适用 |

---

## 11. 设计决策记录

### 决策 1：F-order only

| 属性     | 值                                                                       |
| -------- | ------------------------------------------------------------------------ |
| 决策     | Xenon 仅支持 F-order（列优先），不支持 C-order                           |
| 理由     | BLAS/LAPACK 兼容；科学计算惯例；简化设计减少分支；需求说明书 §7 明确要求 |
| 替代方案 | 同时支持 F/C order — 放弃，增加复杂度且当前版本不需要                    |
| 替代方案 | 仅支持 C-order — 放弃，与 BLAS 不兼容                                    |

### 决策 2：步长使用 usize

| 属性     | 值                                                                           |
| -------- | ---------------------------------------------------------------------------- |
| 决策     | 步长类型为 `usize`（无符号）                                                 |
| 理由     | 需求说明书 §7 明确当前版本不支持负步长；`Strides<D>` 只需表达非负步长与零步长广播 |
| 替代方案 | `isize` — 放弃，会暗示当前版本支持负步长                                     |
| 替代方案 | `i64` — 放弃，与平台 `usize` 不一致                                          |

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

| 约束       | 说明                                   |
| ---------- | -------------------------------------- |
| `std` only | 本模块依赖 `std` 环境，不讨论 `no_std` |
| 单 crate   | 保持单 crate 边界                      |
| SemVer     | 布局类型和 stride 计算变更遵循 SemVer  |
| 最小依赖   | 无新增第三方依赖                       |
| 线程安全   | `Strides<D>` 本身是 immutable 的值类型；构造后不可修改，因此无需同步原语即可跨线程共享（`Send + Sync` 自动推导） |

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

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
