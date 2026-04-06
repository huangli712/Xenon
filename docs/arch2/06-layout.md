# 内存布局模块设计

> 模块路径: `src/layout.rs`  
> 设计阶段: Phase 2 — W1（与 dimension、element、error 并行）  
> 依赖: 无外部依赖，仅依赖 `core`

---

## 1. 模块定位

Layout 模块是 Renon 的内存布局基础设施，负责：

1. **步长计算** — 根据形状与 Order 生成有符号步长（F-order / C-order）
2. **布局标志** — 以 O(1) 复杂度缓存高频派生状态（连续性、对齐、广播、反转）
3. **对齐策略** — 默认 64 字节对齐，小数组优化降级，视图继承与降级逻辑
4. **标志重算** — 为 slice / transpose / reshape 提供标志更新函数

本模块是纯计算模块，**不持有数据**，不涉及内存分配。所有函数接收 `(shape, strides, ptr)` 元组，返回计算结果。`TensorBase` 在创建 / 变形时调用本模块完成标志初始化与更新。

### 设计约束

| 约束 | 说明 |
|------|------|
| 无外部依赖 | 仅使用 `core`，不依赖 `dimension` trait |
| 泛型友好 | 所有函数以 `&[usize]`（shape）和 `&[isize]`（strides）为入参，兼容静态维度和动态维度 |
| 零分配 | 所有计算在栈上完成，不触发堆分配 |
| O(1) 查询 | 标志位预计算后，查询方法均为 O(1) |

---

## 2. 文件位置

```
src/layout.rs    // 单文件模块，~500 行
```

`lib.rs` 中声明：

```rust
pub mod layout;
```

---

## 3. 依赖关系

```
layout.rs
  ├── core（仅此一个依赖）
  │
  └── 被以下模块依赖：
        ├── storage/          // 创建 Owned/ArcRepr 时计算 strides + flags
        ├── tensor.rs         // TensorBase 持有 LayoutFlags 字段
        ├── shape/            // slice/transpose/reshape 调用 compute_flags
        ├── broadcast.rs      // 检测 HAS_ZERO_STRIDE
        ├── indexing.rs       // 判断连续性以选择快速路径
        ├── ops/              // 根据 flags 选择 SIMD/标量路径
        └── ffi.rs            // 查询对齐与连续性
```

**关键设计决策：** Layout 模块不依赖 `Dimension` trait。所有函数以 slice 形式接收 shape 和 strides，由调用方负责从 `D::slice()` / `D::slice_mut()` 转换。这使得 layout 可以在 W1 与 dimension 并行开发。

---

## 4. 公共 API 设计

### 4.1 `Order` 枚举

```rust
/// Memory layout order for multi-dimensional arrays.
///
/// F-order (column-major) is the default, optimized for BLAS/LAPACK interoperability.
/// C-order (row-major) is provided for compatibility with C/Python ecosystems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Order {
    /// Column-major order (Fortran-style).
    /// Strides increase from leftmost to rightmost dimension.
    /// For shape [M, N]: strides = [1, M].
    F = 0,
    /// Row-major order (C-style).
    /// Strides increase from rightmost to leftmost dimension.
    /// For shape [M, N]: strides = [N, 1].
    C = 1,
}
```

**方法：**

```rust
impl Order {
    /// Returns `Order::F` (the default column-major order).
    #[inline]
    pub const fn default() -> Order;

    /// Returns `Order::F` as a static constant.
    pub const COLUMN_MAJOR: Order = Order::F;

    /// Returns `Order::C` as a static constant.
    pub const ROW_MAJOR: Order = Order::C;
}
```

> **注意：** `Order` 不实现 `Default` trait，避免隐式默认值。使用 `Order::default()` 方法显式获取。库内所有构造函数使用 F-order 作为默认参数。

### 4.2 `LayoutFlags` 结构体

```rust
/// Bitflags encoding cached layout properties of a tensor.
///
/// Stored as a single `u8` (5 bits used, 3 bits reserved).
/// All query methods are O(1) — they read precomputed bits.
///
/// # Invariants
///
/// - `F_CONTIGUOUS | C_CONTIGUOUS` implies scalar or 1-D array.
/// - For empty arrays (zero elements), both F and C contiguous are set.
/// - Flags are recomputed on every layout-changing operation (slice, transpose, reshape).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct LayoutFlags(u8);
```

**常量定义（内部使用，通过方法暴露）：**

```rust
// Internal bit positions
const F_CONTIGUOUS_BIT:     u8 = 0b0000_0001;
const C_CONTIGUOUS_BIT:     u8 = 0b0000_0010;
const ALIGNED_BIT:          u8 = 0b0000_0100;
const HAS_ZERO_STRIDE_BIT:  u8 = 0b0000_1000;
const HAS_NEG_STRIDE_BIT:   u8 = 0b0001_0000;
```

**构造方法：**

```rust
impl LayoutFlags {
    /// Empty flags (no properties set).
    pub const NONE: LayoutFlags = LayoutFlags(0);

    /// Creates `LayoutFlags` from a raw `u8` value.
    ///
    /// # Safety
    ///
    /// The caller must ensure only the lower 5 bits are set.
    #[inline]
    pub const fn from_bits(bits: u8) -> LayoutFlags;

    /// Returns the raw underlying `u8` value.
    #[inline]
    pub const fn bits(self) -> u8;
}
```

**查询方法（全部 O(1)）：**

```rust
impl LayoutFlags {
    /// Returns `true` if the data is contiguous in F-order (column-major).
    #[inline]
    pub const fn is_f_contiguous(self) -> bool;

    /// Returns `true` if the data is contiguous in C-order (row-major).
    #[inline]
    pub const fn is_c_contiguous(self) -> bool;

    /// Returns `true` if contiguous in either order.
    #[inline]
    pub const fn is_contiguous(self) -> bool;

    /// Returns `true` if the data pointer is SIMD-aligned (64 bytes).
    #[inline]
    pub const fn is_aligned(self) -> bool;

    /// Returns `true` if any dimension has stride 0 (broadcast dimension).
    #[inline]
    pub const fn has_zero_stride(self) -> bool;

    /// Returns `true` if any dimension has a negative stride (reversed dimension).
    #[inline]
    pub const fn has_neg_stride(self) -> bool;
}
```

**组合/修改方法：**

```rust
impl LayoutFlags {
    /// Sets the given flags (bitwise OR).
    #[inline]
    pub const fn insert(self, other: LayoutFlags) -> LayoutFlags;

    /// Removes the given flags (bitwise AND NOT).
    #[inline]
    pub const fn remove(self, other: LayoutFlags) -> LayoutFlags;

    /// Returns `true` if any of the given flags are set.
    #[inline]
    pub const fn intersects(self, other: LayoutFlags) -> bool;

    /// Returns `true` if all of the given flags are set.
    #[inline]
    pub const fn contains(self, other: LayoutFlags) -> bool;
}
```

### 4.3 步长计算函数

```rust
/// Computes strides for the given shape in the specified order.
///
/// Strides are in element units (not bytes), signed.
///
/// # Arguments
///
/// * `shape` - The array dimensions.
/// * `order` - Memory layout order (F or C).
///
/// # Returns
///
/// A `Vec<isize>` of strides with the same length as `shape`.
///
/// # Panics
///
/// Panics if any dimension is 0 and a non-zero stride would overflow `isize`.
/// For shape with a zero dimension, all strides are 0.
///
/// # Examples
///
/// ```ignore
/// let strides = compute_strides(&[3, 4], Order::F);
/// assert_eq!(strides, vec![1, 3]);
///
/// let strides = compute_strides(&[3, 4], Order::C);
/// assert_eq!(strides, vec![4, 1]);
/// ```
pub fn compute_strides(shape: &[usize], order: Order) -> Vec<isize>;
```

```rust
/// Computes strides into a pre-allocated buffer (no heap allocation for static dims).
///
/// # Arguments
///
/// * `shape` - The array dimensions.
/// * `order` - Memory layout order.
/// * `out` - Output buffer, must have the same length as `shape`.
///
/// # Panics
///
/// Panics if `out.len() != shape.len()`.
pub fn compute_strides_into(shape: &[usize], order: Order, out: &mut [isize]);
```

```rust
/// Returns the total number of elements for the given shape.
///
/// Returns 0 if any dimension is 0. Returns 1 for empty shape (Ix0 / scalar).
///
/// # Panics
///
/// Panics on overflow.
#[inline]
pub fn shape_to_elem_count(shape: &[usize]) -> usize;
```

### 4.4 对齐查询函数

```rust
/// Default SIMD alignment in bytes (AVX-512 cache line).
pub const DEFAULT_ALIGNMENT: usize = 64;

/// Checks whether the given pointer is aligned to the specified byte boundary.
///
/// # Arguments
///
/// * `ptr` - Data pointer (may be `*const T` or `*mut T` cast to `usize`).
/// * `element_size` - Size of a single element in bytes (`size_of::<T>()`).
/// * `alignment` - Required alignment in bytes (must be a power of 2).
///
/// # Returns
///
/// `true` if `(ptr_addr / element_size) * element_size` is divisible by `alignment`.
/// For element-level strides, checks that the element index is aligned.
#[inline]
pub fn is_ptr_aligned(ptr_addr: usize, element_size: usize, alignment: usize) -> bool;

/// Validates that the given alignment value is a valid power of 2
/// and is >= the natural alignment of the element type.
///
/// # Arguments
///
/// * `alignment` - Requested alignment in bytes.
/// * `natural_alignment` - Natural alignment of the element type (`align_of::<T>()`).
///
/// # Returns
///
/// `Ok(())` if valid, `Err(AlignmentError)` otherwise.
pub fn validate_alignment(alignment: usize, natural_alignment: usize) -> Result<(), AlignmentError>;
```

```rust
/// Error type for invalid alignment values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlignmentError {
    /// The alignment is not a power of 2.
    NotPowerOfTwo { value: usize },
    /// The alignment is smaller than the element's natural alignment.
    BelowNatural { requested: usize, natural: usize },
}
```

### 4.5 标志计算函数

```rust
/// Computes all layout flags from shape, strides, and pointer address.
///
/// This is the primary entry point called by `TensorBase` constructors and
/// layout-changing operations (slice, transpose, reshape).
///
/// # Arguments
///
/// * `shape` - Array dimensions.
/// * `strides` - Strides in element units (same length as `shape`).
/// * `ptr_addr` - Address of the data pointer (for alignment check).
/// * `element_size` - Size of a single element in bytes.
/// * `alignment` - Required alignment boundary in bytes.
///
/// # Returns
///
/// Complete `LayoutFlags` with all 5 bits computed.
///
/// # Complexity
///
/// O(ndim) — called once per layout change, then queries are O(1).
///
/// # Panics
///
/// Panics if `shape.len() != strides.len()`.
pub fn compute_flags(
    shape: &[usize],
    strides: &[isize],
    ptr_addr: usize,
    element_size: usize,
    alignment: usize,
) -> LayoutFlags;
```

### 4.6 标志增量更新函数

以下函数用于特定操作后的高效标志重算。当操作类型已知时，可避免完整重算。

```rust
/// Recomputes flags after a slice operation.
///
/// Slicing preserves strides but changes shape and potentially the data offset.
/// Alignment may degrade if the new offset breaks SIMD alignment.
///
/// # Arguments
///
/// * `shape` - New shape after slicing.
/// * `strides` - Strides (unchanged by slice).
/// * `new_ptr_addr` - New data pointer address after applying offset.
/// * `element_size` - Size of a single element in bytes.
/// * `alignment` - Required alignment boundary in bytes.
pub fn flags_after_slice(
    shape: &[usize],
    strides: &[isize],
    new_ptr_addr: usize,
    element_size: usize,
    alignment: usize,
) -> LayoutFlags;
```

```rust
/// Recomputes flags after a transpose operation.
///
/// Transpose swaps axes, which swaps the corresponding strides.
/// This changes contiguity but preserves alignment and zero/negative stride properties.
///
/// # Arguments
///
/// * `shape` - Shape after transpose (axes permuted).
/// * `strides` - Strides after transpose (axes permuted).
/// * `ptr_addr` - Data pointer address (unchanged by transpose).
/// * `element_size` - Size of a single element in bytes.
/// * `alignment` - Required alignment boundary in bytes.
pub fn flags_after_transpose(
    shape: &[usize],
    strides: &[isize],
    ptr_addr: usize,
    element_size: usize,
    alignment: usize,
) -> LayoutFlags;
```

```rust
/// Recomputes flags after a reshape operation.
///
/// Reshape may change both shape and strides. If the source is contiguous,
/// strides are recalculated for the new shape. Non-contiguous sources may
/// not be reshapeable.
///
/// # Arguments
///
/// * `shape` - New shape.
/// * `strides` - New strides (recalculated for the new shape).
/// * `ptr_addr` - Data pointer address (unchanged by reshape).
/// * `element_size` - Size of a single element in bytes.
/// * `alignment` - Required alignment boundary in bytes.
pub fn flags_after_reshape(
    shape: &[usize],
    strides: &[isize],
    ptr_addr: usize,
    element_size: usize,
    alignment: usize,
) -> LayoutFlags;
```

### 4.7 小数组对齐优化

```rust
/// Determines the effective alignment for an array allocation.
///
/// For small arrays where `elem_count * elem_size <= DEFAULT_ALIGNMENT`,
/// the alignment may be downgraded to the element's natural alignment
/// to avoid wasting memory.
///
/// # Arguments
///
/// * `elem_count` - Total number of elements.
/// * `elem_size` - Size of a single element in bytes.
/// * `natural_alignment` - Natural alignment of the element type.
///
/// # Returns
///
/// The effective alignment to use (power of 2, >= natural alignment).
#[inline]
pub fn effective_alignment(elem_count: usize, elem_size: usize, natural_alignment: usize) -> usize;
```

---

## 5. 内部实现设计

### 5.1 标志位计算逻辑（`compute_flags`）

```
compute_flags(shape, strides, ptr_addr, elem_size, alignment):
    flags = NONE
    
    // 1. F-order contiguity
    if is_f_contiguous_impl(shape, strides):
        flags |= F_CONTIGUOUS
    
    // 2. C-order contiguity
    if is_c_contiguous_impl(shape, strides):
        flags |= C_CONTIGUOUS
    
    // 3. SIMD alignment
    if is_ptr_aligned(ptr_addr, elem_size, alignment):
        flags |= ALIGNED
    
    // 4. Zero stride (broadcast)
    for s in strides:
        if s == 0:
            flags |= HAS_ZERO_STRIDE
            break
    
    // 5. Negative stride
    for s in strides:
        if s < 0:
            flags |= HAS_NEG_STRIDE
            break
    
    return flags
```

**F-order 连续性判定算法：**

```
is_f_contiguous_impl(shape, strides):
    // Empty array or scalar: both contiguous
    if shape.is_empty(): return true
    
    expected = 1
    for i in 0..ndim:
        if shape[i] == 1: continue       // size-1 dim: stride irrelevant
        if shape[i] == 0: return true    // zero-size dim: trivially contiguous
        if strides[i] != expected: return false
        expected *= shape[i]
    return true
```

**C-order 连续性判定算法：**

```
is_c_contiguous_impl(shape, strides):
    if shape.is_empty(): return true
    
    expected = 1
    for i in (0..ndim).rev():            // reverse iteration
        if shape[i] == 1: continue
        if shape[i] == 0: return true
        if strides[i] != expected: return false
        expected *= shape[i]
    return true
```

**关键边界情况：**

| 场景 | F_CONTIGUOUS | C_CONTIGUOUS |
|------|:---:|:---:|
| 空形状 `[]`（Ix0 标量） | ✅ | ✅ |
| `[0, 3]`（零元素） | ✅ | ✅ |
| `[1, 1]`（单元素） | ✅ | ✅ |
| `[3]`（1D） | ✅ | ✅ |
| `[3, 4]` strides `[1, 3]` | ✅ | ❌ |
| `[3, 4]` strides `[4, 1]` | ❌ | ✅ |
| `[3, 4]` strides `[2, 6]` | ❌ | ❌ |
| 广播 `[1, 4]` strides `[0, 1]` | ❌ | ❌ |

### 5.2 步长计算

**F-order 步长计算：**

```
compute_strides_f(shape):
    strides = [0isize; ndim]
    s = 1
    for i in 0..ndim:
        strides[i] = s
        if shape[i] != 0:
            s *= shape[i]    // checked multiplication, panic on overflow
    return strides
```

对于 shape `[3, 4, 5]`，F-order strides = `[1, 3, 12]`。

**C-order 步长计算：**

```
compute_strides_c(shape):
    strides = [0isize; ndim]
    s = 1
    for i in (0..ndim).rev():
        strides[i] = s
        if shape[i] != 0:
            s *= shape[i]
    return strides
```

对于 shape `[3, 4, 5]`，C-order strides = `[20, 5, 1]`。

### 5.3 对齐检查逻辑

**指针对齐判定：**

```rust
fn is_ptr_aligned(ptr_addr: usize, element_size: usize, alignment: usize) -> bool {
    // The pointer address must be divisible by alignment.
    // ptr_addr is already in bytes (cast from *const T or *mut T).
    // Since we store strides in element units, we only need to check
    // that the byte address is aligned.
    ptr_addr % alignment == 0
}
```

**小数组对齐降级：**

```rust
fn effective_alignment(elem_count: usize, elem_size: usize, natural_alignment: usize) -> usize {
    let total_bytes = elem_count * elem_size;
    if total_bytes <= DEFAULT_ALIGNMENT {
        // Small array: downgrade to natural alignment
        natural_alignment
    } else {
        DEFAULT_ALIGNMENT
    }
}
```

### 5.4 标志更新规则

#### 切片（Slice）

| 属性 | 规则 |
|------|------|
| 步长 | 不变（继承源数组） |
| 形状 | 由切片范围决定 |
| F/C 连续性 | **可能降级**：若切片跳过了部分元素，步长不变但 shape 变小，连续性保持。但若步长因广播为 0，则非连续 |
| 对齐 | **可能降级**：起始偏移量改变后，新指针地址可能不再 64 字节对齐 |
| 零步长 | 继承源 |
| 负步长 | 继承源 |

实现：`flags_after_slice` 实际上是 `compute_flags` 的别名（全量重算），因为切片后多项属性可能同时变化。命名别名是为了语义清晰。

#### 转置（Transpose）

| 属性 | 规则 |
|------|------|
| 步长 | 按轴交换重排 |
| 形状 | 按轴交换重排 |
| F/C 连续性 | **翻转**：F 连续变为 C 连续（或反之），或都变为不连续 |
| 对齐 | **不变**：数据指针不变 |
| 零步长 | 不变（步长值不变，仅重排） |
| 负步长 | 不变 |

实现：`flags_after_transpose` 也是 `compute_flags` 的语义别名，因为连续性可能翻转。

#### 重塑（Reshape）

| 属性 | 规则 |
|------|------|
| 步长 | 为新 shape 重新计算（仅当源连续时允许） |
| 形状 | 新形状 |
| F/C 连续性 | **取决于新步长** |
| 对齐 | **不变**（数据指针不变，但需重查） |
| 零步长 | 不可能（新分配的步长不含 0） |
| 负步长 | 不可能（新分配的步长不含负值） |

实现：`flags_after_reshape` 调用 `compute_flags`。

#### 广播（Broadcast）

| 属性 | 规则 |
|------|------|
| 步长 | 广播维度设为 0 |
| 形状 | 扩展到目标形状 |
| F/C 连续性 | **通常丢失**（除非所有广播维度 size=1） |
| 对齐 | 继承源 |
| 零步长 | **新增** |
| 负步长 | 继承源 |

#### 反转（Reverse / Flip）

| 属性 | 规则 |
|------|------|
| 步长 | 指定轴步长取反，偏移量调整到末端 |
| 形状 | 不变 |
| F/C 连续性 | **丢失**（负步长不满足连续性判定） |
| 对齐 | **可能降级**（偏移量变化） |
| 零步长 | 继承源 |
| 负步长 | **新增** |

---

## 6. 实现任务拆分

每个任务约 10 分钟，遵循"先测试后实现"原则。

### Task 1: Order 枚举
```
Task:   Implement Order enum with F/C variants and methods
File:   src/layout.rs
Tests:  #[cfg(test)] mod tests::test_order_variants
        #[cfg(test)] mod tests::test_order_default_is_f
前置:   无
预计:   8 min
```
- [ ] 定义 `Order` 枚举（`F`, `C`）
- [ ] 实现 `Order::default()` 返回 `Order::F`
- [ ] 定义 `COLUMN_MAJOR` / `ROW_MAJOR` 常量
- [ ] 测试：`Order::default() == Order::F`、两个变体不等

### Task 2: LayoutFlags 结构体与位运算
```
Task:   Implement LayoutFlags struct with bit operations
File:   src/layout.rs
Tests:  #[cfg(test)] mod tests::test_layout_flags_insert_remove
        #[cfg(test)] mod tests::test_layout_flags_intersects_contains
前置:   无
预计:   10 min
```
- [ ] 定义 `LayoutFlags(u8)` + `#[repr(transparent)]`
- [ ] 定义 5 个内部位常量
- [ ] 实现 `NONE`、`from_bits`、`bits`
- [ ] 实现 `insert`、`remove`、`intersects`、`contains`
- [ ] 测试：位运算正确性

### Task 3: LayoutFlags 查询方法
```
Task:   Implement LayoutFlags query methods (is_f_contiguous, etc.)
File:   src/layout.rs
Tests:  #[cfg(test)] mod tests::test_flags_query_methods
前置:   Task 2
预计:   8 min
```
- [ ] 实现 6 个 `const fn` 查询方法
- [ ] 测试：各种标志组合的查询结果

### Task 4: 步长计算函数
```
Task:   Implement compute_strides and compute_strides_into
File:   src/layout.rs
Tests:  #[cfg(test)] mod tests::test_compute_strides_f_order
        #[cfg(test)] mod tests::test_compute_strides_c_order
        #[cfg(test)] mod tests::test_compute_strides_1d_scalar
        #[cfg(test)] mod tests::test_compute_strides_zero_dim
前置:   Task 1
预计:   12 min
```
- [ ] 实现 `compute_strides(shape, order) -> Vec<isize>`
- [ ] 实现 `compute_strides_into(shape, order, out)`
- [ ] 实现 `shape_to_elem_count(shape) -> usize`
- [ ] 测试：F-order / C-order 各种形状（0D, 1D, 2D, 3D, 零维度）

### Task 5: 对齐检查与验证
```
Task:   Implement alignment checking functions
File:   src/layout.rs
Tests:  #[cfg(test)] mod tests::test_is_ptr_aligned
        #[cfg(test)] mod tests::test_validate_alignment
        #[cfg(test)] mod tests::test_effective_alignment_small_array
前置:   无
预计:   10 min
```
- [ ] 定义 `AlignmentError` 枚举
- [ ] 实现 `is_ptr_aligned`
- [ ] 实现 `validate_alignment`
- [ ] 实现 `effective_alignment`（小数组降级）
- [ ] 测试：对齐地址、非对齐地址、无效对齐值、小数组降级

### Task 6: 连续性判定内部函数
```
Task:   Implement is_f_contiguous_impl and is_c_contiguous_impl
File:   src/layout.rs
Tests:  #[cfg(test)] mod tests::test_f_contiguous_various
        #[cfg(test)] mod tests::test_c_contiguous_various
        #[cfg(test)] mod tests::test_contiguous_edge_cases
前置:   无
预计:   10 min
```
- [ ] 实现 `is_f_contiguous_impl(shape, strides) -> bool`（内部函数）
- [ ] 实现 `is_c_contiguous_impl(shape, strides) -> bool`（内部函数）
- [ ] 测试：正常情况、size-1 维度、零元素、标量、非连续

### Task 7: compute_flags 主函数
```
Task:   Implement compute_flags function
File:   src/layout.rs
Tests:  #[cfg(test)] mod tests::test_compute_flags_full
        #[cfg(test)] mod tests::test_compute_flags_broadcast
        #[cfg(test)] mod tests::test_compute_flags_neg_stride
前置:   Task 3, Task 5, Task 6
预计:   10 min
```
- [ ] 实现 `compute_flags` 组合所有 5 个标志位
- [ ] 测试：完整标志组合（连续+对齐、广播、负步长、混合情况）

### Task 8: 增量更新函数
```
Task:   Implement flags_after_slice, flags_after_transpose, flags_after_reshape
File:   src/layout.rs
Tests:  #[cfg(test)] mod tests::test_flags_after_slice
        #[cfg(test)] mod tests::test_flags_after_transpose
        #[cfg(test)] mod tests::test_flags_after_reshape
前置:   Task 7
预计:   10 min
```
- [ ] 实现 `flags_after_slice`（语义别名 + 文档说明降级规则）
- [ ] 实现 `flags_after_transpose`（语义别名 + 文档说明翻转规则）
- [ ] 实现 `flags_after_reshape`（语义别名 + 文档说明重算规则）
- [ ] 测试：每种操作后的标志变化

### Task 9: Display 实现
```
Task:   Implement Display for Order, LayoutFlags, AlignmentError
File:   src/layout.rs
Tests:  #[cfg(test)] mod tests::test_display_impls
前置:   Task 1, Task 2, Task 5
预计:   8 min
```
- [ ] `Display for Order` → `"F"` / `"C"`
- [ ] `Display for LayoutFlags` → `"LayoutFlags(F_CONTIGUOUS | ALIGNED)"`
- [ ] `Display for AlignmentError` → 人类可读错误信息
- [ ] `Error for AlignmentError`（或 `thiserror` 派生）
- [ ] 测试：格式化输出

### Task 10: 边界测试与文档注释
```
Task:   Add comprehensive edge-case tests and complete doc comments
File:   src/layout.rs
Tests:  #[cfg(test)] mod tests::test_edge_cases_zero_size
        #[cfg(test)] mod tests::test_edge_cases_single_element
        #[cfg(test)] mod tests::test_overflow_protection
前置:   Task 1-9
预计:   10 min
```
- [ ] 零大小数组（shape 含 0）标志
- [ ] 单元素数组标志（应 F+C+ALIGNED）
- [ ] 标量（Ix0，空 shape/strides）标志
- [ ] 大维度溢出保护测试
- [ ] 补全所有 pub 项的 doc comment

---

## 7. 测试计划

### 7.1 单元测试（`src/layout.rs` 内 `#[cfg(test)] mod tests`）

| 测试函数 | 测试内容 |
|----------|----------|
| `test_order_variants` | Order::F ≠ Order::C |
| `test_order_default_is_f` | Order::default() == F |
| `test_layout_flags_insert_remove` | 位插入/移除 |
| `test_layout_flags_intersects_contains` | 位交集/包含 |
| `test_flags_query_methods` | 6 个查询方法在各种组合下正确 |
| `test_compute_strides_f_order` | F-order 步长（1D/2D/3D） |
| `test_compute_strides_c_order` | C-order 步长（1D/2D/3D） |
| `test_compute_strides_1d_scalar` | 1D 步长 [1]、标量步长 [] |
| `test_compute_strides_zero_dim` | shape 含 0 时步长全为 0 后续位正确 |
| `test_is_ptr_aligned` | 对齐/非对齐地址 |
| `test_validate_alignment` | 幂 2 检查、自然对齐检查 |
| `test_effective_alignment_small_array` | 小数组降级到自然对齐 |
| `test_f_contiguous_various` | F 连续性判定（连续/非连续/size-1） |
| `test_c_contiguous_various` | C 连续性判定（连续/非连续/size-1） |
| `test_contiguous_edge_cases` | 标量、零元素、单元素 |
| `test_compute_flags_full` | 完整标志组合 |
| `test_compute_flags_broadcast` | HAS_ZERO_STRIDE 标志 |
| `test_compute_flags_neg_stride` | HAS_NEG_STRIDE 标志 |
| `test_flags_after_slice` | 切片后标志变化 |
| `test_flags_after_transpose` | 转置后 F↔C 翻转 |
| `test_flags_after_reshape` | 重塑后标志重算 |
| `test_display_impls` | Display 格式化 |
| `test_edge_cases_zero_size` | 零元素数组 |
| `test_edge_cases_single_element` | 单元素数组（双连续+对齐） |
| `test_overflow_protection` | 大 shape 乘法溢出 |

### 7.2 关键测试用例详情

#### F-order 连续性

```rust
// shape=[3, 4], strides=[1, 3] → F contiguous, not C
assert!(is_f_contiguous_impl(&[3, 4], &[1, 3]));
assert!(!is_c_contiguous_impl(&[3, 4], &[1, 3]));

// shape=[3, 4], strides=[1, 3] with alignment at 64 → ALIGNED | F_CONTIGUOUS
let flags = compute_flags(&[3, 4], &[1, 3], 64, 8, 64);
assert!(flags.is_f_contiguous());
assert!(!flags.is_c_contiguous());
assert!(flags.is_aligned());
```

#### C-order 连续性

```rust
// shape=[3, 4], strides=[4, 1] → C contiguous, not F
assert!(!is_f_contiguous_impl(&[3, 4], &[4, 1]));
assert!(is_c_contiguous_impl(&[3, 4], &[4, 1]));
```

#### 1D 数组：双连续

```rust
// shape=[5], strides=[1] → both F and C contiguous
assert!(is_f_contiguous_impl(&[5], &[1]));
assert!(is_c_contiguous_impl(&[5], &[1]));
```

#### 广播：零步长

```rust
// shape=[3, 4], strides=[0, 1] → broadcast in dim 0
let flags = compute_flags(&[3, 4], &[0, 1], 64, 8, 64);
assert!(flags.has_zero_stride());
assert!(!flags.is_f_contiguous());  // stride[0]=0 ≠ 1
```

#### 反转：负步长

```rust
// shape=[3, 4], strides=[-1, 3] → reversed dim 0
let flags = compute_flags(&[3, 4], &[-1, 3], 64, 8, 64);
assert!(flags.has_neg_stride());
assert!(!flags.is_f_contiguous());
```

#### 转置：F↔C 翻转

```rust
// Original: shape=[3, 4], strides=[1, 3] → F_CONTIGUOUS
// Transpose: shape=[4, 3], strides=[3, 1] → C_CONTIGUOUS
let flags = flags_after_transpose(&[4, 3], &[3, 1], 64, 8, 64);
assert!(!flags.is_f_contiguous());
assert!(flags.is_c_contiguous());
assert!(flags.is_aligned());  // ptr unchanged
```

#### 小数组对齐降级

```rust
// 4 x f64 = 32 bytes < 64 → allow natural alignment (8 bytes)
let align = effective_alignment(4, 8, 8);
assert_eq!(align, 8);

// 16 x f64 = 128 bytes > 64 → use full alignment
let align = effective_alignment(16, 8, 8);
assert_eq!(align, 64);
```

### 7.3 集成测试（后续阶段）

以下测试在 `tests/` 目录，依赖 tensor 和 storage 模块完成后编写：

| 文件 | 测试内容 |
|------|----------|
| `tests/construction.rs` | `zeros`/`ones` 创建后 flags 正确 |
| `tests/shape_ops.rs` | slice/transpose/reshape 后 flags 降级/翻转 |
| `tests/broadcasting.rs` | 广播后 HAS_ZERO_STRIDE |
| `tests/ffi.rs` | 指针对齐查询 |
