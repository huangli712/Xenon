# Xenon 内存布局模块设计文档

> 版本: 0.1.0 | 最后更新: 2026-03-28

---

## 1. 模块概述

### 1.1 核心角色

内存布局模块 (`src/layout/`) 是 Xenon 张量库的性能优化核心。它负责描述和管理多维数组在内存中的物理排列方式，直接影响：

| 影响领域 | 说明 |
|----------|------|
| **SIMD 加速** | 连续内存布局可启用向量化路径，非连续布局需回退标量 |
| **缓存效率** | 顺缓存行的访问模式减少 cache miss |
| **BLAS 兼容** | F-order 布局与 Fortran/BLAS 惯例一致 |
| **FFI 互操作** | 正确的步长和布局信息是跨语言调用的基础 |

### 1.2 设计目标

1. **O(1) 布局查询**：通过缓存布局标志位，将高频查询（连续性、对齐等）优化为常数时间
2. **有符号步长支持**：支持负步长以表达反转视图，支持零步长以表达广播维度
3. **对齐感知**：64 字节默认对齐（AVX-512 缓存行），支持运行时对齐状态查询
4. **零成本抽象**：布局信息为纯元数据，不引入运行时开销

### 1.3 在架构中的位置

```
依赖层级：

L2: layout ←── error, dimension
         ↓
L3: storage ←── layout, element
         ↓
L4: tensor ←── storage, layout, dimension
```

内存布局模块依赖维度系统（需要 `Dimension` trait）和错误处理，被存储模块和张量核心模块依赖。

---

## 2. 文件结构

```
src/layout/
├── mod.rs             # Layout 类型定义，公开 API
├── flags.rs           # LayoutFlags: u8 位域定义和操作
├── strides.rs         # 步长计算、验证、负步长处理
└── contiguous.rs      # 连续性检测算法（F/C contiguous）
```

### 2.1 文件职责

| 文件 | 职责 | 核心类型/函数 |
|------|------|---------------|
| `mod.rs` | 模块入口，`Layout` 结构体定义，公开 API | `Layout`, `Order` |
| `flags.rs` | 布局标志位的位定义和位操作 | `LayoutFlags` |
| `strides.rs` | 从 shape 计算步长，步长合法性验证 | `compute_strides()`, `validate_strides()` |
| `contiguous.rs` | 判断 F/C 连续性的算法 | `is_f_contiguous()`, `is_c_contiguous()` |

---

## 3. LayoutFlags 设计

### 3.1 位定义

使用 `u8` 类型存储 5 个布局标志位，占用 1 字节：

```rust
// 位布局（从低位到高位）
// Bit 0: F_CONTIGUOUS
// Bit 1: C_CONTIGUOUS
// Bit 2: ALIGNED
// Bit 3: HAS_ZERO_STRIDE
// Bit 4: HAS_NEG_STRIDE
// Bit 5-7: 保留（未使用）
```

**常量定义**：

```rust
/// F-order 连续性标志
pub const F_CONTIGUOUS: u8 = 0b0000_0001;  // 0x01

/// C-order 连续性标志
pub const C_CONTIGUOUS: u8 = 0b0000_0010;  // 0x02

/// SIMD 对齐标志（64 字节）
pub const ALIGNED: u8 = 0b0000_0100;  // 0x04

/// 零步长标志（广播维度）
pub const HAS_ZERO_STRIDE: u8 = 0b0000_1000;  // 0x08

/// 负步长标志（反转维度）
pub const HAS_NEG_STRIDE: u8 = 0b0001_0000;  // 0x10
```

### 3.2 组合语义

| 标志组合 | 含义 | 示例场景 |
|----------|------|----------|
| `F_CONTIGUOUS \| C_CONTIGUOUS` | 标量或 1D 数组 | `Tensor1<[1]>`, `Tensor0` |
| `F_CONTIGUOUS && !C_CONTIGUOUS` | 纯 F-order 连续 | 新创建的 2D+ 张量（默认） |
| `!F_CONTIGUOUS && C_CONTIGUOUS` | 纯 C-order 连续 | 显式 C-order 创建的张量 |
| `!F_CONTIGUOUS && !C_CONTIGUOUS` | 非连续 | 切片、转置后的视图 |
| `ALIGNED && F_CONTIGUOUS` | SIMD 友好 | 可使用向量化路径 |
| `HAS_ZERO_STRIDE` | 存在广播维度 | 广播后的视图 |
| `HAS_NEG_STRIDE` | 存在反转维度 | `flip()` 后的视图 |

### 3.3 LayoutFlags 结构体

```rust
/// 布局标志位集合
/// 
/// 使用位域存储多个布局属性，支持 O(1) 查询。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LayoutFlags(u8);

impl LayoutFlags {
    /// 空标志（所有位为 0）
    pub const EMPTY: Self = Self(0b0000_0000);
    
    /// 所有有效标志
    pub const ALL: Self = Self(0b0001_1111);
    
    // === 构造方法 ===
    
    /// 从原始位值创建
    pub const fn from_bits(bits: u8) -> Self;
    
    /// 获取原始位值
    pub const fn bits(self) -> u8;
    
    // === 查询方法 ===
    
    /// 是否 F-order 连续
    #[inline]
    pub const fn is_f_contiguous(self) -> bool;
    
    /// 是否 C-order 连续
    #[inline]
    pub const fn is_c_contiguous(self) -> bool;
    
    /// 是否任一方向连续
    #[inline]
    pub const fn is_contiguous(self) -> bool;
    
    /// 是否 64 字节对齐
    #[inline]
    pub const fn is_aligned(self) -> bool;
    
    /// 是否存在零步长
    #[inline]
    pub const fn has_zero_stride(self) -> bool;
    
    /// 是否存在负步长
    #[inline]
    pub const fn has_neg_stride(self) -> bool;
    
    // === 修改方法 ===
    
    /// 设置 F-order 连续标志
    #[inline]
    pub const fn set_f_contiguous(self, val: bool) -> Self;
    
    /// 设置 C-order 连续标志
    #[inline]
    pub const fn set_c_contiguous(self, val: bool) -> Self;
    
    /// 设置对齐标志
    #[inline]
    pub const fn set_aligned(self, val: bool) -> Self;
    
    /// 设置零步长标志
    #[inline]
    pub const fn set_has_zero_stride(self, val: bool) -> Self;
    
    /// 设置负步长标志
    #[inline]
    pub const fn set_has_neg_stride(self, val: bool) -> Self;
}
```

### 3.4 位操作实现示意

```rust
impl LayoutFlags {
    // 查询：使用按位与
    pub const fn is_f_contiguous(self) -> bool {
        (self.0 & F_CONTIGUOUS) != 0
    }
    
    // 设置：使用按位或（设置）和按位与（清除）
    pub const fn set_f_contiguous(self, val: bool) -> Self {
        if val {
            Self(self.0 | F_CONTIGUOUS)
        } else {
            Self(self.0 & !F_CONTIGUOUS)
        }
    }
}
```

---

## 4. Order 枚举

### 4.1 定义

```rust
/// 内存布局顺序
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Order {
    /// Fortran-order（列优先）
    /// 
    /// 第一维度变化最慢，最后一维度变化最快。
    /// 例如 2D 数组 `[M, N]` 在内存中按列存储：
    /// `[0,0], [1,0], ..., [M-1,0], [0,1], [1,1], ...`
    F,
    
    /// C-order（行优先）
    /// 
    /// 最后一维度变化最慢，第一维度变化最快。
    /// 例如 2D 数组 `[M, N]` 在内存中按行存储：
    /// `[0,0], [0,1], ..., [0,N-1], [1,0], [1,1], ...`
    C,
}
```

### 4.2 Order 与步长的关系

对于形状 `[d0, d1, d2, ..., d_{n-1}]` 的 n 维数组：

| Order | 步长计算方向 | 步长公式 |
|-------|--------------|----------|
| F | 从左到右累积 | `stride[i] = d0 * d1 * ... * d_{i-1}` |
| C | 从右到左累积 | `stride[i] = d_{i+1} * ... * d_{n-1}` |

**示例**：形状 `[2, 3, 4]` 的步长

| Order | stride[0] | stride[1] | stride[2] |
|-------|-----------|-----------|-----------|
| F | 1 | 2 | 6 |
| C | 12 | 4 | 1 |

---

## 5. 步长计算

### 5.1 步长类型设计

步长使用有符号整数 `isize` 存储，原因：

1. **负步长支持**：反转视图（如 `flip()`）需要负步长
2. **零步长支持**：广播维度使用零步长
3. **单位为元素个数**：非字节，便于索引计算

```rust
/// 步长数组
/// 
/// 每个元素的值表示沿该轴移动一个索引时，内存偏移量的变化（元素单位）。
pub type Strides = SmallVec<[isize; 6]>;
```

### 5.2 F-order 步长计算算法

**算法描述**：

F-order 下，第 `i` 轴的步长等于前面所有轴的长度乘积。

**伪代码**：

```
function compute_f_strides(shape: [usize; N]) -> [isize; N]:
    strides = array of size N
    cumulative = 1
    
    for i from 0 to N-1:
        strides[i] = cumulative
        cumulative = cumulative * shape[i]
    
    return strides
```

**示例**：

```
shape = [3, 4, 5]
i=0: stride[0] = 1,  cumulative = 1 * 3 = 3
i=1: stride[1] = 3,  cumulative = 3 * 4 = 12
i=2: stride[2] = 12, cumulative = 12 * 5 = 60

结果: strides = [1, 3, 12]
```

### 5.3 C-order 步长计算算法

**算法描述**：

C-order 下，第 `i` 轴的步长等于后面所有轴的长度乘积。

**伪代码**：

```
function compute_c_strides(shape: [usize; N]) -> [isize; N]:
    strides = array of size N
    cumulative = 1
    
    for i from N-1 down to 0:
        strides[i] = cumulative
        cumulative = cumulative * shape[i]
    
    return strides
```

**示例**：

```
shape = [3, 4, 5]
i=2: stride[2] = 1,  cumulative = 1 * 5 = 5
i=1: stride[1] = 5,  cumulative = 5 * 4 = 20
i=0: stride[0] = 20, cumulative = 20 * 3 = 60

结果: strides = [20, 5, 1]
```

### 5.4 通用步长计算 API

```rust
/// 根据形状和顺序计算连续布局的步长
/// 
/// # Arguments
/// * `shape` - 各轴长度
/// * `order` - 布局顺序
/// 
/// # Returns
/// 步长数组，长度与 shape 相同
pub fn compute_strides(shape: &[usize], order: Order) -> SmallVec<[isize; 6]>;
```

### 5.5 负步长对内存访问范围的影响

**核心原则**：负步长表示沿该轴反向遍历，但不改变内存访问的总范围。

**偏移量计算**：

对于索引 `[i0, i1, ..., i_{n-1}]`，内存偏移量为：

```
offset = sum(stride[j] * i[j]) for all j
```

当 `stride[j] < 0` 时，随着 `i[j]` 增加，偏移量减少。

**内存范围边界**：

对于形状 `[s0, s1, ..., s_{n-1}]` 和步长 `[st0, st1, ..., st_{n-1}]`：

1. **最小偏移**：当负步长轴取最大索引，正步长轴取 0
2. **最大偏移**：当正步长轴取最大索引，负步长轴取 0

**示例**：

```
shape = [3, 4]
strides = [4, -1]  # 第一轴正步长，第二轴负步长（行反转）

索引 [0, 0]: offset = 0*4 + 0*(-1) = 0
索引 [0, 3]: offset = 0*4 + 3*(-1) = -3  ← 最小偏移
索引 [2, 0]: offset = 2*4 + 0*(-1) = 8   ← 最大偏移
索引 [2, 3]: offset = 2*4 + 3*(-1) = 5

内存访问范围: [-3, 8]，共 12 个元素
需要确保指针在 [base-3, base+8] 范围内有效
```

**实现注意事项**：

1. 带负步长的视图，其数据指针可能指向原数组的中间位置
2. 计算实际内存范围时需考虑所有步长的符号
3. `offset()` 方法返回相对于视图起始位置的偏移，可能是负数

---

## 6. 连续性检测算法

### 6.1 F-order 连续性检测

**定义**：F-order 连续意味着元素在内存中按 Fortran 顺序紧密排列，无间隙。

**条件**：

1. 按顺序累积的步长与理论 F-order 步长一致
2. 或：从第一个元素开始，按列优先顺序遍历时，物理地址连续递增

**伪代码**：

```
function is_f_contiguous(shape: [usize; N], strides: [isize; N]) -> bool:
    // 空数组或单元素数组始终连续
    if product(shape) <= 1:
        return true
    
    // 检查每个轴
    expected_stride = 1
    for i from 0 to N-1:
        // 跳过大小为 1 的轴（步长可以是任意值）
        if shape[i] != 1 and strides[i] != expected_stride:
            return false
        expected_stride = expected_stride * shape[i]
    
    return true
```

**示例**：

```
shape = [2, 3], strides = [1, 2]
i=0: shape[0]=2, stride[0]=1, expected=1 ✓, expected := 1*2=2
i=1: shape[1]=3, stride[1]=2, expected=2 ✓, expected := 2*3=6
结果: true (F-contiguous)

shape = [2, 3], strides = [3, 1]
i=0: shape[0]=2, stride[0]=3, expected=1 ✗
结果: false (not F-contiguous)
```

### 6.2 C-order 连续性检测

**定义**：C-order 连续意味着元素在内存中按 C 顺序紧密排列，无间隙。

**条件**：

1. 按逆序累积的步长与理论 C-order 步长一致
2. 或：从第一个元素开始，按行优先顺序遍历时，物理地址连续递增

**伪代码**：

```
function is_c_contiguous(shape: [usize; N], strides: [isize; N]) -> bool:
    // 空数组或单元素数组始终连续
    if product(shape) <= 1:
        return true
    
    // 检查每个轴（从后向前）
    expected_stride = 1
    for i from N-1 down to 0:
        // 跳过大小为 1 的轴
        if shape[i] != 1 and strides[i] != expected_stride:
            return false
        expected_stride = expected_stride * shape[i]
    
    return true
```

**示例**：

```
shape = [2, 3], strides = [3, 1]
i=1: shape[1]=3, stride[1]=1, expected=1 ✓, expected := 1*3=3
i=0: shape[0]=2, stride[0]=3, expected=3 ✓, expected := 3*2=6
结果: true (C-contiguous)

shape = [2, 3], strides = [1, 2]
i=1: shape[1]=3, stride[1]=2, expected=1 ✗
结果: false (not C-contiguous)
```

### 6.3 特殊情况处理

| 情况 | F-contiguous | C-contiguous | 说明 |
|------|:------------:|:------------:|------|
| 空数组 (size=0) | true | true | 无元素，视为连续 |
| 标量 (ndim=0) | true | true | 单元素 |
| 1D 数组 | true | true | 一维情况下两个方向等价 |
| 含 size=1 轴 | 检查其他轴 | 检查其他轴 | size=1 轴的步长不影响连续性 |

---

## 7. 对齐策略

### 7.1 默认对齐

| 属性 | 值 | 原因 |
|------|-----|------|
| 默认对齐 | 64 字节 | AVX-512 缓存行宽度 |
| 最小对齐 | `max(align_of::<A>(), 8)` | 元素自然对齐，至少 8 字节 |

### 7.2 小数组优化

**规则**：当 `元素数 × 元素大小 ≤ 对齐值` 时，允许降级到元素自然对齐。

**伪代码**：

```
function compute_alignment(elem_size: usize, elem_count: usize, requested: usize) -> usize:
    DEFAULT_ALIGN = 64
    natural_align = align_of::<A>()
    
    // 使用请求的对齐或默认值
    target_align = if requested > 0 then requested else DEFAULT_ALIGN
    
    // 小数组优化
    total_size = elem_size * elem_count
    if total_size <= target_align:
        return natural_align  // 降级到自然对齐
    
    // 确保对齐值是 2 的幂且 >= 自然对齐
    return max(target_align, natural_align)
```

**示例**：

| 元素类型 | 元素数 | 总大小 | 请求对齐 | 实际对齐 |
|----------|--------|--------|----------|----------|
| f64 | 4 | 32B | 64 | 8（降级）|
| f64 | 16 | 128B | 64 | 64 |
| f32 | 8 | 32B | 64 | 4（降级）|
| f32 | 32 | 128B | 64 | 64 |

### 7.3 视图对齐继承

**规则**：视图继承源数组的对齐状态，但切片后可能降级。

| 操作 | 对齐状态 | 原因 |
|------|----------|------|
| 创建视图 | 继承 | 起始地址不变 |
| 切片 `s![1.., ..]` | 可能降级 | 起始地址偏移后可能不对齐 |
| 转置 `t()` | 不变 | 起始地址不变 |
| 广播 | 不变 | 起始地址不变 |

**对齐检查**：

```
function check_alignment(ptr: *const A, align: usize) -> bool:
    addr = ptr as usize
    return (addr % align) == 0
```

### 7.4 ArcRepr make_mut 对齐

**规则**：`make_mut()` 触发写时复制时，新分配使用默认 64 字节对齐。

**场景**：

```rust
let shared: ArcTensor<f64, _> = ...;  // 引用计数可能 > 1
let exclusive = shared.make_mut();    // 若引用计数 > 1，触发复制
// 新分配的内存使用 64 字节对齐
```

### 7.5 对齐查询 API

```rust
impl Layout {
    /// 返回当前数组的实际对齐值（字节）
    /// 
    /// 注意：视图的对齐值可能低于默认值，取决于偏移量
    pub fn alignment(&self) -> usize;
    
    /// 检查数据指针是否对齐到指定值
    pub fn is_aligned_to(&self, align: usize) -> bool;
}
```

---

## 8. 填充 (Padding)

### 8.1 填充目的

**主维度填充**：在主维度（F-order 下为第一轴）的末尾添加额外空间，使得每列/行的起始地址对齐到缓存行边界。

**适用场景**：
- 大型矩阵运算
- 需要 SIMD 优化的操作
- 多线程并行访问（避免伪共享）

### 8.2 填充计算方法

**F-order 填充计算**：

```
function compute_padding(shape: [usize; N], elem_size: usize, align: usize) -> [usize; N]:
    padded_shape = copy(shape)
    
    // 仅对主维度（第一轴）填充
    leading_dim = shape[0]
    bytes_per_column = leading_dim * elem_size
    
    // 计算需要填充到对齐边界的元素数
    if bytes_per_column % align != 0:
        aligned_bytes = ceil(bytes_per_column / align) * align
        padded_shape[0] = aligned_bytes / elem_size
    
    return padded_shape
```

**示例**：

```
shape = [100, 50], elem_size = 8 (f64), align = 64
bytes_per_column = 100 * 8 = 800
800 % 64 = 32 ≠ 0
aligned_bytes = ceil(800/64) * 64 = 13 * 64 = 832
padded_shape[0] = 832 / 8 = 104
结果: padded_shape = [104, 50]，填充 4 个元素
```

### 8.3 填充区域零初始化保证

**规则**：填充区域必须初始化为零值，原因：

1. **确定性**：避免读取未初始化内存导致 UB
2. **调试友好**：意外访问填充区域时返回可预测值
3. **安全复制**：复制整个内存块时不会暴露敏感数据

**实现要求**：

```rust
/// 分配带填充的存储
/// 
/// # Safety
/// 填充区域必须初始化为零
fn allocate_padded<A>(
    shape: &[usize],
    order: Order,
    align: usize,
) -> (Vec<A>, Vec<usize>)
where
    A: Element,
{
    let padded_shape = compute_padding(shape, size_of::<A>(), align);
    let total_elems = product(&padded_shape);
    
    // 分配并零初始化
    let mut data = Vec::with_capacity(total_elems);
    data.resize(total_elems, A::zero());
    
    (data, padded_shape)
}
```

### 8.4 填充对步长的影响

填充后，步长基于填充后的形状计算：

```
原始 shape = [100, 50], strides = [1, 100]
填充后 shape = [104, 50], strides = [1, 104]

注意：stride[1] 从 100 变为 104
```

---

## 9. 标志位更新规则

### 9.1 更新时机总览

| 时机 | 触发条件 | 标志位更新方式 |
|------|----------|----------------|
| 创建 | `Tensor::zeros()`, `from_vec()` 等 | 全部重新计算 |
| 切片 | `slice()`, `slice_mut()` | 继承 + 重新计算连续性/对齐 |
| 转置 | `t()`, `permute_axes()` | 继承 + 重新计算连续性 |
| Reshape | `reshape()` | 重新计算全部 |
| 视图创建 | `view()`, `view_mut()` | 继承全部 |
| 广播 | `broadcast()` | 继承 + 设置零步长标志 |

### 9.2 创建时初始化

**伪代码**：

```
function init_flags_for_new_tensor(shape, strides, ptr, order, elem_size):
    flags = LayoutFlags::EMPTY
    
    // 连续性
    flags = flags.set_f_contiguous(is_f_contiguous(shape, strides))
    flags = flags.set_c_contiguous(is_c_contiguous(shape, strides))
    
    // 对齐
    flags = flags.set_aligned(check_alignment(ptr, 64))
    
    // 步长特性
    flags = flags.set_has_zero_stride(any(stride == 0 for stride in strides))
    flags = flags.set_has_neg_stride(any(stride < 0 for stride in strides))
    
    return flags
```

### 9.3 切片后更新

**伪代码**：

```
function update_flags_for_slice(source_flags, new_shape, new_strides, new_ptr):
    flags = source_flags
    
    // 重新计算连续性（切片可能破坏连续性）
    flags = flags.set_f_contiguous(is_f_contiguous(new_shape, new_strides))
    flags = flags.set_c_contiguous(is_c_contiguous(new_shape, new_strides))
    
    // 重新检查对齐（偏移可能破坏对齐）
    flags = flags.set_aligned(check_alignment(new_ptr, 64))
    
    // 重新检查步长特性
    flags = flags.set_has_zero_stride(any(s == 0 for s in new_strides))
    flags = flags.set_has_neg_stride(any(s < 0 for s in new_strides))
    
    return flags
```

### 9.4 转置后更新

**伪代码**：

```
function update_flags_for_transpose(source_flags, new_shape, new_strides):
    // 转置交换 F/C 连续性
    flags = LayoutFlags::EMPTY
    flags = flags.set_f_contiguous(source_flags.is_c_contiguous())
    flags = flags.set_c_contiguous(source_flags.is_f_contiguous())
    
    // 对齐不变（起始地址不变）
    flags = flags.set_aligned(source_flags.is_aligned())
    
    // 步长特性重新计算（步长顺序交换）
    flags = flags.set_has_zero_stride(any(s == 0 for s in new_strides))
    flags = flags.set_has_neg_stride(any(s < 0 for s in new_strides))
    
    return flags
```

### 9.5 Reshape 后更新

**规则**：Reshape 要求原数组连续，结果数组的连续性取决于新形状。

```
function update_flags_for_reshape(new_shape, new_strides, order, ptr):
    // 重新计算所有标志（与创建时相同）
    return init_flags_for_new_tensor(new_shape, new_strides, ptr, order, ...)
```

### 9.6 视图创建后更新

**规则**：视图继承源数组标志，不重新计算（起始地址不变）。

```
function update_flags_for_view(source_flags):
    return source_flags  // 直接继承
```

### 9.7 广播后更新

**伪代码**：

```
function update_flags_for_broadcast(source_flags, new_strides):
    flags = source_flags
    
    // 设置零步长标志（广播引入零步长）
    flags = flags.set_has_zero_stride(any(s == 0 for s in new_strides))
    
    // 广播可能破坏连续性
    // 如果任何非广播轴的步长不变，可能保持连续
    // 简化处理：重新计算
    // ...
    
    return flags
```

---

## 10. 布局查询 API

### 10.1 API 签名

```rust
impl Layout {
    /// 是否 F-order 连续
    /// 
    /// 复杂度: O(1)
    #[inline]
    pub fn is_f_contiguous(&self) -> bool {
        self.flags.is_f_contiguous()
    }
    
    /// 是否 C-order 连续
    /// 
    /// 复杂度: O(1)
    #[inline]
    pub fn is_c_contiguous(&self) -> bool {
        self.flags.is_c_contiguous()
    }
    
    /// 是否任一方向连续
    /// 
    /// 复杂度: O(1)
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.flags.is_contiguous()
    }
    
    /// 是否 64 字节对齐
    /// 
    /// 复杂度: O(1)
    #[inline]
    pub fn is_aligned(&self) -> bool {
        self.flags.is_aligned()
    }
    
    /// 是否存在零步长（广播维度）
    /// 
    /// 复杂度: O(1)
    #[inline]
    pub fn has_zero_stride(&self) -> bool {
        self.flags.has_zero_stride()
    }
    
    /// 是否存在负步长（反转维度）
    /// 
    /// 复杂度: O(1)
    #[inline]
    pub fn has_neg_stride(&self) -> bool {
        self.flags.has_neg_stride()
    }
    
    /// 返回完整布局标志
    /// 
    /// 复杂度: O(1)
    #[inline]
    pub fn layout_flags(&self) -> LayoutFlags {
        self.flags
    }
}
```

### 10.2 语义说明

| 方法 | 语义 | 典型用途 |
|------|------|----------|
| `is_f_contiguous()` | 内存按列优先紧密排列 | BLAS 调用判断 |
| `is_c_contiguous()` | 内存按行优先紧密排列 | 与 C 库互操作 |
| `is_contiguous()` | 任一方向连续 | 决定是否需要复制 |
| `is_aligned()` | 起始地址 64B 对齐 | SIMD 路径选择 |
| `has_zero_stride()` | 存在广播维度 | 优化迭代逻辑 |
| `has_neg_stride()` | 存在反转维度 | 确定内存访问方向 |
| `layout_flags()` | 获取完整标志 | 批量检查 |

---

## 11. 与其他模块的交互

### 11.1 与 storage 模块的交互

| 交互点 | 方向 | 说明 |
|--------|------|------|
| `Owned` 分配 | storage → layout | 分配时指定 Order，layout 计算步长 |
| 对齐信息 | storage → layout | storage 提供实际对齐值，layout 更新 ALIGNED 标志 |
| `View` 创建 | storage → layout | layout 继承源数组标志 |

**接口设计**：

```rust
// storage/owned.rs
impl<A> OwnedRepr<A> {
    pub fn allocate(
        shape: &[usize],
        order: Order,
        align: usize,
    ) -> Result<(Self, Layout), AllocError> {
        let strides = compute_strides(shape, order);
        let (ptr, actual_align) = allocate_aligned(product(shape), align)?;
        
        let layout = Layout::new(shape, &strides, ptr, actual_align);
        Ok((Self { ptr, len, cap }, layout))
    }
}
```

### 11.2 与 tensor 模块的交互

| 交互点 | 方向 | 说明 |
|--------|------|------|
| `TensorBase` 组成 | tensor 持有 layout | layout 作为 `TensorBase` 的字段 |
| 切片操作 | tensor → layout | 切片时调用 `layout.update_for_slice()` |
| Reshape | tensor → layout | reshape 时重新计算 layout |

**TensorBase 结构**：

```rust
// tensor/mod.rs
pub struct TensorBase<S, D> {
    storage: S,
    dim: D,           // shape
    strides: D,       // 有符号步长
    offset: usize,    // 数据起始偏移
    layout: Layout,   // 布局标志
}
```

### 11.3 与 simd 模块的交互

| 交互点 | 方向 | 说明 |
|--------|------|------|
| 路径选择 | simd ← layout | simd 查询 `is_aligned()` 和 `is_contiguous()` |
| 步长检查 | simd ← layout | simd 检查步长是否为 1（连续） |

**SIMD 路径选择**：

```rust
// simd/vector.rs
pub fn elementwise_add_simd<A>(
    lhs: &TensorView<A, D>,
    rhs: &TensorView<A, D>,
    out: &mut TensorViewMut<A, D>,
) where A: RealScalar {
    // 检查是否可以使用 SIMD 路径
    if lhs.is_aligned() && lhs.is_contiguous() 
       && rhs.is_contiguous() && out.is_contiguous() {
        // 使用对齐加载的 SIMD 实现
        simd_add_aligned(lhs, rhs, out);
    } else if lhs.is_contiguous() && rhs.is_contiguous() && out.is_contiguous() {
        // 使用非对齐加载的 SIMD 实现
        simd_add_unaligned(lhs, rhs, out);
    } else {
        // 回退到标量实现
        scalar_add(lhs, rhs, out);
    }
}
```

### 11.4 与 ffi 模块的交互

| 交互点 | 方向 | 说明 |
|--------|------|------|
| BLAS 兼容检查 | ffi ← layout | `is_blas_compatible()` 使用布局标志 |
| LDA 计算 | ffi ← layout | 从步长计算 leading dimension |

**BLAS 兼容性检查**：

```rust
// ffi.rs
impl<S, D> TensorBase<S, D> {
    /// 检查是否可直接传递给 BLAS
    pub fn is_blas_compatible(&self) -> bool {
        // BLAS 要求：连续（F 或 C）、正步长、无零步长
        self.is_contiguous() 
            && !self.has_zero_stride()
            && !self.has_neg_stride()
    }
    
    /// 返回 BLAS 布局标识
    pub fn blas_layout(&self) -> Option<BlasLayout> {
        if !self.is_blas_compatible() {
            return None;
        }
        if self.is_f_contiguous() {
            Some(BlasLayout::ColumnMajor)
        } else {
            Some(BlasLayout::RowMajor)
        }
    }
    
    /// 返回 leading dimension（F-order 下为 stride[0]）
    pub fn lda(&self) -> Option<isize>
    where
        D: Dimension,
    {
        if self.ndim() != 2 {
            return None;
        }
        Some(self.strides()[0])
    }
}
```

---

## 12. 实现任务分解

### 任务清单

| # | 任务 | 预估时间 | 依赖 | 产出 |
|---|------|----------|------|------|
| 1 | 实现 `LayoutFlags` 位定义和基础方法 | 10 min | 无 | `flags.rs` |
| 2 | 实现 `LayoutFlags` 所有查询/设置方法 | 10 min | T1 | `flags.rs` |
| 3 | 定义 `Order` 枚举和 `Layout` 结构体 | 10 min | T2 | `mod.rs` |
| 4 | 实现 F-order 步长计算算法 | 10 min | T3 | `strides.rs` |
| 5 | 实现 C-order 步长计算算法 | 10 min | T3 | `strides.rs` |
| 6 | 实现 F-order 连续性检测算法 | 10 min | T3 | `contiguous.rs` |
| 7 | 实现 C-order 连续性检测算法 | 10 min | T3 | `contiguous.rs` |
| 8 | 实现标志位更新规则（创建/切片/转置） | 15 min | T1-T7 | `mod.rs` |
| 9 | 实现布局查询 API 和单元测试 | 15 min | T8 | `mod.rs`, `tests/` |
| 10 | 实现对齐检查和填充计算 | 10 min | T3 | `strides.rs` |

### 任务依赖图

```
T1 ──→ T2 ──→ T3 ──┬──→ T4 ──┐
                   │         │
                   ├──→ T5 ──┼──→ T8 ──→ T9
                   │         │
                   ├──→ T6 ──┤
                   │         │
                   └──→ T7 ──┘
                             │
                   T10 ──────┘
```

### 并行执行建议

- **Wave 1**: T1（可独立开始）
- **Wave 2**: T2（依赖 T1）
- **Wave 3**: T3, T10（依赖 T2，可并行）
- **Wave 4**: T4, T5, T6, T7（依赖 T3，可并行）
- **Wave 5**: T8（依赖 T4-T7）
- **Wave 6**: T9（依赖 T8）

---

## 13. 设计决策记录

### D1: 为什么使用 u8 而不是 bitflags crate?

**决策**: 使用裸 `u8` 包装类型而非 `bitflags` crate。

**理由**:
1. **零依赖**: 符合项目最小依赖原则
2. **简单性**: 仅 5 个标志位，手写位操作足够清晰
3. **性能**: 编译器能充分优化简单的位操作
4. **no_std 兼容**: 不引入额外依赖

### D2: 为什么步长使用 isize 而不是 usize?

**决策**: 步长类型为 `isize`（有符号）。

**理由**:
1. **负步长**: 支持 `flip()` 等反转操作
2. **偏移计算**: 负步长时偏移量可能为负，需要统一的有符号类型
3. **零步长**: 有符号类型仍可表示零步长（广播）
4. **惯例一致**: 与 ndarray 库一致

### D3: 为什么默认 F-order?

**决策**: 默认使用 Fortran-order（列优先）。

**理由**:
1. **BLAS 兼容**: BLAS/LAPACK 使用列优先布局
2. **科学计算惯例**: Fortran 在科学计算领域历史悠久
3. **性能一致性**: 与主流线性代数库保持一致

### D4: 为什么 ALIGNED 标志基于 64 字节?

**决策**: `ALIGNED` 标志表示 64 字节对齐。

**理由**:
1. **AVX-512**: 512-bit = 64 字节，是当前最宽的 SIMD 寄存器
2. **缓存行**: 现代 CPU 缓存行通常为 64 字节
3. **通用性**: 64 字节对齐满足 SSE/AVX/AVX2/AVX-512 所有需求

### D5: 为什么不缓存"创建时布局顺序"?

**决策**: 不设置"创建时布局顺序"标志。

**理由**:
1. **可推导**: 创建时顺序可通过连续性推导
2. **简化**: 减少标志位数量，降低维护复杂度
3. **需求明确**: 需求文档明确删除此标志

### D6: 小数组为何降级对齐?

**决策**: 小数组允许降级到元素自然对齐。

**理由**:
1. **内存效率**: 小数组使用 64B 对齐浪费空间
2. **无性能影响**: 小数组无法利用 SIMD，对齐意义不大
3. **阈值合理**: 当 `size <= align` 时降级，避免过度分配

### D7: 填充区域为何零初始化?

**决策**: 填充区域必须零初始化。

**理由**:
1. **安全性**: 避免读取未初始化内存（UB）
2. **可预测性**: 调试时意外访问返回可预测值
3. **安全性**: 复制内存块不暴露敏感数据

---

## 附录 A: 布局标志快速参考

```
LayoutFlags (u8):
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ 7 │ 6 │ 5 │ 4 │ 3 │ 2 │ 1 │ 0 │
├───┼───┼───┼───┼───┼───┼───┼───┤
│ - │ - │ - │NEG│ZER│ALG│ C │ F │
└───┴───┴───┴───┴───┴───┴───┴───┘

F   = F_CONTIGUOUS    (0x01)
C   = C_CONTIGUOUS    (0x02)
ALG = ALIGNED         (0x04)
ZER = HAS_ZERO_STRIDE (0x08)
NEG = HAS_NEG_STRIDE  (0x10)
-   = 保留位
```

## 附录 B: 步长计算示例

### B.1 3D 数组 F-order

```
shape = [2, 3, 4]

F-order 步长计算:
stride[0] = 1
stride[1] = 1 * 2 = 2
stride[2] = 2 * 3 = 6

内存布局:
索引 [0,0,0] -> 偏移 0
索引 [1,0,0] -> 偏移 1
索引 [0,1,0] -> 偏移 2
索引 [1,1,0] -> 偏移 3
索引 [0,2,0] -> 偏移 4
索引 [1,2,0] -> 偏移 5
索引 [0,0,1] -> 偏移 6
...
```

### B.2 2D 数组转置后

```
原始: shape = [2, 3], strides = [1, 2] (F-order)
转置: shape = [3, 2], strides = [2, 1]

原始数据: [a, b, c, d, e, f]
         列0: a, b
         列1: c, d
         列2: e, f

转置后访问:
[0,0] -> 偏移 0 -> a
[0,1] -> 偏移 1 -> b  (但逻辑上这是转置后的第二列)
[1,0] -> 偏移 2 -> c
[1,1] -> 偏移 3 -> d
[2,0] -> 偏移 4 -> e
[2,1] -> 偏移 5 -> f
```

---

*本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。*
