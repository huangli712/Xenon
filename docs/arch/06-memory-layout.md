# 内存布局模块设计

> 文档编号: 06 | 模块: `src/layout/` | 阶段: Phase 2
> 前置文档: `02-dimension.md`, `05-storage.md`
> 需求参考: 需求说明书 §7

---

## 1. 模块定位

### 1.1 职责边界

| 职责 | 包含 | 不包含 |
|------|------|--------|
| 布局标志位 | `LayoutFlags: u8` 位域定义和操作 | 数据分配（由 `storage/` 提供） |
| 步长计算 | F-order 步长公式、合法性验证 | 元素访问（由 `tensor/` 提供） |
| 连续性检查 | F-连续检测算法 | 运算逻辑（由 `ops/` 提供） |
| 对齐检查 | 指针对齐状态查询 | 实际对齐分配（由 `storage/alloc.rs` 提供） |
| 负步长语义 | 切片/翻转时的负步长处理 | SIMD 路径选择（由 `simd/` 提供） |
| 零步长语义 | 广播维度的零步长标记 | 广播规则实现（由 `broadcast/` 提供） |

### 1.2 设计原则

| 原则 | 体现 |
|------|------|
| F-order only | Xenon 仅支持列优先布局，不支持 C-order |
| O(1) 布局查询 | 通过缓存布局标志位，将高频查询优化为常数时间 |
| 有符号步长 | `isize` 类型支持负步长（反转）和零步长（广播） |
| 零成本抽象 | 布局信息为纯元数据，不引入运行时开销 |

### 1.3 F-order 设计决策

> **重要**：Xenon 仅支持 F-order（列优先），不支持 C-order。
>
> **理由**：
> 1. **BLAS 兼容**：BLAS/LAPACK 使用列优先布局
> 2. **科学计算惯例**：Fortran 在科学计算领域历史悠久
> 3. **简化设计**：去除 C-order 减少分支和复杂度
> 4. **需求明确**：需求说明书 §7 明确"只支持列优先（F-order）布局"

---

## 2. 文件位置

```
src/layout/
├── mod.rs             # Layout 结构体定义，公开 API
├── flags.rs           # LayoutFlags: u8 位域定义和操作
├── strides.rs         # 步长计算、验证、负/零步长处理
└── contiguous.rs      # F-连续性检测算法
```

文件划分理由：步长计算、标志位操作、连续性检测各自独立且职责清晰，拆分后便于独立测试和维护。

---

## 3. 依赖关系

### 3.1 依赖图（ASCII）

```
src/layout/
├── crate::dimension     # Dimension trait, Ix0~Ix6, IxDyn
└── crate::storage       # RawStorage (仅 is_aligned 查询)
```

### 3.2 类型级依赖

| 来源模块 | 使用的类型/trait |
|----------|-----------------|
| `dimension` | `Dimension`, `Ix0`~`Ix6`, `IxDyn` |
| `storage` | `RawStorage::as_ptr()` (对齐检查) |
| `core` | `isize`, `usize` |

### 3.3 依赖方向声明

> **依赖方向：单向向上。** `layout/` 仅消费 `dimension` 和 `storage` 的查询接口，不被它们依赖。`tensor/`、`ops/`、`simd/` 等上层模块消费 layout 的类型和函数。

---

## 4. 公共 API 设计

### 4.1 LayoutFlags（u8 bitflags）

使用 `u8` 类型存储布局标志位，占用 1 字节：

```
LayoutFlags (u8):
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ 7 │ 6 │ 5 │ 4 │ 3 │ 2 │ 1 │ 0 │
├───┼───┼───┼───┼───┼───┼───┼───┤
│ - │ - │ - │NEG│ZER│ALG│ - │ F │
└───┴───┴───┴───┴───┴───┴───┴───┘

F   = F_CONTIGUOUS    (0b00001)  Fortran 连续
ALG = ALIGNED         (0b00100)  64 字节对齐
ZER = HAS_ZERO_STRIDE (0b01000)  包含零步长（广播）
NEG = HAS_NEG_STRIDE  (0b10000)  包含负步长（切片/翻转）
-   = 保留位
```

> **注意**：Xenon 不需要 `C_CONTIGUOUS` 标志位（不支持 C-order）。1D 和 0D 数组天然 F-连续。

```rust
/// 布局标志位集合。
///
/// 使用位域存储多个布局属性，支持 O(1) 查询。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LayoutFlags(u8);

impl LayoutFlags {
    /// 空标志
    pub const EMPTY: Self = Self(0b0000_0000);

    /// F-order 连续性标志
    pub const F_CONTIGUOUS: u8 = 0b0000_0001;  // 0x01

    /// SIMD 对齐标志（64 字节）
    pub const ALIGNED: u8 = 0b0000_0100;        // 0x04

    /// 零步长标志（广播维度）
    pub const HAS_ZERO_STRIDE: u8 = 0b0000_1000; // 0x08

    /// 负步长标志（反转维度）
    pub const HAS_NEG_STRIDE: u8 = 0b0001_0000;  // 0x10

    // === 查询方法 ===

    /// 是否 F-order 连续。
    #[inline]
    pub const fn is_f_contiguous(self) -> bool {
        (self.0 & Self::F_CONTIGUOUS) != 0
    }

    /// 是否 64 字节对齐。
    #[inline]
    pub const fn is_aligned(self) -> bool {
        (self.0 & Self::ALIGNED) != 0
    }

    /// 是否存在零步长。
    #[inline]
    pub const fn has_zero_stride(self) -> bool {
        (self.0 & Self::HAS_ZERO_STRIDE) != 0
    }

    /// 是否存在负步长。
    #[inline]
    pub const fn has_neg_stride(self) -> bool {
        (self.0 & Self::HAS_NEG_STRIDE) != 0
    }

    // === 设置方法 ===

    /// 设置 F-order 连续标志。
    #[inline]
    pub const fn set_f_contiguous(self, val: bool) -> Self {
        if val { Self(self.0 | Self::F_CONTIGUOUS) }
        else { Self(self.0 & !Self::F_CONTIGUOUS) }
    }

    /// 设置对齐标志。
    #[inline]
    pub const fn set_aligned(self, val: bool) -> Self {
        if val { Self(self.0 | Self::ALIGNED) }
        else { Self(self.0 & !Self::ALIGNED) }
    }

    /// 设置零步长标志。
    #[inline]
    pub const fn set_has_zero_stride(self, val: bool) -> Self {
        if val { Self(self.0 | Self::HAS_ZERO_STRIDE) }
        else { Self(self.0 & !Self::HAS_ZERO_STRIDE) }
    }

    /// 设置负步长标志。
    #[inline]
    pub const fn set_has_neg_stride(self, val: bool) -> Self {
        if val { Self(self.0 | Self::HAS_NEG_STRIDE) }
        else { Self(self.0 & !Self::HAS_NEG_STRIDE) }
    }
}
```

### 4.2 步长类型：isize

步长使用有符号整数 `isize` 存储：

```rust
/// 步长数组中每个元素表示沿该轴移动一个索引时，内存偏移量的变化（元素单位）。
///
/// - 正步长：正向遍历
/// - 负步长：反向遍历（如 flip 操作）
/// - 零步长：广播维度（重复同一元素）
```

### 4.3 F-order 步长计算

**算法**：

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
i=0: stride[0] = 1,    cumulative = 1 * 3 = 3
i=1: stride[1] = 3,    cumulative = 3 * 4 = 12
i=2: stride[2] = 12,   cumulative = 12 * 5 = 60

结果: strides = [1, 3, 12]
```

**API**：

```rust
/// 根据形状计算 F-order 连续布局的步长。
///
/// # Arguments
/// * `shape` - 各轴长度
///
/// # Returns
/// 步长数组（isize），长度与 shape 相同
pub fn compute_f_strides<D: Dimension>(shape: &D) -> D;
```

### 4.4 连续性检查算法

**F-连续条件**：

```
function is_f_contiguous(shape: [usize; N], strides: [isize; N]) -> bool:
    // 空数组或单元素始终连续
    if product(shape) <= 1:
        return true

    expected_stride = 1
    for i from 0 to N-1:
        // 跳过 size=1 的轴（步长可以是任意值）
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
结果: true (F-contiguous)

shape = [2, 3], strides = [3, 1]
i=0: shape[0]=2, stride[0]=3, expected=1 ✗
结果: false (not F-contiguous)
```

### 4.5 对齐检查

```rust
/// 检查指针是否满足对齐要求。
#[inline]
pub fn is_aligned_to(ptr: *const u8, align: usize) -> bool {
    (ptr as usize) % align == 0
}

/// 检查指针是否 64 字节对齐。
#[inline]
pub fn is_aligned(ptr: *const u8) -> bool {
    is_aligned_to(ptr, 64)
}
```

### 4.6 负步长语义

负步长表示沿该轴反向遍历。对于索引 `[i0, i1, ..., i_{n-1}]`，内存偏移量为：

```
offset = sum(stride[j] * i[j]) for all j
```

**示例**：

```
shape = [3, 4], strides = [4, -1]  // 行反转

索引 [0, 0]: offset = 0*4 + 0*(-1) = 0
索引 [0, 3]: offset = 0*4 + 3*(-1) = -3  ← 最小偏移
索引 [2, 0]: offset = 2*4 + 0*(-1) = 8   ← 最大偏移
```

带负步长的视图，其数据指针指向原数组的中间位置（非起始），需要通过 `offset` 字段调整。

### 4.7 零步长语义

零步长表示广播维度——该维度被扩展，但所有索引访问同一元素：

```
shape = [3, 4], strides = [1, 0]  // 第二维广播

索引 [0, 0] 和 [0, 1] 和 [0, 2] 和 [0, 3] 访问同一物理元素
```

### 4.8 Layout 结构体

```rust
/// 内存布局描述。
///
/// 包含布局标志位和数据偏移量。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    /// 布局标志位
    flags: LayoutFlags,
    /// 数据起始偏移量（元素单位）
    offset: usize,
}

impl Layout {
    /// 创建新布局。
    pub fn new(flags: LayoutFlags, offset: usize) -> Self {
        Self { flags, offset }
    }

    /// 从形状、步长和指针计算完整布局。
    pub fn compute<D: Dimension>(
        shape: &D,
        strides: &D,
        ptr: *const u8,
    ) -> Self;

    /// 是否 F-order 连续。
    #[inline]
    pub fn is_f_contiguous(&self) -> bool {
        self.flags.is_f_contiguous()
    }

    /// 是否 64 字节对齐。
    #[inline]
    pub fn is_aligned(&self) -> bool {
        self.flags.is_aligned()
    }

    /// 是否存在零步长。
    #[inline]
    pub fn has_zero_stride(&self) -> bool {
        self.flags.has_zero_stride()
    }

    /// 是否存在负步长。
    #[inline]
    pub fn has_neg_stride(&self) -> bool {
        self.flags.has_neg_stride()
    }

    /// 返回完整布局标志。
    #[inline]
    pub fn flags(&self) -> LayoutFlags {
        self.flags
    }

    /// 返回数据偏移量。
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }
}
```

### 4.9 Good/Bad 对比

```rust
// Good - F-order 步长计算
let strides = compute_f_strides(&shape);  // [1, 3, 12] for [3,4,5]

// Bad - 硬编码步长（不通用、易错）
let strides = [1, 3, 12];  // 仅对 [3,4,5] 有效
```

```rust
// Good - 使用 LayoutFlags 查询
if tensor.is_f_contiguous() && tensor.is_aligned() {
    // 使用 SIMD 加速路径
}

// Bad - 每次重新计算连续性
let contig = check_contiguity(tensor.shape(), tensor.strides());  // O(n) 重复计算
```

---

## 5. 内部实现设计

### 5.1 标志位计算算法

```
function compute_flags(shape, strides, ptr):
    flags = LayoutFlags::EMPTY

    // 1. 连续性
    flags = flags.set_f_contiguous(is_f_contiguous(shape, strides))

    // 2. 对齐
    flags = flags.set_aligned(is_aligned(ptr))

    // 3. 步长特性
    flags = flags.set_has_zero_stride(any(stride == 0 for stride in strides))
    flags = flags.set_has_neg_stride(any(stride < 0 for stride in strides))

    return flags
```

### 5.2 特殊情况处理

| 情况 | F-contiguous | 说明 |
|------|:------------:|------|
| 空数组 (size=0) | true | 无元素，视为连续 |
| 标量 (ndim=0) | true | 单元素 |
| 1D 数组 | true | 一维情况天然 F-连续 |
| 含 size=1 轴 | 检查其他轴 | size=1 轴的步长不影响连续性 |

### 5.3 标志位更新规则

| 操作 | 标志位更新方式 |
|------|----------------|
| 创建 | 全部重新计算 |
| 切片 | 继承 + 重新计算连续性/对齐 |
| 转置 | 继承 + 重新计算连续性（F/C 交换） |
| Reshape | 重新计算全部 |
| 视图创建 | 继承全部 |
| 广播 | 继承 + 设置零步长标志 |

### 5.4 安全性论证

Layout 模块不涉及 `unsafe` 操作。标志位计算基于 shape/strides 的只读查询，结果缓存在 `LayoutFlags` 中。

---

## 6. 与 Dimension 模块的接口

步长存储在 `D` 类型（与 shape 同类型）中，但实际步长值为 `isize`。Dimension trait 需提供：

```rust
trait Dimension {
    /// 返回步长切片（isize 类型）。
    fn strides_isize(&self) -> &[isize];

    /// 从 isize 切片构造步长。
    fn from_isize_strides(strides: &[isize]) -> Self;
}
```

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
  - 内容: `LayoutFlags(u8)` 定义，常量（F_CONTIGUOUS/ALIGNED/HAS_ZERO_STRIDE/HAS_NEG_STRIDE），查询/设置方法
  - 测试: `test_flags_set_clear`, `test_flags_default_empty`
  - 前置: T1
  - 预计: 10 min

### Wave 2: 核心算法

- [ ] **T3**: 实现 F-order 步长计算算法
  - 文件: `src/layout/strides.rs`
  - 内容: `compute_f_strides<D: Dimension>(shape: &D) -> D`
  - 测试: `test_f_strides_2d`, `test_f_strides_3d`, `test_f_strides_scalar`
  - 前置: T1
  - 预计: 10 min

- [ ] **T4**: 实现 F-连续性检测算法
  - 文件: `src/layout/contiguous.rs`
  - 内容: `is_f_contiguous<D: Dimension>(shape: &D, strides: &D) -> bool`
  - 测试: `test_f_contig_true`, `test_f_contig_false`, `test_f_contig_empty`, `test_f_contig_scalar`
  - 前置: T1
  - 预计: 10 min

- [ ] **T5**: 实现步长特性检测
  - 文件: `src/layout/strides.rs`
  - 内容: `has_zero_stride<D: Dimension>(strides: &D) -> bool`, `has_neg_stride<D: Dimension>(strides: &D) -> bool`
  - 测试: `test_zero_stride_detect`, `test_neg_stride_detect`
  - 前置: T1
  - 预计: 10 min

### Wave 3: Layout 结构体和集成

- [ ] **T6**: 定义 `Layout` 结构体和构造方法
  - 文件: `src/layout/mod.rs`
  - 内容: `Layout` 结构体，`new`/`compute` 方法，查询方法委托
  - 测试: `test_layout_new`, `test_layout_compute`
  - 前置: T2, T3, T4, T5
  - 预计: 15 min

- [ ] **T7**: 实现对齐检查
  - 文件: `src/layout/strides.rs`
  - 内容: `is_aligned_to`/`is_aligned` 函数
  - 测试: `test_alignment_check`
  - 前置: T1
  - 预计: 10 min

### Wave 4: 测试和文档

- [ ] **T8**: 集成测试和文档完善
  - 文件: `tests/layout.rs`
  - 内容: 综合测试套件：步长计算、连续性检查、负步长、零步长、对齐检查
  - 测试: 完整集成测试
  - 前置: T6, T7
  - 预计: 15 min

### 并行执行图

```
Wave 1: [T1] → [T2]
              ↓
Wave 2: [T3] [T4] [T5] [T7]   (可并行)
              ↓
Wave 3:       [T6]
              ↓
Wave 4:       [T8]
```

---

## 8. 测试计划

### 8.1 测试分类

| 类型 | 位置 | 目的 |
|------|------|------|
| 单元测试 | `#[cfg(test)] mod tests` | 验证单个算法 |
| 集成测试 | `tests/` | 验证跨维度类型交互 |
| 边界测试 | 集成测试中标注 | 空数组、标量、高维 |

### 8.2 单元测试清单

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_f_strides_1d` | 1D: `[5]` → strides `[1]` | 高 |
| `test_f_strides_2d` | 2D: `[3,4]` → strides `[1,3]` | 高 |
| `test_f_strides_3d` | 3D: `[2,3,4]` → strides `[1,2,6]` | 高 |
| `test_f_strides_scalar` | 0D: `()` → strides `()` | 高 |
| `test_f_contig_true` | F-连续数组判定 | 高 |
| `test_f_contig_false` | 非连续数组判定 | 高 |
| `test_f_contig_empty` | 空数组判定为连续 | 中 |
| `test_f_contig_size1_axis` | 含 size=1 轴的连续性判定 | 中 |
| `test_neg_stride_detect` | 负步长检测 | 高 |
| `test_zero_stride_detect` | 零步长检测 | 高 |
| `test_alignment_aligned` | 对齐指针对齐检查 | 高 |
| `test_alignment_unaligned` | 非对齐指针对齐检查 | 高 |
| `test_flags_all_set` | 所有标志位同时设置 | 中 |
| `test_flags_default_empty` | 默认值为空 | 高 |

### 8.3 边界测试场景

| 场景 | 预期行为 |
|------|----------|
| 空数组 `shape=[0, 3]` | F-连续为 true |
| 标量 `shape=()` | F-连续为 true，步长为空 |
| 1D 数组 `shape=[5]` | F-连续为 true |
| 高维 `shape=[2,2,2,2,2,2]` | F-连续，步长 `[1,2,4,8,16,32]` |

### 8.4 属性测试不变量

| 不变量 | 测试方法 |
|--------|----------|
| F-步长乘积 == 总元素数 | `product(strides[i] * shape[i]) == total` |
| 空数组/标量始终 F-连续 | 随机 0D/空 shape |
| `compute_f_strides` 后 `is_f_contiguous` 返回 true | 随机 shape |

---

## 9. 与其他模块的交互

### 9.1 与 Storage 模块

| 交互点 | 方向 | 说明 |
|--------|------|------|
| 对齐信息 | storage → layout | storage 提供指针地址，layout 计算对齐标志 |

### 9.2 与 Tensor 模块

| 交互点 | 方向 | 说明 |
|--------|------|------|
| `TensorBase` 组成 | tensor 持有 layout | `Layout` 作为 `TensorBase` 的计算字段 |
| 切片操作 | tensor → layout | 切片时调用 layout 更新连续性/对齐标志 |
| Reshape | tensor → layout | reshape 时重新计算步长和 layout |

### 9.3 与 SIMD 模块

| 交互点 | 方向 | 说明 |
|--------|------|------|
| 路径选择 | simd ← layout | simd 查询 `is_aligned()` 和 `is_f_contiguous()` |
| 步长检查 | simd ← layout | simd 检查步长是否为 1（连续） |

### 9.4 与 FFI 模块

| 交互点 | 方向 | 说明 |
|--------|------|------|
| BLAS 兼容检查 | ffi ← layout | 连续 + 正步长 + 无零步长 |
| LDA 计算 | ffi ← layout | 从步长计算 leading dimension |

---

## 10. 设计决策记录

### 决策 1：F-order only

| 属性 | 值 |
|------|-----|
| 决策 | Xenon 仅支持 F-order（列优先），不支持 C-order |
| 理由 | BLAS/LAPACK 兼容；科学计算惯例；简化设计减少分支；需求说明书 §7 明确要求 |
| 替代方案 | 同时支持 F/C order — 放弃，增加复杂度且当前版本不需要 |
| 替代方案 | 仅支持 C-order — 放弃，与 BLAS 不兼容 |

### 决策 2：步长使用 isize

| 属性 | 值 |
|------|-----|
| 决策 | 步长类型为 `isize`（有符号） |
| 理由 | 支持负步长（反转操作）；支持零步长（广播）；偏移计算统一有符号类型；与 ndarray 一致 |
| 替代方案 | `usize` — 放弃，无法表示负步长 |
| 替代方案 | `i64` — 放弃，与平台指针大小不一致 |

### 决策 3：64 字节对齐选择

| 属性 | 值 |
|------|-----|
| 决策 | `ALIGNED` 标志基于 64 字节对齐 |
| 理由 | AVX-512 = 512-bit = 64 字节；现代 CPU 缓存行 64 字节；满足所有 SIMD 指令集 |
| 替代方案 | 16 字节 — 放弃，AVX-512 未对齐 |
| 替代方案 | 动态对齐 — 放弃，增加运行时开销 |

### 决策 4：使用裸 u8 而非 bitflags crate

| 属性 | 值 |
|------|-----|
| 决策 | 使用裸 `u8` 包装类型而非 `bitflags` crate |
| 理由 | 零依赖（项目最小依赖原则）；仅 4 个标志位，手写位操作足够清晰；no_std 兼容 |
| 替代方案 | bitflags crate — 放弃，引入不必要依赖 |

---

## 11. 性能考量

| 方面 | 设计决策 |
|------|----------|
| 布局查询 | O(1)，直接读取缓存的 `LayoutFlags` |
| 步长计算 | O(ndim)，仅在创建/reshape 时计算 |
| 连续性检查 | O(ndim)，仅在切片/转置后重算 |
| 缓存友好性 | F-order 列优先访问与内存布局一致，顺序遍历时缓存命中率最优 |

**性能数据（参考）**：

| 操作 | 复杂度 | 说明 |
|------|--------|------|
| `is_f_contiguous()` | O(1) | 读取缓存标志 |
| `is_aligned()` | O(1) | 读取缓存标志 |
| `compute_f_strides()` | O(ndim) | ndim ≤ 6 时可忽略 |
| `is_f_contiguous()`（计算） | O(ndim) | ndim ≤ 6 时可忽略 |

**缓存友好性分析**：

| 遍历模式 | 缓存命中率 | 说明 |
|----------|-----------|------|
| F-order 顺序遍历 | ~95% | 顺序访问内存，预取器友好 |
| 非连续遍历（切片） | ~60% | 跳跃访问，缓存行利用率低 |
| 广播遍历（零步长） | ~95% | 重复访问同一缓存行 |

---

## 版本历史

| 版本 | 日期 |
|------|------|
| 1.0.0 | 2026-04-07 |

---

*本文档由 Xenon 维护。如有问题请提交 Issue 或 PR。*
