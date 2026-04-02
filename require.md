# Xenon 需求说明书 v20

> Rust 科学计算多维数组库
> 版本: v20 | 日期: 2026-03-31 | 许可: MIT | MSRV: Rust 1.85+

---

# 第一部分：项目定位

## 1. 项目概述

Xenon 是一个 Rust 多维数组（张量）库，定位为科学计算的数值基础设施，类似于 Python 生态中 NumPy ndarray 层的多维数组基础设施。

### 1.1 目标用户

| 用户类型 | 核心诉求 |
|----------|----------|
| 库开发者 | 稳定API、零开销抽象、指针级互操作 |
| 系统开发者 | no_std支持、底层内存控制、确定性数值行为、最小依赖 |
| 间接用户 | 性能、正确性、与Python经验的直觉一致性 |

### 1.2 核心原则

| 原则 | 说明 |
|------|------|
| 正确性优先 | 类型安全、内存安全、数值精度 |
| 性能敏感 | 零成本抽象、SIMD 优先路径、缓存友好的内存布局、避免不必要的分配 |
| 清晰抽象 | API 语义明确，无隐式行为 |
| 科学计算导向 | 行优先默认、SIMD 友好、BLAS 兼容内存布局 |
| 最小依赖 | 仅 rayon（可选）和 pulp（可选）。`InlineArray` 为库内自研类型（无外部依赖），用于 IxDyn 动态维度和广播形状推导的栈分配优化（≤ 8 维内联，> 8 维回退堆分配） |

### 1.3 工程约束

| 约束 | 要求 |
|------|------|
| Crate 结构 | 单 crate，遵循 SemVer |
| no_std 支持 | 支持 `no_std`（需 `alloc`），`std` 为默认 feature|
| Feature gates | `std`（默认）、`parallel`（rayon，隐式启用 `std`，因 rayon 依赖线程）、`simd`（pulp） |

---

## 2. 项目范围

### 2.1 范围内

- N 维数组的存储、构造、索引与高级索引、切片、形状操作、逐元素运算、归约、广播、集合操作、显式类型转换
- 矩阵-向量乘法、向量内积/外积、批量矩阵运算
- 原始指针 API，供上游库零开销集成
- 自定义复数类型（FFI友好，不依赖num-complex）
- 临时工作空间（对齐分配，供上游库使用）

### 2.2 范围外

- GEMM（矩阵-矩阵乘法）、矩阵分解、对角化等高级线性代数
- FFT、稀疏矩阵、自动微分
- BLAS/LAPACK 绑定（由上游库通过指针API集成）
- GPU 后端（当前版本不实现，但须预留扩展性）
- serde 序列化、arena 分配器
- 栈分配小数组（固定大小数组建议使用 nalgebra）

---

# 第二部分：核心架构

## 3. 维度系统

### 3.1 维度类型总览

支持 0 至 6 维静态维度 + 动态维度。所有维度类型须实现 `Dimension` trait（见 §3.2），并满足 `Debug + Clone + Eq + Hash + Send + Sync`。

| 类型 | 含义 | 内部表示 | 构造方式 |
|------|------|----------|----------|
| Ix0 | 0 维（标量） | `[usize; 0]`（零大小类型） | `Ix0`（无参，实现 `Default`） |
| Ix1 | 1 维（向量） | `[usize; 1]` | `Ix1(usize)` 或 `From<usize>` |
| Ix2 | 2 维（矩阵） | `[usize; 2]` | `Ix2(usize, usize)` 或 `From<(usize, usize)>` |
| Ix3 | 3 维 | `[usize; 3]` | `Ix3(usize, usize, usize)` 或 `From<(usize, usize, usize)>` |
| Ix4 | 4 维 | `[usize; 4]` | `From<[usize; 4]>` / `From<(usize, usize, usize, usize)>` |
| Ix5 | 5 维 | `[usize; 5]` | `From<[usize; 5]>` / `From<(usize, usize, usize, usize, usize)>` |
| Ix6 | 6 维 | `[usize; 6]` | `From<[usize; 6]>` / `From<(usize, usize, usize, usize, usize, usize)>` |
| IxDyn | 动态维度 | 内联数组 `InlineArray<usize, 8>`（≤ 8 维栈分配，> 8 维回退堆分配） | `From<&[usize]>` / `From<Vec<usize>>` |

**InlineArray<T, N> 说明**：库内自研小型内联数组类型（`pub(crate)`），无外部依赖。行为：N 个元素以内在栈上固定大小数组存储，超过 N 时自动退化为堆分配（`Vec<T>`）。用于 IxDyn 内部存储、广播形状推导（§16.2）、错误类型中的 ShapeData（§27）等需要动态长度但通常维度较低的场景。提供 `Deref<Target=[T]>`、`FromIterator`、`with_capacity`、`push` 等基本操作。当 N=8 时，覆盖 99.9% 科学计算场景（6 维以内零堆分配）。

### 3.2 Dimension trait

所有维度类型须实现 `Dimension` trait。**`Dimension` 为 sealed trait**：下游 crate 不可为自定义类型实现。实现者仅限库内预定义的 Ix0~Ix6 和 IxDyn。此限制确保维度类型的语义一致性——所有支持的维度类型均在库内掌控，`DimensionMut`（`slice_mut()`）等内部能力可安全限定为 `pub(crate)`，不会出现外部类型因缺少 `slice_mut()` 而在 reshape 等操作中产生不友好编译错误的情况。

| 关联类型/方法 | 签名 | 说明 |
|--------------|------|------|
| `type Pattern` | 关联类型 | 索引模式类型，用于 `Index` trait 实现 |
| `type Larger` | 关联类型：`Dimension` | 插入一个轴后的维度类型（Ix0 → Ix1, ..., Ix5 → Ix6, Ix6 → IxDyn, IxDyn → IxDyn）。所有维度类型均可升维，因此定义在 `Dimension` 而非独立 trait。`Larger` 约束为 `Dimension`（不含 `RemoveAxis`），以避免 `IxDyn::Larger = IxDyn` 导致的 coinductive cycle（`IxDyn: Dimension + RemoveAxis` → `IxDyn::Larger: RemoveAxis` → 无限递归）。需要 `Larger` 同时支持 `RemoveAxis` 的泛型代码，须在调用处显式添加 `where D::Larger: RemoveAxis` bound。升维→降维为常见操作链（如 `stack` 后再 `collapse`），推荐定义组合 trait 或 helper 函数封装此约束 |
| `NDIM` | 关联常量：`Option<usize>` | 编译时维度数。静态维度（Ix0~Ix6）返回 `Some(N)`，IxDyn 返回 `None`。用于 reshape 等操作在编译时校验目标维度匹配。若需要编译时维度数，使用 `D::NDIM.unwrap()`（静态维度）或回退到 `d.ndim()`（IxDyn） |
| `ndim()` | `&self -> usize` | 返回维度数。静态维度（Ix0~Ix6）时返回编译时常量；IxDyn 为普通函数 |
| `slice()` | `&self -> &[usize]` | 返回 shape 切片 |
| `slice_mut()` | `&mut self -> &mut [usize]` | 返回可变 shape 切片。**此方法不在公开 `Dimension` trait 中定义**，而通过 `pub(crate)` 的内部 sealed trait `DimensionMut` 提供（仅库内 reshape 等操作可调用），防止外部用户通过修改 shape 破坏 Tensor 不变量 |
| `size()` | `&self -> usize` | 返回总元素数（各轴乘积）。Ix0（空维度）返回 1（空乘积的数学约定）。**溢出时 wrapping**（debug 与 release 一致）。此方法仅用于非分配场景（如迭代计数、布局判断），分配场景须使用 `size_checked()`（见下方分配安全规则） |
| `size_checked()` | `&self -> Option<usize>` | 返回总元素数，溢出时返回 `None`。Ix0 返回 `Some(1)` |
| `size_wrapping()` | `&self -> usize` | 返回总元素数，溢出时 wrapping。仅在明确不需要用于内存分配的场景使用（如统计日志） |

**分配安全规则**（强制性约束）：所有涉及内存分配的路径（包括但不限于 Owned 构造、ArcRepr::make_mut 深拷贝、reshape/transpose 需要新缓冲区时）**必须**使用 `size_checked()` 而非 `size()`。当 `size_checked()` 返回 `None` 时，须返回 `ShapeError::Overflow` 错误（见 §27），不得 panic 或静默继续。`size()` 方法仅用于非分配场景（如迭代计数、布局判断），其 wrapping 行为在这些场景下不影响内存安全。

| `insert_axis()` | `&self, axis: usize -> Self::Larger` | 在指定位置插入长度为 1 的新轴，返回升维后的维度。axis 须在 `[0, self.ndim()]` 范围内，否则 panic。用于 `unsqueeze`、`stack` 等升维操作。IxDyn 实现涉及堆分配（Vec 中插入元素） |
| `default_strides()` | `&self -> Self` | 计算 C-order 默认步长，以 `usize` 存储于 `Self` 内部 |
| `default_strides_f()` | `&self -> Self` | 计算 F-order 默认步长，以 `usize` 存储于 `Self` 内部 |
| `into_dyn()` | `self -> IxDyn` | 将任意维度类型转换为动态维度 `IxDyn`，消费 self。静态维度（Ix0~Ix6）将内部数组数据拷贝到 `Vec<usize>`（堆分配，O(ndim)，ndim ≤ 6）；`IxDyn` 为零开销 move（直接返回 self，无 clone）。签名使用 `self` 而非 `&self`，使 IxDyn→IxDyn 路径无分配。此为静态→动态维度转换的唯一入口，用于需要运行时维度类型的场景（如动态 unsqueeze、跨维度类型操作）。转换总是成功（infallible）。反向转换（动态→静态）使用标准库 `TryFrom<IxDyn>` trait，失败时返回 `DimensionMismatch`（见 §3.5） |

**步长存储设计**：

shape 与 strides 共用同一维度类型 `D`，内部均以 `usize` 存储。步长在运算时按 `isize` 解释——因 `usize` 与 `isize` 内存布局等价（同为 pointer-sized integer、二进制补码，所有 Rust 支持的平台上 `size_of` 和 `align_of` 相等），`strides()` 返回 `&[isize]` 时通过显式转换函数 `stride_to_offset(n: usize, stride: usize) -> isize` 将 stride 解释为有符号偏移，无逐元素转换开销。负步长存储为 `usize` 的二进制补码表示（如 `-3isize` 存储为 `usize::MAX - 1`），按 `isize` 解释时自动恢复负值。`Debug` 实现须将 strides 以 `isize` 形式输出（而非原始 `usize`），以确保调试可读性。实现须包含以下编译时断言，在所有目标平台上验证此等价假设：

```rust
const _: () = assert!(size_of::<usize>() == size_of::<isize>());
const _: () = assert!(align_of::<usize>() == align_of::<isize>());
```

步长切片访问（`&[isize]`）由 `TensorBase` 的 `strides()` 方法提供（见 §8.3），而非在 `Dimension` trait 层面定义——因为步长的有符号解释仅在张量上下文中有意义。

**步长值域约束**：所有步长的绝对值须 ≤ `isize::MAX as usize`。此约束保证：1）usize→isize 显式转换后符号正确；2）任何以 usize 比较步长绝对值的代码不会因负步长的补码表示而产生错误判断。`TensorBase` 层面提供 `stride_isize(&self, axis: usize) -> isize` 方法，封装显式转换逻辑，避免散落的类型转换调用。

**关联类型映射**：

| 维度类型 D | D::Pattern | D::Larger |
|------------|------------|-----------|
| Ix0 | `()` | Ix1 |
| Ix1 | `usize` | Ix2 |
| Ix2 | `(usize, usize)` | Ix3 |
| Ix3 | `(usize, usize, usize)` | Ix4 |
| Ix4~Ix6 | `(usize, ..., usize)` | Ix(N+1)（Ix6 的 Larger 为 IxDyn） |
| IxDyn | `InlineArray<usize, 8>` | IxDyn |

### 3.3 RemoveAxis trait

`RemoveAxis` 是独立的 trait，仅由可降维的维度类型实现。**Ix0 不实现此 trait**，从编译时保证零维张量不会调用 `remove_axis`。`type Smaller` 定义在此 trait 而非 `Dimension`，使 Ix0 完全无需定义无意义的降维类型。

| 关联类型/方法 | 签名 | 说明 |
|--------------|------|------|
| `type Smaller` | 关联类型：`Dimension` | 移除一个轴后的维度类型 |
| `remove_axis()` | `&self, axis: usize -> Self::Smaller` | 移除指定轴，返回降维后的维度。IxDyn 实现涉及堆分配（Vec 中移除元素并返回新 Vec） |

**实现者与关联类型映射**（Ix0 不实现）：

| 维度类型 D | \<D as RemoveAxis\>::Smaller |
|------------|------------------------------|
| Ix1 | Ix0 |
| Ix2 | Ix1 |
| Ix3 | Ix2 |
| Ix4~Ix6 | Ix(N-1) |
| IxDyn | IxDyn |

### 3.4 IntoDimension trait

`IntoDimension` 是泛型转换 trait，将各种输入类型统一转换为 `Dimension` 实现者。`TensorBase<S, D>` 的构造函数、`reshape`、`into_dimension` 等方法通过此 trait 接受灵活的 shape 参数。

**Trait 定义**：

| 关联类型/方法 | 签名 | 说明 |
|--------------|------|------|
| `type Dim` | 关联类型：`Dimension` | 转换目标维度类型 |
| `into_dimension()` | `self -> Self::Dim` | 消费 self，返回目标维度类型实例 |

**实现者**：

| 输入类型 | `Self::Dim` | 说明 |
|----------|-------------|------|
| `()` | Ix0 | 空元组 → 0 维 |
| `usize` | Ix1 | 单个值 → 1 维 |
| `(usize, usize)` | Ix2 | 二元组 → 2 维 |
| `(usize, usize, usize)` | Ix3 | 三元组 → 3 维 |
| `(usize, usize, usize, usize)` | Ix4 | 四元组 → 4 维 |
| `(usize, usize, usize, usize, usize)` | Ix5 | 五元组 → 5 维 |
| `(usize, usize, usize, usize, usize, usize)` | Ix6 | 六元组 → 6 维 |
| `[usize; N]`（N = 0~6） | IxN | 数组 → 对应静态维度 |
| `Vec<usize>` | IxDyn | 动态向量 → 动态维度 |
| `IxDyn` | IxDyn | identity |
| `D` where `D: Dimension` | `D` | 所有维度类型实现 identity 转换 |

### 3.5 维度互转规则

**静态 → 动态（IxN → IxDyn）**

| 转换 | 行为 | 失败条件 |
|------|------|----------|
| Ix0 → IxDyn | 总是成功，生成 ndim=0 的 IxDyn（空 Vec） | 无 |
| Ix1~Ix6 → IxDyn | 总是成功，保留 shape 信息 | 无 |

**动态 → 静态（IxDyn → IxN）**

| 转换 | 成功条件 | 失败处理 |
|------|----------|----------|
| IxDyn → Ix0 | ndim == 0（Vec 长度为 0） | 返回 `DimensionMismatch` |
| IxDyn → Ix1~Ix6 | ndim == N | 返回 `DimensionMismatch` |

**维度互转 API**：

| 转换方向 | API | 所在位置 | 说明 |
|----------|-----|----------|------|
| 静态 → 动态 | `dim.into_dyn()` | `Dimension` trait 方法（见 §3.2） | Infallible，消费 self，返回 `IxDyn`。静态维度拷贝数组数据到 `Vec`（堆分配），`IxDyn` 为零开销 move |
| 动态 → 静态 | `IxN::try_from(dyn_dim)` | 标准库 `TryFrom<IxDyn>` impl | Fallible，返回 `Result<IxN, DimensionMismatch>`。各静态维度类型 `Ix0`~`Ix6` 均实现 `TryFrom<IxDyn>` |

**注意**：`IntoDimension` trait（§3.4）仅处理"输入参数→Dimension"的 identity 转换（如 `usize → Ix1`、`(usize,usize) → Ix2`），不负责跨类型维度转换。静态↔动态维度互转必须使用上述专用 API。

### 3.6 边界情况与设计约束

**Ix0 边界语义**

| 属性 | 值 |
|------|-----|
| shape | `&[]`（空切片） |
| ndim | 0 |
| 元素数 | 1 |
| 与 IxDyn 互转 | IxDyn.ndim() == 0 时可双向转换 |

**default_strides 边界行为**

| 边界情况 | 行为 |
|----------|------|
| Ix0 | 返回 Ix0（空，无步长可计算） |
| shape 含零维度（如 `[3, 0, 2]`） | 含零维度的数组同时标记为 F-contiguous 和 C-contiguous；对应轴步长值不影响实际寻址（因为该轴 size=0 永远不会被实际索引） |

**reshape 与维度类型约束**

- reshape 不改变维度类型（静态保持静态，动态保持动态）
- 显式转换维度类型使用 `into_dyn()`（静态→动态，见 §3.2）或 `IxN::try_from()`（动态→静态，见 §3.5）
- 静态维度 reshape 时目标维度数须匹配类型（如 Ix3 reshape 后仍为 3 维）
- Ix6 不可 `unsqueeze`（已达最大静态维度，且 reshape 不改变维度类型）。`Ix6::Larger = IxDyn` 表示类型系统能表达增长，但 `unsqueeze` 保持维度类型不变，因此 Ix6 无法 unsqueeze。需要插入轴时，须先 `into_dyn()` 转换为 `IxDyn` 再操作。实现层面通过 sealed sub-trait `UnsqueezeDim`（仅 Ix0~Ix5 + IxDyn 实现）在编译时拒绝，见 §17.8

---

## 4. 元素类型体系

### 4.1 类型体系

四层元素类型体系（上层继承下层，并额外提供能力）：

| 层次 | 类型 | Trait 约束 |
|------|------|------------|
| 基础层 | 整数、浮点、复数、bool、usize | `Element` |
| 数值层 | 整数、浮点、复数（不含 bool、usize） | `Numeric: Element`（在基础层之上支持四则运算） |
| 实数层 | f32, f64 | `RealScalar: Numeric`（在数值层之上提供数学函数） |
| 复数层 | Complex<f32>, Complex<f64> | `ComplexScalar: Numeric`（在数值层之上提供复数运算） |

**约束**：bool 和 usize 仅属于基础层，不实现 `Numeric`，不支持四则运算。不支持自动类型提升，类型转换须显式。

### 4.2 Sealed Trait 策略

四层 trait 全部 sealed，下游 crate 不可为自定义类型实现。
Element trait 为 sealed（防止外部实现），但提供 `unsafe impl Element` 的 escape hatch 供高级用户为自定义类型实现。**安全契约**（违反导致未定义行为）：(1) 类型的内存布局必须为 `#[repr(C)]` 或 `#[repr(transparent)]`，确保 FFI 兼容和按位复制安全；(2) 类型必须满足 `Copy + Clone + Send + Sync`，且 `clone()` 等价于按位复制（`memcpy`）；(3) `Default` 值必须为全零字节（与 `zeroed()` 分配兼容）；(4) 所有字节模式必须为合法值（无无效位模式，即 `from_bytes` 安全）。`unsafe impl Element` 通过 `unsafe trait private::Sealed` 实现——用户须同时 `unsafe impl private::Sealed`，明确承担上述安全契约的责任。详见§4.4。

| 设计决策 | 说明 |
|----------|------|
| sealed 范围 | Element、Numeric、RealScalar、ComplexScalar 全部 sealed |
| 实现方式 | 私有模块 `mod private { pub trait Sealed {} }` + 各 trait 继承 `private::Sealed` |
| sealed 实现者 | 仅库内预定义类型：i32/i64/f32/f64/bool/usize/Complex<f32>/Complex<f64>。其中 `usize` 仅实现 `Element`（用于 argmin/argmax 等索引操作的返回类型），不实现 `Numeric` |

**v1 整数类型限制说明**：当前版本不支持 u8/u16/u32/u64/i8/i16 等整数类型。主要原因：(1) 图像处理（u8）、信号处理（i8/i16）、位运算（u64）等场景需额外设计整数运算语义（如整数除零 panic vs wrapping、整数溢出策略）；(2) 整数类型需要 `Numeric` trait 的 `zero()`/`one()` 但语义与浮点有微妙差异（如整数除法截断行为）。后续版本计划扩展支持，届时需明确整数除零行为（当前 `Numeric` 要求 `Div<Output=Self>`，整数除零会 panic，此为接受的行为）。

**扩展性说明**：当前版本（v1）四层 trait 全部 sealed，下游 crate 不可为自定义类型（如定点数、对偶数、有理数等）实现。此限制确保数值语义的完整性和 trait 约束的一致性。未来版本计划提供以下扩展机制之一（具体方案待定）：(1) `Newtype` wrapper 允许包装已有数值类型；(2) `unsafe` 扩展 trait（类似 `unsafe trait TrustedLen`）允许高级用户承担正确性责任；(3) 公开 `Sealed` trait 配合文档化的一致性要求。
bool 的 sum() 返回 usize（count of true 值），等价于 count_true()。bool 不支持 prod()。usize 仅实现 `Element`（不实现 `Numeric`），但可作为 `sum()` 等归约操作的返回类型。

### 4.3 基础层（Element）

| 方法/常量 | 说明 |
|-----------|------|
| Copy + Clone | 值语义，可按位复制 |
| Default | 默认值（整数: 0, 浮点: 0.0, bool: false, 复数: (0, 0), usize: 0） |
| PartialEq | 相等比较 |
| Debug + Display | 格式化输出 |
| Send + Sync | 线程安全（用于并行迭代） |

### 4.4 数值层（Numeric）

继承Element，额外约束四则运算并提供单位元。仅数值类型（整数、浮点、复数）实现，bool不实现。

| 约束 | 说明 |
|------|------|
| zero() | 加法单位元（整数: 0, 浮点: 0.0, 复数: (0, 0)） |
| one() | 乘法单位元（整数: 1, 浮点: 1.0, 复数: (1, 0)） |
| signum() | 符号函数（整数/浮点: 正→1, 零→0, 负→-1；复数: z / |z|，零时返回零） |
| Add<Output=Self> | 加法 |
| Sub<Output=Self> | 减法 |
| Mul<Output=Self> | 乘法 |
| Div<Output=Self> | 除法 |
| Neg<Output=Self> | 负号 |

### 4.5 实数层（RealScalar）

在数值层之上额外提供：

**数学函数**

| 方法 | 说明 |
|------|------|
| abs() | 绝对值 |
| sqrt(), cbrt() | 平方根、立方根 |
| hypot(other) | `sqrt(self² + other²)` |
| ln(), log2(), log10() | 对数函数 |
| exp(), exp2() | 指数函数 |
| sin(), cos(), tan() | 三角函数 |
| asin(), acos(), atan(), atan2() | 反三角函数 |
| sinh(), cosh(), tanh() | 双曲函数 |
| floor(), ceil(), round() | 取整 |
| powi(), powf() | 幂运算 |

**常量与检测**

| 方法/常量 | 说明 |
|-----------|------|
| epsilon() | 机器精度 |
| min_positive() | 最小正规数 |
| max_value() | 最大有限值 |
| infinity(), neg_infinity() | 无穷大 |
| nan() | NaN 常量 |
| is_nan(), is_infinite(), is_finite() | 特殊值检测 |
| min(), max() | NaN 传播（任一参数为 NaN 则返回 NaN） |
| PartialOrd | 偏序比较（NaN 与任何值比较返回 false） |

### 4.6 复数层（ComplexScalar）

在数值层之上额外提供。

**关联类型**：

| 关联类型 | 约束 | 说明 |
|----------|------|------|
| `type Real` | `RealScalar` | 底层实数类型（`Complex<f64>::Real = f64`，`Complex<f32>::Real = f32`） |

**方法与常量**：

| 方法/常量 | 签名 | 说明 |
|-----------|------|------|
| re() | `&self -> Self::Real` | 实部 |
| im() | `&self -> Self::Real` | 虚部 |
| conj() | `&self -> Self` | 共轭 |
| norm() | `&self -> Self::Real` | 模（使用 hypot 避免溢出） |
| norm_sqr() | `&self -> Self::Real` | 模的平方（re² + im²）|
| arg() | `&self -> Self::Real` | 辐角（返回 (-π, π]） |
| exp() | `&self -> Self` | 复数指数 |
| ln() | `&self -> Self` | 复数对数 |
| sqrt() | `&self -> Self` | 复数平方根（主值） |
| from_polar(r, theta) | `Self::Real, Self::Real -> Self` | 极坐标构造 |
| i() | `-> Self` | 虚数单位 |

### 4.7 NaN/Inf 处理约定

| 场景 | 行为 |
|------|------|
| 归约（sum/prod）含 NaN | 结果为 NaN |
| min/max 含 NaN | 任一参数为 NaN 则返回 NaN（NaN 传播语义） |
| 排序/比较含 NaN | NaN 不参与排序，PartialOrd 返回 None |
| Inf 参与算术 | 遵循 IEEE 754 规则 |
| 0.0 / 0.0 | 返回 NaN |
| 1.0 / 0.0 | 返回 Inf |

---

## 5. 复数类型

### 5.1 类型定义

- 自定义 Complex<T> 类型，`#[repr(C)]` 布局，不依赖外部 crate
- 实现 Copy + Clone（当 T: Copy 时），值语义
- Default 值为 Complex { re: T::default(), im: T::default() }，即 Complex(0, 0)
- Numeric trait的 zero() → Complex(0, 0)，one() → Complex(1, 0)

### 5.2 格式化输出

| trait | 格式 | 示例 |
|-------|------|------|
| Debug | `Complex {{ re: {re:?}, im: {im:?} }}` | `Complex { re: 1.0, im: 2.0 }` |
| Display | `{re}{sign}{im}i`，虚部为负时 sign 为 `-`，否则为 `+`。| `"1+2i"`, `"1-2i"`, `"0+1i"`, `"3+0i"` |

### 5.3 类型转换

| 转换 | 行为 |
|------|------|
| `From<T> for Complex<T>` | 其中 T: Element，构造 Complex(re, 0) |
| `From<(T, T)> for Complex<T>` | 从元组构造 Complex(t.0, t.1) |
| `From<Complex<T>> for (T, T)` | 反向转换，返回 (re, im) |
| 实数 → 复数 | 须使用 `Complex::from(value)` 或 `Complex::new(value, 0)`（参见 §23.1） |
| 复数 → 实数 | 不允许隐式转换，须使用 `.re` 显式取实部 |

### 5.4 算术运算

**复数与复数运算**

| 操作 | 行为 |
|------|------|
| Complex + Complex | 逐分量加：`(re1+re2, im1+im2)` |
| Complex - Complex | 逐分量减：`(re1-re2, im1-im2)` |
| Complex * Complex | `(re1*re2 - im1*im2, re1*im2 + im1*re2)`，遵循 IEEE 754 逐分量行为 |
| Complex / Complex | `(a+bi)/(c+di)` 使用 Smith 方法避免中间结果溢出：若 `|d| ≤ |c|`，令 `r = d/c`，`denom = c + d*r`，结果为 `((a_re + a_im*r)/denom, (a_im - a_re*r)/denom)`；否则令 `r = c/d`，`denom = c*r + d`，结果为 `((a_re*r + a_im)/denom, (a_im*r - a_re)/denom)`。分母为零时返回 NaN + NaN\*i。此方法避免直接计算 `c²+d²` 导致的溢出（如 `c` 或 `d` 很大时 `c²` 超出浮点范围），同时保持数值精度。需 `T: RealScalar`（需要 `abs()`） |

> **SIMD 性能提示**：Smith 方法的条件分支在 SIMD 场景下可能引入性能回退。实现时应 benchmark Smith 除法与标准复数除法在目标平台上的性能差异后再做选择。

**复数与实数互操作**

| 操作 | 行为 |
|------|------|
| Complex + 实数 | 实数隐式提升为 Complex(r, 0.0)，结果为 Complex。仅限同精度（Complex\<f64\> + f64） |
| 实数 + Complex | 同上，交换律成立 |
| Complex * 实数 | 标量乘法，等价于 (re\*r, im\*r)。仅限同精度 |
| Complex / 实数 | 标量除法，等价于 (re/r, im/r)。仅限同精度 |
| 实数 / Complex | `a / (c+di) = a(c-di) / (c²+d²) = (ac/(c²+d²), -ad/(c²+d²))`。分母为零时遵循 IEEE 754。仅限同精度 |
| Complex 与整数 | 不支持隐式互操作，须先将整数转为浮点 |
| 跨精度（如 f32 + Complex\<f64\>） | 不支持隐式互操作，须先显式 cast 到同精度 |

**复合赋值与一元运算**

| 操作 | 行为 |
|------|------|
| Complex += Complex | 逐分量加后赋值 |
| Complex += 实数 | 同精度实数加后赋值 |
| Complex -= Complex | 逐分量减后赋值 |
| Complex -= 实数 | 同精度实数减后赋值 |
| Complex \*= Complex | 复数乘法后赋值 |
| Complex \*= 实数 | 同精度标量乘后赋值 |
| Complex /= Complex | 复数除法后赋值 |
| Complex /= 实数 | 同精度标量除后赋值 |
| -Complex | 一元取负，返回 Complex(-re, -im) |

### 5.5 相等与比较

| 属性 | 要求 |
|------|------|
| PartialEq | 逐分量比较（re == re && im == im），NaN != NaN |
| Eq | 不实现（因 NaN 破坏自反性） |
| PartialOrd | 不实现（复数无自然全序） |
| Ord | 不实现 |
| Hash | 不实现（因无 Eq，NaN 使 hash 语义不一致） |
| 近似相等 | 提供 `approx_eq(&self, other: &Self, epsilon: T) -> bool` 方法，逐分量判断：`|self.re - other.re| <= epsilon && |self.im - other.im| <= epsilon` |
| 全序比较（工具方法） | 提供 `total_cmp(&self, other: &Self) -> Ordering` 方法，字典序：先比较 re（`total_cmp`），再比较 im（`total_cmp`）。**不作为数学序关系**，仅供集合操作（如 §15.2 unique）内部排序使用 |

**设计说明**：`total_cmp` 不实现为 `Ord` trait，因为字典序不反映复数的数学性质。集合操作通过 `sort_by(\|a, b\| a.total_cmp(b))` 调用。

### 5.6 内存布局

| 属性 | 要求 |
|------|------|
| 内存布局 | `#[repr(C)]`，保证 [re, im] 连续排列 |
| 大小 | size_of::<Complex<T>>() == 2 * size_of::<T>() |
| 对齐 | align_of::<Complex<T>>() == align_of::<T>() |
| 与 C 互操作 | 内存布局兼容 C99 `_Complex`。FFI 仅通过指针传递安全（`*const`/`*mut`），不保证按值传递的 ABI 兼容性 |
| 数组布局 | Complex<f64> 数组与交错实虚 f64 数组内存等价 |
| SIMD 对齐 | Complex<f64> 数组建议 16 字节对齐，以支持 SSE2/NEON 向量化加载 |
| no_std 注意 | norm() 依赖 hypot，no_std 环境需依赖 libm |

---

## 6. 存储系统

### 6.1 分层 Storage trait

存储须区分三种访问级别：**只读**、**可写**、**拥有**（编译时保证）。采用分层 trait 设计：所有存储模式实现 `Storage`（只读），可写存储额外实现 `StorageMut`（可写）。Rust 编译器在泛型约束 `where S: StorageMut` 处自动拒绝只读存储（如 ViewRepr），实现编译时访问控制。

**`Storage` trait**（所有存储模式实现 — 只读访问）：

| 关联类型/方法 | 签名 | 说明 |
|--------------|------|------|
| `type Elem` | 关联类型 | 元素类型 |
| `type Device` | 关联类型：`Device` | 存储所在的计算设备（当前仅 `Cpu`） |
| `capacity()` | `&self -> usize` | 返回存储的缓冲区容量（含 padding，即物理槽位数）。注意：此值 ≥ 逻辑元素数（`TensorBase::len()`），差异来自 SIMD 对齐填充。两方法语义不同：`Storage::capacity()` 反映物理缓冲区大小，`TensorBase::len()` 反映逻辑元素数 |
| `as_ptr()` | `&self -> *const Self::Elem` | 底层数据指针（只读） |
| `manages_memory()` | `&self -> bool` | 存储是否管理内存生命周期（决定 drop 时是否释放内存）。Owned 和 ArcRepr 返回 true（drop 时触发内存回收：Owned 直接释放，ArcRepr 在引用计数归零时释放）；ViewRepr 和 ViewMutRepr 返回 false（不管理内存，drop 仅释放视图元数据）。注意：对 ArcRepr 返回 true 不意味着独占所有权，仅表示其参与内存管理（共享所有权） |

**`StorageMut` trait**（仅可写存储实现 — 在 `Storage` 之上添加可写能力）：

| 方法 | 签名 | 说明 |
|------|------|------|
| `as_mut_ptr()` | `&mut self -> *mut Self::Elem` | 底层数据指针（可写）。仅 Owned、ViewMutRepr 实现此 trait |

**设计说明**：

| 设计决策 | 理由 |
|----------|------|
| `Storage` 不包含 `as_mut_ptr()` | 确保只读存储（ViewRepr）在编译时无法获得可写指针，无需运行时检查 |
| `StorageMut: Storage` | 可写存储必然可读，子 trait 约束反映这一语义 |
| 泛型代码用 `where S: StorageMut` 要求可写 | 编译器在 monomorphization 阶段拒绝 ViewRepr 等只读存储 |
| ArcRepr 不实现 `StorageMut` | ArcRepr 的可写访问必须通过 `make_mut()` 方法（见 §6.3），该方法包含写时复制逻辑；直接暴露 `as_mut_ptr()` 会绕过 CoW 保护，导致共享数据被意外修改 |

### 6.2 四种存储模式

| 存储模式 | 拥有数据 | 可读 | 可写 | 克隆语义 | 分配方式 | 典型用途 |
|----------|---------|------|------|----------|----------|----------|
| Owned | 是 | 是 | 是 | 深拷贝 | 64 字节对齐堆分配 | 数组创建、运算结果 |
| ViewRepr | 否（借用） | 是 | 否 | 拷贝视图元数据（O(1)） | 无分配 | 切片、子数组只读访问 |
| ViewMutRepr | 否（独占借用） | 是 | 是 | 不可克隆（独占语义） | 无分配 | 原地修改子区域 |
| ArcRepr | 共享（Arc） | 是 | 通过 make_mut() 写时复制 | 浅拷贝（引用计数+1） | 写时按需分配 | 跨线程共享、延迟复制、函数参数传递 |

**Owned 分配机制**：

Owned 存储使用 `std::alloc` 系统调用保证 64 字节（或更大）对齐：

```rust
struct Owned<A> {
    ptr: NonNull<A>,       // 64-byte aligned (或用户指定的更大对齐)
    capacity: usize,       // 元素数（含 padding）
    alignment: usize,      // 实际对齐值（≥ 64）
    _marker: PhantomData<A>,
}
```

- **分配**：使用 `Layout::from_size_align(capacity * size_of::<A>(), alignment)` + `alloc::alloc()`。`alignment` 至少为 `max(64, align_of::<A>())`。**ZST 守卫**：当 `size_of::<A>() == 0` 时（零大小类型，如 `()`），`ptr` 设为 `NonNull::dangling()`，`capacity` 设为元素数（逻辑值），不调用 `alloc::alloc()`（零大小分配为未定义行为）。ZST tensor 的 `Drop` 同样跳过 `dealloc`。此路径不影响正常类型（`f32`/`f64`/`i32`/`i64`/`Complex` 均为非零大小）
- **释放**：使用相同 `Layout` 调用 `alloc::dealloc`（Drop 实现）
- **重新分配**：不使用 `realloc`（因标准 `realloc` 不保证对齐保持），而是手动分配新缓冲区 + 拷贝 + 释放旧缓冲区
- **容量计算**：`capacity` 为实际分配的元素槽位数（含 padding），与逻辑元素数 `size()` 不同

**各存储模式的 trait 实现关系**：

| 存储模式 | `Storage` | `StorageMut` | `capacity()` | `as_ptr()` | `as_mut_ptr()`（via StorageMut） | `manages_memory()` | `Clone` |
|----------|-----------|--------------|--------------|------------|----------------------------------|---------------------|---------|
| Owned | ✅ | ✅ | buffer 元素数（含 padding） | buffer 起始指针 | 可用 | true（独占） | 深拷贝（新分配） |
| ViewRepr | ✅ | ❌ 不实现 | 同源 Tensor 的 capacity | 指向源数据 | 不可用（编译拒绝 — ViewRepr 未实现 StorageMut） | false | O(1) 拷贝元数据 |
| ViewMutRepr | ✅ | ✅ | 同源 Tensor 的 capacity | 指向源数据 | 可用 | false | 不可 Clone |
| ArcRepr | ✅ | ❌ 不实现 | buffer 元素数（含 padding） | buffer 起始指针 | 不可用（编译拒绝 — 可写访问须通过 `make_mut()` 方法，见 §6.3） | true（共享） | 浅拷贝（Arc ref +1） |

### 6.3 ArcRepr 语义

**ArcRepr::make_mut() 语义：**

| 属性 | 行为 |
|------|------|
| 写时复制 | 若引用计数 > 1，深拷贝数据到新分配，原 Arc 引用计数减 1 |
| 原子性保证 | 引用计数使用 `AtomicUsize`，所有引用计数操作使用 `Ordering::AcqRel`（fetch_sub）/ `Ordering::Acquire`（load）。make_mut() 先以 `Acquire` order 读取引用计数，若 > 1 则**先分配新缓冲区并深拷贝数据，再替换内部指针**（旧 Arc 引用计数在指针替换/drop 时自然递减）。整个过程中引用计数仅在指针赋值/drop 时递减，不在拷贝之前递减，避免其他线程在拷贝期间误判独占。多线程并发 make_mut() 由 `&mut self` 保证句柄级别串行化，不会导致数据竞争或重复拷贝 |
| 返回值 | 返回 `&mut [A]`，独占访问保证无其他引用可读取数据 |
| 分配对齐 | 新分配使用**与原缓冲区相同的对齐值**（至少 64 字节），确保新缓冲区 capacity ≥ 原始 capacity，strides 保持有效 |
| 深拷贝 buffer 大小 | 新缓冲区容量（`capacity()`）须 ≥ 原 buffer 的 `capacity()`，确保 strides 仍然有效。具体而言：若原 buffer 含 padding（PADDED=true），深拷贝时复制原 buffer 的全部物理槽位（含 padding 区域），strides 和 PADDED 标志保持不变；若原 buffer 无 padding，深拷贝复制逻辑元素数对应的槽位，strides 和标志均保持不变。这意味着 make_mut() 不改变 TensorBase 层面的 shape/strides/flags，仅替换底层存储指针 |
| 可见部分优化 | **v2+ 功能**，当前版本始终克隆整个缓冲区（容量 ≥ 原 capacity，strides 安全）。理论上当视图仅引用底层缓冲区的小部分（< 50%）时，可仅克隆可见部分以减少内存拷贝（与 ndarray 的 OwnedArcRepr 策略一致）。但此优化在 padded 数组场景下存在安全性风险：若仅克隆可见部分，新缓冲区 capacity 小于原 buffer，导致 strides 中包含 padding 的偏移量超出新缓冲区范围（越界访问）。v2 实现时须确保：(1) 非 padded 数组的简单场景可安全优化；(2) padded 数组须正确处理 strides 和 PADDED 标志 |
| 性能提示 | 若引用计数为 1，make_mut() 直接返回 `&mut [A]`，无需拷贝（最优路径）；多线程场景下，建议使用 Arc::try_unwrap() 尝试获取独占所有权，或确保单写多读模式以减少写时复制开销 |

**ArcRepr 性能陷阱提示**：

| 场景 | 行为 | 建议 |
|------|------|------|
| 小 Tensor 频繁修改 | 每次修改触发 make_mut() 深拷贝，即使引用计数为 1（Arc 管理开销本身 > 数据拷贝） | 小 Tensor（<1KB）优先使用 `Owned`（`Tensor`），避免 ArcRepr |
| clone 后立即修改 | `arc_tensor.clone()` 增加 Arc 引用计数至 2，后续 `make_mut()` 必触发深拷贝 | 使用 `Arc::try_unwrap(arc_tensor)` 尝试直接获取 Owned，避免不必要的 Arc 开销 |
| 多线程并发写入 | 每个线程独立深拷贝，峰值内存 = 线程数 × buffer_size | 预先 `split_at()` 分割为独立 Owned Tensor 分发给各线程，避免运行时 CoW |

**ArcRepr 与视图的安全边界：**

ArcTensor 创建的视图（TensorView / TensorViewMut）的生命周期绑定到 `&ArcTensor` / `&mut ArcTensor` 自身，而非底层 Arc 数据。这意味着：

| 场景 | 行为 | 保证机制 |
|------|------|----------|
| 存在活跃 TensorView 时调用 make_mut() | 编译错误——view 持有 `&self` 借用，make_mut() 需要 `&mut self`，Rust 借用检查器拒绝 | 借用检查器（编译时） |
| view 已释放后调用 make_mut() | 正常执行，无冲突 | 借用检查器（编译时） |
| clone ArcTensor 后，原件调用 make_mut() | 正常执行——clone 仅增加 Arc 引用计数，不创建借用关系；make_mut() 触发深拷贝，两个 ArcTensor 各自独立 | Arc 写时复制语义 |
| 多线程各持有 ArcTensor clone，同时 make_mut() | 各线程独立深拷贝，无数据竞争——原子引用计数保证 | Arc 原子操作 |
| ArcTensor 的 view 传递到其他函数 | view 的生命周期不超过创建它的 `&ArcTensor` 借用；函数返回后 view 必须释放 | Rust 生命周期系统 |

**设计要点**：不提供从 ArcTensor 创建"脱离生命周期"的独立视图（即视图不持有 Arc clone）。这确保 make_mut() 与视图之间的互斥完全由编译器静态保证，无需运行时检查。

**ArcRepr 常见操作的成本模型**：

| 操作链 | 内存操作 | 实际分配次数 |
|--------|---------|-------------|
| `arc.clone()` | Arc 引用计数 +1 | 0（仅原子 increment） |
| `arc.clone(); arc.make_mut()` | 引用计数 +1 再深拷贝 buffer | 1（make_mut 触发） |
| `arc.clone().mapv_into(f)` | clone 引用计数 +1，mapv_into 消耗所有权（若引用计数为 1 则直接复用 buffer，否则深拷贝后逐元素应用 f） | 0 或 1（取决于 clone 后引用计数） |
| `Arc::try_unwrap(arc)` | 尝试获取独占所有权 | 0（成功时零拷贝，失败时返回 Err 原 Arc） |

**建议**：需要从 ArcTensor 执行变换时，优先使用 `Arc::try_unwrap()` 尝试获取 Owned 再操作；若失败再接受 make_mut 深拷贝开销。

**CowRepr<'a, A>** — Copy-on-Write 存储表示。**v2+ 功能**，当前版本不实现。设计方向：可以持有 owned 数据或 borrowed 视图，首次修改时按需克隆。适用于需要统一接受 owned/borrowed 输入的 API。CowRepr 的开销比 ArcRepr 更小（无 Arc 引用计数），适合单线程场景。v2 实现时须补充：§6.2 存储模式表条目、Storage/StorageMut trait 实现、类型别名（CowTensor）、Send/Sync 语义。

### 6.4 设备扩展性

Storage / StorageMut trait 预留 `type Device` 关联类型，当前版本仅支持 `Cpu`。

**`Device` trait 定义**：

| 项目 | 定义 | 说明 |
|------|------|------|
| Trait bound | `Device: Debug + Clone + Eq + Send + Sync` | 标记 trait，标识存储所在的计算设备 |
| Cpu | `#[derive(Debug, Clone, Eq, PartialEq, Hash)]` `struct Cpu;` | CPU 设备（默认） |
| 未来扩展 | — | 可扩展为 `Gpu<Cuda>` / `Gpu<Vulkan>` 等 |

**设计约束（当前版本须满足）**：

| 约束 | 说明 |
|------|------|
| 存储层与设备解耦 | `TensorBase<S, D>` 的 `S` 参数承载设备信息（通过 `S::Device` 关联类型），而非在 `TensorBase` 层面增加独立泛型参数。这保证 `TensorBase<S, D>` 签名不变即可支持新设备 |
| 迭代器与设备无关的接口分离 | 当前所有迭代器（`iter()` / `iter_mut()` 等）隐式假设数据在 Host 内存可解引用。GPU 存储不可在 Host 端直接解引用，因此 GPU 后端须提供独立的执行模型（如 kernel launch），不可复用现有迭代器 API |
| Shape / Stride / Layout 可复用 | 形状、步长、布局标志为纯元数据，与存储位置无关。GPU 后端可直接复用 `Shape<D>` 和 `LayoutFlags`，无需重新定义 |

### 6.5 禁止事项

- 不可将任何 GPU 相关类型或 trait 引入公开 API
- 不可在 `Storage` 或 `StorageMut` trait 中添加 `Gpu` 相关的关联类型或方法（除 `type Device` 占位外）
- 不可在 `Element` trait 中添加 GPU 语义的约束

---

## 7. 内存布局

### 7.1 布局规则

| 规则 | 要求 |
|------|------|
| 布局顺序 | 支持 F-order（列优先）和 C-order（行优先），默认 C-order |
| 步长类型 | 有符号类型，单位为元素个数（非字节），支持负步长 |
| 默认对齐 | 自有存储默认 64 字节对齐（AVX-512 缓存行） |
| 可配置对齐 | 构造时可指定对齐值（须为 2 的幂，≥ 元素自然对齐） |
| 填充轴 | F-order 填充 axis 0（第一轴），C-order 填充 last axis（最后一轴）。填充后步长（strides）反映实际物理内存布局（含填充量），详见 §7.6 |
| 填充量 | `M_padded = ⌈M × size_of::<A>() / alignment⌉`（向上取整到 alignment 字节边界），默认 alignment = 64 字节（AVX-512 缓存行），确保每列（F-order）或每行（C-order）起始地址对齐 |
| 填充初始值 | 填充区域零初始化。Padding bytes 必须在分配时零初始化 |

### 7.2 Order 枚举

内存布局顺序枚举，用于构造函数和操作中指定或查询布局方向。

| 属性 | 定义 |
|------|------|
| 类型 | `#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)] #[non_exhaustive] pub enum Order` |
| `Order::F` | 列优先（Fortran）顺序 |
| `Order::C` | 行优先（C）顺序 |
| 默认值 | 项目全局默认 C-order（§7.1）。各 API 中 `order: Option<Order>` 参数传 `None` 时使用默认 C-order |
| `#[non_exhaustive]` | 未来可扩展新变体不构成 breaking change |

**使用场景**：

| 场景 | API 示例 | 说明 |
|------|----------|------|
| 构造函数指定布局 | `zeros(shape, order: Option<Order>)` | `None` 时使用默认 C-order |
| 迭代顺序 | `iter_ordered(order: Order)` | 强制按指定逻辑顺序遍历 |
| 拼接/堆叠输出布局 | `cat(axis, &[...], order: Option<Order>)` | `None` 时默认 C-contiguous |

### 7.3 布局标志系统

采用 7 标志位设计（2 字节 u16，低 7 位有效，高 9 位保留，须为 0），缓存高频查询的派生状态：

| 标志 | 位 | 信息 | 理由 |
|------|----|------|------|
| F_CONTIGUOUS | bit 0 | F-order 连续性 | 核心属性，高频查询 |
| C_CONTIGUOUS | bit 1 | C-order 连续性 | 核心属性，高频查询 |
| ALIGNED | bit 2 | SIMD 对齐（64 字节） | SIMD 路径选择依赖 |
| HAS_ZERO_STRIDE | bit 3 | 零步长（缓存） | 存在步长为 0 的维度。两种来源：(1) 广播产生的 size-1 维度（逻辑上重复该元素）；(2) unsqueeze 插入的 size-1 轴（§17.8，步长设为 0 以保持连续性）。优化路径据此判断不可按简单步长递增遍历，但连续性标志不受影响。**空数组（任意 axis 的 shape[k]==0）时该标志为 false（无元素被访问，无实际步长语义），且 F_CONTIGUOUS 和 C_CONTIGUOUS 均为 true（与 §3.6 一致）** |
| HAS_NEG_STRIDE | bit 4 | 负步长（缓存） | 反转检测，避免 O(ndim) 遍历 |
| PADDED | bit 5 | 主维度填充 | 数组的主维度（F-order 的 axis 0，C-order 的 last axis）经过填充以对齐 SIMD 访问。PADDED 为 true 时，步长反映填充后的物理布局（strides[1] ≥ shape[0] for F-order），但严格连续性标志（F_CONTIGUOUS / C_CONTIGUOUS）仍按无填充的 shape 乘积判定。SIMD 路径应使用宽松连续性检查（`is_f_padded_contiguous()` / `is_c_padded_contiguous()`，见 §7.4） |
| OWNS_DATA | bit 6 | 数据所有权 | 标记此张量是否拥有其底层数据（而非视图/引用）。用于快速判断是否可就地修改，避免遍历存储类型层级 |
| F_PADDED_CONTIGUOUS | bit 7 | F-order 宽松连续 | F-order 宽松连续（允许主维度填充），由 `is_f_padded_contiguous()` 查询。仅在 PADDED=true 时可能为 true；PADDED=false 时与 F_CONTIGUOUS 一致。用于 SIMD 路径快速判断，避免 O(ndim) 遍历 |
| C_PADDED_CONTIGUOUS | bit 8 | C-order 宽松连续 | C-order 宽松连续（允许主维度填充），由 `is_c_padded_contiguous()` 查询。仅在 PADDED=true 时可能为 true；PADDED=false 时与 C_CONTIGUOUS 一致。用于 SIMD 路径快速判断，避免 O(ndim) 遍历 |
| — | bit 9-15 | 保留 | 须为 0，留作未来扩展 |

**组合语义：**
- `F_CONTIGUOUS | C_CONTIGUOUS`：标量、1D 数组、或至多一个轴 size > 1 的多维数组（两种方向的步长模式重合）
- `!F_CONTIGUOUS && !C_CONTIGUOUS`：非连续数组（切片/转置后常见）
- `PADDED && !F_CONTIGUOUS && !C_CONTIGUOUS`：填充数组（严格连续性为 false，但 SIMD 可通过宽松连续性路径处理，见 §7.4）

**更新时机：**
- 创建时：根据分配方式初始化全部标志
- 切片/转置/reshape：重新计算全部标志
- 视图创建：继承源数组标志，按需降级（如对齐）

### 7.4 布局查询方法

以下方法定义于 TensorBase（完整方法签名见 §8.3 布局查询），此处列出各方法的判定规则与复杂度：

| 方法 | 说明 | 复杂度 |
|------|------|--------|
| is_f_contiguous() | 是否 F-order 严格连续（步长严格等于 shape 乘积，不含填充） | O(1) |
| is_c_contiguous() | 是否 C-order 严格连续（步长严格等于 shape 乘积，不含填充） | O(1) |
| is_contiguous() | 任一方向严格连续即为 true | O(1) |
| is_f_padded_contiguous() | 是否 F-order 宽松连续（允许主维度填充）。判定规则：ndim≤1 时等价 `is_f_contiguous()`；ndim≥2 时要求 |strides[0]|==1、|strides[1]|≥shape[0]、|strides[i]|==|strides[i-1]|×shape[i-1]（i≥2 且 shape[i-1]>1） | O(1) |
| is_c_padded_contiguous() | 是否 C-order 宽松连续（允许主维度填充）。判定规则：ndim≤1 时等价 `is_c_contiguous()`；ndim≥2 时要求 |strides[ndim-1]|==1、|strides[ndim-2]|≥shape[ndim-1]、|strides[i]|==|strides[i+1]|×shape[i+1]（i≤ndim-3 且 shape[i+1]>1） | O(1) |
| is_padded_contiguous() | 任一方向宽松连续即为 true | O(1) |
| is_axis_contiguous(axis) | 指定轴是否连续（|strides[axis]|==1），用于判断 SIMD 是否可沿该轴处理 | O(1) |
| is_aligned() | 是否 SIMD 对齐（64 字节） | O(1) |
| has_zero_stride() | 是否存在零步长（广播维度） | O(1) |
| has_neg_stride() | 是否存在负步长（反转维度） | O(1) |
| is_padded() | 是否存在主维度填充（PADDED 标志） | O(1) |
| layout_flags() | 返回完整布局标志 | O(1) |

### 7.5 对齐策略

| 属性 | 要求 |
|------|------|
| 默认对齐 | 64 字节（AVX-512 缓存行） |
| ALIGNED 标志语义 | ALIGNED=true 表示数据起始地址严格满足 64 字节对齐。小数组降级后 ALIGNED 必须为 false（不满足 64 字节对齐），SIMD 路径不得依赖 ALIGNED 标志选择加载指令（见 §9.1.4） |
| 小数组优化 | 当 `元素数 × size_of::<A>() ≤ 64` 时，允许降级到 `max(align_of::<A>(), 8)` 字节对齐（即元素自然对齐，但至少 8 字节以保证基本 SIMD 安全）。降级后 ALIGNED 标志为 false。阈值 64 与默认对齐一致，避免小数组浪费内存。**设计意图**：阈值 64 字节 = 一条 AVX-512 缓存行宽度，确保小数组至少可被一条 SIMD 指令完整处理。对于 f32（4 字节），最多 16 个元素可降级，恰好覆盖 1 条 AVX-512 向量（16 × f32） |
| 对齐查询 | 提供 alignment() 方法返回当前数组的实际对齐值 |
| 视图对齐 | 视图继承源数组的对齐状态；切片后对齐可能降级（起始地址偏移），布局标志须反映实际对齐。对齐判定算法：连续视图切片（所有 stride 为正且依次递增）计算 `(base_ptr_addr + offset × size_of::<A>()) % 64 == 0`，结果决定 ALIGNED 标志值；非连续视图（任一轴 stride ≠ 期望值）ALIGNED 设为 false |
| ArcRepr 对齐 | make_mut() 触发复制时，新分配使用默认对齐（64 字节） |

### 7.6 填充语义

#### 7.6.1 填充轴与步长

| 布局 | 填充轴 | 步长效果 |
|------|--------|----------|
| F-order | axis 0（第一轴） | `strides[1] = M_padded ≥ shape[0]`；`strides[i] = strides[i-1] × shape[i-1]`（i ≥ 2 且 shape[i-1] > 1） |
| C-order | last axis（最后一轴） | `strides[ndim-2] = K_padded ≥ shape[ndim-1]`；`strides[i] = strides[i+1] × shape[i+1]`（i ≤ ndim-3 且 shape[i+1] > 1） |
| 0D/1D | 不适用 | 1D 数组无填充（单轴连续无需填充） |

**示例**：F-order 矩阵 `shape=[10, 5]`，f64 元素，M_padded=16：
- `strides = [1, 16]`（物理步长，含填充）
- 每列占用 16 × 8 = 128 字节（64 字节对齐）
- 严格 `is_f_contiguous()` = false（strides[1]=16 ≠ shape[0]=10）
- `is_f_padded_contiguous()` = true（strides[0]=1 ✓, strides[1]=16≥10 ✓）

#### 7.6.2 填充量计算

`M_padded = ⌈M × size_of::<A>() / alignment⌉ × (alignment / size_of::<A>())`（先将逻辑元素总字节向上取整到 alignment 字节边界，再转回元素数），默认 alignment = 64 字节。等价实现：`padded_bytes = ⌈M × size_of::<A>() / alignment⌉ × alignment`，`M_padded = padded_bytes / size_of::<A>()`。注意：`alignment` 须为 `size_of::<A>()` 的整数倍（由构造函数验证），保证 `padded_bytes / size_of::<A>()` 为整数。**溢出安全**：计算过程中所有中间步骤（`M × size_of::<A>()`、`padded_bytes`）须使用 checked 算术（`checked_mul`/`checked_add`），溢出时返回 `ShapeError::Overflow`。最终 `padded_bytes` 须 ≤ `isize::MAX`（与 §8.1 (2) stride 不变量一致，因为 stride 需要表示为 isize）。

| 元素类型 | M（逻辑） | alignment | M_padded（物理） | 每列/行字节 |
|----------|-----------|-----------|-----------------|-------------|
| f64 | 10 | 64 | 16 | 128 |
| f32 | 10 | 64 | 16 | 64 |
| f64 | 8 | 64 | 8（无需填充） | 64 |
| f64 | 5 | 64 | 8 | 64 |

#### 7.6.3 填充分配 API

| 方法 | 说明 |
|------|------|
| `with_padding(self, alignment: usize)` | 将已有 Tensor 转换为填充布局（需拷贝），返回新的 Owned Tensor，PADDED 标志设为 true |
| `zeros_padded(shape, alignment)` | 创建填充布局的零数组，PADDED 标志设为 true |
| 标准构造函数 | 默认不填充（无额外内存开销），PADDED 标志为 false |

#### 7.6.4 操作后填充行为

| 操作 | 填充保持 | 说明 |
|------|----------|------|
| 逐元素运算（+, -, *, /） | 不保持 | 输出使用标准无填充分配，PADDED = false |
| reshape | 不保持 | 连续路径保持连续但移除填充；非连续路径拷贝后同样无填充 |
| 切片 | 可能保持 | 若切片保留填充轴（如沿 axis≥2 切片），strides 仍含填充，PADDED 不变；若切片消除填充轴，PADDED = false |
| 转置 | 保持 | strides 转置后 PADDED 仍为 true（填充轴改变） |
| clone / to_owned | 保持 | 深拷贝保留填充布局 |
| 归约 | 不保持 | 输出为低维数组，PADDED = false |
| make_mut (ArcRepr) | 保持 | 深拷贝时新 buffer 容量 ≥ 原 capacity，strides 和 PADDED 标志保持不变（见 §6.3） |

#### 7.6.5 填充字节的值约束

| 规则 | 说明 |
|------|------|
| 初始化 | 填充区域零初始化 |
| 逐元素写操作 | 只写入逻辑元素范围，填充字节值变为**未定义**（不保证为零） |
| SIMD 读取 | SIMD 操作读取填充字节是安全的（无副作用，值可能非零但不会触发 UB） |
| SIMD 写入 | SIMD 写操作可能覆写填充字节（如尾部向量 store），覆写后值未定义 |
| 逻辑元素访问 | 填充字节**不得作为逻辑元素暴露给用户**（迭代器、视图、索引不包含填充区域） |
| SIMD tail 处理 | SIMD 操作须按**逻辑元素边界**处理尾部，不得将填充字节计入运算结果 |

#### 7.6.6 填充与连续性关系

| 概念 | 检查方法 | 适用场景 |
|------|----------|----------|
| 严格 F-contiguous | `is_f_contiguous()` | reshape 零拷贝判定（要求步长严格等于 shape 乘积） |
| 宽松 F-contiguous | `is_f_padded_contiguous()` | SIMD 路径选择（允许主维度步长 ≥ shape 乘积） |
| 严格 C-contiguous | `is_c_contiguous()` | reshape 零拷贝判定 |
| 宽松 C-contiguous | `is_c_padded_contiguous()` | SIMD 路径选择 |
| 逐轴连续 | `is_axis_contiguous(axis)` | 沿指定轴 SIMD 处理（\|strides[axis]\|==1 即可） |

---

**Padded 数组与 reshape 交互**：Padded 数组的 `is_f_contiguous()` / `is_c_contiguous()` 返回 `false`（仅 `is_f_padded_contiguous()` / `is_c_padded_contiguous()` 返回 `true`）。因此 reshape 操作对 padded 数组默认触发数据拷贝。这是有意的设计选择：padded 数组的主维度步长包含 padding 字节，零拷贝 reshape 需要重新计算 strides 以保持 padding 对齐，复杂度较高且容易出错。若需要在 padded 数组上执行零拷贝 reshape，需先通过 `unpad()` 去除 padding（此操作会拷贝数据），再执行 reshape。


## 8. 张量核心抽象

### 8.1 泛型设计

核心数据结构为双参数泛型 `TensorBase<S, D>`：
- S：存储模式
- D：维度类型

**TensorBase 内部组成** :

| 组件 | 说明 |
|------|------|
| storage: S | 底层数据存储（决定所有权与访问权限） |
| shape: D | 各轴长度 |
| strides: D | 各轴步长。与 shape 共用同一类型 `D`，内部以 `usize` 存储，运算时按 `isize` 解释（见 §3.2 步长存储说明） |
| offset: usize | 数据起始偏移量（单位为元素个数，非字节），始终 ≥ 0。与 strides 的有符号类型 `isize` 配合使用：元素地址计算为 `data_ptr() + Σ(index[k] * stride[k])`，其中 `data_ptr() = storage.as_ptr() + offset`（见下方指针语义说明），strides 可为负值（反转轴）。offset 使用 `usize` 而非 `isize` 是因为起始偏移在物理上不可能为负（数据存储从缓冲区头部开始），而 strides 需要 `isize` 以表达反转视图的负步长 |
| flags: u16 | 布局标志位（低 7 位有效，高 9 位保留为 0）。缓存高频查询的派生状态（F/C 连续性、对齐、零步长、负步长、填充），使布局查询方法达到 O(1) 复杂度（见 §7.3 标志定义与 §7.4 查询方法）。创建时根据分配方式初始化，切片/转置/reshape 时重新计算，视图创建时继承并按需降级 |

**指针语义分层**：TensorBase 中的指针存在两级语义，须严格区分：

| 层级 | 来源 | 含义 | 用途 |
|------|------|------|------|
| 缓冲区起始指针 | `storage.as_ptr()`（§6.1 Storage trait） | 分配的内存缓冲区起始地址，不含 offset | 内部实现：地址计算的基准点 |
| 数据起始指针 | `as_ptr()`（TensorBase 方法，见 §8.3 及 §25.1） | `storage.as_ptr() + offset`，即首个逻辑元素的地址 | 对外 API：用户和 FFI 通过此指针访问数据 |

元素地址计算公式：`as_ptr() + Σ(index[k] * stride[k])`，展开为 `storage.as_ptr() + offset + Σ(index[k] * stride[k])`。`as_mut_ptr()` 同理，返回 `storage.as_mut_ptr() + offset`。

**偏移量算术安全性**：元素地址计算 `offset + Σ(index[k] * stride[k])` 须始终落在 `[0, storage_capacity)` 范围内。实现须保证：

(1) **shape 不变量**：每个轴 `shape[k] ≤ isize::MAX as usize`（即 `usize::MAX / 2`），由 `TensorBase` 构造函数在入口处验证，违反时返回 `ShapeError::DimensionOverflow`（见 §27）。

(2) **stride 不变量**：每个轴的步长绝对值 `|stride[k]|` 不得超过 `isize::MAX`。构造函数须验证默认步长（F/C-order）的累积乘积不超出 `isize::MAX`——即对于所有 k，`product(shape[0..k]) ≤ isize::MAX`（F-order）或 `product(shape[(k+1)..ndim]) ≤ isize::MAX`（C-order）。违反时返回 `ShapeError::Overflow`。注意：仅约束单个 axis ≤ isize::MAX 不足以保证此不变量，因为 stride 是多轴累积乘积。

(3) **地址计算中间精度**：元素地址公式 `offset + Σ(index[k] * stride[k])` 中，单项 `index[k] * stride[k]` 的乘积在 64-bit 平台上可能超出 `isize` 范围（即使满足上述两个不变量，当 ndim ≥ 2 时累积乘积仍可能溢出）。**运行时热路径**（元素寻址）须使用 `isize` 中间精度进行乘法与累加，最终结果转为 `usize` 后验证 < 存储容量。**构造函数冷路径**（constructor-time overflow validation）可使用 `i128` 中间精度验证 shape×stride 不超出 `isize::MAX` 范围，避免构造函数中手动模拟 128-bit 比较的复杂度。构造函数验证仅在创建 tensor 时执行一次，不影响运行时性能。

(4) **边界访问验证**：最终偏移量 `offset + Σ(...)` 转为 `usize` 后须 < 存储容量，越界访问在 safe API 中返回错误或 panic（如 `IndexOutOfBounds`，见 §27），在 `unsafe` API（如 `get_unchecked`）中为调用方责任。

(5) 构造函数（包括 `from_raw_parts`）须验证：(a) shape × strides 不产生越界偏移；(b) `shape.size_checked()` 返回 `Some`（即总元素数不溢出 usize），违反时返回 `ShapeError::Overflow`。

**统一步长验证函数**：所有构造路径必须调用统一的 `validate_strides(shape, strides) -> Result<(), InvalidStrides>` 函数，验证：(1) stride >= 0 或符合负步长规则，(2) shape×stride 不超过 `isize::MAX`，(3) 零步长仅用于 size-1 或 size-0 维度。此函数为唯一权威的步长合法性检查点，确保所有构造路径（包括 `from_raw_parts`、`from_shape_vec`、视图构造等）的验证逻辑一致。

### 8.2 类型别名体系

**主类型别名**

| 别名 | 展开 | 说明 |
|------|------|------|
| Tensor<A, D> | TensorBase<Owned<A>, D> | 拥有数据的数组 |
| TensorView<'a, A, D> | TensorBase<ViewRepr<&'a A>, D> | 不可变视图 |
| TensorViewMut<'a, A, D> | TensorBase<ViewMutRepr<&'a mut A>, D> | 可变视图 |
| ArcTensor<A, D> | TensorBase<ArcRepr<A>, D> | 原子引用计数共享（内置写时复制） |

**视图生命周期说明**：`TensorView<'a, A, D>` 和 `TensorViewMut<'a, A, D>` 中的 `'a` 绑定到源数组（Owned/ArcRepr）的借用，而非元素本身的 lifetime。具体规则：(1) 从 `&Tensor<A, D>` 创建视图时，`'a` 等于该 `&Tensor` 的 lifetime；(2) 从 `&mut Tensor<A, D>` 创建可变视图时，`'a` 等于该 `&mut Tensor` 的 lifetime；(3) shape 和 strides 的数据由视图自身持有（按值存储），不依赖外部 lifetime——只有底层数据指针通过 `'a` 借用源数组。

**维度便捷别名**（以 Tensor 为例，其他存储模式同理）

| 别名 | 展开 | 说明 |
|------|------|------|
| Tensor0<A> | Tensor<A, Ix0> | 0 维标量 |
| Tensor1<A> | Tensor<A, Ix1> | 1 维向量 |
| Tensor2<A> | Tensor<A, Ix2> | 2 维矩阵 |
| Tensor3<A> | Tensor<A, Ix3> | 3 维 |
| Tensor4<A> | Tensor<A, Ix4> | 4 维 |
| Tensor5<A> | Tensor<A, Ix5> | 5 维 |
| Tensor6<A> | Tensor<A, Ix6> | 6 维 |
| TensorD<A> | Tensor<A, IxDyn> | 动态维度 |

### 8.3 基础查询方法

所有 TensorBase<S, D> 均提供以下查询方法（不区分存储模式）：

**维度与形状查询**

| 方法 | 签名 | 说明 |
|------|------|------|
| `len()` | `&self -> usize` | 返回总元素数（各轴长度乘积），委托到 `self.shape.size()`，溢出时 wrapping（与 `Dimension::size()` 一致，见 §3.2）。**契约**：对通过本库构造函数创建的 tensor，shape 乘积不溢出 usize（构造函数已验证 `size_checked()`，见 §8.1 (5)），`len()` 返回正确值；对通过 `from_raw_parts` 创建的 tensor，调用方须确保 shape 乘积不溢出。非分配场景可安全使用；分配场景须使用 `len_checked()` |
| `len_checked()` | `&self -> Option<usize>` | 返回总元素数，委托到 `self.shape.size_checked()`，溢出时返回 `None`。用于需要精确元素计数的场景（如 FFI 缓冲区大小传递、预分配工作空间） |
| `is_empty()` | `&self -> bool` | 等价于 `len() == 0` |
| `ndim()` | `&self -> usize` | 返回维度数，等价于 `shape().len()` |
| `shape()` | `&self -> &[usize]` | 返回各轴长度的切片 |
| `strides()` | `&self -> &[isize]` | 返回各轴步长的切片（元素单位，有符号）。内部以 `D` 类型存储（`usize`），返回时转为 `&[isize]` 切片（见 §3.2 步长存储说明） |
| `offset()` | `&self -> usize` | 返回数据起始偏移量（元素单位） |

**指针查询**（亦用于 FFI，详见 §25.1）

| 方法 | 签名 | 说明 |
|------|------|------|
| `as_ptr()` | `&self -> *const A` | 返回数据起始位置的不可变原始指针，即 `storage.as_ptr() + offset`（见 §8.1 指针语义分层） |
| `as_mut_ptr()` | `&mut self -> *mut A` | 返回数据起始位置的可变原始指针，即 `storage.as_mut_ptr() + offset`。须可写存储（`S: StorageMut`） |
| `as_ptr_unchecked()` | `unsafe &self -> *const A` | 不检查偏移量有效性的 `as_ptr()` 变体，用于性能敏感路径 |
| `capacity()` | `&self -> usize` | 返回物理缓冲区容量（含 padding，元素单位），委托到 `self.storage.capacity()`（§6.1）。FFI 调用方据此验证裸指针写入边界：写入范围须在 `[offset, offset + capacity)` 内（相对于 `storage.as_ptr()`）。注意：此值 ≥ `len()`，差异来自 SIMD 对齐填充 |

**布局查询**（基于 `flags` 字段的 O(1) 查询，标志定义见 §7.3）

| 方法 | 签名 | 说明 |
|------|------|------|
| `is_f_contiguous()` | `&self -> bool` | 是否 F-order 严格连续 |
| `is_c_contiguous()` | `&self -> bool` | 是否 C-order 严格连续 |
| `is_contiguous()` | `&self -> bool` | 任一方向严格连续 |
| `is_f_padded_contiguous()` | `&self -> bool` | 是否 F-order 宽松连续（允许填充） |
| `is_c_padded_contiguous()` | `&self -> bool` | 是否 C-order 宽松连续（允许填充） |
| `is_padded_contiguous()` | `&self -> bool` | 任一方向宽松连续 |
| `is_aligned()` | `&self -> bool` | 是否 SIMD 对齐（64 字节） |
| `has_zero_stride()` | `&self -> bool` | 是否存在零步长（广播维度） |
| `has_neg_stride()` | `&self -> bool` | 是否存在负步长（反转维度） |
| `is_padded()` | `&self -> bool` | 是否存在主维度填充 |
| `layout_flags()` | `&self -> u16` | 返回完整布局标志原始值 |

各布局查询方法的详细判定规则见 §7.4。

---

# 第三部分：计算模型

## 9. 计算后端

### 9.1 SIMD

#### 9.1.1 基本配置

| 属性 | 要求 |
|------|------|
| 启用方式 | 通过 feature gate `simd` 启用，默认关闭 |
| 实现方式 | 使用 pulp crate 抽象 SIMD 指令集，不直接使用 std::arch |

#### 9.1.2 支持的指令集

基于 pulp crate 的平台条件编译模型。各目标架构有独立的 `Arch` enum（均标记 `#[non_exhaustive]`），通过 `arch.dispatch()` 运行时分发到最优实现。**`Arch` 的变体为 tuple variant**（包含对应 SIMD 后端的结构体实例），不是简单枚举：

**x86_64 目标架构**（`#[cfg(target_arch = "x86_64")]`）：

| 变体 | Feature flag | 包含的指令集 | 说明 |
|------|-------------|-------------|------|
| `Arch::V4(V4)` | `x86-v4`（须显式启用，见下方说明） | AVX-512（F/BW/CD/DQ/VL），向下兼容 V3 | pulp `x86-v4` feature 默认不启用，且依赖 `bytemuck/avx512_simd` |
| `Arch::V3(V3)` | 默认启用 | AVX2 + FMA + BMI1/2 + LZCNT + SSE4.2 等（x86-v3 baseline） | pulp `x86-v3` feature 默认启用 |
| `Arch::Scalar` | — | 无 SIMD | 不支持 AVX2 的 x86 CPU（2012 年前）退化为标量 |

**注**：当对应 feature 未启用时变体不存在（如未启用 `x86-v4` 时 `Arch` 不含 `V4` 变体），`match` 时须使用 `_` 通配符处理（`#[non_exhaustive]` 强制要求）。

**aarch64 目标架构**（`#[cfg(target_arch = "aarch64")]`）：

| 变体 | Feature flag | 包含的指令集 | 说明 |
|------|-------------|-------------|------|
| `Arch::Neon(Neon)` | 默认启用 | NEON + 可选 FCMA | — |
| `Arch::Scalar` | — | 无 SIMD | — |

**wasm32 目标架构**（`#[cfg(target_arch = "wasm32")]`，pulp 内建支持）：

| 变体 | Feature flag | 包含的指令集 | 说明 |
|------|-------------|-------------|------|
| `Arch::Simd128(Simd128)` | 内建（`target_arch` 条件编译） | WASM simd128 | 无需额外 feature flag |
| `Arch::RelaxedSimd(RelaxedSimd)` | 内建（`target_arch` 条件编译） | WASM Relaxed SIMD | 同上 |
| `Arch::Scalar` | — | 无 SIMD | — |

**重要说明**：

| 项目 | 说明 |
|------|------|
| `#[non_exhaustive]` | 所有平台的 `Arch` enum 均标记 `#[non_exhaustive]`，`match` 时须使用 `_` 通配符。这保证 pulp 未来添加新变体时不破坏下游代码 |
| 无独立 V1/V2 路径 | pulp 源码中存在 V1/V2 类型定义，但不在 `Arch` dispatch 路径中。不支持 AVX2 的 x86 CPU 直接退化为 `Scalar` |
| AVX-512 配置 | 使用 AVX-512 须同时启用 Xenon 的 `simd` feature 和 pulp 的 `x86-v4` feature（通过 Cargo.toml 的 `pulp = { features = ["x86-v4"] }` 或 Xenon 提供的子 feature `simd-avx512`） |
| WASM SIMD | pulp 已具备 wasm32 条件编译支持（包含 Simd128 和 RelaxedSimd 两个 SIMD 变体）。当前版本未纳入 Xenon 的 SIMD 路径，后续版本可考虑支持 |
| dispatch 机制 | `Arch::new()` 运行时检测最优指令集，`arch.dispatch(WithSimd)` 通过 `WithSimd` trait + `NullaryFnOnce` bridge 调用闭包。`Arch` 为 `Copy` 类型，可在库初始化时缓存（见 §9.1.4） |
| Simd trait 操作覆盖 | pulp 的 `Simd` trait 提供：浮点/整数算术、复数（c32/c64）、比较、位运算、abs/sqrt/min/max/neg、内存操作（splat/load/store/gather/scatter）、mask 操作。**不提供超越函数**（sin/cos/exp/ln），这些操作回退标量路径（见 §9.3 注） |

#### 9.1.3 SIMD 适用操作

**SIMD 路径的通用前提条件**（适用于以下所有 SIMD-eligible 操作）：

| 条件 | 说明 |
|------|------|
| 宽松连续内存 | `is_padded_contiguous()` 为 true（见 §7.4） |
| **正步长方向** | 所有步长（stride）须为**正值**（对 size-1 维度步长可为 0）。负步长数组（如反向切片 `s![..;-1]`）即使满足 `is_padded_contiguous()` 也**不得进入 SIMD 路径**，须回退标量路径。理由：SIMD 连续加载（load/store）按地址递增方向迭代，负步长数组的物理内存是反向排列的，强制正向遍历会导致越界访问（UB）。参考：ndarray 通过 `is_layout_c`/`is_layout_f` 将 stride 转为 `isize` 后与正值比较，自动排除负步长（负值永远不等于正值） |
| 对齐要求 | 见 §9.1.4 |

| 操作类别 | 示例 | 额外 SIMD 要求 |
|----------|------|---------------|
| 逐元素一元 | abs, neg, sqrt, exp, ln | 无额外要求；满足通用前提即可 |
| 逐元素二元（浮点/复数） | add, sub, mul, div | 两操作数均满足通用前提，形状兼容，**且布局方向一致**（见下方说明）。任一条件不满足时回退标量路径 |
| 逐元素二元（整数） | add, sub, mul | 同上。**注**：pulp 的整数 SIMD `mul` 不支持 `u8`/`i8`（仅支持 u16/i16/u32/i32/u64/i64），`u8`/`i8` 乘法回退标量路径 |
| 逐元素二元（整数除法） | div（整数类型） | 不走 SIMD 路径，始终标量。理由：主流 SIMD 指令集（AVX2/SSE/NEON）不提供整数除法指令，pulp 亦不提供整数 SIMD 除法；即使连续内存也回退标量路径 |
| 归约（浮点/复数） | sum, prod, min, max | 满足通用前提即可；不满足时回退标量路径 |
| 归约（整数 sum/prod） | sum, prod | 不走 SIMD 路径，始终标量。理由：整数 checked 算术（§14.3）要求每步检测溢出，SIMD 指令集无对应操作 |
| 归约（整数 min/max） | min, max | 满足通用前提即可；不满足时回退标量路径 |
| 内积 | dot | 满足通用前提即可 |
| 比较 | eq, lt, gt（生成 mask） | 满足通用前提即可；不满足时回退标量路径 |

**bool 类型归约说明**：bool 类型：`sum()` 返回 `usize`（统计 true 的数量），`prod()` 返回 `bool`（全部为 true 则 true）。推荐使用专用的 `count_true()`/`count_false()`/`any()`/`all()` 方法。bool 归约可使用 SIMD 路径（AND/OR 归约，见 §14.3 all/any 表格）。

**二元 SIMD 的布局方向一致性要求**：

二元 SIMD 操作（如 `a + b`）要求两操作数不仅满足各自的通用前提，还须**布局方向一致**。具体规则：

| 布局组合 | SIMD 行为 | 说明 |
|----------|----------|------|
| 同为 C-padded（行优先宽松连续） | ✅ SIMD 路径 | 两操作数按行优先顺序同步迭代 |
| 同为 F-padded（列优先宽松连续） | ✅ SIMD 路径 | 两操作数按列优先顺序同步迭代 |
| C-padded + F-padded（混合） | ❌ 回退标量路径 | 两操作数的最内层迭代方向不一致：C-padded 沿行连续、F-padded 沿列连续。SIMD 按单一方向迭代无法同时匹配两者的物理元素顺序，会导致错误结果 |
| 任一为非宽松连续 | ❌ 回退标量路径 | 通用前提不满足 |

实现须在二元 SIMD 入口处检查 `layout_direction_compatible(a, b)`（返回 true 当且仅当两操作数同为 C-padded 或同为 F-padded），不满足时回退标量路径。

**注 — 逐轴 SIMD 优化**：上述条件为 SIMD 路径的充分条件。实现可进一步支持逐轴 SIMD（沿目标轴 `is_axis_contiguous(axis)` 为 true 即可，不需要整个数组连续），但当前版本不做强制要求。逐轴 SIMD 允许非连续数组（如转置视图）沿连续轴使用 SIMD 处理。

**注 — 广播操作数的 SIMD 策略**：当二元运算的操作数之一为广播视图（`has_zero_stride() == true`）时，不满足上述宽松连续条件。但广播视图有专用 SIMD 优化路径：单轴广播使用 SIMD broadcast 指令（如 `_mm256_set1_ps`）将标量值加载到所有 lane 后与连续操作数进行向量运算。详见 §16.7。

#### 9.1.4 SIMD 对齐与回退

| 属性 | 要求 |
|------|------|
| 对齐加载判定 | **不得依赖 ALIGNED 布局标志**选择加载指令（小数组降级后 ALIGNED=false 但地址可能恰好对齐，反之亦然）。须在运行时检查数据起始地址 `ptr as usize % 64 == 0`，对齐时使用对齐加载（如 `_mm256_load_ps`），否则使用非对齐加载（如 `_mm256_loadu_ps`）。现代 CPU 上非对齐加载性能已接近对齐加载，对齐加载仅为保守优化 |
| 非连续数据 | 不满足 `is_padded_contiguous()` 时回退到标量路径；逐轴 SIMD 优化下，`is_axis_contiguous(axis)` 不满足时回退标量路径 |
| 负步长数据 | 即使满足 `is_padded_contiguous()`，若任一轴步长为负值，须回退标量路径。负步长表示逻辑方向与物理内存方向相反，SIMD 连续加载/store 无法正确处理反向遍历。实现须在 SIMD 入口处检查 `all_strides_positive_or_zero()`（所有步长 > 0，size-1 维度步长允许为 0） |
| 二元布局不兼容 | 二元操作的两操作数布局方向不一致（C-padded 混合 F-padded）时回退标量路径，见 §9.1.3 |
| 填充字节处理 | 填充数组的 SIMD 加载可能包含填充区域的字节（见 §7.6.5），这是安全的：读取无副作用；写入可能覆写填充字节（值变为未定义），但不影响逻辑元素 |
| 尾部处理 | 元素数非 SIMD 宽度整数倍时，尾部使用标量处理，**按逻辑元素边界**处理（不得将填充字节计入运算结果） |
| 运行时检测 | 使用 pulp 的 `Arch::new()` 检测最佳指令集，通过 `arch.dispatch()` 分发到对应实现；`Arch` 为 `Copy` 类型，可在库初始化时缓存一次供后续复用，避免重复检测开销 |

### 9.2 并行

#### 9.2.1 基本配置

| 属性 | 要求 |
|------|------|
| 启用方式 | 通过 feature gate `parallel` 启用，默认关闭 |
| 实现方式 | 使用 rayon crate 实现数据并行 |

#### 9.2.2 并行策略

| 属性 | 要求 |
|------|------|
| 并行阈值 | 默认数据量（`元素数 × size_of::<A>()`）≥ 256KB 时启用并行，可通过全局配置或单次调用覆盖。基于内存大小而非固定元素数，确保不同元素类型（f32/f64/Complex<f64>等）的调度开销与计算收益比例一致 |
| 支持并行的操作 | 逐元素运算、归约、map/mapv、zip 迭代 |
| 不支持并行的操作 | 矩阵乘法（由 BLAS 内部管理线程）、单元素操作、小数组操作 |
| 分块策略 | 按连续内存块分割，每块不小于 4K 元素。**Cache line 对齐**：分割边界须对齐到 64 字节（主流 CPU cache line 大小），避免相邻线程块共享同一 cache line 导致 false sharing。对于 padded 数组，分割须在 padding 边界对齐（即分割点须为逻辑元素边界），确保 SIMD 写入 padding 字节时不与其他线程的 cache line 交叉 |
| 嵌套并行 | 禁止嵌套并行（内层操作强制单线程），避免线程池饥饿。**强制机制**：使用 `thread_local!` 标记并行上下文，并行操作入口处检查标记——若已在并行上下文中，内层操作强制走单线程路径（不 spawn 新任务）。实现示例：`thread_local! { static IN_PARALLEL: Cell<bool> = Cell::new(false); }`，并行入口设置 `IN_PARALLEL.set(true)`，内层检查 `if IN_PARALLEL.get() { return scalar_path(); }` |
| 线程数 | 默认使用 rayon 全局线程池，可通过自定义线程池覆盖 |
| 浮点归约算法 | 并行浮点归约（sum/mean/var/std）须使用 **Kahan-Babuška-Neumaier** 补偿求和算法。具体策略：单线程路径使用 Neumaier 补偿求和（O(1) 额外空间，适合顺序累加）；并行路径各线程先独立执行 Neumaier 累加，最后汇总阶段使用 **Pairwise 递归求和**合并各线程的部分和（保证不同分块策略下结果的精度上界可控）。**不使用朴素 Kahan**（精度不如 Neumaier）。单线程浮点归约同样使用 Neumaier 补偿求和。整数归约不适用（使用 checked 算术） |

### 9.3 性能分层

运行时根据数据规模、内存布局和步长方向自动选择执行路径。以下表格按**优先级从高到低**排列（首个匹配的行即为执行路径）：

| 优先级 | 条件 | 执行路径 |
|--------|------|----------|
| 1 | 元素数 < SIMD 宽度 | 标量路径 |
| 2 | 元素数 ≥ SIMD 宽度 且 宽松连续内存（`is_padded_contiguous()`）且 **所有步长为正** 且 simd 启用 且 操作有 SIMD 实现（见注） | SIMD 路径（单线程） |
| 3 | 元素数 ≥ 并行阈值 且 parallel 启用 且 宽松连续内存（`is_padded_contiguous()`）且 **所有步长为正** 且 simd 启用 且 操作有 SIMD 实现 | SIMD + 并行路径（每线程内部使用 SIMD） |
| 4 | 元素数 ≥ 并行阈值 且 parallel 启用 且 非宽松连续内存 | 标量 + 并行路径（每线程标量迭代，按块分割非连续数据。宽松连续不是并行路径的前提条件——非连续数据也可并行，只是每线程内部不走 SIMD） |
| 5 | 元素数 ≥ 并行阈值 且 parallel 启用 且 simd 未启用 | 标量 + 并行路径 |
| 6 | 元素数 ≥ 并行阈值 且 parallel 启用 且 宽松连续内存 但 **存在负步长**（满足 padded contiguous 但步长方向不对） | 标量 + 并行路径（SIMD 条件不满足，回退标量但可并行） |
| 7 | 宽松连续内存 且 **所有步长为正** 且 simd 未启用 且 元素数 < 并行阈值（或 parallel 未启用） | 标量路径（连续内存但 SIMD 未启用，单线程标量迭代） |
| 8 | 非宽松连续内存 或 **存在负步长** 且 元素数 < 并行阈值（或 parallel 未启用） | 标量路径（单线程，步长跳跃遍历） |

**条件依赖关系图**：

```
is_padded_contiguous() && all_strides_positive() && simd_enabled && has_simd_impl
    ├─ YES && element_count >= parallel_threshold && parallel_enabled → 路径 3 (SIMD+并行)
    ├─ YES && element_count < parallel_threshold || !parallel_enabled → 路径 2 (SIMD)
    └─ NO (任一条件不满足)
        ├─ element_count >= parallel_threshold && parallel_enabled → 路径 4/5/6 (标量+并行)
        └─ element_count < parallel_threshold || !parallel_enabled → 路径 7/8 (标量)
```

**注 — 无 SIMD 实现的操作（即使连续内存也走标量路径）**：

| 操作 | 原因 |
|------|------|
| 整数除法（i32/i64/u32/u64 的 div） | 主流 SIMD 指令集（AVX2/SSE/NEON）不提供整数除法指令，pulp 亦不提供整数 SIMD 除法 |
| u8/i8 乘法（mul） | pulp 的整数 SIMD 乘法仅支持 16 位及以上类型（u16/i16/u32/i32/u64/i64），不提供 u8/i8 SIMD 乘法 |
| 超越函数（sin, cos, tan, exp, ln, log2, log10, pow 等） | pulp 不提供 SIMD 超越函数；当前版本回退标量路径，后续版本可引入独立 SIMD 数学库 |

上述操作在并行路径下仍可并行（每线程标量迭代），仅不走SIMD。

---

## 10. 线程安全

### 10.1 Send/Sync 保证

| 存储模式 | Send | Sync | 条件 |
|----------|------|------|------|
| Owned | 是 | 是 | 元素类型为 Send+Sync |
| ViewRepr | 是 | 是 | 元素类型为 Send+Sync |
| ViewMutRepr | 是 | 否 | 元素类型为 Send |
| ArcRepr | 是 | 是 | 元素类型为 Send+Sync |

**Send/Sync 语义说明：**

| 存储模式 | Send 条件解释 | Sync 条件解释 |
|----------|---------------|---------------|
| Owned | 元素可跨线程移动，要求 T: Send | 元素可跨线程共享引用，要求 T: Sync |
| ViewRepr | `&T` 跨线程移动要求 T: Sync（View 内含 &T） | `&&T` 跨线程共享要求 T: Sync |
| ViewMutRepr | `&mut T` 可跨线程移动（转移独占访问权），要求 T: Send | `&mut T` 不可共享（Rust 独占借用规则），故永远不是 Sync |
| ArcRepr | Arc 内部使用原子计数，要求 T: Send+Sync 才能跨线程共享 | 多个 Arc 可同时持有 &T，要求 T: Sync |

**ViewMutRepr: Send 的前提条件：**

- 元素类型须实现 `Send`（可安全跨线程移动）
- 视图转移后，原线程不再持有任何引用（独占语义保证）
- 不存在其他指向同一内存的引用（Rust 借用检查器保证）

### 10.2 并行迭代安全

| 规则 | 要求 |
|------|------|
| 访问隔离 | 并行迭代须保证各线程访问不重叠的元素区间 |
| 步长处理 | 非连续数组的并行迭代须正确处理步长跳跃，不得访问逻辑元素之外的内存 |
| 独占保证 | 可变并行迭代须保证独占访问（无别名） |

### 10.3 Padding 字节并发规则

| 规则 | 要求 |
|------|------|
| SIMD 访问许可 | SIMD 操作可读取 padding 字节（无副作用，值可能非零但安全）；SIMD 写操作可能覆写 padding 字节（值变为未定义）。这是填充支持 SIMD 的必要条件（见 §7.6.5） |
| 禁止逻辑暴露 | Padding 字节不得作为逻辑元素暴露给用户：视图切片不得包含 padding 区域，迭代器不得产出 padding 元素，索引访问不得触及 padding 区域 |
| Padding 初始化 | Padding bytes 在分配时必须已零初始化。同一 padding cache line 的并发写入：由调用者保证不同时写入同一 padding 行，或使用原子操作。
| 并行工作区间 | 并行迭代须按**逻辑元素边界**分割工作区间，不得将 padding 字节分配给任何线程。线程间不得同时写同一 cache line 的 padding 区域（避免 false sharing） |
| 写操作范围 | 逐元素写操作（fill, map_inplace, 复合赋值等）只写入逻辑元素范围，padding 值未定义（见 §7.6.5） |

---

## 11. 迭代器

### 11.1 迭代类型

| 类型 | 说明 |
|------|------|
| 元素迭代 | 按内存布局顺序 |
| 轴迭代 | 沿指定轴，每次产出降维子视图 |
| 窗口迭代 | 滑动窗口 |
| 索引迭代 | 带多维索引的元素迭代 |
| 并行迭代 | rayon 支持 |
| 多数组同步迭代 | zip |
| 遍历顺序 | 可指定 F/C |

### 11.2 迭代器协议

| 场景 | 行为 |
|------|------|
| zip 形状完全一致 | 按指定遍历顺序同步产出元素 |
| zip 形状不一致但可广播 | 自动广播后迭代，等价于先 broadcast 再 zip。广播视图使用零步长实现逻辑重复（无额外堆分配），步长为 0 的轴在迭代时重复产出同一位置的数据。**遍历顺序规则**：若未指定遍历顺序，使用库默认 C-order（广播视图本身无物理内存，"物理布局顺序"在此场景下无意义，统一使用 C-order 保证行为确定性）。若显式指定遍历顺序（F 或 C），则按指定逻辑顺序迭代，可能产生非连续访问 |
| zip 形状不一致且不可广播 | 返回 BroadcastError |
| 遍历顺序未指定 | 默认按物理内存布局顺序（连续数组最优）。对于广播视图等无明确物理布局的场景，使用 C-order |
| 遍历顺序指定为 F/C | 强制按指定逻辑顺序，可能非连续访问 |
| 并行迭代 | 将元素按连续块分割给线程，块大小 ≥ 并行阈值 |
| 窗口迭代越界 | 不产出不完整窗口（窗口数 = shape - window_size + 1） |
| 空数组迭代 | 立即结束，产出零个元素 |

**zip 广播迭代的内存分配行为**：

广播迭代器使用零步长（stride=0）模拟维度扩展，不产生实际的数据拷贝。但迭代器内部需要维护多维坐标计数器以跟踪遍历位置：

| 维度类型 | 坐标计数器实现 | 堆分配 |
|----------|---------------|--------|
| 静态维度（Ix0~Ix6） | 固定大小数组 `[usize; N]`（栈分配） | 无 |
| 动态维度（IxDyn） | `InlineArray<usize, 8>`（≤ 8 维栈分配，> 8 维回退堆分配），与 IxDyn 内部表示一致 | 通常无（6 维以内零堆分配） |

此设计与 §1.2 "性能敏感 — 避免不必要的分配" 一致。`no_std` 环境下，IxDyn 迭代器使用 `alloc::vec::Vec` 作为坐标计数器（`no_std` 需 `alloc` feature，见 §1.3）。

### 11.3 广播视图的可变迭代安全性

广播通过零步长（stride=0）模拟维度扩展。当对广播后的可变视图调用 `iter_mut()` / `par_iter_mut()` 时，多个逻辑迭代位置会映射到同一物理元素，构成别名写入。

**设计决策：禁止对广播视图进行可变迭代**

| 场景 | 行为 | 理由 |
|------|------|------|
| `TensorViewMut` 存在零步长轴时调用 `iter_mut()` | panic（运行时检查） | 多个 `&mut` 指向同一元素违反 Rust 别名规则，属于未定义行为 |
| `TensorViewMut` 存在零步长轴时调用 `par_iter_mut()` | panic（运行时检查） | 同上，且并行场景构成数据竞争 |
| `zip_mut` 等内部可变操作遇到广播目标 | panic（运行时检查） | 写入广播目标会导致同一位置被多次写入，结果依赖迭代顺序，语义不明确 |
| `TensorViewMut` 无零步长轴（非广播视图） | 正常执行 | 无别名冲突 |

**实现要求**：

- `iter_mut()` / `par_iter_mut()` 入口处须检查 `has_zero_stride()`，若为 true 则 panic
- panic 信息须明确标注原因：`"cannot iter_mut on broadcast view (zero-stride axis detected)"`
- 调用方可通过 `has_zero_stride()` 预检，避免运行时 panic
- 此规则不适用于只读迭代（`iter()` / `par_iter()`），因为多个 `&T` 指向同一元素是安全的

### 11.4 迭代器 Trait 覆盖

所有迭代器须实现以下标准库 trait：

| Trait | 适用范围 | 说明 |
|-------|----------|------|
| `Iterator` | 所有迭代器 | 必须实现 `Item`、`next()`。所有迭代器的 `size_hint()` 须返回精确值（即 `(len, Some(len))`，与 `ExactSizeIterator::len()` 一致），以确保并行迭代的工作分割均匀（见 §9.2 并行阈值与分块策略） |
| `IntoIterator` | Tensor, TensorView, TensorViewMut, ArcTensor | 详见 §11.5 |
| `ExactSizeIterator` | 元素迭代、轴迭代、索引迭代 | 提供 `len()`，长度在迭代前已知 |
| `DoubleEndedIterator` | `ContiguousIter`（仅连续数组的迭代器） | 详见 §11.7 设计理由 |
| `FusedIterator` | 所有迭代器 | 迭代结束后 `next()` 永远返回 `None` |
| `IntoParallelIterator` | Tensor, TensorView, TensorViewMut, ArcTensor（feature `parallel`） | 需要 `A: Send`；按连续块分割给线程。详见 §11.9 |
| `ParallelIterator` | 并行迭代器（feature `parallel`） | rayon `ParallelIterator`，提供 `par_iter()`、`par_iter_mut()` |

**`size_hint()` 精确性要求**：

所有迭代器（包括窗口迭代器、zip 迭代器等）的 `size_hint()` 必须返回精确值。理由：(1) 并行迭代的任务分割依赖精确的长度信息（§9.2 "每块不小于 4K 元素"），不精确的 `size_hint()` 会导致不均匀的工作负载或丢失尾部元素；(2) `ExactSizeIterator` trait 的语义要求 `size_hint()` 与 `len()` 一致；(3) 即使未实现 `ExactSizeIterator` 的迭代器（如不定长链式迭代），也应在每个独立迭代器层面保证精确性。

### 11.5 IntoIterator 实现表

所有 TensorBase<S, D> 相关类型须实现 `IntoIterator`。各存储模式及引用方式的 `Item` 类型如下：

**按值消费（`into_iter()`）**：

| 类型 | `IntoIterator::IntoIter` | `Item` | 行为 |
|------|--------------------------|--------|------|
| `Tensor<A, D>`（Owned） | `IntoIter<A, D>` | `A` | 消耗所有权，逐元素 move/copy 出来。实现须保证：连续数组按指针递增顺序产出（可 SIMD 友好）；非连续数组按 strides 跳跃寻址产出。内部通过 `unsafe` 逐元素读出并 forget 原存储，避免 double drop |
| `TensorView<'a, A, D>` | `Iter<'a, A, D>` | `&'a A` | 视图本身是借用，无法 move 元素所有权。按值消费视图仅释放视图元数据（shape/strides），底层数据不受影响 |
| `TensorViewMut<'a, A, D>` | `IterMut<'a, A, D>` | `&'a mut A` | 同上，可变视图按值消费后释放元数据 |
| `ArcTensor<A, D>` | `IntoIter<A, D>` | `A` | 消耗 ArcTensor 所有权。若 Arc 引用计数 == 1，等效于 `Arc::try_unwrap()` 获取 Owned 后逐元素 move；若引用计数 > 1，先 `make_mut()` 深拷贝获取独立 Owned 再逐元素 move。此语义保证每次 `next()` 返回的 `A` 拥有独立所有权，无需额外 clone |

**借用迭代（`&Tensor` / `&mut Tensor` 等）**：

| 类型 | `IntoIterator::IntoIter` | `Item` | 说明 |
|------|--------------------------|--------|------|
| `&Tensor<A, D>` | `Iter<'a, A, D>` | `&'a A` | 不可变借用迭代，不消耗所有权 |
| `&mut Tensor<A, D>` | `IterMut<'a, A, D>` | `&'a mut A` | 可变借用迭代，不消耗所有权 |
| `&TensorView<'a, A, D>` | `Iter<'a, A, D>` | `&'a A` | 再借用，等效于视图的不可变迭代 |
| `&mut TensorViewMut<'a, A, D>` | `IterMut<'a, A, D>` | `&'a mut A` | 再借用，等效于视图的可变迭代 |
| `&ArcTensor<A, D>` | `Iter<'a, A, D>` | `&'a A` | 不可变借用迭代，不增加 Arc 引用计数（借用 `&self` 即可读取数据） |
| `&mut ArcTensor<A, D>` | `IterMut<'a, A, D>` | `&'a mut A` | 可变借用迭代，需要 `&mut self` 保证独占访问，不触发 `make_mut()`（借用检查器保证无其他引用） |

**设计说明**：

- `IntoIter<A, D>`（Owned 消费迭代器）与 `Iter<'a, A, D>`（借用迭代器）是不同类型。前者消费存储所有权，后者仅持有引用
- ArcTensor 的 `into_iter()` 在引用计数 > 1 时触发深拷贝——这与 `Arc::try_unwrap()` 语义一致。性能敏感场景下，建议先 `Arc::try_unwrap()` 获取 Owned 再迭代，避免隐式拷贝
- ViewRepr 和 ViewMutRepr 的 `into_iter()` 不消费底层数据（仅释放视图元数据），因此 `Item` 为引用类型

### 11.6 轴迭代器

沿指定轴遍历，每次 `next()` 产出一个降维子视图（移除指定轴后维度减少 1）。

**不可变轴迭代**：

| 属性 | 说明 |
|------|------|
| 签名 | `fn axis_iter(&self, axis: usize) -> AxisIter<'_, A, D> where D: RemoveAxis` |
| `Item` | `TensorView<'a, A, D::Smaller>` |
| 长度 | `shape[axis]` |
| 约束 | `axis` 须在 `[0, ndim)` 范围内，否则 panic（`"axis {axis} is out of bounds for array of dimension {ndim}"`） |
| 空轴 | 轴长度为 0 时产出零个元素，迭代器立即结束 |
| 内存布局 | 每个视图共享源数组底层存储（零拷贝）。第 i 个视图等价于 `self.index_axis(axis, i)`，但直接通过 strides 计算偏移，避免逐次索引计算开销 |
| trait 实现 | `Iterator` + `ExactSizeIterator` + `FusedIterator` |
| Ix0 | 不适用（Ix0 未实现 `RemoveAxis`，编译时拒绝） |

**可变轴迭代**：

| 属性 | 说明 |
|------|------|
| 签名 | `fn axis_iter_mut(&mut self, axis: usize) -> AxisIterMut<'_, A, D> where S: StorageMut<Elem = A>, D: RemoveAxis` |
| `Item` | `TensorViewMut<'a, A, D::Smaller>` |
| 长度 | `shape[axis]` |
| 约束 | 同 `axis_iter`；额外要求 `S: StorageMut`（编译时排除只读存储） |
| 空轴 | 同 `axis_iter` |
| trait 实现 | `Iterator` + `ExactSizeIterator` + `FusedIterator` |

**与 `index_axis` 的关系**：`axis_iter(axis)` 的第 i 次迭代结果与 `index_axis(axis, i)` 语义等价，但实现上直接在迭代器内部维护偏移量递增（`offset += stride[axis]`），避免每次调用 `index_axis` 时重新计算索引映射。

**与 `RemoveAxis` 的关系**：轴迭代沿指定轴遍历，每次产出 N-1 维子视图。返回类型的维度为 `<D as RemoveAxis>::Smaller`，因此需要 `D: RemoveAxis` 约束。Ix0 未实现 `RemoveAxis`（§3.3），从编译时保证零维张量无法调用轴迭代。

### 11.7 元素迭代器方法签名

**核心方法**：

| 方法 | 签名 | `Item` | 适用存储 | 说明 |
|------|------|--------|----------|------|
| `iter()` | `fn iter(&self) -> Iter<'_, A, D>` | `&A` | 所有存储模式 | 按物理内存布局顺序遍历所有元素。非连续数组按 strides 跳跃寻址 |
| `iter_mut()` | `fn iter_mut(&mut self) -> IterMut<'_, A, D> where S: StorageMut` | `&mut A` | Owned, ViewMutRepr | 可变元素迭代。**入口处须检查 `has_zero_stride()`**，若为 true 则 panic（见 §11.3）。`S: StorageMut` 约束在编译时排除 ViewRepr 和 ArcRepr |
| `iter_contiguous()` | `fn iter_contiguous(&self) -> Option<ContiguousIter<'_, A>>` | `&A` | 所有存储模式 | 仅**严格连续**数组（`is_contiguous()` = true，无 padding）返回 `Some`。底层为 slice 迭代器，实现 `DoubleEndedIterator` + `ExactSizeIterator` + `FusedIterator`。**注意**：宽松连续（`is_padded_contiguous()` 但非严格连续，即含 padding）的数组返回 `None`，须使用 `iter()` 代替。严格连续要求所有逻辑元素在内存中紧密排列，无 padding 字节穿插，这样 slice 迭代器才能正确遍历 |
| `iter_contiguous_mut()` | `fn iter_contiguous_mut(&mut self) -> Option<ContiguousIterMut<'_, A>> where S: StorageMut` | `&mut A` | Owned, ViewMutRepr | 可变连续迭代。同上要求严格连续（`is_contiguous()`），检查 `has_zero_stride()` 和 `S: StorageMut` |
| `iter_ordered()` | `fn iter_ordered(&self, order: Order) -> OrderedIter<'_, A, D>` | `&A` | 所有存储模式 | 显式指定遍历顺序（`Order::F` 或 `Order::C`）。强制按指定逻辑顺序遍历，即使与物理布局不一致（可能产生非连续访问） |
| `iter_ordered_mut()` | `fn iter_ordered_mut(&mut self, order: Order) -> OrderedIterMut<'_, A, D> where S: StorageMut` | `&mut A` | Owned, ViewMutRepr | 可变有序迭代。同上检查 |

**`ContiguousIter` 与 `DoubleEndedIterator` 设计理由**：

Rust 的 trait 系统要求 trait 实现在编译时确定，无法基于运行时状态（如连续性）条件实现 `DoubleEndedIterator`。因此采用类型分层策略：

- `iter()` → `Iter`：通用迭代器，支持所有数组（严格连续、宽松连续、非连续），**不实现 `DoubleEndedIterator`**
- `iter_contiguous()` → `Option<ContiguousIter>`：仅严格连续数组（`is_contiguous()` = true）可用，**实现 `DoubleEndedIterator`**（底层为 slice 迭代，`next_back()` 零开销）

**严格连续 vs 宽松连续的迭代器区分**：

| 连续类型 | 判定方法 | `iter_contiguous()` | `iter()` | 说明 |
|----------|---------|---------------------|----------|------|
| 严格连续 | `is_contiguous()` = true | `Some(ContiguousIter)` | `Iter` | 元素在内存中紧密排列，无 padding。两种迭代器均可用 |
| 宽松连续（有 padding） | `is_padded_contiguous()` = true 且 `is_contiguous()` = false | `None` | `Iter` | 行间有 padding 字节，slice 迭代器不适用。须使用 `iter()` |
| 非连续 | `is_padded_contiguous()` = false | `None` | `Iter` | 步长跳跃，只能用 `iter()` |

此设计的用户体验代价是调用方需要分支处理：

```rust
// 需要双向迭代时的推荐模式
if let Some(it) = arr.iter_contiguous() {
    // 可以使用 next_back()、rev() 等
    let last = it.next_back();
} else {
    // 回退到单向迭代
    for elem in arr.iter() { /* ... */ }
}
```

**替代方案及取舍**：

| 方案 | 优点 | 缺点 | 取舍 |
|------|------|------|------|
| 当前方案（类型分层 + Option） | 编译时安全，trait 实现正确 | 用户需分支处理 | ✅ 采用 |
| `as_slice()` 返回 `Option<&[A]>` | 用户获得完整 slice 迭代能力 | 仅适用于连续数组，语义不统一 | 作为辅助方法提供（见 §8.3） |
| 运行时 panicking `next_back()` | API 统一 | 运行时 panic 不可接受，且无法实现 trait | ❌ 拒绝 |

**0 维张量（Ix0）的迭代行为**：

| 场景 | 行为 | 说明 |
|------|------|------|
| `Tensor0<A>.iter()` | 产出恰好 1 个元素 `&A` | Ix0 元素数 = 1（§3.6），迭代一次后结束 |
| `Tensor0<A>.iter_mut()` | 产出恰好 1 个元素 `&mut A` | 同上 |
| `Tensor0<A>.iter_contiguous()` | 返回 `Some(ContiguousIter)`，产出 1 个元素 | 0D 数组始终连续（无步长，无维度） |
| `Tensor0<A>.into_iter()`（Owned） | 产出恰好 1 个 `A` 值 | 消费所有权，返回唯一元素 |
| `Tensor0<A>.axis_iter()` | 编译错误 | Ix0 未实现 `RemoveAxis`（§3.3），编译时拒绝 |
| `Tensor0<A>.indexed_iter()` | 产出恰好 1 个 `((), &A)` | 索引模式为 `()`（Ix0::Pattern） |
| `Tensor0<A>.windows()` | 编译错误或 panic | 0D 无轴可滑动，须在 `window_size.ndim() == 0` 时 panic（§17.16 要求 ndim 一致） |

### 11.8 索引迭代器

带多维索引的元素迭代，每次 `next()` 同时产出元素的多维坐标和元素引用。

**不可变索引迭代**：

| 属性 | 说明 |
|------|------|
| 签名 | `fn indexed_iter(&self) -> IndexedIter<'_, A, D>` |
| `Item` | `(D::Pattern, &'a A)` |
| 遍历顺序 | 按物理内存布局顺序产出，但索引按逻辑坐标递增（F-order 坐标顺序） |
| 长度 | `self.len()` |
| trait 实现 | `Iterator` + `ExactSizeIterator` + `FusedIterator` |

**可变索引迭代**：

| 属性 | 说明 |
|------|------|
| 签名 | `fn indexed_iter_mut(&mut self) -> IndexedIterMut<'_, A, D> where S: StorageMut` |
| `Item` | `(D::Pattern, &'a mut A)` |
| 约束 | 同 `iter_mut()`：须检查 `has_zero_stride()`，若为 true 则 panic（见 §11.3） |
| trait 实现 | `Iterator` + `ExactSizeIterator` + `FusedIterator` |

**各维度 `Item` 类型映射**：

| 维度类型 D | `D::Pattern` | `Item`（不可变） | 示例 |
|------------|-------------|-----------------|------|
| Ix0 | `()` | `((), &A)` | `((), &42.0)` |
| Ix1 | `usize` | `(usize, &A)` | `(3, &1.5)` |
| Ix2 | `(usize, usize)` | `((usize, usize), &A)` | `((1, 2), &3.14)` |
| Ix3 | `(usize, usize, usize)` | `((usize, usize, usize), &A)` | `((0, 1, 2), &2.71)` |
| Ix4~Ix6 | `(usize, ..., usize)` | `((usize, ..., usize), &A)` | — |
| IxDyn | `Vec<usize>` | `(Vec<usize>, &A)` | `(vec![1, 2, 0], &1.0)` |

**设计说明**：

- 索引使用 `D::Pattern`（§3.2 Dimension 关联类型），与 `Index` trait 的索引模式一致，保持 API 统一
- **IxDyn 索引分配**：`IndexedIter` 内部使用 `InlineArray<usize, 8>` 作为坐标计数器（与广播迭代器一致，见 §11.2）。由于 `IxDyn::Pattern` 已改为 `InlineArray<usize, 8>`（见 §3.2 关联类型映射），`next()` 输出的坐标类型也为 `InlineArray<usize, 8>`，每次 `next()` 无堆分配（内部计数器 clone 到输出，≤ 8 维为栈拷贝）。若需 `Vec<usize>` 输出，可调用 `.collect::<Vec<_>>()`。对于极少见的 > 8 维情况，`InlineArray` 自动退化为堆分配，行为与原 `Vec<usize>` 一致。
- 索引始终按逻辑坐标递增（F-order 坐标顺序，最右轴变化最快），不受物理内存布局影响。这保证索引 `(0, 0, 0) → (0, 0, 1) → ... → (shape[0]-1, shape[1]-1, shape[2]-1)` 的确定性顺序

### 11.9 并行迭代

**适用存储模式**：

| 方法 | 签名 | 适用类型 | 约束 |
|------|------|----------|------|
| `par_iter()` | `fn par_iter(&self) -> ParIter<'_, A, D>` | Tensor, TensorView, TensorViewMut, ArcTensor（不可变借用） | `A: Send + Sync`（多线程共享引用要求 Sync） |
| `par_iter_mut()` | `fn par_iter_mut(&mut self) -> ParIterMut<'_, A, D> where S: StorageMut` | Tensor, TensorViewMut（可变借用） | `A: Send`（独占移动跨线程）。ArcRepr 不适用（未实现 StorageMut，见 §6.2） |

**IntoParallelIterator 实现**：

| 类型 | `Item` | 约束 | 说明 |
|------|--------|------|------|
| `Tensor<A, D>` | `A` | `A: Send` | 消费所有权，按连续块分割后逐线程 move 元素 |
| `&Tensor<A, D>` | `&A` | `A: Send + Sync` | 借用迭代 |
| `&mut Tensor<A, D>` | `&mut A` | `A: Send` | 可变借用迭代 |
| `TensorView<'a, A, D>` | `&'a A` | `A: Send + Sync` | 视图的并行只读迭代 |
| `TensorViewMut<'a, A, D>` | `&'a mut A` | `A: Send` | 视图的并行可变迭代。独占语义（`&mut self`）保证各线程写入不重叠 |
| `ArcTensor<A, D>` | `&A` | `A: Send + Sync` | 共享引用的并行只读迭代 |
| `&ArcTensor<A, D>` | `&A` | `A: Send + Sync` | 借用的并行只读迭代 |

**TensorViewMut 的并行安全保证**：`TensorViewMut` 实现独占语义（同一时间只能有一个 `&mut` 引用）。`par_iter_mut()` 按连续块分割后，每个线程获得独立的 `&mut A` 引用区间，Rust 借用检查器在调用侧（`&mut self`）保证独占，rayon 在运行时保证区间不重叠。线程安全要求 `A: Send`（元素可跨线程移动），不需要 `A: Sync`（不存在共享引用）。

**ArcRepr 不可并行可变迭代**：ArcRepr 未实现 `StorageMut`（§6.2），因此 `par_iter_mut()` 的 `where S: StorageMut` 约束在编译时排除 ArcRepr。ArcRepr 的可写访问须通过 `make_mut()` 获取 `&mut [A]`（§6.3），再构造临时 Tensor 进行并行可变迭代。

### 11.10 窗口迭代器补充

窗口迭代器在 §17.16 定义了基本语义。以下补充非连续源数组的行为和性能特性：

**非连续源数组的窗口迭代**：

| 属性 | 说明 |
|------|------|
| 支持非连续源 | 窗口迭代器支持任意 strides 的源数组（包括转置视图、切片等） |
| 窗口内步幅 | 继承源数组步幅（可能非连续） |
| 窗口间步幅 | 等于源数组各轴步长（即窗口滑动 1 个逻辑位置），可能非连续 |
| 缓存性能 | 非连续源数组（如转置视图）的窗口迭代产生非顺序内存访问，缓存命中率低于连续数组。性能敏感场景下，建议先 `as_f_contiguous()` / `as_c_contiguous()` 转为连续布局再迭代 |

**窗口视图的生命周期**：

| 属性 | 说明 |
|------|------|
| 生命周期绑定 | 窗口视图 `TensorView` 的生命周期绑定到 `Windows` 迭代器持有的源数组借用 `&'a TensorBase<S, D>` |
| 迭代中收集 | 迭代期间产出的 `TensorView` 与迭代器共享同一借用。调用 `collect::<Vec<_>>()` 收集所有窗口是安全的（所有视图共享同一借用源） |
| 迭代后使用 | `Windows` 迭代器被消费（`for` 循环结束或 `collect` 完成后），借用的生命周期结束，产出的视图自动失效。这与 Rust 标准借用规则一致 |
| 不实现 `DoubleEndedIterator` | 窗口视图为 N 维子数组，非连续内存布局不支持高效的反向遍历。仅实现 `Iterator` + `ExactSizeIterator` + `FusedIterator` |

---

# 第四部分：API 规范

## 12. 逐元素运算

### 12.1 操作分类与适用类型

**算术运算**（通过运算符重载提供，见 §20）：

| 操作 | 运算符 | 适用元素类型 | Trait Bound |
|------|--------|-------------|-------------|
| 加法 | `+` (Add) | 整数、浮点、复数 | `A: Numeric` |
| 减法 | `-` (Sub) | 整数、浮点、复数 | `A: Numeric` |
| 乘法 | `*` (Mul) | 整数、浮点、复数 | `A: Numeric` |
| 除法 | `/` (Div) | 整数、浮点、复数 | `A: Numeric` |

**一元数值函数**（方法调用）：

| 操作 | 方法名 | 适用元素类型 | Trait Bound | 语义与边界行为 |
|------|--------|-------------|-------------|---------------|
| 绝对值 | `abs` | 整数、浮点 | `A: Numeric` | `|x|`；`abs(i32::MIN)` panic（|MIN| 超出 i32 范围）；`abs(-0.0)` → `0.0` |
| 绝对值（复数） | `abs` | Complex | `A: ComplexScalar` | 返回 `Tensor<A::Real, D>`，值为复数模 `|z|`（实数类型）。与 `norm()` 行为完全一致（`abs` 为 `norm` 的别名，两者等价）。保持与实数 `abs` 的直觉一致性：`abs` 对任何类型返回非负实数值 |
| 模（复数） | `norm` | Complex | `A: ComplexScalar` | 返回 `Tensor<A::Real, D>`（元素类型变为实数）。值为 `|z|`，使用 hypot 避免溢出（见 §4.6） |
| 符号函数 | `signum` | 整数、浮点、复数 | `A: Numeric` | 与 §4.4 `Numeric::signum()` 一致。方法名使用 `signum`（与 Rust 标准库 `i32::signum()` / `f64::signum()` 一致）。整数：正→1，零→0，负→-1；浮点同整数并传播 NaN；复数：z / |z|，零时返回零 |
| 平方 | `square` | 整数、浮点、复数 | `A: Numeric` | 等价于 `x * x`；对复数为 `z * z`（非模的平方）。整数溢出遵循 §12.8 策略 |
| 倒数 | `reciprocal` | 整数、浮点、复数 | `A: Numeric` | 等价于 `A::one() / x`；`reciprocal(0)` 对浮点返回 Inf（IEEE 754），对整数 panic（除零） |
| 整数幂 | `powi` | 整数、浮点、复数 | `A: Numeric` | `powi(&self, exp: i32) -> Tensor<A, D>`，逐元素 `x.powi(exp)`；整数溢出遵循 §12.8 策略 |

**一元运算符**（见 §20.1 运算符重载）：

| 运算符 | 适用元素类型 | Trait Bound | 说明 |
|--------|-------------|-------------|------|
| `-` (Neg) | 整数、浮点、复数 | `A: Numeric` | 逐元素取反，`Neg` trait 运算符重载 |
| `!` (Not) | 仅 bool | — | 逻辑非。整数类型不提供 `!` 运算符重载（见 §12.3 约束 1） |

**数学函数**（仅 RealScalar：f32, f64，见 §4.5）：

| 操作 | 方法名 | 语义 | NaN/Inf 行为 |
|------|--------|------|-------------|
| 正弦 | `sin` | `sin(x)` | `sin(NaN)` → NaN；`sin(±Inf)` → NaN |
| 余弦 | `cos` | `cos(x)` | `cos(NaN)` → NaN；`cos(±Inf)` → NaN |
| 正切 | `tan` | `tan(x)` | `tan(NaN)` → NaN；`tan(±Inf)` → NaN |
| 反正弦 | `asin` | `asin(x)` | `asin(NaN)` → NaN；`|x| > 1` → NaN |
| 反余弦 | `acos` | `acos(x)` | `acos(NaN)` → NaN；`|x| > 1` → NaN |
| 反正切 | `atan` | `atan(x)` | `atan(NaN)` → NaN；`atan(±Inf)` → ±π/2 |
| 二元反正切 | `atan2` | `atan2(y, x)` | 二元运算（见 §12.5）；遵循 IEEE 754 `atan2` 规则 |
| 双曲正弦 | `sinh` | `sinh(x)` | `sinh(NaN)` → NaN |
| 双曲余弦 | `cosh` | `cosh(x)` | `cosh(NaN)` → NaN |
| 双曲正切 | `tanh` | `tanh(x)` | `tanh(NaN)` → NaN |
| 平方根 | `sqrt` | `sqrt(x)` | `sqrt(NaN)` → NaN；`sqrt(-1.0)` → NaN；`sqrt(+Inf)` → +Inf |
| 立方根 | `cbrt` | `cbrt(x)` | `cbrt(NaN)` → NaN；`cbrt(负数)` → 负数（实数立方根支持负数） |
| 指数 | `exp` | `eˣ` | `exp(NaN)` → NaN；`exp(+Inf)` → +Inf；`exp(-Inf)` → 0.0 |
| 自然对数 | `ln` | `ln(x)` | `ln(NaN)` → NaN；`ln(0)` → -Inf；`ln(-1.0)` → NaN；`ln(+Inf)` → +Inf |
| 二进制对数 | `log2` | `log₂(x)` | 同 `ln` 的边界行为 |
| 常用对数 | `log10` | `log₁₀(x)` | 同 `ln` 的边界行为 |
| 下取整 | `floor` | `⌊x⌋` | `floor(NaN)` → NaN；`floor(±Inf)` → ±Inf |
| 上取整 | `ceil` | `⌈x⌉` | `ceil(NaN)` → NaN；`ceil(±Inf)` → ±Inf |
| 四舍五入 | `round` | round-to-nearest-even | `round(NaN)` → NaN |
| 浮点幂运算 | `powf` | `xʸ` | 二元运算（见 §12.5）；遵循 IEEE 754 |

**复数张量运算**（需 `A: ComplexScalar`，见 §4.6）：

| 操作 | 方法名 | 返回元素类型 | 语义 |
|------|--------|-------------|------|
| 共轭 | `conj` | `A`（Complex） | 逐元素共轭 `z̄`，返回同类型 |
| 复数指数 | `cexp` | `A`（Complex） | 逐元素 eᶻ（ComplexScalar::exp） |
| 复数对数 | `cln` | `A`（Complex） | 逐元素 ln(z)（ComplexScalar::ln） |
| 复数平方根 | `csqrt` | `A`（Complex） | 逐元素 sqrt(z)（ComplexScalar::sqrt）；`csqrt(-1+0i)` → `0+1i` |
| 辐角 | `arg` | `A::Real` | 返回 `Tensor<A::Real, D>`；值为 arg(z)，范围 (-π, π] |
| 模的平方 | `norm_sqr` | `A::Real` | 返回 `Tensor<A::Real, D>`；值为 re² + im² |

> **设计说明 — 实数与复数同名运算的区分**：RealScalar（§4.5）和 ComplexScalar（§4.6）均提供 `exp`/`ln`/`sqrt` 方法，但语义不同（如 `sqrt(-1.0)` 对 RealScalar → NaN，对 ComplexScalar → `i`）。复数张量使用前缀 `c` 命名（`cexp`/`cln`/`csqrt`），理由：(1) Rust 不允许同一类型上多个 inherent impl 定义同名方法（即使 where 约束不重叠），统一方法名需引入额外 dispatch trait 增加复杂度；(2) 显式命名避免 IDE 自动补全歧义，用户可明确知道调用的语义；(3) 与 `conj`/`arg`/`norm` 等复数专用方法保持一致的命名空间

### 12.2 比较操作语义

| 操作 | 方法 | 返回类型 | 元素约束 | 说明 |
|------|------|----------|----------|------|
| 等于 | `eq(&self, other) -> Tensor<bool, D>` | `Tensor<bool, D>` | `A: PartialEq` | 逐元素比较，NaN != NaN |
| 不等于 | `ne(&self, other) -> Tensor<bool, D>` | `Tensor<bool, D>` | `A: PartialEq` | 逐元素比较，NaN != NaN 为 true |
| 小于 | `lt(&self, other) -> Tensor<bool, D>` | `Tensor<bool, D>` | `A: PartialOrd` | 逐元素比较，NaN 比较返回 false |
| 小于等于 | `le(&self, other) -> Tensor<bool, D>` | `Tensor<bool, D>` | `A: PartialOrd` | 逐元素比较，NaN 比较返回 false |
| 大于 | `gt(&self, other) -> Tensor<bool, D>` | `Tensor<bool, D>` | `A: PartialOrd` | 逐元素比较，NaN 比较返回 false |
| 大于等于 | `ge(&self, other) -> Tensor<bool, D>` | `Tensor<bool, D>` | `A: PartialOrd` | 逐元素比较，NaN 比较返回 false |

### 12.3 约束

**约束 1**：逐元素算术运算（add, sub, mul, div）和数值函数（abs, signum, square, reciprocal, powi）仅适用于数值类型（整数、浮点、复数），不适用于 bool。数学函数（sin, cos, tan, exp, ln, sqrt 等）仅适用于 RealScalar（f32, f64），不适用于整数和复数。bool 类型仅支持逻辑非（`!` Not 运算符）和比较运算（eq, ne）。bool **不支持**位运算（`& | ^`）的运算符重载——需要位运算时，应先 `.map(|b| b as u8)` 转为整数类型后操作。

**约束 2**：`eq`/`ne` 仅要求 `PartialEq`；`lt`/`le`/`gt`/`ge` 要求 `PartialOrd`（浮点类型因 NaN 为偏序）。所有比较方法仅通过方法调用提供，不提供 `< <= > >=` 运算符语法（见 §20.6 设计决策）。支持广播（见 §12.6）。返回的 bool Tensor 可用于 `select()`、`mask()`、`compress()` 等条件操作。

### 12.4 签名规范

所有逐元素运算方法遵循以下签名模式：

**一元运算**（`&self` → 新 `Tensor<A, D>`）：

| 模式 | 签名 | 说明 |
|------|------|------|
| Numeric 一元 | `fn <op>(&self) -> Tensor<A, D> where A: Numeric` | abs, signum, square, reciprocal, powi。返回新分配的 Owned 数组 |
| RealScalar 数学 | `fn <op>(&self) -> Tensor<A, D> where A: RealScalar` | sin, cos, exp, ln, sqrt 等。返回新分配的 Owned 数组 |
| ComplexScalar 专用 | `fn <op>(&self) -> Tensor<A, D> where A: ComplexScalar` | conj, cexp, cln, csqrt。返回新分配的 Owned 数组 |
| ComplexScalar → Real | `fn <op>(&self) -> Tensor<A::Real, D> where A: ComplexScalar` | norm, norm_sqr, arg。元素类型由 Complex 变为 Real |

**二元运算**（见 §12.5）：

| 模式 | 签名 | 说明 |
|------|------|------|
| 张量-张量 | `fn <op>(&self, other: &Tensor<A, D>) -> Tensor<A, D>` | 支持广播（见 §12.6） |
| 张量-标量 | `fn <op>_s(&self, scalar: A) -> Tensor<A, D>` | 标量自动广播为与 self 相同形状 |

**返回类型规则**：

| 规则 | 说明 |
|------|------|
| 一元运算（默认） | 返回 `Tensor<A, D>`（新分配 Owned），元素类型不变 |
| 比较运算 | 返回 `Tensor<bool, D>`（元素类型固定为 bool） |
| norm/norm_sqr/arg | 返回 `Tensor<A::Real, D>`（元素类型由 Complex 变为 Real） |
| 算术二元运算 | 返回 `Tensor<A, D>`（新分配 Owned 或复用存储，见 §20.2 所有权矩阵） |
| 广播后输出形状 | 由广播规则推导（见 §12.6），可能不同于任一输入形状 |

### 12.5 二元运算与标量运算

**二元数值运算**（通过运算符重载，见 §20）：

| 操作 | 运算符 | 签名 | Trait Bound |
|------|--------|------|-------------|
| 加法 | `+` | `fn add(self, rhs) -> Tensor<A, D>` | `A: Numeric` |
| 减法 | `-` | `fn sub(self, rhs) -> Tensor<A, D>` | `A: Numeric` |
| 乘法 | `*` | `fn mul(self, rhs) -> Tensor<A, D>` | `A: Numeric` |
| 除法 | `/` | `fn div(self, rhs) -> Tensor<A, D>` | `A: Numeric` |

> 运算符的 `self` 和 `rhs` 可为值或引用（`Tensor` 或 `&Tensor`），所有权语义影响存储复用，详见 §20.2。

**二元数学函数**（方法调用）：

| 操作 | 方法名 | 签名 | Trait Bound | 语义 |
|------|--------|------|-------------|------|
| 浮点幂运算 | `powf` | `fn powf(&self, exp: &Tensor<A, D>) -> Tensor<A, D>` | `A: RealScalar` | 逐元素 `x.powf(y)`，支持广播 |
| 二元反正切 | `atan2` | `fn atan2(&self, x: &Tensor<A, D>) -> Tensor<A, D>` | `A: RealScalar` | 逐元素 `atan2(y, x)`，遵循 IEEE 754 |

**标量运算**（方法调用，标量自动广播）：

| 模式 | 签名示例 | 说明 |
|------|----------|------|
| 标量加法 | `fn add_s(&self, scalar: A) -> Tensor<A, D> where A: Numeric` | 等价于 `self + broadcast(scalar)`，但避免构造临时张量 |
| 标量乘法 | `fn mul_s(&self, scalar: A) -> Tensor<A, D> where A: Numeric` | 同上 |
| 其他标量 | `fn <op>_s(&self, scalar: A) -> Tensor<A, D>` | sub_s, div_s 等，模式相同 |

> **设计说明**：标量运算也可通过运算符实现（`tensor + scalar`，见 §20.2 矩阵中 `A`（标量）行），`_s` 方法提供不消耗 self 所有权的显式替代。

### 12.6 广播行为

逐元素运算的广播遵循 §16 定义的 NumPy 风格广播规则。本节描述广播在逐元素运算中的具体应用。

**适用范围**：

| 运算类别 | 是否支持广播 | 说明 |
|----------|-------------|------|
| 算术二元运算（Add/Sub/Mul/Div） | 是 | 通过运算符重载隐式广播 |
| 比较运算（eq/ne/lt/le/gt/ge） | 是 | 逐元素比较，返回广播后形状的 `Tensor<bool, D>` |
| 二元数学函数（powf, atan2） | 是 | 两个输入形状按 §16.1 规则广播 |
| 标量运算（`_s` 方法） | 是 | 标量视为 0D 张量，广播到 self 形状 |
| 一元运算（sin, cos, abs 等） | 不适用 | 仅一个输入，无广播需求 |

**广播与存储复用的交互**（见 §20.2）：

- 当广播发生时（输出形状 > 任一输入形状），输出必须分配新数组，不存在存储复用
- 当广播未发生（输入形状完全相同）时，值语义的操作数可复用存储（见 §20.2 矩阵）
- 广播产生的 size-1 维度使用零步长（见 §7.3 `HAS_ZERO_STRIDE` 标志）

### 12.7 原地运算

逐元素运算提供以下原地（in-place）变体，避免额外的内存分配：

**复合赋值运算符**（通过 trait 重载，见 §20.5）：

| 运算符 | Trait | 签名 | 约束 |
|--------|-------|------|------|
| `+=` | AddAssign | `fn add_assign(&mut self, rhs)` | 左操作数须为可写存储（Owned 或 ViewMut） |
| `-=` | SubAssign | `fn sub_assign(&mut self, rhs)` | 同上 |
| `*=` | MulAssign | `fn mul_assign(&mut self, rhs)` | 同上 |
| `/=` | DivAssign | `fn div_assign(&mut self, rhs)` | 同上 |

> 右操作数支持广播到左操作数形状。左操作数广播禁止——须拥有完整存储。

**通用原地映射**：

| 方法 | 签名 | 说明 |
|------|------|------|
| `mapv_into` | `fn mapv_into<F>(self, f: F) -> Tensor<A, D> where F: Fn(A) -> A` | 消耗 self（Owned），逐元素应用 `f`，结果写回原存储。适用于链式调用中复用存储，如 `t.mapv_into(|x| x * 2.0)`。Owned 存储保证拥有完整数据（无零步长广播视图），因此不存在零步长写入问题。但前提条件为连续内存（非连续输入的 SIMD 优化路径不可用，此时退化为标量遍历） |
| `mapv_into_if` | `fn mapv_into_if<F, M>(self, f: F, mask: M) -> Tensor<A, D>` | 条件原地映射，仅对 mask 为 true 的位置应用 `f`（见 §21.1 mask 相关操作） |

**通用二元原地操作**：

| 方法 | 签名 | 说明 |
|------|------|------|
| `zip_mut_with` | `fn zip_mut_with<B, F>(&mut self, other: &Tensor<B, D>, f: F) where F: Fn(&mut A, &B)` | 将 self 与 other 逐元素配对，对每对元素调用 `f`。other 支持广播。适用于自定义二元原地运算 |
| `mapv_mut` | `fn mapv_mut<F>(&mut self, f: F) where F: Fn(A) -> A` | 原地映射，等价于 `self.mapv_into(f)` 但不消耗所有权（`&mut self`）。**若 `self` 存在零步长（`has_zero_stride()` 为 true），则 panic**（与 `fill` 行为一致，见 §21.4；理由：零步长写入导致同一物理位置被反复覆写，语义等价于对共享引用的 `&mut`，见 §11.3） |

### 12.8 NaN/Inf 与溢出行为

**浮点运算**（IEEE 754 语义）：

- 所有浮点逐元素运算遵循 IEEE 754 标准，NaN 和 Inf 按标准规则传播
- NaN 输入 → NaN 输出（传播语义），除非运算本身对 NaN 有特殊定义（如 `atan2(NaN, 0)` → NaN）
- 涉及 NaN 的比较运算（lt, le, gt, ge）返回 `false`（与 `PartialOrd` 语义一致）；eq 对 NaN 返回 `false`，ne 对 NaN 返回 `true`
- 各运算的具体 NaN/Inf 行为见 §12.1 数学函数表的"NaN/Inf 行为"列

**整数运算溢出策略**：

| 场景 | 行为 | 说明 |
|------|------|------|
| 加减乘溢出（逐元素运算） | **panic**（debug） / **wrapping**（release） | 与 Rust 默认行为一致：debug 模式使用 `checked` 语义（溢出 panic），release 模式使用 `wrapping` 语义（溢出回绕）。**不使用** `saturating` 语义。**例外**：归约运算（sum/prod/cumsum/cumprod）在 debug 和 release 模式均使用 checked 算术（溢出 panic），见 §14.3 |
| 整数除法 / reciprocal(0) | **panic** | 整数除零始终 panic，与 Rust `i32::div` 行为一致 |
| abs(i32::MIN) / abs(i64::MIN) | **panic** | |MIN| 超出类型范围（如 `|i32::MIN| = 2147483648 > i32::MAX`） |
| powi 溢出 | 同加减乘溢出策略 | debug panic / release wrapping |
| square 溢出 | 同加减乘溢出策略 | debug panic / release wrapping |

> **设计说明**：整数溢出策略选择与 Rust 默认一致（debug panic / release wrapping），而非 NumPy 的固定宽度回绕，理由：(1) Rust 惯用法——整数溢出在 debug 模式下 panic 是 Rust 社区的标准实践（`cargo test` 默认 debug 模式，溢出会被捕获）；(2) 安全性——debug 模式下溢出作为错误暴露，release 模式下 wrapping 提供确定性行为；(3) 与 `#[debug_assertions]` 配合——用户可通过编译配置控制行为，无需额外 API

---

## 13. 矩阵运算

### 13.1 支持的操作

| 类别 | 操作 |
|------|------|
| 基本运算 | matvec、dot（`inner` 为 `dot` 的别名，两者行为完全一致）、outer |
| 批量运算 | batch_matvec、batch_dot |

> **关于 `batch_add` 和 `batch_scale`**：这两个操作在功能上与逐元素运算完全等价（`batch_add` 等价于支持广播的逐元素加法 `&a + &b`，`batch_scale` 等价于标量乘法 `&a * scalar`，见 §20 运算符重载），不涉及维度归约或维度变化，因此不纳入矩阵运算章节。调用方应直接使用逐元素运算符。

### 13.2 矩阵运算语义

**matvec 语义**：

| 属性 | 行为 |
|------|------|
| 签名 | `fn matvec(&self, other: &Tensor<A, Ix1>) -> Tensor<A, Ix1> where A: Numeric` |
| 操作 | 矩阵-向量乘法：self(M, N) × other(N) → 结果(M)，其中 result[i] = Σ(self[i, k] * other[k]) |
| 维度约束 | self 须为 2D 且 other 须为 1D，否则返回 `InvalidShape` |
| 形状约束 | other 长度须等于 self 的第 1 轴（N），否则返回 `ShapeMismatch` |
| 非连续输入 | 支持非连续输入，使用步长跳跃遍历（标量路径）。不自动拷贝——调用方若需最佳性能，应先确保输入连续（`is_padded_contiguous()`，见 §7.4）以触发 SIMD 路径 |
| 填充输入 | 支持填充数组（见 §7.6），使用 stride 信息直接计算，填充区域的元素不参与运算 |
| 输出布局 | F-contiguous（BLAS 兼容输出，列优先。与库默认布局 C-order 不同，见 §7.1。此设计因 matvec 通常委托 BLAS 实现，BLAS 原生产出 F-contiguous 布局） |（`inner` 为 `dot` 的别名，两者行为完全一致）：

| 属性 | 行为 |
|------|------|
| 签名 | `fn dot(&self, other: &Tensor<A, Ix1>) -> A where A: Numeric` |
| 操作（实数） | 向量内积：self(N) · other(N) → 标量，`result = Σ(self[k] * other[k])` |
| 操作（复数） | Hermitian 内积：`result = Σ(self[k].conj() * other[k])`，与 BLAS `cdotc`/`zdotc` 及 NumPy `np.vdot` 一致。**注意**：非共轭内积 `Σ(self[k] * other[k])` 不满足正定性，不是数学意义上的"内积"，Xenon 不提供此操作的独立方法（调用方可通过 `(a.conj() * b).sum()` 或 `zip(a, b).map(|(a, b)| a * b).sum()` 手动计算） |
| 错误处理 | **所有错误场景均 panic**（不返回 Result）。理由：`dot` 返回裸标量 `A`，签名无法返回 `Result<A, XenonError>`。这与 `outer`（返回 Tensor，可返回 Result）不同 |
| 维度约束 | self 和 other 均须为 1D，否则 **panic**（`InvalidShape`） |
| 形状约束 | self 和 other 长度须相等，否则 **panic**（`ShapeMismatch`） |
| 空数组 | self 或 other 长度为 0 时返回 `A::zero()`（与 `sum` 对空数组的行为一致，因 `dot` 语义上等价于 `(self * other).sum()`） |
| 非连续输入 | 支持非连续输入，使用步长跳跃遍历（标量路径）。不自动拷贝——调用方若需最佳性能，应先确保输入连续以触发 SIMD 路径 |
| 填充输入 | 支持填充数组，使用 stride 信息直接计算，填充区域的元素不参与运算 |
| 浮点精度 | 并行或大数组场景下使用补偿求和算法（Neumaier，见 §9.2.2），保证精度上界可控 |

**outer 语义**：

| 属性 | 行为 |
|------|------|
| 签名 | `fn outer(&self, other: &Tensor<A, Ix1>) -> Tensor<A, Ix2> where A: Numeric` |
| 操作 | 向量外积：self(M) ⊗ other(N) → 结果(M, N)，其中 result[i, j] = self[i] * other[j] |
| 约束 | self 和 other 均须为 1D，否则返回 `InvalidShape` |
| 非连续输入 | 支持非连续输入，使用步长跳跃遍历读取 |
| 输出布局 | F-contiguous（BLAS 兼容布局，见 §25.3） |

### 13.3 批量运算维度约定

| 约定 | 规则 |
|------|------|
| batch 轴位置 | 最前面的轴为 batch 轴（轴 0, 1, ..., ndim-3 为 batch，最后 2 维为矩阵/向量） |
| batch 形状约束 | 所有操作数的 batch 维度须形状一致或可广播 |
| batch 广播 | batch 维度遵循 NumPy 广播规则（见 §16），结果 batch 形状为广播后形状 |
| batch_matvec 维度关系 | self ndim ≥ 2；other ndim = self ndim - 1（即 other 比 self 少一维：other 的所有轴对应 self 除最后二维外的 batch 部分加上最后的 N 向量维度）。输入：self(..., M, N) × other(..., N) → 输出：(..., M)。示例：self shape `(2, 3, 4, 5)` → other shape 须为 `(2, 3, 5)`（batch 相同）或可广播形状（如 `(1, 3, 5)`）；不支持 other 为 `(5,)`（需先 unsqueeze 补齐 batch 维度再调用） |
| batch_matvec 运算 | 对每个 batch 索引 b，执行 `self[b, ..., :, :] × other[b, ..., :] → output[b, ..., :]`。batch 维度的广播在运行时按 §16 NumPy 广播规则处理 |
| batch_dot | 输入：self(..., N) × other(..., N) → 输出：(...)。两个输入 ndim 相同，batch 维度须一致或可广播 |

### 13.4 批量运算签名与错误处理

| 运算 | 签名 | 输入约束 | 错误 |
|------|------|----------|------|
| batch_matvec | `fn batch_matvec(&self, other: &Tensor<A, D::Smaller>) -> Tensor<A, D::Smaller> where A: Numeric, D: RemoveAxis` | self(D: ndim ≥ 2) × other(D::Smaller: ndim = self.ndim - 1) → output(D::Smaller)；self 最后 2 维为 (M, N)，other 最后 1 维为 N，其余 batch 维度须一致或可广播 | batch 维度不可广播 → `BroadcastError`；self.shape[ndim-1] ≠ other.shape[ndim-2]（即 N 不匹配）→ `ShapeMismatch` |
| batch_dot | `fn batch_dot(&self, other: &Tensor<A, D>) -> Tensor<A, D::Smaller> where A: Numeric, D: Dimension + RemoveAxis` | self(..., N) × other(..., N) → (...)；两个输入 ndim 相同，batch 维度须一致或可广播 | batch 维度不可广播 → `BroadcastError`；向量长度不匹配 → `ShapeMismatch` |

**batch_matvec 维度推导**：`batch_matvec` 的 self 维度类型为 `D`，other 和输出维度类型为 `D::Smaller`（self 降一维）。`D: RemoveAxis` 约束（见 §3.3）确保编译时维度推导正确。静态维度推导规则：self Ix2 × other Ix1 → output Ix1；self Ix3 × other Ix2 → output Ix2；self Ix4 × other Ix3 → output Ix3；...；IxDyn → IxDyn。self Ix1 × other Ix0 不适用（Ix1 的 matvec 无矩阵语义，至少需要 2D self——虽然类型系统允许，但运行时检查 self.ndim < 2 时返回 `InvalidShape`）。

**batch_dot 维度推导**：`batch_dot` 沿最后一轴做内积后消除该轴，输出 ndim = 输入 ndim - 1。签名使用 `D::Smaller` 关联类型表达降维后的维度，并要求 `D: RemoveAxis`（见 §3.3 RemoveAxis trait）。静态维度推导规则：Ix1 → Ix0, Ix2 → Ix1, ..., IxDyn → IxDyn。Ix0 输入编译错误（Ix0 未实现 RemoveAxis，无轴可归约）。

**复数 batch_dot 语义**：复数类型的 `batch_dot` 使用 Hermitian 内积：`result[b, ...] = Σ(self[b, ..., k].conj() * other[b, ..., k])`，与 `dot` 的复数语义一致（见 §13.2）。

**非连续输入行为**：以上批量运算对非连续输入自动拷贝为 F-contiguous 后执行。不返回布局错误——调用方若需零拷贝，应先确保输入连续。**注意**：此行为与基本运算（`dot`、`matvec`）不同——基本运算对非连续输入使用步进遍历（strided traversal）而不拷贝（见 §13.2）。批量运算需要 F-contiguous 布局的原因：批量 GEMV/GEMM 通常委托 BLAS 实现（见 §25.3），BLAS 要求连续内存布局。性能敏感场景中，此隐式拷贝可能构成主要开销——调用方应在热循环外预布局输入为 F-contiguous。

**填充输入行为**：填充数组作为批量运算输入时，自动拷贝过程中移除填充（输出为标准 F-contiguous，PADDED = false）。若需保留填充优化以获得最佳性能，调用方应直接使用 BLAS FFI（见 §25.3，`is_blas_compatible()` 对填充数组返回 true，`lda()` 返回填充后的物理步长）。

**与 `dot` 返回类型差异**：`dot` 返回 `A`（标量值），`batch_dot` 返回 `Tensor<A, D::Smaller>`（标量张量）。两者定位不同：`dot` 是 1D×1D → 标量的便捷方法；`batch_dot` 支持任意批次维度，统一返回 Tensor 以保持泛型一致性（`D::Smaller` 推导需 Tensor 包装）。对 Ix1 输入，`batch_dot` 返回 `Tensor0<A>`（0D 标量张量），可调用 `.into_scalar()` 获取裸值。

---

## 14. 归约操作

### 14.1 归约类型

| 类型 | 操作 |
|------|------|
| 全局 | sum, prod, mean, var, std, min, max, argmin, argmax, all, any |
| 沿轴 | 以上所有运算均支持沿指定轴归约 |
| 累积 | cumsum, cumprod（沿指定轴，返回同形状数组） |
| NaN 忽略 | nansum, nanprod, nanmean, nanvar, nanstd, nanmin, nanmax, nanargmin, nanargmax（跳过 NaN 值的归约变体，见 §14.4） |

### 14.2 沿轴归约输出维度规则

| 属性 | 行为 |
|------|------|
| 默认行为 | 归约轴消除，输出 ndim = 输入 ndim - 1。例如 Ix3 沿 axis=1 归约 → Ix2 |
| keepdims 参数 | 提供 `sum_axis_keepdim(axis)` 变体，归约轴保留为 size-1，输出 ndim 不变 |
| argmin/argmax keepdim | 提供 `argmin_keepdim(axis)` / `argmax_keepdim(axis)` 变体，归约轴保留为 size-1，返回 `Tensor<usize, D>`（而非 `Tensor<usize, D::Smaller>`）。典型用途：配合 `take_along_axis` 实现反向索引（见 §18.3） |
| 静态维度推导 | 归约后维度类型由编译时推导：Ix(N+1) 沿任意轴归约 → IxN。Ix0 不可执行沿轴归约（无轴可归约） |
| 动态维度 | IxDyn 沿轴归约后仍为 IxDyn（ndim 减少 1） |
| argmin/argmax | 沿轴归约时，返回对应轴上的索引数组（usize 类型），输出维度规则同上。要求 `A: PartialOrd`（Complex 类型因无 `PartialOrd` 实现而编译错误，见 §5.5） |
| 全局归约 | 不指定轴时，所有轴归约为单个标量。大部分操作返回 `A`；例外：`argmin`/`argmax` 返回 `usize`（扁平索引），`all`/`any` 返回 `bool` |

### 14.3 约束

**var/std 统计定义**：默认有偏估计（除以 N，即 ddof=0），与 NumPy 默认行为一致。提供 `var_ddof(ddof: usize)` / `std_ddof(ddof: usize)` 变体以支持无偏估计（ddof=1，除以 N-1）等自定义自由度。ddof 须 < 元素数，否则 panic（单元素数组 ddof=1 无意义）。

**var/std 数值算法**：`var`/`std` 须使用 **Welford 在线算法**计算方差，以确保数值稳定性。算法维护三元组累加器 `(count, mean, M2)`，逐元素更新：

```
count += 1
delta = x - mean
mean += delta / count
delta2 = x - mean
M2 += delta * delta2
```

最终 `variance = M2 / (count - ddof)`。

**理由**：朴素公式 `E[X²] - E[X]²` 存在灾难性取消问题——当均值远大于标准差时（如 `[1e8, 1e8+1, 1e8-1]`），即使两个 sum 使用补偿求和，减法步骤仍会丢失全部有效数字。Welford 算法仅涉及 `delta` 量级的乘法，从根本上避免此问题。Welford 天然支持并行归约（两个累加器可合并），与 §9.2.2 并行策略兼容。

**并行归约**：并行方差归约时，各线程维护独立的 Welford 累加器，最终按以下公式合并：

```
count = count_a + count_b
delta = mean_b - mean_a
mean = mean_a + delta * count_b / count
M2 = M2_a + M2_b + delta² * count_a * count_b / count
```

**mean/var/std 类型约束**：`mean()`、`var()`、`std()` 仅对浮点类型（`f32`/`f64`）实现。整数数组调用这些方法将产生编译错误。实现机制：这些方法的 `where A: RealScalar` 约束（见 §4.5）在编译时排除了整数类型（整数仅实现 `Numeric`，不实现 `RealScalar`）。用户需先手动转换类型，例如 `.mapv(|x| x as f64).mean()`。理由：Xenon 不做隐式类型提升，整数除法截断会产生误导性结果。

**mean 补偿求和**：`mean()` 使用 Neumaier 补偿求和算法，与 `sum()` 保持一致的数值精度。

**整数归约溢出行为**：`sum/prod` 作用于整数数组时，溢出将 panic（debug 和 release 模式均如此）。实现须使用 `checked_add`/`checked_mul` 显式检测溢出，因为 Rust release 模式默认为 wrapping 算术（不 panic），仅 debug 模式默认 panic。**此策略与 §12.8 逐元素运算的溢出策略不同**：归约运算涉及多步累加/累乘，wrapping 导致的错误值会传播到最终结果且难以诊断；逐元素运算的每步结果独立，wrapping 语义可预测且与 Rust 默认一致。

**整数归约的 wrapping 替代**：若用户需要 wrapping 语义（溢出回绕而非 panic），可先 `cast` 为更大的整数类型执行归约再 `cast` 回来（如 `i32` → `i64` → `sum` → `i32`），或使用 `.mapv(|x| x as WrappingI32).sum()` 配合自定义 `WrappingI32` newtype（实现 `Numeric` trait 并使用 `wrapping_add`/`wrapping_mul`）。Xenon 不提供 `sum_wrapping()` / `prod_wrapping()` 内置方法——理由：溢出静默回绕在归约场景中几乎总是 bug，不应作为默认选项暴露于公开 API。显式 newtype 模式让用户表明"我确实知道这里会溢出且我需要 wrapping"。

**整数归约与 SIMD 路径**：整数类型的 `sum`/`prod` 因 checked 算术约束（每步需检测溢出）**不使用 SIMD 路径**，始终回退标量路径。主流 SIMD 指令集（AVX2/SSE/NEON）不提供整数溢出检测指令，`checked_add`/`checked_mul` 需要在每步加法/乘法后检测 carry/overflow flag，这在 SIMD 寄存器中没有对应操作。浮点和复数类型的 `sum`/`prod` 可使用 SIMD 路径（见 §9.1.3）。

**cumsum/cumprod 边界行为**：沿指定轴正方向（index 0 → N-1）累积，与 NumPy `np.cumsum(axis=...)` 行为一致。浮点和复数类型：遇到 NaN 时传播 NaN（后续元素均为 NaN）。整数类型不涉及 NaN。空数组返回同形状空数组。

**cumsum/cumprod 整数溢出行为**：`cumsum`/`cumprod` 作用于整数数组时，每步累加/累乘均检测溢出，溢出时 panic（debug 和 release 模式均如此）。实现须使用 `checked_add`/`checked_mul` 显式检测（理由同 `sum/prod`：Rust release 默认 wrapping 算术不会 panic）。若需 wrapping 行为，用户应先 `cast` 为更大的整数类型再执行累积操作。

**cumsum/cumprod 与 SIMD 路径**：整数类型的 `cumsum`/`cumprod` 因 checked 算术约束**不使用 SIMD 路径**，始终回退标量路径（理由同整数 `sum`/`prod`，见上方"整数归约与 SIMD 路径"）。浮点和复数类型的 `cumsum`/`cumprod` 可使用 SIMD 路径。

**cumsum/cumprod 溢出检查性能影响**：逐元素 checked 溃出检测相比 wrapping 累积有额外分支开销。对于 `cumsum`（加法），现代 CPU 的分支预测器对"几乎不溢出"的场景预测准确率高，实际性能损失通常 < 5%。对于 `cumprod`（乘法），若元素绝对值普遍 > 1，溢出更早触发分支误预测。若性能敏感且用户能保证不溢出，可先 `cast` 为更大类型后累积再 `cast` 回来以规避检查开销。

**cumsum/cumprod 与并行路径**：`cumsum`/`cumprod` **不使用并行路径**（即使启用 `parallel` feature），因为每步的输出依赖前一步的结果（串行数据依赖，`output[i] = output[i-1] + input[i]`）。理论上可使用并行前缀和（parallel prefix sum）算法（O(n log n) work, O(log n) span），但对典型数组大小（≤ 10^7 元素），串行实现的缓存局部性优势通常优于并行前缀和的额外内存带宽开销。若未来需要大规模并行前缀和（≥ 10^8 元素），可作为单独的 opt-in API 提供（如 `par_cumsum`）。

**argmin/argmax 多值行为**：存在多个相同最小/最大值时，返回第一个出现的索引（按遍历顺序），与 NumPy/ndarray 行为一致。

**argmin/argmax 类型约束**：`argmin`/`argmax` 要求 `A: PartialOrd`。Complex 类型不实现 `PartialOrd`（见 §5.5，复数无自然全序），因此对 Complex 数组调用 `argmin`/`argmax` 将产生编译错误。如需复数极值查找，使用 `.mapv(|x| x.norm_sqr())` 转换后操作。

**argmin/argmax NaN 行为**：浮点数组含 NaN 时，`argmin`/`argmax` 使用以下确定性规则：

1. 遍历所有元素，维护当前最小/最大值及其索引
2. **NaN 元素跳过**：NaN 与任何值比较均返回 `false`（`NaN < x` 和 `NaN > x` 均为 `false`），NaN 不参与大小比较，不更新当前极值
3. **全 NaN 数组**：所有元素均为 NaN 时，返回第一个 NaN 的索引（按 F-order 遍历顺序），与"返回第一个出现的索引"的多值规则一致
4. **空数组**：返回 `EmptyArray` 错误（见 §14.3 空数组归约行为表）

此规则保证：相同输入在 debug/release 模式、单线程/并行路径下产生相同结果。NaN 忽略变体 `nanargmin`/`nanargmax` 始终跳过 NaN（见 §14.4）。

**min/max 类型约束**：`min`/`max` 要求 `A: PartialOrd`。Complex 类型不实现 `PartialOrd`，编译错误。浮点类型使用 NaN 传播语义（§4.7）：任一参数为 NaN 则返回 NaN。

**all/any 归约语义**：

| 属性 | all | any |
|------|-----|-----|
| 输入类型 | `bool` | `bool` |
| 语义 | 是否所有元素为 `true` | 是否存在元素为 `true` |
| 空数组 | 返回 `true`（vacuous truth） | 返回 `false` |
| 沿轴返回类型 | `Tensor<bool, D::Smaller>` | `Tensor<bool, D::Smaller>` |
| SIMD 适用性 | AND 归约，可 SIMD | OR 归约，可 SIMD |
| 短路 | 不保证短路（SIMD/并行路径需完整遍历） | 不保证短路 |

**空数组归约行为**：

| 操作 | 空数组返回 | 说明 |
|------|-----------|------|
| `sum` | `A::zero()` | 加法单位元 |
| `prod` | `A::one()` | 乘法单位元 |
| `mean`/`var`/`std` | `EmptyArray` 错误 | 除零无意义，返回可恢复错误（见 §27.3） |
| `min`/`max` | `EmptyArray` 错误 | 空集无最小/大值，返回可恢复错误 |
| `argmin`/`argmax` | `EmptyArray` 错误 | 空集无索引 |
| `all` | `true` | vacuous truth |
| `any` | `false` | 无元素满足条件 |
| `cumsum`/`cumprod` | 同形状空数组 | 无元素可累积 |

### 14.4 NaN 忽略归约变体

NaN 忽略变体在归约时跳过 NaN 值，仅对浮点和复数类型实现（整数类型无 NaN，调用 NaN 忽略变体等价于对应的普通归约，但为 API 一致性仍提供）。

| 操作 | 说明 | 返回类型 |
|------|------|----------|
| `nansum` | 跳过 NaN 求和，全 NaN 或空数组返回 `A::zero()` | `A`（全局）或 `Tensor<A, D::Smaller>`（沿轴） |
| `nanprod` | 跳过 NaN 求积，全 NaN 或空数组返回 `A::one()` | 同上 |
| `nanmean` | 跳过 NaN 求均值，全 NaN 或空数组返回 `NaN` | 同上 |
| `nanvar` | 跳过 NaN 计算方差（Welford 算法，跳过 NaN 输入），全 NaN 或空数组返回 `NaN` | 同上 |
| `nanstd` | `nanvar` 的平方根 | 同上 |
| `nanmin` | 跳过 NaN 取最小值，全 NaN 或空数组返回 `NaN` | 同上 |
| `nanmax` | 跳过 NaN 取最大值，全 NaN 或空数组返回 `NaN` | 同上 |
| `nanargmin` | 跳过 NaN 返回最小值索引，全 NaN 或空数组时 panic（`EmptyArray`） | `usize`（全局）或 `Tensor<usize, D::Smaller>`（沿轴） |
| `nanargmax` | 跳过 NaN 返回最大值索引，行为同 `nanargmin` | 同上 |

**设计说明**：`nanargmin`/`nanargmax` 在全 NaN 或空数组时 panic，与 NumPy `np.nanargmin`/`np.nanargmax` 行为一致（NumPy 在此情况下 raise `ValueError`）。沿轴版本返回 `Tensor<usize, D::Smaller>`，每个位置独立判断：若该切片全为 NaN 或为空，panic。**不使用 `Option<usize>`** 作为返回元素类型——`Option<usize>` 不实现 `Element` trait（§4.3 sealed 实现者列表），无法构造 `Tensor<Option<usize>, D>`。全局版本返回裸 `usize`，与 `argmin`/`argmax`（§14.2）返回类型一致。

**NaN 忽略变体的沿轴归约**：以上所有操作均支持沿轴归约（`nan*_axis`），输出维度规则与对应普通归约一致（见 §14.2）。

---

## 15. 集合操作

### 15.1 集合操作类型

| 操作 | 说明 | 输入类型约束 | 返回类型 | 输入维度 |
|------|------|--------------|----------|----------|
| unique | 返回唯一值（排序后） | Sortable（见 §15.1.1） | Tensor1<A> | 任意维度（先按 C-order 展平为一维） |
| unique_counts | 返回唯一值及出现次数 | Sortable（见 §15.1.1） | (Tensor1<A>, Tensor1<usize>) | 任意维度（先按 C-order 展平为一维） |
| unique_inverse | 返回唯一值及原数组索引 | Sortable（见 §15.1.1） | (Tensor1<A>, Tensor1<usize>) | 任意维度（先按 C-order 展平为一维） |
| bincount | 统计非负整数出现次数 | i32 / i64（值须 ≥ 0）或 usize | Tensor1<usize> 或 Tensor1\<W\>（带权重时，W: Numeric） | 仅一维 |
| histogram | 统计落入各 bin 的元素数 | RealScalar | Tensor1<usize> | 任意维度（先按 C-order 展平为一维） |
| histogram_bin_edges | 返回 bin 边界 | RealScalar | Tensor1<A> | 任意维度（用于推断 bin 边界范围） |

#### 15.1.1 Sortable trait

`Sortable` 是 sealed trait，为集合操作提供统一的排序能力。与标准库 `Ord` 不同，`Sortable` 通过关联方法 `sort_cmp` 定义排序语义，使不支持 `Ord` 的类型（如浮点、复数）也能参与集合操作。

| 关联方法 | 签名 | 说明 |
|----------|------|------|
| `sort_cmp` | `&self, other: &Self -> Ordering` | 用于集合操作内部排序和去重 |

**实现者**：

| 类型 | `sort_cmp` 语义 |
|------|-----------------|
| i32, i64 | 委托给 `Ord::cmp`（自然全序） |
| f32, f64 | 委托给 `total_cmp`（IEEE 754 totalOrder：`-NaN < -Inf < ... < -0.0 < +0.0 < ... < +Inf < +NaN`） |
| Complex\<f32\>, Complex\<f64\> | 字典序：先比较实部（`total_cmp`），实部相等时比较虚部（`total_cmp`） |

**设计说明**：

| 决策 | 理由 |
|------|------|
| 不使用 `Ord` trait bound | `f32`/`f64` 的 `Ord` 未实现（因 NaN 为偏序）；`Complex` 不实现 `Ord`（§5.5，复数无自然全序）。使用独立的 `Sortable` trait 可统一覆盖所有需要排序的类型 |
| sealed | 与 §4.2 Sealed 策略一致，下游 crate 不可为自定义类型实现 |
| `bool` 不实现 | bool 的 unique 无实际意义（结果固定为至多 `[false, true]`），不属于集合操作的目标类型 |
| `usize` 不实现 | `usize` 定位为索引返回类型（§4.2），不作为数据值参与集合操作。`bincount` 虽接受 `usize` 输入（§15.5），但其语义是"统计非负整数出现次数"而非排序/去重，不需要 `Sortable` 约束，两者不矛盾 |
| `Sortable` 在类型体系中的位置 | 应添加至 §4 类型体系，作为 `Element` 的子 trait（`Sortable: Element`），与 `Numeric`/`RealScalar`/`ComplexScalar` 并列的独立约束层 |

#### 15.1.2 多维输入展平规则

| 规则 | 说明 |
|------|------|
| 展平顺序 | 按 C-order（行优先）展平为一维。与库默认布局一致（§7.1） |
| 展平开销 | 连续输入零拷贝（返回 1D 视图）；非连续输入需 O(n) 拷贝 |
| `unique_inverse` 索引语义 | `inverse` 中的索引为展平后的一维索引（0-based），非原始多维坐标。用户可通过 `unravel_index(index, original_shape)` 将一维索引还原为多维坐标 |
| `bincount` 不支持多维 | bincount 的语义（统计每个非负整数值的出现次数）与多维结构无关，仅接受一维输入 |

### 15.2 unique 语义

| 属性 | 行为 |
|------|------|
| 排序 | 返回的唯一值按升序排列 |
| 空数组 | 返回空 Tensor1 |
| 返回值 | Tensor1<A>，长度等于唯一值数量 |
| 排序机制 | 使用 `Sortable::sort_cmp`（§15.1.1）排序后，通过 `PartialEq::ne` 去重（相邻元素 `!=` 时保留） |
| 浮点比较 | 浮点类型使用 `total_cmp`（IEEE 754 totalOrder）进行排序和去重 |
| Complex 排序 | Complex 类型使用字典序排序：先比较实部（re），实部相等时比较虚部（im）。排序使用实部和虚部的 `total_cmp`。相等判定为 `re == re && im == im`。例：`[1+2i, 1+1i, 2+0i].unique()` → `[1+1i, 1+2i, 2+0i]` |
| NaN 处理 | 所有 NaN 视为相等，合并为一个，排在末尾（大于 +∞）。例：`[1.0, NaN, 2.0, NaN].unique()` → `[1.0, 2.0, NaN]` |
| ±0.0 处理 | `+0.0` 与 `-0.0` 在 IEEE 754 下 `==` 为 true，在 `total_cmp` 排序时 `-0.0 < +0.0`。unique 使用 `total_cmp` 排序后通过 `!=` 去重：因 `-0.0 == +0.0`（`!=` 为 false），二者合并为一个值。合并结果为排序后的首个出现，即 `-0.0`。例：`[+0.0, -0.0, 1.0].unique()` → `[-0.0, 1.0]` |
| 复杂度 | 时间 O(n log n)（排序）+ O(n)（去重），空间 O(n)。n 为输入元素数（展平后） |
| 并行 | 排序阶段可使用并行排序（rayon `par_sort`，需 `parallel` feature）；去重阶段为线性扫描，无需并行 |
| 与 NumPy 差异 | **±0.0**：Xenon 将 `+0.0` 和 `-0.0` 视为相同值（因 `PartialEq` 判定相等），合并结果为排序后首出现（`-0.0`）。NumPy `np.unique` 将二者视为不同值（基于位模式区分）。若需区分 ±0.0，可先 `.mapv(\|x\| x.to_bits())` 转为整数后 unique |

> **⚠️ 警告：浮点 ±0.0 合并行为**
>
> `unique` 对浮点类型使用 `PartialEq::ne` 去重。由于 IEEE 754 下 `+0.0 == -0.0`（`!=` 为 false），二者会被合并为排序后的首出现值（`-0.0`，因 `total_cmp` 排序时 `-0.0 < +0.0`）。
>
> **此行为与 NumPy 不同**：NumPy `np.unique` 基于位模式区分 `+0.0` 和 `-0.0`。
>
> **更微妙的影响**：去重结果取决于输入中是否同时包含两种零值。例如 `[+0.0]` 的 `unique` 结果为 `[+0.0]`，但 `[+0.0, -0.0]` 的 `unique` 结果为 `[-0.0]`。对同一组数值的不同排列可能产生不同的零值表示。
>
> **若需区分 ±0.0**：使用 `.mapv(|x| x.to_bits())` 转换为整数后调用 `unique`，再 `.mapv(|b| f64::from_bits(b))` 转回。

### 15.3 unique_counts 语义

| 属性 | 行为 |
|------|------|
| 返回值 | (values, counts)，values 为唯一值（升序），counts 为对应出现次数 |
| 约束 | values.len() == counts.len() |
| NaN 处理 | 同 unique：所有 NaN 合并为一个，计入同一 count |
| 复杂度 | 同 unique：时间 O(n log n)，空间 O(n) |

### 15.4 unique_inverse 语义

| 属性 | 行为 |
|------|------|
| 返回值 | (values, inverse)，values 为唯一值（升序），inverse 为原数组每个元素在 values 中的索引 |
| 约束 | inverse.len() == input.len()，inverse[i] ∈ [0, values.len()) |
| 重建原数组 | values[inverse[i]] == input[i] |
| NaN 处理 | 同 unique：所有 NaN 映射到同一索引 |
| inverse 索引语义 | `inverse` 中的索引对应展平后的一维序列（§15.1.2），即 `inverse[i]` 表示展平后第 i 个元素在 `values` 中的位置 |
| 复杂度 | 时间 O(n log n)（排序）+ O(n log k)（构建 inverse 映射，k 为唯一值数，使用排序后二分查找），空间 O(n + k) |

### 15.5 bincount 语义

| 属性 | 行为 |
|------|------|
| 输入约束 | 仅支持整数类型 `i32` / `i64` / `usize`（§4.2 sealed 实现者列表中所有整数类型）。`i32`/`i64` 的值须 ≥ 0（运行时检查），`usize` 天然满足非负约束。**不支持** `bool` 和浮点类型 |
| minlength 参数 | 指定输出最小长度，若 `max(input) + 1 < minlength`，则输出长度为 `minlength`。默认值为 0（即不指定时输出长度由输入最大值决定） |
| 空数组 | 返回长度为 `minlength`（默认 0）的全零数组。若 `minlength == 0` 且输入为空，返回空 `Tensor1<usize>` |
| 全零输入 | 输入全为 0 时，输出长度为 `max(1, minlength)`，`output[0] = input.len()` |
| 负值输入 | `i32`/`i64` 输入含负值时 panic（运行时检查，panic 信息：`"bincount: negative value at index {i}"`） |
| 输出长度溢出保护 | 输出长度为 `max(input) + 1`。若 `max(input) + 1` 超过 `isize::MAX`（即输出数组逻辑大小超出安全分配范围），返回 `InvalidInput` 错误（而非 panic 或 OOM）。理由：`i64::MAX + 1` 远超物理内存，不应触发无意义的巨大分配 |
| 返回值 | `Tensor1<usize>`，长度为 `max(input) + 1`（或 `minlength`，取较大值），`output[i] = count of i in input` |
| 权重参数 | 可选 `weights: Tensor1<W> where W: Numeric`，权重元素类型 W 独立于输入整数类型。带权重时返回 `Tensor1<W>`（而非 `Tensor1<usize>`），`output[i] = sum of weights[j] where input[j] == i`。`weights` 长度须与 `input` 长度相等，否则 panic |
| 复杂度 | 时间 O(n + m)，其中 n 为输入长度，m 为输出长度（`max(input) + 1`）；空间 O(m) |

### 15.6 histogram 语义

**`Bins<A>` 枚举定义**：

```rust
/// Histogram bin specification.
pub enum Bins<A: RealScalar> {
    /// Equal-width bins: divide the range into `count` bins of equal width.
    Count(usize),
    /// Custom bin edges: a strictly monotonically increasing array of boundary values.
    /// Length must be ≥ 2 (at least 1 bin, at least 2 edges).
    Edges(Tensor1<A>),
}

impl<A: RealScalar> From<usize> for Bins<A> {
    fn from(count: usize) -> Self { Bins::Count(count) }
}

impl<A: RealScalar> From<Tensor1<A>> for Bins<A> {
    fn from(edges: Tensor1<A>) -> Self { Bins::Edges(edges) }
}
```

`histogram` 签名使用 `impl Into<Bins<A>>` 以同时接受 `usize`（等宽 bin 数）和 `Tensor1<A>`（自定义边界数组），通过 `From` 实现隐式转换。

| 属性 | 行为 |
|------|------|
| bins 参数 | `impl Into<Bins<A>>`，接受 `usize`（等宽 bin 数，转为 `Bins::Count`）或 `Tensor1<A>`（自定义 bin 边界数组，转为 `Bins::Edges`）。类型定义和 `From` 实现见上方 `Bins<A>` 枚举 |
| range 参数 | `(min, max)` 元组，指定统计范围，超出范围的值不计入。**与 bins 的交互规则**：(1) 当 bins 为整数时，range 指定等宽分割的上下界（未指定时默认使用数据的最小/最大值）。(2) 当 bins 为自定义边界数组时，**不允许**同时提供 range 参数——边界已由数组显式指定，同时提供 bins 数组和 range 返回 `InvalidInput` 错误（非 `InvalidShape`，因这是语义冲突而非形状不匹配） |
| 空数组 | 返回全零数组（长度等于 bin 数） |
| 返回值 | `Tensor1<usize>`，长度等于 bin 数 |
| NaN 输入 | NaN 不计入任何 bin，且不触发 panic。行为分两种情况：(1) **range 已指定**（调用方提供 `(min, max)` 或 bins 为自定义边界数组时）：NaN 被静默跳过，返回的计数仅基于非 NaN 元素。(2) **range 未指定**（bins 为整数且未提供 range 参数）：需从数据推断范围，此时若输入含 NaN，`min()`/`max()` 将返回 NaN，无法确定有效的 bin 边界，函数返回 `InvalidInput` 错误（错误信息："autodetected range is not finite: input contains NaN and no explicit range provided"）。调用方若需在含 NaN 数据上使用 histogram，应显式提供 range 参数 |
| bin 边界规则 | 设 bin 边界数组为 `[e0, e1, ..., ek]`（共 k+1 个边界，k 个 bin），则第 i 个 bin（0 ≤ i < k-1）的区间为 `[ei, ei+1)`（左闭右开），最后一个 bin（第 k-1 个）的区间为 `[ek-1, ek]`（左闭右闭）。与 NumPy `np.histogram` 行为一致 |
| 自定义边界数组验证 | 当 bins 为自定义边界数组 `Tensor1<A>` 时，须满足以下约束，否则返回 `InvalidInput` 错误：(1) **长度 ≥ 2**（至少 1 个 bin，至少 2 个边界）；(2) **严格单调递增**（`e0 < e1 < ... < ek`，非递增数组无法定义有效区间）；(3) **不含 NaN**（NaN 边界无法定义区间）；(4) **不含 Inf**（`±Inf` 边界导致区间语义不明确） |
| 复杂度 | 时间 O(n + k)，其中 n 为输入元素数（展平后），k 为 bin 数；空间 O(k) |

### 15.7 histogram_bin_edges 语义

计算 histogram 的 bin 边界数组。是 `histogram` 内部步骤的公开 API 暴露，供用户在多次 histogram 操作间复用边界。

| 属性 | 行为 |
|------|------|
| 签名 | `fn histogram_bin_edges(bins: Bins, range: Option<(A, A)>) -> Result<Tensor1<A>, InvalidInput> where A: RealScalar` |
| bins 参数 | 整数（等宽 bin 数）或 `Tensor1<A>`（自定义边界，此时直接返回该数组的 clone） |
| range 参数 | `(min, max)` 元组，指定等宽分割的上下界。bins 为自定义边界数组时不可提供（同 §15.6 规则） |
| 等宽边界计算 | 给定 bin 数 `k` 和 range `(lo, hi)`：`step = (hi - lo) / k`，边界为 `[lo, lo + step, lo + 2*step, ..., hi]`，共 `k + 1` 个边界 |
| 默认 range | bins 为整数且未提供 range 时，从输入数据推断：`lo = input.min()`，`hi = input.max()`。输入含 NaN 时返回 `InvalidInput`（同 §15.6） |
| 返回值 | `Tensor1<A>`，长度等于 bin 数 + 1（k 个 bin 产生 k+1 个边界） |
| 空输入 | bins 为整数且输入为空数组时，返回 `InvalidInput`（无法从空数据推断 range）。调用方须显式提供 range |

---

## 16. 广播

### 16.1 基本规则

遵循 NumPy 广播规则（右对齐，size-1 维度拉伸）。广播是零拷贝操作——通过零步长（stride=0）模拟维度扩展，不产生实际数据拷贝（见 §17.1 零拷贝操作分类）。

**广播步骤**：

| 步骤 | 说明 |
|------|------|
| 1. 维度对齐 | 从最右维度开始对齐，维度数不足的数组在左侧补 1（仅逻辑上，不改变原数组） |
| 2. 兼容检查 | 对应维度相等，或其中一个为 1。不满足时返回 `BroadcastError`（见 §27.2） |
| 3. 输出形状 | 每个轴：若一个为 1，取另一个；若相等，取该值。详见 §16.4 算法 |
| 4. 零步长 | size-1 维度广播为目标大小时，该轴步长设为 0，逻辑上重复该维度的单个元素 |

**设计约束**：

| 约束 | 说明 |
|------|------|
| 广播视图为只读 | 广播产生的视图始终为只读（写入语义不明确——无法确定应写入原始元素还是所有广播位置；因此禁止写入，而非技术限制） |
| 算术运算符隐式支持广播 | 二元运算符自动触发广播（见 §12.6、§20） |
| 标量广播 | 标量视为 0 维数组（`Tensor0<A>`），可与任意维度数组广播 |

### 16.2 广播细节

| 规则 | 说明 |
|------|------|
| 维度对齐 | 从最右维度开始对齐，维度数不足的数组在左侧补 1 |
| 兼容条件 | 对应维度相等，或其中一个为 1 |
| 零步长语义 | size-1 维度广播时步长设为 0，逻辑上重复该维度的单个元素 |
| 广播视图可写性 | 广播产生的视图始终为只读（设计约束：广播视图使用零步长实现逻辑重复，写入语义不明确——无法确定应写入原始元素还是所有广播位置；因此禁止写入，而非技术限制） |
| 原地广播运算 | `a += b` 中 b 可被广播，a 不可被广播。左操作数须为连续内存且无零步长（`has_zero_stride() == false`），确保每个逻辑位置有唯一的物理写入目标。右操作数通过零步长广播到左操作数形状 |
| 标量广播 | 标量视为 0 维数组，可与任意维度数组广播 |

### 16.3 显式广播 API

| 项目 | 定义 |
|------|------|
| 签名 | `fn broadcast<D2>(&self, shape: D2) -> Result<TensorView<'_, A, D2::Dim>, BroadcastError> where D2: IntoDimension` |
| 语义 | 创建零步长视图，逻辑上将数组扩展到目标形状，无数据拷贝。仅构造新的 shape 和 strides 元数据 |
| 成功条件 | self 形状与目标形状按 §16.1 规则兼容（逐轴 size-1 或相等） |
| 失败 | 返回 `BroadcastError{left, right, conflicting_axis}`，指示不兼容的轴（见 §27.2） |
| 返回维度类型 | 由 `D2::Dim` 参数决定（传入 `Ix3` 返回 `TensorView<'_, A, Ix3>`，传入 `IxDyn` 返回动态维度） |
| 视图生命周期 | 绑定到 `&self`（与 §17 生命周期约定一致） |
| 布局标志 | `HAS_ZERO_STRIDE = true`（广播维度）；连续性标志按新步长重新计算；`ALIGNED` 继承源数组；`PADDED = false`；详见 §16.5.4 |

**广播视图步长计算规则**：

给定 self 的原始形状 `S`（左侧补 1 对齐到目标 ndim）和目标形状 `T`，步长按以下规则计算：

| 轴情况 | 步长 | 说明 |
|--------|------|------|
| 原始轴为左侧补的 1（self 无此轴） | 0 | 新增的虚拟轴 |
| 原始轴 size=1，目标轴 size>1 | 0 | 广播扩展轴 |
| 原始轴 size == 目标轴 size | 原始步长 | 保持原始步长 |

示例：`shape=[3, 1]`, `strides=[1, 3]` 广播到 `[3, 4]`：
- axis 0: size 3==3, stride=1
- axis 1: size 1→4, stride=0
- 结果：`strides=[1, 0]`, shape=[3, 4]

### 16.4 广播形状推导算法

**公共工具函数**：提供内部函数 `broadcast_shape(shape_a: &[usize], shape_b: &[usize]) -> Result<InlineArray<usize, 8>, BroadcastError>`，供所有需要广播形状推导的操作（逐元素运算、zip、batch 操作）复用。

**算法**：

```
fn broadcast_shape(sa: &[usize], sb: &[usize]) -> Result<InlineArray<usize, 8>, BroadcastError> {
    let ndim = max(sa.len(), sb.len());
    let mut result = InlineArray::with_capacity(ndim);
    for i in 0..ndim {
        let a = if i >= ndim - sa.len() { sa[sa.len() - ndim + i] } else { 1 };
        let b = if i >= ndim - sb.len() { sb[sb.len() - ndim + i] } else { 1 };
        match (a, b) {
            (_, 1) => result.push(a),      // b 为 1，取 a
            (1, _) => result.push(b),      // a 为 1，取 b
            (a, b) if a == b => result.push(a), // 相等
            _ => return Err(BroadcastError {
                left: sa.to_vec(),
                right: sb.to_vec(),
                conflicting_axis: i,
            }),
        }
    }
    Ok(result)
}
```

**关键语义**：结果轴大小取非 1 的值，而非 `max(a, b)`。这对 size-0 维度至关重要：`a=0, b=1` 时结果为 `0`（b=1 广播到 a=0），而 `max(0,1)=1` 是错误的。

**match arm 顺序不可调换**：算法中 match 的三个 arm 顺序具有语义重要性，不可调换：(1) `(_, 1)` 必须在 `(1, _)` 之前——当 `(a=1, b=1)` 时两者均可匹配，但无论匹配哪个结果均为 1，不影响正确性；(2) `(a, b) if a == b` 必须在最后——它需要显式排除 `(1, _)` 和 `(_, 1)` 的情况（否则 `(1, 1)` 会错误地走到相等分支，虽然结果碰巧正确但语义不清晰）。调换前两个 arm 的顺序不影响正确性，但第三个 arm 必须保持最后。

**广播形状推导示例**：

| shape_a | shape_b | 输出形状 | 说明 |
|---------|---------|----------|------|
| `[3, 4]` | `[3, 4]` | `[3, 4]` | 形状完全一致，无广播 |
| `[3, 1]` | `[1, 4]` | `[3, 4]` | 双方各一个轴被广播 |
| `[3, 4]` | `[4]` | `[3, 4]` | 右侧自动补 1 为 `[1, 4]` |
| `[3, 4]` | `[1]` | `[3, 4]` | 标量广播 |
| `[]` (0D) | `[3, 4]` | `[3, 4]` | 0D 广播（标量） |
| `[1, 3, 4]` | `[2, 1, 4]` | `[2, 3, 4]` | 多轴广播 |
| `[3, 4]` | `[5, 4]` | `BroadcastError` | axis 0 不兼容（3 ≠ 5，均非 1） |

### 16.5 边界行为

#### 16.5.1 0D（标量）广播

| 属性 | 行为 |
|------|------|
| 0D 参与广播 | 0D 数组等价于左侧补 1 直到目标 ndim，所有轴步长均为 0。例：`Tensor0<f64>` 广播到 `[3, 4]` 时，strides=[0, 0] |
| 标量运算 | 运算符重载中 `A`（标量）参与二元运算时，等价于构造 `Tensor0<A>` 后广播（见 §20.2）。`_s` 方法（§12.5）避免显式构造临时张量 |

#### 16.5.2 含 size-0 维度的广播

size-0 维度遵循标准广播规则：size-0 与 size-1 兼容（结果为 0），size-0 与 size-0 兼容（结果为 0），size-0 与 size-N（N>1 且 N≠0）不兼容（报错）。

| shape_a | shape_b | 输出形状 | 说明 |
|---------|---------|----------|------|
| `[0, 3]` | `[1, 3]` | `[0, 3]` | axis 0: b=1 广播到 a=0，结果 size=0 |
| `[0, 3]` | `[0, 1]` | `[0, 3]` | axis 0: 0==0 ✓；axis 1: 1→3 broadcast |
| `[0, 3]` | `[2, 3]` | `BroadcastError` | axis 0: 0 ≠ 2，均非 1 |
| `[0, 3]` | `[0, 4]` | `BroadcastError` | axis 1: 3 ≠ 4，均非 1 |
| `[3, 0]` | `[3, 1]` | `[3, 0]` | 结果为空数组（0 个元素），迭代器立即结束 |
| `[1, 0, 3]` | `[2, 1, 3]` | `[2, 0, 3]` | 多轴广播含 size-0，结果仍为空数组 |

**设计说明**：广播结果为空数组时，迭代器立即结束（零个元素），无实际计算开销。

#### 16.5.3 广播后维度数 > 6（静态维度溢出）

| 场景 | 行为 |
|------|------|
| 两个静态维度输入，广播后 ndim > 6 | 须使用 `IxDyn`。调用方须先将输入 `into_dyn()` 再广播，或使用 `broadcast` API 时传入 `IxDyn` 目标形状 |
| 运算符隐式广播 | 运算符要求两操作数维度类型相同（`D` 一致，见 §16.6），不产生维度提升。调用方须确保维度类型足够容纳广播结果 |
| 建议 | ndim > 6 的场景较少见；若预期可能超过 6 维，建议从一开始使用 `IxDyn` |

#### 16.5.4 广播视图的布局标志计算

| 标志 | 计算规则 | 典型值 |
|------|----------|--------|
| `HAS_ZERO_STRIDE` | 始终为 true（广播必有 size-1 轴被扩展） | true |
| `F_CONTIGUOUS` | 按新步长判定：ndim≤1 时为 true；ndim≥2 时要求所有非零步长轴满足 F-order 步长模式（跳过 stride=0 的轴）。通常因零步长存在而为 false | 通常 false |
| `C_CONTIGUOUS` | 同理，按 C-order 步长模式判定（跳过 stride=0 的轴） | 通常 false |
| `PADDED` | false（广播视图不引入填充） | false |
| `ALIGNED` | 继承源数组的 ALIGNED 状态（数据起始偏移量不变） | 与源一致 |
| `HAS_NEG_STRIDE` | 继承源数组（广播不产生新的负步长） | 与源一致 |

**设计说明**：广播视图通常不满足严格连续性（零步长打破了步长严格递增的模式），但 SIMD 路径可通过识别零步长轴进行特殊优化（见 §16.7）。

### 16.6 广播后维度类型推导

**运算符维度类型规则**：二元运算符（`Add/Sub/Mul/Div`）要求两操作数的维度类型 `D` 相同。广播仅在运行时由内部迭代器处理（见 §11.2 Zip），不改变返回维度类型。

| 场景 | 约束 | 示例 |
|------|------|------|
| 两个相同维度类型 | 正常，返回 `Tensor<A, D>` | `Tensor<f64, Ix2> + Tensor<f64, Ix2>` → `Tensor<f64, Ix2>` |
| 不同维度类型 | **编译错误**——须先显式转换 | `Tensor<f64, Ix2> + Tensor<f64, Ix3>` 须改为 `a.into_dyn() + b.into_dyn()` → `Tensor<f64, IxDyn>` |
| 隐式广播（相同 D） | 运行时按 §16.1 规则广播，维度类型不变 | `Tensor<f64, Ix2>{shape=[3,1]} + Tensor<f64, Ix2>{shape=[1,4]}` → `Tensor<f64, Ix2>{shape=[3,4]}` |

**与 ndarray 的设计对比**：ndarray 同样要求二元运算符两操作数维度类型一致，广播由 `Zip` 在运行时处理。Xenon 遵循相同设计，避免类型级编程的复杂性。

**显式广播 API 的维度类型**（§16.3）：`broadcast()` 方法通过泛型参数 `D2: IntoDimension` 允许输出不同的维度类型，调用方显式控制维度类型变化。

### 16.7 广播与 SIMD 路径

广播视图因零步长轴存在，不满足 `is_padded_contiguous()` 的严格步长检查（见 §7.4、§9.1.3），因此标准的"连续内存块 SIMD 加载"路径不适用。但广播场景有专用优化策略：

| 广播类型 | SIMD 策略 | 说明 |
|----------|----------|------|
| 单轴广播（某轴 size=1→size=N） | 使用标量广播指令 | 将单个元素加载到 SIMD 寄存器的所有 lane（如 `_mm256_set1_ps`），然后与连续加载的另一操作数进行向量运算。例：`[1]+[3,4,5,6,7,8]` → 广播 1 到所有 lane |
| 0D 广播（标量→ND） | 同上 | 0D 元素 `set1` 后与连续数组运算 |
| 多轴广播（≥2 个 size-1 轴同时广播） | 标量路径 | 多个轴同时广播时，SIMD broadcast 指令无法高效处理，回退标量。实现可按轴逐级展开优化，但当前版本不做强制要求 |
| 非连续操作数 + 广播 | 标量路径 | 两操作数均非连续时回退标量 |

**二元运算广播 SIMD kernel 推荐实现策略**：

1. 检测广播：通过 `has_zero_stride()` 判断哪些操作数有广播轴
2. 若恰好一个操作数连续（或宽松连续）且另一个操作数的广播轴全部为 size-1→size-N：使用 SIMD broadcast + 连续加载组合
3. 若两个操作数均无广播（形状一致）：使用标准 SIMD 路径（见 §9.1.3）
4. 其他情况：标量路径

**注**：以上为性能优化策略，不影响正确性。实现者可根据目标平台的 SIMD 指令集选择不同程度的优化。

---

## 17. 形状操作

**生命周期约定**：本节所有零拷贝操作返回的 `TensorView` / `TensorViewMut` 的生命周期绑定到 `&self`（不可变）或 `&mut self`（可变）。签名中省略生命周期标注 `'_` 以减少噪音，完整签名为 `TensorView<'a, A, D>` 其中 `'a` 与 `&'a self` 一致。

### 17.0 维度类型传播规则

形状操作中，维度数变化的操作的返回类型须使用 §3 中定义的关联类型精确表达：

**升维操作（ndim + 1）**：

| 操作 | 返回维度类型 | 关联类型 | 约束 |
|------|-------------|----------|------|
| unsqueeze | `D::Larger` | `Dimension::Larger`（§3.2） | Ix6 不可调用（Ix6::Larger = IxDyn，与"保持维度类型"矛盾），编译错误。通过 sealed sub-trait `UnsqueezeDim` 实现，见 §17.8 |
| stack | `D::Larger` | `Dimension::Larger`（§3.2） | Ix6 同理，编译错误。IxDyn → IxDyn（ndim + 1）。Ix6 的替代方案：`stack_dyn()` 返回 `Tensor<A, IxDyn>`，不要求 `D::Larger` |

**降维操作（ndim - 1）**：

| 操作 | 返回维度类型 | 关联类型 | 约束 |
|------|-------------|----------|------|
| squeeze | `<D as RemoveAxis>::Smaller` | `RemoveAxis::Smaller`（§3.3） | Ix0 不实现 RemoveAxis，编译时阻止 |
| index_axis | `<D as RemoveAxis>::Smaller` | `RemoveAxis::Smaller`（§3.3） | 同上 |

**维度数不变的操作**：返回类型仍为 `D`（reshape、transpose、permute、swapaxes、moveaxis、flatten、repeat、tile 等）。

**维度类型由参数决定的操作**：broadcast（§16.3）由 `D2: IntoDimension` 参数决定返回维度类型。

**静态维度 vs IxDyn 规则**：
- 静态维度（Ix0..Ix6）：升/降维通过关联类型在编译时确定精确的维度类型
- IxDyn：所有关联类型映射到自身（IxDyn::Larger = IxDyn，IxDyn::Smaller = IxDyn），维度数在运行时变化

**设计原则**：维度类型传播确保调用方无需 turbofish 标注即可获得正确的返回类型。对于需要跨静态/动态边界的情况（如 Ix6 的 stack），编译错误迫使调用方显式 `into_dyn()` 转换，避免静默降级。

### 17.1 操作分类

| 类型 | 操作 |
|------|------|
| 零拷贝 | reshape, transpose, slice, squeeze, unsqueeze, permute, broadcast, swapaxes, moveaxis, split/chunk, index_axis, unstack |
| 需拷贝 | cat, stack, pad, repeat/tile |
| 视情况 | flatten |

### 17.2 reshape 语义

| 属性 | 行为 |
|------|------|
| 操作 | `reshape(shape) -> Result<Tensor<A, D>>` 改变数组形状，元素总数须不变。形状合法时始终成功（连续零拷贝，非连续自动拷贝）；仅当 `-1` 推导失败时返回 `InvalidShape`。**维度类型 `D` 不变**——shape 参数的长度（ndim）须在编译时匹配 `D::NDIM`（静态维度通过类型系统强制；IxDyn 接受任意长度的 shape） |
| 连续输入 | 仅修改 shape/strides 元数据，复用底层存储（零拷贝）。返回的新 Tensor **独占**该存储（Owned 语义），原 Tensor 在 reshape 后不再可用（移动语义）。"共享底层缓冲区"仅发生在视图层面（如原始数据被 ArcRepr 持有），Owned-to-Owned reshape 通过移动所有权实现零拷贝 |
| 非连续输入 | 自动拷贝为 C-contiguous 布局后再 reshape，返回新分配的 Tensor |
| 输出布局 | 连续路径：继承输入的连续性方向（F-contiguous → F-order strides；C-contiguous → C-order strides）；非连续路径：输出为 C-contiguous |
| -1 自动推导 | shape 中至多一个维度可为 `-1`，由元素总数除以其余维度乘积自动推导；多个 `-1`、推导结果非整数、或元素总数不匹配时返回 `Err(InvalidShape)`。**API 签名**：静态维度接受 `&[isize]` 切片（`-1isize` 表示自动推导，合法正值自动转为 `usize`）；IxDyn 接受 `&IxDyn` 或 `Vec<isize>`（内部同样用 `-1isize` 标记推导维度） |
| 空数组 | 允许 reshape 为任意元素数为 0 的形状（如 `(0, 5)` → `(0, 3, 2)` 或 `(0,)`） |
| 维度类型 | 不改变维度类型（D 不变）。静态维度（如 Ix3）reshape 后仍为 Ix3；IxDyn reshape 后仍为 IxDyn。需要改变维度类型时使用 `into_dimension::<D2>()`（§3.4） |
| 静态维度约束 | 静态维度通过签名强制 shape 的 ndim 匹配 `D::NDIM`。实现方式：`reshape` 接受 `impl IntoDimension<Dim = D>` 参数（§3.4），编译器在调用点验证 shape 维度数与 D 一致。即 `Tensor<A, Ix3>.reshape([3])` 是编译错误（`[usize; 1]` 的 `IntoDimension::Dim = Ix1 ≠ Ix3`），须先 `into_dimension::<Ix1>()` 转换维度类型。IxDyn 接受任意长度的 shape，无此约束 |

> **设计决策**：统一为单一 `reshape()` 方法，返回 `Result`：形状合法时始终成功（连续零拷贝、非连续自动拷贝），仅 `-1` 推导失败时返回错误。移除原 `reshape_owned()` 和双 API 设计。理由：(1) 消除用户在运行时根据内存布局选择 API 的认知负担，与 §1.2 "清晰抽象，无隐式行为" 一致；(2) 连续输入自动零拷贝、非连续输入自动拷贝，行为符合最小惊讶原则；(3) `-1` 推导失败是合法的运行时错误场景（shape 可能来自用户输入），用 Result 而非 panic 返回。

> **关于"无隐式行为"原则与 reshape 自动拷贝的一致性**：reshape 对非连续输入的自动拷贝**不违反** §1.2 "无隐式行为"原则。"无隐式行为"指 API 不应有隐藏的语义副作用（如静默修改输入数据、隐式全局状态变更），而非禁止内部内存分配——所有构造操作（`zeros`/`ones`/`map` 等）都需要分配内存，这是构造语义的固有部分。reshape 的语义是"返回指定形状的新 Tensor"，其分配行为与其他构造方法一致，且通过 `is_contiguous()` 预检和性能提示文档已充分告知调用方。相比之下，若 reshape 对非连续输入静默修改源数组的数据（而非复制），才是真正的"隐式行为"。

> **性能提示**：`reshape()` 对非连续输入会隐式执行完整数据拷贝。性能敏感场景下，调用方可通过 `is_contiguous()` 预检：若返回 `true`，后续 `reshape()` 保证零拷贝；若返回 `false`，调用方可决定是否先调用 `as_f_contiguous()` / `as_c_contiguous()`（见 §22）获取零拷贝视图再操作，或接受拷贝开销。

### 17.3 transpose 语义

| 属性 | 行为 |
|------|------|
| 操作 | `transpose() -> TensorView<A, D>` 反转所有轴顺序 |
| 零拷贝 | 始终零拷贝，仅翻转 strides 和 shape 数组 |
| 输出维度 | ndim 不变，shape 和 strides 均反转 |
| 步长变换 | `strides_out[i] = strides_in[ndim - 1 - i]` |
| 连续性 | F-contiguous 输入转置后变为 C-contiguous，反之亦然 |
| 1D 数组 | 转置为 no-op（shape 和 strides 不变） |
| 0D 数组 | 转置为 no-op |

**指定轴转置 `transpose(axes)`：**

| 属性 | 行为 |
|------|------|
| 操作 | `transpose(axes: &[usize]) -> TensorView<A, D>` 按指定排列重排轴 |
| 约束 | `axes` 须为 `[0, ndim)` 的排列（长度 == ndim，无重复，无遗漏），否则 panic |
| 步长变换 | `strides_out[i] = strides_in[axes[i]]` |

### 17.4 permute 语义

| 属性 | 行为 |
|------|------|
| 操作 | `permute(axes: &[usize]) -> TensorView<A, D>` 按指定排列重排轴（与 `transpose(axes)` 语义等价） |
| 零拷贝 | 始终零拷贝 |
| 与 transpose 关系 | `permute` 为 `transpose(axes)` 的别名，提供 NumPy/PyTorch 兼容命名 |
| 推荐用法 | 推荐使用 `permute`，命名更直观（Permute axes to this order），`transpose(axes)` 为底层实现保留 |

### 17.5 swapaxes 语义

| 属性 | 行为 |
|------|------|
| 操作 | `swapaxes(a: usize, b: usize) -> TensorView<A, D>` 交换两个轴 |
| 零拷贝 | 始终零拷贝，仅交换两个轴的 shape 和 strides 值 |
| 约束 | `a` 和 `b` 须在 `[0, ndim)` 范围内，否则 panic |
| a == b | no-op |
| 连续性 | 交换 F-contiguous 的首尾轴 → 变为非连续（中间轴交换同理） |

### 17.6 moveaxis 语义

| 属性 | 行为 |
|------|------|
| 操作 | `moveaxis(source: usize, destination: usize) -> TensorView<A, D>` 将轴从 source 移到 destination 位置 |
| 零拷贝 | 始终零拷贝，等效于构造特定排列后调用 permute |
| 约束 | source 和 destination 须在 `[0, ndim)` 范围内，否则 panic |
| 与 swapaxes 区别 | moveaxis 仅移动一个轴，其余轴顺序保持不变；swapaxes 仅交换两个轴 |

### 17.7 squeeze 语义

| 属性 | 行为 |
|------|------|
| 操作 | `squeeze(axis: usize) -> TensorView<A, <D as RemoveAxis>::Smaller>` 消除指定轴（该轴长度须为 1）。返回类型通过 `RemoveAxis::Smaller` 关联类型（§3.3）在编译时确定：`Ix3::Smaller = Ix2`，`IxDyn::Smaller = IxDyn` |
| 零拷贝 | 始终零拷贝 |
| 输出维度 | ndim - 1，shape 和 strides 移除该轴 |
| 约束 | 指定轴长度须为 1，否则返回 `InvalidShape` 错误 |
| 多轴挤压 | `squeeze_axes(axes: &[usize])` 同时消除多个 size-1 轴。**返回类型因维度类型而异**：(1) **IxDyn**：返回 `TensorView<A, IxDyn>`，在签名中明确 `IxDyn` 输出。(2) **静态维度**（Ix1-Ix6）：返回 `TensorView<A, IxDyn>`，**降维数量在编译时无法确定**（`axes.len()` 是运行时值），无法递归应用 `RemoveAxis` 推导静态输出类型。若需保留静态维度类型，使用 `squeeze()`（单轴）多次调用或链式调用。此设计选择避免了为每种 `(输入维度, 挤压轴数)` 组合提供不同方法签名 |
| 0D/1D | Ix0 不实现 RemoveAxis trait，编译时阻止 squeeze；Ix1 squeeze 指定唯一轴后得到 Ix0 |
| 动态维度 | IxDyn 支持 squeeze，输出仍为 IxDyn（ndim 减少） |

### 17.8 unsqueeze 语义

| 属性 | 行为 |
|------|------|
| 操作 | `unsqueeze(axis: usize) -> TensorView<A, D::Larger>` 在指定位置插入长度为 1 的新轴。返回类型通过 `Dimension::Larger` 关联类型（§3.2）在编译时确定：`Ix2::Larger = Ix3`，`IxDyn::Larger = IxDyn` |
| 零拷贝 | 始终零拷贝，仅扩展 shape（插入 1）和 strides（插入适当值） |
| 输出维度 | ndim + 1 |
| 新轴步长 | 设为 0（与广播语义一致：size-1 维度的步长为 0，表示逻辑上重复该单个元素）。插入后数组的连续性不变（F-contiguous 输入仍 F-contiguous，C-contiguous 输入仍 C-contiguous） |
| 返回可写性 | **返回不可变视图 `TensorView`（非 `TensorViewMut`）**。理由：新轴步长为 0 意味着该轴所有位置映射到同一物理内存，写入会导致同一位置被静默覆盖（`v[[0,0,0]] = 1.0; v[[0,1,0]] = 2.0` 实际写入同一地址）。与广播视图的设计一致（§16.2 广播产生的零步长视图为只读）。若需写入，须先 `.to_owned()` 获取 Owned 数组 |
| axis 范围 | `0 <= axis <= ndim`（可在最后一轴之后插入） |
| 静态维度约束 | Ix6 不可 unsqueeze（Ix6::Larger = IxDyn，与"保持静态维度类型"矛盾），编译错误。IxDyn 无此限制。**实现方式**：定义 sealed sub-trait `UnsqueezeDim: Dimension`（或等效机制），仅 Ix0~Ix5 和 IxDyn 实现该 trait，`unsqueeze` 方法要求 `where D: UnsqueezeDim`。Ix6 未实现此 trait，调用时编译器在 trait 求解阶段报错 |

### 17.9 cat 语义

| 属性 | 行为 |
|------|------|
| 操作 | `cat(axis, &[TensorView], order: Option<Order>)` 沿指定轴拼接多个数组，返回新 Owned 数组（需拷贝）。`order` 为 `None` 时默认 C-contiguous，`Some(Order::F)` 时输出 F-contiguous |
| 维度约束 | 所有输入数组的 ndim 须相同；除拼接轴外，其余各轴长度须完全一致 |
| 拼接轴长度 | 结果在拼接轴上的长度等于所有输入在该轴上长度之和 |
| 输入数量 | 至少 1 个输入；0 个输入返回 `InvalidShape` 错误 |
| 空数组输入 | 允许部分输入在拼接轴上长度为 0，等价于跳过该输入 |
| 元素顺序 | 按输入列表顺序排列，第 i 个输入的元素在结果中排在第 i-1 个之后 |
| 输出布局 | 默认 C-contiguous；可通过参数指定 F-contiguous |
| 返回类型 | `Tensor<A, D>`（Owned） |

### 17.10 stack 语义

| 属性 | 行为 |
|------|------|
| 操作 | `stack(axis, &[TensorView], order: Option<Order>)` 沿新轴堆叠多个数组，返回新 Owned 数组（需拷贝）。`order` 为 `None` 时默认 C-contiguous，`Some(Order::F)` 时输出 F-contiguous |
| 维度约束 | 所有输入数组的形状须完全一致（shape 相同） |
| 结果维度 | ndim = 输入 ndim + 1，新轴插入在 `axis` 位置 |
| 新轴长度 | 等于输入数组的数量 |
| axis 范围 | `0 <= axis <= input_ndim`（可在最后一轴之后插入） |
| 输入数量 | 至少 1 个输入；0 个输入返回 `InvalidShape` 错误 |
| 元素顺序 | 沿新轴的第 i 个切片对应第 i 个输入数组 |
| 输出布局 | 默认 C-contiguous；可通过参数指定 F-contiguous |
| 返回类型 | `Tensor<A, D::Larger>`（Owned），通过 `Dimension::Larger` 关联类型（§3.2）在编译时确定升维后的精确类型。如输入 `Ix2` → 返回 `Ix3`；输入 `IxDyn` → 返回 `IxDyn` |
| Ix6 上限 | 对 Ix6 数组调用 stack 时，`Ix6::Larger = IxDyn`，编译错误（stack 保持维度类型不变的要求与 Ix6::Larger = IxDyn 矛盾，见 §17.0）。如需在 Ix6 上堆叠，须先 `into_dyn()` 转为 IxDyn。或者使用 `stack_dyn(axis, &[...])` 方法，显式返回 `Tensor<A, IxDyn>`（不要求 `D::Larger`，仅要求 `D: Dimension`），适用于编译时维度类型不确定的场景 |
| 与 cat 的关系 | `stack(axis, arrays)` 等价于先对每个输入 `unsqueeze(axis)` 再 `cat(axis, ...)` |

**stack_dyn — 动态维度堆叠**

| 属性 | 行为 |
|------|------|
| 操作 | `stack_dyn(axis: usize, arrays: &[TensorView<'_, A, D>], order: Option<Order>) -> Tensor<A, IxDyn>` 沿新轴堆叠多个数组，返回 IxDyn 类型（不要求 `D::Larger`，仅要求 `D: Dimension`） |
| 适用场景 | 输入维度类型为 `Ix6`（`stack` 编译错误）或编译时维度类型不确定的场景 |
| 其他行为 | 与 `stack` 一致（维度约束、元素顺序、输出布局等），仅返回类型为 `IxDyn` |

### 17.11 flatten 语义

提供两个独立方法，避免单函数返回两种类型的歧义：

> **决策指南**：性能敏感场景优先使用 `flatten_view()`——成功时零拷贝（通过 `is_contiguous()` 可预判是否成功）；不关心性能或需要始终成功的场景使用 `flatten()`。两个方法的关系类似于 `as_slice()` vs `to_vec()`：视图方法零拷贝但可能失败，owned 方法始终成功但可能分配。

**flatten_view — 零拷贝（可能失败）**

| 属性 | 行为 |
|------|------|
| 签名 | `fn flatten_view(&self) -> Result<TensorView<'_, A, Ix1>, LayoutMismatch>` |
| 成功条件 | 输入为连续内存（F-contiguous 或 C-contiguous）时，返回 1D 视图（零拷贝） |
| 失败条件 | 输入为非连续内存时，返回 `LayoutMismatch` 错误 |
| 展平顺序 | F-contiguous 输入按列优先顺序；C-contiguous 输入按行优先顺序 |
| 空数组 | 返回长度为 0 的 1D 视图 |

**flatten — 始终成功（可能拷贝）**

| 属性 | 行为 |
|------|------|
| 签名 | `fn flatten(&self) -> Tensor<A, Ix1>` |
| 行为 | 始终返回 Owned 数组。连续输入等价于 `flatten_view().to_owned()`；非连续输入按 C-order 拷贝后展平 |
| 可指定顺序 | 提供 `flatten_order(order)` 变体，可显式指定 F/C 展平顺序 |
| 空数组 | 返回长度为 0 的 1D Owned 数组 |

### 17.12 index_axis 语义

| 属性 | 行为 |
|------|------|
| 操作 | `index_axis(axis, index)` 沿指定轴取单个切片，返回 `TensorView<'_, A, <D as RemoveAxis>::Smaller>`（降维视图，ndim - 1）。维度类型通过 `RemoveAxis::Smaller`（§3.3）在编译时确定 |
| axis 参数 | 须在 `[0, ndim)` 范围内，不支持负索引（与 `s![]` 宏不同）。理由：axis 负索引的语义（从末尾计数）在静态维度类型系统中无法表达返回类型的维度映射（与 `remove_axis` 的 `Reduced` 类型同理），统一要求非负值消除了歧义 |
| index 参数 | 须在 `[0, shape[axis])` 范围内，不支持负索引。需要末尾索引时使用 `shape[axis] - 1` |
| 内存布局 | 返回的视图共享源数组底层存储 |
| 对齐继承 | 视图的实际对齐取决于偏移后的起始地址，布局标志须反映实际对齐状态 |
| BLAS 兼容性 | 从 3D batch tensor 中沿最外层 batch 轴索引取出的 2D 视图，若源数组 F-contiguous 则保持 F-contiguous 和原 LDA |
| 连续性 | 沿最外层轴（F-order 下为最后一轴）索引时保持连续性；沿内层轴索引可能导致非连续 |

### 17.13 unstack 语义

| 属性 | 行为 |
|------|------|
| 操作 | `unstack(axis)` 沿指定轴拆分为 n 个降维视图（ndim - 1），n 为该轴长度 |
| 返回类型 | `Vec<TensorView<'a, A, <D as RemoveAxis>::Smaller>>`，长度等于指定轴的 size。生命周期 `'a` 绑定到 `&self`，返回的视图借用源数组（§17.0 维度类型传播规则） |
| 内存布局 | 每个视图共享源数组底层存储，零拷贝 |
| 与 index_axis 关系 | `unstack(axis)[i]` 等价于 `index_axis(axis, i)` |
| 空轴 | 轴长度为 0 时返回空 Vec |

### 17.14 split/chunk 语义

| 属性 | 行为 |
|------|------|
| split(axis, indices) | 沿指定轴按索引列表分割，返回 `Vec<TensorView<'_, A, D>>`，零拷贝 |
| split indices 约束 | indices 须为严格递增的非负整数序列，每个值 < 轴长度。重复、非递减、或越界的索引返回 `InvalidShape` 错误。空 indices 返回包含完整轴的单个视图 |
| chunk(axis, n_chunks) | 沿指定轴均匀分割为 n 块；若轴长度不能整除，前 `(len % n)` 块各多 1 个元素 |
| 返回类型 | `Vec<TensorView<'_, A, D>>`，每个视图共享源数组底层存储 |
| n_chunks = 0 | 返回空 Vec |
| n_chunks > 轴长度 | 返回轴长度个大小为 1 的块（多余的 chunk 数被忽略） |
| 空轴 | 轴长度为 0 时，split 返回 1 个空视图（indices 为空时）或按 indices 分割的空视图；chunk 返回 n_chunks 个空视图 |

### 17.15 pad 语义

| 属性 | 行为 |
|------|------|
| pad(widths, mode) | 沿各轴两侧填充，返回新 Owned 数组（需拷贝） |
| widths 参数 | 每轴一对 (before, after)，指定前后填充宽度 |
| mode: Constant(value) | 用指定常量填充 |
| mode: Edge | 用最近的边缘元素重复填充。对于轴长度为 N 的数组，pad_before 位置的填充值等于 index[0]（首元素），pad_after 位置的填充值等于 index[N-1]（末元素）。等效于 NumPy 的 `mode='edge'` |
| mode: Reflect | 镜像反射填充（不含边缘元素）。具体语义：关于边界反射，填充位置 `i` 取 `data[axis_len - 1 - i]`（before 侧）或 `data[axis_len - 1 - (i - pad_after_offset)]`（after 侧），不包含边界元素本身。等效于 NumPy 的 `mode='reflect'`。注意与 `symmetric` 模式的区别：`symmetric` 包含边界元素（等效 NumPy `mode='symmetric'`），当前版本未提供 symmetric 模式 |
| 零宽度填充 | 等价于拷贝，不改变形状 |
| Reflect 边界约束 | 填充宽度不得超过该轴数据长度减 1（即 `pad_before < axis_len` 且 `pad_after < axis_len`），否则 panic。理由：Reflect 不含边缘元素，填充宽度 ≥ axis_len 时无有效源元素可反射 |

### 17.16 windows 语义

| 属性 | 行为 |
|------|------|
| 签名 | `fn windows(&self, window_size: D) -> Windows<'_, A, D>` |
| 操作 | 沿所有轴生成滑动窗口视图，返回迭代器 |
| 输出形状 | 每个窗口的形状为 `window_size`；窗口总数沿每轴为 `shape[i] - window_size[i] + 1`，总窗口数为各轴窗口数之积 |
| 步幅 | 窗口间的步幅等于源数组各轴步长（即步长 1 个元素）；窗口内部步幅继承源数组步幅 |
| 边界行为 | 不产出不完整窗口——当 `shape[i] < window_size[i]` 时，该轴窗口数为 0，总窗口数为 0，迭代器立即结束 |
| 内存布局 | 每个窗口为 `TensorView`，共享源数组底层存储，零拷贝 |
| 约束 | `window_size` 各轴须 > 0，否则 panic；`window_size` 的 ndim 须与源数组一致。不支持自定义步幅（当前版本步幅固定为 1），自定义步幅需求可通过 `.slice()` 配合迭代实现 |
| window_size > shape | 当 `window_size[i] > shape[i]` 时，该轴窗口数为 0，总窗口数为 0，迭代器立即结束（等价于空迭代器，不 panic） |
| 空数组 | 源数组任意轴长度为 0 时，迭代器立即结束（窗口数为 0） |
| 迭代顺序 | 按 F-order（列优先，最右轴变化最快）产出窗口 |
| 迭代器 trait | 实现 `Iterator`（Item = TensorView）、`ExactSizeIterator`、`FusedIterator` |

### 17.17 repeat/tile 语义

提供两个独立方法，语义不同（与 NumPy `np.repeat` / `np.tile` 对应）：

**repeat — 沿轴逐元素重复**

| 属性 | 行为 |
|------|------|
| 签名 | `fn repeat(&self, axis: usize, count: usize) -> Tensor<A, D>` |
| 操作 | 沿指定轴将每个元素重复 `count` 次，返回新 Owned 数组。如 `[1, 2, 3].repeat(0, 2)` → `[1, 1, 2, 2, 3, 3]` |
| 约束 | `axis` 须在 `[0, ndim)` 范围内；`count = 0` 时指定轴长度变为 0，结果为空数组 |
| 输出形状 | 指定轴长度 = 输入轴长度 × count，其余轴不变 |

**tile — 沿各轴重复整个数组**

| 属性 | 行为 |
|------|------|
| 签名 | `fn tile(&self, reps: &[usize]) -> Tensor<A, D>` |
| 操作 | 沿各轴重复整个数组 `reps[i]` 次（类似 NumPy `np.tile`） |
| reps 参数 | 每轴一个重复次数；reps 长度不足时左侧补 1。静态维度要求 `reps.len() <= D::NDIM`（超出时编译错误或 panic，实现可选），补 1 后长度须等于 ndim |
| reps 含 0 | 对应轴长度变为 0，结果为空数组 |
| reps 全为 1 | 等价于拷贝 |

---

## 18. 索引操作

### 18.1 索引类型

| 类型 | 说明 |
|------|------|
| 多维索引 | `[i, j, k]` 形式，i, j, k 为元素索引 |
| 范围索引 | 切片语法 |
| 切片宏 | `s![]` 形式（语义见 §18.9） |
| 高级索引 | take, take_along_axis, mask, compress, put, argwhere/nonzero |
| 条件选择 | select(condition, x, y) |

### 18.2 take 语义

| 属性 | 行为 |
|------|------|
| 签名 | `fn take(&self, indices: &Tensor1<usize>, axis: Option<usize>) -> Result<Tensor<A, IxDyn>>` |
| 操作 | 沿指定轴按索引数组提取元素，返回新数组 |
| axis=None | 将输入展平后按一维索引取值 |
| axis=Some(i) | 沿第 i 轴取值，其余轴不变，指定轴长度变为 indices.len() |
| 约束 | 索引值须 < 轴长度，否则返回 `IndexOutOfBounds` 错误 |
| 拷贝 | 始终拷贝，返回 Owned |

**返回类型说明**：`take` 统一返回 `Tensor<A, IxDyn>`（动态维度），即使输入为静态维度。理由：`axis: Option<usize>` 在运行时决定是否展平，输出维度数量无法在编译时确定。若需保留静态维度信息，使用 `take_along_axis`（指定轴，维度类型不变）。

### 18.3 take_along_axis 语义

| 属性 | 行为 |
|------|------|
| 签名 | `fn take_along_axis(&self, indices: &Tensor<usize, D>, axis: usize) -> Result<Tensor<A, D>, IndexOutOfBounds>` |
| 操作 | 沿指定轴按索引数组取值。除指定轴外，`indices` 其余轴长度须与 `self` 对应轴长度相等；`indices` 在指定轴上的长度可与 `self` 不同（决定输出在该轴的长度） |
| 约束 | 除指定轴外，indices 其余轴长度须与 self 相同，否则返回 `ShapeMismatch` 错误；索引值 < 指定轴长度，否则返回 `IndexOutOfBounds` 错误 |
| 典型用途 | `argmax` 结果的反向索引：`a.take_along_axis(&a.argmax_keepdim(axis), axis)?` |
| 拷贝 | 始终拷贝，返回 Owned |

### 18.4 mask 语义

| 属性 | 行为 |
|------|------|
| 签名 | `fn mask(&self, mask: &Tensor<bool, D>) -> Tensor1<A>` |
| 操作 | 按布尔掩码提取元素，返回一维数组 |
| 约束 | mask 形状须与 self 形状完全一致，否则返回 `ShapeMismatch` 错误 |
| 返回值 | Tensor1<A>，长度为 mask 中 true 的数量，元素按 F-order 遍历顺序收集 |
| 拷贝 | 始终拷贝 |

### 18.5 compress 语义

| 属性 | 行为 |
|------|------|
| 签名 | `fn compress(&self, mask: &Tensor1<bool>, axis: usize) -> Tensor<A, D>` |
| 操作 | 沿指定轴按布尔掩码提取，保留维度 |
| 约束 | mask 长度须等于指定轴长度，否则返回 `ShapeMismatch` 错误 |
| 返回值 | 维度类型与 self 相同（`D`），指定轴长度为 mask 中 true 的数量。其余轴长度不变 |
| 与 mask 区别 | mask 展平为一维；compress 保留维度结构 |
| 拷贝 | 始终拷贝 |

### 18.6 put 语义

| 属性 | 行为 |
|------|------|
| 签名 | `fn put(&mut self, indices: &Tensor1<usize>, values: &Tensor1<A>, axis: Option<usize>) -> Result<()>` |
| 操作 | 按索引数组将值写入指定位置，原地修改 |
| 约束 | 索引值须 < 轴长度（axis=Some 时为指定轴长度，axis=None 时为元素总数）；values 长度须与 indices 长度相等 |
| 错误 | 索引越界返回 `IndexOutOfBounds`（含越界索引值、轴长度、轴号）；values 与 indices 长度不匹配返回 `ShapeMismatch` |
| 与 take 对称 | `put` 为 `take` 的逆操作 |
| 重复索引 | 当 indices 包含重复值时，同一位置被多次写入，最终值为最后一次写入的值（last-write-wins），与 NumPy `np.put` 行为一致 |

### 18.7 argwhere 语义

| 属性 | 行为 |
|------|------|
| 签名 | `fn argwhere(&self) -> Tensor2<usize> where A: Element` |
| 操作 | 返回非零（`!= A::default()`）元素的多维索引。对数值类型，`default()` 等于 `zero()`；对 `bool`，`default()` 为 `false`（即查找 `true`）；对 `Complex`，`default()` 为 `Complex(0, 0)` |
| 返回值 | Tensor2<usize>，形状为 (n_nonzero, ndim) |
| 空数组 | 若无非零元素，返回形状 (0, ndim) |

### 18.8 nonzero 语义

| 属性 | 行为 |
|------|------|
| 签名 | `fn nonzero(&self) -> Vec<Tensor1<usize>> where A: Element` |
| 操作 | 返回各轴的非零索引，类似 NumPy `np.nonzero()` |
| 返回值 | Vec 长度为 ndim，每个元素为该轴上非零位置的索引数组 |
| 空数组 | 各轴均返回空 Tensor1 |

**argwhere vs nonzero 返回类型差异）**：两者提供同一数据的不同视角——`argwhere` 返回 `Tensor2<usize>`（行=坐标点，列=轴），适合逐点访问；`nonzero` 返回 `Vec<Tensor1<usize>>`（每轴一个索引数组），适合按轴花式索引（如 `a[nonzero_result]`）。这两种返回类型对应 NumPy 中 `np.argwhere` 和 `np.nonzero` 的设计，保持 API 兼容性。两者之间可通过转置互转：`argwhere` 的结果转置后按行拆分即得 `nonzero` 的结果。

**"非零"判定规则**（`argwhere`/`nonzero` 共用）：

| 元素类型 | "非零"判定 | 说明 |
|---------|-----------|------|
| 整数（i8..i64, u8..usize） | `x != 0` | `default()` = `0` |
| 浮点（f32, f64） | `x != 0.0`（NaN 视为非零） | `default()` = `0.0`；`NaN != 0.0` 为 `true`，故 NaN 被视为非零 |
| `bool` | `x != false`，即 `x == true` | `default()` = `false` |
| `Complex<T>` | `x != Complex(T::zero(), T::zero())` | 实部或虚部非零即为非零；`Complex(NaN, 0)` 视为非零 |

### 18.9 切片宏 s![]语义

`s![]` 宏用于构造多维切片描述符，语法糖对应 `SliceInfo` 类型。

> **兼容性提示**：`s![]` 宏名称可能与 ndarray 的同名宏冲突。如需与 ndarray 共存，可使用 crate-qualified 路径 `xenon::s![]`，或改用 `slice![]` 别名。

**语法元素**

| 语法 | 含义 | 示例 |
|------|------|------|
| `i` | 单个索引（降维） | `s![2, ..]` — 第 0 轴取索引 2，第 1 轴全选 |
| `a..b` | 范围（左闭右开） | `s![1..4]` — 索引 1, 2, 3 |
| `a..` | 从 a 到末尾 | `s![2..]` |
| `..b` | 从开头到 b（不含） | `s![..3]` |
| `..` | 全选 | `s![..]` |
| `a..b;step` | 带步长的范围 | `s![0..6;2]` — 索引 0, 2, 4 |
| `..;step` | 全选带步长 | `s![..;-1]` — 反转 |

**负索引**

| 规则 | 说明 |
|------|------|
| 负起始/终止 | 从末尾倒数，`-1` 表示最后一个元素。`s![-3..-1]` 取倒数第 3、第 2 个元素 |
| 负步长 | 反转遍历方向。`s![..;-1]` 等价于 `flip`；`s![4..1;-1]` 取索引 4, 3, 2 |
| 越界裁剪 | 起始/终止超出轴长度时自动裁剪到有效范围（与 NumPy 一致），不 panic |

**降维规则**

| 场景 | 行为 |
|------|------|
| 单个索引 `s![i, ..]` | 该轴被消除，结果 ndim - 1 |
| 范围 `s![a..b, ..]` | 该轴保留，长度为 `ceil((b-a) / step)` |
| 全选 `s![.., ..]` | 该轴保留，长度不变 |

**返回类型**

| 场景 | 返回 |
|------|------|
| 不含单索引 | `TensorView<A, D>`（同维度） |
| 含单索引 | `TensorView<A, D'>`，`D'` 为降维后的维度类型 |
| 可变切片 | `TensorViewMut<A, D'>` |

**SliceInfo 类型定义**

`s![]` 宏展开为 `SliceInfo` 类型，描述多维切片描述符：

```
pub struct SliceInfo<T: SliceNextDim, D: Dimension, DOut: Dimension> {
    // 内部存储各轴的切片描述（索引/范围/全选）
    // T: 递归类型参数，编码各轴切片类型链
    // D: 输入维度类型
    // DOut: 输出维度类型（降维后）
    inner: T,
    _phantom: PhantomData<(D, DOut)>,
}
```

**Index trait 实现**

```
impl<S, A, D, DOut, T> Index<SliceInfo<T, D, DOut>> for TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    DOut: Dimension,
    T: SliceNextDim,
{
    type Output = TensorView<A, DOut>;
    // 返回切片视图（零拷贝）
}

impl<S, A, D, DOut, T> IndexMut<SliceInfo<T, D, DOut>> for TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    D: Dimension,
    DOut: Dimension,
    T: SliceNextDim,
{
    // 返回可变切片视图（零拷贝）
}
```

**维度推导映射**

| 输入维度 | 切片描述示例 | 输出维度 | 说明 |
|---------|------------|---------|------|
| Ix2 | `s![2, ..]` | Ix1 | 第 0 轴单索引降维 |
| Ix3 | `s![1..4, .., 2]` | Ix1 | 第 0、2 轴单索引降维，第 1 轴保留 |
| Ix3 | `s![1..4, .., ..]` | Ix3 | 全为范围，无降维 |
| Ix3 | `s![2]` | Ix2 | 单索引，省略的后缀轴全选 |
| IxDyn | `s![2, ..]` | IxDyn | 动态维度始终输出 IxDyn |
| Ix4 | `s![.., .., 1..3;2, ..]` | Ix4 | 全为范围/步长范围，无降维 |

**边界行为**

| 场景 | 行为 |
|------|------|
| 空范围（a >= b 且 step > 0） | 返回该轴长度为 0 的视图 |
| step = 0 | 静态维度：宏展开时对**字面量** `0` 报编译错误（仅检测 `;0` 的 token 模式，无法检测变量值为 0 的情况）；动态维度：运行时 panic。注意：若 step 为非常量表达式（如 `s![0..6;n]`，运行时 `n == 0`），静态维度也会在运行时 panic |
| 轴数与 ndim 不匹配 | 运行时 panic（所有维度类型）。`macro_rules!` 宏在展开时无法获知维度类型信息（类型检查在宏展开之后），因此静态维度也无法在编译时检测轴数不匹配。未来可考虑 proc macro 实现编译时检查 |

### 18.10 select 语义

| 属性 | 行为 |
|------|------|
| 签名 | `fn select(condition: &Tensor<bool, D>, x: &Tensor<A, D>, y: &Tensor<A, D>) -> Tensor<A, D>`（三者维度类型参数 `D` 须在编译时完全相同，形状须一致或可广播为同一形状。若需不同维度类型，须先手动转换维度类型，如 `.into_dimension::<IxDyn>()`） |
| 操作 | `select(condition, x, y)` 按条件逐元素选择，condition 为 true 取 x，否则取 y |
| condition 类型 | bool 数组（或可广播为目标形状的 bool 数组） |
| x/y 类型约束 | x 和 y 须为同类型 Tensor（元素类型相同）或同类型标量值（`A`）。标量值等价于构造 `Tensor0<A>` 后广播。x 和 y 不支持隐式类型转换——若 x 为 `Tensor<f64, D>` 则 y 也须为 `f64` 类型 |
| 广播 | condition、x、y 三者须形状一致或可广播，结果形状为广播后形状 |
| 返回维度 | 输出维度类型与输入 `D` 相同（三者须为同一 `D`）。形状为三者广播后的形状 |
| 返回类型 | 新分配的 Owned 数组 |

---

## 19. 构造操作

### 19.1 构造方法

| 方法 | 说明 |
|------|------|
| zeros/ones/full | 支持指定 Order |
| eye/identity/diag | 单位矩阵和对角矩阵（见下方语义表） |
| from_vec/from_slice | 从一维数据源构造一维 Tensor（见 §19.7） |
| from_shape_vec/from_shape_slice | 从数据源 + 指定形状构造多维 Tensor（见 §19.7） |
| arange/linspace/logspace | 序列生成 |
| unsafe empty_uninit | 返回 `UninitTensor<A, D>`，未初始化内存（见 §19.3 empty_uninit 语义） |

### 19.2 eye/identity/diag 语义

| 方法 | 签名 | 说明 |
|------|------|------|
| eye | `fn eye(n: usize) -> Tensor<A, Ix2>` | n×n 单位矩阵，对角线为 `one()`，其余为 `zero()` |
| identity | `fn identity(n: usize) -> Tensor<A, Ix2>` | `eye` 的别名，语义等价 |
| diag | `fn diag(diag: &Tensor1<A>) -> Tensor<A, Ix2>` | 以一维数组为主对角线构造方阵，非对角位置为 `zero()`（等价于 `diag_with_offset(diag, 0)`） |
| diag_with_offset | `fn diag_with_offset(diag: &Tensor1<A>, offset: isize) -> Tensor<A, Ix2>` | 偏移对角线：`offset > 0` 上移，`offset < 0` 下移；输出形状为 `(n + |offset|) × (n + |offset|)` |

| 约束 | 要求 |
|------|------|
| Numeric 约束 | 调用方须满足 `A: Numeric`（需要 `zero()`/`one()`，定义在 Numeric trait 上而非 Element） |
| 维度固定 | 返回类型固定为 `Tensor<A, Ix2>`（二维），不支持高维 |
| 内存布局 | 默认 C-contiguous（行优先，与 §6.1 项目默认布局一致） |

### 19.3 empty_uninit 语义

**设计背景**：`Tensor<MaybeUninit<A>, D>` 中的 `MaybeUninit<A>` 不满足 `Element` trait（缺少 `PartialEq`、`Debug` 等约束），因此无法使用标准 `TensorBase<Owned<A>, D>` 类型。采用独立的 `UninitTensor` 类型封装未初始化内存。

| 属性 | 行为 |
|------|------|
| 签名 | `unsafe fn empty_uninit<A, D: Dimension>(shape: D) -> UninitTensor<A, D>` |
| UninitTensor 定义 | 独立结构体，内部持有 `shape: D`、`strides: D`、`offset: usize`、`flags: u16` 和对齐堆分配的 `MaybeUninit<A>` 缓冲区。不实现 `Storage` trait（因 `MaybeUninit<A>` 不满足 `Element`），不暴露任何读取方法 |
| 安全性 | `unsafe`：调用方须在读取前初始化所有元素，否则为未定义行为 |
| 初始化完成 | 调用方通过 `assume_init()` 将 `UninitTensor<A, D>` 转换为 `Tensor<A, D>`，该方法同样为 `unsafe`。签名见下方 |
| 用途 | 性能关键路径中避免零初始化开销（如上游库立即填充全部元素） |
| 对齐 | 使用默认 64 字节对齐分配 |
| 布局 | 默认 C-contiguous（与项目全局布局一致，见 §7.1） |
| 不提供 safe `empty` | 避免与 `zeros` 语义重复或引入隐式零初始化歧义 |
| Drop 行为 | `UninitTensor<A, D>` 的 drop 仅释放缓冲区内存，不调用元素的 drop glue（`MaybeUninit<T>` 的 drop 是 no-op） |

**assume_init 签名**：

```rust
impl<A: Element, D: Dimension> UninitTensor<A, D> {
    /// # Safety
    /// All elements must be initialized before calling this method.
    /// Calling this on a partially initialized tensor is undefined behavior.
    pub unsafe fn assume_init(self) -> Tensor<A, D>;
}
```

| 属性 | 行为 |
|------|------|
| 签名 | `unsafe fn assume_init(self) -> Tensor<A, D> where A: Element` |
| 消耗 self | 消耗 `UninitTensor`，避免未初始化数据被二次访问 |
| 前置条件 | 所有元素须已初始化 |
| 返回值 | `Tensor<A, D>`，复用同一块内存缓冲区（零拷贝），形状和步长不变 |

### 19.4 arange 语义

| 属性 | 行为 |
|------|------|
| 签名 | `fn arange(start: A, end: A, step: A) -> Tensor1<A> where A: Numeric + PartialOrd` |
| 范围 | 半开区间 `[start, end)`，不含 end |
| 空范围 | `start >= end`（step > 0）或 `start <= end`（step < 0）时返回空 Tensor1 |
| step = 0 | panic（运行时） |
| step 精度 | step 类型与元素类型相同（泛型参数 A） |
| 元素数 | 浮点类型：`max(0, ceil((end - start) / step))`。整数类型：使用专门的 ceiling 除法 `let d = end - start; let s = step; if d > 0 { (d + s - 1) / s } else { 0 }`，避免转浮点后精度丢失（大整数场景下 `ceil((end - start) as f64 / step as f64)` 可能因 f64 尾数不足而算错） |
| 整数溢出 | 整数 arange 的累加使用 `checked_add`，溢出时 panic（debug 和 release 模式均如此）。与 `sum/prod/cumsum` 的整数溢出策略一致 |
| 浮点精度 | 浮点 step 的累积误差可能导致最后一个元素的值略偏离预期 end；与 NumPy `np.arange` 行为一致 |
| 返回类型 | 始终 `Tensor1<A>`，布局为 F-contiguous（1D 即 C-contiguous）。设计理由：序列生成本质上是一维操作（单轴等差/等比序列），固定返回 Tensor1 语义最清晰。若需高维，可对结果调用 `reshape()`（如 `arange(0.0, 12.0, 1.0).reshape((3, 4))`） |

### 19.5 linspace 语义

| 属性 | 行为 |
|------|------|
| 签名 | `fn linspace(start: A, end: A, num: usize) -> Tensor1<A> where A: RealScalar` |
| 端点 | 默认包含 end 点（与 NumPy `np.linspace` 默认一致），共 `num` 个等间距点 |
| num = 0 | 返回空 Tensor1 |
| num = 1 | 返回仅含 `start` 的 Tensor1。此时步长公式 `(end - start) / (num - 1)` 的除数为零，实现须跳过步长计算，直接构造单元素数组 |
| num ≥ 2 | 步长为 `(end - start) / (num - 1)`，浮点累积误差不可避免 |
| 仅限 RealScalar | linspace 仅对浮点类型实现（整数等间距无意义） |

### 19.6 logspace 语义

| 属性 | 行为 |
|------|------|
| 签名 | `fn logspace(base: A, start: A, end: A, num: usize) -> Tensor1<A> where A: RealScalar` |
| 语义 | 等价于 `linspace(start, end, num).mapv(|x| base.powf(x))` |
| base 约束 | base 须 > 0 且 ≠ 1；若 base ≤ 0 或 base == 1，panic（违背数学意义，与 "无隐式行为" 原则一致） |
| base = 10 | 常见用法，`logspace(10.0, 0.0, 4.0, 5)` → `[1.0, 10.0, 100.0, 1000.0, 10000.0]`（内部 linspace(0.0, 4.0, 5) 产生 [0, 1, 2, 3, 4]，10^x 即得结果） |
| 仅限 RealScalar | 与 linspace 一致，仅对浮点类型实现 |

### 19.7 from_vec / from_slice / from_shape_vec / from_shape_slice 语义

#### 一维构造（from_vec / from_slice）

| 属性 | 行为 |
|------|------|
| from_vec 签名 | `fn from_vec(v: Vec<A>) -> Tensor1<A> where A: Element` |
| from_slice 签名 | `fn from_slice(s: &[A]) -> Tensor1<A> where A: Element` |
| 形状 | 结果形状为 `[v.len()]` / `[s.len()]`，即一维 |
| 所有权 | `from_vec` 接收 `Vec<A>`，零拷贝复用堆分配；`from_slice` 从切片拷贝数据到新分配缓冲区 |
| 空输入 | `Vec::new()` / `&[]` 返回长度为 0 的空 Tensor1 |
| 布局 | F-contiguous（1D 即 C-contiguous） |

#### 多维构造（from_shape_vec / from_shape_slice）

| 属性 | 行为 |
|------|------|
| from_shape_vec 签名 | `fn from_shape_vec<D: Dimension>(shape: D, v: Vec<A>) -> Result<Tensor<A, D>, ShapeError> where A: Element` |
| from_shape_slice 签名 | `fn from_shape_slice<D: Dimension>(shape: D, s: &[A]) -> Result<Tensor<A, D>, ShapeError> where A: Element` |
| 形状验证 | `shape.size()` 须等于 `v.len()` / `s.len()`，否则返回 `Err(ShapeError)` |
| ShapeError | 枚举类型，至少包含 `IncompatibleLayout { expected: usize, got: usize }` 变体，描述元素数不匹配 |
| 所有权 | `from_shape_vec` 零拷贝复用 Vec 缓冲区；`from_shape_slice` 拷贝到新分配 |
| 对齐与 ALIGNED 标志 | `from_shape_vec` 零拷贝复用 Vec 缓冲区时，`ALIGNED` 标志取决于传入 Vec 的实际对齐。标准 `Vec<A>` 的对齐由 Rust 全局分配器决定（通常 8 或 16 字节），**不保证 64 字节对齐**。因此 `ALIGNED` 标志通常为 `false`，SIMD 路径不会对其优化（见 §9.1.3 对齐检查）。`from_shape_slice` 拷贝到新分配时使用 Xenon 的 64 字节对齐分配器，`ALIGNED` 标志为 `true`。若从 `from_shape_vec` 构建的 Tensor 需 SIMD 优化，应先调用 `to_f_contiguous()`（见 §22.1）重新分配到对齐缓冲区 |
| 空输入 | 形状为全零维度时接受空 Vec/切片，返回零元素 Tensor |
| 布局 | 默认 C-contiguous（与 §7.1 全局布局一致），步长由形状按 C-order 推导 |
| Order 参数变体 | 提供 `from_shape_vec_order` / `from_shape_slice_order` 额外接受 `order: Option<Order>` 参数（见 §7.2），允许调用方指定行优先或列优先布局 |

---

## 20. 运算符重载

### 20.1 重载类型

| 类别 | 运算符/方法 | 说明 |
|------|-------------|------|
| 四则运算 | Add, Sub, Mul, Div | 标准算术运算符重载 |
| 比较运算 | `.lt()`/`.le()`/`.gt()`/`.ge()` | 仅方法调用，不提供运算符语法（见 §20.6 比较运算符设计） |
| 标量相等 | PartialEq `==`/`!=` | 返回 `bool` |
| 复合赋值 | AddAssign, SubAssign, MulAssign, DivAssign | 原地运算 |
| 一元运算 | Neg, Not | 取反、逻辑非 |
| 广播 | 所有二元运算符 | 隐式支持广播 |

### 20.2 二元运算符所有权矩阵

**设计决策**：存储复用由操作数的**所有权语义**（值 vs 引用）显式决定——传入 `Tensor`（值，消耗所有权）表示允许复用存储，传入 `&Tensor`（引用，不消耗所有权）表示不复用。用户通过选择值或引用调用，可完全控制是否复用存储，不存在运行时隐式判断。具体规则见下表(以 Add 为例，Sub/Mul/Div 同理)。

| 左操作数 | 右操作数 | 存储复用 | 返回类型 |
|----------|----------|----------|----------|
| `&Tensor<A, D>` | `&Tensor<A, D>` | 无，分配新数组 | `Tensor<A, D>` |
| `Tensor<A, D>` | `&Tensor<A, D>` | 复用左操作数存储（原地写入） | `Tensor<A, D>` |
| `&Tensor<A, D>` | `Tensor<A, D>` | 复用右操作数存储（原地写入） | `Tensor<A, D>` |
| `Tensor<A, D>` | `Tensor<A, D>` | 复用左操作数存储（原地写入） | `Tensor<A, D>` |
| `&Tensor<A, D>` | `A`（标量） | 无，分配新数组 | `Tensor<A, D>` |
| `Tensor<A, D>` | `A`（标量） | 复用左操作数存储（原地写入） | `Tensor<A, D>` |
| `A`（标量） | `&Tensor<A, D>` | 无，分配新数组 | `Tensor<A, D>` |
| `A`（标量） | `Tensor<A, D>` | 复用右操作数存储（原地写入） | `Tensor<A, D>` |

**规则总结**：值语义的操作数（`Tensor<A, D>`，非引用）允许复用存储；引用语义的操作数（`&Tensor<A, D>`）不参与复用，始终分配新数组。当两个操作数均为值时，优先复用左操作数。

**存储复用前提**：被复用的操作数须为连续内存且形状与输出一致（含广播展开后）。若不满足，回退到分配新数组。此回退为静默行为——用户无法从返回类型区分是否发生了存储复用。设计理由：广播场景下操作数形状可能与输出不一致（如 `(3,4) + (4,)` 中右操作数形状不等于输出形状 `(3,4)`），强制运行时检查是不可避免的。

**存储复用与填充**：当被复用的操作数为填充数组（`PADDED = true`，见 §7.6）时，存储复用仍可发生——填充区域不参与逻辑运算（见 §9.1.4）。但复用后的输出 Tensor 继承源数组的填充布局（`PADDED = true`），逻辑元素正确，填充字节值未定义。若需无填充的输出，调用方应在运算后调用 `to_f_contiguous()` / `to_c_contiguous()`（见 §22）移除填充。

**跨维度运算**：二元运算符要求两操作数的维度类型 `D` 相同（见 §16.6）。不同维度类型（如 `Tensor<A, Ix2>` 与 `Tensor<A, Ix3>`）的运算会产生编译错误——调用方须先通过 `into_dyn()` 统一为 `IxDyn`。广播由内部迭代器（Zip，见 §11.2）在运行时按 §16.1 规则处理，不改变维度类型。所有权矩阵中所有规则均基于相同维度类型 `D` 的前提。

**标量运算符实现约束（孤儿规则）**：

Rust 的孤儿规则（Orphan Rule）禁止为外部类型实现外部 trait。具体影响：

| 方向 | impl 形式 | 合法性 | 原因 |
|------|-----------|--------|------|
| RHS 标量 | `impl<S, A, D> Add<A> for TensorBase<S, D> where A: Numeric` | ✅ 合法 | `TensorBase` 是本地类型，Self 位置被覆盖 |
| LHS 标量（泛型） | `impl<S, A, D> Add<TensorBase<S, D>> for A where A: Numeric` | ❌ E0210 | `A` 是未覆盖的类型参数，不满足"至少一个覆盖类型参数"要求 |
| LHS 标量（具体类型） | `impl<S, A, D> Add<TensorBase<S, D>> for f64` | ✅ 合法 | `f64` 虽为外部类型，但 `TensorBase<S, D>` 作为具体类型出现在 trait 参数中，满足覆盖要求 |

因此 LHS 标量运算须通过宏为每种具体标量类型生成独立的 `impl` 块：

```
// 宏模式（以 Add 为例，Sub/Mul/Div 同理）
// 关键：将 $scalar 直接绑定到 Storage<Elem = $scalar>，确保标量类型与张量元素类型一致
macro_rules! impl_scalar_add_lhs {
    ($scalar:ty) => {
        impl<S, D> Add<TensorBase<S, D>> for $scalar
        where
            S: Storage<Elem = $scalar>,
            D: Dimension,
            $scalar: Numeric,
        {
            type Output = Tensor<$scalar, D>;
            fn add(self, rhs: TensorBase<S, D>) -> Self::Output { ... }
        }

        impl<S, D> Add<&TensorBase<S, D>> for $scalar
        where
            S: Storage<Elem = $scalar>,
            D: Dimension,
            $scalar: Numeric,
        {
            type Output = Tensor<$scalar, D>;
            fn add(self, rhs: &TensorBase<S, D>) -> Self::Output { ... }
        }
    };
}

// 为每种支持运算的标量类型展开（须实现 Numeric，见 §4.4）
impl_scalar_add_lhs!(f32);
impl_scalar_add_lhs!(f64);
impl_scalar_add_lhs!(i32);
impl_scalar_add_lhs!(i64);
impl_scalar_add_lhs!(Complex<f32>);
impl_scalar_add_lhs!(Complex<f64>);
```

**LHS 标量支持的类型清单**（须实现 `Numeric`，见 §4.4；`bool` 和 `usize` 不实现 `Numeric`，不支持运算符重载）：`f32, f64, i32, i64, Complex<f32>, Complex<f64>`。

**不支持泛型 LHS 标量**：`impl<T: Numeric> Add<Tensor<A,D>> for T` 在 Rust 类型系统下不可能编译通过。用户无法为自定义类型添加 `Tensor + CustomType` 的运算符重载（与 §4.2 Sealed Trait 策略一致，类型体系封闭）。

### 20.3 单态化控制

运算符重载的实现须采用以下策略控制泛型实例化爆炸：

1. **统一 Storage 泛型**：每个运算符的 `impl` 块基于 `TensorBase<S, D>` 统一泛型编写（S: Storage），覆盖 Owned/View/ViewMut/Arc 四种存储模式，而非为每种存储模式编写独立 impl。编译器仅对用户实际调用的 `(S, D, A)` 组合生成代码，未使用的组合不产生编译开销。

2. **内部 monomorphic kernel**：运算符 impl 体仅做类型转换和参数准备，核心计算逻辑委托给按 `(运算符, 元素类型)` 分派的 monomorphic 函数（如 `add_f64_raw(ptr, ptr, ptr, len)`）。泛型 impl 被内联后，编译器可识别不同实例共享同一 kernel，消除重复代码生成。

**二元运算符 impl 展开模式**（以 Add 为例）：

所有权矩阵中的 4 种 Tensor 组合通过宏 `impl_binary_op!` 生成。宏为 `(owned, owned)`、`(owned, &ref)`、`(&ref, owned)`、`(&ref, &ref)` 各生成一个 `impl` 块，RHS 标量同理。核心计算统一调用同一个 monomorphic kernel：

```
// 4 种所有权组合的 impl 展开（简化示意）
macro_rules! impl_binary_op {
    ($op:ident, $method:ident) => {
        // (&Tensor, &Tensor) — 无复用，分配新数组
        impl<'a, 'b, S1, S2, A, D> $op<&'b TensorBase<S2, D>> for &'a TensorBase<S1, D>
        where
            S1: Storage<Elem = A>,
            S2: Storage<Elem = A>,
            D: Dimension,
            A: Numeric,
        {
            type Output = Tensor<A, D>;
            fn $method(self, rhs: &'b TensorBase<S2, D>) -> Self::Output {
                // 分配新数组，调用 monomorphic kernel
            }
        }

        // (Tensor, &Tensor) — 复用左操作数存储
        impl<'b, S1, S2, A, D> $op<&'b TensorBase<S2, D>> for TensorBase<S1, D>
        where
            S1: StorageMut<Elem = A>,  // StorageMut: 可写入（Owned/ViewMut）
            S2: Storage<Elem = A>,
            D: Dimension,
            A: Numeric,
        {
            type Output = Tensor<A, D>;
            fn $method(self, rhs: &TensorBase<S2, D>) -> Self::Output {
                // 若 self 连续且形状匹配 → 原地写入
                // 否则 → 分配新数组（静默回退）
            }
        }

        // (Tensor, Tensor) — 复用左操作数存储
        // (&Tensor, Tensor) — 复用右操作数存储
        // ... 同理展开
    };
}

impl_binary_op!(Add, add);
impl_binary_op!(Sub, sub);
impl_binary_op!(Mul, mul);
impl_binary_op!(Div, div);
```

### 20.4 视图参与运算

| 操作数类型 | 行为 |
|------------|------|
| `&TensorView` / `&TensorViewMut` | 等同于 `&Tensor`，始终分配新数组 |
| `TensorViewMut`（消耗） | 始终分配新数组。`TensorViewMut` 是对其他 Tensor 存储的 `&mut` 借用，不拥有数据所有权，无法将底层 buffer 转移为 Owned tensor。消耗 ViewMut 仅表示运算完成后该可变借用失效（调用方不可再通过原 ViewMut 写入），不代表存储复用。若需存储复用，应对 Owned `Tensor` 调用 `.view_mut()` 获取 ViewMut 后原地写入（如 `zip_mut_with`），或直接消耗 Owned `Tensor` 参与运算 |
| `TensorView`（消耗） | 视图无自有存储，消耗后仍需分配新数组。**API 设计建议**：考虑不为 `TensorView`（值）实现二元运算符 trait（仅支持 `&TensorView`），避免用户误以为消耗视图能获得存储复用。若提供值传递 impl，文档须明确标注"消耗视图不获得存储复用" |
| `&ArcTensor` | 等同于 `&Tensor`，始终分配新数组（不触发 make_mut） |
| `ArcTensor`（消耗） | 通过 `Arc::try_unwrap()` 尝试获取独占所有权：成功则复用存储（等效 Owned）；失败（引用计数 > 1）则分配新数组 |

### 20.5 复合赋值约束

| 规则 | 说明 |
|------|------|
| 左操作数 | 须为可写存储（Owned 或 ViewMut），View 和 ArcRepr 不支持复合赋值 |
| 右操作数广播 | 右操作数可被广播到左操作数形状 |
| 左操作数广播 | 禁止——左操作数须拥有完整存储，不可为广播视图 |

**ArcRepr 复合赋值说明**：`ArcTensor` 不实现 `AddAssign` 等复合赋值 trait（编译期拒绝）。虽然 §20.4 中 `ArcTensor`（消耗）可通过 `Arc::try_unwrap()` 获取独占所有权参与二元运算，但复合赋值（如 `a += b`）要求左操作数为 `&mut Self`，而 `ArcTensor` 无法提供独占 `&mut` 保证（引用计数可能 > 1）。因此复合赋值的限制比二元运算更严格——ArcRepr 在编译期即被排除，不存在运行时回退。

**View 复合赋值说明**：`TensorView`（不可变视图）不实现复合赋值 trait（编译期拒绝）。这与 §20.4 中 `TensorView`（消耗）无法复用存储的规则一致——不可变视图不提供写权限。

### 20.6 比较运算符

比较运算符不参与上述所有权/存储复用矩阵，因为输出为 `Tensor<bool, D>`（类型不同于输入），始终分配新数组。

**设计决策**：逐元素比较返回 `Tensor<bool, D>`，但 `std::cmp::PartialOrd` trait 的方法签名返回 `bool`，无法覆盖为返回 Tensor。因此**不通过 `PartialOrd` 实现运算符重载**，而是采用以下方案：

- `< <= > >=` 运算符**不提供**（`PartialOrd` 返回 `bool`，与逐元素 Tensor 语义冲突）
- 逐元素比较**仅通过方法调用**：`.lt()`、`.le()`、`.gt()`、`.ge()`（见 §12.2 比较操作语义），返回 `Tensor<bool, D>`
- **命名说明**：`.lt()` 等方法名与 `PartialOrd` trait 的固有方法同名，但 `TensorBase` **不实现 `PartialOrd`**，因此不存在 trait 方法冲突。此处选择 `.lt()` 而非 `.elem_lt()` 等替代命名，是因为：(1) 与 NumPy 的 `np.less()` / ndarray 的 `.lt()` 命名惯例一致；(2) 返回类型 `Tensor<bool, D>` 已明确区分于 `PartialOrd::lt() -> bool`；(3) IDE 自动补全不会提示 `PartialOrd::lt()`（因未实现该 trait）
- `==` / `!=`：`PartialEq` trait 返回 `bool`（标量语义，判断两个 Tensor 是否完全相等）；逐元素布尔掩码使用 `.eq()` / `.ne()` 方法

| 方法 | 签名 | 返回类型 | 元素约束 |
|------|------|----------|----------|
| `.lt(&self, other) -> Tensor<bool, D>` | 逐元素 `<` | `Tensor<bool, D>` | `A: PartialOrd` |
| `.le(&self, other) -> Tensor<bool, D>` | 逐元素 `<=` | `Tensor<bool, D>` | `A: PartialOrd` |
| `.gt(&self, other) -> Tensor<bool, D>` | 逐元素 `>` | `Tensor<bool, D>` | `A: PartialOrd` |
| `.ge(&self, other) -> Tensor<bool, D>` | 逐元素 `>=` | `Tensor<bool, D>` | `A: PartialOrd` |

以上方法支持广播（与 §16.1 广播规则一致），始终分配新数组。`PartialEq` 的 `==`/`!=` 运算符保持标量语义（`Tensor<A, D> == Tensor<A, D> -> bool`，判断两个数组形状和所有元素是否完全相等）。**NaN 行为**：由于 `PartialEq` 遵循 IEEE 754（`NaN != NaN`），两个形状相同且在相同位置均为 NaN 的 Tensor，`==` 比较结果为 `false`（因为逐元素比较时 NaN != NaN）。此 NaN 语义与 §18.8 布尔归约中"NaN 视为非零"的处理一致——NaN 不等于任何值（包括自身）。若需 NaN 敏感的相等判断，使用 `allclose(rtol=0.0, atol=0.0)`（但注意 `allclose` 同样将 NaN 比较为 false，见 §21.5）。

**`PartialEq` 的局限性与 `exact_eq` 方法**：`PartialEq` 返回 `bool`，无法区分"形状不同"和"形状相同但元素不等"两种失败情况——两者均返回 `false`。为支持错误诊断，提供以下方法：

| 方法 | 签名 | 返回类型 | 说明 |
|------|------|----------|------|
| `exact_eq` | `fn exact_eq(&self, other: &Tensor<A, D>) -> Result<bool, ShapeMismatch> where A: PartialEq` | `Result<bool, ShapeMismatch>` | 形状不同时返回 `Err(ShapeMismatch)`；形状相同时逐元素比较，所有元素相等返回 `Ok(true)`，否则返回 `Ok(false)`。NaN 语义与 `PartialEq` 一致（`NaN != NaN`） |

`exact_eq` 适用于需要区分"形状不匹配"与"元素不等"的场景（如测试断言、数据校验）。若仅需要布尔结果，`PartialEq` 的 `==` 更简洁。

---

## 21. 实用操作

### 21.1 基本实用操作

| 操作 | 说明 |
|------|------|
| copy_to, fill | 复制和填充 |
| is_close/allclose | 近似比较（语义见 §21.5 is_close/allclose 语义） |
| clip | 裁剪 |
| flip/flipud/fliplr | 翻转 |
| to_owned/into_owned | 转换为拥有 |

### 21.2 flip / flipud / fliplr 语义

**方法签名**：

| 方法 | 签名 | 说明 |
|------|------|------|
| `flip`（Owned/ArcRepr） | `fn flip(&self, axes: &[usize]) -> TensorView<'_, A, D>` | 沿指定轴翻转，零拷贝（步长取反）。返回视图的生命周期绑定到 `&self` 借用 |
| `flip`（View 消耗） | `fn flip(self, axes: &[usize]) -> TensorView<'a, A, D>` | 消耗 `TensorView<'a, A, D>`，返回保留原始生命周期 `'a` 的新视图。避免视图链式操作时生命周期逐步缩短 |
| `flip`（ViewMut 消耗） | `fn flip(self, axes: &[usize]) -> TensorViewMut<'a, A, D>` | 消耗 `TensorViewMut<'a, A, D>`，返回保留原始生命周期 `'a` 的新可变视图 |
| `flipud` | 同 `flip`，等价于 `flip(&[0])` | 沿轴 0 翻转。ndim ≥ 1 时反转第 0 轴元素顺序；0D 数组为 no-op |
| `fliplr` | 同 `flip`，等价于 `flip(&[1])` | 沿轴 1 翻转。1D 数组返回 `InvalidAxis`（无轴 1） |

**边界行为**：

| 场景 | 行为 | 说明 |
|------|------|------|
| 轴索引越界 | 返回 `InvalidAxis` | axes 中的轴索引须在 `[0, ndim)` 范围内 |
| 重复轴 | 等价 no-op（步长取反两次恢复原值） | 不报错，与 NumPy `np.flip` 行为一致 |
| 空 axes | 返回原视图的拷贝（无修改） | 不翻转任何轴 |
| 空数组（shape 含 0 维度） | 正常返回视图，步长取反但无实际元素访问 | `HAS_NEG_STRIDE` 标志被设置；当 shape 含零维度时布局标志遵循 §7.2 约定（值未定义） |
| 0D 数组 | `flipud` / `fliplr` 为 no-op（无轴可翻转） | `flip(&[])` 返回原视图拷贝 |

### 21.3 clip 语义

| 属性 | 行为 |
|------|------|
| 签名 | `fn clip(&self, min: A, max: A) -> Tensor<A, D> where A: PartialOrd + Clone` |
| 操作 | 逐元素裁剪到 `[min, max]` 范围内，小于 min 的值设为 min，大于 max 的值设为 max |
| min > max | panic（调用方违反前置条件） |
| 返回类型 | 新分配的 Owned 数组 |
| 元素为 NaN（浮点类型） | NaN 比较为 false，clip 不修改 NaN（NaN 不满足 min/max 约束，保持不变） |
| min 为 NaN（浮点类型） | panic（调用方违反前置条件）。理由：`x < NaN` 恒为 false，导致下限裁剪失效且行为依赖实现方式（使用 `f64::min` 则 NaN 传播使结果全为 NaN，使用 if-else 则下限失效）。显式 panic 比静默错误或平台相关行为更安全。与 "min > max → panic" 的前置条件检查一致 |
| max 为 NaN（浮点类型） | panic（调用方违反前置条件）。理由同上：`x > NaN` 恒为 false，行为依赖实现且不可预测 |
| min 和 max 均为 NaN（浮点类型） | panic（同上） |
| 整数类型 | NaN 相关规则不适用（整数类型无 NaN）。clip 对整数类型仅执行 `min > max` 前置条件检查，无 NaN panic 风险 |

### 21.4 copy_to / fill 语义

| 方法 | 签名 | 说明 |
|------|------|------|
| `copy_to` | `fn copy_to<S2>(&self, dst: &mut TensorBase<S2, D>) where S2: StorageMut<Elem = A>` | 将 self 的数据拷贝到 dst。两者形状须完全一致，否则返回 `ShapeMismatch`。按物理内存顺序拷贝。目标可为任意可写存储（Owned、ViewMut），使用泛型约束 `S2: StorageMut` 而非硬编码 `TensorViewMut`，避免用户在拷贝到 Owned Tensor 时需要额外的 `.view_mut()` 调用 |
| `fill` | `fn fill(&mut self, value: A) where A: Clone, S: StorageMut` | 用指定值填充所有逻辑元素。要求可变存储（Owned 或 ViewMut，由 `S: StorageMut` 约束在编译时排除只读存储）。若 `self` 为广播视图（存在零步长，即 `has_zero_stride()` 为 true），则 panic，与 §11.3 `iter_mut()` 行为保持一致。理由：广播视图的同一物理位置映射到多个逻辑位置，`fill` 写入会通过零步长反复覆写同一内存，语义上等价于对共享引用的 `&mut`，因此必须在入口处拒绝 |

### 21.5 is_close / allclose 语义

#### 21.5.1 RealScalar 变体

| 方法 | 签名 | 返回类型 |
|------|------|----------|
| `is_close` | `fn is_close<S2, D2>(&self, other: &TensorBase<S2, D2>, rtol: A, atol: A) -> Tensor<bool, D> where S2: Storage<Elem = A>, D2: Dimension, A: RealScalar` | `Tensor<bool, D>` |
| `allclose` | `fn allclose<S2, D2>(&self, other: &TensorBase<S2, D2>, rtol: A, atol: A) -> bool where S2: Storage<Elem = A>, D2: Dimension, A: RealScalar` | `bool` |

**返回维度说明**：`is_close` 返回 `Tensor<bool, D>`（与 self 同维度类型），输出形状为 self 与 other 广播后的形状（见下方广播规则）。当广播后形状与 `self.shape()` 不同时，返回的 `D` 实例的内部值会被更新为广播后形状。这要求 `D: Dimension` 的 `slice_mut()` 方法可用于原地更新 shape（见 §3.2）。**Strides 重计算**：广播后 strides 按 §16.2 零步长规则重新计算——广播新增的 size-1 维度使用零步长，已有维度的步长保持不变。输出 Tensor 为新分配的 Owned 数组（非视图），其 strides 按 F-contiguous 顺序从广播后形状重新推导，不继承输入的零步长 |

#### 21.5.2 ComplexScalar 变体

| 方法 | 签名 | 返回类型 |
|------|------|----------|
| `is_close` | `fn is_close<S2, D2>(&self, other: &TensorBase<S2, D2>, rtol: A::Real, atol: A::Real) -> Tensor<bool, D> where S2: Storage<Elem = A>, D2: Dimension, A: ComplexScalar` | `Tensor<bool, D>` |
| `allclose` | `fn allclose<S2, D2>(&self, other: &TensorBase<S2, D2>, rtol: A::Real, atol: A::Real) -> bool where S2: Storage<Elem = A>, D2: Dimension, A: ComplexScalar` | `bool` |

**复数比较公式**：`|a - b| ≤ atol + rtol * |b|`，其中 `|·|` 为复数模（使用 `ComplexScalar::norm()`，底层为 hypot 避免溢出）。`rtol` / `atol` 类型为 `A::Real`（实数类型），与实数变体保持一致的接口设计。

#### 21.5.3 通用语义（实数与复数共享）

| 属性 | 行为 |
|------|------|
| 比较公式 | `|a - b| ≤ atol + rtol * |b|`（与 NumPy `np.isclose` / `np.allclose` 一致；注意此公式非对称——交换 a/b 可能改变结果。这是 NumPy 的设计选择，Xenon 予以保留以保持兼容性） |
| rtol / atol | 相对容差和绝对容差，均须 ≥ 0，否则 panic |
| 广播 | `is_close` 支持广播（self 与 other 形状按 §16.1 规则广播），返回广播后形状的 bool Tensor |
| NaN 处理 | 两个 NaN 比较为 `false`（与 NumPy `np.isclose` 一致）；若需 NaN 相等判定，须用户自行处理 |
| Inf 处理 | `+Inf == +Inf` 为 `true`，`+Inf != -Inf`，遵循 IEEE 754 |
| allclose 语义 | 等价于 `is_close().all()`，所有元素近似相等时返回 `true` |
| 空数组 | `allclose` 对空数组返回 `true`（vacuous truth） |

---

## 22. 连续性保证

连续性保证方法提供三组操作：
- `to_*` 系列：接受 `&self`，始终返回新分配的 Owned Tensor（`Tensor<A, D>`），保证严格连续布局
- `as_*` 系列：接受 `&self`，零拷贝视图路径，仅当已满足连续性时返回 `Some(TensorView)`，否则返回 `None`
- `into_*` 系列：接受 `self`（仅 Owned），已连续时零拷贝返回 `self`，否则拷贝后返回新 Tensor

### 22.1 判定标准

连续性判定使用**严格连续性**（`is_f_contiguous()` / `is_c_contiguous()`，见 §7.4），不使用宽松连续性（`is_f_padded_contiguous()` 等）。理由：输出保证严格连续步长（无填充），严格连续是零拷贝 reshape 判定的必要条件（见 §7.6.6），宽松连续不满足此条件。

**与 §9 SIMD 路径的宽松连续的区别**：§9.1.3 的 SIMD 路径使用宽松连续性（`is_padded_contiguous()`）作为触发条件，允许 SIMD 操作利用填充字节。连续性保证方法（本章）使用严格连续性，因为其目的是保证输出可安全用于零拷贝 reshape——填充数组在 reshape 后填充字节可能被误读为逻辑元素。两种连续性标准服务于不同目的，不冲突。

**填充数组行为**：填充数组（`PADDED = true`，`is_f_padded_contiguous() = true` 但 `is_f_contiguous() = false`）被视为非严格连续。`to_f_contiguous()` 会将其重排为严格 F-contiguous（移除填充），`as_f_contiguous()` 返回 `None`。需要保留填充优化的场景，调用方应直接检查 `is_f_padded_contiguous()` 并使用 `to_owned()` 而非连续性保证方法。

**广播视图行为**：广播视图（`has_zero_stride() = true`）的零步长轴不满足严格连续性判定（`is_f_contiguous()` 要求步长严格等于 shape 乘积，stride=0 不满足），因此 `as_*_contiguous()` 返回 `None`，`to_*_contiguous()` 总是执行拷贝。这与 numpy 的 `ascontiguousarray` 行为一致。

### 22.2 方法签名与行为

**`to_*` 系列（引用输入，始终返回新 Owned）**

| 方法 | 签名 | 行为 |
|------|------|------|
| `to_f_contiguous()` | `fn to_f_contiguous(&self) -> Tensor<A, D> where A: Clone, D: Clone` | 始终返回新的 Owned Tensor。若输入已严格 F-contiguous，等价于 `to_owned()`（拷贝数据到新分配）；否则重新排列为严格 F-contiguous 布局并返回新 Tensor |
| `to_c_contiguous()` | `fn to_c_contiguous(&self) -> Tensor<A, D> where A: Clone, D: Clone` | 始终返回新的 Owned Tensor。若输入已严格 C-contiguous，等价于 `to_owned()`（拷贝数据到新分配）；否则重新排列为严格 C-contiguous 布局并返回新 Tensor |
| `to_contiguous()` | `fn to_contiguous(&self) -> Tensor<A, D> where A: Clone, D: Clone` | 始终返回新的 Owned Tensor。若输入已严格 F-contiguous，输出 F-contiguous；若输入已严格 C-contiguous（但非 F-contiguous），输出 C-contiguous；否则输出 F-contiguous。**注意**：输出布局方向取决于输入的连续性方向——调用方若需确保固定布局方向，应使用 `to_f_contiguous()` 或 `to_c_contiguous()` 显式指定 |

**`as_*` 系列（零拷贝视图路径）**

| 方法 | 签名 | 行为 |
|------|------|------|
| `as_f_contiguous()` | `fn as_f_contiguous(&self) -> Option<TensorView<'_, A, D>>` | 若输入已严格 F-contiguous，返回 `Some(TensorView)`（零拷贝视图，生命周期绑定到 `&self`）；否则返回 `None` |
| `as_c_contiguous()` | `fn as_c_contiguous(&self) -> Option<TensorView<'_, A, D>>` | 若输入已严格 C-contiguous，返回 `Some(TensorView)`（零拷贝视图，生命周期绑定到 `&self`）；否则返回 `None` |
| `as_contiguous()` | `fn as_contiguous(&self) -> Option<TensorView<'_, A, D>>` | 若输入已严格连续（F 或 C），返回 `Some(TensorView)`（零拷贝视图，保持原布局方向）；否则返回 `None` |

**`into_*` 系列（消耗 self，按需拷贝）**

仅对 `Tensor<A, D>`（Owned 存储）实现。对 View / ViewMut / ArcRepr 等非 Owned 存储，须先转为 Owned（`to_owned()`）或使用 `to_*` 系列。

| 方法 | 签名 | 行为 |
|------|------|------|
| `into_f_contiguous()` | `fn into_f_contiguous(self) -> Tensor<A, D> where A: Clone, D: Clone` | 消耗 `self`。若已严格 F-contiguous，零拷贝返回 `self`（无分配）；否则拷贝为严格 F-contiguous 布局的新 Tensor |
| `into_c_contiguous()` | `fn into_c_contiguous(self) -> Tensor<A, D> where A: Clone, D: Clone` | 消耗 `self`。若已严格 C-contiguous，零拷贝返回 `self`（无分配）；否则拷贝为严格 C-contiguous 布局的新 Tensor |
| `into_contiguous()` | `fn into_contiguous(self) -> Tensor<A, D> where A: Clone, D: Clone` | 消耗 `self`。若已严格 F-contiguous，零拷贝返回 `self`；若已严格 C-contiguous（但非 F-contiguous），零拷贝返回 `self`；否则拷贝为 F-contiguous 布局的新 Tensor。**注意**：输出布局方向取决于输入的连续性方向——与 `to_contiguous()` 行为一致 |

### 22.3 输出布局属性

所有 `to_*` 和 `into_*` 方法输出的 Tensor 满足：

| 属性 | 值 | 说明 |
|------|-----|------|
| 步长 | 严格连续步长（无填充） | F-contiguous：`strides = [1, s₀, s₀×s₁, ...]`；C-contiguous：`strides = [..., s₂×s₃, s₃, 1]` |
| PADDED 标志 | `false` | 输出不含填充 |
| F/C_CONTIGUOUS 标志 | `F_CONTIGUOUS = true`（F 系列）或 `C_CONTIGUOUS = true`（C 系列） | 严格连续 |
| 对齐 | 新分配路径：64 字节默认对齐；零拷贝路径（`into_*` 已连续时）：继承原对齐 | 见 §7.5 |

### 22.4 适用存储模式

| 方法系列 | 适用存储 | 说明 |
|----------|---------|------|
| `to_*` | 所有 `S: Storage` | View / ViewMut / ArcRepr 均可调用，返回新 Owned Tensor |
| `as_*` | 所有 `S: Storage` | 返回的 `TensorView<'_, A, D>` 生命周期绑定到 `&self` 的借用。ArcTensor 的视图受限于 `&ArcTensor` 借用（见 §6.3），view 活跃期间 `make_mut()` 被编译器互斥（借用检查器保证） |
| `into_*` | 仅 `Tensor<A, D>`（Owned） | 消耗 self。其他存储模式须先 `to_owned()` 转为 Owned 再调用，或直接使用 `to_*` 系列 |

### 22.5 边界情况

| 输入 | `to_f_contiguous()` | `as_f_contiguous()` | `into_f_contiguous()` |
|------|---------------------|---------------------|----------------------|
| 0D 标量（Ix0） | `to_owned()`（1 元素，无步长） | `Some(view)`（0D 始终连续） | 零拷贝返回 `self` |
| 空张量（shape 含 0） | `to_owned()`（0 元素，F-contiguous 步长） | `Some(view)`（空数组同时 F/C-contiguous，见 §3.6） | 零拷贝返回 `self` |
| 1D 数组 | `to_owned()`（步长 `[1]`，F/C 等价） | `Some(view)`（1D 始终连续） | 零拷贝返回 `self` |
| 填充数组（PADDED=true） | 拷贝并重排为严格 F-contiguous（移除填充） | `None` | 拷贝并重排为严格 F-contiguous |
| 广播视图（has_zero_stride） | 拷贝为 F-contiguous（移除零步长） | `None` | N/A（非 Owned，须先 `to_owned()`） |
| 非连续视图（如转置） | 拷贝并重排为 F-contiguous | `None` | N/A（非 Owned，须先 `to_owned()`） |
| 非连续 Owned（如转置后 `to_owned()`） | 拷贝并重排为 F-contiguous | `None` | 拷贝并重排为严格 F-contiguous |

---

## 23. 转换操作

### 23.1 cast 语义

**方法签名**：

| 方法 | 签名 | 说明 |
|------|------|------|
| `cast` | `fn cast<B>(self) -> Tensor<B, D> where B: Element` | 消耗 Owned 数组，逐元素类型转换。**仅对 `Tensor<A, D>`（Owned）实现**。View 须先 `to_owned()` 再 cast |
| `cast_same` | `fn cast_same(&self) -> Tensor<A, D>` | 返回同类型拷贝（等价于 `to_owned()`）。设计意图：在泛型管线中提供与 `cast::<B>()` 对称的 API，避免调用方在 `B == A` 分支中使用不同的方法名。与 `.clone()` 的区别：`clone` 要求 `Clone`（`Tensor<A, D>` 始终满足），`cast_same` 强调"类型转换管线中的同类型路径"语义 |

**泛型约束**：`B: Element` 约束保证目标类型在本库的类型体系内（Sealed trait，见 §4.2）。转换规则为逐元素按值转换，始终成功（溢出使用饱和语义，见下方精度表）。不支持跨类别隐式转换（如 `Complex → 实数` 须显式取 `re()`）。当 `A == B` 时，`cast` 等价于深拷贝（消耗 self，分配新的 `Tensor<A, D>`），与 `to_owned()` 行为一致——语义上合法但无实际类型转换效果，调用方应优先使用 `to_owned()` 或 `clone()` 以避免歧义。

**存储模式支持**：

| 输入类型 | cast 行为 |
|----------|-----------|
| `Tensor<A, D>`（Owned） | 消耗 self，分配新的 `Tensor<B, D>` |
| `TensorView<A, D>` | 不支持，须先 `.to_owned().cast::<B>()` |
| `ArcTensor<A, D>` | 不支持，须先 `.to_owned().cast::<B>()` |

**精度行为**：

> **类别定义**：表中"整数"指 `i32`、`i64`、`usize`；"浮点"指 `f32`、`f64`；"数值"指整数与浮点的并集（即所有 `Numeric` 实现者）加 `usize`；"复数"指 `Complex<f32>`、`Complex<f64>`。所有精度行为与 Rust 1.45+ 的 `as` 转换语义一致（实现可直接使用 `as` 或等效内在函数）。

| 转换方向 | 行为 |
|----------|------|
| 浮点 → 浮点（高精度→低精度） | 按 IEEE 754 round-to-nearest-even 截断 |
| 浮点 → 浮点（低精度→高精度） | 精确转换，无精度损失 |
| 浮点 → 整数 | 向零截断（truncate），溢出为饱和（saturating cast） |
| 整数 → 浮点 | 最近偶数舍入（round-to-nearest-even） |
| 整数 → 整数（窄化） | 饱和截断（saturating cast） |
| 整数 → 整数（同宽/宽化） | 精确转换，无损失 |
| NaN → 整数 | 结果为 0 |
| Inf → 整数 | 饱和到目标类型的 MAX/MIN |
| bool → 数值 | true = 1, false = 0 |
| 数值 → bool | 非零 = true, 零 = false |
| 实数 → 复数 | 虚部为 0 |
| 复数 → 实数 | 不允许隐式转换，须显式取 re() |
| 复数 → 复数（高精度→低精度） | 实部虚部分别按 IEEE 754 round-to-nearest-even 截断 |
| 复数 → 复数（低精度→高精度） | 实部虚部分别精确转换，无精度损失 |
| usize ↔ i32/i64/f32/f64 | 按对应的"整数"规则处理（见上方）。`usize` 仅实现 `Element`（见 §4.2），但 `cast` 的约束为 `B: Element`，因此 `usize` 可作为源或目标类型 |

### 23.2 标准类型转换

**From/Into 转换**

| 源类型 | 目标类型 | 维度推导 | 说明 |
|--------|----------|----------|------|
| `Vec<A>` | `Tensor1<A>` | 1D，长度 = vec.len() | F-contiguous（1D 即 C-contiguous） |
| `&[A]` | `TensorView<A, Ix1>` | 1D，长度 = slice.len() | 借用，零拷贝 |
| `[A; N]` | `Tensor1<A>` | 1D，长度 = N | 从栈数组移动 |
| `Vec<Vec<A>>` | `Tensor2<A>` | 2D，shape = (outer.len(), inner.len()) | 所有内层 Vec 长度须一致，否则 panic。提供 `TryFrom` 变体返回 `Result<Tensor2<A>, InconsistentLengths>`，适用于不可信数据源（用户输入、文件解析等）。`From` 实现内部等价于 `TryFrom::try_from().expect(...)` |
| `A`（标量） | `Tensor0<A>` | 0D | 零维标量张量 |

**维度类型转换**

| 转换 | 行为 | 失败处理 |
|------|------|----------|
| 静态 → 动态（IxN → IxDyn） | 总是成功 | 无（见 §3.2 `into_dyn()`） |
| 动态 → 静态（IxDyn → IxN） | ndim 须匹配 | 返回 `DimensionMismatch` |

**存储模式转换**

| 转换 | 方法 | 开销 |
|------|------|------|
| 任意 → Owned | `to_owned()` / `into_owned()` | 数据拷贝（Owned 直接返回） |
| Owned → View | 自动借用 `&Tensor` → `TensorView` | 零开销（Rust 借用） |
| Owned → ViewMut | 自动借用 `&mut Tensor` → `TensorViewMut` | 零开销（Rust 借用） |
| Owned → ArcTensor | `into_arc()` | O(1)（包装为 Arc） |
| ArcTensor → Owned | `to_owned()` | 数据拷贝（或 `Arc::try_unwrap()` 零拷贝，若引用计数为 1） |

**标准库 Trait 实现**：

| Trait | 实现者 | 说明 |
|-------|--------|------|
| `AsRef<TensorView<A, D>>` | `Tensor<A, D>`, `ArcTensor<A, D>` | `&self` → `TensorView` 零开销借用 |
| `AsMut<TensorViewMut<A, D>>` | `Tensor<A, D>` | `&mut self` → `TensorViewMut` 零开销借用（ArcRepr 不实现，因写时复制语义与 `AsMut` 冲突） |
| `Borrow<TensorView<A, D>>` | `Tensor<A, D>` | 语义与 `AsRef` 一致，支持 `HashMap` 等集合场景。`ArcTensor` 不实现 `Borrow`（与 `AsRef` 不对称）——`ArcTensor` 通常不作为 `HashMap` key 使用，且 `Borrow` 要求借出引用与 `Hash`/`Eq` 行为一致，`ArcTensor` 的 `PartialEq` 比较 Arc 指针而非张量内容，会导致语义混淆 |
| `Default` | `Tensor<A, D>` | **仅为 `Tensor<A, Ix0>` 实现**（即 0D 标量张量）。实现条件形式化为 `A: Default + Element`，维度类型固定为 `Ix0`（而非泛化到 `D: Default`——虽然 `Ix1`~`Ix6` 和 `IxDyn` 在 §3 中均实现 `Default`，但对非零维度类型返回 `Default` 张量缺乏明确的 shape 语义，因此不予实现）。返回包含 `A::default()` 的 0D 标量张量。实际用途有限——创建数组推荐使用 `zeros(shape)` 或 `full(shape, value)` |
| `From<Vec<A>>` | `Tensor1<A>` | 见 §23.2 标准类型转换 |
| `From<[A; N]>` | `Tensor1<A>` | 见 §23.2 标准类型转换 |

---

## 24. 格式化输出

### 24.1 设计原则

| 原则 | 说明 |
|------|------|
| Display 面向用户 | 简洁、可读，只显示数据内容，不显示内部实现细节（存储模式、步长、偏移量） |
| Debug 面向诊断 | 完整、精确，显示所有内部元数据（存储模式、步长、偏移量），便于调试非连续视图和切片 |
| NumPy 风格 | 外层包裹 `array(...)`，数据部分使用方括号 `[]` 包裹，逗号+空格分隔元素 |

**总体格式模板**：

| trait | 格式模板 |
|-------|----------|
| Display | `array({数据内容})` |
| Debug | `array({数据内容}, shape={shape}, dtype={dtype}[, strides={strides}][, {存储模式}][, offset={offset}])` |

Debug 中方括号 `[]` 表示条件显示（仅当条件满足时追加对应字段），字段间以 `, ` 分隔，字段顺序固定。

**Debug 格式补充说明**：Debug 格式必须包含 strides 向量和 layout flags（如 F_CONTIGUOUS | ALIGNED），这对调试非连续数组至关重要。strides 字段须以 `isize` 形式输出（而非原始 `usize`），layout flags 须包含连续性、对齐、填充等布尔标记的组合。

### 24.2 dtype 显示名称

每种 `Element` 实现者（见 §4.3）对应固定的 dtype 字符串，用于 Debug 输出：

| Rust 类型 | dtype 字符串 |
|-----------|-------------|
| `i32` | `"i32"` |
| `i64` | `"i64"` |
| `f32` | `"f32"` |
| `f64` | `"f64"` |
| `bool` | `"bool"` |
| `usize` | `"usize"` |
| `Complex<f32>` | `"complex64"` |
| `Complex<f64>` | `"complex128"` |

dtype 字符串通过 `Element` trait 的关联常量或方法提供（实现细节由 §4.3 Element trait 定义约束）。

### 24.3 元素级格式化规则

各元素类型在数组格式化中的显示规则：

| 元素类型 | Display（数组内） | Debug（数组内） |
|----------|-------------------|----------------|
| 整数（i32/i64/usize） | 直接使用 `Display` trait，如 `42`、`-7`、`0` | 与 Display 相同 |
| 浮点（f32/f64） | 使用 Rust 标准库的 `Display` trait（最短表示，Ryu 算法），如 `1`、`0.5`、`3.14`。**不**固定 6 位有效数字 | 同 Display，完整精度输出 |
| bool | `"true"` / `"false"` | 与 Display 相同 |
| Complex | 使用 Complex 的 Display 格式（见 §5.2），如 `1+2i`、`3-4i`、`0+1i` | 使用 Complex 的 Debug 格式（见 §5.2），如 `Complex { re: 1.0, im: 2.0 }` |

**NaN/Inf 显示规则**（适用于浮点和复数元素）：

| 值 | 显示 | 示例 |
|----|------|------|
| NaN | `NaN` | `array([NaN, 1.0])` |
| +Inf | `inf` | `array([inf, -inf])` |
| -Inf | `-inf` | `array([inf, -inf])` |

### 24.4 Display 输出规范

Display 只显示数据内容，不显示 shape、dtype、存储模式、步长或偏移量。

**格式规则**：

| 属性 | 规则 |
|------|------|
| 外层包裹 | `array(...)` |
| 1D 数组 | `array([元素0, 元素1, ..., 元素N-1])`，逗号+空格分隔 |
| 2D 数组 | `array([[行0],\n [行1], ..., [行M-1]])`，每行缩进 1 空格，行内逗号+空格分隔 |
| 3D 及以上 | 递归嵌套，每增加一维增加 1 空格缩进，最内层为行向量 |
| 0D 标量 | `array(标量值)`，无中括号 |
| 空数组 | `array([])` |

**Display 示例**：

```
// 0D 标量 (f64)
array(42.0)

// 1D 整数 (i64), shape=[4]
array([1, 2, 3, 4])

// 1D 浮点 (f64), shape=[4]
array([1, 0.5, 3.14, -7.0])

// 1D bool, shape=[3]
array([true, false, true])

// 1D 复数 (Complex<f64>), shape=[3]
array([1+2i, 3-4i, 0+1i])

// 2D 浮点 (f64), shape=[2, 3]
array([[1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0]])

// 3D 浮点 (f64), shape=[2, 2, 3]
array([[[1.0, 2.0, 3.0],
       [4.0, 5.0, 6.0]],
      [[7.0, 8.0, 9.0],
       [10.0, 11.0, 12.0]]])

// 空数组 (f64), shape=[0]
array([])

// 空数组 (f64), shape=[2, 0, 3]
array([])

// NaN/Inf
array([NaN, inf, -inf, 1.0])
```

### 24.5 Debug 输出规范

Debug 在 Display 的数据内容基础上，追加完整的元数据字段。

**格式规则**：

| 属性 | 规则 |
|------|------|
| 基础格式 | `array({Display 的数据内容}, shape={shape}, dtype={dtype}[, strides={strides}][, {存储模式}][, offset={offset}])` |
| shape 字段 | 始终显示，格式 `shape=[d0, d1, ...]`，与 `shape()` 返回值一致 |
| dtype 字段 | 始终显示，使用 §24.2 定义的 dtype 字符串 |
| strides 字段 | 当 `!is_contiguous()` **且** `!is_padded_contiguous()` 时显示（即既非严格连续也非宽松连续），格式 `strides=[s0, s1, ...]`，值为 `isize`（与 `strides()` 方法返回值一致，见 §8.3）。填充数组（`is_padded_contiguous()` 为 true）不显示步长——虽然其严格连续性为 false，但 SIMD 意义上实质连续，显示步长会误导用户 |
| 存储模式字段 | 始终显示，取值：`owned`（Owned）、`view`（ViewRepr）、`view_mut`（ViewMutRepr）、`arc`（ArcRepr） |
| offset 字段 | 当 `offset() != 0` 时显示，格式 `offset=N`，N 为元素单位的偏移量 |
| 字段顺序 | 固定：shape → dtype → strides（条件） → 存储模式 → offset（条件） |

**Debug 示例**：

```
// 0D 标量 (f64), Owned
array(42.0, shape=[], dtype=f64, owned)

// 1D 整数 (i64), Owned, shape=[4]
array([1, 2, 3, 4], shape=[4], dtype=i64, owned)

// 2D 浮点 (f64), Owned, shape=[2, 3]
array([[1.0, 2.0, 3.0],
       [4.0, 5.0, 6.0]], shape=[2, 3], dtype=f64, owned)

// 2D 浮点 (f64), 只读视图 (ViewRepr), shape=[2, 3]
array([[1.0, 2.0, 3.0],
       [4.0, 5.0, 6.0]], shape=[2, 3], dtype=f64, view)

// 2D 浮点 (f64), 非连续视图, shape=[3, 2], strides=[1, 3]
array([[1.0, 4.0],
       [2.0, 5.0],
       [3.0, 6.0]], shape=[3, 2], dtype=f64, strides=[1, 3], view)

// 2D 浮点 (f64), 视图带偏移, shape=[2, 2], offset=3
array([[4.0, 5.0],
       [7.0, 8.0]], shape=[2, 2], dtype=f64, view, offset=3)

// 2D 浮点 (f64), ArcRepr, shape=[2, 3]
array([[1.0, 2.0, 3.0],
       [4.0, 5.0, 6.0]], shape=[2, 3], dtype=f64, arc)

// 空数组 (f64), shape=[0]
array([], shape=[0], dtype=f64, owned)

// 填充数组 (f64), shape=[10, 5], strides=[1, 16] — 不显示步长（padded_contiguous 为 true）
array([[0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0]], shape=[10, 5], dtype=f64, owned)

// 复数 (Complex<f64>), shape=[2]
array([Complex { re: 1.0, im: 2.0 }, Complex { re: 3.0, im: -4.0 }], shape=[2], dtype=complex128, owned)
```

### 24.6 大数组截断规则

当元素总数 > 1000 时触发截断，避免输出过长。截断采用两层策略：

**截断算法**：

| 步骤 | 规则 | 说明 |
|------|------|------|
| 1. 触发判断 | `len() > 1000` | 元素总数超过阈值时触发 |
| 2. 行级截断（沿 axis 0） | 保留前 `edge_items` 行和后 `edge_items` 行，中间替换为 `...` | `edge_items` 默认为 3。沿最高维度（axis 0）截断 |
| 3. 列级截断（最后一轴） | 当最后一轴长度 > 100 时，保留前 `edge_items` 个和后 `edge_items` 个元素，中间替换为 `...` | 仅在最后一轴过长时触发，避免单行输出过宽 |
| 4. 尾部标注 | 在数据内容结束后、元数据字段之前，追加 `, ... showing X of Y elements` | X 为实际显示的元素数，Y 为总元素数（`len()`）。仅 Debug 输出中标注；Display 输出不标注，仅以 `...` 标记省略位置 |

**截断参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `threshold` | 1000 | 触发截断的元素总数阈值 |
| `edge_items` | 3 | 截断时首尾保留的行数/列数 |

当前版本使用固定默认值，不提供用户可配置的截断参数。

**可配置性规划**：截断阈值（默认 1000 元素）和边缘显示数量（默认 3）可通过 `set_print_options()` 配置，类似 numpy 的 printoptions。后续版本将提供 `PrintOptions` 结构体，支持 `threshold`、`edge_items`、`linewidth`、`precision` 等参数的运行时调整。

**截断示例（Display）**：

```
// 1D f64, shape=[5000], 元素总数 > 1000
array([0.0, 1.0, 2.0, ..., 4997.0, 4998.0, 4999.0])

// 2D f64, shape=[1000, 5], 元素总数 5000 > 1000
array([[0.0, 1.0, 2.0, 3.0, 4.0],
       [5.0, 6.0, 7.0, 8.0, 9.0],
       [10.0, 11.0, 12.0, 13.0, 14.0],
       ...,
       [4985.0, 4986.0, 4987.0, 4988.0, 4989.0],
       [4990.0, 4991.0, 4992.0, 4993.0, 4994.0],
       [4995.0, 4996.0, 4997.0, 4998.0, 4999.0]])
```

**截断示例（Debug）**：

```
// 1D f64, shape=[5000], owned
array([0.0, 1.0, 2.0, ..., 4997.0, 4998.0, 4999.0], shape=[5000], dtype=f64, owned, ... showing 6 of 5000 elements)

// 2D f64, shape=[1000, 5], owned
array([[0.0, 1.0, 2.0, 3.0, 4.0],
       [5.0, 6.0, 7.0, 8.0, 9.0],
       [10.0, 11.0, 12.0, 13.0, 14.0],
       ...,
       [4985.0, 4986.0, 4987.0, 4988.0, 4989.0],
       [4990.0, 4991.0, 4992.0, 4993.0, 4994.0],
       [4995.0, 4996.0, 4997.0, 4998.0, 4999.0]], shape=[1000, 5], dtype=f64, owned, ... showing 30 of 5000 elements)
```

### 24.7 空数组与 0D 标量的完整格式

**空数组**：shape 中任一轴为 0 的数组。

| 场景 | Display | Debug |
|------|---------|-------|
| shape=[0] | `array([])` | `array([], shape=[0], dtype=f64, owned)` |
| shape=[0, 3] | `array([])` | `array([], shape=[0, 3], dtype=f64, owned)` |
| shape=[2, 0] | `array([])` | `array([], shape=[2, 0], dtype=f64, owned)` |

空数组无元素可显示，数据部分固定为 `[]`。Debug 中仍显示完整的 shape/dtype/存储模式。

**0D 标量**（shape=[], ndim=0, len()=1）：

| 场景 | Display | Debug |
|------|---------|-------|
| f64 值 42.0 | `array(42.0)` | `array(42.0, shape=[], dtype=f64, owned)` |
| i64 值 7 | `array(7)` | `array(7, shape=[], dtype=i64, owned)` |
| bool 值 true | `array(true)` | `array(true, shape=[], dtype=bool, owned)` |
| Complex\<f64\> 值 1+2i | `array(1+2i)` | `array(Complex { re: 1.0, im: 2.0 }, shape=[], dtype=complex128, owned)` |

0D 标量的数据部分**不带中括号**，直接显示标量值（元素自身的 Display/Debug 格式）。

### 24.8 辅助格式化 trait

| trait | 支持状态 | 说明 |
|-------|----------|------|
| `fmt::Display` | ✅ 实现 | 面向用户的简洁输出（见 §24.4） |
| `fmt::Debug` | ✅ 实现 | 面向诊断的完整输出（见 §24.5） |
| `fmt::LowerExp` | ❌ 当前版本不支持 | 科学计数法格式化（`{:.e}`）。未来版本可考虑支持 |
| `fmt::UpperExp` | ❌ 当前版本不支持 | 大写科学计数法格式化（`{:.E}`）。未来版本可考虑支持 |
| `fmt::Binary` 等 | ❌ 不支持 | 无计划支持 |

**注**：科学计数法格式化是科学计算库的常见需求（如 `{:.3e}` 控制精度），当前版本暂不实现，但须在 `fmt::Display` 实现中预留扩展点（如内部使用独立的数据格式化函数，便于未来添加精度控制参数）。

---

## 25. FFI 集成

### 25.1 指针 API

FFI 指针 API 直接调用 TensorBase 的指针方法（定义见 §8.3 指针查询），此处列出 FFI 场景下的使用说明：

| 方法 | 说明 |
|------|------|
| as_ptr() | 返回数据起始位置的不可变原始指针 `*const A`，即 `storage.as_ptr() + offset`（见 §8.1 指针语义分层） |
| as_mut_ptr() | 返回数据起始位置的可变原始指针 `*mut A`，即 `storage.as_mut_ptr() + offset`（须可写存储）。FFI 调用方须确保写入不越过 `capacity() - offset` 边界（`capacity()` 已作为 TensorBase 公共方法暴露，见 §8.3） |
| as_ptr_unchecked() | unsafe，不检查偏移量有效性 |
| capacity() | 返回物理缓冲区容量（含 padding，元素单位）。FFI 调用方据此验证裸指针写入边界：写入范围须在 `[offset, offset + capacity())` 内（相对于 `storage.as_ptr()`） |

**ArcRepr FFI 安全警告**：

ArcRepr 是 `Send + Sync`（§10.1），多线程可共享同一底层 buffer。从 ArcTensor 获取的裸指针不阻止其他线程触发 `make_mut()` 深拷贝。若 Arc 引用计数 > 1，任何时刻的 `make_mut()` 都可能使已获取的裸指针失效（旧 buffer 被 drop）。FFI 场景下的安全策略：

| 策略 | 方法 | 说明 |
|------|------|------|
| 独占所有权 | `Arc::try_unwrap(arc_tensor)` 转为 Owned 后再获取裸指针 | 零拷贝，推荐 |
| 借用保护 | 通过 `&ArcTensor` 借用获取 `as_ptr()`，借用期间编译器阻止 `make_mut()` | Rust 借用检查器保证 |
| 引用计数锁定 | 确保 Arc 引用计数为 1 后再获取裸指针 | 运行时检查 |

### 25.2 不安全构造函数

#### 25.2.1 from_raw_parts（从外部内存构造 Tensor）

从裸指针构造 Tensor，获取内存所有权。构造后 Tensor 在 drop 时释放缓冲区内存。

| 方法 | 签名 |
|------|------|
| `Tensor::from_raw_parts` | `unsafe fn from_raw_parts(ptr: *mut A, shape: D, strides: D, offset: usize, capacity: usize) -> Self where D: Dimension` |

**Safety 前置条件**（调用方须保证）：

| 条件 | 说明 |
|------|------|
| ptr 非空且对齐 | `ptr` 须非空且满足 `align_of::<A>()` 对齐要求 |
| ptr 指向有效内存 | `ptr` 指向的缓冲区须包含 `capacity` 个有效的 `A` 类型元素（可能包含 padding 元素） |
| 内存所有权 | `ptr` 指向的内存须通过兼容的分配器分配（与 Xenon 内部使用的分配器一致），构造后 Tensor 获得 ownership，drop 时将释放该内存 |
| shape × strides 安全 | 任意有效索引 `index[k] ∈ [0, shape[k])` 计算的偏移量 `offset + Σ(index[k] * strides[k])` 须落在 `[0, capacity)` 范围内。实现须验证此条件，违反时 panic |
| strides 存储格式 | strides 以 `usize` 存储（与 Tensor 内部一致），按 `isize` 解释时为有符号步长（见 §3.2） |
| 初始化状态 | 若 `A` 需要 drop（非 trivially-copyable），则 `capacity` 个元素须全部已初始化 |

#### 25.2.2 from_raw_parts_view（从外部内存构造只读视图）

从裸指针构造只读视图，不获取内存所有权。视图的生命周期由调用方通过 `'a` 参数指定。

| 方法 | 签名 |
|------|------|
| `TensorView::from_raw_parts_view` | `unsafe fn from_raw_parts_view<'a>(ptr: *const A, shape: D, strides: D, offset: usize) -> TensorView<'a, A, D> where D: Dimension` |

**Safety 前置条件**：

| 条件 | 说明 |
|------|------|
| ptr 非空且对齐 | 同 `from_raw_parts` |
| 生命周期 | `'a` 期间 ptr 指向的内存须保持有效且不可被可变访问 |
| shape × strides 安全 | 同 `from_raw_parts` |
| 初始化 | 所有通过 shape × strides 可达的元素须已初始化 |

#### 25.2.3 from_raw_parts_view_mut（从外部内存构造可变视图）

| 方法 | 签名 |
|------|------|
| `TensorViewMut::from_raw_parts_view_mut` | `unsafe fn from_raw_parts_view_mut<'a>(ptr: *mut A, shape: D, strides: D, offset: usize) -> TensorViewMut<'a, A, D> where D: Dimension` |

**Safety 前置条件**：同 `from_raw_parts_view`，额外要求 `'a` 期间 ptr 指向的内存无其他访问（独占语义）。

#### 25.2.4 into_raw_parts（所有权转移到外部）

解构 Tensor 为原始组件，阻止 Rust 释放内存。调用后 Tensor 被 consume，调用方负责后续内存管理。

| 方法 | 签名 |
|------|------|
| `Tensor::into_raw_parts` | `fn into_raw_parts(self) -> (*mut A, D, D, usize, usize)` |

**返回值**：

| 返回值 | 说明 |
|--------|------|
| `*mut A` | 缓冲区起始指针（`storage.as_ptr()`，不含 offset） |
| `D`（第一个） | shape |
| `D`（第二个） | strides（内部 `usize` 格式，按 `isize` 解释见 §3.2） |
| `usize`（第一个） | offset（元素单位） |
| `usize`（第二个） | capacity（元素单位，含 padding） |

**使用约束**：

- 仅 `Owned` 存储模式支持此方法。对 ViewRepr / ViewMutRepr 调用编译错误（视图不拥有内存），对 ArcRepr 调用编译错误（Arc 内存管理不可拆解）
- 调用方须通过以下方式之一释放内存：(1) 使用 `Tensor::from_raw_parts` 重建 Tensor 让 Rust drop 释放；(2) 使用与 Xenon 兼容的分配器手动释放（如 `alloc::alloc::dealloc`，须确保与分配时的对齐一致）
- 调用后原始 Tensor 的 drop 不会执行（`mem::forget` 语义）

### 25.3 形状与步长查询

`shape()` / `strides()` / `offset()` 在 §8.3 中定义，FFI 集成直接调用即可。此处补充字节单位 API：

| 方法 | 签名 | 说明 |
|------|------|------|
| strides_bytes() | `&self -> InlineArray<isize, 8>` | 返回各轴步长的栈分配数组（字节单位，有符号），每次调用重新计算 `O(ndim)`，逐元素计算 `stride_in_bytes = (stride as isize) * size_of::<A>()`。使用 `InlineArray<isize, 8>`（与 `IxDyn` 内部存储一致，见 §3.1）避免堆分配——ndim 通常 ≤ 6（Ix0-Ix6），远小于 8 的 inline 容量。对 `IxDyn`（理论上 ndim 可超过 8），超过 inline 容量时自动回退堆分配。若需调用方控制缓冲区，可使用 `fn strides_bytes_to(&self, buf: &mut [isize])` 将结果写入调用方提供的缓冲区（要求 `buf.len() >= self.ndim()`） |
| offset_bytes() | `&self -> usize` | 返回数据起始偏移量（字节单位），即 `self.offset() * size_of::<A>()`。FFI 场景（如 `memcpy`、GPU 上传）需要字节级起始偏移 |

### 25.4 BLAS 兼容性

#### 25.4.1 BLAS 兼容性 API

| 方法 | 签名 | 说明 |
|------|------|------|
| lda() | `&self -> usize` | 返回 leading dimension（正整数），用于 BLAS 的 LDA/LDB/LDC 参数。详细语义见 §25.4.2 |
| is_blas_compatible() | `&self -> bool` | 检查内存布局是否可直接传递给 BLAS。详细规则见 §25.4.3 |
| blas_layout() | `&self -> Option<Order>` | 返回 BLAS 布局标识（F/C/None），None 表示不兼容 |
| blas_trans() | `&self -> Trans` | 返回 BLAS Trans 参数，基于当前内存布局的物理存储方向推断。详细规则见 §25.4.4 |
| ensure_blas_compatible() | `&self, layout: Order -> CowTensor<'_, A, D>` | 确保内存布局为 BLAS 兼容。若已兼容则返回借用视图（零拷贝），否则拷贝到新的连续布局。详见 §25.4.5 |

#### 25.4.2 lda() 详细语义

`lda()` 返回物理存储的 leading dimension，始终返回 `usize`（正整数，取绝对值）。

**定义**：

| 布局方向 | lda() 返回值 | 语义 |
|----------|-------------|------|
| F-order（F-contiguous 或 F-padded-contiguous） | `|strides[1]|`（取绝对值） | F-order 的 leading dimension 是列间距（物理列长度，含 padding） |
| C-order（C-contiguous 或 C-padded-contiguous） | `|strides[ndim-2]|`（取绝对值） | C-order 的 leading dimension 是行间距 |

**边界情况**：

| 场景 | 行为 | 说明 |
|------|------|------|
| 0D 标量 | panic（`"lda() requires ndim >= 2, got {ndim}"`） | 标量无 leading dimension 概念 |
| 1D 向量 | panic（同上） | 1D 无 leading dimension 概念 |
| 非连续数组 | panic（`"lda() requires contiguous layout"`） | 非连续数组无确定的 leading dimension，须先通过 `ensure_blas_compatible()` 转换 |
| 填充数组 | 返回填充后的物理步长（如 F-order M_padded） | 可直接传递给 BLAS 的 LDA 参数 |
| 子矩阵视图（切片自大矩阵） | 返回源矩阵的物理步长（如 `strides[1]=10`） | BLAS LDA 要求物理存储间距，非逻辑维度 |
| 负步长 | 返回 `|stride|`（绝对值） | BLAS LDA 参数须为正整数 |
| batch 视图（3D+ 的 2D 切片） | 返回该 2D 切片的 leading dimension，不受 batch 轴影响 | 适用于批量 BLAS 调用 |

#### 25.4.3 is_blas_compatible() 判定规则

检查内存布局是否可直接传递给 BLAS。须同时满足以下所有条件：

| 条件 | 说明 |
|------|------|
| 宽松连续 | `is_padded_contiguous()` 为 true（严格 F/C-contiguous 或填充后的 F/C-padded-contiguous） |
| 正步长 | 所有轴步长 > 0（无负步长）。BLAS 不支持负步长布局 |
| 无零步长 | `has_zero_stride() == false`（无广播维度）。广播视图不可直接传 BLAS |
| ndim ≥ 1 | 0D 标量不兼容 BLAS |

**设计说明**：BLAS 的 LDA 参数天然支持 leading dimension 不等于逻辑维度，因此填充数组是 BLAS 兼容的。

#### 25.4.4 blas_trans() 判定逻辑

`blas_trans()` 根据当前 Tensor 的**物理内存布局**推断需要传递给 BLAS 的转置参数。BLAS 函数通常接受 layout 参数（如 `CblasRowMajor` / `CblasColMajor`）和 trans 参数（`CblasNoTrans` / `CblasTrans`），两者的组合表达了"内存中的数据排列"与"逻辑上的矩阵含义"之间的关系。

**维度限制**：`blas_trans()` 和 `blas_layout()` 仅对 **≤2D** Tensor 有明确语义。3D 及以上 Tensor 的 `.t()` 仅交换最后两个轴（见 §17.6），不产生全局转置，因此 `blas_layout()` 返回 `None`，`blas_trans()` 在 3D+ 上调用会 panic。对 3D+ batch tensor 调用 BLAS 时，应通过 `index_axis()` 或 `.slice()` 提取 2D 子视图后再使用 BLAS API。

**基础判定真值表**：

| 布局状态 | blas_layout() | blas_trans() | 说明 |
|----------|---------------|--------------|------|
| F-contiguous 或 F-padded-contiguous | Some(Order::F) | Trans::None | 内存按列优先排列，与 BLAS 列优先 layout 一致，无需转置 |
| C-contiguous 或 C-padded-contiguous | Some(Order::C) | Trans::None | 内存按行优先排列，与 BLAS 行优先 layout 一致，无需转置 |
| 非连续（非 F 亦非 C） | None | 不可调用（panic） | 须先通过 `ensure_blas_compatible()` 转换为连续布局 |
| 广播视图（has_zero_stride） | None | 不可调用（panic） | 广播视图无确定的物理布局 |

**带转置的视图**：

对于通过 `.t()` 或 `.reversed_axes()` 获得的转置视图，其物理内存布局与逻辑布局相反。`blas_layout()` 返回的是**原始内存的物理方向**（即数据实际在内存中的排列顺序），而 `blas_trans()` 补偿逻辑方向与物理方向的差异：

| 原始 Tensor 布局 | 转置后物理布局 | blas_layout() | blas_trans() | 说明 |
|------------------|---------------|---------------|--------------|------|
| F-contiguous | C-contiguous（步长反转） | Some(Order::F) | Trans::Transpose | 物理内存仍为 F，但逻辑为 C。告诉 BLAS "按 F layout 读取，但需要转置" |
| C-contiguous | F-contiguous（步长反转） | Some(Order::C) | Trans::Transpose | 物理内存仍为 C，但逻辑为 F。告诉 BLAS "按 C layout 读取，但需要转置" |

**ConjTranspose 语义**：

| 元素类型 | Trans::ConjTranspose 行为 |
|----------|--------------------------|
| 实数（f32/f64） | 等价于 `Trans::Transpose`（实数无虚部，共轭为 identity） |
| 复数（Complex\<f32\>/Complex\<f64\>） | 共轭转置（转置 + 取共轭） |

`blas_trans()` 本身不自动推断共轭——因为 Tensor 的布局信息不携带"是否需要共轭"的语义。用户需在 BLAS 调用时手动将 `Trans::Transpose` 替换为 `Trans::ConjTranspose`。仅当用户显式请求共轭转置时（如对 Tensor 调用 `.conj().t()`），应在 BLAS 参数中传递 `'C'`。

#### 25.4.5 ensure_blas_compatible() 布局转换

确保 Tensor 内存布局为 BLAS 兼容的标准化方法。避免上游库重复实现"检查兼容 → 不兼容则拷贝"的逻辑。

**CowTensor 定义**（亦适用于 §8.2 类型别名体系）：

| 类型 | 定义 | 说明 |
|------|------|------|
| `CowTensor<'a, A, D>` | 枚举类型 | Copy-on-Write Tensor 引用 |
| `CowTensor::Borrowed(TensorView<'a, A, D>)` | 变体 | 借用源 Tensor（零拷贝） |
| `CowTensor::Owned(Tensor<A, D>)` | 变体 | 新分配的连续 Tensor |

**行为**：

| 当前布局 | 请求布局 | 返回值 | 内存操作 |
|----------|----------|--------|----------|
| 严格 F-contiguous | F | Borrowed（借用视图） | 零拷贝 |
| F-padded-contiguous | F | Borrowed（借用视图） | 零拷贝（填充是 BLAS 兼容的） |
| 严格 C-contiguous | C | Borrowed（借用视图） | 零拷贝 |
| C-padded-contiguous | C | Borrowed（借用视图） | 零拷贝 |
| 非连续 | 任意 | Owned（新分配） | 拷贝到请求布局的连续数组 |
| F-contiguous | C | Owned（新分配） | 拷贝为 C-order |
| C-contiguous | F | Owned（新分配） | 拷贝为 F-order |

### 25.5 Trans 枚举定义

| 变体 | 含义 | BLAS 字符 |
|------|------|-----------|
| `Trans::None` | 不转置（原始布局） | `'N'` |
| `Trans::Transpose` | 转置（行列互换） | `'T'` |
| `Trans::ConjTranspose` | 共轭转置（仅 Complex 类型有意义） | `'C'` |

Trans 枚举用于 FFI 与 BLAS 交互时描述矩阵的转置状态。详细判定逻辑见 §25.4.4。

### 25.6 索引转换

| 方法 | 签名 | 说明 |
|------|------|------|
| index_to_ptr(index) | `fn index_to_ptr(&self, index: I) -> *const A where I: IntoDimension<Dim = D>` | 将多维索引转换为对应元素的原始指针。索引维度类型须与 Tensor 维度类型 `D` 匹配（编译时检查） |
| index_to_ptr_unchecked(index) | `unsafe fn index_to_ptr_unchecked(&self, index: I) -> *const A where I: IntoDimension<Dim = D>` | 不检查索引越界的变体 |
| index_to_offset(index) | `fn index_to_offset(&self, index: I) -> usize where I: IntoDimension<Dim = D>` | 将多维索引转换为相对于数据起始位置（`as_ptr()`）的元素偏移量 |
| index_to_offset_unchecked(index) | `unsafe fn index_to_offset_unchecked(&self, index: I) -> usize where I: IntoDimension<Dim = D>` | 不检查索引越界的变体 |

**越界行为**：

| 方法 | 越界时行为 |
|------|----------|
| index_to_ptr | panic（`IndexOutOfBounds`） |
| index_to_ptr_unchecked | UB（调用方须保证索引有效） |
| index_to_offset | panic（`IndexOutOfBounds`） |
| index_to_offset_unchecked | UB（调用方须保证索引有效） |

### 25.7 FFI 使用模式

#### 25.7.1 模式一：将 Tensor 数据传递给 C 库（只读）

```
// 方法 A：借用 Tensor（推荐，同步场景）
let ptr = tensor.as_ptr();
let shape = tensor.shape();
let strides = tensor.strides_bytes();
c_function(ptr, shape.as_ptr(), strides.as_ptr());
// tensor 在此期间存活，ptr 有效

// 方法 B：转移所有权（异步场景）
let (ptr, shape, strides, offset, cap) = tensor.into_raw_parts();
c_function_async(ptr, shape.slice().as_ptr(), ...);
// 异步完成后回收所有权：
let tensor = unsafe { Tensor::from_raw_parts(ptr, shape, strides, offset, cap) };
```

#### 25.7.2 模式二：将 Tensor 数据传递给 C 库（可写）

```
let ptr = tensor.as_mut_ptr();
let cap = tensor.capacity();  // C 库写入不得超过此边界
let lda = tensor.lda();
c_blas_function(ptr, lda, ...);
```

#### 25.7.3 模式三：从 C 库接收数据构造 Tensor

```
// C 库返回裸指针和形状信息（须确保分配器兼容）
let ptr: *mut f64 = c_library_alloc(rows, cols);
let shape = Ix2(rows, cols);
let strides = shape.default_strides();
let capacity = rows * cols;
let tensor = unsafe { Tensor::from_raw_parts(ptr, shape, strides, 0, capacity) };
// Tensor drop 时将释放 ptr 指向的内存
```

#### 25.7.4 模式四：BLAS 集成

```
let compat = tensor.ensure_blas_compatible(Order::F);
if let Some(layout) = compat.blas_layout() {
    let trans = compat.blas_trans();
    let lda = compat.lda();
    let ptr = compat.as_ptr();
    // blas_dgemm(layout, trans, ..., ptr, lda, ...);
}
```

---

## 26. 临时工作空间

### 26.1 工作空间属性

| 属性 | 要求 |
|------|------|
| 对齐 | 默认 64 字节对齐，须支持调用方指定更大对齐（如 128 字节） |
| 生命周期 | 借用语义：工作空间借出期间不可再次借出，归还后可复用 |
| 增长策略 | 请求超出当前容量时自动扩容，不缩容；扩容后保持对齐。提供 `shrink_to(new_len)` 方法将未使用空间归还给分配器，调用方须保证 `new_len ≤ 当前已分配容量`，否则 panic |
| 线程亲和性 | 工作空间不绑定线程；线程安全由调用方通过借用规则保证 |
| 零初始化 | 不保证零初始化（性能考虑），调用方须自行初始化 |
| 分配模型 | 采用 bump allocator 模型：内部维护一个递增指针，每次 `borrow_*` 推进指针（含对齐填充），`reset()` 将指针重置到缓冲区起始位置。归还（`&mut` 引用生命周期结束）后指针不回退——内存仅在调用 `reset()` 后可重用 |
| no_std 兼容 | 支持 `no_std` + `alloc` 环境，使用 `alloc::alloc` 全局分配器进行堆分配 |
**注意**：bump allocator 不支持单个分配的释放。调用者应确保 `reset()` 调用频率足够高以避免内存累积。对于嵌套并行操作，推荐使用 Scope 语义（`split_at` 返回的子 workspace 在父 scope 重置时一并回收）。

### 26.2 对齐填充规则

每次借出操作在返回缓冲区前须按以下规则处理对齐：

1. 计算对齐后的起始偏移：`aligned_offset = align_up(current_ptr, requested_alignment)`
2. 计算实际消耗的缓冲区空间：`consumed = aligned_offset - buffer_start + len_bytes`
3. 检查 `consumed ≤ capacity`，若不足则按 §26.1 增长策略扩容
4. 返回 `[aligned_offset, aligned_offset + len_bytes)` 区间
5. 推进 `current_ptr = aligned_offset + len_bytes`

**各借用方法的对齐请求值**：

| 方法 | `requested_alignment` 取值 |
|------|---------------------------|
| `borrow_bytes` | 工作空间的当前对齐值（构造时指定，默认 64） |
| `borrow_aligned::<A>` | `max(align_of::<A>(), 工作空间当前对齐)` |
| `borrow_scratch` | `ScratchRequirement::align` |

**示例**：工作空间 `alignment=64`，`current_ptr=72`，调用 `borrow_aligned::<f64>(3)`：
- `requested_alignment = max(8, 64) = 64`
- `aligned_offset = align_up(72, 64) = 128`
- `len_bytes = 3 × 8 = 24`
- 返回 `[128, 152)`，`current_ptr` 推进到 `152`

### 26.3 分割与嵌套

| 属性 | 要求 |
|------|------|
| 嵌套借用 | 支持从同一工作空间分割多个不重叠的子缓冲区 |
| 并行分割 | 支持 `split_at(mid)` 将工作空间分割为两个不重叠的子工作空间，各自可独立借出，O(1)，不涉及内存分配。`split_at` 按值获取 `self`（消费式分割），返回两个独立的 owned `Workspace`，各自的内部指针从各自的缓冲区起始位置开始 |
| 递归分割 | 子工作空间可继续分割（支持递归二分），最小分割粒度为 16 字节（小于 16 字节的分割请求 panic）。**设计理由**：防止无限递归分割产生零大小子空间导致的无限循环；16 字节下限覆盖所有常见 SIMD 类型的对齐需求 |

**`split_at` 签名与语义**：

| 方法 | 签名 | 说明 |
|------|------|------|
| `split_at` | `fn split_at(self, mid: usize) -> (Workspace, Workspace)` | 消费 self，在 `mid` 字节处分割为两个独立工作空间。`mid` 须满足 `16 ≤ mid ≤ capacity - 16`，否则 panic（保证每个子工作空间至少 16 字节） |

### 26.4 工作空间线程安全

`Workspace` 实现 `Send` 但不实现 `Sync`——工作空间可在线程间移动（Send），但不可被多线程同时访问（非 Sync）。

`split_at` 按值消费父工作空间，返回的两个子 `Workspace` 各自为 `Send`，可分别移动到不同线程使用。每个子工作空间在同一时刻只能被一个线程访问（由 `&mut self` 方法签名在编译时保证——`borrow_*` 方法接受 `&mut self`，独占访问）。父工作空间被消费后自然不存在，"不可借出"由所有权系统（而非借用检查器）保证。

### 26.5 扩容安全性约束

| 约束 | 说明 |
|------|------|
| 借用期间不可扩容 | `borrow_*` 方法接受 `&mut self`，返回的 `&mut [u8]` / `&mut [MaybeUninit<A>]` 生命周期绑定到 `&mut Workspace`。活跃借用期间无法再次调用任何 `&mut self` 方法，因此扩容不会在存在活跃借用时发生 |
| 分割后不可扩容 | `split_at` 消费父工作空间，子工作空间拥有独立缓冲区。子工作空间的扩容与原工作空间无关——子工作空间扩容可能触发 `realloc`，但仅影响该子工作空间自身的缓冲区，不影响其他子工作空间 |

### 26.6 ScratchRequirement 类型

`ScratchRequirement` 描述操作所需的临时缓冲区大小和对齐要求，供 `borrow_scratch` 使用。

| 属性 | 定义 |
|------|------|
| 类型 | `#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub struct ScratchRequirement { pub size: usize, pub align: usize }` |
| `size` | 所需字节数 |
| `align` | 所需对齐（须为 2 的幂） |
| 构造方式 | `ScratchRequirement::new(size: usize, align: usize)`；`align` 非 2 的幂时 panic |
| 便捷方法 | `ScratchRequirement::for_elements::<A>(len: usize) -> Self`，等价于 `ScratchRequirement::new(len * size_of::<A>(), align_of::<A>())` |

### 26.7 借出与归还 API

| 方法 | 签名 | 说明 |
|------|------|------|
| `borrow_bytes` | `fn borrow_bytes(&mut self, len: usize) -> &mut [u8]` | 借出指定字节数的缓冲区，按当前对齐对齐（对齐填充规则见 §26.2）。借出期间不可再次借出同一区间 |
| `borrow_aligned` | `fn borrow_aligned<A>(&mut self, len: usize) -> &mut [MaybeUninit<A>]` | 借出指定元素数的类型化缓冲区，按 `max(align_of::<A>(), 当前对齐)` 对齐。返回 `&mut [MaybeUninit<A>]` 而非 `&mut [A]`——工作空间不保证零初始化（见 §26.1），返回已初始化引用将构成 UB。调用方须自行初始化后通过 `MaybeUninit::assume_init` 转换。对齐不足时 panic |
| `borrow_scratch` | `fn borrow_scratch(&mut self, req: ScratchRequirement) -> &mut [u8]` | 根据 `ScratchRequirement` 借出足够大小和指定对齐的缓冲区（类型定义见 §26.6） |
| 归还 | 借出的 `&mut` 引用生命周期绑定到 `&mut Workspace`，引用释放后自动归还，无需显式调用 | 借用检查器保证借出期间不存在二次借出 |

### 26.8 查询与管理 API

| 方法 | 签名 | 说明 |
|------|------|------|
| `capacity` | `fn capacity(&self) -> usize` | 返回当前已分配的字节容量 |
| `remaining` | `fn remaining(&self) -> usize` | 返回从当前内部指针到缓冲区末尾的剩余可用字节数（不含对齐填充开销） |
| `reset` | `fn reset(&mut self)` | 将内部指针重置到缓冲区起始位置，使所有已借出空间可重用。要求无活跃借用（由 `&mut self` 保证）。已借出的内存在借用引用生命周期结束前仍有效 |
| `shrink_to` | `fn shrink_to(&mut self, new_len: usize)` | 将未使用空间归还给分配器。调用方须保证 `new_len ≤ capacity()`，否则 panic。要求无活跃借用（由 `&mut self` 保证） |

---

# 第五部分：工程保障

## 27. 错误处理

### 27.1 错误分类

**所有公开错误枚举均标注 `#[non_exhaustive]`**，确保未来新增错误变体不构成 breaking change。

**错误类型体系**（两级架构）：

| 层级 | 错误类型 | 枚举变体名 | 触发场景 | 处理方式 |
|------|----------|------------|----------|----------|
| **独立错误** | ShapeMismatch | `XenonError::ShapeMismatch` | 二元运算/zip 形状不兼容且无法广播 | Result |
| | BroadcastError | `XenonError::BroadcastError` | 广播规则不满足（非 size-1 维度不等） | Result |
| | LayoutMismatch | `XenonError::LayoutMismatch` | 要求连续布局但输入非连续（如 flatten_view 非连续数组） | Result |
| | InvalidAxis | `XenonError::InvalidAxis` | 轴索引超出维度数 | Result |
| | InvalidShape | `XenonError::InvalidShape` | 构造参数形状无效（如 cat 输入 ndim 不一致） | Result |
| | DimensionMismatch | `XenonError::DimensionMismatch` | 静态维度与动态维度互转时维度数不匹配 | Result |
| | EmptyArray | `XenonError::EmptyArray` | 对空数组执行 min/max/argmin/argmax 或 mean/var/std | Result |
| | InvalidInput | `XenonError::InvalidInput` | 参数语义不合法（非结构/索引类问题）：如 histogram bins 为负数/零、自定义边界数组不严格递增或含 NaN/Inf、同时提供 bins 数组和 range 参数、从空数据推断 range、bincount 输出长度溢出 | Result |
| | InvalidLayout | `XenonError::InvalidLayout` | 布局/对齐验证失败：`from_raw_parts` 等 unsafe 构造器中 shape × strides 不一致（如 strides 导致越界偏移）、指针非空但未满足 `align_of::<A>()` 对齐要求、对齐值非 2 的幂或小于元素自然对齐（见 §4.3） | Result（safe 构造路径）/ panic（unsafe 构造路径的前置条件违反，见 §25.2） |
| | AllocationError | `XenonError::AllocationError` | 内存分配失败（超大数组、系统内存不足） | panic |
| | IndexOutOfBounds | `XenonError::IndexOutOfBounds` | 索引越界（含多维索引、take/put 索引数组） | 多维索引：panic（checked）/ UB（unchecked）；take/put 索引数组：Result |
| | InconsistentLengths | `XenonError::InconsistentLengths` | `TryFrom<Vec<Vec<A>>>` 内层 Vec 长度不一致 | Result |

**`InvalidInput` 与其他错误类型的边界**：`InvalidInput` 覆盖"参数语义不合法"（如负数 bins、NaN 边界），区别于 `InvalidShape`（结构不合法，如 ndim 不一致）和 `IndexOutOfBounds`（索引超范围）。当参数问题可归为更具体的错误类型时，优先使用更具体的类型。

### 27.2 错误类型诊断字段

所有返回 `Result` 的错误类型须携带以下诊断信息，以 `Display` 输出人类可读描述：

| 错误类型 | 诊断字段 | 说明 |
|----------|----------|------|
| ShapeMismatch | `expected: ShapeData`, `actual: ShapeData` | 期望与实际形状。`ShapeData` 使用 `InlineArray<usize, 8>` 实现，低维度（≤ 8）时栈分配避免堆开销，高维度时自动退化为堆分配 |
| BroadcastError | `left: ShapeData`, `right: ShapeData`, `conflicting_axis: usize` | 冲突的两个形状及冲突轴。`ShapeData` 同上 |
| LayoutMismatch | `requested: &'static str`, `actual_layout: &'static str` | 期望布局（如 "C-contiguous"/"F-contiguous"）与实际布局描述 |
| InvalidAxis | `axis: usize`, `ndim: usize` | 请求轴与实际维度数 |
| InvalidShape | `reason: &'static str` | 人类可读原因（如 "0 inputs" / "ndim mismatch: expected 3, got 2"） |
| DimensionMismatch | `expected: usize`, `actual: usize` | 期望与实际维度数 |
| EmptyArray | `operation: &'static str` | 触发操作名（如 "min"/"argmax"/"mean"/"var"） |
| InvalidInput | `reason: Cow<'static, str>` | 人类可读原因描述（如 "bins must be >= 1, got 0" / "bin edges must be strictly monotonic" / "autodetected range is not finite: input contains NaN and no explicit range provided"）。使用 `Cow<'static, str>` 允许静态字符串和动态格式化消息零开销共存 |
| InvalidLayout | `reason: &'static str` | 人类可读原因（如 "strides produce out-of-bounds offset" / "pointer alignment 4 does not meet required alignment 8" / "alignment must be a power of 2"） |
| IndexOutOfBounds | `index: usize`, `axis_len: usize`, `axis: usize` | 越界索引值、轴长度、轴编号（仅 take/put Result 场景） |
| InconsistentLengths | `expected: usize`, `found: Vec<usize>` | 期望的行长度与各实际行长度 |

panic 类错误（`AllocationError`）通过 panic message 传递诊断信息，不使用结构化字段。`IndexOutOfBounds` 在 panic 场景（多维索引越界）通过 panic message 传递诊断信息，在 Result 场景（take/put 索引数组越界）使用结构化字段。`InvalidLayout` 在 unsafe 构造路径（`from_raw_parts`，见 §25.2）中通过 panic message 传递诊断信息（调用方违反 Safety 前置条件），在 safe 构造路径中使用结构化字段返回 `Result`。

### 27.3 API错误类型映射

| API 类别 | 返回的错误类型 | 说明 |
|----------|---------------|------|
| reshape | `InvalidShape` | 仅形状不匹配时返回错误（布局不匹配时自动拷贝） |
| flatten_view | `LayoutMismatch` | 独立错误，仅检查连续性 |
| cat / stack | `InvalidShape`（0 个输入、ndim 不一致）/ `ShapeMismatch`（非拼接轴长度不等：cat 除拼接轴外各轴长度须一致，stack 所有输入形状须完全一致） | 独立错误；两者触发条件互斥，不重叠 |
| squeeze | `InvalidShape` | 独立错误 |
| broadcast（显式广播 API，§16.3） | `BroadcastError` | 广播形状不兼容，含冲突轴诊断信息（见 §27.2） |
| 逐元素运算隐式广播 | **panic**（`BroadcastError`） | 通过运算符重载（`Add`/`Sub`/`Mul`/`Div`）触发的隐式广播：因运算符 trait 的 `type Output = Tensor<A, D>` 无法返回 `Result`，广播失败时 **panic**（panic message 包含 `BroadcastError` 诊断信息，含冲突轴细节）。通过方法调用（`.add()`/`.sub()` 等）的显式广播也遵循相同行为——这些方法与运算符共享实现，同样 panic。若需可恢复的广播错误处理，使用 `broadcast` API（§16.3）显式广播后再运算 |
| 轴操作（sum_axis / cumsum / slice / index_axis 等） | `InvalidAxis` | axis >= ndim 时返回 `InvalidAxis`（Result）。注意：s![] 宏中轴数不匹配的行为不同——静态维度编译错误，动态维度 panic（见 §13.9） |
| 维度互转（into_dimension） | `DimensionMismatch` | 独立错误 |
| 逐元素运算（形状不一致，非广播场景） | `ShapeMismatch` | 形状不同且不满足广播条件（不应发生——广播会先捕获不兼容形状）；独立错误 |
| min/max/argmin/argmax（空数组） | `EmptyArray` | 独立错误 |
| mean/var/std（空数组） | `EmptyArray` | 除零无意义，独立错误 |
| 内存分配（构造器） | `AllocationError`（panic） | 不可恢复 |
| 索引访问 | `IndexOutOfBounds`（panic） / unchecked UB | 编程错误（多维索引硬编码或循环变量） |
| take / put | `IndexOutOfBounds`（Result） | 索引来自运行时动态数据（索引数组），属数据驱动错误 |
| `TryFrom<Vec<Vec<A>>>` | `InconsistentLengths`（Result） | `From` 实现使用 `expect(...)` 转为 panic；`TryFrom` 提供可恢复错误路径 |
| histogram / histogram_bin_edges（§15.6） | `InvalidInput`（Result） | bins 参数语义不合法（零/负数）、自定义边界数组不严格递增/含 NaN/Inf、bins 数组与 range 同时提供、从空数据推断 range、输入含 NaN 且未提供 range |
| bincount（§15.6） | `InvalidInput`（Result） | 输出长度溢出（`max(input) + 1 > isize::MAX`） |
| from_raw_parts（§25.2，unsafe） | `InvalidLayout`（panic） | unsafe 构造路径：调用方违反 Safety 前置条件（strides 导致越界偏移、指针未对齐）时 panic |
| safe 布局/对齐验证（构造器） | `InvalidLayout`（Result） | safe 路径：对齐值非 2 的幂或小于元素自然对齐、shape × strides 不一致 |

### 27.4 处理策略

**判断原则**：panic 用于"调用方违反了 API 前置条件"（编程错误，合理使用无法触发），Result 用于"输入数据本身的属性导致操作无法完成"（合法调用路径，调用方应处理）。具体规则：

| 策略 | 适用场景 | 典型例子 |
|------|----------|----------|
| 可恢复错误（形状、布局、轴、数据驱动索引越界） | 返回 Result | squeeze 非 size-1 轴 → `InvalidShape`；空数组 min/max/mean/var → `EmptyArray`；take/put 索引数组越界 → `IndexOutOfBounds` |
| 可恢复错误（参数语义不合法） | 返回 Result | histogram bins 为零/负数 → `InvalidInput`；自定义边界数组不严格递增 → `InvalidInput` |
| 编程错误（多维索引越界） | panic，同时提供 unsafe unchecked 变体 | `tensor[i,j]` 索引越界 → `IndexOutOfBounds` panic |
| 前置条件违反 | panic | bincount 负值输入（文档要求 ≥ 0）；`Vec<Vec<A>>` 内层长度不一致（文档要求一致）；arange step=0（文档要求 ≠ 0） |
| 内存分配失败 | panic（与 Rust 标准库 `Vec::push` 等行为一致）。调用方若需 graceful 处理，可在调用前预检元素数和预估内存占用。未来版本可考虑提供 `try_*` 变体返回 `Result` | |
| 错误信息 | 见下方「Error trait 条件编译」规则 | |

**Error trait 条件编译**：

本库支持 `no_std`（需 `alloc`，见 §1.3），所有错误类型须满足以下条件编译策略：

| Feature | trait 实现 | 说明 |
|---------|-----------|------|
| `std`（默认启用） | `impl std::error::Error for XenonError` 及各子类型 | 完整的 `Error` trait 支持，可与 `anyhow`/`eyre` 等生态无缝集成 |
| `no_std`（`default-features = false`） | 仅要求 `Display + Debug + Clone + Send + Sync` | `std::error::Error` 不可用；错误类型仍实现 `Display` 以提供人类可读描述，但**不实现** `Error` trait |

实现方式：使用 `cfg(feature = "std")` 条件编译块，在 `std` feature 下 `impl std::error::Error`，`no_std` 下省略。不引入额外依赖（不使用 `thiserror` 或 `core-error`）。

**AllocationError 与 no_std**：

`AllocationError` 在 `std` 环境下 panic（`panic!`），在 `no_std` 环境下的行为取决于全局分配器配置：

| 环境 | AllocationError 行为 | 说明 |
|------|---------------------|------|
| `std`（默认） | `panic!("allocation failed: requested {size} bytes")` | 与 `Vec::push` 等标准库行为一致 |
| `no_std` + `alloc` | 由 `alloc::alloc::handle_alloc_error` 决定（默认 abort） | 行为由调用方配置的全局分配器决定，Xenon 不额外干预 |

无论哪种环境，panic message 均包含请求的分配大小（字节），便于诊断。

### 27.5 Panic Safety

**原则**：panic 发生后，Tensor 的内部状态须满足：(1) 可安全 drop（无未定义行为、无内存泄漏）；(2) 不保证数据一致性（已修改的部分元素可能留在中间状态）。

| 操作场景 | Panic 时保证 | 说明 |
|----------|-------------|------|
| `make_mut()` 触发深拷贝时 panic（分配失败） | 原 ArcTensor 数据未被修改（引用计数不变）；已分配的新缓冲区被 drop 释放 | 深拷贝先分配新缓冲区再复制，分配失败时原数据完好 |
| `empty_uninit` 构造后 `assume_init` 前 panic | Tensor 内含 `MaybeUninit` 元素，drop 时不会读取未初始化数据（`MaybeUninit<T>` 的 drop 是 no-op） | 必须保证 `Tensor<MaybeUninit<A>, D>` 的 drop impl 不触及元素内容 |
| `checked_add` / `checked_mul` 累积运算 panic（如 `sum`、`cumsum`） | 中间结果被丢弃；Tensor 可安全 drop | 溢出检查在每步执行，panic 时无悬垂指针或未初始化内存 |
| `iter_mut()` 对广播视图 panic（见 §10.3） | 未修改任何元素；Tensor 保持原状态 | panic 在写入前触发，数据完好 |

**实现要求**：

- 所有 `drop` 实现不得 panic（遵循 `Drop` trait 的惯用规则）
- `Tensor<MaybeUninit<A>, D>` 的 drop 不得调用元素的 drop glue——仅释放缓冲区内存
- 涉及资源分配的操作（如 `make_mut` 深拷贝）须确保分配失败时原有数据不受影响

**并行操作 panic 安全**（feature `parallel`，需 `std`，见 §1.3）：

并行操作使用 rayon 线程池（见 §9.2），当单个线程 panic 时的行为遵循以下规则：

| 场景 | Panic 时保证 | 说明 |
|------|-------------|------|
| `par_iter()` / `par_iter_mut()` 逐元素操作 | 未 panic 的线程继续完成当前块；rayon `join` 在收集结果时 re-throw panic | rayon 的默认行为：单个任务 panic 不影响其他任务的内存安全性，`join` 返回时将 panic 传播到调用线程 |
| 并行归约（sum / prod 等） | 中间结果被丢弃；输出 Tensor 可安全 drop | 归约的累加器为线程局部变量，panic 时被 drop，无共享状态污染 |
| `make_mut()` 触发深拷贝时 panic（分配失败） | 原 ArcTensor 数据未被修改（引用计数不变）；已分配的新缓冲区被 drop 释放 | 深拷贝先分配新缓冲区再复制，分配失败时原数据完好 |
| 并行写入（`par_iter_mut()`）部分完成后 panic | 已修改的元素保持在最终写入状态；未修改的元素保持原值；Tensor 可安全 drop | 不保证数据一致性（部分修改），但保证内存安全（可安全 drop，无 UB） |

**实现约束**：

- 不使用 `catch_unwind` 包裹并行任务——panic 应立即传播，不吞没
- 共享缓冲区（如 ArcTensor 的引用计数）的修改须使用原子操作，确保 panic 时计数正确
- 并行迭代器的任务分割须确保每个任务的数据范围互不重叠（rayon 按 chunk 分割，每个线程独占一段连续区间）

---

## 28. 质量要求

### 28.1 文档

| 要求 | 范围 |
|------|------|
| doc comment | 所有公开 API |
| Safety 文档节 | 所有 unsafe 函数 |
| 使用示例 | 关键 API |

### 28.2 测试

| 要求 | 指标 |
|------|------|
| 测试类型 | 单元测试 + 集成测试 + 边界测试 |
| 行覆盖率（整体） | ≥ 90%（使用 `cargo llvm-cov` 测量，数值基础设施对正确性要求极高，高覆盖率是必要的质量保障） |
| unsafe 代码块覆盖率 | ≥ 95%（单独统计，覆盖指针运算、`get_unchecked`、`from_raw_parts` 等关键 unsafe 路径） |
| 分支覆盖率（关键路径） | ≥ 85%（覆盖边界检查、溢出检测、SIMD 回退、布局标志判定等分支） |

### 28.3 数值精度

**浮点与复数精度表**：

| 运算类别 | f64 精度 | f32 精度 | Complex\<f64\> 精度 | Complex\<f32\> 精度 |
|----------|----------|----------|---------------------|---------------------|
| 加减乘 | ≤ 1 ULP | ≤ 1 ULP | re/im 各 ≤ 1 ULP | re/im 各 ≤ 1 ULP |
| 除法 | ≤ 1 ULP | ≤ 1 ULP | re/im 各 ≤ 4 ULP | re/im 各 ≤ 4 ULP |
| 归约（sum/prod） | 相对误差 ≤ 1e-14（N ≤ 10⁶） | 相对误差 ≤ 1e-5（N ≤ 10⁶） | 模相对误差 ≤ 1e-13（N ≤ 10⁶） | 模相对误差 ≤ 1e-4（N ≤ 10⁶） |
| 超越函数 | 相对误差 ≤ 1e-14，ULP ≤ 4 | 相对误差 ≤ 1e-5，ULP ≤ 4 | 模相对误差 ≤ 1e-13 | 模相对误差 ≤ 1e-4 |

**归约精度条件说明**：

- 归约精度指标适用于元素数 N ≤ 10⁶ 的场景。对于 N > 10⁶ 的数组，`sum` 须使用补偿求和（如 Neumaier/Kahan summation）以达到表中精度要求；`prod` 的精度随 N 增长自然退化，不强制要求补偿精度
- 并行浮点归约：各线程独立累加后合并，合并顺序与单线程不一致时允许精度容差内差异（见 §28.5"并行确定性"行）
- 浮点 `sum` 在实现时须默认使用补偿求和算法（Neumaier 或 Kahan），以保证大数组场景下的数值稳定性

**整数运算正确性要求**：

| 测试类别 | 要求 |
|----------|------|
| 溢出检测 | `sum`/`prod`/`cumsum`/`cumprod` 作用于整数类型（i8/i16/i32/i64/u8/u16/u32/u64）时，须验证 debug 和 release 模式下溢出均正确 panic。使用 `checked_add`/`checked_mul` 实现（见 §14.3） |
| release 防护 | 确保 release 模式下 checked 算术未退化为 wrapping（Rust release 默认 wrapping，须显式使用 checked 操作） |
| 大整数归约 | 对大数组（≥ 10⁶ 元素）验证 i32/i64 归约不会在未溢出时误 panic |
| 边界值 | 覆盖 `i8::MAX` 累加溢出、`i32::MAX * i32::MAX` 乘法溢出、`u64::MAX` 累加溢出、空数组归约返回单位元 |

**数值精度验证方法论**：

| 属性 | 要求 |
|------|------|
| 参考值来源 | 浮点使用 Rust 标准库 `f32::sin` / `f64::sin` 等作为参考实现；复数使用 `num-complex`（仅测试依赖，非库依赖）或手写参考实现作为对照 |
| 精度度量 | 超越函数使用相对误差 `\|computed - expected\| / \|expected\|`；归约使用混合误差：`\|computed - expected\| ≤ atol + rtol * \|expected\|`，其中 atol = 0，rtol = 精度表中归约行对应值；加减乘除使用 ULP（Unit in the Last Place）验证 |
| 容差 | 不超过本节（§28.3）精度表中规定的阈值 |
| 边界值测试 | 必须覆盖：0、±1、极大值（MAX）、极小正规数（MIN_POSITIVE）、subnormal、±Inf、NaN、π 的倍数（三角函数）、整数幂（pow）。复数额外覆盖：纯虚数（0+1i）、零虚部、零实部、极大模 |
| 回归测试 | 使用已知正确结果的测试向量（golden test），固定随机种子确保可重复性 |
| 跨平台一致性 | 同一输入在不同平台（x86_64 vs aarch64）上的结果须在精度容差内一致（SIMD 近似可能引入跨平台差异，须在容差范围内） |

### 28.4 边界覆盖

须覆盖以下边界情况：
- 空张量
- 单元素
- 大张量
- 极端值（NaN/Inf/subnormal）
- 非连续布局
- 高维（≥4维）
- 广播形状不兼容（返回错误）
- ArcRepr 写时复制（引用计数 1/2/多时的行为差异）
- 小数组对齐降级（ALIGNED 标志正确性）
- 填充布局操作（reshape/slice/transpose 后的步长正确性）
- IxDyn 与 IxN 互转（维度数匹配/不匹配）
- no_std 构建验证（`cargo build --no-default-features` 通过）

### 28.5 并行与 SIMD 测试策略

| 测试类别 | 要求 | 说明 |
|----------|------|------|
| ThreadSanitizer | CI 中使用 nightly 工具链运行 `cargo +nightly test --all-features --target x86_64-unknown-linux-gnu -Zsanitizer=thread` | 检测数据竞争。`-Zsanitizer` 需要 nightly Rust（见 §28.6 CI 要求中的 nightly 行） |
| Miri | CI 中运行 `cargo +nightly miri test --no-default-features --features std --target x86_64-unknown-linux-gnu`（禁用 `simd` 和 `parallel` feature，因 Miri 不支持 SIMD intrinsics 和 rayon 线程） | 检测未定义行为、越界访问。注意：Miri 性能较慢，建议仅对核心模块（存储、索引、广播、ArcRepr）运行 |
| 跨平台 SIMD | 相同输入在 x86_64（AVX2）和 aarch64（NEON）上的逐元素运算结果须一致（精度容差内） | 检测指令集实现差异 |
| 标量 vs SIMD 对照 | SIMD 路径结果须与标量路径结果在精度容差内一致（每个超越函数均须验证） | 确保 SIMD 优化不引入精度回归 |
| 并行确定性 | 相同输入的并行归约结果须与单线程结果一致（浮点归约允许精度容差内差异，见 §28.3 归约行说明） | 确保分块策略正确性 |
| 并行边界 | 测试元素数略高于/低于并行阈值（§9.2.2，默认 64K）的边界情况 | 确保阈值切换无遗漏 |

### 28.6 CI 要求

| 要求 | 说明 |
|------|------|
| 平台矩阵 | Linux (x86_64, aarch64)、macOS (x86_64, aarch64)、Windows (x86_64) |
| Rust 版本矩阵 | 当前 stable + MSRV 版本 + nightly（用于 sanitizer/Miri 测试，见 §28.5） |
| Feature 矩阵 | 默认 features、`--no-default-features`、`--all-features` |
| 必须通过 | `cargo test`、`cargo test --all-features`、`cargo doc`（无 broken doc link） |
| MSRV 验证 | 使用 `cargo msrv` 或 CI 脚本验证 MSRV 声明准确 |
| 格式检查 | `cargo fmt --check` |
| Lint | `cargo clippy --all-features -- -D warnings` |
| 性能回归 | 提供 `cargo bench` 基准，CI 中记录历史结果，人工审查异常波动（暂不自动门控） |

---

*Xenon v20 — 需求说明书*
