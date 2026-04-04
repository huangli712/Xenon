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
| 科学计算导向 | 列优先（F-order）默认、SIMD 友好、BLAS 兼容内存布局 |
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
- 向量内积
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

**InlineArray<T, N> 说明**：库内自研小型内联数组类型（`pub`），无外部依赖。行为：N 个元素以内在栈上固定大小数组存储，超过 N 时自动退化为堆分配（`Vec<T>`）。用于 IxDyn 内部存储、广播形状推导（§16.1）、错误类型中的 ShapeData（§27）等需要动态长度但通常维度较低的场景。提供 `Deref<Target=[T]>`、`FromIterator`、`with_capacity`、`push` 等基本操作。当 N=8 时，覆盖 99.9% 科学计算场景（6 维以内零堆分配）。**可见性理由**：`InlineArray` 作为 `IxDyn::Pattern`（见 §3.2 关联类型）和 `TensorBase<S, IxDyn>` 的 `Index` trait 索引类型出现在公共 API 中，下游 crate 需要命名该类型（如 `let idx: InlineArray<usize, 8> = ...; arr[idx]`），因此必须为 `pub`。`InlineArray` 的 API 表面最小化（`Deref`、`FromIterator`、`with_capacity`、`push`、`From<&[T]>`/`From<Vec<T>>`），不构成维护负担。

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
| `size()` | `&self -> usize` | 返回总元素数（各轴乘积）。Ix0（空维度）返回 1（空乘积的数学约定）。**溢出时 wrapping**（debug 与 release 一致）。此方法仅用于非分配场景（如迭代计数、布局判断），分配场景须使用 `size_checked()`（见下方分配安全规则）。**开发期安全网**：debug 构建中须包含 `debug_assert` 检测溢出——当 `size_checked()` 返回 `None` 时触发断言失败，提示开发者应使用 `size_checked()`。release 构建中此断言不生效，保持 wrapping 行为的性能。实现示例：`fn size(&self) -> usize { let checked = self.size_checked(); debug_assert!(checked.is_some(), "Dimension::size() overflow — use size_checked() for allocation paths"); checked.unwrap_or_else(|| self.size_wrapping()) }` |
| `size_checked()` | `&self -> Option<usize>` | 返回总元素数，溢出时返回 `None`。Ix0 返回 `Some(1)` |
| `size_wrapping()` | `&self -> usize` | 返回总元素数，溢出时 wrapping。仅在明确不需要用于内存分配的场景使用（如统计日志） |

**分配安全规则**（强制性约束）：所有涉及内存分配的路径（包括但不限于 Owned 构造、ArcRepr::make_mut 深拷贝、reshape/transpose 需要新缓冲区时）**必须**使用 `size_checked()` 而非 `size()`。当 `size_checked()` 返回 `None` 时，须返回 `ShapeError::Overflow` 错误（见 §27），不得 panic 或静默继续。`size()` 方法仅用于非分配场景（如迭代计数、布局判断），其 wrapping 行为在这些场景下不影响内存安全。

| `insert_axis()` | `&self, axis: usize -> Self::Larger` | 在指定位置插入长度为 1 的新轴，返回升维后的维度。axis 须在 `[0, self.ndim()]` 范围内，否则 panic。用于 `unsqueeze`、`stack` 等升维操作。IxDyn 实现使用 InlineArray 的 insert 操作（ndim ≤ 8 时栈拷贝，ndim > 8 时可能涉及堆分配） |
| `default_strides()` | `&self -> Self` | 计算 F-order 默认步长（列优先），以 `usize` 存储于 `Self` 内部 |
| `into_dyn()` | `self -> IxDyn` | 将任意维度类型转换为动态维度 `IxDyn`，消费 self。静态维度（Ix0~Ix6）将内部数组数据拷贝到 `InlineArray<usize, 8>`（ndim ≤ 6 时为栈拷贝，无堆分配）；`IxDyn` 为零开销 move（直接返回 self，无 clone）。签名使用 `self` 而非 `&self`，使 IxDyn→IxDyn 路径无分配。此为静态→动态维度转换的唯一入口，用于需要运行时维度类型的场景（如动态 unsqueeze、跨维度类型操作）。转换总是成功（infallible）。反向转换（动态→静态）使用标准库 `TryFrom<IxDyn>` trait，失败时返回 `DimensionMismatch`（见 §3.5） |

**步长存储设计**：

shape 与 strides 共用同一维度类型 `D`，内部均以 `usize` 存储。步长在运算时按 `isize` 解释——因 `usize` 与 `isize` 内存布局等价（同为 pointer-sized integer、二进制补码，所有 Rust 支持的平台上 `size_of` 和 `align_of` 相等），`strides()` 返回 `&[isize]` 时通过显式转换函数 `stride_to_offset(n: usize, stride: usize) -> isize` 将 stride 解释为有符号偏移，无逐元素转换开销。负步长存储为 `usize` 的二进制补码表示（如 `-3isize` 存储为 `usize::MAX - 1`），按 `isize` 解释时自动恢复负值。`Debug` 实现须将 strides 以 `isize` 形式输出（而非原始 `usize`），以确保调试可读性。实现须包含以下编译时断言，在所有目标平台上验证此等价假设：

```rust
const _: () = assert!(size_of::<usize>() == size_of::<isize>());
const _: () = assert!(align_of::<usize>() == align_of::<isize>());
```

步长切片访问（`&[isize]`）由 `TensorBase` 的 `strides()` 方法提供（见 §8.3），而非在 `Dimension` trait 层面定义——因为步长的有符号解释仅在张量上下文中有意义。`TensorBase` 同时提供 `stride_at(axis: usize) -> isize` 方法（见 §8.3），不依赖切片重解释，为单个轴步长访问的推荐方式。

**步长值域约束**：所有步长的绝对值须 ≤ `isize::MAX as usize`。此约束保证：1）usize→isize 显式转换后符号正确；2）任何以 usize 比较步长绝对值的代码不会因负步长的补码表示而产生错误判断。`TensorBase` 层面提供 `stride_at(&self, axis: usize) -> isize` 方法（见 §8.3），封装显式转换逻辑，避免散落的类型转换调用。注意：此方法名统一为 `stride_at`，取代此前设计中的 `stride_isize` 命名。

**关联类型映射**：

| 维度类型 D | D::Pattern | D::Larger |
|------------|------------|-----------|
| Ix0 | `()` | Ix1 |
| Ix1 | `usize` | Ix2 |
| Ix2 | `(usize, usize)` | Ix3 |
| Ix3 | `(usize, usize, usize)` | Ix4 |
| Ix4~Ix6 | `(usize, ..., usize)` | Ix(N+1)（Ix6 的 Larger 为 IxDyn） |
| IxDyn | `InlineArray<usize, 8>` | IxDyn |

**IxDyn 索引导捷性设计**：`IxDyn::Pattern = InlineArray<usize, 8>` 作为 `Index` trait 的模式类型，用于 `indexed_iter` 等内部场景。为提升动态维度数组的用户交互体验，须为 `TensorBase<S, IxDyn>` 额外实现以下 `Index` 变体（利用 Rust 允许为不同索引类型实现多个 `Index` trait 的特性）：

| 索引类型 | 实现方式 | 示例 |
|----------|----------|------|
| `&[usize]` | `Index<&[usize]> for TensorBase<S, IxDyn>` | `arr.index(&[1, 2, 3])` |
| `[usize; N]` | `Index<[usize; N]> for TensorBase<S, IxDyn>`（const generic） | `arr[[1, 2, 3]]` |
| `Vec<usize>` | `Index<Vec<usize>> for TensorBase<S, IxDyn>` | `arr.index(vec![1, 2, 3])` |

这些变体内部将索引转换为 `InlineArray<usize, 8>` 后委托到 `Index<InlineArray<usize, 8>>` 实现，无额外性能开销。**设计理由**：`InlineArray` 不在 Rust 标准库中，无法通过字面量或常见类型自然构造；提供切片/数组/Vec 索引变体使动态维度数组的索引操作与 `ndarray` crate 的 `arr.index(&[...])` 惯例一致。

### 3.3 RemoveAxis trait

`RemoveAxis` 是独立的 trait，仅由可降维的维度类型实现。**Ix0 不实现此 trait**，从编译时保证零维张量不会调用 `remove_axis`。`type Smaller` 定义在此 trait 而非 `Dimension`，使 Ix0 完全无需定义无意义的降维类型。

| 关联类型/方法 | 签名 | 说明 |
|--------------|------|------|
| `type Smaller` | 关联类型：`Dimension` | 移除一个轴后的维度类型 |
| `remove_axis()` | `&self, axis: usize -> Self::Smaller` | 移除指定轴，返回降维后的维度。IxDyn 实现使用 InlineArray 的 remove 操作（ndim ≤ 8 时栈拷贝，ndim > 8 时可能涉及堆分配） |

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
| 静态 → 动态 | `dim.into_dyn()` | `Dimension` trait 方法（见 §3.2） | Infallible，消费 self，返回 `IxDyn`。静态维度拷贝数组数据到 `InlineArray<usize, 8>`（ndim ≤ 6 时栈拷贝，无堆分配），`IxDyn` 为零开销 move |
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
| shape 含零维度（如 `[3, 0, 2]`） | 含零维度的数组标记为 F-contiguous；对应轴步长值不影响实际寻址（因为该轴 size=0 永远不会被实际索引） |

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
Element trait 为 sealed（防止外部实现），但提供 `unsafe impl Element` 的 escape hatch 供高级用户为自定义类型实现。**安全契约**（违反导致未定义行为）：(1) 类型的内存布局必须为 `#[repr(C)]` 或 `#[repr(transparent)]`，确保 FFI 兼容和按位复制安全；(2) 类型必须满足 `Copy + Clone + Send + Sync`，且 `clone()` 等价于按位复制（`memcpy`）；(3) `Default` 值必须为全零字节（与 `zeroed()` 分配兼容）；(4) 所有字节模式必须为合法值（无无效位模式，即 `from_bytes` 安全）；(5) **类型大小须为 2 的幂**（1/2/4/8/16/...字节）——非 2 幂大小（如 12 字节结构体）因无法满足 SIMD padding 的对齐要求（§7.6.2 要求 alignment 须同时为 `size_of::<A>()` 的整数倍和 2 的幂），导致 SIMD padding 不可用，只能使用自然对齐（性能降级但不影响正确性）。**性能降级预期**：非 2 的幂大小类型（如 12 字节结构体、24 字节复合类型）在使用 `unsafe impl Element` 时：(a) SIMD padding 不可用（§7.6），逐元素运算回退到标量路径，预期性能损失约 4-8x（对比 f32/f64 等标准类型的 SIMD 路径，以向量宽度 256-bit/512-bit 估算）；(b) 内存对齐退化为自然对齐（`align_of::<A>()`），无法使用 64 字节缓存行对齐，对缓存不友好的访问模式额外约 10-30% 降级；(c) 总体预期：相比标准数值类型，非 2 幂类型在大规模逐元素运算中性能约降低 5-10x。此降级仅影响计算吞吐量，不影响正确性——所有运算结果与标量路径等价。`unsafe impl Element` 通过 `unsafe trait private::Sealed` 实现——用户须同时 `unsafe impl private::Sealed`，明确承担上述安全契约的责任。详见§4.4。

| 设计决策 | 说明 |
|----------|------|
| sealed 范围 | Element、Numeric、RealScalar、ComplexScalar 全部 sealed |
| 实现方式 | 私有模块 `mod private { pub trait Sealed {} }` + 各 trait 继承 `private::Sealed` |
| sealed 实现者 | 仅库内预定义类型：i32/i64/f32/f64/bool/usize/Complex<f32>/Complex<f64>。其中 `usize` 仅实现 `Element`（用于集合操作等需要索引类型的场景），不实现 `Numeric` |

**v1 整数类型限制说明**：当前版本不支持 u8/u16/u32/u64/i8/i16 等整数类型。主要原因：(1) 图像处理（u8）、信号处理（i8/i16）、位运算（u64）等场景需额外设计整数运算语义（如整数除零 panic vs wrapping、整数溢出策略）；(2) 整数类型需要 `Numeric` trait 的 `zero()`/`one()` 但语义与浮点有微妙差异（如整数除法截断行为）。后续版本计划扩展支持，届时需明确整数除零行为（当前 `Numeric` 要求 `Div<Output=Self>`，整数除零会 panic，此为接受的行为）。

**扩展性说明**：当前版本（v1）四层 trait 全部 sealed，下游 crate 不可为自定义类型（如定点数、对偶数、有理数等）实现。此限制确保数值语义的完整性和 trait 约束的一致性。未来版本计划提供以下扩展机制之一（具体方案待定）：(1) `Newtype` wrapper 允许包装已有数值类型；(2) `unsafe` 扩展 trait（类似 `unsafe trait TrustedLen`）允许高级用户承担正确性责任；(3) 公开 `Sealed` trait 配合文档化的一致性要求。
bool 的 sum() 返回 usize（count of true 值），等价于 count_true()。usize 仅实现 `Element`（不实现 `Numeric`），但可作为 `sum()` 等归约操作的返回类型。

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

**泛型约束说明**：`Complex<T>` 实现 `Numeric` 要求 `T: RealScalar`（而非仅 `T: Element`），因为复数除法（Smith 方法，见 §5.4）需要 `T::abs()` 计算中间结果，此方法仅在 `RealScalar` 层提供。因此 `Complex<T>: Numeric` 的完整约束链为 `T: RealScalar → Complex<T>: Numeric`。当前版本因 sealed trait 限制（Complex 仅用于 f32/f64），此约束不产生用户可见影响，但须在文档中明确标注，避免未来扩展 escape hatch 时产生困惑。

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
| 归约（sum）含 NaN | 结果为 NaN |
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
| no_std 注意 | `norm()` 依赖 `hypot`（避免 `re²+im²` 溢出），`arg()` 依赖 `atan2`，`exp()`/`ln()`/`sqrt()`/`from_polar()` 依赖对应浮点数学函数。`no_std` 环境下这些函数需 `libm`（外部 C 库）或纯 Rust 实现的数学库（如 `libm` crate 的 Rust 实现版本，无 C 依赖，适合裸机环境）。**缓解策略**：(1) `norm_sqr()`（`re²+im²`，无 `sqrt`/`hypot`）为 `no_std` 友好替代，仅需 `Mul + Add`；(2) 可通过 feature gate 在 `no_std` 环境下提供受限的 `Complex` 子集（排除需要数学函数的方法），但当前版本不实现此分级；(3) 若目标平台无浮点单元（如某些嵌入式 MCU），`Complex` 的所有运算均不可用（依赖硬件浮点），须在文档中明确标注 |
| Feature gate 分级（v2 路线图） | **复杂度数学方法分级方案**：将依赖外部数学函数的方法按 feature gate 分层，使 `no_std` 环境可按需启用：(1) **基础层**（`Complex<T>` 本身，无 feature gate）：构造、四则运算、共轭、`norm_sqr()`、`to_polar`/`from_polar`（仅赋值，不调用 `atan2`/`hypot`/`sin`/`cos`）——仅需 `Mul + Add`，`no_std` 零依赖可用；(2) **libm 层**（feature gate `complex-libm`，默认启用）：启用 `norm()`/`arg()`/`exp()`/`ln()`/`sqrt()`/`from_polar()`（计算式）——需 `libm` crate 作为可选依赖（`[dependencies.libm] optional = true`）。`no_std` 用户可通过禁用默认 feature 并仅启用 `complex-libm` 来精确控制。**实现要求**：`Cargo.toml` 须声明 `libm` 为可选依赖；`Complex` impl 块使用 `#[cfg(feature = "complex-libm")]` 条件编译；`no_std` + 不启用 `complex-libm` 时，相关方法不存在（编译期报错而非运行时 panic）。**迁移兼容性**：默认 feature 包含 `complex-libm`，现有 `std` 用户无感迁移；仅 `no_std` + `default-features = false` 用户受影响 |

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

- **分配**：使用 `Layout::from_size_align(capacity * size_of::<A>(), alignment)` + `alloc::alloc()`。`alignment` 至少为 `max(64, align_of::<A>())`。**空分配守卫**：当 `capacity == 0` 或 `size_of::<A>() == 0` 时，`ptr` 设为 `NonNull::dangling()`，不调用 `alloc::alloc()`（零大小分配和空分配均为未定义行为）。对 ZST（`size_of::<A>() == 0`），`capacity` 设为元素数（逻辑值）。对非 ZST 的空数组（如 `shape = [0]` 的 `Tensor<f64, Ix1>`，`capacity = 0`），`ptr` 同样设为 `NonNull::dangling()`。**Drop 守卫**：当 `capacity == 0` 或 `size_of::<A>() == 0` 时跳过 `dealloc`（仅释放过分配的缓冲区）。此路径不影响正常类型（`f32`/`f64`/`i32`/`i64`/`Complex` 均为非零大小），空数组是科学计算中常见场景（过滤结果、形状变换）
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

**CowRepr<'a, A>** — Copy-on-Write 存储表示。**当前版本**：仅在 `ensure_blas_compatible()` 返回值中使用（见 §25.4.5），不实现 Storage/StorageMut trait。**v2+ 扩展**：实现完整 Storage/StorageMut trait，支持运算符重载，可持有 owned 数据或 borrowed 视图，首次修改时按需克隆。适用于需要统一接受 owned/borrowed 输入的 API。CowRepr 的开销比 ArcRepr 更小（无 Arc 引用计数），适合单线程场景。v2 实现时须补充：§6.2 存储模式表条目、Storage/StorageMut trait 实现、类型别名（CowTensor）、Send/Sync 语义。

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
| 布局顺序 | 仅支持 F-order（列优先），F-order 为项目全局默认 |
| 步长类型 | 有符号类型，单位为元素个数（非字节），支持负步长 |
| 默认对齐 | 自有存储默认 64 字节对齐（AVX-512 缓存行） |
| 可配置对齐 | 构造时可指定对齐值（须为 2 的幂，≥ 元素自然对齐） |
| 填充轴 | F-order 填充 axis 0（第一轴）。填充后步长（strides）反映实际物理内存布局（含填充量），详见 §7.6 |
| 填充量 | `M_padded = ⌈M × size_of::<A>() / alignment⌉`（向上取整到 alignment 字节边界），默认 alignment = 64 字节（AVX-512 缓存行），确保每列起始地址对齐 |
| 填充初始值 | 填充区域零初始化。Padding bytes 必须在分配时零初始化 |

### 7.2 Order 枚举

内存布局顺序枚举，用于构造函数和操作中指定或查询布局方向。

| 属性 | 定义 |
|------|------|
| 类型 | `#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)] #[non_exhaustive] pub enum Order` |
| `Order::F` | 列优先（Fortran）顺序 |
| 默认值 | 项目全局默认 F-order（§7.1）。各 API 中 `order: Option<Order>` 参数传 `None` 时使用默认 F-order |
| `#[non_exhaustive]` | 未来可扩展新变体不构成 breaking change |

**使用场景**：

| 场景 | API 示例 | 说明 |
|------|----------|------|
| 构造函数指定布局 | `zeros(shape, order: Option<Order>)` | `None` 时使用默认 F-order |
| 迭代顺序 | `iter_ordered(order: Order)` | 强制按指定逻辑顺序遍历 |
| 拼接/堆叠输出布局 | `cat(axis, &[...], order: Option<Order>)` | `None` 时默认 F-contiguous |

### 7.3 布局标志系统

采用 7 标志位设计（2 字节 u16，低 7 位有效，高 9 位保留，须为 0），缓存高频查询的派生状态：

| 标志 | 位 | 信息 | 理由 |
|------|----|------|------|
| F_CONTIGUOUS | bit 0 | F-order 连续性 | 核心属性，高频查询 |
| ALIGNED | bit 2 | SIMD 对齐（64 字节） | 缓存对齐状态，用于**快速排除**不可能对齐的路径。**当前推荐策略为直接使用非对齐加载**（见 §9.1.4 对齐加载判定），ALIGNED 标志仅用于快速排除不可能对齐的场景（ALIGNED=false 时确定不对齐，ALIGNED=true 时仅作为提示——运行时地址检查已不再推荐，现代 CPU 非对齐加载性能接近对齐加载） |
| HAS_ZERO_STRIDE | bit 3 | 零步长（缓存） | 存在步长为 0 的维度。两种来源：(1) 广播产生的 size-1 维度（逻辑上重复该元素）；(2) unsqueeze 插入的 size-1 轴（§17.8，步长设为 0 以保持连续性）。优化路径据此判断不可按简单步长递增遍历，但连续性标志不受影响。**空数组（任意 axis 的 shape[k]==0）时该标志为 false（无元素被访问，无实际步长语义），且 F_CONTIGUOUS 为 true（与 §3.6 一致）** |
| HAS_NEG_STRIDE | bit 4 | 负步长（缓存） | 反转检测，避免 O(ndim) 遍历 |
| PADDED | bit 5 | 主维度填充 | 数组的主维度（axis 0）经过填充以对齐 SIMD 访问。PADDED 为 true 时，步长反映填充后的物理布局（strides[1] ≥ shape[0]），但严格连续性标志（F_CONTIGUOUS）仍按无填充的 shape 乘积判定。SIMD 路径应使用宽松连续性检查（`is_f_padded_contiguous()`，见 §7.4） |
| OWNS_DATA | bit 6 | 数据所有权 | **仅对 Owned 存储模式为 true**，标记此张量独占拥有其底层数据（而非视图/引用/共享）。ArcRepr 不设置此标志（共享所有权，`manages_memory()` 返回 true 但 OWNS_DATA 为 false）。用于快速判断是否可就地修改且无其他引用，避免遍历存储类型层级 |
| F_PADDED_CONTIGUOUS | bit 7 | F-order 宽松连续 | F-order 宽松连续（允许主维度填充），由 `is_f_padded_contiguous()` 查询。仅在 PADDED=true 时可能为 true；PADDED=false 时与 F_CONTIGUOUS 一致。用于 SIMD 路径快速判断，避免 O(ndim) 遍历 |
| — | bit 8-15 | 保留 | 须为 0，留作未来扩展 |

**组合语义：**
- `F_CONTIGUOUS`：严格 F-contiguous（步长严格等于 shape 乘积）
- `!F_CONTIGUOUS`：非 F-contiguous 数组（切片/转置后常见）
- `PADDED && !F_CONTIGUOUS`：填充数组（严格连续性为 false，但 SIMD 可通过 F-order 宽松连续性路径处理，见 §7.4）

**更新时机：**
- 创建时：根据分配方式初始化全部标志
- 切片/转置/reshape：重新计算全部标志
- 视图创建：继承源数组标志，按需降级（如对齐）

### 7.4 布局查询方法

以下方法定义于 TensorBase（完整方法签名见 §8.3 布局查询），此处列出各方法的判定规则与复杂度：

| 方法 | 说明 | 复杂度 |
|------|------|--------|
| is_f_contiguous() | 是否 F-order 严格连续（步长严格等于 shape 乘积，不含填充） | O(1) |
| is_contiguous() | 等价 is_f_contiguous()（仅支持 F-order） | O(1) |
| is_f_padded_contiguous() | 是否 F-order 宽松连续（允许主维度填充）。判定规则：ndim≤1 时等价 `is_f_contiguous()`；ndim≥2 时要求 |strides[0]|==1、|strides[1]|≥shape[0]、|strides[i]|==|strides[i-1]|×max(shape[i-1], 1)（i≥2）。**注意**：当 shape[i-1]==1 时，条件退化为 |strides[i]|==|strides[i-1]|（即步长须保持一致，不允许任意值），防止 size-1 维度引入非连续跳跃导致 SIMD 路径误判 | O(1) |
| is_padded_contiguous() | 等价 is_f_padded_contiguous()（仅支持 F-order） | O(1) |
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
| 逐元素运算（+, -, *, /） | 不保持 | 输出使用标准无填充分配，PADDED = false。**性能影响**：若输入为填充数组（SIMD 优化路径），逐元素运算后输出丢失填充，后续 SIMD 操作回退到非填充路径（无法使用 `is_f_padded_contiguous()` 快速路径）。在连续链式运算中（如 `a_padded + b_padded -> c -> c * d -> e`），每一步均丢失填充，形成性能降级链。**缓解方式**：(1) 在运算链结束后调用 `.with_padding(64)` 重新填充（代价：一次完整拷贝 + 重新分配）；(2) 对性能关键的内层循环，考虑使用单次融合运算而非链式调用；(3) **v2 路线图**：提供 `_padded` 后缀的运算变体（如 `add_padded()`、`mul_padded()`），输出保持输入的填充布局，避免中间结果的填充丢失 |
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
| 逐轴连续 | `is_axis_contiguous(axis)` | 沿指定轴 SIMD 处理（\|strides[axis]\|==1 即可） |

---

**Padded 数组与 reshape 交互**：Padded 数组的 `is_f_contiguous()` 返回 `false`（仅 `is_f_padded_contiguous()` 返回 `true`）。因此 reshape 操作对 padded 数组默认触发数据拷贝。这是有意的设计选择：padded 数组的主维度步长包含 padding 字节，零拷贝 reshape 需要重新计算 strides 以保持 padding 对齐，复杂度较高且容易出错。若需要在 padded 数组上执行零拷贝 reshape，需先通过 `unpad()` 去除 padding（此操作会拷贝数据），再执行 reshape。


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
| flags: u16 | 布局标志位（低 9 位有效，高 7 位保留为 0）。缓存高频查询的派生状态（F 连续性、对齐、零步长、负步长、填充），使布局查询方法达到 O(1) 复杂度（见 §7.3 标志定义与 §7.4 查询方法）。创建时根据分配方式初始化，切片/转置/reshape 时重新计算，视图创建时继承并按需降级 |

**指针语义分层**：TensorBase 中的指针存在两级语义，须严格区分：

| 层级 | 来源 | 含义 | 用途 |
|------|------|------|------|
| 缓冲区起始指针 | `storage.as_ptr()`（§6.1 Storage trait） | 分配的内存缓冲区起始地址，不含 offset | 内部实现：地址计算的基准点 |
| 数据起始指针 | `as_ptr()`（TensorBase 方法，见 §8.3 及 §25.1） | `storage.as_ptr() + offset`，即首个逻辑元素的地址 | 对外 API：用户和 FFI 通过此指针访问数据 |

元素地址计算公式：`as_ptr() + Σ(index[k] * stride[k])`，展开为 `storage.as_ptr() + offset + Σ(index[k] * stride[k])`。`as_mut_ptr()` 同理，返回 `storage.as_mut_ptr() + offset`。

**偏移量算术安全性**：元素地址计算 `offset + Σ(index[k] * stride[k])` 须始终落在 `[0, storage_capacity)` 范围内。实现须保证：

(1) **shape 不变量**：每个轴 `shape[k] ≤ isize::MAX as usize`（即 `usize::MAX / 2`），由 `TensorBase` 构造函数在入口处验证，违反时返回 `ShapeError::DimensionOverflow`（见 §27）。

(2) **stride 不变量**：每个轴的步长绝对值 `|stride[k]|` 不得超过 `isize::MAX`。构造函数须验证默认 F-order 步长的累积乘积不超出 `isize::MAX`——即对于所有 k，`product(shape[0..k]) ≤ isize::MAX`（F-order）。违反时返回 `ShapeError::Overflow`。注意：仅约束单个 axis ≤ isize::MAX 不足以保证此不变量，因为 stride 是多轴累积乘积。

(3) **地址计算中间精度**：元素地址公式 `offset + Σ(index[k] * stride[k])` 中，单项 `index[k] * stride[k]` 的乘积在 64-bit 平台上可能超出 `isize` 范围（即使满足上述两个不变量，当 ndim ≥ 2 时累积乘积仍可能溢出）。**显式不变量**：合法 tensor 的总分配字节数 ≤ `isize::MAX`（由 §7.6.2 填充量构造保证——`padded_bytes` 验证不超过 `isize::MAX`），因此所有有效偏移量均在 `[0, isize::MAX)` 范围内，`isize` 中间精度不会溢出。**运行时热路径**（元素寻址）须使用 `isize` 中间精度进行乘法与累加，最终结果转为 `usize` 后验证 < 存储容量。**构造函数冷路径**（constructor-time overflow validation）可使用 `i128` 中间精度验证 shape×stride 不超出 `isize::MAX` 范围，避免构造函数中手动模拟 128-bit 比较的复杂度。构造函数验证仅在创建 tensor 时执行一次，不影响运行时性能。

(4) **边界访问验证**：最终偏移量 `offset + Σ(...)` 转为 `usize` 后须 < 存储容量，越界访问在 safe API 中返回错误或 panic（如 `IndexOutOfBounds`，见 §27），在 `unsafe` API（如 `get_unchecked`）中为调用方责任。

(5) 构造函数（包括 `from_raw_parts`）须验证：(a) shape × strides 不产生越界偏移；(b) `shape.size_checked()` 返回 `Some`（即总元素数不溢出 usize），违反时返回 `ShapeError::Overflow`。

**统一步长验证函数**：所有构造路径必须调用统一的 `validate_tensor_layout(shape, strides, offset, capacity) -> Result<(), InvalidLayout>` 函数，验证：(1) stride >= 0 或符合负步长规则，(2) 单轴 shape×stride 不超过 `isize::MAX`，(3) 零步长仅用于 size-1 或 size-0 维度，(4) `offset < capacity`（起始偏移在缓冲区内），(5a) `offset + max_access_offset ≤ capacity`（上界——所有有效索引的最大偏移不越界，`max_access_offset` 由遍历各轴的 `(shape[k] - 1) * |stride[k]|` 累积计算，空轴贡献为 0），(5b) `offset >= min_access_offset`（下界——所有有效索引的最小偏移不越界，`min_access_offset = Σ(负步长轴: |stride[k]| × (shape[k] - 1))`，即 `offset` 须至少覆盖负步长方向的最大回退距离；若无非负步长轴，`min_access_offset = 0`）。**设计理由**：仅检查上界 (5a) 不充分——当存在负步长时，索引 `idx[k] = shape[k] - 1` 在负步长轴上产生负偏移贡献，可能使总偏移量下溢到负值（未定义行为）。下界检查 (5b) 确保即使所有轴均为负步长，`offset` 也足够大以使最小偏移 ≥ 0。此函数为唯一权威的步长与偏移合法性检查点，确保所有构造路径（包括 `from_raw_parts`、视图构造等）的验证逻辑一致。对于视图构造（slice/reshape 等），`capacity` 为源存储的容量；对于新建 tensor，`capacity` 为新分配的缓冲区容量。**溢出安全要求**：`max_access_offset` 和 `min_access_offset` 的计算（`Σ(|stride[k]| × (shape[k] - 1))`）须使用**检查算术**（`checked_add` + `checked_mul`），溢出时返回 `Err(InvalidLayout)` 而非 panic 或回绕。理由：(1) 当 `shape[k]` 接近 `usize::MAX` 且 `|stride[k]| > 1` 时，`|stride[k]| × (shape[k] - 1)` 的乘积可能溢出 `usize`；(2) 高维度（ndim ≥ 3）的累加和可能溢出 `usize`；(3) 回绕会导致通过验证但实际越界访问（安全漏洞）。实现须在每个轴的乘法前 `checked_mul`，轴间累加前 `checked_add`，任一步溢出即返回 `Err(InvalidLayout { reason: "stride/shape overflow in layout validation" })`。

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
| `strides()` | `&self -> &[isize]` | 返回各轴步长的切片（元素单位，有符号）。内部以 `D` 类型存储（`usize`），返回时转为 `&[isize]` 切片。**实现注记**：通过 `unsafe` 指针转换将 `&[usize]` 重解释为 `&[isize]`（`slice::from_raw_parts(self.strides_internal.as_ptr() as *const isize, len)`），安全性依赖 `size_of::<usize>() == size_of::<isize>()` 且 `align_of::<usize>() == align_of::<isize>()`（编译时静态断言保证，见 §3.2）。无逐元素转换开销（见 §3.2 步长存储说明）。**平台假设说明**：此重解释依赖 `usize` 与 `isize` 内存布局等价假设——在所有当前 Rust 支持平台（x86_64、aarch64、riscv64 等）均成立，且编译时断言提供安全网。但此假设并非 Rust 语言规范的显式保证，未来理论上的新平台可能打破。作为不依赖重解释的备选路径，提供 `stride_at()` 方法（见下方）。**Miri 兼容性说明**：`strides()` 的 `&[usize]` → `&[isize]` 指针转换属于 `transmute` 等价操作（不同类型的指针重解释），Miri 的 `-Zmiri-tag-raw-pointers` 模式下可能产生假阳性警告。此转换在语义上是安全的（`usize` 和 `isize` 共享所有位模式），但 Miri 的类型跟踪系统无法识别此等价关系。**缓解策略**：(1) 正常运行不受影响，仅 Miri 测试环境可能报错；(2) Miri CI 脚本须使用 `-Zmiri-ignore-tag-raw-pointers` 或对调用 `strides()` 的测试函数标注 `#[cfg_attr(miri, ignore)]`；(3) 内部热路径优先使用 `stride_at()`（无重解释，Miri 友好），`strides()` 主要用于用户级 API 和调试输出 |
| `stride_at()` | `&self, axis: usize -> isize` | 返回指定轴的步长值（有符号）。**不依赖切片重解释**，直接从内部 `usize` 存储读取并转换为 `isize`（`self.strides_internal[axis] as isize`）。此方法为不依赖 `strides()` 重解释语义的安全替代，供未来移植到可能打破 usize/isize 布局等价假设的平台使用。性能与 `strides()[axis]` 等价（单次内存读取 + 类型转换）。**建议**：当仅需访问单个轴步长时优先使用此方法，避免构造临时 `&[isize]` 切片 |
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
| `is_contiguous()` | `&self -> bool` | 等价 `is_f_contiguous()`（仅支持 F-order） |
| `is_f_padded_contiguous()` | `&self -> bool` | 是否 F-order 宽松连续（允许填充）。判定规则详见 §7.4，size-1 内部维度步长须与前一轴步长一致 |
| `is_padded_contiguous()` | `&self -> bool` | 等价 `is_f_padded_contiguous()`（仅支持 F-order） |
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
| WASM SIMD | pulp 已具备 wasm32 条件编译支持（包含 Simd128 和 RelaxedSimd 两个 SIMD 变体）。**当前版本（v1.0）未纳入 Xenon SIMD dispatch 路径**——`Arch::dispatch()` 在 wasm32 目标下直接使用 `Scalar` 路径。计划在 v1.1 版本加入：需验证 WASM SIMD 的浏览器兼容性（Chrome/Firefox/Safari 均已支持 simd128；RelaxedSimd 仅 V8 支持）并补充对应的 CI 测试。启用后 `Arch::Simd128` 将作为 wasm32 的默认非标量路径，`Arch::RelaxedSimd` 为可选优化路径 |
| dispatch 机制 | `Arch::new()` 运行时检测最优指令集，`arch.dispatch(WithSimd)` 通过 `WithSimd` trait + `NullaryFnOnce` bridge 调用闭包。`Arch` 为 `Copy` 类型，可在库初始化时缓存（见 §9.1.4） |
| Simd trait 操作覆盖 | pulp 的 `Simd` trait 提供：浮点/整数算术、复数（c32/c64）、比较、位运算、abs/sqrt/min/max/neg、内存操作（splat/load/store、masked load/store 即 `mask_load_ptr`/`mask_store_ptr` 系列）、mask 操作。**不提供超越函数**（sin/cos/exp/ln），这些操作回退标量路径（见 §9.3 注）。**注**：pulp 不提供显式的 `gather`/`scatter`（index-vector 分散读写）方法，仅提供基于 mask 的 `mask_load_ptr_*`/`mask_store_ptr_*` 系列方法（功能类似但不等价：以 mask 控制逐元素加载/存储，而非通过 index vector 指定地址偏移）。非连续数据的 SIMD 加载须依赖 mask 操作或回退标量路径 |

#### 9.1.3 SIMD 适用操作

**SIMD 路径的通用前提条件**（适用于以下所有 SIMD-eligible 操作）：

| 条件 | 说明 |
|------|------|
| 宽松连续内存 | `is_padded_contiguous()` 为 true（见 §7.4） |
| **正步长方向** | 所有步长（stride）须为**正值**（对 size-1 维度步长可为 0）。负步长数组即使满足 `is_padded_contiguous()` 也**不得进入 SIMD 路径**，须回退标量路径。理由：SIMD 连续加载（load/store）按地址递增方向迭代，负步长数组的物理内存是反向排列的，强制正向遍历会导致越界访问（UB）。参考：ndarray 通过 `is_layout_c`/`is_layout_f` 将 stride 转为 `isize` 后与正值比较，自动排除负步长（负值永远不等于正值） |
| 对齐要求 | 见 §9.1.4 |

| 操作类别 | 示例 | 额外 SIMD 要求 |
|----------|------|---------------|
| 逐元素一元 | abs, neg, sqrt, exp, ln | 无额外要求；满足通用前提即可 |
| 逐元素二元（浮点/复数） | add, sub, mul, div | 两操作数均满足通用前提，形状兼容，**且布局方向一致**（见下方说明）。任一条件不满足时回退标量路径 |
| 逐元素二元（整数） | add, sub, mul | 同上。v1 支持的整数类型为 i32/i64，pulp 均提供 SIMD 乘法支持 |
| 逐元素二元（整数除法） | div（整数类型） | 不走 SIMD 路径，始终标量。理由：主流 SIMD 指令集（AVX2/SSE/NEON）不提供整数除法指令，pulp 亦不提供整数 SIMD 除法；即使连续内存也回退标量路径 |
| 归约（浮点/复数） | sum | 满足通用前提即可；不满足时回退标量路径 |
| 归约（整数 sum） | sum | **默认使用 widening 累加 + SIMD 路径**。具体策略（v1 支持的整数类型）：`i32` 累加到 `i64`（widening 避免溢出）；`i64`/`u64` 因无更大标准整数类型，**回退标量 checked 路径**（每步 `checked_add` 检测溢出，溢出时 panic，与 §14.3 整数归约溢出策略一致）。参考：归约溢出 wrapping 在几乎所有场景下均为 bug，因此 i64/u64 也使用 checked 语义（与 rayon `wrapping_add`/NumPy 不同——Xenon 选择安全性优于一致性） |
| 内积 | dot | 满足通用前提即可 |
| 比较 | eq, lt, gt（生成 mask） | 满足通用前提即可；不满足时回退标量路径 |

**bool 类型归约说明**：bool 类型：`sum()` 返回 `usize`（统计 true 的数量）。推荐使用专用的 `count_true()`/`count_false()` 方法。

**二元 SIMD 的布局方向一致性要求**：

二元 SIMD 操作（如 `a + b`）要求两操作数不仅满足各自的通用前提，还须**布局方向一致**。具体规则：

| 布局组合 | SIMD 行为 | 说明 |
|----------|----------|------|
| 同为 F-padded（列优先宽松连续） | ✅ SIMD 路径 | 两操作数按列优先顺序同步迭代 |
| 任一为非宽松连续 | ❌ 回退标量路径 | 通用前提不满足 |

实现须在二元 SIMD 入口处检查 `layout_direction_compatible(a, b)`（返回 true 当且仅当两操作数同为 F-padded），不满足时回退标量路径。

**注 — 逐轴 SIMD 优化**：上述条件为 SIMD 路径的充分条件。实现可进一步支持逐轴 SIMD（沿目标轴 `is_axis_contiguous(axis)` 为 true 即可，不需要整个数组连续），但当前版本不做强制要求。逐轴 SIMD 允许非连续数组（如转置视图）沿连续轴使用 SIMD 处理。

**注 — 广播操作数的 SIMD 策略**：当二元运算的操作数之一为广播视图（`has_zero_stride() == true`）时，不满足上述宽松连续条件。但广播视图有专用 SIMD 优化路径：单轴广播使用 SIMD broadcast 指令（如 `_mm256_set1_ps`）将标量值加载到所有 lane 后与连续操作数进行向量运算。

#### 9.1.4 SIMD 对齐与回退

| 属性 | 要求 |
|------|------|
| 对齐加载判定 | **推荐直接使用非对齐加载（如 `_mm256_loadu_ps`），不做运行时对齐检查**。理由：(1) 现代 CPU（Haswell 及以后，2013+）上非对齐加载性能已接近对齐加载（L1 命中时 ~1 周期差异，且仅当跨 cache line 时才有惩罚）；(2) 运行时对齐检查（`ptr % alignment == 0`）每次 SIMD 操作多一次取模运算，开销可能超过对齐加载的收益；(3) pulp 的 `Simd` trait 的 `load`/`store` 方法本身已是非对齐语义。**不得依赖 ALIGNED 布局标志**选择加载指令（小数组降级后 ALIGNED=false 但地址可能恰好对齐，反之亦然）。若实现者坚持对齐优化，对齐阈值须与当前 SIMD 宽度匹配（AVX-512 → 64 字节，AVX2 → 32 字节，NEON → 16 字节），可通过 pulp 的 Simd trait 关联类型获取宽度信息 |
| 非连续数据 | 不满足 `is_padded_contiguous()` 时回退到标量路径；逐轴 SIMD 优化下，`is_axis_contiguous(axis)` 不满足时回退标量路径 |
| 负步长数据 | 即使满足 `is_padded_contiguous()`，若任一轴步长为负值，须回退标量路径。负步长表示逻辑方向与物理内存方向相反，SIMD 连续加载/store 无法正确处理反向遍历。实现须在 SIMD 入口处检查 `all_strides_positive_or_zero()`（所有步长 > 0，size-1 维度步长允许为 0） |
| 二元布局不兼容 | 二元操作的两操作数布局方向不一致（非同为 F-padded）时回退标量路径，见 §9.1.3 |
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
| 并行阈值 | 默认数据量（`元素数 × size_of::<A>()`）≥ 256KB 时启用并行，可通过全局配置或单次调用覆盖。基于内存大小而非固定元素数，确保不同元素类型（f32/f64/Complex<f64>等）的调度开销与计算收益比例一致。**配置 API**：(1) 全局配置：`xenon::parallel::set_threshold(size: usize)` 设置全局默认阈值（单位字节），通过 `AtomicUsize` 存储（所有线程共享，包括 rayon 工作线程——确保主线程设置的阈值在并行计算中生效）。(2) 单次调用覆盖：`arr.par_iter().with_threshold(64 * 1024)` 链式 API 指定本次操作的阈值（单位字节），覆盖全局配置。(3) 查询：`xenon::parallel::threshold() -> usize` 返回当前全局阈值。**设计变更说明**：使用 `AtomicUsize`（`Ordering::Relaxed` 读写，因阈值仅为性能调优提示，不涉及内存安全）替代 `thread_local!`，解决 `thread_local!` 存储不跨线程传播的问题——此前主线程调用 `set_threshold()` 后，rayon 工作线程仍使用默认值，导致用户配置被静默忽略 |
| 支持并行的操作 | 逐元素运算、归约、map/mapv、zip 迭代 |
| 不支持并行的操作 | 矩阵乘法（由 BLAS 内部管理线程）、单元素操作、小数组操作 |
| 分块策略 | 按连续内存块分割，每块不小于 4K 元素。**Cache line 对齐**：分割边界须对齐到 64 字节（主流 CPU cache line 大小），避免相邻线程块共享同一 cache line 导致 false sharing。对于 padded 数组，分割须在 padding 边界对齐（即分割点须为逻辑元素边界），确保 SIMD 写入 padding 字节时不与其他线程的 cache line 交叉。**非连续数组分块算法**：按以下优先级选择分块轴和策略：**(1) 连续轴优先**：扫描所有轴找到 `|strides[axis]| == 1` 的轴，优先沿该轴分块（块内元素在物理内存中连续，缓存友好）。分块将轴切分为 `[start..end)` 索引范围，每块元素数 ≥ 4K。**(2) 最外轴回退**：若无连续轴（所有 `|strides[k]| > 1`），沿最外轴（axis 0）分块，每块为逻辑索引范围 `[start..end)` × 其余轴全范围，块内逐元素按 strides 跳跃寻址。**(3) 分块大小约束**：每块元素数 ≥ 4K 且起始边界对齐到 64 字节（若沿连续轴分块，起始索引 × `size_of::<A>()` 须为 64 的倍数）。**(4) 负步长处理**：含负步长的轴的分块按绝对步长计算物理偏移，每块内部正向遍历（从块的逻辑起始到逻辑终止），物理地址可能递减，但块内范围在构造时已验证不越界（见 §8.1 统一步长验证函数） |
| 嵌套并行 | 禁止嵌套并行（内层操作强制单线程），避免线程池饥饿。**强制机制**：使用 `rayon::current_thread_index().is_some()` 检测当前是否在 rayon 工作线程上——rayon 线程池内返回 `Some(thread_index)`，外部（主线程等）返回 `None`。并行操作入口处检查：若已在 rayon 线程池内，内层操作强制走单线程路径（不 spawn 新任务）。**设计变更说明**：原方案使用 `thread_local!` + RAII scope guard 标记并行上下文，但 `thread_local!` 是 per-OS-thread 的——主线程设置的标记对 rayon 工作线程完全不可见（每个线程有独立的 `Cell<bool> = false`），嵌套操作在 rayon worker 上执行时检测必然失败。`rayon::current_thread_index()` 是 rayon 官方提供的 API（用于 `par_bridge.rs` 的递归检测），语义精确且无需手动管理状态，不存在 panic 安全问题（无 RAII guard 需要维护）。实现示例：`if rayon::current_thread_index().is_some() { return scalar_path(); } /* 否则正常走并行路径 */` |
| 线程数 | 默认使用 rayon 全局线程池，可通过自定义线程池覆盖 |
| 浮点归约算法 | 并行浮点归约（sum）须使用 **Kahan-Babuška-Neumaier** 补偿求和算法。具体策略：单线程路径使用 Neumaier 补偿求和（O(1) 额外空间，适合顺序累加）；并行路径各线程先独立执行 Neumaier 累加，最后汇总阶段使用 **Pairwise 递归求和**合并各线程的部分和（保证不同分块策略下结果的精度上界可控）。**不使用朴素 Kahan**（精度不如 Neumaier）。单线程浮点归约同样使用 Neumaier 补偿求和。整数归约使用 widening 累加（见 §9.1.3），不使用补偿求和 |

### 9.3 性能分层

运行时根据数据规模、内存布局和步长方向自动选择执行路径。以下表格按**优先级从高到低**排列（首个匹配的行即为执行路径）：

定义简写：`simd_ok = is_padded_contiguous() && all_strides_positive_or_zero() && simd_enabled && has_simd_impl`

**`has_simd_impl` 谓词定义**：

`has_simd_impl` 表示当前操作是否存在 SIMD 实现。具体判定规则：

| 判定方式 | 说明 |
|----------|------|
| 编译时确定 | 通过 `ElementWiseOp` trait（见 §9.3 实现策略建议）的关联常量 `const HAS_SIMD_IMPL: bool` 声明。每个操作的 trait 实现显式标注是否提供 SIMD 路径 |
| 参考标准 | 下列操作 `has_simd_impl = false`（见下方 "无 SIMD 实现" 表格）：整数除法、超越函数、i64/u64 整数 sum（因无更大标准整数类型，回退标量 checked 路径，见 §14.3）。其余操作 `has_simd_impl = true`（包括 i32 整数 sum，使用 widening SIMD 路径） |
| 新增操作 | 新增操作须在 trait impl 中显式设置 `HAS_SIMD_IMPL`。默认为 `false`（保守策略：新增操作须显式声明支持 SIMD，避免遗漏标量回退） |

| 优先级 | 条件 | 执行路径 |
|--------|------|----------|
| 1 | 元素数 < SIMD 宽度 | 标量路径（元素太少，向量化无收益） |
| 2 | simd_ok 且 元素数 < 并行阈值（或 parallel 未启用） | SIMD 路径（单线程） |
| 3 | simd_ok 且 元素数 ≥ 并行阈值 且 parallel 启用 | SIMD + 并行路径（每线程内部使用 SIMD） |
| 4 | !simd_ok（任何原因：非连续、负步长、simd 未启用、操作无 SIMD 实现） 且 元素数 ≥ 并行阈值 且 parallel 启用 | 标量 + 并行路径（每线程标量迭代。非连续数据按块分割，块内按 strides 跳跃寻址；宽松连续但有负步长的数据同此路径） |
| 5 | !simd_ok 且 元素数 < 并行阈值（或 parallel 未启用） | 标量路径（单线程，步长跳跃遍历） |

**条件依赖关系图**：

```
simd_ok = is_padded_contiguous() && all_strides_positive_or_zero() && simd_enabled && has_simd_impl
    ├─ YES
    │   ├─ element_count >= parallel_threshold && parallel_enabled → 路径 3 (SIMD+并行)
    │   └─ element_count < parallel_threshold || !parallel_enabled → 路径 2 (SIMD)
    └─ NO (任一条件不满足)
        ├─ element_count >= parallel_threshold && parallel_enabled → 路径 4 (标量+并行)
        └─ element_count < parallel_threshold || !parallel_enabled → 路径 5 (标量)
```

**注 — 无 SIMD 实现的操作（即使连续内存也走标量路径）**：

| 操作 | 原因 |
|------|------|
| 整数除法（i32/i64 的 div） | 主流 SIMD 指令集（AVX2/SSE/NEON）不提供整数除法指令，pulp 亦不提供整数 SIMD 除法 |
| 超越函数（sin, cos, tan, exp, ln, log2, log10, pow 等） | pulp 不提供 SIMD 超越函数；当前版本回退标量路径，后续版本可引入独立 SIMD 数学库 |

**注 — i64/u64 整数 sum**：i64/u64 无更大标准整数类型作为 widening 累加器，回退**标量 checked 路径**（每步 `checked_add` 检测溢出，溢出时 panic），无 SIMD 加速（`has_simd_impl = false`）。此策略与 §14.3 整数归约溢出行为一致——归约溢出 wrapping 在几乎所有场景下均为 bug，Xenon 选择安全性优于性能。

上述操作在并行路径下仍可并行（每线程标量迭代），仅不走 SIMD。

**二元操作的 dispatch 扩展规则**：

上述 5 级分层表描述的是单操作数（一元/归约）的 dispatch 逻辑。二元操作（如 `a + b`）须在进入分层表之前执行额外的前置检查，通过后再按输出数组的属性走分层表。具体流程：

```
binary_dispatch(a, b, op):
    // 阶段 1：布局兼容性检查
    if !layout_direction_compatible(a, b):  // 见 §9.1.3
        → 标量路径（两者布局方向不一致）

    // 阶段 2：相同形状快速路径（最常见的二元操作场景）
    if a.shape() == b.shape():
        // 跳过广播步长计算，直接确定输出形状 = a.shape()
        // 立即判定存储复用可行性
        → skip to 阶段 3 with output_shape = a.shape()

    // 阶段 2.5：广播 SIMD 快速路径（一方为广播标量/低维数组）
    // 连续操作数须同时满足 padded_contiguous 和正步长（见 §9.1.3 通用前提条件）
    if a.is_padded_contiguous() && a.all_strides_positive_or_zero() && b.has_zero_stride():
        → 广播 SIMD 路径（broadcast + SIMD）
    if b.is_padded_contiguous() && b.all_strides_positive_or_zero() && a.has_zero_stride():
        → 广播 SIMD 路径（broadcast + SIMD）

    // 阶段 3：存储复用判定
    // 复用条件：(a) 某操作数拥有存储所有权（Owned），(b) 输出对齐/容量满足要求
    // 复用优先级：优先复用 a 的存储（左侧操作数），次选 b
    // 不满足复用条件时分配新的输出存储
    output = select_output_storage(a, b)

    // 阶段 4：按输出属性走 5 级分层表
    // 输出数组的 contiguous/stride/size 属性决定最终执行路径
    → fallback to tier_table(output)
```

| 检查阶段 | 条件 | 执行路径 | 对应章节 |
|----------|------|----------|----------|
| 1. 布局兼容性 | 两操作数布局方向不一致 | 标量路径 | §9.1.3 |
| 2. 相同形状 | `a.shape() == b.shape()` | 跳过广播计算，直接进入存储复用 | 本节 |
| 2.5. 广播 SIMD | 一方 padded contiguous + 正步长 且另一方 zero-stride | 广播 SIMD 路径 | 本节 |
| 3. 存储复用 | 输出可复用某操作数存储（所有权/对齐/容量满足） | 就地写入 | 本节 |
| 4. 分层回退 | 按输出数组属性匹配 5 级分层表 | 对应层级路径 | §9.3 主表 |

**`select_output_storage` 决策表**：

`select_output_storage(a, b)` 按以下优先级确定输出存储（`output_shape` 由广播规则计算，当阶段 2 命中时为 `a.shape()`）：

| 优先级 | 条件 | 输出存储 | 说明 |
|--------|------|----------|------|
| 1 | `a` 为 Owned 且未发生广播（`a.shape() == output_shape`）且 `a.capacity() >= output_padded_capacity` | 复用 `a` 的存储，输出继承 `a` 的 flags（PADDED/ALIGNED 等）和布局方向 | 最高优先级：左侧操作数的存储复用 |
| 2 | `b` 为 Owned 且未发生广播（`b.shape() == output_shape`）且 `b.capacity() >= output_padded_capacity` | 复用 `b` 的存储，输出继承 `b` 的 flags 和布局方向 | 次选：右侧操作数的存储复用 |
| 3 | 以上均不满足 | 新分配 Owned 存储，默认 64 字节对齐，PADDED = false，布局方向为 F-order | 新分配保证容量和对齐 |

**注**：阶段 2.5（广播 SIMD）要求广播操作数的步长为 0 且被广播维度 size 为 1，连续操作数须满足 `is_padded_contiguous()` 且 `all_strides_positive_or_zero()`。不满足时跳过阶段 2.5，直接进入阶段 3。

**实现策略建议**：

5 级分层表 + 二元 dispatch 规则覆盖 30+ 操作，每个操作手动实现分支会导致大量重复代码。推荐以下实现策略：

| 策略 | 适用场景 | 示例 |
|------|----------|------|
| **泛型 dispatch trait**（推荐） | 所有操作的统一入口 | 定义 `trait ElementWiseOp { fn scalar(&self, val: A) -> A; unsafe fn simd<S: Simd>(&self, vec: S::Vec) -> S::Vec; }`，dispatch 函数接收 `impl ElementWiseOp`，自动匹配分层表 |
| 壁纸宏 | 批量注册操作到分层表 | `impl_element_op!(Abs, |v| v.abs(), |s,v| s.abs(v));` 展开为 trait impl + dispatch 入口 |
| 闭包分发 | 一次性/非通用操作 | `dispatch_elementwise(arr, \|v\| v.exp(), \|s, v\| s.exp(v))` 闭包参数化标量/SIMD 实现 |

**推荐方案**：以泛型 dispatch trait 为主（类型安全、可测试、无宏卫生问题），壁纸宏辅助批量注册常见操作。避免纯闭包方案（难以单独测试、无法缓存 SIMD 实例）。

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
| Padding 初始化 | Padding bytes 在分配时必须已零初始化。同一 padding cache line 的并发写入安全由 Xenon 内部保证（见下方 "并行工作区间" 规则），用户无需关心 padding 区域的并发行为 |
| 并行工作区间 | 并行迭代须按**逻辑元素边界**分割工作区间，不得将 padding 字节分配给任何线程。线程间不得同时写同一 cache line 的 padding 区域（避免 false sharing）。**Xenon 内部分块责任**：当数组 PADDED=true 时，Xenon 的并行分块策略（见 §9.2.2）必须在 **padded 边界**（而非仅逻辑元素边界）分割工作区间。即分割点须为 padded stride 的整数倍位置，确保 SIMD 写入尾部 padding 字节时不与其他线程的数据 cache line 交叉。由于 SIMD 写覆写 padding 是 Xenon 内部行为（非用户可控），Xenon 必须自行保证分割安全性，不得推给调用者 |
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
| zip 形状不一致但可广播 | 自动广播后迭代，等价于先 broadcast 再 zip。广播视图使用零步长实现逻辑重复（无额外堆分配），步长为 0 的轴在迭代时重复产出同一位置的数据。**遍历顺序规则**：若未指定遍历顺序，使用库默认 F-order（广播视图本身无物理内存，"物理布局顺序"在此场景下无意义，统一使用 F-order 保证行为确定性）。若显式指定 F-order 遍历顺序，则按指定逻辑顺序迭代，可能产生非连续访问。**性能提示**：当所有非广播操作数均为 F-contiguous 且均为大数组时，使用 F-order 遍历可获得最优缓存性能 |
| zip 形状不一致且不可广播 | 返回 BroadcastError |
| 遍历顺序未指定 | 默认按物理内存布局顺序（连续数组最优）。对于广播视图等无明确物理布局的场景，使用 F-order |
| 遍历顺序指定为 F/C | 强制按指定逻辑顺序，可能非连续访问 |
| 并行迭代 | 将元素按连续块分割给线程，块大小 ≥ 并行阈值 |
| 窗口迭代越界 | 不产出不完整窗口（窗口数 = shape - window_size + 1） |
| 空数组迭代 | 立即结束，产出零个元素 |

**zip 广播迭代的内存分配行为**：广播迭代使用零步长模拟维度扩展，不产生实际数据拷贝。迭代器内部坐标计数器在静态维度下无堆分配，动态维度（IxDyn）下通常无堆分配（≤ 8 维栈分配，> 8 维回退堆分配）。

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

**`size_hint()` 精确性要求**：所有迭代器（包括窗口迭代器、zip 迭代器等）的 `size_hint()` 必须返回精确值（即 `(len, Some(len))`），以确保并行迭代的任务分割正确性。

### 11.5 IntoIterator 实现表

所有 TensorBase<S, D> 相关类型须实现 `IntoIterator`。各存储模式及引用方式的 `Item` 类型如下：

**按值消费（`into_iter()`）**：

| 类型 | `IntoIterator::IntoIter` | `Item` | 行为 |
|------|--------------------------|--------|------|
| `Tensor<A, D>`（Owned） | `IntoIter<A, D>` | `A` | 消耗所有权，逐元素 move/copy 出来。连续数组按指针递增顺序产出；非连续数组按 strides 跳跃寻址产出 |
| `TensorView<'a, A, D>` | `Iter<'a, A, D>` | `&'a A` | 视图本身是借用，无法 move 元素所有权。按值消费视图仅释放视图元数据（shape/strides），底层数据不受影响 |
| `TensorViewMut<'a, A, D>` | `IterMut<'a, A, D>` | `&'a mut A` | 同上，可变视图按值消费后释放元数据 |
| `ArcTensor<A, D>` | `IntoIter<A, D>` | `A` | 消耗 ArcTensor 所有权。若 Arc 引用计数 == 1，等效于 `Arc::try_unwrap()` 获取 Owned 后逐元素 move；若引用计数 > 1，先深拷贝获取独立 Owned 再逐元素 move |

**借用迭代（`&Tensor` / `&mut Tensor` 等）**：

| 类型 | `IntoIterator::IntoIter` | `Item` | 说明 |
|------|--------------------------|--------|------|
| `&Tensor<A, D>` | `Iter<'a, A, D>` | `&'a A` | 不可变借用迭代，不消耗所有权 |
| `&mut Tensor<A, D>` | `IterMut<'a, A, D>` | `&'a mut A` | 可变借用迭代，不消耗所有权 |
| `&TensorView<'a, A, D>` | `Iter<'a, A, D>` | `&'a A` | 再借用，等效于视图的不可变迭代 |
| `&mut TensorViewMut<'a, A, D>` | `IterMut<'a, A, D>` | `&'a mut A` | 再借用，等效于视图的可变迭代 |
| `&ArcTensor<A, D>` | `Iter<'a, A, D>` | `&'a A` | 不可变借用迭代，不增加 Arc 引用计数（借用 `&self` 即可读取数据） |
| `&mut ArcTensor<A, D>` | `IterMut<'a, A, D>` | `&'a mut A` | 可变借用迭代，需要 `&mut self` 保证独占访问，不触发 `make_mut()`（借用检查器保证无其他引用） |

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
| 内存布局 | 每个视图共享源数组底层存储（零拷贝） |
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


### 11.7 元素迭代器方法签名

**核心方法**：

| 方法 | 签名 | `Item` | 适用存储 | 说明 |
|------|------|--------|----------|------|
| `iter()` | `fn iter(&self) -> Iter<'_, A, D>` | `&A` | 所有存储模式 | 按物理内存布局顺序遍历所有元素。非连续数组按 strides 跳跃寻址 |
| `iter_mut()` | `fn iter_mut(&mut self) -> IterMut<'_, A, D> where S: StorageMut` | `&mut A` | Owned, ViewMutRepr | 可变元素迭代。**入口处须检查 `has_zero_stride()`**，若为 true 则 panic（见 §11.3）。`S: StorageMut` 约束在编译时排除 ViewRepr 和 ArcRepr |
| `iter_contiguous()` | `fn iter_contiguous(&self) -> Option<ContiguousIter<'_, A>>` | `&A` | 所有存储模式 | 仅**严格连续**数组（`is_contiguous()` = true，无 padding）返回 `Some`。实现 `DoubleEndedIterator` + `ExactSizeIterator` + `FusedIterator`。宽松连续（含 padding）的数组返回 `None`，须使用 `iter_padded_contiguous()` 或 `iter()` |
| `iter_contiguous_mut()` | `fn iter_contiguous_mut(&mut self) -> Option<ContiguousIterMut<'_, A>> where S: StorageMut` | `&mut A` | Owned, ViewMutRepr | 可变连续迭代。同上要求严格连续（`is_contiguous()`），检查 `has_zero_stride()` 和 `S: StorageMut` |
| `iter_padded_contiguous()` | `fn iter_padded_contiguous(&self) -> Option<PaddedContiguousIter<'_, A, D>>` | `&A` | 所有存储模式 | 宽松连续数组（`is_padded_contiguous()` = true）返回 `Some`，包括严格连续数组和含 padding 的填充连续数组。迭代器在每列末尾自动跳过 padding 字节（F-order 跳过 axis 0 尾部填充），对外表现为仅遍历逻辑元素。实现 `DoubleEndedIterator` + `ExactSizeIterator` + `FusedIterator` |
| `iter_padded_contiguous_mut()` | `fn iter_padded_contiguous_mut(&mut self) -> Option<PaddedContiguousIterMut<'_, A, D>> where S: StorageMut` | `&mut A` | Owned, ViewMutRepr | 可变 padded 连续迭代。同上但提供可变引用。检查 `has_zero_stride()` 和 `S: StorageMut` |
| `iter_ordered()` | `fn iter_ordered(&self, order: Order) -> OrderedIter<'_, A, D>` | `&A` | 所有存储模式 | 显式指定遍历顺序（`Order::F`，按列优先逻辑顺序）。强制按指定逻辑顺序遍历，即使与物理布局不一致（可能产生非连续访问） |
| `iter_ordered_mut()` | `fn iter_ordered_mut(&mut self, order: Order) -> OrderedIterMut<'_, A, D> where S: StorageMut` | `&mut A` | Owned, ViewMutRepr | 可变有序迭代。同上检查 |

**迭代器类型与 `DoubleEndedIterator`**：

| 迭代器 | 连续性要求 | `DoubleEndedIterator` |
|--------|-----------|----------------------|
| `Iter` | 无（支持所有数组） | 不实现 |
| `ContiguousIter` | 严格连续（`is_contiguous()` = true） | 实现 |
| `PaddedContiguousIter` | 宽松连续（`is_padded_contiguous()` = true） | 实现（须正确跳过列尾 padding） |

**严格连续 vs 宽松连续的迭代器选择**：

| 连续类型 | `iter_contiguous()` | `iter_padded_contiguous()` | `iter()` |
|----------|---------------------|---------------------------|----------|
| 严格连续 | `Some(ContiguousIter)` | `Some(PaddedContiguousIter)` | `Iter` |
| 宽松连续（有 padding） | `None` | `Some(PaddedContiguousIter)` | `Iter` |
| 非连续 | `None` | `None` | `Iter` |

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
| 遍历顺序 | 始终按 F-order（最左轴变化最快）产出 `(index, element)` 对。索引按逻辑坐标 F-order 递增：`(0,0,0) → (1,0,0) → … → (shape[0]-1, shape[1]-1, shape[2]-1)`，元素按对应逻辑位置取出（与物理内存布局一致） |
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

| 维度类型 D | `D::Pattern` | `Item`（不可变） |
|------------|-------------|-----------------|
| Ix0 | `()` | `((), &A)` |
| Ix1 | `usize` | `(usize, &A)` |
| Ix2 | `(usize, usize)` | `((usize, usize), &A)` |
| Ix3 | `(usize, usize, usize)` | `((usize, usize, usize), &A)` |
| Ix4~Ix6 | `(usize, ..., usize)` | `((usize, ..., usize), &A)` |
| IxDyn | `InlineArray<usize, 8>` | `(InlineArray<usize, 8>, &A)` |

索引使用 `D::Pattern`（§3.2），与 `Index` trait 的索引模式一致。索引始终按 F-order 逻辑坐标递增（最左轴变化最快），不受物理内存布局影响。

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



### 11.10 窗口迭代器补充

窗口迭代器在 §17.16 定义了基本语义。以下补充非连续源数组的行为和边界情况：

**非连续源数组的窗口迭代**：

| 属性 | 说明 |
|------|------|
| 支持非连续源 | 窗口迭代器支持任意 strides 的源数组（包括转置视图、切片等） |
| 窗口内步幅 | 继承源数组步幅（可能非连续） |
| 窗口间步幅 | 等于源数组各轴步长（即窗口滑动 1 个逻辑位置），可能非连续 |
| 缓存性能 | 非连续源数组（如转置视图）的窗口迭代产生非顺序内存访问，缓存命中率低于连续数组。性能敏感场景下，建议先 `as_f_contiguous()` 转为连续布局再迭代 |

**边界行为**：

| 场景 | 行为 | 说明 |
|------|------|------|
| 窗口尺寸某轴 > 源数组对应轴 | 产出零个窗口（迭代器立即结束） | 窗口数 = `shape[axis] - window_size[axis] + 1`，当 `window_size[axis] > shape[axis]` 时结果 < 0 → 钳位为 0 |
| 窗口尺寸某轴 == 源数组对应轴 | 产出恰好 1 个窗口（整个轴范围） | 窗口数 = 1 |
| 窗口尺寸 == 源数组形状（所有轴） | 产出恰好 1 个窗口（等于整个数组） | 退化为单次全数组视图 |
| 源数组某轴长度为 0 | 产出零个窗口 | 空数组无元素可迭代 |
| 窗口尺寸 ndim != 源数组 ndim | panic（`"window size dimension {wnd_ndim} does not match array dimension {arr_ndim}"`） | 维度不匹配，编译时无法检查（窗口尺寸为运行时值），须运行时 panic |
| 窗口视图的 `as_slice()` | 非连续窗口返回 `None`；仅当源数组严格连续且窗口为完整数组时返回 `Some` | 窗口视图可能继承源数组的非连续 strides |

**窗口视图的生命周期**：

| 属性 | 说明 |
|------|------|
| 生命周期绑定 | 窗口视图 `TensorView` 的生命周期绑定到 `Windows` 迭代器持有的源数组借用 |
| 不实现 `DoubleEndedIterator` | 仅实现 `Iterator` + `ExactSizeIterator` + `FusedIterator` |

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
| 模（复数） | `norm` | Complex | `A: ComplexScalar` | 返回 `Tensor<A::Real, D>`（元素类型变为实数）。值为 `|z|`（复数模），使用 hypot 避免溢出（见 §4.6）。**复数类型不提供 `abs` 方法**——因 `ComplexScalar: Numeric`，复数类型上同时定义 `abs()` 的两个 inherent impl（返回类型不同）会导致 Rust 编译错误 "multiple applicable items in scope"。复数用户应使用 `norm()` 获取模（数学语义等价于绝对值） |
| 符号函数 | `signum` | 整数、浮点、复数 | `A: Numeric` | 与 §4.4 `Numeric::signum()` 一致。方法名使用 `signum`（与 Rust 标准库 `i32::signum()` / `f64::signum()` 一致）。整数：正→1，零→0，负→-1；浮点同整数并传播 NaN；复数：z / |z|，零时返回零 |
| 平方 | `square` | 整数、浮点、复数 | `A: Numeric` | 等价于 `x * x`；对复数为 `z * z`（非模的平方）。整数溢出遵循 §12.8 策略 |
| 倒数 | `reciprocal` | 整数、浮点、复数 | `A: Numeric` | 等价于 `A::one() / x`；`reciprocal(0)` 对浮点返回 Inf（IEEE 754），对整数 panic（除零） |
| 整数幂 | `powi` | 整数、浮点、复数 | `A: Numeric` | `powi(&self, exp: i32) -> Tensor<A, D>`，逐元素 `x.powi(exp)`；整数溢出遵循 §12.8 策略。**负数幂行为**：浮点类型 `exp < 0` 正常计算（返回浮点结果，如 `2.0.powi(-1) = 0.5`）；**整数类型 `exp < 0` 时 panic**——整数除法无法表示非整数结果（与 Rust 标准库 `i32::pow(self, u32)` 只接受非负指数一致）。若需对整数进行负数幂运算，应先转换为浮点类型 |

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

> **设计说明 — 实数与复数同名运算的区分**：RealScalar（§4.5）和 ComplexScalar（§4.6）均提供 `exp`/`ln`/`sqrt` 方法，但语义不同（如 `sqrt(-1.0)` 对 RealScalar → NaN，对 ComplexScalar → `i`）。复数张量使用前缀 `c` 命名（`cexp`/`cln`/`csqrt`），理由：(1) Rust 不允许同一类型上多个 inherent impl 定义同名方法（即使 where 约束不重叠），统一方法名需引入额外 dispatch trait 增加复杂度；(2) 显式命名避免 IDE 自动补全歧义，用户可明确知道调用的语义；(3) 与 `conj`/`arg`/`norm` 等复数专用方法保持一致的命名空间。**`abs` 同理遵循此规则**：复数类型不提供 `abs`（因 `ComplexScalar: Numeric`，两个 inherent impl 的返回类型不同会导致编译冲突），复数用户应使用 `norm()` 获取模（数学语义等价于 `|z|`）

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

**约束 2**：`eq`/`ne` 仅要求 `PartialEq`；`lt`/`le`/`gt`/`ge` 要求 `PartialOrd`（浮点类型因 NaN 为偏序）。所有比较方法仅通过方法调用提供，不提供 `< <= > >=` 运算符语法（见 §20.6 设计决策）。支持广播（见 §12.6）。返回的 bool Tensor 可用于 `mask()` 等条件操作。

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
| 加减乘溢出（逐元素运算） | **panic**（debug） / **wrapping**（release） | 与 Rust 默认行为一致：debug 模式使用 `checked` 语义（溢出 panic），release 模式使用 `wrapping` 语义（溢出回绕）。**不使用** `saturating` 语义。**例外**：归约运算（sum）在 debug 和 release 模式均使用 checked 算术（溢出 panic），见 §14.3 |
| 整数除法 / reciprocal(0) | **panic** | 整数除零始终 panic，与 Rust `i32::div` 行为一致 |
| abs(i32::MIN) / abs(i64::MIN) | **panic** | |MIN| 超出类型范围（如 `|i32::MIN| = 2147483648 > i32::MAX`） |
| powi 溢出 | 同加减乘溢出策略 | debug panic / release wrapping |
| square 溢出 | 同加减乘溢出策略 | debug panic / release wrapping |

> **设计说明**：整数溢出策略选择与 Rust 默认一致（debug panic / release wrapping），而非 NumPy 的固定宽度回绕，理由：(1) Rust 惯用法——整数溢出在 debug 模式下 panic 是 Rust 社区的标准实践（`cargo test` 默认 debug 模式，溢出会被捕获）；(2) 安全性——debug 模式下溢出作为错误暴露，release 模式下 wrapping 提供确定性行为；(3) 与 `#[debug_assertions]` 配合——用户可通过编译配置控制行为，无需额外 API

---

## 13. 矩阵运算

### 13.1 支持的操作

| 操作 | 说明 |
|------|------|
| dot | 向量内积 |

### 13.2 矩阵运算语义

**dot 语义**：

| 属性 | 行为 |
|------|------|
| 签名 | `fn dot(&self, other: &Tensor<A, Ix1>) -> A where A: Numeric` |
| 操作（实数） | 向量内积：self(N) · other(N) → 标量，`result = Σ(self[k] * other[k])` |
| 操作（复数） | Hermitian 内积：`result = Σ(self[k].conj() * other[k])`，与 BLAS `cdotc`/`zdotc` 及 NumPy `np.vdot` 一致 |
| 错误处理 | 维度或形状不匹配时 panic。提供 `try_dot()` 返回 `Result<A, XenonError>` |
| 空数组 | self 或 other 长度为 0 时返回 `A::zero()` |

---

## 14. 归约操作

### 14.1 归约类型

| 类型 | 操作 |
|------|------|
| 全局 | sum |
| 沿轴 | 以上所有运算均支持沿指定轴归约 |

### 14.2 沿轴归约输出维度规则

| 属性 | 行为 |
|------|------|
| 默认行为 | 归约轴消除，输出 ndim = 输入 ndim - 1。例如 Ix3 沿 axis=1 归约 → Ix2 |
| keepdims 参数 | 提供 `sum_axis_keepdim(axis)` 变体，归约轴保留为 size-1，输出 ndim 不变 |
| 静态维度推导 | 归约后维度类型由编译时推导：Ix(N+1) 沿任意轴归约 → IxN。Ix0 不可执行沿轴归约（无轴可归约） |
| 动态维度 | IxDyn 沿轴归约后仍为 IxDyn（ndim 减少 1） |
| 全局归约 | 不指定轴时，所有轴归约为单个标量，返回 `A` |

### 14.3 约束

**整数归约溢出行为**：`sum` 作用于整数数组时，溢出**始终 panic**（debug 和 release 模式均如此）。**此策略与 §12.8 逐元素运算的溢出策略不同**：归约运算涉及多步累加，wrapping 导致的错误值会传播到最终结果且难以诊断；逐元素运算的每步结果独立，wrapping 语义可预测且与 Rust 默认一致。

> **溢出策略对比：归约 vs 逐元素运算**
>
> | 场景 | debug 模式 | release 模式 | 说明 |
> |------|-----------|-------------|------|
> | 逐元素 `a + b`（整数） | panic（overflow check） | **wrapping**（回绕） | 与 Rust 默认 `Add` 行为一致 |
> | 归约 `arr.sum()`（整数） | panic（溢出检测） | panic（同左） | 归约溢出几乎总是 bug |
> | 归约 `arr.sum()`（浮点） | 不涉及溢出 | 不涉及溢出 | 浮点无整数溢出概念 |

**整数归约的 wrapping 替代**：若用户需要 wrapping 语义（溢出回绕而非 panic），可先 `cast` 为更大的整数类型执行归约再 `cast` 回来，或使用自定义 newtype 实现 `Numeric` trait 并使用 `wrapping_add`。Xenon 不提供 `sum_wrapping()` 内置方法——理由：溢出静默回绕在归约场景中几乎总是 bug，不应作为默认选项暴露于公开 API。

**空数组归约行为**：

| 操作 | 空数组返回 | 说明 |
|------|-----------|------|
| `sum` | `A::zero()` | 加法单位元 |

---

## 15. 集合操作

### 15.1 集合操作类型

| 操作 | 说明 | 输入类型约束 | 返回类型 | 输入维度 |
|------|------|--------------|----------|----------|
| unique | 返回唯一值（排序后） | 可排序类型（整数、浮点、复数） | Tensor1<A> | 任意维度（先按 F-order 展平为一维） |

**排序能力需求**：集合操作中的 unique 需要对元素进行排序和去重。由于浮点类型（NaN 为偏序）和复数类型（无自然全序）不实现 `Ord`，需要一个独立的排序能力 trait（sealed），覆盖整数（自然全序）、浮点（IEEE 754 totalOrder）、复数（字典序：先实部后虚部，使用 totalOrder）。bool 不需要（unique 无实际意义），usize 仅作为索引返回类型不参与排序。此 trait 应为 `Element` 的子 trait，与 `Numeric`/`RealScalar`/`ComplexScalar` 并列。

**多维输入展平规则**：

| 规则 | 说明 |
|------|------|
| 展平顺序 | 按 F-order（列优先）展平为一维 |
| 展平开销 | 连续输入零拷贝；非连续输入需拷贝 |

### 15.2 unique 语义

| 属性 | 行为 |
|------|------|
| 排序 | 返回的唯一值按升序排列 |
| 空数组 | 返回空 Tensor1 |
| 返回值 | Tensor1<A>，长度等于唯一值数量 |
| 浮点比较 | 浮点类型使用 IEEE 754 totalOrder 进行排序和去重 |
| 复数排序 | 复数使用字典序（先实部后虚部），各分量使用 totalOrder |
| NaN 处理 | 所有 NaN 视为相等，合并为一个，排在末尾（大于 +∞） |
| ±0.0 处理 | +0.0 与 -0.0 视为相同值（IEEE 754 下 `==` 为 true），合并为排序后首出现（-0.0）。**此行为与 NumPy 不同**（NumPy 基于位模式区分）。提供 `unique_by_key` 方法供用户按自定义键去重（如按位模式区分 ±0.0） |

---

## 16. 广播

### 16.1 基本规则

遵循 NumPy 广播规则（右对齐，size-1 维度拉伸）。广播是零拷贝操作——通过零步长（stride=0）模拟维度扩展，不产生实际数据拷贝（见 §17.1 零拷贝操作分类）。

**广播规则**：

| 规则 | 说明 |
|------|------|
| 维度对齐 | 从最右维度开始对齐，维度数不足的数组在左侧补 1（仅逻辑上，不改变原数组） |
| 兼容检查 | 对应维度相等，或其中一个为 1。不满足时返回 `BroadcastError`（见 §27.2） |
| 输出形状 | 每个轴取两个输入中非 1 的值；均为非 1 时须相等。结果轴大小取非 1 的值，而非 `max(a, b)`——这对 size-0 维度至关重要（`a=0, b=1` 结果为 `0`，而非 `max(0,1)=1`） |
| 零步长 | size-1 维度广播为目标大小时，该轴步长设为 0，逻辑上重复该维度的单个元素 |

**广播形状示例**：

| shape_a | shape_b | 输出形状 | 说明 |
|---------|---------|----------|------|
| `[3, 4]` | `[3, 4]` | `[3, 4]` | 形状一致，无广播 |
| `[3, 1]` | `[1, 4]` | `[3, 4]` | 双方各一个轴被广播 |
| `[3, 4]` | `[4]` | `[3, 4]` | 右侧自动补 1 为 `[1, 4]` |
| `[]` (0D) | `[3, 4]` | `[3, 4]` | 0D 广播（标量） |
| `[1, 3, 4]` | `[2, 1, 4]` | `[2, 3, 4]` | 多轴广播 |
| `[3, 4]` | `[5, 4]` | `BroadcastError` | axis 0 不兼容（3 ≠ 5，均非 1） |

**设计约束**：

| 约束 | 说明 |
|------|------|
| 广播视图为只读 | 广播产生的视图始终为只读（写入语义不明确） |
| 算术运算符隐式支持广播 | 二元运算符自动触发广播（见 §12.6、§20） |
| 标量广播 | 标量视为 0 维数组（`Tensor0<A>`），可与任意维度数组广播 |
| 原地广播运算 | `a += b` 中 b 可被广播，a 不可被广播。左操作数须为连续内存且无零步长，确保每个逻辑位置有唯一的物理写入目标 |

### 16.2 显式广播 API

| 项目 | 定义 |
|------|------|
| 签名 | `fn broadcast<D2>(&self, shape: D2) -> Result<TensorView<'_, A, D2::Dim>, BroadcastError> where D2: IntoDimension` |
| 语义 | 创建零步长视图，逻辑上将数组扩展到目标形状，无数据拷贝 |
| 成功条件 | self 形状与目标形状按 §16.1 规则兼容 |
| 失败 | 返回 `BroadcastError{left, right, conflicting_axis}`（见 §27.2） |
| 返回维度类型 | 由 `D2::Dim` 参数决定 |
| 视图生命周期 | 绑定到 `&self` |

### 16.3 边界行为

**0D（标量）广播**：0D 数组等价于左侧补 1 直到目标 ndim，所有轴步长均为 0。运算符重载中 `A`（标量）参与二元运算时，等价于构造 `Tensor0<A>` 后广播（见 §20.2）。

**含 size-0 维度的广播**：size-0 与 size-1 兼容（结果为 0），size-0 与 size-0 兼容（结果为 0），size-0 与 size-N（N>1 且 N≠0）不兼容（报错）。广播结果为空数组时，迭代器立即结束，无实际计算开销。

**广播后维度数 > 6**：须使用 `IxDyn`。运算符要求两操作数维度类型相同（`D` 一致），不产生维度提升。

### 16.4 广播后维度类型推导

| 场景 | 约束 |
|------|------|
| 两个相同维度类型 | 正常，返回 `Tensor<A, D>` |
| 不同维度类型 | **编译错误**——须先显式转换（如 `into_dyn()`） |
| 隐式广播（相同 D） | 运行时按 §16.1 规则广播，维度类型不变 |

`broadcast()` 方法通过泛型参数 `D2: IntoDimension` 允许输出不同的维度类型。

### 16.5 可恢复广播运算（try_ 变体）

二元运算符因 trait 签名 `Output = Tensor<A, D>` 限制，广播形状不兼容时只能 panic（见 §27.3）。提供以下 `try_` 前缀方法作为可恢复错误路径：

| 方法 | 签名 | 说明 |
|------|------|------|
| `try_add` | `fn try_add(&self, other: &Tensor<A, D>) -> Result<Tensor<A, D>, XenonError>` | 广播失败返回 `Err(BroadcastError)` |
| `try_sub` | `fn try_sub(&self, other: &Tensor<A, D>) -> Result<Tensor<A, D>, XenonError>` | 同上 |
| `try_mul` | `fn try_mul(&self, other: &Tensor<A, D>) -> Result<Tensor<A, D>, XenonError>` | 同上 |
| `try_div` | `fn try_div(&self, other: &Tensor<A, D>) -> Result<Tensor<A, D>, XenonError>` | 同上 |

`try_*` 方法的运算语义与对应运算符完全一致，仅错误处理方式不同。

---

## 17. 形状操作

### 17.1 transpose 语义

| 属性 | 行为 |
|------|------|
| 操作 | `transpose()` 反转所有轴顺序 |
| 零拷贝 | 始终零拷贝，仅翻转 strides 和 shape 数组 |
| 输出维度 | ndim 不变，shape 和 strides 均反转 |
| 步长变换 | `strides_out[i] = strides_in[ndim - 1 - i]` |
| 1D/0D 数组 | 转置为 no-op |

**指定轴转置 `transpose(axes)`：**

| 属性 | 行为 |
|------|------|
| 操作 | 按指定排列重排轴 |
| 约束 | `axes` 须为 `[0, ndim)` 的排列（长度 == ndim，无重复，无遗漏），否则 panic |
| 步长变换 | `strides_out[i] = strides_in[axes[i]]` |

---

## 18. 索引操作

### 18.1 索引类型

| 类型 | 说明 |
|------|------|
| 多维索引 | `[i, j, k]` 形式，i, j, k 为元素索引 |
| 范围索引 | 切片语法 |
| 高级索引 | mask |

### 18.2 mask 语义

| 属性 | 行为 |
|------|------|
| 签名 | `fn mask(&self, mask: &Tensor<bool, D>) -> Tensor1<A>` |
| 操作 | 按布尔掩码提取元素，返回一维数组 |
| 约束 | mask 形状须与 self 形状完全一致，否则返回 `ShapeMismatch` 错误 |
| 返回值 | Tensor1<A>，长度为 mask 中 true 的数量，元素按 F-order 遍历顺序收集 |
| 拷贝 | 始终拷贝 |

---

## 19. 构造操作

### 19.1 构造方法

| 方法 | 说明 |
|------|------|
| zeros/ones | 支持指定 Order |
| eye | 单位矩阵（见下方语义表） |

### 19.2 eye 语义

| 方法 | 签名 | 说明 |
|------|------|------|
| eye | `fn eye(n: usize) -> Tensor<A, Ix2>` | n×n 单位矩阵，对角线为 `one()`，其余为 `zero()` |

| 约束 | 要求 |
|------|------|
| Numeric 约束 | 调用方须满足 `A: Numeric`（需要 `zero()`/`one()`，定义在 Numeric trait 上而非 Element） |
| 维度固定 | 返回类型固定为 `Tensor<A, Ix2>`（二维），不支持高维 |
| 内存布局 | 默认 F-contiguous（列优先，与 §7.1 项目默认布局一致） |

---

## 20. 运算符重载

### 20.1 重载类型

| 类别 | 运算符 | 说明 |
|------|--------|------|
| 四则运算 | Add, Sub, Mul, Div | 标准算术运算符重载，隐式支持广播 |

**不在当前版本范围内的运算符**：比较运算符、复合赋值运算符、一元运算符（Neg/Not）、位运算符。

### 20.2 二元运算符所有权矩阵

存储复用由操作数的**所有权语义**（值 vs 引用）显式决定——传入 `Tensor`（值，消耗所有权）表示允许复用存储，传入 `&Tensor`（引用，不消耗所有权）表示不复用。

| 左操作数 | 右操作数 | 存储复用 | 返回类型 |
|----------|----------|----------|----------|
| `&Tensor<A, D>` | `&Tensor<A, D>` | 无，分配新数组 | `Tensor<A, D>` |
| `Tensor<A, D>` | `&Tensor<A, D>` | 复用左操作数存储 | `Tensor<A, D>` |
| `&Tensor<A, D>` | `Tensor<A, D>` | 复用右操作数存储 | `Tensor<A, D>` |
| `Tensor<A, D>` | `Tensor<A, D>` | 复用左操作数存储 | `Tensor<A, D>` |
| `&Tensor<A, D>` | `A`（标量） | 无，分配新数组 | `Tensor<A, D>` |
| `Tensor<A, D>` | `A`（标量） | 复用左操作数存储 | `Tensor<A, D>` |
| `A`（标量） | `&Tensor<A, D>` | 无，分配新数组 | `Tensor<A, D>` |
| `A`（标量） | `Tensor<A, D>` | 复用右操作数存储 | `Tensor<A, D>` |

**存储复用前提**：被复用的操作数须为连续内存且形状与输出一致（含广播展开后）。若不满足，静默回退到分配新数组。

**跨维度运算**：二元运算符要求两操作数的维度类型 `D` 相同（见 §16.4）。不同维度类型的运算会产生编译错误，须先通过 `into_dyn()` 统一为 `IxDyn`。

**标量运算符约束**：受 Rust 孤儿规则限制，LHS 标量运算（如 `scalar + tensor`）须为每种具体标量类型（`f32, f64, i32, i64, Complex<f32>, Complex<f64>`）生成独立的 impl 块，不支持泛型 LHS 标量。

---

## 21. 实用操作

### 21.1 clip 语义

| 属性 | 行为 |
|------|------|
| 操作 | 逐元素裁剪到 `[min, max]` 范围内 |
| min > max | panic（调用方违反前置条件） |
| 返回类型 | 新分配的 Owned 数组 |
| 元素为 NaN | NaN 不满足 min/max 约束，保持不变 |
| min 为 NaN | panic（调用方违反前置条件） |
| max 为 NaN | panic（调用方违反前置条件） |
| 整数类型 | 仅执行 `min > max` 前置条件检查，无 NaN 相关行为 |

### 21.2 fill 语义

| 方法 | 说明 |
|------|------|
| `fill` | 用指定值填充所有逻辑元素。要求可写存储。广播视图（`has_zero_stride()` 为 true）时 panic |

---

## 22. 连续性保证

连续性保证方法提供三组操作：
- `to_*` 系列：接受 `&self`，始终返回新分配的 Owned Tensor，保证严格 F-contiguous 布局
- `as_*` 系列：接受 `&self`，零拷贝视图路径，仅当已满足连续性时返回 `Some(TensorView)`，否则返回 `None`
- `into_*` 系列：接受 `self`（仅 Owned），已连续时零拷贝返回 `self`，否则拷贝后返回新 Tensor

### 22.1 判定标准

连续性判定使用严格连续性（`is_f_contiguous()`，见 §7.4）。填充数组（`PADDED = true`）被视为非严格连续；广播视图（`has_zero_stride() = true`）同样不满足严格连续性。

### 22.2 方法签名与行为

**`to_*` 系列（引用输入，始终返回新 Owned）**

| 方法 | 说明 |
|------|------|
| `to_f_contiguous()` | 始终返回新的 Owned Tensor。若输入已严格 F-contiguous，等价于 `to_owned()`；否则重新排列为严格 F-contiguous 布局 |
| `to_contiguous()` | 等价于 `to_f_contiguous()`（仅支持 F-order） |

**`as_*` 系列（零拷贝视图路径）**

| 方法 | 说明 |
|------|------|
| `as_f_contiguous()` | 若输入已严格 F-contiguous，返回 `Some(TensorView)`（零拷贝，生命周期绑定到 `&self`）；否则返回 `None` |
| `as_contiguous()` | 等价于 `as_f_contiguous()`（仅支持 F-order） |

**`into_*` 系列（消耗 self，按需拷贝）**

仅对 `Tensor<A, D>`（Owned 存储）实现。

| 方法 | 说明 |
|------|------|
| `into_f_contiguous()` | 消耗 `self`。若已严格 F-contiguous，零拷贝返回 `self`；否则拷贝为严格 F-contiguous 布局的新 Tensor |
| `into_contiguous()` | 等价于 `into_f_contiguous()`（仅支持 F-order） |

### 22.3 宽松连续性变体

提供宽松连续性变体，允许默认构造的填充张量零拷贝获取视图：

| 方法 | 说明 |
|------|------|
| `as_f_padded_contiguous()` | 若输入满足宽松 F-连续性（`is_f_padded_contiguous()`），返回 `Some(TensorView)`；否则 `None` |

宽松变体不适用于零拷贝 reshape，仅适用于 SIMD 运算、BLAS FFI 等能正确处理填充的场景。

### 22.4 输出布局属性

所有 `to_*` 和 `into_*` 方法输出的 Tensor 满足：

| 属性 | 值 |
|------|-----|
| 步长 | 严格连续步长（无填充） |
| PADDED 标志 | `false` |
| F_CONTIGUOUS 标志 | `true` |
| 对齐 | 新分配路径：64 字节默认对齐；零拷贝路径：继承原对齐 |

### 22.5 适用存储模式

| 方法系列 | 适用存储 |
|----------|---------|
| `to_*` | 所有 `S: Storage`（返回新 Owned Tensor） |
| `as_*` | 所有 `S: Storage`（返回的视图生命周期绑定到 `&self`） |
| `into_*` | 仅 `Tensor<A, D>`（Owned） |

### 22.6 边界情况

| 输入 | `to_f_contiguous()` | `as_f_contiguous()` | `into_f_contiguous()` |
|------|---------------------|---------------------|----------------------|
| 0D 标量 | `to_owned()` | `Some(view)` | 零拷贝返回 `self` |
| 空张量 | `to_owned()` | `Some(view)` | 零拷贝返回 `self` |
| 1D 数组 | `to_owned()` | `Some(view)` | 零拷贝返回 `self` |
| 填充数组 | 拷贝并重排为严格 F-contiguous | `None` | 拷贝并重排为严格 F-contiguous |
| 广播视图 | 拷贝为 F-contiguous | `None` | N/A（非 Owned） |
| 非连续视图 | 拷贝并重排为 F-contiguous | `None` | N/A（非 Owned） |
| 非连续 Owned | 拷贝并重排为 F-contiguous | `None` | 拷贝并重排为严格 F-contiguous |

---

## 23. 转换操作

### 23.1 cast 语义

**方法签名**：

| 方法 | 签名 | 说明 |
|------|------|------|
| `cast` | `fn cast<B>(self) -> Tensor<B, D> where B: Element` | 消耗 Owned 数组，逐元素类型转换。**仅对 `Tensor<A, D>`（Owned）实现**。View 须先 `to_owned()` 再 cast |
| `cast_same` | `fn cast_same(&self) -> Tensor<A, D>` | 返回同类型拷贝（等价于 `to_owned()`）。设计意图：在泛型管线中提供与 `cast::<B>()` 对称的 API，避免调用方在 `B == A` 分支中使用不同的方法名。与 `.clone()` 的区别：`clone` 要求 `Clone`（`Tensor<A, D>` 始终满足），`cast_same` 强调"类型转换管线中的同类型路径"语义 |

**泛型约束**：`B: Element` 约束保证目标类型在本库的类型体系内（Sealed trait，见 §4.2）。转换规则为逐元素按值转换，始终成功（溢出使用饱和语义，见下方精度表）。不支持跨类别隐式转换（如 `Complex → 实数` 须显式取 `re()`）。当 `A == B` 时，`cast` 等价于深拷贝（消耗 self，分配新的 `Tensor<A, D>`），与 `to_owned()` 行为一致——语义上合法但无实际类型转换效果，调用方应优先使用 `to_owned()` 或 `clone()` 以避免歧义。

**编译优化提示**：当 `A == B` 时，`cast::<A>()` 的泛型单态化会生成一个逐元素拷贝循环。虽然 LLVM 可能将其优化为 `memcpy`，但不可依赖——调用方应在类型已知的非泛型上下文中直接使用 `clone()` / `to_owned()` 而非 `cast::<SameType>()`，以避免不必要的单态化代码膨胀。

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
| `Vec<A>` | `Tensor1<A>` | 1D，长度 = vec.len() | F-contiguous |
| `&[A]` | `TensorView<A, Ix1>` | 1D，长度 = slice.len() | 借用，零拷贝 |
| `[A; N]` | `Tensor1<A>` | 1D，长度 = N | 从栈数组移动 |
| `Vec<Vec<A>>` | `Tensor2<A>` | 2D，shape = (outer.len(), inner.len()) | **仅提供 `TryFrom`**，不实现 `From`。所有内层 Vec 长度须一致，否则返回 `Err(InconsistentLengths)`。理由：`From` trait 暗示 infallible 转换，但 `Vec<Vec<A>>` 的内层长度一致性无法在编译时验证，运行时检查失败应返回 `Result` 而非 panic。使用方式：`Tensor2::try_from(nested_vec)?` 或 `TryInto::<Tensor2<A>>::try_into(nested_vec)?` |
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
| `Default` | `Tensor<A, D>` | **仅为 `Tensor<A, Ix0>` 实现**（即 0D 标量张量）。实现条件形式化为 `A: Default + Element`，维度类型固定为 `Ix0`（而非泛化到 `D: Default`——虽然 `Ix1`~`Ix6` 和 `IxDyn` 在 §3 中均实现 `Default`，但对非零维度类型返回 `Default` 张量缺乏明确的 shape 语义，因此不予实现）。返回包含 `A::default()` 的 0D 标量张量。实际用途有限——创建数组推荐使用 `zeros(shape)` |
| `From<Vec<A>>` | `Tensor1<A>` | 见 §23.2 标准类型转换 |
| `From<[A; N]>` | `Tensor1<A>` | 见 §23.2 标准类型转换 |

---

## 24. 格式化输出

### 24.1 设计原则

| 原则 | 说明 |
|------|------|
| Display 面向用户 | 简洁、可读，只显示数据内容，不显示内部实现细节（存储模式、步长、偏移量） |
| NumPy 风格 | 外层包裹 `array(...)`，数据部分使用方括号 `[]` 包裹，逗号+空格分隔元素 |

格式模板：`array({数据内容})`

### 24.2 元素级格式化规则

各元素类型在数组格式化中的显示规则：

| 元素类型 | Display（数组内） |
|----------|-------------------|
| 整数（i32/i64/usize） | 使用 `Display` trait |
| 浮点（f32/f64） | 使用 Rust 标准库的 `Display` trait（最短表示），不固定有效数字位数 |
| bool | `"true"` / `"false"` |
| Complex | 使用 Complex 的 Display 格式（见 §5.2），始终保持 `re+imi` 格式 |

| 特殊值 | 显示 |
|--------|------|
| NaN | `NaN` |
| +Inf | `inf` |
| -Inf | `-inf` |

### 24.3 Display 输出规范

Display 只显示数据内容，不显示 shape、dtype、存储模式、步长或偏移量。

| 属性 | 规则 |
|------|------|
| 外层包裹 | `array(...)` |
| 1D 数组 | `array([元素0, 元素1, ..., 元素N-1])`，逗号+空格分隔 |
| 2D 数组 | `array([[行0],\n [行1], ..., [行M-1]])`，每行缩进 1 空格，行内逗号+空格分隔 |
| 3D 及以上 | 递归嵌套，每增加一维增加 1 空格缩进，最内层为行向量 |
| 0D 标量 | `array(标量值)`，无中括号 |
| 空数组 | `array([])` |

### 24.4 大数组截断规则

当元素总数超过阈值时触发截断，避免输出过长。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `threshold` | 1000 | 触发截断的元素总数阈值 |
| `edge_items` | 3 | 截断时首尾保留的行数/列数 |

截断行为：

| 行为 | 说明 |
|------|------|
| 行级截断（沿 axis 0） | 保留前 `edge_items` 行和后 `edge_items` 行，中间替换为 `...` |

当前版本使用固定默认值，不提供用户可配置的截断参数。

### 24.5 空数组与 0D 标量

**空数组**：shape 中任一轴为 0 的数组。数据部分固定为 `[]`。

**0D 标量**（shape=[], ndim=0, len()=1）：数据部分**不带中括号**，直接显示标量值（元素自身的 Display 格式）。

---

## 25. FFI 集成

### 25.1 指针 API

提供原始指针访问，供上游库零开销集成：

| 方法 | 说明 |
|------|------|
| as_ptr() | 返回数据起始位置的不可变原始指针 |
| as_mut_ptr() | 返回数据起始位置的可变原始指针（须可写存储） |
| as_ptr_unchecked() | unsafe，不检查偏移量有效性 |
| capacity() | 返回物理缓冲区容量（含 padding，元素单位） |

ArcRepr 的裸指针须注意 `make_mut()` 可能使指针失效（旧 buffer 被 drop）。FFI 场景下须通过独占所有权、借用保护或引用计数锁定确保安全。

### 25.2 不安全构造函数

| 方法 | 说明 |
|------|------|
| `Tensor::from_raw_parts` | 从裸指针构造 Tensor，获取内存所有权（drop 时释放） |
| `TensorView::from_raw_parts_view` | 从裸指针构造只读视图，不获取所有权 |
| `TensorViewMut::from_raw_parts_view_mut` | 从裸指针构造可变视图，不获取所有权（独占语义） |
| `Tensor::into_raw_parts` | 解构 Tensor 为裸指针 + shape + strides + offset + capacity，阻止 Rust 释放内存 |

以上均为 `unsafe`（`into_raw_parts` 除外），调用方须保证指针非空、对齐、指向有效内存、shape × strides 安全、分配器兼容等前置条件。`into_raw_parts` 仅支持 `Owned` 存储模式。

### 25.3 形状与步长查询（字节单位）

| 方法 | 说明 |
|------|------|
| strides_bytes() | 返回各轴步长（字节单位，有符号） |
| offset_bytes() | 返回数据起始偏移量（字节单位） |

### 25.4 BLAS 兼容性

提供 BLAS 兼容性查询与布局转换 API，支持上游库通过指针直接与 BLAS 交互：

| 方法 | 说明 |
|------|------|
| lda() | 返回 leading dimension，不可用时返回 `LdaError` |
| is_blas_compatible() | 检查内存布局是否可直接传递给 BLAS |
| blas_layout() | 返回 BLAS 布局标识（F/C/None） |
| blas_trans() | 返回 BLAS 转置参数（基于物理内存布局推断） |
| ensure_blas_compatible() | 确保内存布局为 BLAS 兼容，已兼容时零拷贝返回借用视图，否则拷贝为连续布局 |

**LdaError**：`InsufficientDimension`（ndim < 2）和 `NonContiguous`（非连续布局）。

**BLAS 兼容条件**：宽松连续、正步长、无零步长、ndim ≥ 1。

**blas_trans() 语义**：仅对 ≤2D Tensor 有明确语义；根据物理内存布局推断转置参数，不自动推断共轭。

**ensure_blas_compatible() 返回 CowRepr**（Copy-on-Write 存储表示）：已兼容时返回 `Borrowed`（零拷贝），不兼容时返回 `Owned`（拷贝为连续布局）。CowRepr 当前仅在 `ensure_blas_compatible()` 返回值中使用，不实现 Storage/StorageMut trait。

### 25.5 Trans 枚举

| 变体 | 含义 |
|------|------|
| `Trans::None` | 不转置 |
| `Trans::Transpose` | 转置 |
| `Trans::ConjTranspose` | 共轭转置（仅 Complex 类型有意义） |

### 25.6 索引转换

| 方法 | 说明 |
|------|------|
| index_to_ptr(index) | 多维索引 → 元素原始指针 |
| index_to_ptr_unchecked(index) | unsafe，不检查越界 |
| index_to_offset(index) | 多维索引 → 元素偏移量 |
| index_to_offset_unchecked(index) | unsafe，不检查越界 |

checked 变体越界时 panic，unchecked 变体越界为 UB。

### 25.7 FFI 使用模式

| 模式 | 说明 |
|------|------|
| 只读传递给 C 库 | 借用 Tensor 获取 `as_ptr()`（同步），或 `into_raw_parts()` 转移所有权（异步） |
| 可写传递给 C 库 | 获取 `as_mut_ptr()` + `capacity()` + `lda()`，C 库写入不得超过容量边界 |
| 从 C 库接收数据构造 Tensor | C 库返回裸指针和形状信息，通过 `from_raw_parts()` 构造（须确保分配器兼容） |
| BLAS 集成 | `ensure_blas_compatible()` 获取兼容布局，配合 `blas_layout()` / `blas_trans()` / `lda()` / `as_ptr()` 调用 BLAS 函数 |

---

## 26. 临时工作空间

### 26.1 工作空间属性

| 属性 | 要求 |
|------|------|
| 对齐 | 默认 64 字节对齐，须支持调用方指定更大对齐（如 128 字节） |
| 生命周期 | 借用语义：工作空间借出期间不可再次借出，归还后可复用 |
| 增长策略 | 请求超出当前容量时自动扩容，不缩容；扩容后保持对齐。提供将未使用空间归还给分配器的能力 |
| 线程亲和性 | 工作空间不绑定线程；线程安全由调用方通过借用规则保证 |
| 零初始化 | 不保证零初始化（性能考虑），调用方须自行初始化 |
| 分配模型 | 使用递增指针分配模型，借出时推进指针，`reset()` 后可重用已分配内存 |
| no_std 兼容 | 支持 `no_std` + `alloc` 环境 |

### 26.2 对齐填充规则

每次借出操作须保证返回的缓冲区按请求的对齐值对齐，对齐填充消耗的空间计入工作空间使用量。

### 26.3 分割与嵌套

| 属性 | 要求 |
|------|------|
| 嵌套借用 | 支持从同一工作空间分割多个不重叠的子缓冲区 |
| 并行分割 | 支持 `split_at` 将工作空间分割为两个不重叠的子工作空间，各自可独立借出 |
| 递归分割 | 子工作空间可继续递归分割，设有最小分割粒度下限 |

### 26.4 工作空间线程安全

| 属性 | 要求 |
|------|------|
| Send | 工作空间可在线程间移动 |
| Sync | 不实现 Sync，不可被多线程同时访问 |
| 并行使用 | 通过 `split_at` 分割后的子工作空间可分别移动到不同线程独立使用 |

### 26.5 扩容安全性约束

| 约束 | 说明 |
|------|------|
| 借用期间不可扩容 | 活跃借用期间不允许扩容，由借用生命周期保证 |
| 分割后扩容 | 子工作空间扩容时脱离共享缓冲区，创建独立分配，不影响其他子工作空间 |

### 26.6 ScratchRequirement 类型

描述操作所需的临时缓冲区大小和对齐要求。

| 属性 | 说明 |
|------|------|
| `size` | 所需字节数 |
| `align` | 所需对齐（须为 2 的幂） |
| 便捷构造 | 支持按元素类型和数量构造 |

### 26.7 借出与归还 API

| 方法 | 说明 |
|------|------|
| `borrow_bytes` | 借出指定字节数的缓冲区，按工作空间当前对齐值对齐 |
| `borrow_aligned` | 借出指定元素数的类型化缓冲区，按元素类型对齐和工作空间对齐的较大值对齐。返回未初始化缓冲区，调用方须自行初始化 |
| `borrow_aligned_init` | 借出并初始化为指定值的类型化缓冲区，返回已初始化引用 |
| `borrow_aligned_zeroed` | 借出并零初始化的类型化缓冲区，返回已初始化引用 |
| `borrow_aligned_write` | 借出缓冲区并将源数据拷贝写入，返回已初始化引用 |
| `borrow_scratch` | 按 `ScratchRequirement` 借出缓冲区 |
| 归还 | 借出的引用生命周期结束后自动归还，无需显式调用 |

### 26.8 查询与管理 API

| 方法 | 说明 |
|------|------|
| `capacity` | 返回当前已分配的字节容量 |
| `remaining` | 返回剩余可用字节数 |
| `reset` | 重置分配指针，使已分配空间可重用 |
| `shrink_to` | 将未使用空间归还给分配器 |

---

# 第五部分：工程保障

## 27. 错误处理

### 27.1 错误分类

所有公开错误枚举须标注 `#[non_exhaustive]`。库须定义覆盖以下场景的错误类型体系：

**可恢复错误**（返回 Result）：

| 错误场景 | 说明 |
|----------|------|
| 形状不兼容 | 二元运算/zip 形状不兼容且无法广播 |
| 广播规则不满足 | 非兼容维度无法广播 |
| 布局不满足要求 | 要求连续布局但输入非连续 |
| 轴索引无效 | 轴索引超出维度数 |
| 构造参数形状无效 | 如 cat 输入 ndim 不一致 |
| 维度互转失败 | 静态/动态维度互转时维度数不匹配 |
| 空数组归约 | 对空数组执行要求至少一个元素的归约操作 |
| 参数语义不合法 | 参数值不合法（非结构/索引类问题） |
| 布局/对齐验证失败 | shape × strides 不一致、指针未对齐、对齐值无效等（safe 路径返回 Result，unsafe 路径前置条件违反时 panic，见 §25.2） |
| 嵌套容器长度不一致 | 如 Vec<Vec<A>> 内层长度不一致 |

**不可恢复错误**（panic）：

| 错误场景 | 说明 |
|----------|------|
| 内存分配失败 | 超大数组或系统内存不足 |
| 索引越界 | 编程错误，同时提供 unsafe unchecked 变体 |

错误类型之间须有清晰的边界：参数语义不合法、结构不合法、索引超范围等应归为不同错误类型；当问题可归为更具体的类型时，优先使用更具体的类型。

### 27.2 错误诊断信息

所有返回 Result 的错误类型须携带充分的诊断信息，使调用方能定位问题根因。诊断信息须包含人类可读描述（通过 `Display` 输出）。panic 类错误通过 panic message 传递诊断信息。

### 27.3 处理策略

**判断原则**：panic 用于"调用方违反 API 前置条件"（编程错误），Result 用于"输入数据属性导致操作无法完成"（合法调用路径，调用方应处理）。

| 策略 | 适用场景 |
|------|----------|
| 返回 Result | 形状/布局/轴错误、参数语义不合法、维度互转失败等可恢复错误 |
| panic | 索引越界（同时提供 unchecked 变体）、前置条件违反、内存分配失败 |

运算符重载触发的隐式广播失败时 panic（因运算符 trait 无法返回 Result）；须同时提供 `try_*` 系列方法（见 §16.5）返回 Result 作为可恢复路径。也可使用 `broadcast` API（§16.2）显式广播后再运算。

### 27.4 no_std 兼容

错误类型在 `std` feature 下实现 `std::error::Error` trait，在 `no_std` 下仅要求 `Display + Debug + Clone + Send + Sync`，不引入额外依赖。内存分配失败在 `std` 环境下 panic，在 `no_std` 环境下由全局分配器决定行为。无论哪种环境，诊断信息均须包含请求的分配大小。

### 27.5 Panic Safety

panic 发生后须保证：(1) 可安全 drop（无 UB、无内存泄漏）；(2) 不保证数据一致性。

| 操作场景 | Panic 时保证 |
|----------|------------|
| 深拷贝（如 make_mut 分配失败） | 原数据未修改 |
| 溢出检查（如整数 sum） | Tensor 可安全 drop |
| 广播视图 iter_mut | 数据保持原状态（panic 在写入前触发） |
| 并行写入部分完成 | 已修改元素保持在最终写入状态，Tensor 可安全 drop |

并行操作 panic 时：单个线程 panic 不影响其他线程内存安全；panic 须立即传播，不得吞没。所有 drop 实现不得 panic。

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
| 行覆盖率（整体） | ≥ 90% |
| unsafe 代码块覆盖率 | ≥ 95%（单独统计） |
| 分支覆盖率（关键路径） | ≥ 85% |

### 28.3 数值精度

**浮点与复数精度表**：

| 运算类别 | f64 精度 | f32 精度 | Complex\<f64\> 精度 | Complex\<f32\> 精度 |
|----------|----------|----------|---------------------|---------------------|
| 加减乘 | ≤ 1 ULP | ≤ 1 ULP | re/im 各 ≤ 1 ULP | re/im 各 ≤ 1 ULP |
| 除法 | ≤ 1 ULP | ≤ 1 ULP | re/im 各 ≤ 4 ULP | re/im 各 ≤ 4 ULP |
| 归约（sum） | 相对误差 ≤ 1e-14（N ≤ 10⁶） | 相对误差 ≤ 1e-5（N ≤ 10⁶） | 模相对误差 ≤ 1e-13（N ≤ 10⁶） | 模相对误差 ≤ 1e-4（N ≤ 10⁶） |
| 超越函数 | 相对误差 ≤ 1e-14，ULP ≤ 4 | 相对误差 ≤ 1e-5，ULP ≤ 4 | 模相对误差 ≤ 1e-13 | 模相对误差 ≤ 1e-4 |

归约精度指标适用于 N ≤ 10⁶；大数组归约须使用补偿求和以保证精度。并行浮点归约允许精度容差内差异（见 §28.5）。浮点 `sum` 须默认使用补偿求和算法以保证数值稳定性。

**整数运算正确性**：整数归约溢出时须 panic（debug 和 release 模式一致），不得静默 wrapping。

**精度验证要求**：

| 属性 | 要求 |
|------|------|
| 边界值覆盖 | 0、±1、极大值、极小正规数、subnormal、±Inf、NaN、π 的倍数（三角函数）、整数幂（pow）；复数额外覆盖纯虚数、零虚部、零实部、极大模 |
| 回归测试 | 使用固定随机种子的 golden test |
| 跨平台一致性 | 同一输入在不同平台结果须在精度容差内一致 |

### 28.4 边界覆盖

须覆盖以下边界情况：
- 空张量、单元素、大张量
- 极端值（NaN/Inf/subnormal）
- 非连续布局
- 高维（≥4维）
- 广播形状不兼容（返回错误）
- ArcRepr 写时复制（引用计数不同时的行为差异）
- 小数组对齐降级
- 填充布局操作后的步长正确性
- IxDyn 与 IxN 互转（匹配/不匹配）
- no_std 构建验证

### 28.5 并行与 SIMD 测试

| 测试类别 | 要求 |
|----------|------|
| 数据竞争检测 | CI 中使用 ThreadSanitizer |
| 未定义行为检测 | CI 中使用 Miri 检测核心模块 |
| SIMD 正确性 | SIMD 路径结果须与标量路径在精度容差内一致 |
| 并行确定性 | 并行归约结果须与单线程一致（浮点允许精度容差内差异） |
| 跨平台 SIMD | 不同 SIMD 指令集上结果须一致 |
| 并行边界 | 元素数在并行阈值附近的边界情况 |

### 28.6 CI 要求

| 要求 | 说明 |
|------|------|
| 平台矩阵 | Linux (x86_64, aarch64)、macOS (x86_64, aarch64)、Windows (x86_64) |
| Rust 版本矩阵 | stable + MSRV + nightly |
| Feature 矩阵 | 默认 features、no-default-features、all-features |
| 必须通过 | cargo test（含 all-features）、cargo doc（无 broken link） |
| MSRV 验证 | 验证声明的 MSRV 准确 |
| 格式与 Lint | cargo fmt --check、cargo clippy 零 warning |
| 性能回归 | 提供基准测试，记录历史结果，人工审查异常波动 |

---

*Xenon v20 — 需求说明书*
