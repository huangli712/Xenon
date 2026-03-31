# Xenon 需求说明书 v20

> Rust 科学计算多维数组库
> 版本: v20 | 日期: 2026-03-31 | 许可: MIT | MSRV: Rust 1.85+

---

# 第一部分：项目定位

## 1. 项目概述

Xenon 是一个 Rust 多维数组（张量）库，定位为科学计算的数值基础设施，类似于 Python 生态中 NumPy 的角色。

### 1.1 目标用户

| 用户类型 | 核心诉求 |
|----------|----------|
| 库开发者 | 稳定 API、清晰抽象、零开销 |
| 系统开发者 | 底层控制、性能敏感、FFI 友好 |

### 1.2 核心原则

| 原则 | 说明 |
|------|------|
| 正确性优先 | 类型安全、内存安全、数值精度 |
| 清晰抽象 | API 语义明确，无隐式行为 |
| 科学计算导向 | 列优先默认、SIMD 友好、BLAS 兼容内存布局 |
| 最小依赖 | 仅 rayon（可选）和 pulp（可选） |

### 1.3 工程约束

| 约束 | 要求 |
|------|------|
| Crate 结构 | 单 crate，遵循 SemVer |
| no_std 支持 | 支持 `no_std`（需 `alloc`），`std` 为默认 feature|
| Feature gates | `std`（默认）、`parallel`（rayon，隐式启用 `std`，因 rayon 依赖线程）、`simd`（pulp） |

---

## 2. 项目范围

### 2.1 范围内

- N 维数组的存储、索引、切片、变形、逐元素运算、归约、广播
- 矩阵-向量乘法、向量内积/外积、批量运算
- 原始指针 API，供上游库零开销集成
- 自定义复数类型（FFI 友好，不依赖 num-complex）
- 临时工作空间（对齐分配，供上游库使用）

### 2.2 范围外

- GEMM（矩阵-矩阵乘法）、矩阵分解、对角化等高级线性代数
- FFT、稀疏矩阵、自动微分
- BLAS/LAPACK 绑定（由上游库通过指针API集成）
- GPU 后端（当前版本不实现，但须预留扩展性）
- serde 序列化、arena 分配器
- 栈分配小数组

---

# 第二部分：核心架构

## 3. 维度系统

### 3.1 静态维度与动态维度
支持 0 至 6 维静态维度 + 动态维度，静态维度与动态维度可互转。

| 类型 | 含义 | 内部表示 |
|------|------|----------|
| Ix0 | 0 维（标量） | 零长度数组 |
| Ix1 | 1 维（向量） | [usize; 1] |
| Ix2 | 2 维（矩阵） | [usize; 2] |
| Ix3 | 3 维 | [usize; 3] |
| Ix4 | 4 维 | [usize; 4] |
| Ix5 | 5 维 | [usize; 5] |
| Ix6 | 6 维 | [usize; 6] |
| IxDyn | 动态维度 | 堆分配 Vec<usize> |

### 3.2 Dimension trait 定义

所有维度类型须实现 `Dimension` trait：

| 关联类型/方法 | 签名 | 说明 |
|--------------|------|------|
| `type Stride` | 关联类型 | 步长容器类型（`[isize; N]` 或 `Vec<isize>`） |
| `type Pattern` | 关联类型 | 索引模式类型（如 `[usize; N]`、`Vec<usize>`） |
| `ndim()` | `&self -> usize` | 返回维度数 |
| `slice()` | `&self -> &[usize]` | 返回 shape 切片 |
| `slice_mut()` | `&mut self -> &mut [usize]` | 返回可变 shape 切片 |
| `size()` | `&self -> usize` | 返回总元素数（各轴乘积） |
| `default_strides()` | `&self -> Self::Stride` | 计算 F-order 默认步长 |
| `default_strides_c()` | `&self -> Self::Stride` | 计算 C-order 默认步长 |
| `type Reduced` | 关联类型 | 沿一轴归约后的维度类型（Ix1 → Ix0, Ix2 → Ix1, ..., IxDyn → IxDyn）|
| `remove_axis()` | `&self, axis: usize -> Self::Reduced` | 移除指定轴，返回降维后的维度类型 |

**Trait bound**：`Dimension: Debug + Clone + Eq + Hash + Send + Sync`

### 3.3 对应关系

| 维度类型 D | D::Stride | D::Pattern | D::Reduced |
|------------|-----------|------------|------------|
| Ix0 | `[isize; 0]` | `()` | —（不可归约） |
| Ix1 | `[isize; 1]` | `usize` | Ix0 |
| Ix2 | `[isize; 2]` | `(usize, usize)` | Ix1 |
| Ix3 | `[isize; 3]` | `(usize, usize, usize)` | Ix2 |
| Ix4~Ix6 | `[isize; N]` | `(usize, ..., usize)` | Ix(N-1) |
| IxDyn | `Vec<isize>` | `Vec<usize>` | IxDyn |

### 3.4 维度互转规则

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

### 3.5 特别设计

**Ix0 边界语义**

| 属性 | 值 |
|------|-----|
| shape | `&[]`（空切片） |
| ndim | 0 |
| 元素数 | 1 |
| 与 IxDyn 互转 | IxDyn.ndim() == 0 时可双向转换 |

**reshape 与维度互转**

- reshape 不改变维度类型（静态保持静态，动态保持动态）
- 显式转换维度类型使用 `into_dimension::<D>()` 方法
- 静态维度 reshape 时目标维度数须匹配类型（如 Ix3 reshape 后仍为 3 维）

---

## 4. 元素类型体系

### 4.1 类型体系

四层元素类型体系（上层继承下层，并额外提供能力）：

| 层次 | 类型 | Trait 约束 |
|------|------|------------|
| 基础层 | 整数、浮点、复数、bool | `Element` |
| 数值层 | 整数、浮点、复数（不含 bool） | `Numeric: Element`（在基础层之上支持四则运算） |
| 实数层 | f32, f64 | `RealScalar: Numeric`（在数值层之上提供数学函数） |
| 复数层 | Complex<f32>, Complex<f64> | `ComplexScalar: Numeric`（在数值层之上提供复数运算） |

> **约束**：bool 仅属于基础层，不实现 `Numeric`，不支持四则运算。不支持自动类型提升，类型转换须显式。

### 4.2 Sealed Trait 策略

四层 trait 全部 sealed，下游 crate 不可为自定义类型实现。

| 设计决策 | 说明 |
|----------|------|
| sealed 范围 | Element、Numeric、RealScalar、ComplexScalar 全部 sealed |
| 实现方式 | 私有模块 `mod private { pub trait Sealed {} }` + 各 trait 继承 `private::Sealed` |
| sealed 实现者 | 仅库内预定义类型：i32/i64/f32/f64/bool/Complex\<f32\>/Complex\<f64\> |

### 4.3 基础层（Element）

| 方法/常量 | 说明 |
|-----------|------|
| zero() | 加法单位元 |
| one() | 乘法单位元 |
| Copy + Clone | 值语义，可按位复制 |
| PartialEq | 相等比较 |
| Debug + Display | 格式化输出 |
| Send + Sync | 线程安全（用于并行迭代） |

### 4.4 数值层（Numeric）

继承 Element，额外约束四则运算。仅数值类型（整数、浮点、复数）实现，bool 不实现。

| 约束 | 说明 |
|------|------|
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

在数值层之上额外提供：

| 方法/常量 | 说明 |
|-----------|------|
| re(), im() | 实部、虚部访问 |
| conj() | 共轭 |
| norm() | 模（使用 hypot 避免溢出） |
| arg() | 辐角（返回 (-π, π]） |
| exp(), ln() | 复数指数/对数 |
| sqrt() | 复数平方根（主值） |
| from_polar(r, theta) | 极坐标构造 |
| i() | 虚数单位 |

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
- norm() 须使用 hypot 算法避免溢出

### 5.2 与实数的算术互操作

| 操作 | 行为 |
|------|------|
| Complex + 实数 | 实数隐式提升为 Complex(r, 0.0)，结果为 Complex。仅限同精度（Complex<f64> + f64） |
| 实数 + Complex | 同上，交换律成立 |
| Complex * 实数 | 标量乘法，等价于 (re*r, im*r)。仅限同精度 |
| Complex / 实数 | 标量除法，等价于 (re/r, im/r)。仅限同精度 |
| 实数 / Complex | 结果为 Complex，须正确处理分母共轭。仅限同精度 |
| Complex 与整数 | 不支持隐式互操作，须先将整数转为浮点 |
| 跨精度（如 f32 + Complex<f64>） | 不支持隐式互操作，须先显式 cast 到同精度 |

### 5.3 相等与比较

| 属性 | 要求 |
|------|------|
| PartialEq | 逐分量比较（re == re && im == im），NaN != NaN |
| Eq | 不实现（因 NaN） |
| PartialOrd | 不实现（复数无自然全序） |
| Ord | 不实现 |
| 近似相等 | 提供 approx_eq(other, epsilon) 方法，逐分量判断 |

### 5.4 内存与序列化

| 属性 | 要求 |
|------|------|
| 内存布局 | `#[repr(C)]`，保证 [re, im] 连续排列 |
| 大小 | size_of::<Complex<T>>() == 2 * size_of::<T>() |
| 对齐 | align_of::<Complex<T>>() == align_of::<T>() |
| 与 C 互操作 | 布局兼容 C99 \_Complex，可安全 transmute 指针 |
| 数组布局 | Complex<f64> 数组与交错实虚 f64 数组内存等价 |

---

## 6. 存储系统

### 6.1 Storage trait

存储须区分三种访问级别：**只读**、**可写**、**拥有**（编译时保证）。所有存储模式须实现统一的 `Storage` trait：

| 关联类型/方法 | 签名 | 说明 |
|--------------|------|------|
| `type Elem` | 关联类型 | 元素类型 |
| `type Device` | 关联类型：`Device` | 存储所在的计算设备（当前仅 `Cpu`） |
| `len()` | `&self -> usize` | 返回存储中的元素数（含 padding） |
| `as_ptr()` | `&self -> *const Self::Elem` | 底层数据指针（只读） |
| `as_mut_ptr()` | `&mut self -> *mut Self::Elem where Self::Elem: Sized` | 底层数据指针（可写）；只读存储类型不应实现此方法 |
| `is_owned()` | `&self -> bool` | 是否拥有数据（决定 drop 时是否释放内存） |

### 6.2 四种存储模式

| 存储模式 | 拥有数据 | 可读 | 可写 | 克隆语义 | 分配方式 | 典型用途 |
|----------|---------|------|------|----------|----------|----------|
| Owned | 是 | 是 | 是 | 深拷贝 | 64 字节对齐堆分配 | 数组创建、运算结果 |
| ViewRepr | 否（借用） | 是 | 否 | 拷贝视图元数据（O(1)） | 无分配 | 切片、子数组只读访问 |
| ViewMutRepr | 否（独占借用） | 是 | 是 | 不可克隆（独占语义） | 无分配 | 原地修改子区域 |
| ArcRepr | 共享（Arc） | 是 | 通过 make_mut() 写时复制 | 浅拷贝（引用计数+1） | 写时按需分配 | 跨线程共享、延迟复制、函数参数传递 |

**各存储模式的 Storage impl 特性**：

| 存储模式 | `len()` | `as_ptr()` | `as_mut_ptr()` | `is_owned()` | `Clone` |
|----------|---------|------------|----------------|--------------|---------|
| Owned | buffer 元素数（含 padding） | buffer 起始指针 | 可用 | true | 深拷贝（新分配） |
| ViewRepr | 同源 Tensor 的 len | 指向源数据 | 不可用（编译拒绝） | false | O(1) 拷贝元数据 |
| ViewMutRepr | 同源 Tensor 的 len | 指向源数据 | 可用 | false | 不可 Clone |
| ArcRepr | buffer 元素数（含 padding） | buffer 起始指针 | 通过 `make_mut()` 间接可用 | true（共享） | 浅拷贝（Arc ref +1） |

### 6.3 ArcRepr 语义

**ArcRepr::make_mut() 语义：**

| 属性 | 行为 |
|------|------|
| 写时复制 | 若引用计数 > 1，深拷贝数据到新分配，原 Arc 引用计数减 1 |
| 原子性保证 | 引用计数使用 `AtomicUsize`，所有引用计数操作使用 `Ordering::AcqRel`（fetch_sub）/ `Ordering::Acquire`（load）。make_mut() 先以 `Acquire` order 读取引用计数，若 > 1 则以 `AcqRel` order 递减并深拷贝。多线程并发 make_mut() 不会导致数据竞争或重复拷贝 |
| 返回值 | 返回 `&mut [A]`，独占访问保证无其他引用可读取数据 |
| 分配对齐 | 新分配使用默认 64 字节对齐 |
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

### 6.4 设备扩展性

Storage trait 预留 `type Device` 关联类型，当前版本仅支持 `Cpu`。

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
- 不可在 `Storage` trait 中添加 `Gpu` 相关的关联类型或方法（除 `type Device` 占位外）
- 不可在 `Element` trait 中添加 GPU 语义的约束

---

## 7. 内存布局

### 7.1 布局规则

| 规则 | 要求 |
|------|------|
| 布局顺序 | 支持 F-order（列优先）和 C-order（行优先），默认 F-order |
| 步长类型 | 有符号类型，单位为元素个数（非字节），支持负步长 |
| 默认对齐 | 自有存储默认 64 字节对齐（AVX-512 缓存行） |
| 可配置对齐 | 构造时可指定对齐值（须为 2 的幂，≥ 元素自然对齐） |
| 填充 | 支持主维度填充以保证 SIMD 对齐，填充区域须零初始化 |

### 7.2 布局标志系统

采用 5 标志位设计（1 字节 u8，低 5 位有效，高 3 位保留，须为 0），缓存高频查询的派生状态：

| 标志 | 位 | 信息 | 理由 |
|------|----|------|------|
| F_CONTIGUOUS | bit 0 | F-order 连续性 | 核心属性，高频查询 |
| C_CONTIGUOUS | bit 1 | C-order 连续性 | 核心属性，高频查询 |
| ALIGNED | bit 2 | SIMD 对齐（64 字节） | SIMD 路径选择依赖 |
| HAS_ZERO_STRIDE | bit 3 | 零步长（缓存） | 存在步长为 0 的维度。两种来源：(1) 广播产生的 size-1 维度（逻辑上重复该元素）；(2) unsqueeze 插入的 size-1 轴（§12.8，步长设为 0 以保持连续性）。优化路径据此判断不可按简单步长递增遍历，但连续性标志不受影响 |
| HAS_NEG_STRIDE | bit 4 | 负步长（缓存） | 反转检测，避免 O(ndim) 遍历 |
| — | bit 5-7 | 保留 | 须为 0，留作未来扩展 |

**组合语义：**
- `F_CONTIGUOUS | C_CONTIGUOUS`：标量或 1D 数组（两个方向都连续）
- `!F_CONTIGUOUS && !C_CONTIGUOUS`：非连续数组（切片/转置后常见）

**更新时机：**
- 创建时：根据分配方式初始化全部标志
- 切片/转置/reshape：重新计算全部标志
- 视图创建：继承源数组标志，按需降级（如对齐）

### 7.3 布局查询方法

| 方法 | 说明 | 复杂度 |
|------|------|--------|
| is_f_contiguous() | 是否 F-order 连续 | O(1) |
| is_c_contiguous() | 是否 C-order 连续 | O(1) |
| is_contiguous() | 任一方向连续即为 true | O(1) |
| is_aligned() | 是否 SIMD 对齐（64 字节） | O(1) |
| has_zero_stride() | 是否存在零步长（广播维度） | O(1) |
| has_neg_stride() | 是否存在负步长（反转维度） | O(1) |
| layout_flags() | 返回完整布局标志 | O(1) |

### 7.4 对齐策略

| 属性 | 要求 |
|------|------|
| 默认对齐 | 64 字节（AVX-512 缓存行） |
| 小数组优化 | 当 `元素数 × size_of::<A>() ≤ 64` 时，允许降级到 `max(align_of::<A>(), 8)` 字节对齐（即元素自然对齐，但至少 8 字节以保证基本 SIMD 安全）。阈值 64 与默认对齐一致，避免小数组浪费内存 |
| 对齐查询 | 提供 alignment() 方法返回当前数组的实际对齐值 |
| 视图对齐 | 视图继承源数组的对齐状态；切片后对齐可能降级（起始地址偏移），布局标志须反映实际对齐 |
| ArcRepr 对齐 | make_mut() 触发复制时，新分配使用默认对齐（64 字节） |

---

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
| strides: D::Stride | 各轴步长（有符号 `isize`，单位为元素个数）。每个维度类型 `D` 关联一个步长类型 `D::Stride`（如 Ix2 → `[isize; 2]`，IxDyn → `Vec<isize>`），与 shape 的 `[usize; N]` 表示区分 |
| offset: usize | 数据起始偏移量（单位为元素个数，非字节），始终 ≥ 0。与 strides 的有符号类型 `isize` 配合使用：元素地址计算为 `base_ptr + offset + Σ(index[k] * stride[k])`，其中 offset 为无偏移基准点，strides 可为负值（反转轴）。offset 使用 `usize` 而非 `isize` 是因为起始偏移在物理上不可能为负（数据存储从缓冲区头部开始），而 strides 需要 `isize` 以表达反转视图的负步长 |

**偏移量算术安全性**：元素地址计算 `offset + Σ(index[k] * stride[k])` 须始终落在 `[0, storage_capacity)` 范围内。实现须保证：(1) `index[k] * stride[k]` 的中间结果使用 `isize`（不会溢出，因为 index ≤ shape[k] ≤ usize::MAX / 2，stride 的绝对值 ≤ shape 的乘积，实际场景远小于 isize::MAX）；(2) 最终偏移量 `offset + Σ(...)` 转为 `usize` 后须 < 存储容量，越界访问在 safe API 中返回错误或 panic（如 `IndexOutOfBounds`），在 `unsafe` API（如 `get_unchecked`）中为调用方责任；(3) 构造函数（包括 `from_raw_parts`）须验证 shape × strides 不产生越界偏移。

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
| TensorD<A> | Tensor<A, IxDyn> | 动态维度 |

### 8.3 基础查询方法

所有 TensorBase<S, D> 均提供以下查询方法（不区分存储模式）：

| 方法 | 签名 | 说明 |
|------|------|------|
| `len()` | `&self -> usize` | 返回总元素数（各轴长度乘积），等价于 `shape().iter().product()` |
| `is_empty()` | `&self -> bool` | 等价于 `len() == 0` |
| `ndim()` | `&self -> usize` | 返回维度数，等价于 `shape().len()` |
| `shape()` | `&self -> &[usize]` | 返回各轴长度的切片 |
| `strides()` | `&self -> &[isize]` | 返回各轴步长的切片（元素单位，有符号）。内部存储为 `D::Stride` 类型（如 `[isize; N]` 或 `Vec<isize>`），返回时统一借用为 `&[isize]` 切片 |
| `offset()` | `&self -> usize` | 返回数据起始偏移量（元素单位） |

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

| 指令集 | 架构 | 寄存器宽度 | 优先级 |
|--------|------|-----------|--------|
| AVX-512 | x86_64 | 512 bit | 最高 |
| AVX2 + FMA | x86_64 | 256 bit | 高 |
| SSE4.2（含 SSE4.1、SSSE3、POPCNT） | x86_64 | 128 bit | 中 |
| NEON | aarch64 | 128 bit | 高（ARM 平台） |
| 标量回退 | 所有 | - | 最低 |

#### 9.1.3 SIMD 适用操作

| 操作类别 | 示例 | SIMD 要求 |
|----------|------|-----------|
| 逐元素一元 | abs, neg, sqrt, exp, ln | 连续内存（步长=1）；步长≠1 回退标量路径 |
| 逐元素二元 | add, sub, mul, div | 两操作数均连续内存（步长=1）且形状兼容；任一操作数步长≠1 回退标量路径 |
| 归约 | sum, prod, min, max | 连续内存（步长=1）；步长≠1 回退标量路径 |
| 内积 | dot | 连续内存（步长=1） |
| 比较 | eq, lt, gt（生成 mask） | 连续内存（步长=1） |

#### 9.1.4 SIMD 对齐与回退

| 属性 | 要求 |
|------|------|
| 对齐要求 | 数据起始地址 64 字节对齐时使用对齐加载，否则使用非对齐加载 |
| 非连续数据 | 步长 ≠ 1 时回退到标量路径 |
| 尾部处理 | 元素数非 SIMD 宽度整数倍时，尾部使用标量处理 |
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
| 并行阈值 | 默认元素数 ≥ 64K 时启用并行，可通过全局配置或单次调用覆盖 |
| 支持并行的操作 | 逐元素运算、归约、map/mapv、zip 迭代 |
| 不支持并行的操作 | 矩阵乘法（由 BLAS 内部管理线程）、单元素操作、小数组操作 |
| 分块策略 | 按连续内存块分割，每块不小于 4K 元素 |
| 嵌套并行 | 禁止嵌套并行（内层操作强制单线程），避免线程池饥饿 |
| 线程数 | 默认使用 rayon 全局线程池，可通过自定义线程池覆盖 |
| 浮点归约算法 | 并行浮点归约（sum/mean/var/std）须使用补偿求和算法（Neumaier 或 Pairwise），以保证不同分块策略下结果的精度上界可控。单线程浮点归约同样使用补偿求和。整数归约不适用（使用 checked 算术） |

### 9.3 性能分层

运行时根据数据规模和内存布局自动选择执行路径：

| 条件 | 执行路径 |
|------|----------|
| 元素数 < SIMD 宽度 | 标量路径 |
| 元素数 ≥ SIMD 宽度 且 连续内存 且 simd 启用 | SIMD 路径 |
| 元素数 ≥ 并行阈值 且 parallel 启用 且 连续内存 且 simd 启用 | SIMD + 并行路径（每线程内部使用 SIMD） |
| 元素数 ≥ 并行阈值 且 parallel 启用 且 非连续内存（步长 ≠ 1） | 标量 + 并行路径（每线程使用标量迭代，按块分割非连续数据。连续内存不是并行路径的前提条件——非连续数据也可并行，只是每线程内部不走 SIMD） |
| simd 未启用 且 元素数 ≥ 并行阈值 且 parallel 启用 | 标量 + 并行路径 |
| 非连续内存 且 元素数 < 并行阈值（或 parallel 未启用） | 标量路径（单线程，步长跳跃遍历） |

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
| 禁止访问 | Padding 字节不属于任何逻辑元素，任何线程不得读写 padding 字节（初始化时除外） |
| 禁止分配 | 并行迭代不得将 padding 字节分配给任何线程的工作区间 |
| 禁止暴露 | 视图切片不得暴露 padding 字节为可访问元素 |

---

## 11. 迭代器

### 11.1 迭代类型

| 类型 | 说明 |
|------|------|
| 元素迭代 | 按内存布局顺序 |
| 轴迭代 | 沿指定轴 |
| 窗口迭代 | 滑动窗口 |
| 索引迭代 | 带索引的元素迭代 |
| 并行迭代 | rayon 支持 |
| 多数组同步迭代 | zip |
| 遍历顺序 | 可指定 F/C |

### 11.2 迭代器协议

| 场景 | 行为 |
|------|------|
| zip 形状完全一致 | 按指定遍历顺序同步产出元素 |
| zip 形状不一致但可广播 | 自动广播后迭代，等价于先 broadcast 再 zip。广播后的迭代顺序由广播结果的逻辑布局决定（若未指定遍历顺序，按物理内存布局顺序；广播视图中，步长为 0 的轴在迭代时重复产出同一位置的数据） |
| zip 形状不一致且不可广播 | 返回 BroadcastError |
| 遍历顺序未指定 | 默认按物理内存布局顺序（连续数组最优） |
| 遍历顺序指定为 F/C | 强制按指定逻辑顺序，可能非连续访问 |
| 并行迭代 | 将元素按连续块分割给线程，块大小 ≥ 并行阈值 |
| 窗口迭代越界 | 不产出不完整窗口（窗口数 = shape - window_size + 1） |
| 空数组迭代 | 立即结束，产出零个元素 |

### 11.3 广播视图的可变迭代安全性

广播通过零步长（stride=0）模拟维度扩展。当对广播后的可变视图调用 `iter_mut()` / `par_iter_mut()` 时，多个逻辑迭代位置会映射到同一物理元素，构成别名写入。

**设计决策：禁止对广播视图进行可变迭代**

| 场景 | 行为 | 理由 |
|------|------|------|
| `TensorViewMut` 存在零步长轴时调用 `iter_mut()` | panic（运行时检查） | 多个 `&mut` 指向同一元素违反 Rust 别名规则，属于未定义行为 |
| `TensorViewMut` 存在零步长轴时调用 `par_iter_mut()` | panic（运行时检查） | 同上，且并行场景构成数据竞争 |
| `mapv_inplace` / `zip_mut` 等内部可变操作遇到广播目标 | panic（运行时检查） | 写入广播目标会导致同一位置被多次写入，结果依赖迭代顺序，语义不明确 |
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
| `Iterator` | 所有迭代器 | 必须实现 `Item`、`next()` |
| `IntoIterator` | Tensor, TensorView, TensorViewMut, ArcTensor | `into_iter()` 消耗数组产出元素；`&Tensor` 借用迭代产出引用 |
| `ExactSizeIterator` | 元素迭代、轴迭代、索引迭代 | 提供 `len()`，长度在迭代前已知 |
| `DoubleEndedIterator` | `ContiguousIter`（仅连续数组的迭代器） | 支持 `next_back()`，从两端遍历。Rust 无法基于运行时状态条件实现 trait，因此采用类型分层：`iter()` 返回通用 `Iter`（不实现 `DoubleEndedIterator`）；连续数组额外提供 `iter_contiguous() -> Option<ContiguousIter>`（仅连续时返回 `Some`），`ContiguousIter` 实现 `DoubleEndedIterator` + `ExactSizeIterator`。非连续数组调用 `iter_contiguous()` 返回 `None` |
| `FusedIterator` | 所有迭代器 | 迭代结束后 `next()` 永远返回 `None` |
| `IntoParallelIterator` | Tensor, TensorView, ArcTensor（feature `parallel`） | 需要 `A: Send`；按连续块分割给线程 |
| `ParallelIterator` | 并行迭代器（feature `parallel`） | rayon `ParallelIterator`，提供 `par_iter()`、`par_iter_mut()` |

---

# 第四部分：API 规范

## 12. 逐元素运算

### 12.1 基本类型

| 类别 | 操作 |
|------|------|
| 算术 | add, sub, mul, div |
| 三角函数 | sin, cos, tan, asin, acos, atan |
| 指数/对数 | exp, ln, log2, log10 |
| 数值函数 | abs, sign, floor, ceil, round, square, reciprocal, pow |
| 比较 | eq, ne, lt, le, gt, ge |

### 12.2 比较操作语义

| 操作 | 方法 | 返回类型 | 说明 |
|------|------|----------|------|
| 等于 | `eq(&self, other) -> Tensor<bool, D>` | bool Tensor | 逐元素比较，与 `PartialEq` 一致；NaN != NaN |
| 不等于 | `ne(&self, other) -> Tensor<bool, D>` | bool Tensor | 逐元素比较，NaN != NaN 为 true |
| 小于 | `lt(&self, other) -> Tensor<bool, D>` | bool Tensor | 逐元素比较，要求 `Element: PartialOrd` |
| 小于等于 | `le(&self, other) -> Tensor<bool, D>` | bool Tensor | 逐元素比较，要求 `Element: PartialOrd` |
| 大于 | `gt(&self, other) -> Tensor<bool, D>` | bool Tensor | 逐元素比较，要求 `Element: PartialOrd` |
| 大于等于 | `ge(&self, other) -> Tensor<bool, D>` | bool Tensor | 逐元素比较，要求 `Element: PartialOrd` |

### 12.3 约束

**约束 1**：逐元素算术运算（add, sub, mul, div, 三角函数, 指数/对数, 数值函数）仅适用于数值类型（整数、浮点、复数），不适用于 bool。bool 类型仅支持逻辑运算（`all/any`）和比较运算（eq, ne）。bool **不支持**位运算（`& | ^ !`）的运算符重载——需要位运算时，应先 `.map(|b| b as u8)` 转为整数类型后操作。

**约束 2**：`eq`/`ne` 仅要求 `PartialEq`；`lt`/`le`/`gt`/`ge` 要求 `PartialOrd`（浮点类型因 NaN 为偏序）。所有比较方法仅通过方法调用提供，不提供 `< <= > >=` 运算符语法（见 §14.2 设计决策）。支持广播。返回的 bool Tensor 可用于 `select()`、`mask()`、`compress()` 等条件操作。

---

## 13. 矩阵运算

### 13.1 支持的操作

| 类别 | 操作 |
|------|------|
| 基本运算 | matvec、dot/inner、outer |
| 批量运算 | batch_matvec、batch_dot、batch_add、batch_scale |

### 13.2 矩阵运算语义

**matvec 语义**：

| 属性 | 行为 |
|------|------|
| 签名 | `fn matvec(&self, other: &Tensor<A, Ix1>) -> Tensor<A, Ix1> where A: Numeric` |
| 操作 | 矩阵-向量乘法：self(M, N) × other(N) → 结果(M) |
| 约束 | self 须为 2D，other 须为 1D 且长度等于 self 的第 1 轴（N），否则返回 `ShapeMismatch` |

**dot/inner 语义**：

| 属性 | 行为 |
|------|------|
| 签名 | `fn dot(&self, other: &Tensor<A, Ix1>) -> A where A: Numeric` |
| 操作 | 向量内积：self(N) · other(N) → 标量 |
| 约束 | self 和 other 均须为 1D 且长度相等，否则返回 `ShapeMismatch` |

**outer 语义**：

| 属性 | 行为 |
|------|------|
| 签名 | `fn outer(&self, other: &Tensor<A, Ix1>) -> Tensor<A, Ix2> where A: Numeric` |
| 操作 | 向量外积：self(M) ⊗ other(N) → 结果(M, N)，其中 result[i, j] = self[i] * other[j] |
| 约束 | self 和 other 均须为 1D，否则返回 `InvalidShape` |
| 输出布局 | F-contiguous（默认布局，见 §7.1） |

### 13.3 批量运算维度约定

| 约定 | 规则 |
|------|------|
| batch 轴位置 | 最前面的轴为 batch 轴（轴 0, 1, ..., ndim-3 为 batch，最后 2 维为矩阵/向量） |
| batch 形状约束 | 所有操作数的 batch 维度须形状一致或可广播 |
| batch 广播 | batch 维度遵循 NumPy 广播规则，结果 batch 形状为广播后形状 |
| batch_matvec | 输入：(..., M, N) × (..., N) → 输出：(..., M) |
| batch_dot | 输入：(..., N) × (..., N) → 输出：(...) |
| batch_add/scale | 输入形状须一致或可广播，逐元素操作 |

### 13.4 批量运算签名与错误处理

| 运算 | 签名 | 输入约束 | 错误 |
|------|------|----------|------|
| batch_matvec | `fn batch_matvec(&self, other: &Tensor<A, D>) -> Tensor<A, D> where A: Numeric` | self(..., M, N) × other(..., N) → (..., M)；batch 维度须一致或可广播 | batch 维度不可广播 → `BroadcastError`；最后 2 维形状不匹配 → `ShapeMismatch` |
| batch_dot | `fn batch_dot(&self, other: &Tensor<A, D>) -> Tensor<A, D::Reduced> where A: Numeric, D: Dimension` | self(..., N) × other(..., N) → (...)；batch 维度须一致或可广播 | batch 维度不可广播 → `BroadcastError`；向量长度不匹配 → `ShapeMismatch` |
| batch_add | `fn batch_add(&self, other: &Tensor<A, D>) -> Tensor<A, D> where A: Numeric` | 输入形状须一致或可广播，逐元素加法 | 不可广播 → `BroadcastError` |
| batch_scale | `fn batch_scale(&self, scalar: A) -> Tensor<A, D> where A: Numeric` | 任意形状，逐元素乘以标量 | 无错误 |

**非连续输入行为**：以上批量运算对非连续输入自动拷贝为 F-contiguous 后执行。不返回布局错误——调用方若需零拷贝，应先确保输入连续。

**batch_dot 维度推导**：`batch_dot` 沿最后一轴做内积后消除该轴，输出 ndim = 输入 ndim - 1。签名使用 `D::Reduced` 关联类型表达降维后的维度（见 §3.2 Dimension trait）。静态维度推导规则：Ix1 → Ix0, Ix2 → Ix1, ..., IxDyn → IxDyn。Ix0 输入编译错误（无轴可归约）。

**与 `dot` 返回类型差异**：`dot` 返回 `A`（标量值），`batch_dot` 返回 `Tensor<A, D::Reduced>`（标量张量）。两者定位不同：`dot` 是 1D×1D → 标量的便捷方法；`batch_dot` 支持任意批次维度，统一返回 Tensor 以保持泛型一致性（`D::Reduced` 推导需 Tensor 包装）。对 Ix1 输入，`batch_dot` 返回 `Tensor0<A>`（0D 标量张量），可调用 `.into_scalar()` 获取裸值。

---

## 14. 归约操作

### 14.1 归约类型

| 类型 | 操作 |
|------|------|
| 全局 | sum, prod, mean, var, std, min, max, argmin, argmax, all, any |
| 沿轴 | 以上所有运算均支持沿指定轴归约 |
| 累积 | cumsum, cumprod（沿指定轴，返回同形状数组） |

### 14.2 沿轴归约输出维度规则

| 属性 | 行为 |
|------|------|
| 默认行为 | 归约轴消除，输出 ndim = 输入 ndim - 1。例如 Ix3 沿 axis=1 归约 → Ix2 |
| keepdims 参数 | 提供 `sum_axis_keepdim(axis)` 变体，归约轴保留为 size-1，输出 ndim 不变 |
| 静态维度推导 | 归约后维度类型由编译时推导：Ix(N+1) 沿任意轴归约 → IxN。Ix0 不可执行沿轴归约（无轴可归约） |
| 动态维度 | IxDyn 沿轴归约后仍为 IxDyn（ndim 减少 1） |
| argmin/argmax | 沿轴归约时，返回对应轴上的索引数组（usize 类型），输出维度规则同上 |
| 全局归约 | 不指定轴时，所有轴归约为单个标量（返回 `A`，不是 Tensor） |

### 14.3 约束

**var/std 统计定义**：默认有偏估计（除以 N，即 ddof=0），与 NumPy 默认行为一致。提供 `var_ddof(ddof: usize)` / `std_ddof(ddof: usize)` 变体以支持无偏估计（ddof=1，除以 N-1）等自定义自由度。ddof 须 < 元素数，否则 panic（单元素数组 ddof=1 无意义）。

**mean/var/std 类型约束**：`mean()`、`var()`、`std()` 仅对浮点类型（`f32`/`f64`）实现。整数数组调用这些方法将产生编译错误。实现机制：这些方法的 `where A: RealScalar` 约束（见 §3.3）在编译时排除了整数类型（整数仅实现 `Numeric`，不实现 `RealScalar`）。用户需先手动转换类型，例如 `.mapv(|x| x as f64).mean()`。理由：Xenon 不做隐式类型提升，整数除法截断会产生误导性结果。

**整数归约溢出行为**：`sum/prod` 作用于整数数组时，溢出将 panic（debug 和 release 模式均如此）。实现须使用 `checked_add`/`checked_mul` 显式检测溢出，因为 Rust release 模式默认为 wrapping 算术（不 panic），仅 debug 模式默认 panic。

**cumsum/cumprod 边界行为**：沿指定轴正方向（index 0 → N-1）累积，与 NumPy `np.cumsum(axis=...)` 行为一致。遇到 NaN 时传播 NaN（后续元素均为 NaN）；空数组返回同形状空数组。

**cumsum/cumprod 整数溢出行为**：`cumsum`/`cumprod` 作用于整数数组时，每步累加/累乘均检测溢出，溢出时 panic（debug 和 release 模式均如此）。实现须使用 `checked_add`/`checked_mul` 显式检测（理由同 `sum/prod`：Rust release 默认 wrapping 算术不会 panic）。若需 wrapping 行为，用户应先 `cast` 为更大的整数类型再执行累积操作。

**cumsum/cumprod 溢出检查性能影响**：逐元素 checked 溃出检测相比 wrapping 累积有额外分支开销。对于 `cumsum`（加法），现代 CPU 的分支预测器对"几乎不溢出"的场景预测准确率高，实际性能损失通常 < 5%。对于 `cumprod`（乘法），若元素绝对值普遍 > 1，溢出更早触发分支误预测。若性能敏感且用户能保证不溢出，可先 `cast` 为更大类型后累积再 `cast` 回来以规避检查开销。

**argmin/argmax 多值行为**：存在多个相同最小/最大值时，返回第一个出现的索引（按遍历顺序），与 NumPy/ndarray 行为一致。

---

## 15. 集合操作

### 15.1 集合操作类型

| 操作 | 说明 | 输入类型约束 | 返回类型 |
|------|------|--------------|----------|
| unique | 返回唯一值（排序后） | Element + Ord（浮点使用 total_cmp；Complex 使用实部/虚部字典序 total_cmp） | Tensor1<A> |
| unique_counts | 返回唯一值及出现次数 | Element + Ord（同 unique） | (Tensor1<A>, Tensor1<usize>) |
| unique_inverse | 返回唯一值及原数组索引 | Element + Ord（同 unique） | (Tensor1<A>, Tensor1<usize>) |
| bincount | 统计非负整数出现次数 | 整数类型（i32/i64/u8/u16/u32/u64，值须 ≥ 0；无符号类型天然满足非负约束，推荐优先使用） | Tensor1<usize> |
| histogram | 统计落入各 bin 的元素数 | RealScalar | Tensor1<usize> |
| histogram_bin_edges | 返回 bin 边界 | RealScalar | Tensor1<A> |

### 15.2 unique 语义

| 属性 | 行为 |
|------|------|
| 排序 | 返回的唯一值按升序排列 |
| 空数组 | 返回空 Tensor1 |
| 返回值 | Tensor1<A>，长度等于唯一值数量 |
| 浮点比较 | 浮点类型使用 `total_cmp`（IEEE 754 totalOrder）进行排序和去重 |
| Complex 排序 | Complex 类型使用字典序排序：先比较实部（re），实部相等时比较虚部（im）。排序使用实部和虚部的 `total_cmp`。相等判定为 `re == re && im == im`。例：`[1+2i, 1+1i, 2+0i].unique()` → `[1+1i, 1+2i, 2+0i]` |
| NaN 处理 | 所有 NaN 视为相等，合并为一个，排在末尾（大于 +∞）。例：`[1.0, NaN, 2.0, NaN].unique()` → `[1.0, 2.0, NaN]` |
| ±0.0 处理 | `+0.0` 与 `-0.0` 在 IEEE 754 下 `==` 为 true，在 `total_cmp` 排序时 `-0.0 < +0.0`。unique 使用 `total_cmp` 排序后通过 `!=` 去重：因 `-0.0 == +0.0`（`!=` 为 false），二者合并为一个值。合并结果为排序后的首个出现，即 `-0.0`。例：`[+0.0, -0.0, 1.0].unique()` → `[-0.0, 1.0]` |

### 15.3 unique_counts 语义

| 属性 | 行为 |
|------|------|
| 返回值 | (values, counts)，values 为唯一值（升序），counts 为对应出现次数 |
| 约束 | values.len() == counts.len() |
| NaN 处理 | 同 unique：所有 NaN 合并为一个，计入同一 count |

### 15.4 unique_inverse 语义

| 属性 | 行为 |
|------|------|
| 返回值 | (values, inverse)，values 为唯一值（升序），inverse 为原数组每个元素在 values 中的索引 |
| 约束 | inverse.len() == input.len()，inverse[i] ∈ [0, values.len()) |
| 重建原数组 | values[inverse[i]] == input[i] |
| NaN 处理 | 同 unique：所有 NaN 映射到同一索引 |

### 15.5 bincount 语义

| 属性 | 行为 |
|------|------|
| 输入约束 | 仅支持整数类型，所有值须 ≥ 0 |
| minlength 参数 | 指定输出最小长度，若最大值+1 < minlength，则输出长度为 minlength |
| 空数组 | 返回长度为 minlength（默认 0）的全零数组 |
| 负值输入 | panic（运行时检查） |
| 返回值 | Tensor1<usize>，长度为 max(input) + 1（或 minlength），output[i] = count of i in input |
| 权重参数 | 可选 weights: Tensor1<W> where W: Numeric，权重元素类型 W 独立于输入整数类型。带权重时返回 Tensor1<W>（而非 Tensor1<usize>），output[i] = sum of weights[j] where input[j] == i。weights 长度须与 input 长度相等，否则 panic |

### 15.6 histogram 语义

| 属性 | 行为 |
|------|------|
| bins 参数 | 整数（等宽 bin 数）或 Tensor1<A>（自定义 bin 边界） |
| range 参数 | (min, max) 元组，指定统计范围，超出范围的值不计入。**与 bins 的交互**：当 bins 为整数时，range 指定等宽分割的上下界（默认使用数据的最小/最大值）；当 bins 为自定义边界数组时，range 参数被忽略（边界已由数组显式指定），若调用方同时提供 bins 数组和 range，返回 `InvalidShape` 错误 |
| 空数组 | 返回全零数组 |
| 返回值 | Tensor1<usize>，长度等于 bin 数 |
| NaN 输入 | NaN 不计入任何 bin，且不触发 panic。行为分两种情况：(1) **range 已指定**（调用方提供 `(min, max)` 或 bins 为自定义边界数组时）：NaN 被静默跳过，返回的计数仅基于非 NaN 元素。(2) **range 未指定**（bins 为整数且未提供 range 参数）：需从数据推断范围，此时若输入含 NaN，`min()`/`max()` 将返回 NaN，无法确定有效的 bin 边界，函数返回 `InvalidInput` 错误（错误信息："autodetected range is not finite: input contains NaN and no explicit range provided"）。调用方若需在含 NaN 数据上使用 histogram，应显式提供 range 参数 |
| bin 边界规则 | 设 bin 边界数组为 `[e0, e1, ..., ek]`（共 k+1 个边界，k 个 bin），则第 i 个 bin（0 ≤ i < k-1）的区间为 `[ei, ei+1)`（左闭右开），最后一个 bin（第 k-1 个）的区间为 `[ek-1, ek]`（左闭右闭）。与 NumPy `np.histogram` 行为一致 |

---

## 16. 广播

### 16.1 基本规则

- 遵循 NumPy 广播规则（右对齐，size-1 维度拉伸）
- 广播视图为只读
- 算术运算符隐式支持广播

### 16.2 广播细节

| 规则 | 说明 |
|------|------|
| 维度对齐 | 从最右维度开始对齐，维度数不足的数组在左侧补 1 |
| 兼容条件 | 对应维度相等，或其中一个为 1 |
| 零步长语义 | size-1 维度广播时步长设为 0，逻辑上重复该维度的单个元素 |
| 广播视图可写性 | 广播产生的视图始终为只读（设计约束：广播视图使用零步长实现逻辑重复，写入语义不明确——无法确定应写入原始元素还是所有广播位置；因此禁止写入，而非技术限制） |
| 原地广播运算 | `a += b` 中 b 可被广播，a 不可被广播（a 须拥有完整存储） |
| 标量广播 | 标量视为 0 维数组，可与任意维度数组广播 |

---

## 17. 形状操作

**生命周期约定**：本节所有零拷贝操作返回的 `TensorView` / `TensorViewMut` 的生命周期绑定到 `&self`（不可变）或 `&mut self`（可变）。签名中省略生命周期标注 `'_` 以减少噪音，完整签名为 `TensorView<'a, A, D>` 其中 `'a` 与 `&'a self` 一致。

### 17.1 操作分类

| 类型 | 操作 |
|------|------|
| 零拷贝 | reshape, transpose, slice, squeeze, unsqueeze, permute, broadcast, swapaxes, moveaxis, split/chunk, index_axis, unstack |
| 需拷贝 | cat, stack, pad, repeat/tile |
| 视情况 | flatten |

### 17.2 reshape 语义

| 属性 | 行为 |
|------|------|
| 操作 | `reshape(shape) -> Result<Tensor<A, D>>` 改变数组形状，元素总数须不变。形状合法时始终成功（连续零拷贝，非连续自动拷贝）；仅当 `-1` 推导失败时返回 `InvalidShape` |
| 连续输入 | 仅修改 shape/strides 元数据，复用底层存储（零拷贝）。返回的新 Tensor **独占**该存储（Owned 语义），原 Tensor 在 reshape 后不再可用（移动语义）。"共享底层缓冲区"仅发生在视图层面（如原始数据被 ArcRepr 持有），Owned-to-Owned reshape 通过移动所有权实现零拷贝 |
| 非连续输入 | 自动拷贝为 F-contiguous 布局后再 reshape，返回新分配的 Tensor |
| 输出布局 | 连续路径：继承输入的连续性方向（F-contiguous → F-order strides；C-contiguous → C-order strides）；非连续路径：输出为 F-contiguous |
| -1 自动推导 | shape 中至多一个维度可为 `-1`，由元素总数除以其余维度乘积自动推导；多个 `-1`、推导结果非整数、或元素总数不匹配时返回 `Err(InvalidShape)`。**API 签名**：静态维度接受 `&[isize]` 切片（`-1isize` 表示自动推导，合法正值自动转为 `usize`）；IxDyn 接受 `&IxDyn` 或 `Vec<isize>`（内部同样用 `-1isize` 标记推导维度） |
| 空数组 | 允许 reshape 为任意元素数为 0 的形状（如 `(0, 5)` → `(0, 3, 2)` 或 `(0,)`） |
| 维度类型 | 不改变维度类型（静态保持静态，动态保持动态）；显式转换维度类型使用 `into_dimension::<D>()` |
| 静态维度约束 | 静态维度（如 Ix3）reshape 后维度数须匹配类型（Ix3 reshape 后仍为 3 维）。即 `Tensor<A, Ix3>.reshape(&[3])` 是编译错误（3 维 → 1 维），须先 `into_dimension::<Ix1>()` 转换维度类型。IxDyn 无此约束（维度数可任意变化） |

> **设计决策**：统一为单一 `reshape()` 方法，返回 `Result`：形状合法时始终成功（连续零拷贝、非连续自动拷贝），仅 `-1` 推导失败时返回错误。移除原 `reshape_owned()` 和双 API 设计。理由：(1) 消除用户在运行时根据内存布局选择 API 的认知负担，与 §1.2 "清晰抽象，无隐式行为" 一致；(2) 连续输入自动零拷贝、非连续输入自动拷贝，行为符合最小惊讶原则；(3) `-1` 推导失败是合法的运行时错误场景（shape 可能来自用户输入），用 Result 而非 panic 返回。

> **关于"无隐式行为"原则与 reshape 自动拷贝的一致性**：reshape 对非连续输入的自动拷贝**不违反** §1.2 "无隐式行为"原则。"无隐式行为"指 API 不应有隐藏的语义副作用（如静默修改输入数据、隐式全局状态变更），而非禁止内部内存分配——所有构造操作（`zeros`/`ones`/`map` 等）都需要分配内存，这是构造语义的固有部分。reshape 的语义是"返回指定形状的新 Tensor"，其分配行为与其他构造方法一致，且通过 `is_contiguous()` 预检和性能提示文档已充分告知调用方。相比之下，若 reshape 对非连续输入静默修改源数组的数据（而非复制），才是真正的"隐式行为"。

> **性能提示**：`reshape()` 对非连续输入会隐式执行完整数据拷贝。性能敏感场景下，调用方可通过 `is_contiguous()` 预检：若返回 `true`，后续 `reshape()` 保证零拷贝；若返回 `false`，调用方可决定是否先调用 `as_f_contiguous()` / `as_c_contiguous()`（见 §14.4）获取零拷贝视图再操作，或接受拷贝开销。

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
| 操作 | `squeeze(axis: usize) -> TensorView<A, D'>` 消除指定轴（该轴长度须为 1） |
| 零拷贝 | 始终零拷贝 |
| 输出维度 | ndim - 1，shape 和 strides 移除该轴 |
| 约束 | 指定轴长度须为 1，否则返回 `InvalidShape` 错误 |
| 多轴挤压 | `squeeze_axes(axes: &[usize])` 同时消除多个 size-1 轴 |
| 0D/1D | Ix0 不可 squeeze；Ix1 squeeze 指定唯一轴后得到 Ix0 |
| 动态维度 | IxDyn 支持 squeeze，输出仍为 IxDyn（ndim 减少） |

### 17.8 unsqueeze 语义

| 属性 | 行为 |
|------|------|
| 操作 | `unsqueeze(axis: usize) -> TensorView<A, D'>` 在指定位置插入长度为 1 的新轴 |
| 零拷贝 | 始终零拷贝，仅扩展 shape（插入 1）和 strides（插入适当值） |
| 输出维度 | ndim + 1 |
| 新轴步长 | 设为 0（与广播语义一致：size-1 维度的步长为 0，表示逻辑上重复该单个元素）。插入后数组的连续性不变（F-contiguous 输入仍 F-contiguous，C-contiguous 输入仍 C-contiguous） |
| axis 范围 | `0 <= axis <= ndim`（可在最后一轴之后插入） |
| 静态维度约束 | Ix6 不可 unsqueeze（已达最大静态维度）；IxDyn 无此限制 |

### 17.9 cat 语义

| 属性 | 行为 |
|------|------|
| 操作 | `cat(axis, &[TensorView], order: Option<Order>)` 沿指定轴拼接多个数组，返回新 Owned 数组（需拷贝）。`order` 为 `None` 时默认 F-contiguous，`Some(Order::C)` 时输出 C-contiguous |
| 维度约束 | 所有输入数组的 ndim 须相同；除拼接轴外，其余各轴长度须完全一致 |
| 拼接轴长度 | 结果在拼接轴上的长度等于所有输入在该轴上长度之和 |
| 输入数量 | 至少 1 个输入；0 个输入返回 `InvalidShape` 错误 |
| 空数组输入 | 允许部分输入在拼接轴上长度为 0，等价于跳过该输入 |
| 元素顺序 | 按输入列表顺序排列，第 i 个输入的元素在结果中排在第 i-1 个之后 |
| 输出布局 | 默认 F-contiguous；可通过参数指定 C-contiguous |
| 返回类型 | `Tensor<A, D>`（Owned） |

### 17.10 stack 语义

| 属性 | 行为 |
|------|------|
| 操作 | `stack(axis, &[TensorView], order: Option<Order>)` 沿新轴堆叠多个数组，返回新 Owned 数组（需拷贝）。`order` 为 `None` 时默认 F-contiguous，`Some(Order::C)` 时输出 C-contiguous |
| 维度约束 | 所有输入数组的形状须完全一致（shape 相同） |
| 结果维度 | ndim = 输入 ndim + 1，新轴插入在 `axis` 位置 |
| 新轴长度 | 等于输入数组的数量 |
| axis 范围 | `0 <= axis <= input_ndim`（可在最后一轴之后插入） |
| 输入数量 | 至少 1 个输入；0 个输入返回 `InvalidShape` 错误 |
| 元素顺序 | 沿新轴的第 i 个切片对应第 i 个输入数组 |
| 输出布局 | 默认 F-contiguous；可通过参数指定 C-contiguous |
| 返回类型 | 输入为 `D` 维时，返回类型的维度为 `D+1`（静态维度须在编译时确定目标维度类型） |
| Ix6 上限 | 对 Ix6 数组调用 stack 时，结果维度为 7，超出静态维度上限（Ix6），编译错误。如需在 Ix6 上堆叠，须先转为 IxDyn |
| 与 cat 的关系 | `stack(axis, arrays)` 等价于先对每个输入 `unsqueeze(axis)` 再 `cat(axis, ...)` |

### 17.11 flatten 语义

提供两个独立方法，避免单函数返回两种类型的歧义：

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
| 行为 | 始终返回 Owned 数组。连续输入等价于 `flatten_view().to_owned()`；非连续输入按 F-order 拷贝后展平 |
| 可指定顺序 | 提供 `flatten_order(order)` 变体，可显式指定 F/C 展平顺序 |
| 空数组 | 返回长度为 0 的 1D Owned 数组 |

### 17.12 index_axis 语义

| 属性 | 行为 |
|------|------|
| 操作 | `index_axis(axis, index)` 沿指定轴取单个切片，返回降维视图（ndim - 1） |
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
| 返回类型 | Vec<TensorView>，长度等于指定轴的 size |
| 内存布局 | 每个视图共享源数组底层存储，零拷贝 |
| 与 index_axis 关系 | `unstack(axis)[i]` 等价于 `index_axis(axis, i)` |
| 空轴 | 轴长度为 0 时返回空 Vec |

### 17.14 split/chunk 语义

| 属性 | 行为 |
|------|------|
| split(axis, indices) | 沿指定轴按索引列表分割，返回 Vec<TensorView>，零拷贝 |
| chunk(axis, n_chunks) | 沿指定轴均匀分割为 n 块；若轴长度不能整除，前 (len % n) 块各多 1 个元素 |
| 返回类型 | Vec<TensorView>，每个视图共享源数组底层存储 |
| n_chunks = 0 | 返回空 Vec |
| n_chunks > 轴长度 | 返回轴长度个大小为 1 的块（多余的 chunk 数被忽略） |
| 空轴 | 轴长度为 0 时返回 n_chunks 个空视图 |

### 17.15 pad 语义

| 属性 | 行为 |
|------|------|
| pad(widths, mode) | 沿各轴两侧填充，返回新 Owned 数组（需拷贝） |
| widths 参数 | 每轴一对 (before, after)，指定前后填充宽度 |
| mode: Constant(value) | 用指定常量填充 |
| mode: Edge | 用最近的边缘元素重复填充。对于轴长度为 N 的数组，pad_before 位置的填充值等于 index[0]（首元素），pad_after 位置的填充值等于 index[N-1]（末元素）。等效于 NumPy 的 `mode='edge'` |
| mode: Reflect | 镜像反射填充（不含边缘元素） |
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
| 约束 | `window_size` 各轴须 > 0，否则 panic；`window_size` 的 ndim 须与源数组一致 |
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
| reps 参数 | 每轴一个重复次数；reps 长度不足时左侧补 1 |
| reps 含 0 | 对应轴长度变为 0，结果为空数组 |
| reps 全为 1 | 等价于拷贝 |

---

## 18. 索引操作

### 18.1 索引类型

| 类型 | 说明 |
|------|------|
| 多维索引 | `[i, j, k]` 形式，i, j, k 为元素索引 |
| 范围索引 | 切片语法 |
| 切片宏 | `s![]` 形式（语义见 §13.9） |
| 高级索引 | take, take_along_axis, mask, compress, put, argwhere/nonzero |
| 条件选择 | select(condition, x, y) |

### 18.2 take 语义

| 属性 | 行为 |
|------|------|
| 签名 | `fn take(&self, indices: &Tensor1<usize>, axis: Option<Axis>) -> Result<Tensor<A, IxDyn>>` |
| 操作 | 沿指定轴按索引数组提取元素，返回新数组 |
| axis=None | 将输入展平后按一维索引取值 |
| axis=Some(i) | 沿第 i 轴取值，其余轴不变，指定轴长度变为 indices.len() |
| 约束 | 索引值须 < 轴长度，否则返回 `IndexOutOfBounds` 错误 |
| 拷贝 | 始终拷贝，返回 Owned |

**返回类型说明**：`take` 统一返回 `Tensor<A, IxDyn>`（动态维度），即使输入为静态维度。

### 18.3 take_along_axis 语义

| 属性 | 行为 |
|------|------|
| 签名 | `fn take_along_axis(&self, indices: &Tensor<A, D>, axis: Axis) -> Tensor<A, D>` |
| 操作 | 沿指定轴按索引数组取值，索引数组形状须与源张量一致（指定轴除外） |
| 约束 | 除指定轴外，indices 其余轴长度须与 self 相同；索引值 < 指定轴长度 |
| 典型用途 | `argmax` 结果的反向索引：`a.take_along_axis(&a.argmax(axis), axis)` |
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
| 签名 | `fn compress(&self, mask: &Tensor1<bool>, axis: Axis) -> Tensor<A, D>` |
| 操作 | 沿指定轴按布尔掩码提取，保留维度 |
| 约束 | mask 长度须等于指定轴长度，否则返回 `ShapeMismatch` 错误 |
| 返回值 | 维度类型与 self 相同（`D`），指定轴长度为 mask 中 true 的数量。其余轴长度不变 |
| 与 mask 区别 | mask 展平为一维；compress 保留维度结构 |
| 拷贝 | 始终拷贝 |

### 18.6 put 语义

| 属性 | 行为 |
|------|------|
| 签名 | `fn put(&mut self, indices: &Tensor1<usize>, values: &Tensor1<A>, axis: Option<Axis>) -> Result<()>` |
| 操作 | 按索引数组将值写入指定位置，原地修改 |
| 约束 | 索引值须 < 轴长度（axis=Some 时为指定轴长度，axis=None 时为元素总数）；values 长度须与 indices 长度相等 |
| 错误 | 索引越界返回 `IndexOutOfBounds`（含越界索引值、轴长度、轴号）；values 与 indices 长度不匹配返回 `ShapeMismatch` |
| 与 take 对称 | `put` 为 `take` 的逆操作 |

### 18.7 argwhere 语义

| 属性 | 行为 |
|------|------|
| 签名 | `fn argwhere(&self) -> Tensor2<usize> where A: PartialEq` |
| 操作 | 返回非零（`!= A::zero()` 或对 bool 类型为 `true`）元素的多维索引 |
| 返回值 | Tensor2<usize>，形状为 (n_nonzero, ndim) |
| 空数组 | 若无非零元素，返回形状 (0, ndim) |

### 18.8 nonzero 语义

| 属性 | 行为 |
|------|------|
| 签名 | `fn nonzero(&self) -> Vec<Tensor1<usize>> where A: PartialEq` |
| 操作 | 返回各轴的非零索引，类似 NumPy `np.nonzero()` |
| 返回值 | Vec 长度为 ndim，每个元素为该轴上非零位置的索引数组 |
| 空数组 | 各轴均返回空 Tensor1 |

**argwhere vs nonzero 返回类型差异）**：两者提供同一数据的不同视角——`argwhere` 返回 `Tensor2<usize>`（行=坐标点，列=轴），适合逐点访问；`nonzero` 返回 `Vec<Tensor1<usize>>`（每轴一个索引数组），适合按轴花式索引（如 `a[nonzero_result]`）。这两种返回类型对应 NumPy 中 `np.argwhere` 和 `np.nonzero` 的设计，保持 API 兼容性。两者之间可通过转置互转：`argwhere` 的结果转置后按行拆分即得 `nonzero` 的结果。

### 18.9 切片宏 s![]语义

`s![]` 宏用于构造多维切片描述符，语法糖对应 `SliceInfo` 类型。

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

**边界行为**

| 场景 | 行为 |
|------|------|
| 空范围（a >= b 且 step > 0） | 返回该轴长度为 0 的视图 |
| step = 0 | 静态维度：宏展开时对**字面量** `0` 报编译错误（仅检测 `;0` 的 token 模式，无法检测变量值为 0 的情况）；动态维度：运行时 panic。注意：若 step 为非常量表达式（如 `s![0..6;n]`，运行时 `n == 0`），静态维度也会在运行时 panic |
| 轴数与 ndim 不匹配 | 编译错误（静态维度）或 panic（动态维度） |

### 18.10 select 语义

| 属性 | 行为 |
|------|------|
| 操作 | `select(condition, x, y)` 按条件逐元素选择，condition 为 true 取 x，否则取 y |
| condition 类型 | bool 数组（或可广播为目标形状的 bool 数组） |
| x/y 类型约束 | x 和 y 须为同类型 Tensor（元素类型相同）或同类型标量值（`A`）。标量值等价于构造 `Tensor0<A>` 后广播。x 和 y 不支持隐式类型转换——若 x 为 `Tensor<f64, D>` 则 y 也须为 `f64` 类型 |
| 广播 | condition、x、y 三者须形状一致或可广播，结果形状为广播后形状 |
| 返回类型 | 新分配的 Owned 数组 |

---

## 19. 构造操作

### 19.1 构造方法

| 方法 | 说明 |
|------|------|
| zeros/ones/full | 支持指定 Order |
| eye/identity/diag | 单位矩阵和对角矩阵（见下方语义表） |
| from_vec/from_slice | 从数据源构造 |
| arange/linspace/logspace | 序列生成 |
| unsafe empty_uninit | 返回 `Tensor<MaybeUninit<A>, D>`，未初始化内存（见 §14.1 empty_uninit 语义） |

### 19.2 eye/identity/diag 语义

| 方法 | 签名 | 说明 |
|------|------|------|
| eye | `fn eye(n: usize) -> Tensor<A, Ix2>` | n×n 单位矩阵，对角线为 `one()`，其余为 `zero()` |
| identity | `fn identity(n: usize) -> Tensor<A, Ix2>` | `eye` 的别名，语义等价 |
| diag | `fn diag(diag: &Tensor1<A>) -> Tensor<A, Ix2>` | 以一维数组为主对角线构造方阵，非对角位置为 `zero()`（等价于 `diag_with_offset(diag, 0)`） |
| diag_with_offset | `fn diag_with_offset(diag: &Tensor1<A>, offset: isize) -> Tensor<A, Ix2>` | 偏移对角线：`offset > 0` 上移，`offset < 0` 下移；输出形状为 `(n + |offset|) × (n + |offset|)` |

| 约束 | 要求 |
|------|------|
| Element 约束 | 调用方须满足 `A: Element`（需要 `zero()`/`one()`） |
| 维度固定 | 返回类型固定为 `Tensor<A, Ix2>`（二维），不支持高维 |
| 内存布局 | 默认 F-contiguous（列优先，与 §6.1 项目默认布局一致） |

### 19.3 empty_uninit 语义

| 属性 | 行为 |
|------|------|
| 签名 | `unsafe fn empty_uninit(shape: D) -> Tensor<MaybeUninit<A>, D>` |
| 安全性 | `unsafe`：调用方须在读取前初始化所有元素，否则为未定义行为 |
| 初始化完成 | 调用方通过 `assume_init()` 将 `Tensor<MaybeUninit<A>, D>` 转换为 `Tensor<A, D>`，该方法同样为 `unsafe` |
| 用途 | 性能关键路径中避免零初始化开销（如上游库立即填充全部元素） |
| 对齐 | 使用默认 64 字节对齐分配 |
| 不提供 safe `empty` | 避免与 `zeros` 语义重复或引入隐式零初始化歧义 |

### 19.4 arange 语义

| 属性 | 行为 |
|------|------|
| 签名 | `fn arange(start: A, end: A, step: A) -> Tensor1<A> where A: Numeric` |
| 范围 | 半开区间 `[start, end)`，不含 end |
| 空范围 | `start >= end`（step > 0）或 `start <= end`（step < 0）时返回空 Tensor1 |
| step = 0 | panic（运行时） |
| step 精度 | step 类型与元素类型相同（泛型参数 A） |
| 元素数 | `max(0, ceil((end - start) / step))` |
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
| base = 10 | 常见用法，`logspace(10.0, 0.0, 3.0, 5)` → `[1.0, 10.0, 100.0, 1000.0, 10000.0]` |
| 仅限 RealScalar | 与 linspace 一致，仅对浮点类型实现 |

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

**存储复用前提**：被复用的操作数须为连续内存且形状与输出一致（含广播展开后）。若不满足，回退到分配新数组。

### 20.3 单态化控制

运算符重载的实现须采用以下策略控制泛型实例化爆炸：

1. **统一 Storage 泛型**：每个运算符的 `impl` 块基于 `TensorBase<S, D>` 统一泛型编写（S: Storage），覆盖 Owned/View/ViewMut/Arc 四种存储模式，而非为每种存储模式编写独立 impl。编译器仅对用户实际调用的 `(S, D, A)` 组合生成代码，未使用的组合不产生编译开销。

2. **内部 monomorphic kernel**：运算符 impl 体仅做类型转换和参数准备，核心计算逻辑委托给按 `(运算符, 元素类型)` 分派的 monomorphic 函数（如 `add_f64_raw(ptr, ptr, ptr, len)`）。泛型 impl 被内联后，编译器可识别不同实例共享同一 kernel，消除重复代码生成。

### 20.4 视图参与运算

| 操作数类型 | 行为 |
|------------|------|
| `&TensorView` / `&TensorViewMut` | 等同于 `&Tensor`，始终分配新数组 |
| `TensorViewMut`（消耗） | 复用其底层存储，结果为 Owned（前提：存储连续且形状匹配，否则分配新数组） |
| `TensorView`（消耗） | 视图无自有存储，消耗后仍需分配新数组 |
| `&ArcTensor` | 等同于 `&Tensor`，始终分配新数组（不触发 make_mut） |
| `ArcTensor`（消耗） | 通过 `Arc::try_unwrap()` 尝试获取独占所有权：成功则复用存储（等效 Owned）；失败（引用计数 > 1）则分配新数组 |

### 20.5 复合赋值约束

| 规则 | 说明 |
|------|------|
| 左操作数 | 须为可写存储（Owned 或 ViewMut），View 和 ArcRepr 不支持复合赋值 |
| 右操作数广播 | 右操作数可被广播到左操作数形状 |
| 左操作数广播 | 禁止——左操作数须拥有完整存储，不可为广播视图 |

### 20.6 比较运算符

比较运算符不参与上述所有权/存储复用矩阵，因为输出为 `Tensor<bool, D>`（类型不同于输入），始终分配新数组。

**设计决策**：逐元素比较返回 `Tensor<bool, D>`，但 `std::cmp::PartialOrd` trait 的方法签名返回 `bool`，无法覆盖为返回 Tensor。因此**不通过 `PartialOrd` 实现运算符重载**，而是采用以下方案：

- `< <= > >=` 运算符**不提供**（`PartialOrd` 返回 `bool`，与逐元素 Tensor 语义冲突）
- 逐元素比较**仅通过方法调用**：`.lt()`、`.le()`、`.gt()`、`.ge()`（见 §11.1 比较表），返回 `Tensor<bool, D>`
- **命名说明**：`.lt()` 等方法名与 `PartialOrd` trait 的固有方法同名，但 `TensorBase` **不实现 `PartialOrd`**，因此不存在 trait 方法冲突。此处选择 `.lt()` 而非 `.elem_lt()` 等替代命名，是因为：(1) 与 NumPy 的 `np.less()` / ndarray 的 `.lt()` 命名惯例一致；(2) 返回类型 `Tensor<bool, D>` 已明确区分于 `PartialOrd::lt() -> bool`；(3) IDE 自动补全不会提示 `PartialOrd::lt()`（因未实现该 trait）
- `==` / `!=`：`PartialEq` trait 返回 `bool`（标量语义，判断两个 Tensor 是否完全相等）；逐元素布尔掩码使用 `.eq()` / `.ne()` 方法

| 方法 | 签名 | 返回类型 | 元素约束 |
|------|------|----------|----------|
| `.lt(&self, other) -> Tensor<bool, D>` | 逐元素 `<` | `Tensor<bool, D>` | `A: PartialOrd` |
| `.le(&self, other) -> Tensor<bool, D>` | 逐元素 `<=` | `Tensor<bool, D>` | `A: PartialOrd` |
| `.gt(&self, other) -> Tensor<bool, D>` | 逐元素 `>` | `Tensor<bool, D>` | `A: PartialOrd` |
| `.ge(&self, other) -> Tensor<bool, D>` | 逐元素 `>=` | `Tensor<bool, D>` | `A: PartialOrd` |

以上方法支持广播（与 §11.4 广播规则一致），始终分配新数组。`PartialEq` 的 `==`/`!=` 运算符保持标量语义（`Tensor<A, D> == Tensor<A, D> -> bool`，判断两个数组形状和所有元素是否完全相等）。**NaN 行为**：由于 `PartialEq` 遵循 IEEE 754（`NaN != NaN`），两个形状相同且在相同位置均为 NaN 的 Tensor，`==` 比较结果为 `false`（因为逐元素比较时 NaN != NaN）。若需 NaN 敏感的相等判断，使用 `allclose(rtol=0.0, atol=0.0)`（但注意 `allclose` 同样将 NaN 比较为 false，见 §14.3）。

---

## 21. 实用操作

### 21.1 基本实用操作

| 操作 | 说明 |
|------|------|
| copy_to, fill | 复制和填充 |
| is_close/allclose | 近似比较（语义见 §14.3 is_close/allclose 语义） |
| clip | 裁剪 |
| flip/flipud/fliplr | 翻转 |
| to_owned/into_owned | 转换为拥有 |

### 21.2 flip / flipud / fliplr 语义

| 方法 | 签名 | 说明 |
|------|------|------|
| `flip` | `fn flip(&self, axes: &[usize]) -> TensorView<A, D>` | 沿指定轴翻转，零拷贝（步长取反）。axes 中的轴索引须在 `[0, ndim)` 范围内，否则返回 `InvalidAxis` |
| `flipud` | `fn flipud(&self) -> TensorView<A, D>` | 沿轴 0 翻转（等价于 `flip(&[0])`）。ndim ≥ 1 时反转第 0 轴元素顺序；0D 数组为 no-op |
| `fliplr` | `fn fliplr(&self) -> TensorView<A, D>` | 沿轴 1 翻转（等价于 `flip(&[1])`）。1D 数组返回 `InvalidAxis`（无轴 1） |

### 21.3 clip 语义

| 属性 | 行为 |
|------|------|
| 签名 | `fn clip(&self, min: A, max: A) -> Tensor<A, D> where A: PartialOrd + Clone` |
| 操作 | 逐元素裁剪到 `[min, max]` 范围内，小于 min 的值设为 min，大于 max 的值设为 max |
| min > max | panic（调用方违反前置条件） |
| 返回类型 | 新分配的 Owned 数组 |
| NaN 处理 | NaN 比较为 false，clip 不修改 NaN（NaN 不满足 min/max 约束，保持不变） |

### 21.4 copy_to / fill 语义

| 方法 | 签名 | 说明 |
|------|------|------|
| `copy_to` | `fn copy_to(&self, dst: &mut TensorViewMut<A, D>)` | 将 self 的数据拷贝到 dst。两者形状须完全一致，否则返回 `ShapeMismatch`。按物理内存顺序拷贝 |
| `fill` | `fn fill(&mut self, value: A) where A: Clone` | 用指定值填充所有元素。要求可变引用（Owned 或 ViewMut） |

### 21.5 is_close / allclose 语义

| 方法 | 签名 | 返回类型 |
|------|------|----------|
| `is_close` | `fn is_close(&self, other: &Self, rtol: A, atol: A) -> Tensor<bool, D> where A: RealScalar` | `Tensor<bool, D>` |
| `allclose` | `fn allclose(&self, other: &Self, rtol: A, atol: A) -> bool where A: RealScalar` | `bool` |

| 属性 | 行为 |
|------|------|
| 比较公式 | `|a - b| ≤ atol + rtol * |b|`（与 NumPy `np.isclose` / `np.allclose` 一致；注意此公式非对称——交换 a/b 可能改变结果。这是 NumPy 的设计选择，Xenon 予以保留以保持兼容性） |
| rtol / atol | 相对容差和绝对容差，均须 ≥ 0，否则 panic |
| 广播 | `is_close` 支持广播（condition 与 x/y 形状可广播，与 §11.4 一致），返回广播后形状的 bool Tensor |
| NaN 处理 | 两个 NaN 比较为 `false`（与 NumPy `np.isclose` 一致）；若需 NaN 相等判定，须用户自行处理 |
| Inf 处理 | `+Inf == +Inf` 为 `true`，`+Inf != -Inf`，遵循 IEEE 754 |
| allclose 语义 | 等价于 `is_close().all()`，所有元素近似相等时返回 `true` |
| 空数组 | `allclose` 对空数组返回 `true`（vacuous truth） |

---

## 22. 连续性保证

返回类型始终为 `Tensor<A, D>`，保证连续布局。即使输入已连续，也返回新分配的 Owned 数组（数据拷贝）。

| 方法 | 行为 |
|------|------|
| to_f_contiguous() | 若输入已 F-contiguous，返回 `to_owned()`（拷贝数据到新分配）；否则重新排列为 F-contiguous 布局并返回新 Tensor |
| to_c_contiguous() | 若输入已 C-contiguous，返回 `to_owned()`（拷贝数据到新分配）；否则重新排列为 C-contiguous 布局并返回新 Tensor |
| to_contiguous() | 若输入已连续（F 或 C），返回 `to_owned()`（拷贝数据到新分配）；否则输出 F-contiguous 布局的新 Tensor |
| as_f_contiguous() | 若输入已 F-contiguous，返回 `Some(TensorView)`（零拷贝视图）；否则返回 `None` |
| as_c_contiguous() | 若输入已 C-contiguous，返回 `Some(TensorView)`（零拷贝视图）；否则返回 `None` |
| as_contiguous() | 若输入已连续（F 或 C），返回 `Some(TensorView)`（零拷贝视图，保持原布局）；否则返回 `None` |

---

## 23. 转换操作

### 23.1 cast 语义

**方法签名**：

| 方法 | 签名 | 说明 |
|------|------|------|
| `cast` | `fn cast<B>(self) -> Tensor<B, D> where B: Element` | 消耗 Owned 数组，逐元素类型转换。**仅对 `Tensor<A, D>`（Owned）实现**。View 须先 `to_owned()` 再 cast |
| `cast_same` | `fn cast_same(&self) -> Tensor<A, D>` | 返回同类型拷贝（等价于 `to_owned()`），用于显式语义表达 |

**泛型约束**：`B: Element` 约束保证目标类型在本库的类型体系内（Sealed trait，见 §3）。转换规则为逐元素按值转换，始终成功（溢出使用饱和语义，见下方精度表）。不支持跨类别隐式转换（如 `Complex → 实数` 须显式取 `re()`）。

**存储模式支持**：

| 输入类型 | cast 行为 |
|----------|-----------|
| `Tensor<A, D>`（Owned） | 消耗 self，分配新的 `Tensor<B, D>` |
| `TensorView<A, D>` | 不支持，须先 `.to_owned().cast::<B>()` |
| `ArcTensor<A, D>` | 不支持，须先 `.to_owned().cast::<B>()` |

**精度行为**：
| 转换方向 | 行为 |
|----------|------|
| 浮点 → 浮点（高精度→低精度） | 按 IEEE 754 round-to-nearest-even 截断 |
| 浮点 → 浮点（低精度→高精度） | 精确转换，无精度损失 |
| 浮点 → 整数 | 向零截断（truncate），溢出为饱和（saturating cast） |
| 整数 → 浮点 | 最近偶数舍入（round-to-nearest-even） |
| 整数 → 整数（窄化） | 饱和截断（saturating cast） |
| NaN → 整数 | 结果为 0 |
| Inf → 整数 | 饱和到目标类型的 MAX/MIN |
| bool → 数值 | true = 1, false = 0 |
| 数值 → bool | 非零 = true, 零 = false |
| 实数 → 复数 | 虚部为 0 |
| 复数 → 实数 | 不允许隐式转换，须显式取 re() |

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
| 静态 → 动态（IxN → IxDyn） | 总是成功 | 无（见 §2.1） |
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
| `Borrow<TensorView<A, D>>` | `Tensor<A, D>` | 语义与 `AsRef` 一致，支持 `HashMap` 等集合场景 |
| `Default` | `Tensor<A, D>` | 仅当 `A: Default + Element` 且 `D: Default`（即静态维度 Ix0，shape 为空）时实现，返回包含 `A::default()` 的 0D 标量张量。其他维度类型不实现 `Default`（无法确定 shape）。实际用途有限——创建数组推荐使用 `zeros(shape)` 或 `full(shape, value)` |
| `From<Vec<A>>` | `Tensor1<A>` | 见 §14.6 From/Into 转换表 |
| `From<[A; N]>` | `Tensor1<A>` | 见 §14.6 From/Into 转换表 |

---

## 24. 格式化输出

**Debug / Display 格式规范**：

| 属性 | 要求 |
|------|------|
| 风格 | NumPy 风格（`array([...])`），包含 shape 和 dtype 信息 |
| 浮点精度 | Display：按 `Display` trait 输出（通常 6 位有效数字）；Debug：完整精度（`{:?}` 输出所有有效位） |
| 多维缩进 | 2D 及以上数组每维缩进 4 空格，行内元素空格分隔 |
| 大数组截断 | 元素总数 > 1000 时省略中间元素，首尾各显示最多 3 个（`..., ` 分隔），行末标注 `... (N elements)` |
| 空数组 | 显示为 `[]`（标注 shape） |
| 0D 标量 | 显示为标量值（不带中括号） |
| NaN/Inf | NaN 显示为 `NaN`，+Inf 显示为 `inf`，-Inf 显示为 `-inf` |
| 存储模式标注 | Display：不标注存储模式（面向用户，输出简洁）。Debug：标注存储模式（`view`/`view_mut`/`owned`/`arc`）、偏移量（offset ≠ 0 时显示）和步长信息（非默认步长时显示），便于调试非连续视图和切片 |
| 步长显示 | Debug 中，当数组非连续（`!is_contiguous()`）时，在 shape 信息后追加 strides 显示，如 `shape=[3,4], strides=[8,1]` |

---

## 25. FFI 集成

### 15.1 指针 API

| 方法 | 说明 |
|------|------|
| as_ptr() | 返回数据起始位置的不可变原始指针 `*const A` |
| as_mut_ptr() | 返回数据起始位置的可变原始指针 `*mut A`（须可写存储） |
| as_ptr_unchecked() | unsafe，不检查偏移量有效性 |

### 15.2 形状与步长查询

以下方法的语义和签名定义见 §6.3（`shape()`、`strides()`、`offset()`），本节补充 FFI 场景专用的查询方法：

| 方法 | 说明 |
|------|------|
| strides_bytes() | 返回各轴步长的切片（字节单位，有符号）。FFI 场景下外部库通常需要字节单位步长 |

> `shape()` / `strides()` / `offset()` 在 §6.3 中定义，FFI 集成直接调用即可。此处仅补充 `strides_bytes()`，因为 `§6.3` 的 `strides()` 返回元素单位步长，而 C/BLAS 接口通常需要字节单位。

### 15.3 BLAS 兼容性

| 方法 | 说明 |
|------|------|
| lda() | 返回 leading dimension（F-order 下为第一轴步长），仅 2D 及以上。对 batch 视图（如 3D 数组在最后一维上的 2D 切片），lda() 返回该 2D 切片的第一轴步长，不受 batch 轴影响 |
| is_blas_compatible() | 检查内存布局是否可直接传递给 BLAS（F-contiguous 或 C-contiguous，正步长，无零步长） |
| blas_layout() | 返回 BLAS 布局标识（F/C/None），None 表示不兼容 |
| blas_trans() | 返回 BLAS Trans 参数（N/T/C），基于当前布局与目标布局的关系 |

**Trans 枚举定义**：

| 变体 | 含义 | BLAS 字符 |
|------|------|-----------|
| `Trans::None` | 不转置（原始布局） | `'N'` |
| `Trans::Transpose` | 转置（行列互换） | `'T'` |
| `Trans::ConjTranspose` | 共轭转置（仅 Complex 类型有意义） | `'C'` |

> Trans 枚举用于 FFI 与 BLAS 交互时描述矩阵的转置状态。`blas_trans()` 根据当前 Tensor 的内存布局（F-contiguous 或 C-contiguous）自动推断需要传递给 BLAS 的转置参数。对实数类型，`Trans::ConjTranspose` 等价于 `Trans::Transpose`。

### 15.4 索引转换

| 方法 | 说明 |
|------|------|
| index_to_ptr(index) | 将多维索引转换为对应元素的原始指针 |
| index_to_offset(index) | 将多维索引转换为相对于数据起始位置的元素偏移量 |

### 15.5 原始部件构造与解构

| 方法 | 签名 | 说明 |
|------|------|------|
| from_raw_parts | `unsafe fn from_raw_parts<'a>(ptr: *const A, shape: D, strides: D::Stride, offset: usize) -> TensorView<'a, A, D>` | 从原始部件构造 TensorView。`shape` 和 `strides` 接受维度类型对应的内部表示（静态维度为 `[usize; N]`/`[isize; N]`，IxDyn 为 `Vec<usize>`/`Vec<isize>`）。提供 `from_raw_parts_slice` 便捷变体，接受 `&[usize]`/`&[isize]` 切片并自动转为 `D` |
| from_raw_parts_mut | `unsafe fn from_raw_parts_mut<'a>(ptr: *mut A, shape: D, strides: D::Stride, offset: usize) -> TensorViewMut<'a, A, D>` | 同上，构造可变视图 |

> **生命周期 `'a` 的语义**：`'a` 表示"底层内存保持有效的时长"，由调用方通过类型标注指定。`'a` 的来源取决于内存所有权方式：(1) 从 `&[A]` 切片构造时，`'a` 绑定到切片的 lifetime；(2) 从 `Arc<[A]>` 构造时，调用方可选择 `'a = 'static`（若 Arc 永不释放）或手动管理 lifetime（如将 Arc 和 TensorView 放入同一结构体）；(3) 从 `Box<[A]>` 构造时，`'a` 绑定到 Box 所有权持有者的 lifetime。**调用方责任**：确保 `'a` 不超过底层内存的实际存活时间——若内存被释放后视图仍存在，则访问视图为 UB。
| into_raw_parts | `fn into_raw_parts(self) -> RawParts<A, D>` | 消耗数组，返回原始部件。**内存释放责任**：调用方获得原始指针后须自行管理内存生命周期——通过 `from_raw_parts` 重新构造为 Tensor 交由 Drop 释放，或使用全局分配器的 `dealloc` 手动释放（须与返回的 `Layout` 一致）。忘记释放将导致内存泄漏 |

**`RawParts<A, D>` 结构体**：

```rust
pub struct RawParts<A, D: Dimension> {
    pub ptr: *mut A,
    pub shape: D,
    pub strides: D::Stride,
    pub offset: usize,
    pub layout: Layout,  // 分配时的 Layout，用于安全释放
    pub len: usize,      // 分配的总元素数（≥ 实际使用的元素数，含 padding）
}
```

> **`layout` 字段说明**：记录分配时的 `alloc::alloc::Layout`（含大小和对齐），使调用方可安全调用 `alloc::dealloc`。若数组通过 `zeros`/`full` 等构造函数分配，`layout` 完整记录分配信息；若数组由 `from_raw_parts` 构造（非 Xenon 分配的内存），`layout` 为 `None`，调用方须自行追踪分配信息。`len` 字段记录分配的总元素数（含 padding），与 `shape.size()` 可能不同（后者为逻辑元素数，不含 padding）。

**Safety 前提条件（from_raw_parts / from_raw_parts_mut）：**

调用方须保证以下条件，否则导致未定义行为：

| 前提条件 | 说明 |
|----------|------|
| 指针有效性 | `ptr` 须非空、非悬垂，且对齐到 `align_of::<A>()` |
| 内存范围 | `ptr` 起始的内存范围须足够大，能覆盖所有可访问元素（考虑 offset、shape、strides） |
| 生命周期 | 内存须在返回的视图生命周期内保持有效 |
| 别名规则 | `from_raw_parts`：内存可被共享读取，但不可被写入；`from_raw_parts_mut`：内存须无其他引用（独占访问） |
| 布局一致性 | `shape` 与 `strides` 长度须一致；strides 须为合法的元素步长 |
| 边界安全 | 任意合法索引计算出的偏移量不得越界访问内存 |
| 元素初始化 | 所有可访问元素须已正确初始化（符合 `MaybeUninit` 语义） |
| 线程安全 | 若视图跨线程使用，底层内存须满足 Send/Sync 要求 |

---

## 16. 临时工作空间

### 16.1 工作空间属性

| 属性 | 要求 |
|------|------|
| 对齐 | 默认 64 字节对齐，须支持调用方指定更大对齐（如 128 字节） |
| 生命周期 | 借用语义：工作空间借出期间不可再次借出，归还后可复用 |
| 增长策略 | 请求超出当前容量时自动扩容，不缩容；扩容后保持对齐。提供 `shrink_to(new_len)` 方法将未使用空间归还给分配器，调用方须保证 `new_len ≤ 当前已分配容量`，否则 panic。**设计理由**：长时间运行的算法可能反复使用不同大小的临时空间，若 workspace 只增不减，内存占用将持续增长至峰值。`shrink_to` 允许调用方在当前操作完成后释放多余空间，`shrink_to(0)` 可完全清空（保留最小容量和对齐元数据） |
| 线程亲和性 | 工作空间不绑定线程；线程安全由调用方通过借用规则保证 |
| 零初始化 | 不保证零初始化（性能考虑），调用方须自行初始化 |

### 16.2 分割与嵌套

| 属性 | 要求 |
|------|------|
| 嵌套借用 | 支持从同一工作空间分割多个不重叠的子缓冲区 |
| 并行分割 | 支持 `split_at(mid)` 将工作空间分割为两个不重叠的子工作空间，各自可独立借出，O(1)，不涉及内存分配 |
| 递归分割 | 子工作空间可继续分割（支持递归二分），最小分割粒度为 16 字节（小于 16 字节的分割请求返回空子工作空间）。**设计理由**：防止无限递归分割产生零大小子空间导致的无限循环；16 字节下限覆盖所有常见 SIMD 类型的对齐需求 |

> **Workspace 线程安全**：`Workspace` 实现 `Send` 但不实现 `Sync`——工作空间可在线程间移动（Send），但不可被多线程同时访问（非 Sync）。`split_at` 返回的两个子工作空间各自为 `Send`，可分别移动到不同线程使用，但每个子工作空间在同一时刻只能被一个线程访问（由 `&mut` 借用规则在编译时保证）。父工作空间在子工作空间活跃期间不可借出（借用检查器保证）。



### 16.3 借出与归还 API

| 方法 | 签名 | 说明 |
|------|------|------|
| `borrow_bytes` | `fn borrow_bytes(&mut self, len: usize) -> &mut [u8]` | 借出指定字节数的缓冲区，按当前对齐对齐。借出期间不可再次借出同一区间 |
| `borrow_aligned` | `fn borrow_aligned<A>(&mut self, len: usize) -> &mut [MaybeUninit<A>]` | 借出指定元素数的类型化缓冲区，按 `max(align_of::<A>(), 当前对齐)` 对齐。返回 `&mut [MaybeUninit<A>]` 而非 `&mut [A]`——工作空间不保证零初始化（见 §16.1），返回已初始化引用将构成 UB。调用方须自行初始化后通过 `MaybeUninit::assume_init` 转换。对齐不足时 panic |
| `borrow_scratch` | `fn borrow_scratch(&mut self, req: ScratchRequirement) -> &mut [u8]` | 根据 `ScratchRequirement` 借出足够大小和指定对齐的缓冲区 |
| 归还 | 借出的 `&mut` 引用生命周期绑定到 `&mut Workspace`，引用释放后自动归还，无需显式调用 | 借用检查器保证借出期间不存在二次借出 |

### 16.4 Scratch 查询 API

上游库（如线性代数库）须能在执行操作前查询所需临时内存大小，以便预分配或复用工作空间。

| 属性 | 要求 |
|------|------|
| scratch_size 查询 | 提供静态方法，根据操作类型和矩阵尺寸返回所需字节数 |
| 返回类型 | 返回 `ScratchRequirement` 结构体（大小 + 对齐），而非裸 usize。定义：`#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)] pub struct ScratchRequirement { pub size: usize, pub alignment: usize }`，其中 `size` 为所需字节数，`alignment` 为最小对齐要求（须为 2 的幂，构造时通过 `debug_assert!` 验证） |
| 可组合性 | 多个 scratch 需求可合并：`size` 字段顺序执行取 max、并行执行取 sum；`alignment` 字段始终取 max（对齐要求不可缩减） |
| 与工作空间集成 | 工作空间可接受 scratch 需求描述，一次性分配足够空间 |
| 零运行时开销 | scratch 查询为纯计算，不分配内存 |
| 上游库用法 | 上游库定义自己的操作枚举并实现 scratch 查询；Xenon 仅提供基础设施（需求描述类型 + 合并逻辑 + 工作空间分配接口） |

---

# 第五部分：工程保障

## 17. 错误处理

### 17.1 错误分类

**所有公开错误枚举均标注 `#[non_exhaustive]`**，确保未来新增错误变体不构成 breaking change。

**下游处理建议**：`#[non_exhaustive]` 强制下游 match 使用 `_ =>` 兜底分支。推荐下游 crate 按以下优先级处理未知变体：
1. **显式处理所有已知变体**，对 `_ =>` 使用 `log::warn!("unhandled error variant: {:?}", err)` 记录日志（而非静默忽略），便于调试时发现新增变体
2. **若无法合理处理未知变体**，使用 `Err(e) =>` 将错误向上传播，而非在 `_ =>` 中吞掉
3. **避免空兜底**（`_ => {}`），因为这会静默吞掉新增变体，导致逻辑错误而非编译错误——这正是 `#[non_exhaustive]` 的已知代价

**错误类型体系**（两级架构）：

| 层级 | 错误类型 | 触发场景 | 处理方式 |
|------|----------|----------|----------|
| **独立错误** | ShapeMismatch | 二元运算/zip 形状不兼容且无法广播 | Result |
| | BroadcastError | 广播规则不满足（非 size-1 维度不等） | Result |
| | LayoutMismatch | 要求连续布局但输入非连续（如 flatten_view 非连续数组） | Result |
| | InvalidAxis | 轴索引超出维度数 | Result |
| | InvalidShape | 构造参数形状无效（如 cat 输入 ndim 不一致） | Result |
| | DimensionMismatch | 静态维度与动态维度互转时维度数不匹配 | Result |
| | EmptyArray | 对空数组执行 min/max/argmin/argmax | Result |
| | AllocationError | 内存分配失败（超大数组、系统内存不足） | panic |
| | IndexOutOfBounds | 索引越界（含多维索引、take/put 索引数组） | 多维索引：panic（checked）/ UB（unchecked）；take/put 索引数组：Result |
| ~~复合错误~~ | ~~ShapeError~~ | ~~形状操作中可能遇到的聚合错误~~ | ~~已移除~~ |

> **设计决策**：v19 定义了 `ShapeError` 复合枚举（聚合 `InvalidShape` + `LayoutMismatch`），但审查发现当前无任何 API 需要同时返回这两种错误——reshape 布局不匹配时自动拷贝仅返回 `InvalidShape`，flatten_view 返回独立 `LayoutMismatch`。因此移除 `ShapeError` 复合类型。若未来出现需要组合多种形状/布局错误的 API，可重新引入复合枚举。

**错误类型诊断字段**：所有返回 `Result` 的错误类型须携带以下诊断信息，以 `Display` 输出人类可读描述：

| 错误类型 | 诊断字段 | 说明 |
|----------|----------|------|
| ShapeMismatch | `expected: Vec<usize>`, `actual: Vec<usize>` | 期望与实际形状 |
| BroadcastError | `left: Vec<usize>`, `right: Vec<usize>`, `conflicting_axis: usize` | 冲突的两个形状及冲突轴 |
| LayoutMismatch | `requested: &'static str`, `actual_layout: &'static str` | 期望布局（如 "C-contiguous"/"F-contiguous"）与实际布局描述 |
| InvalidAxis | `axis: usize`, `ndim: usize` | 请求轴与实际维度数 |
| InvalidShape | `reason: &'static str` | 人类可读原因（如 "0 inputs" / "ndim mismatch: expected 3, got 2"） |
| DimensionMismatch | `expected: usize`, `actual: usize` | 期望与实际维度数 |
| EmptyArray | `operation: &'static str` | 触发操作名（如 "min"/"argmax"） |
| IndexOutOfBounds | `index: usize`, `axis_len: usize`, `axis: usize` | 越界索引值、轴长度、轴编号（仅 take/put Result 场景） |

panic 类错误（`AllocationError`）通过 panic message 传递诊断信息，不使用结构化字段。`IndexOutOfBounds` 在 panic 场景（多维索引越界）通过 panic message 传递诊断信息，在 Result 场景（take/put 索引数组越界）使用结构化字段。

**API → 错误类型映射**：

| API 类别 | 返回的错误类型 | 说明 |
|----------|---------------|------|
| reshape | `InvalidShape` | 仅形状不匹配时返回错误（布局不匹配时自动拷贝） |
| flatten_view | `LayoutMismatch` | 独立错误，仅检查连续性 |
| cat / stack | `InvalidShape`（0 个输入、ndim 不一致）/ `ShapeMismatch`（非拼接轴长度不等：cat 除拼接轴外各轴长度须一致，stack 所有输入形状须完全一致） | 独立错误；两者触发条件互斥，不重叠 |
| squeeze / broadcast | `InvalidShape` | 独立错误 |
| 轴操作（sum_axis / cumsum / slice / index_axis 等） | `InvalidAxis` | axis >= ndim 时返回 `InvalidAxis`（Result）。注意：s![] 宏中轴数不匹配的行为不同——静态维度编译错误，动态维度 panic（见 §13.9） |
| 维度互转（into_dimension） | `DimensionMismatch` | 独立错误 |
| 逐元素运算 / zip | `ShapeMismatch` 或 `BroadcastError` | 独立错误 |
| min/max/argmin/argmax（空数组） | `EmptyArray` | 独立错误 |
| 内存分配（构造器） | `AllocationError`（panic） | 不可恢复 |
| 索引访问 | `IndexOutOfBounds`（panic） / unchecked UB | 编程错误（多维索引硬编码或循环变量） |
| take / put | `IndexOutOfBounds`（Result） | 索引来自运行时动态数据（索引数组），属数据驱动错误 |

### 17.2 处理策略

**判断原则**：panic 用于"调用方违反了 API 前置条件"（编程错误，合理使用无法触发），Result 用于"输入数据本身的属性导致操作无法完成"（合法调用路径，调用方应处理）。具体规则：

| 策略 | 适用场景 | 典型例子 |
|------|----------|----------|
| 可恢复错误（形状、布局、轴、数据驱动索引越界） | 返回 Result | squeeze 非 size-1 轴 → `InvalidShape`；空数组 min/max → `EmptyArray`；take/put 索引数组越界 → `IndexOutOfBounds` |
| 编程错误（多维索引越界） | panic，同时提供 unsafe unchecked 变体 | `tensor[i,j]` 索引越界 → `IndexOutOfBounds` panic |
| 前置条件违反 | panic | bincount 负值输入（文档要求 ≥ 0）；`Vec<Vec<A>>` 内层长度不一致（文档要求一致）；arange step=0（文档要求 ≠ 0） |
| 内存分配失败 | panic（与 Rust 标准库 `Vec::push` 等行为一致）。调用方若需 graceful 处理，可在调用前预检元素数和预估内存占用。未来版本可考虑提供 `try_*` 变体返回 `Result` | |
| 错误信息 | 所有错误类型须实现 `Display` 和 `Error`，包含上下文信息（期望值 vs 实际值） | |

### 17.3 Panic Safety

**原则**：panic 发生后，Tensor 的内部状态须满足：(1) 可安全 drop（无未定义行为、无内存泄漏）；(2) 不保证数据一致性（已修改的部分元素可能留在中间状态）。

| 操作场景 | Panic 时保证 | 说明 |
|----------|-------------|------|
| `mapv` / `zip` 中用户闭包 panic | 已遍历的元素可能已修改，未遍历的元素保持原值；Tensor 可安全 drop | 闭包 panic 导致部分写入，这是可接受的——调用方已失去控制流，Tensor 的值已不可用 |
| `mapv_inplace` 中闭包 panic | 原地修改的部分元素处于中间状态；Tensor 可安全 drop | 同上，原地操作的中间状态不可避免 |
| `make_mut()` 触发深拷贝时 panic（分配失败） | 原 ArcTensor 数据未被修改（引用计数不变）；已分配的新缓冲区被 drop 释放 | 深拷贝先分配新缓冲区再复制，分配失败时原数据完好 |
| `empty_uninit` 构造后 `assume_init` 前 panic | Tensor 内含 `MaybeUninit` 元素，drop 时不会读取未初始化数据（`MaybeUninit<T>` 的 drop 是 no-op） | 必须保证 `Tensor<MaybeUninit<A>, D>` 的 drop impl 不触及元素内容 |
| `checked_add` / `checked_mul` 累积运算 panic（如 `sum`、`cumsum`） | 中间结果被丢弃；Tensor 可安全 drop | 溢出检查在每步执行，panic 时无悬垂指针或未初始化内存 |
| `iter_mut()` 对广播视图 panic（见 §10.3） | 未修改任何元素；Tensor 保持原状态 | panic 在写入前触发，数据完好 |

**实现要求**：

- 所有 `drop` 实现不得 panic（遵循 `Drop` trait 的惯用规则）
- `Tensor<MaybeUninit<A>, D>` 的 drop 不得调用元素的 drop glue——仅释放缓冲区内存
- 涉及资源分配的操作（如 `make_mut` 深拷贝）须确保分配失败时原有数据不受影响

---

## 18. 质量要求

### 18.1 文档

| 要求 | 范围 |
|------|------|
| doc comment | 所有公开 API |
| Safety 文档节 | 所有 unsafe 函数 |
| 使用示例 | 关键 API |

### 18.2 测试

| 要求 | 指标 |
|------|------|
| 测试类型 | 单元测试 + 集成测试 + 边界测试 |
| 行覆盖率 | ≥ 90%（数值基础设施对正确性要求极高，高覆盖率是必要的质量保障） |

### 18.3 性能基准

**基准要求**：

| 基准类别 | 操作 | 参照 | 要求 |
|----------|------|------|------|
| 逐元素运算 | add/mul/sin/exp（1D, 1M 元素，F-contiguous） | ndarray 对应操作 | 不慢于 1.2× |
| 归约 | sum/prod/min/max（1D, 1M 元素） | 手写标量循环 + SIMD | 不慢于 1.1× |
| 形状操作 | reshape（连续）、transpose、slice（连续→视图） | 零拷贝路径 | O(1) 元数据操作，无数据拷贝 |
| 内存分配 | zeros/full（1M 元素 f64） | Vec::with_capacity + fill | 不慢于 1.1× |
| 广播 | 标量 + Tensor、size-1 广播 | 视图零拷贝路径 | 无额外分配 |

**编译耗时基准**：

| 指标 | 要求 |
|------|------|
| 单文件增量编译（修改一个 mapv 调用处） | ≤ 5 秒（debug 构建） |
| 全量编译（clean build，默认 features） | ≤ 120 秒（debug 构建，8 核并行） |
| 全量编译（`--all-features`） | ≤ 180 秒（debug 构建，8 核并行） |
| 缓解策略 | 若编译耗时超标，考虑将 SIMD 分发、批量运算拆分为可选子模块（通过 feature gate 隔离） |

### 18.4 数值精度

| 运算类别 | f64 精度 | f32 精度 |
|----------|----------|----------|
| 加减乘 | 精确 | 精确 |
| 归约 | 相对误差 ≤ 1e-14 | 相对误差 ≤ 1e-5 |
| 超越函数 | 相对误差 ≤ 1e-14 | 相对误差 ≤ 1e-5 |

**数值精度验证方法论**：

| 属性 | 要求 |
|------|------|
| 参考值来源 | 使用 `libm`（Rust 标准库 `f32::sin` / `f64::sin` 等）作为参考实现；测试中对比 Xenon 结果与参考值的偏差 |
| 精度度量 | 超越函数使用相对误差 `\|computed - expected\| / \|expected\|`；归约使用混合误差：`\|computed - expected\| ≤ atol + rtol * \|expected\|`，其中 atol = 0，rtol = 精度表中归约行对应值；加减乘使用 ULP（Unit in the Last Place）验证（允许 ≤ 1 ULP） |
| 容差 | 不超过 §18.4 精度表中规定的阈值；超越函数的 ULP 阈值：f64 ≤ 4 ULP，f32 ≤ 4 ULP |
| 边界值测试 | 必须覆盖：0、±1、极大值（MAX）、极小正规数（MIN_POSITIVE）、subnormal、±Inf、NaN、π 的倍数（三角函数）、整数幂（pow） |
| 回归测试 | 使用已知正确结果的测试向量（golden test），固定随机种子确保可重复性 |
| 跨平台一致性 | 同一输入在不同平台（x86_64 vs aarch64）上的结果须在精度容差内一致（SIMD 近似可能引入跨平台差异，须在容差范围内） |

### 18.5 边界覆盖

须覆盖以下边界情况：
- 空张量
- 单元素
- 大张量
- 极端值（NaN/Inf/subnormal）
- 非连续布局
- 高维（≥4维）

### 18.6 并行与 SIMD 测试策略

| 测试类别 | 要求 | 说明 |
|----------|------|------|
| ThreadSanitizer | CI 中运行 `cargo test --all-features --target x86_64-unknown-linux-gnu -Zsanitizer=thread` | 检测数据竞争 |
| Miri | CI 中对 `--target x86_64-unknown-linux-gnu` 运行 `cargo miri test`（不含 SIMD 代码） | 检测未定义行为、越界访问 |
| 跨平台 SIMD | 相同输入在 x86_64（AVX2）和 aarch64（NEON）上的逐元素运算结果须一致（精度容差内） | 检测指令集实现差异 |
| 标量 vs SIMD 对照 | SIMD 路径结果须与标量路径结果在精度容差内一致（每个超越函数均须验证） | 确保 SIMD 优化不引入精度回归 |
| 并行确定性 | 相同输入的并行归约结果须与单线程结果一致（浮点归约允许精度容差内差异） | 确保分块策略正确性 |
| 并行边界 | 测试元素数略高于/低于并行阈值的边界情况 | 确保阈值切换无遗漏 |

### 18.7 CI 要求

| 要求 | 说明 |
|------|------|
| 平台矩阵 | Linux (x86_64, aarch64)、macOS (x86_64, aarch64)、Windows (x86_64) |
| Rust 版本矩阵 | 当前 stable + MSRV 版本 |
| Feature 矩阵 | 默认 features、`--no-default-features`、`--all-features` |
| 必须通过 | `cargo test`、`cargo test --all-features`、`cargo doc`（无 broken doc link） |
| MSRV 验证 | 使用 `cargo msrv` 或 CI 脚本验证 MSRV 声明准确 |
| 格式检查 | `cargo fmt --check` |
| Lint | `cargo clippy --all-features -- -D warnings` |
| 性能回归 | 提供 `cargo bench` 基准，CI 中记录历史结果，人工审查异常波动（暂不自动门控） |

---

*Xenon v19 — 需求说明书*
