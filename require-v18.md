# Xenon 需求说明书 v18

> Rust 科学计算多维数组库
> 版本: v18 | 日期: 2026-03-27 | 许可: MIT | MSRV: Rust 1.85+

---

# 第一部分：项目定位

## 1. 项目概述与范围

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
| no_std 支持 | 支持 `no_std`（需 `alloc`），`std` 为默认 feature |
| Feature gates | `std`（默认）、`parallel`（rayon）、`simd`（pulp） |

### 1.4 范围内

- N 维数组的存储、索引、切片、变形、逐元素运算、归约、广播
- 矩阵-向量乘法、向量内积/外积、批量运算
- 原始指针 API，供上游库零开销集成
- 自定义复数类型（FFI 友好，不依赖 num-complex）
- 临时工作空间（对齐分配，供上游库使用）

### 1.5 范围外

- GEMM（矩阵-矩阵乘法）、矩阵分解、对角化等高级线性代数
- FFT、稀疏矩阵、自动微分
- BLAS/LAPACK 绑定（由上游库通过指针API集成）
- GPU 后端（当前版本不实现，但须预留扩展性）
- serde 序列化、arena 分配器
- 栈分配小数组

---

# 第二部分：核心架构

## 2. 维度与元素类型

### 2.1 维度系统

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

### 2.1.1 维度互转规则

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

### 2.2 元素类型体系

四层元素类型体系（上层继承下层，并额外提供能力）：

| 层次 | 类型 | Trait 约束 |
|------|------|------------|
| 基础层 | 整数、浮点、复数、bool | `Element` |
| 数值层 | 整数、浮点、复数（不含 bool） | `Numeric: Element`（在基础层之上支持四则运算） |
| 实数层 | f32, f64 | `RealScalar: Numeric`（在数值层之上提供数学函数） |
| 复数层 | Complex<f32>, Complex<f64> | `ComplexScalar: Numeric`（在数值层之上提供复数运算） |

> **约束**：bool 仅属于基础层，不实现 `Numeric`，不支持四则运算。不支持自动类型提升，类型转换须显式。

### 2.3 基础层（Element）

| 方法/常量 | 说明 |
|-----------|------|
| zero() | 加法单位元 |
| one() | 乘法单位元 |
| Copy + Clone | 值语义，可按位复制 |
| PartialEq | 相等比较 |
| Debug + Display | 格式化输出 |
| Send + Sync | 线程安全（用于并行迭代） |

### 2.4 数值层（Numeric）

继承 Element，额外约束四则运算。仅数值类型（整数、浮点、复数）实现，bool 不实现。

| 约束 | 说明 |
|------|------|
| Add<Output=Self> | 加法 |
| Sub<Output=Self> | 减法 |
| Mul<Output=Self> | 乘法 |
| Div<Output=Self> | 除法 |
| Neg<Output=Self> | 负号 |

### 2.5 实数层（RealScalar）

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

### 2.6 复数层（ComplexScalar）

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

### 2.7 NaN/Inf 处理约定

| 场景 | 行为 |
|------|------|
| 归约（sum/prod）含 NaN | 结果为 NaN |
| min/max 含 NaN | 任一参数为 NaN 则返回 NaN（NaN 传播语义） |
| 排序/比较含 NaN | NaN 不参与排序，PartialOrd 返回 None |
| Inf 参与算术 | 遵循 IEEE 754 规则 |
| 0.0 / 0.0 | 返回 NaN |
| 1.0 / 0.0 | 返回 Inf |

---

## 3. 复数类型

### 3.1 类型定义

- 自定义 Complex<T> 类型，`#[repr(C)]` 布局，不依赖外部 crate
- norm() 须使用 hypot 算法避免溢出

### 3.2 与实数的算术互操作

| 操作 | 行为 |
|------|------|
| Complex + 实数 | 实数隐式提升为 Complex(r, 0.0)，结果为 Complex。仅限同精度（Complex<f64> + f64） |
| 实数 + Complex | 同上，交换律成立 |
| Complex * 实数 | 标量乘法，等价于 (re*r, im*r)。仅限同精度 |
| Complex / 实数 | 标量除法，等价于 (re/r, im/r)。仅限同精度 |
| 实数 / Complex | 结果为 Complex，须正确处理分母共轭。仅限同精度 |
| Complex 与整数 | 不支持隐式互操作，须先将整数转为浮点 |
| 跨精度（如 f32 + Complex<f64>） | 不支持隐式互操作，须先显式 cast 到同精度 |

### 3.3 相等与比较

| 属性 | 要求 |
|------|------|
| PartialEq | 逐分量比较（re == re && im == im），NaN != NaN |
| Eq | 不实现（因 NaN） |
| PartialOrd | 不实现（复数无自然全序） |
| Ord | 不实现 |
| 近似相等 | 提供 approx_eq(other, epsilon) 方法，逐分量判断 |

### 3.4 内存与序列化

| 属性 | 要求 |
|------|------|
| 内存布局 | `#[repr(C)]`，保证 [re, im] 连续排列 |
| 大小 | size_of::<Complex<T>>() == 2 * size_of::<T>() |
| 对齐 | align_of::<Complex<T>>() == align_of::<T>() |
| 与 C 互操作 | 布局兼容 C99 \_Complex，可安全 transmute 指针 |
| 数组布局 | Complex<f64> 数组与交错实虚 f64 数组内存等价 |

---

## 4. 存储系统

### 4.1 存储模式

#### 4.1.1 访问级别

存储须区分三种访问级别：**只读**、**可写**、**拥有**（编译时保证）。

#### 4.1.2 四种存储模式

| 存储模式 | 拥有数据 | 可读 | 可写 | 克隆语义 | 分配方式 | 典型用途 |
|----------|---------|------|------|----------|----------|----------|
| Owned | 是 | 是 | 是 | 深拷贝 | 64 字节对齐堆分配 | 数组创建、运算结果 |
| ViewRepr | 否（借用） | 是 | 否 | 拷贝视图元数据（O(1)） | 无分配 | 切片、子数组只读访问 |
| ViewMutRepr | 否（独占借用） | 是 | 是 | 不可克隆（独占语义） | 无分配 | 原地修改子区域 |
| ArcRepr | 共享（Arc） | 是 | 通过 make_mut() 写时复制 | 浅拷贝（引用计数+1） | 写时按需分配 | 跨线程共享、延迟复制、函数参数传递 |

**ArcRepr::make_mut() 语义：**

| 属性 | 行为 |
|------|------|
| 写时复制 | 若引用计数 > 1，深拷贝数据到新分配，原 Arc 引用计数减 1 |
| 原子性保证 | 引用计数检查与递减为原子操作，多线程并发 make_mut() 不会导致数据竞争或重复拷贝 |
| 返回值 | 返回 `&mut [A]`，独占访问保证无其他引用可读取数据 |
| 分配对齐 | 新分配使用默认 64 字节对齐 |
| 性能提示 | 单线程场景下，先 clone() 再 make_mut() 可避免不必要的拷贝；多线程场景下，建议使用 Arc::try_unwrap() 或确保单写多读模式 |

#### 4.1.3 设备扩展性

Storage trait 预留 `type Device` 关联类型，当前版本仅支持 `Cpu`。GPU 支持计划在下一版本实现，优先级低。未来通过在 Storage trait 中实现不同的 Device 类型（如 `Cuda`）实现，无需修改 `TensorBase<S, D>` 签名。

---

## 5. 内存布局

### 5.1 布局规则

| 规则 | 要求 |
|------|------|
| 布局顺序 | 支持 F-order（列优先）和 C-order（行优先），默认 F-order |
| 步长类型 | 有符号类型，单位为元素个数（非字节），支持负步长 |
| 默认对齐 | 自有存储默认 64 字节对齐（AVX-512 缓存行） |
| 可配置对齐 | 构造时可指定对齐值（须为 2 的幂，≥ 元素自然对齐） |
| 填充 | 支持主维度填充以保证 SIMD 对齐，填充区域须零初始化 |

### 5.2 布局标志系统

采用 5 标志位设计（1 字节 u8），缓存高频查询的派生状态：

| 标志 | 信息 | 理由 |
|------|------|------|
| F_CONTIGUOUS | F-order 连续性 | 核心属性，高频查询 |
| C_CONTIGUOUS | C-order 连续性 | 核心属性，高频查询 |
| ALIGNED | SIMD 对齐（64 字节） | SIMD 路径选择依赖 |
| HAS_ZERO_STRIDE | 零步长（缓存） | 广播检测，避免 O(ndim) 遍历 |
| HAS_NEG_STRIDE | 负步长（缓存） | 反转检测，避免 O(ndim) 遍历 |
| — | 可写性 | **类型系统表达**（Owned/ViewMut 可写，View 只读） |
| — | 所有权 | **类型系统表达**（Owned/ArcRepr 拥有，View 不拥有） |
| — | 创建时布局顺序 | **删除**（可通过连续性推导） |

**组合语义：**
- `F_CONTIGUOUS | C_CONTIGUOUS`：标量或 1D 数组（两个方向都连续）
- `!F_CONTIGUOUS && !C_CONTIGUOUS`：非连续数组（切片/转置后常见）

**更新时机：**
- 创建时：根据分配方式初始化全部标志
- 切片/转置/reshape：重新计算全部标志
- 视图创建：继承源数组标志，按需降级（如对齐）

### 5.3 布局查询方法

| 方法 | 说明 | 复杂度 |
|------|------|--------|
| is_f_contiguous() | 是否 F-order 连续 | O(1) |
| is_c_contiguous() | 是否 C-order 连续 | O(1) |
| is_contiguous() | 任一方向连续即为 true | O(1) |
| is_aligned() | 是否 SIMD 对齐（64 字节） | O(1) |
| has_zero_stride() | 是否存在零步长（广播维度） | O(1) |
| has_neg_stride() | 是否存在负步长（反转维度） | O(1) |
| layout_flags() | 返回完整布局标志 | O(1) |

### 5.4 对齐策略

| 属性 | 要求 |
|------|------|
| 默认对齐 | 64 字节（AVX-512 缓存行） |
| 小数组优化 | 元素数 × 元素大小 ≤ 对齐值时，允许自动降级到元素自然对齐 |
| 对齐查询 | 提供 alignment() 方法返回当前数组的实际对齐值 |
| 视图对齐 | 视图继承源数组的对齐状态；切片后对齐可能降级（起始地址偏移），布局标志须反映实际对齐 |
| ArcRepr 对齐 | make_mut() 触发复制时，新分配使用默认对齐（64 字节） |

---

## 6. 张量核心抽象

### 6.1 泛型设计

核心数据结构为双参数泛型 `TensorBase<S, D>`：
- S：存储模式
- D：维度类型

#### TensorBase 内部组成

| 组件 | 说明 |
|------|------|
| storage: S | 底层数据存储（决定所有权与访问权限） |
| shape: D | 各轴长度 |
| strides: D | 各轴步长（有符号，单位为元素个数） |
| offset: usize | 数据起始偏移量（支持切片视图） |

### 6.2 类型别名体系

**主类型别名**

| 别名 | 展开 | 说明 |
|------|------|------|
| Tensor<A, D> | TensorBase<Owned<A>, D> | 拥有数据的数组 |
| TensorView<'a, A, D> | TensorBase<ViewRepr<&'a A>, D> | 不可变视图 |
| TensorViewMut<'a, A, D> | TensorBase<ViewMutRepr<&'a mut A>, D> | 可变视图 |
| ArcTensor<A, D> | TensorBase<ArcRepr<A>, D> | 原子引用计数共享（内置写时复制） |

**维度便捷别名**（以 Tensor 为例，其他存储模式同理）

| 别名 | 展开 | 说明 |
|------|------|------|
| Tensor0<A> | Tensor<A, Ix0> | 0 维标量 |
| Tensor1<A> | Tensor<A, Ix1> | 1 维向量 |
| Tensor2<A> | Tensor<A, Ix2> | 2 维矩阵 |
| Tensor3<A> | Tensor<A, Ix3> | 3 维 |
| TensorD<A> | Tensor<A, IxDyn> | 动态维度 |

---

# 第三部分：计算能力

## 7. 计算后端

### 7.1 SIMD

#### 7.1.1 基本配置

| 属性 | 要求 |
|------|------|
| 启用方式 | 通过 feature gate `simd` 启用，默认关闭 |
| 实现方式 | 使用 pulp crate 抽象 SIMD 指令集，不直接使用 std::arch |

#### 7.1.2 支持的指令集

| 指令集 | 架构 | 寄存器宽度 | 优先级 |
|--------|------|-----------|--------|
| AVX-512 | x86_64 | 512 bit | 最高 |
| AVX2 + FMA | x86_64 | 256 bit | 高 |
| SSE4.1 | x86_64 | 128 bit | 中 |
| NEON | aarch64 | 128 bit | 高（ARM 平台） |
| 标量回退 | 所有 | - | 最低 |

#### 7.1.3 SIMD 适用操作

| 操作类别 | 示例 | 要求 |
|----------|------|------|
| 逐元素一元 | abs, neg, sqrt, exp, ln | 连续内存或固定步长 |
| 逐元素二元 | add, sub, mul, div | 两操作数形状兼容 |
| 归约 | sum, prod, min, max | 连续内存优先 |
| 内积 | dot | 连续内存 |
| 比较 | eq, lt, gt（生成 mask） | 连续内存 |

#### 7.1.4 SIMD 对齐与回退

| 属性 | 要求 |
|------|------|
| 对齐要求 | 数据起始地址 64 字节对齐时使用对齐加载，否则使用非对齐加载 |
| 非连续数据 | 步长 ≠ 1 时回退到标量路径 |
| 尾部处理 | 元素数非 SIMD 宽度整数倍时，尾部使用标量处理 |
| 运行时检测 | 使用 pulp 的 `Arch::new()` 检测最佳指令集，通过 `arch.dispatch()` 分发到对应实现；`Arch` 为 `Copy` 类型，可在库初始化时缓存一次供后续复用，避免重复检测开销 |

### 7.2 并行

#### 7.2.1 基本配置

| 属性 | 要求 |
|------|------|
| 启用方式 | 通过 feature gate `parallel` 启用，默认关闭 |
| 实现方式 | 使用 rayon crate 实现数据并行 |

#### 7.2.2 并行策略

| 属性 | 要求 |
|------|------|
| 并行阈值 | 默认元素数 ≥ 64K 时启用并行，可通过全局配置或单次调用覆盖 |
| 支持并行的操作 | 逐元素运算、归约、map/mapv、zip 迭代 |
| 不支持并行的操作 | 矩阵乘法（由 BLAS 内部管理线程）、单元素操作、小数组操作 |
| 分块策略 | 按连续内存块分割，每块不小于 4K 元素 |
| 嵌套并行 | 禁止嵌套并行（内层操作强制单线程），避免线程池饥饿 |
| 线程数 | 默认使用 rayon 全局线程池，可通过自定义线程池覆盖 |

### 7.3 性能分层

运行时根据数据规模和内存布局自动选择执行路径：

| 条件 | 执行路径 |
|------|----------|
| 元素数 < SIMD 宽度 | 标量路径 |
| 元素数 ≥ SIMD 宽度 且 连续内存 且 simd 启用 | SIMD 路径 |
| 元素数 ≥ 并行阈值 且 parallel 启用 | SIMD + 并行路径（每线程内部使用 SIMD） |
| 非连续内存（步长 ≠ 1） | 标量路径（不论数据规模） |
| simd 未启用 且 元素数 ≥ 并行阈值 且 parallel 启用 | 标量 + 并行路径 |

---

## 8. 线程安全

### 8.1 Send/Sync 保证

| 存储模式 | Send | Sync | 条件 |
|----------|------|------|------|
| Owned | 是 | 是 | 元素类型为 Send+Sync |
| ViewRepr | 是 | 是 | 元素类型为 Sync |
| ViewMutRepr | 是 | 否 | 元素类型为 Send |
| ArcRepr | 是 | 是 | 元素类型为 Send+Sync |

**Send/Sync 语义说明：**

| 存储模式 | Send 条件解释 | Sync 条件解释 |
|----------|---------------|---------------|
| Owned | 元素可跨线程移动 | 元素可跨线程共享引用 |
| ViewRepr | `&T` 跨线程共享要求 T: Sync | `&&T` 跨线程共享要求 T: Sync |
| ViewMutRepr | `&mut T` 可跨线程移动（转移独占访问权），要求 T: Send | `&mut T` 不可共享（Rust 独占借用规则），故永远不是 Sync |
| ArcRepr | Arc 内部使用原子计数，要求 T: Send+Sync 才能跨线程共享 | 多个 Arc 可同时持有 &T，要求 T: Sync |

**ViewMutRepr: Send 的前提条件：**

- 元素类型须实现 `Send`（可安全跨线程移动）
- 视图转移后，原线程不再持有任何引用（独占语义保证）
- 不存在其他指向同一内存的引用（Rust 借用检查器保证）

### 8.2 并行迭代安全

| 规则 | 要求 |
|------|------|
| 访问隔离 | 并行迭代须保证各线程访问不重叠的元素区间 |
| 步长处理 | 非连续数组的并行迭代须正确处理步长跳跃，不得访问逻辑元素之外的内存 |
| 独占保证 | 可变并行迭代须保证独占访问（无别名） |

### 8.3 Padding 字节并发规则

| 规则 | 要求 |
|------|------|
| 禁止访问 | Padding 字节不属于任何逻辑元素，任何线程不得读写 padding 字节（初始化时除外） |
| 禁止分配 | 并行迭代不得将 padding 字节分配给任何线程的工作区间 |
| 禁止暴露 | 视图切片不得暴露 padding 字节为可访问元素 |

---

# 第四部分：API 规范

## 9. 迭代器

### 9.1 迭代类型

| 类型 | 说明 |
|------|------|
| 元素迭代 | 按内存布局顺序 |
| 轴迭代 | 沿指定轴 |
| 窗口迭代 | 滑动窗口 |
| 索引迭代 | 带索引的元素迭代 |
| 并行迭代 | rayon 支持 |
| 多数组同步迭代 | zip |
| 遍历顺序 | 可指定 F/C |

### 9.2 迭代器协议

| 场景 | 行为 |
|------|------|
| zip 形状完全一致 | 按指定遍历顺序同步产出元素 |
| zip 形状不一致但可广播 | 自动广播后迭代，等价于先 broadcast 再 zip |
| zip 形状不一致且不可广播 | 返回 BroadcastError |
| 遍历顺序未指定 | 默认按物理内存布局顺序（连续数组最优） |
| 遍历顺序指定为 F/C | 强制按指定逻辑顺序，可能非连续访问 |
| 并行迭代 | 将元素按连续块分割给线程，块大小 ≥ 并行阈值 |
| 窗口迭代越界 | 不产出不完整窗口（窗口数 = shape - window_size + 1） |
| 空数组迭代 | 立即结束，产出零个元素 |

---

## 10. 运算操作

### 10.1 逐元素运算

> **约束**：逐元素算术运算（add, sub, mul, div, 三角函数, 指数/对数, 数值函数）仅适用于数值类型（整数、浮点、复数），不适用于 bool。bool 类型仅支持逻辑运算（`all/any`、位运算、条件选择）。

| 类别 | 操作 |
|------|------|
| 算术 | add, sub, mul, div |
| 三角函数 | sin, cos, tan, asin, acos, atan |
| 指数/对数 | exp, ln, log2, log10 |
| 数值函数 | abs, sign, floor, ceil, round, square, reciprocal, pow |

### 10.2 矩阵运算

#### 10.2.1 支持的操作

- matvec、dot/inner、outer
- 批量：batch_matvec、batch_dot、batch_add、batch_scale
- **不包含** GEMM

#### 10.2.2 批量运算维度约定

| 约定 | 规则 |
|------|------|
| batch 轴位置 | 最前面的轴为 batch 轴（轴 0, 1, ..., ndim-3 为 batch，最后 2 维为矩阵/向量） |
| batch 形状约束 | 所有操作数的 batch 维度须形状一致或可广播 |
| batch 广播 | batch 维度遵循 NumPy 广播规则，结果 batch 形状为广播后形状 |
| batch_matvec | 输入：(..., M, N) × (..., N) → 输出：(..., M) |
| batch_dot | 输入：(..., N) × (..., N) → 输出：(...) |
| batch_add/scale | 输入形状须一致或可广播，逐元素操作 |

### 10.3 归约

#### 10.3.1 归约类型

| 类型 | 操作 |
|------|------|
| 全局 | sum, prod, mean, var, std, min, max, argmin, argmax, all, any |
| 沿轴 | 以上所有运算均支持沿指定轴归约 |
| 累积 | cumsum, cumprod（沿指定轴，返回同形状数组） |

**var/std 统计定义**：默认有偏估计（除以 N，即 ddof=0），与 NumPy 默认行为一致。

**整数归约溢出行为**：`sum/prod` 作用于整数数组时，溢出将 panic（debug 和 release 模式均如此），与 Rust 默认的 checked 算术一致。

**cumsum/cumprod 边界行为**：遇到 NaN 时传播 NaN（后续元素均为 NaN）；空数组返回同形状空数组。

**argmin/argmax 多值行为**：存在多个相同最小/最大值时，返回第一个出现的索引（按遍历顺序），与 NumPy/ndarray 行为一致。

#### 10.3.2 集合操作

| 操作 | 说明 | 输入类型约束 | 返回类型 |
|------|------|--------------|----------|
| unique | 返回唯一值（排序后） | Element | Tensor1<A> |
| unique_counts | 返回唯一值及出现次数 | Element | (Tensor1<A>, Tensor1<usize>) |
| unique_inverse | 返回唯一值及原数组索引 | Element | (Tensor1<A>, Tensor1<usize>) |
| bincount | 统计非负整数出现次数 | 整数类型（u8/u16/u32/u64/i8/i16/i32/i64，值须 ≥ 0） | Tensor1<usize> |
| histogram | 统计落入各 bin 的元素数 | RealScalar | Tensor1<usize> |
| histogram_bin_edges | 返回 bin 边界 | RealScalar | Tensor1<A> |

**unique 语义**：

| 属性 | 行为 |
|------|------|
| 排序 | 返回的唯一值按升序排列 |
| 空数组 | 返回空 Tensor1 |
| 返回值 | Tensor1<A>，长度等于唯一值数量 |

**unique_counts 语义**：

| 属性 | 行为 |
|------|------|
| 返回值 | (values, counts)，values 为唯一值（升序），counts 为对应出现次数 |
| 约束 | values.len() == counts.len() |

**unique_inverse 语义**：

| 属性 | 行为 |
|------|------|
| 返回值 | (values, inverse)，values 为唯一值（升序），inverse 为原数组每个元素在 values 中的索引 |
| 约束 | inverse.len() == input.len()，inverse[i] ∈ [0, values.len()) |
| 重建原数组 | values[inverse[i]] == input[i] |

**bincount 语义**：

| 属性 | 行为 |
|------|------|
| 输入约束 | 仅支持整数类型，所有值须 ≥ 0 |
| minlength 参数 | 指定输出最小长度，若最大值+1 < minlength，则输出长度为 minlength |
| 空数组 | 返回长度为 minlength（默认 0）的全零数组 |
| 负值输入 | panic（运行时检查） |
| 返回值 | Tensor1<usize>，长度为 max(input) + 1（或 minlength），output[i] = count of i in input |
| 权重参数 | 可选 weights: Tensor1<A>，output[i] = sum of weights[j] where input[j] == i |

**histogram 语义**：

| 属性 | 行为 |
|------|------|
| bins 参数 | 整数（等宽 bin 数）或 Tensor1<A>（自定义 bin 边界） |
| range 参数 | (min, max) 元组，指定统计范围，超出范围的值不计入 |
| 空数组 | 返回全零数组 |
| 返回值 | Tensor1<usize>，长度等于 bin 数 |
| bin 边界规则 | 左闭右开 [left, right)，最后一个 bin 为 [left, right] |

### 10.4 广播

#### 10.4.1 基本规则

- 遵循 NumPy 广播规则（右对齐，size-1 维度拉伸）
- 广播视图为只读
- 算术运算符隐式支持广播

#### 10.4.2 广播细节

| 规则 | 说明 |
|------|------|
| 维度对齐 | 从最右维度开始对齐，维度数不足的数组在左侧补 1 |
| 兼容条件 | 对应维度相等，或其中一个为 1 |
| 零步长语义 | size-1 维度广播时步长设为 0，逻辑上重复该维度的单个元素 |
| 广播视图可写性 | 广播产生的视图始终为只读；不允许对广播视图进行写操作 |
| 原地广播运算 | `a += b` 中 b 可被广播，a 不可被广播（a 须拥有完整存储） |
| 标量广播 | 标量视为 0 维数组，可与任意维度数组广播 |

---

## 11. 形状操作

### 11.1 操作分类

| 类型 | 操作 |
|------|------|
| 零拷贝 | reshape, transpose, slice, squeeze, unsqueeze, permute, broadcast, swapaxes, moveaxis, split/chunk, index_axis, unstack |
| 需拷贝 | cat, stack, pad, repeat/tile |
| 视情况 | flatten |

### 11.2 index_axis 语义

| 属性 | 行为 |
|------|------|
| 操作 | `index_axis(axis, index)` 沿指定轴取单个切片，返回降维视图（ndim - 1） |
| 内存布局 | 返回的视图共享源数组底层存储 |
| 对齐继承 | 视图的实际对齐取决于偏移后的起始地址，布局标志须反映实际对齐状态 |
| BLAS 兼容性 | 从 3D batch tensor 中沿最外层 batch 轴索引取出的 2D 视图，若源数组 F-contiguous 则保持 F-contiguous 和原 LDA |
| 连续性 | 沿最外层轴（F-order 下为最后一轴）索引时保持连续性；沿内层轴索引可能导致非连续 |

### 11.3 unstack 语义

| 属性 | 行为 |
|------|------|
| 操作 | `unstack(axis)` 沿指定轴拆分为 n 个降维视图（ndim - 1），n 为该轴长度 |
| 返回类型 | Vec<TensorView>，长度等于指定轴的 size |
| 内存布局 | 每个视图共享源数组底层存储，零拷贝 |
| 与 index_axis 关系 | `unstack(axis)[i]` 等价于 `index_axis(axis, i)` |
| 空轴 | 轴长度为 0 时返回空 Vec |

### 11.4 split/chunk 语义

| 属性 | 行为 |
|------|------|
| split(axis, indices) | 沿指定轴按索引列表分割，返回 Vec<TensorView>，零拷贝 |
| chunk(axis, n_chunks) | 沿指定轴均匀分割为 n 块；若轴长度不能整除，前 (len % n) 块各多 1 个元素 |
| 返回类型 | Vec<TensorView>，每个视图共享源数组底层存储 |
| n_chunks = 0 | 返回空 Vec |
| n_chunks > 轴长度 | 返回轴长度个大小为 1 的块（多余的 chunk 数被忽略） |
| 空轴 | 轴长度为 0 时返回 n_chunks 个空视图 |

### 11.5 pad 语义

| 属性 | 行为 |
|------|------|
| pad(widths, mode) | 沿各轴两侧填充，返回新 Owned 数组（需拷贝） |
| widths 参数 | 每轴一对 (before, after)，指定前后填充宽度 |
| mode: Constant(value) | 用指定常量填充 |
| mode: Edge | 用边缘元素重复填充 |
| mode: Reflect | 镜像反射填充（不含边缘元素） |
| 零宽度填充 | 等价于拷贝，不改变形状 |

### 11.6 repeat/tile 语义

| 属性 | 行为 |
|------|------|
| repeat(reps) | 沿各轴重复指定次数，返回新 Owned 数组（需拷贝） |
| reps 参数 | 每轴一个重复次数；reps 长度不足时左侧补 1（与 NumPy np.tile 一致） |
| reps 含 0 | 对应轴长度变为 0，结果为空数组 |
| reps 全为 1 | 等价于拷贝 |

---

## 12. 索引操作

### 12.1 索引类型

| 类型 | 说明 |
|------|------|
| 多维索引 | `[i, j, k]` 形式，i, j, k 为元素索引 |
| 范围索引 | 切片语法 |
| 切片宏 | `s![]` 形式, 语法参考 ndarray 的 `s![]` 宏 |
| 高级索引 | take, take_along_axis, mask, compress, put, argwhere/nonzero |
| 条件选择 | where(condition, x, y) |

### 12.2 where 语义

| 属性 | 行为 |
|------|------|
| 操作 | `where(condition, x, y)` 按条件逐元素选择，condition 为 true 取 x，否则取 y |
| condition 类型 | bool 数组（或可广播为目标形状的 bool 数组） |
| 广播 | condition、x、y 三者须形状一致或可广播，结果形状为广播后形状 |
| 返回类型 | 新分配的 Owned 数组 |
| 惰性求值 | 不支持；x 和 y 均完整求值 |

---

## 13. 构造与转换

### 13.1 构造方法

| 方法 | 说明 |
|------|------|
| zeros/ones/full/empty | 支持指定 Order |
| eye/identity/diag | 单位矩阵和对角矩阵 |
| from_vec/from_slice/from_fn | 从数据源构造 |
| arange/linspace/logspace | 序列生成 |

### 13.2 运算符重载

- 四则运算
- 复合赋值
- 一元运算（Neg, Not）
- PartialEq
- 所有二元运算符隐式支持广播

### 13.3 实用操作

| 操作 | 说明 |
|------|------|
| copy_to, fill | 复制和填充 |
| is_close/allclose | 近似比较 |
| clip | 裁剪 |
| flip/flipud/fliplr | 翻转 |
| to_owned/into_owned | 转换为拥有 |
| map/mapv/mapv_inplace | 映射操作，语义参考 ndarray 实现 |
| cast | 显式类型转换 |

### 13.4 连续性保证方法

返回类型始终为 `Tensor<A, D>`，保证连续布局。即使输入已连续，也返回新分配的 Owned 数组（数据拷贝）。

| 方法 | 行为 |
|------|------|
| to_f_contiguous() | 返回 F-contiguous 布局的新 Tensor（若输入已 F-contiguous 则等价于 to_owned()） |
| to_c_contiguous() | 返回 C-contiguous 布局的新 Tensor（若输入已 C-contiguous 则等价于 to_owned()） |
| to_contiguous() | 返回连续布局的新 Tensor：优先 F-contiguous，若输入已 C-contiguous 则保持 C-contiguous |

**设计理由**：统一返回 Owned 类型，避免引入 Cow 语义，简化 API。调用方若需零拷贝优化，可先调用 `is_f_contiguous()` / `is_c_contiguous()` 检查后自行处理。

### 13.5 cast 精度行为

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

### 13.6 标准类型转换

- Vec、切片、数组到张量的 From 转换
- 自有到视图的自动借用
- 静态维度到动态维度的转换

### 13.7 格式化输出

- Debug/Display，NumPy 风格，大数组省略

---

## 14. FFI 集成

### 14.1 指针 API

| 方法 | 说明 |
|------|------|
| as_ptr() | 返回数据起始位置的不可变原始指针 `*const A` |
| as_mut_ptr() | 返回数据起始位置的可变原始指针 `*mut A`（须可写存储） |
| as_ptr_unchecked() | unsafe，不检查偏移量有效性 |

### 14.2 形状与步长查询

| 方法 | 说明 |
|------|------|
| shape() | 返回各轴长度的切片 `&[usize]` |
| strides() | 返回各轴步长的切片（元素单位，有符号） |
| strides_bytes() | 返回各轴步长的切片（字节单位，有符号） |
| offset() | 返回数据起始偏移量（元素单位） |

### 14.3 BLAS 兼容性

| 方法 | 说明 |
|------|------|
| lda() | 返回 leading dimension（F-order 下为第一轴步长），仅 2D |
| is_blas_compatible() | 检查内存布局是否可直接传递给 BLAS（F-contiguous 或 C-contiguous，正步长，无零步长） |
| blas_layout() | 返回 BLAS 布局标识（F/C/None），None 表示不兼容 |
| blas_trans() | 返回 BLAS Trans 参数（N/T/C），基于当前布局与目标布局的关系 |

### 14.4 索引转换

| 方法 | 说明 |
|------|------|
| index_to_ptr(index) | 将多维索引转换为对应元素的原始指针 |
| index_to_offset(index) | 将多维索引转换为相对于数据起始位置的元素偏移量 |

### 14.5 原始部件构造与解构

| 方法 | 说明 |
|------|------|
| from_raw_parts(ptr, shape, strides, offset) | unsafe，从原始部件构造 TensorView |
| from_raw_parts_mut(ptr, shape, strides, offset) | unsafe，从原始部件构造 TensorViewMut |
| into_raw_parts() | 消耗数组，返回 (ptr, shape, strides, offset) 元组 |

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

## 15. 临时工作空间

### 15.1 工作空间属性

| 属性 | 要求 |
|------|------|
| 对齐 | 默认 64 字节对齐，须支持调用方指定更大对齐（如 128 字节） |
| 生命周期 | 借用语义：工作空间借出期间不可再次借出，归还后可复用 |
| 增长策略 | 请求超出当前容量时自动扩容，不缩容；扩容后保持对齐 |
| 线程亲和性 | 工作空间不绑定线程；线程安全由调用方通过借用规则保证 |
| 零初始化 | 不保证零初始化（性能考虑），调用方须自行初始化 |

### 15.2 分割与嵌套

| 属性 | 要求 |
|------|------|
| 嵌套借用 | 支持从同一工作空间分割多个不重叠的子缓冲区 |
| 并行分割 | 支持 `split_at(mid)` 将工作空间分割为两个不重叠的子工作空间，各自可独立借出，O(1)，不涉及内存分配 |
| 递归分割 | 子工作空间可继续分割（支持递归二分） |

### 15.3 Scratch 查询 API

上游库（如线性代数库）须能在执行操作前查询所需临时内存大小，以便预分配或复用工作空间。

| 属性 | 要求 |
|------|------|
| scratch_size 查询 | 提供静态方法，根据操作类型和矩阵尺寸返回所需字节数 |
| 返回类型 | 返回结构化的内存需求描述（大小 + 对齐），而非裸 usize |
| 可组合性 | 多个 scratch 需求可合并（顺序执行取 max，并行执行取 sum） |
| 与工作空间集成 | 工作空间可接受 scratch 需求描述，一次性分配足够空间 |
| 零运行时开销 | scratch 查询为纯计算，不分配内存 |
| 上游库用法 | 上游库定义自己的操作枚举并实现 scratch 查询；Xenon 仅提供基础设施（需求描述类型 + 合并逻辑 + 工作空间分配接口） |

---

# 第五部分：工程保障

## 16. 错误处理

### 16.1 错误分类

| 错误类型 | 触发场景 | 处理方式 |
|----------|----------|----------|
| ShapeMismatch | 二元运算/zip 形状不兼容且无法广播 | Result |
| BroadcastError | 广播规则不满足（非 size-1 维度不等） | Result |
| LayoutMismatch | 要求连续布局但输入非连续（如 reshape 非连续数组） | Result |
| InvalidAxis | 轴索引超出维度数 | Result |
| InvalidShape | reshape 目标元素总数与源不一致 | Result |
| DimensionMismatch | 静态维度与动态维度互转时维度数不匹配 | Result |
| IndexOutOfBounds | 多维索引越界 | panic（checked）/ UB（unchecked） |
| EmptyArray | 对空数组执行 min/max/argmin/argmax | Result |

### 16.2 处理策略

| 策略 | 适用场景 |
|------|----------|
| 可恢复错误（形状、布局、轴） | 返回 Result |
| 编程错误（索引越界） | panic，同时提供 unsafe unchecked 变体 |
| 错误信息 | 所有错误类型须实现 `Display` 和 `Error`，包含上下文信息（期望值 vs 实际值） |

---

## 17. 质量要求

### 17.1 文档

| 要求 | 范围 |
|------|------|
| doc comment | 所有公开 API |
| Safety 文档节 | 所有 unsafe 函数 |
| 使用示例 | 关键 API |

### 17.2 测试

| 要求 | 指标 |
|------|------|
| 测试类型 | 单元测试 + 集成测试 + 边界测试 |
| 行覆盖率 | ≥ 80% |

### 17.3 数值精度

| 运算类别 | f64 精度 | f32 精度 |
|----------|----------|----------|
| 加减乘 | 精确 | 精确 |
| 归约 | 1e-15 | 1e-6 |
| 超越函数 | 1e-14 | 1e-5 |

### 17.4 边界覆盖

须覆盖以下边界情况：
- 空张量
- 单元素
- 大张量
- 极端值（NaN/Inf/subnormal）
- 非连续布局
- 高维（≥4维）

---

*Xenon v18 — 需求说明书*
