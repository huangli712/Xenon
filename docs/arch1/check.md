# 规范 A（01-coding-standards.md）问题清单

> 基于 `require-v18.md` 需求说明书逐条对照审查

---

## 一、与需求文档的矛盾（Critical）

### 1.1 错误类型命名完全偏离需求

- **现状**：§4.2 定义 `ShapeError { IncompatibleShape, IncompatibleSize, IncompatibleOrder }`
- **需求**（§16.1）：应定义 8 种错误类型 — `ShapeMismatch`, `BroadcastError`, `LayoutMismatch`, `InvalidAxis`, `InvalidShape`, `DimensionMismatch`, `IndexOutOfBounds`, `EmptyArray`
- **问题**：错误枚举名称和变体与需求完全不匹配；缺少 `BroadcastError`, `LayoutMismatch`, `InvalidAxis`, `EmptyArray` 等关键错误类型

### 1.2 num-traits 依赖违规

- **现状**：§1.3 示例使用 `Numeric: Element + Num`（`Num` 来自 num-traits）；§9.2 Cargo.toml 显式声明 `num-traits = { version = "0.2", default-features = false }`
- **需求**（§1.3）："最小依赖原则：仅允许 rayon（可选）和 pulp（可选）作为外部依赖"
- **问题**：num-traits 是被禁止的重型依赖

### 1.3 F-order 默认布局未提及

- **现状**：全文未提及 F-order（列优先）为默认布局
- **需求**（§5.1）："布局顺序：支持 F-order（列优先）和 C-order（行优先），默认 F-order"
- **问题**：科学计算库的默认布局是最基础的架构决策，完全遗漏

### 1.4 有符号步长未提及

- **现状**：所有步长示例使用 `Vec<Ix>` / `usize`，无法表示负步长
- **需求**（§5.1）："步长类型：有符号类型，单位为元素个数（非字节），支持负步长"
- **问题**：步长类型选择直接影响 reshape/transpose/flip 等操作的正确性

### 1.5 布局标志系统完全缺失

- **现状**：无任何关于布局标志的讨论
- **需求**（§5.2）：5 标志位设计 — `F_CONTIGUOUS`, `C_CONTIGUOUS`, `ALIGNED`, `HAS_ZERO_STRIDE`, `HAS_NEG_STRIDE`
- **问题**：布局标志是 O(1) 查询的核心缓存机制，影响广播检测、SIMD 路径选择、BLAS 兼容性等关键路径

### 1.6 ArcRepr/ArcTensor 存储模式缺失

- **现状**：§3.4 仅展示 `OwnedRepr<A>` 和 `ViewRepr` 两种存储
- **需求**（§4.1.2）：四种存储模式 — Owned, ViewRepr, ViewMutRepr, ArcRepr（含 make_mut() 写时复制）
- **问题**：ArcRepr 的引用计数共享和写时复制是跨线程共享的核心机制

### 1.7 Complex<T> 自定义复数类型缺失

- **现状**：无任何关于复数类型的讨论
- **需求**（§3.1-3.4）：自定义 Complex<T>，`#[repr(C)]` 布局，不依赖外部 crate，C99 互操作
- **问题**：复数类型是科学计算的基线需求

### 1.8 四层元素类型体系未详述

- **现状**：§1.3 仅在示例中提及 `Element`, `Numeric`, `RealScalar` trait 名，无约束定义
- **需求**（§2.2-2.6）：Element（基础层）→ Numeric（数值层）→ RealScalar（实数层）/ ComplexScalar（复数层），各层有明确的 trait 约束
- **问题**：缺少 bool 仅属于 Element 不实现 Numeric 的约束说明；缺少各层完整的方法/常量列表

### 1.9 FFI 指针 API 完全缺失

- **现状**：无任何 FFI 相关 API 规范
- **需求**（§14.1-14.5）：`as_ptr()`, `as_mut_ptr()`, `as_ptr_unchecked()`, `from_raw_parts()`, `from_raw_parts_mut()`, `into_raw_parts()`, `lda()`, `is_blas_compatible()`, `blas_layout()`, `blas_trans()`, `strides_bytes()`, `offset()`, `index_to_ptr()`, `index_to_offset()`
- **问题**：FFI API 是上游库零开销集成的基础

### 1.10 临时工作空间（Workspace）缺失

- **现状**：无任何关于临时工作空间的讨论
- **需求**（§15.1-15.3）：对齐分配、借用语义、分割与嵌套、scratch_size 查询 API
- **问题**：工作空间是上游库（如线性代数库）集成的基础设施

### 1.11 TensorBase 缺少 offset 字段

- **现状**：§1.6 和 §3.4 示例中 TensorBase 仅含 `storage, dim, strides`
- **需求**（§6.1）：`storage: S, shape: D, strides: D, offset: usize`
- **问题**：缺少 offset 字段意味着切片视图无法表示数据起始偏移

### 1.12 维度互转规则缺失

- **现状**：无任何维度互转规则的讨论
- **需求**（§2.1.1）：静态→动态（总是成功）和动态→静态（可能失败返回 DimensionMismatch）的完整转换规则
- **问题**：维度互转是 reshape 和类型转换的基础

### 1.13 NaN/Inf 处理约定缺失

- **现状**：§7.5 仅有 `test_nan_handling` 和 `test_inf_handling` 两个测试，无处理规范
- **需求**（§2.7）：NaN 传播语义（sum/prod 含 NaN 结果为 NaN；min/max 任一参数为 NaN 则返回 NaN；排序中 NaN 不参与）
- **问题**：NaN 语义是科学计算库正确性的基本保障

### 1.14 Send/Sync 表格缺失

- **现状**：无任何 Send/Sync 规范
- **需求**（§8.1）：四种存储模式各自的 Send/Sync 条件及详细语义说明
- **问题**：并行迭代的安全性依赖于正确的 Send/Sync 实现

### 1.15 广播机制规范缺失

- **现状**：无任何广播机制的讨论
- **需求**（§10.4）：NumPy 广播规则、零步长语义、广播视图只读、原地广播运算约束、标量广播

### 1.16 BLAS 兼容性 API 缺失

- **现状**：无任何 BLAS 相关内容
- **需求**（§14.3）：`lda()`, `is_blas_compatible()`, `blas_layout()`, `blas_trans()`

### 1.17 运算符重载规范缺失

- **现状**：无运算符重载的指导
- **需求**（§13.2）：四则运算、复合赋值、一元运算、PartialEq，所有二元运算符隐式支持广播

### 1.18 类型别名体系缺失

- **现状**：§1.2 仅展示 `type Tensor<A, D> = TensorBase<OwnedRepr<A>, D>`
- **需求**（§6.2）：完整别名体系 — `TensorView`, `TensorViewMut`, `ArcTensor`，以及 `Tensor0`~`Tensor6`, `TensorD` 便捷别名

### 1.19 计算后端（SIMD/并行）规范缺失

- **现状**：§9.1 和 §9.3 提及 feature gate 但无计算策略
- **需求**（§7.1-7.3）：SIMD 指令集优先级、并行阈值（64K）、分块策略、性能分层自动选择

### 1.20 线程安全规范缺失

- **现状**：无任何线程安全讨论
- **需求**（§8.1-8.3）：Send/Sync 表格、并行迭代安全、Padding 字节并发规则

---

## 二、规范内部的技术错误（High）

### 2.1 no_std 下 Debug impl 编译冲突

- **位置**：§4.2
- **问题**：`#[derive(Debug)]` 已生成 `impl Debug for ShapeError`，又在 `#[cfg(not(feature = "std"))]` 下手动 `impl core::fmt::Debug for ShapeError`，两者在 no_std 编译时同时存在，导致重复 impl 编译错误
- **修复建议**：移除 `#[derive(Debug)]`，在 std 和 no_std 下分别手动实现 Debug；或统一使用 `#[derive(Debug)]` 并在 no_std 下不覆盖

### 2.2 sum() 返回类型自相矛盾

- **位置**：§4.4 展示 `sum() -> Result<A, EmptyError>`，§7.5 展示 `arr.sum() == 0`（空数组直接返回零值）
- **问题**：一处暗示空数组是错误（返回 Result），另一处暗示空数组是正常情况（返回零值）
- **需求**（§16.1）：`EmptyArray` 错误仅适用于 min/max/argmin/argmax，sum 不在其中
- **修复建议**：sum() 返回 `A`（非 Result），空数组返回 `A::zero()`

### 2.3 expect() 禁令与 panic 策略矛盾

- **位置**：§4.4 "禁止使用 unwrap()、expect()"
- **问题**：§4.1 允许对"编程错误"使用 panic（含 assert_eq!），而 expect() 与 assert_eq! 在语义上等价（都是"满足前置条件否则 panic"）。此外 require-v18 §10.3.1 要求整数归约溢出 panic，实现中必然使用 expect 或 checked_*().expect()
- **修复建议**：区分"禁止在可恢复错误路径上使用 expect/unwrap"与"允许在断言前置条件时使用 expect"，或明确"库代码中仅允许 expect() 用于断言不变量/前置条件"

### 2.4 `as` 类型转换禁令过严

- **位置**：§3.1 "禁止使用 as 进行数值类型转换"
- **问题**：require-v18 §13.5 要求 `cast` 方法实现浮点→整数、整数→浮点等转换，底层实现不可避免地使用 `as`（或 `to_bits`/`from_bits` 等更底层的方式）。完全禁止 `as` 不切实际
- **修复建议**：将禁令改为"禁止在公开 API 中直接使用 as 进行数值转换；内部实现中使用 as 时须注释说明安全性"

### 2.5 std/no_std 条件编译的 to_vec 示例无意义

- **位置**：§9.1
- **问题**：
  ```rust
  #[cfg(feature = "std")]
  pub fn to_vec(&self) -> Vec<A> { ... }
  #[cfg(not(feature = "std"))]
  pub fn to_vec(&self) -> alloc::vec::Vec<A> { ... }
  ```
  `Vec<A>` 和 `alloc::vec::Vec<A>` 是同一个类型，条件编译无必要
- **修复建议**：统一使用 `alloc::vec::Vec<A>` 或 `Vec<A>`（通过 `use alloc::vec::Vec`）

### 2.6 Element trait 多余约束

- **位置**：§1.3 `pub trait Element: Copy + Clone`
- **问题**：`Copy` 已经蕴含 `Clone`，同时写两个是冗余的
- **修复建议**：`pub trait Element: Copy` 或参考 require-v18 §2.3 的完整约束列表（Copy + Clone + PartialEq + Debug + Display + Send + Sync）

---

## 三、覆盖不足（Medium）

### 3.1 测试覆盖率要求缺失

- **现状**：无覆盖率要求
- **需求**（§17.2）："行覆盖率 ≥ 80%"

### 3.2 高维测试缺失

- **现状**：§7.5 边界用例未包含高维测试
- **需求**（§17.4）："高维（≥4维）"

### 3.3 大张量测试缺失

- **现状**：无大张量测试
- **需求**（§17.4）："大张量"

### 3.4 subnormal 浮点值测试缺失

- **现状**：无极值测试
- **需求**（§17.4）："极端值（NaN/Inf/subnormal）"中的 subnormal 未覆盖

### 3.5 数值精度规范缺失

- **现状**：无精度要求
- **需求**（§17.3）：f64/f32 在加减乘、归约、超越函数的精度指标

### 3.6 归约溢出行为规范缺失

- **现状**：无整数归约溢出行为说明
- **需求**（§10.3.1）："sum/prod 作用于整数数组时，溢出将 panic（debug 和 release 模式均如此）"

### 3.7 ViewMutRepr 不可克隆语义未提及

- **现状**：无
- **需求**（§4.1.2）：ViewMutRepr "不可克隆（独占语义）"

### 3.8 Storage trait 的 Device 关联类型未预留

- **现状**：§3.3 Storage trait 定义无 Device 关联类型
- **需求**（§4.1.3）：Storage trait 预留 `type Device` 关联类型，当前版本仅支持 `Cpu`

### 3.9 广播视图只读约束未提及

- **现状**：无
- **需求**（§10.4.2）："广播产生的视图始终为只读；不允许对广播视图进行写操作"

### 3.10 迭代器遍历顺序规范缺失

- **现状**：无
- **需求**（§9.1-9.2）：按内存布局顺序、可指定 F/C、默认按物理内存布局顺序

### 3.11 集合操作（unique/bincount/histogram）未提及

- **现状**：无
- **需求**（§10.3.2）：unique, unique_counts, unique_inverse, bincount, histogram, histogram_bin_edges

### 3.12 形状操作（split/chunk/pad/repeat/tile/unstack）未提及

- **现状**：无
- **需求**（§11.1-11.6）：零拷贝操作和需拷贝操作的完整分类

### 3.13 高级索引操作（take/mask/compress/put/argwhere）未提及

- **现状**：无
- **需求**（§12.1）：take, take_along_axis, mask, compress, put, argwhere/nonzero, where(condition, x, y)

### 3.14 构造函数完整列表缺失

- **现状**：§1.8 有少量方法示例但无系统化构造方法列表
- **需求**（§13.1）：zeros/ones/full/empty, eye/identity/diag, from_vec/from_slice/from_fn, arange/linspace/logspace

### 3.15 连续性保证方法缺失

- **现状**：无
- **需求**（§13.4）：`to_f_contiguous()`, `to_c_contiguous()`, `to_contiguous()` 及其语义

### 3.16 cast 精度行为缺失

- **现状**：§3.1 仅说"禁止 as"，无替代方案的精度行为规范
- **需求**（§13.5）：完整的转换精度行为表（浮点→整数、整数→浮点、NaN→整数等）

### 3.17 Debug/Display 格式化输出规范缺失

- **现状**：无
- **需求**（§13.7）：NumPy 风格，大数组省略

### 3.18 安全规范的 unsafe impl Send/Sync 文档要求缺失

- **现状**：§5 仅讨论 unsafe fn 和 unsafe 块
- **需求**（§8.1）：涉及 unsafe impl Send/Sync 的 Safety 文档

### 3.19 实用操作缺失

- **现状**：无
- **需求**（§13.3）：copy_to, fill, is_close/allclose, clip, flip/flipud/fliplr, map/mapv/mapv_inplace

---

## 四、规范质量改进建议（Low）

### 4.1 CI 缺少行覆盖率检查

- **位置**：附录 D CI 检查清单
- **建议**：添加 `cargo tarpaulin` 或 `cargo llvm-cov` 检查，断言 ≥80% 覆盖率

### 4.2 CI 缺少 miri / sanitizer

- **建议**：添加 `cargo miri test` 用于检测 unsafe 代码中的 UB

### 4.3 §4.2 Display impl 中的错误信息使用 `{:?}` 格式化 Vec

- **位置**：`write!(f, "incompatible shapes: expected {:?}, got {:?}", expected, actual)`
- **建议**：`{:?}` 输出的是 Debug 格式（`[1, 2, 3]`），对用户不够友好。考虑使用自定义格式（如 `(1, 2, 3)` 或 `1×2×3`）

### 4.4 生命周期命名约定仅支持短名

- **位置**：§1.7 "生命周期使用短名：'a, 'b, 'c"
- **建议**：对于复杂的生命周期关系（如 `View<'a, A, D>` 中 `'a` 表示数据借用），应允许在类型签名中添加注释说明生命周期含义

### 4.5 §6.5 lint 列表可补充

- **现状**：`missing_docs`, `missing_debug_implementations`, `rust_2024_compatibility`
- **建议**：考虑添加 `#![deny(unsafe_op_in_unsafe_fn)]`（Rust 2024 edition 默认已启用，但显式声明更清晰）

### 4.6 rustfmt.toml 缺少 `comment_width` 配置

- **位置**：附录 A
- **现状**：有 `wrap_comments = true` 但未设置 `comment_width`，默认值可能导致注释行过长或过短

---

## 统计

| 类别 | 数量 |
|------|------|
| Critical（与需求矛盾） | 20 |
| High（内部技术错误） | 6 |
| Medium（覆盖不足） | 19 |
| Low（质量改进） | 6 |
| **总计** | **51** |
