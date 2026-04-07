# 类型转换模块设计

> 文档编号: 21 | 模块: `src/convert.rs` | 阶段: Phase 4
> 前置文档: `07-tensor.md`, `03-element-types.md`
> 需求参考: 需求说明书 §23

---

## 1. 模块定位

### 1.1 职责边界

| 职责 | 包含 | 不包含 |
|------|------|--------|
| 逐元素类型转换 | `cast<B>(&self) -> Tensor<B, D>` | 隐式类型提升（需显式调用） |
| 同类型拷贝 | `to_owned`、`into_owned` | 跨模块转换逻辑（如 reshape） |
| 存储模式互转 | Owned ↔ ViewRepr ↔ ArcRepr | 隐式 Deref 转换 |
| 标准库接口 | `From`/`Into`/`TryFrom`/`TryInto` 实现 | 非标准转换（如序列化） |

### 1.2 设计原则

| 原则 | 体现 |
|------|------|
| 显式转换 | 所有类型转换须显式调用 `cast()`，无隐式提升 |
| 溢出安全 | 整数截断遵循饱和语义（saturating），浮点→整数 NaN 返回 0 |
| 存储约束 | `cast` 仅适用于持有数据的存储模式（Owned/Arc），View 须先 `to_owned` |
| no_std 兼容 | 所有转换逻辑不依赖 `std`，仅使用 `core` 和 `alloc` |

### 1.3 在架构中的位置

```
依赖层级：
L0: error, private
L1: dimension, element, complex
L2: layout (依赖 dimension)
L3: storage (依赖 layout)
L4: tensor (依赖 storage, dimension)
L5: convert  ← 当前模块
```

---

## 2. 文件位置

```
src/
└── convert.rs    # 类型转换（cast/to_owned/存储模式互转/From trait）
```

单文件设计：类型转换功能内聚性高，代码量适中（~400 行），无需拆分。

---

## 3. 依赖关系

### 3.1 依赖图

```
src/convert.rs
├── crate::tensor        # TensorBase<S, D>, Tensor, TensorView
├── crate::dimension     # Dimension trait
├── crate::storage       # Storage, StorageMut, StorageOwned trait
├── crate::element       # Element, CastTo trait
├── crate::layout        # MemoryOrder, 连续性查询
└── crate::error         # ShapeError
```

### 3.2 类型级依赖

| 来源模块 | 使用的类型/trait |
|----------|-----------------|
| `tensor` | `TensorBase<S, D>`, `Tensor<A, D>`, `.shape()`, `.strides()`, `.memory_order()` |
| `dimension` | `Dimension`, `Ix0`~`Ix6`, `IxDyn` |
| `storage` | `Storage<Elem=A>`, `StorageMut`, `Owned<A>`, `ViewRepr`, `ViewMutRepr`, `ArcRepr` |
| `element` | `Element`, `CastTo<B>` |
| `layout` | `MemoryOrder`, `is_f_contiguous()`, `is_c_contiguous()` |
| `error` | `ShapeError` |

### 3.3 依赖方向声明

> **依赖方向：单向向上。** `convert` 仅消费 `tensor`、`storage` 等核心模块，不被它们依赖。

---

## 4. 公共 API 设计

### 4.1 CastTo trait

```rust
/// Element-wise type conversion trait.
///
/// Defines explicit type conversion rules from `Self` to `T`.
/// Overflow behavior is explicitly defined per implementation (see §4.3).
pub trait CastTo<T> {
    /// Performs the type conversion.
    fn cast_to(self) -> T;
}
```

### 4.2 cast 方法

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element,
{
    /// Element-wise type conversion.
    ///
    /// Only applicable to data-owning storage modes (Owned/Arc).
    /// View types must call `to_owned()` before `cast()`.
    ///
    /// # Type Parameters
    ///
    /// * `B` - Target element type
    ///
    /// # Examples
    ///
    /// ```
    /// let a = Tensor1::from_vec(vec![1.5, 2.7, 3.9]);
    /// let b: Tensor1<i32> = a.cast();
    /// // [1, 2, 3]  — truncate toward zero
    /// ```
    pub fn cast<B>(&self) -> Tensor<B, D>
    where
        B: Element,
        A: CastTo<B>,
    {
        self.mapv(|x| x.cast_to())
    }
}
```

> **设计决策：** `cast` 定义在 `Storage` 约束（而非仅 `StorageOwned`）上，这是因为 View 上的 cast 在语义上合理（只读操作）。调用方可通过类型约束限制为 Owned。

### 4.3 类型转换路径表

| 源类型 | 目标类型 | 转换行为 | 示例 |
|--------|----------|----------|------|
| `f64` | `f32` | round-to-nearest-even | `1.23456789_f64 → 1.234568_f32` |
| `f32` | `f64` | 精确（无损） | `1.5_f32 → 1.5_f64` |
| `f32/f64` | `i32/i64` | truncate + saturating | `1.9 → 1`, `NaN → 0` |
| `f32/f64` (NaN) | 整数 | 返回 0 | `f64::NAN → 0i32` |
| `f32/f64` (+Inf) | 整数 | 饱和到 MAX | `f64::INFINITY → i32::MAX` |
| `f32/f64` (-Inf) | 整数 | 饱和到 MIN | `f64::NEG_INFINITY → i32::MIN` |
| 整数 | `f32/f64` | round-to-nearest-even | `123456789i32 → f32 可能不精确` |
| 整数 → 整数（窄化） | 整数 | saturating | `300u16 → u8::MAX (255)` |
| 整数 → 整数（扩展） | 整数 | 零扩展/符号扩展 | `u8 → u16` 零扩展 |
| `bool` | 数值 | `true → 1`, `false → 0` | `true → 1i32` |
| 数值 | `bool` | 非零 → `true`，零 → `false` | `0 → false`, `1 → true` |
| `f32/f64` | `Complex<f32/f64>` | 虚部为 0 | `1.5 → Complex { re: 1.5, im: 0.0 }` |
| `Complex<T>` | `T` | **编译错误** | 须显式取 `.re()` |

### 4.4 Good / Bad 对比

```rust
// Good - explicit cast, intent is clear
let a: Tensor<f64, Ix1> = Tensor::from_vec(vec![1.5, 2.7]);
let b: Tensor<i32, Ix1> = a.cast();

// Bad - implicit type promotion (Xenon does not support this)
let c: Tensor<i32, Ix1> = a + 1;  // Compile error: type mismatch

// Good - explicit handling of complex to real
let complex_t: Tensor<Complex<f64>, Ix1> = /* ... */;
let re_parts: Tensor<f64, Ix1> = complex_t.mapv(|c| c.re);

// Bad - attempting to directly cast complex to real
let real_t: Tensor<f64, Ix1> = complex_t.cast();  // Compile error: CastTo not implemented
```

### 4.5 to_owned / into_owned

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + Clone,
{
    /// Clones data into a new owned tensor.
    ///
    /// Always allocates new memory and copies data, even if input is already Owned.
    ///
    /// # Examples
    ///
    /// ```
    /// let view: TensorView<f64, Ix1> = tensor.view();
    /// let owned: Tensor<f64, Ix1> = view.to_owned();
    /// ```
    pub fn to_owned(&self) -> Tensor<A, D> {
        let mut result = Tensor::zeros_order(self.shape().clone(), self.memory_order());
        self.copy_to(&mut result);
        result
    }
}

impl<S, D, A> TensorBase<S, D>
where
    S: StorageIntoOwned<Elem = A>,
    D: Dimension,
    A: Element,
{
    /// Consumes the tensor, converting to owned.
    ///
    /// - `Tensor`: returned directly, O(1)
    /// - `TensorView`/`TensorViewMut`: copies data, O(n)
    /// - `ArcTensor`: returned directly if ref count is 1, otherwise copied
    pub fn into_owned(self) -> Tensor<A, D>;
}
```

### 4.6 存储模式互转

| 源 → 目标 | 操作 | 复杂度 |
|------------|------|--------|
| Owned → ViewRepr | 借用（`view()`） | O(1) |
| Owned → ViewMutRepr | 可变借用（`view_mut()`） | O(1) |
| Owned → ArcRepr | Arc 包装（`into_shared()`） | O(1) |
| ArcRepr → Owned | `make_mut()` 或 clone | O(1) 或 O(n) |
| ViewRepr → Owned | `to_owned()`（拷贝） | O(n) |
| ViewMutRepr → Owned | `to_owned()`（拷贝） | O(n) |

### 4.7 标准库类型转换接口

```rust
// Vec<A> → Tensor<A, Ix1>
impl<A: Element> From<Vec<A>> for Tensor<A, Ix1> {
    fn from(vec: Vec<A>) -> Self { Self::from_vec(vec) }
}

// &[A] → Tensor<A, Ix1>
impl<A: Element + Clone> From<&[A]> for Tensor<A, Ix1> {
    fn from(slice: &[A]) -> Self { Self::from_slice(slice) }
}

// [A; N] → Tensor<A, Ix1>
impl<A: Element, const N: usize> From<[A; N]> for Tensor<A, Ix1> {
    fn from(arr: [A; N]) -> Self { Self::from_vec(arr.into_iter().collect()) }
}

// &Tensor → TensorView
impl<'a, A: Element, D: Dimension> From<&'a Tensor<A, D>> for TensorView<'a, A, D> {
    fn from(tensor: &'a Tensor<A, D>) -> Self { tensor.view() }
}

// &mut Tensor → TensorViewMut
impl<'a, A: Element, D: Dimension> From<&'a mut Tensor<A, D>> for TensorViewMut<'a, A, D> {
    fn from(tensor: &'a mut Tensor<A, D>) -> Self { tensor.view_mut() }
}
```

---

## 5. 内部实现设计

### 5.1 CastTo 实现（核心转换路径）

```rust
// === Float → Float ===
impl CastTo<f32> for f64 {
    #[inline]
    fn cast_to(self) -> f32 { self as f32 }  // IEEE 754 round-to-nearest-even
}

impl CastTo<f64> for f32 {
    #[inline]
    fn cast_to(self) -> f64 { self as f64 }  // Exact conversion
}

// === Float → Integer (truncate + saturating) ===
impl CastTo<i32> for f64 {
    #[inline]
    fn cast_to(self) -> i32 {
        if self.is_nan() { return 0; }
        if self >= i32::MAX as f64 { return i32::MAX; }
        if self <= i32::MIN as f64 { return i32::MIN; }
        self.trunc() as i32
    }
}

// === Integer → Integer (saturating) ===
impl CastTo<i32> for i64 {
    #[inline]
    fn cast_to(self) -> i32 { self.clamp(i32::MIN as i64, i32::MAX as i64) as i32 }
}

// === Real → Complex ===
impl CastTo<Complex<f64>> for f64 {
    #[inline]
    fn cast_to(self) -> Complex<f64> { Complex { re: self, im: 0.0 } }
}

// Note: CastTo<f64> for Complex<f64> is intentionally not implemented
// Users must explicitly use .re() or .im()
```

### 5.2 溢出行为汇总

| 输入值 | 目标类型 | 输出值 | 说明 |
|--------|----------|--------|------|
| `f64::NAN` | `i32` | `0` | NaN → 整数 = 0 |
| `f64::NAN` | `f32` | `f32::NAN` | NaN 在浮点间传播 |
| `f64::INFINITY` | `i32` | `i32::MAX` | +Inf 饱和到 MAX |
| `f64::NEG_INFINITY` | `i32` | `i32::MIN` | -Inf 饱和到 MIN |
| `300i32` | `u8` | `255` | 窄化整数饱和 |

---

## 6. no_std 兼容性

| 依赖 | 来源 | no_std |
|------|------|:------:|
| `CastTo` trait | 自定义 | ✅ |
| `From`/`Into` | `core::convert` | ✅ |
| `TensorBase` | 本 crate | ✅ |
| `alloc::vec::Vec` | alloc crate | ✅ |

所有转换逻辑仅依赖 `core` 和 `alloc`，不依赖 `std`。

---

## 7. 实现任务拆分

### Wave 1: 基础设施

- [ ] **T1**: 定义 `CastTo` trait 及核心转换实现
  - 文件: `src/convert.rs`
  - 内容: `CastTo<T>` trait 定义，f64↔f32、f64→i32、i32→f64 等基础 impl
  - 测试: `test_cast_f64_to_i32`, `test_cast_f32_to_f64`, `test_cast_nan_to_int`
  - 前置: element 模块完成
  - 预计: 15 min

- [ ] **T2**: 创建 `convert.rs` 模块骨架
  - 文件: `src/convert.rs`, `src/lib.rs`
  - 内容: 模块声明、`pub use` 导出、`cast()` 方法骨架
  - 测试: 编译通过
  - 前置: T1, tensor 模块完成
  - 预计: 5 min

### Wave 2: 核心方法

- [ ] **T3**: 实现 `to_owned` / `into_owned`
  - 文件: `src/convert.rs`
  - 内容: `to_owned()` 克隆方法、`into_owned()` 消费方法
  - 测试: `test_to_owned_from_view`, `test_into_owned_from_tensor`, `test_into_owned_from_arc`
  - 前置: T2
  - 预计: 10 min

- [ ] **T4**: 实现 `cast` 方法
  - 文件: `src/convert.rs`
  - 内容: `cast<B>(&self) -> Tensor<B, D>` 方法实现
  - 测试: `test_cast_f64_to_f32`, `test_cast_i32_to_f64`, `test_cast_overflow`
  - 前置: T2
  - 预计: 10 min

### Wave 3: From trait 实现

- [ ] **T5**: 实现标准库 `From` trait
  - 文件: `src/convert.rs`
  - 内容: `From<Vec<A>>`, `From<&[A]>`, `From<[A; N]>`, `From<&Tensor> for View` 等
  - 测试: `test_from_vec`, `test_from_slice`, `test_from_array`, `test_from_tensor_view`
  - 前置: T3
  - 预计: 10 min

- [ ] **T6**: 扩展 CastTo 实现（整数↔整数、bool↔数值、实数→复数）
  - 文件: `src/convert.rs`
  - 内容: 窄化整数饱和、bool↔数值、实数→复数转换
  - 测试: `test_cast_int_narrowing`, `test_cast_bool_to_int`, `test_cast_real_to_complex`
  - 前置: T1
  - 预计: 15 min

### 并行执行图

```
Wave 1: [T1] [T2]
            │
Wave 2: [T3] [T4]
            │
Wave 3: [T5] [T6]
```

---

## 8. 测试计划

### 8.1 单元测试清单

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_cast_f64_to_f32` | f64→f32 精度损失符合 IEEE 754 | 高 |
| `test_cast_f32_to_f64` | f32→f64 无损转换 | 高 |
| `test_cast_float_to_int_truncate` | 浮点→整数截断行为 | 高 |
| `test_cast_nan_to_int` | NaN → 0 | 高 |
| `test_cast_inf_to_int` | ±Inf → MAX/MIN 饱和 | 高 |
| `test_cast_int_narrowing` | 整数窄化饱和 | 高 |
| `test_cast_bool_roundtrip` | bool↔数值往返正确 | 中 |
| `test_cast_real_to_complex` | 实数→复数虚部为 0 | 中 |
| `test_to_owned_from_view` | View → Owned 数据一致 | 高 |
| `test_to_owned_from_arc` | Arc → Owned 正确复制 | 高 |
| `test_into_owned_tensor` | Owned → Owned 零拷贝 | 高 |
| `test_from_vec` | Vec → Tensor1 | 高 |
| `test_from_slice` | &[A] → Tensor1 | 中 |
| `test_from_tensor_view` | &Tensor → TensorView | 高 |

### 8.2 边界测试场景

| 场景 | 预期行为 |
|------|----------|
| 空张量 `cast` | 返回空张量，形状不变 |
| 单元素 `cast` | 正确转换唯一元素 |
| 非连续 View `cast` | 正确处理步长跳转 |
| i64::MAX → i32 | 饱和到 i32::MAX |
| f64::NAN → f32 | 传播 NaN |
| `0.0 → bool` | `false` |

### 8.3 属性测试不变量

| 不变量 | 测试方法 |
|--------|----------|
| `cast().shape() == original.shape()` | 随机形状 |
| `cast::<A>().cast::<A>()` 往返近似相等（浮点） | 随机数据 |
| `to_owned().shape() == view.shape()` | 随机形状 |

---

## 9. 设计决策记录(ADR)

### 决策 1：溢出语义选择

| 属性 | 值 |
|------|-----|
| 决策 | 浮点→整数使用 truncate + saturating 语义 |
| 理由 | 与 NumPy 的 `astype` 行为一致；NaN → 0 避免未定义行为；饱和避免 UB |
| 替代方案 | wrapping（Rust `as` 默认） — 放弃，科学计算中 wrapping 语义不直观 |
| 替代方案 | panic on overflow — 放弃，性能不可预测 |

### 决策 2：cast 定义在 Storage 而非 StorageOwned 上

| 属性 | 值 |
|------|-----|
| 决策 | `cast()` 的约束为 `S: Storage`（只读即可） |
| 理由 | View 上的 cast 在语义上合理（只读操作）；调用方可通过类型约束限制 |
| 替代方案 | 限制为 `StorageOwned` — 放弃，强制调用方先 `to_owned()` 增加样板代码 |

### 决策 3：存储模式转换策略

| 属性 | 值 |
|------|-----|
| 决策 | 提供显式方法（`view()`, `view_mut()`, `into_shared()`），不使用 `Into` trait |
| 理由 | 显式方法命名更清晰，避免隐式行为；`From` 仅用于标准库接口 |
| 替代方案 | 为所有模式对实现 `From` — 放弃，组合爆炸（N×N 对） |

---

## 10. 性能考量

| 操作 | 时间复杂度 | 空间复杂度 | 说明 |
|------|-----------|-----------|------|
| `cast` | O(n) | O(n) | 新分配一个张量 |
| `to_owned` | O(n) | O(n) | 总是拷贝 |
| `into_owned`（Owned） | O(1) | O(1) | 直接返回 |
| `into_owned`（View） | O(n) | O(n) | 拷贝 |
| `view()` | O(1) | O(1) | 仅元数据 |
| `into_shared()` | O(1) | O(1) | Arc 包装 |

---

## 版本历史

| 版本 | 日期 |
|------|------|
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-08 |

---

*本文档由 Xenon 维护。如有问题请提交 Issue 或 PR。*
