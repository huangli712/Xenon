# 类型转换模块设计

> 文档编号: 21 | 模块: `src/convert/` | 阶段: Phase 4
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
└── convert/                 # 类型转换（目录模块）
    ├── mod.rs               # 模块根，re-exports
    ├── cast.rs              # CastTo trait、cast() 方法、类型转换路径
    ├── owned.rs             # to_owned、into_owned、存储模式互转
    ├── from_impl.rs         # From/TryFrom trait 实现
    └── contiguous.rs        # to_f_contiguous 辅助函数（公共 API to_contiguous 定义于 20-utility-ops.md）
```

多文件设计：按转换职责拆分，便于后续扩展（如新增转换路径、存储模式等）。

### 2.1 文件职责

| 文件 | 职责 | 预估行数 |
|------|------|----------|
| `mod.rs` | 模块根，re-exports 所有公共类型 | ~20 |
| `cast.rs` | `CastTo<T>` trait 定义、所有类型转换 impl、`cast()` 方法 | ~200 |
| `owned.rs` | `to_owned()`、`into_owned()`、存储模式互转（view/view_mut/into_shared） | ~100 |
| `from_impl.rs` | `From<Vec<A>>`、`From<&[A]>`、`From<[A; N]>`、`From<&Tensor> for View` 等 | ~80 |
| `contiguous.rs` | `to_contiguous()` 连续化转换 | ~50 |

---

## 3. 依赖关系

### 3.1 依赖图

```
src/convert/
├── mod.rs          # re-exports: CastTo, cast, to_owned, into_owned, From impls
├── cast.rs         # 依赖 element (CastTo), tensor (TensorBase)
├── owned.rs        # 依赖 tensor (TensorBase), storage, layout
├── from_impl.rs    # 依赖 tensor (Tensor/TensorView), construct (from_vec)
└── contiguous.rs   # 依赖 tensor (TensorBase), layout

外部依赖:
├── crate::tensor        # TensorBase<S, D>, Tensor, TensorView
├── crate::dimension     # Dimension trait
├── crate::storage       # Storage, StorageMut, StorageOwned trait
├── crate::element       # Element, CastTo trait
├── crate::layout        # is_f_contiguous 查询
└── crate::error         # XenonError, Result<T>
```

### 3.2 类型级依赖

| 来源模块 | 使用的类型/trait |
|----------|-----------------|
| `tensor` | `TensorBase<S, D>`, `Tensor<A, D>`, `.shape()`, `.strides()`, `.memory_order()`（参见 `07-tensor.md` §4） |
| `dimension` | `Dimension`, `Ix0`~`Ix6`, `IxDyn`（参见 `02-dimension.md` §4） |
| `storage` | `Storage<Elem=A>`, `StorageMut`, `Owned<A>`, `ViewRepr`, `ViewMutRepr`, `ArcRepr`（参见 `05-storage.md` §4） |
| `element` | `Element`, `CastTo<B>`（参见 `03-element-types.md` §4.8） |
| `layout` | `is_f_contiguous()`（参见 `06-memory-layout.md` §4） |
| `error` | `XenonError`, `Result<T>`（参见 `26-error-handling.md` §4） |

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
impl<A, D> TensorBase<Owned<A>, D>
where
    D: Dimension,
    A: Element,
{
    /// Element-wise type conversion.
    ///
    /// Only applicable to the `Owned` storage mode (per requirement §23).
    /// For other storage modes, call `to_owned()` first: `view.to_owned().cast::<B>()`.
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
        self.mapv(|x| x.cast_to())  // mapv 参见 `11-elementwise-ops.md` §4.1
    }
}
```

> **设计决策（修订）**：根据需求说明书 §23，`cast()` 仅适用于持有数据的存储模式。
> 因此 `cast()` 分别在 `Owned<A>` 和 `ArcRepr<A>` 存储储模式上实现，
> `ViewRepr`/`ViewMutRepr` 须先调用 `to_owned()`。

```rust
impl<A, D> TensorBase<ArcRepr<A>, D>
where
    D: Dimension,
    A: Element,
{
    /// Element-wise type conversion for Arc-backed tensors.
    ///
    /// Makes a private copy of the Arc data, then casts elements.
    ///
    /// # Type Parameters
    ///
    /// * `B` - Target element type
    pub fn cast<B>(&self) -> Tensor<B, D>
    where
        B: Element,
        A: CastTo<B>,
    {
        self.to_owned().cast()
    }
}
```

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
| `Complex<f32>` | `Complex<f64>` | 实部/虚部各自提升精度 | `Complex { re: 1.0_f32, im: 2.0_f32 } → Complex { re: 1.0_f64, im: 2.0_f64 }` |
| `Complex<f64>` | `Complex<f32>` | 实部/虚部各自 round-to-nearest-even | `Complex { re: 1.23456789, im: 2.0 } → Complex { re: 1.234568, im: 2.0 }` |
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
        // Always produce F-order (Xenon only supports F-order, see requirement §7).
        // Iterate in F-order and collect into a new aligned allocation.
        let shape = self.shape();
        let mut data: Vec<A> = Vec::with_capacity(self.len());
        for &elem in self.iter() {
            data.push(elem);
        }
        // from_vec_aligned: copies into a 64-byte aligned allocation (see 05-storage.md §5.1)
        Tensor::from_shape_vec_aligned(self.raw_dim(), data)
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

### 4.5b from_vec_aligned — 对齐构造辅助

`from_vec_aligned` 是 `Owned<A>` 的内部辅助方法，将 `Vec<A>` 的数据复制到 64 字节对齐的新分配中（参见 `05-storage.md §5.1`）。

```rust
// Defined in src/storage/owned.rs (see 05-storage.md §5.1)
impl<A> Owned<A> {
    /// Creates Owned by copying data into a 64-byte aligned allocation.
    ///
    /// The user's original Vec allocation is discarded; a fresh aligned allocation
    /// is made and data is copied over.
    pub fn from_vec_aligned(data: Vec<A>) -> Self where A: Clone {
        let len = data.len();
        // Allocate aligned memory, copy elements, return
        // ... (implementation detail: uses AlignedAlloc + Vec::from_raw_parts)
    }
}
```

`from_shape_vec_aligned` 是 `Tensor` 上的对应便捷方法（无需验证长度，调用方已保证）：

```rust
impl<A, D> TensorBase<Owned<A>, D> where A: Element, D: Dimension {
    /// Constructs a Tensor from pre-validated data with aligned allocation.
    /// Called internally after size validation is complete.
    pub(crate) fn from_shape_vec_aligned(shape: D, data: Vec<A>) -> Self {
        let strides = shape.strides_for_f_order();
        let storage = Owned::from_vec_aligned(data);
        TensorBase { storage, shape, strides, offset: 0, flags: LayoutFlags::from_order(Order::F) }
    }
}
```

### 4.5c to_f_contiguous — 非连续数据重排

`to_f_contiguous()` 将任意布局的张量重排为 F-order 连续拷贝，供 `to_contiguous()` 在非连续情况下调用（参见 `20-utility-ops.md §4.3`）：

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + Clone,
{
    /// Copies data into a new F-contiguous owned tensor.
    ///
    /// Iterates elements in F-order and re-packs them into a fresh aligned allocation.
    /// Always returns an F-contiguous `Tensor<A, D>`.
    ///
    /// This method is called by `to_contiguous()` when the source is non-contiguous.
    pub fn to_f_contiguous(&self) -> Tensor<A, D> {
        let mut data = Vec::with_capacity(self.len());
        // iter() traverses in F-order (see 10-iterator.md §5.1 fast/slow paths)
        for &elem in self.iter() {
            data.push(elem);
        }
        Tensor::from_shape_vec_aligned(self.raw_dim(), data)
    }
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
}  // Complex 结构体定义参见 `04-complex-type.md` §4

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

## 6. 实现任务拆分

### Wave 1: 基础设施

- [ ] **T1**: 定义 `CastTo` trait 及核心转换实现
  - 文件: `src/convert/cast.rs`
  - 内容: `CastTo<T>` trait 定义，f64↔f32、f64→i32、i32→f64 等基础 impl
  - 测试: `test_cast_f64_to_i32`, `test_cast_f32_to_f64`, `test_cast_nan_to_int`
  - 前置: element 模块完成
  - 预计: 15 min

- [ ] **T2**: 创建 `convert/` 模块骨架
  - 文件: `src/convert/mod.rs`, `src/lib.rs`
  - 内容: 子模块声明、`pub use` 导出
  - 测试: 编译通过
  - 前置: T1
  - 预计: 5 min

### Wave 2: 核心方法

- [ ] **T3**: 实现 `to_owned` / `into_owned` 及存储模式互转
  - 文件: `src/convert/owned.rs`
  - 内容: `to_owned()` 克隆方法、`into_owned()` 消费方法、view/view_mut/into_shared
  - 测试: `test_to_owned_from_view`, `test_into_owned_from_tensor`, `test_into_owned_from_arc`
  - 前置: T2, tensor 模块完成
  - 预计: 15 min

- [ ] **T4**: 实现 `cast` 方法
  - 文件: `src/convert/cast.rs`
  - 内容: `cast<B>(&self) -> Tensor<B, D>` 方法实现
  - 测试: `test_cast_f64_to_f32`, `test_cast_i32_to_f64`, `test_cast_overflow`
  - 前置: T2, tensor 模块完成
  - 预计: 10 min

- [ ] **T5**: 实现 `to_contiguous` 连续化转换
  - 文件: `src/convert/contiguous.rs`
  - 内容: `to_contiguous()` 方法实现
  - 测试: `test_to_contiguous_from_view`, `test_to_contiguous_already_contiguous`
  - 前置: T2, tensor 模块完成
  - 预计: 10 min

### Wave 3: From trait 实现

- [ ] **T6**: 实现标准库 `From` trait
  - 文件: `src/convert/from_impl.rs`
  - 内容: `From<Vec<A>>`, `From<&[A]>`, `From<[A; N]>`, `From<&Tensor> for View` 等
  - 测试: `test_from_vec`, `test_from_slice`, `test_from_array`, `test_from_tensor_view`
  - 前置: T3
  - 预计: 10 min

- [ ] **T7**: 扩展 CastTo 实现（整数↔整数、bool↔数值、实数→复数）
  - 文件: `src/convert/cast.rs`
  - 内容: 窄化整数饱和、bool↔数值、实数→复数转换
  - 测试: `test_cast_int_narrowing`, `test_cast_bool_to_int`, `test_cast_real_to_complex`
  - 前置: T1
  - 预计: 15 min

### 并行执行图

```
Wave 1: [T1] ──▶ [T2]
                    │
Wave 2: [T3] [T4] [T5]  (并行)
             │
Wave 3: [T6] [T7]  (并行)
```

---

## 7. 测试计划

### 7.1 单元测试清单

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

### 7.2 边界测试场景

| 场景 | 预期行为 |
|------|----------|
| 空张量 `cast` | 返回空张量，形状不变 |
| 单元素 `cast` | 正确转换唯一元素 |
| 非连续 View `cast` | 正确处理步长跳转 |
| i64::MAX → i32 | 饱和到 i32::MAX |
| f64::NAN → f32 | 传播 NaN |
| `0.0 → bool` | `false` |

### 7.3 属性测试不变量

| 不变量 | 测试方法 |
|--------|----------|
| `cast().shape() == original.shape()` | 随机形状 |
| `cast::<A>().cast::<A>()` 往返近似相等（浮点） | 随机数据 |
| `to_owned().shape() == view.shape()` | 随机形状 |

---

## 8. 与其他模块的交互

| 交互模块 | 方向 | 说明 |
|----------|------|------|
| `tensor` | convert → tensor | `cast()` 定义在 `TensorBase<S, D>` 的 impl 块上，消费 `.shape()`、`.memory_order()`；`to_owned()`/`into_owned()` 消费 `Storage`/`StorageIntoOwned`（参见 `07-tensor.md` §4） |
| `element` | convert → element | 泛型约束 `A: CastTo<B>` 驱动逐元素转换；`CastTo` trait 定义在 element 模块（参见 `03-element-types.md` §4） |
| `elementwise_ops` | convert → elementwise_ops | `cast()` 内部调用 `mapv()` 执行逐元素映射（参见 `11-elementwise-ops.md` §4.1） |
| `storage` | convert → storage | `into_owned()` 消费 `StorageIntoOwned` trait；存储模式互转依赖 `Owned`/`ViewRepr`/`ArcRepr`（参见 `05-storage.md` §4） |
| `layout` | convert → layout | `to_owned()` 调用 `is_f_contiguous()` 判断是否需要重排（参见 `06-memory-layout.md` §4） |
| `complex` | convert → complex | `CastTo<Complex<T>>` 实现依赖 `Complex` 结构体定义；反向转换（Complex → T）故意不提供（参见 `04-complex-type.md` §4） |

---

## 9. 设计决策记录

### 决策 1：溢出语义选择

| 属性 | 值 |
|------|-----|
| 决策 | 浮点→整数使用 truncate + saturating 语义 |
| 理由 | 与 NumPy 的 `astype` 行为一致；NaN → 0 避免未定义行为；饱和避免 UB |
| 替代方案 | wrapping（Rust `as` 默认） — 放弃，科学计算中 wrapping 语义不直观 |
| 替代方案 | panic on overflow — 放弃，性能不可预测 |

### 决策 2：cast() 仅在持有数据的存储模式上实现

| 属性 | 值 |
|------|-----|
| 决策 | `cast()` 在 `Owned<A>` 和 `ArcRepr<A>` 上实现 |
| 理由 | 需求 §23 明确要求"类型转换仅适用于持有数据的存储模式"。View 和 ViewMut 需要先调用 `to_owned()` 获得拥有所有权的张量，再进行类型转换。 |
| 替代方案 | 在 `Storage` 约束上实现（允许 View 直接 cast） — 放弃，与需求 §23 冲突 |
| 替代方案 | 仅在 `Owned` 上实现 — 放弃，`ArcRepr` 也是持有数据的存储模式，应同样支持 |

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

## 11. no_std 兼容性

| 依赖 | 来源 | no_std |
|------|------|:------:|
| `CastTo` trait | 自定义 | ✅ |
| `From`/`Into` | `core::convert` | ✅ |
| `TensorBase` | 本 crate | ✅ |
| `alloc::vec::Vec` | alloc crate | ✅ |

所有转换逻辑仅依赖 `core` 和 `alloc`，不依赖 `std`。存储模式的 `no_std` 兼容性参见 `05-storage.md` §11。

---

## 版本历史

| 版本 | 日期 |
|------|------|
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-08 |
| 1.0.2 | 2026-04-08 |
| 1.1.0 | 2026-04-08 |
| 1.1.1 | 2026-04-08 |
| 1.2.0 | 2026-04-08 |
| 1.2.1 | 2026-04-08 |

---

*本文档由 Xenon 维护。如有问题请提交 Issue 或 PR。*
