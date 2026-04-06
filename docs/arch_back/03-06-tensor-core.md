# 张量核心抽象模块设计文档

> **模块路径**: `src/tensor/`
> **版本**: v18
> **日期**: 2026-03-28
> **前置文档**: 02-project-architecture.md, 03-02-element-type-system.md

---

## 1. 模块概述

### 1.1 定位

`TensorBase<S, D>` 是 Senon 库的中心数据结构，定义了所有张量类型的统一抽象。它是一个双参数泛型结构体：

- **S（存储模式）**: 决定数据的所有权语义和访问权限
- **D（维度类型）**: 决定数组的维度数和形状表示方式

### 1.2 设计理念

| 原则 | 说明 |
|------|------|
| 零开销抽象 | 不同存储模式在运行时无额外开销 |
| 类型安全 | 通过泛型约束在编译期保证访问权限 |
| 统一接口 | 所有张量类型共享相同的核心 API |
| 最小核心 | 核心结构仅包含必要字段，功能通过扩展方法提供 |

### 1.3 核心角色

```
┌─────────────────────────────────────────────────────────────┐
│                     TensorBase<S, D>                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ storage: S    ← 底层数据（所有权/借用）              │   │
│  │ shape: D      ← 形状描述                            │   │
│  │ strides: D    ← 步长（有符号，元素单位）             │   │
│  │ offset: usize ← 数据起始偏移量                       │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
         ↓ 类型别名展开
┌────────────────┐ ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
│ Tensor<A,D>    │ │TensorView<A,D> │ │TensorViewMut   │ │ ArcTensor<A,D> │
│ Owned<A>       │ │ ViewRepr<&A>   │ │ ViewMutRepr    │ │ ArcRepr<A>     │
└────────────────┘ └────────────────┘ └────────────────┘ └────────────────┘
```

---

## 2. 文件结构

```
src/tensor/
├── mod.rs             # TensorBase<S, D> 结构体定义 + 公开导出
├── impls.rs           # 核心查询方法实现
├── aliases.rs         # 类型别名定义
└── construct.rs       # 内部构造方法（unsafe 底层构造）
```

### 2.1 各文件职责

| 文件 | 职责 | 可见性 |
|------|------|--------|
| `mod.rs` | 结构体定义、模块导出、公开 API 重新导出 | pub |
| `impls.rs` | 所有核心查询方法（shape、strides、view 等） | pub（通过 re-export） |
| `aliases.rs` | 类型别名定义 | pub |
| `construct.rs` | unsafe 构造方法（from_raw_parts 等） | pub（unsafe 方法） |

---

## 3. TensorBase<S, D> 结构体设计

### 3.1 完整定义

```rust
/// 多维数组的核心抽象。
///
/// # 类型参数
///
/// * `S` - 存储模式，决定所有权和访问权限
/// * `D` - 维度类型，决定维度数和形状表示
///
/// # 内存布局
///
/// 结构体大小取决于 S 和 D 的具体实例化。对于静态维度（Ix0-Ix6），
/// D 为栈分配的固定大小数组；对于动态维度（IxDyn），D 包含堆分配的 Vec。
#[repr(C)]
pub struct TensorBase<S, D> {
    /// 底层数据存储。
    ///
    /// 存储模式决定了：
    /// - 所有权：拥有数据（Owned/ArcRepr）还是借用（View/ViewMut）
    /// - 可写性：可写（Owned/ViewMut/ArcRepr）还是只读（View）
    storage: S,

    /// 各轴长度。
    ///
    /// 对于 Ix0（标量）：零长度数组表示
    /// 对于 Ix1-Ix6：[usize; N] 数组
    /// 对于 IxDyn：堆分配的 Vec<usize>
    shape: D,

    /// 各轴步长（有符号，元素单位）。
    ///
    /// 步长表示沿某轴移动一个位置时，在内存中需要跳跃的元素数。
    /// - 正步长：正向遍历
    /// - 负步长：反向遍历（如翻转操作）
    /// - 零步长：广播维度（重复同一元素）
    ///
    /// 类型与 shape 相同，便于维度相关操作。
    strides: D,

    /// 数据起始偏移量（元素单位）。
    ///
    /// 指向 storage 中第一个逻辑元素的位置。
    /// 支持切片视图的零拷贝实现：切片操作仅调整 offset 和 shape/strides，
    /// 而不复制底层数据。
    offset: usize,
}
```

### 3.2 字段详细说明

#### 3.2.1 storage: S

| 存储模式 | 类型实例化 | 拥有数据 | 可读 | 可写 |
|----------|-----------|---------|------|------|
| Owned | `Owned<A>` | 是 | 是 | 是 |
| View | `ViewRepr<&'a A>` | 否（借用） | 是 | 否 |
| ViewMut | `ViewMutRepr<&'a mut A>` | 否（独占借用） | 是 | 是 |
| Arc | `ArcRepr<A>` | 共享 | 是 | 通过 make_mut |

**设计约束**：
- S 必须实现 `Storage` trait（提供读取能力）
- 可写存储必须实现 `StorageMut` trait（继承 Storage，提供写入能力）

#### 3.2.2 shape: D

| 维度类型 | 内部表示 | ndim() | 示例 |
|----------|----------|--------|------|
| Ix0 | `()` （单元类型） | 0 | 标量 |
| Ix1 | `[usize; 1]` | 1 | 向量 |
| Ix2 | `[usize; 2]` | 2 | 矩阵 |
| Ix3 | `[usize; 3]` | 3 | 3D 数组 |
| Ix4-Ix6 | `[usize; N]` | N | 高维数组 |
| IxDyn | `Vec<usize>` | 运行时确定 | 动态维度 |

**设计约束**：
- D 必须实现 `Dimension` trait

#### 3.2.3 strides: D

**使用 D 类型的设计理由**：

| 考量 | 说明 |
|------|------|
| 类型一致性 | strides 与 shape 维度数必须相同，使用相同类型在编译期保证 |
| 静态优化 | 静态维度（IxN）的 strides 也是固定大小数组，避免堆分配 |
| 负步长支持 | 使用 `isize` 语义存储，但类型为 D（内部封装） |

**实现方案**：

```rust
// 方案 A：strides 内部存储 isize，但类型为 D
// 需要在 Dimension trait 中提供 isize 访问方法

// 方案 B：strides 使用独立类型
// 如 Strides<D>，但这增加了类型复杂度

// 推荐方案 A，保持类型统一
```

#### 3.2.4 offset: usize

**作用**：支持切片视图的零拷贝偏移

```
原始数组 storage: [a, b, c, d, e, f, g, h]
shape: [8], strides: [1], offset: 0

切片 [2..5] 后：
storage: [a, b, c, d, e, f, g, h]  // 共享，不复制
shape: [3], strides: [1], offset: 2  // 仅调整元数据
逻辑视图: [c, d, e]
```

**设计优势**：
- 切片操作 O(1)，仅修改元数据
- 无需复制底层数据
- 适用于所有存储模式

---

## 4. 核心方法 (impls.rs)

### 4.1 形状与步长查询

```rust
impl<S, D> TensorBase<S, D>
where
    D: Dimension,
{
    /// 返回各轴长度的切片。
    ///
    /// # 示例
    ///
    /// ```
    /// let t = Tensor2::<f64>::zeros([3, 4]);
    /// assert_eq!(t.shape(), &[3, 4]);
    /// ```
    pub fn shape(&self) -> &[usize];

    /// 返回各轴步长的切片（元素单位，有符号）。
    ///
    /// 步长可能为负（反转维度）或零（广播维度）。
    pub fn strides(&self) -> &[isize];

    /// 返回维度数。
    ///
    /// 对于静态维度（Ix0-Ix6），此值为编译期常量。
    /// 对于动态维度（IxDyn），此值为运行时值。
    pub fn ndim(&self) -> usize;

    /// 返回元素总数（所有维度长度的乘积）。
    ///
    /// 空数组返回 0（任一维度长度为 0）。
    /// 标量（Ix0）返回 1。
    pub fn len(&self) -> usize;

    /// 返回数组是否为空（任一维度长度为 0）。
    pub fn is_empty(&self) -> bool;

    /// 返回数据起始偏移量（元素单位）。
    ///
    /// 偏移量相对于 storage 的起始位置。
    pub fn offset(&self) -> usize;

    /// 返回维度类型的克隆。
    ///
    /// 对于静态维度，返回栈分配的值。
    /// 对于动态维度，返回堆分配的克隆。
    pub fn raw_dim(&self) -> D;
}
```

### 4.2 指针访问

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// 返回数据起始位置的不可变原始指针。
    ///
    /// 指针指向第一个逻辑元素（考虑 offset）。
    ///
    /// # Safety
    ///
    /// 调用方须保证指针使用期间底层数据保持有效。
    pub fn as_ptr(&self) -> *const A;

    /// 不检查偏移量有效性的指针访问。
    ///
    /// # Safety
    ///
    /// 调用方须保证 offset 有效且数据已初始化。
    pub unsafe fn as_ptr_unchecked(&self) -> *const A;
}

impl<S, D, A> TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    D: Dimension,
{
    /// 返回数据起始位置的可变原始指针。
    ///
    /// 仅对可写存储（Owned、ViewMut、ArcRepr）可用。
    ///
    /// # Safety
    ///
    /// 调用方须保证：
    /// - 指针使用期间底层数据保持有效
    /// - 无其他引用同时访问同一数据（别名规则）
    pub fn as_mut_ptr(&mut self) -> *mut A;
}
```

### 4.3 视图创建

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// 创建不可变视图。
    ///
    /// 返回的视图共享底层数据，不进行复制。
    pub fn view(&self) -> TensorView<'_, A, D>;

    /// 创建指定维度类型的不可变视图。
    ///
    /// 用于维度类型转换（如 Ix3 -> IxDyn）。
    pub fn view_into<D2>(&self) -> TensorView<'_, A, D2>
    where
        D2: Dimension,
        D: IntoDimension<D2>;
}

impl<S, D, A> TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    D: Dimension,
{
    /// 创建可变视图。
    ///
    /// 仅对可写存储可用。返回的视图独占访问底层数据。
    pub fn view_mut(&mut self) -> TensorViewMut<'_, A, D>;

    /// 创建指定维度类型的可变视图。
    pub fn view_mut_into<D2>(&mut self) -> TensorViewMut<'_, A, D2>
    where
        D2: Dimension,
        D: IntoDimension<D2>;
}
```

### 4.4 所有权转换

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Clone,
{
    /// 克隆数据到新的拥有型数组。
    ///
    /// 总是分配新内存并复制数据，即使输入已是 Owned。
    pub fn to_owned(&self) -> Tensor<A, D>;
}

impl<S, D, A> TensorBase<S, D>
where
    S: StorageIntoOwned<Elem = A>,
    D: Dimension,
{
    /// 消费数组，转换为拥有型数组。
    ///
    /// 对于 Owned：直接返回，不复制。
    /// 对于 View/ViewMut：复制数据到新分配。
    /// 对于 ArcRepr：若引用计数为 1 则直接返回，否则复制。
    pub fn into_owned(self) -> Tensor<A, D>;
}
```

### 4.5 布局查询委托

```rust
impl<S, D> TensorBase<S, D>
where
    D: Dimension,
{
    /// 是否 F-order（列优先）连续。
    ///
    /// 委托给 layout 模块实现。
    pub fn is_f_contiguous(&self) -> bool;

    /// 是否 C-order（行优先）连续。
    pub fn is_c_contiguous(&self) -> bool;

    /// 是否任一方向连续。
    pub fn is_contiguous(&self) -> bool {
        self.is_f_contiguous() || self.is_c_contiguous()
    }

    /// 是否 SIMD 对齐（64 字节）。
    pub fn is_aligned(&self) -> bool;

    /// 是否存在零步长（广播维度）。
    pub fn has_zero_stride(&self) -> bool;

    /// 是否存在负步长（反转维度）。
    pub fn has_neg_stride(&self) -> bool;

    /// 返回完整布局标志。
    pub fn layout_flags(&self) -> LayoutFlags;
}
```

### 4.6 字节步长查询

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// 返回各轴步长的切片（字节单位，有符号）。
    ///
    /// 等价于 `self.strides().map(|s| s * size_of::<A>())`。
    pub fn strides_bytes(&self) -> &[isize];
}
```

---

## 5. 类型别名体系 (aliases.rs)

### 5.1 主类型别名

```rust
/// 拥有数据的多维数组。
///
/// 创建时分配内存，drop 时释放内存。
/// 支持完整的读写操作。
pub type Tensor<A, D> = TensorBase<Owned<A>, D>;

/// 不可变视图。
///
/// 借用底层数据，不拥有所有权。
/// 仅支持读取操作。
pub type TensorView<'a, A, D> = TensorBase<ViewRepr<&'a A>, D>;

/// 可变视图。
///
/// 独占借用底层数据，不拥有所有权。
/// 支持读取和写入操作。
pub type TensorViewMut<'a, A, D> = TensorBase<ViewMutRepr<&'a mut A>, D>;

/// 原子引用计数共享数组。
///
/// 多个 ArcTensor 可以共享同一份数据。
/// 写入时自动触发写时复制（通过 make_mut）。
pub type ArcTensor<A, D> = TensorBase<ArcRepr<A>, D>;
```

### 5.2 维度便捷别名

#### 5.2.1 Owned 系列

```rust
/// 0 维标量（拥有数据）。
pub type Tensor0<A> = Tensor<A, Ix0>;

/// 1 维向量（拥有数据）。
pub type Tensor1<A> = Tensor<A, Ix1>;

/// 2 维矩阵（拥有数据）。
pub type Tensor2<A> = Tensor<A, Ix2>;

/// 3 维数组（拥有数据）。
pub type Tensor3<A> = Tensor<A, Ix3>;

/// 4 维数组（拥有数据）。
pub type Tensor4<A> = Tensor<A, Ix4>;

/// 5 维数组（拥有数据）。
pub type Tensor5<A> = Tensor<A, Ix5>;

/// 6 维数组（拥有数据）。
pub type Tensor6<A> = Tensor<A, Ix6>;

/// 动态维度数组（拥有数据）。
pub type TensorD<A> = Tensor<A, IxDyn>;
```

#### 5.2.2 View 系列

```rust
/// 0 维标量（不可变视图）。
pub type TensorView0<'a, A> = TensorView<'a, A, Ix0>;

/// 1 维向量（不可变视图）。
pub type TensorView1<'a, A> = TensorView<'a, A, Ix1>;

/// 2 维矩阵（不可变视图）。
pub type TensorView2<'a, A> = TensorView<'a, A, Ix2>;

/// 3 维数组（不可变视图）。
pub type TensorView3<'a, A> = TensorView<'a, A, Ix3>;

/// 动态维度数组（不可变视图）。
pub type TensorViewD<'a, A> = TensorView<'a, A, IxDyn>;
```

#### 5.2.3 ViewMut 系列

```rust
/// 0 维标量（可变视图）。
pub type TensorViewMut0<'a, A> = TensorViewMut<'a, A, Ix0>;

/// 1 维向量（可变视图）。
pub type TensorViewMut1<'a, A> = TensorViewMut<'a, A, Ix1>;

/// 2 维矩阵（可变视图）。
pub type TensorViewMut2<'a, A> = TensorViewMut<'a, A, Ix2>;

/// 3 维数组（可变视图）。
pub type TensorViewMut3<'a, A> = TensorViewMut<'a, A, Ix3>;

/// 动态维度数组（可变视图）。
pub type TensorViewMutD<'a, A> = TensorViewMut<'a, A, IxDyn>;
```

#### 5.2.4 Arc 系列

```rust
/// 0 维标量（Arc 共享）。
pub type ArcTensor0<A> = ArcTensor<A, Ix0>;

/// 1 维向量（Arc 共享）。
pub type ArcTensor1<A> = ArcTensor<A, Ix1>;

/// 2 维矩阵（Arc 共享）。
pub type ArcTensor2<A> = ArcTensor<A, Ix2>;

/// 3 维数组（Arc 共享）。
pub type ArcTensor3<A> = ArcTensor<A, Ix3>;

/// 动态维度数组（Arc 共享）。
pub type ArcTensorD<A> = ArcTensor<A, IxDyn>;
```

### 5.3 命名规范

| 规范 | 示例 | 说明 |
|------|------|------|
| 主类型 + 维度数 | `Tensor2<A>` | 静态维度的拥有型数组 |
| 主类型 + D 后缀 | `TensorD<A>` | 动态维度的拥有型数组 |
| 存储前缀 + 维度 | `TensorView2<'a, A>` | 不可变视图 |
| 存储前缀 + D 后缀 | `TensorViewMutD<'a, A>` | 动态维度的可变视图 |

---

## 6. 内部构造方法 (construct.rs)

### 6.1 from_raw_parts 系列

```rust
impl<'a, A, D> TensorBase<ViewRepr<&'a A>, D>
where
    D: Dimension,
{
    /// 从原始部件构造不可变视图。
    ///
    /// # Safety
    ///
    /// 调用方须保证：
    /// - `ptr` 非空、非悬垂，且对齐到 `align_of::<A>()`
    /// - `ptr` 起始的内存范围能覆盖所有可访问元素
    /// - 内存在返回视图的生命周期 `'a` 内保持有效
    /// - 内存可被共享读取，但不可被写入
    /// - `shape` 与 `strides` 长度一致
    /// - 任意合法索引计算出的偏移量不越界
    /// - 所有可访问元素已正确初始化
    ///
    /// # 示例
    ///
    /// ```ignore
    /// let data = [1.0, 2.0, 3.0, 4.0];
    /// let view = unsafe {
    ///     TensorView1::from_raw_parts(
    ///         data.as_ptr(),
    ///         Ix1([4]),
    ///         Ix1([1]),
    ///         0,
    ///     )
    /// };
    /// ```
    pub unsafe fn from_raw_parts(
        ptr: *const A,
        shape: D,
        strides: D,
        offset: usize,
    ) -> Self;
}

impl<'a, A, D> TensorBase<ViewMutRepr<&'a mut A>, D>
where
    D: Dimension,
{
    /// 从原始部件构造可变视图。
    ///
    /// # Safety
    ///
    /// 调用方须保证：
    /// - `ptr` 非空、非悬垂，且对齐到 `align_of::<A>()`
    /// - `ptr` 起始的内存范围能覆盖所有可访问元素
    /// - 内存在返回视图的生命周期 `'a` 内保持有效
    /// - 内存无其他引用（独占访问）
    /// - `shape` 与 `strides` 长度一致
    /// - 任意合法索引计算出的偏移量不越界
    /// - 所有可访问元素已正确初始化
    ///
    /// # 注意
    ///
    /// 与 `from_raw_parts` 不同，此方法要求独占访问。
    /// 返回的视图拥有对底层数据的唯一写入权。
    pub unsafe fn from_raw_parts_mut(
        ptr: *mut A,
        shape: D,
        strides: D,
        offset: usize,
    ) -> Self;
}
```

### 6.2 into_raw_parts

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: StorageIntoRaw<Elem = A>,
    D: Dimension,
{
    /// 消费数组，返回原始部件。
    ///
    /// 返回 (ptr, shape, strides, offset) 元组。
    /// 调用方负责确保内存安全释放。
    ///
    /// # Safety
    ///
    /// 返回的指针所有权转移给调用方。
    /// 对于 Owned 存储，调用方须最终释放内存。
    /// 对于 View 存储，返回的指针生命周期与原借用绑定。
    pub fn into_raw_parts(self) -> (*mut A, D, D, usize);
}
```

### 6.3 内部构造辅助方法

```rust
impl<S, D> TensorBase<S, D>
where
    D: Dimension,
{
    /// 内部构造方法，不检查不变量。
    ///
    /// # Safety
    ///
    /// 调用方须保证所有不变量成立：
    /// - shape 与 strides 长度一致
    /// - storage 与 shape/strides/offset 组合有效
    ///
    /// 此方法仅供内部模块使用，不对外暴露。
    #[inline]
    pub(crate) unsafe fn new_unchecked(
        storage: S,
        shape: D,
        strides: D,
        offset: usize,
    ) -> Self {
        TensorBase {
            storage,
            shape,
            strides,
            offset,
        }
    }
}
```

---

## 7. 泛型约束策略

### 7.1 Trait 约束层次

```
                    ┌──────────────┐
                    │   Storage    │ ← 只读访问
                    │  - as_ptr()  │
                    │  - len()     │
                    └──────┬───────┘
                           │
           ┌───────────────┴───────────────┐
           │                               │
    ┌──────┴──────┐                 ┌──────┴──────┐
    │ StorageMut  │                 │StorageOwned │
    │- as_mut_ptr│                 │ - into_ptr  │
    └─────────────┘                 └─────────────┘
           │                               │
           └───────────────┬───────────────┘
                           │
                    ┌──────┴──────┐
                    │StorageInto  │
                    │   Owned     │
                    │- into_owned │
                    └─────────────┘
```

### 7.2 方法约束矩阵

| 方法 | S 约束 | A 约束 | 说明 |
|------|--------|--------|------|
| `shape()` | - | - | 无约束，访问 shape 字段 |
| `strides()` | - | - | 无约束，访问 strides 字段 |
| `len()` | - | - | 无约束，计算 shape 乘积 |
| `as_ptr()` | `S: Storage<Elem = A>` | - | 需要读取能力 |
| `as_mut_ptr()` | `S: StorageMut<Elem = A>` | - | 需要写入能力 |
| `view()` | `S: Storage<Elem = A>` | - | 创建只读视图 |
| `view_mut()` | `S: StorageMut<Elem = A>` | - | 创建可变视图 |
| `to_owned()` | `S: Storage<Elem = A>` | `A: Clone` | 需要克隆能力 |
| `into_owned()` | `S: StorageIntoOwned<Elem = A>` | - | 消费转换 |
| `fill()` | `S: StorageMut<Elem = A>` | `A: Clone` | 填充需要写入和克隆 |
| `map()` | `S: Storage<Elem = A>` | `A: Clone` | 映射需要克隆 |
| `mapv_inplace()` | `S: StorageMut<Elem = A>` | - | 原地映射仅需写入 |

### 7.3 存储模式与 Trait 实现

| 存储模式 | Storage | StorageMut | StorageOwned | StorageIntoOwned |
|----------|---------|------------|--------------|------------------|
| `Owned<A>` | ✓ | ✓ | ✓ | ✓（无拷贝） |
| `ViewRepr<&'a A>` | ✓ | ✗ | ✗ | ✓（拷贝） |
| `ViewMutRepr<&'a mut A>` | ✓ | ✓ | ✗ | ✓（拷贝） |
| `ArcRepr<A>` | ✓ | ✓（通过 make_mut） | ✓ | ✓（引用计数为 1 时无拷贝） |

### 7.4 Element 约束使用场景

| 场景 | 约束 | 理由 |
|------|------|------|
| 归约（sum/prod） | `A: Numeric` | 需要加法/乘法单位元 |
| 数学函数（sin/cos） | `A: RealScalar` | 仅实数支持 |
| 复数运算 | `A: ComplexScalar` | 仅复数支持 |
| 比较（min/max） | `A: Element` | 基础比较即可 |
| 填充 | `A: Clone` | 仅需克隆 |

---

## 8. 生命周期管理

### 8.1 View 的生命周期

```rust
pub type TensorView<'a, A, D> = TensorBase<ViewRepr<&'a A>, D>;
```

**生命周期绑定**：
- `'a` 是底层数据的借用生命周期
- `TensorView` 不能比 `'a` 活得更久
- Rust 借用检查器自动保证

```rust
fn example() {
    let data = vec![1.0, 2.0, 3.0];
    let tensor = Tensor1::from_vec(data);
    
    let view: TensorView1<f64> = tensor.view();
    // view 的生命周期绑定到 tensor 的借用
    
    // tensor 仍然可用（共享借用）
    let view2 = tensor.view();
    
    // view 和 view2 可以同时存在
}
```

### 8.2 ViewMut 的生命周期

```rust
pub type TensorViewMut<'a, A, D> = TensorBase<ViewMutRepr<&'a mut A>, D>;
```

**独占语义**：
- `'a` 是底层数据的独占借用生命周期
- `TensorViewMut` 存在期间，不能有任何其他引用（包括 `&T`）

```rust
fn example() {
    let mut tensor = Tensor1::from_vec(vec![1.0, 2.0, 3.0]);
    
    {
        let view: TensorViewMut1<f64> = tensor.view_mut();
        // 在 view 存在期间，不能访问 tensor
        
        view[0] = 10.0;
    }
    // view 离开作用域，独占借用结束
    
    // 现在可以再次访问 tensor
    let val = tensor[0]; // 10.0
}
```

### 8.3 生命周期传播

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    // view() 的生命周期与 self 绑定
    pub fn view(&self) -> TensorView<'_, A, D>;
    //                         ^^ 生命周期省略，等同于 'self
}

impl<'a, A, D> TensorBase<ViewRepr<&'a A>, D>
where
    D: Dimension,
{
    // 从已有视图创建新视图，生命周期不能延长
    pub fn view(&self) -> TensorView<'a, A, D>;
    //                        ^^ 使用原始生命周期 'a
}
```

### 8.4 ArcRepr 的生命周期

`ArcTensor` 不使用生命周期参数，因为：
- 数据所有权由 `Arc` 管理
- 引用计数保证数据在所有引用释放后才被释放

```rust
fn example() {
    let tensor = ArcTensor1::from_vec(vec![1.0, 2.0, 3.0]);
    
    let clone = tensor.clone(); // 引用计数 +1
    
    drop(tensor); // 引用计数 -1，数据仍存在
    
    // clone 仍可用
    let val = clone[0];
}
```

---

## 9. Deref/AsRef 策略

### 9.1 设计决策

**不实现 `Deref<Target = TensorView>`**

| 考量 | 说明 |
|------|------|
| 显式优于隐式 | `.view()` 调用清晰表达意图 |
| 避免歧义 | 防止 `&Tensor` 自动转换为 `TensorView` 导致意外借用 |
| 生命周期明确 | 显式调用让生命周期关系更清晰 |
| 与 ndarray 一致 | ndarray 也不实现 Deref 到视图 |

### 9.2 提供的转换 trait

```rust
// AsRef 实现：允许泛型代码接受任何可视图化的类型
impl<S, D, A> AsRef<TensorBase<ViewRepr<&A>, D>> for TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    fn as_ref(&self) -> &TensorBase<ViewRepr<&A>, D> {
        // unsafe transmute，因为视图和原数组布局兼容
        // 实际上返回的是临时创建的视图的引用
        // 这种设计需要仔细考虑
    }
}

// 推荐方案：仅提供显式的 view() 方法
// 不实现 AsRef/Deref
```

### 9.3 推荐的转换模式

```rust
// 显式视图创建
let view = tensor.view();

// 泛型约束使用 Storage trait
fn process<S, D, A>(tensor: &TensorBase<S, D>)
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    // 内部使用 view()
    let view = tensor.view();
}

// 而非
fn process<A, D>(tensor: &TensorView<A, D>)
where
    D: Dimension,
{
    // ...
}
```

### 9.4 From/Into 实现

```rust
// Owned -> View（隐式借用）
impl<'a, A, D> From<&'a Tensor<A, D>> for TensorView<'a, A, D>
where
    D: Dimension,
{
    fn from(tensor: &'a Tensor<A, D>) -> Self {
        tensor.view()
    }
}

// Owned -> ViewMut（显式借用）
impl<'a, A, D> From<&'a mut Tensor<A, D>> for TensorViewMut<'a, A, D>
where
    D: Dimension,
{
    fn from(tensor: &'a mut Tensor<A, D>) -> Self {
        tensor.view_mut()
    }
}
```

---

## 10. 与其他模块的交互

### 10.1 模块依赖图

```
┌─────────────────────────────────────────────────────────────┐
│                        tensor (核心)                        │
│                     TensorBase<S, D>                        │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   storage     │   │   dimension   │   │    layout     │
│ - Owned<A>    │   │ - Ix0-Ix6     │   │ - LayoutFlags │
│ - ViewRepr    │   │ - IxDyn       │   │ - is_f_contig │
│ - ViewMutRepr │   │ - Dimension   │   │ - is_c_contig │
│ - ArcRepr     │   │   trait       │   │   etc.        │
│ - Storage     │   │               │   │               │
│   trait       │   │               │   │               │
└───────────────┘   └───────────────┘   └───────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      ops (操作)                             │
│ - arithmetic (逐元素运算)                                   │
│ - reduction (归约)                                          │
│ - broadcast (广播)                                          │
│ - linalg (矩阵运算)                                         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      iter (迭代器)                          │
│ - Elements (元素迭代)                                       │
│ - AxisIter (轴迭代)                                         │
│ - Windows (窗口迭代)                                        │
│ - Indexed (索引迭代)                                        │
│ - Zip (多数组同步迭代)                                      │
└─────────────────────────────────────────────────────────────┘
```

### 10.2 与 storage 模块的接口

```rust
// TensorBase 使用 Storage trait 的关联类型
trait Storage {
    type Elem;
    
    fn as_ptr(&self) -> *const Self::Elem;
    fn len(&self) -> usize;
}

// TensorBase 约束
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    // ...
}
```

### 10.3 与 dimension 模块的接口

```rust
// Dimension trait 提供形状操作
trait Dimension: Clone + Debug {
    fn ndim(&self) -> usize;
    fn slice(&self) -> &[usize];
    fn size(&self) -> usize;
    
    // 步长相关
    fn default_strides(&self) -> Self;
    fn fortran_strides(&self) -> Self;
}

// TensorBase 使用
impl<S, D> TensorBase<S, D>
where
    D: Dimension,
{
    pub fn shape(&self) -> &[usize] {
        self.shape.slice()
    }
}
```

### 10.4 与 layout 模块的接口

```rust
// LayoutFlags 由 layout 模块定义
mod layout {
    pub struct LayoutFlags(u8);
    
    pub fn compute_flags<D: Dimension>(
        shape: &D,
        strides: &D,
        ptr: *const u8,
    ) -> LayoutFlags;
}

// TensorBase 委托
impl<S, D> TensorBase<S, D>
where
    D: Dimension,
{
    pub fn is_f_contiguous(&self) -> bool {
        layout::is_f_contiguous(&self.shape, &self.strides)
    }
}
```

### 10.5 与 ops 模块的接口

```rust
// ops 模块为 TensorBase 实现运算
mod ops {
    impl<S1, S2, D, A> Add<TensorBase<S2, D>> for TensorBase<S1, D>
    where
        S1: Storage<Elem = A>,
        S2: Storage<Elem = A>,
        D: Dimension,
        A: Numeric,
    {
        type Output = Tensor<A, D>;
        
        fn add(self, rhs: TensorBase<S2, D>) -> Self::Output {
            // 调用逐元素加法实现
            add_elements(&self, &rhs)
        }
    }
}
```

### 10.6 与 iter 模块的接口

```rust
// iter 模块提供迭代器类型
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// 按内存布局顺序迭代元素。
    pub fn iter(&self) -> iter::Elements<'_, A, D> {
        iter::Elements::new(self.view())
    }
    
    /// 沿指定轴迭代。
    pub fn axis_iter(&self, axis: usize) -> iter::AxisIter<'_, A, D::Smaller>
    where
        D: RemoveAxis,
    {
        iter::AxisIter::new(self.view(), axis)
    }
}
```

---

## 11. 实现任务分解

### 任务 1：结构体定义 (mod.rs)

**预计时间**: 10 分钟

**任务内容**:
- 定义 `TensorBase<S, D>` 结构体
- 添加四个字段：storage、shape、strides、offset
- 添加文档注释

**验收标准**:
- 结构体编译通过
- `#[repr(C)]` 属性正确
- 所有字段有完整文档

---

### 任务 2：Dimension trait 依赖

**预计时间**: 10 分钟

**任务内容**:
- 确保 `Dimension` trait 已定义（依赖 dimension 模块）
- 在 `TensorBase` impl 块中添加 `D: Dimension` 约束

**验收标准**:
- 所有需要 `D: Dimension` 的方法有正确约束

---

### 任务 3：Storage trait 依赖

**预计时间**: 10 分钟

**任务内容**:
- 确保 `Storage` 和 `StorageMut` trait 已定义（依赖 storage 模块）
- 在需要的方法中添加 `S: Storage<Elem = A>` 约束

**验收标准**:
- `as_ptr()` 有 `S: Storage` 约束
- `as_mut_ptr()` 有 `S: StorageMut` 约束

---

### 任务 4：形状与步长查询方法 (impls.rs)

**预计时间**: 10 分钟

**任务内容**:
- 实现 `shape()`、`strides()`、`ndim()`、`len()`、`is_empty()`
- 实现 `offset()`、`raw_dim()`
- 添加文档注释和示例

**验收标准**:
- 所有方法签名正确
- 文档完整
- 单元测试通过

---

### 任务 5：指针访问方法 (impls.rs)

**预计时间**: 10 分钟

**任务内容**:
- 实现 `as_ptr()`、`as_ptr_unchecked()`
- 实现 `as_mut_ptr()`
- 添加 Safety 文档

**验收标准**:
- unsafe 方法有 Safety 文档节
- 约束正确（Storage vs StorageMut）

---

### 任务 6：视图创建方法 (impls.rs)

**预计时间**: 10 分钟

**任务内容**:
- 实现 `view()`、`view_into()`
- 实现 `view_mut()`、`view_mut_into()`
- 处理生命周期关系

**验收标准**:
- 生命周期正确绑定
- 约束正确

---

### 任务 7：类型别名 (aliases.rs)

**预计时间**: 10 分钟

**任务内容**:
- 定义 4 个主类型别名（Tensor、TensorView、TensorViewMut、ArcTensor）
- 定义维度便捷别名（Tensor0-Tensor6、TensorD 等）
- 为 View/ViewMut/Arc 定义对应别名

**验收标准**:
- 所有别名编译通过
- 命名符合规范

---

### 任务 8：from_raw_parts 系列 (construct.rs)

**预计时间**: 10 分钟

**任务内容**:
- 实现 `from_raw_parts`（不可变）
- 实现 `from_raw_parts_mut`（可变）
- 添加详细 Safety 文档

**验收标准**:
- Safety 文档列出所有前提条件
- unsafe 块最小化

---

### 任务 9：into_raw_parts 和内部构造 (construct.rs)

**预计时间**: 10 分钟

**任务内容**:
- 实现 `into_raw_parts`
- 实现 `new_unchecked`（内部方法）
- 定义 `StorageIntoRaw` trait

**验收标准**:
- 所有权正确转移
- 内部方法可见性正确

---

### 任务 10：布局查询委托 (impls.rs)

**预计时间**: 10 分钟

**任务内容**:
- 实现 `is_f_contiguous()`、`is_c_contiguous()`、`is_contiguous()`
- 实现 `is_aligned()`、`has_zero_stride()`、`has_neg_stride()`
- 实现 `layout_flags()`
- 委托给 layout 模块

**验收标准**:
- 所有方法正确委托
- O(1) 复杂度

---

## 12. 设计决策记录

### 12.1 为什么选择 TensorBase<S, D> 双参数设计？

| 方案 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **TensorBase<S, D>** | 存储模式 + 维度类型 | 类型安全、零开销、统一接口 | 泛型参数多 |
| TensorBase<A, S, D> | 额外元素类型参数 | 显式元素类型 | 冗余（A 可从 S 推导） |
| 分离类型（Tensor/TensorView） | 每种存储独立结构体 | 简单直观 | 代码重复、难扩展 |
| 单一 Tensor<A, D> + 运行时标志 | 运行时区分存储模式 | 类型简单 | 运行时开销、不安全 |

**选择理由**：
1. **零开销**：不同存储模式编译为不同类型，无运行时判断
2. **类型安全**：编译期保证不可变视图不能写入
3. **代码复用**：所有存储模式共享实现
4. **扩展性**：新增存储模式只需实现 trait

### 12.2 为什么 strides 使用 D 类型？

| 方案 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **strides: D** | 与 shape 同类型 | 类型一致性、静态优化 | 需要封装 isize |
| strides: Vec<isize> | 统一使用 Vec | 简单 | 静态维度也要堆分配 |
| strides: [isize; N] | 固定数组 | 栈分配 | 不支持动态维度 |
| Strides<D> | 独立类型 | 可区分 shape/strides | 增加类型复杂度 |

**选择理由**：
1. **编译期保证**：strides 与 shape 维度数相同
2. **性能**：静态维度使用栈分配数组
3. **简洁**：不需要额外的 Strides 类型

### 12.3 为什么需要 offset 字段？

| 场景 | 无 offset | 有 offset |
|------|-----------|-----------|
| 切片 [2..5] | 复制数据到新存储 | 仅修改元数据 |
| 切片性能 | O(n) | O(1) |
| 内存使用 | 重复存储 | 共享存储 |

**选择理由**：
1. **零拷贝切片**：切片操作不复制数据
2. **统一机制**：所有存储模式使用相同机制
3. **BLAS 兼容**：支持从数组中间传递指针

### 12.4 为什么不实现 Deref<Target=TensorView>？

| 方案 | 优点 | 缺点 |
|------|------|------|
| **显式 .view()** | 意图清晰、生命周期明确 | 稍显冗长 |
| Deref 到视图 | 隐式转换方便 | 生命周期隐式传播、可能意外借用 |

**选择理由**：
1. **显式优于隐式**：`.view()` 清晰表达创建视图的意图
2. **避免歧义**：防止 `&Tensor` 自动转换为视图导致的意外行为
3. **与 ndarray 一致**：降低用户学习成本

### 12.5 为什么使用泛型约束而非枚举？

| 方案 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **泛型约束** | `S: Storage` | 零开销、编译期检查 | 泛型复杂 |
| 枚举 | `enum Storage { Owned, View, ... }` | 简单 | 运行时匹配、不安全访问 |

**选择理由**：
1. **编译期安全**：不可变视图编译期禁止写入
2. **零开销**：无运行时匹配开销
3. **类型推导**：编译器帮助推导正确类型

---

## 附录 A：完整类型关系图

```
                        TensorBase<S, D>
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
    TensorBase<        TensorBase<          TensorBase<
      Owned<A>,         ViewRepr<           ViewMutRepr<
         D>            &'a A>, D>          &'a mut A>, D>
          │                   │                   │
          ▼                   ▼                   ▼
      Tensor<A,D>      TensorView<'a,A,D>  TensorViewMut<'a,A,D>
          │                   │                   │
    ┌─────┴─────┐       ┌─────┴─────┐       ┌─────┴─────┐
    │           │       │           │       │           │
 Tensor1<A> TensorD<A> TensorView1 TensorViewD TensorViewMut1 TensorViewMutD
    │           │       │           │       │           │
   ...         ...     ...         ...     ...         ...
```

---

## 附录 B：命名约定速查

| 模式 | 示例 | 含义 |
|------|------|------|
| `Tensor{N}` | `Tensor2<A>` | N 维拥有型数组 |
| `TensorD` | `TensorD<A>` | 动态维度拥有型数组 |
| `TensorView{N}` | `TensorView2<'a, A>` | N 维不可变视图 |
| `TensorViewMut{N}` | `TensorViewMut2<'a, A>` | N 维可变视图 |
| `ArcTensor{N}` | `ArcTensor2<A>` | N 维 Arc 共享数组 |
| `TensorViewD` | `TensorViewD<'a, A>` | 动态维度不可变视图 |

---

## 附录 C：Trait 约束速查

| 方法 | 完整签名 |
|------|----------|
| `shape()` | `fn shape(&self) -> &[usize] where D: Dimension` |
| `as_ptr()` | `fn as_ptr(&self) -> *const A where S: Storage<Elem = A>, D: Dimension` |
| `as_mut_ptr()` | `fn as_mut_ptr(&mut self) -> *mut A where S: StorageMut<Elem = A>, D: Dimension` |
| `view()` | `fn view(&self) -> TensorView<'_, A, D> where S: Storage<Elem = A>, D: Dimension` |
| `view_mut()` | `fn view_mut(&mut self) -> TensorViewMut<'_, A, D> where S: StorageMut<Elem = A>, D: Dimension` |
| `to_owned()` | `fn to_owned(&self) -> Tensor<A, D> where S: Storage<Elem = A>, D: Dimension, A: Clone` |
| `into_owned()` | `fn into_owned(self) -> Tensor<A, D> where S: StorageIntoOwned<Elem = A>, D: Dimension` |

---

*文档结束*
