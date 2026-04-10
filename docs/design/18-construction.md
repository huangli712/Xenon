# 构造操作模块设计

> 文档编号: 18 | 模块: `src/construct/` | 阶段: Phase 4
> 前置文档: `07-tensor.md`, `05-storage.md`
> 需求参考: 需求说明书 §19

---

## 1. 模块定位

### 1.1 职责边界

| 职责 | 包含 | 不包含 |
|------|------|--------|
| 零初始化构造 | `zeros<A, D>(shape)` 全零张量 | arange / linspace / logspace 序列生成（当前版本不提供） |
| 常量填充构造 | `ones<A, D>(shape)` / `full<A, D>(shape, scalar)` 全一/填充张量 | 随机数构造（当前版本不提供） |
| 单位矩阵 | `eye<A>(n)` 单位矩阵 | 从文件加载（当前版本不提供） |
| 从 Vec 构造 | `from_shape_vec(shape, vec)` 消费输入 Vec 并构造张量 | 从迭代器/生成器构造（由 `from_fn` 提供） |
| 从切片构造 | `from_shape_slice(shape, slice)` 从切片拷贝数据 | 从文件/网络加载（当前版本不提供） |
| 从固定数组构造 | `from_array<A, N>(arr, shape)` 从固定大小数组构造 | 从文件加载（当前版本不提供） |
| 从标量构造 | `from_scalar<A>(scalar)` 零维张量 | — |
| 从函数构造 | `from_fn<A, D, F>(shape, f)` 从闭包构造 | 从迭代器/生成器构造（由 `from_fn` 提供） |
| 合法性验证 | 所有构造路径验证形状/大小合法性 | 隐式类型转换（由 `convert` 模块提供） |

### 1.2 设计原则

| 原则 | 体现 |
|------|------|
| 合法性验证 | 所有构造路径须验证合法性，防止越界访问（需求说明书 §8） |
| F-order 默认 | 构造时数据按 F-order 存放，默认列优先布局 |
| 对齐分配 | `zeros`/`ones` 使用对齐分配器，满足 BLAS 兼容性（参见 `23-ffi.md` §4.5） |
| 对齐优先 | `from_shape_vec` 将输入 `Vec<A>` 的数据复制到新分配的 64 字节对齐内存中（通过 `Owned::from_vec_aligned()`），确保 SIMD 友好的内存对齐；不承诺复用原始 Vec 的分配 |
| 类型安全 | 形状和元素类型通过泛型约束在编译期检查 |

### 1.3 在架构中的位置

```
依赖层级：
L0: error, private
L1: dimension, element, complex
L2: layout (依赖 dimension)
 L3: storage (独立于 layout，由 tensor 持有并消费 layout 结果)
L4: tensor (依赖 storage, dimension)
L5: construct  ← 当前模块（依赖 storage, layout, dimension, element）
```

---

## 2. 文件位置

```
src/
└── construct/               # 张量构造模块（多文件设计）
    ├── mod.rs               # 模块根，re-exports 所有公共 API
    ├── fill.rs              # zeros, ones, fill（填充构造）
    ├── eye.rs               # eye（单位矩阵）
├── from_data.rs         # from_shape_vec, from_shape_slice, from_array（从数据源构造）
    └── from_fn.rs           # from_fn, from_scalar（从闭包/标量构造）
```

多文件设计：每个子文件对应一类构造方式，便于后期扩展（如新增 `from_diag.rs`、`linspace.rs` 等）。`mod.rs` 负责统一 re-exports，对外保持单一模块接口。

---

## 3. 依赖关系

### 3.1 依赖图（ASCII）

```
                    ┌──────────────┐
                    │   tensor     │
                    │ TensorBase   │
                    └──────┬───────┘
                           │ 使用
              ┌────────────▼───────────┐
              │  construct/            │
              │  mod.rs (re-exports)   │
              └──┬───────────┬─────────┘
                 │ 使用       │ 使用
          ┌──────▼───┐ ┌──────▼────────┐
           │ storage  │ │ layout        │
          │ Owned<A> │ │ LayoutFlags   │
          │ Storage  │ │ Order         │
          └──────────┘ └───────────────┘
```

### 3.2 类型级依赖

| 来源模块 | 使用的类型/trait |
|----------|-----------------|
| `tensor` | `TensorBase<S, D>`, `Tensor<A, D>`, 类型别名 `Tensor0`~`Tensor6`（参见 `07-tensor.md` §4） |
| `storage` | `Owned<A>`, `Storage<Elem = A>`, `from_vec_aligned()`（参见 `05-storage.md` §4） |
| `layout` | `LayoutFlags`, `Strides<D>`, F-order 步长计算（参见 `06-memory.md` §3） |
| `dimension` | `Dimension`, `Ix0`~`Ix6`, `IxDyn`, `IntoDimension`（参见 `02-dimension.md` §4） |
| `element` | `Element`（`zero()` / `one()` 由 `Element` 提供，参见 `03-element.md` §3） |
| `error` | `XenonError::InvalidShape`（用于构造时的 shape/length 基数不匹配，参见 `26-error.md` §4） |

### 3.3 依赖方向声明

> **依赖方向：单向向上。** `construct` 消费 `tensor`、`storage`、`layout`、`dimension`、`element` 的 trait 和类型，不被它们依赖。

---

## 4. 公共 API 设计

### 4.1 zeros / ones / full

```rust
impl<A, D> TensorBase<Owned<A>, D>
where
    A: Element,
    D: Dimension,
{
    /// Create a zero-initialized tensor (F-order).
    ///
    /// # Examples
    /// ```
    /// let t = Tensor::<f64, _>::zeros([3, 4]);
    /// assert_eq!(t.shape(), &[3, 4]);
    /// assert!(t.iter().all(|&x| x == 0.0));
    /// ```
    pub fn zeros<Sh>(shape: Sh) -> Self
    where
        A: Element,  // A::zero() is provided by the Element trait (see 03-element.md §4.1)
        Sh: IntoDimension<Dim = D>,
    {
        let dim = shape.into_dimension();
        let len = dim.size();
        let strides = dim.strides_for_f_order();
        let storage = Owned::zeros(len);
        TensorBase { storage, shape: dim, strides, offset: 0, flags: LayoutFlags::from_order(Order::F) }
    }

    /// Create a tensor filled with ones (F-order).
    ///
    /// # Examples
    /// ```
    /// let t = Tensor::<f64, _>::ones([2, 3]);
    /// assert!(t.iter().all(|&x| x == 1.0));
    /// ```
    pub fn ones<Sh>(shape: Sh) -> Self
    where
        A: Element,  // A::one() is provided by the Element trait (see 03-element.md §4.1)
        Sh: IntoDimension<Dim = D>,
    {
        Self::full(shape, A::one())
    }

    /// Create a tensor filled with the specified value.
    ///
    /// # Examples
    /// ```
    /// let t = Tensor::<f64, _>::full([2, 2], 3.14);
    /// assert!(t.iter().all(|&x| x == 3.14));
    /// ```
    pub fn full<Sh>(shape: Sh, value: A) -> Self
    where
        A: Clone,
        Sh: IntoDimension<Dim = D>,
    {
        let dim = shape.into_dimension();
        let len = dim.size();
        let strides = dim.strides_for_f_order();
        let storage = Owned::from_elem(len, value);
        TensorBase { storage, shape: dim, strides, offset: 0, flags: LayoutFlags::from_order(Order::F) }
    }
}
```

### 4.2 eye（单位矩阵）

```rust
impl<A> TensorBase<Owned<A>, Ix2>
where
    A: Element,
{
    /// Create an n×n identity matrix.
    ///
    /// Diagonal elements are 1, all others are 0. F-order layout.
    ///
    /// # Examples
    /// ```
    /// let e = Tensor::<f64, Ix2>::eye(3);
    /// assert_eq!(e[[0, 0]], 1.0);
    /// assert_eq!(e[[0, 1]], 0.0);
    /// assert_eq!(e[[1, 1]], 1.0);
    /// ```
    pub fn eye(n: usize) -> Self
    where
        A: Element,
    {
        let mut result = Self::zeros([n, n]);
        for i in 0..n {
            result[[i, i]] = A::one();
        }
        result
    }
}
```

### 4.3 from_shape_vec / from_shape_slice / from_array

```rust
impl<A, D> TensorBase<Owned<A>, D>
where
    A: Element,
    D: Dimension,
{
    /// Construct a tensor from a Vec (with specified shape, F-order).
    ///
    /// Validates that the Vec length matches the total number of elements in the shape.
    ///
    /// # Data Layout
    /// The elements in `data` must be laid out in **column-major (F-order)** order.
    /// That is, the first dimension varies fastest.
    ///
    /// # Errors
    /// Returns `XenonError::InvalidShape` if `data.len() != shape.size()`.
    ///
    /// # Examples
    /// ```
    /// let t = Tensor::<f64, _>::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6])?;
    /// assert_eq!(t.shape(), &[2, 3]);
    /// ```
    pub fn from_shape_vec<Sh>(shape: Sh, data: Vec<A>) -> Result<Self, XenonError>
    where
        Sh: IntoDimension<Dim = D>,
    {
        let dim = shape.into_dimension();
        let expected = dim.size();
        if data.len() != expected {
            return Err(XenonError::InvalidShape {
                from: data.len(),
                to: expected,
            });
        }
        let strides = dim.strides_for_f_order();
        // from_vec_aligned: defined in 05-storage.md §5.1 and 21-type.md §5.1;
        // copies data into Xenon's aligned allocation for SIMD compatibility.
        let storage = Owned::from_vec_aligned(data);
        Ok(TensorBase { storage, shape: dim, strides, offset: 0, flags: LayoutFlags::from_order(Order::F) })
    }

    /// Construct a tensor from a slice (copies data).
    ///
    /// # Errors
    /// Returns `XenonError::InvalidShape` if `slice.len() != shape.size()`.
    ///
    /// # Examples
    /// ```
    /// let data = [1.0, 2.0, 3.0, 4.0];
    /// let t = Tensor::<f64, _>::from_shape_slice([2, 2], &data)?;
    /// ```
    pub fn from_shape_slice<Sh>(shape: Sh, slice: &[A]) -> Result<Self, XenonError>
    where
        A: Clone,
        Sh: IntoDimension<Dim = D>,
    {
        let dim = shape.into_dimension();
        let expected = dim.size();
        if slice.len() != expected {
            return Err(XenonError::InvalidShape {
                from: slice.len(),
                to: expected,
            });
        }
        Self::from_shape_vec(dim, slice.to_vec())
    }

    /// Construct a tensor from a fixed-size array.
    ///
    /// # Examples
    /// ```
    /// let arr = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let t = Tensor::<f64, _>::from_array([2, 3], arr)?;
    /// ```
    pub fn from_array<Sh, const N: usize>(
        shape: Sh,
        arr: [A; N],
    ) -> Result<Self, XenonError>
    where
        Sh: IntoDimension<Dim = D>,
    {
        Self::from_shape_vec(shape, arr.into_iter().collect())
    }
}
```

### 4.4 from_scalar / from_fn

```rust
impl<A> TensorBase<Owned<A>, Ix0>
where
    A: Element,
{
    /// Construct a zero-dimensional tensor from a scalar.
    ///
    /// # Examples
    /// ```
    /// let t = Tensor::<f64, Ix0>::from_scalar(3.14);
    /// assert_eq!(t[[]], 3.14);
    /// ```
    pub fn from_scalar(scalar: A) -> Self {
        TensorBase {
            storage: Owned::from_vec_aligned(vec![scalar]),
            shape: Ix0(),
            strides: Strides::<Ix0>::from_slice(&[]),
            offset: 0,
            flags: LayoutFlags::from_order(Order::F),
        }
    }
}

impl<A, D> TensorBase<Owned<A>, D>
where
    A: Element,
    D: Dimension,
{
    /// Construct a tensor from a closure.
    ///
    /// The closure receives the multi-dimensional index of each element and returns its value.
    /// Data is filled in F-order.
    ///
    /// # Examples
    /// ```
    /// let t = Tensor::<i32, _>::from_fn([3, 3], |idx| {
    ///     (idx[0] * 3 + idx[1]) as i32
    /// });
    /// assert_eq!(t[[1, 2]], 5);
    /// ```
    pub fn from_fn<Sh, F>(shape: Sh, mut f: F) -> Self
    where
        Sh: IntoDimension<Dim = D>,
        F: FnMut(&[usize]) -> A,
    {
        let dim = shape.into_dimension();
        let len = dim.size();
        let strides = dim.strides_for_f_order();
        let mut data = Vec::with_capacity(len);
        // Iterate indices in F-order
        let mut idx = vec![0usize; dim.ndim()];
        for _ in 0..len {
            data.push(f(&idx));
            // F-order index increment
            increment_index_f(&dim, &mut idx);
        }
        let storage = Owned::from_vec_aligned(data);
        TensorBase { storage, shape: dim, strides, offset: 0, flags: LayoutFlags::from_order(Order::F) }
    }
}
```

### 4.5 Good / Bad 对比

```rust
// Good - use Result to handle potential shape mismatch
fn create_matrix(data: Vec<f64>) -> Result<Tensor<f64, Ix2>, XenonError> {
    let n = (data.len() as f64).sqrt() as usize;
    if n * n != data.len() {
        return Err(XenonError::InvalidShape {
            from: data.len(),
            to: n * n,
        });
    }
    Tensor::from_shape_vec([n, n], data)
}

// Bad - using unwrap for shape errors in library code
fn create_matrix_bad(data: Vec<f64>) -> Tensor<f64, Ix2> {
    let n = (data.len() as f64).sqrt() as usize;
    Tensor::from_shape_vec([n, n], data).unwrap()  // Forbidden: may panic
}
```

---

## 5. 内部实现设计

### 5.1 内存初始化策略

| 构造方法 | 分配策略 | 初始化 |
|----------|----------|--------|
| `zeros` | 对齐分配 + 零初始化 | `ptr::write_bytes(0)` |
| `ones` | 对齐分配 + 批量填充 | `ptr::write(A::one())` |
| `full` | 对齐分配 + 批量克隆 | `clone` 填充 |
| `from_shape_vec` | 复制到对齐分配（按元素类型选择 Xenon 的标准对齐） | 用户提供数据 |
| `from_shape_slice` | 先复制到临时 Vec，再复制到对齐分配 | 两次数据搬运（保持 API 简洁） |
| `from_fn` | 先写入临时 Vec，再复制到对齐分配 | 闭包逐元素调用 + 一次最终复制 |
| `from_scalar` | 对齐分配（1 元素） | 单元素写入 |
| `eye` | 先 `zeros` 再对角线写入 | 两步：零初始化 + 对角线 |

### 5.2 F-order 数据填充顺序

`from_fn` 中数据按 F-order 顺序填充：

```rust
function increment_index_f(shape, index):
    for i in 0..ndim:
        index[i] += 1
        if index[i] < shape[i]:
            return  // no carry
        index[i] = 0  // carry to next dimension
```

示例（`shape=[2,3]`）：
```
填充顺序: [0,0] → [1,0] → [0,1] → [1,1] → [0,2] → [1,2]
内存布局: data[0], data[1], data[2], data[3], data[4], data[5]
```

### 5.3 安全性论证

- `from_shape_vec`: 验证 `data.len() == dim.size()`，不匹配返回错误；通过后 `Owned::from_vec_aligned` 消费输入 Vec 并复制到对齐存储
- `from_shape_slice`: 验证长度后拷贝，原始切片不再被引用
- `from_fn`: 闭包接收合法索引（`0 <= idx[i] < shape[i]`），无越界风险
- `dim.size()` 溢出或分配失败：构造路径须在实现中转为 `XenonError` 或 panic-by-policy，不得产生未定义行为；ZST 与空张量路径须与 `05-storage.md` 的约束保持一致
- `eye`: 内部使用已验证的 `zeros` 和合法索引 `[[i, i]]`（`0 <= i < n`），无越界风险

---

## 6. 实现任务拆分

### Wave 1: 基础构造

- [ ] **T1**: 创建 `src/construct/` 模块骨架 + `zeros`/`ones`/`full`
  - 文件: `src/construct/mod.rs`, `src/construct/fill.rs`
  - 内容: 模块声明、`zeros`/`ones`/`full` 实现
  - 测试: `test_zeros`, `test_ones`, `test_full`
  - 前置: `tensor` 模块完成
  - 预计: 10 min

- [ ] **T2**: 实现 `eye` 单位矩阵
  - 文件: `src/construct/eye.rs`
  - 内容: 单位矩阵构造
  - 测试: `test_eye`, `test_eye_zero_size`
  - 前置: T1
  - 预计: 5 min

### Wave 2: 从数据源构造

- [ ] **T3**: 实现 `from_shape_vec` 和 `from_shape_slice`
  - 文件: `src/construct/from_data.rs`
  - 内容: 消费输入 Vec 并复制到对齐存储、从切片拷贝
  - 测试: `test_from_shape_vec`, `test_from_shape_vec_mismatch`, `test_from_shape_slice`
  - 前置: T1
  - 预计: 10 min

- [ ] **T4**: 实现 `from_array` 和 `from_scalar`
  - 文件: `src/construct/from_data.rs`, `src/construct/from_fn.rs`
  - 内容: 从固定数组构造、零维张量
  - 测试: `test_from_array`, `test_from_scalar`
  - 前置: T3
  - 预计: 5 min

### Wave 3: 高级构造

- [ ] **T5**: 实现 `from_fn`
  - 文件: `src/construct/from_fn.rs`
  - 内容: 从闭包构造、F-order 索引递增
  - 测试: `test_from_fn`, `test_from_fn_rect`
  - 前置: T3
  - 预计: 10 min

### Wave 4: 集成与测试

- [ ] **T6**: 编写综合测试
  - 文件: `tests/construct.rs`
  - 内容: 各构造方法的集成测试、边界情况
  - 测试: 覆盖所有公共 API
  - 前置: T1-T5
  - 预计: 10 min

### 并行执行图

```
Wave 1: [T1] → [T2]
                  │
Wave 2:      [T3] → [T4]
                      │
Wave 3:           [T5]
                      │
Wave 4:           [T6]
```

---

## 7. 测试计划

### 7.0 测试分类表

| 测试分类 | 位置 | 说明 |
|----------|------|------|
| 单元测试 | `#[cfg(test)] mod tests` | 验证各类构造 API 的核心正确性 |
| 集成测试 | `tests/` | 验证 `construction` 与 `dimension`、`storage`、`layout`、`tensor`、`index` 的协同路径 |
| 边界测试 | 同模块测试中标注 | 覆盖空张量、零维张量和大张量分配等边界 |

### 7.1 单元测试清单

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_zeros_shape` | `zeros([3, 4])` 形状正确 | 高 |
| `test_zeros_values` | 所有元素为零 | 高 |
| `test_ones_values` | 所有元素为一 | 高 |
| `test_full_custom` | 自定义值填充 | 高 |
| `test_eye_3x3` | 3×3 单位矩阵对角线为 1 | 高 |
| `test_eye_zero` | `eye(0)` 空矩阵 | 中 |
| `test_from_shape_vec_success` | 合法 Vec 构造成功 | 高 |
| `test_from_shape_vec_mismatch` | Vec 长度不匹配返回错误 | 高 |
| `test_from_shape_slice_success` | 从切片构造 | 高 |
| `test_from_shape_slice_mismatch` | 切片长度不匹配 | 高 |
| `test_from_array_success` | 从固定数组构造 | 中 |
| `test_from_scalar` | 零维张量 | 高 |
| `test_from_fn_identity` | `from_fn([3,3], |i| i[0]*3+i[1])` | 高 |
| `test_from_fn_empty` | 空数组 | 中 |

### 7.2 边界测试场景

| 场景 | 预期行为 |
|------|----------|
| `zeros([0])` | 空张量，`len() == 0` |
| `zeros([0, 3])` | 空张量，`len() == 0` |
| `eye(0)` | 空 0×0 矩阵 |
| `from_scalar(42)` | 零维张量，`ndim() == 0` |
| `from_shape_vec([0], vec![])` | 空 1D 张量 |
| `from_fn([0, 5], \|\_)` | 空 2D 张量 |
| 大张量 `zeros([1000, 1000])` | 分配成功，F-order 连续 |

### 7.3 属性测试不变量

| 不变量 | 测试方法 |
|--------|----------|
| `zeros(s).iter().all(\|x\| x == Element::zero())` | 随机形状 |
| `ones(s).iter().all(\|x\| x == Element::one())` | 随机形状 |
| `from_shape_vec(s, v).len() == s.size()` | 随机形状和匹配数据 |
| `from_fn(s, f).shape() == s` | 随机形状 |

### 7.4 集成测试

| 测试文件 | 测试内容 |
|----------|----------|
| `tests/construction.rs` | 构造 API 与 `dimension`、`storage`、`layout`、`tensor`、`index` 的端到端集成 |

---

## 8. 与其他模块的交互

### 8.1 接口约定

| 方向 | 对方模块 | 接口/类型 | 约定 |
|------|----------|-----------|------|
| `construct → tensor` | `tensor` | `TensorBase` | 构造张量实例，参见 `07-tensor.md` §4.1 |
| `construct → storage` | `storage` | `Owned::zeros()` / `from_vec_aligned()` | 使用对齐存储完成底层分配，参见 `05-storage.md` §4.2 |
| `construct → layout` | `layout` | F-order 步长 | 构造阶段计算 F-order 步长，参见 `06-memory.md` §4 |
| `construct → dimension` | `dimension` | `IntoDimension` | 接受灵活形状参数并归一化，参见 `02-dimension.md` §4.3 |
| `construct → element` | `element` | `Element` | 通过 `Element::zero()` / `Element::one()` 约束构造 API，参见 `03-element.md` §3 |
| `construct → error` | `error` | `XenonError::InvalidShape` | shape 与长度基数不匹配时返回错误，参见 `26-error.md` §4 |
| `construct → index` | `index` | 索引访问语义 | 构造后的张量继续复用索引路径，参见 `17-indexing.md` §4 |

### 8.2 数据流描述

```text
用户调用 zeros / from_shape_vec / from_fn / eye
    │
    ├── dimension 模块先规范化输入 shape
    ├── layout 计算 F-order strides 与初始 flags
    ├── storage 分配 aligned owned buffer 并写入数据
    └── tensor 模块封装成 TensorBase<Owned<_>, D>，随后可被 index / iter / math 使用
```

---

## 9. 设计决策记录

### ADR-1: from_fn 使用闭包而非迭代器

| 属性 | 值 |
|------|-----|
| 决策 | `from_fn` 接收 `FnMut(&[usize]) -> A` 闭包，按 F-order 遍历 |
| 理由 | 灵活性高（任意初始化逻辑）；F-order 遍历保证数据布局一致性（参见 `06-memory.md` §3.2） |
| 替代方案 | 接收 `Iterator<Item=A>` — 放弃，不提供多维索引信息 |
| 替代方案 | 接收 `FnMut(usize) -> A`（线性索引） — 放弃，用户需要自行计算多维索引 |

### ADR-2: eye 实现方式

| 属性 | 值 |
|------|-----|
| 决策 | `eye` 先 `zeros` 再逐个写入对角线元素 |
| 理由 | 实现简单；对角线元素数量远少于总元素（n vs n²），写入开销可忽略 |
| 替代方案 | 使用 `from_fn` 构造 — 放弃，每次调用闭包需要条件判断，性能差 |
| 替代方案 | 一次性分配并手动设置对角线 — 放弃，代码复杂度高 |

---

## 10. 性能考量

### 10.1 复杂度

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| `zeros(n)` | O(n) | O(n) |
| `ones(n)` | O(n) | O(n) |
| `full(n, v)` | O(n) | O(n) |
| `eye(n)` | O(n²) | O(n²) |
| `from_shape_vec(n)` | O(n) 拷贝到对齐内存 | O(n)（新分配） |
| `from_shape_slice(n)` | O(n) 拷贝 | O(n) |
| `from_array(n)` | O(n) 拷贝 | O(n) |
| `from_scalar()` | O(1) | O(1) |
| `from_fn(n, f)` | O(n × f_cost) | O(n) |

### 10.2 对齐分配

| 元素类型 | 对齐要求 | 分配方式 |
|----------|----------|----------|
| `f64` | 64 字节（AVX-512） | `alloc_aligned(64)` |
| `f32` | 32 字节（AVX） | `alloc_aligned(32)` |
| `Complex<f64>` | 64 字节 | `alloc_aligned(64)` |
| `i32`/`i64` | 16 字节 | `alloc_aligned(16)` |
| `bool` | 1 字节 | `alloc_aligned(1)` |

### 10.3 批量初始化优化

| 场景 | 优化方式 | 预期性能 |
|------|----------|----------|
| `zeros` 大数组 | `ptr::write_bytes(0)` | ~10 GB/s（memset 速度） |
| `ones` 大数组 | `ptr::write(value)` 循环 | ~5 GB/s |
| `full` 大数组 | `clone` 循环 | ~3 GB/s（含克隆开销） |
| `from_shape_vec` | 拷贝到对齐内存 | O(n)（拷贝到 64 字节对齐分配） |
| `eye` 大矩阵 | 先零后对角 | n 次 `write` + n² 次 `write_bytes` |

> **注意**：`from_array` 当前存在双重拷贝（源数组 → Vec → 对齐分配），未来版本可优化为直接构建。

---

## 11. no_std 兼容性

构造操作模块在 `no_std` 环境下可用，但需 `alloc` 支持以进行内存分配。所有构造方法均需要堆分配来存储张量数据。存储层的 `no_std` 兼容性参见 `05-storage.md` §11。

```rust
#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
```

| 组件 | no_std 支持 | 说明 |
|------|:----------:|------|
| `zeros()` | ✅ | 需 `no_std + alloc`，对齐分配 + 零初始化 |
| `ones()` | ✅ | 需 `no_std + alloc`，对齐分配 + 批量填充 |
| `full()` | ✅ | 需 `no_std + alloc`，对齐分配 + 批量克隆 |
| `eye()` | ✅ | 需 `no_std + alloc`，先 `zeros` 再写入对角线 |
| `from_shape_vec()` | ✅ | 需 `no_std + alloc`，拷贝数据到 64 字节对齐内存（O(n)） |
| `from_shape_slice()` | ✅ | 需 `no_std + alloc`，拷贝到新 `Vec` |
| `from_array()` | ✅ | 需 `no_std + alloc`，转换为 `Vec` |
| `from_scalar()` | ✅ | 需 `no_std + alloc`，单元素 `Vec` |
| `from_fn()` | ✅ | 需 `no_std + alloc`，闭包填充 `Vec` |

条件编译处理：

```rust
// All constructors allocate via Owned storage → alloc::vec::Vec
// Alignment uses core::alloc::Layout + alloc::alloc::alloc_zeroed

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
```

---

## 版本历史

| 版本 | 日期 |
|------|------|
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-07 |
| 1.0.2 | 2026-04-08 |
| 1.0.3 | 2026-04-08 |
| 1.0.4 | 2026-04-08 |
| 1.1.0 | 2026-04-08 |
| 1.1.1 | 2026-04-08 |
| 1.1.2 | 2026-04-10 |

---

*本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。*
