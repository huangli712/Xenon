# 构造操作模块设计

> 文档编号: 18 | 模块: `src/construct.rs` | 阶段: Phase 4
> 前置文档: `07-tensor.md`, `05-storage.md`
> 需求参考: 需求说明书 §19

---

## 1. 模块定位

### 1.1 职责边界

| 职责 | 包含 | 不包含 |
|------|------|--------|
| 零初始化构造 | `zeros<A, D>(shape)` 全零张量 | arange / linspace / logspace 序列生成（当前版本不提供） |
| 常量填充构造 | `ones<A, D>(shape)` / `fill<A, D>(scalar, shape)` 全一/填充张量 | 随机数构造（当前版本不提供） |
| 单位矩阵 | `eye<A>(n)` 单位矩阵 | 从文件加载（当前版本不提供） |
| 从 Vec 构造 | `from_vec<A>(vec, shape)` 从 Vec 转移所有权 | 从迭代器/生成器构造（由 `from_fn` 提供） |
| 从切片构造 | `from_slice<A>(slice, shape)` 从切片拷贝数据 | 从文件/网络加载（当前版本不提供） |
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
| 零拷贝优先 | `from_vec` 转移所有权，不拷贝数据 |
| 类型安全 | 彣状和元素类型通过泛型约束在编译期检查 |

### 1.3 在架构中的位置

```
依赖层级：
L0: error, private
L1: dimension, element, complex
L2: layout (依赖 dimension)
L3: storage (依赖 layout)
L4: tensor (依赖 storage, dimension)
L5: construct  ← 当前模块
```

---

## 2. 文件位置

```
src/
└── construct.rs       # 张量构造方法（单文件）
```

单文件设计：构造方法逻辑高度内聚，不依赖复杂子模块拆分。包含所有构造方法的实现。

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
              │  construct             │
              │  construct.rs          │
              └──┬───────────┬─────────┘
                 │ 使用       │ 使用
          ┌──────▼───┐ ┌──────▼────────┐
          │ storage  │ │ memory-layout │
          │ Owned<A> │ │ LayoutFlags   │
          │ Storage  │ │ Order         │
          └──────────┘ └───────────────┘
```

### 3.2 类型级依赖

| 来源模块 | 使用的类型/trait |
|----------|-----------------|
| `tensor` | `TensorBase<S, D>`, `Tensor<A, D>`, 类型别名 `Tensor0`~`Tensor6`（参见 `07-tensor.md` §4） |
| `storage` | `Owned<A>`, `Storage<Elem = A>`, `from_vec_aligned()`（参见 `05-storage.md` §4） |
| `memory_layout` | `LayoutFlags`, `Order::F`, F-order 步长计算（参见 `06-memory-layout.md` §3） |
| `dimension` | `Dimension`, `Ix0`~`Ix6`, `IxDyn`, `IntoDimension`（参见 `02-dimension.md` §4） |
| `element` | `Element`, `Zero`, `One`（参见 `03-element-types.md` §3） |
| `error` | `XenonError::InvalidShape`（参见 `26-error-handling.md` §4） |

### 3.3 依赖方向声明

> **依赖方向：单向向上。** `construct` 消费 `tensor`、`storage`、`memory_layout`、`dimension`、`element` 的 trait 和类型，不被它们依赖。

---

## 4. 公共 API 设计

### 4.1 zeros / ones / fill

```rust
impl<A, D> Tensor<A, D>
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
    /// assert!(t.iter().all(|&x| x == &0.0));
    /// ```
    pub fn zeros<Sh>(shape: Sh) -> Self
    where
        A: Zero,
        Sh: IntoDimension<Dim = D>,
    {
        let dim = shape.into_dimension();
        let len = dim.size();
        let strides = dim.strides_for_f_order();
        let storage = Owned::zeros(len);
        TensorBase { storage, shape: dim, strides, offset: 0 }
    }

    /// Create a tensor filled with ones (F-order).
    ///
    /// # Examples
    /// ```
    /// let t = Tensor::<f64, _>::ones([2, 3]);
    /// assert!(t.iter().all(|&x| x == &1.0));
    /// ```
    pub fn ones<Sh>(shape: Sh) -> Self
    where
        A: Zero + One,
        Sh: IntoDimension<Dim = D>,
    {
        Self::full(shape, A::one())
    }

    /// Create a tensor filled with the specified value.
    ///
    /// # Examples
    /// ```
    /// let t = Tensor::<f64, _>::fill(3.14, [2, 2]);
    /// assert!(t.iter().all(|&x| x == &3.14));
    /// ```
    pub fn fill<Sh>(value: A, shape: Sh) -> Self
    where
        A: Clone,
        Sh: IntoDimension<Dim = D>,
    {
        let dim = shape.into_dimension();
        let len = dim.size();
        let strides = dim.strides_for_f_order();
        let storage = Owned::from_elem(len, value);
        TensorBase { storage, shape: dim, strides, offset: 0 }
    }
}
```

### 4.2 eye（单位矩阵）

```rust
impl<A> Tensor<A, Ix2>
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
        A: Zero + One,
    {
        let mut result = Self::zeros([n, n]);
        for i in 0..n {
            result[[i, i]] = A::one();
        }
        result
    }
}
```

### 4.3 from_vec / from_slice / from_array

```rust
impl<A, D> Tensor<A, D>
where
    A: Element,
    D: Dimension,
{
    /// Construct a tensor from a Vec (with specified shape, F-order).
    ///
    /// Validates that the Vec length matches the total number of elements in the shape.
    ///
    /// # Errors
    /// Returns `InvalidShape` if `data.len() != shape.size()`.
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
                expected,
                actual: data.len(),
                reason: "element count mismatch",
            });
        }
        let strides = dim.strides_for_f_order();
        let storage = Owned::from_vec_aligned(data);
        Ok(TensorBase { storage, shape: dim, strides, offset: 0 })
    }

    /// Construct a tensor from a slice (copies data).
    ///
    /// # Errors
    /// Returns `InvalidShape` if `slice.len() != shape.size()`.
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
                expected,
                actual: slice.len(),
                reason: "element count mismatch",
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
impl<A> Tensor<A, Ix0>
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
            strides: Ix0(),
            offset: 0,
        }
    }
}

impl<A, D> Tensor<A, D>
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
        TensorBase { storage, shape: dim, strides, offset: 0 }
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
            expected: n * n,
            actual: data.len(),
            reason: "not a perfect square",
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
| `ones` | 对齐分配 + 批量填充 | `ptr::write(One::one())` |
| `fill` | 对齐分配 + 批量克隆 | `clone` 填充 |
| `from_vec` | 对齐转移（不拷贝） | 用户提供数据 |
| `from_slice` | 对齐分配 + 拷贝 | `copy_from_slice` |
| `from_fn` | 对齐分配 + 闭包填充 | 闭包逐元素调用 |
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

- `from_vec`: 验证 `data.len() == dim.size()`，不匹配返回错误；通过后 `Owned::from_vec_aligned` 转移所有权
- `from_slice`: 验证长度后拷贝，原始切片不再被引用
- `from_fn`: 闭包接收合法索引（`0 <= idx[i] < shape[i]`），无越界风险
- `eye`: 内部使用已验证的 `zeros` 和合法索引 `[[i, i]]`（`0 <= i < n`），无越界风险

---

## 6. 实现任务拆分

### Wave 1: 基础构造

- [ ] **T1**: 创建 `src/construct.rs` 骨架 + `zeros`/`ones`/`fill`
  - 文件: `src/construct.rs`
  - 内容: 模块声明、`zeros`/`ones`/`fill` 实现
  - 测试: `test_zeros`, `test_ones`, `test_fill`
  - 前置: `tensor` 模块完成
  - 预计: 10 min

- [ ] **T2**: 实现 `eye` 单位矩阵
  - 文件: `src/construct.rs`
  - 内容: 单位矩阵构造
  - 测试: `test_eye`, `test_eye_zero_size`
  - 前置: T1
  - 预计: 5 min

### Wave 2: 从数据源构造

- [ ] **T3**: 实现 `from_vec` 和 `from_slice`
  - 文件: `src/construct.rs`
  - 内容: 从 Vec 转移所有权、从切片拷贝
  - 测试: `test_from_vec`, `test_from_vec_mismatch`, `test_from_slice`
  - 前置: T1
  - 预计: 10 min

- [ ] **T4**: 实现 `from_array` 和 `from_scalar`
  - 文件: `src/construct.rs`
  - 内容: 从固定数组构造、零维张量
  - 测试: `test_from_array`, `test_from_scalar`
  - 前置: T3
  - 预计: 5 min

### Wave 3: 高级构造

- [ ] **T5**: 实现 `from_fn`
  - 文件: `src/construct.rs`
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

### 7.1 单元测试

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_zeros_shape` | `zeros([3, 4])` 形状正确 | 高 |
| `test_zeros_values` | 所有元素为零 | 高 |
| `test_ones_values` | 所有元素为一 | 高 |
| `test_fill_custom` | 自定义值填充 | 高 |
| `test_eye_3x3` | 3×3 单位矩阵对角线为 1 | 高 |
| `test_eye_zero` | `eye(0)` 空矩阵 | 中 |
| `test_from_vec_success` | 合法 Vec 构造成功 | 高 |
| `test_from_vec_mismatch` | Vec 长度不匹配返回错误 | 高 |
| `test_from_slice_success` | 从切片构造 | 高 |
| `test_from_slice_mismatch` | 切片长度不匹配 | 高 |
| `test_from_array_success` | 从固定数组构造 | 中 |
| `test_from_scalar` | 零维张量 | 高 |
| `test_from_fn_identity` | `from_fn([3,3], |i| i[0]*3+i[1])` | 高 |
| `test_from_fn_empty` | 空数组 | 中 |

### 7.2 边界测试

| 场景 | 预期行为 |
|------|----------|
| `zeros([0])` | 空张量，`len() == 0` |
| `zeros([0, 3])` | 空张量，`len() == 0` |
| `eye(0)` | 空 0×0 矩阵 |
| `from_scalar(42)` | 零维张量，`ndim() == 0` |
| `from_vec(vec![], [0])` | 空 1D 张量 |
| `from_fn([0, 5], \|\_)` | 空 2D 张量 |
| 大张量 `zeros([1000, 1000])` | 分配成功，F-order 连续 |

### 7.3 属性测试不变量

| 不变量 | 测试方法 |
|--------|----------|
| `zeros(s).iter().all(\|x\| x == Zero::zero())` | 随机形状 |
| `ones(s).iter().all(\|x\| x == One::one())` | 随机形状 |
| `from_vec(v, s).len() == s.size()` | 随机形状和匹配数据 |
| `from_fn(s, f).shape() == s` | 随机形状 |

---

## 8. 与其他模块的交互

| 交互模块 | 方向 | 说明 |
|----------|------|------|
| `tensor` | construct → tensor | 构造 `TensorBase` 实例（参见 `07-tensor.md` §4.1） |
| `storage` | construct → storage | 使用 `Owned::zeros()`/`from_vec_aligned()`（参见 `05-storage.md` §4.2） |
| `memory_layout` | construct → memory_layout | 计算 F-order 步长（参见 `06-memory-layout.md` §4） |
| `dimension` | construct → dimension | 使用 `IntoDimension` 接受灵活形状参数（参见 `02-dimension.md` §4.3） |
| `element` | construct → element | 使用 `Element`/`Zero`/`One` trait 约束（参见 `03-element-types.md` §3） |
| `error` | construct → error | 返回 `XenonError::InvalidShape`（参见 `26-error-handling.md` §4.2） |
| `index` | index ← construct | 构造后可通过索引访问元素（参见 `17-indexing.md` §4） |

---

## 9. 设计决策记录

### 决策 1: from_fn 使用闭包而非迭代器

| 属性 | 值 |
|------|-----|
| 决策 | `from_fn` 接收 `FnMut(&[usize]) -> A` 闭包，按 F-order 遍历 |
| 理由 | 灵活性高（任意初始化逻辑）；F-order 遍历保证数据布局一致性（参见 `06-memory-layout.md` §3.2） |
| 替代方案 | 接收 `Iterator<Item=A>` — 放弃，不提供多维索引信息 |
| 替代方案 | 接收 `FnMut(usize) -> A`（线性索引） — 放弃，用户需要自行计算多维索引 |

### 决策 2: eye 实现方式

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
| `fill(n, v)` | O(n) | O(n) |
| `eye(n)` | O(n²) | O(n²) |
| `from_vec(n)` | O(1) 验证 + 对齐 | O(n)（转移） |
| `from_slice(n)` | O(n) 拷贝 | O(n) |
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
| `fill` 大数组 | `clone` 循环 | ~3 GB/s（含克隆开销） |
| `from_vec` | 转移所有权 | O(1)（不拷贝） |
| `eye` 大矩阵 | 先零后对角 | n 次 `write` + n² 次 `write_bytes` |

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
| `fill()` | ✅ | 需 `no_std + alloc`，对齐分配 + 批量克隆 |
| `eye()` | ✅ | 需 `no_std + alloc`，先 `zeros` 再写入对角线 |
| `from_shape_vec()` | ✅ | 需 `no_std + alloc`，转移 `Vec` 所有权 |
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

---

*本文档由 Xenon 维护。如有问题请提交 Issue 或 PR。*
