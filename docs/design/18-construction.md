# 构造操作模块设计

> 文档编号: 18 | 模块: `src/construct/` | 阶段: Phase 4
> 前置文档: `07-tensor.md`, `05-storage.md`
> 需求参考: 需求说明书 §19
> 范围声明: 范围内

---

## 1. 模块定位

### 1.1 职责边界

| 职责           | 包含                                                 | 不包含                                                         |
| -------------- | ---------------------------------------------------- | -------------------------------------------------------------- |
| 零初始化构造   | `zeros<A, D>(shape)` 全零张量                        | arange / linspace / logspace 序列生成（当前版本不提供）        |
| 常量构造       | `ones<A, D>(shape)` 全一张量                         | `full<A, D>(shape, scalar)` 与随机数构造（当前版本不提供）     |
| 单位矩阵       | `eye<A>(n)` 单位矩阵                                 | 从文件加载（当前版本不提供）                                   |
| 从 Vec 构造    | `from_shape_vec(shape, vec)` 消费输入 Vec 并构造张量 | 从迭代器/生成器构造（当前版本不提供）                         |
| 从切片构造     | `from_shape_slice(shape, slice)` 从切片拷贝数据      | 从文件/网络加载（当前版本不提供）                              |
| 从固定数组构造 | `from_array<A, N>(shape, arr)` 从固定大小数组构造    | 从文件加载（当前版本不提供）                                   |
| 从标量构造     | `from_scalar<A>(scalar)` 零维张量                    | —                                                              |
| 未来扩展构造   | —                                                    | `from_fn<A, D, F>(shape, f)` 与从迭代器/生成器构造留待后续版本 |
| 合法性验证     | 所有构造路径验证形状/大小合法性                      | 隐式类型转换（由 `convert` 模块提供）                          |

### 1.2 设计原则

| 原则         | 体现                                                                                                                                                             |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 合法性验证   | 所有构造路径须验证合法性，防止越界访问（需求说明书 §8）                                                                                                          |
| F-order 默认 | 构造时数据按 F-order 存放，默认列优先布局                                                                                                                        |
| 对齐分配     | `zeros`/`ones` 使用对齐分配器，满足 BLAS 兼容性（参见 `23-ffi.md` §4.5）                                                                                         |
| 对齐优先     | `from_shape_vec` 将输入 `Vec<A>` 的数据复制到新分配的 64 字节对齐内存中（通过 `Owned::from_vec_aligned()`），确保 SIMD 友好的内存对齐；不承诺复用原始 Vec 的分配 |
| 类型安全     | 形状和元素类型通过泛型约束在编译期检查                                                                                                                           |

### 1.3 在架构中的位置

```
Dependency layers:
L0: error, private
L1: dimension, element, complex
L2: layout (depends on dimension)
L3: storage (independent of layout; owned by tensor and consumes layout results)
L4: tensor (depends on storage, dimension)
L5: construct  <- current module (depends on storage, layout, dimension, element)
```

---

## 2. 需求映射与范围约束

| 类型     | 内容 |
| -------- | ---- |
| 需求映射 | 需求说明书 §19 |
| 范围内   | `zeros` / `ones` / `eye` / `from_shape_vec` / `from_shape_slice` / `from_array` / `from_scalar`。 |
| 范围外   | arange、linspace、from_fn、随机构造器与其他序列生成 API。 |
| 非目标   | 不新增新的构造器家族，不改变 F-order / 对齐分配基线，也不引入第三方随机或数据加载依赖。 |

---

## 3. 文件位置

```
src/
└── construct/               # tensor construction module (multi-file design)
    ├── mod.rs               # module root, re-exports all public APIs
    ├── init.rs              # zeros, ones (basic initialization constructors)
    ├── eye.rs               # eye (identity matrix)
    ├── from_data.rs         # from_shape_vec, from_shape_slice, from_array (construction from data sources)
    └── scalar.rs            # from_scalar (scalar construction)
```

多文件设计：每个子文件对应一类构造方式，便于后期扩展（如新增 `from_diag.rs` 等）。`mod.rs` 负责统一 re-exports，对外保持单一模块接口。

> 注：`fill()` 不属于 construction 模块；该能力由 `20-utility.md` 定义，对应 `src/util/fill.rs`。

---

## 4. 依赖关系

### 4.1 依赖图（ASCII）

```
                    ┌──────────────┐
                    │    tensor    │
                    │ TensorBase   │
                    └──────┬───────┘
                           │ uses
              ┌────────────▼───────────┐
              │   construct/           │
              │   mod.rs (re-exports)  │
              └──┬───────────┬─────────┘
                 │ uses      │ uses
          ┌──────▼───┐ ┌──────▼────────┐
          │ storage  │ │ layout        │
          │ Owned<A> │ │ LayoutFlags   │
          │ Storage  │ │ Order         │
          └──────────┘ └───────────────┘
```

### 4.2 类型级依赖

| 来源模块    | 使用的类型/trait                                                                           |
| ----------- | ------------------------------------------------------------------------------------------ |
| `tensor`    | `TensorBase<S, D>`, `Tensor<A, D>`, 类型别名 `Tensor0`~`Tensor6`（参见 `07-tensor.md` §5） |
| `storage`   | `Owned<A>`, `Storage<Elem = A>`, `from_vec_aligned()`（参见 `05-storage.md` §5）           |
| `layout`    | `LayoutFlags`, `Strides<D>`, F-order 步长计算（参见 `06-layout.md` §3）                    |
| `dimension` | `Dimension`, `Ix0`~`Ix6`, `IxDyn`, `IntoDimension`（参见 `02-dimension.md` §5）            |
| `element`   | `Element`（`zero()` / `one()` 由 `Element` 提供，参见 `03-element.md` §5.1）               |
| `error`     | `XenonError::InvalidShape`（用于构造时的 shape/length 基数不匹配，参见 `26-error.md` §4）  |

### 4.3 依赖方向声明

> **依赖方向：单向向上。** `construct` 消费 `tensor`、`storage`、`layout`、`dimension`、`element` 的 trait 和类型，不被它们依赖。

### 4.4 依赖合法性与替代方案

| 项目           | 说明 |
| -------------- | ---- |
| 新增第三方依赖 | 无 |
| 合法性结论     | 合法；当前设计仅复用 Xenon 既有模块、标准库以及文档中已声明的项目内可选能力。 |
| 替代方案       | 不适用；当前范围内无需额外第三方依赖。 |

---

## 5. 公共 API 设计

### 5.1 zeros / ones

````rust
# use crate::dimension::{Dimension, IntoDimension};
# use crate::element::Element;
# use crate::error::XenonError;
# use crate::layout::{LayoutFlags, Order};
# use crate::storage::Owned;
# use crate::tensor::{Tensor, TensorBase};
impl<A, D> TensorBase<Owned<A>, D>
where
    A: Element,
    D: Dimension,
{
    /// Create a zero-initialized tensor (F-order).
    ///
    /// # Examples
    /// ```
    /// let t = Tensor::<f64, _>::zeros([3, 4])?;
    /// assert_eq!(t.shape(), &[3, 4]);
    /// assert!(t.iter().all(|&x| x == 0.0));
    /// ```
    pub fn zeros<Sh>(shape: Sh) -> Result<Self, XenonError>
    where
        A: Element,  // A::zero() is provided by the Element trait (see 03-element.md §5.1)
        Sh: IntoDimension<Dim = D>,
    {
        let dim = shape.into_dimension();
        let len = dim.checked_size().ok_or(XenonError::InvalidShape {
            operation: "zeros",
            shape: dim.slice().to_vec(),
            expected_elements: 0,
            actual_elements: usize::MAX,
            offending_dim: None,
        })?;
        let strides = dim.strides_for_f_order();
        let storage = Owned::zeros(len);
        Ok(TensorBase { storage, shape: dim, strides, offset: 0, flags: LayoutFlags::from_order(Order::F) })
    }

    /// Create a tensor filled with ones (F-order).
    ///
    /// # Examples
    /// ```
    /// let t = Tensor::<f64, _>::ones([2, 3])?;
    /// assert!(t.iter().all(|&x| x == 1.0));
    /// ```
    pub fn ones<Sh>(shape: Sh) -> Result<Self, XenonError>
    where
        A: Element,  // A::one() is provided by the Element trait (see 03-element.md §5.1)
        Sh: IntoDimension<Dim = D>,
    {
        let dim = shape.into_dimension();
        let len = dim.checked_size().ok_or(XenonError::InvalidShape {
            operation: "ones",
            shape: dim.slice().to_vec(),
            expected_elements: 0,
            actual_elements: usize::MAX,
            offending_dim: None,
        })?;
        let strides = dim.strides_for_f_order();
        let storage = Owned::ones(len);
        Ok(TensorBase { storage, shape: dim, strides, offset: 0, flags: LayoutFlags::from_order(Order::F) })
    }
}
````

> **范围决策：** `full()` 超出 `require.md` §19 的当前最小构造集合。
> 本文仅承诺 `zeros()`、`ones()`、`eye()`、`from_shape_vec()`、`from_vec()`、`from_shape_slice()`、`from_array()` 与 `from_scalar()`；
> `full()` 留待后续版本单独设计。

> **`bool` 特殊值说明：** `zeros::<bool>()` 对应 `false`，`ones::<bool>()` 对应 `true`（`require.md` §19）。

### 5.2 eye（单位矩阵）

````rust
# use crate::complex::Complex;
# use crate::element::Element;
# use crate::error::XenonError;
# use crate::layout::{LayoutFlags, Order};
# use crate::storage::Owned;
# use crate::tensor::{Tensor, TensorBase};
# use crate::dimension::Ix2;
# pub trait EyeElement: Element {}
# impl EyeElement for i32 {}
# impl EyeElement for i64 {}
# impl EyeElement for f32 {}
# impl EyeElement for f64 {}
# impl EyeElement for Complex<f32> {}
# impl EyeElement for Complex<f64> {}
impl<A> TensorBase<Owned<A>, Ix2>
where
    A: EyeElement,
{
    /// Create an n×n identity matrix.
    ///
    /// Diagonal elements are 1, all others are 0. F-order layout.
    ///
    /// # Examples
    /// ```
    /// let e = Tensor::<f64, Ix2>::eye(3).unwrap();
    /// assert_eq!(e[[0, 0]], 1.0);
    /// assert_eq!(e[[0, 1]], 0.0);
    /// assert_eq!(e[[1, 1]], 1.0);
    /// ```
    pub fn eye(n: usize) -> Result<Self, XenonError>
    where
        A: EyeElement,
    {
        let mut result = Self::zeros([n, n])?;
        for i in 0..n {
            result[[i, i]] = A::one();
        }
        Ok(result)
    }
}
````

```rust
pub trait EyeElement: Element {}

impl EyeElement for i32 {}
impl EyeElement for i64 {}
impl EyeElement for f32 {}
impl EyeElement for f64 {}
impl EyeElement for Complex<f32> {}
impl EyeElement for Complex<f64> {}
```

> **类型约束说明：** `eye()` 不对 `bool` 开放；其适用类型严格限定为 `i32`、`i64`、`f32`、`f64`、`Complex<f32>` 与 `Complex<f64>`，以符合 `require.md` §19。

### 5.3 from_shape_vec / from_shape_slice / from_array

````rust
# use crate::dimension::{Dimension, IntoDimension, Ix1};
# use crate::element::Element;
# use crate::error::XenonError;
# use crate::layout::{LayoutFlags, Order};
# use crate::storage::Owned;
# use crate::tensor::{Tensor, TensorBase};
impl<A, D> TensorBase<Owned<A>, D>
where
    A: Element,
    D: Dimension,
{
    /// Construct a tensor from a Vec with explicit shape.
    ///
    /// Validates that the Vec length matches the total number of elements in the shape.
    ///
    /// # Data Layout
    /// The order of elements in `data` defines the logical index-to-element mapping.
    /// Callers provide values following Xenon's F-order logical indexing semantics,
    /// while the library itself is responsible for materializing the internal
    /// F-order contiguous storage.
    ///
    /// # Errors
    /// Returns `XenonError::InvalidShape` if `checked_size(shape)` overflows or
    /// `data.len() != checked_size(shape)`.
    ///
    /// # Examples
    /// ```
    /// let t = Tensor::<f64, _>::from_shape_vec([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    /// assert_eq!(t.shape(), &[2, 3]);
    ///
    /// // F-order construction example:
    /// // shape = [2, 3], data = vec![1, 2, 3, 4, 5, 6]
    /// //
    /// // F-order mapping (column-major):
    /// // Logical index [0,0] = 1  (data[0])
    /// // Logical index [1,0] = 2  (data[1])
    /// // Logical index [0,1] = 3  (data[2])
    /// // Logical index [1,1] = 4  (data[3])
    /// // Logical index [0,2] = 5  (data[4])
    /// // Logical index [1,2] = 6  (data[5])
    /// //
    /// // Result tensor:
    /// // | 1  3  5 |
    /// // | 2  4  6 |
    /// ```
    pub fn from_shape_vec<Sh>(shape: Sh, data: Vec<A>) -> Result<Self, XenonError>
    where
        Sh: IntoDimension<Dim = D>,
    {
        let dim = shape.into_dimension();
        let expected = dim.checked_size().ok_or(XenonError::InvalidShape {
            operation: "from_shape_vec",
            shape: dim.slice().to_vec(),
            expected_elements: 0,
            actual_elements: data.len(),
            offending_dim: None,
        })?;
        if data.len() != expected {
            return Err(XenonError::InvalidShape {
                operation: "from_shape_vec",
                shape: dim.slice().to_vec(),
                expected_elements: expected,
                actual_elements: data.len(),
                offending_dim: None,
            });
        }
        let strides = dim.strides_for_f_order();
        // from_vec_aligned: defined in 05-storage.md §5.1;
        // copies data into Xenon's aligned allocation for SIMD compatibility.
        let storage = Owned::from_vec_aligned(data);
        Ok(TensorBase { storage, shape: dim, strides, offset: 0, flags: LayoutFlags::from_order(Order::F) })
    }

    /// Construct a tensor from a slice (copies data).
    ///
    /// # Errors
    /// Returns `XenonError::InvalidShape` if `checked_size(shape)` overflows or
    /// `slice.len() != checked_size(shape)`.
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
        let expected = dim.checked_size().ok_or(XenonError::InvalidShape {
            operation: "from_shape_slice",
            shape: dim.slice().to_vec(),
            expected_elements: 0,
            actual_elements: slice.len(),
            offending_dim: None,
        })?;
        if slice.len() != expected {
            return Err(XenonError::InvalidShape {
                operation: "from_shape_slice",
                shape: dim.slice().to_vec(),
                expected_elements: expected,
                actual_elements: slice.len(),
                offending_dim: None,
            });
        }
        Self::from_shape_vec(dim, slice.to_vec())
    }

    /// Construct a 1D tensor directly from a Vec.
    ///
    /// This is a convenience wrapper around
    /// `from_shape_vec(Ix1(data.len()), data)` for 1D construction.
    pub fn from_vec(data: Vec<A>) -> Tensor<A, Ix1> {
        Self::from_shape_vec(Ix1(data.len()), data)
            .expect("Vec -> Tensor1 shape is always valid")
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
````

### 5.4 from_scalar

````rust
# use crate::dimension::Ix0;
# use crate::element::Element;
# use crate::layout::{LayoutFlags, Order, Strides};
# use crate::storage::Owned;
# use crate::tensor::{Tensor, TensorBase};
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
            shape: Ix0,
            strides: Strides::<Ix0>::from_slice(&[]),
            offset: 0,
            flags: LayoutFlags::from_order(Order::F),
        }
    }
}

````

> **范围决策：** `from_fn()` 不在 `require.md` §19 当前版本强制集合中。
> 若后续版本确需提供闭包驱动构造器，应另行评估其 API 粒度、性能模型与错误语义。

### 5.5 Good / Bad 对比

```rust
# use crate::dimension::Ix2;
# use crate::error::XenonError;
# use crate::tensor::Tensor;
// Good - use Result to handle potential shape mismatch
fn create_matrix(data: Vec<f64>) -> Result<Tensor<f64, Ix2>, XenonError> {
    let n = (data.len() as f64).sqrt() as usize;
    if n * n != data.len() {
        return Err(XenonError::InvalidShape {
            operation: "create_matrix",
            shape: vec![n, n],
            expected_elements: n * n,
            actual_elements: data.len(),
            offending_dim: None,
        });
    }
    Tensor::from_shape_vec([n, n], data)
}

// Bad - using unwrap for shape errors in library code
fn create_matrix_bad(data: Vec<f64>) -> Tensor<f64, Ix2> {
    let n = (data.len() as f64).sqrt() as usize;
    Tensor::from_shape_vec([n, n], data).unwrap()  // do not discard the recoverable shape error
}
```

---

## 6. 内部实现设计

### 6.1 内存初始化策略

| 构造方法           | 分配策略                                          | 初始化                        |
| ------------------ | ------------------------------------------------- | ----------------------------- |
| `zeros`            | 对齐分配 + 零初始化                               | `ptr::write_bytes(0)`         |
| `ones`             | 对齐分配 + 批量填充                               | `ptr::write(A::one())`        |
| `from_shape_vec`   | 复制到对齐分配（按元素类型选择 Xenon 的标准对齐） | 用户提供数据                  |
| `from_shape_slice` | 先复制到临时 Vec，再复制到对齐分配                | 两次数据搬运（保持 API 简洁） |
| `from_scalar`      | 对齐分配（1 元素）                                | 单元素写入                    |
| `eye`              | 先 `zeros` 再对角线写入                           | 两步：零初始化 + 对角线       |

### 6.2 范围外能力记录

`full()` 与 `from_fn()` 属于后续版本候选能力，当前文档不继续展开其内部填充策略或任务拆分。

### 6.3 安全性论证

- `from_shape_vec`: 先通过共享 checked helper 验证 `dim.checked_size()`，再验证 `data.len() == expected`；通过后 `Owned::from_vec_aligned` 消费输入 Vec 并复制到对齐存储
- `from_shape_slice`: 先通过共享 checked helper 验证 `dim.checked_size()`，长度匹配后再拷贝，原始切片不再被引用
- `dim.size()` / 步长计算溢出：构造路径须在共享的 checked helper 中统一转为 `XenonError::InvalidShape`，再决定是否由上层便捷构造包装；不得产生未定义行为。ZST 与空张量路径须与 `05-storage.md` 的约束保持一致
- `eye`: 内部使用已验证的 `zeros` 和合法索引 `[[i, i]]`（`0 <= i < n`），无越界风险

---

## 7. 实现任务拆分

### Wave 1: 基础构造

- [ ] **T1**: 创建 `src/construct/` 模块骨架 + `zeros`/`ones`
  - 文件: `src/construct/mod.rs`, `src/construct/init.rs`
  - 内容: 模块声明、`zeros`/`ones` 实现
  - 测试: `test_zeros`, `test_ones`
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
  - 文件: `src/construct/from_data.rs`, `src/construct/scalar.rs`
  - 内容: 从固定数组构造、零维张量
  - 测试: `test_from_array`, `test_from_scalar`
  - 前置: T3
  - 预计: 5 min

- [ ] **T5**: 实现 `from_vec` 一维便捷构造
  - 文件: `src/construct/from_data.rs`
  - 内容: `from_vec(data: Vec<A>) -> Tensor<A, Ix1>`，内部委托到 `from_shape_vec(Ix1(data.len()), data)`
  - 测试: `test_from_vec`
  - 前置: T3
  - 预计: 5 min

### Wave 3: 集成与测试

- [ ] **T6**: 编写综合测试
  - 文件: `tests/test_construction.rs`
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
                  [T5]
                      │
Wave 3:            [T6]
```

---

## 8. 测试计划

### 8.1 测试分类表

| 测试分类 | 位置                     | 说明                                                                                  |
| -------- | ------------------------ | ------------------------------------------------------------------------------------- |
| 单元测试 | `#[cfg(test)] mod tests` | 验证各类构造 API 的核心正确性                                                         |
| 集成测试 | `tests/`                 | 验证 `construction` 与 `dimension`、`storage`、`layout`、`tensor`、`index` 的协同路径 |
| 边界测试 | 同模块测试中标注         | 覆盖空张量、零维张量和大张量分配等边界                                                |
| 属性测试 | `tests/property/`        | 验证 zeros/ones/from_shape_vec 的形状与元素不变量                                     |

### 8.2 单元测试清单

| 测试函数                         | 测试内容                     | 优先级 |
| -------------------------------- | ---------------------------- | ------ |
| `test_zeros_shape`               | `zeros([3, 4])` 形状正确     | 高     |
| `test_zeros_values`              | 所有元素为零                 | 高     |
| `test_ones_values`               | 所有元素为一                 | 高     |
| `test_eye_3x3`                   | 3×3 单位矩阵对角线为 1       | 高     |
| `test_eye_bool_not_supported`    | `eye::<bool>` 在类型层被拒绝 | 高     |
| `test_eye_zero`                  | `eye(0)` 空矩阵              | 中     |
| `test_from_shape_vec_success`    | 合法 Vec 构造成功            | 高     |
| `test_from_shape_vec_mismatch`   | Vec 长度不匹配返回错误       | 高     |
| `test_from_shape_slice_success`  | 从切片构造                   | 高     |
| `test_from_shape_slice_mismatch` | 切片长度不匹配               | 高     |
| `test_from_array_success`        | 从固定数组构造               | 中     |
| `test_from_scalar`               | 零维张量                     | 高     |

### 8.3 边界测试场景

| 场景                          | 预期行为                |
| ----------------------------- | ----------------------- |
| `zeros([0])`                  | 空张量，`len() == 0`    |
| `zeros([0, 3])`               | 空张量，`len() == 0`    |
| `eye(0)`                      | 空 0×0 矩阵             |
| `from_scalar(42)`             | 零维张量，`ndim() == 0` |
| `from_shape_vec([0], vec![])` | 空 1D 张量              |
| 大张量 `zeros([1000, 1000])`  | 分配成功，F-order 连续  |

### 8.4 属性测试不变量

| 不变量                                                     | 测试方法           |
| ---------------------------------------------------------- | ------------------ |
| `zeros(s).iter().all(\|x\| x == Element::zero())`          | 随机形状           |
| `ones(s).iter().all(\|x\| x == Element::one())`            | 随机形状           |
| `from_shape_vec(s, v).len()` 等于 `s` 对应的已验证元素总数 | 随机形状和匹配数据 |

### 8.5 集成测试

| 测试文件                     | 测试内容                                                                     |
| ---------------------------- | ---------------------------------------------------------------------------- |
| `tests/test_construction.rs` | 构造 API 与 `dimension`、`storage`、`layout`、`tensor`、`index` 的端到端集成 |

### 8.6 Feature gate / 配置测试

| 配置 | 验证点 |
| ---- | ---- |
| 默认配置 | 所有构造器在默认构建下保持 F-order、对齐分配与错误语义契约。 |
| 其他 feature 组合 | 不适用；当前模块无额外 feature gate。 |

### 8.7 类型边界 / 编译期测试

| 场景 | 测试方式 |
| ---- | ---- |
| `eye::<bool>` 不被支持 | 编译期测试。 |
| `from_scalar` 仅产出 `Ix0` | 编译期签名检查与运行时断言。 |
| arange / linspace / from_fn / random constructors 不属于当前 API | API 缺失断言。 |

---

## 9. 与其他模块的交互

### 9.1 接口约定

| 方向                    | 对方模块    | 接口/类型                               | 约定                                                                                                                          |
| ----------------------- | ----------- | --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `construct → tensor`    | `tensor`    | `TensorBase`                            | 构造张量实例，参见 `07-tensor.md` §5.1                                                                                        |
| `construct → storage`   | `storage`   | `Owned::zeros()` / `from_vec_aligned()` | 使用对齐存储完成底层分配；`from_vec_aligned()` 当前仅用于 Xenon 封闭元素集合中的 `Copy` 元素快路径，参见 `05-storage.md` §5.1 |
| `construct → layout`    | `layout`    | F-order 步长                            | 构造阶段计算 F-order 步长，参见 `06-layout.md` §4                                                                             |
| `construct → dimension` | `dimension` | `IntoDimension`                         | 接受灵活形状参数并归一化，参见 `02-dimension.md` §5.4                                                                         |
| `construct → element`   | `element`   | `Element`                               | 通过 `Element::zero()` / `Element::one()` 约束构造 API，参见 `03-element.md` §5.1                                             |
| `construct → error`     | `error`     | `XenonError::InvalidShape`              | shape 与长度基数不匹配时返回错误，参见 `26-error.md` §4                                                                       |
| `construct → index`     | `index`     | 索引访问语义                            | 构造后的张量继续复用索引路径，参见 `17-indexing.md` §4                                                                        |

### 9.2 数据流描述

```text
User calls zeros / from_shape_vec / eye
    │
    ├── dimension normalizes the input shape and validates element count
    ├── layout computes F-order strides and initial flags
    ├── storage allocates aligned owned memory and writes data
    └── tensor wraps the result as TensorBase<Owned<_>, D> for later use
```

---

## 10. 错误处理与语义边界

| 主题 | 内容 |
| ---- | ---- |
| Recoverable error | shape 与长度基数不匹配、`checked_size()` 溢出等情况返回 `XenonError::InvalidShape`，携带实际与期望元素数。 |
| Panic | 公开构造 API 不定义额外 panic 语义；失败统一走 `Result`。 |
| 路径一致性 | 所有构造路径都必须产出 canonical F-order owned 张量，并保持一致的 shape / strides / flags 语义。 |
| 容差边界 | 不适用。 |

---

## 11. 设计决策记录

### ADR-1: 当前版本不纳入 `from_fn`

| 属性     | 值                                                             |
| -------- | -------------------------------------------------------------- |
| 决策     | `from_fn` 留待后续版本单独设计，不在当前阶段承诺               |
| 理由     | `require.md` §19 当前仅要求标准构造方法与从数据源/标量构造能力 |
| 替代方案 | 在本阶段继续纳入 `from_fn` — 放弃，会扩大当前版本范围          |

### ADR-2: eye 实现方式

| 属性     | 值                                                                   |
| -------- | -------------------------------------------------------------------- |
| 决策     | `eye` 先 `zeros` 再逐个写入对角线元素                                |
| 理由     | 实现简单；对角线元素数量远少于总元素（n vs n²），写入开销可忽略      |
| 替代方案 | 使用未来的 `from_fn` 构造 — 放弃，在当前范围内没有必要引入额外构造器 |
| 替代方案 | 一次性分配并手动设置对角线 — 放弃，代码复杂度高                      |

---

## 12. 性能考量

### 12.1 复杂度

| 操作                  | 时间复杂度          | 空间复杂度     |
| --------------------- | ------------------- | -------------- |
| `zeros(n)`            | O(n)                | O(n)           |
| `ones(n)`             | O(n)                | O(n)           |
| `eye(n)`              | O(n²)               | O(n²)          |
| `from_shape_vec(n)`   | O(n) 拷贝到对齐内存 | O(n)（新分配） |
| `from_shape_slice(n)` | O(n) 拷贝           | O(n)           |
| `from_array(n)`       | O(n) 拷贝           | O(n)           |
| `from_scalar()`       | O(1)                | O(1)           |

### 12.2 对齐分配

| 元素类型       | 对齐要求           | 分配方式            |
| -------------- | ------------------ | ------------------- |
| `f64`          | 64 字节（统一策略） | `alloc_aligned(64)` |
| `f32`          | 64 字节（统一策略） | `alloc_aligned(64)` |
| `Complex<f64>` | 64 字节（统一策略） | `alloc_aligned(64)` |
| `i32`/`i64`    | 64 字节（统一策略） | `alloc_aligned(64)` |
| `bool`         | 64 字节（统一策略） | `alloc_aligned(64)` |

> **对齐策略说明**：当前版本所有元素类型统一使用 64 字节对齐分配（覆盖 AVX-512 需求），而非按元素类型选择不同对齐值。这简化了分配器设计，代价是对 `i32` / `i64` / `bool` 类型有少量内存浪费。

### 12.3 批量初始化优化

| 场景             | 优化方式                 | 预期性能                           |
| ---------------- | ------------------------ | ---------------------------------- |
| `zeros` 大数组   | `ptr::write_bytes(0)`    | ~10 GB/s（memset 速度）            |
| `ones` 大数组    | `ptr::write(value)` 循环 | ~5 GB/s                            |
| `from_shape_vec` | 拷贝到对齐内存           | O(n)（拷贝到 64 字节对齐分配）     |
| `eye` 大矩阵     | 先零后对角               | n 次 `write` + n² 次 `write_bytes` |

> **注意**：`from_array` 当前存在双重拷贝（源数组 → Vec → 对齐分配），未来版本可优化为直接构建。

---

## 13. 平台与工程约束

| 约束       | 说明                                                                |
| ---------- | ------------------------------------------------------------------- |
| `std` only | Xenon 当前版本仅支持 `std` 环境，本文不再讨论 `no_std` 路径         |
| 单 crate   | `construct` 设计保持在现有 crate 内，不引入额外 crate               |
| SemVer     | 文档当前收敛到 `require.md` §19 的最小构造集合，不扩大公开 API 承诺 |
| 最小依赖   | 本模块不新增第三方依赖                                              |

---

## 版本历史

| 版本  | 日期       |
| ----- | ---------- |
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-07 |
| 1.0.2 | 2026-04-08 |
| 1.0.3 | 2026-04-08 |
| 1.0.4 | 2026-04-08 |
| 1.1.0 | 2026-04-08 |
| 1.1.1 | 2026-04-08 |
| 1.1.2 | 2026-04-10 |
| 1.1.3 | 2026-04-14 |
| 1.1.4 | 2026-04-15 |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
