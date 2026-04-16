# 构造操作模块设计

> 文档编号: 18 | 模块: `src/construct/` | 阶段: Phase 4
> 前置文档: `07-tensor.md`, `05-storage.md`
> 需求参考: `需求说明书 §7`, `需求说明书 §8`, `需求说明书 §19`, `需求说明书 §27`, `需求说明书 §28.2`, `需求说明书 §28.4`
> 范围声明: 范围内

---

## 1. 模块定位

### 1.1 职责边界

| 职责           | 包含                                                 | 不包含                                                         |
| -------------- | ---------------------------------------------------- | -------------------------------------------------------------- |
| 零初始化构造   | `zeros<A, D>(shape)` 全零张量                        | arange / linspace / logspace 序列生成（当前版本不提供）        |
| 常量构造       | `ones<A, D>(shape)` 全一张量                         | `full<A, D>(shape, scalar)` 与随机数构造（当前版本不提供）     |
| 单位矩阵       | `eye<A>(n)` 单位矩阵                                 | 从文件加载（当前版本不提供）                                   |
| 从 Vec 构造    | `from_shape_vec(shape, vec)` 消费输入 Vec 并构造张量 | 从迭代器/生成器构造（当前版本不提供）                          |
| 从切片构造     | `from_shape_slice(shape, slice)` 从切片拷贝数据      | 从文件/网络加载（当前版本不提供）                              |
| 从固定数组构造 | `from_array<A, N>(shape, arr)` 从固定大小数组构造    | 从文件加载（当前版本不提供）                                   |
| 从标量构造     | `from_scalar<A>(scalar)` 零维张量                    | —                                                              |
| 未来扩展构造   | —                                                    | `from_fn<A, D, F>(shape, f)` 与从迭代器/生成器构造留待后续版本 |
| 合法性验证     | 所有构造路径验证形状/大小合法性                      | 隐式类型转换（由 `convert` 模块提供）                          |

### 1.2 设计原则

| 原则         | 体现                                                                                                                                    |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| 合法性验证   | 所有构造路径须验证合法性，防止越界访问（`需求说明书 §8`）                                                                               |
| F-order 默认 | 构造时数据按 F-order 存放，默认列优先布局                                                                                               |
| 对齐分配     | `zeros`/`ones` 使用项目统一的对齐分配策略，满足拥有型连续存储的实现需求；具体对齐值不作为构造 API 的公开语义                            |
| 对齐优先     | `from_shape_vec` 可复用项目统一的 owned 存储构造路径；是否复制输入 `Vec<A>`、以及采用何种对齐值，均属于内部实现选择，不单独形成公开承诺 |
| 类型安全     | 形状和元素类型通过泛型约束在编译期检查                                                                                                  |

### 1.3 在架构中的位置

```
Dependency layers:
L0: error, private
L1: dimension, element, complex
L2: layout (depends on dimension)
L3: storage (independent of layout; owned by tensor and consumes layout results)
L4: tensor (depends on storage, dimension)
L7: construct  <- current module (depends on storage, layout, dimension, element)
```

---

## 2. 需求映射与范围约束

| 类型     | 内容                                                                                                                                                  |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| 需求映射 | `需求说明书 §7`, `需求说明书 §8`, `需求说明书 §19`, `需求说明书 §27`, `需求说明书 §28.2`, `需求说明书 §28.4`                                          |
| 范围内   | `zeros` / `ones` / `eye` / `from_shape_vec` / `from_shape_slice` / `from_array` / `from_scalar`，以及空张量 / 零维张量 / ZST 路径的合法性与错误语义。 |
| 范围外   | arange、linspace、from_fn、随机构造器与其他序列生成 API。                                                                                             |
| 非目标   | 不新增新的构造器家族，不改变 F-order / 对齐分配基线，也不引入第三方随机或数据加载依赖。                                                               |

> **范围说明：** 当前版本将 `需求说明书 §19` 中的“动态数组”解释为 `Vec<A>`。`Box<[A]>` 等其他持有容器不单独承诺。

> **固定数组范围说明：** 当前版本“固定数组构造”仅承诺线性数组入口 `[A; N]`。嵌套数组（如 `[[A; M]; N]`）自动推导 shape 不在范围内。

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
          │ storage  │ │ layout              │
          │ Owned<A> │ │ LayoutFlags         │
          │ Storage  │ │ flags_for_f_layout  │
          └──────────┘ └───────────────┘
```

### 4.2 类型级依赖

| 来源模块    | 使用的类型/trait                                                                                                  |
| ----------- | ----------------------------------------------------------------------------------------------------------------- |
| `tensor`    | `TensorBase<S, D>`, `Tensor<A, D>`, 类型别名 `Tensor0`~`Tensor6`（参见 `07-tensor.md` §5）                        |
| `storage`   | `Owned<A>`, `Storage<Elem = A>`, `from_vec_aligned()`（参见 `05-storage.md` §5）                                  |
| `layout`    | `LayoutFlags`, `Strides<D>`, 共享 checked/layout helper（元素总数与 F-order stride 计算，参见 `06-layout.md` §3） |
| `dimension` | `Dimension`, `Ix0`~`Ix6`, `IxDyn`, `IntoDimension`（参见 `02-dimension.md` §5）                                   |
| `element`   | `Element`（`zero()` / `one()` 由 `Element` 提供，参见 `03-element.md` §5.1）                                      |
| `error`     | `XenonError::InvalidShape`（用于构造时的 shape/length 基数不匹配，参见 `26-error.md` §4）                         |

### 4.3 依赖方向声明

> **依赖方向：单向向上。** `construct` 消费 `tensor`、`storage`、`layout`、`dimension`、`element` 的 trait 和类型，不被它们依赖。

### 4.4 依赖合法性与替代方案

| 项目           | 说明                                                                          |
| -------------- | ----------------------------------------------------------------------------- |
| 新增第三方依赖 | 无                                                                            |
| 合法性结论     | 合法；当前设计仅复用 Xenon 既有模块、标准库以及文档中已声明的项目内可选能力。 |
| 替代方案       | 不适用；当前范围内无需额外第三方依赖。                                        |

---

## 5. 公共 API 设计

### 5.1 zeros / ones

````rust,ignore
# use crate::dimension::{Dimension, IntoDimension};
# use crate::element::Element;
# use crate::error::XenonError;
# use crate::layout;
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
    /// ```ignore
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
        let len = layout::checked_element_count(&dim).ok_or(XenonError::InvalidShape {
            operation: "zeros",
            shape: dim.slice().to_vec(),
            expected_elements: 0,
            actual_elements: 0,
            offending_dim: None,
            reason: Some("element count overflow".into()),
        })?;
        let strides = layout::compute_f_strides(&dim)?;
        let storage = Owned::zeros(len);
        let flags = layout::flags_for_f_layout(storage.is_aligned(), false);
        Ok(TensorBase { storage, shape: dim, strides, offset: 0, flags })
    }

    /// Create a tensor filled with ones (F-order).
    ///
    /// # Examples
    /// ```ignore
    /// let t = Tensor::<f64, _>::ones([2, 3])?;
    /// assert!(t.iter().all(|&x| x == 1.0));
    /// ```
    pub fn ones<Sh>(shape: Sh) -> Result<Self, XenonError>
    where
        A: Element,  // A::one() is provided by the Element trait (see 03-element.md §5.1)
        Sh: IntoDimension<Dim = D>,
    {
        let dim = shape.into_dimension();
        let len = layout::checked_element_count(&dim).ok_or(XenonError::InvalidShape {
            operation: "ones",
            shape: dim.slice().to_vec(),
            expected_elements: 0,
            actual_elements: 0,
            offending_dim: None,
            reason: Some("element count overflow".into()),
        })?;
        let strides = layout::compute_f_strides(&dim)?;
        let storage = Owned::ones(len);
        let flags = layout::flags_for_f_layout(storage.is_aligned(), false);
        Ok(TensorBase { storage, shape: dim, strides, offset: 0, flags })
    }
}
````

> **范围决策：** `full()` 超出 `需求说明书 §19` 的当前最小构造集合。
> 如实现后续自愿提供 `from_vec()` 这类 1D 便捷包装，也仅属于非规范 convenience layer；本文的稳定承诺仍以 `需求说明书 §19` 已覆盖的标准构造语义为准，不把该便捷入口纳入范围、任务或测试承诺。

> **helper 命名说明：** `06-layout.md` 的权威 stride API 名称为 `compute_f_strides()`；本节中“元素总数 checked helper”仍为示意性命名，具体 helper 名称与归属以 `06-layout.md` 的实现约定为准。

> **`bool` 特殊值说明：** `zeros::<bool>()` 对应 `false`，`ones::<bool>()` 对应 `true`（`需求说明书 §19`）。

### 5.2 eye（单位矩阵）

````rust,ignore
# use crate::complex::Complex;
# use crate::element::Element;
# use crate::error::XenonError;
# use crate::layout;
# use crate::storage::Owned;
# use crate::tensor::{Tensor, TensorBase};
# use crate::dimension::Ix2;
# pub trait EyeElement: crate::private::Sealed + Element {}
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
    /// ```ignore
    /// let e = Tensor::<f64, Ix2>::eye(3).unwrap();
    /// assert_eq!(*e.get(&[0, 0]).unwrap(), 1.0);
    /// assert_eq!(*e.get(&[0, 1]).unwrap(), 0.0);
    /// assert_eq!(*e.get(&[1, 1]).unwrap(), 1.0);
    /// ```
    pub fn eye(n: usize) -> Result<Self, XenonError>
    where
        A: EyeElement,
    {
        let mut result = Self::zeros([n, n])?;
        for i in 0..n {
            // SAFETY: `i < n`, so `[i, i]` is always in-bounds for the validated
            // `[n, n]` shape created above. `eye()` uses unchecked indexing
            // internally and does not rely on the public `IndexMut` panic sugar.
            unsafe {
                *result.get_unchecked_mut(&[i, i]) = A::one();
            }
        }
        Ok(result)
    }
}
````

```rust,ignore
pub trait EyeElement: crate::private::Sealed + Element {}

impl EyeElement for i32 {}
impl EyeElement for i64 {}
impl EyeElement for f32 {}
impl EyeElement for f64 {}
impl EyeElement for Complex<f32> {}
impl EyeElement for Complex<f64> {}
```

> **类型约束说明：** `eye()` 不对 `bool` 开放；其适用类型严格限定为 `i32`、`i64`、`f32`、`f64`、`Complex<f32>` 与 `Complex<f64>`，以符合 `需求说明书 §19`。`EyeElement` 必须保持 sealed，避免下游扩展突破 `需求说明书 §4` 对元素类型封闭集合的要求。

> **范围说明：** 当前版本的 `eye()` 仅提供 `n×n` 方阵构造。矩形对角矩阵构造器不在范围内。

### 5.3 from_shape_vec / from_shape_slice / from_array

````rust,ignore
# use crate::dimension::{Dimension, IntoDimension};
# use crate::element::Element;
# use crate::error::XenonError;
# use crate::layout;
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
    /// Returns `XenonError::InvalidShape` if the shared checked helper reports
    /// element-count overflow or `data.len()` does not match the validated count.
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
        let expected = layout::checked_element_count(&dim).ok_or(XenonError::InvalidShape {
            operation: "from_shape_vec",
            shape: dim.slice().to_vec(),
            expected_elements: 0,
            actual_elements: 0,
            offending_dim: None,
            reason: Some("element count overflow".into()),
        })?;
        if data.len() != expected {
            return Err(XenonError::InvalidShape {
                operation: "from_shape_vec",
                shape: dim.slice().to_vec(),
                expected_elements: expected,
                actual_elements: data.len(),
                offending_dim: None,
                reason: None,
            });
        }
        let strides = layout::compute_f_strides(&dim)?;
        // from_vec_aligned stands for Xenon's internal owned construction path.
        // The public contract only requires validated F-order logical mapping and
        // an owned result; whether the implementation reuses the original Vec
        // allocation or materializes a new aligned allocation is intentionally
        // left as an internal choice.
        let storage = Owned::from_vec_aligned(data);
        let flags = layout::flags_for_f_layout(storage.is_aligned(), false);
        Ok(TensorBase { storage, shape: dim, strides, offset: 0, flags })
    }

    /// Construct a tensor from a slice (copies data).
    ///
    /// # Errors
    /// Returns `XenonError::InvalidShape` if the shared checked helper reports
    /// element-count overflow or `slice.len()` does not match the validated count.
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
        let expected = layout::checked_element_count(&dim).ok_or(XenonError::InvalidShape {
            operation: "from_shape_slice",
            shape: dim.slice().to_vec(),
            expected_elements: 0,
            actual_elements: 0,
            offending_dim: None,
            reason: Some("element count overflow".into()),
        })?;
        if slice.len() != expected {
            return Err(XenonError::InvalidShape {
                operation: "from_shape_slice",
                shape: dim.slice().to_vec(),
                expected_elements: expected,
                actual_elements: slice.len(),
                offending_dim: None,
                reason: None,
            });
        }
        // The baseline implementation materializes an owned buffer from the slice
        // and then delegates to the shared owned construction path. This may imply
        // an extra data move when the owned path chooses to re-pack into Xenon's
        // preferred allocation, but it keeps all length validation and F-order
        // materialization logic centralized in one place.
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

````

### 5.4 from_scalar

````rust,ignore
# use crate::dimension::Ix0;
# use crate::element::Element;
# use crate::layout::{self, Strides};
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
    /// assert_eq!(*t.get(&[]).unwrap(), 3.14);
    /// ```
    pub fn from_scalar(scalar: A) -> Self {
        let storage = Owned::from_vec_aligned(vec![scalar]);
        let flags = layout::flags_for_f_layout(storage.is_aligned(), false);
        TensorBase {
            storage,
            shape: Ix0,
            strides: Strides::<Ix0>::from_slice(&[]),
            offset: 0,
            flags,
        }
    }
}

````

> **范围决策：** `from_fn()` 不在 `需求说明书 §19` 当前版本强制集合中。
> 若后续版本确需提供闭包驱动构造器，应另行评估其 API 粒度、性能模型与错误语义。

### 5.5 Good / Bad 对比

```rust,ignore
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
            reason: None,
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

| 构造方法           | 分配策略                                                  | 初始化                                         |
| ------------------ | --------------------------------------------------------- | ---------------------------------------------- |
| `zeros`            | 对齐分配 + 零初始化                                       | `ptr::write_bytes(0)`                          |
| `ones`             | 对齐分配 + 批量填充                                       | `ptr::write(A::one())`                         |
| `from_shape_vec`   | 走共享 owned 构造路径；可按实现需要复用或重打包输入缓冲区 | 用户提供数据                                   |
| `from_shape_slice` | 先物化 owned 缓冲区，再委托共享 owned 构造路径            | 至少一次切片拷贝；后续是否再搬运取决于内部实现 |
| `from_scalar`      | 对齐分配（1 元素）                                        | 单元素写入                                     |
| `eye`              | 先 `zeros` 再对角线写入                                   | 两步：零初始化 + 对角线                        |

### 6.2 范围外能力记录

`full()` 与 `from_fn()` 属于后续版本候选能力，当前文档不继续展开其内部填充策略或任务拆分。

### 6.3 安全性论证

- `zeros`: 全零字节初始化仅对当前封闭元素集合合法：`i32` / `i64` 的 `0`、`f32` / `f64` 的 `+0.0`（IEEE 754）、`bool` 的 `false`、`Complex<T>` 的 `(0 + 0i)`。若未来新增元素类型，必须重新验证“全零字节可表示合法值”这一不变量。
- `ones`: 逐元素写入过程中，若 `A::one()` 的 copy 在理论上发生 panic（当前封闭类型集合中不预期出现），未初始化内存仍须由 `Owned` 析构路径基于“已初始化长度跟踪”正确回收。
- `from_shape_vec`: 先通过 layout 层共享 checked helper 验证元素总数，再验证 `data.len() == expected`；通过后进入共享 owned 构造路径。是否复用原始 `Vec` 分配、是否进行额外重打包，均属于内部实现选择，不影响公开语义
- `from_shape_slice`: 先通过 layout 层共享 checked helper 验证元素总数，长度匹配后先把切片物化为 owned 缓冲区，再委托给 `from_shape_vec`；这样把 F-order 映射、长度约束与 owned 结果语义统一收敛到单一路径
- 元素总数 / F-order stride 计算溢出：构造路径须在共享的 layout checked helper 中统一转为 `XenonError::InvalidShape`，不得把这些职责下沉到 `Dimension` trait。具体到 F-order stride，必须逐步验证 `stride[i] = product(shape[0..i])` 的乘积不会溢出 `usize`。ZST、空张量路径与 `05-storage.md` 约束保持一致
- `eye`: 内部使用已验证的 `zeros` 与 unchecked 写入；因为循环变量满足 `0 <= i < n`，所以 `[i, i]` 索引必然合法，不依赖公开 `IndexMut` panic 语法糖

---

## 7. 实现任务拆分

### Wave 1: 基础构造

- [ ] **T1**: 创建 `src/construct/` 模块骨架 + `zeros`/`ones`
  - 文件: `src/construct/mod.rs`, `src/construct/init.rs`
  - 内容: 模块声明、`zeros`/`ones` 实现
  - 测试: `test_zeros`, `test_ones`
  - 前置: `tensor` 模块完成
  - 预计: 10 min

### Wave 2: 基础构造扩展与数据源构造入口

- [ ] **T2**: 实现 `eye` 单位矩阵
  - 文件: `src/construct/eye.rs`
  - 内容: 单位矩阵构造
  - 测试: `test_eye`, `test_eye_zero_size`
  - 前置: T1
  - 预计: 5 min

- [ ] **T3**: 实现 `from_shape_vec` 和 `from_shape_slice`
  - 文件: `src/construct/from_data.rs`
  - 内容: 消费输入 Vec 进入共享 owned 构造路径；是否复用或重打包底层缓冲区由内部决定；从切片拷贝
  - 测试: `test_from_shape_vec`, `test_from_shape_vec_mismatch`, `test_from_shape_slice`
  - 前置: T1
  - 预计: 10 min

### Wave 3: 补齐剩余构造入口

- [ ] **T4**: 实现 `from_array` 和 `from_scalar`
  - 文件: `src/construct/from_data.rs`, `src/construct/scalar.rs`
  - 内容: 从固定数组构造、零维张量
  - 测试: `test_from_array`, `test_from_scalar`
  - 前置: T3
  - 预计: 5 min

### Wave 4: 集成与测试

- [ ] **T5**: 编写综合测试
  - 文件: `tests/test_construction.rs`
  - 内容: 各构造方法的集成测试、边界情况
  - 测试: 覆盖所有公共 API
  - 前置: T1-T4
  - 预计: 10 min

### 并行执行图

```
Wave 1: [T1]
           │
           ├──────────────┐
           ▼              ▼
Wave 2: [T2]           [T3]
                          │
                          ▼
Wave 3:                 [T4]
                          │
                          ▼
Wave 4:                 [T5]
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

| 测试函数                            | 测试内容                                   | 优先级 |
| ----------------------------------- | ------------------------------------------ | ------ |
| `test_zeros_shape`                  | `zeros([3, 4])` 形状正确                   | 高     |
| `test_zeros_values`                 | 所有元素为零                               | 高     |
| `test_ones_values`                  | 所有元素为一                               | 高     |
| `test_eye_3x3`                      | 3×3 单位矩阵对角线为 1                     | 高     |
| `test_eye_bool_not_supported`       | `eye::<bool>` 在类型层被拒绝               | 高     |
| `test_eye_zero`                     | `eye(0)` 空矩阵                            | 中     |
| `test_from_shape_vec_success`       | 合法 Vec 构造成功                          | 高     |
| `test_from_shape_vec_mismatch`      | Vec 长度不匹配返回错误                     | 高     |
| `test_from_shape_slice_success`     | 从切片构造                                 | 高     |
| `test_from_shape_slice_mismatch`    | 切片长度不匹配                             | 高     |
| `test_from_array_success`           | 从固定数组构造                             | 中     |
| `test_from_scalar`                  | 零维张量                                   | 高     |
| `test_construction_large_tensor`    | 大张量构造保持合法错误/成功语义            | 中     |
| `test_construction_high_rank_ixdyn` | 高 rank 构造路径与 F-order stride 校验正确 | 中     |
| `test_eye_overflow`                 | `eye()` 在元素总数溢出时返回结构化错误     | 中     |

### 8.3 边界测试场景

| 场景                          | 预期行为                                    |
| ----------------------------- | ------------------------------------------- |
| `zeros([0])`                  | 空张量，`len() == 0`                        |
| `zeros([0, 3])`               | 空张量，`len() == 0`                        |
| `eye(0)`                      | 空 0×0 矩阵                                 |
| `from_scalar(42)`             | 零维张量，`ndim() == 0`                     |
| `from_shape_vec([0], vec![])` | 空 1D 张量                                  |
| 大张量 `zeros([3162, 3162])`  | 分配成功，F-order 连续                      |
| 极大 shape 导致元素总数溢出   | 返回 `InvalidShape`，不伪造 expected/actual |
| 高 rank `IxDyn` shape         | 正确计算或拒绝 F-order stride               |
| `eye(n)` 接近溢出边界         | 在 `n * n` 溢出前返回结构化错误             |

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

| 配置              | 验证点                                                                                              |
| ----------------- | --------------------------------------------------------------------------------------------------- |
| 默认配置          | 所有构造器在默认构建下保持 F-order 与错误语义契约；具体对齐值与是否重打包输入缓冲区不作为公开断言。 |
| 其他 feature 组合 | 不适用；当前模块无额外 feature gate。                                                               |

### 8.7 类型边界 / 编译期测试

| 场景                                                             | 测试方式                     |
| ---------------------------------------------------------------- | ---------------------------- |
| `eye::<bool>` 不被支持                                           | 编译期测试。                 |
| `from_scalar` 仅产出 `Ix0`                                       | 编译期签名检查与运行时断言。 |
| arange / linspace / from_fn / random constructors 不属于当前 API | API 缺失断言。               |

---

## 9. 与其他模块的交互

### 9.1 接口约定

| 方向                    | 对方模块    | 接口/类型                            | 约定                                                                                               |
| ----------------------- | ----------- | ------------------------------------ | -------------------------------------------------------------------------------------------------- |
| `construct → tensor`    | `tensor`    | `TensorBase`                         | 构造张量实例，参见 `07-tensor.md` §5.1                                                             |
| `construct → storage`   | `storage`   | `Owned::zeros()` / owned 构造 helper | 使用项目统一的 owned 存储构造路径完成底层分配；具体对齐值、是否重打包输入缓冲区由 storage 内部负责 |
| `construct → layout`    | `layout`    | F-order 步长                         | 构造阶段计算 F-order 步长，参见 `06-layout.md` §4                                                  |
| `construct → dimension` | `dimension` | `IntoDimension`                      | 接受灵活形状参数并归一化，参见 `02-dimension.md` §5.4                                              |
| `construct → element`   | `element`   | `Element`                            | 通过 `Element::zero()` / `Element::one()` 约束构造 API，参见 `03-element.md` §5.1                  |
| `construct → error`     | `error`     | `XenonError::InvalidShape`           | shape 与长度基数不匹配时返回错误，参见 `26-error.md` §4                                            |
| `construct → index`     | `index`     | 索引访问语义                         | 构造后的张量继续复用索引路径，参见 `17-indexing.md` §4                                             |

### 9.2 数据流描述

```text
User calls zeros / from_shape_vec / eye
    │
    ├── dimension normalizes the input shape only
    ├── layout shared checked helpers validate element count and compute F-order strides / initial flags
    ├── storage allocates owned memory and writes data according to Xenon's internal policy
    └── tensor wraps the result as TensorBase<Owned<_>, D> for later use
```

---

## 10. 错误处理与语义边界

| 主题              | 内容                                                                                                                                                                                                                                                                                        |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Recoverable error | shape 与长度基数不匹配、共享 checked helper 报告元素总数溢出等情况返回 `XenonError::InvalidShape`。所有 `InvalidShape` 示例都保持 `26-error.md` 的 canonical 字段集；长度不匹配时携带实际与期望元素数，元素总数溢出时额外以 `reason = Some("element count overflow".into())` 标识溢出原因。 |
| Panic             | 公开构造 API 不定义额外 panic 语义；失败统一走 `Result`。                                                                                                                                                                                                                                   |
| 路径一致性        | 所有构造路径都必须产出 canonical F-order owned 张量，并保持一致的 shape / strides / flags 语义。                                                                                                                                                                                            |
| 容差边界          | 不适用。                                                                                                                                                                                                                                                                                    |

---

## 11. 设计决策记录

### ADR-1: 当前版本不纳入 `from_fn`

| 属性     | 值                                                             |
| -------- | -------------------------------------------------------------- |
| 决策     | `from_fn` 留待后续版本单独设计，不在当前阶段承诺               |
| 理由     | `需求说明书 §19` 当前仅要求标准构造方法与从数据源/标量构造能力 |
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

| 操作                  | 时间复杂度                              | 空间复杂度 |
| --------------------- | --------------------------------------- | ---------- |
| `zeros(n)`            | O(n)                                    | O(n)       |
| `ones(n)`             | O(n)                                    | O(n)       |
| `eye(n)`              | O(n²)                                   | O(n²)      |
| `from_shape_vec(n)`   | O(n)                                    | O(n)       |
| `from_shape_slice(n)` | O(n) 起步；若内部重打包则仍为 O(n) 级别 | O(n)       |
| `from_array(n)`       | O(n) 拷贝                               | O(n)       |
| `from_scalar()`       | O(1)                                    | O(1)       |

### 12.2 对齐分配

| 主题     | 说明                                                                                                  |
| -------- | ----------------------------------------------------------------------------------------------------- |
| 公共语义 | `需求说明书 §7` 只要求支持对齐布局，并允许存在仅用于实现目的的填充区域；构造 API 不额外暴露具体对齐值 |
| 实现选择 | `Owned` 存储可按 SIMD / FFI / allocator 约束选择合适的对齐策略                                        |
| 兼容性   | 只要逻辑元素值、访问结果与 F-order 语义不变，更换具体对齐值不构成构造模块语义变化                     |

> **对齐策略说明：** 文档只要求构造结果遵守 Xenon 的 owned/F-order/合法布局语义；实际对齐值由 storage 层统一决定，可随实现演进而调整，无需成为构造 API 的稳定承诺。

### 12.3 批量初始化优化

| 场景             | 优化方式                 | 预期性能                           |
| ---------------- | ------------------------ | ---------------------------------- |
| `zeros` 大数组   | `ptr::write_bytes(0)`    | ~10 GB/s（memset 速度）            |
| `ones` 大数组    | `ptr::write(value)` 循环 | ~5 GB/s                            |
| `from_shape_vec` | 共享 owned 构造路径      | O(n)                               |
| `eye` 大矩阵     | 先零后对角               | n 次 `write` + n² 次 `write_bytes` |

> **注意**：`from_array` 当前存在双重拷贝（源数组 → Vec → 对齐分配），未来版本可优化为直接构建。

---

## 13. 平台与工程约束

| 约束       | 说明                                                                |
| ---------- | ------------------------------------------------------------------- |
| `std` only | Xenon 当前版本仅支持 `std` 环境，本文不再讨论 `no_std` 路径         |
| 单 crate   | `construct` 设计保持在现有 crate 内，不引入额外 crate               |
| SemVer     | 文档当前收敛到 `需求说明书 §19` 的最小构造集合，不扩大公开 API 承诺 |
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
