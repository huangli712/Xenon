# 运算符重载模块设计

> 文档编号: 19
> 模块目录: src/overload/
> 任务阶段: Phase 4
> 前置文档: 07-tensor.md, 11-math.md, 15-broadcast.md

---

## 1. 模块定位

### 1.1 职责边界

| 职责               | 包含                                                                          | 
| ------------------ | ----------------------------------------------------------------------------- | 
| 四则运算运算符语法 | `+`/`-`/`*`/`/` 运算符重载                                                    | 
| 张量×张量运算      | 同形状运算、广播运算                                                          | 
| 张量×标量运算      | `tensor op scalar`、`Scalar(scalar) op tensor` 与常用原生左标量 `scalar op tensor` |
| 广播支持           | 运算符语法内建支持广播                                                        | 
| 新张量产生         | 所有组合产生新的独立张量                                                      | 
| 借用形式           | `&Tensor op &Tensor`/`&Tensor op Tensor` 等组合                               |

| 职责               | 不包含                                           |
| ------------------ | ------------------------------------------------ |
| 四则运算运算符语法 | 原地运算符 `+=`/`-=`/`*=`/`/=`（当前版本不提供） |
| 张量×张量运算      | 矩阵乘法（将来由 `matrix` 提供）                 |
| 张量×标量运算      | 完全泛型的 `T op Tensor<T>` blanket impl         |
| 广播支持           | 比较运算符（在 `math` 提供）                     |
| 新张量产生         | 原地修改运算                                     |

### 1.2 设计原则

| 原则       | 体现                                            |
| ---------- | ----------------------------------------------- |
| 委托模式   | 运算符重载委托给逐元素运算，运算符仅为语法糖    |
| 深拷贝结果 | 所有组合均产生新的独立张量，不共享内存          |
| 广播透明   | 运算符语法内建支持广播，用户无需手动处理        |
| 借用优先   | 鼓励使用 `&a + &b` 形式，避免不必要的所有权转移 |

---

## 2. 需求映射与范围约束

| 类型     | 内容 |
| -------- | ---- |
| 需求映射 | 需求说明书 §12、§20、§27、§28 |
| 范围内   | `+` / `-` / `*` / `/` 的张量×张量、张量×标量及常用左标量重载，含广播与 borrowed 组合。 |
| 范围外   | 位运算、比较运算符、赋值运算符、原地运算与其他超出四则运算范围的 operator API。 |
| 非目标   | 不把运算符层扩展为新的计算后端，不新增第三方依赖，也不在本文设计原地广播语义。 |

---

## 3. 文件位置

```
src/overload/
├── arithmetic.rs       # arithmetic operator overloading
└── mod.rs              # module entry
```

---

## 4. 依赖关系与实现约束

### 4.1 依赖图（ASCII）

```
src/overload/
|
├── mod.rs
│   └── re-exports from arithmetic
|
└── arithmetic.rs
    ├── crate::math        # add() / sub() / mul() / div() / scalar helpers
    ├── crate::tensor      # TensorBase<S, D>, Tensor<A, D>, TensorView, .view()
    ├── crate::element     # Numeric trait
    ├── crate::dimension   # Dimension, Ix0~Ix6, IxDyn, BroadcastDim<E>
    └── crate::error       # XenonError (used in impl Output = Result<..., XenonError>)
```

### 4.2 类型级依赖

| 来源模块    | 使用的类型/trait                                                                        |
| ----------- | --------------------------------------------------------------------------------------- |
| `math`      | `add()` / `sub()` / `mul()` / `div()` / `*_scalar()` 等方法型逐元素运算（参见 `11-math.md` §5） |
| `tensor`    | `TensorBase<S, D>`, `Tensor<A, D>`, `TensorView`, `.view()`（参见 `07-tensor.md` §5）   |
| `element`   | `Numeric` trait 约束（排除 `bool` 与 `usize`）（参见 `03-element.md` §5.2）             |
| `error`     | `XenonError`（运算符 impl 的 `Output = Result<..., XenonError>` 关联类型中使用，参见 `26-error.md §5`） |
| `dimension` | `Dimension`, `Ix0`~`Ix6`, `IxDyn`, `BroadcastDim<E>`（参见 `02-dimension.md §5.10`）    |

### 4.3 依赖合法性

| 项目           | 说明                                                                          |
| -------------- | ----------------------------------------------------------------------------- |
| 新增第三方依赖 | 无                                                                            |
| 合法性结论     | 合法；当前设计仅复用 Xenon 既有模块、标准库以及文档中已声明的项目内可选能力。 |
| 替代方案       | 不适用；当前范围内无需额外第三方依赖。                                        |

### 4.4 依赖方向声明

依赖方向：单向向上。`overload` 仅消费 `math`、`tensor`、`element`、`dimension`、`error` 的 trait 和类型，不被它们依赖。

---

## 5. 公共 API 设计

### 5.1 运算符 trait 实现矩阵

**当前稳定承诺的 `impl` 组合表（以 `Add` 为例，`Sub`/`Mul`/`Div` 同理）：**

- 本表列举本版本 SemVer 稳定承诺的组合，不是所有可能 owned/借用组合的笛卡尔积。
- 表中已覆盖：所有 owned-or-borrowed × owned-or-borrowed 的张量×张量 16 种组合中实际承诺的子集。
- 其它未列出的按值/借用排列（如 `TensorView<A, D> + TensorView<A, E>` 按值×按值），调用方应通过显式 `&view + &other` 或 `view.add(other)?` 表达。
- `Tensor<A, D> + Tensor<A, E>`（owned×owned）虽列在表中，但生产代码推荐使用 `&a + &b` 避免不必要的所有权消费。
- 详细按值 vs 按引用决策见 §6.2。

| Lhs                  | Rhs                  | Output         | 广播     | impl 签名                                                                           |
| -------------------- | -------------------- | -------------- | -------- | ----------------------------------------------------------------------------------- |
| `Tensor<A, D>`       | `Tensor<A, E>`       | `Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>` | ✓ | `impl<...> Add<TensorBase<Owned<A>,E>> for TensorBase<Owned<A>,D>`                 |
| `&Tensor<A, D>`      | `&Tensor<A, E>`      | `Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>` | ✓ | `impl<...> Add<&TensorBase<Owned<A>,E>> for &TensorBase<Owned<A>,D>`               |
| `Tensor<A, D>`       | `&Tensor<A, E>`      | `Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>` | ✓ | `impl<...> Add<&TensorBase<Owned<A>,E>> for TensorBase<Owned<A>,D>`                |
| `&Tensor<A, D>`      | `Tensor<A, E>`       | `Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>` | ✓ | `impl<...> Add<TensorBase<Owned<A>,E>> for &TensorBase<Owned<A>,D>`                |
| `&TensorView<A, D>`  | `&TensorView<A, E>`  | `Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>` | ✓ | `impl<...> Add<&TensorBase<ViewRepr<'b, A>,E>> for &TensorBase<ViewRepr<'a, A>,D>` |
| `&TensorView<A, D>`  | `&Tensor<A, E>`      | `Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>` | ✓ | `impl<...> Add<&TensorBase<Owned<A>,E>> for &TensorBase<ViewRepr<'a, A>,D>`        |
| `&Tensor<A, D>`      | `&TensorView<A, E>`  | `Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>` | ✓ | `impl<...> Add<&TensorBase<ViewRepr<'b, A>,E>> for &TensorBase<Owned<A>,D>`        |
| | | | | |
| `Tensor<A, D>`       | `A`                  | `Tensor<A, D>` | 标量广播 | `impl<...> Add<A> for TensorBase<Owned<A>,D>`                                       |
| `&Tensor<A, D>`      | `A`                  | `Tensor<A, D>` | 标量广播 | `impl<...> Add<A> for &TensorBase<Owned<A>,D>`                                      |
| `Scalar<A>`          | `Tensor<A, D>`       | `Tensor<A, D>` | 标量广播 | `impl<...> Add<TensorBase<Owned<A>,D>> for Scalar<A>`                               |
| `Scalar<A>`          | `&Tensor<A, D>`      | `Tensor<A, D>` | 标量广播 | `impl<...> Add<&TensorBase<Owned<A>,D>> for Scalar<A>`                              |
| `T`（逐类型：`f32`/`f64`/`i32`/`i64`/`Complex<f32>`/`Complex<f64>`） | `Tensor<T, D>` | `Tensor<T, D>` | 标量广播 | 逐具体类型生成（不是泛型 `T`），例如 `impl<D: Dimension> Add<TensorBase<Owned<f32>, D>> for f32 { ... }`、`impl<D: Dimension> Add<TensorBase<Owned<i32>, D>> for i32 { ... }`，每个受支持的算术标量一份。Rust 孤儿规则允许这种 impl：`Self = f32` 是确定的外部类型，但 trait 类型参数 `Add<TensorBase<...>>` 含本地类型 `TensorBase`，因此本 crate 拥有 impl 权（详见 §5.4 与 §6） |
| `T`（逐类型：`f32`/`f64`/`i32`/`i64`/`Complex<..>`） | `&Tensor<T, D>` | `Tensor<T, D>` | 标量广播 | `impl<D> Add<&TensorBase<Owned<T>,D>> for T`（逐类型生成，`T == A`） |
| `TensorView<A, D>`   | `A`                  | `Tensor<A, D>` | 标量广播 | `impl<...> Add<A> for TensorBase<ViewRepr<'a, A>,D>`                                   |
| `&TensorView<A, D>`  | `A`                  | `Tensor<A, D>` | 标量广播 | `impl<...> Add<A> for &TensorBase<ViewRepr<'a, A>,D>`                                  |
| `Scalar<A>`          | `TensorView<A, D>`   | `Tensor<A, D>` | 标量广播 | `impl<...> Add<TensorBase<ViewRepr<'a, A>,D>> for Scalar<A>`                           |
| `T`（逐类型：`f32`/`f64`/`i32`/`i64`/`Complex<f32>`/`Complex<f64>`） | `TensorView<T, D>` | `Tensor<T, D>` | 标量广播 | 逐具体类型生成（不是泛型 `T`），例如 `impl<'a, D: Dimension> Add<TensorBase<ViewRepr<'a, f32>, D>> for f32 { ... }`，每个受支持的算术标量一份。Rust 孤儿规则允许此模式：`Self = f32` 是非泛型外部类型，trait 类型参数含本地 `TensorBase`，禁止的写法是 `impl<T> Add<TensorBase<..., T, D>> for T`（泛型 `T` 在 trait 第一个本地类型参数之前）。 |

- 上表仅列出当前稳定承诺。张量×张量/视图路径（前 7 行）与标量路径（后 10 行）通过空行分隔；原生左标量逐类型 impl 只覆盖 `T op Tensor<T, D>` / `T op TensorView<T, D>`，不提供 `T op Tensor<A, D>` 的混合元素类型提升。
- `TensorView` 相关组合已纳入当前稳定范围，与 `broadcast_to()` / `transpose()` / `slice()` 返回视图的既有设计保持一致。`TensorView` 参与张量×张量/视图路径的运算符重载，同时标量运算符重载（`TensorView + scalar`、`&TensorView + scalar`、`Scalar(s) + TensorView`、原生左标量）也已覆盖 `TensorView`（只读视图）。`TensorViewMut` 不直接参与标量运算符重载——使用方需先调用 `.view()` 转为 `TensorView` 后再使用运算符，或显式调用 `.add_scalar()` 等方法。
- `BroadcastDim` 定义于 `02-dimension.md §5.10`，被 `01-architecture.md §11` 记为“公开 sealed trait”（允许命名但禁止外部实现）。由于它出现在 `broadcast` / `overload` 的公开签名与 trait bound 中，稳定承诺要求用户可在签名中命名该 trait，但不要求用户自行实现它。

**同形状 vs 异形状路径使用建议**：

虽然张量×张量运算符的 `Output` 类型一律为 `Result<Tensor<A, F>, XenonError>`，但实际调用语义按 LHS / RHS 形状关系可分为两个使用场景，推荐选择不同的写法以提高代码可读性：

| 场景                  | 形状关系                                       | 失败可能性                                     | 推荐写法                                | 理由                                                                                  |
| --------------------- | ---------------------------------------------- | ---------------------------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------- |
| 同形状路径            | `a.shape() == b.shape()`（且 rank 类型一致）   | 无（广播必然成功，结果永远 `Ok`）          | `a + b`（直接用运算符）                 | 调用点最简洁；`.unwrap()` 安全，可在静态已知同形状的上下文中通过类型层进一步收敛       |
| 异形状路径 / 广播路径 | `a.shape() != b.shape()` 或维度类型不同        | 有（广播可能失败 → `Err::BroadcastError`） | `a.add(b)?`（显式方法 + `?` 传播）      | 显式方法名 `add` 比运算符 `+` 更突出"可能失败"语义，与 Rust `Result`-传播习惯一致     |

具体含义：

- **同形状路径**：当编译期或运行期已知 `a` 与 `b` 同形状（含 rank 类型一致），`a + b` 永远返回 `Ok(...)`，对返回值 `.unwrap()` 是安全的；运算符语法的简洁优势在此场景最大化。
- **异形状路径**：当 `a` 与 `b` 形状不同（含 rank 不同或某些轴维度不同），运算符 `a + b` 仍合法，但是否成功取决于 `BroadcastDim` 的运行期校验；此时显式方法 `a.add(&b)?`（由 `11-math.md §5.3` 提供）更能突出失败可恢复的语义。
- **API 等价性**：运算符 `a + b` 与方法 `a.add(&b)` 在语义、错误模型、性能上完全等价（前者委托给后者，参见 §6.1）；区别仅在调用点风格。本节是风格建议，不是语义约束——任意场景下两种写法都合法。
- **不引入新方法**：`11-math.md §5` 现有 `add()` / `sub()` / `mul()` / `div()` 方法已经返回 `Result<Tensor<A, F>, XenonError>`，自身扮演 `try_add` 角色；本模块**不**新增独立 `try_add` / `try_sub` / `try_mul` / `try_div` 方法。运算符与方法的二元划分已经足够，引入第三套命名只会增加 API 表面噪音。

### 5.2 张量×张量运算符

**方法解析说明**：以下伪代码体内出现的 `self.add(rhs)` / `self.sub(rhs)` / `self.mul(rhs)` / `self.div(rhs)` 调用必须解析到 `11-math.md §5` 定义的 inherent method，不能递归到本 trait `core::ops::Add` 实现，否则会无限递归并 stack overflow。Rust 方法解析优先级是"inherent method 优先于 trait method"，因此只要 `TensorBase` 上存在同名 inherent `add` / `sub` / `mul` / `div`，调用即正确委托——这正是 §5.1 选择"trait 委托 inherent"模式的原因。委托关系由 `tests/compile/operator_inherent_method_resolution.rs` 守护（详见 §5.3 末尾）。如实现层为了消除任何歧义，可改为完全限定调用形式 `<TensorBase<Owned<A>, D>>::add(self, &rhs)` 或在 inherent method 改名为 `add_impl` / `_add` 等内部名称——这两种形态本质等价；当前文档保留 `self.add(rhs)` 是为了与公开 API 命名一致。

```rust,ignore
// Tensor + Tensor (owned + owned)
impl<A, D, E> Add<TensorBase<Owned<A>, E>> for TensorBase<Owned<A>, D>
where
    A: Numeric,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>;

    fn add(self, rhs: TensorBase<Owned<A>, E>) -> Self::Output {
        self.add(&rhs)
    }
}

// &Tensor + &Tensor (most common form)
impl<'a, 'b, A, D, E> Add<&'b TensorBase<Owned<A>, E>> for &'a TensorBase<Owned<A>, D>
where
    A: Numeric,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>;

    fn add(self, rhs: &'b TensorBase<Owned<A>, E>) -> Self::Output {
        self.add(rhs)
    }
}

// &TensorView + &TensorView (stable: view operations return TensorView)
impl<'a, 'b, A, D, E> Add<&'b TensorBase<ViewRepr<'b, A>, E>>
    for &'a TensorBase<ViewRepr<'a, A>, D>
where
    A: Numeric,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>;

    fn add(self, rhs: &'b TensorBase<ViewRepr<'b, A>, E>) -> Self::Output {
        self.add(rhs)
    }
}

// &TensorView + &Tensor (stable: mixed view/owned path)
impl<'a, 'b, A, D, E> Add<&'b TensorBase<Owned<A>, E>> for &'a TensorBase<ViewRepr<'a, A>, D>
where
    A: Numeric,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>;

    fn add(self, rhs: &'b TensorBase<Owned<A>, E>) -> Self::Output {
        self.add(rhs)
    }
}

// &Tensor + &TensorView (stable: mixed owned/view path)
impl<'a, 'b, A, D, E> Add<&'b TensorBase<ViewRepr<'b, A>, E>> for &'a TensorBase<Owned<A>, D>
where
    A: Numeric,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>;

    fn add(self, rhs: &'b TensorBase<ViewRepr<'b, A>, E>) -> Self::Output {
        self.add(rhs)
    }
}
```

- 与 `15-broadcast.md` 保持一致；对称张量×张量运算须同时满足 `D: BroadcastDim<E>` 与 `E: BroadcastDim<D>`，以保证输出维度类型可双向收敛到同一关联类型。
- 张量×张量运算符直接委托给 `11-math.md §5` 的方法型逐元素 API（`TensorBase::add()` / `TensorBase::sub()` / `TensorBase::mul()` / `TensorBase::div()`）。运算符 impl 中的 `self.add(rhs)` 不会产生递归：Rust 的方法解析规则优先匹配固有方法（`math` 模块提供的 `pub fn add(&self, ...)`），而非 `Add` trait 自身的 `fn add(self, ...)`。

- 广播失败走 `Err(XenonError::BroadcastError)` ；整数除零、整数溢出与结果不可表示仍保持 panic。
- 当前稳定承诺覆盖 `Owned×Owned`、`TensorView×TensorView`、`TensorView×Tensor`、`Tensor×TensorView` 以及标量路径。实现优先级：`Owned×Owned` > `Owned/View` 混合路径。
- 当前稳定 API 直接承诺只读 `TensorView` 参与运算符重载，这样 `broadcast_to()`、`transpose()`、`slice()` 等返回视图的操作结果可以继续参与四则运算。
- `TensorViewMut` 不直接参与运算符重载。若要使用运算符，必须先调用 `.view()` 获取只读 `TensorView`，再对该只读视图应用运算符。若调用方需要从视图结果恢复 owned / shared 链（如 `tensor.transpose().to_owned().into_shared()`），参见 `21-type.md §5.5` 以及 storage shared ownership 相关设计。
- 无论输入组合如何，成功结果都分配新的 owned 张量，不提供原地写回或视图就地更新。

**TensorViewMut 与运算符**：

`TensorViewMut<'a, A, D>` 不直接参与运算符重载。原因：可变独占借用语义与运算符的“输入按值或共享引用消费”模式冲突。使用方需要先调用 `.view()` 转为 `TensorView` 再使用运算符：

```rust,ignore
let mut tensor: Tensor<f64, _> = ...;
let mut_view = tensor.view_mut();
let immut_view = mut_view.view();   // reborrow as TensorView
let result = immut_view + 5.0;      // operators work on TensorView
```

或显式调用 `.add_scalar()` 等方法，这些方法在 `TensorViewMut` 上仍可用。

### 5.3 张量×标量运算符

```rust,ignore
/// Newtype wrapper for scalar values, enabling a generic left-scalar path.
///
/// Rust orphan rules forbid blanket impls such as
/// `impl<T> Add<TensorBase<...>> for T`, because the foreign `Self = T`
/// appears before the first local type. However, concrete primitive left-hand
/// sides like `impl Add<Tensor<f32, D>> for f32` remain legal and should be
/// provided for Xenon's supported scalar set when stable syntax requires it.
///
/// Exported via `xenon::overload::Scalar` only — intentionally excluded from
/// the prelude and top-level re-exports. Most users should prefer the direct
/// native left-scalar forms (`5.0 + tensor`) or the right-scalar path
/// (`tensor + scalar`); `Scalar(x)` is only needed in generic code where the
/// concrete scalar type is a type parameter.
pub struct Scalar<A>(pub A);

// Tensor + scalar
impl<A, D> Add<A> for TensorBase<Owned<A>, D>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn add(self, rhs: A) -> Self::Output {
        self.add_scalar(rhs)
    }
}

// &Tensor + scalar
impl<'a, A, D> Add<A> for &'a TensorBase<Owned<A>, D>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn add(self, rhs: A) -> Self::Output {
        self.add_scalar(rhs)
    }
}

// Scalar<A> + Tensor (scalar on the left)
impl<A, D> Add<TensorBase<Owned<A>, D>> for Scalar<A>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn add(self, rhs: TensorBase<Owned<A>, D>) -> Self::Output {
        rhs.add_scalar(self.0)
    }
}

// Scalar<A> + &Tensor
impl<'a, A, D> Add<&'a TensorBase<Owned<A>, D>> for Scalar<A>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn add(self, rhs: &'a TensorBase<Owned<A>, D>) -> Self::Output {
        rhs.add_scalar(self.0)
    }
}

// TensorView + scalar
impl<'a, A, D> Add<A> for TensorBase<ViewRepr<'a, A>, D>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn add(self, rhs: A) -> Self::Output {
        self.add_scalar(rhs)
    }
}

// &TensorView + scalar
impl<'a, 'b, A, D> Add<A> for &'b TensorBase<ViewRepr<'a, A>, D>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn add(self, rhs: A) -> Self::Output {
        self.add_scalar(rhs)
    }
}

// Scalar<A> + TensorView (commutative -- Add/Mul)
impl<'a, A, D> Add<TensorBase<ViewRepr<'a, A>, D>> for Scalar<A>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn add(self, rhs: TensorBase<ViewRepr<'a, A>, D>) -> Self::Output {
        rhs.add_scalar(self.0)
    }
}
```

- `Scalar<A>` 包装器是实现“泛型左标量 + 张量”时的工程性折中，而不是原生`scalar + tensor` 整体不可行的证明。对 Xenon 支持的具体标量类型（`i32`、`i64`、`f32`、`f64`、`Complex<f32>`、`Complex<f64>`），可以逐类型生成 `impl Add<TensorBase<...>> for T`；真正不可行的是 `impl<T> Add<TensorBase<...>> for T` 这种 blanket impl。因此“常用原生标量”在本文中明确指上述 6 个受支持算术元素类型，而不包括 `bool`、`usize` 或其他范围外类型。
- `Scalar<A>` 保持 `pub` 可见以满足泛型左标量路径的编译需求，但不通过 prelude 或 crate 根导出，仅通过 `xenon::overload::Scalar` 可访问。其定位是孤儿规则下的工程设施，非核心抽象——绝大多数场景下用户应使用 `tensor + scalar`、`scalar + tensor`（原生类型）或方法调用 `.add_scalar()` 等更直接的路径，仅在编写泛型函数 `fn foo<A: Numeric>(a: A, t: Tensor<A, D>)` 且需要 `a + t` 语法时才需引入 `Scalar(a)`。
- 标量运算符的 LHS/RHS 组合通过宏生成，覆盖矩阵参见 §5.4。
- 标量路径无形状不兼容风险，不返回 `Result`；运算符返回 `Tensor` 直接。整数溢出仍遵循 panic 语义。
- 当前版本不稳定承诺 `&A` 形式的标量运算符重载。公开契约仅保证值形式 `tensor + scalar`、`Scalar(scalar) + tensor`，以及常用原生左标量（如 `5.0 + tensor`）。
- 标量运算符重载已覆盖 owned `Tensor` 和 `TensorView`（只读视图）。`TensorViewMut` 不直接参与标量运算符重载——使用方需先调用 `.view()` 转为 `TensorView` 后再使用运算符，或显式调用 `.add_scalar()` 等方法，参见 `11-math.md §5.9`。
- 标量路径的委托分为两类，对 owned `Tensor` 和 `TensorView` 均适用：
  - **右标量与交换性左标量**：`tensor op scalar`、`scalar + tensor`、`scalar * tensor` 直接调用 `11-math.md §5.9` 的公开标量方法（`.add_scalar()` / `.sub_scalar()` / `.mul_scalar()` / `.div_scalar()`），不重复实现。
  - **非交换性左标量**：`scalar - tensor`、`scalar / tensor` 委托到 `11-math.md §5.9` 的 `pub(crate)` 内部入口 `tensor.sub_from_scalar(scalar)` / `tensor.div_from_scalar(scalar)`。本模块不得自行实现逐元素遍历——所有逐元素执行骨架（dispatch 路径选择、SIMD/Parallel admission、checked arithmetic、panic 语义）由 `11-math.md` 单点维护，避免左标量路径游离于 math 执行优化之外。本模块内部 helper（如 `sub_scalar_left_impl` / `div_scalar_left_impl`）仅作为 trait impl 的薄桥接，函数体应仅一行委托：`fn sub_scalar_left_impl(scalar, tensor) -> Tensor { tensor.sub_from_scalar(scalar) }`，不得包含逐元素循环或 dispatch 调用。

### 5.4 Sub / Mul / Div

`Sub`、`Mul`、`Div` 的实现模式与 `Add` 完全相同，需覆盖与 `Add` 对称的张量/引用/视图/标量/`Scalar<A>` 组合；其中当前稳定范围内的张量×张量（含 `TensorView` 参与的只读组合）路径返回 `Result<Tensor<A, F>, XenonError>`，标量路径返回 `Tensor<A, D>`：

```rust,ignore
// Sub: |a, b| a - b
// Mul: |a, b| a * b
// Div: |a, b| a / b
```

**除法语义补充**： 

对整数类型，`Div` 路径中的除以零和结果不可表示（如最小负值除以 `-1`）均遵循 `需求说明书 §12` 与 `需求说明书 §27` 的统一 panic 语义；运算符重载仅把广播不兼容报告为 `Result::Err`，不额外吞掉或包装这类不可恢复错误。

**TensorView 标量路径**：

与 owned `Tensor` 对称，覆盖 3 种 LHS/RHS 组合 × 4 种运算符。

```rust,ignore
// ── Sub ─────────────────────────────────────────────

// TensorView - scalar
impl<'a, A, D> Sub<A> for TensorBase<ViewRepr<'a, A>, D>
where A: Numeric, D: Dimension
{ type Output = Tensor<A, D>; fn sub(self, rhs: A) -> Self::Output { self.sub_scalar(rhs) } }

// &TensorView - scalar
impl<'a, 'b, A, D> Sub<A> for &'b TensorBase<ViewRepr<'a, A>, D>
where A: Numeric, D: Dimension
{ type Output = Tensor<A, D>; fn sub(self, rhs: A) -> Self::Output { self.sub_scalar(rhs) } }

// Scalar<A> - TensorView (non-commutative → sub_scalar_left_impl)
//
// `sub_scalar_left_impl` is a 1-line bridge defined as:
//     pub(crate) fn sub_scalar_left_impl<S, D, A>(scalar: A, t: &TensorBase<S, D>) -> Tensor<A, D>
//     where S: Storage<Elem = A>, D: Dimension, A: Numeric
//     { t.sub_from_scalar(scalar) }
// All element-wise execution lives in `11-math.md §5.9`; this helper exists
// only to satisfy trait-impl call-site ergonomics, never as a parallel impl path.
impl<'a, A, D> Sub<TensorBase<ViewRepr<'a, A>, D>> for Scalar<A>
where A: Numeric, D: Dimension
{
    type Output = Tensor<A, D>;
    fn sub(self, rhs: TensorBase<ViewRepr<'a, A>, D>) -> Self::Output {
        sub_scalar_left_impl(self.0, &rhs)
    }
}

// ── Mul ─────────────────────────────────────────────

// TensorView * scalar
impl<'a, A, D> Mul<A> for TensorBase<ViewRepr<'a, A>, D>
where A: Numeric, D: Dimension
{ type Output = Tensor<A, D>; fn mul(self, rhs: A) -> Self::Output { self.mul_scalar(rhs) } }

// &TensorView * scalar
impl<'a, 'b, A, D> Mul<A> for &'b TensorBase<ViewRepr<'a, A>, D>
where A: Numeric, D: Dimension
{ type Output = Tensor<A, D>; fn mul(self, rhs: A) -> Self::Output { self.mul_scalar(rhs) } }

// Scalar<A> * TensorView (commutative)
impl<'a, A, D> Mul<TensorBase<ViewRepr<'a, A>, D>> for Scalar<A>
where A: Numeric, D: Dimension
{
    type Output = Tensor<A, D>;
    fn mul(self, rhs: TensorBase<ViewRepr<'a, A>, D>) -> Self::Output {
        rhs.mul_scalar(self.0)
    }
}

// ── Div ─────────────────────────────────────────────

// TensorView / scalar
impl<'a, A, D> Div<A> for TensorBase<ViewRepr<'a, A>, D>
where A: Numeric, D: Dimension
{ type Output = Tensor<A, D>; fn div(self, rhs: A) -> Self::Output { self.div_scalar(rhs) } }

// &TensorView / scalar
impl<'a, 'b, A, D> Div<A> for &'b TensorBase<ViewRepr<'a, A>, D>
where A: Numeric, D: Dimension
{ type Output = Tensor<A, D>; fn div(self, rhs: A) -> Self::Output { self.div_scalar(rhs) } }

// Scalar<A> / TensorView (non-commutative → div_scalar_left_impl)
impl<'a, A, D> Div<TensorBase<ViewRepr<'a, A>, D>> for Scalar<A>
where A: Numeric, D: Dimension
{
    type Output = Tensor<A, D>;
    fn div(self, rhs: TensorBase<ViewRepr<'a, A>, D>) -> Self::Output {
        div_scalar_left_impl(self.0, &rhs)
    }
}
```

- 上述 12 个 impl 与 owned `Tensor` 的标量路径完全对称：右标量路径和交换性左标量路径委托给 `11-math.md §5.9` 的公开标量方法；非交换左标量路径（`Scalar<A> - TensorView` / `Scalar<A> / TensorView`）通过本模块内部 helper（`sub_scalar_left_impl`/`div_scalar_left_impl`）转委托——helper 自身仅一行薄桥接 `t.sub_from_scalar(scalar)` / `t.div_from_scalar(scalar)`，逐元素执行骨架由 `11-math.md §5.9` 单点维护。
- 同 owned `Tensor` 路径，本处 `TensorView` 标量运算符组合也通过宏生成实际代码。
- 原生左标量（如 `5.0 + tensor_view`、`3.0 / tensor_view`）的 `TensorView` 版本同样受支持。

**标量重载委托路径**：

覆盖全部 `Numeric` 类型：`i32`、`i64`、`f32`、`f64`、`Complex<f32>`、`Complex<f64>`；表中 `tensor` 指 owned `Tensor` 与 `TensorView` 两者。"交换性"指运算满足交换律，左标量可复用右标量的 math 方法（如 `scalar + tensor` → `tensor.add_scalar(scalar)`）。`TensorView` 标量路径与 owned `Tensor` 标量路径在生成宏中合并，共享同一委托规则。

| 运算符 | `tensor op scalar` | `scalar op tensor`（交换性） | `scalar op tensor`（非交换性） |
| ------ | ------------------ | ---------------------------- | ------------------------------ |
| `+`    | → `.add_scalar()`  | 原生左标量 / `Scalar<A>` → `.add_scalar()` | — |
| `-`    | → `.sub_scalar()`  | —                            | 原生左标量 / `Scalar<A>` → `sub_scalar_left_impl` |
| `*`    | → `.mul_scalar()`  | 原生左标量 / `Scalar<A>` → `.mul_scalar()` | — |
| `/`    | → `.div_scalar()`  | —                            | 原生左标量 / `Scalar<A>` → `div_scalar_left_impl` |

### 5.5 Good / Bad 对比

```rust,ignore
// Good - same shape (no broadcast failure) — operator preferred for clarity
fn compute_same_shape(a: &Tensor<f64, Ix2>, b: &Tensor<f64, Ix2>) -> Result<Tensor<f64, Ix2>, XenonError> {
    a + b  // &Tensor + &Tensor -> Result<Tensor, XenonError> ; operator is safe & idiomatic here
}

// Good - different shapes (broadcast may fail) — explicit method emphasises possible Err
fn compute_broadcast(a: &Tensor<f64, Ix2>, b: &Tensor<f64, Ix1>) -> Result<Tensor<f64, Ix2>, XenonError> {
    a.add(b)  // Explicit method makes the failure path obvious
}

// Also fine - operator works for broadcast too, but `.add()` is recommended in broadcast contexts
fn compute_broadcast_with_op(a: &Tensor<f64, Ix2>, b: &Tensor<f64, Ix1>) -> Result<Tensor<f64, Ix2>, XenonError> {
    a + b  // Equivalent to a.add(b) ; legal but slightly less explicit
}

// Bad - mixing owned and borrowed (unnecessarily consumes a)
fn compute_bad(a: Tensor<f64, Ix2>, b: &Tensor<f64, Ix2>) -> Result<Tensor<f64, Ix2>, XenonError> {
    a + b  // a is consumed, cannot be used afterwards
}
```

---

## 6. 内部实现设计

### 6.1 委托模式

运算符重载的核心设计模式是委托：

```
Operator syntax (arithmetic.rs)
     |
     | delegates to TensorBase::add() / sub() / mul() / div() (math methods)
     v
Element-wise math (math methods)
     |
     | internally broadcasts via broadcast_shape() + broadcast_to() (see 11-math.md §6.2)
     v
Broadcast module (broadcast.rs) -- iterate broadcast views, write result
```

运算符 `a + b` 展开为：

1. 运算符 impl 中调用 `self.add(rhs)`，委托给 `TensorBase::add()`（`11-math.md §5.3`）
2. `TensorBase::add()` 内部通过 `broadcast_shape()` 计算共同 shape，再分别 `broadcast_to()` 构造广播视图（`11-math.md §6.2`）；等价于调用 `broadcast_with(&a.view(), &b.view())` 完成广播
3. 逐元素遍历广播后视图并写入新结果张量

### 6.2 深拷贝保证

所有运算符在成功路径上产生的新张量是独立的：

- 方法型逐元素运算分配新的 `Owned` 存储并逐元素写入
- 新张量与输入张量不共享内存
- `Tensor<A, D>` 类型保证所有权独占

### 6.3 标量路径优化

标量×张量运算使用专门的标量方法型 API，而非广播视图：

```
tensor + scalar:
    tensor.add_scalar(scalar)

    Advantages:
    1. No broadcast view allocation
    2. Direct iteration inside scalar methods, easier for inlining/vectorization
    3. Cache-friendly contiguous access
```

---

## 7. 实现任务拆分

### Wave 1: 基础运算符

- [ ] **T1**: 创建 `src/overload/arithmetic.rs` 骨架
  - 文件: `src/overload/arithmetic.rs`
  - 内容: 模块声明、导入
  - 测试: 编译通过
  - 前置: `math` 完成、`broadcast` 完成
  - 预计: 5 min

### Wave 2: Owned 张量运算符

- [ ] **T2**: 实现 `Add` trait（张量×张量，所有权形式）
  - 文件: `src/overload/arithmetic.rs`
  - 内容: `Tensor + Tensor` impl
  - 测试: `test_add_same_shape`, `test_add_broadcast`
  - 前置: T1
  - 预计: 10 min

### Wave 3: 借用与标量形式

- [ ] **T3**: 实现 `Add` trait（&张量×&张量、混合形式）
  - 文件: `src/overload/arithmetic.rs`
  - 内容: 4 种借用组合
  - 测试: `test_add_ref_ref`, `test_add_owned_ref`, `test_add_ref_owned`
  - 前置: T2
  - 预计: 10 min

- [ ] **T4**: 实现 `Add` trait（张量×标量、标量×张量）
  - 文件: `src/overload/arithmetic.rs`
  - 内容: 标量组合 impl
  - 测试: `test_add_scalar`, `test_scalar_wrapper_add_tensor`, `test_native_scalar_add_tensor_f64`, `test_native_scalar_add_tensor_i32`
  - 前置: T2
  - 预计: 10 min

### Wave 4: 其他运算符

- [ ] **T5**: 实现 `Sub`/`Mul`/`Div`（复制 `Add` 模式）
  - 文件: `src/overload/arithmetic.rs`
  - 内容: Sub/Mul/Div 所有组合
  - 测试: `test_sub`, `test_mul`, `test_div`
  - 前置: T3, T4
  - 预计: 10 min

### Wave 5: 测试

- [ ] **T6**: 编写综合测试
  - 文件: `tests/test_overload.rs`
  - 内容: 广播组合、标量组合、类型组合、深拷贝验证
  - 测试: 覆盖所有公共 API
  - 前置: T1-T5
  - 预计: 10 min

---

## 8. 测试计划

### 8.1 测试分类表

| 测试分类 | 位置                     | 说明                                                        |
| -------- | ------------------------ | ----------------------------------------------------------- |
| 单元测试 | `#[cfg(test)] mod tests` | 验证运算符语法、广播分派与结果所有权语义                    |
| 集成测试 | `tests/`                 | 验证 `overload` 与 `broadcast`、`math`、`tensor` 的协同路径 |
| 边界测试 | 同模块测试中标注         | 覆盖标量、空张量和广播不兼容等边界                          |
| 属性测试 | `tests/property/`        | 验证广播后输出形状、借用/所有权等价性与标量路径不变量       |

### 8.2 单元测试清单

| 测试函数                            | 测试内容                                                           | 优先级 |
| ----------------------------------- | ------------------------------------------------------------------ | ------ |
| `test_add_same_shape`               | `[2,3] + [2,3]` 返回 `Ok(...)`，并逐元素验证                       | 高     |
| `test_add_broadcast`                | `[2,1,3] + [3]` 返回 `Ok(...)`，广播后相加                         | 高     |
| `test_add_ref_ref`                  | `&a + &b` 返回 `Ok(...)`，所有权保留                               | 高     |
| `test_add_owned_ref`                | `a + &b` 返回 `Ok(...)`，a 被消费                                  | 中     |
| `test_add_ref_owned`                | `&a + b` 返回 `Ok(...)`，b 被消费                                  | 中     |
| `test_add_scalar`                   | `tensor + 5.0` 直接返回 `Tensor`                                   | 高     |
| `test_scalar_wrapper_add_tensor`    | `Scalar(5.0) + tensor` 直接返回 `Tensor`                           | 高     |
| `test_native_scalar_add_tensor_f64` | 原生 `5.0 + tensor`（具体类型 impl）直接返回 `Tensor`              | 高     |
| `test_native_scalar_add_tensor_i32` | 原生 `5i32 + tensor`（具体类型 impl）直接返回 `Tensor`             | 中     |
| `test_sub_basic`                    | `a - b` 返回 `Ok(...)` 且结果正确                                  | 高     |
| `test_mul_basic`                    | `a * b` 返回 `Ok(...)` 且结果正确                                  | 高     |
| `test_div_basic`                    | `a / b` 返回 `Ok(...)` 且结果正确                                  | 高     |
| `test_broadcast_incompatible`       | 不兼容形状时运算符与方法路径都返回 `Result::Err(XenonError::BroadcastError)` | 中     |
| `test_result_ownership`             | `Ok` 中结果张量与输入不共享内存                                    | 高     |
| `test_i32_tensor`                   | `i32` 类型张量运算返回 `Ok(...)`                                   | 中     |
| `test_complex_tensor`               | `Complex<f64>` 类型张量运算返回 `Ok(...)`                          | 中     |

### 8.3 边界测试场景

| 场景                                     | 预期行为                       |
| ---------------------------------------- | ------------------------------ |
| 0 维张量 + 0 维张量                      | 返回 `Ok`，执行张量×张量广播语义 |
| 空张量 + 空张量                          | 返回 `Ok`，得到空张量结果      |
| `[1, 1000] + [1000, 1]`                  | 返回 `Ok`，广播到 `[1000, 1000]` |
| 标量 + 0 维张量                          | 直接返回 `Tensor`，正常运算    |
| 大张量 `[10000, 10000] + [10000, 10000]` | 返回 `Ok`，正确完成            |
| `[2, 3] + [4, 5]`                        | 返回 `Err(XenonError::BroadcastError)` |

### 8.4 属性测试不变量

| 不变量                                                     | 测试方法                     |
| ---------------------------------------------------------- | ---------------------------- |
| `(a + b).unwrap().shape() == broadcast_shape(a.shape(), b.shape())` | 随机形状对（仅对可广播输入） |
| `(&a + &b) == (a.clone() + b.clone())`                     | 借用与所有权 `Result` 一致   |
| `(a + scalar) == a.add_scalar(scalar)`                     | 标量路径结果等价             |
| `Scalar(s) + tensor == tensor + s`                         | 包装器左标量与右标量路径等价 |
| 结果张量与输入张量不共享内存（`ptr` 不同）                 | 对 `Ok` 结果做指针比较       |

### 8.5 集成测试

| 测试文件                 | 测试内容                                                              |
| ------------------------ | --------------------------------------------------------------------- |
| `tests/test_overload.rs` | 运算符语法与 `broadcast`、`math`、`tensor` 返回所有权语义的端到端集成 |

### 8.6 Feature gate / 配置测试

| 配置 | 验证点 |
| ---- | ------ |
| 默认配置 | 运算符语法在纯标量后端下与方法型 API 语义保持一致，包括广播失败返回 `Result::Err`。 |
| 启用 SIMD 相关 feature（按 `Cargo.toml` 实际命名） | 通过 `math` 委托的 SIMD 路径不改变广播、`Result` 与结果所有权语义。 |
| 启用 rayon 并行 feature（按 `Cargo.toml` 实际命名） | 通过 `math` 委托的并行路径不改变广播、错误边界与结果所有权语义。 |

### 8.7 类型边界 / 编译期测试

| 场景 | 测试方式 |
| ---- | ---- |
| `AddAssign` / `SubAssign` / `MulAssign` / `DivAssign` 不属于当前 API | API 缺失断言。 |
| bitwise / comparison operators 不在本模块范围内 | 编译期失败测试或 API 缺失断言。 |
| 常用原生左标量仅对受支持具体类型提供实现 | 编译期测试。 |
| `bool` 不参与四则运算符重载 | compile-fail 测试。 |
| `usize` 不属于运算符元素类型 | compile-fail 测试。 |
| 混合元素类型（如 `Tensor<f64> + Tensor<i32>`）不自动提升 | compile-fail 测试。 |

---

## 9. 与其他模块的交互

### 9.1 接口约定

| 方向                     | 对方模块    | 接口/类型                  | 约定                                                            |
| ------------------------ | ----------- | -------------------------- | --------------------------------------------------------------- |
| `arithmetic → math`      | `math`      | `add() / sub() / mul() / div()` / scalar helpers | 张量路径走方法型逐元素运算，标量路径走内部 scalar helper，参见 `11-math.md` §5 |
| `arithmetic → tensor`    | `tensor`    | `Tensor<A, D>` / `.view()`        | 构造 owned 结果并在需要时创建视图，参见 `07-tensor.md` §5        |
| `arithmetic → element`   | `element`   | `Numeric`                         | 通过元素约束排除不支持的类型，参见 `03-element.md` §5.2          |
| `arithmetic → dimension` | `dimension` | `<D as BroadcastDim<E>>::Output`  | 通过维度级关联类型推导广播输出形状，参见 `02-dimension.md` §5.10 |

### 9.2 数据流描述

```text
User writes a + b / tensor + scalar / Scalar(x) + tensor
    │
    ├── overload selects the matching trait impl
    ├── tensor×tensor delegates to broadcast_shape() + broadcast_to() + method dispatch (see 11-math.md §6.2)
    ├── tensor×scalar delegates to scalar method dispatch
    └── tensor / storage allocate a new owned result tensor; tensor×tensor paths return `Result`, scalar paths return `Tensor`
```

---

## 10. 错误处理与语义边界

| 主题 | 内容 |
| ---- | ---- |
| Recoverable error | 项目级稳定的可恢复错误语义由运算符路径与显式方法路径共同承担；`+` / `-` / `*` / `/` 以及 `broadcast_with()`、方法型逐元素 API 均返回 `XenonError::BroadcastError`；若方法参数本身非法，则继续使用 `XenonError::InvalidArgument`。 |
| Panic | 广播不兼容不再 panic；整数除零、溢出与结果不可表示继续沿用 `math` 的 panic 语义，且 panic 消息须包含操作类型、元素类型与第一个失败元素索引（若可确定）。 |
| 路径一致性 | 借用 / owned / 标量以及由 `math` 触发的标量 / SIMD 路径必须保持相同输出 shape 与数值语义。 |
| 容差边界 | 当前不引入额外容差；容差基线以 `需求说明书 §28.3` 为权威。若底层 `math` 使用 SIMD，仍须与该基线及标量路径语义一致。 |

---

## 11. 设计决策记录

### 决策 1：是否支持 += 原地运算符

| 属性     | 值                                                                                           |
| -------- | -------------------------------------------------------------------------------------------- |
| 决策     | 当前版本不提供 `+=`/`-=`/`*=`/`/=` 原地运算符                                                |
| 理由     | `需求说明书 §20` 明确"四则运算以外的运算符语法不在当前范围内"；原地运算符涉及 LHS 广播约束复杂 |
| 替代方案 | 提供 `AddAssign` 等 impl — 留待未来版本                                                      |
| 拒绝原因 | 会把当前文档从纯表达式语法扩展到原地写入语义，增加广播别名与可变借用复杂度                   |

### 决策 2：广播错误处理方式

| 属性     | 值                                                                                        |
| -------- | ----------------------------------------------------------------------------------------- |
| 决策     | 运算符重载在广播不兼容时返回 `Result`；方法型 API 保持相同的 `Result` 返回                |
| 理由     | 为与 `需求说明书 §12` / `需求说明书 §27` 保持一致，广播错误必须以返回值形式报告；虽然这偏离 `std::ops` 的常见习惯，但 Xenon 的错误模型优先 |
| 替代方案 | 保持“运算符 panic / 方法 Result”分离语义，或让方法型 API 也 panic                         |
| 拒绝原因 | 前者直接违背需求约束；后者会抹掉 Xenon 公开 API 的可恢复错误通道                          |

### 决策 3：运算符返回 Result

| 属性     | 值 |
| -------- | --- |
| 决策     | 四则运算符的 `Output` 类型为 `Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>` |
| 理由     | 广播不兼容时须返回可恢复错误（`需求说明书 §20` / `需求说明书 §27`）；运算符是唯一的公开入口，不可静默 panic |
| 替代方案 | 运算符 panic + 提供 `try_add` / `try_sub` 系列方法 — 放弃，因为需求明确要求广播不兼容为可恢复错误，panic 违反语义 |
| 替代方案 | 运算符不返回 `Result`，广播失败由单独的 broadcast 步骤处理 — 放弃，增加调用复杂度 |
| 确认     | 本决策经跨模块评审后确认，现作为项目级稳定 API 风格决策生效 |

---

## 12. 性能考量

### 12.1 复杂度

| 操作                  | 时间复杂度  | 空间复杂度  | 说明                |
| --------------------- | ----------- | ----------- | ------------------- |
| 张量 + 张量（同形状） | O(n)        | O(n)        | 无广播开销          |
| 张量 + 张量（广播）   | O(output_n) | O(output_n) | 广播视图 O(1) 创建  |
| 张量 + 标量           | O(n)        | O(n)        | `*_scalar` 直接迭代 |
| 标量 + 张量           | O(n)        | O(n)        | `*_scalar` 直接迭代 |

### 12.2 SIMD 路径

当 SIMD feature 启用时，方法型逐元素运算与标量方法会在满足前提时自动选择 SIMD 路径（参见 `08-simd.md` §5）：

| 运算符    | SIMD 指令           | 加速比 |
| --------- | ------------------- | ------ |
| `+` (f32) | `AVX _mm256_add_ps` | 4-8x   |
| `+` (f64) | `AVX _mm256_add_pd` | 2-4x   |
| `*` (f32) | `AVX _mm256_mul_ps` | 4-8x   |
| `/` (f64) | `AVX _mm256_div_pd` | 2-4x   |

### 12.3 借用引用优化

```rust,ignore
// &a + &b: no ownership transfer, borrow only
// Internally: self.view() creates a lightweight view (O(1))
// Result: Ok(new Tensor allocation) or Err(XenonError) (O(n) on success)

// a + b: a and b are consumed
// If a/b are not used afterwards, the owned form avoids explicit borrow overhead
// However, the & form is recommended to avoid accidental consumption
```

---

## 13. 平台与工程约束

| 约束       | 说明                                                        |
| ---------- | ----------------------------------------------------------- |
| `std` only | Xenon 当前版本仅支持 `std` 环境，本文不再讨论 `no_std` 路径 |
| MSRV       | Rust 1.85+                                                  |
| 单 crate   | `overload` 设计保持在现有 crate 内，不引入额外 crate        |
| 最小依赖   | 本模块不新增第三方依赖                                      |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
