# 运算符重载模块设计

> 文档编号: 19 | 模块: `src/overload/` | 阶段: Phase 4
> 前置文档: `11-math.md`, `15-broadcast.md`
> 需求参考: 需求说明书 §20, §27, §28.4
> 范围声明: 范围内

---

## 1. 模块定位

### 1.1 职责边界

| 职责               | 包含                                                                          | 不包含                                           |
| ------------------ | ----------------------------------------------------------------------------- | ------------------------------------------------ |
| 四则运算运算符语法 | `+`/`-`/`*`/`/` 运算符重载                                                    | 原地运算符 `+=`/`-=`/`*=`/`/=`（当前版本不提供） |
| 张量×张量运算      | 同形状运算、广播运算                                                          | 矩阵乘法（由 `matrix` 提供）                     |
| 张量×标量运算      | `tensor op scalar`、`Scalar(scalar) op tensor` 与常用原生左标量 `scalar op tensor` | 完全泛型的 `T op Tensor<T>` blanket impl         |
| 广播支持           | 运算符语法内建支持广播                                                        | 比较运算符（在 `math` 提供）                     |
| 新张量产生         | 所有组合产生新的独立张量                                                      | 原地修改运算                                     |
| 借用形式           | `&Tensor op &Tensor`/`&Tensor op Tensor` 等组合                               | 若后续提供 `[]` 运算符，由 `index` 模块承接；当前不属稳定 API |

### 1.2 设计原则

| 原则       | 体现                                            |
| ---------- | ----------------------------------------------- |
| 委托模式   | 运算符重载委托给逐元素运算，运算符仅为语法糖    |
| 深拷贝结果 | 所有组合均产生新的独立张量，不共享内存          |
| 广播透明   | 运算符语法内建支持广播，用户无需手动处理        |
| 借用优先   | 鼓励使用 `&a + &b` 形式，避免不必要的所有权转移 |

### 1.3 在架构中的位置

```
Dependency levels:
L0: error, private
L1: dimension, element, complex
L2: layout (depends on dimension)
L3: storage (independent of layout; tensor owns storage and consumes layout results)
L4: tensor (depends on storage, dimension)
L5: broadcast, iter
L6: math (element-wise operations)
L7: overload  <- current module (depends on broadcast, math)
```

---

## 2. 需求映射与范围约束

| 类型     | 内容 |
| -------- | ---- |
| 需求映射 | 需求说明书 §20, §27, §28.4 |
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

运算符重载文件 `arithmetic.rs` 独立于逐元素运算 `math`，职责清晰：前者提供语法糖，后者提供计算能力。

---

## 4. 依赖关系与实现约束

### 4.1 不变量

| 不变量 | 说明 |
| ------ | ---- |
| 语法糖边界 | `+` / `-` / `*` / `/` 只是对方法型逐元素运算的语法糖；成功时返回新的 owned 张量。 |
| 广播前提 | 张量×张量运算必须先满足广播兼容；广播结果 shape 由 `<D as BroadcastDim<E>>::Output` 推导。 |
| 错误边界 | 运算符路径与方法型 API 在广播失败时均返回 `Result<Tensor<A, F>, XenonError>`；整数除零、整数溢出与结果不可表示仍沿用 panic 语义。 |
| 标量路径 | 张量×标量委托给 `*_scalar` helper；标量路径不涉及广播错误，返回值直接为 `Tensor<A, D>`。 |
| 结果所有权 | 所有运算符组合在成功时都返回独立的新张量，不与输入共享存储。 |

### 4.2 错误场景

| 场景 | 对外语义 |
| ---- | -------- |
| 方法型 API 广播失败 | 返回 `XenonError::BroadcastError { operation: "add", lhs_shape: lhs.shape().into(), rhs_shape: rhs.shape().into(), attempted_target_shape: None, axis: None }`（`sub` / `mul` / `div` 同理）。当两个 shape 本身不兼容时，不人为伪造目标 shape。 |
| 运算符路径广播失败 | 返回 `Err(XenonError::BroadcastError { ... })`；其项目级稳定 ADR 定位见本节后文 `ADR-OVERLOAD-RESULT`。 |
| 整数除零、整数溢出、结果不可表示 | 沿用底层逐元素方法的 panic 语义，不包装为 `Result`；panic 消息须携带操作类型（`add` / `sub` / `mul` / `div`）、元素类型，以及触发位置（第一个溢出的元素索引，若可确定）。 |
| 标量路径参数合法 | `tensor op scalar`、`Scalar(scalar) op tensor` 与常用原生左标量路径不产生广播错误，直接返回 `Tensor`；整数溢出仍遵循 panic 语义。 |

> [!IMPORTANT]
> **ADR-OVERLOAD-RESULT（项目级稳定）**：经过跨模块评审，张量×张量运算符返回 `Result<Tensor, XenonError>` 已确认为项目级稳定契约。规范实现路径为 `impl TensorBinOp for TensorBase<Owned<A>, D>` 返回 `Result<Tensor<A, D>, XenonError>`；本文后续提及“运算符返回 `Result`”时，均指向这一项目级稳定设计结论。

### 4.3 依赖图（ASCII）

```
                    ┌───────────────────┐
                    │      math         │
                    │ add/sub/mul/div   │
                    └─────────┬─────────┘
                              │ uses
                    ┌─────────▼─────────┐
                    │    arithmetic     │
                    │   arithmetic.rs   │
                    └─────────┬─────────┘
                              │ uses
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
      ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
      │   broadcast   │ │    tensor     │ │   element     │
      │ broadcast_*   │ │ TensorBase    │ │ Numeric trait │
      └───────────────┘ └───────────────┘ └───────────────┘
```

### 4.4 类型级依赖

| 来源模块    | 使用的类型/trait                                                                                                     |
| ----------- | -------------------------------------------------------------------------------------------------------------------- |
| `math`      | `add()` / `sub()` / `mul()` / `div()` 等方法型逐元素运算（参见 `11-math.md` §5）                                     |
| `broadcast` | `broadcast_shape()`, `broadcast_with()`, `can_broadcast()`（参见 `15-broadcast.md` §5）                              |
| `tensor`    | `TensorBase<S, D>`, `Tensor<A, D>`, `TensorView`, `.view()`（参见 `07-tensor.md` §5）                                |
| `element`   | `Numeric` trait 约束（排除 `bool` 与 `usize`）（参见 `03-element.md` §5.2）                                          |
| `dimension` | `Dimension`, `Ix0`~`Ix6`, `IxDyn`, `BroadcastDim<E>`（该 trait 定义于 `02-dimension.md §5.9`，计算广播后的维度类型） |

> **Numeric 隐含 Copy：** `Numeric` trait 继承自 `Element`，而 `Element: Copy`（见 `03-element.md` §5.1）。因此所有 `Numeric` 类型均满足 `Copy`，可以在标量运算中安全地按值传递而无需额外约束。

> [!IMPORTANT]
> 张量×张量运算符返回 `Result<Tensor, XenonError>` 的项目级稳定 ADR 见 §4.2 `ADR-OVERLOAD-RESULT`；此处仅引用其当前结论：张量×张量 → `Result`（广播可能失败），张量×标量 → `Tensor`（标量总可广播）。

### 4.5 依赖方向声明

> **依赖方向：单向向上。** `arithmetic` 仅消费 `math`、`broadcast`、`tensor`、`element` 的 trait 和类型，不被它们依赖。`arithmetic` 是最上层的用户 API 模块。

### 4.6 依赖合法性与替代方案

| 项目           | 说明 |
| -------------- | ---- |
| 新增第三方依赖 | 无 |
| 合法性结论     | 合法；当前设计仅复用 Xenon 既有模块、标准库以及文档中已声明的项目内可选能力。 |
| 替代方案       | 不适用；当前范围内无需额外第三方依赖。 |

---

## 5. 公共 API 设计

### 5.1 运算符 trait 实现矩阵

完整的 `impl` 组合表（以 `Add` 为例，`Sub`/`Mul`/`Div` 同理）：

> **稳定性分层说明：** 下表仅列出当前稳定承诺；`TensorView` / `ArcTensor` 相关增强候选已移至文末“附录：增强候选”，避免与当前版本范围混淆。

| Lhs                  | Rhs                  | Output         | 广播     | impl 签名                                                                           |
| -------------------- | -------------------- | -------------- | -------- | ----------------------------------------------------------------------------------- |
| `Tensor<A, D>`       | `Tensor<A, E>`       | `Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>` | ✓        | `impl<...> Add<TensorBase<Owned<A>,E>> for TensorBase<Owned<A>,D>`                  |
| `&Tensor<A, D>`      | `&Tensor<A, E>`      | `Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>` | ✓        | `impl<...> Add<&TensorBase<Owned<A>,E>> for &TensorBase<Owned<A>,D>`                |
| `Tensor<A, D>`       | `&Tensor<A, E>`      | `Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>` | ✓        | `impl<...> Add<&TensorBase<Owned<A>,E>> for TensorBase<Owned<A>,D>`                 |
| `&Tensor<A, D>`      | `Tensor<A, E>`       | `Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>` | ✓        | `impl<...> Add<TensorBase<Owned<A>,E>> for &TensorBase<Owned<A>,D>`                 |
| `Tensor<A, D>`       | `A`                  | `Tensor<A, D>` | 标量广播 | `impl<...> Add<A> for TensorBase<Owned<A>,D>`                                       |
| `&Tensor<A, D>`      | `A`                  | `Tensor<A, D>` | 标量广播 | `impl<...> Add<A> for &TensorBase<Owned<A>,D>`                                      |
| `Scalar<A>`          | `Tensor<A, D>`       | `Tensor<A, D>` | 标量广播 | `impl<...> Add<TensorBase<Owned<A>,D>> for Scalar<A>`                               |
| `Scalar<A>`          | `&Tensor<A, D>`      | `Tensor<A, D>` | 标量广播 | `impl<...> Add<&TensorBase<Owned<A>,D>> for Scalar<A>`                              |

> **说明**：`F` 为广播后的维度类型，由 `<D as BroadcastDim<E>>::Output` 关联类型计算。
> `BroadcastDim` 定义于 `02-dimension.md §5.9`。

### 5.2 张量×张量运算符

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
        self.add_tensor_impl(&rhs)
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
        self.add_tensor_impl(rhs)
    }
}
```

> **设计决策引用：** 此处沿用 §4.2 `ADR-OVERLOAD-RESULT` 的项目级稳定结论：`+` / `-` / `*` / `/` 在广播不兼容时返回 `Result<Tensor<A, F>, XenonError>`，而不是 panic。

> **BroadcastDim 约束说明：** 与 `15-broadcast.md` 保持一致；对称张量×张量运算须同时满足 `D: BroadcastDim<E>` 与 `E: BroadcastDim<D>`，以保证输出维度类型可双向收敛到同一关联类型。

> **实现说明：** 委托示例中的 `add_tensor_impl()` 代表与 trait 方法同名的内部/固有辅助入口，用于避免 `fn add(self, rhs) { self.add(&rhs) }` 这类写法产生对 trait 方法自身的递归歧义。

> **语义边界说明：** 广播失败走 `Err(XenonError::BroadcastError { ... })` 的项目级稳定定性见 §4.2 `ADR-OVERLOAD-RESULT`；整数除零、整数溢出与结果不可表示仍保持 panic。本文 §11 的决策 2a / ADR-2b 仅记录该 ADR 在本模块中的细化范围。

> **范围收敛说明：** 当前稳定承诺聚焦 `Owned×Owned`、其借用变体以及标量路径；`TensorView` / `ArcTensor` 相关组合已移至文末“附录：增强候选”，是否进入稳定面以后续版本决议为准。实现优先级：`Owned×Owned` > `Owned×Scalar` > `View×View`。

> **说明**：当前稳定 API 不直接承诺 `TensorView` 与 `ArcRepr` 参与运算符重载；若后续版本需要这些组合，统一参考文末附录。
> `TensorViewMut` **不**直接参与运算符重载。若要使用运算符，必须先调用 `.view()` 获取只读 `TensorView`，再对该只读视图应用运算符。
> 张量×张量路径在广播失败时返回 `Result<Tensor<A, F>, XenonError>`；标量路径无广播失败分支，直接返回 `Tensor<A, D>`。两类路径成功值都为 owned 结果，因为视图本身不拥有数据，无法作为运算结果的存储。

### 5.3 张量×标量运算符

```rust,ignore
/// Newtype wrapper for scalar values, enabling a generic left-scalar path.
///
/// Rust orphan rules forbid blanket impls such as
/// `impl<T> Add<TensorBase<...>> for T`, because the foreign `Self = T`
/// appears before the first local type. However, concrete primitive left-hand
/// sides like `impl Add<Tensor<f32, D>> for f32` remain legal and should be
/// provided for Xenon's supported scalar set when stable syntax requires it.
pub struct Scalar<A>(pub A);

// Tensor + scalar
impl<A, D> Add<A> for TensorBase<Owned<A>, D>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn add(self, rhs: A) -> Self::Output {
        self.add_scalar_impl(rhs)
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
        self.add_scalar_impl(rhs)
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
        rhs.add_scalar_impl(self.0)
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
        rhs.add_scalar_impl(self.0)
    }
}
```

> **说明**：`Scalar<A>` 包装器是实现“泛型左标量 + 张量”时的工程性折中，而不是原生
> `scalar + tensor` 整体不可行的证明。对 Xenon 支持的具体标量类型（`i32`、`i64`、`f32`、`f64`、`Complex<f32>`、`Complex<f64>`），
> 可以逐类型生成 `impl Add<TensorBase<...>> for T`；真正不可行的是 `impl<T> Add<TensorBase<...>> for T` 这种 blanket impl。
> 因此“常用原生标量”在本文中明确指上述 6 个受支持算术元素类型，而不包括 `bool`、`usize` 或其他范围外类型。

> **宏生成说明：** 标量运算符的 LHS/RHS 组合通过宏生成，覆盖矩阵参见 §5.3-5.4。

> **标量路径返回说明：** 标量路径无形状不兼容风险，不返回 `Result`；运算符返回 `Tensor` 直接。整数溢出仍遵循 panic 语义。

> **说明**：当前版本**不**稳定承诺 `&A` 形式的标量运算符重载。公开契约仅保证值形式 `tensor + scalar`、`Scalar(scalar) + tensor`，以及常用原生左标量（如 `5.0 + tensor`）。若后续版本需要 `&A` 支持，应以独立议题评估。

> **说明**：若后续版本采纳附录中的 `TensorView` 增强候选，`Scalar<A>` 可沿用相同的标量运算模式。`TensorViewMut` 若需使用标量运算符，同样必须先调用 `.view()` 转为只读 `TensorView`。

> **设计决策：** 标量运算委托给 `add_scalar_impl()` / `sub_scalar_impl()` / `mul_scalar_impl()` / `div_scalar_impl()` 等内部 helper，
> 其内部直接遍历输入与结果张量，而不是额外暴露通用 helper 作为稳定实现描述。

对左标量的非交换运算，需显式区分 helper：

- `scalar - tensor`：使用 `sub_scalar_left_impl(scalar, tensor)`，逐元素计算 `scalar - each_element`
- `scalar / tensor`：使用 `div_scalar_left_impl(scalar, tensor)`，逐元素计算 `scalar / each_element`
- 这两条路径不能复用现有 `tensor.sub_scalar_impl(scalar)` / `tensor.div_scalar_impl(scalar)`，因为减法与除法不满足交换律

### 5.4 Sub / Mul / Div

`Sub`、`Mul`、`Div` 的实现模式与 `Add` 完全相同，需覆盖与 `Add` 对称的张量/引用/标量/`Scalar<A>` 组合；其中当前稳定范围内的张量×张量路径返回 `Result<Tensor<A, F>, XenonError>`，标量路径返回 `Tensor<A, D>`。附录中的视图 / `ArcTensor` 增强候选沿用相同返回边界。仅替换运算符和对应闭包：

```rust,ignore
// Sub: |a, b| a - b
// Mul: |a, b| a * b
// Div: |a, b| a / b   (constraint A: Numeric + Div<Output = A>)
```

> **除法语义补充：** 对整数类型，`Div` 路径中的除以零和结果不可表示（如最小负值除以 `-1`）
> 均遵循 `require.md` §12 与 §27 的统一 panic 语义；运算符重载仅把广播不兼容报告为 `Result::Err`，不额外吞掉或包装这类不可恢复错误。

#### 5.4.1 标量重载覆盖矩阵

| 算术类型 | 运算符 | `tensor op scalar` | `scalar op tensor` |
| -------- | ------ | ------------------ | ------------------ |
| `i32` | `+` | `add_scalar_impl` | 原生左标量 / `Scalar<A>` → `add_scalar_impl` |
| `i32` | `-` | `sub_scalar_impl` | 原生左标量 / `Scalar<A>` → `sub_scalar_left_impl` |
| `i32` | `*` | `mul_scalar_impl` | 原生左标量 / `Scalar<A>` → `mul_scalar_impl` |
| `i32` | `/` | `div_scalar_impl` | 原生左标量 / `Scalar<A>` → `div_scalar_left_impl` |
| `i64` | `+` | `add_scalar_impl` | 原生左标量 / `Scalar<A>` → `add_scalar_impl` |
| `i64` | `-` | `sub_scalar_impl` | 原生左标量 / `Scalar<A>` → `sub_scalar_left_impl` |
| `i64` | `*` | `mul_scalar_impl` | 原生左标量 / `Scalar<A>` → `mul_scalar_impl` |
| `i64` | `/` | `div_scalar_impl` | 原生左标量 / `Scalar<A>` → `div_scalar_left_impl` |
| `f32` | `+` | `add_scalar_impl` | 原生左标量 / `Scalar<A>` → `add_scalar_impl` |
| `f32` | `-` | `sub_scalar_impl` | 原生左标量 / `Scalar<A>` → `sub_scalar_left_impl` |
| `f32` | `*` | `mul_scalar_impl` | 原生左标量 / `Scalar<A>` → `mul_scalar_impl` |
| `f32` | `/` | `div_scalar_impl` | 原生左标量 / `Scalar<A>` → `div_scalar_left_impl` |
| `f64` | `+` | `add_scalar_impl` | 原生左标量 / `Scalar<A>` → `add_scalar_impl` |
| `f64` | `-` | `sub_scalar_impl` | 原生左标量 / `Scalar<A>` → `sub_scalar_left_impl` |
| `f64` | `*` | `mul_scalar_impl` | 原生左标量 / `Scalar<A>` → `mul_scalar_impl` |
| `f64` | `/` | `div_scalar_impl` | 原生左标量 / `Scalar<A>` → `div_scalar_left_impl` |
| `Complex<f32>` | `+` | `add_scalar_impl` | 原生左标量 / `Scalar<A>` → `add_scalar_impl` |
| `Complex<f32>` | `-` | `sub_scalar_impl` | 原生左标量 / `Scalar<A>` → `sub_scalar_left_impl` |
| `Complex<f32>` | `*` | `mul_scalar_impl` | 原生左标量 / `Scalar<A>` → `mul_scalar_impl` |
| `Complex<f32>` | `/` | `div_scalar_impl` | 原生左标量 / `Scalar<A>` → `div_scalar_left_impl` |
| `Complex<f64>` | `+` | `add_scalar_impl` | 原生左标量 / `Scalar<A>` → `add_scalar_impl` |
| `Complex<f64>` | `-` | `sub_scalar_impl` | 原生左标量 / `Scalar<A>` → `sub_scalar_left_impl` |
| `Complex<f64>` | `*` | `mul_scalar_impl` | 原生左标量 / `Scalar<A>` → `mul_scalar_impl` |
| `Complex<f64>` | `/` | `div_scalar_impl` | 原生左标量 / `Scalar<A>` → `div_scalar_left_impl` |

> **矩阵生成说明：** 上表由宏展开后的规范化结果表示；实际实现中，标量运算符的 LHS/RHS 组合通过宏生成，覆盖矩阵参见 §5.3-5.4。

### 5.5 Good / Bad 对比

```rust,ignore
// Good - use borrowed form to avoid ownership transfer
fn compute(a: &Tensor<f64, Ix2>, b: &Tensor<f64, Ix2>) -> Result<Tensor<f64, Ix2>, XenonError> {
    a + b  // &Tensor + &Tensor -> Result<new Tensor, XenonError>
}

// Good - use explicit API for broadcast safety
fn compute_safe(a: &Tensor<f64, Ix2>, b: &Tensor<f64, Ix1>) -> Result<Tensor<f64, Ix2>, XenonError> {
    a.add(b)
}

// Bad - mixing owned and borrowed (unnecessarily consumes a)
fn compute_bad(a: Tensor<f64, Ix2>, b: &Tensor<f64, Ix2>) -> Result<Tensor<f64, Ix2>, XenonError> {
    a + b  // a is consumed, cannot be used afterwards
}
```

---

## 6. 内部实现设计

### 6.1 委托模式

运算符重载的核心设计模式是 **委托**：

```
Operator syntax (arithmetic.rs)
     |
     | delegates to
     v
Element-wise math (math methods)
     |
     | uses
     v
Broadcast module (broadcast.rs) -- memory access (storage)
```

运算符 `a + b` 展开为：

1. `broadcast_with(&a.view(), &b.view())` — 广播两个张量
2. `add()` / `sub()` / `mul()` / `div()` — 直接逐元素遍历广播后视图并写入新结果张量

### 6.2 深拷贝保证

所有运算符在成功路径上产生的新张量是独立的：

- 方法型逐元素运算分配新的 `Owned` 存储并逐元素写入
- 新张量与输入张量不共享内存
- `Tensor<A, D>` 类型保证所有权独占

### 6.3 标量路径优化

标量×张量运算使用专门的标量方法型 API，而非广播视图：

```
tensor + scalar:
    tensor.add_scalar_impl(scalar)

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

- [ ] **T2**: 实现 `Add` trait（张量×张量，所有权形式）
  - 文件: `src/overload/arithmetic.rs`
  - 内容: `Tensor + Tensor` impl
  - 测试: `test_add_same_shape`, `test_add_broadcast`
  - 前置: T1
  - 预计: 10 min

### Wave 2: 借用形式

- [ ] **T3**: 实现 `Add` trait（&张量×&张量、混合形式）
  - 文件: `src/overload/arithmetic.rs`
  - 内容: 4 种借用组合
  - 测试: `test_add_ref_ref`, `test_add_owned_ref`, `test_add_ref_owned`
  - 前置: T2
  - 预计: 10 min

### Wave 3: 标量运算符

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

### 并行执行图

```
Wave 1: [T1] → [T2]
                  │
Wave 2:      [T3]
                  │
Wave 3:      [T4]
                  │
Wave 4:      [T5]
                  │
Wave 5:      [T6]
```

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
| `test_broadcast_incompatible`       | 不兼容形状时运算符与方法路径都返回 `Result::Err(XenonError::BroadcastError { .. })` | 中     |
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
| `[2, 3] + [4, 5]`                        | 返回 `Err(XenonError::BroadcastError { .. })` |

### 8.4 §28.4 边界测试占位

| 占位场景 | 说明 |
| -------- | ---- |
| 高维广播链 | 预留给 `require.md §28.4` 的高维广播边界用例（如 `Ix6` / `IxDyn` 混合广播） |
| 大规模整数 panic 诊断 | 预留给 `require.md §28.4` 的整数溢出/除零诊断验证，断言 panic 消息包含操作类型、元素类型与首个失败索引 |
| `ArcRepr` 只读参与 | 预留给 `require.md §28.4` 的共享只读张量参与四则运算边界用例 |

### 8.5 属性测试不变量

| 不变量                                                     | 测试方法                     |
| ---------------------------------------------------------- | ---------------------------- |
| `(a + b).unwrap().shape() == broadcast_shape(a.shape(), b.shape())` | 随机形状对（仅对可广播输入） |
| `(&a + &b) == (a.clone() + b.clone())`                     | 借用与所有权 `Result` 一致   |
| `(a + scalar) == a.add_scalar_impl(scalar)`                | 标量路径结果等价            |
| `Scalar(s) + tensor == tensor + s`                         | 包装器左标量与右标量路径等价 |
| 结果张量与输入张量不共享内存（`ptr` 不同）                 | 对 `Ok` 结果做指针比较       |

### 8.6 集成测试

| 测试文件                 | 测试内容                                                              |
| ------------------------ | --------------------------------------------------------------------- |
| `tests/test_overload.rs` | 运算符语法与 `broadcast`、`math`、`tensor` 返回所有权语义的端到端集成 |

### 8.7 Feature gate / 配置测试

| 配置 | 验证点 |
| ---- | ---- |
| 默认配置 | 运算符语法在纯标量后端下与方法型 API 语义保持一致，包括广播失败返回 `Result::Err`。 |
| 启用 `simd` | 通过 `math` 委托的 SIMD 路径不改变广播、`Result` 与结果所有权语义。 |
| 启用并行 | 通过 `math` 委托的并行路径不改变广播、错误边界与结果所有权语义。 |

### 8.8 类型边界 / 编译期测试

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
| `arithmetic → math`      | `math`      | `add()` / `sub()` / `mul()` / `div()` / scalar helpers | 张量路径走方法型逐元素运算，标量路径走内部 scalar helper，参见 `11-math.md` §5 |
| `arithmetic → broadcast` | `broadcast` | `broadcast_with()`                  | 先把两个操作数广播到公共形状，参见 `15-broadcast.md` §5         |
| `arithmetic → tensor`    | `tensor`    | `Tensor<A, D>` / `.view()`          | 构造 owned 结果并在需要时创建视图，参见 `07-tensor.md` §5       |
| `arithmetic → element`   | `element`   | `Numeric`                           | 通过元素约束排除不支持的类型，参见 `03-element.md` §5.2         |
| `arithmetic → dimension` | `dimension` | `<D as BroadcastDim<E>>::Output`    | 通过维度级关联类型推导广播输出形状，参见 `02-dimension.md` §5.9 |

### 9.2 数据流描述

```text
User writes a + b / tensor + scalar / Scalar(x) + tensor
    │
    ├── overload selects the matching trait impl
    ├── tensor×tensor delegates to broadcast_with() + method dispatch
    ├── tensor×scalar delegates to scalar method dispatch
    └── tensor / storage allocate a new owned result tensor; tensor×tensor paths return `Result`, scalar paths return `Tensor`
```

---

## 10. 错误处理与语义边界

| 主题 | 内容 |
| ---- | ---- |
| Recoverable error | 项目级稳定的可恢复错误语义由运算符路径与显式方法路径共同承担；`+` / `-` / `*` / `/` 以及 `broadcast_with()`、方法型逐元素 API 均返回 `XenonError::BroadcastError { operation: &'static str, lhs_shape: Vec<usize>, rhs_shape: Vec<usize>, attempted_target_shape: Option<Vec<usize>>, axis: Option<usize> }`；若方法参数本身非法，则继续使用 `XenonError::InvalidArgument { operation: Cow<'static, str>, argument: Cow<'static, str>, expected: Cow<'static, str>, actual: Cow<'static, str>, axis: Option<usize>, shape: Option<Vec<usize>> }`。 |
| Panic | 广播不兼容不再 panic；整数除零、溢出与结果不可表示继续沿用 `math` 的 panic 语义，且 panic 消息须包含操作类型、元素类型与第一个失败元素索引（若可确定）。 |
| 路径一致性 | 借用 / owned / 标量以及由 `math` 触发的标量 / SIMD 路径必须保持相同输出 shape 与数值语义。 |
| 容差边界 | 当前不引入额外容差；若底层 `math` 使用 SIMD，仍须与标量路径语义一致。 |

---

## 11. 设计决策记录

> [!WARNING]
> 参见 §4.2 `ADR-OVERLOAD-RESULT`。本节补充该项目级稳定 ADR 的细化记录：`a + b` 需要 `?` 或 `unwrap()` 来获取结果，而 `a + scalar` 不需要；这一差异现已属于项目级 API 风格约束。

### 决策 1：是否支持 += 原地运算符

| 属性     | 值                                                                                           |
| -------- | -------------------------------------------------------------------------------------------- |
| 决策     | 当前版本不提供 `+=`/`-=`/`*=`/`/=` 原地运算符                                                |
| 理由     | 需求说明书 §20 明确"四则运算以外的运算符语法不在当前范围内"；原地运算符涉及 LHS 广播约束复杂 |
| 替代方案 | 提供 `AddAssign` 等 impl — 留待未来版本                                                      |
| 拒绝原因 | 会把当前文档从纯表达式语法扩展到原地写入语义，增加广播别名与可变借用复杂度                    |

### 决策 2：广播错误处理方式

| 属性     | 值                                                                                        |
| -------- | ----------------------------------------------------------------------------------------- |
| 决策     | 运算符重载在广播不兼容时返回 `Result`；方法型 API 保持相同的 `Result` 返回                   |
| 理由     | 为与 `require.md` §12 / §27 保持一致，广播错误必须以返回值形式报告；虽然这偏离 `std::ops` 的常见习惯，但 Xenon 的错误模型优先 |
| 替代方案 | 保持“运算符 panic / 方法 Result”分离语义，或让方法型 API 也 panic                           |
| 拒绝原因 | 前者直接违背需求约束；后者会抹掉 Xenon 公开 API 的可恢复错误通道                             |

> **补充**：运算符与方法型 API 现在共享广播错误的恢复主路径；整数除零、整数溢出和结果不可表示等不可恢复错误则继续遵循 `require.md` §12 / §27 的 panic 语义。

### 决策 2a：运算符返回 Result

| 属性     | 值 |
| -------- | --- |
| 决策     | 四则运算符的 `Output` 类型为 `Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>` |
| 理由     | 广播不兼容时须返回可恢复错误（`require.md` §20 / §27）；运算符是唯一的公开入口，不可静默 panic |
| 替代方案 | 运算符 panic + 提供 `try_add` / `try_sub` 系列方法 — 放弃，因为需求明确要求广播不兼容为可恢复错误，panic 违反语义 |
| 替代方案 | 运算符不返回 `Result`，广播失败由单独的 broadcast 步骤处理 — 放弃，增加调用复杂度 |
| 确认     | 本决策经跨模块评审后确认，现作为项目级稳定 API 风格决策生效 |

### 运算符返回 `Result` 的代价分析

| 方面 | 影响 |
|------|------|
| 链式表达式 | `a + b + c` 须改写为 `(a + b)? + c`，每层运算均需 `?` 传播 |
| 泛型互操作 | `std::ops::Add<Output=Result<_,_>>` 与标准库运算符语义不兼容 |
| 用户心智成本 | 所有张量运算表达式均需考虑错误处理路径 |
| 设计理由 | 广播不兼容须返回可恢复错误（需求说明书 §20），运算符无法通过类型系统排除不兼容输入 |

> **决策**：接受上述代价，运算符统一返回 `Result`。这是在"运算符便利性"与"需求要求广播错误可恢复"之间的显式权衡。

### ADR-2b：仅张量×张量路径共享 Result 边界

| 属性     | 值 |
| -------- | --- |
| 决策     | 仅张量×张量/视图路径在广播失败时返回 `Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>`；标量路径直接返回 `Tensor<A, D>` |
| 理由     | `require.md §20` 只要求广播支持张量与标量之间的逐元素运算，但标量路径不存在形状不兼容分支；因此只把真正可能出现的广播错误保留在张量×张量路径，既满足 `require.md §12 / §27` 的可恢复错误约束，也避免为无错误分支的标量路径强加 `Result` |
| 替代方案 | 所有运算符路径统一返回 `Result`，或让张量×张量路径也 panic |
| 拒绝原因 | 前者会给无广播失败分支的标量路径引入无依据的错误包装；后者违反需求中“可恢复错误须以返回值形式报告”的约束 |

### 决策 3：标量路径使用直接标量方法而非广播视图

| 属性     | 值 |
| -------- | --- |
| 决策     | 张量×标量运算委托给 `*_scalar` 方法，由方法内部直接遍历并写入结果，而非创建广播视图 |
| 理由     | 更高效（直接迭代 vs 间接寻址），同时避免把通用映射 helper 误写成当前版本的稳定设计依赖 |
| 替代方案 | 创建标量广播视图 `Tensor0::from_scalar(scalar).view().broadcast_to(shape)` |
| 拒绝原因 | 会增加间接寻址与额外中间视图概念，不符合当前最小实现描述 |

---

## 12. 性能考量

### 12.1 复杂度

| 操作                  | 时间复杂度  | 空间复杂度  | 说明               |
| --------------------- | ----------- | ----------- | ------------------ |
| 张量 + 张量（同形状） | O(n)        | O(n)        | 无广播开销         |
| 张量 + 张量（广播）   | O(output_n) | O(output_n) | 广播视图 O(1) 创建 |
| 张量 + 标量           | O(n)        | O(n)        | `*_scalar` 直接迭代 |
| 标量 + 张量           | O(n)        | O(n)        | `*_scalar` 直接迭代 |

### 12.2 性能数据（参考）

| 场景                                        | 路径          | 预计性能 |
| ------------------------------------------- | ------------- | -------- |
| `[1000, 1000] + [1000, 1000]` (f64)         | 方法分发 + SIMD | ~1ms   |
| `[1000, 1000] + [1, 1000]` (广播)           | 方法分发 + 广播 | ~1.2ms |
| `[1000, 1000] + 5.0` (标量)                 | `add_scalar`    | ~0.8ms |
| `[1000, 1000] + [1000, 1000]` (f64, 非SIMD) | 方法分发 + 标量 | ~4ms   |

### 12.3 SIMD 路径

当 SIMD feature 启用时，方法型逐元素运算与标量方法会在满足前提时自动选择 SIMD 路径（参见 `08-simd.md` §5）：

| 运算符    | SIMD 指令           | 加速比 |
| --------- | ------------------- | ------ |
| `+` (f32) | `AVX _mm256_add_ps` | 4-8x   |
| `+` (f64) | `AVX _mm256_add_pd` | 2-4x   |
| `*` (f32) | `AVX _mm256_mul_ps` | 4-8x   |
| `/` (f64) | `AVX _mm256_div_pd` | 2-4x   |

### 12.4 借用引用优化

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
| SemVer     | §4.2 `ADR-OVERLOAD-RESULT` 已确认项目级稳定 API 边界；张量×张量路径的规范 `Output` 类型为 `Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>` |
| 最小依赖   | 本模块不新增第三方依赖                                      |

---

## 附录：增强候选

以下增强候选不在当前版本范围内，仅作为未来版本设计参考。

### A.1 运算符 trait 实现矩阵补充

| Lhs                  | Rhs                  | Output         | 广播     | impl 签名                                                                           |
| -------------------- | -------------------- | -------------- | -------- | ----------------------------------------------------------------------------------- |
| `&TensorView<A, D>`（增强候选） | `&TensorView<A, E>`（增强候选） | `Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>` | ✓        | `impl<...> Add<&TensorBase<ViewRepr<'b, A>,E>> for &TensorBase<ViewRepr<'a, A>,D>`（增强候选） |
| `&TensorView<A, D>`（增强候选） | `&Tensor<A, E>`      | `Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>` | ✓        | `impl<...> Add<&TensorBase<Owned<A>,E>> for &TensorBase<ViewRepr<'a, A>,D>`（增强候选） |
| `&Tensor<A, D>`      | `&TensorView<A, E>`（增强候选） | `Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>` | ✓        | `impl<...> Add<&TensorBase<ViewRepr<'b, A>,E>> for &TensorBase<Owned<A>,D>`（增强候选） |
| `&TensorView<A, D>`（增强候选） | `&ArcTensor<A, E>`（增强候选）  | `Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>` | ✓        | `impl<...> Add<&TensorBase<ArcRepr<A>,E>> for &TensorBase<ViewRepr<'a, A>,D>`（增强候选） |
| `&ArcTensor<A, D>`（增强候选）  | `&TensorView<A, E>`（增强候选） | `Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>` | ✓        | `impl<...> Add<&TensorBase<ViewRepr<'b, A>,E>> for &TensorBase<ArcRepr<A>,D>`（增强候选） |
| `&ArcTensor<A, D>`（增强候选）  | `&Tensor<A, E>`      | `Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>` | ✓        | `impl<...> Add<&TensorBase<Owned<A>,E>> for &TensorBase<ArcRepr<A>,D>`（增强候选） |
| `&Tensor<A, D>`      | `&ArcTensor<A, E>`（增强候选）  | `Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>` | ✓        | `impl<...> Add<&TensorBase<ArcRepr<A>,E>> for &TensorBase<Owned<A>,D>`（增强候选） |
| `&ArcTensor<A, D>`（增强候选）  | `&ArcTensor<A, E>`（增强候选）  | `Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>` | ✓        | `impl<...> Add<&TensorBase<ArcRepr<A>,E>> for &TensorBase<ArcRepr<A>,D>`（增强候选） |

### A.2 视图 / 共享只读组合运算符

```rust,ignore
// Enhancement candidate: &ArcTensor + &Tensor (shared read-only + owned reference)
impl<'a, 'b, A, D, E> Add<&'b TensorBase<Owned<A>, E>> for &'a TensorBase<ArcRepr<A>, D>
where
    A: Numeric,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>;

    fn add(self, rhs: &'b TensorBase<Owned<A>, E>) -> Self::Output {
        self.add_tensor_impl(rhs)
    }
}

// Enhancement candidate: &ArcTensor + &ArcTensor (shared read-only + shared read-only)
impl<'a, 'b, A, D, E> Add<&'b TensorBase<ArcRepr<A>, E>> for &'a TensorBase<ArcRepr<A>, D>
where
    A: Numeric,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>;

    fn add(self, rhs: &'b TensorBase<ArcRepr<A>, E>) -> Self::Output {
        self.add_tensor_impl(rhs)
    }
}

// Enhancement candidate: &Tensor + &ArcTensor (owned reference + shared read-only)
impl<'a, 'b, A, D, E> Add<&'b TensorBase<ArcRepr<A>, E>> for &'a TensorBase<Owned<A>, D>
where
    A: Numeric,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>;

    fn add(self, rhs: &'b TensorBase<ArcRepr<A>, E>) -> Self::Output {
        self.add_tensor_impl(rhs)
    }
}

// Enhancement candidate: &TensorView + &TensorView (reference + reference)
impl<'a, 'b, A, D, E> Add<&'b TensorBase<ViewRepr<'b, A>, E>>
    for &'a TensorBase<ViewRepr<'a, A>, D>
where
    A: Numeric,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>;

    fn add(self, rhs: &'b TensorBase<ViewRepr<'b, A>, E>) -> Self::Output {
        self.add_tensor_impl(rhs)
    }
}

// Enhancement candidate: &TensorView + &Tensor (view + owned reference)
impl<'a, 'b, A, D, E> Add<&'b TensorBase<Owned<A>, E>>
    for &'a TensorBase<ViewRepr<'a, A>, D>
where
    A: Numeric,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>;

    fn add(self, rhs: &'b TensorBase<Owned<A>, E>) -> Self::Output {
        self.add_tensor_impl(rhs)
    }
}

// Enhancement candidate: &TensorView + &ArcTensor (view + shared read-only reference)
impl<'a, 'b, A, D, E> Add<&'b TensorBase<ArcRepr<A>, E>>
    for &'a TensorBase<ViewRepr<'a, A>, D>
where
    A: Numeric,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>;

    fn add(self, rhs: &'b TensorBase<ArcRepr<A>, E>) -> Self::Output {
        self.add_tensor_impl(rhs)
    }
}

// Enhancement candidate: &ArcTensor + &TensorView (shared read-only + view reference)
impl<'a, 'b, A, D, E> Add<&'b TensorBase<ViewRepr<'b, A>, E>>
    for &'a TensorBase<ArcRepr<A>, D>
where
    A: Numeric,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>;

    fn add(self, rhs: &'b TensorBase<ViewRepr<'b, A>, E>) -> Self::Output {
        self.add_tensor_impl(rhs)
    }
}
```

> **说明**：`Sub`/`Mul`/`Div` 的视图 / 共享只读组合模式与 `Add` 相同，仅替换运算符和闭包，并统一返回 `Result<Tensor<A, F>, XenonError>`。`ArcRepr<A>` 仅通过 `&self` 解引用为只读视图参与运算，不提供消费式写入语义。`TensorViewMut` **不**直接参与这些组合；需要先调用 `.view()` 转为只读 `TensorView`。

### A.3 可选验证（enhancement candidates）

| 测试函数                            | 测试内容                                                           | 优先级 |
| ----------------------------------- | ------------------------------------------------------------------ | ------ |
| `test_add_view_view`                | `&TensorView + &TensorView` 组合在增强候选实现启用时返回 `Ok(...)` | 中     |
| `test_add_view_tensor`              | `&TensorView + &Tensor` 组合在增强候选实现启用时返回 `Ok(...)`     | 中     |
| `test_add_arc_tensor`               | `&ArcTensor` 参与的组合在增强候选实现启用时返回 `Ok(...)`          | 中     |

> **说明：** 本节仅用于增强候选的可选验证，不属于当前稳定 API 测试基线；若相应增强未实现或未进入稳定面，可不作为默认必过项。

---

## 版本历史

| 版本  | 日期       |
| ----- | ---------- |
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-08 |
| 1.0.2 | 2026-04-08 |
| 1.0.3 | 2026-04-08 |
| 1.0.4 | 2026-04-08 |
| 1.1.0 | 2026-04-08 |
| 1.1.1 | 2026-04-10 |
| 1.1.2 | 2026-04-10 |
| 1.1.3 | 2026-04-10 |
| 1.1.4 | 2026-04-14 |
| 1.1.5 | 2026-04-15 |
| 1.1.6 | 2026-04-15 |
| 1.1.7 | 2026-04-15 |
| 1.1.8 | 2026-04-16 |
| 1.1.9 | 2026-04-16 |
| 1.2.0 | 2026-04-16 |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
