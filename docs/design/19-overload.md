# 运算符重载模块设计

> 文档编号: 19 | 模块: `src/overload/arithmetic.rs` | 阶段: Phase 4
> 前置文档: `11-math.md`, `15-broadcast.md`
> 需求参考: 需求说明书 §20

---

## 1. 模块定位

### 1.1 职责边界

| 职责 | 包含 | 不包含 |
|------|------|--------|
| 四则运算运算符语法 | `+`/`-`/`*`/`÷` 运算符重载 | 原地运算符 `+=`/`-=`/`*=`/`/=`（当前版本不提供） |
| 张量×张量运算 | 同形状运算、广播运算 | 矩阵乘法（由 `matrix` 提供） |
| 张量×标量运算 | 标量广播到张量形状 | 负数运算符 `-`（在 `math` 提供） |
| 广播支持 | 运算符语法内建支持广播 | 比较运算符（在 `math` 提供） |
| 新张量产生 | 所有组合产生新的独立张量 | 原地修改运算 |
| 借用形式 | `&Tensor op &Tensor`/`&Tensor op Tensor` 等组合 | 索引运算符 `[]`（在 `index` 提供） |

### 1.2 设计原则

| 原则 | 体现 |
|------|------|
| 委托模式 | 运算符重载委托给逐元素运算，运算符仅为语法糖 |
| 深拷贝结果 | 所有组合均产生新的独立张量，不共享内存 |
| 广播透明 | 运算符语法内建支持广播，用户无需手动处理 |
| 借用优先 | 鼓励使用 `&a + &b` 形式，避免不必要的所有权转移 |

### 1.3 在架构中的位置

```
依赖层级：
L0: error, private
L1: dimension, element, complex
L2: layout (依赖 dimension)
L3: storage (依赖 layout)
L4: tensor (依赖 storage, dimension)
L5: broadcast, iter
L6: math（逐元素运算）
L7: overload  ← 当前模块（依赖 broadcast, math）
```

---

## 2. 文件位置

```
src/overload/
├── arithmetic.rs       # 四则运算运算符重载
└── mod.rs              # 模块入口
```

运算符重载文件 `arithmetic.rs` 独立于逐元素运算 `math`，职责清晰：前者提供语法糖，后者提供计算能力。

---

## 3. 依赖关系

### 3.1 依赖图（ASCII）

```
                    ┌───────────────────┐
                    │ math               │
                    │ zip_with / mapv    │
                    └─────────┬─────────┘
                              │ 使用
                    ┌─────────▼─────────┐
                    │  arithmetic        │
                    │  arithmetic.rs     │
                    └─────────┬─────────┘
                              │ 使用
              ┌───────────────────┼───────────────────┐
              │                   │                   │
              ▼                   ▼                   ▼
      ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
      │  broadcast    │   │   tensor      │   │  element      │
      │ broadcast_    │   │ TensorBase    │   │ Numeric      │
      │ shape()       │   │ Tensor<A,D>  │   │ trait         │
      └───────────────┘   └───────────────┘   └───────────────┘
```

### 3.2 类型级依赖

| 来源模块 | 使用的类型/trait |
|----------|-----------------|
| `math` | `zip_with()`, `mapv()`, 二元逐元素运算（参见 `11-math.md` §4） |
| `broadcast` | `broadcast_shape()`, `broadcast_with()`, `can_broadcast()`（参见 `15-broadcast.md` §4） |
| `tensor` | `TensorBase<S, D>`, `Tensor<A, D>`, `TensorView`, `.view()`（参见 `07-tensor.md` §4） |
| `element` | `Numeric` trait 约束（排除 `bool` 与 `usize`）（参见 `03-element.md` §3） |
| `dimension` | `Dimension`, `Ix0`~`Ix6`, `IxDyn`, `BroadcastDim<E>`（该 trait 定义于 `02-dimension.md §4.9`，计算广播后的维度类型） |

> **Numeric 隐含 Copy：** `Numeric` trait 继承自 `Element`，而 `Element: Copy`（见 `03-element.md` §4.1）。因此所有 `Numeric` 类型均满足 `Copy`，可以在标量运算中安全地按值传递而无需额外约束。

### 3.3 依赖方向声明

> **依赖方向：单向向上。** `arithmetic` 仅消费 `math`、`broadcast`、`tensor`、`element` 的 trait 和类型，不被它们依赖。`arithmetic` 是最上层的用户 API 模块。

---

## 4. 公共 API 设计

### 4.1 运算符 trait 实现矩阵

完整的 `impl` 组合表（以 `Add` 为例，`Sub`/`Mul`/`Div` 同理）：

| Lhs | Rhs | Output | 广播 | impl 签名 |
|-----|-----|--------|------|-----------|
| `Tensor<A, D>` | `Tensor<A, E>` | `Tensor<A, F>` | ✓ | `impl<...> Add<TensorBase<Owned<A>,E>> for TensorBase<Owned<A>,D>` |
| `&Tensor<A, D>` | `&Tensor<A, E>` | `Tensor<A, F>` | ✓ | `impl<...> Add<&TensorBase<Owned<A>,E>> for &TensorBase<Owned<A>,D>` |
| `Tensor<A, D>` | `&Tensor<A, E>` | `Tensor<A, F>` | ✓ | `impl<...> Add<&TensorBase<Owned<A>,E>> for TensorBase<Owned<A>,D>` |
| `&Tensor<A, D>` | `Tensor<A, E>` | `Tensor<A, F>` | ✓ | `impl<...> Add<TensorBase<Owned<A>,E>> for &TensorBase<Owned<A>,D>` |
| `Tensor<A, D>` | `A` | `Tensor<A, D>` | 标量广播 | `impl<...> Add<A> for TensorBase<Owned<A>,D>` |
| `&Tensor<A, D>` | `&A` | `Tensor<A, D>` | 标量广播 | `impl<...> Add<&A> for &TensorBase<Owned<A>,D>` |
| `Scalar<A>` | `Tensor<A, D>` | `Tensor<A, D>` | 标量广播 | `impl<...> Add<TensorBase<Owned<A>,D>> for Scalar<A>` |
| `Scalar<A>` | `&Tensor<A, D>` | `Tensor<A, D>` | 标量广播 | `impl<...> Add<&TensorBase<Owned<A>,D>> for Scalar<A>` |
| `&TensorView<A, D>` | `&TensorView<A, E>` | `Tensor<A, F>` | ✓ | `impl<...> Add<&TensorBase<ViewRepr<&A>,E>> for &TensorBase<ViewRepr<&A>,D>` |
| `&TensorView<A, D>` | `&Tensor<A, E>` | `Tensor<A, F>` | ✓ | `impl<...> Add<&TensorBase<Owned<A>,E>> for &TensorBase<ViewRepr<&A>,D>` |
| `&Tensor<A, D>` | `&TensorView<A, E>` | `Tensor<A, F>` | ✓ | `impl<...> Add<&TensorBase<ViewRepr<&A>,E>> for &TensorBase<Owned<A>,D>` |

> **说明**：`F` 为广播后的维度类型，由 `D::BroadcastDim<E>::Output` 关联类型计算。
> `BroadcastDim` 定义于 `02-dimension.md §4.9`。

> **说明**：`TensorView` 和 `TensorViewMut` 通过引用模式参与运算符重载（如 `&view + &tensor`）。
> 运算结果始终为 `Tensor<A, F>`（Owned 存储），因为视图本身不拥有数据，无法作为运算结果的存储。

### 4.2 张量×张量运算符

```rust
// Tensor + Tensor (owned + owned)
impl<A, D, E> Add<TensorBase<Owned<A>, E>> for TensorBase<Owned<A>, D>
where
    A: Numeric,
    D: Dimension,
    E: Dimension,
    D: BroadcastDim<E>,
{
    type Output = Tensor<A, <D as BroadcastDim<E>>::Output>;

    fn add(self, rhs: TensorBase<Owned<A>, E>) -> Self::Output {
        let (a_bc, b_bc) = broadcast_with(&self.view(), &rhs.view())
            .expect("broadcast: incompatible shapes");
        zip_with(&a_bc, &b_bc, |a, b| a + b)
            .expect("internal invariant: zip_with should succeed after successful broadcast")
    }
}

// &Tensor + &Tensor (most common form)
impl<'a, 'b, A, D, E> Add<&'b TensorBase<Owned<A>, E>> for &'a TensorBase<Owned<A>, D>
where
    A: Numeric,
    D: Dimension,
    E: Dimension,
    D: BroadcastDim<E>,
{
    type Output = Tensor<A, <D as BroadcastDim<E>>::Output>;

    fn add(self, rhs: &'b TensorBase<Owned<A>, E>) -> Self::Output {
        let (a_bc, b_bc) = broadcast_with(&self.view(), &rhs.view())
            .expect("broadcast: incompatible shapes");
        zip_with(&a_bc, &b_bc, |a, b| a + b)
            .expect("internal invariant: zip_with should succeed after successful broadcast")
    }
}
```

> **设计决策：** 形状不兼容时使用 `expect` panic。
> 这与集中错误语义中的语法糖例外一致：运算符重载 panic 而非返回 Result（参见 `26-error.md` §4.7, §5.2）。
> 在运算符实现内部，`zip_with` 的 `Result` 通过 `expect` 收束为内部不变量：若广播已经成功，则逐元素 zip 不应再失败。
> 用户需要广播安全时可直接使用 `broadcast_with` + `zip_with`。

### 4.2b 视图×视图/张量运算符

```rust
// &TensorView + &TensorView (reference + reference)
impl<'a, 'b, A, D, E> Add<&'b TensorBase<ViewRepr<&'b A>, E>>
    for &'a TensorBase<ViewRepr<&'a A>, D>
where
    A: Numeric,
    D: Dimension,
    E: Dimension,
    D: BroadcastDim<E>,
{
    type Output = Tensor<A, <D as BroadcastDim<E>>::Output>;

    fn add(self, rhs: &'b TensorBase<ViewRepr<&'b A>, E>) -> Self::Output {
        // delegates to math::add with broadcast
        let (a_bc, b_bc) = broadcast_with(&self.view(), &rhs.view())
            .expect("broadcast: incompatible shapes");
        zip_with(&a_bc, &b_bc, |a, b| a + b)
            .expect("internal invariant: zip_with should succeed after successful broadcast")
    }
}

// &TensorView + &Tensor (view + owned reference)
impl<'a, 'b, A, D, E> Add<&'b TensorBase<Owned<A>, E>>
    for &'a TensorBase<ViewRepr<&'a A>, D>
where
    A: Numeric,
    D: Dimension,
    E: Dimension,
    D: BroadcastDim<E>,
{
    type Output = Tensor<A, <D as BroadcastDim<E>>::Output>;

    fn add(self, rhs: &'b TensorBase<Owned<A>, E>) -> Self::Output {
        // delegates to math::add with broadcast
        let (a_bc, b_bc) = broadcast_with(&self.view(), &rhs.view())
            .expect("broadcast: incompatible shapes");
        zip_with(&a_bc, &b_bc, |a, b| a + b)
            .expect("internal invariant: zip_with should succeed after successful broadcast")
    }
}
```

> **说明**：`Sub`/`Mul`/`Div` 的视图组合模式与 `Add` 相同，仅替换运算符和闭包。`ViewMutRepr` 通过 `&*view_mut` 隐式转为 `&TensorBase<ViewRepr<...>>` 后参与运算。

### 4.3 张量×标量运算符

```rust
/// Newtype wrapper for scalar values, enabling `scalar + tensor` syntax.
///
/// Required by Rust's orphan rules: we cannot impl `Add<TensorBase<...>> for A`
/// because both `Add` (external trait) and `A` (unconstrained type param) are
/// foreign to our crate. `Scalar<A>` is a local type, satisfying the orphan rule.
pub struct Scalar<A>(pub A);

// Tensor + scalar
impl<A, D> Add<A> for TensorBase<Owned<A>, D>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn add(self, rhs: A) -> Self::Output {
        self.mapv(|x| x + rhs)
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
        self.mapv(|x| x + rhs)
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
        rhs.mapv(|x| self.0 + x)
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
        rhs.mapv(|x| self.0 + x)
    }
}
```

> **说明**：`Scalar<A>` 包装器是 Rust 孤儿规则所必需的。由于 `Add` 是外部 trait，`A` 是无约束泛型，直接 `impl Add<TensorBase<...>> for A` 违反孤儿规则。使用 `Scalar<A>` 作为本地类型解决此问题。用户需要写 `Scalar(5.0) + tensor` 而非 `5.0 + tensor`。

> **说明**：对于涉及 `&A` 的组合（上表第 2 行），由于 `A: Numeric` 要求 `A: Copy`，`tensor + &scalar` 通过 Rust 的 auto-deref 隐式解引用为值形式，自动调用基于值的 `Add` impl，无需单独实现。

> **说明**：`Scalar<A>` 同样适用于 `TensorView` 和 `TensorViewMut` 的标量运算。

> **设计决策：** 标量运算使用 `mapv` 而非创建广播视图。
> 原因：`mapv` 直接迭代更高效，避免广播视图的间接寻址开销。

### 4.4 Sub / Mul / Div

`Sub`、`Mul`、`Div` 的实现模式与 `Add` 完全相同，仅替换运算符和对应闭包：

```rust
// Sub: |a, b| a - b
// Mul: |a, b| a * b
// Div: |a, b| a / b   (constraint A: Numeric + Div<Output = A>)
```

### 4.5 Good / Bad 对比

```rust
// Good - use borrowed form to avoid ownership transfer
fn compute(a: &Tensor<f64, Ix2>, b: &Tensor<f64, Ix2>) -> Tensor<f64, Ix2> {
    a + b  // &Tensor + &Tensor -> new Tensor
}

// Good - use explicit API for broadcast safety
fn compute_safe(a: &Tensor<f64, Ix2>, b: &Tensor<f64, Ix1>) -> Result<Tensor<f64, Ix2>, XenonError> {
    let (a_bc, b_bc) = broadcast_with(&a.view(), &b.view())?;
    zip_with(&a_bc, &b_bc, |x, y| x + y)
}

// Bad - mixing owned and borrowed (unnecessarily consumes a)
fn compute_bad(a: Tensor<f64, Ix2>, b: &Tensor<f64, Ix2>) -> Tensor<f64, Ix2> {
    a + b  // a is consumed, cannot be used afterwards
}
```

---

## 5. 内部实现设计

### 5.1 委托模式

运算符重载的核心设计模式是 **委托**：

```
运算符语法 (arithmetic.rs)
     │
     │ 委托
     ▼
逐元素运算 (elementwise.rs)
     │
     │ 使用
     ▼
广播模块 (broadcast.rs) ── 内存访问 (storage)
```

运算符 `a + b` 展开为：
1. `broadcast_with(&a.view(), &b.view())` — 广播两个张量
2. `zip_with(&a_bc, &b_bc, |x, y| x + y)` — 逐元素运算

### 5.2 深拷贝保证

所有运算符产生的新张量是独立的：

- `zip_with` 分配新的 `Owned` 存储并逐元素写入
- 新张量与输入张量不共享内存
- `Tensor<A, D>` 类型保证所有权独占

### 5.3 标量路径优化

标量×张量运算使用 `mapv` 而非广播视图：

```
tensor + scalar:
    tensor.mapv(|x| x + scalar)

    优势:
    1. 无需创建广播视图
    2. mapv 内部直接迭代，编译器更容易内联和向量化
    3. 缓存友好（连续访问）
```

---

## 6. 实现任务拆分

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
  - 测试: `test_add_scalar`, `test_scalar_add_tensor`
  - 前置: T2
  - 预计: 10 min

### Wave 4: 其他运算符

- [ ] **T5**: 实现 `Sub`/`Mul`/`Div`（复制 `Add` 模式）
  - 文件: `src/overload/arithmetic.rs`
  - 内容: Sub/Mul/Div 所有组合
  - 测试: `test_sub`, `test_mul`, `test_div`
  - 前置: T3, T4
  - 预计: 15 min

### Wave 5: 测试

- [ ] **T6**: 编写综合测试
  - 文件: `tests/arithmetic.rs`
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

## 7. 测试计划

### 7.1 单元测试

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_add_same_shape` | `[2,3] + [2,3]`，逐元素验证 | 高 |
| `test_add_broadcast` | `[2,1,3] + [3]`，广播后相加 | 高 |
| `test_add_ref_ref` | `&a + &b`，所有权保留 | 高 |
| `test_add_owned_ref` | `a + &b`，a 被消费 | 中 |
| `test_add_ref_owned` | `&a + b`，b 被消费 | 中 |
| `test_add_scalar` | `tensor + 5.0` | 高 |
| `test_scalar_add_tensor` | `Scalar(5.0) + tensor` | 高 |
| `test_sub_basic` | `a - b` 正确性 | 高 |
| `test_mul_basic` | `a * b` 正确性 | 高 |
| `test_div_basic` | `a / b` 正确性 | 高 |
| `test_broadcast_incompatible` | 不兼容形状 panic | 中 |
| `test_result_ownership` | 结果张量与输入不共享内存 | 高 |
| `test_i32_tensor` | `i32` 类型张量运算 | 中 |
| `test_complex_tensor` | `Complex<f64>` 类型张量运算 | 中 |

### 7.2 边界测试

| 场景 | 预期行为 |
|------|----------|
| 0 维张量 + 0 维张量 | 标量加法 |
| 空张量 + 空张量 | 空张量结果 |
| `[1, 1000] + [1000, 1]` | 广播到 `[1000, 1000]` |
| 标量 + 0 维张量 | 正常运算 |
| 大张量 `[10000, 10000] + [10000, 10000]` | 正确完成 |

### 7.3 属性测试不变量

| 不变量 | 测试方法 |
|--------|----------|
| `(a + b).shape() == broadcast_shape(a.shape(), b.shape())` | 随机形状对 |
| `(&a + &b) == (a.clone() + b.clone())` | 借用与所有权结果一致 |
| `(a + scalar) == a.mapv(\|x\| x + scalar)` | 标量路径等价 |
| 结果张量与输入张量不共享内存（`ptr` 不同） | 指针比较 |

---

## 8. 与其他模块的交互

| 交互模块 | 方向 | 说明 |
|----------|------|------|
| `math` | arithmetic → math | `zip_with()` 执行逐元素运算，`mapv()` 执行标量运算（参见 `11-math.md` §4） |
| `broadcast` | arithmetic → broadcast | `broadcast_with()` 广播两个张量到公共形状（参见 `15-broadcast.md` §4） |
| `tensor` | arithmetic → tensor | 构造结果 `Tensor<A, D>`，使用 `.view()` 创建视图（参见 `07-tensor.md` §4） |
| `element` | arithmetic → element | `Numeric` trait 约束排除 `bool` 与 `usize` 类型（参见 `03-element.md` §3） |
| `dimension` | arithmetic → dimension | `BroadcastDim<E>::Output` 关联类型（参见 `02-dimension.md` §4） |

---

## 9. 设计决策记录

### 决策 1：是否支持 += 原地运算符

| 属性 | 值 |
|------|-----|
| 决策 | 当前版本不提供 `+=`/`-=`/`*=`/`/=` 原地运算符 |
| 理由 | 需求说明书 §20 明确"四则运算以外的运算符语法不在当前范围内"；原地运算符涉及 LHS 广播约束复杂 |
| 替代方案 | 提供 `AddAssign` 等 impl — 留待未来版本 |

### 决策 2：广播错误处理方式

| 属性 | 值 |
|------|-----|
| 决策 | 运算符重载中形状不兼容时 panic（使用 `expect`） |
| 理由 | 这是 Xenon 集中错误语义中的显式语法糖例外；运算符返回类型固定，无法返回 Result；用户需要安全路径可直接使用 `broadcast_with` + `zip_with`（参见 `15-broadcast.md` §4.1、`11-math.md` §4.1、`26-error.md` §4.7） |
| 替代方案 | 返回 `Result<Tensor, XenonError>` — 放弃，Rust 运算符 trait 不支持 Result 返回类型 |
| 替代方案 | 使用 `PartialEq` 运算符返回 `Result` — 放弃，不自然 |

> **补充**：方法型 API 仍然是错误恢复的主路径；运算符重载只是语法糖快捷入口，不改变 `math` 模块中 `Result` 风格接口的地位。

### 决策 3：标量路径使用 mapv 而非广播视图

| 属性 | 值 |
|------|-----|
| 决策 | 张量×标量运算使用 `mapv(|x| x op scalar)` 而非创建广播视图 |
| 理由 | 更高效（直接迭代 vs 间接寻址）；编译器更容易内联和向量化 `mapv`；缓存友好 |
| 替代方案 | 创建标量广播视图 `Tensor0::from(scalar).view().broadcast_to(shape)` — 放弃，间接寻址开销不必要 |

---

## 10. 性能考量

### 10.1 复杂度

| 操作 | 时间复杂度 | 空间复杂度 | 说明 |
|------|-----------|-----------|------|
| 张量 + 张量（同形状） | O(n) | O(n) | 无广播开销 |
| 张量 + 张量（广播） | O(output_n) | O(output_n) | 广播视图 O(1) 创建 |
| 张量 + 标量 | O(n) | O(n) | mapv 直接迭代 |
| 标量 + 张量 | O(n) | O(n) | mapv 直接迭代 |

### 10.2 性能数据（参考）

| 场景 | 路径 | 预计性能 |
|------|------|
| `[1000, 1000] + [1000, 1000]` (f64) | zip_with SIMD | ~1ms |
| `[1000, 1000] + [1, 1000]` (广播) | zip_with 广播 | ~1.2ms |
| `[1000, 1000] + 5.0` (标量) | mapv | ~0.8ms |
| `[1000, 1000] + [1000, 1000]` (f64, 非SIMD) | zip_with 标量 | ~4ms |

### 10.3 SIMD 路径

当 SIMD feature 启用时，`zip_with` 和 `mapv` 自动选择 SIMD 路径（参见 `08-simd.md` §4）：

| 运算符 | SIMD 指令 | 加速比 |
|--------|----------|--------|
| `+` (f32) | `AVX _mm256_add_ps` | 4-8x |
| `+` (f64) | `AVX _mm256_add_pd` | 2-4x |
| `*` (f32) | `AVX _mm256_mul_ps` | 4-8x |
| `/` (f64) | `AVX _mm256_div_pd` | 2-4x |

### 10.4 借用引用优化

```rust
// &a + &b: no ownership transfer, borrow only
// Internally: self.view() creates a lightweight view (O(1))
// Result: new Tensor allocation (O(n))

// a + b: a and b are consumed
// If a/b are not used afterwards, the owned form avoids explicit borrow overhead
// However, the & form is recommended to avoid accidental consumption
```

---

## 11. no_std 兼容性

运算符重载模块在 `no_std` 环境下可用，但需 `alloc` 支持以分配结果张量。运算符重载委托给逐元素运算和广播模块，其 `no_std` 兼容性参见对应文档。

```rust
#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
```

| 组件 | no_std 支持 | 说明 |
|------|:----------:|------|
| `Add` / `Sub` / `Mul` / `Div`（张量×张量） | ✅ | 委托 `zip_with` + `broadcast_with`，需 `no_std + alloc` |
| `Add` / `Sub` / `Mul` / `Div`（张量×标量） | ✅ | 委托 `mapv`，需 `no_std + alloc` |
| `Add` / `Sub` / `Mul` / `Div`（标量×张量） | ✅ | 委托 `mapv`，需 `no_std + alloc` |
| 借用形式（`&Tensor op &Tensor`） | ✅ | 创建视图（O(1)）+ 新 `Tensor`，需 `no_std + alloc` |
| 广播错误 panic | ✅ | `expect()` 使用 `core::panic`，无堆依赖 |

条件编译处理：

```rust
// Operator overloading delegates to:
//   - math::zip_with() → alloc (result Tensor)
//   - math::mapv()     → alloc (result Tensor)
//   - broadcast::broadcast_with() → alloc (SmallVec)

#[cfg(not(feature = "std"))]
extern crate alloc;

// 参见 `11-math.md` §11
// 参见 `15-broadcast.md` §11
```

---

## 版本历史

| 版本 | 日期 |
|------|------|
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-08 |
| 1.0.2 | 2026-04-08 |
| 1.0.3 | 2026-04-08 |
| 1.0.4 | 2026-04-08 |
| 1.1.0 | 2026-04-08 |
| 1.1.1 | 2026-04-10 |
| 1.1.2 | 2026-04-10 |
| 1.1.3 | 2026-04-10 |

---

*本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。*
