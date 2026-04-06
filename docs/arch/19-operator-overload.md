# 运算符重载模块设计

> 文档编号: 19 | 模块: `src/ops/arithmetic.rs` | 阶段: Phase 4
> 前置文档: `11-elementwise-ops.md`, `15-broadcast.md`
> 需求参考: 需求说明书 §20

---

## 1. 模块定位

### 1.1 职责边界

| 职责 | 包含 | 不包含 |
|------|------|--------|
| 四则运算运算符语法 | `+`/`-`/`*`/`÷` 运算符重载 | 原地运算符 `+=`/`-=`/`*=`/`/=`（当前版本不提供） |
| 张量×张量运算 | 同形状运算、广播运算 | 矩阵乘法（由 `matrix_ops` 提供） |
| 张量×标量运算 | 标量广播到张量形状 | 负数运算符 `-`（在 `elementwise-ops` 提供） |
| 广播支持 | 运算符语法内建支持广播 | 比较运算符（在 `elementwise-ops` 提供） |
| 新张量产生 | 所有组合产生新的独立张量 | 原地修改运算 |
| 借用形式 | `&Tensor op &Tensor`/`&Tensor op Tensor` 等组合 | 索引运算符 `[]`（在 `index` 提供） |

### 1.2 设计原则

| 原则 | 体现 |
|------|------|
| 委托模式 | 运算符重载委托给逐元素运算，运算符仅为语法糖 |
| 深拷贝结果 | 所有组合均产生新的独立张量，不共享内存 |
| 广播透明 | 运算符语法内建支持广播，用户无需手动处理 |
| 借用优先 | 鼓励使用 `&a + &b` 形式，避免不必要的所有权转移 |

---

## 2. 文件位置

```
src/ops/
├── arithmetic.rs       # 四则运算运算符重载
├── elementwise.rs      # 逐元素运算（运算符重载委托目标）
└── mod.rs              # 模块入口
```

运算符重载文件 `arithmetic.rs` 独立于逐元素运算 `elementwise.rs`，职责清晰：前者提供语法糖，后者提供计算能力。

---

## 3. 依赖关系

### 3.1 依赖图（ASCII）

```
                    ┌───────────────────┐
                    │ elementwise-ops    │
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
| `elementwise_ops` | `zip_with()`, `mapv()`, 二元逐元素运算 |
| `broadcast` | `broadcast_shape()`, `broadcast_with()`, `can_broadcast()` |
| `tensor` | `TensorBase<S, D>`, `Tensor<A, D>`, `TensorView`, `.view()` |
| `element` | `Numeric` trait 约束（排除 `bool`） |
| `dimension` | `Dimension`, `Ix0`~`Ix6`, `IxDyn` |

### 3.3 依赖方向声明

> **依赖方向：单向向上。** `arithmetic` 仅消费 `elementwise_ops`、`broadcast`、`tensor`、`element` 的 trait 和类型，不被它们依赖。`arithmetic` 是最上层的用户 API 模块。

---

## 4. 公共 API 设计

### 4.1 运算符 trait 实现矩阵

完整的 `impl` 组合表（以 `Add` 为例，`Sub`/`Mul`/`Div` 同理）：

| Lhs | Rhs | Output | 广播 | impl 签名 |
|-----|-----|--------|------|-----------|
| `Tensor<A, D>` | `Tensor<A, E>` | `Tensor<A, F>` | ✓ | `impl<...> Add<Tensor<A,E>> for Tensor<A,D>` |
| `&Tensor<A, D>` | `&Tensor<A, E>` | `Tensor<A, F>` | ✓ | `impl<...> Add<&Tensor<A,E>> for &Tensor<A,D>` |
| `Tensor<A, D>` | `&Tensor<A, E>` | `Tensor<A, F>` | ✓ | `impl<...> Add<&Tensor<A,E>> for Tensor<A,D>` |
| `&Tensor<A, D>` | `Tensor<A, E>` | `Tensor<A, F>` | ✓ | `impl<...> Add<Tensor<A,E>> for &Tensor<A,D>` |
| `Tensor<A, D>` | `A` | `Tensor<A, D>` | 标量广播 | `impl<...> Add<A> for Tensor<A,D>` |
| `&Tensor<A, D>` | `&A` | `Tensor<A, D>` | 标量广播 | `impl<...> Add<&A> for &Tensor<A,D>` |
| `A` | `Tensor<A, D>` | `Tensor<A, D>` | 标量广播 | `impl<...> Add<Tensor<A,D>> for A` |
| `A` | `&Tensor<A, D>` | `Tensor<A, D>` | 标量广播 | `impl<...> Add<&Tensor<A,D>> for A` |

> **说明**：`F` 为广播后的维度类型，由 `D::BroadcastDim<E>::Output` 关联类型计算。

### 4.2 张量×张量运算符

```rust
// 张量 + 张量（所有权 + 所有权）
impl<A, D, E> Add<Tensor<A, E>> for Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
    E: Dimension,
    D: BroadcastDim<E>,
{
    type Output = Tensor<A, <D as BroadcastDim<E>>::Output>;

    fn add(self, rhs: Tensor<A, E>) -> Self::Output {
        let (a_bc, b_bc) = broadcast_with(&self.view(), &rhs.view())
            .expect("broadcast: incompatible shapes");
        zip_with(&a_bc, &b_bc, |a, b| a + b)
    }
}

// &张量 + &张量（最常用形式）
impl<'a, 'b, A, D, E> Add<&'b Tensor<A, E>> for &'a Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
    E: Dimension,
    D: BroadcastDim<E>,
{
    type Output = Tensor<A, <D as BroadcastDim<E>>::Output>;

    fn add(self, rhs: &'b Tensor<A, E>) -> Self::Output {
        let (a_bc, b_bc) = broadcast_with(&self.view(), &rhs.view())
            .expect("broadcast: incompatible shapes");
        zip_with(&a_bc, &b_bc, |a, b| a + b)
    }
}
```

> **设计决策：** 形状不兼容时使用 `expect` panic。
> 这与 Rust 标准库 `Index` trait 的惯例一致：运算符重载 panic 而非返回 Result。
> 用户需要广播安全时可直接使用 `broadcast_with` + `zip_with`。

### 4.3 张量×标量运算符

```rust
// 张量 + 标量
impl<A, D> Add<A> for Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn add(self, rhs: A) -> Self::Output {
        self.mapv(|x| x + rhs)
    }
}

// &张量 + 标量
impl<'a, A, D> Add<A> for &'a Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn add(self, rhs: A) -> Self::Output {
        self.mapv(|x| x + rhs)
    }
}

// 标量 + 张量
impl<A, D> Add<Tensor<A, D>> for A
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn add(self, rhs: Tensor<A, D>) -> Self::Output {
        rhs.mapv(|x| self + x)
    }
}

// 标量 + &张量
impl<'a, A, D> Add<&'a Tensor<A, D>> for A
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn add(self, rhs: &'a Tensor<A, D>) -> Self::Output {
        rhs.mapv(|x| self + x)
    }
}
```

> **设计决策：** 标量运算使用 `mapv` 而非创建广播视图。
> 原因：`mapv` 直接迭代更高效，避免广播视图的间接寻址开销。

### 4.4 Sub / Mul / Div

`Sub`、`Mul`、`Div` 的实现模式与 `Add` 完全相同，仅替换运算符和对应闭包：

```rust
// Sub: |a, b| a - b
// Mul: |a, b| a * b
// Div: |a, b| a / b   (约束 A: Numeric + Div<Output = A>)
```

### 4.5 Good / Bad 对比

```rust
// Good - 使用借用形式避免所有权转移
fn compute(a: &Tensor<f64, Ix2>, b: &Tensor<f64, Ix2>) -> Tensor<f64, Ix2> {
    a + b  // &Tensor + &Tensor → 新 Tensor
}

// Good - 需要广播安全时使用显式 API
fn compute_safe(a: &Tensor<f64, Ix2>, b: &Tensor<f64, Ix1>) -> Result<Tensor<f64, Ix2>, BroadcastError> {
    let (a_bc, b_bc) = broadcast_with(&a.view(), &b.view())?;
    Ok(zip_with(&a_bc, &b_bc, |x, y| x + y))
}

// Bad - 混用所有权和借用（不必要地消费 a）
fn compute_bad(a: Tensor<f64, Ix2>, b: &Tensor<f64, Ix2>) -> Tensor<f64, Ix2> {
    a + b  // a 被消费，后续无法使用
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

- [ ] **T1**: 创建 `src/ops/arithmetic.rs` 骨架
  - 文件: `src/ops/arithmetic.rs`
  - 内容: 模块声明、导入
  - 测试: 编译通过
  - 前置: `elementwise-ops` 完成、`broadcast` 完成
  - 预计: 5 min

- [ ] **T2**: 实现 `Add` trait（张量×张量，所有权形式）
  - 文件: `src/ops/arithmetic.rs`
  - 内容: `Tensor + Tensor` impl
  - 测试: `test_add_same_shape`, `test_add_broadcast`
  - 前置: T1
  - 预计: 10 min

### Wave 2: 借用形式

- [ ] **T3**: 实现 `Add` trait（&张量×&张量、混合形式）
  - 文件: `src/ops/arithmetic.rs`
  - 内容: 4 种借用组合
  - 测试: `test_add_ref_ref`, `test_add_owned_ref`, `test_add_ref_owned`
  - 前置: T2
  - 预计: 10 min

### Wave 3: 标量运算符

- [ ] **T4**: 实现 `Add` trait（张量×标量、标量×张量）
  - 文件: `src/ops/arithmetic.rs`
  - 内容: 标量组合 impl
  - 测试: `test_add_scalar`, `test_scalar_add_tensor`
  - 前置: T2
  - 预计: 10 min

### Wave 4: 其他运算符

- [ ] **T5**: 实现 `Sub`/`Mul`/`Div`（复制 `Add` 模式）
  - 文件: `src/ops/arithmetic.rs`
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
| `test_scalar_add_tensor` | `5.0 + tensor` | 高 |
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
| `elementwise_ops` | arithmetic → elementwise | `zip_with()` 执行逐元素运算，`mapv()` 执行标量运算 |
| `broadcast` | arithmetic → broadcast | `broadcast_with()` 广播两个张量到公共形状 |
| `tensor` | arithmetic → tensor | 构造结果 `Tensor<A, D>`，使用 `.view()` 创建视图 |
| `element` | arithmetic → element | `Numeric` trait 约束排除 `bool` 类型 |
| `dimension` | arithmetic → dimension | `BroadcastDim<E>::Output` 关联类型 |

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
| 理由 | 与 Rust 标准库 `Index` trait 惯例一致；运算符返回类型固定，无法返回 Result；用户需要安全路径可直接使用 `broadcast_with` + `zip_with` |
| 替代方案 | 返回 `Result<Tensor, BroadcastError>` — 放弃，Rust 运算符 trait 不支持 Result 返回类型 |
| 替代方案 | 使用 `PartialEq` 运算符返回 `Result` — 放弃，不自然 |

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
|------|------|----------|
| `[1000, 1000] + [1000, 1000]` (f64) | zip_with SIMD | ~1ms |
| `[1000, 1000] + [1, 1000]` (广播) | zip_with 广播 | ~1.2ms |
| `[1000, 1000] + 5.0` (标量) | mapv | ~0.8ms |
| `[1000, 1000] + [1000, 1000]` (f64, 非SIMD) | zip_with 标量 | ~4ms |

### 10.3 SIMD 路径

当 SIMD feature 启用时，`zip_with` 和 `mapv` 自动选择 SIMD 路径：

| 运算符 | SIMD 指令 | 加速比 |
|--------|----------|--------|
| `+` (f32) | `AVX _mm256_add_ps` | 4-8x |
| `+` (f64) | `AVX _mm256_add_pd` | 2-4x |
| `*` (f32) | `AVX _mm256_mul_ps` | 4-8x |
| `/` (f64) | `AVX _mm256_div_pd` | 2-4x |

### 10.4 借用引用优化

```rust
// &a + &b: 无所有权转移，仅借用
// 运算符内部: self.view() 创建轻量视图（O(1)）
// 结果: 新 Tensor 分配（O(n)）

// a + b: a 和 b 被消费
// 如果 a/b 后续不再使用，所有权形式避免显式借用开销
// 但推荐使用 & 形式避免意外消费
```

---

*本文档由 Xenon 维护。如有问题请提交 Issue 或 PR。*
