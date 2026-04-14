# 集合操作模块设计

> 文档编号: 14 | 模块: `src/set/` | 阶段: Phase 4
> 前置文档: `03-element.md`, `04-complex.md`, `07-tensor.md`, `10-iterator.md`
> 需求参考: 需求说明书 §15
> 范围声明: 范围内

---

## 1. 模块定位

### 1.1 职责边界

| 职责        | 包含                                             | 不包含                        |
| ----------- | ------------------------------------------------ | ----------------------------- |
| unique 操作 | 返回不重复元素组成的新 1D 张量；结果顺序不作要求 | intersection/union/difference |
| 支持类型    | i32, i64, f32, f64, Complex<f32>, Complex<f64>   | bincount/histogram            |
| 不支持类型  | bool（仅 2 种值，unique 无意义）                 | argmin/argmax                 |

> **注意**：当前版本仅支持 unique 操作！不包含 intersection/union/difference/bincount/histogram 等。

### 1.2 设计原则

| 原则           | 体现                                                          |
| -------------- | ------------------------------------------------------------- |
| 最小范围       | 当前仅实现 unique，其他集合操作留待未来扩展                   |
| 类型安全       | bool 显式排除；仅对受支持的元素类型开放                       |
| 相等语义优先   | `unique` 仅基于逐元素相等关系去重，不承诺排序结果             |
| IEEE 754 一致  | `NaN != NaN`，因此每个 `NaN` 单独保留；`-0.0 == 0.0` 视为同值 |
| 复数按分量判等 | 复数去重按实部/虚部分别比较，并沿用对应实数语义               |

### 1.3 在架构中的位置

```
依赖层级：
L0: error, private
L1: dimension, element, complex
L2: layout (依赖 dimension)
L3: storage (仅依赖 core/alloc，不依赖 layout)
L4: tensor (依赖 storage, dimension)
L5: set  ← 当前模块
```

---

## 2. 需求映射与范围约束

| 类型     | 内容 |
| -------- | ---- |
| 需求映射 | 需求说明书 §15 |
| 范围内   | `unique()` 去重、NaN / `±0.0` 语义、复数按分量判等，以及 1D owned 结果构造。 |
| 范围外   | sort、unique counts、bincount、intersection / union / difference 等其他集合操作。 |
| 非目标   | 不引入排序契约、不新增第三方去重依赖，也不扩展到 histogram 类 API。 |

---

## 3. 文件位置

```
src/set/
├── mod.rs              # 模块入口
└── unique.rs           # 集合操作（本模块）
```

单文件设计理由：当前仅实现 unique，未来扩展时再拆分。

---

## 4. 依赖关系

### 4.1 依赖图

```
src/set/unique.rs
├── crate::tensor        # TensorBase<S, D>, Tensor<A, Ix1>
├── crate::element       # Element, ComplexScalar
├── crate::complex       # Complex<f32>, Complex<f64>
└── crate::iter          # Elements for collection
```

### 4.2 类型级依赖

| 来源模块  | 使用的类型/trait                                                                  |
| --------- | --------------------------------------------------------------------------------- |
| `tensor`  | `TensorBase<S, D>`, `Tensor<A, Ix1>`, `.iter()`, `.len()`，参见 `07-tensor.md` §5 |
| `storage` | `Storage` trait（consuming tensor storage for data access）                       |
| `dimension` | `Dimension`, `Ix1`（output dimension type for flatten result）                  |
| `element` | `Element`, `ComplexScalar`，参见 `03-element.md` §5.1 / §5.4                      |
| `element` | `Sealed`（sealed trait pattern for `UniqueElement`）                              |
| `complex` | `Complex<f32>`, `Complex<f64>`，参见 `04-complex.md` §5                           |
| `iter`    | `Elements`（遍历收集元素），参见 `10-iterator.md` §5.1                            |

### 4.3 依赖方向

> **依赖方向：单向向上。** `set` 仅消费 `tensor`、`element`、`complex`、`iter` 模块。

### 4.4 依赖合法性与替代方案

| 项目           | 说明 |
| -------------- | ---- |
| 新增第三方依赖 | 无 |
| 合法性结论     | 合法；当前设计仅复用 Xenon 既有模块、标准库以及文档中已声明的项目内可选能力。 |
| 替代方案       | 不适用；当前范围内无需额外第三方依赖。 |

---

## 5. 公共 API 设计

### 5.1 unique 操作

````rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: UniqueElement,  // UniqueElement: Element, correct constraint
{
    /// Returns unique elements; order is unspecified and may vary between calls.
    ///
    /// # Supported types
    ///
/// i32, i64, f32, f64, Complex<f32>, Complex<f64>
    ///
/// # Unsupported types
///
/// - bool: only 2 values (true/false), unique is meaningless for bool
    ///
    /// # Equality behavior
    ///
    /// Each NaN is preserved because `NaN != NaN`, while `-0.0` and `0.0`
    /// are treated as equal.
    ///
    /// # Complex equality rule
    ///
    /// Complex values are compared component-wise using the corresponding
    /// real-number equality semantics.
    ///
    /// # Empty array behavior
    ///
    /// Empty array returns an empty Tensor1.
    ///
    /// # Complexity
    ///
    /// Implementation-defined; external semantics do not depend on result order.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = Tensor1::from_vec(vec![3, 1, 2, 1, 3, 2]);
    /// let u = a.unique();
    /// assert_eq!(u.len(), 3);
    /// assert!(u.iter().all(|x| [1, 2, 3].contains(x)));
    ///
    /// // empty
    /// let empty: Tensor1<i32> = Tensor1::zeros([0]);
    /// assert_eq!(empty.unique().len(), 0);
    /// ```
    pub fn unique(&self) -> Tensor<A, Ix1>;
}
````

### 5.2 Good / Bad 对比示例

```rust
// Good - use unique to get unique elements with unspecified order
let a = Tensor1::from_vec(vec![3, 1, 2, 1, 3]);
let u = a.unique();
assert_eq!(u.len(), 3);

// Good - empty array returns empty Tensor1
let empty: Tensor1<i32> = Tensor1::zeros([0]);
assert_eq!(empty.unique().len(), 0);

// Bad - calling unique on a bool tensor (compile error)
// let b = Tensor1::from_vec(vec![true, false, true]);
// b.unique();  // compile error: bool does not implement UniqueElement trait

```

---

## 6. 内部实现设计

### 6.1 unique 实现步骤

```
unique(self):
    1. 收集所有元素到 Vec<A>
    2. 逐个检查元素是否已在结果集中出现
    3. 若未出现，则追加到结果；若已出现，则跳过
    4. 从 Vec 构造 Tensor<A, Ix1>
```

### 6.2 浮点判等处理

```
- 对非 NaN 浮点值，按 Rust / IEEE 754 的 `==` 语义判等
- `NaN != NaN`，因此输入中的每个 `NaN` 都须单独保留，不参与去重
- `+0.0 == -0.0`，因此两者视为同一个唯一值
- 文档只约束相等关系，不约束内部使用哈希、线性扫描或其它去重实现
```

### 6.3 复数判等规则

```
复数判等策略（component-wise equality）：
- 两个复数相等，当且仅当 `re` 与 `im` 两个分量分别相等
- 分量比较沿用对应实数语义：`NaN != NaN`，`-0.0 == 0.0`
- 因此，带有 NaN 分量的复数不会因为“都是 NaN”而被去重
- 文档不定义任何 lexicographic order、模排序或其它排序关系
```

### 6.4 类型排除实现

```rust
/// Trait for types that support the unique operation.
///
/// Provides the equality semantics required by `unique`.
/// `bool` does not implement this trait.
///
/// Note: `Ord` is intentionally not required because `unique` does not
/// define or expose any sorting contract.
///
/// # Why in set/unique.rs, not element module?
///
/// UniqueElement is defined here rather than in the element module because
/// its semantic (equality for deduplication) is operation-specific, not a
/// fundamental element property. This avoids making the element module depend
/// on `unique`-specific rules.
///
/// # Sealing
///
/// Directly inherits Sealed for defensive sealing, even though Element
/// already provides it. This makes the sealing intent explicit at the
/// trait definition site.
pub trait UniqueElement: Element + Sealed {
    /// Equality check used by `unique`.
    fn unique_eq(&self, other: &Self) -> bool;
}

impl UniqueElement for i32 {
    fn unique_eq(&self, other: &Self) -> bool { self == other }
}

impl UniqueElement for i64 {
    fn unique_eq(&self, other: &Self) -> bool { self == other }
}

impl UniqueElement for f32 {
    fn unique_eq(&self, other: &Self) -> bool { self == other }
}

impl UniqueElement for f64 {
    fn unique_eq(&self, other: &Self) -> bool { self == other }
}

impl UniqueElement for Complex<f32> {
    fn unique_eq(&self, other: &Self) -> bool {
        self.re == other.re && self.im == other.im
    }
}

impl UniqueElement for Complex<f64> {
    fn unique_eq(&self, other: &Self) -> bool {
        self.re == other.re && self.im == other.im
    }
}
// bool does not implement this
```

---

## 7. 实现任务拆分

### Wave 1: 基础设施

- [ ] **T1**: 创建 `src/set/unique.rs` 骨架
  - 文件: `src/set/unique.rs`
  - 内容: 模块声明、UniqueElement trait 定义
  - 测试: 编译通过
  - 前置: `07-tensor.md` 完成
  - 预计: 5 min

### Wave 2: 核心实现

- [ ] **T2**: 实现 `unique` 方法
  - 文件: `src/set/unique.rs`
  - 内容: 元素收集、基于相等关系去重、Tensor 构造
  - 测试: `test_unique_basic`, `test_unique_empty`, `test_unique_single`, `test_unique_duplicates`
  - 前置: T1
  - 预计: 10 min

### Wave 3: 浮点与复数扩展

- [ ] **T3**: 实现浮点 NaN / `±0.0` 判等处理
  - 文件: `src/set/unique.rs`
  - 内容: 保留每个 `NaN`，并将 `-0.0` 与 `0.0` 视为同值
  - 测试: `test_unique_nan_preserved_f32`, `test_unique_signed_zero_equal_f32`
  - 前置: T2
  - 预计: 10 min

- [ ] **T4**: 实现复数按分量判等规则
  - 文件: `src/set/unique.rs`
  - 内容: 实部和虚部分别沿用对应实数语义，不引入排序语义
  - 测试: `test_unique_complex_componentwise`
  - 前置: T2
  - 预计: 10 min

### Wave 4: TensorBase 入口集成

- [ ] **T5**: 在 TensorBase 上添加 `unique()` 入口方法
  - 文件: `src/set/unique.rs`（或 trait extension）
  - 内容: `unique()` 方法绑定到 TensorBase
  - 测试: `test_unique_integration`
  - 前置: T2, T3, T4
  - 预计: 5 min

### 并行执行分组图

```
Wave 1: [T1]
           │
Wave 2: [T2]
         │ │
Wave 3: [T3] [T4]
           │
Wave 4: [T5]
```

---

## 8. 测试计划

### 8.1 测试分类表

| 测试分类 | 位置                     | 说明                                                            |
| -------- | ------------------------ | --------------------------------------------------------------- |
| 单元测试 | `#[cfg(test)] mod tests` | 验证 `unique()` 的去重语义、顺序非承诺与类型特例                |
| 集成测试 | `tests/`                 | 验证 `set` 与 `tensor`、`iter`、`element`、`complex` 的协同路径 |
| 边界测试 | 同模块测试中标注         | 覆盖空张量、单元素、NaN、`±0.0` 与复数分量判等等边界            |
| 属性测试 | `tests/property/`        | 验证结果无重复、元素集合与输入等价等不变量                      |

### 8.2 单元测试清单

| 测试函数                            | 测试内容                                     | 优先级 |
| ----------------------------------- | -------------------------------------------- | ------ |
| `test_unique_basic_i32`             | 噪声去除后结果包含且仅包含 1/2/3；不要求顺序 | 高     |
| `test_unique_basic_i64`             | i64 类型正确性                               | 高     |
| `test_unique_basic_f32`             | f32 类型正确性                               | 高     |
| `test_unique_basic_f64`             | f64 类型正确性                               | 高     |
| `test_unique_basic_complex`         | Complex<f64> 类型正确性                      | 高     |
| `test_unique_empty`                 | 空数组返回空 Tensor1                         | 高     |
| `test_unique_single`                | 单元素返回自身                               | 中     |
| `test_unique_all_same`              | 所有元素相同返回单元素                       | 中     |
| `test_unique_nan_preserved_f32`     | 每个 f32 NaN 都被保留                        | 高     |
| `test_unique_nan_preserved_f64`     | 每个 f64 NaN 都被保留                        | 高     |
| `test_unique_signed_zero_equal_f32` | `-0.0` 与 `0.0` 视为同值                     | 高     |
| `test_unique_signed_zero_equal_f64` | `-0.0` 与 `0.0` 视为同值                     | 高     |
| `test_unique_complex_componentwise` | 复数按分量判等并沿用实数语义                 | 高     |
| `test_unique_2d`                    | 2D 张量 unique 返回 1D                       | 中     |
| `test_unique_order_unspecified`     | 同一输入仅验证集合语义，不断言固定顺序       | 中     |

### 8.3 边界测试场景

| 场景                         | 预期行为                                   |
| ---------------------------- | ------------------------------------------ |
| 空张量 `shape=[0]`           | 返回空 Tensor1                             |
| 单元素 `[42]`                | 返回单元素结果                             |
| 全部相同 `[5, 5, 5]`         | 返回单个 `5`                               |
| NaN + 实数 `[1.0, NaN, 2.0]` | 返回长度为 3 的结果，且该 NaN 被保留       |
| 多个 NaN `[NaN, NaN]`        | 返回长度为 2 的结果                        |
| `[-0.0, 0.0]`                | 返回长度为 1 的结果                        |
| 复数 `[1+NaNi, 1+NaNi]`      | 返回长度为 2 的结果（因为 NaN 分量不相等） |

### 8.4 属性测试不变量

| 不变量                                  | 测试方法                             |
| --------------------------------------- | ------------------------------------ |
| 输出无重复（按 `unique_eq` 定义）       | 任意两个保留元素都不满足 `unique_eq` |
| 非 NaN 输入时输出元素集合与输入集合相同 | 以参考集合语义对比                   |
| NaN 元素按出现次数保留                  | 统计输入/输出中的 NaN 数量并比较     |
| 多维输入始终返回 1D 结果                | 随机 2D/3D 形状输入                  |

### 8.5 集成测试

| 测试文件            | 测试内容                                                                           |
| ------------------- | ---------------------------------------------------------------------------------- |
| `tests/test_set.rs` | `unique()` 与 `tensor`、`iter`、`element`、`complex`、`alloc` 路径的端到端协同验证 |

### 8.6 Feature gate / 配置测试

| 配置 | 验证点 |
| ---- | ---- |
| 默认配置 | `unique()` 在默认构建下保持 NaN 保留、`-0.0 == 0.0` 与顺序非承诺语义。 |
| 其他 feature 组合 | 不适用；当前模块无额外 feature gate。 |

### 8.7 类型边界 / 编译期测试

| 场景 | 测试方式 |
| ---- | ---- |
| `bool` 不实现 `UniqueElement` | 编译期测试。 |
| 多维输入统一返回 `Tensor<A, Ix1>` | 编译期签名检查与运行时 shape 断言。 |
| sort / bincount / unique counts 不属于当前 API | API 缺失断言。 |

---

## 9. 与其他模块的交互

### 9.1 接口约定

| 方向            | 对方模块  | 接口/类型                             | 约定                                                                   |
| --------------- | --------- | ------------------------------------- | ---------------------------------------------------------------------- |
| `set → tensor`  | `tensor`  | `TensorBase<S, D>` / `Tensor<A, Ix1>` | 消费输入张量并返回 1D owned 结果，参见 `07-tensor.md` §5               |
| `set → iter`    | `iter`    | `Elements`                            | 使用元素迭代器收集逻辑元素，参见 `10-iterator.md` §5.1                 |
| `set → element` | `element` | `Element`, `ComplexScalar`            | 复用元素类型边界与复数标量语义，参见 `03-element.md` §5.1 / §5.4       |
| `set → set`     | `set`     | `UniqueElement`                       | `UniqueElement` 定义在 `src/set/unique.rs`，通过 `unique_eq` 约束去重语义 |

### 9.2 数据流描述

```text
User calls unique()
    │
    ├── set collects logical elements through iter
    ├── UniqueElement::unique_eq drives deduplication semantics
    ├── complex inputs reuse component-wise equality rules
    └── the module builds a new owned 1D tensor for the result
```

---

## 10. 错误处理与语义边界

| 主题 | 内容 |
| ---- | ---- |
| Recoverable error | 不适用；当前 `unique()` API 直接返回结果张量，不暴露模块级 `Result` 错误路径。 |
| Panic | 不适用；除分配失败等通用运行时故障外，模块不定义额外 panic 语义。 |
| 路径一致性 | 当前仅有单一路径；无 SIMD / 并行分支，外部语义必须只由 `unique_eq` 决定。 |
| 容差边界 | 不适用。 |

---

## 11. 设计决策记录

### 决策 1：bool 排除理由

| 属性     | 值                                                                                                                                                               |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 决策     | unique 不支持 bool 类型                                                                                                                                          |
| 理由     | bool 只有 true/false 两种值，unique 对 bool 无意义；用户可直接用 `== true` / `== false` 判断；且 bool 的 unique 语义不明确（是返回 [false, true] 还是 [0, 1]？） |
| 替代方案 | 支持 bool unique，返回 [false, true]                                                                                                                             |
| 拒绝原因 | 增加维护负担，收益几乎为零；需求说明书 §15 "bool 不适用"                                                                                                         |

### 决策 2：NaN 处理策略

| 属性     | 值                                                                      |
| -------- | ----------------------------------------------------------------------- |
| 决策     | `unique` 严格沿用 IEEE 754 / Rust 相等语义：`NaN != NaN`，`-0.0 == 0.0` |
| 理由     | 直接满足 `require.md §15`，避免文档额外发明“canonical NaN”语义          |
| 替代方案 | 归并全部 NaN                                                            |
| 替代方案 | 把 `-0.0` 与 `0.0` 视为不同值                                           |
| 拒绝原因 | 均与需求说明书冲突                                                      |

### 决策 3：复数按分量判等

| 属性     | 值                                                                    |
| -------- | --------------------------------------------------------------------- |
| 决策     | 复数去重仅按实部与虚部逐分量判等                                      |
| 理由     | `require.md §15` 只要求 component-wise equality，并未授权任何排序语义 |
| 替代方案 | lexicographic order                                                   |
| 拒绝原因 | 会把排序错误地写入公开契约，并掩盖 NaN 分量应逐个保留的要求           |

---

## 12. 性能考量

### 12.1 复杂度

- 对外语义不承诺具体算法复杂度
- 参考实现可采用线性扫描去重（O(N^2)），也可采用不改变外部语义的辅助索引结构优化
- 无论内部实现如何，结果顺序都不是稳定契约的一部分

### 12.2 内存开销

- 收集元素: O(N) 临时 Vec
- 去重辅助状态: 取决于具体实现，可为 O(1) 到 O(N)
- 结果: O(U) 其中 U 为保留后的元素数量（含每个被保留的 NaN）

### 12.3 大数组性能（参考）

| 说明       | 内容                                                         |
| ---------- | ------------------------------------------------------------ |
| 稳定契约   | 不对结果顺序和实现路径做性能承诺                             |
| 实现自由度 | 可在满足 `require.md §15` 的前提下选择线性扫描或辅助索引结构 |

---

## 13. 平台与工程约束

集合操作模块须遵循项目统一工程约束，不单独定义 `no_std` 方案：

- 仅支持 `std` 环境（参见 `require.md §1.3`）
- 保持单 crate 结构
- 遵循 SemVer
- 不引入超出项目基线的第三方依赖

| 项目       | 约束                                              |
| ---------- | ------------------------------------------------- |
| 平台       | 仅 `std`                                          |
| crate 结构 | 单 crate                                          |
| 依赖       | 不新增第三方依赖                                  |
| 语义一致性 | SIMD / 并行等执行路径不得改变 `unique` 的外部语义 |

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
| 1.1.1 | 2026-04-10 |
| 1.1.2 | 2026-04-14 |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
