# 集合操作模块设计

> 文档编号: 14
> 模块目录: src/set/
> 任务阶段: Phase 4
> 前置文档: 03-element.md, 04-complex.md, 07-tensor.md, 10-iterator.md

---

## 1. 模块定位

### 1.1 职责边界

| 职责         | 包含                                                          |
| ------------ | ------------------------------------------------------------- |
| 集合操作     | unique: 返回不重复元素组成的新 1D 张量；返回顺序不作 API 契约 |
| 支持类型     | i32, i64, f32, f64, Complex<f32>, Complex<f64>                |

| 职责         | 不包含                                              |
| ------------ | --------------------------------------------------- |
| 集合操作     | intersection / union / difference                   |
| 统计操作     | bincount / histogram                                |
| 归约索引     | argmin / argmax                                     |
| 支持类型     | `需求说明书 §15` 明确将 bool 排除在 `unique` 之外   |

### 1.2 设计原则

| 原则           | 体现                                                          |
| -------------- | ------------------------------------------------------------- |
| 最小范围       | 当前仅实现 unique，其他集合操作留待未来扩展                   |
| 类型安全       | bool 显式排除；仅对受支持的元素类型开放                       |
| 相等语义优先   | `unique` 基于逐元素相等关系去重，不承诺任何排序结果           |
| 顺序未定义     | 输出顺序不作 API 契约——允许同输入下结果顺序不稳定             |
| IEEE 754 一致  | `NaN != NaN`，因此每个 `NaN` 单独保留；`-0.0 == 0.0` 视为同值 |
| 复数按分量判等 | 复数去重按实部/虚部分别比较，并沿用对应实数语义               |

---

## 2. 需求映射与范围约束

| 类型     | 内容                                                                              |
| -------- | --------------------------------------------------------------------------------- |
| 需求映射 | 需求说明书 §4、§15、§28                                                           |
| 范围内   | `unique()` 去重、NaN / `±0.0` 语义、复数按分量判等，以及 1D owned 结果构造。      |
| 范围外   | sort、unique counts、bincount、intersection / union / difference 等其他集合操作。 |
| 非目标   | 不引入排序契约、不新增第三方去重依赖，也不扩展到 histogram 类 API。               |

---

## 3. 文件位置

```
src/set/
├── mod.rs              # module entry
└── unique.rs           # set operations (this module)
```

由 `mod.rs` 承担模块入口、`unique.rs` 承担唯一公开集合操作实现，保持导出边界与语义实现分离。

---

## 4. 依赖关系

### 4.1 依赖图

```
src/set/unique.rs
├── crate::tensor        # TensorBase<S, D>, Tensor<A, Ix1>
├── crate::storage       # Storage
├── crate::dimension     # Dimension, Ix1
├── crate::element       # Element (Copy supertrait)
├── crate::complex       # Complex<f32>, Complex<f64> (concrete UniqueElement impls)
└── crate::iter          # Iter for collection
```

### 4.2 类型级依赖

| 来源模块    | 使用的类型/trait                                                                   |
| ----------- | ---------------------------------------------------------------------------------- |
| `tensor`    | `TensorBase<S, D>`, `Tensor<A, Ix1>`, `.iter()`, `.len()`，参见 `07-tensor.md` §5  |
| `storage`   | `Storage<Elem = A>` trait（read-only element access via `Storage<Elem = A>`）      |
| `dimension` | `Dimension`, `Ix1`（output dimension type for flatten result）                     |
| `element`   | `Element`，参见 `03-element.md` §5.1                                               |
| `complex`   | `Complex<f32>`, `Complex<f64>`，参见 `04-complex.md` §5                            |
| `iter`      | `Iter`（遍历收集元素），参见 `10-iterator.md` §5.1                             |

### 4.3 依赖合法性

| 项目           | 说明                                                                          |
| -------------- | ----------------------------------------------------------------------------- |
| 新增第三方依赖 | 无                                                                            |
| 合法性结论     | 合法；当前设计仅复用 Xenon 既有模块、标准库以及文档中已声明的项目内可选能力。 |
| 替代方案       | 不适用；当前范围内无需额外第三方依赖。                                        |

### 4.4 依赖方向

依赖方向：单向向上。`set` 仅消费 `tensor`、`storage`、`dimension`、`element`、`complex`、`iter` 模块。

---

## 5. 公共 API 设计

### 5.1 unique 操作

````rust,ignore
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: UniqueElement,
{
    /// Returns unique elements as a 1D owned tensor.
    ///
    /// # Output ordering contract
    ///
    /// Implementations MAY internally choose a deterministic order (such as
    /// F-order first-occurrence) for performance or debugging convenience,
    /// but that choice is an internal implementation detail and is NOT part
    /// of the public API contract — see decision 4. Tests MUST assert
    /// set-equality, not vector-equality.
    ///
    /// # Supported types
    ///
    /// i32, i64, f32, f64, Complex<f32>, Complex<f64>
    ///
    /// # Unsupported types
    ///
    /// - bool: `false` and `true` are still distinct values, but `requirements specification §15`
    ///   explicitly excludes bool from the current `unique` contract
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
    /// Empty array returns an empty `Tensor<A, Ix1>`.
    ///
    /// # Multi-dimensional input
    ///
    /// For inputs of any dimension, `unique()` logically flattens all elements
    /// into a 1D sequence (in F-order) before deduplication. The output is always
    /// a 1D tensor (`Tensor<A, Ix1>`) with owned contiguous F-order storage;
    /// element order within the output is unspecified.
    ///
    /// # Trait bound rationale
    ///
    /// `UniqueElement: Element`, and `Element: Copy` (see `03-element.md §5.1`),
    /// so the implementation can collect / clone elements freely. The output
    /// `Tensor<A, Ix1>` is constructed by allocating an owned F-order buffer
    /// and copying the retained representatives by value.
    ///
    /// # Complexity
    ///
    /// Implementation-defined. Reference implementations may use linear scan
    /// for small inputs (O(N²)); large inputs SHOULD use a hash-aided structure
    /// (close to O(N) amortized) per §6.5. External semantics do not depend on
    /// the chosen strategy: deduplication results (as a multiset) are identical
    /// across strategies; element order is unspecified regardless.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use std::collections::HashSet;
    /// let a = Tensor::<i32, Ix1>::from_shape_vec(Ix1(6), vec![3, 1, 2, 1, 3, 2])?;
    /// let u = a.unique();
    /// // Order is unspecified — assert as a set, not as a vector.
    /// let got: HashSet<i32> = u.iter().copied().collect();
    /// let want: HashSet<i32> = [3, 1, 2].into_iter().collect();
    /// assert_eq!(got, want);
    ///
    /// // empty
    /// let empty: Tensor<i32, Ix1> = Tensor::zeros([0])?;
    /// assert_eq!(empty.unique().len(), 0);
    /// ```
    pub fn unique(&self) -> Tensor<A, Ix1>;
}
````

### 5.2 Good / Bad 对比示例

```rust,ignore
// Good - use unique; output order is unspecified — assert set membership, not order.
use std::collections::HashSet;
let a = Tensor::<i32, Ix1>::from_shape_vec(Ix1(5), vec![3, 1, 2, 1, 3])?;
let u = a.unique();
assert_eq!(u.len(), 3);
let got: HashSet<i32> = u.iter().copied().collect();
let want: HashSet<i32> = [1, 2, 3].into_iter().collect();
assert_eq!(got, want);

// Bad - relying on a specific element order in the output
// assert_eq!(u.iter().copied().collect::<Vec<_>>(), vec![3, 1, 2]); // BUG: order not contractually guaranteed

// Good - empty array returns empty `Tensor<A, Ix1>`
let empty: Tensor<i32, Ix1> = Tensor::zeros([0])?;
assert_eq!(empty.unique().len(), 0);

// Bad - calling unique on a bool tensor (compile error)
// let b = Tensor::<bool, Ix1>::from_shape_vec(Ix1(3), vec![true, false, true])?;
// b.unique();  // compile error: bool does not implement UniqueElement trait
```

---

## 6. 内部实现设计

### 6.1 unique 实现步骤

```
unique(self):
    1. Iterate logical elements of `self` in F-order (column-major). The
       iteration order is an implementation choice — not part of the API
       contract.
    2. For each element x, decide whether x already has a representative in
       the output set:
         - For non-NaN values: equivalent under `unique_eq`.
         - For NaN values (per IEEE 754): every NaN is its own class; do not
           merge with any prior NaN. Keep all NaN instances.
       Use linear scan for small inputs, or hash-aided lookup for large inputs
       (key construction rules in §6.1.1 below).
    3. If x is the first occurrence of its class (or x is NaN), append x to
       the output sequence.
    4. Construct `Tensor<A, Ix1>` from the appended sequence (owned F-order
       contiguous buffer).

Ordering contract (decision 4, post v2.0.2):
    - Output element order is UNSPECIFIED. Two calls on the same input MAY
      produce outputs in different orders.
    - The reference implementation above happens to be deterministic (driven
      by input F-order iteration), but that is an implementation detail, not
      a contract. Future implementations MAY use, for example, a HashMap
      iteration order as the output sequence (which is randomized per process
      under Rust's default hasher).
    - Implementations are free to choose any output order, including one that
      varies between calls; the only multiset-level contract is that the
      output is the deduplication of input under `unique_eq`, with NaN values
      preserved per IEEE 754.
```

对 `f32` / `f64` 及 `Complex<f32>` / `Complex<f64>` 的 `unique` 实现，不得直接依赖标准 Rust `Hash` / `Eq` 语义，也不得直接建立在 `BTreeSet` / `HashSet` 这类标准集合之上；必须使用线性扫描或自定义哈希键策略，以严格满足本文档定义的判等规则：

 1. `NaN != NaN`，因此每个 `NaN` 都必须单独保留，不能因为"同为 NaN"而被合并。
 2. `-0.0 == 0.0`，因此两者必须视为同一个 unique 值。
 3. 复数按分量比较，且每个分量分别沿用对应实数的上述语义。
 4. 若实现采用哈希优化，键规范固定如下：
   - NaN 元素不进入普通去重键路径。
   - 实现须对 NaN 单独旁路处理，保证输入中的每个 NaN（无论位模式是否相同）均被保留。
   - 普通哈希键仅用于非 NaN 元素。
   - `i32` / `i64` 直接以数值作为键。
   - `f32` / `f64` 对所有 `+0.0` / `-0.0` 归一到同一键。
   - `Complex<T>` 的键为 `(re_key, im_key)`。

### 6.2 浮点判等处理

- 非 NaN 浮点值的相等判定遵循 Rust / IEEE 754 `==` 语义
- `NaN != NaN`，因此输入中的每个 `NaN` 必须独立保留，不参与去重
- `+0.0 == -0.0`，因此两者视为同一个 unique 值
- 输出顺序不构成 API 契约。不限制实现使用哈希、线性扫描或其他查重策略

### 6.3 复数判等规则

- 两个复数相等当且仅当 `re` 和 `im` 分量分别相等
- 分量比较沿用对应实数语义：`NaN != NaN`，`-0.0 == 0.0`
- 因此，含 NaN 分量的复数不会仅因为"都是 NaN"而被去重合并
- 本文档不定义任何字典序、模长序或其他排序关系

### 6.4 类型排除实现

```rust,ignore
/// Trait for types that support the unique operation.
///
/// Provides the equality semantics required by `unique`.
/// `bool` does not implement this trait.
///
/// `UniqueElement` is a sealed trait. It reuses the shared `crate::private::Sealed`
/// infrastructure (defined in `03-element.md §5.7`), consistent with all other
/// public element capability traits. It is implemented only inside this crate for 
/// supported element types, so the closed element set is preserved.
pub trait UniqueElement: crate::private::Sealed + Element {
    /// Equality check used by `unique`.
    fn unique_eq(&self, other: &Self) -> bool;
}

// No local `mod private` needed — `UniqueElement` reuses the shared
// `crate::private::Sealed` already implemented for all seven element types
// (i32, i64, f32, f64, Complex<f32>, Complex<f64>, bool).
// `UniqueElement` is simply not implemented for `bool`, which excludes it
// at compile time without requiring a separate sealing mechanism.

impl UniqueElement for i32 {
    fn unique_eq(&self, other: &Self) -> bool { self == other }
}

impl UniqueElement for i64 {
    fn unique_eq(&self, other: &Self) -> bool { self == other }
}

// IEEE 754 == semantics:
// - NaN != NaN: each NaN is preserved as distinct (no merging)
// - +0.0 == -0.0: treated as equal (single output)
impl UniqueElement for f32 {
    fn unique_eq(&self, other: &Self) -> bool { self == other }
}

// IEEE 754 == semantics (same as f32 above):
// - NaN != NaN: each NaN is preserved as distinct (no merging)
// - +0.0 == -0.0: treated as equal (single output)
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

### 6.5 推荐实现策略

| 场景              | 推荐策略          | 说明                                                                      |
| ----------------- | ----------------- | ------------------------------------------------------------------------- |
| 小输入或原型实现 | 线性扫描 | 直接复用 `unique_eq`，最坏 O(N²)；不引入额外内存分配，常数项小，对短输入更快。|
| 大输入主路径 | 哈希查重 | 用哈希表作为查重索引，重复检测降到近似 O(N) 摊销。可用 `Vec<A>` 按输入迭代顺序追加，也可直接以 `HashMap` iteration 序输出，由实现自行选择。|
| 浮点/复数特殊值 | 专门分支处理 | `NaN != NaN`，因此哈希或索引策略也必须显式保留每个 `NaN`，不得把它们合并。|

**何时必须用哈希路径**：当输入规模导致线性扫描的 O(N²) 内存或 CPU 成本不可接受时，必须切换到哈希路径以避免大张量上的不可接受性能。

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

---

## 8. 测试计划

### 8.1 测试分类表

| 测试分类 | 位置                     | 说明                                                            |
| -------- | ------------------------ | --------------------------------------------------------------- |
| 单元测试 | `#[cfg(test)] mod tests` | 验证 `unique()` 的去重语义、首次出现顺序契约与类型特例          |
| 集成测试 | `tests/`                 | 验证 `set` 与 `tensor`、`iter`、`element`、`complex` 的协同路径 |
| 边界测试 | 同模块测试中标注         | 覆盖空张量、单元素、NaN、`±0.0` 与复数分量判等等边界            |
| 属性测试 | `tests/property/`        | 验证结果无重复、元素集合与输入等价等不变量                      |

### 8.2 单元测试清单

| 测试函数                                    | 测试内容                                     | 优先级 |
| ------------------------------------------- | -------------------------------------------- | ------ |
| `test_unique_basic_i32`                     | 输入 `[3,1,2,1,3,2]` 输出按 multiset 等于 `{1, 2, 3}`（顺序未定义） | 高     |
| `test_unique_basic_i64`                     | i64 类型正确性                               | 高     |
| `test_unique_basic_f32`                     | f32 类型正确性                               | 高     |
| `test_unique_basic_f64`                     | f64 类型正确性                               | 高     |
| `test_unique_basic_complex`                 | Complex<f64> 类型正确性                      | 高     |
| `test_unique_empty`                         | 空数组返回空 `Tensor<A, Ix1>`                | 高     |
| `test_unique_single`                        | 单元素返回自身                               | 中     |
| `test_unique_all_same`                      | 所有元素相同返回单元素                       | 中     |
| `test_unique_nan_preserved_f32`             | 每个 f32 NaN 都被保留                        | 高     |
| `test_unique_nan_preserved_f64`             | 每个 f64 NaN 都被保留                        | 高     |
| `test_unique_signed_zero_equal_f32`         | `-0.0` 与 `0.0` 视为同值                     | 高     |
| `test_unique_signed_zero_equal_f64`         | `-0.0` 与 `0.0` 视为同值                     | 高     |
| `test_unique_complex_componentwise`         | 复数按分量判等并沿用实数语义                 | 高     |
| `test_unique_2d`                            | 2D 张量 unique 返回 1D                       | 中     |
| `test_unique_non_contiguous_view`           | 切片视图输入仍按逻辑元素去重                 | 高     |
| `test_unique_transposed_view`               | 转置视图输入仍按逻辑元素去重                 | 高     |
| `test_unique_padded_tensor_ignores_padding` | padding 区域不应暴露到 unique 语义中         | 高     |
| `test_unique_order_unspecified`             | 文档化：测试不依赖顺序——仅验证 multiset 相等 | 高     |
| `test_unique_set_equality`                  | 输入与输出按 multiset 语义相等（不依赖顺序） | 高     |
| `test_unique_large_tensor_high_dup`         | `10^7` 元素高重复输入主路径保持正确          | 中     |
| `test_unique_high_rank_ixdyn`               | `IxDyn` rank 5+ 输入仍统一展平到 1D          | 中     |
| `test_unique_extreme_i64_values`            | `i32` / `i64` 极值去重语义正确               | 中     |

### 8.3 边界测试场景

| 场景                                              | 预期行为                                                   |
| ------------------------------------------------- | ---------------------------------------------------------- |
| 空张量 `shape=[0]`                                | 返回空 `Tensor<A, Ix1>`                                    |
| 单元素 `[42]`                                     | 返回单元素结果                                             |
| 全部相同 `[5, 5, 5]`                              | 返回单个 `5`                                               |
| NaN + 实数 `[1.0, NaN, 2.0]`                      | 返回长度为 3 的结果，且该 NaN 被保留                       |
| 多个 NaN `[NaN, NaN]`                             | 返回长度为 2 的结果                                        |
| `[-0.0, 0.0]`                                     | 返回长度为 1 的结果                                        |
| 复数 `[1+NaNi, 1+NaNi]`                           | 返回长度为 2 的结果（因为 NaN 分量不相等）                 |
| 大张量（`10^7` 元素，高重复）                     | 结果仍满足 unique 语义，且不改变 1D owned F-order 输出契约 |
| `IxDyn` rank 5+ 高维输入                          | 逻辑展平后去重，结果仍为 1D owned contiguous F-order 张量  |
| `i32::MIN` / `i32::MAX` / `i64::MIN` / `i64::MAX` | 极值按值语义去重，不发生额外归一化                         |
| 非连续切片视图                                    | 仅基于逻辑元素去重，不遗漏或重复                           |
| 转置视图                                          | 仅基于逻辑元素去重，不引入布局相关语义漂移                 |
| 含 padding 的张量                                 | padding 区域不参与 `unique()` 输入集合                     |

### 8.4 属性测试不变量

| 不变量                                       | 测试方法                                                              |
| -------------------------------------------- | --------------------------------------------------------------------- |
| 输出无重复（按 `unique_eq` 定义）            | 任意两个保留元素都不满足 `unique_eq`                                  |
| 非 NaN 输入时输出元素集合与输入集合相同      | 以参考集合语义对比                                                    |
| NaN 元素按出现次数保留                       | 统计输入/输出中的 NaN 数量并比较                                      |
| 多维输入始终返回 1D 结果                     | 随机 2D/3D 形状输入                                                   |
| 输出与输入按 multiset 语义相等（除 NaN 外）  | 输入去重后与输出按 `HashSet` / multiset 比较；NaN 元素按出现次数比较  |
| 输出元素两两不满足 `unique_eq`               | 任意两元素 `unique_eq` 为 `false`（NaN 永远互不相等）                 |

### 8.5 集成测试

| 测试文件            | 测试内容                                                                           |
| ------------------- | ---------------------------------------------------------------------------------- |
| `tests/test_set.rs` | `unique()` 与 `tensor`、`iter`、`element`、`complex`、`alloc` 路径的端到端协同验证 |

### 8.6 Feature gate / 配置测试

| 配置              | 验证点                                                                 |
| ----------------- | ---------------------------------------------------------------------- |
| 默认配置          | `unique()` 在默认构建下保持 NaN 保留、`-0.0 == 0.0` 与"输出顺序未定义"契约。|
| 其他 feature 组合 | 不适用；当前模块无额外 feature gate。                                  |

### 8.7 类型边界 / 编译期测试

| 场景                                           | 测试方式                            |
| ---------------------------------------------- | ----------------------------------- |
| `bool` 不实现 `UniqueElement`                  | 编译期测试。                        |
| 多维输入统一返回 `Tensor<A, Ix1>`              | 编译期签名检查与运行时 shape 断言。 |
| sort / bincount / unique counts 不属于当前 API | API 缺失断言。                      |

---

## 9. 与其他模块的交互

### 9.1 接口约定

| 方向            | 对方模块  | 接口/类型                             | 约定                                    |
| --------------- | --------- | ------------------------------------- | --------------------------------------- |
| `set → tensor`  | `tensor`  | `TensorBase<S, D>` / `Tensor<A, Ix1>` | 消费输入张量并返回 1D owned 结果        |
| `set → iter`    | `iter`    | `Iter`                            | 使用元素迭代器收集逻辑元素              |
| `set → element` | `element` | `Element`                             | 元素 trait 边界由 `UniqueElement` 提供  |
| `set → set`     | `set`     | `UniqueElement`                       | 通过 `unique_eq` 约束去重语义           |

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

| 主题              | 内容                                                                           |
| ----------------- | ------------------------------------------------------------------------------ |
| Recoverable error | 不适用；当前 `unique()` API 直接返回结果张量，不暴露模块级 `Result` 错误路径。 |
| Panic             | 不适用；除分配失败等通用运行时故障外，模块不定义额外 panic 语义。              |
| 路径一致性        | 本模块不接入 SIMD / 并行后端。外部语义由 `unique_eq`单独决定。                 |
| 容差边界          | 不适用。                                                                       |

---

## 11. 设计决策记录

### 决策 1：bool 排除理由

| 属性     | 值                                                          |
| -------- | ----------------------------------------------------------- |
| 决策     | unique 不支持 bool 类型                                     |
| 理由     | `需求说明书 §15` 已明确将 bool 排除在当前版本范围之外       |
| 替代方案 | 支持 bool unique，返回 [false, true]                        |
| 拒绝原因 | 增加维护负担，收益几乎为零；`需求说明书 §15` "bool 不适用"  |

### 决策 2：NaN / signed-zero 处理策略

| 属性          | 值                                                                       |
| ------------- | ------------------------------------------------------------------------ |
| 决策          | `unique` 严格沿用 IEEE 754 / Rust 相等语义：`NaN != NaN`，`-0.0 == 0.0`  |
| 理由          | 直接满足 `需求说明书 §15`，避免文档额外发明"canonical NaN"语义           |
| 替代方案 (a)  | 归并全部 NaN                                                             |
| 替代方案 (b)  | 把 `-0.0` 与 `0.0` 视为不同值                                            |
| 拒绝原因      | 均与需求说明书冲突                                                       |

### 决策 3：复数按分量判等

| 属性     | 值                                                                    |
| -------- | --------------------------------------------------------------------- |
| 决策     | 复数去重仅按实部与虚部逐分量判等                                      |
| 理由     | `需求说明书 §15` 只要求 component-wise equality，并未授权任何排序语义 |
| 替代方案 | lexicographic order                                                   |
| 拒绝原因 | 会把排序错误地写入公开契约，并掩盖 NaN 分量应逐个保留的要求           |

### 决策 4：当前版本不引入 SIMD / 并行路径

| 属性          | 值                                                                     |
| ------------- | ---------------------------------------------------------------------- |
| 决策          | 当前版本只有单一执行路径（标量），不接入 SIMD / 并行后端               |
| 理由          | `unique` 的难点在于查重逻辑而非元素吞吐；标量路径已可满足当前性能需求  |
| 替代方案      | 当前版本就引入 SIMD / 并行                                             |
| 拒绝原因      | 复杂度收益不成正比                                                     |

---

## 12. 性能考量

### 12.1 复杂度

- 对外语义不承诺具体算法复杂度
- 参考实现可采用线性扫描去重，但对大张量主路径应优先采用不改变外部语义的哈希辅助结构
- 当 O(N²) CPU 成本不可接受时应切换到哈希路径
- 输出顺序未定义；实现可任选 first-occurrence、hash-iteration 等顺序

### 12.2 内存开销

- 收集元素: O(N) 临时 Vec
- 去重辅助状态: 取决于具体实现，可为 O(1) 到 O(N)
- 结果: O(U) 其中 U 为保留后的元素数量（含每个被保留的 NaN）

---

## 13. 平台与工程约束

| 约束       | 说明                                                         |
| ---------- | ------------------------------------------------------------ |
| `std` only | Xenon 当前版本仅支持 `std` 环境，本文不再讨论 `no_std` 路径  |
| MSRV       | Rust 1.85+                                                   |
| 单 crate   | `set` 设计保持在现有 crate 内，不引入额外 crate              |
| SemVer     | 遵循SemVer                                                   |
| 最小依赖   | 本模块不新增第三方依赖                                       |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
