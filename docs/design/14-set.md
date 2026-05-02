# 集合操作模块设计

> 文档编号: 14
> 模块目录: src/set/
> 任务阶段: Phase 4
> 前置文档: 03-element.md, 04-complex.md, 07-tensor.md, 10-iterator.md
> 需求参考: 需求说明书 §4、§15、§28
> 范围声明: 范围内

---

## 1. 模块定位

### 1.1 职责边界

| 职责         | 包含                                                      |
| ------------ | --------------------------------------------------------- |
| 集合操作     | unique: 返回不重复元素组成的新 1D 张量；按"逻辑首次出现顺序"输出（参见 §1.2 / 决策 4） |
| 支持类型     | i32, i64, f32, f64, Complex<f32>, Complex<f64>            |

| 职责         | 不包含                                              |
| ------------ | --------------------------------------------------- |
| 集合操作     | intersection / union / difference                   |
| 统计操作     | bincount / histogram                                |
| 归约索引     | argmin / argmax                                     |
| 支持类型     | `需求说明书 §15` 明确将 bool 排除在 `unique` 之外） |

### 1.2 设计原则

| 原则           | 体现                                                          |
| -------------- | ------------------------------------------------------------- |
| 最小范围       | 当前仅实现 unique，其他集合操作留待未来扩展                   |
| 类型安全       | bool 显式排除；仅对受支持的元素类型开放                       |
| 相等语义优先   | `unique` 基于逐元素相等关系去重，不承诺排序结果（不进行 lexicographic / 模长 / 数值排序）|
| 顺序可复现     | 输出元素按"逻辑首次出现顺序"排列（按 F-order 元素遍历顺序的首次出现位置）；同一进程内对相同输入的多次调用结果完全一致（决策 4，对应需求说明书 §15）|
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

双文件设计理由：当前范围由 `mod.rs` 承担模块入口、`unique.rs` 承担唯一公开集合操作实现，保持导出边界与语义实现分离。

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
└── crate::iter          # Elements for collection
```

### 4.2 类型级依赖

| 来源模块    | 使用的类型/trait                                                                   |
| ----------- | ---------------------------------------------------------------------------------- |
| `tensor`    | `TensorBase<S, D>`, `Tensor<A, Ix1>`, `.iter()`, `.len()`，参见 `07-tensor.md` §5  |
| `storage`   | `Storage<Elem = A>` trait（read-only element access via `Storage<Elem = A>`）      |
| `dimension` | `Dimension`, `Ix1`（output dimension type for flatten result）                     |
| `element`   | `Element`（参见 `03-element.md` §5.1）；`UniqueElement: Element` 蕴含 `Copy`（`Element: Copy`），无需额外约束元素层 trait。`ComplexScalar` **未实际使用**——`Complex<f32>` / `Complex<f64>` 通过为这两个具体类型分别 impl `UniqueElement` 来支持，而不是通过 `ComplexScalar` 泛型。 |
| `complex`   | `Complex<f32>`, `Complex<f64>`，参见 `04-complex.md` §5                            |
| `iter`      | `Elements`（遍历收集元素），参见 `10-iterator.md` §5.1                             |

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
    /// Returns unique elements in **first-occurrence order**.
    ///
    /// # Output ordering contract
    ///
    /// Elements are returned in the order each unique value is **first**
    /// encountered when iterating logical elements of `self` in F-order
    /// (column-major). Two calls on the same input within the same process
    /// produce bit-identical outputs (decision 4); cross-process / cross-platform
    /// reproducibility is not promised because float bit patterns may differ.
    /// No sorting is performed (no lexicographic / magnitude / numeric ordering).
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
    /// element order within the output is the first-occurrence order described above.
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
    /// the chosen strategy: ordering and deduplication results are identical.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let a = Tensor::<i32, Ix1>::from_shape_vec(Ix1(6), vec![3, 1, 2, 1, 3, 2])?;
    /// let u = a.unique();
    /// assert_eq!(u.len(), 3);
    /// assert!(u.iter().all(|x| [1, 2, 3].contains(x)));
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
// Good - use unique; output is in first-occurrence order (deterministic per input)
let a = Tensor::<i32, Ix1>::from_shape_vec(Ix1(5), vec![3, 1, 2, 1, 3])?;
let u = a.unique();
assert_eq!(u.len(), 3);

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
    1. Iterate logical elements of `self` in F-order (column-major).
    2. For each element x, decide whether x already has a representative in the
       output set:
         - For non-NaN values: equivalent under `unique_eq`.
         - For NaN values (per IEEE 754): every NaN is its own class; do not
           merge with any prior NaN. Keep all NaN instances in encounter order.
       Use linear scan for small inputs, or hash-aided lookup for large inputs
       (key construction rules in §6.1.1 below).
    3. If x is the first occurrence of its class (or x is NaN), append x to
       the output sequence.
    4. Construct `Tensor<A, Ix1>` from the appended sequence (owned F-order
       contiguous buffer).

Ordering contract (decision 4):
    - Output elements appear in F-order first-occurrence order.
    - Same input in the same process gives bit-identical output.
    - Implementations MUST NOT use a strategy that produces order-varying
      output (e.g., a HashMap with default randomized hasher used as the
      *output* container; using a hash table only as a lookup index for
      first-occurrence detection is fine because the output sequence is
      driven by the iteration order of the input).
```

#### 6.1.1 哈希键规范（用于大输入快路径）

 **实现约束**:

 对 `f32` / `f64` 及 `Complex<f32>` / `Complex<f64>` 的 `unique` 实现，**不得**直接依赖标准 Rust `Hash` / `Eq` 语义，也**不得**直接建立在 `BTreeSet` / `HashSet` 这类标准集合之上；必须使用线性扫描或自定义哈希键策略，以严格满足本文档定义的判等规则：

 1. `NaN != NaN`，因此每个 `NaN` 都必须单独保留，不能因为"同为 NaN"而被合并。
 2. `-0.0 == 0.0`，因此两者必须视为同一个 unique 值。
 3. 复数按分量比较，且每个分量分别沿用对应实数的上述语义。
 4. 若实现采用哈希优化，键规范固定如下：NaN 元素不进入普通去重键路径。实现须对 NaN 单独旁路处理，保证输入中的每个 NaN（无论位模式是否相同）均被保留。普通哈希键仅用于非 NaN 元素；其中 `i32` / `i64` 直接以数值作为键，`f32` / `f64` 对所有 `+0.0` / `-0.0` 归一到同一键，`Complex<T>` 的键为 `(re_key, im_key)`，并对含 NaN 的分量同样走旁路保留逻辑。
 5. **哈希表只作为查重索引使用，不作为输出容器**。输出顺序必须由"输入逻辑迭代顺序"驱动（决策 4），因此哈希表的迭代顺序（即使是默认随机化 hasher 的）不会泄漏到输出。

换言之，若实现采用哈希优化，则键设计必须显式编码这些语义；若无法保证，则应退回线性扫描，禁止使用与本文档语义不一致的默认集合判重行为。

### 6.2 浮点判等处理

- 非 NaN 浮点值的相等判定遵循 Rust / IEEE 754 `==` 语义
- `NaN != NaN`，因此输入中的每个 `NaN` 必须独立保留，不参与去重
- `+0.0 == -0.0`，因此两者视为同一个 unique 值
- 本文档约束相等语义与输出顺序（按 §1.2 / 决策 4 的"逻辑首次出现顺序"）；不限制实现使用哈希、线性扫描或其他**查重**策略，但策略不得改变输出顺序契约

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
/// `UniqueElement` is a sealed trait. It reuses the shared `crate::private::Sealed`
/// infrastructure (defined in `03-element.md §5.7`), consistent with all other
/// public element capability traits.
/// It is implemented only inside this crate for supported element types,
/// so the closed element set is preserved.
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
| 小输入或原型实现（约 N ≤ 64 的简单类型；阈值由实现选取） | 线性扫描          | 直接复用 `unique_eq`，最坏 O(N²)；不引入额外内存分配，常数项小，对短输入更快。|
| 大输入主路径      | 哈希查重 + 顺序输出列表 | 用哈希表作为"该元素是否已见过"的查重索引，输出序列另用 `Vec<A>` 按输入逻辑顺序追加。重复检测降到近似 O(N) 摊销，输出顺序与线性扫描严格一致（决策 4）。|
| 浮点/复数特殊值 | 专门分支处理        | `NaN != NaN`，因此哈希或索引策略也必须显式保留每个 `NaN`（旁路路径直接追加到输出列表，不进入哈希表），不得把它们合并。|

**何时必须用哈希路径**：实现可自行选择阈值（如 N > 1024 或元素总数与去重比例的启发式），但当输入规模导致线性扫描的 O(N²) 内存或 CPU 成本不可接受时，必须切换到哈希路径以避免大张量上的不可接受性能。

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
| `test_unique_basic_i32`                     | 噪声去除后结果包含且仅包含 1/2/3；不要求顺序 | 高     |
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
| `test_unique_first_occurrence_order`        | 同一输入断言完整输出向量按 F-order 首次出现顺序排列 | 高     |
| `test_unique_reproducible_within_process`   | 同一输入连续两次调用结果 bit-identical（决策 4）    | 高     |
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

| 不变量                                                  | 测试方法                                                              |
| ------------------------------------------------------- | --------------------------------------------------------------------- |
| 输出无重复（按 `unique_eq` 定义）                       | 任意两个保留元素都不满足 `unique_eq`                                  |
| 非 NaN 输入时输出元素集合与输入集合相同                 | 以参考集合语义对比                                                    |
| NaN 元素按出现次数保留                                  | 统计输入/输出中的 NaN 数量并比较                                      |
| 多维输入始终返回 1D 结果                                | 随机 2D/3D 形状输入                                                   |
| 输出按 F-order 首次出现顺序排列（决策 4）               | 与参考线性扫描实现的输出向量逐位比较                                  |
| 同进程内同输入两次调用 bit-identical（决策 4）          | `a.unique() == a.unique()`，对所有受支持类型的随机输入                  |

### 8.5 集成测试

| 测试文件            | 测试内容                                                                           |
| ------------------- | ---------------------------------------------------------------------------------- |
| `tests/test_set.rs` | `unique()` 与 `tensor`、`iter`、`element`、`complex`、`alloc` 路径的端到端协同验证 |

### 8.6 Feature gate / 配置测试

| 配置              | 验证点                                                                 |
| ----------------- | ---------------------------------------------------------------------- |
| 默认配置          | `unique()` 在默认构建下保持 NaN 保留、`-0.0 == 0.0` 与"逻辑首次出现顺序、同输入可复现"契约（决策 4）。|
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

| 方向            | 对方模块  | 接口/类型                             | 约定                                         |
| --------------- | --------- | ------------------------------------- | -------------------------------------------- |
| `set → tensor`  | `tensor`  | `TensorBase<S, D>` / `Tensor<A, Ix1>` | 消费输入张量并返回 1D owned 结果，参见 `07-tensor.md` §5                  |
| `set → iter`    | `iter`    | `Elements`                            | 使用元素迭代器收集逻辑元素，参见 `10-iterator.md` §5.1                    |
| `set → element` | `element` | `Element`（其 `Copy` 继承用于元素值复制）| 元素 trait 边界由 `UniqueElement: Element` 提供；`ComplexScalar` 未直接使用，复数支持通过对 `Complex<f32>` / `Complex<f64>` 分别 impl `UniqueElement` 完成（参见 `03-element.md` §5.1）|
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

| 主题              | 内容                                                                           |
| ----------------- | ------------------------------------------------------------------------------ |
| Recoverable error | 不适用；当前 `unique()` API 直接返回结果张量，不暴露模块级 `Result` 错误路径。 |
| Panic             | 不适用；除分配失败等通用运行时故障外，模块不定义额外 panic 语义。              |
| 路径一致性        | 当前仅有单一执行路径（标量）；本模块不接入 SIMD / 并行后端（决策 5）。外部语义由 `unique_eq` 与决策 4 的"逻辑首次出现顺序"共同决定。|
| 容差边界          | 不适用。                                                                       |

---

## 11. 设计决策记录

### 决策 1：bool 排除理由

| 属性     | 值                                                                      |
| -------- | ----------------------------------------------------------------------- |
| 决策     | unique 不支持 bool 类型                                                 |
| 理由     | `需求说明书 §15` 已明确将 bool 排除在当前版本范围之外                   |
| 替代方案 | 支持 bool unique，返回 [false, true]                                    |
| 拒绝原因 | 增加维护负担，收益几乎为零；`需求说明书 §15` "bool 不适用"              |

### 决策 2：NaN / signed-zero 处理策略

| 属性          | 值                                                                              |
| ------------- | ------------------------------------------------------------------------------- |
| 决策          | `unique` 严格沿用 IEEE 754 / Rust 相等语义：`NaN != NaN`，`-0.0 == 0.0`          |
| 理由          | 直接满足 `需求说明书 §15`，避免文档额外发明"canonical NaN"语义                  |
| 替代方案 (a)  | 归并全部 NaN                                                                    |
| 替代方案 (b)  | 把 `-0.0` 与 `0.0` 视为不同值                                                   |
| 拒绝原因      | 均与需求说明书冲突                                                              |

### 决策 3：复数按分量判等

| 属性     | 值                                                                    |
| -------- | --------------------------------------------------------------------- |
| 决策     | 复数去重仅按实部与虚部逐分量判等                                      |
| 理由     | `需求说明书 §15` 只要求 component-wise equality，并未授权任何排序语义 |
| 替代方案 | lexicographic order                                                   |
| 拒绝原因 | 会把排序错误地写入公开契约，并掩盖 NaN 分量应逐个保留的要求           |

### 决策 4：输出顺序按"逻辑首次出现"，同进程同输入可复现

| 属性          | 值                                                                                                                                                                                                                                       |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 决策          | `unique()` 输出按输入逻辑迭代顺序（F-order）每个等价类首次出现的位置排列；同一进程内对相同输入的多次调用产生 bit-identical 输出                                                                                                          |
| 理由          | (1) 用户广泛预期数组库 `unique` 输出可复现（NumPy `np.unique` 无序但单调；Pandas / R `unique` 严格首次出现序）；(2) 测试可断言完整向量而不只是元素集合，调试与回归基线更稳定；(3) 不要求排序，避免引入排序成本与"哪种 order"的语义负担 |
| 替代方案 (a)  | 输出顺序完全未定义（v1.x 设计）                                                                                                                                                                                                          |
| 拒绝原因      | 数组库用户对 `unique` 输出可复现性的预期强烈；"未定义"会让单元测试只能断言集合语义而不能断言完整向量，调试经验差                                                                                                                          |
| 替代方案 (b)  | 数值排序                                                                                                                                                                                                                                 |
| 拒绝原因      | 强制排序会引入 O(N log N) 成本；NaN 与复数排序语义需要额外定义；且 `需求说明书 §15` 未要求排序                                                                                                                                            |
| 实现影响      | 哈希优化路径下，哈希表只作为"是否已见过"的查重索引，不作为输出容器；输出序列必须由输入迭代顺序驱动                                                                                                                                          |
| 跨进程 / 跨平台 | 不承诺。浮点 NaN 位模式可能不同；不同平台 SIMD / 并行配置不影响顺序（决策 5），但实现版本变更可能影响重复元素的"首次"判定来源（取最早索引），故承诺仅限同实现版本                                                                          |

### 决策 5：当前版本不引入 SIMD / 并行路径，未来若引入需保留顺序契约

| 属性          | 值                                                                                                                                                |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| 决策          | 当前版本只有单一执行路径（标量），不接入 SIMD / 并行后端                                                                                          |
| 理由          | `unique` 的难点在于查重逻辑而非元素吞吐；标量路径已可满足当前性能需求                                                                            |
| 未来约束      | 若未来引入 SIMD / 并行优化，必须保留决策 4 的"逻辑首次出现顺序、同输入可复现"契约——即并行实现也要按 chunk index 升序合并首次出现位置             |
| 替代方案      | 当前版本就引入 SIMD / 并行                                                                                                                        |
| 拒绝原因      | 复杂度收益不成正比；引入会让顺序契约的并行实现策略提前固化                                                                                        |

---

## 12. 性能考量

### 12.1 复杂度

- 对外语义不承诺具体算法复杂度
- 参考实现可采用线性扫描去重（O(N^2)），但对大张量主路径应优先采用不改变外部语义的哈希或索引辅助结构
- 无论内部实现如何，结果顺序都不是稳定契约的一部分

### 12.2 内存开销

- 收集元素: O(N) 临时 Vec
- 去重辅助状态: 取决于具体实现，可为 O(1) 到 O(N)
- 结果: O(U) 其中 U 为保留后的元素数量（含每个被保留的 NaN）

---

## 13. 平台与工程约束

| 约束       | 说明                                                                   |
| ---------- | ---------------------------------------------------------------------- |
| `std` only | Xenon 当前版本仅支持 `std` 环境，本文不再讨论 `no_std` 路径            |
| MSRV       | Rust 1.85+                                                             |
| 单 crate   | `set` 设计保持在现有 crate 内，不引入额外 crate                        |
| SemVer     | 遵循SemVer                                                             |
| 最小依赖   | 本模块不新增第三方依赖                                                 |
| 语义一致性 | 当前版本只有标量执行路径（决策 5）；若未来引入 SIMD / 并行等执行路径，必须保留 `unique` 的外部语义，包括决策 4 的"逻辑首次出现顺序、同进程同输入可复现" |

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
| 1.1.3 | 2026-04-15 |
| 2.0.0 | 2026-05-02 |

### v2.0.0 (2026-05-02) — 顺序契约与一致性更新

> 本版本是与用户决策 B5.c 协同的破坏性契约更新；同时纳入 26-error v3.0.0 / 03-element 一致性更新。

**契约破坏（B5.c 用户已批准）**：

- §1.1 / §1.2 / §5.1：`unique()` 输出顺序从"unspecified, may vary between calls"改为"按 F-order 逻辑首次出现顺序排列；同进程同输入可复现"（决策 4）。这是公开 API 契约的破坏性更新——v1.x 用户若依赖"顺序不稳定"假设进行容错测试，行为变得更严格但不会出错；若用户假设"无法测试输出向量"，新契约提供了更强保证，可向后兼容。

**协同与一致性更新**：

- §1.2 设计原则表新增"顺序可复现"行；移除 v1.x 隐含的"顺序不作要求"。
- §4.2 / §4.1 / §9.1：移除对 `ComplexScalar` 的依赖（实际未使用），改为 `Complex<f32>` / `Complex<f64>` 的具体类型 impl；明确 `UniqueElement: Element` 通过 `Element: Copy` 继承得到 `Copy` 能力，trait bound 不再不闭合。
- §5.1：doc comment 重写"Output ordering contract"段；新增"Trait bound rationale"段说明 `Copy` 继承；新增"Complexity"段引用 §6.5 大输入哈希策略。
- §6.1：unique 实现步骤重写为"输入迭代驱动 + 哈希查重索引（不作输出容器）"模型；明确哈希表只用于查重，不作为输出容器，输出序列必须由输入迭代顺序驱动。
- §6.2：本文档约束扩展到"相等语义 + 输出顺序"。
- §6.5 推荐策略表细化：小输入用线性扫描（约 N ≤ 64 阈值由实现选取），大输入用"哈希查重 + 顺序输出列表"；新增"何时必须用哈希路径"段。
- §10：路径一致性表述改为"当前仅有单一标量执行路径（决策 5）"，去掉 v1.x 含糊的"无 SIMD/并行分支"。
- §11：决策 2 修复重复"替代方案"行的 markdown 结构错误；新增决策 4（顺序契约）和决策 5（当前不引入 SIMD/并行 + 未来约束）。
- §13 平台约束："SIMD/并行不得改变外部语义"重写为"当前只有标量执行路径，未来若引入需保留决策 4 的顺序契约"。
- §8.2 单元测试：新增 `test_unique_first_occurrence_order`、`test_unique_reproducible_within_process`；删除 `test_unique_order_unspecified`。
- §8.4 属性测试：新增"输出按 F-order 首次出现顺序排列"和"同进程同输入两次调用 bit-identical"两条不变量。

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
