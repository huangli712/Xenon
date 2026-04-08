# 集合操作模块设计

> 文档编号: 14 | 模块: `src/ops/set_ops.rs` | 阶段: Phase 4
> 前置文档: `03-element-types.md`, `07-tensor.md`
> 需求参考: 需求说明书 §15

---

## 1. 模块定位

### 1.1 职责边界

| 职责 | 包含 | 不包含 |
|------|------|--------|
| unique 操作 | 返回排序后的不重复元素作为新 1D 张量 | intersection/union/difference |
| 支持类型 | i32, i64, f32, f64, Complex<f32>, Complex<f64> | bincount/histogram |
| 不支持类型 | bool（仅 2 种值，unique 无意义）、usize（仅用于索引） | argmin/argmax |
| — | — | usize 类型的 unique 操作（usize 仅用于索引，对应需求§4中"整数"特指 i32/i64） |

> **注意**：当前版本仅支持 unique 操作！不包含 intersection/union/difference/bincount/histogram 等。

> **usize 排除说明**：需求说明书 §4 中提到的"整数"特指 i32/i64，usize 仅用于索引，不参与集合操作。

### 1.2 设计原则

| 原则 | 体现 |
|------|------|
| 最小范围 | 当前仅实现 unique，其他集合操作留待未来扩展 |
| 类型安全 | bool 和 usize 显式排除 |
| NaN 处理明确 | 浮点 NaN 掂与排序有明确定义行为 |
| 复数排序明确 | 先按实部再按虚部 |

### 1.3 在架构中的位置

```
依赖层级：
L0: error, private
L1: dimension, element, complex
L2: layout (依赖 dimension)
L3: storage (依赖 layout)
L4: tensor (依赖 storage, dimension)
L5: ops/set_ops  ← 当前模块
```

---

## 2. 文件位置

```
src/ops/
├── mod.rs              # 模块入口
└── set_ops.rs           # 集合操作（本模块）
```

单文件设计理由：当前仅实现 unique，未来扩展时再拆分。

---

## 3. 依赖关系

### 3.1 依赖图

```
src/ops/set_ops.rs
├── crate::tensor        # TensorBase<S, D>, Tensor<A, Ix1>
├── crate::element       # Element, Numeric, ComplexScalar
└── crate::iter          # Elements（收集元素）
```

### 3.2 类型级依赖

| 来源模块 | 使用的类型/trait |
|----------|-----------------|
| `tensor` | `TensorBase<S, D>`, `Tensor<A, Ix1>`, `.iter()`, `.len()`，参见 `07-tensor.md` §4 |
| `element` | `Element`, `ComplexScalar`，参见 `03-element-types.md` §3 |
| `iter` | `Elements`（遍历收集元素），参见 `10-iterator.md` §3 |

### 3.3 依赖方向

> **依赖方向：单向向上。** `ops/set_ops` 仅消费 `tensor`、`element`、`iter` 模块。

---

## 4. 公共 API 设计

### 4.1 unique 操作

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: UniqueElement,  // UniqueElement: Element, correct constraint
{
    /// Returns sorted unique elements as a new 1D tensor.
    ///
    /// # Supported types
    ///
    /// i32, i64, f32, f64, Complex<f32>, Complex<f64>
    ///
    /// # Unsupported types
    ///
    /// - bool: only 2 values (true/false), unique is meaningless for bool
    /// - usize: used only for indexing, no set operation semantics
    ///
    /// # Float sorting behavior
    ///
    /// NaN is placed last (NaN > any real number).
    ///
    /// # Complex sorting rule
    ///
    /// Sort by real part first, then by imaginary part if real parts are equal.
    ///
    /// # Empty array behavior
    ///
    /// Empty array returns an empty Tensor1.
    ///
    /// # Complexity
    ///
    /// O(N log N)
    ///
    /// # Examples
    ///
    /// ```
    /// let a = Tensor1::from_vec(vec![3, 1, 2, 1, 3, 2]);
    /// let u = a.unique();
    /// assert_eq!(u, Tensor1::from_vec(vec![1, 2, 3]));
    ///
    /// // empty
    /// let empty: Tensor1<i32> = Tensor1::zeros([0]);
    /// assert_eq!(empty.unique().len(), 0);
    /// ```
    pub fn unique(&self) -> Tensor<A, Ix1>;
}
```

### 4.2 Good / Bad 对比示例

```rust
// Good - use unique to get sorted unique elements
let a = Tensor1::from_vec(vec![3, 1, 2, 1, 3]);
let u = a.unique();
assert_eq!(u.as_slice(), &[1, 2, 3]);

// Good - empty array returns empty Tensor1
let empty: Tensor1<i32> = Tensor1::zeros([0]);
assert_eq!(empty.unique().len(), 0);

// Bad - calling unique on a bool tensor (compile error)
// let b = Tensor1::from_vec(vec![true, false, true]);
// b.unique();  // compile error: bool does not implement UniqueElement trait

// Bad - calling unique on a usize tensor (compile error)
// let idx = Tensor1::from_vec(vec![3usize, 1, 2]);
// idx.unique();  // compile error: usize does not implement UniqueElement trait
```

---

## 5. 内部实现设计

### 5.1 unique 实现步骤

```
unique(self):
    1. 收集所有元素到 Vec<A>
    2. 排序: vec.sort_by(|a, b| a.total_cmp(b))
    3. 去重: vec.dedup_by(|a, b| a.total_eq(b))
    4. 从 Vec 构造 Tensor<A, Ix1>
```

### 5.2 浮点排序处理

```
浮点比较策略（基于 f64::total_cmp / f32::total_cmp）：
- NaN 被视为大于任何实数（NaN > f64::MAX）
- 排序后 NaN 出现在数组末尾
- -NaN 和 NaN 视为相等（通过 total_eq 去重）
- 与 IEEE 754 totalOrder 一致
```

### 5.3 复数排序规则

```
复数比较策略（lexicographic order）：
- 先比较实部（re）
- 实部相同再比较虚部（im）
- Complex { re: 1.0, im: 2.0 } < Complex { re: 1.0, im: 3.0 }
- Complex { re: 1.0, im: 2.0 } < Complex { re: 2.0, im: 0.0 }
```

### 5.4 类型排除实现

```rust
/// Trait for types that support the unique operation.
///
/// Provides a total ordering for sorting and deduplication.
/// `bool` and `usize` do not implement this trait.
///
/// Note: `Ord` is NOT used as a supertrait because `f32`/`f64`
/// do not implement `Ord`, and `Complex` does not implement `PartialOrd`.
/// Instead, each type defines its own `total_cmp` method.
///
/// # Why in set_ops.rs, not element module?
///
/// UniqueElement is defined here rather than in the element module because
/// its semantic (total ordering for deduplication) is operation-specific,
/// not a fundamental element property. This avoids making the element module
/// depend on ordering semantics.
///
/// # Sealing
///
/// Directly inherits Sealed for defensive sealing, even though Element
/// already provides it. This makes the sealing intent explicit at the
/// trait definition site.
pub trait UniqueElement: Element + Sealed {
    /// Total ordering comparison for sorting and deduplication.
    ///
    /// Must satisfy: reflexivity, antisymmetry, transitivity, totality.
    fn total_cmp(&self, other: &Self) -> core::cmp::Ordering;

    /// Equality check consistent with `total_cmp`.
    fn total_eq(&self, other: &Self) -> bool {
        self.total_cmp(other) == core::cmp::Ordering::Equal
    }
}

impl UniqueElement for i32 {
    fn total_cmp(&self, other: &Self) -> core::cmp::Ordering { Ord::cmp(self, other) }
}

impl UniqueElement for i64 {
    fn total_cmp(&self, other: &Self) -> core::cmp::Ordering { Ord::cmp(self, other) }
}

impl UniqueElement for f32 {
    fn total_cmp(&self, other: &Self) -> core::cmp::Ordering {
        // Uses f32::total_cmp (stable since Rust 1.62).
        // NaN is sorted to the end; -NaN and NaN are considered equal.
        f32::total_cmp(self, other)
    }
}

impl UniqueElement for f64 {
    fn total_cmp(&self, other: &Self) -> core::cmp::Ordering {
        f64::total_cmp(self, other)
    }
}

impl UniqueElement for Complex<f32> {
    fn total_cmp(&self, other: &Self) -> core::cmp::Ordering {
        // Lexicographic order: compare real part first, then imaginary part.
        f32::total_cmp(&self.re, &other.re)
            .then_with(|| f32::total_cmp(&self.im, &other.im))
    }
}

impl UniqueElement for Complex<f64> {
    fn total_cmp(&self, other: &Self) -> core::cmp::Ordering {
        f64::total_cmp(&self.re, &other.re)
            .then_with(|| f64::total_cmp(&self.im, &other.im))
    }
}
// bool and usize do not implement this
```

---

## 6. 实现任务拆分

### Wave 1: 基础设施

- [ ] **T1**: 创建 `src/ops/set_ops.rs` 骨架
  - 文件: `src/ops/set_ops.rs`
  - 内容: 模块声明、UniqueElement trait 定义
  - 测试: 编译通过
  - 前置: `07-tensor.md` 完成
  - 预计: 5 min

### Wave 2: 核心实现

- [ ] **T2**: 实现 `unique` 方法
  - 文件: `src/ops/set_ops.rs`
  - 内容: 元素收集、排序、去重、Tensor 构造
  - 测试: `test_unique_basic`, `test_unique_empty`, `test_unique_single`, `test_unique_duplicates`
  - 前置: T1
  - 预计: 10 min

### Wave 3: 浮点与复数扩展

- [ ] **T3**: 实现浮点 NaN 排序处理
  - 文件: `src/ops/set_ops.rs`
  - 内容: NaN 比任何实数大，排序后 NaN 在末尾
  - 测试: `test_unique_nan_f32`, `test_unique_nan_f64`
  - 前置: T2
  - 预计: 10 min

- [ ] **T4**: 实现复数排序规则
  - 文件: `src/ops/set_ops.rs`
  - 内容: 先按实部再按虚部的 lexicographic order
  - 测试: `test_unique_complex_order`
  - 前置: T2
  - 预计: 10 min

### Wave 4: TensorBase 入口集成

- [ ] **T5**: 在 TensorBase 上添加 `unique()` 入口方法
  - 文件: `src/ops/set_ops.rs`（或 trait extension）
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

## 7. 测试计划

### 7.1 单元测试清单

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_unique_basic_i32` | 噪声去除: [3,1,2,1,3,2] → [1,2,3] | 高 |
| `test_unique_basic_i64` | i64 类型正确性 | 高 |
| `test_unique_basic_f32` | f32 类型正确性 | 高 |
| `test_unique_basic_f64` | f64 类型正确性 | 高 |
| `test_unique_basic_complex` | Complex<f64> 类型正确性 | 高 |
| `test_unique_empty` | 空数组返回空 Tensor1 | 高 |
| `test_unique_single` | 单元素返回自身 | 中 |
| `test_unique_all_same` | 所有元素相同返回单元素 | 中 |
| `test_unique_nan_f32` | f32 NaN 排在末尾 | 高 |
| `test_unique_nan_f64` | f64 NaN 排在末尾 | 高 |
| `test_unique_nan_mixed` | NaN + 正常数混合排序正确 | 高 |
| `test_unique_nan_dedup` | 含 NaN 和 -NaN 的浮点数组：NaN 与 -NaN 被视为相等，去重后只保留一个 NaN | 高 |
| `test_unique_complex_order` | 复数先实部再虚部排序 | 高 |
| `test_unique_2d` | 2D 张量 unique 返回 1D | 中 |
| `test_unique_preserves_order` | 去重后排序正确 | 中 |

### 7.2 边界测试场景

| 场景 | 预期行为 |
|------|----------|
| 空张量 `shape=[0]` | 返回空 Tensor1 |
| 单元素 `[42]` | 返回 `[42]` |
| 全部相同 `[5, 5, 5]` | 返回 `[5]` |
| NaN + 实数 `[1.0, NaN, 2.0]` | 返回 `[1.0, 2.0, NaN]` |
| 复数 `[1+2i, 1+1i, 2+0i]` | 返回 `[1+1i, 1+2i, 2+0i]` |
| 已排序输入 `[1, 2, 3]` | 返回 `[1, 2, 3]`（无拷贝优化） |
| 逆序输入 `[3, 2, 1]` | 返回 `[1, 2, 3]` |

---

## 8. 与其他模块的交互

| 交互模块 | 接口约定 |
|----------|----------|
| `tensor` | 消费 `TensorBase<S, D>`，返回 `Tensor<A, Ix1>`，参见 `07-tensor.md` §4 |
| `iter` | 使用 `Elements` 迭代器收集元素，参见 `10-iterator.md` §3 |
| `element` | 泛型约束 `UniqueElement: Element`（排除 bool/usize，提供 `total_cmp`），参见 `03-element-types.md` §3 |

---

## 9. 设计决策记录

### 决策 1：bool 排除理由

| 属性 | 值 |
|------|-----|
| 决策 | unique 不支持 bool 类型 |
| 理由 | bool 只有 true/false 两种值，unique 对 bool 无意义；用户可直接用 `== true` / `== false` 判断；且 bool 的 unique 语义不明确（是返回 [false, true] 还是 [0, 1]？） |
| 替代方案 | 支持 bool unique，返回 [false, true] |
| 拒绝原因 | 增加维护负担，收益几乎为零；需求说明书 §15 "bool 不适用" |

### 决策 2：NaN 处理策略

| 属性 | 值 |
|------|-----|
| 决策 | NaN 排在排序结果的末尾（NaN > 任何实数） |
| 理由 | 与 IEEE 754 的 `totalOrder` 比较一致（NaN 被视为大于所有值）；与 NumPy `np.unique` 行为一致；多个 NaN 视为相同值（去重后只保留一个 NaN） |
| 替代方案 | NaN 排在最前 |
| 替代方案 | 抛弃所有 NaN |
| 拒绝原因 | 与 IEEE 754 不一致；可能丢失数据信息 |

### 决策 3：复数排序规则

| 属性 | 值 |
|------|-----|
| 决策 | 复数按 lexicographic order 排序：先实部再虚部 |
| 理由 | 这是数学中最常见的复数全序关系；与 C++ `std::complex` 的 `<` 运算符一致；实现简单直观 |
| 替代方案 | 按模排序 |
| 拒绝原因 | 模（绝对值）排序不是全序关系（模相等的复数无法排序）；与标准库不一致 |

---

## 10. 性能考量

### 10.1 复杂度

- 排序: O(N log N)，其中 N 为元素数量
- 去重: O(N)，在线性扫描
- 总体: O(N log N)

### 10.2 内存开销

- 收集元素: O(N) 临时 Vec
- 排序: O(log N) 栈空间（快排）
- 结果: O(U) 其中 U 为唯一值数量

### 10.3 大数组性能（参考）

| 元素数 | 排序时间 | 去重时间 | 总时间 |
|--------|----------|----------|--------|
| 1K | ~0.01ms | ~0.001ms | ~0.01ms |
| 1M | ~15ms | ~1ms | ~16ms |
| 100M | ~2s | ~100ms | ~2.1s |

---

## 11. no_std 兼容性

集合操作模块在 `no_std` 环境下可用，但需 `alloc` 支持以分配临时 `Vec` 和结果张量。排序使用 `alloc::vec::Vec::sort`。

```rust
#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
```

| 组件 | no_std 支持 | 说明 |
|------|:----------:|------|
| `unique()` | ✅ | 需 `no_std + alloc`，收集元素到 `Vec` + 排序 + 去重，参见 `05-storage.md` §11 |
| `UniqueElement` trait | ✅ | 纯 trait 定义，无依赖 |
| 浮点 NaN 排序 | ✅ | `f32::total_cmp` / `f64::total_cmp`，`core` 内建（Rust 1.62+） |
| 复数排序 | ✅ | lexicographic `total_cmp` 实现，参见 `04-complex-type.md` §3 |

条件编译处理：

```rust
// unique() requires:
//   1. Collect elements into Vec → alloc::vec::Vec
//   2. Sort in-place → Vec::sort (available in alloc)
//   3. Dedup → Vec::dedup (available in alloc)
//   4. Construct result Tensor → alloc::vec::Vec

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
| 1.0.3 | 2026-04-08 |
| 1.0.4 | 2026-04-08 |
| 1.1.0 | 2026-04-08 |

---

*本文档由 Xenon 维护。如有问题请提交 Issue 或 PR。*
