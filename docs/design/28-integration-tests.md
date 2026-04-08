# 集成测试模块设计

> 文档编号: 28 | 模块: `tests/` | 阶段: Phase 6
> 前置文档: 所有前置文档（`00-coding-standards.md` ~ `27-benchmark.md`）
> 需求参考: 需求说明书 §28.2, §28.4

---

## 1. 模块定位

### 1.1 职责边界

| 职责 | 包含 | 不包含 |
|------|------|--------|
| 跨模块验证 | 维度、存储、布局、运算等模块的协同行为（参见 `01-architecture-overview.md §5`） | 单函数测试（由 `#[cfg(test)] mod tests` 覆盖） |
| 边界覆盖 | 空张量、单元素、大张量、极端值、非连续、高维 | 性能测量（由 benchmark 覆盖） |
| 数值精度 | IEEE 754 精度验证 | 微观 benchmark |
| 属性测试 | 代数不变量验证 | 内存泄漏检测 |
| 并行安全 | 无数据竞争、并行/串行一致性 | 并行性能（由 benchmark 覆盖） |

### 1.2 设计原则

| 原则 | 体现 |
|------|------|
| 全面性 | 覆盖需求说明书中所有 API 的关键行为 |
| 独立性 | 每个测试文件可独立运行，无跨文件依赖 |
| 可读性 | 测试名称描述预期行为，失败信息包含上下文 |
| 快速反馈 | 完整集成测试 < 5min |

### 1.3 在架构中的位置

```
依赖层级：
L0: error, private
L1: dimension, element, complex
L2: layout (依赖 dimension)
L3: storage (依赖 layout)
L4: tensor (依赖 storage, dimension)
L5: ops/, iter/, index/, shape_ops/, broadcast/, construct/, ffi/, convert/, format/

外部（非 crate 模块）：
tests/  ← 当前模块（仅消费 crate 公共 API）
```

---

## 2. 文件位置

### 2.1 目录结构

```
tests/
├── common/
│   ├── mod.rs                  # 共享工具导出
│   ├── assertions.rs           # 自定义断言宏（assert_tensor_close）
│   └── generators.rs           # 测试数据生成器
│
├── test_tensor.rs              # 张量核心功能（创建/查询/类型别名）
├── test_ops.rs                 # 逐元素运算（算术/数学/比较/逻辑）
├── test_broadcast.rs           # 广播机制（标量/向量/矩阵广播）
├── test_index.rs               # 索引操作（多维索引/范围切片）
├── test_construction.rs        # 构造方法（zeros/ones/eye/from_vec/arange/linspace）
├── test_reduction.rs           # 归约运算（sum/沿轴sum/unique）
├── test_shape_ops.rs           # 形状操作（transpose/reshape）
├── test_conversion.rs          # 类型转换（cast/存储模式转换）
├── test_ffi.rs                 # FFI 集成（原始指针/BLAS 兼容）
├── test_parallel.rs            # 并行计算（一致性/数据竞争）
├── test_simd.rs                # SIMD 计算（结果一致性）
├── test_no_std.rs              # no_std 兼容性（编译验证）
├── test_error.rs               # 错误处理（所有错误类型）
│
└── property/
    ├── mod.rs                  # 属性测试入口
    ├── tensor_props.rs         # 张量不变量（reshape 保元素数等）
    ├── ops_props.rs            # 运算不变量（交换律/结合律等）
    └── shape_props.rs          # 形状不变量（transpose 自反等）
```

### 2.2 划分理由

按测试领域分文件，而非按源码模块：集成测试关注跨模块行为而非单个模块内部。

---

## 3. 依赖关系

### 3.1 依赖图

```
tests/
├── crate::tensor           # TensorBase, Tensor, TensorView, TensorViewMut, ArcTensor
├── crate::dimension        # Ix0~Ix6, IxDyn, Dimension
├── crate::element          # Element, Numeric, RealScalar, ComplexScalar
├── crate::complex          # Complex<f32>, Complex<f64>
├── crate::storage          # Owned, ViewRepr, ViewMutRepr, ArcRepr
├── crate::layout           # LayoutFlags, Order
├── crate::ops              # 逐元素运算、归约、内积
├── crate::broadcast        # broadcast_shape
├── crate::shape_ops        # transpose, reshape
├── crate::index            # 多维索引、范围切片
├── crate::construct        # zeros, ones, eye, from_vec, arange, linspace
├── crate::set_ops          # unique
├── crate::ffi              # as_ptr, as_mut_ptr, from_raw_parts
├── crate::workspace        # Workspace
├── crate::error            # XenonError
└── crate::simd/parallel    # 条件编译模块
```

### 3.2 依赖精确到类型级

| 来源模块 | 使用的类型/trait |
|----------|-----------------|
| `tensor` | `Tensor<A, D>`, `TensorView`, `TensorViewMut`, `ArcTensor`, `.shape()`, `.strides()`（参见 `07-tensor.md §4`） |
| `dimension` | `Ix0`~`Ix6`, `IxDyn`, `Dimension`, `DimensionMismatch`（参见 `02-dimension.md §4`） |
| `element` | `Element`, `Numeric`, `RealScalar`, `ComplexScalar`（参见 `03-element-types.md §4`） |
| `complex` | `Complex<f32>`, `Complex<f64>`（参见 `04-complex-type.md §4`） |
| `storage` | `Owned`, `ViewRepr`, `ViewMutRepr`, `ArcRepr`, `Storage`（参见 `05-storage.md §4`） |
| `layout` | `LayoutFlags`, `Order`（参见 `06-memory-layout.md §4`） |
| `error` | `XenonError`, `Result<T>`（参见 `26-error-handling.md §4`） |

### 3.3 依赖方向声明

> **依赖方向：单向消费。** `tests/` 仅消费 crate 公共 API（参见 `01-architecture-overview.md §10`），不被任何模块依赖。

---

## 4. 公共工具设计

### 4.1 tests/common/mod.rs

```rust
// tests/common/mod.rs
pub mod assertions;
pub mod generators;
```

### 4.2 tests/common/assertions.rs

```rust
// tests/common/assertions.rs

/// Assert two tensors are element-wise approximately equal (absolute tolerance).
pub fn assert_tensor_close<A, D>(
    actual: &TensorBase<impl Storage<Elem = A>, D>,
    expected: &TensorBase<impl Storage<Elem = A>, D>,
    atol: A,
    msg: &str,
) where
    A: RealScalar,
    D: Dimension,
{
    assert_eq!(actual.shape(), expected.shape(),
        "{}: shape mismatch: {:?} vs {:?}", msg, actual.shape(), expected.shape());

    for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (*a - *e).abs();
        assert!(diff <= atol,
            "{}: element {} differs: actual={}, expected={}, diff={}",
            msg, idx, a, e, diff);
    }
}

/// Assert a tensor operation returns the expected error variant.
macro_rules! assert_xenon_error {
    ($expr:expr, $variant:pat) => {
        match $expr {
            Err($variant) => {},
            Err(other) => panic!("expected {}, got {:?}", stringify!($variant), other),
            Ok(val) => panic!("expected error {}, got Ok({:?})", stringify!($variant), val),
        }
    };
}
```

### 4.3 tests/common/generators.rs

```rust
// tests/common/generators.rs

/// Standard 2D shapes for parameterized testing.
pub fn standard_shapes_2d() -> Vec<(usize, usize)> {
    vec![
        (0, 0), (1, 1), (1, 5), (5, 1),
        (3, 4), (4, 3), (8, 8), (64, 64),
    ]
}

/// Generate a non-contiguous 2D tensor (transposed view).
pub fn non_contiguous_2d(rows: usize, cols: usize) -> Tensor2<f64> {
    let t = Tensor2::<f64>::from_fn([cols, rows], |[i, j]| (i * rows + j) as f64);
    t.t().to_owned()
}
```

---

## 5. 完整测试函数清单

### 5.1 test_tensor.rs

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_tensor_shape_strides` | shape(), strides(), ndim(), len() 正确性 | 高 |
| `test_tensor_is_empty` | 空数组 is_empty() == true | 高 |
| `test_tensor_view_creation` | view(), view_mut() 创建与读取 | 高 |
| `test_tensor_to_owned` | to_owned() 深拷贝，修改不影响原始 | 高 |
| `test_tensor_into_owned` | into_owned() 消耗转换 | 中 |
| `test_type_aliases` | Tensor0~Tensor3, TensorD 别名正确 | 高 |
| `test_tensor_debug_display` | Debug/Display 格式化输出 | 中 |
| `test_arc_tensor_clone` | ArcTensor clone 为浅拷贝 | 中 |
| `test_arc_tensor_make_mut` | make_mut CoW 行为 | 中 |

### 5.2 test_ops.rs

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_add_same_shape` | 同形状加法 | 高 |
| `test_add_scalar` | 标量加法 | 高 |
| `test_sub_mul_div` | 减法、乘法、除法正确性 | 高 |
| `test_neg` | 一元负号 | 中 |
| `test_abs_sign` | abs, sign 行为 | 中 |
| `test_sin_cos_exp_ln` | 三角/指数/对数精度 | 高 |
| `test_complex_elementwise` | 复数逐元素运算 | 中 |
| `test_bool_not` | bool 逻辑非 | 中 |
| `test_compare_eq_ne` | 等于/不等于比较 | 高 |
| `test_compare_lt_gt` | 小于/大于比较 | 中 |
| `test_compound_assign` | +=, -=, *=, /= 原地运算 | 高 |
| `test_square_reciprocal` | square, reciprocal | 中 |

### 5.3 test_broadcast.rs

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_broadcast_scalar` | 标量与任意维度广播 | 高 |
| `test_broadcast_row_col` | 行/列向量广播到矩阵 | 高 |
| `test_broadcast_left_pad` | 维度不足左侧补 1 | 高 |
| `test_broadcast_incompatible` | 不兼容形状返回错误 | 高 |
| `test_broadcast_view_readonly` | 广播视图为只读 | 高 |
| `test_broadcast_inplace` | a += b 中 b 可广播 | 中 |
| `test_broadcast_zero_stride` | 广播后步长为 0 | 中 |

### 5.4 test_index.rs

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_multi_dim_index` | [i, j, k] 多维索引 | 高 |
| `test_index_out_of_bounds` | 越界 panic | 高 |
| `test_slice_range` | 范围切片 | 高 |
| `test_slice_negative_index` | 负索引 | 中 |
| `test_slice_step` | 步长切片 | 中 |

### 5.5 test_construction.rs

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_zeros_ones_full` | zeros, ones, full 构造 | 高 |
| `test_eye_identity` | 单位矩阵 | 高 |
| `test_from_vec_slice` | from_vec, from_slice 构造 | 高 |
| `test_from_fn` | from_fn 函数构造 | 中 |
| `test_arange` | arange 序列 | 中 |
| `test_linspace` | linspace 序列 | 中 |
| `test_from_fixed_array` | 从固定数组构造 | 中 |

### 5.6 test_reduction.rs

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_sum_global` | 全局 sum | 高 |
| `test_sum_axis` | 沿轴 sum | 高 |
| `test_sum_keepdims` | sum 保留被归约轴为长度 1 | 中 |
| `test_sum_empty` | 空数组 sum 返回加法单位元 | 高 |
| `test_sum_nan` | sum 含 NaN 结果为 NaN | 中 |
| `test_unique_sorted` | unique 返回排序后不重复元素 | 高 |
| `test_unique_integers` | 整数 unique | 中 |
| `test_unique_complex` | 复数 unique | 中 |
| `test_dot_product` | 向量内积 | 高 |
| `test_dot_complex` | 复数内积共轭线性 | 高 |
| `test_dot_shape_mismatch` | 内积维度不匹配返回错误 | 高 |
| `test_integer_sum_overflow` | 整数 sum 溢出视为不可恢复错误 | 中 |

### 5.7 test_shape_ops.rs

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_reshape_contiguous` | 连续数组 reshape 零拷贝 | 高 |
| `test_reshape_element_count` | 元素总数不匹配返回错误 | 高 |
| `test_transpose_2d` | 2D 转置 | 高 |
| `test_transpose_high_dim` | 高维转置 | 中 |

### 5.8 test_conversion.rs

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_cast_f32_to_f64` | f32→f64 精度保持 | 高 |
| `test_cast_f64_to_f32` | f64→f32 精度截断 | 高 |
| `test_cast_real_to_complex` | 实数→复数（虚部为 0） | 中 |
| `test_cast_nan_to_int` | NaN→整数行为 | 中 |
| `test_cast_bool_numeric` | bool↔数值转换 | 中 |
| `test_copy_to_fill` | copy_to, fill | 中 |
| `test_is_close_allclose` | 近似比较 | 中 |
| `test_clip` | 裁剪操作 | 中 |

### 5.9 test_ffi.rs

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_as_ptr` | as_ptr 返回正确指针 | 高 |
| `test_as_mut_ptr` | as_mut_ptr 返回可变指针 | 高 |
| `test_strides_bytes` | strides_bytes 正确转换 | 高 |
| `test_lda` | lda 返回 leading dimension | 中 |
| `test_is_blas_compatible` | BLAS 兼容性检查 | 高 |
| `test_from_raw_parts_roundtrip` | into_raw_parts → from_raw_parts 往返 | 高 |
| `test_index_to_offset` | index_to_offset 正确计算 | 高 |

### 5.10 test_parallel.rs

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_par_sum_consistency` | 并行 sum 与串行 sum 结果一致（参见 `09-parallel-backend.md §7`） | 高 |
| `test_par_add_consistency` | 并行 add 与串行 add 结果一致 | 高 |
| `test_parallel_read` | 多线程并发只读访问安全（参见 `25-thread-safety.md §4.5`） | 高 |
| `test_no_nested_parallel` | 嵌套并行被拒绝 | 中 |

### 5.11 test_simd.rs

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_simd_add_consistency` | SIMD add 与标量 add 结果一致（参见 `08-simd-backend.md §7`） | 高 |
| `test_simd_sum_consistency` | SIMD sum 与标量 sum 结果一致 | 高 |
| `test_simd_fallback_small` | 小数组 SIMD 回退到标量 | 中 |

### 5.12 test_no_std.rs

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_no_std_compile` | `--no-default-features --features alloc` 编译通过（参见 `01-architecture-overview.md §6`） | 高 |

### 5.13 test_error.rs

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_shape_mismatch_error` | 不兼容形状运算返回 ShapeMismatch（参见 `26-error-handling.md §4.1`） | 高 |
| `test_broadcast_error` | 不可广播返回 BroadcastError | 高 |
| `test_invalid_shape_error` | reshape 元素数不匹配返回 InvalidShape | 高 |
| `test_invalid_axis_error` | 轴越界返回 InvalidAxis | 高 |
| `test_dimension_mismatch_error` | 维度互转失败返回 DimensionMismatch | 高 |
| `test_error_display` | 所有错误类型的 Display 包含上下文 | 中 |

---

## 6. 边界测试场景

### 6.1 边界覆盖（参照需求说明书 §28.4）

| 边界类别 | 测试场景 | 覆盖的操作 |
|----------|----------|-----------|
| 空张量 | shape 含 0 的数组（如 `[0, 3]`） | 构造、迭代、归约、形状操作、索引 |
| 单元素 | shape 全为 1（如 `[1, 1]`） | 所有运算、归约、广播 |
| 大张量 | 元素数 > 1M | 归约精度、并行正确性 |
| 极端值（NaN） | 含 NaN 的浮点数组 | 归约、比较、cumsum |
| 极端值（Inf） | 含 Inf 的浮点数组 | 算术、归约、cast |
| 极端值（Subnormal） | 含次正规数 | 算术精度 |
| 非连续布局 | 转置/切片后的视图 | 所有运算、迭代、归约 |
| 高维（≥4 维） | 4D~6D 数组 | 形状操作、广播、迭代 |

### 6.2 边界测试示例

```rust
#[test]
fn test_empty_tensor_properties() {
    let t = Tensor2::<f64>::zeros([0, 5]);
    assert!(t.is_empty());
    assert_eq!(t.len(), 0);
    assert_eq!(t.ndim(), 2);
    assert_eq!(t.shape(), &[0, 5]);
    assert_eq!(t.iter().count(), 0);
}

#[test]
fn test_empty_tensor_sum() {
    let t = Tensor1::<f64>::zeros([0]);
    assert_eq!(t.sum(), 0.0); // additive identity
}

#[test]
fn test_single_element() {
    let t = Tensor0::from(42.0f64);
    assert_eq!(t.len(), 1);
    assert_eq!(t.sum(), 42.0);
}

#[test]
fn test_nan_sum_propagation() {
    let t = Tensor1::from_vec(vec![1.0, f64::NAN, 3.0]);
    assert!(t.sum().is_nan());
}

#[test]
fn test_high_dim_operations() {
    let t4 = Tensor::<f64, Ix4>::zeros([2, 3, 4, 5]);
    assert_eq!(t4.len(), 120);
    assert_eq!(t4.sum(), 0.0);

    let t6 = Tensor::<f64, Ix6>::ones([2, 2, 2, 2, 2, 2]);
    assert_eq!(t6.len(), 64);
    assert_eq!(t6.sum(), 64.0);
}
```

---

## 7. 数值精度规范

### 7.1 IEEE 754 精度要求

| 运算类别 | f64 容差 | f32 容差 | 参考实现 |
|----------|----------|----------|----------|
| 加减乘 | 精确 (0 ULP) | 精确 (0 ULP) | 直接计算 |
| 归约 (sum) | rtol < 1e-15 | rtol < 1e-6 | Kahan 求和 |
| 超越函数 (sin/cos) | rtol < 1e-14 | rtol < 1e-5 | `std::f64::sin` |
| 超越函数 (exp/ln) | rtol < 1e-14 | rtol < 1e-5 | `std::f64::exp` |

### 7.2 浮点比较方式

所有浮点测试使用 **相对容差**（rtol）比较：

```rust
/// Compare two floats with relative tolerance.
pub fn rtol_eq(actual: f64, expected: f64, rtol: f64) -> bool {
    if expected == 0.0 {
        actual.abs() < rtol
    } else {
        (actual - expected).abs() / expected.abs() < rtol
    }
}
```

---

## 8. 并行与 SIMD 测试

### 8.1 并行无数据竞争

线程安全测试方案（参见 `25-thread-safety.md §7`）：

| 方式 | 说明 |
|------|------|
| `thread::scope` | 使用 scoped thread 并发访问 TensorView（只读） |
| ArcTensor | 多线程共享 ArcTensor 并发读取 |
| `loom` 模型检查 | 对 ArcRepr make_mut 的并发安全性进行模型检查（可选） |

### 8.2 并行归约一致性

```rust
#[test]
fn test_par_sum_consistency() {
    let n = 1_000_000;
    let t = Tensor1::<f64>::from_fn([n], |[i]| (i as f64).sin());

    let serial_sum = t.sum();

    #[cfg(feature = "parallel")]
    {
        let par_sum = t.par_sum();
        let rtol = (serial_sum - par_sum).abs() / serial_sum.abs();
        assert!(rtol < 1e-14, "parallel sum rtol: {}", rtol);
    }
}
```

### 8.3 SIMD 结果一致性

```rust
#[test]
fn test_simd_add_consistency() {
    let a = Tensor1::<f64>::from_fn([1024], |[i]| (i as f64).sin());
    let b = Tensor1::<f64>::from_fn([1024], |[i]| (i as f64).cos());

    let result = &a + &b;

    // Verify against scalar loop
    for i in 0..1024 {
        let expected = a[[i]] + b[[i]];
        let diff = (result[[i]] - expected).abs();
        assert!(diff < 1e-15, "SIMD add mismatch at {}: diff={}", i, diff);
    }
}
```

---

## 9. 属性测试不变量设计

### 9.1 不变量清单

| 不变量 | 测试方法 | 优先级 |
|--------|----------|--------|
| `reshape` 保元素数 | 随机形状 → reshape → `len()` 不变 | 高 |
| `reshape` + `reshape` 回到原形状 | 连续数组 reshape 再 reshape 回原形状（参见 `16-shape-ops.md §4`） | 高 |
| `sum` 保加法单位元 | 空数组 sum == 0（参见 `13-reduction.md §4`） | 高 |
| `transpose` 自反性 | `t.t().t()` == `t`（参见 `16-shape-ops.md §4`） | 高 |
| 加法交换律 | `a + b` == `b + a`（近似） | 中 |
| 加法结合律 | `(a + b) + c` == `a + (b + c)`（近似） | 中 |
| `unique` 保元素数 | `unique(a).len()` ≤ `a.len()` | 中 |
| `unique` 不含重复 | unique 结果无相邻相等元素 | 中 |
| 广播形状一致性 | 广播后形状 = max 对应维度（参见 `15-broadcast.md §5`） | 高 |

### 9.2 属性测试框架

使用 `proptest` 进行随机测试：

```rust
// tests/property/tensor_props.rs
use proptest::prelude::*;

proptest! {
    #[test]
    fn prop_reshape_preserves_len(r in 1..64usize, c in 1..64usize) {
        let t = Tensor2::<f64>::zeros([r, c]);
        let flat = t.reshape([r * c]);
        match flat {
            Ok(f) => prop_assert_eq!(f.len(), t.len()),
            Err(_) => prop_assert!(false, "contiguous reshape should succeed"),
        }
    }

    #[test]
    fn prop_transpose_involution(r in 1..32usize, c in 1..32usize) {
        let t = Tensor2::from_fn([r, c], |[i, j]| (i * c + j) as f64);
        let tt = t.t().t().to_owned();
        for (a, b) in t.iter().zip(tt.iter()) {
            prop_assert_eq!(a, b);
        }
    }

    #[test]
    fn prop_add_commutative(data in proptest::collection::vec(any::<f64>(), 1..256)) {
        let a = Tensor1::from_vec(data.clone());
        let b = Tensor1::from_vec(data.iter().map(|x| x + 1.0).collect());
        let ab = &a + &b;
        let ba = &b + &a;
        for (x, y) in ab.iter().zip(ba.iter()) {
            let diff = (x - y).abs();
            prop_assert!(diff < 1e-10 || (x.is_nan() && y.is_nan()));
        }
    }
}
```

---

## 10. 实现任务拆分

### Wave 1: 基础设施

- [ ] **T1**: 创建 `tests/` 目录结构和 `common/` 共享工具
  - 文件: `tests/common/mod.rs`, `tests/common/assertions.rs`, `tests/common/generators.rs`
  - 内容: assert_tensor_close 宏、标准形状常量、数据生成函数
  - 测试: 编译通过
  - 前置: 无
  - 预计: 10 min

### Wave 2: 核心功能测试（可并行）

- [ ] **T2**: 实现 `tests/test_tensor.rs`
  - 文件: `tests/test_tensor.rs`
  - 内容: 张量核心功能（shape/strides/view/to_owned/type_aliases/debug_display/arc）
  - 测试: `cargo test --test test_tensor`
  - 前置: T1
  - 预计: 10 min

- [ ] **T3**: 实现 `tests/test_ops.rs`
  - 文件: `tests/test_ops.rs`
  - 内容: 逐元素运算（算术/数学/比较/逻辑/原地）
  - 测试: `cargo test --test test_ops`
  - 前置: T1
  - 预计: 10 min

- [ ] **T4**: 实现 `tests/test_broadcast.rs`
  - 文件: `tests/test_broadcast.rs`
  - 内容: 广播机制（标量/行/列/不兼容/只读/原地）
  - 测试: `cargo test --test test_broadcast`
  - 前置: T1
  - 预计: 10 min

- [ ] **T5**: 实现 `tests/test_index.rs`
  - 文件: `tests/test_index.rs`
  - 内容: 索引操作（多维索引/越界/切片/负索引/步长）
  - 测试: `cargo test --test test_index`
  - 前置: T1
  - 预计: 10 min

- [ ] **T6**: 实现 `tests/test_construction.rs`
  - 文件: `tests/test_construction.rs`
  - 内容: 构造方法（zeros/ones/eye/from_vec/from_fn/arange/linspace）
  - 测试: `cargo test --test test_construction`
  - 前置: T1
  - 预计: 10 min

- [ ] **T7**: 实现 `tests/test_reduction.rs`
  - 文件: `tests/test_reduction.rs`
  - 内容: 归约运算（sum/sum_axis/keepdims/empty/NaN/unique/dot/overflow）
  - 测试: `cargo test --test test_reduction`
  - 前置: T1
  - 预计: 10 min

- [ ] **T8**: 实现 `tests/test_shape_ops.rs`
  - 文件: `tests/test_shape_ops.rs`
  - 内容: 形状操作（transpose/reshape/高维）
  - 测试: `cargo test --test test_shape_ops`
  - 前置: T1
  - 预计: 10 min

- [ ] **T9**: 实现 `tests/test_conversion.rs`
  - 文件: `tests/test_conversion.rs`
  - 内容: 类型转换（cast/存储模式转换/clip/is_close）
  - 测试: `cargo test --test test_conversion`
  - 前置: T1
  - 预计: 10 min

- [ ] **T10**: 实现 `tests/test_error.rs`
  - 文件: `tests/test_error.rs`
  - 内容: 错误处理（所有错误类型验证/display 输出）
  - 测试: `cargo test --test test_error`
  - 前置: T1
  - 预计: 10 min

### Wave 3: 特化测试（可并行）

- [ ] **T11**: 实现 `tests/test_ffi.rs`
  - 文件: `tests/test_ffi.rs`
  - 内容: FFI 集成（指针/BLAS/roundtrip/offset）
  - 测试: `cargo test --test test_ffi`
  - 前置: T2
  - 预计: 10 min

- [ ] **T12**: 实现 `tests/test_parallel.rs`
  - 文件: `tests/test_parallel.rs`
  - 内容: 并行计算一致性（par_sum/par_add/并发读取/嵌套禁止）
  - 测试: `cargo test --test test_parallel --features parallel`
  - 前置: T3, T7
  - 预计: 10 min

- [ ] **T13**: 实现 `tests/test_simd.rs`
  - 文件: `tests/test_simd.rs`
  - 内容: SIMD 结果一致性（add/sum/fallback）
  - 测试: `cargo test --test test_simd --features simd`
  - 前置: T3, T7
  - 预计: 10 min

- [ ] **T14**: 实现 `tests/test_no_std.rs`
  - 文件: `tests/test_no_std.rs`
  - 内容: no_std 编译验证
  - 测试: `cargo test --test test_no_std --no-default-features --features alloc`
  - 前置: T2
  - 预计: 5 min

### Wave 4: 属性测试

- [ ] **T15**: 实现 `tests/property/` 属性测试模块
  - 文件: `tests/property/mod.rs`, `tests/property/tensor_props.rs`, `tests/property/ops_props.rs`, `tests/property/shape_props.rs`
  - 内容: reshape 保元素数/transpose 自反/加法交换律/unique 不含重复
  - 测试: `cargo test --test property`
  - 前置: T3, T7, T8
  - 预计: 10 min

### Wave 5: CI 集成

- [ ] **T16**: 配置 CI 测试矩阵
  - 文件: `.github/workflows/test.yml`
  - 内容: 多 OS × 多 Rust 版本 × 多 feature 组合，覆盖率门禁
  - 测试: CI 触发运行
  - 前置: T1-T15
  - 预计: 10 min

### 并行执行分组图

```
Wave 1: [T1]
           │
Wave 2: [T2] [T3] [T4] [T5] [T6] [T7] [T8] [T9] [T10]
           │    │              │    │
           └────┴──────────────┴────┘
                       │
Wave 3:       [T11] [T12] [T13] [T14]
                  │
Wave 4:       [T15]
                  │
Wave 5:       [T16]
```

---

## 11. 测试计划

### 11.1 测试分类表

| 类型 | 位置 | 目的 |
|------|------|------|
| 单元测试 | `#[cfg(test)] mod tests` | 验证单个函数/方法 |
| 集成测试 | `tests/` | 验证跨模块交互 |
| 边界测试 | 集成测试中标注 | 空张量、单元素、NaN/Inf、非连续、高维 |
| 属性测试 | `tests/property/` | 随机生成验证不变量 |

### 11.2 CI 测试矩阵

```yaml
# .github/workflows/test.yml
test:
    strategy:
        matrix:
            os: [ubuntu-latest, macos-latest, windows-latest]
            rust: [1.85.0, stable]
            features:
                - ""
                - "--features parallel"
                - "--features simd"
                - "--all-features"
                - "--no-default-features --features alloc"
    steps:
        - name: Unit + Integration tests
          run: cargo test --lib --tests ${{ matrix.features }}

        - name: Doc tests
          run: cargo test --doc ${{ matrix.features }}
```

---

## 12. ADR 决策记录

### 决策 1：按测试领域分文件

| 属性 | 值 |
|------|-----|
| 决策 | 按测试领域（ops/broadcast/reduction 等）而非按源码模块分文件 |
| 理由 | 集成测试关注跨模块行为；独立运行；编译并行化；失败定位清晰 |
| 替代方案 | 按源码模块分（test_dimension.rs, test_storage.rs）— 放弃，跨模块边界模糊 |

### 决策 2：使用 proptest 进行属性测试

| 属性 | 值 |
|------|-----|
| 决策 | 使用 proptest 进行属性测试 |
| 理由 | 比 quickcheck 更好的失败案例缩小（shrinking）；Rust 生态成熟 |
| 替代方案 | quickcheck — 可接受，但 shrinking 能力较弱 |

### 决策 3：浮点比较使用相对容差

| 属性 | 值 |
|------|-----|
| 决策 | 浮点测试使用 rtol（相对容差）而非 atol（绝对容差） |
| 理由 | 不同数量级的值需要不同精度基准；rtol 更符合科学计算习惯 |
| 替代方案 | 仅用 atol — 放弃，大数值时精度要求过严，小数值时过松 |

### 决策 4：并行一致性测试为必须项

| 属性 | 值 |
|------|-----|
| 决策 | 并行归约和逐元素运算必须与串行结果完全一致 |
| 理由 | 需求说明书 §28.5 明确要求并行归约与单线程一致 |
| 替代方案 | 允许有限误差 — 放弃，违反需求 |

### 决策 5：no_std 测试为编译验证

| 属性 | 值 |
|------|-----|
| 决策 | no_std 测试仅验证编译通过，不运行完整测试 |
| 理由 | 集成测试依赖 std（println/assert 格式化），完整运行需单独环境 |
| 替代方案 | 运行完整测试 — 放弃，CI 复杂度过高 |

---

## 版本历史

| 版本 | 日期 |
|------|------|
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-08 |

---

*本文档由 Xenon 维护。如有问题请提交 Issue 或 PR。*
