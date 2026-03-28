# Xenon 集成测试设计文档

> **文档版本**: v1.0  
> **最后更新**: 2026-03-28  
> **模块路径**: `tests/`  
> **需求来源**: require-v18.md §17

---

## 1. 模块概述

### 1.1 职责定义

集成测试模块负责验证 Xenon 库各组件协同工作的正确性。

| 职责 | 说明 |
|------|------|
| 跨模块验证 | 验证维度、存储、布局、运算等模块的协同行为 |
| 边界覆盖 | 覆盖空数组、单元素、大数组、NaN/Inf、非连续布局、高维等边界 |
| 数值精度 | 验证归约和超越函数满足精度要求 |
| 回归防护 | 防止已修复的 bug 再次出现 |

### 1.2 设计目标

| 目标 | 实现方式 |
|------|----------|
| **全面性** | 覆盖 require-v18.md 中所有 API 的关键行为 |
| **独立性** | 每个测试文件可独立运行，无跨文件依赖 |
| **可读性** | 测试名称描述预期行为，失败信息包含上下文 |
| **快速反馈** | 单元测试 < 30s，完整集成测试 < 5min |

### 1.3 需求来源

来自 `require-v18.md` 第 17 节：

- 测试类型：单元测试 + 集成测试 + 边界测试
- 行覆盖率：≥ 80%
- 边界覆盖：空张量、单元素、大张量、NaN/Inf/subnormal、非连续布局、高维(≥4维)

---

## 2. 测试架构

### 2.1 测试分层

```
测试层次
├── L1: 单元测试（src/ 内 #[cfg(test)] mod tests）
│   └── 每个源文件内部，测试单个函数/方法的行为
│
├── L2: 集成测试（tests/ 目录）
│   └── 跨模块交互，从公开 API 角度验证
│
└── L3: 属性测试（tests/property/）
    └── 使用 quickcheck/proptest 验证代数性质
```

### 2.2 目录结构

```
tests/
├── dimension.rs            — 维度系统集成测试
├── element.rs              — 元素类型体系测试
├── complex.rs              — 复数类型测试
├── storage.rs              — 存储系统测试
├── layout.rs               — 内存布局测试
├── tensor_core.rs          — TensorBase 核心功能测试
├── iterator.rs             — 迭代器测试
├── elementwise.rs          — 逐元素运算测试
├── matrix_ops.rs           — 矩阵运算测试
├── reduction.rs            — 归约运算测试
├── broadcast.rs            — 广播测试
├── shape_ops.rs            — 形状操作测试
├── indexing.rs             — 索引操作测试
├── construction.rs         — 构造与转换测试
├── ffi.rs                  — FFI 集成测试
├── workspace.rs            — 临时工作空间测试
├── thread_safety.rs        — 线程安全测试
├── error_handling.rs       — 错误处理测试
├── edge_cases.rs           — 统一边界情况测试
├── precision.rs            — 数值精度测试
│
├── property/               — 属性测试
│   ├── mod.rs
│   ├── algebraic.rs        — 代数性质（交换律、结合律、分配律）
│   ├── broadcast_prop.rs   — 广播性质
│   └── layout_prop.rs      — 布局不变量
│
└── common/                 — 共享测试工具
    ├── mod.rs
    ├── assertions.rs       — 自定义断言宏
    └── generators.rs       — 测试数据生成器
```

### 2.3 共享测试工具

```rust
// tests/common/mod.rs
pub mod assertions;
pub mod generators;
```

```rust
// tests/common/assertions.rs

/// Assert two tensors are element-wise approximately equal.
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

```rust
// tests/common/generators.rs

/// Generate tensors of various shapes for parameterized testing.
pub fn standard_shapes_2d() -> Vec<(usize, usize)> {
    vec![
        (0, 0), (1, 1), (1, 5), (5, 1),
        (3, 4), (4, 3), (8, 8), (64, 64),
    ]
}

/// Generate a non-contiguous tensor (transposed view).
pub fn non_contiguous_2d(rows: usize, cols: usize) -> Tensor2<f64> {
    let t = Tensor2::<f64>::from_fn([cols, rows], |[i, j]| (i * rows + j) as f64);
    t.t().to_owned()
}
```

---

## 3. 测试分类详细设计

### 3.1 维度系统测试 (dimension.rs)

| 测试 | 验证内容 |
|------|----------|
| `test_ix0_properties` | Ix0: ndim=0, shape=&[], 元素数=1 |
| `test_ix1_to_ix6_properties` | IxN: ndim=N, shape 长度=N |
| `test_ixdyn_properties` | IxDyn: 动态 ndim, 堆分配 |
| `test_static_to_dynamic` | Ix0-Ix6 → IxDyn 总是成功 |
| `test_dynamic_to_static_success` | IxDyn(ndim=N) → IxN 成功 |
| `test_dynamic_to_static_fail` | IxDyn(ndim≠N) → IxN 返回 DimensionMismatch |
| `test_ix0_ixdyn_roundtrip` | Ix0 → IxDyn → Ix0 往返转换 |
| `test_dimension_equality` | 相同 shape 的不同维度类型比较 |

### 3.2 元素类型体系测试 (element.rs)

| 测试 | 验证内容 |
|------|----------|
| `test_element_zero_one` | 所有 Element 类型的 zero()/one() |
| `test_numeric_arithmetic` | 数值类型四则运算正确性 |
| `test_bool_not_numeric` | bool 不实现 Numeric（编译时保证，此为文档测试） |
| `test_real_scalar_math` | f32/f64 数学函数（sin, cos, exp, ln 等） |
| `test_complex_scalar_ops` | Complex 复数运算（conj, norm, arg 等） |
| `test_nan_propagation` | NaN 传播行为（min, max, sum） |
| `test_inf_arithmetic` | Inf 算术遵循 IEEE 754 |

### 3.3 复数类型测试 (complex.rs)

| 测试 | 验证内容 |
|------|----------|
| `test_complex_creation` | Complex::new(re, im) 构造 |
| `test_complex_arithmetic` | 加减乘除正确性 |
| `test_complex_real_interop` | Complex + 实数、实数 + Complex |
| `test_complex_conj_norm_arg` | conj(), norm()(hypot), arg() |
| `test_complex_from_polar` | from_polar 与 norm/arg 往返 |
| `test_complex_repr_c` | size_of, align_of, 内存布局验证 |
| `test_complex_partial_eq` | NaN != NaN, 正常值相等 |
| `test_complex_no_partial_ord` | 编译时保证无 PartialOrd（文档测试） |
| `test_complex_approx_eq` | approx_eq 逐分量判断 |

### 3.4 存储系统测试 (storage.rs)

| 测试 | 验证内容 |
|------|----------|
| `test_owned_deep_copy` | Owned clone 为深拷贝 |
| `test_view_metadata_copy` | View clone 为 O(1) 元数据拷贝 |
| `test_view_mut_not_clone` | ViewMut 不可 clone（编译时保证） |
| `test_arc_shallow_copy` | ArcRepr clone 为引用计数+1 |
| `test_arc_make_mut_single_ref` | 单引用 make_mut 不拷贝 |
| `test_arc_make_mut_multi_ref` | 多引用 make_mut 触发深拷贝 |
| `test_arc_make_mut_alignment` | make_mut 新分配使用 64 字节对齐 |
| `test_owned_alignment` | Owned 存储 64 字节对齐 |

### 3.5 内存布局测试 (layout.rs)

| 测试 | 验证内容 |
|------|----------|
| `test_f_contiguous_default` | 默认构造为 F-contiguous |
| `test_c_contiguous_explicit` | 指定 C-order 构造 |
| `test_layout_flags_creation` | 创建时标志位正确 |
| `test_layout_flags_after_transpose` | 转置后标志位更新 |
| `test_layout_flags_after_slice` | 切片后标志位更新 |
| `test_alignment_64_byte` | 默认 64 字节对齐 |
| `test_alignment_small_array` | 小数组对齐降级 |
| `test_zero_stride_broadcast` | 广播后零步长标志 |
| `test_neg_stride_flip` | 翻转后负步长标志 |
| `test_1d_both_contiguous` | 1D 数组同时 F 和 C 连续 |
| `test_scalar_both_contiguous` | 0D 标量同时 F 和 C 连续 |

### 3.6 张量核心测试 (tensor_core.rs)

| 测试 | 验证内容 |
|------|----------|
| `test_tensor_shape_strides` | shape(), strides(), ndim(), len() |
| `test_tensor_is_empty` | 空数组 is_empty() = true |
| `test_tensor_view_creation` | view(), view_mut() 创建 |
| `test_tensor_to_owned` | to_owned() 深拷贝 |
| `test_tensor_into_owned` | into_owned() 消耗转换 |
| `test_type_aliases` | Tensor0-Tensor3, TensorD 别名正确 |
| `test_tensor_debug_display` | Debug/Display 格式化输出 |

### 3.7 迭代器测试 (iterator.rs)

| 测试 | 验证内容 |
|------|----------|
| `test_iter_elements_order` | 元素迭代按内存布局顺序 |
| `test_iter_axis` | 轴迭代产出正确子视图 |
| `test_iter_window` | 窗口迭代：窗口数 = shape - window_size + 1 |
| `test_iter_window_no_incomplete` | 不产出不完整窗口 |
| `test_iter_indexed` | 带索引迭代，索引正确 |
| `test_zip_same_shape` | 同形状 zip 同步产出 |
| `test_zip_broadcast` | 可广播形状 zip 自动广播 |
| `test_zip_incompatible` | 不可广播形状 zip 返回 BroadcastError |
| `test_iter_empty_array` | 空数组迭代立即结束 |
| `test_iter_f_order` | 指定 F-order 遍历 |
| `test_iter_c_order` | 指定 C-order 遍历 |

### 3.8 逐元素运算测试 (elementwise.rs)

| 测试 | 验证内容 |
|------|----------|
| `test_add_same_shape` | 同形状加法 |
| `test_add_broadcast` | 广播加法 |
| `test_add_scalar` | 标量加法 |
| `test_sub_mul_div` | 减法、乘法、除法 |
| `test_compound_assign` | +=, -=, *=, /= |
| `test_neg` | 一元负号 |
| `test_trig_functions` | sin, cos, tan 精度 |
| `test_exp_ln` | exp, ln 精度 |
| `test_abs_sign` | abs, sign 行为 |
| `test_floor_ceil_round` | 取整函数 |
| `test_square_reciprocal` | square, reciprocal |
| `test_pow` | pow 运算 |
| `test_complex_elementwise` | 复数逐元素运算 |
| `test_bool_no_arithmetic` | bool 不支持算术（编译时保证） |

### 3.9 矩阵运算测试 (matrix_ops.rs)

| 测试 | 验证内容 |
|------|----------|
| `test_matvec_basic` | 基本矩阵-向量乘法 |
| `test_matvec_f_order` | F-order 矩阵 matvec |
| `test_matvec_c_order` | C-order 矩阵 matvec |
| `test_dot_product` | 向量内积 |
| `test_outer_product` | 向量外积 |
| `test_batch_matvec` | 批量 matvec 维度约定 |
| `test_batch_dot` | 批量内积 |
| `test_batch_broadcast` | batch 维度广播 |
| `test_matvec_noncontiguous` | 非连续矩阵 matvec |

### 3.10 归约运算测试 (reduction.rs)

| 测试 | 验证内容 |
|------|----------|
| `test_sum_global` | 全局 sum |
| `test_sum_axis` | 沿轴 sum |
| `test_prod_global` | 全局 prod |
| `test_mean_var_std` | mean, var(ddof=0), std |
| `test_min_max` | min, max |
| `test_argmin_argmax` | argmin, argmax（多值取第一个） |
| `test_all_any` | all, any（bool 数组） |
| `test_cumsum_cumprod` | 累积求和/乘积 |
| `test_cumsum_nan` | cumsum 遇 NaN 传播 |
| `test_sum_nan` | sum 含 NaN 结果为 NaN |
| `test_min_max_nan` | min/max NaN 传播 |
| `test_empty_array_reduction` | 空数组 min/max 返回 EmptyArray 错误 |
| `test_integer_overflow_panic` | 整数 sum 溢出 panic |
| `test_unique` | unique 排序去重 |
| `test_unique_counts` | unique_counts 正确计数 |
| `test_unique_inverse` | unique_inverse 可重建原数组 |
| `test_bincount` | bincount 基本行为 |
| `test_bincount_negative_panic` | bincount 负值 panic |
| `test_histogram` | histogram 基本行为 |

### 3.11 广播测试 (broadcast.rs)

| 测试 | 验证内容 |
|------|----------|
| `test_broadcast_scalar` | 标量与任意维度广播 |
| `test_broadcast_row_col` | 行/列向量广播到矩阵 |
| `test_broadcast_left_pad` | 维度不足左侧补 1 |
| `test_broadcast_incompatible` | 不兼容形状返回 BroadcastError |
| `test_broadcast_view_readonly` | 广播视图为只读 |
| `test_broadcast_inplace` | a += b 中 b 可广播，a 不可 |
| `test_broadcast_zero_stride` | 广播后步长为 0 |

### 3.12 形状操作测试 (shape_ops.rs)

| 测试 | 验证内容 |
|------|----------|
| `test_reshape_contiguous` | 连续数组 reshape 零拷贝 |
| `test_reshape_noncontiguous_error` | 非连续数组 reshape 返回 LayoutMismatch |
| `test_reshape_element_count` | 元素总数不匹配返回 InvalidShape |
| `test_transpose_2d` | 2D 转置 |
| `test_permute_axes` | 多维轴排列 |
| `test_swapaxes` | 交换两轴 |
| `test_moveaxis` | 移动轴位置 |
| `test_squeeze` | 移除 size-1 维度 |
| `test_unsqueeze` | 添加 size-1 维度 |
| `test_index_axis` | 沿轴索引降维 |
| `test_index_axis_blas_compat` | F-contiguous batch 索引保持 F-contiguous |
| `test_unstack` | 沿轴拆分为视图列表 |
| `test_unstack_empty_axis` | 空轴返回空 Vec |
| `test_split_indices` | 按索引分割 |
| `test_chunk_even` | 均匀分块 |
| `test_chunk_uneven` | 不整除分块（前几块多 1） |
| `test_chunk_zero` | n_chunks=0 返回空 Vec |
| `test_chunk_exceeds_axis` | n_chunks > 轴长度 |
| `test_cat_axis` | 沿轴拼接 |
| `test_stack_new_axis` | 新轴堆叠 |
| `test_pad_constant` | 常量填充 |
| `test_pad_edge` | 边缘填充 |
| `test_pad_reflect` | 反射填充 |
| `test_repeat_tile` | repeat/tile 基本行为 |
| `test_repeat_zero` | reps 含 0 结果为空 |
| `test_flatten_contiguous` | 连续数组 flatten 零拷贝 |
| `test_flatten_noncontiguous` | 非连续数组 flatten 需拷贝 |

### 3.13 索引操作测试 (indexing.rs)

| 测试 | 验证内容 |
|------|----------|
| `test_multi_dim_index` | [i, j, k] 索引 |
| `test_index_out_of_bounds_panic` | 越界 panic |
| `test_slice_macro` | s![] 宏基本语法 |
| `test_slice_negative_index` | 负索引 |
| `test_slice_step` | 步长切片 |
| `test_take` | take 操作 |
| `test_take_along_axis` | take_along_axis |
| `test_mask_bool` | bool mask 索引 |
| `test_compress` | compress 操作 |
| `test_put` | put 操作 |
| `test_argwhere_nonzero` | argwhere/nonzero |
| `test_where_select` | where(condition, x, y) |
| `test_where_broadcast` | where 三参数广播 |

### 3.14 构造与转换测试 (construction.rs)

| 测试 | 验证内容 |
|------|----------|
| `test_zeros_ones_full` | zeros, ones, full 构造 |
| `test_empty` | empty 构造（未初始化） |
| `test_eye_identity` | 单位矩阵 |
| `test_diag` | 对角矩阵 |
| `test_from_vec_slice_fn` | from_vec, from_slice, from_fn |
| `test_arange` | arange 序列 |
| `test_linspace` | linspace 序列 |
| `test_logspace` | logspace 序列 |
| `test_cast_precision` | cast 精度行为（所有转换方向） |
| `test_cast_nan_to_int` | NaN → 整数 = 0 |
| `test_cast_inf_to_int` | Inf → 整数 = MAX/MIN |
| `test_cast_bool_numeric` | bool ↔ 数值转换 |
| `test_cast_real_complex` | 实数 → 复数（虚部为 0） |
| `test_to_f_contiguous` | to_f_contiguous 返回 F-contiguous Owned |
| `test_to_c_contiguous` | to_c_contiguous 返回 C-contiguous Owned |
| `test_from_conversions` | Vec/slice/array → Tensor 的 From 转换 |
| `test_copy_to_fill` | copy_to, fill |
| `test_is_close_allclose` | 近似比较 |
| `test_clip` | 裁剪 |
| `test_flip` | flip/flipud/fliplr |
| `test_map_mapv` | map, mapv, mapv_inplace |

### 3.15 FFI 集成测试 (ffi.rs)

| 测试 | 验证内容 |
|------|----------|
| `test_as_ptr` | as_ptr 返回正确指针 |
| `test_as_mut_ptr` | as_mut_ptr 返回可变指针 |
| `test_strides_bytes` | strides_bytes 正确转换 |
| `test_lda` | lda 返回 leading dimension |
| `test_is_blas_compatible` | BLAS 兼容性检查 |
| `test_blas_layout` | blas_layout 返回 F/C/None |
| `test_from_raw_parts_roundtrip` | into_raw_parts → from_raw_parts 往返 |
| `test_index_to_ptr` | index_to_ptr 正确计算 |
| `test_index_to_offset` | index_to_offset 正确计算 |

### 3.16 临时工作空间测试 (workspace.rs)

| 测试 | 验证内容 |
|------|----------|
| `test_workspace_alloc` | 基本分配和释放 |
| `test_workspace_alignment` | 64 字节对齐 |
| `test_workspace_custom_alignment` | 自定义对齐（128 字节） |
| `test_workspace_grow` | 自动扩容 |
| `test_workspace_no_shrink` | 不缩容 |
| `test_workspace_split_at` | split_at 分割 |
| `test_workspace_recursive_split` | 递归分割 |
| `test_scratch_size_query` | scratch 查询 API |
| `test_scratch_combine` | scratch 需求合并（max/sum） |

### 3.17 线程安全测试 (thread_safety.rs)

| 测试 | 验证内容 |
|------|----------|
| `test_owned_send_sync` | Owned 可跨线程发送和共享 |
| `test_view_send_sync` | View 可跨线程发送和共享 |
| `test_view_mut_send_not_sync` | ViewMut 可发送但不可共享 |
| `test_arc_send_sync` | ArcRepr 可跨线程发送和共享 |
| `test_parallel_read` | 多线程并发只读访问 |
| `test_arc_concurrent_make_mut` | 并发 make_mut 无数据竞争 |


---

## 4. 边界情况测试 (edge_cases.rs)

### 4.1 统一边界覆盖

基于 `require-v18.md` 第 17.4 节，以下边界情况须在所有相关操作中覆盖：

| 边界类别 | 测试场景 | 覆盖的操作 |
|----------|----------|-----------|
| 空张量 | shape 含 0 的数组 | 构造、迭代、归约、形状操作、索引 |
| 单元素 | shape 全为 1 的数组 | 所有运算、归约、广播 |
| 大张量 | 元素数 > 1M | 归约精度、并行正确性 |
| NaN | 含 NaN 的浮点数组 | 归约、min/max、比较、cumsum |
| Inf | 含 Inf 的浮点数组 | 算术、归约、cast |
| Subnormal | 含次正规数的数组 | 算术精度 |
| 非连续布局 | 转置/切片后的视图 | 所有运算、迭代、归约 |
| 高维 (≥4维) | 4D-6D 数组 | 形状操作、广播、迭代 |

### 4.2 边界测试模板

```rust
// tests/edge_cases.rs

/// Parameterized edge case testing macro.
macro_rules! test_with_edge_cases {
    ($test_fn:ident, $op:expr) => {
        #[test]
        fn $test_fn() {
            // Empty tensor
            let empty = Tensor1::<f64>::zeros([0]);
            $op(&empty);

            // Single element
            let single = Tensor1::from_vec(vec![42.0f64]);
            $op(&single);

            // Contains NaN
            let with_nan = Tensor1::from_vec(vec![1.0, f64::NAN, 3.0]);
            $op(&with_nan);

            // Contains Inf
            let with_inf = Tensor1::from_vec(vec![1.0, f64::INFINITY, 3.0]);
            $op(&with_inf);

            // Non-contiguous (every other element)
            let full = Tensor1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
            let strided = full.slice(s![..;2]);
            $op(&strided);
        }
    };
}

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
    assert_eq!(t.sum(), 0.0); // sum of empty = additive identity
}

#[test]
fn test_empty_tensor_min_error() {
    let t = Tensor1::<f64>::zeros([0]);
    assert!(t.min().is_err()); // EmptyArray error
}

#[test]
fn test_nan_sum_propagation() {
    let t = Tensor1::from_vec(vec![1.0, f64::NAN, 3.0]);
    assert!(t.sum().is_nan());
}

#[test]
fn test_nan_min_propagation() {
    let t = Tensor1::from_vec(vec![1.0, f64::NAN, 3.0]);
    let m = t.min().unwrap();
    assert!(m.is_nan()); // NaN propagation
}

#[test]
fn test_high_dim_operations() {
    // 4D tensor
    let t4 = Tensor::<f64, Ix4>::zeros([2, 3, 4, 5]);
    assert_eq!(t4.len(), 120);
    assert_eq!(t4.sum(), 0.0);

    // 6D tensor
    let t6 = Tensor::<f64, Ix6>::ones([2, 2, 2, 2, 2, 2]);
    assert_eq!(t6.len(), 64);
    assert_eq!(t6.sum(), 64.0);
}
```

---

## 5. 数值精度测试 (precision.rs)

### 5.1 精度验证策略

| 运算类别 | f64 容差 | f32 容差 | 参考实现 |
|----------|----------|----------|----------|
| 加减乘 | 精确 (0.0) | 精确 (0.0) | 直接计算 |
| 归约 (sum) | 1e-15 | 1e-6 | Kahan 求和 |
| 归约 (prod) | 1e-15 | 1e-6 | 逐元素累乘 |
| 超越函数 (sin/cos) | 1e-14 | 1e-5 | std::f64::sin |
| 超越函数 (exp/ln) | 1e-14 | 1e-5 | std::f64::exp |
| mean/var/std | 1e-15 | 1e-6 | 两遍算法 |

### 5.2 精度测试示例

```rust
// tests/precision.rs

#[test]
fn test_sum_precision_f64() {
    // Worst case: many small values that accumulate error
    let n = 1_000_000;
    let t = Tensor1::from_fn([n], |[i]| 1.0f64 / (i as f64 + 1.0));

    let result = t.sum();

    // Kahan reference
    let mut kahan_sum = 0.0f64;
    let mut compensation = 0.0f64;
    for i in 0..n {
        let y = 1.0 / (i as f64 + 1.0) - compensation;
        let t = kahan_sum + y;
        compensation = (t - kahan_sum) - y;
        kahan_sum = t;
    }

    let rel_error = (result - kahan_sum).abs() / kahan_sum.abs();
    assert!(rel_error < 1e-15,
        "f64 sum relative error: {} (expected < 1e-15)", rel_error);
}

#[test]
fn test_sum_precision_f32() {
    let n = 100_000;
    let t = Tensor1::from_fn([n], |[i]| 1.0f32 / (i as f32 + 1.0));

    let result = t.sum();

    // f64 reference for f32 precision check
    let reference: f64 = (0..n).map(|i| 1.0f64 / (i as f64 + 1.0)).sum();

    let rel_error = ((result as f64) - reference).abs() / reference.abs();
    assert!(rel_error < 1e-6,
        "f32 sum relative error: {} (expected < 1e-6)", rel_error);
}

#[test]
fn test_trig_precision_f64() {
    let values = vec![0.0, 0.1, 0.5, 1.0, 2.0, 3.14159];
    let t = Tensor1::from_vec(values.clone());

    let sin_result = t.sin();
    for (i, &v) in values.iter().enumerate() {
        let expected = v.sin();
        let diff = (sin_result[[i]] - expected).abs();
        assert!(diff < 1e-14,
            "sin({}) error: {} (expected < 1e-14)", v, diff);
    }
}

#[test]
fn test_var_precision() {
    // Shifted data to test numerical stability
    let base = 1e8;
    let t = Tensor1::from_vec(vec![base + 1.0, base + 2.0, base + 3.0]);

    let v = t.var();
    let expected = 2.0 / 3.0; // var with ddof=0

    let rel_error = (v - expected).abs() / expected.abs();
    assert!(rel_error < 1e-15,
        "var relative error: {} (expected < 1e-15)", rel_error);
}
```

---

## 6. 属性测试 (property/)

### 6.1 代数性质测试 (algebraic.rs)

```rust
// tests/property/algebraic.rs
use quickcheck::quickcheck;

quickcheck! {
    // Addition commutativity: a + b == b + a
    fn prop_add_commutative(a_data: Vec<f64>, b_data: Vec<f64>) -> bool {
        if a_data.len() != b_data.len() || a_data.is_empty() { return true; }
        let n = a_data.len().min(b_data.len());
        let a = Tensor1::from_vec(a_data[..n].to_vec());
        let b = Tensor1::from_vec(b_data[..n].to_vec());
        let ab = &a + &b;
        let ba = &b + &a;
        ab.iter().zip(ba.iter()).all(|(x, y)| x == y || (x.is_nan() && y.is_nan()))
    }

    // Addition associativity: (a + b) + c == a + (b + c)
    // (approximate due to floating point)
    fn prop_add_associative_approx(
        a_data: Vec<f64>, b_data: Vec<f64>, c_data: Vec<f64>
    ) -> bool {
        let n = a_data.len().min(b_data.len()).min(c_data.len());
        if n == 0 { return true; }
        let a = Tensor1::from_vec(a_data[..n].to_vec());
        let b = Tensor1::from_vec(b_data[..n].to_vec());
        let c = Tensor1::from_vec(c_data[..n].to_vec());
        let lhs = &(&a + &b) + &c;
        let rhs = &a + &(&b + &c);
        lhs.iter().zip(rhs.iter()).all(|(x, y)| {
            (x - y).abs() < 1e-10 || (x.is_nan() && y.is_nan())
        })
    }

    // Multiplication distributivity: a * (b + c) == a*b + a*c
    fn prop_mul_distributive(
        a_data: Vec<f64>, b_data: Vec<f64>, c_data: Vec<f64>
    ) -> bool {
        let n = a_data.len().min(b_data.len()).min(c_data.len());
        if n == 0 { return true; }
        let a = Tensor1::from_vec(a_data[..n].to_vec());
        let b = Tensor1::from_vec(b_data[..n].to_vec());
        let c = Tensor1::from_vec(c_data[..n].to_vec());
        let lhs = &a * &(&b + &c);
        let rhs = &(&a * &b) + &(&a * &c);
        lhs.iter().zip(rhs.iter()).all(|(x, y)| {
            (x - y).abs() < 1e-10 * x.abs().max(1.0)
            || (x.is_nan() && y.is_nan())
        })
    }
}
```

### 6.2 广播性质测试 (broadcast_prop.rs)

```rust
quickcheck! {
    // Broadcasting is shape-consistent
    fn prop_broadcast_shape(
        a_shape: (usize, usize), b_shape: (usize, usize)
    ) -> bool {
        let (ar, ac) = (a_shape.0 % 16 + 1, a_shape.1 % 16 + 1);
        let (br, bc) = (b_shape.0 % 16 + 1, b_shape.1 % 16 + 1);
        let a = Tensor2::<f64>::zeros([ar, ac]);
        let b = Tensor2::<f64>::zeros([br, bc]);
        match (&a + &b) {
            Ok(c) => {
                c.shape()[0] == ar.max(br) && c.shape()[1] == ac.max(bc)
            }
            Err(_) => {
                // Should fail only if dimensions are incompatible
                (ar != br && ar != 1 && br != 1)
                || (ac != bc && ac != 1 && bc != 1)
            }
        }
    }
}
```

### 6.3 布局不变量测试 (layout_prop.rs)

```rust
quickcheck! {
    // Reshape preserves element count
    fn prop_reshape_preserves_len(shape: (usize, usize)) -> bool {
        let (r, c) = (shape.0 % 64 + 1, shape.1 % 64 + 1);
        let t = Tensor2::<f64>::zeros([r, c]);
        let flat = t.reshape([r * c]);
        match flat {
            Ok(f) => f.len() == t.len(),
            Err(_) => false, // contiguous tensor reshape should succeed
        }
    }

    // Transpose is involution: t.t().t() == t
    fn prop_transpose_involution(shape: (usize, usize)) -> bool {
        let (r, c) = (shape.0 % 32 + 1, shape.1 % 32 + 1);
        let t = Tensor2::from_fn([r, c], |[i, j]| (i * c + j) as f64);
        let tt = t.t().t().to_owned();
        t.iter().zip(tt.iter()).all(|(a, b)| a == b)
    }
}
```

---

## 7. 错误处理测试 (error_handling.rs)

| 测试 | 验证内容 |
|------|----------|
| `test_shape_mismatch_error` | 不兼容形状运算返回 ShapeMismatch |
| `test_broadcast_error` | 不可广播返回 BroadcastError |
| `test_layout_mismatch_error` | 非连续 reshape 返回 LayoutMismatch |
| `test_invalid_axis_error` | 轴越界返回 InvalidAxis |
| `test_invalid_shape_error` | reshape 元素数不匹配返回 InvalidShape |
| `test_dimension_mismatch_error` | 维度互转失败返回 DimensionMismatch |
| `test_empty_array_error` | 空数组 min/max 返回 EmptyArray |
| `test_index_out_of_bounds_panic` | 索引越界 panic |
| `test_error_display` | 所有错误类型的 Display 输出包含上下文 |
| `test_error_is_std_error` | std feature 下实现 std::error::Error |

---

## 8. CI 集成

### 8.1 测试矩阵

```yaml
# .github/workflows/test.yml (conceptual)
test:
  strategy:
    matrix:
      os: [ubuntu-latest, macos-latest, windows-latest]
      rust: [1.85.0, stable, nightly]
      features:
        - ""                    # default (std only)
        - "--features parallel"
        - "--features simd"
        - "--all-features"
        - "--no-default-features --features alloc"  # no_std

  steps:
    - name: Unit tests
      run: cargo test --lib ${{ matrix.features }}

    - name: Integration tests
      run: cargo test --tests ${{ matrix.features }}

    - name: Doc tests
      run: cargo test --doc ${{ matrix.features }}

    - name: Property tests
      run: cargo test --test property ${{ matrix.features }}
      if: matrix.rust == 'stable'
```

### 8.2 覆盖率配置

```yaml
coverage:
  steps:
    - name: Install cargo-llvm-cov
      run: cargo install cargo-llvm-cov

    - name: Generate coverage
      run: cargo llvm-cov --all-features --lcov --output-path lcov.info

    - name: Check threshold
      run: |
        COVERAGE=$(cargo llvm-cov --all-features --summary-only | grep 'TOTAL' | awk '{print $NF}')
        echo "Coverage: $COVERAGE"
        # Fail if below 80%
```

---

## 9. 与其他模块的交互

```
07-integration-tests
├── 依赖所有模块设计文档（03.01-05.11）
│   └── 每个模块对应一个集成测试文件
├── 依赖 03.07 错误处理
│   └── 错误类型验证
├── 依赖 08 文档设计
│   └── doctest 也是测试的一部分
└── 被 06 benchmark 依赖
    └── benchmark 复用测试数据生成器
```

---

## 10. 实现任务分解

| 任务 | 描述 | 预计时间 | 依赖 |
|------|------|----------|------|
| T1 | 创建 tests/ 目录结构和 common/ 共享工具 | 10 min | 无 |
| T2 | 实现 tests/dimension.rs | 10 min | 03.01 完成 |
| T3 | 实现 tests/element.rs | 10 min | 03.02 完成 |
| T4 | 实现 tests/complex.rs | 10 min | 03.03 完成 |
| T5 | 实现 tests/storage.rs | 10 min | 03.04 完成 |
| T6 | 实现 tests/layout.rs | 10 min | 03.05 完成 |
| T7 | 实现 tests/tensor_core.rs | 10 min | 03.06 完成 |
| T8 | 实现 tests/error_handling.rs | 10 min | 03.07 完成 |
| T9 | 实现 tests/iterator.rs | 10 min | 05.01 完成 |
| T10 | 实现 tests/elementwise.rs | 10 min | 05.02 完成 |
| T11 | 实现 tests/matrix_ops.rs | 10 min | 05.03 完成 |
| T12 | 实现 tests/reduction.rs | 10 min | 05.04 完成 |
| T13 | 实现 tests/broadcast.rs | 10 min | 05.05 完成 |
| T14 | 实现 tests/shape_ops.rs | 10 min | 05.06 完成 |
| T15 | 实现 tests/indexing.rs | 10 min | 05.07 完成 |
| T16 | 实现 tests/construction.rs | 10 min | 05.08 完成 |
| T17 | 实现 tests/ffi.rs | 10 min | 05.09 完成 |
| T18 | 实现 tests/workspace.rs | 10 min | 05.10 完成 |
| T19 | 实现 tests/thread_safety.rs | 10 min | 05.11 完成 |
| T20 | 实现 tests/edge_cases.rs | 10 min | T7 |
| T21 | 实现 tests/precision.rs | 10 min | T10, T12 |
| T22 | 实现 tests/property/ 属性测试 | 10 min | T10, T13 |
| T23 | 配置 CI 测试矩阵和覆盖率 | 10 min | T1-T22 |

### 10.1 并行执行分组

```
Wave 1 (无依赖):
  T1

Wave 2 (依赖 T1 + 各模块实现，可并行):
  T2, T3, T4, T5, T6, T7, T8

Wave 3 (依赖 Wave 2 + 各模块实现，可并行):
  T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19

Wave 4 (依赖 Wave 3):
  T20, T21, T22

Wave 5 (依赖 Wave 4):
  T23
```

---

## 11. 设计决策记录

### 11.1 决策：集成测试按模块分文件

| 属性 | 值 |
|------|-----|
| 决策 | 每个模块对应一个集成测试文件 |
| 理由 | 可独立运行；编译并行化；失败定位清晰 |
| 替代方案 | 按功能分（如 test_creation, test_arithmetic）— 放弃，跨模块边界模糊 |

### 11.2 决策：使用 quickcheck 属性测试

| 属性 | 值 |
|------|-----|
| 决策 | 使用 quickcheck 进行代数性质验证 |
| 理由 | 自动生成测试用例；覆盖人工难以想到的边界 |
| 替代方案 | proptest — 可接受，但 quickcheck 更轻量 |

### 11.3 决策：覆盖率阈值 80%

| 属性 | 值 |
|------|-----|
| 决策 | 行覆盖率 ≥ 80% |
| 理由 | 需求文档明确要求；80% 平衡了覆盖率与维护成本 |
| 替代方案 | 90% — 放弃，unsafe 代码和错误路径难以达到 90% |

---

*本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。*
