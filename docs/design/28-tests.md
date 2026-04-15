# 集成测试模块设计

> 文档编号: 28 | 影响范围: `tests/`, doctest 与测试 CI 矩阵 | 阶段: Phase 6
> 前置文档: 所有前置文档（`00-coding.md` ~ `27-benchmark.md`）
> 需求参考: 需求说明书 §28.2, §28.3, §28.4, §28.5
> 范围声明: 范围内

---

## 1. 模块定位

### 1.1 职责边界

| 职责       | 包含                                                                   | 不包含                                         |
| ---------- | ---------------------------------------------------------------------- | ---------------------------------------------- |
| 跨模块验证 | 维度、存储、布局、运算等模块的协同行为（参见 `01-architecture.md §5`） | 单函数测试（由 `#[cfg(test)] mod tests` 覆盖） |
| 边界覆盖   | 空张量、单元素、大张量、极端值、非连续、高维                           | 性能测量（由 benchmark 覆盖）                  |
| 数值精度   | IEEE 754 精度验证                                                      | 微观 benchmark                                 |
| 属性测试   | 代数不变量验证                                                         | 内存泄漏检测                                   |
| 并行安全   | 无数据竞争、并行/串行一致性                                            | 并行性能（由 benchmark 覆盖）                  |

### 1.2 设计原则

| 原则     | 体现                                                                                                         |
| -------- | ------------------------------------------------------------------------------------------------------------ |
| 全面性   | 覆盖需求说明书中所有 API 的关键行为                                                                          |
| 独立性   | 每个测试文件可独立运行，无跨文件依赖                                                                         |
| 可读性   | 测试名称描述预期行为，失败信息包含上下文                                                                     |
| 快速反馈 | 分层执行：smoke test ~2 min（每次 PR）/ required test ~10 min（每次 PR）/ extended test ~30 min（每周/发版） |

### 1.3 在架构中的位置

```
Dependency layers:
L0: error, private
L1: dimension, element, complex
L2: layout (depends on dimension)
L3: storage (independent of layout; owned by tensor and consumes layout results)
L4: tensor (depends on storage, dimension)
L5: overload/, iter/, index/, shape/, broadcast/, construct/, ffi/, convert/, format/

External (non-crate modules):
tests/  <- current module (consumes only the crate's public API)
```

## 2. 需求映射与范围约束

| 类型     | 内容                                                             |
| -------- | ---------------------------------------------------------------- |
| 需求映射 | `require.md §28.2`, `§28.3`, `§28.4`, `§28.5`                    |
| 范围内   | 集成测试矩阵、边界测试、属性测试、并行与 SIMD 一致性验证         |
| 范围外   | benchmark、生产环境监控、额外平台专用测试基础设施                |
| 非目标   | 通过测试文档引入新的产品能力、运行时依赖或超出需求范围的测试契约 |

---

## 3. 文件位置

### 3.1 目录结构

```
tests/
├── common/
│   ├── mod.rs                  # Shared utility exports
│   ├── assertions.rs           # Custom assertion helpers and macros
│   └── generators.rs           # Test data generators
│
├── compile_fail_tests.rs       # Repository-local compile-fail harness（top-level for Cargo discovery）
├── compile-fail/
│   └── ui/
│       ├── wrong_dimension_type.rs
│       ├── missing_element_bound.rs
│       └── mismatched_storage_type.rs
│
├── test_tensor.rs              # Tensor core functionality (creation/query/type aliases)
├── test_math.rs                # Element-wise operations (arithmetic/math/comparison/logic)
├── test_broadcast.rs           # Broadcasting (scalar/vector/matrix broadcasting)
├── test_index.rs               # Indexing operations (multi-dimensional indexing/range slicing)
├── test_construction.rs        # Constructors (zeros/ones/eye/from_shape_vec/from_shape_slice/from_scalar/from_array; `from_vec` convenience coverage only)
├── test_iterator.rs            # Iterators (elements/by-axis/by-index)
├── test_reduction.rs           # Reduction operations (sum/sum along axis)
├── test_matrix.rs              # Vector dot product (dot)
├── test_set.rs                 # Set operations (unique)
├── test_shape.rs               # Shape operations (transpose)
├── test_conversion.rs          # Type conversion (cast)
├── test_utility.rs             # Utility operations (fill/clip/to_contiguous)
├── test_output.rs              # NumPy-style formatted output (Display/Debug/truncation)
├── test_ffi.rs                 # FFI integration (raw pointers/BLAS compatibility)
├── test_workspace.rs           # Workspace-specific errors and borrow/split/growth
├── test_parallel.rs            # Parallel computation (consistency/data races)
├── test_simd.rs                # SIMD computation (result consistency)
├── test_error.rs               # Error handling (all error types)
│
├── property.rs                # Property-test entry point (integration test target)
└── property/
    ├── tensor_props.rs         # Tensor invariants (transpose involution, unique boundaries, etc.)
    ├── ops_props.rs            # Operation invariants (commutativity/associativity, etc.)
    └── shape_props.rs          # Shape invariants (transpose involution, etc.)
```

### 3.2 划分理由

按测试领域分文件，而非按源码模块：集成测试关注跨模块行为而非单个模块内部。

---

## 4. 依赖关系

### 4.1 依赖图

```
tests/
├── crate::tensor           # TensorBase, Tensor, TensorView, TensorViewMut, ArcTensor
├── crate::dimension        # Ix0~Ix6, IxDyn, Dimension
├── crate::element          # Element, Numeric, RealScalar, ComplexScalar
├── crate::complex          # Complex<f32>, Complex<f64>
├── crate::storage          # Owned, ViewRepr, ViewMutRepr, ArcRepr
├── crate::layout           # LayoutFlags
├── crate::math             # Element-wise operations
├── crate::broadcast        # broadcast_shape
├── crate::shape            # transpose
├── crate::index            # Multi-dimensional indexing and range slicing
├── crate::construct        # zeros, ones, eye, from_shape_vec, from_shape_slice, from_array, from_scalar (`from_vec` convenience path only)
├── crate::set              # unique
├── crate::ffi              # as_ptr, as_mut_ptr, from_raw_parts
├── crate::workspace        # Workspace
├── crate::error            # XenonError
└── crate::simd/parallel    # Conditionally compiled modules
```

### 4.2 依赖精确到类型级

| 来源模块    | 使用的类型/trait                                                                                               |
| ----------- | -------------------------------------------------------------------------------------------------------------- |
| `tensor`    | `Tensor<A, D>`, `TensorView`, `TensorViewMut`, `ArcTensor`, `.shape()`, `.strides()`（参见 `07-tensor.md §5`） |
| `dimension` | `Ix0`~`Ix6`, `IxDyn`, `Dimension`（参见 `02-dimension.md §5`）                                                 |
| `element`   | `Element`, `Numeric`, `RealScalar`, `ComplexScalar`（参见 `03-element.md §5`）                                 |
| `complex`   | `Complex<f32>`, `Complex<f64>`（参见 `04-complex.md §5`）                                                      |
| `storage`   | `Owned`, `ViewRepr`, `ViewMutRepr`, `ArcRepr`, `Storage`（参见 `05-storage.md §5`）                            |
| `layout`    | `LayoutFlags`（参见 `06-layout.md §5`）                                                                        |
| `error`     | `XenonError`, `Result<T>`（参见 `26-error.md §4`）                                                             |

### 4.3 依赖方向声明

> **依赖方向：单向消费。** `tests/` 仅消费 crate 公共 API（参见 `01-architecture.md §10`），不被任何模块依赖。

### 4.4 依赖合法性与新增依赖说明

| 项目           | 说明                                                                              |
| -------------- | --------------------------------------------------------------------------------- |
| 新增第三方依赖 | 当前基线不新增测试专用 dev-dependency；编译期边界验证使用标准工具链与仓库内脚本   |
| 合法性结论     | 仅依赖 `cargo test`、`cargo check`、`rustc` 与既有 feature 矩阵，符合最小依赖约束 |
| 替代方案       | 若后续希望引入专用 compile-fail 框架，须先单独裁决 dev-dependency 政策            |

---

## 5. 公共 API 设计

### 5.1 tests/common/mod.rs

```rust
// tests/common/mod.rs
pub mod assertions;
pub mod generators;
```

### 5.2 tests/common/assertions.rs

```rust
// tests/common/assertions.rs

/// Assert two real-valued tensors satisfy max(1 ULP, epsilon * |scalar_result|).
///
/// This is the default comparison helper for exact arithmetic paths.
/// Mathematical functions that are explicitly allowed to use dedicated
/// tolerances must call a separate helper instead of changing this default.
pub fn assert_tensor_close_real<A, D>(
    actual: &TensorBase<impl Storage<Elem = A>, D>,
    expected: &TensorBase<impl Storage<Elem = A>, D>,
    max_ulp: u64,
    msg: &str,
) where
    A: RealScalar + CastTo<f64>,
    D: Dimension,
{
    assert_eq!(actual.shape(), expected.shape(),
        "{}: shape mismatch: {:?} vs {:?}", msg, actual.shape(), expected.shape());

    for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let a_f: f64 = (*a).cast_to().unwrap();
        let e_f: f64 = (*e).cast_to().unwrap();
        assert!(ulp_eq_f64(a_f, e_f, max_ulp),
            "{}: element {} differs: actual={}, expected={}, tolerance=max(1 ULP, epsilon * |scalar_result|); configured ulp_limit={}",
            msg, idx, a, e, max_ulp);
    }
}

/// Assert two complex tensors satisfy max(1 ULP, epsilon * |scalar_result|) per component.
///
/// The real and imaginary components are compared independently so the helper
/// stays aligned with the current CastTo-based scalar conversion boundary.
pub fn assert_tensor_close_complex<A, D>(
    actual: &TensorBase<impl Storage<Elem = A>, D>,
    expected: &TensorBase<impl Storage<Elem = A>, D>,
    max_ulp: u64,
    msg: &str,
) where
    A: ComplexScalar,
    A::Real: RealScalar + CastTo<f64>,
    D: Dimension,
{
    assert_eq!(actual.shape(), expected.shape(),
        "{}: shape mismatch: {:?} vs {:?}", msg, actual.shape(), expected.shape());

    for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let a_re: f64 = a.re().cast_to().unwrap();
        let a_im: f64 = a.im().cast_to().unwrap();
        let e_re: f64 = e.re().cast_to().unwrap();
        let e_im: f64 = e.im().cast_to().unwrap();

        assert!(ulp_eq_f64(a_re, e_re, max_ulp),
            "{}: element {} real part differs: actual={}, expected={}, tolerance=max(1 ULP, epsilon * |scalar_result|); configured ulp_limit={}",
            msg, idx, a_re, e_re, max_ulp);
        assert!(ulp_eq_f64(a_im, e_im, max_ulp),
            "{}: element {} imaginary part differs: actual={}, expected={}, tolerance=max(1 ULP, epsilon * |scalar_result|); configured ulp_limit={}",
            msg, idx, a_im, e_im, max_ulp);
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

### 5.3 tests/common/generators.rs

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
pub struct NonContiguous2D {
    pub owner: Tensor2<f64>,
}

impl NonContiguous2D {
    pub fn view(&self) -> TensorView2<'_, f64> {
        self.owner.transpose()
    }
}

pub fn non_contiguous_2d(rows: usize, cols: usize) -> NonContiguous2D {
    NonContiguous2D {
        owner: Tensor2::<f64>::from_shape_vec(
            [cols, rows],
            (0..cols * rows).map(|idx| idx as f64).collect(),
        )
        .expect("shape and data length must match"),
    }
}
```

---

## 6. 内部实现设计

| 方面 | 说明 |
| ---- | ---- |
| 断言辅助 | 以 `assert_tensor_close_real` / `assert_tensor_close_complex`、错误断言宏与共享生成器统一测试表达，避免各测试文件重复定义比较逻辑 |
| 数值比较 | 默认比较遵循 mixed tolerance contract；复数按实部/虚部分量分别校验；仅数学函数场景允许引用单独容差规则 |
| 编译期边界 | compile-fail harness 负责验证 `usize`/非法无符号元素类型、错误 trait bound 与不合法存储组合 |
| 数据生成 | 顺序数据、三角函数样本、非连续视图与大张量样本均在共享工具中集中生成，保证可重复性 |

## 7. 实现任务拆分

| Wave | 目标 | 说明 |
| ---- | ---- | ---- |
| Wave 1 | 基础设施 | 建立 `tests/common/`、compile-fail harness 与共享数据生成工具 |
| Wave 2 | 核心功能测试 | 覆盖 tensor / math / broadcast / index / construction / reduction / iterator / matrix / set / shape / conversion / utility / output / error |
| Wave 3 | 特化测试 | 覆盖 FFI、parallel、SIMD 与 `std` 环境测试矩阵 |
| Wave 4 | 属性测试 | 固化 transpose、自反性、交换律、broadcast 形状规则等不变量 |
| Wave 5 | CI 集成 | 收敛默认、`parallel`、`simd`、`all-features` 的仓库测试矩阵 |

> 详细任务清单继续采用 Wave 形式维护，见后文“详细任务清单”。

---

## 8. 测试计划

### 8.1 test_tensor.rs

| 测试函数                    | 测试内容                                 | 优先级 |
| --------------------------- | ---------------------------------------- | ------ |
| `test_tensor_shape_strides` | shape(), strides(), ndim(), len() 正确性 | 高     |
| `test_tensor_is_empty`      | 空数组 is_empty() == true                | 高     |
| `test_tensor_view_creation` | view(), view_mut() 创建与读取            | 高     |
| `test_tensor_to_owned`      | to_owned() 深拷贝，修改不影响原始        | 高     |
| `test_tensor_into_owned`    | into_owned() 消耗转换                    | 中     |
| `test_type_aliases`         | Tensor0~Tensor3, TensorD 别名正确        | 高     |
| `test_tensor_debug_display` | Debug/Display 格式化输出                 | 中     |
| `test_arc_tensor_clone`     | ArcTensor clone 为浅拷贝                 | 中     |
| `test_arc_tensor_alias_isolation_on_write` | 共享后通过公开写入口修改副本时，不影响原值且保持别名隔离 | 中     |

### 8.2 test_math.rs

| 测试函数                          | 测试内容                     | 优先级 |
| --------------------------------- | ---------------------------- | ------ |
| `test_add_same_shape`             | 同形状加法                   | 高     |
| `test_add_scalar`                 | 标量加法                     | 高     |
| `test_sub_mul_div`                | 减法、乘法、除法正确性       | 高     |
| `test_neg`                        | 一元负号                     | 中     |
| `test_abs_signum`                 | abs, signum 行为（含 NaN、±0.0、±Inf） | 中     |
| `test_sin_sqrt_exp_ln_floor_ceil` | 三角/开方/指数/对数/取整精度 | 高     |
| `test_complex_math`               | 复数逐元素运算               | 中     |
| `test_bool_not`                   | bool 逻辑非                  | 中     |
| `test_compare_eq_ne`              | 等于/不等于比较              | 高     |
| `test_compare_lt_gt`              | 小于/大于比较                | 中     |
| `test_square`                     | square 逐元素平方            | 中     |
| `test_integer_add_overflow_panics` | 整数加法溢出触发 panic     | 高     |
| `test_integer_divide_by_zero_panics` | 整数除以零触发 panic     | 高     |
| `test_integer_min_abs_panics`     | 最小负值取绝对值触发 panic | 高     |
| `test_integer_min_div_neg_one_panics` | 最小负值除以 `-1` 触发 panic | 高 |

### 8.3 test_broadcast.rs

| 测试函数                       | 测试内容            | 优先级 |
| ------------------------------ | ------------------- | ------ |
| `test_broadcast_scalar`        | 标量与任意维度广播  | 高     |
| `test_broadcast_row_col`       | 行/列向量广播到矩阵 | 高     |
| `test_broadcast_left_pad`      | 维度不足左侧补 1    | 高     |
| `test_broadcast_incompatible`  | 不兼容形状返回错误  | 高     |
| `test_broadcast_view_readonly` | 广播视图为只读      | 高     |
| `test_broadcast_zero_stride`   | 广播后步长为 0      | 中     |

### 8.4 test_index.rs

| 测试函数                            | 测试内容                     | 优先级 |
| ----------------------------------- | ---------------------------- | ------ |
| `test_multi_dim_index`              | [i, j, k] 多维索引           | 高     |
| `test_index_out_of_bounds`          | 越界返回可恢复错误           | 高     |
| `test_slice_range`                  | 范围切片                     | 高     |
| `test_slice_subrange`               | 范围切片子区间               | 中     |
| `test_slice_mut_broadcast_rejected` | 广播视图上的可变切片在编译期/API 缺失层被拒绝 | 高     |

### 8.5 test_construction.rs

| 测试函数                      | 测试内容                                                     | 优先级 |
| ----------------------------- | ------------------------------------------------------------ | ------ |
| `test_zeros_ones_from_scalar` | zeros, ones, from_scalar 构造                                | 高     |
| `test_eye_identity`           | 单位矩阵                                                     | 高     |
| `test_from_data_constructors` | 以 `from_shape_vec`、`from_shape_slice` 为主；`from_vec` 仅覆盖 Ix1 convenience path | 高     |
| `test_from_fixed_array`       | 从固定数组构造                                               | 中     |

### 8.6 test_reduction.rs

| 测试函数                    | 测试内容                      | 优先级 |
| --------------------------- | ----------------------------- | ------ |
| `test_sum_global`           | 全局 sum                      | 高     |
| `test_sum_axis`             | 沿轴 sum                      | 高     |
| `test_sum_keepdims`         | sum 保留被归约轴为长度 1      | 中     |
| `test_sum_empty`            | 空数组 sum 返回加法单位元     | 高     |
| `test_sum_nan`              | sum 含 NaN 结果为 NaN         | 中     |
| `test_integer_sum_overflow` | 整数 sum 溢出视为不可恢复错误 | 中     |

### 8.7 test_iterator.rs

| 测试函数             | 测试内容                        | 优先级 |
| -------------------- | ------------------------------- | ------ |
| `test_iter_elements` | 按元素遍历与 `len()` 一致       | 高     |
| `test_axis_iter`     | 按轴遍历产出数量与形状一致      | 高     |
| `test_axis_iter_ix0_runtime_error` | 0D 张量上的 `axis_iter` 返回可恢复错误 | 高     |
| `test_indexed_iter`  | 按索引遍历返回 F-order 逻辑索引 | 中     |

### 8.8 test_matrix.rs

| 测试函数                  | 测试内容                  | 优先级 |
| ------------------------- | ------------------------- | ------ |
| `test_dot_product`        | 向量内积                  | 高     |
| `test_dot_complex`        | 复数内积共轭线性          | 高     |
| `test_dot_shape_mismatch` | 内积维度不匹配返回错误    | 高     |
| `test_dot_empty`          | 空向量 dot 返回加法单位元 | 中     |

### 8.9 test_set.rs

| 测试函数                        | 测试内容                                          | 优先级 |
| ------------------------------- | ------------------------------------------------- | ------ |
| `test_unique_order_unspecified` | unique 返回不重复元素，结果无需排序且顺序不作要求 | 高     |
| `test_unique_integers`          | 整数 unique                                       | 中     |
| `test_unique_nan_preserved`     | 浮点 `NaN != NaN`，输入中的每个 NaN 都应保留      | 高     |
| `test_unique_signed_zero_equal` | `-0.0` 与 `0.0` 视为相等，仅保留一个零值          | 高     |
| `test_unique_complex`           | 复数 unique                                       | 中     |

### 8.10 test_shape.rs

| 测试函数                  | 测试内容 | 优先级 |
| ------------------------- | -------- | ------ |
| `test_transpose_2d`       | 2D 转置  | 高     |
| `test_transpose_high_dim` | 高维转置 | 中     |

### 8.11 test_conversion.rs

| 测试函数                    | 测试内容                                       | 优先级 |
| --------------------------- | ---------------------------------------------- | ------ |
| `test_cast_f32_to_f64`      | `cast()` 成功执行 f32→f64 无损转换             | 高     |
| `test_cast_f64_to_f32`      | `cast()` 对默认禁止的 f64→f32 有损转换返回错误 | 高     |
| `test_cast_real_to_complex` | `cast()` 执行实数→复数转换并将虚部补为 0       | 中     |
| `test_cast_complex_to_real_zero_imag` | `cast()` 仅在虚部为 0 时允许复数→实数成功 | 高     |
| `test_cast_complex_to_real_nonzero_imag` | `cast()` 对虚部非 0 的复数→实数返回 `TypeConversion` 错误 | 高 |
| `test_bool_not_participating_in_cast` | `bool` 不参与逐元素类型转换，相关入口在类型层或运行时被拒绝 | 高 |
| `test_cast_nan_to_int`      | `cast()` 对 NaN→整数返回 `TypeConversion` 错误 | 中     |

### 8.12 test_utility.rs

| 测试函数                   | 测试内容                | 优先级 |
| -------------------------- | ----------------------- | ------ |
| `test_fill_inplace`        | 原地 fill / 非连续 fill | 中     |
| `test_clip`                | 裁剪操作                | 中     |
| `test_clip_non_contiguous` | 非连续布局裁剪          | 中     |
| `test_to_contiguous`       | 连续化保持逻辑元素顺序  | 高     |

### 8.13 test_output.rs

| 测试函数                       | 测试内容                          | 优先级 |
| ------------------------------ | --------------------------------- | ------ |
| `test_display_small_tensor`    | 小张量 NumPy 风格输出             | 高     |
| `test_display_truncated`       | 超阈值触发截断                    | 高     |
| `test_debug_includes_metadata` | Debug 包含 shape/stride/type 信息 | 中     |
| `test_output_complex`          | 复数格式化输出                    | 中     |

### 8.14 test_ffi.rs

| 测试函数                        | 测试内容                                   | 优先级 |
| ------------------------------- | ------------------------------------------ | ------ |
| `test_as_ptr`                   | as_ptr 返回正确指针                        | 高     |
| `test_as_mut_ptr`               | as_mut_ptr 返回可变指针                    | 高     |
| `test_lda`                      | lda 返回 leading dimension                 | 中     |
| `test_is_blas_layout_compatible` | BLAS 兼容性检查                           | 高     |
| `test_export_roundtrip`         | `export` 导出只读视图并验证元数据往返      | 高     |
| `test_export_mut_roundtrip`     | `export_mut` 导出可变视图并验证独占写路径   | 高     |
| `test_from_raw_parts_mut_reject_overlap` | `from_raw_parts_mut` 对地址重叠/别名冲突执行检查 | 高     |
| `test_try_offset_of`            | try_offset_of 正确计算                     | 高     |

### 8.15 test_workspace.rs

| 测试函数                               | 测试内容                                                                   | 优先级 |
| -------------------------------------- | -------------------------------------------------------------------------- | ------ |
| `test_workspace_new_invalid_alignment` | 非法对齐返回 `XenonError::Workspace(WorkspaceError::InvalidLayout { .. })` | 高     |
| `test_workspace_borrow_rules`          | 借用守卫与复借用约束                                                       | 高     |
| `test_workspace_split`                 | split 后子工作空间边界正确                                                 | 中     |
| `test_workspace_ensure_capacity`       | 扩容不破坏已借用安全性                                                     | 高     |
| `test_workspace_assume_init_prefix`    | `assume_init_*` 只允许访问调用方已证明初始化的前缀                         | 高     |

### 8.16 test_parallel.rs

| 测试函数                                    | 测试内容                                                 | 优先级 |
| ------------------------------------------- | -------------------------------------------------------- | ------ |
| `test_sum_parallel_feature_consistency`     | 启用 `parallel` feature 后，同一公开 `sum()` 语义与默认配置一致（参见 `09-parallel.md §8`） | 高     |
| `test_par_add_consistency`                  | 并行 add 与串行 add 结果一致                             | 高     |
| `test_parallel_read`                        | 多线程并发只读访问安全（参见 `25-safety.md §5`）         | 高     |
| `test_nested_parallel_falls_back_to_serial` | 嵌套并行检测后自动回退串行                               | 中     |

### 8.17 test_simd.rs

| 测试函数                    | 测试内容                                                                    | 优先级 |
| --------------------------- | --------------------------------------------------------------------------- | ------ |
| `test_simd_add_consistency` | SIMD add 与显式标量 baseline 结果一致（参见 `08-simd.md §8`）               | 高     |
| `test_simd_sum_consistency` | SIMD sum 与显式标量 baseline 结果一致                                       | 高     |
| `test_simd_fallback_small`  | 小数组 SIMD 回退到标量                                                      | 中     |
| `test_simd_complex_path`    | 验证 `Complex<f32>` / `Complex<f64>` SIMD kernel 与标量路径在文档容差内一致 | 中     |

### 8.18 compile_fail_tests.rs

| 测试函数                              | 测试内容                                           | 优先级 |
| ------------------------------------- | -------------------------------------------------- | ------ |
| `ui_wrong_dimension_type`             | 非法维度类型在编译期被拒绝                         | 高     |
| `ui_missing_element_bound`            | 非法元素类型或缺失元素约束在编译期被拒绝           | 高     |
| `ui_mismatched_storage_type`          | 不合法存储表示组合在编译期被拒绝                   | 高     |
| `ui_unsigned_tensor_element_rejected` | `usize` 等无符号整数不能作为张量元素类型           | 高     |
| `ui_invalid_unsigned_element_rejected` | `u8` / `u16` / `u32` / `u64` 等非法无符号元素类型在编译期被拒绝 | 高 |

### 8.19 property.rs

| 测试函数                  | 测试内容                                  | 优先级 |
| ------------------------- | ----------------------------------------- | ------ |
| `prop_transpose_involution` | `transpose().transpose()` 与原张量相等 | 高     |
| `prop_add_commutative`    | 逐元素加法满足交换律（在文档化容差内）    | 中     |
| `prop_unique_len_bound`   | `unique(a).len()` 不超过 `a.len()`        | 中     |
| `prop_broadcast_shape_rule` | 广播结果形状遵循 NumPy 规则             | 高     |

### 8.20 test_error.rs

| 测试函数                      | 测试内容                                                                                                     | 优先级 |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------ | ------ |
| `test_broadcast_shape_error`  | 逐元素/广播不兼容形状返回 `XenonError::BroadcastError`；`ShapeMismatch` 仅用于非广播双输入形状冲突（如 dot） | 高     |
| `test_broadcast_error`        | 不可广播返回 `XenonError::BroadcastError`                                                                    | 高     |
| `test_invalid_axis_error`     | 轴越界返回 InvalidAxis                                                                                       | 高     |
| `test_invalid_argument_error` | 非法参数返回 `XenonError::InvalidArgument`                                                                   | 高     |
| `test_invalid_shape_error`    | 非法 shape/元素数不匹配返回 `XenonError::InvalidShape`                                                       | 高     |
| `test_error_display`          | 所有错误类型的 Display 包含上下文                                                                            | 中     |
| `test_send_sync_contracts`    | 各 storage mode 的 Send/Sync 边界与 `25-safety.md` 一致                                                      | 高     |
| `test_complex_c99_layout`     | `Complex<T>` 的 C-compatible 布局与 FFI 约定一致                                                             | 高     |
| `test_ix0_iter_single`        | 零维张量元素迭代恰好产出 1 个元素                                                                            | 高     |
| `test_zst_storage_no_ub`      | 验证内部实现对零大小类型的安全处理（内部不变量测试，非公开 API 契约）                                       | 高     |

---

### 8.21 边界测试场景

#### 8.21.1 边界覆盖（参照需求说明书 §28.4）

| 边界类别            | 测试场景                                        | 覆盖的操作                       |
| ------------------- | ----------------------------------------------- | -------------------------------- |
| 空张量              | shape 含 0 的数组（如 `[0, 3]`）                | 构造、迭代、归约、形状操作、索引 |
| 单元素              | shape 全为 1（如 `[1, 1]`）                     | 所有运算、归约、广播             |
| 大张量              | 元素数量达到 `10^7` 量级或总字节数达到 GiB 量级 | 归约精度、并行正确性             |
| 极端值（NaN）       | 含 NaN 的浮点数组                               | 归约、比较                       |
| 极端值（Inf）       | 含 Inf 的浮点数组                               | 算术、归约、cast                 |
| 极端值（Subnormal） | 含次正规数                                      | 算术精度                         |
| 非连续布局          | 转置/切片后的视图                               | 所有运算、迭代、归约             |
| 高维（≥4 维）       | 4D~6D 数组                                      | 形状操作、广播、迭代             |
| 非法元素类型        | 违反 §4 约束的输入（如 `usize`、`String`）      | 编译期失败或 trait 约束拒绝      |

补充覆盖要求：

- 大张量边界须覆盖 `10^7` 元素量级，例如 shape 为 `[10000, 1000]` 的连续张量；测试目标包括 `zeros` / `ones` / `from_shape_vec` 构造、基础索引访问、以及至少一种只读运算或归约路径。
- 对 `10^7` 元素张量，须验证内存分配成功、测试过程不因设计缺陷触发 OOM，并显式检查 shape 乘积与线性索引换算在 `usize` 边界内不发生溢出。
- 大张量索引须覆盖首元素、末元素及跨行边界元素（如 `[0, 0]`、`[9999, 999]`、`[1, 0]`），确保逻辑索引到线性偏移的映射正确。
- 高维场景除 `Ix4`~`Ix6` 外，还须覆盖 `IxDyn` 动态维度：rank 0（标量）、1、2，直至 12 的代表性张量，满足需求说明书 §28.4 对“高 rank 的动态维度张量”的覆盖要求。
- `IxDyn` 测试须验证与固定维度 `Ix1`~`Ix6` 的 shape/dimension 转换、广播、动态索引与迭代行为；当前设计以 rank 12 作为跨平台上限的测试目标，不假定平台支持无限 rank。

#### 8.21.2 边界测试示例

```rust
#[test]
fn test_empty_tensor_properties() {
    let t = Tensor2::<f64>::zeros([0, 5]).expect("shape is valid");
    assert!(t.is_empty());
    assert_eq!(t.len(), 0);
    assert_eq!(t.ndim(), 2);
    assert_eq!(t.shape(), &[0, 5]);
    assert_eq!(t.iter().count(), 0);
}

#[test]
fn test_empty_tensor_sum() {
    let t = Tensor1::<f64>::zeros([0]).expect("shape is valid");
    assert_eq!(t.sum(), 0.0); // additive identity
}

#[test]
fn test_single_element() {
    let t = Tensor::<f64, Ix0>::from_scalar(42.0f64);  // see 18-construction.md §5.4
    assert_eq!(t.len(), 1);
    assert_eq!(t.sum(), 42.0);
}

#[test]
fn test_nan_sum_propagation() {
    let t = Tensor1::from_shape_vec([3], vec![1.0, f64::NAN, 3.0])
        .expect("shape and data length must match");
    assert!(t.sum().is_nan());
}

#[test]
fn test_high_dim_operations() {
    let t4 = Tensor::<f64, Ix4>::zeros([2, 3, 4, 5]).expect("shape is valid");
    assert_eq!(t4.len(), 120);
    assert_eq!(t4.sum(), 0.0);

    let t6 = Tensor::<f64, Ix6>::ones([2, 2, 2, 2, 2, 2]).expect("shape is valid");
    assert_eq!(t6.len(), 64);
    assert_eq!(t6.sum(), 64.0);
}

#[test]
fn test_large_tensor_boundary_10m_elements() {
    let shape = [10_000, 1_000];
    let len = shape[0] * shape[1];
    assert_eq!(len, 10_000_000);
    assert!(len.checked_mul(core::mem::size_of::<f32>()).is_some());

    let data = vec![1.0f32; len];
    let t = Tensor2::<f32>::from_shape_vec(shape, data)
        .expect("10^7 element tensor allocation must succeed");

    assert_eq!(t.shape(), &shape);
    assert_eq!(t[[0, 0]], 1.0);
    assert_eq!(t[[1, 0]], 1.0);
    assert_eq!(t[[9_999, 999]], 1.0);
}

#[test]
fn test_ixdyn_high_rank_scenarios() {
    for rank in 0..=12usize {
        let shape = vec![1usize; rank];
        let t = Tensor::<i32, IxDyn>::zeros(IxDyn::from_slice(&shape)).expect("shape is valid");
        assert_eq!(t.ndim(), rank);
        assert_eq!(t.shape(), shape.as_slice());
    }

    let fixed = Tensor2::<i32>::zeros([2, 3]).expect("shape is valid");
    let dyn_tensor = Tensor::<i32, IxDyn>::from_shape_vec(
        IxDyn::from_slice(fixed.shape()),
        fixed.iter().copied().collect(),
    )
    .expect("fixed tensor should convert to IxDyn shape");
    assert_eq!(dyn_tensor.shape(), &[2, 3]);

    let lhs = Tensor::<i32, IxDyn>::ones(IxDyn::from_slice(&[3, 1, 4]))
        .expect("shape is valid");
    let rhs = Tensor::<i32, IxDyn>::ones(IxDyn::from_slice(&[1, 5, 4]))
        .expect("shape is valid");
    let sum = (&lhs + &rhs).unwrap();
    assert_eq!(sum.shape(), &[3, 5, 4]);
    assert_eq!(sum[IxDyn::from_slice(&[2, 4, 3])], 2);
}
```

说明：上述示例中的大张量测试以 `f32` 作为默认元素类型，以在 `10^7` 元素量级下控制测试内存占用；若目标平台内存预算更紧，可保留元素数量目标不变并避免并发叠加分配。`IxDyn` 高 rank 测试以 rank 12 为上限，兼顾需求覆盖与测试执行成本。

---

### 8.22 数值精度规范

#### 8.22.1 IEEE 754 精度要求

| 运算类别                    | 默认比较规则                                                | 参考实现              |
| --------------------------- | ----------------------------------------------------------- | --------------------- |
| 加减乘                      | `max(1 ULP, epsilon * |scalar_result|)` 混合容差           | 显式标量 helper       |
| 归约 (`sum`)                | 与标量基线一致；默认按混合容差契约判定                      | 显式标量基线路径      |
| 复数加减乘                  | 实部/虚部分别满足对应实数类型的混合容差上限                 | 显式标量复数 helper   |
| 超越函数 (`sin`/`exp`/`ln`) | 不继承默认比较；仅可引用各 API 单独容差规则 | `std` 对应数学函数 |

对 `Complex<f32>` / `Complex<f64>` 的默认容差采用**分量级混合判定**：分别比较实部与虚部，各自必须满足 `max(1 ULP, epsilon * |scalar_result|)`；不得把复数整体模长误差替代为分量误差。

#### 8.22.2 浮点比较方式

浮点测试默认使用 **mixed tolerance contract** 比较；并行与 SIMD 路径对浮点/复数结果的确定性要求须对齐 `require.md §28.3`：同执行路径下结果确定，但跨执行路径（标量/SIMD/并行）只允许出现处于文档化混合容差契约内的舍入差异；整数和布尔路径仍须逐位一致。

默认混合容差契约适用于不同执行路径（串行/SIMD/并行）下的浮点和复数比较。只有需求或对应 API 文档明确放宽时，`sin` / `exp` / `ln` 等数学函数才可使用单独容差 helper；`usize` 与其他无符号整数约束属于 `require.md §28.5` 的编译期边界，不适用该数值规则。

```rust
/// Compare two finite floats with the default mixed tolerance contract.
///
/// This is the default helper for floating-point result comparison.
/// NaN / Inf semantics must be asserted explicitly in dedicated tests.
pub fn ulp_eq_f64(actual: f64, expected: f64, max_ulp: u64) -> bool {
    if !actual.is_finite() || !expected.is_finite() {
        return false;
    }

    let ulp_ok = ulp_distance(actual, expected) <= max_ulp;
    let epsilon_ok = (actual - expected).abs() <= f64::EPSILON * expected.abs();

    ulp_ok || epsilon_ok
}

/// Compare two complex values component-wise under the mixed tolerance contract.
pub fn ulp_eq_complex64(
    actual: Complex<f64>,
    expected: Complex<f64>,
    max_ulp: u64,
) -> bool {
    ulp_eq_f64(actual.re, expected.re, max_ulp)
        && ulp_eq_f64(actual.im, expected.im, max_ulp)
}
```

---

### 8.23 并行与 SIMD 测试

#### 8.23.1 并行无数据竞争

线程安全测试方案（参见 `25-safety.md §8`）：

| 方式            | 说明                                                     |
| --------------- | -------------------------------------------------------- |
| `thread::scope` | 使用 scoped thread 并发访问 TensorView（只读）           |
| ArcTensor       | 多线程共享 ArcTensor 并发读取                            |
| 类型系统验证    | 通过 `Send`/`Sync` 约束和并发只读/独占写测试验证线程安全 |

#### 8.23.2 并行归约一致性

```rust
#[test]
fn test_sum_parallel_feature_consistency() {
    let n = 1_000_000;
    let t = Tensor1::<f64>::from_shape_vec(
        [n],
        (0..n).map(|idx| (idx as f64).sin()).collect(),
    )
    .expect("shape and data length must match");

    let scalar_baseline: f64 = t.iter().copied().sum();

    #[cfg(feature = "parallel")]
    {
        assert!(
            ulp_eq_f64(t.sum(), scalar_baseline, 1),
            "sum() under the parallel feature must stay within the documented mixed tolerance contract"
        );
    }

    #[cfg(not(feature = "parallel"))]
    {
        assert!(ulp_eq_f64(t.sum(), scalar_baseline, 1));
    }
}
```

#### 8.23.3 SIMD 结果一致性

```rust
#[test]
fn test_simd_add_consistency() {
    let a = Tensor1::<f64>::from_shape_vec(
        [1024],
        (0..1024).map(|idx| (idx as f64).sin()).collect(),
    )
    .expect("shape and data length must match");
    let b = Tensor1::<f64>::from_shape_vec(
        [1024],
        (0..1024).map(|idx| ((idx as f64) / 1024.0).exp()).collect(),
    )
    .expect("shape and data length must match");

    let result = (&a + &b).unwrap();
    let expected: Vec<f64> = a
        .iter()
        .zip(b.iter())
        .map(|(lhs, rhs)| lhs + rhs)
        .collect();

    // Verify against an explicit scalar baseline
    for i in 0..1024 {
        let expected_value = expected[i];
        assert!(
            ulp_eq_f64(result[[i]], expected_value, 1),
            "SIMD add mismatch at {}",
            i
        );
    }
}
```

---

### 8.24 属性测试不变量设计

#### 8.24.1 不变量清单

| 不变量             | 测试方法                                                                                          | 优先级 |
| ------------------ | ------------------------------------------------------------------------------------------------- | ------ |
| `sum` 保加法单位元 | 空数组 sum == 0（参见 `13-reduction.md §5.1`）                                                    | 高     |
| `transpose` 自反性 | `transpose().transpose()` == 原张量（参见 `16-shape.md §5.1`）                                   | 高     |
| 加法交换律         | `a + b` == `b + a`（近似）                                                                        | 中     |
| `unique` 保元素数  | `unique(a).len()` ≤ `a.len()`                                                                     | 中     |
| `unique` 不含重复  | 对非 `NaN` 元素，结果中不得重复；`NaN` 按 IEEE 754 自反不相等语义逐个保留                         | 中     |
| 广播形状一致性     | 广播结果形状遵循 NumPy 规则：相等取该值，一方为 1 取另一方，否则报错（参见 `15-broadcast.md §5`） | 高     |

#### 8.24.2 属性测试框架

使用受控参数化数据生成与固定种子样本进行覆盖；不引入额外属性测试框架依赖：

```rust
// tests/property/tensor_props.rs
#[test]
fn prop_transpose_involution() {
    for r in 1..32usize {
        for c in 1..32usize {
            let t = Tensor2::from_shape_vec(
                [r, c],
                (0..r * c).map(|idx| idx as f64).collect(),
            )
            .expect("shape and data length must match");
            let tt = t.transpose().transpose().to_owned();
            for (a, b) in t.iter().zip(tt.iter()) {
                assert_eq!(a, b);
            }
        }
    }
}

#[test]
fn prop_add_commutative() {
    for len in 1..256usize {
        let a = Tensor1::from_shape_vec([len], (0..len).map(|idx| idx as f64).collect())
            .expect("shape and data length must match");
        let b = Tensor1::from_shape_vec([len], (0..len).map(|idx| idx as f64 + 1.0).collect())
            .expect("shape and data length must match");
        let ab = (&a + &b).unwrap();
        let ba = (&b + &a).unwrap();
        for (x, y) in ab.iter().zip(ba.iter()) {
            assert!(ulp_eq_f64(*x, *y, 1));
        }
    }
}
```

---

### 8.25 内部实现补充

#### 8.25.1 assert_tensor_close helper 实现细节

> 默认比较 helper 应基于已冻结的元素转换接口实现（如实数路径上的 `CastTo<f64>`），避免依赖未在 `03-element.md` 中正式冻结的附加转换约定；复数路径须拆分为独立 helper，并按实部/虚部分量比较。

`assert_tensor_close_real` / `assert_tensor_close_complex` 的核心逻辑：

1. **形状检查**：先比较 shape，不匹配立即断言失败并附上下文
2. **逐元素比较**：默认路径使用 `max(1 ULP, epsilon * |scalar_result|)` 混合判定；复数路径按实部/虚部分量分别比较；仅在 API 文档明确允许的数学函数场景使用单独容差 helper
3. **错误信息**：包含测试名称、元素索引、实际值、期望值和比较契约（混合容差或单独容差）

```rust
// Internal implementation detail of the default tensor-close helpers
//
// 1. shape comparison: O(1) early exit if shapes differ
// 2. element-wise iteration: uses iter() according to the iterator contract
//    from 10-iterator.md; tests compare logical element sequences, not raw
//    storage order assumptions
// 3. default comparison uses the mixed tolerance contract; complex values compare real and
//    imaginary components independently
// 4. dedicated tolerance helpers are opt-in for mathematically allowed APIs
// 5. error message includes: msg, index, actual, expected, and comparison contract
```

#### 8.25.2 参数化策略设计

| 不变量类型       | 参数化策略                  | 说明                            |
| ---------------- | --------------------------- | ------------------------------- |
| 形状参数         | `1..64usize`                | 避免零/过大形状                 |
| 浮点数据         | `from_shape_vec` + 固定变换 | 覆盖正常值、NaN、Inf、Subnormal |
| 浮点数据（过滤） | 手写筛选后的确定性样本      | 排除 NaN/Inf                    |
| 整数数据         | 固定边界样本集              | 含负数、零与边界值              |
| 1D 向量          | `for len in 1..256`         | 含边界长度                      |

#### 8.25.3 测试数据生成策略

| 数据类型   | 生成方法                                                     | 用途                     |
| ---------- | ------------------------------------------------------------ | ------------------------ |
| 顺序填充   | `from_shape_vec([n], (0..n).map(|idx| idx as f64).collect())` | 可重复、确定性           |
| 三角函数值 | `from_shape_vec([n], (0..n).map(|idx| (idx as f64).sin()).collect())` | 非平凡浮点值             |
| 非连续视图 | `t.transpose()` 或 `t.slice(s![0..n])`                      | 测试子区间与转置视图处理 |
| 空/单元素  | `zeros([0])?` / `Tensor::<f64, Ix0>::from_scalar(42.0)`      | 边界测试                 |
| 受控数据   | `from_shape_vec(...)` / 顺序生成                             | 属性测试                 |

---

### 8.26 Good/Bad 对比示例

#### 8.26.1 Good — 正确的集成测试模式

```rust
// Good: Use the default mixed-tolerance helper for floating point comparison
#[test]
fn test_add_result() {
    let a = Tensor1::from_shape_vec([3], vec![1.0, 2.0, 3.0])
        .expect("shape and data length must match");
    let b = Tensor1::from_shape_vec([3], vec![4.0, 5.0, 6.0])
        .expect("shape and data length must match");
    let result = (&a + &b).unwrap();
    let expected = Tensor1::from_shape_vec([3], vec![5.0, 7.0, 9.0])
        .expect("shape and data length must match");
    assert_tensor_close_real(&result, &expected, 1, "add");
}

// Good: Test a supported error path that returns Result
#[test]
fn test_axis_iter_invalid_axis() {
    let t = Tensor2::<f64>::zeros([2, 3]).expect("shape is valid");
    assert_xenon_error!(t.axis_iter(Axis(2)), XenonError::InvalidAxis { .. });
}

// Good: Parameterized test with standard shapes
#[test]
fn test_transpose_shapes() {
    for (r, c) in standard_shapes_2d() {
        let t = Tensor2::<f64>::zeros([r, c]).expect("shape is valid");
        let tt = t.transpose();
        assert_eq!(tt.shape(), &[c, r]);
    }
}
```

#### 8.26.2 Bad — 错误的集成测试模式

```rust
// Bad: Using exact equality for floating point
#[test]
fn test_add_bad() {
    let a = Tensor1::from_shape_vec([2], vec![0.1, 0.2])
        .expect("shape and data length must match");
    let b = Tensor1::from_shape_vec([2], vec![0.3, 0.4])
        .expect("shape and data length must match");
    let result = (&a + &b).unwrap();
    assert_eq!(result[[0]], 0.4);  // Floating point exact comparison may fail
}

// Bad: Ignoring a recoverable error path
#[test]
fn test_axis_iter_bad() {
    let t = Tensor2::<f64>::zeros([2, 3]).expect("shape is valid");
    let _ = t.axis_iter(Axis(2));  // Silently ignoring the Result
}

// Bad: Hardcoded magic numbers without context
#[test]
fn test_bad_magic() {
    let t = Tensor1::<f64>::zeros([100]).expect("shape is valid");
    assert_eq!(t.sum(), 0.0);  // What is 100? Why zeros?
}
```

---

## 9. 模块交互设计

### 9.1 Test files到被测模块映射

| Test files             | 被测模块            | 对应设计文档                    |
| ---------------------- | ------------------- | ------------------------------- |
| `test_tensor.rs`       | `tensor`, `storage` | `07-tensor.md`, `05-storage.md` |
| `test_math.rs`         | `math`              | `11-math.md`                    |
| `test_broadcast.rs`    | `broadcast`         | `15-broadcast.md`               |
| `test_index.rs`        | `index`             | `17-indexing.md`                |
| `test_construction.rs` | `construct`         | `18-construction.md`            |
| `test_reduction.rs`    | `reduction`         | `13-reduction.md`               |
| `test_iterator.rs`     | `iter`              | `10-iterator.md`                |
| `test_matrix.rs`       | `matrix`            | `12-matrix.md`                  |
| `test_set.rs`          | `set`               | `14-set.md`                     |
| `test_shape.rs`        | `shape`             | `16-shape.md`                   |
| `test_conversion.rs`   | `convert`           | `21-type.md`                    |
| `test_workspace.rs`    | `workspace`         | `24-workspace.md`               |
| `test_utility.rs`      | `utility`           | `20-utility.md`                 |
| `test_output.rs`       | `format`            | `22-output.md`                  |
| `test_ffi.rs`          | `ffi`, `workspace`  | `23-ffi.md`, `24-workspace.md`  |
| `test_parallel.rs`     | `parallel`          | `09-parallel.md`                |
| `test_simd.rs`         | `simd`              | `08-simd.md`                    |
| `test_error.rs`        | `error`             | `26-error.md`                   |

> **说明**：workspace 模块可保留内部 `WorkspaceError` 载荷，但公开 API 边界统一验证 `XenonError::Workspace(...)`；`test_workspace.rs` 关注 workspace 语义与内层载荷字段，`test_error.rs` 关注统一公开错误边界。

### 9.2 数据流

```
Test files
    │
    ├── call crate public APIs (Tensor::zeros, +, sum, transpose(), ...)
    │       │
    │       └── internal path: storage → tensor → overload → simd/parallel
    │
    ├── use common/ utilities
│       ├── use the default mixed-tolerance assertion helpers for numeric comparison
    │       └── generators produce test data
    │
    └── parameterized data generation
            └── enumerate standard inputs → verify invariants
```

---

## 附录：详细任务清单

### Wave 1: 基础设施

- [ ] **T1**: 创建 `tests/` 目录结构和 `common/` 共享工具
  - 文件: `tests/common/mod.rs`, `tests/common/assertions.rs`, `tests/common/generators.rs`
  - 内容: `assert_tensor_close_real` / `assert_tensor_close_complex` helper、标准形状常量、数据生成函数
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

- [ ] **T3**: 实现 `tests/test_math.rs`
  - 文件: `tests/test_math.rs`
  - 内容: 逐元素运算（算术/数学/比较/逻辑）
  - 测试: `cargo test --test test_math`
  - 前置: T1
  - 预计: 10 min

- [ ] **T4**: 实现 `tests/test_broadcast.rs`
  - 文件: `tests/test_broadcast.rs`
  - 内容: 广播机制（标量/行/列/不兼容/只读）
  - 测试: `cargo test --test test_broadcast`
  - 前置: T1
  - 预计: 10 min

- [ ] **T5**: 实现 `tests/test_index.rs`
  - 文件: `tests/test_index.rs`
  - 内容: 索引操作（多维索引/越界返回错误/切片/步长）
  - 测试: `cargo test --test test_index`
  - 前置: T1
  - 预计: 10 min

- [ ] **T6**: 实现 `tests/test_construction.rs`
  - 文件: `tests/test_construction.rs`
  - 内容: 构造方法（zeros/ones/eye/from_shape_vec/from_shape_slice/from_scalar/from_array；`from_vec` 仅作为 Ix1 convenience coverage）
  - 测试: `cargo test --test test_construction`
  - 前置: T1
  - 预计: 10 min

- [ ] **T7**: 实现 `tests/test_reduction.rs`
  - 文件: `tests/test_reduction.rs`
  - 内容: 归约运算（sum/sum_axis/keepdims/empty/NaN/overflow）
  - 测试: `cargo test --test test_reduction`
  - 前置: T1
  - 预计: 10 min

- [ ] **T7a**: 实现 `tests/test_iterator.rs`
  - 文件: `tests/test_iterator.rs`
  - 内容: 迭代器（elements/axis/indexed）
  - 测试: `cargo test --test test_iterator`
  - 前置: T1
  - 预计: 10 min

- [ ] **T7b**: 实现 `tests/test_matrix.rs`
  - 文件: `tests/test_matrix.rs`
  - 内容: 向量内积（dot/complex/shape mismatch）
  - 测试: `cargo test --test test_matrix`
  - 前置: T1
  - 预计: 10 min

- [ ] **T7c**: 实现 `tests/test_set.rs`
  - 文件: `tests/test_set.rs`
  - 内容: 集合操作（unique 无序结果/整数/复数/NaN/±0.0）
  - 测试: `cargo test --test test_set`
  - 前置: T1
  - 预计: 10 min

- [ ] **T8**: 实现 `tests/test_shape.rs`
  - 文件: `tests/test_shape.rs`
  - 内容: 形状操作（transpose/高维）
  - 测试: `cargo test --test test_shape`
  - 前置: T1
  - 预计: 10 min

- [ ] **T9**: 实现 `tests/test_conversion.rs`
  - 文件: `tests/test_conversion.rs`
  - 内容: 类型转换（cast/存储模式转换）
  - 测试: `cargo test --test test_conversion`
  - 前置: T1
  - 预计: 10 min

- [ ] **T9a**: 实现 `tests/test_utility.rs`
  - 文件: `tests/test_utility.rs`
  - 内容: 实用操作（fill/clip/to_contiguous）
  - 测试: `cargo test --test test_utility`
  - 前置: T1
  - 预计: 10 min

- [ ] **T9b**: 实现 `tests/test_output.rs`
  - 文件: `tests/test_output.rs`
  - 内容: NumPy 风格输出（Display/Debug/截断/复数）
  - 测试: `cargo test --test test_output`
  - 前置: T1
  - 预计: 10 min

- [ ] **T10**: 实现 `tests/test_error.rs`
  - 文件: `tests/test_error.rs`
  - 内容: `XenonError` 边界与 display 输出验证（其中 workspace 相关公开边界统一断言 `XenonError::Workspace(...)`，内层 `WorkspaceError` 载荷继续在 `tests/test_workspace.rs` 中覆盖）
  - 测试: `cargo test --test test_error`
  - 前置: T1
  - 预计: 10 min

### Wave 3: 特化测试（可并行）

- [ ] **T11**: 实现 `tests/test_ffi.rs`
  - 文件: `tests/test_ffi.rs`
  - 内容: FFI 集成（指针/BLAS 兼容性辅助判断/export/export_mut/offset）
  - 测试: `cargo test --test test_ffi`
  - 前置: T2
  - 预计: 10 min

- [ ] **T12**: 实现 `tests/test_parallel.rs`
  - 文件: `tests/test_parallel.rs`
  - 内容: 启用 `parallel` feature 后公开 `sum`/`add` 行为一致性、并发读取、嵌套禁止
  - 测试: `cargo test --test test_parallel --features parallel`
  - 前置: T3, T7
  - 预计: 10 min

- [ ] **T13**: 实现 `tests/test_simd.rs`
  - 文件: `tests/test_simd.rs`
  - 内容: SIMD 结果一致性（add/sum/fallback）
  - 测试: `cargo test --test test_simd --features simd`
  - 前置: T3, T7
  - 预计: 10 min

- [ ] **T14**: 校验测试矩阵仅覆盖 `std` 环境
  - 文件: `.github/workflows/test.yml`
  - 内容: 维持 lib/tests/doctest 的 `std` 环境测试矩阵，不增加额外平台编译验证分支
  - 测试: CI 中运行默认测试矩阵
  - 前置: T2
  - 预计: 5 min

### Wave 4: 属性测试

- [ ] **T15**: 实现 `tests/property` 属性测试模块
  - 文件: `tests/property.rs`, `tests/property/tensor_props.rs`, `tests/property/ops_props.rs`, `tests/property/shape_props.rs`
  - 内容: transpose 自反/加法交换律/unique 不含重复
  - 测试: `cargo test --test property`
  - 前置: T3, T7, T8
  - 预计: 10 min

### Wave 5: CI 集成

- [ ] **T16**: 配置 CI 测试矩阵
  - 文件: `.github/workflows/test.yml`
  - 内容: `std` 环境下的默认/`parallel`/`simd`/全 feature 组合；不额外扩展平台矩阵，不设置覆盖率门禁
  - 测试: CI 触发运行
  - 前置: T1-T15
  - 预计: 10 min

### 并行执行分组图

```
Wave 1: [T1]
           │
Wave 2: [T2] [T3] [T4] [T5] [T6] [T7] [T7a] [T7b] [T7c] [T8] [T9] [T9a] [T9b] [T10]
           │    │              │                           │    │
           └────┴──────────────┴───────────────────────────┴────┘
                                   │
Wave 3:                   [T11] [T12] [T13] [T14]
                                   │
Wave 4:                           [T15]
                                   │
Wave 5:                           [T16]
```

---

## 附录：CI 测试矩阵与专项约束

### A.1 测试分类表

| 类型     | 位置                     | 目的                                  |
| -------- | ------------------------ | ------------------------------------- |
| 单元测试 | `#[cfg(test)] mod tests` | 验证单个函数/方法                     |
| 集成测试 | `tests/`                 | 验证跨模块交互                        |
| 边界测试 | 集成测试中标注           | 空张量、单元素、NaN/Inf、非连续、高维 |
| 属性测试 | `tests/property/`        | 参数化输入验证不变量                  |

### A.2 CI 测试矩阵

```yaml
# .github/workflows/test.yml
test:
  strategy:
    matrix:
      features:
        - ""
        - "--features parallel"
        - "--features simd"
        - "--all-features"
  steps:
    - name: Unit + Integration tests
      run: cargo test --lib --tests ${{ matrix.features }}

    - name: Doc tests
      run: cargo test --doc ${{ matrix.features }}
```

### A.3 Feature gate / 配置测试

| 配置       | 验证点                                                  |
| ---------- | ------------------------------------------------------- |
| 默认配置   | 核心集成测试与 doctest 在 `std` 默认配置下通过          |
| 启用并行   | `test_parallel.rs` 与相关一致性断言在 `parallel` 下通过 |
| 启用 SIMD  | `test_simd.rs` 与相关回退断言在 `simd` 下通过           |
| 全 feature | 测试矩阵组合启用时无冲突、无遗漏                        |

### A.4 类型边界 / 编译期测试

| 场景                      | 测试方式                                  |
| ------------------------- | ----------------------------------------- |
| `Send` / `Sync` 约束传播  | 编译期断言辅助函数与 `test_error.rs` 校验 |
| 非法元素类型与 trait 边界 | 编译期失败测试或等价约束验证              |
| feature gate 导出边界     | `cargo test` 配置矩阵与条件编译测试       |

### A.5 `require.md §28.5` 专项约束测试

| 约束                                                      | 验证方式                        |
| --------------------------------------------------------- | ------------------------------- |
| 验证 `usize` 仅承担元数据角色（索引、轴、形状、切片边界） | 编译期断言 + compile-fail 用例  |
| 验证 `usize` 不属于张量元素类型                           | `Tensor<usize, D>` compile-fail |
| 验证 `usize` 不参与逐元素算术或类型转换                   | trait-bound / compile-fail 用例 |
| 验证其他非法无符号元素类型边界不被放宽                    | trait-bound / compile-fail 用例 |

编译期失败测试采用仓库内 harness：通过 `cargo test` 驱动，并在测试体中调用 `rustc` / `cargo check` 校验预期失败，不新增测试专用 dev-dependency：

```text
tests/
├── compile_fail_tests.rs    # repository-local harness (top-level for Cargo discovery)
└── compile-fail/
    └── ui/
        ├── wrong_dimension_type.rs
        ├── missing_element_bound.rs
        ├── mismatched_storage_type.rs
        ├── unsigned_tensor_element_rejected.rs
        └── invalid_unsigned_element_rejected.rs
```

建议的 harness 形式如下（伪代码，需避免与辅助函数同名）：

```rust
// tests/compile_fail_tests.rs
#[test]
fn compile_fail_harness() {
    for case in collect_compile_fail_cases() {
        case.assert_compile_fail();
    }
}
```

编译期失败场景至少覆盖以下三类：

- 错误的维度类型：例如将不满足 `Dimension` 约束的类型传入 `Tensor<A, D>`，或把错误 rank 的 `IxDyn::from_slice(...)` 传给固定维度转换入口。
- 非法 trait bound：例如元素类型缺失 `Element` / `Numeric` / `RealScalar` 等必要约束，违反第 4 节元素类型边界。
- 不匹配的存储类型：例如要求 `StorageMut` 的 API 传入只读 view，或将不兼容的 storage representation 组合到同一签名中。
- `usize` / 非法无符号元素类型边界：例如拒绝 `Tensor<usize, Ix1>`、拒绝把 `usize` 送入逐元素算术或数值 cast，同时允许 `usize` 仅出现在 shape、axis、index 与切片边界等元数据位置。

这些 compile-fail 用例与运行时错误测试互补：前者验证“错误代码无法通过编译”，后者验证“合法代码在非法输入下返回正确错误语义”。

---

## 10. 错误处理与语义边界

本文档不直接定义错误类型，但要求所有测试断言、失败用例与 panic 校验统一遵循 `26-error.md` 的错误语义边界；测试层负责验证外部语义，不自创与实现分离的错误分类。

---

## 11. 设计决策记录

### 决策 1：按测试领域分文件

| 属性     | 值                                                                       |
| -------- | ------------------------------------------------------------------------ |
| 决策     | 按测试领域（overload/broadcast/reduction 等）而非按源码模块分文件        |
| 理由     | 集成测试关注跨模块行为；独立运行；编译并行化；失败定位清晰               |
| 替代方案 | 按源码模块分（test_dimension.rs, test_storage.rs）— 放弃，跨模块边界模糊 |

### 决策 2：使用参数化属性测试

| 属性     | 值                                                        |
| -------- | --------------------------------------------------------- |
| 决策     | 使用标准形状集合 + 参数化循环验证属性不变量               |
| 理由     | 不引入额外依赖，仍能稳定覆盖 transpose/交换律等关键不变量 |
| 替代方案 | 外部属性测试框架 — 放弃，与最小依赖原则冲突               |

### 决策 3：默认浮点比较采用混合容差契约

| 属性     | 值                                                                                                  |
| -------- | --------------------------------------------------------------------------------------------------- |
| 决策     | 浮点/复数默认比较遵循 `max(1 ULP, epsilon * |scalar_result|)` 混合容差契约；复数按实部/虚部分量分别判定。 |
| 理由     | 与统一数值契约一致，且能覆盖接近零值、跨执行路径舍入差异以及复数分量级比较的需求。                      |
| 替代方案 | 默认改用 NumPy 风格 `atol + rtol` — 放弃，与当前混合容差契约不一致。                                  |
| 替代方案 | 仅做逐位完全相等比较 — 放弃，会把允许的浮点舍入差异误判为失败。                                     |

### 决策 4：并行一致性测试为必须项

| 属性     | 值                                                                                                   |
| -------- | ---------------------------------------------------------------------------------------------------- |
| 决策     | 并行归约和逐元素运算在同执行路径下结果确定；浮点/复数跨执行路径允许文档化容差，整数/布尔仍须逐位一致 |
| 理由     | 与 `require.md §28.3` 对齐：浮点/复数允许舍入差异，整数与布尔仍要求严格一致                          |
| 替代方案 | 要求所有路径位级完全一致 — 放弃，会把允许的浮点舍入差异误判为失败                                    |

### 决策 5：测试矩阵仅覆盖 `std` 环境

| 属性     | 值                                                                                 |
| -------- | ---------------------------------------------------------------------------------- |
| 决策     | 集成测试、doctest 与 CI 测试矩阵仅覆盖 `std` 环境                                  |
| 理由     | 需求说明书 §1.3 已将平台支持限定为 `std`；继续维护额外平台测试分支会制造超范围契约 |
| 替代方案 | 保留额外平台编译检查 — 放弃，与当前版本范围矛盾                                    |

---

## 13. 平台与工程约束

| 约束项     | 约束内容                                           |
| ---------- | -------------------------------------------------- |
| 平台支持   | 集成测试仅覆盖 `std` 环境                          |
| crate 结构 | 测试方案依附当前单 crate，不拆分额外测试 crate     |
| 依赖约束   | 维持标准测试工具链，不为测试矩阵引入额外第三方依赖 |

---

## 版本历史

| 版本  | 日期       |
| ----- | ---------- |
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-08 |
| 1.1.0 | 2026-04-08 |
| 1.2.0 | 2026-04-08 |
| 1.2.1 | 2026-04-08 |
| 1.2.2 | 2026-04-10 |
| 1.2.3 | 2026-04-14 |
| 1.2.4 | 2026-04-15 |
| 1.2.5 | 2026-04-15 |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
