# 集成测试方案设计

> 本文档定义 Xenon 的集成测试策略、测试套件设计、属性测试方案及 CI 集成方案。
> 文档编号: 23 | 状态: 草案 | 依赖: `00-rust-standards.md`, `01-architecture-overview.md`, `require-v18.md`

---

## 1. 模块定位

### 1.1 集成测试目标

集成测试位于 `tests/` 目录，与 `src/` 内的单元测试（`#[cfg(test)] mod tests`）互为补充。单元测试验证单一函数/方法级别的正确性；集成测试验证**跨模块交互**与**端到端场景**的正确性。

| 目标 | 说明 |
|------|------|
| 跨模块正确性 | 验证 construction → shape ops → element ops → reduction 等链路的数据流通 |
| Feature-gate 组合 | 验证 `std`/`parallel`/`simd` 三种 feature 的合法组合均能编译并产生正确结果 |
| no_std 兼容性 | 验证 `no_std` + `alloc` 模式下 crate 可成功编译并链接 |
| FFI 安全性 | 验证所有 unsafe API 的前提条件，确保 safe 封装不会暴露 UB |
| 数值正确性 | 以 NumPy 为参考实现，验证运算结果在精度要求范围内 |
| 边界条件 | 空张量、单元素、NaN/Inf、非连续布局、高维（≥4维） |

### 1.2 与单元测试的边界

| 范畴 | 单元测试（`src/` 内） | 集成测试（`tests/`） |
|------|----------------------|---------------------|
| 维度互转 IxN ↔ IxDyn | ✅ | 仅验证跨模块使用场景 |
| 单个运算符正确性 | ✅ | 端到端链路验证 |
| Storage 单独行为 | ✅ | 存储 × 运算 × 形状 的组合验证 |
| Layout flags 计算 | ✅ | 切片/转置后 flags 的传递验证 |
| 错误类型构造 | ✅ | 错误在完整场景下的传播 |
| feature-gate 编译 | ❌ | ✅（集成测试独有） |
| no_std 构建 | ❌ | ✅（集成测试独有） |
| FFI 不变量 | ❌ | ✅（集成测试独有） |
| >1GB 内存测试 | ❌ | ✅（CI 分离执行） |

---

## 2. 文件位置

### 2.1 目录结构

```
tests/
├── common/                        # 共享测试工具模块
│   ├── mod.rs                     # 公共 re-export
│   ├── assertions.rs              # 自定义断言宏 (assert_close!, assert_tensor_eq!)
│   ├── fixtures.rs                # 测试数据工厂 (随机张量、参考数据)
│   └── numpy_ref.rs               # NumPy 参考值生成/加载工具
│
├── tensor_lifecycle.rs            # 创建、视图、修改、销毁全流程
├── feature_combinations.rs        # std/parallel/simd feature 组合矩阵
├── no_std_compat.rs               # no_std 构建验证
├── ffi_safety.rs                  # unsafe API 不变量验证
├── complex_ops.rs                 # Complex<T> 端到端测试
├── large_tensors.rs               # 大张量测试 (>1GB)
├── numpy_compat.rs                # NumPy 行为对比验证
│
├── construction.rs                # 构造函数集成测试
├── arithmetic.rs                  # 算术运算集成测试
├── broadcasting.rs                # 广播集成测试
├── indexing.rs                    # 索引集成测试
├── shape_ops.rs                   # 形状操作集成测试
├── reduction.rs                   # 归约集成测试
├── matrix_ops.rs                  # 矩阵运算集成测试
├── set_ops.rs                     # 集合操作集成测试
├── iter_integration.rs            # 迭代器集成测试
├── workspace.rs                   # 工作空间集成测试
├── edge_cases.rs                  # 边界用例 (空张量、NaN/Inf、非连续)
│
└── property/                      # 属性测试
    ├── mod.rs                     # Strategy 定义、共享配置
    ├── algebraic_props.rs         # 代数性质 (结合律、交换律、分配律)
    ├── broadcast_rules.rs         # 广播不变量
    ├── reduction_props.rs         # 归约不变量
    ├── shape_invariants.rs        # 形状操作不变量
    └── complex_props.rs           # 复数运算不变量
```

### 2.2 各文件职责概要

| 文件 | 核心验证点 | 预计测试数 |
|------|-----------|-----------|
| `tensor_lifecycle.rs` | 存储 × 维度 × 操作 的完整生命周期 | ~40 |
| `feature_combinations.rs` | 7 种 feature 组合的编译与结果一致性 | ~30 |
| `no_std_compat.rs` | `--no-default-features` 构建通过 | ~5 |
| `ffi_safety.rs` | 所有 unsafe API 的前提条件与边界 | ~35 |
| `complex_ops.rs` | Complex 算术、共轭、极坐标、FFI 布局 | ~50 |
| `large_tensors.rs` | >1GB 分配、并行正确性、内存回收 | ~10 |
| `numpy_compat.rs` | NumPy 参考值对比（重点运算） | ~60 |
| `construction.rs` | 构造函数 + 维度 + 元素类型组合 | ~30 |
| `arithmetic.rs` | 逐元素运算 + 运算符 + 类型提升 | ~40 |
| `broadcasting.rs` | 广播规则 + shape 兼容性 + 错误 | ~25 |
| `indexing.rs` | 多维索引 + 切片 + 高级索引 | ~35 |
| `shape_ops.rs` | reshape/transpose/squeeze/split/pad | ~40 |
| `reduction.rs` | 全局/沿轴/累积归约 + 边界 | ~35 |
| `matrix_ops.rs` | matvec/dot/outer/batch + 布局 | ~30 |
| `set_ops.rs` | unique/bincount/histogram | ~25 |
| `iter_integration.rs` | 元素/轴/窗口/zip 迭代 + 遍历顺序 | ~30 |
| `workspace.rs` | 分配/分割/嵌套/scratch 查询 | ~20 |
| `edge_cases.rs` | 空张量、单元素、NaN/Inf、非连续、高维 | ~50 |
| `property/` | 代数性质、不变量（随机化） | ~25 |

---

## 3. 依赖关系

### 3.1 测试专用依赖 (`[dev-dependencies]`)

```toml
[dev-dependencies]
# Property-based testing
proptest = "1.6"

# NumPy reference value generation (Python interop)
# 仅在本地生成参考数据时需要，CI 中使用预生成数据

# Temporary for no_std build verification
# 使用 cargo build --no-default-features 而非运行测试
```

> **设计决策**：不引入 `ndarray` 或 `nalgebra` 作为对比参考。NumPy 参考值通过预生成 JSON 文件提供，避免运行时 Python 依赖。

### 3.2 测试模块内部依赖

```
common/
  ├── assertions.rs       ← 被所有测试文件引用
  ├── fixtures.rs         ← 被 tensor_lifecycle, edge_cases, property/ 引用
  └── numpy_ref.rs        ← 被 numpy_compat, complex_ops 引用

各测试文件之间无依赖，可独立编译和执行。
```

### 3.3 NumPy 参考数据管理

```
tests/
└── data/
    └── numpy_references/
        ├── arithmetic_f64.json        # {input_shape, a, b, expected_add, expected_sub, ...}
        ├── reduction_f64.json         # {input, expected_sum, expected_mean, ...}
        ├── trigonometry_f64.json      # {input, expected_sin, expected_cos, ...}
        ├── complex_f64.json           # {input_re, input_im, expected_exp, expected_ln, ...}
        ├── broadcast_cases.json       # {a_shape, b_shape, a, b, expected_add, ...}
        ├── linalg_f64.json            # {mat_shape, vec, expected_matvec, ...}
        └── generate_refs.py           # NumPy 参考值生成脚本
```

**参考数据生成流程**：

1. 本地运行 `python3 tests/data/numpy_references/generate_refs.py`
2. 脚本输出 JSON 文件到 `tests/data/numpy_references/`
3. JSON 文件提交到 Git（二进制稳定性）
4. 集成测试通过 `numpy_ref.rs` 工具模块加载和解析

**JSON 格式约定**：

```json
{
    "description": "element-wise add, f64, shape [3,4]",
    "dtype": "float64",
    "shape": [3, 4],
    "a": [0.0, 1.0, 2.0, ...],
    "b": [10.0, 20.0, 30.0, ...],
    "expected_add": [10.0, 21.0, 32.0, ...],
    "tolerance": 1e-15
}
```

---

## 4. 每个 Test Suite 的具体设计

### 4.1 `tensor_lifecycle.rs` — 创建、视图、修改、销毁

验证 TensorBase<S, D> 四种存储模式 × 多种维度类型的完整生命周期。

#### 测试场景

| 场景 | 验证点 |
|------|--------|
| Owned 创建 → 读取 → 修改 → drop | 数据正确、对齐正确、无内存泄漏 |
| Owned → View → 读取 → View drop → Owned 读取 | 视图共享底层数据、视图 drop 不影响 Owned |
| Owned → ViewMut → 修改 → ViewMut drop → Owned 读取 | 独占修改正确反映到 Owned |
| Owned → ArcTensor clone → 两者读取 → 各自 drop | 引用计数正确、浅拷贝语义 |
| ArcTensor::make_mut() 修改 → 检查是否 CoW | 引用计数 >1 时深拷贝、=1 时原地修改 |
| View → reshape → 读取 | reshape 视图数据正确 |
| ViewMut → slice → 修改 | 多级视图修改正确 |
| Ix2 → slice → Ix1 视图 | 降维视图步长正确 |
| IxDyn 创建 → 操作 → 转回 Ix2 | 动态维度交互操作正确 |

#### 关键测试函数签名

```rust
#[test]
fn test_owned_create_read_modify_drop_f64_ix2()

#[test]
fn test_view_shares_underlying_data()

#[test]
fn test_view_mut_modifications_visible_in_owned()

#[test]
fn test_arc_tensor_shallow_clone_and_cow()

#[test]
fn test_arc_make_mut_triggers_deep_copy_when_ref_count_gt_1()

#[test]
fn test_arc_make_mut_in_place_when_ref_count_eq_1()

#[test]
fn test_chained_views_shared_data_integrity()

#[test]
fn test_owned_to_view_to_view_mut_lifetime_ordering()

#[test]
fn test_reshape_view_data_correctness()

#[test]
fn test_ixdyn_roundtrip_with_ix2()
```

#### 预期结果

- 所有视图操作零拷贝（仅元数据变更）
- Layout flags 在视图创建后正确更新
- Arc 引用计数在 clone/drop 后精确
- make_mut 在 ref_count > 1 时产生独立的内存分配

---

### 4.2 `feature_combinations.rs` — Feature 组合矩阵

验证 7 种合法 feature 组合下，核心运算结果一致。

#### Feature 组合矩阵

| 编号 | Features | 说明 |
|------|----------|------|
| F1 | `std` | 默认配置（基准） |
| F2 | `std + parallel` | 启用 rayon 并行 |
| F3 | `std + simd` | 启用 pulp SIMD |
| F4 | `std + parallel + simd` | 全特性 |
| F5 | `parallel` only | 无 std，仅有并行 |
| F6 | `simd` only | 无 std，仅有 SIMD |
| F7 | `no default features` | 纯 no_std + alloc |

> **注意**：F5/F6/F7 需要在 `--no-default-features` 基础上添加对应 feature。实际测试中，`parallel` 和 `simd` 的 no_std 组合是否可行取决于 rayon/pulp 的 no_std 支持。若不可行则标记为 compile-fail 测试。

#### 测试策略

由于 `cargo test` 一次只能使用一组 feature，此文件使用 **条件编译** 组织测试：

```rust
// 所有 feature 组合共享的测试逻辑
// 通过 cfg 守卫确保只在对应 feature 下运行

#[cfg(feature = "std")]
#[test]
fn test_std_arithmetic_results() {
    // 基准运算结果
}

#[cfg(feature = "parallel")]
#[test]
fn test_parallel_results_match_scalar() {
    // 验证并行结果与标量结果一致
}
```

#### CI 执行方式

通过 CI matrix 为每个 feature 组合执行独立的 `cargo test`：

```yaml
strategy:
  matrix:
    features:
      - "--features std"
      - "--features std,parallel"
      - "--features std,simd"
      - "--features std,parallel,simd"
      - "--no-default-features --features parallel"
      - "--no-default-features --features simd"
      - "--no-default-features"
```

#### 一致性验证

| 运算 | 验证方式 |
|------|----------|
| 逐元素 add/mul (f64, 10000 元素) | 所有组合结果逐位相同 |
| sum 归约 (f64, 100000 元素) | 误差 < 1e-15 |
| matvec (f64, 1000×1000) | 误差 < 1e-13 |
| 非连续数组运算 | 结果与连续数组 to_owned() 后一致 |

---

### 4.3 `no_std_compat.rs` — no_std 构建验证

验证 `--no-default-features`（即 `no_std + alloc`）模式下 crate 可编译。

#### 测试策略：仅构建验证（无运行时测试）

`no_std` 目标无法在标准 CI 环境运行测试（无 `std::io`、无 test harness）。策略为 **编译通过即通过**。

```rust
// 此文件验证 no_std 下公共 API 的可用性
// 仅编译，不运行（no_std 目标无 test runner）

#![no_std]

extern crate alloc;

use xenon::{Tensor, TensorView, Ix2, zeros, ones};
use xenon::{Element, Numeric};

// 验证公共 API 在 no_std 下可引用
fn _verify_api_availability() {
    let _shape = [3usize, 4];
    // 不能调用 zeros/ones（需要运行时），但验证类型解析
}
```

#### CI 验证方式

```bash
# 编译验证（不运行测试）
cargo build --no-default-features

# 交叉编译到裸机目标（可选，Phase 5+）
cargo build --no-default-features --target thumbv7em-none-eabihf
```

#### QEMU 执行策略（远期）

| 阶段 | 策略 | 说明 |
|------|------|------|
| Phase 5 | 编译验证 | 仅 `cargo build --no-default-features` |
| 远期 | QEMU 执行 | 使用 `cargo test --target aarch64-unknown-none` + QEMU user mode |

---

### 4.4 `ffi_safety.rs` — unsafe API 不变量验证

验证所有 unsafe 函数的前提条件，确保 safe 封装的正确性。

#### 测试的不变量

| 不变量 | 验证方式 |
|--------|----------|
| `as_ptr()` 返回非空、对齐指针 | 检查指针非 null 且地址 % align == 0 |
| `as_mut_ptr()` 独占访问保证 | 通过 borrow checker 静态验证；运行时验证指针值正确 |
| `from_raw_parts()` 生命周期安全 | 构造合法视图并验证数据正确 |
| `from_raw_parts()` 非法参数检测 | 验证文档中声明的前提条件（null 指针等） |
| `as_ptr_unchecked()` offset 计算 | 与 checked 版本对比，结果一致 |
| `index_to_offset()` 正确性 | 与手动计算的偏移量对比 |
| `into_raw_parts()` / `from_raw_parts()` 往返 | 解构后重构，数据不变 |
| BLAS 兼容性检查 | F-contiguous 数组返回正确的 lda/layout |

#### 关键测试函数

```rust
#[test]
fn test_as_ptr_alignment_64bytes() {
    let t: Tensor<f64, Ix2> = zeros([8, 8]);
    let ptr = t.as_ptr();
    assert!(!ptr.is_null());
    assert_eq!(ptr as usize % 64, 0, "pointer should be 64-byte aligned");
}

#[test]
fn test_raw_parts_roundtrip_preserves_data() {
    let original: Tensor<f64, Ix2> = from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    let (ptr, shape, strides, offset) = original.into_raw_parts();
    // SAFETY: ptr is valid, shape/strides come from a valid tensor
    let view = unsafe { TensorView::from_raw_parts(ptr, shape, strides, offset) };
    // verify data integrity...
}

#[test]
fn test_blas_layout_f_contiguous_matrix() {
    let t: Tensor<f64, Ix2> = zeros([4, 5]).with_order(Order::F);
    assert!(t.is_blas_compatible());
    assert_eq!(t.blas_layout(), Some(BlasLayout::ColumnMajor));
    assert_eq!(t.lda(), 4); // leading dimension = first axis size
}

#[test]
fn test_blas_layout_transposed_not_contiguous() {
    let t: Tensor<f64, Ix2> = zeros([4, 5]).with_order(Order::F);
    let transposed = t.t();
    assert!(!transposed.is_blas_compatible());
}

#[test]
fn test_index_to_offset_various_strides() {
    let t: Tensor<f64, Ix2> = zeros([3, 4]).with_order(Order::F);
    // F-order strides: [1, 3]
    assert_eq!(t.index_to_offset(&[0, 0]), 0);
    assert_eq!(t.index_to_offset(&[1, 0]), 1);
    assert_eq!(t.index_to_offset(&[0, 1]), 3);
    assert_eq!(t.index_to_offset(&[2, 3]), 2 + 3 * 3);
}
```

#### 安全封装验证

```rust
#[test]
fn test_checked_vs_unchecked_index_equivalence() {
    let t: Tensor<f64, Ix2> = from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    let index = [1, 2];
    let checked = t.get(&index).unwrap();
    // SAFETY: index is within bounds
    let unchecked = unsafe { t.get_unchecked(&index) };
    assert_eq!(checked, unchecked);
}

#[test]
#[should_panic(expected = "index out of bounds")]
fn test_checked_index_panics_on_oob() {
    let t: Tensor<f64, Ix2> = zeros([3, 4]);
    let _ = t.get(&[5, 5]);
}
```

---

### 4.5 `complex_ops.rs` — Complex\<T\> 端到端测试

验证自定义 `Complex<T>` 类型的全链路正确性。

#### 测试场景

| 场景 | 验证点 |
|------|--------|
| Complex 构造与字段访问 | `Complex::new(3.0, 4.0).re == 3.0` |
| Complex 算术运算 | 加减乘除、与实数互操作 |
| Complex 运算符 | `+`, `-`, `*`, `/`, unary `-` |
| Complex 特殊方法 | `conj()`, `norm()`, `arg()`, `from_polar()` |
| Complex 超越函数 | `exp()`, `ln()`, `sqrt()` |
| Complex 数组运算 | `Tensor<Complex<f64>, Ix2>` 的逐元素运算 |
| Complex 归约 | `sum()`、`dot()` 对复数数组 |
| Complex 广播 | 复数 × 实数广播 |
| Complex 近似比较 | `approx_eq()` 方法验证 |
| Complex 内存布局 | `#[repr(C)]`，大小 = 2 × size_of::<T>() |

#### 关键测试函数

```rust
#[test]
fn test_complex_memory_layout_repr_c() {
    use core::mem::{size_of, align_of};
    assert_eq!(size_of::<Complex<f64>>(), 2 * size_of::<f64>());
    assert_eq!(align_of::<Complex<f64>>(), align_of::<f64>());
    // Verify field order: [re, im]
    let c = Complex::new(1.0f64, 2.0f64);
    let ptr = &c as *const _ as *const f64;
    assert_eq!(unsafe { *ptr }, 1.0);       // re
    assert_eq!(unsafe { *ptr.add(1) }, 2.0); // im
}

#[test]
fn test_complex_norm_uses_hypot() {
    // hypot avoids overflow for large components
    let c = Complex::new(1e200f64, 1e200f64);
    let n = c.norm();
    assert!(n.is_finite(), "norm should use hypot, expected finite but got {n}");
    assert!((n - 1.4142135623730951e200).abs() < 1e186);
}

#[test]
fn test_complex_tensor_element_wise_mul() {
    let a: Tensor1<Complex<f64>> = from_vec(vec![
        Complex::new(1.0, 2.0),
        Complex::new(3.0, 4.0),
    ]);
    let b: Tensor1<Complex<f64>> = from_vec(vec![
        Complex::new(5.0, 6.0),
        Complex::new(7.0, 8.0),
    ]);
    let result = &a * &b;
    // (1+2i)(5+6i) = 5+6i+10i+12i² = -7+16i
    assert_eq!(result[0], Complex::new(-7.0, 16.0));
    // (3+4i)(7+8i) = 21+24i+28i+32i² = -11+52i
    assert_eq!(result[1], Complex::new(-11.0, 52.0));
}

#[test]
fn test_complex_real_interop_add() {
    let c = Complex::new(3.0f64, 4.0f64);
    let r: f64 = 5.0;
    assert_eq!(c + r, Complex::new(8.0, 4.0));
    assert_eq!(r + c, Complex::new(8.0, 4.0)); // commutativity
}

#[test]
fn test_complex_real_interop_div() {
    // real / complex: r / (a+bi) = r(a-bi)/(a²+b²)
    let c = Complex::new(3.0f64, 4.0f64); // norm² = 25
    let r: f64 = 10.0;
    let result = r / c;
    // 10(3-4i)/25 = 30/25 - 40/25 i = 1.2 - 1.6i
    assert!(result.approx_eq(Complex::new(1.2, -1.6), 1e-14));
}

#[test]
fn test_complex_exp_euler_identity() {
    // e^(iπ) + 1 = 0
    let i_pi = Complex::new(0.0f64, core::f64::consts::PI);
    let result = i_pi.exp();
    assert!(result.approx_eq(Complex::new(-1.0, 0.0), 1e-14));
}
```

#### NumPy 参考验证

```rust
#[test]
fn test_complex_tensor_sum_matches_numpy() {
    // Reference: numpy.sum([1+2j, 3+4j, 5+6j]) = (9+12j)
    let t: Tensor1<Complex<f64>> = from_vec(vec![
        Complex::new(1.0, 2.0),
        Complex::new(3.0, 4.0),
        Complex::new(5.0, 6.0),
    ]);
    let sum = t.sum();
    assert_eq!(sum, Complex::new(9.0, 12.0));
}
```

---

### 4.6 `large_tensors.rs` — 大张量测试 (>1GB)

验证大内存分配和并行计算的正确性。**此测试套件默认跳过**，仅在 CI 中通过环境变量选择性执行。

#### 触发条件

```rust
// 每个 large tensor test 都需要显式 opt-in
fn should_run_large_tests() -> bool {
    std::env::var("XENON_LARGE_TESTS")
        .map(|v| v == "1" || v == "true")
        .unwrap_or(false)
}

#[test]
fn test_large_allocation_f64() {
    if !should_run_large_tests() { return; }
    // ...
}
```

#### 测试场景

| 场景 | 规格 | 验证点 |
|------|------|--------|
| 1D 大数组分配 | 200M × f64 ≈ 1.6GB | 分配成功、数据可读写 |
| 2D 矩阵分配 | 10000 × 10000 × f64 ≈ 800MB | F-order 步长正确、对齐正确 |
| 并行填充 + 归约 | 200M 元素，random fill + sum | 并行结果与串行一致 |
| 广播大数组 | (1, 100000) + (100000, 1) | 广播结果正确、无越界 |
| 大矩阵 matvec | 10000 × 10000 × 10000 | 结果正确、无 OOM |

#### 内存安全验证

```rust
#[test]
fn test_large_tensor_parallel_sum_correctness() {
    if !should_run_large_tests() { return; }
    let n = 200_000_000usize; // ~1.6GB for f64
    let t: Tensor1<f64> = full(n, 0.5);
    let expected = 0.5f64 * n as f64;
    let result = t.sum();
    let rel_err = (result - expected).abs() / expected;
    assert!(rel_err < 1e-13, "parallel sum relative error: {rel_err}");
}
```

---

### 4.7 `numpy_compat.rs` — NumPy 行为对比验证

以 NumPy 为参考实现，验证 Xenon 运算结果在精度范围内一致。

#### 参考值覆盖范围

| 类别 | 运算 | 数据类型 | 精度阈值 |
|------|------|----------|----------|
| 算术 | add, sub, mul, div | f64 | 1e-15 |
| 三角函数 | sin, cos, tan | f64 | 1e-14 |
| 指数/对数 | exp, ln, log2, log10 | f64 | 1e-14 |
| 归约 | sum, prod, mean, var, std | f64 | 1e-15 / 1e-13 |
| 矩阵 | matvec, dot, outer | f64 | 1e-13 |
| 形状 | reshape, transpose, pad | f64 | 精确 |
| 广播 | 各种形状组合 | f64 | 1e-15 |
| 集合 | unique, bincount, histogram | f64, i64 | 精确 |

#### 测试结构

```rust
#[test]
fn test_add_f64_2d_matches_numpy() {
    let ref_data = NumpyRef::load("arithmetic_f64.json");
    let a: Tensor2<f64> = from_vec(ref_data.a, ref_data.shape);
    let b: Tensor2<f64> = from_vec(ref_data.b, ref_data.shape);
    let result = &a + &b;
    assert_tensor_close!(&result, &ref_data.expected_add, ref_data.tolerance);
}
```

#### NumPy 参考值生成脚本设计

`generate_refs.py` 覆盖以下场景：

```python
import numpy as np
import json

def generate_arithmetic_refs():
    """Generate reference values for element-wise arithmetic."""
    np.random.seed(42)
    shape = (3, 4)
    a = np.random.randn(*shape)
    b = np.random.randn(*shape) + 0.5  # avoid zeros for div
    return {
        "description": "element-wise arithmetic, f64, shape [3,4]",
        "dtype": "float64",
        "shape": list(shape),
        "a": a.flatten('F').tolist(),   # F-order flatten
        "b": b.flatten('F').tolist(),
        "expected_add": (a + b).flatten('F').tolist(),
        "expected_sub": (a - b).flatten('F').tolist(),
        "expected_mul": (a * b).flatten('F').tolist(),
        "expected_div": (a / b).flatten('F').tolist(),
        "tolerance": 1e-15,
    }
```

---

### 4.8 各领域测试套件补充设计

#### `construction.rs` — 构造函数集成测试

| 测试 | 验证点 |
|------|--------|
| `zeros` × (Ix0, Ix1, Ix2, Ix3, IxDyn) × (f32, f64, i32, bool) | 形状和元素正确 |
| `ones` + `full` + `empty` + `fill` | 填充值正确性 |
| `eye` / `identity` 对角矩阵 | 对角线为 1、其余为 0 |
| `diag` 从向量构造对角矩阵 | 正确提取/构造 |
| `arange` / `linspace` / `logspace` | 序列值和步长正确 |
| `from_vec` F-order vs C-order | 步长计算正确 |

#### `arithmetic.rs` — 算术运算集成测试

| 测试 | 验证点 |
|------|--------|
| 逐元素 add/sub/mul/div × (f32, f64, i32) | 类型特化正确 |
| 运算符重载 `&a + &b`, `a + b`, `&a + 1.0` | 所有权和广播 |
| 复合赋值 `a += &b` | 原地修改正确 |
| Neg, Not 一元运算 | 符号反转和逻辑非 |
| cast 后运算 | 跨类型运算精度 |
| bool 仅支持逻辑运算 | `all`/`any`/位运算/条件选择 |

#### `broadcasting.rs` — 广播集成测试

| 测试 | 验证点 |
|------|--------|
| (3,1) + (1,4) → (3,4) | 基本广播 |
| (5,) + (3,5) → (3,5) | 1D + 2D |
| 标量 + 矩阵 | 0-dim 广播 |
| 不兼容广播 → BroadcastError | 错误返回 |
| 广播视图只读 | 写操作编译期拒绝 |
| 原地广播 `a += &b` | b 可广播、a 不可广播 |

#### `indexing.rs` — 索引集成测试

| 测试 | 验证点 |
|------|--------|
| 多维索引 `[i, j, k]` | 值正确 |
| 范围索引 `s![0..2, ..]` | 切片边界正确 |
| `take` / `take_along_axis` | 高级索引正确 |
| `put` / `mask` / `compress` | 写入索引正确 |
| `where` 条件选择 | 广播 + 条件 |
| `argwhere` / `nonzero` | 索引返回正确 |
| 越界 panic | `#[should_panic]` |

#### `shape_ops.rs` — 形状操作集成测试

| 测试 | 验证点 |
|------|--------|
| reshape 保持元素总数 | 数据不变、步长重新计算 |
| transpose 步长交换 | F→C 连续性变化 |
| squeeze / unsqueeze | 降维/升维正确 |
| split / chunk / unstack | 分割结果正确、零拷贝 |
| pad 三种模式 | Constant/Edge/Reflect |
| repeat / tile | 重复正确 |
| permute / swapaxes / moveaxis | 轴重排正确 |

#### `reduction.rs` — 归约集成测试

| 测试 | 验证点 |
|------|--------|
| sum/prod/mean/var/std 全局归约 | 数值精度 |
| 沿轴归约 | shape 降维正确 |
| cumsum/cumprod | 累积结果正确 |
| argmin/argmax | 第一个出现的索引 |
| min/max 含 NaN | NaN 传播 |
| 空数组归约 | 返回 Result |
| 整数 sum 溢出 | panic |

#### `matrix_ops.rs` — 矩阵运算集成测试

| 测试 | 验证点 |
|------|--------|
| matvec F/C-order | 结果一致 |
| dot / inner / outer | 维度和值正确 |
| batch_matvec / batch_dot | 批量轴正确 |
| batch_add / batch_scale | 广播 + 批量 |
| BLAS 兼容布局 | lda/layout 正确 |

#### `set_ops.rs` — 集合操作集成测试

| 测试 | 验证点 |
|------|--------|
| unique 排序、空数组 | 升序唯一值 |
| unique_counts | 值和计数对齐 |
| unique_inverse | 重建原数组验证 |
| bincount minlength、负值 panic | 计数正确 |
| histogram bins 和 range | bin 边界正确 |

#### `iter_integration.rs` — 迭代器集成测试

| 测试 | 验证点 |
|------|--------|
| 元素迭代 F/C 顺序 | 遍历顺序正确 |
| 轴迭代 | 沿轴产出子视图 |
| 窗口迭代 | 窗口大小和步进 |
| zip 同形/广播 | 同步产出正确 |
| 空数组迭代 | 零产出 |
| 并行迭代 (`parallel` feature) | 结果与串行一致 |

#### `workspace.rs` — 工作空间集成测试

| 测试 | 验证点 |
|------|--------|
| 分配 + 对齐 | 64 字节对齐 |
| split_at 分割 | 不重叠、独立借出 |
| 递归分割 | 多级分割正确 |
| 扩容 | 增长后保持对齐 |
| scratch_size 查询 | 纯计算、零分配 |
| ScratchNeed 合并 | sum/max 组合正确 |

#### `edge_cases.rs` — 边界用例

| 类别 | 测试 |
|------|------|
| 空张量 | 0 长度轴、空归约、空迭代 |
| 单元素 | Ix0 标量运算、单元素归约 |
| NaN/Inf | NaN 传播、Inf 算术、0.0/0.0→NaN、1.0/0.0→Inf |
| subnormal | 运算不 panic |
| 非连续布局 | 转置后运算、切片后归约、广播非连续 |
| 高维 (≥4维) | Ix4/Ix5/Ix6 构造、索引、归约 |
| 大步长 | 负步长翻转、零步长广播 |
| 极端形状 | (1,1,...,1) 全 1 形状、(1000000,) 长向量 |

---

## 5. 属性测试 (Property-based Testing)

### 5.1 框架选择

使用 `proptest` crate，原因：
- 支持自定义 Strategy 和 Shrinking
- 与 Rust 生态系统成熟集成
- 支持确定性复现（seed-based）

### 5.2 Strategy 设计

#### 张量生成 Strategy

```rust
use proptest::prelude::*;

/// Generate a random shape with ndim in [1, 4], each dimension in [1, 20].
fn shape_strategy() -> impl Strategy<Value = Vec<usize>> {
    (1..=4usize).prop_flat_map(|ndim| {
        proptest::collection::vec(1..=20usize, ndim)
    })
}

/// Generate a random Tensor<f64, IxDyn> with the given shape.
fn tensor_f64_strategy() -> impl Strategy<Value = Tensor<f64, IxDyn>> {
    shape_strategy().prop_flat_map(|shape| {
        let n: usize = shape.iter().product();
        proptest::collection::vec(-100.0f64..100.0f64, n)
            .prop_map(move |data| {
                Tensor::from_vec(data, shape.clone()).unwrap()
            })
    })
}

/// Generate two broadcastable shapes.
fn broadcastable_shapes_strategy() -> impl Strategy<Value = (Vec<usize>, Vec<usize>)> {
    shape_strategy().prop_flat_map(|base_shape| {
        let ndim = base_shape.len();
        // Generate a compatible shape by replacing some dims with 1
        let compatible: Vec<usize> = base_shape.iter().map(|&d| {
            if d == 1 { 1 } else { 1usize } // simplified
        }).collect();
        // This is illustrative; real strategy is more nuanced
        Just((base_shape.clone(), compatible))
    })
}
```

### 5.3 待测属性

#### 代数性质

| 属性 | 运算 | 条件 | Strategy |
|------|------|------|----------|
| 交换律 | add, mul | 同形状 f64 张量 | `tensor_f64_strategy()` × 2 |
| 结合律 | add, mul | 同形状 f64 张量 | `tensor_f64_strategy()` × 3 |
| 分配律 | mul over add | 同形状 f64 张量 | `tensor_f64_strategy()` × 3 |
| 零元 | add + zero tensor | 任意 f64 张量 | `tensor_f64_strategy()` |
| 单位元 | mul + one tensor | 任意 f64 张量 | `tensor_f64_strategy()` |

```rust
proptest! {
    #[test]
    fn prop_add_commutative(a in tensor_f64_strategy(), b in tensor_f64_strategy()) {
        // Only test when shapes match
        if a.shape() == b.shape() {
            let sum_ab = &a + &b;
            let sum_ba = &b + &a;
            assert_tensor_close!(&sum_ab, &sum_ba, 1e-14);
        }
    }

    #[test]
    fn prop_add_associative(
        a in tensor_f64_strategy(),
        b in tensor_f64_strategy(),
        c in tensor_f64_strategy()
    ) {
        if a.shape() == b.shape() && b.shape() == c.shape() {
            let left = &(&a + &b) + &c;
            let right = &a + &(&b + &c);
            assert_tensor_close!(&left, &right, 1e-13);
        }
    }

    #[test]
    fn prop_mul_distributes_over_add(
        a in tensor_f64_strategy(),
        b in tensor_f64_strategy(),
        c in tensor_f64_strategy()
    ) {
        if a.shape() == b.shape() && b.shape() == c.shape() {
            let left = &a * &(&b + &c);
            let right = &(&a * &b) + &(&a * &c);
            assert_tensor_close!(&left, &right, 1e-12);
        }
    }
}
```

#### 归约不变量

| 属性 | 验证 |
|------|------|
| sum(全零) = 0 | 对任意形状 |
| prod(全一) = 1 | 对任意形状 |
| sum 沿轴 = 全局 sum | 沿轴后求和 = 直接全局求和 |
| min ≤ mean ≤ max | 对不含 NaN 的有限值数组 |
| cumsum[-1] == sum | 累积求和末元素 = 全局求和 |

#### 形状操作不变量

| 属性 | 验证 |
|------|------|
| reshape 再 reshape 回原形 | 数据不变 |
| transpose 两次 = 原始 | 数据不变 |
| broadcast shape 正确 | 结果形状 = 广播规则计算 |
| split + cat = 原始 | 分割再拼接恢复 |
| pad 零宽度 = 原始 | 等价于 to_owned() |

#### 复数运算不变量

| 属性 | 验证 |
|------|------|
| z * conj(z) = norm(z)² | 任意复数 |
| exp(ln(z)) = z | z ≠ 0 |
| from_polar(norm(z), arg(z)) = z | z ≠ 0 |

### 5.4 Shrinking 策略

proptest 内置 shrinking 对数值型 Strategy 表现良好。对张量 Strategy 的额外 shrinking 配置：

```rust
proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,           // 256 test cases per property
        max_shrink_iters: 1024,
        ..Default::default()
    })]

    // ...
}
```

对形状的 shrinking：优先缩小维度数（ndim 向 1 收缩），再缩小各轴长度（向 1 收缩）。这确保发现失败时的最小复现形状尽可能简单。

---

## 6. CI 集成

### 6.1 测试矩阵

#### 平台矩阵

| 平台 | 目标 | Feature 组合 | 说明 |
|------|------|-------------|------|
| Linux x86_64 | `x86_64-unknown-linux-gnu` | 全部 7 组 | 主要 CI |
| macOS ARM64 | `aarch64-apple-darwin` | F1, F4 | 验证 NEON 路径 |
| Windows x86_64 | `x86_64-pc-windows-msvc` | F1, F4 | 基本验证 |
| Linux ARM64 | `aarch64-unknown-linux-gnu` | F1 | 交叉编译验证 |
| no_std 目标 | `thumbv7em-none-eabihf` | F7 | 编译验证 |

#### Feature 组合详细命令

```yaml
test_matrix:
  - name: "std (default)"
    command: "cargo test"
    features: ""

  - name: "std + parallel"
    command: "cargo test --features parallel"
    features: "parallel"

  - name: "std + simd"
    command: "cargo test --features simd"
    features: "simd"

  - name: "full (std + parallel + simd)"
    command: "cargo test --features full"
    features: "full"

  - name: "no_std build"
    command: "cargo build --no-default-features"
    features: ""

  - name: "large tensor tests"
    command: "XENON_LARGE_TESTS=1 cargo test --test large_tensors --features full"
    features: "full"
```

### 6.2 覆盖率测量

使用 `cargo-llvm-cov` 测量覆盖率：

```bash
# Install
cargo install cargo-llvm-cov

# Run with coverage
cargo llvm-cov --features full --html

# Enforce threshold
cargo llvm-cov --features full --fail-under-lines 80
```

#### 覆盖率目标

| 模块 | 行覆盖率目标 |
|------|-------------|
| `dimension` | ≥ 90% |
| `element` / `complex` | ≥ 90% |
| `storage` | ≥ 85% |
| `layout` | ≥ 85% |
| `tensor` | ≥ 85% |
| `construction` | ≥ 80% |
| `ops/` | ≥ 80% |
| `shape/` | ≥ 80% |
| `ffi` | ≥ 80% |
| `backend/` | ≥ 70% (SIMD 路径依赖运行时 CPU) |
| **整体** | **≥ 80%** |

### 6.3 超时与 Flaky 测试处理

#### 超时配置

| 测试类别 | 超时 |
|----------|------|
| 常规集成测试 | 60s / test |
| large_tensors | 300s / test |
| 属性测试 (proptest) | 120s / test |

```yaml
# CI configuration
env:
  RUST_TEST_TIME_INTEGRATION: "60,120"     # default, max
  RUST_TEST_TIME_LARGE: "300,600"
```

#### Flaky 测试策略

| 策略 | 实现 |
|------|------|
| 确定性种子 | proptest 使用固定 seed，CI 中通过 `PROPTEST_SEED` 环境变量控制 |
| 失败重跑 | CI 中失败的测试自动重跑一次 |
| 隔离 | 每个测试文件独立编译、独立运行 |
| 并行测试分离 | `large_tensors` 和 `property/` 与常规测试分离执行 |
| 日志 | 测试失败时打印完整的环境信息（feature、platform、seed） |

```yaml
# Retry flaky tests
retry:
  max_attempts: 2
  only: ["failed"]
```

### 6.4 CI Pipeline 概览

```
┌─────────────────────────────────────────────┐
│                 CI Pipeline                  │
├─────────────────────────────────────────────┤
│                                              │
│  ┌─── Stage 1: Quick Checks ─────────────┐  │
│  │  cargo fmt --check                    │  │
│  │  cargo clippy --features full         │  │
│  │  cargo test (default features)        │  │
│  └───────────────────────────────────────┘  │
│                    │                         │
│  ┌─── Stage 2: Feature Matrix ───────────┐  │
│  │  Parallel jobs for each feature combo │  │
│  │  ├── std + parallel                   │  │
│  │  ├── std + simd                       │  │
│  │  ├── std + parallel + simd            │  │
│  │  └── no_std build                     │  │
│  └───────────────────────────────────────┘  │
│                    │                         │
│  ┌─── Stage 3: Cross-platform ───────────┐  │
│  │  macOS ARM64: F1, F4                  │  │
│  │  Windows x86_64: F1, F4              │  │
│  └───────────────────────────────────────┘  │
│                    │                         │
│  ┌─── Stage 4: Extended Tests ───────────┐  │
│  │  cargo test --test large_tensors      │  │
│  │    (XENON_LARGE_TESTS=1, full)        │  │
│  │  cargo test --test property/          │  │
│  └───────────────────────────────────────┘  │
│                    │                         │
│  ┌─── Stage 5: Coverage ─────────────────┐  │
│  │  cargo llvm-cov --features full       │  │
│  │  enforce ≥ 80% line coverage          │  │
│  └───────────────────────────────────────┘  │
│                                              │
└─────────────────────────────────────────────┘
```

---

## 7. 实现任务拆分

> 每个任务约 10 分钟，遵循 `00-rust-standards.md` 中的任务模板。

### 7.1 基础设施（前置条件）

- [ ] **T1**: 创建 `tests/common/mod.rs`，定义公共模块结构
  - 文件: `tests/common/mod.rs`
  - 预计: 5 min

- [ ] **T2**: 实现 `tests/common/assertions.rs` — `assert_close!` 和 `assert_tensor_eq!` 宏
  - 文件: `tests/common/assertions.rs`
  - 验证: 宏能正确比较 f64 浮点数和张量
  - 预计: 10 min

- [ ] **T3**: 实现 `tests/common/fixtures.rs` — 测试数据工厂函数
  - 文件: `tests/common/fixtures.rs`
  - 包含: `random_tensor()`, `linspace_tensor()`, `identity_tensor()` 等工厂
  - 预计: 10 min

- [ ] **T4**: 实现 `tests/common/numpy_ref.rs` — NumPy 参考值加载工具
  - 文件: `tests/common/numpy_ref.rs`
  - 包含: `NumpyRef` 结构体、JSON 加载逻辑
  - 预计: 10 min

- [ ] **T5**: 创建 `tests/data/numpy_references/` 目录和 `generate_refs.py` 脚本
  - 文件: `tests/data/numpy_references/generate_refs.py`
  - 包含: 算术、归约、三角函数、矩阵运算的参考值生成
  - 预计: 15 min

- [ ] **T6**: 在 `Cargo.toml` 中添加 `[dev-dependencies]` — proptest
  - 文件: `Cargo.toml`
  - 预计: 3 min

### 7.2 核心 Test Suite

- [ ] **T7**: 实现 `tests/tensor_lifecycle.rs` — Owned/View/ViewMut 基本生命周期
  - 文件: `tests/tensor_lifecycle.rs`
  - 测试: `test_owned_create_read_modify_drop_f64_ix2`, `test_view_shares_underlying_data`, `test_view_mut_modifications_visible_in_owned`
  - 预计: 10 min

- [ ] **T8**: 实现 `tests/tensor_lifecycle.rs` — ArcRepr 和 CoW 语义
  - 文件: `tests/tensor_lifecycle.rs` (追加)
  - 测试: `test_arc_tensor_shallow_clone_and_cow`, `test_arc_make_mut_triggers_deep_copy_when_ref_count_gt_1`, `test_arc_make_mut_in_place_when_ref_count_eq_1`
  - 前置: T7
  - 预计: 10 min

- [ ] **T9**: 实现 `tests/tensor_lifecycle.rs` — 视图链和维度转换
  - 文件: `tests/tensor_lifecycle.rs` (追加)
  - 测试: `test_chained_views_shared_data_integrity`, `test_reshape_view_data_correctness`, `test_ixdyn_roundtrip_with_ix2`
  - 前置: T7
  - 预计: 10 min

- [ ] **T10**: 实现 `tests/ffi_safety.rs` — 指针 API 和对齐验证
  - 文件: `tests/ffi_safety.rs`
  - 测试: `test_as_ptr_alignment_64bytes`, `test_raw_parts_roundtrip_preserves_data`, `test_index_to_offset_various_strides`
  - 预计: 10 min

- [ ] **T11**: 实现 `tests/ffi_safety.rs` — BLAS 兼容性和 checked/unchecked 等价
  - 文件: `tests/ffi_safety.rs` (追加)
  - 测试: `test_blas_layout_f_contiguous_matrix`, `test_blas_layout_transposed_not_contiguous`, `test_checked_vs_unchecked_index_equivalence`
  - 前置: T10
  - 预计: 10 min

- [ ] **T12**: 实现 `tests/complex_ops.rs` — Complex 构造、算术、内存布局
  - 文件: `tests/complex_ops.rs`
  - 测试: `test_complex_memory_layout_repr_c`, `test_complex_tensor_element_wise_mul`, `test_complex_real_interop_add`, `test_complex_real_interop_div`
  - 预计: 10 min

- [ ] **T13**: 实现 `tests/complex_ops.rs` — Complex 特殊方法和超越函数
  - 文件: `tests/complex_ops.rs` (追加)
  - 测试: `test_complex_norm_uses_hypot`, `test_complex_exp_euler_identity`, `test_complex_conj`, `test_complex_from_polar_roundtrip`
  - 前置: T12
  - 预计: 10 min

### 7.3 领域测试

- [ ] **T14**: 实现 `tests/construction.rs` — zeros/ones/full/eye/diag 构造
  - 文件: `tests/construction.rs`
  - 测试: 多种维度 × 多种元素类型组合
  - 预计: 10 min

- [ ] **T15**: 实现 `tests/construction.rs` — arange/linspace/logspace 序列生成
  - 文件: `tests/construction.rs` (追加)
  - 前置: T14
  - 预计: 10 min

- [ ] **T16**: 实现 `tests/arithmetic.rs` — 逐元素运算和运算符重载
  - 文件: `tests/arithmetic.rs`
  - 测试: add/sub/mul/div × (f32, f64, i32) + 运算符变体
  - 预计: 10 min

- [ ] **T17**: 实现 `tests/broadcasting.rs` — 广播规则和错误处理
  - 文件: `tests/broadcasting.rs`
  - 测试: 广播形状计算、不兼容错误、只读视图
  - 预计: 10 min

- [ ] **T18**: 实现 `tests/indexing.rs` — 多维索引和切片
  - 文件: `tests/indexing.rs`
  - 测试: 多维索引、范围切片、take/put/mask/where
  - 预计: 10 min

- [ ] **T19**: 实现 `tests/shape_ops.rs` — reshape/transpose/squeeze
  - 文件: `tests/shape_ops.rs`
  - 测试: reshape 数据保持、transpose 步长交换、squeeze/unsqueeze
  - 预计: 10 min

- [ ] **T20**: 实现 `tests/shape_ops.rs` — split/chunk/pad/repeat
  - 文件: `tests/shape_ops.rs` (追加)
  - 前置: T19
  - 预计: 10 min

- [ ] **T21**: 实现 `tests/reduction.rs` — 全局和沿轴归约
  - 文件: `tests/reduction.rs`
  - 测试: sum/prod/mean/var/std/min/max 全局 + 沿轴 + NaN 传播
  - 预计: 10 min

- [ ] **T22**: 实现 `tests/reduction.rs` — 累积归约和 argmin/argmax
  - 文件: `tests/reduction.rs` (追加)
  - 前置: T21
  - 预计: 10 min

- [ ] **T23**: 实现 `tests/matrix_ops.rs` — matvec/dot/outer
  - 文件: `tests/matrix_ops.rs`
  - 测试: F/C-order matvec、dot/inner/outer、BLAS 兼容布局
  - 预计: 10 min

- [ ] **T24**: 实现 `tests/matrix_ops.rs` — batch 运算
  - 文件: `tests/matrix_ops.rs` (追加)
  - 前置: T23
  - 预计: 10 min

- [ ] **T25**: 实现 `tests/set_ops.rs` — unique/bincount/histogram
  - 文件: `tests/set_ops.rs`
  - 测试: unique 排序、bincount 计数、histogram bin 边界
  - 预计: 10 min

- [ ] **T26**: 实现 `tests/iter_integration.rs` — 元素/轴/窗口迭代
  - 文件: `tests/iter_integration.rs`
  - 测试: F/C 顺序、轴迭代、窗口边界、zip 广播
  - 预计: 10 min

- [ ] **T27**: 实现 `tests/workspace.rs` — 工作空间分配和分割
  - 文件: `tests/workspace.rs`
  - 测试: 对齐分配、split_at、递归分割、scratch 查询
  - 预计: 10 min

- [ ] **T28**: 实现 `tests/edge_cases.rs` — 空张量和单元素
  - 文件: `tests/edge_cases.rs`
  - 测试: 空张量归约/迭代、Ix0 标量运算
  - 预计: 10 min

- [ ] **T29**: 实现 `tests/edge_cases.rs` — NaN/Inf 和非连续布局
  - 文件: `tests/edge_cases.rs` (追加)
  - 前置: T28
  - 预计: 10 min

### 7.4 Feature 和兼容性

- [ ] **T30**: 实现 `tests/feature_combinations.rs` — 条件编译测试框架
  - 文件: `tests/feature_combinations.rs`
  - 包含: `#[cfg(feature = "...")]` 守卫的测试骨架
  - 预计: 10 min

- [ ] **T31**: 实现 `tests/feature_combinations.rs` — parallel/simd 结果一致性
  - 文件: `tests/feature_combinations.rs` (追加)
  - 前置: T30
  - 预计: 10 min

- [ ] **T32**: 实现 `tests/no_std_compat.rs` — no_std 编译验证
  - 文件: `tests/no_std_compat.rs`
  - 内容: `#![no_std]` + API 可用性验证
  - 预计: 5 min

### 7.5 大张量和 NumPy 兼容

- [ ] **T33**: 实现 `tests/large_tensors.rs` — 大数组分配和并行归约
  - 文件: `tests/large_tensors.rs`
  - 测试: `test_large_allocation_f64`, `test_large_tensor_parallel_sum_correctness`
  - 预计: 10 min

- [ ] **T34**: 实现 `tests/numpy_compat.rs` — 算术和归约参考值对比
  - 文件: `tests/numpy_compat.rs`
  - 前置: T5 (参考数据)
  - 预计: 10 min

- [ ] **T35**: 实现 `tests/numpy_compat.rs` — 三角函数和矩阵运算参考值对比
  - 文件: `tests/numpy_compat.rs` (追加)
  - 前置: T5, T34
  - 预计: 10 min

### 7.6 属性测试

- [ ] **T36**: 实现 `tests/property/mod.rs` — Strategy 定义和共享配置
  - 文件: `tests/property/mod.rs`
  - 包含: `tensor_f64_strategy()`, `shape_strategy()`, `ProptestConfig`
  - 预计: 10 min

- [ ] **T37**: 实现 `tests/property/algebraic_props.rs` — 代数性质
  - 文件: `tests/property/algebraic_props.rs`
  - 前置: T36
  - 测试: 交换律、结合律、分配律、零元、单位元
  - 预计: 10 min

- [ ] **T38**: 实现 `tests/property/broadcast_rules.rs` — 广播不变量
  - 文件: `tests/property/broadcast_rules.rs`
  - 前置: T36
  - 测试: 广播形状正确性、广播结果一致性
  - 预计: 10 min

- [ ] **T39**: 实现 `tests/property/reduction_props.rs` — 归约不变量
  - 文件: `tests/property/reduction_props.rs`
  - 前置: T36
  - 测试: sum(全零)=0、min≤mean≤max、cumsum[-1]==sum
  - 预计: 10 min

- [ ] **T40**: 实现 `tests/property/shape_invariants.rs` — 形状操作不变量
  - 文件: `tests/property/shape_invariants.rs`
  - 前置: T36
  - 测试: reshape 往返、transpose 双次、split+cat
  - 预计: 10 min

- [ ] **T41**: 实现 `tests/property/complex_props.rs` — 复数运算不变量
  - 文件: `tests/property/complex_props.rs`
  - 前置: T36
  - 测试: z*conj(z)=norm²、exp(ln(z))=z、from_polar 往返
  - 预计: 10 min

---

## 附录 A：自定义断言宏设计

```rust
/// Assert that two floating-point values are close within tolerance.
#[macro_export]
macro_rules! assert_close {
    ($left:expr, $right:expr, $tol:expr) => {{
        let left = $left;
        let right = $right;
        let tol = $tol;
        let diff = (left - right).abs();
        assert!(
            diff <= tol,
            "assertion failed: `(left ≈ right)`\n  left: `{}`\n right: `{}`\n   tol: `{}`\n  diff: `{}`",
            left, right, tol, diff
        );
    }};
}

/// Assert that two tensors have the same shape and close values.
#[macro_export]
macro_rules! assert_tensor_close {
    ($left:expr, $right:expr, $tol:expr) => {{
        let left = $left;
        let right = $right;
        assert_eq!(left.shape(), right.shape(), "tensor shapes differ");
        for (i, (l, r)) in left.iter().zip(right.iter()).enumerate() {
            let diff = (l - r).abs();
            assert!(
                diff <= $tol,
                "assertion failed at index {}: left={}, right={}, diff={}, tol={}",
                i, l, r, diff, $tol
            );
        }
    }};
}
```

## 附录 B：NumPy 参考值生成脚本骨架

```python
#!/usr/bin/env python3
"""Generate NumPy reference values for Xenon integration tests."""

import numpy as np
import json
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__))

def save_ref(filename: str, data: dict):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Generated: {path}")

def generate_arithmetic_refs():
    np.random.seed(42)
    shape = (3, 4)
    a = np.random.randn(*shape)
    b = np.random.randn(*shape) + 0.5
    save_ref("arithmetic_f64.json", {
        "description": "element-wise arithmetic, f64, shape [3,4]",
        "dtype": "float64",
        "shape": list(shape),
        "a": a.flatten('F').tolist(),
        "b": b.flatten('F').tolist(),
        "expected_add": (a + b).flatten('F').tolist(),
        "expected_sub": (a - b).flatten('F').tolist(),
        "expected_mul": (a * b).flatten('F').tolist(),
        "expected_div": (a / b).flatten('F').tolist(),
        "tolerance": 1e-15,
    })

def generate_reduction_refs():
    np.random.seed(123)
    shape = (5, 6)
    a = np.random.randn(*shape)
    save_ref("reduction_f64.json", {
        "description": "reduction operations, f64, shape [5,6]",
        "dtype": "float64",
        "shape": list(shape),
        "a": a.flatten('F').tolist(),
        "expected_sum": float(np.sum(a)),
        "expected_mean": float(np.mean(a)),
        "expected_var": float(np.var(a)),     # ddof=0
        "expected_std": float(np.std(a)),     # ddof=0
        "expected_min": float(np.min(a)),
        "expected_max": float(np.max(a)),
        "expected_sum_axis0": np.sum(a, axis=0).flatten('F').tolist(),
        "expected_sum_axis1": np.sum(a, axis=1).flatten('F').tolist(),
        "tolerance": 1e-13,
    })

def generate_trig_refs():
    np.random.seed(456)
    x = np.linspace(-np.pi, np.pi, 24).reshape(4, 6)
    save_ref("trigonometry_f64.json", {
        "description": "trigonometric functions, f64, shape [4,6]",
        "dtype": "float64",
        "shape": list(x.shape),
        "x": x.flatten('F').tolist(),
        "expected_sin": np.sin(x).flatten('F').tolist(),
        "expected_cos": np.cos(x).flatten('F').tolist(),
        "expected_tan": np.tan(x).flatten('F').tolist(),
        "expected_exp": np.exp(x).flatten('F').tolist(),
        "expected_ln": np.log(x + 4.0).flatten('F').tolist(),  # shift to positive
        "tolerance": 1e-14,
    })

def generate_linalg_refs():
    np.random.seed(789)
    m, n = 4, 5
    mat = np.random.randn(m, n)
    vec = np.random.randn(n)
    save_ref("linalg_f64.json", {
        "description": "linear algebra operations, f64",
        "dtype": "float64",
        "mat_shape": [m, n],
        "mat": mat.flatten('F').tolist(),
        "vec": vec.tolist(),
        "expected_matvec": mat.dot(vec).tolist(),
        "vec_a": np.random.randn(6).tolist(),
        "vec_b": np.random.randn(6).tolist(),
        "tolerance": 1e-13,
    })

if __name__ == "__main__":
    generate_arithmetic_refs()
    generate_reduction_refs()
    generate_trig_refs()
    generate_linalg_refs()
```

---

## 附录 C：CI 配置骨架 (GitHub Actions)

```yaml
name: Integration Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  CARGO_TERM_COLOR: always
  PROPTEST_SEED: "0x000102030405060708090A0B0C0D0E0F"

jobs:
  fmt-and-clippy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt, clippy
      - run: cargo fmt --check
      - run: cargo clippy --features full -- -D warnings

  test-default:
    runs-on: ubuntu-latest
    needs: fmt-and-clippy
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo test

  test-features:
    runs-on: ubuntu-latest
    needs: fmt-and-clippy
    strategy:
      fail-fast: false
      matrix:
        features:
          - "--features parallel"
          - "--features simd"
          - "--features full"
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo test ${{ matrix.features }}

  test-nostd:
    runs-on: ubuntu-latest
    needs: fmt-and-clippy
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo build --no-default-features

  test-cross-platform:
    runs-on: ${{ matrix.os }}
    needs: fmt-and-clippy
    strategy:
      fail-fast: false
      matrix:
        os: [macos-latest, windows-latest]
        features: ["", "--features full"]
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo test ${{ matrix.features }}

  test-large:
    runs-on: ubuntu-latest
    needs: test-default
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: XENON_LARGE_TESTS=1 cargo test --test large_tensors --features full
        timeout-minutes: 15

  coverage:
    runs-on: ubuntu-latest
    needs: test-features
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo install cargo-llvm-cov
      - run: cargo llvm-cov --features full --fail-under-lines 80
```

---

*Xenon 集成测试方案设计 — 文档 23*
