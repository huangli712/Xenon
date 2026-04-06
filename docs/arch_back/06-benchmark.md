# Senon Benchmark 设计文档

> **文档版本**: v1.0  
> **最后更新**: 2026-03-28  
> **模块路径**: `benches/`  
> **需求来源**: require-v18.md §7, §17

---

## 1. 模块概述

### 1.1 职责定义

Benchmark 模块负责 Senon 库的性能基准测试，用于：

| 职责 | 说明 |
|------|------|
| 性能回归检测 | 每次提交自动运行，检测性能退化 |
| 优化验证 | SIMD/并行优化效果的量化验证 |
| 竞品对比 | 与 ndarray 等同类库的性能对比（可选） |
| 性能档案 | 记录各操作在不同数据规模下的吞吐量 |

### 1.2 设计目标

| 目标 | 实现方式 |
|------|----------|
| **可重复性** | 固定数据规模、固定随机种子、预热后测量 |
| **全面覆盖** | 覆盖所有性能关键路径（逐元素、归约、矩阵、形状操作） |
| **多维度对比** | 同一操作在不同数据规模、布局、类型下的性能 |
| **低噪声** | 使用 criterion 统计分析，过滤测量噪声 |

### 1.3 需求来源

来自 `require-v18.md` 第 17 节质量要求，以及第 7 节计算后端的性能分层设计。

---

## 2. 基础设施

### 2.1 Cargo.toml 配置

```toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name = "elementwise"
harness = false

[[bench]]
name = "reduction"
harness = false

[[bench]]
name = "matrix_ops"
harness = false

[[bench]]
name = "shape_ops"
harness = false

[[bench]]
name = "iterator"
harness = false

[[bench]]
name = "construction"
harness = false

[[bench]]
name = "indexing"
harness = false

[[bench]]
name = "broadcast"
harness = false
```

### 2.2 目录结构

```
benches/
├── elementwise.rs      — 逐元素运算 benchmark
├── reduction.rs        — 归约运算 benchmark
├── matrix_ops.rs       — 矩阵运算 benchmark
├── shape_ops.rs        — 形状操作 benchmark
├── iterator.rs         — 迭代器 benchmark
├── construction.rs     — 构造方法 benchmark
├── indexing.rs         — 索引操作 benchmark
├── broadcast.rs        — 广播运算 benchmark
└── utils/
    ├── mod.rs          — 共享工具函数
    └── data_gen.rs     — 测试数据生成器
```

### 2.3 共享工具模块

```rust
// benches/utils/mod.rs
pub mod data_gen;

/// Standard benchmark sizes covering small to large arrays.
pub const SIZES_1D: &[usize] = &[64, 256, 1024, 4096, 16384, 65536, 262144, 1048576];

/// Standard 2D matrix sizes (square).
pub const SIZES_2D: &[(usize, usize)] = &[
    (8, 8), (32, 32), (64, 64), (128, 128),
    (256, 256), (512, 512), (1024, 1024),
];

/// Standard 3D tensor sizes.
pub const SIZES_3D: &[(usize, usize, usize)] = &[
    (8, 8, 8), (16, 16, 16), (32, 32, 32), (64, 64, 64),
];
```

```rust
// benches/utils/data_gen.rs
use Senon::prelude::*;

/// Generate a 1D tensor with sequential values for reproducible benchmarks.
pub fn sequential_1d(n: usize) -> Tensor1<f64> {
    Tensor1::from_fn([n], |[i]| i as f64)
}

/// Generate a 2D F-order tensor with sequential values.
pub fn sequential_2d(rows: usize, cols: usize) -> Tensor2<f64> {
    Tensor2::from_fn([rows, cols], |[i, j]| (i + j * rows) as f64)
}

/// Generate a non-contiguous view (every other element).
pub fn strided_1d(n: usize) -> Tensor1<f64> {
    let full = sequential_1d(n * 2);
    full.slice(s![..;2]).to_owned()
}
```

### 2.4 Benchmark 模板

```rust
// benches/example.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

mod utils;
use utils::{SIZES_1D, data_gen};

fn bench_operation(c: &mut Criterion) {
    let mut group = c.benchmark_group("operation_name");

    for &size in SIZES_1D {
        let a = data_gen::sequential_1d(size);

        group.bench_with_input(
            BenchmarkId::new("f64/contiguous", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(a.sum());
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_operation);
criterion_main!(benches);
```

---

## 3. Benchmark 分类

### 3.1 逐元素运算 (elementwise.rs)

| Benchmark | 操作 | 变量维度 |
|-----------|------|----------|
| add_contiguous | a + b（两个连续数组） | 数据规模、元素类型(f32/f64) |
| add_broadcast | a + scalar | 数据规模 |
| add_noncontiguous | 非连续数组相加 | 数据规模 |
| mul_contiguous | a * b | 数据规模、元素类型 |
| sin_contiguous | sin(a) | 数据规模、元素类型 |
| exp_contiguous | exp(a) | 数据规模 |
| abs_contiguous | abs(a) | 数据规模 |
| complex_add | Complex 加法 | 数据规模 |
| complex_mul | Complex 乘法 | 数据规模 |
| compound_assign | a += b（原地运算） | 数据规模 |

**性能分层验证**:

| 条件 | 预期路径 | 验证方式 |
|------|----------|----------|
| 小数组 (< SIMD 宽度) | 标量 | 对比 SIMD 关闭/开启无差异 |
| 中数组 (SIMD 宽度 ~ 64K) | SIMD | 对比标量有 2-8x 加速 |
| 大数组 (>= 64K) | SIMD + 并行 | 对比单线程有近线性加速 |
| 非连续 | 标量 | 对比连续数组有性能下降 |

### 3.2 归约运算 (reduction.rs)

| Benchmark | 操作 | 变量维度 |
|-----------|------|----------|
| sum_1d | 全局 sum | 数据规模 |
| sum_2d_axis0 | 沿轴 0 归约 | 矩阵大小 |
| sum_2d_axis1 | 沿轴 1 归约 | 矩阵大小 |
| mean_1d | 全局 mean | 数据规模 |
| var_1d | 全局 var | 数据规模 |
| min_max_1d | 全局 min/max | 数据规模 |
| argmin_1d | 全局 argmin | 数据规模 |
| cumsum_1d | 累积求和 | 数据规模 |
| sum_noncontiguous | 非连续数组 sum | 数据规模 |
| unique_1d | unique 操作 | 数据规模、唯一值比例 |
| histogram_1d | histogram | 数据规模、bin 数量 |

### 3.3 矩阵运算 (matrix_ops.rs)

| Benchmark | 操作 | 变量维度 |
|-----------|------|----------|
| matvec_f64 | 矩阵-向量乘法 | 矩阵大小 |
| matvec_f32 | 矩阵-向量乘法 (f32) | 矩阵大小 |
| dot_1d | 向量内积 | 向量长度 |
| outer_product | 向量外积 | 向量长度 |
| batch_matvec | 批量矩阵-向量乘法 | batch 大小、矩阵大小 |
| batch_dot | 批量内积 | batch 大小、向量长度 |
| matvec_noncontiguous | 非连续矩阵 matvec | 矩阵大小 |
| matvec_c_order | C-order 矩阵 matvec | 矩阵大小 |

### 3.4 形状操作 (shape_ops.rs)

| Benchmark | 操作 | 变量维度 |
|-----------|------|----------|
| reshape_contiguous | reshape（零拷贝） | 数据规模 |
| transpose_2d | 2D 转置 | 矩阵大小 |
| slice_contiguous | 连续切片 | 数据规模、切片比例 |
| cat_axis0 | 沿轴 0 拼接 | 数组数量、每个数组大小 |
| stack_new_axis | 新轴堆叠 | 数组数量、每个数组大小 |
| pad_constant | 常量填充 | 数据规模、填充宽度 |
| repeat_tile | repeat/tile | 数据规模、重复次数 |
| split_chunk | split/chunk | 数据规模、分块数 |
| flatten | flatten | 数据规模、连续 vs 非连续 |

### 3.5 迭代器 (iterator.rs)

| Benchmark | 操作 | 变量维度 |
|-----------|------|----------|
| iter_elements | 元素迭代 | 数据规模 |
| iter_axis | 轴迭代 | 数据规模、轴 |
| iter_window | 窗口迭代 | 数据规模、窗口大小 |
| iter_indexed | 带索引迭代 | 数据规模 |
| zip_2 | 两数组 zip | 数据规模 |
| zip_3 | 三数组 zip | 数据规模 |
| map_inplace | mapv_inplace | 数据规模 |
| par_iter | 并行迭代 | 数据规模 |

### 3.6 构造方法 (construction.rs)

| Benchmark | 操作 | 变量维度 |
|-----------|------|----------|
| zeros_1d | zeros 构造 | 数据规模 |
| zeros_2d | 2D zeros | 矩阵大小 |
| from_vec | from_vec 构造 | 数据规模 |
| from_fn | from_fn 构造 | 数据规模 |
| arange | arange 序列 | 数据规模 |
| linspace | linspace 序列 | 数据规模 |
| clone_tensor | 深拷贝 | 数据规模 |
| to_f_contiguous | 布局转换 | 数据规模 |
| cast_f32_f64 | 类型转换 | 数据规模 |

### 3.7 索引操作 (indexing.rs)

| Benchmark | 操作 | 变量维度 |
|-----------|------|----------|
| index_single | 单元素索引 | 维度数 |
| index_axis | index_axis | 数据规模 |
| take_1d | take 操作 | 数据规模、索引数量 |
| mask_bool | bool mask | 数据规模、true 比例 |
| where_select | where 条件选择 | 数据规模 |
| advanced_index | 高级索引 | 数据规模 |

### 3.8 广播 (broadcast.rs)

| Benchmark | 操作 | 变量维度 |
|-----------|------|----------|
| broadcast_scalar | 标量广播 | 数据规模 |
| broadcast_row | 行向量广播到矩阵 | 矩阵大小 |
| broadcast_col | 列向量广播到矩阵 | 矩阵大小 |
| broadcast_add | 广播加法 | 数据规模 |
| broadcast_mul | 广播乘法 | 数据规模 |


---

## 4. 性能分层验证策略

### 4.1 SIMD 加速验证

通过对比 `simd` feature 开启/关闭时的性能差异，验证 SIMD 路径的有效性。

```rust
// Benchmark pattern for SIMD verification
fn bench_simd_verification(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_verification");

    for &size in &[1024, 16384, 262144] {
        let a = data_gen::sequential_1d(size);
        let b = data_gen::sequential_1d(size);

        // This benchmark runs with whatever features are enabled.
        // Compare results between:
        //   cargo bench --bench elementwise
        //   cargo bench --bench elementwise --no-default-features
        //   cargo bench --bench elementwise --features simd
        group.bench_with_input(
            BenchmarkId::new("add", size),
            &size,
            |bench, _| {
                bench.iter(|| black_box(&a + &b));
            },
        );
    }

    group.finish();
}
```

**预期加速比**:

| 操作 | 标量 → SSE4.1 | 标量 → AVX2 | 标量 → AVX-512 |
|------|--------------|-------------|----------------|
| f64 add | ~2x | ~4x | ~8x |
| f32 add | ~4x | ~8x | ~16x |
| f64 sum | ~2x | ~4x | ~8x |
| f64 sin/exp | ~2x | ~2-4x | ~4-8x |

### 4.2 并行加速验证

```rust
// Benchmark pattern for parallel verification
fn bench_parallel_verification(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_verification");

    // Only meaningful for large arrays
    for &size in &[65536, 262144, 1048576, 4194304] {
        let a = data_gen::sequential_1d(size);

        group.bench_with_input(
            BenchmarkId::new("sum", size),
            &size,
            |bench, _| {
                bench.iter(|| black_box(a.sum()));
            },
        );
    }

    group.finish();
}
```

**预期加速比** (4 核):

| 数据规模 | 预期加速 | 说明 |
|----------|----------|------|
| 64K | ~1x | 接近并行阈值，开销抵消收益 |
| 256K | ~2-3x | 并行收益显现 |
| 1M | ~3-4x | 接近线性加速 |
| 4M | ~3.5-4x | 接近理论上限 |

### 4.3 内存布局影响验证

```rust
fn bench_layout_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("layout_impact");

    let size = 1024;
    let a_f = Tensor2::<f64>::zeros([size, size]); // F-order (default)
    let a_c = Tensor2::<f64>::zeros_with_order([size, size], Order::C);

    // Sum along axis 0: F-order should be faster (contiguous access)
    group.bench_function("sum_axis0_f_order", |b| {
        b.iter(|| black_box(a_f.sum_axis(0)));
    });

    group.bench_function("sum_axis0_c_order", |b| {
        b.iter(|| black_box(a_c.sum_axis(0)));
    });

    // Sum along axis 1: C-order should be faster
    group.bench_function("sum_axis1_f_order", |b| {
        b.iter(|| black_box(a_f.sum_axis(1)));
    });

    group.bench_function("sum_axis1_c_order", |b| {
        b.iter(|| black_box(a_c.sum_axis(1)));
    });

    group.finish();
}
```

---

## 5. 数值精度验证

### 5.1 精度 Benchmark

不测量速度，而是测量数值精度，确保满足 `require-v18.md` 第 17.3 节要求。

```rust
// benches/precision.rs (not a criterion benchmark, but a test)
// This is better placed in tests/, but documented here for completeness.

/// Verify reduction precision meets requirements.
fn verify_sum_precision() {
    // Kahan summation reference
    let n = 1_000_000;
    let a = Tensor1::from_fn([n], |[i]| 1.0f64 / (i as f64 + 1.0));

    let result = a.sum();
    let reference = kahan_sum(&a); // high-precision reference

    let rel_error = (result - reference).abs() / reference.abs();
    assert!(rel_error < 1e-15, "f64 sum precision: {}", rel_error);
}
```

### 5.2 精度要求速查

| 运算类别 | f64 精度 | f32 精度 |
|----------|----------|----------|
| 加减乘 | 精确 | 精确 |
| 归约 (sum/prod) | 1e-15 | 1e-6 |
| 超越函数 (sin/exp/ln) | 1e-14 | 1e-5 |

---

## 6. CI 集成

### 6.1 CI Benchmark 策略

| 策略 | 说明 |
|------|------|
| 触发条件 | PR 合并到 main 分支时运行 |
| 基准线 | 与上一次 main 分支的结果对比 |
| 回归阈值 | 性能下降超过 10% 时标记为 warning，超过 20% 时 fail |
| 报告格式 | criterion HTML 报告，存储为 CI artifact |
| 运行环境 | 固定硬件配置，避免虚拟化噪声 |

### 6.2 CI 配置示例

```yaml
# .github/workflows/bench.yml (conceptual)
benchmark:
  runs-on: self-hosted  # Fixed hardware for reproducibility
  steps:
    - uses: actions/checkout@v4

    - name: Run benchmarks
      run: cargo bench --all-features -- --output-format bencher

    - name: Store results
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: cargo
        output-file-path: target/criterion/output.txt
        alert-threshold: "120%"
        fail-on-alert: true
```

### 6.3 本地运行指南

```bash
# Run all benchmarks
cargo bench --all-features

# Run specific benchmark group
cargo bench --bench elementwise

# Run with specific filter
cargo bench --bench elementwise -- "add_contiguous"

# Compare with baseline
cargo bench --all-features -- --save-baseline main
# ... make changes ...
cargo bench --all-features -- --baseline main

# Generate HTML report
# Results in target/criterion/report/index.html
```

---

## 7. 与其他模块的交互

### 7.1 依赖关系

```
06-benchmark
├── 依赖 04.01 SIMD 后端
│   └── SIMD 加速验证 benchmark
├── 依赖 04.02 并行后端
│   └── 并行加速验证 benchmark
├── 依赖 05.01 迭代器
│   └── 迭代器性能 benchmark
├── 依赖 05.02 逐元素运算
│   └── 逐元素运算 benchmark
├── 依赖 05.03 矩阵运算
│   └── 矩阵运算 benchmark
├── 依赖 05.04 归约运算
│   └── 归约运算 benchmark
├── 依赖 05.05 广播
│   └── 广播运算 benchmark
├── 依赖 05.06 形状操作
│   └── 形状操作 benchmark
├── 依赖 05.07 索引操作
│   └── 索引操作 benchmark
└── 依赖 05.08 构造与转换
    └── 构造方法 benchmark
```

---

## 8. 实现任务分解

| 任务 | 描述 | 预计时间 | 依赖 |
|------|------|----------|------|
| T1 | 配置 Cargo.toml bench 入口和 criterion 依赖 | 5 min | 无 |
| T2 | 实现 benches/utils/ 共享工具模块（数据生成、常量） | 10 min | T1 |
| T3 | 实现 benches/elementwise.rs（逐元素运算 benchmark） | 10 min | T2, 05.02 完成 |
| T4 | 实现 benches/reduction.rs（归约运算 benchmark） | 10 min | T2, 05.04 完成 |
| T5 | 实现 benches/matrix_ops.rs（矩阵运算 benchmark） | 10 min | T2, 05.03 完成 |
| T6 | 实现 benches/shape_ops.rs（形状操作 benchmark） | 10 min | T2, 05.06 完成 |
| T7 | 实现 benches/iterator.rs（迭代器 benchmark） | 10 min | T2, 05.01 完成 |
| T8 | 实现 benches/construction.rs（构造方法 benchmark） | 10 min | T2, 05.08 完成 |
| T9 | 实现 benches/indexing.rs（索引操作 benchmark） | 10 min | T2, 05.07 完成 |
| T10 | 实现 benches/broadcast.rs（广播运算 benchmark） | 10 min | T2, 05.05 完成 |
| T11 | 实现 SIMD 加速验证 benchmark（对比 feature 开关） | 10 min | T3, 04.01 完成 |
| T12 | 实现并行加速验证 benchmark（对比 feature 开关） | 10 min | T3, 04.02 完成 |
| T13 | 实现内存布局影响验证 benchmark | 10 min | T4 |
| T14 | 配置 CI benchmark 工作流 | 10 min | T3-T10 |

### 8.1 并行执行分组

```
Wave 1 (无依赖):
  T1

Wave 2 (依赖 T1):
  T2

Wave 3 (依赖 T2 + 各模块实现，可并行):
  T3, T4, T5, T6, T7, T8, T9, T10

Wave 4 (依赖 Wave 3):
  T11, T12, T13

Wave 5 (依赖 Wave 4):
  T14
```

---

## 9. 设计决策记录

### 9.1 决策：使用 criterion.rs

| 属性 | 值 |
|------|-----|
| 决策 | 使用 criterion.rs 作为 benchmark 框架 |
| 理由 | 统计分析（置信区间、异常值检测）；HTML 报告；与 CI 集成成熟 |
| 替代方案 | `#[bench]` nightly — 放弃，需要 nightly 编译器 |
| 替代方案 | divan — 放弃，生态不如 criterion 成熟 |

### 9.2 决策：按操作类别分文件

| 属性 | 值 |
|------|-----|
| 决策 | 每个操作类别一个 benchmark 文件 |
| 理由 | 可独立运行（`cargo bench --bench elementwise`）；编译时间可控 |
| 替代方案 | 单文件 — 放弃，编译慢、难以选择性运行 |

### 9.3 决策：固定数据规模序列

| 属性 | 值 |
|------|-----|
| 决策 | 使用 2 的幂次序列（64, 256, 1024, ...） |
| 理由 | 覆盖 SIMD 宽度边界、并行阈值边界；对齐友好；便于对比 |
| 替代方案 | 任意规模 — 放弃，不利于跨版本对比 |

### 9.4 决策：性能回归阈值 10%/20%

| 属性 | 值 |
|------|-----|
| 决策 | 10% warning，20% fail |
| 理由 | 10% 以内可能是测量噪声；20% 以上几乎确定是真实回归 |
| 替代方案 | 5% fail — 放弃，CI 环境噪声可能导致误报 |

---

*本文档由 Senon 项目维护。如有问题请提交 Issue 或 PR。*
