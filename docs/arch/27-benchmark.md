# 基准测试模块设计

> 文档编号: 27 | 模块: `benches/` | 阶段: Phase 6
> 前置文档: 所有前置文档（`00-coding-standards.md` ~ `26-error-handling.md`）
> 需求参考: 需求说明书 §28.2

---

## 1. 模块定位

### 1.1 职责边界

| 职责 | 包含 | 不包含 |
|------|------|--------|
| 性能回归检测 | 每次 PR 自动运行，检测性能退化 | 功能正确性验证（由集成测试覆盖） |
| 优化验证 | SIMD/并行优化效果的量化验证 | 编译时间测量 |
| 性能档案 | 记录各操作在不同参数下的吞吐量 | 内存泄漏检测 |
| 参数矩阵 | 多规模、多类型、多布局组合 | 跨语言对比（非核心目标） |

### 1.2 设计原则

| 原则 | 体现 |
|------|------|
| 可重复性 | 固定数据规模、固定随机种子、预热后测量 |
| 全面覆盖 | 覆盖所有性能关键路径（逐元素、归约、内积、形状操作） |
| 低噪声 | 使用 criterion 统计分析，过滤测量噪声 |
| 分级执行 | CI 三级工作流：Smoke / Regression / Full |

---

## 2. 文件位置

```
benches/
├── utils/
│   ├── mod.rs              # 共享常量与工具导出
│   └── data_gen.rs         # 测试数据生成器
├── elementwise.rs          # 逐元素运算 benchmark
├── reduction.rs            # 归约运算 benchmark（sum）
├── dot_product.rs          # 向量内积 benchmark
├── set_ops.rs              # 集合操作 benchmark（unique）
├── broadcast.rs            # 广播运算 benchmark
├── shape_ops.rs            # 形状操作 benchmark（transpose/reshape）
├── simd_comparison.rs      # SIMD 对比 benchmark
├── parallel_comparison.rs  # 并行对比 benchmark
└── construction.rs         # 构造方法 benchmark
```

按操作类别分文件：可独立运行（`cargo bench --bench elementwise`），编译时间可控。

---

## 3. 依赖关系

### 3.1 依赖图

```
benches/
├── crate::tensor           # TensorBase, Tensor, TensorView 等
├── crate::dimension        # Ix0~Ix6, IxDyn, Dimension trait
├── crate::element          # Element, Numeric, RealScalar, ComplexScalar
├── crate::ops              # 逐元素运算、归约、内积
├── crate::shape_ops        # transpose, reshape
├── crate::broadcast        # broadcast_shape
├── crate::set_ops          # unique
├── crate::construct        # zeros, ones, from_vec
└── criterion (dev-dep)     # benchmark 框架
```

### 3.2 依赖精确到类型级

| 来源模块 | 使用的类型/trait |
|----------|-----------------|
| `tensor` | `Tensor<A, D>`, `TensorView`, `TensorViewMut`, `.shape()`, `.sum()` |
| `dimension` | `Ix1`, `Ix2`, `Ix3`, `IxDyn`, `Dimension` |
| `element` | `Element`, `Numeric`, `RealScalar`, `ComplexScalar` |
| `ops` | `add`, `sub`, `mul`, `div`, `sin`, `exp`, `abs`, `sum_axis` |
| `shape_ops` | `reshape`, `transpose` |
| `set_ops` | `unique` |
| `construct` | `zeros`, `ones`, `from_vec`, `from_fn` |
| `broadcast` | `broadcast_shape`, 广播运算符 |

### 3.3 依赖方向声明

> **依赖方向：单向消费。** `benches/` 仅消费 crate 公共 API 和 criterion，不被任何模块依赖。

---

## 4. Benchmark 框架选择

### 4.1 Cargo.toml 配置

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
name = "dot_product"
harness = false

[[bench]]
name = "set_ops"
harness = false

[[bench]]
name = "broadcast"
harness = false

[[bench]]
name = "shape_ops"
harness = false

[[bench]]
name = "simd_comparison"
harness = false

[[bench]]
name = "parallel_comparison"
harness = false

[[bench]]
name = "construction"
harness = false
```

### 4.2 共享工具模块

```rust
// benches/utils/mod.rs
pub mod data_gen;

/// Standard benchmark sizes: Small / Medium / Large (1D).
pub const SIZES_1D: &[usize] = &[64, 65_536, 16_777_216];

/// Standard benchmark sizes: Small / Medium / Large (2D square).
pub const SIZES_2D: &[[(usize, usize); 3]] = &[
    [(8, 8), (256, 256), (4096, 4096)],
];

/// Standard benchmark sizes: Small / Medium / Large (3D).
pub const SIZES_3D: &[[(usize, usize, usize); 3]] = &[
    [(4, 4, 4), (64, 32, 32), (256, 256, 256)],
];
```

```rust
// benches/utils/data_gen.rs
use xenon::prelude::*;

/// Generate a 1D tensor with sequential f64 values.
pub fn sequential_1d(n: usize) -> Tensor1<f64> {
    Tensor1::from_fn([n], |[i]| i as f64)
}

/// Generate a 2D F-order tensor with sequential f64 values.
pub fn sequential_2d(rows: usize, cols: usize) -> Tensor2<f64> {
    Tensor2::from_fn([rows, cols], |[i, j]| (i + j * rows) as f64)
}

/// Generate a non-contiguous view by slicing every other element.
pub fn strided_view_1d(n: usize) -> Tensor1<f64> {
    let full = sequential_1d(n * 2);
    full.slice(s![..;2]).to_owned()
}
```

---

## 5. 四级分类体系

```
Benchmark 分类
├── Micro-benchmarks        # 单操作、单函数级别（如 zeros 构造）
├── Kernel benchmarks       # 核心计算内核（如 add, sum, dot）
├── Workflow benchmarks     # 真实使用模式（如 broadcast + add 链）
└── Comparison benchmarks   # 外部对比（SIMD 开/关，并行 开/关）
```

| 级别 | 示例 | 用途 |
|------|------|------|
| Micro | `zeros_1d`, `from_vec` | 基础开销基线 |
| Kernel | `add_f64`, `sum_f64`, `dot_f64` | 核心路径性能 |
| Workflow | `broadcast_add_row`, `transpose_then_sum` | 真实场景吞吐 |
| Comparison | `add_simd_vs_scalar`, `sum_parallel_vs_serial` | 优化效果验证 |

---

## 6. 参数矩阵

### 6.1 输入规模

| 级别 | 1D | 2D | 3D |
|------|-----|-----|-----|
| **Small** | 64 | 8×8 | 4×4×4 |
| **Medium** | 65,536 | 256×256 | 64×32×32 |
| **Large** | 16,777,216 | 4096×4096 | 256×256×256 |

### 6.2 数据类型

| 类型 | 优先级 | 说明 |
|------|--------|------|
| `f64` | **必测** | 科学计算默认精度 |
| `f32` | **必测** | SIMD 向量宽度更大 |
| `Complex<f64>` | **必测** | 复数运算开销验证 |

### 6.3 内存布局

| 布局 | 构造方式 | 验证目标 |
|------|----------|----------|
| F-contiguous | `zeros(shape)` | 默认路径性能基线 |
| Non-contiguous | `tensor.slice(s![.., 0..n-1])` | 非连续路径标量回退惩罚 |

> **注意**：Xenon 仅支持 F-order 布局，不存在 C-order 路径。非连续布局通过切片/转置视图产生。

---

## 7. Benchmark 清单

| 组 ID | 操作 | 规模 | 类型 | 布局 | 说明 |
|-------|------|------|------|------|------|
| `elem_add_f64` | `a + b` | S/M/L | f64 | F-contiguous | 连续数组逐元素加法 |
| `elem_add_f32` | `a + b` | S/M/L | f32 | F-contiguous | f32 加法，SIMD 向量宽度更大 |
| `elem_add_complex` | `a + b` | S/M/L | Complex\<f64\> | F-contiguous | 复数加法开销 |
| `elem_mul_f64` | `a * b` | S/M/L | f64 | F-contiguous | 逐元素乘法 |
| `elem_sin_f64` | `sin(a)` | S/M/L | f64 | F-contiguous | 超越函数逐元素 |
| `elem_add_sliced` | `a + b`（b 为切片视图） | M | f64 | Non-contiguous | 非连续惩罚 |
| `sum_1d` | 全局 sum | S/M/L | f64 | F-contiguous | 1D 归约 |
| `sum_2d_axis0` | 沿轴 0 sum | S/M/L | f64 | F-contiguous | 2D 沿轴归约 |
| `sum_2d_axis1` | 沿轴 1 sum | S/M/L | f64 | F-contiguous | 2D 沿轴归约 |
| `sum_sliced` | 非连续 sum | M | f64 | Non-contiguous | 非连续归约惩罚 |
| `dot_1d_f64` | 向量内积 | S/M/L | f64 | F-contiguous | 基本内积 |
| `dot_1d_complex` | 复数内积 | S/M/L | Complex\<f64\> | F-contiguous | 复数内积（含共轭） |
| `unique_1d` | unique 操作 | S/M/L | f64 | F-contiguous | 排序去重 |
| `broadcast_scalar` | 标量广播加法 | S/M/L | f64 | F-contiguous | 标量广播开销 |
| `broadcast_row` | 行向量广播到矩阵 | S/M/L | f64 | F-contiguous | 行广播 |
| `broadcast_col` | 列向量广播到矩阵 | S/M/L | f64 | F-contiguous | 列广播 |
| `transpose_2d` | 2D 转置（零拷贝） | S/M/L | f64 | F-contiguous | 转置视图创建 |
| `reshape_contiguous` | 连续 reshape（零拷贝） | S/M/L | f64 | F-contiguous | reshape 元数据操作 |
| `reshape_noncontiguous` | 非连续 reshape（需拷贝） | M | f64 | Non-contiguous | reshape 数据拷贝 |
| `simd_add_compare` | `a + b` (SIMD vs 标量) | M | f32/f64 | F-contiguous | SIMD 加速比 |
| `simd_sum_compare` | sum (SIMD vs 标量) | M | f32/f64 | F-contiguous | SIMD 归约加速 |
| `par_sum_compare` | sum (并行 vs 串行) | L | f64 | F-contiguous | 并行加速比 |
| `par_add_compare` | `a + b` (并行 vs 串行) | L | f64 | F-contiguous | 并行逐元素加速 |
| `zeros_1d` | zeros 构造 | S/M/L | f64 | F-contiguous | 构造开销 |
| `from_fn_2d` | from_fn 构造 | S/M/L | f64 | F-contiguous | 函数构造开销 |

---

## 8. CI 三级工作流

| 工作流 | 基准数量 | 预计时间 | 频率 |
|--------|----------|----------|------|
| **Smoke Test** | 3 个核心文件 × `--quick` | ~5 min | 每次 PR |
| **Regression Check** | 1-2 个热点基准 | ~10 min | 每次 PR |
| **Full Benchmark** | 全部文件 × 4 feature 组合 | ~60 min | 每周/合并到 main |

### 8.1 Smoke Test 覆盖范围

| 文件 | 组 | 说明 |
|------|-----|------|
| `elementwise.rs` | `elem_add_f64` (Medium) | 核心逐元素路径 |
| `reduction.rs` | `sum_1d` (Medium) | 核心归约路径 |
| `construction.rs` | `zeros_1d` (Medium) | 基础构造路径 |

### 8.2 CI 配置示例

```yaml
# .github/workflows/bench.yml
benchmark-smoke:
    runs-on: self-hosted  # Fixed hardware for reproducibility
    steps:
        - uses: actions/checkout@v4

        - name: Smoke benchmarks
          run: |
            cargo bench --bench elementwise -- "elem_add_f64" --quick
            cargo bench --bench reduction -- "sum_1d" --quick
            cargo bench --bench construction -- "zeros_1d" --quick

        - name: Store results
          uses: benchmark-action/github-action-benchmark@v1
          with:
            tool: cargo
            output-file-path: target/criterion/output.txt
            alert-threshold: "120%"
            fail-on-alert: true
```

---

## 9. 回归阈值

| 级别 | 阈值 | 动作 |
|------|------|------|
| **警告** | > 5% 变慢 | CI warning（不阻塞合并） |
| **失败** | > 20% 变慢 | CI failure（阻塞合并） |
| **改善** | > 5% 变快 | CI note（记录性能改善） |

> **设计决策**：5% 以内视为测量噪声，20% 以上几乎确定是真实回归。

---

## 10. Benchmark 模板

```rust
// benches/elementwise.rs
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
mod utils;
use utils::{SIZES_1D, data_gen};

fn bench_elem_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("elem_add_f64");

    for &size in SIZES_1D {
        let a = data_gen::sequential_1d(size);
        let b = data_gen::sequential_1d(size);

        group.bench_with_input(
            BenchmarkId::new("f64/contiguous", size),
            &size,
            |b, _| {
                b.iter(|| black_box(&a + &b));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_elem_add);
criterion_main!(benches);
```

---

## 11. Good / Bad 示例

### 11.1 Good — 正确的 benchmark 模式

```rust
// Good: 使用 black_box 防止编译器消除死代码
fn bench_sum(c: &mut Criterion) {
    let data = data_gen::sequential_1d(65_536);
    c.bench_function("sum_f64_65k", |b| {
        b.iter(|| black_box(data.sum()));
    });
}
```

### 11.2 Bad — 错误的 benchmark 模式

```rust
// Bad: 未使用 black_box，编译器可能消除运算
fn bench_sum_bad(c: &mut Criterion) {
    let data = data_gen::sequential_1d(65_536);
    c.bench_function("sum_f64_65k", |b| {
        b.iter(|| data.sum());  // 编译器可能优化掉
    });
}

// Bad: 在 iter 回调中构造数据，混入构造开销
fn bench_sum_bad2(c: &mut Criterion) {
    c.bench_function("sum_f64_65k", |b| {
        b.iter(|| {
            let data = data_gen::sequential_1d(65_536);  // 构造开销混入
            data.sum()
        });
    });
}
```

---

## 12. 实现任务拆分

### Wave 1: 基础设施

- [ ] **T1**: 配置 `Cargo.toml` bench 入口和 criterion 依赖
  - 文件: `Cargo.toml`
  - 内容: 添加 `criterion` dev-dependency，9 个 `[[bench]]` 入口
  - 测试: `cargo bench --bench elementwise -- --list` 输出正常
  - 前置: 无
  - 预计: 5 min

- [ ] **T2**: 实现 `benches/utils/mod.rs` 和 `benches/utils/data_gen.rs`
  - 文件: `benches/utils/mod.rs`, `benches/utils/data_gen.rs`
  - 内容: 共享常量（`SIZES_1D/2D/3D`），数据生成函数
  - 测试: 编译通过
  - 前置: T1
  - 预计: 10 min

### Wave 2: 核心基准

- [ ] **T3**: 实现 `benches/elementwise.rs`
  - 文件: `benches/elementwise.rs`
  - 内容: add/sub/mul/div/sin/exp/abs，覆盖 f32/f64/Complex\<f64\> + 非连续
  - 测试: `cargo bench --bench elementwise -- "elem_add" --quick`
  - 前置: T2
  - 预计: 10 min

- [ ] **T4**: 实现 `benches/reduction.rs`
  - 文件: `benches/reduction.rs`
  - 内容: sum_1d/sum_2d_axis0/sum_2d_axis1/sum_sliced
  - 测试: `cargo bench --bench reduction -- "sum" --quick`
  - 前置: T2
  - 预计: 10 min

- [ ] **T5**: 实现 `benches/dot_product.rs`
  - 文件: `benches/dot_product.rs`
  - 内容: dot_1d_f64/dot_1d_complex
  - 测试: `cargo bench --bench dot_product -- --quick`
  - 前置: T2
  - 预计: 10 min

- [ ] **T6**: 实现 `benches/set_ops.rs`
  - 文件: `benches/set_ops.rs`
  - 内容: unique_1d（不同规模、不同唯一值比例）
  - 测试: `cargo bench --bench set_ops -- --quick`
  - 前置: T2
  - 预计: 10 min

- [ ] **T7**: 实现 `benches/broadcast.rs`
  - 文件: `benches/broadcast.rs`
  - 内容: broadcast_scalar/broadcast_row/broadcast_col
  - 测试: `cargo bench --bench broadcast -- --quick`
  - 前置: T2
  - 预计: 10 min

### Wave 3: 形状与构造

- [ ] **T8**: 实现 `benches/shape_ops.rs`
  - 文件: `benches/shape_ops.rs`
  - 内容: transpose_2d/reshape_contiguous/reshape_noncontiguous
  - 测试: `cargo bench --bench shape_ops -- --quick`
  - 前置: T2
  - 预计: 10 min

- [ ] **T9**: 实现 `benches/construction.rs`
  - 文件: `benches/construction.rs`
  - 内容: zeros_1d/from_fn_2d/from_vec
  - 测试: `cargo bench --bench construction -- --quick`
  - 前置: T2
  - 预计: 10 min

### Wave 4: 对比基准

- [ ] **T10**: 实现 `benches/simd_comparison.rs`
  - 文件: `benches/simd_comparison.rs`
  - 内容: add/sum 在 `--features simd` 开/关时的对比
  - 测试: 分别以两种 feature 配置运行，对比结果
  - 前置: T3, T4
  - 预计: 10 min

- [ ] **T11**: 实现 `benches/parallel_comparison.rs`
  - 文件: `benches/parallel_comparison.rs`
  - 内容: sum/add 在 `--features parallel` 开/关时的对比
  - 测试: 分别以两种 feature 配置运行，对比结果
  - 前置: T3, T4
  - 预计: 10 min

### Wave 5: CI 集成

- [ ] **T12**: 配置 CI benchmark 工作流
  - 文件: `.github/workflows/bench.yml`
  - 内容: Smoke/Regression/Full 三级工作流，回归阈值配置
  - 测试: CI 触发运行
  - 前置: T3-T11
  - 预计: 10 min

### 并行执行分组图

```
Wave 1: [T1]
           │
Wave 2: [T2]
           │
Wave 3: [T3] [T4] [T5] [T6] [T7] [T8] [T9]
           │    │              │    │
           └────┴──────────────┴────┘
                       │
Wave 4:       [T10] [T11]
                 │
Wave 5:       [T12]
```

---

## 13. 测试计划

| 类型 | 位置 | 目的 |
|------|------|------|
| 编译验证 | `cargo bench --bench X -- --list` | 每个文件可编译和列出基准 |
| 单点运行 | `cargo bench --bench X -- "Y" --quick` | 快速验证基准可运行 |
| 回归检测 | CI `benchmark-action` | 与基线对比，检测 >20% 回归 |

---

## 14. ADR 决策记录

### 决策 1：使用 criterion.rs

| 属性 | 值 |
|------|-----|
| 决策 | 使用 criterion.rs 0.5 作为 benchmark 框架 |
| 理由 | 统计分析（置信区间、异常值检测）；HTML 报告；与 CI 集成成熟；stable Rust 可用 |
| 替代方案 | `#[bench]` nightly — 放弃，需要 nightly 编译器 |
| 替代方案 | divan — 放弃，生态不如 criterion 成熟 |

### 决策 2：按操作类别分文件

| 属性 | 值 |
|------|-----|
| 决策 | 每个操作类别一个 benchmark 文件 |
| 理由 | 可独立运行（`cargo bench --bench elementwise`）；编译时间可控；编译并行化 |
| 替代方案 | 单文件 — 放弃，编译慢、难以选择性运行 |

### 决策 3：参数矩阵使用 2 的幂次序列

| 属性 | 值 |
|------|-----|
| 决策 | 使用 64 / 65536 / 16777216 三级（2^6, 2^16, 2^24） |
| 理由 | 覆盖 SIMD 宽度边界、并行阈值边界；对齐友好；便于跨版本对比 |
| 替代方案 | 任意规模 — 放弃，不利于跨版本对比 |

### 决策 4：回归阈值 5%/20%

| 属性 | 值 |
|------|-----|
| 决策 | >5% warning，>20% fail，>5% 快记录改善 |
| 理由 | 5% 以内可能是测量噪声；20% 以上几乎确定是真实回归；改善记录供优化参考 |
| 替代方案 | 5% fail — 放弃，CI 环境噪声可能导致误报 |

### 决策 5：仅 F-order 布局测试

| 属性 | 值 |
|------|-----|
| 决策 | 不包含 C-order benchmark，非连续通过切片视图产生 |
| 理由 | Xenon 仅支持 F-order 布局（需求说明书 §7），C-order 不在范围内 |
| 替代方案 | 添加 C-order 测试 — 放弃，与需求矛盾 |

---

*本文档由 Xenon 维护。如有问题请提交 Issue 或 PR。*
