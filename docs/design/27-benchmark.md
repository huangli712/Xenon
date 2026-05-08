# 基准测试规范

> 文档编号: 27
> 模块目录: benches/
> 任务阶段: Phase 6
> 前置文档: 所有前置文档（`00-coding.md` ~ `26-error.md`）

---

## 1. 主题定位

本文档是"性能观测规范"，用于约定 benchmark 采样口径、趋势记录与可选回归检测；它不是完整质量门禁规范。功能正确性、错误语义、并发/UB 边界等质量要求统一由 `28-tests.md` 承担。

### 1.1 职责边界

| 职责         | 包含                           | 不包含                           |
| ------------ | ------------------------------ | -------------------------------- |
| 性能回归检测 | 可选工程增强：在仓库需要时运行，观测性能退化趋势 | 功能正确性验证（由集成测试覆盖） |
| 优化验证     | SIMD/并行优化效果的量化验证    | 编译时间测量                     |
| 性能档案     | 记录各操作在不同参数下的吞吐量 | 内存泄漏检测                     |
| 参数矩阵     | 多规模、多类型、多布局组合     | 跨语言对比（非核心目标）         |

### 1.2 设计原则

| 原则     | 体现                                                                        |
| -------- | --------------------------------------------------------------------------- |
| 可重复性 | 固定数据规模、固定随机种子、预热后测量                                      |
| 全面覆盖 | 覆盖所有性能关键路径（逐元素、归约、内积、转置、广播）                      |
| 低噪声   | 使用固定输入与稳定运行环境收集趋势；benchmark 结果不作为 crate 合约的一部分 |
| 分级执行 | 可选 CI 分级工作流：Smoke / Regression / Full，不阻塞默认交付               |

---

## 2. 需求映射与范围约束

| 类型     | 内容                                                                       |
| -------- | -------------------------------------------------------------------------- |
| 需求映射 | 需求说明书 §9、§28                                                         |
| 范围内   | benchmark 分类、参数矩阵、基准 harness 与结果汇总口径                      |
| 范围外   | 生产运行时性能调优、跨语言基准、额外平台专用测量框架                       |
| 非目标   | 通过 benchmark 文档扩展 crate 公共 API、引入非必要运行时依赖或改变需求边界 |

---

## 3. 文件位置

```
benches/
├── utils/
│   ├── mod.rs              # Shared constants and utility exports
│   └── generators.rs       # Test data generators
├── math.rs                 # Element-wise operation benchmarks
├── reduction.rs            # Reduction benchmarks (sum)
├── dot.rs                  # Vector dot-product benchmarks
├── set.rs                  # Set-operation benchmarks (unique)
├── broadcast.rs            # Broadcast-operation benchmarks
├── shape.rs                # Shape-operation benchmarks (transpose)
├── simd_comparison.rs      # SIMD comparison benchmarks
├── parallel_comparison.rs  # Parallel comparison benchmarks
└── construction.rs         # Constructor benchmarks
```

按操作类别分文件：可独立运行（`cargo bench --bench math`），编译时间可控。

---

## 4. 依赖关系

### 4.1 依赖图（ASCII）

```
benches/
├── crate::tensor           # TensorBase, Tensor, TensorView, etc.
├── crate::dimension        # Ix0~Ix6, IxDyn, Dimension trait
├── crate::element          # Element, Numeric, RealScalar, ComplexScalar
├── crate::math             # Element-wise ops
├── crate::reduction        # sum, sum_axis
├── crate::matrix           # dot
├── crate::shape            # transpose
├── crate::broadcast        # broadcast_shape
├── crate::set              # unique
├── crate::construct        # zeros, ones, from_shape_vec
└── optional local benchmark tools  # used only for maintenance-time performance observation
```

### 4.2 类型级依赖

| 来源模块    | 使用的 API（类型/trait/函数）                                                   |
| ----------- | ------------------------------------------------------------------------------- |
| `tensor`    | `Tensor`, `TensorView`, `TensorViewMut`, `.shape()`（参见 `07-tensor.md §5`）   |
| `dimension` | `Ix1`, `Ix2`, `Ix3`, `IxDyn`, `Dimension`（参见 `02-dimension.md §5`）          |
| `element`   | `Element`, `Numeric`, `RealScalar`, `ComplexScalar`（参见 `03-element.md §5`）  |
| `math`      | `add`, `sub`, `mul`, `div`, `sin`, `exp`, `abs`（参见 `11-math.md §5`）         |
| `reduction` | `sum`, `sum_axis`（参见 `13-reduction.md §5`）                                  |
| `matrix`    | `dot`（参见 `12-matrix.md §5`）                                                 |
| `shape`     | `transpose`（参见 `16-shape.md §5`）                                            |
| `set`       | `unique`（参见 `14-set.md §5`）                                                 |
| `construct` | `zeros`, `ones`, `from_shape_vec`（参见 `18-construction.md §5`）               |
| `broadcast` | `broadcast_shape`（参见 `15-broadcast.md §5`）、广播视图构造                    |

### 4.3 依赖合法性

| 项目           | 说明                                                                                           |
| -------------- | ---------------------------------------------------------------------------------------------- |
| 新增第三方依赖 | 当前基线不新增 benchmark 专用 dev-dependency；若后续希望引入外部框架，须单独裁决               |
| 合法性结论     | 以 `std::time::Instant`、`std::hint::black_box` 与仓库内脚本作为当前版本基线，符合最小依赖约束 |
| 替代方案       | 外部 benchmark 框架可作为未来方案评估，但不属于当前版本默认要求                                |

### 4.4 依赖方向声明

依赖方向：单向消费。`benches` 消费 crate 公共 API 与部分 `pub(crate)` 内部 API。benchmark 工具链属于可选维护设施，不被任何模块依赖。

---

## 5. Benchmark 结构设计

### 5.1 Cargo.toml 配置

```toml
[[bench]]
name = "math"
harness = false

[[bench]]
name = "reduction"
harness = false

[[bench]]
name = "dot"
harness = false

[[bench]]
name = "set"
harness = false

[[bench]]
name = "broadcast"
harness = false

[[bench]]
name = "shape"
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

### 5.2 共享工具模块

```rust,ignore
// benches/utils/mod.rs
pub mod generators;

/// Standard benchmark sizes: Small / Medium / Large (1D).
pub const SIZES_1D: &[usize] = &[64, 65_536, 16_777_216];

/// Standard benchmark sizes: Small / Medium / Large (2D square).
pub const SIZES_2D: &[(usize, usize)] = &[
    (8, 8), (256, 256), (4096, 4096),
];
```

```rust,ignore
// benches/utils/generators.rs
use xenon::prelude::*;

/// Generate a 1D tensor with sequential f64 values.
pub fn sequential_1d(n: usize) -> Tensor1<f64> {
    Tensor1::from_shape_vec([n], (0..n).map(|idx| idx as f64).collect())
        .expect("shape and data length must match")
}

/// Generate a 2D F-order tensor with sequential f64 values.
pub fn sequential_2d(rows: usize, cols: usize) -> Tensor2<f64> {
    Tensor2::from_shape_vec(
        [rows, cols],
        (0..rows * cols).map(|idx| idx as f64).collect(),
    )
    .expect("shape and data length must match")
}

/// Generate a truly non-contiguous 1D view from an F-order 2D owner.
/// NOTE: The slice construction below uses pseudo-code; actual API shape
/// must follow `17-indexing.md` frozen interface at implementation time.
pub struct StridedFixture1D {
    pub owner: Tensor2<f64>,
}

impl StridedFixture1D {
    /// Returns a non-contiguous view by taking a row slice of the F-order 2D owner.
    /// Actual slicing API follows `17-indexing.md`; the expression below
    /// is illustrative and must be adapted to the final slice syntax at
    /// implementation. `SliceInfo::new` performs structural validation only;
    /// shape-aware bounds checking happens inside `TensorBase::slice` (see
    /// 17-indexing §5.1).
    pub fn view(&self) -> TensorView1<'_, f64> {
        // Illustrative: take row 1 from the F-order 2D owner, yielding a
        // non-contiguous 1D view with stride == ncols (not 1).
        self.owner
            .view()
            .slice(SliceInfo::new(/* [Index(1), Range { start: 0, end: self.owner.ncols() }] */)
                .expect("structurally valid slice info"))
            .expect("slice bounds within owner shape")
    }
}

pub fn strided_view_1d(n: usize) -> StridedFixture1D {
    StridedFixture1D {
        owner: sequential_2d(2, n),
    }
}
```

### 5.3 四级分类体系

```
Benchmark categories
├── Micro-benchmarks        # Single operation / single function (for example zeros)
├── Kernel benchmarks       # Core compute kernels (for example add, sum, dot)
├── Workflow benchmarks     # Real usage flows (for example broadcast + add chain)
└── Comparison benchmarks   # In-repo baseline comparison (SIMD/scalar, parallel/serial)
```

| 级别       | 示例                                           | 用途           |
| ---------- | ---------------------------------------------- | -------------- |
| Micro      | `zeros_1d`, `from_shape_vec_1d`                | 基础开销基线   |
| Kernel     | `add_f64`, `sum_f64`, `dot_f64`                | 核心路径性能   |
| Workflow   | `broadcast_row`                                | 真实场景吞吐   |
| Comparison | `add_simd_vs_scalar`, `sum_parallel_vs_serial` | 仓库内路径对比 |

### 5.4 参数矩阵

#### 5.4.1 输入规模

| 级别       | 1D         | 2D        |
| ---------- | ---------- | --------- |
| **Small**  | 64         | 8×8       |
| **Medium** | 65,536     | 256×256   |
| **Large**  | 16,777,216 | 4096×4096 |

#### 5.4.2 数据类型

| 类型           | 优先级   | 说明              |
| -------------- | -------- | ----------------- |
| `f64`          | 必测     | 科学计算默认精度  |
| `f32`          | 必测     | SIMD 向量宽度更大 |
| `Complex<f64>` | 必测     | 复数运算开销验证  |

Comparison benchmark 可额外使用 `i32`、`i64` 等整数类型，以覆盖对应后端路径的特定加速场景（参见 `08-simd.md §5.6`、`09-parallel.md §5`）。

#### 5.4.3 内存布局

| 布局           | 构造方式                          | 验证目标           |
| -------------- | --------------------------------- | ------------------ |
| F-contiguous   | `zeros(shape)`                    | 默认路径性能基线   |
| Non-contiguous | F-order 2D 张量的行视图或转置视图 | 非连续路径性能惩罚 |

**注意**：

- Xenon 仅支持 F-order 布局，不存在 C-order 路径。非连续布局通过切片/转置视图产生（参见 `06-layout.md §5.3` / `§5.7`）。
- 数据竞争和 UB 验证由 `28-tests.md` 覆盖。benchmark 仅验证性能指标，不承担正确性验证职责。

### 5.5 Benchmark 清单

| 组 ID                      | 操作                    | 规模  | 类型           | 布局           | 说明                                            |
| -------------------------- | ----------------------- | ----- | -------------- | -------------- | ----------------------------------------------- |
| `elem_add_f64`             | `a + b`                 | S/M/L | f64            | F-contiguous   | 连续数组逐元素加法                              |
| `elem_add_f32`             | `a + b`                 | S/M/L | f32            | F-contiguous   | f32 加法，SIMD 向量宽度更大                     |
| `elem_add_complex`         | `a + b`                 | S/M/L | Complex<f64>   | F-contiguous   | 复数加法开销                                    |
| `elem_mul_f64`             | `a * b`                 | S/M/L | f64            | F-contiguous   | 逐元素乘法                                      |
| `elem_sin_f64`             | `sin(a)`                | S/M/L | f64            | F-contiguous   | 超越函数逐元素                                  |
| `elem_sin_f32`             | `sin(a)`                | S/M/L | f32            | F-contiguous   | f32 超越函数，SIMD 宽度更大                     |
| `elem_sub_f64`             | `a - b`                 | S/M/L | f64            | F-contiguous   | 逐元素减法                                      |
| `elem_div_f64`             | `a / b`                 | S/M/L | f64            | F-contiguous   | 逐元素除法                                      |
| `elem_exp_f64`             | `exp(a)`                | S/M/L | f64            | F-contiguous   | 指数函数逐元素                                  |
| `elem_abs_f64`             | `abs(a)`                | S/M/L | f64            | F-contiguous   | 绝对值逐元素                                    |
| `elem_add_sliced`          | `a + b`（b 为切片视图） | M     | f64            | Non-contiguous | 非连续惩罚                                      |
| `sum_1d_f64`               | 全局 sum                | S/M/L | f64            | F-contiguous   | 1D 归约                                         |
| `sum_2d_axis0`             | 沿轴 0 sum              | S/M/L | f64            | F-contiguous   | 2D 沿轴归约                                     |
| `sum_2d_axis1`             | 沿轴 1 sum              | S/M/L | f64            | F-contiguous   | 2D 沿轴归约                                     |
| `sum_sliced`               | 非连续 sum              | M     | f64            | Non-contiguous | 非连续归约惩罚                                  |
| `sum_2d_keepdims`          | 沿轴 sum 并保留维度     | S/M/L | f64            | F-contiguous   | keepdims 路径性能（参见 `13-reduction.md §5`）  |
| `dot_1d_f64`               | 向量内积                | S/M/L | f64            | F-contiguous   | 基本内积                                        |
| `dot_1d_complex`           | 复数内积                | S/M/L | Complex<f64>   | F-contiguous   | 复数内积（含共轭）                              |
| `unique_1d`                | unique 操作             | S/M/L | f64            | F-contiguous   | 返回不重复元素，结果无需排序且顺序不作要求      |
| `broadcast_scalar`         | 标量广播加法            | S/M/L | f64            | F-contiguous   | 标量广播开销                                    |
| `broadcast_row`            | 行向量广播到矩阵        | S/M/L | f64            | F-contiguous   | 行广播                                          |
| `broadcast_col`            | 列向量广播到矩阵        | S/M/L | f64            | F-contiguous   | 列广播                                          |
| `broadcast_with`           | 双张量广播协作          | S/M/L | f64            | F-contiguous   | broadcast_with 协作模式（参见 `15-broadcast.md §5`） |
| `transpose_2d`             | 2D 转置（零拷贝）       | S/M/L | f64            | F-contiguous   | 转置视图创建                                    |
| `simd_add_compare`         | `a + b` (SIMD vs 标量)  | M     | f32/f64        | F-contiguous   | SIMD 加速比（参见 `08-simd.md §12`）            |
| `simd_sum_compare`         | sum (SIMD vs 标量)      | M     | i32/f32/f64    | F-contiguous   | SIMD sum 对比；i32 覆盖整数 admission / scalar fallback 路径 |
| `simd_dot_compare`         | dot (SIMD vs 标量)      | M     | f32/f64        | F-contiguous   | SIMD dot kernel 已在 `08-simd.md` 中设计        |
| `par_sum_compare`          | sum (并行 vs 串行)      | L     | i64            | F-contiguous   | 并行加速比（参见 `09-parallel.md §12`）         |
| `par_add_compare`          | `a + b` (并行 vs 串行)  | L     | f64            | F-contiguous   | 并行逐元素加速                                  |
| `par_dot_compare`          | dot (并行 vs 串行)      | L     | f64/Complex<f64> | F-contiguous | 并行内积加速比，覆盖实数与复数内积              |
| `zeros_1d`                 | zeros 构造              | S/M/L | f64            | F-contiguous   | 构造开销                                        |
| `from_shape_vec_1d`        | from_shape_vec 构造     | S/M/L | f64            | F-contiguous   | 顺序数据构造开销                                |
| `eye_2d`                   | eye 单位矩阵构造        | S/M/L | f64            | F-contiguous   | 单位矩阵构造开销（参见 `18-construction.md §5`） |

### 5.6 可选 CI 三级工作流

| 工作流               | 基准数量                                                                    | 启用方式                     |
| -------------------- | --------------------------------------------------------------------------- | ---------------------------- |
| **Smoke Test**       | 3 个核心文件 × `--quick`                                                    | 仓库可按需启用；不阻塞交付   |
| **Regression Check** | `elem_add_f64` 和 `sum_1d_f64`                                              | 存在 baseline 时可选启用     |
| **Full Benchmark**   | 全部文件 × 4 feature 组合（`default`、`simd`、`parallel`、`simd+parallel`） | 维护期按需运行               |

**Smoke Test 覆盖范围**：

Smoke Test 仅验证 benchmark 代码可以正常编译和运行，不用于性能判断，也不执行回归阈值门禁。其调用约定统一使用 `--quick`；若仓库选择启用性能回归检测，再由 Regression Check 承担趋势观测。

| 文件              | 组                      | 说明           |
| ----------------- | ----------------------- | -------------- |
| `math.rs`         | `elem_add_f64` (Medium) | 核心逐元素路径 |
| `reduction.rs`    | `sum_1d_f64` (Medium)   | 核心归约路径   |
| `construction.rs` | `zeros_1d` (Medium)     | 基础构造路径   |

**Regression Check 覆盖范围**：

- 当仓库显式启用 Regression Check 时，可监测以下核心基准：`elem_add_f64`（逐元素加法，f64，65536 元素）和 `sum_1d_f64`（一维归约，f64，65536 元素）。其中 `65536` 与 §5.4.1 的 1D Medium 规模保持一致。
- 本文档统一使用同一组规模基线：Small = `64` / `8×8`，Medium = `65,536` / `256×256`，Large = `16,777,216` / `4096×4096`。
- Large 规模基准测试（如 `4096×4096`）仅在 weekly/full benchmark 流水线中执行。PR 级别的 Smoke Test 仅使用 Small/Medium 规模。CI 配置须设置合理的内存上限；当可用内存低于 Large 用例估算峰值的 1.5× 安全阈值时，大张量测试应跳过并标记为 skipped。

**CI 配置示例**：

```yaml
# .github/workflows/bench.yml
benchmark-smoke:
    runs-on: self-hosted  # Fixed hardware for reproducibility
    steps:
        - uses: actions/checkout@v4

        - name: Smoke benchmarks
          run: |
            cargo bench --bench math -- "elem_add_f64" --quick
            cargo bench --bench reduction -- "sum_1d_f64" --quick
            cargo bench --bench construction -- "zeros_1d" --quick

        - name: Summarize results
          run: python tools/bench/report.py --input target/benchmark-results --output target/benchmark-results/regression.json
```

- **说明**：若仓库选择为 benchmark 增加 CI 摘要，则须通过仓库内脚本显式导出到约定路径（如 `target/benchmark-results/regression.json`），而不是依赖第三方 GitHub Action 或外部服务。
- **baseline 管理**：若启用 Regression Check，则以上一轮 main 分支通过的结果作为 baseline；当性能改善或已知噪声需要更新基线时，应在专门的 benchmark PR 中更新并记录原因。未启用时不影响默认交付。

### 5.7 可选回归阈值

| 级别     | 阈值       | 动作                                      |
| -------- | ---------- | ----------------------------------------- |
| **警告** | > 5% 变慢  | 可选 CI warning（不阻塞默认交付）         |
| **失败** | > 20% 变慢 | 可选 CI failure gate（仅仓库显式启用时）  |
| **改善** | > 5% 变快  | 可选 CI note（记录性能改善）              |

**设计决策**：5% 以内视为测量噪声，20% 以上通常可视为真实回归；这些门限只在仓库显式启用回归门禁时生效，不构成默认必需交付。

### 5.8 Benchmark 模板

```rust,ignore
// benches/math.rs
use std::hint::black_box;
use std::time::Instant;
mod utils;
use utils::{SIZES_1D, generators};

const WARMUP_ITERATIONS: usize = 10;
const ROUNDS: usize = 10;
const ITERATIONS_PER_ROUND: usize = 100;

fn bench_elem_add(quick: bool) {
    let (rounds, iters) = if quick { (1, 10) } else { (ROUNDS, ITERATIONS_PER_ROUND) };
    for &size in SIZES_1D {
        let a = generators::sequential_1d(size);
        let b = generators::sequential_1d(size);
        // Same-shape add path always succeeds; unwrap is a sanity check, not hot-path noise.
        for _ in 0..WARMUP_ITERATIONS {
            black_box((&a + &b).unwrap());
        }
        let mut timings = Vec::with_capacity(rounds);
        for _round in 0..rounds {
            let started_at = Instant::now();
            for _iteration in 0..iters {
                let _result = black_box((&a + &b).unwrap());
            }
            timings.push(started_at.elapsed().as_nanos());
        }
        timings.sort();
        let median = timings[timings.len() / 2];
        println!("elem_add_f64/{size}: {median} ns (median of {rounds} rounds)");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let quick = args.iter().any(|arg| arg == "--quick");
    let filter = args.iter().find(|a| !a.starts_with("--"));

    let should_run = |name: &str| -> bool {
        match &filter {
            Some(f) => name.contains(f.as_str()),
            None => true,
        }
    };

    if should_run("elem_add_f64") {
        bench_elem_add(quick);
    }
}
```

### 5.9 Good / Bad 示例

```rust,ignore
// Good: Use black_box and a dedicated timing loop
fn bench_sum() {
    let data = generators::sequential_1d(65_536);
    let started_at = Instant::now();
    for _iteration in 0..100 {
        let _result = black_box(data.sum());
    }
    println!("sum_f64_65k: {:?}", started_at.elapsed());
}
```

```rust,ignore
// Bad: Not using black_box, compiler may eliminate the operation
fn bench_sum_bad() {
    let data = generators::sequential_1d(65_536);
    for _iteration in 0..100 {
        let _result = data.sum();  // Compiler may optimize this away
        let _ = _result;
    }
}

// Bad: Constructing data inside the timed loop, mixing in construction overhead
fn bench_sum_bad2() {
    let started_at = Instant::now();
    for _iteration in 0..100 {
        let data = generators::sequential_1d(65_536);  // Construction overhead mixed in
        let _result = data.sum();
        let _ = _result;
    }
    println!("sum_f64_65k_bad: {:?}", started_at.elapsed());
}
```

---

## 6. 内部实现设计

### 6.1 测量方法论

| 阶段            | 说明                                                     |
| --------------- | -------------------------------------------------------- |
| 预热 (warm-up)  | 每个基准先运行固定轮数预热，消除冷启动效应               |
| 采样 (sampling) | 固定迭代次数，并进行多轮采样（至少 10 轮）；Smoke Test 使用缩减轮数 |
| 汇总            | 由仓库脚本统一读取 stdout / JSON 摘要，使用 median 而非 mean 计算回归百分比 |
| 报告            | CLI 文本摘要；如后续需要更复杂报告，须单独裁决引入新工具 |

使用仓库内轻量 benchmark harness：
- 预热后以 `Instant` 计时，并由脚本统一汇总结果。
- 记录环境信息：至少包含 CPU、编译配置、feature flags。
- 可选 CPU pinning：如仓库环境允许，可将 benchmark 进程绑定到固定核心以降低抖动。
- 当前测量方法为简化版本。回归阈值（5%/20%）为参考值，CI 环境波动可能导致误判。未来可升级为统计显著性检验。

若仓库启用 benchmark 结果持久化，结果文件应至少包含以下字段：`benchmark_id`、`feature_set`、`input_shape`、`element_type`、`layout_kind`、`warmup_iterations`、`measured_iterations`、`wall_time_ns`、`timestamp`、`git_commit`、`environment_label`。JSON / CSV 载体可自行选择，但字段语义必须保持稳定，以便 Regression Check 消费。

### 6.2 数据生成策略

| 策略          | 实现                                                            | 说明                   |
| ------------- | --------------------------------------------------------------- | ---------------------- |
| 顺序填充      | 预先构造顺序 `Vec<f64>` 后用 `from_shape_vec` 导入；`from_vec` 仅作为 Ix1 非规范便捷层对照 | 可重复，无随机性       |
| 预分配 + 复用 | 数据在计时循环外生成                                            | 避免测量中混入构造开销 |
| 非连续视图    | 行视图或转置视图                                                | 模拟真实非连续访问场景 |

**设计决策**：所有数据在迭代回调外预生成，确保仅测量目标操作本身的性能。

### 6.3 black_box 使用规范

```rust,ignore
// Good: black_box wraps the entire expression to prevent dead code elimination
let _result = black_box((&a + &b).unwrap());

// Good: black_box wraps inputs to prevent constant folding
let _result = black_box(&a) + black_box(&b);

// Bad: forgetting black_box allows compiler to optimize away
let _result = (&a + &b).unwrap();
```

`black_box` 的作用是告诉编译器"此值可能被以任何方式使用"，防止编译器将结果视为死代码而消除整个计算。

### 6.4 基线消费表

| 基准 ID        | 输入规模 | baseline 来源              | 报告指标             | 阈值消费者       |
| -------------- | -------- | -------------------------- | -------------------- | ---------------- |
| `elem_add_f64` | 65,536   | 最近一次 main 分支通过结果 | wall time / change % | 可选 Regression Check |
| `sum_1d_f64`   | 65,536   | 最近一次 main 分支通过结果 | wall time / change % | 可选 Regression Check |

### 6.5 数值正确性引用边界

benchmark 不定义正确性容差，也不在本文件内重复维护 `MathTolerance`（`{ ulp, abs }`）或其他数值比较契约。所有数值正确性判断统一引用 `28-tests.md` 中冻结的数值契约：默认比较遵循 ULP-based contract，仅在文档明确允许的数学函数场景使用单独容差规则。benchmark 侧只允许复用这些契约来说明“性能观测建立在既有正确性测试之上”，不得把 benchmark 结果或阈值升级为新的正确性门禁。参见 `需求说明书 §28.3`。

---

## 7. 实现任务拆分

### Wave 1: 基础设施

- [ ] **T1**: 配置 `Cargo.toml` bench 入口和仓库内基准 harness
  - 文件: `Cargo.toml`
  - 内容: 添加 9 个 `[[bench]]` 入口，不新增 benchmark 专用第三方依赖
  - 测试: `cargo bench --bench math -- "elem_add_f64" --quick` 能编译并冒烟运行至少一个基准组（自定义 `harness = false` 模板仅识别 `--quick` 与单个 filter 字符串，不支持 `--list`）
  - 前置: 无
  - 预计: 5 min

- [ ] **T2**: 实现 `benches/utils/mod.rs` 和 `benches/utils/generators.rs`
  - 文件: `benches/utils/mod.rs`, `benches/utils/generators.rs`
  - 内容: 共享常量（`SIZES_1D`、`SIZES_2D`），数据生成函数
  - 测试: 编译通过
  - 前置: T1
  - 预计: 10 min

### Wave 2: 核心基准

- [ ] **T3**: 实现 `benches/math.rs`
  - 文件: `benches/math.rs`
  - 内容: add/sub/mul/div/sin/exp/abs，覆盖 f32/f64/Complex\<f64\> + 非连续
  - 测试: `cargo bench --bench math -- "elem_add" --quick`
  - 前置: T2
  - 预计: 10 min

- [ ] **T4**: 实现 `benches/reduction.rs`
  - 文件: `benches/reduction.rs`
  - 内容: sum_1d_f64/sum_2d_axis0/sum_2d_axis1/sum_sliced/sum_2d_keepdims
  - 测试: `cargo bench --bench reduction -- "sum" --quick`
  - 前置: T2
  - 预计: 10 min

- [ ] **T5**: 实现 `benches/dot.rs`
  - 文件: `benches/dot.rs`
  - 内容: dot_1d_f64/dot_1d_complex
  - 测试: `cargo bench --bench dot -- --quick`
  - 前置: T2
  - 预计: 10 min

- [ ] **T6**: 实现 `benches/set.rs`
  - 文件: `benches/set.rs`
  - 内容: unique_1d（不同规模、不同唯一值比例）
  - 测试: `cargo bench --bench set -- --quick`
  - 前置: T2
  - 预计: 10 min

- [ ] **T7**: 实现 `benches/broadcast.rs`
  - 文件: `benches/broadcast.rs`
  - 内容: broadcast_scalar/broadcast_row/broadcast_col/broadcast_with
  - 测试: `cargo bench --bench broadcast -- --quick`
  - 前置: T2
  - 预计: 10 min

### Wave 3: 形状与构造

- [ ] **T8**: 实现 `benches/shape.rs`
  - 文件: `benches/shape.rs`
  - 内容: transpose_2d
  - 测试: `cargo bench --bench shape -- --quick`
  - 前置: T2
  - 预计: 10 min

- [ ] **T9**: 实现 `benches/construction.rs`
  - 文件: `benches/construction.rs`
  - 内容: zeros_1d/from_shape_vec_1d/eye_2d
  - 测试: `cargo bench --bench construction -- --quick`
  - 前置: T2
  - 预计: 10 min

### Wave 4: 对比基准

- [ ] **T10**: 实现 `benches/simd_comparison.rs`
  - 文件: `benches/simd_comparison.rs`
  - 内容: add/sum/dot 在 `--features simd` 开/关时的性能对比；路径切换正确性由 `28-tests.md` 验证
  - 测试: 分别以两种 feature 配置运行，对比结果
  - 前置: T3, T4, T5
  - 预计: 10 min

- [ ] **T11**: 实现 `benches/parallel_comparison.rs`
  - 文件: `benches/parallel_comparison.rs`
  - 内容: sum/add/dot 在 `--features parallel` 开/关时的性能对比；自动切换与嵌套回退正确性由 `28-tests.md` 验证
  - 测试: 分别以两种 feature 配置运行，对比结果
  - 前置: T3, T4, T5
  - 预计: 10 min

### Wave 5: CI 集成

- [ ] **T12**: 配置可选 CI benchmark 工作流
  - 文件: `.github/workflows/bench.yml`, `tools/bench/report.py`
  - 内容: Smoke/Regression/Full 三级工作流与可选回归阈值配置；`tools/bench/report.py` 负责读取 stdout 或 JSON 摘要并输出回归判定结果；不作为默认交付阻塞项
  - 测试: CI 触发运行
  - 前置: T3-T11
  - 预计: 15 min

---

## 8. 测试计划

### 8.1 测试分类表

| 类型     | 位置                                   | 目的                                                     |
| -------- | -------------------------------------- | -------------------------------------------------------- |
| 单元验证 | `cargo bench --bench X -- "<name>" --quick` | 每个 benchmark 文件可编译并冒烟运行至少一个基准组   |
| 集成验证 | `cargo bench --bench X -- "Y" --quick` | 快速验证单个 benchmark 可运行，输入与 feature 组合正确   |
| 边界验证 | CI smoke/regression                    | 验证 benchmark 分组、输入规模、feature 组合与回归阈值流程符合预期 |
| 基线校验 | benchmark 输入准备与路径选择检查       | 验证 benchmark 分组、输入规模与 feature 组合符合预期     |

### 8.2 Feature gate / 配置测试

| 配置        | 验证点                                                                 |
| ----------- | ---------------------------------------------------------------------- |
| 默认配置    | benchmark 能在默认 `std` 配置下编译并执行核心基线                      |
| 启用 `simd` | SIMD comparison 组在 `simd` feature 下可用，并记录路径切换后的性能     |
| 启用并行    | parallel comparison 组在 `parallel` feature 下可用，并覆盖阈值切换行为 |
| 全 feature  | benchmark 入口、结果采集与 CI 工作流在组合配置下保持有效               |

### 8.3 类型边界 / 编译期测试

| 场景                         | 测试方式                                           |
| ---------------------------- | -------------------------------------------------- |
| feature-gated benchmark 入口 | `cargo bench --bench ... --features ... -- "<filter>" --quick`（自定义 `harness=false` runner 仅识别 filter + `--quick`，不实现 `--list`） |
| SIMD / 并行比较组导出边界    | 配置矩阵编译检查                                   |
| 非法 feature 组合            | CI 配置矩阵与 smoke check                          |

---

## 9. 模块交互设计

### 9.1 Benchmark 文件到被测模块映射

| Benchmark 文件           | 被测模块            | 对应设计文档         |
| ------------------------ | ------------------- | -------------------- |
| `math.rs`                | `math`              | `11-math.md`         |
| `reduction.rs`           | `reduction`         | `13-reduction.md`    |
| `dot.rs`                 | `matrix`            | `12-matrix.md`       |
| `set.rs`                 | `set`               | `14-set.md`          |
| `broadcast.rs`           | `broadcast`         | `15-broadcast.md`    |
| `shape.rs`               | `shape`             | `16-shape.md`        |
| `simd_comparison.rs`     | `simd` + `math`     | `08-simd.md`         |
| `parallel_comparison.rs` | `parallel` + `math` | `09-parallel.md`     |
| `construction.rs`        | `construct`         | `18-construction.md` |

### 9.2 数据流

```
benchmark files
    │
    ├── call crate public APIs (Tensor::from_shape_vec, zeros, +, sum, ...)
    │       │
    │       └── internal path: storage -> tensor -> overload -> simd/parallel
    │
    └── repository-local harness measures end-to-end runtime
```

---

## 10. 错误处理与语义边界

本文档不直接定义错误类型，但要求 benchmark 代码、回归脚本与 CI 汇总逻辑在遇到被测 API 失败时遵循 `26-error.md` 的错误语义边界；基准层只允许记录或传播既有错误，不得自定义新的公开错误语义。

---

## 11. 设计决策记录

### 决策 1：不引入额外 benchmark 框架

| 属性     | 值                                                                                            |
| -------- | --------------------------------------------------------------------------------------------- |
| 决策     | 当前版本 benchmark 基线使用 `std::time::Instant`、`std::hint::black_box` 与仓库内结果汇总脚本 |
| 理由     | 满足最小依赖约束，并可在 stable Rust 下落地                                                   |
| 替代方案 | `#[bench]` nightly — 放弃，需要 nightly 编译器                                                |
| 替代方案 | 外部 benchmark 框架 — 暂不采用，需先单独裁决 dev-dependency 政策                              |

### 决策 2：按操作类别分文件

| 属性     | 值                                                                 |
| -------- | ------------------------------------------------------------------ |
| 决策     | 每个操作类别一个 benchmark 文件                                    |
| 理由     | 可独立运行（`cargo bench --bench math`）；编译时间可控；编译并行化 |
| 替代方案 | 单文件 — 放弃，编译慢、难以选择性运行                              |

### 决策 3：参数矩阵使用 2 的幂次序列

| 属性     | 值                                                         |
| -------- | ---------------------------------------------------------- |
| 决策     | 使用 64 / 65536 / 16777216 三级（2^6, 2^16, 2^24）         |
| 理由     | 覆盖 SIMD 宽度边界、并行阈值边界；对齐友好；便于跨版本对比 |
| 替代方案 | 任意规模 — 放弃，不利于跨版本对比                          |

### 决策 4：CI 回归阈值属于可选工程增强

| 属性     | 值                                                                    |
| -------- | --------------------------------------------------------------------- |
| 决策     | 若仓库后续启用 Regression Check，可采用 >5% warning、>20% fail、>5% 改善记录 的工程门限 |
| 理由     | 该门限有助于持续观察性能趋势，但属于 CI/维护流程增强，不是 `需求说明书` 当前版本的必需交付 |
| 替代方案 | 完全不设门限 — 允许，当前版本至少需保留可重复 benchmark 方案与结果汇总口径 |

### 决策 5：仅 F-order 布局测试

| 属性     | 值                                                             |
| -------- | -------------------------------------------------------------- |
| 决策     | 不包含 C-order benchmark，非连续通过切片视图产生               |
| 理由     | Xenon 仅支持 F-order 布局（`需求说明书 §7`），C-order 不在范围内 |
| 替代方案 | 添加 C-order 测试 — 放弃，与需求矛盾                           |

---

## 12. 性能描述

| 方面 | 说明 |
| ---- | ---- |
| 覆盖范围 | 重点观察逐元素运算、归约、内积、广播、转置、构造等当前版本范围内的性能关键路径 |
| 对比口径 | 默认记录 wall time 与相对变化；SIMD / 并行仅在对应 feature 启用时进行对比 |
| 非目标 | 不把 CI 回归阈值、baseline 门禁或固定硬件流程视为当前版本必需交付 |
| 语义边界 | benchmark 结果只描述性能表现，不改变 `需求说明书` 规定的公开 API 语义与正确性契约 |

---

## 13. 平台与工程约束

| 约束项     | 约束内容                                              |
| ---------- | ----------------------------------------------------- |
| 平台支持   | benchmark 方案仅覆盖 `std` 环境                       |
| MSRV       | Rust 1.85+                                            |
| crate 结构 | 保持单 crate，不为 benchmark 拆分独立 crate           |
| SemVer     | 无影响；benchmark 与性能观测流程不构成 crate 稳定 API 合约 |
| 最小依赖   | 不引入 benchmark 专用第三方依赖；不扩展额外运行时依赖 |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
