# 基准测试方案设计

> 文档编号: 22 | 目录: `benches/` | 阶段: Phase 5（测试与文档）
> 前置文档: `00-rust-standards.md`, `01-architecture-overview.md`, `20-simd.md`, `21-parallel.md`
> 需求参考: 需求说明书 §7.1~§7.3, §17.2

---

## 1. 模块定位

基准测试模块是 Renon 性能保障体系的核心，为持续迭代提供可量化的性能基线。其价值不仅在于"测多快"，更在于回答以下问题：

| 问题 | 基准测试的回答方式 |
|------|-------------------|
| 这次重构是否引入性能退化？ | 回归检测：对比分支/版本间的 criterion 报告 |
| SIMD 路径是否真的比标量快？ | 特性对比：同一操作在 `--no-default-features` 与 `--features full` 下的吞吐量 |
| 并行阈值（64K）是否合理？ | 参数扫描：不同元素规模下串行 vs 并行的 crossover 点 |
| 某操作是否可安全用于热路径？ | 绝对延迟：μs 级 wall-clock 时间 + 元素吞吐量 |
| 非连续布局的惩罚有多大？ | 布局对比：contiguous vs sliced vs transposed 的性能倍率 |
| Renon 与 ndarray 的差距在哪？ | 对比基准：相同操作的第三方库对照 |

### 核心设计目标

| 目标 | 体现 |
|------|------|
| 回归检测 | criterion 统计分析 + CI 阈值告警 |
| 性能分层验证 | 标量 / SIMD / 并行 / SIMD+并行 四档对比 |
| 布局敏感 | F-contiguous / C-contiguous / 非连续（切片）三种布局 |
| 类型覆盖 | f32 / f64 / Complex\<f64\> 三种核心数据类型 |
| 规模梯度 | 小（8×8）/ 中（256×256）/ 大（4096×4096）三级输入规模 |
| 可比性 | 与 ndarray / raw loop / BLAS 的对照基准 |
| CI 友好 | 快速烟雾测试 + 完整基准分离 |

### 职责边界

| 职责 | 包含 | 不包含 |
|------|------|--------|
| 性能度量 | wall-clock 时间、吞吐量（元素/秒、GFLOPS） | 正确性验证（由集成测试负责） |
| 回归检测 | criterion 统计分析 + CI 阈值 | 功能回归（由 CI test 负责） |
| 特性对比 | scalar vs SIMD vs parallel 分档对比 | 编译时间基准 |
| 内存度量 | 分配次数/大小（可选） | 内存泄漏检测（由 Miri/ASan 负责） |
| 竞品对照 | ndarray / raw loop 对照基准 | 第三方库 API 兼容性 |

---

## 2. 文件位置

```
benches/
├── tensor_construct.rs    # 张量构造：zeros/ones/full/eye/from_vec/arange
├── element_ops.rs         # 逐元素运算：add/sub/mul/div/sin/exp/abs
├── matrix_ops.rs          # 矩阵运算：matvec/dot/outer/batch_matvec
├── reduction.rs           # 归约：sum/prod/min/max/mean/var/argmin
├── shape_ops.rs           # 形状操作：reshape/transpose/slice/squeeze/permute
├── iterator.rs            # 迭代器：iter/axis_iter/zip/windows
├── simd.rs                # SIMD 专项：SIMD vs 标量对比、对齐/非对齐、尾部处理
├── parallel.rs            # 并行专项：并行 vs 串行对比、阈值扫描、分块策略
└── ffi.rs                 # FFI：指针 API 开销、BLAS 兼容性检查
```

在 `Cargo.toml` 中的声明：

```toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
ndarray = "0.16"           # comparison benchmarks (optional)
rand = "0.8"               # test data generation
approx = "0.5"             # floating-point comparison in validation

[[bench]]
name = "tensor_construct"
harness = false

[[bench]]
name = "element_ops"
harness = false

[[bench]]
name = "matrix_ops"
harness = false

[[bench]]
name = "reduction"
harness = false

[[bench]]
name = "shape_ops"
harness = false

[[bench]]
name = "iterator"
harness = false

[[bench]]
name = "simd"
harness = false
required-features = ["simd"]

[[bench]]
name = "parallel"
harness = false
required-features = ["parallel"]

[[bench]]
name = "ffi"
harness = false
```

**文件规模预估：**

| 文件 | 预计行数 | 基准组数 | 说明 |
|------|----------|----------|------|
| tensor_construct.rs | ~250 | 6 | 构造函数开销小，基准数量少 |
| element_ops.rs | ~500 | 12 | 多操作 × 多类型 × 多布局 |
| matrix_ops.rs | ~400 | 8 | matvec/dot 是热路径，需精细测试 |
| reduction.rs | ~400 | 10 | 全局 + 沿轴 + 累积 |
| shape_ops.rs | ~300 | 8 | 零拷贝操作验证 |
| iterator.rs | ~350 | 8 | 迭代模式对比 |
| simd.rs | ~450 | 10 | SIMD 专项对比 |
| parallel.rs | ~400 | 8 | 并行专项 + 阈值扫描 |
| ffi.rs | ~200 | 5 | 指针 API 微基准 |
| **合计** | **~3250** | **75** | — |

---

## 3. 依赖关系

### 3.1 外部依赖

| 依赖 | 版本 | 用途 | 约束 |
|------|------|------|------|
| `criterion` | 0.5 | 基准测试框架：统计分析、HTML 报告、回归检测 | dev-dependency，启用 `html_reports` feature |
| `ndarray` | 0.16 | 对比基准：同操作 vs ndarray 实现 | dev-dependency，仅用于 bencher 内部 |
| `rand` | 0.8 | 生成随机测试数据 | dev-dependency |
| `approx` | 0.5 | 浮点结果验证（确保基准测试正确性） | dev-dependency |

### 3.2 内部依赖（benches 对 src/ 的引用）

```
benches/tensor_construct.rs
├── crate::construction     # zeros, ones, full, eye, arange, linspace
├── crate::tensor           # Tensor, TensorBase
├── crate::dimension        # Ix1, Ix2, IxDyn
├── crate::element          # RealScalar
└── crate::layout           # Order

benches/element_ops.rs
├── crate::ops              # add, sub, mul, div, sin, cos, exp, ln, abs
├── crate::tensor           # Tensor
├── crate::construction     # zeros (setup)
└── crate::element          # Numeric, RealScalar, ComplexScalar

benches/matrix_ops.rs
├── crate::ops::matrix      # matvec, dot, outer, batch_matvec
├── crate::tensor           # Tensor, Tensor2
├── crate::construction     # zeros, from_vec
└── crate::element          # Numeric

benches/reduction.rs
├── crate::ops::reduction   # sum, prod, min, max, mean, var, argmin, cumsum
├── crate::tensor           # Tensor
├── crate::construction     # zeros (setup)
└── crate::element          # Numeric, RealScalar

benches/shape_ops.rs
├── crate::shape            # reshape, transpose, slice, squeeze, permute
├── crate::tensor           # Tensor, TensorView
├── crate::construction     # zeros (setup)
├── crate::macros           # s![] for slicing
└── crate::dimension        # Ix2, Ix3, IxDyn

benches/iterator.rs
├── crate::iter             # iter, axis_iter, windows, zip
├── crate::tensor           # Tensor
└── crate::construction     # zeros (setup)

benches/simd.rs
├── crate::ops              # element-wise ops (SIMD accelerated)
├── crate::backend::simd    # low-level SIMD kernels
├── crate::tensor           # Tensor
├── crate::layout           # LayoutFlags, alignment checks
└── crate::element          # RealScalar

benches/parallel.rs
├── crate::ops              # element-wise, reduction (parallel enabled)
├── crate::backend::parallel # parallel dispatch
├── crate::tensor           # Tensor
└── crate::element          # Numeric

benches/ffi.rs
├── crate::ffi              # as_ptr, as_mut_ptr, lda, blas_layout
├── crate::tensor           # Tensor
└── crate::construction     # zeros (setup)
```

---

## 4. Benchmark 分类

### 4.1 四级分类体系

```
Benchmark 分类
├── Micro-benchmarks        # 单操作、单函数级别
│   ├── 构造：zeros, ones, from_vec, eye
│   ├── 索引：get, get_unchecked
│   └── 布局：is_contiguous, layout_flags
│
├── Kernel benchmarks       # 核心计算内核
│   ├── 逐元素：add, mul, sin, exp (连续/非连续)
│   ├── 归约：sum, mean, var, argmin
│   ├── 矩阵：matvec, dot, batch_matvec
│   └── 累积：cumsum, cumprod
│
├── Workflow benchmarks     # 真实使用模式
│   ├── 线性回归前向/反向
│   ├── 信号处理（FFT 前置：加窗 + 归一化）
│   └── 批量推理（batch_matvec + broadcast add + 归约）
│
└── Comparison benchmarks   # 外部对比
    ├── vs ndarray：同操作的 ndarray 实现
    ├── vs raw loop：手写循环基线
    └── vs BLAS（通过 FFI 调用）：matvec 对照
```

### 4.2 分类在各文件中的分布

| 文件 | Micro | Kernel | Workflow | Comparison |
|------|-------|--------|----------|------------|
| tensor_construct.rs | ✅ 构造开销 | — | — | vs ndarray::Array |
| element_ops.rs | ✅ 单操作 | ✅ 多类型/布局 | — | vs ndarray/loop |
| matrix_ops.rs | — | ✅ matvec/dot | — | vs ndarray |
| reduction.rs | ✅ 全局归约 | ✅ 沿轴/累积 | — | vs ndarray |
| shape_ops.rs | ✅ reshape/transpose | — | — | vs ndarray |
| iterator.rs | ✅ iter 开销 | — | ✅ zip 工作流 | — |
| simd.rs | — | ✅ SIMD kernels | — | vs scalar |
| parallel.rs | — | ✅ 并行 kernels | — | vs serial |
| ffi.rs | ✅ 指针开销 | — | — | — |

---

## 5. 每个 Benchmark 的具体设计

### 5.1 通用输入参数矩阵

每个 benchmark 在设计时须覆盖以下参数组合的**有意义子集**（非全排列，避免指数爆炸）：

#### 5.1.1 输入规模

| 级别 | 1D | 2D | 3D | 用途 |
|------|-----|-----|-----|------|
| **Small** | 64 | 8×8 (64) | 4×4×4 (64) | 验证冷启动/小数组开销 |
| **Medium** | 65_536 | 256×256 (65K) | 64×32×32 (65K) | 典型工作负载 |
| **Large** | 16_777_216 | 4096×4096 (16M) | 256×256×256 (16M) | 压力测试、并行/SIMD 充分利用 |
| **XL** | — | 8192×8192 (67M) | — | 并行专项：大数组饱和测试 |

#### 5.1.2 数据类型

| 类型 | 优先级 | 覆盖场景 |
|------|--------|----------|
| `f64` | **必测** | 科学计算默认精度，最高频使用 |
| `f32` | **必测** | SIMD 向量宽度更大（8×f32 vs 4×f64），机器学习场景 |
| `Complex<f64>` | **必测** | 复数运算开销验证，步长=2 的特殊内存模式 |

#### 5.1.3 内存布局

| 布局 | 构造方式 | 验证目标 |
|------|----------|----------|
| **F-contiguous** | `zeros(shape, Order::F)` | 默认路径性能基线 |
| **C-contiguous** | `zeros(shape, Order::C)` | C-order 兼容性 |
| **Non-contiguous (sliced)** | `tensor.slice(s![.., 0..n-1])` | 非连续路径标量回退惩罚 |
| **Non-contiguous (transposed)** | `tensor.t()` | 转置视图性能 |
| **Broadcasted** | `tensor.broadcast(shape)` | 零步长迭代开销 |

#### 5.1.4 Feature 组合

| 组合 | 编译命令 | 验证目标 |
|------|----------|----------|
| Scalar only | `--no-default-features --features std` | 纯标量基线 |
| + SIMD | `--features std,simd` | SIMD 加速倍率 |
| + Parallel | `--features std,parallel` | 并行加速倍率 |
| SIMD + Parallel | `--features full` | 最大性能 |

> **注意**：Feature 组合在 cargo bench 层面切换，不是单个 benchmark 内部切换。CI 中分四个 job 运行。

---

### 5.2 tensor_construct.rs — 张量构造

#### 基准组列表

| 组 ID | 操作 | 规模 | 类型 | 说明 |
|-------|------|------|------|------|
| `construct_zeros` | `zeros::<f64, Ix2>(shape)` | S/M/L | f64 | 全零数组创建（含分配+初始化） |
| `construct_zeros_f32` | `zeros::<f32, Ix2>(shape)` | S/M/L | f32 | f32 构造对比 |
| `construct_zeros_c_order` | `zeros(shape, Order::C)` | M | f64 | C-order 构造开销 |
| `construct_ones` | `ones::<f64, Ix2>(shape)` | S/M/L | f64 | 全一数组 |
| `construct_from_vec` | `Tensor::from_vec(vec, shape)` | S/M/L | f64 | 从 Vec 构造（无初始化，仅元数据） |
| `construct_eye` | `eye::<f64>(n)` | 8/64/256/1024 | f64 | 单位矩阵（对角填充） |
| `construct_arange` | `arange(0.0, n as f64, 1.0)` | 64/65K/16M | f64 | 序列生成 |
| `construct_linspace` | `linspace(0.0, 1.0, n)` | 64/65K/16M | f64 | 线性等分 |
| `construct_zeros_dyn` | `zeros::<f64, IxDyn>(shape)` | M | f64 | 动态维度构造开销 |
| `construct_zeros_ndarray` | `ndarray::Array2::<f64>::zeros(shape)` | S/M/L | f64 | 对比：ndarray 构造 |

#### 设计要点

- **测量策略**：wall-clock 时间 + throughput（elements/sec）
- **关键对比**：Renon zeros vs ndarray zeros — 验证分配+初始化开销
- **避坑**：构造函数可能被 LLVM 优化掉（dead code elimination），须使用 `black_box` 或 `criterion::BatchSize`
- **代码示例**（仅展示模式，非实现）：

```rust
fn construct_zeros(c: &mut Criterion) {
    let mut group = c.benchmark_group("construct_zeros");
    for &size in &[8, 256, 4096] {
        group.throughput(Throughput::Elements((size * size) as u64));
        group.bench_with_input(
            BenchmarkId::new("f64", format!("{size}x{size}")),
            &size,
            |b, &sz| {
                b.iter_batched(
                    || (),  // no setup
                    |_| black_box(zeros::<f64, Ix2>([sz, sz])),
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}
```

---

### 5.3 element_ops.rs — 逐元素运算

#### 基准组列表

| 组 ID | 操作 | 规模 | 类型 | 布局 |
|-------|------|------|------|------|
| `elem_add_f64` | `a + b` | S/M/L | f64 | F-contiguous |
| `elem_add_f32` | `a + b` | S/M/L | f32 | F-contiguous |
| `elem_add_complex` | `a + b` | M | Complex\<f64\> | F-contiguous |
| `elem_mul_f64` | `a * b` | S/M/L | f64 | F-contiguous |
| `elem_sin_f64` | `a.sin()` | S/M/L | f64 | F-contiguous |
| `elem_exp_f64` | `a.exp()` | S/M/L | f64 | F-contiguous |
| `elem_abs_f64` | `a.abs()` | S/M/L | f64 | F-contiguous |
| `elem_add_sliced` | `a + b`（其中 b 为切片视图） | M | f64 | Non-contiguous |
| `elem_add_transposed` | `a + b.t()` | M | f64 | Transposed |
| `elem_add_broadcast` | `a + scalar`（标量广播） | M | f64 | Broadcasted |
| `elem_add_inplace` | `a += b` | S/M/L | f64 | F-contiguous |
| `elem_add_ndarray` | `ndarray a + b` | S/M/L | f64 | — |
| `elem_add_rawloop` | raw for loop | S/M/L | f64 | — |

#### 设计要点

- **测量策略**：wall-clock + throughput（elements/sec + GFLOPS）
- **GFLOPS 计算**：`gflops = (2 * n) / time_ns / 1e9`（加法：1 add + 1 load per element；乘法：1 mul + 1 load per element）
- **inplace 测量**：须在 setup 中预分配目标数组，避免测量分配开销
- **非连续布局**：使用 `slice(s![.., 0..n-1])` 创建每隔一个元素的访问模式
- **广播测量**：标量 + 张量的广播加法，验证零步长路径

```rust
fn elem_add_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("elem_add_f64");
    for &size in &[8, 256, 4096] {
        let n = size * size;
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{size}x{size}")),
            &size,
            |b, &sz| {
                b.iter_batched(
                    || {
                        let a = zeros::<f64, Ix2>([sz, sz]);
                        let b = zeros::<f64, Ix2>([sz, sz]);
                        (a, b)
                    },
                    |(a, b)| black_box(&a + &b),
                    BatchSize::LargeInput,
                )
            },
        );
    }
    group.finish();
}
```

---

### 5.4 matrix_ops.rs — 矩阵运算

#### 基准组列表

| 组 ID | 操作 | 规模 (M×N) | 类型 | 说明 |
|-------|------|------------|------|------|
| `matvec_f64` | `matvec(&A, &x)` | 8×8, 256×256, 4096×4096 | f64 | 矩阵-向量乘法 |
| `matvec_f32` | `matvec(&A, &x)` | 256×256 | f32 | f32 matvec |
| `dot_f64` | `a.dot(&b)` | 64, 65K, 16M | f64 | 内积 |
| `outer_f64` | `a.outer(&b)` | 64, 256 | f64 | 外积 |
| `batch_matvec_f64` | `batch_matvec(&A, &x)` | 32×256×256 | f64 | 批量 matvec |
| `matvec_c_order` | `matvec(&A, &x)` A 为 C-order | 256×256 | f64 | C-order matvec 开销 |
| `matvec_sliced` | `matvec(&A.slice(..), &x)` | 256×256 | f64 | 切片视图 matvec |
| `matvec_ndarray` | `ndarray::linalg::mat_vec_mul` | 256×256 | f64 | 对比基准 |

#### 设计要点

- **GFLOPS 计算**：`gflops = (2 * M * N) / time_ns / 1e9`（matvec: M×N 乘加）
- **批量运算**：验证 batch 轴迭代开销是否线性
- **布局影响**：C-order 矩阵的 matvec 须转置访问，测量缓存惩罚
- **与 ndarray 对比**：ndarray 的 `mat_vec_mul` 使用相同输入规模

```rust
fn matvec_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("matvec_f64");
    for &(m, n) in &[(8, 8), (256, 256), (4096, 4096)] {
        let flops = 2 * m * n;
        group.throughput(Throughput::Elements(flops as u64));
        group.bench_with_input(
            BenchmarkId::new("f64", format!("{m}x{n}")),
            &(m, n),
            |b, &(m, n)| {
                b.iter_batched(
                    || {
                        let a = zeros::<f64, Ix2>([m, n]);
                        let x = zeros::<f64, Ix1>([n]);
                        (a, x)
                    },
                    |(a, x)| black_box(matvec(&a, &x)),
                    BatchSize::LargeInput,
                )
            },
        );
    }
    group.finish();
}
```

---

### 5.5 reduction.rs — 归约操作

#### 基准组列表

| 组 ID | 操作 | 规模 | 类型 | 说明 |
|-------|------|------|------|------|
| `sum_global_f64` | `a.sum()` | S/M/L | f64 | 全局求和 |
| `sum_axis_f64` | `a.sum_axis(0)` | M (256×256) | f64 | 沿轴求和 |
| `prod_global_f64` | `a.prod()` | M | f64 | 全局求积 |
| `min_global_f64` | `a.min()` | S/M/L | f64 | 全局最小值 |
| `mean_global_f64` | `a.mean()` | S/M/L | f64 | 全局均值 |
| `var_global_f64` | `a.var()` | S/M/L | f64 | 全局方差（Welford 算法） |
| `argmin_f64` | `a.argmin()` | S/M/L | f64 | 全局 argmin |
| `cumsum_f64` | `a.cumsum(axis)` | M | f64 | 累积求和 |
| `sum_sliced_f64` | `a.slice(..).sum()` | M | f64 | 非连续归约惩罚 |
| `sum_ndarray` | `ndarray a.sum()` | S/M/L | f64 | 对比基准 |

#### 设计要点

- **吞吐量度量**：elements/sec（而非 GFLOPS，因为归约的计算密度低）
- **var/std 重点**：Welford 算法对数值稳定性的影响 + 性能开销
- **沿轴 vs 全局**：沿轴归约需要分配输出数组 + 按轴遍历，开销不同于全局归约
- **非连续惩罚**：sliced 视图的归约须使用标量路径，测量性能倍率
- **cumsum**：前缀扫描不是 SIMD 友好的，测量串行性能

---

### 5.6 shape_ops.rs — 形状操作

#### 基准组列表

| 组 ID | 操作 | 规模 | 说明 |
|-------|------|------|------|
| `reshape_contiguous` | `a.reshape(new_shape)` | M | 连续数组 reshape（应 O(1)） |
| `reshape_non_contiguous` | `a.slice(..).reshape(..)` | M | 非连续 reshape（需拷贝） |
| `transpose_2d` | `a.t()` | M | 2D 转置视图（零拷贝） |
| `transpose_3d` | `a.permute([2, 1, 0])` | M (64×32×32) | 3D 转置 |
| `slice_view` | `a.slice(s![0..n/2, ..])` | M | 切片视图创建（O(1)） |
| `squeeze` | `a.squeeze()` | M | 去除长度 1 的维度 |
| `broadcast_view` | `a.broadcast(new_shape)` | M | 广播视图创建 |
| `to_f_contiguous` | `a.to_f_contiguous()` | M | 强制连续化（含拷贝） |

#### 设计要点

- **核心验证**：reshape/transpose/slice 理论上应为 O(1)（仅元数据操作），benchmark 须证实这一点
- **非连续 reshape**：若输入非连续，reshape 须拷贝数据，测量拷贝开销
- **to_f_contiguous**：这涉及完整数据拷贝 + 重排，属于"昂贵的形状操作"
- **throughput 度量**：用 elements/sec 而非时间，因为零拷贝操作的纳秒级时间可能不稳定

---

### 5.7 iterator.rs — 迭代器

#### 基准组列表

| 组 ID | 操作 | 规模 | 说明 |
|-------|------|------|------|
| `iter_elements` | `a.iter().for_each(\|x\| black_box(x))` | S/M/L | 元素迭代（内存布局顺序） |
| `iter_axis` | `a.axis_iter(0).for_each(\|x\| ..)` | M | 沿轴迭代 |
| `iter_windows` | `a.windows([3, 3]).for_each(\|w\| ..)` | M | 滑动窗口 |
| `iter_indexed` | `a.indexed_iter().for_each(..)` | M | 带索引迭代 |
| `zip_two` | `a.iter().zip(b.iter()).for_each(..)` | M | 双数组 zip |
| `zip_three` | `Zip::new(&a, &b, &c).for_each(..)` | M | 三数组 zip |
| `iter_sliced` | `a.slice(..).iter().for_each(..)` | M | 非连续迭代 |
| `iter_fold` | `a.iter().fold(0.0, \|acc, &x\| acc + x)` | M | fold 模式 |

#### 设计要点

- **对比目标**：迭代器 vs 原始 for 循环 — 验证零开销抽象
- **zip 测量**：验证 Zip 迭代器的编译优化质量
- **窗口迭代**：滑动窗口的缓存局部性是关键瓶颈

---

### 5.8 simd.rs — SIMD 专项

> **前提**：此文件受 `required-features = ["simd"]` 保护。

#### 基准组列表

| 组 ID | 操作 | 规模 | 对比目标 |
|-------|------|------|----------|
| `simd_add_f64` | `a + b` (SIMD path) | M/L | vs scalar path |
| `simd_add_f32` | `a + b` (SIMD path) | M/L | 验证 f32 向量宽度优势 |
| `simd_mul_f64` | `a * b` (SIMD path) | M/L | vs scalar path |
| `simd_sum_f64` | `a.sum()` (SIMD path) | M/L | vs scalar sum |
| `simd_sin_f64` | `a.sin()` (SIMD path) | M/L | 超越函数 SIMD |
| `simd_aligned_vs_unaligned` | aligned vs unaligned load/store | M | 对齐对性能的影响 |
| `simd_tail_handling` | 非向量宽度整数倍的长度 | 63/65/129 | 尾部标量处理开销 |
| `simd_complex_add` | Complex\<f64\> add | M | 复数 SIMD 路径 |
| `simd_dot_f64` | `a.dot(&b)` (SIMD path) | M/L | 内积 SIMD |
| `simd_vs_ndarray` | 同操作 vs ndarray | M/L | SIMD 优势验证 |

#### 设计要点

- **对比方式**：在基准内部，分别调用标量路径和 SIMD 路径的同一操作，对比 throughput
- **对齐测试**：构造 64 字节对齐和非对齐的数组，测量加载/存储差异
- **尾部处理**：选择非 SIMD 宽度整数倍的长度（如 f64×AVX2=4，测试 63/65/129 元素），验证尾部处理无显著回退
- **GFLOPS 报告**：SIMD 基准须报告 GFLOPS，便于与理论峰值对比

```rust
fn simd_vs_scalar_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_vs_scalar_add");
    let size = 256;
    let n = size * size;
    group.throughput(Throughput::Elements(n as u64));

    // SIMD path (compiled with --features simd)
    group.bench_function("simd_f64", |b| {
        b.iter_batched(
            || (zeros::<f64, Ix2>([size, size]), zeros::<f64, Ix2>([size, size])),
            |(a, b)| black_box(&a + &b),
            BatchSize::LargeInput,
        )
    });

    // Scalar path (force scalar via non-contiguous input)
    group.bench_function("scalar_f64", |b| {
        b.iter_batched(
            || {
                let a = zeros::<f64, Ix2>([size, size]);
                let a = a.slice(s![.., ..]).to_owned(); // force non-optimal path
                let b = zeros::<f64, Ix2>([size, size]);
                (a, b)
            },
            |(a, b)| black_box(&a + &b),
            BatchSize::LargeInput,
        )
    });

    group.finish();
}
```

---

### 5.9 parallel.rs — 并行专项

> **前提**：此文件受 `required-features = ["parallel"]` 保护。

#### 基准组列表

| 组 ID | 操作 | 规模 | 说明 |
|-------|------|------|------|
| `parallel_add_f64` | `a + b` (parallel) | M/L/XL | 并行逐元素加法 |
| `parallel_sum_f64` | `a.sum()` (parallel) | M/L/XL | 并行归约 |
| `parallel_map_f64` | `a.mapv(\|x\| x * 2.0)` | M/L/XL | 并行 map |
| `parallel_zip_f64` | `Zip::new(&a, &b).for_each(..)` | M/L/XL | 并行 zip |
| `threshold_scan` | `a + b` at various sizes | 1K..1M | 搜索并行 crossover 点 |
| `chunk_size_impact` | `a.sum()` with varying chunks | L | 分块大小对性能的影响 |
| `nested_parallel_overhead` | 并行循环内嵌套并行操作 | M | 验证嵌套并行禁止机制 |
| `thread_scaling` | `a.sum()` with 1/2/4/8 threads | XL | 线程扩展性 |

#### 设计要点

- **阈值扫描**：使用对数刻度（1K, 4K, 16K, 64K, 256K, 1M），找出并行 overhead 抵消并行收益的 crossover 点，验证 64K 阈值的合理性
- **线程扩展性**：通过 `rayon::ThreadPoolBuilder` 设置不同线程数，测量 speedup 曲线
- **分块影响**：测试 1K/4K/16K/64K 分块大小，验证 4K 最小分块的合理性
- **speedup 报告**：并行基准须报告 speedup 倍率（vs 串行基线）

```rust
fn threshold_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("threshold_scan");
    for &n in &[1_000, 4_000, 16_000, 65_536, 256_000, 1_000_000] {
        let side = (n as f64).sqrt() as usize;
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new("add", n),
            &side,
            |b, &sz| {
                b.iter_batched(
                    || (zeros::<f64, Ix2>([sz, sz]), zeros::<f64, Ix2>([sz, sz])),
                    |(a, b)| black_box(&a + &b),
                    BatchSize::LargeInput,
                )
            },
        );
    }
    group.finish();
}
```

---

### 5.10 ffi.rs — FFI 基准

#### 基准组列表

| 组 ID | 操作 | 规模 | 说明 |
|-------|------|------|------|
| `ffi_as_ptr` | `tensor.as_ptr()` | M | 指针获取开销（应 O(1)） |
| `ffi_index_to_ptr` | `tensor.index_to_ptr([i, j])` | M | 多维索引→指针转换 |
| `ffi_blas_check` | `tensor.is_blas_compatible()` | M | BLAS 兼容性检查 |
| `ffi_lda` | `tensor.lda()` | M | leading dimension 查询 |
| `ffi_raw_parts_roundtrip` | `tensor.into_raw_parts()` → `from_raw_parts()` | M | 原始部件解构+重构 |

#### 设计要点

- **FFI 微基准**：测量指针 API 的调用开销，确保纳秒级
- **BLAS 检查**：`is_blas_compatible()` 涉及 LayoutFlags 检查，须验证 O(1)
- **roundtrip**：`into_raw_parts` → `from_raw_parts` 的开销，验证零拷贝 FFI 集成的可行性

---

## 6. 测量策略

### 6.1 Wall-clock 时间（criterion 默认）

criterion 使用迭代采样 + 线性回归估算每次迭代的平均时间。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| sample_size | 100 | 采样次数 |
| measurement_time | 5s | 每个基准的测量时间 |
| warm_up_time | 3s | 预热时间 |

**调整策略：**

| 场景 | 调整 | 原因 |
|------|------|------|
| 极快操作（< 100ns） | 增大 `sample_size` 至 1000 | 减少测量噪声 |
| 极慢操作（> 10s） | 减少 `measurement_time` 至 2s | 避免 CI 超时 |
| 不稳定操作 | 增大 `warm_up_time` 至 5s | 确保 CPU 频率稳定 |

### 6.2 Throughput 度量

每个基准组须声明吞吐量类型，便于横向对比：

| 操作类型 | Throughput 类型 | 计算 |
|----------|-----------------|------|
| 逐元素运算 | `Throughput::Elements(n)` | 输出元素数 |
| 矩阵运算 | `Throughput::Elements(flops)` | 2×M×N（matvec） |
| 归约 | `Throughput::Elements(n)` | 输入元素数 |
| 构造 | `Throughput::Elements(n)` | 输出元素数 |
| 形状操作 | `Throughput::Elements(n)` | 涉及的元素数 |

### 6.3 GFLOPS 计算（矩阵/运算类基准）

在 HTML 报告中附加 GFLOPS 信息：

```rust
group.throughput(Throughput::Elements(flops as u64));
// GFLOPS = reported throughput (elements/sec) / 1e9
```

### 6.4 内存分配跟踪（可选，Phase 2 增强）

使用 criterion 的 `Mem profiler` 或手动统计：

```rust
#[cfg(feature = "std")]
fn measure_allocations() {
    use std::alloc::GlobalAlloc;
    // Wrap global allocator to count allocations
    // Report in benchmark output as "allocations per iteration"
}
```

**Phase 1 不实现内存跟踪**，留作后续增强。原因：需要自定义全局分配器，与 criterion 集成较复杂，且 wall-clock + throughput 已覆盖主要需求。

### 6.5 Cache 行为（可选，Phase 3 增强）

通过 `perf` 采集 cache miss 数据：

```bash
# Manual perf profiling
perf stat -e cache-misses,cache-references,cycles,instructions \
    cargo bench --bench element_ops -- elem_add_f64
```

**Phase 1 不集成到 criterion**，仅作为手动分析工具。原因：perf 需要 root 权限或 `perf_event_paranoid` 配置，不适合 CI 环境。

---

## 7. CI 集成

### 7.1 CI 工作流设计

#### Smoke Test（每次 PR）

```yaml
# .github/workflows/bench-smoke.yml
name: Bench Smoke Test
on: [pull_request]
jobs:
  smoke:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - name: Run smoke benchmarks
        run: |
          cargo bench --bench tensor_construct -- --quick
          cargo bench --bench element_ops -- --quick
          cargo bench --bench reduction -- --quick
        env:
          RUSTFLAGS: "-C target-cpu=native"
```

**`--quick` 标志**：减少采样次数，smoke test 在 5 分钟内完成。

#### Full Benchmark（每周/每版本）

```yaml
# .github/workflows/bench-full.yml
name: Full Benchmark
on:
  schedule:
    - cron: '0 2 * * 1'  # Weekly Monday 2AM UTC
  workflow_dispatch:       # Manual trigger
jobs:
  bench:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        features:
          - "std"                    # scalar only
          - "std,simd"              # + SIMD
          - "std,parallel"          # + parallel
          - "std,simd,parallel"     # full
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - name: Run full benchmarks
        run: cargo bench --features "${{ matrix.features }}"
        env:
          RUSTFLAGS: "-C target-cpu=native"
      - name: Upload criterion reports
        uses: actions/upload-artifact@v4
        with:
          name: bench-report-${{ matrix.features }}
          path: target/criterion/
```

#### Regression Detection（每次 PR）

```yaml
# .github/workflows/bench-regression.yml
name: Benchmark Regression Check
on: [pull_request]
jobs:
  regression:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - name: Cache criterion baseline
        uses: actions/cache@v4
        with:
          path: target/criterion
          key: criterion-${{ runner.os }}-${{ hashFiles('src/**') }}
      - name: Run regression check
        run: |
          cargo bench --bench element_ops -- elem_add_f64 2>&1 | tee bench_output.txt
          # Check for regressions in output
          grep -q "Performance has regressed" bench_output.txt && exit 1 || true
```

### 7.2 回归阈值

| 级别 | 阈值 | 动作 |
|------|------|------|
| **警告** | > 5% 变慢 | CI warning（不阻塞合并） |
| **失败** | > 20% 变慢 | CI failure（阻塞合并，须人工确认） |
| **改善** | > 5% 变快 | CI note（记录性能改善） |

criterion 的默认回归检测使用 Bootstrap 置信区间分析，可通过以下方式配置阈值：

```rust
// In benchmark code, configure regression threshold
let mut group = c.benchmark_group("elem_add");
group.significance_level(0.05);      // 5% significance
group.sample_size(100);
// criterion automatically detects regressions relative to saved baseline
```

### 7.3 结果存储与趋势可视化

#### 本地存储

criterion 自动将结果保存到 `target/criterion/` 目录：

```
target/criterion/
├── elem_add_f64/
│   ├── 8x8/
│   │   ├── new/          # 本次运行
│   │   ├── baseline/     # 上次运行（用于回归检测）
│   │   └── change/       # 差异统计
│   ├── 256x256/
│   └── 4096x4096/
└── ...
```

#### 趋势可视化方案

| 方案 | 复杂度 | 推荐阶段 |
|------|--------|----------|
| criterion 内置 HTML 报告 | 低（开箱即用） | Phase 1 |
| GitHub Pages 托管报告 | 中 | Phase 2 |
| custom.co + criterion 导出 | 高 | Phase 3（可选） |

**Phase 1 采用 criterion 内置 HTML 报告**，后续根据需要增强。

### 7.4 CI 运行时间预算

| 工作流 | 基准数量 | 预计时间 | 频率 |
|--------|----------|----------|------|
| Smoke Test | 3 个文件 × `--quick` | ~5 min | 每次 PR |
| Regression Check | 1-2 个热点基准 | ~10 min | 每次 PR |
| Full Benchmark | 全部 9 文件 × 4 feature 组合 | ~60 min | 每周 |

---

## 8. 公共辅助基础设施

各 bench 文件共享的辅助功能集中在一个模块中，避免重复代码。

### 8.1 辅助模块位置

```
benches/
├── common/                # 共享辅助代码
│   ├── mod.rs             # pub mod 所有子模块
│   ├── sizes.rs           # 标准规模常量
│   ├── inputs.rs          # 测试数据生成
│   └── metrics.rs         # GFLOPS 计算等辅助函数
├── tensor_construct.rs
├── element_ops.rs
└── ...
```

### 8.2 sizes.rs — 标准规模定义

```rust
// benches/common/sizes.rs

/// Standard 2D sizes for benchmark parameterization.
pub const SMALL_2D: usize = 8;
pub const MEDIUM_2D: usize = 256;
pub const LARGE_2D: usize = 4096;

/// Standard 1D sizes.
pub const SMALL_1D: usize = 64;
pub const MEDIUM_1D: usize = 65_536;
pub const LARGE_1D: usize = 16_777_216;

/// All 2D sizes as a slice for iteration.
pub const SIZES_2D: &[usize] = &[SMALL_2D, MEDIUM_2D, LARGE_2D];

/// All 1D sizes as a slice for iteration.
pub const SIZES_1D: &[usize] = &[SMALL_1D, MEDIUM_1D, LARGE_1D];
```

### 8.3 inputs.rs — 测试数据生成

```rust
// benches/common/inputs.rs

use Renon::{Tensor, Ix2, Ix1, zeros};
use rand::Rng;

/// Create a 2D tensor filled with random f64 values in [0.0, 1.0).
pub fn random_tensor_2d(rows: usize, cols: usize) -> Tensor<f64, Ix2> {
    let mut rng = rand::thread_rng();
    let data: Vec<f64> = (0..rows * cols).map(|_| rng.gen()).collect();
    Tensor::from_vec(data, [rows, cols]).unwrap()
}

/// Create a 1D tensor filled with random f64 values in [0.0, 1.0).
pub fn random_tensor_1d(n: usize) -> Tensor<f64, Ix1> {
    let mut rng = rand::thread_rng();
    let data: Vec<f64> = (0..n).map(|_| rng.gen()).collect();
    Tensor::from_vec(data, [n]).unwrap()
}
```

### 8.4 metrics.rs — GFLOPS 辅助

```rust
// benches/common/metrics.rs

/// Calculate GFLOPS from operation count and elapsed nanoseconds.
pub fn gflops(flops: u64, time_ns: f64) -> f64 {
    flops as f64 / time_ns / 1e9
}
```

---

## 9. 实现任务拆分

以下任务每个约 10 分钟，按依赖顺序排列。

### Phase A: 基础设施（4 个任务）

- [ ] **A1**: 创建 `benches/common/mod.rs`, `sizes.rs`, `inputs.rs`, `metrics.rs` 辅助模块
  - 文件: `benches/common/*.rs`
  - 内容: 规模常量 + 随机数据生成 + GFLOPS 计算
  - 预计: 10 min

- [ ] **A2**: 在 `Cargo.toml` 添加 `criterion`, `ndarray`, `rand`, `approx` dev-dependency + 9 个 `[[bench]]` 声明
  - 文件: `Cargo.toml`
  - 内容: criterion 0.5 + html_reports, ndarray 0.16, 各 bench 入口声明
  - 预计: 10 min

- [ ] **A3**: 创建 `benches/tensor_construct.rs` 骨架 + criterion 入口 + `construct_zeros` 组
  - 文件: `benches/tensor_construct.rs`
  - 内容: `criterion_group!`, `criterion_main!`, zeros S/M/L 三组
  - 预计: 10 min

- [ ] **A4**: 完善 `benches/tensor_construct.rs`：ones, from_vec, eye, arange, linspace, zeros_dyn, ndarray 对比
  - 文件: `benches/tensor_construct.rs`
  - 内容: 补充剩余 8 个基准组
  - 预计: 10 min

### Phase B: 核心运算基准（5 个任务）

- [ ] **B1**: 创建 `benches/element_ops.rs`：add f64 (S/M/L) + add f32 + add complex + ndarray 对比
  - 文件: `benches/element_ops.rs`
  - 内容: `elem_add_f64`, `elem_add_f32`, `elem_add_complex`, `elem_add_ndarray`, `elem_add_rawloop`
  - 预计: 10 min

- [ ] **B2**: 完善 `benches/element_ops.rs`：mul, sin, exp, abs + inplace + sliced/transposed/broadcast
  - 文件: `benches/element_ops.rs`
  - 内容: `elem_mul_f64`, `elem_sin_f64`, `elem_exp_f64`, `elem_abs_f64`, `elem_add_inplace`, `elem_add_sliced`, `elem_add_transposed`, `elem_add_broadcast`
  - 预计: 10 min

- [ ] **B3**: 创建 `benches/matrix_ops.rs`：matvec f64 (S/M/L) + matvec f32 + dot + outer
  - 文件: `benches/matrix_ops.rs`
  - 内容: `matvec_f64`, `matvec_f32`, `dot_f64`, `outer_f64`, GFLOPS 报告
  - 预计: 10 min

- [ ] **B4**: 完善 `benches/matrix_ops.rs`：batch_matvec + C-order + sliced + ndarray 对比
  - 文件: `benches/matrix_ops.rs`
  - 内容: `batch_matvec_f64`, `matvec_c_order`, `matvec_sliced`, `matvec_ndarray`
  - 预计: 10 min

- [ ] **B5**: 创建 `benches/reduction.rs`：sum/prod/min/mean/var 全局 + 沿轴 + cumsum + ndarray 对比
  - 文件: `benches/reduction.rs`
  - 内容: `sum_global_f64`, `sum_axis_f64`, `prod_global_f64`, `min_global_f64`, `mean_global_f64`, `var_global_f64`, `argmin_f64`, `cumsum_f64`, `sum_sliced_f64`, `sum_ndarray`
  - 预计: 10 min

### Phase C: 形状/迭代/FFI 基准（3 个任务）

- [ ] **C1**: 创建 `benches/shape_ops.rs`：reshape, transpose, slice, squeeze, broadcast, to_f_contiguous
  - 文件: `benches/shape_ops.rs`
  - 内容: 全部 8 个基准组
  - 预计: 10 min

- [ ] **C2**: 创建 `benches/iterator.rs`：iter, axis_iter, windows, indexed, zip (2/3), sliced, fold
  - 文件: `benches/iterator.rs`
  - 内容: 全部 8 个基准组
  - 预计: 10 min

- [ ] **C3**: 创建 `benches/ffi.rs`：as_ptr, index_to_ptr, is_blas_compatible, lda, raw_parts roundtrip
  - 文件: `benches/ffi.rs`
  - 内容: 全部 5 个基准组
  - 预计: 10 min

### Phase D: 后端专项基准（2 个任务）

- [ ] **D1**: 创建 `benches/simd.rs`：SIMD vs scalar add/mul/sum/sin + aligned vs unaligned + tail + dot + complex + ndarray
  - 文件: `benches/simd.rs`
  - 内容: 全部 10 个基准组，受 `required-features = ["simd"]` 保护
  - 预计: 10 min

- [ ] **D2**: 创建 `benches/parallel.rs`：parallel add/sum/map/zip + threshold scan + chunk size + thread scaling
  - 文件: `benches/parallel.rs`
  - 内容: 全部 8 个基准组，受 `required-features = ["parallel"]` 保护
  - 预计: 10 min

### Phase E: CI 集成（3 个任务）

- [ ] **E1**: 创建 `.github/workflows/bench-smoke.yml`：PR 触发，3 个核心 bench 文件 `--quick` 模式
  - 文件: `.github/workflows/bench-smoke.yml`
  - 内容: ubuntu-latest, rust stable, cargo bench --quick, 5 min timeout
  - 预计: 10 min

- [ ] **E2**: 创建 `.github/workflows/bench-full.yml`：每周定时 + 手动触发，4 个 feature 组合 × 全部 bench 文件
  - 文件: `.github/workflows/bench-full.yml`
  - 内容: matrix strategy, artifact upload, criterion HTML 报告
  - 预计: 10 min

- [ ] **E3**: 创建 `.github/workflows/bench-regression.yml`：PR 触发，criterion baseline 对比 + 5%/20% 阈值
  - 文件: `.github/workflows/bench-regression.yml`
  - 内容: cache baseline, regression check, threshold alert
  - 预计: 10 min

---

## 10. 附录

### 10.1 Benchmark 运行命令速查

```bash
# Run all benchmarks (default features)
cargo bench

# Run a specific benchmark file
cargo bench --bench element_ops

# Run a specific benchmark group
cargo bench --bench element_ops -- elem_add_f64

# Run with SIMD feature
cargo bench --features simd

# Run with full features (SIMD + parallel)
cargo bench --features full

# Quick mode for smoke testing
cargo bench -- --quick

# Save baseline for regression detection
cargo bench -- --save-baseline main

# Compare against saved baseline
cargo bench -- --baseline main

# Generate HTML reports (default behavior)
# Reports saved to target/criterion/report/index.html
```

### 10.2 预期性能参考值

以下为参考目标值（基于典型 x86_64 平台，AVX2），实际值取决于硬件和编译器优化：

| 操作 | 规模 | 标量基线 | SIMD 加速倍率 | 并行加速倍率 |
|------|------|----------|---------------|-------------|
| add (f64) | 256×256 | ~0.05 ms | 2-4× | 2-8×（4 核） |
| mul (f64) | 256×256 | ~0.05 ms | 2-4× | 2-8× |
| sum (f64) | 256×256 | ~0.02 ms | 2-4× | 2-8× |
| matvec (f64) | 256×256 | ~0.1 ms | N/A | N/A |
| sin (f64) | 256×256 | ~0.5 ms | 1.5-3× | 2-8× |
| reshape | 256×256 | < 10 ns | N/A | N/A |
| transpose (view) | 256×256 | < 10 ns | N/A | N/A |

> **注意**：这些值仅作为数量级参考，不作为回归阈值依据。回归阈值基于 criterion 统计分析自动确定。

### 10.3 与 ndarray 对比的注意事项

| 注意点 | 说明 |
|--------|------|
| 版本锁定 | ndarray 版本须锁定在 `Cargo.toml` dev-dependency 中，避免版本差异引入噪声 |
| 相同输入 | 对比基准须使用相同的随机种子，确保输入数据一致 |
| 相同规模 | 矩阵维度须完全一致（包括内存布局：ndarray 默认 C-order，Renon 默认 F-order） |
| 编译优化 | 两者须使用相同的 `RUSTFLAGS`（特别是 `target-cpu=native`） |
| 结果验证 | 对比基准须验证结果近似相等（`approx::assert_relative_eq`），确保测量的是正确操作 |
