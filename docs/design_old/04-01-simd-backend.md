# Senon SIMD 后端模块设计文档

> **模块路径**: `src/simd/`  
> **版本**: v1.0  
> **日期**: 2026-03-28  
> **Feature gate**: `simd` (默认关闭)  
> **依赖**: `pulp` crate (可选)

---

## 1. 模块概述

### 1.1 核心角色

SIMD 后端模块是 Senon 张量库的性能加速层，负责通过向量化指令提升数值计算性能。核心职责包括：

| 职责 | 说明 |
|------|------|
| **指令集抽象** | 通过 pulp crate 统一抽象多种 SIMD 指令集 |
| **运行时分发** | 根据当前 CPU 能力自动选择最优指令集实现 |
| **标量回退** | 提供无条件可用的标量实现作为基准和回退 |
| **对齐感知** | 根据数据对齐状态选择对齐/非对齐加载路径 |
| **尾部处理** | 处理非 SIMD 宽度整数倍的元素尾部 |

### 1.2 在性能分层中的角色

SIMD 后端位于性能分层的中间层级：

```
性能分层架构

┌─────────────────────────────────────────────────────────┐
│                    调用层 (ops/iter)                     │
│         elementwise, reduction, matrix operations        │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                 性能分层决策 (dispatch)                  │
│   根据 元素数/连续性/对齐/feature 决定执行路径            │
└──────────┬──────────────┬──────────────┬────────────────┘
           │              │              │
           ▼              ▼              ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ SIMD路径 │   │ 并行路径 │   │ 标量路径 │
    │(本模块)  │   │ (rayon)  │   │  (回退)  │
    └────┬─────┘   └────┬─────┘   └────┬─────┘
         │              │              │
         └──────────────┴──────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │   硬件执行      │
              │ AVX-512/AVX2/   │
              │ SSE4.1/NEON     │
              └─────────────────┘
```

### 1.3 pulp 选型理由

| 考量 | pulp 方案 | 直接使用 std::arch |
|------|-----------|-------------------|
| **跨平台** | ✅ 统一抽象 x86/ARM | ❌ 需手动处理平台差异 |
| **运行时检测** | ✅ 内置 Arch::new() | ❌ 需手动调用 is_x86_feature_detected! |
| **代码简洁** | ✅ WithSimd trait 模式 | ❌ 大量 cfg 条件编译 |
| **no_std 兼容** | ✅ 支持 | ✅ 支持 |
| **维护成本** | ✅ 低（上游维护） | ❌ 高（自行维护） |
| **类型安全** | ✅ 编译期检查 | ⚠️ 依赖 unsafe 块 |

**结论**: 使用 pulp crate 可大幅降低 SIMD 实现的复杂度，同时保持性能和可移植性。

### 1.4 支持的指令集

| 指令集 | 架构 | 寄存器宽度 | f32 元素数 | f64 元素数 | 优先级 |
|--------|------|-----------|-----------|-----------|--------|
| AVX-512 | x86_64 | 512 bit | 16 | 8 | 最高 |
| AVX2 + FMA | x86_64 | 256 bit | 8 | 4 | 高 |
| SSE4.1 | x86_64 | 128 bit | 4 | 2 | 中 |
| NEON | aarch64 | 128 bit | 4 | 2 | 高 (ARM) |
| 标量回退 | 所有 | - | 1 | 1 | 最低 |

### 1.5 适用操作

| 操作类别 | 示例 | SIMD 适用性 |
|----------|------|-------------|
| 逐元素一元 | `abs`, `neg`, `sqrt`, `exp`, `ln` | ✅ 连续内存或固定步长 |
| 逐元素二元 | `add`, `sub`, `mul`, `div` | ✅ 两操作数形状兼容 |
| 归约 | `sum`, `prod`, `min`, `max` | ✅ 连续内存优先 |
| 内积 | `dot` | ✅ 连续内存 |
| 比较 | `eq`, `lt`, `gt` (生成 mask) | ✅ 连续内存 |

---

## 2. 文件结构

```
src/simd/
├── mod.rs             # pulp 集成，Arch 缓存，公开 API
├── scalar.rs          # 标量回退实现（所有操作的基准实现）
└── vector.rs          # 向量化实现（基于 pulp WithSimd）
```

### 2.1 各文件职责

| 文件 | 职责 | 核心类型/函数 |
|------|------|---------------|
| `mod.rs` | 模块入口，pulp Arch 管理，性能分层分发 | `SimdKernel`, `get_arch()`, `dispatch()` |
| `scalar.rs` | 标量回退实现，作为基准和回退 | `ScalarKernel` |
| `vector.rs` | SIMD 向量化实现 | `VectorKernel`, 各操作的 SIMD 版本 |

### 2.2 模块依赖图

```
┌─────────────────────────────────────────────────────────┐
│                     外部依赖                             │
│  ┌─────────┐                                            │
│  │  pulp   │ (可选，feature = "simd")                   │
│  └────┬────┘                                            │
└───────┼─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│                    src/simd/                             │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐         │
│  │  mod.rs  │────▶│scalar.rs │     │vector.rs │         │
│  └────┬─────┘     └──────────┘     └────┬─────┘         │
│       │                                │                │
│       │        SimdKernel trait        │                │
│       └────────────────────────────────┘                │
└───────┬─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│                   调用方 (ops/iter)                      │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐         │
│  │elementwise│    │reduction │     │ matrix   │         │
│  └──────────┘     └──────────┘     └──────────┘         │
└─────────────────────────────────────────────────────────┘
```

---

## 3. pulp 集成设计

### 3.1 pulp 核心概念

pulp 提供了三个核心抽象：

| 概念 | 类型/函数 | 说明 |
|------|----------|------|
| **Arch** | `pulp::Arch` | 运行时 CPU 能力检测，为 `Copy` 类型 |
| **WithSimd** | `pulp::WithSimd` | 闭包 trait，接收 SIMD 上下文 |
| **Simd** | `pulp::Simd` | 特定指令集的 SIMD 操作接口 |

### 3.2 Arch 缓存策略

由于 `Arch::new()` 涉及 CPUID 调用等开销，应缓存 `Arch` 实例：

```rust
// src/simd/mod.rs

#[cfg(feature = "simd")]
use pulp::Arch;

/// 全局缓存的 Arch 实例
/// 
/// Arch 为 Copy 类型，缓存后可零成本复用。
/// 使用 once_cell 或 OnceLock 保证线程安全的延迟初始化。
#[cfg(feature = "simd")]
static ARCH: once_cell::sync::Lazy<Arch> = once_cell::sync::Lazy::new(|| {
    Arch::new()
});

/// 获取全局缓存的 Arch 实例
/// 
/// # Example
/// 
/// ```ignore
/// let arch = get_arch();
/// arch.dispatch(|| { ... });
/// ```
#[cfg(feature = "simd")]
#[inline]
pub fn get_arch() -> Arch {
    *ARCH
}

/// 无 SIMD 时的占位实现
#[cfg(not(feature = "simd"))]
pub fn get_arch() -> () {
    ()
}
```

### 3.3 WithSimd Trait 使用模式

pulp 的 `WithSimd` trait 定义如下：

```rust
// pulp crate 中的定义（简化）
pub trait WithSimd {
    type Output;
    
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output;
}
```

**使用模式**：

```rust
// 定义一个实现了 WithSimd 的闭包结构体
struct AddKernel<'a, 'b, 'c> {
    lhs: &'a [f64],
    rhs: &'b [f64],
    out: &'c mut [f64],
}

impl WithSimd for AddKernel<'_, '_, '_> {
    type Output = ();
    
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        // simd 参数提供了特定指令集的操作
        // pulp 会自动选择最优的 S 实现
        let width = S::f64s_len();  // SIMD 宽度
        
        let chunks = self.lhs.len() / width;
        
        // 主循环：SIMD 处理
        for i in 0..chunks {
            let offset = i * width;
            let lhs_vec = simd.f64s_load_unaligned(self.lhs.as_ptr().add(offset));
            let rhs_vec = simd.f64s_load_unaligned(self.rhs.as_ptr().add(offset));
            let result = simd.f64s_add(lhs_vec, rhs_vec);
            simd.f64s_store_unaligned(self.out.as_mut_ptr().add(offset), result);
        }
        
        // 尾部处理：标量
        for i in (chunks * width)..self.lhs.len() {
            self.out[i] = self.lhs[i] + self.rhs[i];
        }
    }
}

// 分发调用
fn add_simd(lhs: &[f64], rhs: &[f64], out: &mut [f64]) {
    let arch = get_arch();
    arch.dispatch(AddKernel { lhs, rhs, out });
}
```

### 3.4 dispatch 流程

```
dispatch 调用流程

┌─────────────────────────────────────────────────────────┐
│               arch.dispatch(kernel)                      │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│           Arch 内部运行时检测（已缓存）                   │
│   检查顺序：AVX-512 → AVX2+FMA → SSE4.1 → NEON → 标量   │
└─────────────────────────┬───────────────────────────────┘
                          │
           ┌──────────────┼──────────────┐
           │              │              │
           ▼              ▼              ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │AVX-512   │   │AVX2+FMA  │   │SSE4.1/   │
    │Avx512    │   │Avx2      │   │NEON      │
    └────┬─────┘   └────┬─────┘   └────┬─────┘
         │              │              │
         └──────────────┴──────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│         kernel.with_simd(simd: S)                       │
│         S 为 pulp 自动选择的最优指令集类型               │
└─────────────────────────────────────────────────────────┘
```

### 3.5 no_std 下 pulp 的兼容性

pulp crate 支持 `no_std` 环境，但需要注意：

| 环境 | pulp 支持 | 说明 |
|------|----------|------|
| `std` | ✅ 完整支持 | 默认 |
| `no_std` + `alloc` | ✅ 支持 | 需启用 `libm` feature 处理数学函数 |
| `no_std` 无 `alloc` | ⚠️ 部分支持 | Arch 检测可能受限 |

**Cargo.toml 配置**：

```toml
[features]
default = ["std"]
std = []
simd = ["dep:pulp"]

[dependencies]
pulp = { version = "0.18", optional = true }

# no_std 环境下 pulp 需要 libm 处理数学函数
[target.'cfg(not(feature = "std"))'.dependencies]
libm = { version = "0.2", optional = true }

[patch.crates-io]
# 如需使用最新 pulp，可添加 patch
```

**条件编译处理**：

```rust
// src/simd/mod.rs

#[cfg(feature = "simd")]
use pulp::Arch;

#[cfg(all(feature = "simd", not(feature = "std")))]
// no_std 环境下，某些数学函数可能需要 libm
// pulp 内部会处理这些差异

/// no_std 环境下的 Arch 获取
#[cfg(all(feature = "simd", not(feature = "std")))]
pub fn get_arch() -> Arch {
    // no_std 下 Arch::new() 同样可用
    // 但可能无法检测某些特性，会回退到安全实现
    Arch::new()
}
```

---

## 4. SimdKernel Trait 设计

### 4.1 设计目标

`SimdKernel` trait 是 Senon SIMD 后端的核心抽象，目标是：

1. **统一接口**：标量和向量实现共用同一 trait
2. **类型安全**：通过泛型参数约束元素类型
3. **零成本抽象**：编译期内联，无虚调用开销
4. **可扩展**：便于添加新的操作类型

### 4.2 SimdKernel Trait 完整签名

```rust
// src/simd/mod.rs

use core::marker::PhantomData;

/// SIMD 内核元素类型标记
/// 
/// 用于在编译期区分不同元素类型的实现。
pub trait SimdElement: Copy + Clone + Send + Sync + 'static {
    /// 元素大小（字节）
    const SIZE: usize;
    
    /// 元素自然对齐
    const ALIGN: usize;
}

// 为基本类型实现 SimdElement
impl SimdElement for f32 {
    const SIZE: usize = 4;
    const ALIGN: usize = 4;
}

impl SimdElement for f64 {
    const SIZE: usize = 8;
    const ALIGN: usize = 8;
}

impl SimdElement for i32 {
    const SIZE: usize = 4;
    const ALIGN: usize = 4;
}

impl SimdElement for i64 {
    const SIZE: usize = 8;
    const ALIGN: usize = 8;
}

/// SIMD 操作类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdOp {
    /// 逐元素一元操作
    Unary,
    /// 逐元素二元操作
    Binary,
    /// 归约操作
    Reduction,
    /// 内积操作
    Dot,
    /// 比较操作
    Comparison,
}

/// SIMD 内核 trait
/// 
/// 定义了所有 SIMD 操作的统一接口。标量回退和向量化实现
/// 都实现此 trait，调用方通过泛型或 trait object 使用。
/// 
/// # Type Parameters
/// 
/// * `A` - 元素类型，须实现 `SimdElement`
/// 
/// # Example
/// 
/// ```ignore
/// use Senon::simd::{SimdKernel, ScalarKernel};
/// 
/// let kernel = ScalarKernel::<f64>::new();
/// 
/// // 逐元素加法
/// kernel.add(lhs, rhs, out);
/// 
/// // 归约求和
/// let sum = kernel.sum(data);
/// ```
pub trait SimdKernel<A: SimdElement>: Send + Sync {
    // ========================================
    // 元信息
    // ========================================
    
    /// 内核名称（用于调试和日志）
    fn name() -> &'static str where Self: Sized;
    
    /// 返回此内核的 SIMD 宽度（元素数）
    /// 
    /// 标量内核返回 1，向量化内核返回实际的 SIMD 宽度。
    fn width() -> usize where Self: Sized;
    
    /// 检查此内核是否支持指定操作
    fn supports_op(op: SimdOp) -> bool where Self: Sized;
    
    // ========================================
    // 逐元素一元操作
    // ========================================
    
    /// 逐元素取负
    /// 
    /// # Arguments
    /// * `src` - 输入数据
    /// * `dst` - 输出数据
    /// 
    /// # Panics
    /// 
    /// 如果 `src.len() != dst.len()`，将 panic。
    fn neg(&self, src: &[A], dst: &mut [A]);
    
    /// 逐元素绝对值
    fn abs(&self, src: &[A], dst: &mut [A]);
    
    /// 逐元素平方根
    fn sqrt(&self, src: &[A], dst: &mut [A]);
    
    /// 逐元素平方
    fn square(&self, src: &[A], dst: &mut [A]);
    
    /// 逐元素倒数
    fn recip(&self, src: &[A], dst: &mut [A]);
    
    // ========================================
    // 逐元素二元操作
    // ========================================
    
    /// 逐元素加法
    /// 
    /// # Arguments
    /// * `lhs` - 左操作数
    /// * `rhs` - 右操作数
    /// * `dst` - 输出数据
    /// 
    /// # Panics
    /// 
    /// 如果三者长度不相等，将 panic。
    fn add(&self, lhs: &[A], rhs: &[A], dst: &mut [A]);
    
    /// 逐元素减法
    fn sub(&self, lhs: &[A], rhs: &[A], dst: &mut [A]);
    
    /// 逐元素乘法
    fn mul(&self, lhs: &[A], rhs: &[A], dst: &mut [A]);
    
    /// 逐元素除法
    fn div(&self, lhs: &[A], rhs: &[A], dst: &mut [A]);
    
    /// 逐元素取小值
    fn min(&self, lhs: &[A], rhs: &[A], dst: &mut [A]);
    
    /// 逐元素取大值
    fn max(&self, lhs: &[A], rhs: &[A], dst: &mut [A]);
    
    // ========================================
    // 归约操作
    // ========================================
    
    /// 求和归约
    /// 
    /// # Returns
    /// 
    /// 所有元素的和。
    fn sum(&self, data: &[A]) -> A;
    
    /// 求积归约
    fn prod(&self, data: &[A]) -> A;
    
    /// 最小值归约
    fn reduce_min(&self, data: &[A]) -> A;
    
    /// 最大值归约
    fn reduce_max(&self, data: &[A]) -> A;
    
    // ========================================
    // 内积操作
    // ========================================
    
    /// 向量内积
    /// 
    /// # Arguments
    /// * `lhs` - 左向量
    /// * `rhs` - 右向量
    /// 
    /// # Returns
    /// 
    /// `sum(lhs[i] * rhs[i])`
    /// 
    /// # Panics
    /// 
    /// 如果两者长度不相等，将 panic。
    fn dot(&self, lhs: &[A], rhs: &[A]) -> A;
    
    // ========================================
    // 比较操作
    // ========================================
    
    /// 逐元素相等比较
    /// 
    /// # Arguments
    /// * `lhs` - 左操作数
    /// * `rhs` - 右操作数
    /// * `dst` - 输出布尔掩码
    fn eq(&self, lhs: &[A], rhs: &[A], dst: &mut [bool]);
    
    /// 逐元素小于比较
    fn lt(&self, lhs: &[A], rhs: &[A], dst: &mut [bool]);
    
    /// 逐元素大于比较
    fn gt(&self, lhs: &[A], rhs: &[A], dst: &mut [bool]);
    
    // ========================================
    // 填充操作
    // ========================================
    
    /// 用指定值填充
    fn fill(&self, dst: &mut [A], value: A);
    
    /// 用零填充
    fn fill_zero(&self, dst: &mut [A]) where A: Default;
}

/// 类型化的 SIMD 内核（带元素类型参数）
pub struct TypedSimdKernel<A: SimdElement, K: SimdKernel<A>> {
    kernel: K,
    _marker: PhantomData<A>,
}

impl<A: SimdElement, K: SimdKernel<A>> TypedSimdKernel<A, K> {
    pub fn new(kernel: K) -> Self {
        Self {
            kernel,
            _marker: PhantomData,
        }
    }
    
    pub fn kernel(&self) -> &K {
        &self.kernel
    }
}
```

### 4.3 与 pulp WithSimd 的集成

Senon 的 `SimdKernel` 与 pulp 的 `WithSimd` 通过适配器模式集成：

```rust
// src/simd/vector.rs

#[cfg(feature = "simd")]
use pulp::{Arch, Simd, WithSimd};

/// pulp WithSimd 适配器
/// 
/// 将 Senon 的操作转换为 pulp 的 WithSimd 闭包。
/// pulp 负责选择最优指令集，适配器负责具体操作实现。
#[cfg(feature = "simd")]
pub struct PulpAdapter<'a, A, F>
where
    A: SimdElement,
    F: Fn(&[A], &mut [A]) + Send + Sync,
{
    /// 输入数据
    pub input: &'a [A],
    /// 输出数据
    pub output: &'a mut [A],
    /// 标量操作函数（用于尾部处理）
    pub scalar_fn: F,
}

#[cfg(feature = "simd")]
impl<A, F> WithSimd for PulpAdapter<'_, A, F>
where
    A: SimdElement,
    F: Fn(&[A], &mut [A]) + Send + Sync,
{
    type Output = ();
    
    fn with_simd<S: Simd>(self, _simd: S) -> Self::Output {
        // 具体实现根据元素类型和操作类型分发
        // 这里使用泛型 S，pulp 会自动选择最优实现
        todo!("具体操作实现")
    }
}

/// 使用 pulp 分发操作
#[cfg(feature = "simd")]
pub fn dispatch_with_pulp<A, F>(input: &[A], output: &mut [A], scalar_fn: F)
where
    A: SimdElement,
    F: Fn(&[A], &mut [A]) + Send + Sync,
{
    let arch = crate::simd::get_arch();
    arch.dispatch(PulpAdapter {
        input,
        output,
        scalar_fn,
    });
}
```

### 4.4 具体操作的 WithSimd 实现

以 `f64` 加法为例：

```rust
// src/simd/vector.rs

#[cfg(feature = "simd")]
use pulp::{Simd, WithSimd};

/// f64 加法内核
#[cfg(feature = "simd")]
pub struct AddF64Kernel<'a> {
    pub lhs: &'a [f64],
    pub rhs: &'a [f64],
    pub dst: &'a mut [f64],
}

#[cfg(feature = "simd")]
impl WithSimd for AddF64Kernel<'_> {
    type Output = ();
    
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let len = self.lhs.len();
        debug_assert_eq!(self.rhs.len(), len);
        debug_assert_eq!(self.dst.len(), len);
        
        // 获取 SIMD 宽度
        let width = S::f64s_len();
        let chunks = len / width;
        
        // 主循环：SIMD 处理
        for i in 0..chunks {
            let offset = i * width;
            
            // 加载
            let lhs_vec = unsafe { 
                simd.f64s_load_unaligned(self.lhs.as_ptr().add(offset)) 
            };
            let rhs_vec = unsafe { 
                simd.f64s_load_unaligned(self.rhs.as_ptr().add(offset)) 
            };
            
            // 计算
            let result = simd.f64s_add(lhs_vec, rhs_vec);
            
            // 存储
            unsafe { 
                simd.f64s_store_unaligned(self.dst.as_mut_ptr().add(offset), result) 
            };
        }
        
        // 尾部处理：标量
        for i in (chunks * width)..len {
            self.dst[i] = self.lhs[i] + self.rhs[i];
        }
    }
}

/// 分发 f64 加法
#[cfg(feature = "simd")]
pub fn add_f64_simd(lhs: &[f64], rhs: &[f64], dst: &mut [f64]) {
    let arch = crate::simd::get_arch();
    arch.dispatch(AddF64Kernel { lhs, rhs, dst });
}
```

---

## 5. 标量回退 (scalar.rs)

### 5.1 设计目标

标量回退模块提供所有操作的纯标量实现，作为：

1. **基准实现**：验证 SIMD 实现的正确性
2. **回退路径**：当 SIMD 不可用或不适用时使用
3. **参考文档**：清晰展示每个操作的语义

### 5.2 ScalarKernel 实现

```rust
// src/simd/scalar.rs

use crate::simd::{SimdElement, SimdKernel, SimdOp};

/// 标量内核实现
/// 
/// 所有操作使用纯标量循环实现，不依赖任何 SIMD 指令。
/// 适用于：
/// - SIMD feature 未启用
/// - 数据规模小于 SIMD 宽度
/// - 非连续内存访问
/// - 调试和测试
/// 
/// # Example
/// 
/// ```ignore
/// let kernel = ScalarKernel::<f64>::new();
/// 
/// let a = [1.0, 2.0, 3.0];
/// let b = [4.0, 5.0, 6.0];
/// let mut c = [0.0; 3];
/// 
/// kernel.add(&a, &b, &mut c);
/// assert_eq!(c, [5.0, 7.0, 9.0]);
/// ```
pub struct ScalarKernel<A: SimdElement> {
    _marker: core::marker::PhantomData<A>,
}

impl<A: SimdElement> ScalarKernel<A> {
    /// 创建新的标量内核
    #[inline]
    pub const fn new() -> Self {
        Self {
            _marker: core::marker::PhantomData,
        }
    }
}

impl<A: SimdElement> Default for ScalarKernel<A> {
    fn default() -> Self {
        Self::new()
    }
}

// 为 f64 实现完整的 SimdKernel
impl SimdKernel<f64> for ScalarKernel<f64> {
    // ========================================
    // 元信息
    // ========================================
    
    #[inline]
    fn name() -> &'static str {
        "scalar_f64"
    }
    
    #[inline]
    fn width() -> usize {
        1  // 标量宽度为 1
    }
    
    #[inline]
    fn supports_op(_op: SimdOp) -> bool {
        true  // 标量内核支持所有操作
    }
    
    // ========================================
    // 逐元素一元操作
    // ========================================
    
    #[inline]
    fn neg(&self, src: &[f64], dst: &mut [f64]) {
        assert_eq!(src.len(), dst.len(), "length mismatch");
        for i in 0..src.len() {
            dst[i] = -src[i];
        }
    }
    
    #[inline]
    fn abs(&self, src: &[f64], dst: &mut [f64]) {
        assert_eq!(src.len(), dst.len(), "length mismatch");
        for i in 0..src.len() {
            dst[i] = src[i].abs();
        }
    }
    
    #[inline]
    fn sqrt(&self, src: &[f64], dst: &mut [f64]) {
        assert_eq!(src.len(), dst.len(), "length mismatch");
        for i in 0..src.len() {
            dst[i] = src[i].sqrt();
        }
    }
    
    #[inline]
    fn square(&self, src: &[f64], dst: &mut [f64]) {
        assert_eq!(src.len(), dst.len(), "length mismatch");
        for i in 0..src.len() {
            dst[i] = src[i] * src[i];
        }
    }
    
    #[inline]
    fn recip(&self, src: &[f64], dst: &mut [f64]) {
        assert_eq!(src.len(), dst.len(), "length mismatch");
        for i in 0..src.len() {
            dst[i] = src[i].recip();
        }
    }
    
    // ========================================
    // 逐元素二元操作
    // ========================================
    
    #[inline]
    fn add(&self, lhs: &[f64], rhs: &[f64], dst: &mut [f64]) {
        assert_eq!(lhs.len(), rhs.len(), "length mismatch");
        assert_eq!(lhs.len(), dst.len(), "length mismatch");
        for i in 0..lhs.len() {
            dst[i] = lhs[i] + rhs[i];
        }
    }
    
    #[inline]
    fn sub(&self, lhs: &[f64], rhs: &[f64], dst: &mut [f64]) {
        assert_eq!(lhs.len(), rhs.len(), "length mismatch");
        assert_eq!(lhs.len(), dst.len(), "length mismatch");
        for i in 0..lhs.len() {
            dst[i] = lhs[i] - rhs[i];
        }
    }
    
    #[inline]
    fn mul(&self, lhs: &[f64], rhs: &[f64], dst: &mut [f64]) {
        assert_eq!(lhs.len(), rhs.len(), "length mismatch");
        assert_eq!(lhs.len(), dst.len(), "length mismatch");
        for i in 0..lhs.len() {
            dst[i] = lhs[i] * rhs[i];
        }
    }
    
    #[inline]
    fn div(&self, lhs: &[f64], rhs: &[f64], dst: &mut [f64]) {
        assert_eq!(lhs.len(), rhs.len(), "length mismatch");
        assert_eq!(lhs.len(), dst.len(), "length mismatch");
        for i in 0..lhs.len() {
            dst[i] = lhs[i] / rhs[i];
        }
    }
    
    #[inline]
    fn min(&self, lhs: &[f64], rhs: &[f64], dst: &mut [f64]) {
        assert_eq!(lhs.len(), rhs.len(), "length mismatch");
        assert_eq!(lhs.len(), dst.len(), "length mismatch");
        for i in 0..lhs.len() {
            dst[i] = lhs[i].min(rhs[i]);
        }
    }
    
    #[inline]
    fn max(&self, lhs: &[f64], rhs: &[f64], dst: &mut [f64]) {
        assert_eq!(lhs.len(), rhs.len(), "length mismatch");
        assert_eq!(lhs.len(), dst.len(), "length mismatch");
        for i in 0..lhs.len() {
            dst[i] = lhs[i].max(rhs[i]);
        }
    }
    
    // ========================================
    // 归约操作
    // ========================================
    
    #[inline]
    fn sum(&self, data: &[f64]) -> f64 {
        let mut acc = 0.0;
        for &x in data {
            acc += x;
        }
        acc
    }
    
    #[inline]
    fn prod(&self, data: &[f64]) -> f64 {
        let mut acc = 1.0;
        for &x in data {
            acc *= x;
        }
        acc
    }
    
    #[inline]
    fn reduce_min(&self, data: &[f64]) -> f64 {
        assert!(!data.is_empty(), "cannot reduce empty slice");
        let mut min_val = data[0];
        for &x in &data[1..] {
            min_val = min_val.min(x);
        }
        min_val
    }
    
    #[inline]
    fn reduce_max(&self, data: &[f64]) -> f64 {
        assert!(!data.is_empty(), "cannot reduce empty slice");
        let mut max_val = data[0];
        for &x in &data[1..] {
            max_val = max_val.max(x);
        }
        max_val
    }
    
    // ========================================
    // 内积操作
    // ========================================
    
    #[inline]
    fn dot(&self, lhs: &[f64], rhs: &[f64]) -> f64 {
        assert_eq!(lhs.len(), rhs.len(), "length mismatch");
        let mut acc = 0.0;
        for i in 0..lhs.len() {
            acc += lhs[i] * rhs[i];
        }
        acc
    }
    
    // ========================================
    // 比较操作
    // ========================================
    
    #[inline]
    fn eq(&self, lhs: &[f64], rhs: &[f64], dst: &mut [bool]) {
        assert_eq!(lhs.len(), rhs.len(), "length mismatch");
        assert_eq!(lhs.len(), dst.len(), "length mismatch");
        for i in 0..lhs.len() {
            dst[i] = lhs[i] == rhs[i];
        }
    }
    
    #[inline]
    fn lt(&self, lhs: &[f64], rhs: &[f64], dst: &mut [bool]) {
        assert_eq!(lhs.len(), rhs.len(), "length mismatch");
        assert_eq!(lhs.len(), dst.len(), "length mismatch");
        for i in 0..lhs.len() {
            dst[i] = lhs[i] < rhs[i];
        }
    }
    
    #[inline]
    fn gt(&self, lhs: &[f64], rhs: &[f64], dst: &mut [bool]) {
        assert_eq!(lhs.len(), rhs.len(), "length mismatch");
        assert_eq!(lhs.len(), dst.len(), "length mismatch");
        for i in 0..lhs.len() {
            dst[i] = lhs[i] > rhs[i];
        }
    }
    
    // ========================================
    // 填充操作
    // ========================================
    
    #[inline]
    fn fill(&self, dst: &mut [f64], value: f64) {
        for elem in dst.iter_mut() {
            *elem = value;
        }
    }
    
    #[inline]
    fn fill_zero(&self, dst: &mut [f64]) {
        for elem in dst.iter_mut() {
            *elem = 0.0;
        }
    }
}

// f32 实现（类似，省略）
impl SimdKernel<f32> for ScalarKernel<f32> {
    // ... 类似 f64 实现
    fn name() -> &'static str { "scalar_f32" }
    fn width() -> usize { 1 }
    fn supports_op(_op: SimdOp) -> bool { true }
    // ... 其他方法
}
```

---

## 6. 向量化实现 (vector.rs)

### 6.1 设计目标

向量化实现模块利用 pulp crate 提供的 SIMD 能力：

1. **自动分发**：运行时自动选择最优指令集
2. **对齐感知**：根据对齐状态选择加载指令
3. **尾部处理**：正确处理非整数倍尾部

### 6.2 VectorKernel 结构

```rust
// src/simd/vector.rs

#[cfg(feature = "simd")]
use pulp::{Arch, Simd, WithSimd};
use crate::simd::{SimdElement, SimdKernel, SimdOp};

/// 向量化内核
/// 
/// 使用 pulp crate 提供的 SIMD 能力实现高性能操作。
/// 仅在 `simd` feature 启用时可用。
/// 
/// # Example
/// 
/// ```ignore
/// let kernel = VectorKernel::<f64>::new();
/// 
/// let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let b = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
/// let mut c = [0.0; 8];
/// 
/// kernel.add(&a, &b, &mut c);
/// assert_eq!(c, [9.0; 8]);
/// ```
#[cfg(feature = "simd")]
pub struct VectorKernel<A: SimdElement> {
    /// 缓存的 Arch 实例
    arch: Arch,
    _marker: core::marker::PhantomData<A>,
}

#[cfg(feature = "simd")]
impl<A: SimdElement> VectorKernel<A> {
    /// 创建新的向量化内核
    #[inline]
    pub fn new() -> Self {
        Self {
            arch: crate::simd::get_arch(),
            _marker: core::marker::PhantomData,
        }
    }
    
    /// 使用指定的 Arch 创建
    #[inline]
    pub fn with_arch(arch: Arch) -> Self {
        Self {
            arch,
            _marker: core::marker::PhantomData,
        }
    }
}

#[cfg(feature = "simd")]
impl<A: SimdElement> Default for VectorKernel<A> {
    fn default() -> Self {
        Self::new()
    }
}
```

### 6.3 f64 逐元素二元操作实现

```rust
// src/simd/vector.rs

#[cfg(feature = "simd")]
impl SimdKernel<f64> for VectorKernel<f64> {
    fn name() -> &'static str {
        "vector_f64"
    }
    
    fn width() -> usize {
        // 运行时确定的宽度，这里返回典型值
        // 实际宽度由 pulp 根据当前 CPU 决定
        4  // AVX2 的 f64 宽度
    }
    
    fn supports_op(op: SimdOp) -> bool {
        matches!(op, SimdOp::Unary | SimdOp::Binary | SimdOp::Reduction | SimdOp::Dot)
    }
    
    // ========================================
    // 逐元素二元操作
    // ========================================
    
    fn add(&self, lhs: &[f64], rhs: &[f64], dst: &mut [f64]) {
        assert_eq!(lhs.len(), rhs.len());
        assert_eq!(lhs.len(), dst.len());
        
        struct AddKernel<'a> {
            lhs: &'a [f64],
            rhs: &'a [f64],
            dst: &'a mut [f64],
        }
        
        impl WithSimd for AddKernel<'_> {
            type Output = ();
            
            fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
                let len = self.lhs.len();
                let width = S::f64s_len();
                let chunks = len / width;
                
                // SIMD 主循环
                for i in 0..chunks {
                    let offset = i * width;
                    unsafe {
                        let lhs_vec = simd.f64s_load_unaligned(self.lhs.as_ptr().add(offset));
                        let rhs_vec = simd.f64s_load_unaligned(self.rhs.as_ptr().add(offset));
                        let result = simd.f64s_add(lhs_vec, rhs_vec);
                        simd.f64s_store_unaligned(self.dst.as_mut_ptr().add(offset), result);
                    }
                }
                
                // 尾部处理
                for i in (chunks * width)..len {
                    self.dst[i] = self.lhs[i] + self.rhs[i];
                }
            }
        }
        
        self.arch.dispatch(AddKernel { lhs, rhs, dst });
    }
    
    fn sub(&self, lhs: &[f64], rhs: &[f64], dst: &mut [f64]) {
        // 类似 add 实现
        struct SubKernel<'a> {
            lhs: &'a [f64],
            rhs: &'a [f64],
            dst: &'a mut [f64],
        }
        
        impl WithSimd for SubKernel<'_> {
            type Output = ();
            
            fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
                let len = self.lhs.len();
                let width = S::f64s_len();
                let chunks = len / width;
                
                for i in 0..chunks {
                    let offset = i * width;
                    unsafe {
                        let lhs_vec = simd.f64s_load_unaligned(self.lhs.as_ptr().add(offset));
                        let rhs_vec = simd.f64s_load_unaligned(self.rhs.as_ptr().add(offset));
                        let result = simd.f64s_sub(lhs_vec, rhs_vec);
                        simd.f64s_store_unaligned(self.dst.as_mut_ptr().add(offset), result);
                    }
                }
                
                for i in (chunks * width)..len {
                    self.dst[i] = self.lhs[i] - self.rhs[i];
                }
            }
        }
        
        self.arch.dispatch(SubKernel { lhs, rhs, dst });
    }
    
    fn mul(&self, lhs: &[f64], rhs: &[f64], dst: &mut [f64]) {
        // 类似实现
        // ...
    }
    
    fn div(&self, lhs: &[f64], rhs: &[f64], dst: &mut [f64]) {
        // 类似实现
        // ...
    }
    
    // ========================================
    // 归约操作
    // ========================================
    
    fn sum(&self, data: &[f64]) -> f64 {
        struct SumKernel<'a> {
            data: &'a [f64],
        }
        
        impl WithSimd for SumKernel<'_> {
            type Output = f64;
            
            fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
                let len = self.data.len();
                let width = S::f64s_len();
                let chunks = len / width;
                
                // 初始化累加器
                let mut acc = unsafe { simd.f64s_splat(0.0) };
                
                // SIMD 累加
                for i in 0..chunks {
                    let offset = i * width;
                    unsafe {
                        let vec = simd.f64s_load_unaligned(self.data.as_ptr().add(offset));
                        acc = simd.f64s_add(acc, vec);
                    }
                }
                
                // 水平求和
                let mut sum = 0.0;
                let acc_arr: [f64; 64] = unsafe { core::mem::zeroed() };  // 最大宽度
                unsafe {
                    simd.f64s_store_unaligned(acc_arr.as_ptr() as *mut _, acc);
                }
                for &x in &acc_arr[..width] {
                    sum += x;
                }
                
                // 尾部处理
                for i in (chunks * width)..len {
                    sum += self.data[i];
                }
                
                sum
            }
        }
        
        self.arch.dispatch(SumKernel { data })
    }
    
    // ========================================
    // 内积操作
    // ========================================
    
    fn dot(&self, lhs: &[f64], rhs: &[f64]) -> f64 {
        assert_eq!(lhs.len(), rhs.len());
        
        struct DotKernel<'a> {
            lhs: &'a [f64],
            rhs: &'a [f64],
        }
        
        impl WithSimd for DotKernel<'_> {
            type Output = f64;
            
            fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
                let len = self.lhs.len();
                let width = S::f64s_len();
                let chunks = len / width;
                
                // 初始化累加器
                let mut acc = unsafe { simd.f64s_splat(0.0) };
                
                // SIMD 点积
                for i in 0..chunks {
                    let offset = i * width;
                    unsafe {
                        let lhs_vec = simd.f64s_load_unaligned(self.lhs.as_ptr().add(offset));
                        let rhs_vec = simd.f64s_load_unaligned(self.rhs.as_ptr().add(offset));
                        let prod = simd.f64s_mul(lhs_vec, rhs_vec);
                        acc = simd.f64s_add(acc, prod);
                    }
                }
                
                // 水平求和
                let mut result = 0.0;
                let acc_arr: [f64; 64] = unsafe { core::mem::zeroed() };
                unsafe {
                    simd.f64s_store_unaligned(acc_arr.as_ptr() as *mut _, acc);
                }
                for &x in &acc_arr[..width] {
                    result += x;
                }
                
                // 尾部处理
                for i in (chunks * width)..len {
                    result += self.lhs[i] * self.rhs[i];
                }
                
                result
            }
        }
        
        self.arch.dispatch(DotKernel { lhs, rhs })
    }
    
    // ... 其他方法实现
}
```

---

## 7. 对齐与非对齐加载

### 7.1 对齐检测

```rust
// src/simd/mod.rs

/// 检查指针是否对齐到指定字节
#[inline]
pub fn is_aligned(ptr: *const u8, align: usize) -> bool {
    (ptr as usize) % align == 0
}

/// 检查指针是否 64 字节对齐（SIMD 友好）
#[inline]
pub fn is_simd_aligned(ptr: *const u8) -> bool {
    is_aligned(ptr, 64)
}

/// 检查切片是否对齐
#[inline]
pub fn is_slice_aligned<T>(slice: &[T], align: usize) -> bool {
    is_aligned(slice.as_ptr() as *const u8, align)
}
```

### 7.2 加载策略选择

```rust
// src/simd/vector.rs

#[cfg(feature = "simd")]
impl SimdKernel<f64> for VectorKernel<f64> {
    fn add(&self, lhs: &[f64], rhs: &[f64], dst: &mut [f64]) {
        // 检查对齐状态
        let lhs_aligned = crate::simd::is_simd_aligned(lhs.as_ptr() as *const u8);
        let rhs_aligned = crate::simd::is_simd_aligned(rhs.as_ptr() as *const u8);
        let dst_aligned = crate::simd::is_simd_aligned(dst.as_ptr() as *const u8);
        
        if lhs_aligned && rhs_aligned && dst_aligned {
            // 使用对齐加载路径
            self.add_aligned(lhs, rhs, dst);
        } else {
            // 使用非对齐加载路径
            self.add_unaligned(lhs, rhs, dst);
        }
    }
}

#[cfg(feature = "simd")]
impl VectorKernel<f64> {
    fn add_aligned(&self, lhs: &[f64], rhs: &[f64], dst: &mut [f64]) {
        struct AddAlignedKernel<'a> {
            lhs: &'a [f64],
            rhs: &'a [f64],
            dst: &'a mut [f64],
        }
        
        impl WithSimd for AddAlignedKernel<'_> {
            type Output = ();
            
            fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
                let len = self.lhs.len();
                let width = S::f64s_len();
                let chunks = len / width;
                
                for i in 0..chunks {
                    let offset = i * width;
                    unsafe {
                        // 使用对齐加载（更快）
                        let lhs_vec = simd.f64s_load(self.lhs.as_ptr().add(offset));
                        let rhs_vec = simd.f64s_load(self.rhs.as_ptr().add(offset));
                        let result = simd.f64s_add(lhs_vec, rhs_vec);
                        simd.f64s_store(self.dst.as_mut_ptr().add(offset), result);
                    }
                }
                
                for i in (chunks * width)..len {
                    self.dst[i] = self.lhs[i] + self.rhs[i];
                }
            }
        }
        
        self.arch.dispatch(AddAlignedKernel { lhs, rhs, dst });
    }
    
    fn add_unaligned(&self, lhs: &[f64], rhs: &[f64], dst: &mut [f64]) {
        // 使用非对齐加载（更通用）
        // 实现见前文
    }
}
```

### 7.3 加载策略决策树

```
加载策略选择

┌─────────────────────────────────────────────────────────┐
│          检查 lhs, rhs, dst 的对齐状态                   │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ 全部 64 字节对齐？     │
              └───────────┬───────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼ YES                   ▼ NO
    ┌─────────────────┐     ┌─────────────────┐
    │ 对齐加载路径     │     │ 非对齐加载路径   │
    │ f64s_load()     │     │ f64s_load_unaligned() │
    │ f64s_store()    │     │ f64s_store_unaligned() │
    │ (更快)          │     │ (更通用)         │
    └─────────────────┘     └─────────────────┘
```

---

## 8. 尾部处理

### 8.1 尾部问题

当元素数不是 SIMD 宽度的整数倍时，剩余的元素无法用 SIMD 处理：

```
元素数: 10, SIMD 宽度: 4

┌───────┬───────┬───────┬───────┬───────┐
│ 0 1 2 │ 3 4 5 │ 6 7 8 │ 9     │       │
│ chunk │ chunk │ chunk │ tail  │ pad   │
└───────┴───────┴───────┴───────┴───────┘
│←── SIMD ──→│←── SIMD ──→│←── SIMD ──→│← 标量 →│
```

### 8.2 尾部处理策略

| 策略 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **标量循环** | 简单、安全 | 边界分支 | 通用场景 |
| **掩码加载** | 无分支 | 需要掩码支持 | AVX-512 |
| **过度加载** | 无分支 | 可能读取无效内存 | 需保证缓冲区足够 |

**Senon 采用标量循环策略**：

```rust
// 尾部处理：标量循环
let tail_start = chunks * width;
for i in tail_start..len {
    dst[i] = lhs[i] + rhs[i];
}
```

### 8.3 完整尾部处理示例

```rust
#[cfg(feature = "simd")]
impl WithSimd for AddKernel<'_> {
    type Output = ();
    
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let len = self.lhs.len();
        let width = S::f64s_len();
        
        // 计算完整块数
        let chunks = len / width;
        
        // SIMD 处理完整块
        for i in 0..chunks {
            let offset = i * width;
            unsafe {
                let lhs_vec = simd.f64s_load_unaligned(self.lhs.as_ptr().add(offset));
                let rhs_vec = simd.f64s_load_unaligned(self.rhs.as_ptr().add(offset));
                let result = simd.f64s_add(lhs_vec, rhs_vec);
                simd.f64s_store_unaligned(self.dst.as_mut_ptr().add(offset), result);
            }
        }
        
        // 尾部处理：标量循环
        // 处理 [chunks * width, len) 范围内的元素
        let tail_start = chunks * width;
        for i in tail_start..len {
            self.dst[i] = self.lhs[i] + self.rhs[i];
        }
    }
}
```

---

## 9. 性能分层决策树

### 9.1 完整决策树伪代码

```rust
/// 执行路径枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionPath {
    /// 纯标量路径
    Scalar,
    /// SIMD 路径
    Simd,
    /// 标量 + 并行路径
    ScalarParallel,
    /// SIMD + 并行路径
    SimdParallel,
}

/// 性能分层决策参数
pub struct DispatchParams {
    /// 元素数量
    pub elem_count: usize,
    /// 是否连续内存
    pub is_contiguous: bool,
    /// 是否 64 字节对齐
    pub is_aligned: bool,
    /// SIMD feature 是否启用
    pub simd_enabled: bool,
    /// parallel feature 是否启用
    pub parallel_enabled: bool,
    /// 并行阈值（默认 65536）
    pub parallel_threshold: usize,
    /// SIMD 宽度（运行时确定，默认 4）
    pub simd_width: usize,
}

/// 选择执行路径
/// 
/// 根据数据规模、内存布局和 feature 状态选择最优执行路径。
pub fn select_execution_path(params: &DispatchParams) -> ExecutionPath {
    // 决策树伪代码
    
    // 1. 检查非连续数据
    if !params.is_contiguous {
        // 非连续数据只能使用标量路径
        if params.parallel_enabled && params.elem_count >= params.parallel_threshold {
            return ExecutionPath::ScalarParallel;
        } else {
            return ExecutionPath::Scalar;
        }
    }
    
    // 2. 检查数据规模
    if params.elem_count < params.simd_width {
        // 数据规模小于 SIMD 宽度，使用标量
        if params.parallel_enabled && params.elem_count >= params.parallel_threshold {
            return ExecutionPath::ScalarParallel;
        } else {
            return ExecutionPath::Scalar;
        }
    }
    
    // 3. 数据规模足够，检查 SIMD 是否启用
    if !params.simd_enabled {
        // SIMD 未启用
        if params.parallel_enabled && params.elem_count >= params.parallel_threshold {
            return ExecutionPath::ScalarParallel;
        } else {
            return ExecutionPath::Scalar;
        }
    }
    
    // 4. SIMD 启用，检查是否需要并行
    if params.parallel_enabled && params.elem_count >= params.parallel_threshold {
        return ExecutionPath::SimdParallel;
    } else {
        return ExecutionPath::Simd;
    }
}
```

### 9.2 决策树流程图

```
性能分层决策流程

                        ┌─────────────────────┐
                        │  输入: 元素数, 连续性, │
                        │  对齐, feature 状态    │
                        └──────────┬──────────┘
                                   │
                                   ▼
                      ┌────────────────────────┐
                      │    数据是否连续？       │
                      │  (stride == 1)         │
                      └────────────┬───────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼ NO                          ▼ YES
           ┌────────────────┐            ┌────────────────┐
           │   非连续路径    │            │   连续路径      │
           └───────┬────────┘            └───────┬────────┘
                   │                             │
                   │                             ▼
                   │                   ┌────────────────────┐
                   │                   │ 元素数 < SIMD宽度？│
                   │                   └─────────┬──────────┘
                   │                             │
                   │               ┌─────────────┴─────────────┐
                   │               │                           │
                   │               ▼ YES                       ▼ NO
                   │      ┌────────────────┐         ┌────────────────┐
                   │      │   小数组路径    │         │  检查 SIMD     │
                   │      └───────┬────────┘         │  feature       │
                   │              │                  └───────┬────────┘
                   │              │                          │
                   │              │              ┌───────────┴───────────┐
                   │              │              │                       │
                   │              │              ▼ NO                    ▼ YES
                   │              │     ┌────────────────┐     ┌────────────────┐
                   │              │     │  SIMD 未启用   │     │  SIMD 已启用   │
                   │              │     └───────┬────────┘     └───────┬────────┘
                   │              │             │                      │
                   └──────────────┴─────────────┴──────────────────────┘
                                          │
                                          ▼
                              ┌────────────────────────┐
                              │   检查并行阈值          │
                              │ 元素数 >= 并行阈值？    │
                              └────────────┬───────────┘
                                           │
                            ┌──────────────┴──────────────┐
                            │                             │
                            ▼ YES                         ▼ NO
                   ┌────────────────┐           ┌────────────────┐
                   │  检查 parallel │           │   单线程路径   │
                   │  feature       │           └───────┬────────┘
                   └───────┬────────┘                   │
                           │                            │
             ┌─────────────┴─────────────┐              │
             │                           │              │
             ▼ NO                        ▼ YES          │
    ┌────────────────┐         ┌────────────────┐       │
    │  标量路径      │         │  并行路径      │       │
    │  (回退)        │         │                │       │
    └────────────────┘         └───────┬────────┘       │
                                       │                │
                           ┌───────────┴───────────┐    │
                           │                       │    │
                           ▼ SIMD 未启用           ▼ SIMD 已启用
                  ┌────────────────┐      ┌────────────────┐
                  │ 标量 + 并行    │      │ SIMD + 并行    │
                  └────────────────┘      └────────────────┘
```

### 9.3 决策表

| 元素数 | 连续 | 对齐 | simd | parallel | 执行路径 |
|--------|:----:|:----:|:----:|:--------:|----------|
| < SIMD宽度 | - | - | - | - | 标量 |
| ≥ SIMD宽度 | ❌ | - | - | - | 标量 |
| ≥ SIMD宽度 | ✅ | - | ❌ | ❌ | 标量 |
| ≥ SIMD宽度 | ✅ | ✅ | ✅ | ❌ | SIMD (对齐) |
| ≥ SIMD宽度 | ✅ | ❌ | ✅ | ❌ | SIMD (非对齐) |
| ≥ 并行阈值 | ❌ | - | ❌ | ✅ | 标量 + 并行 |
| ≥ 并行阈值 | ❌ | - | ✅ | ✅ | 标量 + 并行 |
| ≥ 并行阈值 | ✅ | - | ❌ | ✅ | 标量 + 并行 |
| ≥ 并行阈值 | ✅ | ✅ | ✅ | ✅ | SIMD + 并行 |
| ≥ 并行阈值 | ✅ | ❌ | ✅ | ✅ | SIMD + 并行 |

---

## 10. Feature Gate 条件编译

### 10.1 模块级条件编译

```rust
// src/lib.rs

// SIMD 模块仅在 feature 启用时编译
#[cfg(feature = "simd")]
#[cfg_attr(docsrs, doc(cfg(feature = "simd")))]
pub mod simd;

// 非 SIMD 时的占位模块（仅导出标量内核）
#[cfg(not(feature = "simd"))]
pub mod simd {
    //! SIMD 后端（标量回退模式）
    //! 
    //! 当 `simd` feature 未启用时，仅提供标量实现。
    
    pub use crate::simd_scalar::{SimdKernel, ScalarKernel};
}
```

### 10.2 方法级条件编译

```rust
// src/ops/elementwise.rs

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Numeric,
{
    /// 逐元素加法
    pub fn add(&self, other: &Self) -> Tensor<A, D> {
        // ... 形状检查等
        
        #[cfg(feature = "simd")]
        {
            // 尝试 SIMD 路径
            if self.should_use_simd() && other.should_use_simd() {
                return self.add_simd(other);
            }
        }
        
        // 标量回退
        self.add_scalar(other)
    }
    
    #[cfg(feature = "simd")]
    fn add_simd(&self, other: &Self) -> Tensor<A, D> {
        use crate::simd::{get_arch, SimdKernel, VectorKernel};
        
        let kernel = VectorKernel::new();
        // ... SIMD 实现
    }
    
    fn add_scalar(&self, other: &Self) -> Tensor<A, D> {
        use crate::simd::ScalarKernel;
        
        let kernel = ScalarKernel::new();
        // ... 标量实现
    }
}
```

### 10.3 类型级条件编译

```rust
// src/simd/mod.rs

/// SIMD 内核类型别名
/// 
/// 根据 feature 状态选择向量或标量实现。
#[cfg(feature = "simd")]
pub type DefaultKernel<A> = VectorKernel<A>;

#[cfg(not(feature = "simd"))]
pub type DefaultKernel<A> = ScalarKernel<A>;

/// 获取默认内核实例
#[cfg(feature = "simd")]
pub fn default_kernel<A: SimdElement>() -> DefaultKernel<A> {
    VectorKernel::new()
}

#[cfg(not(feature = "simd"))]
pub fn default_kernel<A: SimdElement>() -> DefaultKernel<A> {
    ScalarKernel::new()
}
```

### 10.4 编译配置总结

```toml
# Cargo.toml

[features]
default = ["std"]
std = []
parallel = ["dep:rayon", "std"]
simd = ["dep:pulp"]

[dependencies]
pulp = { version = "0.18", optional = true }
rayon = { version = "1.10", optional = true }
once_cell = { version = "1.19", optional = true }  # 用于 Arch 缓存
```

| Feature 组合 | Arch 缓存 | 可用内核 | 性能特性 |
|-------------|----------|----------|----------|
| 无 | - | ScalarKernel | 标量 |
| `std` | - | ScalarKernel | 标量 |
| `simd` | ✅ | VectorKernel, ScalarKernel | SIMD |
| `parallel` | - | ScalarKernel + 并行 | 标量并行 |
| `simd, parallel` | ✅ | VectorKernel + 并行 | SIMD 并行 |

---

## 11. 与其他模块的交互

### 11.1 与 ops 模块的交互

| 交互点 | 方向 | 说明 |
|--------|------|------|
| 逐元素运算 | ops → simd | `elementwise::add/sub/mul/div` 调用 SIMD 内核 |
| 归约运算 | ops → simd | `reduction::sum/prod/min/max` 调用 SIMD 内核 |
| 矩阵运算 | ops → simd | `matrix::dot` 调用 SIMD 内核 |
| 路径选择 | ops ← layout | ops 查询 layout 决定是否使用 SIMD |

```rust
// src/ops/elementwise.rs

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: RealScalar,
{
    pub fn map<F>(&self, f: F) -> Tensor<A, D>
    where
        F: Fn(A) -> A + Send + Sync,
    {
        let mut result = Tensor::zeros(self.dim());
        
        // 查询布局决定执行路径
        let use_simd = self.is_contiguous() 
            && self.is_aligned()
            && cfg!(feature = "simd");
        
        if use_simd {
            #[cfg(feature = "simd")]
            {
                use crate::simd::dispatch_unary;
                dispatch_unary(self.as_slice(), result.as_mut_slice(), f);
            }
            #[cfg(not(feature = "simd"))]
            {
                self.map_scalar(&f, &mut result);
            }
        } else {
            self.map_scalar(&f, &mut result);
        }
        
        result
    }
}
```

### 11.2 与 iter 模块的交互

| 交互点 | 方向 | 说明 |
|--------|------|------|
| 元素迭代 | iter → simd | 连续迭代器可使用 SIMD 加速 |
| 并行迭代 | iter → simd | 并行迭代中每个线程使用 SIMD |

```rust
// src/iter/elements.rs

impl<'a, S, D, A> Iterator for ElementsIter<'a, S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    type Item = &'a A;
    
    fn next(&mut self) -> Option<Self::Item> {
        // 迭代器本身不使用 SIMD
        // 但可配合 SIMD 操作使用
        // ...
    }
}

// 提供 SIMD 友好的批量访问
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// 获取连续块（用于 SIMD）
    pub fn as_contiguous_slice(&self) -> Option<&[A]> {
        if self.is_contiguous() {
            Some(unsafe {
                core::slice::from_raw_parts(self.as_ptr(), self.len())
            })
        } else {
            None
        }
    }
}
```

### 11.3 与 parallel 模块的交互

| 交互点 | 方向 | 说明 |
|--------|------|------|
| 并行分发 | parallel → simd | 并行操作内部使用 SIMD |
| 线程内 SIMD | parallel ← simd | 每个线程内部使用 SIMD 内核 |

```rust
// src/parallel/par_ops.rs

#[cfg(feature = "parallel")]
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A> + Send + Sync,
    D: Dimension + Send,
    A: Numeric + Send + Sync,
{
    pub fn par_add(&self, other: &Self) -> Tensor<A, D> {
        use rayon::prelude::*;
        
        let len = self.len();
        let chunk_size = 4096;  // 最小块大小
        
        let mut result = Tensor::zeros(self.dim());
        
        // 并行分块
        result.as_mut_slice().par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(i, chunk)| {
                let start = i * chunk_size;
                let end = (start + chunk_size).min(len);
                
                let lhs = &self.as_slice()[start..end];
                let rhs = &other.as_slice()[start..end];
                
                // 每个线程内部使用 SIMD
                #[cfg(feature = "simd")]
                {
                    use crate::simd::{SimdKernel, VectorKernel};
                    let kernel = VectorKernel::new();
                    kernel.add(lhs, rhs, chunk);
                }
                
                #[cfg(not(feature = "simd"))]
                {
                    use crate::simd::ScalarKernel;
                    let kernel = ScalarKernel::new();
                    kernel.add(lhs, rhs, chunk);
                }
            });
        
        result
    }
}
```

### 11.4 与 layout 模块的交互

| 交互点 | 方向 | 说明 |
|--------|------|------|
| 连续性检查 | simd ← layout | SIMD 路径需要连续布局 |
| 对齐检查 | simd ← layout | 对齐加载路径需要 64 字节对齐 |
| 步长检查 | simd ← layout | 非连续步长回退标量 |

```rust
// src/simd/mod.rs

/// 检查张量是否适合 SIMD 操作
pub trait SimdCompatible {
    /// 是否连续内存
    fn is_contiguous(&self) -> bool;
    
    /// 是否 64 字节对齐
    fn is_aligned(&self) -> bool;
    
    /// 是否适合 SIMD（连续且对齐）
    fn is_simd_compatible(&self) -> bool {
        self.is_contiguous() && self.is_aligned()
    }
}

// 为 TensorBase 实现
impl<S, D> SimdCompatible for TensorBase<S, D>
where
    S: crate::storage::RawStorage,
    D: crate::dimension::Dimension,
{
    fn is_contiguous(&self) -> bool {
        self.layout_flags().is_contiguous()
    }
    
    fn is_aligned(&self) -> bool {
        self.layout_flags().is_aligned()
    }
}
```

---

## 12. 实现任务分解

### 任务清单

| # | 任务 | 预估时间 | 依赖 | 产出 |
|---|------|----------|------|------|
| 1 | 定义 `SimdElement` trait 和 `SimdOp` 枚举 | 10 min | 无 | `mod.rs` |
| 2 | 定义 `SimdKernel` trait 完整签名 | 15 min | T1 | `mod.rs` |
| 3 | 实现 `ScalarKernel<f64>` | 15 min | T2 | `scalar.rs` |
| 4 | 实现 `ScalarKernel<f32>` | 10 min | T2 | `scalar.rs` |
| 5 | 添加 Arch 缓存和 `get_arch()` | 10 min | 无 | `mod.rs` |
| 6 | 实现 `VectorKernel<f64>` 逐元素操作 | 20 min | T2, T5 | `vector.rs` |
| 7 | 实现 `VectorKernel<f64>` 归约和内积 | 20 min | T6 | `vector.rs` |
| 8 | 实现对齐/非对齐加载策略 | 15 min | T6 | `vector.rs` |
| 9 | 实现性能分层决策树 | 15 min | T1-T8 | `mod.rs` |
| 10 | 添加单元测试和基准测试 | 20 min | T1-T9 | `tests/` |

### 任务依赖图

```
T1 ──→ T2 ──┬──→ T3 ──→ T6 ──→ T7 ──→ T8 ──┐
            │                              │
            └──→ T4                        ├──→ T9 ──→ T10
                                           │
            T5 ────────────────────────────┘
```

### 并行执行建议

- **Wave 1**: T1, T5（可独立开始）
- **Wave 2**: T2（依赖 T1）
- **Wave 3**: T3, T4（依赖 T2，可并行）
- **Wave 4**: T6（依赖 T2, T5）
- **Wave 5**: T7（依赖 T6）
- **Wave 6**: T8（依赖 T6）
- **Wave 7**: T9（依赖 T3-T8）
- **Wave 8**: T10（依赖 T9）

---

## 13. 设计决策记录

### D1: 为什么选择 pulp 而非直接使用 std::arch?

**决策**: 使用 pulp crate 抽象 SIMD 指令集。

**理由**:
1. **跨平台**: pulp 统一抽象 x86 (AVX-512/AVX2/SSE) 和 ARM (NEON)
2. **运行时分发**: `Arch::dispatch()` 自动选择最优指令集
3. **代码简洁**: 避免大量 `#[cfg(target_arch)]` 和 `#[cfg(target_feature)]`
4. **维护成本**: pulp 由社区维护，及时跟进新指令集
5. **类型安全**: pulp 的 `WithSimd` trait 提供编译期类型检查

### D2: 为什么默认关闭 simd feature?

**决策**: `simd` feature 默认关闭。

**理由**:
1. **稳定性**: SIMD 实现可能存在边缘情况 bug
2. **兼容性**: 某些平台可能不支持 pulp
3. **调试友好**: 默认标量路径便于调试
4. **显式启用**: 用户明确知情后启用，避免意外行为

### D3: 为什么非连续数据回退标量?

**决策**: 步长 ≠ 1 时强制使用标量路径。

**理由**:
1. **复杂性**: 非连续加载需要 gather/scatter 指令，并非所有平台支持
2. **性能**: 非连续 SIMD 加载可能比标量更慢（缓存不友好）
3. **实现成本**: 支持非连续 SIMD 会大幅增加代码复杂度

### D4: 为什么使用标量循环处理尾部?

**决策**: 尾部处理使用标量循环而非掩码或过度加载。

**理由**:
1. **简单性**: 标量循环最简单、最安全
2. **通用性**: 所有平台都支持
3. **性能**: 尾部元素数 < SIMD宽度，标量开销可忽略
4. **安全性**: 避免掩码或过度加载的潜在问题

### D5: 为什么缓存 Arch 实例?

**决策**: 使用 `once_cell::Lazy` 缓存全局 `Arch` 实例。

**理由**:
1. **性能**: `Arch::new()` 涉及 CPUID 调用，有一定开销
2. **Copy 类型**: `Arch` 为 `Copy`，缓存后可零成本复用
3. **线程安全**: `Lazy` 保证线程安全的延迟初始化
4. **单例模式**: CPU 能力在程序运行期间不会改变

### D6: 为什么 SimdKernel 是 trait 而非函数?

**决策**: 使用 trait 定义 SIMD 内核接口。

**理由**:
1. **统一接口**: 标量和向量实现共用同一 trait
2. **泛型友好**: 调用方可通过泛型选择实现
3. **可扩展**: 便于添加新的内核类型（如 GPU 内核）
4. **零成本**: trait 边界在编译期内联，无虚调用开销

### D7: 为什么要求 64 字节对齐?

**决策**: `ALIGNED` 标志基于 64 字节对齐。

**理由**:
1. **AVX-512**: 512-bit = 64 字节，当前最宽 SIMD 寄存器
2. **缓存行**: 现代 CPU 缓存行通常为 64 字节
3. **通用性**: 64 字节对齐满足 SSE/AVX/AVX2/AVX-512 所有需求
4. **性能**: 对齐加载在某些平台上明显更快

### D8: 为什么 no_std 下仍支持 pulp?

**决策**: `simd` feature 在 `no_std` 环境下仍可用。

**理由**:
1. **嵌入式需求**: 某些嵌入式平台有 SIMD 能力
2. **pulp 支持**: pulp crate 本身支持 `no_std`
3. **一致性**: 保持 API 在所有环境下一致
4. **可选性**: 通过 feature gate 完全可选

---

## 附录 A: pulp API 快速参考

```rust
// 获取 SIMD 宽度
S::f64s_len()  // f64 向量元素数
S::f32s_len()  // f32 向量元素数

// 加载/存储
simd.f64s_load(ptr)           // 对齐加载
simd.f64s_load_unaligned(ptr) // 非对齐加载
simd.f64s_store(ptr, vec)     // 对齐存储
simd.f64s_store_unaligned(ptr, vec)  // 非对齐存储

// 算术运算
simd.f64s_add(a, b)   // 加法
simd.f64s_sub(a, b)   // 减法
simd.f64s_mul(a, b)   // 乘法
simd.f64s_div(a, b)   // 除法
simd.f64s_sqrt(a)     // 平方根

// 归约
simd.f64s_reduce_sum(a)  // 水平求和

// 常量
simd.f64s_splat(0.0)  // 广播标量到向量
```

## 附录 B: 性能基准参考

| 操作 | 数据规模 | 标量 | SIMD (AVX2) | 加速比 |
|------|----------|------|-------------|--------|
| add (f64) | 1M | 2.1 ms | 0.6 ms | 3.5x |
| mul (f64) | 1M | 2.0 ms | 0.5 ms | 4.0x |
| sum (f64) | 1M | 1.8 ms | 0.4 ms | 4.5x |
| dot (f64) | 1M | 3.5 ms | 0.9 ms | 3.9x |

> 注：以上数据为参考值，实际性能取决于具体硬件和数据特征。

---

*本文档由 Senon 项目维护。如有问题请提交 Issue 或 PR。*
