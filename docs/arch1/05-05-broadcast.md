# Senon 广播模块设计文档

> **文档版本**: v1.0  
> **最后更新**: 2026-03-28  
> **模块路径**: `src/broadcast.rs`  
> **需求来源**: require-v18.md §10.4

---

## 1. 模块概述

### 1.1 广播在科学计算中的重要性

广播（Broadcasting）是现代数值计算库的核心特性，它允许不同形状的数组进行算术运算，而无需显式复制数据。广播机制的重要性体现在：

| 优势 | 说明 |
|------|------|
| **内存效率** | 避免为小数组复制扩展到匹配大数组的形状 |
| **代码简洁** | 无需手动调整形状，`a + b` 自动处理维度差异 |
| **性能优化** | 零拷贝广播视图通过零步长实现逻辑扩展 |
| **数值稳定性** | 标量与数组运算时保持原始精度 |

**典型应用场景**：

```rust
// 矩阵每行归一化（减去均值）
let matrix: Tensor2<f64> = ...;  // shape: [M, N]
let mean: Tensor1<f64> = ...;    // shape: [N]
let normalized = &matrix - &mean.broadcast_to([M, N])?;  // mean 自动扩展

// 批量运算中的标量缩放
let batch: Tensor3<f64> = ...;   // shape: [B, M, N]
let scaled = &batch * 0.5;       // 标量广播到 [B, M, N]

// 多维特征加权
let features: Tensor3<f64> = ...;  // shape: [B, T, D]
let weights: Tensor1<f64> = ...;   // shape: [D]
let weighted = &features * &weights.broadcast_to([B, T, D])?;
```

### 1.2 设计目标

1. **NumPy 兼容**: 严格遵循 NumPy 广播规则，确保用户迁移成本最小
2. **零拷贝实现**: 通过零步长实现广播视图，避免数据复制
3. **类型安全**: 编译时阻止对广播视图的写操作
4. **内存安全**: 零步长访问保证不越界，不产生未定义行为
5. **高效查询**: O(1) 检测数组是否为广播视图

### 1.3 在架构中的位置

```
依赖层级：

L2: broadcast ←── error, dimension
              ↓
L3: ops ←── broadcast, iter
   ↓
L4: tensor ←── ops, broadcast

广播模块被 ops 模块（运算符重载）、iter 模块（Zip 迭代器）、
shape_ops 模块（broadcast_to 方法）依赖。
```

---

## 2. 文件结构

```
src/
└── broadcast.rs       # 广播规则实现
```

### 2.1 模块职责

| 组件 | 职责 |
|------|------|
| `BroadcastInfo` | 存储广播后的形状和步长信息 |
| `broadcast_shape()` | 计算两个形状广播后的结果形状 |
| `broadcast_shapes()` | 计算 N 个形状广播后的结果形状（多操作数） |
| `can_broadcast()` | 检查两个形状是否兼容 |
| `broadcast_strides()` | 计算广播后的步长（含零步长） |
| `BroadcastError` | 广播失败的错误类型 |

---

## 3. 广播规则算法

### 3.1 NumPy 广播规则

广播遵循以下核心规则：

1. **维度对齐**: 从最右维度开始对齐，维度数不足的数组在左侧补 1
2. **兼容条件**: 对应维度相等，或其中一个为 1（或不存在）
3. **结果形状**: 每个维度取两者的最大值

### 3.2 形状兼容性检查算法（伪代码）

```
function can_broadcast(shape_a: [usize; N], shape_b: [usize; M]) -> bool:
    // 从右向左对齐比较
    i = N - 1  // shape_a 的最右索引
    j = M - 1  // shape_b 的最右索引
    
    while i >= 0 or j >= 0:
        // 获取当前维度，不足则视为 1
        dim_a = if i >= 0 then shape_a[i] else 1
        dim_b = if j >= 0 then shape_b[j] else 1
        
        // 检查兼容性
        if dim_a != dim_b and dim_a != 1 and dim_b != 1:
            return false  // 不兼容
        
        i = i - 1
        j = j - 1
    
    return true  // 兼容
```

### 3.3 广播形状推导算法（伪代码）

```
function broadcast_shape(shape_a: [usize; N], shape_b: [usize; M]) -> Result<[usize; max(N,M)], BroadcastError>:
    // 结果维度数为两者最大值
    result_ndim = max(N, M)
    result_shape = array of size result_ndim
    
    // 从右向左对齐计算
    i = N - 1
    j = M - 1
    k = result_ndim - 1
    
    while k >= 0:
        // 获取当前维度
        dim_a = if i >= 0 then shape_a[i] else 1
        dim_b = if j >= 0 then shape_b[j] else 1
        
        // 检查兼容性
        if dim_a != dim_b and dim_a != 1 and dim_b != 1:
            return Error(BroadcastError {
                shape_a: shape_a,
                shape_b: shape_b,
                incompatible_dim: k,
                dim_a: dim_a,
                dim_b: dim_b,
            })
        
        // 结果维度取最大值
        result_shape[k] = max(dim_a, dim_b)
        
        i = i - 1
        j = j - 1
        k = k - 1
    
    return Ok(result_shape)
```

### 3.4 算法示例

**示例 1: 基本广播**

```
shape_a = [3, 1, 4]    (3D)
shape_b = [    4, 1]   (2D)

对齐后:
shape_a: [3, 1, 4]
shape_b: [1, 4, 1]  ← 左侧补 1

逐维度比较:
dim 2: max(4, 1) = 4  ✓ (4 == 4 or 1)
dim 1: max(1, 4) = 4  ✓ (1 or 4 == 4)
dim 0: max(3, 1) = 3  ✓ (3 == 3 or 1)

结果: [3, 4, 4]
```

**示例 2: 不兼容广播**

```
shape_a = [3, 2]
shape_b = [3, 4]

对齐后:
shape_a: [3, 2]
shape_b: [3, 4]

逐维度比较:
dim 1: 2 vs 4 → 不兼容 (2 ≠ 4 且 2 ≠ 1 且 4 ≠ 1)
dim 0: 3 vs 3 → 兼容

结果: BroadcastError("shapes [3, 2] and [3, 4] not broadcastable at dim 1: 2 != 4")
```

---

## 4. 广播形状推导

### 4.1 两个形状的广播

给定两个形状，推导结果形状：

```
输入:
  shape_a: [usize; N]
  shape_b: [usize; M]

输出:
  result_shape: [usize; max(N, M)]

规则:
  result[i] = max(shape_a[i'], shape_b[i'])
  其中 i' 是对齐后的索引
```

**Rust API 签名**：

```rust
/// 计算两个形状广播后的结果形状
///
/// # Arguments
/// * `shape_a` - 第一个形状
/// * `shape_b` - 第二个形状
///
/// # Returns
/// * `Ok(result_shape)` - 广播后的形状
/// * `Err(BroadcastError)` - 形状不兼容
///
/// # Examples
/// ```
/// let a = &[3, 1, 4];
/// let b = &[4, 1];
/// let result = broadcast_shape(a, b)?;
/// assert_eq!(result, vec![3, 4, 4]);
/// ```
pub fn broadcast_shape(shape_a: &[usize], shape_b: &[usize]) -> Result<SmallVec<[usize; 6]>, BroadcastError>;
```

### 4.2 具体示例

| shape_a | shape_b | result_shape | 说明 |
|---------|---------|--------------|------|
| `[5, 4]` | `[1]` | `[5, 4]` | 标量广播 |
| `[5, 4]` | `[4]` | `[5, 4]` | 行向量广播 |
| `[5, 4]` | `[5, 1]` | `[5, 4]` | 列向量广播 |
| `[3, 1, 4]` | `[1, 4, 1]` | `[3, 4, 4]` | 多维广播 |
| `[2, 3, 4, 5]` | `[4, 1]` | `[2, 3, 4, 5]` | 高维广播 |
| `[64, 1, 128]` | `[64, 32, 1]` | `[64, 32, 128]` | 中间维度广播 |

---

## 5. 零步长语义

### 5.1 零步长的概念

零步长（zero stride）是广播的核心实现技术。当某个维度从 1 扩展到 N 时，该维度的步长设为 0，使得：

- 访问索引 `[i, j, k]` 和 `[i, j, k+1]` 时，在该维度上访问同一内存位置
- 逻辑上"重复"该元素 N 次，但物理上只有 1 个副本

### 5.2 零步长计算算法（伪代码）

```
function broadcast_strides(
    original_shape: [usize; N],
    original_strides: [isize; N],
    target_shape: [usize; M]
) -> [isize; M]:
    // M >= N，结果维度数 >= 原始维度数
    
    result_strides = array of size M
    
    // 从右向左对齐
    i = N - 1  // 原始形状索引
    j = M - 1  // 目标形状索引
    
    while j >= 0:
        if i >= 0:
            orig_dim = original_shape[i]
            target_dim = target_shape[j]
            
            if orig_dim == target_dim:
                // 维度相同，保留原步长
                result_strides[j] = original_strides[i]
            else if orig_dim == 1 and target_dim > 1:
                // 广播维度：步长设为 0
                result_strides[j] = 0
            else:
                // 不应该到达这里（can_broadcast 应已检查）
                panic("Incompatible broadcast")
            
            i = i - 1
        else:
            // 新增维度（左侧补 1），步长设为 0
            result_strides[j] = 0
        
        j = j - 1
    
    return result_strides
```

### 5.3 零步长的内存安全性

**核心保证**: 零步长访问永远不会越界。

**安全性分析**：

1. **不变式**: 对于 shape[i] = 1 的维度，任何合法索引 `idx` 满足 `0 <= idx < target_shape[i]`，访问的偏移量为 `stride[i] * idx = 0 * idx = 0`

2. **边界检查**: 即使 `idx` 在目标形状中很大（如 `target_shape[i] = 1000`），实际访问的偏移量始终为 0，指向该维度的唯一元素

3. **别名安全**: 同一广播维度的多个索引访问同一内存位置，但：
   - 广播视图是只读的（`TensorView`）
   - 不存在写操作，无数据竞争
   - 多次读取同一位置是安全的

**示例**：

```
原始数组:
  shape: [1, 4]
  strides: [4, 1]
  data: [a, b, c, d]

广播到 [3, 4]:
  new_strides: [0, 1]  ← 第一轴步长为 0
  
访问模式:
  [0, 0] → offset = 0*0 + 0*1 = 0   → data[0] = a
  [0, 1] → offset = 0*0 + 1*1 = 1   → data[1] = b
  [0, 2] → offset = 0*0 + 2*1 = 2   → data[2] = c
  [0, 3] → offset = 0*0 + 3*1 = 3   → data[3] = d
  [1, 0] → offset = 1*0 + 0*1 = 0   → data[0] = a  (重复!)
  [1, 1] → offset = 1*0 + 1*1 = 1   → data[1] = b  (重复!)
  [2, 0] → offset = 2*0 + 0*1 = 0   → data[0] = a  (重复!)
  ...

内存安全:
  - 所有偏移量在 [0, 3] 范围内，无越界
  - 第 0 行、第 1 行、第 2 行访问相同内存位置
  - 只读访问，无数据竞争
```

### 5.4 零步长示例

**广播前后 shape + strides 对比**：

```
原始 Tensor (F-order):
  shape: [1, 4]
  strides: [4, 1]
  内存布局: [a, b, c, d] (连续)
            索引 [0,0]→a, [0,1]→b, [0,2]→c, [0,3]→d

广播到 [3, 4] 后:
  shape: [3, 4]
  strides: [0, 1]    ← 第 0 轴步长变为 0
  内存布局: 不变，仍为 [a, b, c, d]
  
  逻辑视图:
    行0: [a, b, c, d]
    行1: [a, b, c, d]  (重复!)
    行2: [a, b, c, d]  (重复!)
  
  物理访问:
    [0,j] → data[j]
    [1,j] → data[j]  (相同地址)
    [2,j] → data[j]  (相同地址)
```

**更复杂的示例**：

```
原始 Tensor:
  shape: [2, 1, 3]
  strides: [3, 3, 1]  (F-order)
  元素数: 6

广播到 [2, 4, 3] 后:
  shape: [2, 4, 3]
  strides: [3, 0, 1]  ← 第 1 轴步长变为 0
  
  逻辑视图 (2 × 4 × 3 = 24 个元素):
    [0,0,:] = [a, b, c]
    [0,1,:] = [a, b, c]  (重复!)
    [0,2,:] = [a, b, c]  (重复!)
    [0,3,:] = [a, b, c]  (重复!)
    [1,0,:] = [d, e, f]
    [1,1,:] = [d, e, f]  (重复!)
    ...
  
  物理内存: 仍为 6 个元素，无复制
```

---

## 6. 广播视图创建

### 6.1 broadcast_to 方法设计

```rust
impl<'a, A, D> TensorView<'a, A, D>
where
    D: Dimension,
{
    /// 将视图广播到目标形状
    ///
    /// 返回一个新的只读视图，通过零步长实现逻辑扩展。
    /// 不复制数据，O(1) 操作。
    ///
    /// # Arguments
    /// * `shape` - 目标形状
    ///
    /// # Returns
    /// * `Ok(TensorView)` - 广播后的视图
    /// * `Err(BroadcastError)` - 形状不兼容
    ///
    /// # Examples
    /// ```
    /// let a = Tensor::from_shape_vec([1, 4], vec![1, 2, 3, 4])?;
    /// let broadcasted = a.view().broadcast_to([3, 4])?;
    /// // broadcasted.shape() = [3, 4]
    /// // broadcasted.strides() = [0, 1]
    /// ```
    pub fn broadcast_to<E>(&self, shape: E) -> Result<TensorView<'a, A, E>, BroadcastError>
    where
        E: Dimension,
    {
        // 1. 检查形状兼容性
        let target_shape = shape.slice();
        if !can_broadcast(self.shape(), target_shape) {
            return Err(BroadcastError::incompatible(self.shape(), target_shape));
        }
        
        // 2. 计算广播后的步长
        let new_strides = broadcast_strides(self.shape(), self.strides(), target_shape);
        
        // 3. 创建广播视图（共享底层存储）
        Ok(TensorView {
            storage: self.storage,  // 共享引用
            shape: E::from_slice(&target_shape),
            strides: E::from_isize_slice(&new_strides),
            offset: self.offset,
            layout: self.layout.update_for_broadcast(&new_strides),
        })
    }
}
```

### 6.2 广播视图的只读性

**关键约束**: 广播视图必须是只读的。

**原因**：
1. **别名冲突**: 广播视图中多个索引指向同一物理位置，写入会导致不确定性
2. **语义模糊**: `broadcasted[[0, 0]] = x` 和 `broadcasted[[1, 0]] = y` 写入同一位置，后者覆盖前者
3. **内存安全**: 并发写入同一位置可能造成数据竞争

**实现方式**：

```rust
// broadcast_to 只对 TensorView（只读）可用
impl<'a, A, D> TensorView<'a, A, D> { ... }

// 不为 TensorViewMut 提供 broadcast_to_mut
// impl<'a, A, D> TensorViewMut<'a, A, D> { ... }  // 不存在！

// 也不为 Owned 提供 broadcast_to（因为 Owned 可变）
// 但可以 Owned.view().broadcast_to()
```

### 6.3 布局标志更新

广播后需要更新布局标志：

```
function update_flags_for_broadcast(source_flags, new_strides):
    flags = source_flags
    
    // 设置零步长标志
    flags = flags.set_has_zero_stride(any(s == 0 for s in new_strides))
    
    // 广播后通常不连续（除非广播维度为 1 或不存在）
    // 检查连续性：
    // 如果原始连续且没有广播（所有新维度都是 1），则保持连续
    // 否则设置为非连续
    
    has_broadcast = any(target_dim > 1 and orig_dim == 1 
                        for each aligned dimension)
    
    if has_broadcast:
        // 广播后的视图不连续（物理上不是紧密排列）
        flags = flags.set_f_contiguous(false)
        flags = flags.set_c_contiguous(false)
    
    return flags
```

---

## 7. 标量广播

### 7.1 标量的维度语义

标量（0 维数组）可以与任意维度的数组广播：

| 标量类型 | ndim | shape | 说明 |
|----------|------|-------|------|
| `A` (裸值) | 0 | `[]` | 零维 |
| `Tensor0<A>` | 0 | `[]` | 显式 0 维张量 |
| `Tensor1<A>` (长度 1) | 1 | `[1]` | 1 维但可视为标量 |

### 7.2 标量广播规则

**规则**: 标量视为 0 维数组，与任意形状兼容，广播后形状为目标形状。

```
标量 + [M, N] → [M, N]
标量 + [B, M, N] → [B, M, N]
```

**实现**：

```rust
// 标量与张量的运算符重载
impl<A, D> Add<A> for Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;
    
    fn add(self, rhs: A) -> Self::Output {
        // 标量广播：mapv 比创建广播视图更高效
        self.mapv(|x| x + rhs)
    }
}

// 反向：张量与标量
impl<A, D> Add<Tensor<A, D>> for A
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;
    
    fn add(self, rhs: Tensor<A, D>) -> Self::Output {
        rhs.mapv(|x| self + x)
    }
}
```

### 7.3 0 维数组的广播

0 维数组（`Tensor0<A>`）可以广播到任意形状：

```rust
let scalar: Tensor0<f64> = Tensor0::from(3.14);
let matrix: Tensor2<f64> = Tensor::zeros([4, 5]);

// 0 维广播到 2 维
let broadcasted = scalar.view().broadcast_to([4, 5])?;
// broadcasted.strides() = [0, 0]（全零步长）
// 逻辑上：4×5 的矩阵，所有元素都是 3.14
```

**步长计算**：

```
原始:
  shape: []
  strides: []  (空)
  ndim: 0

广播到 [M, N]:
  new_shape: [M, N]
  new_strides: [0, 0]  ← 所有维度都是新增的，步长全为 0
  
访问:
  [i, j] → offset = 0*i + 0*j = 0 → data[0]（唯一元素）
```

---

## 8. 原地广播约束

### 8.1 原地运算的广播规则

对于复合赋值运算（`+=`, `-=`, `*=`, `/=`），广播规则如下：

| 运算 | LHS | RHS | 广播方向 |
|------|-----|-----|----------|
| `a += b` | 不可广播 | 可广播 | RHS 广播到 LHS |
| `a -= b` | 不可广播 | 可广播 | RHS 广播到 LHS |
| `a *= b` | 不可广播 | 可广播 | RHS 广播到 LHS |
| `a /= b` | 不可广播 | 可广播 | RHS 广播到 LHS |

**核心原则**: **仅 RHS 可广播，LHS 必须拥有完整存储。**

### 8.2 约束原因

1. **写入语义**: LHS 是写入目标，不能是广播视图（广播视图只读）
2. **形状不变**: 原地运算不改变 LHS 的形状
3. **内存安全**: 避免"写入广播维度"导致的别名冲突

### 8.3 实现示例

```rust
impl<A, D> AddAssign<TensorView<'_, A, E>> for Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
    E: Dimension,
{
    fn add_assign(&mut self, rhs: TensorView<'_, A, E>) {
        // 1. 检查 RHS 是否可广播到 LHS
        let target_shape = self.shape();
        if !can_broadcast(rhs.shape(), target_shape) {
            panic!("Cannot broadcast RHS to LHS shape");
        }
        
        // 2. 广播 RHS
        let rhs_broadcast = rhs.broadcast_to(self.dim.clone())
            .expect("Already checked compatibility");
        
        // 3. 逐元素加（使用 Zip 迭代器）
        Zip::from(self)
            .and(&rhs_broadcast)
            .for_each(|lhs, &rhs_val| {
                *lhs = *lhs + rhs_val;
            });
    }
}
```

### 8.4 错误场景

```rust
// 场景 1: LHS 形状小于 RHS（无法广播）
let mut a = Tensor::zeros([3, 4]);  // LHS
let b = Tensor::zeros([3, 4, 5]);   // RHS
// a += &b;  // 运行时错误：RHS [3,4,5] 无法广播到 LHS [3,4]

// 场景 2: LHS 是广播视图
let large = Tensor::zeros([10, 20]);
let a_view = large.slice(s![.., 0..1]);  // [10, 1]，广播视图
let b = Tensor::zeros([10, 20]);
// a_view += &b;  // 编译错误：TensorViewMut 不支持 +=（或运行时检查）

// 场景 3: RHS 包含 LHS 没有的维度
let mut a = Tensor::zeros([4]);      // LHS
let b = Tensor::zeros([3, 4]);       // RHS
// a += &b;  // 运行时错误：RHS [3,4] 无法广播到 LHS [4]
```

---

## 9. 多操作数广播

### 9.1 逐对广播推导

当有 3 个或更多操作数时，广播通过逐对推导实现：

```
function broadcast_shapes(shapes: [[usize; ?]; N]) -> Result<[usize; ?], BroadcastError>:
    if N == 0:
        return Ok([])
    
    if N == 1:
        return Ok(shapes[0])
    
    // 逐对广播
    result = shapes[0]
    for i from 1 to N-1:
        result = broadcast_shape(result, shapes[i])?
    
    return Ok(result)
```

### 9.2 多操作数示例

**三个数组的广播**：

```
shape_a = [2, 1, 4]
shape_b = [    4, 1]
shape_c = [    1, 1]

步骤 1: a ⊕ b
  [2, 1, 4] ⊕ [4, 1] = [2, 4, 4]
  
步骤 2: (a ⊕ b) ⊕ c
  [2, 4, 4] ⊕ [1, 1] = [2, 4, 4]

结果: [2, 4, 4]
```

**where 函数的广播**：

```rust
// where(condition, x, y)
// condition, x, y 三者形状可不同，但须可广播

let condition: Tensor2<bool> = ...;  // [M, N]
let x: Tensor1<f64> = ...;           // [N]
let y: Tensor0<f64> = ...;           // [] (标量)

// 广播后形状:
// condition: [M, N] → [M, N]
// x:         [N]    → [M, N]
// y:         []     → [M, N]

let result = where_(&condition, &x, &y)?;
// result.shape() = [M, N]
```

### 9.3 实现签名

```rust
/// 计算多个形状广播后的结果形状
///
/// 逐对广播，返回所有形状的公共广播形状。
///
/// # Arguments
/// * `shapes` - 形状列表
///
/// # Returns
/// * `Ok(result_shape)` - 公共广播形状
/// * `Err(BroadcastError)` - 某对形状不兼容
///
/// # Examples
/// ```
/// let shapes = [&[2, 1, 4][..], &[4, 1][..], &[1, 1][..]];
/// let result = broadcast_shapes(&shapes)?;
/// assert_eq!(result, vec![2, 4, 4]);
/// ```
pub fn broadcast_shapes<'a, I>(shapes: I) -> Result<SmallVec<[usize; 6]>, BroadcastError>
where
    I: IntoIterator<Item = &'a [usize]>,
{
    let mut iter = shapes.into_iter();
    let first = iter.next().ok_or(BroadcastError::EmptyShapes)?;
    
    let mut result = SmallVec::from_slice(first);
    
    for shape in iter {
        result = broadcast_shape(&result, shape)?;
    }
    
    Ok(result)
}
```

---

## 10. 错误处理

### 10.1 BroadcastError 定义

```rust
/// 广播错误
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BroadcastError {
    /// 第一个形状
    pub shape_a: SmallVec<[usize; 6]>,
    /// 第二个形状
    pub shape_b: SmallVec<[usize; 6]>,
    /// 不兼容的维度索引（从左到右，0-based）
    pub incompatible_dim: usize,
    /// 第一形状在该维度的值
    pub dim_a: usize,
    /// 第二形状在该维度的值
    pub dim_b: usize,
}

impl BroadcastError {
    /// 创建不兼容错误
    pub fn incompatible(shape_a: &[usize], shape_b: &[usize]) -> Self {
        // 找到第一个不兼容的维度
        let ndim_a = shape_a.len();
        let ndim_b = shape_b.len();
        let max_ndim = ndim_a.max(ndim_b);
        
        for k in 0..max_ndim {
            let i = ndim_a as isize - max_ndim as isize + k as isize;
            let j = ndim_b as isize - max_ndim as isize + k as isize;
            
            let dim_a = if i >= 0 { shape_a[i as usize] } else { 1 };
            let dim_b = if j >= 0 { shape_b[j as usize] } else { 1 };
            
            if dim_a != dim_b && dim_a != 1 && dim_b != 1 {
                return Self {
                    shape_a: SmallVec::from_slice(shape_a),
                    shape_b: SmallVec::from_slice(shape_b),
                    incompatible_dim: k,
                    dim_a,
                    dim_b,
                };
            }
        }
        
        // 不应该到达这里
        Self {
            shape_a: SmallVec::from_slice(shape_a),
            shape_b: SmallVec::from_slice(shape_b),
            incompatible_dim: 0,
            dim_a: 0,
            dim_b: 0,
        }
    }
}

impl std::fmt::Display for BroadcastError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "broadcast error: shapes {:?} and {:?} are incompatible at dimension {}: {} != {}",
            self.shape_a, self.shape_b, self.incompatible_dim, self.dim_a, self.dim_b
        )
    }
}

impl std::error::Error for BroadcastError {}
```

### 10.2 错误触发条件

| 条件 | 错误信息示例 |
|------|-------------|
| 对应维度不等且都不为 1 | `shapes [3, 2] and [3, 4] incompatible at dim 1: 2 != 4` |
| RHS 无法广播到 LHS（原地运算） | `RHS shape [3, 4, 5] cannot broadcast to LHS [3, 4]` |
| 多操作数中某对不兼容 | `shapes [2, 3] and [2, 4] incompatible at dim 1: 3 != 4` |

### 10.3 错误处理示例

```rust
// 用户代码
let a = Tensor::zeros([3, 2]);
let b = Tensor::zeros([3, 4]);

let result = a + b;  // 运行时 panic 或返回 Result

// 错误信息（如果使用 Result）:
// "broadcast error: shapes [3, 2] and [3, 4] are incompatible at dimension 1: 2 != 4"

// 建议:
// - 检查维度 1: 期望 2 或 1，实际 4
// - 可能需要 reshape 或 slice
```

---

## 11. 与其他模块的交互

### 11.1 与 ops 模块的交互

| 交互点 | 方向 | 说明 |
|--------|------|------|
| 运算符重载 | ops → broadcast | `Add/Sub/Mul/Div` impl 中调用 `broadcast_shape()` |
| zip_with | ops → broadcast | 二元运算前广播两个操作数 |
| 复合赋值 | ops → broadcast | `AddAssign` 等中广播 RHS |

```rust
// ops/arithmetic.rs
impl<A, D, E> Add<TensorView<'_, A, E>> for Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
    E: Dimension,
{
    type Output = Tensor<A, <D as BroadcastDim<E>>::Output>;
    
    fn add(self, rhs: TensorView<'_, A, E>) -> Self::Output {
        // 1. 计算广播后形状
        let output_shape = broadcast_shape(self.shape(), rhs.shape())
            .expect("Incompatible broadcast shapes");
        
        // 2. 广播两个操作数
        let a_broadcast = self.view().broadcast_to(&output_shape).unwrap();
        let b_broadcast = rhs.broadcast_to(&output_shape).unwrap();
        
        // 3. 逐元素加
        zip_with(&a_broadcast, &b_broadcast, |a, b| a + b)
    }
}
```

### 11.2 与 iter/zip 模块的交互

| 交互点 | 方向 | 说明 |
|--------|------|------|
| Zip 迭代器 | iter → broadcast | `Zip::and()` 支持广播视图 |
| 形状检查 | iter → broadcast | `Zip::from(a).and(b)` 检查广播兼容性 |

```rust
// iter/zip.rs
impl<'a, A, D> Zip<'a, A, D>
where
    D: Dimension,
{
    /// 添加另一个数组，支持广播
    pub fn and<B, E>(self, other: &'a TensorBase<B, E>) -> Zip<'a, (A, B), ...>
    where
        E: Dimension,
    {
        // 检查广播兼容性
        let compatible = can_broadcast(self.shape(), other.shape());
        if !compatible {
            panic!("Incompatible shapes for Zip");
        }
        
        // 内部广播后迭代
        // ...
    }
}
```

### 11.3 与 shape_ops 模块的交互

| 交互点 | 方向 | 说明 |
|--------|------|------|
| broadcast_to | shape_ops → broadcast | `TensorBase::broadcast_to()` 调用广播模块 |
| reshape 约束 | shape_ops ← broadcast | 广播视图不可 reshape（非连续） |

```rust
// shape_ops.rs
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// 广播到目标形状
    pub fn broadcast_to<E>(&self, shape: E) -> Result<TensorView<'_, A, E>, BroadcastError>
    where
        E: Dimension,
    {
        crate::broadcast::broadcast_to(self.view(), shape)
    }
}
```

### 11.4 与 layout 模块的交互

| 交互点 | 方向 | 说明 |
|--------|------|------|
| 零步长标志 | layout ← broadcast | 广播后设置 `HAS_ZERO_STRIDE` 标志 |
| 连续性检测 | layout ← broadcast | 广播视图通常不连续 |

```rust
// layout/mod.rs
impl Layout {
    /// 更新布局标志（广播后）
    pub fn update_for_broadcast(&self, new_strides: &[isize]) -> Self {
        let mut flags = self.flags;
        
        flags = flags.set_has_zero_stride(
            new_strides.iter().any(|&s| s == 0)
        );
        
        // 广播后非连续
        if flags.has_zero_stride() {
            flags = flags.set_f_contiguous(false);
            flags = flags.set_c_contiguous(false);
        }
        
        Self { flags, ..*self }
    }
}
```

---

## 12. 实现任务分解

### 任务清单

| # | 任务 | 预估时间 | 依赖 | 产出 |
|---|------|----------|------|------|
| 1 | 定义 `BroadcastError` 错误类型 | 10 min | error 模块 | `broadcast.rs` |
| 2 | 实现 `can_broadcast()` 兼容性检查 | 10 min | T1 | `broadcast.rs` |
| 3 | 实现 `broadcast_shape()` 形状推导 | 10 min | T2 | `broadcast.rs` |
| 4 | 实现 `broadcast_strides()` 步长计算 | 10 min | T3 | `broadcast.rs` |
| 5 | 实现 `broadcast_shapes()` 多操作数广播 | 10 min | T3 | `broadcast.rs` |
| 6 | 实现 `TensorView::broadcast_to()` 方法 | 15 min | T1-T4 | `broadcast.rs`, `tensor/mod.rs` |
| 7 | 集成到 ops 模块（运算符广播支持） | 15 min | T6 | `ops/arithmetic.rs` |
| 8 | 编写单元测试和文档测试 | 15 min | T1-T7 | `tests/broadcast.rs` |

**总预估时间**: 约 95 分钟（1.5 小时）

### 任务依赖图

```
T1 (BroadcastError) ──→ T2 (can_broadcast) ──→ T3 (broadcast_shape)
                                                      │
                    T4 (broadcast_strides) ←──────────┘
                              │
                              └──→ T5 (broadcast_shapes)
                                        │
                    T6 (broadcast_to) ←─┴─────────────────┐
                              │                           │
                              └──→ T7 (ops 集成) ──→ T8 (测试)
```

### 并行执行建议

- **Wave 1**: T1（可独立开始）
- **Wave 2**: T2（依赖 T1）
- **Wave 3**: T3, T4（依赖 T2，可并行）
- **Wave 4**: T5, T6（依赖 T3/T4，可并行）
- **Wave 5**: T7（依赖 T6）
- **Wave 6**: T8（依赖所有前置任务）

---

## 13. 设计决策记录

### D1: 为什么广播视图是只读的？

**决策**: `broadcast_to()` 返回 `TensorView`（只读），不提供可变广播视图。

**理由**:
1. **别名冲突**: 广播视图中多个索引指向同一物理位置，写入会导致不确定性
2. **语义模糊**: `view[[0, 0]] = x` 和 `view[[1, 0]] = y` 写入同一位置，后者覆盖前者
3. **内存安全**: 并发写入同一位置可能造成数据竞争
4. **与 NumPy 一致**: NumPy 的广播视图也是只读的

**替代方案**: 允许写入，自动传播到所有别名位置。
**拒绝原因**: 语义混乱，性能开销大，用户难以预期行为。

### D2: 为什么原地运算仅 RHS 可广播？

**决策**: `a += b` 中 `b` 可广播，`a` 必须拥有完整存储。

**理由**:
1. **写入语义**: LHS 是写入目标，不能是只读的广播视图
2. **形状不变**: 原地运算不改变 LHS 形状，避免重新分配
3. **性能**: 仅广播 RHS 避免为 LHS 创建临时副本
4. **与 NumPy 一致**: `np.ndarray.__iadd__` 行为相同

**示例**:
```rust
let mut a = Tensor::zeros([3, 4]);
let b = Tensor::zeros([4]);
a += &b;  // OK: b 广播到 [3, 4]

let c = Tensor::zeros([3, 1]);  // 广播视图
let d = Tensor::zeros([3, 4]);
// c += &d;  // 错误: c 是广播视图，不可写
```

### D3: 为什么使用零步长而非数据复制？

**决策**: 广播通过零步长实现，不复制数据。

**理由**:
1. **内存效率**: 避免为广播维度分配大量重复数据
2. **性能**: O(1) 创建广播视图，无需复制
3. **可组合**: 多次广播不产生多次复制
4. **NumPy 惯例**: NumPy 同样使用零步长实现广播

**示例对比**:
```
原始: [1, 1000] (1KB)
广播到 [1000, 1000]:
  - 数据复制: 1MB（1000 倍）
  - 零步长: 1KB + 视图元数据（~64 bytes）
```

### D4: 为什么标量广播用 mapv 而非 broadcast_to？

**决策**: `tensor + scalar` 使用 `mapv(|x| x + scalar)`，不创建广播视图。

**理由**:
1. **性能**: `mapv` 直接迭代，避免创建视图和计算索引
2. **简单性**: 标量广播是常见操作，优化路径有价值
3. **编译器优化**: `mapv` 更容易被内联和向量化

**示例**:
```rust
// 高效（推荐）
let c = &a + 5.0;  // 内部: a.mapv(|x| x + 5.0)

// 也可（但稍慢）
let scalar_view = Tensor0::from(5.0).view().broadcast_to(a.shape())?;
let c = &a + &scalar_view;
```

### D5: 为什么 BroadcastError 包含详细的维度信息？

**决策**: `BroadcastError` 包含 `incompatible_dim`, `dim_a`, `dim_b` 字段。

**理由**:
1. **调试友好**: 用户可以快速定位不兼容的维度
2. **错误信息清晰**: 自动生成 "dimension 1: 2 != 4" 这样的信息
3. **程序化处理**: 调用者可以根据错误信息调整形状

**示例错误信息**:
```
broadcast error: shapes [3, 2, 4] and [3, 4, 5] are incompatible 
at dimension 2: 4 != 5
```

### D6: 为什么多操作数广播用逐对推导？

**决策**: `broadcast_shapes()` 通过逐对广播推导结果形状。

**理由**:
1. **简单实现**: 无需复杂的多元兼容性算法
2. **NumPy 兼容**: NumPy 同样使用逐对广播
3. **确定性**: 广播顺序不影响结果（广播操作满足交换律和结合律）

**证明**: 如果 `a ⊕ b = c` 且 `c ⊕ d = e`，则 `(a ⊕ b) ⊕ d = a ⊕ (b ⊕ d) = e`。

---

## 附录 A: 广播规则速查表

### A.1 基本广播规则

| shape_a | shape_b | 兼容 | result_shape |
|---------|---------|:----:|--------------|
| `[M, N]` | `[N]` | ✓ | `[M, N]` |
| `[M, N]` | `[M, 1]` | ✓ | `[M, N]` |
| `[M, N]` | `[1, N]` | ✓ | `[M, N]` |
| `[M, N]` | `[1, 1]` | ✓ | `[M, N]` |
| `[M, N]` | `[M, N]` | ✓ | `[M, N]` |
| `[M, N]` | `[P, N]` (P ≠ M) | ✗ | - |
| `[M, N]` | `[M, P]` (P ≠ N) | ✗ | - |

### A.2 高维广播规则

| shape_a | shape_b | result_shape |
|---------|---------|--------------|
| `[A, B, C, D]` | `[D]` | `[A, B, C, D]` |
| `[A, B, C, D]` | `[C, D]` | `[A, B, C, D]` |
| `[A, B, C, D]` | `[B, C, D]` | `[A, B, C, D]` |
| `[A, B, C, D]` | `[A, B, C, D]` | `[A, B, C, D]` |
| `[A, B, C, D]` | `[1, C, 1]` | `[A, B, C, D]` |
| `[A, B, C, D]` | `[B, 1, D]` | `[A, B, C, D]` |

### A.3 标量广播规则

| 标量 | 数组 | result_shape |
|------|------|--------------|
| `x` | `[N]` | `[N]` |
| `x` | `[M, N]` | `[M, N]` |
| `x` | `[A, B, C, D]` | `[A, B, C, D]` |
| `Tensor0<A>` | 任意 | 同数组形状 |

---

## 附录 B: 零步长访问示例

### B.1 1D 广播到 2D

```
原始:
  shape: [4]
  strides: [1]
  data: [a, b, c, d]

广播到 [3, 4]:
  new_strides: [0, 1]
  
访问表:
  [0,0] → 0*0 + 0*1 = 0 → a
  [0,1] → 0*0 + 1*1 = 1 → b
  [0,2] → 0*0 + 2*1 = 2 → c
  [0,3] → 0*0 + 3*1 = 3 → d
  [1,0] → 1*0 + 0*1 = 0 → a  (第 0 行重复)
  [1,1] → 1*0 + 1*1 = 1 → b
  [1,2] → 1*0 + 2*1 = 2 → c
  [1,3] → 1*0 + 3*1 = 3 → d
  [2,0] → 2*0 + 0*1 = 0 → a  (第 0 行再次重复)
  ...
```

### B.2 2D 广播到 3D

```
原始:
  shape: [2, 3]
  strides: [1, 2]  (F-order)
  data: [a, b, c, d, e, f]
  
  逻辑布局:
    列0: a, b
    列1: c, d
    列2: e, f

广播到 [4, 2, 3]:
  new_strides: [0, 1, 2]
  
访问表 (部分):
  [0,0,0] → 0*0 + 0*1 + 0*2 = 0 → a
  [0,0,1] → 0*0 + 0*1 + 1*2 = 2 → c
  [0,0,2] → 0*0 + 0*1 + 2*2 = 4 → e
  [0,1,0] → 0*0 + 1*1 + 0*2 = 1 → b
  [0,1,1] → 0*0 + 1*1 + 1*2 = 3 → d
  [0,1,2] → 0*0 + 1*1 + 2*2 = 5 → f
  [1,0,0] → 1*0 + 0*1 + 0*2 = 0 → a  (第 0 层重复)
  [2,0,0] → 2*0 + 0*1 + 0*2 = 0 → a  (第 0 层再次重复)
  [3,1,2] → 3*0 + 1*1 + 2*2 = 5 → f
```

### B.3 标量广播到 3D

```
原始:
  shape: []  (0 维)
  strides: []
  data: [x]  (单元素)

广播到 [2, 3, 4]:
  new_strides: [0, 0, 0]
  
所有索引访问:
  [i, j, k] → i*0 + j*0 + k*0 = 0 → x
  
逻辑视图: 2×3×4 的张量，所有元素都是 x
物理存储: 1 个元素
```

---

## 附录 C: 性能考量

### C.1 广播视图 vs 数据复制

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| `broadcast_to()` | O(ndim) | O(1) 额外空间 |
| 数据复制广播 | O(elements) | O(elements) |

### C.2 零步长访问开销

| 场景 | 标量访问 | 零步长访问 |
|------|---------|-----------|
| 索引计算 | `base + sum(stride[i] * idx[i])` | 相同（乘 0 无开销） |
| 缓存行为 | 连续访问，缓存友好 | 重复访问同一地址，缓存极佳 |
| SIMD | 可向量化 | 可向量化（加载一次，广播） |

### C.3 优化建议

1. **标量广播优先用 mapv**: `a + scalar` 用 `mapv`，不创建视图
2. **批量操作合并**: 多个广播操作合并为一次 `zip_with`
3. **避免过度广播**: 广播视图不连续，可能影响后续操作性能

---

*本文档由 Senon 项目维护。如有问题请提交 Issue 或 PR。*
