# FFI 接口模块设计

> 文档编号: 23 | 模块: `src/ffi.rs` | 阶段: Phase 4
> 前置文档: `07-tensor.md`, `06-memory-layout.md`
> 需求参考: 需求说明书 §25

---

## 1. 模块定位

### 1.1 职责边界

| 职责 | 包含 | 不包含 |
|------|------|--------|
| 原始指针 API | `as_ptr()`/`as_mut_ptr()` | BLAS 绑定实现（由上游库通过 `blas-sys` crate 提供） |
| 裸指针构造张量 | `from_raw_parts`/`from_raw_parts_mut` | GPU 内存操作 |
| 裸指针解构张量 | `into_raw_parts` | 跨进程共享内存 |
| BLAS 兼容性 API | `blas_layout()`/`is_blas_compatible()` | 自动调用 BLAS（由上游库负责） |
| 多维索引转换 | `offset_of()`/`ptr_at()` | 序列化/反序列化 |

### 1.2 设计原则

| 原则 | 体现 |
|------|------|
| 零拷贝 | 指针 API 无数据拷贝，O(1) 开销 |
| 安全边界清晰 | 所有 unsafe 函数有详尽 Safety 文档 |
| BLAS 友好 | 提供完整的 BLAS 兼容性检查和布局查询 |
| 最小约束 | FFI 方法避免重复安全检查（调用方已 unsafe） |

### 1.3 在架构中的位置

```
依赖层级：
L0: error, private
L1: dimension, element, complex
L2: layout (依赖 dimension)
L3: storage (依赖 layout)
L4: tensor (依赖 storage, dimension)
L5: ffi  ← 当前模块
```

---

## 2. 文件位置

```
src/
└── ffi.rs    # FFI API 实现（单文件模块）
```

单文件设计：FFI 功能紧密相关，代码量 ~300 行，无需拆分。

---

## 3. 依赖关系

### 3.1 依赖图

```
src/ffi.rs
├── crate::tensor        # TensorBase<S, D>, as_ptr(), offset()
├── crate::dimension     # Dimension trait
├── crate::storage       # Storage, StorageMut, StorageIntoRaw
├── crate::layout        # is_f_contiguous, is_c_contiguous
└── crate::error         # WorkspaceError (仅内存分配失败)
```

### 3.2 类型级依赖

| 来源模块 | 使用的类型/trait |
|----------|-----------------|
| `tensor` | `TensorBase<S, D>`, `.shape()`, `.strides()`, `.as_ptr()`, `.as_mut_ptr()`, `.offset()` |
| `dimension` | `Dimension`, `Ix0`~`Ix6`, `IxDyn` |
| `storage` | `Storage<Elem=A>`, `StorageMut<Elem=A>`, `StorageIntoRaw` |
| `layout` | `is_f_contiguous()`, `is_c_contiguous()`, `has_zero_stride()`, `has_neg_stride()` |

### 3.3 依赖方向声明

> **依赖方向：单向向上。** `ffi` 仅消费 `tensor`、`storage` 等核心模块，为上游库提供接口。

---

## 4. 公共 API 设计

### 4.1 辅助类型

```rust
/// BLAS 矩阵布局标识。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlasLayout {
    /// 列优先（Fortran order）。
    /// 对应 BLAS `CblasColMajor`（102）。
    ColumnMajor,
    /// 行优先（C order）。
    /// 对应 BLAS `CblasRowMajor`（101）。
    RowMajor,
}

/// BLAS 转置标识。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlasTrans {
    /// 不转置。
    NoTrans,
    /// 转置。
    Trans,
    /// 共轭转置（仅复数）。
    ConjTrans,
}
```

### 4.2 原始指针 API

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// 返回数据起始位置的只读原始指针。
    ///
    /// 指针指向第一个逻辑元素（考虑 offset）。
    /// 返回的指针在 `self` 被修改或释放后无效。
    ///
    /// # Example
    ///
    /// ```
    /// let tensor = Tensor2::<f64>::zeros([3, 4]);
    /// let ptr = tensor.as_ptr();
    /// // 可传递给只读 C 函数
    /// ```
    pub fn as_ptr(&self) -> *const A {
        unsafe {
            self.storage.as_ptr().add(self.offset)
        }
    }
}

impl<S, D, A> TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    D: Dimension,
{
    /// 返回数据起始位置的可变原始指针。
    ///
    /// 仅对可写存储（Owned、ViewMut、ArcRepr）可用。
    ///
    /// # Example
    ///
    /// ```
    /// let mut tensor = Tensor2::<f64>::zeros([3, 4]);
    /// let ptr = tensor.as_mut_ptr();
    /// // 可传递给需要可变指针的 C 函数
    /// ```
    pub fn as_mut_ptr(&mut self) -> *mut A {
        unsafe {
            self.storage.as_mut_ptr().add(self.offset)
        }
    }
}
```

### 4.3 从裸指针构造张量

```rust
impl<'a, A, D> TensorBase<ViewRepr<&'a A>, D>
where
    D: Dimension,
{
    /// 从裸指针构造不可变视图。
    ///
    /// # Arguments
    ///
    /// * `ptr` - 数据起始指针（不可变）
    /// * `shape` - 各轴长度
    /// * `strides` - 各轴步长（元素单位，有符号）
    /// * `offset` - 数据起始偏移量（元素单位）
    ///
    /// # Returns
    ///
    /// 新的 `TensorView<'a, A, D>` 实例。
    ///
    /// # Safety
    ///
    /// 调用方须保证以下所有条件：
    ///
    /// | 前提条件 | 说明 |
    /// |----------|------|
    /// | 指针有效性 | `ptr` 须非空、非悬垂，且对齐到 `align_of::<A>()` |
    /// | 内存范围 | `ptr` 起始的内存须覆盖所有可访问元素（考虑 offset、shape、strides）） |
    /// | 生命周期 | 内存须在生命周期 `'a` 内保持有效 |
    /// | 别名规则 | 内存可被共享读取，但不可被写入 |
    /// | 布局一致性 | `shape` 与 `strides` 长度须一致 |
    /// | 元素初始化 | 所有可访问元素须已正确初始化 |
    ///
    /// # Example
    ///
    /// ```
    /// let data: [f64; 12] = [0.0; 12];
    /// let view = unsafe {
    ///     TensorView2::from_raw_parts(
    ///         data.as_ptr(),
    ///         [3, 4],
    ///         [1, 3],
    ///         0,
    ///     )
    /// };
    /// ```
    pub unsafe fn from_raw_parts(
        ptr: *const A,
        shape: D,
        strides: D,
        offset: usize,
    ) -> Self {
        // 实现...
    }
}

impl<'a, A, D> TensorBase<ViewMutRepr<&'a mut A>, D>
where
    D: Dimension,
{
    /// 从裸指针构造可变视图。
    ///
    /// 与 `from_raw_parts` 相同，但要求独占访问（无其他引用）。
    ///
    /// # Safety
    ///
    /// 与 `from_raw_parts` 相同,但额外要求：内存无其他引用。
    ///
    /// # Example
    ///
    /// ```
    /// let mut data: [f64; 12] = [0.0; 12];
    /// let view = unsafe {
    ///     TensorViewMut2::from_raw_parts_mut(
    ///         data.as_mut_ptr(),
    ///         [3, 4],
    ///         [1, 3],
    ///         0,
    ///     )
    /// };
    /// ```
    pub unsafe fn from_raw_parts_mut(
        ptr: *mut A,
        shape: D,
        strides: D,
        offset: usize,
    ) -> Self {
        // 实现...
    }
}
```

### 4.4 将张量解构为裸指针

```rust
impl<A, D> TensorBase<Owned<A>, D>
where
    D: Dimension,
{
    /// 消费张量，返回原始部件。
    ///
    /// 调用方负责释放返回的内存。
    ///
    /// # Returns
    ///
    /// 元组 `(ptr, shape, strides, offset)`。
    ///
    /// # Example
    ///
    /// ```
    /// let tensor = Tensor2::<f64>::zeros([3, 4]);
    /// let (ptr, shape, strides, offset) = tensor.into_raw_parts();
    /// // 调用方现在拥有 ptr，负责释放
    /// ```
    pub fn into_raw_parts(self) -> (*mut A, D, D, usize) {
        let ptr = self.storage.into_raw();
        let shape = self.shape;
        let strides = self.strides;
        let offset = self.offset;

        // 防止 Drop 释放内存
        core::mem::forget(self);

        (ptr, shape, strides, offset)
    }
}
```

> **设计决策：** `into_raw_parts` 仅适用于 Owned 存储。View/ViewMut 的数据仍由原借用绑定，调用方应谨慎处理。

### 4.5 BLAS 兼容性 API

```rust
impl<S, D> TensorBase<S, D>
where
    S: Storage,
    D: Dimension,
{
    /// 检查内存布局是否可直接传递给 BLAS。
    ///
    /// # BLAS 兼容性条件
    ///
    /// | 条件 | 说明 |
    /// |------|------|
    /// | 连续性 | F-contiguous 或 C-contiguous |
    /// | 正步长 | 所有步长 > 0（无反转维度） |
    /// | 无零步长 | 无广播维度 |
    ///
    /// # Returns
    ///
    /// `true` 表示可直接传递给 BLAS；`false` 表示需先复制。
    ///
    /// # Example
    ///
    /// ```
    /// let a = Tensor2::<f64>::zeros([3, 4]);
    /// assert!(a.is_blas_compatible());
    ///
    /// let b = a.slice(s![.., 1..3]);
    /// assert!(!b.is_blas_compatible());
    /// ```
    pub fn is_blas_compatible(&self) -> bool {
        self.is_contiguous()
            && !self.has_zero_stride()
            && !self.has_neg_stride()
    }
}
```

### 4.6 blas_layout 和 BlasLayout 结构体

```rust
/// BLAS 矩阵信息。
///
/// 包含传递给 BLAS 函数所需的全部参数。
pub struct BlasInfo {
    /// 数据指针。
    pub data_ptr: *const u8,
    /// Leading dimension（元素单位）。
    pub leading_dim: i32,
    /// 行数。
    pub rows: i32,
    /// 列数。
    pub cols: i32,
}

impl<S, D> TensorBase<S, D>
where
    S: Storage,
    D: Dimension,
{
    /// 返回 BLAS 布局标识及参数信息。
    ///
    /// # Returns
    ///
    /// - `Some(BlasInfo)`：兼容条件满足
    /// - `None`：不兼容 BLAS
    ///
    /// # Example
    ///
    /// ```
    /// let a = Tensor2::<f64>::zeros([3, 4]);
    /// let info = a.blas_info().unwrap();
    /// assert_eq!(info.rows, 3);
    /// assert_eq!(info.cols, 4);
    /// ```
    pub fn blas_info(&self) -> Option<BlasInfo> {
        if !self.is_blas_compatible() || self.ndim() != 2 {
            return None;
        }

        let data_ptr = self.as_ptr() as *const u8;
        let lda = self.lda()? as i32;
        let rows = self.shape()[0] as i32;
        let cols = self.shape()[1] as i32;

        Some(BlasInfo {
            data_ptr,
            leading_dim: lda,
            rows,
            cols,
        })
    }
}
```

### 4.7 LDA 查询

```rust
impl<S, D> TensorBase<S, D>
where
    S: Storage,
    D: Dimension,
{
    /// 返回 leading dimension（仅 2D 数组有意义）。
    ///
    /// 对于 F-order 矩阵 `A[M, N]`，`LDA = stride[1]`。
    ///
    /// # Returns
    ///
    /// - `Some(isize)`: 2D 数组的 LDA
    /// - `None`: 非 2D 数组
    ///
    /// # Example
    ///
    /// ```
    /// let a = Tensor2::<f64>::zeros([3, 4]);
    /// assert_eq!(a.lda(), Some(3));  // F-order, LDA = M = 3
    /// ```
    pub fn lda(&self) -> Option<isize> {
        if self.ndim() != 2 {
            return None;
        }
        let strides = self.strides();
        Some(strides[1] as isize)
    }
}
```

### 4.8 多维索引到指针偏移

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// 将多维索引转换为元素偏移量。
    ///
    /// 偏移量 = Σ(stride[i] * index[i]) for all i in [0, ndim)
    ///
    /// # Panics
    ///
    /// - 索引长度与维度数不匹配
    /// - 索引越界
    ///
    /// # Example
    ///
    /// ```
    /// let tensor = Tensor2::<f64>::zeros([3, 4]);
    /// // shape=[3,4], strides=[1,3], F-order
    /// // index [1, 2] → offset = 1*1 + 2*3 = 7
    /// assert_eq!(tensor.offset_of(&[1, 2]), 7);
    /// ```
    pub fn offset_of(&self, index: &[usize]) -> usize {
        assert!(index.len() == self.ndim(), "index dimension mismatch");

        let shape = self.shape();
        let strides = self.strides();
        let mut offset: usize = 0;
        for i in 0..self.ndim() {
            assert!(index[i] < shape[i], "index out of bounds");
            offset += index[i] * (strides[i] as usize);
        }
        offset
    }

    /// 将多维索引转换为对应元素的原始指针。
    ///
    /// # Panics
    ///
    /// - 索引长度与维度数不匹配
    /// - 索引越界
    ///
    /// # Example
    ///
    /// ```
    /// let tensor = Tensor1::<i32>::from_vec(vec![10, 20, 30, 40]);
    /// let ptr = tensor.ptr_at(&[2]);
    /// assert_eq!(unsafe { *ptr }, 30);
    /// ```
    pub fn ptr_at(&self, index: &[usize]) -> *const A {
        let offset = self.offset_of(index);
        // SAFETY: offset_of guarantees valid access within storage bounds
        unsafe { self.as_ptr().add(offset) }
    }
}
```

### 4.9 Good/Bad 对比

```rust
// Good - 使用 BLAS 兼容性检查后再传递
if tensor.is_blas_compatible() {
    let info = tensor.blas_info().unwrap();
    unsafe {
        call_blas_dgemm(info, tensor.blas_trans(), ...);
    }
} else {
    let contiguous = tensor.to_f_contiguous();
    let info = contiguous.blas_info().unwrap();
    unsafe {
        call_blas_dgemm(info, contiguous.blas_trans(), ...);
    }
}

// Bad - 不检查 BLAS 兼容性，直接传递
unsafe {
    call_blas_dgemm(CblasColMajor, CblasNoTrans, ...,
        tensor.as_ptr(), tensor.lda().unwrap(),
        ...,
    );  // UB if tensor is non-contiguous!
}
```

---

## 5. 内部实现设计

### 5.1 指针有效性论证

`as_ptr()` 和 `as_mut_ptr()` 的返回值有效性由 Rust 借用检查器保证（`NonNull` 指针在 `Owned` 枃持下）。

对于 View 类型，`offset` 在 storage 范围内则结果合法。数据来自原始 Tensor 的 storage，生命周期由原始引用保证。

`from_raw_parts` 的 Safety 由调用方保证：所有前提条件文档为运行时"契约"，违反任何条件都将导致未定义行为。

### 5.2 BLAS 兼容性检查流程

```
is_blas_compatible():
    │
    ├── is_contiguous()? ─── No ──→ false
    │
    ├── has_zero_stride()? ── Yes ──→ false
    │
    └── has_neg_stride()? ── Yes ──→ false
    │
    └── All passed ────────────────→ true
```

---

## 6. 实现任务拆分

### Wave 1: 基础设施

- [ ] **T1**: 定义 `BlasLayout`/`BlasTrans` 枚举和 `BlasInfo` 结构体
  - 文件: `src/ffi.rs`
  - 内容: 枚举定义、常量映射
  - 测试: `test_blas_layout_column_major`, `test_blas_trans_variants`
  - 前置: 无
  - 预计: 10 min

### Wave 2: 指针 API

- [ ] **T2**: 实现 `as_ptr()` 和 `as_mut_ptr()`
  - 文件: `src/ffi.rs`
  - 内容: 指针访问方法（含 offset 计算）
  - 测试: `test_as_ptr_basic`, `test_as_mut_ptr_basic`
  - 前置: T1
  - 预计: 10 min

### Wave 3: 构造/解构

- [ ] **T3**: 实现 `from_raw_parts` 和 `from_raw_parts_mut` 及 Safety 文档
  - 文件: `src/ffi.rs`
  - 内容: 从裸指针构造视图
  - 测试: `test_from_raw_parts_roundtrip`, `test_from_raw_parts_mut_roundtrip`
  - 前置: T2
  - 预计: 15 min

- [ ] **T4**: 实现 `into_raw_parts`
  - 文件: `src/ffi.rs`
  - 内容: 张量解构（Owned only）+ `mem::forget`
  - 测试: `test_into_raw_parts`, `test_into_raw_parts_memory_leak`
  - 前置: T2
  - 预计: 10 min

### Wave 4: BLAS 和索引

- [ ] **T5**: 实现 BLAS 兼容性 API（`is_blas_compatible`/`blas_info`/`lda`）
  - 文件: `src/ffi.rs`
  - 内容: BLAS 检查和参数查询
  - 测试: `test_is_blas_compatible_f_order`, `test_is_blas_compatible_non_contiguous`, `test_lda_f_order`
  - 前置: T1
  - 预计: 15 min

- [ ] **T6**: 实现 `offset_of()` 和 `ptr_at()`
  - 文件: `src/ffi.rs`
  - 内容: 多维索引到偏移量和指针转换
  - 测试: `test_offset_of_various`, `test_ptr_at_various`
  - 前置: T2
  - 预计: 10 min

### 并行执行图

```
Wave 1: [T1]
            │
Wave 2: [T2]
            │
Wave 3: [T3] [T4]
            │
Wave 4: [T5] [T6]
```

---

## 7. 测试计划

### 7.1 单元测试清单

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_as_ptr_basic` | `as_ptr()` 返回有效指针 | 高 |
| `test_as_mut_ptr_basic` | `as_mut_ptr()` 返回有效可写指针 | 高 |
| `test_as_ptr_offset` | 指针考虑 offset 后指向正确元素 | 高 |
| `test_is_blas_compatible_f_order` | F-order 连续数组兼容 | 高 |
| `test_is_blas_compatible_c_order` | C-order 连续数组兼容 | 高 |
| `test_is_blas_compatible_non_contiguous` | 非连续切片不兼容 | 高 |
| `test_is_blas_compatible_broadcast` | 广播维度（零步长）不兼容 | 高 |
| `test_is_blas_compatible_flipped` | 负步长（翻转）不兼容 | 高 |
| `test_blas_info_f_order` | F-order 返回正确 BlasInfo | 高 |
| `test_lda_f_order` | F-order [3,4] 返回 3 | 高 |
| `test_lda_c_order` | C-order [3,4] 返回 4 | 中 |
| `test_from_raw_parts_roundtrip` | 构造 → 读取一致性 | 高 |
| `test_from_raw_parts_mut_roundtrip` | 可变构造 → 修改 → 读取 | 高 |
| `test_into_raw_parts` | Owned 张量解构后指针有效 | 高 |
| `test_into_raw_parts_memory_leak` | 解构后正确释放 | 中 |
| `test_offset_of_various` | 各种索引的偏移量正确性 | 高 |
| `test_ptr_at_various` | 各种索引的指针正确性 | 高 |

### 7.2 边界测试场景

| 场景 | 预期行为 |
|------|----------|
| 空张量 | `as_ptr()` 返回非空但不应解引用 |
| 单元素张量 | `as_ptr()` 指向唯一元素 |
| 非连续切片 | `is_blas_compatible()` 返回 `false` |
| 广播维度 | `is_blas_compatible()` 返回 `false` |
| 1D 张量 | `lda()` 返回 `None` |
| 零维张量 | `offset_of(&[])` 返回 0 |
| 未对齐指针 | `from_raw_parts` 的 Safety 文档需说明对齐要求 |

### 7.3 内存安全测试

| 场景 | 验证方式 |
|------|----------|
| `from_raw_parts` + Drop | 无内存泄漏（借用语义） |
| `into_raw_parts` + 手动释放 | 正确释放（通过 allocator API） |
| `from_raw_parts` 野指针 | AddressSanitizer 检测 |

---

## 8. 与其他模块的交互

| 交互点 | 方向 | 说明 |
|--------|------|------|
| 指针访问 | ffi → tensor | 通过 `TensorBase` 的 storage 获取指针 |
| BLAS 检查 | ffi ← layout | 使用 `is_contiguous()`、`has_zero_stride()`、`has_neg_stride()` |
| 解构 | ffi → storage | `into_raw_parts` 使用 `StorageIntoRaw` trait |
| BLAS 参数 | 上游库 ← ffi | 上游 BLAS 库调用 `blas_info()`、`lda()` 等获取参数 |

---

## 9. 设计决策记录(ADR)

### 决策 1: BLAS 兼容 API 设计

| 属性 | 值 |
|------|-----|
| 决策 | 提供结构化的 `BlasInfo` 查询方法，而非仅返回布尔值 |
| 理由 | 上游库需要完整的 BLAS 参数（data ptr、lda、rows、cols），结构体返回比单独方法调用更便捷 |
| 替代方案 | 仅返回 `bool is_blas_compatible()` — 放弃，上游库需要重复获取多个参数 |
| 替代方案 | 返回 raw C 常量 — 放弃，不符合 Rust 惯例 |

### 决策 2: Safety 独立边界

| 属性 | 值 |
|------|-----|
| 决策 | `from_raw_parts` 和 `from_raw_parts_mut` 使用最小 Safety 模约束集 |
| 理由 | 将安全责任尽可能交给调用方，库本身不做额外假设；与 `std::slice::from_raw_parts` 设计一致 |
| 替代方案 | 库内部验证所有 Safety 条件 — 放弃，运行时开销过大（O(n) 检查） |

### 决策 3: 性能 — 零拷贝优先

| 属性 | 值 |
|------|-----|
| 决策 | FFI 方法避免重复安全检查 |
| 理由 | FFI 的核心价值是零开销；重复检查会增加不必要的运行时开销；调用方已在 unsafe 块中，可自行决定安全级别 |
| 替代方案 | 每次调用都检查连续性 — 放弃，与零开销目标冲突 |

---

## 10. 性能考量

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| `as_ptr()` | O(1) | 仅指针加法 |
| `as_mut_ptr()` | O(1) | 仅指针加法 |
| `is_blas_compatible()` | O(1) | 检查布局标志 |
| `blas_info()` | O(1) | 包含 `is_blas_compatible()` + 构造 |
| `lda()` | O(1) | 步长查询 |
| `offset_of()` | O(ndim) | 逐轴计算 |
| `ptr_at()` | O(ndim) | `offset_of()` + 指针加法 |
| `from_raw_parts()` | O(1) | 仅构造视图 |
| `into_raw_parts()` | O(1) | 提取字段 + `forget` |

**性能提示**:

- `as_ptr()` 和 `as_mut_ptr()` 应标注 `#[inline]`
- `offset_of()` 在热路径中可能需要内联
- `is_blas_compatible()` 检查现有 `LayoutFlags`，无需重新计算

---

## 版本历史

| 版本 | 日期 |
|------|------|
| 1.0.0 | 2026-04-07 |

---

*本文档由 Xenon 维护。如有问题请提交 Issue 或 PR。*
