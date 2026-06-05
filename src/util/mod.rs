//! Element-wise and layout utility operations on tensors.
//!
//! This module provides inherent methods on [`TensorBase`] for common
//! element-wise and memory-layout operations:
//!
//! | Operation          | Description                                      |
//! |--------------------|--------------------------------------------------|
//! | [`clip`]           | Clamp elements into `[min, max]`.                |
//! | [`fill`]           | Write `value` to all logical elements in place.  |
//! | [`try_fill`]       | Fallible variant of `fill`.                      |
//! | [`to_contiguous`]  | Produce a canonical F-order copy.                |
//! | [`into_contiguous`]| Convert into a canonical F-order owned tensor.   |
//!
//! [`TensorBase`]: crate::tensor::TensorBase
//! [`clip`]: crate::tensor::TensorBase::clip
//! [`fill`]: crate::tensor::TensorBase::fill
//! [`try_fill`]: crate::tensor::TensorBase::try_fill
//! [`to_contiguous`]: crate::tensor::TensorBase::to_contiguous
//! [`into_contiguous`]: crate::tensor::TensorBase::into_contiguous

mod impls;
