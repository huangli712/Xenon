//! Public API implementations for reduction operations.
//!
//! This file contains the `impl TensorBase` blocks that define the public
//! reduction methods, delegating to the private implementations in `sum.rs`.

use crate::dimension::{Axis, Dimension, RemoveAxis};
use crate::element::Numeric;
use crate::error::XenonError;
use crate::storage::Storage;
use crate::tensor::{Tensor, TensorBase};

// ── Public API: TensorBase::sum() ──

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Numeric + Copy + 'static,
{
    /// Returns the sum of all logical elements. See `13-reduction.md §5.1`.
    ///
    /// Empty arrays return the additive identity `A::zero()`.
    /// Rank-0 (scalar) tensors return their single element.
    /// Integer overflow is unrecoverable and panics.
    pub fn sum(&self) -> A {
        crate::reduction::sum_all(self)
    }
}

// ── Public API: TensorBase::sum_axis() ──

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension + RemoveAxis,
    A: Numeric + Copy + 'static,
{
    /// Reduces along `axis` and removes that axis from the output shape.
    /// See `13-reduction.md §5.1`.
    ///
    /// # Errors
    ///
    /// Returns `XenonError::InvalidAxis` when `axis.index() >= self.ndim()`.
    pub fn sum_axis(&self, axis: Axis) -> Result<Tensor<A, D::Smaller>, XenonError> {
        crate::reduction::sum_axis_impl(self, axis)
    }
}

// ── Public API: TensorBase::sum_axis_keepdims() ──

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Numeric + Copy + 'static,
{
    /// Reduces along `axis` and keeps the reduced axis with length 1.
    /// See `13-reduction.md §5.1`. For 0D tensors, every `axis` returns
    /// `XenonError::InvalidAxis` (no axis is valid at rank 0).
    ///
    /// # Errors
    ///
    /// Returns `XenonError::InvalidAxis` when `axis.index() >= self.ndim()`.
    pub fn sum_axis_keepdims(&self, axis: Axis) -> Result<Tensor<A, D>, XenonError> {
        crate::reduction::sum_axis_keepdims_impl(self, axis)
    }
}
