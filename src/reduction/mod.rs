//! Reduction operations.
//!
//! The public API is exposed as methods on [`TensorBase`].

mod sum;
mod impls;

#[cfg(feature = "simd")]
mod sum_simd;

#[cfg(feature = "simd")]
mod driver;

pub(crate) use sum::{sum_impl, sum_axis_impl, sum_axis_keepdims_impl};
