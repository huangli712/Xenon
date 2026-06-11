//! Matrix operations.
//!
//! Exposes the vector dot product. The free function `dot_impl` backs
//! the method-style `TensorBase::dot()` API defined in `impls.rs`.

mod dot;
mod impls;

#[cfg(feature = "simd")]
mod dot_simd;

#[cfg(feature = "simd")]
mod driver;

pub(crate) use dot::dot_impl;

#[cfg(feature = "parallel")]
pub(crate) mod dot_parallel;

#[cfg(feature = "parallel")]
pub(crate) use dot::{dot_mul_step, dot_reduce_step};
