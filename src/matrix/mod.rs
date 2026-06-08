//! Matrix operations.
//!
//! Exposes the vector dot product. The free function `dot_impl` backs
//! the method-style `TensorBase::dot()` API defined in `impls.rs`.

mod dot;
mod impls;

pub(crate) use dot::dot_impl;
