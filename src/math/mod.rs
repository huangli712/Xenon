//! Element-wise math operations (arithmetic, unary, comparison).
//!
//! Public API is exposed as inherent methods on `TensorBase` from
//! the `binary`, `unary`, and `comparison` submodules. No `pub use`
//! re-exports are needed: method visibility is governed by the
//! `impl<...> TensorBase<...>` blocks themselves.
//!

mod binary;
mod comparison;
mod unary;

pub(crate) use binary::BinaryArith;
