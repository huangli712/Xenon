//! Element-wise math operations: arithmetic, unary, and comparison.
//!
//! Public API is exposed as inherent methods on `TensorBase` from the
//! `binary`, `unary`, and `compare` submodules. No `pub use` re-exports are
//! needed: method visibility is governed by the `impl<...> TensorBase<...>`
//! blocks themselves.

mod binary;
mod unary;
mod compare;

#[cfg(feature = "simd")]
mod types;

#[cfg(feature = "simd")]
mod binary_simd;

#[cfg(feature = "simd")]
mod unary_simd;

#[cfg(feature = "simd")]
mod driver;

pub(crate) use binary::BinaryArith;
