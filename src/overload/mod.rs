//! Operator overloading for `Tensor` / `TensorView` arithmetic.
//!
 //! Public entry per `19-overload.md §5`. Actual `impl` blocks live in
 //! `owned` and `view` sub-modules and are exposed through Rust's usual
 //! trait-impl visibility; only `Scalar` needs to be named explicitly by
 //! user code.

pub mod scalar;
pub mod add;
pub mod sub;
pub mod mul;
pub mod div;
pub mod view;

pub use scalar::Scalar;
