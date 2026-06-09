//! Operator overloading for `Tensor` / `TensorView` arithmetic.
//!
//! Actual `impl` blocks live in the `add`, `sub`, `mul`, `div`
//! sub-modules and are exposed through Rust's usual trait-impl
//! visibility. Each module covers both owned `Tensor` operations
//! and `TensorView` cross-combinations. Only [`Scalar`] needs to
//! be named explicitly by user code.

pub mod scalar;
pub mod add;
pub mod sub;
pub mod mul;
pub mod div;

pub use scalar::Scalar;
