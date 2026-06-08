//! Tensor construction methods.
//!
//! Factory methods for creating owned tensors with common initialization
//! patterns. All constructors return `Result<Tensor, XenonError>` to enforce
//! shape validation at construction time.
//!
//! ## Constructor overview
//!
//! | Method | Input | Returns | Description |
//! |--------|-------|---------|-------------|
//! | `zeros` | shape | `Tensor<A, D>` | All elements zero |
//! | `ones` | shape | `Tensor<A, D>` | All elements one |
//! | `eye` | `n: usize` | `Tensor<A, Ix2>` | n×n identity matrix |
//! | `from_shape_vec` | shape + `Vec<A>` | `Tensor<A, D>` | Consumes data |
//! | `from_vec` | `Vec<A>` | `Tensor<A, Ix1>` | 1-D, shape inferred |
//! | `from_shape_slice` | shape + `&[A]` | `Tensor<A, D>` | Copies data |
//! | `from_array` | shape + `[A; N]` | `Tensor<A, D>` | Fixed-size input |
//! | `from_scalar` | `A` | `Tensor<A, Ix0>` | 0-dimensional scalar |
//!
//! ## Error types
//!
//! - `InvalidShapeKind::ProductOverflow` — shape element count overflows
//! - `InvalidShapeKind::ElementCountMismatch` — data length ≠ expected size
//! - `AllocationFailed` — underlying allocator cannot satisfy request

pub mod impls;
