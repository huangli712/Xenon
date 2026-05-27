//! Complex number type `Complex<T>` with a sealed component bound.
//!
//! The module provides the `Complex<T>` struct, its constructors and
//! accessors, and the `ComplexFloat` sealed trait exclusive to `f32`
//! and `f64`. Arithmetic operations, formatting, and math methods
//! are implemented in sub-modules.

mod display;
mod math;
mod ops;

mod types;

pub use types::{Complex, ComplexFloat};
