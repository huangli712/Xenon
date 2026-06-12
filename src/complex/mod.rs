//! Complex number type `Complex<T>` with a sealed component bound.
//!
//! The module provides the `Complex<T>` struct, its constructors and
//! accessors, and the `ComplexFloat` sealed trait exclusive to `f32`
//! and `f64`. Arithmetic operations, formatting, and math methods
//! are implemented in sub-modules.

mod types;
mod ops;
mod math;
mod display;

pub use types::{Complex, ComplexFloat};
