//! Complex number type `Complex<T>` with a sealed component bound.
//!
//! W5T1 provides the minimal skeleton: the `Complex<T>` struct, its `new()`
//! constructor, and the `ComplexFloat` sealed trait with `Sealed + Copy +
//! Default` supertraits. Subsequent Wave-5 tasks extend `ComplexFloat` and
//! add arithmetic, formatting, conversion, and math methods.

mod display;
mod math;
mod ops;

mod types;

pub use types::{Complex, ComplexFloat};
