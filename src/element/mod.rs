//! Element type hierarchy: base traits, type discriminants, and arithmetic
//! contracts for Xenon's closed set of 7 element types.
//!
//! # Organisation
//!
//! | Submodule | Contents |
//! |-----------|----------|
//! | [`types`] | `Element`, `ElementType`, `CastTo`, `CastElement`, `OrderedCompareElement`, `BoolElement` |
//! | [`numeric`](numeric) | `Numeric` — arithmetic operators + `conjugate()` |
//! | [`real`](real) | `RealScalar` — IEEE‑754 math functions (`abs`, `sqrt`, `sin`, …) |
//! | [`complex`](complex) | `ComplexScalar` — complex component accessors |
//! | [`checked`](checked) | `CheckedAdd` / `CheckedSub` / … — integer overflow‑safe arithmetic |
//! | [`primitives`] | Concrete impls of all element traits for `i32`, `i64`, `f32`, `f64`, `bool`, `Complex<f32>`, `Complex<f64>` |
//!
//! See [`types`] for the detailed rationale: why `usize` is excluded, the cast
//! error semantics, and the `bool` exclusion from `CastTo<T>`.

mod types;
pub use types::{
    CastElement, CastTo, Element, ElementType, OrderedCompareElement, element_type_name_of,
    element_type_of,
};
pub(crate) use types::BoolElement;

mod numeric;
pub use numeric::Numeric;

mod real;
pub use real::RealScalar;

mod complex;
pub use complex::ComplexScalar;

/// Element trait implementations for standard numeric types.
pub mod primitives;

mod checked;
pub(crate) use checked::{CheckedAdd, CheckedDiv, CheckedMul, CheckedNeg, CheckedSub};
