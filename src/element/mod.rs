//! Element type hierarchy — closed set of 7 types with sealed traits.
//!
//! The module defines the `Element` base trait, `ElementType` discriminant,
//! and specialised sub-traits (`Numeric`, `RealScalar`, `ComplexScalar`,
//! `CastTo`, `CheckedAdd`, etc.). Arithmetic contracts, type conversion,
//! and concrete primitive implementations are provided by sub-modules.

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
