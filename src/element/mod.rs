//! Element type hierarchy — closed set of 7 types with sealed traits.
//!
//! The module defines the `Element` base trait, `ElementType` discriminant,
//! and specialised sub-traits (`Numeric`, `RealScalar`, `ComplexScalar`,
//! `CastTo`, `CheckedAdd`, etc.). Arithmetic contracts, type conversion,
//! and concrete primitive implementations are provided by sub-modules.

mod types;
mod order;
mod checked;

mod primitives;
mod numeric;
mod real;
mod complex;

pub(crate) use types::{ElementType, element_type_of};
pub(crate) use order::OrderedCompareElement;
pub(crate) use checked::{CheckedAdd, CheckedDiv, CheckedMul, CheckedNeg, CheckedSub};

pub use primitives::Element;
pub use numeric::Numeric;
pub use real::RealScalar;
pub use complex::ComplexScalar;
