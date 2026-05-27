//! Element type hierarchy — closed set of 7 types with sealed traits.
//!
//! The module defines the `Element` base trait, `ElementType` discriminant,
//! and specialised sub-traits (`Numeric`, `RealScalar`, `ComplexScalar`,
//! `CastTo`, `CheckedAdd`, etc.). Arithmetic contracts, type conversion,
//! and concrete primitive implementations are provided by sub-modules.

mod types;
mod primitives;
mod numeric;
mod real;
mod complex;
mod order;
mod checked;

pub use types::{ElementType, element_type_name_of, element_type_of};
pub use primitives::Element;
pub use numeric::Numeric;
pub use real::RealScalar;
pub use complex::ComplexScalar;
pub use order::OrderedCompareElement;
pub(crate) use checked::{CheckedAdd, CheckedDiv, CheckedMul, CheckedNeg, CheckedSub};
