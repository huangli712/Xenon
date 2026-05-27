//! Element type hierarchy.

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
