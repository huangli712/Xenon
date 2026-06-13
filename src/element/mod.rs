//! Element type hierarchy — closed set of 7 types with sealed traits.
//!
//! # Supported element types
//!
//! The closed set of element types consists of 7 members:
//!
//! | Type | `Element` | `Numeric` | `RealScalar` | `ComplexScalar` |
//! |------|-----------|-----------|--------------|-----------------|
//! | `i32` | ✓ | ✓ | | |
//! | `i64` | ✓ | ✓ | | |
//! | `f32` | ✓ | ✓ | ✓ | |
//! | `f64` | ✓ | ✓ | ✓ | |
//! | `Complex<f32>` | ✓ | ✓ | | ✓ |
//! | `Complex<f64>` | ✓ | ✓ | | ✓ |
//! | `bool` | ✓ | | | |
//!
//! # `usize` is NOT an element type
//!
//! `usize` is reserved for indexing, shape metadata, and dimension
//! expressions. It does not implement `Element` because:
//!
//! * It lacks an additive inverse (no negative values), preventing
//!   it from forming the algebraic structure required by `Element`.
//! * `Element` types require `zero()` and `one()` identities; `usize`
//!   has no consistent negation semantics in this context.
//!
//! # Sub-modules
//!
//! | Sub-module | Contents |
//! |------------|----------|
//! | `primitives` | `Element` trait + impls for all 7 types |
//! | `types` | `ElementType` discriminant + free functions |
//! | `numeric` | `Numeric` trait + impls |
//! | `real` | `RealScalar` trait + impls |
//! | `complex` | `ComplexScalar` trait + impls |
//! | `order` | `OrderedCompareElement` marker trait |
//! | `simd` | `SimdElement` marker trait |
//! | `sealed` | `SealedElement` marker trait (`eye`/`unique`/`cast`) |
//! | `checked` | Checked arithmetic traits |

mod types;
mod order;
mod simd;
mod sealed;
mod checked;

mod primitives;
mod numeric;
mod real;
mod complex;

pub(crate) use types::element_type_of;
pub(crate) use order::OrderedCompareElement;
pub(crate) use simd::SimdElement;
pub(crate) use sealed::SealedElement;
pub(crate) use checked::{CheckedAdd, CheckedDiv, CheckedMul, CheckedNeg, CheckedSub};

pub use types::ElementType;
pub use primitives::Element;
pub use numeric::Numeric;
pub use real::RealScalar;
pub use complex::ComplexScalar;
