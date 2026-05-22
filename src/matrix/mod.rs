//! Matrix operations.
//!
//! Currently this module exposes the vector dot product. The
//! method-style `TensorBase::dot()` mirror is added by W17T3 once the
//! scalar inner product is implemented.
//!
//! Design reference: 12-matrix §3, §5.1.

mod dot;

pub use dot::dot;
