//! N-dimensional indexing and slicing.
//!
//! This module provides the `NdIndex` trait for multi-dimensional indexing,
//! `SliceInfo` and friends for describing tensor slices, and the inherent
//! access methods on `TensorBase` (`try_at`, `get`, `slice`, and their
//! mutable and unchecked variants).
//!
//! # Sub-modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | `ndindex` | `NdIndex` trait + tuple/`&[usize]` implementations |
//! | `slice` | `SliceInfoElem`, `SliceInfoIndices`, `SliceInfo` types |
//! | `impls` | `TensorBase` inherent methods: `try_at`, `get`, `slice` etc. |

pub mod impls;
pub mod ndindex;
pub mod slice;

pub use ndindex::NdIndex;
// pub use impls::{/* inherent methods live on TensorBase */};
pub use slice::{SliceInfo, SliceInfoElem, SliceInfoIndices};
