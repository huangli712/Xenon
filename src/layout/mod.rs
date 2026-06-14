//! F-order memory layout: strides, contiguity, alignment, and flags.
//!
//! The layout module models how tensor elements are arranged in memory.
//! All tensors in this crate are **Fortran-contiguous by default**:
//! the fastest-varying axis is axis 0 (stride 1), and strides increase
//! monotonically for axes with extent > 1.
//!
//! # Submodules
//!
//! | Module    | Purpose                                                  |
//! |-----------|----------------------------------------------------------|
//! | `strides` | `Strides` carrier and F-order stride computation         |
//! | `flags`   | `LayoutFlags` bitfield, `LayoutState` enum, `classify()` |
//! | `compute` | `compute_layout_flags` central entry point               |

mod flags;
mod strides;
mod compute;

pub use flags::{LayoutFlags, LayoutState};
pub use strides::Strides;
pub(crate) use compute::compute_layout_flags;
