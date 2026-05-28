//! F-order memory layout: strides, contiguity, alignment, and flags.
//!
//! The layout module models how tensor elements are arranged in memory.
//! All tensors in this crate are **Fortran-contiguous by default**:
//! the fastest-varying axis is axis 0 (stride 1), and strides increase
//! monotonically for axes with extent > 1.
//!
//! # Submodules
//!
//! | Module      | Purpose |
//! |-------------|---------|
//! | `strides`   | [`Strides`] carrier and F-order stride computation |
//! | `contiguous`| `is_f_contiguous` recognition algorithm |
//! | `aligned`   | Raw-pointer alignment checks |
//! | `flags`     | [`LayoutFlags`] bitfield, [`LayoutState`] enum, and the central `compute_layout_flags` entry point |
//!
//! # Public API
//!
//! - [`Strides`] — stride storage and helper methods
//! - `is_f_contiguous` — check F-contiguity from raw shape + strides
//! - [`LayoutFlags`] — packed contiguity/alignment/broadcast flags
//! - [`LayoutState`] — three-way classification: `FContiguous`, `NonContiguous`, `BroadcastView`

mod aligned;
mod contiguous;
mod flags;
mod strides;

pub use contiguous::is_f_contiguous;
pub use flags::{LayoutFlags, LayoutState};
pub use strides::Strides;

pub(crate) use flags::compute_layout_flags;
