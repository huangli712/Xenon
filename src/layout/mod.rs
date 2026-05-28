//! Layout module: F-order strides, contiguity, flags and alignment.
//!
//! See `docs/design/06-layout.md`.

mod aligned;
mod contiguous;
mod flags;
mod strides;

pub use contiguous::is_f_contiguous;
pub use flags::{LayoutFlags, LayoutState};
pub(crate) use flags::compute_layout_flags;
pub use strides::{Strides, compute_f_strides};
