//! Layout module: F-order strides, contiguity, flags and alignment.

mod aligned;
mod contiguous;
mod flags;
mod strides;

pub use contiguous::is_f_contiguous;
pub use flags::{LayoutFlags, LayoutState};
pub use strides::Strides;

pub(crate) use flags::compute_layout_flags;
pub(crate) use strides::compute_f_strides;
