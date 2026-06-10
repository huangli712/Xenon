//! Broadcasting utilities.
//!
//! Public API list:
//! - `broadcast_shape(a, b) -> Result<IxDyn, XenonError>`
//! - `can_broadcast(a, b) -> bool`
//! - `broadcast_strides(orig_shape, orig_strides, target_shape) -> Result<Vec<usize>, XenonError>`
//! - `broadcast_with` (pub(crate) — shared broadcast prologue for math operators)

mod shape;
mod view;

pub use shape::{broadcast_shape, broadcast_strides, can_broadcast};
pub(crate) use view::broadcast_with;