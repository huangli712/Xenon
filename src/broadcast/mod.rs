//! Broadcasting utilities.
//!
//! Public API list (see `15-broadcast.md §5.1`):
//! - `broadcast_shape(a, b) -> Result<IxDyn, XenonError>`
//! - `can_broadcast(a, b) -> bool`
//! - `broadcast_strides(orig_shape, orig_strides, target_shape) -> Result<Vec<usize>, XenonError>`
//! - `TensorBase::broadcast_to<E: IntoDimension>(&self, shape: E)`
//!   — inherent method on `TensorBase`, defined in `view.rs`; visible via `TensorBase`.

mod shape;
mod view;

pub use shape::{broadcast_shape, broadcast_strides, can_broadcast};
pub(crate) use view::broadcast_with;
