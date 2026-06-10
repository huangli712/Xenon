//! Numpy-style broadcasting: shape computation, stride derivation, and
//! zero-copy broadcast view construction.

mod shape;
mod view;

pub use shape::{broadcast_shape, broadcast_strides, can_broadcast};
pub(crate) use view::broadcast_with;
