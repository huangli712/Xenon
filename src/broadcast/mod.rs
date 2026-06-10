//! Numpy-style broadcasting: shape computation, stride derivation, and
//! zero-copy broadcast view construction.

mod shape;
mod view;

pub use shape::{can_broadcast, broadcast_shape, broadcast_strides};
pub(crate) use view::broadcast_with;
