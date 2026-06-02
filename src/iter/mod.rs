//! Tensor iterators.
//!
//! This module provides the public iterator surface area for tensors:
//! - `Iter` / `IterMut`: flat element iterators.
//! - `AxisIter` / `AxisIterMut`: axis-wise sub-view iterators.
//! - `IndexedIter` / `IndexedIterMut`: index-paired iterators.
//!
//! Entry methods on `TensorBase` provide access to these iterators.

mod axis;
mod primitives;
mod types;
mod indexed;
mod impls;

pub use axis::{AxisIter, AxisIterMut};
pub use primitives::{Iter, IterMut};
pub use indexed::{IndexedIter, IndexedIterMut};
