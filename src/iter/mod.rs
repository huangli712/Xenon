//! Tensor iterators.
//!
//! This module provides the public iterator surface area for tensors:
//! - `Iter` / `IterMut`: flat element iterators.
//! - `AxisIter` / `AxisIterMut`: axis-wise sub-view iterators.
//! - `IndexedIter` / `IndexedIterMut`: index-paired iterators.
//!
//! Entry methods on `TensorBase` provide access to these iterators.

mod primitives;
mod indexed;
mod axis;
mod types;
mod impls;

pub use primitives::{Iter, IterMut};
pub use indexed::{IndexedIter, IndexedIterMut};
pub use axis::{AxisIter, AxisIterMut};
