//! Tensor iterators.
//!
//! Defines the public iterator surface area as specified in `10-iterator.md §5`:
//! - `Iter` / `IterMut` (flat element iterators, §5.1, wired up in W12T3 / W12T4)
//! - `AxisIter` / `AxisIterMut` (axis-wise sub-view iterators, §5.2, wired up in W12T5)
//! - `IndexedIter` / `IndexedIterMut` (index-paired iterators, §5.4, wired up in W12T6)
//!
//! Entry methods on `TensorBase` (§5.5) are added in W12T7.

mod axis;
mod primitives;
mod types;
mod indexed;
mod impls;

pub use axis::{AxisIter, AxisIterMut};
pub use primitives::Iter;
pub use primitives::IterMut;
pub use indexed::{IndexedIter, IndexedIterMut};
