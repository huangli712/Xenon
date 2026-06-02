//! N-dimensional indexing and slicing.
//!
//! See `design/17-indexing.md` for the full design. This module declares the
//! three sub-modules and re-exports their public surface as each sub-module
//! is implemented. Each `pub use` line is enabled by the task indicated in
//! the adjacent comment.

pub mod impls;
pub mod ndindex;
pub mod slice;

// Re-exports — enabled incrementally. Each line is activated by the task
// named in its trailing comment; until then it stays commented out to avoid
// referencing symbols that do not yet exist.
//
pub use ndindex::NdIndex; // W21T2
// pub use impls::{/* inherent methods live on TensorBase */};    // W21T3/W21T5
pub use slice::{SliceInfo, SliceInfoElem, SliceInfoIndices}; // W21T4

