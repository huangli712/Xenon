//! Dimensions describe tensor rank and shape metadata.
//!
//! `Ix0` through `Ix6` represent statically ranked tensors, while `IxDyn`
//! stores a runtime-rank shape. The `Dimension` and `Reverse` traits are
//! sealed so the crate can keep stride, indexing, and broadcasting invariants
//! coherent.
//!
//! ## Type overview
//!
//! | Type | Rank | Description |
//! |------|------|-------------|
//! | `Ix0` | 0 | Scalar (zero-dimensional) |
//! | `Ix1` | 1 | Vector |
//! | `Ix2` | 2 | Matrix |
//! | `Ix3`–`Ix6` | 3–6 | Higher-rank tensors |
//! | `IxDyn` | runtime | Dynamically-sized dimensions |
//!
//! Conversion from tuples (`(usize,)`, etc.), arrays (`[usize; N]`),
//! and slices is provided through `IntoDimension`.

mod axes;
pub mod broadcast;
pub mod dynamic;
pub mod fixed;
pub mod into;

// Public re-exports — the canonical access path for dimension types.
pub use axes::Axis;
pub use broadcast::BroadcastDim;
pub use dynamic::IxDyn;
pub use fixed::{Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6};
pub use into::IntoDimension;

mod types;

pub use types::{Dimension, Reverse, RemoveAxis};
pub use types::MAX_DIMENSION;

