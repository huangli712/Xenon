//! Pre-allocated, aligned scratch buffer with atomic borrow-state tracking.
//!
//! Guards (`WorkspaceBorrow`, `WorkspaceBorrowMut`, `SplitBorrowMut`) are
//! `!Send + !Sync` and use RAII to restore borrow state on drop.

mod borrow;
mod split;
mod space;

pub use borrow::{WorkspaceBorrow, WorkspaceBorrowMut};
pub use split::SplitBorrowMut;
pub use space::Workspace;
