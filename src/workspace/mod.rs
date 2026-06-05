//! Temporary aligned workspace for internal scratch buffers.
//!
//! # Design
//!
//! - Allocation requests an aligned raw buffer via `std::alloc`.
//! - Bytes are modeled as `MaybeUninit<u8>` until callers prove initialization.
//! - Borrow state is tracked with atomic tags; at most one active guard per workspace.
//! - `ensure_capacity` only grows, never shrinks.
//! - Splits are O(1) pointer arithmetic (no allocation).
//!
//! # Thread Safety
//!
//! `Workspace` and all guards are `!Send + !Sync` — enforced by
//! `PhantomData<*mut ()>` and `&'a Workspace` lifetime binding.
//!
//! # Example
//!
//! ```text
//! use xenon::workspace::Workspace;
//!
//! let mut ws = Workspace::new(1024, 64)?;
//!
//! // Mutable borrow
//! let mut buf = ws.borrow_mut()?;
//! let scratch = buf.as_maybe_uninit_slice();
//! // initialize scratch...
//! drop(buf); // RAII return
//!
//! // Split
//! let (a, b) = ws.split_at_mut(512)?;
//! # Ok::<(), xenon::error::XenonError>(())
//! ```

mod borrow;
mod split;
mod space;

pub use borrow::{WorkspaceBorrow, WorkspaceBorrowMut};
pub use split::SplitBorrowMut;
pub use space::Workspace;



