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
//! `Workspace` and all guards are `!Send + !Sync` — verified at compile time
//! (see `compile_time_negative_assertions` below).
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
#[expect(
    clippy::module_inception,
    reason = "intentional: workspace::workspace holds the Workspace type definition"
)]
mod workspace;

pub use borrow::{WorkspaceBorrow, WorkspaceBorrowMut};
pub use split::SplitBorrowMut;
pub use workspace::Workspace;

#[cfg(test)]
mod tests {
    //! Compile-time verification that workspace types are `!Send + !Sync`.

    use super::{SplitBorrowMut, Workspace, WorkspaceBorrow, WorkspaceBorrowMut};
    use static_assertions::assert_not_impl_all;

    // Each `assert_not_impl_all!` is a zero-runtime-cost check — if any of
    // these auto-traits become accidentally `impl`'d, the build fails.
    assert_not_impl_all!(Workspace: Send);
    assert_not_impl_all!(Workspace: Sync);
    assert_not_impl_all!(WorkspaceBorrow<'static>: Send);
    assert_not_impl_all!(WorkspaceBorrow<'static>: Sync);
    assert_not_impl_all!(WorkspaceBorrowMut<'static>: Send);
    assert_not_impl_all!(WorkspaceBorrowMut<'static>: Sync);
    assert_not_impl_all!(SplitBorrowMut<'static>: Send);
    assert_not_impl_all!(SplitBorrowMut<'static>: Sync);
}


