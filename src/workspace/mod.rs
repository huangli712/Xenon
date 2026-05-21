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
mod expand;
mod split;
#[expect(clippy::module_inception, reason = "intentional: workspace::workspace follows design doc §3")]
mod workspace;

pub use borrow::{WorkspaceBorrow, WorkspaceBorrowMut};
pub use split::SplitBorrowMut;
pub use workspace::Workspace;

#[cfg(test)]
mod compile_time_negative_assertions {
    //! Compile-time verification that workspace types are `!Send + !Sync`.

    use static_assertions::assert_not_impl_all;
    use super::{SplitBorrowMut, Workspace, WorkspaceBorrow, WorkspaceBorrowMut};

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

// ── W9T7 typed-slice rejection tests ──
#[cfg(test)]
mod tests {
    use crate::workspace::Workspace;
    use crate::error::{WorkspaceErrorCategory, TypedViewRejection};

    /// Exercise the reachable `TypedViewRejection` paths and the
    /// `SplitOutOfBounds` rejection for typed views.
    ///
    /// **Note on `AlignmentMismatch`**: this branch cannot be reached via
    /// the public API when the workspace satisfies `alignment >= MIN_ALIGNMENT = 8`
    /// and Element types all have `align_of::<T>() <= 8` (bool / i32 / i64 /
    /// f32 / f64 / Complex<f32> / Complex<f64> under `#[repr(C)]`). The
    /// branch is instead covered by property tests under `tests/property/`
    /// which construct misaligned sub-regions through `#[cfg(test)]`
    /// visibility hooks.
    #[test]
    fn test_typed_slice_rejections() {
        // Allocate a workspace aligned to 8 bytes (the minimum allowed).
        let mut ws = Workspace::new(64, 8).expect("64-byte workspace");
        let mut guard = ws.borrow_mut().expect("mutable borrow in test");

        // TypedByteLengthOverflow rejection (f64 implements Element).
        let result =
            unsafe { guard.as_maybe_uninit_typed_slice::<f64>(usize::MAX) };
        match result {
            Err(crate::error::XenonError::Workspace {
                category:
                    WorkspaceErrorCategory::TypedViewRejected {
                        detail:
                            TypedViewRejection::TypedByteLengthOverflow { .. },
                    },
                ..
            }) => {}
            other => {
                panic!("expected TypedByteLengthOverflow, got {:?}", other)
            }
        }

        // SplitOutOfBounds (byte_len > workspace capacity) is reported via
        // the `SplitOutOfBounds` category, not `TypedViewRejected`.
        let result =
            unsafe { guard.as_maybe_uninit_typed_slice::<f64>(100) };
        assert!(matches!(
            result,
            Err(crate::error::XenonError::Workspace {
                category: WorkspaceErrorCategory::SplitOutOfBounds { .. },
                ..
            })
        ));
    }
}