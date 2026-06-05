//! Split borrow guard for partitioning a [`Workspace`] into sub-spaces.
//!
//! [`SplitBorrowMut`] represents one contiguous slice of a workspace after
//! [`Workspace::split_at_mut`](super::Workspace::split_at_mut). Multiple
//! split guards coexist over non-overlapping memory regions, reference-counted
//! via an [`AtomicUsize`]. The workspace is released only when the last split
//! guard drops.

use core::ptr::NonNull;
use core::sync::atomic::{AtomicUsize, Ordering};

use super::workspace::Workspace;
use crate::error::XenonError;

/// Borrow guard for a split sub-space.
///
/// Multiple `SplitBorrowMut` guards from the same root `split_at_mut` call
/// coexist over non-overlapping memory regions. `!Send + !Sync` via
/// `&'a Workspace`.
#[derive(Debug)]
pub struct SplitBorrowMut<'a> {
    /// Start pointer of this split sub-space.
    pub(crate) ptr: NonNull<u8>,
    /// Length of this split sub-space in bytes.
    pub(crate) len: usize,
    /// Parent workspace whose borrow state is restored when all splits drop.
    pub(crate) workspace: &'a Workspace,
    /// Reference to the split count. Top-level `split_at_mut` initializes the
    /// counter to 2 (binary split); recursive `split_at_mut` `fetch_add(1)`s.
    /// Drop `fetch_sub(1)`s; only on `prev == 1` is `borrow_state` reset.
    pub(crate) split_count: &'a AtomicUsize,
}

impl<'a> SplitBorrowMut<'a> {
    /// Returns the sub-space length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the sub-space is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the split sub-space as possibly-uninitialized bytes.
    pub fn as_maybe_uninit_slice(&mut self) -> &mut [core::mem::MaybeUninit<u8>] {
        // SAFETY: split guards expose scratch memory as possibly uninitialized;
        // the pointer/length range is disjoint from sibling splits and the
        // borrow is exclusive within this sub-space.
        unsafe {
            core::slice::from_raw_parts_mut(
                self.ptr.as_ptr() as *mut core::mem::MaybeUninit<u8>,
                self.len,
            )
        }
    }

    /// Continue splitting (recursive binary split). O(1) — pointer arithmetic.
    ///
    /// **Safety design**: `split_at_mut` *consumes* `self` rather than
    /// borrowing, ensuring each `SplitBorrowMut` has an independent lifetime.
    /// The original guard's `Drop` is bypassed via `ManuallyDrop::new(self)`.
    ///
    /// **Reference count invariant**: because `Drop` is skipped (−1 avoided)
    /// but two new guards are produced (+2 expected), the net active-guard
    /// change is +1. We `fetch_add(1, Release)` before constructing the
    /// children so the last sibling's Drop correctly waits for ALL active
    /// sub-splits before resetting `borrow_state`.
    ///
    /// # Errors
    ///
    /// Returns `WorkspaceBorrowKind::SplitOutOfBounds` when `mid > self.len`.
    pub fn split_at_mut(
        self,
        mid: usize,
    ) -> crate::error::Result<(SplitBorrowMut<'a>, SplitBorrowMut<'a>)> {
        if mid > self.len {
            return Err(XenonError::workspace_split_oob(
                "SplitBorrowMut::split_at_mut",
                mid,
                self.len,
            ));
        }

        // Skip the original guard's Drop — its conceptual slot in the count
        // is transferred to the children (see invariant above).
        let this = core::mem::ManuallyDrop::new(self);

        // Increment for the additional sub-space; `Release` so the children's
        // Drop observes the up-to-date counter.
        this.split_count.fetch_add(1, Ordering::Release);

        let left_ptr = this.ptr;
        // SAFETY: mid <= this.len (checked above), so the offset stays within
        // the parent split's region.
        let right_ptr = unsafe { NonNull::new_unchecked(this.ptr.as_ptr().add(mid)) };

        Ok((
            SplitBorrowMut {
                ptr: left_ptr,
                len: mid,
                workspace: this.workspace,
                split_count: this.split_count,
            },
            SplitBorrowMut {
                ptr: right_ptr,
                len: this.len - mid,
                workspace: this.workspace,
                split_count: this.split_count,
            },
        ))
    }
}

/// Drop releases the exclusive borrow on the workspace.
///
/// Reference counting: each top-level `split_at_mut()` sets split_count to
/// the number of sub-spaces (2 for binary split); each recursive
/// `split_at_mut()` atomically increments by 1 (net +1 active guard). Each
/// `SplitBorrowMut::drop()` atomically decrements. Only when split_count
/// reaches 0 (i.e., `prev == 1`) is `borrow_state` reset to `BORROW_NONE`.
///
/// # Safety Invariant
///
/// After `drop`, the caller must not use any existing references into the
/// workspace memory. The Rust borrow checker enforces this via `'a`. Any
/// pair of active split guards (including recursively produced descendants)
/// must always cover disjoint, non-overlapping byte ranges.
impl<'a> Drop for SplitBorrowMut<'a> {
    fn drop(&mut self) {
        // `AcqRel`: Release for our own decrement to be visible to other
        // drops; Acquire to read the freshest count when deciding whether
        // to reset borrow_state.
        let prev = self.split_count.fetch_sub(1, Ordering::AcqRel);
        if prev == 1 {
            self.workspace
                .borrow_state
                .store(Workspace::BORROW_NONE, Ordering::Release);
        }
    }
}


