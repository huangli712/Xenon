use core::sync::atomic::Ordering;
use std::borrow::Cow;

use super::workspace::current_borrow_state;
use super::workspace::Workspace;
use crate::error::{WorkspaceBorrowKind, WorkspaceErrorCategory, XenonError};

impl Workspace {
    /// Ensure capacity is at least `min_capacity`.
    ///
    /// If current capacity is insufficient, a larger memory region is
    /// allocated. New capacity = max(min_capacity, current × 1.5).
    ///
    /// Growth may preserve existing bytes in the current implementation, but
    /// callers MUST NOT rely on that. After growth, all previous views and
    /// borrows are invalidated. Treat the entire scratch region as
    /// unspecified until it is re-initialized. Growth only guarantees that
    /// capacity satisfies the new request and that alignment is unchanged.
    ///
    /// # Errors
    ///
    /// - `XenonError::Workspace { BorrowConflict }` — residual borrow state
    ///   (defensive; should be unreachable under correct `&mut self` semantics)
    /// - `XenonError::Workspace { GrowOverflow }` — `capacity * 1.5` overflows
    /// - `XenonError::Workspace { InvalidLayout | AllocFailed }` — allocator failure
    pub fn ensure_capacity(&mut self, min_capacity: usize) -> crate::error::Result<()> {
        if min_capacity <= self.capacity {
            return Ok(());
        }

        // Compile-time exclusivity from `&mut self` already prevents an active
        // guard from coexisting with growth. The Acquire load is
        // defense-in-depth (pairs with guard `Drop`'s Release store) to catch
        // residual state from a buggy split chain or torn shutdown.
        let state = self.borrow_state.load(Ordering::Acquire);
        if state != Self::BORROW_NONE {
            return Err(XenonError::workspace_borrow_conflict(
                "Workspace::ensure_capacity",
                WorkspaceBorrowKind::Exclusive,
                current_borrow_state(self),
            ));
        }

        // 1.5x growth. `checked_mul` first to surface overflow as
        // `GrowOverflow` rather than silently wrapping; division by the
        // (non-zero) denominator is plain `/`.
        let grown = self
            .capacity
            .checked_mul(Self::GROWTH_FACTOR_NUMERATOR)
            .ok_or_else(|| {
                XenonError::workspace_grow_overflow(
                    "Workspace::ensure_capacity",
                    self.capacity,
                    min_capacity,
                )
            })?
            / Self::GROWTH_FACTOR_DENOMINATOR;
        let new_capacity = grown.max(min_capacity);

        self.reallocate(new_capacity)
    }

    /// Internal reallocation. Not part of the public API.
    fn reallocate(&mut self, new_capacity: usize) -> crate::error::Result<()> {
        let new_layout = std::alloc::Layout::from_size_align(new_capacity, self.alignment)
            .map_err(|_| XenonError::Workspace {
                operation: Cow::Borrowed("Workspace::reallocate"),
                category: WorkspaceErrorCategory::InvalidLayout {
                    size: new_capacity,
                    align: self.alignment,
                },
                
            })?;

        // SAFETY: layout is valid (checked above).
        let new_ptr = unsafe { std::alloc::alloc(new_layout) };
        let new_ptr = core::ptr::NonNull::new(new_ptr).ok_or(XenonError::Workspace {
            operation: Cow::Borrowed("Workspace::reallocate"),
            category: WorkspaceErrorCategory::AllocFailed {
                size: new_capacity,
                align: self.alignment,
            },
            
        })?;

        // Implementation detail: bytes MAY be copied during growth, but this
        // is NOT part of the stable public contract. Callers must not rely
        // on content preservation. All previous views are invalid after
        // reallocation; the scratch region must be treated as unspecified.
        // SAFETY: src and dst are non-overlapping; copy min(old, new) bytes.
        unsafe {
            core::ptr::copy_nonoverlapping(
                self.ptr.as_ptr(),
                new_ptr.as_ptr(),
                self.capacity.min(new_capacity),
            );
        }

        // Free old memory.
        // SAFETY: old layout was valid at allocation time and unchanged.
        unsafe {
            let old_layout =
                std::alloc::Layout::from_size_align_unchecked(self.capacity, self.alignment);
            std::alloc::dealloc(self.ptr.as_ptr(), old_layout);
        }

        self.ptr = new_ptr;
        self.capacity = new_capacity;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::workspace::Workspace;
    use core::sync::atomic::Ordering;

    #[test]
    fn test_ensure_capacity_no_grow() {
        let mut workspace = Workspace::new(64, 64).expect("64-byte workspace");
        workspace
            .ensure_capacity(32)
            .expect("ensure capacity in test");
        assert_eq!(workspace.capacity(), 64);
    }

    #[test]
    fn test_ensure_capacity_grow() {
        let mut workspace = Workspace::new(64, 64).expect("64-byte workspace");
        workspace
            .ensure_capacity(128)
            .expect("ensure capacity in test");
        assert!(workspace.capacity() >= 128);
        // Alignment must be preserved across growth.
        assert_eq!(workspace.alignment(), 64);
    }

    /// Runtime defense-in-depth: if `borrow_state` somehow carries residual
    /// non-NONE state (which the compile-time `&mut self` would normally
    /// preclude), `ensure_capacity` must surface a structured `BorrowConflict`
    /// rather than racing with a phantom guard.
    ///
    /// We simulate the residual state by directly poking `borrow_state` —
    /// this is a `#[cfg(test)]`-only path exercising the Acquire load on
    /// the growth path.
    #[test]
    fn test_ensure_capacity_while_borrowed_fails() {
        let mut workspace = Workspace::new(64, 64).expect("64-byte workspace");
        // Inject residual exclusive state. Field access is allowed here
        // because `workspace.rs` declares `borrow_state` as `pub(crate)`.
        workspace
            .borrow_state
            .store(Workspace::BORROW_EXCLUSIVE, Ordering::Release);
        let err = workspace.ensure_capacity(256);
        assert!(err.is_err());
        // Restore so Drop is balanced.
        workspace
            .borrow_state
            .store(Workspace::BORROW_NONE, Ordering::Release);
    }
}
