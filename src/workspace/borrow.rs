use core::ptr::NonNull;
use core::sync::atomic::Ordering;
use std::borrow::Cow;

use super::workspace::Workspace;
use crate::error::{
    TypedViewRejection, WorkspaceBorrowKind, WorkspaceBorrowState, WorkspaceErrorCategory,
    XenonError,
};

/// Immutable borrow guard — `!Send + !Sync` via `&'a Workspace`.
#[derive(Debug)]
pub struct WorkspaceBorrow<'a> {
    ptr: NonNull<u8>,
    len: usize,
    workspace: &'a Workspace,
}

/// Mutable borrow guard — `!Send + !Sync` via `&'a Workspace`.
#[derive(Debug)]
pub struct WorkspaceBorrowMut<'a> {
    ptr: NonNull<u8>,
    len: usize,
    workspace: &'a Workspace,
}

// =============================================================================
// WorkspaceBorrow methods
// =============================================================================

impl<'a> WorkspaceBorrow<'a> {
    /// Returns the borrow length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the borrow is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the raw data pointer.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Returns the scratch region as possibly-uninitialized bytes.
    ///
    /// Takes `&mut self` so that the `MaybeUninit<u8>` view and the
    /// initialized `&[u8]` view are mutually exclusive at the borrow-guard
    /// level — safe code cannot hold both simultaneously.
    pub fn as_maybe_uninit_slice(&mut self) -> &[core::mem::MaybeUninit<u8>] {
        // SAFETY: `MaybeUninit<u8>` permits uninitialized bytes; the
        // pointer/length range is owned by the workspace whose guard we hold.
        unsafe {
            core::slice::from_raw_parts(
                self.ptr.as_ptr() as *const core::mem::MaybeUninit<u8>,
                self.len,
            )
        }
    }

    /// Interprets an initialized prefix as `&[u8]`.
    ///
    /// Takes `&mut self` for the same mutual-exclusion reason as
    /// `as_maybe_uninit_slice`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that the first `initialized_len` bytes have
    /// been fully initialized before calling this method.
    pub unsafe fn assume_init_slice(
        &mut self,
        initialized_len: usize,
    ) -> crate::error::Result<&[u8]> {
        if initialized_len > self.len {
            return Err(XenonError::workspace_split_oob(
                "WorkspaceBorrow::assume_init_slice",
                initialized_len,
                self.len,
            ));
        }
        // SAFETY: bounded by the check above; caller's `# Safety` precondition
        // covers initialization. `!Send + !Sync` precludes aliasing.
        Ok(unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), initialized_len) })
    }
}

// =============================================================================
// WorkspaceBorrowMut methods
// =============================================================================

impl<'a> WorkspaceBorrowMut<'a> {
    /// Returns the borrow length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the borrow is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the mutable data pointer.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Returns the mutable scratch region as possibly-uninitialized bytes.
    pub fn as_maybe_uninit_slice(&mut self) -> &mut [core::mem::MaybeUninit<u8>] {
        // SAFETY: same as `WorkspaceBorrow::as_maybe_uninit_slice`, mut variant.
        unsafe {
            core::slice::from_raw_parts_mut(
                self.ptr.as_ptr() as *mut core::mem::MaybeUninit<u8>,
                self.len,
            )
        }
    }

    /// Interprets an initialized prefix as `&mut [u8]`.
    ///
    /// # Errors
    ///
    /// Returns [`XenonError::Workspace`] with
    /// [`WorkspaceErrorCategory::SplitOutOfBounds`] if `initialized_len`
    /// exceeds the borrow length.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that the first `initialized_len` bytes have
    /// been fully initialized before calling this method.
    pub unsafe fn assume_init_slice(
        &mut self,
        initialized_len: usize,
    ) -> crate::error::Result<&mut [u8]> {
        if initialized_len > self.len {
            return Err(XenonError::workspace_split_oob(
                "WorkspaceBorrowMut::assume_init_slice",
                initialized_len,
                self.len,
            ));
        }
        // SAFETY: bounded by the check above; caller asserts initialization.
        Ok(unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), initialized_len) })
    }

    /// Typed access to possibly-uninitialized scratch memory.
    ///
    /// `T: Element` keeps the typed-view API closed over Xenon's supported
    /// element type set.
    ///
    /// # Errors
    ///
    /// - `TypedViewRejected::ZeroSizedType` — `size_of::<T>() == 0`
    /// - `TypedViewRejected::TypedByteLengthOverflow` — `count * size_of::<T>()` overflowed `usize`
    /// - `SplitOutOfBounds` — requested byte length exceeds borrow length
    /// - `TypedViewRejected::AlignmentMismatch` — buffer base not `T`-aligned
    ///
    /// # Safety
    ///
    /// Caller must still uphold `T` initialization model; size / alignment /
    /// count violations are returned as `Result` errors rather than UB.
    pub unsafe fn as_maybe_uninit_typed_slice<T: crate::element::Element>(
        &mut self,
        count: usize,
    ) -> crate::error::Result<&mut [core::mem::MaybeUninit<T>]> {
        const OP: &str = "WorkspaceBorrowMut::as_maybe_uninit_typed_slice";
        if core::mem::size_of::<T>() == 0 {
            return Err(XenonError::Workspace {
                operation: Cow::Borrowed(OP),
                category: WorkspaceErrorCategory::TypedViewRejected {
                    detail: TypedViewRejection::ZeroSizedType,
                },
                cause: None,
            });
        }
        let byte_len =
            count
                .checked_mul(core::mem::size_of::<T>())
                .ok_or(XenonError::Workspace {
                    operation: Cow::Borrowed(OP),
                    category: WorkspaceErrorCategory::TypedViewRejected {
                        detail: TypedViewRejection::TypedByteLengthOverflow {
                            count,
                            elem_size: core::mem::size_of::<T>(),
                        },
                    },
                    cause: None,
                })?;
        if byte_len > self.len {
            return Err(XenonError::workspace_split_oob(OP, byte_len, self.len));
        }
        let actual_addr = self.ptr.as_ptr() as usize;
        if !actual_addr.is_multiple_of(core::mem::align_of::<T>()) {
            return Err(XenonError::Workspace {
                operation: Cow::Borrowed(OP),
                category: WorkspaceErrorCategory::TypedViewRejected {
                    detail: TypedViewRejection::AlignmentMismatch {
                        required: core::mem::align_of::<T>(),
                        actual: actual_addr % core::mem::align_of::<T>(),
                    },
                },
                cause: None,
            });
        }
        // SAFETY: bounds and alignment checked above; `MaybeUninit<T>` permits
        // uninitialized representation. Exclusive borrow + `!Send + !Sync`
        // forbid aliasing.
        Ok(unsafe {
            core::slice::from_raw_parts_mut(
                self.ptr.as_ptr() as *mut core::mem::MaybeUninit<T>,
                count,
            )
        })
    }

    /// Interprets the first `count` elements as initialized `T` values.
    ///
    /// # Errors
    ///
    /// - `TypedViewRejected::ZeroSizedType` — `size_of::<T>() == 0`
    /// - `TypedViewRejected::TypedByteLengthOverflow` — `count * size_of::<T>()` overflowed `usize`
    /// - `SplitOutOfBounds` — requested byte length exceeds borrow length
    /// - `TypedViewRejected::AlignmentMismatch` — buffer base not `T`-aligned
    ///
    /// # Safety
    ///
    /// Caller must guarantee:
    /// - the first `count` typed elements are fully initialized and valid `T`,
    /// - `count * size_of::<T>() <= self.len()`,
    /// - and the scratch region satisfies `T` alignment.
    pub unsafe fn assume_init_typed_slice<T: crate::element::Element>(
        &mut self,
        count: usize,
    ) -> crate::error::Result<&mut [T]> {
        const OP: &str = "WorkspaceBorrowMut::assume_init_typed_slice";
        if core::mem::size_of::<T>() == 0 {
            return Err(XenonError::Workspace {
                operation: Cow::Borrowed(OP),
                category: WorkspaceErrorCategory::TypedViewRejected {
                    detail: TypedViewRejection::ZeroSizedType,
                },
                cause: None,
            });
        }
        let byte_len =
            count
                .checked_mul(core::mem::size_of::<T>())
                .ok_or(XenonError::Workspace {
                    operation: Cow::Borrowed(OP),
                    category: WorkspaceErrorCategory::TypedViewRejected {
                        detail: TypedViewRejection::TypedByteLengthOverflow {
                            count,
                            elem_size: core::mem::size_of::<T>(),
                        },
                    },
                    cause: None,
                })?;
        if byte_len > self.len {
            return Err(XenonError::workspace_split_oob(OP, byte_len, self.len));
        }
        let actual_addr = self.ptr.as_ptr() as usize;
        if !actual_addr.is_multiple_of(core::mem::align_of::<T>()) {
            return Err(XenonError::Workspace {
                operation: Cow::Borrowed(OP),
                category: WorkspaceErrorCategory::TypedViewRejected {
                    detail: TypedViewRejection::AlignmentMismatch {
                        required: core::mem::align_of::<T>(),
                        actual: actual_addr % core::mem::align_of::<T>(),
                    },
                },
                cause: None,
            });
        }
        // SAFETY: bounds and alignment checked; caller's `# Safety` covers
        // initialization. Exclusive borrow + `!Send + !Sync` forbid aliasing.
        Ok(unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut T, count) })
    }
}

// =============================================================================
// borrow / borrow_mut on Workspace + diagnostic helper
// =============================================================================

/// Internal helper: read current borrow state in structured form for error
/// reporting. Loads `borrow_state` and `split_count` with `Relaxed` because
/// the loads are diagnostic-only — borrow safety is enforced by the CAS in
/// `borrow()` and by `&mut self` in `borrow_mut`/`split_at_mut`/`ensure_capacity`.
pub(crate) fn current_borrow_state(ws: &Workspace) -> WorkspaceBorrowState {
    let bs = ws.borrow_state.load(Ordering::Relaxed);
    let sc = ws.split_count.load(Ordering::Relaxed);
    match (bs, sc) {
        (Workspace::BORROW_NONE, _) => WorkspaceBorrowState::None,
        (Workspace::BORROW_READ, _) => WorkspaceBorrowState::Shared,
        (Workspace::BORROW_EXCLUSIVE, 0) => WorkspaceBorrowState::Exclusive,
        (Workspace::BORROW_EXCLUSIVE, count) => WorkspaceBorrowState::SplitActive { count },
        _ => WorkspaceBorrowState::None,
    }
}

impl Workspace {
    /// Acquire the workspace for read-only inspection of the scratch region.
    ///
    /// Takes `&self`: at most one active read guard is enforced at runtime
    /// by the internal `AtomicU8` state machine — multiple read guards are
    /// mutually exclusive but do not require compile-time exclusivity.
    ///
    /// # Errors
    ///
    /// Returns [`XenonError::Workspace`] with
    /// [`WorkspaceErrorCategory::BorrowConflict`] (`requested: Shared`) if
    /// the workspace already has an active borrow (shared or exclusive). At
    /// most one active read guard is allowed by design.
    pub fn borrow(&self) -> crate::error::Result<WorkspaceBorrow<'_>> {
        let prev = self.borrow_state.compare_exchange(
            Self::BORROW_NONE,
            Self::BORROW_READ,
            Ordering::Acquire,
            Ordering::Relaxed,
        );
        if prev.is_err() {
            return Err(XenonError::workspace_borrow_conflict(
                "Workspace::borrow",
                WorkspaceBorrowKind::Shared,
                current_borrow_state(self),
            ));
        }
        Ok(WorkspaceBorrow {
            ptr: self.ptr,
            len: self.capacity,
            workspace: self,
        })
    }

    /// Mutably borrow the workspace.
    ///
    /// Takes `&mut self`: compile-time exclusivity makes "exclusive borrow
    /// while another guard exists" a static type error. The internal
    /// `AtomicU8` CAS still runs as defense-in-depth.
    ///
    /// # Errors
    ///
    /// Returns [`XenonError::Workspace`] with
    /// [`WorkspaceErrorCategory::BorrowConflict`] (`requested: Exclusive`) if
    /// the internal `AtomicU8` CAS observes a non-`None` borrow state. In
    /// practice this branch is unreachable while `&mut self` is held, but the
    /// check remains as defense-in-depth.
    pub fn borrow_mut(&mut self) -> crate::error::Result<WorkspaceBorrowMut<'_>> {
        let prev = self.borrow_state.compare_exchange(
            Self::BORROW_NONE,
            Self::BORROW_EXCLUSIVE,
            Ordering::Acquire,
            Ordering::Relaxed,
        );
        if prev.is_err() {
            return Err(XenonError::workspace_borrow_conflict(
                "Workspace::borrow_mut",
                WorkspaceBorrowKind::Exclusive,
                current_borrow_state(self),
            ));
        }
        Ok(WorkspaceBorrowMut {
            ptr: self.ptr,
            len: self.capacity,
            workspace: self,
        })
    }

    /// Crate-internal probe used by `expand.rs` (W9T6) to verify no residual
    /// borrow state before reallocation. Uses `Acquire` to pair with the
    /// `Release` stores performed by guard `Drop` impls.
    #[expect(dead_code, reason = "defense-in-depth helper; W9T6 uses direct load")]
    pub(crate) fn is_borrowed(&self) -> bool {
        self.borrow_state.load(Ordering::Acquire) != Self::BORROW_NONE
    }
}

// =============================================================================
// Drop impls — release the exclusive/shared borrow on the workspace.
// =============================================================================

impl<'a> Drop for WorkspaceBorrow<'a> {
    fn drop(&mut self) {
        self.workspace
            .borrow_state
            .store(Workspace::BORROW_NONE, Ordering::Release);
    }
}

impl<'a> Drop for WorkspaceBorrowMut<'a> {
    fn drop(&mut self) {
        self.workspace
            .borrow_state
            .store(Workspace::BORROW_NONE, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use crate::workspace::Workspace;

    #[test]
    fn test_borrow_basic() {
        let workspace = Workspace::new(64, 64).expect("64-byte workspace");
        let mut guard = workspace.borrow().expect("immutable borrow");
        let view = guard.as_maybe_uninit_slice();
        assert_eq!(view.len(), 64);
    }

    #[test]
    fn test_borrow_mut_basic() {
        let mut workspace = Workspace::new(64, 64).expect("64-byte workspace");
        let mut guard = workspace.borrow_mut().expect("mutable borrow");
        let view = guard.as_maybe_uninit_slice();
        assert_eq!(view.len(), 64);
        // Write through the MaybeUninit view to demonstrate it is writable.
        for slot in view.iter_mut() {
            slot.write(0xAA);
        }
    }

    #[test]
    fn test_borrow_double_fails() {
        let workspace = Workspace::new(64, 64).expect("64-byte workspace");
        let _g1 = workspace.borrow().expect("first borrow");
        // Second shared borrow must conflict — current design allows at most
        // one active read guard.
        assert!(workspace.borrow().is_err());
    }

    #[test]
    fn test_borrow_after_drop() {
        let workspace = Workspace::new(64, 64).expect("64-byte workspace");
        {
            let _g = workspace.borrow().expect("scoped borrow");
        }
        // Re-borrow after the previous guard is dropped must succeed.
        assert!(workspace.borrow().is_ok());
    }

    #[test]
    fn test_assume_init_requires_initialized_prefix() {
        let mut workspace = Workspace::new(64, 64).expect("64-byte workspace");
        let mut guard = workspace.borrow_mut().expect("mutable borrow");
        // OOB request rejected with structured error rather than UB.
        let err = unsafe { guard.assume_init_slice(128) };
        assert!(err.is_err());
        // Within bounds (caller takes responsibility for true initialization).
        let ok = unsafe { guard.assume_init_slice(0) };
        assert!(ok.is_ok());
    }
}

/// `test_workspace_borrow_views_are_mutually_exclusive` is realized as a
/// compile-fail doctest because the violation must be a *static* type error
/// — `as_maybe_uninit_slice` and `assume_init_slice` both take `&mut self`,
/// so safe code cannot hold both views from the same borrow at once.
///
/// ```compile_fail
/// # use xenon::workspace::Workspace;
/// let workspace = Workspace::new(64, 64).unwrap();
/// let mut guard = workspace.borrow().unwrap();
/// let a = guard.as_maybe_uninit_slice();
/// // Second concurrent view from the same borrow — must fail to compile.
/// let b = unsafe { guard.assume_init_slice(0) }.unwrap();
/// let _ = (a, b);
/// ```
#[cfg(doctest)]
struct ViewsMutuallyExclusiveDoctest;
