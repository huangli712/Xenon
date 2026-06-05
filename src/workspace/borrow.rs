use core::ptr::NonNull;
use core::sync::atomic::Ordering;
use std::borrow::Cow;

use super::workspace::Workspace;
use crate::error::{
    TypedViewRejection, WorkspaceErrorCategory,
    XenonError,
};

/// Immutable borrow guard — `!Send + !Sync` via `&'a Workspace`.
#[derive(Debug)]
pub struct WorkspaceBorrow<'a> {
    pub(crate) ptr: NonNull<u8>,
    pub(crate) len: usize,
    pub(crate) workspace: &'a Workspace,
}

/// Mutable borrow guard — `!Send + !Sync` via `&'a Workspace`.
#[derive(Debug)]
pub struct WorkspaceBorrowMut<'a> {
    pub(crate) ptr: NonNull<u8>,
    pub(crate) len: usize,
    pub(crate) workspace: &'a Workspace,
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
                
            });
        }
        // SAFETY: bounds and alignment checked; caller's `# Safety` covers
        // initialization. Exclusive borrow + `!Send + !Sync` forbid aliasing.
        Ok(unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut T, count) })
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

