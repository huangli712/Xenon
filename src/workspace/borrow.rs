//! Immutable and mutable borrow guards for [`Workspace`] scratch regions.
//!
//! - [`WorkspaceBorrow`]: read-only guard — at most one active at a time,
//!   enforced by an [`AtomicU8`](core::sync::atomic::AtomicU8) CAS in
//!   [`Workspace::borrow`](super::Workspace::borrow).
//! - [`WorkspaceBorrowMut`]: exclusive mutable guard — compile-time
//!   exclusivity via `&mut self`, with CAS as defense-in-depth.
//!
//! Both guards release the borrow state on [`Drop`] via `Release` ordering,
//! making the workspace re-borrowable.

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
    /// Raw pointer to the start of data in the scratch region.
    pub(crate) ptr: NonNull<u8>,
    /// Length of the borrow in bytes.
    pub(crate) len: usize,
    /// Reference to the parent workspace (guarantees `!Send + !Sync`).
    pub(crate) workspace: &'a Workspace,
}

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

impl<'a> Drop for WorkspaceBorrow<'a> {
    fn drop(&mut self) {
        self.workspace
            .borrow_state
            .store(Workspace::BORROW_NONE, Ordering::Release);
    }
}

/// Mutable borrow guard — `!Send + !Sync` via `&'a Workspace`.
#[derive(Debug)]
pub struct WorkspaceBorrowMut<'a> {
    /// Raw pointer to the start of data in the scratch region.
    pub(crate) ptr: NonNull<u8>,
    /// Length of the borrow in bytes.
    pub(crate) len: usize,
    /// Reference to the parent workspace (guarantees `!Send + !Sync`).
    pub(crate) workspace: &'a Workspace,
}

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


impl<'a> Drop for WorkspaceBorrowMut<'a> {
    fn drop(&mut self) {
        self.workspace
            .borrow_state
            .store(Workspace::BORROW_NONE, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use crate::error::{TypedViewRejection, WorkspaceErrorCategory};
    use crate::workspace::Workspace;
    use std::mem::MaybeUninit;

    // ── WorkspaceBorrow ──

    /// `len()` returns the workspace capacity.
    #[test]
    fn test_borrow_len() {
        let ws = Workspace::new(128, 64).expect("workspace");
        let guard = ws.borrow().expect("borrow");
        assert_eq!(guard.len(), 128);
    }

    /// `is_empty()` returns false for non-zero-capacity workspace.
    #[test]
    fn test_borrow_is_empty() {
        let ws = Workspace::new(64, 64).expect("workspace");
        let guard = ws.borrow().expect("borrow");
        assert!(!guard.is_empty());
    }

    /// `as_ptr()` returns the workspace base address.
    #[test]
    fn test_borrow_as_ptr() {
        let ws = Workspace::new(64, 64).expect("workspace");
        let guard = ws.borrow().expect("borrow");
        assert_eq!(guard.as_ptr(), ws.ptr.as_ptr());
    }

    // ── WorkspaceBorrowMut ──

    /// `len()` returns the workspace capacity.
    #[test]
    fn test_borrow_mut_len() {
        let mut ws = Workspace::new(128, 64).expect("workspace");
        let guard = ws.borrow_mut().expect("borrow_mut");
        assert_eq!(guard.len(), 128);
    }

    /// `is_empty()` returns false for non-zero-capacity workspace.
    #[test]
    fn test_borrow_mut_is_empty() {
        let mut ws = Workspace::new(64, 64).expect("workspace");
        let guard = ws.borrow_mut().expect("borrow_mut");
        assert!(!guard.is_empty());
    }

    /// `as_mut_ptr()` returns a writable pointer.
    #[test]
    fn test_borrow_mut_as_mut_ptr() {
        let mut ws = Workspace::new(64, 64).expect("workspace");
        let mut guard = ws.borrow_mut().expect("borrow_mut");
        let ptr = guard.as_mut_ptr();
        unsafe { ptr.write(0x42); }
        assert_eq!(unsafe { ptr.read() }, 0x42);
    }

    /// `as_maybe_uninit_typed_slice` returns a typed uninit view of correct length.
    #[test]
    fn test_borrow_mut_typed_slice_basic() {
        let mut ws = Workspace::new(64, 64).expect("workspace");
        let mut guard = ws.borrow_mut().expect("borrow_mut");
        let result = unsafe { guard.as_maybe_uninit_typed_slice::<f64>(4) };
        assert!(result.is_ok());
        let view = result.expect("typed slice within capacity");
        assert_eq!(view.len(), 4);
        view[0].write(1.5);
    }

    /// TypedByteLengthOverflow and SplitOutOfBounds rejections for typed views.
    #[test]
    fn test_borrow_mut_typed_slice_rejections() {
        let mut ws = Workspace::new(64, 8).expect("64-byte workspace");
        let mut guard = ws.borrow_mut().expect("mutable borrow in test");

        // TypedByteLengthOverflow rejection (f64 implements Element).
        let result = unsafe { guard.as_maybe_uninit_typed_slice::<f64>(usize::MAX) };
        match result {
            Err(crate::error::XenonError::Workspace {
                category:
                    WorkspaceErrorCategory::TypedViewRejected {
                        detail: TypedViewRejection::TypedByteLengthOverflow { .. },
                    },
                ..
            }) => {},
            other => {
                panic!("expected TypedByteLengthOverflow, got {:?}", other)
            },
        }

        // SplitOutOfBounds (byte_len > workspace capacity) is reported via
        // the `SplitOutOfBounds` category, not `TypedViewRejected`.
        let result = unsafe { guard.as_maybe_uninit_typed_slice::<f64>(100) };
        assert!(matches!(
            result,
            Err(crate::error::XenonError::Workspace {
                category: WorkspaceErrorCategory::SplitOutOfBounds { .. },
                ..
            })
        ));
    }

    /// `assume_init_typed_slice` returns a valid typed mutable view.
    #[test]
    fn test_borrow_mut_assume_init_typed_basic() {
        let mut ws = Workspace::new(64, 64).expect("workspace");
        let mut guard = ws.borrow_mut().expect("borrow_mut");
        {
            let raw = guard.as_maybe_uninit_slice();
            let (head, _) = raw.split_at_mut(16);
            let f64s: &mut [MaybeUninit<f64>] = unsafe {
                std::slice::from_raw_parts_mut(
                    head.as_mut_ptr() as *mut MaybeUninit<f64>,
                    2,
                )
            };
            f64s[0].write(1.0);
            f64s[1].write(2.0);
        }
        let view = unsafe { guard.assume_init_typed_slice::<f64>(2) };
        assert!(view.is_ok());
        assert_eq!(view.expect("f64 values initialized"), &mut [1.0_f64, 2.0]);
    }

    /// Drop releases the borrow state so workspace is re-borrowable.
    #[test]
    fn test_borrow_mut_drop_releases() {
        let mut ws = Workspace::new(64, 64).expect("workspace");
        {
            let _guard = ws.borrow_mut().expect("borrow_mut");
        }
        assert!(ws.borrow_mut().is_ok());
    }
}
