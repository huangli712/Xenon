//! Aligned allocator.
//!
//! Crate-internal 64-byte aligned memory allocator. ZST and `size == 0`
//! paths are handled by callers; this allocator requires `size > 0`.

use core::ptr::NonNull;
use std::borrow::Cow;
use std::alloc::{Layout, alloc, alloc_zeroed, dealloc};

use crate::error::{XenonError, WorkspaceErrorCategory};

/// Aligned memory allocator.
pub(crate) struct AlignedAlloc;

impl AlignedAlloc {
    /// Current default alignment: 64 bytes.
    pub(crate) const DEFAULT_ALIGNMENT: usize = 64;

    /// Allocates a memory block of the given size and alignment, without
    /// initialization.
    ///
    /// # Errors
    ///
    /// Returns [`XenonError`] when:
    /// - `align` is not a power of two
    /// - `size` is 0
    /// - the requested layout is invalid
    /// - the allocator reports allocation failure
    ///
    /// For ZST or zero-sized allocations, callers must skip this allocator
    /// and use `NonNull::dangling()` directly.
    pub(crate) fn alloc(
        size: usize,
        align: usize
    ) -> Result<NonNull<u8>, XenonError> {
        if size == 0 {
            return Err(XenonError::Workspace {
                operation: Cow::Borrowed("AlignedAlloc::alloc"),
                category: WorkspaceErrorCategory::AllocFailed { size, align },
                cause: None,
            });
        }
        let layout = Layout::from_size_align(size, align).map_err(|_| XenonError::Workspace {
            operation: Cow::Borrowed("AlignedAlloc::alloc"),
            category: WorkspaceErrorCategory::InvalidLayout { size, align },
            cause: None,
        })?;
        let ptr = unsafe { alloc(layout) };
        NonNull::new(ptr).ok_or(XenonError::Workspace {
            operation: Cow::Borrowed("AlignedAlloc::alloc"),
            category: WorkspaceErrorCategory::AllocFailed { size, align },
            cause: None,
        })
    }

    /// Allocates and zero-initializes.
    pub(crate) fn alloc_zeroed(size: usize, align: usize) -> Result<NonNull<u8>, XenonError> {
        if size == 0 {
            return Err(XenonError::Workspace {
                operation: Cow::Borrowed("AlignedAlloc::alloc_zeroed"),
                category: WorkspaceErrorCategory::AllocFailed { size, align },
                cause: None,
            });
        }
        let layout = Layout::from_size_align(size, align).map_err(|_| XenonError::Workspace {
            operation: Cow::Borrowed("AlignedAlloc::alloc_zeroed"),
            category: WorkspaceErrorCategory::InvalidLayout { size, align },
            cause: None,
        })?;
        let ptr = unsafe { alloc_zeroed(layout) };
        NonNull::new(ptr).ok_or(XenonError::Workspace {
            operation: Cow::Borrowed("AlignedAlloc::alloc_zeroed"),
            category: WorkspaceErrorCategory::AllocFailed { size, align },
            cause: None,
        })
    }

    /// Deallocates memory.
    ///
    /// # Safety
    ///
    /// - `ptr` must have been returned by `alloc` or `alloc_zeroed`
    /// - `size` and `align` must be the same as during allocation
    /// - caller must not use `ptr` after this call
    pub(crate) unsafe fn dealloc(ptr: NonNull<u8>, size: usize, align: usize) {
        let layout = Layout::from_size_align(size, align).expect("invalid allocation layout");
        // SAFETY: caller guarantees ptr/size/align match the original allocation
        unsafe {
            dealloc(ptr.as_ptr(), layout);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Allocates and deallocates 64-byte aligned memory successfully.
    #[test]
    fn test_aligned_alloc_64() {
        let size = 8 * core::mem::size_of::<f64>();
        let ptr = AlignedAlloc::alloc(size, AlignedAlloc::DEFAULT_ALIGNMENT)
            .expect("allocation succeeds");
        assert_eq!(ptr.as_ptr() as usize % AlignedAlloc::DEFAULT_ALIGNMENT, 0);
        unsafe {
            AlignedAlloc::dealloc(ptr, size, AlignedAlloc::DEFAULT_ALIGNMENT);
        }
    }

    /// Allocates zeroed memory and verifies all bytes are zero.
    #[test]
    fn test_aligned_alloc_zeroed() {
        let size = 8 * core::mem::size_of::<u64>();
        let ptr = AlignedAlloc::alloc_zeroed(size, AlignedAlloc::DEFAULT_ALIGNMENT)
            .expect("zeroed allocation succeeds");
        assert_eq!(ptr.as_ptr() as usize % AlignedAlloc::DEFAULT_ALIGNMENT, 0);
        let values = unsafe { core::slice::from_raw_parts(ptr.as_ptr() as *const u64, 8) };
        assert!(values.iter().all(|value| *value == 0));
        unsafe {
            AlignedAlloc::dealloc(ptr, size, AlignedAlloc::DEFAULT_ALIGNMENT);
        }
    }

    /// Rejects an impossibly large allocation with an error.
    #[test]
    fn test_aligned_alloc_invalid_layout_returns_error() {
        let err = AlignedAlloc::alloc(isize::MAX as usize, AlignedAlloc::DEFAULT_ALIGNMENT)
            .expect_err("layout should be rejected");
        match err {
            XenonError::Workspace {
                category: WorkspaceErrorCategory::InvalidLayout { size, align },
                ..
            } => {
                assert_eq!(size, isize::MAX as usize);
                assert_eq!(align, AlignedAlloc::DEFAULT_ALIGNMENT);
            },
            other => panic!("expected InvalidLayout, got {other:?}"),
        }
    }
}
