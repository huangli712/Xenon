use crate::error::{Result, WorkspaceErrorCategory};
use core::marker::PhantomData;
use core::ptr::NonNull;
use core::sync::atomic::AtomicU8;
use core::sync::atomic::AtomicUsize;
use std::borrow::Cow;

/// Temporary aligned workspace for internal scratch buffers.
///
/// # Fields
///
/// All fields except `_not_send_sync` are `pub(crate)` so that sibling
/// modules (`borrow`, `split`, `expand`) can implement their logic directly
/// on the workspace's internal state (e.g., CAS on `borrow_state`, pointer
/// arithmetic on `ptr`) without indirection through getters/setters.
///
/// `_not_send_sync` remains private to guarantee `!Send + !Sync` at the
/// type-system level.
#[derive(Debug)]
pub struct Workspace {
    /// Raw pointer to the start of the aligned allocation.
    pub(crate) ptr: NonNull<u8>,
    /// Total byte length of the allocation.
    pub(crate) capacity: usize,
    /// Allocation alignment in bytes.
    pub(crate) alignment: usize,
    /// Borrow-state tag: `BORROW_NONE` / `BORROW_READ` / `BORROW_EXCLUSIVE`.
    pub(crate) borrow_state: AtomicU8,
    /// Number of active split guards (ref-counted).
    pub(crate) split_count: AtomicUsize,
    /// Negative auto-trait marker: `!Send + !Sync`.
    _not_send_sync: PhantomData<*mut ()>,
}

impl Workspace {
    /// Default allocation alignment (64 bytes — cache-line friendly).
    pub const DEFAULT_ALIGNMENT: usize = 64;
    /// Minimum allowed alignment (8 bytes — required by the `Element` type
    /// set's alignment guarantees).
    pub const MIN_ALIGNMENT: usize = 8;
    /// Default allocation capacity (4 KiB).
    pub const DEFAULT_CAPACITY: usize = 4096;

    /// No active borrow.
    pub(crate) const BORROW_NONE: u8 = 0;
    /// One shared (immutable) borrow.
    pub(crate) const BORROW_READ: u8 = 1;
    /// One exclusive (mutable / split) borrow.
    pub(crate) const BORROW_EXCLUSIVE: u8 = 2;

    /// Growth multiplier numerator (×1.5 factor).
    pub(crate) const GROWTH_FACTOR_NUMERATOR: usize = 3;
    /// Growth multiplier denominator.
    pub(crate) const GROWTH_FACTOR_DENOMINATOR: usize = 2;

    /// Returns the current capacity in bytes.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the alignment in bytes.
    pub fn alignment(&self) -> usize {
        self.alignment
    }

    /// Allocate a new workspace with the given `capacity` (bytes) and
    /// `alignment`.
    ///
    /// # Errors
    ///
    /// - `InvalidLayout` — `alignment` is not a power of two or is below
    ///   `MIN_ALIGNMENT`.
    /// - `AllocFailed` — the global allocator returned `null`.
    pub fn new(capacity: usize, alignment: usize) -> Result<Self> {
        if !alignment.is_power_of_two() || alignment < Self::MIN_ALIGNMENT {
            return Err(crate::error::XenonError::Workspace {
                operation: Cow::Borrowed("Workspace::new"),
                category: WorkspaceErrorCategory::InvalidLayout {
                    size: capacity,
                    align: alignment,
                },
                
            });
        }
        let size = capacity.max(1);
        let layout = std::alloc::Layout::from_size_align(size, alignment).map_err(|_| {
            crate::error::XenonError::Workspace {
                operation: Cow::Borrowed("Workspace::new"),
                category: WorkspaceErrorCategory::InvalidLayout {
                    size: capacity,
                    align: alignment,
                },
                
            }
        })?;
        let ptr = if size == 0 {
            NonNull::dangling()
        } else {
            // SAFETY: layout has non-zero size and valid alignment.
            let raw = unsafe { std::alloc::alloc(layout) };
            NonNull::new(raw).ok_or({
                crate::error::XenonError::Workspace {
                    operation: Cow::Borrowed("Workspace::new"),
                    category: WorkspaceErrorCategory::AllocFailed {
                        size,
                        align: alignment,
                    },
                    
                }
            })?
        };
        Ok(Self {
            ptr,
            capacity: size,
            alignment,
            borrow_state: AtomicU8::new(Self::BORROW_NONE),
            split_count: AtomicUsize::new(0),
            _not_send_sync: PhantomData,
        })
    }

    /// Allocate a workspace with default capacity and alignment.
    ///
    /// # Errors
    ///
    /// Returns errors from the underlying allocator, wrapped as
    /// `XenonError::Workspace { AllocationFailed }`.
    pub fn with_default_capacity() -> Result<Self> {
        Self::new(Self::DEFAULT_CAPACITY, Self::DEFAULT_ALIGNMENT)
    }
}

impl Drop for Workspace {
    fn drop(&mut self) {
        // SAFETY: layout was valid at allocation time and ptr is unchanged.
        unsafe {
            let layout =
                std::alloc::Layout::from_size_align_unchecked(self.capacity, self.alignment);
            std::alloc::dealloc(self.ptr.as_ptr(), layout);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workspace_new_default() {
        let ws = Workspace::with_default_capacity().expect("default workspace");
        assert_eq!(ws.capacity(), Workspace::DEFAULT_CAPACITY);
        assert_eq!(ws.alignment(), Workspace::DEFAULT_ALIGNMENT);
    }

    #[test]
    fn test_workspace_new() {
        let ws = Workspace::new(1024, 64).expect("1024-byte workspace");
        assert_eq!(ws.capacity(), 1024);
        assert_eq!(ws.alignment(), 64);
    }

    #[test]
    fn test_workspace_constants() {
        assert_eq!(Workspace::DEFAULT_ALIGNMENT, 64);
        assert_eq!(Workspace::MIN_ALIGNMENT, 8);
        assert_eq!(Workspace::DEFAULT_CAPACITY, 4096);
    }

    #[test]
    fn test_workspace_new_invalid_alignment() {
        // alignment not a power of two → InvalidLayout
        let err = Workspace::new(1024, 7).expect_err("invalid alignment");
        match err {
            crate::error::XenonError::Workspace {
                category:
                    WorkspaceErrorCategory::InvalidLayout {
                        size: 1024,
                        align: 7,
                    },
                ..
            } => {},
            other => panic!("expected InvalidLayout, got {:?}", other),
        }

        // alignment below MIN_ALIGNMENT → InvalidLayout
        let err = Workspace::new(1024, 4).expect_err("below min alignment");
        match err {
            crate::error::XenonError::Workspace {
                category:
                    WorkspaceErrorCategory::InvalidLayout {
                        size: 1024,
                        align: 4,
                    },
                ..
            } => {},
            other => panic!("expected InvalidLayout, got {:?}", other),
        }
    }

    #[test]
    fn test_workspace_drop_no_leak() {
        let ws = Workspace::new(1024, 64).expect("workspace for drop");
        drop(ws);
    }
}
