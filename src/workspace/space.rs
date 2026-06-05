//! Temporary aligned scratch buffer with borrow-state tracking.
//!
//! [`Workspace`] provides a pre-allocated, aligned memory region that can be
//! borrowed (immutably or mutably), split into sub-spaces, and grown on demand.
//! Borrow state is managed via atomic CAS with RAII guards.

use core::marker::PhantomData;
use core::ptr::NonNull;
use core::sync::atomic::AtomicU8;
use core::sync::atomic::AtomicUsize;
use core::sync::atomic::Ordering;

use std::alloc::{Layout, alloc};
use std::borrow::Cow;

use crate::error::{Result, XenonError};
use crate::error::{WorkspaceBorrowKind, WorkspaceBorrowState, WorkspaceErrorCategory};
use super::borrow::{WorkspaceBorrow, WorkspaceBorrowMut};
use super::split::SplitBorrowMut;

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

/// Temporary aligned workspace for internal scratch buffers.
///
/// # Fields
///
/// All fields except `_not_send_sync` are `pub(crate)` so that sibling
/// modules (`borrow`, `split`) can implement their logic directly
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
            return Err(XenonError::Workspace {
                operation: Cow::Borrowed("Workspace::new"),
                category: WorkspaceErrorCategory::InvalidLayout {
                    size: capacity,
                    align: alignment,
                },
            });
        }
        let size = capacity.max(1);
        let layout = Layout::from_size_align(size, alignment).map_err(|_| {
            XenonError::Workspace {
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
            let raw = unsafe { alloc(layout) };
            NonNull::new(raw).ok_or({
                XenonError::Workspace {
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

    /// Crate-internal probe to verify no residual borrow state before
    /// reallocation. Uses `Acquire` to pair with the `Release` stores
    /// performed by guard `Drop` impls.
    #[expect(dead_code, reason = "defense-in-depth helper; ensure_capacity uses direct load")]
    pub(crate) fn is_borrowed(&self) -> bool {
        self.borrow_state.load(Ordering::Acquire) != Self::BORROW_NONE
    }

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

    /// Split the workspace mutably at the specified position into two sub-spaces.
    ///
    /// # Complexity
    ///
    /// O(1) — pointer arithmetic only, no memory allocation.
    ///
    /// # RAII Behavior
    ///
    /// Dropping **the last** `SplitBorrowMut` releases the workspace for
    /// re-use. Reference counting ensures the workspace is not re-borrowable
    /// until ALL sub-spaces (including those from recursive `split_at_mut`
    /// calls) are dropped.
    ///
    /// # Errors
    ///
    /// - `XenonError::Workspace { SplitOutOfBounds }` — `mid > capacity`
    /// - `XenonError::Workspace { BorrowConflict }` — already borrowed
    pub fn split_at_mut(
        &mut self,
        mid: usize,
    ) -> crate::error::Result<(SplitBorrowMut<'_>, SplitBorrowMut<'_>)> {
        // 1. Bounds check FIRST — must not leave borrow_state in a partially
        //    transitioned state if mid is out of range.
        if mid > self.capacity {
            return Err(XenonError::workspace_split_oob(
                "Workspace::split_at_mut",
                mid,
                self.capacity,
            ));
        }

        // 2. CAS to acquire exclusive borrow.
        if self
            .borrow_state
            .compare_exchange(
                Self::BORROW_NONE,
                Self::BORROW_EXCLUSIVE,
                Ordering::Acquire,
                Ordering::Relaxed,
            )
            .is_err()
        {
            return Err(XenonError::workspace_borrow_conflict(
                "Workspace::split_at_mut",
                WorkspaceBorrowKind::Split,
                current_borrow_state(self),
            ));
        }

        // 3. Initialize split_count to 2 (two sub-spaces about to be created).
        //    `Release` so subsequent guard Drops see the initialization.
        self.split_count.store(2, Ordering::Release);

        let left_ptr = self.ptr;
        // SAFETY: mid <= capacity (checked above), so ptr + mid is within
        // the allocation.
        let right_ptr = unsafe { NonNull::new_unchecked(self.ptr.as_ptr().add(mid)) };

        Ok((
            SplitBorrowMut {
                ptr: left_ptr,
                len: mid,
                workspace: self,
                split_count: &self.split_count,
            },
            SplitBorrowMut {
                ptr: right_ptr,
                len: self.capacity - mid,
                workspace: self,
                split_count: &self.split_count,
            },
        ))
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

    /// Verify default workspace has correct capacity and alignment.
    #[test]
    fn test_workspace_new_default() {
        let ws = Workspace::with_default_capacity().expect("default workspace");
        assert_eq!(ws.capacity(), Workspace::DEFAULT_CAPACITY);
        assert_eq!(ws.alignment(), Workspace::DEFAULT_ALIGNMENT);
    }

    /// Verify custom-sized workspace allocation.
    #[test]
    fn test_workspace_new() {
        let ws = Workspace::new(1024, 64).expect("1024-byte workspace");
        assert_eq!(ws.capacity(), 1024);
        assert_eq!(ws.alignment(), 64);
    }

    /// Verify public constant values.
    #[test]
    fn test_workspace_constants() {
        assert_eq!(Workspace::DEFAULT_ALIGNMENT, 64);
        assert_eq!(Workspace::MIN_ALIGNMENT, 8);
        assert_eq!(Workspace::DEFAULT_CAPACITY, 4096);
    }

    /// Invalid alignment (non-power-of-two, below minimum) must be rejected.
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

    /// Dropping a workspace must not leak memory.
    #[test]
    fn test_workspace_drop_no_leak() {
        let ws = Workspace::new(1024, 64).expect("workspace for drop");
        drop(ws);
    }

    /// Immutable borrow returns a guard covering the full workspace.
    #[test]
    fn test_borrow_basic() {
        let workspace = Workspace::new(64, 64).expect("64-byte workspace");
        let mut guard = workspace.borrow().expect("immutable borrow");
        let view = guard.as_maybe_uninit_slice();
        assert_eq!(view.len(), 64);
    }

    /// Mutable borrow returns a writable view of the workspace.
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

    /// Second borrow while first is active must fail.
    #[test]
    fn test_borrow_double_fails() {
        let workspace = Workspace::new(64, 64).expect("64-byte workspace");
        let _g1 = workspace.borrow().expect("first borrow");
        // Second shared borrow must conflict — current design allows at most
        // one active read guard.
        assert!(workspace.borrow().is_err());
    }

    /// After dropping a borrow guard, re-borrowing must succeed.
    #[test]
    fn test_borrow_after_drop() {
        let workspace = Workspace::new(64, 64).expect("64-byte workspace");
        {
            let _g = workspace.borrow().expect("scoped borrow");
        }
        // Re-borrow after the previous guard is dropped must succeed.
        assert!(workspace.borrow().is_ok());
    }

    /// Out-of-bounds `assume_init_slice` rejects, within-bounds accepts.
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

    /// `ensure_capacity` with smaller value must not reallocate.
    #[test]
    fn test_ensure_capacity_no_grow() {
        let mut workspace = Workspace::new(64, 64).expect("64-byte workspace");
        workspace
            .ensure_capacity(32)
            .expect("ensure capacity in test");
        assert_eq!(workspace.capacity(), 64);
    }

    /// `ensure_capacity` with larger value must grow and preserve alignment.
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
            .store(Workspace::BORROW_EXCLUSIVE, core::sync::atomic::Ordering::Release);
        let err = workspace.ensure_capacity(256);
        assert!(err.is_err());
        // Restore so Drop is balanced.
        workspace
            .borrow_state
            .store(Workspace::BORROW_NONE, core::sync::atomic::Ordering::Release);
    }

    /// Basic binary split produces correct sub-space lengths.
    #[test]
    fn test_split_at_mut_basic() {
        let mut workspace = Workspace::new(100, 64).expect("workspace in test");
        let (left, right) = workspace.split_at_mut(40).expect("split in test");
        assert_eq!(left.len(), 40);
        assert_eq!(right.len(), 60);
    }

    /// Recursive split produces three sub-spaces with correct lengths.
    #[test]
    fn test_split_at_mut_recursive() {
        let mut workspace = Workspace::new(100, 64).expect("workspace in test");
        let (left, right) = workspace.split_at_mut(40).expect("split in test");
        let (right_a, right_b) = right.split_at_mut(30).expect("split in test");
        assert_eq!(left.len(), 40);
        assert_eq!(right_a.len(), 30);
        assert_eq!(right_b.len(), 30);
    }

    /// Out-of-bounds split must fail without corrupting borrow state.
    #[test]
    fn test_split_at_mut_oob() {
        let mut workspace = Workspace::new(8, 64).expect("workspace in test");
        // Bounds check happens BEFORE CAS, so borrow_state must remain
        // BORROW_NONE and a subsequent borrow must succeed.
        assert!(workspace.split_at_mut(9).is_err());
        assert!(workspace.borrow().is_ok());
    }

    /// Drop ordering is irrelevant to correctness: regardless of which sibling
    /// drops last, the workspace must be re-usable only AFTER the last sibling
    /// drops.
    #[test]
    fn test_recursive_split_drop_order_independent() {
        // Scenario A: drop in [left, right_a, right_b] order.
        {
            let mut workspace = Workspace::new(100, 64).expect("workspace in test");
            let (left, right) = workspace.split_at_mut(40).expect("split in test");
            let (right_a, right_b) = right.split_at_mut(30).expect("split in test");
            drop(left);
            // Workspace still has 2 active guards — must not be re-borrowable.
            drop(right_a);
            // Still 1 active guard.
            drop(right_b);
            // All guards dropped — must be re-borrowable.
            assert!(workspace.borrow().is_ok());
        }
        // Scenario B: drop in [right_b, right_a, left] order.
        {
            let mut workspace = Workspace::new(100, 64).expect("workspace in test");
            let (left, right) = workspace.split_at_mut(40).expect("split in test");
            let (right_a, right_b) = right.split_at_mut(30).expect("split in test");
            drop(right_b);
            drop(right_a);
            drop(left);
            assert!(workspace.borrow().is_ok());
        }
        // Scenario C: drop in [right_a, left, right_b] order — interleaved.
        {
            let mut workspace = Workspace::new(100, 64).expect("workspace in test");
            let (left, right) = workspace.split_at_mut(40).expect("split in test");
            let (right_a, right_b) = right.split_at_mut(30).expect("split in test");
            drop(right_a);
            drop(left);
            drop(right_b);
            assert!(workspace.borrow().is_ok());
        }
    }
}

