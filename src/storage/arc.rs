//! Arc-based shared storage (W7T16).
//!
//! `ArcRepr<A>` is a shared read-only storage representation backed by
//! `Arc<SharedBuf<A>>`. O(1) shallow `Clone` via reference-count bump.
//! Public API stays read-only.

use std::sync::Arc;

use crate::element::Element;
use crate::error::XenonError;
use crate::storage::StorageIntoOwned;
#[cfg(test)]
use crate::storage::alloc::AlignedAlloc;
use crate::storage::owned::AlignedBuf;
use crate::storage::traits::IsShared;
use crate::storage::{RawStorage, Storage, StorageShared, StorageSharedExt};

/// Internal shared buffer wrapping an `AlignedBuf`.
///
/// When the last `Arc` is dropped, `AlignedBuf`'s `Drop` releases the
/// aligned allocation with the original layout.
#[derive(Debug)]
pub(crate) struct SharedBuf<A> {
    buf: AlignedBuf<A>,
}

/// Shared read-only storage.
#[derive(Debug)]
pub struct ArcRepr<A> {
    inner: Arc<SharedBuf<A>>,
}

impl<A> ArcRepr<A> {
    /// Creates an empty `ArcRepr<A>`.
    pub fn new() -> Self {
        Self::from_aligned_buf(AlignedBuf::empty())
    }

    pub(crate) fn from_aligned_buf(buf: AlignedBuf<A>) -> Self {
        Self {
            inner: Arc::new(SharedBuf { buf }),
        }
    }

    /// Constructs shared storage from a `Vec`, copying into an aligned buffer.
    pub fn from_vec(data: Vec<A>) -> Result<Self, XenonError>
    where
        A: Copy,
    {
        let buf = AlignedBuf::from_vec(data)?;
        Ok(Self::from_aligned_buf(buf))
    }

    /// Core implementation of `from_vec` with explicit alignment copy.
    pub fn from_vec_aligned(data: Vec<A>) -> Result<Self, XenonError>
    where
        A: Copy,
    {
        let buf = AlignedBuf::from_vec_aligned(data)?;
        Ok(Self::from_aligned_buf(buf))
    }

    /// Creates shared storage filled with zeros.
    ///
    /// # Errors
    ///
    /// Propagates from `AlignedBuf::zeros`:
    /// - `XenonError::InvalidShape { kind: ProductOverflow }` — `len *
    ///   size_of::<A>()` (with alignment slack) exceeds `isize::MAX`.
    /// - `XenonError::AllocationFailed` — the underlying allocator could not
    ///   provide the requested zero-filled aligned buffer.
    pub fn zeros(len: usize) -> Result<Self, XenonError>
    where
        A: Element + Default,
    {
        let buf = AlignedBuf::zeros(len)?;
        Ok(Self::from_aligned_buf(buf))
    }

    /// Creates shared storage filled with clones of `value`.
    ///
    /// # Errors
    ///
    /// Propagates from `AlignedBuf::from_elem`:
    /// - `XenonError::InvalidShape { kind: ProductOverflow }` — `len *
    ///   size_of::<A>()` (with alignment slack) exceeds `isize::MAX`.
    /// - `XenonError::AllocationFailed` — the underlying allocator could not
    ///   provide the requested aligned buffer.
    pub fn from_elem(len: usize, value: A) -> Result<Self, XenonError>
    where
        A: Element + Clone,
    {
        let buf = AlignedBuf::from_elem(len, value)?;
        Ok(Self::from_aligned_buf(buf))
    }

    /// Returns the current reference count. Crate-internal helper.
    pub(crate) fn ref_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }

    /// Checks if this is the sole owner. Crate-internal helper.
    pub(crate) fn is_unique(&self) -> bool {
        Arc::strong_count(&self.inner) == 1
    }
}

impl<A> Clone for ArcRepr<A> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

// SAFETY: ArcRepr owns a single Arc<SharedBuf<A>> whose AlignedBuf<A>
// maintains a non-null, aligned, fully initialized range of len elements
// within one allocation. as_ptr/len forward that stable storage-visible range.
unsafe impl<A: Element> RawStorage for ArcRepr<A> {
    type Elem = A;

    fn as_ptr(&self) -> *const A {
        self.inner.buf.as_ptr()
    }

    fn len(&self) -> usize {
        self.inner.buf.len()
    }
}

// SAFETY: ArcRepr exposes only shared read-only access to the initialized
// AlignedBuf<A> range described by RawStorage.
unsafe impl<A: Element> Storage for ArcRepr<A> {}
// SAFETY: ArcRepr is the crate-controlled shared read-only storage mode.
// Cloning only bumps the Arc refcount and never exposes mutable access.
unsafe impl<A: Element> StorageShared for ArcRepr<A> {}

impl<A: Element> StorageSharedExt for ArcRepr<A> {
    fn is_unique(&self) -> bool {
        self.is_unique()
    }

    fn ref_count(&self) -> usize {
        self.ref_count()
    }
}

impl<A> crate::private::Sealed for ArcRepr<A> {}
// SAFETY: ArcRepr satisfies RawStorage and Sealed, and represents Xenon's
// shared read-only storage category.
unsafe impl<A: Element> IsShared for ArcRepr<A> {}

impl<A: Element + Clone> StorageIntoOwned for ArcRepr<A> {
    fn into_owned_storage(self) -> crate::storage::Owned<A>
    where
        Self::Elem: Clone,
    {
        let align = core::mem::align_of::<A>().max(64);
        let mut buf: AlignedBuf<A> = AlignedBuf::with_capacity_aligned(self.len(), align)
            .expect("allocation failed in ArcRepr::into_owned_storage");
        for i in 0..self.len() {
            // SAFETY: i < len, both src and dst pointers are valid
            unsafe {
                core::ptr::write(buf.as_mut_ptr().add(i), *self.inner.buf.as_ptr().add(i));
            }
            // Increment length after each successful write so that
            // a panic during a later clone() will still drop the
            // prefix elements via AlignedBuf::Drop.
            unsafe {
                buf.set_len(i + 1);
            }
        }
        crate::storage::Owned { data: buf }
    }
}

// ---------------------------------------------------------------------------
// W7T17: Send/Sync + Default + TryFrom for ArcRepr<A>
// ---------------------------------------------------------------------------

// SAFETY: `ArcRepr<A>` shares storage through atomic reference counting.
// Moving the handle across threads is sound when `A` can be sent and shared
// across threads, matching `Arc<[A]>` requirements.
unsafe impl<A: Send + Sync> Send for ArcRepr<A> {}

// SAFETY: `&ArcRepr<A>` may be accessed concurrently and exposes shared reads
// of `A`. Concurrent shared reads are sound when `A: Sync`, and cloned handles
// may move between threads when `A: Send`.
unsafe impl<A: Send + Sync> Sync for ArcRepr<A> {}

impl<A> Default for ArcRepr<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Element + Copy> TryFrom<Vec<A>> for ArcRepr<A> {
    type Error = crate::error::XenonError;

    fn try_from(value: Vec<A>) -> Result<Self, Self::Error> {
        Self::from_vec(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arc_ref_count() {
        let arc = ArcRepr::from_vec(vec![1_i32, 2])
            .expect("ArcRepr::from_vec should succeed for small i32 input");
        let cloned = arc.clone();
        assert_eq!(arc.as_slice(), cloned.as_slice());
        assert_eq!(arc.ref_count(), 2);
        assert!(!arc.is_unique());
    }

    #[test]
    fn test_arc_clone_o1() {
        let arc = ArcRepr::from_vec(vec![1_i32, 2, 3])
            .expect("ArcRepr::from_vec should succeed for small i32 input");
        let ptr = arc.as_ptr();
        let cloned = arc.clone();

        assert_eq!(arc.as_ptr(), ptr);
        assert_eq!(cloned.as_ptr(), ptr);
        assert_eq!(arc.ref_count(), 2);
    }

    #[test]
    fn test_arc_alignment_preserved() {
        let arc = ArcRepr::from_vec(vec![1_i32, 2, 3, 4])
            .expect("ArcRepr::from_vec should succeed for small i32 input");
        assert_eq!((arc.as_ptr() as usize) % AlignedAlloc::DEFAULT_ALIGNMENT, 0);
    }

    #[test]
    fn test_arc_zst() {
        // ArcRepr::from_elem requires A: Element + Clone, which () does not
        // satisfy. Test the ZST code path through AlignedBuf directly.
        let buf = AlignedBuf::<()>::from_elem(1024, ())
            .expect("AlignedBuf::from_elem should succeed for ZST input");
        assert_eq!(buf.len(), 1024);
    }

    #[test]
    fn test_arc_empty() {
        let arc = ArcRepr::<i32>::from_vec(Vec::new())
            .expect("ArcRepr::from_vec should succeed for empty input");
        assert!(arc.is_empty());
        assert!(arc.is_unique());
    }

    // W7T17 tests
    #[test]
    fn test_arc_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ArcRepr<i32>>();
    }

    #[test]
    fn test_arc_default() {
        let arc = ArcRepr::<i32>::default();
        assert!(arc.is_empty());
    }

    #[test]
    fn test_arc_try_from_vec() {
        let arc = ArcRepr::<i32>::try_from(vec![1, 2, 3])
            .expect("ArcRepr::try_from should succeed for small i32 input");
        assert_eq!(arc.len(), 3);
        assert_eq!(arc.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_arc_concurrent_read() {
        let arc = ArcRepr::from_vec(vec![1_i32, 2, 3])
            .expect("ArcRepr::from_vec should succeed for small i32 input");
        let left = arc.clone();
        let right = arc.clone();
        let a = std::thread::spawn(move || {
            left.as_slice().iter().copied().sum::<i32>()
        });
        let b = std::thread::spawn(move || {
            right.as_slice().iter().copied().sum::<i32>()
        });
        assert_eq!(a.join().expect("thread should not panic"), 6);
        assert_eq!(b.join().expect("thread should not panic"), 6);
    }

    #[test]
    fn test_arc_cloned_handles_preserve_read_only_data() {
        let arc = ArcRepr::from_vec(vec![4_i64, 5])
            .expect("ArcRepr::from_vec should succeed for small i64 input");
        let cloned = arc.clone();
        assert_eq!(arc.as_slice(), cloned.as_slice());
    }
}
