//! Immutable view storage (W7T14).
//!
//! `ViewRepr<'a, A>` is a shared-borrow read-only storage representation.
//! O(1) `Copy`/`Clone` metadata, no allocation.

use crate::storage::Owned;
use crate::storage::RawStorage;
use crate::storage::Storage;
use crate::storage::StorageIntoOwned;
use crate::storage::owned::AlignedBuf;
use crate::storage::traits::IsView;

/// Immutable view over borrowed data.
#[derive(Debug, Clone, Copy)]
pub struct ViewRepr<'a, A> {
    ptr: *const A,
    len: usize,
    _marker: core::marker::PhantomData<&'a A>,
}

impl<'a, A> crate::private::Sealed for ViewRepr<'a, A> {}

impl<'a, A> ViewRepr<'a, A> {
    /// Creates a `ViewRepr` from a raw pointer and length.
    ///
    /// # Safety
    ///
    /// Caller must guarantee that `ptr` is non-null, aligned to
    /// `align_of::<A>()`, points to `len` initialized elements inside one
    /// allocation, remains valid for lifetime `'a`, and has no mutable alias
    /// for the same memory during `'a`.
    pub unsafe fn from_raw_parts(ptr: *const A, len: usize) -> Self {
        Self {
            ptr,
            len,
            _marker: core::marker::PhantomData,
        }
    }

    /// Creates a `ViewRepr` from a shared slice.
    pub fn from_slice(slice: &'a [A]) -> Self {
        Self {
            ptr: slice.as_ptr(),
            len: slice.len(),
            _marker: core::marker::PhantomData,
        }
    }

    /// Returns a sub-view of `self`.
    ///
    /// # Panics
    ///
    /// Panics if `start > end` or `end > self.len`.
    pub fn view(&self, start: usize, end: usize) -> Self {
        assert!(start <= end && end <= self.len);
        Self {
            // SAFETY: start <= end <= len, so start is within bounds
            ptr: unsafe { self.ptr.add(start) },
            len: end - start,
            _marker: core::marker::PhantomData,
        }
    }

    /// Returns a sub-view using a `Range<usize>`.
    pub fn slice(&self, range: core::ops::Range<usize>) -> Self {
        self.view(range.start, range.end)
    }
}

// SAFETY: ptr is non-null, aligned, within one allocation; len is known.
unsafe impl<'a, A> RawStorage for ViewRepr<'a, A> {
    type Elem = A;

    fn as_ptr(&self) -> *const A {
        self.ptr
    }

    fn len(&self) -> usize {
        self.len
    }
}

unsafe impl<'a, A> Storage for ViewRepr<'a, A> {}
unsafe impl<'a, A> IsView for ViewRepr<'a, A> {}

// SAFETY: ViewRepr only allows shared (read-only) access.
// Safe to send between threads if A: Sync (shared refs are Send when T: Sync).
unsafe impl<'a, A: Sync> Send for ViewRepr<'a, A> {}
// SAFETY: Multiple threads can hold &ViewRepr simultaneously if A: Sync.
unsafe impl<'a, A: Sync> Sync for ViewRepr<'a, A> {}

impl<'a, A: Clone> StorageIntoOwned for ViewRepr<'a, A> {
    fn into_owned_storage(self) -> Owned<A>
    where
        Self::Elem: Clone,
    {
        let align = core::mem::align_of::<A>().max(64);
        let mut buf: AlignedBuf<A> = AlignedBuf::with_capacity_aligned(self.len, align)
            .expect("allocation failed in ViewRepr::into_owned_storage");
        for i in 0..self.len {
            // SAFETY: i < len, both src and dst pointers are valid
            unsafe {
                core::ptr::write(buf.as_mut_ptr().add(i), (*self.ptr.add(i)).clone());
            }
            // Increment length after each successful write so that
            // a panic during a later clone() will still drop the
            // prefix elements via AlignedBuf::Drop.
            unsafe {
                buf.set_len(i + 1);
            }
        }
        Owned { data: buf }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_view_from_slice() {
        let data = [1_i32, 2, 3];
        let view = ViewRepr::from_slice(&data);
        assert_eq!(view.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_view_clone_o1() {
        let data = [1_i32, 2, 3];
        let view = ViewRepr::from_slice(&data);
        let cloned = view;

        assert_eq!(view.as_ptr(), cloned.as_ptr());
        assert_eq!(view.len(), cloned.len());
    }

    #[test]
    fn test_view_lifetime() {
        fn assert_lifetime<'a>(slice: &'a [i32]) -> ViewRepr<'a, i32> {
            ViewRepr::from_slice(slice)
        }

        let data = [1_i32, 2, 3];
        let view = assert_lifetime(&data);
        assert_eq!(view.as_slice(), &[1, 2, 3]);
    }
}
