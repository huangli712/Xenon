//! Immutable view storage.
//!
//! `ViewRepr<'a, A>` is a shared-borrow read-only storage representation.
//! O(1) `Copy`/`Clone` metadata, no allocation.

use core::ptr::write;
use core::marker::PhantomData;

use crate::private::Sealed;

use super::alloc::AlignedAlloc;
use super::buffer::AlignedBuf;
use super::Owned;
use super::{RawStorage, Storage, StorageIntoOwned};

/// Immutable borrowed read-only view storage.
///
/// `ViewRepr` stores a raw pointer and element count with a `PhantomData<&'a A>`
/// lifetime marker — no allocation, no reference counting. `Copy` and `Clone`
/// are O(1) metadata-only operations. The public API is read-only; mutable
/// access requires converting to [`Owned`] via
/// [`StorageIntoOwned::into_owned_storage`] (O(n) deep copy).
///
/// # Thread Safety
///
/// `ViewRepr<'a, A>` is `Send` and `Sync` when `A: Sync`, allowing
/// cross-thread shared reads without requiring `A: Send` (the view itself
/// contains no owned `A` values).
#[derive(Debug, Clone, Copy)]
pub struct ViewRepr<'a, A> {
    ptr: *const A,
    len: usize,
    _marker: PhantomData<&'a A>,
}

/// Short alias for [`ViewRepr`].
pub type View<'a, A> = ViewRepr<'a, A>;

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
            _marker: PhantomData,
        }
    }

    /// Creates a `ViewRepr` from a shared slice.
    pub fn from_slice(slice: &'a [A]) -> Self {
        Self {
            ptr: slice.as_ptr(),
            len: slice.len(),
            _marker: PhantomData,
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
            _marker: PhantomData,
        }
    }

    /// Returns a sub-view using a `Range<usize>`.
    pub fn slice(&self, range: core::ops::Range<usize>) -> Self {
        self.view(range.start, range.end)
    }
}

// ---------------------------------------------------------------------------
// Send/Sync for ViewRepr<'a, A>
// ---------------------------------------------------------------------------

// SAFETY: `ViewRepr` is a borrowed read-only view. Moving it to another
// thread only moves shared access to `A` values, which is sound exactly when
// `A: Sync`. The lifetime `'a` still prevents outliving the borrowed storage.
unsafe impl<'a, A: Sync> Send for ViewRepr<'a, A> {}

// SAFETY: Sharing `&ViewRepr` across threads permits only shared reads of
// `A` through the original borrow. Shared reads are thread-safe when `A: Sync`.
unsafe impl<'a, A: Sync> Sync for ViewRepr<'a, A> {}

impl<'a, A> Sealed for ViewRepr<'a, A> {}

// SAFETY: ptr is non-null, aligned, within one allocation; len is known.
unsafe impl<'a, A> RawStorage for ViewRepr<'a, A> {
    type Elem = A;

    /// Returns the base pointer to the borrowed data.
    fn as_ptr(&self) -> *const A {
        self.ptr
    }

    /// Returns the number of elements in the view.
    fn len(&self) -> usize {
        self.len
    }
}

// SAFETY: ViewRepr exposes only shared read-only access to the initialized
// range described by RawStorage.
unsafe impl<'a, A> Storage for ViewRepr<'a, A> {}

impl<'a, A: Clone> StorageIntoOwned for ViewRepr<'a, A> {
    /// Copies the borrowed data into a fresh `Owned` buffer (O(n)).
    ///
    /// Elements are cloned one-by-one into a new 64-byte aligned allocation.
    /// The result is independent of the original borrowed data.
    fn into_owned_storage(self) -> Owned<A>
    where
        Self::Elem: Clone,
    {
        let align = core::mem::align_of::<A>().max(AlignedAlloc::DEFAULT_ALIGNMENT);
        let mut buf: AlignedBuf<A> = AlignedBuf::with_capacity_aligned(self.len, align)
            .expect("allocation failed in ViewRepr::into_owned_storage");
        for i in 0..self.len {
            // SAFETY: i < len, both src and dst pointers are valid
            unsafe {
                write(buf.as_mut_ptr().add(i), (*self.ptr.add(i)).clone());
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
    use std::thread;

    use super::*;
    use crate::StorageMut;
    use crate::dimension::Ix1;
    use crate::tensor::Tensor1;

    /// Creates a view from a slice and verifies data access.
    #[test]
    fn test_view_from_slice() {
        let data = [1_i32, 2, 3];
        let view = ViewRepr::from_slice(&data);
        assert_eq!(view.as_slice(), &[1, 2, 3]);
    }

    /// Clone is O(1) — both handles point to the same data.
    #[test]
    fn test_view_clone_o1() {
        let data = [1_i32, 2, 3];
        let view = ViewRepr::from_slice(&data);
        let cloned = view;

        assert_eq!(view.as_ptr(), cloned.as_ptr());
        assert_eq!(view.len(), cloned.len());
    }

    /// View lifetime is tied to the borrowed data.
    #[test]
    fn test_view_lifetime() {
        fn assert_lifetime<'a>(slice: &'a [i32]) -> ViewRepr<'a, i32> {
            ViewRepr::from_slice(slice)
        }

        let data = [1_i32, 2, 3];
        let view = assert_lifetime(&data);
        assert_eq!(view.as_slice(), &[1, 2, 3]);
    }

    /// Constructs a view via `from_raw_parts` and verifies data access.
    #[test]
    fn test_view_from_raw_parts() {
        let data = [10_i32, 20, 30];
        let view = unsafe { ViewRepr::from_raw_parts(data.as_ptr(), data.len()) };
        assert_eq!(view.as_slice(), &[10, 20, 30]);
        assert_eq!(view.len(), 3);
    }

    /// `view()` returns a valid sub-range; panics when start > end or
    /// end > len.
    #[test]
    fn test_view_sub_view() {
        let data = [5_i32, 10, 15, 20];
        let view = ViewRepr::from_slice(&data);

        let sub = view.view(1, 3);
        assert_eq!(sub.as_slice(), &[10, 15]);
        assert_eq!(sub.len(), 2);

        // empty sub-view
        let empty = view.view(2, 2);
        assert!(empty.is_empty());
    }

    /// `view()` panics when start > end.
    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_view_sub_view_panics_on_inverted_range() {
        let data = [1_i32, 2];
        let view = ViewRepr::from_slice(&data);
        let _ = view.view(2, 1);
    }

    /// `view()` panics when end > len.
    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_view_sub_view_panics_on_oob() {
        let data = [1_i32, 2];
        let view = ViewRepr::from_slice(&data);
        let _ = view.view(0, 3);
    }

    /// `into_owned_storage` produces a detached deep copy.
    #[test]
    fn test_view_into_owned_storage() {
        let data = [7_i32, 8, 9];
        let view = ViewRepr::from_slice(&data);
        let mut owned = view.into_owned_storage();

        assert_eq!(owned.as_slice(), &[7, 8, 9]);

        owned.as_mut_slice()[0] = 99;

        // mutated copy does not affect original data
        assert_eq!(owned.as_slice(), &[99, 8, 9]);
        assert_eq!(data, [7, 8, 9]);
    }

    /// View implements Send + Sync when the element type does.
    #[test]
    fn test_view_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ViewRepr<'_, f64>>();
    }

    /// View can be sent across threads for shared reads.
    #[test]
    fn test_view_cross_thread() {
        let tensor = Tensor1::from_shape_vec(Ix1(3), vec![1_i32, 2, 3])
            .expect("Tensor1::from_shape_vec should succeed for valid shape");
        thread::scope(|scope| {
            let view = tensor.view();
            let handle = scope.spawn(move || view.len());
            assert_eq!(handle.join().expect("thread should not panic"), 3);
        });
    }

    /// Multiple threads can concurrently read through a view.
    #[test]
    fn test_view_read_only_across_threads() {
        let tensor = Tensor1::from_shape_vec(Ix1(2), vec![10_i32, 20])
            .expect("Tensor1::from_shape_vec should succeed for valid shape");
        thread::scope(|scope| {
            let view = tensor.view();
            let handle = scope.spawn(move || view.iter().copied().sum::<i32>());
            assert_eq!(handle.join().expect("thread should not panic"), 30);
        });
    }
}
