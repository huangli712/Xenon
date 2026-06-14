//! Mutable view storage.
//!
//! `ViewMutRepr<'a, A>` is an exclusive-borrow storage representation.
//! Does not implement `Clone` or `Copy`.

use core::ptr::write;
use core::marker::PhantomData;

use crate::private::Sealed;

use super::alloc::AlignedAlloc;
use super::buffer::AlignedBuf;
use super::Owned;
use super::ViewRepr;
use super::{RawStorage, Storage, StorageMut, StorageIntoOwned};

/// Exclusive mutable borrowed view storage.
///
/// `ViewMutRepr` stores a raw mutable pointer and element count with a
/// `PhantomData<&'a mut A>` lifetime marker — no allocation, no reference
/// counting. Does not implement `Clone` or `Copy` because the exclusive borrow
/// guarantees unique ownership of the data. Mutable access is provided through
/// [`StorageMut`]; conversion to owned storage is available via
/// [`StorageIntoOwned::into_owned_storage`] (O(n) deep copy).
///
/// # Thread Safety
///
/// `ViewMutRepr<'a, A>` is `Send` when `A: Send`, allowing transfer of the
/// exclusive write capability to another thread. It intentionally does NOT
/// implement `Sync` — sharing `&ViewMutRepr` across threads would create
/// multiple write-capable aliases, violating the exclusive borrow contract.
#[derive(Debug)]
pub struct ViewMutRepr<'a, A> {
    ptr: *mut A,
    len: usize,
    _marker: PhantomData<&'a mut A>,
}

/// Short alias for [`ViewMutRepr`].
pub type ViewMut<'a, A> = ViewMutRepr<'a, A>;

impl<'a, A> ViewMutRepr<'a, A> {
    /// Creates a `ViewMutRepr` from a raw mutable pointer and length.
    ///
    /// # Safety
    ///
    /// Caller must guarantee an exclusive mutable borrow of the range for
    /// lifetime `'a`: `ptr` is non-null, aligned to `align_of::<A>()`,
    /// points to `len` initialized elements inside one allocation, and no
    /// other mutable or shared alias accesses the same memory during `'a`.
    pub unsafe fn from_raw_parts_mut(ptr: *mut A, len: usize) -> Self {
        Self {
            ptr,
            len,
            _marker: PhantomData,
        }
    }

    /// Builds a mutable view from a mutable slice borrow.
    pub fn from_mut_slice(slice: &'a mut [A]) -> Self {
        Self {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
            _marker: PhantomData,
        }
    }

    /// Returns a read-only reborrow tied to `&self` (`'_`), not the
    /// original `'a`, so the read alias cannot outlive this borrow of
    /// the mutable view.
    pub fn view(&self) -> ViewRepr<'_, A> {
        // SAFETY: self.ptr is non-null, aligned, len elements initialized.
        // The returned ViewRepr has lifetime tied to &self, enforcing
        // that it cannot outlive this mutable view borrow.
        unsafe { ViewRepr::from_raw_parts(self.ptr as *const A, self.len) }
    }

    /// Returns a sub-view of `self`.
    ///
    /// # Panics
    ///
    /// Panics if `start > end` or `end > self.len`.
    pub fn view_mut(&mut self, start: usize, end: usize) -> ViewMutRepr<'_, A> {
        assert!(start <= end && end <= self.len);
        ViewMutRepr {
            // SAFETY: start <= end <= len, so start is within bounds
            ptr: unsafe { self.ptr.add(start) },
            len: end - start,
            _marker: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// Send for ViewMutRepr<'a, A>
// ---------------------------------------------------------------------------

// SAFETY: `ViewMutRepr` represents an exclusive mutable borrow of a logical
// tensor region. Moving it to another thread transfers that exclusive access;
// it does not create aliases. Moving contained element access across threads is
// sound exactly when `A: Send`.
unsafe impl<'a, A: Send> Send for ViewMutRepr<'a, A> {}

impl<'a, A> Sealed for ViewMutRepr<'a, A> {}

// SAFETY: ViewMutRepr is created only from an exclusive mutable borrow or an
// unsafe raw-parts constructor whose caller guarantees a non-null, aligned,
// initialized single-allocation range valid for 'a.
unsafe impl<'a, A> RawStorage for ViewMutRepr<'a, A> {
    type Elem = A;

    /// Returns the base pointer to the borrowed mutable data.
    fn as_ptr(&self) -> *const A {
        self.ptr as *const A
    }

    /// Returns the number of elements in the mutable view.
    fn len(&self) -> usize {
        self.len
    }
}

// SAFETY: ViewMutRepr forwards the same initialized storage-visible range as
// RawStorage and can expose shared reads for the duration of &self.
unsafe impl<'a, A> Storage for ViewMutRepr<'a, A> {}

// SAFETY: ViewMutRepr grants exclusive mutable access over an exclusive
// borrow, so mutable references and slices derived from it are unique, and
// `&mut self` guarantees the storage-visible range tracked by ptr/len.
unsafe impl<'a, A> StorageMut for ViewMutRepr<'a, A> {
    /// Returns the raw mutable pointer to the borrowed data.
    fn as_mut_ptr(&mut self) -> *mut A {
        self.ptr
    }
}

// Intentionally no `Sync` impl: sharing `&ViewMutRepr` would share an exclusive
// write capability and violate the aliasing model.
impl<'a, A: Clone> StorageIntoOwned for ViewMutRepr<'a, A> {
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
            .expect("allocation failed in ViewMutRepr::into_owned_storage");
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
    use crate::dimension::Ix1;
    use crate::tensor::Tensor1;

    /// Mutable view provides exclusive write access.
    #[test]
    fn test_view_mut_exclusive() {
        let mut data = [1_i32, 2, 3];
        let mut view = ViewMutRepr::from_mut_slice(&mut data);
        *view
            .get_mut(1)
            .expect("index 1 should be in bounds for length-3 mutable view") = 5;
        assert_eq!(view.as_slice(), &[1, 5, 3]);
    }

    /// Constructs a mutable view via `from_raw_parts_mut` and verifies
    /// data access and mutation.
    #[test]
    fn test_view_mut_from_raw_parts_mut() {
        let mut data = [10_i32, 20, 30];
        let mut view =
            unsafe { ViewMutRepr::from_raw_parts_mut(data.as_mut_ptr(), data.len()) };
        assert_eq!(view.as_slice(), &[10, 20, 30]);
        view.fill(0);
        assert_eq!(data, [0, 0, 0]);
    }

    /// `view()` creates a read-only reborrow from a mutable view.
    #[test]
    fn test_view_mut_view() {
        let mut data = [3_i32, 1, 4];
        let view_mut = ViewMutRepr::from_mut_slice(&mut data);
        let ro = view_mut.view();
        assert_eq!(ro.as_slice(), &[3, 1, 4]);
    }

    /// `view_mut()` returns a valid sub-range; panics when start > end or
    /// end > len.
    #[test]
    fn test_view_mut_sub_view() {
        let mut data = [5_i32, 10, 15, 20];
        let mut view = ViewMutRepr::from_mut_slice(&mut data);
        // `view_mut()` borrows `&mut self`, so the returned sub-view
        // prevents further access to the original `view`. Wrap the sub-view
        // in a block so the borrow ends at the closing brace, freeing
        // `view` for the subsequent `as_slice()` assertion.
        {
            let mut sub = view.view_mut(1, 3);
            assert_eq!(sub.as_slice(), &[10, 15]);
            sub.fill(99);
        }
        assert_eq!(view.as_slice(), &[5, 99, 99, 20]);

        // empty sub-view
        let empty = view.view_mut(2, 2);
        assert!(empty.is_empty());
    }

    /// `view_mut()` panics when start > end.
    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_view_mut_sub_view_panics_on_inverted_range() {
        let mut data = [1_i32, 2];
        let mut view = ViewMutRepr::from_mut_slice(&mut data);
        let _ = view.view_mut(2, 1);
    }

    /// `view_mut()` panics when end > len.
    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_view_mut_sub_view_panics_on_oob() {
        let mut data = [1_i32, 2];
        let mut view = ViewMutRepr::from_mut_slice(&mut data);
        let _ = view.view_mut(0, 3);
    }

    /// `into_owned_storage` produces a detached deep copy.
    #[test]
    fn test_view_mut_into_owned_storage() {
        let mut data = [7_i32, 8, 9];
        let view = ViewMutRepr::from_mut_slice(&mut data);
        let mut owned = view.into_owned_storage();

        assert_eq!(owned.as_slice(), &[7, 8, 9]);

        owned.as_mut_slice()[0] = 99;

        // mutated copy does not affect original data
        assert_eq!(owned.as_slice(), &[99, 8, 9]);
        assert_eq!(data, [7, 8, 9]);
    }

    /// Compile-time assertion: ViewMutRepr must NOT implement Clone.
    /// If it did, the blanket impl for `T: Clone` would conflict.
    #[test]
    fn test_view_mut_no_clone() {
        trait FailsIfViewMutBecomesClone {}
        impl<T: Clone> FailsIfViewMutBecomesClone for T {}
        impl<'a, A> FailsIfViewMutBecomesClone for ViewMutRepr<'a, A> {}

        let mut data = [1_i32, 2, 3];
        let view = ViewMutRepr::from_mut_slice(&mut data);
        fn assert_marker<T: FailsIfViewMutBecomesClone>(_: T) {}
        assert_marker(view);
    }

    /// Compile-time assertion: ViewMutRepr must NOT implement Sync.
    /// Sharing `&ViewMutRepr` across threads would expose a second
    /// write-capable alias, violating the exclusive-borrow model; this
    /// guards against an accidental `Sync` impl being added later.
    ///
    /// This is the ambiguity trick that `assert_not_impl_any!` expands to:
    /// the `Sync`-gated `u8` impl applies only if the type is `Sync`. For a
    /// concrete `!Sync` type only the blanket `()` impl matches, so the `_`
    /// resolves unambiguously and this compiles. A future `Sync` impl would
    /// make both impls apply, leaving `_` ambiguous and failing the build.
    #[test]
    fn test_view_mut_not_sync() {
        trait AmbiguousIfSync<A> {
            fn marker() {}
        }
        impl<T> AmbiguousIfSync<()> for T {}
        impl<T: Sync> AmbiguousIfSync<u8> for T {}

        // Resolves only while `ViewMutRepr<'static, i32>: !Sync`.
        let _ = <ViewMutRepr<'static, i32> as AmbiguousIfSync<_>>::marker;
    }

    /// Mutable view implements Send for Send-compatible element types.
    #[test]
    fn test_view_mut_send() {
        fn assert_send<T: Send>() {}
        assert_send::<ViewMutRepr<'_, f64>>();
    }

    /// A mutable view can be sent across threads for exclusive writes.
    #[test]
    fn test_view_mut_cross_thread_write() {
        let mut tensor = Tensor1::from_shape_vec(Ix1(2), vec![1_i32, 2])
            .expect("Tensor1::from_shape_vec should succeed for valid shape");
        thread::scope(|scope| {
            let mut view = tensor.view_mut();
            let handle = scope.spawn(move || {
                view.fill(7);
            });
            handle.join().expect("thread should not panic");
        });
        assert_eq!(
            tensor.as_slice().expect("from_shape_vec produces F-contiguous tensor"),
            &[7, 7]
        );
    }
}
