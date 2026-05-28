//! Owned aligned storage (W7T8-W7T11).
//!
//! `Owned<A>` is the owning heap-allocated storage backed by `AlignedBuf<A>`.
//! Construction goes through `AlignedAlloc` for 64-byte alignment.

use core::mem::{align_of, size_of};

use crate::element::Element;
use crate::error::XenonError;
#[cfg(test)]
use crate::error::InvalidShapeKind;
use crate::storage::ArcRepr;
use crate::storage::RawStorage;
use crate::storage::Storage;
use crate::storage::StorageIntoOwned;
use crate::storage::alloc::AlignedAlloc;
use crate::storage::buffer::{AlignedBuf, allocation_size};
use crate::storage::traits::IsOwned;
use crate::storage::{RawStorageMut, StorageMut, StorageOwned};

// ---------------------------------------------------------------------------
// Owned<A> — owning storage (W7T8)
// ---------------------------------------------------------------------------

/// Owned storage with SIMD-friendly 64-byte alignment.
#[derive(Debug)]
pub struct Owned<A> {
    pub(crate) data: AlignedBuf<A>,
}

impl<A> Owned<A> {
    /// Default alignment: 64 bytes (AVX-512 cache line).
    pub const DEFAULT_ALIGNMENT: usize = 64;

    /// Creates an empty owned storage.
    pub(crate) fn new() -> Self {
        Self {
            data: AlignedBuf::empty(),
        }
    }

    /// Creates owned storage with the given capacity, 64-byte aligned.
    pub(crate) fn with_capacity(cap: usize) -> Result<Self, XenonError>
    where
        A: Element,
    {
        let align = align_of::<A>().max(Self::DEFAULT_ALIGNMENT);
        Ok(Self {
            data: AlignedBuf::with_capacity_aligned(cap, align)?,
        })
    }

    /// Constructs owned storage from a `Vec`, copying into an aligned buffer.
    ///
    /// # Errors
    ///
    /// Propagates from [`Self::from_vec_aligned`]:
    /// - `XenonError::InvalidShape { kind: ProductOverflow }` — `data.len() *
    ///   size_of::<A>()` (with alignment slack) exceeds `isize::MAX`.
    /// - `XenonError::AllocationFailed` — the underlying allocator could not
    ///   provide the requested 64-byte aligned buffer.
    pub fn from_vec(data: Vec<A>) -> Result<Self, XenonError>
    where
        A: Copy,
    {
        Self::from_vec_aligned(data)
    }

    /// Core implementation of `from_vec`: copies into a 64-byte aligned buffer.
    ///
    /// # Errors
    ///
    /// - `XenonError::InvalidShape { kind: ProductOverflow }` — `data.len() *
    ///   size_of::<A>()` (with alignment slack) exceeds `isize::MAX`.
    /// - `XenonError::AllocationFailed` (via `AlignedAlloc::alloc`) — the
    ///   underlying allocator could not provide the requested 64-byte aligned
    ///   buffer.
    pub fn from_vec_aligned(data: Vec<A>) -> Result<Self, XenonError>
    where
        A: Copy,
    {
        let len = data.len();
        if size_of::<A>() == 0 {
            return Ok(Self {
                data: AlignedBuf::zst(len),
            });
        }
        if len == 0 {
            return Ok(Self {
                data: AlignedBuf::empty(),
            });
        }
        let align = align_of::<A>().max(Self::DEFAULT_ALIGNMENT);
        let size = allocation_size::<A>(len, align, "Owned::from_vec_aligned")?;
        let ptr = AlignedAlloc::alloc(size, align)?;
        let typed_ptr = ptr.as_ptr() as *mut A;
        // SAFETY: typed_ptr and data.as_ptr() are valid for len elements,
        // non-overlapping (typed_ptr is freshly allocated)
        unsafe {
            core::ptr::copy_nonoverlapping(data.as_ptr(), typed_ptr, len);
        }
        drop(data);
        // SAFETY: ptr was allocated by AlignedAlloc; len elements initialized
        Ok(Self {
            data: unsafe { AlignedBuf::from_raw_parts(typed_ptr, len, len, align) },
        })
    }

    /// Creates owned storage filled with zeros.
    ///
    /// # Errors
    ///
    /// - `XenonError::InvalidShape { kind: ProductOverflow }` — `len *
    ///   size_of::<A>()` (with alignment slack) exceeds `isize::MAX`.
    /// - `XenonError::AllocationFailed` (via `AlignedAlloc::alloc_zeroed`) —
    ///   the underlying allocator could not provide the requested zero-filled
    ///   aligned buffer.
    pub fn zeros(len: usize) -> Result<Self, XenonError>
    where
        A: Element + Default,
    {
        if size_of::<A>() == 0 {
            return Ok(Self {
                data: AlignedBuf::zst(len),
            });
        }
        if len == 0 {
            return Ok(Self {
                data: AlignedBuf::empty(),
            });
        }
        let align = align_of::<A>().max(Self::DEFAULT_ALIGNMENT);
        let size = allocation_size::<A>(len, align, "Owned::zeros")?;
        let ptr = AlignedAlloc::alloc_zeroed(size, align)?;
        // SAFETY: alloc_zeroed returned valid zeroed memory for len elements
        Ok(Self {
            data: unsafe { AlignedBuf::from_raw_parts(ptr.as_ptr() as *mut A, len, len, align) },
        })
    }

    /// Creates owned storage filled with clones of `value`.
    ///
    /// # Errors
    ///
    /// Propagates from `Self::with_capacity`:
    /// - `XenonError::InvalidShape { kind: ProductOverflow }` — `len *
    ///   size_of::<A>()` (with alignment slack) overflows `isize::MAX`.
    /// - `XenonError::AllocationFailed` — the underlying `AlignedAlloc` call
    ///   could not provide the requested aligned buffer.
    pub fn from_elem(len: usize, value: A) -> Result<Self, XenonError>
    where
        A: Element + Clone,
    {
        let mut owned = Self::with_capacity(len)?;
        for index in 0..len {
            // SAFETY: capacity >= len, ptr is valid for len elements
            unsafe {
                core::ptr::write(owned.data.as_mut_ptr().add(index), value);
            }
        }
        // SAFETY: all len elements have been initialized
        unsafe {
            owned.data.set_len(len);
        }
        Ok(owned)
    }
}

// ---------------------------------------------------------------------------
// W7T9: RawStorage impl for Owned<A>
// ---------------------------------------------------------------------------

impl<A> crate::private::Sealed for Owned<A> {}

/// # Safety
///
/// `Owned<A>` implements `RawStorage` because `AlignedBuf<A>` maintains the
/// storage invariants required by `05-storage.md` §5.3 and §5.5:
///
/// 1. `data.ptr` is always non-null: real allocation for non-empty non-ZST
///    buffers, or `NonNull::dangling()` for empty/ZST buffers.
/// 2. `data.ptr` satisfies `data.align`, and `data.align >= align_of::<A>()`.
/// 3. The `len` initialized elements are within one allocation owned by
///    `AlignedBuf`, never spanning multiple allocations.
/// 4. `AlignedBuf` constructors reject `len * size_of::<A>()` overflow, so the
///    backing range satisfies the `isize::MAX` bound used by slice creation.
/// 5. `as_ptr()` forwards the stable base pointer stored in `AlignedBuf`; tensor
///    offsets are handled by `TensorBase`, not by storage.
unsafe impl<A> RawStorage for Owned<A> {
    type Elem = A;

    fn as_ptr(&self) -> *const A {
        self.data.as_ptr()
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

// ---------------------------------------------------------------------------
// W7T10: Storage impl for Owned<A>
// ---------------------------------------------------------------------------

/// # Safety
///
/// `Owned<A>` implements `Storage` because W7T9's `RawStorage` impl exposes
/// the same base pointer and length maintained by `AlignedBuf<A>`, and
/// `AlignedBuf` guarantees the `Storage::as_slice` preconditions from
/// `05-storage.md` §5.5: non-null aligned pointer, one allocation, `len`
/// initialized elements, `isize::MAX` range limit, and no mutable alias
/// for the duration of `&self`.
unsafe impl<A: Element> Storage for Owned<A> {}

// ---------------------------------------------------------------------------
// W7T11: StorageMut + StorageOwned + Clone + IsOwned for Owned<A>
// ---------------------------------------------------------------------------

/// # Safety
///
/// `Owned<A>` implements `RawStorageMut` because it uniquely owns the
/// `AlignedBuf<A>` allocation. `&mut self` gives exclusive access to the
/// `len` initialized elements, and `data.as_mut_ptr()` is the same stable,
/// non-null, aligned base pointer established by W7T9.
unsafe impl<A> RawStorageMut for Owned<A> {
    fn as_mut_ptr(&mut self) -> *mut A {
        self.data.as_mut_ptr()
    }
}

/// # Safety
///
/// `Owned<A>` implements `StorageMut` because `Owned` has unique ownership of
/// its `AlignedBuf<A>` allocation. During any `&mut self` borrow there can be
/// no shared or mutable aliases into the same range, satisfying
/// `05-storage.md` §5.6 for `get_mut`, `get_unchecked_mut` and `as_mut_slice`
/// default methods.
unsafe impl<A: Element> StorageMut for Owned<A> {}

/// # Safety
///
/// `Owned<A>` satisfies `IsOwned`'s `RawStorage + Sealed` bounds because
/// W7T9 provides `RawStorage` and `Sealed`.
unsafe impl<A: Element> IsOwned for Owned<A> {}

// Owned<A>: Clone uses deep_clone semantics (W7T11).
impl<A: Element + Clone> Clone for Owned<A> {
    fn clone(&self) -> Self {
        self.deep_clone()
    }
}

impl<A: Element> Owned<A> {
    /// Moves elements from a `Vec` into a fresh aligned allocation without
    /// requiring `Copy`.
    fn from_vec_moved(data: Vec<A>) -> Result<Self, XenonError> {
        let len = data.len();
        let mut owned = Self::with_capacity(len)?;
        for (index, value) in data.into_iter().enumerate() {
            // SAFETY: capacity >= len, ptr valid for len elements
            unsafe {
                core::ptr::write(owned.data.as_mut_ptr().add(index), value);
            }
        }
        // SAFETY: all len elements initialized
        unsafe {
            owned.data.set_len(len);
        }
        Ok(owned)
    }
}

/// # Safety
///
/// `Owned<A>` implements `StorageOwned` because it has exclusive ownership of
/// the `AlignedBuf<A>` allocation, supports mutable access through
/// `StorageMut`, and all owned constructors preserve the `AlignedBuf`
/// invariants from `05-storage.md` §6.1: non-null aligned pointer, one
/// allocation, initialized logical elements, capacity metadata, and no
/// ZST/empty deallocation.
unsafe impl<A: Element + Clone> StorageOwned for Owned<A> {
    fn zeros(len: usize) -> Self
    where
        Self::Elem: Default,
    {
        <Owned<A>>::zeros(len).expect("allocation failed during StorageOwned::zeros")
    }

    fn from_elem(len: usize, value: Self::Elem) -> Self
    where
        Self::Elem: Clone,
    {
        <Owned<A>>::from_elem(len, value).expect("allocation failed during StorageOwned::from_elem")
    }

    fn from_vec(vec: Vec<Self::Elem>) -> Result<Self, XenonError>
    where
        Self::Elem: Copy,
    {
        <Owned<A>>::from_vec(vec)
    }

    fn from_iter<I: IntoIterator<Item = Self::Elem>>(iter: I) -> Self {
        let data: Vec<Self::Elem> = iter.into_iter().collect();
        <Owned<A>>::from_vec_moved(data).expect("allocation failed during StorageOwned::from_iter")
    }

    fn into_vec(self) -> Vec<Self::Elem> {
        let mut this = core::mem::ManuallyDrop::new(self);
        let mut out = Vec::with_capacity(this.data.len());
        for index in 0..this.data.len() {
            // SAFETY: index < len, element is initialized
            unsafe {
                out.push(core::ptr::read(this.data.as_ptr().add(index)));
            }
        }
        // All elements have been moved out. Prevent AlignedBuf::Drop
        // from double-dropping elements, then manually release the
        // underlying aligned allocation.
        unsafe { this.data.set_len(0) };
        // SAFETY: all elements have been moved out. Only the allocation
        // metadata remains, which is safe to drop (frees aligned memory).
        unsafe {
            core::ptr::drop_in_place(&mut this.data as *mut AlignedBuf<A>);
        }
        out
    }

    fn deep_clone(&self) -> Self {
        let mut cloned = <Owned<A>>::with_capacity(self.len())
            .expect("allocation failed during StorageOwned::deep_clone");
        for index in 0..self.len() {
            // SAFETY: index < len for both src and dst, no overlap
            unsafe {
                core::ptr::write(
                    cloned.data.as_mut_ptr().add(index),
                    *self.as_ptr().add(index),
                );
            }
        }
        // SAFETY: all len elements initialized
        unsafe {
            cloned.data.set_len(self.len());
        }
        cloned
    }

    fn capacity(&self) -> usize {
        self.data.capacity()
    }

    fn try_reserve(&mut self, new_capacity: usize) -> Result<(), XenonError> {
        if new_capacity <= self.capacity() {
            return Ok(());
        }
        let mut grown = <Owned<A>>::with_capacity(new_capacity)?;
        for index in 0..self.len() {
            // SAFETY: src and dst are non-overlapping, index < len
            unsafe {
                core::ptr::write(
                    grown.data.as_mut_ptr().add(index),
                    core::ptr::read(self.data.as_ptr().add(index)),
                );
            }
        }
        // SAFETY: all elements moved to grown
        unsafe {
            grown.data.set_len(self.len());
        }
        unsafe { self.data.set_len(0) };
        *self = grown;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// W7T12: into_shared + Send/Sync for Owned<A>
// ---------------------------------------------------------------------------

impl<A: Element + Clone> StorageIntoOwned for Owned<A> {
    fn into_owned_storage(self) -> Owned<A>
    where
        Self::Elem: Clone,
    {
        self
    }
}

// ---------------------------------------------------------------------------

impl<A> Owned<A> {
    /// Zero-copy conversion from `Owned<A>` to shared read-only `ArcRepr<A>`.
    pub fn into_shared(self) -> ArcRepr<A> {
        ArcRepr::from_aligned_buf(self.data)
    }

    /// Returns the allocator alignment in bytes.
    pub(crate) fn alignment(&self) -> usize {
        self.data.alignment()
    }

    /// Returns a mutable pointer to the storage base without requiring
    /// `&mut self`. This is safe because `Owned<A>` has exclusive ownership.
    pub(crate) fn as_mut_ptr_unchecked(&self) -> *mut A {
        self.data.ptr.as_ptr()
    }

    /// Constructs `Owned<A>` from raw allocator components.
    ///
    /// # Safety
    ///
    /// Same preconditions as [`AlignedBuf::from_raw_parts`].
    pub(crate) unsafe fn from_raw_parts(ptr: *mut A, len: usize, cap: usize, align: usize) -> Self {
        Self {
            data: unsafe { AlignedBuf::from_raw_parts(ptr, len, cap, align) },
        }
    }
}

// SAFETY: `Owned<A>` has exclusive ownership of its allocation and moving it to
// another thread moves the only owner. Element values are only moved across
// threads when `A: Send`, so no non-Send element can cross a thread boundary.
unsafe impl<A: Send> Send for Owned<A> {}

// SAFETY: Shared access to `Owned<A>` only exposes shared access to initialized
// `A` elements and immutable metadata. Sharing those elements across threads is
// sound exactly when `A: Sync`.
unsafe impl<A: Sync> Sync for Owned<A> {}

// ---------------------------------------------------------------------------
// W7T13: TryFrom<Vec<A>> + Default for Owned<A>
// ---------------------------------------------------------------------------

impl<A> Default for Owned<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Element + Copy> TryFrom<Vec<A>> for Owned<A> {
    type Error = crate::error::XenonError;

    fn try_from(value: Vec<A>) -> Result<Self, Self::Error> {
        Self::from_vec(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_owned_new_empty() {
        let owned = Owned::<f64>::new();
        assert_eq!(owned.data.len(), 0);
        assert_eq!(owned.data.capacity(), 0);
    }

    #[test]
    fn test_owned_zeros() {
        let owned =
            Owned::<f64>::zeros(100).expect("Owned::zeros should succeed for small f64 input");
        assert_eq!(owned.data.len(), 100);
        for index in 0..owned.data.len() {
            assert_eq!(
                // SAFETY: index < len, all elements initialized to zero
                unsafe { *owned.data.as_ptr().add(index) },
                0.0
            );
        }
    }

    #[test]
    fn test_owned_from_vec() {
        let owned = Owned::from_vec(vec![1_i32, 2, 3])
            .expect("Owned::from_vec should succeed for small i32 input");
        assert_eq!(owned.data.len(), 3);
        assert_eq!(
            // SAFETY: index 0 < len
            unsafe { *owned.data.as_ptr().add(0) },
            1
        );
        assert_eq!(
            // SAFETY: index 2 < len
            unsafe { *owned.data.as_ptr().add(2) },
            3
        );
    }

    #[test]
    fn test_owned_zeros_layout_overflow_returns_error() {
        let err = match Owned::<bool>::zeros(isize::MAX as usize) {
            Ok(_) => panic!("layout overflow should return error"),
            Err(err) => err,
        };
        match err {
            XenonError::InvalidShape {
                kind: InvalidShapeKind::ProductOverflow,
                ..
            } => {},
            other => panic!("expected InvalidShape::ProductOverflow, got {other:?}"),
        }
    }

    #[test]
    fn test_owned_alignment_from_zeros() {
        let owned =
            Owned::<f64>::zeros(8).expect("Owned::zeros should succeed for small f64 input");
        assert_eq!(
            (owned.data.as_ptr() as usize) % Owned::<f64>::DEFAULT_ALIGNMENT,
            0
        );
    }

    #[test]
    fn test_owned_alignment_from_vec() {
        let owned = Owned::from_vec(vec![1.0_f64, 2.0])
            .expect("Owned::from_vec should succeed for small f64 input");
        assert_eq!(
            (owned.data.as_ptr() as usize) % Owned::<f64>::DEFAULT_ALIGNMENT,
            0
        );
    }

    // W7T9 tests
    #[test]
    fn test_owned_raw_storage_len() {
        let owned = Owned::from_vec(vec![1_i32, 2, 3])
            .expect("Owned::from_vec should succeed for small i32 input");
        assert_eq!(owned.len(), 3);
        assert_eq!(
            // SAFETY: index 2 < len
            unsafe { *owned.as_ptr().add(2) },
            3
        );
    }

    #[test]
    fn test_owned_raw_storage_ptr() {
        let owned = Owned::from_vec(vec![1_i32, 2, 3])
            .expect("Owned::from_vec should succeed for small i32 input");
        let first = owned.as_ptr();
        let second = owned.as_ptr();
        assert_eq!(first, second);
        assert!(!owned.is_empty());
        assert!(owned.is_aligned());
        assert!(owned.is_aligned_to(64));
    }

    // W7T10 tests
    #[test]
    fn test_owned_storage_as_slice() {
        let owned = Owned::from_vec(vec![1_i32, 2, 3])
            .expect("Owned::from_vec should succeed for small i32 input");
        assert_eq!(owned.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_owned_storage_get() {
        let owned = Owned::from_vec(vec![1.0_f64])
            .expect("Owned::from_vec should succeed for single f64 input");
        assert_eq!(owned.get(0), Some(&1.0));
        assert_eq!(owned.get(1), None);
    }

    #[test]
    fn test_owned_storage_get_unchecked() {
        let owned = Owned::from_vec(vec![1_i32, 2, 3])
            .expect("Owned::from_vec should succeed for small i32 input");
        assert_eq!(unsafe { *owned.get_unchecked(2) }, 3);
    }

    // W7T11 tests
    #[test]
    fn test_owned_storage_mut() {
        let mut owned = Owned::from_vec(vec![1_i32, 2])
            .expect("Owned::from_vec should succeed for small i32 input");
        *owned
            .get_mut(0)
            .expect("index 0 should be in bounds for length-2 owned storage") = 9;
        assert_eq!(owned.as_slice(), &[9, 2]);
    }

    #[test]
    fn test_owned_storage_mut_unchecked_and_slice() {
        let mut owned = Owned::from_vec(vec![1_i32, 2, 3])
            .expect("Owned::from_vec should succeed for small i32 input");
        unsafe {
            *owned.get_unchecked_mut(1) = 7;
        }
        owned.as_mut_slice()[2] = 8;
        assert_eq!(owned.as_slice(), &[1, 7, 8]);
    }

    #[test]
    fn test_owned_clone_deep() {
        let original = Owned::from_vec(vec![1_i32, 2, 3])
            .expect("Owned::from_vec should succeed for small i32 input");
        let mut cloned = original.deep_clone();
        *cloned
            .get_mut(0)
            .expect("index 0 should be in bounds for cloned length-3 owned storage") = 9;
        assert_eq!(original.as_slice(), &[1, 2, 3]);
        assert_eq!(cloned.as_slice(), &[9, 2, 3]);
    }

    #[test]
    fn test_owned_into_vec() {
        let owned = Owned::from_vec(vec![1_i32, 2, 3])
            .expect("Owned::from_vec should succeed for small i32 input");
        assert_eq!(owned.into_vec(), vec![1, 2, 3]);
    }

    #[test]
    fn test_storage_owned_capacity() {
        let mut owned = <Owned<f64> as StorageOwned>::zeros(4);
        assert!(owned.capacity() >= 4);
        owned
            .try_reserve(16)
            .expect("try_reserve should succeed for small growth request");
        assert!(owned.capacity() >= 16);
        assert_eq!(owned.len(), 4);
    }

    #[test]
    fn test_storage_owned_from_elem_and_from_iter() {
        let owned = <Owned<i32> as StorageOwned>::from_elem(3, 5);
        assert_eq!(owned.as_slice(), &[5, 5, 5]);
        let iter_owned = <Owned<i32> as StorageOwned>::from_iter([1, 2, 3]);
        assert_eq!(iter_owned.as_slice(), &[1, 2, 3]);
    }

    // W7T12 tests
    #[test]
    fn test_owned_into_shared() {
        let owned = Owned::from_vec(vec![1_i32, 2])
            .expect("Owned::from_vec should succeed for small i32 input");
        let shared = owned.into_shared();
        assert_eq!(shared.as_slice(), &[1, 2]);
    }

    #[test]
    fn test_owned_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<Owned<i32>>();
        assert_sync::<Owned<i32>>();
    }

    #[test]
    fn test_owned_cross_thread() {
        let owned = Owned::from_vec(vec![1_i32, 2, 3])
            .expect("Owned::from_vec should succeed for small i32 input");
        let handle = std::thread::spawn(move || owned.len());
        assert_eq!(handle.join().expect("thread should not panic"), 3);
    }

    // W7T13 tests
    #[test]
    fn test_owned_default() {
        let owned = Owned::<i32>::default();
        assert!(owned.is_empty());
    }

    #[test]
    fn test_owned_from_vec_try_from() {
        let v = vec![1i32, 2, 3];
        let owned = Owned::try_from(v).expect("from_vec succeeds");
        assert_eq!(owned.as_slice(), &[1, 2, 3]);
        assert_eq!(owned.as_ptr().align_offset(64), 0);
    }
}
