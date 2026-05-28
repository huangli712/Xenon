//! Storage traits for Xenon backing buffers — raw pointer access, safe read,
//! mutable access, ownership, shared access, and marker traits.
//!
//! This module defines the complete storage trait hierarchy:
//!
//! - Marker traits: `IsOwned`, `IsView`, `IsViewMut`, `IsShared`
//! - Core trait hierarchy:
//!   `RawStorage` → `Storage` / `RawStorageMut` → `StorageMut` →
//!   `StorageOwned` / `StorageShared`
//! - Conversion trait: `StorageIntoOwned`

use core::ptr::NonNull;

use crate::private::Sealed;

// ---------------------------------------------------------------------------
// Marker traits (IsOwned, IsView, IsViewMut, IsShared)
// ---------------------------------------------------------------------------

/// Marker trait for owned storage representations.
///
/// # Safety
///
/// Implementors must satisfy the [`RawStorage`] contract and represent a
/// storage mode with exclusive ownership semantics controlled by this crate.
/// This trait is sealed and must not be implemented outside Xenon.
///
/// # Sealed
///
/// This trait is sealed and cannot be implemented outside of `Xenon`.
pub unsafe trait IsOwned: RawStorage + Sealed {}

/// Marker trait for immutable borrowed view storage representations.
///
/// # Safety
///
/// Implementors must satisfy the [`RawStorage`] contract and represent a
/// read-only borrowed storage mode controlled by this crate. This trait is
/// sealed and must not be implemented outside Xenon.
///
/// # Sealed
///
/// This trait is sealed and cannot be implemented outside of `Xenon`.
pub unsafe trait IsView: RawStorage + Sealed {}

/// Marker trait for mutable borrowed view storage representations.
///
/// # Safety
///
/// Implementors must satisfy the [`RawStorage`] contract and represent an
/// exclusive mutable borrowed storage mode controlled by this crate. This
/// trait is sealed and must not be implemented outside Xenon.
///
/// # Sealed
///
/// This trait is sealed and cannot be implemented outside of `Xenon`.
pub unsafe trait IsViewMut: RawStorage + Sealed {}

/// Marker trait for shared read-only storage representations.
///
/// # Safety
///
/// Implementors must satisfy the [`RawStorage`] contract and represent a
/// shared read-only storage mode controlled by this crate. This trait is
/// sealed and must not be implemented outside Xenon.
///
/// # Sealed
///
/// This trait is sealed and cannot be implemented outside of `Xenon`.
pub unsafe trait IsShared: RawStorage + Sealed {}

// ---------------------------------------------------------------------------
// RawStorage — raw pointer access to underlying storage
// ---------------------------------------------------------------------------

/// Raw pointer access to underlying storage.
///
/// # Safety
///
/// Implementors must uphold: `as_ptr()` remains valid for the
/// storage lifetime, repeated calls return the same address, the pointer is
/// non-null and properly aligned, the `len()` range is initialized within one
/// allocation, and the total range does not exceed `isize::MAX`.
///
/// # Sealed
///
/// This trait is sealed and cannot be implemented outside of `Xenon`.
pub unsafe trait RawStorage: Sealed {
    /// The element type of the storage.
    type Elem;

    /// Returns the raw storage base pointer.
    fn as_ptr(&self) -> *const Self::Elem;

    /// Returns the number of elements in storage.
    fn len(&self) -> usize;

    /// Checks if the storage is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Checks if the pointer satisfies the specified alignment requirement.
    ///
    /// Returns `false` for `align == 0` or `align` that is not a power of two,
    /// rather than panicking.
    fn is_aligned_to(&self, align: usize) -> bool {
        align != 0 && align.is_power_of_two() && (self.as_ptr() as usize).is_multiple_of(align)
    }

    /// Checks if the storage satisfies the current default alignment (64 bytes).
    fn is_aligned(&self) -> bool {
        self.is_aligned_to(64)
    }
}

// ---------------------------------------------------------------------------
// Storage — safe read access to backing storage
// ---------------------------------------------------------------------------

/// Safe read access to the entire backing storage.
///
/// # Safety
///
/// Implementors must uphold the [`RawStorage`] contract and guarantee that
/// the storage-visible range exposed through safe shared access remains
/// fully initialized, aligned, and valid for the duration of `&self`.
///
/// # Sealed
///
/// This trait is sealed and cannot be implemented outside of `Xenon`.
pub unsafe trait Storage: RawStorage + Sealed {
    /// Returns an immutable reference to the element at the given index.
    fn get(&self, index: usize) -> Option<&Self::Elem> {
        if index < self.len() {
            // SAFETY: index is bounds-checked above
            Some(unsafe { self.get_unchecked(index) })
        } else {
            None
        }
    }

    /// Returns an immutable reference to the element at the given index
    /// without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure `index < self.len()`.
    unsafe fn get_unchecked(&self, index: usize) -> &Self::Elem {
        // SAFETY: caller guarantees `index < self.len()`, and `Storage`'s
        // unsafe contract ensures `as_ptr()` points to a valid initialized
        // range of `len()` elements.
        unsafe { &*self.as_ptr().add(index) }
    }

    /// Returns a slice view of the storage-visible backing range.
    ///
    /// This is a storage-level API. It starts at the storage base pointer and
    /// spans exactly `len()` initialized elements.
    fn as_slice(&self) -> &[Self::Elem] {
        // SAFETY: `Storage`'s unsafe supertrait contract guarantees a single,
        // aligned, fully initialized, non-aliased range of `len()` elements
        // starting at `as_ptr()`.
        unsafe { core::slice::from_raw_parts(self.as_ptr(), self.len()) }
    }
}

// ---------------------------------------------------------------------------
// RawStorageMut + StorageMut — mutable storage access
// ---------------------------------------------------------------------------

/// Raw pointer access for mutable storage.
///
/// # Safety
///
/// Implementors must ensure the pointer returned by `as_mut_ptr()` remains
/// valid for the storage's lifetime and that no other mutable references
/// to the same data exist (aliasing rules).
///
/// # Sealed
///
/// This trait is sealed and cannot be implemented outside of `Xenon`.
pub unsafe trait RawStorageMut: RawStorage + Sealed {
    /// Returns a raw mutable pointer to the start of the data.
    fn as_mut_ptr(&mut self) -> *mut Self::Elem;

    /// Converts the storage to a NonNull pointer.
    ///
    /// # Safety
    ///
    /// For empty storage, this returns `NonNull::dangling()` as a sentinel.
    /// Callers must check `self.len() > 0` before dereferencing the result.
    unsafe fn as_non_null(&mut self) -> NonNull<Self::Elem> {
        if self.len() == 0 {
            NonNull::dangling()
        } else {
            // SAFETY: caller ensures the storage is non-empty; RawStorageMut
            // contract guarantees as_mut_ptr() is valid and non-null.
            unsafe { NonNull::new_unchecked(self.as_mut_ptr()) }
        }
    }
}

/// Safe read-write access to storage.
///
/// # Safety
///
/// Implementors must uphold the contracts of both [`Storage`] and
/// [`RawStorageMut`], and guarantee exclusive mutable access to the
/// storage-visible range for the duration of `&mut self`.
///
/// # Sealed
///
/// This trait is sealed and cannot be implemented outside of `Xenon`.
pub unsafe trait StorageMut: Storage + RawStorageMut + Sealed {
    /// Returns a mutable reference to the element at the given index.
    fn get_mut(&mut self, index: usize) -> Option<&mut Self::Elem> {
        if index < self.len() {
            // SAFETY: index is bounds-checked above
            Some(unsafe { self.get_unchecked_mut(index) })
        } else {
            None
        }
    }

    /// Returns a mutable reference to the element at the given index
    /// without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure `index < self.len()`.
    unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut Self::Elem {
        // SAFETY: caller guarantees `index < self.len()`, and `StorageMut`'s
        // unsafe contract ensures exclusive mutable access.
        unsafe { &mut *self.as_mut_ptr().add(index) }
    }

    /// Returns a mutable slice view of the storage-visible backing range.
    ///
    /// Like `Storage::as_slice()`, this is a storage-level API over the
    /// storage base pointer and `len()` initialized elements.
    fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
        // SAFETY: `StorageMut`'s unsafe supertrait contract guarantees an
        // exclusive, aligned, fully initialized range of `len()` elements
        // starting at `as_mut_ptr()`.
        unsafe { core::slice::from_raw_parts_mut(self.as_mut_ptr(), self.len()) }
    }

    /// Fills the entire storage-visible backing range with the given value.
    ///
    /// This is a storage-layer API with `fill_all()` semantics.
    fn fill(&mut self, value: Self::Elem)
    where
        Self::Elem: Copy,
    {
        self.as_mut_slice().fill(value);
    }
}

// ---------------------------------------------------------------------------
// StorageOwned + StorageShared + StorageIntoOwned
// ---------------------------------------------------------------------------

/// Storage that owns data.
///
/// # Safety
///
/// Implementors must uphold the [`StorageMut`] contract, own their backing
/// allocation exclusively, and ensure all constructors and conversions
/// preserve the storage invariants required by this module.
///
/// # Sealed
///
/// This trait is sealed and cannot be implemented outside of `Xenon`.
pub unsafe trait StorageOwned: StorageMut + Clone + Sealed {
    /// Allocates storage of the given size, zero-filled.
    fn zeros(len: usize) -> Self
    where
        Self::Elem: Default;

    /// Allocates storage of the given size, filled with the given value.
    fn from_elem(len: usize, value: Self::Elem) -> Self
    where
        Self::Elem: Clone;

    /// Constructs storage from a Vec.
    ///
    /// # Errors
    ///
    /// Returns `crate::error::XenonError` when the implementor cannot
    /// allocate the backing buffer — typically:
    /// - `XenonError::InvalidShape { kind: ProductOverflow }` if
    ///   `vec.len() * size_of::<Self::Elem>()` (with alignment slack)
    ///   exceeds `isize::MAX`.
    /// - `XenonError::AllocationFailed` if the underlying allocator
    ///   cannot satisfy the request.
    fn from_vec(vec: Vec<Self::Elem>) -> Result<Self, crate::error::XenonError>
    where
        Self::Elem: Copy;

    /// Constructs storage from an iterator.
    fn from_iter<I: IntoIterator<Item = Self::Elem>>(iter: I) -> Self;

    /// Converts storage into a Vec.
    fn into_vec(self) -> Vec<Self::Elem>;

    /// Creates a deep copy of the storage.
    fn deep_clone(&self) -> Self;

    /// Returns the capacity of the storage.
    fn capacity(&self) -> usize;

    /// Attempts to ensure total capacity is at least `new_capacity`.
    ///
    /// `new_capacity` is the target total capacity, not an additional amount.
    ///
    /// # Errors
    ///
    /// Returns `crate::error::XenonError` when the implementor cannot grow
    /// the backing buffer to `new_capacity` — typically:
    /// - `XenonError::InvalidShape { kind: ProductOverflow }` if
    ///   `new_capacity * size_of::<Self::Elem>()` (with alignment slack)
    ///   exceeds `isize::MAX`.
    /// - `XenonError::AllocationFailed` if the underlying allocator
    ///   cannot satisfy the request.
    fn try_reserve(&mut self, new_capacity: usize) -> Result<(), crate::error::XenonError>;
}

/// Marker trait for shared read-only storage.
///
/// # Sealed
///
/// This trait is sealed and cannot be implemented outside of `Xenon`.
///
/// # Safety
///
/// Implementors must uphold the [`Storage`] contract and represent a
/// shared read-only storage mode whose aliasing and thread-safety
/// invariants are controlled by this crate.
pub unsafe trait StorageShared: Storage + Clone + Sealed {}

/// Storage types that can be converted into an owned tensor by consuming self.
///
/// - `Owned<A>` → O(1), returns self directly
/// - `ViewRepr`/`ViewMutRepr` → O(n), copies data
/// - `ArcRepr` → O(n), always allocates and copies into a fresh owned buffer
pub trait StorageIntoOwned: Storage {
    /// Consume this storage, returning an `Owned<A>` storage.
    ///
    /// This is a storage-layer method. Tensor-level `into_owned()` handles
    /// shape, strides, and offset logic separately.
    fn into_owned_storage(self) -> crate::storage::Owned<Self::Elem>
    where
        Self::Elem: Clone;
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::{Owned, ViewRepr, ViewMutRepr, ArcRepr};

    /// Verify that all four marker traits exist and are properly sealed.
    #[test]
    fn test_marker_traits() {
        fn _sealed<T: Sealed>() {}
        fn _bound_owned<T: IsOwned>() {}
        fn _bound_view<T: IsView>() {}
        fn _bound_view_mut<T: IsViewMut>() {}
        fn _bound_shared<T: IsShared>() {}
    }

    // -----------------------------------------------------------------------
    // Basic module compile check
    // -----------------------------------------------------------------------

    /// Verifies that the storage module compiles as a basic sanity check.
    #[test]
    fn test_storage_module_compile() {
        assert_eq!(0, 0);
    }

    /// An empty mock to verify `RawStorage` trait methods compile and behave correctly.
    struct MockEmpty;

    impl Sealed for MockEmpty {}

    unsafe impl RawStorage for MockEmpty {
        type Elem = f64;

        fn as_ptr(&self) -> *const Self::Elem {
            core::ptr::NonNull::<Self::Elem>::dangling().as_ptr()
        }

        fn len(&self) -> usize {
            0
        }
    }

    /// Verifies `RawStorage` default methods (`is_empty`, `is_aligned`)
    /// on an empty mock implementor.
    #[test]
    fn test_raw_storage_compile() {
        let storage = MockEmpty;
        assert!(storage.is_empty());
        // dangling f64 pointer (address = align_of::<f64>() = 8) is not
        // 64-byte aligned; is_aligned() correctly reports false
        assert!(!storage.is_aligned());
    }

    /// A mock implementing `RawStorage` + `Storage` backed by a fixed-size array.
    struct MockStorage {
        data: [i32; 3],
    }

    impl Sealed for MockStorage {}

    unsafe impl RawStorage for MockStorage {
        type Elem = i32;

        fn as_ptr(&self) -> *const Self::Elem {
            self.data.as_ptr()
        }

        fn len(&self) -> usize {
            self.data.len()
        }
    }

    unsafe impl Storage for MockStorage {}

    /// Verifies `Storage::get` bounds checking and `as_slice`
    /// on a mock implementor.
    #[test]
    fn test_storage_compile() {
        let storage = MockStorage { data: [1, 2, 3] };
        assert_eq!(storage.get(1), Some(&2));
        assert_eq!(storage.get(3), None);
        assert_eq!(storage.as_slice(), &[1, 2, 3]);
    }

    /// A mock implementing `RawStorageMut` + `StorageMut` backed by a fixed-size array.
    struct MockStorageMut {
        data: [i32; 3],
    }

    impl Sealed for MockStorageMut {}

    unsafe impl RawStorage for MockStorageMut {
        type Elem = i32;

        fn as_ptr(&self) -> *const Self::Elem {
            self.data.as_ptr()
        }

        fn len(&self) -> usize {
            self.data.len()
        }
    }

    unsafe impl RawStorageMut for MockStorageMut {
        fn as_mut_ptr(&mut self) -> *mut Self::Elem {
            self.data.as_mut_ptr()
        }
    }

    unsafe impl Storage for MockStorageMut {}
    unsafe impl StorageMut for MockStorageMut {}

    /// Verifies `StorageMut::fill` and `RawStorageMut::as_non_null`
    /// on a mock mutable implementor.
    #[test]
    fn test_storage_mut_compile() {
        let mut storage = MockStorageMut { data: [1, 2, 3] };
        storage.fill(7);
        assert_eq!(storage.as_slice(), &[7, 7, 7]);
        let ptr = unsafe { storage.as_non_null().as_ptr() };
        assert_eq!(ptr, storage.as_mut_ptr());
    }

    /// A mock implementing `StorageOwned` backed by a Vec.
    #[derive(Clone)]
    struct MockOwned {
        data: Vec<i32>,
    }

    /// A mock implementing `StorageShared` backed by a Vec.
    #[derive(Clone)]
    struct MockShared {
        data: Vec<i32>,
    }

    impl Sealed for MockOwned {}
    impl Sealed for MockShared {}

    unsafe impl RawStorage for MockOwned {
        type Elem = i32;

        fn as_ptr(&self) -> *const Self::Elem {
            self.data.as_ptr()
        }

        fn len(&self) -> usize {
            self.data.len()
        }
    }

    unsafe impl RawStorageMut for MockOwned {
        fn as_mut_ptr(&mut self) -> *mut Self::Elem {
            self.data.as_mut_ptr()
        }
    }

    unsafe impl Storage for MockOwned {}
    unsafe impl StorageMut for MockOwned {}

    unsafe impl StorageOwned for MockOwned {
        fn zeros(len: usize) -> Self
        where
            Self::Elem: Default,
        {
            Self {
                data: vec![Self::Elem::default(); len],
            }
        }

        fn from_elem(len: usize, value: Self::Elem) -> Self
        where
            Self::Elem: Clone,
        {
            Self {
                data: vec![value; len],
            }
        }

        fn from_vec(vec: Vec<Self::Elem>) -> Result<Self, crate::error::XenonError>
        where
            Self::Elem: Copy,
        {
            Ok(Self { data: vec })
        }

        fn from_iter<I: IntoIterator<Item = Self::Elem>>(iter: I) -> Self {
            Self {
                data: iter.into_iter().collect(),
            }
        }

        fn into_vec(self) -> Vec<Self::Elem> {
            self.data
        }

        fn deep_clone(&self) -> Self {
            self.clone()
        }

        fn capacity(&self) -> usize {
            self.data.capacity()
        }

        fn try_reserve(&mut self, new_capacity: usize) -> Result<(), crate::error::XenonError> {
            if new_capacity > self.capacity() {
                self.data.reserve(new_capacity - self.capacity());
            }
            Ok(())
        }
    }

    unsafe impl RawStorage for MockShared {
        type Elem = i32;

        fn as_ptr(&self) -> *const Self::Elem {
            self.data.as_ptr()
        }

        fn len(&self) -> usize {
            self.data.len()
        }
    }

    unsafe impl Storage for MockShared {}
    unsafe impl StorageShared for MockShared {}

    /// Compile-time helper asserting a type satisfies `StorageOwned`.
    fn assert_storage_owned<S: StorageOwned>() {}
    /// Compile-time helper asserting a type satisfies `StorageShared`.
    fn assert_storage_shared<S: StorageShared>() {}

    /// Verifies that mock implementors satisfy the `StorageOwned`
    /// and `StorageShared` trait bounds.
    #[test]
    fn test_storage_traits_compile() {
        assert_storage_owned::<MockOwned>();
        assert_storage_shared::<MockShared>();
    }

    // -----------------------------------------------------------------------
    // Concrete storage type export tests
    // -----------------------------------------------------------------------

    /// Verifies that the concrete `Owned`, `ViewRepr`, and `ArcRepr`
    /// storage types are exported and usable.
    #[test]
    fn test_storage_exports_compile() {
        let owned = Owned::from_vec(vec![1_i32])
            .expect("Owned::from_vec should succeed for single i32 input");
        let view = ViewRepr::from_slice(owned.as_slice());
        let shared = ArcRepr::try_from(vec![1_i32])
            .expect("ArcRepr::try_from should succeed for single i32 input");

        assert_eq!(view.get(0), Some(&1));
        assert_eq!(shared.get(0), Some(&1));
    }

    /// Verifies that concrete storage types satisfy the `Storage`,
    /// `StorageOwned`, `StorageShared`, and `StorageIntoOwned` trait bounds.
    #[test]
    fn test_storage_trait_exports_compile() {
        fn assert_storage<S: Storage>(_: &S) {}
        fn assert_owned<S: StorageOwned>(_: &S) {}
        fn assert_shared<S: StorageShared>(_: &S) {}
        fn assert_into_owned<S: StorageIntoOwned>(_: &S) {}

        let owned = Owned::from_vec(vec![1_i32])
            .expect("Owned::from_vec should succeed for single i32 input");
        let shared = ArcRepr::try_from(vec![1_i32])
            .expect("ArcRepr::try_from should succeed for single i32 input");

        assert_storage(&owned);
        assert_owned(&owned);
        assert_into_owned(&owned);
        assert_storage(&shared);
        assert_shared(&shared);
        assert_into_owned(&shared);
    }

    // -----------------------------------------------------------------------
    // Integration tests
    // -----------------------------------------------------------------------

    /// Integration test verifying that `Owned`, `ViewRepr`, and `ArcRepr`
    /// round-trip data through `as_slice`.
    #[test]
    fn test_storage_module_compiles() {
        let owned = Owned::from_vec(vec![1_i32, 2, 3])
            .expect("Owned::from_vec should succeed for small i32 input");
        let view = ViewRepr::from_slice(owned.as_slice());
        let shared = ArcRepr::try_from(vec![1_i32, 2, 3])
            .expect("ArcRepr::try_from should succeed for small i32 input");

        assert_eq!(view.as_slice(), &[1, 2, 3]);
        assert_eq!(shared.as_slice(), &[1, 2, 3]);
    }

    /// Verifies that the four marker traits (`IsOwned`, `IsView`,
    /// `IsViewMut`, `IsShared`) are implemented by the concrete storage types.
    #[test]
    fn test_marker_traits_sealed() {
        fn assert_owned<T: IsOwned>() {}
        fn assert_view<T: IsView>() {}
        fn assert_view_mut<T: IsViewMut>() {}
        fn assert_shared<T: IsShared>() {}

        assert_owned::<Owned<i32>>();
        assert_view::<ViewRepr<'_, i32>>();
        assert_view_mut::<ViewMutRepr<'_, i32>>();
        assert_shared::<ArcRepr<i32>>();
    }

    /// Verifies that `StorageIntoOwned::into_owned_storage` works
    /// correctly across `Owned`, `ViewRepr`, and `ArcRepr`.
    #[test]
    fn test_storage_into_owned_matrix() {
        let owned = Owned::from_vec(vec![1_i32, 2, 3])
            .expect("Owned::from_vec should succeed for small i32 input");
        let shared = owned.clone().into_shared();
        let owned_view = ViewRepr::from_slice(owned.as_slice());
        let copied_from_view = owned_view.into_owned_storage();
        let shared_view = ViewRepr::from_slice(shared.as_slice());
        let copied_from_shared = shared.clone().into_owned_storage();

        assert_eq!(owned.as_slice(), &[1, 2, 3]);
        assert_eq!(shared.as_slice(), &[1, 2, 3]);
        assert_eq!(owned_view.as_slice(), &[1, 2, 3]);
        assert_eq!(copied_from_view.as_slice(), &[1, 2, 3]);
        assert_eq!(shared_view.as_slice(), &[1, 2, 3]);
        assert_eq!(copied_from_shared.as_slice(), &[1, 2, 3]);
    }

    /// Verifies that `ArcRepr::into_owned_storage` produces a detached
    /// deep copy that does not alias the shared buffer.
    #[test]
    fn test_arc_into_owned_storage_is_detached_copy() {
        let shared = ArcRepr::try_from(vec![1_i32, 2, 3])
            .expect("ArcRepr::try_from should succeed for small i32 input");
        let mut copied = shared.clone().into_owned_storage();

        copied.as_mut_slice()[0] = 9;

        assert_eq!(copied.as_slice(), &[9, 2, 3]);
        assert_eq!(shared.as_slice(), &[1, 2, 3]);
    }

    /// Verifies `Send`/`Sync` bounds for all concrete storage types.
    #[test]
    fn test_storage_thread_safety_matrix_compile() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<Owned<i32>>();
        assert_sync::<Owned<i32>>();
        assert_send::<ViewRepr<'_, i32>>();
        assert_sync::<ViewRepr<'_, i32>>();
        assert_send::<ViewMutRepr<'_, i32>>();
        assert_send::<ArcRepr<i32>>();
        assert_sync::<ArcRepr<i32>>();
    }
}
