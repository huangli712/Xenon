//! FFI pointer-export API and raw-parts re-exports.
//!
//! See `docs/design/23-ffi.md` §5.4 (`export` / `export_mut`) and §5.7-§5.8
//! (raw-parts ownership semantics) for the authoritative contract.

use core::marker::PhantomData;
use core::ptr::NonNull;

use super::private::{TensorExport, TensorExportMut};
use super::types::{TensorExportMutRaw, TensorExportRaw};
use crate::dimension::Dimension;
use crate::element::{element_type_of, Element};
use crate::storage::{Storage, StorageMut};

/// Re-exports for FFI consumers to access raw-parts metadata and the
/// tensor type via the stable `crate::ffi::*` path.
///
/// `OwnedRawParts` is defined in `crate::tensor` (see `07-tensor.md`
/// §5.7). The inherent methods `from_raw_parts` / `from_raw_parts_mut`
/// / `into_raw_parts` / `from_raw_parts_owned` are defined on
/// `TensorBase` and ride on the type itself — they become callable as
/// soon as `TensorBase` is in scope, with no extra `pub use` required
/// (inherent methods have no free-function path symbol).
///
/// Note: a single `pub use` brings `TensorBase` into the current scope
/// (so that `impl TensorBase { ... }` below works) **and** simultaneously
/// re-exports it from `crate::ffi::ptr`.
pub use crate::tensor::{OwnedRawParts, TensorBase};

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    A: Element,
    D: Dimension,
{
    /// Export tensor data as a C-compatible raw structure.
    ///
    /// The returned `TensorExportRaw` borrows the tensor's data and
    /// metadata. The consumer must ensure the tensor outlives the export.
    /// This method does not fail; it always returns a valid export.
    ///
    /// `data` always carries the **storage base pointer**; the logical
    /// first element address is derived from `data.add(offset)` for
    /// non-empty tensors. Empty tensors (`len() == 0`) get a valid
    /// aligned `dangling` pointer that must not be dereferenced;
    /// `shape`, `strides`, and `offset` still describe the empty
    /// tensor metadata.
    ///
    /// **Visibility & return type** (`23-ffi.md` §5.4 line 547-555):
    /// This is the public FFI entry; it returns `TensorExportRaw` (the
    /// C-visible non-generic raw descriptor). The intermediate generic
    /// descriptor `TensorExport<'_, A>` is `pub(crate)` Rust-only
    /// borrowing evidence and cannot appear in a `pub fn` return type.
    pub fn export(&self) -> TensorExportRaw {
        self.export_internal().into()
    }

    /// `pub(crate)` internal helper: produces the typed generic descriptor
    /// for in-crate borrow tracking and lifetime evidence. Not exposed to
    /// downstream consumers; use `export()` for the public FFI surface.
    pub(crate) fn export_internal(&self) -> TensorExport<'_, A> {
        let data = if self.is_empty() {
            // Empty tensor: a valid aligned non-dereferenceable pointer.
            // Do NOT call as_storage_ptr() — the backing storage may be
            // empty or even unallocated (e.g. zero-cap Vec).
            NonNull::<A>::dangling().as_ptr() as *const A
        } else {
            self.as_storage_ptr()
        };
        TensorExport {
            data,
            element_type: element_type_of::<A>(),
            ndim: self.ndim(),
            shape: self.shape().as_ptr(),
            strides: self.strides().as_ptr(),
            storage_len: self.storage_len(),
            offset: self.offset(),
            _marker: PhantomData,
        }
    }
}

impl<S, D, A> TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    A: Element,
    D: Dimension,
{
    /// Export tensor data with mutable access.
    ///
    /// **Visibility & return type**: Public FFI entry; returns
    /// `TensorExportMutRaw`. The intermediate generic
    /// `TensorExportMut<'_, A>` is `pub(crate)` Rust-only borrowing
    /// evidence and cannot appear in `pub fn` return type.
    ///
    /// This API is only implemented for writable storage, so read-only
    /// storage modes are rejected at the trait boundary rather than at
    /// runtime. No additional fallible validation is performed beyond the
    /// existing `&mut self` + `S: StorageMut` exclusivity boundary.
    pub fn export_mut(&mut self) -> TensorExportMutRaw {
        self.export_mut_internal().into()
    }

    /// `pub(crate)` internal helper: produces the typed mutable generic
    /// descriptor for in-crate borrow tracking and lifetime evidence.
    pub(crate) fn export_mut_internal(&mut self) -> TensorExportMut<'_, A> {
        let data = if self.is_empty() {
            NonNull::<A>::dangling().as_ptr()
        } else {
            self.as_storage_mut_ptr()
        };
        // Capture metadata *before* moving the mutable borrow into `data`.
        let element_type = element_type_of::<A>();
        let ndim = self.ndim();
        let shape = self.shape().as_ptr();
        let strides = self.strides().as_ptr();
        let storage_len = self.storage_len();
        let offset = self.offset();
        TensorExportMut {
            data,
            element_type,
            ndim,
            shape,
            strides,
            storage_len,
            offset,
            _marker: PhantomData,
        }
    }
}

/// Doctest enforcing that `export_mut` cannot be called on a read-only view.
///
/// ```compile_fail
/// use xenon::tensor::Tensor;
/// use xenon::dimension::Ix1;
/// let t: Tensor<i32, Ix1> = Tensor::from_shape_vec([2], vec![1, 2]).unwrap();
/// let v = t.view();
/// // `view()` returns a `ViewRepr`-backed TensorBase, which is `Storage`
/// // but NOT `StorageMut`. `export_mut` must therefore fail to compile.
/// let _ = (&v).export_mut();
/// ```
fn _doctest_export_mut_rejects_view() {}

/// Doctest enforcing that `into_raw_parts` cannot be called on a view.
///
/// ```compile_fail
/// use xenon::tensor::Tensor;
/// use xenon::dimension::Ix1;
/// let t: Tensor<i32, Ix1> = Tensor::from_shape_vec([2], vec![1, 2]).unwrap();
/// let v = t.view();
/// // `into_raw_parts` is only implemented for Owned storage.
/// let _ = v.into_raw_parts();
/// ```
fn _doctest_into_raw_parts_rejects_view() {}

#[cfg(test)]
mod tests {
    use crate::ffi::types::ElementType;
    use crate::dimension::Ix1;
    use crate::layout::Strides;
    use crate::tensor::{TensorView, TensorViewMut};

    /// §8.2 test_export_contract — exports match source metadata, and
    /// crucially `data == as_storage_ptr()` (NOT `as_ptr()`).
    /// Constructed via W8T7 `TensorView::from_raw_parts` to avoid W22.
    #[test]
    fn test_export_contract_data_is_storage_base() {
        let data = [10_i32, 20, 30];
        // SAFETY: F-order [3] view over data; offset = 0.
        let tensor = unsafe {
            TensorView::<i32, Ix1>::from_raw_parts(
                data.as_ptr(),
                data.len(),
                Ix1(3),
                Strides::from_slice(&[1_usize]).expect("valid strides"),
                0,
            )
        }
        .expect("valid F-order [3] view");
        let raw = tensor.export();
        assert_eq!(raw.ndim, 1);
        assert_eq!(raw.storage_len, 3);
        assert_eq!(raw.element_type, ElementType::I32);
        // The critical contract: §5.4 mandates `data` carries the storage
        // base pointer, not the logical first. For offset = 0 views the two
        // are equal; the offset != 0 case is tested in
        // `test_export_contract_data_storage_base_with_nonzero_offset`.
        assert_eq!(raw.data as *const i32, tensor.as_storage_ptr());
    }

    /// Critical: `data == as_storage_ptr()` AND `data != as_ptr()` when
    /// `offset != 0`. This is the gate guarding against the historical
    /// regression where `data` was set to `as_ptr()` (logical first)
    /// instead of `as_storage_ptr()` (storage base).
    #[test]
    fn test_export_contract_data_storage_base_with_nonzero_offset() {
        let data = [10_i32, 20, 30, 40, 50];
        // SAFETY: F-order [2] view starting at offset 2 over data;
        // storage_base = data.as_ptr(); logical_first = data.as_ptr() + 2.
        let tensor = unsafe {
            TensorView::<i32, Ix1>::from_raw_parts(
                data.as_ptr(),          // ptr = storage base
                data.len(),             // storage_len = 5
                Ix1(2),                 // shape
                Strides::from_slice(&[1_usize]).expect("valid strides"),
                2,                      // offset ≠ 0
            )
        }
        .expect("F-order [2] view starting at offset 2");
        let raw = tensor.export();
        // Storage-base contract: `data` field equals storage base.
        assert_eq!(raw.data as *const i32, tensor.as_storage_ptr());
        // Offset preserved.
        assert_eq!(raw.offset, 2);
        // The logical first pointer is `data + offset` (NOT just `data`).
        // SAFETY: offset < storage_len ensures pointer stays within object.
        let logical_first = unsafe { (raw.data as *const i32).add(raw.offset) };
        assert_eq!(logical_first, tensor.as_ptr());
        // And it differs from `data` because offset ≠ 0.
        assert_ne!(raw.data as *const i32, tensor.as_ptr());
    }

    /// Empty tensors must export a dangling but valid aligned pointer.
    #[test]
    fn test_export_empty_tensor_dangling() {
        let data = Vec::<f64>::new();
        // SAFETY: shape [0] product 0 → validate_access_range early-returns;
        // storage_len = 0; F-order strides [1].
        let tensor = unsafe {
            TensorView::<f64, Ix1>::from_raw_parts(
                data.as_ptr(),
                0,
                Ix1(0),
                Strides::from_slice(&[1_usize]).expect("valid strides"),
                0,
            )
        }
        .expect("empty F-order [0] view");
        let raw = tensor.export();
        assert_eq!(raw.ndim, 1);
        assert_eq!(raw.storage_len, 0);
        // dangling pointers are non-null and properly aligned.
        assert!(!raw.data.is_null());
        assert_eq!((raw.data as usize) % core::mem::align_of::<f64>(), 0);
    }

    /// §8.2 test_export_mut_contract — `data` writable; only callable
    /// on `StorageMut`. We verify writability by storing through the raw
    /// pointer and reading back via the original tensor.
    #[test]
    fn test_export_mut_contract_writable() {
        let mut data = vec![1_i32, 2];
        // SAFETY: F-order [2] mutable view over data; offset = 0.
        let mut tensor = unsafe {
            TensorViewMut::<i32, Ix1>::from_raw_parts_mut(
                data.as_mut_ptr(),
                data.len(),
                Ix1(2),
                Strides::from_slice(&[1_usize]).expect("valid strides"),
                0,
            )
        }
        .expect("valid F-order [2] mutable view");
        let storage_base_before = tensor.as_storage_ptr() as usize;
        let raw = tensor.export_mut();
        assert_eq!(raw.ndim, 1);
        assert_eq!(raw.data as usize, storage_base_before);
        // SAFETY: raw.data is the storage base; storage_len=2; we write
        // within the borrowed range while no other reference exists
        // (export_mut consumed `&mut tensor` for its lifetime).
        unsafe {
            (raw.data as *mut i32).add(0).write(99);
        }
        // Reading back through `tensor` is sound after `raw` is dropped
        // because `TensorExportMutRaw` does not own the lifetime —
        // the borrow ends at end-of-expression for `raw`.
        let _ = raw;
        let observed = tensor.as_slice().expect("tensor is non-empty")[0];
        assert_eq!(observed, 99);
    }
}
