//! Generic Rust-only descriptors backing the C-visible FFI exports.
//!
//! These types are `#[doc(hidden)] pub(crate)`: they are not part of any
//! public surface (neither Rust nor C). See `docs/design/23-ffi.md` §5.3.1
//! / §5.3.2 for the cbindgen-emission-gate rationale.

use core::marker::PhantomData;

use super::types::{TensorExportMutRaw, TensorExportRaw};
use crate::element::{Element, ElementType};

/// Raw tensor data export for FFI consumers (read-only).
///
/// # Safety
///
/// - All pointer fields (`data`, `shape`, `strides`) borrow the source
///   tensor's internal storage and metadata. They become invalid
///   immediately after the source tensor is dropped.
/// - C consumers must use `ndim` as the length of both `shape` and
///   `strides` arrays.
/// - For `bool` element type, interoperability with C `_Bool` / C23
///   `bool` is only documented for explicitly supported platforms/ABIs.
///
/// **Visibility & file location**: Generic descriptors are `pub(crate)`
/// Rust-only borrowing evidence and live in `src/ffi/private.rs`
/// (see §3 file layout + §5.3.2 cbindgen gate #2: generic descriptors
/// are physically isolated and excluded from cbindgen emission set).
/// They are NOT part of any C ABI surface; the C-visible raw descriptors
/// are `TensorExportRaw` / `TensorExportMutRaw`. `#[doc(hidden)]` keeps
/// them out of public rustdoc as well.
#[doc(hidden)]
#[repr(C)]
pub(crate) struct TensorExport<'a, A> {
    /// Typed pointer to the storage base pointer.
    ///
    /// For non-empty tensors this points at the underlying storage base.
    /// For empty tensors (`len() == 0`), this is still a valid aligned
    /// pointer but must not be dereferenced.
    ///
    /// `strides` and `offset` use element units of `A`.
    /// The logical first element address is `data.add(offset)` when
    /// `len() != 0`.
    pub data: *const A,
    /// Element type identifier (matches `ElementType` enum).
    pub element_type: ElementType,
    /// Number of dimensions.
    ///
    /// Must be used as the length of both `shape` and `strides` arrays.
    pub ndim: usize,
    /// Shape array (length = `ndim`).
    pub shape: *const usize,
    /// Stride array (length = `ndim`), in element units (not bytes).
    pub strides: *const usize,
    /// Storage length in elements for safe view reconstruction.
    pub storage_len: usize,
    /// Logical offset metadata in element units, preserved for raw-parts
    /// roundtrip / reconstruction contracts.
    pub offset: usize,
    /// Lifetime marker tying the export to the source tensor borrow.
    ///
    /// **Must be the last field** in this `#[repr(C)]` struct: as a ZST
    /// it contributes 0 bytes in the C ABI, but placing it in the middle
    /// can produce unspecified behavior across compiler versions.
    pub _marker: PhantomData<&'a A>,
}

/// Raw mutable tensor data export for FFI consumers.
///
/// Field semantics are identical to `TensorExport` unless noted: `data`
/// is `*mut A` (writable) and `_marker` uses `PhantomData<&'a mut A>`
/// (exclusive borrow).
///
/// **Visibility & file location**: same `pub(crate)` Rust-only scope and
/// `src/ffi/private.rs` location as `TensorExport`. Not part of any C
/// ABI surface; C consumers see `TensorExportMutRaw` instead.
#[doc(hidden)]
#[repr(C)]
pub(crate) struct TensorExportMut<'a, A> {
    /// Writable typed pointer to the storage base pointer.
    pub data: *mut A,
    /// Element type identifier (matches `ElementType` enum).
    pub element_type: ElementType,
    /// Number of dimensions.
    pub ndim: usize,
    /// Shape array (length = `ndim`).
    pub shape: *const usize,
    /// Stride array (length = `ndim`), in element units.
    pub strides: *const usize,
    /// Storage length in elements.
    pub storage_len: usize,
    /// Logical offset metadata in element units.
    pub offset: usize,
    /// Lifetime marker; `PhantomData<&'a mut A>` enforces exclusive borrow.
    ///
    /// Must be the last field (ZST), same rationale as `TensorExport::_marker`.
    pub _marker: PhantomData<&'a mut A>,
}

impl<'a, A: Element> From<TensorExport<'a, A>> for TensorExportRaw {
    fn from(e: TensorExport<'a, A>) -> Self {
        TensorExportRaw {
            data: e.data.cast(),
            element_type: e.element_type as u8,
            ndim: e.ndim,
            shape: e.shape,
            strides: e.strides,
            storage_len: e.storage_len,
            offset: e.offset,
        }
    }
}

impl<'a, A: Element> From<TensorExportMut<'a, A>> for TensorExportMutRaw {
    fn from(e: TensorExportMut<'a, A>) -> Self {
        TensorExportMutRaw {
            data: e.data.cast(),
            element_type: e.element_type as u8,
            ndim: e.ndim,
            shape: e.shape,
            strides: e.strides,
            storage_len: e.storage_len,
            offset: e.offset,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify generic-to-raw conversion preserves all 7 metadata fields.
    /// Mirrors design §8.2 "raw-parts roundtrip" intent: every byte of the
    /// internal descriptor must survive the FFI-boundary transition.
    #[test]
    fn test_tensor_export_to_raw_conversion() {
        let shape = [2_usize, 3];
        let strides = [1_usize, 2];
        let export = TensorExport::<f64> {
            // Use dangling pointer (§5.4 line 568-575: "valid aligned pointer")
            // rather than null() to match the design's empty-tensor semantics.
            data: core::ptr::NonNull::<f64>::dangling().as_ptr(),
            element_type: ElementType::F64,
            ndim: 2,
            shape: shape.as_ptr(),
            strides: strides.as_ptr(),
            storage_len: 6,
            offset: 1,
            _marker: PhantomData,
        };
        let raw: TensorExportRaw = export.into();
        assert_eq!(raw.ndim, 2);
        assert_eq!(raw.storage_len, 6);
        assert_eq!(raw.offset, 1);
        assert_eq!(raw.element_type, ElementType::F64 as u8);
        assert_eq!(raw.shape, shape.as_ptr());
        assert_eq!(raw.strides, strides.as_ptr());
    }

    #[test]
    fn test_tensor_export_mut_to_raw_conversion() {
        let mut data = [1.0_f64, 2.0, 3.0];
        let shape = [3_usize];
        let strides = [1_usize];
        let export = TensorExportMut::<f64> {
            data: data.as_mut_ptr(),
            element_type: ElementType::F64,
            ndim: 1,
            shape: shape.as_ptr(),
            strides: strides.as_ptr(),
            storage_len: 3,
            offset: 0,
            _marker: PhantomData,
        };
        let raw: TensorExportMutRaw = export.into();
        assert_eq!(raw.ndim, 1);
        assert_eq!(raw.storage_len, 3);
        assert_eq!(raw.offset, 0);
        assert_eq!(raw.element_type, ElementType::F64 as u8);
        assert!(!raw.data.is_null());
        assert_eq!(raw.shape, shape.as_ptr());
        assert_eq!(raw.strides, strides.as_ptr());
    }

    /// `_marker` is a ZST and must sit at the tail of the `#[repr(C)]`
    /// struct so that C consumers see fields ending at `offset: usize`.
    /// `size_of` is bounded below by the sum of the non-ZST fields plus
    /// any padding required by `#[repr(C)]`.
    #[test]
    fn test_phantom_data_is_zst_and_trailing() {
        use core::mem::size_of;
        // Non-ZST trailing fields sum (with worst-case padding) ≤ struct size.
        let non_zst_min =
            size_of::<*const f64>() + size_of::<ElementType>() + 5 * size_of::<usize>();
        assert!(size_of::<TensorExport<'_, f64>>() >= non_zst_min);
        // PhantomData<&'_ A> is ZST, so it should not change struct size
        // relative to a hypothetical layout without it.
        assert_eq!(size_of::<PhantomData<&'_ f64>>(), 0);
    }
}
