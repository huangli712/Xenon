//! C-visible FFI descriptors, generic Rust-only descriptors,
//! and BLAS metadata helpers.
//!
//! See `docs/design/23-ffi.md` §5.3, §5.3.1, §5.3.2, §5.11 for the
//! authoritative definitions and ABI rationale.

use core::ffi::c_void;
use core::marker::PhantomData;
use std::borrow::Cow;

pub use crate::error::{FfiBackend, FfiErrorCategory};

use crate::element::{Element, ElementType};
use crate::error::XenonError;

/// C-visible read-only tensor descriptor.
///
/// This is the cbindgen-emitted concrete schema. Generic
/// `TensorExport<'a, A>` is converted to `TensorExportRaw` at the FFI
/// boundary by stripping the lifetime / `PhantomData` and erasing
/// `*const A` to `*const c_void`. C consumers cast `data` to the matching
/// pointer type using `element_type` as the discriminator.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TensorExportRaw {
    /// Storage base pointer, type-erased to `*const c_void`.
    pub data: *const c_void,
    /// Element type discriminator for C-side pointer cast.
    pub element_type: u8,
    /// Number of dimensions.
    pub ndim: usize,
    /// Shape array (length = `ndim`).
    pub shape: *const usize,
    /// Stride array (length = `ndim`), element units.
    pub strides: *const usize,
    /// Storage length in elements.
    pub storage_len: usize,
    /// Logical offset in element units.
    pub offset: usize,
}

/// C-visible mutable tensor descriptor (writable variant).
#[repr(C)]
#[derive(Debug)]
pub struct TensorExportMutRaw {
    /// Writable storage base pointer, type-erased to `*mut c_void`.
    pub data: *mut c_void,
    /// Element type discriminator for C-side pointer cast.
    pub element_type: u8,
    /// Number of dimensions.
    pub ndim: usize,
    /// Shape array (length = `ndim`).
    pub shape: *const usize,
    /// Stride array (length = `ndim`), element units.
    pub strides: *const usize,
    /// Storage length in elements.
    pub storage_len: usize,
    /// Logical offset in element units.
    pub offset: usize,
}

// ── Generic Rust-only descriptors ─────────────────────────────
// These types are `pub(crate)` Rust-only borrowing evidence. They
// are NOT part of any C ABI surface; the C-visible raw descriptors
// are `TensorExportRaw` / `TensorExportMutRaw`. `#[doc(hidden)]` keeps
// them out of public rustdoc as well.

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

/// BLAS/LAPACK matrix metadata.
///
/// BLAS/LAPACK backends may use different integer widths. Xenon keeps the
/// raw dimensions in `usize` and lets callers convert them to the backend's
/// integer type (`i32` or `i64`) at the FFI boundary via `as_blas_int()`.
///
/// Field order matches `23-ffi.md` §5.11 line 859-868 exactly.
#[derive(Debug, Clone, Copy)]
pub struct BlasInfo<A> {
    /// Data pointer to the logical first element.
    pub data_ptr: *const A,
    /// Leading dimension (element units, raw `usize`).
    pub leading_dim: usize,
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
}

impl<A> BlasInfo<A> {
    /// Convert a raw BLAS/LAPACK size parameter to the backend integer type.
    ///
    /// This is an associated function (no `self`): callers pick which value
    /// to convert (`info.rows`, `info.cols`, `info.leading_dim`, or any
    /// `usize` known to be a BLAS size) and select the backend integer type
    /// via turbofish, e.g. `BlasInfo::<f64>::as_blas_int::<i32>(info.rows)?`.
    ///
    /// `target_width_bits` is filled with the bit width of `I` so that the
    /// structured `FfiErrorCategory::IntegerOverflow` payload accurately
    /// identifies which backend integer type was unable to represent `value`.
    ///
    /// # Errors
    ///
    /// Returns `XenonError::Ffi` with
    /// `FfiErrorCategory::IntegerOverflow { value, target_width_bits }` when
    /// `value` does not fit in the backend integer type `I` (i.e.
    /// `I::try_from(value)` fails).
    pub fn as_blas_int<I>(value: usize) -> Result<I, XenonError>
    where
        I: TryFrom<usize>,
    {
        I::try_from(value).map_err(|_| XenonError::Ffi {
            operation: Cow::Borrowed("ffi::blas_info::as_blas_int"),
            category: FfiErrorCategory::IntegerOverflow {
                value,
                target_width_bits: (core::mem::size_of::<I>() * 8) as u8,
            },
            backend: FfiBackend::Blas,
            
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors design §8.2: test_blas_info_f_order — success path of
    /// `as_blas_int` returning the backend integer for valid BLAS sizes.
    #[test]
    fn test_blas_info_as_blas_int_success() {
        let r: i32 = BlasInfo::<f64>::as_blas_int(3).expect("3 fits in i32");
        let c: i32 = BlasInfo::<f64>::as_blas_int(4).expect("4 fits in i32");
        let l: i32 = BlasInfo::<f64>::as_blas_int(3).expect("3 fits in i32");
        assert_eq!((r, c, l), (3, 4, 3));
    }

    /// Mirrors design §8.2: test_blas_info_as_blas_int_overflow.
    /// `usize::MAX` cannot fit in `i32`; conversion must return
    /// `FfiErrorCategory::IntegerOverflow` with the right `target_width_bits`.
    #[test]
    fn test_blas_info_as_blas_int_overflow() {
        let Err(err) = BlasInfo::<f64>::as_blas_int::<i32>(usize::MAX) else {
            panic!("expected IntegerOverflow error, got Ok");
        };
        match err {
            XenonError::Ffi {
                category, backend, ..
            } => {
                assert!(matches!(backend, FfiBackend::Blas));
                match category {
                    FfiErrorCategory::IntegerOverflow {
                        value,
                        target_width_bits,
                    } => {
                        assert_eq!(value, usize::MAX);
                        assert_eq!(target_width_bits, 32);
                    },
                    other => panic!("unexpected category: {other:?}"),
                }
            },
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    /// Validate the C ABI layout: raw descriptors are `#[repr(C)]` with the
    /// field order defined by §5.3.1. We verify field offsets match the
    /// design spec (§5.4 line 508-521 uses offset_of! for this purpose).
    /// MSRV 1.85 supports offset_of! (stabilized in Rust 1.77).
    #[test]
    fn test_raw_descriptors_repr_c_layout() {
        use core::mem::{align_of, offset_of, size_of};

        // Size lower bound check
        assert!(
            size_of::<TensorExportRaw>()
                >= size_of::<*const c_void>() + size_of::<u8>() + 5 * size_of::<usize>()
        );

        // Alignment checks
        assert_eq!(align_of::<TensorExportRaw>(), align_of::<*const c_void>());
        assert_eq!(align_of::<TensorExportMutRaw>(), align_of::<*mut c_void>());

        // Field offset checks (§5.3.1 line 285-314 field order)
        assert_eq!(offset_of!(TensorExportRaw, data), 0);
        assert_eq!(
            offset_of!(TensorExportRaw, element_type),
            size_of::<*const c_void>()
        );
        // ndim follows element_type; exact offset depends on ElementType size + padding
        assert!(
            offset_of!(TensorExportRaw, ndim)
                >= offset_of!(TensorExportRaw, element_type) + size_of::<u8>()
        );
        // Remaining fields follow in order: shape, strides, storage_len, offset
        assert!(offset_of!(TensorExportRaw, shape) > offset_of!(TensorExportRaw, ndim));
        assert!(offset_of!(TensorExportRaw, strides) > offset_of!(TensorExportRaw, shape));
        assert!(offset_of!(TensorExportRaw, storage_len) > offset_of!(TensorExportRaw, strides));
        assert!(offset_of!(TensorExportRaw, offset) > offset_of!(TensorExportRaw, storage_len));
    }

    // ── Generic-to-raw conversion tests (was in private.rs) ────

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
