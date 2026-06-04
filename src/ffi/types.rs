//! C-visible FFI descriptors and BLAS metadata helpers.
//!
//! See `docs/design/23-ffi.md` §5.3, §5.3.1, §5.11 for the
//! authoritative definitions and ABI rationale.

use core::ffi::c_void;
use std::borrow::Cow;

pub use crate::error::{FfiBackend, FfiErrorCategory};

use crate::error::XenonError;

/// C-visible read-only tensor descriptor.
///
/// This is the cbindgen-emitted concrete schema. `export()` builds it
/// directly from the source tensor, erasing the typed `*const A`
/// storage pointer to `*const c_void`. C consumers cast `data` to the
/// matching pointer type using `element_type` as the discriminator.
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
}
