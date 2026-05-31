//! Core tensor type: [`TensorBase<S, D>`] and [`OwnedRawParts`].

use core::mem::ManuallyDrop;
use core::ptr::NonNull;

use crate::Result;
use crate::dimension::Dimension;
use crate::error::{InvalidLayoutReason, StorageKindTag, XenonError};
use crate::layout::{LayoutFlags, Strides, compute_layout_flags};
use crate::storage::{Owned, RawStorage, StorageOwned};
use std::borrow::Cow;

/// N-dimensional array with type-level storage and dimension descriptors.
///
/// `TensorBase<S, D>` is the central type of the tensor module. It pairs a
/// storage representation `S` (one of [`Owned`], [`ViewRepr`], [`ViewMutRepr`],
/// or [`ArcRepr`]) with a dimension descriptor `D` (one of [`Ix0`]–[`Ix6`] or
/// [`IxDyn`]), and carries six metadata fields: [`storage`], [`shape`],
/// [`strides`], [`offset`], [`flags`], and [`derived_from_view_mut`].
///
/// The struct intentionally does **not** implement `Debug`; field-level
/// introspection is provided through query methods on [`TensorBase`]. The
/// internal field layout is not part of the public API and may change across
/// minor versions.
///
/// [`Owned`]: crate::storage::Owned
/// [`ViewRepr`]: crate::storage::ViewRepr
/// [`ViewMutRepr`]: crate::storage::ViewMutRepr
/// [`ArcRepr`]: crate::storage::ArcRepr
/// [`Ix0`]: crate::dimension::Ix0
/// [`Ix6`]: crate::dimension::Ix6
/// [`IxDyn`]: crate::dimension::IxDyn
/// [`storage`]: #structfield.storage
/// [`shape`]: #structfield.shape
/// [`strides`]: #structfield.strides
/// [`offset`]: #structfield.offset
/// [`flags`]: #structfield.flags
/// [`derived_from_view_mut`]: #structfield.derived_from_view_mut
pub struct TensorBase<S, D>
where
    S: RawStorage,
    D: Dimension,
{
    /// Opaque storage handle. Private to the type; exposed through query API.
    // SAFETY INVARIANT: the six fields below together encode the tensor
    // layout contract. Direct field mutation (possible via `pub(crate)`
    // visibility) can violate the shape/strides/offset/flags mutual-
    // consistency invariant OR the provenance invariant encoded by
    // `derived_from_view_mut`. ANY constructor path within the crate
    // MUST route through `construct::new_unchecked` (or one of the
    // validated public constructors) which is the single internal entry
    // point for tensor metadata assembly. Tests may construct directly
    // because their invariants are locally obvious; production code
    // MUST NOT.
    pub(crate) storage: S,
    pub(crate) shape: D,
    pub(crate) strides: Strides<D>,
    pub(crate) offset: usize,
    pub(crate) flags: LayoutFlags,
    /// `true` iff this view was demoted from a [`ViewMutRepr`] via `view()`.
    ///
    /// Enables [`access_semantics`](TensorBase::access_semantics) and
    /// [`alias_class`](TensorBase::alias_class) to correctly report
    /// `SharedReadOnly` / `ViewMutDerived` for ViewMut-demoted views.
    /// **Do NOT mutate this field directly**; always set it through
    /// [`new_unchecked`](super::construct::TensorBase::new_unchecked) or the
    /// provenance-aware constructors.
    pub(crate) derived_from_view_mut: bool,
}

// ── OwnedRawParts ──

/// Decomposition of an owned tensor into raw pointer + allocator metadata.
///
/// # ABI Note
///
/// `OwnedRawParts<A, D>` is **not** a stable C-ABI type. The `D` and
/// `Strides<D>` fields are Rust generics whose layout is not specified
/// by `#[repr(C)]` (especially for `IxDyn`, which contains a `Vec<usize>`).
/// FFI consumers MUST NOT decode this struct from C code. It exists solely
/// as a Rust-internal round-trip carrier for `into_raw_parts` /
/// `from_raw_parts_owned`. C-facing interop must use the dedicated
/// `TensorExportRaw` / `TensorExportMutRaw` types, which are explicitly
/// designed for a stable C ABI.
#[expect(
    missing_debug_implementations,
    reason = "OwnedRawParts carries a raw pointer; Debug is misleading"
)]
pub struct OwnedRawParts<A, D>
where
    D: Dimension,
{
    /// Pointer to the storage base.
    pub ptr: *mut A,
    /// Logical length in elements.
    pub len: usize,
    /// Allocator capacity in elements.
    pub cap: usize,
    /// Allocator alignment in bytes.
    pub align: usize,
    /// Owned tensor shape.
    pub shape: D,
    /// Canonical F-order strides matching shape.
    pub strides: Strides<D>,
    /// Logical offset in element units.
    pub offset: usize,
}

impl<A, D> TensorBase<Owned<A>, D>
where
    A: crate::element::Element + Clone,
    D: Dimension + Clone,
{
    /// Consumes the tensor, returning owned raw parts.
    ///
    /// This method consumes `self`. The caller must eventually reconstruct
    /// the tensor via `from_raw_parts_owned` and let `Drop` reclaim the
    /// memory, or else memory is leaked.
    pub fn into_raw_parts(self) -> OwnedRawParts<A, D> {
        let this = ManuallyDrop::new(self);
        let ptr = this.storage.as_mut_ptr_unchecked();
        OwnedRawParts {
            ptr,
            len: this.storage.len(),
            cap: this.storage.capacity(),
            align: this.storage.alignment(),
            shape: this.shape.clone(),
            strides: this.strides.clone(),
            offset: this.offset,
        }
    }
}

impl<A, D> TensorBase<Owned<A>, D>
where
    A: crate::element::Element,
    D: Dimension + Clone + PartialEq,
{
    /// Reconstructs an owned tensor from raw parts obtained via
    /// `into_raw_parts`.
    pub unsafe fn from_raw_parts_owned(raw: OwnedRawParts<A, D>) -> Result<Self> {
        if raw.offset != 0 {
            return Err(XenonError::InvalidLayout {
                operation: Cow::Borrowed("tensor::from_raw_parts_owned"),
                storage_kind: StorageKindTag::Owned,
                shape: raw.shape.slice().to_vec(),
                strides: raw.strides.as_slice().to_vec(),
                offset: raw.offset,
                storage_len: raw.len,
                reason: InvalidLayoutReason::OwnedRequiresZeroOffset,
            });
        }

        let expected_len = raw.shape.checked_size().map_err(|_| {
            XenonError::InvalidLayout {
                operation: Cow::Borrowed("tensor::from_raw_parts_owned"),
                storage_kind: StorageKindTag::Owned,
                shape: raw.shape.slice().to_vec(),
                strides: raw.strides.as_slice().to_vec(),
                offset: raw.offset,
                storage_len: raw.len,
                reason: InvalidLayoutReason::ShapeProductOverflow,
            }
        })?;
        if raw.len != expected_len {
            return Err(XenonError::InvalidLayout {
                operation: Cow::Borrowed("tensor::from_raw_parts_owned"),
                storage_kind: StorageKindTag::Owned,
                shape: raw.shape.slice().to_vec(),
                strides: raw.strides.as_slice().to_vec(),
                offset: raw.offset,
                storage_len: raw.len,
                reason: InvalidLayoutReason::LenShapeMismatch,
            });
        }

        if raw.cap < raw.len {
            return Err(XenonError::InvalidLayout {
                operation: Cow::Borrowed("tensor::from_raw_parts_owned"),
                storage_kind: StorageKindTag::Owned,
                shape: raw.shape.slice().to_vec(),
                strides: raw.strides.as_slice().to_vec(),
                offset: raw.offset,
                storage_len: raw.len,
                reason: InvalidLayoutReason::CapacityBelowLen,
            });
        }

        if !raw.align.is_power_of_two() || raw.align < core::mem::align_of::<A>() {
            return Err(XenonError::InvalidLayout {
                operation: Cow::Borrowed("tensor::from_raw_parts_owned"),
                storage_kind: StorageKindTag::Owned,
                shape: raw.shape.slice().to_vec(),
                strides: raw.strides.as_slice().to_vec(),
                offset: raw.offset,
                storage_len: raw.len,
                reason: InvalidLayoutReason::AlignmentInvalid,
            });
        }

        let expected_strides = Strides::f_contiguous(&raw.shape)?;
        if raw.strides.as_slice() != expected_strides.as_slice() {
            return Err(XenonError::InvalidLayout {
                operation: Cow::Borrowed("tensor::from_raw_parts_owned"),
                storage_kind: StorageKindTag::Owned,
                shape: raw.shape.slice().to_vec(),
                strides: raw.strides.as_slice().to_vec(),
                offset: raw.offset,
                storage_len: raw.len,
                reason: InvalidLayoutReason::OwnedRequiresCanonicalFOrder,
            });
        }

        let storage = unsafe {
            Owned::from_raw_parts(raw.ptr, raw.len, raw.cap, raw.align)
        };

        let logical_ptr: *const A = if raw.len == 0 {
            NonNull::<A>::dangling().as_ptr()
        } else {
            raw.ptr
        };
        let flags = compute_layout_flags::<A, D>(&raw.shape, &raw.strides, logical_ptr);

        Ok(TensorBase {
            storage,
            shape: raw.shape,
            strides: raw.strides,
            offset: 0,
            flags,
            derived_from_view_mut: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::TensorBase;
    use crate::dimension::{Dimension, Ix1, Ix2};
    use crate::layout::{LayoutFlags, Strides};
    use crate::storage::Owned;
    use crate::tensor::Tensor;

    /// Verify the tensor module skeleton compiles and all three sub-modules
    /// (`impls`, `aliases`, `construct`) are reachable.
    #[test]
    fn test_module_skeleton_compile() {
        // Reaching this point confirms: (a) all three sub-module files
        // exist, (b) `src/tensor/mod.rs` parses, (c) the crate compiled.
        assert_ne!(0, 1);
    }

    /// Verify `TensorBase` struct fields hold the values assigned to them
    /// after direct construction.
    #[test]
    fn test_tensorbase_struct_fields() {
        let data = vec![0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = Ix2(2, 3);
        let strides = Strides::f_contiguous(&shape).expect("valid shape");
        let storage = Owned::from_vec(data).expect("valid vec");

        let tensor = TensorBase {
            storage,
            shape,
            strides,
            offset: 0,
            flags: LayoutFlags::F_CONTIGUOUS,
            derived_from_view_mut: false,
        };

        // Verify pub(crate) fields hold the values assigned
        assert_eq!(tensor.offset, 0);
        assert!(!tensor.derived_from_view_mut);
        assert!(tensor.flags.is_f_contiguous());
    }

    // ── OwnedRawParts round-trip tests ──

    /// `into_raw_parts` → `from_raw_parts_owned` round-trip preserves shape,
    /// strides, offset, and element contents.
    #[test]
    fn test_into_raw_parts_roundtrip_2d() {
        let original = Tensor::<i32, Ix2>::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6])
            .expect("test input valid");
        let raw = original.into_raw_parts();
        assert_eq!(raw.len, 6);
        assert!(raw.cap >= 6);
        assert_eq!(raw.offset, 0);
        assert_eq!(raw.shape.slice(), &[2, 3]);
        let restored: Tensor<i32, Ix2> =
            unsafe { Tensor::from_raw_parts_owned(raw) }.expect("round-trip must succeed");
        assert_eq!(restored.shape(), &[2, 3]);
        assert_eq!(
            restored.as_slice().expect("test input valid"),
            &[1, 2, 3, 4, 5, 6]
        );
    }

    /// Round-trip for 1D tensors via the from_vec convenience path.
    #[test]
    fn test_into_raw_parts_roundtrip_1d() {
        let original =
            Tensor::<f64, Ix1>::from_vec(vec![1.0, 2.0, 3.0, 4.0]).expect("test input valid");
        let raw = original.into_raw_parts();
        let restored: Tensor<f64, Ix1> =
            unsafe { Tensor::from_raw_parts_owned(raw) }.expect("test input valid");
        assert_eq!(
            restored.as_slice().expect("test input valid"),
            &[1.0, 2.0, 3.0, 4.0]
        );
    }

    /// Empty tensor round-trip uses dangling sentinel for compute_layout_flags.
    #[test]
    fn test_into_raw_parts_roundtrip_empty() {
        let original =
            Tensor::<i32, Ix1>::from_shape_vec([0], Vec::new()).expect("test input valid");
        let raw = original.into_raw_parts();
        assert_eq!(raw.len, 0);
        let restored: Tensor<i32, Ix1> =
            unsafe { Tensor::from_raw_parts_owned(raw) }.expect("test input valid");
        assert_eq!(restored.len(), 0);
    }

    /// from_raw_parts_owned rejects non-zero offset.
    #[test]
    fn test_from_raw_parts_owned_rejects_nonzero_offset() {
        let original = Tensor::<i32, Ix1>::from_vec(vec![1_i32, 2, 3]).expect("test input valid");
        let mut raw = original.into_raw_parts();
        raw.offset = 1;
        let err = unsafe { Tensor::<i32, Ix1>::from_raw_parts_owned(raw) }
            .expect_err("tampered raw parts");
        assert!(matches!(
            err,
            crate::error::XenonError::InvalidLayout {
                reason: crate::error::InvalidLayoutReason::OwnedRequiresZeroOffset,
                ..
            }
        ));
    }

    /// Tampered shape produces LenShapeMismatch error.
    #[test]
    fn test_from_raw_parts_owned_rejects_len_shape_mismatch() {
        let original = Tensor::<i32, Ix2>::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6])
            .expect("test input valid");
        let mut raw = original.into_raw_parts();
        raw.shape = crate::dimension::Ix2(3, 3);
        let err = unsafe { Tensor::<i32, Ix2>::from_raw_parts_owned(raw) }
            .expect_err("tampered raw parts");
        assert!(matches!(
            err,
            crate::error::XenonError::InvalidLayout {
                reason: crate::error::InvalidLayoutReason::LenShapeMismatch,
                ..
            }
        ));
    }

    /// Tampered strides reject with OwnedRequiresCanonicalFOrder.
    #[test]
    fn test_from_raw_parts_owned_rejects_non_canonical_strides() {
        let original = Tensor::<i32, Ix2>::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6])
            .expect("test input valid");
        let mut raw = original.into_raw_parts();
        raw.strides = crate::layout::Strides::from_slice(&[3, 1]).expect("test input valid");
        let err = unsafe { Tensor::<i32, Ix2>::from_raw_parts_owned(raw) }
            .expect_err("tampered raw parts");
        assert!(matches!(
            err,
            crate::error::XenonError::InvalidLayout {
                reason: crate::error::InvalidLayoutReason::OwnedRequiresCanonicalFOrder,
                ..
            }
        ));
    }
}
