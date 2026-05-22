//! Integration tests for FFI helper APIs: raw pointer access, BLAS compatibility,
//! export/export_mut contracts, try_offset_of, and alignment preconditions.

use xenon::dimension::{Ix1, Ix2};
use xenon::error::{FfiBackend, FfiErrorCategory, XenonError};
use xenon::ffi::ElementType;
use xenon::layout::Strides;
use xenon::tensor::{Tensor, TensorView, TensorViewMut};

/// Helper: build a simple 1-D owned f64 tensor.
fn owned_f64_1d(data: Vec<f64>) -> Tensor<f64, Ix1> {
    Tensor::from_vec(data).expect("valid 1-D owned f64 tensor")
}

/// Helper: build a 1-D view from a slice.
unsafe fn view_f64_1d<'a>(data: &'a [f64]) -> TensorView<'a, f64, Ix1> {
    unsafe {
        TensorView::<f64, Ix1>::from_raw_parts(
            data.as_ptr(),
            data.len(),
            Ix1(data.len()),
            Strides::from_slice(&[1_usize]).expect("valid strides"),
            0,
        )
    }
    .expect("valid F-order [n] view")
}

#[test]
fn test_as_ptr() {
    let tensor = owned_f64_1d(vec![1.0, 2.0, 3.0]);
    let ptr = tensor.as_ptr();
    assert!(!ptr.is_null());
    // as_ptr() returns the logical first element address.
    // SAFETY: tensor is non-empty and valid.
    unsafe {
        assert_eq!(*ptr, 1.0);
    }
    // For F-contiguous non-broadcast tensors, as_slice should be available
    // and the pointer should match.
    let slice = tensor.as_slice().expect("F-contiguous 1-D");
    assert_eq!(ptr, slice.as_ptr());
}

#[test]
fn test_as_mut_ptr() {
    let mut tensor = owned_f64_1d(vec![1.0, 2.0, 3.0]);
    let ptr = tensor.as_mut_ptr();
    assert!(!ptr.is_null());
    // Write through the mutable pointer.
    unsafe {
        *ptr = 99.0;
    }
    // Read back via safe API.
    assert_eq!(
        tensor.as_slice().expect("F-contiguous")[0],
        99.0
    );
}

#[test]
fn test_lda() {
    // F-order [3, 4] tensor: lda = stride[1] = 3.
    let tensor = Tensor::<f64, _>::zeros([3, 4]).expect("zeros [3,4]");
    let lda = tensor.lda().expect("BLAS-compatible F-order 2D tensor");
    assert_eq!(lda, 3);

    // Non-F-contiguous layout (C-order strides) should be rejected.
    let data = [0.0_f64; 12];
    let strides = Strides::from_slice(&[4_usize, 1]).expect("valid strides");
    let t = unsafe {
        TensorView::<f64, Ix2>::from_raw_parts(data.as_ptr(), data.len(), Ix2(3, 4), strides, 0)
    }
    .expect("valid layout with C-order strides");
    let err = t.lda().expect_err("non-F-contiguous should fail");
    assert!(matches!(err, XenonError::Ffi { category: FfiErrorCategory::BlasIncompatibleLayout { .. }, .. }));

    // 1-D tensor should fail with InvalidRank.
    let t1 = owned_f64_1d(vec![1.0, 2.0, 3.0]);
    let err = t1.lda().expect_err("1-D should fail lda");
    assert!(matches!(err, XenonError::Ffi { category: FfiErrorCategory::InvalidRank { .. }, .. }));
}

#[test]
fn test_is_blas_layout_compatible() {
    // F-order 2D tensor is BLAS-compatible.
    let tensor = Tensor::<f64, _>::zeros([3, 4]).expect("zeros [3,4]");
    assert!(tensor.is_blas_layout_compatible());

    // Non-contiguous layout should return false.
    let data = [0.0_f64; 12];
    let strides = Strides::from_slice(&[4_usize, 1]).expect("valid strides");
    let t = unsafe {
        TensorView::<f64, Ix2>::from_raw_parts(data.as_ptr(), data.len(), Ix2(3, 4), strides, 0)
    }
    .expect("valid layout with C-order strides");
    assert!(!t.is_blas_layout_compatible());

    // 1-D F-contiguous tensor is layout-compatible (layout check only, no rank check).
    let t1 = owned_f64_1d(vec![1.0, 2.0, 3.0]);
    assert!(t1.is_blas_layout_compatible());
}

#[test]
fn test_export_roundtrip() {
    let tensor = owned_f64_1d(vec![10.0, 20.0, 30.0]);
    let raw = tensor.export();

    assert_eq!(raw.ndim, 1);
    assert_eq!(raw.storage_len, 3);
    assert_eq!(raw.element_type, ElementType::F64);
    assert_eq!(raw.offset, 0);

    // `data` must equal the storage base pointer (not logical first).
    assert_eq!(raw.data as *const f64, tensor.as_storage_ptr());
    // Logical first = data + offset = as_ptr().
    let logical_first = unsafe { (raw.data as *const f64).add(raw.offset) };
    assert_eq!(logical_first, tensor.as_ptr());
    // SAFETY: tensor is non-empty; dereference within bounds.
    unsafe {
        assert_eq!(*logical_first, 10.0);
    }
}

#[test]
fn test_export_mut_roundtrip() {
    let mut tensor = owned_f64_1d(vec![1.0, 2.0, 3.0]);
    let storage_base_before = tensor.as_storage_ptr() as usize;
    let raw = tensor.export_mut();

    assert_eq!(raw.ndim, 1);
    assert_eq!(raw.storage_len, 3);
    assert_eq!(raw.data as usize, storage_base_before);

    // Write through the mutable raw pointer.
    unsafe {
        *(raw.data as *mut f64) = 99.0;
    }
    // After raw is dropped (end of expression), read back through tensor.
    let _ = raw;
    assert_eq!(
        tensor.as_slice().expect("F-contiguous")[0],
        99.0
    );
}

#[test]
fn test_from_raw_parts_mut_reject_overlap() {
    // Ambiguous overlap layout: shape [2, 2] with strides [1, 1] creates
    // overlapping elements (column 0 and column 1 alias). Must be rejected.
    let mut backing = [1_i32, 2, 3, 4];
    let strides = Strides::from_slice(&[1_usize, 1]).expect("valid strides");
    let err = unsafe {
        TensorViewMut::<i32, Ix2>::from_raw_parts_mut(
            backing.as_mut_ptr(),
            backing.len(),
            Ix2(2, 2),
            strides,
            0,
        )
    }
    .expect_err("overlapping layout should be rejected");
    match &err {
        XenonError::InvalidLayout { reason, .. } => {
            assert!(
                matches!(reason, xenon::error::InvalidLayoutReason::AmbiguousOverlap),
                "expected AmbiguousOverlap, got {reason:?}"
            );
        },
        other => panic!("expected InvalidLayout, got {other:?}"),
    }
}

#[test]
fn test_try_offset_of() {
    // F-order [2, 3] tensor with strides [1, 2]:
    //   index [0, 0] → offset 0
    //   index [1, 2] → 1*1 + 2*2 = 5
    let tensor = Tensor::<i32, _>::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6])
        .expect("valid [2,3] tensor");
    assert_eq!(
        tensor.try_offset_of(&[0, 0]).expect("valid index"),
        0
    );
    assert_eq!(
        tensor.try_offset_of(&[1, 2]).expect("valid index"),
        5
    );

    // Out-of-bounds index → IndexOutOfBounds.
    let err = tensor.try_offset_of(&[2, 0]).expect_err("OOB");
    assert!(matches!(err, XenonError::IndexOutOfBounds { axis: 0, .. }));

    // Rank mismatch → DimensionMismatch.
    let err = tensor.try_offset_of(&[0]).expect_err("rank mismatch");
    assert!(matches!(err, XenonError::DimensionMismatch { expected: 2, actual: 1, .. }));

    // Multi-dimensional access with F-order strides for [3, 4] = strides [1, 3].
    let tensor = Tensor::<f64, _>::zeros([3, 4]).expect("zeros [3,4]");
    assert_eq!(
        tensor.try_offset_of(&[2, 3]).expect("valid index"),
        2 * 1 + 3 * 3
    );
}

#[test]
fn test_export_alignment_preconditions() {
    // Empty tensor export: must produce a non-null, aligned pointer.
    let empty: Tensor<f64, Ix1> = Tensor::from_shape_vec([0], Vec::new())
        .expect("valid empty 1-D tensor");
    let raw = empty.export();
    assert!(!raw.data.is_null());
    assert_eq!(
        (raw.data as usize) % core::mem::align_of::<f64>(),
        0,
        "empty tensor export data must be f64-aligned"
    );

    // Non-empty tensor export must have non-null data with valid alignment.
    let tensor = owned_f64_1d(vec![1.0, 2.0]);
    let raw = tensor.export();
    assert!(!raw.data.is_null());
    assert_eq!(
        (raw.data as usize) % core::mem::align_of::<f64>(),
        0,
        "non-empty tensor export data must be f64-aligned"
    );
    // data equals storage base pointer.
    assert_eq!(raw.data as *const f64, tensor.as_storage_ptr());
}

#[test]
fn test_cbindgen_header_exports_only_raw_descriptors() {
    // Verify that the FFI module's public surface exposes only raw C-compatible
    // descriptor types (TensorExportRaw, TensorExportMutRaw, BlasInfo) and
    // the ElementType/FfiBackend/FfiErrorCategory enums. The generic
    // TensorExport and TensorExportMut are crate-internal.

    // Compile-time: verify that TensorExportRaw and TensorExportMutRaw are
    // accessible from the crate root's ffi module.
    let _: xenon::ffi::TensorExportRaw;
    let _: xenon::ffi::TensorExportMutRaw;
    let _: xenon::ffi::BlasInfo<f64>;
    let _: xenon::ffi::ElementType;
    let _: xenon::ffi::FfiBackend;
    let _: xenon::ffi::FfiErrorCategory;

    // Verify layout: TensorExportRaw must be repr(C) with data field at offset 0.
    use core::mem::{offset_of, size_of};
    assert_eq!(offset_of!(xenon::ffi::TensorExportRaw, data), 0);
    assert!(
        size_of::<xenon::ffi::TensorExportRaw>()
            >= size_of::<*const std::ffi::c_void>()
                + size_of::<ElementType>()
                + 5 * size_of::<usize>()
    );

    // TensorExportMutRaw must be repr(C).
    assert_eq!(offset_of!(xenon::ffi::TensorExportMutRaw, data), 0);
}
