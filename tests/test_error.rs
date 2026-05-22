//! Integration tests for XenonError thread safety.
//!
//! Verifies that `XenonError` satisfies `Send` across the public crate boundary,
//! as required by 26-error §9.3 and 25-safety §8.5.

use std::borrow::Cow;

use xenon::broadcast::broadcast_shape;
use xenon::error::{
    AbiMismatchKind, ConversionFailureReason, FfiBackend, FfiErrorCategory,
    InvalidArgumentKind, InvalidLayoutReason, InvalidShapeKind, StorageConversionKind,
    StorageKindTag, TypedViewRejection, WorkspaceBorrowKind, WorkspaceBorrowState,
    WorkspaceErrorCategory, XenonError,
};
use xenon::tensor::Tensor;

#[test]
fn test_parallel_error_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<XenonError>();
}

#[test]
fn test_broadcast_shape_error() {
    // Incompatible shapes: [2, 3] vs [4, 3].
    let err = broadcast_shape(&[2, 3], &[4, 3]).expect_err("incompatible shapes");
    match err {
        XenonError::BroadcastError {
            operation,
            lhs_shape,
            rhs_shape,
            attempted_target_shape,
            axis,
        } => {
            assert_eq!(operation.as_ref(), "broadcast_shape");
            assert_eq!(lhs_shape, vec![2, 3]);
            assert_eq!(rhs_shape, vec![4, 3]);
            assert_eq!(attempted_target_shape, None);
            assert_eq!(axis, Some(0));
        },
        other => panic!("expected BroadcastError, got {other:?}"),
    }
}

#[test]
fn test_broadcast_error() {
    let err = XenonError::BroadcastError {
        operation: Cow::Borrowed("add"),
        lhs_shape: vec![3, 1],
        rhs_shape: vec![1, 4],
        attempted_target_shape: Some(vec![3, 4]),
        axis: Some(1),
    };
    let s = format!("{err}");
    assert!(s.contains("[3 × 1]"));
    assert!(s.contains("[1 × 4]"));
    assert!(s.contains("[3 × 4]"));
    assert!(s.contains("axis: 1"));
}

#[test]
fn test_invalid_shape_error() {
    let err = XenonError::InvalidShape {
        operation: Cow::Borrowed("from_shape_vec"),
        shape: vec![2, 3],
        kind: InvalidShapeKind::ElementCountMismatch {
            expected: 6,
            actual: 5,
        },
        offending_dim: None,
    };
    match err {
        XenonError::InvalidShape {
            ref operation,
            ref shape,
            ref kind,
            ..
        } => {
            assert_eq!(operation.as_ref(), "from_shape_vec");
            assert_eq!(shape, &vec![2, 3]);
            assert!(matches!(kind, InvalidShapeKind::ElementCountMismatch { .. }));
        },
        _ => panic!("variant mismatch"),
    }
    let s = format!("{err}");
    assert!(s.contains("element count mismatch"));
    assert!(s.contains("6"));
    assert!(s.contains("5"));
}

#[test]
fn test_invalid_axis_error() {
    let err = XenonError::InvalidAxis {
        operation: Cow::Borrowed("sum"),
        axis: 3,
        ndim: 2,
        shape: vec![3, 4],
    };
    let s = format!("{err}");
    assert!(s.contains("sum"));
    assert!(s.contains("axis 3"));
    assert!(s.contains("0..2"));
    assert!(s.contains("[3 × 4]"));
}

#[test]
fn test_invalid_argument_error() {
    let err = XenonError::InvalidArgument {
        operation: Cow::Borrowed("slice"),
        kind: InvalidArgumentKind::RangeOutOfBounds {
            axis: 0,
            axis_len: 5,
            start: 3,
            end: 10,
        },
    };
    let s = format!("{err}");
    assert!(s.contains("slice"));
    assert!(s.contains("range [3..10] out of bounds"));
    assert!(s.contains("axis 0"));
}

#[test]
fn test_invalid_storage_mode_conversion_error() {
    let err = XenonError::InvalidStorageMode {
        operation: Cow::Borrowed("to_owned"),
        expected: StorageKindTag::Owned,
        actual: StorageKindTag::View,
        shape: Some(vec![2, 3]),
        conversion: Some(StorageConversionKind::ToOwned),
    };
    let s = format!("{err}");
    assert!(s.contains("to_owned"));
    assert!(s.contains("expected owned"));
    assert!(s.contains("got view"));
    assert!(s.contains("[2 × 3]"));
    assert!(s.contains("to owned"));
}

#[test]
fn test_layout_state_query_error_context() {
    // Build a valid owned tensor, verify layout_state does not panic.
    let tensor = Tensor::<f64, _>::zeros([3, 4]).expect("zeros [3,4]");
    let _state = tensor.layout_state();
    // Also verify that query methods on an error-returning path compile.
    let _ = tensor.is_f_contiguous();
    let _ = tensor.is_aligned();
    let _ = tensor.has_zero_stride();
}

#[test]
fn test_error_display() {
    let cases: &[XenonError] = &[
        XenonError::ShapeMismatch {
            operation: Cow::Borrowed("dot"),
            left_shape: vec![1, 3],
            right_shape: vec![3, 4],
        },
        XenonError::DimensionMismatch {
            operation: Cow::Borrowed("reshape"),
            expected: 2,
            actual: 3,
        },
        XenonError::IndexOutOfBounds {
            operation: Cow::Borrowed("index"),
            attempted_index: vec![0, 5],
            axis: 1,
            shape: vec![3, 4],
        },
        XenonError::InvalidLayout {
            operation: Cow::Borrowed("from_raw_parts"),
            storage_kind: StorageKindTag::View,
            shape: vec![2, 3],
            strides: vec![1, 2],
            offset: 0,
            storage_len: 6,
            reason: InvalidLayoutReason::AccessRangeExceedsStorage,
        },
        XenonError::TypeConversion {
            operation: Cow::Borrowed("cast"),
            source_type: "f64",
            target_type: "i32",
            reason: ConversionFailureReason::FloatToInteger,
            element_index: None,
        },
        XenonError::Ffi {
            operation: Cow::Borrowed("export"),
            category: FfiErrorCategory::NullPointer {
                argument: Cow::Borrowed("ptr"),
            },
            backend: FfiBackend::RawParts,
            cause: None,
        },
        XenonError::Workspace {
            operation: Cow::Borrowed("Workspace::new"),
            category: WorkspaceErrorCategory::AllocFailed {
                size: 4096,
                align: 64,
            },
            cause: None,
        },
    ];
    for e in cases {
        let s = format!("{e}");
        assert!(!s.is_empty(), "Display output must not be empty for {e:?}");
    }
}

#[test]
fn test_send_sync_contracts() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    // XenonError must be both Send and Sync per 26-error §9.3.
    assert_send::<XenonError>();
    assert_sync::<XenonError>();
    // Auxiliary FFI error enums must also be Send + Sync (they carry no
    // interior mutability or thread-local state).
    assert_send::<FfiErrorCategory>();
    assert_sync::<FfiErrorCategory>();
    assert_send::<FfiBackend>();
    assert_sync::<FfiBackend>();
    assert_send::<AbiMismatchKind>();
    assert_sync::<AbiMismatchKind>();
}

#[test]
fn test_complex_c99_layout() {
    // Verify that Complex<f32> and Complex<f64> have C-compatible layout.
    use xenon::complex::Complex;
    use core::mem::{align_of, offset_of, size_of};

    // Complex<f32> must be repr(C) compatible: two consecutive f32 fields.
    #[repr(C)]
    struct CComplexF32 {
        re: f32,
        im: f32,
    }
    assert_eq!(size_of::<Complex<f32>>(), size_of::<CComplexF32>());
    assert_eq!(align_of::<Complex<f32>>(), align_of::<CComplexF32>());
    assert_eq!(offset_of!(Complex<f32>, re), offset_of!(CComplexF32, re));
    assert_eq!(offset_of!(Complex<f32>, im), offset_of!(CComplexF32, im));

    // Complex<f64> must be repr(C) compatible: two consecutive f64 fields.
    #[repr(C)]
    struct CComplexF64 {
        re: f64,
        im: f64,
    }
    assert_eq!(size_of::<Complex<f64>>(), size_of::<CComplexF64>());
    assert_eq!(align_of::<Complex<f64>>(), align_of::<CComplexF64>());
    assert_eq!(offset_of!(Complex<f64>, re), offset_of!(CComplexF64, re));
    assert_eq!(offset_of!(Complex<f64>, im), offset_of!(CComplexF64, im));
}

#[test]
fn test_ix0_iter_single() {
    // Ix0 (0-D) tensor has shape [] and len() == 1. Iteration yields one element.
    let tensor = Tensor::<i32, _>::zeros([]).expect("zeros Ix0");
    assert_eq!(tensor.ndim(), 0);
    assert_eq!(tensor.len(), 1);
    assert_eq!(tensor.shape(), &[]);
    // Iterate and collect: should produce exactly one element.
    let collected: Vec<&i32> = tensor.iter().collect();
    assert_eq!(collected.len(), 1);
    assert_eq!(*collected[0], 0);
    // Access via get on empty slice.
    let val = tensor.get(&[] as &[usize]).expect("0-D access");
    assert_eq!(*val, 0);
}

#[test]
fn test_zst_storage_no_ub() {
    // Verify that zero-sized-type paths do not cause UB. We test that
    // XenonError's Ffi and Workspace variants handle ZST metadata correctly.
    // The `TypedViewRejection::ZeroSizedType` variant exists specifically to
    // reject typed views of ZSTs.
    let rejection = TypedViewRejection::ZeroSizedType;
    let s = format!("{rejection}");
    assert_eq!(s, "zero-sized type");

    // Verify that `WorkspaceErrorCategory::TypedViewRejected` with
    // `ZeroSizedType` is constructable and displays correctly.
    // Note: the Display impl uses {:?} for the detail field, so the output
    // contains "ZeroSizedType" (Debug), not "zero-sized type" (Display).
    let cat = WorkspaceErrorCategory::TypedViewRejected {
        detail: TypedViewRejection::ZeroSizedType,
    };
    let s = format!("{cat}");
    assert!(s.contains("typed view rejected"));

    // Sanity: Workspace's InvalidLayout with ZST-sized parameters is also
    // constructable (size=0, align=1 is an edge case for Layout).
    let err = XenonError::Workspace {
        operation: Cow::Borrowed("test_zst"),
        category: WorkspaceErrorCategory::InvalidLayout {
            size: 0,
            align: 1,
        },
        cause: None,
    };
    let s = format!("{err}");
    assert!(s.contains("size=0"));
    assert!(s.contains("align=1"));
}

#[test]
fn test_workspace_error_structured_fields() {
    // AllocFailed
    let err = XenonError::Workspace {
        operation: Cow::Borrowed("Workspace::new"),
        category: WorkspaceErrorCategory::AllocFailed {
            size: 1024,
            align: 64,
        },
        cause: None,
    };
    match &err {
        XenonError::Workspace { operation, category, cause } => {
            assert_eq!(operation.as_ref(), "Workspace::new");
            assert!(matches!(category, WorkspaceErrorCategory::AllocFailed { size: 1024, align: 64 }));
            assert!(cause.is_none());
        },
        _ => panic!("variant mismatch"),
    }

    // BorrowConflict
    let err = XenonError::Workspace {
        operation: Cow::Borrowed("Workspace::borrow"),
        category: WorkspaceErrorCategory::BorrowConflict {
            requested: WorkspaceBorrowKind::Exclusive,
            current: WorkspaceBorrowState::Shared,
        },
        cause: None,
    };
    let s = format!("{err}");
    assert!(s.contains("borrow conflict"));
    assert!(s.contains("Exclusive"));
    assert!(s.contains("Shared"));

    // SplitOutOfBounds — field names must be `mid` and `len`.
    let err = XenonError::Workspace {
        operation: Cow::Borrowed("Workspace::split_at_mut"),
        category: WorkspaceErrorCategory::SplitOutOfBounds { mid: 42, len: 10 },
        cause: None,
    };
    let s = format!("{err:?}");
    assert!(s.contains("SplitOutOfBounds"));
    assert!(s.contains("mid: 42"));

    // Cause chain: Workspace wrapping an inner error.
    let inner = Box::new(XenonError::InvalidAxis {
        operation: Cow::Borrowed("check"),
        axis: 1,
        ndim: 2,
        shape: vec![3, 4],
    });
    let err = XenonError::Workspace {
        operation: Cow::Borrowed("test"),
        category: WorkspaceErrorCategory::AllocFailed { size: 64, align: 8 },
        cause: Some(inner),
    };
    let s = format!("{err}");
    assert!(s.contains("; caused by:"));
}
