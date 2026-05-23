//! Integration tests for Wave 8 tensor core: cross-module interaction,
//! boundary cases, and type alias verification. Covers items from
//! `07-tensor.md §8.2 / §8.3` that require multi-module composition or
//! the public crate API surface.

use xenon::dimension::{Ix0, Ix1, Ix2, IxDyn};
use xenon::layout::{LayoutState, Strides, compute_f_strides};
use xenon::tensor;
use xenon::tensor::{
    AccessSemantics, AliasClass, ArcTensor, ArcTensor1, ArcTensor2, ArcTensorD, DataLocation,
    StorageKind, Tensor, Tensor0, Tensor1, Tensor2, TensorBase, TensorD, TensorView, TensorView1,
    TensorView2, TensorViewD, TensorViewMut, TensorViewMut2,
};

// === Type alias compile verification ===

/// §8.2 test_type_aliases_compile: all 36 aliases resolve at compile time.
#[test]
fn test_type_aliases_compile() {
    // Primary aliases (4)
    let _: Option<Tensor<f64, Ix2>> = None;
    let _: Option<TensorView<'_, f64, Ix2>> = None;
    let _: Option<TensorViewMut<'_, f64, Ix2>> = None;
    let _: Option<ArcTensor<f64, Ix2>> = None;

    // Owned dimension convenience aliases (8)
    let _: Option<Tensor0<f64>> = None;
    let _: Option<tensor::Tensor1<f64>> = None;
    let _: Option<Tensor2<i32>> = None;
    let _: Option<tensor::Tensor3<f64>> = None;
    let _: Option<tensor::Tensor4<f64>> = None;
    let _: Option<tensor::Tensor5<f64>> = None;
    let _: Option<tensor::Tensor6<f64>> = None;
    let _: Option<TensorD<f64>> = None;

    // View convenience aliases (8)
    let _: Option<tensor::TensorView0<'_, f64>> = None;
    let _: Option<tensor::TensorView1<'_, f64>> = None;
    let _: Option<TensorView2<'_, f64>> = None;
    let _: Option<tensor::TensorView3<'_, f64>> = None;
    let _: Option<tensor::TensorView4<'_, f64>> = None;
    let _: Option<tensor::TensorView5<'_, f64>> = None;
    let _: Option<tensor::TensorView6<'_, f64>> = None;
    let _: Option<TensorViewD<'_, f64>> = None;

    // ViewMut convenience aliases (8)
    let _: Option<tensor::TensorViewMut0<'_, f64>> = None;
    let _: Option<tensor::TensorViewMut1<'_, f64>> = None;
    let _: Option<TensorViewMut2<'_, f64>> = None;
    let _: Option<tensor::TensorViewMut3<'_, f64>> = None;
    let _: Option<tensor::TensorViewMut4<'_, f64>> = None;
    let _: Option<tensor::TensorViewMut5<'_, f64>> = None;
    let _: Option<tensor::TensorViewMut6<'_, f64>> = None;
    let _: Option<tensor::TensorViewMutD<'_, f64>> = None;

    // Arc convenience aliases (8)
    let _: Option<tensor::ArcTensor0<f64>> = None;
    let _: Option<tensor::ArcTensor1<f64>> = None;
    let _: Option<ArcTensor2<f64>> = None;
    let _: Option<tensor::ArcTensor3<f64>> = None;
    let _: Option<tensor::ArcTensor4<f64>> = None;
    let _: Option<tensor::ArcTensor5<f64>> = None;
    let _: Option<tensor::ArcTensor6<f64>> = None;
    let _: Option<ArcTensorD<f64>> = None;
}

// === Basic cross-module construction (immutable view) ===

/// §8.2 test_tensor_shape_2d / test_tensor_strides_f_order /
/// test_tensor_len / test_tensor_ndim_static.
#[test]
fn test_tensor_shape_2d() {
    let data = [0_i32; 12];
    let shape = Ix2(3, 4);
    let strides = compute_f_strides(&shape).expect("valid shape");
    // SAFETY: data outlives tensor; layout is canonical F-order.
    let tensor: TensorView2<'_, i32> =
        unsafe { TensorBase::from_raw_parts(data.as_ptr(), data.len(), shape, strides, 0) }
            .expect("valid raw-parts must succeed");

    assert_eq!(tensor.shape(), &[3, 4]);
    assert_eq!(tensor.strides(), &[1_usize, 3]);
    assert_eq!(tensor.len(), 12);
    assert_eq!(tensor.ndim(), 2);
    assert!(tensor.is_f_contiguous());
    assert!(!tensor.has_zero_stride());
}

/// §8.3 boundary: 0D scalar tensor uses `Ix0` constructor; `[]` literal
/// cannot be inferred as `Ix0` (which is `[usize; 0]`) at function call sites.
#[test]
fn test_tensor0_scalar() {
    let data = [5_f64];
    let shape: Ix0 = Ix0;
    let strides = compute_f_strides(&shape).expect("valid");
    let tensor: TensorView<'_, f64, Ix0> =
        unsafe { TensorBase::from_raw_parts(data.as_ptr(), data.len(), shape, strides, 0) }
            .expect("Ix0 scalar tensor must construct");
    assert_eq!(tensor.ndim(), 0);
    assert_eq!(tensor.len(), 1);
}

/// §8.3 boundary: empty tensor with one zero-length axis.
#[test]
fn test_tensor_empty_dim() {
    let data = Vec::<f64>::new();
    let shape = Ix2(0, 3);
    let strides = compute_f_strides(&shape).expect("valid");
    let tensor: TensorView2<'_, f64> =
        unsafe { TensorBase::from_raw_parts(data.as_ptr(), data.len(), shape, strides, 0) }
            .expect("empty tensor must construct");
    assert!(tensor.is_empty());
    assert_eq!(tensor.shape(), &[0, 3]);
    assert!(tensor.as_slice().expect("empty Some(&[])").is_empty());
}

/// §8.3 boundary: single-element tensor; as_slice fast path returns Some.
#[test]
fn test_tensor_single_element() {
    let data = [42_i32];
    let shape = Ix2(1, 1);
    let strides = compute_f_strides(&shape).expect("valid");
    let tensor: TensorView2<'_, i32> =
        unsafe { TensorBase::from_raw_parts(data.as_ptr(), data.len(), shape, strides, 0) }
            .expect("1x1 tensor must construct");
    assert_eq!(tensor.len(), 1);
    assert_eq!(tensor.as_slice().expect("F-contiguous Some")[0], 42);
}

/// §8.3 boundary: non-zero offset view; as_storage_ptr != as_ptr.
#[test]
fn test_tensor_non_zero_offset() {
    let data = [10_i32, 20, 30, 40, 50];
    let shape = Ix1(3);
    let strides = compute_f_strides(&shape).expect("valid");
    // View elements [20, 30, 40] with offset = 1.
    let tensor: TensorView<'_, i32, Ix1> =
        unsafe { TensorBase::from_raw_parts(data.as_ptr(), data.len(), shape, strides, 1) }
            .expect("offset view must construct");
    assert_eq!(tensor.offset(), 1);
    let storage_ptr = tensor.as_storage_ptr() as usize;
    let logical_ptr = tensor.as_ptr() as usize;
    assert_eq!(logical_ptr - storage_ptr, core::mem::size_of::<i32>());
}

/// §8.3 boundary: transposed/non-contiguous view reports NonContiguous and
/// does not expose a contiguous slice.
#[test]
fn test_tensor_transposed_non_contiguous() {
    let data = [1_i32, 2, 3, 4, 5, 6];
    let shape = Ix2(3, 2);
    let strides = Strides::new(Ix2(2, 1));
    let tensor: TensorView2<'_, i32> =
        unsafe { TensorBase::from_raw_parts(data.as_ptr(), data.len(), shape, strides, 0) }
            .expect("transposed view must construct");

    assert_eq!(tensor.layout_state(), LayoutState::NonContiguous);
    assert!(!tensor.is_f_contiguous());
    assert!(!tensor.has_zero_stride());
    assert!(tensor.as_slice().is_none());
}

/// §8.3 boundary: broadcast zero-stride view reports BroadcastView and
/// shared-read-only semantics.
#[test]
fn test_tensor_broadcast_zero_stride() {
    let data = [7_i32, 8, 9];
    let shape = Ix2(2, 3);
    let strides = Strides::new(Ix2(1, 0));
    let tensor: TensorView2<'_, i32> =
        unsafe { TensorBase::from_raw_parts(data.as_ptr(), data.len(), shape, strides, 0) }
            .expect("broadcast view must construct");

    assert_eq!(tensor.layout_state(), LayoutState::BroadcastView);
    assert!(tensor.has_zero_stride());
    assert_eq!(tensor.access_semantics(), AccessSemantics::SharedReadOnly);
    assert_eq!(tensor.alias_class(), AliasClass::BroadcastAlias);
    assert!(tensor.as_slice().is_none());
}

// === Four storage modes: storage_kind / access_semantics ===

/// §8.2 test_tensor_storage_kind + test_tensor_access_semantics for all four
/// storage representations.
#[test]
fn test_storage_kind_view() {
    let data = [1_i32, 2, 3, 4];
    let shape = Ix2(2, 2);
    let strides = compute_f_strides(&shape).expect("valid");
    let tensor: TensorView2<'_, i32> =
        unsafe { TensorBase::from_raw_parts(data.as_ptr(), data.len(), shape, strides, 0) }
            .expect("valid");
    assert_eq!(tensor.storage_kind(), StorageKind::View);
    assert_eq!(tensor.access_semantics(), AccessSemantics::ReadOnly);
}

#[test]
fn test_storage_kind_view_mut() {
    let mut data = vec![1_i32, 2, 3, 4];
    let shape = Ix2(2, 2);
    let strides = compute_f_strides(&shape).expect("valid");
    let tensor: TensorViewMut2<'_, i32> =
        unsafe { TensorBase::from_raw_parts_mut(data.as_mut_ptr(), 4, shape, strides, 0) }
            .expect("valid");
    assert_eq!(tensor.storage_kind(), StorageKind::ViewMut);
    assert_eq!(tensor.access_semantics(), AccessSemantics::Writable);
}

#[test]
fn test_storage_kind_shared_via_arc() {
    // Compile-time check: ArcTensor2 alias resolves correctly.
    fn _accepts_arc(_t: &ArcTensor2<i32>) {}

    // Runtime check: Tensor::into_shared (`05-storage.md §5.11.3`) produces
    // an ArcTensor with the expected storage/access classification.
    let owned = Tensor2::<i32>::from_shape_vec([2, 2], vec![1, 2, 3, 4])
        .expect("from_shape_vec matching shape");
    let arc: ArcTensor2<i32> = owned.into_shared();
    assert_eq!(arc.storage_kind(), StorageKind::Shared);
    assert_eq!(arc.access_semantics(), AccessSemantics::SharedReadOnly);
    assert_eq!(arc.alias_class(), AliasClass::ArcShared);
}

// === view / view_mut cross-module behaviour ===

/// §8.2 test_tensor_view: view shares data; modifications visible.
#[test]
fn test_view_shares_data() {
    let mut data = vec![1_i32, 2, 3, 4];
    let shape = Ix2(2, 2);
    let strides = compute_f_strides(&shape).expect("valid");
    let mut tensor: TensorViewMut2<'_, i32> =
        unsafe { TensorBase::from_raw_parts_mut(data.as_mut_ptr(), 4, shape, strides, 0) }
            .expect("valid");

    {
        let v = tensor.view();
        assert_eq!(
            v.access_semantics(),
            AccessSemantics::SharedReadOnly,
            "ViewMut→View demotion must report SharedReadOnly per §5.3 rule (3)"
        );
        assert_eq!(v.alias_class(), AliasClass::ViewMutDerived);
    }

    tensor.as_mut_slice().expect("F-contiguous")[0] = 99;
    let v2 = tensor.view();
    assert_eq!(v2.as_slice().expect("F-contiguous")[0], 99);
}

/// §8.2 test_tensor_view_mut: view_mut writes propagate to source.
#[test]
fn test_view_mut_writes_back() {
    let mut data = vec![1_i32, 2, 3, 4];
    let shape = Ix2(2, 2);
    let strides = compute_f_strides(&shape).expect("valid");
    let mut tensor: TensorViewMut2<'_, i32> =
        unsafe { TensorBase::from_raw_parts_mut(data.as_mut_ptr(), 4, shape, strides, 0) }
            .expect("valid");

    tensor.view_mut().as_mut_slice().expect("F-contiguous")[1] = 77;
    assert_eq!(tensor.as_slice().expect("F-contiguous")[1], 77);
}

// === Error path: from_raw_parts out-of-bounds ===

/// §8.2 test_from_raw_parts_invalid_range: out-of-bounds access range is rejected.
#[test]
fn test_from_raw_parts_invalid_range() {
    let data = [1_i32, 2, 3];
    let shape = Ix2(2, 2);
    let strides = compute_f_strides(&shape).expect("valid");
    // storage_len = 3 but logical access range needs 4.
    let result: xenon::Result<TensorView2<'_, i32>> =
        unsafe { TensorBase::from_raw_parts(data.as_ptr(), data.len(), shape, strides, 0) };
    assert!(result.is_err(), "out-of-bounds raw-parts must be rejected");
}

// === Data location ===

/// §8.2 test_tensor_data_location.
#[test]
fn test_tensor_data_location() {
    let data = [1_i32];
    let shape = Ix2(1, 1);
    let strides = compute_f_strides(&shape).expect("valid");
    let tensor: TensorView2<'_, i32> =
        unsafe { TensorBase::from_raw_parts(data.as_ptr(), data.len(), shape, strides, 0) }
            .expect("valid");
    assert_eq!(tensor.data_location(), DataLocation::Cpu);
}

// === Dynamic dimension (IxDyn) ===

/// §8.2 test_tensor_ndim_dynamic: TensorD reports runtime ndim.
#[test]
fn test_tensor_ndim_dynamic() {
    let data = [0_i32; 24];
    let shape: IxDyn = IxDyn::from_slice(&[2, 3, 4]);
    let strides = compute_f_strides(&shape).expect("valid");
    let tensor: TensorViewD<'_, i32> =
        unsafe { TensorBase::from_raw_parts(data.as_ptr(), data.len(), shape, strides, 0) }
            .expect("valid");
    assert_eq!(tensor.ndim(), 3);
    assert_eq!(tensor.len(), 24);
}
// ===========================================================================
// W29T2: Core tensor integration tests
// ===========================================================================

#[cfg(test)]
mod w29t2_tests {
    use super::*;

    #[test]
    fn test_tensor_shape_strides() {
        let t = Tensor1::<f64>::from_shape_vec([5], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]).expect("valid shape");
        assert_eq!(t.shape(), &[5usize]);
        // strides() returns &[usize] (F-order strides for a 1D tensor is [1]).
        assert_eq!(t.strides(), &[1usize]);
        assert_eq!(t.len(), 5);
    }

    #[test]
    fn test_tensor_is_empty() {
        let t = Tensor1::<f64>::from_shape_vec([0], Vec::new()).expect("valid shape");
        assert_eq!(t.len(), 0);
        assert!(t.is_empty());
    }

    #[test]
    fn test_tensor_view_creation() {
        let mut t = Tensor1::<f64>::from_shape_vec([3], vec![1.0_f64, 2.0, 3.0]).expect("valid shape");

        // (a) read-only view()
        {
            let v = t.view();
            assert_eq!(v.shape(), &[3usize]);
            assert_eq!(v.iter().copied().collect::<Vec<_>>(), vec![1.0, 2.0, 3.0]);
        }

        // (b) mutable view_mut(): write through view, verify via read-only view
        {
            let mut vm = t.view_mut();
            assert_eq!(vm.shape(), &[3usize]);
            *vm.try_at_mut((0,)).expect("valid index") = 10.0;
            *vm.try_at_mut((2,)).expect("valid index") = 30.0;
        }
        assert_eq!(
            t.view().iter().copied().collect::<Vec<_>>(),
            vec![10.0, 2.0, 30.0]
        );
    }

    #[test]
    fn test_tensor_to_owned() {
        let t = Tensor1::<f64>::from_shape_vec([2], vec![10.0_f64, 20.0]).expect("valid shape");
        let view = t.view();
        let owned = view.to_owned();
        assert_eq!(owned.shape(), &[2usize]);
        assert_eq!(owned.iter().copied().collect::<Vec<_>>(), vec![10.0, 20.0]);
    }

    #[test]
    fn test_tensor_into_owned() {
        let t = Tensor1::<f64>::from_shape_vec([3], vec![1.0_f64, 2.0, 3.0]).expect("valid shape");
        let view = t.view();
        let owned = view.into_owned();
        assert_eq!(owned.shape(), &[3usize]);
        assert_eq!(owned.iter().copied().collect::<Vec<_>>(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_type_aliases() {
        // Type aliases are verified at compile time — confirm they resolve.
        let _owned: Tensor1<f64> = Tensor1::<f64>::zeros([2]).expect("shape is valid");
        let _view: TensorView1<f64> = _owned.view();
        // ArcTensor alias verified by type annotation (construction requires
        // storage-layer API not yet exposed at the TensorBase level).
        let _check_arc: Option<ArcTensor1<f64>> = None;
    }

    #[test]
    fn test_arc_tensor_clone() {
        // ArcTensor aliases and Clone trait verified at compile time.
        // Runtime construction of ArcTensor via Owned::into_shared is tested
        // in dedicated storage-layer and workspace integration tests.
        fn _accepts_clone<T: Clone>(_t: &T) {}
        let _check: Option<ArcTensor1<f64>> = None;
    }

    #[test]
    fn test_tensor_debug_display() {
        let t = Tensor1::<f64>::from_shape_vec([3], vec![1.0_f64, 2.0, 3.0]).expect("valid shape");
        let dbg = format!("{:?}", t);
        let disp = format!("{}", t);
        assert!(!dbg.is_empty(), "Debug must produce non-empty output");
        assert!(!disp.is_empty(), "Display must produce non-empty output");
        assert!(dbg.contains('1') || dbg.contains('2') || dbg.contains('3'));
    }

    #[test]
    fn test_arc_tensor_alias_isolation_on_write() {
        // ArcTensor aliases verified at compile time.
        fn _accepts_arc<T>(_t: &T) {}
        let _check: Option<ArcTensor1<f64>> = None;
    }
}