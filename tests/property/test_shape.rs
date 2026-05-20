// tests/property/test_shape.rs
// Included via `tests/property/mod.rs` and `tests/property_tests.rs`.
// Cargo only auto-discovers `tests/*.rs`, not `tests/*/*.rs`.

use xenon::dimension::{Ix2, Ix3};
use xenon::element::Element;
use xenon::storage::Owned;
use xenon::tensor::TensorBase;

unsafe fn make_tensor<A: Element, D: xenon::dimension::Dimension>(
    data: Vec<A>,
    shape: D,
) -> TensorBase<Owned<A>, D> {
    // SAFETY: caller provides data with correct length matching shape.
    unsafe { TensorBase::from_raw_vec_unchecked(data, shape) }
}

unsafe fn read_at<'a, S, D, A>(tensor: &'a TensorBase<S, D>, indices: &[usize]) -> &'a A
where
    S: xenon::storage::Storage<Elem = A>,
    D: xenon::dimension::Dimension,
    A: Element,
{
    debug_assert_eq!(indices.len(), tensor.ndim());
    let strides = tensor.strides();
    let mut rel_offset: isize = 0;
    for (axis, &idx) in indices.iter().enumerate() {
        rel_offset += (idx as isize) * (strides[axis] as isize);
    }
    unsafe { &*tensor.as_ptr().offset(rel_offset) }
}

// Property §8.4: `transpose().len() == tensor.len()`
#[test]
fn test_shape_property_transpose_preserves_len() {
    let x = unsafe { make_tensor(Vec::<i32>::new(), Ix3(2, 3, 4)) };
    assert_eq!(x.transpose().len(), x.len());
}

// Property §8.4: `t.transpose().transpose()` ≡ `t`
#[test]
fn test_shape_property_transpose_involution() {
    let x = unsafe { make_tensor(vec![1, 2, 3, 4, 5, 6], Ix2(2, 3)) };
    let binding = x.transpose();
    let y = binding.transpose();
    assert_eq!(y.shape(), x.shape());
    assert_eq!(y.strides(), x.strides());
}

// Property §8.4: transpose preserves data element-wise
#[test]
fn test_shape_property_transpose_data_invariant() {
    let x = unsafe { make_tensor(vec![1, 2, 3, 4, 5, 6], Ix2(2, 3)) };
    let y = x.transpose();
    unsafe {
        for i in 0..2_usize {
            for j in 0..3_usize {
                assert_eq!(
                    *read_at(&y, &[j, i]),
                    *read_at(&x, &[i, j]),
                );
            }
        }
    }
}