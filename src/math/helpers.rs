//! Shared element-wise traversal skeletons for the math module.
//!
//! Visibility: `pub(in crate::math)`. Consumed by `binary` / `unary` /
//! `comparison`. Never re-exported to the crate root.

use crate::dimension::Dimension;
use crate::element::{ComplexScalar, Element};
use crate::storage::Storage;
use crate::tensor::{Tensor, TensorBase};

/// Same-type unary traversal — per 11-math §6.1 lines 537-543.
///
/// Output element type equals input element type. Type-changing
/// traversal (`Complex<T> → T`) is handled by `apply_complex_to_real`.
#[inline]
pub(in crate::math) fn apply_unary<A, S, D, F>(
    input: &TensorBase<S, D>,
    mut f: F,
) -> Tensor<A, D>
where
    A: Element,
    S: Storage<Elem = A>,
    D: Dimension,
    F: FnMut(A) -> A,
{
    let mut result = Tensor::<A, D>::zeros(input.raw_dim())
        .expect("input dimension must be valid since input tensor exists");
    for (dst, src) in result.iter_mut().zip(input.iter()) {
        *dst = f(*src);
    }
    result
}

/// Type-changing traversal for `Complex<T> → T` — per 11-math §6.1 line 546.
///
/// Used by `modulus()` (W16T5). Input is a complex tensor; output is a
/// real tensor of the same shape. Not exposed publicly.
#[inline]
pub(in crate::math) fn apply_complex_to_real<A, S, D, F>(
    input: &TensorBase<S, D>,
    mut f: F,
) -> Tensor<<A as ComplexScalar>::Real, D>
where
    A: ComplexScalar,
    S: Storage<Elem = A>,
    D: Dimension,
    F: FnMut(A) -> <A as ComplexScalar>::Real,
{
    let mut result =
        Tensor::<<A as ComplexScalar>::Real, D>::zeros(input.raw_dim())
            .expect("input dimension must be valid since input tensor exists");
    for (dst, src) in result.iter_mut().zip(input.iter()) {
        *dst = f(*src);
    }
    result
}

/// Same-type unary traversal with element-index and shape context — variant
/// of `apply_unary` that propagates `(idx, &shape)` into the kernel closure.
///
/// Required by W16T3 integer monomorphizations of `abs` / `neg` / `square`
/// so that panic messages can embed `element_index` + `shape` per 11-math
/// §10 line 785–790 ("panic 信息至少包含 `operation`、`type`、`trigger`、
/// `element_index`，并在适用时附带 `shape`"). Non-integer monomorphizations
/// have no panic path; the `idx` / `shape` parameters are zero-cost dropped
/// after inlining.
///
/// Visibility: `pub(in crate::math)` — consumed by `unary.rs` (W16T3).
#[inline]
pub(in crate::math) fn apply_unary_indexed<A, S, D, F>(
    input: &TensorBase<S, D>,
    mut f: F,
) -> Tensor<A, D>
where
    A: Element,
    S: Storage<Elem = A>,
    D: Dimension,
    F: FnMut(A, usize, &[usize]) -> A,
{
    let dim = input.raw_dim();
    let shape_slice: Vec<usize> = dim.slice().to_vec();
    let mut result = Tensor::<A, D>::zeros(dim)
        .expect("input dimension must be valid since input tensor exists");
    for (idx, (dst, src)) in result.iter_mut().zip(input.iter()).enumerate() {
        *dst = f(*src, idx, &shape_slice);
    }
    result
}