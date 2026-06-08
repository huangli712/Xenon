//! Public API implementations for matrix dot product.
//!
//! This file contains the `impl TensorBase` block that defines the
//! method-style `dot()` API, delegating to the free function in `dot.rs`.

use crate::dimension::Dimension;
use crate::element::Numeric;
use crate::error::XenonError;
use crate::storage::Storage;
use crate::tensor::TensorBase;

// ── TensorBase::dot method ──

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Numeric + Copy + 'static + Send + Sync,
{
    /// Stable method-style API; semantically equivalent to
    /// `matrix::dot(self, other)`. See 12-matrix §5.1.
    ///
    /// # Errors
    ///
    /// Returns `XenonError::InvalidArgument` when either tensor is not
    /// 1-dimensional. Returns `XenonError::ShapeMismatch` when the two
    /// tensors have different element counts.
    pub fn dot<S2, D2>(&self, other: &TensorBase<S2, D2>) -> Result<A, XenonError>
    where
        S2: Storage<Elem = A>,
        D2: Dimension,
    {
        crate::matrix::dot(self, other)
    }
}

// ── Unit tests ──

#[cfg(test)]
mod tests {
    use crate::dimension::Ix1;
    use crate::matrix::dot;
    use crate::tensor::Tensor1;

    // W17T3
    #[test]
    fn test_dot_basic() {
        let a = Tensor1::from_shape_vec(Ix1(3), vec![1_i32, 2, 3]).expect("valid construction");
        let b = Tensor1::from_shape_vec(Ix1(3), vec![4_i32, 5, 6]).expect("valid construction");
        assert_eq!(dot(&a, &b).expect("valid construction"), 32_i32);
        assert_eq!(a.dot(&b).expect("valid construction"), 32_i32);
    }
}
