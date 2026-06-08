//! Public API implementations for matrix dot product.
//!
//! Defines the method-style `TensorBase::dot()` API, which delegates to
//! the internal `dot_impl` free function in `dot.rs`.

use crate::error::XenonError;
use crate::dimension::Dimension;
use crate::element::Numeric;
use crate::storage::Storage;
use crate::tensor::TensorBase;

use super::dot_impl;

// --- TensorBase::dot method -------------------------------------------------

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Numeric + Copy + 'static + Send + Sync,
{
    /// Vector dot product of two 1‑dimensional tensors.
    ///
    /// Delegates to `dot_impl`.
    ///
    /// # Errors
    ///
    /// Returns `XenonError::InvalidArgument` when either tensor is not
    /// 1‑dimensional. Returns `XenonError::ShapeMismatch` when the two
    /// tensors have different element counts.
    pub fn dot<S2, D2>(&self, other: &TensorBase<S2, D2>) -> Result<A, XenonError>
    where
        S2: Storage<Elem = A>,
        D2: Dimension,
    {
        dot_impl(self, other)
    }
}

#[cfg(test)]
mod tests {
    use crate::dimension::Ix1;
    use crate::tensor::Tensor1;

    use super::dot_impl;

    /// `TensorBase::dot` produces the same result as the free function
    /// `dot_impl` for a basic integer dot product.
    #[test]
    fn test_dot_basic() {
        let a = Tensor1::from_shape_vec(Ix1(3), vec![1_i32, 2, 3])
            .expect("valid construction");
        let b = Tensor1::from_shape_vec(Ix1(3), vec![4_i32, 5, 6])
            .expect("valid construction");
        assert_eq!(dot_impl(&a, &b).expect("valid construction"), 32_i32);
        assert_eq!(a.dot(&b).expect("valid construction"), 32_i32);
    }
}
