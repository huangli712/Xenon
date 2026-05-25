//! Tensor access paths: `try_at` / `get` / `get_unchecked` and mutable variants.
//!
//! Implemented in W21T3 (read) and W21T5 (mutable). See `design/17-indexing.md §5.2`.

use crate::dimension::Dimension;
use crate::error::{InvalidLayoutReason, Result, StorageKindTag, XenonError};
use crate::index::NdIndex;
use crate::storage::Storage;
use crate::tensor::TensorBase;

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// Canonical safe read entry point — accepts any `NdIndex<D>` (tuples,
    /// `&[usize]` for `D == IxDyn`).
    ///
    /// # Errors
    ///
    /// Per `17-indexing §5.2`:
    /// - rank mismatch → `XenonError::DimensionMismatch`
    /// - per-axis out of bounds → `XenonError::IndexOutOfBounds`
    /// - offset arithmetic overflow → `XenonError::InvalidLayout`
    pub fn try_at<I>(&self, index: I) -> Result<&A>
    where
        I: NdIndex<D>,
    {
        let offset = index.index_checked(&self.shape, &self.strides)?;
        // SAFETY: index_checked verified bounds and checked-offset arithmetic.
        Ok(unsafe { self.storage.get_unchecked(self.offset() + offset) })
    }

    /// Convenience wrapper accepting `&[usize]`. Independent of `try_at`'s
    /// trait dispatch path — see `17-indexing §5.2` line 280 for rationale.
    ///
    /// # Errors
    ///
    /// Per `17-indexing §5.2`:
    /// - rank mismatch (`index.len() != self.ndim()`) → `XenonError::DimensionMismatch`
    /// - per-axis out of bounds (`index[i] >= shape[i]`) → `XenonError::IndexOutOfBounds`
    /// - `strides[i] * index[i]` or the accumulator overflows `usize`
    ///   → `XenonError::InvalidLayout { reason: AccessRangeExceedsStorage }`
    pub fn get(&self, index: &[usize]) -> Result<&A> {
        let shape = self.shape();
        let strides = self.strides();
        // Rank mismatch → DimensionMismatch per 17-indexing §5.2 line 280.
        if index.len() != shape.len() {
            return Err(XenonError::DimensionMismatch {
                operation: "TensorBase::get".into(),
                expected: shape.len(),
                actual: index.len(),
            });
        }
        let mut offset = 0usize;
        for (axis, ((&idx, &extent), &stride)) in index.iter().zip(shape).zip(strides).enumerate() {
            if idx >= extent {
                return Err(XenonError::IndexOutOfBounds {
                    operation: "TensorBase::get".into(),
                    attempted_index: index.to_vec(),
                    axis,
                    shape: shape.to_vec(),
                });
            }
            // storage_kind / storage_len placeholders — see W21T2 convention.
            // Final form depends on W7's StorageKindTag API.
            let term = idx
                .checked_mul(stride)
                .ok_or_else(|| XenonError::InvalidLayout {
                    operation: "TensorBase::get".into(),
                    storage_kind: StorageKindTag::View, // placeholder; W7-dependent
                    shape: shape.to_vec(),
                    strides: strides.to_vec(),
                    offset,
                    storage_len: 0, // placeholder; W7-dependent
                    reason: InvalidLayoutReason::AccessRangeExceedsStorage,
                })?;
            offset = offset
                .checked_add(term)
                .ok_or_else(|| XenonError::InvalidLayout {
                    operation: "TensorBase::get".into(),
                    storage_kind: StorageKindTag::View, // placeholder; W7-dependent
                    shape: shape.to_vec(),
                    strides: strides.to_vec(),
                    offset,
                    storage_len: 0, // placeholder; W7-dependent
                    reason: InvalidLayoutReason::AccessRangeExceedsStorage,
                })?;
        }
        // SAFETY: rank and per-axis bounds verified above; offset computed
        // with checked arithmetic.
        Ok(unsafe { self.storage.get_unchecked(self.offset() + offset) })
    }

    /// Unsafe dual of `get`. Signature is `&[usize]` per `17-indexing §5.2`
    /// line 251 — NOT generic over `NdIndex` (that would duplicate `try_at`).
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// - `index.len() == self.ndim()`
    /// - each `index[i] < shape[i]`
    /// - resulting offset does not overflow `usize`
    pub unsafe fn get_unchecked(&self, index: &[usize]) -> &A {
        let strides = self.strides();
        let mut offset = 0usize;
        for (&idx, &stride) in index.iter().zip(strides) {
            // SAFETY: caller's # Safety contract guarantees no overflow.
            debug_assert!(idx.checked_mul(stride).is_some());
            let term = unsafe { idx.unchecked_mul(stride) };
            debug_assert!(offset.checked_add(term).is_some());
            offset = unsafe { offset.unchecked_add(term) };
        }
        debug_assert!(self.offset().checked_add(offset).is_some());
        // SAFETY: caller guarantees total offset is within bounds.
        unsafe {
            self.storage
                .get_unchecked(self.offset().unchecked_add(offset))
        }
    }
}

// ── Mutable access (W21T5) ──

use crate::storage::StorageMut;

impl<S, D, A> TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    D: Dimension,
{
    /// Mutable dual of `try_at`. Gated on `StorageMut`.
    ///
    /// # Errors
    ///
    /// Per `17-indexing §5.2`:
    /// - rank mismatch → `XenonError::DimensionMismatch`
    /// - per-axis out of bounds → `XenonError::IndexOutOfBounds`
    /// - offset arithmetic overflow → `XenonError::InvalidLayout`
    pub fn try_at_mut<I>(&mut self, index: I) -> Result<&mut A>
    where
        I: NdIndex<D>,
    {
        let offset = index.index_checked(&self.shape, &self.strides)?;
        // SAFETY: index_checked verified bounds + checked-offset arithmetic.
        Ok(unsafe { self.storage.get_unchecked_mut(self.offset() + offset) })
    }

    /// Mutable dual of `get`. Independent of `try_at_mut` trait dispatch.
    ///
    /// # Errors
    ///
    /// Per `17-indexing §5.2`:
    /// - rank mismatch (`index.len() != self.ndim()`) → `XenonError::DimensionMismatch`
    /// - per-axis out of bounds (`index[i] >= shape[i]`) → `XenonError::IndexOutOfBounds`
    /// - `strides[i] * index[i]` or the accumulator overflows `usize`
    ///   → `XenonError::InvalidLayout { reason: AccessRangeExceedsStorage }`
    pub fn get_mut(&mut self, index: &[usize]) -> Result<&mut A> {
        let (shape, strides_vec, off) = {
            let shape = self.shape().to_vec();
            let strides = self.strides().to_vec();
            (shape, strides, self.offset())
        };
        if index.len() != shape.len() {
            return Err(XenonError::DimensionMismatch {
                operation: "TensorBase::get_mut".into(),
                expected: shape.len(),
                actual: index.len(),
            });
        }
        let mut offset = 0usize;
        for (axis, ((&idx, &extent), &stride)) in
            index.iter().zip(&shape).zip(&strides_vec).enumerate()
        {
            if idx >= extent {
                return Err(XenonError::IndexOutOfBounds {
                    operation: "TensorBase::get_mut".into(),
                    attempted_index: index.to_vec(),
                    axis,
                    shape: shape.clone(),
                });
            }
            let term = idx
                .checked_mul(stride)
                .ok_or_else(|| XenonError::InvalidLayout {
                    operation: "TensorBase::get_mut".into(),
                    storage_kind: StorageKindTag::View,
                    shape: shape.clone(),
                    strides: strides_vec.clone(),
                    offset,
                    storage_len: 0,
                    reason: InvalidLayoutReason::AccessRangeExceedsStorage,
                })?;
            offset = offset
                .checked_add(term)
                .ok_or_else(|| XenonError::InvalidLayout {
                    operation: "TensorBase::get_mut".into(),
                    storage_kind: StorageKindTag::View,
                    shape: shape.clone(),
                    strides: strides_vec.clone(),
                    offset,
                    storage_len: 0,
                    reason: InvalidLayoutReason::AccessRangeExceedsStorage,
                })?;
        }
        // SAFETY: rank + per-axis bounds verified above; offset via checked arithmetic.
        Ok(unsafe { self.storage.get_unchecked_mut(off + offset) })
    }

    /// Unsafe dual of `get_mut`. Signature is `&[usize]`.
    ///
    /// # Safety
    /// Caller must ensure: rank match, per-axis bounds, no offset overflow,
    /// and exclusive mutable access.
    pub unsafe fn get_unchecked_mut(&mut self, index: &[usize]) -> &mut A {
        let off = self.offset();
        let mut offset = 0usize;
        {
            let strides = self.strides();
            for (&idx, &stride) in index.iter().zip(strides) {
                debug_assert!(idx.checked_mul(stride).is_some());
                let term = unsafe { idx.unchecked_mul(stride) };
                debug_assert!(offset.checked_add(term).is_some());
                offset = unsafe { offset.unchecked_add(term) };
            }
        }
        debug_assert!(off.checked_add(offset).is_some());
        unsafe { self.storage.get_unchecked_mut(off.unchecked_add(offset)) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::Ix2;
    use crate::tensor::Tensor;

    fn tensor_ix2<A: crate::element::Element>(data: Vec<A>, shape: Ix2) -> Tensor<A, Ix2> {
        unsafe { Tensor::from_raw_vec_unchecked(data, shape) }
    }

    #[test]
    fn test_try_at_2d() {
        let tensor = tensor_ix2(vec![1, 2, 3, 4, 5, 6], Ix2(2, 3));
        assert_eq!(*tensor.try_at((1usize, 2usize)).expect("valid index"), 6);
    }

    #[test]
    fn test_try_at_out_of_bounds() {
        let tensor = tensor_ix2(vec![1, 2, 3, 4, 5, 6], Ix2(2, 3));
        let err = tensor.try_at((2usize, 0usize)).expect_err("out of bounds");
        assert!(matches!(err, XenonError::IndexOutOfBounds { axis: 0, .. }));
    }

    #[test]
    fn test_get_returns_index_out_of_bounds() {
        let tensor = tensor_ix2(vec![1, 2, 3, 4], Ix2(2, 2));
        let err = tensor.get(&[2, 0]).expect_err("out of bounds");
        assert!(matches!(err, XenonError::IndexOutOfBounds { .. }));
    }

    #[test]
    fn test_get_rank_mismatch_is_dimension_mismatch() {
        let tensor = tensor_ix2(vec![1, 2, 3, 4], Ix2(2, 2));
        let err = tensor.get(&[0, 0, 0]).expect_err("rank mismatch");
        assert!(matches!(
            err,
            XenonError::DimensionMismatch {
                expected: 2,
                actual: 3,
                ..
            }
        ));
    }
}

#[cfg(test)]
mod mut_tests {
    use super::*;
    use crate::dimension::Ix2;
    use crate::tensor::Tensor;

    fn tensor_ix2_mut<A: crate::element::Element>(data: Vec<A>, shape: Ix2) -> Tensor<A, Ix2> {
        unsafe { Tensor::from_raw_vec_unchecked(data, shape) }
    }

    #[test]
    fn test_try_at_mut_requires_storage_mut() {
        let mut tensor = tensor_ix2_mut(vec![1, 2, 3, 4], Ix2(2, 2));
        *tensor
            .try_at_mut((1usize, 1usize))
            .expect("valid mut index") = 9;
        assert_eq!(*tensor.try_at((1usize, 1usize)).expect("valid index"), 9);
    }

    #[test]
    fn test_get_mut_out_of_bounds() {
        let mut tensor = tensor_ix2_mut(vec![1, 2, 3, 4], Ix2(2, 2));
        let err = tensor.get_mut(&[2, 0]).expect_err("out of bounds");
        assert!(matches!(err, XenonError::IndexOutOfBounds { .. }));
    }

    #[test]
    fn test_get_mut_rank_mismatch_is_dimension_mismatch() {
        let mut tensor = tensor_ix2_mut(vec![1, 2, 3, 4], Ix2(2, 2));
        let err = tensor.get_mut(&[0, 0, 0]).expect_err("rank mismatch");
        assert!(matches!(
            err,
            XenonError::DimensionMismatch {
                expected: 2,
                actual: 3,
                ..
            }
        ));
    }
}
