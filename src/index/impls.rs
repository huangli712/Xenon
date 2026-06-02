//! Inherent [`TensorBase`] methods for element access and slicing.
//!
//! Provides the safe entry points `try_at`, `get`, `slice`, their mutable
//! counterparts (`try_at_mut`, `get_mut`, `get_unchecked`), and the
//! unsafe unchecked variants (`get_unchecked`, `get_unchecked_mut`).

use crate::error::{InvalidArgumentKind, InvalidLayoutReason, Result, StorageKindTag, XenonError};
use crate::dimension::Dimension;
use crate::layout::{Strides, compute_layout_flags};
use crate::storage::{Storage, ViewRepr};
use crate::tensor::{StorageKind, TensorBase, TensorView};

use super::{NdIndex, SliceInfo, SliceInfoElem};

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// Canonical safe read entry point — accepts any [`NdIndex`]`<D>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use xenon::dimension::Ix2;
    /// use xenon::tensor::Tensor;
    ///
    /// let tensor = unsafe { Tensor::from_raw_vec_unchecked(vec![1,2,3,4,5,6], Ix2(2,3)) };
    /// let val = tensor.try_at((1usize, 2usize)).unwrap();
    /// assert_eq!(*val, 6);
    /// ```
    ///
    /// # Errors
    ///
    /// - rank mismatch → [`XenonError::DimensionMismatch`]
    /// - per-axis out of bounds → [`XenonError::IndexOutOfBounds`]
    /// - offset arithmetic overflow → [`XenonError::InvalidLayout`]
    pub fn try_at<I>(&self, index: I) -> Result<&A>
    where
        I: NdIndex<D>,
    {
        let offset = index.index_checked(&self.shape, &self.strides)?;
        // SAFETY: index_checked verified bounds and checked-offset arithmetic.
        Ok(unsafe { self.storage.get_unchecked(self.offset() + offset) })
    }

    /// Element access by slice reference — a direct path independent of
    /// `try_at`'s trait-dispatch mechanism.
    ///
    /// # Errors
    ///
    /// - rank mismatch (`index.len() != self.ndim()`) → [`XenonError::DimensionMismatch`]
    /// - per-axis out of bounds (`index[i] >= shape[i]`) → [`XenonError::IndexOutOfBounds`]
    /// - offset arithmetic overflow → [`XenonError::InvalidLayout`]
    pub fn get(&self, index: &[usize]) -> Result<&A> {
        let shape = self.shape();
        let strides = self.strides();

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
            let term = idx
                .checked_mul(stride)
                .ok_or_else(|| XenonError::InvalidLayout {
                    operation: "TensorBase::get".into(),
                    storage_kind: StorageKindTag::View,
                    shape: shape.to_vec(),
                    strides: strides.to_vec(),
                    offset,
                    storage_len: 0,
                    reason: InvalidLayoutReason::AccessRangeExceedsStorage,
                })?;
            offset = offset
                .checked_add(term)
                .ok_or_else(|| XenonError::InvalidLayout {
                    operation: "TensorBase::get".into(),
                    storage_kind: StorageKindTag::View,
                    shape: shape.to_vec(),
                    strides: strides.to_vec(),
                    offset,
                    storage_len: 0,
                    reason: InvalidLayoutReason::AccessRangeExceedsStorage,
                })?;
        }
        // SAFETY: rank and per-axis bounds verified above; offset computed
        // with checked arithmetic.
        Ok(unsafe { self.storage.get_unchecked(self.offset() + offset) })
    }

    /// Unsafe dual of [`get`](Self::get). Accepts `&[usize]`.
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

use crate::storage::StorageMut;

impl<S, D, A> TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    D: Dimension,
{
    /// Mutable dual of [`try_at`](Self::try_at). Gated on [`StorageMut`].
    ///
    /// # Errors
    ///
    /// - rank mismatch → [`XenonError::DimensionMismatch`]
    /// - per-axis out of bounds → [`XenonError::IndexOutOfBounds`]
    /// - offset arithmetic overflow → [`XenonError::InvalidLayout`]
    pub fn try_at_mut<I>(&mut self, index: I) -> Result<&mut A>
    where
        I: NdIndex<D>,
    {
        let offset = index.index_checked(&self.shape, &self.strides)?;
        // SAFETY: index_checked verified bounds + checked-offset arithmetic.
        Ok(unsafe { self.storage.get_unchecked_mut(self.offset() + offset) })
    }

    /// Mutable dual of [`get`](Self::get). Independent of `try_at_mut`'s
    /// trait-dispatch path.
    ///
    /// # Errors
    ///
    /// - rank mismatch (`index.len() != self.ndim()`) → [`XenonError::DimensionMismatch`]
    /// - per-axis out of bounds (`index[i] >= shape[i]`) → [`XenonError::IndexOutOfBounds`]
    /// - offset arithmetic overflow → [`XenonError::InvalidLayout`]
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

    /// Unsafe dual of [`get_mut`](Self::get_mut). Accepts `&[usize]`.
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

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A> + crate::tensor::StorageSemantics,
    D: Dimension,
{
    /// Creates a read-only sliced view of the tensor.
    ///
    /// The resulting [`TensorView`] shares the underlying storage and
    /// exposes a subset of axes defined by the [`SliceInfo`] descriptor.
    ///
    /// # Errors
    ///
    /// - [`XenonError::IndexOutOfBounds`] — a [`SliceInfoElem::Index`]`(idx)` has
    ///   `idx >= self.shape()[axis]`.
    /// - [`XenonError::InvalidArgument`] with
    ///   [`RangeOutOfBounds`] — a [`SliceInfoElem::Range`]
    ///   has `end > self.shape()[axis]`.
    /// - [`XenonError::InvalidLayout`] — offset arithmetic overflows `usize`.
    /// - [`XenonError::DimensionMismatch`] — the output shape's rank does not
    ///   match the static rank of `I` (fixed-rank `I` only).
    ///
    /// [`RangeOutOfBounds`]: InvalidArgumentKind::RangeOutOfBounds
    pub fn slice<I>(&self, info: SliceInfo<I, D>) -> Result<TensorView<'_, A, I>>
    where
        I: Dimension,
    {
        debug_assert_eq!(info.input_dim().ndim(), self.ndim());

        let shape = self.shape();
        let strides = self.strides();
        let mut out_shape = Vec::with_capacity(info.output_dim().ndim());
        let mut out_strides = Vec::with_capacity(info.output_dim().ndim());
        let mut slice_delta = 0usize;

        let overflow_err = |partial_offset: usize| XenonError::InvalidLayout {
            operation: "TensorBase::slice".into(),
            storage_kind: StorageKindTag::View,
            shape: shape.to_vec(),
            strides: strides.to_vec(),
            offset: partial_offset,
            storage_len: 0,
            reason: InvalidLayoutReason::AccessRangeExceedsStorage,
        };

        for (axis, elem) in info.indices().iter().enumerate() {
            match elem {
                SliceInfoElem::Index(idx) => {
                    if idx >= shape[axis] {
                        return Err(XenonError::IndexOutOfBounds {
                            operation: "TensorBase::slice".into(),
                            attempted_index: vec![idx],
                            axis,
                            shape: shape.to_vec(),
                        });
                    }
                    let term = idx
                        .checked_mul(strides[axis])
                        .ok_or_else(|| overflow_err(slice_delta))?;
                    slice_delta = slice_delta
                        .checked_add(term)
                        .ok_or_else(|| overflow_err(slice_delta))?;
                },
                SliceInfoElem::Range { start, end } => {
                    if end > shape[axis] {
                        return Err(XenonError::InvalidArgument {
                            operation: "TensorBase::slice".into(),
                            kind: InvalidArgumentKind::RangeOutOfBounds {
                                axis,
                                axis_len: shape[axis],
                                start,
                                end,
                            },
                        });
                    }
                    let term = start
                        .checked_mul(strides[axis])
                        .ok_or_else(|| overflow_err(slice_delta))?;
                    slice_delta = slice_delta
                        .checked_add(term)
                        .ok_or_else(|| overflow_err(slice_delta))?;
                    out_shape.push(end - start);
                    out_strides.push(strides[axis]);
                },
            }
        }

        let new_offset = self
            .offset()
            .checked_add(slice_delta)
            .ok_or_else(|| overflow_err(slice_delta))?;

        let new_dim = I::try_from_slice(&out_shape)?;
        let new_strides = Strides::<I>::from_slice(&out_strides)?;

        let is_empty = out_shape.contains(&0);
        let logical_ptr: *const A = if is_empty {
            core::ptr::NonNull::<A>::dangling().as_ptr()
        } else {
            // SAFETY: slice_delta validated via per-axis bounds and
            // checked-offset arithmetic; the resulting offset lies within
            // the source's reachable storage range.
            unsafe { self.as_ptr().add(slice_delta) }
        };
        let new_flags = compute_layout_flags::<A, I>(&new_dim, &new_strides, logical_ptr);

        let derived_from_view_mut = match self.storage_kind() {
            StorageKind::ViewMut => true,
            StorageKind::View => self.derived_from_view_mut,
            _ => false,
        };

        // SAFETY: ViewRepr::from_raw_parts with valid ptr/len from storage contract.
        let view_storage: ViewRepr<'_, A> =
            unsafe { ViewRepr::from_raw_parts(self.storage.as_ptr(), self.storage.len()) };
        // SAFETY: all metadata fields validated above.
        let view = unsafe {
            TensorBase::new_unchecked(
                view_storage,
                new_dim,
                new_strides,
                new_offset,
                new_flags,
                derived_from_view_mut,
            )
        };
        Ok(view)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::{Ix0, Ix1, Ix2, IxDyn};
    use crate::index::slice::{SliceInfo, SliceInfoElem, SliceInfoIndices};
    use crate::tensor::Tensor;

    /// Build a 2D owned tensor from a `Vec` and a fixed shape.
    ///
    /// # Safety
    ///
    /// Panics if `data.len() != shape.size()` — the caller is responsible
    /// for passing consistent arguments (this is a test helper).
    fn tensor_ix2<A: crate::element::Element>(data: Vec<A>, shape: Ix2) -> Tensor<A, Ix2> {
        unsafe { Tensor::from_raw_vec_unchecked(data, shape) }
    }

    /// `try_at` with a valid 2D tuple returns the correct element.
    #[test]
    fn test_try_at_2d() {
        let tensor = tensor_ix2(vec![1, 2, 3, 4, 5, 6], Ix2(2, 3));
        assert_eq!(*tensor.try_at((1usize, 2usize)).expect("valid index"), 6);
    }

    /// `try_at` returns [`IndexOutOfBounds`] when an axis index exceeds
    /// the shape bound.
    #[test]
    fn test_try_at_out_of_bounds() {
        let tensor = tensor_ix2(vec![1, 2, 3, 4, 5, 6], Ix2(2, 3));
        let err = tensor.try_at((2usize, 0usize)).expect_err("out of bounds");
        assert!(matches!(err, XenonError::IndexOutOfBounds { axis: 0, .. }));
    }

    /// `get` returns [`IndexOutOfBounds`] for an out-of-range slice index.
    #[test]
    fn test_get_returns_index_out_of_bounds() {
        let tensor = tensor_ix2(vec![1, 2, 3, 4], Ix2(2, 2));
        let err = tensor.get(&[2, 0]).expect_err("out of bounds");
        assert!(matches!(err, XenonError::IndexOutOfBounds { .. }));
    }

    /// `get` returns [`DimensionMismatch`] when the index slice length
    /// differs from the tensor's rank.
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

    /// Build a mutable 2D owned tensor from a `Vec` and a fixed shape.
    ///
    /// # Safety
    ///
    /// Panics if `data.len() != shape.size()` — the caller is responsible
    /// for passing consistent arguments (this is a test helper).
    fn tensor_ix2_mut<A: crate::element::Element>(data: Vec<A>, shape: Ix2) -> Tensor<A, Ix2> {
        unsafe { Tensor::from_raw_vec_unchecked(data, shape) }
    }

    /// `try_at_mut` returns a mutable reference that can be written through,
    /// and the write is visible through a subsequent `try_at`.
    #[test]
    fn test_try_at_mut_requires_storage_mut() {
        let mut tensor = tensor_ix2_mut(vec![1, 2, 3, 4], Ix2(2, 2));
        *tensor
            .try_at_mut((1usize, 1usize))
            .expect("valid mut index") = 9;
        assert_eq!(*tensor.try_at((1usize, 1usize)).expect("valid index"), 9);
    }

    /// `get_mut` returns [`IndexOutOfBounds`] for an out-of-range index.
    #[test]
    fn test_get_mut_out_of_bounds() {
        let mut tensor = tensor_ix2_mut(vec![1, 2, 3, 4], Ix2(2, 2));
        let err = tensor.get_mut(&[2, 0]).expect_err("out of bounds");
        assert!(matches!(err, XenonError::IndexOutOfBounds { .. }));
    }

    /// `get_mut` returns [`DimensionMismatch`] for a rank-mismatched index.
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

    /// `slice` produces a view with the expected output shape and data.
    #[test]
    fn test_slice_layout_recomputed() {
        let tensor = tensor_ix2((0i32..20).collect(), Ix2(4, 5));
        let info = SliceInfo::new(
            SliceInfoIndices::from_vec(vec![
                SliceInfoElem::Range { start: 1, end: 4 },
                SliceInfoElem::Index(2),
            ]),
            Ix2(4, 5),
            Ix1(3),
        )
        .expect("valid slice");
        let view = tensor.slice(info).expect("valid slice");
        assert_eq!(view.shape(), &[3]);
        assert_eq!(view.as_slice(), Some(&[9, 10, 11][..]));
    }

    /// Chained `slice` calls produce the expected output shape.
    #[test]
    fn test_slice_chain() {
        let tensor = tensor_ix2((0i32..12).collect(), Ix2(3, 4));
        let info1 = SliceInfo::new(
            SliceInfoIndices::from_vec(vec![
                SliceInfoElem::Range { start: 0, end: 2 },
                SliceInfoElem::Range { start: 1, end: 3 },
            ]),
            Ix2(3, 4),
            Ix2(2, 2),
        )
        .expect("valid slice 1");
        let view1 = tensor.slice(info1).expect("valid slice 1");
        let info2 = SliceInfo::new(
            SliceInfoIndices::from_vec(vec![
                SliceInfoElem::Index(1),
                SliceInfoElem::Range { start: 0, end: 2 },
            ]),
            Ix2(2, 2),
            Ix1(2),
        )
        .expect("valid slice 2");
        let view2 = view1.slice(info2).expect("valid slice 2");
        assert_eq!(view2.shape(), &[2]);
    }

    /// `slice` works on a 7-dimensional `IxDyn` tensor,
    /// collapsing all axes via `Index(0)`.
    #[test]
    fn test_slice_high_rank_ixdyn() {
        let dyn_shape = IxDyn::from_slice(&[2, 2, 2, 2, 2, 2, 2]);
        let total: usize = dyn_shape.slice().iter().product();
        // SAFETY: shape size == data.len().
        let tensor = unsafe {
            Tensor::from_raw_vec_unchecked((0i32..total as i32).collect(), dyn_shape.clone())
        };
        let elems: Vec<SliceInfoElem> = (0..7).map(|_| SliceInfoElem::Index(0)).collect();
        let info = SliceInfo::new(
            SliceInfoIndices::from_vec(elems),
            dyn_shape,
            IxDyn::from_slice(&[]),
        )
        .expect("valid high-rank slice");
        let view = tensor.slice(info).expect("valid slice");
        assert_eq!(view.ndim(), 0);
    }

    /// `slice` with a valid descriptor at the upper edge succeeds.
    #[test]
    fn test_slice_extreme_offset_checked() {
        let tensor = tensor_ix2(vec![0i32, 1, 2, 3], Ix2(2, 2));
        let info_ok = SliceInfo::new(
            SliceInfoIndices::from_vec(vec![
                SliceInfoElem::Index(1),
                SliceInfoElem::Range { start: 0, end: 2 },
            ]),
            Ix2(2, 2),
            Ix1(2),
        )
        .expect("valid slice");
        assert!(tensor.slice(info_ok).is_ok());
    }

    /// `slice` with a large tensor (3162×3162) indexing the last element
    /// does not overflow.
    #[test]
    fn test_index_large_tensor_offset_boundary() {
        const N: usize = 3162;
        let data: Vec<i32> = (0..(N * N) as i32).collect();
        // SAFETY: shape size == data.len().
        let tensor = unsafe { Tensor::from_raw_vec_unchecked(data, Ix2(N, N)) };

        let info_end = SliceInfo::<Ix0, Ix2>::new(
            SliceInfoIndices::from_vec(vec![
                SliceInfoElem::Index(N - 1),
                SliceInfoElem::Index(N - 1),
            ]),
            Ix2(N, N),
            Ix0,
        )
        .expect("valid end slice");
        let view = tensor.slice(info_end).expect("valid slice");
        assert_eq!(view.ndim(), 0);
    }
}
