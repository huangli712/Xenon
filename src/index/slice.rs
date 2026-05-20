//! Slice descriptor types and the `TensorBase::slice` method.
//!
//! `SliceInfo*` defined in W21T4; `TensorBase::slice` implemented in W21T6.
//! See `design/17-indexing.md §5.1` and `§6.3`.

use std::borrow::Cow;

use crate::dimension::Dimension;
use crate::error::{InvalidArgumentKind, Result, XenonError};

/// A single element of a slice description: either an index or a range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceInfoElem {
    /// Select a single element along the axis (axis folded in output).
    Index(usize),
    /// Select a range of elements along the axis (axis preserved in output).
    Range {
        /// Inclusive start index.
        start: usize,
        /// Exclusive end index.
        end: usize,
    },
}

/// Storage-optimized container for slice index descriptors.
///
/// Prefers a fixed-capacity inline representation when `len() <= 6`,
/// falling back to heap allocation for higher-rank `IxDyn` slices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SliceInfoIndices {
    /// Fixed-capacity inline representation; covers Ix0..Ix6 slice descriptors
    /// without heap allocation (17-indexing §5.1 line 232).
    Inline {
        /// Number of valid elements in `elems`.
        len: u8,
        /// Slice descriptors; prefix `[..len]` are always `Some(..)`.
        elems: [Option<SliceInfoElem>; 6],
    },
    /// Heap-backed fallback for IxDyn with rank > 6.
    Dynamic(Vec<SliceInfoElem>),
}

impl SliceInfoIndices {
    /// Constructor that automatically selects the optimal representation.
    pub fn from_vec(elems: Vec<SliceInfoElem>) -> Self {
        if elems.len() <= 6 {
            let mut buf: [Option<SliceInfoElem>; 6] = [None; 6];
            let len = elems.len() as u8;
            for (slot, elem) in buf.iter_mut().zip(elems) {
                *slot = Some(elem);
            }
            Self::Inline { len, elems: buf }
        } else {
            Self::Dynamic(elems)
        }
    }

    /// Inline constructor for compile-time-sized slice descriptors.
    pub fn from_array<const N: usize>(arr: [SliceInfoElem; N]) -> Self {
        if N <= 6 {
            let mut buf: [Option<SliceInfoElem>; 6] = [None; 6];
            for (slot, elem) in buf.iter_mut().zip(arr.iter()) {
                *slot = Some(*elem);
            }
            Self::Inline {
                len: N as u8,
                elems: buf,
            }
        } else {
            Self::Dynamic(arr.to_vec())
        }
    }

    /// Number of elements in this slice description.
    pub fn len(&self) -> usize {
        match self {
            Self::Inline { len, .. } => *len as usize,
            Self::Dynamic(v) => v.len(),
        }
    }

    /// Returns true if this slice description is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Yields each `SliceInfoElem` by value — `SliceInfoElem: Copy`.
    pub fn iter(&self) -> SliceInfoIter<'_> {
        SliceInfoIter {
            source: self,
            pos: 0,
        }
    }
}

/// Iterator over `SliceInfoIndices` yielding `SliceInfoElem` by value.
#[derive(Debug)]
pub struct SliceInfoIter<'a> {
    source: &'a SliceInfoIndices,
    pos: usize,
}

impl Iterator for SliceInfoIter<'_> {
    type Item = SliceInfoElem;

    fn next(&mut self) -> Option<Self::Item> {
        match self.source {
            SliceInfoIndices::Inline { len, elems } => {
                if self.pos < *len as usize {
                    let slot = &elems[self.pos];
                    debug_assert!(slot.is_some());
                    // SAFETY: prefix elems[..len] is always Some.
                    let elem = unsafe { slot.unwrap_unchecked() };
                    self.pos += 1;
                    Some(elem)
                } else {
                    None
                }
            }
            SliceInfoIndices::Dynamic(v) => {
                let elem = *v.get(self.pos)?;
                self.pos += 1;
                Some(elem)
            }
        }
    }
}

/// A validated slice description coupling indices, input dim, and output dim.
#[derive(Debug)]
pub struct SliceInfo<I: Dimension, D: Dimension> {
    indices: SliceInfoIndices,
    in_dim: D,
    out_dim: I,
}

impl<I: Dimension, D: Dimension> SliceInfo<I, D> {
    /// Constructs a `SliceInfo` with three structural checks.
    pub fn new(indices: SliceInfoIndices, in_dim: D, out_dim: I) -> Result<Self> {
        // Check 1: rank match.
        if indices.len() != in_dim.ndim() {
            return Err(XenonError::InvalidArgument {
                operation: Cow::Borrowed("SliceInfo::new"),
                kind: InvalidArgumentKind::OperationSpecific {
                    argument: Cow::Borrowed("slice"),
                    constraint: Cow::Borrowed("slice rank does not match input dimension"),
                },
            });
        }
        // Check 2: output rank == Range count.
        let ranges = indices
            .iter()
            .filter(|elem| matches!(elem, SliceInfoElem::Range { .. }))
            .count();
        if ranges != out_dim.ndim() {
            return Err(XenonError::InvalidArgument {
                operation: Cow::Borrowed("SliceInfo::new"),
                kind: InvalidArgumentKind::OperationSpecific {
                    argument: Cow::Borrowed("slice"),
                    constraint: Cow::Borrowed("slice output rank does not match Range count"),
                },
            });
        }
        // Check 3: Range start <= end.
        for (axis, elem) in indices.iter().enumerate() {
            if let SliceInfoElem::Range { start, end } = elem
                && start > end
            {
                return Err(XenonError::InvalidArgument {
                    operation: Cow::Borrowed("SliceInfo::new"),
                    kind: InvalidArgumentKind::RangeStartAfterEnd { axis, start, end },
                });
            }
        }
        Ok(Self {
            indices,
            in_dim,
            out_dim,
        })
    }

    /// Returns the slice index descriptors.
    pub fn indices(&self) -> &SliceInfoIndices {
        &self.indices
    }
    /// Returns the input dimension.
    pub fn input_dim(&self) -> &D {
        &self.in_dim
    }
    /// Returns the output dimension.
    pub fn output_dim(&self) -> &I {
        &self.out_dim
    }
}

// ── TensorBase::slice (W21T6) ──

use crate::error::{InvalidLayoutReason, StorageKindTag};
use crate::layout::{compute_layout_flags, Strides};
use crate::storage::{Storage, ViewRepr};
use crate::tensor::{StorageKind, TensorBase, TensorView};

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A> + crate::tensor::StorageSemantics,
    D: Dimension,
{
    /// Creates a read-only sliced view of the tensor (17-indexing §6.3).
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
                }
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
                }
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
    use crate::dimension::{Ix1, Ix2};

    #[test]
    fn test_slice_basic() {
        let info = SliceInfo::new(
            SliceInfoIndices::from_vec(vec![
                SliceInfoElem::Index(1),
                SliceInfoElem::Range { start: 0, end: 3 },
            ]),
            Ix2(2, 3),
            Ix1(3),
        );
        assert!(info.is_ok());
    }

    #[test]
    fn test_slice_info_rank_mismatch() {
        let err = SliceInfo::new(
            SliceInfoIndices::from_vec(vec![SliceInfoElem::Index(0)]),
            Ix2(2, 3),
            Ix1(0),
        )
        .expect_err("rank mismatch");
        assert!(matches!(
            err,
            XenonError::InvalidArgument {
                kind: InvalidArgumentKind::OperationSpecific { .. },
                ..
            }
        ));
    }

    #[test]
    fn test_slice_info_output_rank_mismatch() {
        let err = SliceInfo::new(
            SliceInfoIndices::from_vec(vec![
                SliceInfoElem::Range { start: 0, end: 2 },
                SliceInfoElem::Range { start: 0, end: 3 },
            ]),
            Ix2(2, 3),
            Ix1(2),
        )
        .expect_err("output rank mismatch");
        assert!(matches!(
            err,
            XenonError::InvalidArgument {
                kind: InvalidArgumentKind::OperationSpecific { .. },
                ..
            }
        ));
    }

    #[test]
    fn test_slice_info_range_start_after_end() {
        let err = SliceInfo::new(
            SliceInfoIndices::from_vec(vec![
                SliceInfoElem::Range { start: 5, end: 2 },
                SliceInfoElem::Index(0),
            ]),
            Ix2(10, 3),
            Ix1(3),
        )
        .expect_err("start > end");
        assert!(matches!(
            err,
            XenonError::InvalidArgument {
                kind: InvalidArgumentKind::RangeStartAfterEnd {
                    axis: 0,
                    start: 5,
                    end: 2
                },
                ..
            }
        ));
    }

    #[test]
    fn test_slice_info_indices_prefers_inline() {
        let indices = SliceInfoIndices::from_vec(vec![
            SliceInfoElem::Index(0),
            SliceInfoElem::Range { start: 0, end: 2 },
        ]);
        assert!(matches!(
            indices,
            SliceInfoIndices::Inline { len: 2, .. }
        ));
    }

    #[test]
    fn test_slice_info_indices_falls_back_to_dynamic() {
        let elems: Vec<SliceInfoElem> = (0..7).map(SliceInfoElem::Index).collect();
        let indices = SliceInfoIndices::from_vec(elems);
        assert!(matches!(indices, SliceInfoIndices::Dynamic(_)));
    }
}

#[cfg(test)]
mod slice_tests {
    use super::*;
    use crate::dimension::{Ix0, Ix1, Ix2, IxDyn};
    use crate::tensor::Tensor;

    fn tensor_ix2<A: crate::element::Element>(data: Vec<A>, shape: Ix2) -> Tensor<A, Ix2> {
        unsafe { Tensor::from_raw_vec_unchecked(data, shape) }
    }

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

    #[test]
    fn test_slice_high_rank_ixdyn() {
        let dyn_shape = IxDyn::from_slice(&[2, 2, 2, 2, 2, 2, 2]);
        let total: usize = dyn_shape.slice().iter().product();
        // SAFETY: shape size == data.len().
        let tensor = unsafe {
            Tensor::from_raw_vec_unchecked(
                (0i32..total as i32).collect(),
                dyn_shape.clone(),
            )
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