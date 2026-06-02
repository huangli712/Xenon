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
            },
            SliceInfoIndices::Dynamic(v) => {
                let elem = *v.get(self.pos)?;
                self.pos += 1;
                Some(elem)
            },
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
    ///
    /// # Errors
    ///
    /// Returns `XenonError::InvalidArgument` for any of:
    /// - `InvalidArgumentKind::OperationSpecific` with constraint
    ///   `"slice rank does not match input dimension"` — `indices.len()
    ///   != in_dim.ndim()`.
    /// - `InvalidArgumentKind::OperationSpecific` with constraint
    ///   `"slice output rank does not match Range count"` — the number of
    ///   `SliceInfoElem::Range` entries in `indices` does not equal
    ///   `out_dim.ndim()`.
    /// - `InvalidArgumentKind::RangeStartAfterEnd { axis, start, end }` —
    ///   some `Range { start, end }` has `start > end`.
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
        assert!(matches!(indices, SliceInfoIndices::Inline { len: 2, .. }));
    }

    #[test]
    fn test_slice_info_indices_falls_back_to_dynamic() {
        let elems: Vec<SliceInfoElem> = (0..7).map(SliceInfoElem::Index).collect();
        let indices = SliceInfoIndices::from_vec(elems);
        assert!(matches!(indices, SliceInfoIndices::Dynamic(_)));
    }
}

