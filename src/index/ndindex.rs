//! The [`NdIndex`] trait and tuple/slice index implementations.

use crate::private::Sealed;
use crate::dimension::{Dimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};
use crate::error::{InvalidLayoutReason, Result, StorageKindTag, XenonError};
use crate::layout::Strides;

/// Compute the linear offset for a multi-dimensional index, with full
/// per-axis bounds and overflow checking.
///
/// Returns the offset as `usize` on success, or the appropriate [`XenonError`]
/// on failure.
fn checked_offset(index: &[usize], shape: &[usize], strides: &[usize]) -> Result<usize> {
    // --- Rank check ---
    if index.len() != shape.len() {
        return Err(XenonError::DimensionMismatch {
            operation: "NdIndex::index_checked".into(),
            expected: shape.len(),
            actual: index.len(),
        });
    }

    let mut offset = 0usize;

    // Standard multi-dimensional linear offset:
    //
    //   offset = Σ (index[axis] × strides[axis])
    //
    // Each term is computed with checked arithmetic to prevent overflow.
    for (axis, ((&idx, &extent), &stride)) in
        index.iter().zip(shape).zip(strides).enumerate()
    {
        // --- Per-axis bounds ---
        if idx >= extent {
            return Err(XenonError::IndexOutOfBounds {
                operation: "NdIndex::index_checked".into(),
                attempted_index: index.to_vec(),
                axis,
                shape: shape.to_vec(),
            });
        }

        // --- term = idx * stride ---
        // Checked multiply: idx × stride must not overflow usize.
        let term = idx
            .checked_mul(stride)
            .ok_or_else(|| XenonError::InvalidLayout {
                operation: "NdIndex::index_checked".into(),
                storage_kind: StorageKindTag::View,
                shape: shape.to_vec(),
                strides: strides.to_vec(),
                offset,
                storage_len: 0,
                reason: InvalidLayoutReason::AccessRangeExceedsStorage,
            })?;

        // --- offset += term ---
        // Checked add: accumulator must not overflow usize.
        offset = offset
            .checked_add(term)
            .ok_or_else(|| XenonError::InvalidLayout {
                operation: "NdIndex::index_checked".into(),
                storage_kind: StorageKindTag::View,
                shape: shape.to_vec(),
                strides: strides.to_vec(),
                offset,
                storage_len: 0,
                reason: InvalidLayoutReason::AccessRangeExceedsStorage,
            })?;
    }
    Ok(offset)
}

/// Compute the linear offset for a multi-dimensional index without any bounds
/// or overflow checking.
///
/// # Safety
///
/// The caller must ensure that `index.len() == strides.len()`, each
/// per-axis component is within bounds, and no `usize` overflow occurs
/// during arithmetic.
fn unchecked_offset(index: &[usize], strides: &[usize]) -> usize {
    index.iter().zip(strides).map(|(i, s)| i * s).sum()
}

// Each concrete index type must individually implement `Sealed` so that
// `NdIndex<D>: Sealed` forms a closed set — no downstream crate can add
// new index types. Tuple arities Ix0..Ix6 are open-coded because
// `(usize,)`, `(usize, usize)`, … are distinct, unrelated types in Rust.
//
// Sealed implementations

impl Sealed for () {}
impl Sealed for (usize,) {}
impl Sealed for (usize, usize) {}
impl Sealed for (usize, usize, usize) {}
impl Sealed for (usize, usize, usize, usize) {}
impl Sealed for (usize, usize, usize, usize, usize) {}
impl Sealed for (usize, usize, usize, usize, usize, usize) {}
impl Sealed for &[usize] {}

/// Sealed trait for types that can be used as multi-dimensional indices.
pub trait NdIndex<D: Dimension>: Sealed {
    /// Validates the index against `dim` and computes the linear offset
    /// via `strides`.
    ///
    /// # Errors
    ///
    /// - rank mismatch → [`XenonError::DimensionMismatch`]
    /// - per-axis out of bounds → [`XenonError::IndexOutOfBounds`]
    /// - offset arithmetic overflow → [`XenonError::InvalidLayout`]
    fn index_checked(&self, dim: &D, strides: &Strides<D>) -> Result<usize>;

    /// Computes the linear offset without any validation.
    ///
    /// # Safety
    /// The caller must ensure rank match, per-axis bounds, and no offset overflow.
    unsafe fn index_unchecked(&self, strides: &Strides<D>) -> usize;

    /// Converts the index to a flat [`Vec`]`<usize>` for error diagnostics.
    fn to_index_vec(&self) -> Vec<usize>;
}

impl NdIndex<Ix0> for () {
    fn index_checked(&self, _dim: &Ix0, _strides: &Strides<Ix0>) -> Result<usize> {
        Ok(0)
    }

    unsafe fn index_unchecked(&self, _strides: &Strides<Ix0>) -> usize {
        0
    }

    fn to_index_vec(&self) -> Vec<usize> {
        vec![]
    }
}

impl NdIndex<Ix1> for (usize,) {
    fn index_checked(&self, dim: &Ix1, strides: &Strides<Ix1>) -> Result<usize> {
        checked_offset(
            &[self.0],
            dim.slice(),
            strides.as_slice(),
        )
    }
   
    unsafe fn index_unchecked(&self, strides: &Strides<Ix1>) -> usize {
        unchecked_offset(
            &[self.0],
            strides.as_slice(),
        )
    }
    
    fn to_index_vec(&self) -> Vec<usize> {
        vec![self.0]
    }
}

impl NdIndex<Ix2> for (usize, usize) {
    fn index_checked(&self, dim: &Ix2, strides: &Strides<Ix2>) -> Result<usize> {
        checked_offset(
            &[self.0, self.1],
            dim.slice(),
            strides.as_slice(),
        )
    }

    unsafe fn index_unchecked(&self, strides: &Strides<Ix2>) -> usize {
        unchecked_offset(
            &[self.0, self.1],
            strides.as_slice(),
        )
    }

    fn to_index_vec(&self) -> Vec<usize> {
        vec![self.0, self.1]
    }
}

impl NdIndex<Ix3> for (usize, usize, usize) {
    fn index_checked(&self, dim: &Ix3, strides: &Strides<Ix3>) -> Result<usize> {
        checked_offset(
            &[self.0, self.1, self.2],
            dim.slice(),
            strides.as_slice(),
        )
    }

    unsafe fn index_unchecked(&self, strides: &Strides<Ix3>) -> usize {
        unchecked_offset(
            &[self.0, self.1, self.2],
            strides.as_slice(),
        )
    }

    fn to_index_vec(&self) -> Vec<usize> {
        vec![self.0, self.1, self.2]
    }
}

impl NdIndex<Ix4> for (usize, usize, usize, usize) {
    fn index_checked(&self, dim: &Ix4, strides: &Strides<Ix4>) -> Result<usize> {
        checked_offset(
            &[self.0, self.1, self.2, self.3],
            dim.slice(),
            strides.as_slice(),
        )
    }

    unsafe fn index_unchecked(&self, strides: &Strides<Ix4>) -> usize {
        unchecked_offset(
            &[self.0, self.1, self.2, self.3],
            strides.as_slice(),
        )
    }

    fn to_index_vec(&self) -> Vec<usize> {
        vec![self.0, self.1, self.2, self.3]
    }
}

impl NdIndex<Ix5> for (usize, usize, usize, usize, usize) {
    fn index_checked(&self, dim: &Ix5, strides: &Strides<Ix5>) -> Result<usize> {
        checked_offset(
            &[self.0, self.1, self.2, self.3, self.4],
            dim.slice(),
            strides.as_slice(),
        )
    }

    unsafe fn index_unchecked(&self, strides: &Strides<Ix5>) -> usize {
        unchecked_offset(
            &[self.0, self.1, self.2, self.3, self.4],
            strides.as_slice(),
        )
    }
    
    fn to_index_vec(&self) -> Vec<usize> {
        vec![self.0, self.1, self.2, self.3, self.4]
    }
}

impl NdIndex<Ix6> for (usize, usize, usize, usize, usize, usize) {
    fn index_checked(&self, dim: &Ix6, strides: &Strides<Ix6>) -> Result<usize> {
        checked_offset(
            &[self.0, self.1, self.2, self.3, self.4, self.5],
            dim.slice(),
            strides.as_slice(),
        )
    }
    
    unsafe fn index_unchecked(&self, strides: &Strides<Ix6>) -> usize {
        unchecked_offset(
            &[self.0, self.1, self.2, self.3, self.4, self.5],
            strides.as_slice(),
        )
    }
    
    fn to_index_vec(&self) -> Vec<usize> {
        vec![self.0, self.1, self.2, self.3, self.4, self.5]
    }
}

impl NdIndex<IxDyn> for &[usize] {
    fn index_checked(&self, dim: &IxDyn, strides: &Strides<IxDyn>) -> Result<usize> {
        checked_offset(self, dim.slice(), strides.as_slice())
    }
    
    unsafe fn index_unchecked(&self, strides: &Strides<IxDyn>) -> usize {
        unchecked_offset(self, strides.as_slice())
    }
    
    fn to_index_vec(&self) -> Vec<usize> {
        self.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::Ix2;

    /// `checked_offset` returns [`IndexOutOfBounds`] when an index component
    /// exceeds the corresponding shape dimension.
    ///
    /// [`IndexOutOfBounds`]: XenonError::IndexOutOfBounds
    #[test]
    fn test_checked_offset_out_of_bounds() {
        let err = checked_offset(&[2, 0], &[2, 3], &[1, 2]).expect_err("out of bounds");
        assert!(matches!(err, XenonError::IndexOutOfBounds { .. }));
    }

    /// `checked_offset` returns [`DimensionMismatch`] when the index slice
    /// has a different length than the shape.
    ///
    /// [`DimensionMismatch`]: XenonError::DimensionMismatch
    #[test]
    fn test_checked_offset_rank_mismatch() {
        let err = checked_offset(&[0, 0, 0], &[2, 3], &[1, 2]).expect_err("rank mismatch");
        assert!(matches!(
            err,
            XenonError::DimensionMismatch {
                expected: 2,
                actual: 3,
                ..
            }
        ));
    }

    /// A valid 2D tuple index computes the correct linear offset via `index_checked`.
    #[test]
    fn test_ndindex_tuple_2d_checked() {
        let dim = Ix2(2, 3);
        let strides = Strides::from_slice(&[1, 2]).expect("known-valid stride");
        let idx = (1usize, 2usize);
        assert_eq!(idx.index_checked(&dim, &strides).expect("valid index"), 5);
    }

    /// An out-of-bounds 2D tuple index reports [`IndexOutOfBounds`]
    /// via `index_checked`.
    #[test]
    fn test_ndindex_tuple_2d_out_of_bounds() {
        let dim = Ix2(2, 3);
        let strides = Strides::from_slice(&[1, 2]).expect("known-valid stride");
        let err = (2usize, 0usize)
            .index_checked(&dim, &strides)
            .expect_err("out of bounds");
        assert!(matches!(err, XenonError::IndexOutOfBounds { .. }));
    }
}
