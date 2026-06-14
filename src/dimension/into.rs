//! Dimension conversion traits.

use super::Dimension;
use super::IxDyn;
use super::{Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6};

/// Trait for types that can be converted into a dimension.
///
/// # Examples
///
/// ```
/// use xenon::dimension::{Dimension, IntoDimension, Ix3, IxDyn};
/// let d3: Ix3 = (2, 3, 4).into_dimension();
/// assert_eq!(d3.slice(), &[2, 3, 4]);
/// let dyn_dim: IxDyn = vec![5, 6].into_dimension();
/// assert_eq!(dyn_dim.slice(), &[5, 6]);
/// ```
pub trait IntoDimension {
    /// The resulting dimension type.
    type Dim: Dimension;

    /// Convert into a dimension.
    fn into_dimension(self) -> Self::Dim;
}

// --- Identity impl ----------------------------------------------------------

impl<D: Dimension> IntoDimension for D {
    type Dim = D;
    
    #[inline]
    fn into_dimension(self) -> Self::Dim {
        self
    }
}

// --- Tuple impls ------------------------------------------------------------

impl IntoDimension for () {
    type Dim = Ix0;
    
    #[inline]
    fn into_dimension(self) -> Ix0 {
        Ix0
    }
}

impl IntoDimension for (usize,) {
    type Dim = Ix1;
    
    #[inline]
    fn into_dimension(self) -> Ix1 {
        Ix1(self.0)
    }
}

impl IntoDimension for (usize, usize) {
    type Dim = Ix2;
    
    #[inline]
    fn into_dimension(self) -> Ix2 {
        Ix2(self.0, self.1)
    }
}

impl IntoDimension for (usize, usize, usize) {
    type Dim = Ix3;
    
    #[inline]
    fn into_dimension(self) -> Ix3 {
        Ix3(self.0, self.1, self.2)
    }
}

impl IntoDimension for (usize, usize, usize, usize) {
    type Dim = Ix4;
    
    #[inline]
    fn into_dimension(self) -> Ix4 {
        Ix4(self.0, self.1, self.2, self.3)
    }
}

impl IntoDimension for (usize, usize, usize, usize, usize) {
    type Dim = Ix5;
    
    #[inline]
    fn into_dimension(self) -> Ix5 {
        Ix5(self.0, self.1, self.2, self.3, self.4)
    }
}

impl IntoDimension for (usize, usize, usize, usize, usize, usize) {
    type Dim = Ix6;
    
    #[inline]
    fn into_dimension(self) -> Ix6 {
        Ix6(self.0, self.1, self.2, self.3, self.4, self.5)
    }
}

// --- Array impls ------------------------------------------------------------

impl IntoDimension for [usize; 0] {
    type Dim = Ix0;
    
    #[inline]
    fn into_dimension(self) -> Ix0 {
        Ix0
    }
}

impl IntoDimension for [usize; 1] {
    type Dim = Ix1;
    
    #[inline]
    fn into_dimension(self) -> Ix1 {
        Ix1(self[0])
    }
}

impl IntoDimension for [usize; 2] {
    type Dim = Ix2;
    
    #[inline]
    fn into_dimension(self) -> Ix2 {
        Ix2(self[0], self[1])
    }
}

impl IntoDimension for [usize; 3] {
    type Dim = Ix3;
    
    #[inline]
    fn into_dimension(self) -> Ix3 {
        Ix3(self[0], self[1], self[2])
    }
}

impl IntoDimension for [usize; 4] {
    type Dim = Ix4;
    
    #[inline]
    fn into_dimension(self) -> Ix4 {
        Ix4(self[0], self[1], self[2], self[3])
    }
}

impl IntoDimension for [usize; 5] {
    type Dim = Ix5;
    
    #[inline]
    fn into_dimension(self) -> Ix5 {
        Ix5(self[0], self[1], self[2], self[3], self[4])
    }
}

impl IntoDimension for [usize; 6] {
    type Dim = Ix6;
    
    #[inline]
    fn into_dimension(self) -> Ix6 {
        Ix6(self[0], self[1], self[2], self[3], self[4], self[5])
    }
}

// --- Slice / Vec impls ------------------------------------------------------

impl IntoDimension for &[usize] {
    type Dim = IxDyn;

    #[inline]
    fn into_dimension(self) -> IxDyn {
        IxDyn::from_slice(self)
    }
}

impl IntoDimension for Vec<usize> {
    type Dim = IxDyn;
    
    #[inline]
    fn into_dimension(self) -> IxDyn {
        IxDyn::from_vec(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tuples preserve static dimensionality.
    #[test]
    fn test_tuple_into_dimension() {
        let d0: Ix0 = ().into_dimension();
        let _ = d0;

        let d1: Ix1 = (5,).into_dimension();
        assert_eq!(d1, Ix1(5));

        let d2: Ix2 = (2, 3).into_dimension();
        assert_eq!(d2, Ix2(2, 3));

        let d3: Ix3 = (2, 3, 4).into_dimension();
        assert_eq!(d3, Ix3(2, 3, 4));

        let d4: Ix4 = (2, 3, 4, 5).into_dimension();
        assert_eq!(d4, Ix4(2, 3, 4, 5));

        let d5: Ix5 = (2, 3, 4, 5, 6).into_dimension();
        assert_eq!(d5, Ix5(2, 3, 4, 5, 6));

        let d6: Ix6 = (2, 3, 4, 5, 6, 7).into_dimension();
        assert_eq!(d6, Ix6(2, 3, 4, 5, 6, 7));
    }

    /// Array impls preserve static dimensionality.
    #[test]
    fn test_array_into_dimension() {
        let d0: Ix0 = [].into_dimension();
        let _ = d0;
        let d3: Ix3 = [2, 3, 4].into_dimension();
        assert_eq!(d3, Ix3(2, 3, 4));
        let d6: Ix6 = [1, 2, 3, 4, 5, 6].into_dimension();
        assert_eq!(d6, Ix6(1, 2, 3, 4, 5, 6));
    }

    /// `&[usize]` becomes `IxDyn`.
    #[test]
    fn test_slice_to_ixdyn() {
        let dyn_dim: IxDyn = (&[5, 6, 7][..]).into_dimension();
        assert_eq!(dyn_dim.slice(), &[5, 6, 7]);
    }

    /// Vec<usize> becomes IxDyn (zero-copy move).
    #[test]
    fn test_vec_to_ixdyn() {
        let dyn_dim: IxDyn = vec![1, 2, 3].into_dimension();
        assert_eq!(dyn_dim.slice(), &[1, 2, 3]);
    }

    /// Identity impl: any Dimension converts to itself.
    #[test]
    fn test_dimension_identity() {
        let d3 = Ix3(2, 3, 4);
        let same: Ix3 = d3.into_dimension();
        assert_eq!(same, d3);
        let dyn_dim = IxDyn::from_slice(&[1, 2]);
        let same_dyn: IxDyn = dyn_dim.clone().into_dimension();
        assert_eq!(same_dyn, dyn_dim);
    }
}
