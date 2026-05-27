//! Compile-time broadcast dimension inference.
//!
//! This module provides the type-level half of the broadcasting system:
//! it derives the result *type* (rank / static vs IxDyn) at compile time,
//! with no statement about per-axis length compatibility. Runtime length
//! checks live in the broadcast shape/strides logic.

use crate::private::Sealed;
use crate::dimension::{Dimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};

/// Trait for computing the output dimension type when broadcasting two arrays.
///
/// Same static dimension broadcasts to itself: `IxN BroadcastDim IxN → IxN`.
/// Cross static dimensions: higher-rank wins. Any IxDyn mixed: `IxDyn`.
///
/// Sealed: closed to `Ix0..Ix6` and `IxDyn` only.
pub trait BroadcastDim<Other: Dimension>: Dimension + Sealed {
    /// The output dimension type after broadcasting.
    type Output: Dimension;
}

// --- Same static dimension: IxN BroadcastDim IxN → IxN (7 impls) ---

impl BroadcastDim<Ix0> for Ix0 {
    type Output = Ix0;
}
impl BroadcastDim<Ix1> for Ix1 {
    type Output = Ix1;
}
impl BroadcastDim<Ix2> for Ix2 {
    type Output = Ix2;
}
impl BroadcastDim<Ix3> for Ix3 {
    type Output = Ix3;
}
impl BroadcastDim<Ix4> for Ix4 {
    type Output = Ix4;
}
impl BroadcastDim<Ix5> for Ix5 {
    type Output = Ix5;
}
impl BroadcastDim<Ix6> for Ix6 {
    type Output = Ix6;
}

// --- Cross static rank: take higher rank (42 impls, all bidirectional pairs) ---

// Ix0 × {Ix1..Ix6}: 12 impls (6 each direction).

impl BroadcastDim<Ix0> for Ix1 {
    type Output = Ix1;
}
impl BroadcastDim<Ix0> for Ix2 {
    type Output = Ix2;
}
impl BroadcastDim<Ix0> for Ix3 {
    type Output = Ix3;
}
impl BroadcastDim<Ix0> for Ix4 {
    type Output = Ix4;
}
impl BroadcastDim<Ix0> for Ix5 {
    type Output = Ix5;
}
impl BroadcastDim<Ix0> for Ix6 {
    type Output = Ix6;
}
impl BroadcastDim<Ix1> for Ix0 {
    type Output = Ix1;
}
impl BroadcastDim<Ix2> for Ix0 {
    type Output = Ix2;
}
impl BroadcastDim<Ix3> for Ix0 {
    type Output = Ix3;
}
impl BroadcastDim<Ix4> for Ix0 {
    type Output = Ix4;
}
impl BroadcastDim<Ix5> for Ix0 {
    type Output = Ix5;
}
impl BroadcastDim<Ix6> for Ix0 {
    type Output = Ix6;
}

// Ix1 × {Ix2..Ix6}: 10 impls.

impl BroadcastDim<Ix1> for Ix2 {
    type Output = Ix2;
}
impl BroadcastDim<Ix1> for Ix3 {
    type Output = Ix3;
}
impl BroadcastDim<Ix1> for Ix4 {
    type Output = Ix4;
}
impl BroadcastDim<Ix1> for Ix5 {
    type Output = Ix5;
}
impl BroadcastDim<Ix1> for Ix6 {
    type Output = Ix6;
}
impl BroadcastDim<Ix2> for Ix1 {
    type Output = Ix2;
}
impl BroadcastDim<Ix3> for Ix1 {
    type Output = Ix3;
}
impl BroadcastDim<Ix4> for Ix1 {
    type Output = Ix4;
}
impl BroadcastDim<Ix5> for Ix1 {
    type Output = Ix5;
}
impl BroadcastDim<Ix6> for Ix1 {
    type Output = Ix6;
}

// Ix2 × {Ix3..Ix6}: 8 impls.

impl BroadcastDim<Ix2> for Ix3 {
    type Output = Ix3;
}
impl BroadcastDim<Ix2> for Ix4 {
    type Output = Ix4;
}
impl BroadcastDim<Ix2> for Ix5 {
    type Output = Ix5;
}
impl BroadcastDim<Ix2> for Ix6 {
    type Output = Ix6;
}
impl BroadcastDim<Ix3> for Ix2 {
    type Output = Ix3;
}
impl BroadcastDim<Ix4> for Ix2 {
    type Output = Ix4;
}
impl BroadcastDim<Ix5> for Ix2 {
    type Output = Ix5;
}
impl BroadcastDim<Ix6> for Ix2 {
    type Output = Ix6;
}

// Ix3 × {Ix4..Ix6}: 6 impls.

impl BroadcastDim<Ix3> for Ix4 {
    type Output = Ix4;
}
impl BroadcastDim<Ix3> for Ix5 {
    type Output = Ix5;
}
impl BroadcastDim<Ix3> for Ix6 {
    type Output = Ix6;
}
impl BroadcastDim<Ix4> for Ix3 {
    type Output = Ix4;
}
impl BroadcastDim<Ix5> for Ix3 {
    type Output = Ix5;
}
impl BroadcastDim<Ix6> for Ix3 {
    type Output = Ix6;
}

// Ix4 × {Ix5..Ix6}: 4 impls.

impl BroadcastDim<Ix4> for Ix5 {
    type Output = Ix5;
}
impl BroadcastDim<Ix4> for Ix6 {
    type Output = Ix6;
}
impl BroadcastDim<Ix5> for Ix4 {
    type Output = Ix5;
}
impl BroadcastDim<Ix6> for Ix4 {
    type Output = Ix6;
}

// Ix5 × Ix6: 2 impls.

impl BroadcastDim<Ix5> for Ix6 {
    type Output = Ix6;
}
impl BroadcastDim<Ix6> for Ix5 {
    type Output = Ix6;
}

// --- Static × IxDyn: always IxDyn (14 impls, 7 bidirectional pairs) ---

impl BroadcastDim<IxDyn> for Ix0 {
    type Output = IxDyn;
}
impl BroadcastDim<IxDyn> for Ix1 {
    type Output = IxDyn;
}
impl BroadcastDim<IxDyn> for Ix2 {
    type Output = IxDyn;
}
impl BroadcastDim<IxDyn> for Ix3 {
    type Output = IxDyn;
}
impl BroadcastDim<IxDyn> for Ix4 {
    type Output = IxDyn;
}
impl BroadcastDim<IxDyn> for Ix5 {
    type Output = IxDyn;
}
impl BroadcastDim<IxDyn> for Ix6 {
    type Output = IxDyn;
}
impl BroadcastDim<Ix0> for IxDyn {
    type Output = IxDyn;
}
impl BroadcastDim<Ix1> for IxDyn {
    type Output = IxDyn;
}
impl BroadcastDim<Ix2> for IxDyn {
    type Output = IxDyn;
}
impl BroadcastDim<Ix3> for IxDyn {
    type Output = IxDyn;
}
impl BroadcastDim<Ix4> for IxDyn {
    type Output = IxDyn;
}
impl BroadcastDim<Ix5> for IxDyn {
    type Output = IxDyn;
}
impl BroadcastDim<Ix6> for IxDyn {
    type Output = IxDyn;
}

// --- IxDyn × IxDyn: 1 impl ---

impl BroadcastDim<IxDyn> for IxDyn {
    type Output = IxDyn;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::any::TypeId;

    /// Helper: assert that `<A as BroadcastDim<B>>::Output` is type-equal to
    /// `O`. Compiles only if the bound holds.
    fn assert_output<A, B, O>()
    where
        A: BroadcastDim<B, Output = O>,
        B: Dimension,
        O: Dimension + 'static,
    {
        // Runtime sanity: TypeId of O matches itself. The real assertion is
        // the compile-time bound `Output = O`.
        assert_eq!(TypeId::of::<O>(), TypeId::of::<O>());
        let _ = TypeId::of::<A>();
        let _ = TypeId::of::<B>();
    }

    /// Same-rank cases (7).
    #[test]
    fn test_broadcast_dim_same_rank() {
        assert_output::<Ix0, Ix0, Ix0>();
        assert_output::<Ix1, Ix1, Ix1>();
        assert_output::<Ix2, Ix2, Ix2>();
        assert_output::<Ix3, Ix3, Ix3>();
        assert_output::<Ix4, Ix4, Ix4>();
        assert_output::<Ix5, Ix5, Ix5>();
        assert_output::<Ix6, Ix6, Ix6>();
    }

    /// Cross-static representative pairs (covers each rank-gap once).
    #[test]
    fn test_broadcast_dim_cross_static() {
        assert_output::<Ix0, Ix6, Ix6>();
        assert_output::<Ix6, Ix0, Ix6>();
        assert_output::<Ix1, Ix3, Ix3>();
        assert_output::<Ix3, Ix1, Ix3>();
        assert_output::<Ix2, Ix5, Ix5>();
        assert_output::<Ix5, Ix2, Ix5>();
    }

    /// Static × IxDyn (any side IxDyn ⇒ IxDyn).
    #[test]
    fn test_broadcast_dim_with_ixdyn() {
        assert_output::<Ix0, IxDyn, IxDyn>();
        assert_output::<IxDyn, Ix0, IxDyn>();
        assert_output::<Ix3, IxDyn, IxDyn>();
        assert_output::<IxDyn, Ix3, IxDyn>();
        assert_output::<Ix6, IxDyn, IxDyn>();
        assert_output::<IxDyn, Ix6, IxDyn>();
        assert_output::<IxDyn, IxDyn, IxDyn>();
    }

    /// Symmetry guarantee: for every (D, E), the bidirectional
    /// Output type must agree. We sample one case per rank-gap.
    #[test]
    fn test_broadcast_dim_symmetry() {
        // Helper that compiles only if both directions yield identical Output.
        fn assert_symmetric<A, B, O>()
        where
            A: BroadcastDim<B, Output = O>,
            B: BroadcastDim<A, Output = O>,
            O: Dimension + 'static,
        {
            assert_eq!(TypeId::of::<O>(), TypeId::of::<O>());
            let _ = TypeId::of::<A>();
            let _ = TypeId::of::<B>();
        }
        assert_symmetric::<Ix0, Ix0, Ix0>();
        assert_symmetric::<Ix1, Ix1, Ix1>();
        assert_symmetric::<Ix0, Ix3, Ix3>();
        assert_symmetric::<Ix2, Ix5, Ix5>();
        assert_symmetric::<Ix4, IxDyn, IxDyn>();
        assert_symmetric::<IxDyn, Ix4, IxDyn>();
        assert_symmetric::<IxDyn, IxDyn, IxDyn>();
    }
}
