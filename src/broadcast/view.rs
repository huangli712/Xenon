//! Broadcast view construction and the shared two-input broadcast prologue.

use crate::error::XenonError;
use crate::dimension::{BroadcastDimension, Dimension, IntoDimension};
use crate::element::Element;
use crate::layout::{Strides, compute_layout_flags};
use crate::storage::{Storage, ViewRepr};
use crate::tensor::{TensorBase, TensorView};

use super::shape::{broadcast_shape, broadcast_strides};

/// The two broadcast views plus the common output dimension produced by
/// [`broadcast_with`].
pub(crate) type BroadcastViews<'a, 'b, A, D1, D2> = (
    TensorView<'a, A, <D1 as BroadcastDimension<D2>>::Output>,
    TensorView<'b, A, <D1 as BroadcastDimension<D2>>::Output>,
    <D1 as BroadcastDimension<D2>>::Output,
);

/// Broadcast both operands to their common shape, returning the two
/// broadcast views together with the output dimension. Shared `pub(crate)`
/// entry point for two-input broadcast; consumers (e.g. `math`) must route
/// double-input broadcast through here rather than redefining the rule.
pub(crate) fn broadcast_with<'a, 'b, A, S1, S2, D1, D2>(
    a: &'a TensorBase<S1, D1>,
    b: &'b TensorBase<S2, D2>,
) -> Result<BroadcastViews<'a, 'b, A, D1, D2>, XenonError>
where
    A: Element,
    S1: Storage<Elem = A>,
    S2: Storage<Elem = A>,
    D1: Dimension + BroadcastDimension<D2>,
    D2: Dimension,
{
    let out_shape = broadcast_shape(a.shape(), b.shape())?;
    let out_dim = <D1 as BroadcastDimension<D2>>::Output::try_from_slice(out_shape.slice())
        .expect("broadcast_shape validated the output shape");
    let a_view = broadcast_to(a, out_dim.clone())?;
    let b_view = broadcast_to(b, out_dim.clone())?;
    Ok((a_view, b_view, out_dim))
}

/// Implementation backing [`TensorBase::broadcast_to`], extracted as a
/// `pub(crate)` free function so internal call sites can broadcast a single
/// tensor without method syntax. The public inherent method forwards here; see
/// it for the public contract, error semantics, and the read-only guarantee.
pub(crate) fn broadcast_to<S, A, D, E>(
    tensor: &TensorBase<S, D>,
    shape: E,
) -> Result<TensorView<'_, A, E::Dim>, XenonError>
where
    S: Storage<Elem = A>,
    A: Element,
    D: Dimension,
    E: IntoDimension,
{
    let target_dim: E::Dim = shape.into_dimension();
    let target_shape: &[usize] = target_dim.slice();

    // Error path: `broadcast_strides` returns BroadcastError or InvalidArgument on
    // shape-level failure. No additional `broadcast_shape` pre-check is performed —
    // it would be redundant and semantically wrong (`broadcast_shape` is
    // bidirectional; `broadcast_to` is single-direction).
    let strides_vec: Vec<usize> = broadcast_strides(
        tensor.shape(),
        tensor.strides(),
        target_shape
    )?;
    let strides: Strides<E::Dim> = Strides::<E::Dim>::from_slice(
        &strides_vec
    )?;

    // Layout flags are computed by `compute_layout_flags` — the single source of
    // truth. We do NOT construct layout flags by hand or branch on zero strides
    // ourselves (`compute_layout_flags` already handles `HAS_ZERO_STRIDE` including
    // empty-array degeneracy).
    //
    // `ptr` argument: the logical-first pointer of the result view. `broadcast_to`
    // preserves `offset`, so the result's logical-first pointer equals the source's
    // logical-first pointer = `tensor.as_ptr()` (which returns the logical-first
    // pointer, NOT the storage base).
    let flags = compute_layout_flags::<A, E::Dim>(
        &target_dim,
        &strides,
        tensor.as_ptr()
    );

    // (1) Build the `ViewRepr<'_, A>` that borrows the source storage.
    //     ViewRepr holds `(storage_base_ptr, storage_len)` — NOT the
    //     logical-first pointer. The base pointer is
    //     `tensor.as_storage_ptr()` (which returns `storage.as_ptr()` directly,
    //     WITHOUT adding offset); `storage_len()` reports the source storage
    //     extent.
    //
    // SAFETY (ViewRepr::from_raw_parts):
    //   - `tensor.as_storage_ptr()` is non-null and aligned (carried unchanged
    //     from the source storage that was already constructed and validated).
    //   - `[base, base + storage_len)` lies inside a single allocation, all
    //     `storage_len` elements are initialized values of `A`.
    //   - The returned `ViewRepr<'_, A>` lifetime is bound to `&tensor`; no
    //     mutable alias to the same memory is alive during that borrow.
    //   - Empty-storage case: `as_storage_ptr()` returns the dangling sentinel,
    //     `storage_len == 0` makes the range empty, and the sentinel is never
    //     dereferenced.
    let view_storage: ViewRepr<'_, A> = unsafe {
        ViewRepr::from_raw_parts(
            tensor.as_storage_ptr(),
            tensor.storage_len()
        )
    };

    // (2) Finalize via `TensorBase::new_unchecked` — the canonical `pub(crate)`
    //     unsafe constructor.
    //
    // Why NOT `TensorView::from_raw_parts`:
    //   (a) Its `ptr` argument must be the storage base; passing
    //       `tensor.as_ptr()` (logical-first) + `tensor.offset()` would
    //       double-apply the offset → UB.
    //   (b) It always sets `derived_from_view_mut := false`, preventing
    //       propagation from `ViewMut` sources; `new_unchecked` accepts the
    //       flag explicitly.
    //   (c) All internal unchecked constructors forward to `new_unchecked`
    //       rather than defining parallel safety invariants — broadcast is
    //       exactly such an internal caller.
    //
    // `derived_from_view_mut` propagation:
    //   - `true` ONLY when source is a `ViewMutRepr` being demoted, or a
    //     `ViewRepr` already carrying `derived_from_view_mut == true`.
    //   - `tensor.derived_from_view_mut` (field access) reports the combined
    //     classification; forwarding it satisfies the propagation rule for
    //     every supported source storage type.
    //   - For `Owned<A>` sources the field is `false`.
    //
    // SAFETY (TensorBase::new_unchecked):
    //   - `target_dim` + `strides` + `tensor.offset()`: target shape is
    //     rank-checked (`Strides::from_slice` succeeded). `broadcast_strides`
    //     already produced a layout that broadcasts the source onto
    //     `target_dim`, so the result's logical access range equals the
    //     source's (each `target_dim` index resolves to a source index via
    //     zero-stride/repeat axes), which still fits the source `storage_len`
    //     carried by `view_storage`.
    //   - `flags` was produced by `compute_layout_flags` on the same
    //     `(target_dim, strides, tensor.as_ptr())` triple — `tensor.as_ptr()`
    //     is the logical-first pointer the result view will expose via its own
    //     `as_ptr()` (offset unchanged).
    //   - `derived_from_view_mut` is forwarded from the source per the
    //     propagation rule above.
    //   - The layout family is valid for an immutable view: zero-stride
    //     broadcast layouts are explicitly accepted on read-only paths.
    let view: TensorView<'_, A, E::Dim> = unsafe {
        TensorBase::<ViewRepr<'_, A>, E::Dim>::new_unchecked(
            view_storage,
            target_dim,
            strides,
            tensor.offset(),
            flags,
            tensor.derived_from_view_mut,
        )
    };

    Ok(view)
}

impl<S, A, D> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    A: Element,
    D: Dimension,
{
    /// Broadcast `self` to `shape`. Returns a read-only zero-copy view sharing
    /// the underlying storage.
    ///
    /// # Errors
    ///
    /// - `XenonError::BroadcastError` — `self.shape()` is not broadcast-compatible
    ///   with `shape` (rank exceeds target, or a non-singleton source axis differs
    ///   from the target axis).
    /// - `XenonError::InvalidArgument` — defensive: `self.shape().len()` does not
    ///   match `self.strides().len()` (caller bug; unreachable under correct
    ///   `TensorBase` invariants).
    /// - `XenonError::DimensionMismatch` — the broadcast stride vector length does
    ///   not match the rank of `E::Dim` (caller-provided target rank mismatch for
    ///   fixed-rank `E`).
    ///
    pub fn broadcast_to<E>(
        &self, shape: E
    ) -> Result<TensorView<'_, A, E::Dim>, XenonError>
    where
        E: IntoDimension,
    {
        broadcast_to(self, shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::XenonError;
    use crate::layout::LayoutState;
    use crate::tensor::Tensor2;

    // --- broadcast_with tests -----------------------------------------------

    /// Two-input broadcast where each operand expands a *different* axis:
    /// `a=[1,3]` broadcasts axis 0, `b=[2,1]` broadcasts axis 1, both reaching
    /// `[2,3]`. Verifies the prologue broadcasts each side independently.
    #[test]
    fn test_broadcast_with_mutual() {
        let a: Tensor2<f64> = Tensor2::from_shape_vec(
            [1, 3],
            vec![1.0, 2.0, 3.0]
        ).expect("valid test input");
        let b: Tensor2<f64> = Tensor2::from_shape_vec(
            [2, 1],
            vec![10.0, 20.0]
        ).expect("valid test input");
        let (a_view, b_view, out_dim) = broadcast_with(&a, &b)
            .expect("compatible shapes");
        assert_eq!(out_dim.slice(), &[2, 3]);
        
        // a expands axis 0 (stride 0) and keeps axis 1.
        assert_eq!(a_view.shape(), &[2, 3]);
        assert_eq!(a_view.strides(), &[0, 1]);
        
        // b expands axis 1 (stride 0) and keeps axis 0.
        assert_eq!(b_view.shape(), &[2, 3]);
        assert_eq!(b_view.strides(), &[1, 0]);
        
        // Zero-copy: each view points at its own source.
        assert_eq!(a_view.as_ptr(), a.as_ptr());
        assert_eq!(b_view.as_ptr(), b.as_ptr());
    }

    /// 3-tuple contract: the returned `out_dim` equals both views' shape.
    #[test]
    fn test_broadcast_with_out_dim_matches_views() {
        let a: Tensor2<f64> = Tensor2::from_shape_vec(
            [1, 4],
            vec![1.0, 2.0, 3.0, 4.0]
        ).expect("valid test input");
        let b: Tensor2<f64> = Tensor2::from_shape_vec(
            [3, 1],
            vec![1.0, 2.0, 3.0]
        ).expect("valid test input");
        let (a_view, b_view, out_dim) = broadcast_with(&a, &b)
            .expect("compatible shapes");
        assert_eq!(out_dim.slice(), &[3, 4]);
        assert_eq!(out_dim.slice(), a_view.shape());
        assert_eq!(out_dim.slice(), b_view.shape());
        assert_eq!(a_view.shape(), b_view.shape());
    }

    /// Same-shape shortcut: identical input shapes broadcast to the same
    /// shape with original strides preserved (no zero strides introduced)
    /// and no `BroadcastView` classification.
    #[test]
    fn test_broadcast_with_same_shape_preserves_layout() {
        let a: Tensor2<f64> = Tensor2::from_shape_vec(
            [2, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        ).expect("valid test input");
        let b: Tensor2<f64> = Tensor2::from_shape_vec(
            [2, 3],
            vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        ).expect("valid test input");
        let (a_view, b_view, out_dim) = broadcast_with(&a, &b)
            .expect("compatible shapes");
        assert_eq!(out_dim.slice(), &[2, 3]);
        // No axis is broadcast: original strides survive, no zero strides.
        assert_eq!(a_view.strides(), a.strides());
        assert_eq!(b_view.strides(), b.strides());
        assert!(!a_view.flags().has_zero_stride());
        assert_ne!(a_view.layout_state(), LayoutState::BroadcastView);
    }

    /// Error propagation: incompatible shapes surface `BroadcastError`
    /// straight out of `broadcast_with` (raised by the internal
    /// `broadcast_shape` call).
    #[test]
    fn test_broadcast_with_error_propagation() {
        let a: Tensor2<f64> = Tensor2::zeros([2, 3])
            .expect("valid test input");
        let b: Tensor2<f64> = Tensor2::zeros([4, 3])
            .expect("valid test input");
        let err = broadcast_with(&a, &b)
            .expect_err("incompatible shapes");
        match err {
            XenonError::BroadcastError { operation, .. } => {
                assert_eq!(operation.as_ref(), "broadcast_shape");
            },
            other => panic!("expected BroadcastError, got {:?}", other),
        }
    }

    // --- broadcast_to tests -------------------------------------------------

    /// Tests basic broadcast: a `[1,3]` tensor broadcast to `[2,3]` produces
    /// a view with the target shape, stride 0 on the broadcast axis, and
    /// zero-copy pointer sharing.
    #[test]
    fn test_broadcast_to_basic() {
        let tensor: Tensor2<f64> = Tensor2::from_shape_vec(
            [1, 3],
            vec![1.0, 2.0, 3.0]
        ).expect("valid test input");
        let view = tensor
            .broadcast_to([2, 3])
            .expect("valid test input");
        assert_eq!(view.shape(), &[2, 3]);
        
        // Broadcast axis 0: stride 0.
        assert_eq!(view.strides()[0], 0);
        
        // Zero-copy: pointer is identical.
        assert_eq!(view.as_ptr(), tensor.as_ptr());
        
        // Offset preserved.
        assert_eq!(view.offset(), tensor.offset());
    }

    /// Both-singleton same-rank broadcast: a `[1, 1]` source expands *both*
    /// axes to `[2, 3]`, so both result strides are 0. (True rank-increasing
    /// broadcast is covered by the integration test `test_broadcast_left_pad`.)
    #[test]
    fn test_broadcast_to_scalar_to_higher_rank() {
        let scalar: Tensor2<f64> = Tensor2::from_shape_vec(
            [1, 1],
            vec![42.0]
        ).expect("valid test input");
        let view = scalar
            .broadcast_to([2, 3])
            .expect("valid test input");
        assert_eq!(view.shape(), &[2, 3]);
        // Both axes are broadcast: both strides are 0.
        assert_eq!(view.strides(), &[0, 0]);
    }

    /// Tests that broadcasting to an incompatible shape returns a structured
    /// `BroadcastError` with the correct fields.
    #[test]
    fn test_broadcast_to_error() {
        let tensor: Tensor2<f64> = Tensor2::zeros([2, 3])
            .expect("valid test input");
        let err = tensor
            .broadcast_to([4, 3])
            .expect_err("expected error");
        match err {
            XenonError::BroadcastError {
                operation,
                lhs_shape,
                rhs_shape,
                attempted_target_shape,
                axis,
            } => {
                assert_eq!(operation.as_ref(), "broadcast_strides");
                assert_eq!(lhs_shape, vec![2, 3]);
                assert_eq!(rhs_shape, vec![4, 3]);
                assert_eq!(attempted_target_shape, Some(vec![4, 3]));
                assert_eq!(axis, Some(0));
            },
            other => panic!("expected BroadcastError, got {:?}", other),
        }
    }

    /// Non-empty broadcast view (`product(shape) > 0` and `any(stride == 0)`)
    /// classifies as `BroadcastView`.
    #[test]
    fn test_broadcast_to_layout_flags_recomputed() {
        let tensor: Tensor2<f64> = Tensor2::zeros([1, 3])
            .expect("valid test input");
        let view = tensor
            .broadcast_to([2, 3])
            .expect("valid test input");
        assert_eq!(view.layout_state(), LayoutState::BroadcastView);
        assert!(view.flags().has_zero_stride());
    }

    /// Empty-array degenerate zero stride (`1 -> 0`) does NOT trigger
    /// `BroadcastView` classification.
    #[test]
    fn test_broadcast_to_empty_does_not_classify_as_broadcast_view() {
        let tensor: Tensor2<f64> = Tensor2::zeros([1, 3])
            .expect("valid test input");
        let view = tensor
            .broadcast_to([0, 3])
            .expect("valid test input");
        // `product(shape) == 0` ⇒ HAS_ZERO_STRIDE = false ⇒ not BroadcastView.
        assert_ne!(view.layout_state(), LayoutState::BroadcastView);
        assert!(!view.flags().has_zero_stride());
    }

    /// The broadcast view is iterable in read-only fashion, as guaranteed by
    /// the `TensorView` return type (which has no `&mut` access methods).
    #[test]
    fn test_broadcast_to_read_only_iterable() {
        let tensor: Tensor2<f64> = Tensor2::from_shape_vec(
            [1, 3],
            vec![1.0, 2.0, 3.0]
        ).expect("valid test input");
        let view = tensor
            .broadcast_to([2, 3])
            .expect("valid test input");
        let n: usize = view.iter().count();
        assert_eq!(n, 6);
    }
}
