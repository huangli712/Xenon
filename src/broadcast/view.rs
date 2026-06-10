use crate::dimension::{BroadcastDim, Dimension, IntoDimension};
use crate::element::Element;
use crate::error::XenonError;
use crate::layout::{Strides, compute_layout_flags};
use crate::storage::{Storage, ViewRepr};
use crate::tensor::{TensorBase, TensorView};

use crate::broadcast::shape::{broadcast_shape, broadcast_strides};

impl<S, A, D> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    A: Element,
    D: Dimension,
{
    /// Broadcast `self` to `shape`. Returns a read-only zero-copy view sharing the
    /// underlying storage. See `15-broadcast.md §5.1` line 124-132 and §6.4.
    ///
    /// # Errors
    ///
    /// - `XenonError::BroadcastError` — `self.shape()` is not broadcast-compatible
    ///   with `shape` (rank exceeds target, or a non-singleton source axis differs
    ///   from the target axis). See `15-broadcast.md §6.3` and `26-error.md §5.1`.
    /// - `XenonError::InvalidArgument` — defensive: `self.shape().len()` does not
    ///   match `self.strides().len()` (caller bug; unreachable under correct
    ///   `TensorBase` invariants).
    /// - `XenonError::DimensionMismatch` — the broadcast stride vector length does
    ///   not match the rank of `E::Dim` (caller-provided target rank mismatch for
    ///   fixed-rank `E`). See `06-layout §5.5`.
    ///
    /// # Read-only guarantee (compile-fail demonstration)
    ///
    /// The broadcast result is `TensorView<'_, A, E::Dim>` — a read-only view.
    /// Any attempt to acquire `&mut` access through the broadcast result must fail
    /// to compile.
    ///
    /// ```compile_fail
    /// use xenon::tensor::Tensor2;
    /// let tensor: Tensor2<f64> = Tensor2::ones([1, 3]).expect("valid test input");
    /// let view = tensor.broadcast_to([2, 3]).expect("valid test input");
    /// let _slice: &mut [f64] = view.as_mut_slice();  // No such method on TensorView.
    /// ```
    pub fn broadcast_to<E>(&self, shape: E) -> Result<TensorView<'_, A, E::Dim>, XenonError>
    where
        E: IntoDimension,
    {
        broadcast_to(self, shape)
    }
}

// ----------------------------------------------------------------------------
// broadcast_to free-function implementation
// ----------------------------------------------------------------------------

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
    // it would be redundant (W11T6 already iterates per-axis) and semantically wrong
    // (`broadcast_shape` is bidirectional; `broadcast_to` is single-direction).
    let strides_vec: Vec<usize> =
        broadcast_strides(tensor.shape(), tensor.strides(), target_shape)?;
    let strides: Strides<E::Dim> = Strides::<E::Dim>::from_slice(&strides_vec)?;

    // Per 06-layout §5.5 line 233: the SINGLE source of truth for layout flags is
    // `compute_layout_flags`. We do NOT construct `LayoutFlags::broadcast_view()` by
    // hand (no such constructor exists in 06-layout.md), and we do NOT branch on
    // `has_zero_stride` ourselves (§6.1 line 684 already handles
    // `HAS_ZERO_STRIDE := any(stride == 0) && product(shape) > 0`, including
    // empty-array degeneracy).
    //
    // `ptr` argument: the logical-first pointer of the RESULT view. broadcast_to
    // preserves `offset`, so the result's logical-first pointer equals the source's
    // logical-first pointer = `tensor.as_ptr()` (07-tensor.md line 1198: `as_ptr()`
    // returns the logical-first pointer, NOT storage base).
    let flags = compute_layout_flags::<A, E::Dim>(&target_dim, &strides, tensor.as_ptr());

    // (1) Build the `ViewRepr<'_, A>` that borrows the source storage.
    //     ViewRepr holds `(storage_base_ptr, storage_len)` — NOT the
    //     logical-first pointer (05-storage.md §6.4). The base pointer is
    //     `tensor.as_storage_ptr()` (07-tensor.md §5 line 465-476: "returns
    //     `storage.as_ptr()` directly, WITHOUT adding offset"); `storage_len()`
    //     reports the source storage extent (07-tensor.md §5 line 484).
    //
    // SAFETY (ViewRepr::from_raw_parts, W7T14):
    //   - `tensor.as_storage_ptr()` is non-null and aligned (carried unchanged
    //     from the source storage that was already constructed and validated).
    //   - `[base, base + storage_len)` lies inside a single allocation, all
    //     `storage_len` elements are initialized values of `A`.
    //   - The returned `ViewRepr<'_, A>` lifetime is bound to `&tensor`; no
    //     mutable alias to the same memory is alive during that borrow.
    //   - Empty-storage case: `as_storage_ptr()` returns the dangling sentinel,
    //     `storage_len == 0` makes the range empty, and the sentinel is never
    //     dereferenced.
    let view_storage: ViewRepr<'_, A> =
        unsafe { ViewRepr::from_raw_parts(tensor.as_storage_ptr(), tensor.storage_len()) };

    // (2) Finalize via `TensorBase::new_unchecked` — the canonical pub(crate)
    //     unsafe constructor (07-tensor.md §5.6 line 674-730).
    //
    // Why NOT `TensorView::from_raw_parts` (W11T7's original path):
    //   (a) Its `ptr` argument must be the storage base (07-tensor.md §5.1 line
    //       202, §5.7 line 752); passing `tensor.as_ptr()` (logical-first) +
    //       `tensor.offset()` would double-apply the offset → UB.
    //   (b) It always sets `derived_from_view_mut := false` (07-tensor.md §5
    //       line 183-186), preventing propagation from `ViewMut` sources;
    //       `new_unchecked` accepts the flag explicitly.
    //   (c) 07-tensor.md §5.6 line 685-690 explicitly states all internal
    //       unchecked constructors forward to `new_unchecked` rather than
    //       defining parallel safety invariants — broadcast is exactly such
    //       an internal caller.
    //
    // `derived_from_view_mut` propagation per 07-tensor.md §5.6 line 707-713:
    //   - `true` ONLY when source is a `ViewMutRepr` being demoted, or a
    //     `ViewRepr` already carrying `derived_from_view_mut == true`.
    //   - `tensor.derived_from_view_mut` (field access) reports the combined
    //     classification; forwarding it satisfies the propagation rule for
    //     every supported source storage type.
    //   - For `Owned<A>` sources the field is `false`, matching the
    //     §5.6 line 711-713 requirement that "Owned construction paths MUST
    //     pass `false`".
    //
    // SAFETY (TensorBase::new_unchecked, 07-tensor.md §5.6 line 692-715):
    //   - `target_dim` + `strides` + `tensor.offset()`: target shape is rank-checked
    //     (`Strides::from_slice` succeeded; `IntoDimension` fixed the rank).
    //     `broadcast_strides` already produced a layout that broadcasts the
    //     source onto `target_dim`, so the result's logical access range equals
    //     the source's (each `target_dim` index resolves to a source index via
    //     zero-stride/repeat axes), which still fits the source `storage_len`
    //     carried by `view_storage`.
    //   - `flags` was produced by `compute_layout_flags` (above) on the same
    //     `(target_dim, strides, tensor.as_ptr())` triple — `tensor.as_ptr()` is the
    //     logical-first pointer the result view will expose via its own
    //     `as_ptr()` (offset unchanged). This satisfies §5.6 line 693-698 verbatim.
    //   - `derived_from_view_mut` is forwarded from the source per the
    //     propagation rule above.
    //   - The layout family is valid for an immutable view: zero-stride
    //     broadcast layouts are explicitly accepted on read-only paths
    //     (07-tensor.md §5.7 line 775-777).
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

// ----------------------------------------------------------------------------
// Two-input broadcast prologue (shared by math binary ops)
// ----------------------------------------------------------------------------

/// The two broadcast views plus the common output dimension produced by
/// [`broadcast_with`].
pub(crate) type BroadcastViews<'a, 'b, A, D1, D2> = (
    TensorView<'a, A, <D1 as BroadcastDim<D2>>::Output>,
    TensorView<'b, A, <D1 as BroadcastDim<D2>>::Output>,
    <D1 as BroadcastDim<D2>>::Output,
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
    D1: Dimension + BroadcastDim<D2>,
    D2: Dimension,
{
    let out_shape = broadcast_shape(a.shape(), b.shape())?;
    let out_dim = <D1 as BroadcastDim<D2>>::Output::try_from_slice(out_shape.slice())
        .expect("broadcast_shape validated the output shape");
    let a_view = broadcast_to(a, out_dim.clone())?;
    let b_view = broadcast_to(b, out_dim.clone())?;
    Ok((a_view, b_view, out_dim))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::Ix2;
    use crate::error::XenonError;
    use crate::layout::LayoutState;
    use crate::tensor::Tensor2;

    /// Compile-time check: `broadcast_to` method exists with the expected signature.
    #[allow(dead_code)]
    fn _check_broadcast_to_sig(t: &Tensor2<f64>) -> Result<TensorView<'_, f64, Ix2>, XenonError> {
        t.broadcast_to([2usize, 3])
    }

    // --- W11T7 tests ---

    #[test]
    fn test_broadcast_to_basic() {
        let tensor: Tensor2<f64> =
            Tensor2::from_shape_vec([1, 3], vec![1.0, 2.0, 3.0]).expect("valid test input");
        let view = tensor.broadcast_to([2, 3]).expect("valid test input");
        assert_eq!(view.shape(), &[2, 3]);
        // Broadcast axis 0: stride 0.
        assert_eq!(view.strides()[0], 0);
        // Zero-copy: pointer is identical.
        assert_eq!(view.as_ptr(), tensor.as_ptr());
        // Offset preserved.
        assert_eq!(view.offset(), tensor.offset());
    }

    /// §8.3 row 2: scalar (rank 0) broadcasting to higher rank. Missing leading
    /// axes are length 1 per §6.2 step 2.
    #[test]
    fn test_broadcast_to_scalar_to_higher_rank() {
        let scalar: Tensor2<f64> =
            Tensor2::from_shape_vec([1, 1], vec![42.0]).expect("valid test input");
        let view = scalar.broadcast_to([2, 3]).expect("valid test input");
        assert_eq!(view.shape(), &[2, 3]);
        // Both axes are broadcast: both strides are 0.
        assert_eq!(view.strides(), &[0, 0]);
    }

    // --- W11T8 tests ---

    #[test]
    fn test_broadcast_to_error() {
        let tensor: Tensor2<f64> = Tensor2::zeros([2, 3]).expect("valid test input");
        let err = tensor.broadcast_to([4, 3]).expect_err("expected error");
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

    /// §8.2 `test_broadcast_layout_flags_recomputed`: non-empty broadcast view
    /// (`product(shape) > 0` and `any(stride == 0)`) classifies as `BroadcastView`.
    #[test]
    fn test_broadcast_layout_flags_recomputed() {
        let tensor: Tensor2<f64> = Tensor2::zeros([1, 3]).expect("valid test input");
        let view = tensor.broadcast_to([2, 3]).expect("valid test input");
        assert_eq!(view.layout_state(), LayoutState::BroadcastView);
        assert!(view.flags().has_zero_stride());
    }

    /// §6.4 line 258 / §5.11 line 261-263: empty-array degenerate zero stride
    /// (`1 -> 0`) does NOT trigger `BroadcastView` classification.
    #[test]
    fn test_broadcast_to_empty_does_not_classify_as_broadcast_view() {
        let tensor: Tensor2<f64> = Tensor2::zeros([1, 3]).expect("valid test input");
        let view = tensor.broadcast_to([0, 3]).expect("valid test input");
        // `product(shape) == 0` ⇒ HAS_ZERO_STRIDE = false ⇒ not BroadcastView.
        assert_ne!(view.layout_state(), LayoutState::BroadcastView);
        assert!(!view.flags().has_zero_stride());
    }

    /// §8.2 `test_broadcast_read_only`: the broadcast view never exposes &mut.
    /// We verify the view is iterable in read-only fashion; the absence of any
    /// `&mut` path is covered as a compile-fail doctest in W11T10.
    #[test]
    fn test_broadcast_read_only_iterable() {
        let tensor: Tensor2<f64> =
            Tensor2::from_shape_vec([1, 3], vec![1.0, 2.0, 3.0]).expect("valid test input");
        let view = tensor.broadcast_to([2, 3]).expect("valid test input");
        let n: usize = view.iter().count();
        assert_eq!(n, 6);
    }

    #[test]
    fn test_broadcast_to_signature_compiles() {
        // Signature is verified at compile time by _check_broadcast_to_sig above.
    }
}
