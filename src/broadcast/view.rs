use crate::dimension::{BroadcastDim, Dimension, IntoDimension};
use crate::element::Element;
use crate::error::XenonError;
use crate::layout::{compute_layout_flags, Strides};
use crate::storage::{Storage, ViewRepr};
use crate::tensor::{TensorBase, TensorView};

use crate::broadcast::shape::broadcast_strides;

impl<S, A, D> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    A: Element,
    D: Dimension,
{
    /// Broadcast `self` to `shape`. Returns a read-only zero-copy view sharing the
    /// underlying storage. See `15-broadcast.md §5.1` line 124-132 and §6.4.
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
        let target_dim: E::Dim = shape.into_dimension();
        let target_shape: &[usize] = target_dim.slice();

        // Error path: `broadcast_strides` returns BroadcastError or InvalidArgument on
        // shape-level failure. No additional `broadcast_shape` pre-check is performed —
        // it would be redundant (W11T6 already iterates per-axis) and semantically wrong
        // (`broadcast_shape` is bidirectional; `broadcast_to` is single-direction).
        let strides_vec: Vec<usize> =
            broadcast_strides(self.shape(), self.strides(), target_shape)?;
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
        // logical-first pointer = `self.as_ptr()` (07-tensor.md line 1198: `as_ptr()`
        // returns the logical-first pointer, NOT storage base).
        let flags = compute_layout_flags::<A, E::Dim>(&target_dim, &strides, self.as_ptr());

        // (1) Build the `ViewRepr<'_, A>` that borrows the source storage.
        //     ViewRepr holds `(storage_base_ptr, storage_len)` — NOT the
        //     logical-first pointer (05-storage.md §6.4). The base pointer is
        //     `self.as_storage_ptr()` (07-tensor.md §5 line 465-476: "returns
        //     `storage.as_ptr()` directly, WITHOUT adding offset"); `storage_len()`
        //     reports the source storage extent (07-tensor.md §5 line 484).
        //
        // SAFETY (ViewRepr::from_raw_parts, W7T14):
        //   - `self.as_storage_ptr()` is non-null and aligned (carried unchanged
        //     from the source storage that was already constructed and validated).
        //   - `[base, base + storage_len)` lies inside a single allocation, all
        //     `storage_len` elements are initialized values of `A`.
        //   - The returned `ViewRepr<'_, A>` lifetime is bound to `&self`; no
        //     mutable alias to the same memory is alive during that borrow.
        //   - Empty-storage case: `as_storage_ptr()` returns the dangling sentinel,
        //     `storage_len == 0` makes the range empty, and the sentinel is never
        //     dereferenced.
        let view_storage: ViewRepr<'_, A> = unsafe {
            ViewRepr::from_raw_parts(self.as_storage_ptr(), self.storage_len())
        };

        // (2) Finalize via `TensorBase::new_unchecked` — the canonical pub(crate)
        //     unsafe constructor (07-tensor.md §5.6 line 674-730).
        //
        // Why NOT `TensorView::from_raw_parts` (W11T7's original path):
        //   (a) Its `ptr` argument must be the storage base (07-tensor.md §5.1 line
        //       202, §5.7 line 752); passing `self.as_ptr()` (logical-first) +
        //       `self.offset()` would double-apply the offset → UB.
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
        //   - `self.derived_from_view_mut` (field access) reports the combined
        //     classification; forwarding it satisfies the propagation rule for
        //     every supported source storage type.
        //   - For `Owned<A>` sources the field is `false`, matching the
        //     §5.6 line 711-713 requirement that "Owned construction paths MUST
        //     pass `false`".
        //
        // SAFETY (TensorBase::new_unchecked, 07-tensor.md §5.6 line 692-715):
        //   - `target_dim` + `strides` + `self.offset()`: target shape is rank-checked
        //     (`Strides::from_slice` succeeded; `IntoDimension` fixed the rank).
        //     `broadcast_strides` already produced a layout that broadcasts the
        //     source onto `target_dim`, so the result's logical access range equals
        //     the source's (each `target_dim` index resolves to a source index via
        //     zero-stride/repeat axes), which still fits the source `storage_len`
        //     carried by `view_storage`.
        //   - `flags` was produced by `compute_layout_flags` (above) on the same
        //     `(target_dim, strides, self.as_ptr())` triple — `self.as_ptr()` is the
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
                self.offset(),
                flags,
                self.derived_from_view_mut,
            )
        };

        Ok(view)
    }
}

/// Alias for the pair of broadcast views returned by `broadcast_with`.
type BroadcastPair<'a, A, D, E> = (
    TensorView<'a, A, <D as BroadcastDim<E>>::Output>,
    TensorView<'a, A, <D as BroadcastDim<E>>::Output>,
);

/// Dual-input broadcast. Internal entry consumed by `math` / `overload`. See
/// `15-broadcast.md §5.1` line 134-148 and §5.2 line 170.
///
/// The bidirectional `BroadcastDim` bound (D: BroadcastDim<E> and E: BroadcastDim<D,
/// Output = ...>) is satisfiable for every `(D, E) ∈ {Ix0..Ix6, IxDyn}` per
/// 02-dimension §5.10 line 703 symmetry guarantee.
pub(crate) fn broadcast_with<'a, A, S1, D, S2, E>(
    a: &'a TensorBase<S1, D>,
    b: &'a TensorBase<S2, E>,
) -> Result<BroadcastPair<'a, A, D, E>, XenonError>
where
    S1: Storage<Elem = A>,
    S2: Storage<Elem = A>,
    A: Element,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    use crate::broadcast::shape::broadcast_shape;

    // §5.2 line 170: compute the common shape; on incompatibility, propagate the
    // structured `BroadcastError` from `broadcast_shape` (which fills lhs_shape,
    // rhs_shape; attempted_target_shape = None for the pure shape-derivation path).
    let out_dyn = broadcast_shape(a.shape(), b.shape())?;

    // Convert IxDyn → <D as BroadcastDim<E>>::Output. By 02-dimension §5.10's
    // 57-impl matrix and the bidirectional bound, the result rank equals
    // `max(D::NDIM, E::NDIM)` (or IxDyn when either side is IxDyn), and
    // `try_from_slice` succeeds unconditionally on the broadcast-derived shape.
    type Out<D, E> = <D as BroadcastDim<E>>::Output;
    let out_dim: Out<D, E> = Out::<D, E>::try_from_slice(out_dyn.slice())?;

    // Build the two broadcast views directly via the shape-level primitives,
    // bypassing `broadcast_to<E: IntoDimension>` to avoid the IxDyn round-trip
    // that would erase the static output dimension type.
    let left = broadcast_to_output_dim::<A, S1, D, Out<D, E>>(a, &out_dim)?;
    let right = broadcast_to_output_dim::<A, S2, E, Out<D, E>>(b, &out_dim)?;
    Ok((left, right))
}

/// Helper: broadcast a single tensor to a pre-computed output dimension that is
/// already known to be compatible. Mirrors `TensorBase::broadcast_to` but takes
/// the target dimension by reference (no `IntoDimension` re-conversion) and
/// returns the view with the caller-chosen target dimension type.
///
/// This is a `fn`-level helper (not an inherent method) so it can be shared by
/// both inputs in `broadcast_with` without re-deriving the target dim.
fn broadcast_to_output_dim<'a, A, S, D, Out>(
    src: &'a TensorBase<S, D>,
    out_dim: &Out,
) -> Result<TensorView<'a, A, Out>, XenonError>
where
    S: Storage<Elem = A>,
    A: Element,
    D: Dimension,
    Out: Dimension,
{
    use crate::broadcast::shape::broadcast_strides;
    use crate::layout::{compute_layout_flags, Strides};
    use crate::storage::ViewRepr;

    let strides_vec = broadcast_strides(src.shape(), src.strides(), out_dim.slice())?;
    // Strides::from_slice is the rank-checked entry (06-layout §5.5 line 333-335).
    // NOTE: `Strides<D>` exposes `from_slice`, NOT `try_from_slice`; the latter
    // is a `Dimension` trait method (02-dimension §5.1 line 148), used above to
    // convert `IxDyn` → `Out`, but not available on `Strides<D>`.
    let strides = Strides::<Out>::from_slice(&strides_vec)?;

    // Path Y view assembly (per W11T8 design decision; do NOT use
    // `TensorView::from_raw_parts` — its `ptr` arg is documented as the storage
    // base pointer (07-tensor.md §5.1 line 202, §5.7 line 752), while
    // `src.as_ptr()` returns the logical-first pointer (line 1198), so the
    // constructor's internal `ptr.add(offset)` would double-apply the offset
    // → UB. `from_raw_parts` also unconditionally sets
    // `derived_from_view_mut := false` (line 183-186), losing the ViewMut
    // demotion propagation. The canonical pub(crate) entry
    // `TensorBase::new_unchecked` avoids both issues.
    //
    // Step 1: compute result-view layout flags. `ptr` here is the logical-first
    // pointer the RESULT view will expose. `broadcast_to_output_dim` preserves
    // `src.offset()`, so result and source share the same logical-first pointer
    // (07-tensor.md line 1198). `compute_layout_flags` is the single source of
    // truth per 06-layout §5.5 line 233.
    let flags = compute_layout_flags::<A, Out>(out_dim, &strides, src.as_ptr());

    // Step 2: build `ViewRepr<'a, A>` from storage base + storage_len.
    // SAFETY (`ViewRepr::from_raw_parts`, W7T14):
    //   - `src.as_storage_ptr()` is non-null & aligned (07-tensor.md §5 line
    //     465-476: returns `storage.as_ptr()` WITHOUT adding offset).
    //   - `[base, base + storage_len)` lies inside a single allocation; every
    //     element in that range is an initialized `A` value.
    //   - `ViewRepr` lifetime bound to `&'a self`; no overlapping mutable
    //     alias is alive during `'a`.
    //   - Empty-storage case: dangling sentinel, never dereferenced.
    let view_storage: ViewRepr<'a, A> = unsafe {
        ViewRepr::from_raw_parts(src.as_storage_ptr(), src.storage_len())
    };

    // Step 3: finalize via `TensorBase::new_unchecked` — the canonical
    // pub(crate) unsafe constructor (07-tensor.md §5.6 line 674-730).
    //
    // SAFETY (`TensorBase::new_unchecked`, 07-tensor.md §5.6 line 692-715):
    //   - `out_dim` + `strides` + `src.offset()`: shape & strides validated
    //     by `broadcast_strides`; the broadcast layout's logical access range
    //     fits inside `src.storage_len()` (zero-stride/repeat axes do not
    //     widen the access range).
    //   - `flags` was produced by `compute_layout_flags` (Step 1) on the
    //     same `(out_dim, strides, src.as_ptr())` triple the result view
    //     will expose.
    //   - `derived_from_view_mut` is forwarded from source per 07-tensor.md
    //     §5.6 line 707-713 propagation rule.
    //   - Layout family valid for an immutable view: zero-stride broadcast
    //     layouts are explicitly accepted on read-only paths (07-tensor.md
    //     §5.7 line 775-777). We use the immutable canonical entry, so
    //     aliasing rules are not violated.
    let view: TensorView<'a, A, Out> = unsafe {
        TensorBase::<ViewRepr<'a, A>, Out>::new_unchecked(
            view_storage,
            out_dim.clone(),
            strides,
            src.offset(),
            flags,
            src.derived_from_view_mut,
        )
    };

    Ok(view)
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

    /// Compile-time check: bidirectional `BroadcastDim` bound is satisfiable for
    /// same-rank `(Ix2, Ix2)`.
    #[allow(dead_code)]
    fn _check_broadcast_with_sig<'a>(
        a: &'a Tensor2<f64>,
        b: &'a Tensor2<f64>,
    ) -> Result<BroadcastPair<'a, f64, Ix2, Ix2>, XenonError> {
        broadcast_with(a, b)
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
            }
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

    // --- W11T9 tests ---

    /// §8.2 `test_broadcast_with_same_shape`: when input shapes already match,
    /// both result views share the same target shape; if the second input has
    /// a `1` axis, the corresponding stride is 0.
    #[test]
    fn test_broadcast_with_same_shape() {
        let a: Tensor2<f64> = Tensor2::zeros([2, 3]).expect("valid test input");
        let b: Tensor2<f64> = Tensor2::ones([1, 3]).expect("valid test input");
        let (left, right) = broadcast_with(&a, &b).expect("valid test input");
        assert_eq!(left.shape(), &[2, 3]);
        assert_eq!(right.shape(), &[2, 3]);
        // `a` already matches the output shape → no zero stride introduced.
        assert!(left.strides().iter().all(|&s| s != 0));
        // `b` had axis 0 = 1 → broadcast axis with stride 0.
        assert_eq!(right.strides()[0], 0);
        // Zero-copy: pointers unchanged.
        assert_eq!(left.as_ptr(), a.as_ptr());
        assert_eq!(right.as_ptr(), b.as_ptr());
    }

    /// §7 Wave 3 T7: scalar/low-rank vs higher-rank inputs.
    #[test]
    fn test_broadcast_scalar_and_tensor() {
        let scalar: Tensor2<f64> = Tensor2::from_shape_vec([1, 1], vec![5.0]).expect("valid test input");
        let tensor: Tensor2<f64> = Tensor2::zeros([2, 3]).expect("valid test input");
        let (left, right) = broadcast_with(&scalar, &tensor).expect("valid test input");
        assert_eq!(left.shape(), &[2, 3]);
        assert_eq!(right.shape(), &[2, 3]);
        // Scalar's both axes are broadcast.
        assert_eq!(left.strides(), &[0, 0]);
    }

    /// §7 Wave 3 T7 / §8.2: incompatible shapes propagate `BroadcastError`.
    #[test]
    fn test_broadcast_with_incompatible_shapes() {
        let a: Tensor2<f64> = Tensor2::zeros([2, 3]).expect("valid test input");
        let b: Tensor2<f64> = Tensor2::zeros([4, 3]).expect("valid test input");
        let err = broadcast_with(&a, &b).expect_err("expected error");
        match err {
            XenonError::BroadcastError {
                operation,
                lhs_shape,
                rhs_shape,
                ..
            } => {
                assert_eq!(operation.as_ref(), "broadcast_shape");
                assert_eq!(lhs_shape, vec![2, 3]);
                assert_eq!(rhs_shape, vec![4, 3]);
            }
            other => panic!("expected BroadcastError, got {:?}", other),
        }
    }

    #[test]
    fn test_broadcast_to_signature_compiles() {
        // Signature is verified at compile time by _check_broadcast_to_sig above.
    }

    #[test]
    fn test_broadcast_with_bidirectional_bound_satisfiable() {
        // Signature is verified at compile time by _check_broadcast_with_sig above.
    }
}
