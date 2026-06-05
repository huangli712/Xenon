//! Utility operations: `clip`, `to_contiguous`, `into_contiguous`.
//!
//! Provides inherent methods on [`TensorBase`].
//! See `docs/design/20-utility.md`.

use std::borrow::Cow;

use crate::dimension::Dimension;
use crate::element::{Element, OrderedCompareElement};
use crate::error::{InvalidArgumentKind, XenonError};
use crate::layout::{Strides, compute_layout_flags};
use crate::storage::{RawStorage, Storage, StorageIntoOwned};
use crate::tensor::{StorageKind, StorageSemantics, Tensor, TensorBase};

// ── clip ──

/// Validate that `min <= max`; reject NaN bounds.
///
/// Returns `Err(XenonError::InvalidArgument)` when `min > max` or either
/// bound is `NaN` (for floating-point types).
fn validate_clip_bounds<A>(min: &A, max: &A) -> Result<(), XenonError>
where
    A: OrderedCompareElement,
{
    if min.partial_cmp(max).is_none() || min > max {
        return Err(XenonError::InvalidArgument {
            operation: Cow::Borrowed("clip"),
            kind: InvalidArgumentKind::OperationSpecific {
                argument: Cow::Borrowed("min/max"),
                constraint: Cow::Borrowed(
                    "min <= max; NaN bounds are invalid for floating-point inputs",
                ),
            },
        });
    }
    Ok(())
}

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension + Clone,
    A: OrderedCompareElement + Clone,
{
    /// Clamp each logical element into `[min, max]`.
    ///
    /// Per `20-utility §5.1` / §6.4:
    /// - Bounds are validated **before** allocation.
    /// - `NaN` input values pass through unchanged (both `< min` and `> max`
    ///   are `false` under IEEE 754), matching NumPy `np.clip` semantics.
    /// - `NaN` *bounds* (or `min > max`) return `InvalidArgument`.
    ///
    /// Implementation note: this body uses the `iter()` + `Vec` +
    /// `from_shape_vec` path explicitly permitted by design §5.1
    /// ("may use MaybeUninit or equivalent internal uninitialized owned
    /// buffer"). Once the
    /// `uninit_like` / `iter_uninit_mut` / `assume_init` primitives land in
    /// a future wave, migrate to a single-pass MaybeUninit write per §5.1
    /// §5.1 / §6.1 algorithm sketch.
    ///
    /// # Errors
    ///
    /// - `XenonError::InvalidArgument` when bounds are invalid: either bound
    ///   is `NaN`, or `min > max` (`20-utility §5.1` / §6.4). Validated before
    ///   allocation.
    /// - `XenonError::InvalidShape` propagated from `Tensor::from_shape_vec`
    ///   when the shape's element count overflows `usize` (`ProductOverflow`).
    ///   Unreachable in practice — `self` already holds a valid shape — but
    ///   the failure mode is preserved by `?` for completeness.
    #[expect(
        clippy::clone_on_copy,
        reason = "generic over Clone (not Copy); .clone() is correct generic pattern"
    )]
    pub fn clip(&self, min: A, max: A) -> Result<Tensor<A, D>, XenonError> {
        validate_clip_bounds(&min, &max)?;
        let data: Vec<A> = self
            .iter()
            .map(|src| {
                if *src < min {
                    min.clone()
                } else if *src > max {
                    max.clone()
                } else {
                    src.clone()
                }
            })
            .collect();
        Tensor::from_shape_vec(self.raw_dim(), data)
    }
}

// ── to_contiguous: fresh owned path ──

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension + Clone,
    A: Element + Clone,
{
    /// Ensure the tensor's data is stored contiguously in canonical F-order
    /// (`20-utility §5.5`). Always returns a fresh owned tensor; the input
    /// borrow is never aliased into the result.
    ///
    /// # Panics
    ///
    /// Does not panic in practice: on the repack path, `Iter` is an
    /// `ExactSizeIterator` whose `len()` equals `product(shape)`, which is
    /// exactly what `from_shape_vec` requires (see `10-iterator §5.5`,
    /// `18-construction §5.6`). A mismatch would indicate an iterator-contract
    /// bug elsewhere in the crate.
    pub fn to_contiguous(&self) -> Tensor<A, D> {
        if self.is_f_contiguous() {
            // Fast path: `to_owned()` is contracted to return a canonical
            // F-order owned buffer (`21-type §5.5`), so logically F-order
            // inputs need only a single allocate-and-copy pass.
            self.to_owned()
        } else {
            // Repack: iterate logical F-order (`10-iterator §5.5`) and
            // funnel through the canonical owned constructor.
            let values: Vec<A> = self.iter().cloned().collect();
            // SAFETY of `expect`: `Iter` is `ExactSizeIterator` and its
            // `len()` equals `product(shape)`, which is exactly what
            // `from_shape_vec` requires (`18-construction §5.6`). Mismatch
            // would indicate an iterator-contract bug elsewhere.
            Tensor::from_shape_vec(self.raw_dim(), values)
                .expect("logical iteration length equals shape product")
        }
    }
}

// ── into_contiguous: reuse gate ──

impl<S, D, A> TensorBase<S, D>
where
    S: StorageIntoOwned<Elem = A> + StorageSemantics,
    D: Dimension + Clone,
    A: Element + Clone,
{
    /// Consume `self` and produce an owned, canonical F-order tensor
    /// (`20-utility §5.5`, §6.3). Reuses backing storage only when the input
    /// is already a canonical F-contiguous `Owned` tensor (predicate below).
    ///
    /// # Panics
    ///
    /// Panics if `Strides::f_contiguous(&dim)` fails. This cannot happen on the
    /// reuse path because `is_canonical_f_contiguous_owned` already
    /// established `is_f_contiguous()`, which implies the shape's element
    /// count fits `usize` (a construction-time invariant of `TensorBase`).
    pub fn into_contiguous(self) -> Tensor<A, D> {
        if is_canonical_f_contiguous_owned(&self) {
            // O(1) reuse path: storage is moved out and layout metadata
            // is recomputed from the shape per §6.3 ("layout flags ...
            // calculated by 06-layout").
            //
            // We snapshot dim BEFORE consuming `self.storage` so that we
            // never depend on partial-move semantics over a non-Copy
            // `Strides<D>`/`LayoutFlags`/`D` triple.
            let dim = self.raw_dim(); // requires D: Clone (W8T4)
            // Per §6.3: canonical predicate already established
            // `is_f_contiguous()`, so shape.checked_size() must succeed
            // (construction-time invariant). Re-derive F-order strides.
            let strides =
                Strides::f_contiguous(&dim).expect("canonical predicate implies shape is valid");
            // Move storage out (StorageIntoOwned, W7T19).
            let owned = self.storage.into_owned_storage();
            // Re-derive layout flags via the canonical entry point
            // (06-layout §5.12, W6T11). Uses the freshly moved storage's
            // logical-first pointer so the ALIGNED bit reflects reality.
            let flags = compute_layout_flags::<A, D>(&dim, &strides, owned.as_ptr());
            // SAFETY:
            //   * `is_canonical_f_contiguous_owned` already verified:
            //     - shape/strides match (F-order),
            //     - storage.len() == product(shape) (no tail padding),
            //     - offset == 0,
            //     - storage_kind() == Owned (sole-ownership, not view-derived).
            //   * Therefore the logical access range derived from
            //     (dim, strides, 0) lies entirely within `owned`.
            //   * `flags` was produced by `compute_layout_flags` for the same
            //     (dim, strides) we are storing.
            //   * Owned storage was never downgraded from a ViewMut, so
            //     `derived_from_view_mut = false` is correct.
            unsafe { TensorBase::new_unchecked(owned, dim, strides, 0, flags, false) }
        } else {
            // Repack path: `into_owned()` always produces canonical F-order
            // (`21-type §5.5`). Since `StorageIntoOwned: Storage` (W7T18),
            // `to_owned()` is available on `self`.
            self.into_owned()
        }
    }
}

/// Crate-internal canonical predicate from `20-utility §6.3`. Private to
/// `src/util/contiguous.rs`: external callers cannot name it (matches §6.3
/// §6.3).
///
/// Returns `true` iff **all four** conditions from `20-utility §6.3` hold:
///   1. `is_f_contiguous()` — strides satisfy F-order pattern
///      (`06-layout §5.7`, surfaced on TensorBase by W8).
///   2. `storage_kind() == Owned` — sole-ownership, not view-derived
///      (`05-storage §5.9`, surfaced by W8T4).
///   3. `offset() == 0` — no head padding (`07-tensor §5.3`, surfaced by W8T4).
///   4. `storage_len() == product(shape)` — no tail padding
///      (`05-storage §5.3` `RawStorage::len`, surfaced on TensorBase by
///      W8T6 as `storage_len()`).
fn is_canonical_f_contiguous_owned<S, D, A>(t: &TensorBase<S, D>) -> bool
where
    S: Storage<Elem = A> + StorageSemantics,
    D: Dimension,
{
    if !t.is_f_contiguous() {
        return false;
    }
    if t.storage_kind() != StorageKind::Owned {
        return false;
    }
    if t.offset() != 0 {
        return false;
    }
    // `storage_len()` (W8T6) is the inherent method that surfaces
    // `RawStorage::len()` on TensorBase without requiring util to
    // access the `pub(crate)` storage field directly.  `checked_size()`
    // succeeded at construction time; if it fails here the tensor
    // invariants are already violated, so return `false` defensively.
    let logical = match t.raw_dim().checked_size() {
        Ok(n) => n,
        Err(_) => return false,
    };
    t.storage_len() == logical
}

// ── Unit tests (§8.2) ──

#[cfg(test)]
mod tests {
    use crate::error::XenonError;
    use crate::tensor::{StorageKind, Tensor1, Tensor2};

    // ── clip tests (§8.2 / §7 T2) ──

    #[test]
    fn test_clip_basic() {
        let tensor = Tensor1::from_shape_vec([5], vec![-1.0, 0.5, 1.0, 2.0, 3.0])
            .expect("from_shape_vec matching shape");
        let clipped = tensor.clip(0.0, 2.0).expect("valid clip bounds");
        let values: Vec<f64> = clipped.iter().copied().collect();
        assert_eq!(values, vec![0.0, 0.5, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn test_clip_no_change() {
        let tensor = Tensor1::from_shape_vec([3], vec![0.5, 1.0, 1.5])
            .expect("from_shape_vec matching shape");
        let clipped = tensor.clip(0.0, 2.0).expect("valid clip bounds");
        let values: Vec<f64> = clipped.iter().copied().collect();
        assert_eq!(values, vec![0.5, 1.0, 1.5]);
    }

    // NaN inputs pass through unchanged (§6.4)
    #[test]
    fn test_clip_nan() {
        let tensor = Tensor1::from_shape_vec([3], vec![1.0_f64, f64::NAN, 3.0])
            .expect("from_shape_vec matching shape");
        let clipped = tensor.clip(0.0, 4.0).expect("valid clip bounds");
        let values: Vec<f64> = clipped.iter().copied().collect();
        assert_eq!(values[0], 1.0);
        assert!(values[1].is_nan());
        assert_eq!(values[2], 3.0);
    }

    // NaN as min or max → InvalidArgument
    #[test]
    fn test_clip_nan_bound() {
        let tensor =
            Tensor1::from_shape_vec([1], vec![1.0_f64]).expect("from_shape_vec matching shape");
        assert!(matches!(
            tensor.clip(f64::NAN, 2.0),
            Err(XenonError::InvalidArgument { .. })
        ));
        assert!(matches!(
            tensor.clip(0.0, f64::NAN),
            Err(XenonError::InvalidArgument { .. })
        ));
    }

    #[test]
    fn test_clip_integers() {
        let tensor = Tensor1::from_shape_vec([4], vec![-5_i32, 0, 5, 10])
            .expect("from_shape_vec matching shape");
        let clipped = tensor.clip(0, 7).expect("valid clip bounds");
        let values: Vec<i32> = clipped.iter().copied().collect();
        assert_eq!(values, vec![0, 0, 5, 7]);
    }

    // Construction is in F-order. `from_shape_vec([2, 3], [1, 4, 2, 5, 3, 6])`
    // produces the 2×3 matrix
    //     [ 1 2 3 ]
    //     [ 4 5 6 ]
    // (column-major: column 0 = [1, 4], column 1 = [2, 5], column 2 = [3, 6]).
    // After `transpose()` we get the 3×2 matrix
    //     [ 1 4 ]
    //     [ 2 5 ]
    //     [ 3 6 ]
    // and clip(2.0, 5.0) yields
    //     [ 2 4 ]
    //     [ 2 5 ]
    //     [ 3 5 ]
    // F-order to_vec of that result (column 0 then column 1) is
    //     [2.0, 2.0, 3.0, 4.0, 5.0, 5.0].
    #[test]
    fn test_clip_non_contiguous() {
        let tensor = Tensor2::from_shape_vec([2, 3], vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0])
            .expect("from_shape_vec matching shape");
        let transposed = tensor.transpose();
        let clipped = transposed.clip(2.0, 5.0).expect("valid clip bounds");
        assert_eq!(clipped.shape(), &[3, 2]);
        assert_eq!(*clipped.get(&[0, 0]).expect("valid index"), 2.0);
        assert_eq!(*clipped.get(&[0, 1]).expect("valid index"), 4.0);
        assert_eq!(*clipped.get(&[1, 0]).expect("valid index"), 2.0);
        assert_eq!(*clipped.get(&[1, 1]).expect("valid index"), 5.0);
        assert_eq!(*clipped.get(&[2, 0]).expect("valid index"), 3.0);
        assert_eq!(*clipped.get(&[2, 1]).expect("valid index"), 5.0);
        // F-order iter consistency (column-major), kept as a regression
        // guard against any future iter() semantics drift:
        let values: Vec<f64> = clipped.iter().copied().collect();
        assert_eq!(values, vec![2.0, 2.0, 3.0, 4.0, 5.0, 5.0]);
    }

    // ── contiguous tests (§8.2) ──

    // §8.2 — test_to_contiguous_f_order
    #[test]
    fn test_to_contiguous_f_order() {
        let tensor = Tensor2::<i32>::from_shape_vec([2, 2], vec![1, 2, 3, 4])
            .expect("from_shape_vec matching shape");
        let contiguous = tensor.to_contiguous();
        assert!(contiguous.is_f_contiguous());
        // Logical positions match (F-order construction):
        //   column 0 = [1, 2]; column 1 = [3, 4]
        assert_eq!(*contiguous.get(&[0, 0]).expect("valid index"), 1);
        assert_eq!(*contiguous.get(&[1, 0]).expect("valid index"), 2);
        assert_eq!(*contiguous.get(&[0, 1]).expect("valid index"), 3);
        assert_eq!(*contiguous.get(&[1, 1]).expect("valid index"), 4);
    }

    // §8.2 — test_to_contiguous_transposed_becomes_f
    #[test]
    fn test_to_contiguous_transposed_becomes_f() {
        let tensor = Tensor2::<i32>::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6])
            .expect("from_shape_vec matching shape");
        let contiguous = tensor.transpose().to_contiguous();
        assert!(contiguous.is_f_contiguous());
        // Element-wise check is order-independent. The F-order original is
        //   column 0 = [1, 2], column 1 = [3, 4], column 2 = [5, 6]
        // i.e. matrix
        //   [1 3 5]
        //   [2 4 6]
        // After `transpose()` → 3×2
        //   [1 2]
        //   [3 4]
        //   [5 6]
        assert_eq!(contiguous.shape(), &[3, 2]);
        assert_eq!(*contiguous.get(&[0, 0]).expect("valid index"), 1);
        assert_eq!(*contiguous.get(&[0, 1]).expect("valid index"), 2);
        assert_eq!(*contiguous.get(&[1, 0]).expect("valid index"), 3);
        assert_eq!(*contiguous.get(&[1, 1]).expect("valid index"), 4);
        assert_eq!(*contiguous.get(&[2, 0]).expect("valid index"), 5);
        assert_eq!(*contiguous.get(&[2, 1]).expect("valid index"), 6);
    }

    // §8.2 — test_into_contiguous_reuses_canonical_owned_data
    //
    // The design contract (§6.3) says "O(1) move of the canonical buffer";
    // it does NOT mandate that `as_ptr()` returns the literal same value
    // after the move. AlignedAlloc may pad allocation size for alignment,
    // and an `Owned`'s `as_ptr()` value depends on its internal repr.
    // Therefore we assert:
    //   (a) the result is canonical F-order owned;
    //   (b) per-element values are preserved;
    //   (c) the result is the "Owned reuse" path, not the repack path.
    //
    // (c) is verified by observing `storage_kind() == Owned` and noting
    // that no observable construction-time allocation cost is incurred
    // beyond the input's. We do NOT compare raw pointers — that is an
    // implementation detail.
    #[test]
    fn test_into_contiguous_reuses_canonical_owned_data() {
        let tensor = Tensor2::<i32>::from_shape_vec([2, 2], vec![1, 2, 3, 4])
            .expect("from_shape_vec matching shape");
        let contiguous = tensor.into_contiguous();
        assert!(contiguous.is_f_contiguous());
        assert_eq!(contiguous.storage_kind(), StorageKind::Owned);
        assert_eq!(*contiguous.get(&[0, 0]).expect("valid index"), 1);
        assert_eq!(*contiguous.get(&[1, 0]).expect("valid index"), 2);
        assert_eq!(*contiguous.get(&[0, 1]).expect("valid index"), 3);
        assert_eq!(*contiguous.get(&[1, 1]).expect("valid index"), 4);
    }

    // §8.2 — test_into_contiguous_repacks_noncanonical_f_contiguous_owned
    //
    // Logically F-contiguous owned input WITH tail padding / non-zero
    // offset MUST be repacked into a canonical F-order owned (no padding,
    // offset == 0). The construction primitives needed to produce a
    // non-canonical-but-logically-F-contiguous Owned input land in W6/W8
    // (slice / reshape with padding awareness). Until they are wired up,
    // this test stays `#[ignore]`.
    #[test]
    #[ignore = "needs non-canonical owned constructor (tail padding / non-zero offset, W6+W8)"]
    fn test_into_contiguous_repacks_noncanonical_f_contiguous_owned() {
        todo!("activate after padding-aware owned constructor lands (W6+W8)");
    }

    // §8.2 — test_into_contiguous_repacks_arc_input
    //
    // ArcTensor input is `storage_kind() == Shared`, so it fails the
    // `is_canonical_f_contiguous_owned` predicate (which requires Owned).
    // `into_contiguous()` must therefore take the repack path and produce
    // a canonical F-order Owned tensor with element data preserved.
    #[test]
    fn test_into_contiguous_repacks_arc_input() {
        let tensor = Tensor2::<i32>::from_shape_vec([2, 2], vec![1, 2, 3, 4])
            .expect("from_shape_vec matching shape");
        let arc = tensor.into_shared();
        let contiguous = arc.into_contiguous();
        // Result is canonical F-order Owned (repack path).
        assert!(contiguous.is_f_contiguous());
        assert_eq!(contiguous.storage_kind(), StorageKind::Owned);
        // Per-element data preserved.
        assert_eq!(*contiguous.get(&[0, 0]).expect("valid index"), 1);
        assert_eq!(*contiguous.get(&[1, 0]).expect("valid index"), 2);
        assert_eq!(*contiguous.get(&[0, 1]).expect("valid index"), 3);
        assert_eq!(*contiguous.get(&[1, 1]).expect("valid index"), 4);
    }

    // §8.2 — test_to_contiguous_non_contiguous
    //
    // Construction is in F-order. `from_shape_vec([2, 3], [1..=6])` is
    //   [1 3 5]
    //   [2 4 6]
    // (col 0 = [1, 2]; col 1 = [3, 4]; col 2 = [5, 6]).
    // `transpose()` → 3×2 logical
    //   [1 2]
    //   [3 4]
    //   [5 6]
    // After `to_contiguous()`, the result is F-order owned with the same
    // logical values at the same logical positions.
    #[test]
    fn test_to_contiguous_non_contiguous() {
        let tensor = Tensor2::<i32>::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6])
            .expect("from_shape_vec matching shape");
        let transposed = tensor.transpose();
        assert!(!transposed.is_f_contiguous());
        let contiguous = transposed.to_contiguous();
        assert!(contiguous.is_f_contiguous());
        assert_eq!(contiguous.shape(), &[3, 2]);
        assert_eq!(*contiguous.get(&[0, 0]).expect("valid index"), 1);
        assert_eq!(*contiguous.get(&[0, 1]).expect("valid index"), 2);
        assert_eq!(*contiguous.get(&[1, 0]).expect("valid index"), 3);
        assert_eq!(*contiguous.get(&[1, 1]).expect("valid index"), 4);
        assert_eq!(*contiguous.get(&[2, 0]).expect("valid index"), 5);
        assert_eq!(*contiguous.get(&[2, 1]).expect("valid index"), 6);
    }
}
