//! Fill operation for tensors.
//!
//! Provides `fill()` (in-place) and `try_fill()` (fallible) inherent
//! methods on [`TensorBase`]. See `docs/design/20-utility.md` §5.2–§5.4.
//!
//! [`TensorBase`]: crate::tensor::TensorBase

use std::borrow::Cow;

use crate::dimension::Dimension;
use crate::element::Element;
use crate::error::{StorageKindTag, XenonError};
use crate::storage::{ArcRepr, Owned, StorageMut, ViewMutRepr, ViewRepr};
use crate::tensor::TensorBase;

// ── Primary in-place fill (compile-time writable gate) ──

impl<S, D, A> TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    D: Dimension,
    A: Element + Clone,
{
    /// Fill all logical elements with `value` in place
    /// (`20-utility §5.2`, primary public entry point).
    ///
    /// Stride-aware: iterates via `iter_mut()` so non-contiguous layouts and
    /// tensors with internal padding only have their logical elements
    /// touched (`§5.4`).
    #[expect(
        clippy::clone_on_copy,
        reason = "generic over Clone (not Copy); .clone() is the correct generic pattern"
    )]
    pub fn fill(&mut self, value: A) {
        for slot in self.iter_mut() {
            *slot = value.clone();
        }
    }
}

// ── Crate-internal helpers (collapsed `fill_try_dispatch` body from §5.2) ──

/// Read-only branch: §5.3 row "View / Shared → InvalidStorageMode".
/// `tag` is the [`StorageKindTag`] for the concrete storage in this arm.
pub(crate) fn fill_try_read_only_err(tag: StorageKindTag) -> XenonError {
    XenonError::InvalidStorageMode {
        operation: Cow::Borrowed("Tensor::try_fill"),
        expected: StorageKindTag::Owned,
        actual: tag,
        shape: None,
        conversion: None,
    }
}

// ── Public entry points: 4 concrete inherent impls realizing §5.2 ──
//
// Every `S` below trivially satisfies design §5.2's `S: Storage<Elem = A>`
// public constraint (4 concrete storages all implement `Storage`).
//
// W7T6 marker traits (`IsOwned`, `IsViewMut`, `IsView`, `IsShared`) serve
// only to document which §5.3 dispatch arm each impl corresponds to. They
// are NOT added as `where` bounds because doing so triggers
// rust-lang/rust#152409: a where-bound mentioning e.g. `Owned<A>: IsOwned`
// shadows the compiler's knowledge that `Owned<A>::Elem = A`, preventing
// the associated-type equality from being used in the method body.

impl<D, A> TensorBase<Owned<A>, D>
where
    D: Dimension,
    A: Element + Clone,
    // W7T6 tag (not a where-bound — see note above): Owned → §5.3 row 1 (writable)
{
    /// Fallible fill (`20-utility §5.2`, secondary entry on Owned).
    ///
    /// §5.3 dispatch arm: Owned → `iter_mut()` write path.
    ///
    /// # Errors
    ///
    /// Infallible: always returns `Ok(())`. The `Result` return type exists for
    /// API uniformity with the read-only `ViewRepr` / `ArcRepr` variants of
    /// `try_fill`, which return `XenonError::InvalidStorageMode`.
    #[expect(
        clippy::clone_on_copy,
        reason = "generic over Clone (not Copy); .clone() is the correct generic pattern"
    )]
    pub fn try_fill(&mut self, value: A) -> Result<(), XenonError> {
        for slot in self.iter_mut() {
            *slot = value.clone();
        }
        Ok(())
    }
}

impl<'a, D, A> TensorBase<ViewMutRepr<'a, A>, D>
where
    D: Dimension,
    A: Element + Clone,
    // W7T6 tag (not a where-bound): ViewMut → §5.3 row 1 (writable)
{
    /// Fallible fill (`20-utility §5.2`, secondary entry on ViewMut).
    ///
    /// §5.3 dispatch arm: ViewMut → `iter_mut()` write path.
    ///
    /// # Errors
    ///
    /// Infallible: always returns `Ok(())`. The `Result` return type exists for
    /// API uniformity with the read-only `ViewRepr` / `ArcRepr` variants of
    /// `try_fill`, which return `XenonError::InvalidStorageMode`.
    #[expect(
        clippy::clone_on_copy,
        reason = "generic over Clone (not Copy); .clone() is the correct generic pattern"
    )]
    pub fn try_fill(&mut self, value: A) -> Result<(), XenonError> {
        for slot in self.iter_mut() {
            *slot = value.clone();
        }
        Ok(())
    }
}

impl<'a, D, A> TensorBase<ViewRepr<'a, A>, D>
where
    D: Dimension,
    A: Element + Clone,
    // W7T6 tag (not a where-bound): View → §5.3 row 2 (read-only)
{
    /// Fallible fill (`20-utility §5.2`, secondary entry on View).
    ///
    /// §5.3 dispatch arm: View / SharedReadOnly → `InvalidStorageMode`.
    /// Covers BOTH the plain `ReadOnly` ViewRepr and the runtime-tagged
    /// `SharedReadOnly` ViewRepr cases (derived_from_view_mut and zero-stride
    /// broadcast — see W8T4 `access_semantics()`): both collapse to the
    /// same `InvalidStorageMode` outcome here.
    ///
    /// # Errors
    ///
    /// Always returns `XenonError::InvalidStorageMode` with
    /// `storage_kind: StorageKindTag::View` — a `View` (read-only) tensor
    /// cannot be mutated through `try_fill`.
    pub fn try_fill(&mut self, _value: A) -> Result<(), XenonError> {
        Err(fill_try_read_only_err(StorageKindTag::View))
    }
}

impl<D, A> TensorBase<ArcRepr<A>, D>
where
    D: Dimension,
    A: Element + Clone,
    // W7T6 tag (not a where-bound): Shared → §5.3 row 2 (read-only)
{
    /// Fallible fill (`20-utility §5.2`, secondary entry on Arc).
    ///
    /// §5.3 dispatch arm: Shared (read-only) → `InvalidStorageMode`.
    ///
    /// # Errors
    ///
    /// Always returns `XenonError::InvalidStorageMode` with
    /// `storage_kind: StorageKindTag::Shared` — an `ArcRepr` (shared,
    /// read-only) tensor cannot be mutated through `try_fill`.
    pub fn try_fill(&mut self, _value: A) -> Result<(), XenonError> {
        Err(fill_try_read_only_err(StorageKindTag::Shared))
    }
}

// ── Unit tests (§8.2 / §7 T1) ──

#[cfg(test)]
mod tests {
    use crate::error::XenonError;
    use crate::tensor::Tensor1;

    // §8.2 — test_fill_basic
    #[test]
    fn test_fill_basic() {
        let mut tensor = Tensor1::<f64>::zeros([3]).expect("zeros(valid shape)");
        tensor.fill(2.5);
        assert_eq!(
            tensor.iter().copied().collect::<Vec<_>>(),
            vec![2.5, 2.5, 2.5]
        );
    }

    // §8.2 — test_try_fill_read_only_returns_error
    #[test]
    fn test_try_fill_read_only_returns_error() {
        let tensor =
            Tensor1::from_shape_vec([2], vec![1_i32, 2]).expect("from_shape_vec matching shape");
        let mut view = tensor.view();
        let error = view.try_fill(7).expect_err("view is read-only");
        assert!(matches!(error, XenonError::InvalidStorageMode { .. }));
    }

    // §7 T1 — test_try_fill_read_only_returns_read_only_storage
    //
    // Design §7 T1 and §8.2 list the SAME behavioural requirement under
    // two different names. Build MUST NOT pick one and drop the other — we
    // publish both as independent `#[test]` items. The body is identical
    // to `test_try_fill_read_only_returns_error`; keeping both names keeps
    // `cargo test` output traceable to either design citation.
    #[test]
    fn test_try_fill_read_only_returns_read_only_storage() {
        let tensor =
            Tensor1::from_shape_vec([2], vec![1_i32, 2]).expect("from_shape_vec matching shape");
        let mut view = tensor.view();
        let error = view.try_fill(7).expect_err("view is read-only");
        assert!(matches!(error, XenonError::InvalidStorageMode { .. }));
    }

    // §8.2 — test_fill_non_contiguous
    //
    // **DEPENDENCY GAP**: requires writable non-contiguous view constructor
    // (transpose returns read-only ViewRepr per 16-shape §5; no writable
    // non-contiguous view constructor scheduled in current SUMMARY.md).
    #[test]
    #[ignore = "needs writable non-contiguous view primitive (transpose returns read-only ViewRepr per 16-shape §5; no writable non-contiguous view constructor scheduled in current SUMMARY.md)"]
    fn test_fill_non_contiguous() {
        todo!("activate after writable non-contiguous view constructor lands");
    }

    // §8.2 — test_fill_padded_writes_logical_only
    //
    // **DEPENDENCY GAP**: needs writable strided sub-view primitive.
    #[test]
    #[ignore = "needs writable strided sub-view primitive (not scheduled in current SUMMARY.md indexing waves W21T1–W21T6)"]
    fn test_fill_padded_writes_logical_only() {
        todo!("activate after writable strided slice constructor lands");
    }

    // §8.2 — test_try_fill_writable_matches_fill
    #[test]
    fn test_try_fill_writable_matches_fill() {
        let mut t1 = Tensor1::<f64>::zeros([3]).expect("zeros(valid shape)");
        let mut t2 = Tensor1::<f64>::zeros([3]).expect("zeros(valid shape)");
        t1.fill(2.71);
        t2.try_fill(2.71).expect("try_fill on owned is writable");
        let v1: Vec<_> = t1.iter().copied().collect();
        let v2: Vec<_> = t2.iter().copied().collect();
        assert_eq!(v1, v2);
    }

    // §8.2 — test_fill_empty
    #[test]
    fn test_fill_empty() {
        let mut tensor = Tensor1::<f64>::zeros([0]).expect("zeros(empty)");
        tensor.fill(1.0); // must not panic
        assert_eq!(tensor.len(), 0);
    }

    // §8.4 invariant — property-style coverage of `fill(v) ⇒ all == v`.
    // Light, deterministic coverage (no proptest dependency required).
    #[test]
    fn test_fill_invariant_all_equal_value() {
        for &n in &[1_usize, 2, 3, 5, 8, 13] {
            let mut t = Tensor1::<i64>::zeros([n]).expect("zeros(valid n)");
            t.fill(42);
            assert!(t.iter().all(|&x| x == 42), "n={}", n);
        }
    }
}
