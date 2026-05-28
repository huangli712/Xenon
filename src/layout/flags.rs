//! Layout flags and state classification.
//!
//! Bitfield constants, query/setter methods, `LayoutFlags::classify()`,
//! `flags_for_f_layout` fast-path, and `compute_layout_flags` central
//! entry point are implemented here.

use crate::dimension::Dimension;
use super::contiguous::is_f_contiguous;
use super::strides::{is_aligned, should_set_zero_stride_flag, Strides};

/// 8-bit packed layout flags. Concrete bit layout in `06-layout §5.1`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LayoutFlags(u8);

/// Classification of tensor memory layout contiguity (`06-layout §5.3`).
///
/// Variants are mutually exclusive. `BroadcastView` applies only when
/// `product(shape) > 0 && any(stride == 0)`; empty tensors with degenerate
/// zero strides remain `FContiguous`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutState {
    /// Fortran-contiguous: first stride = 1, F-order progression.
    FContiguous,
    /// Arbitrary non-broadcast view that is not F-contiguous.
    NonContiguous,
    /// Non-empty view with at least one zero-stride axis (broadcast).
    BroadcastView,
}

impl LayoutFlags {
    // === Constants ===

    /// Empty flags — all bits cleared.
    pub const EMPTY: Self = Self(0);

    /// F-order contiguity flag (bit 0, 0x01).
    pub const F_CONTIGUOUS: Self = Self(0b0000_0001);

    /// SIMD alignment flag (bit 2, 0x04) — 64-byte aligned.
    pub const ALIGNED: Self = Self(0b0000_0100);

    /// Zero stride flag (bit 3, 0x08) — set for broadcast-induced
    /// zero strides (`product(shape) > 0`), never for empty-array
    /// degenerate metadata.
    pub const HAS_ZERO_STRIDE: Self = Self(0b0000_1000);

    // === Query methods ===

    /// Returns `true` if the F-order contiguity flag is set.
    #[inline]
    pub const fn is_f_contiguous(self) -> bool {
        (self.0 & Self::F_CONTIGUOUS.0) != 0
    }

    /// Returns `true` if the 64-byte alignment flag is set.
    #[inline]
    pub const fn is_aligned(self) -> bool {
        (self.0 & Self::ALIGNED.0) != 0
    }

    /// Returns `true` if the broadcast zero-stride flag is set.
    #[inline]
    pub const fn has_zero_stride(self) -> bool {
        (self.0 & Self::HAS_ZERO_STRIDE.0) != 0
    }

    // === Setter methods (const, builder pattern) ===

    /// Sets or clears the F-order contiguity flag.
    #[inline]
    pub const fn set_f_contiguous(self, val: bool) -> Self {
        if val {
            Self(self.0 | Self::F_CONTIGUOUS.0)
        } else {
            Self(self.0 & !Self::F_CONTIGUOUS.0)
        }
    }

    /// Sets or clears the 64-byte alignment flag.
    #[inline]
    pub const fn set_aligned(self, val: bool) -> Self {
        if val {
            Self(self.0 | Self::ALIGNED.0)
        } else {
            Self(self.0 & !Self::ALIGNED.0)
        }
    }

    /// Sets or clears the broadcast zero-stride flag.
    #[inline]
    pub const fn set_has_zero_stride(self, val: bool) -> Self {
        if val {
            Self(self.0 | Self::HAS_ZERO_STRIDE.0)
        } else {
            Self(self.0 & !Self::HAS_ZERO_STRIDE.0)
        }
    }

    // === LayoutState classification (§5.3) ===

    /// Classifies the current layout into a `LayoutState` variant.
    ///
    /// Deterministic priority: `BroadcastView` → `FContiguous` → `NonContiguous`.
    ///
    /// # Invariant
    ///
    /// Correct **only** for `LayoutFlags` produced by
    /// `compute_layout_flags(shape, strides, ptr)`: that entry
    /// point is the sole authority that sets `HAS_ZERO_STRIDE` and
    /// guarantees `HAS_ZERO_STRIDE` is set iff
    /// `any(stride == 0) && product(shape) > 0`. As a consequence,
    /// `classify` does NOT (and cannot — it has no `shape` argument)
    /// re-check the `product(shape) > 0` half of the rule.
    #[inline]
    pub const fn classify(self) -> LayoutState {
        if self.has_zero_stride() {
            LayoutState::BroadcastView
        } else if self.is_f_contiguous() {
            LayoutState::FContiguous
        } else {
            LayoutState::NonContiguous
        }
    }
}

/// Compute canonical `LayoutFlags` for an already-validated F-order layout
/// (`06-layout §5.2`).
///
/// Fast path: use only when the caller has already established that the
/// layout is F-order (e.g., immediately after a successful
/// `compute_f_strides()`). For the general case, use
/// `compute_layout_flags(shape, strides, ptr)` (§5.12).
///
/// # Arguments
///
/// * `aligned` - whether the logical-first pointer satisfies 64-byte
///   alignment, OR whether the layout describes an empty tensor (§5.9).
/// * `is_broadcast_zero_stride` - whether the layout contains broadcast-induced
///   zero strides. Empty-array degenerate zero strides (`product(shape) == 0`)
///   MUST be passed as `false` (their `F_CONTIGUOUS` bit is retained).
#[inline]
#[allow(
    dead_code,
    reason = "06-layout §5.2 fast-path API — canonical LayoutFlags constructor \
              for already-validated F-order layouts. Pairs with the general \
              `compute_layout_flags` (§5.12); no production caller currently \
              chooses the fast path (all sites go through the general one). \
              Implementation + tests are complete. (`allow` rather than \
              `expect` because dead_code only fires without `--tests`; \
              test-mode use suppresses the lint, so `expect` would be \
              unfulfilled.)"
)]
pub(crate) const fn flags_for_f_layout(
    aligned: bool,
    is_broadcast_zero_stride: bool,
) -> LayoutFlags {
    LayoutFlags::EMPTY
        .set_f_contiguous(!is_broadcast_zero_stride)
        .set_aligned(aligned)
        .set_has_zero_stride(is_broadcast_zero_stride)
}

/// Central entry for computing `LayoutFlags` from `shape + strides + ptr`
/// (`06-layout §5.12`).
///
/// This function is the **single source of truth** for the
/// `HAS_ZERO_STRIDE` bit and for the F-order / alignment flags as cached
/// in `TensorBase`. Downstream callers (construction / broadcast /
/// transpose / slice paths) MUST route through this function rather than
/// recomputing flags themselves.
///
/// # Preconditions
///
/// `product(shape)` must be representable in `usize`; fallible shape
/// validation belongs to the caller (e.g., `Dimension::checked_size` or
/// `compute_f_strides`). This function does NOT return `Result`.
///
/// # Pointer safety (§6.5)
///
/// `ptr` is inspected only as an integer address (modulo alignment) and
/// is never dereferenced. The pointer is permitted to be dangling (e.g.,
/// for empty tensors).
pub(crate) fn compute_layout_flags<A, D: Dimension>(
    shape: &D,
    strides: &Strides<D>,
    ptr: *const A,
) -> LayoutFlags {
    // §6.1 step 1: HAS_ZERO_STRIDE := any(stride==0) && product(shape) > 0.
    let is_broadcast_zero_stride = should_set_zero_stride_flag(shape, strides);

    // §6.1 step 2: F_CONTIGUOUS := !is_broadcast_zero_stride
    //                              && is_f_contiguous(shape, strides).
    //
    // Empty-array degenerate metadata (product == 0) keeps
    // `is_broadcast_zero_stride == false` and can therefore still be
    // F_CONTIGUOUS, matching §5.11 edge-case table row 3.
    let f_contig = !is_broadcast_zero_stride && is_f_contiguous(shape, strides);

    // §6.1 step 3 + §5.9: empty tensors report ALIGNED = true regardless
    // of the dangling pointer; otherwise inspect the address.
    let is_empty = shape.slice().contains(&0);
    let aligned = if is_empty {
        true
    } else {
        is_aligned(ptr as *const u8)
    };

    LayoutFlags::EMPTY
        .set_f_contiguous(f_contig)
        .set_aligned(aligned)
        .set_has_zero_stride(is_broadcast_zero_stride)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::{Ix0, Ix2, Ix3};
    use crate::layout::compute_f_strides;

    #[test]
    fn test_flags_default_empty() {
        let flags = LayoutFlags::default();
        assert_eq!(flags, LayoutFlags::EMPTY);
        assert!(!flags.is_f_contiguous());
        assert!(!flags.is_aligned());
        assert!(!flags.has_zero_stride());
    }

    #[test]
    fn test_flags_set_clear() {
        let on = LayoutFlags::EMPTY
            .set_f_contiguous(true)
            .set_aligned(true)
            .set_has_zero_stride(true);
        assert!(on.is_f_contiguous());
        assert!(on.is_aligned());
        assert!(on.has_zero_stride());

        let off = on
            .set_f_contiguous(false)
            .set_aligned(false)
            .set_has_zero_stride(false);
        assert_eq!(off, LayoutFlags::EMPTY);
    }

    #[test]
    fn test_flags_all_set() {
        let flags = LayoutFlags::EMPTY
            .set_f_contiguous(true)
            .set_aligned(true)
            .set_has_zero_stride(true);
        assert!(flags.is_f_contiguous());
        assert!(flags.is_aligned());
        assert!(flags.has_zero_stride());
    }

    #[test]
    fn test_classify_broadcast_view_priority() {
        let flags = LayoutFlags::EMPTY
            .set_f_contiguous(true)
            .set_has_zero_stride(true);
        assert_eq!(flags.classify(), LayoutState::BroadcastView);
    }

    #[test]
    fn test_classify_f_contiguous() {
        let flags = LayoutFlags::EMPTY.set_f_contiguous(true);
        assert_eq!(flags.classify(), LayoutState::FContiguous);
    }

    #[test]
    fn test_classify_non_contiguous() {
        let flags = LayoutFlags::EMPTY.set_aligned(true);
        assert_eq!(flags.classify(), LayoutState::NonContiguous);
    }

    // === §5.2 flags_for_f_layout ===

    #[test]
    fn test_flags_for_f_layout_aligned_no_broadcast() {
        // Construction path: known F-order + aligned + no broadcast.
        let flags = flags_for_f_layout(/*aligned=*/ true, /*broadcast=*/ false);
        assert!(flags.is_f_contiguous());
        assert!(flags.is_aligned());
        assert!(!flags.has_zero_stride());
        assert_eq!(flags.classify(), LayoutState::FContiguous);
    }

    #[test]
    fn test_flags_for_f_layout_broadcast_clears_f_contig() {
        // §5.2: when caller declares broadcast zero stride, the fast path
        // MUST clear F_CONTIGUOUS regardless of the F-order assumption.
        let flags = flags_for_f_layout(true, true);
        assert!(!flags.is_f_contiguous());
        assert!(flags.has_zero_stride());
        assert_eq!(flags.classify(), LayoutState::BroadcastView);
    }

    // === §5.12 compute_layout_flags ===

    fn dangling_u8() -> *const u8 {
        core::ptr::NonNull::<u8>::dangling().as_ptr()
    }

    #[test]
    fn test_compute_layout_flags_construction_f_order() {
        let shape = Ix3(2, 3, 4);
        let strides = compute_f_strides(&shape).expect("valid test shape");
        let flags = compute_layout_flags::<u8, Ix3>(&shape, &strides, dangling_u8());
        assert!(flags.is_f_contiguous());
        assert!(!flags.has_zero_stride());
    }

    #[test]
    fn test_compute_layout_flags_broadcast_view() {
        let shape = Ix2(3, 4);
        let strides = Strides::new(Ix2(1, 0));
        let flags = compute_layout_flags::<u8, Ix2>(&shape, &strides, dangling_u8());
        assert!(flags.has_zero_stride());
        assert!(!flags.is_f_contiguous());
        assert_eq!(flags.classify(), LayoutState::BroadcastView);
    }

    #[test]
    fn test_compute_layout_flags_empty_degenerate_zero_stride() {
        let shape = Ix2(0, 3);
        let strides = Strides::new(Ix2(1, 0));
        let flags = compute_layout_flags::<u8, Ix2>(&shape, &strides, dangling_u8());
        assert!(!flags.has_zero_stride());
        assert!(flags.is_aligned());
        assert!(flags.is_f_contiguous());
        assert_eq!(flags.classify(), LayoutState::FContiguous);
    }

    #[test]
    fn test_compute_layout_flags_transpose_non_contiguous() {
        let shape = Ix2(3, 2);
        let strides = Strides::new(Ix2(2, 1));
        let flags = compute_layout_flags::<u8, Ix2>(&shape, &strides, dangling_u8());
        assert!(!flags.is_f_contiguous());
        assert!(!flags.has_zero_stride());
        assert_eq!(flags.classify(), LayoutState::NonContiguous);
    }

    #[test]
    fn test_compute_layout_flags_slice_size1_axis() {
        let shape = Ix3(5, 1, 4);
        let strides = Strides::new(Ix3(1, 999, 5));
        let flags = compute_layout_flags::<u8, Ix3>(&shape, &strides, dangling_u8());
        assert!(flags.is_f_contiguous());
        assert!(!flags.has_zero_stride());
    }

    #[test]
    fn test_compute_layout_flags_scalar() {
        let shape = Ix0;
        let strides = Strides::new(Ix0);
        let flags = compute_layout_flags::<u8, Ix0>(&shape, &strides, dangling_u8());
        assert!(flags.is_f_contiguous());
        assert!(!flags.has_zero_stride());
    }

    #[test]
    fn test_compute_layout_flags_aligned_non_empty() {
        use std::alloc::{Layout, alloc, dealloc};
        let layout = Layout::from_size_align(128, 64).expect("valid layout");
        let ptr = unsafe { alloc(layout) };
        assert!(!ptr.is_null(), "allocator returned null");
        let shape = Ix2(4, 8);
        let strides = compute_f_strides(&shape).expect("valid test shape");
        let flags = compute_layout_flags::<u8, Ix2>(&shape, &strides, ptr);
        assert!(flags.is_f_contiguous());
        assert!(flags.is_aligned());
        assert!(!flags.has_zero_stride());
        assert_eq!(flags.classify(), LayoutState::FContiguous);
        unsafe {
            dealloc(ptr, layout);
        }
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::dimension::{Ix0, Ix2, Ix3};
    use crate::layout::compute_f_strides;

    fn dangling_u8() -> *const u8 {
        core::ptr::NonNull::<u8>::dangling().as_ptr()
    }

    #[test]
    fn test_layout_integration_construction_f_order() {
        let shape = Ix3(2, 3, 4);
        let strides = compute_f_strides(&shape).expect("valid test shape");
        assert_eq!(strides.as_slice(), &[1, 2, 6]);
        assert!(is_f_contiguous(&shape, &strides));

        let dangling = dangling_u8();
        let flags = compute_layout_flags::<u8, Ix3>(&shape, &strides, dangling);
        assert!(flags.is_f_contiguous());
        assert!(!flags.has_zero_stride());
        assert_eq!(flags.classify(), LayoutState::FContiguous);
    }

    #[test]
    fn test_layout_integration_broadcast_view() {
        let shape = Ix2(5, 4);
        let strides = Strides::new(Ix2(1, 0));
        let dangling = dangling_u8();
        let flags = compute_layout_flags::<u8, Ix2>(&shape, &strides, dangling);
        assert!(flags.has_zero_stride());
        assert!(!flags.is_f_contiguous());
        assert_eq!(flags.classify(), LayoutState::BroadcastView);
    }

    #[test]
    fn test_layout_integration_empty_degenerate_zero_stride() {
        let shape = Ix2(0, 3);
        let strides = Strides::new(Ix2(1, 0));
        let dangling = dangling_u8();
        let flags = compute_layout_flags::<u8, Ix2>(&shape, &strides, dangling);
        assert!(
            !flags.has_zero_stride(),
            "empty array degenerate zero stride must NOT set HAS_ZERO_STRIDE"
        );
        assert!(
            flags.is_f_contiguous(),
            "empty F-order metadata should remain F_CONTIGUOUS (§5.11)"
        );
        assert!(flags.is_aligned());
        assert_eq!(flags.classify(), LayoutState::FContiguous);
    }

    #[test]
    fn test_layout_integration_transpose_non_contiguous() {
        let shape = Ix2(3, 2);
        let strides = Strides::new(Ix2(2, 1));
        let dangling = dangling_u8();
        let flags = compute_layout_flags::<u8, Ix2>(&shape, &strides, dangling);
        assert!(!flags.is_f_contiguous());
        assert!(!flags.has_zero_stride());
        assert_eq!(flags.classify(), LayoutState::NonContiguous);
    }

    #[test]
    fn test_layout_integration_slice_size1_axis() {
        let shape = Ix3(5, 1, 4);
        let strides = Strides::new(Ix3(1, 999, 5));
        let dangling = dangling_u8();
        let flags = compute_layout_flags::<u8, Ix3>(&shape, &strides, dangling);
        assert!(flags.is_f_contiguous());
        assert!(!flags.has_zero_stride());
    }

    #[test]
    fn test_layout_integration_scalar() {
        let shape = Ix0;
        let strides = Strides::new(Ix0);
        let dangling = dangling_u8();
        let flags = compute_layout_flags::<u8, Ix0>(&shape, &strides, dangling);
        assert!(flags.is_f_contiguous());
        assert!(!flags.has_zero_stride());
        assert_eq!(flags.classify(), LayoutState::FContiguous);
    }
}
