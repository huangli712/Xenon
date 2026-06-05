//! Layout flags, state classification, and the `compute_layout_flags`
//! central entry point.
//!
//! Bitfield constants, query/setter methods, `LayoutFlags::classify()`
//! fast-path constructor, and `compute_layout_flags` are implemented here.

use crate::dimension::Dimension;
use super::aligned::is_aligned;
use super::contiguous::is_f_contiguous;
use super::strides::Strides;

/// Classification of tensor memory layout contiguity status.
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

impl LayoutState {
    /// Returns a human-readable label for the layout classification.
    pub fn as_str(self) -> &'static str {
        match self {
            LayoutState::FContiguous => "f-contiguous",
            LayoutState::BroadcastView => "broadcast",
            LayoutState::NonContiguous => "non-contiguous",
        }
    }
}

/// 8-bit packed layout flags: F_CONTIGUOUS (bit 0), ALIGNED (bit 2),
/// HAS_ZERO_STRIDE (bit 3). Bits 1, 4-7 are reserved.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LayoutFlags(u8);

impl LayoutFlags {
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

    /// Classifies the current layout into a `LayoutState` variant.
    ///
    /// Priority: `BroadcastView` → `FContiguous` → `NonContiguous`.
    ///
    /// # Invariant
    ///
    /// Correct only for `LayoutFlags` produced by `compute_layout_flags()`: that
    /// entry point is the sole authority that sets `HAS_ZERO_STRIDE` and guarantees
    /// `HAS_ZERO_STRIDE` is set if `any(stride == 0) && product(shape) > 0`.
    ///
    /// As a consequence, `classify()` does NOT (and cannot — it has no `shape`
    /// argument) re-check the `product(shape) > 0` half of the rule.
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

/// Central entry for computing `LayoutFlags` from `shape + strides + ptr`.
///
/// This function is the **single source of truth** for the `HAS_ZERO_STRIDE`
/// bit and for the F-order / alignment flags as cached in `TensorBase`.
/// Downstream callers (construction / broadcast / transpose / slice paths)
/// MUST route through this function rather than recomputing flags themselves.
///
/// # Preconditions
///
/// `product(shape)` must be representable in `usize`; fallible shape
/// validation belongs to the caller (e.g., `Dimension::checked_size` or
/// `Strides::f_contiguous`). This function does NOT return `Result`.
///
/// # Pointer safety
///
/// `ptr` is inspected only as an integer address (modulo alignment) and
/// is never dereferenced. The pointer is permitted to be dangling (e.g.,
/// for empty tensors).
pub(crate) fn compute_layout_flags<A, D: Dimension>(
    shape: &D,
    strides: &Strides<D>,
    ptr: *const A,
) -> LayoutFlags {
    // Step 1:
    //
    // HAS_ZERO_STRIDE := any(stride==0) && product(shape) > 0.
    let is_broadcast_zero_stride = strides.should_set_zero_stride_flag(shape);

    // Step 2:
    //
    // F_CONTIGUOUS := !is_broadcast_zero_stride && is_f_contiguous(shape, strides).
    //
    // Empty-array degenerate metadata (product == 0) keeps
    // `is_broadcast_zero_stride == false` and can therefore still be F_CONTIGUOUS.
    let f_contig = !is_broadcast_zero_stride && is_f_contiguous(shape, strides);

    // Step 3:
    // 
    // empty tensors report ALIGNED = true regardless of the dangling pointer;
    // otherwise inspect the address.
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
    use super::Strides;

    // --- LayoutFlags bitfield -----------------------------------------------

    /// Default LayoutFlags has all bits cleared.
    #[test]
    fn test_flags_default_empty() {
        let flags = LayoutFlags::default();
        assert_eq!(flags, LayoutFlags::EMPTY);
        assert!(!flags.is_f_contiguous());
        assert!(!flags.is_aligned());
        assert!(!flags.has_zero_stride());
    }

    /// Setting and then clearing each flag returns to EMPTY.
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

    /// All three flags can be set simultaneously.
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

    // --- LayoutFlags::classify ----------------------------------------------

    /// BroadcastView takes priority over FContiguous in classify().
    #[test]
    fn test_classify_broadcast_view_priority() {
        let flags = LayoutFlags::EMPTY
            .set_f_contiguous(true)
            .set_has_zero_stride(true);
        assert_eq!(flags.classify(), LayoutState::BroadcastView);
    }

    /// F_CONTIGUOUS alone classifies as FContiguous.
    #[test]
    fn test_classify_f_contiguous() {
        let flags = LayoutFlags::EMPTY.set_f_contiguous(true);
        assert_eq!(flags.classify(), LayoutState::FContiguous);
    }

    /// ALIGNED alone classifies as NonContiguous.
    #[test]
    fn test_classify_non_contiguous() {
        let flags = LayoutFlags::EMPTY.set_aligned(true);
        assert_eq!(flags.classify(), LayoutState::NonContiguous);
    }

    /// EMPTY (all bits cleared) classifies as NonContiguous.
    #[test]
    fn test_classify_empty_non_contiguous() {
        assert_eq!(LayoutFlags::EMPTY.classify(), LayoutState::NonContiguous);
    }

    // --- compute_layout_flags -----------------------------------------------

    /// Non-dereferenceable `u8` pointer for pointer-alignment-only tests.
    fn dangling_u8() -> *const u8 {
        core::ptr::NonNull::<u8>::dangling().as_ptr()
    }

    /// Normal construction path: F-order strides ⇒ F-contiguous flags.
    #[test]
    fn test_compute_layout_flags_construction_f_order() {
        let shape = Ix3(2, 3, 4);
        let strides = Strides::f_contiguous(&shape).expect("valid test shape");
        assert_eq!(strides.as_slice(), &[1, 2, 6]);
        assert!(is_f_contiguous(&shape, &strides));

        let flags = compute_layout_flags::<u8, Ix3>(&shape, &strides, dangling_u8());
        assert!(flags.is_f_contiguous());
        assert!(!flags.has_zero_stride());
        assert_eq!(flags.classify(), LayoutState::FContiguous);
    }

    /// Non-empty broadcast view sets HAS_ZERO_STRIDE and clears F_CONTIGUOUS.
    #[test]
    fn test_compute_layout_flags_broadcast_view() {
        let shape = Ix2(5, 4);
        let strides = Strides::new(Ix2(1, 0));
        let flags = compute_layout_flags::<u8, Ix2>(&shape, &strides, dangling_u8());
        assert!(flags.has_zero_stride());
        assert!(!flags.is_f_contiguous());
        assert_eq!(flags.classify(), LayoutState::BroadcastView);
    }

    /// Empty shape with degenerate zero stride keeps F_CONTIGUOUS.
    #[test]
    fn test_compute_layout_flags_empty_degenerate_zero_stride() {
        let shape = Ix2(0, 3);
        let strides = Strides::new(Ix2(1, 0));
        let flags = compute_layout_flags::<u8, Ix2>(&shape, &strides, dangling_u8());
        assert!(
            !flags.has_zero_stride(),
            "empty array degenerate zero stride must NOT set HAS_ZERO_STRIDE"
        );
        assert!(
            flags.is_f_contiguous(),
            "empty F-order metadata should remain F_CONTIGUOUS"
        );
        assert!(flags.is_aligned());
        assert_eq!(flags.classify(), LayoutState::FContiguous);
    }

    /// Transposed strides produce NonContiguous layout.
    #[test]
    fn test_compute_layout_flags_transpose_non_contiguous() {
        let shape = Ix2(3, 2);
        let strides = Strides::new(Ix2(2, 1));
        let flags = compute_layout_flags::<u8, Ix2>(&shape, &strides, dangling_u8());
        assert!(!flags.is_f_contiguous());
        assert!(!flags.has_zero_stride());
        assert_eq!(flags.classify(), LayoutState::NonContiguous);
    }

    /// Size-1 axis with arbitrary stride is still F-contiguous.
    #[test]
    fn test_compute_layout_flags_slice_size1_axis() {
        let shape = Ix3(5, 1, 4);
        let strides = Strides::new(Ix3(1, 999, 5));
        let flags = compute_layout_flags::<u8, Ix3>(&shape, &strides, dangling_u8());
        assert!(flags.is_f_contiguous());
        assert!(!flags.has_zero_stride());
    }

    /// 0-D scalar is always F-contiguous.
    #[test]
    fn test_compute_layout_flags_scalar() {
        let shape = Ix0;
        let strides = Strides::new(Ix0);
        let flags = compute_layout_flags::<u8, Ix0>(&shape, &strides, dangling_u8());
        assert!(flags.is_f_contiguous());
        assert!(!flags.has_zero_stride());
        assert_eq!(flags.classify(), LayoutState::FContiguous);
    }

    /// Non-empty tensor with 64-byte-aligned pointer reports aligned.
    #[test]
    fn test_compute_layout_flags_aligned_non_empty() {
        use std::alloc::{Layout, alloc, dealloc};
        let layout = Layout::from_size_align(128, 64).expect("valid layout");
        let ptr = unsafe { alloc(layout) };
        assert!(!ptr.is_null(), "allocator returned null");
        let shape = Ix2(4, 8);
        let strides = Strides::f_contiguous(&shape).expect("valid test shape");
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
