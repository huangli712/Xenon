//! `LayoutFlags` bitfield: packed contiguity / alignment / broadcast flags.
//!
//! Bitfield constants, query/setter methods, and `LayoutFlags::classify()`
//! fast-path constructor are implemented here. `LayoutState` lives in
//! `state`; flag computation (`compute_layout_flags`) lives in `compute`.

use super::state::LayoutState;

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
