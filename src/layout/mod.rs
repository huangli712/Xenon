//! Layout module: F-order strides, contiguity, flags and alignment.
//!
//! See `docs/design/06-layout.md`.

mod contiguous;
mod flags;
mod strides;

pub use contiguous::is_f_contiguous;
pub use flags::{LayoutFlags, LayoutState};
pub use strides::{Strides, compute_f_strides, has_zero_stride, is_aligned, is_aligned_to};

use crate::dimension::Dimension;

use strides::should_set_zero_stride_flag;

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

    fn dangling_u8() -> *const u8 {
        core::ptr::NonNull::<u8>::dangling().as_ptr()
    }

    #[test]
    #[allow(unused_imports)]
    fn test_layout_module_skeleton_compiles() {
        // Compile-time verification: submodule files must exist.
        use super::{contiguous as _, flags as _, strides as _};

        let module_path = module_path!();
        assert!(
            module_path.contains("layout"),
            "module_path! should reference layout module, got: {module_path}"
        );
        assert!(
            module_path.contains("tests"),
            "module_path! should reference tests submodule, got: {module_path}"
        );
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

    // === §5.12 construction context ===

    #[test]
    fn test_compute_layout_flags_construction_f_order() {
        let shape = Ix3(2, 3, 4);
        let strides = compute_f_strides(&shape).expect("valid test shape");
        let flags = compute_layout_flags::<u8, Ix3>(&shape, &strides, dangling_u8());
        assert!(flags.is_f_contiguous());
        assert!(!flags.has_zero_stride());
    }

    // === §5.12 broadcast context (§5.11 row 2) ===

    #[test]
    fn test_compute_layout_flags_broadcast_view() {
        let shape = Ix2(3, 4);
        let strides = Strides::new(Ix2(1, 0));
        let flags = compute_layout_flags::<u8, Ix2>(&shape, &strides, dangling_u8());
        assert!(flags.has_zero_stride());
        assert!(!flags.is_f_contiguous());
        assert_eq!(flags.classify(), LayoutState::BroadcastView);
    }

    // === §5.11 row 3: empty array degenerate zero stride ===

    #[test]
    fn test_compute_layout_flags_empty_degenerate_zero_stride() {
        let shape = Ix2(0, 3);
        let strides = Strides::new(Ix2(1, 0));
        let flags = compute_layout_flags::<u8, Ix2>(&shape, &strides, dangling_u8());
        // §5.11: empty array ⇒ HAS_ZERO_STRIDE MUST stay false.
        assert!(!flags.has_zero_stride());
        // §5.9: empty array ⇒ ALIGNED true regardless of pointer.
        assert!(flags.is_aligned());
        // §5.11: F_CONTIGUOUS retained for empty F-order metadata.
        assert!(flags.is_f_contiguous());
        assert_eq!(flags.classify(), LayoutState::FContiguous);
    }

    // === §5.12 shape (transpose) context ===

    #[test]
    fn test_compute_layout_flags_transpose_non_contiguous() {
        // Transposed shape [3, 2] with strides [2, 1] ⇒ NonContiguous.
        let shape = Ix2(3, 2);
        let strides = Strides::new(Ix2(2, 1));
        let flags = compute_layout_flags::<u8, Ix2>(&shape, &strides, dangling_u8());
        assert!(!flags.is_f_contiguous());
        assert!(!flags.has_zero_stride());
        assert_eq!(flags.classify(), LayoutState::NonContiguous);
    }

    // === §5.12 index (slice with size=1 axis) context ===

    #[test]
    fn test_compute_layout_flags_slice_size1_axis() {
        // §5.7: size=1 axis may have arbitrary stride; still F-contiguous.
        let shape = Ix3(5, 1, 4);
        let strides = Strides::new(Ix3(1, 999, 5));
        let flags = compute_layout_flags::<u8, Ix3>(&shape, &strides, dangling_u8());
        assert!(flags.is_f_contiguous());
        assert!(!flags.has_zero_stride());
    }

    // === §5.11 row 5: 0-D scalar ===

    #[test]
    fn test_compute_layout_flags_scalar() {
        let shape = Ix0;
        let strides = Strides::new(Ix0);
        let flags = compute_layout_flags::<u8, Ix0>(&shape, &strides, dangling_u8());
        assert!(flags.is_f_contiguous());
        assert!(!flags.has_zero_stride());
    }

    // === §5.12 aligned non-empty shape (§5.9) ===

    #[test]
    fn test_compute_layout_flags_aligned_non_empty() {
        // Verify that a non-empty tensor with a 64-byte-aligned pointer
        // correctly reports `is_aligned() == true`.
        use std::alloc::{Layout, alloc, dealloc};
        let layout = Layout::from_size_align(128, 64).expect("valid layout");
        // SAFETY: layout is non-zero size with valid align.
        let ptr = unsafe { alloc(layout) };
        assert!(!ptr.is_null(), "allocator returned null");
        let shape = Ix2(4, 8);
        let strides = compute_f_strides(&shape).expect("valid test shape");
        let flags = compute_layout_flags::<u8, Ix2>(&shape, &strides, ptr);
        assert!(flags.is_f_contiguous());
        assert!(flags.is_aligned());
        assert!(!flags.has_zero_stride());
        assert_eq!(flags.classify(), LayoutState::FContiguous);
        // SAFETY: ptr was obtained from `alloc(layout)`.
        unsafe {
            dealloc(ptr, layout);
        }
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::dimension::{Ix0, Ix2, Ix3};

    fn dangling_u8() -> *const u8 {
        core::ptr::NonNull::<u8>::dangling().as_ptr()
    }

    // === Construction context (§5.12 row 4) ===

    #[test]
    fn test_layout_integration_construction_f_order() {
        // §5.6/§5.7 symmetry: compute_f_strides ⇒ is_f_contiguous == true.
        let shape = Ix3(2, 3, 4);
        let strides = compute_f_strides(&shape).expect("valid test shape");
        assert_eq!(strides.as_slice(), &[1, 2, 6]);
        assert!(is_f_contiguous(&shape, &strides));

        // §5.12: compute_layout_flags on this layout ⇒ F_CONTIGUOUS bit set,
        // HAS_ZERO_STRIDE clear; classify ⇒ FContiguous.
        // Pointer alignment is undefined here; test only the contiguity bit.
        let dangling = dangling_u8();
        let flags = compute_layout_flags::<u8, Ix3>(&shape, &strides, dangling);
        assert!(flags.is_f_contiguous());
        assert!(!flags.has_zero_stride());
        assert_eq!(flags.classify(), LayoutState::FContiguous);
    }

    // === Broadcast context (§5.11 + §5.12 row 1) ===

    #[test]
    fn test_layout_integration_broadcast_view() {
        // Non-empty broadcast view: shape [5, 4], strides [1, 0].
        let shape = Ix2(5, 4);
        let strides = Strides::new(Ix2(1, 0));
        let dangling = dangling_u8();
        let flags = compute_layout_flags::<u8, Ix2>(&shape, &strides, dangling);
        assert!(flags.has_zero_stride());
        // §5.11: broadcast view ⇒ NOT F_CONTIGUOUS.
        assert!(!flags.is_f_contiguous());
        assert_eq!(flags.classify(), LayoutState::BroadcastView);
    }

    #[test]
    fn test_layout_integration_empty_degenerate_zero_stride() {
        // §5.11 boundary: empty shape `[0, 3]` with degenerate zero stride
        // ⇒ HAS_ZERO_STRIDE MUST stay false; F_CONTIGUOUS may stay true.
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
        // §5.9: empty tensor ⇒ ALIGNED true regardless of pointer.
        assert!(flags.is_aligned());
        assert_eq!(flags.classify(), LayoutState::FContiguous);
    }

    // === Shape (transpose) context (§5.12 row 2) ===

    #[test]
    fn test_layout_integration_transpose_non_contiguous() {
        // Transpose of [2, 3] (F-order strides [1, 2]) ⇒ shape [3, 2],
        // strides [2, 1] — no longer F-contiguous, no zero strides.
        let shape = Ix2(3, 2);
        let strides = Strides::new(Ix2(2, 1));
        let dangling = dangling_u8();
        let flags = compute_layout_flags::<u8, Ix2>(&shape, &strides, dangling);
        assert!(!flags.is_f_contiguous());
        assert!(!flags.has_zero_stride());
        assert_eq!(flags.classify(), LayoutState::NonContiguous);
    }

    // === Index (slice) context (§5.12 row 3) ===

    #[test]
    fn test_layout_integration_slice_size1_axis() {
        // size=1 axis along axis 1 with arbitrary stride; full layout
        // still F-contiguous (§5.7 size=1 axis rule).
        let shape = Ix3(5, 1, 4);
        let strides = Strides::new(Ix3(1, 999, 5));
        let dangling = dangling_u8();
        let flags = compute_layout_flags::<u8, Ix3>(&shape, &strides, dangling);
        assert!(flags.is_f_contiguous());
        assert!(!flags.has_zero_stride());
    }

    // === Scalar / 0-D ===

    #[test]
    fn test_layout_integration_scalar() {
        // §5.11 boundary: 0-D scalar (`product == 1`) ⇒ F-contiguous,
        // no zero stride; alignment depends on pointer (use aligned alloc).
        let shape = Ix0;
        let strides = Strides::new(Ix0);
        let dangling = dangling_u8();
        let flags = compute_layout_flags::<u8, Ix0>(&shape, &strides, dangling);
        assert!(flags.is_f_contiguous());
        assert!(!flags.has_zero_stride());
        assert_eq!(flags.classify(), LayoutState::FContiguous);
    }
}
