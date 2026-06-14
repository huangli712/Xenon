//! `compute_layout_flags` — central entry point that combines alignment,
//! contiguity, and stride checks to produce `LayoutFlags`.

use crate::dimension::Dimension;
use super::flags::LayoutFlags;
use super::strides::Strides;

/// Check whether `ptr` satisfies the alignment requirement.
///
/// 64 bytes (cache-line width) is the minimum useful alignment for most
/// SIMD paths.
///
/// Returns `false` for `align == 0` or non-power-of-two `align`; never
/// panics. The pointer is inspected only as an integer address (modulo
/// `align`); it is **not** dereferenced, and is permitted to be dangling
/// (e.g., for empty tensors).
#[inline]
fn is_aligned_to(ptr: *const u8, align: usize) -> bool {
    if align == 0 || !align.is_power_of_two() {
        return false;
    }
    (ptr as usize).is_multiple_of(align)
}

/// Convenience: check whether `ptr` is 64-byte aligned.
#[inline]
fn is_aligned(ptr: *const u8) -> bool {
    is_aligned_to(ptr, 64)
}

/// Returns `true` if the tensor is F-contiguous.
///
/// An F-contiguous layout has `stride[0] == 1` and strictly increasing
/// strides for axes with extent > 1. Size-1 axes may have arbitrary
/// strides. Empty and single-element tensors are always contiguous.
fn is_f_contiguous<D: Dimension>(shape: &D, strides: &Strides<D>) -> bool {
    let shape = shape.slice();
    let strides = strides.as_slice();

    // Fast path: empty / scalar / single-element layouts are always
    // contiguous, regardless of stride values.
    let mut size: usize = 1;
    for &extent in shape.iter() {
        size = match size.checked_mul(extent) {
            Some(v) => v,
            None => break, // overflow ⇒ definitely > 1 ⇒ go to general path
        };
        if size == 0 {
            return true;
        }
    }
    if size <= 1 {
        return true;
    }

    // General path: expected stride accumulates the product(shape[0..i]);
    // axes with shape[i] == 1 are skipped (stride may be arbitrary).
    let mut expected: usize = 1;
    for (&extent, &stride) in shape.iter().zip(strides.iter()) {
        if extent != 1 && stride != expected {
            return false;
        }
        // `expected_stride` accumulator: overflow saturates conservatively;
        // any subsequent stride that has to equal a saturated value will
        // simply fail the equality check and short-circuit to `false`.
        expected = match expected.checked_mul(extent) {
            Some(v) => v,
            None => return false,
        };
    }
    true
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
    use crate::dimension::{Ix0, Ix1, Ix2, Ix3};
    use super::super::LayoutState;
    use std::alloc::{Layout, alloc, dealloc};

    // --- alignment helpers --------------------------------------------------

    /// Non-dereferenceable `u8` pointer for pointer-alignment-only tests.
    fn dangling_u8() -> *const u8 {
        core::ptr::NonNull::<u8>::dangling().as_ptr()
    }

    /// 64-byte-aligned pointer passes all alignment checks.
    #[test]
    fn test_alignment_aligned() {
        let layout = Layout::from_size_align(256, 64).expect("valid layout");
        let ptr = unsafe { alloc(layout) };
        assert!(!ptr.is_null(), "allocator returned null");
        assert!(is_aligned(ptr));
        assert!(is_aligned_to(ptr, 64));
        assert!(is_aligned_to(ptr, 32));
        assert!(is_aligned_to(ptr, 1));
        unsafe {
            dealloc(ptr, layout);
        }
    }

    /// Unaligned pointer and invalid align arguments return false.
    #[test]
    fn test_alignment_unaligned() {
        let values = [1_u8, 2, 3];
        let ptr = unsafe { values.as_ptr().add(1) };
        assert!(!is_aligned_to(ptr, 64));
        assert!(!is_aligned_to(values.as_ptr(), 0));
        assert!(!is_aligned_to(values.as_ptr(), 3));
    }

    // --- contiguity detection -----------------------------------------------

    /// Empty shape with zero stride is still F-contiguous.
    #[test]
    fn test_f_contig_empty() {
        let shape = Ix2(0, 3);
        let strides = Strides::new(Ix2(1, 0));
        assert!(is_f_contiguous(&shape, &strides));
    }

    /// F-order strides for [2, 3] are [1, 2] ⇒ contiguous.
    #[test]
    fn test_f_contig_true() {
        let shape = Ix2(2, 3);
        let strides = Strides::new(Ix2(1, 2));
        assert!(is_f_contiguous(&shape, &strides));
    }

    /// C-order strides [3, 1] for [2, 3] ⇒ NOT F-contiguous.
    #[test]
    fn test_f_contig_false() {
        let shape = Ix2(2, 3);
        let strides = Strides::new(Ix2(3, 1));
        assert!(!is_f_contiguous(&shape, &strides));
    }

    /// 0-D scalar always F-contiguous.
    #[test]
    fn test_f_contig_scalar() {
        let shape = Ix0;
        let strides = Strides::new(Ix0);
        assert!(is_f_contiguous(&shape, &strides));
    }

    /// Size-1 axis with arbitrary stride is still F-contiguous.
    #[test]
    fn test_f_contig_size1_axis() {
        let shape = Ix3(5, 1, 4);
        let strides = Strides::new(Ix3(1, 999, 5));
        assert!(is_f_contiguous(&shape, &strides));
    }

    /// 1-D arrays are F-contiguous when stride[0] == 1.
    #[test]
    fn test_f_contig_1d() {
        let shape = Ix1(5);
        let strides = Strides::new(Ix1(1));
        assert!(is_f_contiguous(&shape, &strides));
    }

    /// `Strides::f_contiguous` output is always recognised as F-contiguous.
    #[test]
    fn test_f_contiguous_round_trip() {
        let shape = Ix3(2, 3, 4);
        let s = Strides::f_contiguous(&shape).expect("valid test shape");
        assert_eq!(s.as_slice(), &[1, 2, 6]);
        assert!(is_f_contiguous(&shape, &s));

        let shape = Ix2(4, 5);
        let s = Strides::f_contiguous(&shape).expect("valid test shape");
        assert_eq!(s.as_slice(), &[1, 4]);
        assert!(is_f_contiguous(&shape, &s));
    }

    // --- compute_layout_flags -----------------------------------------------

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
