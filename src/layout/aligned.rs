//! Pointer alignment checks.
//!
//! `is_aligned` / `is_aligned_to` — raw-pointer alignment verification.

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
pub(crate) fn is_aligned_to(ptr: *const u8, align: usize) -> bool {
    if align == 0 || !align.is_power_of_two() {
        return false;
    }
    (ptr as usize).is_multiple_of(align)
}

/// Convenience: check whether `ptr` is 64-byte aligned.
#[inline]
pub(crate) fn is_aligned(ptr: *const u8) -> bool {
    is_aligned_to(ptr, 64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 64-byte-aligned pointer passes all alignment checks.
    #[test]
    fn test_alignment_aligned() {
        use std::alloc::{Layout, alloc, dealloc};
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
}
