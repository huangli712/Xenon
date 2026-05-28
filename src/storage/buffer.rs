//! Buffers wrapping aligned allocations.
//!
//! `AlignedBuf<A>` is the internal aligned heap buffer used by all storage
//! representations. `SharedBuf<A>` wraps an `AlignedBuf` in an `Arc` for
//! shared read-only access.
//!
//! When the last owner is dropped, `AlignedBuf`'s `Drop` releases the
//! aligned allocation with the original layout.

use core::marker::PhantomData;
use core::mem::{align_of, size_of};
use core::ptr::NonNull;
use std::borrow::Cow;

use crate::element::Element;
use crate::error::{InvalidShapeKind, XenonError};
use super::alloc::AlignedAlloc;

// ---------------------------------------------------------------------------
// Internal helper
// ---------------------------------------------------------------------------

pub(crate) fn allocation_size<A>(
    len: usize,
    align: usize,
    operation: &'static str,
) -> Result<usize, XenonError> {
    let size = len
        .checked_mul(size_of::<A>())
        .ok_or_else(|| XenonError::InvalidShape {
            operation: Cow::Borrowed(operation),
            shape: vec![len],
            kind: InvalidShapeKind::ProductOverflow,
            offending_dim: Some(0),
        })?;
    let max_size = (isize::MAX as usize).saturating_sub(align.saturating_sub(1));
    if size > max_size {
        return Err(XenonError::InvalidShape {
            operation: Cow::Borrowed(operation),
            shape: vec![len],
            kind: InvalidShapeKind::ProductOverflow,
            offending_dim: Some(0),
        });
    }
    Ok(size)
}

// ---------------------------------------------------------------------------
// AlignedBuf<A> — internal aligned buffer
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub(crate) struct AlignedBuf<A> {
    pub(crate) ptr: NonNull<A>,
    len: usize,
    cap: usize,
    align: usize,
    _marker: PhantomData<A>,
}

impl<A> AlignedBuf<A> {
    pub(crate) fn empty() -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            cap: 0,
            align: align_of::<A>().max(1),
            _marker: PhantomData,
        }
    }

    pub(crate) fn zst(len: usize) -> Self {
        debug_assert_eq!(size_of::<A>(), 0);
        Self {
            ptr: NonNull::dangling(),
            len,
            cap: usize::MAX,
            align: align_of::<A>().max(1),
            _marker: PhantomData,
        }
    }

    pub(crate) fn with_capacity_aligned(cap: usize, align: usize) -> Result<Self, XenonError> {
        let align = align.max(align_of::<A>().max(1));
        if size_of::<A>() == 0 {
            return Ok(Self::zst(0));
        }
        if cap == 0 {
            return Ok(Self::empty());
        }
        let size = allocation_size::<A>(cap, align, "AlignedBuf::with_capacity_aligned")?;
        let ptr = AlignedAlloc::alloc(size, align)?;
        // SAFETY: AlignedAlloc::alloc returned a valid aligned allocation
        Ok(unsafe { Self::from_raw_parts(ptr.as_ptr() as *mut A, 0, cap, align) })
    }

    /// Copies elements from a `Vec` into a 64-byte aligned buffer.
    pub(crate) fn from_vec(data: Vec<A>) -> Result<Self, XenonError>
    where
        A: Copy,
    {
        Self::from_vec_aligned(data)
    }

    /// Core implementation of `from_vec` with explicit alignment.
    pub(crate) fn from_vec_aligned(data: Vec<A>) -> Result<Self, XenonError>
    where
        A: Copy,
    {
        let len = data.len();
        if size_of::<A>() == 0 {
            return Ok(Self::zst(len));
        }
        if len == 0 {
            return Ok(Self::empty());
        }
        let align = align_of::<A>().max(AlignedAlloc::DEFAULT_ALIGNMENT);
        let size = allocation_size::<A>(len, align, "AlignedBuf::from_vec_aligned")?;
        let ptr = AlignedAlloc::alloc(size, align)?;
        let typed_ptr = ptr.as_ptr() as *mut A;
        // SAFETY: typed_ptr and data.as_ptr() are valid, non-overlapping
        unsafe {
            core::ptr::copy_nonoverlapping(data.as_ptr(), typed_ptr, len);
        }
        drop(data);
        // SAFETY: allocated by AlignedAlloc, len elements initialized
        Ok(unsafe { Self::from_raw_parts(typed_ptr, len, len, align) })
    }

    /// Creates a 64-byte aligned buffer filled with zeros.
    pub(crate) fn zeros(len: usize) -> Result<Self, XenonError>
    where
        A: Element + Default,
    {
        if size_of::<A>() == 0 {
            return Ok(Self::zst(len));
        }
        if len == 0 {
            return Ok(Self::empty());
        }
        let align = align_of::<A>().max(AlignedAlloc::DEFAULT_ALIGNMENT);
        let size = allocation_size::<A>(len, align, "AlignedBuf::zeros")?;
        let ptr = AlignedAlloc::alloc_zeroed(size, align)?;
        // SAFETY: alloc_zeroed returned valid zeroed memory
        Ok(unsafe { Self::from_raw_parts(ptr.as_ptr() as *mut A, len, len, align) })
    }

    /// Creates a 64-byte aligned buffer filled with clones of `value`.
    pub(crate) fn from_elem(len: usize, value: A) -> Result<Self, XenonError>
    where
        A: Clone,
    {
        let align = align_of::<A>().max(AlignedAlloc::DEFAULT_ALIGNMENT);
        let mut buf = Self::with_capacity_aligned(len, align)?;
        for index in 0..len {
            // SAFETY: capacity >= len, ptr valid
            unsafe {
                core::ptr::write(buf.as_mut_ptr().add(index), value.clone());
            }
        }
        // SAFETY: all len elements initialized
        unsafe {
            buf.set_len(len);
        }
        Ok(buf)
    }

    /// Creates an `AlignedBuf` from raw components.
    ///
    /// # Safety
    ///
    /// - `ptr` must be non-null, allocated by `AlignedAlloc` (or
    ///   `NonNull::dangling` for ZST/empty buffers), and valid for `cap`
    ///   elements of type `A`.
    /// - The first `len` elements must be initialized (unless ZST).
    /// - `align` must equal the actual allocation alignment and satisfy
    ///   `align >= align_of::<A>()`.
    /// - `cap` must be the actual allocated capacity in elements.
    pub(crate) unsafe fn from_raw_parts(ptr: *mut A, len: usize, cap: usize, align: usize) -> Self {
        Self {
            // SAFETY: caller guarantees ptr is non-null
            ptr: unsafe { NonNull::new_unchecked(ptr) },
            len,
            cap,
            align,
            _marker: PhantomData,
        }
    }

    pub(crate) fn as_ptr(&self) -> *const A {
        self.ptr.as_ptr()
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut A {
        self.ptr.as_ptr()
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn capacity(&self) -> usize {
        self.cap
    }

    pub(crate) fn alignment(&self) -> usize {
        self.align
    }

    /// Sets the logical length of the buffer.
    ///
    /// # Safety
    ///
    /// - `len` must not exceed `self.cap` (unless ZST, where `cap` is
    ///   `usize::MAX`).
    /// - Elements in `self.len..len` must be initialized (unless ZST).
    pub(crate) unsafe fn set_len(&mut self, len: usize) {
        debug_assert!(len <= self.cap || size_of::<A>() == 0);
        self.len = len;
    }
}

impl<A> Drop for AlignedBuf<A> {
    fn drop(&mut self) {
        if size_of::<A>() == 0 || self.cap == 0 {
            return;
        }
        for index in 0..self.len {
            // SAFETY: ptr is valid for len elements; each element is
            // initialized by construction contract
            unsafe {
                core::ptr::drop_in_place(self.ptr.as_ptr().add(index));
            }
        }
        let size = self.cap * size_of::<A>();
        // SAFETY: ptr was allocated by AlignedAlloc with (size, self.align)
        unsafe {
            AlignedAlloc::dealloc(self.ptr.cast(), size, self.align);
        }
    }
}

// ---------------------------------------------------------------------------
// SharedBuf<A> — Arc-wrapped shared buffer
// ---------------------------------------------------------------------------

/// Internal shared buffer wrapping an `AlignedBuf`.
///
/// When the last `Arc` is dropped, `AlignedBuf`'s `Drop` releases the
/// aligned allocation with the original layout.
#[derive(Debug)]
pub(crate) struct SharedBuf<A> {
    pub(crate) buf: AlignedBuf<A>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // SharedBuf tests
    // -----------------------------------------------------------------------

    /// SharedBuf compiles and wraps an AlignedBuf correctly.
    #[test]
    fn test_shared_buf_compile() {
        let buf = AlignedBuf::<i32>::zeros(4)
            .expect("AlignedBuf::zeros should succeed for small i32 input");
        let shared = SharedBuf { buf };
        assert_eq!(shared.buf.len(), 4);
        assert!(!shared.buf.as_ptr().is_null());
    }

    // -----------------------------------------------------------------------
    // AlignedBuf tests (moved from owned.rs)
    // -----------------------------------------------------------------------

    /// Alignment is clamped to at least the element type's alignment.
    #[test]
    fn test_aligned_buf_with_capacity_clamps_alignment_to_element_requirement() {
        let buf = AlignedBuf::<u128>::with_capacity_aligned(4, 1)
            .expect("AlignedBuf::with_capacity_aligned should honor element alignment");
        assert_eq!((buf.as_ptr() as usize) % align_of::<u128>(), 0);
        assert_eq!(buf.capacity(), 4);
    }

    /// ZST buffers work without triggering undefined behavior.
    #[test]
    fn test_zst_no_ub() {
        // Owned::zeros requires A: Element + Default, which () does not
        // satisfy. Test the ZST code path through AlignedBuf directly.
        let buf = AlignedBuf::<()>::zst(1000);
        assert_eq!(buf.len(), 1000);
    }
}
