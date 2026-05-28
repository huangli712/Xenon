//! Shared buffer wrapping an aligned allocation.
//!
//! When the last `Arc` to the buffer is dropped, `AlignedBuf`'s `Drop`
//! releases the aligned allocation with the original layout.

use crate::storage::owned::AlignedBuf;

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

    /// SharedBuf compiles and wraps an AlignedBuf correctly.
    #[test]
    fn test_shared_buf_compile() {
        let buf = AlignedBuf::<i32>::zeros(4)
            .expect("AlignedBuf::zeros should succeed for small i32 input");
        let shared = SharedBuf { buf };
        assert_eq!(shared.buf.len(), 4);
        assert!(!shared.buf.as_ptr().is_null());
    }
}
