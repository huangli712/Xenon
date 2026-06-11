//! Display implementations and formatting helpers for error types.
//!
//! Provides [`Display`] impls for all error enums and the [`FmtShape`]
//! helper for rendering `[usize]` slices in numpy-style bracket notation
//! (e.g., `[2 × 3 × 4]`).

use core::fmt::{self, Display, Formatter};

/// Helper for formatting `[usize]` shape/stride slices in error messages.
///
/// Output format: `[]`、`[5]`、`[2 × 3 × 4]` — numpy style.
pub(super) struct FmtShape<'a>(pub(super) &'a [usize]);

impl<'a> Display for FmtShape<'a> {
    /// Formats the shape slice in numpy-style bracket notation.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, dim) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, " × ")?;
            }
            write!(f, "{dim}")?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::FmtShape;

    /// Verify empty `FmtShape` renders as `[]`.
    #[test]
    fn test_fmt_shape_empty() {
        assert_eq!(format!("{}", FmtShape(&[])), "[]");
    }

    /// Verify single-dimension `FmtShape` renders as `[N]`.
    #[test]
    fn test_fmt_shape_1d() {
        assert_eq!(format!("{}", FmtShape(&[5])), "[5]");
    }

    /// Verify multi-dimension `FmtShape` renders as `[a × b × c]`.
    #[test]
    fn test_fmt_shape_3d() {
        assert_eq!(format!("{}", FmtShape(&[2, 3, 4])), "[2 × 3 × 4]");
    }
}
