use core::fmt::{self, Formatter};

/// Tracks the current character-column position on the most recent line.
///
/// Used by the pretty-printing pipeline to decide when to soft-wrap at
/// innermost-axis element boundaries: when the current column reaches or
/// exceeds [`FormatConfig::line_width`], the next element starts on a
/// new line with indentation.
///
/// Column counts Unicode scalar values (`char`), not bytes or grapheme
/// clusters, consistent with `line_width` being a per-character budget.
pub(crate) struct LineWriter<'a, 'b> {
    inner: &'a mut Formatter<'b>,
    column: usize,
}

impl<'a, 'b> LineWriter<'a, 'b> {
    /// Creates a new `LineWriter` wrapping the given formatter, starting
    /// at column 0 (beginning of a line).
    pub(crate) fn new(inner: &'a mut Formatter<'b>) -> Self {
        Self { inner, column: 0 }
    }

    /// Returns the current column position (number of Unicode scalar
    /// values since the last newline).
    pub(crate) fn column(&self) -> usize {
        self.column
    }
}

impl fmt::Write for LineWriter<'_, '_> {
    /// Writes the string slice to the underlying formatter and updates
    /// the column counter.
    ///
    /// If the inner formatter fails partway through writing, the column
    /// may be left inconsistent — there is no rollback API in `fmt::Write`.
    /// This is the same trade-off every `Write` implementation makes.
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.inner.write_str(s)?;
        // Reset column after the last '\n'; otherwise accumulate.
        match s.rfind('\n') {
            Some(p) => self.column = s[p + 1..].chars().count(),
            None => self.column += s.chars().count(),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::fmt::Write;

    /// Drive a `LineWriter` through a `Display` impl and return both the
    /// formatted output string and the final column position.
    fn run_probe(ops: impl Fn(&mut LineWriter<'_, '_>) -> fmt::Result) -> (String, usize) {
        struct Probe<F> {
            ops: F,
            column: std::cell::Cell<usize>,
        }
        impl<F: Fn(&mut LineWriter<'_, '_>) -> fmt::Result> fmt::Display for Probe<F> {
            fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                let mut w = LineWriter::new(f);
                (self.ops)(&mut w)?;
                self.column.set(w.column());
                Ok(())
            }
        }
        let probe = Probe {
            ops,
            column: std::cell::Cell::new(0),
        };
        let mut s = String::new();
        let _ = write!(&mut s, "{}", probe);
        (s, probe.column.get())
    }

    /// A fresh `LineWriter` starts at column 0.
    #[test]
    fn test_line_writer_initial_column() {
        let (_, col) = run_probe(|_| Ok(()));
        assert_eq!(col, 0);
    }

    /// Writing an empty string produces no output and does not change
    /// the column.
    #[test]
    fn test_line_writer_empty_string() {
        let (output, col) = run_probe(|w| w.write_str(""));
        assert_eq!(output, "");
        assert_eq!(col, 0);
    }

    /// Writing a string without newlines accumulates the character count.
    #[test]
    fn test_line_writer_no_newline() {
        let (output, col) = run_probe(|w| w.write_str("hello"));
        assert_eq!(output, "hello");
        assert_eq!(col, 5);
    }

    /// A trailing newline resets the column to 0.
    #[test]
    fn test_line_writer_trailing_newline() {
        let (output, col) = run_probe(|w| w.write_str("abc\n"));
        assert_eq!(output, "abc\n");
        assert_eq!(col, 0);
    }

    /// Writing only a newline produces just the newline and column 0.
    #[test]
    fn test_line_writer_only_newline() {
        let (output, col) = run_probe(|w| w.write_str("\n"));
        assert_eq!(output, "\n");
        assert_eq!(col, 0);
    }

    /// A newline in the middle resets the column to count characters
    /// after the last newline.
    #[test]
    fn test_line_writer_newline_in_middle() {
        let (output, col) = run_probe(|w| w.write_str("abc\ndef"));
        assert_eq!(output, "abc\ndef");
        assert_eq!(col, 3);
    }

    /// Multiple newlines — only the characters after the last one count
    /// toward the column.
    #[test]
    fn test_line_writer_multiple_newlines() {
        let (output, col) = run_probe(|w| w.write_str("a\nb\ncd"));
        assert_eq!(output, "a\nb\ncd");
        assert_eq!(col, 2);
    }

    /// Column accumulation across multiple `write_str` calls adds
    /// correctly.
    #[test]
    fn test_line_writer_multi_write_accumulation() {
        let col_after_first = std::cell::Cell::new(0usize);
        let (output, final_col) = run_probe(|w| {
            w.write_str("ab")?;
            col_after_first.set(w.column());
            w.write_str("cd")?;
            Ok(())
        });
        assert_eq!(output, "abcd");
        assert_eq!(col_after_first.get(), 2);
        assert_eq!(final_col, 4);
    }

    /// A newline inside a `write_str` call resets the column to what
    /// follows the newline.
    #[test]
    fn test_line_writer_multi_write_with_newline() {
        let (output, col) = run_probe(|w| {
            w.write_str("ab")?;
            w.write_str("\nde")?;
            Ok(())
        });
        assert_eq!(output, "ab\nde");
        assert_eq!(col, 2);
    }

    /// Column counts Unicode scalar values, not UTF-8 bytes.
    /// "你好" is 2 chars in 6 bytes.
    #[test]
    fn test_line_writer_unicode_chars() {
        let (output, col) = run_probe(|w| w.write_str("你好"));
        assert_eq!(output, "你好");
        assert_eq!(col, 2);
    }

    /// After a newline, column counts only the Unicode scalars on the
    /// current line.
    #[test]
    fn test_line_writer_unicode_after_newline() {
        let (output, col) = run_probe(|w| w.write_str("x\n你好"));
        assert_eq!(output, "x\n你好");
        assert_eq!(col, 2);
    }

    /// Column counts Unicode scalar values, not grapheme clusters.
    /// A decomposed 'é' (e + combining acute accent) is 2 scalars.
    #[test]
    fn test_line_writer_combining_chars_scalar_count() {
        let (output, col) = run_probe(|w| w.write_str("e\u{301}"));
        assert_eq!(output, "e\u{301}");
        assert_eq!(col, 2);
    }
}
