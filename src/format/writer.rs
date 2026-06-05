use core::fmt::{self, Formatter};

/// Formatter wrapper that tracks the current column (characters since the
/// last newline). Used by `write_separator` to soft-wrap at innermost-axis
/// element boundaries when column >= `FormatConfig::line_width`
/// (`22-output §5.6` line 392).
///
/// `column` counts `char` units (Unicode scalar values), consistent with
/// `FormatConfig::line_width: usize` being a character budget, not a byte
/// budget. The helper re-uses `s.chars().count()` on every written fragment;
/// short scalar / separator writes make this O(fragment size).
pub(crate) struct LineWriter<'a, 'b> {
    inner: &'a mut Formatter<'b>,
    column: usize,
}

impl<'a, 'b> LineWriter<'a, 'b> {
    pub(crate) fn new(inner: &'a mut Formatter<'b>) -> Self {
        Self { inner, column: 0 }
    }

    pub(crate) fn column(&self) -> usize {
        self.column
    }
}

impl fmt::Write for LineWriter<'_, '_> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.inner.write_str(s)?;
        // If the inner Formatter fails, column is left inconsistent — the
        // Formatter may have partially written before erroring and there is
        // no rollback API. This is the same tradeoff every fmt::Write impl
        // makes.
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

    /// Drive a `LineWriter` from inside a `Display` impl and return both
    /// the formatted output and the final column position.
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

    #[test]
    fn test_line_writer_initial_column() {
        let (_, col) = run_probe(|_| Ok(()));
        assert_eq!(col, 0);
    }

    #[test]
    fn test_line_writer_empty_string() {
        let (output, col) = run_probe(|w| w.write_str(""));
        assert_eq!(output, "");
        assert_eq!(col, 0);
    }

    #[test]
    fn test_line_writer_no_newline() {
        let (output, col) = run_probe(|w| w.write_str("hello"));
        assert_eq!(output, "hello");
        assert_eq!(col, 5);
    }

    #[test]
    fn test_line_writer_trailing_newline() {
        let (output, col) = run_probe(|w| w.write_str("abc\n"));
        assert_eq!(output, "abc\n");
        assert_eq!(col, 0);
    }

    #[test]
    fn test_line_writer_only_newline() {
        let (output, col) = run_probe(|w| w.write_str("\n"));
        assert_eq!(output, "\n");
        assert_eq!(col, 0);
    }

    #[test]
    fn test_line_writer_newline_in_middle() {
        let (output, col) = run_probe(|w| w.write_str("abc\ndef"));
        assert_eq!(output, "abc\ndef");
        assert_eq!(col, 3);
    }

    #[test]
    fn test_line_writer_multiple_newlines() {
        let (output, col) = run_probe(|w| w.write_str("a\nb\ncd"));
        assert_eq!(output, "a\nb\ncd");
        assert_eq!(col, 2);
    }

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

    #[test]
    fn test_line_writer_unicode_chars() {
        // "你好" is 2 chars but 6 UTF-8 bytes.
        let (output, col) = run_probe(|w| w.write_str("你好"));
        assert_eq!(output, "你好");
        assert_eq!(col, 2);
    }

    #[test]
    fn test_line_writer_unicode_after_newline() {
        let (output, col) = run_probe(|w| w.write_str("x\n你好"));
        assert_eq!(output, "x\n你好");
        assert_eq!(col, 2);
    }

    #[test]
    fn test_line_writer_combining_chars_scalar_count() {
        // Column counts Unicode scalar values, not grapheme clusters.
        // Decomposed 'é' = 'e' + combining acute: 2 scalars, 1 grapheme.
        let (output, col) = run_probe(|w| w.write_str("e\u{301}"));
        assert_eq!(output, "e\u{301}");
        assert_eq!(col, 2);
    }
}
