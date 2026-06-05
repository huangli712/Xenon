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
        // Reset column after the last '\n'; otherwise accumulate.
        match s.rfind('\n') {
            Some(p) => self.column = s[p + 1..].chars().count(),
            None => self.column += s.chars().count(),
        }
        Ok(())
    }
}
