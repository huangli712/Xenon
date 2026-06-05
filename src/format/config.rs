//! Tensor output configuration — truncation thresholds, precision, and
//! line-wrap parameters.

/// Formatting output configuration.
///
/// Controls truncation behavior and display parameters for large arrays.
/// All fields are public so callers can construct instances with struct
/// literal syntax.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FormatConfig {
    /// Number of edge items shown on each side when truncation is active.
    /// Defaults to 3. A value of 0 is normalized to 1 at the point of use.
    pub edge_items: usize,

    /// Total element count strictly greater than this triggers truncation.
    /// Defaults to 1000.
    pub threshold: usize,

    /// Optional floating-point decimal precision. `None` means use the
    /// type's default Display formatting (no `.precision` modifier).
    pub precision: Option<usize>,

    /// Maximum line width in characters for soft line-break decisions.
    /// Only affects line breaks; never changes truncation outcomes.
    /// Defaults to 80.
    pub line_width: usize,
}

impl FormatConfig {
    /// Returns `edge_items` clamped to a minimum of 1.
    ///
    /// Used internally by the pretty printers so that a user-configured
    /// value of 0 still produces at least one edge item.
    pub(crate) fn normalized_edge_items(self) -> usize {
        self.edge_items.max(1)
    }
}

impl Default for FormatConfig {
    /// Returns a `FormatConfig` with sensible defaults:
    /// `edge_items=3`, `threshold=1000`, `precision=None`, `line_width=80`.
    fn default() -> Self {
        Self {
            edge_items: 3,
            threshold: 1000,
            precision: None,
            line_width: 80,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify all four default values match the spec: `edge_items=3`,
    /// `threshold=1000`, `precision=None`, `line_width=80`.
    #[test]
    fn test_format_config_default() {
        let config = FormatConfig::default();
        assert_eq!(config.edge_items, 3);
        assert_eq!(config.threshold, 1000);
        assert_eq!(config.precision, None);
        assert_eq!(config.line_width, 80);
    }

    /// `edge_items=0` normalizes to 1 so truncation always shows
    /// at least one head and one tail element.
    #[test]
    fn test_format_config_edge_items_zero_normalizes_to_one() {
        let config = FormatConfig {
            edge_items: 0,
            ..Default::default()
        };
        assert_eq!(config.normalized_edge_items(), 1);
    }

    /// Non-zero `edge_items` values pass through `normalized_edge_items`
    /// unchanged (no clamping when already ≥ 1).
    #[test]
    fn test_format_config_edge_items_passthrough() {
        // Non-zero values pass through unchanged.
        let config = FormatConfig {
            edge_items: 5,
            ..Default::default()
        };
        assert_eq!(config.normalized_edge_items(), 5);
    }
}
