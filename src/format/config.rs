/// Formatting output configuration.
///
/// Controls truncation behavior and display parameters for large arrays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FormatConfig {
    /// Number of edge items shown on each side when truncation is active.
    /// Defaults to 3. Configured 0 is normalized to 1 at use site.
    pub edge_items: usize,

    /// Element count strictly greater than this triggers truncation.
    /// Defaults to 1000.
    pub threshold: usize,

    /// Optional floating-point decimal precision; `None` = type's default
    /// Display formatting (no `.precision` modifier).
    pub precision: Option<usize>,

    /// Maximum line width in characters for soft line-break decisions.
    /// Only affects line breaks; never changes truncation outcomes.
    /// Defaults to 80.
    pub line_width: usize,
}

impl Default for FormatConfig {
    fn default() -> Self {
        Self {
            edge_items: 3,
            threshold: 1000,
            precision: None,
            line_width: 80,
        }
    }
}

impl FormatConfig {
    /// Crate-private helper used by pretty printers to honor §5.6 line 384:
    /// `edge_items = 0` behaves as 1 without mutating the user's config.
    pub(crate) fn normalized_edge_items(self) -> usize {
        self.edge_items.max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_config_default() {
        let config = FormatConfig::default();
        assert_eq!(config.edge_items, 3);
        assert_eq!(config.threshold, 1000);
        assert_eq!(config.precision, None);
        assert_eq!(config.line_width, 80);
    }

    #[test]
    fn test_format_config_edge_items_zero_normalizes_to_one() {
        let config = FormatConfig {
            edge_items: 0,
            ..Default::default()
        };
        assert_eq!(config.normalized_edge_items(), 1);
    }

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
