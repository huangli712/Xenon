//! `TensorDisplay` wrapper and `Display` impl — configurable tensor
//! formatting adapter constructed via `TensorBase::display_with`.

use core::fmt;

use crate::dimension::Dimension;
use crate::element::Element;
use crate::storage::Storage;
use crate::tensor::TensorBase;

use super::config::FormatConfig;
use super::pretty::format_tensor_display;

/// A wrapper that formats a tensor with a specific [`FormatConfig`].
///
/// Constructed via `TensorBase::display_with`. Implements `core::fmt::Display`
/// so it can be used directly in `format!` / `write!` macros.
pub struct TensorDisplay<'a, S, D, A>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// Borrowed reference to the underlying tensor.
    pub(crate) tensor: &'a TensorBase<S, D>,

    /// Configuration for formatting: edge_items, threshold, precision, etc.
    pub(crate) config: FormatConfig,
}

impl<'a, S, D, A> fmt::Display for TensorDisplay<'a, S, D, A>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + fmt::Display,
{
    /// Delegates to the central display formatting pipeline with the config
    /// stored in this wrapper.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        format_tensor_display(f, self.tensor, self.config)
    }
}

impl<'a, S, D, A> fmt::Debug for TensorDisplay<'a, S, D, A>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + fmt::Debug,
{
    /// Formats the wrapper as `TensorDisplay { tensor: …, config: … }`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorDisplay")
            .field("tensor", &self.tensor)
            .field("config", &self.config)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::Ix1;

    /// Verifies that `FormatConfig::precision` controls the number of
    /// decimal places in float Display output.
    #[test]
    fn test_fmt_float_precision() {
        let tensor = unsafe {
            TensorBase::from_raw_vec_unchecked(
                vec![1.234_f64],
                Ix1(1)
            )
        };
        let config = FormatConfig {
            precision: Some(2),
            ..Default::default()
        };
        assert_eq!(format!("{}", tensor.display_with(config)), "[1.23]");
    }
}
