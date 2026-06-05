//! `TensorDisplay` wrapper and `Display` impl — configurable tensor formatting
//! adapter constructed via `TensorBase::display_with`.

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
    /// Manual impl: `#[derive(Debug)]` would add wrong bounds
    /// (`S: Debug, D: Debug, A: Debug`); `TensorBase`'s `Debug` actually
    /// requires `A: Element + fmt::Debug`, mirrored here.
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
    use crate::complex::Complex;
    use crate::dimension::Ix0;

    #[test]
    fn test_display_tensor() {
        let tensor =
            unsafe { TensorBase::from_raw_vec_unchecked(vec![1, 2, 3], crate::dimension::Ix1(3)) };
        assert_eq!(format!("{}", tensor), "[1, 2, 3]");
    }

    /// Verifies that `FormatConfig::precision` controls the number of
    /// decimal places in float Display output.
    #[test]
    fn test_fmt_float_precision() {
        let tensor = unsafe {
            TensorBase::from_raw_vec_unchecked(vec![1.234_f64], crate::dimension::Ix1(1))
        };
        let config = FormatConfig {
            precision: Some(2),
            ..Default::default()
        };
        assert_eq!(format!("{}", tensor.display_with(config)), "[1.23]");
    }

    /// Verifies that a scalar tensor (zero dimensions) renders with the
    /// explicit `Tensor0(...)` marker, not just the bare value.
    #[test]
    fn test_fmt_zero_dim() {
        let tensor = unsafe { TensorBase::from_raw_vec_unchecked(vec![42_i32], Ix0) };
        assert_eq!(format!("{}", tensor), "Tensor0(42)");
    }

    /// Verifies that Complex values with a positive imaginary part render
    /// with a `+` separator (e.g. `1+2j`).
    #[test]
    fn test_display_complex_f64() {
        let tensor = unsafe {
            TensorBase::from_raw_vec_unchecked(
                vec![
                    Complex::new(1.0_f64, 2.0_f64),
                    Complex::new(3.0_f64, 4.0_f64),
                ],
                crate::dimension::Ix1(2),
            )
        };
        assert_eq!(format!("{}", tensor), "[1+2j, 3+4j]");
    }

    /// Verifies that Complex values with a negative imaginary part render
    /// with a `-` separator (e.g. `1-2j`).
    #[test]
    fn test_display_complex_negative_imag() {
        let tensor = unsafe {
            TensorBase::from_raw_vec_unchecked(
                vec![
                    Complex::new(1.0_f64, -2.0_f64),
                    Complex::new(-3.0_f64, -4.0_f64),
                ],
                crate::dimension::Ix1(2),
            )
        };
        assert_eq!(format!("{}", tensor), "[1-2j, -3-4j]");
    }

    /// Verifies that f64 special values (NaN, ±∞) pass through the Display
    /// pipeline correctly — `f64::Display` produces `"NaN"`, `"inf"`, `"-inf"`.
    #[test]
    fn test_display_nan_inf_f64() {
        let tensor = unsafe {
            TensorBase::from_raw_vec_unchecked(
                vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY],
                crate::dimension::Ix1(3),
            )
        };
        assert_eq!(format!("{}", tensor), "[NaN, inf, -inf]");
    }
}
