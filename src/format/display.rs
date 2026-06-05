use core::fmt;

use crate::dimension::Dimension;
use crate::element::Element;
use crate::storage::Storage;
use crate::tensor::TensorBase;

use super::config::FormatConfig;
use super::pretty::format_tensor_display;

/// A wrapper that formats a tensor with a specific [`FormatConfig`](crate::format::FormatConfig).
///
/// Constructed via [`TensorBase::display_with`]. Implements [`core::fmt::Display`]
/// so it can be used directly in `format!` / `write!` macros.
#[expect(
    missing_debug_implementations,
    reason = "wrapper type; only used as a formatting adapter"
)]
pub struct TensorDisplay<'a, S, D, A>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    tensor: &'a TensorBase<S, D>,
    config: FormatConfig,
}

impl<'a, S, D, A> fmt::Display for TensorDisplay<'a, S, D, A>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        format_tensor_display(f, self.tensor, self.config)
    }
}

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element,
{
    /// Returns a display wrapper that formats this tensor with the given config.
    pub fn display_with(&self, config: FormatConfig) -> TensorDisplay<'_, S, D, A> {
        TensorDisplay {
            tensor: self,
            config,
        }
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

    #[test]
    fn test_fmt_zero_dim() {
        // 18-construction.md §5 — scalar tensor constructed via raw parts.
        let tensor = unsafe { TensorBase::from_raw_vec_unchecked(vec![42_i32], Ix0) };
        assert_eq!(format!("{}", tensor), "Tensor0(42)");
    }

    /// 22-output.md §8.2 line 681 (medium priority) — Complex default formatting
    /// (positive imaginary part). Verifies W5T7 `Complex<T>::Display`
    /// reaches tensor output via W26T4 routing and still respects
    /// §6.1 line 502-515 concatenation rules.
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
        // Rust f64 Display: `1.0_f64` defaults to "1"; Complex Display uses
        // "+" separator when im ≥ 0 (§6.1 line 504 row).
        assert_eq!(format!("{}", tensor), "[1+2j, 3+4j]");
    }

    /// 22-output.md §8.2 line 682 (medium priority) — Complex negative imag part.
    /// Verifies §6.1 line 507 row: when im < 0, `is_sign_negative()` selects "-".
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
        // im=-2 selects "-", mag=|im|=2 outputs "2" (f64 Display default).
        // im=-4 same. §6.1 line 514-515: `im.abs()` unifies magnitude.
        assert_eq!(format!("{}", tensor), "[1-2j, -3-4j]");
    }

    /// 22-output.md §8.3 line 703 — NaN/Inf boundary scenario.
    /// Verifies that float special values pass through W26T3
    /// `fmt_scalar_display` delegating to Rust's default
    /// `f64::Display`: NaN→"NaN", ±∞→"±inf" (§6.1 line 378).
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
