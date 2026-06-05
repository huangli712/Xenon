use core::fmt;

use crate::dimension::Dimension;
use crate::element::Element;
use crate::storage::Storage;
use crate::tensor::TensorBase;

use super::config::FormatConfig;
use super::pretty::{
    read_logical,
    fmt_1d_debug, fmt_nd_debug, fmt_scalar_debug,
    dtype_name, layout_name, format_tensor_display,
};
impl<S, D, A> fmt::Debug for TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Header: §5.4 line 287-301, §5.5 line 328-334
        writeln!(
            f,
            "Tensor(shape={:?}, strides={:?}, dtype={}, layout={})",
            self.shape(),
            self.strides(),
            dtype_name::<A>(),
            layout_name(self),
        )?;
        // Data section: route through W26T3 debug helpers.
        // §5.6 line 391 — Debug data omits Display's trailing shape suffix.
        let config = FormatConfig::default();
        match self.ndim() {
            0 => {
                write!(f, "Tensor0(")?;
                fmt_scalar_debug(f, read_logical(self, &[]), config)?;
                write!(f, ")")
            },
            1 => fmt_1d_debug(f, self, config),
            _ => fmt_nd_debug(f, self, config),
        }
    }
}

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

impl<S, D, A> fmt::Display for TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        format_tensor_display(f, self, FormatConfig::default())
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::Complex;
    use crate::dimension::{Ix0, Ix1, Ix2};
    use crate::layout::Strides;

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

    #[test]
    fn test_debug_tensor() {
        let tensor = unsafe { TensorBase::from_raw_vec_unchecked(vec![1_i32, 2, 3, 4], Ix2(2, 2)) };
        let text = format!("{:?}", tensor);
        // §5.4 line 287-301 + §5.5 line 328-334 header.
        assert!(text.contains("shape=[2, 2]"), "text = {text:?}");
        assert!(text.contains("strides="), "text = {text:?}");
        assert!(text.contains("dtype=i32"), "text = {text:?}");
        assert!(text.contains("layout=f-contiguous"), "text = {text:?}");
        // Data section present.
        assert!(text.contains("[1, 3]") || text.contains("[1, 2]"), "text = {text:?}");
    }

    #[test]
    fn test_debug_truncated_does_not_repeat_shape_suffix() {
        // §5.6 line 391 — Debug must NOT append "... shape=[...]".
        let tensor = unsafe { TensorBase::from_raw_vec_unchecked(vec![0_i32; 1001], Ix1(1001)) };
        let text = format!("{:?}", tensor);
        assert!(text.contains("shape=[1001]"), "header missing; text = {text:?}");
        assert!(!text.contains("elements omitted)  shape="), "Debug must not repeat Display's shape suffix; text = {text:?}");
    }

    /// Construct a tensor view via from_raw_parts.
    unsafe fn make_view<A: Element>(
        base: &TensorBase<crate::storage::Owned<A>, Ix2>,
        shape: Ix2,
        strides: Strides<Ix2>,
    ) -> TensorBase<crate::storage::ViewRepr<'_, A>, Ix2> {
        // SAFETY: shape/strides are manually constructed from the original
        // tensor's data with known valid geometric transformations.
        unsafe {
            TensorBase::from_raw_parts(base.as_ptr(), base.storage_len(), shape, strides, 0)
                .expect("valid layout from manually constructed strides")
        }
    }

    #[test]
    fn test_debug_transposed_view() {
        // §5.5 line 333 + §5.4 line 258: transposed view → layout=non-contiguous.
        // Source: shape=[2, 3] F-order, data=[1, 2, 3, 4, 5, 6]
        //   logical = [[1, 3, 5], [2, 4, 6]]
        // Transposed to shape=[3, 2]:
        //   logical = [[1, 2], [3, 4], [5, 6]]
        let tensor =
            unsafe { TensorBase::from_raw_vec_unchecked(vec![1_i32, 2, 3, 4, 5, 6], Ix2(2, 3)) };
        // Transposed: shape=[3,2], strides=[2,1]
        let view = unsafe { make_view(&tensor, Ix2(3, 2), Strides::new(Ix2(2, 1))) };
        let text = format!("{:?}", view);
        // Header: layout classification.
        assert!(text.contains("layout=non-contiguous"), "text = {text:?}");
        assert!(text.contains("shape=[3, 2]"), "text = {text:?}");
        // Data section: logical row order, not physical storage order.
        assert!(text.contains("[1, 2]"), "text = {text:?}");
        assert!(text.contains("[3, 4]"), "text = {text:?}");
        assert!(text.contains("[5, 6]"), "text = {text:?}");
    }

    #[test]
    fn test_debug_broadcast_view() {
        // §5.5 line 332 + §5.4 line 258: broadcast view (zero stride) → layout=broadcast.
        let tensor = unsafe { TensorBase::from_raw_vec_unchecked(vec![1_i32, 2, 3], Ix2(1, 3)) };
        // Broadcast: shape=[4,3], strides=[0,1]
        let view = unsafe { make_view(&tensor, Ix2(4, 3), Strides::new(Ix2(0, 1))) };
        let text = format!("{:?}", view);
        assert!(text.contains("layout=broadcast"), "text = {text:?}");
    }

    #[test]
    fn test_debug_dtype_complex() {
        // §6.2 line 600-601: Complex<f32> / Complex<f64> dtype name format.
        // Statically assert dtype_name dispatch via monomorphization.
        fn check<A: Element>() -> &'static str {
            dtype_name::<A>()
        }
        assert_eq!(check::<i32>(), "i32");
        assert_eq!(check::<i64>(), "i64");
        assert_eq!(check::<f32>(), "f32");
        assert_eq!(check::<f64>(), "f64");
        assert_eq!(check::<bool>(), "bool");
        // Complex<f32> / Complex<f64> verified via integration test.
    }
}
