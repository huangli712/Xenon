//! `Debug` and `Display` trait implementations for `TensorBase`, plus the
//! `display_with` constructor method.

use core::fmt;

use crate::dimension::Dimension;
use crate::element::Element;
use crate::storage::Storage;
use crate::tensor::TensorBase;

use super::config::FormatConfig;
use super::display::TensorDisplay;
use super::pretty::{format_tensor_display, format_tensor_debug};

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element,
{
    /// Returns a display wrapper that formats this tensor with the
    /// given config.
    ///
    /// The returned `TensorDisplay` implements `core::fmt::Display`, so
    /// it can be used directly in `format!` / `write!` macros.
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
    /// Renders the tensor data in logical index order using Display
    /// formatting. Zero-dimension tensors get an explicit `Tensor0(...)`
    /// marker. Truncation is controlled by [`FormatConfig::default()`].
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        format_tensor_display(f, self, FormatConfig::default())
    }
}

impl<S, D, A> fmt::Debug for TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + fmt::Debug,
{
    /// Writes a header line with shape, strides, dtype, and layout
    /// metadata, followed by the tensor data rendered in logical
    /// index order using Debug formatting for each element.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        format_tensor_debug(f, self, FormatConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::Complex;
    use crate::dimension::{Ix0, Ix1, Ix2};
    use crate::layout::Strides;
    use crate::element::element_type_of;

    /// Verifies that Debug output contains the expected header fields
    /// (shape, strides, dtype, layout) and data section.
    #[test]
    fn test_debug_tensor() {
        let tensor = unsafe { TensorBase::from_raw_vec_unchecked(vec![1_i32, 2, 3, 4], Ix2(2, 2)) };
        let text = format!("{:?}", tensor);
        assert!(text.contains("shape=[2, 2]"), "text = {text:?}");
        assert!(text.contains("strides="), "text = {text:?}");
        assert!(text.contains("dtype=i32"), "text = {text:?}");
        assert!(text.contains("layout=f-contiguous"), "text = {text:?}");
        assert!(text.contains("[1, 3]") || text.contains("[1, 2]"), "text = {text:?}");
    }

    /// Verifies that Debug truncation does NOT append the Display-only
    /// trailing shape suffix (` ... (N omitted)  shape=[...]`).
    #[test]
    fn test_debug_truncated_does_not_repeat_shape_suffix() {
        let tensor = unsafe { TensorBase::from_raw_vec_unchecked(vec![0_i32; 1001], Ix1(1001)) };
        let text = format!("{:?}", tensor);
        assert!(text.contains("shape=[1001]"), "header missing; text = {text:?}");
        assert!(!text.contains("elements omitted)  shape="), "Debug must not repeat Display's shape suffix; text = {text:?}");
    }

    /// Construct a tensor view via `from_raw_parts` with manually specified
    /// shape and strides.
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

    /// Verifies that a transposed view reports `layout=non-contiguous` in
    /// the Debug header and renders elements in logical row order.
    #[test]
    fn test_debug_transposed_view() {
        // Source: shape=[2, 3] F-order, data=[1, 2, 3, 4, 5, 6]
        //   logical = [[1, 3, 5], [2, 4, 6]]
        // Transposed to shape=[3, 2]:
        //   logical = [[1, 2], [3, 4], [5, 6]]
        let tensor =
            unsafe { TensorBase::from_raw_vec_unchecked(vec![1_i32, 2, 3, 4, 5, 6], Ix2(2, 3)) };
        // Transposed: shape=[3,2], strides=[2,1]
        let view = unsafe { make_view(&tensor, Ix2(3, 2), Strides::new(Ix2(2, 1))) };
        let text = format!("{:?}", view);
        assert!(text.contains("layout=non-contiguous"), "text = {text:?}");
        assert!(text.contains("shape=[3, 2]"), "text = {text:?}");
        assert!(text.contains("[1, 2]"), "text = {text:?}");
        assert!(text.contains("[3, 4]"), "text = {text:?}");
        assert!(text.contains("[5, 6]"), "text = {text:?}");
    }

    /// Verifies that a broadcast view (zero stride on an axis) reports
    /// `layout=broadcast` in the Debug header.
    #[test]
    fn test_debug_broadcast_view() {
        let tensor = unsafe { TensorBase::from_raw_vec_unchecked(vec![1_i32, 2, 3], Ix2(1, 3)) };
        // Broadcast: shape=[4,3], strides=[0,1]
        let view = unsafe { make_view(&tensor, Ix2(4, 3), Strides::new(Ix2(0, 1))) };
        let text = format!("{:?}", view);
        assert!(text.contains("layout=broadcast"), "text = {text:?}");
    }

    /// Verifies that `element_type_of::<A>().name()` returns the correct
    /// human-readable dtype name for each supported element type.
    #[test]
    fn test_debug_dtype_complex() {
        fn check<A: Element>() -> &'static str {
            element_type_of::<A>().name()
        }
        assert_eq!(check::<i32>(), "i32");
        assert_eq!(check::<i64>(), "i64");
        assert_eq!(check::<f32>(), "f32");
        assert_eq!(check::<f64>(), "f64");
        assert_eq!(check::<bool>(), "bool");
    }

    #[test]
    fn test_display_tensor() {
        let tensor =
            unsafe { TensorBase::from_raw_vec_unchecked(vec![1, 2, 3], Ix1(3)) };
        assert_eq!(format!("{}", tensor), "[1, 2, 3]");
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
                Ix1(2),
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
                Ix1(2),
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
                Ix1(3),
            )
        };
        assert_eq!(format!("{}", tensor), "[NaN, inf, -inf]");
    }
}
