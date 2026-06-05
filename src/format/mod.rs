//! Tensor formatting support.
//!
//! Provides user-facing [`core::fmt::Display`] and developer-facing
//! [`core::fmt::Debug`] for tensors, with configurable truncation through
//! [`FormatConfig`](crate::format::FormatConfig) and Numpy-style logical-index ordering.
//!
//! ## In scope
//!
//! - Numpy-style 1D / ND output, nested brackets, matrix form.
//! - Configurable truncation (`edge_items`, `threshold`).
//! - Optional float precision through [`FormatConfig::precision`](crate::format::FormatConfig::precision).
//! - Distinct zero-dim marker `Tensor0(...)`.
//!
//! ## Out of scope (per `docs/design/22-output.md` §2)
//!
//! - Binary / JSON serialization.
//! - File I/O.
//! - HTML / rich-text rendering.
//! - Custom formatter registration.

mod config;
mod display;
mod pretty;
mod writer;

pub use config::FormatConfig;
pub use display::TensorDisplay;

#[cfg(test)]
mod tests {
    use super::{FormatConfig, TensorDisplay};
    use crate::dimension::Ix1;
    use crate::tensor::TensorBase;

    #[test]
    fn test_display_compile() {
        let tensor = unsafe { TensorBase::from_raw_vec_unchecked(vec![true, false], Ix1(2)) };
        let display = format!("{}", tensor);
        let debug = format!("{:?}", tensor);
        // §8.2 line 684 — bool element type renders via Display.
        assert!(display.contains("true"), "display = {display:?}");
        assert!(display.contains("false"), "display = {display:?}");
        // §5.4 line 287-301 — Debug header carries `shape=`.
        assert!(debug.contains("shape="), "debug = {debug:?}");
    }

    #[test]
    fn test_format_config_reexported() {
        // §8.7 line 729-731 — public re-export path is reachable.
        let _config: FormatConfig = FormatConfig::default();
    }

    #[test]
    fn test_tensor_display_reexported() {
        // §8.7 line 729-731 — TensorDisplay wrapper is reachable through
        // the re-export, and `display_with` returns it.
        let tensor = unsafe { TensorBase::from_raw_vec_unchecked(vec![1_i32, 2, 3], Ix1(3)) };
        let config = FormatConfig::default();
        let wrapper: TensorDisplay<'_, _, _, _> = tensor.display_with(config);
        let text = format!("{}", wrapper);
        assert_eq!(text, "[1, 2, 3]");
    }
}
