//! Layout module: F-order strides, contiguity, flags and alignment.
//!
//! See `docs/design/06-layout.md`.

mod contiguous;
mod flags;
mod strides;

pub use contiguous::is_f_contiguous;
pub use flags::{LayoutFlags, LayoutState};
#[allow(unused_imports)]
pub(crate) use flags::{compute_layout_flags, flags_for_f_layout};
pub use strides::{Strides, compute_f_strides, has_zero_stride, is_aligned, is_aligned_to};

#[cfg(test)]
mod tests {
    #[test]
    #[allow(unused_imports)]
    fn test_layout_module_skeleton_compiles() {
        // Compile-time verification: submodule files must exist.
        use super::{contiguous as _, flags as _, strides as _};

        let module_path = module_path!();
        assert!(
            module_path.contains("layout"),
            "module_path! should reference layout module, got: {module_path}"
        );
        assert!(
            module_path.contains("tests"),
            "module_path! should reference tests submodule, got: {module_path}"
        );
    }
}
