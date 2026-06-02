//! Tensor construction methods.
//!
//! Provides factory methods for creating tensors with common initialization
//! patterns. All constructors return `Result<Tensor, XenonError>` to enforce
//! validation at construction time.
//!
//! ## Core constructors
//!
//! | Method | Description | Returns |
//! |--------|-------------|---------|
//! | `zeros` | Zero-initialized tensor | `Result<Tensor<A, D>>` |
//! | `ones` | One-initialized tensor | `Result<Tensor<A, D>>` |
//! | `eye` | Identity matrix (2D only) | `Result<Tensor<A, Ix2>>` |
//! | `from_shape_vec` | From flat Vec with shape validation | `Result<Tensor<A, D>>` |
//! | `from_scalar` | Scalar repeated across shape | `Result<Tensor<A, D>>` |
//!
//! Construction errors include `InvalidShapeKind::ProductOverflow` for
//! overflowed element counts and `InvalidShapeKind::ElementCountMismatch`
//! for mismatched shape-Vec sizes.
//!
//! ## Implementation
//!
//! Constructors use `<Owned<A> as StorageOwned>::from_elem(len, value)` for
//! element-level initialization with canonical F-order strides.

pub use types::EyeElement;
/// Tensor constructors (`zeros`, `ones`, `eye`, `from_shape_vec`, `from_vec`,
/// `from_shape_slice`, `from_array`, `from_scalar`).
pub mod impls;
/// Construction trait definitions (`EyeElement`).
pub mod types;

#[cfg(test)]
mod tests {
    /// Compile-time anchor: each sub-module path must resolve. If any of the
    /// two `pub mod` declarations is removed or its target file is missing,
    /// this `use` block fails to compile.
    #[allow(unused_imports)]
    use super::{impls, types};

    /// Verify that all sub-module declarations resolve correctly.
    #[test]
    fn compile_anchor_construct_submodule_paths_resolve() {
        // No assertion needed — the `use super::{impls, types};`
        // statement above is itself the test. Constructor behavior is tested
        // by the individual sub-modules.
    }
}
