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

// Construction module skeleton (W22T1).
//
// Implementations are added by sub-tasks in their respective files:
//   W22T2 → impl<A: Element, D: Dimension> TensorBase<Owned<A>, D> { fn zeros }  in impls.rs
//   W22T3 → impl<A: Element, D: Dimension> TensorBase<Owned<A>, D> { fn ones }   in impls.rs
//   W22T4 → pub trait EyeElement + impl<A: EyeElement> TensorBase<Owned<A>, Ix2> { fn eye }
//   W22T4 trait definition + EyeElement impls are in types.rs.
//   W22T4 additionally adds `pub use types::EyeElement;` here.
//   W22T5 → impl<A: Element, D: Dimension> TensorBase<Owned<A>, D> { fn from_shape_vec }
//         + impl<A: Element> TensorBase<Owned<A>, Ix1> { fn from_vec }  in impls.rs
//   W22T6 → impl<A: Element + Clone, D: Dimension> TensorBase<Owned<A>, D> { fn from_shape_slice }
//           in impls.rs
//   W22T7 → impl<A: Element, D: Dimension> TensorBase<Owned<A>, D> { fn from_array<const N: usize> }
//           in impls.rs
//   W22T8 → impl<A: Element> TensorBase<Owned<A>, Ix0> { fn from_scalar }  in impls.rs
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
    /// this `use` block fails to compile, surfacing the breakage at the
    /// W22T1 acceptance gate rather than at a downstream sub-task.
    #[allow(unused_imports)]
    use super::{impls, types};

    #[test]
    fn compile_anchor_construct_submodule_paths_resolve() {
        // No assertion needed — the `use super::{impls, types};`
        // statement above is itself the test. The empty body documents that
        // constructor behavior is tested by W22T2 (zeros), W22T3 (ones),
        // W22T4 (eye + EyeElement), W22T5 (from_shape_vec + from_vec),
        // W22T6 (from_shape_slice), W22T7 (from_array), W22T8 (from_scalar),
        // and W22T9 (cross-API integration tests).
    }
}
