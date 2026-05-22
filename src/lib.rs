//! # Xenon — N-dimensional array library
//!
//! Xenon is a pure Rust N-dimensional array library for scientific computing.
//!
//! Public API examples are added by later documentation tasks after the
//! corresponding types and functions exist.

#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
#![warn(rust_2024_compatibility)]
#![warn(unsafe_op_in_unsafe_fn)]
#![warn(clippy::unwrap_used)]

/// Internal sealed-trait infrastructure.
mod private;

/// Structured error types for fallible operations.
pub mod error;

/// Static and dynamic dimension types with compile-time rank checking.
pub mod dimension;

/// Curated re-exports of the most commonly used types.
pub mod prelude;

/// Complex number type with a sealed real-component bound.
pub mod complex;

/// Factory functions: zeros, ones, eye, from_shape_vec, from_scalar.
pub mod construct;

/// Element type hierarchy: base traits and type discriminants.
pub mod element;

/// Memory layout: strides, contiguity, and layout flags.
pub mod layout;

/// Storage backends: owned, arc-shared, and view representations.
pub mod storage;

/// Element-level type conversion: casts, to_owned, into_owned.
pub mod convert;

/// Tensor core: TensorBase, type aliases, and raw-parts construction.
pub mod tensor;

/// Display and Debug formatting with configurable truncation.
pub mod format;

/// N-dimensional indexing and slicing traits.
pub mod index;

/// Shape operations: full-axis transpose.
pub mod shape;

/// Tensor iterators: element, axis, and indexed traversal.
pub mod iter;

/// Utility operations: clip, fill, to_contiguous, into_contiguous.
pub mod util;

/// Set operations: unique deduplication.
pub mod set;

/// FFI helper APIs: raw-pointer access and BLAS compatibility.
pub mod ffi;

/// Element-wise math operations: arithmetic, unary, comparison.
pub mod math;

/// Execution-path dispatch: Serial, Simd, and Parallel arbitration.
pub(crate) mod dispatch;

/// Broadcasting: shape compatibility, stride expansion, zero-copy views.
pub mod broadcast;

/// Aligned scratch workspace for internal temporary buffers.
pub mod workspace;

/// SIMD vectorized computation backend (opt-in via `simd` feature).
#[cfg(feature = "simd")]
pub(crate) mod simd;

/// Parallel computation backend (opt-in via `parallel` feature).
/// Module is always compiled; only rayon-dependent items are gated.
pub(crate) mod parallel;

pub use error::XenonError;
pub use prelude::*;

#[cfg(test)]
mod tests {
    //! Crate-level integrity verification tests.

    /// Verifies that Cargo.toml metadata matches the expected crate identity.
    /// `env!` reads these values from Cargo.toml at compile time.
    #[test]
    fn test_crate_metadata() {
        assert_eq!(
            env!("CARGO_PKG_NAME"),
            "xenon",
            "Cargo.toml [package] name must be 'xenon'"
        );
        assert_eq!(
            env!("CARGO_PKG_VERSION"),
            "0.0.2",
            "Cargo.toml [package] version must be '0.0.2'"
        );
    }

    /// Verifies that src/prelude.rs exists on disk.
    /// Created in Step 4 of this task and required by the `pub mod prelude;`
    /// declaration in Step 3. W1T4 will later overwrite this placeholder
    /// file with the full prelude module skeleton.
    #[test]
    fn test_prelude_module_file_exists() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/prelude.rs");
        assert!(
            std::path::Path::new(path).exists(),
            "src/prelude.rs must exist (created in Step 4)"
        );
    }
}
