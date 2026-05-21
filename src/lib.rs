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

/// Internal infrastructure for sealed traits.
mod private;

/// Structured error types for fallible operations.
pub mod error;

/// Dimension types for compile-time and runtime shape specification.
pub mod dimension;

/// Common exports for convenient use.
pub mod prelude;

/// Complex number type with sealed component bound.
pub mod complex;

/// Tensor construction: zeros, ones, eye, from_shape_vec, from_scalar, etc.
pub mod construct;

/// Element type hierarchy: base traits and type discriminants.
pub mod element;

/// Layout module: F-order strides, contiguity, flags and alignment.
pub mod layout;

/// Storage system: trait hierarchy and concrete storage representations.
pub mod storage;

/// Type conversion: element-level cast dispatch and `to_owned` / `into_owned`.
pub mod convert;

/// Tensor core: TensorBase, type aliases, query methods, and raw-parts
/// construction. See `07-tensor.md`.
pub mod tensor;

/// Tensor formatting support.
///
/// Provides user-facing `Display` and developer-facing `Debug` for tensors,
/// with configurable truncation and Numpy-style logical-index ordering.
pub mod format;

/// N-dimensional indexing and slicing.
pub mod index;

/// Shape operations.
///
/// The current public shape operation is full-axis transpose, exposed as a
/// method on `TensorBase`.
pub mod shape;
/// Tensor iterators.
///
/// Defines the public iterator surface: `Iter`, `IterMut`, `AxisIter`,
/// `AxisIterMut`, `IndexedIter`, `IndexedIterMut`.
pub mod iter;

/// Utility operations: clip, fill, to/into_contiguous.
/// See `docs/design/20-utility.md`.
pub mod util;


/// Execution-path dispatch: three-way arbitration (Serial / Simd / Parallel),
/// threshold management, and nested-parallel guard. See `30-dispatch.md`.
pub(crate) mod dispatch;

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
