//! # Xenon — N-dimensional Tensor Library for Rust
//!
//! Xenon is a high-performance N-dimensional array (tensor) library for Rust,
//! designed as numerical infrastructure for scientific computing.
//!
//! ## Quick Start
//!
//! ```
//! # use xenon::prelude::*;
//! # use xenon::tensor::Tensor;
//!
//! # fn demo() -> xenon::Result<()> {
//! // Create two 2×3 f64 tensors filled with zeros and ones
//! let a = Tensor::<f64, _>::zeros([2, 3])?;
//! let b = Tensor::<f64, _>::ones([2, 3])?;
//!
//! // Element-wise addition (same shape)
//! let c = (&a + &b)?;
//! assert_eq!(c.shape(), &[2, 3]);
//!
//! // Reduction: sum of all elements
//! assert_eq!(c.sum(), 6.0);
//! # Ok(())
//! # }
//! ```
//!
//! ## Runtime Environment
//!
//! Xenon supports only the `std` environment.
//! It does not need or provide a `std` feature toggle.
//! All documentation assumes a `std` environment.
//!
//! ## Optional Features
//!
//! | Feature | Default | Description |
//! |---------|:-------:|-------------|
//! | `parallel` | ✗ | Data parallelism via rayon |
//! | `simd` | ✗ | SIMD acceleration via pulp |
//!
//! ## Supported Element Types
//!
//! | Level | Types | Trait Bound |
//! |-------|-------|-------------|
//! | Base | i32, i64, f32, f64, `Complex<f32>`, `Complex<f64>`, bool | `Element` |
//! | Numeric | i32, i64, f32, f64, `Complex<f32>`, `Complex<f64>` | `Numeric: Element` |
//! | Real | f32, f64 | `RealScalar: Numeric` |
//! | Complex | `Complex<f32>`, `Complex<f64>` | `ComplexScalar: Numeric` |
//!
//! `usize` is reserved for shape and index metadata, not as a tensor element type.
//!
//! ## Memory Layout
//!
//! Default layout is **F-order (column-major)**.
//! Xenon provides helper APIs that make upstream BLAS/LAPACK integration easier,
//! but not every legal layout is natively BLAS/LAPACK-compatible.
//!
//! ## License
//!
//! Xenon is distributed under the terms of the MIT license.
//! See [LICENSE](https://github.com/Xenon/LICENSE) for details.

#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
#![warn(rust_2024_compatibility)]
#![warn(unsafe_op_in_unsafe_fn)]
#![warn(clippy::disallowed_methods)]
#![warn(rustdoc::missing_crate_level_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(rustdoc::private_intra_doc_links)]
#![warn(clippy::missing_errors_doc)]
#![warn(clippy::missing_panics_doc)]
#![warn(clippy::missing_safety_doc)]

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

// ── Test-only re-exports ────────────────────────────────────────────
//
// Integration tests under `tests/` are external crates and cannot reach
// `pub(crate)` items inside `dispatch`. The items below are re-exported
// solely so those tests can observe dispatch decisions and tweak
// thresholds. They are NOT a stable public API: marked `#[doc(hidden)]`
// to keep them out of generated documentation.

#[doc(hidden)]
pub use crate::dispatch::{
    ExecPath, reset_simd_threshold, select_exec_path, set_simd_threshold,
};

#[cfg(feature = "parallel")]
#[doc(hidden)]
pub use crate::dispatch::{
    ParallelExecStrategy, ParallelGuard, reset_parallel_threshold, set_parallel_threshold,
};

/// Broadcasting: shape compatibility, stride expansion, zero-copy views.
pub mod broadcast;

/// Operator overloading for `Tensor` / `TensorView` arithmetic.
/// Delegates to inherent methods in [`crate::math`].
pub mod overload;

/// Aligned scratch workspace for internal temporary buffers.
pub mod workspace;

/// SIMD vectorized computation backend (opt-in via `simd` feature).
#[cfg(feature = "simd")]
pub(crate) mod simd;

/// Parallel computation backend (opt-in via `parallel` feature).
/// Module is always compiled; only rayon-dependent items are gated.
pub(crate) mod parallel;

// ── Test-only re-exports ────────────────────────────────────────────
//
// Integration tests under `tests/` are external crates and cannot reach
// `pub(crate)` items inside `parallel`. The items below are re-exported
// solely so those tests can exercise the parallel kernels directly.
// They are NOT a stable public API: marked `#[doc(hidden)]` to keep
// them out of generated documentation.

#[cfg(feature = "parallel")]
#[doc(hidden)]
pub use crate::parallel::map::par_map;

#[cfg(feature = "parallel")]
#[doc(hidden)]
pub use crate::parallel::reduce::{par_dot, par_sum};

/// Reduction operations: sum, sum_axis, sum_axis_keepdims.
pub mod reduction;

pub mod matrix;
pub use matrix::dot;

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
            "0.0.7",
            "Cargo.toml [package] version must be '0.0.7'"
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
