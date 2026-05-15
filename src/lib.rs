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

/// Common exports for convenient use.
pub mod prelude;

#[expect(unused_imports)]
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
            "0.0.1",
            "Cargo.toml [package] version must be '0.0.1'"
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
