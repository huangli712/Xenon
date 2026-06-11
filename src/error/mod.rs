//! Xenon error types.
//!
//! All recoverable errors are represented by the [`XenonError`] enum.
//! The crate uses `Result<T, XenonError>` (aliased as [`Result`]) for
//! all fallible operations.

mod display;
mod enums;
mod root;

pub use enums::*;
pub use root::{Result, XenonError};
