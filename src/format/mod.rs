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
mod impls;
mod pretty;
mod writer;

pub use config::FormatConfig;
pub use display::TensorDisplay;

