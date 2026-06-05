//! Tensor formatting support.
//!
//! Provides user-facing [`core::fmt::Display`] and developer-facing
//! [`core::fmt::Debug`] for tensors, with configurable truncation through
//! [`FormatConfig`](crate::format::FormatConfig) and column-major
//! logical-index ordering.
//!
//! # Features
//!
//! - Numpy-style 1D / ND output with nested brackets and indented matrix form.
//! - Configurable truncation via
//!   [`FormatConfig::edge_items`](crate::format::FormatConfig::edge_items) and
//!   [`FormatConfig::threshold`](crate::format::FormatConfig::threshold).
//! - Optional float precision through
//!   [`FormatConfig::precision`](crate::format::FormatConfig::precision).
//! - Distinct zero-dimension marker `Tensor0(...)`.
//! - Soft line-wrapping controlled by
//!   [`FormatConfig::line_width`](crate::format::FormatConfig::line_width).
//!
//! # Architecture
//!
//! | Module | Role |
//! |--------|------|
//! | `config` | [`FormatConfig`](crate::format::FormatConfig) struct and defaults. |
//! | `display` | [`TensorDisplay`](crate::format::TensorDisplay) wrapper for custom-config Display. |
//! | `impls` | Adds [`core::fmt::Display`] and [`core::fmt::Debug`] implementations to [`TensorBase`](crate::tensor::TensorBase). |
//! | `pretty` | Core formatting pipelines: scalar, 1D, and recursive ND axis walkers. |
//! | `writer` | Column-tracking `LineWriter` used for soft-wrap decisions. |

mod config;
mod display;
mod impls;
mod pretty;
mod writer;

pub use config::FormatConfig;
pub use display::TensorDisplay;
