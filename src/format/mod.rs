//! Tensor pretty-printing with configurable truncation and soft-wrap.
//!
//! | Module | Role |
//! |--------|------|
//! | `config` | [`FormatConfig`][FormatConfig] — edge items, threshold, precision, line width. |
//! | `display` | [`TensorDisplay`][TensorDisplay] — custom-config `Display` adapter. |
//! | `impls` | `Display` + `Debug` trait impls on [`TensorBase`][crate::tensor::TensorBase]. |
//! | `pretty` | Core formatting pipeline — scalar, 1D, and recursive ND walkers. |
//! | `writer` | [`LineWriter`][crate::format::writer::LineWriter] — column tracker for soft-wrap. |

mod config;
mod display;
mod impls;
mod pretty;
mod writer;

pub use config::FormatConfig;
pub use display::TensorDisplay;
