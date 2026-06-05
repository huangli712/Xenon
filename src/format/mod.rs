//! Tensor pretty-printing with configurable truncation and soft-wrap.
//!
//! | Module | Role |
//! |--------|------|
//! | `config`  | `FormatConfig` — edge items, threshold, precision, line width.   |
//! | `display` | `TensorDisplay` — custom-config `Display` adapter.               |
//! | `impls`   | `Display` + `Debug` trait impls on `TensorBase`.                 |
//! | `pretty`  | Core formatting pipeline — scalar, 1D, and recursive ND walkers. |
//! | `writer`  | Column-tracking `LineWriter` for soft-wrap decisions.            |

mod config;
mod display;
mod impls;
mod pretty;
mod writer;

pub use config::FormatConfig;
pub use display::TensorDisplay;
