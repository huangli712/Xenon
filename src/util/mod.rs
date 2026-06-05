//! Utility operations: `clip`, `fill` / `try_fill`, `to_contiguous` /
//! `into_contiguous`. See `docs/design/20-utility.md`.
//!
//! All public entry points are exposed as inherent methods on `TensorBase`
//! in the submodules below; this module root only wires them into the crate.

mod impls;
