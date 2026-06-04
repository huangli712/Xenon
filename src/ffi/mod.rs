//! FFI helper APIs.
//!
//! # Pointer / Offset Conventions
//!
//! - `as_ptr()` = logical first element pointer (storage base + offset).
//! - `as_storage_ptr()` = storage base pointer.
//! - `export().data` = storage base pointer.
//! - `try_offset_of(idx)` = offset relative to `as_ptr()` (logical first).
//!
//! Invariant: `export().data + export().offset == as_ptr()`, and
//! `try_ptr_at(idx) == as_ptr() + try_offset_of(idx)`.
//!
//! # Panic Safety
//!
//! Xenon defines no `pub extern "C"` functions. Downstream FFI wrappers
//! must use `std::panic::catch_unwind` at their `extern "C"` boundary —
//! a Rust panic crossing `extern "C"` is undefined behavior.

mod types;
mod impls;

// Public FFI re-exports, grouped here to give C/FFI consumers a single
// stable `crate::ffi::*` path: the tensor type and raw-parts metadata
// come from `crate::tensor`, the C-visible descriptors from `types`, and
// the FFI error enums from `crate::error`.
pub use types::{BlasInfo, TensorExportMutRaw, TensorExportRaw};
pub use crate::error::{FfiBackend, FfiErrorCategory};
pub use crate::tensor::{OwnedRawParts, TensorBase};
