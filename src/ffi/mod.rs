//! FFI helper APIs.
//!
//! # Pointer / Offset Semantic Boundary
//!
//! The `offset` field appears across several FFI paths with subtly
//! different conventions; this section is the single source of truth:
//!
//! - `TensorBase::as_ptr()` returns the **logical first element**
//!   pointer. It has already absorbed `self.offset()` (i.e.
//!   `storage_base.add(self.offset())`).
//! - `TensorBase::as_storage_ptr()` returns the **storage base pointer**.
//! - `export().data` carries the storage base pointer, NOT the logical
//!   first. The logical first address is `data.add(offset)` for non-empty
//!   tensors.
//! - `try_offset_of(idx)` returns an offset relative to `as_ptr()`
//!   (logical first), NOT relative to the storage base. `try_ptr_at(idx)`
//!   therefore forwards via `as_ptr().add(offset)` and does NOT add
//!   `self.offset()` again.
//!
//! Invariant: `export().data + export().offset == as_ptr()`, and
//! `try_ptr_at(idx) == as_ptr() + try_offset_of(idx)`.
//!
//! # FFI Panic Boundary (Doc Example)
//!
//! Xenon does NOT define any `pub extern "C"` exported functions. The
//! example below shows how downstream consumers should wrap Xenon API
//! calls in their own `extern "C"` boundary using
//! `std::panic::catch_unwind` to prevent Rust panics from crossing the C
//! ABI.
//!
//! ```text
//! // Upstream library defines an extern "C" wrapper that calls into
//! // Xenon and catches any panic at the FFI boundary.
//! #[repr(C)]
//! pub enum XenonFfiStatus { Ok = 0, Error = 1, Panic = 2 }
//!
//! #[no_mangle]
//! pub extern "C" fn upstream_call_xenon() -> XenonFfiStatus {
//!     match std::panic::catch_unwind(|| {
//!         // Build or borrow Xenon tensors; use xenon::ffi::* APIs:
//!         //   tensor.export()         -> TensorExportRaw
//!         //   tensor.blas_info()      -> Result<BlasInfo<A>, XenonError>
//!         //   tensor.try_offset_of(.) -> Result<usize, XenonError>
//!         // Do not let C store the borrowed pointers after this fn returns.
//!     }) {
//!         Ok(()) => XenonFfiStatus::Ok,
//!         Err(_panic) => XenonFfiStatus::Panic,
//!     }
//! }
//! ```
//!
//! Without `catch_unwind`, a Rust panic crossing `extern "C"` is
//! undefined behavior. Wrappers must capture the panic and convert it to
//! an upstream ABI error code (or use `panic = "abort"`).

mod impls;
mod types;

// Public FFI re-exports, grouped here to give C/FFI consumers a single
// stable `crate::ffi::*` path: the tensor type and raw-parts metadata
// come from `crate::tensor`, the C-visible descriptors from `types`, and
// the FFI error enums from `crate::error`.
pub use crate::tensor::{OwnedRawParts, TensorBase};
pub use types::{BlasInfo, TensorExportMutRaw, TensorExportRaw};
pub use crate::error::{FfiBackend, FfiErrorCategory};
