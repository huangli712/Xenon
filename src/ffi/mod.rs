//! FFI helper APIs.
//!
//! See `docs/design/23-ffi.md` for the public API surface, lifetime
//! contracts, BLAS compatibility rules, and Safety considerations.
//!
//! # Pointer / Offset Semantic Boundary
//!
//! The `offset` field appears across several FFI paths with subtly
//! different conventions; this section is the single source of truth:
//!
//! - `TensorBase::as_ptr()` (W8T6, defined in `07-tensor.md` §5.2)
//!   returns the **logical first element** pointer. It has already
//!   absorbed `self.offset()` (i.e. `storage_base.add(self.offset())`).
//! - `TensorBase::as_storage_ptr()` returns the **storage base pointer**.
//! - `export().data` (W13T4 / `§5.4 line 542-543`) carries the storage
//!   base pointer, NOT the logical first. The logical first address is
//!   `data.add(offset)` for non-empty tensors. This is the historical
//!   regression point.
//! - `try_offset_of(idx)` (W13T6 / `§5.13 line 1085`) returns an offset
//!   relative to `as_ptr()` (logical first), NOT relative to the storage
//!   base. `try_ptr_at(idx)` therefore forwards via `as_ptr().add(offset)`
//!   and does NOT add `self.offset()` again.
//!
//! Invariant: `export().data + export().offset == as_ptr()`, and
//! `try_ptr_at(idx) == as_ptr() + try_offset_of(idx)`. Any path diverging
//! from these invariants is a regression and must be caught by
//! W13T4/W13T6 unit tests.
//!
//! # FFI Panic Boundary (Doc Example)
//!
//! Xenon does NOT define any `pub extern "C"` exported functions. The
//! example below shows how downstream consumers should wrap Xenon API
//! calls in their own `extern "C"` boundary using
//! `std::panic::catch_unwind` to prevent Rust panics from crossing the C
//! ABI (`23-ffi.md` §5.4 line 642-675, §8.2 line 1364).
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

// Public re-exports are intentionally absent at W13T1.
// Each downstream task (W13T2 / W13T4) adds the `pub use` lines for the
// items it introduces, following the module-declaration-evolution
// protocol defined by W1T3.
pub use crate::tensor::{OwnedRawParts, TensorBase};
pub use types::{BlasInfo, TensorExportMutRaw, TensorExportRaw};
pub use crate::error::{FfiBackend, FfiErrorCategory};
