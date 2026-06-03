//! Internal execution-path dispatch.
//!
//! Arbitrates between Serial, SIMD, and Parallel execution paths based on
//! input length, contiguity, alignment hints, runtime thresholds, and
//! feature gates. Holds the nested-parallel guard (TLS flag) so library
//! code never starts a parallel region inside another parallel region.
//!
//! All items here are `pub(crate)`. A minimal subset is re-exported
//! through `crate::prelude` so integration tests under `tests/` (which
//! are external crates) can observe dispatch decisions, tweak
//! thresholds, and exercise parallel kernels directly. Those
//! re-exports are NOT a stable public API.

mod threshold;
mod types;
mod exec;
mod support;

// --- threshold re-exports ---

#[cfg(any(test, feature = "parallel"))]
pub use threshold::{set_parallel_threshold, reset_parallel_threshold};
#[cfg(any(test, feature = "simd"))]
pub use threshold::{set_simd_threshold, reset_simd_threshold};
pub(crate) use threshold::{get_parallel_threshold, get_simd_threshold};

// --- types re-exports ---

pub use types::{ExecPath, ParallelGuard};
#[cfg(feature = "parallel")]
pub use types::ParallelExecStrategy;

// --- exec re-exports ---

pub use exec::select_exec_path;
#[cfg(feature = "parallel")]
pub(crate) use exec::with_parallel_worker_context;

// --- support re-exports ---

#[cfg(any(test, feature = "parallel", feature = "simd"))]
pub use support::ThresholdTestGuard;
