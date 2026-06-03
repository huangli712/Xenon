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

#[cfg(any(test, feature = "parallel", feature = "simd"))]
mod support;

#[cfg(any(test, feature = "parallel", feature = "simd"))]
pub use self::support::ThresholdTestGuard;

mod threshold;
#[cfg(any(test, feature = "parallel"))]
pub use self::threshold::{set_parallel_threshold, reset_parallel_threshold};
#[cfg(any(test, feature = "simd"))]
pub use self::threshold::{set_simd_threshold, reset_simd_threshold};

pub(crate) use self::threshold::{DEFAULT_PARALLEL_THRESHOLD, DEFAULT_SIMD_THRESHOLD};
pub(crate) use self::threshold::{get_parallel_threshold, get_simd_threshold};

mod types;
pub use self::types::ExecPath;
#[cfg(feature = "parallel")]
pub use self::types::ParallelExecStrategy;
pub use self::types::ParallelGuard;

mod exec;
pub use self::exec::select_exec_path;
#[cfg(feature = "parallel")]
pub(crate) use self::exec::with_parallel_worker_context;
