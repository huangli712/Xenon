//! N-dimensional indexing and slicing.
//!
//! See `design/17-indexing.md` for the full design. This module declares the
//! three sub-modules and re-exports their public surface as each sub-module
//! is implemented. Each `pub use` line is enabled by the task indicated in
//! the adjacent comment.

pub mod access;
pub mod ndindex;
pub mod slice;

// Re-exports — enabled incrementally. Each line is activated by the task
// named in its trailing comment; until then it stays commented out to avoid
// referencing symbols that do not yet exist.
//
pub use ndindex::NdIndex; // W21T2
// pub use access::{/* inherent methods live on TensorBase */};    // W21T3/W21T5
pub use slice::{SliceInfo, SliceInfoElem, SliceInfoIndices}; // W21T4

#[cfg(test)]
mod tests {
    // Path-based smoke test: each `use` resolves only if the corresponding
    // sub-module file exists and parses. If any of the three files is
    // missing or malformed, the test target refuses to compile.
    #[allow(unused_imports)]
    use crate::index::access;
    #[allow(unused_imports)]
    use crate::index::ndindex;
    #[allow(unused_imports)]
    use crate::index::slice;

    #[test]
    fn test_index_module_skeleton_compiles() {
        // The real behavioral tests for `NdIndex`, `try_at`, `SliceInfo`
        // and `slice()` live in W21T2–W21T6 per `design/17-indexing.md §8.2`.
        // This smoke test only asserts that the module skeleton itself
        // compiles; the `use` lines above are the actual assertion.
    }
}
