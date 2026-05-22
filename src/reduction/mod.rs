//! Reduction operations.
//!
//! The public API is exposed as methods on [`TensorBase`].
//!
//! [`TensorBase`]: crate::tensor::TensorBase

mod sum;

pub(crate) use sum::{sum_all, sum_axis_impl, sum_axis_keepdims_impl};

#[cfg(test)]
mod tests {
    // Smoke test: verify the reduction module skeleton and re-exports compile.
    // sum() methods are implemented in W18T2+, full feature tests in W18T6.

    #[test]
    fn test_reduction_module_compiles() {
        // T1 only ensures mod sum and pub(crate) use compile in mod.rs.
        // fn sum() is not yet implemented in this task, so no behaviour verified.
    }
}
