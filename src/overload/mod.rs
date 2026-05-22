//! Operator overloading for `Tensor` / `TensorView` arithmetic.
//!
//! Public entry per `19-overload.md §5`. Actual `impl` blocks live in
//! `arithmetic` and are exposed through Rust's usual trait-impl visibility;
//! only `Scalar` needs to be named explicitly by user code.

pub mod arithmetic;

pub use arithmetic::Scalar;

#[cfg(test)]
mod tests {
    /// Compile-time anchor: the `arithmetic` sub-module path must resolve.
    /// If the `pub mod arithmetic;` declaration is removed or its target
    /// file is missing, this `use` block fails to compile, surfacing the
    /// breakage at the W23T1 acceptance gate rather than at a downstream
    /// sub-task. Behavioural coverage (operator semantics, `Scalar`
    /// re-export, broadcast paths) is provided by W23T2–W23T11.
    #[allow(unused_imports)]
    use super::arithmetic;

    #[test]
    fn compile_anchor_overload_submodule_path_resolves() {
        // No assertion needed — the `use super::arithmetic;` statement
        // above is itself the test.
    }

    #[test]
    fn test_scalar_reexport_visible_at_module_root() {
        // Compile-time surface check: `Scalar` must be re-exported at module
        // root so user code can reference `xenon::overload::Scalar`
        // (per 19-overload §5.3 line 268-272).
        let _opt: Option<super::Scalar<i32>> = None;
    }
}
