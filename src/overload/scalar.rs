//! Operator overloading shared types.
//!
//! Contains the [`Scalar`] newtype wrapper used by both
//! [`super::owned`] and [`super::view`].

/// Newtype wrapper for scalar values, enabling a generic left-scalar path.
///
/// Rust orphan rules forbid blanket impls such as
/// `impl<T> Add<TensorBase<...>> for T`. Concrete primitive left-hand sides
/// like `impl Add<Tensor<f32, D>> for f32` remain legal and are provided for
/// Xenon's supported scalar set in `W23T5`–`W23T8` / `W23T10`.
///
/// Exported via `xenon::overload::Scalar` only — intentionally excluded from
/// the prelude and top-level re-exports.
#[expect(
    missing_debug_implementations,
    reason = "Scalar is a trivial newtype wrapper; Debug is not part of its public contract per 19-overload §5.3"
)]
pub struct Scalar<A>(pub A);
