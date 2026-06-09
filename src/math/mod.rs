//! Element-wise math operations (arithmetic, unary, comparison).
//!
//! Public API is exposed as inherent methods on `TensorBase` from
//! the `binary`, `unary`, and `comparison` submodules. No `pub use`
//! re-exports are needed: method visibility is governed by the
//! `impl<...> TensorBase<...>` blocks themselves.
//!

mod binary;
mod comparison;
mod unary;

pub(crate) use binary::BinaryArith;

#[cfg(test)]
mod tests {
    // Module skeleton verification: empty submodules + crate-root
    // `pub mod math;` declaration are validated by `cargo check`.
    //
    // Functional tests for math operations live in W16T2–W16T11
    // (each task adds its own `#[cfg(test)] mod tests` block in the
    // submodule it edits).
}
