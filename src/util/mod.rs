//! Utility operations: `clip`, `fill` / `try_fill`, `to_contiguous` /
//! `into_contiguous`. See `docs/design/20-utility.md`.
//!
//! All public entry points are exposed as inherent methods on `TensorBase`
//! in the submodules below; this module root only wires them into the crate.

mod clip;
mod contiguous;
mod fill;

#[cfg(test)]
mod tests {
    // Module-root existence test. `fill` / `clip` / `to_contiguous` /
    // `into_contiguous` are inherent methods on `TensorBase` defined in
    // the submodules below, NOT free functions, so we deliberately do NOT
    // `use super::*` nor reference them here — that would imply a
    // free-function API surface the design doc does not promise.
    //
    // Behavioral coverage lives in:
    //   - fill / try_fill  → `src/util/fill.rs`        (W24T2, §8.2)
    //   - clip             → `src/util/clip.rs`        (W24T3, §8.2)
    //   - to_contiguous /
    //     into_contiguous  → `src/util/contiguous.rs`  (W24T4, §8.2)
    //   - integration      → `tests/test_utility.rs`   (W24T5, §8.5)

    #[test]
    fn test_util_module_tree_compiles() {
        // Intentionally empty: reaching this point proves that `mod clip;`,
        // `mod contiguous;`, `mod fill;` all parsed and type-checked, which
        // is the only thing this module root can guarantee at its level.
    }
}
