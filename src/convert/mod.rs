//! Type conversion module root.
//!
//! Submodules:
//! - `cast`: `ConvertTo` sealed shim, tier impls, and `cast()` / `to_owned()`
//!   / `into_owned()` methods on `TensorBase`.

mod cast;
pub use cast::{CastElement, CastTo};

#[cfg(test)]
mod tests {
    #[test]
    fn test_convert_module_skeleton_compiles() {
        // Mere compilation of this test proves the convert/ module skeleton
        // is wired up correctly:
        //   - src/lib.rs declares `pub mod convert;`
        //   - src/convert/mod.rs declares `mod cast;`
        //   - src/convert/cast.rs exists as a resolvable file
        // ConvertTo trait behavior is verified in W25T2-W25T7 tests, after
        // the trait and its impls are introduced.
        let _module_path = module_path!();
        assert!(_module_path.contains("convert"));
    }
}
