//! Shape operations.
//!
//! The current public shape operation is full-axis transpose.
//!
//! Public re-exports are intentionally empty at this skeleton stage:
//! `transpose` is exposed as a method on `TensorBase` (W20T2 signature,
//! W20T3 body), not as a free function, so there is nothing to re-export
//! from this module. The 16-shape.md §7 T1 line 265 phrase "公共导出声明"
//! refers to this deliberate "no re-exports" choice — the public surface
//! is kept minimal per §1.1 职责边界.

mod transpose;

#[cfg(test)]
mod tests {
    // Verify the module compiles — no full transpose test at this stage.
    #[test]
    fn test_shape_module_compiles() {
        // This test trivially passes; its purpose is to confirm the crate
        // still builds after adding the shape module skeleton.
    }
}
