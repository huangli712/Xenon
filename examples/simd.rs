//! Example: SIMD-accelerated computation
//!
//! Run with: `cargo run --example simd --features simd`
//!
//! Demonstrates how the `simd` feature enables internal SIMD acceleration
//! while preserving the public API surface.

use xenon::tensor::Tensor;

fn main() -> xenon::Result<()> {
    // Create two large same-shaped tensors for element-wise operations.
    // When `simd` is enabled, internal kernels may use SIMD instructions
    // for contiguous data; the public add() API remains identical.
    let size = 1_000_000;
    let a = Tensor::<f64, _>::ones([size])?;
    let b = Tensor::<f64, _>::ones([size])?;

    let c = (&a + &b)?;
    let total = c.sum();

    assert_eq!(total, 2.0 * size as f64);
    println!(
        "SIMD example: {} elements, result sum = {} (expected {})",
        size,
        total,
        2.0 * size as f64
    );

    Ok(())
}
