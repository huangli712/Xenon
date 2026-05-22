//! Example: Optional feature behavior
//!
//! Run with: `cargo run --example features --features parallel,simd`
//!
//! Demonstrates how `parallel` and `simd` features affect public API behavior.
//! The public API remains identical; features influence internal execution paths.

use xenon::tensor::Tensor;

fn main() -> xenon::Result<()> {
    // Create a large 1D tensor
    let size = 1_000_000;
    let t = Tensor::<f64, _>::ones([size])?;

    // Reduction — public API is always `sum()`, but internal dispatch
    // selects serial, SIMD, or parallel paths based on enabled features.
    #[cfg(feature = "parallel")]
    println!("parallel feature enabled");

    #[cfg(feature = "simd")]
    println!("simd feature enabled");

    #[cfg(not(any(feature = "parallel", feature = "simd")))]
    println!("default (serial) path");

    let total = t.sum();
    assert_eq!(total, size as f64);
    println!("sum of {} elements = {}", size, total);

    Ok(())
}
