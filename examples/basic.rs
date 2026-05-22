//! Example: Basic tensor operations
//!
//! Run with: `cargo run --example basic`
//!
//! Demonstrates tensor creation, element-wise addition, reduction, and display.

use xenon::tensor::Tensor;

fn main() -> xenon::Result<()> {
    // Step 1: Create tensors
    let a = Tensor::<f64, _>::zeros([2, 3])?;
    let b = Tensor::<f64, _>::ones([2, 3])?;
    println!("Created a 2×3 zero tensor and a 2×3 one tensor");

    // Step 2: Element-wise addition
    let c = (&a + &b)?;
    assert_eq!(c.shape(), &[2, 3]);
    println!("shape={:?}, sum={}", c.shape(), c.sum());

    // Step 3: Print result with default formatting
    assert_eq!(c.sum(), 6.0);
    println!("{}", c);

    Ok(())
}
