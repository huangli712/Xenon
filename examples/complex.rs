//! Example: Complex number operations
//!
//! Run with: `cargo run --example complex`
//!
//! Demonstrates complex tensor construction, same-type arithmetic, and display.

use xenon::complex::Complex;
use xenon::prelude::*;
use xenon::tensor::Tensor;

fn main() -> xenon::Result<()> {
    // Step 1: Create complex-valued tensors
    let a = Tensor::<Complex<f64>, _>::from_shape_vec(
        [2],
        vec![Complex::new(1.0, 2.0), Complex::new(3.0, -4.0)],
    )?;
    let b = Tensor::<Complex<f64>, _>::ones([2])?;
    println!(
        "Created complex tensors: a = {}, b = {}",
        a.display_with(Default::default()),
        b.display_with(Default::default())
    );

    // Step 2: Element-wise addition of complex tensors
    let c = (&a + &b)?;
    println!("a + b = {}", c);

    // Step 3: Reduction — sum of all elements
    let total: Complex<f64> = c.sum();
    println!("sum = {}", total);

    Ok(())
}
