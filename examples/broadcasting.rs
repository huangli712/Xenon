//! Example: Broadcasting operations
//!
//! Run with: `cargo run --example broadcasting`

use xenon::tensor::Tensor;

fn main() -> xenon::Result<()> {
    // Step 1: Create a 2×3 matrix and a row vector (1×3)
    let matrix = Tensor::<f64, _>::from_shape_vec([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let row = Tensor::<f64, _>::from_shape_vec([1, 3], vec![10.0, 20.0, 30.0])?;

    println!("Matrix (2×3):\n{}", matrix);
    println!("Row vector (1×3):\n{}", row);

    // Step 2: Broadcast row across matrix using explicit add
    let result = matrix.add(&row)?;
    println!("Matrix + broadcast row:\n{}", result);

    // Step 3: Transpose to demonstrate shape operations
    let transposed = result.transpose();
    println!("Transposed:\n{}", transposed);

    Ok(())
}
