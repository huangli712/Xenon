//! Example: FFI integration with C / BLAS / LAPACK
//!
//! Run with: `cargo run --example ffi`
//!
//! Demonstrates raw pointer access and layout compatibility checks for
//! upstream C / BLAS / LAPACK integration.

use xenon::ffi;
use xenon::tensor::Tensor;

fn main() -> xenon::Result<()> {
    // Step 1: Create an F-order contiguous tensor
    let t = Tensor::<f64, _>::from_shape_vec([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;

    // Step 2: Access raw pointer and storage info for FFI
    println!(
        "Tensor: shape={:?}, strides={:?}, len={}",
        t.shape(),
        t.strides(),
        t.len()
    );

    // Step 3: Raw pointer access for FFI
    let ptr = t.as_ptr();
    let storage_ptr = t.as_storage_ptr();
    println!(
        "Logical first element: {:p}, storage base: {:p}",
        ptr, storage_ptr
    );

    // Step 4: Export metadata for C interop
    let exported = t.export();
    let shape_slice = unsafe { std::slice::from_raw_parts(exported.shape, exported.ndim) };
    let strides_slice = unsafe { std::slice::from_raw_parts(exported.strides, exported.ndim) };
    println!(
        "Exported: data={:p}, storage_len={}, ndim={}, shape={:?}, strides={:?}, offset={}",
        exported.data,
        exported.storage_len,
        exported.ndim,
        shape_slice,
        strides_slice,
        exported.offset,
    );

    Ok(())
}
