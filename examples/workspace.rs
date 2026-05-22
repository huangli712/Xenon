//! Example: Workspace allocation and borrow/split workflow
//!
//! Run with: `cargo run --example workspace`
//!
//! Demonstrates scratch workspace creation, mutable borrowing, splitting,
//! and growth semantics.

use xenon::workspace::Workspace;

fn main() -> xenon::Result<()> {
    // Step 1: Create a workspace with 64-byte alignment
    let mut ws = Workspace::new(1024, 64)?;
    println!("Workspace created: capacity={}", ws.capacity());

    // Step 2: Borrow a mutable buffer from the workspace
    {
        let mut buf = ws.borrow_mut()?;
        let scratch = buf.as_maybe_uninit_slice();
        println!("Borrowed mutable buffer: len={}", scratch.len());
    } // buf dropped here — RAII return

    // Step 3: Grow the workspace and verify
    ws.ensure_capacity(2048)?;
    println!("After ensure_capacity: capacity={}", ws.capacity());

    // Step 4: Split the workspace for independent sub-buffers
    let (left, right) = ws.split_at_mut(512)?;
    println!(
        "Split workspace: left={}, right={}",
        left.len(),
        right.len()
    );

    Ok(())
}
