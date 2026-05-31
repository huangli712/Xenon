//! Shape operations.
//!
//! The current public shape operation is full-axis transpose.
//!
//! Public re-exports are intentionally empty: `transpose` is exposed as a
//! method on `TensorBase`, not as a free function, so there is nothing to
//! re-export from this module. The public surface is kept deliberately
//! minimal.

mod transpose;
