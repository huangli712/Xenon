//! Xenon prelude module.
//!
//! Provides a convenient way to import the most commonly used types and traits
//! from the Xenon crate.
//!
//! # Usage
//!
//! ```
//! use xenon::prelude::*;
//! ```

// --- Core types (added incrementally as modules are implemented) ---
// Order strictly follows 01-architecture.md §7 prelude export list (lines 542-576).

// Tensor types — available after W8
// pub use crate::tensor::{TensorBase,
//                          Tensor, TensorView, TensorViewMut, ArcTensor};

// Dimension types — available after W3
// pub use crate::dimension::{Dimension, IntoDimension,
//                            Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6,
//                            IxDyn, Axis};

// Element traits — available after W4
// pub use crate::element::{Element, Numeric, RealScalar, ComplexScalar};

// Complex type — available after W5
// pub use crate::complex::Complex;

// Error types — available after W2
// pub use crate::error::{XenonError, Result};

// Construction convenience helpers — available after W22
// pub use crate::construct::{zeros, ones, eye, from_shape_vec};
