//! Private infrastructure for sealing public traits.
//!
//! Types in this module are not re-exported and are not part of the public API.

/// Sealed supertrait that prevents external implementations of
/// key library traits (e.g. `Dimension`, `Element`).
pub trait Sealed {}

use crate::dimension::{Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6};
use crate::dimension::IxDyn;

impl Sealed for Ix0 {}
impl Sealed for Ix1 {}
impl Sealed for Ix2 {}
impl Sealed for Ix3 {}
impl Sealed for Ix4 {}
impl Sealed for Ix5 {}
impl Sealed for Ix6 {}
impl Sealed for IxDyn {}
impl Sealed for f32 {}
impl Sealed for f64 {}
