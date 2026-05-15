//! Private infrastructure for sealing public traits.
//!
//! Types in this module are not re-exported and are not part of the public API.

/// Sealed supertrait that prevents external implementations of
/// key library traits (e.g. `Dimension`, `Element`).
///
/// Individual `impl Sealed for T` blocks are added by the respective
/// Wave's downstream tasks (W3T15 for `Dimension`).
pub trait Sealed {}
