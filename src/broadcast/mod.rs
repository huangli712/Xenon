//! Broadcasting utilities.
//!
//! Public API list (see `15-broadcast.md §5.1`):
//! - `broadcast_shape(a, b) -> Result<IxDyn, XenonError>`
//! - `can_broadcast(a, b) -> bool`
//! - `broadcast_strides(orig_shape, orig_strides, target_shape) -> Result<Vec<usize>, XenonError>`
//! - `TensorBase::broadcast_to<E: IntoDimension>(&self, shape: E)`
//!   — inherent method on `TensorBase`, defined in `view.rs`; visible via `TensorBase`.

mod shape;
mod view;

pub use shape::{broadcast_shape, broadcast_strides, can_broadcast};
pub(crate) use view::broadcast_with;

#[cfg(test)]
mod tests {
    use crate::broadcast::{broadcast_shape, broadcast_strides, can_broadcast};
    use crate::dimension::IxDyn;
    use crate::error::XenonError;

    /// Compile-only check that the broadcast module's public exports are reachable
    /// from `crate::broadcast` with the exact signatures declared in
    /// `15-broadcast.md §5.1`. Does NOT call the functions (W11T2 stubs panic).
    #[test]
    fn test_broadcast_module_exports_compile() {
        type StridesFn = fn(&[usize], &[usize], &[usize]) -> Result<Vec<usize>, XenonError>;
        let _: fn(&[usize], &[usize]) -> bool = can_broadcast;
        let _: fn(&[usize], &[usize]) -> Result<IxDyn, XenonError> = broadcast_shape;
        let _: StridesFn = broadcast_strides;
    }
}
