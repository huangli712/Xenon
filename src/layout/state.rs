//! Layout state classification: the `LayoutState` enum.

/// Classification of tensor memory layout contiguity status.
///
/// Variants are mutually exclusive. `BroadcastView` applies only when
/// `product(shape) > 0 && any(stride == 0)`; empty tensors with degenerate
/// zero strides remain `FContiguous`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutState {
    /// Fortran-contiguous: first stride = 1, F-order progression.
    FContiguous,

    /// Arbitrary non-broadcast view that is not F-contiguous.
    NonContiguous,

    /// Non-empty view with at least one zero-stride axis (broadcast).
    BroadcastView,
}

impl LayoutState {
    /// Returns a human-readable label for the layout classification.
    pub fn as_str(self) -> &'static str {
        match self {
            LayoutState::FContiguous => "f-contiguous",
            LayoutState::BroadcastView => "broadcast",
            LayoutState::NonContiguous => "non-contiguous",
        }
    }
}
