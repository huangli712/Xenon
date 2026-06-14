//! External integration tests for the layout module.
//!
//! Limited to the public layout API (`LayoutFlags`, `LayoutState`).

use xenon::layout::{LayoutFlags, LayoutState};

#[test]
fn external_flags_default_classify_non_contiguous() {
    // Default flags (all zero) ⇒ classify() returns NonContiguous
    // (no F_CONTIGUOUS bit, no HAS_ZERO_STRIDE bit).
    let flags = LayoutFlags::default();
    assert_eq!(flags.classify(), LayoutState::NonContiguous);
}
