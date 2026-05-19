//! External integration tests for the layout module.
//!
//! Limited to the public layout API (`compute_f_strides`,
//! `is_f_contiguous`, `Strides::*`, `is_aligned*`, `LayoutFlags`,
//! `LayoutState`). The crate-internal `compute_layout_flags` and
//! `flags_for_f_layout` are exercised by the in-module
//! `integration_tests` mod inside `src/layout/mod.rs`.
//!
//! Per `06-layout.md §8.5`, cross-module integration with tensor / shape /
//! index / ffi / simd is owned by W29 (those test files are created in
//! W29Tx) — not by W6T10.

use xenon::dimension::{Ix2, Ix3};
use xenon::layout::{compute_f_strides, is_f_contiguous, LayoutFlags, LayoutState, Strides};

#[test]
fn external_f_strides_symmetry() {
    // §5.6/§5.7 round-trip on common shape patterns.
    let cases: [&dyn Fn() -> bool; 4] = [
        &|| {
            let shape = Ix3(2, 3, 4);
            let s = compute_f_strides(&shape).expect("valid test shape");
            s.as_slice() == [1, 2, 6] && is_f_contiguous(&shape, &s)
        },
        &|| {
            let shape = Ix2(3, 1);
            let s = compute_f_strides(&shape).expect("valid test shape");
            is_f_contiguous(&shape, &s)
        },
        &|| {
            let shape = Ix2(1, 3);
            let s = compute_f_strides(&shape).expect("valid test shape");
            is_f_contiguous(&shape, &s)
        },
        &|| {
            let shape = Ix2(4, 5);
            let s = compute_f_strides(&shape).expect("valid test shape");
            s.as_slice() == [1, 4] && is_f_contiguous(&shape, &s)
        },
    ];
    for (idx, case) in cases.iter().enumerate() {
        assert!(case(), "case {idx} failed F-order symmetry");
    }
}

#[test]
fn external_c_order_is_not_f_contiguous() {
    // [2, 3] with C-order strides [3, 1] ⇒ NOT F-contiguous.
    let shape = Ix2(2, 3);
    let c_strides = Strides::new(Ix2(3, 1));
    assert!(!is_f_contiguous(&shape, &c_strides));
}

#[test]
fn external_flags_default_classify_non_contiguous() {
    // Default flags (all zero) ⇒ classify() returns NonContiguous
    // (no F_CONTIGUOUS bit, no HAS_ZERO_STRIDE bit).
    let flags = LayoutFlags::default();
    assert_eq!(flags.classify(), LayoutState::NonContiguous);
}