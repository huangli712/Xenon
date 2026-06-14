//! Integration tests for Workspace allocation, borrowing, splitting, and error paths.
//!
//! Covers the public Workspace API surface per 24-workspace.md.

use xenon::error::{
    WorkspaceErrorCategory, XenonError,
};
use xenon::workspace::Workspace;

#[test]
fn test_workspace_new_invalid_alignment() {
    // Alignment that is not a power of two → InvalidLayout.
    let err = Workspace::new(1024, 7).expect_err("non-power-of-two alignment");
    match &err {
        XenonError::Workspace {
            category: WorkspaceErrorCategory::InvalidLayout { size, align },
            ..
        } => {
            assert_eq!(*size, 1024);
            assert_eq!(*align, 7);
        },
        other => panic!("expected InvalidLayout, got {other:?}"),
    }
    let s = format!("{err}");
    assert!(s.contains("invalid layout"));

    // Alignment below MIN_ALIGNMENT (8) → InvalidLayout.
    let err = Workspace::new(1024, 4).expect_err("below minimum alignment");
    match &err {
        XenonError::Workspace {
            category: WorkspaceErrorCategory::InvalidLayout { size, align },
            ..
        } => {
            assert_eq!(*size, 1024);
            assert_eq!(*align, 4);
        },
        other => panic!("expected InvalidLayout, got {other:?}"),
    }
}

#[test]
fn test_workspace_ensure_capacity() {
    // No-grow case: capacity already sufficient.
    let mut ws = Workspace::new(64, 64).expect("64-byte workspace");
    ws.ensure_capacity(32).expect("no-grow ensure_capacity");
    assert_eq!(ws.capacity(), 64);

    // Grow case: request more than current capacity.
    let mut ws = Workspace::new(64, 64).expect("64-byte workspace");
    ws.ensure_capacity(128).expect("grow ensure_capacity");
    assert!(ws.capacity() >= 128);
    // Alignment must be preserved.
    assert_eq!(ws.alignment(), 64);

    // Ensure capacity with a value larger than current triggers growth.
    let mut ws = Workspace::new(64, 64).expect("64-byte workspace");
    ws.ensure_capacity(200).expect("larger ensure_capacity");
    assert!(ws.capacity() >= 200);
}

#[test]
fn test_workspace_borrow_rules() {
    // Successfully borrow immutably.
    let ws = Workspace::new(64, 64).expect("64-byte workspace");
    let guard = ws.borrow().expect("immutable borrow");
    assert_eq!(guard.len(), 64);
    drop(guard);

    // Successfully borrow mutably.
    let mut ws = Workspace::new(64, 64).expect("64-byte workspace");
    let guard = ws.borrow_mut().expect("mutable borrow");
    assert_eq!(guard.len(), 64);
    drop(guard);

    // Second immutable borrow while one is active must fail (only one guard
    // allowed at a time).
    let ws = Workspace::new(64, 64).expect("64-byte workspace");
    let _g1 = ws.borrow().expect("first borrow");
    let err = ws.borrow().expect_err("second borrow should conflict");
    match &err {
        XenonError::Workspace {
            category: WorkspaceErrorCategory::BorrowConflict { requested, current: _ },
            ..
        } => {
            assert!(format!("{requested:?}").contains("Shared"));
        },
        other => panic!("expected BorrowConflict, got {other:?}"),
    }
    drop(_g1);

    // After dropping the first guard, re-borrow succeeds.
    let _g2 = ws.borrow().expect("re-borrow after drop");
}

#[test]
fn test_workspace_split() {
    // Basic binary split.
    let mut ws = Workspace::new(100, 64).expect("100-byte workspace");
    let (left, right) = ws.split_at_mut(40).expect("split");
    assert_eq!(left.len(), 40);
    assert_eq!(right.len(), 60);
    drop(left);
    drop(right);

    // Split out of bounds must return structured error and must not leave
    // the workspace in a borrowed state.
    let mut ws = Workspace::new(8, 64).expect("8-byte workspace");
    let err = ws.split_at_mut(9).expect_err("split OOB");
    match &err {
        XenonError::Workspace {
            category: WorkspaceErrorCategory::SplitOutOfBounds { mid, len },
            ..
        } => {
            assert_eq!(*mid, 9);
            assert_eq!(*len, 8);
        },
        other => panic!("expected SplitOutOfBounds, got {other:?}"),
    }
    // Workspace must be re-borrowable after failed split.
    assert!(ws.borrow().is_ok());

    // Recursive split.
    let mut ws = Workspace::new(100, 64).expect("100-byte workspace");
    let (left, right) = ws.split_at_mut(40).expect("split");
    let (right_a, right_b) = right.split_at_mut(30).expect("recursive split");
    assert_eq!(left.len(), 40);
    assert_eq!(right_a.len(), 30);
    assert_eq!(right_b.len(), 30);
    drop(left);
    drop(right_a);
    drop(right_b);

    // Workspace must be re-borrowable after all splits dropped.
    assert!(ws.borrow().is_ok());
}

#[test]
fn test_workspace_assume_init_prefix() {
    let mut ws = Workspace::new(64, 64).expect("64-byte workspace");
    let mut guard = ws.borrow_mut().expect("mutable borrow");

    // Write some known bytes via the MaybeUninit view.
    {
        let view = guard.as_maybe_uninit_slice();
        for (i, slot) in view.iter_mut().enumerate() {
            slot.write((i % 256) as u8);
        }
    }

    // assume_init_slice with valid length should succeed.
    let initialized =
        unsafe { guard.assume_init_slice(10).expect("assume_init prefix") };
    assert_eq!(initialized.len(), 10);
    assert_eq!(initialized[0], 0);
    assert_eq!(initialized[9], 9);

    // assume_init_slice with length exceeding borrow length must fail.
    let err = unsafe { guard.assume_init_slice(128) };
    assert!(err.is_err());

    // assume_init_slice with length == 0 must succeed.
    let empty = unsafe { guard.assume_init_slice(0).expect("empty prefix") };
    assert!(empty.is_empty());
}

#[test]
fn test_workspace_error_boundary_mapping() {
    // Verify that Workspace::new returns InvalidLayout for alignment=0.
    let err = Workspace::new(64, 0).expect_err("alignment 0 is invalid");
    match &err {
        XenonError::Workspace {
            category: WorkspaceErrorCategory::InvalidLayout { size, align },
            ..
        } => {
            assert_eq!(*size, 64);
            assert_eq!(*align, 0);
        },
        other => panic!("expected InvalidLayout, got {other:?}"),
    }

    // Verify that a borrow conflict error is properly structured.
    let ws = Workspace::new(64, 64).expect("64-byte workspace");
    let _guard = ws.borrow().expect("first borrow");
    let err = ws.borrow().expect_err("second borrow conflict");
    let msg = format!("{err}");
    assert!(msg.contains("borrow conflict"));
    assert!(msg.contains("Shared"));

    // Verify that the workspace error Display does not panic for any category.
    let categories = [
        WorkspaceErrorCategory::AllocFailed { size: 0, align: 1 },
        WorkspaceErrorCategory::InvalidLayout { size: 0, align: 3 },
        WorkspaceErrorCategory::BorrowConflict {
            requested: xenon::error::WorkspaceBorrowKind::Shared,
            current: xenon::error::WorkspaceBorrowState::None,
        },
        WorkspaceErrorCategory::SplitOutOfBounds { mid: 0, len: 0 },
        WorkspaceErrorCategory::GrowOverflow {
            current_capacity: usize::MAX,
            additional: 1,
        },
        WorkspaceErrorCategory::TypedViewRejected {
            detail: xenon::error::TypedViewRejection::ZeroSizedType,
        },
    ];
    for cat in &categories {
        let s = format!("{cat}");
        assert!(!s.is_empty(), "Display for {cat:?} must not be empty");
    }
}

#[test]
fn test_workspace_not_send_not_sync() {
    // Workspace must be !Send + !Sync because it contains PhantomData<*mut ()>.
    // Verified at compile time via the ambiguity trick (the mechanism
    // assert_not_impl_any! expands to): the Send-/Sync-gated impls apply only
    // if Workspace implements that trait, so for a !Send + !Sync type only the
    // blanket `()` impl matches and `_` resolves. A future Send or Sync impl
    // would make resolution ambiguous and fail the build.
    trait AmbiguousIfImpl<A> {
        fn marker() {}
    }
    impl<T> AmbiguousIfImpl<()> for T {}
    impl<T: Send> AmbiguousIfImpl<u8> for T {}
    impl<T: Sync> AmbiguousIfImpl<u16> for T {}

    let _ = <Workspace as AmbiguousIfImpl<_>>::marker;
}
