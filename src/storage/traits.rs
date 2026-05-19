//! Storage marker traits (W7T6).
//!
//! Sealed marker traits for the four concrete storage categories.
//! Concrete `unsafe impl`s are added by W7T11 (IsOwned), W7T14 (IsView),
//! W7T15 (IsViewMut), and W7T16 (IsShared).

use crate::storage::RawStorage;

/// Marker trait for owned storage representations.
///
/// # Safety
///
/// Implementors must satisfy the [`RawStorage`] contract and represent a
/// storage mode with exclusive ownership semantics controlled by this crate.
/// This trait is sealed and must not be implemented outside Xenon.
pub unsafe trait IsOwned: RawStorage + crate::private::Sealed {}

/// Marker trait for immutable borrowed view storage representations.
///
/// # Safety
///
/// Implementors must satisfy the [`RawStorage`] contract and represent a
/// read-only borrowed storage mode controlled by this crate. This trait is
/// sealed and must not be implemented outside Xenon.
pub unsafe trait IsView: RawStorage + crate::private::Sealed {}

/// Marker trait for mutable borrowed view storage representations.
///
/// # Safety
///
/// Implementors must satisfy the [`RawStorage`] contract and represent an
/// exclusive mutable borrowed storage mode controlled by this crate. This
/// trait is sealed and must not be implemented outside Xenon.
pub unsafe trait IsViewMut: RawStorage + crate::private::Sealed {}

/// Marker trait for shared read-only storage representations.
///
/// # Safety
///
/// Implementors must satisfy the [`RawStorage`] contract and represent a
/// shared read-only storage mode controlled by this crate. This trait is
/// sealed and must not be implemented outside Xenon.
pub unsafe trait IsShared: RawStorage + crate::private::Sealed {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that all four marker traits exist and require both the sealed
    /// bound and the `RawStorage` super-trait per `05-storage.md` §6.8.
    /// Concrete `unsafe impl`s are added in W7T11 (IsOwned), W7T14 (IsView),
    /// W7T15 (IsViewMut), and W7T16 (IsShared).
    #[test]
    fn test_marker_traits() {
        fn _sealed<T: crate::private::Sealed>() {}
        fn _bound_owned<T: IsOwned>() {}
        fn _bound_view<T: IsView>() {}
        fn _bound_view_mut<T: IsViewMut>() {}
        fn _bound_shared<T: IsShared>() {}
    }
}
