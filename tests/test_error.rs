//! Integration tests for XenonError thread safety.
//!
//! Verifies that `XenonError` satisfies `Send` across the public crate boundary,
//! as required by 26-error §9.3 and 25-safety §8.5.

use xenon::error::XenonError;

#[test]
fn test_parallel_error_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<XenonError>();
}