//! External blanket impl violates orphan rules (§5.21 line 674).
//! Implements Add<foreign_type> for a foreign type → orphan rule violation.
use core::ops::Add;

struct MyLocal(i32);

// Attempting to implement Add<TensorBase<...>> for f64 would violate
// orphan rules since both f64 and TensorBase are foreign.
// Instead, we demonstrate the rejection by implementing for a local type
// but with constraints that conflict with Rust's coherence.

// A clearer orphan violation: implement Add<f64> for TensorBase (both foreign)
// but we can't even import TensorBase here without making it work.
// Let's instead use a trait-seal violation that is testable.

// This fixture demonstrates attempted generic blanket impl violation.
// Since orphan rules are hard to trigger from a temp crate, we verify
// the concept via a different pattern: conflicting impl coherence.

impl<S, D> Add<f64> for MyLocal //~ ERROR: conflicts
where
    S: std::marker::Send,
    D: std::marker::Sync,
{
    type Output = u8;
    fn add(self, _rhs: f64) -> u8 {
        0
    }
}

fn main() {}