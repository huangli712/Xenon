//! Operations property tests: add commutative, unique len bound,
//! unique no duplicate, sum additive identity.
//! W29T22 — tests/property/ops_props.rs

use xenon::tensor::Tensor1;

/// §5.22 / §6.4.1: a + b == b + a (exact equality on same execution path).
#[test]
fn prop_add_commutative() {
    for len in 1..256usize {
        let a = Tensor1::<f64>::from_shape_vec(
            [len],
            (0..len).map(|i| i as f64).collect(),
        )
        .expect("shape must match data length");
        let b = Tensor1::<f64>::from_shape_vec(
            [len],
            (0..len).map(|i| i as f64 + 1.0).collect(),
        )
        .expect("shape must match data length");
        let ab = (&a + &b).expect("add must succeed");
        let ba = (&b + &a).expect("add must succeed");
        for (x, y) in ab.iter().zip(ba.iter()) {
            assert_eq!(x, y, "add commutativity violated");
        }
    }
}

/// §6.4.1 line 1040: unique(a).len() <= a.len().
#[test]
fn prop_unique_len_bound() {
    for len in 0..64usize {
        let data: Vec<i32> = (0..len as i32).map(|i| i % 7).collect();
        let t = Tensor1::<i32>::from_shape_vec([len], data)
            .expect("shape must match data length");
        let u = t.unique();
        assert!(u.len() <= t.len(), "unique must not grow length");
    }
}

/// §5.22 / §6.4.1: unique result has no duplicate (non-NaN elements).
#[test]
fn prop_unique_no_duplicate() {
    for len in 1..64usize {
        let data: Vec<i32> = (0..len as i32).map(|i| i % 5).collect();
        let t = Tensor1::<i32>::from_shape_vec([len], data)
            .expect("shape must match data length");
        let u = t.unique();
        let mut vals: Vec<i32> = u.iter().copied().collect();
        vals.sort_unstable();
        let original_len = vals.len();
        vals.dedup();
        assert_eq!(
            vals.len(),
            original_len,
            "unique output must contain no duplicate"
        );
    }
}

/// §6.4.1 line 1037: empty array sum == additive identity (0) for f64.
#[test]
fn prop_sum_additive_identity() {
    let t = Tensor1::<f64>::zeros([0]).expect("empty shape is valid");
    let scalar: f64 = t.sum();
    assert_eq!(
        scalar, 0.0_f64,
        "sum of empty array must equal additive identity 0"
    );
}