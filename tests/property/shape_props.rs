//! Shape property tests: broadcast shape rule (§6.4.1 / §5.22).
//! W29T22 — tests/property/shape_props.rs

use xenon::tensor::Tensor2;

/// §6.4.1 / §5.22 / 15-broadcast §5:
/// Numpy broadcast — equal dims keep, 1 expands.
#[test]
fn prop_broadcast_shape_rule() {
    for r in 1..16usize {
        for c in 1..16usize {
            let a = Tensor2::<f64>::from_shape_vec(
                [r, 1],
                (0..r).map(|i| i as f64).collect(),
            )
            .expect("shape must match data length");
            let b = Tensor2::<f64>::from_shape_vec(
                [1, c],
                (0..c).map(|i| i as f64).collect(),
            )
            .expect("shape must match data length");
            let r_ab = (&a + &b).expect("broadcast add must succeed");
            assert_eq!(
                r_ab.shape(),
                &[r, c],
                "broadcast result shape must follow Numpy rule"
            );
        }
    }
}