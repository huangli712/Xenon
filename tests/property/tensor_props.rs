//! Tensor property tests: transpose involution (§6.4.2 parameterized loop).
//! W29T22 — tests/property/tensor_props.rs

use xenon::tensor::Tensor2;

/// §6.4.1 / §6.4.2: transpose().transpose() == original,
/// parameterized over 1..32 x 1..32.
#[test]
fn prop_transpose_involution() {
    for r in 1..32usize {
        for c in 1..32usize {
            let data: Vec<f64> = (0..r * c).map(|i| i as f64).collect();
            let t = Tensor2::from_shape_vec([r, c], data)
                .expect("shape must match data length");
            let tr = t.transpose();
            let tt = tr.transpose();
            assert_eq!(tt.shape(), t.shape(), "double transpose must restore shape");
            for (a, b) in t.iter().zip(tt.iter()) {
                assert_eq!(a, b, "double transpose must preserve element");
            }
        }
    }
}