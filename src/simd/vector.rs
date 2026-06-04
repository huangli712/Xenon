//! Integration tests for the SIMD backend.

#[cfg(all(test, feature = "simd"))]
mod tests {
    use crate::simd::{BinaryOp, UnaryOp};

    // ---- threshold rejection ----

    #[test]
    fn test_vector_sub_mul_div_below_threshold_rejects() {
        let lhs: Vec<f32> = (0..32).map(|v| v as f32).collect();
        let rhs: Vec<f32> = (0..32).map(|v| v as f32).collect();
        let mut dst = vec![99.0f32; lhs.len()];

        // len=32 < threshold 64 — must reject
        assert!(!crate::simd::dispatch_vector_binary_op(
            BinaryOp::Add,
            &lhs,
            &rhs,
            &mut dst
        ));
        assert!(!crate::simd::dispatch_vector_binary_op(
            BinaryOp::Mul,
            &lhs,
            &rhs,
            &mut dst
        ));
        assert!(!crate::simd::dispatch_vector_unary_op(
            UnaryOp::Neg,
            &lhs,
            &mut dst
        ));
        // dst should remain unchanged on rejection
        for &v in &dst {
            assert_eq!(v, 99.0_f32, "dst must be untouched on SIMD rejection");
        }
    }

    // ---- integer stub ----

    const CASES: usize = 32;
    const MAX_LEN: usize = 4096;

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn gen_len(state: &mut u64, max_len: usize) -> usize {
        (splitmix64(state) as usize) % (max_len + 1)
    }

    fn gen_i32_no_overflow(state: &mut u64) -> i32 {
        ((splitmix64(state) % 2001) as i32) - 1000
    }

    fn prop_integer_no_panic_i32(seed: u64) {
        let mut rng = seed;
        for _case in 0..CASES {
            let len = gen_len(&mut rng, MAX_LEN);
            let data: Vec<i32> = (0..len).map(|_| gen_i32_no_overflow(&mut rng)).collect();
            // i32 SIMD is currently not available (per W14T0 spike),
            // so try_sum_i32 should always return None.
            assert!(
                crate::simd::try_sum_i32(&data).is_none(),
                "i32 SIMD sum should not be available (widening unavailable)"
            );
        }
    }

    #[test]
    fn prop_integer_no_panic() {
        prop_integer_no_panic_i32(0x4001);
    }
}
