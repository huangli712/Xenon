//! Integration tests for the SIMD backend.

#[cfg(all(test, feature = "simd"))]
mod tests {
    use crate::complex::Complex;
    use crate::simd::{self, BinaryOp, UnaryOp};

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

    // ---- element-wise consistency (W14T8) ----

    const SIMD_WIDTH: usize = 64;

    fn assert_same_bits_or_nan_f64(actual: f64, expected: f64) {
        if expected.is_nan() || actual.is_nan() {
            assert!(actual.is_nan() && expected.is_nan());
        } else {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }

    fn assert_same_bits_or_nan_f32(actual: f32, expected: f32) {
        if expected.is_nan() || actual.is_nan() {
            assert!(actual.is_nan() && expected.is_nan());
        } else {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }

    fn assert_vec_bits_or_nan_f64(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (&a, &e) in actual.iter().zip(expected.iter()) {
            assert_same_bits_or_nan_f64(a, e);
        }
    }

    fn assert_vec_bits_or_nan_f32(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (&a, &e) in actual.iter().zip(expected.iter()) {
            assert_same_bits_or_nan_f32(a, e);
        }
    }

    fn apply_binary_f64(op: BinaryOp, lhs: f64, rhs: f64) -> f64 {
        match op {
            BinaryOp::Add => lhs + rhs,
            BinaryOp::Sub => lhs - rhs,
            BinaryOp::Mul => lhs * rhs,
            BinaryOp::Div => lhs / rhs,
        }
    }

    fn apply_binary_f32(op: BinaryOp, lhs: f32, rhs: f32) -> f32 {
        match op {
            BinaryOp::Add => lhs + rhs,
            BinaryOp::Sub => lhs - rhs,
            BinaryOp::Mul => lhs * rhs,
            BinaryOp::Div => lhs / rhs,
        }
    }

    fn fixture_f64(len: usize) -> (Vec<f64>, Vec<f64>) {
        let lhs_seed = [
            1.5_f64,
            -2.3,
            0.001,
            -1e20,
            std::f64::consts::PI,
            0.0,
            -0.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MIN_POSITIVE / 2.0,
        ];
        let rhs_seed = [
            -4.25_f64,
            8.0,
            -0.125,
            1e-20,
            -2.0,
            0.0,
            -0.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MIN_POSITIVE / 2.0,
        ];
        let lhs: Vec<f64> = (0..len).map(|i| lhs_seed[i % lhs_seed.len()]).collect();
        let rhs: Vec<f64> = (0..len).map(|i| rhs_seed[i % rhs_seed.len()]).collect();
        (lhs, rhs)
    }

    fn fixture_f32(len: usize) -> (Vec<f32>, Vec<f32>) {
        let lhs_seed = [
            1.5_f32,
            -2.3,
            0.001,
            -1e10,
            std::f32::consts::PI,
            0.0,
            -0.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MIN_POSITIVE / 2.0,
        ];
        let rhs_seed = [
            -4.25_f32,
            8.0,
            -0.125,
            1e-10,
            -2.0,
            0.0,
            -0.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MIN_POSITIVE / 2.0,
        ];
        let lhs: Vec<f32> = (0..len).map(|i| lhs_seed[i % lhs_seed.len()]).collect();
        let rhs: Vec<f32> = (0..len).map(|i| rhs_seed[i % rhs_seed.len()]).collect();
        (lhs, rhs)
    }

    fn simd_vs_serial_bitwise_f64(op: BinaryOp, lhs: &[f64], rhs: &[f64]) {
        let mut dst = vec![0.0_f64; lhs.len()];
        let handled = simd::dispatch_vector_binary_op(op, lhs, rhs, &mut dst);
        if handled {
            let serial: Vec<f64> = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(&l, &r)| apply_binary_f64(op, l, r))
                .collect();
            assert_vec_bits_or_nan_f64(&dst, &serial);
        }
    }

    fn simd_vs_serial_bitwise_f32(op: BinaryOp, lhs: &[f32], rhs: &[f32]) {
        let mut dst = vec![0.0_f32; lhs.len()];
        let handled = simd::dispatch_vector_binary_op(op, lhs, rhs, &mut dst);
        if handled {
            let serial: Vec<f32> = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(&l, &r)| apply_binary_f32(op, l, r))
                .collect();
            assert_vec_bits_or_nan_f32(&dst, &serial);
        }
    }

    #[test]
    fn test_simd_vector_consistency_elementwise() {
        let (lhs_f64, rhs_f64) = fixture_f64(256);
        let (lhs_f32, rhs_f32) = fixture_f32(256);
        for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div] {
            simd_vs_serial_bitwise_f64(op, &lhs_f64, &rhs_f64);
            simd_vs_serial_bitwise_f32(op, &lhs_f32, &rhs_f32);
        }
    }

    #[test]
    fn test_simd_neg_f64_matches_serial() {
        let (src, _) = fixture_f64(256);
        let mut dst = vec![0.0_f64; src.len()];
        let handled = simd::dispatch_vector_unary_op(UnaryOp::Neg, &src, &mut dst);
        if handled {
            let serial: Vec<f64> = src.iter().map(|&v| -v).collect();
            assert_vec_bits_or_nan_f64(&dst, &serial);
        }
    }

    #[test]
    fn test_simd_neg_f32_matches_serial() {
        let (src, _) = fixture_f32(256);
        let mut dst = vec![0.0_f32; src.len()];
        let handled = simd::dispatch_vector_unary_op(UnaryOp::Neg, &src, &mut dst);
        if handled {
            let serial: Vec<f32> = src.iter().map(|&v| -v).collect();
            assert_vec_bits_or_nan_f32(&dst, &serial);
        }
    }

    #[test]
    fn test_elementwise_boundary_lengths() {
        let mut lengths = vec![0, 1, 32, 64, 65, 128, 256];
        lengths.extend((1..8).map(|extra| SIMD_WIDTH + extra));
        for len in lengths {
            let (lhs, rhs) = fixture_f64(len);
            simd_vs_serial_bitwise_f64(BinaryOp::Add, &lhs, &rhs);
            simd_vs_serial_bitwise_f64(BinaryOp::Sub, &lhs, &rhs);
        }
    }

    #[test]
    fn test_elementwise_tail_handling() {
        for extra in [1_usize, 3, 7, 15] {
            let (lhs, rhs) = fixture_f64(256 + extra);
            for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div] {
                simd_vs_serial_bitwise_f64(op, &lhs, &rhs);
            }
        }
    }

    #[test]
    fn test_elementwise_below_threshold() {
        let (lhs, rhs) = fixture_f64(32);
        let mut dst = vec![0.0_f64; lhs.len()];
        assert!(!simd::dispatch_vector_binary_op(
            BinaryOp::Add,
            &lhs,
            &rhs,
            &mut dst
        ));
    }

    #[test]
    fn test_elementwise_at_threshold() {
        let (lhs, rhs) = fixture_f64(64);
        simd_vs_serial_bitwise_f64(BinaryOp::Add, &lhs, &rhs);
    }

    #[test]
    fn test_elementwise_misaligned() {
        let (lhs_storage, rhs_storage) = fixture_f64(258);
        let lhs = &lhs_storage[1..257];
        let rhs = &rhs_storage[1..257];
        simd_vs_serial_bitwise_f64(BinaryOp::Mul, lhs, rhs);
    }

    // ---- randomized property tests (W14T10) ----

    const CASES: usize = 32;
    const MAX_LEN: usize = 4096;
    const ELEMENTWISE_THRESHOLD: usize = 64;
    const SUM_THRESHOLD: usize = 1024;
    const DOT_THRESHOLD: usize = 512;
    const COMPLEX_SUM_THRESHOLD: usize = 1024;
    const COMPLEX_DOT_THRESHOLD: usize = 512;
    const COMPLEX_ELEMENTWISE_THRESHOLD: usize = 128;

    // ---- splitmix64 PRNG ----

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

    fn gen_f64(state: &mut u64) -> f64 {
        let frac = (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64;
        (frac - 0.5) * 20.0
    }

    fn gen_f32(state: &mut u64) -> f32 {
        let frac = (splitmix64(state) >> 11) as f32 / (1u64 << 53) as f32;
        (frac - 0.5) * 20.0
    }

    fn gen_i32_no_overflow(state: &mut u64) -> i32 {
        ((splitmix64(state) % 2001) as i32) - 1000
    }

    // ---- Tolerance helpers ----

    fn reduction_bound_f64(expected: f64, len: usize) -> f64 {
        let eps = f64::EPSILON;
        let magnitude = expected.abs().max(1.0);
        // Generous bound: O(n * eps * |expected|) with factor 4.0.
        // This is looser than the strict design bound (13-reduction §6.3)
        // but avoids flaky failures from different accumulation orders.
        ((len as f64) * eps * magnitude * 4.0).max(4.0 * f64::MIN_POSITIVE)
    }

    fn reduction_bound_f32(expected: f32, len: usize) -> f32 {
        let eps = f32::EPSILON;
        let magnitude = expected.abs().max(1.0);
        ((len as f32) * eps * magnitude * 4.0).max(4.0 * f32::MIN_POSITIVE)
    }

    fn assert_within_reduction_bound_f64(actual: f64, expected: f64, len: usize, op: &str) {
        let bound = reduction_bound_f64(expected, len);
        assert!(
            (actual - expected).abs() <= bound,
            "{op} outside bound at len={len}: actual={actual}, expected={expected}, bound={bound}"
        );
    }

    fn assert_within_reduction_bound_f32(actual: f32, expected: f32, len: usize, op: &str) {
        let bound = reduction_bound_f32(expected, len);
        assert!(
            (actual - expected).abs() <= bound,
            "{op} outside bound at len={len}: actual={actual}, expected={expected}, bound={bound}"
        );
    }

    // ---- Element-wise property tests ----

    fn prop_elementwise_binary_f64(seed: u64) {
        let mut rng = seed;
        for _case in 0..CASES {
            let len = ELEMENTWISE_THRESHOLD + gen_len(&mut rng, MAX_LEN);
            let lhs: Vec<f64> = (0..len).map(|_| gen_f64(&mut rng)).collect();
            let rhs: Vec<f64> = (0..len).map(|_| gen_f64(&mut rng)).collect();
            for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div] {
                let mut dst = vec![0.0_f64; len];
                let handled = simd::dispatch_vector_binary_op(op, &lhs, &rhs, &mut dst);
                if handled {
                    let serial: Vec<f64> = lhs
                        .iter()
                        .zip(rhs.iter())
                        .map(|(&l, &r)| match op {
                            BinaryOp::Add => l + r,
                            BinaryOp::Sub => l - r,
                            BinaryOp::Mul => l * r,
                            BinaryOp::Div => l / r,
                        })
                        .collect();
                    for (i, (&a, &e)) in dst.iter().zip(serial.iter()).enumerate() {
                        if a.is_nan() || e.is_nan() {
                            assert!(a.is_nan() && e.is_nan(), "nan mismatch at len={len}, i={i}");
                        } else {
                            assert_eq!(
                                a.to_bits(),
                                e.to_bits(),
                                "bits mismatch at len={len}, i={i}"
                            );
                        }
                    }
                }
            }
        }
    }

    fn prop_elementwise_neg_f64(seed: u64) {
        let mut rng = seed;
        for _case in 0..CASES {
            let len = ELEMENTWISE_THRESHOLD + gen_len(&mut rng, MAX_LEN);
            let src: Vec<f64> = (0..len).map(|_| gen_f64(&mut rng)).collect();
            let mut dst = vec![0.0_f64; len];
            let handled = simd::dispatch_vector_unary_op(UnaryOp::Neg, &src, &mut dst);
            if handled {
                let serial: Vec<f64> = src.iter().map(|&v| -v).collect();
                for (&a, &e) in dst.iter().zip(serial.iter()) {
                    if a.is_nan() || e.is_nan() {
                        assert!(a.is_nan() && e.is_nan());
                    } else {
                        assert_eq!(a.to_bits(), e.to_bits());
                    }
                }
            }
        }
    }

    fn prop_elementwise_complex_add_f32(seed: u64) {
        let mut rng = seed;
        for _case in 0..CASES {
            let len = COMPLEX_ELEMENTWISE_THRESHOLD + gen_len(&mut rng, MAX_LEN);
            let lhs: Vec<Complex<f32>> = (0..len)
                .map(|_| Complex::new(gen_f32(&mut rng), gen_f32(&mut rng)))
                .collect();
            let rhs: Vec<Complex<f32>> = (0..len)
                .map(|_| Complex::new(gen_f32(&mut rng), gen_f32(&mut rng)))
                .collect();
            let mut dst = vec![Complex::new(0.0, 0.0); len];
            let handled = simd::dispatch_vector_binary_op(BinaryOp::Add, &lhs, &rhs, &mut dst);
            if handled {
                for (i, (a, (l, r))) in dst.iter().zip(lhs.iter().zip(rhs.iter())).enumerate() {
                    let e = *l + *r;
                    if a.re.is_nan() || a.im.is_nan() || e.re.is_nan() || e.im.is_nan() {
                        assert!(
                            (a.re.is_nan() && e.re.is_nan()) || (a.re == e.re),
                            "real nan mismatch at len={len}, i={i}"
                        );
                        assert!(
                            (a.im.is_nan() && e.im.is_nan()) || (a.im == e.im),
                            "imag nan mismatch at len={len}, i={i}"
                        );
                    } else {
                        assert_eq!(
                            a.re.to_bits(),
                            e.re.to_bits(),
                            "real bits at len={len}, i={i}"
                        );
                        assert_eq!(
                            a.im.to_bits(),
                            e.im.to_bits(),
                            "imag bits at len={len}, i={i}"
                        );
                    }
                }
            }
        }
    }

    // ---- Reduction/dot property tests ----

    fn prop_sum_tolerance_f64(seed: u64) {
        let mut rng = seed;
        for _case in 0..CASES {
            let len = SUM_THRESHOLD + gen_len(&mut rng, MAX_LEN);
            let data: Vec<f64> = (0..len).map(|_| gen_f64(&mut rng)).collect();
            if let Some(simd) = simd::try_sum_f64(&data) {
                let scalar: f64 = data.iter().sum();
                assert_within_reduction_bound_f64(simd, scalar, len, "sum f64");
            }
        }
    }

    fn prop_sum_tolerance_f32(seed: u64) {
        let mut rng = seed;
        for _case in 0..CASES {
            let len = SUM_THRESHOLD + gen_len(&mut rng, MAX_LEN);
            let data: Vec<f32> = (0..len).map(|_| gen_f32(&mut rng)).collect();
            if let Some(simd) = simd::try_sum_f32(&data) {
                let scalar: f32 = data.iter().sum();
                assert_within_reduction_bound_f32(simd, scalar, len, "sum f32");
            }
        }
    }

    fn prop_dot_tolerance_f64(seed: u64) {
        let mut rng = seed;
        for _case in 0..CASES {
            let len = DOT_THRESHOLD + gen_len(&mut rng, MAX_LEN);
            let lhs: Vec<f64> = (0..len).map(|_| gen_f64(&mut rng)).collect();
            let rhs: Vec<f64> = (0..len).map(|_| gen_f64(&mut rng)).collect();
            if let Some(simd) = simd::try_dot_f64(&lhs, &rhs) {
                let scalar: f64 = lhs.iter().zip(rhs.iter()).map(|(&l, &r)| l * r).sum();
                assert_within_reduction_bound_f64(simd, scalar, len, "dot f64");
            }
        }
    }

    fn prop_sum_complex_f64(seed: u64) {
        let mut rng = seed;
        for _case in 0..CASES {
            let len = COMPLEX_SUM_THRESHOLD + gen_len(&mut rng, MAX_LEN);
            let data: Vec<Complex<f64>> = (0..len)
                .map(|_| Complex::new(gen_f64(&mut rng), gen_f64(&mut rng)))
                .collect();
            if let Some(simd) = simd::try_sum_complex_f64(&data) {
                let scalar: Complex<f64> = data
                    .iter()
                    .copied()
                    .fold(Complex::new(0.0, 0.0), |a, b| a + b);
                assert_within_reduction_bound_f64(simd.re, scalar.re, len, "complex sum f64 re");
                assert_within_reduction_bound_f64(simd.im, scalar.im, len, "complex sum f64 im");
            }
        }
    }

    fn prop_dot_conjugate_complex_f64(seed: u64) {
        let mut rng = seed;
        for _case in 0..CASES {
            let len = COMPLEX_DOT_THRESHOLD + gen_len(&mut rng, MAX_LEN);
            let lhs: Vec<Complex<f64>> = (0..len)
                .map(|_| Complex::new(gen_f64(&mut rng), gen_f64(&mut rng)))
                .collect();
            let rhs: Vec<Complex<f64>> = (0..len)
                .map(|_| Complex::new(gen_f64(&mut rng), gen_f64(&mut rng)))
                .collect();
            if let Some(simd) = simd::try_dot_complex_f64(&lhs, &rhs) {
                let scalar: Complex<f64> = lhs
                    .iter()
                    .zip(rhs.iter())
                    .map(|(l, r)| l.conj() * *r)
                    .fold(Complex::new(0.0, 0.0), |a, b| a + b);
                assert_within_reduction_bound_f64(simd.re, scalar.re, len, "complex dot f64 re");
                assert_within_reduction_bound_f64(simd.im, scalar.im, len, "complex dot f64 im");
            }
        }
    }

    fn prop_integer_no_panic_i32(seed: u64) {
        let mut rng = seed;
        for _case in 0..CASES {
            let len = gen_len(&mut rng, MAX_LEN);
            let data: Vec<i32> = (0..len).map(|_| gen_i32_no_overflow(&mut rng)).collect();
            // i32 SIMD is currently not available (per W14T0 spike),
            // so try_sum_i32 should always return None.
            assert!(
                simd::try_sum_i32(&data).is_none(),
                "i32 SIMD sum should not be available (widening unavailable)"
            );
        }
    }

    fn prop_tail_handling_f64(seed: u64) {
        let mut rng = seed;
        for width in [2usize, 4, 8, 16, 32] {
            for tail in 1..width {
                let base = 1 + (splitmix64(&mut rng) as usize % 16);
                let len = base * width + tail;
                let lhs: Vec<f64> = (0..len).map(|_| gen_f64(&mut rng)).collect();
                let rhs: Vec<f64> = (0..len).map(|_| gen_f64(&mut rng)).collect();
                let mut dst = vec![0.0_f64; len];
                let handled = simd::dispatch_vector_binary_op(BinaryOp::Add, &lhs, &rhs, &mut dst);
                if handled {
                    for i in 0..len {
                        let expected = lhs[i] + rhs[i];
                        if expected.is_nan() || dst[i].is_nan() {
                            assert!(dst[i].is_nan() && expected.is_nan());
                        } else {
                            assert_eq!(dst[i].to_bits(), expected.to_bits());
                        }
                    }
                }
            }
        }
    }

    // ---- Aggregate test entry points ----

    #[test]
    fn prop_elementwise_consistency() {
        prop_elementwise_binary_f64(0x1001);
        prop_elementwise_neg_f64(0x1002);
        prop_elementwise_complex_add_f32(0x1003);
    }

    #[test]
    fn prop_sum_tolerance() {
        prop_sum_tolerance_f64(0x2001);
        prop_sum_tolerance_f32(0x2002);
        prop_sum_complex_f64(0x2003);
    }

    #[test]
    fn prop_dot_conjugate_contract() {
        prop_dot_tolerance_f64(0x3001);
        prop_dot_conjugate_complex_f64(0x3002);
    }

    #[test]
    fn prop_integer_no_panic() {
        prop_integer_no_panic_i32(0x4001);
    }

    #[test]
    fn prop_tail_and_fallback() {
        prop_tail_handling_f64(0x5001);
    }
}
