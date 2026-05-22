//! Data generators for parameterized integration tests.
//!
//! Provides standard shapes and helper constructors used across W29
//! integration test files (28-tests §5.3).

use xenon::tensor::Tensor2;

/// Standard 2D shapes for parameterized testing (`28-tests §5.3 L394-399`).
pub fn standard_shapes_2d() -> Vec<(usize, usize)> {
    vec![
        (0, 0),
        (1, 1),
        (1, 5),
        (5, 1),
        (3, 4),
        (4, 3),
        (8, 8),
        (64, 64),
    ]
}

/// Generate a 2D tensor suitable for producing a non-contiguous transposed
/// view via `owner.transpose()` (`28-tests §5.3 L405-411`).
///
/// The tensor is constructed with shape `[cols, rows]` so that a subsequent
/// `transpose()` yields a `[rows, cols]` view with non-contiguous strides.
pub fn non_contiguous_2d_owner(rows: usize, cols: usize) -> Tensor2<f64> {
    Tensor2::<f64>::from_shape_vec(
        [cols, rows],
        (0..cols * rows).map(|idx| idx as f64).collect(),
    )
    .expect("shape and data length must match")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generators_standard_shapes_2d() {
        let shapes = standard_shapes_2d();
        assert!(shapes.contains(&(0, 0)));
        assert!(shapes.contains(&(8, 8)));
        assert_eq!(shapes.len(), 8);
    }

    #[test]
    fn test_generators_non_contiguous_2d_owner_transpose_view() {
        let owner = non_contiguous_2d_owner(3, 4);
        assert_eq!(owner.shape(), &[4, 3]);
        let view = owner.transpose();
        assert_eq!(view.shape(), &[3, 4]);
    }
}