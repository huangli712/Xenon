use xenon::complex::Complex;
use xenon::dimension::{Ix1, Ix2};
use xenon::index::{SliceInfo, SliceInfoElem, SliceInfoIndices};
use xenon::tensor::{Tensor1, Tensor2, TensorView1};

pub fn sequential_1d(size: usize) -> Tensor1<f64> {
    Tensor1::from_shape_vec([size], (0..size).map(|idx| idx as f64).collect())
        .expect("shape and data length must match")
}

pub fn sequential_1d_f32(size: usize) -> Tensor1<f32> {
    Tensor1::from_shape_vec([size], (0..size).map(|idx| idx as f32).collect())
        .expect("shape and data length must match")
}

pub fn sequential_1d_i32(size: usize) -> Tensor1<i32> {
    // Used by simd_sum_compare's i32 admission / scalar-fallback path
    // (27-benchmark §5.5 simd_sum_compare; §5.4.2 integer extension).
    Tensor1::from_shape_vec([size], (0..size).map(|idx| idx as i32).collect())
        .expect("shape and data length must match")
}

pub fn sequential_1d_i64(size: usize) -> Tensor1<i64> {
    // Used by par_sum_compare which is fixed to i64 (27-benchmark §5.5).
    Tensor1::from_shape_vec([size], (0..size).map(|idx| idx as i64).collect())
        .expect("shape and data length must match")
}

pub fn sequential_2d(rows: usize, cols: usize) -> Tensor2<f64> {
    Tensor2::from_shape_vec([rows, cols], (0..rows * cols).map(|idx| idx as f64).collect())
        .expect("shape and data length must match")
}

pub fn complex_1d(size: usize) -> Tensor1<Complex<f64>> {
    Tensor1::from_shape_vec(
        [size],
        (0..size).map(|idx| Complex::new(idx as f64, -(idx as f64))).collect(),
    )
    .expect("shape and data length must match")
}

/// Non-contiguous 1D fixture: an F-order 2×n owner, with `view()` returning
/// row 1 as a stride-n TensorView1. The owner is held by the caller so the
/// borrow checker enforces the view's lifetime — no leaking.
///
/// Layout (27-benchmark §5.2):
///   owner = sequential_2d(2, n)            // F-order, shape [2, n]
///   view  = owner[1, 0..n]                 // stride != 1, len = n
pub struct StridedFixture1D {
    pub owner: Tensor2<f64>,
}

impl StridedFixture1D {
    /// Returns a non-contiguous TensorView1 over row 1 of the 2×n owner.
    /// Uses the SliceInfo three-argument constructor frozen in 17-indexing.md §5.1.
    pub fn view(&self) -> TensorView1<'_, f64> {
        let n = self.owner.shape()[1];
        let mut elems = [None; 6];
        elems[0] = Some(SliceInfoElem::Index(1));
        elems[1] = Some(SliceInfoElem::Range { start: 0, end: n });
        let info: SliceInfo<Ix1, Ix2> = SliceInfo::new(
            SliceInfoIndices::Inline { len: 2, elems },
            Ix2::default(),
            Ix1::default(),
        )
        .expect("structurally valid slice info");
        // Call slice() directly on the owner; the returned view borrows
        // from self.owner which is alive for &self's duration.
        self.owner
            .slice(info)
            .expect("slice bounds within owner shape")
    }
}

/// Construct a StridedFixture1D backed by an F-order 2×n owner.
/// Caller MUST keep the returned fixture alive for as long as the view is used.
/// (27-benchmark §5.2 fixture constructor)
pub fn strided_view_1d(n: usize) -> StridedFixture1D {
    StridedFixture1D {
        owner: sequential_2d(2, n),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_sequential_1d_shape_and_values() {
        let tensor = sequential_1d(4);
        assert_eq!(tensor.shape(), &[4]);
        assert_eq!(tensor.iter().copied().collect::<Vec<_>>(), vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_sequential_2d_shape() {
        let tensor = sequential_2d(2, 3);
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.len(), 6);
    }

    #[test]
    fn test_strided_fixture_view_shape() {
        let fixture = strided_view_1d(8);
        let view = fixture.view();
        assert_eq!(view.shape(), &[8]);
        // view is row 1 of a 2×8 F-order matrix → stride != 1
        assert_ne!(view.strides()[0], 1);
    }
}
