//! Storage representation mismatch: ViewRepr vs OwnedRepr (§5.21 line 668).
use xenon::dimension::Ix1;
use xenon::storage::Owned;
use xenon::tensor::{Tensor1, TensorBase, TensorView1};

fn take_owned_only<D>(_t: TensorBase<Owned<f64>, D>)
where
    D: xenon::dimension::Dimension,
{
}

fn main() {
    let t = Tensor1::<f64>::from_shape_vec([2], vec![1.0, 2.0]).unwrap();
    let v: TensorView1<f64> = t.view(); // ViewRepr<&f64>
    take_owned_only(v); //~ ERROR: expected
}