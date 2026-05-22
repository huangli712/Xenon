//! bool does not participate in unique operation (§5.21 line 672).
use xenon::tensor::Tensor1;

fn main() {
    let t = Tensor1::<bool>::from_shape_vec([2], vec![true, false]).unwrap();
    let _u = t.unique(); //~ ERROR: bool
}