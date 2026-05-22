//! bool does not participate in arithmetic (§5.21 line 673).
use xenon::tensor::Tensor1;

fn main() {
    let a = Tensor1::<bool>::from_shape_vec([2], vec![true, false]).unwrap();
    let b = Tensor1::<bool>::from_shape_vec([2], vec![false, true]).unwrap();
    let _r = &a + &b; //~ ERROR: Add
}