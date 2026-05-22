//! usize is not a valid element type (§5.21 line 669).
use xenon::tensor::Tensor1;

fn main() {
    let _t: Tensor1<usize> = Tensor1::<usize>::from_shape_vec([3], vec![1usize, 2, 3]).unwrap(); //~ ERROR: usize
}