//! u8/u16/u32/u64 are not valid element types (§5.21 line 670).
use xenon::tensor::Tensor1;

fn main() {
    let _t: Tensor1<u32> = Tensor1::<u32>::from_shape_vec([2], vec![1u32, 2]).unwrap(); //~ ERROR: u32
}