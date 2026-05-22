use xenon::dimension::Ix1;
use xenon::tensor::Tensor;

struct NotADimension;
type WrongDimTensor = Tensor<f64, NotADimension>; //~ ERROR: Dimension

fn main() {
    let _x: WrongDimTensor;
}