use xenon::dimension::Ix1;
use xenon::tensor::Tensor;

#[derive(Clone, Debug)]
struct NotElement;

type BadTensor = Tensor<NotElement, Ix1>; //~ ERROR: Element

fn main() {
    let _x: BadTensor;
}