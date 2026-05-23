use xenon::tensor::Tensor1;

#[derive(Clone, Debug)]
struct NotElement;

fn main() {
    // `zeros` is declared on `impl TensorBase<Owned<A: Element>, D>`,
    // so monomorphising it with `A = NotElement` must be rejected with
    // an unsatisfied `Element` bound. A bare type alias would NOT trip
    // this bound because `TensorBase<S, D>` itself only requires
    // `S: RawStorage`, and `Owned<A>` implements `RawStorage` for any
    // `A` regardless of `Element`.
    let _x = Tensor1::<NotElement>::zeros([3]); //~ ERROR: Element
}