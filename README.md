# Xenon

Rust N-dimensional tensor library for scientific computing.

## Features

- N-dimensional arrays with static (0-6D) and dynamic dimensions (`IxDyn` for
  runtime-rank tensors)
- Column-major (F-order) default, with helper APIs and compatibility checks
  for upstream BLAS/LAPACK integration when the layout preconditions are
  satisfied
- Custom FFI-friendly complex number type
- Optional SIMD (pulp) and parallel (rayon) acceleration

## Quick Start

```rust
use xenon::prelude::*;
use xenon::tensor::Tensor;

fn main() -> xenon::Result<()> {
    let a = Tensor::<f64, _>::zeros([2, 3])?;
    let b = Tensor::<f64, _>::ones([2, 3])?;
    let c = (&a + &b)?;
    assert_eq!(c.shape(), &[2, 3]);
    assert_eq!(c.sum(), 6.0);
    Ok(())
}
```

## Installation

```toml
[dependencies]
xenon = "0.0.8"
```

## Documentation

Full API documentation is published on [docs.rs](https://docs.rs/xenon).

## License

MIT
