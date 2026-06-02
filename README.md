# Xenon

Rust N-dimensional tensor library for scientific computing.

## Features

- N-dimensional arrays with static (0-6D) and dynamic (IxDyn) dimension types
- Column-major (F-order) contiguous storage with BLAS/LAPACK-compatible layout
- Custom FFI-friendly complex number type (`Complex`)
- Type-level broadcasting via `BroadcastDim` trait
- Optional SIMD acceleration via `pulp` crate
- Optional parallel computing support via `rayon` crate
- Convenient dimension creation from tuples, arrays, and slices

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
xenon = "0.0.18"
```

## Documentation

Full API documentation is published on [docs.rs](https://docs.rs/xenon).

## License

MIT
