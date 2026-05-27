//! Dimensions describe tensor rank and shape metadata.
//!
//! `Ix0` through `Ix6` represent statically ranked tensors, while `IxDyn`
//! stores a runtime-rank shape. The `Dimension` and `Reverse` traits are
//! sealed so the crate can keep stride, indexing, and broadcasting invariants
//! coherent.
//!
//! ## Type overview
//!
//! | Type | Rank | Description |
//! |------|------|-------------|
//! | `Ix0` | 0 | Scalar (zero-dimensional) |
//! | `Ix1` | 1 | Vector |
//! | `Ix2` | 2 | Matrix |
//! | `Ix3`–`Ix6` | 3–6 | Higher-rank tensors |
//! | `IxDyn` | runtime | Dynamically-sized dimensions |
//!
//! Conversion from tuples (`(usize,)`, etc.), arrays (`[usize; N]`),
//! and slices is provided through `IntoDimension`.

pub mod axes;
pub mod broadcast;
pub mod dynamic;
pub mod fixed;
pub mod into;

// Public re-exports — the canonical access path for dimension types.
pub use axes::Axis;
pub use broadcast::BroadcastDim;
pub use dynamic::IxDyn;
pub use fixed::{Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6};
pub use into::IntoDimension;

mod types;

pub use types::{Dimension, Reverse, RemoveAxis};

/// Maximum number of dimensions representable on this platform.
pub const MAX_DIMENSION: usize = usize::MAX;

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile-time path check: ensure all four submodules are reachable
    /// via the public dimension module path. If any submodule is renamed
    /// or removed, these `use` statements fail to compile.
    #[test]
    fn test_dimension_submodules_reachable() {
        #[allow(unused_imports)]
        use crate::dimension::axes;
        #[allow(unused_imports)]
        use crate::dimension::dynamic;
        #[allow(unused_imports)]
        use crate::dimension::fixed;
        #[allow(unused_imports)]
        use crate::dimension::into;
    }

    /// §8.2: MAX_DIMENSION constant value.
    #[test]
    fn test_max_dimension_is_usize_max() {
        assert_eq!(MAX_DIMENSION, usize::MAX);
    }


    /// Public re-exports are reachable via `crate::dimension::*`.
    #[test]
    fn test_public_exports_reachable() {
        let _: Ix0 = Ix0;
        let _: Ix1 = Ix1(1);
        let _: IxDyn = IxDyn::new();
        let _: Axis = Axis::new(0);
        // IntoDimension trait reachable: tuple → Ix3.
        let _: Ix3 = (1, 2, 3).into_dimension();
    }

    /// §8.2: Mirrors the canonical doc examples on Ix2 / Axis /
    /// IntoDimension to ensure they exercise real public API paths.
    #[test]
    fn test_public_doc_examples_execute() {
        // Ix2 example
        let dim = Ix2(10, 20);
        assert_eq!(dim.ndim(), 2);
        assert_eq!(dim.slice(), &[10, 20]);
        assert_eq!(dim.checked_size(), Ok(200));
        assert_eq!(dim[0], 10);

        // Axis example
        let ax = Axis::new(0);
        assert!(ax.is_first());

        // IntoDimension example
        let d3: Ix3 = (2, 3, 4).into_dimension();
        assert_eq!(d3.slice(), &[2, 3, 4]);
    }
}
