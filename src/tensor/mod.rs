//! Tensor module: the central `TensorBase<S, D>` type, storage semantics,
//! type aliases, and all method implementations.
//!
//! | Sub-module | Responsibility |
//! |---|---|
//! | `types` | `TensorBase`, `OwnedRawParts`, and semantic query enums |
//! | `traits` | `StorageSemantics` trait and its 4 storage-type impls |
//! | `aliases` | 36 type aliases (re‑exported via `aliases::*`) |
//! | `impls` | All method implementations, validators, constructors, and tests |

mod types;
mod traits;
mod aliases;
mod impls;

pub use types::{TensorBase, OwnedRawParts};
pub use types::{DataLocation, StorageKind, AccessSemantics, AliasClass};
pub use traits::StorageSemantics;
pub use aliases::*;
