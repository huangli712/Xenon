//! Storage representations for Xenon backing buffers.
//!
//! The storage layer provides four concrete modes from the public storage
//! taxonomy:
//!
//! - `Owned<A>` owns readable and writable data and clones by deep copy.
//! - `ViewRepr<'a, A>` is a borrowed read-only view and clones by O(1)
//!   metadata copy.
//! - `ViewMutRepr<'a, A>` is an exclusive borrowed mutable view and is not
//!   cloneable.
//! - `ArcRepr<A>` owns shared read-only data and clones by reference-count
//!   increment.
//!
//! `ArcRepr<A>` and `ViewRepr<'a, A>` both expose read-only access, but their
//! ownership models differ: `ArcRepr<A>` is an owning shared handle, while
//! `ViewRepr<'a, A>` is a borrowed read-only view tied to an external lifetime.
//!
//! # Thread Safety
//!
//! Storage thread-safety follows the matrix below:
//!
//! | Storage               | Send | Sync | Condition              |
//! |-----------------------|------|------|------------------------|
//! | `Owned<A>`            | yes  | yes  | `A: Send` / `A: Sync`  |
//! | `ViewRepr<'a, A>`     | yes  | yes  | `A: Sync`              |
//! | `ViewMutRepr<'a, A>`  | yes  | no   | `A: Send`              |
//! | `ArcRepr<A>`          | yes  | yes  | `A: Send + Sync`       |
//!
//! `ViewMutRepr` is intentionally not `Sync`; parallel mutable execution must
//! split exclusive access into non-overlapping chunks before crossing threads.

mod alloc;
mod arc;
mod owned;
mod traits;
mod view;
mod viewmut;

pub use arc::ArcRepr;
pub use owned::Owned;
pub use traits::{
    IsOwned, IsShared, IsView, IsViewMut,
    RawStorage, RawStorageMut, Storage, StorageIntoOwned, StorageMut,
    StorageOwned, StorageShared,
};
pub use view::ViewRepr;
pub use viewmut::ViewMutRepr;

/// Short alias for [`ViewRepr`].
pub type View<'a, A> = ViewRepr<'a, A>;
/// Short alias for [`ViewMutRepr`].
pub type ViewMut<'a, A> = ViewMutRepr<'a, A>;
