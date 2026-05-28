//! Short type aliases for storage representations.

use crate::storage::ViewMutRepr;
use crate::storage::ViewRepr;

/// Short alias for [`ViewRepr`].
pub type View<'a, A> = ViewRepr<'a, A>;
/// Short alias for [`ViewMutRepr`].
pub type ViewMut<'a, A> = ViewMutRepr<'a, A>;
