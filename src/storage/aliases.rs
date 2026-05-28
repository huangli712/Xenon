//! Short type aliases for storage representations.

use super::ViewMutRepr;
use super::ViewRepr;

/// Short alias for [`ViewRepr`].
pub type View<'a, A> = ViewRepr<'a, A>;
/// Short alias for [`ViewMutRepr`].
pub type ViewMut<'a, A> = ViewMutRepr<'a, A>;
