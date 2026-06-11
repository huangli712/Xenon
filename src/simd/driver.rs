//! SIMD runtime support shared by the element-wise and reduction kernels:
//! the cached `pulp::Arch` singleton and the SIMD lane-width query. The
//! dispatch facades now live with their kernels in `crate::math`,
//! `crate::matrix`, and `crate::reduction`.

use pulp::Arch;

use std::sync::OnceLock;

use crate::element::SimdElement;

// ----------------------------------------------------------------------------
// Arch cache
// ----------------------------------------------------------------------------

/// Returns a reference to the lazily-initialized static `pulp::Arch`.
///
/// The `OnceLock` is placed inside the function body so that external
/// code cannot bypass the accessor to read the cache directly.
pub(crate) fn get_arch() -> &'static Arch {
    static ARCH: OnceLock<Arch> = OnceLock::new();
    ARCH.get_or_init(Arch::new)
}

// ----------------------------------------------------------------------------
// Capability query
// ----------------------------------------------------------------------------

/// Returns the SIMD lane width for `T` on the current platform.
///
/// `Some(width > 1)` means the platform exposes a usable SIMD lane width.
/// `None` means the feature is disabled, the type is unsupported, or no
/// suitable ISA is available.
///
#[allow(dead_code, reason = "returns None until ISA dispatch is wired")]
pub(crate) fn simd_vector_width<T: SimdElement>() -> Option<usize> {
    None
}
