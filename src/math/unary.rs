//! Unary element-wise operations: abs, neg, signum, square,
//! math functions sin/sqrt/exp/ln/floor/ceil,
//! complex conjugate/modulus, logical not.

use crate::complex::{Complex, ComplexFloat};
use crate::dimension::Dimension;
#[cfg(feature = "parallel")]
use crate::dispatch::ParallelExecStrategy;
use crate::dispatch::{ExecPath, select_exec_path};
use crate::element::{
    CheckedNeg, ComplexScalar, Element, Numeric, OrderedCompareElement, RealScalar,
};

use crate::storage::Storage;
use crate::tensor::{Tensor, TensorBase};

// ============================================================================
// Private per-type dispatch traits
// ============================================================================

/// Per-type unary step for `neg` and `square`.
///
/// - Integer types (`i32`, `i64`): use checked arithmetic; overflow cases such
///   as negating `i{32,64}::MIN` panic with a diagnostic message.
/// - Floating types (`f32`, `f64`): delegate to the native operators; NaN /
///   Inf propagate per IEEE 754.
/// - Complex types (`Complex<f32>`, `Complex<f64>`): use the `Neg` / `Mul`
///   standard operators. `abs` / `signum` do not cover complex numbers (the
///   trait bound `OrderedCompareElement` excludes them at compile time).
///
/// Sealed via `Numeric: Sealed`. Consumers cannot add new impls; all six
/// concrete `Numeric` types provide their own impl in this file.
///
/// The `*_step_with_ctx` variants propagate `element_index` and `shape` into
/// the integer panic text. Default implementations forward to the
/// context-free `*_step` (zero cost on non-integer paths after `#[inline]`);
/// integer impls override these to embed the full diagnostic fields.
///
/// The `SimdElement` supertrait keeps this trait usable from SIMD-aware
/// callers without adding a feature gate. All six `UnaryArith` impls
/// (`i32` / `i64` / `f32` / `f64` / `Complex<f32>` / `Complex<f64>`) already
/// implement `SimdElement`, so adding the supertrait does not narrow the
/// sealed set.
trait UnaryArith: Numeric + crate::element::SimdElement + 'static {
    /// Element-wise negation; integer path panics on `MIN`.
    fn neg_step(x: Self) -> Self;
    /// Element-wise square `x * x`; integer path panics on overflow.
    fn square_step(x: Self) -> Self;

    /// Context-aware variant used by integer monomorphizations to embed
    /// `element_index` and `shape` into panic text. The default impl
    /// forwards to the context-free `*_step` (non-integer types have no
    /// panic path, so the index and shape are simply dropped).
    #[inline]
    fn neg_step_with_ctx(x: Self, _idx: usize, _shape: &[usize]) -> Self {
        Self::neg_step(x)
    }
    /// Context-aware variant of `square_step`; see `neg_step_with_ctx`.
    #[inline]
    fn square_step_with_ctx(x: Self, _idx: usize, _shape: &[usize]) -> Self {
        Self::square_step(x)
    }
}

// ============================================================================
// Integer impls (checked arithmetic)
// ============================================================================

/// Checked-arithmetic unary step for `i32`: `neg` and `square` panic with a
/// diagnostic message on overflow (`i32::MIN` for `neg`, large magnitudes
/// for `square`).
impl UnaryArith for i32 {
    #[inline]
    fn neg_step(x: Self) -> Self {
        <Self as CheckedNeg>::checked_neg(x).unwrap_or_else(|| {
            panic!(
                "integer overflow: operation=neg, type={}, trigger={}",
                "i32", x
            )
        })
    }
    #[inline]
    fn square_step(x: Self) -> Self {
        x.checked_mul(x).unwrap_or_else(|| {
            panic!(
                "integer overflow: operation=square, type={}, trigger={}",
                "i32", x
            )
        })
    }
    #[inline]
    fn neg_step_with_ctx(x: Self, idx: usize, shape: &[usize]) -> Self {
        <Self as CheckedNeg>::checked_neg(x).unwrap_or_else(|| {
            panic!(
                "integer overflow: operation=neg, type={}, trigger={}, \
                 element_index={}, shape={:?}",
                "i32", x, idx, shape
            )
        })
    }
    #[inline]
    fn square_step_with_ctx(x: Self, idx: usize, shape: &[usize]) -> Self {
        x.checked_mul(x).unwrap_or_else(|| {
            panic!(
                "integer overflow: operation=square, type={}, trigger={}, \
                 element_index={}, shape={:?}",
                "i32", x, idx, shape
            )
        })
    }
}

/// Checked-arithmetic unary step for `i64`: `neg` and `square` panic with a
/// diagnostic message on overflow (`i64::MIN` for `neg`, large magnitudes
/// for `square`).
impl UnaryArith for i64 {
    #[inline]
    fn neg_step(x: Self) -> Self {
        <Self as CheckedNeg>::checked_neg(x).unwrap_or_else(|| {
            panic!(
                "integer overflow: operation=neg, type={}, trigger={}",
                "i64", x
            )
        })
    }
    #[inline]
    fn square_step(x: Self) -> Self {
        x.checked_mul(x).unwrap_or_else(|| {
            panic!(
                "integer overflow: operation=square, type={}, trigger={}",
                "i64", x
            )
        })
    }
    #[inline]
    fn neg_step_with_ctx(x: Self, idx: usize, shape: &[usize]) -> Self {
        <Self as CheckedNeg>::checked_neg(x).unwrap_or_else(|| {
            panic!(
                "integer overflow: operation=neg, type={}, trigger={}, \
                 element_index={}, shape={:?}",
                "i64", x, idx, shape
            )
        })
    }
    #[inline]
    fn square_step_with_ctx(x: Self, idx: usize, shape: &[usize]) -> Self {
        x.checked_mul(x).unwrap_or_else(|| {
            panic!(
                "integer overflow: operation=square, type={}, trigger={}, \
                 element_index={}, shape={:?}",
                "i64", x, idx, shape
            )
        })
    }
}

// ============================================================================
// Float impls (IEEE 754)
// ============================================================================

/// IEEE 754 unary step for `f32`: native `-x` and `x * x`, NaN / Inf
/// propagate.
impl UnaryArith for f32 {
    #[inline]
    fn neg_step(x: Self) -> Self {
        -x
    }
    #[inline]
    fn square_step(x: Self) -> Self {
        x * x
    }
}

/// IEEE 754 unary step for `f64`: native `-x` and `x * x`, NaN / Inf
/// propagate.
impl UnaryArith for f64 {
    #[inline]
    fn neg_step(x: Self) -> Self {
        -x
    }
    #[inline]
    fn square_step(x: Self) -> Self {
        x * x
    }
}

// ============================================================================
// Complex impls (only neg / square; abs / signum excluded at compile time
// via `OrderedCompareElement` — Complex does NOT implement it).
// ============================================================================

/// Unary step for `Complex<f32>`: standard `Neg` and `Mul` operators; NaN /
/// Inf propagate per IEEE 754 on the real and imaginary components.
impl UnaryArith for crate::complex::Complex<f32> {
    #[inline]
    fn neg_step(x: Self) -> Self {
        -x
    }
    #[inline]
    fn square_step(x: Self) -> Self {
        x * x
    }
}

/// Unary step for `Complex<f64>`: standard `Neg` and `Mul` operators; NaN /
/// Inf propagate per IEEE 754 on the real and imaginary components.
impl UnaryArith for crate::complex::Complex<f64> {
    #[inline]
    fn neg_step(x: Self) -> Self {
        -x
    }
    #[inline]
    fn square_step(x: Self) -> Self {
        x * x
    }
}

/// Per-type unary step for `abs` and `signum`, restricted to ordered types.
///
/// The trait bound `Numeric + OrderedCompareElement` is sealed to `i32`,
/// `i64`, `f32`, and `f64`; complex types are excluded at compile time.
///
/// The `*_step_with_ctx` variants follow the same contract as `UnaryArith`:
/// default implementations forward to `*_step`; integer impls override to
/// embed `element_index` and `shape` into panic text.
///
/// The `SimdElement` supertrait keeps this trait usable from SIMD-aware
/// callers without adding a feature gate. All four `OrderedUnaryArith`
/// impls (`i32` / `i64` / `f32` / `f64`) already implement `SimdElement`,
/// so the supertrait does not narrow the sealed set.
trait OrderedUnaryArith:
    Numeric + OrderedCompareElement + crate::element::SimdElement + 'static
{
    /// Element-wise absolute value; integer path panics on `MIN`.
    fn abs_step(x: Self) -> Self;
    /// Element-wise signum. Integers: `-1` / `0` / `1`.
    /// Floats: delegates to `RealScalar::signum` (IEEE 754).
    fn signum_step(x: Self) -> Self;

    /// Context-aware variant of `abs_step`; integer impls override to embed
    /// `element_index` and `shape` into panic text.
    #[inline]
    fn abs_step_with_ctx(x: Self, _idx: usize, _shape: &[usize]) -> Self {
        Self::abs_step(x)
    }
    /// Context-aware variant of `signum_step`; default impl forwards to the
    /// context-free `signum_step` since neither integers nor floats have a
    /// panic path.
    #[inline]
    fn signum_step_with_ctx(x: Self, _idx: usize, _shape: &[usize]) -> Self {
        Self::signum_step(x)
    }
}

/// Ordered unary step for `i32`: `abs` returns the non-negative value via
/// checked negation and panics on `i32::MIN`; `signum` returns `-1`, `0`,
/// or `1`.
impl OrderedUnaryArith for i32 {
    #[inline]
    fn abs_step(x: Self) -> Self {
        if x >= 0 {
            x
        } else {
            <Self as CheckedNeg>::checked_neg(x).unwrap_or_else(|| {
                panic!(
                    "integer overflow: operation=abs, type={}, trigger={}",
                    "i32", x
                )
            })
        }
    }
    #[inline]
    fn signum_step(x: Self) -> Self {
        x.signum()
    }
    #[inline]
    fn abs_step_with_ctx(x: Self, idx: usize, shape: &[usize]) -> Self {
        if x >= 0 {
            x
        } else {
            <Self as CheckedNeg>::checked_neg(x).unwrap_or_else(|| {
                panic!(
                    "integer overflow: operation=abs, type={}, trigger={}, \
                     element_index={}, shape={:?}",
                    "i32", x, idx, shape
                )
            })
        }
    }
}

/// Ordered unary step for `i64`: `abs` returns the non-negative value via
/// checked negation and panics on `i64::MIN`; `signum` returns `-1`, `0`,
/// or `1`.
impl OrderedUnaryArith for i64 {
    #[inline]
    fn abs_step(x: Self) -> Self {
        if x >= 0 {
            x
        } else {
            <Self as CheckedNeg>::checked_neg(x).unwrap_or_else(|| {
                panic!(
                    "integer overflow: operation=abs, type={}, trigger={}",
                    "i64", x
                )
            })
        }
    }
    #[inline]
    fn signum_step(x: Self) -> Self {
        x.signum()
    }
    #[inline]
    fn abs_step_with_ctx(x: Self, idx: usize, shape: &[usize]) -> Self {
        if x >= 0 {
            x
        } else {
            <Self as CheckedNeg>::checked_neg(x).unwrap_or_else(|| {
                panic!(
                    "integer overflow: operation=abs, type={}, trigger={}, \
                     element_index={}, shape={:?}",
                    "i64", x, idx, shape
                )
            })
        }
    }
}

/// Ordered unary step for `f32`: `abs` and `signum` delegate to
/// `RealScalar`, following IEEE 754 (NaN propagates, signed zeros handled).
impl OrderedUnaryArith for f32 {
    #[inline]
    fn abs_step(x: Self) -> Self {
        <Self as RealScalar>::abs(x)
    }
    #[inline]
    fn signum_step(x: Self) -> Self {
        <Self as RealScalar>::signum(x)
    }
}

/// Ordered unary step for `f64`: `abs` and `signum` delegate to
/// `RealScalar`, following IEEE 754 (NaN propagates, signed zeros handled).
impl OrderedUnaryArith for f64 {
    #[inline]
    fn abs_step(x: Self) -> Self {
        <Self as RealScalar>::abs(x)
    }
    #[inline]
    fn signum_step(x: Self) -> Self {
        <Self as RealScalar>::signum(x)
    }
}

// ============================================================================
// Public unary methods on TensorBase
// ============================================================================

/// Tensor methods for `abs` and `signum`, available on ordered element types
/// (`i32` / `i64` / `f32` / `f64`).
// abs / signum: ordered types (i32/i64/f32/f64).
#[expect(
    private_bounds,
    reason = "OrderedUnaryArith is a private sealed trait; public API bound is equivalent to Numeric + OrderedCompareElement"
)]
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: OrderedUnaryArith,
{
    /// Element-wise absolute value. The integer path panics on `MIN`; the
    /// float path follows IEEE 754 (`abs(NaN) = NaN`).
    ///
    /// Integer types retain the indexed traversal so that overflow panics
    /// can carry `element_index` and `shape` diagnostics. Float types route
    /// through the dispatch-aware helper (no SIMD kernel for `abs` yet, so
    /// the SIMD path falls back to scalar; parallel acceleration kicks in
    /// when the executor selects it).
    pub fn abs(&self) -> Tensor<A, D> {
        #[cfg(feature = "simd")]
        {
            use core::any::TypeId;
            if TypeId::of::<A>() != TypeId::of::<i32>() && TypeId::of::<A>() != TypeId::of::<i64>()
            {
                return apply_unary_with_dispatch(
                    self,
                    |x| <A as OrderedUnaryArith>::abs_step(x),
                    None,
                );
            }
        }
        apply_unary_indexed(self, |x, idx, shape| {
            <A as OrderedUnaryArith>::abs_step_with_ctx(x, idx, shape)
        })
    }

    /// Element-wise signum. Integers return sign-based `-1` / `0` / `1`.
    /// Floats follow IEEE 754 semantics via `RealScalar::signum` (handles
    /// signed zeros and NaN).
    pub fn signum(&self) -> Tensor<A, D> {
        #[cfg(feature = "simd")]
        {
            use core::any::TypeId;
            if TypeId::of::<A>() != TypeId::of::<i32>() && TypeId::of::<A>() != TypeId::of::<i64>()
            {
                return apply_unary_with_dispatch(
                    self,
                    |x| <A as OrderedUnaryArith>::signum_step(x),
                    None,
                );
            }
        }
        apply_unary_indexed(self, |x, idx, shape| {
            <A as OrderedUnaryArith>::signum_step_with_ctx(x, idx, shape)
        })
    }
}

/// Tensor methods for `neg` and `square`, available on every `Numeric`
/// element type (including complex).
// neg / square: all Numeric (including Complex).
#[expect(
    private_bounds,
    reason = "UnaryArith is a private sealed trait; public API bound is equivalent to Numeric"
)]
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: UnaryArith,
{
    /// Element-wise negation. The integer path panics on `MIN`; the float
    /// path uses `-x` (IEEE 754); the complex path uses the `Neg`
    /// operator.
    ///
    /// Integer types retain the indexed traversal for overflow diagnostics.
    /// Float types route through the dispatch-aware helper with a SIMD
    /// `Neg` op tag, so the SIMD path can accelerate when available.
    pub fn neg(&self) -> Tensor<A, D> {
        #[cfg(feature = "simd")]
        {
            use core::any::TypeId;
            if TypeId::of::<A>() != TypeId::of::<i32>() && TypeId::of::<A>() != TypeId::of::<i64>()
            {
                return apply_unary_with_dispatch(
                    self,
                    |x| <A as UnaryArith>::neg_step(x),
                    Some(crate::simd::UnaryOp::Neg),
                );
            }
        }
        apply_unary_indexed(self, |x, idx, shape| {
            <A as UnaryArith>::neg_step_with_ctx(x, idx, shape)
        })
    }

    /// Element-wise square (`x * x`). The integer path panics on overflow;
    /// the float and complex paths use `*` with IEEE 754 propagation.
    pub fn square(&self) -> Tensor<A, D> {
        #[cfg(feature = "simd")]
        {
            use core::any::TypeId;
            if TypeId::of::<A>() != TypeId::of::<i32>() && TypeId::of::<A>() != TypeId::of::<i64>()
            {
                return apply_unary_with_dispatch(
                    self,
                    |x| <A as UnaryArith>::square_step(x),
                    None,
                );
            }
        }
        apply_unary_indexed(self, |x, idx, shape| {
            <A as UnaryArith>::square_step_with_ctx(x, idx, shape)
        })
    }
}

// ============================================================================
// Math functions for RealScalar (f32, f64)
// ============================================================================

/// Tensor methods that compute element-wise real-valued math functions
/// (`sin`, `sqrt`, `exp`, `ln`, `floor`, `ceil`) on `RealScalar` elements.
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: RealScalar,
{
    /// Element-wise sine. IEEE 754 NaN propagates: `sin(NaN) = NaN`.
    ///
    /// Routes through `apply_unary_real_dispatch` so that large tensors can
    /// benefit from parallel acceleration. No SIMD kernel for `sin` is
    /// available yet, so the SIMD path falls back to scalar.
    pub fn sin(&self) -> Tensor<A, D> {
        apply_unary_real_dispatch(self, |x| x.sin())
    }

    /// Element-wise square root. `sqrt(-1.0) = NaN` per IEEE 754.
    pub fn sqrt(&self) -> Tensor<A, D> {
        apply_unary_real_dispatch(self, |x| x.sqrt())
    }

    /// Element-wise exponential. `exp(Inf) = Inf`, `exp(-Inf) = 0.0`.
    pub fn exp(&self) -> Tensor<A, D> {
        apply_unary_real_dispatch(self, |x| x.exp())
    }

    /// Element-wise natural logarithm. `ln(0.0) = -Inf`, `ln(-1.0) = NaN`.
    pub fn ln(&self) -> Tensor<A, D> {
        apply_unary_real_dispatch(self, |x| x.ln())
    }

    /// Element-wise floor. Exact (no tolerance).
    pub fn floor(&self) -> Tensor<A, D> {
        apply_unary_real_dispatch(self, |x| x.floor())
    }

    /// Element-wise ceil. Exact (no tolerance).
    pub fn ceil(&self) -> Tensor<A, D> {
        apply_unary_real_dispatch(self, |x| x.ceil())
    }
}

// ============================================================================
// Complex ops — modulus (type-changing Complex<T> → T)
// ============================================================================

/// Tensor method that computes the element-wise modulus of a complex tensor,
/// returning a real tensor of the same shape.
impl<S, D, T> TensorBase<S, D>
where
    S: Storage<Elem = Complex<T>>,
    D: Dimension,
    T: RealScalar + ComplexFloat,
    Complex<T>: ComplexScalar<Real = T>,
{
    /// Element-wise modulus: `|a + bi| = sqrt(a*a + b*b)`.
    /// Returns a real tensor of the same dimension.
    pub fn modulus(&self) -> Tensor<T, D> {
        apply_unary_map(self, |c| c.norm())
    }
}

// ============================================================================
// Complex ops — conjugate (single generic impl)
// ============================================================================

/// Tensor method that computes the element-wise complex conjugate.
impl<S, D, T> TensorBase<S, D>
where
    S: Storage<Elem = Complex<T>>,
    D: Dimension,
    T: ComplexFloat,
    Complex<T>: Element + Numeric,
{
    /// Element-wise complex conjugate: `(a + bi) → (a - bi)`. Delegates to
    /// `Numeric::conjugate`.
    ///
    /// Routes through `apply_unary_real_dispatch` so that large tensors can
    /// benefit from parallel acceleration. No SIMD kernel for `conjugate`
    /// is available yet (it would require cross-lane operations), so the
    /// SIMD path falls back to scalar.
    pub fn conjugate(&self) -> Tensor<Complex<T>, D> {
        apply_unary_real_dispatch(self, <Complex<T> as Numeric>::conjugate)
    }
}

// ============================================================================
// Logical NOT for bool tensors with dispatch wiring
// ============================================================================

/// Tensor method that computes the element-wise logical NOT of a bool
/// tensor, with execution-path dispatch.
impl<S, D> TensorBase<S, D>
where
    S: Storage<Elem = bool>,
    D: Dimension,
{
    /// Element-wise logical NOT. Returns a bool tensor of the same shape.
    ///
    /// `select_exec_path` routes between Serial and SIMD (both fall back to
    /// the scalar baseline because no bool SIMD kernel exists), and selects
    /// the Parallel path via `par_map` when the `parallel` feature is
    /// enabled and the executor chooses it.
    ///
    /// # Panics
    ///
    /// Panics if `Tensor::zeros(self.raw_dim())` fails. This cannot happen
    /// because `self.raw_dim()` originates from a valid `TensorBase` whose
    /// shape was already validated at construction.
    pub fn not(&self) -> Tensor<bool, D> {
        let len = self.len();
        let is_contiguous = self.is_f_contiguous();
        let alignment_ok = self.is_aligned();
        let (path, guard) = select_exec_path(len, is_contiguous, alignment_ok);
        match path {
            // No bool SIMD kernel exists; Serial and Simd both fall through
            // to the scalar `apply_unary_map` baseline. Parallel routes
            // through `par_map`.
            ExecPath::Serial | ExecPath::Simd => {
                let _ = guard;
                apply_unary_map(self, |x| !x)
            },
            ExecPath::Parallel => {
                #[cfg(feature = "parallel")]
                {
                    let strat = ParallelExecStrategy::auto();
                    let g = guard.expect("ExecPath::Parallel must carry a ParallelGuard");
                    crate::parallel::unary::par_map(self, &strat, g, |x| !*x)
                }
                #[cfg(not(feature = "parallel"))]
                {
                    let _ = guard;
                    apply_unary_map(self, |x| !x)
                }
            },
        }
    }
}

// ============================================================================
// Shared traversal helpers
// ============================================================================

/// Element-wise unary traversal.
///
/// Maps each input element through `f`, producing an output tensor of the
/// same shape. The output element type `O` may differ from the input type
/// `A`, covering both same-type ops (`A → A`, e.g. `neg` / `square`) and
/// type-changing ops (`Complex<T> → T`, e.g. `modulus`).
#[inline]
fn apply_unary_map<A, O, S, D, F>(input: &TensorBase<S, D>, mut f: F) -> Tensor<O, D>
where
    A: Element,
    O: Element,
    S: Storage<Elem = A>,
    D: Dimension,
    F: FnMut(A) -> O,
{
    let mut result = Tensor::<O, D>::zeros(input.raw_dim())
        .expect("input dimension must be valid since input tensor exists");
    for (dst, src) in result.iter_mut().zip(input.iter()) {
        *dst = f(*src);
    }
    result
}

/// Same-type unary traversal with element-index and shape context — variant
/// of `apply_unary_map` that propagates `(idx, &shape)` into the kernel
/// closure.
///
/// Required by the integer monomorphizations of `abs` / `neg` / `square`
/// so that panic messages can embed `operation`, `type`, `trigger`,
/// `element_index`, and (when applicable) `shape`. Non-integer
/// monomorphizations have no panic path; the `idx` / `shape` parameters
/// are zero-cost dropped after inlining.
#[inline]
fn apply_unary_indexed<A, S, D, F>(input: &TensorBase<S, D>, mut f: F) -> Tensor<A, D>
where
    A: Element,
    S: Storage<Elem = A>,
    D: Dimension,
    F: FnMut(A, usize, &[usize]) -> A,
{
    let dim = input.raw_dim();
    let shape_slice: Vec<usize> = dim.slice().to_vec();
    let mut result = Tensor::<A, D>::zeros(dim)
        .expect("input dimension must be valid since input tensor exists");
    for (idx, (dst, src)) in result.iter_mut().zip(input.iter()).enumerate() {
        *dst = f(*src, idx, &shape_slice);
    }
    result
}

// ============================================================================
// Dispatch-aware unary helper (serial/SIMD/parallel routing)
// ============================================================================

/// Dispatch-aware unary helper for methods bounded by `A: RealScalar` or
/// `A: ComplexFloat`-derived element traits that do NOT include
/// `SimdElement` in their supertrait set.
///
/// **Why this exists**: The user-facing math API (`sin`, `sqrt`, `exp`,
/// `ln`, `floor`, `ceil`, `conjugate`) is bounded by the public
/// `RealScalar` and `ComplexFloat` traits. These traits intentionally do
/// NOT include the `pub(crate) SimdElement` trait as a supertrait, because
/// SIMD types must not appear in the public API surface (`pub(crate)`
/// only). Therefore `apply_unary_with_dispatch` (which requires
/// `A: SimdElement`) cannot be called from these methods.
///
/// **Coverage today**: `Sin` / `Sqrt` / `Exp` / `Ln` / `Floor` / `Ceil` /
/// `Conjugate` are reserved for future SIMD coverage, so the SIMD branch
/// of this helper always falls through to scalar.
///
/// **Real acceleration today**: Parallel path via
/// `crate::parallel::unary::par_map` when the `parallel` feature is
/// enabled and `select_exec_path` returns `ExecPath::Parallel`.
///
/// **Future-proof**: When SIMD coverage is extended (e.g. a new
/// `UnaryOp::Sin`), this helper can be upgraded internally to invoke
/// `dispatch_vector_unary_op` for the supported types without changing
/// any method body (since the public bound `A: RealScalar` remains).
fn apply_unary_real_dispatch<A, S, D, F>(input: &TensorBase<S, D>, op: F) -> Tensor<A, D>
where
    A: Element,
    S: Storage<Elem = A>,
    D: Dimension,
    F: Fn(A) -> A + Copy + Send + Sync,
{
    let len = input.len();
    let is_contiguous = input.is_f_contiguous();
    let alignment_ok = input.is_aligned();
    let (path, guard) = select_exec_path(len, is_contiguous, alignment_ok);
    match path {
        // No SIMD kernel for these ops yet; Serial and Simd both route to
        // the scalar baseline.
        ExecPath::Serial | ExecPath::Simd => {
            let _ = guard;
            apply_unary_map(input, op)
        },
        ExecPath::Parallel => {
            #[cfg(feature = "parallel")]
            {
                let strat = ParallelExecStrategy::auto();
                let g = guard.expect("ExecPath::Parallel must carry a ParallelGuard");
                crate::parallel::unary::par_map(input, &strat, g, |x| op(*x))
            }
            #[cfg(not(feature = "parallel"))]
            {
                let _ = guard;
                apply_unary_map(input, op)
            }
        },
    }
}

/// Dispatch-aware variant of `apply_unary_map` — routes between Serial,
/// SIMD, and Parallel paths. The scalar baseline `apply_unary_map` is the
/// SIMD-fallback target.
///
/// `op_tag: Option<simd::UnaryOp>` encodes which SIMD kernel to attempt.
/// `None` causes the SIMD path to be skipped, falling back to Serial /
/// Parallel.
#[cfg(feature = "simd")]
fn apply_unary_with_dispatch<A, S, D, F>(
    input: &TensorBase<S, D>,
    op: F,
    op_tag: Option<crate::simd::UnaryOp>,
) -> Tensor<A, D>
where
    A: Element + crate::element::SimdElement,
    S: Storage<Elem = A>,
    D: Dimension,
    F: Fn(A) -> A + Copy + Send + Sync,
{
    let len = input.len();
    let is_contiguous = input.is_f_contiguous();
    let alignment_ok = input.is_aligned();

    let (path, guard) = select_exec_path(len, is_contiguous, alignment_ok);
    match path {
        ExecPath::Serial => apply_unary_map(input, op),
        ExecPath::Simd => {
            try_simd_unary_via_slice(input, op_tag).unwrap_or_else(|| apply_unary_map(input, op))
        },
        ExecPath::Parallel => {
            #[cfg(feature = "parallel")]
            {
                let strat = ParallelExecStrategy::auto();
                let g = guard.expect("ExecPath::Parallel must carry a ParallelGuard");
                crate::parallel::unary::par_map(input, &strat, g, |x| op(*x))
            }
            #[cfg(not(feature = "parallel"))]
            {
                let _ = guard;
                apply_unary_map(input, op)
            }
        },
    }
}

/// Attempt a slice-based SIMD unary kernel. Returns `None` if `op_tag` is
/// `None`, the kernel returned false, or the input cannot be viewed as
/// `&[A]` (non-contiguous — defense-in-depth).
#[cfg(feature = "simd")]
fn try_simd_unary_via_slice<A, S, D>(
    input: &TensorBase<S, D>,
    op_tag: Option<crate::simd::UnaryOp>,
) -> Option<Tensor<A, D>>
where
    A: Element + crate::element::SimdElement,
    S: Storage<Elem = A>,
    D: Dimension,
{
    let tag = op_tag?;
    let src: &[A] = input.as_slice()?;
    let mut result = Tensor::<A, D>::zeros(input.raw_dim()).expect("input dimension must be valid");
    let dst: &mut [A] = result.as_mut_slice()?;
    if crate::simd::dispatch_vector_unary_op(tag, src, dst) {
        Some(result)
    } else {
        None
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::Ix1;
    use crate::tensor::Tensor;

    /// `abs` on an `i32` tensor returns the non-negative magnitude of each
    /// element (`abs(-3) == 3`, `abs(0) == 0`, `abs(5) == 5`).
    #[test]
    fn test_abs() {
        // abs(-3) = 3 for i32; abs(-2.5) = 2.5 for f64.
        let t =
            Tensor::<i32, Ix1>::from_shape_vec([3], vec![-3, 0, 5]).expect("valid tensor shape");
        let r = t.abs();
        assert_eq!(*r.get(&[0]).expect("valid index"), 3);
        assert_eq!(*r.get(&[1]).expect("valid index"), 0);
        assert_eq!(*r.get(&[2]).expect("valid index"), 5);
    }

    /// `neg` on an `f64` tensor flips the sign of each element; `-0.0`
    /// compares equal to `0.0` even though the bit pattern differs.
    #[test]
    fn test_neg() {
        let t = Tensor::<f64, Ix1>::from_shape_vec([3], vec![1.0, -2.0, 0.0])
            .expect("valid tensor shape");
        let r = t.neg();
        assert_eq!(*r.get(&[0]).expect("valid index"), -1.0);
        assert_eq!(*r.get(&[1]).expect("valid index"), 2.0);
        // -0.0 in IEEE 754 is distinct from +0.0 but compares equal.
        assert_eq!(*r.get(&[2]).expect("valid index"), 0.0);
    }

    /// `signum` on an `i32` tensor returns `-1` for negatives, `0` for
    /// zero, and `1` for positives.
    #[test]
    fn test_signum() {
        // i32: -1 / 0 / 1; f64 follows IEEE 754 signum (NaN→NaN, ±0.0→±1.0).
        let t =
            Tensor::<i32, Ix1>::from_shape_vec([3], vec![-7, 0, 4]).expect("valid tensor shape");
        let r = t.signum();
        assert_eq!(*r.get(&[0]).expect("valid index"), -1);
        assert_eq!(*r.get(&[1]).expect("valid index"), 0);
        assert_eq!(*r.get(&[2]).expect("valid index"), 1);
    }

    /// `square` on an `i32` tensor containing `i32::MAX` panics because
    /// `i32::MAX * i32::MAX` overflows the checked multiplication.
    #[test]
    fn test_square_checked_overflow() {
        // i32::MAX squared overflows i32 — square() must panic via
        // UnaryArith::square_step for i32.
        let t =
            Tensor::<i32, Ix1>::from_shape_vec([1], vec![i32::MAX]).expect("valid tensor shape");
        let result = std::panic::catch_unwind(|| t.square());
        assert!(result.is_err(), "i32::MAX squared must panic on overflow");
    }

    // ── Math function tests ──

    /// `sin` on `[0.0, π/2]` returns `[0.0, 1.0]` within a 1e-10 tolerance.
    #[test]
    fn test_sin() {
        let t = Tensor::<f64, Ix1>::from_shape_vec([2], vec![0.0, std::f64::consts::FRAC_PI_2])
            .expect("valid tensor shape");
        let r = t.sin();
        assert!((*r.get(&[0]).expect("valid index") - 0.0).abs() < 1e-10);
        assert!((*r.get(&[1]).expect("valid index") - 1.0).abs() < 1e-10);
    }

    /// `sqrt` returns 2.0 for `4.0` and NaN for `-1.0` per IEEE 754.
    #[test]
    fn test_sqrt() {
        let t =
            Tensor::<f64, Ix1>::from_shape_vec([2], vec![4.0, -1.0]).expect("valid tensor shape");
        let r = t.sqrt();
        assert!((*r.get(&[0]).expect("valid index") - 2.0).abs() < 1e-10);
        assert!(r.get(&[1]).expect("valid index").is_nan());
    }

    /// `ln` then `exp` recovers the original value (`exp(ln(2.0)) ≈ 2.0`)
    /// within a 1e-10 tolerance.
    #[test]
    fn test_exp_ln_roundtrip() {
        let t = Tensor::<f64, Ix1>::from_shape_vec([1], vec![2.0]).expect("valid tensor shape");
        let r = t.ln().exp();
        assert!((*r.get(&[0]).expect("valid index") - 2.0).abs() < 1e-10);
    }

    /// `floor(1.7) == 1.0` and `ceil(1.3) == 2.0`.
    #[test]
    fn test_floor_ceil() {
        let t =
            Tensor::<f64, Ix1>::from_shape_vec([2], vec![1.7, 1.3]).expect("valid tensor shape");
        let f = t.floor();
        let c = t.ceil();
        assert_eq!(*f.get(&[0]).expect("valid index"), 1.0);
        assert_eq!(*c.get(&[1]).expect("valid index"), 2.0);
    }

    // ── Complex op tests ──

    /// `modulus` of `3 + 4i` returns `5.0` (within a 1e-10 tolerance) and
    /// produces a real tensor of the same shape.
    #[test]
    fn test_modulus() {
        let t = Tensor::<Complex<f64>, Ix1>::from_shape_vec([1], vec![Complex::new(3.0_f64, 4.0)])
            .expect("valid tensor shape");
        let r = t.modulus();
        assert!(
            (*r.get(&[0]).expect("valid index") - 5.0).abs() < 1e-10,
            "modulus(3+4i) should be 5.0, got {}",
            r.get(&[0]).expect("valid index")
        );
    }

    /// `conjugate` of `1 + 2i` returns `1 - 2i` (real part unchanged,
    /// imaginary part negated).
    #[test]
    fn test_conjugate() {
        let t = Tensor::<Complex<f64>, Ix1>::from_shape_vec([1], vec![Complex::new(1.0_f64, 2.0)])
            .expect("valid tensor shape");
        let r = t.conjugate();
        assert_eq!(r.get(&[0]).expect("valid index").re(), 1.0);
        assert_eq!(r.get(&[0]).expect("valid index").im(), -2.0);
    }

    // ── Logical NOT test ──

    /// `not` on a bool tensor flips each element (`true → false`,
    /// `false → true`) and preserves the shape.
    #[test]
    fn test_not_bool() {
        let t = Tensor::<bool, Ix1>::from_shape_vec([3], vec![true, false, true])
            .expect("valid tensor shape");
        let result = t.not();
        assert!(!(*result.get(&[0]).expect("valid index")));
        assert!(*result.get(&[1]).expect("valid index"));
        assert!(!(*result.get(&[2]).expect("valid index")));
    }
}
