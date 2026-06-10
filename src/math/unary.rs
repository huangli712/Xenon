//! Unary element-wise operations:
//!
//! - abs, neg, signum, square,
//! - math functions sin/sqrt/exp/ln/floor/ceil,
//! - complex conjugate/modulus, logical not.

use core::any::TypeId;

use crate::dispatch::{ExecPath, select_exec_path};

use crate::complex::{Complex, ComplexFloat};
use crate::dimension::Dimension;
use crate::element::{Element, Numeric, RealScalar, ComplexScalar};
use crate::element::{CheckedNeg, OrderedCompareElement, SimdElement};
use crate::storage::Storage;
use crate::tensor::{Tensor, TensorBase};

#[cfg(feature = "parallel")]
use crate::dispatch::ParallelExecStrategy;

#[cfg(feature = "parallel")]
use crate::parallel::unary::par_map;

#[cfg(feature = "simd")]
use crate::simd::{UnaryOp, dispatch_vector_unary_op};

/// Selector for the dispatch-routed unary arithmetic ops
/// (`abs` / `neg` / `square` / `signum`). Defined unconditionally so the
/// dispatch layer can name an operation without depending on the `simd`
/// feature.
#[derive(Copy, Clone)]
enum UnaryArithOp {
    Abs,
    Neg,
    Square,
    Signum,
}

/// Maps the feature-independent [`UnaryArithOp`] selector to the
/// SIMD-internal [`UnaryOp`] tag. Only `Neg` has a SIMD kernel today; the
/// rest fall through to scalar. Only compiled when `simd` is enabled.
#[cfg(feature = "simd")]
#[inline]
fn simd_unary_op_tag(op: UnaryArithOp) -> Option<UnaryOp> {
    match op {
        UnaryArithOp::Neg => Some(UnaryOp::Neg),
        UnaryArithOp::Abs | UnaryArithOp::Square | UnaryArithOp::Signum => None,
    }
}

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
trait UnaryArith: Numeric + SimdElement + 'static {
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

/// Unary step for `Complex<f32>`: standard `Neg` and `Mul` operators; NaN /
/// Inf propagate per IEEE 754 on the real and imaginary components.
impl UnaryArith for Complex<f32> {
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
impl UnaryArith for Complex<f64> {
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
    Numeric + OrderedCompareElement + SimdElement + 'static
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

// ----------------------------------------------------------------------------
// Public unary methods on TensorBase
// ----------------------------------------------------------------------------

/// Tensor methods for `abs` and `signum`, available on ordered element types
/// (`i32` / `i64` / `f32` / `f64`).
// abs / signum: ordered types (i32/i64/f32/f64).
#[expect(private_bounds)]
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
        apply_unary_with_dispatch(
            self,
            |x, idx, shape| <A as OrderedUnaryArith>::abs_step_with_ctx(
                x,
                idx,
                shape
            ),
            UnaryArithOp::Abs,
        )
    }

    /// Element-wise signum. Integers return sign-based `-1` / `0` / `1`.
    /// Floats follow IEEE 754 semantics via `RealScalar::signum` (handles
    /// signed zeros and NaN).
    pub fn signum(&self) -> Tensor<A, D> {
        apply_unary_with_dispatch(
            self,
            |x, idx, shape| <A as OrderedUnaryArith>::signum_step_with_ctx(
                x,
                idx,
                shape
            ),
            UnaryArithOp::Signum,
        )
    }
}

/// Tensor methods for `neg` and `square`, available on every `Numeric`
/// element type (including complex).
// neg / square: all Numeric (including Complex).
#[expect(private_bounds)]
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
        apply_unary_with_dispatch(
            self,
            |x, idx, shape| <A as UnaryArith>::neg_step_with_ctx(
                x,
                idx,
                shape
            ),
            UnaryArithOp::Neg,
        )
    }

    /// Element-wise square (`x * x`). The integer path panics on overflow;
    /// the float and complex paths use `*` with IEEE 754 propagation.
    pub fn square(&self) -> Tensor<A, D> {
        apply_unary_with_dispatch(
            self,
            |x, idx, shape| <A as UnaryArith>::square_step_with_ctx(
                x,
                idx,
                shape
            ),
            UnaryArithOp::Square,
        )
    }
}

// ----------------------------------------------------------------------------
// Math functions for RealScalar (f32, f64)
// ----------------------------------------------------------------------------

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
    /// Routes through `apply_unary_with_real_dispatch` so that large tensors can
    /// benefit from parallel acceleration. No SIMD kernel for `sin` is
    /// available yet, so the SIMD path falls back to scalar.
    pub fn sin(&self) -> Tensor<A, D> {
        apply_unary_with_real_dispatch(self, |x| x.sin())
    }

    /// Element-wise square root. `sqrt(-1.0) = NaN` per IEEE 754.
    pub fn sqrt(&self) -> Tensor<A, D> {
        apply_unary_with_real_dispatch(self, |x| x.sqrt())
    }

    /// Element-wise exponential. `exp(Inf) = Inf`, `exp(-Inf) = 0.0`.
    pub fn exp(&self) -> Tensor<A, D> {
        apply_unary_with_real_dispatch(self, |x| x.exp())
    }

    /// Element-wise natural logarithm. `ln(0.0) = -Inf`, `ln(-1.0) = NaN`.
    pub fn ln(&self) -> Tensor<A, D> {
        apply_unary_with_real_dispatch(self, |x| x.ln())
    }

    /// Element-wise floor. Exact (no tolerance).
    pub fn floor(&self) -> Tensor<A, D> {
        apply_unary_with_real_dispatch(self, |x| x.floor())
    }

    /// Element-wise ceil. Exact (no tolerance).
    pub fn ceil(&self) -> Tensor<A, D> {
        apply_unary_with_real_dispatch(self, |x| x.ceil())
    }
}

// ----------------------------------------------------------------------------
// Complex ops — modulus (type-changing Complex<T> → T)
// ----------------------------------------------------------------------------

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
        apply_unary_serial(self, |c| c.norm())
    }
}

// ----------------------------------------------------------------------------
// Complex ops — conjugate (single generic impl)
// ----------------------------------------------------------------------------

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
    /// Routes through `apply_unary_with_real_dispatch` so that large tensors can
    /// benefit from parallel acceleration. No SIMD kernel for `conjugate`
    /// is available yet (it would require cross-lane operations), so the
    /// SIMD path falls back to scalar.
    pub fn conjugate(&self) -> Tensor<Complex<T>, D> {
        apply_unary_with_real_dispatch(self, <Complex<T> as Numeric>::conjugate)
    }
}

// ----------------------------------------------------------------------------
// Logical NOT for bool tensors with dispatch wiring
// ----------------------------------------------------------------------------

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
            // to the scalar `apply_unary_serial` baseline. Parallel routes
            // through `par_map`.
            ExecPath::Serial | ExecPath::Simd => {
                let _ = guard;
                apply_unary_serial(self, |x| !x)
            },
            ExecPath::Parallel => {
                #[cfg(feature = "parallel")]
                {
                    let strat = ParallelExecStrategy::auto();
                    let g = guard.expect("ExecPath::Parallel must carry a ParallelGuard");
                    par_map(self, &strat, g, |x| !*x)
                }
                #[cfg(not(feature = "parallel"))]
                {
                    let _ = guard;
                    apply_unary_serial(self, |x| !x)
                }
            },
        }
    }
}

// ----------------------------------------------------------------------------
// Shared traversal helpers
// ----------------------------------------------------------------------------

/// Element-wise unary traversal.
///
/// Maps each input element through `f`, producing an output tensor of the
/// same shape. The output element type `O` may differ from the input type
/// `A`, covering both same-type ops (`A → A`, e.g. `neg` / `square`) and
/// type-changing ops (`Complex<T> → T`, e.g. `modulus`).
#[inline]
fn apply_unary_serial<A, O, S, D, F>(input: &TensorBase<S, D>, mut f: F) -> Tensor<O, D>
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
/// of `apply_unary_serial` that propagates `(idx, &shape)` into the kernel
/// closure.
///
/// Required by the integer monomorphizations of `abs` / `neg` / `square`
/// so that panic messages can embed `operation`, `type`, `trigger`,
/// `element_index`, and (when applicable) `shape`. Non-integer
/// monomorphizations have no panic path; the `idx` / `shape` parameters
/// are zero-cost dropped after inlining.
#[inline]
fn apply_unary_checked<A, S, D, F>(input: &TensorBase<S, D>, mut f: F) -> Tensor<A, D>
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

// ----------------------------------------------------------------------------
// Dispatch-aware unary helpers (serial/SIMD/parallel routing)
// ----------------------------------------------------------------------------

/// Dispatch-aware unary helper for methods bounded by `A: RealScalar` or
/// `A: ComplexFloat`-derived element traits that do NOT include
/// `SimdElement` in their supertrait set.
///
/// **Why this exists**: The user-facing math API (`sin`, `sqrt`, `exp`,
/// `ln`, `floor`, `ceil`, `conjugate`) is bounded by the public
/// `RealScalar` and `ComplexFloat` traits. These traits intentionally do
/// NOT include the `pub(crate) SimdElement` trait as a supertrait, because
/// SIMD types must not appear in the public API surface (`pub(crate)`
/// only). Therefore the SIMD kernel attempt (which requires
/// `A: SimdElement`) cannot be called from these methods, so there is no
/// SIMD path here: the SIMD execution path collapses to the scalar
/// baseline (`apply_unary_serial`).
///
/// **Real acceleration today**: the Parallel path via `par_map` when the
/// `parallel` feature is enabled and `select_exec_path` returns
/// `ExecPath::Parallel` — reachable independent of the `simd` feature.
fn apply_unary_with_real_dispatch<A, S, D, F>(input: &TensorBase<S, D>, op: F) -> Tensor<A, D>
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
        // No SIMD kernel for these ops; Serial and Simd both use the
        // scalar baseline.
        ExecPath::Serial | ExecPath::Simd => {
            let _ = guard;
            apply_unary_serial(input, op)
        },
        ExecPath::Parallel => {
            #[cfg(feature = "parallel")]
            {
                let strat = ParallelExecStrategy::auto();
                let g = guard.expect("ExecPath::Parallel must carry a ParallelGuard");
                par_map(input, &strat, g, |x| op(*x))
            }
            #[cfg(not(feature = "parallel"))]
            {
                let _ = guard;
                apply_unary_serial(input, op)
            }
        },
    }
}

/// Unified unary arithmetic dispatch for `abs` / `neg` / `square` /
/// `signum`.
///
/// - Integer types (`i32`, `i64`): serial checked traversal via
///   `apply_unary_checked`, carrying the per-element index and shape so
///   overflow panics keep their diagnostic context.
/// - Float / complex types: routed through the Serial / SIMD / Parallel
///   execution path chosen by `select_exec_path`.
///
/// NOT gated on the `simd` feature — only the inner SIMD kernel attempt is
/// conditional — so the Parallel path stays reachable whenever `parallel`
/// is enabled, independent of `simd`.
///
/// The `step` closure carries `(idx, &shape)` for the integer path; the
/// float path adapts it to a context-free kernel via `|x| step(x, 0, &[])`,
/// which is zero-cost since float / complex impls ignore those parameters.
fn apply_unary_with_dispatch<A, S, D, F>(
    input: &TensorBase<S, D>,
    step: F,
    op: UnaryArithOp,
) -> Tensor<A, D>
where
    A: Element + SimdElement + 'static,
    S: Storage<Elem = A>,
    D: Dimension,
    F: Fn(A, usize, &[usize]) -> A + Copy + Send + Sync,
{
    // Integer carve-out: i32 / i64 keep the per-element panic diagnostic
    // context, so they take the serial checked path.
    if TypeId::of::<A>() == TypeId::of::<i32>() || TypeId::of::<A>() == TypeId::of::<i64>() {
        return apply_unary_checked(input, step);
    }
    // Float / complex: drop the index/shape context and route through the
    // Serial / SIMD / Parallel execution path chosen by `select_exec_path`.
    let scalar_op = move |x| step(x, 0, &[]);
    let len = input.len();
    let is_contiguous = input.is_f_contiguous();
    let alignment_ok = input.is_aligned();
    let (path, guard) = select_exec_path(len, is_contiguous, alignment_ok);
    match path {
        ExecPath::Serial => apply_unary_serial(input, scalar_op),
        ExecPath::Simd => {
            #[cfg(feature = "simd")]
            {
                try_simd_unary(input, simd_unary_op_tag(op))
                    .unwrap_or_else(|| apply_unary_serial(input, scalar_op))
            }
            #[cfg(not(feature = "simd"))]
            {
                let _ = op;
                apply_unary_serial(input, scalar_op)
            }
        },
        ExecPath::Parallel => {
            #[cfg(feature = "parallel")]
            {
                let strat = ParallelExecStrategy::auto();
                let g = guard.expect("ExecPath::Parallel must carry a ParallelGuard");
                par_map(input, &strat, g, |x| scalar_op(*x))
            }
            #[cfg(not(feature = "parallel"))]
            {
                let _ = guard;
                apply_unary_serial(input, scalar_op)
            }
        },
    }
}

/// Attempt a slice-based SIMD unary kernel. Returns `None` if `op_tag` is
/// `None`, the kernel returned false, or the input cannot be viewed as
/// `&[A]` (non-contiguous — defense-in-depth).
#[cfg(feature = "simd")]
fn try_simd_unary<A, S, D>(
    input: &TensorBase<S, D>,
    op_tag: Option<UnaryOp>,
) -> Option<Tensor<A, D>>
where
    A: Element + SimdElement,
    S: Storage<Elem = A>,
    D: Dimension,
{
    let tag = op_tag?;
    let src: &[A] = input.as_slice()?;
    let mut result = Tensor::<A, D>::zeros(input.raw_dim()).expect("input dimension must be valid");
    let dst: &mut [A] = result.as_mut_slice()?;
    if dispatch_vector_unary_op(tag, src, dst) {
        Some(result)
    } else {
        None
    }
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::Ix1;
    use crate::tensor::Tensor;
    use std::panic::catch_unwind;

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
        let result = catch_unwind(|| t.square());
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

    // ── Integer overflow panic diagnostics ──

    /// `neg(i32::MIN)` overflows the checked negation and panics.
    #[test]
    fn test_neg_i32_min_overflow_panic() {
        let t = Tensor::<i32, Ix1>::from_shape_vec([1], vec![i32::MIN])
            .expect("valid tensor shape");
        let result = catch_unwind(|| t.neg());
        assert!(result.is_err(), "neg(i32::MIN) must panic on overflow");
    }

    /// `abs(i32::MIN)` overflows the checked negation and panics.
    #[test]
    fn test_abs_i32_min_overflow_panic() {
        let t = Tensor::<i32, Ix1>::from_shape_vec([1], vec![i32::MIN])
            .expect("valid tensor shape");
        let result = catch_unwind(|| t.abs());
        assert!(result.is_err(), "abs(i32::MIN) must panic on overflow");
    }

    // ── Float abs / signum ──

    /// `abs` on an `f64` tensor returns the IEEE 754 magnitude.
    #[test]
    fn test_abs_f64() {
        let t = Tensor::<f64, Ix1>::from_shape_vec([3], vec![-3.5, 0.0, 2.5])
            .expect("valid tensor shape");
        let r = t.abs();
        assert!((*r.get(&[0]).expect("valid index") - 3.5).abs() < 1e-10);
        assert!((*r.get(&[1]).expect("valid index") - 0.0).abs() < 1e-10);
        assert!((*r.get(&[2]).expect("valid index") - 2.5).abs() < 1e-10);
    }

    /// `signum` on an `f64` tensor returns `-1.0` / `1.0` per IEEE 754
    /// (signed-zero / NaN handling delegated to `RealScalar::signum`).
    #[test]
    fn test_signum_f64() {
        let t = Tensor::<f64, Ix1>::from_shape_vec([3], vec![-7.0, 4.0, -0.5])
            .expect("valid tensor shape");
        let r = t.signum();
        assert!((*r.get(&[0]).expect("valid index") + 1.0).abs() < 1e-10);
        assert!((*r.get(&[1]).expect("valid index") - 1.0).abs() < 1e-10);
        assert!((*r.get(&[2]).expect("valid index") + 1.0).abs() < 1e-10);
    }

    // ── Complex neg / square (independent dispatch branch) ──

    /// `neg` on `Complex<f64>` negates both components: `-(1+2i) = -1-2i`.
    #[test]
    fn test_neg_complex() {
        let t = Tensor::<Complex<f64>, Ix1>::from_shape_vec([1], vec![Complex::new(1.0, 2.0)])
            .expect("valid tensor shape");
        let r = t.neg();
        let v = r.get(&[0]).expect("valid index");
        assert!((v.re() + 1.0).abs() < 1e-10);
        assert!((v.im() + 2.0).abs() < 1e-10);
    }

    /// `square` on `Complex<f64>`: `(1+2i)^2 = -3+4i`.
    #[test]
    fn test_square_complex() {
        let t = Tensor::<Complex<f64>, Ix1>::from_shape_vec([1], vec![Complex::new(1.0, 2.0)])
            .expect("valid tensor shape");
        let r = t.square();
        let v = r.get(&[0]).expect("valid index");
        assert!((v.re() + 3.0).abs() < 1e-10);
        assert!((v.im() - 4.0).abs() < 1e-10);
    }

    // ── Parallel-path cross-consistency (parallel feature only) ──

    /// Float `abs`/`neg`/`square`/`sin` produce identical results on the
    /// Parallel path (forced via a threshold of 1) as on the Serial path
    /// (parallel disabled via the 0 sentinel). Guards the inlined
    /// `ExecPath::Parallel` arm of both `apply_unary_with_dispatch`
    /// (abs/neg/square) and `apply_unary_with_real_dispatch` (sin).
    #[cfg(feature = "parallel")]
    #[test]
    fn test_unary_parallel_matches_serial_f64() {
        use crate::dispatch::ThresholdTestGuard;
        use crate::dispatch::set_parallel_threshold;

        let t = Tensor::<f64, Ix1>::from_shape_vec([128], (0..128).map(|x| x as f64 - 64.0).collect())
            .expect("valid tensor shape");

        let _guard = ThresholdTestGuard::new();
        set_parallel_threshold(0);
        let abs_serial = t.abs();
        let neg_serial = t.neg();
        let square_serial = t.square();
        let sin_serial = t.sin();
        set_parallel_threshold(1);
        let abs_par = t.abs();
        let neg_par = t.neg();
        let square_par = t.square();
        let sin_par = t.sin();

        for i in 0..128 {
            let ix = [i];
            assert_eq!(
                abs_par.get(&ix).expect("valid index"),
                abs_serial.get(&ix).expect("valid index"),
                "abs parallel/serial mismatch at {i}"
            );
            assert_eq!(
                neg_par.get(&ix).expect("valid index"),
                neg_serial.get(&ix).expect("valid index"),
                "neg parallel/serial mismatch at {i}"
            );
            assert_eq!(
                square_par.get(&ix).expect("valid index"),
                square_serial.get(&ix).expect("valid index"),
                "square parallel/serial mismatch at {i}"
            );
            assert_eq!(
                sin_par.get(&ix).expect("valid index"),
                sin_serial.get(&ix).expect("valid index"),
                "sin parallel/serial mismatch at {i}"
            );
        }
    }

    /// Integer `neg` keeps exact byte-level equality between the Parallel path
    /// (forced) and the Serial checked path.
    #[cfg(feature = "parallel")]
    #[test]
    fn test_neg_parallel_matches_serial_i32() {
        use crate::dispatch::ThresholdTestGuard;
        use crate::dispatch::set_parallel_threshold;

        let t = Tensor::<i32, Ix1>::from_shape_vec([128], (0..128).map(|x| x - 64).collect())
            .expect("valid tensor shape");

        let _guard = ThresholdTestGuard::new();
        set_parallel_threshold(0);
        let serial = t.neg();
        set_parallel_threshold(1);
        let parallel = t.neg();
        for i in 0..128 {
            assert_eq!(
                parallel.get(&[i]).expect("valid index"),
                serial.get(&[i]).expect("valid index"),
                "i32 neg parallel/serial mismatch at {i}"
            );
        }
    }
}
