//! Unary element-wise operations: abs, neg, signum, square (W16T3),
//! math functions sin/sqrt/exp/ln/floor/ceil (W16T4),
//! complex conjugate/modulus (W16T5), logical not (W16T7).

use crate::complex::{Complex, ComplexFloat};
use crate::dimension::Dimension;
#[cfg(feature = "parallel")]
use crate::dispatch::ParallelExecStrategy;
use crate::dispatch::{ExecPath, select_exec_path};
use crate::element::{CheckedNeg, ComplexScalar, Element, Numeric, OrderedCompareElement, RealScalar};

use crate::storage::Storage;
use crate::tensor::{Tensor, TensorBase};

// ============================================================================
// Private per-type dispatch traits
// ============================================================================

/// Per-type unary step for abs / neg / square.
///
/// - Integer types (`i32`, `i64`): checked arithmetic; overflow / `i{32,64}::MIN`
///   等不可表示情形 panic，符合 11-math §5.4 line 257-259 的整数规约。
/// - Floating types (`f32`, `f64`): delegate to `RealScalar` intrinsics.
/// - Complex types (`Complex<f32>`, `Complex<f64>`): use `Neg` /
///   `Mul` 标准运算符（IEEE 754 传播 NaN / Inf）；`abs` / `signum` 不覆盖
///   复数（trait bound `OrderedCompareElement` 编译期排除）。
///
/// Sealed via `Numeric: Sealed` (03-element §5.2). Consumers cannot add
/// new impls; all six concrete `Numeric` types provide their own impl in
/// this file.
///
/// `*_step_with_ctx` variants propagate `element_index` + `shape` into
/// integer panic text per 11-math §10 line 785-790. Default implementations
/// forward to context-free `*_step` (zero cost on non-integer paths after
/// `#[inline]`); integer impls override these to embed the full diagnostic
/// fields.
/// `SimdElement` supertrait — see `BinaryArith` for the rationale. Since
/// `SimdElement` lives in `crate::element` (ungated), the bound holds in
/// every feature configuration. All six `UnaryArith` impls
/// (i32/i64/f32/f64/Complex<f32>/Complex<f64>) already implement
/// `SimdElement` per W14T1, so adding the supertrait does not narrow
/// the sealed set.
trait UnaryArith: Numeric + crate::element::SimdElement + 'static {
    /// Element-wise negation; integer path panics on `MIN`.
    fn neg_step(x: Self) -> Self;
    /// Element-wise square `x * x`; integer path panics on overflow.
    fn square_step(x: Self) -> Self;

    /// Context-aware variant used by integer monomorphizations to embed
    /// `element_index` + `shape` into panic text per 11-math §10. Default
    /// impl forwards to the context-free `*_step` (non-integer types have
    /// no panic path, so the index / shape are simply dropped).
    #[inline]
    fn neg_step_with_ctx(x: Self, _idx: usize, _shape: &[usize]) -> Self {
        Self::neg_step(x)
    }
    #[inline]
    fn square_step_with_ctx(x: Self, _idx: usize, _shape: &[usize]) -> Self {
        Self::square_step(x)
    }
}

/// Per-type unary step for abs / signum, restricted to ordered types.
///
/// Trait bound `Numeric + OrderedCompareElement` 在 03-element §5.5 sealed
/// 到 `i32` / `i64` / `f32` / `f64`；复数编译时被排除。
///
/// `*_step_with_ctx` variants follow the same contract as `UnaryArith`:
/// default implementations forward to `*_step`; integer impls override to
/// embed `element_index` + `shape` into panic text per 11-math §10.
/// `SimdElement` supertrait. Since `SimdElement` lives in `crate::element`
/// (ungated), the bound holds in every feature configuration. All four
/// `OrderedUnaryArith` impls (i32/i64/f32/f64) already implement
/// `SimdElement` per W14T1, so the supertrait does not narrow the
/// sealed set.
trait OrderedUnaryArith: Numeric + OrderedCompareElement + crate::element::SimdElement + 'static {
    /// Element-wise absolute value; integer path panics on `MIN`.
    fn abs_step(x: Self) -> Self;
    /// Element-wise signum. Integers: `-1` / `0` / `1`.
    /// Floats: delegates to `RealScalar::signum` (IEEE 754).
    fn signum_step(x: Self) -> Self;

    #[inline]
    fn abs_step_with_ctx(x: Self, _idx: usize, _shape: &[usize]) -> Self {
        Self::abs_step(x)
    }
    #[inline]
    fn signum_step_with_ctx(x: Self, _idx: usize, _shape: &[usize]) -> Self {
        Self::signum_step(x)
    }
}

// ============================================================================
// Integer impls (checked arithmetic)
// ============================================================================

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

// ============================================================================
// Float impls (IEEE 754)
// ============================================================================

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
// Complex impls (only neg / square; abs / signum excluded at compile time
// via `OrderedCompareElement` — Complex does NOT implement it per
// 03-element §5.5).
// ============================================================================

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

// ============================================================================
// Shared traversal helpers (merged from helpers.rs)
// ============================================================================

/// Same-type unary traversal — per 11-math §6.1 lines 537-543.
///
/// Output element type equals input element type. Type-changing
/// traversal (`Complex<T> → T`) is handled by `apply_complex_to_real`.
#[inline]
fn apply_unary<A, S, D, F>(input: &TensorBase<S, D>, mut f: F) -> Tensor<A, D>
where
    A: Element,
    S: Storage<Elem = A>,
    D: Dimension,
    F: FnMut(A) -> A,
{
    let mut result = Tensor::<A, D>::zeros(input.raw_dim())
        .expect("input dimension must be valid since input tensor exists");
    for (dst, src) in result.iter_mut().zip(input.iter()) {
        *dst = f(*src);
    }
    result
}

/// Type-changing traversal for `Complex<T> → T` — per 11-math §6.1 line 546.
///
/// Used by `modulus()` (W16T5). Input is a complex tensor; output is a
/// real tensor of the same shape.
#[inline]
fn apply_complex_to_real<A, S, D, F>(
    input: &TensorBase<S, D>,
    mut f: F,
) -> Tensor<<A as ComplexScalar>::Real, D>
where
    A: ComplexScalar,
    S: Storage<Elem = A>,
    D: Dimension,
    F: FnMut(A) -> <A as ComplexScalar>::Real,
{
    let mut result = Tensor::<<A as ComplexScalar>::Real, D>::zeros(input.raw_dim())
        .expect("input dimension must be valid since input tensor exists");
    for (dst, src) in result.iter_mut().zip(input.iter()) {
        *dst = f(*src);
    }
    result
}

/// Same-type unary traversal with element-index and shape context — variant
/// of `apply_unary` that propagates `(idx, &shape)` into the kernel closure.
///
/// Required by W16T3 integer monomorphizations of `abs` / `neg` / `square`
/// so that panic messages can embed `element_index` + `shape` per 11-math
/// §10 line 785–790 ("panic 信息至少包含 `operation`、`type`、`trigger`、
/// `element_index`，并在适用时附带 `shape`"). Non-integer monomorphizations
/// have no panic path; the `idx` / `shape` parameters are zero-cost dropped
/// after inlining.
#[inline]
fn apply_unary_indexed<A, S, D, F>(
    input: &TensorBase<S, D>,
    mut f: F,
) -> Tensor<A, D>
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
// Public unary methods on TensorBase
// ============================================================================

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
    /// Element-wise absolute value. Integer path panics on `MIN`;
    /// float path follows IEEE 754 (`abs(NaN) = NaN`).
    ///
    /// W16T11 Step 2: integer types retain `apply_unary_indexed` for §10
    /// `element_index` diagnostics. Float types route through
    /// `apply_unary_with_dispatch` (op_tag=None — W14 has no abs SIMD
    /// kernel yet, so Serial/Simd both fall to scalar; Parallel deferred).
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

    /// Element-wise signum. Integers: sign-based `-1` / `0` / `1`.
    /// Floats: IEEE 754 semantics via `RealScalar::signum` (handles
    /// ±0.0 and NaN per 11-math §5.4).
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
    /// Element-wise negation. Integer path panics on `MIN`; float path
    /// uses `-x` (IEEE 754); complex path uses `Neg` operator.
    ///
    /// W16T11 Step 2: integer types retain `apply_unary_indexed`; float
    /// types route through `apply_unary_with_dispatch` with op_tag=Some(Neg)
    /// — W14's only unary SIMD kernel per W14T1 Step 4.
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

    /// Element-wise square: `x * x`. Integer path panics on overflow;
    /// float / complex path uses `*` (IEEE 754 propagation).
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
// W16T4: Math functions for RealScalar (f32, f64)
// ============================================================================

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: RealScalar,
{
    /// Element-wise sine. IEEE 754 NaN propagates: `sin(NaN) = NaN`.
    ///
    /// W16T11 Step 2: routes through `apply_unary_real_dispatch` for
    /// parallel acceleration on large tensors. W14 has no SIMD kernel
    /// for `sin` yet, so the SIMD path falls back to scalar.
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
// W16T5: Complex ops — modulus (type-changing Complex<T> → T)
// ============================================================================

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
        apply_complex_to_real(self, |c| c.norm())
    }
}

// ============================================================================
// W16T5: Complex ops — conjugate (single generic impl)
// ============================================================================

impl<S, D, T> TensorBase<S, D>
where
    S: Storage<Elem = Complex<T>>,
    D: Dimension,
    T: ComplexFloat,
    Complex<T>: Element + Numeric,
{
    /// Element-wise complex conjugate: `(a + bi) → (a - bi)`.
    /// Delegates to `Numeric::conjugate` per 03-element §5.2.
    ///
    /// W16T11 Step 2: routes through `apply_unary_real_dispatch` for
    /// parallel acceleration on large tensors. W14 has no SIMD kernel
    /// for `conjugate` yet (requires cross-lane operations), so the
    /// SIMD path falls back to scalar.
    pub fn conjugate(&self) -> Tensor<Complex<T>, D> {
        apply_unary_real_dispatch(self, <Complex<T> as Numeric>::conjugate)
    }
}

// ============================================================================
// W16T7 + W16T11: Logical NOT for bool tensors with dispatch wiring
// ============================================================================

impl<S, D> TensorBase<S, D>
where
    S: Storage<Elem = bool>,
    D: Dimension,
{
    /// Element-wise logical NOT. Returns a bool tensor of the same shape.
    ///
    /// Dispatch wiring per W16T11 Step 10: `select_exec_path` routes
    /// between Serial and SIMD (both fall to scalar — W14 has no bool
    /// SIMD kernel), with Parallel path deferred to a future wave when
    /// the `par_map` API is finalized.
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
            // No bool SIMD kernel in W14; Serial and Simd fall through to
            // the inline scalar loop. Parallel routes through par_map (W15).
            ExecPath::Serial | ExecPath::Simd => {
                let _ = guard;
                let mut result =
                    Tensor::zeros(self.raw_dim()).expect("input dimension must be valid");
                for (dst, &src) in result.iter_mut().zip(self.iter()) {
                    *dst = !src;
                }
                result
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
                    let mut result =
                        Tensor::zeros(self.raw_dim()).expect("input dimension must be valid");
                    for (dst, &src) in result.iter_mut().zip(self.iter()) {
                        *dst = !src;
                    }
                    result
                }
            },
        }
    }
}

// ============================================================================
// W16T11: Dispatch-aware unary helper (serial/SIMD/parallel routing)
// ============================================================================

/// Dispatch-aware unary helper for methods bounded by `A: RealScalar` or
/// `A: ComplexFloat`-derived element traits that do NOT include
/// `SimdElement` in their supertrait set.
///
/// **Why this exists**: The user-facing math API (`sin`, `sqrt`, `exp`,
/// `ln`, `floor`, `ceil`, `conjugate`) is bounded by public `RealScalar`
/// (`03-element.md §5.3`) and `ComplexFloat` (`02-complex.md`) traits.
/// These traits intentionally do NOT include the `pub(crate) SimdElement`
/// trait as a supertrait, because `08-simd.md §5.1` mandates that SIMD
/// types must not appear in the public API surface (`pub(crate)` only).
/// Therefore `apply_unary_with_dispatch` (which requires
/// `A: SimdElement`) cannot be called from these methods.
///
/// **Coverage today**: W14T1 line 132 reserves `Sin`/`Sqrt`/`Exp`/`Ln`/
/// `Floor`/`Ceil`/`Conjugate` for future SIMD coverage. So the SIMD
/// branch of this helper always falls through to scalar.
///
/// **Real acceleration today**: Parallel path via
/// `crate::parallel::unary::par_map` (W15T3) when `parallel` feature is
/// enabled and `select_exec_path` returns `ExecPath::Parallel`.
///
/// **Future-proof**: When W14 extends coverage (e.g. adds
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
        // W14 has no SIMD kernel for these ops yet; Serial and Simd both
        // route to the scalar baseline.
        ExecPath::Serial | ExecPath::Simd => {
            let _ = guard;
            apply_unary(input, op)
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
                apply_unary(input, op)
            }
        },
    }
}

/// Dispatch-aware variant of `apply_unary` — routes between Serial, SIMD,
/// and Parallel paths per 11-math §5.2 / §6.3. The scalar baseline
/// `apply_unary` (W16T1 helpers) is the SIMD-fallback target.
///
/// `op_tag: Option<simd::UnaryOp>` encodes which SIMD kernel to attempt.
/// `None` → SIMD path is skipped, falling back to Serial/Parallel.
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
        ExecPath::Serial => apply_unary(input, op),
        ExecPath::Simd => {
            try_simd_unary_via_slice(input, op_tag).unwrap_or_else(|| apply_unary(input, op))
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
                apply_unary(input, op)
            }
        },
    }
}

/// Helper: attempt a slice-based SIMD unary kernel. Returns `None` if
/// op_tag is None, the kernel returned false, or the input cannot be
/// viewed as `&[A]` (non-contiguous — defense-in-depth).
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

    #[test]
    fn test_signum() {
        // i32: -1 / 0 / 1; f64: follows IEEE 754 signum (NaN→NaN, ±0.0→±1.0).
        let t =
            Tensor::<i32, Ix1>::from_shape_vec([3], vec![-7, 0, 4]).expect("valid tensor shape");
        let r = t.signum();
        assert_eq!(*r.get(&[0]).expect("valid index"), -1);
        assert_eq!(*r.get(&[1]).expect("valid index"), 0);
        assert_eq!(*r.get(&[2]).expect("valid index"), 1);
    }

    #[test]
    fn test_square_checked_overflow() {
        // i32::MAX squared overflows i32 — square() must panic via
        // UnaryArith::square_step for i32 (Step 2 per-type impl).
        let t =
            Tensor::<i32, Ix1>::from_shape_vec([1], vec![i32::MAX]).expect("valid tensor shape");
        let result = std::panic::catch_unwind(|| t.square());
        assert!(result.is_err(), "i32::MAX squared must panic on overflow");
    }

    // ── W16T4: Math function tests ──

    #[test]
    fn test_sin() {
        let t = Tensor::<f64, Ix1>::from_shape_vec([2], vec![0.0, std::f64::consts::FRAC_PI_2])
            .expect("valid tensor shape");
        let r = t.sin();
        assert!((*r.get(&[0]).expect("valid index") - 0.0).abs() < 1e-10);
        assert!((*r.get(&[1]).expect("valid index") - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sqrt() {
        let t =
            Tensor::<f64, Ix1>::from_shape_vec([2], vec![4.0, -1.0]).expect("valid tensor shape");
        let r = t.sqrt();
        assert!((*r.get(&[0]).expect("valid index") - 2.0).abs() < 1e-10);
        assert!(r.get(&[1]).expect("valid index").is_nan());
    }

    #[test]
    fn test_exp_ln_roundtrip() {
        let t = Tensor::<f64, Ix1>::from_shape_vec([1], vec![2.0]).expect("valid tensor shape");
        let r = t.ln().exp();
        assert!((*r.get(&[0]).expect("valid index") - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_floor_ceil() {
        let t =
            Tensor::<f64, Ix1>::from_shape_vec([2], vec![1.7, 1.3]).expect("valid tensor shape");
        let f = t.floor();
        let c = t.ceil();
        assert_eq!(*f.get(&[0]).expect("valid index"), 1.0);
        assert_eq!(*c.get(&[1]).expect("valid index"), 2.0);
    }

    // ── W16T5: Complex op tests ──

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

    #[test]
    fn test_conjugate() {
        let t = Tensor::<Complex<f64>, Ix1>::from_shape_vec([1], vec![Complex::new(1.0_f64, 2.0)])
            .expect("valid tensor shape");
        let r = t.conjugate();
        assert_eq!(r.get(&[0]).expect("valid index").re(), 1.0);
        assert_eq!(r.get(&[0]).expect("valid index").im(), -2.0);
    }

    // ── W16T7: Logical NOT test ──

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
