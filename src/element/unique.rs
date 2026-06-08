//! `UniqueElement` sealed marker trait for the `unique()` set operation.
//!
//! Defines the public `UniqueElement` trait that restricts `unique()` to a
//! closed set of numeric types, excluding `bool`.

use crate::complex::Complex;
use crate::private::Sealed;
use super::Element;

/// Sealed marker trait for types that support the `unique` operation.
///
/// Implementors: `i32`, `i64`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`.
/// `bool` is intentionally excluded per requirements specification §15.
///
/// Deduplication equality is `PartialEq` (`Element: PartialEq`): each `NaN`
/// is preserved (`NaN != NaN`), `-0.0` and `0.0` compare equal, and complex
/// values compare component-wise.
pub trait UniqueElement: Sealed + Element {}

impl UniqueElement for i32 {}
impl UniqueElement for i64 {}
impl UniqueElement for f32 {}
impl UniqueElement for f64 {}
impl UniqueElement for Complex<f32> {}
impl UniqueElement for Complex<f64> {}
