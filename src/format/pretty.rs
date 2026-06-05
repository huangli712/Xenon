//! Pretty-printing pipeline — 0D / 1D / ND display and debug formatters,
//! scalar rendering helpers, and the recursive axis walkers that produce
//! Numpy-style nested-bracket output.

use core::fmt::{self, Formatter, Write as _};

use crate::dimension::Dimension;
use crate::element::{Element, element_type_of};
use crate::storage::Storage;
use crate::tensor::TensorBase;

use super::config::FormatConfig;
use super::writer::LineWriter;

// --- Top-level dispatch -----------------------------------------------------

/// Internal Display dispatch.
///
/// Zero-dimension tensors render as `Tensor0(value)`.
/// 1D tensors delegate to [`fmt_1d_display`].
/// ND (n ≥ 2) tensors delegate to [`fmt_nd_display`].
pub(crate) fn format_tensor_display<S, D, A>(
    f: &mut Formatter<'_>,
    tensor: &TensorBase<S, D>,
    config: FormatConfig,
) -> fmt::Result
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + fmt::Display,
{
    match tensor.ndim() {
        0 => {
            write!(f, "Tensor0(")?;
            fmt_scalar_display(f, read_logical(tensor, &[]), config)?;
            write!(f, ")")
        },
        1 => fmt_1d_display(f, tensor, config),
        _ => fmt_nd_display(f, tensor, config),
    }
}

/// Internal Debug dispatch.
///
/// Writes a header line with metadata (shape, strides, dtype, layout),
/// then dispatches the data section through the same ndim routing as
/// Display, using Debug formatting helpers. Debug output does NOT
/// include the trailing shape suffix that Display uses.
pub(crate) fn format_tensor_debug<S, D, A>(
    f: &mut Formatter<'_>,
    tensor: &TensorBase<S, D>,
    config: FormatConfig,
) -> fmt::Result
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + fmt::Debug,
{
    writeln!(
        f,
        "Tensor(shape={:?}, strides={:?}, dtype={}, layout={})",
        tensor.shape(),
        tensor.strides(),
        element_type_of::<A>().name(),
        tensor.layout_state().as_str(),
    )?;
    match tensor.ndim() {
        0 => {
            write!(f, "Tensor0(")?;
            fmt_scalar_debug(f, read_logical(tensor, &[]), config)?;
            write!(f, ")")
        },
        1 => fmt_1d_debug(f, tensor, config),
        _ => fmt_nd_debug(f, tensor, config),
    }
}

// --- Scalar helpers ---------------------------------------------------------

/// Display-mode scalar rendering.
///
/// Honors [`FormatConfig::precision`] for types that support `{:.N}`;
/// types that do not (e.g. `bool`, integers) ignore it.
///
/// Generic over `W: fmt::Write` so it accepts both `&mut Formatter<'_>`
/// (used by 0D entries) and `&mut LineWriter<'_, '_>` (used by 1D/ND
/// walkers to track column positions for line_width-based soft-wrapping).
pub(crate) fn fmt_scalar_display<W, A>(
    w: &mut W,
    value: &A,
    config: FormatConfig
) -> fmt::Result
where
    W: fmt::Write,
    A: fmt::Display,
{
    match config.precision {
        Some(p) => write!(w, "{value:.p$}"),
        None => write!(w, "{value}"),
    }
}

/// Debug-mode scalar rendering.
///
/// `FormatConfig::precision` does NOT apply to Debug output — precision
/// is a Display-only concern.
pub(crate) fn fmt_scalar_debug<W, A>(
    w: &mut W,
    value: &A,
    _config: FormatConfig
) -> fmt::Result
where
    W: fmt::Write,
    A: fmt::Debug,
{
    write!(w, "{value:?}")
}

// --- Logical element access -------------------------------------------------

/// Read the element at logical index `indices` (length must equal `ndim()`).
///
/// Returns a reference obtained via raw pointer arithmetic. Callers must
/// ensure each index is in range; the pretty-printing pipeline only
/// generates in-range coordinates.
pub(crate) fn read_logical<'a, S, D, A>(
    tensor: &'a TensorBase<S, D>,
    indices: &[usize]
) -> &'a A
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element,
{
    debug_assert_eq!(indices.len(), tensor.ndim());
    let strides = tensor.strides();
    let mut rel_offset: isize = 0;
    for (axis, &idx) in indices.iter().enumerate() {
        // strides[axis] is `usize`; casting to `isize` is safe because a
        // valid tensor's max offset fits in `isize`.
        rel_offset += (idx as isize) * (strides[axis] as isize);
    }
    // SAFETY:
    // - `tensor.as_ptr()` returns `storage_base.add(tensor.offset())`,
    //   pointing at the first logical element.
    //
    // - The pretty-printing pipeline only produces indices satisfying
    //   `indices[i] < shape[i]`, so `rel_offset` stays within the
    //   contiguous span of the logical view. Broadcast zero-strides alias
    //   back to valid positions, never out-of-bounds.
    //
    // - The resulting `&A` lifetime is tied to `'a`, which is bounded by
    //   the immutable borrow of `tensor`, preserving aliasing rules.
    unsafe { &*tensor.as_ptr().offset(rel_offset) }
}

// --- 1D Display / Debug -----------------------------------------------------

/// Entry point for 1D Display formatting.
///
/// Wraps the formatter in a [`LineWriter`] and delegates to
/// `fmt_1d_display_into` for soft-wrap-capable output.
pub(crate) fn fmt_1d_display<S, D, A>(
    f: &mut Formatter<'_>,
    tensor: &TensorBase<S, D>,
    config: FormatConfig,
) -> fmt::Result
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + fmt::Display,
{
    debug_assert_eq!(tensor.ndim(), 1);
    let mut w = LineWriter::new(f);
    fmt_1d_display_into(&mut w, tensor, config)
}

/// Core 1D Display rendering using a [`LineWriter`].
///
/// Handles two truncation stages:
///
/// 1. **Global truncation** — triggered when `tensor.len() > threshold`.
/// 2. **Axis-level truncation** — triggered when the axis length exceeds
///    `2 * edge_items`. When active, renders edge items from both ends
///    with an ellipsis in between, and appends a trailing shape suffix
///    `(... (N elements omitted)  shape=[...])`.
///
/// Column-tracking via [`LineWriter`] enables soft wrapping at
/// `config.line_width` without changing truncation outcomes.
fn fmt_1d_display_into<S, D, A>(
    w: &mut LineWriter<'_, '_>,
    tensor: &TensorBase<S, D>,
    config: FormatConfig,
) -> fmt::Result
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + fmt::Display,
{
    let len = tensor.len();
    let edge = config.normalized_edge_items();
    let truncated = len > config.threshold;
    let axis_truncated = truncated && len > 2 * edge;
    // 1D is an innermost axis (axis=0, ndim=1), so `write_separator`
    // uses the flat-row path with `config.line_width` for soft-wrap.
    let axis: usize = 0;
    let ndim: usize = 1;

    write!(w, "[")?;
    if axis_truncated {
        // Head: indices [0, edge)
        for i in 0..edge {
            if i > 0 {
                write_separator(w, axis, ndim, config)?;
            }
            fmt_scalar_display(w, read_logical(tensor, &[i]), config)?;
        }
        write_separator(w, axis, ndim, config)?;
        write!(w, "...")?;
        // Tail: indices [len-edge, len)
        for i in (len - edge)..len {
            write_separator(w, axis, ndim, config)?;
            fmt_scalar_display(w, read_logical(tensor, &[i]), config)?;
        }
        write!(w, "]")?;
        // Display appends a trailing shape suffix when truncated.
        let omitted = len - edge * 2;
        write!(
            w,
            " ... ({omitted} elements omitted)  shape={:?}",
            tensor.shape()
        )
    } else {
        for i in 0..len {
            if i > 0 {
                write_separator(w, axis, ndim, config)?;
            }
            fmt_scalar_display(w, read_logical(tensor, &[i]), config)?;
        }
        write!(w, "]")
    }
}

/// Entry point for 1D Debug formatting.
///
/// Wraps the formatter in a [`LineWriter`] and delegates to `fmt_1d_debug_into`.
pub(crate) fn fmt_1d_debug<S, D, A>(
    f: &mut Formatter<'_>,
    tensor: &TensorBase<S, D>,
    config: FormatConfig,
) -> fmt::Result
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + fmt::Debug,
{
    debug_assert_eq!(tensor.ndim(), 1);
    let mut w = LineWriter::new(f);
    fmt_1d_debug_into(&mut w, tensor, config)
}

/// Core 1D Debug rendering using a [`LineWriter`].
///
/// Structurally identical to `fmt_1d_display_into` but renders elements
/// via `fmt_scalar_debug` and does NOT append the trailing shape suffix
/// when truncated.
fn fmt_1d_debug_into<S, D, A>(
    w: &mut LineWriter<'_, '_>,
    tensor: &TensorBase<S, D>,
    config: FormatConfig,
) -> fmt::Result
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + fmt::Debug,
{
    let len = tensor.len();
    let edge = config.normalized_edge_items();
    let truncated = len > config.threshold;
    let axis_truncated = truncated && len > 2 * edge;
    let axis: usize = 0;
    let ndim: usize = 1;

    write!(w, "[")?;
    if axis_truncated {
        for i in 0..edge {
            if i > 0 {
                write_separator(w, axis, ndim, config)?;
            }
            fmt_scalar_debug(w, read_logical(tensor, &[i]), config)?;
        }
        write_separator(w, axis, ndim, config)?;
        write!(w, "...")?;
        for i in (len - edge)..len {
            write_separator(w, axis, ndim, config)?;
            fmt_scalar_debug(w, read_logical(tensor, &[i]), config)?;
        }
        write!(w, "]")?;
        // Debug does NOT append the trailing shape suffix.
        let omitted = len - edge * 2;
        write!(w, " ... ({omitted} elements omitted)")
    } else {
        for i in 0..len {
            if i > 0 {
                write_separator(w, axis, ndim, config)?;
            }
            fmt_scalar_debug(w, read_logical(tensor, &[i]), config)?;
        }
        write!(w, "]")
    }
}

// --- ND Display / Debug -----------------------------------------------------

/// ND Display rendering.
///
/// Sets up the index buffer and decides whether truncation is globally
/// active, then delegates to a recursive axis walker. After the walk,
/// appends a trailing shape suffix if truncation occurred.
pub(crate) fn fmt_nd_display<S, D, A>(
    f: &mut Formatter<'_>,
    tensor: &TensorBase<S, D>,
    config: FormatConfig,
) -> fmt::Result
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + fmt::Display,
{
    let total = tensor.len();
    let truncated = total > config.threshold;
    let mut indices: Vec<usize> = Vec::with_capacity(tensor.ndim());
    let mut w = LineWriter::new(f);
    let visible = walk_axis_display(
        &mut w,
        tensor,
        config,
        0,
        &mut indices, truncated
    )?;

    if truncated {
        let omitted = total.saturating_sub(visible);
        if omitted > 0 {
            write!(
                w,
                " ... ({omitted} elements omitted)  shape={:?}",
                tensor.shape()
            )?;
        }
    }
    Ok(())
}

/// ND Debug rendering.
///
/// Same structure as [`fmt_nd_display`], but elements are rendered via
/// Debug formatting and no trailing shape suffix is appended.
pub(crate) fn fmt_nd_debug<S, D, A>(
    f: &mut Formatter<'_>,
    tensor: &TensorBase<S, D>,
    config: FormatConfig,
) -> fmt::Result
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + fmt::Debug,
{
    let total = tensor.len();
    let truncated = total > config.threshold;
    let mut indices: Vec<usize> = Vec::with_capacity(tensor.ndim());
    let mut w = LineWriter::new(f);
    let visible = walk_axis_debug(
        &mut w,
        tensor,
        config,
        0,
        &mut indices, truncated
    )?;

    if truncated {
        let omitted = total.saturating_sub(visible);
        if omitted > 0 {
            write!(w, " ... ({omitted} elements omitted)")?;
        }
    }
    Ok(())
}

/// Recursive axis walker (Display).
///
/// Drives Numpy-style nested bracket output with optional truncation at
/// each axis level. Returns the number of visible logical elements
/// rendered under this subtree, used to compute the `omitted` count
/// for the truncation suffix.
fn walk_axis_display<S, D, A>(
    w: &mut LineWriter<'_, '_>,
    tensor: &TensorBase<S, D>,
    config: FormatConfig,
    axis: usize,
    indices: &mut Vec<usize>,
    truncated: bool,
) -> Result<usize, fmt::Error>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + fmt::Display,
{
    if axis == tensor.ndim() {
        fmt_scalar_display(w, read_logical(tensor, indices), config)?;
        return Ok(1);
    }

    let axis_len = tensor.shape()[axis];
    let edge = config.normalized_edge_items();
    let axis_truncated = truncated && axis_len > 2 * edge;

    write!(w, "[")?;
    let mut visible: usize = 0;
    if !axis_truncated {
        for i in 0..axis_len {
            if i > 0 {
                write_separator(w, axis, tensor.ndim(), config)?;
            }
            indices.push(i);
            visible += walk_axis_display(
                w,
                tensor,
                config,
                axis + 1,
                indices,
                truncated
            )?;
            indices.pop();
        }
    } else {
        // Head: indices [0, edge)
        for i in 0..edge {
            if i > 0 {
                write_separator(w, axis, tensor.ndim(), config)?;
            }
            indices.push(i);
            visible += walk_axis_display(
                w,
                tensor,
                config,
                axis + 1,
                indices,
                truncated
            )?;
            indices.pop();
        }
        // Ellipsis marker between head and tail along this axis.
        write_separator(w, axis, tensor.ndim(), config)?;
        write!(w, "...")?;
        // Tail: indices [axis_len - edge, axis_len)
        for i in (axis_len - edge)..axis_len {
            write_separator(w, axis, tensor.ndim(), config)?;
            indices.push(i);
            visible += walk_axis_display(
                w,
                tensor,
                config,
                axis + 1,
                indices,
                truncated
            )?;
            indices.pop();
        }
    }
    write!(w, "]")?;
    Ok(visible)
}

/// Recursive axis walker (Debug).
///
/// Same recursive structure as [`walk_axis_display`], but delegates
/// element rendering to `fmt_scalar_debug`. No trailing shape suffix
/// is emitted.
fn walk_axis_debug<S, D, A>(
    w: &mut LineWriter<'_, '_>,
    tensor: &TensorBase<S, D>,
    config: FormatConfig,
    axis: usize,
    indices: &mut Vec<usize>,
    truncated: bool,
) -> Result<usize, fmt::Error>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + fmt::Debug,
{
    if axis == tensor.ndim() {
        fmt_scalar_debug(w, read_logical(tensor, indices), config)?;
        return Ok(1);
    }

    let axis_len = tensor.shape()[axis];
    let edge = config.normalized_edge_items();
    let axis_truncated = truncated && axis_len > 2 * edge;

    write!(w, "[")?;
    let mut visible: usize = 0;
    if !axis_truncated {
        for i in 0..axis_len {
            if i > 0 {
                write_separator(w, axis, tensor.ndim(), config)?;
            }
            indices.push(i);
            visible += walk_axis_debug(
                w,
                tensor,
                config,
                axis + 1,
                indices,
                truncated
            )?;
            indices.pop();
        }
    } else {
        for i in 0..edge {
            if i > 0 {
                write_separator(w, axis, tensor.ndim(), config)?;
            }
            indices.push(i);
            visible += walk_axis_debug(
                w,
                tensor,
                config,
                axis + 1,
                indices,
                truncated
            )?;
            indices.pop();
        }
        write_separator(w, axis, tensor.ndim(), config)?;
        write!(w, "...")?;
        for i in (axis_len - edge)..axis_len {
            write_separator(w, axis, tensor.ndim(), config)?;
            indices.push(i);
            visible += walk_axis_debug(
                w,
                tensor,
                config,
                axis + 1,
                indices,
                truncated
            )?;
            indices.pop();
        }
    }
    write!(w, "]")?;
    Ok(visible)
}

/// Inter-element separator with soft-wrap support.
///
/// - **Outer axes** (non-innermost): always emit `,\n` followed by
///   indentation proportional to the current axis depth, producing
///   Numpy-style nested matrix layout.
///
/// - **Innermost axis** (`axis + 1 == ndim`): emits `, ` by default.
///   When the current column has reached or exceeded
///   [`FormatConfig::line_width`], soft-wraps at this element boundary
///   by emitting `,\n` + depth indent instead. This keeps truncation
///   outcomes unchanged and only affects line breaks.
fn write_separator(
    w: &mut LineWriter<'_, '_>,
    axis: usize,
    ndim: usize,
    config: FormatConfig,
) -> fmt::Result {
    let outer_break = |w: &mut LineWriter<'_, '_>, depth: usize| -> fmt::Result {
        writeln!(w, ",")?;
        for _ in 0..=depth {
            write!(w, " ")?;
        }
        Ok(())
    };

    if axis + 1 == ndim {
        // Innermost axis — decide flat vs. soft-wrap by current column.
        if w.column() >= config.line_width {
            // Soft-wrap at this element boundary with indentation to
            // the innermost axis depth, so the wrapped row aligns with
            // the opening `[` of the current inner list.
            outer_break(w, axis)
        } else {
            write!(w, ", ")
        }
    } else {
        // Outer axis — always newline + indent by depth.
        outer_break(w, axis)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::{Ix1, Ix2};
    use crate::layout::Strides;
    use crate::storage::{Owned, ViewRepr};

    /// Internal helper: invoke `fmt_1d_display` via a type that implements
    /// `Display`, returning the formatted string.
    fn fmt_1d_display_string<A: Element + fmt::Display>(
        tensor: &TensorBase<Owned<A>, Ix1>,
        config: FormatConfig,
    ) -> String {
        let mut s = String::new();
        struct Wrap<'a, A: Element>(
            &'a TensorBase<Owned<A>, Ix1>,
            FormatConfig,
        );
        impl<'a, A: Element + fmt::Display> fmt::Display for Wrap<'a, A> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt_1d_display(f, self.0, self.1)
            }
        }
        write!(&mut s, "{}", Wrap(tensor, config))
            .expect("formatting to String is infallible");
        s
    }

    /// Construct a tensor view by manually assembling raw parts (shape
    /// and strides).
    unsafe fn make_view<A: Element>(
        base: &TensorBase<Owned<A>, Ix2>,
        shape: Ix2,
        strides: Strides<Ix2>,
    ) -> TensorBase<ViewRepr<'_, A>, Ix2> {
        // SAFETY: shape and strides are manually constructed from the
        // original tensor's data and known geometric transformations
        // (transpose or broadcast), keeping all logical indices within
        // the valid storage range.
        unsafe {
            TensorBase::from_raw_parts(
                base.as_ptr(),
                base.storage_len(),
                shape,
                strides,
                0
            ).expect("valid layout from manually constructed strides")
        }
    }

    /// Small 1D tensor renders the full contents in a single flat row.
    #[test]
    fn test_fmt_1d_full() {
        let tensor = unsafe {
            TensorBase::from_raw_vec_unchecked(vec![1, 2, 3], Ix1(3))
        };
        assert_eq!(
            fmt_1d_display_string(&tensor, FormatConfig::default()),
            "[1, 2, 3]"
        );
    }

    /// When the axis exceeds `2 * edge_items`, the tensor is truncated
    /// with an ellipsis and a trailing shape suffix.
    #[test]
    fn test_fmt_1d_truncated() {
        let tensor = unsafe {
            TensorBase::from_raw_vec_unchecked(
                vec![1, 2, 3, 4, 5, 6, 7],
                Ix1(7)
            )
        };
        let config = FormatConfig {
            edge_items: 2,
            threshold: 4,
            ..Default::default()
        };
        let text = fmt_1d_display_string(&tensor, config);
        assert!(
            text.contains("... (3 elements omitted)  shape=[7]"),
            "text = {text:?}"
        );
        assert!(text.contains("[1, 2"), "text = {text:?}");
        assert!(text.contains("6, 7]"), "text = {text:?}");
    }

    /// An empty 1D tensor renders as `[]`.
    #[test]
    fn test_fmt_1d_empty() {
        let tensor = unsafe {
            TensorBase::from_raw_vec_unchecked(Vec::<i32>::new(), Ix1(0))
        };
        assert_eq!(
            fmt_1d_display_string(&tensor, FormatConfig::default()),
            "[]"
        );
    }

    /// A single-element 1D tensor renders without commas.
    #[test]
    fn test_fmt_1d_single() {
        let tensor = unsafe {
            TensorBase::from_raw_vec_unchecked(vec![42], Ix1(1))
        };
        assert_eq!(
            fmt_1d_display_string(&tensor, FormatConfig::default()),
            "[42]"
        );
    }

    /// 2D F-order tensor renders in logical index order:
    /// physical `[1..9]` with shape `[3,3]` → rows `[1,4,7]`, `[2,5,8]`, `[3,6,9]`.
    #[test]
    fn test_fmt_2d_logical_order() {
        let tensor = unsafe {
            TensorBase::from_raw_vec_unchecked(
                vec![1, 2, 3, 4, 5, 6, 7, 8, 9],
                Ix2(3, 3)
            )
        };
        struct Wrap<'a>(&'a TensorBase<Owned<i32>, Ix2>);
        impl<'a> fmt::Display for Wrap<'a> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt_nd_display(f, self.0, FormatConfig::default())
            }
        }
        let text = format!("{}", Wrap(&tensor));
        assert!(text.contains("[1, 4, 7]"), "text = {text:?}");
        assert!(text.contains("[2, 5, 8]"), "text = {text:?}");
        assert!(text.contains("[3, 6, 9]"), "text = {text:?}");
        assert!(text.starts_with("[["), "text = {text:?}");
        assert!(text.ends_with("]]"), "text = {text:?}");
    }

    /// Large 2D array `[100,100]` with `edge_items=3, threshold=1000`
    /// triggers both-axis truncation. Verifies that head/tail rows and
    /// elements are present and the outer suffix reports 9964 omitted.
    #[test]
    fn test_fmt_large_2d_truncated() {
        let data: Vec<i32> = (1..=10_000).collect();
        let tensor = unsafe {
            TensorBase::from_raw_vec_unchecked(data, Ix2(100, 100))
        };
        struct Wrap<'a>(&'a TensorBase<Owned<i32>, Ix2>);
        impl<'a> fmt::Display for Wrap<'a> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt_nd_display(f, self.0, FormatConfig::default())
            }
        }
        let text = format!("{}", Wrap(&tensor));
        assert!(
            text.contains("... (9964 elements omitted)  shape=[100, 100]"),
            "text = {text:?}"
        );
        assert!(text.contains("..."), "text = {text:?}");
        assert!(text.contains("1, 101, 201"), "text = {text:?}");
        assert!(text.contains("9900, 10000"), "text = {text:?}");
    }

    /// Broadcast view must render in logical index order, not physical
    /// zero-stride aliasing. A `[1,3]` source broadcast to `[4,3]`
    /// should produce 4 identical logical rows.
    #[test]
    fn test_fmt_broadcast_view() {
        let base = unsafe {
            TensorBase::from_raw_vec_unchecked(vec![1_i32, 2, 3], Ix2(1, 3))
        };
        let view = unsafe {
            make_view(&base, Ix2(4, 3), Strides::new(Ix2(0, 1)))
        };
        struct Wrap<'a>(&'a TensorBase<ViewRepr<'a, i32>, Ix2>);
        impl<'a> fmt::Display for Wrap<'a> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt_nd_display(f, self.0, FormatConfig::default())
            }
        }
        let text = format!("{}", Wrap(&view));
        let row_count = text.matches("[1, 2, 3]").count();
        assert_eq!(row_count, 4, "expected 4 broadcast rows; text = {text:?}");
    }

    /// Transposed view must render in logical row/column order, not
    /// physical F-order storage order.
    #[test]
    fn test_fmt_transposed_view() {
        // Source: shape=[2,3] F-order, data=[1..6]
        //   logical = [[1,3,5], [2,4,6]]
        // Transposed to [3,2]: logical = [[1,2], [3,4], [5,6]]
        let base =
            unsafe { TensorBase::from_raw_vec_unchecked(vec![1_i32, 2, 3, 4, 5, 6], Ix2(2, 3)) };
        let view = unsafe { make_view(&base, Ix2(3, 2), Strides::new(Ix2(2, 1))) };
        struct Wrap<'a>(&'a TensorBase<ViewRepr<'a, i32>, Ix2>);
        impl<'a> fmt::Display for Wrap<'a> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt_nd_display(f, self.0, FormatConfig::default())
            }
        }
        let text = format!("{}", Wrap(&view));
        assert!(text.contains("[1, 2]"), "text = {text:?}");
        assert!(text.contains("[3, 4]"), "text = {text:?}");
        assert!(text.contains("[5, 6]"), "text = {text:?}");
        assert!(text.starts_with("[["), "text = {text:?}");
        assert!(text.ends_with("]]"), "text = {text:?}");
    }

    /// 3D tensor with shape `[2,2,2]` renders with triple-nested brackets
    /// and F-order logical rows.
    #[test]
    fn test_fmt_3d() {
        use crate::dimension::Ix3;
        let tensor = unsafe {
            TensorBase::from_raw_vec_unchecked(vec![1_i32, 2, 3, 4, 5, 6, 7, 8], Ix3(2, 2, 2))
        };
        struct Wrap<'a>(&'a TensorBase<Owned<i32>, Ix3>);
        impl<'a> fmt::Display for Wrap<'a> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt_nd_display(f, self.0, FormatConfig::default())
            }
        }
        let text = format!("{}", Wrap(&tensor));
        assert!(text.contains("[1, 5]"), "text = {text:?}");
        assert!(text.contains("[3, 7]"), "text = {text:?}");
        assert!(text.contains("[2, 6]"), "text = {text:?}");
        assert!(text.contains("[4, 8]"), "text = {text:?}");
        assert!(text.starts_with("[[["), "text = {text:?}");
        assert!(text.ends_with("]]]"), "text = {text:?}");
    }

    /// Long 1D row must soft-wrap when the accumulated column exceeds
    /// `config.line_width`. Truncation must remain unchanged by
    /// line-wrapping decisions.
    #[test]
    fn test_line_width_wrapping() {
        let tensor = unsafe {
            TensorBase::from_raw_vec_unchecked(
                vec![100_i32, 200, 300, 400, 500, 600, 700, 800],
                Ix1(8),
            )
        };
        let config = FormatConfig {
            line_width: 20,
            ..Default::default()
        };
        let text = fmt_1d_display_string(&tensor, config);
        for v in [100, 200, 300, 400, 500, 600, 700, 800] {
            assert!(
                text.contains(&v.to_string()),
                "missing {v}: text = {text:?}"
            );
        }
        assert!(
            text.contains('\n'),
            "expected soft-wrap newline; text = {text:?}"
        );
        assert!(
            !text.contains("elements omitted"),
            "line_width must not trigger truncation; text = {text:?}"
        );
    }

    /// Narrow `line_width` triggers more frequent soft-wraps than a wide
    /// one, while keeping the element set and ordering identical.
    #[test]
    fn test_line_width_narrow() {
        let tensor =
            unsafe { TensorBase::from_raw_vec_unchecked(vec![10_i32, 20, 30, 40, 50, 60], Ix1(6)) };
        let wide = FormatConfig {
            line_width: 200,
            ..Default::default()
        };
        let narrow = FormatConfig {
            line_width: 8,
            ..Default::default()
        };
        let wide_text = fmt_1d_display_string(&tensor, wide);
        let narrow_text = fmt_1d_display_string(&tensor, narrow);
        assert_eq!(wide_text, "[10, 20, 30, 40, 50, 60]");
        let wide_nl = wide_text.matches('\n').count();
        let narrow_nl = narrow_text.matches('\n').count();
        assert!(
            narrow_nl > wide_nl,
            "narrow must wrap more; wide={wide_nl}, narrow={narrow_nl}, text={narrow_text:?}"
        );
        for v in [10, 20, 30, 40, 50, 60] {
            assert!(
                narrow_text.contains(&v.to_string()),
                "missing {v} in narrow: {narrow_text:?}"
            );
        }
    }
}
