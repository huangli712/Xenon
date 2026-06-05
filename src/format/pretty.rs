use core::fmt::{self, Formatter, Write as _};

use crate::dimension::Dimension;
use crate::element::{Element, element_type_of};
use crate::storage::Storage;
use crate::tensor::TensorBase;

use super::config::FormatConfig;

use super::writer::LineWriter;

/// Internal Display dispatch: 0D → `Tensor0(...)`; 1D → `fmt_1d_display`;
/// ND (n ≥ 2) → `fmt_nd_display`. Mirrors `22-output §5.3` line 235-249.
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
            // §5.3 line 237-242 + Decision 3 line 793-799: explicit Tensor0(...) marker.
            write!(f, "Tensor0(")?;
            fmt_scalar_display(f, read_logical(tensor, &[]), config)?;
            write!(f, ")")
        },
        1 => fmt_1d_display(f, tensor, config),
        _ => fmt_nd_display(f, tensor, config),
    }
}

/// Internal Debug dispatch: writes a header line then routes the data section
/// through the same ndim-dispatch as Display, using Debug helpers.
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
    // Header: §5.4 line 287-301, §5.5 line 328-334
    writeln!(
        f,
        "Tensor(shape={:?}, strides={:?}, dtype={}, layout={})",
        tensor.shape(),
        tensor.strides(),
        element_type_of::<A>().name(),
            tensor.layout_state().as_str(),    )?;
    // Data section: route through W26T3 debug helpers.
    // §5.6 line 391 — Debug data omits Display's trailing shape suffix.
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

/// Display-mode scalar rendering. Honors `precision` for types that
/// support `{:.N}`; types that do not (e.g. `bool`, integers) ignore it.
///
/// Generic over `W: fmt::Write` so it accepts both `&mut Formatter<'_>`
/// (used by 0D entries in W26T4 / W26T5) and `&mut LineWriter<'_, '_>`
/// (used by 1D / ND walkers to track column positions for line_width).
pub(crate) fn fmt_scalar_display<W, A>(w: &mut W, value: &A, config: FormatConfig) -> fmt::Result
where
    W: fmt::Write,
    A: fmt::Display,
{
    match config.precision {
        Some(p) => write!(w, "{value:.p$}"),
        None => write!(w, "{value}"),
    }
}

/// Debug-mode scalar rendering. `precision` does NOT propagate into Debug
/// (per `22-output §6.1` line 493 — precision is a Display concern).
pub(crate) fn fmt_scalar_debug<W, A>(w: &mut W, value: &A, _config: FormatConfig) -> fmt::Result
where
    W: fmt::Write,
    A: fmt::Debug,
{
    write!(w, "{value:?}")
}

/// Read the element at logical index `indices` (length must equal `ndim()`).
/// Returns `&A` obtained via raw pointer arithmetic. Caller must ensure each
/// index is in range; pretty.rs only generates in-range coordinates.
pub(crate) fn read_logical<'a, S, D, A>(tensor: &'a TensorBase<S, D>, indices: &[usize]) -> &'a A
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element,
{
    debug_assert_eq!(indices.len(), tensor.ndim());
    let strides = tensor.strides();
    let mut rel_offset: isize = 0;
    for (axis, &idx) in indices.iter().enumerate() {
        // strides[axis] is `usize`; casting to `isize` is safe because a valid
        // tensor's max offset fits in isize (05-storage.md §5 + 06-layout.md §5.8.3).
        rel_offset += (idx as isize) * (strides[axis] as isize);
    }
    // SAFETY:
    // - `tensor.as_ptr()` returns `storage_base.add(tensor.offset())`, pointing
    //   at the first logical element (07-tensor.md §5.4 line 463).
    // - pretty.rs only produces `indices` satisfying `indices[i] < shape[i]`,
    //   so `rel_offset` stays within the contiguous span of the logical view
    //   (06-layout.md §5.8.3 max-offset bound; broadcast zero-strides alias
    //   back to valid positions, never out-of-bounds).
    // - The resulting `&A` lifetime is tied to `'a`, which is bounded by the
    //   immutable borrow of `tensor`, preserving aliasing rules.
    unsafe { &*tensor.as_ptr().offset(rel_offset) }
}

// ── 1D Display / Debug ──

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
    // Two-stage check mirroring §5.6 pseudocode & ND path:
    //   - `truncated`: global trigger (§5.6 line 383, `len > threshold`)
    //   - `axis_truncated`: axis-level rule (§5.6 line 386, `axis_len > 2*edge`)
    let truncated = len > config.threshold;
    let axis_truncated = truncated && len > 2 * edge;
    // 1D is an innermost axis (axis = 0, ndim = 1). `write_separator`
    // interprets `axis + 1 == ndim` as the flat-row path and consults
    // `config.line_width` for soft-wrap decisions (§5.6 line 392).
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
        // §5.6 line 390 — Display-only outer suffix with shape.
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
    // Two-stage check, symmetric to fmt_1d_display (§5.6 line 383 & 386).
    let truncated = len > config.threshold;
    let axis_truncated = truncated && len > 2 * edge;
    // 1D innermost axis; shares the line_width soft-wrap path with Display.
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
        // §5.6 line 391 — Debug does NOT append the trailing shape suffix.
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

// ── ND Display / Debug ──

/// ND Display rendering. Outer entry; sets up the index buffer and decides
/// whether truncation is globally active, then delegates to a recursive walker.
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
    let visible = walk_axis_display(&mut w, tensor, config, 0, &mut indices, truncated)?;

    if truncated {
        let omitted = total.saturating_sub(visible);
        if omitted > 0 {
            // §5.6 line 390 + §5.5 line 360 — outer-most truncation suffix.
            write!(
                w,
                " ... ({omitted} elements omitted)  shape={:?}",
                tensor.shape()
            )?;
        }
    }
    Ok(())
}

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
    let visible = walk_axis_debug(&mut w, tensor, config, 0, &mut indices, truncated)?;

    if truncated {
        let omitted = total.saturating_sub(visible);
        if omitted > 0 {
            // §5.6 line 391 — Debug omits the shape suffix.
            write!(w, " ... ({omitted} elements omitted)")?;
        }
    }
    Ok(())
}

/// Recursive axis walker (Display). Returns the number of visible logical
/// elements rendered under this subtree, used to compute `omitted`.
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
            visible += walk_axis_display(w, tensor, config, axis + 1, indices, truncated)?;
            indices.pop();
        }
    } else {
        // Head [0, edge)
        for i in 0..edge {
            if i > 0 {
                write_separator(w, axis, tensor.ndim(), config)?;
            }
            indices.push(i);
            visible += walk_axis_display(w, tensor, config, axis + 1, indices, truncated)?;
            indices.pop();
        }
        // Ellipsis marker between head and tail along this axis
        write_separator(w, axis, tensor.ndim(), config)?;
        write!(w, "...")?;
        // Tail [axis_len - edge, axis_len)
        for i in (axis_len - edge)..axis_len {
            write_separator(w, axis, tensor.ndim(), config)?;
            indices.push(i);
            visible += walk_axis_display(w, tensor, config, axis + 1, indices, truncated)?;
            indices.pop();
        }
    }
    write!(w, "]")?;
    Ok(visible)
}

/// Recursive axis walker (Debug). Same structure as Display, but element
/// rendering routes through `fmt_scalar_debug`. No trailing shape suffix.
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
            visible += walk_axis_debug(w, tensor, config, axis + 1, indices, truncated)?;
            indices.pop();
        }
    } else {
        for i in 0..edge {
            if i > 0 {
                write_separator(w, axis, tensor.ndim(), config)?;
            }
            indices.push(i);
            visible += walk_axis_debug(w, tensor, config, axis + 1, indices, truncated)?;
            indices.pop();
        }
        write_separator(w, axis, tensor.ndim(), config)?;
        write!(w, "...")?;
        for i in (axis_len - edge)..axis_len {
            write_separator(w, axis, tensor.ndim(), config)?;
            indices.push(i);
            visible += walk_axis_debug(w, tensor, config, axis + 1, indices, truncated)?;
            indices.pop();
        }
    }
    write!(w, "]")?;
    Ok(visible)
}

/// Inter-element separator.
///
/// - Outer axes (non-innermost): always emit `,\n` + indentation by depth
///   (`22-output §5.5` line 322-326 Numpy-style nested matrix layout).
/// - Innermost axis (`axis + 1 == ndim`): default `, `; when the current
///   column after emitting `, ` and the next scalar would plausibly exceed
///   `config.line_width`, soft-wrap at this element boundary per
///   `22-output §5.6` line 392-393.
///   The check is done `before` writing the separator: if the current
///   `w.column() >= config.line_width`, emit a newline-based separator
///   (`,\n` + depth indent) instead of flat `, `. This keeps truncation
///   outcomes unchanged (§5.6 line 392) and only affects line breaks.
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
        // Innermost axis. Decide flat vs. soft-wrap by current column.
        if w.column() >= config.line_width {
            // Soft wrap at this element boundary. Indent to the innermost
            // axis depth so the wrapped row aligns with the opening `[`
            // of the current inner list (§5.6 line 393 «axis boundary»).
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
    use crate::dimension::Ix1;
    use crate::layout::Strides;

    /// Internal helper: invoke fmt_1d_display via a tiny adapter type.
    fn fmt_1d_display_string<A: Element + core::fmt::Display>(
        tensor: &TensorBase<crate::storage::Owned<A>, crate::dimension::Ix1>,
        config: FormatConfig,
    ) -> String {
        let mut s = String::new();
        struct Wrap<'a, A: Element>(
            &'a TensorBase<crate::storage::Owned<A>, crate::dimension::Ix1>,
            FormatConfig,
        );
        impl<'a, A: Element + core::fmt::Display> core::fmt::Display for Wrap<'a, A> {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                fmt_1d_display(f, self.0, self.1)
            }
        }
        write!(&mut s, "{}", Wrap(tensor, config)).expect("formatting to String is infallible");
        s
    }

    #[test]
    fn test_fmt_1d_full() {
        let tensor = unsafe { TensorBase::from_raw_vec_unchecked(vec![1, 2, 3], Ix1(3)) };
        assert_eq!(
            fmt_1d_display_string(&tensor, FormatConfig::default()),
            "[1, 2, 3]"
        );
    }

    #[test]
    fn test_fmt_1d_truncated() {
        let tensor =
            unsafe { TensorBase::from_raw_vec_unchecked(vec![1, 2, 3, 4, 5, 6, 7], Ix1(7)) };
        let config = FormatConfig {
            edge_items: 2,
            threshold: 4,
            ..Default::default()
        };
        let text = fmt_1d_display_string(&tensor, config);
        // §5.6 line 390 — Display appends "... (N elements omitted)  shape=[...]".
        assert!(
            text.contains("... (3 elements omitted)  shape=[7]"),
            "text = {text:?}"
        );
        // Head + tail values present, middle omitted.
        assert!(text.contains("[1, 2"), "text = {text:?}");
        assert!(text.contains("6, 7]"), "text = {text:?}");
    }

    #[test]
    fn test_fmt_1d_empty() {
        let tensor = unsafe { TensorBase::from_raw_vec_unchecked(Vec::<i32>::new(), Ix1(0)) };
        assert_eq!(
            fmt_1d_display_string(&tensor, FormatConfig::default()),
            "[]"
        );
    }

    #[test]
    fn test_fmt_1d_single() {
        let tensor = unsafe { TensorBase::from_raw_vec_unchecked(vec![42], Ix1(1)) };
        assert_eq!(
            fmt_1d_display_string(&tensor, FormatConfig::default()),
            "[42]"
        );
    }

    #[test]
    fn test_fmt_2d_logical_order() {
        use crate::dimension::Ix2;
        // F-order storage: physical = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        // Logical [i, j] reads tensor[i, j]; §5.5 line 322-326 expects
        // [[1, 4, 7], [2, 5, 8], [3, 6, 9]] with newline+indent between rows.
        let tensor = unsafe {
            TensorBase::from_raw_vec_unchecked(vec![1, 2, 3, 4, 5, 6, 7, 8, 9], Ix2(3, 3))
        };
        struct Wrap<'a>(&'a TensorBase<crate::storage::Owned<i32>, Ix2>);
        impl<'a> core::fmt::Display for Wrap<'a> {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
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

    #[test]
    fn test_fmt_large_2d_truncated() {
        use crate::dimension::Ix2;
        // §8.2 line 688 (high priority): large 2D array with both-axis truncation.
        // §5.5 line 351-361: shape=[100, 100], edge_items=3, threshold=1000 →
        // each axis shows 6 indices → visible = 36, omitted = 9964.
        let data: Vec<i32> = (1..=10_000).collect();
        let tensor = unsafe { TensorBase::from_raw_vec_unchecked(data, Ix2(100, 100)) };
        struct Wrap<'a>(&'a TensorBase<crate::storage::Owned<i32>, Ix2>);
        impl<'a> core::fmt::Display for Wrap<'a> {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                fmt_nd_display(f, self.0, FormatConfig::default())
            }
        }
        let text = format!("{}", Wrap(&tensor));
        // Outer truncation suffix present (§5.6 line 390, two spaces before shape=).
        assert!(
            text.contains("... (9964 elements omitted)  shape=[100, 100]"),
            "text = {text:?}"
        );
        // Row-level ellipsis between head rows and tail rows (§5.5 line 357).
        assert!(text.contains("..."), "text = {text:?}");
        // First logical row [1, 101, 201, ..., 9801, 9901] — F-order logical read.
        assert!(text.contains("1, 101, 201"), "text = {text:?}");
        // Last logical row [100, 200, 300, ..., 9900, 10000].
        assert!(text.contains("9900, 10000"), "text = {text:?}");
    }

    /// Construct a tensor view by manually assembling raw parts.
    unsafe fn make_view<A: Element>(
        base: &TensorBase<crate::storage::Owned<A>, crate::dimension::Ix2>,
        shape: crate::dimension::Ix2,
        strides: Strides<crate::dimension::Ix2>,
    ) -> TensorBase<crate::storage::ViewRepr<'_, A>, crate::dimension::Ix2> {
        // SAFETY: shape and strides are manually constructed from the
        // original tensor's data and known geometric transformations
        // (transpose or broadcast), keeping all logical indices within
        // the valid storage range.
        unsafe {
            TensorBase::from_raw_parts(base.as_ptr(), base.storage_len(), shape, strides, 0)
                .expect("valid layout from manually constructed strides")
        }
    }

    #[test]
    fn test_fmt_broadcast_view() {
        use crate::dimension::Ix2;
        // §8.2 line 689 (high priority): broadcast view must render in logical
        // index order, not according to physical zero-stride aliasing.
        // Source: shape=[1, 3], data=[1, 2, 3]; broadcast to [4, 3] produces a
        // view where row 0 == row 1 == row 2 == row 3 == [1, 2, 3].
        let base = unsafe { TensorBase::from_raw_vec_unchecked(vec![1_i32, 2, 3], Ix2(1, 3)) };
        // Broadcast: shape=[4,3], strides=[0,1] (F-order for shape [1,3] is strides=[1,1];
        // broadcast adds zero stride for the broadcast axis).
        let view = unsafe { make_view(&base, Ix2(4, 3), Strides::new(Ix2(0, 1))) };
        struct Wrap<'a>(&'a TensorBase<crate::storage::ViewRepr<'a, i32>, crate::dimension::Ix2>);
        impl<'a> core::fmt::Display for Wrap<'a> {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                fmt_nd_display(f, self.0, FormatConfig::default())
            }
        }
        let text = format!("{}", Wrap(&view));
        // All 4 logical rows must render as [1, 2, 3] in logical order,
        // regardless of the zero-stride physical aliasing.
        let row_count = text.matches("[1, 2, 3]").count();
        assert_eq!(row_count, 4, "expected 4 broadcast rows; text = {text:?}");
    }

    #[test]
    fn test_fmt_transposed_view() {
        use crate::dimension::Ix2;
        // §8.2 line 690 (high priority): transposed view must render in
        // logical row/column order, not physical F-order storage order.
        // Source: shape=[2, 3] F-order, data = [1, 2, 3, 4, 5, 6]
        //   logical matrix = [[1, 3, 5], [2, 4, 6]]
        // Transposed to shape=[3, 2]:
        //   logical matrix = [[1, 2], [3, 4], [5, 6]]
        let base =
            unsafe { TensorBase::from_raw_vec_unchecked(vec![1_i32, 2, 3, 4, 5, 6], Ix2(2, 3)) };
        // Original F-order strides for [2,3] = [1, 2]
        // Transposed: shape=[3,2], strides=[2,1]
        let view = unsafe { make_view(&base, Ix2(3, 2), Strides::new(Ix2(2, 1))) };
        struct Wrap<'a>(&'a TensorBase<crate::storage::ViewRepr<'a, i32>, crate::dimension::Ix2>);
        impl<'a> core::fmt::Display for Wrap<'a> {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                fmt_nd_display(f, self.0, FormatConfig::default())
            }
        }
        let text = format!("{}", Wrap(&view));
        // Logical row ordering (§5.3 line 213-214, §8.3 line 705).
        assert!(text.contains("[1, 2]"), "text = {text:?}");
        assert!(text.contains("[3, 4]"), "text = {text:?}");
        assert!(text.contains("[5, 6]"), "text = {text:?}");
        // Outer matrix structure preserved.
        assert!(text.starts_with("[["), "text = {text:?}");
        assert!(text.ends_with("]]"), "text = {text:?}");
    }

    #[test]
    fn test_fmt_3d() {
        use crate::dimension::Ix3;
        // §8.2 line 679 (medium priority) + §5.5 line 336-343 example:
        // shape=[2, 2, 2], F-order storage = [1, 2, 3, 4, 5, 6, 7, 8] →
        // logical = [[[1, 5], [3, 7]], [[2, 6], [4, 8]]].
        let tensor = unsafe {
            TensorBase::from_raw_vec_unchecked(vec![1_i32, 2, 3, 4, 5, 6, 7, 8], Ix3(2, 2, 2))
        };
        struct Wrap<'a>(&'a TensorBase<crate::storage::Owned<i32>, Ix3>);
        impl<'a> core::fmt::Display for Wrap<'a> {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                fmt_nd_display(f, self.0, FormatConfig::default())
            }
        }
        let text = format!("{}", Wrap(&tensor));
        // Innermost rows: [1, 5] / [3, 7] / [2, 6] / [4, 8].
        assert!(text.contains("[1, 5]"), "text = {text:?}");
        assert!(text.contains("[3, 7]"), "text = {text:?}");
        assert!(text.contains("[2, 6]"), "text = {text:?}");
        assert!(text.contains("[4, 8]"), "text = {text:?}");
        // Triple-nested brackets — 3D envelope.
        assert!(text.starts_with("[[["), "text = {text:?}");
        assert!(text.ends_with("]]]"), "text = {text:?}");
    }

    #[test]
    fn test_line_width_wrapping() {
        // §8.2 line 691 (medium priority): long 1D row must soft-wrap when
        // the accumulated column width exceeds config.line_width. Truncation
        // must remain unchanged (§5.6 line 392).
        let tensor = unsafe {
            TensorBase::from_raw_vec_unchecked(
                vec![100_i32, 200, 300, 400, 500, 600, 700, 800],
                Ix1(8),
            )
        };
        // Default threshold=1000 means len=8 ≤ threshold → no truncation.
        // Narrow line_width forces soft-wrap on element boundaries.
        let config = FormatConfig {
            line_width: 20,
            ..Default::default()
        };
        let text = fmt_1d_display_string(&tensor, config);
        // All elements present — soft-wrap must not drop any value.
        for v in [100, 200, 300, 400, 500, 600, 700, 800] {
            assert!(
                text.contains(&v.to_string()),
                "missing {v}: text = {text:?}"
            );
        }
        // At least one soft-wrap inserted (newline present inside the row).
        assert!(
            text.contains('\n'),
            "expected soft-wrap newline; text = {text:?}"
        );
        // No truncation suffix — `line_width` only affects line breaks,
        // never truncation outcomes (§5.6 line 392).
        assert!(
            !text.contains("elements omitted"),
            "line_width must not trigger truncation; text = {text:?}"
        );
    }

    #[test]
    fn test_line_width_narrow() {
        // §8.2 line 692 (medium priority): narrow line_width triggers more
        // frequent soft-wraps than a wide one, while keeping element set
        // and ordering identical.
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
        // Wide config fits on a single line.
        assert_eq!(wide_text, "[10, 20, 30, 40, 50, 60]");
        // Narrow config inserts more newlines than the wide config.
        let wide_nl = wide_text.matches('\n').count();
        let narrow_nl = narrow_text.matches('\n').count();
        assert!(
            narrow_nl > wide_nl,
            "narrow must wrap more; wide={wide_nl}, narrow={narrow_nl}, text={narrow_text:?}"
        );
        // Element set preserved despite wrapping.
        for v in [10, 20, 30, 40, 50, 60] {
            assert!(
                narrow_text.contains(&v.to_string()),
                "missing {v} in narrow: {narrow_text:?}"
            );
        }
    }
}
