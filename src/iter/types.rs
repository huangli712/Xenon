//! F-order stride state machine.
//!
//! `StrideState` is an internal implementation detail. It is `pub(crate)` so
//! `IndexedIter` / `Iter` in sibling modules can reuse it without leaking the
//! type into the public API surface.

// ── StrideState ──

#[derive(Debug, Clone)]
pub(crate) struct StrideState {
    shape: Vec<usize>,
    index: Vec<usize>,
    finished: bool,
}

impl StrideState {
    /// Build a new state for the given shape.
    ///
    /// Empty shape (`Ix0` / rank-0 `IxDyn`) starts at the empty index and yields
    /// exactly one position before finishing. Shapes containing a zero dimension
    /// start as finished (zero yields total).
    pub(crate) fn new(shape: &[usize]) -> Self {
        let finished = shape.contains(&0);
        Self {
            shape: shape.to_vec(),
            index: vec![0; shape.len()],
            finished,
        }
    }

    /// Current logical index (length == `ndim`).
    pub(crate) fn index(&self) -> &[usize] {
        &self.index
    }

    /// Advance one step in F-order; mark finished after the last position.
    ///
    /// Pseudocode:
    ///
    /// ```text
    /// for i in 0..ndim:
    ///     index[i] += 1
    ///     if index[i] < shape[i]:
    ///         return  // no carry
    ///     index[i] = 0  // carry to next dimension
    /// ```
    pub(crate) fn advance(&mut self) {
        if self.finished {
            return;
        }
        if self.shape.is_empty() {
            // Ix0 / rank-0 IxDyn: yield exactly one (empty) index, then finish.
            self.finished = true;
            return;
        }
        for axis in 0..self.shape.len() {
            self.index[axis] += 1;
            if self.index[axis] < self.shape[axis] {
                return;
            }
            self.index[axis] = 0;
        }
        // All axes carried out → exhausted.
        self.finished = true;
    }

    /// Whether the state machine has exhausted all logical positions.
    #[cfg_attr(not(test), expect(dead_code, reason = "used in tests"))]
    /// Whether the state machine has exhausted all logical positions.
    pub(crate) fn is_finished(&self) -> bool {
        self.finished
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::StrideState;

    /// F-order increment: index[0] varies fastest; carry propagates to higher axes.
    #[test]
    fn test_stride_state_increment() {
        let mut state = StrideState::new(&[2, 3]);
        assert!(!state.is_finished());
        assert_eq!(state.index(), &[0, 0]);

        state.advance();
        assert_eq!(state.index(), &[1, 0]);

        state.advance();
        // Carry from axis 0 (2 -> 0) into axis 1 (0 -> 1).
        assert_eq!(state.index(), &[0, 1]);

        state.advance();
        assert_eq!(state.index(), &[1, 1]);

        state.advance();
        assert_eq!(state.index(), &[0, 2]);

        state.advance();
        assert_eq!(state.index(), &[1, 2]);

        // Final advance exhausts the state machine.
        state.advance();
        assert!(state.is_finished());

        // Further advances are no-ops.
        state.advance();
        assert!(state.is_finished());
    }

    /// Ix0 / rank-0 IxDyn yields exactly one (empty) index before finishing.
    #[test]
    fn test_stride_state_ix0() {
        let mut state = StrideState::new(&[]);
        assert!(!state.is_finished());
        assert_eq!(state.index(), &[] as &[usize]);

        state.advance();
        assert!(state.is_finished());
    }

    /// Empty array (`shape=[0, 3]`) finishes before producing any index.
    #[test]
    fn test_stride_state_empty() {
        let state = StrideState::new(&[0, 3]);
        assert!(state.is_finished());
    }
}
