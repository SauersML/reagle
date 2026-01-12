//! # HMM State Management
//!
//! Provides efficient storage and access for composite HMM states.
//! This matches Java `imp/ImpStates.java` mosaic haplotype logic.
//!
//! Key concepts:
//! - States are "composite haplotypes" - mosaics of reference haplotype segments
//! - `ThreadedHaps` stores segments in a linked-list arena for O(1) updates
//! - `MosaicCursor` provides SIMD-friendly state access for the HMM hot path

/// Arena-based storage for composite haplotypes using threaded indices.
///
/// This avoids the O(N) shift cost of standard SoA insertion and the
/// heap fragmentation of `Vec<CompositeHap>`.
///
/// # Memory Layout
/// Segments are stored in a global arena (vectors). Each composite haplotype
/// is a linked list of segment indices.
#[derive(Clone, Debug)]
pub struct ThreadedHaps {
    // --- Arena Storage ---
    segments_hap: Vec<u32>,
    segments_end: Vec<u32>,
    segments_next: Vec<u32>,

    // --- State Pointers ---
    state_heads: Vec<u32>,
    state_tails: Vec<u32>,
    state_cursors: Vec<u32>,

    // --- Metadata ---
    n_markers: usize,
}

impl ThreadedHaps {
    const NIL: u32 = u32::MAX;

    /// Create new threaded haps arena
    pub fn new(initial_capacity_states: usize, initial_capacity_segments: usize, n_markers: usize) -> Self {
        Self {
            segments_hap: Vec::with_capacity(initial_capacity_segments),
            segments_end: Vec::with_capacity(initial_capacity_segments),
            segments_next: Vec::with_capacity(initial_capacity_segments),
            state_heads: Vec::with_capacity(initial_capacity_states),
            state_tails: Vec::with_capacity(initial_capacity_states),
            state_cursors: Vec::with_capacity(initial_capacity_states),
            n_markers,
        }
    }

    /// Clear all states (keeps capacity)
    pub fn clear(&mut self) {
        self.segments_hap.clear();
        self.segments_end.clear();
        self.segments_next.clear();
        self.state_heads.clear();
        self.state_tails.clear();
        self.state_cursors.clear();
    }

    /// Number of active states
    pub fn n_states(&self) -> usize {
        self.state_heads.len()
    }

    /// Initialize a new state starting with given hap. Returns the state index.
    pub fn push_new(&mut self, start_hap: u32) -> usize {
        let state_idx = self.state_heads.len();

        let seg_idx = self.segments_hap.len() as u32;
        self.segments_hap.push(start_hap);
        self.segments_end.push(self.n_markers as u32);
        self.segments_next.push(Self::NIL);

        self.state_heads.push(seg_idx);
        self.state_tails.push(seg_idx);
        self.state_cursors.push(seg_idx);

        state_idx
    }

    /// Add a segment to an existing state
    pub fn add_segment(&mut self, state_idx: usize, hap: u32, start_marker: usize) {
        let tail_idx = self.state_tails[state_idx] as usize;

        self.segments_end[tail_idx] = start_marker as u32;

        let new_seg_idx = self.segments_hap.len() as u32;
        self.segments_hap.push(hap);
        self.segments_end.push(self.n_markers as u32);
        self.segments_next.push(Self::NIL);

        self.segments_next[tail_idx] = new_seg_idx;
        self.state_tails[state_idx] = new_seg_idx;
    }

    /// Get haplotype at marker for a state (advances cursor)
    ///
    /// Optimized with fast path: most calls don't cross segment boundaries,
    /// so we check the common case first before entering the advancement loop.
    #[inline]
    pub fn hap_at_raw(&mut self, state_idx: usize, marker: usize) -> u32 {
        let cur = self.state_cursors[state_idx] as usize;

        // Fast path: marker is within current segment (common case ~90%+)
        // This avoids the loop overhead when no advancement is needed
        if marker < self.segments_end[cur] as usize {
            return self.segments_hap[cur];
        }

        // Slow path: need to advance through segments
        self.advance_cursor_slow(state_idx, cur, marker)
    }

    /// Slow path for cursor advancement (called when segment boundary crossed)
    #[cold]
    #[inline(never)]
    fn advance_cursor_slow(&mut self, state_idx: usize, mut cur: usize, marker: usize) -> u32 {
        loop {
            let next = self.segments_next[cur];
            if next == Self::NIL {
                break;
            }
            cur = next as usize;
            if marker < self.segments_end[cur] as usize {
                break;
            }
        }

        self.state_cursors[state_idx] = cur as u32;
        self.segments_hap[cur]
    }

    /// Reset cursors for all states (for new iteration pass)
    pub fn reset_cursors(&mut self) {
        self.state_cursors.copy_from_slice(&self.state_heads);
    }

    /// Materialize haplotypes for a single marker without mutating cursors.
    ///
    /// This method takes `&self` (immutable) since it walks the segment linked lists
    /// from scratch without using the internal cursors. O(segments) per call.
    ///
    /// Use this for sparse access patterns (Stage 2). For dense access across all
    /// markers, prefer `materialize_all()` which is O(markers + segments) total.
    #[inline]
    pub fn materialize_at(&self, marker: usize, out: &mut [u32]) {
        let n_states = self.state_heads.len();
        assert!(out.len() >= n_states);

        for state_idx in 0..n_states {
            let mut cur = self.state_heads[state_idx] as usize;

            // Walk linked list to find segment containing marker
            while marker >= self.segments_end[cur] as usize {
                let next = self.segments_next[cur];
                if next == Self::NIL {
                    break;
                }
                cur = next as usize;
            }

            out[state_idx] = self.segments_hap[cur];
        }
    }

    /// Fill allele array with marker-major iteration order.
    ///
    /// This variant processes markers in the outer loop, allowing the caller
    /// to hoist per-marker computations (like alignment lookups) outside
    /// the state loop for better performance.
    ///
    /// # Arguments
    /// * `out` - Output allele array of size n_markers * n_states
    /// * `per_marker` - Called once per marker, returns a closure that maps hap_idx -> allele
    #[inline]
    pub fn fill_alleles_marker_major<F, G>(&self, out: &mut [u8], mut per_marker: F)
    where
        F: FnMut(usize) -> G,
        G: Fn(u32) -> u8,
    {
        let n_states = self.state_heads.len();
        let n_markers = self.n_markers;

        // Initialize cursors for all states
        let mut cursors: Vec<usize> = (0..n_states)
            .map(|s| self.state_heads[s] as usize)
            .collect();
        let mut seg_ends: Vec<usize> = cursors.iter()
            .map(|&c| self.segments_end[c] as usize)
            .collect();
        let mut haps: Vec<u32> = cursors.iter()
            .map(|&c| self.segments_hap[c])
            .collect();

        for m in 0..n_markers {
            // Get the hap-to-allele function for this marker (allows hoisting per-marker work)
            let to_allele = per_marker(m);

            let base = m * n_states;
            for state_idx in 0..n_states {
                // Advance cursor if needed
                while m >= seg_ends[state_idx] {
                    let next = self.segments_next[cursors[state_idx]];
                    if next == Self::NIL {
                        break;
                    }
                    cursors[state_idx] = next as usize;
                    seg_ends[state_idx] = self.segments_end[cursors[state_idx]] as usize;
                    haps[state_idx] = self.segments_hap[cursors[state_idx]];
                }
                out[base + state_idx] = to_allele(haps[state_idx]);
            }
        }
    }
}

// ============================================================================
// StateSwitch: Event Record for Backward Pass Rewinding
// ============================================================================

/// Record of a state switch for backward pass "rewinding".
///
/// During the forward pass, whenever a state crosses a segment boundary,
/// we record the switch. During the backward pass, we replay these events
/// in reverse to restore the cursor to its previous state.
///
/// This approach is O(switches) memory vs O(markers × states) for snapshots.
#[derive(Debug, Clone, Copy)]
pub struct StateSwitch {
    /// Marker index where switch occurred (inclusive start of new segment)
    pub marker: u32,
    /// State index that switched
    pub state_idx: u32,
    /// Haplotype index *before* the switch (for restoration)
    pub old_hap: u32,
}

// ============================================================================
// MosaicCursor: SIMD-Friendly State Access for HMM Hot Path
// ============================================================================

/// High-performance cursor for HMM forward-backward loops.
///
/// Separates state management from math kernel:
/// - **Phase A**: State maintenance (integer logic, branch-predictable)
/// - **Phase B**: Allele materialization (memory fetch into scratch buffer)
/// - **Phase C**: Math kernel (SIMD on contiguous data)
#[derive(Clone, Debug)]
pub struct MosaicCursor {
    /// Current active haplotype index for each state
    active_haps: Vec<u32>,
    /// Marker index where each state switches to next segment
    next_switch: Vec<usize>,
    /// Current segment arena index for each state
    cursor_indices: Vec<u32>,
}

impl MosaicCursor {
    /// Create a cursor from ThreadedHaps
    pub fn from_threaded(th: &ThreadedHaps) -> Self {
        let n = th.n_states();
        let mut active_haps = Vec::with_capacity(n);
        let mut next_switch = Vec::with_capacity(n);
        let mut cursor_indices = Vec::with_capacity(n);

        for state in 0..n {
            let head = th.state_heads[state] as usize;
            active_haps.push(th.segments_hap[head]);
            next_switch.push(th.segments_end[head] as usize);
            cursor_indices.push(th.state_heads[state]);
        }

        Self {
            active_haps,
            next_switch,
            cursor_indices,
        }
    }

    /// Get slice of all active haplotypes (for SIMD materialization)
    #[inline]
    pub fn active_haps(&self) -> &[u32] {
        &self.active_haps
    }

    fn advance_state(&mut self, state: usize, marker: usize, th: &ThreadedHaps) {
        let mut cur = self.cursor_indices[state] as usize;

        loop {
            let next = th.segments_next[cur];
            if next == ThreadedHaps::NIL {
                break;
            }
            cur = next as usize;

            if marker < th.segments_end[cur] as usize {
                break;
            }
        }

        self.cursor_indices[state] = cur as u32;
        self.active_haps[state] = th.segments_hap[cur];
        self.next_switch[state] = th.segments_end[cur] as usize;
    }

    /// Phase A with history: Advance to marker, recording state switches.
    ///
    /// Same as `advance_to_marker` but pushes `StateSwitch` events onto the
    /// history stack whenever a state crosses a segment boundary. This enables
    /// efficient backward pass rewinding without O(M×K) snapshots.
    #[inline]
    pub fn advance_with_history(
        &mut self,
        marker: usize,
        th: &ThreadedHaps,
        history: &mut Vec<StateSwitch>,
    ) {
        for state in 0..self.active_haps.len() {
            if marker >= self.next_switch[state] {
                // Record the switch BEFORE updating
                history.push(StateSwitch {
                    marker: self.next_switch[state] as u32,
                    state_idx: state as u32,
                    old_hap: self.active_haps[state],
                });
                self.advance_state(state, marker, th);
            }
        }
    }

    /// Rewind cursor to a previous marker by replaying history in reverse.
    ///
    /// Pops all events that occurred after `target_marker`, restoring
    /// `active_haps` to the state it was in at that marker.
    ///
    /// Note: This only restores `active_haps`, not internal cursor pointers.
    /// This is safe because the backward pass only reads `active_haps`.
    #[inline]
    pub fn rewind(&mut self, target_marker: usize, history: &mut Vec<StateSwitch>) {
        while let Some(event) = history.last() {
            if event.marker as usize > target_marker {
                let event = history.pop().unwrap();
                self.active_haps[event.state_idx as usize] = event.old_hap;
            } else {
                break;
            }
        }
    }

    /// Reset cursor to the initial state (marker 0).
    #[cfg(test)]
    pub fn reset(&mut self, th: &ThreadedHaps) {
        for state in 0..th.n_states() {
            let head = th.state_heads[state] as usize;
            self.active_haps[state] = th.segments_hap[head];
            self.next_switch[state] = th.segments_end[head] as usize;
            self.cursor_indices[state] = th.state_heads[state];
        }
    }
}

/// Scratch buffer for Phase B: allele materialization
#[derive(Clone, Debug)]
pub struct AlleleScratch {
    /// Alleles for each state at current marker
    pub alleles: Vec<u8>,
}

impl AlleleScratch {
    /// Create scratch buffer for n_states
    pub fn new(n_states: usize) -> Self {
        Self {
            alleles: vec![0; n_states],
        }
    }

    /// Phase B: Materialize alleles for current marker using cursor
    #[inline]
    pub fn materialize<F>(&mut self, cursor: &MosaicCursor, marker: usize, get_allele: F)
    where
        F: Fn(usize, u32) -> u8,
    {
        for (state, &hap) in cursor.active_haps.iter().enumerate() {
            self.alleles[state] = get_allele(marker, hap);
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threaded_haps_basic() {
        let mut th = ThreadedHaps::new(4, 16, 100);

        th.push_new(10);
        th.push_new(20);

        assert_eq!(th.n_states(), 2);
        assert_eq!(th.hap_at_raw(0, 0), 10);
        assert_eq!(th.hap_at_raw(1, 50), 20);
    }

    #[test]
    fn test_threaded_haps_segments() {
        let mut th = ThreadedHaps::new(2, 8, 100);

        th.push_new(10);
        th.add_segment(0, 15, 50);
        th.push_new(20);

        assert_eq!(th.hap_at_raw(0, 25), 10);
        assert_eq!(th.hap_at_raw(0, 75), 15);

        th.reset_cursors();
        assert_eq!(th.hap_at_raw(0, 10), 10);
        assert_eq!(th.hap_at_raw(0, 60), 15);
    }

    #[test]
    fn test_mosaic_cursor_threaded() {
        let mut th = ThreadedHaps::new(2, 8, 100);

        th.push_new(10);
        th.add_segment(0, 15, 50);
        th.push_new(20);

        let mut cursor = MosaicCursor::from_threaded(&th);
        let mut history = Vec::new();

        assert_eq!(cursor.active_haps()[0], 10);
        assert_eq!(cursor.active_haps()[1], 20);

        cursor.advance_with_history(25, &th, &mut history);
        assert_eq!(cursor.active_haps()[0], 10);

        cursor.advance_with_history(60, &th, &mut history);
        assert_eq!(cursor.active_haps()[0], 15);
        assert_eq!(cursor.active_haps()[1], 20);

        cursor.reset(&th);
        assert_eq!(cursor.active_haps()[0], 10);
    }

    #[test]
    fn test_allele_scratch() {
        let mut th = ThreadedHaps::new(3, 3, 100);
        th.push_new(0);
        th.push_new(1);
        th.push_new(2);

        let cursor = MosaicCursor::from_threaded(&th);
        let mut scratch = AlleleScratch::new(3);

        scratch.materialize(&cursor, 5, |_, h| h as u8);

        assert_eq!(scratch.alleles[0], 0);
        assert_eq!(scratch.alleles[1], 1);
        assert_eq!(scratch.alleles[2], 2);
    }

    #[test]
    fn test_mosaic_cursor_rewind() {
        // Create a mosaic setup with 2 states, each having segment boundaries
        let mut th = ThreadedHaps::new(2, 8, 100);
        
        // State 0: hap 10 for [0, 40), hap 15 for [40, 70), hap 20 for [70, 100)
        th.push_new(10);
        th.add_segment(0, 15, 40);
        th.add_segment(0, 20, 70);
        
        // State 1: hap 30 for [0, 50), hap 35 for [50, 100)
        th.push_new(30);
        th.add_segment(1, 35, 50);
        
        let mut cursor = MosaicCursor::from_threaded(&th);
        let mut history: Vec<StateSwitch> = Vec::new();
        
        // Advance through all markers recording history
        for m in 0..100 {
            cursor.advance_with_history(m, &th, &mut history);
        }
        
        // Verify cursor is at end state
        assert_eq!(cursor.active_haps()[0], 20); // State 0 at marker 99
        assert_eq!(cursor.active_haps()[1], 35); // State 1 at marker 99
        
        // History should have recorded 3 switches total:
        // - State 0 switched at marker 40, 70
        // - State 1 switched at marker 50
        assert_eq!(history.len(), 3);
        
        // Now test rewinding
        // Rewind to marker 80 - should stay at hap 20, 35
        cursor.rewind(80, &mut history);
        assert_eq!(cursor.active_haps()[0], 20);
        assert_eq!(cursor.active_haps()[1], 35);
        assert_eq!(history.len(), 3); // No events popped
        
        // Rewind to marker 60 - state 1 stays at 35, state 0 reverts to 15
        cursor.rewind(60, &mut history);
        assert_eq!(cursor.active_haps()[0], 15);
        assert_eq!(cursor.active_haps()[1], 35);
        assert_eq!(history.len(), 2); // One event popped (state 0's switch at 70)
        
        // Rewind to marker 45 - state 1 reverts to 30, state 0 stays at 15
        cursor.rewind(45, &mut history);
        assert_eq!(cursor.active_haps()[0], 15);
        assert_eq!(cursor.active_haps()[1], 30);
        assert_eq!(history.len(), 1); // Two events popped total
        
        // Rewind to marker 30 - state 0 reverts to 10, state 1 stays at 30
        cursor.rewind(30, &mut history);
        assert_eq!(cursor.active_haps()[0], 10);
        assert_eq!(cursor.active_haps()[1], 30);
        assert_eq!(history.len(), 0); // All events popped
    }

    #[test]
    fn test_materialize_at() {
        let mut th = ThreadedHaps::new(3, 8, 100);

        // State 0: hap 10 for [0, 50), hap 15 for [50, 100)
        th.push_new(10);
        th.add_segment(0, 15, 50);

        // State 1: hap 20 for all markers
        th.push_new(20);

        // State 2: hap 30 for [0, 25), hap 35 for [25, 100)
        th.push_new(30);
        th.add_segment(2, 35, 25);

        let mut buffer = vec![0u32; 3];

        // Test at marker 10 - before any segment transitions
        th.materialize_at(10, &mut buffer);
        assert_eq!(buffer[0], 10);
        assert_eq!(buffer[1], 20);
        assert_eq!(buffer[2], 30);

        // Test at marker 30 - after state 2's transition
        th.materialize_at(30, &mut buffer);
        assert_eq!(buffer[0], 10);
        assert_eq!(buffer[1], 20);
        assert_eq!(buffer[2], 35);

        // Test at marker 60 - after state 0's transition
        th.materialize_at(60, &mut buffer);
        assert_eq!(buffer[0], 15);
        assert_eq!(buffer[1], 20);
        assert_eq!(buffer[2], 35);

        // Reset and test again to verify cursor handling
        th.reset_cursors();
        th.materialize_at(5, &mut buffer);
        assert_eq!(buffer[0], 10);
        assert_eq!(buffer[1], 20);
        assert_eq!(buffer[2], 30);
    }
}
