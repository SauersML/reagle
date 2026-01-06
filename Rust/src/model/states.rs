//! # HMM State Management
//!
//! Provides efficient storage and access for composite HMM states.
//! This matches Java `imp/ImpStates.java` mosaic haplotype logic.
//!
//! Key concepts:
//! - States are "composite haplotypes" - mosaics of reference haplotype segments
//! - `ThreadedHaps` stores segments in a linked-list arena for O(1) updates
//! - `MosaicCursor` provides SIMD-friendly state access for the HMM hot path

use crate::data::haplotype::HapIdx;

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

    /// Create ThreadedHaps from a static list of haplotypes (one segment per state)
    ///
    /// This is used when states are selected via PBWT and remain constant
    /// for the entire forward-backward run (e.g., in phasing).
    pub fn from_static_haps(haps: &[HapIdx], n_markers: usize) -> Self {
        let n_states = haps.len();
        let mut th = Self::new(n_states, n_states, n_markers);
        for &hap in haps {
            th.push_new(hap.0);
        }
        th
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
    #[inline]
    pub fn hap_at_raw(&mut self, state_idx: usize, marker: usize) -> u32 {
        let mut cur = self.state_cursors[state_idx] as usize;

        while marker >= self.segments_end[cur] as usize {
            let next = self.segments_next[cur];
            if next == Self::NIL {
                break;
            }
            cur = next as usize;
        }

        self.state_cursors[state_idx] = cur as u32;
        self.segments_hap[cur]
    }

    /// Reset cursors for all states (for new iteration pass)
    pub fn reset_cursors(&mut self) {
        self.state_cursors.copy_from_slice(&self.state_heads);
    }
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

    /// Phase A: Advance to marker, updating any states that need to switch.
    #[inline]
    pub fn advance_to_marker(&mut self, marker: usize, th: &ThreadedHaps) {
        for state in 0..self.active_haps.len() {
            if marker >= self.next_switch[state] {
                self.advance_state(state, marker, th);
            }
        }
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

    /// Reset cursor to beginning (for backward pass or new iteration)
    pub fn reset(&mut self, th: &ThreadedHaps) {
        for state in 0..self.active_haps.len() {
            let head = th.state_heads[state] as usize;
            self.cursor_indices[state] = head as u32;
            self.active_haps[state] = th.segments_hap[head];
            self.next_switch[state] = th.segments_end[head] as usize;
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

        assert_eq!(cursor.active_haps()[0], 10);
        assert_eq!(cursor.active_haps()[1], 20);

        cursor.advance_to_marker(25, &th);
        assert_eq!(cursor.active_haps()[0], 10);

        cursor.advance_to_marker(60, &th);
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

        scratch.materialize(&cursor, 5, |marker, h| if marker > 0 { h as u8 } else { h as u8 });

        assert_eq!(scratch.alleles[0], 0);
        assert_eq!(scratch.alleles[1], 1);
        assert_eq!(scratch.alleles[2], 2);
    }
}
