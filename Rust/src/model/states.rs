//! # HMM State Management
//!
//! Provides efficient storage and access for composite HMM states.
//! This matches Java `imp/ImpStates.java` mosaic haplotype logic.
//!
//! Key concepts:
//! - States are "composite haplotypes" - mosaics of reference haplotype segments
//! - `ThreadedHaps` stores segments in a linked-list arena for O(1) updates
//! - `StateProvider` trait abstracts state access for the HMM

use crate::data::haplotype::HapIdx;

/// Provides reference haplotype indices for HMM states at specific markers.
///
/// This trait abstracts the state lookup to support both:
/// - Static states (same haplotype for entire chromosome)
/// - Dynamic/mosaic states (different haplotypes at different positions)
pub trait StateProvider {
    /// Number of HMM states
    fn n_states(&self) -> usize;

    /// Get haplotype index for state k at marker m (O(1) amortized)
    fn hap_at(&mut self, state_idx: usize, marker_idx: usize) -> HapIdx;

    /// Reset cursors for a new forward/backward pass
    fn reset_cursors(&mut self);
}

/// A segment of a composite haplotype
#[derive(Clone, Debug)]
pub struct HapSegment {
    /// Reference haplotype index for this segment
    pub hap: u32,
    /// End marker (exclusive) for this segment
    pub end: usize,
}

/// Arena-based storage for composite haplotypes using threaded indices.
///
/// This avoids the O(N) shift cost of standard SoA insertion and the
/// heap fragmentation of `Vec<CompositeHap>`.
///
/// # Memory Layout
/// Segments are stored in a global arena (vectors). Each composite haplotype
/// is a linked list of segment indices.
///
/// - `segments_hap`: Haplotype ID for the segment
/// - `segments_end`: End marker for the segment
/// - `segments_next`: Index of the next segment in the arena (or u32::MAX)
#[derive(Clone, Debug)]
pub struct ThreadedHaps {
    // --- Arena Storage ---
    segments_hap: Vec<u32>,
    segments_end: Vec<u32>,
    segments_next: Vec<u32>,

    // --- State Pointers ---
    // Index of the first segment for each state
    state_heads: Vec<u32>,
    // Index of the last segment for each state (for O(1) append)
    state_tails: Vec<u32>,
    // Current iteration cursor for each state
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

    /// Initialize a new state starting with given hap
    /// Returns the state index
    pub fn push_new(&mut self, start_hap: u32) -> usize {
        let state_idx = self.state_heads.len();

        // Create first segment
        let seg_idx = self.segments_hap.len() as u32;
        self.segments_hap.push(start_hap);
        self.segments_end.push(self.n_markers as u32);
        self.segments_next.push(Self::NIL);

        // Register state
        self.state_heads.push(seg_idx);
        self.state_tails.push(seg_idx);
        self.state_cursors.push(seg_idx);

        state_idx
    }

    /// Add a segment to an existing state
    ///
    /// # Arguments
    /// * `state_idx` - Index of the state to extend
    /// * `hap` - Reference haplotype for the new segment
    /// * `start_marker` - Start marker of the new segment (truncates previous segment)
    pub fn add_segment(&mut self, state_idx: usize, hap: u32, start_marker: usize) {
        let tail_idx = self.state_tails[state_idx] as usize;

        // Truncate previous segment
        self.segments_end[tail_idx] = start_marker as u32;

        // Create new segment
        let new_seg_idx = self.segments_hap.len() as u32;
        self.segments_hap.push(hap);
        self.segments_end.push(self.n_markers as u32);
        self.segments_next.push(Self::NIL);

        // Link
        self.segments_next[tail_idx] = new_seg_idx;
        self.state_tails[state_idx] = new_seg_idx;
    }

    /// Get haplotype at marker for a state (advances cursor)
    #[inline]
    pub fn hap_at_raw(&mut self, state_idx: usize, marker: usize) -> u32 {
        let mut cur = self.state_cursors[state_idx] as usize;

        // Advance while marker is past this segment's end
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

impl StateProvider for ThreadedHaps {
    fn n_states(&self) -> usize {
        self.state_heads.len()
    }

    fn hap_at(&mut self, state_idx: usize, marker_idx: usize) -> HapIdx {
        HapIdx::new(self.hap_at_raw(state_idx, marker_idx))
    }

    fn reset_cursors(&mut self) {
        ThreadedHaps::reset_cursors(self)
    }
}

/// Static state provider for simple cases (same haplotype for entire chromosome)
///
/// This is useful for testing or when a fixed set of reference haplotypes
/// is used without mosaic switching.
#[derive(Clone, Debug)]
pub struct StaticStates {
    /// State-to-haplotype mapping (constant across all markers)
    haps: Vec<HapIdx>,
}

impl StaticStates {
    /// Create from a fixed list of haplotype indices
    pub fn new(haps: Vec<HapIdx>) -> Self {
        Self { haps }
    }
}

impl StateProvider for StaticStates {
    fn n_states(&self) -> usize {
        self.haps.len()
    }

    #[allow(unused_variables)]
    fn hap_at(&mut self, state_idx: usize, marker_idx: usize) -> HapIdx {
        self.haps[state_idx]
    }

    fn reset_cursors(&mut self) {
        // No-op for static states
    }
}

// ============================================================================
// MosaicCursor: SIMD-Friendly State Access for HMM Hot Path
// ============================================================================

/// High-performance cursor for HMM forward-backward loops.
///
/// The key optimization is separating:
/// - **Phase A**: State maintenance (integer logic, branch-predictable)
/// - **Phase B**: Allele materialization (memory fetch into scratch buffer)
/// - **Phase C**: Math kernel (SIMD on contiguous data)
///
/// This allows the math kernel to operate on flat, pre-fetched data,
/// enabling auto-vectorization (AVX2/AVX-512).
#[derive(Clone, Debug)]
pub struct MosaicCursor {
    /// Current active haplotype index for each state
    active_haps: Vec<u32>,
    /// Marker index where each state switches to next segment (or usize::MAX if no more)
    next_switch: Vec<usize>,
    /// Current segment arena index for each state (into ThreadedHaps)
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

    /// Create a cursor from static states (no switching ever happens)
    pub fn from_static(states: &StaticStates) -> Self {
        let n = states.haps.len();
        Self {
            active_haps: states.haps.iter().map(|h| h.0).collect(),
            next_switch: vec![usize::MAX; n], // Never switch
            cursor_indices: vec![0; n],       // Not used for static
        }
    }

    /// Number of states
    #[inline]
    pub fn n_states(&self) -> usize {
        self.active_haps.len()
    }

    /// Get the current active haplotype for a state (no cursor movement)
    #[inline]
    pub fn active_hap(&self, state: usize) -> u32 {
        self.active_haps[state]
    }

    /// Get slice of all active haplotypes (for SIMD materialization)
    #[inline]
    pub fn active_haps(&self) -> &[u32] {
        &self.active_haps
    }

    /// Phase A: Advance to marker, updating any states that need to switch.
    /// Returns true if any state switched (rare, cold path).
    ///
    /// This should be called BEFORE materializing alleles for marker `m`.
    #[inline]
    pub fn advance_to_marker(&mut self, marker: usize, th: &ThreadedHaps) -> bool {
        let mut any_switch = false;

        for state in 0..self.active_haps.len() {
            if marker >= self.next_switch[state] {
                // Cold path: state needs to switch to next segment
                self.advance_state(state, marker, th);
                any_switch = true;
            }
        }

        any_switch
    }

    /// Advance a single state to the segment containing `marker`
    fn advance_state(&mut self, state: usize, marker: usize, th: &ThreadedHaps) {
        let mut cur = self.cursor_indices[state] as usize;

        loop {
            let next = th.segments_next[cur];
            if next == ThreadedHaps::NIL {
                // No more segments, stay at current
                break;
            }
            cur = next as usize;

            if marker < th.segments_end[cur] as usize {
                // Found the segment containing this marker
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
///
/// Pre-allocated buffer to hold alleles for current marker,
/// enabling SIMD-friendly access in the math kernel.
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
    ///
    /// Fetches reference alleles into contiguous buffer for SIMD math.
    #[inline]
    pub fn materialize<F>(&mut self, cursor: &MosaicCursor, marker: usize, get_allele: F)
    where
        F: Fn(usize, u32) -> u8, // (marker, hap) -> allele
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
    fn test_static_states() {
        let haps = vec![HapIdx::new(0), HapIdx::new(5), HapIdx::new(10)];
        let mut states = StaticStates::new(haps);

        assert_eq!(states.n_states(), 3);
        assert_eq!(states.hap_at(0, 0), HapIdx::new(0));
        assert_eq!(states.hap_at(1, 100), HapIdx::new(5));
        assert_eq!(states.hap_at(2, 999), HapIdx::new(10));
    }

    #[test]
    fn test_threaded_haps_basic() {
        let mut th = ThreadedHaps::new(4, 16, 100);

        // Add 2 states
        th.push_new(10);
        th.push_new(20);

        assert_eq!(th.n_states(), 2);
        assert_eq!(th.hap_at_raw(0, 0), 10);
        assert_eq!(th.hap_at_raw(1, 50), 20);
    }

    #[test]
    fn test_threaded_haps_segments() {
        let mut th = ThreadedHaps::new(2, 8, 100);

        // State 0: hap 10 for markers 0-49, hap 15 for markers 50-99
        th.push_new(10);
        th.add_segment(0, 15, 50);

        // State 1: single segment
        th.push_new(20);

        // Query state 0
        assert_eq!(th.hap_at_raw(0, 25), 10);
        assert_eq!(th.hap_at_raw(0, 75), 15);

        // Reset and query again
        th.reset_cursors();
        assert_eq!(th.hap_at_raw(0, 10), 10);
        assert_eq!(th.hap_at_raw(0, 60), 15);
    }

    #[test]
    fn test_mosaic_cursor_static() {
        let haps = vec![HapIdx::new(5), HapIdx::new(10), HapIdx::new(15)];
        let states = StaticStates::new(haps);
        let cursor = MosaicCursor::from_static(&states);

        assert_eq!(cursor.n_states(), 3);
        assert_eq!(cursor.active_hap(0), 5);
        assert_eq!(cursor.active_hap(1), 10);
        assert_eq!(cursor.active_hap(2), 15);
    }

    #[test]
    fn test_mosaic_cursor_threaded() {
        let mut th = ThreadedHaps::new(2, 8, 100);

        // State 0: hap 10 for 0-49, hap 15 for 50-99
        th.push_new(10);
        th.add_segment(0, 15, 50);

        // State 1: hap 20 for entire range
        th.push_new(20);

        let mut cursor = MosaicCursor::from_threaded(&th);

        // Initial state
        assert_eq!(cursor.active_hap(0), 10);
        assert_eq!(cursor.active_hap(1), 20);

        // Advance to marker 25 - no switches needed
        let switched = cursor.advance_to_marker(25, &th);
        assert!(!switched);
        assert_eq!(cursor.active_hap(0), 10);

        // Advance to marker 60 - state 0 should switch
        let switched = cursor.advance_to_marker(60, &th);
        assert!(switched);
        assert_eq!(cursor.active_hap(0), 15);
        assert_eq!(cursor.active_hap(1), 20); // unchanged

        // Reset and verify
        cursor.reset(&th);
        assert_eq!(cursor.active_hap(0), 10);
    }

    #[test]
    fn test_allele_scratch() {
        let haps = vec![HapIdx::new(0), HapIdx::new(1), HapIdx::new(2)];
        let states = StaticStates::new(haps);
        let cursor = MosaicCursor::from_static(&states);
        let mut scratch = AlleleScratch::new(3);

        // Simulate reference panel: marker m, hap h -> h as allele
        #[allow(unused_variables)]
        scratch.materialize(&cursor, 5, |m, h| h as u8);

        assert_eq!(scratch.alleles[0], 0);
        assert_eq!(scratch.alleles[1], 1);
        assert_eq!(scratch.alleles[2], 2);
    }
}
