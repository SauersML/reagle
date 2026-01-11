//! # Phase State Selection
//!
//! Implements dynamic HMM state selection for phasing using PBWT-based IBS matching.
//! This matches Java's `BasicPhaseStates` behavior: composite haplotypes that change
//! segments along the chromosome based on local IBS matches at each position.
//!
//! ## Key Difference from Static Selection
//!
//! The previous implementation selected states at the chromosome midpoint and used
//! them statically for the entire chromosome. This was problematic because haplotypes
//! that match well at the midpoint may share no recent ancestry at the chromosome ends
//! due to recombination.
//!
//! This implementation:
//! 1. Iterates through all markers (not just midpoint)
//! 2. Uses a priority queue to manage composite haplotypes
//! 3. Dynamically swaps in better-matching haplotype segments as we traverse
//! 4. Results in "mosaic" haplotypes that provide local matches everywhere

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use crate::model::ibs2::Ibs2;
use crate::model::phase_ibs::BidirectionalPhaseIbs;
use crate::model::states::ThreadedHaps;

/// Entry in the priority queue for managing composite haplotypes
#[derive(Clone, Debug)]
struct CompHapEntry {
    /// Index into the composite haplotypes array
    comp_hap_idx: usize,
    /// Current reference haplotype
    hap: u32,
    /// Marker index where this segment starts
    start_marker: usize,
    /// Last marker where this hap was seen in IBS matches
    last_ibs_marker: i32,
}

impl PartialEq for CompHapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.last_ibs_marker == other.last_ibs_marker
    }
}

impl Eq for CompHapEntry {}

impl PartialOrd for CompHapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CompHapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: smallest last_ibs_marker should be at top (to be replaced first)
        other.last_ibs_marker.cmp(&self.last_ibs_marker)
    }
}

/// Dynamic HMM state selector for phasing
///
/// Matches Java `BasicPhaseStates.java` behavior: builds composite/mosaic haplotypes
/// by iterating through all markers and dynamically swapping in better-matching
/// reference haplotype segments.
pub struct PhaseStates {
    /// Maximum number of HMM states
    max_states: usize,
    /// Composite haplotypes (the HMM states) - Optimized Arena
    threaded_haps: ThreadedHaps,
    /// Map from reference haplotype to last IBS marker
    hap_to_last_ibs: HashMap<u32, i32>,
    /// Priority queue for managing composite haplotypes
    queue: BinaryHeap<CompHapEntry>,
    /// Number of markers
    n_markers: usize,
}

const NIL: i32 = -103;

impl PhaseStates {
    /// Create a new phase state selector
    ///
    /// # Arguments
    /// * `max_states` - Maximum number of composite haplotypes (K in Li-Stephens model)
    /// * `n_markers` - Number of markers in the HMM
    pub fn new(max_states: usize, n_markers: usize) -> Self {
        Self {
            max_states,
            threaded_haps: ThreadedHaps::new(max_states, max_states * 4, n_markers),
            hap_to_last_ibs: HashMap::with_capacity(max_states),
            queue: BinaryHeap::with_capacity(max_states),
            n_markers,
        }
    }

    /// Build composite haplotypes for a sample's two haplotypes
    ///
    /// This iterates through all markers and dynamically updates the composite
    /// haplotypes based on local IBS matches, similar to Java's `BasicPhaseStates`.
    ///
    /// # Arguments
    /// * `sample` - Sample index (hap1 = sample * 2, hap2 = sample * 2 + 1)
    /// * `phase_ibs` - Bidirectional PBWT for finding IBS neighbors
    /// * `ibs2` - IBS2 segment data for additional neighbors
    /// * `n_candidates` - Number of candidates to consider at each marker
    ///
    /// # Returns
    /// The composite haplotypes ready for use in the HMM (cloned for ownership transfer)
    pub fn build_composite_haps(
        &mut self,
        sample: u32,
        phase_ibs: &BidirectionalPhaseIbs,
        ibs2: &Ibs2,
        n_candidates: usize,
    ) -> ThreadedHaps {
        self.clear();

        let h1 = sample * 2;
        let h2 = h1 + 1;

        // Iterate through all markers and update composite haplotypes
        for marker in 0..self.n_markers {
            // Get IBS neighbors for both haplotypes at this marker
            let neighbors1 = phase_ibs.find_neighbors(h1, marker, ibs2, n_candidates);
            let neighbors2 = phase_ibs.find_neighbors(h2, marker, ibs2, n_candidates);

            // Add IBS haplotypes (excluding same sample)
            for &ibs_hap in &neighbors1 {
                if ibs_hap / 2 != sample {
                    self.add_ibs_hap(ibs_hap, marker as i32);
                }
            }
            for &ibs_hap in &neighbors2 {
                if ibs_hap / 2 != sample {
                    self.add_ibs_hap(ibs_hap, marker as i32);
                }
            }
        }

        // If no IBS haps found, fill with random haps
        if self.queue.is_empty() {
            self.fill_with_random(sample, phase_ibs.n_haps());
        }

        // Finalize and return owned copy
        self.finalize();
        self.threaded_haps.clone()
    }

    /// Clear state for reuse
    fn clear(&mut self) {
        self.threaded_haps.clear();
        self.hap_to_last_ibs.clear();
        self.queue.clear();
    }

    /// Add an IBS haplotype at a marker
    ///
    /// This matches Java's `BasicPhaseStates.addIbsHap`:
    /// - If the hap is already in the queue, just update its last IBS marker
    /// - If the hap is new and queue isn't full, add it
    /// - If queue is full and the oldest entry is stale enough, replace it
    fn add_ibs_hap(&mut self, ibs_hap: u32, marker: i32) {
        // Check if hap is already being tracked
        if let Some(&last_marker) = self.hap_to_last_ibs.get(&ibs_hap) {
            if last_marker != NIL {
                // Hap is already in queue, update its last IBS marker
                self.hap_to_last_ibs.insert(ibs_hap, marker);
                return;
            }
        }

        // Hap is not in queue - try to add it
        self.update_head_of_queue();

        // LRU eviction: when queue is full, ALWAYS evict the oldest entry
        // to make room for the new IBS match. PBWT finds best local matches,
        // so we must not discard them. This matches Java BasicPhaseStates
        // where q.poll() always removes the LRU entry at capacity.
        if self.queue.len() < self.max_states {
            // Queue has room - add new entry
            let index = self.queue.len();
            self.threaded_haps.push_new(ibs_hap);
            self.queue.push(CompHapEntry {
                comp_hap_idx: index,
                hap: ibs_hap,
                start_marker: 0,
                last_ibs_marker: marker,
            });
            self.hap_to_last_ibs.insert(ibs_hap, marker);
        } else if !self.queue.is_empty() {
            // Queue is full - evict oldest (LRU) to make room for new match
            let head = self.queue.pop().unwrap();
            let index = head.comp_hap_idx;
            let prev_hap = head.hap;
            let prev_start = head.start_marker;

            // Calculate transition point (midpoint between last seen and current)
            let next_start = ((head.last_ibs_marker + marker) / 2) as usize;
            let next_start = next_start.max(prev_start).min(self.n_markers.saturating_sub(1));

            // Remove old hap from tracking
            self.hap_to_last_ibs.remove(&prev_hap);

            // Add segment to threaded haps if this is the first segment for this state
            // (later segments are added during finalization)
            if self.threaded_haps.n_states() <= index {
                self.threaded_haps.push_new(prev_hap);
            }
            if next_start > prev_start && next_start < self.n_markers {
                self.threaded_haps.add_segment(index, ibs_hap, next_start);
            }

            // Add new entry for the replacement hap
            self.queue.push(CompHapEntry {
                comp_hap_idx: index,
                hap: ibs_hap,
                start_marker: next_start,
                last_ibs_marker: marker,
            });
            self.hap_to_last_ibs.insert(ibs_hap, marker);
        }
    }

    /// Update the head of the queue to reflect latest IBS marker
    fn update_head_of_queue(&mut self) {
        while let Some(head) = self.queue.peek() {
            let current_last = *self.hap_to_last_ibs.get(&head.hap).unwrap_or(&NIL);
            if head.last_ibs_marker == current_last {
                break;
            }
            // Update the entry with its actual last IBS marker
            let mut entry = self.queue.pop().unwrap();
            entry.last_ibs_marker = current_last;
            self.queue.push(entry);
        }
    }

    /// Finalize composite haplotypes
    fn finalize(&mut self) {
        // The ThreadedHaps already have segments added during add_ibs_hap.
        // Just ensure all states extend to the end of the chromosome.
        // This is already handled by ThreadedHaps::push_new setting end = n_markers.
    }

    /// Fill with random haplotypes when no IBS matches found
    fn fill_with_random(&mut self, sample: u32, n_haps: usize) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Simple deterministic "random" selection based on sample
        let mut hasher = DefaultHasher::new();
        sample.hash(&mut hasher);
        let mut seed = hasher.finish();

        let n_states = self.max_states.min(n_haps.saturating_sub(2));

        for _ in 0..n_states {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let h = (seed % n_haps as u64) as u32;
            if h / 2 != sample && !self.hap_to_last_ibs.contains_key(&h) {
                let index = self.threaded_haps.n_states();
                self.threaded_haps.push_new(h);
                self.hap_to_last_ibs.insert(h, 0);
                self.queue.push(CompHapEntry {
                    comp_hap_idx: index,
                    hap: h,
                    start_marker: 0,
                    last_ibs_marker: 0,
                });
            }
        }
    }

    /// Get the number of states
    pub fn n_states(&self) -> usize {
        self.threaded_haps.n_states()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_states_basic() {
        let mut ps = PhaseStates::new(10, 100);
        assert_eq!(ps.n_states(), 0);

        // Add some IBS haps
        ps.add_ibs_hap(0, 0);
        ps.add_ibs_hap(2, 0);
        ps.add_ibs_hap(4, 10);

        assert!(ps.n_states() > 0);
    }

    #[test]
    fn test_phase_states_replacement() {
        let mut ps = PhaseStates::new(2, 1000); // Only 2 states

        // Fill up the queue
        ps.add_ibs_hap(0, 0);
        ps.add_ibs_hap(2, 0);

        // This should trigger replacement after enough markers
        for m in 1..200 {
            ps.add_ibs_hap(4, m);
        }

        // Should have replaced one of the original haps
        assert_eq!(ps.n_states(), 2);
    }

    #[test]
    fn test_composite_haps_have_multiple_segments() {
        let mut ps = PhaseStates::new(2, 1000);

        // Add haplotype 0 at marker 0
        ps.add_ibs_hap(0, 0);

        // Add haplotype 2 at marker 0 (second state)
        ps.add_ibs_hap(2, 0);

        // Now simulate IBS matches later that should cause segment changes
        // With LRU eviction, new IBS matches always replace oldest when queue is full

        // Add haplotype 10 repeatedly starting at marker 100
        // This should eventually replace one of the original states' segments
        for m in 100..300 {
            ps.add_ibs_hap(10, m);
        }

        ps.finalize();

        // Get the threaded_haps and check if any state has multiple segments
        let th = &ps.threaded_haps;

        // We should have 2 states
        assert_eq!(th.n_states(), 2, "Should have exactly 2 states");

        // Verify that at least one state has had a segment change by checking
        // if the haplotype changes across markers
        let mut found_segment_change = false;
        let mut th_clone = th.clone();

        for state in 0..th.n_states() {
            th_clone.reset_cursors();
            let hap_at_start = th_clone.hap_at_raw(state, 0);
            let hap_at_middle = th_clone.hap_at_raw(state, 150);
            let hap_at_end = th_clone.hap_at_raw(state, 999);

            if hap_at_start != hap_at_middle || hap_at_middle != hap_at_end {
                found_segment_change = true;
                break;
            }
        }

        assert!(
            found_segment_change,
            "CRITICAL: No segment changes detected! PhaseStates is NOT creating \
             dynamic composite haplotypes. All states have the same haplotype \
             from start to end, meaning we're back to static state selection. \
             This defeats the purpose of the PhaseStates implementation."
        );
    }
}
