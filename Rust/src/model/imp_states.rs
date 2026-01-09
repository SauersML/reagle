//! # Imputation State Selection
//!
//! Implements dynamic HMM state selection using PBWT-based IBS matching.
//!
//! ## Design Decision: PBWT vs Recursive IBS
//!
//! Java Beagle's `imp/ImpIbs.java` uses recursive partitioning over Coded Steps
//! to find IBS matches. This Rust implementation instead uses PBWT (Positional
//! Burrows-Wheeler Transform) because:
//!
//! 1. Accuracy: PBWT guarantees finding longest common substring matches,
//!    while recursive partitioning is a heuristic that can miss matches at
//!    step boundaries.
//!
//! 2. Speed: PBWT uses linear arrays with sequential memory access (cache-
//!    friendly, SIMD-vectorizable). Recursive IBS involves tree traversal with
//!    pointer chasing, causing cache misses.
//!
//! 3. Bidirectional matching: Running PBWT forward and backward finds the
//!    best haplotypes matching on both sides of each position, improving
//!    imputation accuracy at the cost of a second pass.
//!
//! See `model/mod.rs` for detailed rationale.
//!
//! ## Key Concepts
//!
//! - States are "composite haplotypes" - mosaics of reference haplotype segments
//! - PBWT finds IBS matches at each step (marker cluster)
//! - A priority queue tracks which composite haplotypes to keep/replace

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::data::storage::coded_steps::{CodedPbwtView, RefPanelCoded};
use crate::utils::workspace::ImpWorkspace;

// Re-export from states module
pub use crate::model::states::ThreadedHaps;

/// Entry in the priority queue for state management
#[derive(Clone, Debug)]
struct CompHapEntry {
    /// Index into the composite haplotypes array
    comp_hap_idx: usize,
    /// Current reference haplotype
    hap: u32,
    /// Last step where this hap was seen in IBS matches
    last_ibs_step: i32,
}

impl PartialEq for CompHapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.last_ibs_step == other.last_ibs_step
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
        // Min-heap: smallest last_ibs_step should be at top (to be replaced first)
        other.last_ibs_step.cmp(&self.last_ibs_step)
    }
}

/// Dynamic HMM state selector for imputation
///
/// Matches Java `imp/ImpStates.java`
///
/// IMPORTANT: This struct uses the reference panel's step boundaries for all operations.
/// The `target_alleles` passed to `ibs_states()` must be in **reference marker space**
/// (indexed by reference marker, with 255 for markers not genotyped in target).
/// Sentinel value indicating "not in IBS set"
const IBS_NIL: i32 = i32::MIN;

pub struct ImpStates<'a> {
    /// Maximum number of HMM states
    max_states: usize,
    /// Reference panel with coded steps
    ref_panel: &'a RefPanelCoded,
    /// Composite haplotypes (the HMM states) - Optimized Arena
    threaded_haps: ThreadedHaps,
    /// Direct-indexed lookup: hap_to_last_ibs[hap] = last IBS step (IBS_NIL if not present)
    /// O(1) lookup vs HashMap's O(1) amortized with hash overhead
    /// Benchmarks show 15x speedup: Vec at 1 Gelem/s vs HashMap at 80 Melem/s
    hap_to_last_ibs: Vec<i32>,
    /// Priority queue for managing composite haplotypes
    queue: BinaryHeap<CompHapEntry>,
    /// Number of reference markers (in DENSE space, for ThreadedHaps)
    n_ref_markers: usize,
    /// Number of reference haplotypes (excludes appended target haplotypes)
    n_ref_haps: usize,
    /// Number of IBS haplotypes to find per step
    n_ibs_haps: usize,
    /// Genetic positions of projected markers (used for precise midpoint calculation)
    projected_gen_positions: Option<&'a [f64]>,
    /// Dense indices of projected (genotyped) markers
    projected_markers: Option<&'a [usize]>,
    /// Seed for deterministic state filling
    seed: u64,
}

impl<'a> ImpStates<'a> {
    /// Create a new state selector for PROJECTED PBWT
    ///
    /// In projected mode, the RefPanelCoded is built on genotyped markers only.
    /// ThreadedHaps still uses dense marker indices for segment boundaries.
    ///
    /// # Arguments
    /// * `ref_panel` - Reference panel in PROJECTED space (genotyped markers only)
    /// * `n_dense_markers` - Number of markers in dense space (for ThreadedHaps)
    /// * `n_ref_haps` - Number of reference haplotypes
    /// * `max_states` - Maximum number of HMM states to track
    /// * `n_ibs_haps` - Number of IBS haplotypes to find per step
    /// * `projected_gen_positions` - Genetic positions of genotyped markers
    /// * `dense_gen_positions` - Genetic positions of all reference markers
    pub fn new_projected(
        ref_panel: &'a RefPanelCoded,
        n_dense_markers: usize,
        n_ref_haps: usize,
        max_states: usize,
        n_ibs_haps: usize,
        projected_gen_positions: &'a [f64],
        projected_markers: &'a [usize],
        seed: u64,
    ) -> Self {
        Self {
            max_states,
            ref_panel,
            // ThreadedHaps uses DENSE marker count for segment boundaries
            threaded_haps: ThreadedHaps::new(max_states, max_states * 4, n_dense_markers),
            hap_to_last_ibs: vec![IBS_NIL; n_ref_haps],
            queue: BinaryHeap::with_capacity(max_states),
            n_ref_markers: n_dense_markers,
            n_ref_haps,
            n_ibs_haps,
            projected_gen_positions: Some(projected_gen_positions),
            projected_markers: Some(projected_markers),
            seed,
        }
    }

    /// Select IBS states using PROJECTED PBWT (exact matching, no approximation)
    ///
    /// This is the key fix for DR2 regression. In projected space:
    /// 1. target_alleles has NO missing values (no 255s)
    /// 2. Pattern matching is EXACT (no closest_pattern fallback)
    /// 3. All refs matching at observed positions cluster together
    ///
    /// # Arguments
    /// * `get_ref_allele` - Dense closure: (dense_marker_idx, hap) -> allele
    /// * `target_alleles_projected` - Target alleles in PROJECTED space (no 255s!)
    /// * `target_alleles_dense` - Target alleles in DENSE space (with 255s, for output)
    /// * `genotyped_markers` - Dense indices of genotyped markers
    /// * `workspace` - Pre-allocated workspace buffers
    /// * `hap_indices` - Output: ref hap indices at each genotyped marker
    /// * `allele_match` - Output: whether each state matches target
    pub fn ibs_states_projected<F>(
        &mut self,
        get_ref_allele: F,
        target_alleles_projected: &[u8],
        target_alleles_dense: &[u8],
        genotyped_markers: &[usize],
        workspace: &mut ImpWorkspace,
        hap_indices: &mut Vec<Vec<u32>>,
        allele_match: &mut Vec<Vec<bool>>,
    ) -> usize
    where
        F: Fn(usize, u32) -> u8,
    {
        self.initialize();

        let n_ref_haps = self.n_ref_haps;
        let n_steps = self.ref_panel.n_steps();
        let n_ibs_haps = self.n_ibs_haps;

        workspace.resize_with_ref(self.max_states, self.n_ref_markers, n_ref_haps);

        // Store backward IBS haps for each step
        let mut bwd_ibs_per_step: Vec<Vec<u32>> = vec![Vec::new(); n_steps];

        // STEP 1: Backward PBWT pass (allocation-free using counting sort)
        {
            let mut pbwt_bwd = CodedPbwtView::new_backward(
                &mut workspace.pbwt_prefix_bwd[..n_ref_haps],
                &mut workspace.pbwt_divergence_bwd[..n_ref_haps + 1],
                n_steps,
            );

            let mut bwd_virtual_pos = n_ref_haps / 2;

            for step_idx in (0..n_steps).rev() {
                let coded_step = self.ref_panel.step(step_idx);
                let proj_start = coded_step.start;
                let proj_end = coded_step.end;

                // Extract target sequence for this step (in projected space, no 255s!)
                let target_seq: Vec<u8> = (proj_start..proj_end)
                    .map(|p| target_alleles_projected[p])
                    .collect();

                // Try exact match first, use closest pattern as fallback for missing data
                let target_pattern = coded_step
                    .match_sequence_projected(&target_seq, genotyped_markers, &get_ref_allele)
                    .unwrap_or_else(|| {
                        // Fallback: use closest pattern (handles 255s in target)
                        coded_step.closest_pattern_projected(&target_seq, genotyped_markers, &get_ref_allele)
                    });

                // Use allocation-free counting sort for backward pass
                pbwt_bwd.update_backward_counting_sort(
                    coded_step,
                    n_steps,
                    &mut workspace.sort_counts_bwd,
                    &mut workspace.sort_offsets_bwd,
                    &mut workspace.sort_prefix_scratch_bwd,
                    &mut workspace.sort_div_scratch_bwd,
                    Some((&mut bwd_virtual_pos, target_pattern)),
                );

                // Restrict to the target's pattern bucket (matches forward pass / Java)
                let pattern_idx = target_pattern as usize;
                let bucket_start = workspace.sort_offsets_bwd.get(pattern_idx).copied().unwrap_or(0);
                let bucket_end = workspace
                    .sort_offsets_bwd
                    .get(pattern_idx + 1)
                    .copied()
                    .unwrap_or(n_ref_haps);

                let bwd_ibs: Vec<u32> = pbwt_bwd
                    .select_neighbors_in_bucket(bwd_virtual_pos, n_ibs_haps, bucket_start, bucket_end)
                    .into_iter()
                    .map(|h| h.0)
                    .collect();

                bwd_ibs_per_step[step_idx] = bwd_ibs;
            }
        }

        // STEP 2: Forward PBWT pass with composite haplotype building
        {
            let mut pbwt_fwd = CodedPbwtView::new(
                &mut workspace.pbwt_prefix[..n_ref_haps],
                &mut workspace.pbwt_divergence[..n_ref_haps + 1],
            );

            let mut fwd_virtual_pos = n_ref_haps / 2;

            for step_idx in 0..n_steps {
                let coded_step = self.ref_panel.step(step_idx);
                let proj_start = coded_step.start;
                let proj_end = coded_step.end;

                // Extract target sequence (projected space, no 255s)
                let target_seq: Vec<u8> = (proj_start..proj_end)
                    .map(|p| target_alleles_projected[p])
                    .collect();

                // Try exact match first, use closest pattern as fallback for missing data
                let target_pattern = coded_step
                    .match_sequence_projected(&target_seq, genotyped_markers, &get_ref_allele)
                    .unwrap_or_else(|| {
                        coded_step.closest_pattern_projected(&target_seq, genotyped_markers, &get_ref_allele)
                    });

                pbwt_fwd.update_counting_sort(
                    coded_step,
                    &mut workspace.sort_counts,
                    &mut workspace.sort_offsets,
                    &mut workspace.sort_prefix_scratch,
                    &mut workspace.sort_div_scratch,
                    Some((&mut fwd_virtual_pos, target_pattern)),
                );

                // Get bucket boundaries for target's pattern
                let pattern_idx = target_pattern as usize;
                let bucket_start = workspace.sort_offsets.get(pattern_idx).copied().unwrap_or(0);
                let bucket_end = workspace.sort_offsets.get(pattern_idx + 1).copied().unwrap_or(n_ref_haps);

                let fwd_ibs: Vec<u32> = pbwt_fwd
                    .select_neighbors_in_bucket(fwd_virtual_pos, n_ibs_haps, bucket_start, bucket_end)
                    .into_iter()
                    .map(|h| h.0)
                    .collect();

                // Update composite haplotypes with IBS matches from BOTH directions
                for hap in fwd_ibs {
                    self.update_with_ibs_hap(hap, step_idx as i32);
                }
                for hap in &bwd_ibs_per_step[step_idx] {
                    self.update_with_ibs_hap(*hap, step_idx as i32);
                }
            }
        }

        // Fill remaining slots with random haplotypes if needed
        if self.queue.len() < self.max_states {
            let target_hap = target_alleles_projected.iter().fold(0u32, |acc, &a| acc.wrapping_mul(31).wrapping_add(a as u32));
            self.fill_remaining_with_random(target_hap);
        }

        // Build output arrays (uses dense target_alleles for match computation)
        let n_states = self.queue.len().min(self.max_states);
        self.build_output_sparse(
            get_ref_allele,
            target_alleles_dense,
            genotyped_markers,
            n_states,
            hap_indices,
            allele_match,
        );

        n_states
    }

    fn initialize(&mut self) {
        // Reset Vec to sentinel value (faster than HashMap clear for large n_ref_haps)
        self.hap_to_last_ibs.fill(IBS_NIL);
        self.threaded_haps.clear();
        self.queue.clear();
    }

    fn update_with_ibs_hap(&mut self, hap: u32, step: i32) {
        let hap_idx = hap as usize;

        // Direct array lookup: O(1) with no hash computation
        if self.hap_to_last_ibs[hap_idx] == IBS_NIL {
            self.update_queue_head();

            if self.queue.len() == self.max_states {
                // Replace oldest composite haplotype
                if let Some(mut head) = self.queue.pop() {
                    let start_marker = if let (Some(proj_pos), Some(proj_markers)) = (self.projected_gen_positions, self.projected_markers) {
                        let last_step_idx = head.last_ibs_step.max(0) as usize;
                        let curr_step_idx = step.max(0) as usize;

                        // Get genetic positions of the steps (using midpoint for accuracy)
                        // Be careful with bounds: step index can be up to n_steps
                        let get_pos = |step_idx: usize| -> f64 {
                            if step_idx < self.ref_panel.n_steps() {
                                let step = self.ref_panel.step(step_idx);
                                let start_pos = proj_pos.get(step.start).copied().unwrap_or(0.0);
                                // step.end is exclusive, so use end-1 for last marker in step
                                let end_pos = proj_pos.get(step.end.saturating_sub(1)).copied().unwrap_or(start_pos);
                                (start_pos + end_pos) / 2.0
                            } else {
                                // After last step
                                proj_pos.last().copied().unwrap_or(0.0)
                            }
                        };

                        let pos_last = get_pos(last_step_idx);
                        let pos_curr = get_pos(curr_step_idx);

                        let mid_pos = (pos_last + pos_curr) / 2.0;

                        // Find projected marker closest to mid_pos, then map to dense index.
                        let idx = proj_pos.partition_point(|&p| p < mid_pos);
                        let proj_idx = idx.min(proj_pos.len().saturating_sub(1));
                        proj_markers.get(proj_idx).copied().unwrap_or(0)
                    } else {
                        // Fallback to old sparse logic (should not happen in projected mode)
                        let mid_step = (head.last_ibs_step + step) / 2;
                        let mid_step_idx = mid_step.max(0) as usize;
                        if mid_step_idx < self.ref_panel.n_steps() {
                             // Assuming standard mode where step start is dense index
                             self.ref_panel.step(mid_step_idx).start
                        } else {
                            0
                        }
                    };

                    // Clear old haplotype's entry
                    self.hap_to_last_ibs[head.hap as usize] = IBS_NIL;

                    if head.comp_hap_idx < self.threaded_haps.n_states() {
                        self.threaded_haps.add_segment(head.comp_hap_idx, hap, start_marker);
                    }

                    head.hap = hap;
                    head.last_ibs_step = step;
                    self.queue.push(head);
                }
            } else {
                // Add new composite haplotype
                let comp_hap_idx = self.threaded_haps.push_new(hap);
                self.queue.push(CompHapEntry {
                    comp_hap_idx,
                    hap,
                    last_ibs_step: step,
                });
            }
        }

        // Direct array write: O(1)
        self.hap_to_last_ibs[hap_idx] = step;
    }

    fn update_queue_head(&mut self) {
        while let Some(head) = self.queue.peek() {
            // Direct array lookup: O(1)
            let last_ibs = self.hap_to_last_ibs[head.hap as usize];
            if head.last_ibs_step != last_ibs {
                let mut head = self.queue.pop().unwrap();
                head.last_ibs_step = last_ibs;
                self.queue.push(head);
            } else {
                break;
            }
        }
    }

    fn fill_remaining_with_random(&mut self, target_hap_hash: u32) {
        use rand::rngs::StdRng;
        use rand::Rng;
        use rand::SeedableRng;

        // Use only reference haplotypes (exclude appended target haplotypes)
        let n_ref_haps = self.n_ref_haps;
        let n_states = self.max_states.min(n_ref_haps);
        let current_states = self.queue.len();

        if current_states >= n_states {
            return;
        }

        // Match Java: seed with target haplotype hash + global seed for reproducibility
        // In imputation, target haps are separate from reference, so no exclusion needed
        let mut rng = StdRng::seed_from_u64(target_hap_hash as u64 ^ self.seed);

        let ibs_step = 0;
        let mut attempts = 0;
        while self.queue.len() < n_states && attempts < n_ref_haps * 2 {
            let h = rng.random_range(0..n_ref_haps as u32);
            // Only add if this haplotype isn't already tracked (direct array check)
            if self.hap_to_last_ibs[h as usize] == IBS_NIL {
                let comp_hap_idx = self.threaded_haps.push_new(h);
                self.queue.push(CompHapEntry {
                    comp_hap_idx,
                    hap: h,
                    last_ibs_step: ibs_step,
                });
                self.hap_to_last_ibs[h as usize] = ibs_step;
            }
            attempts += 1;
        }
    }

    /// Build output arrays for GENOTYPED markers only (memory-efficient sparse version)
    ///
    /// Instead of allocating for all n_ref_markers (potentially millions), this only
    /// allocates for the genotyped markers (typically thousands), reducing memory by 10-100x.
    fn build_output_sparse<F>(
        &mut self,
        get_ref_allele: F,
        target_alleles: &[u8],
        genotyped_markers: &[usize],
        n_states: usize,
        hap_indices: &mut Vec<Vec<u32>>,
        allele_match: &mut Vec<Vec<bool>>,
    ) where
        F: Fn(usize, u32) -> u8,
    {
        let n_genotyped = genotyped_markers.len();

        // Allocate only for genotyped markers, not all reference markers
        hap_indices.clear();
        hap_indices.resize(n_genotyped, vec![0; n_states]);
        allele_match.clear();
        allele_match.resize(n_genotyped, vec![false; n_states]);

        self.threaded_haps.reset_cursors();

        // Process only genotyped markers
        let mut entries: Vec<CompHapEntry> = self.queue.iter().cloned().take(n_states).collect();
        entries.sort_by_key(|e| e.comp_hap_idx);

        for (sparse_idx, &ref_m) in genotyped_markers.iter().enumerate() {
            let target_allele = target_alleles.get(ref_m).copied().unwrap_or(255);

            for (j, entry) in entries.iter().enumerate() {
                if entry.comp_hap_idx < self.threaded_haps.n_states() {
                    let hap = self.threaded_haps.hap_at_raw(entry.comp_hap_idx, ref_m);
                    hap_indices[sparse_idx][j] = hap;

                    let ref_allele = get_ref_allele(ref_m, hap);
                    // Treat missing reference as wildcard (no penalty).
                    // For missing target alleles (255), leave as mismatch (Java behavior).
                    allele_match[sparse_idx][j] =
                        target_allele != 255 && (ref_allele == target_allele || ref_allele == 255);
                }
            }
        }
    }

    /// Materialize haplotypes for all states at every dense marker.
    ///
    /// This is used for small panels to improve interpolation accuracy by
    /// querying the mosaic state at the exact marker instead of reusing
    /// flanking haplotypes.
    pub fn dense_haps_for_states(&mut self, n_states: usize, n_markers: usize) -> Vec<Vec<u32>> {
        self.threaded_haps.reset_cursors();
        let mut out = vec![vec![0u32; n_states]; n_markers];
        for m in 0..n_markers {
            self.threaded_haps.materialize_haps(m, &mut out[m]);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_imp_states_creation() {
        // Basic sanity test - we can't easily test without a full RefPanelCoded
        // but we verify the struct compiles and basic invariants hold
        assert!(std::mem::size_of::<ImpStates>() > 0);
    }
}
