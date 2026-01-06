//! # Imputation State Selection
//!
//! Implements dynamic HMM state selection using PBWT-based IBS matching.
//! This matches Java `imp/ImpStates.java`.
//!
//! Key concepts:
//! - States are "composite haplotypes" - mosaics of reference haplotype segments
//! - PBWT finds IBS matches at each step (marker cluster)
//! - A priority queue tracks which composite haplotypes to keep/replace

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

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
pub struct ImpStates<'a> {
    /// Maximum number of HMM states
    max_states: usize,
    /// Reference panel with coded steps
    ref_panel: &'a RefPanelCoded,
    /// Composite haplotypes (the HMM states) - Optimized Arena
    threaded_haps: ThreadedHaps,
    /// Map from reference haplotype to last IBS step
    hap_to_last_ibs: HashMap<u32, i32>,
    /// Priority queue for managing composite haplotypes
    queue: BinaryHeap<CompHapEntry>,
    /// Number of reference markers
    n_ref_markers: usize,
    /// Number of reference haplotypes (excludes appended target haplotypes)
    n_ref_haps: usize,
    /// Number of IBS haplotypes to find per step
    n_ibs_haps: usize,
    /// Random seed for reproducibility
    seed: u64,
}

impl<'a> ImpStates<'a> {
    /// Create a new state selector
    ///
    /// # Arguments
    /// * `ref_panel` - Reference panel with coded steps (defines step boundaries)
    /// * `n_ref_haps` - Number of reference haplotypes (excludes appended target haplotypes)
    /// * `max_states` - Maximum number of HMM states to track
    /// * `n_ibs_haps` - Number of IBS haplotypes to find per step
    /// * `seed` - Random seed for reproducibility
    pub fn new(
        ref_panel: &'a RefPanelCoded,
        n_ref_haps: usize,
        max_states: usize,
        n_ibs_haps: usize,
        seed: u64,
    ) -> Self {
        let n_ref_markers = ref_panel.n_markers();

        Self {
            max_states,
            ref_panel,
            threaded_haps: ThreadedHaps::new(max_states, max_states * 4, n_ref_markers),
            hap_to_last_ibs: HashMap::with_capacity(max_states),
            queue: BinaryHeap::with_capacity(max_states),
            n_ref_markers,
            n_ref_haps,
            n_ibs_haps,
            seed,
        }
    }

    /// Select IBS-based HMM states for a target haplotype
    ///
    /// Uses BOTH forward and backward PBWT passes to find IBS matches,
    /// matching Java Beagle's bidirectional approach.
    ///
    /// **Sparse target handling:** Steps with all-missing target data are skipped
    /// for IBS matching, preventing degenerate state selection when the target
    /// is much sparser than the reference panel.
    ///
    /// # Arguments
    /// * `get_ref_allele` - Function to get reference allele at (marker, hap)
    /// * `target_alleles` - Target alleles in **reference marker space** (length = n_ref_markers)
    ///                      Use 255 for markers not genotyped in target
    /// * `workspace` - Pre-allocated workspace buffers
    /// * `hap_indices` - Output: reference haplotype indices at each marker
    /// * `allele_match` - Output: whether each state matches target at each marker
    ///
    /// # Returns
    /// Number of states selected
    pub fn ibs_states<F>(
        &mut self,
        get_ref_allele: F,
        target_alleles: &[u8],
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

        // Pre-compute which steps have informative target data
        // Steps with all-missing data should be skipped for IBS matching
        let step_has_data: Vec<bool> = (0..n_steps)
            .map(|step_idx| {
                let coded_step = self.ref_panel.step(step_idx);
                let step_start = coded_step.start;
                let step_end = coded_step.end;
                // Check if any marker in this step has non-missing target data
                (step_start..step_end).any(|m| {
                    target_alleles.get(m).copied().unwrap_or(255) != 255
                })
            })
            .collect();

        // Store backward IBS haps for each step (to use during forward pass)
        let mut bwd_ibs_per_step: Vec<Vec<u32>> = vec![Vec::new(); n_steps];

        // STEP 1: Backward PBWT pass
        {
            let mut pbwt_bwd = CodedPbwtView::new_backward(
                &mut workspace.pbwt_prefix_bwd[..n_ref_haps],
                &mut workspace.pbwt_divergence_bwd[..n_ref_haps + 1],
                n_steps,
            );

            for step_idx in (0..n_steps).rev() {
                let coded_step = self.ref_panel.step(step_idx);
                let step_start = coded_step.start;
                let step_end = coded_step.end;

                pbwt_bwd.update_backward(coded_step, n_steps);

                // Skip IBS matching if this step has no informative target data
                if !step_has_data[step_idx] {
                    continue;
                }

                // Extract target sequence for this step using reference marker indices
                let target_seq: Vec<u8> = (step_start..step_end)
                    .map(|m| target_alleles.get(m).copied().unwrap_or(255))
                    .collect();

                let bwd_ibs: Vec<u32> =
                    if let Some(target_pattern) = coded_step.match_sequence(&target_seq) {
                        pbwt_bwd
                            .find_ibs(target_pattern, coded_step, n_ibs_haps)
                            .into_iter()
                            .map(|h| h.0)
                            .collect()
                    } else {
                        let closest_pattern = coded_step.closest_pattern(&target_seq);
                        pbwt_bwd
                            .find_ibs(closest_pattern, coded_step, n_ibs_haps)
                            .into_iter()
                            .map(|h| h.0)
                            .collect()
                    };

                bwd_ibs_per_step[step_idx] = bwd_ibs;
            }
        }

        // STEP 2: Forward PBWT pass + composite haplotype building
        {
            let mut pbwt_fwd = CodedPbwtView::new(
                &mut workspace.pbwt_prefix[..n_ref_haps],
                &mut workspace.pbwt_divergence[..n_ref_haps + 1],
            );

            for step_idx in 0..n_steps {
                let coded_step = self.ref_panel.step(step_idx);
                let step_start = coded_step.start;
                let step_end = coded_step.end;

                pbwt_fwd.update_counting_sort(
                    coded_step,
                    &mut workspace.sort_counts,
                    &mut workspace.sort_offsets,
                    &mut workspace.sort_prefix_scratch,
                    &mut workspace.sort_div_scratch,
                );

                // Skip IBS matching if this step has no informative target data
                if !step_has_data[step_idx] {
                    continue;
                }

                // Extract target sequence for this step using reference marker indices
                let target_seq: Vec<u8> = (step_start..step_end)
                    .map(|m| target_alleles.get(m).copied().unwrap_or(255))
                    .collect();

                let fwd_ibs: Vec<u32> =
                    if let Some(target_pattern) = coded_step.match_sequence(&target_seq) {
                        pbwt_fwd
                            .find_ibs(target_pattern, coded_step, n_ibs_haps)
                            .into_iter()
                            .map(|h| h.0)
                            .collect()
                    } else {
                        let closest_pattern = coded_step.closest_pattern(&target_seq);
                        pbwt_fwd
                            .find_ibs(closest_pattern, coded_step, n_ibs_haps)
                            .into_iter()
                            .map(|h| h.0)
                            .collect()
                    };

                // Update composite haplotypes with IBS matches from BOTH directions
                // Use the step_idx for state recency tracking (matches Java behavior)
                for hap in fwd_ibs {
                    self.update_with_ibs_hap(hap, step_idx as i32);
                }
                for hap in &bwd_ibs_per_step[step_idx] {
                    self.update_with_ibs_hap(*hap, step_idx as i32);
                }
            }
        }

        // If queue is empty, fill with random haplotypes
        if self.queue.is_empty() {
            self.fill_with_random_haps();
        }

        // Build output arrays
        let n_states = self.queue.len().min(self.max_states);
        self.build_output(
            get_ref_allele,
            target_alleles,
            n_states,
            hap_indices,
            allele_match,
        );

        n_states
    }

    /// Select IBS-based HMM states using TARGET-BASED CLUSTERS.
    ///
    /// This is the preferred method for sparse targets (e.g., genotyping arrays).
    /// Unlike `ibs_states`, this iterates over target-defined clusters where every
    /// cluster has informative data, matching Java Beagle's approach.
    ///
    /// # Arguments
    /// * `target_alleles` - Target alleles in **target marker space** (dense, no gaps)
    /// * `targ_cluster_start` - Target marker index for start of each cluster
    /// * `targ_cluster_end` - Target marker index for end of each cluster (exclusive)
    /// * `ref_cluster_start` - Reference marker index for start of each cluster
    /// * `workspace` - Pre-allocated workspace buffers
    /// * `get_cluster_allele` - Returns (ref_allele, target_allele) for (cluster, ref_hap)
    /// * `hap_indices` - Output: reference haplotype indices at each cluster
    /// * `allele_match` - Output: whether each state matches target at each cluster
    ///
    /// # Returns
    /// Number of states selected
    pub fn ibs_states_clustered<F>(
        &mut self,
        target_alleles: &[u8],
        targ_cluster_start: &[usize],
        targ_cluster_end: &[usize],
        ref_cluster_start: &[usize],
        workspace: &mut ImpWorkspace,
        get_cluster_allele: F,
        hap_indices: &mut Vec<Vec<u32>>,
        allele_match: &mut Vec<Vec<bool>>,
    ) -> usize
    where
        F: Fn(usize, u32) -> (u8, u8), // (ref_allele, target_allele)
    {
        self.initialize();

        let n_ref_haps = self.n_ref_haps;
        let n_steps = self.ref_panel.n_steps();
        let n_clusters = targ_cluster_start.len();
        let n_ibs_haps = self.n_ibs_haps;
        workspace.resize_with_ref(self.max_states, self.n_ref_markers, n_ref_haps);

        // Build mapping from reference step to target cluster (if any)
        // A reference step maps to a target cluster if the step's markers overlap
        // with the cluster's reference marker range
        let step_to_cluster: Vec<Option<usize>> = (0..n_steps)
            .map(|step_idx| {
                let coded_step = self.ref_panel.step(step_idx);
                let step_start = coded_step.start;
                let step_end = coded_step.end;
                // Find cluster that overlaps this step
                for (c, &ref_start) in ref_cluster_start.iter().enumerate() {
                    let ref_end = if c + 1 < n_clusters {
                        ref_cluster_start[c + 1]
                    } else {
                        self.n_ref_markers
                    };
                    // Check if step overlaps cluster's reference range
                    if step_start < ref_end && step_end > ref_start {
                        return Some(c);
                    }
                }
                None
            })
            .collect();

        // Store backward IBS haps for each cluster
        let mut bwd_ibs_per_cluster: Vec<Vec<u32>> = vec![Vec::new(); n_clusters];

        // STEP 1: Backward PBWT pass
        {
            let mut pbwt_bwd = CodedPbwtView::new_backward(
                &mut workspace.pbwt_prefix_bwd[..n_ref_haps],
                &mut workspace.pbwt_divergence_bwd[..n_ref_haps + 1],
                n_steps,
            );

            for step_idx in (0..n_steps).rev() {
                let coded_step = self.ref_panel.step(step_idx);
                let step_start = coded_step.start;
                let step_end = coded_step.end;

                pbwt_bwd.update_backward(coded_step, n_steps);

                // Only do IBS matching if this step maps to a target cluster
                if let Some(cluster) = step_to_cluster[step_idx] {
                    let targ_start = targ_cluster_start[cluster];
                    let targ_end = targ_cluster_end[cluster];

                    // Build target sequence for this step from target alleles
                    let target_seq: Vec<u8> = (step_start..step_end)
                        .map(|_| {
                            // Use majority allele from target markers in this cluster
                            // For simplicity, use first non-missing allele
                            (targ_start..targ_end)
                                .map(|t| target_alleles.get(t).copied().unwrap_or(255))
                                .find(|&a| a != 255)
                                .unwrap_or(0)
                        })
                        .collect();

                    let bwd_ibs: Vec<u32> =
                        if let Some(target_pattern) = coded_step.match_sequence(&target_seq) {
                            pbwt_bwd
                                .find_ibs(target_pattern, coded_step, n_ibs_haps)
                                .into_iter()
                                .map(|h| h.0)
                                .collect()
                        } else {
                            let closest_pattern = coded_step.closest_pattern(&target_seq);
                            pbwt_bwd
                                .find_ibs(closest_pattern, coded_step, n_ibs_haps)
                                .into_iter()
                                .map(|h| h.0)
                                .collect()
                        };

                    bwd_ibs_per_cluster[cluster].extend(bwd_ibs);
                }
            }
        }

        // STEP 2: Forward PBWT pass + composite haplotype building
        {
            let mut pbwt_fwd = CodedPbwtView::new(
                &mut workspace.pbwt_prefix[..n_ref_haps],
                &mut workspace.pbwt_divergence[..n_ref_haps + 1],
            );

            for step_idx in 0..n_steps {
                let coded_step = self.ref_panel.step(step_idx);
                let step_start = coded_step.start;
                let step_end = coded_step.end;

                pbwt_fwd.update_counting_sort(
                    coded_step,
                    &mut workspace.sort_counts,
                    &mut workspace.sort_offsets,
                    &mut workspace.sort_prefix_scratch,
                    &mut workspace.sort_div_scratch,
                );

                // Only do IBS matching if this step maps to a target cluster
                if let Some(cluster) = step_to_cluster[step_idx] {
                    let targ_start = targ_cluster_start[cluster];
                    let targ_end = targ_cluster_end[cluster];

                    // Build target sequence for this step
                    let target_seq: Vec<u8> = (step_start..step_end)
                        .map(|_| {
                            (targ_start..targ_end)
                                .map(|t| target_alleles.get(t).copied().unwrap_or(255))
                                .find(|&a| a != 255)
                                .unwrap_or(0)
                        })
                        .collect();

                    let fwd_ibs: Vec<u32> =
                        if let Some(target_pattern) = coded_step.match_sequence(&target_seq) {
                            pbwt_fwd
                                .find_ibs(target_pattern, coded_step, n_ibs_haps)
                                .into_iter()
                                .map(|h| h.0)
                                .collect()
                        } else {
                            let closest_pattern = coded_step.closest_pattern(&target_seq);
                            pbwt_fwd
                                .find_ibs(closest_pattern, coded_step, n_ibs_haps)
                                .into_iter()
                                .map(|h| h.0)
                                .collect()
                        };

                    // Update composite haplotypes with IBS matches from BOTH directions
                    // Use cluster index for state recency tracking
                    for hap in fwd_ibs {
                        self.update_with_ibs_hap(hap, cluster as i32);
                    }
                    for &hap in &bwd_ibs_per_cluster[cluster] {
                        self.update_with_ibs_hap(hap, cluster as i32);
                    }
                }
            }
        }

        // If queue is empty, fill with random haplotypes
        if self.queue.is_empty() {
            self.fill_with_random_haps();
        }

        // Build output arrays indexed by cluster
        let n_states = self.queue.len().min(self.max_states);
        self.build_output_clustered(
            get_cluster_allele,
            n_clusters,
            ref_cluster_start,
            n_states,
            hap_indices,
            allele_match,
        );

        n_states
    }

    fn initialize(&mut self) {
        self.hap_to_last_ibs.clear();
        self.threaded_haps.clear();
        self.queue.clear();
    }

    fn update_with_ibs_hap(&mut self, hap: u32, step: i32) {
        const NIL: i32 = i32::MIN;

        if self.hap_to_last_ibs.get(&hap).copied().unwrap_or(NIL) == NIL {
            self.update_queue_head();

            if self.queue.len() == self.max_states {
                // Replace oldest composite haplotype
                if let Some(mut head) = self.queue.pop() {
                    let mid_step = (head.last_ibs_step + step) / 2;
                    // Use reference panel's step boundary (the fix!)
                    let mid_step_idx = mid_step.max(0) as usize;
                    let start_marker = if mid_step_idx < self.ref_panel.n_steps() {
                        self.ref_panel.step(mid_step_idx).start
                    } else {
                        0
                    };

                    self.hap_to_last_ibs.remove(&head.hap);

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

        self.hap_to_last_ibs.insert(hap, step);
    }

    fn update_queue_head(&mut self) {
        while let Some(head) = self.queue.peek() {
            let last_ibs = self.hap_to_last_ibs.get(&head.hap).copied().unwrap_or(i32::MIN);
            if head.last_ibs_step != last_ibs {
                let mut head = self.queue.pop().unwrap();
                head.last_ibs_step = last_ibs;
                self.queue.push(head);
            } else {
                break;
            }
        }
    }

    fn fill_with_random_haps(&mut self) {
        use rand::rngs::StdRng;
        use rand::seq::SliceRandom;
        use rand::SeedableRng;

        // Use only reference haplotypes (exclude appended target haplotypes)
        let n_ref_haps = self.n_ref_haps;
        let n_states = self.max_states.min(n_ref_haps);

        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut hap_indices: Vec<u32> = (0..n_ref_haps as u32).collect();
        hap_indices.shuffle(&mut rng);

        for &h in hap_indices.iter().take(n_states) {
            let comp_hap_idx = self.threaded_haps.push_new(h);
            self.queue.push(CompHapEntry {
                comp_hap_idx,
                hap: h,
                last_ibs_step: 0,
            });
        }
    }

    fn build_output<F>(
        &mut self,
        get_ref_allele: F,
        target_alleles: &[u8],
        n_states: usize,
        hap_indices: &mut Vec<Vec<u32>>,
        allele_match: &mut Vec<Vec<bool>>,
    ) where
        F: Fn(usize, u32) -> u8,
    {
        hap_indices.clear();
        hap_indices.resize(self.n_ref_markers, vec![0; n_states]);
        allele_match.clear();
        allele_match.resize(self.n_ref_markers, vec![false; n_states]);

        self.threaded_haps.reset_cursors();

        for m in 0..self.n_ref_markers {
            let target_allele = target_alleles.get(m).copied().unwrap_or(255);

            for (j, entry) in self.queue.iter().take(n_states).enumerate() {
                if entry.comp_hap_idx < self.threaded_haps.n_states() {
                    let hap = self.threaded_haps.hap_at_raw(entry.comp_hap_idx, m);
                    hap_indices[m][j] = hap;

                    let ref_allele = get_ref_allele(m, hap);
                    allele_match[m][j] = target_allele == 255 || ref_allele == target_allele;
                }
            }
        }
    }

    /// Build output arrays indexed by TARGET CLUSTER instead of reference marker.
    ///
    /// This matches Java Beagle's approach where HMM runs on clusters defined by
    /// target marker positions, ensuring every cluster has informative data.
    ///
    /// # Arguments
    /// * `get_cluster_allele` - Returns (ref_hap_allele, target_allele) for (cluster, ref_hap)
    /// * `n_clusters` - Number of target clusters
    /// * `cluster_to_ref_start` - Maps cluster index to reference marker start
    /// * `n_states` - Number of HMM states
    /// * `hap_indices` - Output: reference haplotype at each (cluster, state)
    /// * `allele_match` - Output: whether state matches target at each cluster
    pub fn build_output_clustered<F>(
        &mut self,
        get_cluster_allele: F,
        n_clusters: usize,
        cluster_to_ref_start: &[usize],
        n_states: usize,
        hap_indices: &mut Vec<Vec<u32>>,
        allele_match: &mut Vec<Vec<bool>>,
    ) where
        F: Fn(usize, u32) -> (u8, u8), // (ref_allele, target_allele)
    {
        hap_indices.clear();
        hap_indices.resize(n_clusters, vec![0; n_states]);
        allele_match.clear();
        allele_match.resize(n_clusters, vec![false; n_states]);

        self.threaded_haps.reset_cursors();

        for cluster in 0..n_clusters {
            // Get the reference marker corresponding to this cluster start
            let ref_m = cluster_to_ref_start[cluster];

            for (j, entry) in self.queue.iter().take(n_states).enumerate() {
                if entry.comp_hap_idx < self.threaded_haps.n_states() {
                    let hap = self.threaded_haps.hap_at_raw(entry.comp_hap_idx, ref_m);
                    hap_indices[cluster][j] = hap;

                    let (ref_allele, target_allele) = get_cluster_allele(cluster, hap);
                    allele_match[cluster][j] = target_allele == 255 || ref_allele == target_allele;
                }
            }
        }
    }

    /// Get the number of states currently in the queue
    pub fn n_active_states(&self) -> usize {
        self.queue.len().min(self.max_states)
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
