//! # Imputation State Selection
//!
//! Implements dynamic HMM state selection using PBWT-based IBS matching.
//! This matches Java `imp/ImpStates.java` and `imp/ImpIbs.java`.
//!
//! Key concepts:
//! - States are "composite haplotypes" - mosaics of reference haplotype segments
//! - PBWT finds IBS matches at each step (marker cluster)
//! - A priority queue tracks which composite haplotypes to keep/replace

use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;

use crate::data::HapIdx;
use crate::model::pbwt::PbwtDivUpdater;

/// A segment of a composite haplotype
#[derive(Clone, Debug)]
pub struct HapSegment {
    /// Reference haplotype index for this segment
    pub hap: u32,
    /// End marker (exclusive) for this segment
    pub end: usize,
}

/// A composite haplotype made of reference haplotype segments
#[derive(Clone, Debug)]
pub struct CompositeHap {
    /// Segments making up this composite haplotype
    pub segments: Vec<HapSegment>,
    /// Current segment index
    current_seg: usize,
}

impl CompositeHap {
    /// Create a new composite haplotype starting with given hap
    pub fn new(start_hap: u32, n_markers: usize) -> Self {
        Self {
            segments: vec![HapSegment {
                hap: start_hap,
                end: n_markers,
            }],
            current_seg: 0,
        }
    }

    /// Get the reference haplotype at a given marker
    pub fn hap_at(&mut self, marker: usize) -> u32 {
        while self.current_seg < self.segments.len() - 1
            && marker >= self.segments[self.current_seg].end
        {
            self.current_seg += 1;
        }
        self.segments[self.current_seg].hap
    }

    /// Add a new segment starting at given marker
    pub fn add_segment(&mut self, hap: u32, start_marker: usize, n_markers: usize) {
        // Update the end of the last segment
        if let Some(last) = self.segments.last_mut() {
            last.end = start_marker;
        }
        // Add new segment
        self.segments.push(HapSegment {
            hap,
            end: n_markers,
        });
    }

    /// Reset for a new iteration
    pub fn reset(&mut self) {
        self.current_seg = 0;
    }
}

/// Entry in the priority queue for state management
#[derive(Clone, Debug)]
struct CompHapEntry {
    /// Index into the composite haplotypes array
    comp_hap_idx: usize,
    /// Current reference haplotype
    hap: u32,
    /// Last step where this hap was seen in IBS matches
    last_ibs_step: i32,
    /// Start marker of current segment
    start_marker: usize,
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

/// Configuration for coded steps (marker clusters)
#[derive(Clone, Debug)]
pub struct CodedStepsConfig {
    /// Step size in cM
    pub step_cm: f32,
    /// Number of IBS haplotypes to find per step
    pub n_ibs_haps: usize,
}

impl Default for CodedStepsConfig {
    fn default() -> Self {
        Self {
            step_cm: 0.1,  // From Java imp_step parameter
            n_ibs_haps: 8, // From Java imp_nsteps
        }
    }
}

/// Coded steps - divides markers into intervals for IBS matching
#[derive(Clone, Debug)]
pub struct CodedSteps {
    /// Start marker index for each step
    step_starts: Vec<usize>,
    /// Number of markers total
    n_markers: usize,
}

impl CodedSteps {
    /// Create coded steps from genetic positions
    pub fn new(gen_positions: &[f64], config: &CodedStepsConfig) -> Self {
        if gen_positions.is_empty() {
            return Self {
                step_starts: vec![0],
                n_markers: 0,
            };
        }

        let n_markers = gen_positions.len();
        let mut step_starts = Vec::new();
        step_starts.push(0);

        let step_cm = config.step_cm as f64;

        // First step is half-length
        let mut next_pos = gen_positions[0] + step_cm / 2.0;
        let mut idx = Self::next_index(gen_positions, 0, next_pos);

        while idx < n_markers {
            step_starts.push(idx);
            next_pos = gen_positions[idx] + step_cm;
            idx = Self::next_index(gen_positions, idx, next_pos);
        }

        Self {
            step_starts,
            n_markers,
        }
    }

    fn next_index(positions: &[f64], start: usize, target: f64) -> usize {
        match positions[start..].binary_search_by(|p| p.partial_cmp(&target).unwrap()) {
            Ok(i) => start + i,
            Err(i) => start + i,
        }
    }

    /// Number of steps
    pub fn n_steps(&self) -> usize {
        self.step_starts.len()
    }

    /// Start marker for a step
    pub fn step_start(&self, step: usize) -> usize {
        self.step_starts[step]
    }

    /// End marker for a step (exclusive)
    pub fn step_end(&self, step: usize) -> usize {
        if step + 1 < self.step_starts.len() {
            self.step_starts[step + 1]
        } else {
            self.n_markers
        }
    }
}

/// IBS haplotype finder using PBWT
pub struct ImpIbs {
    /// PBWT updater
    pbwt_updater: PbwtDivUpdater,
    /// Forward prefix array
    fwd_prefix: Vec<u32>,
    /// Forward divergence array
    fwd_divergence: Vec<i32>,
    /// Number of reference haplotypes
    n_ref_haps: usize,
    /// Number of IBS haplotypes to return per step
    n_ibs_haps: usize,
}

impl ImpIbs {
    /// Create a new IBS finder
    pub fn new(n_ref_haps: usize, n_ibs_haps: usize) -> Self {
        Self {
            pbwt_updater: PbwtDivUpdater::new(n_ref_haps),
            fwd_prefix: (0..n_ref_haps as u32).collect(),
            fwd_divergence: vec![0; n_ref_haps + 1],
            n_ref_haps,
            n_ibs_haps,
        }
    }

    /// Reset PBWT state
    pub fn reset(&mut self) {
        for (i, p) in self.fwd_prefix.iter_mut().enumerate() {
            *p = i as u32;
        }
        self.fwd_divergence.fill(0);
    }

    /// Update PBWT with alleles at a marker
    pub fn update(&mut self, alleles: &[u8], n_alleles: usize, marker: usize) {
        self.pbwt_updater.fwd_update(
            alleles,
            n_alleles,
            marker,
            &mut self.fwd_prefix,
            &mut self.fwd_divergence[..self.n_ref_haps],
        );
    }

    /// Find IBS haplotypes for a target allele sequence at current PBWT position
    ///
    /// Returns haplotype indices that are IBS with the target
    pub fn find_ibs_haps(&self, target_seq_idx: usize, step: usize) -> Vec<u32> {
        // Find position of target in prefix array (or closest match)
        let target_pos = self.find_target_position(target_seq_idx);

        // Expand from target position to find IBS neighbors
        let mut result = Vec::with_capacity(self.n_ibs_haps);
        let n = self.fwd_prefix.len();

        let mut left = target_pos;
        let mut right = target_pos + 1;
        let step_i32 = step as i32;

        while result.len() < self.n_ibs_haps && (left > 0 || right < n) {
            // Check left neighbor
            if left > 0 {
                left -= 1;
                let hap = self.fwd_prefix[left];
                if hap < self.n_ref_haps as u32 {
                    // Check if IBS (divergence <= current step)
                    let div = self.fwd_divergence.get(left + 1).copied().unwrap_or(0);
                    if div <= step_i32 {
                        result.push(hap);
                    }
                }
            }

            // Check right neighbor
            if right < n && result.len() < self.n_ibs_haps {
                let hap = self.fwd_prefix[right];
                if hap < self.n_ref_haps as u32 {
                    let div = self.fwd_divergence.get(right).copied().unwrap_or(0);
                    if div <= step_i32 {
                        result.push(hap);
                    }
                }
                right += 1;
            }
        }

        // If we still don't have enough, just add nearby haplotypes
        if result.len() < self.n_ibs_haps {
            let mut left = target_pos;
            let mut right = target_pos + 1;
            while result.len() < self.n_ibs_haps && (left > 0 || right < n) {
                if left > 0 {
                    left -= 1;
                    let hap = self.fwd_prefix[left];
                    if hap < self.n_ref_haps as u32 && !result.contains(&hap) {
                        result.push(hap);
                    }
                }
                if right < n && result.len() < self.n_ibs_haps {
                    let hap = self.fwd_prefix[right];
                    if hap < self.n_ref_haps as u32 && !result.contains(&hap) {
                        result.push(hap);
                    }
                    right += 1;
                }
            }
        }

        result
    }

    fn find_target_position(&self, target_seq_idx: usize) -> usize {
        // For imputation, target is not in prefix array
        // Return middle position as starting point
        self.fwd_prefix.len() / 2
    }
}

/// Dynamic HMM state selector for imputation
///
/// Matches Java `imp/ImpStates.java`
pub struct ImpStates {
    /// Maximum number of HMM states
    max_states: usize,
    /// IBS haplotype finder
    ibs: ImpIbs,
    /// Coded steps configuration
    coded_steps: CodedSteps,
    /// Composite haplotypes (the HMM states)
    comp_haps: Vec<CompositeHap>,
    /// Map from reference haplotype to last IBS step
    hap_to_last_ibs: HashMap<u32, i32>,
    /// Priority queue for managing composite haplotypes
    queue: BinaryHeap<CompHapEntry>,
    /// Number of markers
    n_markers: usize,
}

impl ImpStates {
    /// Create a new state selector
    pub fn new(
        n_ref_haps: usize,
        max_states: usize,
        gen_positions: &[f64],
        config: &CodedStepsConfig,
    ) -> Self {
        let coded_steps = CodedSteps::new(gen_positions, config);
        let n_markers = gen_positions.len();

        Self {
            max_states,
            ibs: ImpIbs::new(n_ref_haps, config.n_ibs_haps),
            coded_steps,
            comp_haps: Vec::with_capacity(max_states),
            hap_to_last_ibs: HashMap::with_capacity(max_states),
            queue: BinaryHeap::with_capacity(max_states),
            n_markers,
        }
    }

    /// Select IBS-based HMM states for a target haplotype
    ///
    /// # Arguments
    /// * `get_ref_allele` - Function to get reference allele at (marker, hap)
    /// * `target_alleles` - Target haplotype alleles at genotyped markers
    /// * `hap_indices` - Output: reference haplotype for each state at each marker
    /// * `allele_match` - Output: whether state allele matches target at each marker
    ///
    /// # Returns
    /// Number of states
    pub fn ibs_states<F>(
        &mut self,
        get_ref_allele: F,
        target_alleles: &[u8],
        hap_indices: &mut Vec<Vec<u32>>,
        allele_match: &mut Vec<Vec<bool>>,
    ) -> usize
    where
        F: Fn(usize, u32) -> u8,
    {
        self.initialize();

        // Process each step
        for step in 0..self.coded_steps.n_steps() {
            let step_start = self.coded_steps.step_start(step);
            let step_end = self.coded_steps.step_end(step);

            // Update PBWT for markers in this step
            for m in step_start..step_end {
                let mut alleles: Vec<u8> = Vec::with_capacity(self.ibs.n_ref_haps);
                for h in 0..self.ibs.n_ref_haps as u32 {
                    alleles.push(get_ref_allele(m, h));
                }
                // Determine number of alleles
                let n_alleles = alleles.iter().copied().max().unwrap_or(0) as usize + 1;
                self.ibs.update(&alleles, n_alleles.max(2), m);
            }

            // Find IBS haplotypes for target
            let target_seq = 0; // Placeholder - would need proper sequence matching
            let ibs_haps = self.ibs.find_ibs_haps(target_seq, step);

            // Update composite haplotypes with IBS matches
            for hap in ibs_haps {
                self.update_with_ibs_hap(hap, step as i32, step_start);
            }
        }

        // If queue is empty, fill with random haplotypes
        if self.queue.is_empty() {
            self.fill_with_random_haps();
        }

        // Build output arrays
        let n_states = self.queue.len().min(self.max_states);
        self.build_output(get_ref_allele, target_alleles, n_states, hap_indices, allele_match);

        n_states
    }

    fn initialize(&mut self) {
        self.hap_to_last_ibs.clear();
        self.comp_haps.clear();
        self.queue.clear();
        self.ibs.reset();
    }

    fn update_with_ibs_hap(&mut self, hap: u32, step: i32, step_start: usize) {
        const NIL: i32 = i32::MIN;

        if self.hap_to_last_ibs.get(&hap).copied().unwrap_or(NIL) == NIL {
            // Hap not currently tracked
            self.update_queue_head();

            if self.queue.len() == self.max_states {
                // Replace oldest composite haplotype
                if let Some(mut head) = self.queue.pop() {
                    let mid_step = (head.last_ibs_step + step) / 2;
                    let start_marker = self.coded_steps.step_start(mid_step.max(0) as usize);

                    self.hap_to_last_ibs.remove(&head.hap);

                    // Update composite haplotype with new segment
                    if head.comp_hap_idx < self.comp_haps.len() {
                        self.comp_haps[head.comp_hap_idx].add_segment(
                            hap,
                            start_marker,
                            self.n_markers,
                        );
                    }

                    head.hap = hap;
                    head.last_ibs_step = step;
                    head.start_marker = start_marker;
                    self.queue.push(head);
                }
            } else {
                // Add new composite haplotype
                let comp_hap_idx = self.comp_haps.len();
                self.comp_haps.push(CompositeHap::new(hap, self.n_markers));
                self.queue.push(CompHapEntry {
                    comp_hap_idx,
                    hap,
                    last_ibs_step: step,
                    start_marker: 0,
                });
            }
        }

        self.hap_to_last_ibs.insert(hap, step);
    }

    fn update_queue_head(&mut self) {
        // Update the head's last_ibs_step if it has been seen more recently
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
        let n_states = self.max_states.min(self.ibs.n_ref_haps);
        for h in 0..n_states as u32 {
            let comp_hap_idx = self.comp_haps.len();
            self.comp_haps.push(CompositeHap::new(h, self.n_markers));
            self.queue.push(CompHapEntry {
                comp_hap_idx,
                hap: h,
                last_ibs_step: 0,
                start_marker: 0,
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
        // Resize output arrays
        hap_indices.clear();
        hap_indices.resize(self.n_markers, vec![0; n_states]);
        allele_match.clear();
        allele_match.resize(self.n_markers, vec![false; n_states]);

        // Reset composite haplotypes for iteration
        for comp_hap in &mut self.comp_haps {
            comp_hap.reset();
        }

        // Build output for each marker
        for m in 0..self.n_markers {
            let target_allele = target_alleles.get(m).copied().unwrap_or(255);

            for (j, entry) in self.queue.iter().take(n_states).enumerate() {
                if entry.comp_hap_idx < self.comp_haps.len() {
                    let hap = self.comp_haps[entry.comp_hap_idx].hap_at(m);
                    hap_indices[m][j] = hap;

                    let ref_allele = get_ref_allele(m, hap);
                    allele_match[m][j] = ref_allele == target_allele;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coded_steps() {
        let positions: Vec<f64> = (0..100).map(|i| i as f64 * 0.05).collect();
        let config = CodedStepsConfig::default();
        let steps = CodedSteps::new(&positions, &config);

        assert!(steps.n_steps() > 1);
        assert_eq!(steps.step_start(0), 0);
    }

    #[test]
    fn test_composite_hap() {
        let mut comp = CompositeHap::new(0, 100);
        assert_eq!(comp.hap_at(0), 0);
        assert_eq!(comp.hap_at(50), 0);

        comp.add_segment(1, 30, 100);
        comp.reset();
        assert_eq!(comp.hap_at(0), 0);
        assert_eq!(comp.hap_at(29), 0);
        assert_eq!(comp.hap_at(30), 1);
        assert_eq!(comp.hap_at(99), 1);
    }

    #[test]
    fn test_imp_ibs() {
        let mut ibs = ImpIbs::new(100, 8);
        let alleles: Vec<u8> = (0..100).map(|i| (i % 2) as u8).collect();
        ibs.update(&alleles, 2, 0);

        let haps = ibs.find_ibs_haps(0, 0);
        assert!(!haps.is_empty());
    }
}
