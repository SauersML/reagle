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

use crate::data::storage::coded_steps::{RefPanelCoded, CodedPbwtView};
use crate::utils::workspace::ImpWorkspace;
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

/// IBS haplotype finder using PBWT (forward and backward)
///
/// Java Beagle uses BOTH forward and backward PBWT to select states,
/// ensuring we find haplotypes that match well in the past AND future.
pub struct ImpIbs {
    /// PBWT updater
    pbwt_updater: PbwtDivUpdater,
    /// Forward prefix array
    fwd_prefix: Vec<u32>,
    /// Forward divergence array
    fwd_divergence: Vec<i32>,
    /// Backward prefix array
    bwd_prefix: Vec<u32>,
    /// Backward divergence array
    bwd_divergence: Vec<i32>,
    /// Number of reference haplotypes
    pub n_ref_haps: usize,
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
            bwd_prefix: (0..n_ref_haps as u32).collect(),
            bwd_divergence: vec![0; n_ref_haps + 1],
            n_ref_haps,
            n_ibs_haps,
        }
    }

    /// Reset PBWT state (both forward and backward)
    pub fn reset(&mut self) {
        for (i, p) in self.fwd_prefix.iter_mut().enumerate() {
            *p = i as u32;
        }
        self.fwd_divergence.fill(0);
        for (i, p) in self.bwd_prefix.iter_mut().enumerate() {
            *p = i as u32;
        }
        self.bwd_divergence.fill(0);
    }

    /// Reset only forward PBWT
    pub fn reset_fwd(&mut self) {
        for (i, p) in self.fwd_prefix.iter_mut().enumerate() {
            *p = i as u32;
        }
        self.fwd_divergence.fill(0);
    }

    /// Reset only backward PBWT
    pub fn reset_bwd(&mut self) {
        for (i, p) in self.bwd_prefix.iter_mut().enumerate() {
            *p = i as u32;
        }
        self.bwd_divergence.fill(0);
    }

    /// Update forward PBWT with alleles at a marker
    pub fn update(&mut self, alleles: &[u8], n_alleles: usize, marker: usize) {
        self.pbwt_updater.fwd_update(
            alleles,
            n_alleles,
            marker,
            &mut self.fwd_prefix,
            &mut self.fwd_divergence[..self.n_ref_haps],
        );
    }

    /// Update backward PBWT with alleles at a marker
    pub fn update_bwd(&mut self, alleles: &[u8], n_alleles: usize, marker: usize) {
        self.pbwt_updater.fwd_update(
            alleles,
            n_alleles,
            marker,
            &mut self.bwd_prefix,
            &mut self.bwd_divergence[..self.n_ref_haps],
        );
    }

    /// Find IBS haplotypes for a target allele at current PBWT position
    ///
    /// This is the key method for imputation: we find reference haplotypes that
    /// are IBS with the target by locating where the target's allele pattern
    /// would sort in the PBWT prefix array.
    ///
    /// # Arguments
    /// * `target_allele` - The target's allele at the last marker of this step
    /// * `ref_alleles` - Reference panel alleles at the last marker (indexed by hap)
    /// * `step` - Current step number (for divergence checking)
    ///
    /// Returns haplotype indices that are IBS with the target
    pub fn find_ibs_haps_for_allele(
        &self,
        target_allele: u8,
        ref_alleles: &[u8],
        step: usize,
    ) -> Vec<u32> {
        let n = self.fwd_prefix.len();
        if n == 0 {
            return Vec::new();
        }

        // Find the region in the prefix array where haplotypes have the same
        // allele as the target. In PBWT, haplotypes are sorted by their prefix
        // sequences, so haplotypes with the same allele at the current position
        // tend to be grouped together.
        let (region_start, region_end) = self.find_allele_region(target_allele, ref_alleles);

        // Start from the middle of the matching region
        let target_pos = if region_end > region_start {
            (region_start + region_end) / 2
        } else {
            n / 2
        };

        // Expand from target position to find IBS neighbors
        let mut result = Vec::with_capacity(self.n_ibs_haps);
        let step_i32 = step as i32;

        let mut left = target_pos;
        let mut right = if target_pos < n { target_pos + 1 } else { n };

        // First, add haplotypes with matching allele and valid IBS
        while result.len() < self.n_ibs_haps && (left > 0 || right < n) {
            // Check left neighbor
            if left > 0 {
                left -= 1;
                let hap = self.fwd_prefix[left];
                if hap < self.n_ref_haps as u32 {
                    let hap_allele = ref_alleles.get(hap as usize).copied().unwrap_or(255);
                    // Prefer haplotypes with matching allele
                    if hap_allele == target_allele {
                        let div = self.fwd_divergence.get(left + 1).copied().unwrap_or(0);
                        if div <= step_i32 {
                            result.push(hap);
                        }
                    }
                }
            }

            // Check right neighbor
            if right < n && result.len() < self.n_ibs_haps {
                let hap = self.fwd_prefix[right];
                if hap < self.n_ref_haps as u32 {
                    let hap_allele = ref_alleles.get(hap as usize).copied().unwrap_or(255);
                    if hap_allele == target_allele {
                        let div = self.fwd_divergence.get(right).copied().unwrap_or(0);
                        if div <= step_i32 {
                            result.push(hap);
                        }
                    }
                }
                right += 1;
            }
        }

        // If we still don't have enough, add nearby haplotypes regardless of allele
        if result.len() < self.n_ibs_haps {
            left = target_pos;
            right = if target_pos < n { target_pos + 1 } else { n };

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

    /// Find the region in prefix array where haplotypes have the target allele
    fn find_allele_region(&self, target_allele: u8, ref_alleles: &[u8]) -> (usize, usize) {
        let mut start = None;
        let mut end = 0;

        for (i, &hap_idx) in self.fwd_prefix.iter().enumerate() {
            let hap_allele = ref_alleles.get(hap_idx as usize).copied().unwrap_or(255);
            if hap_allele == target_allele {
                if start.is_none() {
                    start = Some(i);
                }
                end = i + 1;
            }
        }

        (start.unwrap_or(0), end)
    }

    /// Legacy method for backwards compatibility
    pub fn find_ibs_haps(&self, target_allele: u8, step: usize) -> Vec<u32> {
        // Create a dummy ref_alleles array based on prefix order
        let ref_alleles: Vec<u8> = (0..self.n_ref_haps).map(|_| 0).collect();
        self.find_ibs_haps_for_allele(target_allele, &ref_alleles, step)
    }

    /// Find IBS haplotypes using BACKWARD PBWT
    ///
    /// This finds haplotypes that match well in the FUTURE direction.
    /// Java Beagle calls both fwdIbsHap() and bwdIbsHap() at each step.
    pub fn find_bwd_ibs_haps_for_allele(
        &self,
        target_allele: u8,
        ref_alleles: &[u8],
        step: usize,
    ) -> Vec<u32> {
        let n = self.bwd_prefix.len();
        if n == 0 {
            return Vec::new();
        }

        // Find region in backward prefix array with target allele
        let (region_start, region_end) = self.find_bwd_allele_region(target_allele, ref_alleles);

        let target_pos = if region_end > region_start {
            (region_start + region_end) / 2
        } else {
            n / 2
        };

        let step_i32 = step as i32;
        let mut result = Vec::with_capacity(self.n_ibs_haps);
        let mut left = target_pos;
        let mut right = if target_pos < n { target_pos + 1 } else { n };

        // First pass: prefer haplotypes with matching allele and recent divergence
        while result.len() < self.n_ibs_haps && (left > 0 || right < n) {
            if left > 0 {
                left -= 1;
                let hap = self.bwd_prefix[left];
                if hap < self.n_ref_haps as u32 {
                    let hap_allele = ref_alleles.get(hap as usize).copied().unwrap_or(255);
                    if hap_allele == target_allele {
                        let div = self.bwd_divergence.get(left).copied().unwrap_or(0);
                        if div <= step_i32 {
                            result.push(hap);
                        }
                    }
                }
            }

            if right < n && result.len() < self.n_ibs_haps {
                let hap = self.bwd_prefix[right];
                if hap < self.n_ref_haps as u32 {
                    let hap_allele = ref_alleles.get(hap as usize).copied().unwrap_or(255);
                    if hap_allele == target_allele {
                        let div = self.bwd_divergence.get(right).copied().unwrap_or(0);
                        if div <= step_i32 {
                            result.push(hap);
                        }
                    }
                }
                right += 1;
            }
        }

        // Fallback: add nearby haplotypes regardless of allele
        if result.len() < self.n_ibs_haps {
            left = target_pos;
            right = if target_pos < n { target_pos + 1 } else { n };

            while result.len() < self.n_ibs_haps && (left > 0 || right < n) {
                if left > 0 {
                    left -= 1;
                    let hap = self.bwd_prefix[left];
                    if hap < self.n_ref_haps as u32 && !result.contains(&hap) {
                        result.push(hap);
                    }
                }
                if right < n && result.len() < self.n_ibs_haps {
                    let hap = self.bwd_prefix[right];
                    if hap < self.n_ref_haps as u32 && !result.contains(&hap) {
                        result.push(hap);
                    }
                    right += 1;
                }
            }
        }

        result
    }

    /// Find region in backward prefix array where haplotypes have target allele
    fn find_bwd_allele_region(&self, target_allele: u8, ref_alleles: &[u8]) -> (usize, usize) {
        let mut start = None;
        let mut end = 0;

        for (i, &hap_idx) in self.bwd_prefix.iter().enumerate() {
            let hap_allele = ref_alleles.get(hap_idx as usize).copied().unwrap_or(255);
            if hap_allele == target_allele {
                if start.is_none() {
                    start = Some(i);
                }
                end = i + 1;
            }
        }

        (start.unwrap_or(0), end)
    }
}

/// HMM state selector for phasing (uses ImpIbs on mutable/raw genotypes)
///
/// This version operates on raw allele data and is designed for the phasing
/// pipeline where genotypes are modified during each iteration. For compressed
/// reference panels, use `ImpStates` instead.
pub struct ImpStatesMutable {
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

/// Dynamic HMM state selector for imputation (optimized with CodedPbwt)
///
/// Matches Java `imp/ImpStates.java`
pub struct ImpStates<'a> {
    /// Maximum number of HMM states
    max_states: usize,
    /// Reference panel with coded steps
    ref_panel: &'a RefPanelCoded,
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
    /// Number of IBS haplotypes to find per step
    n_ibs_haps: usize,
}

impl<'a> ImpStates<'a> {
    /// Create a new state selector
    pub fn new(
        ref_panel: &'a RefPanelCoded,
        max_states: usize,
        gen_positions: &[f64],
        config: &CodedStepsConfig,
    ) -> Self {
        let coded_steps = CodedSteps::new(gen_positions, config);
        let n_markers = gen_positions.len();

        Self {
            max_states,
            ref_panel,
            coded_steps,
            comp_haps: Vec::with_capacity(max_states),
            hap_to_last_ibs: HashMap::with_capacity(max_states),
            queue: BinaryHeap::with_capacity(max_states),
            n_markers,
            n_ibs_haps: config.n_ibs_haps,
        }
    }

    /// Select IBS-based HMM states for a target haplotype
    ///
    /// Uses BOTH forward and backward PBWT passes to find IBS matches,
    /// matching Java Beagle's behavior in LowFreqPhaseStates.
    ///
    /// # Arguments
    /// * `get_ref_allele` - Function to get reference allele at (marker, hap)
    /// * `target_alleles` - Target haplotype alleles at genotyped markers
    /// * `workspace` - Workspace with PBWT buffers
    /// * `hap_indices` - Output: reference haplotype for each state at each marker
    /// * `allele_match` - Output: whether state allele matches target at each marker
    ///
    /// # Returns
    /// Number of states
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

        // Create CodedPbwtView from workspace buffers
        let n_ref_haps = self.ref_panel.n_haps();
        let n_steps = self.ref_panel.n_steps();
        workspace.resize_with_ref(self.max_states, self.n_markers, n_ref_haps);

        // Store backward IBS haps for each step (to use during forward pass)
        let mut bwd_ibs_per_step: Vec<Vec<u32>> = vec![Vec::new(); n_steps];

        // STEP 1: Build backward PBWT by processing steps in REVERSE order
        // This allows finding haplotypes that match well going forward from each position
        {
            // Create backward PBWT view
            let mut pbwt_bwd = CodedPbwtView::new(
                &mut workspace.pbwt_prefix_bwd[..n_ref_haps],
                &mut workspace.pbwt_divergence_bwd[..n_ref_haps + 1],
            );

            for step_idx in (0..n_steps).rev() {
                let coded_step = self.ref_panel.step(step_idx);
                let step_start = self.coded_steps.step_start(step_idx);
                let step_end = self.coded_steps.step_end(step_idx);

                // Update backward PBWT (use legacy method since we don't have scratch access here)
                pbwt_bwd.update(coded_step);

                // Extract target alleles for this step range
                let target_seq: Vec<u8> = (step_start..step_end)
                    .map(|m| target_alleles.get(m).copied().unwrap_or(255))
                    .collect();

                // Find backward IBS haplotypes
                let bwd_ibs: Vec<u32> = if let Some(target_pattern) = coded_step.match_sequence(&target_seq) {
                    pbwt_bwd.find_ibs(target_pattern, coded_step, self.n_ibs_haps)
                        .into_iter()
                        .map(|h| h.0)
                        .collect()
                } else {
                    let closest_pattern = coded_step.closest_pattern(&target_seq);
                    pbwt_bwd.find_ibs(closest_pattern, coded_step, self.n_ibs_haps)
                        .into_iter()
                        .map(|h| h.0)
                        .collect()
                };

                bwd_ibs_per_step[step_idx] = bwd_ibs;
            }
        }

        // STEP 2: Build forward PBWT and collect IBS haps from BOTH directions
        // Use counting sort optimization with workspace scratch buffers
        {
            let mut pbwt_fwd = CodedPbwtView::new(
                &mut workspace.pbwt_prefix[..n_ref_haps],
                &mut workspace.pbwt_divergence[..n_ref_haps + 1],
            );

            for step_idx in 0..n_steps {
                let coded_step = self.ref_panel.step(step_idx);
                let step_start = self.coded_steps.step_start(step_idx);
                let step_end = self.coded_steps.step_end(step_idx);

                // Update forward PBWT
                pbwt_fwd.update(coded_step);

                // Extract target alleles for this step range
                let target_seq: Vec<u8> = (step_start..step_end)
                    .map(|m| target_alleles.get(m).copied().unwrap_or(255))
                    .collect();

                // Find forward IBS haplotypes
                let fwd_ibs: Vec<u32> = if let Some(target_pattern) = coded_step.match_sequence(&target_seq) {
                    pbwt_fwd.find_ibs(target_pattern, coded_step, self.n_ibs_haps)
                        .into_iter()
                        .map(|h| h.0)
                        .collect()
                } else {
                    let closest_pattern = coded_step.closest_pattern(&target_seq);
                    pbwt_fwd.find_ibs(closest_pattern, coded_step, self.n_ibs_haps)
                        .into_iter()
                        .map(|h| h.0)
                        .collect()
                };

                // Update composite haplotypes with IBS matches from BOTH directions
                // Java Beagle: addIbsHap(ibsHaps.fwdIbsHap(...)); addIbsHap(ibsHaps.bwdIbsHap(...));
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
        self.build_output(get_ref_allele, target_alleles, n_states, hap_indices, allele_match);

        n_states
    }

    fn initialize(&mut self) {
        self.hap_to_last_ibs.clear();
        self.comp_haps.clear();
        self.queue.clear();
    }

    fn update_with_ibs_hap(&mut self, hap: u32, step: i32) {
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
        use rand::seq::SliceRandom;
        use rand::rng;

        let n_ref_haps = self.ref_panel.n_haps();
        let n_states = self.max_states.min(n_ref_haps);
        let mut rng = rng();

        // Create list of all haplotype indices and shuffle
        let mut hap_indices: Vec<u32> = (0..n_ref_haps as u32).collect();
        hap_indices.shuffle(&mut rng);

        // Take first n_states from shuffled list
        for &h in hap_indices.iter().take(n_states) {
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
                    allele_match[m][j] = target_allele == 255 || ref_allele == target_allele;
                }
            }
        }
    }
}

impl ImpStatesMutable {
    /// Create a new state selector (legacy version for phasing)
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

    /// Select IBS-based HMM states for a target haplotype (legacy version)
    ///
    /// Uses BOTH forward and backward PBWT passes to find IBS matches,
    /// matching Java Beagle's LowFreqPhaseStates behavior.
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

        let n_steps = self.coded_steps.n_steps();

        // First, build backward PBWT by scanning markers in REVERSE order
        // This allows us to find haplotypes that match well going forward from each position
        self.ibs.reset_bwd();
        for step in (0..n_steps).rev() {
            let step_start = self.coded_steps.step_start(step);
            let step_end = self.coded_steps.step_end(step);

            // Update backward PBWT for markers in reverse
            for m in (step_start..step_end).rev() {
                let mut alleles: Vec<u8> = Vec::with_capacity(self.ibs.n_ref_haps);
                for h in 0..self.ibs.n_ref_haps as u32 {
                    alleles.push(get_ref_allele(m, h));
                }
                let n_alleles = alleles.iter().copied().max().unwrap_or(0) as usize + 1;
                self.ibs.update_bwd(&alleles, n_alleles.max(2), m);
            }
        }

        // Now process forward, collecting IBS haps from BOTH directions
        self.ibs.reset_fwd();
        for step in 0..n_steps {
            let step_start = self.coded_steps.step_start(step);
            let step_end = self.coded_steps.step_end(step);

            // Update forward PBWT for markers in this step
            for m in step_start..step_end {
                let mut alleles: Vec<u8> = Vec::with_capacity(self.ibs.n_ref_haps);
                for h in 0..self.ibs.n_ref_haps as u32 {
                    alleles.push(get_ref_allele(m, h));
                }
                let n_alleles = alleles.iter().copied().max().unwrap_or(0) as usize + 1;
                self.ibs.update(&alleles, n_alleles.max(2), m);
            }

            // Find IBS haplotypes for target using target's allele at last marker of step
            let last_marker = step_end.saturating_sub(1);
            let target_allele = target_alleles.get(last_marker).copied().unwrap_or(255);

            // Get reference alleles at the last marker for allele-aware IBS finding
            let ref_alleles: Vec<u8> = (0..self.ibs.n_ref_haps as u32)
                .map(|h| get_ref_allele(last_marker, h))
                .collect();

            // Get IBS haps from FORWARD PBWT (matches from past)
            let fwd_ibs_haps = self.ibs.find_ibs_haps_for_allele(target_allele, &ref_alleles, step);

            // Get IBS haps from BACKWARD PBWT (matches from future)
            // Java Beagle: addIbsHap(ibsHaps.fwdIbsHap(...)); addIbsHap(ibsHaps.bwdIbsHap(...));
            let bwd_ibs_haps = self.ibs.find_bwd_ibs_haps_for_allele(target_allele, &ref_alleles, step);

            // Update composite haplotypes with IBS matches from BOTH directions
            for hap in fwd_ibs_haps {
                self.update_with_ibs_hap(hap, step as i32);
            }
            for hap in bwd_ibs_haps {
                self.update_with_ibs_hap(hap, step as i32);
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

    fn update_with_ibs_hap(&mut self, hap: u32, step: i32) {
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

            self.hap_to_last_ibs.insert(hap, step);
        }
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
        use rand::seq::SliceRandom;
        use rand::rng;

        let n_states = self.max_states.min(self.ibs.n_ref_haps);
        let mut rng = rng();

        // Create list of all haplotype indices and shuffle
        let mut hap_indices: Vec<u32> = (0..self.ibs.n_ref_haps as u32).collect();
        hap_indices.shuffle(&mut rng);

        // Take first n_states from shuffled list
        for &h in hap_indices.iter().take(n_states) {
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
                    allele_match[m][j] = target_allele == 255 || ref_allele == target_allele;
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
