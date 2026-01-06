//! # Imputation State Selection
//!
//! Implements dynamic HMM state selection using PBWT-based IBS matching.
//! This matches Java `imp/ImpStates.java` and `imp/ImpIbs.java`.
//!
//! Key concepts:
//! - States are "composite haplotypes" - mosaics of reference haplotype segments
//! - PBWT finds IBS matches at each step (marker cluster)
//! - A priority queue tracks which composite haplotypes to keep/replace

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use crate::data::storage::coded_steps::{CodedPbwtView, RefPanelCoded};
use crate::model::pbwt::PbwtDivUpdater;
use crate::utils::workspace::ImpWorkspace;

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

// ============================================================================
// Threaded Arena Layout (Linked List in Vecs) for O(1) Updates
// ============================================================================

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
    // List of active state indices (holes are possible if we supported removal, 
    // but here we only add/replace)
    // We assume states are 0..n_states
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
        // Note: In standard Beagle logic, we might need to handle cases where 
        // start_marker <= prev_start (complete overwrite). 
        // But usually we move forward.
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
    pub fn hap_at(&mut self, state_idx: usize, marker: usize) -> u32 {
        let mut cur = self.state_cursors[state_idx] as usize;

        // Advance while marker is past this segment's end
        // (Fast path: most checks just return current)
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
        // Reset cursors to heads
        // Optimized: copy_from_slice is fast
        self.state_cursors.copy_from_slice(&self.state_heads);
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
        if n_alleles <= 2 {
            self.pbwt_updater.fwd_update_biallelic(
                alleles,
                marker,
                &mut self.fwd_prefix,
                &mut self.fwd_divergence[..self.n_ref_haps],
            );
        } else {
            self.pbwt_updater.fwd_update(
                alleles,
                n_alleles,
                marker,
                &mut self.fwd_prefix,
                &mut self.fwd_divergence[..self.n_ref_haps],
            );
        }
    }

    /// Update backward PBWT with alleles at a marker
    ///
    /// NOTE: This correctly uses bwd_update() which has different logic than fwd_update():
    /// - Initializes p array with marker-1 (not marker+1)
    /// - Uses min() for propagation (not max())
    /// - Resets p[allele] with i32::MAX (not i32::MIN)
    pub fn update_bwd(&mut self, alleles: &[u8], n_alleles: usize, marker: usize) {
        if n_alleles <= 2 {
            self.pbwt_updater.bwd_update_biallelic(
                alleles,
                marker,
                &mut self.bwd_prefix,
                &mut self.bwd_divergence[..self.n_ref_haps],
            );
        } else {
            self.pbwt_updater.bwd_update(
                alleles,
                n_alleles,
                marker,
                &mut self.bwd_prefix,
                &mut self.bwd_divergence[..self.n_ref_haps],
            );
        }
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
    /// Composite haplotypes (the HMM states) - Optimized Arena
    threaded_haps: ThreadedHaps,
    /// Map from reference haplotype to last IBS step
    hap_to_last_ibs: HashMap<u32, i32>,
    /// Priority queue for managing composite haplotypes
    queue: BinaryHeap<CompHapEntry>,
    /// Number of markers
    n_markers: usize,
    /// Number of IBS haplotypes to find per step
    n_ibs_haps: usize,
    /// Random seed for reproducibility (combined from global seed + sample index)
    seed: u64,
}

impl<'a> ImpStates<'a> {
    /// Create a new state selector
    ///
    /// # Arguments
    /// * `ref_panel` - Reference panel with coded steps
    /// * `max_states` - Maximum number of HMM states
    /// * `gen_positions` - Genetic positions for markers
    /// * `config` - Coded steps configuration
    /// * `seed` - Random seed for reproducibility (should be combined from global seed + sample index)
    pub fn new(
        ref_panel: &'a RefPanelCoded,
        max_states: usize,
        gen_positions: &[f64],
        config: &CodedStepsConfig,
        seed: u64,
    ) -> Self {
        let coded_steps = CodedSteps::new(gen_positions, config);
        let n_markers = gen_positions.len();

        Self {
            max_states,
            ref_panel,
            coded_steps,
            threaded_haps: ThreadedHaps::new(max_states, max_states * 4, n_markers),
            hap_to_last_ibs: HashMap::with_capacity(max_states),
            queue: BinaryHeap::with_capacity(max_states),
            n_markers,
            n_ibs_haps: config.n_ibs_haps,
            seed,
        }
    }

    /// Select IBS-based HMM states for a target haplotype using recursive partitioning
    ///
    /// This is the correct Java Beagle algorithm from `ImpIbs.java`:
    /// - Uses pre-computed IBS haplotypes from recursive partitioning
    /// - Groups haplotypes by shared allele sequences at each step
    ///
    /// # Arguments
    /// * `get_ref_allele` - Function to get reference allele at (marker, hap)
    /// * `target_alleles` - Target haplotype alleles at genotyped markers
    /// * `ibs_finder` - Pre-computed recursive partitioning IBS finder
    /// Legacy method using PBWT-based IBS (kept for compatibility)
    ///
    /// Uses BOTH forward and backward PBWT passes to find IBS matches.
    /// For the correct Java Beagle behavior, use `ibs_states_partition` instead.
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
        let n_ibs_haps = 8; // Default value
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

                // Update forward PBWT using counting sort optimization
                pbwt_fwd.update_counting_sort(
                    coded_step,
                    &mut workspace.sort_counts,
                    &mut workspace.sort_offsets,
                    &mut workspace.sort_prefix_scratch,
                    &mut workspace.sort_div_scratch,
                );

                // Extract target alleles for this step range
                let target_seq: Vec<u8> = (step_start..step_end)
                    .map(|m| target_alleles.get(m).copied().unwrap_or(255))
                    .collect();

                // Find forward IBS haplotypes
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
        self.build_output(
            get_ref_allele,
            target_alleles,
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
            // Hap not currently tracked
            self.update_queue_head();

            if self.queue.len() == self.max_states {
                // Replace oldest composite haplotype
                if let Some(mut head) = self.queue.pop() {
                    let mid_step = (head.last_ibs_step + step) / 2;
                    let start_marker = self.coded_steps.step_start(mid_step.max(0) as usize);

                    self.hap_to_last_ibs.remove(&head.hap);

                    // Update composite haplotype with new segment
                    if head.comp_hap_idx < self.threaded_haps.n_states() {
                        self.threaded_haps.add_segment(
                            head.comp_hap_idx,
                            hap,
                            start_marker,
                        );
                    }

                    head.hap = hap;
                    head.last_ibs_step = step;
                    head.start_marker = start_marker;
                    self.queue.push(head);
                }
            } else {
                // Add new composite haplotype
                let comp_hap_idx = self.threaded_haps.push_new(hap);
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
            let last_ibs = self
                .hap_to_last_ibs
                .get(&head.hap)
                .copied()
                .unwrap_or(i32::MIN);
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
        use rand::SeedableRng;
        use rand::seq::SliceRandom;
        use rand::rngs::StdRng;

        let n_ref_haps = self.ref_panel.n_haps();
        let n_states = self.max_states.min(n_ref_haps);

        // Use deterministic RNG seeded from stored seed for reproducibility
        let mut rng = StdRng::seed_from_u64(self.seed);

        // Create list of all haplotype indices and shuffle
        let mut hap_indices: Vec<u32> = (0..n_ref_haps as u32).collect();
        hap_indices.shuffle(&mut rng);

        // Take first n_states from shuffled list
        for &h in hap_indices.iter().take(n_states) {
            let comp_hap_idx = self.threaded_haps.push_new(h);
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
        self.threaded_haps.reset_cursors();

        // Build output for each marker
        for m in 0..self.n_markers {
            let target_allele = target_alleles.get(m).copied().unwrap_or(255);

            for (j, entry) in self.queue.iter().take(n_states).enumerate() {
                if entry.comp_hap_idx < self.threaded_haps.n_states() {
                    let hap = self.threaded_haps.hap_at(entry.comp_hap_idx, m);
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
