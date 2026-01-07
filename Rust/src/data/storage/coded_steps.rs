//! # Coded Steps Dictionary Compression
//!
//! Implements dictionary compression for haplotype sequences within genetic intervals ("steps").
//! This matches Java `imp/CodedSteps.java` and `bref/SeqCoder3.java`.
//!
//! Key concepts:
//! - Steps: Small intervals of markers based on genetic distance (default 0.1 cM)
//! - Patterns: Unique allele sequences within a step
//! - Dictionary: Maps haplotypes to their sequence pattern index
//!
//! This dramatically reduces computation for large reference panels by:
//! 1. Collapsing identical haplotype segments into single patterns
//! 2. Operating on pattern indices rather than raw alleles
//! 3. Enabling efficient PBWT operations on the compressed representation

use crate::data::haplotype::HapIdx;
use crate::data::marker::MarkerIdx;
use crate::data::storage::matrix::GenotypeMatrix;
use crate::data::storage::phase_state::PhaseState;

/// A single coded step containing dictionary-compressed haplotypes
///
/// Memory-efficient design: stores representative haplotype indices instead of
/// materialized allele sequences. Alleles are looked up on-demand from the
/// genotype matrix when needed. This matches Java Beagle's approach and
/// dramatically reduces memory usage for large reference panels.
#[derive(Clone, Debug)]
pub struct CodedStep {
    /// Start marker (inclusive)
    pub start: usize,
    /// End marker (exclusive)
    pub end: usize,
    /// Number of unique patterns in this step
    n_patterns: usize,
    /// Mapping from haplotype to pattern index
    hap_to_pattern: Vec<u16>,
    /// Representative haplotype index for each pattern (for on-demand allele lookup)
    /// This replaces Vec<Vec<u8>> to save ~3GB for large reference panels
    pattern_rep_hap: Vec<u32>,
}

impl CodedStep {
    /// Create a new coded step from genotype data
    ///
    /// Memory-efficient: stores only representative haplotype indices, not allele sequences.
    /// Alleles are looked up on-demand when needed, saving ~3GB for large reference panels.
    pub fn new<S: PhaseState>(gt: &GenotypeMatrix<S>, start: usize, end: usize) -> Self {
        let n_haps = gt.n_haplotypes();
        let n_markers = end - start;

        if n_markers == 0 {
            return Self {
                start,
                end,
                n_patterns: 0,
                hap_to_pattern: vec![0; n_haps],
                pattern_rep_hap: Vec::new(),
            };
        }

        // Use HashMap for pattern deduplication during construction
        // Key: allele sequence, Value: (pattern_index, representative_hap)
        use std::collections::HashMap;
        let mut pattern_map: HashMap<Vec<u8>, (u16, u32)> = HashMap::new();
        let mut hap_to_pattern = Vec::with_capacity(n_haps);
        let mut pattern_rep_hap: Vec<u32> = Vec::new();

        // Pre-allocate scratch buffer - reused for each haplotype
        let mut scratch = vec![0u8; n_markers];

        for h in 0..n_haps {
            let hap = HapIdx::new(h as u32);

            // Fill scratch buffer in-place
            for (i, m) in (start..end).enumerate() {
                scratch[i] = gt.allele(MarkerIdx::new(m as u32), hap);
            }

            // Look up or insert pattern
            let pattern_idx = if let Some(&(idx, _)) = pattern_map.get(&scratch) {
                idx
            } else {
                // New unique pattern - store representative haplotype index
                let idx = pattern_rep_hap.len() as u16;
                pattern_rep_hap.push(h as u32);
                pattern_map.insert(scratch.clone(), (idx, h as u32));
                idx
            };

            hap_to_pattern.push(pattern_idx);
        }

        Self {
            start,
            end,
            n_patterns: pattern_rep_hap.len(),
            hap_to_pattern,
            pattern_rep_hap,
        }
    }

    /// Number of markers in this step
    pub fn n_markers(&self) -> usize {
        self.end - self.start
    }

    /// Number of unique patterns
    pub fn n_patterns(&self) -> usize {
        self.n_patterns
    }

    /// Get pattern index for a haplotype
    pub fn pattern(&self, hap: HapIdx) -> u16 {
        self.hap_to_pattern[hap.0 as usize]
    }

    /// Compression ratio (n_haps / n_patterns)
    pub fn compression_ratio(&self) -> f32 {
        if self.n_patterns == 0 {
            1.0
        } else {
            self.hap_to_pattern.len() as f32 / self.n_patterns as f32
        }
    }

    /// Match a target allele sequence to a pattern index
    /// Returns None if the sequence doesn't exist in the reference
    ///
    /// Uses a closure to look up reference alleles on-demand, saving ~3GB
    /// vs storing all pattern allele sequences in memory.
    ///
    /// # Arguments
    /// * `alleles` - Target allele sequence for this step
    /// * `get_allele` - Closure: (marker_idx, hap_idx) -> allele
    pub fn match_sequence_with<F>(&self, alleles: &[u8], get_allele: F) -> Option<u16>
    where
        F: Fn(usize, u32) -> u8,
    {
        if alleles.len() != self.n_markers() {
            return None;
        }

        // Look for exact match by comparing against representative haplotypes
        for (idx, &rep_hap) in self.pattern_rep_hap.iter().enumerate() {
            let matches = (self.start..self.end).enumerate().all(|(i, m)| {
                let ref_allele = get_allele(m, rep_hap);
                alleles[i] == ref_allele || alleles[i] == 255 || ref_allele == 255
            });
            if matches {
                return Some(idx as u16);
            }
        }
        None
    }

    /// Find closest pattern to target sequence (by Hamming distance)
    /// Always returns a pattern (never None)
    ///
    /// Uses a closure to look up reference alleles on-demand.
    ///
    /// # Arguments
    /// * `alleles` - Target allele sequence for this step
    /// * `get_allele` - Closure: (marker_idx, hap_idx) -> allele
    pub fn closest_pattern_with<F>(&self, alleles: &[u8], get_allele: F) -> u16
    where
        F: Fn(usize, u32) -> u8,
    {
        if alleles.len() != self.n_markers() || self.n_patterns == 0 {
            return 0;
        }

        let mut best_pattern = 0u16;
        let mut best_distance = usize::MAX;

        for (idx, &rep_hap) in self.pattern_rep_hap.iter().enumerate() {
            let distance = (self.start..self.end)
                .enumerate()
                .filter(|(i, m)| {
                    let ref_allele = get_allele(*m, rep_hap);
                    let target_allele = alleles[*i];
                    ref_allele != target_allele && ref_allele != 255 && target_allele != 255
                })
                .count();

            if distance < best_distance {
                best_distance = distance;
                best_pattern = idx as u16;
            }
        }

        best_pattern
    }
}

/// Collection of coded steps for a chromosome
#[derive(Clone, Debug)]
pub struct RefPanelCoded {
    /// Coded steps
    steps: Vec<CodedStep>,
    /// Number of markers
    n_markers: usize,
}

impl RefPanelCoded {
    /// Create coded reference panel from genotype matrix
    pub fn new<S: PhaseState>(gt: &GenotypeMatrix<S>, step_starts: &[usize]) -> Self {
        let n_markers = gt.n_markers();

        let mut steps = Vec::with_capacity(step_starts.len());

        for (i, &start) in step_starts.iter().enumerate() {
            let end = step_starts.get(i + 1).copied().unwrap_or(n_markers);
            steps.push(CodedStep::new(gt, start, end));
        }

        Self {
            steps,
            n_markers,
        }
    }

    /// Create from genetic positions with default step size
    pub fn from_gen_positions<S: PhaseState>(gt: &GenotypeMatrix<S>, gen_positions: &[f64], step_cm: f64) -> Self {
        let step_starts = compute_step_starts(gen_positions, step_cm);
        Self::new(gt, &step_starts)
    }

    /// Number of steps
    pub fn n_steps(&self) -> usize {
        self.steps.len()
    }

    /// Get a step
    pub fn step(&self, idx: usize) -> &CodedStep {
        &self.steps[idx]
    }

    /// Total number of patterns across all steps
    pub fn total_patterns(&self) -> usize {
        self.steps.iter().map(|s| s.n_patterns()).sum()
    }

    /// Average compression ratio
    pub fn avg_compression_ratio(&self) -> f32 {
        if self.steps.is_empty() {
            return 1.0;
        }
        let sum: f32 = self.steps.iter().map(|s| s.compression_ratio()).sum();
        sum / self.steps.len() as f32
    }

    /// Number of markers
    pub fn n_markers(&self) -> usize {
        self.n_markers
    }
}

/// Compute step start indices from genetic positions
pub fn compute_step_starts(gen_positions: &[f64], step_cm: f64) -> Vec<usize> {
    if gen_positions.is_empty() {
        return vec![0];
    }

    let mut step_starts = vec![0];

    // First step is half-length
    let mut next_pos = gen_positions[0] + step_cm / 2.0;
    let mut idx = 1;

    while idx < gen_positions.len() {
        if gen_positions[idx] >= next_pos {
            step_starts.push(idx);
            next_pos = gen_positions[idx] + step_cm;
        }
        idx += 1;
    }

    step_starts
}

/// PBWT operations on coded steps using external buffers
///
/// This version uses workspace buffers to avoid repeated allocations
pub struct CodedPbwtView<'a> {
    /// Prefix array (borrowed from workspace)
    prefix: &'a mut [u32],
    /// Divergence array (borrowed from workspace)
    divergence: &'a mut [i32],
}

impl<'a> CodedPbwtView<'a> {
    /// Create forward PBWT view from workspace buffers
    pub fn new(prefix: &'a mut [u32], divergence: &'a mut [i32]) -> Self {
        for (i, p) in prefix.iter_mut().enumerate() {
            *p = i as u32;
        }
        divergence.fill(0);

        Self { prefix, divergence }
    }

    /// Create backward PBWT view from workspace buffers
    ///
    /// Initializes divergence to n_steps (end of chromosome) for min-aggregation
    pub fn new_backward(prefix: &'a mut [u32], divergence: &'a mut [i32], n_steps: usize) -> Self {
        for (i, p) in prefix.iter_mut().enumerate() {
            *p = i as u32;
        }
        divergence.fill(n_steps as i32);

        Self { prefix, divergence }
    }

    /// Update PBWT with a coded step using counting sort
    ///
    /// This optimized version avoids allocating Vec<Vec<>> per step by using
    /// flat counting sort with workspace scratch buffers.
    pub fn update_counting_sort(
        &mut self,
        step: &CodedStep,
        counts: &mut Vec<usize>,
        offsets: &mut Vec<usize>,
        prefix_scratch: &mut [u32],
        div_scratch: &mut [i32],
    ) {
        let n_haps = self.prefix.len();
        let n_patterns = step.n_patterns();

        if n_patterns == 0 || n_haps == 0 {
            return;
        }

        // Step 1: Count frequency of each pattern
        counts.clear();
        counts.resize(n_patterns, 0);
        for i in 0..n_haps {
            let hap = self.prefix[i];
            let pattern = step.pattern(HapIdx::new(hap)) as usize;
            if pattern < n_patterns {
                counts[pattern] += 1;
            }
        }

        // Step 2: Compute cumulative offsets (where each pattern bucket starts)
        offsets.clear();
        offsets.resize(n_patterns + 1, 0);
        let mut running = 0usize;
        for (i, &count) in counts.iter().enumerate() {
            offsets[i] = running;
            running += count;
        }
        offsets[n_patterns] = running;

        // Step 3: Distribute haplotypes to their sorted positions
        // We need to track current write position for each pattern
        let mut write_pos: Vec<usize> = offsets[..n_patterns].to_vec();
        let step_start = 0i32;

        for i in 0..n_haps {
            let hap = self.prefix[i];
            let div = self.divergence[i];
            let pattern = step.pattern(HapIdx::new(hap)) as usize;

            if pattern < n_patterns {
                let bucket_start = offsets[pattern];
                let pos = write_pos[pattern];

                prefix_scratch[pos] = hap;
                // Divergence: first element in bucket gets max(step_start, div)
                div_scratch[pos] = if pos == bucket_start {
                    step_start.max(div)
                } else {
                    div
                };

                write_pos[pattern] += 1;
            }
        }

        // Step 4: Copy results back to main arrays
        self.prefix[..n_haps].copy_from_slice(&prefix_scratch[..n_haps]);
        self.divergence[..n_haps].copy_from_slice(&div_scratch[..n_haps]);
    }

    /// Backward update PBWT with a coded step
    ///
    /// For backward PBWT, divergence represents where the match ENDS (not starts).
    /// We use min aggregation instead of max, and initialize with end-of-chromosome.
    pub fn update_backward(&mut self, step: &CodedStep, n_steps: usize) {
        let n_haps = self.prefix.len();
        let n_patterns = step.n_patterns();

        let mut buckets: Vec<Vec<(u32, i32)>> = vec![Vec::new(); n_patterns.max(1)];

        for i in 0..n_haps {
            let hap = self.prefix[i];
            let div = self.divergence[i];
            let pattern = step.pattern(HapIdx::new(hap)) as usize;
            if pattern < buckets.len() {
                buckets[pattern].push((hap, div));
            }
        }

        let mut idx = 0;
        let step_end = n_steps as i32;

        for bucket in &buckets {
            let bucket_start = idx;
            for &(hap, div) in bucket {
                self.prefix[idx] = hap;
                self.divergence[idx] = if idx == bucket_start {
                    step_end.min(div)
                } else {
                    div
                };
                idx += 1;
            }
        }
    }

    /// Find IBS haplotypes for a target pattern at current step
    pub fn find_ibs(&self, target_pattern: u16, step: &CodedStep, n_matches: usize) -> Vec<HapIdx> {
        let mut target_pos = None;
        for (i, &hap) in self.prefix.iter().enumerate() {
            if step.pattern(HapIdx::new(hap)) == target_pattern {
                target_pos = Some(i);
                break;
            }
        }

        let target_pos = match target_pos {
            Some(p) => p,
            None => self.prefix.len() / 2,
        };

        let mut result = Vec::with_capacity(n_matches);
        let mut left = target_pos;
        let mut right = target_pos + 1;

        while result.len() < n_matches && (left > 0 || right < self.prefix.len()) {
            if left > 0 {
                left -= 1;
                result.push(HapIdx::new(self.prefix[left]));
            }
            if right < self.prefix.len() && result.len() < n_matches {
                result.push(HapIdx::new(self.prefix[right]));
                right += 1;
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::ChromIdx;
    use crate::data::haplotype::Samples;
    use crate::data::marker::{Allele, Marker, Markers};
    use crate::data::storage::GenotypeColumn;
    use std::sync::Arc;

    fn make_test_matrix() -> GenotypeMatrix {
        let samples = Arc::new(Samples::from_ids(vec![
            "S1".to_string(),
            "S2".to_string(),
            "S3".to_string(),
        ]));

        let mut markers = Markers::new();
        markers.add_chrom("chr1");

        // 6 markers, 6 haplotypes
        // Pattern: markers 0-2 are step 1, markers 3-5 are step 2
        let allele_data = vec![
            // marker 0: [0,1, 0,1, 0,0] - 2 patterns: (0), (1)
            vec![0u8, 1, 0, 1, 0, 0],
            // marker 1: [0,1, 0,1, 0,0]
            vec![0, 1, 0, 1, 0, 0],
            // marker 2: [0,1, 0,1, 0,0]
            vec![0, 1, 0, 1, 0, 0],
            // marker 3: [0,0, 1,1, 0,1]
            vec![0, 0, 1, 1, 0, 1],
            // marker 4: [0,0, 1,1, 0,1]
            vec![0, 0, 1, 1, 0, 1],
            // marker 5: [0,0, 1,1, 0,1]
            vec![0, 0, 1, 1, 0, 1],
        ];

        let mut columns = Vec::new();
        for (i, alleles) in allele_data.iter().enumerate() {
            let m = Marker::new(
                ChromIdx::new(0),
                (i * 1000 + 100) as u32,
                None,
                Allele::Base(0),
                vec![Allele::Base(1)],
            );
            markers.push(m);
            columns.push(GenotypeColumn::from_alleles(alleles, 2));
        }

        GenotypeMatrix::new_unphased(markers, columns, samples)
    }

    #[test]
    fn test_coded_step() {
        let gt = make_test_matrix();
        let step = CodedStep::new(&gt, 0, 3);

        assert_eq!(step.n_markers(), 3);
        // n_haps() not exposed on CodedStep, but hap_to_pattern.len() would be 6

        // Should have 2 patterns: [0,0,0] and [1,1,1]
        assert_eq!(step.n_patterns(), 2);

        // Haplotypes 0,2,4 have pattern [0,0,0]
        // Haplotypes 1,3,5 have pattern [1,1,1]
        let p0 = step.pattern(HapIdx::new(0));
        let p2 = step.pattern(HapIdx::new(2));
        let p4 = step.pattern(HapIdx::new(4));
        assert_eq!(p0, p2);
        assert_eq!(p2, p4);

        let p1 = step.pattern(HapIdx::new(1));
        let p3 = step.pattern(HapIdx::new(3));
        assert_eq!(p1, p3);
        assert_ne!(p0, p1);

        // Check compression ratio
        assert!(step.compression_ratio() > 1.0);
    }

    #[test]
    fn test_ref_panel_coded() {
        let gt = make_test_matrix();
        let step_starts = vec![0, 3];
        let coded = RefPanelCoded::new(&gt, &step_starts);

        assert_eq!(coded.n_steps(), 2);
        assert_eq!(coded.n_markers(), 6);
    }

    #[test]
    fn test_compute_step_starts() {
        let gen_pos: Vec<f64> = (0..100).map(|i| i as f64 * 0.05).collect();
        let step_starts = compute_step_starts(&gen_pos, 0.1);

        assert!(!step_starts.is_empty());
        assert_eq!(step_starts[0], 0);

        // Steps should be roughly 0.1 cM apart
        for i in 1..step_starts.len() {
            let prev_pos = gen_pos[step_starts[i - 1]];
            let curr_pos = gen_pos[step_starts[i]];
            let diff = curr_pos - prev_pos;
            assert!(diff >= 0.05 && diff <= 0.2, "Step distance: {}", diff);
        }
    }
}
