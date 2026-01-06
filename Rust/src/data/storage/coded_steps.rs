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
use crate::data::storage::GenotypeMatrix;

/// A single coded step containing dictionary-compressed haplotypes
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
    /// Allele sequences for each pattern: patterns[pattern_idx][marker_offset] = allele
    patterns: Vec<Vec<u8>>,
}

impl CodedStep {
    /// Create a new coded step from genotype data
    ///
    /// OPTIMIZATION: Uses a reusable scratch buffer to avoid allocating a new Vec<u8>
    /// for every haplotype. Only unique patterns are allocated and stored.
    /// This reduces heap allocations from O(n_haps * n_steps) to O(n_unique_patterns).
    pub fn new(gt: &GenotypeMatrix, start: usize, end: usize) -> Self {
        let n_haps = gt.n_haplotypes();
        let n_markers = end - start;

        if n_markers == 0 {
            return Self {
                start,
                end,
                n_patterns: 0,
                hap_to_pattern: vec![0; n_haps],
                patterns: Vec::new(),
            };
        }

        // Pre-allocate scratch buffer - reused for each haplotype (avoids O(n_haps) allocations)
        let mut scratch = vec![0u8; n_markers];

        // Store unique patterns and their indices
        let mut patterns: Vec<Vec<u8>> = Vec::new();
        let mut hap_to_pattern = Vec::with_capacity(n_haps);

        for h in 0..n_haps {
            let hap = HapIdx::new(h as u32);

            // Fill scratch buffer in-place (NO allocation)
            for (i, m) in (start..end).enumerate() {
                scratch[i] = gt.allele(MarkerIdx::new(m as u32), hap);
            }

            // Look up pattern using linear search for small pattern counts
            // This is typically faster than HashMap for <100 patterns due to cache locality,
            // and reference panels typically have high compression ratios (many haps per pattern)
            let pattern_idx = patterns
                .iter()
                .position(|p| p.as_slice() == scratch.as_slice())
                .map(|i| i as u16)
                .unwrap_or_else(|| {
                    // Only allocate when we discover a unique pattern
                    let idx = patterns.len() as u16;
                    patterns.push(scratch.clone()); // Only allocation per unique pattern
                    idx
                });

            hap_to_pattern.push(pattern_idx);
        }

        Self {
            start,
            end,
            n_patterns: patterns.len(),
            hap_to_pattern,
            patterns,
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

    /// Number of haplotypes
    pub fn n_haps(&self) -> usize {
        self.hap_to_pattern.len()
    }

    /// Get pattern index for a haplotype
    pub fn pattern(&self, hap: HapIdx) -> u16 {
        self.hap_to_pattern[hap.0 as usize]
    }

    /// Get allele at a marker for a haplotype
    pub fn hap_allele(&self, hap: HapIdx, marker_offset: usize) -> u8 {
        let pattern = self.hap_to_pattern[hap.0 as usize] as usize;
        self.patterns[pattern][marker_offset]
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
    pub fn match_sequence(&self, alleles: &[u8]) -> Option<u16> {
        if alleles.len() != self.n_markers() {
            return None;
        }

        // Look for exact match in patterns
        self.patterns
            .iter()
            .position(|p| p == alleles)
            .map(|idx| idx as u16)
    }

    /// Find closest pattern to target sequence (by Hamming distance)
    /// Always returns a pattern (never None)
    pub fn closest_pattern(&self, alleles: &[u8]) -> u16 {
        if alleles.len() != self.n_markers() {
            // Return most common pattern (usually index 0)
            return 0;
        }

        let mut best_pattern = 0u16;
        let mut best_distance = usize::MAX;

        for (idx, pattern) in self.patterns.iter().enumerate() {
            let distance = pattern
                .iter()
                .zip(alleles.iter())
                .filter(|(p, a)| **p != **a && **p != 255 && **a != 255)
                .count();

            if distance < best_distance {
                best_distance = distance;
                best_pattern = idx as u16;
            }
        }

        best_pattern
    }

    /// Append a haplotype to this step.
    ///
    /// Checks if the allele sequence matches an existing pattern.
    /// If yes, reuses the pattern index. If no, adds a new pattern.
    /// Returns the pattern index assigned to the haplotype.
    pub fn append_haplotype(&mut self, alleles: &[u8]) -> u16 {
        // Look for existing pattern match
        if let Some(idx) = self
            .patterns
            .iter()
            .position(|p| p.as_slice() == alleles)
        {
            let pattern_idx = idx as u16;
            self.hap_to_pattern.push(pattern_idx);
            pattern_idx
        } else {
            // Add new pattern
            let pattern_idx = self.patterns.len() as u16;
            self.patterns.push(alleles.to_vec());
            self.n_patterns = self.patterns.len();
            self.hap_to_pattern.push(pattern_idx);
            pattern_idx
        }
    }
}

/// Collection of coded steps for a chromosome
#[derive(Clone, Debug)]
pub struct RefPanelCoded {
    /// Coded steps
    steps: Vec<CodedStep>,
    /// Number of markers
    n_markers: usize,
    /// Number of haplotypes
    n_haps: usize,
}

impl RefPanelCoded {
    /// Create coded reference panel from genotype matrix
    pub fn new(gt: &GenotypeMatrix, step_starts: &[usize]) -> Self {
        let n_markers = gt.n_markers();
        let n_haps = gt.n_haplotypes();

        let mut steps = Vec::with_capacity(step_starts.len());

        for (i, &start) in step_starts.iter().enumerate() {
            let end = step_starts.get(i + 1).copied().unwrap_or(n_markers);
            steps.push(CodedStep::new(gt, start, end));
        }

        Self {
            steps,
            n_markers,
            n_haps,
        }
    }

    /// Create from genetic positions with default step size
    pub fn from_gen_positions(gt: &GenotypeMatrix, gen_positions: &[f64], step_cm: f64) -> Self {
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

    /// Get step index for a marker
    pub fn step_for_marker(&self, marker: usize) -> usize {
        for (i, step) in self.steps.iter().enumerate() {
            if marker >= step.start && marker < step.end {
                return i;
            }
        }
        self.steps.len().saturating_sub(1)
    }

    /// Get allele for a haplotype at a marker
    pub fn allele(&self, marker: usize, hap: HapIdx) -> u8 {
        let step_idx = self.step_for_marker(marker);
        let step = &self.steps[step_idx];
        let offset = marker - step.start;
        step.hap_allele(hap, offset)
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

    /// Number of haplotypes
    pub fn n_haps(&self) -> usize {
        self.n_haps
    }

    /// Get mutable reference to a step
    pub fn step_mut(&mut self, idx: usize) -> &mut CodedStep {
        &mut self.steps[idx]
    }

    /// Append target haplotypes to the coded reference panel.
    ///
    /// For each step, extracts the relevant markers from the target genotype matrix
    /// (using the alignment to map target markers to reference markers and handle 
    /// strand flips), then appends each target haplotype to the step.
    ///
    /// # Arguments
    /// * `target_gt` - Target genotype matrix (with target markers)
    /// * `ref_to_target` - For each reference marker, the corresponding target marker index (-1 if not in target)
    /// * `map_allele` - Function to map target allele to reference allele space
    ///
    /// After this call, the panel contains `n_ref_haps + n_target_haps` haplotypes.
    pub fn append_target_haplotypes<F>(
        &mut self,
        target_gt: &GenotypeMatrix,
        ref_to_target: &[i32],
        map_allele: F,
    ) where
        F: Fn(usize, u8) -> u8,
    {
        let n_target_haps = target_gt.n_haplotypes();
        if n_target_haps == 0 {
            return;
        }

        // Process each step
        for step in &mut self.steps {
            let n_markers_in_step = step.end - step.start;
            let mut allele_buffer = vec![0u8; n_markers_in_step];

            // For each target haplotype
            for h in 0..n_target_haps {
                let hap_idx = HapIdx::new(h as u32);

                // Extract alleles for this step's marker range
                for (offset, ref_m) in (step.start..step.end).enumerate() {
                    let target_m_idx = ref_to_target.get(ref_m).copied().unwrap_or(-1);
                    
                    let allele = if target_m_idx >= 0 {
                        let target_m = target_m_idx as usize;
                        let raw_allele = target_gt.allele(
                            crate::data::marker::MarkerIdx::new(target_m as u32),
                            hap_idx,
                        );
                        map_allele(target_m, raw_allele)
                    } else {
                        // Reference marker not in target - use missing value
                        255
                    };
                    
                    allele_buffer[offset] = allele;
                }

                // Append the haplotype to this step
                step.append_haplotype(&allele_buffer);
            }
        }

        // Update total haplotype count
        self.n_haps += n_target_haps;
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

/// PBWT operations on coded steps
///
/// Allows efficient PBWT updates using pattern indices instead of raw alleles
pub struct CodedPbwt {
    /// Prefix array (sorted by pattern)
    prefix: Vec<u32>,
    /// Divergence array
    divergence: Vec<i32>,
    /// Current coded step
    current_step: usize,
}

impl CodedPbwt {
    /// Create new PBWT for coded reference panel
    pub fn new(n_haps: usize) -> Self {
        Self {
            prefix: (0..n_haps as u32).collect(),
            divergence: vec![0; n_haps + 1],
            current_step: 0,
        }
    }

    /// Update PBWT with a coded step
    pub fn update(&mut self, step: &CodedStep) {
        let n_haps = self.prefix.len();
        let n_patterns = step.n_patterns();

        // Bucket sort by pattern
        let mut buckets: Vec<Vec<(u32, i32)>> = vec![Vec::new(); n_patterns.max(1)];

        for i in 0..n_haps {
            let hap = self.prefix[i];
            let div = self.divergence[i];
            let pattern = step.pattern(HapIdx::new(hap)) as usize;
            if pattern < buckets.len() {
                buckets[pattern].push((hap, div));
            }
        }

        // Rebuild arrays
        let mut idx = 0;
        let step_start = self.current_step as i32;

        for bucket in &buckets {
            let bucket_start = idx;
            for &(hap, div) in bucket {
                self.prefix[idx] = hap;
                // Divergence is max of previous div and bucket start
                self.divergence[idx] = if idx == bucket_start {
                    step_start.max(div)
                } else {
                    div
                };
                idx += 1;
            }
        }

        self.current_step += 1;
    }

    /// Find IBS haplotypes for a target pattern at current step
    pub fn find_ibs(&self, target_pattern: u16, step: &CodedStep, n_matches: usize) -> Vec<HapIdx> {
        // Find position of a haplotype with target pattern
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

        // Expand from target position
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

    /// Current step index
    pub fn current_step(&self) -> usize {
        self.current_step
    }
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
    /// Create PBWT view from workspace buffers
    pub fn new(prefix: &'a mut [u32], divergence: &'a mut [i32]) -> Self {
        // Initialize prefix to identity permutation
        for (i, p) in prefix.iter_mut().enumerate() {
            *p = i as u32;
        }
        divergence.fill(0);

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

    /// Update PBWT with a coded step (legacy bucket sort version)
    ///
    /// Kept for compatibility. Use `update_counting_sort` for better performance.
    pub fn update(&mut self, step: &CodedStep) {
        let n_haps = self.prefix.len();
        let n_patterns = step.n_patterns();

        // Allocate buckets based on pattern count
        let mut buckets: Vec<Vec<(u32, i32)>> = vec![Vec::new(); n_patterns.max(1)];

        // Bucket sort by pattern
        for i in 0..n_haps {
            let hap = self.prefix[i];
            let div = self.divergence[i];
            let pattern = step.pattern(HapIdx::new(hap)) as usize;
            if pattern < buckets.len() {
                buckets[pattern].push((hap, div));
            }
        }

        // Rebuild arrays
        let mut idx = 0;
        let step_start = 0i32;

        for bucket in &buckets {
            let bucket_start = idx;
            for &(hap, div) in bucket {
                self.prefix[idx] = hap;
                // Divergence is max of previous div and bucket start
                self.divergence[idx] = if idx == bucket_start {
                    step_start.max(div)
                } else {
                    div
                };
                idx += 1;
            }
        }
    }

    /// Find IBS haplotypes for a target pattern at current step
    pub fn find_ibs(&self, target_pattern: u16, step: &CodedStep, n_matches: usize) -> Vec<HapIdx> {
        // Find position of a haplotype with target pattern
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

        // Expand from target position
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
        assert_eq!(step.n_haps(), 6);

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
        assert_eq!(coded.n_haps(), 6);

        // Check allele access
        assert_eq!(coded.allele(0, HapIdx::new(0)), 0);
        assert_eq!(coded.allele(0, HapIdx::new(1)), 1);
        assert_eq!(coded.allele(3, HapIdx::new(2)), 1);
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

    #[test]
    fn test_coded_pbwt() {
        let gt = make_test_matrix();
        let step = CodedStep::new(&gt, 0, 3);
        let mut pbwt = CodedPbwt::new(6);

        pbwt.update(&step);
        assert_eq!(pbwt.current_step(), 1);

        // Find IBS haplotypes for pattern 0
        let p0 = step.pattern(HapIdx::new(0));
        let ibs = pbwt.find_ibs(p0, &step, 3);
        assert!(!ibs.is_empty());
    }
}
