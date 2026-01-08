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

    /// Create a coded step from a PROJECTED subset of markers
    ///
    /// Unlike `new()` which assumes contiguous marker range [start, end),
    /// this version builds patterns from non-contiguous markers specified
    /// by indices into a projected marker list.
    ///
    /// This is the key fix for DR2 regression: by building patterns on only
    /// the markers where the target has data, we ensure exact matching.
    ///
    /// # Arguments
    /// * `gt` - Full genotype matrix (dense)
    /// * `projected_markers` - Indices of genotyped markers in dense space
    /// * `projected_start` - Start index in projected_markers (inclusive)
    /// * `projected_end` - End index in projected_markers (exclusive)
    pub fn new_projected<S: PhaseState>(
        gt: &GenotypeMatrix<S>,
        projected_markers: &[usize],
        projected_start: usize,
        projected_end: usize,
    ) -> Self {
        let n_haps = gt.n_haplotypes();
        let step_markers = &projected_markers[projected_start..projected_end];
        let n_markers = step_markers.len();

        if n_markers == 0 {
            return Self {
                start: projected_start,
                end: projected_end,
                n_patterns: 0,
                hap_to_pattern: vec![0; n_haps],
                pattern_rep_hap: Vec::new(),
            };
        }

        use std::collections::HashMap;
        let mut pattern_map: HashMap<Vec<u8>, (u16, u32)> = HashMap::new();
        let mut hap_to_pattern = Vec::with_capacity(n_haps);
        let mut pattern_rep_hap: Vec<u32> = Vec::new();
        let mut scratch = vec![0u8; n_markers];

        for h in 0..n_haps {
            let hap = HapIdx::new(h as u32);

            // Fill scratch from NON-CONTIGUOUS markers (the key difference)
            for (i, &dense_m) in step_markers.iter().enumerate() {
                scratch[i] = gt.allele(MarkerIdx::new(dense_m as u32), hap);
            }

            let pattern_idx = if let Some(&(idx, _)) = pattern_map.get(&scratch) {
                idx
            } else {
                let idx = pattern_rep_hap.len() as u16;
                pattern_rep_hap.push(h as u32);
                pattern_map.insert(scratch.clone(), (idx, h as u32));
                idx
            };

            hap_to_pattern.push(pattern_idx);
        }

        Self {
            start: projected_start,
            end: projected_end,
            n_patterns: pattern_rep_hap.len(),
            hap_to_pattern,
            pattern_rep_hap,
        }
    }

    /// Match sequence in PROJECTED space with 255 wildcard handling
    ///
    /// Even in projected space, individual samples may have missing data (255)
    /// at markers that are "genotyped" because other samples have data there.
    /// We treat 255 as a wildcard that matches any reference allele.
    ///
    /// # Arguments
    /// * `alleles` - Target alleles for this step (length = end - start)
    /// * `projected_markers` - Dense indices for this step's markers
    /// * `get_dense_allele` - Closure: (dense_marker_idx, hap) -> allele
    pub fn match_sequence_projected<F>(
        &self,
        alleles: &[u8],
        projected_markers: &[usize],
        get_dense_allele: F,
    ) -> Option<u16>
    where
        F: Fn(usize, u32) -> u8,
    {
        let step_markers = &projected_markers[self.start..self.end];
        if alleles.len() != step_markers.len() {
            return None;
        }

        for (idx, &rep_hap) in self.pattern_rep_hap.iter().enumerate() {
            let matches = step_markers.iter().enumerate().all(|(i, &dense_m)| {
                let ref_allele = get_dense_allele(dense_m, rep_hap);
                let target_allele = alleles[i];
                // Handle 255 (missing) as wildcard - matches anything
                target_allele == ref_allele || target_allele == 255 || ref_allele == 255
            });
            if matches {
                return Some(idx as u16);
            }
        }
        None
    }

    /// Find closest pattern in PROJECTED space (by Hamming distance, 255 = wildcard)
    ///
    /// Fallback when exact match fails due to missing data.
    /// Uses same logic as `closest_pattern_with` but in projected space.
    ///
    /// # Arguments
    /// * `alleles` - Target alleles for this step
    /// * `projected_markers` - Dense indices for this step's markers
    /// * `get_dense_allele` - Closure: (dense_marker_idx, hap) -> allele
    pub fn closest_pattern_projected<F>(
        &self,
        alleles: &[u8],
        projected_markers: &[usize],
        get_dense_allele: F,
    ) -> u16
    where
        F: Fn(usize, u32) -> u8,
    {
        let step_markers = &projected_markers[self.start..self.end];
        if alleles.len() != step_markers.len() || self.n_patterns == 0 {
            return 0;
        }

        let mut best_pattern = 0u16;
        let mut best_distance = usize::MAX;

        for (idx, &rep_hap) in self.pattern_rep_hap.iter().enumerate() {
            let mut distance = 0usize;
            let mut early_exit = false;

            for (i, &dense_m) in step_markers.iter().enumerate() {
                let ref_allele = get_dense_allele(dense_m, rep_hap);
                let target_allele = alleles[i];

                // Skip missing data comparisons (255 = wildcard)
                if ref_allele != target_allele && ref_allele != 255 && target_allele != 255 {
                    distance += 1;
                    if distance >= best_distance {
                        early_exit = true;
                        break;
                    }
                }
            }

            if !early_exit && distance < best_distance {
                best_distance = distance;
                best_pattern = idx as u16;

                if distance == 0 {
                    return best_pattern;
                }
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
}

impl RefPanelCoded {
    /// Create coded reference panel in PROJECTED space (genotyped markers only)
    ///
    /// This is the key fix for DR2 regression. By building patterns on only
    /// the markers where the target has data, we ensure:
    /// 1. Target patterns are EXACT (no 255s, no approximation)
    /// 2. Virtual insertion is EXACT (no closest_pattern fallback)
    /// 3. Reference haps matching at observed positions cluster together
    ///
    /// # Arguments
    /// * `gt` - Full dense genotype matrix
    /// * `projected_markers` - Indices of genotyped markers in dense space
    /// * `projected_gen_positions` - Genetic positions at projected markers only
    /// * `step_cm` - Step size in centiMorgans
    pub fn from_projected_markers<S: PhaseState>(
        gt: &GenotypeMatrix<S>,
        projected_markers: &[usize],
        projected_gen_positions: &[f64],
        step_cm: f64,
    ) -> Self {
        let step_starts = compute_step_starts(projected_gen_positions, step_cm);
        let n_projected = projected_markers.len();

        let mut steps = Vec::with_capacity(step_starts.len());

        for (i, &start) in step_starts.iter().enumerate() {
            let end = step_starts.get(i + 1).copied().unwrap_or(n_projected);
            steps.push(CodedStep::new_projected(gt, projected_markers, start, end));
        }

        Self { steps }
    }

    /// Number of steps
    pub fn n_steps(&self) -> usize {
        self.steps.len()
    }

    /// Get a step
    pub fn step(&self, idx: usize) -> &CodedStep {
        &self.steps[idx]
    }

    /// Average compression ratio
    pub fn avg_compression_ratio(&self) -> f32 {
        if self.steps.is_empty() {
            return 1.0;
        }
        let sum: f32 = self.steps.iter().map(|s| s.compression_ratio()).sum();
        sum / self.steps.len() as f32
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
    ///
    /// # Virtual Insertion (Optional)
    /// If `virtual_pos` is provided as `Some((pos, target_pattern))`, this method
    /// computes where a target with `target_pattern` would be placed in the new
    /// sort order using LF-mapping: `new_pos = Offset[c] + Rank(Prefix[0..pos], c)`.
    ///
    /// Important: The rank is computed on the PRE-SORT prefix array, before mutation.
    pub fn update_counting_sort(
        &mut self,
        step: &CodedStep,
        counts: &mut Vec<usize>,
        offsets: &mut Vec<usize>,
        prefix_scratch: &mut [u32],
        div_scratch: &mut [i32],
        virtual_pos: Option<(&mut usize, u16)>,
    ) {
        let n_haps = self.prefix.len();
        let n_patterns = step.n_patterns();

        if n_patterns == 0 || n_haps == 0 {
            return;
        }

        // Step 1: Count frequency of each pattern AND compute rank for virtual insertion
        // Fused loop: count patterns while tracking rank up to virtual_pos
        // This avoids a second O(N) scan for the rank calculation.
        counts.clear();
        counts.resize(n_patterns, 0);

        // Extract virtual position info for fused computation
        let (has_vpos, target_pattern, current_vpos) = match &virtual_pos {
            Some((vpos, pattern)) => (true, *pattern as usize, (**vpos).min(n_haps)),
            None => (false, 0, 0),
        };
        let mut rank_at_vpos = 0usize;

        for i in 0..n_haps {
            let hap = self.prefix[i];
            let pattern = step.pattern(HapIdx::new(hap)) as usize;
            if pattern < n_patterns {
                // Track rank for virtual insertion (fused with counting)
                // Rank = count of target_pattern haps in prefix[0..current_vpos]
                if has_vpos && pattern == target_pattern && i < current_vpos {
                    rank_at_vpos += 1;
                }
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

        // Step 2.5: Apply LF-mapping for virtual insertion using pre-computed rank
        // u_{k+1} = Offset[c] + Rank(Prefix_k[0..u_k], c)
        if let Some((vpos, _)) = virtual_pos {
            if target_pattern < offsets.len() {
                *vpos = offsets[target_pattern] + rank_at_vpos;
            }
        }

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
    ///
    /// # Virtual Insertion (Optional)
    /// If `virtual_pos` is provided, computes the target's new position using LF-mapping
    /// on the PRE-SORT prefix array, before mutation.
    ///
    /// # Offsets (Optional)
    /// If `offsets` is provided, stores the bucket offsets after sorting.
    /// offsets[i] = start position of pattern i in the sorted array.
    pub fn update_backward(
        &mut self,
        step: &CodedStep,
        n_steps: usize,
        virtual_pos: Option<(&mut usize, u16)>,
        offsets: Option<&mut Vec<usize>>,
    ) {
        let n_haps = self.prefix.len();
        let n_patterns = step.n_patterns();

        let mut buckets: Vec<Vec<(u32, i32)>> = vec![Vec::new(); n_patterns.max(1)];

        // Extract virtual position info for fused rank computation
        let (vpos_ref, target_pattern, current_vpos) = match &virtual_pos {
            Some((vpos, pattern)) => (true, *pattern as usize, (**vpos).min(n_haps)),
            None => (false, 0, 0),
        };
        let mut rank_at_vpos = 0usize;

        // Build buckets AND compute rank in single pass (fused loop)
        for i in 0..n_haps {
            let hap = self.prefix[i];
            let div = self.divergence[i];
            let pattern = step.pattern(HapIdx::new(hap)) as usize;
            if pattern < buckets.len() {
                // Track rank for virtual insertion (fused with bucket building)
                if vpos_ref && pattern == target_pattern && i < current_vpos {
                    rank_at_vpos += 1;
                }
                buckets[pattern].push((hap, div));
            }
        }

        // Calculate new virtual position using pre-computed rank (LF-mapping)
        if let Some((vpos, _)) = virtual_pos {
            if target_pattern < buckets.len() {
                // Compute offset for target pattern (sum of bucket sizes before it)
                let mut pattern_offset = 0;
                for bucket in buckets.iter().take(target_pattern) {
                    pattern_offset += bucket.len();
                }
                *vpos = pattern_offset + rank_at_vpos;
            }
        }

        // Compute and store offsets if requested
        if let Some(offs) = offsets {
            offs.clear();
            offs.reserve(n_patterns + 1);
            let mut running = 0;
            for bucket in &buckets {
                offs.push(running);
                running += bucket.len();
            }
            offs.push(running); // Final offset = n_haps
        }

        // Now scatter to prefix
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

    /// Select neighbors around a virtual position, constrained to a pattern bucket
    ///
    /// This is the core of the "Virtual Insertion" algorithm from Naseri et al.
    /// The target's virtual position represents where it would be inserted in
    /// the PBWT sort order, preserving its history. Neighbors adjacent to this
    /// position share the longest common history with the target.
    ///
    /// IMPORTANT: Neighbors are constrained to the bucket [bucket_start, bucket_end)
    /// to ensure they have the SAME allele pattern as the target. This matches
    /// Java's ImpIbs behavior which only returns haplotypes from the same partition.
    ///
    /// # Arguments
    /// * `virtual_pos` - The target's position in the sort order
    /// * `n_matches` - Number of neighbors to return
    /// * `bucket_start` - Start of the target's pattern bucket (inclusive)
    /// * `bucket_end` - End of the target's pattern bucket (exclusive)
    pub fn select_neighbors_in_bucket(
        &self,
        virtual_pos: usize,
        n_matches: usize,
        bucket_start: usize,
        bucket_end: usize,
    ) -> Vec<HapIdx> {
        let n_haps = self.prefix.len();
        if n_haps == 0 || bucket_start >= bucket_end {
            return Vec::new();
        }

        // Clamp virtual_pos to bucket range
        let center = virtual_pos.clamp(bucket_start, bucket_end.saturating_sub(1));

        let mut result = Vec::with_capacity(n_matches);
        let mut left = center;
        let mut right = center + 1;

        // Expand outward from center, staying within bucket
        while result.len() < n_matches && (left > bucket_start || right < bucket_end) {
            if left > bucket_start {
                left -= 1;
                result.push(HapIdx::new(self.prefix[left]));
            }
            if right < bucket_end && result.len() < n_matches {
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
