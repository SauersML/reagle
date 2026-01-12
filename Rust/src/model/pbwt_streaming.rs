//! # Streaming PBWT for O(N) Memory Phasing
//!
//! This module implements a streaming (wavefront) approach to PBWT that maintains
//! only the current state rather than storing the full index for all markers.
//!
//! ## Memory Comparison
//! - Traditional BidirectionalPhaseIbs: O(M × N × 24 bytes) - stores PPA/div/pos for all markers
//! - Streaming PbwtWavefront: O(N × 24 bytes) - stores only current state
//!
//! For a chromosome with 500k markers and 10k haplotypes:
//! - Traditional: 500,000 × 10,000 × 24 = 120 GB
//! - Streaming: 10,000 × 24 = 240 KB
//!
//! ## Algorithm
//! Uses two passes to get bidirectional neighbors:
//! 1. Forward pass (markers 0→M): Advance forward PBWT, query at sampling points
//! 2. Backward pass (markers M→0): Advance backward PBWT, query at sampling points
//!
//! ## Cache Locality
//! The small wavefront state (~240KB for 10k haplotypes) fits entirely in L2 cache,
//! making neighbor queries extremely fast compared to random access into a large index.

use crate::data::haplotype::SampleIdx;
use crate::model::ibs2::Ibs2;
use crate::model::pbwt::PbwtDivUpdater;

/// Streaming PBWT state that maintains only the current marker's arrays.
///
/// This replaces `BidirectionalPhaseIbs` for memory-efficient phasing.
/// Instead of storing arrays for all markers, we maintain a "wavefront"
/// that advances through the chromosome.
pub struct PbwtWavefront {
    /// Current forward prefix array: ppa[i] = haplotype at sorted position i
    fwd_ppa: Vec<u32>,
    /// Current forward divergence: div[i] = marker where match with predecessor started
    fwd_div: Vec<i32>,
    /// Current backward prefix array
    bwd_ppa: Vec<u32>,
    /// Current backward divergence: div[i] = marker where match with predecessor ends
    bwd_div: Vec<i32>,
    /// Forward inverse index: fwd_inverse[hap] = position of hap in fwd_ppa
    /// Computed lazily only at sampling points
    fwd_inverse: Vec<u32>,
    /// Backward inverse index
    bwd_inverse: Vec<u32>,
    /// Reusable PBWT updater
    updater: PbwtDivUpdater,
    /// Number of haplotypes
    n_haps: usize,
    /// Number of markers in the dataset
    n_markers: usize,
    /// Current marker index for forward pass
    fwd_marker: usize,
    /// Current marker index for backward pass
    bwd_marker: usize,
    /// Whether forward inverse index is valid for current marker
    fwd_inverse_valid: bool,
    /// Whether backward inverse index is valid for current marker
    bwd_inverse_valid: bool,
}

impl PbwtWavefront {
    /// Create a new streaming PBWT wavefront
    pub fn new(n_haps: usize, n_markers: usize) -> Self {
        Self {
            fwd_ppa: (0..n_haps as u32).collect(),
            fwd_div: vec![0; n_haps],
            bwd_ppa: (0..n_haps as u32).collect(),
            bwd_div: vec![n_markers as i32; n_haps],
            fwd_inverse: vec![0; n_haps],
            bwd_inverse: vec![0; n_haps],
            updater: PbwtDivUpdater::new(n_haps),
            n_haps,
            n_markers,
            fwd_marker: 0,
            bwd_marker: n_markers,
            fwd_inverse_valid: false,
            bwd_inverse_valid: false,
        }
    }

    /// Reset for a new forward pass (starts at marker 0)
    pub fn reset_forward(&mut self) {
        for i in 0..self.n_haps {
            self.fwd_ppa[i] = i as u32;
            self.fwd_div[i] = 0;
        }
        self.fwd_marker = 0;
        self.fwd_inverse_valid = false;
    }

    /// Reset for a new backward pass (starts at marker n_markers-1)
    pub fn reset_backward(&mut self) {
        for i in 0..self.n_haps {
            self.bwd_ppa[i] = i as u32;
            self.bwd_div[i] = self.n_markers as i32;
        }
        self.bwd_marker = self.n_markers;
        self.bwd_inverse_valid = false;
    }

    /// Advance forward PBWT by one marker
    ///
    /// # Arguments
    /// * `alleles` - Allele for each haplotype at this marker
    /// * `n_alleles` - Number of distinct alleles (usually 2 for biallelic)
    pub fn advance_forward(&mut self, alleles: &[u8], n_alleles: usize) {
        self.updater.fwd_update(
            alleles,
            n_alleles,
            self.fwd_marker,
            &mut self.fwd_ppa,
            &mut self.fwd_div,
        );
        self.fwd_marker += 1;
        self.fwd_inverse_valid = false;
    }

    /// Advance backward PBWT by one marker (going in reverse)
    ///
    /// # Arguments
    /// * `alleles` - Allele for each haplotype at this marker
    /// * `n_alleles` - Number of distinct alleles
    pub fn advance_backward(&mut self, alleles: &[u8], n_alleles: usize) {
        self.bwd_marker -= 1;
        self.updater.bwd_update(
            alleles,
            n_alleles,
            self.bwd_marker,
            &mut self.bwd_ppa,
            &mut self.bwd_div,
        );
        self.bwd_inverse_valid = false;
    }

    /// Compute forward inverse index (hap -> position) if not already valid
    fn ensure_fwd_inverse(&mut self) {
        if !self.fwd_inverse_valid {
            for (pos, &hap) in self.fwd_ppa.iter().enumerate() {
                self.fwd_inverse[hap as usize] = pos as u32;
            }
            self.fwd_inverse_valid = true;
        }
    }

    /// Compute backward inverse index if not already valid
    fn ensure_bwd_inverse(&mut self) {
        if !self.bwd_inverse_valid {
            for (pos, &hap) in self.bwd_ppa.iter().enumerate() {
                self.bwd_inverse[hap as usize] = pos as u32;
            }
            self.bwd_inverse_valid = true;
        }
    }

    /// Pre-compute forward inverse index for read-only parallel queries
    ///
    /// Call this before using `find_fwd_neighbors_readonly` in a parallel context.
    pub fn prepare_fwd_queries(&mut self) {
        self.ensure_fwd_inverse();
    }

    /// Pre-compute backward inverse index for read-only parallel queries
    pub fn prepare_bwd_queries(&mut self) {
        self.ensure_bwd_inverse();
    }

    /// Find forward neighbors for a haplotype (read-only after prepare_fwd_queries)
    ///
    /// # Safety
    /// Must call `prepare_fwd_queries()` first to ensure inverse is computed.
    pub fn find_fwd_neighbors_readonly(&self, hap_idx: u32, n_candidates: usize) -> Vec<u32> {
        if (hap_idx as usize) >= self.n_haps || !self.fwd_inverse_valid {
            return Vec::new();
        }

        let sorted_pos = self.fwd_inverse[hap_idx as usize] as usize;
        let marker_i32 = (self.fwd_marker.saturating_sub(1)) as i32;

        let mut result = Vec::with_capacity(n_candidates);

        let mut u = sorted_pos;
        let mut v = sorted_pos + 1;
        let mut max_div_up = i32::MIN;
        let mut max_div_down = i32::MIN;

        while result.len() < n_candidates {
            let div_up = if u > 0 {
                self.fwd_div.get(u).copied().unwrap_or(i32::MAX)
            } else {
                i32::MAX
            };
            let div_down = if v < self.n_haps {
                self.fwd_div.get(v).copied().unwrap_or(i32::MAX)
            } else {
                i32::MAX
            };

            let up_valid = u > 0 && max_div_up.max(div_up) <= marker_i32;
            let down_valid = v < self.n_haps && max_div_down.max(div_down) <= marker_i32;

            if !up_valid && !down_valid {
                break;
            }

            let go_up = up_valid && (!down_valid || div_up <= div_down);

            if go_up {
                max_div_up = max_div_up.max(div_up);
                u -= 1;
                let h = self.fwd_ppa[u];
                if h != hap_idx {
                    result.push(h);
                }
            } else {
                max_div_down = max_div_down.max(div_down);
                let h = self.fwd_ppa[v];
                if h != hap_idx {
                    result.push(h);
                }
                v += 1;
            }
        }

        while result.len() < n_candidates && u > 0 {
            u -= 1;
            let h = self.fwd_ppa[u];
            if h != hap_idx {
                result.push(h);
            }
        }
        while result.len() < n_candidates && v < self.n_haps {
            let h = self.fwd_ppa[v];
            if h != hap_idx {
                result.push(h);
            }
            v += 1;
        }

        result
    }

    /// Find backward neighbors for a haplotype (read-only after prepare_bwd_queries)
    pub fn find_bwd_neighbors_readonly(&self, hap_idx: u32, n_candidates: usize) -> Vec<u32> {
        if (hap_idx as usize) >= self.n_haps || !self.bwd_inverse_valid {
            return Vec::new();
        }

        let sorted_pos = self.bwd_inverse[hap_idx as usize] as usize;
        let marker_i32 = self.bwd_marker as i32;

        let mut result = Vec::with_capacity(n_candidates);

        let mut u = sorted_pos;
        let mut v = sorted_pos + 1;
        let mut min_div_up = i32::MAX;
        let mut min_div_down = i32::MAX;

        while result.len() < n_candidates {
            let div_up = if u > 0 {
                self.bwd_div.get(u).copied().unwrap_or(0)
            } else {
                0
            };
            let div_down = if v < self.n_haps {
                self.bwd_div.get(v).copied().unwrap_or(0)
            } else {
                0
            };

            let up_valid = u > 0 && min_div_up.min(div_up) >= marker_i32;
            let down_valid = v < self.n_haps && min_div_down.min(div_down) >= marker_i32;

            if !up_valid && !down_valid {
                break;
            }

            let go_up = up_valid && (!down_valid || div_up >= div_down);

            if go_up {
                min_div_up = min_div_up.min(div_up);
                u -= 1;
                let h = self.bwd_ppa[u];
                if h != hap_idx {
                    result.push(h);
                }
            } else {
                min_div_down = min_div_down.min(div_down);
                let h = self.bwd_ppa[v];
                if h != hap_idx {
                    result.push(h);
                }
                v += 1;
            }
        }

        while result.len() < n_candidates && u > 0 {
            u -= 1;
            let h = self.bwd_ppa[u];
            if h != hap_idx {
                result.push(h);
            }
        }
        while result.len() < n_candidates && v < self.n_haps {
            let h = self.bwd_ppa[v];
            if h != hap_idx {
                result.push(h);
            }
            v += 1;
        }

        result
    }

    /// Find forward neighbors for a haplotype at the current marker
    ///
    /// Returns haplotypes that have matching allele prefixes up to current marker.
    pub fn find_fwd_neighbors(&mut self, hap_idx: u32, n_candidates: usize) -> Vec<u32> {
        if (hap_idx as usize) >= self.n_haps {
            return Vec::new();
        }

        self.ensure_fwd_inverse();

        let sorted_pos = self.fwd_inverse[hap_idx as usize] as usize;
        let marker_i32 = (self.fwd_marker - 1) as i32; // Current marker (after advance)

        let mut result = Vec::with_capacity(n_candidates);

        // Dynamic expansion: choose direction with lower divergence (longer match)
        let mut u = sorted_pos;
        let mut v = sorted_pos + 1;
        let mut max_div_up = i32::MIN;
        let mut max_div_down = i32::MIN;

        while result.len() < n_candidates {
            let div_up = if u > 0 {
                self.fwd_div.get(u).copied().unwrap_or(i32::MAX)
            } else {
                i32::MAX
            };
            let div_down = if v < self.n_haps {
                self.fwd_div.get(v).copied().unwrap_or(i32::MAX)
            } else {
                i32::MAX
            };

            let up_valid = u > 0 && max_div_up.max(div_up) <= marker_i32;
            let down_valid = v < self.n_haps && max_div_down.max(div_down) <= marker_i32;

            if !up_valid && !down_valid {
                break;
            }

            let go_up = up_valid && (!down_valid || div_up <= div_down);

            if go_up {
                max_div_up = max_div_up.max(div_up);
                u -= 1;
                let h = self.fwd_ppa[u];
                if h != hap_idx {
                    result.push(h);
                }
            } else {
                max_div_down = max_div_down.max(div_down);
                let h = self.fwd_ppa[v];
                if h != hap_idx {
                    result.push(h);
                }
                v += 1;
            }
        }

        // Fallback: expand without divergence constraints
        while result.len() < n_candidates && u > 0 {
            u -= 1;
            let h = self.fwd_ppa[u];
            if h != hap_idx {
                result.push(h);
            }
        }
        while result.len() < n_candidates && v < self.n_haps {
            let h = self.fwd_ppa[v];
            if h != hap_idx {
                result.push(h);
            }
            v += 1;
        }

        result
    }

    /// Find backward neighbors for a haplotype at the current marker
    ///
    /// Returns haplotypes that have matching allele suffixes from current marker onward.
    pub fn find_bwd_neighbors(&mut self, hap_idx: u32, n_candidates: usize) -> Vec<u32> {
        if (hap_idx as usize) >= self.n_haps {
            return Vec::new();
        }

        self.ensure_bwd_inverse();

        let sorted_pos = self.bwd_inverse[hap_idx as usize] as usize;
        let marker_i32 = self.bwd_marker as i32;

        let mut result = Vec::with_capacity(n_candidates);

        // Dynamic expansion: choose direction with higher divergence (longer match for bwd)
        let mut u = sorted_pos;
        let mut v = sorted_pos + 1;
        let mut min_div_up = i32::MAX;
        let mut min_div_down = i32::MAX;

        while result.len() < n_candidates {
            let div_up = if u > 0 {
                self.bwd_div.get(u).copied().unwrap_or(0)
            } else {
                0
            };
            let div_down = if v < self.n_haps {
                self.bwd_div.get(v).copied().unwrap_or(0)
            } else {
                0
            };

            // For backward: valid if divergence >= marker (match extends past current)
            let up_valid = u > 0 && min_div_up.min(div_up) >= marker_i32;
            let down_valid = v < self.n_haps && min_div_down.min(div_down) >= marker_i32;

            if !up_valid && !down_valid {
                break;
            }

            // For backward: prefer higher divergence (longer match)
            let go_up = up_valid && (!down_valid || div_up >= div_down);

            if go_up {
                min_div_up = min_div_up.min(div_up);
                u -= 1;
                let h = self.bwd_ppa[u];
                if h != hap_idx {
                    result.push(h);
                }
            } else {
                min_div_down = min_div_down.min(div_down);
                let h = self.bwd_ppa[v];
                if h != hap_idx {
                    result.push(h);
                }
                v += 1;
            }
        }

        // Fallback: expand without divergence constraints
        while result.len() < n_candidates && u > 0 {
            u -= 1;
            let h = self.bwd_ppa[u];
            if h != hap_idx {
                result.push(h);
            }
        }
        while result.len() < n_candidates && v < self.n_haps {
            let h = self.bwd_ppa[v];
            if h != hap_idx {
                result.push(h);
            }
            v += 1;
        }

        result
    }

    /// Find bidirectional neighbors by combining forward and backward results
    ///
    /// This combines the neighbor-finding from both PBWT directions,
    /// plus IBS2 segments if provided.
    pub fn find_neighbors(
        &mut self,
        hap_idx: u32,
        ibs2: &Ibs2,
        n_candidates: usize,
        global_marker: usize,
    ) -> Vec<u32> {
        let mut neighbors = Vec::with_capacity(n_candidates * 2 + 10);
        let sample = SampleIdx::new(hap_idx / 2);

        // Add IBS2 neighbors first (highest priority)
        for seg in ibs2.segments(sample) {
            if seg.contains(global_marker) {
                let other_s = seg.other_sample;
                if other_s != sample {
                    neighbors.push(other_s.hap1().0);
                    neighbors.push(other_s.hap2().0);
                }
            }
        }

        // Add forward PBWT neighbors
        let fwd_neighbors = self.find_fwd_neighbors(hap_idx, n_candidates);
        for h in fwd_neighbors {
            if h != hap_idx && h / 2 != sample.0 {
                neighbors.push(h);
            }
        }

        // Add backward PBWT neighbors
        let bwd_neighbors = self.find_bwd_neighbors(hap_idx, n_candidates);
        for h in bwd_neighbors {
            if h != hap_idx && h / 2 != sample.0 {
                neighbors.push(h);
            }
        }

        neighbors
    }

    /// Get current forward marker index
    pub fn fwd_marker(&self) -> usize {
        self.fwd_marker
    }

    /// Get current backward marker index
    pub fn bwd_marker(&self) -> usize {
        self.bwd_marker
    }

    /// Get number of haplotypes
    pub fn n_haps(&self) -> usize {
        self.n_haps
    }
}

/// Builder for streaming PBWT-based neighbor collection.
///
/// This implements the two-pass streaming algorithm:
/// 1. Forward pass: collect forward neighbors at sampling points
/// 2. Backward pass: collect backward neighbors at sampling points
/// 3. Finalize: combine neighbors into PhaseStates
pub struct StreamingNeighborCollector {
    /// Forward neighbors collected at each sampling point
    /// fwd_neighbors[sample_idx][sampling_point] = Vec<neighbor_haps>
    fwd_neighbors: Vec<Vec<Vec<u32>>>,
    /// Backward neighbors collected at each sampling point
    bwd_neighbors: Vec<Vec<Vec<u32>>>,
    /// Sampling point marker indices
    sampling_points: Vec<usize>,
    /// Number of samples
    n_samples: usize,
    /// Number of candidates per query
    n_candidates: usize,
}

impl StreamingNeighborCollector {
    /// Create a new streaming neighbor collector
    ///
    /// # Arguments
    /// * `n_samples` - Number of target samples
    /// * `n_markers` - Total number of markers
    /// * `n_candidates` - Number of candidates to collect at each sampling point
    pub fn new(n_samples: usize, n_markers: usize, n_candidates: usize) -> Self {
        // Sampling at sparse points (every 64 markers, matching PhaseStates behavior)
        const MAX_SAMPLE_POINTS: usize = 64;
        let step = (n_markers / MAX_SAMPLE_POINTS).max(1);
        let sampling_points: Vec<usize> = (0..n_markers).step_by(step).collect();
        let n_points = sampling_points.len();

        Self {
            fwd_neighbors: vec![vec![Vec::new(); n_points]; n_samples],
            bwd_neighbors: vec![vec![Vec::new(); n_points]; n_samples],
            sampling_points,
            n_samples,
            n_candidates,
        }
    }

    /// Get sampling points
    pub fn sampling_points(&self) -> &[usize] {
        &self.sampling_points
    }

    /// Record forward neighbors for a sample at a sampling point index
    pub fn add_fwd_neighbors(&mut self, sample: usize, point_idx: usize, neighbors: Vec<u32>) {
        if sample < self.n_samples && point_idx < self.sampling_points.len() {
            self.fwd_neighbors[sample][point_idx] = neighbors;
        }
    }

    /// Record backward neighbors for a sample at a sampling point index
    pub fn add_bwd_neighbors(&mut self, sample: usize, point_idx: usize, neighbors: Vec<u32>) {
        if sample < self.n_samples && point_idx < self.sampling_points.len() {
            self.bwd_neighbors[sample][point_idx] = neighbors;
        }
    }

    /// Get all collected neighbors for a sample (combines forward and backward)
    ///
    /// Returns (marker_indices, neighbors_per_marker)
    pub fn get_all_neighbors(&self, sample: usize) -> Vec<(usize, Vec<u32>)> {
        let mut result = Vec::with_capacity(self.sampling_points.len());

        for (point_idx, &marker) in self.sampling_points.iter().enumerate() {
            let mut combined = Vec::new();

            // Add forward neighbors
            if sample < self.fwd_neighbors.len() && point_idx < self.fwd_neighbors[sample].len() {
                combined.extend_from_slice(&self.fwd_neighbors[sample][point_idx]);
            }

            // Add backward neighbors
            if sample < self.bwd_neighbors.len() && point_idx < self.bwd_neighbors[sample].len() {
                combined.extend_from_slice(&self.bwd_neighbors[sample][point_idx]);
            }

            result.push((marker, combined));
        }

        result
    }

    /// Number of candidates per query
    pub fn n_candidates(&self) -> usize {
        self.n_candidates
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wavefront_basic() {
        let n_haps = 100;
        let n_markers = 1000;
        let mut wavefront = PbwtWavefront::new(n_haps, n_markers);

        // Forward pass with simple biallelic data
        wavefront.reset_forward();
        for m in 0..100 {
            let alleles: Vec<u8> = (0..n_haps).map(|h| ((h + m) % 2) as u8).collect();
            wavefront.advance_forward(&alleles, 2);
        }

        // Find neighbors for haplotype 0
        let neighbors = wavefront.find_fwd_neighbors(0, 10);
        assert!(!neighbors.is_empty());
        assert!(neighbors.iter().all(|&h| h != 0));
    }

    #[test]
    fn test_bidirectional() {
        let n_haps = 50;
        let n_markers = 500;
        let mut wavefront = PbwtWavefront::new(n_haps, n_markers);

        // Generate consistent allele data
        let alleles: Vec<Vec<u8>> = (0..n_markers)
            .map(|m| (0..n_haps).map(|h| ((h * 7 + m * 13) % 2) as u8).collect())
            .collect();

        // Forward pass
        wavefront.reset_forward();
        for m in 0..n_markers {
            wavefront.advance_forward(&alleles[m], 2);
        }
        let fwd_neighbors = wavefront.find_fwd_neighbors(5, 10);

        // Backward pass
        wavefront.reset_backward();
        for m in (0..n_markers).rev() {
            wavefront.advance_backward(&alleles[m], 2);
        }
        let bwd_neighbors = wavefront.find_bwd_neighbors(5, 10);

        // Both should return valid neighbors
        assert!(!fwd_neighbors.is_empty());
        assert!(!bwd_neighbors.is_empty());
    }
}
