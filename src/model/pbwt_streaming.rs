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
    #[inline]
    fn finalize_candidates(mut out: Vec<u32>, target: u32, n_candidates: usize) -> Vec<u32> {
        if out.is_empty() {
            return out;
        }
        out.sort_unstable();
        out.dedup();
        if let Ok(pos) = out.binary_search(&target) {
            out.remove(pos);
        }
        if out.len() > n_candidates {
            out.truncate(n_candidates);
        }
        out
    }

    // Geometric spiral selection: dense core with exponentially expanding offsets.
    fn select_spiral_window(
        ppa: &[u32],
        start: usize,
        end: usize,
        sorted_pos: usize,
        target: u32,
        n_candidates: usize,
    ) -> Vec<u32> {
        if n_candidates == 0 || start >= end {
            return Vec::new();
        }
        let block_len = end - start;
        if block_len <= 1 {
            return Vec::new();
        }

        let mut out = Vec::with_capacity(n_candidates);
        let max_left = sorted_pos.saturating_sub(start);
        let max_right = end.saturating_sub(sorted_pos + 1);
        let max_offset = max_left.max(max_right);

        let push_offset = |offset: usize, out: &mut Vec<u32>| {
            if offset == 0 || out.len() >= n_candidates {
                return;
            }
            if offset <= max_left {
                out.push(ppa[sorted_pos - offset]);
            }
            if out.len() >= n_candidates {
                return;
            }
            if offset <= max_right {
                out.push(ppa[sorted_pos + offset]);
            }
        };

        // Fibonacci offsets: 1, 2, 3, 5, 8, ...
        let mut prev = 1usize;
        let mut curr = 2usize;
        if max_offset >= 1 {
            push_offset(prev, &mut out);
        }
        if out.len() < n_candidates && max_offset >= 2 {
            push_offset(curr, &mut out);
        }
        while out.len() < n_candidates {
            let next = prev.saturating_add(curr);
            if next > max_offset {
                break;
            }
            push_offset(next, &mut out);
            prev = curr;
            curr = next;
        }

        // If still short, fill by scanning outward within the bounded window.
        if out.len() < n_candidates {
            let mut u = sorted_pos;
            let mut v = sorted_pos + 1;
            while out.len() < n_candidates && (u > start || v < end) {
                if u > start {
                    u -= 1;
                    out.push(ppa[u]);
                }
                if out.len() < n_candidates && v < end {
                    out.push(ppa[v]);
                    v += 1;
                }
            }
        }

        Self::finalize_candidates(out, target, n_candidates)
    }

    fn fwd_window_bounded(
        &self,
        sorted_pos: usize,
        marker_i32: i32,
        max_span: usize,
    ) -> (usize, usize) {
        let mut start = sorted_pos;
        let mut max_div = i32::MIN;
        let mut u = sorted_pos;
        let mut steps = 0usize;
        while u > 0 && steps < max_span {
            let div = self.fwd_div.get(u).copied().unwrap_or(i32::MAX);
            max_div = max_div.max(div);
            if max_div > marker_i32 {
                break;
            }
            u -= 1;
            start = u;
            steps += 1;
        }

        let mut end = sorted_pos + 1;
        let mut max_div_down = i32::MIN;
        let mut v = sorted_pos + 1;
        steps = 0;
        while v < self.n_haps && steps < max_span {
            let div = self.fwd_div.get(v).copied().unwrap_or(i32::MAX);
            max_div_down = max_div_down.max(div);
            if max_div_down > marker_i32 {
                break;
            }
            v += 1;
            end = v;
            steps += 1;
        }
        (start, end)
    }

    fn bwd_window_bounded(
        &self,
        sorted_pos: usize,
        marker_i32: i32,
        max_span: usize,
    ) -> (usize, usize) {
        let mut start = sorted_pos;
        let mut min_div = i32::MAX;
        let mut u = sorted_pos;
        let mut steps = 0usize;
        while u > 0 && steps < max_span {
            let div = self.bwd_div.get(u).copied().unwrap_or(0);
            min_div = min_div.min(div);
            if min_div < marker_i32 {
                break;
            }
            u -= 1;
            start = u;
            steps += 1;
        }

        let mut end = sorted_pos + 1;
        let mut min_div_down = i32::MAX;
        let mut v = sorted_pos + 1;
        steps = 0;
        while v < self.n_haps && steps < max_span {
            let div = self.bwd_div.get(v).copied().unwrap_or(0);
            min_div_down = min_div_down.min(div);
            if min_div_down < marker_i32 {
                break;
            }
            v += 1;
            end = v;
            steps += 1;
        }
        (start, end)
    }
    /// Create a new streaming PBWT wavefront
    pub fn new(n_haps: usize, n_markers: usize) -> Self {
        Self::with_state(n_haps, n_markers, None)
    }

    pub fn n_haps(&self) -> usize {
        self.n_haps
    }

    pub fn n_markers(&self) -> usize {
        self.n_markers
    }

    /// Create a new streaming PBWT wavefront with optional initial state
    pub fn with_state(
        n_haps: usize,
        n_markers: usize,
        initial_state: Option<&crate::model::pbwt::PbwtState>
    ) -> Self {
        let (fwd_ppa, fwd_div) = if let Some(state) = initial_state {
            assert_eq!(state.ppa.len(), n_haps, "Initial state hap count mismatch");
            (state.ppa.clone(), state.div.clone())
        } else {
            ((0..n_haps as u32).collect(), vec![0; n_haps])
        };

        Self {
            fwd_ppa,
            fwd_div,
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

    /// Get the current PBWT state (PPA and divergence) for handoff
    pub fn get_state(&self) -> crate::model::pbwt::PbwtState {
        crate::model::pbwt::PbwtState::new(
            self.fwd_ppa.clone(),
            self.fwd_div.clone(),
            self.fwd_marker
        )
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

    /// Estimate forward match length with a neighbor carrying the requested allele.
    pub fn fwd_match_len_with_allele(&mut self, hap_idx: u32, allele: u8, alleles: &[u8]) -> i32 {
        if (hap_idx as usize) >= self.n_haps {
            return 0;
        }
        self.ensure_fwd_inverse();
        let pos = self.fwd_inverse[hap_idx as usize] as usize;
        let m = self.fwd_marker as i32;
        let mut best = 0i32;

        if pos > 0 {
            let nb = self.fwd_ppa[pos - 1] as usize;
            if alleles.get(nb).copied().unwrap_or(255) == allele {
                let len = m - self.fwd_div[pos];
                if len > best {
                    best = len;
                }
            }
        }

        if pos + 1 < self.n_haps {
            let nb = self.fwd_ppa[pos + 1] as usize;
            if alleles.get(nb).copied().unwrap_or(255) == allele {
                let len = m - self.fwd_div[pos + 1];
                if len > best {
                    best = len;
                }
            }
        }

        if best < 0 {
            0
        } else {
            best
        }
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
        let max_span = n_candidates.saturating_mul(8).max(32);
        let (start, end) = self.fwd_window_bounded(sorted_pos, marker_i32, max_span);
        Self::select_spiral_window(
            &self.fwd_ppa,
            start,
            end,
            sorted_pos,
            hap_idx,
            n_candidates,
        )
    }

    /// Find backward neighbors for a haplotype (read-only after prepare_bwd_queries)
    pub fn find_bwd_neighbors_readonly(&self, hap_idx: u32, n_candidates: usize) -> Vec<u32> {
        if (hap_idx as usize) >= self.n_haps || !self.bwd_inverse_valid {
            return Vec::new();
        }

        let sorted_pos = self.bwd_inverse[hap_idx as usize] as usize;
        let marker_i32 = self.bwd_marker as i32;
        let max_span = n_candidates.saturating_mul(8).max(32);
        let (start, end) = self.bwd_window_bounded(sorted_pos, marker_i32, max_span);
        Self::select_spiral_window(
            &self.bwd_ppa,
            start,
            end,
            sorted_pos,
            hap_idx,
            n_candidates,
        )
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

        // Find neighbors for haplotype 0 using read-only method
        wavefront.prepare_fwd_queries();
        let neighbors = wavefront.find_fwd_neighbors_readonly(0, 10);
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
        wavefront.prepare_fwd_queries();
        let fwd_neighbors = wavefront.find_fwd_neighbors_readonly(5, 10);

        // Backward pass
        wavefront.reset_backward();
        for m in (0..n_markers).rev() {
            wavefront.advance_backward(&alleles[m], 2);
        }
        wavefront.prepare_bwd_queries();
        let bwd_neighbors = wavefront.find_bwd_neighbors_readonly(5, 10);

        // Both should return valid neighbors
        assert!(!fwd_neighbors.is_empty());
        assert!(!bwd_neighbors.is_empty());
    }
}
