//! # Bidirectional PBWT for Phasing HMM State Selection
//!
//! This module implements bidirectional Positional Burrows-Wheeler Transform (PBWT)
//! based neighbor finding for haplotype phasing. It is used to select HMM states
//! (reference haplotypes) that are likely to match the target haplotype.
//!
//! ## Algorithm Overview
//!
//! The PBWT maintains a sorted order of haplotypes such that those with longer
//! matching prefixes (forward) or suffixes (backward) are adjacent. By storing
//! both directions, we can find haplotypes that match well both upstream and
//! downstream of the current marker.
//!
//! ## Key Concepts
//!
//! - Prefix array (PPA): Permutation of haplotypes sorted by allele history
//! - Divergence array: For each position in PPA, stores where the match started/ended
//! - Forward PBWT: `div[i]` = marker where the match with predecessor started
//! - Backward PBWT: `div[i]` = marker where the match with predecessor ends
//!
//! ## Integration with IBS2
//!
//! IBS2 segments (regions where two samples share both haplotypes) are also
//! included as high-priority neighbors, as they indicate recent common ancestry
//! and strong phase concordance.

use crate::data::haplotype::SampleIdx;
use crate::model::ibs2::Ibs2;
use crate::model::pbwt::PbwtDivUpdater;

/// Manages bidirectional PBWT state for HMM state selection.
///
/// Stores both forward and backward PBWT arrays at each marker to enable
/// selecting haplotypes that match well both upstream and downstream.
/// This is critical for phasing accuracy around recombination hotspots.
///
/// ## Subset Support
///
/// When built for a marker subset (e.g., high-frequency markers in Stage 1),
/// the PBWT operates in subset index space (0..n_subset), but IBS2 segments
/// use global marker indices. The `subset_to_global` mapping handles this
/// coordinate space conversion automatically in `find_neighbors`.
pub struct BidirectionalPhaseIbs {
    /// Forward divergence at each marker: `fwd_div[m]` = divergence array after
    /// processing markers 0..=m. For position i in the sorted order, `div[i]` is
    /// the marker where the match with the haplotype at position i-1 started.
    fwd_div: Vec<Vec<i32>>,
    /// Forward prefix array at each marker: `fwd_ppa[m][i]` = haplotype index at
    /// sorted position i after processing markers 0..=m
    fwd_ppa: Vec<Vec<u32>>,
    /// Backward divergence at each marker: `bwd_div[m]` = divergence array after
    /// processing markers m..n_markers (in reverse). For position i, `div[i]` is
    /// the marker where the match with the haplotype at position i-1 ends.
    bwd_div: Vec<Vec<i32>>,
    /// Backward prefix array at each marker
    bwd_ppa: Vec<Vec<u32>>,
    /// Inverse index for forward PPA: fwd_pos[m][h] = position of haplotype h in fwd_ppa[m]
    /// Enables O(1) position lookup instead of O(n_haps) linear search.
    fwd_pos: Vec<Vec<u32>>,
    /// Inverse index for backward PPA: bwd_pos[m][h] = position of haplotype h in bwd_ppa[m]
    bwd_pos: Vec<Vec<u32>>,
    /// Total number of haplotypes in the PBWT
    n_haps: usize,
    /// Number of markers in the PBWT (may be subset of full chromosome)
    n_markers: usize,
    /// Optional mapping from subset marker index to global marker index.
    /// When Some, IBS2 lookups use the mapped global index since IBS2 segments
    /// are defined in global marker space.
    /// When None (full chromosome), marker indices are used directly.
    subset_to_global: Option<Vec<usize>>,
    /// Stored alleles per marker for O(1) allele lookup.
    /// alleles[m][h] = allele of haplotype h at marker m.
    alleles: Vec<Vec<u8>>,
}

impl BidirectionalPhaseIbs {
    /// Build bidirectional PBWT from genotype data
    ///
    /// # Arguments
    /// * `alleles` - Allele data per marker
    /// * `n_haps` - Number of haplotypes
    /// * `n_markers` - Number of markers
    pub fn build(
        mut alleles: Vec<Vec<u8>>,
        n_haps: usize,
        n_markers: usize,
    ) -> Self {
        let mut fwd_div = Vec::with_capacity(n_markers);
        let mut fwd_ppa = Vec::with_capacity(n_markers);
        let mut fwd_pos = Vec::with_capacity(n_markers);
        let mut bwd_div = vec![Vec::new(); n_markers];
        let mut bwd_ppa = vec![Vec::new(); n_markers];
        let mut bwd_pos = vec![Vec::new(); n_markers];
        let mut n_alleles_by_marker = vec![2usize; n_markers];

        let mut updater = PbwtDivUpdater::new(n_haps);

        let mut ppa: Vec<u32> = (0..n_haps as u32).collect();
        let mut div: Vec<i32> = vec![0; n_haps + 1];

        for m in 0..n_markers {
            let n_alleles = normalize_pbwt_alleles(&mut alleles[m]);
            n_alleles_by_marker[m] = n_alleles;
            updater.fwd_update(&alleles[m], n_alleles, m, &mut ppa, &mut div);

            // Build inverse index: fwd_pos[m][h] = position of haplotype h in fwd_ppa[m]
            let mut pos = vec![0u32; n_haps];
            for (i, &h) in ppa.iter().enumerate() {
                pos[h as usize] = i as u32;
            }
            fwd_pos.push(pos);
            fwd_ppa.push(ppa.clone());
            fwd_div.push(div[..n_haps].to_vec());
        }

        ppa = (0..n_haps as u32).collect();
        div = vec![n_markers as i32; n_haps + 1];

        for m in (0..n_markers).rev() {
            let n_alleles = n_alleles_by_marker[m];
            updater.bwd_update(&alleles[m], n_alleles, m, &mut ppa, &mut div);

            // Build inverse index: bwd_pos[m][h] = position of haplotype h in bwd_ppa[m]
            let mut pos = vec![0u32; n_haps];
            for (i, &h) in ppa.iter().enumerate() {
                pos[h as usize] = i as u32;
            }
            bwd_pos[m] = pos;
            bwd_ppa[m] = ppa.clone();
            bwd_div[m] = div[..n_haps].to_vec();
        }

        Self {
            fwd_div,
            fwd_ppa,
            bwd_div,
            bwd_ppa,
            fwd_pos,
            bwd_pos,
            n_haps,
            n_markers,
            subset_to_global: None,
            alleles,
        }
    }

    /// Build bidirectional PBWT for a marker subset with global index mapping.
    ///
    /// This variant is used when phasing operates on a subset of markers (e.g.,
    /// high-frequency markers in Stage 1 phasing). The subset_to_global mapping
    /// ensures IBS2 segment lookups (which use global indices) work correctly.
    ///
    /// # Arguments
    /// * `alleles` - Allele data per subset marker (2D: [marker][haplotype])
    /// * `n_haps` - Number of haplotypes
    /// * `n_markers` - Number of markers in the subset
    /// * `subset_to_global` - Mapping from subset index to global marker index
    ///
    /// # Example Use Case
    /// Stage 1 phasing uses only high-frequency markers (e.g., MAF > 0.1) to
    /// establish initial phase. The subset indices (0, 1, 2, ...) map to
    /// non-contiguous global indices (e.g., 0, 5, 12, ...).
    pub fn build_for_subset(
        alleles: Vec<Vec<u8>>,
        n_haps: usize,
        n_markers: usize,
        subset_to_global: &[usize],
    ) -> Self {
        let mut result = Self::build(alleles, n_haps, n_markers);
        result.subset_to_global = Some(subset_to_global.to_vec());
        result
    }

    /// Find neighbor haplotypes at a marker using bidirectional PBWT and IBS2.
    ///
    /// This is the main entry point for HMM state selection during phasing.
    /// It combines three sources of potential matching haplotypes:
    ///
    /// 1. **IBS2 segments**: Haplotypes from samples that share both haplotypes
    ///    with the target sample at this marker (highest priority)
    /// 2. **Forward PBWT neighbors**: Haplotypes with matching allele prefixes
    ///    (markers 0..=marker_idx)
    /// 3. **Backward PBWT neighbors**: Haplotypes with matching allele suffixes
    ///    (markers marker_idx..n_markers)
    ///
    /// The combined set excludes the target haplotype and its pair from the
    /// same sample.
    ///
    /// # Arguments
    /// * `hap_idx` - Target haplotype index
    /// * `marker_idx` - Current marker (in subset space if built with subset)
    /// * `ibs2` - IBS2 segment data
    /// * `n_candidates` - Approximate number of neighbors to return
    ///
    /// # Returns
    /// Vector of neighbor haplotype indices (may contain duplicates from
    /// multiple sources, which is intentional for weighting)
    pub fn find_neighbors(
        &self,
        hap_idx: u32,
        marker_idx: usize,
        ibs2: &Ibs2,
        n_candidates: usize,
    ) -> Vec<u32> {
        let mut neighbors = Vec::with_capacity(n_candidates * 2 + 10);
        let sample = SampleIdx::new(hap_idx / 2);

        // Convert marker index to global space for IBS2 lookup
        // IBS2 segments use global marker indices, but when built for a subset,
        // marker_idx is in subset space. The mapping handles this conversion.
        let ibs2_marker_idx = self
            .subset_to_global
            .as_ref()
            .and_then(|mapping| mapping.get(marker_idx).copied())
            .unwrap_or(marker_idx);

        for seg in ibs2.segments(sample) {
            if seg.contains(ibs2_marker_idx) {
                let other_s = seg.other_sample;
                if other_s != sample {
                    neighbors.push(other_s.hap1().0);
                    neighbors.push(other_s.hap2().0);
                }
            }
        }

        let fwd_neighbors = self.find_fwd_neighbors(hap_idx, marker_idx, n_candidates);
        let bwd_neighbors = self.find_bwd_neighbors(hap_idx, marker_idx, n_candidates);

        for h in fwd_neighbors {
            if h != hap_idx && h / 2 != sample.0 {
                neighbors.push(h);
            }
        }

        for h in bwd_neighbors {
            if h != hap_idx && h / 2 != sample.0 {
                neighbors.push(h);
            }
        }

        neighbors
    }

    /// Estimate the best match span (in marker steps) for a haplotype at a marker.
    ///
    /// Uses adjacent PBWT neighbors and divergence arrays to approximate the
    /// longest shared segment around `marker_idx`.
    pub fn best_match_span(&self, hap_idx: u32, marker_idx: usize) -> usize {
        if marker_idx >= self.n_markers || (hap_idx as usize) >= self.n_haps {
            return 0;
        }

        // O(1) position lookup using inverse index
        let pos_fwd = self.fwd_pos[marker_idx][hap_idx as usize] as usize;
        let pos_bwd = self.bwd_pos[marker_idx][hap_idx as usize] as usize;

        let mut best_fwd = 0usize;
        for pos in [pos_fwd.wrapping_sub(1), pos_fwd + 1] {
            if pos < self.n_haps {
                let start = self.fwd_div[marker_idx][pos];
                if marker_idx as i32 >= start {
                    let span = (marker_idx as i32 - start + 1) as usize;
                    if span > best_fwd {
                        best_fwd = span;
                    }
                }
            }
        }

        let mut best_bwd = 0usize;
        for pos in [pos_bwd.wrapping_sub(1), pos_bwd + 1] {
            if pos < self.n_haps {
                let end = self.bwd_div[marker_idx][pos];
                if end >= marker_idx as i32 {
                    let span = (end - marker_idx as i32 + 1) as usize;
                    if span > best_bwd {
                        best_bwd = span;
                    }
                }
            }
        }

        if best_fwd > 0 && best_bwd > 0 {
            best_fwd + best_bwd - 1
        } else {
            best_fwd.max(best_bwd)
        }
    }

    fn find_fwd_neighbors(&self, hap_idx: u32, marker_idx: usize, n_candidates: usize) -> Vec<u32> {
        if marker_idx >= self.n_markers || (hap_idx as usize) >= self.n_haps {
            return Vec::new();
        }

        let ppa = &self.fwd_ppa[marker_idx];
        let div = &self.fwd_div[marker_idx];

        // O(1) position lookup using inverse index
        let sorted_pos = self.fwd_pos[marker_idx][hap_idx as usize] as usize;
        let marker_i32 = marker_idx as i32;

        // For forward PBWT, div[i] = marker where match started (divergence point).
        // A valid match has div[i] <= marker_idx (match still active).
        // Java: continue while d[u] <= step (match started at or before current step)
        // Note: backoff_limit is available for future backoff implementation but not used currently

        let mut result = Vec::with_capacity(n_candidates);

        // Dynamic expansion: choose direction with lower divergence (= longer match = better neighbor)
        // instead of forcing 50/50 split between up/down directions
        let mut u = sorted_pos;
        let mut v = sorted_pos + 1;
        let mut max_div_up = i32::MIN;
        let mut max_div_down = i32::MIN;

        while result.len() < n_candidates {
            // Get next divergence values in each direction
            let div_up = if u > 0 { div.get(u).copied().unwrap_or(i32::MAX) } else { i32::MAX };
            let div_down = if v < self.n_haps { div.get(v).copied().unwrap_or(i32::MAX) } else { i32::MAX };

            // Check if either direction still has valid matches (divergence <= current marker)
            let up_valid = u > 0 && max_div_up.max(div_up) <= marker_i32;
            let down_valid = v < self.n_haps && max_div_down.max(div_down) <= marker_i32;

            if !up_valid && !down_valid {
                break; // No more valid matches in either direction
            }

            // For forward PBWT: lower divergence = longer match = better neighbor
            // Choose direction with lower next divergence value
            let go_up = up_valid && (!down_valid || div_up <= div_down);

            if go_up {
                max_div_up = max_div_up.max(div_up);
                u -= 1;
                let h = ppa[u];
                if h != hap_idx {
                    result.push(h);
                }
            } else {
                max_div_down = max_div_down.max(div_down);
                let h = ppa[v];
                if h != hap_idx {
                    result.push(h);
                }
                v += 1;
            }
        }

        // Fallback: if strict PBWT matching yields too few neighbors,
        // expand outward without divergence constraints to fill the pool.
        while result.len() < n_candidates && u > 0 {
            u -= 1;
            let h = ppa[u];
            if h != hap_idx {
                result.push(h);
            }
        }
        while result.len() < n_candidates && v < self.n_haps {
            let h = ppa[v];
            if h != hap_idx {
                result.push(h);
            }
            v += 1;
        }

        result
    }

    fn find_bwd_neighbors(&self, hap_idx: u32, marker_idx: usize, n_candidates: usize) -> Vec<u32> {
        if marker_idx >= self.n_markers || (hap_idx as usize) >= self.n_haps {
            return Vec::new();
        }

        let ppa = &self.bwd_ppa[marker_idx];
        let div = &self.bwd_div[marker_idx];

        // O(1) position lookup using inverse index
        let sorted_pos = self.bwd_pos[marker_idx][hap_idx as usize] as usize;
        let marker_i32 = marker_idx as i32;

        // For backward PBWT, div[i] = marker where match ENDS (going backward).
        // A valid match has div[i] >= marker_idx (match continues at or past current marker).
        // Java: continue while step <= uNextMatchEnd || step <= vNextMatchEnd
        // i.e., continue while marker_idx <= div[i] (match still active at current marker)
        // Note: backoff_limit is available for future backoff implementation but not used currently

        let mut result = Vec::with_capacity(n_candidates);

        // Dynamic expansion: choose direction with higher divergence (= longer match = better neighbor)
        // For backward PBWT, higher div = match ends later = longer match
        let mut u = sorted_pos;
        let mut v = sorted_pos + 1;
        let mut min_div_up = i32::MAX;
        let mut min_div_down = i32::MAX;

        while result.len() < n_candidates {
            // Get next divergence values in each direction
            let div_up = if u > 0 { div.get(u).copied().unwrap_or(0) } else { 0 };
            let div_down = if v < self.n_haps { div.get(v).copied().unwrap_or(0) } else { 0 };

            // Check if either direction still has valid matches (divergence >= current marker)
            let up_valid = u > 0 && min_div_up.min(div_up) >= marker_i32;
            let down_valid = v < self.n_haps && min_div_down.min(div_down) >= marker_i32;

            if !up_valid && !down_valid {
                break; // No more valid matches in either direction
            }

            // For backward PBWT: higher divergence = longer match = better neighbor
            // Choose direction with higher next divergence value
            let go_up = up_valid && (!down_valid || div_up >= div_down);

            if go_up {
                min_div_up = min_div_up.min(div_up);
                u -= 1;
                let h = ppa[u];
                if h != hap_idx {
                    result.push(h);
                }
            } else {
                min_div_down = min_div_down.min(div_down);
                let h = ppa[v];
                if h != hap_idx {
                    result.push(h);
                }
                v += 1;
            }
        }

        // Fallback: widen search without divergence constraints if needed.
        while result.len() < n_candidates && u > 0 {
            u -= 1;
            let h = ppa[u];
            if h != hap_idx {
                result.push(h);
            }
        }
        while result.len() < n_candidates && v < self.n_haps {
            let h = ppa[v];
            if h != hap_idx {
                result.push(h);
            }
            v += 1;
        }

        result
    }

    /// Get the number of haplotypes
    pub fn n_haps(&self) -> usize {
        self.n_haps
    }

    /// Get the allele of a reference haplotype at a marker.
    ///
    /// This is used during dynamic MCMC to retrieve the reference panel alleles
    /// when computing emissions for the HMM states.
    #[inline]
    pub fn allele(&self, marker: usize, hap: u32) -> u8 {
        self.alleles[marker][hap as usize]
    }

    /// Find neighbors of a reference haplotype state in the PBWT.
    ///
    /// This is the "Latent State" approach: instead of threading the target's alleles
    /// through PBWT (which is O(M*N) and mathematically unsound), we use the HMM's
    /// sampled state. If the HMM decided we're copying from reference haplotype k,
    /// then the neighbors of our target ARE the neighbors of k in the PBWT.
    ///
    /// This gives O(1) position lookup via the precomputed inverse index.
    ///
    /// # Arguments
    /// * `ref_state` - Reference haplotype index from the sampled HMM path
    /// * `marker_idx` - Current marker for neighbor selection
    /// * `sample_idx` - Sample index (for exclusion)
    /// * `n_candidates` - Number of neighbors to return
    pub fn find_neighbors_of_state(
        &self,
        ref_state: u32,
        marker_idx: usize,
        sample_idx: u32,
        n_candidates: usize,
    ) -> Vec<u32> {
        if marker_idx >= self.n_markers || (ref_state as usize) >= self.n_haps {
            return Vec::new();
        }

        let hap1 = sample_idx * 2;
        let hap2 = sample_idx * 2 + 1;

        // O(1) position lookup: where is ref_state in the sorted PBWT at this marker?
        let center_pos = self.fwd_pos[marker_idx][ref_state as usize] as usize;

        let ppa = &self.fwd_ppa[marker_idx];
        let div = &self.fwd_div[marker_idx];
        let marker_i32 = marker_idx as i32;

        let mut neighbors = Vec::with_capacity(n_candidates + 4);

        // Expand outward from center_pos, respecting divergence constraints
        let mut u = center_pos;
        let mut v = center_pos + 1;
        let mut max_div_u = i32::MIN;
        let mut max_div_v = i32::MIN;

        while neighbors.len() < n_candidates {
            let can_go_u = u > 0;
            let can_go_v = v < self.n_haps;

            if !can_go_u && !can_go_v {
                break;
            }

            // Prefer direction with better divergence (longer match)
            let prefer_u = if can_go_u && can_go_v {
                let div_u = div.get(u).copied().unwrap_or(i32::MAX);
                let div_v = div.get(v).copied().unwrap_or(i32::MAX);
                div_u <= div_v
            } else {
                can_go_u
            };

            if prefer_u && can_go_u {
                max_div_u = max_div_u.max(div.get(u).copied().unwrap_or(i32::MAX));
                u -= 1;
                let h = ppa[u];
                if h != hap1 && h != hap2 && h != ref_state {
                    neighbors.push(h);
                }
            } else if can_go_v {
                max_div_v = max_div_v.max(div.get(v).copied().unwrap_or(i32::MAX));
                let h = ppa[v];
                if h != hap1 && h != hap2 && h != ref_state {
                    neighbors.push(h);
                }
                v += 1;
            }

            // Stop expanding in a direction if divergence exceeds marker (match broken)
            if max_div_u > marker_i32 && max_div_v > marker_i32 && neighbors.len() >= n_candidates / 2 {
                break;
            }
        }

        neighbors
    }
}

fn normalize_pbwt_alleles(alleles: &mut [u8]) -> usize {
    let mut max_allele = 1u8;
    for &a in alleles.iter() {
        if a != 255 && a > max_allele {
            max_allele = a;
        }
    }
    let n_alleles = (max_allele as usize + 1).max(2);
    let missing_u8 = n_alleles as u8;
    for a in alleles.iter_mut() {
        if *a == 255 {
            *a = missing_u8;
        }
    }
    n_alleles + 1
}
