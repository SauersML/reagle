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
use crate::model::pbwt::{PbwtDivUpdater, PbwtIndex};

pub trait PbwtStateCache<I: PbwtIndex> {
    fn with_fwd_state<R>(
        this: &BidirectionalPhaseIbsImpl<I>,
        marker_idx: usize,
        f: impl FnOnce(&[I], &[i32]) -> R,
    ) -> R;
    fn with_bwd_state<R>(
        this: &BidirectionalPhaseIbsImpl<I>,
        marker_idx: usize,
        f: impl FnOnce(&[I], &[i32]) -> R,
    ) -> R;
}

/// Checkpoint interval for sparse PBWT storage.
/// PPA/div/pos arrays are only stored at every CHECKPOINT_INTERVAL markers.
/// This reduces memory from O(n_markers × n_haps × 24 bytes) to O(n_markers/INTERVAL × n_haps × 24 bytes).
/// Checkpoints are used for storage, and exact states are recomputed per marker when queried.
const PBWT_CHECKPOINT_INTERVAL: usize = 64;

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
pub struct BidirectionalPhaseIbsImpl<I: crate::model::pbwt::PbwtIndex> {
    /// Forward divergence at checkpoints: `fwd_div[checkpoint_idx]` = divergence array after
    /// processing markers 0..=checkpoint_marker. For position i in the sorted order, `div[i]` is
    /// the marker where the match with the haplotype at position i-1 started.
    /// Sparse storage: only stored at every PBWT_CHECKPOINT_INTERVAL markers.
    fwd_div: Vec<Vec<i32>>,
    /// Forward prefix array at checkpoints: `fwd_ppa[checkpoint_idx][i]` = haplotype index at
    /// sorted position i after processing markers 0..=checkpoint_marker
    fwd_ppa: Vec<Vec<I>>,
    /// Backward divergence at checkpoints: `bwd_div[checkpoint_idx]` = divergence array after
    /// processing markers checkpoint_marker..n_markers (in reverse). For position i, `div[i]` is
    /// the marker where the match with the haplotype at position i-1 ends.
    bwd_div: Vec<Vec<i32>>,
    /// Backward prefix array at checkpoints
    bwd_ppa: Vec<Vec<I>>,
    /// Marker indices for each checkpoint (sorted ascending)
    checkpoint_markers: Vec<usize>,
    /// Total number of haplotypes in the PBWT
    n_haps: usize,
    /// Number of markers in the PBWT (may be subset of full chromosome)
    n_markers: usize,
    /// Number of checkpoints stored (n_markers / PBWT_CHECKPOINT_INTERVAL + 1)
    n_checkpoints: usize,
    /// Optional mapping from subset marker index to global marker index.
    /// When Some, IBS2 lookups use the mapped global index since IBS2 segments
    /// are defined in global marker space.
    /// When None (full chromosome), marker indices are used directly.
    subset_to_global: Option<Vec<usize>>,
    /// Stored alleles in row-major layout for O(1) allele lookup.
    /// alleles_flat[m * n_haps + h] = allele of haplotype h at marker m.
    alleles_flat: Vec<u8>,
    /// Number of distinct alleles per marker (after normalization)
    n_alleles_by_marker: Vec<usize>,

    reference_start_hap: Option<u32>,
}

impl<I: crate::model::pbwt::PbwtIndex> BidirectionalPhaseIbsImpl<I>
where
    BidirectionalPhaseIbsImpl<I>: PbwtStateCache<I>,
{
    #[inline]
    fn marker_allele_row(&self, marker_idx: usize) -> &[u8] {
        let start = marker_idx.saturating_mul(self.n_haps);
        let end = start
            .saturating_add(self.n_haps)
            .min(self.alleles_flat.len());
        &self.alleles_flat[start..end]
    }

    pub fn set_reference_start_hap(&mut self, start: u32) {
        self.reference_start_hap = Some(start);
    }

    /// Build bidirectional PBWT from flat row-major allele data.
    pub fn build_flat(mut alleles_flat: Vec<u8>, n_haps: usize, n_markers: usize) -> Self {
        assert_eq!(
            alleles_flat.len(),
            n_markers.saturating_mul(n_haps),
            "PBWT flat allele buffer has wrong length"
        );
        let mut checkpoint_markers: Vec<usize> =
            (0..n_markers).step_by(PBWT_CHECKPOINT_INTERVAL).collect();
        if let Some(&last) = checkpoint_markers.last() {
            if last + 1 < n_markers {
                checkpoint_markers.push(n_markers - 1);
            }
        } else if n_markers > 0 {
            checkpoint_markers.push(n_markers - 1);
        }
        let n_checkpoints = checkpoint_markers.len();

        let mut fwd_div = Vec::with_capacity(n_checkpoints);
        let mut fwd_ppa = Vec::with_capacity(n_checkpoints);
        let mut bwd_div = vec![Vec::new(); n_checkpoints];
        let mut bwd_ppa = vec![Vec::new(); n_checkpoints];
        let mut n_alleles_by_marker = vec![2usize; n_markers];

        let mut updater = PbwtDivUpdater::new(n_haps);

        let mut ppa: Vec<I> = (0..n_haps).map(I::from_usize).collect();
        let mut div: Vec<i32> = vec![0; n_haps + 1];

        let mut next_checkpoint = 0usize;
        for m in 0..n_markers {
            let row_start = m.saturating_mul(n_haps);
            let row_end = row_start.saturating_add(n_haps);
            let row = &mut alleles_flat[row_start..row_end];
            let n_alleles = normalize_pbwt_alleles(row);
            n_alleles_by_marker[m] = n_alleles;
            updater.fwd_update(row, n_alleles, m, &mut ppa, &mut div);

            if next_checkpoint < n_checkpoints && m == checkpoint_markers[next_checkpoint] {
                fwd_ppa.push(ppa.clone());
                fwd_div.push(div[..n_haps].to_vec());
                next_checkpoint += 1;
            }
        }

        ppa = (0..n_haps).map(I::from_usize).collect();
        div = vec![n_markers as i32; n_haps + 1];

        let mut next_checkpoint = n_checkpoints;
        for m in (0..n_markers).rev() {
            let n_alleles = n_alleles_by_marker[m];
            let row_start = m.saturating_mul(n_haps);
            let row_end = row_start.saturating_add(n_haps);
            let row = &alleles_flat[row_start..row_end];
            updater.bwd_update(row, n_alleles, m, &mut ppa, &mut div);

            if next_checkpoint > 0 && m == checkpoint_markers[next_checkpoint - 1] {
                let checkpoint_idx = next_checkpoint - 1;
                bwd_ppa[checkpoint_idx] = ppa.clone();
                bwd_div[checkpoint_idx] = div[..n_haps].to_vec();
                next_checkpoint -= 1;
            }
        }

        Self {
            fwd_div,
            fwd_ppa,
            bwd_div,
            bwd_ppa,
            checkpoint_markers,
            n_haps,
            n_markers,
            n_checkpoints,
            subset_to_global: None,
            alleles_flat,
            n_alleles_by_marker,
            reference_start_hap: None,
        }
    }

    pub fn n_haps(&self) -> usize {
        self.n_haps
    }

    fn with_fwd_pos_at_marker<R>(
        &self,
        marker_idx: usize,
        hap_idx: u32,
        f: impl FnOnce(&[I], &[i32], usize) -> R,
    ) -> R {
        <BidirectionalPhaseIbsImpl<I> as PbwtStateCache<I>>::with_fwd_state(
            self,
            marker_idx,
            |ppa, div| {
                thread_local! {
                    static FWD_POS_AT: std::cell::RefCell<(usize, Vec<u32>)> =
                        std::cell::RefCell::new((usize::MAX, Vec::new()));
                }
                FWD_POS_AT.with(|cell| {
                    let mut cache = cell.borrow_mut();
                    if cache.0 != marker_idx || cache.1.len() != ppa.len() {
                        cache.1.clear();
                        cache.1.resize(ppa.len(), 0u32);
                        for (i, &h) in ppa.iter().enumerate() {
                            cache.1[h.to_usize()] = i as u32;
                        }
                        cache.0 = marker_idx;
                    }
                    let pos = cache.1[hap_idx as usize] as usize;
                    f(ppa, div, pos)
                })
            },
        )
    }

    fn with_bwd_pos_at_marker<R>(
        &self,
        marker_idx: usize,
        hap_idx: u32,
        f: impl FnOnce(&[I], &[i32], usize) -> R,
    ) -> R {
        <BidirectionalPhaseIbsImpl<I> as PbwtStateCache<I>>::with_bwd_state(
            self,
            marker_idx,
            |ppa, div| {
                thread_local! {
                    static BWD_POS_AT: std::cell::RefCell<(usize, Vec<u32>)> =
                        std::cell::RefCell::new((usize::MAX, Vec::new()));
                }
                BWD_POS_AT.with(|cell| {
                    let mut cache = cell.borrow_mut();
                    if cache.0 != marker_idx || cache.1.len() != ppa.len() {
                        cache.1.clear();
                        cache.1.resize(ppa.len(), 0u32);
                        for (i, &h) in ppa.iter().enumerate() {
                            cache.1[h.to_usize()] = i as u32;
                        }
                        cache.0 = marker_idx;
                    }
                    let pos = cache.1[hap_idx as usize] as usize;
                    f(ppa, div, pos)
                })
            },
        )
    }

    pub fn build_for_subset_flat(
        alleles_flat: Vec<u8>,
        n_haps: usize,
        n_markers: usize,
        subset_to_global: &[usize],
    ) -> Self {
        let mut result = Self::build_flat(alleles_flat, n_haps, n_markers);
        result.subset_to_global = Some(subset_to_global.to_vec());
        result
    }

    #[inline(always)]
    fn marker_to_checkpoint_floor(&self, marker_idx: usize) -> usize {
        match self.checkpoint_markers.binary_search(&marker_idx) {
            Ok(idx) => idx,
            Err(idx) => idx.saturating_sub(1),
        }
    }

    #[inline(always)]
    fn marker_to_checkpoint_ceil(&self, marker_idx: usize) -> usize {
        match self.checkpoint_markers.binary_search(&marker_idx) {
            Ok(idx) => idx,
            Err(idx) => idx.min(self.n_checkpoints.saturating_sub(1)),
        }
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
    /// Note: Uses sparse PBWT storage with exact per-marker recomputation from checkpoints.
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
        let span = self.best_match_span(hap_idx, marker_idx);
        let n_candidates = if span < PBWT_CHECKPOINT_INTERVAL / 2 {
            n_candidates.saturating_mul(2)
        } else {
            n_candidates
        };
        let mut neighbors = Vec::with_capacity(n_candidates * 2 + 10);
        let sample = SampleIdx::new(hap_idx / 2);

        let ref_start = self.reference_start_hap;
        let mut ibs2_fallback: Vec<u32> = Vec::new();

        // Convert marker index to global space for IBS2 lookup
        // IBS2 segments use global marker indices, but when built for a subset,
        // marker_idx is in subset space. The mapping handles this conversion.
        // Strict handling of subset_to_global mapping to prevent silent fallback errors
        // IBS2 segments use global marker indices.
        let ibs2_marker_idx = match &self.subset_to_global {
            Some(mapping) => *mapping
                .get(marker_idx)
                .expect("Marker index out of bounds for subset mapping"),
            None => marker_idx,
        };

        for seg in ibs2.segments(sample) {
            if seg.contains(ibs2_marker_idx) {
                let other_s = seg.other_sample;
                if other_s != sample {
                    if ref_start.is_some() {
                        ibs2_fallback.push(other_s.hap1().0);
                        ibs2_fallback.push(other_s.hap2().0);
                    } else {
                        neighbors.push(other_s.hap1().0);
                        neighbors.push(other_s.hap2().0);
                    }
                }
            }
        }

        let fwd_neighbors = self.find_fwd_neighbors(hap_idx, marker_idx, n_candidates);
        let bwd_neighbors = self.find_bwd_neighbors(hap_idx, marker_idx, n_candidates);

        for h in fwd_neighbors {
            if h != hap_idx && h / 2 != sample.0 {
                if ref_start.map_or(true, |start| h >= start) {
                    neighbors.push(h);
                }
            }
        }

        for h in bwd_neighbors {
            if h != hap_idx && h / 2 != sample.0 {
                if ref_start.map_or(true, |start| h >= start) {
                    neighbors.push(h);
                }
            }
        }

        if ref_start.is_some() && neighbors.len() < n_candidates {
            for h in ibs2_fallback {
                if h != hap_idx && h / 2 != sample.0 {
                    neighbors.push(h);
                }
            }
        }

        neighbors
    }

    /// Estimate the best match span (in marker steps) for a haplotype at a marker.
    ///
    /// Uses adjacent PBWT neighbors and divergence arrays to approximate the
    /// longest shared segment around `marker_idx`. Uses sparse checkpoint storage.
    pub fn best_match_span(&self, hap_idx: u32, marker_idx: usize) -> usize {
        if marker_idx >= self.n_markers || (hap_idx as usize) >= self.n_haps {
            return 0;
        }

        let mut best_fwd = 0usize;
        self.with_fwd_pos_at_marker(marker_idx, hap_idx, |_, div, pos_fwd| {
            for pos in [pos_fwd.wrapping_sub(1), pos_fwd + 1] {
                if pos < self.n_haps {
                    let start = div[pos];
                    if marker_idx as i32 >= start {
                        let span = (marker_idx as i32 - start + 1) as usize;
                        if span > best_fwd {
                            best_fwd = span;
                        }
                    }
                }
            }
        });

        let mut best_bwd = 0usize;
        self.with_bwd_pos_at_marker(marker_idx, hap_idx, |_, div, pos_bwd| {
            for pos in [pos_bwd.wrapping_sub(1), pos_bwd + 1] {
                if pos < self.n_haps {
                    let end = div[pos];
                    if end >= marker_idx as i32 {
                        let span = (end - marker_idx as i32 + 1) as usize;
                        if span > best_bwd {
                            best_bwd = span;
                        }
                    }
                }
            }
        });

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

        self.with_fwd_pos_at_marker(marker_idx, hap_idx, |ppa, div, sorted_pos| {
            let marker_i32 = marker_idx as i32;
            let mut result = Vec::with_capacity(n_candidates);

            let mut u = sorted_pos;
            let mut v = sorted_pos + 1;
            let mut max_div_up = i32::MIN;
            let mut max_div_down = i32::MIN;

            while result.len() < n_candidates {
                let div_up = if u > 0 {
                    div.get(u).copied().unwrap_or(i32::MAX)
                } else {
                    i32::MAX
                };
                let div_down = if v < self.n_haps {
                    div.get(v).copied().unwrap_or(i32::MAX)
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
                    let h = ppa[u].to_u32();
                    if h != hap_idx {
                        result.push(h);
                    }
                } else {
                    max_div_down = max_div_down.max(div_down);
                    let h = ppa[v].to_u32();
                    if h != hap_idx {
                        result.push(h);
                    }
                    v += 1;
                }
            }

            while result.len() < n_candidates && u > 0 {
                u -= 1;
                let h = ppa[u].to_u32();
                if h != hap_idx {
                    result.push(h);
                }
            }
            while result.len() < n_candidates && v < self.n_haps {
                let h = ppa[v].to_u32();
                if h != hap_idx {
                    result.push(h);
                }
                v += 1;
            }

            result
        })
    }

    fn find_bwd_neighbors(&self, hap_idx: u32, marker_idx: usize, n_candidates: usize) -> Vec<u32> {
        if marker_idx >= self.n_markers || (hap_idx as usize) >= self.n_haps {
            return Vec::new();
        }

        self.with_bwd_pos_at_marker(marker_idx, hap_idx, |ppa, div, sorted_pos| {
            let marker_i32 = marker_idx as i32;
            let mut result = Vec::with_capacity(n_candidates);

            let mut u = sorted_pos;
            let mut v = sorted_pos + 1;
            let mut min_div_up = i32::MAX;
            let mut min_div_down = i32::MAX;

            while result.len() < n_candidates {
                let div_up = if u > 0 {
                    div.get(u).copied().unwrap_or(0)
                } else {
                    0
                };
                let div_down = if v < self.n_haps {
                    div.get(v).copied().unwrap_or(0)
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
                    let h = ppa[u].to_u32();
                    if h != hap_idx {
                        result.push(h);
                    }
                } else {
                    min_div_down = min_div_down.min(div_down);
                    let h = ppa[v].to_u32();
                    if h != hap_idx {
                        result.push(h);
                    }
                    v += 1;
                }
            }

            while result.len() < n_candidates && u > 0 {
                u -= 1;
                let h = ppa[u].to_u32();
                if h != hap_idx {
                    result.push(h);
                }
            }
            while result.len() < n_candidates && v < self.n_haps {
                let h = ppa[v].to_u32();
                if h != hap_idx {
                    result.push(h);
                }
                v += 1;
            }

            result
        })
    }

    /// Get the allele of a reference haplotype at a marker.
    ///
    /// This is used during dynamic MCMC to retrieve the reference panel alleles
    /// when computing emissions for the HMM states.
    #[inline]
    pub fn allele(&self, marker: usize, hap: u32) -> u8 {
        let idx = marker
            .saturating_mul(self.n_haps)
            .saturating_add(hap as usize);
        self.alleles_flat.get(idx).copied().unwrap_or(255)
    }

    #[inline]
    pub fn fill_alleles_for_haps(&self, marker: usize, haps: &[u32], out: &mut [u8]) {
        let n = haps.len().min(out.len());
        let base = marker.saturating_mul(self.n_haps);
        for i in 0..n {
            let idx = base.saturating_add(haps[i] as usize);
            out[i] = self.alleles_flat.get(idx).copied().unwrap_or(255);
        }
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

        let exclude_sample = sample_idx != u32::MAX;
        let hap1 = if exclude_sample { sample_idx * 2 } else { 0 };
        let hap2 = if exclude_sample {
            sample_idx * 2 + 1
        } else {
            0
        };

        let ref_start = self.reference_start_hap;

        let requested = if ref_start.is_some() {
            n_candidates.saturating_mul(4)
        } else {
            n_candidates
        };

        self.with_fwd_pos_at_marker(marker_idx, ref_state, |ppa, div, center_pos| {
            let marker_i32 = marker_idx as i32;
            let mut neighbors = Vec::with_capacity(requested + 4);

            let mut u = center_pos;
            let mut v = center_pos + 1;
            let mut max_div_u = i32::MIN;
            let mut max_div_v = i32::MIN;

            while neighbors.len() < requested {
                let can_go_u = u > 0;
                let can_go_v = v < self.n_haps;

                if !can_go_u && !can_go_v {
                    break;
                }

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
                    let h = ppa[u].to_u32();
                    if (!exclude_sample || (h != hap1 && h != hap2)) && h != ref_state {
                        if ref_start.map_or(true, |start| h >= start) {
                            neighbors.push(h);
                        }
                    }
                } else if can_go_v {
                    max_div_v = max_div_v.max(div.get(v).copied().unwrap_or(i32::MAX));
                    let h = ppa[v].to_u32();
                    if (!exclude_sample || (h != hap1 && h != hap2)) && h != ref_state {
                        if ref_start.map_or(true, |start| h >= start) {
                            neighbors.push(h);
                        }
                    }
                    v += 1;
                }

                if max_div_u > marker_i32
                    && max_div_v > marker_i32
                    && neighbors.len() >= n_candidates / 2
                {
                    break;
                }
            }

            if neighbors.len() > n_candidates {
                neighbors.truncate(n_candidates);
            }
            neighbors
        })
    }
}

impl PbwtStateCache<u16> for BidirectionalPhaseIbsImpl<u16> {
    fn with_fwd_state<R>(
        this: &BidirectionalPhaseIbsImpl<u16>,
        marker_idx: usize,
        f: impl FnOnce(&[u16], &[i32]) -> R,
    ) -> R {
        let checkpoint_idx = this.marker_to_checkpoint_floor(marker_idx);
        let checkpoint_marker = this.checkpoint_markers[checkpoint_idx];
        if marker_idx == checkpoint_marker {
            return f(&this.fwd_ppa[checkpoint_idx], &this.fwd_div[checkpoint_idx]);
        }

        thread_local! {
            static FWD_STATE_CACHE: std::cell::RefCell<(usize, usize, Vec<u16>, Vec<i32>)> =
                std::cell::RefCell::new((usize::MAX, usize::MAX, Vec::new(), Vec::new()));
            static FWD_UPDATER: std::cell::RefCell<PbwtDivUpdater<u16>> =
                std::cell::RefCell::new(PbwtDivUpdater::new(0));
        }

        FWD_STATE_CACHE.with(|state_cell| {
            FWD_UPDATER.with(|upd_cell| {
                let mut state = state_cell.borrow_mut();
                if state.0 != checkpoint_idx
                    || state.1 != marker_idx
                    || state.2.len() != this.n_haps
                {
                    state.0 = checkpoint_idx;
                    state.1 = marker_idx;
                    state.2 = this.fwd_ppa[checkpoint_idx].clone();
                    state.3 = this.fwd_div[checkpoint_idx].clone();

                    let mut updater = upd_cell.borrow_mut();
                    if updater.n_haps() != this.n_haps {
                        *updater = PbwtDivUpdater::new(this.n_haps);
                    }
                    let mut ppa = std::mem::take(&mut state.2);
                    let mut div = std::mem::take(&mut state.3);
                    for m in (checkpoint_marker + 1)..=marker_idx {
                        let n_alleles = this.n_alleles_by_marker[m];
                        updater.fwd_update(
                            this.marker_allele_row(m),
                            n_alleles,
                            m,
                            &mut ppa,
                            &mut div,
                        );
                    }
                    state.2 = ppa;
                    state.3 = div;
                }
                f(&state.2, &state.3)
            })
        })
    }

    fn with_bwd_state<R>(
        this: &BidirectionalPhaseIbsImpl<u16>,
        marker_idx: usize,
        f: impl FnOnce(&[u16], &[i32]) -> R,
    ) -> R {
        let checkpoint_idx = this.marker_to_checkpoint_ceil(marker_idx);
        let checkpoint_marker = this.checkpoint_markers[checkpoint_idx];
        if marker_idx == checkpoint_marker {
            return f(&this.bwd_ppa[checkpoint_idx], &this.bwd_div[checkpoint_idx]);
        }

        thread_local! {
            static BWD_STATE_CACHE: std::cell::RefCell<(usize, usize, Vec<u16>, Vec<i32>)> =
                std::cell::RefCell::new((usize::MAX, usize::MAX, Vec::new(), Vec::new()));
            static BWD_UPDATER: std::cell::RefCell<PbwtDivUpdater<u16>> =
                std::cell::RefCell::new(PbwtDivUpdater::new(0));
        }

        BWD_STATE_CACHE.with(|state_cell| {
            BWD_UPDATER.with(|upd_cell| {
                let mut state = state_cell.borrow_mut();
                if state.0 != checkpoint_idx
                    || state.1 != marker_idx
                    || state.2.len() != this.n_haps
                {
                    state.0 = checkpoint_idx;
                    state.1 = marker_idx;
                    state.2 = this.bwd_ppa[checkpoint_idx].clone();
                    state.3 = this.bwd_div[checkpoint_idx].clone();

                    let mut updater = upd_cell.borrow_mut();
                    if updater.n_haps() != this.n_haps {
                        *updater = PbwtDivUpdater::new(this.n_haps);
                    }
                    let mut ppa = std::mem::take(&mut state.2);
                    let mut div = std::mem::take(&mut state.3);
                    for m in (marker_idx..checkpoint_marker).rev() {
                        let n_alleles = this.n_alleles_by_marker[m];
                        updater.bwd_update(
                            this.marker_allele_row(m),
                            n_alleles,
                            m,
                            &mut ppa,
                            &mut div,
                        );
                    }
                    state.2 = ppa;
                    state.3 = div;
                }
                f(&state.2, &state.3)
            })
        })
    }
}

impl PbwtStateCache<u32> for BidirectionalPhaseIbsImpl<u32> {
    fn with_fwd_state<R>(
        this: &BidirectionalPhaseIbsImpl<u32>,
        marker_idx: usize,
        f: impl FnOnce(&[u32], &[i32]) -> R,
    ) -> R {
        let checkpoint_idx = this.marker_to_checkpoint_floor(marker_idx);
        let checkpoint_marker = this.checkpoint_markers[checkpoint_idx];
        if marker_idx == checkpoint_marker {
            return f(&this.fwd_ppa[checkpoint_idx], &this.fwd_div[checkpoint_idx]);
        }

        thread_local! {
            static FWD_STATE_CACHE: std::cell::RefCell<(usize, usize, Vec<u32>, Vec<i32>)> =
                std::cell::RefCell::new((usize::MAX, usize::MAX, Vec::new(), Vec::new()));
            static FWD_UPDATER: std::cell::RefCell<PbwtDivUpdater<u32>> =
                std::cell::RefCell::new(PbwtDivUpdater::new(0));
        }

        FWD_STATE_CACHE.with(|state_cell| {
            FWD_UPDATER.with(|upd_cell| {
                let mut state = state_cell.borrow_mut();
                if state.0 != checkpoint_idx
                    || state.1 != marker_idx
                    || state.2.len() != this.n_haps
                {
                    state.0 = checkpoint_idx;
                    state.1 = marker_idx;
                    state.2 = this.fwd_ppa[checkpoint_idx].clone();
                    state.3 = this.fwd_div[checkpoint_idx].clone();

                    let mut updater = upd_cell.borrow_mut();
                    if updater.n_haps() != this.n_haps {
                        *updater = PbwtDivUpdater::new(this.n_haps);
                    }
                    let mut ppa = std::mem::take(&mut state.2);
                    let mut div = std::mem::take(&mut state.3);
                    for m in (checkpoint_marker + 1)..=marker_idx {
                        let n_alleles = this.n_alleles_by_marker[m];
                        updater.fwd_update(
                            this.marker_allele_row(m),
                            n_alleles,
                            m,
                            &mut ppa,
                            &mut div,
                        );
                    }
                    state.2 = ppa;
                    state.3 = div;
                }
                f(&state.2, &state.3)
            })
        })
    }

    fn with_bwd_state<R>(
        this: &BidirectionalPhaseIbsImpl<u32>,
        marker_idx: usize,
        f: impl FnOnce(&[u32], &[i32]) -> R,
    ) -> R {
        let checkpoint_idx = this.marker_to_checkpoint_ceil(marker_idx);
        let checkpoint_marker = this.checkpoint_markers[checkpoint_idx];
        if marker_idx == checkpoint_marker {
            return f(&this.bwd_ppa[checkpoint_idx], &this.bwd_div[checkpoint_idx]);
        }

        thread_local! {
            static BWD_STATE_CACHE: std::cell::RefCell<(usize, usize, Vec<u32>, Vec<i32>)> =
                std::cell::RefCell::new((usize::MAX, usize::MAX, Vec::new(), Vec::new()));
            static BWD_UPDATER: std::cell::RefCell<PbwtDivUpdater<u32>> =
                std::cell::RefCell::new(PbwtDivUpdater::new(0));
        }

        BWD_STATE_CACHE.with(|state_cell| {
            BWD_UPDATER.with(|upd_cell| {
                let mut state = state_cell.borrow_mut();
                if state.0 != checkpoint_idx
                    || state.1 != marker_idx
                    || state.2.len() != this.n_haps
                {
                    state.0 = checkpoint_idx;
                    state.1 = marker_idx;
                    state.2 = this.bwd_ppa[checkpoint_idx].clone();
                    state.3 = this.bwd_div[checkpoint_idx].clone();

                    let mut updater = upd_cell.borrow_mut();
                    if updater.n_haps() != this.n_haps {
                        *updater = PbwtDivUpdater::new(this.n_haps);
                    }
                    let mut ppa = std::mem::take(&mut state.2);
                    let mut div = std::mem::take(&mut state.3);
                    for m in (marker_idx..checkpoint_marker).rev() {
                        let n_alleles = this.n_alleles_by_marker[m];
                        updater.bwd_update(
                            this.marker_allele_row(m),
                            n_alleles,
                            m,
                            &mut ppa,
                            &mut div,
                        );
                    }
                    state.2 = ppa;
                    state.3 = div;
                }
                f(&state.2, &state.3)
            })
        })
    }
}

pub enum BidirectionalPhaseIbs {
    U16(BidirectionalPhaseIbsImpl<u16>),
    U32(BidirectionalPhaseIbsImpl<u32>),
}

impl BidirectionalPhaseIbs {
    pub fn build_for_subset_flat(
        alleles_flat: Vec<u8>,
        n_haps: usize,
        n_markers: usize,
        subset_to_global: &[usize],
    ) -> Self {
        if n_haps <= u16::MAX as usize {
            Self::U16(BidirectionalPhaseIbsImpl::<u16>::build_for_subset_flat(
                alleles_flat,
                n_haps,
                n_markers,
                subset_to_global,
            ))
        } else {
            Self::U32(BidirectionalPhaseIbsImpl::<u32>::build_for_subset_flat(
                alleles_flat,
                n_haps,
                n_markers,
                subset_to_global,
            ))
        }
    }

    pub fn n_haps(&self) -> usize {
        match self {
            Self::U16(inner) => inner.n_haps(),
            Self::U32(inner) => inner.n_haps(),
        }
    }

    pub fn best_match_span(&self, hap_idx: u32, marker_idx: usize) -> usize {
        match self {
            Self::U16(inner) => inner.best_match_span(hap_idx, marker_idx),
            Self::U32(inner) => inner.best_match_span(hap_idx, marker_idx),
        }
    }

    pub fn set_reference_start_hap(&mut self, start: u32) {
        match self {
            Self::U16(inner) => inner.set_reference_start_hap(start),
            Self::U32(inner) => inner.set_reference_start_hap(start),
        }
    }

    pub fn find_neighbors(
        &self,
        hap_idx: u32,
        marker_idx: usize,
        ibs2: &Ibs2,
        n_candidates: usize,
    ) -> Vec<u32> {
        match self {
            Self::U16(inner) => inner.find_neighbors(hap_idx, marker_idx, ibs2, n_candidates),
            Self::U32(inner) => inner.find_neighbors(hap_idx, marker_idx, ibs2, n_candidates),
        }
    }

    #[inline]
    pub fn allele(&self, marker: usize, hap: u32) -> u8 {
        match self {
            Self::U16(inner) => inner.allele(marker, hap),
            Self::U32(inner) => inner.allele(marker, hap),
        }
    }

    #[inline]
    pub fn fill_alleles_for_haps(&self, marker: usize, haps: &[u32], out: &mut [u8]) {
        match self {
            Self::U16(inner) => inner.fill_alleles_for_haps(marker, haps, out),
            Self::U32(inner) => inner.fill_alleles_for_haps(marker, haps, out),
        }
    }

    pub fn find_neighbors_of_state(
        &self,
        ref_state: u32,
        marker_idx: usize,
        sample_idx: u32,
        n_candidates: usize,
    ) -> Vec<u32> {
        match self {
            Self::U16(inner) => {
                inner.find_neighbors_of_state(ref_state, marker_idx, sample_idx, n_candidates)
            }
            Self::U32(inner) => {
                inner.find_neighbors_of_state(ref_state, marker_idx, sample_idx, n_candidates)
            }
        }
    }
}

fn normalize_pbwt_alleles(alleles: &mut [u8]) -> usize {
    let mut max_allele = 1u8;
    for &a in alleles.iter() {
        if a != 255 && a > max_allele {
            max_allele = a;
        }
    }
    (max_allele as usize + 1).max(2)
}
