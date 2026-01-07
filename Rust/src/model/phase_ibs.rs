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
    /// Forward divergence at each marker: fwd_div[m] = divergence array after processing markers 0..=m
    fwd_div: Vec<Vec<i32>>,
    /// Forward prefix array at each marker
    fwd_ppa: Vec<Vec<u32>>,
    /// Backward divergence at each marker: bwd_div[m] = divergence array after processing markers m..n_markers
    bwd_div: Vec<Vec<i32>>,
    /// Backward prefix array at each marker
    bwd_ppa: Vec<Vec<u32>>,
    n_haps: usize,
    n_markers: usize,
    /// Optional mapping from subset marker index to global marker index.
    /// When Some, IBS2 lookups use the mapped global index.
    /// When None (full chromosome), marker indices are used directly.
    subset_to_global: Option<Vec<usize>>,
}

impl BidirectionalPhaseIbs {
    /// Build bidirectional PBWT from genotype data
    ///
    /// # Arguments
    /// * `alleles` - Allele data per marker
    /// * `n_haps` - Number of haplotypes
    /// * `n_markers` - Number of markers
    pub fn build(
        alleles: &[Vec<u8>],
        n_haps: usize,
        n_markers: usize,
    ) -> Self {
        let mut fwd_div = Vec::with_capacity(n_markers);
        let mut fwd_ppa = Vec::with_capacity(n_markers);
        let mut bwd_div = vec![Vec::new(); n_markers];
        let mut bwd_ppa = vec![Vec::new(); n_markers];

        let mut updater = PbwtDivUpdater::new(n_haps);

        let mut ppa: Vec<u32> = (0..n_haps as u32).collect();
        let mut div: Vec<i32> = vec![0; n_haps + 1];

        for m in 0..n_markers {
            updater.fwd_update(&alleles[m], 2, m, &mut ppa, &mut div);
            fwd_ppa.push(ppa.clone());
            fwd_div.push(div[..n_haps].to_vec());
        }

        ppa = (0..n_haps as u32).collect();
        div = vec![n_markers as i32; n_haps + 1];

        for m in (0..n_markers).rev() {
            updater.bwd_update(&alleles[m], 2, m, &mut ppa, &mut div);
            bwd_ppa[m] = ppa.clone();
            bwd_div[m] = div[..n_haps].to_vec();
        }

        Self {
            fwd_div,
            fwd_ppa,
            bwd_div,
            bwd_ppa,
            n_haps,
            n_markers,
            subset_to_global: None,
        }
    }

    /// Build bidirectional PBWT for a marker subset with global index mapping
    ///
    /// This variant stores the subset-to-global marker mapping so that IBS2
    /// lookups (which use global indices) work correctly when the PBWT is
    /// built on a marker subset (e.g., high-frequency markers in Stage 1).
    ///
    /// # Arguments
    /// * `alleles` - Allele data per subset marker
    /// * `n_haps` - Number of haplotypes
    /// * `n_markers` - Number of markers in subset
    /// * `subset_to_global` - Mapping from subset index to global marker index
    pub fn build_for_subset(
        alleles: &[Vec<u8>],
        n_haps: usize,
        n_markers: usize,
        subset_to_global: &[usize],
    ) -> Self {
        let mut result = Self::build(alleles, n_haps, n_markers);
        result.subset_to_global = Some(subset_to_global.to_vec());
        result
    }

    /// Find neighbors at a marker using both forward and backward PBWT
    ///
    /// When built with `build_for_subset`, automatically converts the subset
    /// marker index to global space for IBS2 segment lookups.
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
            if h != hap_idx && h / 2 != sample.0 && !neighbors.contains(&h) {
                neighbors.push(h);
            }
        }

        for h in bwd_neighbors {
            if h != hap_idx && h / 2 != sample.0 && !neighbors.contains(&h) {
                neighbors.push(h);
            }
        }

        neighbors
    }

    fn find_fwd_neighbors(&self, hap_idx: u32, marker_idx: usize, n_candidates: usize) -> Vec<u32> {
        if marker_idx >= self.n_markers {
            return Vec::new();
        }

        let ppa = &self.fwd_ppa[marker_idx];
        let div = &self.fwd_div[marker_idx];

        let sorted_pos = ppa.iter().position(|&h| h == hap_idx).unwrap_or(0);
        let marker_i32 = marker_idx as i32;

        // For forward PBWT, div[i] = marker where match started (divergence point).
        // A valid match has div[i] <= marker_idx (match still active).
        // Java: continue while d[u] <= step (match started at or before current step)
        // Note: backoff_limit is available for future backoff implementation but not used currently

        let mut result = Vec::with_capacity(n_candidates);

        let mut u = sorted_pos;
        let mut max_div = i32::MIN;
        while result.len() < n_candidates / 2 && u > 0 {
            max_div = max_div.max(div.get(u).copied().unwrap_or(i32::MAX));
            // Forward PBWT: div values are where match started.
            // Stop collecting when max_div > marker_idx (no match at current marker)
            // AND max_div > backoff_limit (started too recently, outside tolerance).
            // But since backoff_limit <= marker_idx, we effectively stop when
            // max_div > backoff_limit (allowing backoff window).
            // Corrected condition: break if divergence exceeds backoff tolerance.
            // Match is valid if div <= backoff_limit + some tolerance
            // Java uses: continue while d[u] <= dMax where dMax = min(matchStart + backoff, step)
            // Simpler: break if max_div > marker_idx (no exact match and not within backoff)
            if max_div > marker_i32 {
                break;
            }
            u -= 1;
            let h = ppa[u];
            if h != hap_idx {
                result.push(h);
            }
        }

        let mut v = sorted_pos + 1;
        max_div = i32::MIN;
        while result.len() < n_candidates && v < self.n_haps {
            max_div = max_div.max(div.get(v).copied().unwrap_or(i32::MAX));
            // Same logic as above
            if max_div > marker_i32 {
                break;
            }
            result.push(ppa[v]);
            v += 1;
        }

        result
    }

    fn find_bwd_neighbors(&self, hap_idx: u32, marker_idx: usize, n_candidates: usize) -> Vec<u32> {
        if marker_idx >= self.n_markers {
            return Vec::new();
        }

        let ppa = &self.bwd_ppa[marker_idx];
        let div = &self.bwd_div[marker_idx];

        let sorted_pos = ppa.iter().position(|&h| h == hap_idx).unwrap_or(0);
        let marker_i32 = marker_idx as i32;

        // For backward PBWT, div[i] = marker where match ENDS (going backward).
        // A valid match has div[i] >= marker_idx (match continues at or past current marker).
        // Java: continue while step <= uNextMatchEnd || step <= vNextMatchEnd
        // i.e., continue while marker_idx <= div[i] (match still active at current marker)
        // Note: backoff_limit is available for future backoff implementation but not used currently

        let mut result = Vec::with_capacity(n_candidates);

        let mut u = sorted_pos;
        let mut min_div = i32::MAX;
        while result.len() < n_candidates / 2 && u > 0 {
            min_div = min_div.min(div.get(u).copied().unwrap_or(0));
            // Backward PBWT: div values are where match ends (going backward).
            // Stop collecting when min_div < marker_idx (match ended before current marker).
            if min_div < marker_i32 {
                break;
            }
            u -= 1;
            let h = ppa[u];
            if h != hap_idx {
                result.push(h);
            }
        }

        let mut v = sorted_pos + 1;
        min_div = i32::MAX;
        while result.len() < n_candidates && v < self.n_haps {
            min_div = min_div.min(div.get(v).copied().unwrap_or(0));
            // Same logic as above
            if min_div < marker_i32 {
                break;
            }
            result.push(ppa[v]);
            v += 1;
        }

        result
    }

    /// Get the number of haplotypes
    pub fn n_haps(&self) -> usize {
        self.n_haps
    }
}
