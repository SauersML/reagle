use crate::data::haplotype::SampleIdx;
use crate::model::ibs2::Ibs2;
use crate::model::pbwt::PbwtDivUpdater;

/// Manages bidirectional PBWT state for HMM state selection.
///
/// Stores both forward and backward PBWT arrays at each marker to enable
/// selecting haplotypes that match well both upstream and downstream.
/// This is critical for phasing accuracy around recombination hotspots.
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
}

impl BidirectionalPhaseIbs {
    /// Build bidirectional PBWT from genotype data
    pub fn build(alleles: &[Vec<u8>], n_haps: usize, n_markers: usize) -> Self {
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
        }
    }

    /// Find neighbors at a marker using both forward and backward PBWT
    pub fn find_neighbors(
        &self,
        hap_idx: u32,
        marker_idx: usize,
        ibs2: &Ibs2,
        n_candidates: usize,
    ) -> Vec<u32> {
        let mut neighbors = Vec::with_capacity(n_candidates * 2 + 10);
        let sample = SampleIdx::new(hap_idx / 2);

        for seg in ibs2.segments(sample) {
            if seg.contains(marker_idx) {
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

        let mut result = Vec::with_capacity(n_candidates);

        let mut u = sorted_pos;
        let mut max_div = i32::MIN;
        while result.len() < n_candidates / 2 && u > 0 {
            max_div = max_div.max(div.get(u).copied().unwrap_or(i32::MAX));
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

        let mut result = Vec::with_capacity(n_candidates);

        let mut u = sorted_pos;
        let mut min_div = i32::MAX;
        while result.len() < n_candidates / 2 && u > 0 {
            min_div = min_div.min(div.get(u).copied().unwrap_or(0));
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
            if min_div < marker_i32 {
                break;
            }
            result.push(ppa[v]);
            v += 1;
        }

        result
    }
}
