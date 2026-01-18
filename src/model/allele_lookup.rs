//! Pre-computed allele lookup for HMM states.
//!
//! Stores alleles as a flat Vec<u8> with layout:
//! [marker0_state0, marker0_state1, ..., marker1_state0, ...]
//! This eliminates per-lookup alignment and reference indirection.

use aligned_vec::{AVec, ConstAlign};

use crate::data::alignment::MarkerAlignment;
use crate::data::haplotype::HapIdx;
use crate::data::marker::MarkerIdx;
use crate::data::storage::phase_state::Phased;
use crate::data::storage::{GenotypeMatrix, MutableGenotypes};
use crate::model::states::ThreadedHaps;

/// Pre-computed allele lookup for HMM states.
pub struct RefAlleleLookup {
    /// alleles[m * n_states + k] = allele for state k at marker m
    alleles: AVec<u8, ConstAlign<32>>,
    n_states: usize,
}

impl RefAlleleLookup {
    /// Create a new lookup directly from ThreadedHaps without intermediate allocation.
    ///
    /// This avoids the O(n_markers × n_states × 4) temporary from materialize_all().
    pub fn new_from_threaded_with_buffer(
        threaded_haps: &ThreadedHaps,
        n_markers: usize,
        n_states: usize,
        n_target_haps: usize,
        ref_geno: &MutableGenotypes,
        reference_gt: Option<&GenotypeMatrix<Phased>>,
        alignment: Option<&MarkerAlignment>,
        marker_map: Option<&[usize]>,
        mut alleles: AVec<u8, ConstAlign<32>>,
    ) -> Self {
        let required = n_markers * n_states;
        if alleles.len() < required {
            alleles = AVec::from_iter(32, std::iter::repeat(0u8).take(required));
        } else {
            alleles[..required].fill(0);
        }

        // Use marker-major iteration to hoist per-marker alignment computation
        threaded_haps.fill_alleles_marker_major(&mut alleles, |m| {
            let orig_m = marker_map.map(|map| map[m]).unwrap_or(m);
            let ref_m_opt = alignment.and_then(|a| a.target_to_ref(orig_m));

            move |hap: u32| {
                let hap = hap as usize;
                if hap < n_target_haps {
                    ref_geno.get(orig_m, HapIdx::new(hap as u32))
                } else {
                    let ref_h = (hap - n_target_haps) as u32;
                    if let (Some(ref_gt), Some(ref_m)) = (reference_gt, ref_m_opt) {
                        let ref_allele = ref_gt.allele(MarkerIdx::new(ref_m as u32), HapIdx::new(ref_h));
                        alignment.unwrap().reverse_map_allele(orig_m, ref_allele)
                    } else {
                        255
                    }
                }
            }
        });

        Self { alleles, n_states }
    }

    #[inline(always)]
    pub fn allele(&self, marker: usize, state: usize) -> u8 {
        self.alleles[marker * self.n_states + state]
    }

    pub fn n_states(&self) -> usize {
        self.n_states
    }

    pub fn into_buffer(self) -> AVec<u8, ConstAlign<32>> {
        self.alleles
    }
}
