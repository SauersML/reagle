//! Packed reference view for fast bit-parallel allele access.
//!
//! This view aligns reference alleles to target markers (via MarkerAlignment)
//! and stores packed reference columns for fast mask construction.

use crate::data::alignment::MarkerAlignment;
use crate::data::marker::AnyMarkerSpace;
use crate::data::marker::AlleleMapping;
use crate::data::storage::GenotypeMatrix;
use crate::data::storage::phase_state::PhaseState;
use crate::io::prescan_cache::PackedRefColumn;

/// Packed reference columns aligned to target marker indices.
#[derive(Clone, Debug)]
pub struct PackedRefView<RefSpace = AnyMarkerSpace> {
    n_ref_haps: usize,
    columns: Vec<Option<PackedRefColumn>>, // indexed by target marker
    allele_maps: Vec<Option<AlleleMapping>>, // indexed by target marker
    phantom: std::marker::PhantomData<RefSpace>,
}

impl<RefSpace> PackedRefView<RefSpace> {
    pub fn n_ref_haps(&self) -> usize {
        self.n_ref_haps
    }

    /// Get the mapped reference allele for a target allele at this marker.
    #[inline]
    fn map_targ_to_ref(&self, marker: usize, targ_allele: u8) -> Option<u8> {
        let mapping = self.allele_maps.get(marker)?.as_ref()?;
        let idx = *mapping.targ_to_ref.get(targ_allele as usize)?;
        if idx < 0 {
            None
        } else {
            Some(idx as u8)
        }
    }

    /// Map a reference allele to target allele space (for emission checks).
    #[inline]
    fn map_ref_to_targ(&self, marker: usize, ref_allele: u8) -> Option<u8> {
        let mapping = self.allele_maps.get(marker)?.as_ref()?;
        mapping.reverse_map_allele(ref_allele)
    }

    /// Fill `out` with a match mask for the given target allele.
    ///
    /// Returns false if the marker isn't aligned or the allele can't be mapped.
    pub fn fill_match_mask(&self, marker: usize, targ_allele: u8, out: &mut [u64]) -> bool {
        let col = match self.columns.get(marker).and_then(|v| v.as_ref()) {
            Some(col) => col,
            None => return false,
        };
        let ref_allele = match self.map_targ_to_ref(marker, targ_allele) {
            Some(a) => a,
            None => return false,
        };

        match col {
            PackedRefColumn::Bits {
                bits,
                n_haps,
                words,
                missing,
            } => {
                if *bits != 1 {
                    // Non-biallelic: fall back to per-hap scan.
                    let n = (*n_haps).min(self.n_ref_haps);
                    out.fill(0);
                    for h in 0..n {
                        if col.allele(h) == ref_allele {
                            let w = h / 64;
                            let b = h % 64;
                            out[w] |= 1u64 << b;
                        }
                    }
                    return true;
                }
                let n_words = out.len();
                out[..n_words].fill(0);
                let miss_words = missing.len().min(n_words);
                let data_words = words.len().min(n_words);
                for w in 0..n_words {
                    let miss = if w < miss_words { missing[w] } else { 0 };
                    let bits_word = if w < data_words { words[w] } else { 0 };
                    let match_word = if ref_allele == 1 {
                        bits_word & !miss
                    } else if ref_allele == 0 {
                        (!bits_word) & !miss
                    } else {
                        0
                    };
                    out[w] = match_word;
                }
                true
            }
            PackedRefColumn::Bytes { alleles } => {
                out.fill(0);
                let n = alleles.len().min(self.n_ref_haps);
                for h in 0..n {
                    if alleles[h] == ref_allele {
                        let w = h / 64;
                        let b = h % 64;
                        out[w] |= 1u64 << b;
                    }
                }
                true
            }
        }
    }

    /// Get the reference allele (mapped to target allele space) for a given hap at marker.
    pub fn ref_allele_targ(&self, marker: usize, hap: usize) -> Option<u8> {
        let col = self.columns.get(marker).and_then(|v| v.as_ref())?;
        let ref_al = col.allele(hap);
        if ref_al == 255 {
            return None;
        }
        self.map_ref_to_targ(marker, ref_al)
    }

}

impl<RefSpace: Send + Sync> PackedRefView<RefSpace> {
    /// Build a PackedRefView aligned to target markers.
    pub fn build<TargetState: PhaseState>(
        target_gt: &GenotypeMatrix<TargetState, AnyMarkerSpace>,
        ref_gt: &GenotypeMatrix<crate::data::storage::phase_state::Phased, RefSpace>,
        alignment: &MarkerAlignment<AnyMarkerSpace, RefSpace>,
    ) -> Self {
        let n_targets = target_gt.n_markers();
        let n_ref_haps = ref_gt.n_haplotypes();
        let mut columns: Vec<Option<PackedRefColumn>> = vec![None; n_targets];
        let mut allele_maps: Vec<Option<AlleleMapping>> = vec![None; n_targets];

        for t in 0..n_targets {
            if let Some(r_idx) = alignment.target_to_ref[t] {
                let col = ref_gt.column(r_idx);
                let packed = PackedRefColumn::pack_from_column(r_idx, ref_gt.markers(), &col);
                columns[t] = Some(packed);
                allele_maps[t] = alignment.allele_mappings[t].clone();
            }
        }

        Self {
            n_ref_haps,
            columns,
            allele_maps,
            phantom: std::marker::PhantomData,
        }
    }
}

/// Utility: mask all bits on for `n` haplotypes.
#[inline]
pub fn mask_all_ones(mask: &mut [u64], n_haps: usize) {
    mask.fill(!0u64);
    let n_words = mask.len();
    if n_words == 0 {
        return;
    }
    let rem = n_haps % 64;
    if rem != 0 {
        let last = n_words - 1;
        let keep = (1u64 << rem) - 1;
        mask[last] = keep;
    }
}

#[inline]
pub fn mask_and_inplace(dst: &mut [u64], src: &[u64]) {
    let n = dst.len().min(src.len());
    for i in 0..n {
        dst[i] &= src[i];
    }
}

#[inline]
pub fn mask_bit_is_set(mask: &[u64], hap: usize) -> bool {
    let w = hap / 64;
    let b = hap % 64;
    if w >= mask.len() {
        return false;
    }
    ((mask[w] >> b) & 1) != 0
}
