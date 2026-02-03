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

    /// Check whether a target allele can be mapped at this marker.
    #[inline]
    pub fn can_map_targ_allele(&self, marker: usize, targ_allele: u8) -> bool {
        self.map_targ_to_ref(marker, targ_allele).is_some()
    }

    /// Map a reference allele to target allele space (for emission checks).
    #[inline]
    fn map_ref_to_targ(&self, marker: usize, ref_allele: u8) -> Option<u8> {
        let mapping = self.allele_maps.get(marker)?.as_ref()?;
        mapping.reverse_map_allele(ref_allele)
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

    /// Build a PackedRefView aligned to a sparse subset of target markers.
    pub fn build_sparse<TargetState: PhaseState>(
        target_gt: &GenotypeMatrix<TargetState, AnyMarkerSpace>,
        ref_gt: &GenotypeMatrix<crate::data::storage::phase_state::Phased, RefSpace>,
        alignment: &MarkerAlignment<AnyMarkerSpace, RefSpace>,
        target_markers: &[usize],
    ) -> Self {
        let n_targets = target_gt.n_markers();
        let n_ref_haps = ref_gt.n_haplotypes();
        let mut columns: Vec<Option<PackedRefColumn>> = vec![None; n_targets];
        let mut allele_maps: Vec<Option<AlleleMapping>> = vec![None; n_targets];

        for &t in target_markers {
            if t >= n_targets {
                continue;
            }
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
