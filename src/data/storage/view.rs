//! # Genotype View
//!
//! A read-only view over different genotype storage types.
//! This provides a unified interface for algorithms (like HMM)
//! to operate on both immutable `GenotypeMatrix` and `MutableGenotypes`.

use crate::data::alignment::MarkerAlignment;
use crate::data::haplotype::HapIdx;
use crate::data::marker::{AnyMarkerSpace, MarkerIdx, Markers};
use crate::data::storage::{GenotypeMatrix, MutableGenotypes, phase_state};

/// A read-only view of genotype data - allows HMM to work with either
/// GenotypeMatrix or MutableGenotypes without caring about concrete type
pub enum GenotypeView<'a, TargetSpace = AnyMarkerSpace, RefSpace = AnyMarkerSpace> {
    /// View over an immutable GenotypeMatrix
    Matrix(&'a GenotypeMatrix<phase_state::Unphased, TargetSpace>),
    /// View over MutableGenotypes
    Mutable(&'a MutableGenotypes),
    /// View over a subset of markers in MutableGenotypes
    MutableSubset {
        geno: &'a MutableGenotypes,
        subset: &'a [usize],
    },
    /// Composite view: Target (mutable) + Reference (immutable)
    /// Haplotype indices 0..n_target_haps -> target, n_target_haps..n_total -> reference
    Composite {
        target: &'a MutableGenotypes,
        reference: &'a GenotypeMatrix<phase_state::Phased, RefSpace>,
        alignment: &'a MarkerAlignment<TargetSpace, RefSpace>,
        n_target_haps: usize,
    },
    /// Composite view over a marker subset (for Stage 1 hi-freq markers)
    /// Combines target + reference with marker subset mapping
    CompositeSubset {
        target: &'a MutableGenotypes,
        reference: &'a GenotypeMatrix<phase_state::Phased, RefSpace>,
        alignment: &'a MarkerAlignment<TargetSpace, RefSpace>,
        subset: &'a [usize],
        n_target_haps: usize,
    },
}

impl<'a, TargetSpace, RefSpace> Copy for GenotypeView<'a, TargetSpace, RefSpace> {}

impl<'a, TargetSpace, RefSpace> Clone for GenotypeView<'a, TargetSpace, RefSpace> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, TargetSpace, RefSpace> GenotypeView<'a, TargetSpace, RefSpace> {
    /// Get the number of markers
    #[inline]
    pub fn n_markers(&self) -> usize {
        match self {
            GenotypeView::Matrix(m) => m.n_markers(),
            GenotypeView::Mutable(geno) => geno.n_markers(),
            GenotypeView::MutableSubset { subset, .. } => subset.len(),
            GenotypeView::Composite { target, .. } => target.n_markers(),
            GenotypeView::CompositeSubset { subset, .. } => subset.len(),
        }
    }

    /// Get the number of haplotypes represented by this view.
    #[inline]
    pub fn n_haps(&self) -> usize {
        match self {
            GenotypeView::Matrix(m) => m.samples_arc().n_haps(),
            GenotypeView::Mutable(geno) => geno.n_haps(),
            GenotypeView::MutableSubset { geno, .. } => geno.n_haps(),
            GenotypeView::Composite {
                target, reference, ..
            } => target.n_haps() + reference.samples_arc().n_haps(),
            GenotypeView::CompositeSubset {
                target, reference, ..
            } => target.n_haps() + reference.samples_arc().n_haps(),
        }
    }

    /// Get an allele at a specific marker and haplotype index
    #[inline]
    pub fn allele(&self, marker: MarkerIdx<TargetSpace>, hap: HapIdx) -> u8 {
        match self {
            GenotypeView::Matrix(m) => m.allele(marker, hap),
            GenotypeView::Mutable(geno) => geno.get(marker.as_usize(), hap),
            GenotypeView::MutableSubset { geno, subset } => {
                let real_idx = subset[marker.as_usize()];
                geno.get(real_idx, hap)
            }
            GenotypeView::Composite {
                target,
                reference,
                alignment,
                n_target_haps,
            } => {
                let hap_idx = hap.as_usize();
                if hap_idx < *n_target_haps {
                    // Target haplotype - direct lookup
                    target.get(marker.as_usize(), hap)
                } else {
                    // Reference haplotype - translate marker index, look up, and map allele to target encoding
                    let ref_hap = hap_idx - n_target_haps;
                    let target_marker = marker.as_usize();
                    if let Some(ref_m) =
                        alignment.target_to_ref(MarkerIdx::new(target_marker as u32))
                    {
                        let ref_allele = reference.allele(ref_m, HapIdx::new(ref_hap as u32));
                        // Map reference allele back to target encoding (handles strand flips)
                        alignment.reverse_map_allele(target_marker, ref_allele)
                    } else {
                        crate::data::storage::AlleleCode::MISSING.raw() // Marker not in reference - return missing
                    }
                }
            }
            GenotypeView::CompositeSubset {
                target,
                reference,
                alignment,
                subset,
                n_target_haps,
            } => {
                let orig_marker = subset[marker.as_usize()]; // Subset index -> original target marker index
                let hap_idx = hap.as_usize();
                if hap_idx < *n_target_haps {
                    // Target haplotype - direct lookup using original marker index
                    target.get(orig_marker, hap)
                } else {
                    // Reference haplotype - translate marker, look up, and map allele to target encoding
                    let ref_hap = hap_idx - n_target_haps;
                    if let Some(ref_m) = alignment.target_to_ref(MarkerIdx::new(orig_marker as u32))
                    {
                        let ref_allele = reference.allele(ref_m, HapIdx::new(ref_hap as u32));
                        // Map reference allele back to target encoding (handles strand flips)
                        alignment.reverse_map_allele(orig_marker, ref_allele)
                    } else {
                        crate::data::storage::AlleleCode::MISSING.raw() // Marker not in reference - return missing
                    }
                }
            }
        }
    }

    /// Fill a batch of alleles for the provided haplotypes at a marker.
    #[inline]
    pub fn fill_batch(&self, marker: MarkerIdx<TargetSpace>, haps: &[HapIdx], out: &mut [u8]) {
        let n = haps.len().min(out.len());
        match self {
            GenotypeView::Matrix(m) => m.fill_batch(marker, &haps[..n], &mut out[..n]),
            GenotypeView::Mutable(geno) => {
                geno.fill_batch(marker.as_usize(), &haps[..n], &mut out[..n]);
            }
            GenotypeView::MutableSubset { geno, subset } => {
                let real_idx = subset[marker.as_usize()];
                geno.fill_batch(real_idx, &haps[..n], &mut out[..n]);
            }
            GenotypeView::Composite {
                target,
                reference,
                alignment,
                n_target_haps,
            } => {
                let target_marker = marker.as_usize();
                let ref_marker = alignment.target_to_ref(MarkerIdx::new(target_marker as u32));
                for i in 0..n {
                    let hap_idx = haps[i].as_usize();
                    if hap_idx < *n_target_haps {
                        out[i] = target.get(target_marker, haps[i]);
                    } else if let Some(ref_m) = ref_marker {
                        let ref_hap = hap_idx - n_target_haps;
                        let ref_allele = reference.allele(ref_m, HapIdx::new(ref_hap as u32));
                        out[i] = alignment.reverse_map_allele(target_marker, ref_allele);
                    } else {
                        out[i] = crate::data::storage::AlleleCode::MISSING.raw();
                    }
                }
            }
            GenotypeView::CompositeSubset {
                target,
                reference,
                alignment,
                subset,
                n_target_haps,
            } => {
                let orig_marker = subset[marker.as_usize()];
                let ref_marker = alignment.target_to_ref(MarkerIdx::new(orig_marker as u32));
                for i in 0..n {
                    let hap_idx = haps[i].as_usize();
                    if hap_idx < *n_target_haps {
                        out[i] = target.get(orig_marker, haps[i]);
                    } else if let Some(ref_m) = ref_marker {
                        let ref_hap = hap_idx - n_target_haps;
                        let ref_allele = reference.allele(ref_m, HapIdx::new(ref_hap as u32));
                        out[i] = alignment.reverse_map_allele(orig_marker, ref_allele);
                    } else {
                        out[i] = crate::data::storage::AlleleCode::MISSING.raw();
                    }
                }
            }
        }
    }
}

/// Conversion from `&GenotypeMatrix` (Unphased) to `GenotypeView`
impl<'a, Space> From<&'a GenotypeMatrix<phase_state::Unphased, Space>> for GenotypeView<'a, Space> {
    fn from(matrix: &'a GenotypeMatrix<phase_state::Unphased, Space>) -> Self {
        GenotypeView::Matrix(matrix)
    }
}

/// Conversion from `&GenotypeMatrix<Phased>` to `GenotypeView`
impl<'a, Space> From<&'a GenotypeMatrix<phase_state::Phased, Space>> for GenotypeView<'a, Space> {
    fn from(matrix: &'a GenotypeMatrix<phase_state::Phased, Space>) -> Self {
        GenotypeView::Matrix(matrix.as_unphased_ref())
    }
}

/// Conversion from `(&'a MutableGenotypes, &'a Markers)` to `GenotypeView`
/// Note: markers are not stored since they're not needed for allele access
impl<'a, Space> From<(&'a MutableGenotypes, &'a Markers<Space>)> for GenotypeView<'a, Space> {
    fn from(tuple: (&'a MutableGenotypes, &'a Markers<Space>)) -> Self {
        GenotypeView::Mutable(tuple.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::ChromIdx;
    use crate::data::haplotype::Samples;
    use crate::data::marker::{Allele, Marker, Nucleotide};
    use crate::data::storage::phase_state::Phased;
    use crate::data::storage::{GenotypeColumn, MutableGenotypes};
    use std::sync::Arc;

    fn make_test_matrix() -> GenotypeMatrix<Phased> {
        let samples = Arc::new(Samples::from_ids(vec!["S1".to_string(), "S2".to_string()]));
        let mut markers = Markers::<crate::data::AnyMarkerSpace>::new();
        markers.add_chrom("chr1");
        markers.push(Marker::new(
            ChromIdx::new(0),
            100,
            None,
            Allele::Base(Nucleotide::A),
            vec![Allele::Base(Nucleotide::C)],
        ));
        let col = GenotypeColumn::from_alleles(&[0, 1, 0, 1], 2);
        GenotypeMatrix::new_phased(markers, vec![col], samples)
    }

    fn make_test_mutable() -> (MutableGenotypes, Markers) {
        let mut markers = Markers::<crate::data::AnyMarkerSpace>::new();
        markers.add_chrom("chr1");
        markers.push(Marker::new(
            ChromIdx::new(0),
            200,
            None,
            Allele::Base(Nucleotide::C),
            vec![Allele::Base(Nucleotide::A)],
        ));
        let geno = MutableGenotypes::from_fn(1, 2, |_, h| h as u8);
        (geno, markers)
    }

    #[test]
    fn test_view_from_matrix() {
        let matrix = make_test_matrix();
        let view = GenotypeView::from(&matrix);

        assert_eq!(view.n_markers(), 1);
        assert_eq!(view.allele(MarkerIdx::new(0), HapIdx::new(1)), 1);
    }

    #[test]
    fn test_view_from_mutable() {
        let (geno, markers) = make_test_mutable();
        let view = GenotypeView::from((&geno, &markers));

        assert_eq!(view.n_markers(), 1);
        assert_eq!(view.allele(MarkerIdx::new(0), HapIdx::new(1)), 1);
    }

    #[test]
    fn test_fill_batch_matches_single() {
        let matrix = make_test_matrix();
        let view = GenotypeView::from(&matrix);
        let haps = [
            HapIdx::new(0),
            HapIdx::new(1),
            HapIdx::new(2),
            HapIdx::new(3),
        ];
        let mut out = [0u8; 4];
        view.fill_batch(MarkerIdx::new(0), &haps, &mut out);
        for (i, hap) in haps.iter().enumerate() {
            assert_eq!(out[i], view.allele(MarkerIdx::new(0), *hap));
        }
    }
}
