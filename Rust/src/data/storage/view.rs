//! # Genotype View
//!
//! A read-only view over different genotype storage types.
//! This provides a unified interface for algorithms (like HMM)
//! to operate on both immutable `GenotypeMatrix` and `MutableGenotypes`.

use crate::data::haplotype::HapIdx;
use crate::data::marker::{Marker, MarkerIdx, Markers};
use crate::data::storage::{GenotypeMatrix, MutableGenotypes, phase_state};
use crate::pipelines::imputation::MarkerAlignment;

/// A read-only view of genotype data - allows HMM to work with either
/// GenotypeMatrix or MutableGenotypes without caring about concrete type
#[derive(Clone, Copy)]
pub enum GenotypeView<'a> {
    /// View over an immutable GenotypeMatrix
    Matrix(&'a GenotypeMatrix),
    /// View over MutableGenotypes with associated markers
    Mutable {
        geno: &'a MutableGenotypes,
        markers: &'a Markers,
    },
    /// View over a subset of markers in a Matrix
    MatrixSubset {
        matrix: &'a GenotypeMatrix,
        subset: &'a [usize],
    },
    /// View over a subset of markers in MutableGenotypes
    MutableSubset {
        geno: &'a MutableGenotypes,
        markers: &'a Markers,
        subset: &'a [usize],
    },
    /// Composite view: Target (mutable) + Reference (immutable)
    /// Haplotype indices 0..n_target_haps -> target, n_target_haps..n_total -> reference
    Composite {
        target: &'a MutableGenotypes,
        reference: &'a GenotypeMatrix<phase_state::Phased>,
        alignment: &'a MarkerAlignment,
        markers: &'a Markers,
        n_target_haps: usize,
    },
    /// Composite view over a marker subset (for Stage 1 hi-freq markers)
    /// Combines target + reference with marker subset mapping
    CompositeSubset {
        target: &'a MutableGenotypes,
        reference: &'a GenotypeMatrix<phase_state::Phased>,
        alignment: &'a MarkerAlignment,
        markers: &'a Markers,
        subset: &'a [usize],
        n_target_haps: usize,
    },
}

impl<'a> GenotypeView<'a> {
    /// Get the number of markers
    #[inline]
    pub fn n_markers(&self) -> usize {
        match self {
            GenotypeView::Matrix(m) => m.n_markers(),
            GenotypeView::Mutable { geno, .. } => geno.n_markers(),
            GenotypeView::MatrixSubset { subset, .. } => subset.len(),
            GenotypeView::MutableSubset { subset, .. } => subset.len(),
            GenotypeView::Composite { target, .. } => target.n_markers(),
            GenotypeView::CompositeSubset { subset, .. } => subset.len(),
        }
    }

    /// Get the number of haplotypes
    #[inline]
    pub fn n_haplotypes(&self) -> usize {
        match self {
            GenotypeView::Matrix(m) => m.n_haplotypes(),
            GenotypeView::Mutable { geno, .. } => geno.n_haps(),
            GenotypeView::MatrixSubset { matrix, .. } => matrix.n_haplotypes(),
            GenotypeView::MutableSubset { geno, .. } => geno.n_haps(),
            GenotypeView::Composite { target, reference, .. } => {
                target.n_haps() + reference.n_haplotypes()
            }
            GenotypeView::CompositeSubset { target, reference, .. } => {
                target.n_haps() + reference.n_haplotypes()
            }
        }
    }

    /// Get the number of samples
    #[inline]
    pub fn n_samples(&self) -> usize {
        match self {
            GenotypeView::Matrix(m) => m.n_samples(),
            GenotypeView::Mutable { geno, .. } => geno.n_samples(),
            GenotypeView::MatrixSubset { matrix, .. } => matrix.n_samples(),
            GenotypeView::MutableSubset { geno, .. } => geno.n_samples(),
            // For composite views, return target samples (we're phasing the target)
            GenotypeView::Composite { target, .. } => target.n_samples(),
            GenotypeView::CompositeSubset { target, .. } => target.n_samples(),
        }
    }

    /// Get an allele at a specific marker and haplotype index
    #[inline]
    pub fn allele(&self, marker: MarkerIdx, hap: HapIdx) -> u8 {
        match self {
            GenotypeView::Matrix(m) => m.allele(marker, hap),
            GenotypeView::Mutable { geno, .. } => geno.get(marker.as_usize(), hap),
            GenotypeView::MatrixSubset { matrix, subset } => {
                let real_idx = subset[marker.as_usize()];
                matrix.allele(MarkerIdx::new(real_idx as u32), hap)
            }
            GenotypeView::MutableSubset { geno, subset, .. } => {
                let real_idx = subset[marker.as_usize()];
                geno.get(real_idx, hap)
            }
            GenotypeView::Composite { target, reference, alignment, n_target_haps, .. } => {
                let hap_idx = hap.as_usize();
                if hap_idx < *n_target_haps {
                    // Target haplotype - direct lookup
                    target.get(marker.as_usize(), hap)
                } else {
                    // Reference haplotype - translate marker index, look up, and map allele to target encoding
                    let ref_hap = hap_idx - n_target_haps;
                    let target_marker = marker.as_usize();
                    if let Some(ref_m) = alignment.target_to_ref(target_marker) {
                        let ref_allele = reference.allele(MarkerIdx::new(ref_m as u32), HapIdx::new(ref_hap as u32));
                        // Map reference allele back to target encoding (handles strand flips)
                        alignment.reverse_map_allele(target_marker, ref_allele)
                    } else {
                        255 // Marker not in reference - return missing
                    }
                }
            }
            GenotypeView::CompositeSubset { target, reference, alignment, subset, n_target_haps, .. } => {
                let orig_marker = subset[marker.as_usize()]; // Subset index -> original target marker index
                let hap_idx = hap.as_usize();
                if hap_idx < *n_target_haps {
                    // Target haplotype - direct lookup using original marker index
                    target.get(orig_marker, hap)
                } else {
                    // Reference haplotype - translate marker, look up, and map allele to target encoding
                    let ref_hap = hap_idx - n_target_haps;
                    if let Some(ref_m) = alignment.target_to_ref(orig_marker) {
                        let ref_allele = reference.allele(MarkerIdx::new(ref_m as u32), HapIdx::new(ref_hap as u32));
                        // Map reference allele back to target encoding (handles strand flips)
                        alignment.reverse_map_allele(orig_marker, ref_allele)
                    } else {
                        255 // Marker not in reference - return missing
                    }
                }
            }
        }
    }

    /// Get marker metadata by index
    #[inline]
    pub fn marker(&self, marker: MarkerIdx) -> &Marker {
        match self {
            GenotypeView::Matrix(m) => m.marker(marker),
            GenotypeView::Mutable { markers, .. } => markers.marker(marker),
            GenotypeView::MatrixSubset { matrix, subset } => {
                let real_idx = subset[marker.as_usize()];
                matrix.marker(MarkerIdx::new(real_idx as u32))
            }
            GenotypeView::MutableSubset { markers, subset, .. } => {
                let real_idx = subset[marker.as_usize()];
                markers.marker(MarkerIdx::new(real_idx as u32))
            }
            GenotypeView::Composite { markers, .. } => markers.marker(marker),
            GenotypeView::CompositeSubset { markers, subset, .. } => {
                let real_idx = subset[marker.as_usize()];
                markers.marker(MarkerIdx::new(real_idx as u32))
            }
        }
    }
}

/// Conversion from `&GenotypeMatrix` (Unphased) to `GenotypeView`
impl<'a> From<&'a GenotypeMatrix> for GenotypeView<'a> {
    fn from(matrix: &'a GenotypeMatrix) -> Self {
        GenotypeView::Matrix(matrix)
    }
}

/// Conversion from `&GenotypeMatrix<Phased>` to `GenotypeView`
impl<'a> From<&'a GenotypeMatrix<phase_state::Phased>> for GenotypeView<'a> {
    fn from(matrix: &'a GenotypeMatrix<phase_state::Phased>) -> Self {
        GenotypeView::Matrix(matrix.as_unphased_ref())
    }
}

/// Conversion from `(&'a MutableGenotypes, &'a Markers)` to `GenotypeView`
impl<'a> From<(&'a MutableGenotypes, &'a Markers)> for GenotypeView<'a> {
    fn from((geno, markers): (&'a MutableGenotypes, &'a Markers)) -> Self {
        GenotypeView::Mutable { geno, markers }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::ChromIdx;
    use crate::data::haplotype::Samples;
    use crate::data::marker::{Allele, Marker};
    use crate::data::storage::phase_state::Phased;
    use crate::data::storage::GenotypeColumn;
    use std::sync::Arc;

    fn make_test_matrix() -> GenotypeMatrix<Phased> {
        let samples = Arc::new(Samples::from_ids(vec!["S1".to_string(), "S2".to_string()]));
        let mut markers = Markers::new();
        markers.add_chrom("chr1");
        markers.push(Marker::new(
            ChromIdx::new(0),
            100,
            None,
            Allele::Base(0),
            vec![Allele::Base(1)],
        ));
        let col = GenotypeColumn::from_alleles(&[0, 1, 0, 1], 2);
        GenotypeMatrix::new_phased(markers, vec![col], samples)
    }

    fn make_test_mutable() -> (MutableGenotypes, Markers) {
        let mut markers = Markers::new();
        markers.add_chrom("chr1");
        markers.push(Marker::new(
            ChromIdx::new(0),
            200,
            None,
            Allele::Base(1),
            vec![Allele::Base(0)],
        ));
        let geno = MutableGenotypes::from_fn(1, 2, |_, h| h as u8);
        (geno, markers)
    }

    #[test]
    fn test_view_from_matrix() {
        let matrix = make_test_matrix();
        let view = GenotypeView::from(&matrix);

        assert_eq!(view.n_markers(), 1);
        assert_eq!(view.n_haplotypes(), 4);
        assert_eq!(view.n_samples(), 2);
        assert_eq!(view.allele(MarkerIdx::new(0), HapIdx::new(1)), 1);
        assert_eq!(view.marker(MarkerIdx::new(0)).pos, 100);
    }

    #[test]
    fn test_view_from_mutable() {
        let (geno, markers) = make_test_mutable();
        let view = GenotypeView::from((&geno, &markers));

        assert_eq!(view.n_markers(), 1);
        assert_eq!(view.n_haplotypes(), 2);
        assert_eq!(view.n_samples(), 1);
        assert_eq!(view.allele(MarkerIdx::new(0), HapIdx::new(1)), 1);
        assert_eq!(view.marker(MarkerIdx::new(0)).pos, 200);
    }
}
