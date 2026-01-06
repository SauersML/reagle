//! # Genotype View
//!
//! A read-only view over different genotype storage types.
//! This provides a unified interface for algorithms (like HMM)
//! to operate on both immutable `GenotypeMatrix` and `MutableGenotypes`.

use crate::data::haplotype::HapIdx;
use crate::data::marker::{Marker, MarkerIdx, Markers};
use crate::data::storage::{GenotypeMatrix, MutableGenotypes};

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
}

impl<'a> GenotypeView<'a> {
    /// Get the number of markers
    #[inline]
    pub fn n_markers(&self) -> usize {
        match self {
            GenotypeView::Matrix(m) => m.n_markers(),
            GenotypeView::Mutable { geno, .. } => geno.n_markers(),
        }
    }

    /// Get the number of haplotypes
    #[inline]
    pub fn n_haplotypes(&self) -> usize {
        match self {
            GenotypeView::Matrix(m) => m.n_haplotypes(),
            GenotypeView::Mutable { geno, .. } => geno.n_haps(),
        }
    }

    /// Get the number of samples
    #[inline]
    pub fn n_samples(&self) -> usize {
        match self {
            GenotypeView::Matrix(m) => m.n_samples(),
            GenotypeView::Mutable { geno, .. } => geno.n_samples(),
        }
    }

    /// Get an allele at a specific marker and haplotype index
    #[inline]
    pub fn allele(&self, marker: MarkerIdx, hap: HapIdx) -> u8 {
        match self {
            GenotypeView::Matrix(m) => m.allele(marker, hap),
            GenotypeView::Mutable { geno, .. } => geno.get(marker.as_usize(), hap),
        }
    }

    /// Get marker metadata by index
    #[inline]
    pub fn marker(&self, marker: MarkerIdx) -> &Marker {
        match self {
            GenotypeView::Matrix(m) => m.marker(marker),
            GenotypeView::Mutable { markers, .. } => markers.marker(marker),
        }
    }

    /// Get the markers collection
    #[inline]
    pub fn markers(&self) -> &Markers {
        match self {
            GenotypeView::Matrix(m) => m.markers(),
            GenotypeView::Mutable { markers, .. } => markers,
        }
    }
}

/// Conversion from `&GenotypeMatrix` to `GenotypeView`
impl<'a> From<&'a GenotypeMatrix> for GenotypeView<'a> {
    fn from(matrix: &'a GenotypeMatrix) -> Self {
        GenotypeView::Matrix(matrix)
    }
}

/// Conversion from `(&'a MutableGenotypes, &'a Markers)` to `GenotypeView`
impl<'a> From<(&'a MutableGenotypes, &'a Markers)> for GenotypeView<'a> {
    fn from((geno, markers): (&'a MutableGenotypes, &'a Markers)) -> Self {
        GenotypeView::Mutable { geno, markers }
    }
}

impl<'a> GenotypeView<'a> {
    /// Create a view over mutable genotypes with associated markers
    pub fn from_mutable(geno: &'a MutableGenotypes, markers: &'a Markers) -> Self {
        GenotypeView::Mutable { geno, markers }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::ChromIdx;
    use crate::data::haplotype::Samples;
    use crate::data::marker::{Allele, Marker};
    use crate::data::storage::GenotypeColumn;
    use std::sync::Arc;

    fn make_test_matrix() -> GenotypeMatrix {
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
        GenotypeMatrix::new(markers, vec![col], samples, true)
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
