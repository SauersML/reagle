//! # Genotype Matrix
//!
//! The main data structure: a matrix of genotypes (markers x haplotypes).
//! Replaces `vcf/RefGT.java`, `vcf/BasicGT.java`, and related classes.

use std::sync::Arc;

use crate::data::haplotype::{HapIdx, SampleIdx, Samples};
use crate::data::marker::{Marker, MarkerIdx, Markers};
use crate::data::storage::GenotypeColumn;

/// The main genotype matrix structure
#[derive(Clone, Debug)]
pub struct GenotypeMatrix {
    /// Marker metadata
    markers: Markers,

    /// Genotype data (one column per marker)
    columns: Vec<GenotypeColumn>,

    /// Sample metadata
    samples: Arc<Samples>,

    /// Whether the data is phased
    is_phased: bool,

    /// Whether markers are in reverse order
    is_reversed: bool,
}

impl GenotypeMatrix {
    /// Create a new genotype matrix
    pub fn new(
        markers: Markers,
        columns: Vec<GenotypeColumn>,
        samples: Arc<Samples>,
        is_phased: bool,
    ) -> Self {
        debug_assert_eq!(markers.len(), columns.len());
        Self {
            markers,
            columns,
            samples,
            is_phased,
            is_reversed: false,
        }
    }

    /// Create an empty matrix
    pub fn empty(samples: Arc<Samples>) -> Self {
        Self {
            markers: Markers::new(),
            columns: Vec::new(),
            samples,
            is_phased: true,
            is_reversed: false,
        }
    }

    /// Number of markers
    pub fn n_markers(&self) -> usize {
        self.markers.len()
    }

    /// Number of samples
    pub fn n_samples(&self) -> usize {
        self.samples.len()
    }

    /// Number of haplotypes
    pub fn n_haplotypes(&self) -> usize {
        self.samples.n_haps()
    }

    /// Check if data is phased
    pub fn is_phased(&self) -> bool {
        self.is_phased
    }

    /// Check if markers are in reverse order
    pub fn is_reversed(&self) -> bool {
        self.is_reversed
    }

    /// Get marker by index
    pub fn marker(&self, idx: MarkerIdx) -> &Marker {
        self.markers.marker(idx)
    }

    /// Get all markers
    pub fn markers(&self) -> &Markers {
        &self.markers
    }

    /// Get samples
    pub fn samples(&self) -> &Samples {
        &self.samples
    }

    /// Get samples Arc
    pub fn samples_arc(&self) -> Arc<Samples> {
        Arc::clone(&self.samples)
    }

    /// Get genotype column for a marker
    pub fn column(&self, idx: MarkerIdx) -> &GenotypeColumn {
        &self.columns[idx.as_usize()]
    }

    /// Get allele at (marker, haplotype)
    #[inline]
    pub fn allele(&self, marker: MarkerIdx, hap: HapIdx) -> u8 {
        self.columns[marker.as_usize()].get(hap)
    }

    /// Get both alleles for a sample at a marker (for diploid)
    pub fn genotype(&self, marker: MarkerIdx, sample: SampleIdx) -> (u8, u8) {
        let hap1 = sample.hap1();
        let hap2 = sample.hap2();
        (self.allele(marker, hap1), self.allele(marker, hap2))
    }

    /// Restrict to a range of markers
    pub fn restrict(&self, start: usize, end: usize) -> Self {
        Self {
            markers: self.markers.restrict(start, end),
            columns: self.columns[start..end].to_vec(),
            samples: Arc::clone(&self.samples),
            is_phased: self.is_phased,
            is_reversed: self.is_reversed,
        }
    }

    /// Restrict to specific marker indices
    pub fn restrict_markers(&self, indices: &[usize]) -> Self {
        let markers = Markers::from_vec(
            indices
                .iter()
                .map(|&i| self.markers.marker(MarkerIdx::new(i as u32)).clone())
                .collect(),
            self.markers.chrom_names().to_vec(),
        );
        let columns = indices.iter().map(|&i| self.columns[i].clone()).collect();

        Self {
            markers,
            columns,
            samples: Arc::clone(&self.samples),
            is_phased: self.is_phased,
            is_reversed: self.is_reversed,
        }
    }

    /// Iterate over markers with their columns
    pub fn iter(&self) -> impl Iterator<Item = (MarkerIdx, &Marker, &GenotypeColumn)> {
        self.markers
            .iter()
            .enumerate()
            .map(move |(i, m)| (MarkerIdx::new(i as u32), m, &self.columns[i]))
    }

    /// Iterate over marker indices
    pub fn marker_indices(&self) -> impl Iterator<Item = MarkerIdx> {
        (0..self.n_markers()).map(|i| MarkerIdx::new(i as u32))
    }

    /// Iterate over haplotype indices
    pub fn haplotype_indices(&self) -> impl Iterator<Item = HapIdx> {
        (0..self.n_haplotypes()).map(|i| HapIdx::new(i as u32))
    }

    /// Get alleles for a haplotype across all markers
    pub fn haplotype(&self, hap: HapIdx) -> Vec<u8> {
        self.columns.iter().map(|col| col.get(hap)).collect()
    }

    /// Get alleles for a marker across all haplotypes
    pub fn alleles_at_marker(&self, marker: MarkerIdx) -> Vec<u8> {
        let col = &self.columns[marker.as_usize()];
        (0..self.n_haplotypes())
            .map(|h| col.get(HapIdx::new(h as u32)))
            .collect()
    }

    /// Set the phased flag
    pub fn set_phased(&mut self, phased: bool) {
        self.is_phased = phased;
    }

    /// Set the reversed flag
    pub fn set_reversed(&mut self, reversed: bool) {
        self.is_reversed = reversed;
    }

    /// Total memory usage in bytes (approximate)
    pub fn size_bytes(&self) -> usize {
        let column_bytes: usize = self.columns.iter().map(|c| c.size_bytes()).sum();
        column_bytes + std::mem::size_of::<Self>()
    }

    /// Get column mutably (for phasing updates)
    pub fn column_mut(&mut self, idx: MarkerIdx) -> &mut GenotypeColumn {
        &mut self.columns[idx.as_usize()]
    }

    /// Replace a column
    pub fn set_column(&mut self, idx: MarkerIdx, column: GenotypeColumn) {
        self.columns[idx.as_usize()] = column;
    }

    /// Add a marker and its genotype column
    pub fn push(&mut self, marker: Marker, column: GenotypeColumn) {
        self.markers.push(marker);
        self.columns.push(column);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::ChromIdx;
    use crate::data::marker::Allele;

    fn make_test_matrix() -> GenotypeMatrix {
        let samples = Arc::new(Samples::from_ids(vec!["S1".to_string(), "S2".to_string()]));
        let mut markers = Markers::new();
        markers.add_chrom("chr1");

        let m1 = Marker::new(
            ChromIdx::new(0),
            100,
            None,
            Allele::Base(0),
            vec![Allele::Base(1)],
        );
        let m2 = Marker::new(
            ChromIdx::new(0),
            200,
            None,
            Allele::Base(0),
            vec![Allele::Base(1)],
        );

        markers.push(m1);
        markers.push(m2);

        let col1 = GenotypeColumn::from_alleles(&[0, 1, 0, 1], 2);
        let col2 = GenotypeColumn::from_alleles(&[1, 1, 0, 0], 2);

        GenotypeMatrix::new(markers, vec![col1, col2], samples, true)
    }

    #[test]
    fn test_matrix_access() {
        let matrix = make_test_matrix();

        assert_eq!(matrix.n_markers(), 2);
        assert_eq!(matrix.n_samples(), 2);
        assert_eq!(matrix.n_haplotypes(), 4);

        assert_eq!(matrix.allele(MarkerIdx::new(0), HapIdx::new(0)), 0);
        assert_eq!(matrix.allele(MarkerIdx::new(0), HapIdx::new(1)), 1);
        assert_eq!(matrix.allele(MarkerIdx::new(1), HapIdx::new(0)), 1);
    }

    #[test]
    fn test_matrix_restrict() {
        let matrix = make_test_matrix();
        let restricted = matrix.restrict(0, 1);

        assert_eq!(restricted.n_markers(), 1);
        assert_eq!(restricted.n_haplotypes(), 4);
    }

    #[test]
    fn test_genotype() {
        let matrix = make_test_matrix();
        let (a1, a2) = matrix.genotype(MarkerIdx::new(0), SampleIdx::new(0));
        assert_eq!(a1, 0);
        assert_eq!(a2, 1);
    }
}
