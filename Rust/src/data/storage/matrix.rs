//! # Genotype Matrix
//!
//! The main data structure: a matrix of genotypes (markers x haplotypes).
//! Replaces `vcf/RefGT.java`, `vcf/BasicGT.java`, and related classes.
//!
//! ## Type State Pattern
//!
//! The matrix uses a generic `State` parameter to track phasing status at compile time:
//!
//! ```ignore
//! // Unphased data from VCF reader
//! let unphased: GenotypeMatrix<Unphased> = vcf_reader.read()?;
//!
//! // Phasing transforms to phased type
//! let phased: GenotypeMatrix<Phased> = phasing_pipeline.run(unphased)?;
//!
//! // Imputation requires phased - enforced at compile time!
//! imputation_pipeline.run(&phased)?;
//! ```

use std::marker::PhantomData;
use std::sync::Arc;

use crate::data::haplotype::{HapIdx, SampleIdx, Samples};
use crate::data::marker::{Marker, MarkerIdx, Markers};
use crate::data::storage::phase_state::{PhaseState, Phased, Unphased};
use crate::data::storage::GenotypeColumn;

/// The main genotype matrix structure.
///
/// Type parameter `State` encodes whether data is phased at compile time,
/// enabling the compiler to enforce correct pipeline usage.
#[derive(Clone, Debug)]
pub struct GenotypeMatrix<State: PhaseState = Unphased> {
    /// Marker metadata
    markers: Markers,

    /// Genotype data (one column per marker)
    columns: Vec<GenotypeColumn>,

    /// Sample metadata
    samples: Arc<Samples>,

    /// Whether markers are in reverse order
    is_reversed: bool,

    /// Phantom data to hold the State type parameter (zero-sized)
    _state: PhantomData<State>,
}

// ============================================================================
// Methods available for ALL phase states
// ============================================================================

impl<S: PhaseState> GenotypeMatrix<S> {
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

    /// Check if data is phased (compile-time constant)
    pub const fn is_phased() -> bool {
        S::IS_PHASED
    }

    /// Get marker by index
    pub fn marker(&self, idx: MarkerIdx) -> &Marker {
        self.markers.marker(idx)
    }

    /// Get all markers
    pub fn markers(&self) -> &Markers {
        &self.markers
    }

    /// Get samples reference
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

    /// Get all columns
    pub fn columns(&self) -> &[GenotypeColumn] {
        &self.columns
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

    /// Restrict to a range of markers (preserves phase state)
    pub fn restrict(&self, start: usize, end: usize) -> Self {
        Self {
            markers: self.markers.restrict(start, end),
            columns: self.columns[start..end].to_vec(),
            samples: Arc::clone(&self.samples),
            is_reversed: self.is_reversed,
            _state: PhantomData,
        }
    }

    /// Total memory usage in bytes (approximate)
    pub fn size_bytes(&self) -> usize {
        let column_bytes: usize = self.columns.iter().map(|c| c.size_bytes()).sum();
        column_bytes + std::mem::size_of::<Self>()
    }

    /// Whether markers are in reverse order
    pub fn is_reversed(&self) -> bool {
        self.is_reversed
    }
}

// ============================================================================
// Methods ONLY for Unphased matrices
// ============================================================================

impl GenotypeMatrix<Unphased> {
    /// Create a new unphased genotype matrix
    pub fn new_unphased(
        markers: Markers,
        columns: Vec<GenotypeColumn>,
        samples: Arc<Samples>,
    ) -> Self {
        debug_assert_eq!(markers.len(), columns.len());
        Self {
            markers,
            columns,
            samples,
            is_reversed: false,
            _state: PhantomData,
        }
    }

    /// Transform into a phased matrix.
    ///
    /// This is the primary way to create a `GenotypeMatrix<Phased>`.
    /// Consumes self to prevent accidental use of unphased data.
    pub fn into_phased(self) -> GenotypeMatrix<Phased> {
        GenotypeMatrix {
            markers: self.markers,
            columns: self.columns,
            samples: self.samples,
            is_reversed: self.is_reversed,
            _state: PhantomData,
        }
    }

    /// Legacy constructor for backwards compatibility.
    ///
    /// Creates an unphased matrix if `is_phased` is false,
    /// or a phased matrix wrapped in an enum if true.
    #[deprecated(
        since = "0.2.0",
        note = "Use new_unphased() or new_phased() for type-safe construction"
    )]
    pub fn new(
        markers: Markers,
        columns: Vec<GenotypeColumn>,
        samples: Arc<Samples>,
        _is_phased: bool,
    ) -> Self {
        Self::new_unphased(markers, columns, samples)
    }
}

// ============================================================================
// Methods ONLY for Phased matrices
// ============================================================================

impl GenotypeMatrix<Phased> {
    /// Create a new phased genotype matrix
    pub fn new_phased(
        markers: Markers,
        columns: Vec<GenotypeColumn>,
        samples: Arc<Samples>,
    ) -> Self {
        debug_assert_eq!(markers.len(), columns.len());
        Self {
            markers,
            columns,
            samples,
            is_reversed: false,
            _state: PhantomData,
        }
    }

    /// Convert to unphased (for algorithms that don't need phase)
    pub fn into_unphased(self) -> GenotypeMatrix<Unphased> {
        GenotypeMatrix {
            markers: self.markers,
            columns: self.columns,
            samples: self.samples,
            is_reversed: self.is_reversed,
            _state: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::ChromIdx;
    use crate::data::marker::Allele;

    fn make_test_matrix_phased() -> GenotypeMatrix<Phased> {
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

        GenotypeMatrix::new_phased(markers, vec![col1, col2], samples)
    }

    fn make_test_matrix_unphased() -> GenotypeMatrix<Unphased> {
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

        GenotypeMatrix::new_unphased(markers, vec![col1, col2], samples)
    }

    #[test]
    fn test_matrix_access() {
        let matrix = make_test_matrix_phased();

        assert_eq!(matrix.n_markers(), 2);
        assert_eq!(matrix.n_samples(), 2);
        assert_eq!(matrix.n_haplotypes(), 4);

        assert_eq!(matrix.allele(MarkerIdx::new(0), HapIdx::new(0)), 0);
        assert_eq!(matrix.allele(MarkerIdx::new(0), HapIdx::new(1)), 1);
        assert_eq!(matrix.allele(MarkerIdx::new(1), HapIdx::new(0)), 1);
    }

    #[test]
    fn test_matrix_restrict() {
        let matrix = make_test_matrix_phased();
        let restricted = matrix.restrict(0, 1);

        assert_eq!(restricted.n_markers(), 1);
        assert_eq!(restricted.n_haplotypes(), 4);
    }

    #[test]
    fn test_genotype() {
        let matrix = make_test_matrix_phased();
        let (a1, a2) = matrix.genotype(MarkerIdx::new(0), SampleIdx::new(0));
        assert_eq!(a1, 0);
        assert_eq!(a2, 1);
    }

    #[test]
    fn test_phase_state_compile_time() {
        // Verify phase state is known at compile time
        assert!(GenotypeMatrix::<Phased>::is_phased());
        assert!(!GenotypeMatrix::<Unphased>::is_phased());
    }

    #[test]
    fn test_phase_transition() {
        let unphased = make_test_matrix_unphased();
        assert!(!GenotypeMatrix::<Unphased>::is_phased());

        // Transform to phased
        let phased = unphased.into_phased();
        assert!(GenotypeMatrix::<Phased>::is_phased());
        assert_eq!(phased.n_markers(), 2);
    }
}
