//! # Genotype Matrix
//!
//! The main data structure: a matrix of genotypes (markers x haplotypes).
//! Replaces `vcf/RefGT.java`, `vcf/BasicGT.java`, and related classes.
//!
//! ## Type State Pattern
//!
//! The matrix uses a generic `State` parameter to track phasing status at compile time.
//! `GenotypeMatrix<Unphased>` represents unphased data, while `GenotypeMatrix<Phased>`
//! represents phased data. This enables compile-time enforcement of pipeline correctness.

use std::marker::PhantomData;
use std::sync::Arc;

use crate::data::haplotype::{HapIdx, Samples};
use crate::data::marker::{Marker, MarkerIdx, Markers};
use crate::data::storage::phase_state::{PhaseState, Phased, Unphased};
use crate::data::storage::GenotypeColumn;

/// A view into a subset of markers in a GenotypeMatrix
///
/// Provides zero-copy access to a range of markers without cloning the underlying data.
/// Marker indices in the view are relative (0-based), but global positions are preserved.
#[derive(Debug)]
pub struct GenotypeView<'a, State: PhaseState> {
    /// Reference to the full matrix
    matrix: &'a GenotypeMatrix<State>,
    /// Start marker index in the full matrix
    start: usize,
    /// End marker index (exclusive) in the full matrix
    end: usize,
}

impl<'a, State: PhaseState> GenotypeView<'a, State> {
    /// Get the number of markers in this view
    pub fn n_markers(&self) -> usize {
        self.end - self.start
    }

    /// Get the number of haplotypes
    pub fn n_haplotypes(&self) -> usize {
        self.matrix.n_haplotypes()
    }

    /// Get allele for a marker/haplotype pair (relative indices)
    pub fn allele(&self, marker: MarkerIdx, hap: HapIdx) -> u8 {
        let global_marker = MarkerIdx::new((self.start + marker.as_usize()) as u32);
        self.matrix.allele(global_marker, hap)
    }

    /// Get marker metadata (relative indices within this view)
    pub fn marker(&self, marker: MarkerIdx) -> &Marker {
        let global_marker = MarkerIdx::new((self.start + marker.as_usize()) as u32);
        self.matrix.marker(global_marker)
    }

    /// Get genetic distance between two markers within this view
    ///
    /// Returns the genetic distance in cM between view-relative marker indices.
    /// Panics if indices are out of bounds.
    pub fn genetic_distance(&self, marker_a: usize, marker_b: usize) -> f64 {
        assert!(marker_a < self.n_markers() && marker_b < self.n_markers());
        let pos_a = self.marker(MarkerIdx::new(marker_a as u32)).pos_cm;
        let pos_b = self.marker(MarkerIdx::new(marker_b as u32)).pos_cm;
        (pos_a - pos_b).abs()
    }

    /// Get physical distance between two markers within this view
    ///
    /// Returns the physical distance in base pairs between view-relative marker indices.
    /// Panics if indices are out of bounds.
    pub fn physical_distance(&self, marker_a: usize, marker_b: usize) -> u32 {
        assert!(marker_a < self.n_markers() && marker_b < self.n_markers());
        let pos_a = self.marker(MarkerIdx::new(marker_a as u32)).pos;
        let pos_b = self.marker(MarkerIdx::new(marker_b as u32)).pos;
        pos_a.max(pos_b) - pos_a.min(pos_b)
    }

    /// Get the marker offset (index of first marker in the full matrix)
    ///
    /// When using markers() to get metadata, add this offset to convert
    /// from view-relative indices to matrix-global indices.
    pub fn marker_offset(&self) -> usize {
        self.start
    }

    /// Get markers metadata from the full matrix
    ///
    /// WARNING: This returns the full matrix's markers, not a subset for this view.
    /// Only use this when you need access to the full chromosome markers for cross-window operations.
    ///
    /// For view-relative access, use `marker(idx)` instead, which handles index translation automatically.
    /// If you find yourself using `view.markers().len()`, you probably want `view.n_markers()` instead.
    pub fn markers(&self) -> &Markers {
        self.matrix.markers()
    }

    /// Get samples metadata
    pub fn samples(&self) -> &Arc<Samples> {
        self.matrix.samples()
    }
}

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

    /// Optional per-sample genotype confidence scores (from GL or DS).
    /// Stored as u8 (0-255) representing confidence 0.0-1.0.
    /// Layout: `confidence[marker][sample]`
    /// None if no confidence information available (assume full confidence).
    confidence: Option<Vec<Vec<u8>>>,

    /// Phantom data to hold the State type parameter (zero-sized)
    phantom: PhantomData<State>,
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

    /// Get marker by index
    pub fn marker(&self, idx: MarkerIdx) -> &Marker {
        self.markers.marker(idx)
    }

    /// Get all markers
    pub fn markers(&self) -> &Markers {
        &self.markers
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

    /// Total memory usage in bytes (approximate)
    pub fn size_bytes(&self) -> usize {
        let column_bytes: usize = self.columns.iter().map(|c| c.size_bytes()).sum();
        let confidence_bytes: usize = self.confidence.as_ref()
            .map(|c| c.iter().map(|v| v.len()).sum())
            .unwrap_or(0);
        column_bytes + confidence_bytes + std::mem::size_of::<Self>()
    }

    /// Check if confidence scores are available
    pub fn has_confidence(&self) -> bool {
        self.confidence.is_some()
    }

    /// Get confidence score for a sample at a marker (0-255 representing 0.0-1.0).
    /// Returns 255 (full confidence) if confidence data is not available.
    #[inline]
    pub fn sample_confidence(&self, marker: MarkerIdx, sample_idx: usize) -> u8 {
        self.confidence.as_ref()
            .and_then(|c| c.get(marker.as_usize()))
            .and_then(|row| row.get(sample_idx))
            .copied()
            .unwrap_or(255)
    }

    /// Get confidence score as f32 (0.0-1.0)
    #[inline]
    pub fn sample_confidence_f32(&self, marker: MarkerIdx, sample_idx: usize) -> f32 {
        self.sample_confidence(marker, sample_idx) as f32 / 255.0
    }

    /// Clone the confidence data (for transferring to a new matrix)
    pub fn confidence_clone(&self) -> Option<Vec<Vec<u8>>> {
        self.confidence.clone()
    }

    /// Create a zero-copy view of a marker range
    pub fn get_window_view(&self, range: std::ops::Range<usize>) -> GenotypeView<'_, S> {
        assert!(range.start <= range.end && range.end <= self.n_markers());
        GenotypeView {
            matrix: self,
            start: range.start,
            end: range.end,
        }
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
            confidence: None,
            phantom: PhantomData,
        }
    }

    /// Create new unphased matrix with confidence scores
    pub fn new_unphased_with_confidence(
        markers: Markers,
        columns: Vec<GenotypeColumn>,
        samples: Arc<Samples>,
        confidence: Vec<Vec<u8>>,
    ) -> Self {
        debug_assert_eq!(markers.len(), columns.len());
        debug_assert_eq!(markers.len(), confidence.len());
        Self {
            markers,
            columns,
            samples,
            is_reversed: false,
            confidence: Some(confidence),
            phantom: PhantomData,
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
            confidence: self.confidence,
            phantom: PhantomData,
        }
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
            confidence: None,
            phantom: PhantomData,
        }
    }

    /// Create a new phased genotype matrix with confidence scores
    pub fn new_phased_with_confidence(
        markers: Markers,
        columns: Vec<GenotypeColumn>,
        samples: Arc<Samples>,
        confidence: Vec<Vec<u8>>,
    ) -> Self {
        debug_assert_eq!(markers.len(), columns.len());
        debug_assert_eq!(markers.len(), confidence.len());
        Self {
            markers,
            columns,
            samples,
            is_reversed: false,
            confidence: Some(confidence),
            phantom: PhantomData,
        }
    }

    /// Get a reference as unphased (zero-cost, same memory layout)
    pub fn as_unphased_ref(&self) -> &GenotypeMatrix<Unphased> {
        // SAFETY: GenotypeMatrix<Phased> and GenotypeMatrix<Unphased> have identical
        // memory layouts (PhantomData is zero-sized), differing only in the type parameter
        unsafe { &*(self as *const GenotypeMatrix<Phased> as *const GenotypeMatrix<Unphased>) }
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
    fn test_phase_transition() {
        let unphased = make_test_matrix_unphased();

        // Transform to phased
        let phased = unphased.into_phased();
        assert_eq!(phased.n_markers(), 2);
    }

    #[test]
    fn test_confidence_scores() {
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

        // Create confidence scores: marker 0 has full confidence, marker 1 has 50% for sample 0
        let confidence = vec![
            vec![255, 255],         // marker 0: full confidence for both samples
            vec![128, 255],         // marker 1: 50% for sample 0, full for sample 1
        ];

        let matrix = GenotypeMatrix::new_unphased_with_confidence(
            markers,
            vec![col1, col2],
            samples,
            confidence,
        );

        assert!(matrix.has_confidence());
        assert_eq!(matrix.sample_confidence(MarkerIdx::new(0), 0), 255);
        assert_eq!(matrix.sample_confidence(MarkerIdx::new(1), 0), 128);
        assert_eq!(matrix.sample_confidence(MarkerIdx::new(1), 1), 255);

        // Check f32 conversion
        assert!((matrix.sample_confidence_f32(MarkerIdx::new(0), 0) - 1.0).abs() < 0.01);
        assert!((matrix.sample_confidence_f32(MarkerIdx::new(1), 0) - 0.502).abs() < 0.01);

        // Verify confidence survives phase transition
        let phased = matrix.into_phased();
        assert!(phased.has_confidence());
        assert_eq!(phased.sample_confidence(MarkerIdx::new(1), 0), 128);
    }

    #[test]
    fn test_no_confidence_defaults_to_full() {
        let matrix = make_test_matrix_unphased();

        // Without confidence data, has_confidence returns false
        assert!(!matrix.has_confidence());

        // But sample_confidence defaults to 255 (full confidence)
        assert_eq!(matrix.sample_confidence(MarkerIdx::new(0), 0), 255);
        assert!((matrix.sample_confidence_f32(MarkerIdx::new(0), 0) - 1.0).abs() < 0.01);
    }
}
