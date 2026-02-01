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

use crate::data::haplotype::{HapIdx, SampleIdx, Samples};
use crate::data::marker::{AnyMarkerSpace, Marker, MarkerIdx, Markers};
use crate::data::storage::GenotypeColumn;
use crate::data::storage::phase_state::{PhaseState, Phased, Unphased};

#[derive(Clone, Debug)]
pub struct PlMatrix {
    n_samples: usize,
    marker_offsets: Vec<u32>,
    marker_strides: Vec<u16>,
    values: Vec<u16>,
}

impl PlMatrix {
    const MISSING: u16 = u16::MAX;

    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    pub fn n_markers(&self) -> usize {
        self.marker_strides.len()
    }

    #[inline]
    pub fn sample_pl(&self, marker: usize, sample_idx: usize) -> Option<&[u16]> {
        if marker >= self.marker_strides.len() || sample_idx >= self.n_samples {
            return None;
        }
        let stride = self.marker_strides[marker] as usize;
        if stride == 0 {
            return None;
        }
        let base = self.marker_offsets[marker] as usize;
        let start = base + sample_idx * stride;
        let end = start + stride;
        let slice = self.values.get(start..end)?;
        if slice.iter().all(|&v| v == Self::MISSING) {
            None
        } else {
            Some(slice)
        }
    }

    pub fn from_marker_blocks(
        n_samples: usize,
        marker_strides: Vec<u16>,
        mut marker_blocks: Vec<Vec<u16>>,
    ) -> Self {
        debug_assert_eq!(marker_strides.len(), marker_blocks.len());

        let mut marker_offsets: Vec<u32> = Vec::with_capacity(marker_strides.len() + 1);
        marker_offsets.push(0);
        let mut values: Vec<u16> = Vec::new();
        let mut running: u32 = 0;
        for (stride_u16, mut block) in marker_strides.iter().copied().zip(marker_blocks.iter_mut())
        {
            let stride = stride_u16 as usize;
            if stride == 0 {
                marker_offsets.push(running);
                continue;
            }
            debug_assert_eq!(block.len(), stride * n_samples);
            values.append(&mut block);
            running = running.saturating_add((stride * n_samples) as u32);
            marker_offsets.push(running);
        }
        let out = Self {
            n_samples,
            marker_offsets,
            marker_strides,
            values,
        };
        debug_assert_eq!(out.n_samples(), n_samples);
        debug_assert_eq!(out.n_markers(), out.marker_strides.len());
        out
    }
}

/// The main genotype matrix structure.
///
/// Type parameter `State` encodes whether data is phased at compile time,
/// enabling the compiler to enforce correct pipeline usage.
#[derive(Debug)]
pub struct GenotypeMatrix<State: PhaseState = Unphased, Space = AnyMarkerSpace> {
    /// Marker metadata
    markers: Markers<Space>,

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

    /// Optional per-sample phase confidence scores (0-255 => 0.0-1.0).
    /// Represents confidence in the current phased orientation at heterozygotes.
    /// Layout: `phase_confidence[marker][sample]`
    phase_confidence: Option<Vec<Vec<u8>>>,

    /// Optional per-sample phasedness mask (1 = phased, 0 = unphased or missing).
    /// Layout: `phase_mask[marker][sample]`
    phase_mask: Option<Vec<Vec<u8>>>,

    likelihoods_pl: Option<Arc<PlMatrix>>,

    /// Phantom data to hold the State type parameter (zero-sized)
    phantom: PhantomData<State>,
}

impl<State: PhaseState, Space> Clone for GenotypeMatrix<State, Space> {
    fn clone(&self) -> Self {
        Self {
            markers: self.markers.clone(),
            columns: self.columns.clone(),
            samples: Arc::clone(&self.samples),
            is_reversed: self.is_reversed,
            confidence: self.confidence.clone(),
            phase_confidence: self.phase_confidence.clone(),
            phase_mask: self.phase_mask.clone(),
            likelihoods_pl: self.likelihoods_pl.clone(),
            phantom: PhantomData,
        }
    }
}

// ============================================================================
// Methods available for ALL phase states
// ============================================================================

impl<S: PhaseState, Space> GenotypeMatrix<S, Space> {
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
    pub fn marker(&self, idx: MarkerIdx<Space>) -> &Marker {
        self.markers.marker(idx)
    }

    /// Get all markers
    pub fn markers(&self) -> &Markers<Space> {
        &self.markers
    }

    /// Get samples reference

    /// Get samples Arc (cloned)
    pub fn samples_arc(&self) -> Arc<Samples> {
        Arc::clone(&self.samples)
    }

    /// Get genotype column for a marker
    pub fn column(&self, idx: MarkerIdx<Space>) -> &GenotypeColumn {
        &self.columns[idx.as_usize()]
    }

    /// Get all genotype columns

    /// Get allele at (marker, haplotype)
    #[inline]
    pub fn allele(&self, marker: MarkerIdx<Space>, hap: HapIdx) -> u8 {
        self.columns[marker.as_usize()].get(hap)
    }

    /// Total memory usage in bytes (approximate)
    pub fn size_bytes(&self) -> usize {
        let column_bytes: usize = self.columns.iter().map(|c| c.size_bytes()).sum();
        let confidence_bytes: usize = self
            .confidence
            .as_ref()
            .map(|c| c.iter().map(|v| v.len()).sum())
            .unwrap_or(0);
        let phase_confidence_bytes: usize = self
            .phase_confidence
            .as_ref()
            .map(|c| c.iter().map(|v| v.len()).sum())
            .unwrap_or(0);
        let phase_mask_bytes: usize = self
            .phase_mask
            .as_ref()
            .map(|c| c.iter().map(|v| v.len()).sum())
            .unwrap_or(0);
        column_bytes
            + confidence_bytes
            + phase_confidence_bytes
            + phase_mask_bytes
            + std::mem::size_of::<Self>()
    }

    /// Check if confidence scores are available

    /// Get confidence score for a sample at a marker (0-255 representing 0.0-1.0).
    /// Returns 255 (full confidence) if confidence data is not available.
    #[inline]
    pub fn sample_confidence(&self, marker: MarkerIdx<Space>, sample_idx: usize) -> u8 {
        if marker.as_usize() >= self.columns.len() {
            return 0;
        }
        if let Some(conf) = self
            .confidence
            .as_ref()
            .and_then(|c| c.get(marker.as_usize()))
            .and_then(|row| row.get(sample_idx))
            .copied()
        {
            return conf;
        }
        let sample = SampleIdx::new(sample_idx as u32);
        let a1 = self.allele(marker, sample.hap1());
        if a1 == 255 {
            return 0;
        }
        if self.samples.is_diploid(sample) {
            let a2 = self.allele(marker, sample.hap2());
            if a2 == 255 {
                return 0;
            }
        }
        255
    }

    /// Get confidence score as f32 (0.0-1.0)
    #[inline]
    pub fn sample_confidence_f32(&self, marker: MarkerIdx<Space>, sample_idx: usize) -> f32 {
        self.sample_confidence(marker, sample_idx) as f32 / 255.0
    }

    /// Clone the confidence data (for transferring to a new matrix)
    pub fn confidence_clone(&self) -> Option<Vec<Vec<u8>>> {
        self.confidence.clone()
    }

    /// Get phase confidence score for a sample at a marker (0-255).
    /// Returns 255 (full confidence) if phase confidence is not available.
    #[inline]
    pub fn sample_phase_confidence(&self, marker: MarkerIdx<Space>, sample_idx: usize) -> u8 {
        self.phase_confidence
            .as_ref()
            .and_then(|c| c.get(marker.as_usize()))
            .and_then(|row| row.get(sample_idx))
            .copied()
            .unwrap_or(255)
    }

    /// Get phase confidence score as f32 (0.0-1.0).
    #[inline]
    pub fn sample_phase_confidence_f32(&self, marker: MarkerIdx<Space>, sample_idx: usize) -> f32 {
        self.sample_phase_confidence(marker, sample_idx) as f32 / 255.0
    }

    /// Clone the phase confidence data (for transferring to a new matrix)
    pub fn phase_confidence_clone(&self) -> Option<Vec<Vec<u8>>> {
        self.phase_confidence.clone()
    }

    pub fn phase_mask(&self) -> Option<&Vec<Vec<u8>>> {
        self.phase_mask.as_ref()
    }


    pub fn with_phase_mask(mut self, phase_mask: Option<Vec<Vec<u8>>>) -> Self {
        if let Some(ref mask) = phase_mask {
            debug_assert_eq!(self.markers.len(), mask.len());
        }
        self.phase_mask = phase_mask;
        self
    }


    pub fn likelihoods_pl_arc(&self) -> Option<Arc<PlMatrix>> {
        self.likelihoods_pl.as_ref().map(Arc::clone)
    }

    #[inline]
    pub fn sample_pl(&self, marker: MarkerIdx<Space>, sample_idx: usize) -> Option<&[u16]> {
        self.likelihoods_pl
            .as_ref()
            .and_then(|pl| pl.sample_pl(marker.as_usize(), sample_idx))
    }
}

// ============================================================================
// Methods ONLY for Unphased matrices
// ============================================================================

impl<Space> GenotypeMatrix<Unphased, Space> {
    /// Create a new unphased genotype matrix
    pub fn new_unphased(
        markers: Markers<Space>,
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
            phase_confidence: None,
            phase_mask: None,
            likelihoods_pl: None,
            phantom: PhantomData,
        }
    }

    /// Create new unphased matrix with confidence scores
    pub fn new_unphased_with_confidence(
        markers: Markers<Space>,
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
            phase_confidence: None,
            phase_mask: None,
            likelihoods_pl: None,
            phantom: PhantomData,
        }
    }

    pub fn new_unphased_with_confidence_and_likelihoods(
        markers: Markers<Space>,
        columns: Vec<GenotypeColumn>,
        samples: Arc<Samples>,
        confidence: Option<Vec<Vec<u8>>>,
        likelihoods_pl: Arc<PlMatrix>,
    ) -> Self {
        debug_assert_eq!(markers.len(), columns.len());
        if let Some(ref conf) = confidence {
            debug_assert_eq!(markers.len(), conf.len());
        }
        Self {
            markers,
            columns,
            samples,
            is_reversed: false,
            confidence,
            phase_confidence: None,
            phase_mask: None,
            likelihoods_pl: Some(likelihoods_pl),
            phantom: PhantomData,
        }
    }

    /// Transform into a phased matrix.
    ///
    /// This is the primary way to create a `GenotypeMatrix<Phased>`.
    /// Consumes self to prevent accidental use of unphased data.
    pub fn into_phased(self) -> GenotypeMatrix<Phased, Space> {
        GenotypeMatrix {
            markers: self.markers,
            columns: self.columns,
            samples: self.samples,
            is_reversed: self.is_reversed,
            confidence: self.confidence,
            phase_confidence: self.phase_confidence,
            phase_mask: self.phase_mask,
            likelihoods_pl: self.likelihoods_pl,
            phantom: PhantomData,
        }
    }
}

// ============================================================================
// Methods ONLY for Phased matrices
// ============================================================================

impl<Space> GenotypeMatrix<Phased, Space> {
    /// Create a new phased genotype matrix
    pub fn new_phased(
        markers: Markers<Space>,
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
            phase_confidence: None,
            phase_mask: None,
            likelihoods_pl: None,
            phantom: PhantomData,
        }
    }

    /// Create a new phased genotype matrix with confidence scores
    pub fn new_phased_with_confidence(
        markers: Markers<Space>,
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
            phase_confidence: None,
            phase_mask: None,
            likelihoods_pl: None,
            phantom: PhantomData,
        }
    }

    pub fn new_phased_with_confidence_and_likelihoods(
        markers: Markers<Space>,
        columns: Vec<GenotypeColumn>,
        samples: Arc<Samples>,
        confidence: Option<Vec<Vec<u8>>>,
        likelihoods_pl: Option<Arc<PlMatrix>>,
    ) -> Self {
        debug_assert_eq!(markers.len(), columns.len());
        if let Some(ref conf) = confidence {
            debug_assert_eq!(markers.len(), conf.len());
        }

        if likelihoods_pl.is_none() {
            if let Some(conf) = confidence {
                return Self::new_phased_with_confidence(markers, columns, samples, conf);
            }
            return Self::new_phased(markers, columns, samples);
        }

        Self {
            markers,
            columns,
            samples,
            is_reversed: false,
            confidence,
            phase_confidence: None,
            phase_mask: None,
            likelihoods_pl,
            phantom: PhantomData,
        }
    }

    /// Attach phase confidence data to a phased matrix.
    pub fn with_phase_confidence(mut self, phase_confidence: Option<Vec<Vec<u8>>>) -> Self {
        if let Some(ref conf) = phase_confidence {
            debug_assert_eq!(self.markers.len(), conf.len());
        }
        self.phase_confidence = phase_confidence;
        self
    }

    /// Get a reference as unphased (zero-cost, same memory layout)
    pub fn as_unphased_ref(&self) -> &GenotypeMatrix<Unphased, Space> {
        // SAFETY: GenotypeMatrix<Phased> and GenotypeMatrix<Unphased> have identical
        // memory layouts (PhantomData is zero-sized), differing only in the type parameter
        unsafe {
            &*(self as *const GenotypeMatrix<Phased, Space> as *const GenotypeMatrix<Unphased, Space>)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::ChromIdx;
    use crate::data::marker::{Allele, Nucleotide};

    fn make_test_matrix_phased() -> GenotypeMatrix<Phased> {
        let samples = Arc::new(Samples::from_ids(vec!["S1".to_string(), "S2".to_string()]));
        let mut markers = Markers::<crate::data::AnyMarkerSpace>::new();
        markers.add_chrom("chr1");

        let m1 = Marker::new(
            ChromIdx::new(0),
            100,
            None,
            Allele::Base(Nucleotide::A),
            vec![Allele::Base(Nucleotide::C)],
        );
        let m2 = Marker::new(
            ChromIdx::new(0),
            200,
            None,
            Allele::Base(Nucleotide::A),
            vec![Allele::Base(Nucleotide::C)],
        );

        markers.push(m1);
        markers.push(m2);

        let col1 = GenotypeColumn::from_alleles(&[0, 1, 0, 1], 2);
        let col2 = GenotypeColumn::from_alleles(&[1, 1, 0, 0], 2);

        GenotypeMatrix::new_phased(markers, vec![col1, col2], samples)
    }

    fn make_test_matrix_unphased() -> GenotypeMatrix<Unphased> {
        let samples = Arc::new(Samples::from_ids(vec!["S1".to_string(), "S2".to_string()]));
        let mut markers = Markers::<crate::data::AnyMarkerSpace>::new();
        markers.add_chrom("chr1");

        let m1 = Marker::new(
            ChromIdx::new(0),
            100,
            None,
            Allele::Base(Nucleotide::A),
            vec![Allele::Base(Nucleotide::C)],
        );
        let m2 = Marker::new(
            ChromIdx::new(0),
            200,
            None,
            Allele::Base(Nucleotide::A),
            vec![Allele::Base(Nucleotide::C)],
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
        let mut markers = Markers::<crate::data::AnyMarkerSpace>::new();
        markers.add_chrom("chr1");

        let m1 = Marker::new(
            ChromIdx::new(0),
            100,
            None,
            Allele::Base(Nucleotide::A),
            vec![Allele::Base(Nucleotide::C)],
        );
        let m2 = Marker::new(
            ChromIdx::new(0),
            200,
            None,
            Allele::Base(Nucleotide::A),
            vec![Allele::Base(Nucleotide::C)],
        );

        markers.push(m1);
        markers.push(m2);

        let col1 = GenotypeColumn::from_alleles(&[0, 1, 0, 1], 2);
        let col2 = GenotypeColumn::from_alleles(&[1, 1, 0, 0], 2);

        // Create confidence scores: marker 0 has full confidence, marker 1 has 50% for sample 0
        let confidence = vec![
            vec![255, 255], // marker 0: full confidence for both samples
            vec![128, 255], // marker 1: 50% for sample 0, full for sample 1
        ];

        let matrix = GenotypeMatrix::new_unphased_with_confidence(
            markers,
            vec![col1, col2],
            samples,
            confidence,
        );

        assert!(matrix.confidence_clone().is_some());
        assert_eq!(matrix.sample_confidence(MarkerIdx::new(0), 0), 255);
        assert_eq!(matrix.sample_confidence(MarkerIdx::new(1), 0), 128);
        assert_eq!(matrix.sample_confidence(MarkerIdx::new(1), 1), 255);

        // Check f32 conversion
        assert!((matrix.sample_confidence_f32(MarkerIdx::new(0), 0) - 1.0).abs() < 0.01);
        assert!((matrix.sample_confidence_f32(MarkerIdx::new(1), 0) - 0.502).abs() < 0.01);

        // Verify confidence survives phase transition
        let phased = matrix.into_phased();
        assert!(phased.confidence_clone().is_some());
        assert_eq!(phased.sample_confidence(MarkerIdx::new(1), 0), 128);
    }

    #[test]
    fn test_no_confidence_defaults_to_full() {
        let matrix = make_test_matrix_unphased();

        // Without confidence data, has_confidence returns false
        assert!(matrix.confidence_clone().is_none());

        // But sample_confidence defaults to 255 (full confidence)
        assert_eq!(matrix.sample_confidence(MarkerIdx::new(0), 0), 255);
        assert!((matrix.sample_confidence_f32(MarkerIdx::new(0), 0) - 1.0).abs() < 0.01);
    }
}
