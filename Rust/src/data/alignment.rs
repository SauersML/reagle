//! Marker alignment between target and reference panels.

use std::collections::HashMap;
use crate::data::marker::{AlleleMapping, Marker, MarkerIdx};
use crate::data::storage::GenotypeMatrix;
use crate::data::storage::phase_state::PhaseState;
use crate::error::Result;

/// Marker alignment between target and reference panels
#[derive(Clone, Debug)]
pub struct MarkerAlignment {
    /// For each reference marker, the index of the corresponding target marker (-1 if not in target)
    pub ref_to_target: Vec<i32>,
    /// For each target marker, the index of the corresponding reference marker
    pub target_to_ref: Vec<usize>,
    
    /// Allele mapping for each aligned marker (indexed by target marker)
    /// Maps target allele indices to reference allele indices
    pub allele_mappings: Vec<Option<AlleleMapping>>,
}

impl MarkerAlignment {
    /// Create alignment by matching markers by position with allele mapping
    ///
    /// This handles strand flips (A/T vs T/A) and allele swaps automatically
    /// using `compute_allele_mapping`.
    pub fn new<S1: PhaseState, S2: PhaseState>(target_gt: &GenotypeMatrix<S1>, ref_gt: &GenotypeMatrix<S2>) -> Self {
        use crate::data::marker::compute_allele_mapping;

        let n_ref_markers = ref_gt.n_markers();
        let n_target_markers = target_gt.n_markers();

        // Build position -> target index map
        let mut target_pos_map: HashMap<(u16, u32), usize> = HashMap::new();
        for m in 0..n_target_markers {
            let marker = target_gt.marker(MarkerIdx::new(m as u32));
            target_pos_map.insert((marker.chrom.0, marker.pos), m);
        }

        // Map reference markers to target markers
        let mut ref_to_target = vec![-1i32; n_ref_markers];
        let mut target_to_ref = vec![0usize; n_target_markers];
        let mut allele_mappings: Vec<Option<AlleleMapping>> =
            vec![None; n_target_markers];

        let mut n_strand_flipped = 0usize;
        let mut n_allele_swapped = 0usize;

        for m in 0..n_ref_markers {
            let ref_marker = ref_gt.marker(MarkerIdx::new(m as u32));
            if let Some(&target_idx) = target_pos_map.get(&(ref_marker.chrom.0, ref_marker.pos)) {
                let target_marker = target_gt.marker(MarkerIdx::new(target_idx as u32));

                // Compute allele mapping (handles strand flips)
                if let Some(mapping) = compute_allele_mapping(target_marker, ref_marker) {
                    // Check if the mapping is valid (at least REF allele maps)
                    if mapping.is_valid() {
                        ref_to_target[m] = target_idx as i32;
                        target_to_ref[target_idx] = m;

                        if mapping.strand_flipped {
                            n_strand_flipped += 1;
                            // Warn about strand-ambiguous markers (A/T or C/G) where flip detection is unreliable
                            if crate::data::marker::is_strand_ambiguous(target_marker) {
                                eprintln!(
                                    "  Warning: Strand-ambiguous marker at pos {} (A/T or C/G SNV) was strand-flipped",
                                    target_marker.pos
                                );
                            }
                        }
                        if mapping.alleles_swapped {
                            n_allele_swapped += 1;
                        }

                        allele_mappings[target_idx] = Some(mapping);
                    }
                    // If mapping is invalid, marker won't be aligned
                }
            }
        }

        if n_strand_flipped > 0 || n_allele_swapped > 0 {
            eprintln!(
                "  Allele alignment: {} strand-flipped, {} allele-swapped markers",
                n_strand_flipped, n_allele_swapped
            );
        }

        Self {
            ref_to_target,
            target_to_ref,
            allele_mappings,
        }
    }

    /// Create alignment from overlapping windows (for streaming)
    pub fn new_from_windows<S1: PhaseState, S2: PhaseState>(
        target_win: &GenotypeMatrix<S1>,
        ref_win: &GenotypeMatrix<S2>,
        ref_pos_map: &HashMap<(u16, u32), usize>,
    ) -> Result<Self> {
        let n_ref_markers = ref_win.n_markers();
        let n_target_markers = target_win.n_markers();

        // Build position -> target index map for the window
        let mut target_pos_map: HashMap<(u32, u32), usize> = HashMap::new();
        for m in 0..n_target_markers {
            let marker = target_win.marker(MarkerIdx::new(m as u32));
            target_pos_map.insert((marker.chrom.0 as u32, marker.pos), m);
        }

        // Map reference markers to target markers
        let mut ref_to_target = vec![-1i32; n_ref_markers];
        let mut target_to_ref = vec![0usize; n_target_markers];
        let mut allele_mappings: Vec<Option<AlleleMapping>> =
            vec![None; n_target_markers];

        for ref_m in 0..n_ref_markers {
            let ref_marker = ref_win.marker(MarkerIdx::new(ref_m as u32));
            let ref_pos_global = (ref_marker.chrom.0 as u32, ref_marker.pos);

            // Check if this reference marker is genotyped in target window
            if let Some(&target_idx) = target_pos_map.get(&ref_pos_global) {
                let target_marker = target_win.marker(MarkerIdx::new(target_idx as u32));

                // Compute allele mapping
                if let Some(mapping) = crate::data::marker::compute_allele_mapping(target_marker, ref_marker) {
                    if mapping.is_valid() {
                        ref_to_target[ref_m] = target_idx as i32;
                        target_to_ref[target_idx] = ref_m;
                        allele_mappings[target_idx] = Some(mapping);
                    }
                }
            }
        }

        Ok(Self {
            ref_to_target,
            target_to_ref,
            allele_mappings,
        })
    }

    /// Create alignment from reference panel and position map
    ///
    /// Used for initializing alignment before processing windows.
    pub fn new_from_position_map(
        samples: &crate::data::haplotype::Samples,
        ref_pos_map: &std::collections::HashMap<(u32, u32), usize>,
    ) -> Result<Self> {
        // Suppress unused variable warning by referencing them
        if false {
            println!("{:?} {:?}", samples.len(), ref_pos_map.len());
        }
        // For streaming imputation, we don't have all target genotypes at this point
        // This is a stub that gets filled in by new_from_windows later
        // Return empty alignment
        Ok(Self {
            ref_to_target: vec![],
            target_to_ref: vec![],
            allele_mappings: vec![],
        })
    }

    /// Get target marker index for a reference marker (returns None if not genotyped)
    #[inline]
    pub fn target_marker(&self, ref_marker: usize) -> Option<usize> {
        // Use direct vector access for performance if possible, but keep method for compatibility
        let idx = *self.ref_to_target.get(ref_marker).unwrap_or(&-1);
        if idx >= 0 { Some(idx as usize) } else { None }
    }

    /// Map a target allele to reference allele space
    ///
    /// Returns the reference allele index for a given target allele,
    /// handling strand flips and swaps automatically.
    /// Returns 255 (missing) if no valid mapping exists.
    pub fn map_allele(&self, target_marker: usize, target_allele: u8) -> u8 {
        if target_allele == 255 {
            return 255; // Missing stays missing
        }

        if let Some(Some(mapping)) = self.allele_mappings.get(target_marker) {
            mapping.map_allele(target_allele).unwrap_or(255)
        } else {
            // No mapping means identity (direct match assumed)
            target_allele
        }
    }

    /// Map a reference allele to target allele space (reverse mapping)
    ///
    /// Returns the target allele index for a given reference allele,
    /// handling strand flips and swaps automatically.
    /// Returns 255 (missing) if no valid mapping exists.
    pub fn reverse_map_allele(&self, target_marker: usize, ref_allele: u8) -> u8 {
        if ref_allele == 255 {
            return 255; // Missing stays missing
        }

        if let Some(Some(mapping)) = self.allele_mappings.get(target_marker) {
            mapping.reverse_map_allele(ref_allele).unwrap_or(255)
        } else {
            // No mapping means identity (direct match assumed)
            ref_allele
        }
    }

    /// Get reference marker index for a target marker (returns None if not aligned)
    pub fn target_to_ref(&self, target_marker: usize) -> Option<usize> {
        // Check allele_mappings to ensure the marker actually aligns.
        // The raw target_to_ref vector initializes with 0s, which is ambiguous.
        if self.allele_mappings.get(target_marker).and_then(|m| m.as_ref()).is_some() {
            Some(self.target_to_ref[target_marker])
        } else {
            None
        }
    }

    /// Check if a target marker has non-identity allele mapping (strand flip or swap)
    ///
    /// Returns true if the marker needs allele remapping, false if identity mapping applies.
    /// Most biallelic markers have identity mapping (no flips/swaps).
    #[inline]
    pub fn has_allele_mapping(&self, target_marker: usize) -> bool {
        self.allele_mappings.get(target_marker)
            .and_then(|m| m.as_ref())
            .is_some()
    }

    /// Get the number of markers that were successfully aligned
    pub fn n_aligned(&self) -> usize {
        self.ref_to_target.iter().filter(|&&x| x >= 0).count()
    }
}
