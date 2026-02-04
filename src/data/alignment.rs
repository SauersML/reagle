//! Marker alignment between target and reference panels.

use crate::data::marker::{AlleleMapping, AnyMarkerSpace, MarkerIdx, Markers};
use crate::data::storage::GenotypeMatrix;
use crate::data::storage::phase_state::PhaseState;
use std::collections::HashMap;

/// Marker alignment between target and reference panels
#[derive(Debug)]
pub struct MarkerAlignment<TargetSpace = AnyMarkerSpace, RefSpace = AnyMarkerSpace> {
    /// For each reference marker, the index of the corresponding target marker (-1 if not in target)
    pub ref_to_target: Vec<Option<MarkerIdx<TargetSpace>>>,
    /// For each target marker, the index of the corresponding reference marker
    pub target_to_ref: Vec<Option<MarkerIdx<RefSpace>>>,

    /// Allele mapping for each aligned marker (indexed by target marker)
    /// Maps target allele indices to reference allele indices
    pub allele_mappings: Vec<Option<AlleleMapping>>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct AlignmentStats {
    pub aligned: usize,
    pub strand_flipped: usize,
    pub allele_swapped: usize,
}

impl<TargetSpace, RefSpace> Clone for MarkerAlignment<TargetSpace, RefSpace> {
    fn clone(&self) -> Self {
        Self {
            ref_to_target: self.ref_to_target.clone(),
            target_to_ref: self.target_to_ref.clone(),
            allele_mappings: self.allele_mappings.clone(),
        }
    }
}

impl<TargetSpace, RefSpace> MarkerAlignment<TargetSpace, RefSpace> {
    /// Create alignment by matching markers by position with allele mapping
    ///
    /// This handles strand flips (A/T vs T/A) and allele swaps automatically
    /// using `compute_allele_mapping`.
    pub fn new<S1: PhaseState, S2: PhaseState>(
        target_gt: &GenotypeMatrix<S1, TargetSpace>,
        ref_gt: &GenotypeMatrix<S2, RefSpace>,
    ) -> Self {
        use crate::data::marker::compute_allele_mapping;

        let n_ref_markers = ref_gt.n_markers();
        let n_target_markers = target_gt.n_markers();

        // Build position -> target index map (keyed by chrom name for stability)
        let mut target_pos_map: HashMap<(String, u32), Vec<usize>> = HashMap::new();
        for m in 0..n_target_markers {
            let marker = target_gt.marker(MarkerIdx::new(m as u32));
            let chrom_name = target_gt.markers().chrom_name(marker.chrom).unwrap_or("");
            let chrom_norm = normalize_chrom(chrom_name).to_string();
            target_pos_map
                .entry((chrom_norm, marker.pos))
                .or_default()
                .push(m);
        }

        // Map reference markers to target markers
        let mut ref_to_target = vec![None; n_ref_markers];
        let mut target_to_ref = vec![None; n_target_markers];
        let mut allele_mappings: Vec<Option<AlleleMapping>> = vec![None; n_target_markers];

        let mut n_strand_flipped = 0usize;
        let mut n_allele_swapped = 0usize;

        let mut used_targets = vec![false; n_target_markers];
        for m in 0..n_ref_markers {
            let ref_marker = ref_gt.marker(MarkerIdx::new(m as u32));
            let ref_chrom = ref_gt.markers().chrom_name(ref_marker.chrom).unwrap_or("");
            let ref_chrom_norm = normalize_chrom(ref_chrom).to_string();
            if let Some(target_candidates) = target_pos_map.get(&(ref_chrom_norm, ref_marker.pos)) {
                for &target_idx in target_candidates {
                    if used_targets[target_idx] {
                        continue;
                    }
                    let target_marker = target_gt.marker(MarkerIdx::new(target_idx as u32));

                    // Compute allele mapping (handles strand flips)
                    if let Some(mapping) = compute_allele_mapping(target_marker, ref_marker) {
                        // Check if the mapping is valid (at least REF allele maps)
                        if mapping.is_valid() {
                            let strand_flipped = mapping.strand_flipped;
                            let alleles_swapped = mapping.alleles_swapped;
                            ref_to_target[m] = Some(MarkerIdx::new(target_idx as u32));
                            target_to_ref[target_idx] = Some(MarkerIdx::new(m as u32));
                            allele_mappings[target_idx] = Some(mapping);
                            used_targets[target_idx] = true;

                            if strand_flipped {
                                n_strand_flipped += 1;
                                // Warn about strand-ambiguous markers (A/T or C/G SNV) where flip detection is unreliable
                                if crate::data::marker::is_strand_ambiguous(target_marker) {
                                    eprintln!(
                                        "  Warning: Strand-ambiguous marker at pos {} (A/T or C/G SNV) was strand-flipped",
                                        target_marker.pos
                                    );
                                }
                            }
                            if alleles_swapped {
                                n_allele_swapped += 1;
                                eprintln!(
                                    "  Allele swapped at pos {}: target {}>{}, ref {}>{}",
                                    target_marker.pos,
                                    target_marker.ref_allele,
                                    target_marker
                                        .alt_alleles
                                        .get(0)
                                        .map(|a| a.to_string())
                                        .unwrap_or_else(|| "-".to_string()),
                                    ref_marker.ref_allele,
                                    ref_marker
                                        .alt_alleles
                                        .get(0)
                                        .map(|a| a.to_string())
                                        .unwrap_or_else(|| "-".to_string()),
                                );
                            }
                            break;
                        }
                        // If mapping is invalid, marker won't be aligned
                    }
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

    /// Create alignment against reference markers without reference genotypes.
    pub fn new_with_ref_markers<S: PhaseState>(
        target_gt: &GenotypeMatrix<S, TargetSpace>,
        ref_markers: &Markers<RefSpace>,
    ) -> Self {
        use crate::data::marker::compute_allele_mapping;

        let n_ref_markers = ref_markers.len();
        let n_target_markers = target_gt.n_markers();

        let mut target_pos_map: HashMap<(String, u32), Vec<usize>> = HashMap::new();
        for m in 0..n_target_markers {
            let marker = target_gt.marker(MarkerIdx::new(m as u32));
            let chrom_name = target_gt.markers().chrom_name(marker.chrom).unwrap_or("");
            let chrom_norm = normalize_chrom(chrom_name).to_string();
            target_pos_map
                .entry((chrom_norm, marker.pos))
                .or_default()
                .push(m);
        }

        let mut ref_to_target = vec![None; n_ref_markers];
        let mut target_to_ref = vec![None; n_target_markers];
        let mut allele_mappings: Vec<Option<AlleleMapping>> = vec![None; n_target_markers];

        let mut n_strand_flipped = 0usize;
        let mut n_allele_swapped = 0usize;

        let mut used_targets = vec![false; n_target_markers];
        let mut unaligned_with_candidates = 0usize;
        let mut unaligned_examples: Vec<String> = Vec::new();
        for m in 0..n_ref_markers {
            let ref_marker = ref_markers.marker(MarkerIdx::new(m as u32));
            let ref_chrom = ref_markers.chrom_name(ref_marker.chrom).unwrap_or("");
            let ref_chrom_norm = normalize_chrom(ref_chrom).to_string();
            if let Some(target_candidates) = target_pos_map.get(&(ref_chrom_norm, ref_marker.pos)) {
                let mut aligned = false;
                for &target_idx in target_candidates {
                    if used_targets[target_idx] {
                        continue;
                    }
                    let target_marker = target_gt.marker(MarkerIdx::new(target_idx as u32));

                    if let Some(mapping) = compute_allele_mapping(target_marker, ref_marker) {
                        if mapping.is_valid() {
                            let strand_flipped = mapping.strand_flipped;
                            let alleles_swapped = mapping.alleles_swapped;
                            ref_to_target[m] = Some(MarkerIdx::new(target_idx as u32));
                            target_to_ref[target_idx] = Some(MarkerIdx::new(m as u32));
                            allele_mappings[target_idx] = Some(mapping);
                            used_targets[target_idx] = true;
                            aligned = true;

                            if strand_flipped {
                                n_strand_flipped += 1;
                                if crate::data::marker::is_strand_ambiguous(target_marker) {
                                    eprintln!(
                                        "  Warning: Strand-ambiguous marker at pos {} (A/T or C/G SNV) was strand-flipped",
                                        target_marker.pos
                                    );
                                }
                            }
                            if alleles_swapped {
                                n_allele_swapped += 1;
                                eprintln!(
                                    "  Allele swapped at pos {}: target {}>{}, ref {}>{}",
                                    target_marker.pos,
                                    target_marker.ref_allele,
                                    target_marker
                                        .alt_alleles
                                        .get(0)
                                        .map(|a| a.to_string())
                                        .unwrap_or_else(|| "-".to_string()),
                                    ref_marker.ref_allele,
                                    ref_marker
                                        .alt_alleles
                                        .get(0)
                                        .map(|a| a.to_string())
                                        .unwrap_or_else(|| "-".to_string()),
                                );
                            }
                            break;
                        }
                    }
                }
                if !aligned {
                    unaligned_with_candidates += 1;
                    if unaligned_examples.len() < 5 {
                        let ref_alt = ref_marker
                            .alt_alleles
                            .iter()
                            .map(|a| a.to_string())
                            .collect::<Vec<_>>()
                            .join(",");
                        unaligned_examples.push(format!(
                            "{}:{} {}>{}",
                            ref_chrom, ref_marker.pos, ref_marker.ref_allele, ref_alt
                        ));
                    }
                }
            }
        }

        if n_strand_flipped > 0 || n_allele_swapped > 0 {
            eprintln!(
                "  Allele alignment: {} strand-flipped, {} allele-swapped markers",
                n_strand_flipped, n_allele_swapped
            );
        }
        if unaligned_with_candidates > 0 {
            if unaligned_examples.is_empty() {
                eprintln!(
                    "  Unaligned positions with candidates: {}",
                    unaligned_with_candidates
                );
            } else {
                eprintln!(
                    "  Unaligned positions with candidates: {} (first {})",
                    unaligned_with_candidates,
                    unaligned_examples.len()
                );
                eprintln!("  Examples: {}", unaligned_examples.join(", "));
            }
        }

        Self {
            ref_to_target,
            target_to_ref,
            allele_mappings,
        }
    }

    /// Build a reference position index for fast streaming alignment.
    pub fn build_ref_pos_index(
        ref_markers: &Markers<RefSpace>,
    ) -> HashMap<(String, u32), Vec<usize>> {
        let mut ref_pos_map: HashMap<(String, u32), Vec<usize>> = HashMap::new();
        for m in 0..ref_markers.len() {
            let marker = ref_markers.marker(MarkerIdx::new(m as u32));
            let chrom_name = ref_markers.chrom_name(marker.chrom).unwrap_or("");
            let chrom_norm = normalize_chrom(chrom_name).to_string();
            ref_pos_map
                .entry((chrom_norm, marker.pos))
                .or_default()
                .push(m);
        }
        ref_pos_map
    }

    /// Create alignment against a pre-built reference position index.
    ///
    /// This avoids scanning all reference markers for each target window. The returned
    /// alignment is sized to the target markers; `ref_to_target` is left empty because
    /// streaming phasing only queries target-to-ref mappings.
    pub fn new_with_ref_index<S: PhaseState>(
        target_gt: &GenotypeMatrix<S, TargetSpace>,
        ref_markers: &Markers<RefSpace>,
        ref_pos_map: &HashMap<(String, u32), Vec<usize>>,
    ) -> (Self, AlignmentStats) {
        use crate::data::marker::compute_allele_mapping;

        let n_target_markers = target_gt.n_markers();
        let mut target_to_ref = vec![None; n_target_markers];
        let mut allele_mappings: Vec<Option<AlleleMapping>> = vec![None; n_target_markers];
        let mut stats = AlignmentStats::default();

        for m in 0..n_target_markers {
            let target_marker = target_gt.marker(MarkerIdx::new(m as u32));
            let chrom_name = target_gt.markers().chrom_name(target_marker.chrom).unwrap_or("");
            let chrom_norm = normalize_chrom(chrom_name).to_string();
            if let Some(ref_candidates) = ref_pos_map.get(&(chrom_norm, target_marker.pos)) {
                for &ref_idx in ref_candidates {
                    let ref_marker = ref_markers.marker(MarkerIdx::new(ref_idx as u32));
                    if let Some(mapping) = compute_allele_mapping(target_marker, ref_marker) {
                        if mapping.is_valid() {
                            let strand_flipped = mapping.strand_flipped;
                            let alleles_swapped = mapping.alleles_swapped;
                            target_to_ref[m] = Some(MarkerIdx::new(ref_idx as u32));
                            if strand_flipped {
                                stats.strand_flipped += 1;
                            }
                            if alleles_swapped {
                                stats.allele_swapped += 1;
                                eprintln!(
                                    "  Allele swapped at pos {}: target {}>{}, ref {}>{}",
                                    target_marker.pos,
                                    target_marker.ref_allele,
                                    target_marker
                                        .alt_alleles
                                        .get(0)
                                        .map(|a| a.to_string())
                                        .unwrap_or_else(|| "-".to_string()),
                                    ref_marker.ref_allele,
                                    ref_marker
                                        .alt_alleles
                                        .get(0)
                                        .map(|a| a.to_string())
                                        .unwrap_or_else(|| "-".to_string()),
                                );
                            }
                            allele_mappings[m] = Some(mapping);
                            stats.aligned += 1;
                            break;
                        }
                    }
                }
            }
        }

        (
            Self {
                ref_to_target: Vec::new(),
                target_to_ref,
                allele_mappings,
            },
            stats,
        )
    }

    /// Get target marker index for a reference marker (returns None if not genotyped)
    #[inline]
    pub fn target_marker(&self, ref_marker: MarkerIdx<RefSpace>) -> Option<MarkerIdx<TargetSpace>> {
        self.ref_to_target
            .get(ref_marker.as_usize())
            .and_then(|idx| *idx)
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
    pub fn target_to_ref(
        &self,
        target_marker: MarkerIdx<TargetSpace>,
    ) -> Option<MarkerIdx<RefSpace>> {
        // Check allele_mappings to ensure the marker actually aligns.
        // The raw target_to_ref vector initializes with 0s, which is ambiguous.
        if self
            .allele_mappings
            .get(target_marker.as_usize())
            .and_then(|m| m.as_ref())
            .is_some()
        {
            self.target_to_ref
                .get(target_marker.as_usize())
                .and_then(|idx| *idx)
        } else {
            None
        }
    }

    /// Get the number of markers that were successfully aligned
    pub fn n_aligned(&self) -> usize {
        self.ref_to_target.iter().filter(|x| x.is_some()).count()
    }
}

#[inline]
fn normalize_chrom(name: &str) -> &str {
    if name.len() >= 3 && name[..3].eq_ignore_ascii_case("chr") {
        &name[3..]
    } else {
        name
    }
}
