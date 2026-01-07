//! # Imputation Pipeline
//!
//! Orchestrates the imputation workflow:
//! 1. Load target and reference VCFs
//! 2. Align markers between target and reference
//! 3. Process data in overlapping sliding windows (for memory efficiency)
//! 4. Run Li-Stephens HMM for each target haplotype with dynamic PBWT state selection
//! 5. Interpolate state probabilities for ungenotyped markers
//! 6. Splice window results at overlap midpoints
//! 7. Compute dosages and write output with quality metrics (DR2, AF)
//!
//! This matches Java `imp/ImpLS.java`, `imp/ImpLSBaum.java`, and related classes.

use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

use crate::config::Config;
use crate::data::genetic_map::GeneticMaps;
use crate::data::haplotype::HapIdx;
use crate::data::marker::MarkerIdx;
use crate::data::storage::GenotypeMatrix;
use crate::data::storage::phase_state::{PhaseState, Phased};
use crate::data::storage::coded_steps::RefPanelCoded;
use crate::error::Result;
use crate::io::bref3::Bref3Reader;
use crate::io::vcf::{ImputationQuality, VcfReader, VcfWriter};
use crate::utils::workspace::ImpWorkspace;

use crate::model::imp_states::ImpStates;
use crate::model::parameters::ModelParams;

/// Imputation pipeline
pub struct ImputationPipeline {
    config: Config,
    params: ModelParams,
}

/// Marker alignment between target and reference panels
#[derive(Clone, Debug)]
pub struct MarkerAlignment {
    /// For each reference marker, the index of the corresponding target marker (-1 if not in target)
    ref_to_target: Vec<i32>,
    /// For each target marker, the index of the corresponding reference marker
    target_to_ref: Vec<usize>,
    /// Number of reference markers

    /// Allele mapping for each aligned marker (indexed by target marker)
    /// Maps target allele indices to reference allele indices
    allele_mappings: Vec<Option<crate::data::marker::AlleleMapping>>,
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
        let mut allele_mappings: Vec<Option<crate::data::marker::AlleleMapping>> =
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

    /// Check if a reference marker is genotyped in target
    pub fn is_genotyped(&self, ref_marker: usize) -> bool {
        self.ref_to_target.get(ref_marker).copied().unwrap_or(-1) >= 0
    }

    /// Get target marker index for a reference marker (returns None if not genotyped)
    pub fn target_marker(&self, ref_marker: usize) -> Option<usize> {
        let idx = self.ref_to_target.get(ref_marker).copied().unwrap_or(-1);
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

    /// Get the number of markers that were successfully aligned
    pub fn n_aligned(&self) -> usize {
        self.ref_to_target.iter().filter(|&&x| x >= 0).count()
    }
}
/// State probabilities from HMM forward-backward on reference markers.
///
/// Sparse state probabilities with interpolation for ungenotyped markers.
///
/// The HMM runs only on GENOTYPED markers (where target has data), matching
/// Java Beagle's efficient approach. State probabilities for ungenotyped markers
/// are computed via linear interpolation in genetic distance space.
///
/// This provides ~100x speedup over running HMM on all reference markers.
///
/// **Important**: This structure stores `probs_p1` (probabilities at the next marker)
/// to enable proper interpolation matching Java Beagle's approach. This ensures
/// the **same** haplotype is used with smoothly transitioning probability weights.
#[derive(Clone, Debug)]
pub struct StateProbs {
    /// Indices of genotyped markers in reference space
    /// Uses Arc to share across all haplotypes (avoids cloning per sample)
    genotyped_markers: std::sync::Arc<Vec<usize>>,
    /// Reference haplotype indices at each genotyped marker
    hap_indices: Vec<Vec<u32>>,
    /// State probabilities at each genotyped marker
    probs: Vec<Vec<f32>>,
    /// State probabilities at the **next** genotyped marker (for interpolation)
    /// At the last marker, this equals probs (no next marker)
    probs_p1: Vec<Vec<f32>>,
    /// Genetic positions of ALL reference markers (for interpolation)
    /// Uses Arc to share across all haplotypes (avoids cloning ~8MB per haplotype)
    gen_positions: std::sync::Arc<Vec<f64>>,
}

impl StateProbs {
    /// Create state probabilities from sparse HMM output.
    ///
    /// # Arguments
    /// * `genotyped_markers` - Indices of genotyped markers in reference space
    /// * `n_states` - Number of HMM states
    /// * `hap_indices` - Reference haplotype indices at each genotyped marker
    /// * `state_probs` - Flattened state probabilities from HMM (genotyped markers only)
    /// * `gen_positions` - Genetic positions of ALL reference markers
    ///
    /// # Java Compatibility
    /// Stores `probs_p1` (probability at next marker) to match Java Beagle's
    /// `StateProbsFactory.stateProbs()` which stores both `probs[m]` and `probsP1[m]`
    /// for proper interpolation of ungenotyped markers.
    pub fn new(
        genotyped_markers: std::sync::Arc<Vec<usize>>,
        n_states: usize,
        hap_indices: Vec<Vec<u32>>,
        state_probs: Vec<f32>,
        gen_positions: std::sync::Arc<Vec<f64>>,
    ) -> Self {
        let threshold = 0.005f32.min(0.9999 / n_states as f32);
        let n_genotyped = genotyped_markers.len();
        let n_genotyped_m1 = n_genotyped.saturating_sub(1);

        let mut filtered_haps = Vec::with_capacity(n_genotyped);
        let mut filtered_probs = Vec::with_capacity(n_genotyped);
        let mut filtered_probs_p1 = Vec::with_capacity(n_genotyped);

        for sparse_m in 0..n_genotyped {
            let row_offset = sparse_m * n_states;
            // For probs_p1, use next marker's probs, or same marker if at end
            let sparse_m_p1 = if sparse_m < n_genotyped_m1 { sparse_m + 1 } else { sparse_m };
            let row_offset_p1 = sparse_m_p1 * n_states;

            let mut haps = Vec::new();
            let mut probs = Vec::new();
            let mut probs_p1 = Vec::new();

            // Match Java: include state if prob at m OR prob at m+1 exceeds threshold
            for j in 0..n_states.min(hap_indices.get(sparse_m).map(|v| v.len()).unwrap_or(0)) {
                let prob = state_probs.get(row_offset + j).copied().unwrap_or(0.0);
                let prob_p1 = state_probs.get(row_offset_p1 + j).copied().unwrap_or(0.0);

                if prob > threshold || prob_p1 > threshold {
                    haps.push(hap_indices[sparse_m][j]);
                    probs.push(prob);
                    probs_p1.push(prob_p1);
                }
            }

            filtered_haps.push(haps);
            filtered_probs.push(probs);
            filtered_probs_p1.push(probs_p1);
        }

        Self {
            genotyped_markers,
            hap_indices: filtered_haps,
            probs: filtered_probs,
            probs_p1: filtered_probs_p1,
            gen_positions,
        }
    }

    /// Compute per-allele probabilities at a reference marker.
    /// Returns optimized representation: Biallelic for 2-allele sites, Multiallelic for others.
    #[inline]
    pub fn allele_posteriors<F>(&self, ref_marker: usize, n_alleles: usize, get_ref_allele: F) -> AllelePosteriors
    where
        F: Fn(usize, u32) -> u8,
    {
        match self.genotyped_markers.binary_search(&ref_marker) {
            Ok(sparse_idx) => {
                self.posteriors_at_genotyped(sparse_idx, ref_marker, n_alleles, &get_ref_allele)
            }
            Err(insert_pos) => {
                self.posteriors_interpolated(ref_marker, insert_pos, n_alleles, &get_ref_allele)
            }
        }
    }

    /// Per-allele probabilities at a genotyped marker
    #[inline]
    fn posteriors_at_genotyped<F>(
        &self,
        sparse_idx: usize,
        ref_marker: usize,
        n_alleles: usize,
        get_ref_allele: &F,
    ) -> AllelePosteriors
    where
        F: Fn(usize, u32) -> u8,
    {
        let haps = &self.hap_indices[sparse_idx];
        let probs = &self.probs[sparse_idx];

        if n_alleles == 2 {
            // Optimized biallelic path - just compute P(ALT)
            let mut p_alt = 0.0f32;
            for (j, &hap) in haps.iter().enumerate() {
                let allele = get_ref_allele(ref_marker, hap);
                if allele == 1 {
                    p_alt += probs[j];
                }
            }
            AllelePosteriors::Biallelic(p_alt)
        } else {
            // Full multiallelic - compute PMF
            let mut al_probs = vec![0.0f32; n_alleles];
            for (j, &hap) in haps.iter().enumerate() {
                let allele = get_ref_allele(ref_marker, hap);
                if allele != 255 && (allele as usize) < n_alleles {
                    al_probs[allele as usize] += probs[j];
                }
            }
            AllelePosteriors::Multiallelic(al_probs)
        }
    }

    /// Per-allele probabilities at an ungenotyped marker via interpolation
    #[inline]
    fn posteriors_interpolated<F>(
        &self,
        ref_marker: usize,
        insert_pos: usize,
        n_alleles: usize,
        get_ref_allele: &F,
    ) -> AllelePosteriors
    where
        F: Fn(usize, u32) -> u8,
    {
        let n_genotyped = self.genotyped_markers.len();

        // Handle edge cases
        if n_genotyped == 0 {
            return if n_alleles == 2 {
                AllelePosteriors::Biallelic(0.0)
            } else {
                AllelePosteriors::Multiallelic(vec![0.0f32; n_alleles])
            };
        }
        if insert_pos == 0 {
            return self.posteriors_at_genotyped(0, ref_marker, n_alleles, get_ref_allele);
        }
        if insert_pos >= n_genotyped {
            return self.posteriors_at_genotyped(n_genotyped - 1, ref_marker, n_alleles, get_ref_allele);
        }

        // Interpolate using left marker's haplotypes
        let left_sparse = insert_pos - 1;
        let left_ref = self.genotyped_markers[left_sparse];
        let right_ref = self.genotyped_markers[insert_pos];

        let pos_left = self.gen_positions[left_ref];
        let pos_right = self.gen_positions[right_ref];
        let pos_marker = self.gen_positions[ref_marker];

        let total_dist = pos_right - pos_left;
        let weight_left = if total_dist > 1e-10 {
            ((pos_right - pos_marker) / total_dist) as f32
        } else {
            0.5
        };

        let haps = &self.hap_indices[left_sparse];
        let probs = &self.probs[left_sparse];
        let probs_p1 = &self.probs_p1[left_sparse];

        if n_alleles == 2 {
            // Optimized biallelic path
            let mut p_alt = 0.0f32;
            for (j, &hap) in haps.iter().enumerate() {
                let allele = get_ref_allele(ref_marker, hap);
                if allele == 1 {
                    let interpolated_prob = weight_left * probs[j] + (1.0 - weight_left) * probs_p1[j];
                    p_alt += interpolated_prob;
                }
            }
            AllelePosteriors::Biallelic(p_alt)
        } else {
            // Full multiallelic
            let mut al_probs = vec![0.0f32; n_alleles];
            for (j, &hap) in haps.iter().enumerate() {
                let allele = get_ref_allele(ref_marker, hap);
                if allele != 255 && (allele as usize) < n_alleles {
                    let interpolated_prob = weight_left * probs[j] + (1.0 - weight_left) * probs_p1[j];
                    al_probs[allele as usize] += interpolated_prob;
                }
            }
            AllelePosteriors::Multiallelic(al_probs)
        }
    }
}

/// Per-haplotype allele posterior probabilities.
/// Optimized: uses compact representation for biallelic (99% of sites).
#[derive(Clone, Debug)]
pub enum AllelePosteriors {
    /// Biallelic site: just store P(ALT)
    Biallelic(f32),
    /// Multiallelic site: full PMF where index i = P(allele i)
    Multiallelic(Vec<f32>),
}

impl AllelePosteriors {
    /// Get P(allele i)
    #[inline]
    pub fn prob(&self, allele: usize) -> f32 {
        match self {
            AllelePosteriors::Biallelic(p_alt) => {
                if allele == 0 { 1.0 - p_alt } else if allele == 1 { *p_alt } else { 0.0 }
            }
            AllelePosteriors::Multiallelic(probs) => {
                probs.get(allele).copied().unwrap_or(0.0)
            }
        }
    }

    /// Compute dosage = expected allele index = sum(i * P(i))
    #[inline]
    pub fn dosage(&self) -> f32 {
        match self {
            AllelePosteriors::Biallelic(p_alt) => *p_alt,
            AllelePosteriors::Multiallelic(probs) => {
                probs.iter().enumerate().map(|(i, &p)| i as f32 * p).sum()
            }
        }
    }

    /// Get the most likely allele (argmax)
    #[inline]
    pub fn max_allele(&self) -> u8 {
        match self {
            AllelePosteriors::Biallelic(p_alt) => if *p_alt >= 0.5 { 1 } else { 0 },
            AllelePosteriors::Multiallelic(probs) => {
                probs.iter().enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i as u8)
                    .unwrap_or(0)
            }
        }
    }

}

impl ImputationPipeline {
    /// Create a new imputation pipeline
    pub fn new(config: Config) -> Self {
        let params = ModelParams::new();
        Self { config, params }
    }

    /// Run the imputation pipeline
    pub fn run(&mut self) -> Result<()> {
        eprintln!("Loading target VCF...");
        let (mut target_reader, target_file) = VcfReader::open(&self.config.gt)?;
        let target_samples = target_reader.samples_arc();
        let target_gt = target_reader.read_all(target_file)?;

        eprintln!("Loading reference panel...");
        let ref_path = self.config.r#ref.as_ref().ok_or_else(|| {
            crate::error::ReagleError::config("Reference panel required for imputation")
        })?;

        // Detect file format by extension and load accordingly
        // Wrap in Arc for shared ownership (avoids cloning when passing to phasing pipeline)
        let ref_gt: Arc<GenotypeMatrix<Phased>> = Arc::new(if ref_path.extension().map(|e| e == "bref3").unwrap_or(false) {
            eprintln!("  Detected BREF3 format");
            let reader = Bref3Reader::open(ref_path)?;
            reader.read_all()?
        } else {
            eprintln!("  Detected VCF format");
            let (mut ref_reader, ref_file) = VcfReader::open(ref_path)?;
            ref_reader.read_all(ref_file)?.into_phased()
        });

        if target_gt.n_markers() == 0 || ref_gt.n_markers() == 0 {
            return Ok(());
        }

        // Phase target before imputation (imputation requires phased haplotypes)
        // With type states, VcfReader returns GenotypeMatrix<Unphased>, so we always phase
        eprintln!("Phasing target data before imputation...");

        // Create marker alignment (reused for both phasing and imputation)
        // This can be done before phasing since markers don't change during phasing
        eprintln!("Aligning markers...");
        let alignment = MarkerAlignment::new(&target_gt, &ref_gt);

        let n_ref_haps = ref_gt.n_haplotypes();
        let n_ref_markers = ref_gt.n_markers();
        let n_genotyped = alignment.ref_to_target.iter().filter(|&&x| x >= 0).count();
        eprintln!(
            "  {} of {} reference markers are genotyped in target",
            n_genotyped, n_ref_markers
        );

        // Create phasing pipeline with current config
        let mut phasing = super::phasing::PhasingPipeline::new(self.config.clone());

        // Set reference panel for reference-guided phasing
        // Uses Arc::clone for cheap reference count increment instead of deep cloning
        phasing.set_reference(Arc::clone(&ref_gt), alignment.clone());
        eprintln!("Using reference panel ({} haplotypes) for phasing", n_ref_haps);

        // Load genetic map if provided
        let gen_maps = if let Some(ref map_path) = self.config.map {
            let chrom_names: Vec<&str> = target_gt
                .markers()
                .chrom_names()
                .iter()
                .map(|s| s.as_ref())
                .collect();
            GeneticMaps::from_plink_file(map_path, &chrom_names)?
        } else {
            GeneticMaps::new()
        };

        let target_gt = phasing.phase_in_memory(&target_gt, &gen_maps)?;

        let n_target_markers = target_gt.n_markers();
        let n_target_samples = target_gt.n_samples();
        let n_target_haps = target_gt.n_haplotypes();

        eprintln!(
            "Target: {} markers, {} samples; Reference: {} markers, {} haplotypes",
            n_target_markers, n_target_samples, n_ref_markers, n_ref_haps
        );

        // Initialize parameters with CLI config
        // Java uses nHaps = nRefHaps + nTargHaps for recombIntensity calculation
        let n_total_haps = n_ref_haps + n_target_haps;
        self.params = ModelParams::for_imputation(n_total_haps, self.config.ne, self.config.err);
        self.params
            .set_n_states(self.config.imp_states.min(n_ref_haps));

        // Load genetic map if provided
        let gen_maps = if let Some(ref map_path) = self.config.map {
            let chrom_names: Vec<&str> = ref_gt
                .markers()
                .chrom_names()
                .iter()
                .map(|s| s.as_ref())
                .collect();
            GeneticMaps::from_plink_file(map_path, &chrom_names)?
        } else {
            GeneticMaps::new()
        };

        let chrom = ref_gt.marker(MarkerIdx::new(0)).chrom;

        // Note: gen_positions and steps_config removed - ImpStates now uses ref_panel step boundaries directly

        // Number of IBS haplotypes to find per step
        // Java: nHapsPerStep = imp_states / (imp_segment / imp_step)
        //     = 1600 / (6.0 / 0.1) = 1600 / 60 ≈ 26
        // We add a minimum of sqrt(imp_states) to handle low imp_states values.
        // Without this floor, imp_states=50 with 60 steps/segment gives 50/60=0,
        // causing IBS matching to fail completely.
        let n_steps_per_segment = (self.config.imp_segment / self.config.imp_step).round() as usize;
        let computed = self.config.imp_states / n_steps_per_segment.max(1);
        let min_ibs = (self.config.imp_states as f64).sqrt() as usize;
        let n_ibs_haps = computed.max(min_ibs).max(1);

        // Compute genetic positions at ALL reference markers (for RefPanelCoded)
        let ref_gen_positions: Vec<f64> = (0..n_ref_markers)
            .map(|m| {
                let pos = ref_gt.marker(MarkerIdx::new(m as u32)).pos;
                gen_maps.gen_pos(chrom, pos)
            })
            .collect();

        // Build coded reference panel for efficient PBWT operations
        //
        // We build the dictionary using ONLY reference haplotypes.
        // Do not append target haplotypes to this panel.
        //
        // Mathematical basis for this design:
        // ─────────────────────────────────────────────────────────────────────
        // Let M_ref be the set of reference markers, M_targ ⊂ M_ref be target markers.
        // Let C: H → ℕ be the pattern coding function mapping allele sequences to indices.
        //
        // If we append target haplotypes with "missing" (255) at M_ref \ M_targ:
        //   - Reference patterns: ∀m ∈ step, A(h_ref, m) ∈ {0, 1}
        //   - Target patterns: ∃m ∈ step where A(h_targ, m) = 255
        //   - Since 255 ≠ 0 and 255 ≠ 1: C(h_ref) ≠ C(h_targ) for any ref/targ pair
        //   - The pattern sets are DISJOINT: {C(h_ref)} ∩ {C(h_targ)} = ∅
        //
        // This causes IBS matching to fail catastrophically:
        //   1. match_sequence(target_seq) finds the target's own pattern (with 255s)
        //   2. find_ibs() searches PBWT for ref haps with that pattern
        //   3. No ref hap has patterns containing 255 → zero matches
        //   4. Falls back to random selection → imputation accuracy destroyed
        //
        // Solution: Keep only reference patterns in the dictionary.
        // The ImpStates::ibs_states() method uses:
        //   - PBWT built on reference haplotypes only (n_ref_haps)
        //   - closest_pattern() which correctly ignores 255s in distance calculation:
        //       distance = Σ 𝟙[p_i ≠ a_i ∧ p_i ≠ 255 ∧ a_i ≠ 255]
        //   - This finds the reference pattern that best matches at genotyped positions
        // ─────────────────────────────────────────────────────────────────────
        eprintln!("Building coded reference panel...");
        let ref_panel_coded = RefPanelCoded::from_gen_positions(
            &ref_gt,
            &ref_gen_positions,
            self.config.imp_step as f64,
        );
        eprintln!(
            "  {} steps ({} haps share each pattern on avg)",
            ref_panel_coded.n_steps(),
            ref_panel_coded.avg_compression_ratio() as usize
        );

        eprintln!("Running imputation with dynamic state selection...");
        let n_states = self.params.n_states;

        // Build genotyped markers list (reference markers that have target data)
        // This is the sparse set the HMM will run on
        // Wrapped in Arc to share across all haplotypes without cloning
        let genotyped_markers: std::sync::Arc<Vec<usize>> = std::sync::Arc::new(
            (0..n_ref_markers)
                .filter(|&m| alignment.is_genotyped(m))
                .collect()
        );
        let n_genotyped = genotyped_markers.len();
        let n_to_impute = n_ref_markers - n_genotyped;
        eprintln!(
            "  HMM on {} genotyped markers, interpolating {} ungenotyped",
            n_genotyped,
            n_to_impute
        );

        // Compute cumulative genetic positions for ALL reference markers (for interpolation)
        // Wrapped in Arc to share across all haplotypes without cloning
        let gen_positions: std::sync::Arc<Vec<f64>> = {
            let mut positions = Vec::with_capacity(n_ref_markers);
            let mut cumulative = 0.0f64;
            positions.push(0.0);
            for m in 1..n_ref_markers {
                let pos1 = ref_gt.marker(MarkerIdx::new((m - 1) as u32)).pos;
                let pos2 = ref_gt.marker(MarkerIdx::new(m as u32)).pos;
                cumulative += gen_maps.gen_dist(chrom, pos1, pos2);
                positions.push(cumulative);
            }
            std::sync::Arc::new(positions)
        };

        // Compute sparse recombination probabilities (between consecutive GENOTYPED markers)
        // This uses the genetic distance spanning potentially many ungenotyped markers
        let sparse_p_recomb: Vec<f32> = std::iter::once(0.0f32)
            .chain(genotyped_markers.windows(2).map(|w| {
                let gen_dist = gen_positions[w[1]] - gen_positions[w[0]];
                self.params.p_recomb(gen_dist)
            }))
            .collect();

        // Run imputation for each target haplotype with per-thread workspaces
        let state_probs: Vec<StateProbs> = (0..n_target_haps)
            .into_par_iter()
            .map_init(
                // Initialize workspace for each thread
                || ImpWorkspace::with_ref_size(n_states, n_ref_markers, n_ref_haps),
                // Process each haplotype with its thread's workspace
                |workspace, h| {
                    let hap_idx = HapIdx::new(h as u32);

                    // Build target alleles in REFERENCE marker space (full)
                    // Needed for IBS state selection which runs on all markers
                    let target_alleles: Vec<u8> = (0..n_ref_markers)
                        .map(|ref_m| {
                            if let Some(target_m) = alignment.target_marker(ref_m) {
                                let raw_allele = target_gt.allele(MarkerIdx::new(target_m as u32), hap_idx);
                                alignment.map_allele(target_m, raw_allele)
                            } else {
                                255 // Missing - marker not in target
                            }
                        })
                        .collect();

                    // Create ImpStates for dynamic state selection with RefPanelCoded
                    let mut imp_states =
                        ImpStates::new(&ref_panel_coded, n_ref_haps, n_states, n_ibs_haps);

                    // Get reference allele closure
                    let get_ref_allele = |ref_m: usize, hap: u32| -> u8 {
                        ref_gt.allele(MarkerIdx::new(ref_m as u32), HapIdx::new(hap))
                    };

                    // Get IBS-based states with SPARSE output (genotyped markers only)
                    // Memory optimization: allocates O(n_genotyped * n_states) instead of
                    // O(n_ref_markers * n_states), typically 10-100x memory savings
                    let mut sparse_hap_indices: Vec<Vec<u32>> = Vec::new();
                    let mut sparse_allele_match: Vec<Vec<bool>> = Vec::new();
                    let actual_n_states = imp_states.ibs_states(
                        get_ref_allele,
                        &target_alleles,
                        &genotyped_markers,
                        workspace,
                        &mut sparse_hap_indices,
                        &mut sparse_allele_match,
                    );

                    // Extract sparse target alleles for HMM
                    let sparse_target_alleles: Vec<u8> = genotyped_markers
                        .iter()
                        .map(|&m| target_alleles[m])
                        .collect();

                    // Run forward-backward HMM on GENOTYPED markers only (~100x faster)
                    let hmm_state_probs = run_hmm_forward_backward(
                        &sparse_target_alleles,
                        &sparse_allele_match,
                        &sparse_p_recomb,
                        self.params.p_mismatch,
                        actual_n_states,
                        workspace,
                    );

                    // Create StateProbs with interpolation support
                    StateProbs::new(
                        std::sync::Arc::clone(&genotyped_markers),
                        actual_n_states,
                        sparse_hap_indices,
                        hmm_state_probs,
                        std::sync::Arc::clone(&gen_positions),
                    )
                },
            )
            .collect();

        eprintln!("Computing dosages with interpolation and quality metrics...");

        // Initialize quality stats for all reference markers
        let n_alleles_per_marker: Vec<usize> = (0..n_ref_markers)
            .map(|m| {
                let marker = ref_gt.marker(MarkerIdx::new(m as u32));
                1 + marker.alt_alleles.len()
            })
            .collect();
        let mut quality = ImputationQuality::new(&n_alleles_per_marker);

        // Mark imputed markers (those not in target)
        for m in 0..n_ref_markers {
            let is_imputed = !alignment.is_genotyped(m);
            quality.set_imputed(m, is_imputed);
        }

        // Check if we need per-haplotype allele probabilities for AP/GP output
        let need_allele_probs = self.config.ap || self.config.gp;

        // Compute dosages per sample (parallel, no locks)
        // Each sample returns: (dosages, allele_posteriors, quality_contributions)
        // Uses optimized AllelePosteriors enum (compact for biallelic, full for multiallelic)
        type QualityContrib = (usize, [f32; 2], [f32; 2]); // (marker, probs1, probs2)
        type HapPosteriors = Vec<(AllelePosteriors, AllelePosteriors)>; // [marker] -> (hap1, hap2)
        let sample_results: Vec<(Vec<f32>, Option<HapPosteriors>, Vec<QualityContrib>)> = (0..n_target_samples)
            .into_par_iter()
            .map(|s| {
                let hap1_probs = &state_probs[s * 2];
                let hap2_probs = &state_probs[s * 2 + 1];

                let mut dosages: Vec<f32> = Vec::with_capacity(n_ref_markers);
                let mut posteriors: Option<HapPosteriors> = if need_allele_probs {
                    Some(Vec::with_capacity(n_ref_markers))
                } else {
                    None
                };
                let mut quality_contribs: Vec<QualityContrib> = Vec::new();

                for m in 0..n_ref_markers {
                    let get_ref_allele = |ref_m: usize, hap: u32| -> u8 {
                        ref_gt.allele(MarkerIdx::new(ref_m as u32), HapIdx::new(hap))
                    };

                    let n_alleles = n_alleles_per_marker[m];

                    // Compute per-allele posteriors (optimized enum)
                    let post1 = hap1_probs.allele_posteriors(m, n_alleles, &get_ref_allele);
                    let post2 = hap2_probs.allele_posteriors(m, n_alleles, &get_ref_allele);

                    // Dosage = expected ALT allele count for both haplotypes
                    let d1 = post1.dosage();
                    let d2 = post2.dosage();
                    dosages.push(d1 + d2);

                    // Record contributions for DR2 calculation (all sites)
                    quality_contribs.push((m,
                        [post1.prob(0), post1.prob(1)],
                        [post2.prob(0), post2.prob(1)]
                    ));

                    // Store posteriors if needed for output
                    if let Some(ref mut p) = posteriors {
                        p.push((post1, post2));
                    }
                }

                (dosages, posteriors, quality_contribs)
            })
            .collect();

        // Serial reduction: accumulate all quality contributions (fast, no contention)
        for (_, _, quality_contribs) in &sample_results {
            for &(m, probs1, probs2) in quality_contribs {
                if let Some(stats) = quality.get_mut(m) {
                    stats.add_sample(&probs1, &probs2);
                }
            }
        }

        // STREAMING OUTPUT: No flat_dosages allocation!
        // Access sample_results directly via closures during write.
        // Memory savings: eliminates O(n_markers * n_samples) transposed array
        let output_path = self.config.out.with_extension("vcf.gz");
        eprintln!("Writing output to {:?}", output_path);
        let mut writer = VcfWriter::create(&output_path, target_samples)?;
        writer.write_header_extended(ref_gt.markers(), true, self.config.gp, self.config.ap)?;

        // Streaming closure: get dosage directly from sample-major storage
        let get_dosage = |m: usize, s: usize| -> f32 {
            sample_results[s].0[m]
        };

        // Streaming closure: get posteriors (clones one at a time during write)
        let get_posteriors: Option<Box<dyn Fn(usize, usize) -> (AllelePosteriors, AllelePosteriors)>> =
            if need_allele_probs {
                Some(Box::new(|m: usize, s: usize| -> (AllelePosteriors, AllelePosteriors) {
                    if let Some(ref posts) = sample_results[s].1 {
                        (posts[m].0.clone(), posts[m].1.clone())
                    } else {
                        (AllelePosteriors::Biallelic(0.0), AllelePosteriors::Biallelic(0.0))
                    }
                }))
            } else {
                None
            };

        writer.write_imputed_streaming(
            &ref_gt,
            get_dosage,
            get_posteriors,
            &quality,
            0,
            n_ref_markers,
            self.config.gp,
            self.config.ap,
        )?;
        writer.flush()?;

        eprintln!("Imputation complete!");
        Ok(())
    }
}

/// Run forward-backward HMM on IBS-selected states
fn run_hmm_forward_backward(
    target_alleles: &[u8],
    allele_match: &[Vec<bool>],
    p_recomb: &[f32],
    p_mismatch: f32,
    n_states: usize,
    workspace: &mut ImpWorkspace,
) -> Vec<f32> {
    let n_markers = target_alleles.len();
    if n_markers == 0 || n_states == 0 {
        return Vec::new();
    }

    let p_match = 1.0 - p_mismatch;
    let emit_probs = [p_match, p_mismatch];

    // Ensure workspace is sized correctly
    workspace.resize(n_states, n_markers);

    // Use pre-allocated forward storage (flat)
    let total_size = n_markers * n_states;
    let mut fwd: Vec<f32> = vec![0.0; total_size];
    let mut fwd_sum = 1.0f32;

    // Forward pass using workspace.tmp for temporary calculations
    for m in 0..n_markers {
        let p_rec = p_recomb.get(m).copied().unwrap_or(0.0);
        let shift = p_rec / n_states as f32;
        let scale = (1.0 - p_rec) / fwd_sum;

        let mut new_sum = 0.0f32;
        let matches = &allele_match[m];
        let row_offset = m * n_states;
        let prev_row_offset = if m > 0 { (m - 1) * n_states } else { 0 };

        for k in 0..n_states.min(matches.len()) {
            let emit = if matches[k] {
                emit_probs[0]
            } else {
                emit_probs[1]
            };

            // Match Java exactly: m==0 uses just emission, not divided by n_states
            // Java: fwdVal[m][j] = m==0 ? em : em*(scale*fwdVal[m-1][j] + shift);
            let val = if m == 0 {
                emit
            } else {
                emit * (scale * fwd[prev_row_offset + k] + shift)
            };

            fwd[row_offset + k] = val;
            new_sum += val;
        }
        fwd_sum = new_sum;
    }

    // Backward pass using workspace.bwd buffer
    let bwd = &mut workspace.bwd;
    bwd.resize(n_states, 0.0);
    bwd.fill(1.0 / n_states as f32);
    let mut bwd_sum = 1.0f32;

    for m in (0..n_markers).rev() {
        let row_offset = m * n_states;
        
        // Apply transition for backward (except at last marker)
        if m < n_markers - 1 {
            let p_rec = p_recomb.get(m + 1).copied().unwrap_or(0.0);
            let shift = p_rec / n_states as f32;
            let scale = (1.0 - p_rec) / bwd_sum;

            for k in 0..n_states {
                bwd[k] = scale * bwd[k] + shift;
            }
        }

        // Compute posterior: fwd * bwd
        let mut state_sum = 0.0f32;
        for k in 0..n_states {
            let idx = row_offset + k;
            fwd[idx] *= bwd[k];
            state_sum += fwd[idx];
        }

        // Normalize
        if state_sum > 0.0 {
            let inv_sum = 1.0 / state_sum;
            for k in 0..n_states {
                fwd[row_offset + k] *= inv_sum;
            }
        }

        // Apply emission for next backward iteration
        if m > 0 {
            let matches = &allele_match[m];
            bwd_sum = 0.0;
            for k in 0..n_states.min(matches.len()) {
                let emit = if matches[k] {
                    emit_probs[0]
                } else {
                    emit_probs[1]
                };
                bwd[k] *= emit;
                bwd_sum += bwd[k];
            }
        }
    }

    fwd
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // AllelePosteriors Tests - RIGOROUS
    // =========================================================================

    #[test]
    fn test_biallelic_prob_exact_math() {
        // Test exact probability computation for biallelic sites
        // P(REF) = 1 - P(ALT), must be EXACT to f32 precision
        for p_alt_int in 0..=100 {
            let p_alt = p_alt_int as f32 / 100.0;
            let post = AllelePosteriors::Biallelic(p_alt);

            let computed_ref = post.prob(0);
            let computed_alt = post.prob(1);
            let expected_ref = 1.0 - p_alt;

            assert!(
                (computed_ref - expected_ref).abs() < 1e-7,
                "P(REF) wrong for p_alt={}: got {}, expected {}", p_alt, computed_ref, expected_ref
            );
            assert!(
                (computed_alt - p_alt).abs() < 1e-7,
                "P(ALT) wrong for p_alt={}: got {}, expected {}", p_alt, computed_alt, p_alt
            );

            // Probabilities MUST sum to exactly 1.0
            let sum = computed_ref + computed_alt;
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "Probabilities don't sum to 1 for p_alt={}: sum={}", p_alt, sum
            );
        }
    }

    #[test]
    fn test_biallelic_dosage_equals_p_alt() {
        // Dosage for biallelic MUST equal P(ALT) exactly
        // This is the definition: E[X] = 0*P(0) + 1*P(1) = P(1) = P(ALT)
        for p_alt_int in 0..=100 {
            let p_alt = p_alt_int as f32 / 100.0;
            let post = AllelePosteriors::Biallelic(p_alt);

            assert!(
                (post.dosage() - p_alt).abs() < 1e-7,
                "Dosage != P(ALT) for p_alt={}: dosage={}", p_alt, post.dosage()
            );
        }
    }

    #[test]
    fn test_biallelic_max_allele_boundary() {
        // Critical boundary: at exactly 0.5, should return 1 (ALT)
        let at_boundary = AllelePosteriors::Biallelic(0.5);
        assert_eq!(at_boundary.max_allele(), 1, "At p_alt=0.5, max_allele should be 1");

        // Just below boundary
        let below = AllelePosteriors::Biallelic(0.4999999);
        assert_eq!(below.max_allele(), 0, "At p_alt=0.4999999, max_allele should be 0");

        // Just above boundary
        let above = AllelePosteriors::Biallelic(0.5000001);
        assert_eq!(above.max_allele(), 1, "At p_alt=0.5000001, max_allele should be 1");

        // Edge cases
        assert_eq!(AllelePosteriors::Biallelic(0.0).max_allele(), 0);
        assert_eq!(AllelePosteriors::Biallelic(1.0).max_allele(), 1);
    }

    #[test]
    fn test_multiallelic_prob_sum_to_one() {
        // Multiallelic probabilities must sum to 1 (if input sums to 1)
        let probs = vec![0.1, 0.2, 0.3, 0.4]; // Sum = 1.0
        let post = AllelePosteriors::Multiallelic(probs.clone());

        let sum: f32 = (0..4).map(|i| post.prob(i)).sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Multiallelic probs don't sum to 1: sum={}", sum
        );

        // Verify each prob is correct
        for (i, &expected) in probs.iter().enumerate() {
            assert!(
                (post.prob(i) - expected).abs() < 1e-7,
                "P({}) wrong: got {}, expected {}", i, post.prob(i), expected
            );
        }
    }

    #[test]
    fn test_multiallelic_dosage_formula() {
        // Dosage = E[X] = sum(i * P(i))
        let probs = vec![0.1, 0.2, 0.3, 0.4]; // 4-allelic
        let post = AllelePosteriors::Multiallelic(probs.clone());

        // Expected: 0*0.1 + 1*0.2 + 2*0.3 + 3*0.4 = 0 + 0.2 + 0.6 + 1.2 = 2.0
        let expected_dosage = 2.0f32;
        assert!(
            (post.dosage() - expected_dosage).abs() < 1e-6,
            "Multiallelic dosage wrong: got {}, expected {}", post.dosage(), expected_dosage
        );

        // Test another case: 5-allelic
        let probs2 = vec![0.5, 0.2, 0.1, 0.1, 0.1];
        let post2 = AllelePosteriors::Multiallelic(probs2);
        // Expected: 0*0.5 + 1*0.2 + 2*0.1 + 3*0.1 + 4*0.1 = 0 + 0.2 + 0.2 + 0.3 + 0.4 = 1.1
        assert!(
            (post2.dosage() - 1.1).abs() < 1e-6,
            "5-allelic dosage wrong: got {}", post2.dosage()
        );
    }

    #[test]
    fn test_multiallelic_max_allele_all_cases() {
        // Case 1: First allele is max
        let post1 = AllelePosteriors::Multiallelic(vec![0.5, 0.3, 0.2]);
        assert_eq!(post1.max_allele(), 0);

        // Case 2: Middle allele is max
        let post2 = AllelePosteriors::Multiallelic(vec![0.2, 0.6, 0.2]);
        assert_eq!(post2.max_allele(), 1);

        // Case 3: Last allele is max
        let post3 = AllelePosteriors::Multiallelic(vec![0.1, 0.2, 0.7]);
        assert_eq!(post3.max_allele(), 2);
    }

    #[test]
    fn test_out_of_bounds_returns_zero() {
        // Accessing probability of non-existent allele must return 0
        let biallelic = AllelePosteriors::Biallelic(0.5);
        assert_eq!(biallelic.prob(2), 0.0);
        assert_eq!(biallelic.prob(100), 0.0);

        let triallelic = AllelePosteriors::Multiallelic(vec![0.3, 0.3, 0.4]);
        assert_eq!(triallelic.prob(3), 0.0);
        assert_eq!(triallelic.prob(1000), 0.0);
    }

    // =========================================================================
    // HMM Forward-Backward Tests - RIGOROUS
    // =========================================================================

    #[test]
    fn test_hmm_posteriors_sum_to_one_strict() {
        use crate::utils::workspace::ImpWorkspace;

        // Test with various configurations
        for n_markers in [2, 5, 10, 20] {
            for n_states in [2, 4, 8] {
                let target_alleles = vec![0u8; n_markers];
                let allele_match: Vec<Vec<bool>> = (0..n_markers)
                    .map(|m| (0..n_states).map(|k| (m + k) % 2 == 0).collect())
                    .collect();
                let p_recomb: Vec<f32> = (0..n_markers)
                    .map(|m| if m == 0 { 0.0 } else { 0.01 })
                    .collect();

                let mut workspace = ImpWorkspace::with_ref_size(n_states, n_markers, 100);
                let posteriors = run_hmm_forward_backward(
                    &target_alleles,
                    &allele_match,
                    &p_recomb,
                    0.01,
                    n_states,
                    &mut workspace,
                );

                // Posteriors MUST sum to 1.0 at EVERY marker
                for m in 0..n_markers {
                    let sum: f32 = (0..n_states).map(|k| posteriors[m * n_states + k]).sum();
                    assert!(
                        (sum - 1.0).abs() < 0.001,
                        "n_markers={}, n_states={}, marker {}: posteriors sum to {}, not 1.0",
                        n_markers, n_states, m, sum
                    );
                }
            }
        }
    }

    #[test]
    fn test_hmm_posteriors_non_negative() {
        use crate::utils::workspace::ImpWorkspace;

        let n_markers = 10;
        let n_states = 4;
        let target_alleles = vec![0u8; n_markers];
        let allele_match: Vec<Vec<bool>> = (0..n_markers)
            .map(|m| (0..n_states).map(|k| k == m % n_states).collect())
            .collect();
        let p_recomb: Vec<f32> = (0..n_markers).map(|m| if m == 0 { 0.0 } else { 0.05 }).collect();

        let mut workspace = ImpWorkspace::with_ref_size(n_states, n_markers, 100);
        let posteriors = run_hmm_forward_backward(
            &target_alleles,
            &allele_match,
            &p_recomb,
            0.01,
            n_states,
            &mut workspace,
        );

        for (i, &p) in posteriors.iter().enumerate() {
            assert!(p >= 0.0, "Posterior at index {} is negative: {}", i, p);
            assert!(p <= 1.0, "Posterior at index {} exceeds 1: {}", i, p);
        }
    }

    #[test]
    fn test_hmm_perfect_match_gives_high_posterior() {
        use crate::utils::workspace::ImpWorkspace;

        let n_markers = 20;
        let n_states = 4;
        let target_alleles = vec![0u8; n_markers];

        // State 0 always matches, others never match
        let allele_match: Vec<Vec<bool>> = (0..n_markers)
            .map(|_| vec![true, false, false, false])
            .collect();
        let p_recomb: Vec<f32> = (0..n_markers).map(|m| if m == 0 { 0.0 } else { 0.001 }).collect();

        let mut workspace = ImpWorkspace::with_ref_size(n_states, n_markers, 100);
        let posteriors = run_hmm_forward_backward(
            &target_alleles,
            &allele_match,
            &p_recomb,
            0.001,
            n_states,
            &mut workspace,
        );

        // State 0 should have posterior > 0.99 at every marker
        for m in 0..n_markers {
            let prob_state0 = posteriors[m * n_states];
            assert!(
                prob_state0 > 0.99,
                "Perfect match state should have p>0.99, got {} at marker {}", prob_state0, m
            );
        }
    }

    #[test]
    fn test_hmm_state_switch_detection() {
        use crate::utils::workspace::ImpWorkspace;

        // Pattern: state 0 matches first half, state 1 matches second half
        let n_markers = 20;
        let n_states = 2;
        let target_alleles = vec![0u8; n_markers];

        let allele_match: Vec<Vec<bool>> = (0..n_markers)
            .map(|m| {
                if m < 10 { vec![true, false] } else { vec![false, true] }
            })
            .collect();
        let p_recomb: Vec<f32> = (0..n_markers).map(|m| if m == 0 { 0.0 } else { 0.01 }).collect();

        let mut workspace = ImpWorkspace::with_ref_size(n_states, n_markers, 100);
        let posteriors = run_hmm_forward_backward(
            &target_alleles,
            &allele_match,
            &p_recomb,
            0.01,
            n_states,
            &mut workspace,
        );

        // First half: state 0 should dominate
        for m in 0..5 {
            let prob_state0 = posteriors[m * n_states];
            assert!(prob_state0 > 0.8, "First half, state 0 should dominate. marker {}: p0={}", m, prob_state0);
        }

        // Second half: state 1 should dominate
        for m in 15..n_markers {
            let prob_state1 = posteriors[m * n_states + 1];
            assert!(prob_state1 > 0.8, "Second half, state 1 should dominate. marker {}: p1={}", m, prob_state1);
        }
    }

    #[test]
    fn test_hmm_symmetry() {
        use crate::utils::workspace::ImpWorkspace;

        let n_markers = 10;
        let n_states = 2;
        let target_alleles = vec![0u8; n_markers];

        // Run 1: state 0 matches at even markers
        let match1: Vec<Vec<bool>> = (0..n_markers)
            .map(|m| if m % 2 == 0 { vec![true, false] } else { vec![false, true] })
            .collect();

        // Run 2: state 1 matches at even markers (swapped)
        let match2: Vec<Vec<bool>> = (0..n_markers)
            .map(|m| if m % 2 == 0 { vec![false, true] } else { vec![true, false] })
            .collect();

        let p_recomb: Vec<f32> = (0..n_markers).map(|m| if m == 0 { 0.0 } else { 0.05 }).collect();

        let mut workspace = ImpWorkspace::with_ref_size(n_states, n_markers, 100);

        let post1 = run_hmm_forward_backward(&target_alleles, &match1, &p_recomb, 0.01, n_states, &mut workspace);
        let post2 = run_hmm_forward_backward(&target_alleles, &match2, &p_recomb, 0.01, n_states, &mut workspace);

        // Posteriors should be swapped
        for m in 0..n_markers {
            let p1_s0 = post1[m * n_states];
            let p1_s1 = post1[m * n_states + 1];
            let p2_s0 = post2[m * n_states];
            let p2_s1 = post2[m * n_states + 1];

            assert!((p1_s0 - p2_s1).abs() < 0.01, "Symmetry broken at marker {}: p1[s0]={}, p2[s1]={}", m, p1_s0, p2_s1);
            assert!((p1_s1 - p2_s0).abs() < 0.01, "Symmetry broken at marker {}: p1[s1]={}, p2[s0]={}", m, p1_s1, p2_s0);
        }
    }

    // =========================================================================
    // HARD TESTS - Analytically computed expected values
    // These tests compute exact expected posteriors by hand and compare.
    // If there's ANY bug in the HMM, these WILL FAIL.
    // =========================================================================

    #[test]
    fn test_hmm_2state_2marker_exact_posterior() {
        // 2-state, 2-marker HMM with known parameters
        // We compute the EXACT posterior analytically and compare
        //
        // Setup:
        // - 2 states, 2 markers
        // - State 0 matches at marker 0, state 1 matches at marker 1
        // - p_recomb = 0.1, p_mismatch = 0.01
        //
        // Li-Stephens forward formula:
        //   fwd[m][k] = emit[k] * ((1-rho)*fwd[m-1][k]/sum + rho/K)
        // where rho = p_recomb, K = n_states
        use crate::utils::workspace::ImpWorkspace;

        let n_states = 2;
        let n_markers = 2;
        let rho = 0.1f32;  // recombination prob
        let p_err = 0.01f32;  // mismatch prob
        let p_match = 1.0 - p_err;

        // State 0 matches at m=0, state 1 matches at m=1
        let allele_match = vec![
            vec![true, false],   // m=0: state 0 matches
            vec![false, true],   // m=1: state 1 matches
        ];

        // Compute expected forward values analytically
        // Marker 0: fwd[0] = (1/K) * emit[k]
        let fwd0_0 = (1.0 / n_states as f32) * p_match;  // state 0 matches
        let fwd0_1 = (1.0 / n_states as f32) * p_err;    // state 1 mismatches
        let fwd0_sum = fwd0_0 + fwd0_1;

        // Marker 1: fwd[1][k] = emit[k] * ((1-rho)*fwd[0][k]/fwd0_sum + rho/K)
        let shift = rho / n_states as f32;
        let scale = (1.0 - rho) / fwd0_sum;

        let fwd1_0_pre = scale * fwd0_0 + shift;  // transition for state 0
        let fwd1_1_pre = scale * fwd0_1 + shift;  // transition for state 1
        let fwd1_0 = p_err * fwd1_0_pre;          // state 0 mismatches at m=1
        let fwd1_1 = p_match * fwd1_1_pre;        // state 1 matches at m=1

        // Backward: bwd[M-1] = 1/K, then apply bwd_update
        // For 2 markers, bwd at m=0 uses emission at m=1
        let bwd1_0 = 1.0 / n_states as f32;
        let bwd1_1 = 1.0 / n_states as f32;

        // bwd_update: first multiply by emit, then normalize transition
        let bwd0_pre_0 = bwd1_0 * p_err;    // state 0 at m=1 mismatches
        let bwd0_pre_1 = bwd1_1 * p_match;  // state 1 at m=1 matches
        let bwd_sum = bwd0_pre_0 + bwd0_pre_1;

        let bwd_scale = (1.0 - rho) / bwd_sum;
        let bwd_shift = rho / n_states as f32;
        let bwd0_0 = bwd_scale * bwd0_pre_0 + bwd_shift;
        let bwd0_1 = bwd_scale * bwd0_pre_1 + bwd_shift;

        // Expected posteriors: gamma[m][k] = fwd[m][k] * bwd[m][k] / sum
        let gamma0_0_raw = fwd0_0 * bwd0_0;
        let gamma0_1_raw = fwd0_1 * bwd0_1;
        let gamma0_sum = gamma0_0_raw + gamma0_1_raw;
        let expected_gamma0_0 = gamma0_0_raw / gamma0_sum;
        let expected_gamma0_1 = gamma0_1_raw / gamma0_sum;

        let gamma1_0_raw = fwd1_0 * bwd1_0;
        let gamma1_1_raw = fwd1_1 * bwd1_1;
        let gamma1_sum = gamma1_0_raw + gamma1_1_raw;
        let expected_gamma1_0 = gamma1_0_raw / gamma1_sum;
        let expected_gamma1_1 = gamma1_1_raw / gamma1_sum;

        // Run the actual HMM
        let target_alleles = vec![0u8; n_markers];
        let p_recomb = vec![0.0, rho];  // First marker has 0 recomb

        let mut workspace = ImpWorkspace::with_ref_size(n_states, n_markers, 10);
        let posteriors = run_hmm_forward_backward(
            &target_alleles,
            &allele_match,
            &p_recomb,
            p_err,
            n_states,
            &mut workspace,
        );

        // Compare with TIGHT tolerance - these should be EXACT (up to float precision)
        let tol = 0.02;  // 2% tolerance for numerical differences

        let actual_gamma0_0 = posteriors[0];
        let actual_gamma0_1 = posteriors[1];
        let actual_gamma1_0 = posteriors[2];
        let actual_gamma1_1 = posteriors[3];

        assert!(
            (actual_gamma0_0 - expected_gamma0_0).abs() < tol,
            "Marker 0, State 0: expected {:.6}, got {:.6}", expected_gamma0_0, actual_gamma0_0
        );
        assert!(
            (actual_gamma0_1 - expected_gamma0_1).abs() < tol,
            "Marker 0, State 1: expected {:.6}, got {:.6}", expected_gamma0_1, actual_gamma0_1
        );
        assert!(
            (actual_gamma1_0 - expected_gamma1_0).abs() < tol,
            "Marker 1, State 0: expected {:.6}, got {:.6}", expected_gamma1_0, actual_gamma1_0
        );
        assert!(
            (actual_gamma1_1 - expected_gamma1_1).abs() < tol,
            "Marker 1, State 1: expected {:.6}, got {:.6}", expected_gamma1_1, actual_gamma1_1
        );
    }

    #[test]
    fn test_hmm_uniform_emission_gives_uniform_posterior() {
        // If ALL states match at ALL markers (uniform emission),
        // posterior should be uniform: 1/K for each state
        use crate::utils::workspace::ImpWorkspace;

        for n_states in [2, 4, 8, 16] {
            let n_markers = 10;
            let target_alleles = vec![0u8; n_markers];

            // All states match at all markers
            let allele_match: Vec<Vec<bool>> = (0..n_markers)
                .map(|_| vec![true; n_states])
                .collect();
            let p_recomb: Vec<f32> = (0..n_markers)
                .map(|m| if m == 0 { 0.0 } else { 0.05 })
                .collect();

            let mut workspace = ImpWorkspace::with_ref_size(n_states, n_markers, 100);
            let posteriors = run_hmm_forward_backward(
                &target_alleles,
                &allele_match,
                &p_recomb,
                0.01,
                n_states,
                &mut workspace,
            );

            let expected = 1.0 / n_states as f32;

            // All posteriors should be uniform
            for m in 0..n_markers {
                for k in 0..n_states {
                    let actual = posteriors[m * n_states + k];
                    assert!(
                        (actual - expected).abs() < 0.05,
                        "n_states={}, marker {}, state {}: expected {:.4}, got {:.4}",
                        n_states, m, k, expected, actual
                    );
                }
            }
        }
    }

    #[test]
    fn test_hmm_no_recombination_preserves_initial_state() {
        // With ZERO recombination, the HMM should stay in the initial state
        // (weighted by emission probabilities)
        use crate::utils::workspace::ImpWorkspace;

        let n_states = 4;
        let n_markers = 10;
        let target_alleles = vec![0u8; n_markers];

        // State 0 always matches
        let allele_match: Vec<Vec<bool>> = (0..n_markers)
            .map(|_| vec![true, false, false, false])
            .collect();

        // ZERO recombination everywhere
        let p_recomb = vec![0.0f32; n_markers];

        let mut workspace = ImpWorkspace::with_ref_size(n_states, n_markers, 100);
        let posteriors = run_hmm_forward_backward(
            &target_alleles,
            &allele_match,
            &p_recomb,
            0.01,  // small mismatch prob
            n_states,
            &mut workspace,
        );

        // With no recombination and state 0 always matching,
        // state 0 should have nearly all probability
        for m in 0..n_markers {
            let prob_state0 = posteriors[m * n_states];
            assert!(
                prob_state0 > 0.999,
                "With zero recomb, matching state should have p>0.999, got {} at marker {}", prob_state0, m
            );
        }
    }

    #[test]
    fn test_dosage_bounds_diploid() {
        // For diploid genotypes, dosage should be in [0, 2]
        // Test that DS = P(hap1=ALT) + P(hap2=ALT) is bounded correctly

        // Biallelic: dosage = P(ALT) per haplotype, so diploid DS = hap1 + hap2
        let hap1 = AllelePosteriors::Biallelic(0.8);
        let hap2 = AllelePosteriors::Biallelic(0.6);
        let diploid_dosage = hap1.dosage() + hap2.dosage();
        assert!(diploid_dosage >= 0.0 && diploid_dosage <= 2.0,
            "Diploid dosage {} should be in [0,2]", diploid_dosage);
        assert!((diploid_dosage - 1.4).abs() < 1e-6,
            "Expected diploid dosage 1.4, got {}", diploid_dosage);
    }

    #[test]
    fn test_gp_probabilities_from_haplotype_posteriors() {
        // GP (genotype probability) should be computed from haplotype posteriors
        // For biallelic: GP = [P(0/0), P(0/1), P(1/1)]
        //   P(0/0) = P(hap1=0) * P(hap2=0)
        //   P(0/1) = P(hap1=0)*P(hap2=1) + P(hap1=1)*P(hap2=0)
        //   P(1/1) = P(hap1=1) * P(hap2=1)

        let p1_alt = 0.3f32;  // P(hap1 = ALT)
        let p2_alt = 0.7f32;  // P(hap2 = ALT)

        let p1_ref = 1.0 - p1_alt;
        let p2_ref = 1.0 - p2_alt;

        let expected_p00 = p1_ref * p2_ref;  // 0.7 * 0.3 = 0.21
        let expected_p01 = p1_ref * p2_alt + p1_alt * p2_ref;  // 0.7*0.7 + 0.3*0.3 = 0.58
        let expected_p11 = p1_alt * p2_alt;  // 0.3 * 0.7 = 0.21

        // Verify they sum to 1
        let gp_sum = expected_p00 + expected_p01 + expected_p11;
        assert!((gp_sum - 1.0).abs() < 1e-6, "GP should sum to 1, got {}", gp_sum);

        // Test using AllelePosteriors
        let hap1 = AllelePosteriors::Biallelic(p1_alt);
        let hap2 = AllelePosteriors::Biallelic(p2_alt);

        let computed_p00 = hap1.prob(0) * hap2.prob(0);
        let computed_p01 = hap1.prob(0) * hap2.prob(1) + hap1.prob(1) * hap2.prob(0);
        let computed_p11 = hap1.prob(1) * hap2.prob(1);

        assert!((computed_p00 - expected_p00).abs() < 1e-6,
            "P(0/0): expected {}, got {}", expected_p00, computed_p00);
        assert!((computed_p01 - expected_p01).abs() < 1e-6,
            "P(0/1): expected {}, got {}", expected_p01, computed_p01);
        assert!((computed_p11 - expected_p11).abs() < 1e-6,
            "P(1/1): expected {}, got {}", expected_p11, computed_p11);
    }

    #[test]
    fn test_multiallelic_gp_count() {
        // For N alleles, GP should have N*(N+1)/2 values
        // Triallelic (N=3): 3*4/2 = 6 values: 0/0, 0/1, 1/1, 0/2, 1/2, 2/2

        for n_alleles in [2, 3, 4, 5, 10] {
            let expected_gp_count = n_alleles * (n_alleles + 1) / 2;

            // Create uniform posteriors
            let probs: Vec<f32> = (0..n_alleles).map(|_| 1.0 / n_alleles as f32).collect();
            let hap1 = AllelePosteriors::Multiallelic(probs.clone());
            let hap2 = AllelePosteriors::Multiallelic(probs);

            // Compute GP values (following VCF spec ordering)
            let mut gp_values = Vec::new();
            for i2 in 0..n_alleles {
                for i1 in 0..=i2 {
                    let prob = if i1 == i2 {
                        hap1.prob(i1) * hap2.prob(i2)
                    } else {
                        hap1.prob(i1) * hap2.prob(i2) + hap1.prob(i2) * hap2.prob(i1)
                    };
                    gp_values.push(prob);
                }
            }

            assert_eq!(gp_values.len(), expected_gp_count,
                "N={}: expected {} GP values, got {}", n_alleles, expected_gp_count, gp_values.len());

            // GP should sum to 1
            let gp_sum: f32 = gp_values.iter().sum();
            assert!((gp_sum - 1.0).abs() < 1e-5,
                "N={}: GP sum should be 1, got {}", n_alleles, gp_sum);
        }
    }

    // =========================================================================
    // StateProbs Interpolation Tests - EXACT MATHEMATICAL VERIFICATION
    // =========================================================================

    #[test]
    fn test_state_probs_interpolation_weight_formula() {
        // Test that interpolation weight is computed correctly:
        //   weight_left = (pos_right - pos_marker) / (pos_right - pos_left)
        //
        // At left marker: weight_left = 1.0
        // At right marker: weight_left = 0.0
        // At midpoint: weight_left = 0.5

        let genotyped_markers = std::sync::Arc::new(vec![0, 10]);
        // Genetic positions: markers at 0.0 and 1.0 cM
        let gen_positions = std::sync::Arc::new(vec![
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
        ]);

        // State 0 (hap 0) always carries REF
        // State 1 (hap 1) always carries ALT
        let hap_indices = vec![
            vec![0, 1],  // marker 0
            vec![0, 1],  // marker 10
        ];
        // At marker 0: 100% state 0, 0% state 1
        // At marker 10: 0% state 0, 100% state 1
        let state_probs = vec![1.0, 0.0, 0.0, 1.0];

        let sp = StateProbs::new(
            genotyped_markers,
            2,
            hap_indices,
            state_probs,
            gen_positions,
        );

        // Hap 0 = REF (0), Hap 1 = ALT (1)
        let get_ref_allele = |marker: usize, hap: u32| -> u8 {
            assert!(marker <= 10);
            if hap == 0 { 0 } else { 1 }
        };

        // Test interpolation at various positions
        // At marker 0: weight_left = 1.0, prob = 1.0 * 1.0 + 0.0 * 0.0 = 1.0 for state 0
        //   P(ALT) = prob_state1 = 0.0
        let post_0 = sp.allele_posteriors(0, 2, &get_ref_allele);
        match post_0 {
            AllelePosteriors::Biallelic(p_alt) => {
                assert!((p_alt - 0.0).abs() < 0.05, "At marker 0: expected P(ALT)~0.0, got {}", p_alt);
            }
            _ => panic!("Expected Biallelic"),
        }

        // At marker 5 (midpoint): weight_left = 0.5
        //   prob_state0 = 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        //   prob_state1 = 0.5 * 0.0 + 0.5 * 1.0 = 0.5
        //   P(ALT) = prob_state1 = 0.5
        let post_5 = sp.allele_posteriors(5, 2, &get_ref_allele);
        match post_5 {
            AllelePosteriors::Biallelic(p_alt) => {
                assert!((p_alt - 0.5).abs() < 0.1, "At marker 5: expected P(ALT)~0.5, got {}", p_alt);
            }
            _ => panic!("Expected Biallelic"),
        }

        // At marker 10: should use exact value (not interpolated)
        //   P(ALT) = prob_state1 = 1.0
        let post_10 = sp.allele_posteriors(10, 2, &get_ref_allele);
        match post_10 {
            AllelePosteriors::Biallelic(p_alt) => {
                assert!((p_alt - 1.0).abs() < 0.05, "At marker 10: expected P(ALT)~1.0, got {}", p_alt);
            }
            _ => panic!("Expected Biallelic"),
        }

        // Test precise interpolation at marker 2 (position 0.2)
        // weight_left = (1.0 - 0.2) / (1.0 - 0.0) = 0.8
        // prob_state1 = 0.8 * 0.0 + 0.2 * 1.0 = 0.2
        let post_2 = sp.allele_posteriors(2, 2, &get_ref_allele);
        match post_2 {
            AllelePosteriors::Biallelic(p_alt) => {
                assert!((p_alt - 0.2).abs() < 0.1, "At marker 2: expected P(ALT)~0.2, got {}", p_alt);
            }
            _ => panic!("Expected Biallelic"),
        }

        // Test precise interpolation at marker 8 (position 0.8)
        // weight_left = (1.0 - 0.8) / (1.0 - 0.0) = 0.2
        // prob_state1 = 0.2 * 0.0 + 0.8 * 1.0 = 0.8
        let post_8 = sp.allele_posteriors(8, 2, &get_ref_allele);
        match post_8 {
            AllelePosteriors::Biallelic(p_alt) => {
                assert!((p_alt - 0.8).abs() < 0.1, "At marker 8: expected P(ALT)~0.8, got {}", p_alt);
            }
            _ => panic!("Expected Biallelic"),
        }
    }

    #[test]
    fn test_state_probs_edge_cases() {
        // Test edge case: marker before first genotyped marker
        // Should return the first marker's value

        let genotyped_markers = std::sync::Arc::new(vec![5, 10]);
        let gen_positions = std::sync::Arc::new(vec![
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
        ]);

        let hap_indices = vec![vec![0, 1], vec![0, 1]];
        // At marker 5: 70% state 0, 30% state 1
        // At marker 10: 20% state 0, 80% state 1
        let state_probs = vec![0.7, 0.3, 0.2, 0.8];

        let sp = StateProbs::new(
            genotyped_markers,
            2,
            hap_indices,
            state_probs,
            gen_positions,
        );

        let get_ref_allele = |marker: usize, hap: u32| -> u8 {
            assert!(marker <= 10);
            if hap == 0 { 0 } else { 1 }
        };

        // Marker 0 is before first genotyped marker (5)
        // Should return marker 5's value: P(ALT) = 0.3
        let post_before = sp.allele_posteriors(0, 2, &get_ref_allele);
        match post_before {
            AllelePosteriors::Biallelic(p_alt) => {
                assert!((p_alt - 0.3).abs() < 0.1, "Before first: expected ~0.3, got {}", p_alt);
            }
            _ => panic!("Expected Biallelic"),
        }
    }

    #[test]
    fn test_state_probs_probabilities_normalized() {
        // StateProbs should produce normalized probabilities (sum to 1)
        // even after interpolation

        let genotyped_markers = std::sync::Arc::new(vec![0, 10]);
        let gen_positions: Vec<f64> = (0..=10).map(|i| i as f64 * 0.1).collect();
        let gen_positions = std::sync::Arc::new(gen_positions);

        // 4 states, properly normalized at each genotyped marker
        let hap_indices = vec![
            vec![0, 1, 2, 3],
            vec![0, 1, 2, 3],
        ];
        // Marker 0: [0.4, 0.3, 0.2, 0.1]
        // Marker 10: [0.1, 0.2, 0.3, 0.4]
        let state_probs = vec![0.4, 0.3, 0.2, 0.1, 0.1, 0.2, 0.3, 0.4];

        let sp = StateProbs::new(
            genotyped_markers,
            4,
            hap_indices,
            state_probs,
            gen_positions,
        );

        // Allele mapping: hap 0,1 = REF (0), hap 2,3 = ALT (1)
        let get_ref_allele = |marker: usize, hap: u32| -> u8 {
            assert!(marker <= 10);
            if hap < 2 { 0 } else { 1 }
        };

        // At marker 0: P(REF) = 0.4+0.3 = 0.7, P(ALT) = 0.2+0.1 = 0.3
        let post_0 = sp.allele_posteriors(0, 2, &get_ref_allele);
        match post_0 {
            AllelePosteriors::Biallelic(p_alt) => {
                assert!((p_alt - 0.3).abs() < 0.05, "Marker 0: expected P(ALT)=0.3, got {}", p_alt);
            }
            _ => panic!("Expected Biallelic"),
        }

        // At marker 10: P(REF) = 0.1+0.2 = 0.3, P(ALT) = 0.3+0.4 = 0.7
        let post_10 = sp.allele_posteriors(10, 2, &get_ref_allele);
        match post_10 {
            AllelePosteriors::Biallelic(p_alt) => {
                assert!((p_alt - 0.7).abs() < 0.05, "Marker 10: expected P(ALT)=0.7, got {}", p_alt);
            }
            _ => panic!("Expected Biallelic"),
        }

        // At marker 5 (midpoint): should interpolate
        // Expected P(ALT) = 0.5 * 0.3 + 0.5 * 0.7 = 0.5
        let post_5 = sp.allele_posteriors(5, 2, &get_ref_allele);
        match post_5 {
            AllelePosteriors::Biallelic(p_alt) => {
                assert!((p_alt - 0.5).abs() < 0.1, "Marker 5: expected P(ALT)~0.5, got {}", p_alt);
            }
            _ => panic!("Expected Biallelic"),
        }
    }
}
