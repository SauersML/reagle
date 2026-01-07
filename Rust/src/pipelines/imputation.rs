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

    /// Compute dosage at a reference marker with interpolation for ungenotyped markers.
    ///
    /// For genotyped markers: direct lookup of HMM posterior probabilities.
    /// For ungenotyped markers: linear interpolation in genetic distance space
    /// between flanking genotyped markers, matching Java Beagle's approach.
    #[inline]
    pub fn dosage<F>(&self, ref_marker: usize, get_ref_allele: F) -> f32
    where
        F: Fn(usize, u32) -> u8,
    {
        // Binary search to find position in genotyped markers
        match self.genotyped_markers.binary_search(&ref_marker) {
            Ok(sparse_idx) => {
                // Exact match - this is a genotyped marker
                self.dosage_at_genotyped(sparse_idx, ref_marker, &get_ref_allele)
            }
            Err(insert_pos) => {
                // Not genotyped - interpolate between flanking genotyped markers
                self.dosage_interpolated(ref_marker, insert_pos, &get_ref_allele)
            }
        }
    }

    /// Compute dosage at a genotyped marker (direct lookup)
    #[inline]
    fn dosage_at_genotyped<F>(&self, sparse_idx: usize, ref_marker: usize, get_ref_allele: &F) -> f32
    where
        F: Fn(usize, u32) -> u8,
    {
        let haps = &self.hap_indices[sparse_idx];
        let probs = &self.probs[sparse_idx];

        let mut dosage = 0.0f32;
        for (j, &hap) in haps.iter().enumerate() {
            let allele = get_ref_allele(ref_marker, hap);
            if allele != 255 {
                dosage += probs[j] * allele as f32;
            }
        }
        dosage
    }

    /// Compute dosage at an ungenotyped marker via interpolation
    ///
    /// # Java Compatibility
    /// Matches Java Beagle's `ImputedVcfWriter.setAlProbs()`:
    /// - Uses the **same** haplotypes from the left flanking marker
    /// - Interpolates **probability weights** using `probs` and `probs_p1`
    /// - Formula: `allele * (wt * prob + (1-wt) * probP1)`
    ///
    /// This ensures smooth probability transitions across marker boundaries,
    /// maintaining calibration and avoiding discontinuities from switching haplotype sets.
    #[inline]
    fn dosage_interpolated<F>(&self, ref_marker: usize, insert_pos: usize, get_ref_allele: &F) -> f32
    where
        F: Fn(usize, u32) -> u8,
    {
        let n_genotyped = self.genotyped_markers.len();

        // Handle edge cases
        if n_genotyped == 0 {
            return 0.0;
        }
        if insert_pos == 0 {
            // Before first genotyped marker - use first marker's probs
            return self.dosage_at_genotyped(0, ref_marker, get_ref_allele);
        }
        if insert_pos >= n_genotyped {
            // After last genotyped marker - use last marker's probs
            return self.dosage_at_genotyped(n_genotyped - 1, ref_marker, get_ref_allele);
        }

        // Use left marker's haplotypes with interpolated probabilities
        // This matches Java's approach of using the same haplotype set
        // with smoothly transitioning probability weights
        let left_sparse = insert_pos - 1;
        let left_ref = self.genotyped_markers[left_sparse];
        let right_ref = self.genotyped_markers[insert_pos];

        // Compute interpolation weight based on genetic distance
        // Java: wt = (cumPos[nextStart] - cumPos[m]) / (cumPos[nextStart] - cumPos[end-1])
        // This gives wt=1.0 at left marker (use probs), wt=0.0 at right marker (use probs_p1)
        let pos_left = self.gen_positions[left_ref];
        let pos_right = self.gen_positions[right_ref];
        let pos_marker = self.gen_positions[ref_marker];

        let total_dist = pos_right - pos_left;
        let weight_left = if total_dist > 1e-10 {
            ((pos_right - pos_marker) / total_dist) as f32
        } else {
            0.5 // Equal weight if markers are at same position
        };

        // Compute dosage using the SAME haplotypes from left marker
        // with interpolated probability weights
        let haps = &self.hap_indices[left_sparse];
        let probs = &self.probs[left_sparse];
        let probs_p1 = &self.probs_p1[left_sparse];

        let mut dosage = 0.0f32;
        for (j, &hap) in haps.iter().enumerate() {
            let allele = get_ref_allele(ref_marker, hap);
            if allele != 255 {
                // Interpolate probability: wt*prob + (1-wt)*probP1
                let interpolated_prob = weight_left * probs[j] + (1.0 - weight_left) * probs_p1[j];
                dosage += interpolated_prob * allele as f32;
            }
        }
        dosage
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

                    // Get IBS-based states (runs on all ref markers for state selection)
                    let mut full_hap_indices: Vec<Vec<u32>> = Vec::new();
                    let mut full_allele_match: Vec<Vec<bool>> = Vec::new();
                    let actual_n_states = imp_states.ibs_states(
                        get_ref_allele,
                        &target_alleles,
                        workspace,
                        &mut full_hap_indices,
                        &mut full_allele_match,
                    );

                    // Extract sparse data for genotyped markers only
                    let sparse_hap_indices: Vec<Vec<u32>> = genotyped_markers
                        .iter()
                        .map(|&m| full_hap_indices.get(m).cloned().unwrap_or_default())
                        .collect();
                    let sparse_allele_match: Vec<Vec<bool>> = genotyped_markers
                        .iter()
                        .map(|&m| full_allele_match.get(m).cloned().unwrap_or_default())
                        .collect();
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
        // Each sample returns: (dosages, allele_probs, quality_contributions)
        // where quality_contributions = Vec<(marker_idx, probs1, probs2)> for biallelic sites
        type QualityContrib = (usize, [f32; 2], [f32; 2]); // (marker, probs1, probs2)
        let sample_results: Vec<(Vec<f32>, Option<Vec<f32>>, Vec<QualityContrib>)> = (0..n_target_samples)
            .into_par_iter()
            .map(|s| {
                let hap1_probs = &state_probs[s * 2];
                let hap2_probs = &state_probs[s * 2 + 1];

                let mut dosages: Vec<f32> = Vec::with_capacity(n_ref_markers);
                let mut allele_probs: Option<Vec<f32>> = if need_allele_probs {
                    Some(Vec::with_capacity(n_ref_markers * 2))
                } else {
                    None
                };
                let mut quality_contribs: Vec<QualityContrib> = Vec::new();

                for m in 0..n_ref_markers {
                    let get_ref_allele = |ref_m: usize, hap: u32| -> u8 {
                        ref_gt.allele(MarkerIdx::new(ref_m as u32), HapIdx::new(hap))
                    };

                    let d1 = hap1_probs.dosage(m, &get_ref_allele);
                    let d2 = hap2_probs.dosage(m, &get_ref_allele);

                    // For biallelic sites, record contributions for DR2 calculation
                    let n_alleles = n_alleles_per_marker[m];
                    if n_alleles == 2 {
                        quality_contribs.push((m, [1.0 - d1, d1], [1.0 - d2, d2]));
                    }

                    dosages.push(d1 + d2);

                    // Store per-haplotype allele probabilities if needed
                    if let Some(ref mut ap) = allele_probs {
                        ap.push(d1); // P(ALT) for hap1
                        ap.push(d2); // P(ALT) for hap2
                    }
                }

                (dosages, allele_probs, quality_contribs)
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

        // Flatten dosages for output (marker-major order for the writer)
        // Reorder from [sample][marker] to [marker][sample]
        let mut flat_dosages: Vec<f32> = Vec::with_capacity(n_ref_markers * n_target_samples);
        for m in 0..n_ref_markers {
            for s in 0..n_target_samples {
                flat_dosages.push(sample_results[s].0[m]);
            }
        }

        // Flatten allele probabilities if needed
        // Layout: [marker][sample*2 + hap_offset] -> P(ALT) for that haplotype
        let flat_allele_probs: Option<Vec<f32>> = if need_allele_probs {
            let mut probs = Vec::with_capacity(n_ref_markers * n_target_samples * 2);
            for m in 0..n_ref_markers {
                for s in 0..n_target_samples {
                    if let Some(ref ap) = sample_results[s].1 {
                        probs.push(ap[m * 2]);     // hap1
                        probs.push(ap[m * 2 + 1]); // hap2
                    }
                }
            }
            Some(probs)
        } else {
            None
        };


        // quality is already owned - no mutex unwrapping needed

        // Write output with quality metrics
        let output_path = self.config.out.with_extension("vcf.gz");
        eprintln!("Writing output to {:?}", output_path);
        let mut writer = VcfWriter::create(&output_path, target_samples)?;
        writer.write_header_extended(ref_gt.markers(), true, self.config.gp, self.config.ap)?;

        // Use appropriate writer method based on AP/GP flags
        if need_allele_probs {
            writer.write_imputed_with_probs(
                &ref_gt,
                &flat_dosages,
                flat_allele_probs.as_deref(),
                &quality,
                0,
                n_ref_markers,
                self.config.gp,
                self.config.ap,
            )?;
        } else {
            writer.write_imputed_with_quality(&ref_gt, &flat_dosages, &quality, 0, n_ref_markers)?;
        }
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
