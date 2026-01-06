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
use std::sync::Mutex;

use crate::config::Config;
use crate::data::genetic_map::GeneticMaps;
use crate::data::haplotype::HapIdx;
use crate::data::marker::MarkerIdx;
use crate::data::storage::GenotypeMatrix;
use crate::data::storage::phase_state::PhaseState;
use crate::data::storage::coded_steps::RefPanelCoded;
use crate::error::Result;
use crate::io::vcf::{ImputationQuality, VcfReader, VcfWriter};
use crate::utils::workspace::ImpWorkspace;

use crate::model::imp_states::{CodedStepsConfig, ImpStates};
use crate::model::parameters::ModelParams;
use crate::model::recursive_ibs::RecursiveIbs;

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
    n_ref_markers: usize,
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
            n_ref_markers,
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

    /// Get reference marker index for a target marker
    pub fn ref_marker(&self, target_marker: usize) -> usize {
        self.target_to_ref[target_marker]
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

    /// Get the ref_to_target mapping array
    ///
    /// For each reference marker, returns the corresponding target marker index (-1 if not in target)
    pub fn ref_to_target(&self) -> &[i32] {
        &self.ref_to_target
    }

    /// Find flanking genotyped markers for a reference marker
    /// Returns (left_ref_marker, right_ref_marker, interpolation_weight)
    pub fn flanking_markers(&self, ref_marker: usize) -> (usize, usize, f32) {
        // Find left genotyped marker
        let mut left = ref_marker;
        while left > 0 && !self.is_genotyped(left) {
            left -= 1;
        }
        if !self.is_genotyped(left) {
            left = 0;
        }

        // Find right genotyped marker
        let mut right = ref_marker;
        while right < self.n_ref_markers - 1 && !self.is_genotyped(right) {
            right += 1;
        }
        if !self.is_genotyped(right) {
            right = self.n_ref_markers - 1;
        }

        // Calculate interpolation weight
        let weight = if left == right {
            0.5
        } else {
            (ref_marker - left) as f32 / (right - left) as f32
        };

        (left, right, weight)
    }
}

/// State probabilities with interpolation support
#[derive(Clone, Debug)]
pub struct StateProbs {
    /// Reference haplotype indices at each genotyped marker
    hap_indices: Vec<Vec<u32>>,
    /// State probabilities at each genotyped marker
    probs: Vec<Vec<f32>>,
    /// State probabilities at marker+1 (for interpolation)
    probs_p1: Vec<Vec<f32>>,
}

impl StateProbs {
    /// Create state probabilities from HMM output
    pub fn new(
        n_markers: usize,
        n_states: usize,
        hap_indices: Vec<Vec<u32>>,
        state_probs: Vec<f32>,
    ) -> Self {
        let threshold = 0.005f32.min(0.9999 / n_states as f32);

        let mut filtered_haps = Vec::with_capacity(n_markers);
        let mut filtered_probs = Vec::with_capacity(n_markers);
        let mut filtered_probs_p1 = Vec::with_capacity(n_markers);

        for m in 0..n_markers {
            let m_p1 = (m + 1).min(n_markers - 1);
            let row_offset = m * n_states;
            let row_offset_p1 = m_p1 * n_states;
            
            let mut haps = Vec::new();
            let mut probs = Vec::new();
            let mut probs_p1 = Vec::new();

            for j in 0..n_states.min(hap_indices.get(m).map(|v| v.len()).unwrap_or(0)) {
                let prob_m = state_probs
                    .get(row_offset + j)
                    .copied()
                    .unwrap_or(0.0);
                let prob_m_p1 = state_probs
                    .get(row_offset_p1 + j)
                    .copied()
                    .unwrap_or(0.0);

                if prob_m > threshold || prob_m_p1 > threshold {
                    haps.push(hap_indices[m][j]);
                    probs.push(prob_m);
                    probs_p1.push(prob_m_p1);
                }
            }

            filtered_haps.push(haps);
            filtered_probs.push(probs);
            filtered_probs_p1.push(probs_p1);
        }

        Self {
            hap_indices: filtered_haps,
            probs: filtered_probs,
            probs_p1: filtered_probs_p1,
        }
    }

    /// Get interpolated dosage at a reference marker
    pub fn interpolated_dosage<F>(
        &self,
        ref_marker: usize,
        alignment: &MarkerAlignment,
        get_ref_allele: F,
    ) -> f32
    where
        F: Fn(usize, u32) -> u8,
    {
        if let Some(target_marker) = alignment.target_marker(ref_marker) {
            // Genotyped marker - use direct state probs
            self.dosage_at_genotyped(target_marker, ref_marker, &get_ref_allele)
        } else {
            // Ungenotyped marker - interpolate
            let (left_ref, right_ref, weight) = alignment.flanking_markers(ref_marker);
            let left_target = alignment.target_marker(left_ref);
            let right_target = alignment.target_marker(right_ref);

            match (left_target, right_target) {
                (Some(lt), Some(_)) => {
                    // Interpolate between two genotyped markers
                    self.interpolated_dosage_between(lt, weight, ref_marker, &get_ref_allele)
                }
                (Some(t), None) | (None, Some(t)) => {
                    // Only one flanking marker - use its probs
                    self.dosage_at_genotyped(t, ref_marker, &get_ref_allele)
                }
                (None, None) => 0.0,
            }
        }
    }

    fn dosage_at_genotyped<F>(
        &self,
        target_marker: usize,
        ref_marker: usize,
        get_ref_allele: &F,
    ) -> f32
    where
        F: Fn(usize, u32) -> u8,
    {
        let haps = &self.hap_indices[target_marker];
        let probs = &self.probs[target_marker];

        let mut dosage = 0.0f32;
        for (j, &hap) in haps.iter().enumerate() {
            let allele = get_ref_allele(ref_marker, hap);
            dosage += probs[j] * allele as f32;
        }
        dosage
    }

    fn interpolated_dosage_between<F>(
        &self,
        left_marker: usize,
        weight: f32,
        ref_marker: usize,
        get_ref_allele: &F,
    ) -> f32
    where
        F: Fn(usize, u32) -> u8,
    {
        // Use probs from left marker and probs_p1 from left marker
        // Linear interpolation: (1-w) * prob_left + w * prob_right
        let left_haps = &self.hap_indices[left_marker];
        let left_probs = &self.probs[left_marker];
        let left_probs_p1 = &self.probs_p1[left_marker];

        let mut dosage = 0.0f32;
        for (j, &hap) in left_haps.iter().enumerate() {
            let allele = get_ref_allele(ref_marker, hap);
            let interp_prob = (1.0 - weight) * left_probs[j] + weight * left_probs_p1[j];
            dosage += interp_prob * allele as f32;
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

        eprintln!("Loading reference VCF...");
        let ref_path = self.config.r#ref.as_ref().ok_or_else(|| {
            crate::error::ReagleError::config("Reference panel required for imputation")
        })?;
        let (mut ref_reader, ref_file) = VcfReader::open(ref_path)?;
        let ref_gt = ref_reader.read_all(ref_file)?;

        if target_gt.n_markers() == 0 || ref_gt.n_markers() == 0 {
            return Ok(());
        }

        // Phase target before imputation (imputation requires phased haplotypes)
        // With type states, VcfReader returns GenotypeMatrix<Unphased>, so we always phase
        eprintln!("Phasing target data before imputation...");

        // Create phasing pipeline with current config
        let mut phasing = super::phasing::PhasingPipeline::new(self.config.clone());

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

        let n_ref_haps = ref_gt.n_haplotypes();
        let n_ref_markers = ref_gt.n_markers();
        let n_target_markers = target_gt.n_markers();
        let n_target_samples = target_gt.n_samples();
        let n_target_haps = target_gt.n_haplotypes();

        eprintln!(
            "Target: {} markers, {} samples; Reference: {} markers, {} haplotypes",
            n_target_markers, n_target_samples, n_ref_markers, n_ref_haps
        );

        // Create marker alignment
        eprintln!("Aligning markers...");
        let alignment = MarkerAlignment::new(&target_gt, &ref_gt);
        let n_genotyped = alignment.ref_to_target.iter().filter(|&&x| x >= 0).count();
        eprintln!(
            "  {} of {} reference markers are genotyped in target",
            n_genotyped, n_ref_markers
        );

        // Initialize parameters
        self.params = ModelParams::for_imputation(n_ref_haps);
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

        // Compute genetic positions at genotyped markers (for ImpStates)
        let gen_positions: Vec<f64> = (0..n_target_markers)
            .map(|m| {
                let ref_m = alignment.ref_marker(m);
                let pos = ref_gt.marker(MarkerIdx::new(ref_m as u32)).pos;
                gen_maps.gen_pos(chrom, pos)
            })
            .collect();

        // Compute recombination probabilities at genotyped markers
        let p_recomb: Vec<f32> = std::iter::once(0.0f32)
            .chain((1..n_target_markers).map(|m| {
                let prev_ref = alignment.ref_marker(m - 1);
                let curr_ref = alignment.ref_marker(m);
                let pos1 = ref_gt.marker(MarkerIdx::new(prev_ref as u32)).pos;
                let pos2 = ref_gt.marker(MarkerIdx::new(curr_ref as u32)).pos;
                let gen_dist = gen_maps.gen_dist(chrom, pos1, pos2);
                self.params.p_recomb(gen_dist)
            }))
            .collect();

        // CodedSteps configuration
        let steps_config = CodedStepsConfig {
            step_cm: self.config.imp_step,
            n_ibs_haps: self.config.imp_nsteps,
        };

        // Compute genetic positions at ALL reference markers (for RefPanelCoded)
        let ref_gen_positions: Vec<f64> = (0..n_ref_markers)
            .map(|m| {
                let pos = ref_gt.marker(MarkerIdx::new(m as u32)).pos;
                gen_maps.gen_pos(chrom, pos)
            })
            .collect();

        // Build coded reference panel for efficient PBWT operations
        eprintln!("Building coded reference panel...");
        let mut ref_panel_coded = RefPanelCoded::from_gen_positions(
            &ref_gt,
            &ref_gen_positions,
            self.config.imp_step as f64,
        );
        eprintln!(
            "  {} steps, {} total patterns, avg compression ratio: {:.1}x",
            ref_panel_coded.n_steps(),
            ref_panel_coded.total_patterns(),
            ref_panel_coded.avg_compression_ratio()
        );

        // Append target haplotypes to coded panel for RecursiveIbs
        eprintln!("Appending target haplotypes to coded panel...");
        ref_panel_coded.append_target_haplotypes(
            &target_gt,
            alignment.ref_to_target(),
            |target_m, allele| alignment.map_allele(target_m, allele),
        );
        eprintln!(
            "  Combined panel: {} haplotypes ({} ref + {} target)",
            ref_panel_coded.n_haps(),
            n_ref_haps,
            n_target_haps
        );

        // Initialize RecursiveIbs for efficient IBS haplotype finding
        eprintln!("Building recursive IBS index...");
        let recursive_ibs = RecursiveIbs::new(
            &ref_panel_coded,
            n_ref_haps,
            n_target_haps,
            self.config.seed as u64,
            self.config.imp_nsteps,
            8, // n_haps_per_step (default from Java Beagle)
        );
        eprintln!(
            "  {} steps, {} IBS haps per step",
            recursive_ibs.n_steps(),
            recursive_ibs.n_haps_per_step()
        );

        eprintln!("Running imputation with dynamic state selection...");
        let n_states = self.params.n_states;

        // Run imputation for each target haplotype with per-thread workspaces
        let state_probs: Vec<StateProbs> = (0..n_target_haps)
            .into_par_iter()
            .map_init(
                // Initialize workspace for each thread
                || ImpWorkspace::with_ref_size(n_states, n_target_markers, n_ref_haps),
                // Process each haplotype with its thread's workspace
                |workspace, h| {
                    let hap_idx = HapIdx::new(h as u32);

                    // Get target alleles at genotyped markers, mapped to reference allele space
                    // This handles strand flips (A/T vs T/A) and allele swaps automatically
                    let target_alleles: Vec<u8> = (0..n_target_markers)
                        .map(|m| {
                            let raw_allele = target_gt.allele(MarkerIdx::new(m as u32), hap_idx);
                            alignment.map_allele(m, raw_allele)
                        })
                        .collect();

                    // Create ImpStates for dynamic state selection with RefPanelCoded
                    // Combine config seed with haplotype index for reproducibility
                    let seed = (self.config.seed as u64).wrapping_add(h as u64);
                    let mut imp_states =
                        ImpStates::new(&ref_panel_coded, n_states, &gen_positions, &steps_config, seed);

                    // Get reference allele closure
                    let get_ref_allele = |m: usize, hap: u32| -> u8 {
                        let ref_m = alignment.ref_marker(m);
                        ref_gt.allele(MarkerIdx::new(ref_m as u32), HapIdx::new(hap))
                    };

                    // Get IBS-based states using workspace
                    let mut hap_indices: Vec<Vec<u32>> = Vec::new();
                    let mut allele_match: Vec<Vec<bool>> = Vec::new();
                    let actual_n_states = imp_states.ibs_states(
                        get_ref_allele,
                        &target_alleles,
                        workspace,
                        &mut hap_indices,
                        &mut allele_match,
                    );

                    // Run forward-backward HMM using workspace buffers
                    let hmm_state_probs = run_hmm_forward_backward(
                        &target_alleles,
                        &allele_match,
                        &p_recomb,
                        self.params.p_mismatch,
                        actual_n_states,
                        workspace,
                    );

                    // Create StateProbs for interpolation
                    StateProbs::new(
                        n_target_markers,
                        actual_n_states,
                        hap_indices,
                        hmm_state_probs,
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
        let quality = Mutex::new(ImputationQuality::new(&n_alleles_per_marker));

        // Mark imputed markers (those not in target)
        for m in 0..n_ref_markers {
            let is_imputed = !alignment.is_genotyped(m);
            quality.lock().unwrap().set_imputed(m, is_imputed);
        }

        // Compute dosages at all reference markers (including ungenotyped)
        // Also accumulate quality statistics
        let sample_dosages: Vec<Vec<f32>> = (0..n_target_samples)
            .into_par_iter()
            .map(|s| {
                let hap1_probs = &state_probs[s * 2];
                let hap2_probs = &state_probs[s * 2 + 1];

                let dosages: Vec<f32> = (0..n_ref_markers)
                    .map(|m| {
                        let get_ref_allele = |ref_m: usize, hap: u32| -> u8 {
                            ref_gt.allele(MarkerIdx::new(ref_m as u32), HapIdx::new(hap))
                        };

                        let d1 = hap1_probs.interpolated_dosage(m, &alignment, &get_ref_allele);
                        let d2 = hap2_probs.interpolated_dosage(m, &alignment, &get_ref_allele);

                        // For biallelic sites, convert dosage to allele probabilities for DR2
                        // d1 = P(ALT) for hap1, d2 = P(ALT) for hap2
                        let n_alleles = n_alleles_per_marker[m];
                        if n_alleles == 2 {
                            let probs1 = vec![1.0 - d1, d1];
                            let probs2 = vec![1.0 - d2, d2];
                            if let Ok(mut q) = quality.lock() {
                                if let Some(stats) = q.get_mut(m) {
                                    stats.add_sample(&probs1, &probs2);
                                }
                            }
                        }

                        d1 + d2
                    })
                    .collect();

                dosages
            })
            .collect();

        // Flatten dosages for output (marker-major order for the writer)
        // Reorder from [sample][marker] to [marker][sample]
        let mut flat_dosages: Vec<f32> = Vec::with_capacity(n_ref_markers * n_target_samples);
        for m in 0..n_ref_markers {
            for s in 0..n_target_samples {
                flat_dosages.push(sample_dosages[s][m]);
            }
        }

        // Get quality stats
        let quality = quality.into_inner().unwrap();

        // Write output with quality metrics
        let output_path = self.config.out.with_extension("vcf.gz");
        eprintln!("Writing output to {:?}", output_path);
        let mut writer = VcfWriter::create(&output_path, target_samples)?;
        writer.write_header_imputed(ref_gt.markers())?;
        writer.write_imputed_with_quality(&ref_gt, &flat_dosages, &quality, 0, n_ref_markers)?;
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
            
            let val = if m == 0 {
                emit / n_states as f32
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
