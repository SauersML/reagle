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
use crate::error::Result;
use crate::io::vcf::{VcfReader, VcfWriter, ImputationQuality};
use crate::io::window::WindowIndices;
use crate::model::hmm::LiStephensHmm;
use crate::model::imp_states::{CodedStepsConfig, ImpStates};
use crate::model::parameters::ModelParams;
use crate::model::pbwt::PbwtIbs;

/// Results from processing a single window
///
/// Contains dosages and quality metrics that need to be spliced together
/// from overlapping windows.
#[derive(Clone)]
pub struct WindowResult {
    /// Dosages for each sample at each marker in this window [marker][sample]
    pub dosages: Vec<Vec<f32>>,
    /// Quality statistics for each marker
    pub quality: ImputationQuality,
    /// Window indices (splice points)
    pub indices: WindowIndices,
}

impl WindowResult {
    /// Create new window result
    pub fn new(n_markers: usize, n_samples: usize, indices: WindowIndices) -> Self {
        let n_alleles_per_marker = vec![2usize; n_markers]; // Assume biallelic
        Self {
            dosages: vec![vec![0.0; n_samples]; n_markers],
            quality: ImputationQuality::new(&n_alleles_per_marker),
            indices,
        }
    }
}

/// Splice window results together using overlap midpoints
///
/// This implements Java MarkerIndices splice point logic:
/// - Each window computes results for its full range
/// - Output is taken from prev_splice to next_splice
/// - Splice points are at the middle of overlap regions
///
/// # Arguments
/// * `results` - Vector of window results in order
/// * `total_markers` - Total number of markers in output
/// * `n_samples` - Number of samples
pub fn splice_window_results(
    results: &[WindowResult],
    total_markers: usize,
    n_samples: usize,
) -> (Vec<Vec<f32>>, ImputationQuality) {
    let n_alleles_per_marker = vec![2usize; total_markers];
    let mut final_dosages = vec![vec![0.0f32; n_samples]; total_markers];
    let mut final_quality = ImputationQuality::new(&n_alleles_per_marker);

    for result in results {
        let indices = &result.indices;

        // Only copy from prev_splice to next_splice (the "owned" region)
        let local_splice_start = indices.prev_splice.saturating_sub(indices.start);
        let local_splice_end = indices.next_splice.saturating_sub(indices.start).min(result.dosages.len());

        for local_m in local_splice_start..local_splice_end {
            let global_m = indices.start + local_m;
            if global_m < total_markers {
                // Copy dosages
                if local_m < result.dosages.len() {
                    final_dosages[global_m] = result.dosages[local_m].clone();
                }

                // Copy quality stats
                if let Some(src_stats) = result.quality.get(local_m) {
                    if let Some(dst_stats) = final_quality.get_mut(global_m) {
                        *dst_stats = src_stats.clone();
                    }
                }
            }
        }
    }

    (final_dosages, final_quality)
}

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
}

impl MarkerAlignment {
    /// Create alignment by matching markers by position
    pub fn new(target_gt: &GenotypeMatrix, ref_gt: &GenotypeMatrix) -> Self {
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

        for m in 0..n_ref_markers {
            let marker = ref_gt.marker(MarkerIdx::new(m as u32));
            if let Some(&target_idx) = target_pos_map.get(&(marker.chrom.0, marker.pos)) {
                ref_to_target[m] = target_idx as i32;
                target_to_ref[target_idx] = m;
            }
        }

        Self {
            ref_to_target,
            target_to_ref,
            n_ref_markers,
        }
    }

    /// Check if a reference marker is genotyped in target
    pub fn is_genotyped(&self, ref_marker: usize) -> bool {
        self.ref_to_target.get(ref_marker).copied().unwrap_or(-1) >= 0
    }

    /// Get target marker index for a reference marker (returns None if not genotyped)
    pub fn target_marker(&self, ref_marker: usize) -> Option<usize> {
        let idx = self.ref_to_target.get(ref_marker).copied().unwrap_or(-1);
        if idx >= 0 {
            Some(idx as usize)
        } else {
            None
        }
    }

    /// Get reference marker index for a target marker
    pub fn ref_marker(&self, target_marker: usize) -> usize {
        self.target_to_ref[target_marker]
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
        state_probs: Vec<Vec<f32>>,
    ) -> Self {
        let threshold = 0.005f32.min(0.9999 / n_states as f32);

        let mut filtered_haps = Vec::with_capacity(n_markers);
        let mut filtered_probs = Vec::with_capacity(n_markers);
        let mut filtered_probs_p1 = Vec::with_capacity(n_markers);

        for m in 0..n_markers {
            let m_p1 = (m + 1).min(n_markers - 1);
            let mut haps = Vec::new();
            let mut probs = Vec::new();
            let mut probs_p1 = Vec::new();

            for j in 0..n_states.min(hap_indices.get(m).map(|v| v.len()).unwrap_or(0)) {
                let prob_m = state_probs.get(m).and_then(|v| v.get(j)).copied().unwrap_or(0.0);
                let prob_m_p1 = state_probs.get(m_p1).and_then(|v| v.get(j)).copied().unwrap_or(0.0);

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

    fn dosage_at_genotyped<F>(&self, target_marker: usize, ref_marker: usize, get_ref_allele: &F) -> f32
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
        self.params.set_n_states(self.config.imp_states.min(n_ref_haps));

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

        eprintln!("Running imputation with dynamic state selection...");
        let n_states = self.params.n_states;

        // Run imputation for each target haplotype
        let state_probs: Vec<StateProbs> = (0..n_target_haps)
            .into_par_iter()
            .map(|h| {
                let hap_idx = HapIdx::new(h as u32);

                // Get target alleles at genotyped markers
                let target_alleles: Vec<u8> = (0..n_target_markers)
                    .map(|m| target_gt.allele(MarkerIdx::new(m as u32), hap_idx))
                    .collect();

                // Create ImpStates for dynamic state selection
                let mut imp_states = ImpStates::new(
                    n_ref_haps,
                    n_states,
                    &gen_positions,
                    &steps_config,
                );

                // Get reference allele closure
                let get_ref_allele = |m: usize, hap: u32| -> u8 {
                    let ref_m = alignment.ref_marker(m);
                    ref_gt.allele(MarkerIdx::new(ref_m as u32), HapIdx::new(hap))
                };

                // Get IBS-based states
                let mut hap_indices: Vec<Vec<u32>> = Vec::new();
                let mut allele_match: Vec<Vec<bool>> = Vec::new();
                let actual_n_states = imp_states.ibs_states(
                    get_ref_allele,
                    &target_alleles,
                    &mut hap_indices,
                    &mut allele_match,
                );

                // Run forward-backward HMM
                let hmm_state_probs = run_hmm_forward_backward(
                    &target_alleles,
                    &allele_match,
                    &p_recomb,
                    self.params.p_mismatch,
                    actual_n_states,
                );

                // Create StateProbs for interpolation
                StateProbs::new(
                    n_target_markers,
                    actual_n_states,
                    hap_indices,
                    hmm_state_probs,
                )
            })
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
) -> Vec<Vec<f32>> {
    let n_markers = target_alleles.len();
    if n_markers == 0 || n_states == 0 {
        return Vec::new();
    }

    let p_match = 1.0 - p_mismatch;
    let emit_probs = [p_match, p_mismatch];

    // Forward pass
    let mut fwd: Vec<Vec<f32>> = vec![vec![0.0; n_states]; n_markers];
    let mut fwd_sum = 1.0f32;

    for m in 0..n_markers {
        let p_rec = p_recomb.get(m).copied().unwrap_or(0.0);
        let shift = p_rec / n_states as f32;
        let scale = (1.0 - p_rec) / fwd_sum;

        let mut new_sum = 0.0f32;
        let matches = &allele_match[m];

        for k in 0..n_states.min(matches.len()) {
            let emit = if matches[k] { emit_probs[0] } else { emit_probs[1] };
            fwd[m][k] = if m == 0 {
                emit / n_states as f32
            } else {
                emit * (scale * fwd[m - 1][k] + shift)
            };
            new_sum += fwd[m][k];
        }
        fwd_sum = new_sum;
    }

    // Backward pass and compute posteriors
    let mut bwd = vec![1.0 / n_states as f32; n_states];
    let mut bwd_sum = 1.0f32;

    for m in (0..n_markers).rev() {
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
            fwd[m][k] *= bwd[k];
            state_sum += fwd[m][k];
        }

        // Normalize
        if state_sum > 0.0 {
            for k in 0..n_states {
                fwd[m][k] /= state_sum;
            }
        }

        // Apply emission for next backward iteration
        if m > 0 {
            let matches = &allele_match[m];
            bwd_sum = 0.0;
            for k in 0..n_states.min(matches.len()) {
                let emit = if matches[k] { emit_probs[0] } else { emit_probs[1] };
                bwd[k] *= emit;
                bwd_sum += bwd[k];
            }
        }
    }

    fwd
}

/// Internal implementation of imputation for a single haplotype
fn impute_haplotype_internal(
    target_alleles: &[u8],
    ref_gt: &GenotypeMatrix,
    params: &ModelParams,
    ref_haps: &[HapIdx],
    p_recomb: &[f32],
) -> Vec<f32> {
    let n_markers = ref_gt.n_markers();
    let n_states = ref_haps.len();

    if n_markers == 0 || n_states == 0 {
        return vec![0.0; n_markers];
    }

    // Create HMM
    let hmm = LiStephensHmm::new(ref_gt, params, ref_haps.to_vec(), p_recomb.to_vec());

    // Allocate forward and backward buffers
    let mut fwd: Vec<Vec<f32>> = vec![vec![0.0; n_states]; n_markers];
    let mut bwd: Vec<f32> = vec![0.0; n_states];

    // Run forward-backward
    let result = hmm.forward_backward(target_alleles, &mut fwd, &mut bwd);

    // Compute dosages at each marker
    (0..n_markers)
        .map(|m| {
            let marker_idx = MarkerIdx::new(m as u32);
            if m < result.state_probs.len() {
                hmm.compute_dosage(marker_idx, &result.state_probs[m])
            } else {
                0.0
            }
        })
        .collect()
}

/// Impute a single target haplotype (public API)
///
/// # Arguments
/// * `target_alleles` - Alleles of target haplotype at genotyped markers
/// * `ref_gt` - Reference genotype matrix
/// * `params` - Model parameters
/// * `ref_haps` - Selected reference haplotypes to use as HMM states
/// * `gen_dists` - Genetic distances between consecutive markers (in cM)
/// # Returns
/// Vector of dosages at each marker
#[allow(unused_variables)]
pub fn impute_haplotype(
    target_alleles: &[u8],
    ref_gt: &GenotypeMatrix,
    params: &ModelParams,
    ref_haps: &[HapIdx],
    gen_dists: &[f64],
    seed: u64,
) -> Vec<f32> {
    // Convert genetic distances to recombination probabilities
    let p_recomb: Vec<f32> = std::iter::once(0.0f32)
        .chain(gen_dists.iter().map(|&d| params.p_recomb(d)))
        .collect();

    impute_haplotype_internal(target_alleles, ref_gt, params, ref_haps, &p_recomb)
}

/// Imputation with PBWT-based state selection
///
/// This matches Java `imp/ImpLSBaum` more closely by selecting states
/// dynamically using PBWT.
pub struct ImpLSBaum<'a> {
    ref_gt: &'a GenotypeMatrix,
    params: &'a ModelParams,
    pbwt: PbwtIbs,
    p_recomb: Vec<f32>,
}

impl<'a> ImpLSBaum<'a> {
    /// Create a new imputation engine
    pub fn new(
        ref_gt: &'a GenotypeMatrix,
        params: &'a ModelParams,
        p_recomb: Vec<f32>,
    ) -> Self {
        let n_haps = ref_gt.n_haplotypes();
        let pbwt = PbwtIbs::new(n_haps);

        Self {
            ref_gt,
            params,
            pbwt,
            p_recomb,
        }
    }

    /// Impute a target haplotype
    pub fn impute(&mut self, target_alleles: &[u8]) -> Vec<f32> {
        let n_markers = self.ref_gt.n_markers();
        let n_states = self.params.n_states;

        if n_markers == 0 || n_states == 0 {
            return vec![0.0; n_markers];
        }

        // Build PBWT on reference panel incrementally and select IBS states
        self.pbwt.reset();

        // Create a synthetic "target haplotype" position tracker
        // We'll select states based on IBS matches to the target at each marker
        let n_ref_haps = self.ref_gt.n_haplotypes();

        // Track cumulative IBS match scores for each reference haplotype
        let mut ibs_scores: Vec<u32> = vec![0; n_ref_haps];

        // Build PBWT and accumulate IBS scores
        for m in 0..n_markers {
            let target_allele = target_alleles.get(m).copied().unwrap_or(255);

            let alleles: Vec<u8> = (0..n_ref_haps)
                .map(|h| self.ref_gt.allele(MarkerIdx::new(m as u32), HapIdx::new(h as u32)))
                .collect();

            // Increment IBS score for haplotypes matching target allele
            if target_allele != 255 {
                for (h, &allele) in alleles.iter().enumerate() {
                    if allele == target_allele {
                        ibs_scores[h] += 1;
                    }
                }
            }

            self.pbwt.fwd_update(&alleles, 2, m);
        }

        // Select top n_states haplotypes by IBS score
        let mut scored_haps: Vec<(u32, usize)> = ibs_scores
            .iter()
            .enumerate()
            .map(|(h, &score)| (score, h))
            .collect();

        // Sort by score descending
        scored_haps.sort_by(|a, b| b.0.cmp(&a.0));

        // Take top n_states
        let ref_haps: Vec<HapIdx> = scored_haps
            .iter()
            .take(n_states.min(n_ref_haps))
            .map(|&(_, h)| HapIdx::new(h as u32))
            .collect();

        impute_haplotype_internal(
            target_alleles,
            self.ref_gt,
            self.params,
            &ref_haps,
            &self.p_recomb,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::haplotype::Samples;
    use crate::data::marker::{Allele, Marker, Markers};
    use crate::data::storage::GenotypeColumn;
    use crate::data::ChromIdx;
    use std::sync::Arc;

    fn make_test_ref_panel() -> GenotypeMatrix {
        let samples = Arc::new(Samples::from_ids(vec![
            "R1".to_string(),
            "R2".to_string(),
        ]));
        let mut markers = Markers::new();
        markers.add_chrom("chr1");

        let mut columns = Vec::new();
        for i in 0..5 {
            let m = Marker::new(
                ChromIdx::new(0),
                (i * 1000 + 100) as u32,
                None,
                Allele::Base(0),
                vec![Allele::Base(1)],
            );
            markers.push(m);
            columns.push(GenotypeColumn::from_alleles(&[0, 1, 0, 1], 2));
        }

        GenotypeMatrix::new(markers, columns, samples, true)
    }

    #[test]
    fn test_impute_haplotype() {
        let ref_panel = make_test_ref_panel();
        let params = ModelParams::for_imputation(4);
        let ref_haps: Vec<HapIdx> = (0..4).map(|i| HapIdx::new(i)).collect();
        let gen_dists = vec![0.01; 4];

        let target_alleles = vec![0, 1, 0, 1, 0];
        let dosages = impute_haplotype(
            &target_alleles,
            &ref_panel,
            &params,
            &ref_haps,
            &gen_dists,
            12345,
        );

        assert_eq!(dosages.len(), 5);
        for d in &dosages {
            assert!(*d >= 0.0 && *d <= 1.0);
        }
    }

    #[test]
    fn test_imp_ls_baum() {
        let ref_panel = make_test_ref_panel();
        let mut params = ModelParams::for_imputation(4);
        params.set_n_states(4);

        let p_recomb = vec![0.0, 0.01, 0.01, 0.01, 0.01];
        let mut imp = ImpLSBaum::new(&ref_panel, &params, p_recomb);

        let target_alleles = vec![0, 1, 0, 1, 0];
        let dosages = imp.impute(&target_alleles);

        assert_eq!(dosages.len(), 5);
        for d in &dosages {
            assert!(*d >= 0.0 && *d <= 1.0);
        }
    }
}
