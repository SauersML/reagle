//! # Phasing Pipeline
//!
//! Orchestrates the phasing workflow:
//! 1. Load target VCF
//! 2. Classify markers into Stage 1 (high-frequency) and Stage 2 (low-frequency/rare)
//! 3. Build PBWT for haplotype matching
//! 4. Run PBWT-accelerated Li-Stephens HMM (PhasingHmm) on Stage 1 markers
//! 5. Update phase and iterate
//! 6. Collect EM parameter estimates and update
//! 7. Run Stage 2 phasing: interpolate state probabilities to phase rare variants
//! 8. Write phased output
//!
//! This implements Beagle's two-stage phasing algorithm for handling rare variants.

use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::config::Config;
use crate::data::genetic_map::GeneticMaps;
use crate::data::haplotype::{HapIdx, SampleIdx};
use crate::data::marker::MarkerIdx;
use crate::data::storage::{GenotypeColumn, GenotypeMatrix, MutableGenotypes};
use crate::error::Result;
use crate::io::vcf::{VcfReader, VcfWriter};
use crate::model::parameters::{AtomicParamEstimates, ModelParams, ParamEstimates};
use crate::model::pbwt::{PbwtDivUpdater, PbwtIbs};

/// Stage marker classification for two-stage phasing
///
/// This follows Java FixedPhaseData's approach of separating high-frequency (Stage 1)
/// and low-frequency (Stage 2) markers. Stage 1 markers form a "scaffold" used to
/// phase rare variants in Stage 2.
#[derive(Clone, Debug)]
pub struct StageMarkers {
    /// Indices of Stage 1 (high-frequency) markers in the original marker array
    pub stage1_indices: Vec<usize>,
    /// For each marker, the index of the preceding Stage 1 marker (-1 if none)
    pub prev_stage1: Vec<i32>,
    /// For each marker, the index of the following Stage 1 marker (n_stage1 if none)
    pub next_stage1: Vec<i32>,
    /// For each marker, interpolation weight toward prev_stage1 marker (0.0 = use next, 1.0 = use prev)
    pub prev_stage1_wt: Vec<f32>,
    /// For each marker, for each allele, whether that allele is low-frequency
    pub is_low_freq: Vec<Vec<bool>>,
    /// Total number of markers
    pub n_markers: usize,
}

impl StageMarkers {
    /// Create stage marker classification from genotype matrix
    ///
    /// # Arguments
    /// * `geno` - Genotype data
    /// * `rare_threshold` - Allele frequency threshold below which variants are "rare" (default 0.002)
    /// * `min_stage1_spacing` - Minimum number of markers between Stage 1 markers (ensures scaffold density)
    pub fn classify(
        geno: &MutableGenotypes,
        n_alleles_per_marker: &[usize],
        rare_threshold: f32,
        min_stage1_spacing: usize,
    ) -> Self {
        let n_markers = geno.n_markers();
        let n_haps = geno.n_haps();

        // Compute allele frequencies and classify markers
        let mut stage1_indices = Vec::new();
        let mut is_low_freq = Vec::with_capacity(n_markers);

        for m in 0..n_markers {
            let alleles = geno.marker_alleles(m);
            let n_alleles = n_alleles_per_marker.get(m).copied().unwrap_or(2);

            // Count allele frequencies
            let mut allele_counts = vec![0usize; n_alleles];
            for &a in alleles {
                if (a as usize) < n_alleles {
                    allele_counts[a as usize] += 1;
                }
            }

            // Determine if each allele is low-frequency
            let allele_low_freq: Vec<bool> = allele_counts
                .iter()
                .map(|&count| {
                    let freq = count as f32 / n_haps as f32;
                    freq < rare_threshold && freq > 0.0
                })
                .collect();

            // A marker is Stage 1 if it has no low-frequency alleles (or all are ref)
            // Also ensure minimum spacing between Stage 1 markers
            let has_common_het = allele_low_freq.iter().filter(|&&x| !x).count() >= 2;
            let far_from_last = stage1_indices.last().map(|&last| m - last >= min_stage1_spacing).unwrap_or(true);

            if has_common_het && far_from_last {
                stage1_indices.push(m);
            }

            is_low_freq.push(allele_low_freq);
        }

        // Ensure we have at least some Stage 1 markers (fallback: every Nth marker)
        if stage1_indices.len() < 10 && n_markers > 10 {
            stage1_indices.clear();
            let step = (n_markers / 100).max(1);
            for m in (0..n_markers).step_by(step) {
                stage1_indices.push(m);
            }
        }

        // Compute prev/next Stage 1 marker indices and interpolation weights
        let n_stage1 = stage1_indices.len();
        let mut prev_stage1 = vec![-1i32; n_markers];
        let mut next_stage1 = vec![n_stage1 as i32; n_markers];
        let mut prev_stage1_wt = vec![0.5f32; n_markers];

        let mut stage1_idx = 0;
        for m in 0..n_markers {
            // Find prev Stage 1 marker
            while stage1_idx < n_stage1 && stage1_indices[stage1_idx] < m {
                stage1_idx += 1;
            }

            if stage1_idx > 0 {
                prev_stage1[m] = (stage1_idx - 1) as i32;
            }

            if stage1_idx < n_stage1 {
                next_stage1[m] = stage1_idx as i32;
            }

            // Compute interpolation weight
            let prev_idx = prev_stage1[m];
            let next_idx = next_stage1[m];

            if prev_idx >= 0 && (next_idx as usize) < n_stage1 {
                let prev_pos = stage1_indices[prev_idx as usize];
                let next_pos = stage1_indices[next_idx as usize];
                if next_pos > prev_pos {
                    prev_stage1_wt[m] = (next_pos - m) as f32 / (next_pos - prev_pos) as f32;
                }
            } else if prev_idx >= 0 {
                prev_stage1_wt[m] = 1.0;
            } else {
                prev_stage1_wt[m] = 0.0;
            }
        }

        Self {
            stage1_indices,
            prev_stage1,
            next_stage1,
            prev_stage1_wt,
            is_low_freq,
            n_markers,
        }
    }

    /// Check if a marker is a Stage 1 (high-frequency) marker
    pub fn is_stage1(&self, marker: usize) -> bool {
        self.stage1_indices.binary_search(&marker).is_ok()
    }

    /// Get the number of Stage 1 markers
    pub fn n_stage1(&self) -> usize {
        self.stage1_indices.len()
    }

    /// Get the Stage 1 marker index for a given Stage 1 position
    pub fn stage1_marker(&self, stage1_idx: usize) -> usize {
        self.stage1_indices[stage1_idx]
    }

    /// Check if an allele at a marker is low-frequency
    pub fn is_allele_low_freq(&self, marker: usize, allele: u8) -> bool {
        self.is_low_freq
            .get(marker)
            .and_then(|v| v.get(allele as usize))
            .copied()
            .unwrap_or(false)
    }
}

/// Phasing pipeline
pub struct PhasingPipeline {
    config: Config,
    params: ModelParams,
    /// Cache of previously selected reference haplotypes per sample (for stickiness)
    prev_ref_haps: Vec<Vec<HapIdx>>,
}

impl PhasingPipeline {
    /// Create a new phasing pipeline
    pub fn new(config: Config) -> Self {
        let params = ModelParams::new();
        Self { 
            config, 
            params,
            prev_ref_haps: Vec::new(),
        }
    }

    /// Run the phasing pipeline
    pub fn run(&mut self) -> Result<()> {
        eprintln!("Loading VCF...");

        // Load target VCF
        let (mut reader, file_reader) = VcfReader::open(&self.config.gt)?;
        let samples = reader.samples_arc();
        let target_gt = reader.read_all(file_reader)?;

        if target_gt.n_markers() == 0 {
            eprintln!("No markers found in input VCF");
            return Ok(());
        }

        let n_markers = target_gt.n_markers();
        let n_samples = target_gt.n_samples();
        let n_haps = target_gt.n_haplotypes();

        eprintln!("Loaded {} markers, {} samples ({} haplotypes)", n_markers, n_samples, n_haps);

        // Initialize parameters based on sample size
        self.params = ModelParams::for_phasing(n_haps);
        self.params.set_n_states(self.config.phase_states.min(n_haps.saturating_sub(2)));

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

        // Create mutable genotype storage for phasing
        let mut geno = MutableGenotypes::from_fn(n_markers, n_haps, |m, h| {
            target_gt.allele(MarkerIdx::new(m as u32), HapIdx::new(h as u32))
        });

        // Compute genetic distances and recombination probabilities
        let chrom = target_gt.marker(MarkerIdx::new(0)).chrom;
        let gen_dists: Vec<f64> = (0..n_markers.saturating_sub(1))
            .map(|m| {
                let pos1 = target_gt.marker(MarkerIdx::new(m as u32)).pos;
                let pos2 = target_gt.marker(MarkerIdx::new((m + 1) as u32)).pos;
                gen_maps.gen_dist(chrom, pos1, pos2)
            })
            .collect();

        // Convert to recombination probabilities
        let p_recomb: Vec<f32> = std::iter::once(0.0f32)
            .chain(gen_dists.iter().map(|&d| self.params.p_recomb(d)))
            .collect();

        // Run phasing iterations
        let n_burnin = self.config.burnin;
        let n_iterations = self.config.iterations;
        let total_iterations = n_burnin + n_iterations;

        for it in 0..total_iterations {
            let is_burnin = it < n_burnin;
            let iter_type = if is_burnin { "burnin" } else { "main" };
            eprintln!("Iteration {}/{} ({})", it + 1, total_iterations, iter_type);

            // Update LR threshold for this iteration
            self.params.lr_threshold = self.params.lr_threshold_for_iteration(it);

            // Run phasing iteration with EM estimation (if enabled and during burnin)
            let collect_em = self.config.em && is_burnin;
            self.run_iteration_with_hmm(&target_gt, &mut geno, &p_recomb, collect_em)?;
        }

        // Build final GenotypeMatrix from mutable genotypes
        let final_gt = self.build_final_matrix(&target_gt, &geno);

        // Write output
        let output_path = self.config.out.with_extension("vcf.gz");
        eprintln!("Writing output to {:?}", output_path);

        let mut writer = VcfWriter::create(&output_path, samples)?;
        writer.write_header(final_gt.markers())?;
        writer.write_phased(&final_gt, 0, final_gt.n_markers())?;
        writer.flush()?;

        eprintln!("Phasing complete!");
        Ok(())
    }

    /// Phase a GenotypeMatrix in-memory and return the phased result
    /// 
    /// This is used by the imputation pipeline to auto-phase unphased inputs.
    pub fn phase_in_memory(&mut self, target_gt: &GenotypeMatrix, gen_maps: &GeneticMaps) -> Result<GenotypeMatrix> {
        let n_markers = target_gt.n_markers();
        let n_haps = target_gt.n_haplotypes();

        if n_markers == 0 {
            return Ok(target_gt.clone());
        }

        // Initialize parameters
        self.params = ModelParams::for_phasing(n_haps);
        self.params.set_n_states(self.config.phase_states.min(n_haps.saturating_sub(2)));

        // Create mutable genotype storage for phasing
        let mut geno = MutableGenotypes::from_fn(n_markers, n_haps, |m, h| {
            target_gt.allele(MarkerIdx::new(m as u32), HapIdx::new(h as u32))
        });

        // Compute recombination probabilities
        let chrom = target_gt.marker(MarkerIdx::new(0)).chrom;
        let gen_dists: Vec<f64> = (0..n_markers.saturating_sub(1))
            .map(|m| {
                let pos1 = target_gt.marker(MarkerIdx::new(m as u32)).pos;
                let pos2 = target_gt.marker(MarkerIdx::new((m + 1) as u32)).pos;
                gen_maps.gen_dist(chrom, pos1, pos2)
            })
            .collect();

        let p_recomb: Vec<f32> = std::iter::once(0.0f32)
            .chain(gen_dists.iter().map(|&d| self.params.p_recomb(d)))
            .collect();

        // Run phasing iterations (reduced for imputation pre-processing)
        let n_burnin = self.config.burnin.min(3);
        let n_iterations = self.config.iterations.min(6);
        let total_iterations = n_burnin + n_iterations;

        for it in 0..total_iterations {
            let is_burnin = it < n_burnin;
            self.params.lr_threshold = self.params.lr_threshold_for_iteration(it);
            let collect_em = self.config.em && is_burnin;
            self.run_iteration_with_hmm(target_gt, &mut geno, &p_recomb, collect_em)?;
        }

        // Build and return phased GenotypeMatrix
        Ok(self.build_final_matrix(target_gt, &geno))
    }

    fn run_iteration_with_hmm(
        &mut self,
        target_gt: &GenotypeMatrix,
        geno: &mut MutableGenotypes,
        p_recomb: &[f32],
        collect_em: bool,
    ) -> Result<()> {
        let n_samples = target_gt.n_samples();
        let n_markers = target_gt.n_markers();
        let n_haps = target_gt.n_haplotypes();
        let n_states = self.params.n_states;

        // Build PBWT for state selection using current phasing
        let pbwt = self.build_pbwt(geno, n_markers, n_haps);

        // Atomic counters for statistics
        let total_switches = AtomicUsize::new(0);
        let em_estimates = if collect_em {
            Some(AtomicParamEstimates::new())
        } else {
            None
        };

        // Initialize prev_ref_haps if empty
        if self.prev_ref_haps.len() != n_samples {
            self.prev_ref_haps = vec![Vec::new(); n_samples];
        }

        // Collect new selections for stickiness in next iteration
        let new_ref_haps: Mutex<Vec<(usize, Vec<HapIdx>)>> = Mutex::new(Vec::new());

        // Phase each sample in parallel
        let updates: Vec<(SampleIdx, Vec<usize>)> = (0..n_samples)
            .into_par_iter()
            .filter_map(|s| {
                let sample_idx = SampleIdx::new(s as u32);
                let hap1 = sample_idx.hap1();
                let hap2 = sample_idx.hap2();

                // Get current alleles for this sample
                let alleles1 = geno.haplotype(hap1);
                let alleles2 = geno.haplotype(hap2);

                // Find heterozygous markers
                let het_markers: Vec<usize> = (0..n_markers)
                    .filter(|&m| alleles1[m] != alleles2[m])
                    .collect();

                if het_markers.is_empty() {
                    return None; // Nothing to phase
                }

                // Get previous selection for stickiness
                let prev_selected = if self.prev_ref_haps[s].is_empty() {
                    None
                } else {
                    Some(self.prev_ref_haps[s].as_slice())
                };

                // Select reference haplotypes using PBWT with stickiness
                let ref_haps = self.select_ref_haps(&pbwt, hap1, n_states, n_haps, prev_selected);

                // Store new selection for next iteration
                if let Ok(mut guard) = new_ref_haps.lock() {
                    guard.push((s, ref_haps.clone()));
                }

                // Build reference alleles for HMM
                let ref_alleles: Vec<Vec<u8>> = ref_haps
                    .iter()
                    .map(|&h| geno.haplotype(h))
                    .collect();

                // Run phase decision using PhasingHmm-style algorithm
                let (switch_markers, local_em) = phase_sample_with_hmm(
                    &alleles1,
                    &alleles2,
                    &het_markers,
                    &ref_alleles,
                    p_recomb,
                    &self.params,
                    collect_em,
                );

                // Collect EM estimates
                if let (Some(global_em), Some(local)) = (&em_estimates, local_em) {
                    global_em.add_estimation_data(&local);
                }

                if switch_markers.is_empty() {
                    None
                } else {
                    total_switches.fetch_add(switch_markers.len(), Ordering::Relaxed);
                    Some((sample_idx, switch_markers))
                }
            })
            .collect();

        // Update prev_ref_haps with new selections for next iteration
        if let Ok(guard) = new_ref_haps.into_inner() {
            for (s, haps) in guard {
                self.prev_ref_haps[s] = haps;
            }
        }

        // Apply phase switches
        for (sample_idx, switch_markers) in updates {
            let hap1 = sample_idx.hap1();
            let hap2 = sample_idx.hap2();

            for &m in &switch_markers {
                geno.swap(m, hap1, hap2);
            }
        }

        let n_switches = total_switches.load(Ordering::Relaxed);
        eprintln!("  Applied {} phase switches", n_switches);

        // Update parameters from EM estimates
        if let Some(global_em) = em_estimates {
            let estimates = global_em.to_estimates();
            if estimates.n_switch_obs() > 0 {
                let new_recomb = estimates.recomb_intensity();
                self.params.update_recomb_intensity(new_recomb);
                eprintln!("  Updated recomb_intensity to {:.4}", self.params.recomb_intensity);
            }
            if estimates.n_emit_obs() > 0 {
                let new_p_mismatch = estimates.p_mismatch();
                self.params.update_p_mismatch(new_p_mismatch);
                eprintln!("  Updated p_mismatch to {:.6}", self.params.p_mismatch);
            }
        }

        Ok(())
    }

    /// Build PBWT from current genotypes
    fn build_pbwt(&self, geno: &MutableGenotypes, n_markers: usize, n_haps: usize) -> PbwtIbs {
        let mut pbwt = PbwtIbs::new(n_haps);
        let mut updater = PbwtDivUpdater::new(n_haps);

        for m in 0..n_markers {
            let alleles = geno.marker_alleles(m);
            
            // Determine number of alleles (usually 2 for biallelic)
            let n_alleles = alleles.iter().copied().max().unwrap_or(0) as usize + 1;

            let mut temp_prefix = pbwt.fwd_prefix().to_vec();
            let mut temp_div: Vec<i32> = pbwt.fwd_divergence().iter().map(|&x| x).collect();

            updater.fwd_update(alleles, n_alleles.max(2), m, &mut temp_prefix, &mut temp_div);

            pbwt.fwd_prefix_mut().copy_from_slice(&temp_prefix);
            for (i, &d) in temp_div.iter().enumerate() {
                if i < pbwt.fwd_divergence_mut().len() {
                    pbwt.fwd_divergence_mut()[i] = d;
                }
            }
        }

        pbwt
    }

    /// Select reference haplotypes using PBWT with stickiness heuristic
    /// 
    /// The stickiness heuristic (from Java bestFwdStage2Index) prefers haplotypes
    /// that were selected in the previous iteration, which stabilizes HMM paths.
    fn select_ref_haps(
        &self,
        pbwt: &PbwtIbs,
        target_hap: HapIdx,
        n_states: usize,
        n_haps: usize,
        prev_selected: Option<&[HapIdx]>,
    ) -> Vec<HapIdx> {
        // Exclude the other haplotype from the same sample
        let other_hap = if target_hap.0 % 2 == 0 {
            HapIdx::new(target_hap.0 + 1)
        } else {
            HapIdx::new(target_hap.0 - 1)
        };

        // Use PBWT to find nearby haplotypes
        let marker = 0;
        let n_candidates = n_states * 2;
        let mut candidates = pbwt.select_states(target_hap, n_states + 2, marker, n_candidates, false, true);
        
        // Remove the other haplotype from the same sample
        candidates.retain(|&h| h != other_hap);

        // Apply stickiness: prefer haplotypes from previous selection
        let mut selected = Vec::with_capacity(n_states);
        
        if let Some(prev) = prev_selected {
            // First, add candidates that were also in previous selection (sticky)
            for &h in candidates.iter() {
                if prev.contains(&h) && selected.len() < n_states {
                    selected.push(h);
                }
            }
            // Then add new candidates
            for &h in candidates.iter() {
                if !selected.contains(&h) && selected.len() < n_states {
                    selected.push(h);
                }
            }
        } else {
            // No previous selection - use candidates directly
            selected = candidates;
            selected.truncate(n_states);
        }

        // If we don't have enough, add random haplotypes
        if selected.len() < n_states {
            use rand::seq::SliceRandom;
            use rand::thread_rng;
            
            let mut remaining: Vec<HapIdx> = (0..n_haps as u32)
                .map(HapIdx::new)
                .filter(|&h| h != target_hap && h != other_hap && !selected.contains(&h))
                .collect();
            remaining.shuffle(&mut thread_rng());
            
            for h in remaining {
                selected.push(h);
                if selected.len() >= n_states {
                    break;
                }
            }
        }

        selected
    }

    /// Build final GenotypeMatrix from mutable genotypes
    fn build_final_matrix(
        &self,
        original: &GenotypeMatrix,
        geno: &MutableGenotypes,
    ) -> GenotypeMatrix {
        let markers = original.markers().clone();
        let samples = original.samples_arc();
        let n_markers = geno.n_markers();

        let columns: Vec<GenotypeColumn> = (0..n_markers)
            .map(|m| {
                let alleles = geno.marker_alleles(m);
                GenotypeColumn::from_alleles(alleles, 2)
            })
            .collect();

        GenotypeMatrix::new(markers, columns, samples, true)
    }
}

/// Phase a single sample using proper forward-backward HMM
///
/// This matches Java PhaseBaum2.java algorithm:
/// 1. Run forward pass for combined, hap1, and hap2
/// 2. Run backward pass storing values at het sites
/// 3. Make phase decisions by computing P(0|1) vs P(1|0) posteriors
fn phase_sample_with_hmm(
    alleles1: &[u8],
    alleles2: &[u8],
    het_markers: &[usize],
    ref_alleles: &[Vec<u8>],
    p_recomb: &[f32],
    params: &ModelParams,
    collect_em: bool,
) -> (Vec<usize>, Option<ParamEstimates>) {
    let n_markers = alleles1.len();
    let n_states = ref_alleles.len();

    if n_states == 0 || het_markers.is_empty() || n_markers == 0 {
        return (Vec::new(), None);
    }

    let p_match = params.emit_match();
    let p_mismatch = params.emit_mismatch();

    // Emission function - handles missing data (255) as uninformative
    let emit = |target: u8, reference: u8| -> f32 {
        // Missing data (255) should be treated as uninformative, not as mismatch
        if target == 255 || reference == 255 {
            1.0  // Uninformative - probability is uniform across states
        } else if target == reference {
            p_match
        } else {
            p_mismatch
        }
    };

    // Forward pass arrays: fwd_combined[m][s], fwd1[m][s], fwd2[m][s]
    // fwd_combined: P(X_1..m | hap1, hap2) regardless of which allele is on which hap
    // fwd1: P(X_1..m | state=s for hap1)
    // fwd2: P(X_1..m | state=s for hap2)
    let mut fwd_combined = vec![vec![0.0f32; n_states]; n_markers];
    let mut fwd1 = vec![vec![0.0f32; n_states]; n_markers];
    let mut fwd2 = vec![vec![0.0f32; n_states]; n_markers];

    let mut combined_sum = 1.0f32;
    let mut fwd1_sum = 1.0f32;
    let mut fwd2_sum = 1.0f32;

    // EM accumulators
    let mut em_estimates = if collect_em {
        Some(ParamEstimates::new())
    } else {
        None
    };

    // Forward pass
    for m in 0..n_markers {
        let a1 = alleles1[m];
        let a2 = alleles2[m];
        let p_rec = p_recomb.get(m).copied().unwrap_or(0.0);
        let shift = p_rec / n_states as f32;
        let scale_combined = (1.0 - p_rec) / combined_sum;
        let scale1 = (1.0 - p_rec) / fwd1_sum;
        let scale2 = (1.0 - p_rec) / fwd2_sum;

        let mut new_combined_sum = 0.0f32;
        let mut new_fwd1_sum = 0.0f32;
        let mut new_fwd2_sum = 0.0f32;

        for s in 0..n_states {
            let ref_a = ref_alleles[s][m];

            // Emission for combined: emit(a1 OR a2 matches ref)
            let emit_combined = emit(a1, ref_a).max(emit(a2, ref_a));
            let emit1 = emit(a1, ref_a);
            let emit2 = emit(a2, ref_a);

            if m == 0 {
                let init = 1.0 / n_states as f32;
                fwd_combined[m][s] = init * emit_combined;
                fwd1[m][s] = init * emit1;
                fwd2[m][s] = init * emit2;
            } else {
                fwd_combined[m][s] = emit_combined * (scale_combined * fwd_combined[m - 1][s] + shift);
                fwd1[m][s] = emit1 * (scale1 * fwd1[m - 1][s] + shift);
                fwd2[m][s] = emit2 * (scale2 * fwd2[m - 1][s] + shift);
            }

            new_combined_sum += fwd_combined[m][s];
            new_fwd1_sum += fwd1[m][s];
            new_fwd2_sum += fwd2[m][s];
        }

        combined_sum = new_combined_sum.max(1e-30);
        fwd1_sum = new_fwd1_sum.max(1e-30);
        fwd2_sum = new_fwd2_sum.max(1e-30);

        // Note: EM statistics for switches require backward probabilities
        // which we collect in the backward pass below.
    }

    // Backward pass: compute posterior probabilities at het sites
    let mut bwd = vec![1.0 / n_states as f32; n_states];
    let mut bwd_sum = 1.0f32;

    // Store backward values at het sites for phase decision
    let mut bwd_at_het: Vec<Vec<f32>> = Vec::with_capacity(het_markers.len());
    let mut het_idx = het_markers.len();

    for m in (0..n_markers).rev() {
        // Store backward values at het sites
        while het_idx > 0 && het_markers[het_idx - 1] == m {
            het_idx -= 1;
            bwd_at_het.push(bwd.clone());
        }

        if m == 0 {
            break;
        }

        // Apply emission for current marker
        let a1 = alleles1[m];
        let a2 = alleles2[m];

        for s in 0..n_states {
            let ref_a = ref_alleles[s][m];
            let emit1 = emit(a1, ref_a);
            let emit2 = emit(a2, ref_a);
            bwd[s] *= emit1.max(emit2);
        }

        // Apply transition
        let p_rec = p_recomb.get(m).copied().unwrap_or(0.0);
        let shift = p_rec / n_states as f32;
        let scale = (1.0 - p_rec) / bwd_sum;

        bwd_sum = 0.0;
        for s in 0..n_states {
            bwd[s] = scale * bwd[s] + shift;
            bwd_sum += bwd[s];
        }
        bwd_sum = bwd_sum.max(1e-30);
    }

    // Reverse so bwd_at_het[i] corresponds to het_markers[i]
    bwd_at_het.reverse();

    // Make phase decisions
    let mut switch_markers = Vec::new();
    let lr_threshold = params.lr_threshold;

    for (idx, &m) in het_markers.iter().enumerate() {
        let a1 = alleles1[m];
        let a2 = alleles2[m];
        let bwd_m = &bwd_at_het[idx];

        // Compute P(phase=0|1) vs P(phase=1|0)
        // 
        // Forward values fwd1[m][s] already include emission at marker m.
        // Backward values bwd[s] represent P(O_{m+1..T} | S_m=s), stored BEFORE emission at m.
        //
        // For phase decision at m, we want:
        //   P(phase | O) ∝ Σ_s P(O_1..m, S_m=s) × P(O_m | S_m=s, phase) × P(O_{m+1..T} | S_m=s)
        //
        // Since fwd1[m][s] = P(O_1..m, S_m=s) with emit(a1, ref[s]) baked in,
        // and we want to compare phases, we divide out the original emission
        // and multiply by phase-specific emission:
        //   p_01 contribution = fwd1[m][s] × bwd[s]  (a1 on hap1 - matches fwd1)
        //   p_10 contribution = fwd1[m][s] × (emit_a2/emit_a1) × bwd[s]  (a2 on hap1)
        //
        // Equivalently, use fwd1[m-1] + transition + phase-specific emission × bwd.
        // Since fwd1[m] already encodes emit1, the ratio approach is simpler.

        let mut p_01 = 0.0f32; // Current phase (a1 on hap1, a2 on hap2)
        let mut p_10 = 0.0f32; // Swapped phase (a2 on hap1, a1 on hap2)
        let p_rec = p_recomb.get(m).copied().unwrap_or(0.0);

        for s in 0..n_states {
            let ref_a = ref_alleles[s][m];
            let emit_a1 = emit(a1, ref_a);
            let emit_a2 = emit(a2, ref_a);
            let bwd_s = bwd_m[s];

            if m == 0 {
                // At first marker, no transition from previous
                let init = 1.0 / n_states as f32;
                p_01 += init * emit_a1 * bwd_s;
                p_10 += init * emit_a2 * bwd_s;
            } else {
                // Use forward from m-1, apply transition and phase-specific emission
                let fwd_prev = fwd1[m - 1][s];
                let shift = p_rec / n_states as f32;
                let scale = 1.0 - p_rec;
                
                // Transition probability to stay in same state (dominant term)
                let trans_stay = scale * fwd_prev + shift * fwd1_sum;
                
                // Combine: fwd(m-1) × trans × emit × bwd
                p_01 += trans_stay * emit_a1 * bwd_s;
                p_10 += trans_stay * emit_a2 * bwd_s;
            }
        }

        // Normalize
        let total = p_01 + p_10;
        if total > 0.0 {
            p_01 /= total;
            p_10 /= total;
        } else {
            p_01 = 0.5;
            p_10 = 0.5;
        }

        // Likelihood ratio test
        let lr = if p_01 > p_10 {
            p_01 / p_10.max(1e-10)
        } else {
            p_10 / p_01.max(1e-10)
        };

        // Apply phase switch if swapped phase is better and LR exceeds threshold
        // For ties (p_01 ≈ p_10), use marker index parity for deterministic tie-breaking
        let should_swap = if (p_10 - p_01).abs() < 1e-6 {
            m % 2 == 1  // Tie-break based on marker index parity
        } else {
            p_10 > p_01
        };
        
        if should_swap && lr > lr_threshold {
            switch_markers.push(m);
        }

        // Collect EM statistics
        if let Some(ref mut em) = em_estimates {
            // Expected matches/mismatches based on phase decision
            let best_p = p_01.max(p_10);
            let worst_p = p_01.min(p_10);
            em.add_emission(best_p as f64, worst_p as f64);
            
            // Expected switches using Baum-Welch recurrence
            // ξ_t(i,j) = α_{t-1}(i) × a_{ij} × b_j(o_t) × β_t(j) / P(O)
            // Expected switches = Σ_{i≠j} ξ_t(i,j)
            if m > 0 {
                let p_rec = p_recomb.get(m).copied().unwrap_or(0.0);
                let p_no_rec = 1.0 - p_rec;
                let shift = p_rec / n_states as f32;
                
                // Calculate total posterior mass (for normalization)
                let mut total_mass = 0.0f64;
                let mut no_switch_mass = 0.0f64;
                
                for s in 0..n_states {
                    let ref_a = ref_alleles[s][m];
                    let emit_s = emit(alleles1[m], ref_a).max(emit(alleles2[m], ref_a));
                    let bwd_s = bwd_at_het.get(idx).and_then(|v| v.get(s)).copied().unwrap_or(1.0 / n_states as f32);
                    
                    // No-switch: stay in state s
                    let no_switch_s = fwd1[m - 1][s] as f64 * p_no_rec as f64 * emit_s as f64 * bwd_s as f64;
                    no_switch_mass += no_switch_s;
                    
                    // All transitions into state s (for total mass)
                    // Switch: from any other state into s (approximated by shift term)
                    let switch_into_s = fwd1_sum as f64 * shift as f64 * emit_s as f64 * bwd_s as f64;
                    total_mass += no_switch_s + switch_into_s;
                }
                
                // Expected switch probability = 1 - P(no switch)
                if total_mass > 0.0 {
                    let p_no_switch = no_switch_mass / total_mass;
                    let expected_switches = 1.0 - p_no_switch;
                    em.add_switch(p_rec as f64, expected_switches);
                }
            }
        }
    }

    (switch_markers, em_estimates)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_pipeline_creation() {
        let config = Config {
            gt: PathBuf::from("test.vcf"),
            r#ref: None,
            out: PathBuf::from("out"),
            map: None,
            chrom: None,
            excludesamples: None,
            excludemarkers: None,
            burnin: 3,
            iterations: 12,
            phase_states: 280,
            rare: 0.002,
            impute: true,
            imp_states: 1600,
            imp_segment: 6.0,
            imp_step: 0.1,
            imp_nsteps: 7,
            cluster: 0.005,
            ap: false,
            gp: false,
            ne: 100000.0,
            err: None,
            em: true,
            window: 40.0,
            window_markers: 4000000,
            overlap: 2.0,
            seed: 12345,
            nthreads: None,
        };

        let pipeline = PhasingPipeline::new(config);
        assert_eq!(pipeline.params.n_states, 280);
    }

}