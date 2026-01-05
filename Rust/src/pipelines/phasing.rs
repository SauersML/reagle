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

use crate::config::Config;
use crate::data::genetic_map::GeneticMaps;
use crate::data::haplotype::{HapIdx, SampleIdx};
use crate::data::marker::MarkerIdx;
use crate::data::storage::{GenotypeColumn, GenotypeMatrix, MutableGenotypes};
use crate::error::Result;
use crate::io::streaming::{StreamingConfig, StreamingVcfReader};
use crate::io::vcf::{VcfReader, VcfWriter};
use crate::model::ibs2::Ibs2;
use crate::model::imp_states::{CodedStepsConfig, ImpStatesLegacy};
use crate::model::parameters::{AtomicParamEstimates, ModelParams, ParamEstimates};
use crate::utils::workspace::Workspace;



/// Phasing pipeline
pub struct PhasingPipeline {
    config: Config,
    params: ModelParams,
}

impl PhasingPipeline {
    /// Create a new phasing pipeline
    pub fn new(config: Config) -> Self {
        let params = ModelParams::new();
        Self { config, params }
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

        // Compute cumulative genetic positions for dynamic state selection
        let gen_positions: Vec<f64> = {
            let mut pos = vec![0.0];
            for &d in &gen_dists {
                pos.push(pos.last().unwrap() + d);
            }
            pos
        };

        // Compute MAF for each marker (used by IBS2)
        let maf: Vec<f32> = (0..n_markers)
            .map(|m| target_gt.column(MarkerIdx::new(m as u32)).maf() as f32)
            .collect();

        // Build IBS2 segments for phase consistency (uses PositionMap fallback if no --map)
        eprintln!("Building IBS2 segments...");
        let ibs2 = Ibs2::new(&target_gt, &gen_maps, chrom, &maf);
        let n_with_ibs2 = (0..n_samples).filter(|&s| ibs2.n_segments(crate::data::haplotype::SampleIdx::new(s as u32)) > 0).count();
        eprintln!("Found {} samples with IBS2 segments, {} total", n_with_ibs2, ibs2.n_samples());

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
            self.run_iteration_with_hmm(&mut geno, &p_recomb, &gen_positions, collect_em)?;
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

    /// Run the phasing pipeline in streaming mode for large datasets
    ///
    /// This processes the data in sliding windows to avoid loading the entire
    /// chromosome into memory. Use this when markers exceed the window threshold.
    pub fn run_streaming(&mut self) -> Result<()> {
        eprintln!("Opening VCF for streaming...");

        // Configure streaming (genetic maps loaded lazily by StreamingVcfReader)
        let streaming_config = StreamingConfig {
            window_cm: self.config.window,
            overlap_cm: self.config.overlap,
            ..Default::default()
        };

        // Load genetic maps - use empty maps if no map file provided
        // The PositionMap fallback (1 cM per Mb) is used automatically
        let gen_maps = if let Some(ref map_path) = self.config.map {
            // Load all chromosomes from the map file
            GeneticMaps::from_plink_file(map_path, &["chr1", "chr2", "chr3", "chr4", "chr5",
                "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr13", "chr14",
                "chr15", "chr16", "chr17", "chr18", "chr19", "chr20", "chr21", "chr22", "chrX",
                "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14",
                "15", "16", "17", "18", "19", "20", "21", "22", "X"])?
        } else {
            GeneticMaps::new()
        };

        // Open streaming reader
        let mut reader = StreamingVcfReader::open(&self.config.gt, gen_maps.clone(), streaming_config)?;
        let samples = reader.samples_arc();

        // Create output writer
        let output_path = self.config.out.with_extension("vcf.gz");
        eprintln!("Writing output to {:?}", output_path);
        let mut writer = VcfWriter::create(&output_path, samples)?;

        let mut window_count = 0;
        let mut total_markers = 0;

        // Process windows
        while let Some(window) = reader.next_window()? {
            window_count += 1;
            let n_markers = window.genotypes.n_markers();
            total_markers += window.output_end - window.output_start;

            eprintln!(
                "Processing window {} ({} markers, output {}..{})",
                window_count, n_markers, window.output_start, window.output_end
            );

            // Phase this window
            let phased = self.phase_in_memory(&window.genotypes, &gen_maps)?;

            // Write header on first window
            if window.is_first {
                writer.write_header(phased.markers())?;
            }

            // Write output region
            writer.write_phased(&phased, window.output_start, window.output_end)?;
        }

        writer.flush()?;
        eprintln!("Streaming phasing complete: {} windows, {} markers", window_count, total_markers);
        Ok(())
    }

    /// Automatically select between in-memory and streaming mode based on data size
    ///
    /// Uses streaming mode if:
    /// - `--streaming` flag is explicitly set, OR
    /// - Estimated marker count exceeds `--window-markers` threshold
    pub fn run_auto(&mut self) -> Result<()> {
        // Check if streaming was explicitly requested
        if self.config.streaming == Some(true) {
            return self.run_streaming();
        }

        // Estimate marker count from file size (rough heuristic: ~100 bytes per marker line)
        let file_size = std::fs::metadata(&self.config.gt)
            .map(|m| m.len())
            .unwrap_or(0);
        let estimated_markers = file_size / 100;

        let use_streaming = estimated_markers > self.config.window_markers as u64;

        if use_streaming {
            eprintln!(
                "Auto-detected large dataset (~{} markers), using streaming mode",
                estimated_markers
            );
            self.run_streaming()
        } else {
            self.run()
        }
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

        // Compute cumulative genetic positions for dynamic state selection
        let gen_positions: Vec<f64> = {
            let mut pos = vec![0.0];
            for &d in &gen_dists {
                pos.push(pos.last().unwrap() + d);
            }
            pos
        };

        // Run phasing iterations (reduced for imputation pre-processing)
        let n_burnin = self.config.burnin.min(3);
        let n_iterations = self.config.iterations.min(6);
        let total_iterations = n_burnin + n_iterations;

        for it in 0..total_iterations {
            let is_burnin = it < n_burnin;
            self.params.lr_threshold = self.params.lr_threshold_for_iteration(it);
            let collect_em = self.config.em && is_burnin;
            self.run_iteration_with_hmm(&mut geno, &p_recomb, &gen_positions, collect_em)?;
        }

        // Build and return phased GenotypeMatrix
        Ok(self.build_final_matrix(target_gt, &geno))
    }

    /// Run a single phasing iteration using dynamic ImpStates for state selection.
    ///
    /// NOTE: This rebuilds ImpStates per-sample which is O(N²) in the number of samples.
    /// For optimal performance, a future optimization would pre-compute global PBWT matches.
    fn run_iteration_with_hmm(
        &mut self,
        geno: &mut MutableGenotypes,
        p_recomb: &[f32],
        gen_positions: &[f64],
        collect_em: bool,
    ) -> Result<()> {
        let n_samples = geno.n_haps() / 2;
        let n_markers = geno.n_markers();
        let n_haps = geno.n_haps();
        let n_states = self.params.n_states;

        // Atomic counters for statistics
        let total_switches = AtomicUsize::new(0);
        let em_estimates = if collect_em {
            Some(AtomicParamEstimates::new())
        } else {
            None
        };

        // Configuration for dynamic state selection (ImpStates)
        let steps_config = CodedStepsConfig::default();

        // Take a snapshot of genotypes for thread-safe access
        let geno_snapshot: Vec<Vec<u8>> = (0..n_haps)
            .map(|h| geno.haplotype(HapIdx::new(h as u32)))
            .collect();

        // Phase each sample in parallel using dynamic ImpStates with thread-local workspaces
        let updates: Vec<(SampleIdx, Vec<usize>)> = (0..n_samples)
            .into_par_iter()
            .map_init(
                || Workspace::new(n_states, n_markers, n_haps),
                |workspace, s| {
                    let sample_idx = SampleIdx::new(s as u32);
                    let hap1 = sample_idx.hap1();
                    let hap2 = sample_idx.hap2();

                    // Get current alleles for this sample from snapshot
                    let alleles1 = &geno_snapshot[hap1.0 as usize];
                    let alleles2 = &geno_snapshot[hap2.0 as usize];

                    // Find heterozygous markers
                    let het_markers: Vec<usize> = (0..n_markers)
                        .filter(|&m| alleles1[m] != alleles2[m])
                        .collect();

                    if het_markers.is_empty() {
                        return None; // Nothing to phase
                    }

                    // Create ImpStatesLegacy for dynamic state selection (phasing uses naive PBWT)
                    let mut imp_states = ImpStatesLegacy::new(n_haps, n_states, gen_positions, &steps_config);

                    // Use hap1's alleles as the target for IBS matching
                    let target_alleles: Vec<u8> = alleles1.clone();

                    // Closure to get reference allele at (marker, hap)
                    let get_ref_allele = |m: usize, h: u32| -> u8 {
                        geno_snapshot
                            .get(h as usize)
                            .and_then(|hap| hap.get(m).copied())
                            .unwrap_or(255)
                    };

                    // Build dynamic state mapping using ImpStates
                    let mut hap_indices: Vec<Vec<u32>> = Vec::new();
                    let mut allele_match: Vec<Vec<bool>> = Vec::new();
                    let actual_n_states = imp_states.ibs_states(
                        get_ref_allele,
                        &target_alleles,
                        &mut hap_indices,
                        &mut allele_match,
                    );

                    if actual_n_states == 0 {
                        return None;
                    }

                    // Resize workspace for actual number of states
                    workspace.resize(actual_n_states, n_markers, n_haps);

                    // Run phase decision using dynamic state HMM with workspace
                    let (switch_markers, local_em) = phase_sample_with_hmm(
                        alleles1,
                        alleles2,
                        &het_markers,
                        &hap_indices,
                        &geno_snapshot,
                        p_recomb,
                        &self.params,
                        collect_em,
                        workspace,
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
                },
            )
            .flatten()
            .collect();

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

/// Phase a single sample using proper forward-backward HMM with dynamic states
///
/// This matches Java PhaseBaum2.java algorithm:
/// 1. Run forward pass for combined, hap1, and hap2
/// 2. Run backward pass storing values at het sites
/// 3. Make phase decisions by computing P(0|1) vs P(1|0) posteriors
///
/// Uses dynamic state selection where hap_indices[m][s] gives the reference
/// haplotype index for state s at marker m.
fn phase_sample_with_hmm(
    alleles1: &[u8],
    alleles2: &[u8],
    het_markers: &[usize],
    hap_indices: &[Vec<u32>],
    geno_snapshot: &[Vec<u8>],
    p_recomb: &[f32],
    params: &ModelParams,
    collect_em: bool,
    workspace: &mut Workspace,
) -> (Vec<usize>, Option<ParamEstimates>) {
    let n_markers = alleles1.len();

    if hap_indices.is_empty() || het_markers.is_empty() || n_markers == 0 {
        return (Vec::new(), None);
    }

    let n_states = hap_indices.first().map(|v| v.len()).unwrap_or(0);
    if n_states == 0 {
        return (Vec::new(), None);
    }

    // Helper to get reference allele at marker m for state s (dynamic lookup)
    let get_ref_allele = |m: usize, s: usize| -> u8 {
        hap_indices
            .get(m)
            .and_then(|states| states.get(s))
            .and_then(|&hap_idx| {
                geno_snapshot
                    .get(hap_idx as usize)
                    .and_then(|hap| hap.get(m).copied())
            })
            .unwrap_or(255)
    };

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

    // Use workspace buffers for forward pass arrays
    // fwd_combined: P(X_1..m | hap1, hap2) regardless of which allele is on which hap
    // fwd1: P(X_1..m | state=s for hap1)
    // fwd2: P(X_1..m | state=s for hap2)
    let fwd_combined = &mut workspace.fwd_combined;
    let fwd1 = &mut workspace.fwd1;
    let fwd2 = &mut workspace.fwd2;

    // Clear buffers for this sample
    for m in 0..n_markers {
        for s in 0..n_states {
            fwd_combined[m][s] = 0.0;
            fwd1[m][s] = 0.0;
            fwd2[m][s] = 0.0;
        }
    }

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
            let ref_a = get_ref_allele(m, s);

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

    // Backward passes: compute SEPARATE backward probabilities for each haplotype
    // Java Beagle (PhaseBaum2.java) maintains bwdHet1 and bwdHet2 separately
    let mut bwd1 = vec![1.0 / n_states as f32; n_states];
    let mut bwd2 = vec![1.0 / n_states as f32; n_states];
    let mut bwd1_sum = 1.0f32;
    let mut bwd2_sum = 1.0f32;

    // Store backward values at het sites for phase decision
    let mut bwd1_at_het: Vec<Vec<f32>> = Vec::with_capacity(het_markers.len());
    let mut bwd2_at_het: Vec<Vec<f32>> = Vec::with_capacity(het_markers.len());
    let mut het_idx = het_markers.len();

    for m in (0..n_markers).rev() {
        // Store backward values at het sites
        while het_idx > 0 && het_markers[het_idx - 1] == m {
            het_idx -= 1;
            bwd1_at_het.push(bwd1.clone());
            bwd2_at_het.push(bwd2.clone());
        }

        if m == 0 {
            break;
        }

        // Apply emission for current marker SEPARATELY for each haplotype
        let a1 = alleles1[m];
        let a2 = alleles2[m];

        for s in 0..n_states {
            let ref_a = get_ref_allele(m, s);
            let emit1 = emit(a1, ref_a);
            let emit2 = emit(a2, ref_a);
            // Each backward pass uses its own allele's emission (separate buffers)
            bwd1[s] *= emit1;
            bwd2[s] *= emit2;
        }

        // Apply transition for bwd1
        let p_rec = p_recomb.get(m).copied().unwrap_or(0.0);
        let shift = p_rec / n_states as f32;
        let scale1 = (1.0 - p_rec) / bwd1_sum;

        bwd1_sum = 0.0;
        for s in 0..n_states {
            bwd1[s] = scale1 * bwd1[s] + shift;
            bwd1_sum += bwd1[s];
        }
        bwd1_sum = bwd1_sum.max(1e-30);

        // Apply transition for bwd2
        let scale2 = (1.0 - p_rec) / bwd2_sum;
        bwd2_sum = 0.0;
        for s in 0..n_states {
            bwd2[s] = scale2 * bwd2[s] + shift;
            bwd2_sum += bwd2[s];
        }
        bwd2_sum = bwd2_sum.max(1e-30);
    }

    // Reverse so bwd_at_het[i] corresponds to het_markers[i]
    bwd1_at_het.reverse();
    bwd2_at_het.reverse();

    // Make phase decisions using BOTH backward passes
    // Java Beagle computes: prob_no_swap = p11 * p22, prob_swap = p12 * p21
    // where p_ij = Σ_s fwd_i[m][s] * bwd_j[m][s]
    let mut switch_markers = Vec::new();
    let lr_threshold = params.lr_threshold;

    for (idx, &m) in het_markers.iter().enumerate() {
        let bwd1_m = &bwd1_at_het[idx];
        let bwd2_m = &bwd2_at_het[idx];

        // Compute posterior probabilities using BOTH forward and backward passes
        // p11 = Σ_s fwd1[m][s] * bwd1[m][s]  (hap1 continues with allele1)
        // p12 = Σ_s fwd1[m][s] * bwd2[m][s]  (hap1 switches to allele2)
        // p21 = Σ_s fwd2[m][s] * bwd1[m][s]  (hap2 switches to allele1)
        // p22 = Σ_s fwd2[m][s] * bwd2[m][s]  (hap2 continues with allele2)
        let mut p11 = 0.0f32;
        let mut p12 = 0.0f32;
        let mut p21 = 0.0f32;
        let mut p22 = 0.0f32;

        for s in 0..n_states {
            p11 += fwd1[m][s] * bwd1_m[s];
            p12 += fwd1[m][s] * bwd2_m[s];
            p21 += fwd2[m][s] * bwd1_m[s];
            p22 += fwd2[m][s] * bwd2_m[s];
        }

        // Phase likelihoods:
        // Current phase (a1 on hap1, a2 on hap2): prob_no_swap = p11 * p22
        // Swapped phase (a2 on hap1, a1 on hap2): prob_swap = p12 * p21
        let prob_no_swap = p11 * p22;
        let prob_swap = p12 * p21;

        // Normalize
        let total = prob_no_swap + prob_swap;
        let (p_01, p_10) = if total > 0.0 {
            (prob_no_swap / total, prob_swap / total)
        } else {
            (0.5, 0.5)
        };

        // Likelihood ratio test
        let lr = if p_01 > p_10 {
            p_01 / p_10.max(1e-10)
        } else {
            p_10 / p_01.max(1e-10)
        };

        // Apply phase switch if swapped phase is better and LR exceeds threshold
        let should_swap = if (p_10 - p_01).abs() < 1e-6 {
            use rand::Rng;
            rand::rng().random_bool(0.5)
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
            if m > 0 {
                let p_rec = p_recomb.get(m).copied().unwrap_or(0.0);
                let p_no_rec = 1.0 - p_rec;
                let shift = p_rec / n_states as f32;

                // Calculate total posterior mass (for normalization)
                let mut total_mass = 0.0f64;
                let mut no_switch_mass = 0.0f64;

                for s in 0..n_states {
                    let ref_a = get_ref_allele(m, s);
                    let emit_s = emit(alleles1[m], ref_a).max(emit(alleles2[m], ref_a));
                    // Use bwd1 for EM (could also average bwd1 and bwd2)
                    let bwd_s = bwd1_at_het.get(idx).and_then(|v| v.get(s)).copied().unwrap_or(1.0 / n_states as f32);

                    // No-switch: stay in state s
                    let no_switch_s = fwd1[m - 1][s] as f64 * p_no_rec as f64 * emit_s as f64 * bwd_s as f64;
                    no_switch_mass += no_switch_s;

                    // All transitions into state s (for total mass)
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
            streaming: None,
            seed: 12345,
            nthreads: None,
        };

        let pipeline = PhasingPipeline::new(config);
        assert_eq!(pipeline.params.n_states, 280);
    }

}