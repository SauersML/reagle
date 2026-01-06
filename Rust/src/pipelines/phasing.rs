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

use bitvec::prelude::*;
use rayon::prelude::*;

use crate::config::Config;
use crate::data::genetic_map::{GeneticMaps, MarkerMap};
use crate::data::haplotype::{HapIdx, SampleIdx};
use crate::data::marker::MarkerIdx;
use crate::data::storage::{GenotypeColumn, GenotypeMatrix, MutableGenotypes, GenotypeView};
use crate::error::Result;
use crate::io::streaming::{StreamingConfig, StreamingVcfReader};
use crate::io::vcf::{VcfReader, VcfWriter};
use crate::model::ibs2::Ibs2;
use crate::model::hmm::LiStephensHmm;
use crate::model::parameters::ModelParams;
use crate::model::phase_ibs::GlobalPhaseIbs;

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

        // Load exclusion lists
        let exclude_samples = self.config.load_exclude_samples()?;
        let exclude_markers = self.config.load_exclude_markers()?;

        if !exclude_samples.is_empty() {
            eprintln!("Excluding {} samples", exclude_samples.len());
        }
        if !exclude_markers.is_empty() {
            eprintln!("Excluding {} markers", exclude_markers.len());
        }

        // Load target VCF with filtering
        let (mut reader, file_reader) = VcfReader::open(&self.config.gt)?;
        reader.set_exclude_samples(&exclude_samples);
        reader.set_exclude_markers(exclude_markers);
        let samples = reader.samples_arc();
        let target_gt = reader.read_all(file_reader)?;

        if target_gt.n_markers() == 0 {
            eprintln!("No markers found in input VCF");
            return Ok(());
        }

        let n_markers = target_gt.n_markers();
        let n_samples = target_gt.n_samples();
        let n_haps = target_gt.n_haplotypes();

        eprintln!(
            "Loaded {} markers, {} samples ({} haplotypes), {:.2} MB",
            n_markers,
            n_samples,
            n_haps,
            target_gt.size_bytes() as f64 / 1024.0 / 1024.0
        );

        // Initialize parameters based on sample size
        self.params = ModelParams::for_phasing(n_haps);
        self.params
            .set_n_states(self.config.phase_states.min(n_haps.saturating_sub(2)));

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
        // 1. Pre-compute missing mask (row-major: [hap][marker]) and init geno
        // Optimize iteration order: iterate markers (cols), update haps (rows)
        let mut missing_mask_vecs: Vec<BitVec<u8, Lsb0>> = vec![BitVec::with_capacity(n_markers); n_haps];
        let mut geno = MutableGenotypes::new(n_markers, n_haps);

        for m in 0..n_markers {
            let m_idx = MarkerIdx::new(m as u32);
            for h in 0..n_haps {
                let h_idx = HapIdx::new(h as u32);
                let allele = target_gt.allele(m_idx, h_idx);
                missing_mask_vecs[h].push(allele == 255);
                if allele != 0 && allele != 255 {
                    geno.set(m, h_idx, 1);
                }
            }
        }
        let missing_mask: Vec<BitBox<u8, Lsb0>> = missing_mask_vecs
            .into_iter()
            .map(|v| v.into_boxed_bitslice())
            .collect();

        // Compute genetic distances and recombination probabilities using MarkerMap
        // This handles map interpolation and minimum distance enforcement
        let chrom = target_gt.marker(MarkerIdx::new(0)).chrom;
        let marker_map = if let Some(map) = gen_maps.get(chrom) {
            MarkerMap::create(target_gt.markers(), map)
        } else {
            MarkerMap::from_positions(target_gt.markers())
        };

        let gen_positions: Vec<f64> = marker_map.gen_positions().to_vec();

        // Compute MAF for each marker (used by IBS2 and two-stage phasing)
        let maf: Vec<f32> = (0..n_markers)
            .map(|m| target_gt.column(MarkerIdx::new(m as u32)).maf() as f32)
            .collect();

        // TWO-STAGE PHASING: Classify markers by frequency
        // Stage 1 (high-frequency): Run full HMM - these markers provide phasing signal
        // Stage 2 (rare): Interpolate from flanking high-frequency markers
        let rare_threshold = self.config.rare;
        let hi_freq_markers: Vec<usize> = (0..n_markers)
            .filter(|&m| maf[m] >= rare_threshold)
            .collect();
        let rare_markers: Vec<usize> = (0..n_markers)
            .filter(|&m| maf[m] < rare_threshold && maf[m] > 0.0) // Exclude monomorphic
            .collect();

        let n_hi_freq = hi_freq_markers.len();
        eprintln!(
            "Two-stage phasing: {} high-frequency markers (MAF >= {}), {} rare markers",
            n_hi_freq,
            rare_threshold,
            rare_markers.len()
        );

        // Create mapping from hi-freq index to original index
        let hi_freq_to_orig: Vec<usize> = hi_freq_markers.clone();

        // Compute genetic distances only for HIGH-FREQUENCY markers
        // This is critical: recombination probabilities must be computed for the
        // inter-marker distances between consecutive hi-freq markers, not all markers
        let stage1_gen_dists: Vec<f64> = if hi_freq_markers.len() > 1 {
            hi_freq_markers
                .windows(2)
                .map(|w| gen_positions[w[1]] - gen_positions[w[0]])
                .collect()
        } else {
            Vec::new()
        };

        // Build IBS2 segments for phase consistency (uses PositionMap fallback if no --map)
        eprintln!("Building IBS2 segments...");
        let ibs2 = Ibs2::new(&target_gt, &gen_maps, chrom, &maf);
        let n_with_ibs2 = (0..n_samples)
            .filter(|&s| ibs2.n_segments(crate::data::haplotype::SampleIdx::new(s as u32)) > 0)
            .count();
        eprintln!(
            "Found {} samples with IBS2 segments, {} total",
            n_with_ibs2,
            ibs2.n_samples()
        );

        // Run phasing iterations (STAGE 1: high-frequency markers only)
        let n_burnin = self.config.burnin;
        let n_iterations = self.config.iterations;
        let total_iterations = n_burnin + n_iterations;

        // Recombination probabilities - mutable so EM can update them
        let mut stage1_p_recomb: Vec<f32> = std::iter::once(0.0f32)
            .chain(stage1_gen_dists.iter().map(|&d| self.params.p_recomb(d)))
            .collect();

        for it in 0..total_iterations {
            let is_burnin = it < n_burnin;
            let iter_type = if is_burnin { "burnin" } else { "main" };
            eprintln!("Iteration {}/{} ({})", it + 1, total_iterations, iter_type);

            // Update LR threshold for this iteration
            self.params.lr_threshold = self.params.lr_threshold_for_iteration(it);

            // Run phasing iteration with EM estimation (if enabled and during burnin)
            let atomic_estimates = if is_burnin && self.config.em {
                Some(crate::model::parameters::AtomicParamEstimates::new())
            } else {
                None
            };

            self.run_phase_baum_iteration_stage1(
                &target_gt,
                &mut geno,
                &missing_mask,
                &stage1_p_recomb,
                &stage1_gen_dists,
                &hi_freq_to_orig,
                &ibs2,
                atomic_estimates.as_ref(),
            )?;

            // Update parameters from EM estimates and recompute recombination probabilities
            if let Some(ref atomic) = atomic_estimates {
                let est = atomic.to_estimates();
                let mut params_updated = false;
                
                if est.n_emit_obs() > 0 {
                    self.params.update_p_mismatch(est.p_mismatch());
                    params_updated = true;
                }
                if est.n_switch_obs() > 0 {
                    self.params.update_recomb_intensity(est.recomb_intensity());
                    params_updated = true;
                }
                
                // Recompute recombination probabilities with updated intensity
                if params_updated {
                    stage1_p_recomb = std::iter::once(0.0f32)
                        .chain(stage1_gen_dists.iter().map(|&d| self.params.p_recomb(d)))
                        .collect();
                }
                
                eprintln!(
                    "  EM update: p_mismatch={:.6}, recomb_intensity={:.4}",
                    self.params.p_mismatch, self.params.recomb_intensity
                );
            }
        }

        // STAGE 2: Interpolate rare marker phases from flanking high-frequency markers
        // Uses genetic-distance-weighted interpolation for improved accuracy
        if !rare_markers.is_empty() {
            eprintln!(
                "Stage 2: Phasing {} rare markers by interpolation...",
                rare_markers.len()
            );
            // Pass genetic positions for recombination-based weighting
            let gen_pos_ref = if !gen_positions.is_empty() {
                Some(gen_positions.as_slice())
            } else {
                None
            };
            self.phase_rare_markers(&mut geno, &rare_markers, &hi_freq_markers, gen_pos_ref);
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
    pub fn run_streaming(&mut self) -> Result<()> {
        eprintln!("Opening VCF for streaming...");

        // Configure streaming (genetic maps loaded lazily by StreamingVcfReader)
        let streaming_config = StreamingConfig {
            window_cm: self.config.window,
            overlap_cm: self.config.overlap,
            ..Default::default()
        };

        // Load genetic maps - use empty maps if no map file provided
        let gen_maps = if let Some(ref map_path) = self.config.map {
            GeneticMaps::from_plink_file(
                map_path,
                &[
                    "chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9",
                    "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17",
                    "chr18", "chr19", "chr20", "chr21", "chr22", "chrX", "1", "2", "3", "4", "5",
                    "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
                    "20", "21", "22", "X",
                ],
            )?
        } else {
            GeneticMaps::new()
        };

        // Open streaming reader
        let mut reader =
            StreamingVcfReader::open(&self.config.gt, gen_maps.clone(), streaming_config)?;
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
                "Processing window {} ({} markers, output {}..{} in global {}:{}..{})",
                window_count,
                n_markers,
                window.output_start,
                window.output_end,
                window.window_num,
                window.global_start,
                window.global_end
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
        eprintln!(
            "Streaming phasing complete: {} windows, {} markers",
            window_count, total_markers
        );
        Ok(())
    }

    /// Automatically select between in-memory and streaming mode based on data size
    pub fn run_auto(&mut self) -> Result<()> {
        if self.config.streaming == Some(true) {
            return self.run_streaming();
        }

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
    pub fn phase_in_memory(
        &mut self,
        target_gt: &GenotypeMatrix,
        gen_maps: &GeneticMaps,
    ) -> Result<GenotypeMatrix<crate::data::storage::phase_state::Phased>> {
        let n_markers = target_gt.n_markers();
        let n_haps = target_gt.n_haplotypes();

        if n_markers == 0 {
            return Ok(target_gt.clone().into_phased());
        }

        self.params = ModelParams::for_phasing(n_haps);
        self.params
            .set_n_states(self.config.phase_states.min(n_haps.saturating_sub(2)));

        let mut missing_mask_vecs: Vec<BitVec<u8, Lsb0>> = vec![BitVec::with_capacity(n_markers); n_haps];
        let mut geno = MutableGenotypes::new(n_markers, n_haps);

        for m in 0..n_markers {
            let m_idx = MarkerIdx::new(m as u32);
            for h in 0..n_haps {
                let h_idx = HapIdx::new(h as u32);
                let allele = target_gt.allele(m_idx, h_idx);
                missing_mask_vecs[h].push(allele == 255);
                if allele != 0 && allele != 255 {
                    geno.set(m, h_idx, 1);
                }
            }
        }
        let missing_mask: Vec<BitBox<u8, Lsb0>> = missing_mask_vecs
            .into_iter()
            .map(|v| v.into_boxed_bitslice())
            .collect();

        let chrom = target_gt.marker(MarkerIdx::new(0)).chrom;
        let gen_dists: Vec<f64> = (0..n_markers.saturating_sub(1))
            .map(|m| {
                let pos1 = target_gt.marker(MarkerIdx::new(m as u32)).pos;
                let pos2 = target_gt.marker(MarkerIdx::new((m + 1) as u32)).pos;
                gen_maps.gen_dist(chrom, pos1, pos2)
            })
            .collect();

        let maf: Vec<f32> = (0..n_markers)
            .map(|m| target_gt.column(MarkerIdx::new(m as u32)).maf() as f32)
            .collect();

        let ibs2 = Ibs2::new(target_gt, gen_maps, chrom, &maf);

        let n_burnin = self.config.burnin.min(3);
        let n_iterations = self.config.iterations.min(6);
        let total_iterations = n_burnin + n_iterations;

        // Recombination probabilities - mutable so EM can update them
        let mut p_recomb: Vec<f32> = std::iter::once(0.0f32)
            .chain(gen_dists.iter().map(|&d| self.params.p_recomb(d)))
            .collect();

        for it in 0..total_iterations {
            let is_burnin = it < n_burnin;
            self.params.lr_threshold = self.params.lr_threshold_for_iteration(it);
            
            let atomic_estimates = if is_burnin && self.config.em {
                Some(crate::model::parameters::AtomicParamEstimates::new())
            } else {
                None
            };

            self.run_phase_baum_iteration(
                &target_gt,
                &mut geno,
                &missing_mask,
                &p_recomb,
                &gen_dists,
                &ibs2,
                atomic_estimates.as_ref(),
            )?;

            // Update parameters from EM estimates and recompute recombination probabilities
            if let Some(ref atomic) = atomic_estimates {
                let est = atomic.to_estimates();
                let mut params_updated = false;
                
                if est.n_emit_obs() > 0 {
                    self.params.update_p_mismatch(est.p_mismatch());
                    params_updated = true;
                }
                if est.n_switch_obs() > 0 {
                    self.params.update_recomb_intensity(est.recomb_intensity());
                    params_updated = true;
                }
                
                // Recompute recombination probabilities with updated intensity
                if params_updated {
                    p_recomb = std::iter::once(0.0f32)
                        .chain(gen_dists.iter().map(|&d| self.params.p_recomb(d)))
                        .collect();
                }
            }
        }

        Ok(self.build_final_matrix(target_gt, &geno))
    }

    /// Select optimal HMM states using PBWT-based dynamic state selection
    ///
    /// This method uses GlobalPhaseIbs to find IBS neighbors at a specific marker,
    /// incorporating divergence information for accurate long-match detection.
    /// This is closer to Beagle's dynamic state selection approach.
    ///
    /// # Arguments
    /// * `hap_idx` - The haplotype to find neighbors for
    /// * `marker_idx` - Current marker index for localized IBS matching
    /// * `phase_ibs` - Global PBWT state tracker
    /// * `ibs2` - IBS2 segments for guaranteed long-range matches
    /// * `n_states_wanted` - Number of reference haplotypes to select
    /// * `n_total_haps` - Total haplotypes available
    fn select_states_pbwt(
        &self,
        hap_idx: HapIdx,
        marker_idx: usize,
        phase_ibs: &GlobalPhaseIbs,
        ibs2: &Ibs2,
        n_states_wanted: usize,
        n_total_haps: usize,
    ) -> Vec<HapIdx> {
        // Use PBWT neighbor finding with divergence-aware selection
        let neighbors = phase_ibs.find_neighbors(
            hap_idx.0,
            marker_idx,
            ibs2,
            n_states_wanted,
        );

        let mut states: Vec<HapIdx> = neighbors.into_iter().map(HapIdx::new).collect();

        // Fill remaining if PBWT didn't find enough neighbors
        if states.len() < n_states_wanted {
            let sample = SampleIdx::new(hap_idx.0 / 2);
            let mut i = 0;
            while states.len() < n_states_wanted && i < n_total_haps {
                let h = HapIdx::new(i as u32);
                if h != sample.hap1() && h != sample.hap2() && !states.contains(&h) {
                    states.push(h);
                }
                i += 1;
            }
        }

        states.truncate(n_states_wanted);
        states
    }

    /// Build and maintain PBWT state for dynamic state selection
    ///
    /// Creates a GlobalPhaseIbs and updates it with allele data, returning it
    /// for use in select_states_pbwt.
    fn build_phase_pbwt(&self, geno: &MutableGenotypes, n_markers: usize, n_haps: usize) -> GlobalPhaseIbs {
        let mut phase_ibs = GlobalPhaseIbs::new(n_haps);

        // Advance PBWT through all markers to build the divergence array
        for m in 0..n_markers {
            // Collect alleles for this marker
            let mut alleles = Vec::with_capacity(n_haps);
            for h in 0..n_haps {
                alleles.push(geno.get(m, HapIdx::new(h as u32)));
            }
            phase_ibs.advance(&alleles, m);
        }

        phase_ibs
    }

    /// Run a single phasing iteration using Forward-Backward Li-Stephens HMM
    ///
    /// This uses the full Forward-Backward algorithm to compute posterior probabilities
    /// of the phase, ensuring that phasing decisions are informed by both upstream
    /// and downstream data.
    fn run_phase_baum_iteration(
        &mut self,
        target_gt: &GenotypeMatrix,
        geno: &mut MutableGenotypes,
        missing_mask: &[BitBox<u8, Lsb0>],
        p_recomb: &[f32],
        gen_dists: &[f64], // Pass genetic distances for EM
        ibs2: &Ibs2,
        atomic_estimates: Option<&crate::model::parameters::AtomicParamEstimates>,
    ) -> Result<()> {
        let n_samples = geno.n_haps() / 2;
        let n_markers = geno.n_markers();
        let n_haps = geno.n_haps();
        let markers = target_gt.markers();

        // Clone current genotypes to use as a frozen reference panel
        let ref_geno = geno.clone();
        let ref_view = GenotypeView::from((&ref_geno, markers));

        // Build PBWT for dynamic state selection (using current phased genotypes)
        let phase_ibs = self.build_phase_pbwt(&ref_geno, n_markers, n_haps);

        // Prepare swap masks (one BitVec per sample)
        let mut swap_masks: Vec<BitVec<u8, Lsb0>> = vec![BitVec::repeat(false, n_markers); n_samples];

        // Process samples in parallel
        swap_masks.par_iter_mut().enumerate().for_each(|(s, mask)| {
            let sample_idx = SampleIdx::new(s as u32);
            let hap1 = sample_idx.hap1();
            let hap2 = sample_idx.hap2();

            // Use midpoint marker for PBWT-based state selection
            let mid_marker = n_markers / 2;
            let states = self.select_states_pbwt(hap1, mid_marker, &phase_ibs, ibs2, self.params.n_states, n_haps);
            
            // 2. Extract current alleles for H1 and H2
            let seq1 = ref_geno.haplotype(hap1);
            let seq2 = ref_geno.haplotype(hap2);

            // 3. Run HMM Forward-Backward for H1
            let hmm1 = LiStephensHmm::new(ref_view, &self.params, states.clone(), p_recomb.to_vec());
            
            // Collect EM statistics if requested
            if let Some(atomic) = atomic_estimates {
                let mut local_est = crate::model::parameters::ParamEstimates::new();
                hmm1.collect_stats(&seq1, gen_dists, &mut local_est);
                
                let hmm2 = LiStephensHmm::new(ref_view, &self.params, states.clone(), p_recomb.to_vec());
                hmm2.collect_stats(&seq2, gen_dists, &mut local_est);
                
                atomic.add_estimation_data(&local_est);
            }

            let mut fwd1 = Vec::new();
            let mut bwd1 = Vec::new();
            hmm1.forward_backward_raw(&seq1, &mut fwd1, &mut bwd1);

            // 4. Run HMM Forward-Backward for H2
            let hmm2 = LiStephensHmm::new(ref_view, &self.params, states.clone(), p_recomb.to_vec());
            let mut fwd2 = Vec::new();
            let mut bwd2 = Vec::new();
            hmm2.forward_backward_raw(&seq2, &mut fwd2, &mut bwd2);

            // 5. Decide Phase using BLOCK-BASED phasing
            // Instead of per-marker decisions, identify contiguous heterozygous blocks
            // and compute block-level likelihoods to preserve linkage structure
            let n_states = states.len();
            
            // Identify heterozygous blocks
            let mut het_blocks: Vec<(usize, usize)> = Vec::new(); // (start, end) inclusive
            let mut block_start: Option<usize> = None;
            
            for m in 0..n_markers {
                let a1 = seq1[m];
                let a2 = seq2[m];
                let is_missing = missing_mask[hap1.as_usize()][m] || missing_mask[hap2.as_usize()][m];
                let is_het = !is_missing && a1 != a2;
                
                if is_het {
                    if block_start.is_none() {
                        block_start = Some(m);
                    }
                } else if let Some(start) = block_start {
                    het_blocks.push((start, m - 1));
                    block_start = None;
                }
            }
            if let Some(start) = block_start {
                het_blocks.push((start, n_markers - 1));
            }
            
            // Compute block-level likelihoods and decide phase for each block
            let mut current_swap = false;
            let lr_threshold = self.params.lr_threshold as f64;
            
            for (block_start_m, block_end_m) in het_blocks {
                // Compute log-likelihood ratio for the entire block
                // L_keep = Π_{m in block} (Σ_k fwd1[m,k] * bwd1[m,k]) * (Σ_k fwd2[m,k] * bwd2[m,k])
                // L_swap = Π_{m in block} (Σ_k fwd1[m,k] * bwd2[m,k]) * (Σ_k fwd2[m,k] * bwd1[m,k])
                // Use log-domain for numerical stability
                let mut log_l_keep = 0.0f64;
                let mut log_l_swap = 0.0f64;
                
                for m in block_start_m..=block_end_m {
                    let a1 = seq1[m];
                    let a2 = seq2[m];
                    let is_missing = missing_mask[hap1.as_usize()][m] || missing_mask[hap2.as_usize()][m];
                    
                    if is_missing || a1 == a2 {
                        continue; // Skip non-het markers in block
                    }
                    
                    let row_start = m * n_states;
                    let row_end = row_start + n_states;
                    
                    let f1 = &fwd1[row_start..row_end];
                    let b1 = &bwd1[row_start..row_end];
                    let f2 = &fwd2[row_start..row_end];
                    let b2 = &bwd2[row_start..row_end];

                    let mut s11 = 0.0f64;
                    let mut s22 = 0.0f64;
                    let mut s12 = 0.0f64;
                    let mut s21 = 0.0f64;

                    for k in 0..n_states {
                        s11 += f1[k] as f64 * b1[k] as f64;
                        s22 += f2[k] as f64 * b2[k] as f64;
                        s12 += f1[k] as f64 * b2[k] as f64;
                        s21 += f2[k] as f64 * b1[k] as f64;
                    }

                    // Avoid log(0) by adding small epsilon
                    let eps = 1e-300f64;
                    log_l_keep += (s11 * s22 + eps).ln();
                    log_l_swap += (s12 * s21 + eps).ln();
                }
                
                // Compare block-level likelihoods considering current phase state
                let (log_l_stay, log_l_flip) = if current_swap {
                    (log_l_swap, log_l_keep)
                } else {
                    (log_l_keep, log_l_swap)
                };
                
                // Flip if log(L_flip) >= log(L_stay) + log(threshold)
                let log_threshold = lr_threshold.ln();
                if log_l_flip >= log_l_stay + log_threshold {
                    current_swap = !current_swap;
                }
                
                // Apply phase to entire block
                if current_swap {
                    for m in block_start_m..=block_end_m {
                        mask.set(m, true);
                    }
                }
            }
        });

        // Apply Swaps
        let mut total_switches = 0;
        for s in 0..n_samples {
            let mask = &swap_masks[s];
            if mask.any() {
                let hap1 = HapIdx::new((s * 2) as u32);
                let hap2 = HapIdx::new((s * 2 + 1) as u32);
                
                // Iterate true bits
                for m in mask.iter_ones() {
                    geno.swap(m, hap1, hap2);
                    total_switches += 1;
                }
            }
        }
        
        eprintln!("Applied {} phase switches (Forward-Backward)", total_switches);
        Ok(())
    }

    /// Run Stage 1 phasing iteration on HIGH-FREQUENCY markers only using FB HMM
    fn run_phase_baum_iteration_stage1(
        &mut self,
        target_gt: &GenotypeMatrix,
        geno: &mut MutableGenotypes,
        missing_mask: &[BitBox<u8, Lsb0>],
        stage1_p_recomb: &[f32],
        stage1_gen_dists: &[f64],
        hi_freq_to_orig: &[usize],
        ibs2: &Ibs2,
        atomic_estimates: Option<&crate::model::parameters::AtomicParamEstimates>,
    ) -> Result<()> {
        let n_samples = geno.n_haps() / 2;
        let n_haps = geno.n_haps();
        let n_markers = geno.n_markers();
        let markers = target_gt.markers();

        // 1. Create Subset View for Stage 1 markers
        let ref_geno = geno.clone();
        let subset_view = GenotypeView::MutableSubset {
            geno: &ref_geno,
            markers: markers,
            subset: hi_freq_to_orig,
        };

        // 2. Build PBWT for dynamic state selection (using current phased genotypes)
        let phase_ibs = self.build_phase_pbwt(&ref_geno, n_markers, n_haps);

        let mut swap_masks: Vec<BitVec<u8, Lsb0>> = vec![BitVec::repeat(false, hi_freq_to_orig.len()); n_samples];

        swap_masks.par_iter_mut().enumerate().for_each(|(s, mask)| {
            let sample_idx = SampleIdx::new(s as u32);
            let hap1 = sample_idx.hap1();
            let hap2 = sample_idx.hap2();

            // Use midpoint marker for PBWT-based state selection
            // This gives a representative set of IBS neighbors for the window
            let mid_marker = hi_freq_to_orig.len() / 2;
            let mid_orig = hi_freq_to_orig.get(mid_marker).copied().unwrap_or(0);
            let states = self.select_states_pbwt(hap1, mid_orig, &phase_ibs, ibs2, self.params.n_states, n_haps);
            
            // Extract Alleles for SUBSET of markers
            let seq1: Vec<u8> = hi_freq_to_orig.iter().map(|&m| ref_geno.get(m, hap1)).collect();
            let seq2: Vec<u8> = hi_freq_to_orig.iter().map(|&m| ref_geno.get(m, hap2)).collect();

            let hmm1 = LiStephensHmm::new(subset_view, &self.params, states.clone(), stage1_p_recomb.to_vec());
            
            // Collect EM statistics if requested
            if let Some(atomic) = atomic_estimates {
                let mut local_est = crate::model::parameters::ParamEstimates::new();
                hmm1.collect_stats(&seq1, stage1_gen_dists, &mut local_est);
                
                let hmm2 = LiStephensHmm::new(subset_view, &self.params, states.clone(), stage1_p_recomb.to_vec());
                hmm2.collect_stats(&seq2, stage1_gen_dists, &mut local_est);
                
                atomic.add_estimation_data(&local_est);
            }

            let mut fwd1 = Vec::new();
            let mut bwd1 = Vec::new();
            hmm1.forward_backward_raw(&seq1, &mut fwd1, &mut bwd1);

            let hmm2 = LiStephensHmm::new(subset_view, &self.params, states.clone(), stage1_p_recomb.to_vec());
            let mut fwd2 = Vec::new();
            let mut bwd2 = Vec::new();
            hmm2.forward_backward_raw(&seq2, &mut fwd2, &mut bwd2);

            // BLOCK-BASED phasing for Stage 1
            let n_states = states.len();
            let n_hi_freq = hi_freq_to_orig.len();
            
            // Identify heterozygous blocks in hi-freq marker space
            let mut het_blocks: Vec<(usize, usize)> = Vec::new(); // (start, end) inclusive in hi-freq index
            let mut block_start: Option<usize> = None;
            
            for i in 0..n_hi_freq {
                let m = hi_freq_to_orig[i];
                let a1 = seq1[i];
                let a2 = seq2[i];
                let is_missing = missing_mask[hap1.as_usize()][m] || missing_mask[hap2.as_usize()][m];
                let is_het = !is_missing && a1 != a2;
                
                if is_het {
                    if block_start.is_none() {
                        block_start = Some(i);
                    }
                } else if let Some(start) = block_start {
                    het_blocks.push((start, i - 1));
                    block_start = None;
                }
            }
            if let Some(start) = block_start {
                het_blocks.push((start, n_hi_freq - 1));
            }
            
            // Compute block-level likelihoods and decide phase for each block
            let mut current_swap = false;
            let lr_threshold = self.params.lr_threshold as f64;
            
            for (block_start_i, block_end_i) in het_blocks {
                let mut log_l_keep = 0.0f64;
                let mut log_l_swap = 0.0f64;
                
                for i in block_start_i..=block_end_i {
                    let m = hi_freq_to_orig[i];
                    let a1 = seq1[i];
                    let a2 = seq2[i];
                    let is_missing = missing_mask[hap1.as_usize()][m] || missing_mask[hap2.as_usize()][m];
                    
                    if is_missing || a1 == a2 {
                        continue;
                    }
                    
                    let row_start = i * n_states;
                    let row_end = row_start + n_states;
                    
                    let f1 = &fwd1[row_start..row_end];
                    let b1 = &bwd1[row_start..row_end];
                    let f2 = &fwd2[row_start..row_end];
                    let b2 = &bwd2[row_start..row_end];

                    let mut s11 = 0.0f64;
                    let mut s22 = 0.0f64;
                    let mut s12 = 0.0f64;
                    let mut s21 = 0.0f64;

                    for k in 0..n_states {
                        s11 += f1[k] as f64 * b1[k] as f64;
                        s22 += f2[k] as f64 * b2[k] as f64;
                        s12 += f1[k] as f64 * b2[k] as f64;
                        s21 += f2[k] as f64 * b1[k] as f64;
                    }

                    let eps = 1e-300f64;
                    log_l_keep += (s11 * s22 + eps).ln();
                    log_l_swap += (s12 * s21 + eps).ln();
                }
                
                let (log_l_stay, log_l_flip) = if current_swap {
                    (log_l_swap, log_l_keep)
                } else {
                    (log_l_keep, log_l_swap)
                };
                
                let log_threshold = lr_threshold.ln();
                if log_l_flip >= log_l_stay + log_threshold {
                    current_swap = !current_swap;
                }
                
                if current_swap {
                    for i in block_start_i..=block_end_i {
                        mask.set(i, true);
                    }
                }
            }
        });

        // Apply Swaps
        let mut total_switches = 0;
        for s in 0..n_samples {
            let mask = &swap_masks[s];
            if mask.any() {
                let hap1 = HapIdx::new((s * 2) as u32);
                let hap2 = HapIdx::new((s * 2 + 1) as u32);
                
                for i in mask.iter_ones() {
                    let m = hi_freq_to_orig[i];
                    geno.swap(m, hap1, hap2);
                    total_switches += 1;
                }
            }
        }
        
        eprintln!("Applied {} phase switches (Stage 1 FB)", total_switches);
        Ok(())
    }
    
    /// Build final GenotypeMatrix from mutable genotypes
    fn build_final_matrix(
        &self,
        original: &GenotypeMatrix,
        geno: &MutableGenotypes,
    ) -> GenotypeMatrix<crate::data::storage::phase_state::Phased> {
        let markers = original.markers().clone();
        let samples = original.samples_arc();
        let n_markers = geno.n_markers();

        let columns: Vec<GenotypeColumn> = (0..n_markers)
            .map(|m| {
                let alleles = geno.marker_alleles(m);
                let bytes: Vec<u8> = alleles.iter().map(|b| *b as u8).collect();
                GenotypeColumn::from_alleles(&bytes, 2)
            })
            .collect();

        GenotypeMatrix::new_phased(markers, columns, samples)
    }

    /// Stage 2: Phase rare markers by interpolation from flanking high-frequency markers
    /// Phase rare markers using interpolation from flanking high-frequency markers.
    ///
    /// This improved method uses genetic-distance-weighted interpolation following
    /// the approach in Java Beagle's Stage2Baum.java. The key idea is:
    ///
    /// 1. For each rare heterozygous marker, find flanking high-frequency het markers
    /// 2. Weight the phase evidence from each flanking marker by genetic distance
    /// 3. Use recombination probability to compute the likelihood of each phase
    /// 4. Choose the phase with higher likelihood
    ///
    /// The probability of no recombination between markers i and j is approximately:
    /// P(no_recomb) = exp(-recomb_intensity * genetic_distance)
    ///
    /// When genetic map is not available, falls back to physical-distance heuristic
    /// with assumed 1 cM/Mb rate.
    fn phase_rare_markers(
        &self,
        geno: &mut MutableGenotypes,
        rare_markers: &[usize],
        hi_freq_markers: &[usize],
        gen_positions: Option<&[f64]>,
    ) {
        let n_samples = geno.n_haps() / 2;

        if rare_markers.is_empty() || hi_freq_markers.is_empty() {
            return;
        }

        // Precompute recombination intensity
        let recomb_intensity = self.params.recomb_intensity as f64;

        for s in 0..n_samples {
            let sample_idx = SampleIdx::new(s as u32);
            let hap1 = sample_idx.hap1();
            let hap2 = sample_idx.hap2();

            let alleles1 = geno.haplotype(hap1);
            let alleles2 = geno.haplotype(hap2);

            // Find rare heterozygotes for this sample
            let rare_het: Vec<usize> = rare_markers
                .iter()
                .copied()
                .filter(|&m| alleles1[m] != alleles2[m] && alleles1[m] != 255 && alleles2[m] != 255)
                .collect();

            if rare_het.is_empty() {
                continue;
            }

            // Find high-frequency heterozygotes for this sample
            let hf_het: Vec<usize> = hi_freq_markers
                .iter()
                .copied()
                .filter(|&m| alleles1[m] != alleles2[m] && alleles1[m] != 255 && alleles2[m] != 255)
                .collect();

            if hf_het.is_empty() {
                continue;
            }

            for &rare_m in &rare_het {
                let left_hf = hf_het.iter().copied().filter(|&m| m < rare_m).max();
                let right_hf = hf_het.iter().copied().filter(|&m| m > rare_m).min();

                // Calculate phase probabilities using genetic distance weighting
                let (p_keep, p_swap) = match (left_hf, right_hf) {
                    (Some(left), Some(right)) => {
                        // Both flanking markers available - interpolate using
                        // probability of no recombination from each side
                        let left_hap1_allele = alleles1[left];
                        let right_hap1_allele = alleles1[right];

                        // Genetic distances (or fallback to physical distance as proxy)
                        let (gen_dist_left, gen_dist_right) = if let Some(gen_pos) = gen_positions {
                            let d_left = (gen_pos[rare_m] - gen_pos[left]).abs();
                            let d_right = (gen_pos[right] - gen_pos[rare_m]).abs();
                            (d_left, d_right)
                        } else {
                            // Fallback: assume 1 cM per marker as rough proxy
                            ((rare_m - left) as f64 * 0.001, (right - rare_m) as f64 * 0.001)
                        };

                        // Probability of no recombination from each flanking marker
                        // P(no_recomb) = exp(-intensity * distance)
                        let p_no_recomb_left = (-recomb_intensity * gen_dist_left).exp();
                        let p_no_recomb_right = (-recomb_intensity * gen_dist_right).exp();

                        // Phase contribution from left marker
                        // If hap1 has REF at left (allele=0), phase_from_left suggests hap1 should
                        // continue with same pattern. If hap1 has ALT (allele=1), suggests swap.
                        let left_suggests_keep = left_hap1_allele == 0;
                        let right_suggests_keep = right_hap1_allele == 0;

                        // Weight the suggestions by recombination probability
                        // Higher p_no_recomb means more confidence from that flanking marker
                        let weight_left = p_no_recomb_left;
                        let weight_right = p_no_recomb_right;
                        let total_weight = weight_left + weight_right;

                        if total_weight < 1e-10 {
                            // Both flanking markers too far - no strong evidence
                            (0.5, 0.5)
                        } else {
                            // Normalized weighted vote
                            let p_keep = (if left_suggests_keep { weight_left } else { 0.0 }
                                + if right_suggests_keep { weight_right } else { 0.0 })
                                / total_weight;
                            let p_swap = 1.0 - p_keep;
                            (p_keep, p_swap)
                        }
                    }
                    (Some(left), None) => {
                        // Only left flanking marker
                        let left_suggests_keep = alleles1[left] == 0;
                        if left_suggests_keep {
                            (0.9, 0.1) // High confidence to keep
                        } else {
                            (0.1, 0.9) // High confidence to swap
                        }
                    }
                    (None, Some(right)) => {
                        // Only right flanking marker
                        let right_suggests_keep = alleles1[right] == 0;
                        if right_suggests_keep {
                            (0.9, 0.1)
                        } else {
                            (0.1, 0.9)
                        }
                    }
                    (None, None) => continue,
                };

                // Decide based on probability comparison
                let rare_hap1_has_0 = alleles1[rare_m] == 0;
                let should_swap = p_swap > p_keep;

                // Apply swap if needed
                // If should_swap XOR rare_hap1_has_0, we need to actually swap
                if should_swap == rare_hap1_has_0 {
                    let a1 = alleles1[rare_m];
                    let a2 = alleles2[rare_m];
                    geno.set(rare_m, hap1, a2);
                    geno.set(rare_m, hap2, a1);
                }
            }
        }
    }
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
            em: false, // Disable EM for unit test to avoid complexity
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
    
    #[test]
    fn test_run_phase() {
        // Create a small pipeline and run phase_in_memory
        use crate::data::storage::matrix::GenotypeMatrix;
        use crate::data::genetic_map::GeneticMaps;
        use crate::data::marker::{Marker, Allele, Markers};
        use crate::data::haplotype::Samples;
        use crate::data::storage::GenotypeColumn;
        use crate::data::ChromIdx;
        use std::sync::Arc;
        
        let n_markers = 50;
        let n_samples = 10;
        
        // Mock Markers
        let mut markers = Markers::new();
        markers.add_chrom("chr1");
        
        for i in 0..n_markers {
            let m = Marker::new(
                ChromIdx::new(0), 
                i as u32 * 1000, 
                Some(format!("m{}", i).into()), 
                Allele::Base(b'A'), 
                vec![Allele::Base(b'T')]
            );
            markers.push(m);
        }
            
        // Mock Samples
        let samples = Arc::new(Samples::from_ids(
            (0..n_samples).map(|i| format!("s{}", i)).collect()
        ));
            
        // Mock Genotypes (Random)
        let columns: Vec<GenotypeColumn> = (0..n_markers)
            .map(|_| {
                let bytes: Vec<u8> = (0..n_samples*2).map(|i| (i % 3) as u8).collect();
                GenotypeColumn::from_alleles(&bytes, 2)
            })
            .collect();
            
        let gt = GenotypeMatrix::new_unphased(
            markers, 
            columns, 
            samples
        );
        
        // Mock Genetic Map (Empty uses default linear rate)
        let gen_maps = GeneticMaps::new(); 
        
        let config = Config {
            gt: PathBuf::from("test.vcf"),
            r#ref: None,
            out: PathBuf::from("out"),
            map: None,
            chrom: None,
            excludesamples: None,
            excludemarkers: None,
            burnin: 2,
            iterations: 2,
            phase_states: 10,
            rare: 0.002,
            impute: true,
            imp_states: 10,
            imp_segment: 6.0,
            imp_step: 0.1,
            imp_nsteps: 7,
            cluster: 0.005,
            ap: false,
            gp: false,
            ne: 10000.0,
            err: None,
            em: false,
            window: 40.0,
            window_markers: 4000000,
            overlap: 2.0,
            streaming: None,
            seed: 12345,
            nthreads: Some(2),
        };
        
        let mut pipeline = PhasingPipeline::new(config);
        
        // Run phasing
        let result = pipeline.phase_in_memory(&gt, &gen_maps);
        
        assert!(result.is_ok());
        let phased = result.unwrap();
        assert_eq!(phased.n_markers(), n_markers);
        assert_eq!(phased.n_haplotypes(), n_samples * 2);
    }
}