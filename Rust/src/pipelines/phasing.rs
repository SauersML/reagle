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
use crate::data::storage::sample_phase::SamplePhase;
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

        // Create SamplePhase instances to track phase state
        let mut sample_phases = self.create_sample_phases(&geno, &missing_mask);

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
                &mut sample_phases,
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

        // Sync final phase state from SamplePhase to MutableGenotypes
        self.sync_sample_phases_to_geno(&sample_phases, &mut geno);

        // STAGE 2: Phase rare markers using HMM state probability interpolation
        // This implements the proper algorithm from Java Beagle's Stage2Baum.java
        if !rare_markers.is_empty() && hi_freq_markers.len() >= 2 {
            eprintln!(
                "Stage 2: Phasing {} rare markers using HMM interpolation...",
                rare_markers.len()
            );
            self.phase_rare_markers_with_hmm(
                &target_gt,
                &mut geno,
                &hi_freq_markers,
                &gen_positions,
                &stage1_p_recomb,
                &ibs2,
                &mut sample_phases,
            );
            
            // Sync again after Stage 2
            self.sync_sample_phases_to_geno(&sample_phases, &mut geno);
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

    /// Create SamplePhase instances for all samples
    ///
    /// This initializes phase tracking state from the current genotype data.
    fn create_sample_phases(
        &self,
        geno: &MutableGenotypes,
        missing_mask: &[BitBox<u8, Lsb0>],
    ) -> Vec<SamplePhase> {
        let n_samples = geno.n_haps() / 2;
        let n_markers = geno.n_markers();

        (0..n_samples)
            .map(|s| {
                let hap1 = HapIdx::new((s * 2) as u32);
                let hap2 = HapIdx::new((s * 2 + 1) as u32);

                let alleles1: Vec<u8> = (0..n_markers).map(|m| geno.get(m, hap1)).collect();
                let alleles2: Vec<u8> = (0..n_markers).map(|m| geno.get(m, hap2)).collect();

                // Identify missing markers
                let missing: Vec<usize> = (0..n_markers)
                    .filter(|&m| missing_mask[hap1.as_usize()][m] || missing_mask[hap2.as_usize()][m])
                    .collect();

                // Initially all hets are unphased
                let unphased: Vec<usize> = (0..n_markers)
                    .filter(|&m| {
                        let a1 = alleles1[m];
                        let a2 = alleles2[m];
                        a1 != a2
                            && !missing_mask[hap1.as_usize()][m]
                            && !missing_mask[hap2.as_usize()][m]
                    })
                    .collect();

                SamplePhase::new(s as u32, n_markers, &alleles1, &alleles2, &unphased, &missing)
            })
            .collect()
    }

    /// Sync SamplePhase alleles back to MutableGenotypes
    fn sync_sample_phases_to_geno(
        &self,
        sample_phases: &[SamplePhase],
        geno: &mut MutableGenotypes,
    ) {
        let n_markers = geno.n_markers();

        for (s, sp) in sample_phases.iter().enumerate() {
            let hap1 = HapIdx::new((s * 2) as u32);
            let hap2 = HapIdx::new((s * 2 + 1) as u32);

            for m in 0..n_markers {
                let a1 = sp.allele1(m);
                let a2 = sp.allele2(m);
                geno.set(m, hap1, a1);
                geno.set(m, hap2, a2);
            }
        }
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
    ///
    /// Uses SamplePhase to track phase state and only phases unphased markers.
    fn run_phase_baum_iteration_stage1(
        &mut self,
        target_gt: &GenotypeMatrix,
        geno: &mut MutableGenotypes,
        _missing_mask: &[BitBox<u8, Lsb0>],
        stage1_p_recomb: &[f32],
        stage1_gen_dists: &[f64],
        hi_freq_to_orig: &[usize],
        ibs2: &Ibs2,
        sample_phases: &mut [SamplePhase],
        atomic_estimates: Option<&crate::model::parameters::AtomicParamEstimates>,
    ) -> Result<()> {
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

        // Collect phase decisions per sample: Vec<(hi_freq_idx, swap, lr)>
        type PhaseDecision = (usize, bool, f64); // (hi_freq_idx, should_swap, log_lr)
        let phase_decisions: Vec<Vec<PhaseDecision>> = sample_phases
            .par_iter()
            .enumerate()
            .map(|(s, sp)| {
                let sample_idx = SampleIdx::new(s as u32);
                let hap1 = sample_idx.hap1();

                // Use midpoint marker for PBWT-based state selection
                let mid_marker = hi_freq_to_orig.len() / 2;
                let mid_orig = hi_freq_to_orig.get(mid_marker).copied().unwrap_or(0);
                let states = self.select_states_pbwt(hap1, mid_orig, &phase_ibs, ibs2, self.params.n_states, n_haps);

                // Extract alleles from SamplePhase for SUBSET of markers
                let seq1: Vec<u8> = hi_freq_to_orig.iter().map(|&m| sp.allele1(m)).collect();
                let seq2: Vec<u8> = hi_freq_to_orig.iter().map(|&m| sp.allele2(m)).collect();

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

                // Identify UNPHASED heterozygous blocks in hi-freq marker space
                let mut het_blocks: Vec<(usize, usize)> = Vec::new(); // (start, end) inclusive
                let mut block_start: Option<usize> = None;

                for i in 0..n_hi_freq {
                    let m = hi_freq_to_orig[i];
                    let a1 = seq1[i];
                    let a2 = seq2[i];
                    let is_het = a1 != a2;
                    // Only consider UNPHASED heterozygotes
                    let is_unphased_het = is_het && sp.is_unphased(m);

                    if is_unphased_het {
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

                // Compute block-level likelihoods and collect phase decisions
                let mut decisions = Vec::new();
                let mut current_swap = false;
                let lr_threshold = self.params.lr_threshold as f64;

                for (block_start_i, block_end_i) in het_blocks {
                    let mut log_l_keep = 0.0f64;
                    let mut log_l_swap = 0.0f64;
                    let mut has_unphased = false;

                    for i in block_start_i..=block_end_i {
                        let m = hi_freq_to_orig[i];
                        let a1 = seq1[i];
                        let a2 = seq2[i];

                        // Only count unphased hets in likelihood
                        if a1 == a2 || !sp.is_unphased(m) {
                            continue;
                        }
                        has_unphased = true;

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

                    if !has_unphased {
                        continue;
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

                    // Compute absolute log-LR for phasing confidence
                    let log_lr = (log_l_keep - log_l_swap).abs();

                    // Record decisions for each unphased marker in block
                    for i in block_start_i..=block_end_i {
                        let m = hi_freq_to_orig[i];
                        if sp.is_unphased(m) && seq1[i] != seq2[i] {
                            decisions.push((i, current_swap, log_lr));
                        }
                    }
                }

                decisions
            })
            .collect();

        // Apply phase decisions to SamplePhase
        let mut total_switches = 0;
        let mut total_phased = 0;
        let log_lr_threshold = (self.params.lr_threshold as f64).ln();

        for (s, decisions) in phase_decisions.into_iter().enumerate() {
            let sp = &mut sample_phases[s];

            for (hi_freq_idx, should_swap, log_lr) in decisions {
                let m = hi_freq_to_orig[hi_freq_idx];

                // Only process if still unphased
                if !sp.is_unphased(m) {
                    continue;
                }

                // Apply swap if needed
                if should_swap {
                    sp.swap_haps(m, m + 1);
                    total_switches += 1;
                }

                // Mark as phased if LR exceeds threshold
                if log_lr >= log_lr_threshold {
                    sp.mark_phased(m);
                    total_phased += 1;
                }
            }
        }

        // Also update MutableGenotypes to keep in sync for next iteration's PBWT
        self.sync_sample_phases_to_geno(sample_phases, geno);

        eprintln!(
            "Applied {} phase switches, {} markers phased (Stage 1 FB)",
            total_switches, total_phased
        );
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

    /// Stage 2: Phase rare markers using HMM state probability interpolation
    ///
    /// This implements the proper algorithm from Java Beagle's Stage2Baum.java:
    ///
    /// 1. Run HMM on high-frequency markers to get state probabilities for each haplotype
    /// 2. For each rare heterozygote:
    ///    - Find flanking high-frequency markers (mkrA, mkrB)
    ///    - Interpolate state probabilities: prob = wt*probsA[j] + (1-wt)*probsB[j]
    ///    - Accumulate allele probabilities from reference haplotypes
    /// 3. Decide phase: p1 = alProbs1[a1] * alProbs2[a2], p2 = alProbs1[a2] * alProbs2[a1]
    ///    Switch if p2 > p1
    ///
    /// **Key fix**: Only phases markers that are currently UNPHASED in SamplePhase.
    fn phase_rare_markers_with_hmm(
        &self,
        target_gt: &GenotypeMatrix,
        geno: &mut MutableGenotypes,
        hi_freq_markers: &[usize],
        gen_positions: &[f64],
        stage1_p_recomb: &[f32],
        ibs2: &Ibs2,
        sample_phases: &mut [SamplePhase],
    ) {
        let n_markers = geno.n_markers();
        let n_haps = geno.n_haps();
        let n_stage1 = hi_freq_markers.len();

        if n_stage1 < 2 {
            return;
        }

        // Build Stage 2 interpolation mappings
        let stage2_phaser = Stage2Phaser::new(hi_freq_markers, gen_positions, n_markers);

        // Clone current genotypes to use as a frozen reference panel
        let ref_geno = geno.clone();
        let markers = target_gt.markers();
        let subset_view = GenotypeView::MutableSubset {
            geno: &ref_geno,
            markers,
            subset: hi_freq_markers,
        };

        // Build PBWT for state selection
        let phase_ibs = self.build_phase_pbwt(&ref_geno, n_markers, n_haps);

        // Process samples in parallel - collect results: (marker, should_swap)
        let phase_changes: Vec<Vec<(usize, bool)>> = sample_phases
            .par_iter()
            .enumerate()
            .map(|(s, sp)| {
                let sample_idx = SampleIdx::new(s as u32);
                let hap1 = sample_idx.hap1();

                // Select HMM states using PBWT
                let mid_marker = n_stage1 / 2;
                let mid_orig = hi_freq_markers.get(mid_marker).copied().unwrap_or(0);
                let states = self.select_states_pbwt(hap1, mid_orig, &phase_ibs, ibs2, self.params.n_states, n_haps);
                let n_states = states.len();

                // Extract Stage 1 alleles from SamplePhase
                let seq1: Vec<u8> = hi_freq_markers.iter().map(|&m| sp.allele1(m)).collect();
                let seq2: Vec<u8> = hi_freq_markers.iter().map(|&m| sp.allele2(m)).collect();

                // Run HMM forward-backward for both haplotypes on Stage 1 markers
                let hmm1 = LiStephensHmm::new(subset_view, &self.params, states.clone(), stage1_p_recomb.to_vec());
                let mut fwd1 = Vec::new();
                let mut bwd1 = Vec::new();
                hmm1.forward_backward_raw(&seq1, &mut fwd1, &mut bwd1);

                let hmm2 = LiStephensHmm::new(subset_view, &self.params, states.clone(), stage1_p_recomb.to_vec());
                let mut fwd2 = Vec::new();
                let mut bwd2 = Vec::new();
                hmm2.forward_backward_raw(&seq2, &mut fwd2, &mut bwd2);

                // Compute posterior state probabilities at each Stage 1 marker
                let probs1 = compute_state_posteriors(&fwd1, &bwd1, n_stage1, n_states);
                let probs2 = compute_state_posteriors(&fwd2, &bwd2, n_stage1, n_states);

                // Collect phase decisions for this sample
                let mut phase_decisions = Vec::new();

                // Helper closure to process a marker
                let mut process_marker = |m: usize| {
                    // **KEY FIX**: Only process if marker is UNPHASED
                    if !sp.is_unphased(m) {
                        return;
                    }

                    let a1 = sp.allele1(m);
                    let a2 = sp.allele2(m);

                    // Only process heterozygotes
                    if a1 == a2 {
                        return;
                    }

                    // Compute interpolated allele probabilities for each haplotype
                    let al_probs1 = stage2_phaser.interpolated_allele_probs(
                        m, &probs1, &states, &ref_geno, n_haps, a1, a2,
                    );
                    let al_probs2 = stage2_phaser.interpolated_allele_probs(
                        m, &probs2, &states, &ref_geno, n_haps, a1, a2,
                    );

                    // p1 = P(hap1 has a1, hap2 has a2)
                    // p2 = P(hap1 has a2, hap2 has a1)
                    let p1 = al_probs1[a1 as usize] * al_probs2[a2 as usize];
                    let p2 = al_probs1[a2 as usize] * al_probs2[a1 as usize];

                    // Record decision: (marker, should_swap)
                    phase_decisions.push((m, p2 > p1));
                };

                // Process markers before first Stage 1 marker
                if !hi_freq_markers.is_empty() {
                    let first_hf = hi_freq_markers[0];
                    for m in 0..first_hf {
                        process_marker(m);
                    }
                }

                // Process all Stage 2 markers (rare markers between Stage 1 markers)
                for start_idx in 0..n_stage1 {
                    let start_m = hi_freq_markers[start_idx];
                    let end_m = if start_idx + 1 < n_stage1 {
                        hi_freq_markers[start_idx + 1]
                    } else {
                        n_markers
                    };

                    for m in (start_m + 1)..end_m {
                        process_marker(m);
                    }
                }

                phase_decisions
            })
            .collect();

        // Apply phase changes to SamplePhase
        let mut total_switches = 0;
        let mut total_phased = 0;

        for (s, decisions) in phase_changes.into_iter().enumerate() {
            let sp = &mut sample_phases[s];

            for (m, should_swap) in decisions {
                // Double-check still unphased (should always be true)
                if !sp.is_unphased(m) {
                    continue;
                }

                if should_swap {
                    sp.swap_haps(m, m + 1);
                    total_switches += 1;
                }

                // Mark as phased after Stage 2 processing
                sp.mark_phased(m);
                total_phased += 1;
            }
        }

        eprintln!(
            "Stage 2: Applied {} phase switches, {} markers phased (HMM interpolation)",
            total_switches, total_phased
        );
    }
}

/// Compute normalized posterior state probabilities from forward-backward arrays
fn compute_state_posteriors(
    fwd: &[f32],
    bwd: &[f32],
    n_markers: usize,
    n_states: usize,
) -> Vec<Vec<f32>> {
    let mut probs = vec![vec![0.0f32; n_states]; n_markers];

    for m in 0..n_markers {
        let row_start = m * n_states;
        let mut sum = 0.0f32;

        for k in 0..n_states {
            let p = fwd[row_start + k] * bwd[row_start + k];
            probs[m][k] = p;
            sum += p;
        }

        // Normalize
        if sum > 0.0 {
            for k in 0..n_states {
                probs[m][k] /= sum;
            }
        }
    }

    probs
}

/// Stage 2 phaser with HMM state probability interpolation
///
/// Implements the algorithm from Java Beagle's Stage2Baum.java for phasing
/// rare variants using interpolated HMM state probabilities.
struct Stage2Phaser {
    /// For each Stage 2 marker, the index of the preceding Stage 1 marker
    prev_stage1_marker: Vec<usize>,
    /// For each Stage 2 marker, the interpolation weight (0.0 to 1.0)
    /// wt = 1.0 means use prev marker fully, wt = 0.0 means use next marker fully
    prev_stage1_wt: Vec<f32>,
    /// Number of Stage 1 markers
    n_stage1: usize,
}

impl Stage2Phaser {
    /// Create a new Stage2Phaser
    ///
    /// # Arguments
    /// * `hi_freq_markers` - Indices of high-frequency (Stage 1) markers in original space
    /// * `gen_positions` - Genetic positions (cM) for all markers
    /// * `n_total_markers` - Total number of markers
    fn new(hi_freq_markers: &[usize], gen_positions: &[f64], n_total_markers: usize) -> Self {
        let n_stage1 = hi_freq_markers.len();

        // Build prevStage1Marker: for each marker, which Stage 1 marker precedes it
        let mut prev_stage1_marker = vec![0usize; n_total_markers];

        if n_stage1 >= 2 {
            // Fill markers before first Stage 1 marker with 0
            let first_hf = hi_freq_markers[0];
            for m in 0..=first_hf {
                prev_stage1_marker[m] = 0;
            }

            // Fill between Stage 1 markers
            for j in 1..n_stage1 {
                let prev_hf = hi_freq_markers[j - 1];
                let curr_hf = hi_freq_markers[j];
                for m in (prev_hf + 1)..=curr_hf {
                    prev_stage1_marker[m] = j - 1;
                }
            }

            // Fill after last Stage 1 marker
            let last_hf = hi_freq_markers[n_stage1 - 1];
            for m in (last_hf + 1)..n_total_markers {
                prev_stage1_marker[m] = n_stage1 - 1;
            }
        }

        // Build prevStage1Wt: interpolation weight based on genetic position
        // wt = (posB - pos) / (posB - posA) where posA is prev Stage1, posB is next Stage1
        let mut prev_stage1_wt = vec![1.0f32; n_total_markers];

        if n_stage1 >= 2 {
            // Markers before first Stage 1 marker: wt = 1.0 (use first marker)
            // Already initialized to 1.0

            // Between Stage 1 markers: interpolate
            for j in 0..(n_stage1 - 1) {
                let start = hi_freq_markers[j];
                let end = hi_freq_markers[j + 1];
                let pos_a = gen_positions[start];
                let pos_b = gen_positions[end];
                let d = pos_b - pos_a;

                prev_stage1_wt[start] = 1.0;

                if d > 1e-10 {
                    for m in (start + 1)..end {
                        prev_stage1_wt[m] = ((pos_b - gen_positions[m]) / d) as f32;
                    }
                } else {
                    // Zero distance, use equal weight
                    for m in (start + 1)..end {
                        prev_stage1_wt[m] = 0.5;
                    }
                }
            }

            // Markers at and after last Stage 1 marker: wt = 1.0
            let last_hf = hi_freq_markers[n_stage1 - 1];
            for m in last_hf..n_total_markers {
                prev_stage1_wt[m] = 1.0;
            }
        }

        Self {
            prev_stage1_marker,
            prev_stage1_wt,
            n_stage1,
        }
    }

    /// Compute interpolated allele probabilities for a rare marker
    ///
    /// Following Java Stage2Baum.unscaledAlProbs:
    /// - For each HMM state, interpolate probability from flanking Stage 1 markers
    /// - Accumulate allele probabilities based on reference haplotype alleles
    fn interpolated_allele_probs(
        &self,
        marker: usize,
        state_probs: &[Vec<f32>], // [stage1_marker][state]
        states: &[HapIdx],        // HMM state haplotype indices
        ref_geno: &MutableGenotypes,
        n_haps: usize,
        a1: u8,
        a2: u8,
    ) -> [f32; 2] {
        let mut al_probs = [0.0f32; 2];

        let mkr_a = self.prev_stage1_marker[marker];
        let mkr_b = (mkr_a + 1).min(self.n_stage1 - 1);
        let wt = self.prev_stage1_wt[marker];

        let probs_a = &state_probs[mkr_a];
        let probs_b = &state_probs[mkr_b];

        for (j, &hap_idx) in states.iter().enumerate() {
            let hap = hap_idx.0 as usize;
            
            // Get allele from reference haplotype at rare marker
            let b1 = ref_geno.get(marker, hap_idx);
            
            // Get allele from paired haplotype (for het reference haplotypes)
            let paired_hap = hap ^ 1;
            let b2 = if paired_hap < n_haps {
                ref_geno.get(marker, HapIdx::new(paired_hap as u32))
            } else {
                b1
            };

            if b1 == 255 || b2 == 255 {
                continue;
            }

            // Interpolate state probability
            let prob = wt * probs_a.get(j).copied().unwrap_or(0.0)
                + (1.0 - wt) * probs_b.get(j).copied().unwrap_or(0.0);

            if b1 == b2 {
                // Homozygous reference haplotype
                if b1 == a1 {
                    al_probs[0] += prob;
                } else if b1 == a2 {
                    al_probs[1] += prob;
                }
            } else {
                // Heterozygous reference haplotype - use rare allele matching
                // Following Java Stage2Baum logic for rare allele disambiguation
                let rare1 = a1; // In our simplified model, assume a1 could be rare
                let rare2 = a2;

                let match1 = rare1 == b1 || rare1 == b2;
                let match2 = rare2 == b1 || rare2 == b2;

                if match1 != match2 {
                    // Only one allele matches - use that information
                    if match1 {
                        al_probs[0] += prob;
                    } else {
                        al_probs[1] += prob;
                    }
                }
                // If both or neither match, no contribution (ambiguous)
            }
        }

        al_probs
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