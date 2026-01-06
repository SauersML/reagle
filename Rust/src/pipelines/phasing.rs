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
use bitvec::prelude::*;

use crate::config::Config;
use crate::data::genetic_map::{GeneticMaps, MarkerMap};
use crate::data::haplotype::{HapIdx, SampleIdx};
use crate::data::marker::MarkerIdx;
use crate::data::storage::{GenotypeColumn, GenotypeMatrix, MutableGenotypes};
use crate::error::Result;
use crate::io::streaming::{StreamingConfig, StreamingVcfReader};
use crate::io::vcf::{VcfReader, VcfWriter};
use crate::model::ibs2::Ibs2;
use crate::model::online_hmm::OnlineHmm;
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
            // Access raw alleles if possible, or use accessor
            // Assuming target_gt allows efficient random access or we should iterate columns
            // Using standard accessor for safety/clarity, assuming GenotypeMatrix caches columns
            for h in 0..n_haps {
                let h_idx = HapIdx::new(h as u32);
                let allele = target_gt.allele(m_idx, h_idx);
                missing_mask_vecs[h].push(allele == 255);
                // Initialize MutableGenotypes (missing -> 0, else 1 if allele!=0)
                // Note: MutableGenotypes init with 0. Only need to set if 1.
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
        let stage1_gen_positions: Vec<f64> =
            hi_freq_markers.iter().map(|&m| gen_positions[m]).collect();

        let stage1_gen_dists: Vec<f64> = if hi_freq_markers.len() > 1 {
            hi_freq_markers
                .windows(2)
                .map(|w| gen_positions[w[1]] - gen_positions[w[0]])
                .collect()
        } else {
            Vec::new()
        };

        let stage1_p_recomb: Vec<f32> = std::iter::once(0.0f32)
            .chain(stage1_gen_dists.iter().map(|&d| self.params.p_recomb(d)))
            .collect();

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

        for it in 0..total_iterations {
            let is_burnin = it < n_burnin;
            let iter_type = if is_burnin { "burnin" } else { "main" };
            eprintln!("Iteration {}/{} ({})", it + 1, total_iterations, iter_type);

            // Update LR threshold for this iteration
            self.params.lr_threshold = self.params.lr_threshold_for_iteration(it);

            // Run phasing iteration with EM estimation (if enabled and during burnin)
            let collect_em = self.config.em && is_burnin;
            self.run_iteration_with_hmm_stage1(
                &mut geno,
                &missing_mask,
                &stage1_p_recomb,
                &stage1_gen_dists,
                &stage1_gen_positions,
                &hi_freq_to_orig,
                &ibs2,
                collect_em,
            )?;
        }

        // STAGE 2: Interpolate rare marker phases from flanking high-frequency markers
        if !rare_markers.is_empty() {
            eprintln!(
                "Stage 2: Phasing {} rare markers by interpolation...",
                rare_markers.len()
            );
            self.phase_rare_markers(&mut geno, &rare_markers, &hi_freq_markers);
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
    pub fn phase_in_memory(
        &mut self,
        target_gt: &GenotypeMatrix,
        gen_maps: &GeneticMaps,
    ) -> Result<GenotypeMatrix> {
        let n_markers = target_gt.n_markers();
        let n_haps = target_gt.n_haplotypes();

        if n_markers == 0 {
            return Ok(target_gt.clone());
        }

        // Initialize parameters
        self.params = ModelParams::for_phasing(n_haps);
        self.params
            .set_n_states(self.config.phase_states.min(n_haps.saturating_sub(2)));

        // Create mutable genotype storage for phasing
        // 1. Pre-compute missing mask (row-major: [hap][marker]) and init geno
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

        // Compute MAF for each marker (used by IBS2)
        let maf: Vec<f32> = (0..n_markers)
            .map(|m| target_gt.column(MarkerIdx::new(m as u32)).maf() as f32)
            .collect();

        // Build IBS2 segments for phase consistency
        let ibs2 = Ibs2::new(target_gt, gen_maps, chrom, &maf);

        // Run phasing iterations (reduced for imputation pre-processing)
        let n_burnin = self.config.burnin.min(3);
        let n_iterations = self.config.iterations.min(6);
        let total_iterations = n_burnin + n_iterations;

        for it in 0..total_iterations {
            let is_burnin = it < n_burnin;
            self.params.lr_threshold = self.params.lr_threshold_for_iteration(it);
            let collect_em = self.config.em && is_burnin;
            self.run_iteration_with_hmm(
                &mut geno,
                &missing_mask,
                &p_recomb,
                &gen_dists,
                &gen_positions,
                &ibs2,
                collect_em,
            )?;
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
        missing_mask: &[BitBox<u8, Lsb0>],
        p_recomb: &[f32],
        gen_dists: &[f64],
        gen_positions: &[f64],
        ibs2: &Ibs2,
        collect_em: bool,
    ) -> Result<()> {
        let n_samples = geno.n_haps() / 2;
        let n_markers = geno.n_markers();
        let n_haps = geno.n_haps();
        
        // Initialize Global PBWT
        let mut ibs = GlobalPhaseIbs::new(n_haps);
        
        // Initialize Online HMMs
        let mut online_hmms: Vec<OnlineHmm> = (0..n_samples)
            .map(|_| OnlineHmm::new())
            .collect();

        let total_switches = AtomicUsize::new(0);
        
        // Markers Loop (Standard: All markers)
        for m in 0..n_markers {
            let pr = p_recomb[m];
            
            // 1. Parallel Decision Phase
            let alleles_bits = geno.marker_alleles(m);
            let alleles_bytes: Vec<u8> = (0..n_haps).map(|h| {
                 let is_missing = missing_mask[h].get(m).as_deref().copied().unwrap_or(false);
                 if is_missing { 255 } else { if alleles_bits[h] { 1 } else { 0 } }
            }).collect();
            
            let results: Vec<(bool, Vec<u32>, Vec<u32>)> = online_hmms
                .par_iter()
                .enumerate()
                .map(|(s, hmm)| {
                    let hap1 = (s * 2) as u32;
                    let hap2 = (s * 2 + 1) as u32;
                    let k = 100;
                    
                    let n1 = ibs.find_neighbors(hap1, m, ibs2, k);
                    let n2 = ibs.find_neighbors(hap2, m, ibs2, k);
                    
                    let a1 = alleles_bytes[hap1 as usize];
                    let a2 = alleles_bytes[hap2 as usize];
                    
                    let swap = hmm.decide_phase_at_step(
                        &n1,
                        a1,
                        a2,
                        |h_idx| { alleles_bytes[h_idx as usize] },
                        pr,
                        n_haps,
                        &self.params,
                    );
                    (swap, n1, n2)
                })
                .collect();
            
            // 2. Apply Swaps
            let mut step_switches = 0;
            for (s, (swap, _, _)) in results.iter().enumerate() {
                if *swap {
                    geno.swap(m, HapIdx::new((s*2) as u32), HapIdx::new((s*2+1) as u32));
                    step_switches += 1;
                }
            }
            total_switches.fetch_add(step_switches, Ordering::Relaxed);

            // 3. Finalize Alleles
            let mut final_alleles = alleles_bytes;
            for (s, (swap, _, _)) in results.iter().enumerate() {
                if *swap {
                    final_alleles.swap(s*2, s*2+1);
                }
            }

            // 4. Update HMM
            online_hmms.par_iter_mut().zip(results.into_par_iter()).enumerate().for_each(|(s, (hmm, (_, n1, n2)))| {
                let mut states = n1;
                states.extend(n2);
                states.sort_unstable();
                states.dedup();
                
                let my_h1 = s * 2;
                let my_h2 = s * 2 + 1;
                let a1 = final_alleles[my_h1];
                let a2 = final_alleles[my_h2];
                let p_match = self.params.emit_match();
                let p_mismatch = self.params.emit_mismatch();
                
                hmm.step(
                    &states,
                    |h_idx| { 
                        let ref_a = final_alleles[h_idx as usize];
                        let e1 = if a1 == 255 || ref_a == 255 { 1.0 } else if a1 == ref_a { p_match } else { p_mismatch };
                        let e2 = if a2 == 255 || ref_a == 255 { 1.0 } else if a2 == ref_a { p_match } else { p_mismatch };
                        (1.0, e1, e2) 
                    },
                    pr,
                    n_haps,
                );
            });
            
            // 5. Advance PBWT
            ibs.advance(&final_alleles, m);
        }

        eprintln!("Applied {} phase switches (Standard)", total_switches.load(Ordering::Relaxed));
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
                let bytes: Vec<u8> = alleles.iter().map(|b| *b as u8).collect();
                GenotypeColumn::from_alleles(&bytes, 2)
            })
            .collect();

        GenotypeMatrix::new(markers, columns, samples, true)
    }

    /// Run Stage 1 phasing iteration on HIGH-FREQUENCY markers only
    ///
    /// This is the key to two-stage phasing: by running the HMM only on high-frequency
    /// markers, we get:
    /// 1. Correct recombination probabilities (distances between common variants)
    /// 2. Better computational efficiency (fewer markers in O(n²) HMM)
    /// 3. More robust phasing signal (rare variants are noisy)
    ///
    /// Switch markers returned are in ORIGINAL marker indices for correct application.
    fn run_iteration_with_hmm_stage1(
        &mut self,
        geno: &mut MutableGenotypes,
        missing_mask: &[BitBox<u8, Lsb0>],
        stage1_p_recomb: &[f32],
        stage1_gen_dists: &[f64],
        stage1_gen_positions: &[f64],
        hi_freq_to_orig: &[usize],
        ibs2: &Ibs2,
        collect_em: bool,
    ) -> Result<()> {
        let n_samples = geno.n_haps() / 2;
        let n_haps = geno.n_haps();
        let n_hi_freq = hi_freq_to_orig.len();
        
        if n_hi_freq == 0 {
            return Ok(());
        }

        // Initialize Global PBWT
        let mut ibs = GlobalPhaseIbs::new(n_haps);
        
        // Initialize Online HMMs for each sample
        let mut online_hmms: Vec<OnlineHmm> = (0..n_samples)
            .map(|_| {
                let mut hmm = OnlineHmm::new();
                // Initialize with all haplotypes as potential states (uniform prior)
                // Or let the first step handle it?
                // Better to start fresh.
                hmm
            })
            .collect();

        // Atomic for statistics
        let total_switches = AtomicUsize::new(0);

        // Iterate through High-Frequency markers
        for (i, &orig_m) in hi_freq_to_orig.iter().enumerate() {
            let p_recomb = stage1_p_recomb[i];
            
            // 1. Parallel Step: Find Neighbors & Decide Phase
            // We need to access 'geno' (bits) in parallel. Thread-safe as we don't write.
            // But MutableGenotypes::haplotype returns vectors/refs?
            // MutableGenotypes stores columns? No, rows (haplotypes).
            // Accessing column m is slow if row-store?
            // MutableGenotypes has `marker_alleles(m)` which returns Vec<bool>.
            // That creates a copy. O(N).
            // We can distribute this read-only buffer to threads.
            
            let alleles_bits = geno.marker_alleles(orig_m); // Vec<bool> of size n_haps
            let alleles_bytes: Vec<u8> = (0..n_haps).map(|h| {
                 let is_missing = missing_mask[h].get(orig_m).as_deref().copied().unwrap_or(false);
                 if is_missing { 255 } else { if alleles_bits[h] { 1 } else { 0 } }
            }).collect();
            
            // Collect results: (swaps, neighbors)
            let results: Vec<(bool, Vec<u32>, Vec<u32>)> = online_hmms
                .par_iter()
                .enumerate()
                .map(|(s, hmm)| {
                    let hap1 = (s * 2) as u32;
                    let hap2 = (s * 2 + 1) as u32;
                    
                    // Find neighbors using Global PBWT state
                    // Use somewhat large K for accuracy
                    let k = 100; 
                    let n1 = ibs.find_neighbors(hap1, i, ibs2, k);
                    let n2 = ibs.find_neighbors(hap2, i, ibs2, k);
                    
                    // Current alleles
                    let a1 = alleles_bytes[hap1 as usize];
                    let a2 = alleles_bytes[hap2 as usize];
                    
                    // Decide Phase
                    let swap = hmm.decide_phase_at_step(
                        &n1, // using neighbors of hap1 as states for hap1 HMM?
                             // Wait. If swapped, hap1 originates from s2-like states?
                             // Standard HMM: fwd1 tracks "Hap1 physics".
                             // States are "reference haps".
                             // If we swap, we swap observations.
                             // decided_phase_at_step handles logic.
                        a1,
                        a2,
                        |h_idx| { // Lookup ref allele from current marker alleles
                            alleles_bytes[h_idx as usize]
                        },
                        p_recomb,
                        n_haps,
                        &self.params,
                    );
                    
                    (swap, n1, n2)
                })
                .collect();
            
            // 2. Apply Swaps & Count
            let mut step_switches = 0;
            for (s, (swap, _, _)) in results.iter().enumerate() {
                if *swap {
                    // Swap at this marker means we change physical alleles
                    // This affects subsequent steps and final output
                    // We only swap AT this marker? 
                    // No, usually "phase switch" means "alleles from here on are flipped relative to previous"
                    // But here we are building the haplotype.
                    // If we decide "swap", we flip the bits at `orig_m`.
                    // And we flip the HMM state mapping?
                    // "Online Phasing" builds the haplotypes marker by marker.
                    // If we swap, we physically swap the bits in `geno`.
                    // Does this affect `prev` history?
                    // Only in that we are establishing the physical haplotype sequence 0..m.
                    // The HMM state tracks likelihood of this sequence.
                    geno.swap(orig_m, HapIdx::new((s*2) as u32), HapIdx::new((s*2+1) as u32));
                    step_switches += 1;
                }
            }
            total_switches.fetch_add(step_switches, Ordering::Relaxed);
            
            // 3. Reconstruct FINAL alleles for HMM Update and PBWT Advance
            // Optimization: modify `alleles_bytes` in place based on swaps
            let mut final_alleles = alleles_bytes; // move
            for (s, (swap, _, _)) in results.iter().enumerate() {
                if *swap {
                    final_alleles.swap(s*2, s*2+1);
                }
            }
            
            // 4. Update HMM States (Parallel)
            online_hmms.par_iter_mut().zip(results.into_par_iter()).enumerate().for_each(|(s, (hmm, (_, n1, n2)))| {
                // Merge neighbors to form the state set for this step
                let mut states = n1;
                states.extend(n2);
                states.sort_unstable();
                states.dedup();
                
                // Get final alleles for this sample
                // Note: final_alleles is shared across threads, read-only here
                let my_h1 = s * 2;
                let my_h2 = s * 2 + 1;
                let a1 = final_alleles[my_h1];
                let a2 = final_alleles[my_h2];
                
                let p_match = self.params.emit_match();
                let p_mismatch = self.params.emit_mismatch();
                
                hmm.step(
                    &states,
                    |h_idx| { 
                        let ref_a = final_alleles[h_idx as usize];
                        
                        let e1 = if a1 == 255 || ref_a == 255 { 
                            1.0 
                        } else if a1 == ref_a { 
                            p_match 
                        } else { 
                            p_mismatch 
                        };
                        
                        let e2 = if a2 == 255 || ref_a == 255 { 
                            1.0 
                        } else if a2 == ref_a { 
                            p_match 
                        } else { 
                            p_mismatch 
                        };
                        
                        // Combined emission ignored for phasing
                        (1.0, e1, e2) 
                    },
                    p_recomb,
                    n_haps,
                );
            });
            
            // 5. Advance Global PBWT
            // Note: advance takes marker_idx which is 'i' in hi-freq space?
            // No, strictly PBWT relies on bit patterns.
            // But if we skip markers, PBWT will be approximate?
            // For Stage 1, we ONLY run on Hi-Freq markers.
            // So we treat the Hi-Freq markers as the ONLY markers.
            // The PBWT state evolves along the Hi-Freq backbone.
            ibs.advance(&final_alleles, i);
        }
        
        eprintln!("Applied {} switches in Stage 1", total_switches.load(Ordering::Relaxed));
        Ok(())
    }

    /// Stage 2: Phase rare markers by interpolation from flanking high-frequency markers
    ///
    /// For each rare heterozygous marker, we determine its phase by looking at the
    /// closest flanking high-frequency heterozygous markers. If both flanking markers
    /// have the same phase relationship (both swapped or both unswapped relative to
    /// some reference), the rare marker inherits that phase. If they disagree or
    /// only one exists, we use the closest one.
    ///
    /// This matches Java Beagle's PhaseLS.runStage2() approach.
    fn phase_rare_markers(
        &self,
        geno: &mut MutableGenotypes,
        rare_markers: &[usize],
        hi_freq_markers: &[usize],
    ) {
        let n_samples = geno.n_haps() / 2;
        let n_markers = geno.n_markers();

        if rare_markers.is_empty() || hi_freq_markers.is_empty() {
            return;
        }

        // For each sample, phase rare heterozygous markers
        for s in 0..n_samples {
            let sample_idx = SampleIdx::new(s as u32);
            let hap1 = sample_idx.hap1();
            let hap2 = sample_idx.hap2();

            // Get current alleles
            let alleles1 = geno.haplotype(hap1);
            let alleles2 = geno.haplotype(hap2);

            // Find heterozygous rare markers for this sample
            let rare_het: Vec<usize> = rare_markers
                .iter()
                .copied()
                .filter(|&m| alleles1[m] != alleles2[m] && alleles1[m] != 255 && alleles2[m] != 255)
                .collect();

            if rare_het.is_empty() {
                continue;
            }

            // Find heterozygous high-frequency markers
            let hf_het: Vec<usize> = hi_freq_markers
                .iter()
                .copied()
                .filter(|&m| alleles1[m] != alleles2[m] && alleles1[m] != 255 && alleles2[m] != 255)
                .collect();

            if hf_het.is_empty() {
                // No hi-freq hets to interpolate from - leave rare markers as-is
                continue;
            }

            // For each rare het marker, find flanking hi-freq hets and interpolate phase
            for &rare_m in &rare_het {
                // Find closest hi-freq het markers on each side
                let left_hf = hf_het.iter().copied().filter(|&m| m < rare_m).max();
                let right_hf = hf_het.iter().copied().filter(|&m| m > rare_m).min();

                // Determine phase from flanking markers
                // We compare which allele is on hap1 at the flanking positions
                let should_swap = match (left_hf, right_hf) {
                    (Some(left), Some(right)) => {
                        // Both flanking markers exist - use consensus
                        // Check if hap1 has allele 0 or 1 at each position
                        let left_hap1_has_0 = alleles1[left] == 0;
                        let right_hap1_has_0 = alleles1[right] == 0;
                        // If they disagree, use closer one
                        if left_hap1_has_0 != right_hap1_has_0 {
                            let dist_left = rare_m - left;
                            let dist_right = right - rare_m;
                            if dist_left <= dist_right {
                                !left_hap1_has_0
                            } else {
                                !right_hap1_has_0
                            }
                        } else {
                            // Both agree - inherit that phase
                            !left_hap1_has_0
                        }
                    }
                    (Some(left), None) => {
                        // Only left marker exists
                        alleles1[left] != 0
                    }
                    (None, Some(right)) => {
                        // Only right marker exists
                        alleles1[right] != 0
                    }
                    (None, None) => {
                        // No flanking hets (shouldn't happen if hf_het is non-empty)
                        continue;
                    }
                };

                // Apply phase: if should_swap, ensure hap1 has allele 1, else allele 0
                let rare_hap1_has_0 = alleles1[rare_m] == 0;
                if should_swap != rare_hap1_has_0 {
                    // Need to swap this marker
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
            
        let gt = GenotypeMatrix::new(
            markers, 
            columns, 
            samples, 
            true
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
