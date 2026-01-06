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
use crate::data::storage::phase_state::Phased;
use crate::data::storage::{GenotypeColumn, GenotypeMatrix, MutableGenotypes, GenotypeView};
use crate::error::Result;
use crate::io::bref3::Bref3Reader;
use crate::io::streaming::{PhasedOverlap, StreamingConfig, StreamingVcfReader};
use crate::io::vcf::{VcfReader, VcfWriter};
use crate::model::ibs2::Ibs2;
use crate::model::hmm::BeagleHmm;
use crate::model::states::ThreadedHaps;
use crate::model::parameters::ModelParams;
use crate::model::phase_ibs::BidirectionalPhaseIbs;
use crate::pipelines::imputation::MarkerAlignment;

/// Phasing pipeline
pub struct PhasingPipeline {
    config: Config,
    params: ModelParams,
    /// Reference panel for reference-guided phasing (optional)
    reference_gt: Option<GenotypeMatrix<Phased>>,
    /// Marker alignment between target and reference
    alignment: Option<MarkerAlignment>,
}

impl PhasingPipeline {
    /// Create a new phasing pipeline
    pub fn new(config: Config) -> Self {
        let params = ModelParams::new();
        Self {
            config,
            params,
            reference_gt: None,
            alignment: None,
        }
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

        // Load reference panel if provided (for reference-guided phasing)
        if let Some(ref_path) = &self.config.r#ref {
            eprintln!("Loading reference panel for phasing...");
            let ref_gt: GenotypeMatrix<Phased> = if ref_path.extension().map(|e| e == "bref3").unwrap_or(false) {
                eprintln!("  Detected BREF3 format");
                let reader = Bref3Reader::open(ref_path)?;
                reader.read_all()?
            } else {
                eprintln!("  Detected VCF format");
                let (mut ref_reader, ref_file) = VcfReader::open(ref_path)?;
                ref_reader.read_all(ref_file)?.into_phased()
            };
            eprintln!(
                "  Reference: {} markers, {} haplotypes",
                ref_gt.n_markers(),
                ref_gt.n_haplotypes()
            );

            // Create marker alignment between target and reference
            let alignment = MarkerAlignment::new(&target_gt, &ref_gt);
            eprintln!("  Aligned {} reference markers to target", alignment.n_aligned());

            // Store in pipeline struct for use during phasing iterations
            self.alignment = Some(alignment);
            self.reference_gt = Some(ref_gt);
        }

        // Compute combined haplotype count
        let n_ref_haps = self.reference_gt.as_ref().map(|r| r.n_haplotypes()).unwrap_or(0);
        let n_total_haps = n_haps + n_ref_haps;

        if n_ref_haps > 0 {
            eprintln!(
                "Combined haplotype space: {} target + {} reference = {} total",
                n_haps, n_ref_haps, n_total_haps
            );
        }

        // Initialize parameters based on TOTAL haplotype count (target + ref)
        self.params = ModelParams::for_phasing(n_total_haps, self.config.ne, self.config.err);
        self.params
            .set_n_states(self.config.phase_states.min(n_total_haps.saturating_sub(2)));

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
        // MutableGenotypes now internally tracks missing data (allele = 255)
        // so we can use from_fn to initialize all values including missing
        let mut geno = MutableGenotypes::from_fn(n_markers, n_haps, |m, h| {
            target_gt.allele(MarkerIdx::new(m as u32), HapIdx::new(h as u32))
        });

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
        let mut sample_phases = self.create_sample_phases(&geno);

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
                &mut geno,
                &stage1_p_recomb,
                &stage1_gen_dists,
                &hi_freq_to_orig,
                &ibs2,
                &mut sample_phases,
                atomic_estimates.as_ref(),
                it,
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
        
        // Track phased overlap from previous window for phase continuity
        let mut phased_overlap: Option<PhasedOverlap> = None;

        // Process windows
        while let Some(mut window) = reader.next_window()? {
            window_count += 1;
            let n_markers = window.genotypes.n_markers();
            total_markers += window.output_end - window.output_start;

            eprintln!(
                "Processing window {} ({} markers, global {}..{}, output {}..{}, overlap: {} markers)",
                window.window_num,
                n_markers,
                window.global_start,
                window.global_end,
                window.output_start,
                window.output_end,
                phased_overlap.as_ref().map(|o| o.n_markers).unwrap_or(0)
            );

            // Set the phased overlap from previous window
            window.phased_overlap = phased_overlap.take();

            // Phase this window with overlap constraint
            let phased = self.phase_in_memory_with_overlap(&window.genotypes, &gen_maps, window.phased_overlap.as_ref())?;

            // Extract overlap for next window (markers from output_end to end of window)
            if !window.is_last() {
                phased_overlap = Some(self.extract_overlap(&phased, window.output_end, n_markers));
            }

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
    
    /// Extract phased overlap region from a phased genotype matrix
    ///
    /// This extracts the overlap region (markers from `start` to `end`) to be used
    /// as a constraint for the next window's phasing, ensuring phase continuity.
    fn extract_overlap(
        &self,
        phased: &GenotypeMatrix<crate::data::storage::phase_state::Phased>,
        start: usize,
        end: usize,
    ) -> PhasedOverlap {
        let n_overlap = end - start;
        let n_haps = phased.n_haplotypes();
        
        let mut alleles = Vec::with_capacity(n_overlap * n_haps);
        
        // Layout: alleles[hap * n_markers + marker]
        for h in 0..n_haps {
            let h_idx = HapIdx::new(h as u32);
            for m in start..end {
                let m_idx = MarkerIdx::new(m as u32);
                alleles.push(phased.allele(m_idx, h_idx));
            }
        }
        
        PhasedOverlap::new(n_overlap, n_haps, alleles)
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

        self.params = ModelParams::for_phasing(n_haps, self.config.ne, self.config.err);
        self.params
            .set_n_states(self.config.phase_states.min(n_haps.saturating_sub(2)));

        // Initialize genotypes preserving actual allele values including missing (255)
        let mut geno = MutableGenotypes::from_fn(n_markers, n_haps, |m, h| {
            target_gt.allele(MarkerIdx::new(m as u32), HapIdx::new(h as u32))
        });

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
                &p_recomb,
                &gen_dists,
                &ibs2,
                atomic_estimates.as_ref(),
                it,
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
    
    /// Phase a GenotypeMatrix in-memory with overlap constraint from previous window
    ///
    /// This is like `phase_in_memory` but seeds the phasing with alleles from the
    /// overlap region of the previous window, ensuring phase continuity at window
    /// boundaries. Based on Java's FixedPhaseData and SplicedGT.
    pub fn phase_in_memory_with_overlap(
        &mut self,
        target_gt: &GenotypeMatrix,
        gen_maps: &GeneticMaps,
        phased_overlap: Option<&PhasedOverlap>,
    ) -> Result<GenotypeMatrix<crate::data::storage::phase_state::Phased>> {
        let n_markers = target_gt.n_markers();
        let n_haps = target_gt.n_haplotypes();

        if n_markers == 0 {
            return Ok(target_gt.clone().into_phased());
        }

        self.params = ModelParams::for_phasing(n_haps, self.config.ne, self.config.err);
        self.params
            .set_n_states(self.config.phase_states.min(n_haps.saturating_sub(2)));

        // Initialize genotypes preserving actual allele values including missing (255)
        let mut geno = MutableGenotypes::from_fn(n_markers, n_haps, |m, h| {
            target_gt.allele(MarkerIdx::new(m as u32), HapIdx::new(h as u32))
        });

        // Build missing mask for overlap constraint handling
        let missing_mask: Vec<BitBox<u8, Lsb0>> = (0..n_haps)
            .map(|h| {
                let bits: BitVec<u8, Lsb0> = (0..n_markers)
                    .map(|m| target_gt.allele(MarkerIdx::new(m as u32), HapIdx::new(h as u32)) == 255)
                    .collect();
                bits.into_boxed_bitslice()
            })
            .collect();

        // Apply overlap constraint: set alleles from previous window's phased overlap
        // This seeds the phasing with the known phase from the overlap region
        let overlap_markers = if let Some(overlap) = phased_overlap {
            self.apply_overlap_constraint(&mut geno, overlap);
            overlap.n_markers
        } else {
            0
        };

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

        // Create sample phases with overlap markers pre-phased
        let mut sample_phases = self.create_sample_phases_with_overlap(&geno, &missing_mask, overlap_markers);

        for it in 0..total_iterations {
            let is_burnin = it < n_burnin;
            self.params.lr_threshold = self.params.lr_threshold_for_iteration(it);
            
            let atomic_estimates = if is_burnin && self.config.em {
                Some(crate::model::parameters::AtomicParamEstimates::new())
            } else {
                None
            };

            self.run_phase_baum_iteration_with_overlap(
                &target_gt,
                &mut geno,
                &p_recomb,
                &gen_dists,
                &ibs2,
                &mut sample_phases,
                overlap_markers,
                atomic_estimates.as_ref(),
                it,
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
        
        // Sync final phase state from SamplePhase to MutableGenotypes
        self.sync_sample_phases_to_geno(&sample_phases, &mut geno);

        Ok(self.build_final_matrix(target_gt, &geno))
    }
    
    /// Apply overlap constraint from previous window's phased output
    ///
    /// This sets the alleles in the overlap region to match the previous window's
    /// phased output, ensuring phase continuity.
    fn apply_overlap_constraint(&self, geno: &mut MutableGenotypes, overlap: &PhasedOverlap) {
        let n_overlap = overlap.n_markers;
        let n_haps = overlap.n_haps.min(geno.n_haps());
        
        for h in 0..n_haps {
            let h_idx = HapIdx::new(h as u32);
            for m in 0..n_overlap {
                let allele = overlap.allele(m, h);
                if allele != 255 {
                    geno.set(m, h_idx, allele);
                }
            }
        }
    }
    
    /// Create SamplePhase instances with overlap markers pre-phased
    ///
    /// Markers in the overlap region (0..overlap_markers) are marked as already
    /// phased since their phase comes from the previous window.
    fn create_sample_phases_with_overlap(
        &self,
        geno: &MutableGenotypes,
        missing_mask: &[BitBox<u8, Lsb0>],
        overlap_markers: usize,
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

                // Hets in the overlap region are already phased (from previous window)
                // Only hets AFTER the overlap region start as unphased
                let unphased: Vec<usize> = (overlap_markers..n_markers)
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

    /// Create SamplePhase instances for all samples
    ///
    /// This initializes phase tracking state from the current genotype data.
    fn create_sample_phases(
        &self,
        geno: &MutableGenotypes,
    ) -> Vec<SamplePhase> {
        let n_samples = geno.n_haps() / 2;
        let n_markers = geno.n_markers();

        (0..n_samples)
            .map(|s| {
                let hap1 = HapIdx::new((s * 2) as u32);
                let hap2 = HapIdx::new((s * 2 + 1) as u32);

                // geno.get() now returns 255 for missing positions
                let alleles1: Vec<u8> = (0..n_markers).map(|m| geno.get(m, hap1)).collect();
                let alleles2: Vec<u8> = (0..n_markers).map(|m| geno.get(m, hap2)).collect();

                // Identify missing markers using the internal missing tracking
                let missing: Vec<usize> = (0..n_markers)
                    .filter(|&m| geno.is_missing(m, hap1) || geno.is_missing(m, hap2))
                    .collect();

                // Initially all hets are unphased (het = different alleles, neither missing)
                let unphased: Vec<usize> = (0..n_markers)
                    .filter(|&m| {
                        let a1 = alleles1[m];
                        let a2 = alleles2[m];
                        a1 != a2 && a1 != 255 && a2 != 255
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

    /// Select HMM states using bidirectional PBWT (forward + backward neighbors)
    fn select_states_bidirectional(
        &self,
        hap_idx: HapIdx,
        marker_idx: usize,
        phase_ibs: &BidirectionalPhaseIbs,
        ibs2: &Ibs2,
        n_states_wanted: usize,
        n_total_haps: usize,
        seed: i64,
        iteration: usize,
    ) -> Vec<HapIdx> {
        use rand::seq::SliceRandom;
        use rand::SeedableRng;

        let n_candidates = (n_states_wanted * 2).min(n_total_haps);
        let neighbors = phase_ibs.find_neighbors(hap_idx.0, marker_idx, ibs2, n_candidates);

        let mut states: Vec<HapIdx> = neighbors.into_iter().map(HapIdx::new).collect();

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

        let combined_seed = (seed as u64)
            .wrapping_add(iteration as u64)
            .wrapping_add(hap_idx.0 as u64);
        let mut rng = rand::rngs::StdRng::seed_from_u64(combined_seed);

        states.shuffle(&mut rng);
        states.truncate(n_states_wanted);
        states
    }

    /// Build bidirectional PBWT for full chromosome phasing
    ///
    /// This stores both forward and backward PBWT arrays to enable selecting
    /// haplotypes that match well both upstream and downstream of each marker.
    fn build_bidirectional_pbwt(&self, geno: &MutableGenotypes, n_markers: usize, n_haps: usize) -> BidirectionalPhaseIbs {
        let mut alleles_by_marker: Vec<Vec<u8>> = Vec::with_capacity(n_markers);
        for m in 0..n_markers {
            let mut alleles = Vec::with_capacity(n_haps);
            for h in 0..n_haps {
                alleles.push(geno.get(m, HapIdx::new(h as u32)));
            }
            alleles_by_marker.push(alleles);
        }
        BidirectionalPhaseIbs::build(&alleles_by_marker, n_haps, n_markers)
    }

    /// Build bidirectional PBWT for a subset of markers (e.g., high-frequency only)
    ///
    /// This ensures state selection is consistent with the HMM marker space
    /// when phasing Stage 1 (high-frequency) markers.
    fn build_bidirectional_pbwt_subset(
        &self,
        geno: &MutableGenotypes,
        marker_indices: &[usize],
        n_haps: usize,
    ) -> BidirectionalPhaseIbs {
        let n_subset = marker_indices.len();
        let mut alleles_by_marker: Vec<Vec<u8>> = Vec::with_capacity(n_subset);
        
        for &orig_m in marker_indices {
            let mut alleles = Vec::with_capacity(n_haps);
            for h in 0..n_haps {
                alleles.push(geno.get(orig_m, HapIdx::new(h as u32)));
            }
            alleles_by_marker.push(alleles);
        }
        
        BidirectionalPhaseIbs::build(&alleles_by_marker, n_haps, n_subset)
    }

    /// Build bidirectional PBWT for combined target + reference haplotype space
    ///
    /// This builds the PBWT over all haplotypes (target and reference) to enable
    /// finding IBS matches in both pools during reference-guided phasing.
    fn build_bidirectional_pbwt_combined<F>(
        &self,
        get_allele: F,
        n_markers: usize,
        n_total_haps: usize,
    ) -> BidirectionalPhaseIbs
    where
        F: Fn(usize, usize) -> u8,  // (marker_idx, hap_idx) -> allele
    {
        let mut alleles_by_marker: Vec<Vec<u8>> = Vec::with_capacity(n_markers);
        for m in 0..n_markers {
            let mut alleles = Vec::with_capacity(n_total_haps);
            for h in 0..n_total_haps {
                alleles.push(get_allele(m, h));
            }
            alleles_by_marker.push(alleles);
        }
        BidirectionalPhaseIbs::build(&alleles_by_marker, n_total_haps, n_markers)
    }

    /// Build bidirectional PBWT for combined haplotype space on a marker subset
    ///
    /// Used for Stage 1 phasing with reference panel - indexes both target and
    /// reference haplotypes but only on high-frequency markers.
    fn build_bidirectional_pbwt_combined_subset<F>(
        &self,
        get_allele: F,
        marker_indices: &[usize],
        n_total_haps: usize,
    ) -> BidirectionalPhaseIbs
    where
        F: Fn(usize, usize) -> u8,  // (orig_marker_idx, hap_idx) -> allele
    {
        let n_subset = marker_indices.len();
        let mut alleles_by_marker: Vec<Vec<u8>> = Vec::with_capacity(n_subset);

        for &orig_m in marker_indices {
            let mut alleles = Vec::with_capacity(n_total_haps);
            for h in 0..n_total_haps {
                alleles.push(get_allele(orig_m, h));
            }
            alleles_by_marker.push(alleles);
        }

        BidirectionalPhaseIbs::build(&alleles_by_marker, n_total_haps, n_subset)
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
        p_recomb: &[f32],
        gen_dists: &[f64],
        ibs2: &Ibs2,
        atomic_estimates: Option<&crate::model::parameters::AtomicParamEstimates>,
        iteration: usize,
    ) -> Result<()> {
        let n_samples = geno.n_haps() / 2;
        let n_markers = geno.n_markers();
        let n_haps = geno.n_haps();
        let markers = target_gt.markers();
        let seed = self.config.seed;

        // Compute total haplotype count (target + reference)
        let n_ref_haps = self.reference_gt.as_ref().map(|r| r.n_haplotypes()).unwrap_or(0);
        let n_total_haps = n_haps + n_ref_haps;

        let ref_geno = geno.clone();

        // Use Composite view when reference panel is available
        let ref_view = if let (Some(ref_gt), Some(alignment)) = (&self.reference_gt, &self.alignment) {
            GenotypeView::Composite {
                target: &ref_geno,
                reference: ref_gt,
                alignment,
                n_target_haps: n_haps,
            }
        } else {
            GenotypeView::from((&ref_geno, markers))
        };

        // Build PBWT over combined haplotype space when reference is available
        let phase_ibs = if let (Some(ref_gt), Some(alignment)) = (&self.reference_gt, &self.alignment) {
            self.build_bidirectional_pbwt_combined(
                |m, h| {
                    if h < n_haps {
                        ref_geno.get(m, HapIdx::new(h as u32))
                    } else {
                        let ref_h = h - n_haps;
                        if let Some(ref_m) = alignment.target_to_ref(m) {
                            let ref_allele = ref_gt.allele(MarkerIdx::new(ref_m as u32), HapIdx::new(ref_h as u32));
                            // Map reference allele to target encoding (handles strand flips)
                            alignment.reverse_map_allele(m, ref_allele)
                        } else {
                            255 // Missing - marker not in reference
                        }
                    }
                },
                n_markers,
                n_total_haps,
            )
        } else {
            self.build_bidirectional_pbwt(&ref_geno, n_markers, n_haps)
        };

        let mut swap_masks: Vec<BitVec<u8, Lsb0>> = vec![BitVec::repeat(false, n_markers); n_samples];

        swap_masks.par_iter_mut().enumerate().for_each(|(s, mask)| {
            let sample_idx = SampleIdx::new(s as u32);
            let hap1 = sample_idx.hap1();
            let hap2 = sample_idx.hap2();

            let mid_marker = n_markers / 2;
            // Use n_total_haps to allow selecting reference haplotypes as states
            let states = self.select_states_bidirectional(hap1, mid_marker, &phase_ibs, ibs2, self.params.n_states, n_total_haps, seed, iteration);
            
            // 2. Extract current alleles for H1 and H2
            let seq1 = ref_geno.haplotype(hap1);
            let seq2 = ref_geno.haplotype(hap2);

            // 3. Run HMM with PER-HETEROZYGOTE phase decisions
            // Following Java PhaseBaum2.java: interleave phase decisions in the forward pass.
            //
            // Key Algorithm (3-Track HMM):
            // 1. Run backward pass for BOTH haplotypes first, storing backward values
            // 2. Run forward pass marker-by-marker for BOTH haplotypes
            // 3. At each het, compute phase decision using fwd and stored bwd
            // 4. If swap decided, swap the allele sequences from that point forward
            // 5. Continue forward pass with (potentially swapped) sequences
            //
            // This ensures forward probabilities after a het correctly reflect the phase decision
            // by using the decided alleles for emission probabilities.
            let n_states = states.len();
            let threaded_haps = ThreadedHaps::from_static_haps(&states, n_markers);
            let hmm = BeagleHmm::new(ref_view.clone(), &self.params, n_states, p_recomb.to_vec());

            // Collect EM statistics if requested (using original sequences)
            if let Some(atomic) = atomic_estimates {
                let mut local_est = crate::model::parameters::ParamEstimates::new();
                hmm.collect_stats(&seq1, &threaded_haps, gen_dists, &mut local_est);
                hmm.collect_stats(&seq2, &threaded_haps, gen_dists, &mut local_est);
                atomic.add_estimation_data(&local_est);
            }

            // 3-Track HMM with Prior-First Approach
            //
            // This implementation avoids the numerically unstable division hack.
            // Instead, we:
            // 1. Run sparse backward passes, storing only at het positions
            // 2. Run forward with prior-first: compute transition before emission
            // 3. At hets: use prior (no emission) to evaluate both hypotheses
            // 4. Apply combined emission after decision for numerical stability

            // Identify heterozygote positions first
            let het_positions: Vec<usize> = (0..n_markers)
                .filter(|&m| {
                    let a1 = seq1[m];
                    let a2 = seq2[m];
                    a1 != 255 && a2 != 255 && a1 != a2
                })
                .collect();

            let p_err = self.params.p_mismatch;
            let p_no_err = 1.0 - p_err;

            // Helper to get reference allele at (marker, state)
            let get_ref_allele = |m: usize, k: usize| -> u8 {
                let h = states[k].as_usize();
                if h < n_haps {
                    ref_geno.get(m, states[k])
                } else {
                    let ref_h = (h - n_haps) as u32;
                    if let (Some(ref_gt_inner), Some(align)) = (&self.reference_gt, &self.alignment) {
                        if let Some(ref_m) = align.target_to_ref(m) {
                            let ref_allele = ref_gt_inner.allele(MarkerIdx::new(ref_m as u32), HapIdx::new(ref_h));
                            align.reverse_map_allele(m, ref_allele)
                        } else { 255 }
                    } else { 255 }
                }
            };

            // Sparse backward pass: only store values at het positions (O(N_hets × K) memory)
            // For each het, store two vectors: bwd assuming seq1's future, bwd assuming seq2's future
            let n_hets = het_positions.len();
            let mut bwd1_cache: Vec<Vec<f32>> = vec![vec![0.0; n_states]; n_hets];
            let mut bwd2_cache: Vec<Vec<f32>> = vec![vec![0.0; n_states]; n_hets];

            // Run backward for seq1
            {
                let init_bwd = 1.0f32 / n_states as f32;
                let mut bwd = vec![init_bwd; n_states];
                let mut het_rev_idx = n_hets;

                for m in (0..n_markers).rev() {
                    // Check if this is a het position (going backwards)
                    if het_rev_idx > 0 && het_positions[het_rev_idx - 1] == m {
                        het_rev_idx -= 1;
                        bwd1_cache[het_rev_idx].copy_from_slice(&bwd);
                    }

                    if m > 0 {
                        // Backward update: bwd[m-1] from bwd[m]
                        let p_recomb_m = p_recomb.get(m).copied().unwrap_or(0.0);
                        let shift = p_recomb_m / n_states as f32;
                        let stay = 1.0 - p_recomb_m;

                        // Compute sum for normalization
                        let mut bwd_sum = 0.0f32;
                        for k in 0..n_states {
                            let ref_al = get_ref_allele(m, k);
                            let emit = if ref_al == seq1[m] || ref_al == 255 || seq1[m] == 255 {
                                p_no_err
                            } else {
                                p_err
                            };
                            bwd[k] *= emit;
                            bwd_sum += bwd[k];
                        }

                        // Apply transition (backward direction)
                        let scale = stay / bwd_sum.max(1e-30);
                        for k in 0..n_states {
                            bwd[k] = scale * bwd[k] + shift;
                        }
                    }
                }
            }

            // Run backward for seq2
            {
                let init_bwd = 1.0f32 / n_states as f32;
                let mut bwd = vec![init_bwd; n_states];
                let mut het_rev_idx = n_hets;

                for m in (0..n_markers).rev() {
                    if het_rev_idx > 0 && het_positions[het_rev_idx - 1] == m {
                        het_rev_idx -= 1;
                        bwd2_cache[het_rev_idx].copy_from_slice(&bwd);
                    }

                    if m > 0 {
                        let p_recomb_m = p_recomb.get(m).copied().unwrap_or(0.0);
                        let shift = p_recomb_m / n_states as f32;
                        let stay = 1.0 - p_recomb_m;

                        let mut bwd_sum = 0.0f32;
                        for k in 0..n_states {
                            let ref_al = get_ref_allele(m, k);
                            let emit = if ref_al == seq2[m] || ref_al == 255 || seq2[m] == 255 {
                                p_no_err
                            } else {
                                p_err
                            };
                            bwd[k] *= emit;
                            bwd_sum += bwd[k];
                        }

                        let scale = stay / bwd_sum.max(1e-30);
                        for k in 0..n_states {
                            bwd[k] = scale * bwd[k] + shift;
                        }
                    }
                }
            }

            // 3-Track Forward Pass (Java PhaseBaum2.java algorithm)
            //
            // Maintain THREE forward tracks:
            // - fwd0: Combined track (match-any emission at hets)
            // - fwd1: Hap1 track (emits hap1's allele at hets)
            // - fwd2: Hap2 track (emits hap2's allele at hets)
            //
            // Between hets, fwd1 and fwd2 DIVERGE because they emit different alleles.
            // At each het decision point, we use fwd1 and fwd2 for likelihoods,
            // then RESET: fwd1 = fwd2 = fwd0
            let mut seq1_working = seq1.clone();
            let mut seq2_working = seq2.clone();

            let init_val = 1.0f32 / n_states as f32;
            let mut fwd0 = vec![init_val; n_states]; // Combined track
            let mut fwd1 = vec![init_val; n_states]; // Hap1 track
            let mut fwd2 = vec![init_val; n_states]; // Hap2 track
            let mut fwd0_prior = vec![0.0f32; n_states];
            let mut fwd1_prior = vec![0.0f32; n_states];
            let mut fwd2_prior = vec![0.0f32; n_states];
            let mut ref_alleles = vec![0u8; n_states];
            let mut fwd0_sum = 1.0f32;
            let mut fwd1_sum = 1.0f32;
            let mut fwd2_sum = 1.0f32;

            let mut het_idx = 0;
            for m in 0..n_markers {
                let allele1 = seq1_working[m];
                let allele2 = seq2_working[m];
                let is_het = het_idx < n_hets && het_positions[het_idx] == m;

                // Cache reference alleles for this marker
                for k in 0..n_states {
                    ref_alleles[k] = get_ref_allele(m, k);
                }

                // Compute PRIOR for all three tracks (transition only, no emission)
                if m > 0 {
                    let p_recomb_m = p_recomb.get(m).copied().unwrap_or(0.0);
                    let shift = p_recomb_m / n_states as f32;

                    let scale0 = (1.0 - p_recomb_m) / fwd0_sum;
                    let scale1 = (1.0 - p_recomb_m) / fwd1_sum;
                    let scale2 = (1.0 - p_recomb_m) / fwd2_sum;

                    for k in 0..n_states {
                        fwd0_prior[k] = scale0 * fwd0[k] + shift;
                        fwd1_prior[k] = scale1 * fwd1[k] + shift;
                        fwd2_prior[k] = scale2 * fwd2[k] + shift;
                    }
                } else {
                    for k in 0..n_states {
                        fwd0_prior[k] = init_val;
                        fwd1_prior[k] = init_val;
                        fwd2_prior[k] = init_val;
                    }
                }

                if is_het {
                    // At heterozygote: use fwd1 and fwd2 priors for phase decision
                    let b1 = &bwd1_cache[het_idx];
                    let b2 = &bwd2_cache[het_idx];

                    // Compute likelihoods using SEPARATE forward tracks
                    // p11 = P(hap1 emitted allele1 before, continues with allele1)
                    // p12 = P(hap1 emitted allele1 before, now emits allele2) - crossover
                    // etc.
                    let mut p11 = 0.0f64;
                    let mut p12 = 0.0f64;
                    let mut p21 = 0.0f64;
                    let mut p22 = 0.0f64;

                    for k in 0..n_states {
                        let ref_al = ref_alleles[k];
                        let emit1 = if ref_al == allele1 || ref_al == 255 || allele1 == 255 {
                            p_no_err
                        } else {
                            p_err
                        };
                        let emit2 = if ref_al == allele2 || ref_al == 255 || allele2 == 255 {
                            p_no_err
                        } else {
                            p_err
                        };

                        let fwd1_k = fwd1_prior[k] as f64;
                        let fwd2_k = fwd2_prior[k] as f64;
                        let b1_k = b1[k] as f64;
                        let b2_k = b2[k] as f64;

                        // fwd1 was tracking hap1's history, fwd2 was tracking hap2's history
                        p11 += fwd1_k * (emit1 as f64) * b1_k; // hap1 continues with allele1
                        p12 += fwd1_k * (emit2 as f64) * b2_k; // hap1 switches to allele2
                        p21 += fwd2_k * (emit1 as f64) * b1_k; // hap2 switches to allele1
                        p22 += fwd2_k * (emit2 as f64) * b2_k; // hap2 continues with allele2
                    }

                    // Compare joint likelihoods: keep (p11*p22) vs swap (p12*p21)
                    let l_keep = p11 * p22;
                    let l_swap = p12 * p21;

                    if l_swap > l_keep {
                        // SWAP: exchange alleles from this position forward
                        for m_swap in m..n_markers {
                            std::mem::swap(&mut seq1_working[m_swap], &mut seq2_working[m_swap]);
                        }
                        // Swap backward caches for future het decisions
                        for h in het_idx..n_hets {
                            std::mem::swap(&mut bwd1_cache[h], &mut bwd2_cache[h]);
                        }
                    }

                    // Apply combined emission to fwd0 (match-any at hets)
                    fwd0_sum = 0.0;
                    for k in 0..n_states {
                        let ref_al = ref_alleles[k];
                        let emit = if ref_al == allele1 || ref_al == allele2 || ref_al == 255 {
                            p_no_err
                        } else {
                            p_err
                        };
                        fwd0[k] = fwd0_prior[k] * emit;
                        fwd0_sum += fwd0[k];
                    }
                    fwd0_sum = fwd0_sum.max(1e-30);

                    // RESET: fwd1 = fwd2 = fwd0 (after decision, tracks converge)
                    fwd1.copy_from_slice(&fwd0);
                    fwd2.copy_from_slice(&fwd0);
                    fwd1_sum = fwd0_sum;
                    fwd2_sum = fwd0_sum;

                    het_idx += 1;
                } else {
                    // Not a het: all tracks emit the observed (homozygous) allele
                    let observed = if allele1 != 255 { allele1 } else { allele2 };

                    fwd0_sum = 0.0;
                    fwd1_sum = 0.0;
                    fwd2_sum = 0.0;

                    for k in 0..n_states {
                        let ref_al = ref_alleles[k];
                        let emit = if ref_al == observed || ref_al == 255 || observed == 255 {
                            p_no_err
                        } else {
                            p_err
                        };
                        fwd0[k] = fwd0_prior[k] * emit;
                        fwd1[k] = fwd1_prior[k] * emit;
                        fwd2[k] = fwd2_prior[k] * emit;
                        fwd0_sum += fwd0[k];
                        fwd1_sum += fwd1[k];
                        fwd2_sum += fwd2[k];
                    }
                    fwd0_sum = fwd0_sum.max(1e-30);
                    fwd1_sum = fwd1_sum.max(1e-30);
                    fwd2_sum = fwd2_sum.max(1e-30);
                }
            }

            // Determine which markers were swapped by comparing final working sequences to originals
            for m in 0..n_markers {
                if seq1_working[m] != seq1[m] {
                    mask.set(m, true);
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
    
    /// Run a phasing iteration with overlap constraint for streaming mode
    ///
    /// Similar to `run_phase_baum_iteration` but respects the overlap constraint:
    /// markers in the overlap region (0..overlap_markers) have their phase locked
    /// from the previous window and will not be changed.
    ///
    /// Uses the same correct per-heterozygote decision algorithm as `run_phase_baum_iteration`:
    /// - Sparse backward pass (O(N_hets × K) memory)
    /// - Prior-first forward (no division)
    /// - Match-any emission for combined track
    fn run_phase_baum_iteration_with_overlap(
        &mut self,
        target_gt: &GenotypeMatrix,
        geno: &mut MutableGenotypes,
        p_recomb: &[f32],
        gen_dists: &[f64],
        ibs2: &Ibs2,
        sample_phases: &mut [SamplePhase],
        overlap_markers: usize,
        atomic_estimates: Option<&crate::model::parameters::AtomicParamEstimates>,
        iteration: usize,
    ) -> Result<()> {
        let n_markers = geno.n_markers();
        let n_haps = geno.n_haps();
        let markers = target_gt.markers();
        let seed = self.config.seed;

        let ref_geno = geno.clone();
        let ref_view = GenotypeView::from((&ref_geno, markers));

        // Build bidirectional PBWT for better state selection around recombination hotspots
        let phase_ibs = self.build_bidirectional_pbwt(&ref_geno, n_markers, n_haps);

        // Collect swap masks per sample
        let n_samples = n_haps / 2;
        let swap_masks: Vec<BitBox<u64, Lsb0>> = sample_phases
            .par_iter()
            .enumerate()
            .map(|(s, sp)| {
                let sample_idx = SampleIdx::new(s as u32);
                let hap1 = sample_idx.hap1();

                // Use midpoint marker for PBWT-based state selection
                let mid_marker = n_markers / 2;
                let states = self.select_states_bidirectional(hap1, mid_marker, &phase_ibs, ibs2, self.params.n_states, n_haps, seed, iteration);

                // Extract current alleles from SamplePhase
                let seq1: Vec<u8> = (0..n_markers).map(|m| sp.allele1(m)).collect();
                let seq2: Vec<u8> = (0..n_markers).map(|m| sp.allele2(m)).collect();

                let n_states = states.len();

                // Collect EM statistics if requested
                if let Some(atomic) = atomic_estimates {
                    let threaded_haps = ThreadedHaps::from_static_haps(&states, n_markers);
                    let hmm = BeagleHmm::new(ref_view.clone(), &self.params, n_states, p_recomb.to_vec());
                    let mut local_est = crate::model::parameters::ParamEstimates::new();
                    hmm.collect_stats(&seq1, &threaded_haps, gen_dists, &mut local_est);
                    hmm.collect_stats(&seq2, &threaded_haps, gen_dists, &mut local_est);
                    atomic.add_estimation_data(&local_est);
                }

                // Identify heterozygote positions (ONLY after overlap region for decisions)
                // But we still need full het list for backward pass
                let het_positions: Vec<usize> = (0..n_markers)
                    .filter(|&m| {
                        let a1 = seq1[m];
                        let a2 = seq2[m];
                        a1 != 255 && a2 != 255 && a1 != a2
                    })
                    .collect();

                // Find first het index after overlap region
                let first_changeable_het = het_positions.iter().position(|&m| m >= overlap_markers).unwrap_or(het_positions.len());

                let p_err = self.params.p_mismatch;
                let p_no_err = 1.0 - p_err;

                // Helper to get reference allele at (marker, state)
                let get_ref_allele = |m: usize, k: usize| -> u8 {
                    let h = states[k].as_usize();
                    if h < n_haps {
                        ref_geno.get(m, states[k])
                    } else {
                        let ref_h = (h - n_haps) as u32;
                        if let (Some(ref_gt_inner), Some(align)) = (&self.reference_gt, &self.alignment) {
                            if let Some(ref_m) = align.target_to_ref(m) {
                                let ref_allele = ref_gt_inner.allele(MarkerIdx::new(ref_m as u32), HapIdx::new(ref_h));
                                align.reverse_map_allele(m, ref_allele)
                            } else { 255 }
                        } else { 255 }
                    }
                };

                // Sparse backward pass: only store values at het positions
                let n_hets = het_positions.len();
                let mut bwd1_cache: Vec<Vec<f32>> = vec![vec![0.0; n_states]; n_hets];
                let mut bwd2_cache: Vec<Vec<f32>> = vec![vec![0.0; n_states]; n_hets];

                // Run backward for seq1
                {
                    let init_bwd = 1.0f32 / n_states as f32;
                    let mut bwd = vec![init_bwd; n_states];
                    let mut het_rev_idx = n_hets;

                    for m in (0..n_markers).rev() {
                        if het_rev_idx > 0 && het_positions[het_rev_idx - 1] == m {
                            het_rev_idx -= 1;
                            bwd1_cache[het_rev_idx].copy_from_slice(&bwd);
                        }

                        if m > 0 {
                            let p_recomb_m = p_recomb.get(m).copied().unwrap_or(0.0);
                            let shift = p_recomb_m / n_states as f32;
                            let stay = 1.0 - p_recomb_m;

                            let mut bwd_sum = 0.0f32;
                            for k in 0..n_states {
                                let ref_al = get_ref_allele(m, k);
                                let emit = if ref_al == seq1[m] || ref_al == 255 || seq1[m] == 255 {
                                    p_no_err
                                } else {
                                    p_err
                                };
                                bwd[k] *= emit;
                                bwd_sum += bwd[k];
                            }

                            let scale = stay / bwd_sum.max(1e-30);
                            for k in 0..n_states {
                                bwd[k] = scale * bwd[k] + shift;
                            }
                        }
                    }
                }

                // Run backward for seq2
                {
                    let init_bwd = 1.0f32 / n_states as f32;
                    let mut bwd = vec![init_bwd; n_states];
                    let mut het_rev_idx = n_hets;

                    for m in (0..n_markers).rev() {
                        if het_rev_idx > 0 && het_positions[het_rev_idx - 1] == m {
                            het_rev_idx -= 1;
                            bwd2_cache[het_rev_idx].copy_from_slice(&bwd);
                        }

                        if m > 0 {
                            let p_recomb_m = p_recomb.get(m).copied().unwrap_or(0.0);
                            let shift = p_recomb_m / n_states as f32;
                            let stay = 1.0 - p_recomb_m;

                            let mut bwd_sum = 0.0f32;
                            for k in 0..n_states {
                                let ref_al = get_ref_allele(m, k);
                                let emit = if ref_al == seq2[m] || ref_al == 255 || seq2[m] == 255 {
                                    p_no_err
                                } else {
                                    p_err
                                };
                                bwd[k] *= emit;
                                bwd_sum += bwd[k];
                            }

                            let scale = stay / bwd_sum.max(1e-30);
                            for k in 0..n_states {
                                bwd[k] = scale * bwd[k] + shift;
                            }
                        }
                    }
                }

                // 3-Track Forward Pass with overlap constraint
                let mut seq1_working = seq1.clone();
                let mut seq2_working = seq2.clone();
                let mut mask = bitbox![u64, Lsb0; 0; n_markers];

                let init_val = 1.0f32 / n_states as f32;
                let mut fwd0 = vec![init_val; n_states]; // Combined track
                let mut fwd1 = vec![init_val; n_states]; // Hap1 track
                let mut fwd2 = vec![init_val; n_states]; // Hap2 track
                let mut fwd0_prior = vec![0.0f32; n_states];
                let mut fwd1_prior = vec![0.0f32; n_states];
                let mut fwd2_prior = vec![0.0f32; n_states];
                let mut ref_alleles = vec![0u8; n_states];
                let mut fwd0_sum = 1.0f32;
                let mut fwd1_sum = 1.0f32;
                let mut fwd2_sum = 1.0f32;

                let mut het_idx = 0;
                for m in 0..n_markers {
                    let allele1 = seq1_working[m];
                    let allele2 = seq2_working[m];
                    let is_het = het_idx < n_hets && het_positions[het_idx] == m;

                    // Cache reference alleles for this marker
                    for k in 0..n_states {
                        ref_alleles[k] = get_ref_allele(m, k);
                    }

                    // Compute PRIOR for all three tracks
                    if m > 0 {
                        let p_recomb_m = p_recomb.get(m).copied().unwrap_or(0.0);
                        let shift = p_recomb_m / n_states as f32;

                        let scale0 = (1.0 - p_recomb_m) / fwd0_sum;
                        let scale1 = (1.0 - p_recomb_m) / fwd1_sum;
                        let scale2 = (1.0 - p_recomb_m) / fwd2_sum;

                        for k in 0..n_states {
                            fwd0_prior[k] = scale0 * fwd0[k] + shift;
                            fwd1_prior[k] = scale1 * fwd1[k] + shift;
                            fwd2_prior[k] = scale2 * fwd2[k] + shift;
                        }
                    } else {
                        for k in 0..n_states {
                            fwd0_prior[k] = init_val;
                            fwd1_prior[k] = init_val;
                            fwd2_prior[k] = init_val;
                        }
                    }

                    if is_het {
                        // Only make phase decisions for hets AFTER overlap region
                        if het_idx >= first_changeable_het {
                            let b1 = &bwd1_cache[het_idx];
                            let b2 = &bwd2_cache[het_idx];

                            // Use fwd1 and fwd2 for proper 3-track likelihoods
                            let mut p11 = 0.0f64;
                            let mut p12 = 0.0f64;
                            let mut p21 = 0.0f64;
                            let mut p22 = 0.0f64;

                            for k in 0..n_states {
                                let ref_al = ref_alleles[k];
                                let emit1 = if ref_al == allele1 || ref_al == 255 || allele1 == 255 {
                                    p_no_err
                                } else {
                                    p_err
                                };
                                let emit2 = if ref_al == allele2 || ref_al == 255 || allele2 == 255 {
                                    p_no_err
                                } else {
                                    p_err
                                };

                                let fwd1_k = fwd1_prior[k] as f64;
                                let fwd2_k = fwd2_prior[k] as f64;
                                let b1_k = b1[k] as f64;
                                let b2_k = b2[k] as f64;

                                p11 += fwd1_k * (emit1 as f64) * b1_k;
                                p12 += fwd1_k * (emit2 as f64) * b2_k;
                                p21 += fwd2_k * (emit1 as f64) * b1_k;
                                p22 += fwd2_k * (emit2 as f64) * b2_k;
                            }

                            let l_keep = p11 * p22;
                            let l_swap = p12 * p21;

                            if l_swap > l_keep {
                                // SWAP: exchange alleles from this position forward
                                for m_swap in m..n_markers {
                                    std::mem::swap(&mut seq1_working[m_swap], &mut seq2_working[m_swap]);
                                }
                                // Swap backward caches for future het decisions
                                for h in het_idx..n_hets {
                                    std::mem::swap(&mut bwd1_cache[h], &mut bwd2_cache[h]);
                                }
                            }
                        }

                        // Apply combined emission to fwd0 (match-any at hets)
                        fwd0_sum = 0.0;
                        for k in 0..n_states {
                            let ref_al = ref_alleles[k];
                            let emit = if ref_al == allele1 || ref_al == allele2 || ref_al == 255 {
                                p_no_err
                            } else {
                                p_err
                            };
                            fwd0[k] = fwd0_prior[k] * emit;
                            fwd0_sum += fwd0[k];
                        }
                        fwd0_sum = fwd0_sum.max(1e-30);

                        // RESET: fwd1 = fwd2 = fwd0
                        fwd1.copy_from_slice(&fwd0);
                        fwd2.copy_from_slice(&fwd0);
                        fwd1_sum = fwd0_sum;
                        fwd2_sum = fwd0_sum;

                        het_idx += 1;
                    } else {
                        // Not a het: all tracks emit the observed allele
                        let observed = if allele1 != 255 { allele1 } else { allele2 };

                        fwd0_sum = 0.0;
                        fwd1_sum = 0.0;
                        fwd2_sum = 0.0;

                        for k in 0..n_states {
                            let ref_al = ref_alleles[k];
                            let emit = if ref_al == observed || ref_al == 255 || observed == 255 {
                                p_no_err
                            } else {
                                p_err
                            };
                            fwd0[k] = fwd0_prior[k] * emit;
                            fwd1[k] = fwd1_prior[k] * emit;
                            fwd2[k] = fwd2_prior[k] * emit;
                            fwd0_sum += fwd0[k];
                            fwd1_sum += fwd1[k];
                            fwd2_sum += fwd2[k];
                        }
                        fwd0_sum = fwd0_sum.max(1e-30);
                        fwd1_sum = fwd1_sum.max(1e-30);
                        fwd2_sum = fwd2_sum.max(1e-30);
                    }
                }

                // Record swaps by comparing final working sequences to originals
                for m in overlap_markers..n_markers {
                    if seq1_working[m] != seq1[m] {
                        mask.set(m, true);
                    }
                }

                mask
            })
            .collect();

        // Apply Swaps
        let mut total_switches = 0;
        for s in 0..n_samples {
            let mask = &swap_masks[s];
            if mask.any() {
                let hap1 = HapIdx::new((s * 2) as u32);
                let hap2 = HapIdx::new((s * 2 + 1) as u32);

                for m in mask.iter_ones() {
                    geno.swap(m, hap1, hap2);
                    total_switches += 1;
                }
            }
        }

        // Update sample_phases to match
        for (s, sp) in sample_phases.iter_mut().enumerate() {
            for m in overlap_markers..n_markers {
                if swap_masks[s][m] {
                    sp.swap_alleles(m);
                }
            }
        }

        eprintln!("Applied {} phase switches (with {} overlap markers locked)", total_switches, overlap_markers);
        Ok(())
    }

    /// Run Stage 1 phasing iteration on HIGH-FREQUENCY markers only using FB HMM
    ///
    /// Uses SamplePhase to track phase state and only phases unphased markers.
    fn run_phase_baum_iteration_stage1(
        &mut self,
        geno: &mut MutableGenotypes,
        stage1_p_recomb: &[f32],
        stage1_gen_dists: &[f64],
        hi_freq_to_orig: &[usize],
        ibs2: &Ibs2,
        sample_phases: &mut [SamplePhase],
        atomic_estimates: Option<&crate::model::parameters::AtomicParamEstimates>,
        iteration: usize,
    ) -> Result<()> {
        let n_haps = geno.n_haps();
        let seed = self.config.seed;

        // Compute total haplotype count (target + reference)
        let n_ref_haps = self.reference_gt.as_ref().map(|r| r.n_haplotypes()).unwrap_or(0);
        let n_total_haps = n_haps + n_ref_haps;

        // 1. Create Subset View for Stage 1 markers
        // Use CompositeSubset when reference panel is available
        let ref_geno = geno.clone();
        let subset_view = if let (Some(ref_gt), Some(alignment)) = (&self.reference_gt, &self.alignment) {
            GenotypeView::CompositeSubset {
                target: &ref_geno,
                reference: ref_gt,
                alignment,
                subset: hi_freq_to_orig,
                n_target_haps: n_haps,
            }
        } else {
            GenotypeView::MutableSubset {
                geno: &ref_geno,
                subset: hi_freq_to_orig,
            }
        };

        // 2. Build bidirectional PBWT on high-frequency markers only
        // When reference is available, include reference haplotypes in the PBWT
        let phase_ibs = if let (Some(ref_gt), Some(alignment)) = (&self.reference_gt, &self.alignment) {
            self.build_bidirectional_pbwt_combined_subset(
                |orig_m, h| {
                    if h < n_haps {
                        ref_geno.get(orig_m, HapIdx::new(h as u32))
                    } else {
                        let ref_h = h - n_haps;
                        if let Some(ref_m) = alignment.target_to_ref(orig_m) {
                            let ref_allele = ref_gt.allele(MarkerIdx::new(ref_m as u32), HapIdx::new(ref_h as u32));
                            // Map reference allele to target encoding (handles strand flips)
                            alignment.reverse_map_allele(orig_m, ref_allele)
                        } else {
                            255 // Missing - marker not in reference
                        }
                    }
                },
                hi_freq_to_orig,
                n_total_haps,
            )
        } else {
            self.build_bidirectional_pbwt_subset(&ref_geno, hi_freq_to_orig, n_haps)
        };

        // Collect phase decisions per sample using correct per-het algorithm
        // Returns: Vec of (hi_freq_idx, should_swap) for each sample
        let phase_decisions: Vec<Vec<(usize, bool)>> = sample_phases
            .par_iter()
            .enumerate()
            .map(|(s, sp)| {
                let sample_idx = SampleIdx::new(s as u32);
                let hap1 = sample_idx.hap1();

                // Use midpoint of subset marker space for state selection
                let mid_marker = hi_freq_to_orig.len() / 2;
                let states = self.select_states_bidirectional(hap1, mid_marker, &phase_ibs, ibs2, self.params.n_states, n_total_haps, seed, iteration);

                // Extract alleles from SamplePhase for SUBSET of markers
                let seq1: Vec<u8> = hi_freq_to_orig.iter().map(|&m| sp.allele1(m)).collect();
                let seq2: Vec<u8> = hi_freq_to_orig.iter().map(|&m| sp.allele2(m)).collect();

                let n_states = states.len();
                let n_hi_freq = hi_freq_to_orig.len();

                // Collect EM statistics if requested
                if let Some(atomic) = atomic_estimates {
                    let threaded_haps = ThreadedHaps::from_static_haps(&states, n_hi_freq);
                    let hmm = BeagleHmm::new(subset_view.clone(), &self.params, n_states, stage1_p_recomb.to_vec());
                    let mut local_est = crate::model::parameters::ParamEstimates::new();
                    hmm.collect_stats(&seq1, &threaded_haps, stage1_gen_dists, &mut local_est);
                    hmm.collect_stats(&seq2, &threaded_haps, stage1_gen_dists, &mut local_est);
                    atomic.add_estimation_data(&local_est);
                }

                // Identify UNPHASED heterozygote positions in hi-freq marker space
                let het_positions: Vec<usize> = (0..n_hi_freq)
                    .filter(|&i| {
                        let m = hi_freq_to_orig[i];
                        let a1 = seq1[i];
                        let a2 = seq2[i];
                        a1 != 255 && a2 != 255 && a1 != a2 && sp.is_unphased(m)
                    })
                    .collect();

                if het_positions.is_empty() {
                    return Vec::new();
                }

                let p_err = self.params.p_mismatch;
                let p_no_err = 1.0 - p_err;

                // Helper to get reference allele at (hi_freq_idx, state)
                let get_ref_allele = |i: usize, k: usize| -> u8 {
                    let h = states[k].as_usize();
                    let orig_m = hi_freq_to_orig[i];
                    if h < n_haps {
                        ref_geno.get(orig_m, states[k])
                    } else {
                        let ref_h = (h - n_haps) as u32;
                        if let (Some(ref_gt_inner), Some(align)) = (&self.reference_gt, &self.alignment) {
                            if let Some(ref_m) = align.target_to_ref(orig_m) {
                                let ref_allele = ref_gt_inner.allele(MarkerIdx::new(ref_m as u32), HapIdx::new(ref_h));
                                align.reverse_map_allele(orig_m, ref_allele)
                            } else { 255 }
                        } else { 255 }
                    }
                };

                // Sparse backward pass: only store values at het positions
                let n_hets = het_positions.len();
                let mut bwd1_cache: Vec<Vec<f32>> = vec![vec![0.0; n_states]; n_hets];
                let mut bwd2_cache: Vec<Vec<f32>> = vec![vec![0.0; n_states]; n_hets];

                // Run backward for seq1
                {
                    let init_bwd = 1.0f32 / n_states as f32;
                    let mut bwd = vec![init_bwd; n_states];
                    let mut het_rev_idx = n_hets;

                    for i in (0..n_hi_freq).rev() {
                        if het_rev_idx > 0 && het_positions[het_rev_idx - 1] == i {
                            het_rev_idx -= 1;
                            bwd1_cache[het_rev_idx].copy_from_slice(&bwd);
                        }

                        if i > 0 {
                            let p_recomb_i = stage1_p_recomb.get(i).copied().unwrap_or(0.0);
                            let shift = p_recomb_i / n_states as f32;
                            let stay = 1.0 - p_recomb_i;

                            let mut bwd_sum = 0.0f32;
                            for k in 0..n_states {
                                let ref_al = get_ref_allele(i, k);
                                let emit = if ref_al == seq1[i] || ref_al == 255 || seq1[i] == 255 {
                                    p_no_err
                                } else {
                                    p_err
                                };
                                bwd[k] *= emit;
                                bwd_sum += bwd[k];
                            }

                            let scale = stay / bwd_sum.max(1e-30);
                            for k in 0..n_states {
                                bwd[k] = scale * bwd[k] + shift;
                            }
                        }
                    }
                }

                // Run backward for seq2
                {
                    let init_bwd = 1.0f32 / n_states as f32;
                    let mut bwd = vec![init_bwd; n_states];
                    let mut het_rev_idx = n_hets;

                    for i in (0..n_hi_freq).rev() {
                        if het_rev_idx > 0 && het_positions[het_rev_idx - 1] == i {
                            het_rev_idx -= 1;
                            bwd2_cache[het_rev_idx].copy_from_slice(&bwd);
                        }

                        if i > 0 {
                            let p_recomb_i = stage1_p_recomb.get(i).copied().unwrap_or(0.0);
                            let shift = p_recomb_i / n_states as f32;
                            let stay = 1.0 - p_recomb_i;

                            let mut bwd_sum = 0.0f32;
                            for k in 0..n_states {
                                let ref_al = get_ref_allele(i, k);
                                let emit = if ref_al == seq2[i] || ref_al == 255 || seq2[i] == 255 {
                                    p_no_err
                                } else {
                                    p_err
                                };
                                bwd[k] *= emit;
                                bwd_sum += bwd[k];
                            }

                            let scale = stay / bwd_sum.max(1e-30);
                            for k in 0..n_states {
                                bwd[k] = scale * bwd[k] + shift;
                            }
                        }
                    }
                }

                // 3-Track Forward Pass for Stage 1 phasing
                let mut seq1_working = seq1.clone();
                let mut seq2_working = seq2.clone();
                let mut decisions = Vec::new();

                let init_val = 1.0f32 / n_states as f32;
                let mut fwd0 = vec![init_val; n_states]; // Combined track
                let mut fwd1 = vec![init_val; n_states]; // Hap1 track
                let mut fwd2 = vec![init_val; n_states]; // Hap2 track
                let mut fwd0_prior = vec![0.0f32; n_states];
                let mut fwd1_prior = vec![0.0f32; n_states];
                let mut fwd2_prior = vec![0.0f32; n_states];
                let mut ref_alleles = vec![0u8; n_states];
                let mut fwd0_sum = 1.0f32;
                let mut fwd1_sum = 1.0f32;
                let mut fwd2_sum = 1.0f32;

                let mut het_idx = 0;
                for i in 0..n_hi_freq {
                    let allele1 = seq1_working[i];
                    let allele2 = seq2_working[i];
                    let is_het = het_idx < n_hets && het_positions[het_idx] == i;

                    // Cache reference alleles for this marker
                    for k in 0..n_states {
                        ref_alleles[k] = get_ref_allele(i, k);
                    }

                    // Compute PRIOR for all three tracks
                    if i > 0 {
                        let p_recomb_i = stage1_p_recomb.get(i).copied().unwrap_or(0.0);
                        let shift = p_recomb_i / n_states as f32;

                        let scale0 = (1.0 - p_recomb_i) / fwd0_sum;
                        let scale1 = (1.0 - p_recomb_i) / fwd1_sum;
                        let scale2 = (1.0 - p_recomb_i) / fwd2_sum;

                        for k in 0..n_states {
                            fwd0_prior[k] = scale0 * fwd0[k] + shift;
                            fwd1_prior[k] = scale1 * fwd1[k] + shift;
                            fwd2_prior[k] = scale2 * fwd2[k] + shift;
                        }
                    } else {
                        for k in 0..n_states {
                            fwd0_prior[k] = init_val;
                            fwd1_prior[k] = init_val;
                            fwd2_prior[k] = init_val;
                        }
                    }

                    if is_het {
                        let b1 = &bwd1_cache[het_idx];
                        let b2 = &bwd2_cache[het_idx];

                        // Use fwd1 and fwd2 for proper 3-track likelihoods
                        let mut p11 = 0.0f64;
                        let mut p12 = 0.0f64;
                        let mut p21 = 0.0f64;
                        let mut p22 = 0.0f64;

                        for k in 0..n_states {
                            let ref_al = ref_alleles[k];
                            let emit1 = if ref_al == allele1 || ref_al == 255 || allele1 == 255 {
                                p_no_err
                            } else {
                                p_err
                            };
                            let emit2 = if ref_al == allele2 || ref_al == 255 || allele2 == 255 {
                                p_no_err
                            } else {
                                p_err
                            };

                            let fwd1_k = fwd1_prior[k] as f64;
                            let fwd2_k = fwd2_prior[k] as f64;
                            let b1_k = b1[k] as f64;
                            let b2_k = b2[k] as f64;

                            p11 += fwd1_k * (emit1 as f64) * b1_k;
                            p12 += fwd1_k * (emit2 as f64) * b2_k;
                            p21 += fwd2_k * (emit1 as f64) * b1_k;
                            p22 += fwd2_k * (emit2 as f64) * b2_k;
                        }

                        let l_keep = p11 * p22;
                        let l_swap = p12 * p21;

                        let should_swap = l_swap > l_keep;
                        decisions.push((i, should_swap));

                        if should_swap {
                            // SWAP: exchange alleles from this position forward
                            for i_swap in i..n_hi_freq {
                                std::mem::swap(&mut seq1_working[i_swap], &mut seq2_working[i_swap]);
                            }
                            // Swap backward caches for future het decisions
                            for h in het_idx..n_hets {
                                std::mem::swap(&mut bwd1_cache[h], &mut bwd2_cache[h]);
                            }
                        }

                        // Apply combined emission to fwd0 (match-any at hets)
                        fwd0_sum = 0.0;
                        for k in 0..n_states {
                            let ref_al = ref_alleles[k];
                            let emit = if ref_al == allele1 || ref_al == allele2 || ref_al == 255 {
                                p_no_err
                            } else {
                                p_err
                            };
                            fwd0[k] = fwd0_prior[k] * emit;
                            fwd0_sum += fwd0[k];
                        }
                        fwd0_sum = fwd0_sum.max(1e-30);

                        // RESET: fwd1 = fwd2 = fwd0
                        fwd1.copy_from_slice(&fwd0);
                        fwd2.copy_from_slice(&fwd0);
                        fwd1_sum = fwd0_sum;
                        fwd2_sum = fwd0_sum;

                        het_idx += 1;
                    } else {
                        // Not a het: all tracks emit the observed allele
                        let observed = if allele1 != 255 { allele1 } else { allele2 };

                        fwd0_sum = 0.0;
                        fwd1_sum = 0.0;
                        fwd2_sum = 0.0;

                        for k in 0..n_states {
                            let ref_al = ref_alleles[k];
                            let emit = if ref_al == observed || ref_al == 255 || observed == 255 {
                                p_no_err
                            } else {
                                p_err
                            };
                            fwd0[k] = fwd0_prior[k] * emit;
                            fwd1[k] = fwd1_prior[k] * emit;
                            fwd2[k] = fwd2_prior[k] * emit;
                            fwd0_sum += fwd0[k];
                            fwd1_sum += fwd1[k];
                            fwd2_sum += fwd2[k];
                        }
                        fwd0_sum = fwd0_sum.max(1e-30);
                        fwd1_sum = fwd1_sum.max(1e-30);
                        fwd2_sum = fwd2_sum.max(1e-30);
                    }
                }

                decisions
            })
            .collect();

        // Apply phase decisions to SamplePhase
        let mut total_switches = 0;
        let mut total_phased = 0;

        for (s, decisions) in phase_decisions.into_iter().enumerate() {
            let sp = &mut sample_phases[s];

            for (hi_freq_idx, should_swap) in decisions {
                let m = hi_freq_to_orig[hi_freq_idx];

                // Apply swap if needed
                if should_swap {
                    sp.swap_alleles(m);
                    total_switches += 1;
                }

                // Mark as phased (we made a decision)
                sp.mark_phased(m);
                total_phased += 1;
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
        let seed = self.config.seed;

        if n_stage1 < 2 {
            return;
        }

        // Compute total haplotype count (target + reference)
        let n_ref_haps = self.reference_gt.as_ref().map(|r| r.n_haplotypes()).unwrap_or(0);
        let n_total_haps = n_haps + n_ref_haps;

        // Build Stage 2 interpolation mappings
        let stage2_phaser = Stage2Phaser::new(hi_freq_markers, gen_positions, n_markers);

        // Clone current genotypes to use as a frozen reference panel
        let ref_geno = geno.clone();

        // Use CompositeSubset view when reference panel is available
        let subset_view = if let (Some(ref_gt), Some(alignment)) = (&self.reference_gt, &self.alignment) {
            GenotypeView::CompositeSubset {
                target: &ref_geno,
                reference: ref_gt,
                alignment,
                subset: hi_freq_markers,
                n_target_haps: n_haps,
            }
        } else {
            GenotypeView::MutableSubset {
                geno: &ref_geno,
                subset: hi_freq_markers,
            }
        };

        // Build bidirectional PBWT on hi-freq markers for consistent state selection
        // When reference is available, include reference haplotypes
        let phase_ibs = if let (Some(ref_gt), Some(alignment)) = (&self.reference_gt, &self.alignment) {
            self.build_bidirectional_pbwt_combined_subset(
                |orig_m, h| {
                    if h < n_haps {
                        ref_geno.get(orig_m, HapIdx::new(h as u32))
                    } else {
                        let ref_h = h - n_haps;
                        if let Some(ref_m) = alignment.target_to_ref(orig_m) {
                            let ref_allele = ref_gt.allele(MarkerIdx::new(ref_m as u32), HapIdx::new(ref_h as u32));
                            // Map reference allele to target encoding (handles strand flips)
                            alignment.reverse_map_allele(orig_m, ref_allele)
                        } else {
                            255 // Missing - marker not in reference
                        }
                    }
                },
                hi_freq_markers,
                n_total_haps,
            )
        } else {
            self.build_bidirectional_pbwt_subset(&ref_geno, hi_freq_markers, n_haps)
        };

        // Process samples in parallel - collect results: (marker, should_swap)
        // Note: This is called after all iterations, so we use iteration=0 for deterministic state selection
        let phase_changes: Vec<Vec<(usize, bool)>> = sample_phases
            .par_iter()
            .enumerate()
            .map(|(s, sp)| {
                let sample_idx = SampleIdx::new(s as u32);
                let hap1 = sample_idx.hap1();

                // Select HMM states using bidirectional PBWT on subset marker space
                let mid_marker = n_stage1 / 2;
                // Use n_total_haps to allow selecting reference haplotypes as states
                let states = self.select_states_bidirectional(hap1, mid_marker, &phase_ibs, ibs2, self.params.n_states, n_total_haps, seed, 0);
                let n_states = states.len();

                // Extract Stage 1 alleles from SamplePhase
                let seq1: Vec<u8> = hi_freq_markers.iter().map(|&m| sp.allele1(m)).collect();
                let seq2: Vec<u8> = hi_freq_markers.iter().map(|&m| sp.allele2(m)).collect();

                // Run HMM forward-backward for both haplotypes on Stage 1 markers
                let threaded_haps = ThreadedHaps::from_static_haps(&states, n_stage1);
                let hmm = BeagleHmm::new(subset_view, &self.params, n_states, stage1_p_recomb.to_vec());

                let mut fwd1 = Vec::new();
                let mut bwd1 = Vec::new();
                hmm.forward_backward_raw(&seq1, &threaded_haps, &mut fwd1, &mut bwd1);

                let mut fwd2 = Vec::new();
                let mut bwd2 = Vec::new();
                hmm.forward_backward_raw(&seq2, &threaded_haps, &mut fwd2, &mut bwd2);

                // Compute posterior state probabilities at each Stage 1 marker
                let probs1 = compute_state_posteriors(&fwd1, &bwd1, n_stage1, n_states);
                let probs2 = compute_state_posteriors(&fwd2, &bwd2, n_stage1, n_states);

                // Collect phase decisions for this sample
                let mut phase_decisions = Vec::new();

                // Closure to get allele for any haplotype (target or reference)
                let get_allele = |marker: usize, hap: usize| -> u8 {
                    if hap < n_haps {
                        // Target haplotype
                        ref_geno.get(marker, HapIdx::new(hap as u32))
                    } else {
                        // Reference haplotype
                        let ref_h = hap - n_haps;
                        if let (Some(ref_gt), Some(alignment)) = (&self.reference_gt, &self.alignment) {
                            if let Some(ref_m) = alignment.target_to_ref(marker) {
                                let ref_allele = ref_gt.allele(MarkerIdx::new(ref_m as u32), HapIdx::new(ref_h as u32));
                                alignment.reverse_map_allele(marker, ref_allele)
                            } else {
                                255 // Missing - marker not in reference
                            }
                        } else {
                            255 // No reference panel
                        }
                    }
                };

                let mut process_marker = |m: usize| {
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
                        m, &probs1, &states, &get_allele, a1, a2,
                    );
                    let al_probs2 = stage2_phaser.interpolated_allele_probs(
                        m, &probs2, &states, &get_allele, a1, a2,
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
    fn interpolated_allele_probs<F>(
        &self,
        marker: usize,
        state_probs: &[Vec<f32>], // [stage1_marker][state]
        states: &[HapIdx],        // HMM state haplotype indices
        get_allele: &F,           // Closure to get allele for any haplotype
        a1: u8,
        a2: u8,
    ) -> [f32; 2]
    where
        F: Fn(usize, usize) -> u8, // (marker, hap_index) -> allele
    {
        let mut al_probs = [0.0f32; 2];

        let mkr_a = self.prev_stage1_marker[marker];
        let mkr_b = (mkr_a + 1).min(self.n_stage1 - 1);
        let wt = self.prev_stage1_wt[marker];

        let probs_a = &state_probs[mkr_a];
        let probs_b = &state_probs[mkr_b];

        for (j, &hap_idx) in states.iter().enumerate() {
            let hap = hap_idx.0 as usize;

            // Get allele from this haplotype at rare marker (handles both target and reference)
            let b1 = get_allele(marker, hap);

            // Get allele from paired haplotype (for het handling)
            // For target: paired_hap = hap ^ 1 (same sample, other haplotype)
            // For reference: paired_hap = hap ^ 1 (same sample in ref panel)
            let paired_hap = hap ^ 1;
            let b2 = get_allele(marker, paired_hap);

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
                let match1 = a1 == b1 || a1 == b2;
                let match2 = a2 == b1 || a2 == b2;

                if match1 && !match2 {
                    // Only a1 matches - favor a1 with heuristic weight
                    al_probs[0] += 0.55 * prob;
                    al_probs[1] += 0.45 * prob;
                } else if match2 && !match1 {
                    // Only a2 matches - favor a2 with heuristic weight
                    al_probs[0] += 0.45 * prob;
                    al_probs[1] += 0.55 * prob;
                } else {
                    // Both match or neither match (ambiguous) - split 50/50
                    // This preserves probabilistic information from the HMM
                    al_probs[0] += 0.5 * prob;
                    al_probs[1] += 0.5 * prob;
                }
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