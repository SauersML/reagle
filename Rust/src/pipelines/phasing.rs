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

use std::sync::Arc;

use bitvec::prelude::*;
use rayon::prelude::*;
use tracing::instrument;

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
use crate::model::parameters::ModelParams;
use crate::model::phase_ibs::BidirectionalPhaseIbs;
use crate::model::phase_states::PhaseStates;
use crate::pipelines::imputation::MarkerAlignment;

/// Phasing pipeline
pub struct PhasingPipeline {
    config: Config,
    params: ModelParams,
    /// Reference panel for reference-guided phasing (optional)
    /// Uses Arc for shared ownership to avoid cloning the large reference panel
    reference_gt: Option<Arc<GenotypeMatrix<Phased>>>,
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

    /// Set reference panel for reference-guided phasing
    ///
    /// When a reference panel is provided, the phasing algorithm uses it to:
    /// 1. Improve state selection (PBWT neighbors from reference)
    /// 2. Guide phase decisions with reference haplotypes
    ///
    /// Uses Arc for shared ownership to avoid cloning the large reference panel.
    pub fn set_reference(&mut self, reference: Arc<GenotypeMatrix<Phased>>, alignment: MarkerAlignment) {
        self.reference_gt = Some(reference);
        self.alignment = Some(alignment);
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
            self.reference_gt = Some(Arc::new(ref_gt));
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

        // Log ploidy information from detected samples
        let samples = target_gt.samples_arc();
        let n_haploid = (0..n_samples)
            .filter(|&s| !samples.is_diploid(SampleIdx::new(s as u32)))
            .count();
        if n_haploid > 0 {
            eprintln!(
                "Detected {} haploid samples ({} diploid), {} true haplotypes",
                n_haploid,
                n_samples - n_haploid,
                samples.n_true_haps()
            );
        }

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
                &maf,
                rare_threshold,
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
    #[instrument(name = "phase_in_memory", skip(self, target_gt, gen_maps))]
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
                target_gt,
                &mut geno,
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
                target_gt,
                &mut geno,
                &p_recomb,
                &gen_dists,
                &ibs2,
                &mut sample_phases,
                overlap_markers,
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

    /// Build bidirectional PBWT for full chromosome phasing
    fn build_bidirectional_pbwt(
        &self,
        geno: &MutableGenotypes,
        n_markers: usize,
        n_haps: usize,
    ) -> BidirectionalPhaseIbs {
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

        BidirectionalPhaseIbs::build_for_subset(
            &alleles_by_marker,
            n_haps,
            n_subset,
            marker_indices,
        )
    }

    /// Build bidirectional PBWT for combined target + reference haplotype space
    fn build_bidirectional_pbwt_combined<F>(
        &self,
        get_allele: F,
        n_markers: usize,
        n_total_haps: usize,
    ) -> BidirectionalPhaseIbs
    where
        F: Fn(usize, usize) -> u8,
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
    fn build_bidirectional_pbwt_combined_subset<F>(
        &self,
        get_allele: F,
        marker_indices: &[usize],
        n_total_haps: usize,
    ) -> BidirectionalPhaseIbs
    where
        F: Fn(usize, usize) -> u8,
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

        BidirectionalPhaseIbs::build_for_subset(
            &alleles_by_marker,
            n_total_haps,
            n_subset,
            marker_indices,
        )
    }

    /// Run a single phasing iteration using Forward-Backward Li-Stephens HMM
    ///
    /// This uses the full Forward-Backward algorithm to compute posterior probabilities
    /// of the phase, ensuring that phasing decisions are informed by both upstream
    /// and downstream data.
    #[instrument(skip_all, fields(n_samples, n_markers))]
    fn run_phase_baum_iteration(
        &mut self,
        target_gt: &GenotypeMatrix,
        geno: &mut MutableGenotypes,
        p_recomb: &[f32],
        gen_dists: &[f64],
        ibs2: &Ibs2,
        atomic_estimates: Option<&crate::model::parameters::AtomicParamEstimates>,
    ) -> Result<()> {
        let n_samples = geno.n_haps() / 2;
        let n_markers = geno.n_markers();
        let n_haps = geno.n_haps();
        let markers = target_gt.markers();

        tracing::Span::current().record("n_samples", n_samples);
        tracing::Span::current().record("n_markers", n_markers);

        // Compute total haplotype count (target + reference)
        let n_ref_haps = self.reference_gt.as_ref().map(|r| r.n_haplotypes()).unwrap_or(0);
        let n_total_haps = n_haps + n_ref_haps;

        let ref_geno = tracing::info_span!("clone_geno").in_scope(|| geno.clone());

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
        let phase_ibs = tracing::info_span!("build_pbwt").in_scope(|| {
            if let (Some(ref_gt), Some(alignment)) = (&self.reference_gt, &self.alignment) {
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
            }
        });

        let mut swap_masks: Vec<BitVec<u8, Lsb0>> = vec![BitVec::repeat(false, n_markers); n_samples];

        tracing::info_span!("hmm_samples").in_scope(|| swap_masks.par_iter_mut().enumerate().for_each(|(s, mask)| {
            let sample_idx = SampleIdx::new(s as u32);
            let hap1 = sample_idx.hap1();
            let hap2 = sample_idx.hap2();

            // Build dynamic composite haplotypes using PhaseStates
            // This iterates through all markers and builds mosaic haplotypes
            // that provide local IBS matches everywhere, not just at midpoint.
            let mut phase_states = PhaseStates::new(self.params.n_states, n_markers);
            let threaded_haps = phase_states.build_composite_haps(
                s as u32,
                &phase_ibs,
                ibs2,
                20, // n_candidates per marker
            );
            let n_states = phase_states.n_states();

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
            let hmm = BeagleHmm::new(ref_view, &self.params, n_states, p_recomb.to_vec());

            // Collect EM statistics if requested (using original sequences)
            if let Some(atomic) = atomic_estimates {
                let mut local_est = crate::model::parameters::ParamEstimates::new();
                hmm.collect_stats(&seq1, &threaded_haps, gen_dists, &mut local_est);
                hmm.collect_stats(&seq2, &threaded_haps, gen_dists, &mut local_est);
                atomic.add_estimation_data(&local_est);
            }

            // 3-Track HMM with Prior-First Approach
            //
            // This implementation avoids the numerically unstable division workaround.
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

            // Pre-compute state->hap mapping for all (marker, state) pairs
            // This is needed because ThreadedHaps uses cursor-based traversal
            // Pre-allocate all memory upfront to avoid clone() overhead in hot loop
            let state_haps: Vec<Vec<u32>> = {
                let mut threaded_haps_mut = threaded_haps.clone();
                let mut state_haps = vec![vec![0u32; n_states]; n_markers];
                for (m, item) in state_haps.iter_mut().enumerate().take(n_markers) {
                    threaded_haps_mut.materialize_haps(m, item);
                }
                state_haps
            };

            // Helper to get reference allele at (marker, state)
            let get_ref_allele = |m: usize, k: usize| -> u8 {
                let h = state_haps[m][k] as usize;
                if h < n_haps {
                    ref_geno.get(m, HapIdx::new(h as u32))
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

            // ═══════════════════════════════════════════════════════════════════════
            // 3-Track Backward Pass with Collapsing at Heterozygotes
            // ═══════════════════════════════════════════════════════════════════════
            //
            // Mathematical Basis:
            // ───────────────────────────────────────────────────────────────────────
            // Let G_{1:M} be the sequence of unphased genotypes.
            // Let Z_m ∈ {1...K} be the hidden state at marker m.
            // Let φ_m ∈ {0,1} be the phase orientation at het m.
            //
            // The backward variable β_m(k) should represent:
            //   β_m(k) = P(G_{m+1:M} | Z_m = k)
            //
            // This is the likelihood of future GENOTYPES, not future haplotype estimates.
            //
            // Bug in previous implementation:
            // ───────────────────────────────────────────────────────────────────────
            // The old code computed β^(1)_m(k) = P(Ĥ^(1)_{m+1:M} | Z_m = k) where Ĥ^(1)
            // is the CURRENT haplotype-1 estimate. This conditions on the arbitrary
            // initial phase assignment, biasing decisions toward preserving random
            // initialization and preventing proper convergence.
            //
            // CORRECT ALGORITHM (Collapsing):
            // ───────────────────────────────────────────────────────────────────────
            // At each unphased het (going backward), the phase is AMBIGUOUS relative
            // to upstream decisions. We must marginalize over this ambiguity:
            //
            //   P(G_{m+1:M} | Z_m) = Σ_{φ_{m+1}} P(G_{m+1:M} | Z_m, φ_{m+1}) P(φ_{m+1})
            //
            // The 3-track model implements this by COLLAPSING at each het:
            //   1. Maintain 3 backward tracks: β^(0) (combined), β^(1), β^(2)
            //   2. At each het position m (going backward):
            //      a. STORE β^(1) and β^(2) for phase decision (they differ from prev het)
            //      b. COLLAPSE: β^(1) ← β^(0), β^(2) ← β^(0)
            //   3. Continue backward with collapsed values
            //
            // This ensures:
            //   - Backward info at het m includes emission at m+1 (one het lookahead)
            //   - Info from m+2, m+3, ... is MARGINALIZED (same for both tracks)
            //   - No bias from initial phase assignment
            // ═══════════════════════════════════════════════════════════════════════

            let n_hets = het_positions.len();
            let mut bwd1_cache: Vec<Vec<f32>> = vec![vec![0.0; n_states]; n_hets];
            let mut bwd2_cache: Vec<Vec<f32>> = vec![vec![0.0; n_states]; n_hets];

            {
                let init_bwd = 1.0f32 / n_states as f32;

                // Three backward tracks (all start equal at the end)
                let mut bwd0 = vec![init_bwd; n_states]; // Combined track (match-any at hets)
                let mut bwd1 = vec![init_bwd; n_states]; // Track 1 (emits allele1 at hets)
                let mut bwd2 = vec![init_bwd; n_states]; // Track 2 (emits allele2 at hets)

                let mut het_rev_idx = n_hets;

                for m in (0..n_markers).rev() {
                    let allele1 = seq1[m];
                    let allele2 = seq2[m];
                    let is_het = allele1 != 255 && allele2 != 255 && allele1 != allele2;

                    // At het positions: STORE then COLLAPSE
                    if is_het && het_rev_idx > 0 && het_positions[het_rev_idx - 1] == m {
                        het_rev_idx -= 1;

                        // Store BEFORE collapsing (tracks differ due to emission at m+1)
                        bwd1_cache[het_rev_idx].copy_from_slice(&bwd1);
                        bwd2_cache[het_rev_idx].copy_from_slice(&bwd2);

                        // COLLAPSE: marginalize over phase ambiguity at this het
                        // After this, backward info flowing to m-1 is phase-agnostic
                        bwd1.copy_from_slice(&bwd0);
                        bwd2.copy_from_slice(&bwd0);
                    }

                    if m > 0 {
                        let p_recomb_m = p_recomb.get(m).copied().unwrap_or(0.0);
                        let shift = p_recomb_m / n_states as f32;
                        let stay = 1.0 - p_recomb_m;

                        let mut bwd0_sum = 0.0f32;
                        let mut bwd1_sum = 0.0f32;
                        let mut bwd2_sum = 0.0f32;

                        for (k, item) in bwd0.iter_mut().enumerate().take(n_states) {
                            let ref_al = get_ref_allele(m, k);

                            // Track 0: Combined emission (match-any at hets)
                            let emit0 = if is_het {
                                if ref_al == allele1 || ref_al == allele2 || ref_al == 255 {
                                    p_no_err
                                } else {
                                    p_err
                                }
                            } else {
                                let obs = if allele1 != 255 { allele1 } else { allele2 };
                                if ref_al == obs || ref_al == 255 || obs == 255 {
                                    p_no_err
                                } else {
                                    p_err
                                }
                            };

                            // Track 1: Emits allele1
                            let emit1 = if ref_al == allele1 || ref_al == 255 || allele1 == 255 {
                                p_no_err
                            } else {
                                p_err
                            };

                            // Track 2: Emits allele2
                            let emit2 = if ref_al == allele2 || ref_al == 255 || allele2 == 255 {
                                p_no_err
                            } else {
                                p_err
                            };

                            *item *= emit0;
                            bwd1[k] *= emit1;
                            bwd2[k] *= emit2;

                            bwd0_sum += *item;
                            bwd1_sum += bwd1[k];
                            bwd2_sum += bwd2[k];
                        }

                        // Apply transition (backward direction) with normalization
                        let scale0 = stay / bwd0_sum.max(1e-30);
                        let scale1 = stay / bwd1_sum.max(1e-30);
                        let scale2 = stay / bwd2_sum.max(1e-30);

                        for (k, item) in bwd0.iter_mut().enumerate().take(n_states) {
                            *item = scale0 * *item + shift;
                            bwd1[k] = scale1 * bwd1[k] + shift;
                            bwd2[k] = scale2 * bwd2[k] + shift;
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
                for (k, item) in ref_alleles.iter_mut().enumerate().take(n_states) {
                    *item = get_ref_allele(m, k);
                }

                // Compute PRIOR for all three tracks (transition only, no emission)
                if m > 0 {
                    let p_recomb_m = p_recomb.get(m).copied().unwrap_or(0.0);
                    let shift = p_recomb_m / n_states as f32;

                    let scale0 = (1.0 - p_recomb_m) / fwd0_sum;
                    let scale1 = (1.0 - p_recomb_m) / fwd1_sum;
                    let scale2 = (1.0 - p_recomb_m) / fwd2_sum;

                    for (k, item) in fwd0_prior.iter_mut().enumerate().take(n_states) {
                        *item = scale0 * fwd0[k] + shift;
                        fwd1_prior[k] = scale1 * fwd1[k] + shift;
                        fwd2_prior[k] = scale2 * fwd2[k] + shift;
                    }
                } else {
                    for (k, item) in fwd0_prior.iter_mut().enumerate().take(n_states) {
                        *item = init_val;
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

                    for (k, ref_al_item) in ref_alleles.iter().enumerate().take(n_states) {
                        let ref_al = *ref_al_item;
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
                            seq1_working.swap(m_swap, m_swap);
                        }
                        // Swap backward caches for future het decisions
                        for h in het_idx..n_hets {
                            bwd1_cache.swap(h, h);
                        }
                    }

                    // Apply combined emission to fwd0 (match-any at hets)
                    fwd0_sum = 0.0;
                    for (k, item) in fwd0.iter_mut().enumerate().take(n_states) {
                        let ref_al = ref_alleles[k];
                        let emit = if ref_al == allele1 || ref_al == allele2 || ref_al == 255 {
                            p_no_err
                        } else {
                            p_err
                        };
                        *item = fwd0_prior[k] * emit;
                        fwd0_sum += *item;
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

                    for (k, item) in fwd0.iter_mut().enumerate().take(n_states) {
                        let ref_al = ref_alleles[k];
                        let emit = if ref_al == observed || ref_al == 255 || observed == 255 {
                            p_no_err
                        } else {
                            p_err
                        };
                        *item = fwd0_prior[k] * emit;
                        fwd1[k] = fwd1_prior[k] * emit;
                        fwd2[k] = fwd2_prior[k] * emit;
                        fwd0_sum += *item;
                        fwd1_sum += fwd1[k];
                        fwd2_sum += fwd2[k];
                    }
                    fwd0_sum = fwd0_sum.max(1e-30);
                    fwd1_sum = fwd1_sum.max(1e-30);
                    fwd2_sum = fwd2_sum.max(1e-30);
                }
            }

            // Determine which markers were swapped by comparing final working sequences to originals
            for (m, mask_bit) in mask.iter_mut().enumerate().take(n_markers) {
                if seq1_working[m] != seq1[m] {
                    mask_bit.set(true);
                }
            }
        }));

        // Apply Swaps
        let mut total_switches = 0;
        for (s, mask) in swap_masks.iter().enumerate().take(n_samples) {
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
    ) -> Result<()> {
        let n_markers = geno.n_markers();
        let n_haps = geno.n_haps();
        let markers = target_gt.markers();

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
                // Build dynamic composite haplotypes using PhaseStates
                let mut phase_states = PhaseStates::new(self.params.n_states, n_markers);
                let threaded_haps = phase_states.build_composite_haps(
                    s as u32,
                    &phase_ibs,
                    ibs2,
                    20,
                );
                let n_states = phase_states.n_states();

                // Extract current alleles from SamplePhase
                let seq1: Vec<u8> = (0..n_markers).map(|m| sp.allele1(m)).collect();
                let seq2: Vec<u8> = (0..n_markers).map(|m| sp.allele2(m)).collect();

                // Collect EM statistics if requested
                if let Some(atomic) = atomic_estimates {
                    let hmm = BeagleHmm::new(ref_view, &self.params, n_states, p_recomb.to_vec());
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

                // Pre-compute state->hap mapping for all (marker, state) pairs
                // Pre-allocate all memory upfront to avoid clone() overhead in hot loop
                let state_haps: Vec<Vec<u32>> = {
                    let mut threaded_haps_mut = threaded_haps.clone();
                    let mut state_haps = vec![vec![0u32; n_states]; n_markers];
                    for m in 0..n_markers {
                        threaded_haps_mut.materialize_haps(m, &mut state_haps[m]);
                    }
                    state_haps
                };

                // Helper to get reference allele at (marker, state)
                let get_ref_allele = |m: usize, k: usize| -> u8 {
                    let h = state_haps[m][k] as usize;
                    if h < n_haps {
                        ref_geno.get(m, HapIdx::new(h as u32))
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

                // ═══════════════════════════════════════════════════════════════════════
                // 3-TRACK BACKWARD PASS WITH COLLAPSING (see run_phase_baum_iteration)
                // ═══════════════════════════════════════════════════════════════════════
                // At each het, we STORE then COLLAPSE to marginalize over phase ambiguity.
                // This prevents the backward pass from conditioning on random initialization.
                let n_hets = het_positions.len();
                let mut bwd1_cache: Vec<Vec<f32>> = vec![vec![0.0; n_states]; n_hets];
                let mut bwd2_cache: Vec<Vec<f32>> = vec![vec![0.0; n_states]; n_hets];

                {
                    let init_bwd = 1.0f32 / n_states as f32;
                    let mut bwd0 = vec![init_bwd; n_states]; // Combined track
                    let mut bwd1 = vec![init_bwd; n_states]; // Track 1
                    let mut bwd2 = vec![init_bwd; n_states]; // Track 2
                    let mut het_rev_idx = n_hets;

                    for m in (0..n_markers).rev() {
                        let allele1 = seq1[m];
                        let allele2 = seq2[m];
                        let is_het = allele1 != 255 && allele2 != 255 && allele1 != allele2;

                        // At het positions: STORE then COLLAPSE
                        if is_het && het_rev_idx > 0 && het_positions[het_rev_idx - 1] == m {
                            het_rev_idx -= 1;
                            bwd1_cache[het_rev_idx].copy_from_slice(&bwd1);
                            bwd2_cache[het_rev_idx].copy_from_slice(&bwd2);
                            // COLLAPSE: marginalize over phase
                            bwd1.copy_from_slice(&bwd0);
                            bwd2.copy_from_slice(&bwd0);
                        }

                        if m > 0 {
                            let p_recomb_m = p_recomb.get(m).copied().unwrap_or(0.0);
                            let shift = p_recomb_m / n_states as f32;
                            let stay = 1.0 - p_recomb_m;

                            let mut bwd0_sum = 0.0f32;
                            let mut bwd1_sum = 0.0f32;
                            let mut bwd2_sum = 0.0f32;

                            for k in 0..n_states {
                                let ref_al = get_ref_allele(m, k);

                                // Track 0: Combined emission
                                let emit0 = if is_het {
                                    if ref_al == allele1 || ref_al == allele2 || ref_al == 255 {
                                        p_no_err
                                    } else {
                                        p_err
                                    }
                                } else {
                                    let obs = if allele1 != 255 { allele1 } else { allele2 };
                                    if ref_al == obs || ref_al == 255 || obs == 255 {
                                        p_no_err
                                    } else {
                                        p_err
                                    }
                                };

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

                                bwd0[k] *= emit0;
                                bwd1[k] *= emit1;
                                bwd2[k] *= emit2;
                                bwd0_sum += bwd0[k];
                                bwd1_sum += bwd1[k];
                                bwd2_sum += bwd2[k];
                            }

                            let scale0 = stay / bwd0_sum.max(1e-30);
                            let scale1 = stay / bwd1_sum.max(1e-30);
                            let scale2 = stay / bwd2_sum.max(1e-30);

                            for k in 0..n_states {
                                bwd0[k] = scale0 * bwd0[k] + shift;
                                bwd1[k] = scale1 * bwd1[k] + shift;
                                bwd2[k] = scale2 * bwd2[k] + shift;
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

        // Collect phase decisions per sample using correct per-het algorithm.
        // Returns: (swap_mask, het_lr_values) per sample where:
        //   - swap_mask[i] = true if hi_freq marker i should be swapped (derived by comparing
        //     final working sequences to originals, capturing cumulative effect of all swaps)
        //   - het_lr_values = (hi_freq_idx, lr) for each het, used for phased marking threshold
        let phase_decisions: Vec<(Vec<bool>, Vec<(usize, f32)>)> = sample_phases
            .par_iter()
            .enumerate()
            .map(|(s, sp)| {
                let n_hi_freq = hi_freq_to_orig.len();

                // Build dynamic composite haplotypes using PhaseStates
                let mut phase_states = PhaseStates::new(self.params.n_states, n_hi_freq);
                let threaded_haps = phase_states.build_composite_haps(
                    s as u32,
                    &phase_ibs,
                    ibs2,
                    20,
                );
                let n_states = phase_states.n_states();

                // Extract alleles from SamplePhase for SUBSET of markers
                let seq1: Vec<u8> = hi_freq_to_orig.iter().map(|&m| sp.allele1(m)).collect();
                let seq2: Vec<u8> = hi_freq_to_orig.iter().map(|&m| sp.allele2(m)).collect();

                // Collect EM statistics if requested
                if let Some(atomic) = atomic_estimates {
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
                    // No hets to phase: no swaps needed, no LR values
                    return (vec![false; n_hi_freq], Vec::new());
                }

                let p_err = self.params.p_mismatch;
                let p_no_err = 1.0 - p_err;

                // Pre-compute state->hap mapping for all (hi_freq_idx, state) pairs
                // Pre-allocate all memory upfront to avoid clone() overhead in hot loop
                let state_haps: Vec<Vec<u32>> = {
                    let mut threaded_haps_mut = threaded_haps.clone();
                    let mut state_haps = vec![vec![0u32; n_states]; n_hi_freq];
                    for m in 0..n_hi_freq {
                        threaded_haps_mut.materialize_haps(m, &mut state_haps[m]);
                    }
                    state_haps
                };

                // Helper to get reference allele at (hi_freq_idx, state)
                let get_ref_allele = |i: usize, k: usize| -> u8 {
                    let h = state_haps[i][k] as usize;
                    let orig_m = hi_freq_to_orig[i];
                    if h < n_haps {
                        ref_geno.get(orig_m, HapIdx::new(h as u32))
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

                // ═══════════════════════════════════════════════════════════════════════
                // 3-TRACK BACKWARD PASS WITH COLLAPSING (see run_phase_baum_iteration)
                // ═══════════════════════════════════════════════════════════════════════
                // At each het, we STORE then COLLAPSE to marginalize over phase ambiguity.
                // This prevents the backward pass from conditioning on random initialization.
                let n_hets = het_positions.len();
                let mut bwd1_cache: Vec<Vec<f32>> = vec![vec![0.0; n_states]; n_hets];
                let mut bwd2_cache: Vec<Vec<f32>> = vec![vec![0.0; n_states]; n_hets];

                {
                    let init_bwd = 1.0f32 / n_states as f32;
                    let mut bwd0 = vec![init_bwd; n_states]; // Combined track
                    let mut bwd1 = vec![init_bwd; n_states]; // Track 1
                    let mut bwd2 = vec![init_bwd; n_states]; // Track 2
                    let mut het_rev_idx = n_hets;

                    for i in (0..n_hi_freq).rev() {
                        let allele1 = seq1[i];
                        let allele2 = seq2[i];
                        let is_het = allele1 != 255 && allele2 != 255 && allele1 != allele2;

                        // At het positions: STORE then COLLAPSE
                        if is_het && het_rev_idx > 0 && het_positions[het_rev_idx - 1] == i {
                            het_rev_idx -= 1;
                            bwd1_cache[het_rev_idx].copy_from_slice(&bwd1);
                            bwd2_cache[het_rev_idx].copy_from_slice(&bwd2);
                            // COLLAPSE: marginalize over phase
                            bwd1.copy_from_slice(&bwd0);
                            bwd2.copy_from_slice(&bwd0);
                        }

                        if i > 0 {
                            let p_recomb_i = stage1_p_recomb.get(i).copied().unwrap_or(0.0);
                            let shift = p_recomb_i / n_states as f32;
                            let stay = 1.0 - p_recomb_i;

                            let mut bwd0_sum = 0.0f32;
                            let mut bwd1_sum = 0.0f32;
                            let mut bwd2_sum = 0.0f32;

                            for k in 0..n_states {
                                let ref_al = get_ref_allele(i, k);

                                // Track 0: Combined emission
                                let emit0 = if is_het {
                                    if ref_al == allele1 || ref_al == allele2 || ref_al == 255 {
                                        p_no_err
                                    } else {
                                        p_err
                                    }
                                } else {
                                    let obs = if allele1 != 255 { allele1 } else { allele2 };
                                    if ref_al == obs || ref_al == 255 || obs == 255 {
                                        p_no_err
                                    } else {
                                        p_err
                                    }
                                };

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

                                bwd0[k] *= emit0;
                                bwd1[k] *= emit1;
                                bwd2[k] *= emit2;
                                bwd0_sum += bwd0[k];
                                bwd1_sum += bwd1[k];
                                bwd2_sum += bwd2[k];
                            }

                            let scale0 = stay / bwd0_sum.max(1e-30);
                            let scale1 = stay / bwd1_sum.max(1e-30);
                            let scale2 = stay / bwd2_sum.max(1e-30);

                            for k in 0..n_states {
                                bwd0[k] = scale0 * bwd0[k] + shift;
                                bwd1[k] = scale1 * bwd1[k] + shift;
                                bwd2[k] = scale2 * bwd2[k] + shift;
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
                        // Compute likelihood ratio for threshold check
                        let lr = if l_swap > l_keep {
                            (l_swap / l_keep.max(1e-300)) as f32
                        } else {
                            (l_keep / l_swap.max(1e-300)) as f32
                        };
                        decisions.push((i, should_swap, lr));

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

                // Compute swap mask by comparing final working sequences to originals.
                // This correctly captures the cumulative effect: when the HMM decides to swap
                // at het i, it swaps seq_working from i to end. A subsequent "keep" at het j
                // means "keep the current (already-swapped) orientation", so markers between
                // i and j remain swapped. Comparing to originals captures this.
                let swap_mask: Vec<bool> = (0..n_hi_freq)
                    .map(|i| seq1_working[i] != seq1[i])
                    .collect();

                // Extract just (position, LR) for het phased-marking threshold checks
                let het_lr_values: Vec<(usize, f32)> = decisions
                    .into_iter()
                    .map(|(idx, _, lr)| (idx, lr))
                    .collect();

                (swap_mask, het_lr_values)
            })
            .collect();

        // Apply phase decisions to SamplePhase
        let mut total_switches = 0;
        let mut total_phased = 0;

        // Determine if we're in burn-in (don't mark as phased during burn-in)
        let is_burnin = iteration < self.config.burnin;
        let lr_threshold = self.params.lr_threshold;

        for (s, (swap_mask, het_lr_values)) in phase_decisions.into_iter().enumerate() {
            let sp = &mut sample_phases[s];

            // Apply swaps using the mask (correctly handles cumulative swap propagation)
            for (hi_freq_idx, should_swap) in swap_mask.into_iter().enumerate() {
                if should_swap {
                    let m = hi_freq_to_orig[hi_freq_idx];
                    sp.swap_alleles(m);
                    total_switches += 1;
                }
            }

            // Mark hets as phased if LR exceeds threshold (independent of swap decision)
            if !is_burnin {
                for (hi_freq_idx, lr) in het_lr_values {
                    if lr >= lr_threshold {
                        let m = hi_freq_to_orig[hi_freq_idx];
                        sp.mark_phased(m);
                        total_phased += 1;
                    }
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
                let bytes: Vec<u8> = alleles.iter().copied().collect();
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
        maf: &[f32],
        rare_threshold: f32,
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

        // Process samples in parallel - collect results: Stage2Decision
        // Note: This is called after all iterations, so we use iteration=0 for deterministic state selection
        let phase_changes: Vec<Vec<Stage2Decision>> = sample_phases
            .par_iter()
            .enumerate()
            .map(|(s, sp)| {
                // Create deterministic RNG for this sample for random tie-breaking
                // Seed combines global seed + sample index + constant for Stage 2 distinction
                use rand::{Rng, SeedableRng};
                let sample_seed = (seed as u64)
                    .wrapping_add(s as u64)
                    .wrapping_add(0xDEAD_BEEF_CAFE_u64); // Stage 2 distinction constant
                let mut rng = rand::rngs::StdRng::seed_from_u64(sample_seed);

                // Build dynamic composite haplotypes using PhaseStates
                let mut phase_states = PhaseStates::new(self.params.n_states, n_stage1);
                let threaded_haps = phase_states.build_composite_haps(
                    s as u32,
                    &phase_ibs,
                    ibs2,
                    20,
                );
                let n_states = phase_states.n_states();

                // Extract Stage 1 alleles from SamplePhase
                let seq1: Vec<u8> = hi_freq_markers.iter().map(|&m| sp.allele1(m)).collect();
                let seq2: Vec<u8> = hi_freq_markers.iter().map(|&m| sp.allele2(m)).collect();
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

                // Pre-compute state->hap mapping for all Stage 1 markers
                // This is needed because ThreadedHaps uses cursor-based traversal
                // Pre-allocate all memory upfront to avoid clone() overhead in hot loop
                let state_haps_stage1: Vec<Vec<u32>> = {
                    let mut threaded_haps_mut = threaded_haps.clone();
                    let mut state_haps = vec![vec![0u32; n_states]; n_stage1];
                    for m in 0..n_stage1 {
                        threaded_haps_mut.materialize_haps(m, &mut state_haps[m]);
                    }
                    state_haps
                };

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

                let mut decisions: Vec<Stage2Decision> = Vec::new();

                // Function to impute a single allele for a haplotype
                // Matches Java Stage2Baum.imputeAllele()
                let impute_allele = |m: usize, probs: &[Vec<f32>], state_haps: &[Vec<u32>]| -> u8 {
                    // For biallelic sites (most common), use 2 alleles
                    // For multiallelic, use a reasonable max
                    let n_alleles = 4usize; // Conservative max for rare multiallelic sites
                    let mut al_probs = vec![0.0f32; n_alleles];

                    let mkr_a = stage2_phaser.prev_stage1_marker[m];
                    let mkr_b = (mkr_a + 1).min(n_stage1.saturating_sub(1));
                    let wt = stage2_phaser.prev_stage1_wt[m];

                    for (j, &hap) in state_haps[mkr_a].iter().enumerate() {
                        let prob_a = probs[mkr_a].get(j).copied().unwrap_or(0.0);
                        let prob_b = probs[mkr_b].get(j).copied().unwrap_or(0.0);
                        let prob = wt * prob_a + (1.0 - wt) * prob_b;

                        let b1 = get_allele(m, hap as usize);
                        let b2 = get_allele(m, (hap ^ 1) as usize);

                        if b1 != 255 && b2 != 255 {
                            if b1 == b2 || (hap as usize) >= n_haps {
                                // Homozygous or reference haplotype
                                if (b1 as usize) < n_alleles {
                                    al_probs[b1 as usize] += prob;
                                }
                            } else {
                                // Heterozygous target haplotype
                                let is_rare1 = maf[m] < rare_threshold && b1 > 0;
                                let is_rare2 = maf[m] < rare_threshold && b2 > 0;
                                if is_rare1 != is_rare2 {
                                    // One rare, one common: favor rare slightly
                                    if is_rare1 && (b1 as usize) < n_alleles {
                                        al_probs[b1 as usize] += 0.55 * prob;
                                    }
                                    if !is_rare1 && (b1 as usize) < n_alleles {
                                        al_probs[b1 as usize] += 0.45 * prob;
                                    }
                                    if is_rare2 && (b2 as usize) < n_alleles {
                                        al_probs[b2 as usize] += 0.55 * prob;
                                    }
                                    if !is_rare2 && (b2 as usize) < n_alleles {
                                        al_probs[b2 as usize] += 0.45 * prob;
                                    }
                                } else {
                                    // Both rare or both common: equal weight
                                    if (b1 as usize) < n_alleles {
                                        al_probs[b1 as usize] += 0.5 * prob;
                                    }
                                    if (b2 as usize) < n_alleles {
                                        al_probs[b2 as usize] += 0.5 * prob;
                                    }
                                }
                            }
                        }
                    }

                    // Return allele with highest probability
                    al_probs
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(idx, _)| idx as u8)
                        .unwrap_or(0)
                };

                let mut process_marker = |m: usize, rng: &mut rand::rngs::StdRng| {
                    let a1 = sp.allele1(m);
                    let a2 = sp.allele2(m);

                    // Handle missing genotypes by imputation
                    if sp.is_missing(m) || a1 == 255 || a2 == 255 {
                        let imp_a1 = impute_allele(m, &probs1, &state_haps_stage1);
                        let imp_a2 = impute_allele(m, &probs2, &state_haps_stage1);
                        decisions.push(Stage2Decision::Impute { marker: m, a1: imp_a1, a2: imp_a2 });
                        return;
                    }

                    // Skip if not unphased heterozygote
                    if !sp.is_unphased(m) {
                        return;
                    }

                    // Skip homozygotes
                    if a1 == a2 {
                        return;
                    }

                    // Determine if each allele is rare (low frequency)
                    // Following Java Stage2Baum.unscaledAlProbs: isLowFreq(m, allele)
                    // For biallelic markers: non-reference allele (a > 0) is rare when MAF < threshold
                    // Reference allele (0) is typically not rare
                    let marker_maf = maf[m];
                    let is_a1_rare = a1 > 0 && marker_maf < rare_threshold;
                    let is_a2_rare = a2 > 0 && marker_maf < rare_threshold;

                    // Compute interpolated allele probabilities for each haplotype
                    let al_probs1 = stage2_phaser.interpolated_allele_probs(
                        m, &probs1, &state_haps_stage1, &get_allele, a1, a2, is_a1_rare, is_a2_rare,
                    );
                    let al_probs2 = stage2_phaser.interpolated_allele_probs(
                        m, &probs2, &state_haps_stage1, &get_allele, a1, a2, is_a1_rare, is_a2_rare,
                    );

                    // p1 = P(hap1 has a1, hap2 has a2)
                    // p2 = P(hap1 has a2, hap2 has a1)
                    // al_probs[0] = P(allele a1), al_probs[1] = P(allele a2) - semantic indices, not allele values
                    let p1 = al_probs1[0] * al_probs2[1];
                    let p2 = al_probs1[1] * al_probs2[0];

                    // Record decision: (marker, should_swap, likelihood_ratio)
                    // When probabilities are equal (ambiguous phase), use random tie-breaking
                    // to prevent systematic bias. This matches Java Beagle behavior:
                    // switchAlleles = (p1<p2 || (p1==p2 && rand.nextBoolean()))
                    let should_swap = p2 > p1 || (p1 == p2 && rng.random_bool(0.5));
                    // Compute likelihood ratio for threshold check
                    let lr = if p2 > p1 {
                        (p2 / p1.max(1e-30)) as f32
                    } else {
                        (p1 / p2.max(1e-30)) as f32
                    };
                    decisions.push(Stage2Decision::Phase { marker: m, should_swap, lr });
                };

                // Process markers before first Stage 1 marker
                if !hi_freq_markers.is_empty() {
                    let first_hf = hi_freq_markers[0];
                    for m in 0..first_hf {
                        process_marker(m, &mut rng);
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
                        process_marker(m, &mut rng);
                    }
                }

                decisions
            })
            .collect::<Vec<_>>();

        // Apply phase changes and imputations to SamplePhase
        let mut total_switches = 0;
        let mut total_phased = 0;
        let mut total_imputed = 0;

        // Stage 2 runs after all iterations, so lr_threshold is typically 1.0
        // (all decisions pass). We still check for consistency with Stage 1.
        let lr_threshold = self.params.lr_threshold;

        for (s, decisions) in phase_changes.into_iter().enumerate() {
            let sp = &mut sample_phases[s];

            for decision in decisions {
                match decision {
                    Stage2Decision::Phase { marker: m, should_swap, lr } => {
                        // Double-check still unphased (should always be true)
                        if !sp.is_unphased(m) {
                            continue;
                        }

                        if should_swap {
                            sp.swap_haps(m, m + 1);
                            total_switches += 1;
                        }

                        // Only mark as phased if likelihood ratio exceeds threshold
                        // (Stage 2 runs after iterations, so threshold is typically 1.0)
                        if lr >= lr_threshold {
                            sp.mark_phased(m);
                            total_phased += 1;
                        }
                    }
                    Stage2Decision::Impute { marker: m, a1, a2 } => {
                        // Set imputed alleles for missing marker
                        sp.set_imputed(m, a1, a2);
                        total_imputed += 1;
                    }
                }
            }
        }

        eprintln!(
            "Stage 2: Applied {} phase switches, {} markers phased, {} markers imputed (HMM interpolation)",
            total_switches, total_phased, total_imputed
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

        for (k, p) in probs[m].iter_mut().enumerate().take(n_states) {
            *p = fwd[row_start + k] * bwd[row_start + k];
            sum += *p;
        }

        // Normalize
        if sum > 0.0 {
            for p in probs[m].iter_mut().take(n_states) {
                *p /= sum;
            }
        }
    }

    probs
}

/// Decision type for Stage 2 marker processing
#[derive(Debug, Clone)]
enum Stage2Decision {
    /// Phase an unphased heterozygote
    Phase { marker: usize, should_swap: bool, lr: f32 },
    /// Impute a missing genotype
    Impute { marker: usize, a1: u8, a2: u8 },
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
            prev_stage1_marker[..=first_hf].fill(0);

            // Fill between Stage 1 markers
            for j in 1..n_stage1 {
                let prev_hf = hi_freq_markers[j - 1];
                let curr_hf = hi_freq_markers[j];
                prev_stage1_marker[(prev_hf + 1)..=curr_hf].fill(j - 1);
            }

            // Fill after last Stage 1 marker
            let last_hf = hi_freq_markers[n_stage1 - 1];
            prev_stage1_marker[(last_hf + 1)..].fill(n_stage1 - 1);
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
                    for (m, wt) in prev_stage1_wt.iter_mut().enumerate().take(end).skip(start + 1) {
                        *wt = ((pos_b - gen_positions[m]) / d) as f32;
                    }
                } else {
                    // Zero distance, use equal weight
                    prev_stage1_wt[(start + 1)..end].fill(0.5);
                }
            }

            // Markers at and after last Stage 1 marker: wt = 1.0
            let last_hf = hi_freq_markers[n_stage1 - 1];
            prev_stage1_wt[last_hf..].fill(1.0);
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
    /// - Only rare alleles trigger the match logic (Java's isLowFreq check)
    /// - Use full probability (1.0), not the 0.55/0.45 heuristic (that's for imputation)
    ///
    /// # Arguments
    /// * `is_a1_rare` - Whether target allele a1 is low frequency
    /// * `is_a2_rare` - Whether target allele a2 is low frequency
    fn interpolated_allele_probs<F>(
        &self,
        marker: usize,
        state_probs: &[Vec<f32>],      // [stage1_marker][state]
        state_haps_stage1: &[Vec<u32>], // [stage1_marker][state] -> hap
        get_allele: &F,                 // Closure to get allele for any haplotype
        a1: u8,
        a2: u8,
        is_a1_rare: bool,
        is_a2_rare: bool,
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

        // Use state haps at flanking Stage 1 marker A (following Java Stage2Baum)
        let haps_at_mkr_a = &state_haps_stage1[mkr_a];
        let n_states = haps_at_mkr_a.len();

        for j in 0..n_states {
            let hap = haps_at_mkr_a[j] as usize;

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
                // Following Java Stage2Baum.unscaledAlProbs:
                // - match1 is true ONLY if a1 is rare AND matches a reference allele
                // - match2 is true ONLY if a2 is rare AND matches a reference allele
                // - Use FULL probability (1.0), NOT the 0.55/0.45 heuristic (that's for imputation)
                let match1 = is_a1_rare && (a1 == b1 || a1 == b2);
                let match2 = is_a2_rare && (a2 == b1 || a2 == b2);

                if match1 && !match2 {
                    // Only a1 matches (and is rare) - add full probability to a1
                    al_probs[0] += prob;
                } else if match2 && !match1 {
                    // Only a2 matches (and is rare) - add full probability to a2
                    al_probs[1] += prob;
                }
                // If both match or neither match (ambiguous), no contribution
                // This is consistent with Java Stage2Baum.unscaledAlProbs behavior
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
            profile: false,
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
            profile: false,
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