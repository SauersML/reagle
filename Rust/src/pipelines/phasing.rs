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
use rand::{Rng, SeedableRng};
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
use mini_mcmc::core::{MarkovChain, Trace};

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

const MOSAIC_BLOCK_SIZE: usize = 128;  // Smaller = less memory per sample

struct RefAlleleLookup<'a> {
    state_haps: &'a [Vec<u32>],
    n_target_haps: usize,
    ref_geno: &'a MutableGenotypes,
    reference_gt: Option<&'a GenotypeMatrix<Phased>>,
    alignment: Option<&'a MarkerAlignment>,
    marker_map: Option<&'a [usize]>,
}

impl<'a> RefAlleleLookup<'a> {
    fn orig_marker(&self, m: usize) -> usize {
        self.marker_map.map(|map| map[m]).unwrap_or(m)
    }

    fn allele(&self, m: usize, state: usize) -> u8 {
        let orig_m = self.orig_marker(m);
        let hap = self.state_haps[m][state] as usize;
        if hap < self.n_target_haps {
            self.ref_geno.get(orig_m, HapIdx::new(hap as u32))
        } else {
            let ref_h = (hap - self.n_target_haps) as u32;
            if let (Some(ref_gt), Some(align)) = (self.reference_gt, self.alignment) {
                if let Some(ref_m) = align.target_to_ref(orig_m) {
                    let ref_allele = ref_gt.allele(MarkerIdx::new(ref_m as u32), HapIdx::new(ref_h));
                    align.reverse_map_allele(orig_m, ref_allele)
                } else {
                    255
                }
            } else {
                255
            }
        }
    }
}

struct FwdCheckpoints {
    block_size: usize,
    n_blocks: usize,
    n_states: usize,
    data: Vec<f32>,
}

impl FwdCheckpoints {
    fn new(n_markers: usize, n_states: usize, block_size: usize) -> Self {
        let block_size = block_size.max(1).min(n_markers.max(1));
        let n_blocks = (n_markers + block_size - 1) / block_size;
        Self {
            block_size,
            n_blocks,
            n_states,
            data: vec![0.0f32; n_blocks * n_states],
        }
    }

    fn block_slice(&self, block_idx: usize) -> &[f32] {
        let start = block_idx * self.n_states;
        &self.data[start..start + self.n_states]
    }

    fn block_slice_mut(&mut self, block_idx: usize) -> &mut [f32] {
        let start = block_idx * self.n_states;
        &mut self.data[start..start + self.n_states]
    }
}

#[derive(Debug, Clone)]
struct MosaicTrace {
    mean_state: f64,
    switch_rate: f64,
    log_likelihood: f64,
}

impl Trace for MosaicTrace {
    fn trace(&self) -> Vec<f64> {
        vec![self.mean_state, self.switch_rate, self.log_likelihood]
    }
}

struct MosaicChain<'a> {
    rng: rand::rngs::SmallRng,
    n_markers: usize,
    n_states: usize,
    p_recomb: &'a [f32],
    seq1: &'a [u8],
    seq2: &'a [u8],
    conf: &'a [f32],
    lookup: &'a RefAlleleLookup<'a>,
    combined_checkpoints: Arc<FwdCheckpoints>,
    hap2_checkpoints: FwdCheckpoints,
    hap2_allele: Vec<u8>,
    hap2_use_combined: Vec<bool>,
    path1: Vec<u32>,  // u32 saves 50% memory vs usize
    path2: Vec<u32>,
    fwd_block: Vec<f32>,
    trace: MosaicTrace,
    p_no_err: f32,
    p_err: f32,
}

impl<'a> MosaicChain<'a> {
    fn new(
        seed: u64,
        n_markers: usize,
        n_states: usize,
        p_recomb: &'a [f32],
        seq1: &'a [u8],
        seq2: &'a [u8],
        conf: &'a [f32],
        lookup: &'a RefAlleleLookup<'a>,
        combined_checkpoints: Arc<FwdCheckpoints>,
        p_no_err: f32,
        p_err: f32,
    ) -> Self {
        let out = Self {
            rng: rand::rngs::SmallRng::seed_from_u64(seed),
            n_markers,
            n_states,
            p_recomb,
            seq1,
            seq2,
            conf,
            lookup,
            combined_checkpoints,
            hap2_checkpoints: FwdCheckpoints::new(n_markers, n_states, MOSAIC_BLOCK_SIZE),
            hap2_allele: vec![255u8; n_markers],
            hap2_use_combined: vec![true; n_markers],
            path1: vec![0u32; n_markers],
            path2: vec![0u32; n_markers],
            fwd_block: vec![0.0f32; n_states * MOSAIC_BLOCK_SIZE],
            trace: MosaicTrace {
                mean_state: 0.0,
                switch_rate: 0.0,
                log_likelihood: 0.0,
            },
            p_no_err,
            p_err,
        };
        out
    }

    fn paths(&self) -> (&[u32], &[u32]) {
        (&self.path1, &self.path2)
    }

    fn update_trace(&mut self) {
        if self.n_markers == 0 {
            self.trace.mean_state = 0.0;
            self.trace.switch_rate = 0.0;
            self.trace.log_likelihood = 0.0;
            return;
        }

        let mut sum = 0.0f64;
        let mut switches = 0usize;
        let mut logp = 0.0f64;
        for i in 0..self.n_markers {
            let s1 = self.path1[i] as f64;
            let s2 = self.path2[i] as f64;
            sum += s1 + s2;
            if i > 0 {
                if self.path1[i] != self.path1[i - 1] {
                    switches += 1;
                }
                if self.path2[i] != self.path2[i - 1] {
                    switches += 1;
                }
            }
            logp += (self.path1[i] as f64 + 1.0).ln();
        }

        let denom = (self.n_markers * 2) as f64;
        self.trace.mean_state = sum / denom;
        self.trace.switch_rate = if self.n_markers > 1 {
            switches as f64 / ((self.n_markers - 1) as f64 * 2.0)
        } else {
            0.0
        };
        self.trace.log_likelihood = logp;
    }

    fn build_hap2_inputs(&mut self) {
        for m in 0..self.n_markers {
            let a1 = self.seq1[m];
            let a2 = self.seq2[m];
            if a1 == 255 && a2 == 255 {
                self.hap2_use_combined[m] = true;
                self.hap2_allele[m] = 255;
                continue;
            }
            if a1 == a2 {
                self.hap2_use_combined[m] = false;
                self.hap2_allele[m] = a1;
                continue;
            }

            let ref_al = self.lookup.allele(m, self.path1[m] as usize);
            if ref_al == a1 {
                self.hap2_use_combined[m] = false;
                self.hap2_allele[m] = a2;
            } else if ref_al == a2 {
                self.hap2_use_combined[m] = false;
                self.hap2_allele[m] = a1;
            } else {
                self.hap2_use_combined[m] = true;
                self.hap2_allele[m] = 255;
            }
        }
    }
}

impl MarkovChain<MosaicTrace> for MosaicChain<'_> {
    fn step(&mut self) -> &MosaicTrace {
        sample_path_from_checkpoints(
            &mut self.path1,
            &self.combined_checkpoints,
            self.n_markers,
            self.n_states,
            self.p_recomb,
            self.seq1,
            self.seq2,
            self.conf,
            &self.hap2_allele,
            &self.hap2_use_combined,
            self.lookup,
            self.p_no_err,
            self.p_err,
            &mut self.rng,
            &mut self.fwd_block,
            EmissionMode::Combined,
        );

        self.build_hap2_inputs();
        build_fwd_checkpoints(
            &mut self.hap2_checkpoints,
            self.n_markers,
            self.n_states,
            self.p_recomb,
            self.seq1,
            self.seq2,
            self.conf,
            &self.hap2_allele,
            &self.hap2_use_combined,
            self.lookup,
            self.p_no_err,
            self.p_err,
            EmissionMode::Hap,
        );
        sample_path_from_checkpoints(
            &mut self.path2,
            &self.hap2_checkpoints,
            self.n_markers,
            self.n_states,
            self.p_recomb,
            self.seq1,
            self.seq2,
            self.conf,
            &self.hap2_allele,
            &self.hap2_use_combined,
            self.lookup,
            self.p_no_err,
            self.p_err,
            &mut self.rng,
            &mut self.fwd_block,
            EmissionMode::Hap,
        );
        self.update_trace();
        &self.trace
    }

    fn current_state(&self) -> &MosaicTrace {
        &self.trace
    }
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

        // Create SamplePhase instances to track phase state (with confidence)
        let confidence_by_sample = build_sample_confidence(&target_gt);
        let mut sample_phases = self.create_sample_phases(&geno, &confidence_by_sample);

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
        let n_ref_haps = self.reference_gt.as_ref().map(|r| r.n_haplotypes()).unwrap_or(0);
        let n_total_haps = n_haps + n_ref_haps;

        if n_markers == 0 {
            return Ok(target_gt.clone().into_phased());
        }

        self.params = ModelParams::for_phasing(n_total_haps, self.config.ne, self.config.err);
        self.params
            .set_n_states(self.config.phase_states.min(n_total_haps.saturating_sub(2)));

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
        let n_ref_haps = self.reference_gt.as_ref().map(|r| r.n_haplotypes()).unwrap_or(0);
        let n_total_haps = n_haps + n_ref_haps;

        if n_markers == 0 {
            return Ok(target_gt.clone().into_phased());
        }

        self.params = ModelParams::for_phasing(n_total_haps, self.config.ne, self.config.err);
        self.params
            .set_n_states(self.config.phase_states.min(n_total_haps.saturating_sub(2)));

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
        let confidence_by_sample = build_sample_confidence(&target_gt);
        let mut sample_phases =
            self.create_sample_phases_with_overlap(&geno, &missing_mask, overlap_markers, &confidence_by_sample);

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
        confidence_by_sample: &[Vec<f32>],
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

                let conf = &confidence_by_sample[s];
                SamplePhase::new(s as u32, n_markers, &alleles1, &alleles2, conf, &unphased, &missing)
            })
            .collect()
    }

    /// Create SamplePhase instances for all samples
    ///
    /// This initializes phase tracking state from the current genotype data.
    fn create_sample_phases(
        &self,
        geno: &MutableGenotypes,
        confidence_by_sample: &[Vec<f32>],
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

                let conf = &confidence_by_sample[s];
                SamplePhase::new(s as u32, n_markers, &alleles1, &alleles2, conf, &unphased, &missing)
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
        BidirectionalPhaseIbs::build(alleles_by_marker, n_haps, n_markers)
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
            alleles_by_marker,
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
        BidirectionalPhaseIbs::build(alleles_by_marker, n_total_haps, n_markers)
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
            alleles_by_marker,
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
            let sample_seed = (self.config.seed as u64)
                .wrapping_add(s as u64)
                .wrapping_add(0xA5A5_5A5A_D00Du64);

            // Build dynamic composite haplotypes using PhaseStates
            // This iterates through all markers and builds mosaic haplotypes
            // that provide local IBS matches everywhere, not just at midpoint.
            let min_segment_len = PhaseStates::min_segment_len(n_markers, gen_dists);
            let mut phase_states = PhaseStates::new(self.params.n_states, n_markers, min_segment_len);
            let n_candidates = self.params.n_states.min(n_total_haps).max(20);
            let threaded_haps = phase_states.build_composite_haps(
                s as u32,
                &phase_ibs,
                ibs2,
                n_candidates,
            );
            let n_states = phase_states.n_states();

            // 2. Extract current alleles for H1 and H2
            let seq1 = ref_geno.haplotype(hap1);
            let seq2 = ref_geno.haplotype(hap2);
            let sample_conf: Vec<f32> = (0..n_markers)
                .map(|m| target_gt.sample_confidence_f32(MarkerIdx::new(m as u32), s))
                .collect();

            // 3. Run HMM with per-heterozygote swap probabilities
            // Following Java PhaseBaum2.java: interleave phase decisions in the forward pass.
            //
            // Key Algorithm (3-Track HMM):
            // 1. Run backward pass for BOTH haplotypes first, storing backward values
            // 2. Run forward pass marker-by-marker for BOTH haplotypes
            // 3. At each het, compute swap probability using fwd and stored bwd
            // 4. After the forward pass, sample a swap mask via MCMC
            // 5. Apply the sampled mask to update phase
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

            let lookup = RefAlleleLookup {
                state_haps: &state_haps,
                n_target_haps: n_haps,
                ref_geno: &ref_geno,
                reference_gt: self.reference_gt.as_deref(),
                alignment: self.alignment.as_ref(),
                marker_map: None,
            };

            let (swap_bits, swap_lr) = sample_swap_bits_mosaic(
                n_markers,
                n_states,
                p_recomb,
                &seq1,
                &seq2,
                &sample_conf,
                &lookup,
                &het_positions,
                sample_seed,
                self.config.mcmc_burnin,
                p_no_err,
                p_err,
            );
            assert!(swap_lr.len() <= n_markers);
            let mut swapped = false;
            let mut swap_idx = 0usize;
            for m in 0..n_markers {
                if swap_idx < het_positions.len() && het_positions[swap_idx] == m {
                    swapped = swap_bits.get(swap_idx).copied().unwrap_or(0) == 1;
                    swap_idx += 1;
                }
                if swapped {
                    mask.set(m, true);
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

        // Compute total haplotype count (target + reference)
        let n_ref_haps = self.reference_gt.as_ref().map(|r| r.n_haplotypes()).unwrap_or(0);
        let n_total_haps = n_haps + n_ref_haps;

        let ref_geno = geno.clone();
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

        // Build bidirectional PBWT for better state selection around recombination hotspots
        let phase_ibs = if let (Some(ref_gt), Some(alignment)) = (&self.reference_gt, &self.alignment) {
            self.build_bidirectional_pbwt_combined(
                |m, h| {
                    if h < n_haps {
                        ref_geno.get(m, HapIdx::new(h as u32))
                    } else {
                        let ref_h = h - n_haps;
                        if let Some(ref_m) = alignment.target_to_ref(m) {
                            let ref_allele = ref_gt.allele(MarkerIdx::new(ref_m as u32), HapIdx::new(ref_h as u32));
                            alignment.reverse_map_allele(m, ref_allele)
                        } else {
                            255
                        }
                    }
                },
                n_markers,
                n_total_haps,
            )
        } else {
            self.build_bidirectional_pbwt(&ref_geno, n_markers, n_haps)
        };

        // Collect swap masks per sample
        let n_samples = n_haps / 2;
        let swap_masks: Vec<BitBox<u64, Lsb0>> = sample_phases
            .par_iter()
            .enumerate()
            .map(|(s, sp)| {
                // Build dynamic composite haplotypes using PhaseStates
                let min_segment_len = PhaseStates::min_segment_len(n_markers, gen_dists);
                let mut phase_states = PhaseStates::new(self.params.n_states, n_markers, min_segment_len);
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
                let sample_conf: Vec<f32> = (0..n_markers).map(|m| sp.confidence(m)).collect();
                let sample_seed = (self.config.seed as u64)
                    .wrapping_add(s as u64)
                    .wrapping_add(0xBEEF_CAFE_55AAu64);

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
                let changeable_positions: Vec<usize> = het_positions[first_changeable_het..].to_vec();
                if changeable_positions.is_empty() {
                    return bitbox![u64, Lsb0; 0; n_markers];
                }

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

                let lookup = RefAlleleLookup {
                    state_haps: &state_haps,
                    n_target_haps: n_haps,
                    ref_geno: &ref_geno,
                    reference_gt: self.reference_gt.as_deref(),
                    alignment: self.alignment.as_ref(),
                    marker_map: None,
                };

                let (swap_bits, swap_lr) = sample_swap_bits_mosaic(
                    n_markers,
                    n_states,
                    p_recomb,
                    &seq1,
                    &seq2,
                    &sample_conf,
                    &lookup,
                    &changeable_positions,
                    sample_seed,
                    self.config.mcmc_burnin,
                    p_no_err,
                    p_err,
                );
                assert!(swap_lr.len() <= changeable_positions.len());
                let mut swap_mask = bitbox![u64, Lsb0; 0; n_markers];
                let mut current_phase = 0u8;
                let mut phase_idx = 0usize;
                for m in overlap_markers..n_markers {
                    if phase_idx < changeable_positions.len() && changeable_positions[phase_idx] == m {
                        current_phase = swap_bits.get(phase_idx).copied().unwrap_or(0);
                        phase_idx += 1;
                    }
                    if current_phase == 1 {
                        swap_mask.set(m, true);
                    }
                }

                swap_mask
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
        //   - swap_mask[i] = true if the sampled phase orientation at marker i is swapped
        //   - het_lr_values = (hi_freq_idx, lr) for each het, used for phased marking threshold
        type PhaseDecision = (Vec<bool>, Vec<(usize, f32)>);
        let phase_decisions: Vec<PhaseDecision> = sample_phases
            .par_iter()
            .enumerate()
            .map(|(s, sp)| {
                let n_hi_freq = hi_freq_to_orig.len();

                // Build dynamic composite haplotypes using PhaseStates
                let min_segment_len = PhaseStates::min_segment_len(n_hi_freq, stage1_gen_dists);
                let mut phase_states = PhaseStates::new(self.params.n_states, n_hi_freq, min_segment_len);
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
                let sample_conf: Vec<f32> = hi_freq_to_orig.iter().map(|&m| sp.confidence(m)).collect();
                let sample_seed = (self.config.seed as u64)
                    .wrapping_add(s as u64)
                    .wrapping_add((iteration as u64) << 32)
                    .wrapping_add(0xFEED_FACE_1234u64);

                // Collect EM statistics if requested
                if let Some(atomic) = atomic_estimates {
                    let hmm = BeagleHmm::new(subset_view, &self.params, n_states, stage1_p_recomb.to_vec());
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

                let lookup = RefAlleleLookup {
                    state_haps: &state_haps,
                    n_target_haps: n_haps,
                    ref_geno: &ref_geno,
                    reference_gt: self.reference_gt.as_deref(),
                    alignment: self.alignment.as_ref(),
                    marker_map: Some(hi_freq_to_orig),
                };

                let (swap_bits, swap_lr) = sample_swap_bits_mosaic(
                    n_hi_freq,
                    n_states,
                    stage1_p_recomb,
                    &seq1,
                    &seq2,
                    &sample_conf,
                    &lookup,
                    &het_positions,
                    sample_seed,
                    self.config.mcmc_burnin,
                    p_no_err,
                    p_err,
                );

                let mut swap_mask = vec![false; n_hi_freq];
                let mut current_phase = 0u8;
                let mut phase_idx = 0usize;
                for i in 0..n_hi_freq {
                    if phase_idx < het_positions.len() && het_positions[phase_idx] == i {
                        current_phase = swap_bits.get(phase_idx).copied().unwrap_or(0);
                        phase_idx += 1;
                    }
                    swap_mask[i] = current_phase == 1;
                }

                let het_lr_values: Vec<(usize, f32)> = het_positions
                    .iter()
                    .copied()
                    .zip(swap_lr.into_iter())
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
                let bytes: Vec<u8> = alleles.to_vec();
                GenotypeColumn::from_alleles(&bytes, 2)
            })
            .collect();

        // Preserve confidence scores from original matrix
        if let Some(confidence) = original.confidence_clone() {
            GenotypeMatrix::new_phased_with_confidence(markers, columns, samples, confidence)
        } else {
            GenotypeMatrix::new_phased(markers, columns, samples)
        }
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

        let stage1_gen_dists: Vec<f64> = if hi_freq_markers.len() > 1 {
            hi_freq_markers
                .windows(2)
                .map(|w| gen_positions[w[1]] - gen_positions[w[0]])
                .collect()
        } else {
            Vec::new()
        };

        // Compute total haplotype count (target + reference)
        let n_ref_haps = self.reference_gt.as_ref().map(|r| r.n_haplotypes()).unwrap_or(0);
        let n_total_haps = n_haps + n_ref_haps;

        // Build Stage 2 interpolation mappings
        let stage2_phaser = Stage2Phaser::new(hi_freq_markers, gen_positions, n_markers);

        // Clone current genotypes to use as a frozen reference panel
        let ref_geno = geno.clone();

        let rare_markers: Vec<usize> = (0..n_markers)
            .filter(|&m| maf[m] < rare_threshold && maf[m] > 0.0)
            .collect();

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

        let get_allele_global = |marker: usize, hap: usize| -> u8 {
            if hap < n_haps {
                ref_geno.get(marker, HapIdx::new(hap as u32))
            } else {
                let ref_h = hap - n_haps;
                if let (Some(ref_gt), Some(alignment)) = (&self.reference_gt, &self.alignment) {
                    if let Some(ref_m) = alignment.target_to_ref(marker) {
                        let ref_allele = ref_gt.allele(MarkerIdx::new(ref_m as u32), HapIdx::new(ref_h as u32));
                        alignment.reverse_map_allele(marker, ref_allele)
                    } else {
                        255
                    }
                } else {
                    255
                }
            }
        };

        let mut carrier_haps: Vec<Vec<u32>> = vec![Vec::new(); n_markers];
        for &m in &rare_markers {
            let mut carriers = Vec::new();
            for h in 0..n_total_haps {
                let allele = get_allele_global(m, h);
                if allele > 0 && allele != 255 {
                    carriers.push(h as u32);
                }
            }
            carrier_haps[m] = carriers;
        }

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
                let min_segment_len = PhaseStates::min_segment_len(n_stage1, &stage1_gen_dists);
                let mut phase_states = PhaseStates::new(self.params.n_states, n_stage1, min_segment_len);
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
                let seq_conf: Vec<f32> = hi_freq_markers.iter().map(|&m| sp.confidence(m)).collect();
                let hmm = BeagleHmm::new(subset_view, &self.params, n_states, stage1_p_recomb.to_vec());

                let mut fwd1 = Vec::new();
                let mut bwd1 = Vec::new();
                hmm.forward_backward_raw(&seq1, Some(&seq_conf), &threaded_haps, &mut fwd1, &mut bwd1);

                let mut fwd2 = Vec::new();
                let mut bwd2 = Vec::new();
                hmm.forward_backward_raw(&seq2, Some(&seq_conf), &threaded_haps, &mut fwd2, &mut bwd2);

                // Compute posterior state probabilities at each Stage 1 marker
                let probs1 = compute_state_posteriors(&fwd1, &bwd1, n_stage1, n_states);
                let probs2 = compute_state_posteriors(&fwd2, &bwd2, n_stage1, n_states);

                // Lazy cache for state->hap mapping - O(1) indexing with Option<Vec>
                let mut hap_cache: Vec<Option<Vec<u32>>> = vec![None; n_stage1];
                let mut threaded_haps_mut = threaded_haps.clone();

                macro_rules! get_haps {
                    ($marker:expr) => {{
                        let m = $marker;
                        if hap_cache[m].is_none() {
                            let mut haps = vec![0u32; n_states];
                            threaded_haps_mut.materialize_haps(m, &mut haps);
                            hap_cache[m] = Some(haps);
                        }
                        hap_cache[m].as_ref().unwrap()
                    }};
                }

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

                // Inline helper macro for imputing a single allele
                // Matches Java Stage2Baum.imputeAllele()
                macro_rules! impute_allele {
                    ($m:expr, $probs:expr) => {{
                        let m = $m;
                        let probs = $probs;
                        let n_alleles = 4usize;
                        let mut al_probs = [0.0f32; 4];

                        let mkr_a = stage2_phaser.prev_stage1_marker[m];
                        let mkr_b = (mkr_a + 1).min(n_stage1.saturating_sub(1));
                        let wt = stage2_phaser.prev_stage1_wt[m];
                        let state_haps = get_haps!(mkr_a);

                        for (j, &hap) in state_haps.iter().enumerate() {
                            let prob_a = probs[mkr_a].get(j).copied().unwrap_or(0.0);
                            let prob_b = probs[mkr_b].get(j).copied().unwrap_or(0.0);
                            let prob = wt * prob_a + (1.0 - wt) * prob_b;

                            let b1 = get_allele(m, hap as usize);
                            let b2 = get_allele(m, (hap ^ 1) as usize);

                            if b1 != 255 && b2 != 255 {
                                if b1 == b2 || (hap as usize) >= n_haps {
                                    if (b1 as usize) < n_alleles {
                                        al_probs[b1 as usize] += prob;
                                    }
                                } else {
                                    let is_rare1 = maf[m] < rare_threshold && b1 > 0;
                                    let is_rare2 = maf[m] < rare_threshold && b2 > 0;
                                    if is_rare1 != is_rare2 {
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

                        al_probs
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                            .map(|(idx, _)| idx as u8)
                            .unwrap_or(0)
                    }};
                }

                // Inline helper macro for carrier score calculation
                macro_rules! carrier_score {
                    ($m:expr, $probs:expr, $carrier_set:expr) => {{
                        let m = $m;
                        let probs = $probs;
                        let carrier_set = $carrier_set;
                        let mkr_a = stage2_phaser.prev_stage1_marker[m];
                        let mkr_b = (mkr_a + 1).min(n_stage1.saturating_sub(1));
                        let wt = stage2_phaser.prev_stage1_wt[m];
                        let state_haps = get_haps!(mkr_a);
                        let mut score = 0.0f32;
                        for (j, &hap) in state_haps.iter().enumerate() {
                            let prob_a = probs[mkr_a].get(j).copied().unwrap_or(0.0);
                            let prob_b = probs[mkr_b].get(j).copied().unwrap_or(0.0);
                            let prob = wt * prob_a + (1.0 - wt) * prob_b;
                            if carrier_set.contains(&hap) {
                                score += prob;
                            }
                        }
                        score
                    }};
                }

                for &m in &rare_markers {
                    let a1 = sp.allele1(m);
                    let a2 = sp.allele2(m);

                    // Handle missing genotypes by imputation
                    if sp.is_missing(m) || a1 == 255 || a2 == 255 {
                        let imp_a1 = impute_allele!(m, &probs1);
                        let imp_a2 = impute_allele!(m, &probs2);
                        decisions.push(Stage2Decision::Impute { marker: m, a1: imp_a1, a2: imp_a2 });
                        continue;
                    }

                    // Skip if not unphased heterozygote
                    if !sp.is_unphased(m) {
                        continue;
                    }

                    // Skip homozygotes
                    if a1 == a2 {
                        continue;
                    }

                    let marker_maf = maf[m];
                    let is_rare_marker = marker_maf < rare_threshold;
                    let carriers = &carrier_haps[m];

                    if is_rare_marker && !carriers.is_empty() {
                        let carrier_set: std::collections::HashSet<u32> =
                            carriers.iter().copied().collect();
                        let score1 = carrier_score!(m, &probs1, &carrier_set);
                        let score2 = carrier_score!(m, &probs2, &carrier_set);

                        if carriers.len() == 1 || (score1 == 0.0 && score2 == 0.0) {
                            let stage1_idx = stage2_phaser.prev_stage1_marker[m];
                            let hap1_idx = (s * 2) as u32;
                            let hap2_idx = (s * 2 + 1) as u32;
                            let span1 = phase_ibs.best_match_span(hap1_idx, stage1_idx);
                            let span2 = phase_ibs.best_match_span(hap2_idx, stage1_idx);
                            let alt_on_hap1 = a1 > 0 && a1 != 255;
                            let alt_on_hap2 = a2 > 0 && a2 != 255;
                            if alt_on_hap1 ^ alt_on_hap2 {
                                let shorter_is_hap1 = span1 < span2;
                                let should_swap = if alt_on_hap1 {
                                    !shorter_is_hap1
                                } else {
                                    shorter_is_hap1
                                };
                                decisions.push(Stage2Decision::Phase { marker: m, should_swap, lr: 1.0 });
                                continue;
                            }
                        }

                        let should_swap = score2 > score1 || (score2 == score1 && rng.random_bool(0.5));
                        let lr = if score2 > score1 {
                            (score2 / score1.max(1e-30)) as f32
                        } else {
                            (score1 / score2.max(1e-30)) as f32
                        };
                        decisions.push(Stage2Decision::Phase { marker: m, should_swap, lr });
                        continue;
                    }

                    // Fallback to interpolated allele probabilities for non-rare markers
                    let is_a1_rare = a1 > 0 && marker_maf < rare_threshold;
                    let is_a2_rare = a2 > 0 && marker_maf < rare_threshold;

                    let mkr_a = stage2_phaser.prev_stage1_marker[m];
                    let state_haps_for_interp = get_haps!(mkr_a);
                    let al_probs1 = stage2_phaser.interpolated_allele_probs(
                        m, &probs1, state_haps_for_interp, &get_allele, a1, a2, is_a1_rare, is_a2_rare,
                    );
                    let al_probs2 = stage2_phaser.interpolated_allele_probs(
                        m, &probs2, state_haps_for_interp, &get_allele, a1, a2, is_a1_rare, is_a2_rare,
                    );

                    let p1 = al_probs1[0] * al_probs2[1];
                    let p2 = al_probs1[1] * al_probs2[0];

                    let should_swap = p2 > p1 || (p1 == p2 && rng.random_bool(0.5));
                    let lr = if p2 > p1 {
                        (p2 / p1.max(1e-30)) as f32
                    } else {
                        (p1 / p2.max(1e-30)) as f32
                    };
                    decisions.push(Stage2Decision::Phase { marker: m, should_swap, lr });
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

fn build_sample_confidence(target_gt: &GenotypeMatrix) -> Vec<Vec<f32>> {
    let n_samples = target_gt.n_samples();
    let n_markers = target_gt.n_markers();

    (0..n_samples)
        .map(|s| {
            (0..n_markers)
                .map(|m| target_gt.sample_confidence_f32(MarkerIdx::new(m as u32), s))
                .collect()
        })
        .collect()
}

#[inline(always)]
fn emit_prob(ref_al: u8, targ_al: u8, conf: f32, p_no_err: f32, p_err: f32) -> f32 {
    let base = if ref_al == targ_al || ref_al == 255 || targ_al == 255 {
        p_no_err
    } else {
        p_err
    };
    base * conf + 0.5 * (1.0 - conf)
}

/// Emission mode for combined diploid genotype
#[derive(Clone, Copy)]
enum CombinedEmitMode {
    AllMissing,           // a1==255 && a2==255: always p_no_err
    Het { a1: u8, a2: u8 }, // a1!=a2: match if ref in {a1,a2,255}
    HomOrHemi { obs: u8 }, // hom or one missing: match if ref==obs or missing
}

#[inline]
fn classify_combined(a1: u8, a2: u8) -> CombinedEmitMode {
    if a1 == 255 && a2 == 255 {
        CombinedEmitMode::AllMissing
    } else if a1 != 255 && a2 != 255 && a1 != a2 {
        CombinedEmitMode::Het { a1, a2 }
    } else {
        let obs = if a1 != 255 { a1 } else { a2 };
        CombinedEmitMode::HomOrHemi { obs }
    }
}

/// Fast emit - assumes conf is already clamped to [0,1]
#[inline(always)]
fn emit_combined_fast(ref_al: u8, mode: CombinedEmitMode, conf: f32, p_no_err: f32, p_err: f32) -> f32 {
    let base = match mode {
        CombinedEmitMode::AllMissing => p_no_err,
        CombinedEmitMode::Het { a1, a2 } => {
            if ref_al == a1 || ref_al == a2 || ref_al == 255 { p_no_err } else { p_err }
        }
        CombinedEmitMode::HomOrHemi { obs } => {
            if ref_al == obs || ref_al == 255 || obs == 255 { p_no_err } else { p_err }
        }
    };
    base * conf + 0.5 * (1.0 - conf)
}

#[inline]
fn emit_combined(ref_al: u8, a1: u8, a2: u8, conf: f32, p_no_err: f32, p_err: f32) -> f32 {
    emit_combined_fast(ref_al, classify_combined(a1, a2), conf, p_no_err, p_err)
}

#[derive(Clone, Copy, Debug)]
enum EmissionMode {
    Combined,
    Hap,
}

fn build_fwd_checkpoints(
    checkpoints: &mut FwdCheckpoints,
    n_markers: usize,
    n_states: usize,
    p_recomb: &[f32],
    seq1: &[u8],
    seq2: &[u8],
    conf: &[f32],
    hap2_allele: &[u8],
    hap2_use_combined: &[bool],
    lookup: &RefAlleleLookup<'_>,
    p_no_err: f32,
    p_err: f32,
    mode: EmissionMode,
) {
    if n_markers == 0 || n_states == 0 {
        return;
    }

    let init = 1.0f32 / n_states as f32;
    let mut fwd = vec![init; n_states];
    let mut fwd_prior = vec![0.0f32; n_states];
    let mut ref_alleles = vec![0u8; n_states];  // Pre-compute alleles per marker
    let mut fwd_sum = 1.0f32;

    for m in 0..n_markers {
        if m > 0 {
            let r = p_recomb.get(m).copied().unwrap_or(0.0);
            let shift = r / n_states as f32;
            let scale = (1.0 - r) / fwd_sum.max(1e-30);
            for k in 0..n_states {
                fwd_prior[k] = scale * fwd[k] + shift;
            }
        } else {
            fwd_prior.fill(init);
        }

        let a1 = seq1[m];
        let a2 = seq2[m];
        let conf_m = conf[m].clamp(0.0, 1.0);

        // Batch lookup: get all ref alleles for this marker at once
        for k in 0..n_states {
            ref_alleles[k] = lookup.allele(m, k);
        }

        fwd_sum = 0.0;
        let use_combined = matches!(mode, EmissionMode::Combined) || hap2_use_combined[m];

        if use_combined {
            // Classify once per marker, then fast emit per state
            let emit_mode = classify_combined(a1, a2);
            for k in 0..n_states {
                let emit = emit_combined_fast(ref_alleles[k], emit_mode, conf_m, p_no_err, p_err);
                fwd[k] = fwd_prior[k] * emit;
                fwd_sum += fwd[k];
            }
        } else {
            let h2_al = hap2_allele[m];
            for k in 0..n_states {
                let emit = emit_prob(ref_alleles[k], h2_al, conf_m, p_no_err, p_err);
                fwd[k] = fwd_prior[k] * emit;
                fwd_sum += fwd[k];
            }
        }
        fwd_sum = fwd_sum.max(1e-30);

        if m % checkpoints.block_size == 0 {
            let block_idx = m / checkpoints.block_size;
            let dst = checkpoints.block_slice_mut(block_idx);
            dst.copy_from_slice(&fwd);
        }
    }
}

fn sample_from_weights(weights: &[f32], rng: &mut rand::rngs::SmallRng) -> usize {
    let total: f32 = weights.iter().sum();
    if total <= 0.0 {
        let idx = rng.random::<u32>() as usize % weights.len().max(1);
        return idx.min(weights.len().saturating_sub(1));
    }

    let mut threshold = rng.random::<f32>() * total;
    for (i, w) in weights.iter().enumerate() {
        threshold -= *w;
        if threshold <= 0.0 {
            return i;
        }
    }
    weights.len().saturating_sub(1)
}

fn sample_path_from_checkpoints(
    path: &mut [u32],
    checkpoints: &FwdCheckpoints,
    n_markers: usize,
    n_states: usize,
    p_recomb: &[f32],
    seq1: &[u8],
    seq2: &[u8],
    conf: &[f32],
    hap2_allele: &[u8],
    hap2_use_combined: &[bool],
    lookup: &RefAlleleLookup<'_>,
    p_no_err: f32,
    p_err: f32,
    rng: &mut rand::rngs::SmallRng,
    fwd_block: &mut [f32],
    mode: EmissionMode,
) {
    if n_markers == 0 || n_states == 0 {
        return;
    }

    let block_size = checkpoints.block_size;
    let n_blocks = checkpoints.n_blocks;
    let mut current_state: Option<usize> = None;

    let mut weights = vec![0.0f32; n_states];

    for block_idx in (0..n_blocks).rev() {
        let start = block_idx * block_size;
        let end = (start + block_size).min(n_markers);
        let block_len = end - start;
        let row_stride = n_states;
        let buf_len = block_len * row_stride;
        let fwd_buf = &mut fwd_block[..buf_len];

        // Seed forward values at block start from checkpoint.
        let seed = checkpoints.block_slice(block_idx);
        fwd_buf[..row_stride].copy_from_slice(seed);
        let mut prev_sum: f32 = seed.iter().sum();
        prev_sum = prev_sum.max(1e-30);

        for m in (start + 1)..end {
            let r = p_recomb.get(m).copied().unwrap_or(0.0);
            let shift = r / n_states as f32;
            let scale = (1.0 - r) / prev_sum;

            let a1 = seq1[m];
            let a2 = seq2[m];
            let conf_m = conf[m];
            let row_idx = (m - start) * row_stride;
            let (prev_part, curr_part) = fwd_buf.split_at_mut(row_idx);
            let prev_row = &prev_part[row_idx - row_stride..];

            let mut row_sum = 0.0f32;
            for k in 0..n_states {
                let prior = scale * prev_row[k] + shift;
                let ref_al = lookup.allele(m, k);
                let emit = match mode {
                    EmissionMode::Combined => emit_combined(ref_al, a1, a2, conf_m, p_no_err, p_err),
                    EmissionMode::Hap => {
                        if hap2_use_combined[m] {
                            emit_combined(ref_al, a1, a2, conf_m, p_no_err, p_err)
                        } else {
                            emit_prob(ref_al, hap2_allele[m], conf_m, p_no_err, p_err)
                        }
                    }
                };
                curr_part[k] = prior * emit;
                row_sum += curr_part[k];
            }
            prev_sum = row_sum.max(1e-30);
        }

        if current_state.is_none() {
            let last_row = &fwd_buf[(block_len - 1) * row_stride..block_len * row_stride];
            let sampled = sample_from_weights(last_row, rng);
            current_state = Some(sampled);
            path[end - 1] = sampled as u32;
        }

        for m in (start + 1..end).rev() {
            let next_state = current_state.unwrap();
            let r = p_recomb.get(m).copied().unwrap_or(0.0);
            let shift = r / n_states as f32;
            // Li-Stephens: P(stay) = (1-r) + r/K, P(switch) = r/K
            let stay = (1.0 - r) + shift;
            let row_idx = (m - 1 - start) * row_stride;
            let prev_row = &fwd_buf[row_idx..row_idx + row_stride];

            for k in 0..n_states {
                let trans = if k == next_state { stay } else { shift };
                weights[k] = prev_row[k] * trans;
            }
            let sampled = sample_from_weights(&weights, rng);
            path[m - 1] = sampled as u32;
            current_state = Some(sampled);
        }
    }
}

/// Sample phase swap decisions using Stochastic EM (single chain MCMC).
///
/// This implements Forward-Filtering Backward-Sampling (FFBS) with a single
/// Markov chain, which is the mathematically correct approach for phasing.
/// Multiple chains would require phase alignment to avoid symmetric mode
/// cancellation, so we use exactly one chain (Stochastic EM).
///
/// The algorithm:
/// 1. Build forward checkpoints for the combined (both haplotypes uncertain) HMM
/// 2. Run burn-in steps to let the chain mix
/// 3. Take exactly ONE sample from the posterior
/// 4. Return swap decisions directly from that sample
fn sample_swap_bits_mosaic(
    n_markers: usize,
    n_states: usize,
    p_recomb: &[f32],
    seq1: &[u8],
    seq2: &[u8],
    conf: &[f32],
    lookup: &RefAlleleLookup<'_>,
    het_positions: &[usize],
    seed: u64,
    burnin: usize,
    p_no_err: f32,
    p_err: f32,
) -> (Vec<u8>, Vec<f32>) {
    if het_positions.is_empty() || n_markers == 0 || n_states == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut combined_checkpoints = FwdCheckpoints::new(n_markers, n_states, MOSAIC_BLOCK_SIZE);
    let dummy_allele = vec![255u8; n_markers];
    let dummy_combined = vec![true; n_markers];
    build_fwd_checkpoints(
        &mut combined_checkpoints,
        n_markers,
        n_states,
        p_recomb,
        seq1,
        seq2,
        conf,
        &dummy_allele,
        &dummy_combined,
        lookup,
        p_no_err,
        p_err,
        EmissionMode::Combined,
    );

    let combined_checkpoints = Arc::new(combined_checkpoints);

    // Pure Stochastic EM: single chain, take the final sample after burn-in.
    // No averaging, no counting - just sample once from the posterior.
    let chain_seed = seed.wrapping_add(0xC0FFEE_BAAD_F00Du64);
    let mut chain = MosaicChain::new(
        chain_seed,
        n_markers,
        n_states,
        p_recomb,
        seq1,
        seq2,
        conf,
        lookup,
        Arc::clone(&combined_checkpoints),
        p_no_err,
        p_err,
    );

    // Burn-in: let the chain mix
    for _ in 0..burnin {
        chain.step();
    }

    // Take the final sample (Stochastic EM uses exactly one sample)
    // Additional samples would be correlated, so we ignore the `samples` parameter
    // and always take exactly one.
    chain.step();
    let (path1, path2) = chain.paths();

    // Determine swap decisions directly from the sampled paths
    let mut swap_bits = Vec::with_capacity(het_positions.len());
    let mut swap_lr = Vec::with_capacity(het_positions.len());

    for &m in het_positions.iter() {
        let a1 = seq1[m];
        let a2 = seq2[m];

        // Skip non-het positions
        if a1 == 255 || a2 == 255 || a1 == a2 {
            swap_bits.push(0);
            swap_lr.push(1.0);
            continue;
        }

        let ref1 = lookup.allele(m, path1[m] as usize);
        let ref2 = lookup.allele(m, path2[m] as usize);

        // Determine orientation from the sampled haplotype paths
        // Orient=0: keep current phase (ref1 matches a1, ref2 matches a2)
        // Orient=1: swap phase (ref1 matches a2, ref2 matches a1)
        let orient = if ref1 == a1 && ref2 == a2 {
            0u8
        } else if ref1 == a2 && ref2 == a1 {
            1u8
        } else if ref1 == 255 && ref2 == a2 {
            0u8
        } else if ref1 == 255 && ref2 == a1 {
            1u8
        } else if ref2 == 255 && ref1 == a1 {
            0u8
        } else if ref2 == 255 && ref1 == a2 {
            1u8
        } else {
            // No clear orientation from reference - keep current phase
            swap_bits.push(0);
            swap_lr.push(1.0);
            continue;
        };

        swap_bits.push(orient);
        // High confidence since this is a deterministic decision from the sample
        swap_lr.push(1e6);
    }

    (swap_bits, swap_lr)
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
        haps_at_mkr_a: &[u32],         // haplotypes at flanking Stage 1 marker
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
            mcmc_burnin: 1,
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
            mcmc_burnin: 1,
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
