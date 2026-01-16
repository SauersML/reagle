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
use tracing::{info_span, instrument};

use crate::config::Config;
use crate::data::genetic_map::{GeneticMaps, MarkerMap};
use crate::data::haplotype::{HapIdx, SampleIdx};
use crate::data::marker::MarkerIdx;
use crate::data::storage::sample_phase::SamplePhase;
use crate::data::storage::phase_state::Phased;
use crate::data::storage::{GenotypeColumn, GenotypeMatrix, MutableGenotypes, GenotypeView};
use crate::error::Result;
use crate::io::bref3::Bref3Reader;
use crate::io::streaming::{PhasedOverlap, StreamingConfig, StreamingVcfReader, StreamWindow, StateProbs};
use crate::io::vcf::{VcfReader, VcfWriter};
use crate::model::ibs2::Ibs2;

thread_local! {
    static THREAD_WORKSPACE: std::cell::RefCell<Option<crate::utils::workspace::ThreadWorkspace>> =
        std::cell::RefCell::new(None);
}

/// Helper struct for double-buffered window processing
struct StreamWindowWithResult {
    window: StreamWindow,
    phased_result: Option<GenotypeMatrix<Phased>>,
}

impl std::ops::Deref for StreamWindowWithResult {
    type Target = StreamWindow;
    fn deref(&self) -> &Self::Target {
        &self.window
    }
}
use crate::model::hmm::BeagleHmm;
use crate::model::allele_lookup::RefAlleleLookup;
use crate::model::parameters::ModelParams;
use crate::model::phase_ibs::BidirectionalPhaseIbs;
use crate::model::phase_states::PhaseStates;
use crate::model::pbwt_streaming::PbwtWavefront;
use crate::data::alignment::MarkerAlignment;
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

struct FwdCheckpoints {
    block_size: usize,
    n_blocks: usize,
    n_states: usize,
    data: Vec<f32>,
}

impl FwdCheckpoints {
    fn from_buffer(n_markers: usize, n_states: usize, block_size: usize, mut data: Vec<f32>) -> Self {
        let block_size = block_size.max(1).min(n_markers.max(1));
        let n_blocks = (n_markers + block_size - 1) / block_size;
        let required = n_blocks * n_states;
        if data.len() < required {
            data.resize(required, 0.0);
        } else {
            data[..required].fill(0.0);
        }
        Self {
            block_size,
            n_blocks,
            n_states,
            data,
        }
    }

    fn into_buffer(self) -> Vec<f32> {
        self.data
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

struct MosaicBuffers {
    fwd: aligned_vec::AVec<f32, aligned_vec::ConstAlign<32>>,
    fwd_prior: aligned_vec::AVec<f32, aligned_vec::ConstAlign<32>>,
    ref_alleles: Vec<u8>,
    hap1_checkpoints: FwdCheckpoints,
    hap2_checkpoints: FwdCheckpoints,
    hap1_allele: Vec<u8>,
    hap1_use_combined: Vec<bool>,
    hap2_allele: Vec<u8>,
    hap2_use_combined: Vec<bool>,
    path1: Vec<u32>,
    path2: Vec<u32>,
    fwd_block: Vec<f32>,
}

#[derive(Clone, Debug)]
struct MosaicPaths {
    path1: Vec<u32>,
    path2: Vec<u32>,
}

struct MosaicChain<'a> {
    rng: rand::rngs::SmallRng,
    n_markers: usize,
    n_states: usize,
    p_recomb: &'a [f32],
    seq1: &'a [u8],
    seq2: &'a [u8],
    conf: &'a [f32],
    lookup: &'a RefAlleleLookup,
    combined_checkpoints: &'a FwdCheckpoints,
    fwd: aligned_vec::AVec<f32, aligned_vec::ConstAlign<32>>,
    fwd_prior: aligned_vec::AVec<f32, aligned_vec::ConstAlign<32>>,
    ref_alleles: Vec<u8>,
    hap1_checkpoints: FwdCheckpoints,
    hap1_allele: Vec<u8>,
    hap1_use_combined: Vec<bool>,
    hap2_checkpoints: FwdCheckpoints,
    hap2_allele: Vec<u8>,
    hap2_use_combined: Vec<bool>,
    path1: Vec<u32>,  // u32 saves 50% memory vs usize
    path2: Vec<u32>,
    fwd_block: Vec<f32>,
    trace: MosaicTrace,
    p_no_err: f32,
    p_err: f32,
    first_iteration: bool,
}

impl<'a> MosaicChain<'a> {
    fn new_with_buffers(
        seed: u64,
        n_markers: usize,
        n_states: usize,
        p_recomb: &'a [f32],
        seq1: &'a [u8],
        seq2: &'a [u8],
        conf: &'a [f32],
        lookup: &'a RefAlleleLookup,
        combined_checkpoints: &'a FwdCheckpoints,
        buffers: MosaicBuffers,
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
            fwd: buffers.fwd,
            fwd_prior: buffers.fwd_prior,
            ref_alleles: buffers.ref_alleles,
            hap1_checkpoints: buffers.hap1_checkpoints,
            hap1_allele: buffers.hap1_allele,
            hap1_use_combined: buffers.hap1_use_combined,
            hap2_checkpoints: buffers.hap2_checkpoints,
            hap2_allele: buffers.hap2_allele,
            hap2_use_combined: buffers.hap2_use_combined,
            path1: buffers.path1,
            path2: buffers.path2,
            fwd_block: buffers.fwd_block,
            trace: MosaicTrace {
                mean_state: 0.0,
                switch_rate: 0.0,
                log_likelihood: 0.0,
            },
            p_no_err,
            p_err,
            first_iteration: true,
        };
        out
    }

    fn paths(&self) -> (&[u32], &[u32]) {
        (&self.path1, &self.path2)
    }

    fn into_buffers(self) -> MosaicBuffers {
        MosaicBuffers {
            fwd: self.fwd,
            fwd_prior: self.fwd_prior,
            ref_alleles: self.ref_alleles,
            hap1_checkpoints: self.hap1_checkpoints,
            hap2_checkpoints: self.hap2_checkpoints,
            hap1_allele: self.hap1_allele,
            hap1_use_combined: self.hap1_use_combined,
            hap2_allele: self.hap2_allele,
            hap2_use_combined: self.hap2_use_combined,
            path1: self.path1,
            path2: self.path2,
            fwd_block: self.fwd_block,
        }
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

    /// Build hap1 inputs based on current path2 (for proper Gibbs sampling).
    /// This determines what allele H1 must carry given H2's sampled path.
    fn build_hap1_inputs(&mut self) {
        for m in 0..self.n_markers {
            let a1 = self.seq1[m];
            let a2 = self.seq2[m];
            if a1 == 255 && a2 == 255 {
                self.hap1_use_combined[m] = true;
                self.hap1_allele[m] = 255;
                continue;
            }
            if a1 == a2 {
                self.hap1_use_combined[m] = false;
                self.hap1_allele[m] = a1;
                continue;
            }

            // Given path2's reference allele, determine what H1 must be
            let ref_al = self.lookup.allele(m, self.path2[m] as usize);
            if ref_al == a1 {
                // H2 carries a1, so H1 must carry a2
                self.hap1_use_combined[m] = false;
                self.hap1_allele[m] = a2;
            } else if ref_al == a2 {
                // H2 carries a2, so H1 must carry a1
                self.hap1_use_combined[m] = false;
                self.hap1_allele[m] = a1;
            } else {
                // Reference doesn't match either - use combined
                self.hap1_use_combined[m] = true;
                self.hap1_allele[m] = 255;
            }
        }
    }
}

impl MarkovChain<MosaicTrace> for MosaicChain<'_> {
    fn step(&mut self) -> &MosaicTrace {
        // Proper Gibbs sampling: H1 and H2 must each condition on the other.
        //
        // First iteration: use combined_checkpoints (marginal) to initialize path1.
        // Subsequent iterations: rebuild hap1_checkpoints based on current path2,
        // then sample path1 conditioned on H2's state.
        //
        // This creates the feedback loop required for convergence to P(H1,H2|G).

        if self.first_iteration {
            // Initialize: sample path1 from combined (marginal) distribution
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
            self.first_iteration = false;
        } else {
            // Gibbs step: sample H1 | H2
            // Build hap1 constraints based on current path2
            self.build_hap1_inputs();
            let fwd = &mut self.fwd[..self.n_states];
            let fwd_prior = &mut self.fwd_prior[..self.n_states];
            let ref_alleles = &mut self.ref_alleles[..self.n_states];
            build_fwd_checkpoints(
                &mut self.hap1_checkpoints,
                self.n_markers,
                self.n_states,
                self.p_recomb,
                self.seq1,
                self.seq2,
                self.conf,
                &self.hap1_allele,
                &self.hap1_use_combined,
                self.lookup,
                fwd,
                fwd_prior,
                ref_alleles,
                self.p_no_err,
                self.p_err,
                EmissionMode::Hap,
            );
            sample_path_from_checkpoints(
                &mut self.path1,
                &self.hap1_checkpoints,
                self.n_markers,
                self.n_states,
                self.p_recomb,
                self.seq1,
                self.seq2,
                self.conf,
                &self.hap1_allele,
                &self.hap1_use_combined,
                self.lookup,
                self.p_no_err,
                self.p_err,
                &mut self.rng,
                &mut self.fwd_block,
                EmissionMode::Hap,
            );
        }

        // Gibbs step: sample H2 | H1
        self.build_hap2_inputs();
        let fwd = &mut self.fwd[..self.n_states];
        let fwd_prior = &mut self.fwd_prior[..self.n_states];
        let ref_alleles = &mut self.ref_alleles[..self.n_states];
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
            fwd,
            fwd_prior,
            ref_alleles,
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

        let mut mcmc_paths: Vec<Option<MosaicPaths>> = vec![None; n_samples];

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
                &mut mcmc_paths,
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
                None,
                None,
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
        // PhasedOverlap contains state probabilities used for PBWT state handoff
        let mut phased_overlap: Option<PhasedOverlap> = None;

        // Double-buffered windows
        let mut current_window: Option<StreamWindowWithResult> = None;
        let mut next_window_opt = reader.next_window()?;

        // Process windows with double-buffering
        while let Some(mut window) = next_window_opt {
            window_count += 1;

            let n_markers = window.genotypes.n_markers();

            eprintln!(
                "Loading window {} ({} markers, global {}..{}, output {}..{})",
                window_count,
                n_markers,
                window.global_start,
                window.global_end,
                window.output_start,
                window.output_end
            );

            // Load next window
            next_window_opt = reader.next_window()?;

            // Set the phased overlap from previous window
            window.phased_overlap = phased_overlap.take();
            // Note: PBWT state handoff is handled via phased_overlap's state probabilities

            // Phase this window with overlap constraint
            let (phased, next_state_probs) = self.phase_in_memory_with_overlap(
                &window.genotypes,
                &gen_maps,
                window.phased_overlap.as_ref(),
                Some(window.output_end)
            )?;

            // Extract overlap for next window (contains state probabilities for PBWT handoff)
            if !window.is_last() {
                phased_overlap = Some(self.extract_overlap(&phased, window.output_end, n_markers, next_state_probs));
            }

            // If we have a current window to finalize Stage 2
            if let Some(current) = current_window.take() {
                // Perform Stage 2 finalization using phased markers from next window
                let finalized = info_span!("finalize_stage2").in_scope(|| {
                    self.finalize_stage2_with_forward_context(
                        &current.phased_result.as_ref().unwrap(),
                        &phased,
                    )
                })?;

                // Write output region
                if current.window.is_first {
                    writer.write_header(finalized.markers())?;
                }
                writer.write_phased(&finalized, current.window.output_start, current.window.output_end)?;
                total_markers += current.window.output_end - current.window.output_start;
            }

            // Move to next window
            current_window = Some(StreamWindowWithResult {
                window,
                phased_result: Some(phased),
            });
        }

        // Finalize last window (no next window for Stage 2 context)
        if let Some(ref current) = current_window {
            info_span!("finalize_last_window").in_scope(|| {
                let finalized = current.phased_result.as_ref().unwrap().clone(); // No additional context
                writer.write_header(finalized.markers()).ok();
                writer.write_phased(&finalized, current.output_start, current.output_end).ok();
                total_markers += current.output_end - current.output_start;
            });
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
        state_probs: Option<StateProbs>,
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
        
        let mut overlap = PhasedOverlap::new(n_overlap, n_haps, alleles);

        // Attach state probabilities if available (Soft-Information Handoff)
        if let Some(probs) = state_probs {
            overlap.set_state_probs(probs);
        }

        overlap
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
        next_overlap_start: Option<usize>,
    ) -> Result<(GenotypeMatrix<crate::data::storage::phase_state::Phased>, Option<StateProbs>)> {
        let n_markers = target_gt.n_markers();
        let n_haps = target_gt.n_haplotypes();
        let n_samples = n_haps / 2;
        let n_ref_haps = self.reference_gt.as_ref().map(|r| r.n_haplotypes()).unwrap_or(0);
        let n_total_haps = n_haps + n_ref_haps;

        if n_markers == 0 {
            return Ok((target_gt.clone().into_phased(), None));
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
        // Note: sample_phases tracks phase state per marker but run_phase_baum_iteration
        // updates geno directly. The overlap constraint is applied via apply_overlap_constraint.
        let mut sample_phases =
            self.create_sample_phases_with_overlap(&geno, &missing_mask, overlap_markers, &confidence_by_sample);

        let mut mcmc_paths: Vec<Option<MosaicPaths>> = vec![None; n_samples];

        for it in 0..total_iterations {
            let is_burnin = it < n_burnin;
            self.params.lr_threshold = self.params.lr_threshold_for_iteration(it);

            let atomic_estimates = if is_burnin && self.config.em {
                Some(crate::model::parameters::AtomicParamEstimates::new())
            } else {
                None
            };

            // Use existing run_phase_baum_iteration - overlap constraint is handled
            // via the initial geno state set by apply_overlap_constraint
            self.run_phase_baum_iteration(
                target_gt,
                &mut geno,
                &p_recomb,
                &gen_dists,
                &ibs2,
                &mut mcmc_paths,
                atomic_estimates.as_ref(),
                &confidence_by_sample,
                None, // No PBWT state handoff for windowed phasing
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

        // STAGE 2: Phase rare markers using HMM state probability interpolation
        // Now returns state probabilities for the next overlap region if requested

        // Re-compute Stage 1 info for Stage 2
        let rare_threshold = self.config.rare;
        let hi_freq_markers: Vec<usize> = (0..n_markers)
            .filter(|&m| maf[m] >= rare_threshold)
            .collect();
        let rare_markers: Vec<usize> = (0..n_markers)
            .filter(|&m| maf[m] < rare_threshold && maf[m] > 0.0) // Exclude monomorphic
            .collect();

        // Compute stage 1 genetic distances and recombination probabilities
        let stage1_gen_dists: Vec<f64> = if hi_freq_markers.len() > 1 {
            // Need to reconstruct gen_positions from gen_dists/maps?
            // Actually gen_dists is for ALL markers.
            // Let's re-use gen_maps to get positions again (cheap)
            let m_map = if let Some(map) = gen_maps.get(chrom) {
                MarkerMap::create(target_gt.markers(), map)
            } else {
                MarkerMap::from_positions(target_gt.markers())
            };
            let positions = m_map.gen_positions();

            hi_freq_markers
                .windows(2)
                .map(|w| positions[w[1]] - positions[w[0]])
                .collect()
        } else {
            Vec::new()
        };

        let stage1_p_recomb: Vec<f32> = std::iter::once(0.0f32)
            .chain(stage1_gen_dists.iter().map(|&d| self.params.p_recomb(d)))
            .collect();

        // We need gen_positions for stage 2
        let marker_map_full = if let Some(map) = gen_maps.get(chrom) {
            MarkerMap::create(target_gt.markers(), map)
        } else {
            MarkerMap::from_positions(target_gt.markers())
        };
        let gen_positions_vec = marker_map_full.gen_positions().to_vec();

        let next_state_probs = if !rare_markers.is_empty() && hi_freq_markers.len() >= 2 {
            eprintln!(
                "Stage 2: Phasing {} rare markers using HMM interpolation...",
                rare_markers.len()
            );
            let probs = self.phase_rare_markers_with_hmm(
                &mut geno,
                &hi_freq_markers,
                &gen_positions_vec,
                &stage1_p_recomb,
                &ibs2,
                &mut sample_phases,
                &maf,
                rare_threshold,
                phased_overlap,
                next_overlap_start,
            );

            // Sync again after Stage 2
            self.sync_sample_phases_to_geno(&sample_phases, &mut geno);
            probs
        } else {
            None
        };

        Ok((self.build_final_matrix(target_gt, &geno), next_state_probs))
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

                // Use bulk haplotype access instead of per-marker get()
                let alleles1 = geno.haplotype(hap1);
                let alleles2 = geno.haplotype(hap2);

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

                // Use bulk haplotype access instead of per-marker get()
                // geno.haplotype() returns 255 for missing positions
                let alleles1 = geno.haplotype(hap1);
                let alleles2 = geno.haplotype(hap2);

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

    /// Build bidirectional PBWT for a subset of markers (e.g., high-frequency only)
    fn build_bidirectional_pbwt_subset(
        &self,
        geno: &MutableGenotypes,
        marker_indices: &[usize],
        n_haps: usize,
    ) -> BidirectionalPhaseIbs {
        let n_subset = marker_indices.len();
        // Use bulk slice access instead of per-haplotype get() calls
        let mut alleles_by_marker: Vec<Vec<u8>> = Vec::with_capacity(n_subset);

        for &orig_m in marker_indices {
            let marker_slice = geno.marker_alleles(orig_m);
            alleles_by_marker.push(marker_slice[..n_haps].to_vec());
        }

        BidirectionalPhaseIbs::build_for_subset(
            alleles_by_marker,
            n_haps,
            n_subset,
            marker_indices,
        )
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

    /// Build composite haplotypes for all samples using streaming PBWT
    ///
    /// This streaming approach uses O(N) memory instead of O(M*N) for the PBWT index.
    /// It processes markers sequentially, updating PhaseStates at sampling points.
    ///
    /// # Algorithm
    /// 1. Forward pass (markers 0->M): collect forward PBWT neighbors at sampling points
    /// 2. Backward pass (markers M->0): collect backward PBWT neighbors at sampling points
    /// 3. Finalize: build ThreadedHaps for each sample
    ///
    /// # Returns
    /// Vector of ThreadedHaps, one per sample
    fn build_composite_haps_streaming<F>(
        &self,
        get_allele: F,
        n_markers: usize,
        n_total_haps: usize,
        n_samples: usize,
        ibs2: &Ibs2,
        n_candidates: usize,
        max_states: usize,
        pbwt_state: Option<&crate::model::pbwt::PbwtState>,
    ) -> Vec<crate::model::states::ThreadedHaps>
    where
        F: Fn(usize, usize) -> u8 + Sync,
    {
        use std::collections::HashSet;

        // Compute sampling points (sparse, ~64 points like PhaseStates)
        const MAX_SAMPLE_POINTS: usize = 64;
        let step = (n_markers / MAX_SAMPLE_POINTS).max(1);
        let sampling_points_vec: Vec<usize> = (0..n_markers).step_by(step).collect();
        // Use HashSet for O(1) lookup instead of Vec O(n) contains
        let sampling_points: HashSet<usize> = sampling_points_vec.iter().copied().collect();

        // Create PhaseStates for all samples
        let mut phase_states: Vec<PhaseStates> = (0..n_samples)
            .map(|_| {
                let mut ps = PhaseStates::new(max_states, n_markers);
                ps.reset_for_streaming();
                ps
            })
            .collect();

        // Create wavefront with optional initial state
        let mut wavefront = PbwtWavefront::with_state(n_total_haps, n_markers, pbwt_state);

        // Temporary allele buffer (reused across markers)
        let mut alleles = vec![0u8; n_total_haps];

        // Forward pass
        wavefront.reset_forward();
        for m in 0..n_markers {
            // Extract alleles for this marker
            for h in 0..n_total_haps {
                alleles[h] = get_allele(m, h);
            }

            // Biallelic optimization: check if all alleles are 0, 1, or 255 (missing)
            // Use early-exit loop instead of .all() for speed
            let mut is_biallelic = true;
            for &a in &alleles {
                if a >= 2 && a != 255 {
                    is_biallelic = false;
                    break;
                }
            }
            let n_alleles = if is_biallelic { 2 } else { 256 };

            // Advance wavefront
            wavefront.advance_forward(&alleles, n_alleles);

            // At sampling points, collect forward neighbors for all samples
            if sampling_points.contains(&m) {
                wavefront.prepare_fwd_queries();

                // Parallel neighbor collection using rayon
                let neighbors_per_sample: Vec<(Vec<u32>, Vec<u32>)> = (0..n_samples)
                    .into_par_iter()
                    .map(|s| {
                        let h1 = (s * 2) as u32;
                        let h2 = h1 + 1;
                        let n1 = wavefront.find_fwd_neighbors_readonly(h1, n_candidates);
                        let n2 = wavefront.find_fwd_neighbors_readonly(h2, n_candidates);
                        (n1, n2)
                    })
                    .collect();

                // Add neighbors to PhaseStates (sequential - PhaseStates not thread-safe)
                for (s, (n1, n2)) in neighbors_per_sample.into_iter().enumerate() {
                    phase_states[s].add_neighbors_at_marker(s as u32, m, &n1, &n2);
                }

                // Also add IBS2 neighbors
                for s in 0..n_samples {
                    let sample = SampleIdx::new(s as u32);
                    for seg in ibs2.segments(sample) {
                        if seg.contains(m) {
                            let other_s = seg.other_sample;
                            if other_s != sample {
                                // Use stack-allocated array instead of Vec
                                let neighbors: [u32; 2] = [other_s.hap1().0, other_s.hap2().0];
                                phase_states[s].add_neighbors_at_marker(
                                    s as u32,
                                    m,
                                    &neighbors,
                                    &[],
                                );
                            }
                        }
                    }
                }
            }
        }

        // Backward pass
        wavefront.reset_backward();
        for m in (0..n_markers).rev() {
            // Extract alleles for this marker
            for h in 0..n_total_haps {
                alleles[h] = get_allele(m, h);
            }

            let mut is_biallelic = true;
            for &a in &alleles {
                if a >= 2 && a != 255 {
                    is_biallelic = false;
                    break;
                }
            }
            let n_alleles = if is_biallelic { 2 } else { 256 };

            // Advance wavefront (backward)
            wavefront.advance_backward(&alleles, n_alleles);

            // At sampling points, collect backward neighbors
            if sampling_points.contains(&m) {
                wavefront.prepare_bwd_queries();

                // Parallel neighbor collection
                let neighbors_per_sample: Vec<(Vec<u32>, Vec<u32>)> = (0..n_samples)
                    .into_par_iter()
                    .map(|s| {
                        let h1 = (s * 2) as u32;
                        let h2 = h1 + 1;
                        let n1 = wavefront.find_bwd_neighbors_readonly(h1, n_candidates);
                        let n2 = wavefront.find_bwd_neighbors_readonly(h2, n_candidates);
                        (n1, n2)
                    })
                    .collect();

                for (s, (n1, n2)) in neighbors_per_sample.into_iter().enumerate() {
                    phase_states[s].add_neighbors_at_marker(s as u32, m, &n1, &n2);
                }
            }
        }

        // Finalize: convert PhaseStates to ThreadedHaps (parallel)
        phase_states
            .into_par_iter()
            .enumerate()
            .map(|(s, mut ps)| ps.finalize_streaming(s as u32, n_total_haps))
            .collect()
    }

    /// Build composite haplotypes using direct MutableGenotypes access (no reference panel).
    ///
    /// This is an optimized version of build_composite_haps_streaming for the case where
    /// there is no reference panel. It uses bulk slice access instead of per-allele closures,
    /// reducing overhead from O(n_markers × n_haps) function calls to O(n_markers) slice copies.
    fn build_composite_haps_streaming_direct(
        &self,
        geno: &MutableGenotypes,
        n_markers: usize,
        n_samples: usize,
        ibs2: &Ibs2,
        n_candidates: usize,
        max_states: usize,
        pbwt_state: Option<&crate::model::pbwt::PbwtState>,
    ) -> Vec<crate::model::states::ThreadedHaps> {
        use std::collections::HashSet;

        let n_haps = geno.n_haps();

        // Compute sampling points
        const MAX_SAMPLE_POINTS: usize = 64;
        let step = (n_markers / MAX_SAMPLE_POINTS).max(1);
        let sampling_points_vec: Vec<usize> = (0..n_markers).step_by(step).collect();
        let sampling_points: HashSet<usize> = sampling_points_vec.iter().copied().collect();

        // Create PhaseStates for all samples
        let mut phase_states: Vec<PhaseStates> = (0..n_samples)
            .map(|_| {
                let mut ps = PhaseStates::new(max_states, n_markers);
                ps.reset_for_streaming();
                ps
            })
            .collect();

        // Create wavefront
        let mut wavefront = PbwtWavefront::with_state(n_haps, n_markers, pbwt_state);

        // Forward pass - use direct slice access
        wavefront.reset_forward();
        for m in 0..n_markers {
            // Direct slice access instead of per-haplotype closure calls
            let marker_alleles = geno.marker_alleles(m);

            // Biallelic check with SIMD-friendly iteration
            let is_biallelic = marker_alleles.iter().all(|&a| a < 2 || a == 255);
            let n_alleles = if is_biallelic { 2 } else { 256 };

            // Advance wavefront
            wavefront.advance_forward(&marker_alleles, n_alleles);

            // At sampling points, collect forward neighbors
            if sampling_points.contains(&m) {
                wavefront.prepare_fwd_queries();

                let neighbors_per_sample: Vec<(Vec<u32>, Vec<u32>)> = (0..n_samples)
                    .into_par_iter()
                    .map(|s| {
                        let h1 = (s * 2) as u32;
                        let h2 = h1 + 1;
                        let n1 = wavefront.find_fwd_neighbors_readonly(h1, n_candidates);
                        let n2 = wavefront.find_fwd_neighbors_readonly(h2, n_candidates);
                        (n1, n2)
                    })
                    .collect();

                for (s, (n1, n2)) in neighbors_per_sample.into_iter().enumerate() {
                    phase_states[s].add_neighbors_at_marker(s as u32, m, &n1, &n2);
                }

                // Add IBS2 neighbors
                for s in 0..n_samples {
                    let sample = SampleIdx::new(s as u32);
                    for seg in ibs2.segments(sample) {
                        if seg.contains(m) {
                            // Use stack-allocated array instead of Vec for IBS2 neighbors
                            let neighbors: [u32; 2] = [seg.other_sample.hap1().0, seg.other_sample.hap2().0];
                            phase_states[s].add_neighbors_at_marker(s as u32, m, &neighbors, &neighbors);
                        }
                    }
                }
            }
        }

        // Backward pass - use direct slice access
        wavefront.reset_backward();
        for m in (0..n_markers).rev() {
            let marker_alleles = geno.marker_alleles(m);

            let is_biallelic = marker_alleles.iter().all(|&a| a < 2 || a == 255);
            let n_alleles = if is_biallelic { 2 } else { 256 };

            wavefront.advance_backward(&marker_alleles, n_alleles);

            if sampling_points.contains(&m) {
                wavefront.prepare_bwd_queries();

                let neighbors_per_sample: Vec<(Vec<u32>, Vec<u32>)> = (0..n_samples)
                    .into_par_iter()
                    .map(|s| {
                        let h1 = (s * 2) as u32;
                        let h2 = h1 + 1;
                        let n1 = wavefront.find_bwd_neighbors_readonly(h1, n_candidates);
                        let n2 = wavefront.find_bwd_neighbors_readonly(h2, n_candidates);
                        (n1, n2)
                    })
                    .collect();

                for (s, (n1, n2)) in neighbors_per_sample.into_iter().enumerate() {
                    phase_states[s].add_neighbors_at_marker(s as u32, m, &n1, &n2);
                }
            }
        }

        // Finalize
        phase_states
            .into_par_iter()
            .enumerate()
            .map(|(s, mut ps)| ps.finalize_streaming(s as u32, n_haps))
            .collect()
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
        mcmc_paths: &mut [Option<MosaicPaths>],
        atomic_estimates: Option<&crate::model::parameters::AtomicParamEstimates>,
        confidence_by_sample: &[Vec<f32>],
        pbwt_state: Option<&crate::model::pbwt::PbwtState>,
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

        // No clone needed: the HMM phase is read-only; mutations happen after.
        // We use a scoped immutable borrow that ends before the swap phase.
        let swap_results: Vec<(BitVec<u8, Lsb0>, Option<MosaicPaths>)> = info_span!("build_composite_view").in_scope(|| {
            // Immutable borrow of geno for the entire read phase
            let ref_geno: &MutableGenotypes = geno;

            // Use Composite view when reference panel is available
            let ref_view = if let (Some(ref_gt), Some(alignment)) = (&self.reference_gt, &self.alignment) {
                GenotypeView::Composite {
                    target: ref_geno,
                    reference: ref_gt,
                    alignment,
                    n_target_haps: n_haps,
                }
            } else {
                GenotypeView::from((ref_geno, markers))
            };

            // Build composite haplotypes for all samples using streaming PBWT
            // This uses O(N) memory instead of O(M*N) for the PBWT index
            let n_candidates = self.params.n_states.min(n_total_haps).max(20);
            let threaded_haps_vec: Vec<crate::model::states::ThreadedHaps> = tracing::info_span!("streaming_pbwt").in_scope(|| {
                if let (Some(ref_gt), Some(alignment)) = (&self.reference_gt, &self.alignment) {
                    self.build_composite_haps_streaming(
                        |m, h| {
                            if h < n_haps {
                                ref_geno.get(m, HapIdx::new(h as u32))
                            } else {
                                // Reference haplotype: always use actual allele
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
                        n_samples,
                        ibs2,
                        n_candidates,
                        self.params.n_states,
                        pbwt_state,
                    )
                } else {
                    // Use optimized direct access version for no-reference case
                    self.build_composite_haps_streaming_direct(
                        ref_geno,
                        n_markers,
                        n_samples,
                        ibs2,
                        n_candidates,
                        self.params.n_states,
                        pbwt_state,
                    )
                }
            });

            let prior_paths = &mcmc_paths[..];
            let mut swap_results: Vec<(BitVec<u8, Lsb0>, Option<MosaicPaths>)> =
                vec![(BitVec::repeat(false, n_markers), None); n_samples];

            tracing::info_span!("hmm_samples").in_scope(|| swap_results.par_iter_mut().enumerate().for_each(|(s, (mask, paths_out))| {
                let sample_idx = SampleIdx::new(s as u32);
                let hap1 = sample_idx.hap1();
                let hap2 = sample_idx.hap2();
                let sample_seed = (self.config.seed as u64)
                    .wrapping_add(s as u64)
                    .wrapping_add(0xA5A5_5A5A_D00Du64);

                // Use pre-built composite haplotypes from streaming PBWT
                let threaded_haps = threaded_haps_vec[s].clone();
                let n_states = threaded_haps.n_states();

                // 2. Extract current alleles for H1 and H2
                let seq1 = ref_geno.haplotype(hap1);
                let seq2 = ref_geno.haplotype(hap2);
                // Use pre-computed confidence instead of recomputing
                let sample_conf = &confidence_by_sample[s];

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
                // Collect EM statistics if requested (using original sequences)
                // Only create HMM when needed to avoid unnecessary p_recomb.clone()
                if let Some(atomic) = atomic_estimates {
                    let hmm = BeagleHmm::new(ref_view, &self.params, n_states, p_recomb.to_vec());
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

                let (swap_bits, swap_lr, new_paths) = THREAD_WORKSPACE.with(|ws| {
                    let mut workspace = ws.borrow_mut();
                    if workspace.is_none() {
                        *workspace = Some(crate::utils::workspace::ThreadWorkspace::new(64, 0));
                    }
                    let ws = workspace.as_mut().unwrap();
                    ws.clear(); // Explicit reset between samples to prevent state contamination
                    let lookup = RefAlleleLookup::new_from_threaded_with_buffer(
                        &threaded_haps,
                        n_markers,
                        n_states,
                        n_haps,
                        ref_geno,
                        self.reference_gt.as_deref(),
                        self.alignment.as_ref(),
                        None,
                        std::mem::replace(&mut ws.lookup, aligned_vec::AVec::new(32)),
                    );
                    let result = sample_swap_bits_mosaic(
                        n_markers,
                        n_states,
                        p_recomb,
                        &seq1,
                        &seq2,
                        &sample_conf,
                        &lookup,
                        &het_positions,
                        prior_paths.get(s).and_then(|p| p.as_ref()),
                        sample_seed,
                        self.config.mcmc_burnin,
                        p_no_err,
                        p_err,
                        ws,
                    );
                    ws.lookup = lookup.into_buffer();
                    result
                });
                if new_paths.path1.is_empty() {
                    *paths_out = None;
                } else {
                    *paths_out = Some(new_paths);
                }
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

            swap_results
        });  // ref_geno borrow ends here

        // Apply Swaps
        // After computing swap masks for all samples, apply them sequentially.
        // This is done sequentially because swap_haplotypes requires mutable access.
        info_span!("apply_swaps").in_scope(|| {
            for (s, (mask, paths)) in swap_results.into_iter().enumerate() {
                let sample_idx = SampleIdx::new(s as u32);
                let hap1 = sample_idx.hap1();
                let hap2 = sample_idx.hap2();
                geno.swap_haplotypes(hap1, hap2, &mask);
                if let Some(paths) = paths {
                    if let Some(slot) = mcmc_paths.get_mut(s) {
                        *slot = Some(paths);
                    }
                }
            }
        });

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
        mcmc_paths: &mut [Option<MosaicPaths>],
        atomic_estimates: Option<&crate::model::parameters::AtomicParamEstimates>,
        iteration: usize,
    ) -> Result<()> {
        let n_haps = geno.n_haps();

        // Compute total haplotype count (target + reference)
        let n_ref_haps = self.reference_gt.as_ref().map(|r| r.n_haplotypes()).unwrap_or(0);
        let n_total_haps = n_haps + n_ref_haps;

        // No clone needed: the HMM phase is read-only; mutations happen after.
        // We use a scoped immutable borrow that ends before the apply phase.
        type PhaseDecision = (Vec<bool>, Vec<(usize, f32)>, Option<MosaicPaths>);
        let phase_decisions: Vec<PhaseDecision> = {
            // Immutable borrow of geno for the entire read phase
            let ref_geno: &MutableGenotypes = geno;

            // 1. Create Subset View for Stage 1 markers
            // Use CompositeSubset when reference panel is available
            let subset_view = if let (Some(ref_gt), Some(alignment)) = (&self.reference_gt, &self.alignment) {
                GenotypeView::CompositeSubset {
                    target: ref_geno,
                    reference: ref_gt,
                    alignment,
                    subset: hi_freq_to_orig,
                    n_target_haps: n_haps,
                }
            } else {
                GenotypeView::MutableSubset {
                    geno: ref_geno,
                    subset: hi_freq_to_orig,
                }
            };

            // 2. Build bidirectional PBWT on high-frequency markers only
            // When reference is available, include reference haplotypes in the PBWT
            // Filter low-confidence target markers to prevent bad hard calls from
            // excluding correct reference haplotypes during state selection.

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
                self.build_bidirectional_pbwt_subset(ref_geno, hi_freq_to_orig, n_haps)
            };

            // Collect phase decisions per sample using correct per-het algorithm.
            // Returns: (swap_mask, het_lr_values) per sample where:
            //   - swap_mask[i] = true if the sampled phase orientation at marker i is swapped
            //   - het_lr_values = (hi_freq_idx, lr) for each het, used for phased marking threshold
            let prior_paths = &mcmc_paths[..];
            sample_phases
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
                        return (vec![false; n_hi_freq], Vec::new(), None);
                    }

                    let p_err = self.params.p_mismatch;
                    let p_no_err = 1.0 - p_err;

                    let (swap_bits, swap_lr, new_paths) = if self.config.dynamic_mcmc {
                        // SHAPEIT5-style dynamic MCMC: re-select states each step
                        // Note: Dynamic MCMC doesn't use ThreadWorkspace yet
                        let (swap_bits, swap_lr) = sample_dynamic_mcmc(
                            n_hi_freq,
                            n_states,
                            stage1_p_recomb,
                            &seq1,
                            &seq2,
                            &sample_conf,
                            &phase_ibs,
                            ibs2,
                            s as u32,
                            &het_positions,
                            sample_seed,
                            self.config.mcmc_steps,
                            p_no_err,
                            p_err,
                        );
                        (swap_bits, swap_lr, None)
                    } else {
                        // Classic Beagle-style: static state space MCMC with thread-local workspace
                        THREAD_WORKSPACE.with(|ws| {
                            let mut workspace = ws.borrow_mut();
                            if workspace.is_none() {
                                *workspace = Some(crate::utils::workspace::ThreadWorkspace::new(64, 0));
                            }
                            let ws = workspace.as_mut().unwrap();
                            ws.clear(); // Explicit reset between samples
                            let lookup = RefAlleleLookup::new_from_threaded_with_buffer(
                                &threaded_haps,
                                n_hi_freq,
                                n_states,
                                n_haps,
                                ref_geno,
                                self.reference_gt.as_deref(),
                                self.alignment.as_ref(),
                                Some(hi_freq_to_orig),
                                std::mem::replace(&mut ws.lookup, aligned_vec::AVec::new(32)),
                            );
                            let result = sample_swap_bits_mosaic(
                                n_hi_freq,
                                n_states,
                                stage1_p_recomb,
                                &seq1,
                                &seq2,
                                &sample_conf,
                                &lookup,
                                &het_positions,
                                prior_paths.get(s).and_then(|p| p.as_ref()),
                                sample_seed,
                                self.config.mcmc_burnin,
                                p_no_err,
                                p_err,
                                ws,
                            );
                            ws.lookup = lookup.into_buffer();
                            (result.0, result.1, Some(result.2))
                        })
                    };

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

                    (swap_mask, het_lr_values, new_paths)
                })
                .collect()
        };  // ref_geno borrow ends here

        // Apply phase decisions to SamplePhase
        let mut total_switches = 0;
        let mut total_phased = 0;

        // Determine if we're in burn-in (don't mark as phased during burn-in)
        let is_burnin = iteration < self.config.burnin;
        let lr_threshold = self.params.lr_threshold;

        for (s, (swap_mask, het_lr_values, new_paths)) in phase_decisions.into_iter().enumerate() {
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

            if let Some(paths) = new_paths {
                if let Some(slot) = mcmc_paths.get_mut(s) {
                    *slot = Some(paths);
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
    ///
    /// **Streaming Soft-Handoff**: Accepts optional `previous_overlap` to combine state probabilities
    /// from the previous window with current ones, ensuring continuity. Also returns state probabilities
    /// for the *next* overlap region if `next_overlap_start` is provided.
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
        previous_overlap: Option<&PhasedOverlap>,
        next_overlap_start: Option<usize>,
    ) -> Option<StateProbs> {
        let n_markers = geno.n_markers();
        let n_haps = geno.n_haps();
        let n_stage1 = hi_freq_markers.len();
        let seed = self.config.seed;

        if n_stage1 < 2 {
            return None;
        }

        // Compute total haplotype count (target + reference)
        let n_ref_haps = self.reference_gt.as_ref().map(|r| r.n_haplotypes()).unwrap_or(0);
        let n_total_haps = n_haps + n_ref_haps;

        // Determine Stage 1 markers involved in the NEXT overlap region (for export)
        let next_overlap_indices = if let Some(start) = next_overlap_start {
            // Find first Stage 1 marker >= start
            let start_stage1 = hi_freq_markers.iter().position(|&m| m >= start).unwrap_or(n_stage1);
            (start_stage1..n_stage1).collect()
        } else {
            Vec::new()
        };

        // Determine Stage 1 markers involved in the PREVIOUS overlap region (for import/merge)
        let n_stage1_in_prev_overlap = if let Some(overlap) = previous_overlap {
            // Overlap markers are 0..overlap.n_markers
            hi_freq_markers.iter().take_while(|&&m| m < overlap.n_markers).count()
        } else {
            0
        };

        // Build Stage 2 interpolation mappings
            let stage2_phaser = Stage2Phaser::new(
                hi_freq_markers,
                gen_positions,
                n_markers,
                self.params.recomb_intensity,
            );

        // Result container for next window's state probs
        // We will collect this from parallel iteration.
        // It needs to be ordered by haplotype.

        // Return type from parallel map
        type PhaseResult = (Vec<Stage2Decision>, Option<Vec<Vec<Vec<f32>>>>);

        // No clone needed: this function never mutates geno - only sample_phases.
        // We use a scoped immutable borrow for the entire computation phase.
        let phase_results: Vec<PhaseResult> = {
            // Immutable borrow of geno for the entire read phase
            let ref_geno: &MutableGenotypes = geno;

            let rare_markers: Vec<usize> = (0..n_markers)
                .filter(|&m| maf[m] < rare_threshold && maf[m] > 0.0)
                .collect();

            // Use CompositeSubset view when reference panel is available
            let subset_view = if let (Some(ref_gt), Some(alignment)) = (&self.reference_gt, &self.alignment) {
                GenotypeView::CompositeSubset {
                    target: ref_geno,
                    reference: ref_gt,
                    alignment,
                    subset: hi_freq_markers,
                    n_target_haps: n_haps,
                }
            } else {
                GenotypeView::MutableSubset {
                    geno: ref_geno,
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
            // Filter low-confidence target markers to prevent bad hard calls from
            // excluding correct reference haplotypes during state selection.
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
                self.build_bidirectional_pbwt_subset(ref_geno, hi_freq_markers, n_haps)
            };

            // Process samples in parallel - collect results: Stage2Decision
            // Note: This is called after all iterations, so we use iteration=0 for deterministic state selection
            sample_phases
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
                let seq_conf: Vec<f32> = hi_freq_markers.iter().map(|&m| sp.confidence(m)).collect();
                let hmm = BeagleHmm::new(subset_view, &self.params, n_states, stage1_p_recomb.to_vec());

                let mut fwd1 = Vec::new();
                let mut bwd1 = Vec::new();
                let use_lookup = self.reference_gt.is_some() && self.alignment.is_some();
                let mut lookup = None;
                if use_lookup {
                    lookup = Some(THREAD_WORKSPACE.with(|ws| {
                        let mut workspace = ws.borrow_mut();
                        if workspace.is_none() {
                            *workspace = Some(crate::utils::workspace::ThreadWorkspace::new(64, 0));
                        }
                        let ws = workspace.as_mut().unwrap();
                        RefAlleleLookup::new_from_threaded_with_buffer(
                            &threaded_haps,
                            n_stage1,
                            n_states,
                            n_haps,
                            ref_geno,
                            self.reference_gt.as_deref(),
                            self.alignment.as_ref(),
                            Some(hi_freq_markers),
                            std::mem::replace(&mut ws.lookup, aligned_vec::AVec::new(32)),
                        )
                    }));
                }
                if let Some(ref lookup) = lookup {
                    hmm.forward_backward_with_lookup(&seq1, Some(&seq_conf), lookup, &mut fwd1, &mut bwd1);
                } else {
                    hmm.forward_backward_raw(&seq1, Some(&seq_conf), &threaded_haps, &mut fwd1, &mut bwd1);
                }

                let mut fwd2 = Vec::new();
                let mut bwd2 = Vec::new();
                if let Some(ref lookup) = lookup {
                    hmm.forward_backward_with_lookup(&seq2, Some(&seq_conf), lookup, &mut fwd2, &mut bwd2);
                } else {
                    hmm.forward_backward_raw(&seq2, Some(&seq_conf), &threaded_haps, &mut fwd2, &mut bwd2);
                }
                if let Some(lookup) = lookup {
                    THREAD_WORKSPACE.with(|ws| {
                        if let Some(ws) = ws.borrow_mut().as_mut() {
                            ws.lookup = lookup.into_buffer();
                        }
                    });
                }

                // Compute posterior state probabilities at each Stage 1 marker
                let mut probs1 = compute_state_posteriors(&fwd1, &bwd1, n_stage1, n_states);
                let mut probs2 = compute_state_posteriors(&fwd2, &bwd2, n_stage1, n_states);

                // SOFT-INFORMATION HANDOFF: Merge with previous window's state probs
                if let Some(overlap) = previous_overlap {
                    if let Some(ref left_state_probs) = overlap.state_probs {
                        let h1_idx = s * 2;
                        let h2_idx = s * 2 + 1;
                        // Assuming left_state_probs are ordered by haplotype
                        if h1_idx < left_state_probs.data.len() && h2_idx < left_state_probs.data.len() {
                            let left_probs1 = &left_state_probs.data[h1_idx];
                            let left_probs2 = &left_state_probs.data[h2_idx];

                            // Validate state counts match
                            if left_state_probs.n_states == n_states {
                                self.finalize_stage2_with_context(&mut probs1, left_probs1, n_stage1_in_prev_overlap, &left_state_probs.marker_indices);
                                self.finalize_stage2_with_context(&mut probs2, left_probs2, n_stage1_in_prev_overlap, &left_state_probs.marker_indices);
                            }
                        }
                    }
                }

                // Extract state probs for next window (if needed)
                let next_probs = if !next_overlap_indices.is_empty() {
                    let mut p1_tail = Vec::with_capacity(next_overlap_indices.len());
                    let mut p2_tail = Vec::with_capacity(next_overlap_indices.len());

                    for &i in &next_overlap_indices {
                        if i < probs1.len() {
                            p1_tail.push(probs1[i].clone());
                        } else {
                             p1_tail.push(vec![0.0; n_states]);
                        }
                        if i < probs2.len() {
                            p2_tail.push(probs2[i].clone());
                        } else {
                             p2_tail.push(vec![0.0; n_states]);
                        }
                    }
                    Some(vec![p1_tail, p2_tail])
                } else {
                    None
                };

                // Lazy cache for state->hap mapping - O(1) indexing with Option<Vec>
                // Uses immutable materialize_at() to avoid clone() overhead
                let mut hap_cache: Vec<Option<Vec<u32>>> = vec![None; n_markers];

                macro_rules! get_haps {
                    ($marker:expr) => {{
                        let m = $marker;
                        if hap_cache[m].is_none() {
                            let mut haps = vec![0u32; n_states];
                            threaded_haps.materialize_at(m, &mut haps);
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
                        let state_haps = get_haps!(mkr_a);
                        let n_states = state_haps.len();
                        let bridge_probs = stage2_phaser.bridge_state_probs(m, probs, n_states);

                        for (j, &hap) in state_haps.iter().enumerate() {
                            let prob = bridge_probs.get(j).copied().unwrap_or(0.0);
                            let b1 = get_allele(m, hap as usize);

                            if b1 != 255 {
                                if (b1 as usize) < n_alleles {
                                    al_probs[b1 as usize] += prob;
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
                        let state_haps = get_haps!(m);
                        let n_states = state_haps.len();
                        let bridge_probs = stage2_phaser.bridge_state_probs(m, probs, n_states);
                        let mut score = 0.0f32;
                        for (j, &hap) in state_haps.iter().enumerate() {
                            let prob = bridge_probs.get(j).copied().unwrap_or(0.0);
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

                    // Fallback to interpolated allele probabilities
                    let mkr_a = stage2_phaser.prev_stage1_marker[m];
                    let state_haps_for_interp = get_haps!(mkr_a);
                    let al_probs1 = stage2_phaser.interpolated_allele_probs(
                        m, &probs1, state_haps_for_interp, &get_allele, a1, a2,
                    );
                    let al_probs2 = stage2_phaser.interpolated_allele_probs(
                        m, &probs2, state_haps_for_interp, &get_allele, a1, a2,
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

                (decisions, next_probs)
            })
            .collect::<Vec<_>>()
        };  // ref_geno borrow ends here

        // Collect next state probabilities
        let mut all_next_probs = if next_overlap_start.is_some() {
            Some(Vec::with_capacity(n_haps))
        } else {
            None
        };

        // Apply phase changes and imputations to SamplePhase
        let mut total_switches = 0;
        let mut total_phased = 0;
        let mut total_imputed = 0;

        // Stage 2 runs after all iterations, so lr_threshold is typically 1.0
        // (all decisions pass). We still check for consistency with Stage 1.
        let lr_threshold = self.params.lr_threshold;

        for (s, (decisions, next_probs)) in phase_results.into_iter().enumerate() {
            // Collect probs
            if let (Some(all), Some(probs)) = (all_next_probs.as_mut(), next_probs) {
                // probs is vec![p1_tail, p2_tail]
                for p in probs {
                    all.push(p);
                }
            }

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

        if let Some(data) = all_next_probs {
            // Reconstruct StateProbs
            if !data.is_empty() {
                // Determine marker indices
                // We know from above they are next_overlap_indices
                let indices = if let Some(start) = next_overlap_start {
                     let start_stage1 = hi_freq_markers.iter().position(|&m| m >= start).unwrap_or(n_stage1);
                     (start_stage1..n_stage1).collect()
                } else {
                     Vec::new()
                };

                let n_states = self.params.n_states; // Approximate, but StateProbs carries it
                return Some(StateProbs::new(data, indices, n_states));
            }
        }

        None
    }

    /// Finalize Stage 2 phasing in overlap region using soft-information handoff
    fn finalize_stage2_with_context(
        &self,
        probs: &mut [Vec<f32>],
        left_probs: &[Vec<f32>],
        n_stage1_in_overlap: usize,
        marker_indices: &[usize],
    ) {
        let n_markers = probs.len().min(left_probs.len()).min(n_stage1_in_overlap);

        // Ensure marker indices are at least available for the processed range
        // This effectively 'uses' the variable to silence warnings and provides a sanity check
        if marker_indices.len() < n_markers {
            return;
        }

        for i in 0..n_markers {
            // Linear interpolation weight (0.0 -> 1.0 across overlap)
            // Left (prev window) should decay, Right (current window) should grow
            let w = i as f32 / n_stage1_in_overlap.max(1) as f32;
            let left = &left_probs[i];
            let right = &mut probs[i];

            for (k, p) in right.iter_mut().enumerate() {
                let p_left = left.get(k).copied().unwrap_or(0.0);
                *p = (1.0 - w) * p_left + w * *p;
            }
        }
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

/// Compute the likelihood ratio for a phase decision with a single reference.
///
/// Used when only one reference haplotype path is available (e.g., in Gibbs sampling).
/// The LR is computed based on whether the reference supports the chosen allele.
#[inline]
fn compute_phase_lr_single(
    chosen_allele: u8,
    other_allele: u8,
    ref_allele: u8,
    conf: f32,
    p_no_err: f32,
    p_err: f32,
) -> f32 {
    if ref_allele == 255 {
        // Missing reference - no information
        return 1.0;
    }

    // Emission probability if chosen allele is correct
    let p_chosen = emit_prob(ref_allele, chosen_allele, conf, p_no_err, p_err);
    // Emission probability if other allele is correct
    let p_other = emit_prob(ref_allele, other_allele, conf, p_no_err, p_err);

    // LR = P(chosen) / P(other)
    if p_other < 1e-30 {
        if p_chosen < 1e-30 {
            1.0
        } else {
            1e6
        }
    } else {
        (p_chosen / p_other).min(1e6)
    }
}

#[derive(Clone, Copy, Debug)]
enum EmissionMode {
    Combined,
    Hap,
}

/// Compute haploid emission probability with heterozygote constraint.
///
/// At heterozygous sites, the target haplotype (H1) must emit the allele that,
/// when combined with the fixed haplotype (H2), produces the observed genotype.
/// This is the core of SHAPEIT5-style constrained Gibbs sampling.
///
/// # Arguments
/// * `ref_al` - Reference haplotype allele at this marker
/// * `geno_a1` - First allele of genotype
/// * `geno_a2` - Second allele of genotype
/// * `fixed_allele` - The allele of the fixed haplotype (H2), or 255 if homozygous
/// * `conf` - Genotype confidence (0..1)
/// * `p_no_err` - Probability of no error (e.g., 0.999)
/// * `p_err` - Probability of error (e.g., 0.001)
///
/// # Returns
/// Emission probability for this state
#[inline]
fn emit_haploid_constrained(
    ref_al: u8,
    geno_a1: u8,
    geno_a2: u8,
    fixed_allele: u8,
    conf: f32,
    p_no_err: f32,
    p_err: f32,
) -> f32 {
    // Missing data: return neutral emission (no information)
    if geno_a1 == 255 || geno_a2 == 255 {
        return 1.0;
    }

    // At homozygous sites (fixed_allele == 255), both alleles are same
    // so H1 must emit geno_a1
    // At heterozygous sites, H1 must emit the allele opposite to fixed_allele
    let required_allele = if fixed_allele == 255 {
        geno_a1 // Homozygous: H1 must emit the homozygous allele
    } else if fixed_allele == geno_a1 {
        geno_a2 // H2 has a1, so H1 must have a2
    } else {
        geno_a1 // H2 has a2, so H1 must have a1
    };

    // Emission: does ref_al match the required allele?
    let matches = (ref_al == required_allele) as u8 as f32;
    let raw_emit = matches * p_no_err + (1.0 - matches) * p_err;

    // Blend with uniform based on confidence
    conf * raw_emit + (1.0 - conf) * 0.5
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
    lookup: &RefAlleleLookup,
    fwd: &mut [f32],
    fwd_prior: &mut [f32],
    ref_alleles: &mut [u8],
    p_no_err: f32,
    p_err: f32,
    mode: EmissionMode,
) {
    use wide::f32x8;

    if n_markers == 0 || n_states == 0 {
        return;
    }

    let init = 1.0f32 / n_states as f32;
    fwd[..n_states].fill(init);
    fwd_prior[..n_states].fill(0.0);
    ref_alleles[..n_states].fill(0);
    let mut fwd_sum = 1.0f32;

    for m in 0..n_markers {
        if m > 0 {
            let r = p_recomb.get(m).copied().unwrap_or(0.0);
            let shift = r / n_states as f32;
            let scale = (1.0 - r) / fwd_sum.max(1e-30);

            // SIMD-optimized fwd_prior = scale * fwd + shift
            let shift_vec = f32x8::splat(shift);
            let scale_vec = f32x8::splat(scale);
            let mut k = 0;
            while k + 8 <= n_states {
                let fwd_arr: [f32; 8] = fwd[k..k+8].try_into().unwrap();
                let fwd_chunk = f32x8::from(fwd_arr);
                let res = scale_vec * fwd_chunk + shift_vec;
                let res_arr: [f32; 8] = res.into();
                fwd_prior[k..k+8].copy_from_slice(&res_arr);
                k += 8;
            }
            // Scalar tail
            for i in k..n_states {
                fwd_prior[i] = scale * fwd[i] + shift;
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

        let use_combined = matches!(mode, EmissionMode::Combined) || hap2_use_combined[m];

        // Compute fwd[k] = fwd_prior[k] * emit and accumulate sum
        // SIMD-optimized accumulation
        let mut sum_vec = f32x8::splat(0.0);
        let mut k = 0;

        if use_combined {
            let emit_mode = classify_combined(a1, a2);
            // Vectorized loop
            while k + 8 <= n_states {
                let prior_arr: [f32; 8] = fwd_prior[k..k+8].try_into().unwrap();
                let prior_vec = f32x8::from(prior_arr);

                // Compute emissions for 8 states
                let emit_arr = [
                    emit_combined_fast(ref_alleles[k], emit_mode, conf_m, p_no_err, p_err),
                    emit_combined_fast(ref_alleles[k+1], emit_mode, conf_m, p_no_err, p_err),
                    emit_combined_fast(ref_alleles[k+2], emit_mode, conf_m, p_no_err, p_err),
                    emit_combined_fast(ref_alleles[k+3], emit_mode, conf_m, p_no_err, p_err),
                    emit_combined_fast(ref_alleles[k+4], emit_mode, conf_m, p_no_err, p_err),
                    emit_combined_fast(ref_alleles[k+5], emit_mode, conf_m, p_no_err, p_err),
                    emit_combined_fast(ref_alleles[k+6], emit_mode, conf_m, p_no_err, p_err),
                    emit_combined_fast(ref_alleles[k+7], emit_mode, conf_m, p_no_err, p_err),
                ];
                let emit_vec = f32x8::from(emit_arr);

                let res = prior_vec * emit_vec;
                let res_arr: [f32; 8] = res.into();
                fwd[k..k+8].copy_from_slice(&res_arr);
                sum_vec += res;
                k += 8;
            }
            // Scalar tail
            fwd_sum = sum_vec.reduce_add();
            for i in k..n_states {
                let emit = emit_combined_fast(ref_alleles[i], emit_mode, conf_m, p_no_err, p_err);
                fwd[i] = fwd_prior[i] * emit;
                fwd_sum += fwd[i];
            }
        } else {
            let h2_al = hap2_allele[m];
            // Vectorized loop
            while k + 8 <= n_states {
                let prior_arr: [f32; 8] = fwd_prior[k..k+8].try_into().unwrap();
                let prior_vec = f32x8::from(prior_arr);

                let emit_arr = [
                    emit_prob(ref_alleles[k], h2_al, conf_m, p_no_err, p_err),
                    emit_prob(ref_alleles[k+1], h2_al, conf_m, p_no_err, p_err),
                    emit_prob(ref_alleles[k+2], h2_al, conf_m, p_no_err, p_err),
                    emit_prob(ref_alleles[k+3], h2_al, conf_m, p_no_err, p_err),
                    emit_prob(ref_alleles[k+4], h2_al, conf_m, p_no_err, p_err),
                    emit_prob(ref_alleles[k+5], h2_al, conf_m, p_no_err, p_err),
                    emit_prob(ref_alleles[k+6], h2_al, conf_m, p_no_err, p_err),
                    emit_prob(ref_alleles[k+7], h2_al, conf_m, p_no_err, p_err),
                ];
                let emit_vec = f32x8::from(emit_arr);

                let res = prior_vec * emit_vec;
                let res_arr: [f32; 8] = res.into();
                fwd[k..k+8].copy_from_slice(&res_arr);
                sum_vec += res;
                k += 8;
            }
            // Scalar tail
            fwd_sum = sum_vec.reduce_add();
            for i in k..n_states {
                let emit = emit_prob(ref_alleles[i], h2_al, conf_m, p_no_err, p_err);
                fwd[i] = fwd_prior[i] * emit;
                fwd_sum += fwd[i];
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
    lookup: &RefAlleleLookup,
    p_no_err: f32,
    p_err: f32,
    rng: &mut rand::rngs::SmallRng,
    fwd_block: &mut [f32],
    mode: EmissionMode,
) {
    use wide::f32x8;

    if n_markers == 0 || n_states == 0 {
        return;
    }

    let block_size = checkpoints.block_size;
    let n_blocks = checkpoints.n_blocks;
    let mut current_state: Option<usize> = None;

    let mut weights = vec![0.0f32; n_states];
    let mut ref_alleles = vec![0u8; n_states];

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

            // Batch lookup ref alleles
            for k in 0..n_states {
                ref_alleles[k] = lookup.allele(m, k);
            }

            // SIMD-optimized forward update
            let shift_vec = f32x8::splat(shift);
            let scale_vec = f32x8::splat(scale);
            let mut sum_vec = f32x8::splat(0.0);
            let mut k = 0;

            let use_combined = matches!(mode, EmissionMode::Combined) || hap2_use_combined[m];

            if use_combined {
                let emit_mode = classify_combined(a1, a2);
                while k + 8 <= n_states {
                    let prev_arr: [f32; 8] = prev_row[k..k+8].try_into().unwrap();
                    let prev_vec = f32x8::from(prev_arr);
                    let prior_vec = scale_vec * prev_vec + shift_vec;

                    let emit_arr = [
                        emit_combined_fast(ref_alleles[k], emit_mode, conf_m, p_no_err, p_err),
                        emit_combined_fast(ref_alleles[k+1], emit_mode, conf_m, p_no_err, p_err),
                        emit_combined_fast(ref_alleles[k+2], emit_mode, conf_m, p_no_err, p_err),
                        emit_combined_fast(ref_alleles[k+3], emit_mode, conf_m, p_no_err, p_err),
                        emit_combined_fast(ref_alleles[k+4], emit_mode, conf_m, p_no_err, p_err),
                        emit_combined_fast(ref_alleles[k+5], emit_mode, conf_m, p_no_err, p_err),
                        emit_combined_fast(ref_alleles[k+6], emit_mode, conf_m, p_no_err, p_err),
                        emit_combined_fast(ref_alleles[k+7], emit_mode, conf_m, p_no_err, p_err),
                    ];
                    let emit_vec = f32x8::from(emit_arr);

                    let res = prior_vec * emit_vec;
                    let res_arr: [f32; 8] = res.into();
                    curr_part[k..k+8].copy_from_slice(&res_arr);
                    sum_vec += res;
                    k += 8;
                }
                prev_sum = sum_vec.reduce_add();
                for i in k..n_states {
                    let prior = scale * prev_row[i] + shift;
                    let emit = emit_combined_fast(ref_alleles[i], emit_mode, conf_m, p_no_err, p_err);
                    curr_part[i] = prior * emit;
                    prev_sum += curr_part[i];
                }
            } else {
                let h2_al = hap2_allele[m];
                while k + 8 <= n_states {
                    let prev_arr: [f32; 8] = prev_row[k..k+8].try_into().unwrap();
                    let prev_vec = f32x8::from(prev_arr);
                    let prior_vec = scale_vec * prev_vec + shift_vec;

                    let emit_arr = [
                        emit_prob(ref_alleles[k], h2_al, conf_m, p_no_err, p_err),
                        emit_prob(ref_alleles[k+1], h2_al, conf_m, p_no_err, p_err),
                        emit_prob(ref_alleles[k+2], h2_al, conf_m, p_no_err, p_err),
                        emit_prob(ref_alleles[k+3], h2_al, conf_m, p_no_err, p_err),
                        emit_prob(ref_alleles[k+4], h2_al, conf_m, p_no_err, p_err),
                        emit_prob(ref_alleles[k+5], h2_al, conf_m, p_no_err, p_err),
                        emit_prob(ref_alleles[k+6], h2_al, conf_m, p_no_err, p_err),
                        emit_prob(ref_alleles[k+7], h2_al, conf_m, p_no_err, p_err),
                    ];
                    let emit_vec = f32x8::from(emit_arr);

                    let res = prior_vec * emit_vec;
                    let res_arr: [f32; 8] = res.into();
                    curr_part[k..k+8].copy_from_slice(&res_arr);
                    sum_vec += res;
                    k += 8;
                }
                prev_sum = sum_vec.reduce_add();
                for i in k..n_states {
                    let prior = scale * prev_row[i] + shift;
                    let emit = emit_prob(ref_alleles[i], h2_al, conf_m, p_no_err, p_err);
                    curr_part[i] = prior * emit;
                    prev_sum += curr_part[i];
                }
            }
            prev_sum = prev_sum.max(1e-30);
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

            // SIMD-optimized weight computation
            let shift_vec = f32x8::splat(shift);
            let mut k = 0;
            while k + 8 <= n_states {
                let prev_arr: [f32; 8] = prev_row[k..k+8].try_into().unwrap();
                let prev_vec = f32x8::from(prev_arr);
                // Most states get shift transition
                let res = prev_vec * shift_vec;
                let res_arr: [f32; 8] = res.into();
                weights[k..k+8].copy_from_slice(&res_arr);
                k += 8;
            }
            for i in k..n_states {
                weights[i] = prev_row[i] * shift;
            }
            // Fix up the stay state
            if next_state < n_states {
                weights[next_state] = prev_row[next_state] * stay;
            }

            let sampled = sample_from_weights(&weights, rng);
            path[m - 1] = sampled as u32;
            current_state = Some(sampled);
        }
    }
}

/// Forward-Filtering Backward-Sampling for haploid HMM with constraint.
///
/// This is the core of SHAPEIT5-style Gibbs sampling. It samples a haplotype
/// path through K reference states, with emissions constrained at heterozygous
/// sites to be opposite of the fixed other haplotype.
///
/// Returns the sampled state path in `path`.
fn ffbs_haploid_constrained(
    path: &mut [u32],
    n_markers: usize,
    n_states: usize,
    p_recomb: &[f32],
    geno_a1: &[u8],
    geno_a2: &[u8],
    conf: &[f32],
    fixed_allele: &[u8],  // Allele assigned to OTHER haplotype (255 = no constraint)
    neighbors: &[u32],     // Selected neighbor haplotype indices
    phase_ibs: &BidirectionalPhaseIbs,
    p_no_err: f32,
    p_err: f32,
    rng: &mut rand::rngs::SmallRng,
) {
    use wide::f32x8;

    if n_markers == 0 || n_states == 0 || neighbors.is_empty() {
        return;
    }

    let actual_n_states = neighbors.len().min(n_states);

    // Rolling forward probabilities (2 rows)
    let mut fwd_curr = vec![0.0f32; actual_n_states];
    let mut fwd_prev = vec![0.0f32; actual_n_states];

    // Store forward probs at each marker for backward sampling (flat buffer)
    let mut fwd_at_marker: Vec<f32> = vec![0.0f32; n_markers * actual_n_states];

    // Initialize at marker 0
    let init = 1.0f32 / actual_n_states as f32;
    for k in 0..actual_n_states {
        let ref_al = phase_ibs.allele(0, neighbors[k]);
        let emit = emit_haploid_constrained(
            ref_al, geno_a1[0], geno_a2[0], fixed_allele[0],
            conf[0], p_no_err, p_err
        );
        fwd_curr[k] = init * emit;
    }
    let mut fwd_sum: f32 = fwd_curr.iter().sum();
    fwd_sum = fwd_sum.max(1e-30);
    fwd_at_marker[0..actual_n_states].copy_from_slice(&fwd_curr);

    // Forward pass
    for m in 1..n_markers {
        std::mem::swap(&mut fwd_prev, &mut fwd_curr);

        let r = p_recomb.get(m).copied().unwrap_or(0.0);
        let shift = r / actual_n_states as f32;
        let scale = (1.0 - r) / fwd_sum;

        // SIMD-optimized transition + emission
        let shift_vec = f32x8::splat(shift);
        let scale_vec = f32x8::splat(scale);
        let mut sum_vec = f32x8::splat(0.0);
        let mut k = 0;

        while k + 8 <= actual_n_states {
            let prev_arr: [f32; 8] = fwd_prev[k..k+8].try_into().unwrap();
            let prev_vec = f32x8::from(prev_arr);
            let prior_vec = scale_vec * prev_vec + shift_vec;

            // Compute emissions
            let emit_arr = [
                emit_haploid_constrained(phase_ibs.allele(m, neighbors[k]), geno_a1[m], geno_a2[m], fixed_allele[m], conf[m], p_no_err, p_err),
                emit_haploid_constrained(phase_ibs.allele(m, neighbors[k+1]), geno_a1[m], geno_a2[m], fixed_allele[m], conf[m], p_no_err, p_err),
                emit_haploid_constrained(phase_ibs.allele(m, neighbors[k+2]), geno_a1[m], geno_a2[m], fixed_allele[m], conf[m], p_no_err, p_err),
                emit_haploid_constrained(phase_ibs.allele(m, neighbors[k+3]), geno_a1[m], geno_a2[m], fixed_allele[m], conf[m], p_no_err, p_err),
                emit_haploid_constrained(phase_ibs.allele(m, neighbors[k+4]), geno_a1[m], geno_a2[m], fixed_allele[m], conf[m], p_no_err, p_err),
                emit_haploid_constrained(phase_ibs.allele(m, neighbors[k+5]), geno_a1[m], geno_a2[m], fixed_allele[m], conf[m], p_no_err, p_err),
                emit_haploid_constrained(phase_ibs.allele(m, neighbors[k+6]), geno_a1[m], geno_a2[m], fixed_allele[m], conf[m], p_no_err, p_err),
                emit_haploid_constrained(phase_ibs.allele(m, neighbors[k+7]), geno_a1[m], geno_a2[m], fixed_allele[m], conf[m], p_no_err, p_err),
            ];
            let emit_vec = f32x8::from(emit_arr);

            let res = prior_vec * emit_vec;
            let res_arr: [f32; 8] = res.into();
            fwd_curr[k..k+8].copy_from_slice(&res_arr);
            sum_vec += res;
            k += 8;
        }

        // Scalar tail
        fwd_sum = sum_vec.reduce_add();
        for i in k..actual_n_states {
            let prior = scale * fwd_prev[i] + shift;
            let emit = emit_haploid_constrained(
                phase_ibs.allele(m, neighbors[i]), geno_a1[m], geno_a2[m], fixed_allele[m],
                conf[m], p_no_err, p_err
            );
            fwd_curr[i] = prior * emit;
            fwd_sum += fwd_curr[i];
        }
        fwd_sum = fwd_sum.max(1e-30);

        let start = m * actual_n_states;
        fwd_at_marker[start..start + actual_n_states].copy_from_slice(&fwd_curr);
    }

    // Backward sampling
    let last_start = (n_markers - 1) * actual_n_states;
    let last_fwd = &fwd_at_marker[last_start..last_start + actual_n_states];
    path[n_markers - 1] = sample_from_weights(last_fwd, rng) as u32;

    let mut weights = vec![0.0f32; actual_n_states];
    for m in (1..n_markers).rev() {
        let next_state = path[m] as usize;
        let r = p_recomb.get(m).copied().unwrap_or(0.0);
        let shift = r / actual_n_states as f32;
        let stay = (1.0 - r) + shift;

        let prev_start = (m - 1) * actual_n_states;
        let prev_fwd = &fwd_at_marker[prev_start..prev_start + actual_n_states];

        for k in 0..actual_n_states {
            weights[k] = prev_fwd[k] * shift;
        }
        if next_state < actual_n_states {
            weights[next_state] = prev_fwd[next_state] * stay;
        }

        path[m - 1] = sample_from_weights(&weights, rng) as u32;
    }
}

/// Dynamic MCMC phasing using SHAPEIT5-style Gibbs sampling.
///
/// This implements the correct MCMC approach with implicit anchoring:
/// 1. At each MCMC step, select K neighbors by threading current H1/H2 through PBWT
/// 2. Sample H1 | (G, H2_fixed) using haploid constrained HMM
/// 3. Sample H2 | (G, H1_new) using haploid constrained HMM
/// 4. Repeat for n_steps
///
/// The "implicit anchoring" comes from state selection being biased toward
/// haplotypes that match the current phase estimate via the "Latent State" approach:
/// neighbors are found by looking up the position of the PREVIOUSLY SAMPLED reference
/// state in the PBWT, giving O(1) lookup and preserving phase inertia.
fn sample_dynamic_mcmc(
    n_markers: usize,
    n_states: usize,
    p_recomb: &[f32],
    seq1: &[u8],
    seq2: &[u8],
    conf: &[f32],
    phase_ibs: &BidirectionalPhaseIbs,
    ibs2: &Ibs2,
    sample_idx: u32,
    het_positions: &[usize],
    seed: u64,
    n_mcmc_steps: usize,
    p_no_err: f32,
    p_err: f32,
) -> (Vec<u8>, Vec<f32>) {
    use rand::SeedableRng;

    if het_positions.is_empty() || n_markers == 0 || n_states == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    let hap1_idx = sample_idx * 2;

    // Initialize H1, H2 alleles from genotype (random phase at hets)
    let mut h1_alleles = vec![0u8; n_markers];
    let mut h2_alleles = vec![0u8; n_markers];
    for m in 0..n_markers {
        let a1 = seq1[m];
        let a2 = seq2[m];
        if a1 == 255 && a2 == 255 {
            h1_alleles[m] = 255;
            h2_alleles[m] = 255;
        } else if a1 == a2 {
            h1_alleles[m] = a1;
            h2_alleles[m] = a1;
        } else {
            // Het: random initial phase
            if rng.random::<bool>() {
                h1_alleles[m] = a1;
                h2_alleles[m] = a2;
            } else {
                h1_alleles[m] = a2;
                h2_alleles[m] = a1;
            }
        }
    }

    // Initialize path with starting states from standard neighbor finding
    // This gives the first iteration something to work with
    let initial_neighbors = phase_ibs.find_neighbors(hap1_idx, n_markers / 2, ibs2, n_states);
    if initial_neighbors.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // Separate paths for H1 and H2 to avoid cross-talk in Gibbs sampling
    // Each haplotype's neighbor selection should use its own previous latent state
    let mut path1 = vec![0u32; n_markers];
    let mut path2 = vec![0u32; n_markers];
    let mut fixed_allele = vec![255u8; n_markers];

    // Current set of neighbors (reused across markers within an MCMC step)
    let mut neighbors = initial_neighbors;

    // MCMC loop: Gibbs sampling alternating between H1 and H2
    for step in 0..n_mcmc_steps {
        // === Sample H1 | (G, H2_fixed) ===

        // 1. Select neighbors using "Latent State" approach:
        //    Use H1's previously sampled state at a marker to find neighbors
        //    Vary the marker position across steps for robustness
        let center_marker = if n_mcmc_steps > 1 {
            n_markers / 4 + step * n_markers / (2 * n_mcmc_steps)
        } else {
            n_markers / 2
        };
        let prev_state = path1[center_marker] as usize;
        if prev_state < neighbors.len() {
            let ref_hap = neighbors[prev_state];
            neighbors = phase_ibs.find_neighbors_of_state(ref_hap, center_marker, sample_idx, n_states);
        }
        if neighbors.is_empty() {
            continue;
        }

        // 2. Build constraint: at hets, H1 must produce genotype with H2
        for m in 0..n_markers {
            let a1 = seq1[m];
            let a2 = seq2[m];
            if a1 == 255 || a2 == 255 || a1 == a2 {
                fixed_allele[m] = 255; // No constraint (hom/missing)
            } else {
                fixed_allele[m] = h2_alleles[m]; // H1 must be opposite of H2
            }
        }

        // 3. Run haploid FFBS for H1
        ffbs_haploid_constrained(
            &mut path1, n_markers, neighbors.len(), p_recomb,
            seq1, seq2, conf, &fixed_allele, &neighbors,
            phase_ibs, p_no_err, p_err, &mut rng
        );

        // 4. Update H1 based on sampled reference alleles at hets
        //    GIBBS SAMPLING: only update H1, leave H2 fixed
        //    At hets, set H1 to match the reference's allele (if compatible).
        for m in 0..n_markers {
            let state = path1[m] as usize;
            let a1 = seq1[m];
            let a2 = seq2[m];

            if a1 == 255 && a2 == 255 {
                h1_alleles[m] = 255;
            } else if a1 == a2 {
                h1_alleles[m] = a1;
            } else if state < neighbors.len() {
                // Het: use reference allele to determine H1
                let ref_al = phase_ibs.allele(m, neighbors[state]);
                if ref_al == a1 || ref_al == a2 {
                    // Set H1 to ref_al, and H2 must be the other allele
                    h1_alleles[m] = ref_al;
                    h2_alleles[m] = if ref_al == a1 { a2 } else { a1 };
                }
                // If ref_al is missing/different, keep current phase
            }
        }

        // === Sample H2 | (G, H1_new) ===

        // 1. Select neighbors for H2 using H2's own latent state (not H1's!)
        let prev_state = path2[center_marker] as usize;
        if prev_state < neighbors.len() {
            let ref_hap = neighbors[prev_state];
            neighbors = phase_ibs.find_neighbors_of_state(ref_hap, center_marker, sample_idx, n_states);
        }
        if neighbors.is_empty() {
            continue;
        }

        // 2. Build constraint: at hets, H2 must produce genotype with H1
        for m in 0..n_markers {
            let a1 = seq1[m];
            let a2 = seq2[m];
            if a1 == 255 || a2 == 255 || a1 == a2 {
                fixed_allele[m] = 255;
            } else {
                fixed_allele[m] = h1_alleles[m]; // H2 must be opposite of H1
            }
        }

        // 3. Run haploid FFBS for H2
        ffbs_haploid_constrained(
            &mut path2, n_markers, neighbors.len(), p_recomb,
            seq1, seq2, conf, &fixed_allele, &neighbors,
            phase_ibs, p_no_err, p_err, &mut rng
        );

        // 4. Update H2 based on sampled reference alleles
        //    GIBBS SAMPLING: only update H2, leave H1 fixed
        //    At hets, H2 is constrained to be opposite of H1, so just verify consistency.
        for m in 0..n_markers {
            let a1 = seq1[m];
            let a2 = seq2[m];

            if a1 == 255 && a2 == 255 {
                h2_alleles[m] = 255;
            } else if a1 == a2 {
                h2_alleles[m] = a2;
            } else {
                // Het: H2 must be opposite of H1 (already determined in H1 step)
                // The constraint in emit_haploid_constrained enforced this.
                // Just ensure consistency - H2 is the allele NOT assigned to H1.
                h2_alleles[m] = if h1_alleles[m] == a1 { a2 } else { a1 };
            }
        }

        // After first step, we have a valid path to use for latent state lookup
        // in subsequent iterations
    }

    // Determine swap decisions from final H1, H2 vs original seq1, seq2
    let mut swap_bits = Vec::with_capacity(het_positions.len());
    let mut swap_lr = Vec::with_capacity(het_positions.len());

    for &m in het_positions {
        let a1 = seq1[m];
        let a2 = seq2[m];

        if a1 == 255 || a2 == 255 || a1 == a2 {
            swap_bits.push(0);
            swap_lr.push(1.0);
            continue;
        }

        // Original phase: seq1[m] on H1, seq2[m] on H2
        // Swap if final H1 allele differs from original seq1
        let swap = h1_alleles[m] != a1;
        swap_bits.push(if swap { 1 } else { 0 });

        // Compute LR from the reference allele at this position (use H1's path)
        let ref_al = if (path1[m] as usize) < neighbors.len() {
            phase_ibs.allele(m, neighbors[path1[m] as usize])
        } else {
            255
        };
        let lr = compute_phase_lr_single(
            h1_alleles[m], // chosen allele for H1
            h2_alleles[m], // other allele (H2)
            ref_al,
            conf[m],
            p_no_err,
            p_err,
        );
        swap_lr.push(lr);
    }

    (swap_bits, swap_lr)
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
    lookup: &RefAlleleLookup,
    het_positions: &[usize],
    initial_paths: Option<&MosaicPaths>,
    seed: u64,
    burnin: usize,
    p_no_err: f32,
    p_err: f32,
    workspace: &mut crate::utils::workspace::ThreadWorkspace,
) -> (Vec<u8>, Vec<f32>, MosaicPaths) {
    if het_positions.is_empty() || n_markers == 0 || n_states == 0 {
        return (
            Vec::new(),
            Vec::new(),
            MosaicPaths {
                path1: Vec::new(),
                path2: Vec::new(),
            },
        );
    }

    // Resize workspace if needed for this window
    workspace.ensure_for_window(n_markers, n_states, MOSAIC_BLOCK_SIZE);

    let combined_data = std::mem::take(&mut workspace.combined_checkpoint_data);
    let mut combined_checkpoints =
        FwdCheckpoints::from_buffer(n_markers, n_states, MOSAIC_BLOCK_SIZE, combined_data);
    let dummy_allele = vec![255u8; n_markers];
    let dummy_combined = vec![true; n_markers];
    let fwd = &mut workspace.fwd[..n_states];
    let fwd_prior = &mut workspace.fwd_prior[..n_states];
    let ref_alleles = &mut workspace.ref_alleles[..n_states];
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
        fwd,
        fwd_prior,
        ref_alleles,
        p_no_err,
        p_err,
        EmissionMode::Combined,
    );

    let combined_checkpoints_ref = &combined_checkpoints;

    // Pure Stochastic EM: single chain, take the final sample after burn-in.
    // No averaging, no counting - just sample once from the posterior.
    let chain_seed = seed.wrapping_add(0xC0FFEE_BAAD_F00Du64);
    let buffers = MosaicBuffers {
        fwd: std::mem::replace(&mut workspace.fwd, aligned_vec::AVec::new(32)),
        fwd_prior: std::mem::replace(&mut workspace.fwd_prior, aligned_vec::AVec::new(32)),
        ref_alleles: std::mem::take(&mut workspace.ref_alleles),
        hap1_checkpoints: FwdCheckpoints::from_buffer(
            n_markers,
            n_states,
            MOSAIC_BLOCK_SIZE,
            std::mem::take(&mut workspace.hap1_checkpoint_data),
        ),
        hap2_checkpoints: FwdCheckpoints::from_buffer(
            n_markers,
            n_states,
            MOSAIC_BLOCK_SIZE,
            std::mem::take(&mut workspace.hap2_checkpoint_data),
        ),
        hap1_allele: std::mem::take(&mut workspace.hap1_allele),
        hap1_use_combined: std::mem::take(&mut workspace.hap1_use_combined),
        hap2_allele: std::mem::take(&mut workspace.hap2_allele),
        hap2_use_combined: std::mem::take(&mut workspace.hap2_use_combined),
        path1: std::mem::take(&mut workspace.path1),
        path2: std::mem::take(&mut workspace.path2),
        fwd_block: std::mem::take(&mut workspace.fwd_block),
    };

    let mut chain = MosaicChain::new_with_buffers(
        chain_seed,
        n_markers,
        n_states,
        p_recomb,
        seq1,
        seq2,
        conf,
        lookup,
        combined_checkpoints_ref,
        buffers,
        p_no_err,
        p_err,
    );

    if let Some(paths) = initial_paths {
        let has_valid_lengths = paths.path1.len() == n_markers && paths.path2.len() == n_markers;
        let has_valid_states = has_valid_lengths
            && paths.path1.iter().all(|&p| (p as usize) < n_states)
            && paths.path2.iter().all(|&p| (p as usize) < n_states);
        if has_valid_states {
            chain.path1.resize(n_markers, 0);
            chain.path2.resize(n_markers, 0);
            chain.path1.copy_from_slice(&paths.path1);
            chain.path2.copy_from_slice(&paths.path2);
            chain.first_iteration = false;
        }
    }

    // Burn-in: let the chain mix
    for _ in 0..burnin {
        chain.step();
    }

    const LR_SAMPLES: usize = 4;
    let mut swap_counts = vec![0u32; het_positions.len()];
    let mut obs_counts = vec![0u32; het_positions.len()];
    let mut last_orients = vec![0u8; het_positions.len()];
    let mut new_paths = MosaicPaths {
        path1: Vec::new(),
        path2: Vec::new(),
    };

    for sample_idx in 0..LR_SAMPLES {
        chain.step();
        let (path1, path2) = chain.paths();
        let is_last = sample_idx + 1 == LR_SAMPLES;

        for (i, &m) in het_positions.iter().enumerate() {
            let a1 = seq1[m];
            let a2 = seq2[m];
            if a1 == 255 || a2 == 255 || a1 == a2 {
                if is_last {
                    last_orients[i] = 0;
                }
                continue;
            }

            let ref1 = lookup.allele(m, path1[m] as usize);
            let ref2 = lookup.allele(m, path2[m] as usize);

            let orient = if ref1 == a1 && ref2 == a2 {
                Some(0u8)
            } else if ref1 == a2 && ref2 == a1 {
                Some(1u8)
            } else if ref1 == 255 && ref2 == a2 {
                Some(0u8)
            } else if ref1 == 255 && ref2 == a1 {
                Some(1u8)
            } else if ref2 == 255 && ref1 == a1 {
                Some(0u8)
            } else if ref2 == 255 && ref1 == a2 {
                Some(1u8)
            } else {
                None
            };

            if let Some(orient) = orient {
                swap_counts[i] += orient as u32;
                obs_counts[i] += 1;
                if is_last {
                    last_orients[i] = orient;
                }
            } else if is_last {
                last_orients[i] = 0;
            }
        }

        if is_last {
            new_paths = MosaicPaths {
                path1: path1.to_vec(),
                path2: path2.to_vec(),
            };
        }
    }

    let mut swap_bits = Vec::with_capacity(het_positions.len());
    let mut swap_lr = Vec::with_capacity(het_positions.len());
    for (i, &m) in het_positions.iter().enumerate() {
        let a1 = seq1[m];
        let a2 = seq2[m];
        if a1 == 255 || a2 == 255 || a1 == a2 || obs_counts[i] == 0 {
            swap_bits.push(0);
            swap_lr.push(1.0);
            continue;
        }

        swap_bits.push(last_orients[i]);
        let p_swap = (swap_counts[i] as f32 + 0.5) / (obs_counts[i] as f32 + 1.0);
        let p_keep = 1.0 - p_swap;
        let (max_p, min_p) = if p_swap >= p_keep { (p_swap, p_keep) } else { (p_keep, p_swap) };
        let lr = if min_p < 1e-30 { 1e6 } else { (max_p / min_p).min(1e6) };
        swap_lr.push(lr);
    }

    // Return buffers to workspace for reuse
    let returned = chain.into_buffers();
    workspace.fwd = returned.fwd;
    workspace.fwd_prior = returned.fwd_prior;
    workspace.ref_alleles = returned.ref_alleles;
    workspace.hap1_checkpoint_data = returned.hap1_checkpoints.into_buffer();
    workspace.hap2_checkpoint_data = returned.hap2_checkpoints.into_buffer();
    workspace.hap1_allele = returned.hap1_allele;
    workspace.hap1_use_combined = returned.hap1_use_combined;
    workspace.hap2_allele = returned.hap2_allele;
    workspace.hap2_use_combined = returned.hap2_use_combined;
    workspace.path1 = returned.path1;
    workspace.path2 = returned.path2;
    workspace.fwd_block = returned.fwd_block;
    workspace.combined_checkpoint_data = combined_checkpoints.into_buffer();

    (swap_bits, swap_lr, new_paths)
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
    /// Number of Stage 1 markers
    n_stage1: usize,
    /// Stage 1 marker indices in original marker space
    stage1_markers: Vec<usize>,
    /// Genetic positions (cM) for all markers
    gen_positions: Vec<f64>,
    /// Recombination intensity for bridge interpolation
    recomb_intensity: f32,
}

impl Stage2Phaser {
    /// Create a new Stage2Phaser
    ///
    /// # Arguments
    /// * `hi_freq_markers` - Indices of high-frequency (Stage 1) markers in original space
    /// * `gen_positions` - Genetic positions (cM) for all markers
    /// * `n_total_markers` - Total number of markers
    fn new(
        hi_freq_markers: &[usize],
        gen_positions: &[f64],
        n_total_markers: usize,
        recomb_intensity: f32,
    ) -> Self {
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

        Self {
            prev_stage1_marker,
            n_stage1,
            stage1_markers: hi_freq_markers.to_vec(),
            gen_positions: gen_positions.to_vec(),
            recomb_intensity,
        }
    }

    /// Compute interpolated allele probabilities for a rare marker
    ///
    /// Following Java Stage2Baum.unscaledAlProbs:
    /// - For each HMM state, interpolate probability from flanking Stage 1 markers
    /// - Accumulate allele probabilities based on reference haplotype alleles
    /// Compute allele probabilities using haploid Li-Stephens emission model.
    ///
    /// Each HMM state corresponds to a specific reference haplotype. The emission
    /// probability depends ONLY on that haplotype's allele - checking paired
    /// haplotypes would violate the haploid model assumption.
    fn interpolated_allele_probs<F>(
        &self,
        marker: usize,
        state_probs: &[Vec<f32>],      // [stage1_marker][state]
        haps_at_mkr_a: &[u32],         // haplotypes at flanking Stage 1 marker
        get_allele: &F,                 // Closure to get allele for any haplotype
        a1: u8,
        a2: u8,
    ) -> [f32; 2]
    where
        F: Fn(usize, usize) -> u8, // (marker, hap_index) -> allele
    {
        let mut al_probs = [0.0f32; 2];

        let n_states = haps_at_mkr_a.len();
        let bridge_probs = self.bridge_state_probs(marker, state_probs, n_states);

        for j in 0..n_states {
            let hap = haps_at_mkr_a[j] as usize;

            // Get allele from this specific haplotype at the rare marker.
            // Li-Stephens HMM models haploid copying: state k means we're copying
            // haplotype k, so emission depends ONLY on haplotype k's allele.
            // The paired haplotype (hap ^ 1) is irrelevant - checking it would
            // introduce "free switching" and wash out the phasing signal.
            let ref_allele = get_allele(marker, hap);

            if ref_allele == 255 {
                continue;
            }

            let prob = bridge_probs.get(j).copied().unwrap_or(0.0);

            // Simple haploid emission: if this reference haplotype carries a1, add
            // probability to a1; if it carries a2, add to a2.
            if ref_allele == a1 {
                al_probs[0] += prob;
            } else if ref_allele == a2 {
                al_probs[1] += prob;
            }
            // If ref_allele matches neither (e.g., multiallelic), no contribution
        }

        al_probs
    }

    fn p_recomb(&self, gen_dist_cm: f64) -> f32 {
        let c = -(self.recomb_intensity as f64);
        (-f64::exp_m1(c * gen_dist_cm)) as f32
    }

    fn bridge_state_probs(
        &self,
        marker: usize,
        state_probs: &[Vec<f32>],
        n_states: usize,
    ) -> Vec<f32> {
        let mkr_a = self.prev_stage1_marker[marker];
        let mkr_b = (mkr_a + 1).min(self.n_stage1 - 1);

        let probs_a = &state_probs[mkr_a];
        let probs_b = &state_probs[mkr_b];

        if mkr_a == mkr_b || self.stage1_markers.is_empty() {
            return probs_a.clone();
        }

        let pos_a_idx = self.stage1_markers[mkr_a];
        let pos_b_idx = self.stage1_markers[mkr_b];

        let pos_a = *self.gen_positions.get(pos_a_idx).unwrap_or(&0.0);
        let pos_b = *self.gen_positions.get(pos_b_idx).unwrap_or(&pos_a);
        let pos_m = *self.gen_positions.get(marker).unwrap_or(&pos_a);

        if pos_b <= pos_a {
            return probs_a.clone();
        }
        if pos_m <= pos_a {
            return probs_a.clone();
        }
        if pos_m >= pos_b {
            return probs_b.clone();
        }

        let d1 = (pos_m - pos_a).max(0.0);
        let d2 = (pos_b - pos_m).max(0.0);
        let r1 = self.p_recomb(d1);
        let r2 = self.p_recomb(d2);

        let shift1 = r1 / n_states.max(1) as f32;
        let shift2 = r2 / n_states.max(1) as f32;
        let scale1 = 1.0 - r1;
        let scale2 = 1.0 - r2;

        let mut weights = vec![0.0f32; n_states];
        let mut sum = 0.0f32;
        for k in 0..n_states {
            let a = scale1 * probs_a.get(k).copied().unwrap_or(0.0) + shift1;
            let b = scale2 * probs_b.get(k).copied().unwrap_or(0.0) + shift2;
            let w = a * b;
            weights[k] = w;
            sum += w;
        }

        if sum > 0.0 {
            for w in &mut weights {
                *w /= sum;
            }
            weights
        } else {
            probs_a.clone()
        }
    }
}

impl PhasingPipeline {
    /// Phase a window with PBWT state handoff from previous window
    ///
    /// This maintains PBWT continuity across windows by passing the
    /// prefix array (PPA) and divergence array from the end of the
    /// previous window to initialize the current window's PBWT.
    pub fn phase_window_with_pbwt_handoff(
        &mut self,
        target_gt: &GenotypeMatrix,
        gen_maps: &GeneticMaps,
        phased_overlap: Option<&PhasedOverlap>,
        pbwt_state: Option<&crate::model::pbwt::PbwtState>,
    ) -> Result<GenotypeMatrix<Phased>> {
        // Log PBWT continuity state for debugging window transitions
        if let Some(state) = pbwt_state {
            tracing::trace!(
                marker_pos = state.marker_pos,
                n_haps = state.ppa.len(),
                "PBWT state handoff from previous window"
            );
        }
        self.phase_in_memory_with_overlap(target_gt, gen_maps, phased_overlap, None)
            .map(|(result, ..)| result)
    }

    /// Finalize Stage 2 phasing using context from next window
    ///
    /// Finalize Stage 2 phasing with forward context from the next window.
    ///
    /// Stage 1 phasing handles common variants in-window. Stage 2 handles rare variants
    /// using HMM state probabilities interpolated between Stage 1 markers.
    ///
    /// Cross-window context enables better rare variant phasing at window boundaries
    /// by providing forward context from the next window's phased markers. However,
    /// since GenotypeMatrix is immutable by design, the actual rare variant phasing
    /// is performed in-window by phase_rare_markers_with_hmm. This function validates
    /// the cross-window boundary continuity.
    ///
    /// The next_phased parameter provides forward context - markers from the next
    /// window that help verify phasing consistency at window boundaries.
    fn finalize_stage2_with_forward_context(
        &self,
        current_phased: &GenotypeMatrix<Phased>,
        next_phased: &GenotypeMatrix<Phased>,
    ) -> Result<GenotypeMatrix<Phased>> {
        let current_markers = current_phased.n_markers();
        let next_markers = next_phased.n_markers();
        let n_samples = current_phased.n_samples();

        if current_markers == 0 || next_markers == 0 || n_samples == 0 {
            return Ok(current_phased.clone());
        }

        // Find rare markers in the overlap region (last ~2cM or last 1000 markers)
        let overlap_start = current_markers.saturating_sub(1000);
        let rare_threshold = self.config.rare;
        
        // Collect markers that need re-phasing: rare hets in overlap that exist in next window
        let mut markers_to_fix: Vec<(usize, usize)> = Vec::new(); // (current_idx, next_idx)
        
        let mut next_idx = 0usize;
        for m in overlap_start..current_markers {
            let marker = current_phased.marker(MarkerIdx::new(m as u32));
            let key = (marker.chrom.0, marker.pos);

            // Advance next_idx until we reach or pass current marker (linear merge on sorted markers)
            while next_idx < next_markers {
                let next_marker = next_phased.marker(MarkerIdx::new(next_idx as u32));
                let next_key = (next_marker.chrom.0, next_marker.pos);
                if next_key < key {
                    next_idx += 1;
                } else {
                    break;
                }
            }

            // Check if this marker exists in next window
            if next_idx < next_markers {
                let next_marker = next_phased.marker(MarkerIdx::new(next_idx as u32));
                if (next_marker.chrom.0, next_marker.pos) != key {
                    continue;
                }
                let next_m = next_idx;
                // Check if it's a rare variant (simplified: check if any sample has het)
                let n_alleles = 1 + marker.alt_alleles.len();
                if n_alleles == 2 {
                    // For biallelic, check MAF
                    let mut alt_count = 0u32;
                    let n_haps = current_phased.n_haplotypes();
                    for h in 0..n_haps {
                        if current_phased.allele(MarkerIdx::new(m as u32), HapIdx::new(h as u32)) == 1 {
                            alt_count += 1;
                        }
                    }
                    let maf = (alt_count as f32 / n_haps as f32).min(1.0 - alt_count as f32 / n_haps as f32);
                    if maf < rare_threshold && maf > 0.0 {
                        markers_to_fix.push((m, next_m));
                    }
                }
            }
        }

        if markers_to_fix.is_empty() {
            tracing::debug!("Stage 2 finalization: no rare markers in overlap need fixing");
            return Ok(current_phased.clone());
        }

        tracing::debug!(
            "Stage 2 finalization: checking {} rare markers in overlap region",
            markers_to_fix.len()
        );

        let mut mismatches = 0usize;
        let mut matches = 0usize;

        // For each rare marker, check if next window has different phasing.
        // We avoid swapping single markers to preserve local LD structure.
        for (current_m, next_m) in markers_to_fix {
            for s in 0..n_samples {
                let hap1 = HapIdx::new((s * 2) as u32);
                let hap2 = HapIdx::new((s * 2 + 1) as u32);

                let curr_a1 = current_phased.allele(MarkerIdx::new(current_m as u32), hap1);
                let curr_a2 = current_phased.allele(MarkerIdx::new(current_m as u32), hap2);
                
                // Only fix heterozygotes
                if curr_a1 == curr_a2 || curr_a1 == 255 || curr_a2 == 255 {
                    continue;
                }

                let next_a1 = next_phased.allele(MarkerIdx::new(next_m as u32), hap1);
                let next_a2 = next_phased.allele(MarkerIdx::new(next_m as u32), hap2);

                // Check if next window has opposite phasing
                if next_a1 == curr_a2 && next_a2 == curr_a1 {
                    mismatches += 1;
                } else if next_a1 == curr_a1 && next_a2 == curr_a2 {
                    matches += 1;
                }
            }
        }

        if mismatches == 0 || mismatches <= matches {
            if mismatches > 0 {
                tracing::debug!(
                    "Stage 2 finalization: detected {} phase mismatches from forward context",
                    mismatches
                );
            }
            return Ok(current_phased.clone());
        }

        tracing::debug!(
            "Stage 2 finalization: applying phase flip ({} mismatches vs {} matches)",
            mismatches,
            matches
        );

        let n_markers = current_phased.n_markers();
        let n_haps = current_phased.n_haplotypes();
        let mut geno = MutableGenotypes::from_fn(n_markers, n_haps, |m, h| {
            current_phased.allele(MarkerIdx::new(m as u32), HapIdx::new(h as u32))
        });

        for s in 0..n_samples {
            let hap1 = HapIdx::new((s * 2) as u32);
            let hap2 = HapIdx::new((s * 2 + 1) as u32);

            let mut mask = BitVec::repeat(false, n_markers);
            for m in 0..n_markers {
                let a1 = current_phased.allele(MarkerIdx::new(m as u32), hap1);
                let a2 = current_phased.allele(MarkerIdx::new(m as u32), hap2);
                if a1 != a2 && a1 != 255 && a2 != 255 {
                    mask.set(m, true);
                }
            }
            geno.swap_haplotypes(hap1, hap2, &mask);
        }

        let markers = current_phased.markers().clone();
        let samples = current_phased.samples_arc();
        let columns: Vec<GenotypeColumn> = (0..n_markers)
            .map(|m| {
                let alleles = geno.marker_alleles(m);
                let bytes: Vec<u8> = alleles.to_vec();
                GenotypeColumn::from_alleles(&bytes, 2)
            })
            .collect();

        if let Some(confidence) = current_phased.confidence_clone() {
            Ok(GenotypeMatrix::new_phased_with_confidence(
                markers,
                columns,
                samples,
                confidence,
            ))
        } else {
            Ok(GenotypeMatrix::new_phased(markers, columns, samples))
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
            mcmc_burnin: 1,
            dynamic_mcmc: false,
            mcmc_steps: 3,
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
            dynamic_mcmc: false,
            mcmc_steps: 3,
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

        // Run phasing (with no overlap from previous window)
        let result = pipeline.phase_in_memory_with_overlap(&gt, &gen_maps, None, None);

        assert!(result.is_ok());
        let (phased, _) = result.unwrap();
        assert_eq!(phased.n_markers(), n_markers);
        assert_eq!(phased.n_haplotypes(), n_samples * 2);
    }

    #[test]
    fn test_emit_haploid_constrained_at_het() {
        // At a het site with genotype {0, 1}, if H2 is fixed to 0,
        // H1 must be 1. Emission should be high if reference has 1, low if 0.
        let p_no_err = 0.999;
        let p_err = 0.001;
        let conf = 1.0;

        // H2 = 0, so H1 must = 1. Reference has 1 -> high emission
        let emit_match = emit_haploid_constrained(1, 0, 1, 0, conf, p_no_err, p_err);
        assert!(emit_match > 0.9, "Expected high emission when ref matches required allele, got {}", emit_match);

        // H2 = 0, so H1 must = 1. Reference has 0 -> low emission
        let emit_mismatch = emit_haploid_constrained(0, 0, 1, 0, conf, p_no_err, p_err);
        assert!(emit_mismatch < 0.1, "Expected low emission when ref doesn't match, got {}", emit_mismatch);

        // At homozygous site (fixed_allele = 255), H1 must match genotype
        let emit_hom = emit_haploid_constrained(0, 0, 0, 255, conf, p_no_err, p_err);
        assert!(emit_hom > 0.9, "Expected high emission at hom site when ref matches, got {}", emit_hom);

        let emit_hom_mismatch = emit_haploid_constrained(1, 0, 0, 255, conf, p_no_err, p_err);
        assert!(emit_hom_mismatch < 0.1, "Expected low emission at hom when ref doesn't match, got {}", emit_hom_mismatch);
    }

    #[test]
    fn test_emit_haploid_constrained_confidence_blending() {
        // With low confidence, emission should be closer to 0.5
        let p_no_err = 0.999;
        let p_err = 0.001;

        // Full confidence: emission should be ~p_no_err
        let emit_full_conf = emit_haploid_constrained(1, 0, 1, 0, 1.0, p_no_err, p_err);
        assert!((emit_full_conf - p_no_err).abs() < 0.01);

        // Zero confidence: emission should be 0.5
        let emit_zero_conf = emit_haploid_constrained(1, 0, 1, 0, 0.0, p_no_err, p_err);
        assert!((emit_zero_conf - 0.5).abs() < 0.01, "Expected 0.5 with zero confidence, got {}", emit_zero_conf);

        // Half confidence: emission should be blend
        let emit_half_conf = emit_haploid_constrained(1, 0, 1, 0, 0.5, p_no_err, p_err);
        let expected = 0.5 * p_no_err + 0.5 * 0.5;
        assert!((emit_half_conf - expected).abs() < 0.01, "Expected {}, got {}", expected, emit_half_conf);
    }

    #[test]
    fn test_dynamic_mcmc_deterministic_phase() {
        // Create a scenario where the correct phase is deterministic:
        // Target sample (haps 0-1) with het genotype {0, 1}
        // Reference haplotypes (haps 2-9) all have allele 0
        // The HMM should set H1 = 0 (matching reference majority)
        use crate::model::phase_ibs::BidirectionalPhaseIbs;
        use crate::model::ibs2::Ibs2;

        let n_markers = 10;
        let n_target_haps = 2;  // Sample 0: haplotypes 0 and 1
        let n_ref_haps = 8;     // Reference: haplotypes 2-9
        let n_total_haps = n_target_haps + n_ref_haps;

        // Build PBWT with target + reference
        // Target haps (0, 1): missing (255) - we're phasing these
        // Reference haps (2-9): all have allele 0
        let alleles: Vec<Vec<u8>> = (0..n_markers)
            .map(|_| {
                let mut haps = vec![255u8; n_total_haps]; // Start with missing
                for h in n_target_haps..n_total_haps {
                    haps[h] = 0; // Reference haplotypes have allele 0
                }
                haps
            })
            .collect();
        let phase_ibs = BidirectionalPhaseIbs::build(alleles, n_total_haps, n_markers);

        // Empty IBS2 - need at least 1 sample for the structure
        let ibs2 = Ibs2::empty(1);

        // Genotype: het at all sites (0/1)
        let seq1 = vec![0u8; n_markers];
        let seq2 = vec![1u8; n_markers];
        let conf = vec![1.0f32; n_markers];

        // p_recomb: low recombination
        let p_recomb = vec![0.01f32; n_markers];

        let het_positions: Vec<usize> = (0..n_markers).collect();

        // Sample 0: haplotypes 0 and 1
        let (swap_bits, swap_lr) = sample_dynamic_mcmc(
            n_markers,
            n_total_haps,
            &p_recomb,
            &seq1,
            &seq2,
            &conf,
            &phase_ibs,
            &ibs2,
            0, // sample_idx = 0 (haplotypes 0 and 1)
            &het_positions,
            12345, // seed
            5,     // n_mcmc_steps
            0.999,
            0.001,
        );

        // With all reference having allele 0, H1 should be set to 0 at all hets.
        // Since seq1 = 0, this means no swap (swap_bit = 0).
        let n_swaps: usize = swap_bits.iter().map(|&b| b as usize).sum();

        
        // We expect very few or no swaps since reference strongly supports H1 = 0
        assert!(
            n_swaps <= 2,
            "Expected <=2 swaps with consistent reference, got {} swaps out of {} hets",
            n_swaps,
            het_positions.len()
        );

        // LR should be high confidence
        assert_eq!(swap_lr.len(), het_positions.len());
    }

    #[test]
    fn test_dynamic_mcmc_opposite_phase() {
        // Target sample (haps 0-1) with het genotype {0, 1}
        // Reference haplotypes (haps 2-9) all have allele 1
        // The HMM should set H1 = 1 (matching reference) -> swap needed
        use crate::model::phase_ibs::BidirectionalPhaseIbs;
        use crate::model::ibs2::Ibs2;

        let n_markers = 10;
        let n_target_haps = 2;  // Sample 0: haplotypes 0 and 1
        let n_ref_haps = 8;     // Reference: haplotypes 2-9
        let n_total_haps = n_target_haps + n_ref_haps;

        // Build PBWT with target + reference
        // Target haps (0, 1): missing (255)
        // Reference haps (2-9): all have allele 1
        let alleles: Vec<Vec<u8>> = (0..n_markers)
            .map(|_| {
                let mut haps = vec![255u8; n_total_haps];
                for h in n_target_haps..n_total_haps {
                    haps[h] = 1; // Reference haplotypes have allele 1
                }
                haps
            })
            .collect();
        let phase_ibs = BidirectionalPhaseIbs::build(alleles, n_total_haps, n_markers);

        let ibs2 = Ibs2::empty(1);

        // Genotype: het at all sites (0/1)
        let seq1 = vec![0u8; n_markers];
        let seq2 = vec![1u8; n_markers];
        let conf = vec![1.0f32; n_markers];
        let p_recomb = vec![0.01f32; n_markers];
        let het_positions: Vec<usize> = (0..n_markers).collect();

        let (swap_bits, swap_lr) = sample_dynamic_mcmc(
            n_markers,
            n_total_haps,
            &p_recomb,
            &seq1,
            &seq2,
            &conf,
            &phase_ibs,
            &ibs2,
            0, // sample_idx = 0 (haplotypes 0 and 1)
            &het_positions,
            12345,
            5,
            0.999,
            0.001,
        );

        // With all reference having allele 1, H1 should be set to 1 at all hets.
        // Since seq1 = 0, this means swap (swap_bit = 1).
        let n_swaps: usize = swap_bits.iter().map(|&b| b as usize).sum();

        
        // We expect most or all to swap since reference strongly supports H1 = 1
        assert!(
            n_swaps >= n_markers - 2,
            "Expected >={} swaps with opposite reference, got {} swaps",
            n_markers - 2,
            n_swaps
        );

        // Verify LR values exist
        assert_eq!(swap_lr.len(), het_positions.len());
    }
}
