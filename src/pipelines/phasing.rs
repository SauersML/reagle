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

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use bitvec::prelude::*;
use linfa::DatasetBase;
use linfa::traits::{Fit, Predict};
use linfa_clustering::KMeans;
use ndarray::{Array1, Array2};
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use tracing::{info_span, instrument};

use crate::config::Config;
use crate::data::genetic_map::{GeneticMaps, MarkerMap};
use crate::data::haplotype::{HapIdx, SampleIdx};
use crate::data::marker::{AnyMarkerSpace, MarkerIdx};
use crate::data::storage::phase_state::Phased;
use crate::data::storage::sample_phase::SamplePhase;
use crate::data::storage::{GenotypeColumn, GenotypeMatrix, GenotypeView, MutableGenotypes};
use crate::error::Result;
use crate::io::bref3::Bref3Reader;
use crate::io::streaming::{
    GlobalHapId, HaplotypePriors, PhasedOverlap, StateProbs, StreamWindow, StreamingConfig,
    StreamingVcfReader,
};
use crate::io::vcf::{VcfReader, VcfWriter};
use crate::model::ibs2::Ibs2;
use crate::model::pl_emission::{
    PlProvider, allele_probs_cond_from_pl, allele_probs_uncond_from_pl, emit_from_allele_probs,
};

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
use crate::data::alignment::{AlignmentStats, MarkerAlignment};
use crate::data::condensed::CondensedTarget;
use crate::data::ref_packed::PackedRefView;
use crate::model::beam::{ActivePool, BeamConfig, BeamPhaser, PbwtBeamIndex, PbwtInjector};
use crate::model::hmm::MosaicHmm;
use crate::model::parameters::ModelParams;
use crate::model::phase_ibs::BidirectionalPhaseIbs;
use crate::model::reference_pbwt::{
    PbwtBiallelicQueryProb, PbwtQueryAllele, RankBeam, ReferencePbwt,
};
use crate::model::state_allocator::allocate_lms_sparse;
use crate::model::states::ThreadedHaps;
use crate::model::types::{CombinedHapId, CombinedHapSpace, RefHapId, combined_from_ref};
use crate::utils::state::{StateAVec32, StateCount, StateVec};
use crate::utils::telemetry::{Stage, TelemetryBlackboard};

#[derive(Default)]
struct Stage1Timing {
    seq_extract_ns: AtomicU64,
    anchor_ns: AtomicU64,
    mcmc_ns: AtomicU64,
    total_sample_ns: AtomicU64,
    samples: AtomicU64,
    n_states_sum: AtomicU64,
    hets_sum: AtomicU64,
    anchors_sum: AtomicU64,
    last_log_ns: AtomicU64,
}

impl Stage1Timing {
    fn add_sample(
        &self,
        seq_extract: u64,
        anchor: u64,
        mcmc: u64,
        total: u64,
        n_states: u64,
        hets: u64,
        anchors: u64,
    ) -> u64 {
        self.seq_extract_ns
            .fetch_add(seq_extract, Ordering::Relaxed);
        self.anchor_ns.fetch_add(anchor, Ordering::Relaxed);
        self.mcmc_ns.fetch_add(mcmc, Ordering::Relaxed);
        self.total_sample_ns.fetch_add(total, Ordering::Relaxed);
        self.n_states_sum.fetch_add(n_states, Ordering::Relaxed);
        self.hets_sum.fetch_add(hets, Ordering::Relaxed);
        self.anchors_sum.fetch_add(anchors, Ordering::Relaxed);
        self.samples.fetch_add(1, Ordering::Relaxed) + 1
    }

    fn should_log(&self, elapsed_ns: u64) -> bool {
        const LOG_EVERY_NS: u64 = 60_000_000_000;
        let last = self.last_log_ns.load(Ordering::Relaxed);
        if elapsed_ns.saturating_sub(last) >= LOG_EVERY_NS {
            self.last_log_ns.store(elapsed_ns, Ordering::Relaxed);
            true
        } else {
            false
        }
    }
}
use mini_mcmc::core::{MarkovChain, Trace};

#[derive(Clone, Copy, Debug, Default)]
struct SampleCohortStats {
    mismatch_mass: f64,
    emission_mass: f64,
    expected_switches: f64,
    genetic_dist_morgans: f64,
    phase_uncertainty_sum: f64,
    phase_uncertainty_count: usize,
}

#[derive(Clone, Copy, Debug)]
struct MismatchProb(f32);

impl MismatchProb {
    fn new_clamped(p: f32) -> Self {
        Self(p.clamp(1e-6, 0.25))
    }

    fn get(self) -> f32 {
        self.0
    }
}

impl SampleCohortStats {
    fn mean_uncertainty(self) -> f64 {
        if self.phase_uncertainty_count == 0 {
            0.5
        } else {
            (self.phase_uncertainty_sum / self.phase_uncertainty_count as f64).clamp(0.0, 1.0)
        }
    }

    fn has_signal(self) -> bool {
        self.emission_mass > 0.0 && self.genetic_dist_morgans > 0.0
    }
}

#[derive(Debug)]
struct CohortCalibration {
    sample_p_mismatch: Vec<f32>,
    cohort_p_mismatch: Vec<f32>,
    cohort_sizes: Vec<usize>,
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let q = q.clamp(0.0, 1.0);
    let pos = q * (sorted.len().saturating_sub(1) as f64);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let w = pos - lo as f64;
        sorted[lo] * (1.0 - w) + sorted[hi] * w
    }
}

fn robust_center_scale(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 1.0);
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = percentile(&sorted, 0.5);
    let q1 = percentile(&sorted, 0.25);
    let q3 = percentile(&sorted, 0.75);
    let iqr = (q3 - q1).abs().max(1e-6);
    (median, iqr)
}

fn fit_cohort_calibration(
    stats: &[SampleCohortStats],
    global_p_mismatch: f32,
) -> Option<CohortCalibration> {
    const MIN_SAMPLES_FOR_CALIBRATION: usize = 50;
    const MIN_COHORT_SIZE: usize = 12;
    const MIN_REL_SPREAD: f32 = 1.15;
    const MIN_ABS_SPREAD_FACTOR: f32 = 0.15;

    if stats.len() < MIN_SAMPLES_FOR_CALIBRATION {
        return None;
    }
    let mut active_indices: Vec<usize> = Vec::new();
    let mut raw_features: Vec<[f64; 3]> = Vec::new();
    for (idx, s) in stats.iter().copied().enumerate() {
        if !s.has_signal() {
            continue;
        }
        let log_e = ((s.mismatch_mass + 1.0) / (s.emission_mass + 2.0)).ln();
        let log_s = ((s.expected_switches + 1.0) / (s.genetic_dist_morgans + 1e-3)).ln();
        let u = s.mean_uncertainty();
        if log_e.is_finite() && log_s.is_finite() && u.is_finite() {
            active_indices.push(idx);
            raw_features.push([log_e, log_s, u]);
        }
    }
    if active_indices.len() < MIN_SAMPLES_FOR_CALIBRATION {
        return None;
    }

    let mut dims: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    dims[0].reserve(raw_features.len());
    dims[1].reserve(raw_features.len());
    dims[2].reserve(raw_features.len());
    for f in &raw_features {
        dims[0].push(f[0]);
        dims[1].push(f[1]);
        dims[2].push(f[2]);
    }
    let (m0, s0) = robust_center_scale(&dims[0]);
    let (m1, s1) = robust_center_scale(&dims[1]);
    let (m2, s2) = robust_center_scale(&dims[2]);

    let mut scaled = Vec::with_capacity(raw_features.len() * 3);
    const FEATURE_CLAMP: f32 = 6.0;
    for f in &raw_features {
        let x0 = ((f[0] - m0) / s0) as f32;
        let x1 = ((f[1] - m1) / s1) as f32;
        let x2 = ((f[2] - m2) / s2) as f32;
        scaled.push(x0.clamp(-FEATURE_CLAMP, FEATURE_CLAMP));
        scaled.push(x1.clamp(-FEATURE_CLAMP, FEATURE_CLAMP));
        scaled.push(x2.clamp(-FEATURE_CLAMP, FEATURE_CLAMP));
    }
    let data = Array2::from_shape_vec((raw_features.len(), 3), scaled).ok()?;
    let dataset = DatasetBase::from(data);
    let n = raw_features.len();
    let max_k = 5usize.min(n.saturating_sub(1));
    if max_k < 2 {
        return None;
    }

    let mut best_model = None;
    let mut best_score = f32::INFINITY;
    for k in 2..=max_k {
        let model = match KMeans::params(k)
            .max_n_iterations(100)
            .tolerance(1e-4)
            .fit(&dataset)
        {
            Ok(m) => m,
            Err(_) => continue,
        };
        let inertia = model.inertia().max(1e-12);
        let penalty = ((k * 3) as f32) * (n as f32).ln().max(1.0);
        let score = (n as f32) * inertia.ln() + penalty;
        if score < best_score {
            best_score = score;
            best_model = Some(model);
        }
    }
    let model = best_model?;
    let labels: Array1<usize> = model.predict(dataset.records());
    let n_clusters = labels.iter().copied().max().unwrap_or(0) + 1;
    if n_clusters == 0 {
        return None;
    }

    let mut cohort_sizes = vec![0usize; n_clusters];
    let mut sum_mismatch = vec![0.0f64; n_clusters];
    let mut sum_emit = vec![0.0f64; n_clusters];
    for (row, &label) in labels.iter().enumerate() {
        let sample_idx = active_indices[row];
        let s = stats[sample_idx];
        cohort_sizes[label] += 1;
        sum_mismatch[label] += s.mismatch_mass;
        sum_emit[label] += s.emission_mass;
    }

    let global = MismatchProb::new_clamped(global_p_mismatch);
    let global_p = global.get();
    let mut cohort_p_mismatch = vec![global_p; n_clusters];
    for c in 0..n_clusters {
        if sum_emit[c] > 0.0 {
            let raw = (sum_mismatch[c] / sum_emit[c]) as f32;
            let lo = (global_p * 0.25).max(1e-6);
            let hi = (global_p * 4.0).min(0.25);
            cohort_p_mismatch[c] = raw.clamp(lo, hi);
        }
    }

    for &size in &cohort_sizes {
        if size < MIN_COHORT_SIZE {
            return None;
        }
    }
    let mut min_cohort = f32::INFINITY;
    let mut max_cohort = f32::NEG_INFINITY;
    for &p in &cohort_p_mismatch {
        min_cohort = min_cohort.min(p);
        max_cohort = max_cohort.max(p);
    }
    if !min_cohort.is_finite() || !max_cohort.is_finite() {
        return None;
    }
    let rel_spread = max_cohort / min_cohort.max(1e-6);
    let abs_spread = max_cohort - min_cohort;
    if rel_spread < MIN_REL_SPREAD || abs_spread < global_p * MIN_ABS_SPREAD_FACTOR {
        return None;
    }

    let mut sample_p_mismatch = vec![global_p; stats.len()];
    for (row, &label) in labels.iter().enumerate() {
        let sample_idx = active_indices[row];
        let s = stats[sample_idx];
        let weight = (s.emission_mass / (s.emission_mass + 500.0)).clamp(0.0, 1.0) as f32;
        let cohort_p = cohort_p_mismatch[label];
        sample_p_mismatch[sample_idx] =
            ((1.0 - weight) * global_p + weight * cohort_p).clamp(1e-6, 0.25);
    }

    Some(CohortCalibration {
        sample_p_mismatch,
        cohort_p_mismatch,
        cohort_sizes,
    })
}

const STAGE1_BLOCK_MIN_CM: f64 = 0.01;
const STAGE1_BLOCK_MAX_CM: f64 = 20.0;
const STAGE1_BLOCK_TARGET_MARKERS: usize = 200;
const STAGE1_BLOCK_TARGET_MARKERS_FAST_MAX: usize = 800;
const STAGE1_BLOCK_MIN_MARKERS: usize = 10;
const PBWT_SELECT_BLOCK_CM: f64 = 0.1;
const PBWT_MIN_MARKER_STEP: usize = 50;
const PBWT_MIN_SAMPLE_POINTS: usize = 10;
const PBWT_PER_WINDOW_MULT: usize = 8;
const PBWT_MIN_PER_HAP: usize = 64;
const PBWT_MAX_PER_HAP: usize = 256;
const PBWT_ADAPTIVE_K_MIN_DIVISOR: usize = 3;
const PBWT_ADAPTIVE_K_FLOOR: usize = 16;
const PBWT_ADAPTIVE_K_MAX_MULT: usize = 2;
const PBWT_WILDCARD_MIN_UNCERTAINTY: f32 = 0.85;
const PBWT_MIN_ENTROPY_EPS: f32 = 1e-12;
const PBWT_FORCE_TOP_HAPS: usize = 32;
const PBWT_ANCHOR_TOP_HAPS: usize = 512;
const EMIT_PROFILE_MAX_PROBE_SAMPLES: usize = 128;
const EMIT_PROFILE_PRIOR_STRENGTH: f32 = 24.0;
const EMIT_PROFILE_HET_CONFIDENCE_GATE: f32 = 0.98;
const EMIT_PROFILE_MIN_CONF_SCALE: f32 = 0.35;
const FAST_BEAM_WIDTH: usize = 16;
const FAST_BEAM_SWITCH_CANDIDATES: usize = 4;
const FAST_BEAM_INJECT_K: usize = 8;
const FAST_BEAM_FIX_CONF: f32 = 0.99;
const SCAN_RAM_FRACTION: f64 = 0.10;
const PHASE_RAM_FRACTION: f64 = 0.15;
const PHASE_AUTO_PRESCAN_MULT: usize = 4;
const PHASE_AUTO_PRESCAN_MIN: usize = 512;
const PHASE_AUTO_PRESCAN_MAX: usize = 2048;
const PHASE_STATE_BUDGET_SAFETY: f64 = 0.6;
const MIN_AVAIL_BYTES_FOR_PLANNING: u64 = 64 * 1024 * 1024;
const INVALID_ALLELE: u8 = 254;
const REDUCTION_SPARSE_MAX_MARKERS: usize = 1024;
const REDUCTION_SPARSE_HET_FRAC_NUM: usize = 3;
const REDUCTION_SPARSE_HET_FRAC_DEN: usize = 4;

struct RefAlleleProvider<'a, TargetSpace = AnyMarkerSpace, RefSpace = AnyMarkerSpace> {
    ref_gt: GenotypeView<'a, TargetSpace, RefSpace>,
    threaded_haps: &'a ThreadedHaps<CombinedHapSpace>,
    state_haps: Vec<HapIdx>,
    cursor: crate::model::states::MosaicCursor<CombinedHapSpace>,
    cursor_marker: Option<usize>,
}

impl<'a, TargetSpace, RefSpace> RefAlleleProvider<'a, TargetSpace, RefSpace> {
    fn new(
        ref_gt: GenotypeView<'a, TargetSpace, RefSpace>,
        threaded_haps: &'a ThreadedHaps<CombinedHapSpace>,
    ) -> Self {
        let n_states = threaded_haps.n_states();
        Self {
            ref_gt,
            threaded_haps,
            state_haps: vec![HapIdx::new(0); n_states],
            cursor: crate::model::states::MosaicCursor::from_threaded(threaded_haps),
            cursor_marker: None,
        }
    }

    #[inline]
    fn fill_ref_alleles(&mut self, marker: usize, out: &mut [u8]) {
        let n_states = self.threaded_haps.n_states().min(out.len());
        if self.state_haps.len() < n_states {
            self.state_haps.resize(n_states, HapIdx::new(0));
        }
        let can_advance = self
            .cursor_marker
            .map(|last| marker >= last)
            .unwrap_or(true);
        if can_advance {
            self.cursor.advance_to_marker(marker, self.threaded_haps);
        } else {
            self.cursor = crate::model::states::MosaicCursor::from_threaded(self.threaded_haps);
            self.cursor.advance_to_marker(marker, self.threaded_haps);
        }
        self.cursor_marker = Some(marker);
        let active = self.cursor.active_haps();
        let marker_idx = MarkerIdx::new(marker as u32);
        for i in 0..n_states {
            self.state_haps[i] = HapIdx::new(active[i].as_u32());
        }
        self.ref_gt.fill_batch(
            marker_idx,
            &self.state_haps[..n_states],
            &mut out[..n_states],
        );
    }

    fn materialize_into(&mut self, n_markers: usize, out: &mut [u8]) {
        let n_states = self.threaded_haps.n_states();
        if n_markers == 0 || n_states == 0 {
            return;
        }
        let needed = n_markers.saturating_mul(n_states);
        assert!(
            out.len() >= needed,
            "materialize_into buffer too small: have {}, need {}",
            out.len(),
            needed
        );
        for m in 0..n_markers {
            let offset = m * n_states;
            self.fill_ref_alleles(m, &mut out[offset..offset + n_states]);
        }
    }
}

fn partition_markers_by_cm(gen_positions: &[f64], block_cm: f64) -> Vec<(usize, usize)> {
    if gen_positions.is_empty() {
        return Vec::new();
    }
    let mut blocks = Vec::new();
    let mut start = 0usize;
    while start < gen_positions.len() {
        let start_pos = gen_positions[start];
        let mut end = start + 1;
        let limit = start_pos + block_cm;
        while end < gen_positions.len() && gen_positions[end] < limit {
            end += 1;
        }
        let min_end = (start + STAGE1_BLOCK_MIN_MARKERS).min(gen_positions.len());
        if end < min_end {
            end = min_end;
        }
        if end <= start {
            end = start.saturating_add(1).min(gen_positions.len());
        }
        blocks.push((start, end));
        start = end;
    }
    blocks
}

fn stage1_block_cm(gen_positions: &[f64]) -> f64 {
    if gen_positions.len() < 2 {
        return STAGE1_BLOCK_MIN_CM;
    }

    let mut sum_dist = 0.0f64;
    let mut sum_sq_dist = 0.0f64;
    let mut n_dist = 0usize;
    for w in gen_positions.windows(2) {
        let d = (w[1] - w[0]).abs().max(f64::EPSILON);
        sum_dist += d;
        sum_sq_dist += d * d;
        n_dist += 1;
    }
    if n_dist == 0 {
        return STAGE1_BLOCK_MIN_CM;
    }

    let avg = sum_dist / n_dist as f64;
    let variance = (sum_sq_dist / n_dist as f64 - avg * avg).max(0.0);
    let std_dev = variance.sqrt();
    let cv = std_dev / avg.max(f64::EPSILON);

    // Fast, accuracy-safe adaptive block sizing:
    // - In stable marker-density regions (low CV), use larger blocks to cut Stage 1 transitions.
    // - On very large windows, increase block marker targets further to reduce MCMC overhead.
    // - In volatile/sparse regions (high CV), stay close to legacy behavior.
    let density_stability_boost = if cv < 0.65 {
        1.0 + ((0.65 - cv) / 0.65) * 1.4
    } else {
        1.0
    };
    let window_scale_boost = match gen_positions.len() {
        0..=4_999 => 1.0,
        5_000..=11_999 => 1.2,
        12_000..=24_999 => 1.5,
        _ => 1.8,
    };
    let target_markers = ((STAGE1_BLOCK_TARGET_MARKERS as f64
        * density_stability_boost
        * window_scale_boost)
        .round() as usize)
        .clamp(
            STAGE1_BLOCK_TARGET_MARKERS,
            STAGE1_BLOCK_TARGET_MARKERS_FAST_MAX,
        );

    let block = avg * target_markers as f64;
    block.clamp(STAGE1_BLOCK_MIN_CM, STAGE1_BLOCK_MAX_CM)
}

fn estimate_phase_state_budget(
    available_bytes: u64,
    n_threads: usize,
    window_markers: usize,
) -> usize {
    if available_bytes == 0 || n_threads == 0 || window_markers == 0 {
        return 0;
    }
    let bytes_state_id = std::mem::size_of::<u32>();
    let bytes_allele = std::mem::size_of::<u8>();
    let bytes_prob = std::mem::size_of::<f32>();
    let bytes_path = bytes_state_id
        .saturating_add(bytes_allele)
        .saturating_add(bytes_prob.saturating_mul(2));
    let per_state_bytes = 64usize.saturating_add(window_markers.saturating_mul(bytes_path));
    if per_state_bytes == 0 {
        return 0;
    }
    let budget = (available_bytes as f64 * PHASE_RAM_FRACTION) as u64;
    let per_thread = budget / n_threads.max(1) as u64;
    let safe_bytes = (per_thread as f64 * PHASE_STATE_BUDGET_SAFETY) as u64;
    (safe_bytes as usize) / per_state_bytes
}

fn estimate_scan_batch_size(
    available_bytes: u64,
    n_ref_haps: usize,
    n_target_haps: usize,
) -> usize {
    if available_bytes == 0 || n_ref_haps == 0 || n_target_haps == 0 {
        return 1;
    }
    let per_hap_bytes = (n_ref_haps as u64).saturating_mul(20);
    if per_hap_bytes == 0 {
        return 1;
    }
    let budget = (available_bytes as f64 * SCAN_RAM_FRACTION) as u64;
    let mut batch = (budget / per_hap_bytes) as usize;
    if batch == 0 {
        batch = 1;
    }
    batch.min(n_target_haps)
}

fn build_sampling_points(
    gen_positions: &[f64],
    step_cm: f64,
    min_marker_step: usize,
    informative: Option<&[bool]>,
) -> Vec<bool> {
    let n = gen_positions.len();
    let mut sampling = vec![false; n];
    if n == 0 {
        return sampling;
    }
    let min_step = min_marker_step.min((n / PBWT_MIN_SAMPLE_POINTS).max(1));
    let step = step_cm.max(1e-6);
    let mut next_cm = gen_positions[0];
    let mut next_marker = 0usize;
    for m in 0..n {
        let cm = gen_positions[m];
        if cm >= next_cm || m >= next_marker {
            sampling[m] = true;
            next_cm = cm + step;
            next_marker = m.saturating_add(min_step);
        }
        if informative
            .and_then(|flags| flags.get(m))
            .copied()
            .unwrap_or(false)
        {
            sampling[m] = true;
        }
    }
    sampling[n - 1] = true;
    let count = sampling.iter().filter(|&&b| b).count();
    eprintln!(
        "[pbwt sampling] markers={} step_cm={:.6} min_step={} sampled={} first_cm={:.6} last_cm={:.6}",
        n,
        step,
        min_step,
        count,
        gen_positions.first().copied().unwrap_or(0.0),
        gen_positions.last().copied().unwrap_or(0.0)
    );
    sampling
}

fn select_top_k(scores: &[f32], k: usize) -> Vec<(usize, f32)> {
    if k == 0 || scores.is_empty() {
        return Vec::new();
    }
    let mut ranked: Vec<(usize, f32)> = scores
        .iter()
        .enumerate()
        .filter(|&(_, &s)| s.is_finite())
        .map(|(i, &s)| (i, s))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if ranked.len() > k {
        ranked.truncate(k);
    }
    ranked
}

#[inline]
fn adaptive_prescan_top_m(scores: &[f32], base_top: usize, n_ref_haps: usize) -> usize {
    if base_top == 0 || n_ref_haps == 0 {
        return 0;
    }
    let base = base_top.min(n_ref_haps).max(1);
    let min_top = (base / 2).max(PBWT_ADAPTIVE_K_FLOOR).min(base);
    if min_top >= base {
        return base;
    }
    const SCORE_DENSITY_SAMPLES: usize = 2048;
    let n = scores.len().min(n_ref_haps);
    if n == 0 {
        return min_top;
    }
    let samples = n.min(SCORE_DENSITY_SAMPLES).max(1);
    let mut finite_hits = 0usize;
    for i in 0..samples {
        let idx = ((i as u128 * n as u128) / samples as u128) as usize;
        let clamped = idx.min(n.saturating_sub(1));
        let s = scores[clamped];
        if s.is_finite() && s > 0.0 {
            finite_hits += 1;
        }
    }
    let uncertainty = (finite_hits as f32 / samples as f32).clamp(0.0, 1.0);
    let span = (base - min_top) as f32;
    (min_top as f32 + span * uncertainty)
        .round()
        .clamp(min_top as f32, base as f32) as usize
}

fn combine_swap_probs(fwd: &[f32], bwd: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(fwd.len());
    for i in 0..fwd.len() {
        let pf = fwd.get(i).copied().unwrap_or(0.5).clamp(1e-6, 1.0 - 1e-6);
        let pb = bwd.get(i).copied().unwrap_or(0.5).clamp(1e-6, 1.0 - 1e-6);
        let lf = (pf / (1.0 - pf)).ln();
        let lb = (pb / (1.0 - pb)).ln();
        let logit = lf + lb;
        let p = 1.0 / (1.0 + (-logit).exp());
        out.push(p.clamp(1e-6, 1.0 - 1e-6));
    }
    out
}

fn build_sparse_scores(
    window_scores: &[Vec<(usize, f32)>],
    abyss: &[bool],
) -> (Vec<usize>, Vec<Vec<(usize, f32)>>) {
    let mut map: HashMap<usize, usize> = HashMap::new();
    let mut candidate_haps: Vec<usize> = Vec::new();
    let mut scores_by_hap: Vec<Vec<(usize, f32)>> = Vec::new();

    for (w, list) in window_scores.iter().enumerate() {
        for &(hap, score) in list.iter() {
            if score <= 0.0 || !score.is_finite() {
                continue;
            }
            if hap < abyss.len() && abyss[hap] {
                continue;
            }
            let idx = *map.entry(hap).or_insert_with(|| {
                candidate_haps.push(hap);
                scores_by_hap.push(Vec::new());
                candidate_haps.len() - 1
            });
            scores_by_hap[idx].push((w, score));
        }
    }
    for scores in scores_by_hap.iter_mut() {
        scores.sort_by_key(|(w, _)| *w);
    }
    (candidate_haps, scores_by_hap)
}

fn compute_ref_freqs<TargetSpace, RefSpace>(
    target_gt: &GenotypeMatrix<impl crate::data::storage::phase_state::PhaseState, TargetSpace>,
    ref_columns: &[GenotypeColumn],
    alignment: Option<&MarkerAlignment<TargetSpace, RefSpace>>,
    marker_map: Option<&[usize]>,
    ref_index_map: Option<&[usize]>,
    n_markers: usize,
) -> Vec<Vec<f32>> {
    let n_ref_haps = ref_columns.first().map(|c| c.n_haplotypes()).unwrap_or(0);
    let mut freqs: Vec<Vec<f32>> = Vec::with_capacity(n_markers);
    for m in 0..n_markers {
        let orig_m = marker_map.map(|map| map[m]).unwrap_or(m);
        let n_alleles = target_gt
            .markers()
            .marker(MarkerIdx::new(orig_m as u32))
            .n_alleles();
        let mut counts = vec![0u32; n_alleles.max(1)];
        let mut total = 0u32;
        if let Some(alignment) = alignment {
            if let Some(ref_m) = alignment.target_to_ref(MarkerIdx::new(orig_m as u32)) {
                let ref_idx = if let Some(map) = ref_index_map {
                    let idx = map[ref_m.as_usize()];
                    if idx == usize::MAX {
                        freqs.push(vec![0.0f32; counts.len()]);
                        continue;
                    }
                    idx
                } else {
                    ref_m.as_usize()
                };
                for rh in 0..n_ref_haps {
                    let ref_a = ref_columns[ref_idx].get(HapIdx::new(rh as u32));
                    let mapped = alignment.reverse_map_allele(orig_m, ref_a);
                    if mapped == 255 {
                        continue;
                    }
                    let idx = mapped as usize;
                    if idx < counts.len() {
                        counts[idx] += 1;
                        total += 1;
                    }
                }
            }
        } else {
            let ref_idx = if let Some(map) = ref_index_map {
                if orig_m >= map.len() {
                    freqs.push(vec![0.0f32; counts.len()]);
                    continue;
                }
                let idx = map[orig_m];
                if idx == usize::MAX {
                    freqs.push(vec![0.0f32; counts.len()]);
                    continue;
                }
                idx
            } else {
                if orig_m >= ref_columns.len() {
                    freqs.push(vec![0.0f32; counts.len()]);
                    continue;
                }
                orig_m
            };
            for rh in 0..n_ref_haps {
                let ref_a = ref_columns[ref_idx].get(HapIdx::new(rh as u32));
                if ref_a == 255 {
                    continue;
                }
                let idx = ref_a as usize;
                if idx < counts.len() {
                    counts[idx] += 1;
                    total += 1;
                }
            }
        }
        let mut out = vec![0.0f32; counts.len()];
        if total > 0 {
            let inv = 1.0 / total as f32;
            for (i, c) in counts.into_iter().enumerate() {
                out[i] = c as f32 * inv;
            }
        }
        freqs.push(out);
    }
    freqs
}

#[inline]
fn pbwt_beam_uncertainty(beam: &RankBeam, n_ref_haps: usize, query: PbwtQueryAllele) -> f32 {
    if n_ref_haps == 0 {
        return 0.0;
    }
    let mut total = 0.0f32;
    let mut sq_sum = 0.0f32;
    let mut n_intervals = 0usize;
    for &(l, r) in beam.intervals() {
        if r <= l {
            continue;
        }
        let len = (r - l) as f32;
        total += len;
        sq_sum += len * len;
        n_intervals += 1;
    }
    if total <= 0.0 {
        return 0.0;
    }
    let coverage = (total / n_ref_haps as f32).clamp(0.0, 1.0);
    let entropy_norm = if n_intervals > 1 && sq_sum > 0.0 {
        let mut entropy = 0.0f32;
        for &(l, r) in beam.intervals() {
            if r <= l {
                continue;
            }
            let p = ((r - l) as f32 / total).clamp(PBWT_MIN_ENTROPY_EPS, 1.0);
            entropy -= p * p.ln();
        }
        let max_entropy = (n_intervals as f32).ln().max(PBWT_MIN_ENTROPY_EPS);
        (entropy / max_entropy).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mut uncertainty = (0.7 * coverage + 0.3 * entropy_norm).clamp(0.0, 1.0);
    if query.is_wildcard() {
        uncertainty = uncertainty.max(PBWT_WILDCARD_MIN_UNCERTAINTY);
    }
    uncertainty
}

#[inline]
fn adaptive_pbwt_donor_k(
    base_k: usize,
    n_ref_haps: usize,
    beam: &RankBeam,
    query: PbwtQueryAllele,
) -> NonZeroUsize {
    if n_ref_haps == 0 {
        return NonZeroUsize::new(1).expect("1 must be non-zero");
    }
    let base = base_k.max(1).min(n_ref_haps);
    let k_min = (base / PBWT_ADAPTIVE_K_MIN_DIVISOR)
        .max(PBWT_ADAPTIVE_K_FLOOR)
        .min(base);
    let k_max = base
        .saturating_mul(PBWT_ADAPTIVE_K_MAX_MULT)
        .max(base)
        .min(n_ref_haps);
    if k_max <= k_min {
        return NonZeroUsize::new(k_min.max(1)).expect("k_min.max(1) must be non-zero");
    }
    let uncertainty = pbwt_beam_uncertainty(beam, n_ref_haps, query);
    let span = (k_max - k_min) as f32;
    let k = (k_min as f32 + span * uncertainty)
        .round()
        .clamp(k_min as f32, k_max as f32) as usize;
    NonZeroUsize::new(k.max(1)).expect("k.max(1) must be non-zero")
}

#[inline]
fn biallelic_haplotype_probs(a1: u8, a2: u8, phase_conf: f32) -> [PbwtBiallelicQueryProb; 2] {
    if a1 == 0 && a2 == 1 {
        let p = phase_conf.clamp(0.0, 1.0);
        [
            PbwtBiallelicQueryProb::new(p, 1.0 - p),
            PbwtBiallelicQueryProb::new(1.0 - p, p),
        ]
    } else if a1 == 1 && a2 == 0 {
        let p = phase_conf.clamp(0.0, 1.0);
        [
            PbwtBiallelicQueryProb::new(1.0 - p, p),
            PbwtBiallelicQueryProb::new(p, 1.0 - p),
        ]
    } else if a1 == 0 && a2 == 0 {
        [
            PbwtBiallelicQueryProb::deterministic(0),
            PbwtBiallelicQueryProb::deterministic(0),
        ]
    } else if a1 == 1 && a2 == 1 {
        [
            PbwtBiallelicQueryProb::deterministic(1),
            PbwtBiallelicQueryProb::deterministic(1),
        ]
    } else {
        [
            PbwtBiallelicQueryProb::uniform(),
            PbwtBiallelicQueryProb::uniform(),
        ]
    }
}

fn score_window_batch_pbwt_segment<TargetState, TargetSpace, RefSpace>(
    batch_haps: &[usize],
    target_gt: &GenotypeMatrix<TargetState, TargetSpace>,
    geno: &MutableGenotypes,
    ref_columns: &[GenotypeColumn],
    phase_mask: Option<&crate::data::storage::matrix::BitMatrix>,
    mask_unphased_hets: Option<&[bool]>,
    alignment: Option<&MarkerAlignment<TargetSpace, RefSpace>>,
    freqs: &[Vec<f32>],
    window: (usize, usize),
    k_per_hap: usize,
    sampling: &[bool],
    window_scores: &mut [Vec<f32>],
    exclude_self: bool,
    marker_map: Option<&[usize]>,
    ref_index_map: Option<&[usize]>,
) where
    TargetState: crate::data::storage::phase_state::PhaseState,
{
    let n_ref_haps = ref_columns.first().map(|c| c.n_haplotypes()).unwrap_or(0);
    if batch_haps.is_empty() || n_ref_haps == 0 {
        return;
    }
    let (start, end) = window;
    if start >= end {
        return;
    }

    let mut pbwt_fwd = ReferencePbwt::new(n_ref_haps);
    let mut beams_fwd: Vec<RankBeam> = (0..batch_haps.len())
        .map(|_| RankBeam::full(n_ref_haps as u32))
        .collect();
    let mut ref_alleles = vec![0u8; n_ref_haps];
    let mut query_alleles = vec![PbwtQueryAllele::missing(); batch_haps.len()];
    let mut query_allele_probs = vec![PbwtBiallelicQueryProb::uniform(); batch_haps.len()];
    let mut donors_buf: Vec<u32> = Vec::new();

    let min_freq = 1.0 / (n_ref_haps.max(1) as f32);
    let mut anchor_seen_fwd: Vec<usize> = vec![0; batch_haps.len()];
    for m in start..end {
        let local_idx = m - start;
        let orig_m = marker_map.and_then(|map| map.get(m).copied()).unwrap_or(m);
        let marker_idx = MarkerIdx::new(orig_m as u32);
        let mut cached_sample_idx = usize::MAX;
        let mut cached_query_pair = [PbwtQueryAllele::missing(); 2];
        let mut cached_query_probs = [PbwtBiallelicQueryProb::uniform(); 2];
        for (i, &hap_idx) in batch_haps.iter().enumerate() {
            let sample_idx = hap_idx / 2;
            let local = hap_idx % 2;
            if sample_idx != cached_sample_idx {
                cached_sample_idx = sample_idx;
                let hap1 = sample_idx * 2;
                let hap2 = hap1 + 1;
                let a1 = geno.get(orig_m, HapIdx::new(hap1 as u32));
                let a2 = geno.get(orig_m, HapIdx::new(hap2 as u32));
                let is_het = a1 != 255 && a2 != 255 && a1 != a2;
                let qa1 = PbwtQueryAllele::allele(a1).unwrap_or_else(PbwtQueryAllele::missing);
                let qa2 = PbwtQueryAllele::allele(a2).unwrap_or_else(PbwtQueryAllele::missing);
                let phased = phase_mask
                    .and_then(|mask| mask.get(orig_m, sample_idx))
                    .unwrap_or(0);
                let wildcard_unphased_het = phased == 0
                    && mask_unphased_hets
                        .and_then(|flags| flags.get(sample_idx))
                        .copied()
                        .unwrap_or(false)
                    && is_het;
                cached_query_probs = biallelic_haplotype_probs(a1, a2, 1.0);
                if wildcard_unphased_het {
                    cached_query_pair = [PbwtQueryAllele::wildcard(), PbwtQueryAllele::wildcard()];
                    cached_query_probs = [
                        PbwtBiallelicQueryProb::uniform(),
                        PbwtBiallelicQueryProb::uniform(),
                    ];
                } else if phased == 0 && a1 != 255 && a1 == a2 {
                    cached_query_pair = [qa1, qa2];
                } else if phased != 0 && is_het {
                    let phase_conf = target_gt
                        .sample_phase_confidence_f32(marker_idx, sample_idx)
                        .clamp(0.0, 1.0);
                    cached_query_probs = biallelic_haplotype_probs(a1, a2, phase_conf);
                    // Phased scaffold markers are explicit orientation constraints.
                    if phase_conf < 0.5 {
                        cached_query_pair = [qa2, qa1];
                    } else {
                        cached_query_pair = [qa1, qa2];
                    }
                } else {
                    cached_query_pair = [qa1, qa2];
                    if phased == 0 && is_het {
                        cached_query_probs = [
                            PbwtBiallelicQueryProb::uniform(),
                            PbwtBiallelicQueryProb::uniform(),
                        ];
                    }
                }
            }
            query_alleles[i] = cached_query_pair[local];
            query_allele_probs[i] = cached_query_probs[local];
        }

        if let Some(alignment) = alignment {
            if let Some(ref_m) = alignment.target_to_ref(MarkerIdx::new(orig_m as u32)) {
                let ref_idx = if let Some(map) = ref_index_map {
                    let idx = map[ref_m.as_usize()];
                    if idx == usize::MAX {
                        ref_alleles.fill(255);
                        continue;
                    }
                    idx
                } else {
                    ref_m.as_usize()
                };
                for rh in 0..n_ref_haps {
                    let ref_a = ref_columns[ref_idx].get(HapIdx::new(rh as u32));
                    ref_alleles[rh] = alignment.reverse_map_allele(orig_m, ref_a);
                }
            } else {
                ref_alleles.fill(255);
            }
        } else {
            let ref_idx = if let Some(map) = ref_index_map {
                if orig_m >= map.len() {
                    ref_alleles.fill(255);
                    continue;
                }
                let idx = map[orig_m];
                if idx == usize::MAX {
                    ref_alleles.fill(255);
                    continue;
                }
                idx
            } else {
                if orig_m >= ref_columns.len() {
                    ref_alleles.fill(255);
                    continue;
                }
                orig_m
            };
            for rh in 0..n_ref_haps {
                ref_alleles[rh] = ref_columns[ref_idx].get(HapIdx::new(rh as u32));
            }
        }

        let mut is_biallelic = true;
        for &a in ref_alleles.iter() {
            if a >= 2 && a != 255 {
                is_biallelic = false;
                break;
            }
        }
        if is_biallelic {
            for &q in &query_alleles {
                if q.is_wildcard() {
                    continue;
                }
                if let Some(a) = q.as_allele() {
                    if a >= 2 {
                        is_biallelic = false;
                        break;
                    }
                }
            }
        }
        let n_alleles = if is_biallelic {
            2
        } else {
            let mut max_allele = 1u8;
            for &a in ref_alleles.iter() {
                if a != 255 && a > max_allele {
                    max_allele = a;
                }
            }
            for &q in &query_alleles {
                if let Some(a) = q.as_allele() {
                    if a > max_allele {
                        max_allele = a;
                    }
                }
            }
            (max_allele as usize).saturating_add(1).max(2)
        };

        pbwt_fwd.advance_with_beams_query_probs(
            &ref_alleles,
            n_alleles,
            local_idx,
            &query_alleles,
            Some(&query_allele_probs),
            &mut beams_fwd,
        );

        if sampling.get(local_idx).copied().unwrap_or(false) {
            for (i, &hap_idx) in batch_haps.iter().enumerate() {
                if query_alleles[i].is_missing() {
                    continue;
                }
                let donor_k =
                    adaptive_pbwt_donor_k(k_per_hap, n_ref_haps, &beams_fwd[i], query_alleles[i]);
                pbwt_fwd.select_donors_into(&beams_fwd[i], donor_k.get(), &mut donors_buf);
                let allele_probs = query_allele_probs[i];
                let p0 = allele_probs.prob_for_allele(0).clamp(0.0, 1.0);
                let p1 = allele_probs.prob_for_allele(1).clamp(0.0, 1.0);
                let query_allele = query_alleles[i].as_allele();
                let allele_certainty = match query_allele {
                    Some(a) if a > 1 => 1.0,
                    _ => (p0 - p1).abs(),
                };
                if allele_certainty <= f32::EPSILON {
                    continue;
                }
                let sample_idx = hap_idx / 2;
                let hap1 = sample_idx * 2;
                let hap2 = hap1 + 1;
                let a1 = geno.get(orig_m, HapIdx::new(hap1 as u32));
                let a2 = geno.get(orig_m, HapIdx::new(hap2 as u32));
                let is_het = a1 != 255 && a2 != 255 && a1 != a2;
                let phased = phase_mask
                    .and_then(|mask| mask.get(orig_m, sample_idx))
                    .unwrap_or(0);
                let full_anchor_scoring = phased != 0 && is_het;
                if full_anchor_scoring {
                    anchor_seen_fwd[i] = anchor_seen_fwd[i].saturating_add(1);
                    // Normalize cumulative anchor contribution so dense anchors do not dominate.
                    let anchor_norm = 1.0 / (anchor_seen_fwd[i] as f32).sqrt().max(1.0);
                    for idx in 0..n_ref_haps {
                        if exclude_self && idx / 2 == hap_idx / 2 {
                            continue;
                        }
                        let ref_allele = ref_alleles[idx];
                        if ref_allele == 255 {
                            continue;
                        }
                        let p_match = if ref_allele <= 1 {
                            allele_probs.prob_for_allele(ref_allele).clamp(0.0, 1.0)
                        } else {
                            match query_allele {
                                Some(a) if a == ref_allele => 1.0,
                                _ => 0.0,
                            }
                        };
                        if p_match <= 0.0 {
                            continue;
                        }
                        let freq = freqs
                            .get(m)
                            .and_then(|f| f.get(ref_allele as usize))
                            .copied()
                            .unwrap_or(0.0);
                        if freq <= 0.0 {
                            continue;
                        }
                        let weight =
                            anchor_norm * allele_certainty * p_match * -(freq.max(min_freq)).ln();
                        let w = &mut window_scores[i][idx];
                        if w.is_finite() {
                            *w += weight;
                        } else {
                            *w = weight;
                        }
                    }
                } else {
                    for &d in donors_buf.iter() {
                        let idx = d as usize;
                        if idx >= n_ref_haps {
                            continue;
                        }
                        if exclude_self && idx / 2 == hap_idx / 2 {
                            continue;
                        }
                        let ref_allele = ref_alleles[idx];
                        if ref_allele == 255 {
                            continue;
                        }
                        let p_match = if ref_allele <= 1 {
                            allele_probs.prob_for_allele(ref_allele).clamp(0.0, 1.0)
                        } else {
                            match query_allele {
                                Some(a) if a == ref_allele => 1.0,
                                _ => 0.0,
                            }
                        };
                        if p_match <= 0.0 {
                            continue;
                        }
                        let freq = freqs
                            .get(m)
                            .and_then(|f| f.get(ref_allele as usize))
                            .copied()
                            .unwrap_or(0.0);
                        if freq <= 0.0 {
                            continue;
                        }
                        let weight = allele_certainty * p_match * -(freq.max(min_freq)).ln();
                        let w = &mut window_scores[i][idx];
                        if w.is_finite() {
                            *w += weight;
                        } else {
                            *w = weight;
                        }
                    }
                }
            }
        }
    }

    let mut pbwt_bwd = ReferencePbwt::new(n_ref_haps);
    let mut beams_bwd: Vec<RankBeam> = (0..batch_haps.len())
        .map(|_| RankBeam::full(n_ref_haps as u32))
        .collect();
    let mut anchor_seen_bwd: Vec<usize> = vec![0; batch_haps.len()];
    for (rev_step, m) in (start..end).rev().enumerate() {
        let local_idx = end - start - 1 - rev_step;
        let orig_m = marker_map.and_then(|map| map.get(m).copied()).unwrap_or(m);
        let marker_idx = MarkerIdx::new(orig_m as u32);
        let mut cached_sample_idx = usize::MAX;
        let mut cached_query_pair = [PbwtQueryAllele::missing(); 2];
        let mut cached_query_probs = [PbwtBiallelicQueryProb::uniform(); 2];
        for (i, &hap_idx) in batch_haps.iter().enumerate() {
            let sample_idx = hap_idx / 2;
            let local = hap_idx % 2;
            if sample_idx != cached_sample_idx {
                cached_sample_idx = sample_idx;
                let hap1 = sample_idx * 2;
                let hap2 = hap1 + 1;
                let a1 = geno.get(orig_m, HapIdx::new(hap1 as u32));
                let a2 = geno.get(orig_m, HapIdx::new(hap2 as u32));
                let is_het = a1 != 255 && a2 != 255 && a1 != a2;
                let qa1 = PbwtQueryAllele::allele(a1).unwrap_or_else(PbwtQueryAllele::missing);
                let qa2 = PbwtQueryAllele::allele(a2).unwrap_or_else(PbwtQueryAllele::missing);
                let phased = phase_mask
                    .and_then(|mask| mask.get(orig_m, sample_idx))
                    .unwrap_or(0);
                let wildcard_unphased_het = phased == 0
                    && mask_unphased_hets
                        .and_then(|flags| flags.get(sample_idx))
                        .copied()
                        .unwrap_or(false)
                    && is_het;
                cached_query_probs = biallelic_haplotype_probs(a1, a2, 1.0);
                if wildcard_unphased_het {
                    cached_query_pair = [PbwtQueryAllele::wildcard(), PbwtQueryAllele::wildcard()];
                    cached_query_probs = [
                        PbwtBiallelicQueryProb::uniform(),
                        PbwtBiallelicQueryProb::uniform(),
                    ];
                } else if phased == 0 && a1 != 255 && a1 == a2 {
                    cached_query_pair = [qa1, qa2];
                } else if phased != 0 && is_het {
                    let phase_conf = target_gt
                        .sample_phase_confidence_f32(marker_idx, sample_idx)
                        .clamp(0.0, 1.0);
                    cached_query_probs = biallelic_haplotype_probs(a1, a2, phase_conf);
                    // Phased scaffold markers are explicit orientation constraints.
                    if phase_conf < 0.5 {
                        cached_query_pair = [qa2, qa1];
                    } else {
                        cached_query_pair = [qa1, qa2];
                    }
                } else {
                    cached_query_pair = [qa1, qa2];
                    if phased == 0 && is_het {
                        cached_query_probs = [
                            PbwtBiallelicQueryProb::uniform(),
                            PbwtBiallelicQueryProb::uniform(),
                        ];
                    }
                }
            }
            query_alleles[i] = cached_query_pair[local];
            query_allele_probs[i] = cached_query_probs[local];
        }

        if let Some(alignment) = alignment {
            if let Some(ref_m) = alignment.target_to_ref(MarkerIdx::new(orig_m as u32)) {
                let ref_idx = if let Some(map) = ref_index_map {
                    let idx = map[ref_m.as_usize()];
                    if idx == usize::MAX {
                        ref_alleles.fill(255);
                        continue;
                    }
                    idx
                } else {
                    ref_m.as_usize()
                };
                for rh in 0..n_ref_haps {
                    let ref_a = ref_columns[ref_idx].get(HapIdx::new(rh as u32));
                    ref_alleles[rh] = alignment.reverse_map_allele(orig_m, ref_a);
                }
            } else {
                ref_alleles.fill(255);
            }
        } else {
            let ref_idx = if let Some(map) = ref_index_map {
                if orig_m >= map.len() {
                    ref_alleles.fill(255);
                    continue;
                }
                let idx = map[orig_m];
                if idx == usize::MAX {
                    ref_alleles.fill(255);
                    continue;
                }
                idx
            } else {
                if orig_m >= ref_columns.len() {
                    ref_alleles.fill(255);
                    continue;
                }
                orig_m
            };
            for rh in 0..n_ref_haps {
                ref_alleles[rh] = ref_columns[ref_idx].get(HapIdx::new(rh as u32));
            }
        }

        let mut is_biallelic = true;
        for &a in ref_alleles.iter() {
            if a >= 2 && a != 255 {
                is_biallelic = false;
                break;
            }
        }
        if is_biallelic {
            for &q in &query_alleles {
                if q.is_wildcard() {
                    continue;
                }
                if let Some(a) = q.as_allele() {
                    if a >= 2 {
                        is_biallelic = false;
                        break;
                    }
                }
            }
        }
        let n_alleles = if is_biallelic {
            2
        } else {
            let mut max_allele = 1u8;
            for &a in ref_alleles.iter() {
                if a != 255 && a > max_allele {
                    max_allele = a;
                }
            }
            for &q in &query_alleles {
                if let Some(a) = q.as_allele() {
                    if a > max_allele {
                        max_allele = a;
                    }
                }
            }
            (max_allele as usize).saturating_add(1).max(2)
        };

        pbwt_bwd.advance_with_beams_query_probs(
            &ref_alleles,
            n_alleles,
            rev_step,
            &query_alleles,
            Some(&query_allele_probs),
            &mut beams_bwd,
        );

        if sampling.get(local_idx).copied().unwrap_or(false) {
            for (i, &hap_idx) in batch_haps.iter().enumerate() {
                if query_alleles[i].is_missing() {
                    continue;
                }
                let donor_k =
                    adaptive_pbwt_donor_k(k_per_hap, n_ref_haps, &beams_bwd[i], query_alleles[i]);
                pbwt_bwd.select_donors_into(&beams_bwd[i], donor_k.get(), &mut donors_buf);
                let allele_probs = query_allele_probs[i];
                let p0 = allele_probs.prob_for_allele(0).clamp(0.0, 1.0);
                let p1 = allele_probs.prob_for_allele(1).clamp(0.0, 1.0);
                let query_allele = query_alleles[i].as_allele();
                let allele_certainty = match query_allele {
                    Some(a) if a > 1 => 1.0,
                    _ => (p0 - p1).abs(),
                };
                if allele_certainty <= f32::EPSILON {
                    continue;
                }
                let sample_idx = hap_idx / 2;
                let hap1 = sample_idx * 2;
                let hap2 = hap1 + 1;
                let a1 = geno.get(orig_m, HapIdx::new(hap1 as u32));
                let a2 = geno.get(orig_m, HapIdx::new(hap2 as u32));
                let is_het = a1 != 255 && a2 != 255 && a1 != a2;
                let phased = phase_mask
                    .and_then(|mask| mask.get(orig_m, sample_idx))
                    .unwrap_or(0);
                let full_anchor_scoring = phased != 0 && is_het;
                if full_anchor_scoring {
                    anchor_seen_bwd[i] = anchor_seen_bwd[i].saturating_add(1);
                    let anchor_norm = 1.0 / (anchor_seen_bwd[i] as f32).sqrt().max(1.0);
                    for idx in 0..n_ref_haps {
                        if exclude_self && idx / 2 == hap_idx / 2 {
                            continue;
                        }
                        let ref_allele = ref_alleles[idx];
                        if ref_allele == 255 {
                            continue;
                        }
                        let p_match = if ref_allele <= 1 {
                            allele_probs.prob_for_allele(ref_allele).clamp(0.0, 1.0)
                        } else {
                            match query_allele {
                                Some(a) if a == ref_allele => 1.0,
                                _ => 0.0,
                            }
                        };
                        if p_match <= 0.0 {
                            continue;
                        }
                        let freq = freqs
                            .get(m)
                            .and_then(|f| f.get(ref_allele as usize))
                            .copied()
                            .unwrap_or(0.0);
                        if freq <= 0.0 {
                            continue;
                        }
                        let weight =
                            anchor_norm * allele_certainty * p_match * -(freq.max(min_freq)).ln();
                        let w = &mut window_scores[i][idx];
                        if w.is_finite() {
                            *w += weight;
                        } else {
                            *w = weight;
                        }
                    }
                } else {
                    for &d in donors_buf.iter() {
                        let idx = d as usize;
                        if idx >= n_ref_haps {
                            continue;
                        }
                        if exclude_self && idx / 2 == hap_idx / 2 {
                            continue;
                        }
                        let ref_allele = ref_alleles[idx];
                        if ref_allele == 255 {
                            continue;
                        }
                        let p_match = if ref_allele <= 1 {
                            allele_probs.prob_for_allele(ref_allele).clamp(0.0, 1.0)
                        } else {
                            match query_allele {
                                Some(a) if a == ref_allele => 1.0,
                                _ => 0.0,
                            }
                        };
                        if p_match <= 0.0 {
                            continue;
                        }
                        let freq = freqs
                            .get(m)
                            .and_then(|f| f.get(ref_allele as usize))
                            .copied()
                            .unwrap_or(0.0);
                        if freq <= 0.0 {
                            continue;
                        }
                        let weight = allele_certainty * p_match * -(freq.max(min_freq)).ln();
                        let w = &mut window_scores[i][idx];
                        if w.is_finite() {
                            *w += weight;
                        } else {
                            *w = weight;
                        }
                    }
                }
            }
        }
    }
}

/// Phasing pipeline
pub struct PhasingPipeline<RefSpace = crate::data::AnyMarkerSpace> {
    config: Config,
    params: ModelParams,
    /// Reference panel for reference-guided phasing (optional)
    /// Uses Arc for shared ownership to avoid cloning the large reference panel
    reference_gt: Option<Arc<GenotypeMatrix<Phased, RefSpace>>>,
    /// Marker alignment between target and reference
    alignment: Option<MarkerAlignment<crate::data::AnyMarkerSpace, RefSpace>>,
    telemetry: Option<Arc<TelemetryBlackboard>>,
}

struct FwdCheckpoints {
    block_starts: Arc<[usize]>,
    n_states: usize,
    data: Vec<f32>,
}

impl FwdCheckpoints {
    fn from_buffer(block_starts: Arc<[usize]>, n_states: usize, mut data: Vec<f32>) -> Self {
        let n_blocks = block_starts.len().max(1);
        let required = n_blocks * n_states;
        if data.len() < required {
            data.resize(required, 0.0);
        } else {
            data[..required].fill(0.0);
        }
        Self {
            n_states,
            block_starts,
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

fn blocks_to_starts(blocks: &[(usize, usize)], n_markers: usize) -> Vec<usize> {
    if n_markers == 0 {
        return Vec::new();
    }
    if blocks.is_empty() {
        return vec![0];
    }
    let mut out = Vec::with_capacity(blocks.len());
    for &(s, _) in blocks {
        if s < n_markers {
            out.push(s);
        }
    }
    if out.first().copied().unwrap_or(usize::MAX) != 0 {
        out.insert(0, 0);
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn max_block_len_from_starts(block_starts: &[usize], n_markers: usize) -> usize {
    if n_markers == 0 {
        return 0;
    }
    if block_starts.is_empty() {
        return n_markers.max(1);
    }
    let mut max_len = 1usize;
    for (i, &s) in block_starts.iter().enumerate() {
        let e = block_starts.get(i + 1).copied().unwrap_or(n_markers);
        if e > s {
            max_len = max_len.max(e - s);
        }
    }
    max_len
}

/// Overlap handoff payload for streaming windows.
///
/// `state_probs` is kept for compatibility and intra-window diagnostics, but
/// cross-window continuity should use `hap_priors`, which is identity-aware.
#[derive(Clone, Debug)]
pub struct Stage2OverlapHandoff {
    state_probs: Option<StateProbs>,
    hap_priors: Option<Vec<HaplotypePriors>>,
    prior_stage1_global_marker: Option<usize>,
    prior_stage1_gen_pos: Option<f64>,
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
    n_states: StateCount,
    fwd: StateAVec32<f32>,
    fwd_prior: StateAVec32<f32>,
    ref_alleles: StateVec<u8>,
    ref_alleles_flat: Vec<u8>,
    weights: StateVec<f32>,
    allele_probs: Vec<f32>,
    hap1_checkpoints: FwdCheckpoints,
    hap2_checkpoints: FwdCheckpoints,
    hap1_allele: Vec<u8>,
    hap1_partner_allele: Vec<u8>,
    hap1_use_combined: Vec<bool>,
    hap1_hard_match: Vec<bool>,
    hap2_allele: Vec<u8>,
    hap2_partner_allele: Vec<u8>,
    hap2_use_combined: Vec<bool>,
    hap2_hard_match: Vec<bool>,
    path1: Vec<u32>,
    path2: Vec<u32>,
    fwd_block: Vec<f32>,
}

#[derive(Clone, Debug)]
struct MosaicPaths {
    path1: Vec<u32>,
    path2: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct GlobalMosaicPaths {
    pub path1: Vec<CombinedHapId>,
    pub path2: Vec<CombinedHapId>,
}

fn local_to_global_paths(
    local: &MosaicPaths,
    threaded: &crate::model::states::ThreadedHaps<CombinedHapSpace>,
    n_markers: usize,
) -> GlobalMosaicPaths {
    let n_states = threaded.n_states();
    let mut cursor = crate::model::states::MosaicCursor::from_threaded(threaded);
    let mut path1 = Vec::with_capacity(n_markers);
    let mut path2 = Vec::with_capacity(n_markers);

    for m in 0..n_markers {
        cursor.advance_to_marker(m, threaded);
        let active = cursor.active_haps();
        let s1 = local.path1[m] as usize;
        let s2 = local.path2[m] as usize;

        path1.push(if s1 < n_states {
            active[s1]
        } else {
            CombinedHapId::from(0u32)
        });
        path2.push(if s2 < n_states {
            active[s2]
        } else {
            CombinedHapId::from(0u32)
        });
    }

    GlobalMosaicPaths { path1, path2 }
}

fn global_to_local_paths(
    global: &GlobalMosaicPaths,
    threaded: &crate::model::states::ThreadedHaps<CombinedHapSpace>,
    n_markers: usize,
) -> Option<MosaicPaths> {
    let mut cursor = crate::model::states::MosaicCursor::from_threaded(threaded);
    let mut path1 = Vec::with_capacity(n_markers);
    let mut path2 = Vec::with_capacity(n_markers);

    for m in 0..n_markers {
        cursor.advance_to_marker(m, threaded);
        let active = cursor.active_haps();

        let g1 = global.path1[m];
        let mut s1 = None;
        for (i, &gid) in active.iter().enumerate() {
            if gid == g1 {
                s1 = Some(i as u32);
                break;
            }
        }

        let g2 = global.path2[m];
        let mut s2 = None;
        for (i, &gid) in active.iter().enumerate() {
            if gid == g2 {
                s2 = Some(i as u32);
                break;
            }
        }

        if let (Some(idx1), Some(idx2)) = (s1, s2) {
            path1.push(idx1);
            path2.push(idx2);
        } else {
            return None;
        }
    }

    Some(MosaicPaths { path1, path2 })
}

struct MosaicChain<'a, RefSpace = crate::data::AnyMarkerSpace> {
    rng: rand::rngs::SmallRng,
    n_markers: usize,
    n_states: StateCount,
    panel_haps: usize,
    p_recomb: &'a [f32],
    seq1: &'a [u8],
    seq2: &'a [u8],
    conf: &'a [f32],
    ref_provider: RefAlleleProvider<'a, AnyMarkerSpace, RefSpace>,
    combined_checkpoints: &'a FwdCheckpoints,
    fwd: StateAVec32<f32>,
    fwd_prior: StateAVec32<f32>,
    ref_alleles: StateVec<u8>,
    ref_alleles_flat: Vec<u8>,
    ref_alleles_flat_ref: Option<&'a [u8]>,
    weights: StateVec<f32>,
    allele_probs: Vec<f32>,
    hap1_checkpoints: FwdCheckpoints,
    hap1_allele: Vec<u8>,
    hap1_partner_allele: Vec<u8>,
    hap1_use_combined: Vec<bool>,
    hap1_hard_match: Vec<bool>,
    hap2_checkpoints: FwdCheckpoints,
    hap2_allele: Vec<u8>,
    hap2_partner_allele: Vec<u8>,
    hap2_use_combined: Vec<bool>,
    hap2_hard_match: Vec<bool>,
    path1: Vec<u32>, // u32 saves 50% memory vs usize
    path2: Vec<u32>,
    fwd_block: Vec<f32>,
    trace: MosaicTrace,
    p_no_err: f32,
    p_err: f32,
    first_iteration: bool,
    pl_provider: Option<PlProvider<'a>>,
    anchor_hap1: Vec<u8>,
    anchor_hap2: Vec<u8>,
    anchor_drop_prob: f32,
}

impl<'a, RefSpace> MosaicChain<'a, RefSpace> {
    fn new_with_buffers(
        seed: u64,
        n_markers: usize,
        p_recomb: &'a [f32],
        seq1: &'a [u8],
        seq2: &'a [u8],
        conf: &'a [f32],
        ref_provider: RefAlleleProvider<'a, AnyMarkerSpace, RefSpace>,
        combined_checkpoints: &'a FwdCheckpoints,
        buffers: MosaicBuffers,
        ref_alleles_flat_ref: Option<&'a [u8]>,
        p_no_err: f32,
        p_err: f32,
        pl_provider: Option<PlProvider<'a>>,
        anchor_hap1: Vec<u8>,
        anchor_hap2: Vec<u8>,
    ) -> Self {
        let n_states = buffers.n_states;
        let anchor_count = anchor_hap1
            .iter()
            .zip(anchor_hap2.iter())
            .filter(|(a1, a2)| **a1 != 255 || **a2 != 255)
            .count();
        let anchor_frac = if n_markers > 0 {
            anchor_count as f32 / n_markers as f32
        } else {
            0.0
        };
        let anchor_drop_prob = (1.0 - anchor_frac).clamp(0.0, 0.5);
        let out = Self {
            rng: rand::rngs::SmallRng::seed_from_u64(seed),
            n_markers,
            n_states,
            panel_haps: ref_provider.ref_gt.n_haps(),
            p_recomb,
            seq1,
            seq2,
            conf,
            ref_provider,
            combined_checkpoints,
            fwd: buffers.fwd,
            fwd_prior: buffers.fwd_prior,
            ref_alleles: buffers.ref_alleles,
            ref_alleles_flat: buffers.ref_alleles_flat,
            ref_alleles_flat_ref,
            weights: buffers.weights,
            allele_probs: buffers.allele_probs,
            hap1_checkpoints: buffers.hap1_checkpoints,
            hap1_allele: buffers.hap1_allele,
            hap1_partner_allele: buffers.hap1_partner_allele,
            hap1_use_combined: buffers.hap1_use_combined,
            hap1_hard_match: buffers.hap1_hard_match,
            hap2_checkpoints: buffers.hap2_checkpoints,
            hap2_allele: buffers.hap2_allele,
            hap2_partner_allele: buffers.hap2_partner_allele,
            hap2_use_combined: buffers.hap2_use_combined,
            hap2_hard_match: buffers.hap2_hard_match,
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
            pl_provider,
            anchor_hap1,
            anchor_hap2,
            anchor_drop_prob,
        };
        out
    }

    fn into_buffers(self) -> MosaicBuffers {
        MosaicBuffers {
            n_states: self.n_states,
            fwd: self.fwd,
            fwd_prior: self.fwd_prior,
            ref_alleles: self.ref_alleles,
            ref_alleles_flat: self.ref_alleles_flat,
            weights: self.weights,
            allele_probs: self.allele_probs,
            hap1_checkpoints: self.hap1_checkpoints,
            hap2_checkpoints: self.hap2_checkpoints,
            hap1_allele: self.hap1_allele,
            hap1_partner_allele: self.hap1_partner_allele,
            hap1_use_combined: self.hap1_use_combined,
            hap1_hard_match: self.hap1_hard_match,
            hap2_allele: self.hap2_allele,
            hap2_partner_allele: self.hap2_partner_allele,
            hap2_use_combined: self.hap2_use_combined,
            hap2_hard_match: self.hap2_hard_match,
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

    #[inline]
    fn ref_row(&mut self, marker: usize) -> &[u8] {
        let n_states = self.n_states.get();
        if let Some(flat) = self.ref_alleles_flat_ref {
            let offset = marker * n_states;
            return &flat[offset..offset + n_states];
        }
        if !self.ref_alleles_flat.is_empty() {
            let offset = marker * n_states;
            return &self.ref_alleles_flat[offset..offset + n_states];
        }
        self.ref_provider
            .fill_ref_alleles(marker, self.ref_alleles.as_mut_slice());
        self.ref_alleles.as_slice()
    }

    fn build_hap2_inputs(&mut self) {
        for m in 0..self.n_markers {
            let a1 = self.seq1[m];
            let a2 = self.seq2[m];
            let p1 = self.path1[m] as usize;
            let ref_row = self.ref_row(m);
            let ref_al = ref_row[p1];
            // Partner allele is always the other haplotype's current reference allele.
            self.hap2_partner_allele[m] = ref_al;
            self.hap2_hard_match[m] = false;
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

            if ref_al == a1 {
                self.hap2_use_combined[m] = false;
                self.hap2_allele[m] = a2;
            } else if ref_al == a2 {
                self.hap2_use_combined[m] = false;
                self.hap2_allele[m] = a1;
            } else {
                // Partner allele incompatible with genotype under current sampled path.
                // Fall back to unconstrained combined emission rather than creating
                // an impossible hard wall in the state space.
                self.hap2_use_combined[m] = true;
                self.hap2_allele[m] = 255;
                self.hap2_hard_match[m] = false;
            }
        }
        if !self.anchor_hap2.is_empty() {
            use rand::Rng;
            for m in 0..self.n_markers {
                let a2 = self.anchor_hap2[m];
                if a2 == 255 {
                    continue;
                }
                if self.anchor_drop_prob > 0.0 && self.rng.random::<f32>() < self.anchor_drop_prob {
                    continue;
                }
                self.hap2_use_combined[m] = false;
                self.hap2_allele[m] = a2;
                self.hap2_hard_match[m] = false;
                let a1 = self.anchor_hap1.get(m).copied().unwrap_or(255);
                if a1 != 255 {
                    self.hap2_partner_allele[m] = a1;
                }
            }
        }
    }

    /// Build hap1 inputs based on current path2 (for proper Gibbs sampling).
    /// This determines what allele H1 must carry given H2's sampled path.
    fn build_hap1_inputs(&mut self) {
        for m in 0..self.n_markers {
            let a1 = self.seq1[m];
            let a2 = self.seq2[m];
            let p2 = self.path2[m] as usize;
            let ref_row = self.ref_row(m);
            let ref_al = ref_row[p2];
            // Partner allele is always the other haplotype's current reference allele.
            self.hap1_partner_allele[m] = ref_al;
            self.hap1_hard_match[m] = false;
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
            if ref_al == a1 {
                // H2 carries a1, so H1 must carry a2
                self.hap1_use_combined[m] = false;
                self.hap1_allele[m] = a2;
            } else if ref_al == a2 {
                // H2 carries a2, so H1 must carry a1
                self.hap1_use_combined[m] = false;
                self.hap1_allele[m] = a1;
            } else {
                // Partner allele incompatible with genotype under current sampled path.
                // Fall back to unconstrained combined emission rather than creating
                // an impossible hard wall in the state space.
                self.hap1_use_combined[m] = true;
                self.hap1_allele[m] = 255;
                self.hap1_hard_match[m] = false;
            }
        }
        if !self.anchor_hap1.is_empty() {
            use rand::Rng;
            for m in 0..self.n_markers {
                let a1 = self.anchor_hap1[m];
                if a1 == 255 {
                    continue;
                }
                if self.anchor_drop_prob > 0.0 && self.rng.random::<f32>() < self.anchor_drop_prob {
                    continue;
                }
                self.hap1_use_combined[m] = false;
                self.hap1_allele[m] = a1;
                self.hap1_hard_match[m] = false;
                let a2 = self.anchor_hap2.get(m).copied().unwrap_or(255);
                if a2 != 255 {
                    self.hap1_partner_allele[m] = a2;
                }
            }
        }
    }
}

impl<RefSpace> MarkovChain<MosaicTrace> for MosaicChain<'_, RefSpace> {
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
            let dummy_target = vec![255u8; self.n_markers];
            let dummy_partner = vec![255u8; self.n_markers];
            let dummy_combined = vec![true; self.n_markers];
            let dummy_hard = vec![false; self.n_markers];

            sample_path_from_checkpoints(
                &mut self.path1,
                &self.combined_checkpoints,
                self.n_markers,
                self.n_states.get(),
                self.panel_haps,
                self.p_recomb,
                self.seq1,
                self.seq2,
                self.conf,
                HapEmissionInputs {
                    target_constraint: &dummy_target,
                    partner_allele: &dummy_partner,
                    use_combined: &dummy_combined,
                    hard_match: &dummy_hard,
                },
                &mut self.ref_provider,
                (!self.ref_alleles_flat.is_empty()).then_some(&self.ref_alleles_flat[..]),
                self.pl_provider.as_ref(),
                self.p_no_err,
                self.p_err,
                &mut self.rng,
                &mut self.fwd_block,
                self.weights.as_mut_slice(),
                self.ref_alleles.as_mut_slice(),
                &mut self.allele_probs,
                EmissionMode::Combined,
            );
            self.first_iteration = false;
        } else {
            // Gibbs step: sample H1 | H2
            // Build hap1 constraints based on current path2
            self.build_hap1_inputs();
            let fwd = self.fwd.as_mut_slice();
            let fwd_prior = self.fwd_prior.as_mut_slice();
            let ref_alleles = self.ref_alleles.as_mut_slice();
            build_fwd_checkpoints(
                &mut self.hap1_checkpoints,
                self.n_markers,
                self.n_states.get(),
                self.panel_haps,
                self.p_recomb,
                self.seq1,
                self.seq2,
                self.conf,
                HapEmissionInputs {
                    target_constraint: &self.hap1_allele,
                    partner_allele: &self.hap1_partner_allele,
                    use_combined: &self.hap1_use_combined,
                    hard_match: &self.hap1_hard_match,
                },
                &mut self.ref_provider,
                (!self.ref_alleles_flat.is_empty()).then_some(&self.ref_alleles_flat[..]),
                self.pl_provider.as_ref(),
                &mut self.allele_probs,
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
                self.n_states.get(),
                self.panel_haps,
                self.p_recomb,
                self.seq1,
                self.seq2,
                self.conf,
                HapEmissionInputs {
                    target_constraint: &self.hap1_allele,
                    partner_allele: &self.hap1_partner_allele,
                    use_combined: &self.hap1_use_combined,
                    hard_match: &self.hap1_hard_match,
                },
                &mut self.ref_provider,
                (!self.ref_alleles_flat.is_empty()).then_some(&self.ref_alleles_flat[..]),
                self.pl_provider.as_ref(),
                self.p_no_err,
                self.p_err,
                &mut self.rng,
                &mut self.fwd_block,
                self.weights.as_mut_slice(),
                self.ref_alleles.as_mut_slice(),
                &mut self.allele_probs,
                EmissionMode::Hap,
            );
        }

        // Gibbs step: sample H2 | H1
        self.build_hap2_inputs();
        let fwd = self.fwd.as_mut_slice();
        let fwd_prior = self.fwd_prior.as_mut_slice();
        let ref_alleles = self.ref_alleles.as_mut_slice();
        build_fwd_checkpoints(
            &mut self.hap2_checkpoints,
            self.n_markers,
            self.n_states.get(),
            self.panel_haps,
            self.p_recomb,
            self.seq1,
            self.seq2,
            self.conf,
            HapEmissionInputs {
                target_constraint: &self.hap2_allele,
                partner_allele: &self.hap2_partner_allele,
                use_combined: &self.hap2_use_combined,
                hard_match: &self.hap2_hard_match,
            },
            &mut self.ref_provider,
            (!self.ref_alleles_flat.is_empty()).then_some(&self.ref_alleles_flat[..]),
            self.pl_provider.as_ref(),
            &mut self.allele_probs,
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
            self.n_states.get(),
            self.panel_haps,
            self.p_recomb,
            self.seq1,
            self.seq2,
            self.conf,
            HapEmissionInputs {
                target_constraint: &self.hap2_allele,
                partner_allele: &self.hap2_partner_allele,
                use_combined: &self.hap2_use_combined,
                hard_match: &self.hap2_hard_match,
            },
            &mut self.ref_provider,
            (!self.ref_alleles_flat.is_empty()).then_some(&self.ref_alleles_flat[..]),
            self.pl_provider.as_ref(),
            self.p_no_err,
            self.p_err,
            &mut self.rng,
            &mut self.fwd_block,
            self.weights.as_mut_slice(),
            self.ref_alleles.as_mut_slice(),
            &mut self.allele_probs,
            EmissionMode::Hap,
        );
        self.update_trace();
        &self.trace
    }

    fn current_state(&self) -> &MosaicTrace {
        &self.trace
    }
}

impl<RefSpace: Send + Sync> PhasingPipeline<RefSpace> {
    /// Create a new phasing pipeline
    pub fn new(config: Config, telemetry: Option<Arc<TelemetryBlackboard>>) -> Self {
        let params = ModelParams::new();
        Self {
            config,
            params,
            reference_gt: None,
            alignment: None,
            telemetry,
        }
    }

    /// Set reference panel for reference-guided phasing
    ///
    /// When a reference panel is provided, the phasing algorithm uses it to:
    /// 1. Improve state selection (PBWT neighbors from reference)
    /// 2. Guide phase decisions with reference haplotypes
    ///
    /// Uses Arc for shared ownership to avoid cloning the large reference panel.
    pub fn set_reference(
        &mut self,
        reference: Arc<GenotypeMatrix<Phased, RefSpace>>,
        alignment: MarkerAlignment<crate::data::AnyMarkerSpace, RefSpace>,
    ) {
        self.reference_gt = Some(reference);
        self.alignment = Some(alignment);
    }
}

impl PhasingPipeline<crate::data::AnyMarkerSpace> {
    /// Run the phasing pipeline
    pub fn run(&mut self) -> Result<()> {
        THREAD_WORKSPACE.with(|ws| *ws.borrow_mut() = None);
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
        let (mut reader, file_reader) = VcfReader::open(&self.config.target)?;
        reader.set_exclude_samples(&exclude_samples);
        reader.set_exclude_markers(exclude_markers);
        let target_gt = reader.read_all(file_reader)?;
        let input_fully_phased = target_gt
            .phase_mask()
            .map(|mask| (0..mask.n_rows()).all(|row| mask.row_all_set(row)))
            .unwrap_or(true);

        if input_fully_phased {
            eprintln!("Input already fully phased; writing passthrough output.");
            let output_path = self.config.out.with_extension("vcf.gz");
            let mut writer = VcfWriter::create(&output_path, target_gt.samples_arc())?;
            writer.write_header(target_gt.markers())?;
            let phased = target_gt.clone().into_phased();
            writer.write_phased(&phased, 0, phased.n_markers())?;
            eprintln!("Phasing complete!");
            return Ok(());
        }

        if target_gt.n_markers() == 0 {
            eprintln!("No markers found in input VCF");
            return Ok(());
        }

        let n_markers = target_gt.n_markers();
        let n_samples = target_gt.n_samples();
        let n_haps = target_gt.n_haplotypes();

        if let Some(bb) = &self.telemetry {
            bb.set_total_samples(n_samples as u64);
            bb.set_samples_processed(0);
            bb.set_total_markers(n_markers as u64);
            bb.set_markers_processed(0);
        }

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
            let ref_gt: GenotypeMatrix<Phased> =
                if ref_path.extension().map(|e| e == "bref3").unwrap_or(false) {
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
            eprintln!(
                "  Aligned {} reference markers to target",
                alignment.n_aligned()
            );

            // Store in pipeline struct for use during phasing iterations
            self.set_reference(Arc::new(ref_gt), alignment);
        }

        // Compute combined haplotype count
        let n_ref_haps = self
            .reference_gt
            .as_ref()
            .map(|r| r.n_haplotypes())
            .unwrap_or(0);
        let n_total_haps = n_haps + n_ref_haps;

        if n_ref_haps > 0 {
            eprintln!(
                "Combined haplotype space: {} target + {} reference = {} total",
                n_haps, n_ref_haps, n_total_haps
            );
        }

        // Initialize parameters based on TOTAL haplotype count (target + ref)
        self.params = ModelParams::for_phasing(n_total_haps, self.config.ne, self.config.err);
        // Keep LR-threshold schedule consistent with the configured iteration plan.
        self.params.burnin = self.config.burnin;
        self.params.iterations = self.config.iterations.max(1);
        if self.config.phase_states > 0 {
            self.params
                .set_n_states(self.config.phase_states.min(n_total_haps.saturating_sub(2)));
        }
        // Calibrate LR threshold schedule for dynamic MCMC.
        // With N Gibbs steps the max achievable LR is (N+0.5)/0.5 = 2N+1.
        // Setting initial_lr just above this ensures only "perfect evidence"
        // hets lock right after burnin, with gradual relaxation thereafter.
        if self.config.dynamic_mcmc {
            let max_lr = 2.0 * self.config.mcmc_steps as f32 + 1.0;
            self.params.initial_lr = (1.2 * max_lr).max(4.0);
        }

        eprintln!(
            "Phasing parameters: p_mismatch={}, recomb_intensity={}, n_states={} (requested phase_states={})",
            self.params.p_mismatch,
            self.params.recomb_intensity,
            self.params.n_states,
            self.config.phase_states
        );

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
        // Includes reference panel allele counts if available, ensuring homozygous
        // markers in small target panels are correctly classified as high-frequency.
        let maf: Vec<f32> =
            if let (Some(ref_gt), Some(alignment)) = (&self.reference_gt, &self.alignment) {
                (0..n_markers)
                    .map(|m| {
                        let m_idx = MarkerIdx::new(m as u32);
                        let mut target_alt = target_gt.column(m_idx).alt_count() as usize;
                        let target_total = target_gt.n_haplotypes();
                        if let Some(ref_m) = alignment.target_to_ref(m_idx) {
                            let ref_col = ref_gt.column(ref_m);
                            let ref_total = ref_gt.n_haplotypes();
                            let ref_alt = ref_col.alt_count() as usize;

                            if let Some(Some(mapping)) = alignment.allele_mappings.get(m) {
                                let ref_equiv_of_alt =
                                    mapping.targ_to_ref.get(1).copied().unwrap_or(-1);
                                if ref_equiv_of_alt == 1 {
                                    target_alt += ref_alt;
                                } else if ref_equiv_of_alt == 0 {
                                    target_alt += ref_total - ref_alt;
                                }
                            }

                            let total = target_total + ref_total;
                            if total == 0 {
                                0.0
                            } else {
                                let freq = target_alt as f32 / total as f32;
                                freq.min(1.0 - freq)
                            }
                        } else {
                            let freq = target_alt as f32 / target_total as f32;
                            freq.min(1.0 - freq)
                        }
                    })
                    .collect()
            } else {
                (0..n_markers)
                    .map(|m| target_gt.column(MarkerIdx::new(m as u32)).maf() as f32)
                    .collect()
            };

        // TWO-STAGE PHASING: Classify markers by frequency
        // Stage 1 (high-frequency): Run full HMM - these markers provide phasing signal
        // Stage 2 (rare): Interpolate from flanking high-frequency markers
        let rare_threshold = self.config.rare;
        let hi_freq_markers: Vec<usize> = (0..n_markers)
            .filter(|&m| maf[m] >= rare_threshold)
            .collect();
        let has_missing = |m: usize| -> bool {
            let m_idx = MarkerIdx::new(m as u32);
            (0..n_haps).any(|h| target_gt.allele(m_idx, HapIdx::new(h as u32)) == 255)
        };
        let rare_markers: Vec<usize> = (0..n_markers)
            .filter(|&m| (maf[m] < rare_threshold && maf[m] > 0.0) || has_missing(m))
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
        let hi_freq_gen_positions: Vec<f64> =
            hi_freq_to_orig.iter().map(|&m| gen_positions[m]).collect();

        let stage1_blocks = partition_markers_by_cm(
            &hi_freq_gen_positions,
            stage1_block_cm(&hi_freq_gen_positions),
        );
        eprintln!("Stage 1 blocks: {}", stage1_blocks.len());

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

        // Log ploidy information from detected samples
        let samples = target_gt.samples_arc();
        let n_haploid = (0..n_samples)
            .filter(|&s| !samples.is_diploid(SampleIdx::new(s as u32)))
            .count();
        if n_haploid > 0 {
            return Err(crate::error::ReagleError::vcf(format!(
                "Detected {} haploid samples. Reagle currently supports diploid samples only.",
                n_haploid
            )));
        }

        // Stage 1 (condensed beam) - single pass
        if let Some(bb) = &self.telemetry {
            bb.set_stage(Stage::PhasingMain);
            bb.set_producer_stage(Stage::PhasingMain);
            bb.set_total_samples(n_samples as u64);
            bb.set_samples_processed(0);
            bb.set_total_markers(hi_freq_markers.len() as u64);
            bb.set_markers_processed(0);
            bb.set_total_iterations(1);
            bb.set_current_iteration(1);
        }

        let confidence_by_sample = build_sample_confidence(&target_gt);
        let phase_mask = target_gt.phase_mask();
        let mut sample_phases = self.create_sample_phases(&geno, &confidence_by_sample, phase_mask);
        let mut stage1_p_recomb: Vec<f32> = std::iter::once(0.0f32)
            .chain(stage1_gen_dists.iter().map(|&d| self.params.p_recomb(d)))
            .collect();

        if self.reference_gt.is_none() {
            // Build IBS2 segments for phase consistency (target-only)
            if let Some(bb) = &self.telemetry {
                bb.set_stage(Stage::PhasingPrescan);
                bb.set_producer_stage(Stage::PhasingPrescan);
                bb.set_op("Phasing prescan: IBS2 segments");
            }
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

            // Fallback: iterative HMM phasing without reference
            let n_burnin = self.config.burnin;
            let n_iterations = self.config.iterations;
            let total_iterations = n_burnin + n_iterations;
            if let Some(bb) = &self.telemetry {
                bb.set_total_samples(n_samples as u64);
                bb.set_samples_processed(0);
                bb.set_total_markers(hi_freq_markers.len() as u64);
                bb.set_markers_processed(0);
                bb.set_total_iterations(total_iterations as u64);
                bb.set_current_iteration(0);
            }

            let ref_gt = self.reference_gt.as_ref().map(|v| v.as_ref());
            let threaded_haps_vec = self.build_phasing_prescan_states(
                &target_gt,
                &geno,
                ref_gt,
                self.alignment.as_ref(),
                n_hi_freq,
                n_samples,
                &hi_freq_gen_positions,
                self.config.imp_step,
                Some(&hi_freq_to_orig),
            )?;
            let mut mcmc_paths: Vec<Option<GlobalMosaicPaths>> = vec![None; n_samples];
            let mut stable_main_iters = 0usize;
            let mut prev_remaining_hets: Option<usize> = None;
            let mut frozen_samples = vec![false; n_samples];
            let mut frozen_streaks = vec![0usize; n_samples];
            let mut cohort_calibration: Option<CohortCalibration> = None;

            for it in 0..total_iterations {
                let is_burnin = it < n_burnin;
                let iter_type = if is_burnin { "burnin" } else { "main" };
                eprintln!("Iteration {}/{} ({})", it + 1, total_iterations, iter_type);
                if let Some(bb) = &self.telemetry {
                    let stage = if is_burnin {
                        Stage::PhasingBurnin
                    } else {
                        Stage::PhasingMain
                    };
                    bb.set_stage(stage);
                    bb.set_producer_stage(stage);
                    bb.set_current_iteration((it + 1) as u64);
                    bb.set_samples_processed(0);
                    bb.set_markers_processed(0);
                }

                self.params.lr_threshold = self.params.lr_threshold_for_iteration(it);
                let atomic_estimates = if is_burnin && self.config.em {
                    Some(crate::model::parameters::AtomicParamEstimates::new())
                } else {
                    None
                };

                let mut cohort_stats = if is_burnin && it == 0 {
                    Some(vec![SampleCohortStats::default(); n_samples])
                } else {
                    None
                };
                let (total_switches, total_phased, sample_changed) = self
                    .run_phase_baum_iteration_stage1(
                        &target_gt,
                        &mut geno,
                        &threaded_haps_vec,
                        &stage1_p_recomb,
                        &stage1_gen_dists,
                        &hi_freq_to_orig,
                        &stage1_blocks,
                        &ibs2,
                        &mut sample_phases,
                        &mut mcmc_paths,
                        if is_burnin {
                            None
                        } else {
                            Some(&frozen_samples)
                        },
                        cohort_calibration
                            .as_ref()
                            .map(|cal| cal.sample_p_mismatch.as_slice()),
                        cohort_stats.as_deref_mut(),
                        atomic_estimates.as_ref(),
                        it,
                    )?;
                if let Some(bb) = &self.telemetry {
                    bb.set_samples_processed(n_samples as u64);
                    bb.set_markers_processed(hi_freq_markers.len() as u64);
                }

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
                    if params_updated {
                        stage1_p_recomb = std::iter::once(0.0f32)
                            .chain(stage1_gen_dists.iter().map(|&d| self.params.p_recomb(d)))
                            .collect();
                    }
                }

                if let Some(stats) = cohort_stats {
                    if let Some(model) = fit_cohort_calibration(&stats, self.params.p_mismatch) {
                        let n_cohorts = model.cohort_p_mismatch.len();
                        let preview: Vec<String> = model
                            .cohort_p_mismatch
                            .iter()
                            .zip(model.cohort_sizes.iter())
                            .map(|(&p, &n)| format!("{:.6} (n={})", p, n))
                            .collect();
                        eprintln!(
                            "Cohort calibration enabled: {} cohorts, p_mismatch={}",
                            n_cohorts,
                            preview.join(", ")
                        );
                        cohort_calibration = Some(model);
                    } else {
                        eprintln!("Cohort calibration skipped: insufficient structure in burn-in");
                    }
                }

                if !is_burnin {
                    let remaining_hets =
                        Self::count_unphased_hets(&sample_phases, &hi_freq_to_orig);
                    let mut newly_frozen = 0usize;
                    for s in 0..n_samples {
                        if frozen_samples[s] {
                            continue;
                        }
                        if sample_changed[s] {
                            frozen_streaks[s] = 0;
                        } else {
                            frozen_streaks[s] += 1;
                            if frozen_streaks[s] >= 2 {
                                frozen_samples[s] = true;
                                newly_frozen += 1;
                            }
                        }
                    }
                    if newly_frozen > 0 {
                        let frozen_total = frozen_samples.iter().filter(|&&v| v).count();
                        eprintln!(
                            "Stage 1 freezing: {} newly frozen samples ({} / {} total)",
                            newly_frozen, frozen_total, n_samples
                        );
                    }

                    let no_progress = total_switches == 0 && total_phased == 0;
                    let unresolved_unchanged = prev_remaining_hets
                        .map(|prev| prev == remaining_hets)
                        .unwrap_or(false);
                    if no_progress && unresolved_unchanged {
                        stable_main_iters += 1;
                        if stable_main_iters >= 2 {
                            eprintln!(
                                "Phasing converged (exact fixed point: no new switches/locks and unresolved hets unchanged for 2 main iterations); stopping early."
                            );
                            break;
                        }
                    } else {
                        stable_main_iters = 0;
                    }
                    prev_remaining_hets = Some(remaining_hets);
                }
            }
        } else {
            let ref_gt = match self.reference_gt.as_ref() {
                Some(r) => r.clone(),
                None => unreachable!(),
            };
            let alignment = self.alignment.as_ref().ok_or_else(|| {
                crate::error::ReagleError::config("Reference alignment missing for beam phasing")
            })?;

            let packed_ref =
                PackedRefView::build_sparse(&target_gt, &ref_gt, alignment, &hi_freq_to_orig)?;

            // Compute allele frequencies for TMRCA-aware beam scoring.
            // For each hi-freq marker, compute (freq_allele0, freq_allele1) from reference.
            let hi_freq_allele_freqs: Option<Vec<(f32, f32)>> = {
                let n_ref_haps = packed_ref.n_ref_haps();
                if n_ref_haps > 0 {
                    Some(
                        hi_freq_to_orig
                            .iter()
                            .map(|&orig_m| {
                                let mut count0 = 0u32;
                                let mut count1 = 0u32;
                                for h in 0..n_ref_haps {
                                    match packed_ref.ref_allele_targ(orig_m, h) {
                                        Some(0) => count0 += 1,
                                        Some(1) => count1 += 1,
                                        _ => {}
                                    }
                                }
                                let total = (count0 + count1).max(1) as f32;
                                (count0 as f32 / total, count1 as f32 / total)
                            })
                            .collect(),
                    )
                } else {
                    None
                }
            };

            let beam_config = BeamConfig::default();
            let mut beam_config_fast = beam_config;
            beam_config_fast.beam_width = FAST_BEAM_WIDTH;
            beam_config_fast.switch_candidates = FAST_BEAM_SWITCH_CANDIDATES;
            beam_config_fast.inject_k = FAST_BEAM_INJECT_K.min(beam_config.inject_k.max(1));
            if beam_config_fast.inject_interval == 0 {
                beam_config_fast.inject_interval = beam_config.inject_interval;
            }
            let beam_index = PbwtBeamIndex::build(
                &ref_gt,
                alignment,
                &hi_freq_to_orig,
                &hi_freq_gen_positions,
                beam_config.inject_k.max(beam_config_fast.inject_k),
                beam_config.inject_interval,
                self.params.recomb_intensity,
            );
            let pbwt_stats: Option<Vec<(f32, f32, f32, f32)>> = Some(
                (0..hi_freq_to_orig.len())
                    .map(|hi_idx| beam_index.stats_for_hi(hi_idx))
                    .collect(),
            );
            let phaser_fast = BeamPhaser::new(&packed_ref, &self.params, beam_config_fast);
            let phaser = BeamPhaser::new(&packed_ref, &self.params, beam_config);

            let ibs2 = Ibs2::new(&target_gt, &gen_maps, chrom, &maf);

            let mut threaded_haps_vec = self.build_phasing_prescan_states(
                &target_gt,
                &geno,
                Some(ref_gt.as_ref()),
                self.alignment.as_ref(),
                n_hi_freq,
                n_samples,
                &hi_freq_gen_positions,
                self.config.imp_step,
                Some(&hi_freq_to_orig),
            )?;

            const PRESCAN_BEAM_SEED: usize = 8;
            let beam_confidence: Vec<std::sync::Mutex<Vec<(usize, u8, u8, f32)>>> = (0..n_samples)
                .map(|_| std::sync::Mutex::new(Vec::new()))
                .collect();
            let beam_donors: Vec<std::sync::Mutex<Vec<usize>>> = (0..n_samples)
                .map(|_| std::sync::Mutex::new(Vec::new()))
                .collect();
            let fast_confidence: Vec<std::sync::Mutex<Vec<(usize, u8, u8, f32)>>> = (0..n_samples)
                .map(|_| std::sync::Mutex::new(Vec::new()))
                .collect();

            let n_target_haps = target_gt.n_haplotypes();
            // Fast pass: fix high-confidence hets using a smaller beam.
            sample_phases.par_iter().enumerate().for_each(|(s, sp)| {
                let mut active_pool = ActivePool::new(packed_ref.n_ref_haps());
                let mut tmp = vec![
                    crate::model::types::CombinedHapId::from(0u32);
                    threaded_haps_vec[s].n_states()
                ];
                let th = &threaded_haps_vec[s];
                th.materialize_at(0, &mut tmp);
                for id in tmp.iter().copied() {
                    let hid = id.as_u32() as usize;
                    if hid >= n_target_haps {
                        let ref_id = hid - n_target_haps;
                        if ref_id < packed_ref.n_ref_haps() {
                            active_pool.add(ref_id);
                        }
                    }
                }

                let mut promoted = 0usize;
                for id in tmp.iter().copied() {
                    if promoted >= PRESCAN_BEAM_SEED {
                        break;
                    }
                    let hid = id.as_u32() as usize;
                    if hid >= n_target_haps {
                        let ref_id = hid - n_target_haps;
                        if ref_id < packed_ref.n_ref_haps() {
                            active_pool.promote(ref_id);
                            promoted += 1;
                        }
                    }
                }

                let hard_threshold_nats = if beam_config_fast.prune_tolerance > 0 {
                    Some((beam_config_fast.prune_tolerance as f64) / 1_000_000.0)
                } else {
                    None
                };
                let constraint_max_expected =
                    Some(beam_config_fast.switch_candidates.max(2) as f64);
                let condensed = CondensedTarget::build(
                    sp,
                    &hi_freq_to_orig,
                    &hi_freq_gen_positions,
                    hi_freq_allele_freqs.as_deref(),
                    pbwt_stats.as_deref(),
                    &packed_ref,
                    &self.params,
                    hard_threshold_nats,
                    constraint_max_expected,
                );

                let mut sp_fast = sp.clone();
                let mut injector = PbwtInjector::new(
                    &beam_index,
                    packed_ref.n_ref_haps(),
                    beam_config_fast.inject_k,
                );
                let fwd = phaser_fast.phase_sample(
                    &condensed,
                    &mut sp_fast,
                    &mut active_pool,
                    &mut injector,
                );
                let condensed_rev = condensed.reversed(&hi_freq_gen_positions);
                let mut active_pool_rev = ActivePool::new(packed_ref.n_ref_haps());
                let mut tmp_rev =
                    vec![crate::model::types::CombinedHapId::from(0u32); th.n_states()];
                th.materialize_at(0, &mut tmp_rev);
                for id in tmp_rev.iter().copied() {
                    let hid = id.as_u32() as usize;
                    if hid >= n_target_haps {
                        let ref_id = hid - n_target_haps;
                        if ref_id < packed_ref.n_ref_haps() {
                            active_pool_rev.add(ref_id);
                        }
                    }
                }
                let mut sp_rev = sp.clone();
                let mut injector_rev = PbwtInjector::new(
                    &beam_index,
                    packed_ref.n_ref_haps(),
                    beam_config_fast.inject_k,
                );
                let bwd = phaser_fast.phase_sample(
                    &condensed_rev,
                    &mut sp_rev,
                    &mut active_pool_rev,
                    &mut injector_rev,
                );
                let mut p_swapped_bwd = bwd.p_swapped;
                p_swapped_bwd.reverse();
                let combined = combine_swap_probs(&fwd.p_swapped, &p_swapped_bwd);
                if let Ok(mut slot) = fast_confidence[s].lock() {
                    slot.clear();
                    for (i, &p) in combined.iter().enumerate() {
                        let call = &condensed.call_sites[i];
                        slot.push((call.marker.as_usize(), call.a1, call.a2, p));
                    }
                }
            });

            let mut fast_fixed: Vec<Vec<usize>> = vec![Vec::new(); n_samples];
            let mut fast_total: u64 = 0;
            let mut fast_fixed_count: u64 = 0;
            for (s, sp) in sample_phases.iter_mut().enumerate() {
                if let Ok(slot) = fast_confidence[s].lock() {
                    for &(m, a1, a2, p) in slot.iter() {
                        fast_total += 1;
                        if !sp.is_unphased(m) {
                            continue;
                        }
                        if !sp.has_input_phase_anchor() {
                            continue;
                        }
                        let conf = p.max(1.0 - p);
                        if conf < FAST_BEAM_FIX_CONF {
                            continue;
                        }
                        let want_swapped = p >= 0.5;
                        let swapped_now = sp.allele1(m) == a2 && sp.allele2(m) == a1;
                        if want_swapped != swapped_now {
                            sp.swap_alleles(m);
                        }
                        sp.mark_phased(m);
                        sp.set_phase_confidence(m, conf);
                        fast_fixed[s].push(m);
                        fast_fixed_count += 1;
                    }
                }
            }
            if let Some(bb) = &self.telemetry {
                bb.set_fast_beam_stats(fast_fixed_count, fast_total);
            }

            let mut original_unphased: Vec<Vec<usize>> = sample_phases
                .iter()
                .map(|sp| {
                    hi_freq_to_orig
                        .iter()
                        .copied()
                        .filter(|&m| sp.is_unphased(m))
                        .collect()
                })
                .collect();
            for s in 0..n_samples {
                if fast_fixed[s].is_empty() {
                    continue;
                }
                let mut fixed = HashSet::with_capacity(fast_fixed[s].len());
                for &m in &fast_fixed[s] {
                    fixed.insert(m);
                }
                original_unphased[s].retain(|m| !fixed.contains(m));
            }

            sample_phases
                .par_iter_mut()
                .enumerate()
                .for_each(|(s, sp)| {
                    let mut active_pool = ActivePool::new(packed_ref.n_ref_haps());
                    let mut tmp = vec![
                        crate::model::types::CombinedHapId::from(0u32);
                        threaded_haps_vec[s].n_states()
                    ];
                    let th = &threaded_haps_vec[s];
                    th.materialize_at(0, &mut tmp);
                    for id in tmp.iter().copied() {
                        let hid = id.as_u32() as usize;
                        if hid >= n_target_haps {
                            let ref_id = hid - n_target_haps;
                            if ref_id < packed_ref.n_ref_haps() {
                                active_pool.add(ref_id);
                            }
                        }
                    }

                    let mut promoted = 0usize;
                    for id in tmp.iter().copied() {
                        if promoted >= PRESCAN_BEAM_SEED {
                            break;
                        }
                        let hid = id.as_u32() as usize;
                        if hid >= n_target_haps {
                            let ref_id = hid - n_target_haps;
                            if ref_id < packed_ref.n_ref_haps() {
                                active_pool.promote(ref_id);
                                promoted += 1;
                            }
                        }
                    }

                    let hard_threshold_nats = if beam_config.prune_tolerance > 0 {
                        Some((beam_config.prune_tolerance as f64) / 1_000_000.0)
                    } else {
                        None
                    };
                    let constraint_max_expected = Some(beam_config.switch_candidates.max(2) as f64);
                    let condensed = CondensedTarget::build(
                        sp,
                        &hi_freq_to_orig,
                        &hi_freq_gen_positions,
                        hi_freq_allele_freqs.as_deref(),
                        pbwt_stats.as_deref(),
                        &packed_ref,
                        &self.params,
                        hard_threshold_nats,
                        constraint_max_expected,
                    );

                    let mut injector = PbwtInjector::new(
                        &beam_index,
                        packed_ref.n_ref_haps(),
                        beam_config.inject_k,
                    );
                    let fwd = phaser.phase_sample(&condensed, sp, &mut active_pool, &mut injector);

                    let condensed_rev = condensed.reversed(&hi_freq_gen_positions);
                    let mut active_pool_rev = ActivePool::new(packed_ref.n_ref_haps());
                    let mut tmp_rev =
                        vec![crate::model::types::CombinedHapId::from(0u32); th.n_states()];
                    th.materialize_at(0, &mut tmp_rev);
                    for id in tmp_rev.iter().copied() {
                        let hid = id.as_u32() as usize;
                        if hid >= n_target_haps {
                            let ref_id = hid - n_target_haps;
                            if ref_id < packed_ref.n_ref_haps() {
                                active_pool_rev.add(ref_id);
                            }
                        }
                    }
                    let mut sp_rev = sp.clone();
                    let mut injector_rev = PbwtInjector::new(
                        &beam_index,
                        packed_ref.n_ref_haps(),
                        beam_config.inject_k,
                    );
                    let bwd = phaser.phase_sample(
                        &condensed_rev,
                        &mut sp_rev,
                        &mut active_pool_rev,
                        &mut injector_rev,
                    );
                    let mut p_swapped_bwd = bwd.p_swapped;
                    p_swapped_bwd.reverse();
                    let combined = combine_swap_probs(&fwd.p_swapped, &p_swapped_bwd);

                    if let Ok(mut slot) = beam_confidence[s].lock() {
                        slot.clear();
                        for (i, &p) in combined.iter().enumerate() {
                            let call = &condensed.call_sites[i];
                            slot.push((call.marker.as_usize(), call.a1, call.a2, p));
                        }
                    }
                    if let Ok(mut slot) = beam_donors[s].lock() {
                        slot.clear();
                        let list = active_pool.list();
                        let cap = beam_config
                            .beam_width
                            .max(beam_config.inject_k.saturating_mul(2))
                            .max(beam_config.switch_candidates.saturating_mul(4));
                        if list.len() > cap {
                            slot.extend_from_slice(&list[list.len() - cap..]);
                        } else {
                            slot.extend_from_slice(list);
                        }
                    }
                });

            for (s, sp) in sample_phases.iter_mut().enumerate() {
                for &m in original_unphased[s].iter() {
                    sp.mark_unphased(m);
                }
            }

            // Feed beam-selected donors back into the HMM state set.
            if packed_ref.n_ref_haps() > 0 {
                let offset = n_target_haps as u32;
                for s in 0..n_samples {
                    let Ok(donors) = beam_donors[s].lock() else {
                        continue;
                    };
                    if donors.is_empty() {
                        continue;
                    }
                    let threaded_haps = &mut threaded_haps_vec[s];
                    let mut existing = vec![CombinedHapId::from(0u32); threaded_haps.n_states()];
                    threaded_haps.materialize_at(0, &mut existing);
                    let mut seen: HashSet<u32> =
                        HashSet::with_capacity(existing.len() + donors.len());
                    for id in existing {
                        seen.insert(id.as_u32());
                    }
                    for &h in donors.iter() {
                        let combined = combined_from_ref(RefHapId::from(h), offset);
                        let id = combined.as_u32();
                        if seen.insert(id) {
                            threaded_haps.push_new(combined);
                        }
                    }
                }
            }

            // Beam-first, then sparse MCMC refinement on remaining unphased hets.
            let mut mcmc_paths: Vec<Option<GlobalMosaicPaths>> = vec![None; n_samples];
            if packed_ref.n_ref_haps() > 0 {
                let offset = n_target_haps as u32;
                for s in 0..n_samples {
                    let Ok(donors) = beam_donors[s].lock() else {
                        continue;
                    };
                    if donors.is_empty() {
                        continue;
                    }
                    let d1 = donors[0];
                    let d2 = *donors.get(1).unwrap_or(&donors[0]);
                    let hap1 = CombinedHapId::new(offset + d1 as u32);
                    let hap2 = CombinedHapId::new(offset + d2 as u32);
                    let path1 = vec![hap1; n_hi_freq];
                    let path2 = vec![hap2; n_hi_freq];
                    mcmc_paths[s] = Some(GlobalMosaicPaths { path1, path2 });
                }
            }
            let n_burnin = self.config.burnin;
            let n_iterations = self.config.iterations;
            let total_iterations = n_burnin + n_iterations;
            let mut stable_main_iters = 0usize;
            let mut prev_remaining_hets: Option<usize> = None;
            let mut frozen_samples = vec![false; n_samples];
            let mut frozen_streaks = vec![0usize; n_samples];
            let mut cohort_calibration: Option<CohortCalibration> = None;
            for it in 0..total_iterations {
                let is_burnin = it < n_burnin;
                let iter_type = if is_burnin { "burnin" } else { "main" };
                eprintln!("Iteration {}/{} ({})", it + 1, total_iterations, iter_type);
                if let Some(bb) = &self.telemetry {
                    let stage = if is_burnin {
                        Stage::PhasingBurnin
                    } else {
                        Stage::PhasingMain
                    };
                    bb.set_stage(stage);
                    bb.set_producer_stage(stage);
                    bb.set_current_iteration((it + 1) as u64);
                    bb.set_samples_processed(0);
                    bb.set_markers_processed(0);
                }

                self.params.lr_threshold = self.params.lr_threshold_for_iteration(it);
                let mut cohort_stats = if is_burnin && it == 0 {
                    Some(vec![SampleCohortStats::default(); n_samples])
                } else {
                    None
                };
                let (total_switches, total_phased, sample_changed) = self
                    .run_phase_baum_iteration_stage1(
                        &target_gt,
                        &mut geno,
                        &threaded_haps_vec,
                        &stage1_p_recomb,
                        &stage1_gen_dists,
                        &hi_freq_to_orig,
                        &stage1_blocks,
                        &ibs2,
                        &mut sample_phases,
                        &mut mcmc_paths,
                        if is_burnin {
                            None
                        } else {
                            Some(&frozen_samples)
                        },
                        cohort_calibration
                            .as_ref()
                            .map(|cal| cal.sample_p_mismatch.as_slice()),
                        cohort_stats.as_deref_mut(),
                        None,
                        it,
                    )?;
                if let Some(bb) = &self.telemetry {
                    bb.set_samples_processed(n_samples as u64);
                    bb.set_markers_processed(hi_freq_markers.len() as u64);
                }

                if !is_burnin {
                    let remaining_hets =
                        Self::count_unphased_hets(&sample_phases, &hi_freq_to_orig);
                    let mut newly_frozen = 0usize;
                    for s in 0..n_samples {
                        if frozen_samples[s] {
                            continue;
                        }
                        if sample_changed[s] {
                            frozen_streaks[s] = 0;
                        } else {
                            frozen_streaks[s] += 1;
                            if frozen_streaks[s] >= 2 {
                                frozen_samples[s] = true;
                                newly_frozen += 1;
                            }
                        }
                    }
                    if newly_frozen > 0 {
                        let frozen_total = frozen_samples.iter().filter(|&&v| v).count();
                        eprintln!(
                            "Stage 1 freezing: {} newly frozen samples ({} / {} total)",
                            newly_frozen, frozen_total, n_samples
                        );
                    }

                    let no_progress = total_switches == 0 && total_phased == 0;
                    let unresolved_unchanged = prev_remaining_hets
                        .map(|prev| prev == remaining_hets)
                        .unwrap_or(false);
                    if no_progress && unresolved_unchanged {
                        stable_main_iters += 1;
                        if stable_main_iters >= 2 {
                            eprintln!(
                                "Phasing converged (exact fixed point: no new switches/locks and unresolved hets unchanged for 2 main iterations); stopping early."
                            );
                            break;
                        }
                    } else {
                        stable_main_iters = 0;
                    }
                    prev_remaining_hets = Some(remaining_hets);
                }

                if let Some(stats) = cohort_stats {
                    if let Some(model) = fit_cohort_calibration(&stats, self.params.p_mismatch) {
                        let preview: Vec<String> = model
                            .cohort_p_mismatch
                            .iter()
                            .zip(model.cohort_sizes.iter())
                            .map(|(&p, &n)| format!("{:.6} (n={})", p, n))
                            .collect();
                        eprintln!(
                            "Cohort calibration enabled: {} cohorts, p_mismatch={}",
                            model.cohort_p_mismatch.len(),
                            preview.join(", ")
                        );
                        cohort_calibration = Some(model);
                    } else {
                        eprintln!("Cohort calibration skipped: insufficient structure in burn-in");
                    }
                }
            }

            // Note: We intentionally do NOT overwrite phase confidence here.
            // The HMM refinement populates calibrated confidence values; reapplying
            // beam confidence would clobber those scores.
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
            if let Some(bb) = &self.telemetry {
                bb.set_stage(Stage::PhasingStage2);
                bb.set_producer_stage(Stage::PhasingStage2);
                bb.set_op("Phasing stage2: HMM interpolation");
                bb.set_total_iterations(0);
                bb.set_current_iteration(0);
                bb.set_total_markers(rare_markers.len() as u64);
                bb.set_markers_processed(0);
                bb.set_samples_processed(0);
            }
            let stage2_handoff = self.phase_rare_markers_with_hmm(
                &target_gt,
                &mut geno,
                &hi_freq_markers,
                &gen_positions,
                &hi_freq_gen_positions,
                &stage1_p_recomb,
                &mut sample_phases,
                &maf,
                rare_threshold,
                None,
                None,
            );
            tracing::trace!(
                has_handoff = stage2_handoff.is_some(),
                "Stage 2 HMM overlap handoff computed"
            );
            if let Some(bb) = &self.telemetry {
                bb.set_markers_processed(rare_markers.len() as u64);
                bb.set_samples_processed(n_samples as u64);
            }

            // Sync again after Stage 2
            self.sync_sample_phases_to_geno(&sample_phases, &mut geno);
        }

        // Build final GenotypeMatrix from mutable genotypes
        let final_gt = self.build_final_matrix(&target_gt, &geno, &sample_phases);

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
        THREAD_WORKSPACE.with(|ws| *ws.borrow_mut() = None);
        eprintln!("Opening VCF for streaming...");

        // Configure streaming (genetic maps loaded lazily by StreamingVcfReader)
        let streaming_config = StreamingConfig {
            window_cm: self.config.window,
            overlap_cm: self.config.overlap,
            max_markers: self.config.window_markers,
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

        // Load reference panel if provided (for reference-guided phasing)
        let mut ref_pos_map = None;
        if let Some(ref_path) = &self.config.r#ref {
            eprintln!("Loading reference panel for streaming phasing...");
            let ref_gt: GenotypeMatrix<Phased> =
                if ref_path.extension().map(|e| e == "bref3").unwrap_or(false) {
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
            ref_pos_map = Some(
                MarkerAlignment::<crate::data::AnyMarkerSpace, _>::build_ref_pos_index(
                    ref_gt.markers(),
                ),
            );
            self.reference_gt = Some(Arc::new(ref_gt));
            self.alignment = None;
        }

        // Open streaming reader
        let mut reader =
            StreamingVcfReader::open(&self.config.target, gen_maps.clone(), streaming_config)?;
        let samples = reader.samples_arc();

        // Check for haploid samples
        let n_samples = samples.len();
        let n_haploid = (0..n_samples)
            .filter(|&s| !samples.is_diploid(SampleIdx::new(s as u32)))
            .count();
        if n_haploid > 0 {
            return Err(crate::error::ReagleError::vcf(format!(
                "Detected {} haploid samples. Reagle currently supports diploid samples only.",
                n_haploid
            )));
        }

        // Create output writer
        let output_path = self.config.out.with_extension("vcf.gz");
        eprintln!("Writing output to {:?}", output_path);
        let mut writer = VcfWriter::create(&output_path, samples)?;

        let mut window_count = 0;
        let mut total_markers = 0;
        let mut wrote_header = false;
        let mut align_stats = AlignmentStats::default();

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

            if let (Some(ref_gt), Some(ref_pos_map)) =
                (self.reference_gt.as_ref(), ref_pos_map.as_ref())
            {
                let (alignment, stats) = MarkerAlignment::new_with_ref_index(
                    &window.genotypes,
                    ref_gt.markers(),
                    ref_pos_map,
                );
                align_stats.aligned += stats.aligned;
                align_stats.strand_flipped += stats.strand_flipped;
                align_stats.allele_swapped += stats.allele_swapped;
                self.alignment = Some(alignment);
            } else {
                self.alignment = None;
            }

            // Phase this window with overlap constraint
            let (phased, next_overlap_handoff) = self.phase_in_memory_with_overlap(
                &window.genotypes,
                &gen_maps,
                window.phased_overlap.as_ref(),
                Some(window.output_end),
            )?;

            // Extract overlap for next window (contains identity-aware priors for handoff)
            if !window.is_last() {
                phased_overlap = Some(self.extract_overlap(
                    &phased,
                    window.output_end,
                    n_markers,
                    next_overlap_handoff,
                ));
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
                if current.window.is_first && !wrote_header {
                    writer.write_header(finalized.markers())?;
                    wrote_header = true;
                }
                writer.write_phased(
                    &finalized,
                    current.window.output_start,
                    current.window.output_end,
                )?;
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
            info_span!("finalize_last_window").in_scope(|| -> Result<()> {
                let finalized = current.phased_result.as_ref().unwrap().clone(); // No additional context
                if current.window.is_first && !wrote_header {
                    writer.write_header(finalized.markers())?;
                }
                writer.write_phased(&finalized, current.output_start, current.output_end)?;
                total_markers += current.output_end - current.output_start;
                Ok(())
            })?;
        }

        writer.flush()?;
        eprintln!(
            "Streaming phasing complete: {} windows, {} markers",
            window_count, total_markers
        );
        if align_stats.aligned > 0
            && (align_stats.strand_flipped > 0 || align_stats.allele_swapped > 0)
        {
            eprintln!(
                "  Alignment summary (streaming): {} strand-flipped, {} allele-swapped markers",
                align_stats.strand_flipped, align_stats.allele_swapped
            );
        }
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
        handoff: Option<Stage2OverlapHandoff>,
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

        // Attach soft-information handoff payloads if available.
        if let Some(handoff) = handoff {
            if let Some(probs) = handoff.state_probs {
                let state_meta = (probs.n_states, probs.marker_indices.len(), probs.data.len());
                tracing::trace!(
                    n_states = state_meta.0,
                    marker_indices = state_meta.1,
                    hap_entries = state_meta.2,
                    "Attaching legacy state_probs handoff"
                );
                overlap.set_state_probs(probs);
            }
            if let Some(priors) = handoff.hap_priors {
                overlap.set_hap_priors(priors);
            }
            if let Some(marker) = handoff.prior_stage1_global_marker {
                overlap.set_prior_stage1_global_marker(marker);
            }
            if let Some(gen_pos) = handoff.prior_stage1_gen_pos {
                overlap.set_prior_stage1_gen_pos(gen_pos);
            }
        }

        overlap
    }

    /// Automatically select between in-memory and streaming mode based on data size
    pub fn run_auto(&mut self) -> Result<()> {
        let threshold = self.config.window_markers.max(1);
        let estimated_markers =
            Self::estimate_marker_count_for_auto(&self.config.target, threshold + 1).ok();
        let use_streaming = estimated_markers.map(|n| n > threshold).unwrap_or_else(|| {
            let file_size = std::fs::metadata(&self.config.target)
                .map(|m| m.len())
                .unwrap_or(0);
            file_size > (threshold as u64).saturating_mul(100)
        });

        if use_streaming {
            if let Some(n) = estimated_markers {
                eprintln!(
                    "Auto-detected large dataset (>{} markers, observed at least {}), using streaming mode",
                    threshold, n
                );
            } else {
                eprintln!("Auto-detected large dataset (fallback heuristic), using streaming mode");
            }
            self.run_streaming()
        } else {
            self.run()
        }
    }

    fn estimate_marker_count_for_auto(path: &std::path::Path, limit: usize) -> Result<usize> {
        let (_, mut reader) = VcfReader::open(path)?;
        let mut line_buf: Vec<u8> = Vec::new();
        let mut markers = 0usize;
        while markers < limit {
            line_buf.clear();
            let bytes_read = std::io::BufRead::read_until(&mut reader, b'\n', &mut line_buf)?;
            if bytes_read == 0 {
                break;
            }
            if line_buf.is_empty() || line_buf[0] == b'#' {
                continue;
            }
            markers += 1;
        }
        Ok(markers)
    }
}

impl<RefSpace: Send + Sync> PhasingPipeline<RefSpace> {
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
    ) -> Result<(
        GenotypeMatrix<crate::data::storage::phase_state::Phased>,
        Option<Stage2OverlapHandoff>,
    )> {
        let n_markers = target_gt.n_markers();
        let n_haps = target_gt.n_haplotypes();
        let n_samples = n_haps / 2;
        let n_ref_haps = self
            .reference_gt
            .as_ref()
            .map(|r| r.n_haplotypes())
            .unwrap_or(0);
        let n_total_haps = n_haps + n_ref_haps;
        let samples = target_gt.samples_arc();

        // Check for haploid samples
        let n_haploid = (0..n_samples)
            .filter(|&s| !samples.is_diploid(SampleIdx::new(s as u32)))
            .count();
        if n_haploid > 0 {
            return Err(crate::error::ReagleError::vcf(format!(
                "Detected {} haploid samples. Reagle currently supports diploid samples only.",
                n_haploid
            )));
        }

        if n_markers == 0 {
            return Ok((target_gt.clone().into_phased(), None));
        }

        self.params = ModelParams::for_phasing(n_total_haps, self.config.ne, self.config.err);
        // Keep LR-threshold schedule consistent with the configured iteration plan.
        self.params.burnin = self.config.burnin;
        self.params.iterations = self.config.iterations.max(1);
        if self.config.phase_states > 0 {
            self.params
                .set_n_states(self.config.phase_states.min(n_total_haps.saturating_sub(2)));
        }
        if self.config.dynamic_mcmc {
            let max_lr = 2.0 * self.config.mcmc_steps as f32 + 1.0;
            self.params.initial_lr = (1.2 * max_lr).max(4.0);
        }

        // Initialize genotypes preserving actual allele values including missing (255)
        let mut geno = MutableGenotypes::from_fn(n_markers, n_haps, |m, h| {
            target_gt.allele(MarkerIdx::new(m as u32), HapIdx::new(h as u32))
        });

        // Build missing mask for overlap constraint handling
        let missing_mask: Vec<BitBox<u8, Lsb0>> = (0..n_haps)
            .map(|h| {
                let bits: BitVec<u8, Lsb0> = (0..n_markers)
                    .map(|m| {
                        target_gt.allele(MarkerIdx::new(m as u32), HapIdx::new(h as u32)) == 255
                    })
                    .collect();
                bits.into_boxed_bitslice()
            })
            .collect();

        // Apply overlap constraint: set alleles from previous window's phased overlap
        // This seeds the phasing with the known phase from the overlap region
        let overlap_markers = if let Some(overlap) = phased_overlap {
            self.apply_overlap_constraint(&mut geno, overlap, &missing_mask);
            overlap.n_markers.min(n_markers)
        } else {
            0
        };

        let chrom = target_gt.marker(MarkerIdx::new(0)).chrom;

        // Compute MAF for each marker (used by IBS2 and two-stage phasing)
        // Includes reference panel allele counts if available, ensuring homozygous
        // markers in small target panels are correctly classified as high-frequency.
        let maf: Vec<f32> =
            if let (Some(ref_gt), Some(alignment)) = (&self.reference_gt, &self.alignment) {
                (0..n_markers)
                    .map(|m| {
                        let m_idx = MarkerIdx::new(m as u32);
                        let mut target_alt = target_gt.column(m_idx).alt_count() as usize;
                        let target_total = target_gt.n_haplotypes();
                        if let Some(ref_m) = alignment.target_to_ref(m_idx) {
                            let ref_col = ref_gt.column(ref_m);
                            let ref_total = ref_gt.n_haplotypes();
                            let ref_alt = ref_col.alt_count() as usize;

                            if let Some(Some(mapping)) = alignment.allele_mappings.get(m) {
                                let ref_equiv_of_alt =
                                    mapping.targ_to_ref.get(1).copied().unwrap_or(-1);
                                if ref_equiv_of_alt == 1 {
                                    target_alt += ref_alt;
                                } else if ref_equiv_of_alt == 0 {
                                    target_alt += ref_total - ref_alt;
                                }
                            }

                            let total = target_total + ref_total;
                            if total == 0 {
                                0.0
                            } else {
                                let freq = target_alt as f32 / total as f32;
                                freq.min(1.0 - freq)
                            }
                        } else {
                            let freq = target_alt as f32 / target_total as f32;
                            freq.min(1.0 - freq)
                        }
                    })
                    .collect()
            } else {
                (0..n_markers)
                    .map(|m| target_gt.column(MarkerIdx::new(m as u32)).maf() as f32)
                    .collect()
            };

        // Build hi-frequency / rare marker sets for Stage 1/2
        let rare_threshold = self.config.rare;
        let hi_freq_markers: Vec<usize> = (0..n_markers)
            .filter(|&m| maf[m] >= rare_threshold)
            .collect();
        let has_missing = |m: usize| -> bool {
            let m_idx = MarkerIdx::new(m as u32);
            (0..n_haps).any(|h| target_gt.allele(m_idx, HapIdx::new(h as u32)) == 255)
        };
        let rare_markers: Vec<usize> = (0..n_markers)
            .filter(|&m| (maf[m] < rare_threshold && maf[m] > 0.0) || has_missing(m))
            .collect();
        let hi_freq_to_orig: Vec<usize> = hi_freq_markers.clone();

        // Genetic map for this window
        let marker_map = if let Some(map) = gen_maps.get(chrom) {
            MarkerMap::create(target_gt.markers(), map)
        } else {
            MarkerMap::from_positions(target_gt.markers())
        };
        let gen_positions_vec = marker_map.gen_positions().to_vec();
        let hi_freq_gen_positions: Vec<f64> = hi_freq_markers
            .iter()
            .map(|&m| gen_positions_vec[m])
            .collect();

        let stage1_gen_dists: Vec<f64> = if hi_freq_markers.len() > 1 {
            hi_freq_markers
                .windows(2)
                .map(|w| gen_positions_vec[w[1]] - gen_positions_vec[w[0]])
                .collect()
        } else {
            Vec::new()
        };

        if let Some(bb) = &self.telemetry {
            bb.set_stage(Stage::PhasingMain);
            bb.set_producer_stage(Stage::PhasingMain);
            bb.set_total_samples(n_samples as u64);
            bb.set_samples_processed(0);
            bb.set_total_markers(hi_freq_markers.len() as u64);
            bb.set_markers_processed(0);
            bb.set_total_iterations(1);
            bb.set_current_iteration(1);
        }

        // Create sample phases with overlap markers pre-phased
        let confidence_by_sample = build_sample_confidence(&target_gt);
        let phase_mask = target_gt.phase_mask();
        let has_input_phase = phase_mask
            .as_ref()
            .map(|mask| (0..mask.n_rows()).any(|row| mask.row_has_any_set(row)))
            .unwrap_or(false);
        let mut sample_phases = self.create_sample_phases_with_overlap(
            &geno,
            &missing_mask,
            overlap_markers,
            phased_overlap,
            &confidence_by_sample,
            phase_mask,
        );

        if self.reference_gt.is_none() {
            // Fallback: iterative HMM phasing without reference (overlap-aware)
            let n_burnin = self.config.burnin.min(3);
            let n_iterations = self.config.iterations.min(6);
            let total_iterations = n_burnin + n_iterations;
            if let Some(bb) = &self.telemetry {
                bb.set_total_samples(n_samples as u64);
                bb.set_samples_processed(0);
                bb.set_total_markers(n_markers as u64);
                bb.set_markers_processed(0);
                bb.set_total_iterations(total_iterations as u64);
                bb.set_current_iteration(0);
            }

            let gen_dists: Vec<f64> = (0..n_markers.saturating_sub(1))
                .map(|m| {
                    let pos1 = target_gt.marker(MarkerIdx::new(m as u32)).pos;
                    let pos2 = target_gt.marker(MarkerIdx::new((m + 1) as u32)).pos;
                    gen_maps.gen_dist(chrom, pos1, pos2)
                })
                .collect();
            let mut p_recomb: Vec<f32> = std::iter::once(0.0f32)
                .chain(gen_dists.iter().map(|&d| self.params.p_recomb(d)))
                .collect();

            let mut mcmc_paths: Vec<Option<GlobalMosaicPaths>> = vec![None; n_samples];
            let mut cohort_calibration: Option<CohortCalibration> = None;
            for it in 0..total_iterations {
                let is_burnin = it < n_burnin;
                self.params.lr_threshold = self.params.lr_threshold_for_iteration(it);
                if let Some(bb) = &self.telemetry {
                    let stage = if is_burnin {
                        Stage::PhasingBurnin
                    } else {
                        Stage::PhasingMain
                    };
                    bb.set_stage(stage);
                    bb.set_producer_stage(stage);
                    bb.set_current_iteration((it + 1) as u64);
                    bb.set_samples_processed(0);
                    bb.set_markers_processed(0);
                }

                let atomic_estimates = if is_burnin && self.config.em {
                    Some(crate::model::parameters::AtomicParamEstimates::new())
                } else {
                    None
                };
                let mut cohort_stats = if is_burnin && it == 0 {
                    Some(vec![SampleCohortStats::default(); n_samples])
                } else {
                    None
                };

                self.run_phase_baum_iteration(
                    target_gt,
                    &mut geno,
                    &p_recomb,
                    &gen_dists,
                    &mut sample_phases,
                    &mut mcmc_paths,
                    cohort_calibration
                        .as_ref()
                        .map(|cal| cal.sample_p_mismatch.as_slice()),
                    cohort_stats.as_deref_mut(),
                    atomic_estimates.as_ref(),
                    &confidence_by_sample,
                )?;
                if let Some(bb) = &self.telemetry {
                    bb.set_samples_processed(n_samples as u64);
                    bb.set_markers_processed(n_markers as u64);
                }

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
                    if params_updated {
                        p_recomb = std::iter::once(0.0f32)
                            .chain(gen_dists.iter().map(|&d| self.params.p_recomb(d)))
                            .collect();
                    }
                }

                if let Some(stats) = cohort_stats {
                    if let Some(model) = fit_cohort_calibration(&stats, self.params.p_mismatch) {
                        let preview: Vec<String> = model
                            .cohort_p_mismatch
                            .iter()
                            .zip(model.cohort_sizes.iter())
                            .map(|(&p, &n)| format!("{:.6} (n={})", p, n))
                            .collect();
                        eprintln!(
                            "Cohort calibration enabled: {} cohorts, p_mismatch={}",
                            model.cohort_p_mismatch.len(),
                            preview.join(", ")
                        );
                        cohort_calibration = Some(model);
                    } else {
                        eprintln!("Cohort calibration skipped: insufficient structure in burn-in");
                    }
                }
            }
        } else {
            let ref_gt = match self.reference_gt.as_ref() {
                Some(r) => r.clone(),
                None => unreachable!(),
            };
            let alignment = self.alignment.as_ref().ok_or_else(|| {
                crate::error::ReagleError::config("Reference alignment missing for beam phasing")
            })?;

            let packed_ref =
                PackedRefView::build_sparse(&target_gt, &ref_gt, alignment, &hi_freq_to_orig)?;
            let beam_config = BeamConfig::default();
            let beam_index = PbwtBeamIndex::build(
                &ref_gt,
                alignment,
                &hi_freq_to_orig,
                &hi_freq_gen_positions,
                beam_config.inject_k,
                beam_config.inject_interval,
                self.params.recomb_intensity,
            );
            let pbwt_stats: Option<Vec<(f32, f32, f32, f32)>> = Some(
                (0..hi_freq_to_orig.len())
                    .map(|hi_idx| beam_index.stats_for_hi(hi_idx))
                    .collect(),
            );
            let phaser = BeamPhaser::new(&packed_ref, &self.params, beam_config);

            // Compute allele frequencies for TMRCA-aware switch costs
            let hi_freq_allele_freqs: Option<Vec<(f32, f32)>> = {
                let n_ref_haps = packed_ref.n_ref_haps();
                if n_ref_haps > 0 {
                    Some(
                        hi_freq_to_orig
                            .iter()
                            .map(|&orig_m| {
                                let mut count0 = 0u32;
                                let mut count1 = 0u32;
                                for h in 0..n_ref_haps {
                                    match packed_ref.ref_allele_targ(orig_m, h) {
                                        Some(0) => count0 += 1,
                                        Some(1) => count1 += 1,
                                        _ => {}
                                    }
                                }
                                let total = (count0 + count1).max(1) as f32;
                                (count0 as f32 / total, count1 as f32 / total)
                            })
                            .collect(),
                    )
                } else {
                    None
                }
            };

            let mut threaded_haps_vec = self.build_phasing_prescan_states(
                target_gt,
                &geno,
                Some(ref_gt.as_ref()),
                self.alignment.as_ref(),
                hi_freq_markers.len(),
                n_samples,
                &hi_freq_gen_positions,
                self.config.imp_step,
                Some(&hi_freq_to_orig),
            )?;

            let n_target_haps = target_gt.n_haplotypes();
            let beam_donors: Vec<std::sync::Mutex<Vec<usize>>> = (0..n_samples)
                .map(|_| std::sync::Mutex::new(Vec::new()))
                .collect();
            sample_phases
                .par_iter_mut()
                .enumerate()
                .for_each(|(s, sp)| {
                    let mut active_pool = ActivePool::new(packed_ref.n_ref_haps());
                    let mut tmp = vec![
                        crate::model::types::CombinedHapId::from(0u32);
                        threaded_haps_vec[s].n_states()
                    ];
                    let th = &threaded_haps_vec[s];
                    th.materialize_at(0, &mut tmp);
                    for id in tmp.iter().copied() {
                        let hid = id.as_u32() as usize;
                        if hid >= n_target_haps {
                            let ref_id = hid - n_target_haps;
                            if ref_id < packed_ref.n_ref_haps() {
                                active_pool.add(ref_id);
                            }
                        }
                    }

                    let hard_threshold_nats = if beam_config.prune_tolerance > 0 {
                        Some((beam_config.prune_tolerance as f64) / 1_000_000.0)
                    } else {
                        None
                    };
                    let constraint_max_expected = Some(beam_config.switch_candidates.max(2) as f64);
                    let condensed = CondensedTarget::build(
                        sp,
                        &hi_freq_to_orig,
                        &hi_freq_gen_positions,
                        hi_freq_allele_freqs.as_deref(),
                        pbwt_stats.as_deref(),
                        &packed_ref,
                        &self.params,
                        hard_threshold_nats,
                        constraint_max_expected,
                    );
                    let mut injector = PbwtInjector::new(
                        &beam_index,
                        packed_ref.n_ref_haps(),
                        beam_config.inject_k,
                    );
                    let fwd = phaser.phase_sample(&condensed, sp, &mut active_pool, &mut injector);

                    if !condensed.call_sites.is_empty() {
                        let mut active_pool_rev = ActivePool::new(packed_ref.n_ref_haps());
                        th.materialize_at(0, &mut tmp);
                        for id in tmp.iter().copied() {
                            let hid = id.as_u32() as usize;
                            if hid >= n_target_haps {
                                let ref_id = hid - n_target_haps;
                                if ref_id < packed_ref.n_ref_haps() {
                                    active_pool_rev.add(ref_id);
                                }
                            }
                        }
                        let condensed_rev = condensed.reversed(&hi_freq_gen_positions);
                        let mut injector_rev = PbwtInjector::new(
                            &beam_index,
                            packed_ref.n_ref_haps(),
                            beam_config.inject_k,
                        );
                        let mut sp_rev = sp.clone();
                        let bwd = phaser.phase_sample(
                            &condensed_rev,
                            &mut sp_rev,
                            &mut active_pool_rev,
                            &mut injector_rev,
                        );
                        let mut p_swapped_bwd = bwd.p_swapped;
                        p_swapped_bwd.reverse();
                        let combined = combine_swap_probs(&fwd.p_swapped, &p_swapped_bwd);
                        for (i, &swapped) in fwd.decisions.iter().enumerate() {
                            let m = condensed.call_sites[i].marker.as_usize();
                            let p = combined.get(i).copied().unwrap_or(0.5);
                            let conf = if has_input_phase {
                                if swapped { p } else { 1.0 - p }
                            } else {
                                0.5
                            };
                            sp.set_phase_confidence(m, conf);
                        }
                    }

                    if let Ok(mut slot) = beam_donors[s].lock() {
                        slot.clear();
                        let list = active_pool.list();
                        let cap = beam_config
                            .beam_width
                            .max(beam_config.inject_k.saturating_mul(2))
                            .max(beam_config.switch_candidates.saturating_mul(4));
                        if list.len() > cap {
                            slot.extend_from_slice(&list[list.len() - cap..]);
                        } else {
                            slot.extend_from_slice(list);
                        }
                    }
                });

            // Feed beam-selected donors back into the HMM state set.
            if packed_ref.n_ref_haps() > 0 {
                let offset = n_target_haps as u32;
                for s in 0..n_samples {
                    let Ok(donors) = beam_donors[s].lock() else {
                        continue;
                    };
                    if donors.is_empty() {
                        continue;
                    }
                    let threaded_haps = &mut threaded_haps_vec[s];
                    let mut existing = vec![CombinedHapId::from(0u32); threaded_haps.n_states()];
                    threaded_haps.materialize_at(0, &mut existing);
                    let mut seen: HashSet<u32> =
                        HashSet::with_capacity(existing.len() + donors.len());
                    for id in existing {
                        seen.insert(id.as_u32());
                    }
                    for &h in donors.iter() {
                        let combined = combined_from_ref(RefHapId::from(h), offset);
                        let id = combined.as_u32();
                        if seen.insert(id) {
                            threaded_haps.push_new(combined);
                        }
                    }
                }
            }

            let stage1_blocks = partition_markers_by_cm(
                &hi_freq_gen_positions,
                stage1_block_cm(&hi_freq_gen_positions),
            );
            let ibs2 = Ibs2::new(target_gt, gen_maps, chrom, &maf);
            let stage1_p_recomb: Vec<f32> = std::iter::once(0.0f32)
                .chain(stage1_gen_dists.iter().map(|&d| self.params.p_recomb(d)))
                .collect();

            // Micro-HMM refinement on hi-frequency markers (single pass).
            let mut mcmc_paths: Vec<Option<GlobalMosaicPaths>> = vec![None; n_samples];
            self.run_phase_baum_iteration_stage1(
                target_gt,
                &mut geno,
                &threaded_haps_vec,
                &stage1_p_recomb,
                &stage1_gen_dists,
                &hi_freq_to_orig,
                &stage1_blocks,
                &ibs2,
                &mut sample_phases,
                &mut mcmc_paths,
                None,
                None,
                None,
                None,
                0,
            )?;
        }

        // Sync final phase state from SamplePhase to MutableGenotypes
        self.sync_sample_phases_to_geno(&sample_phases, &mut geno);

        // STAGE 2: Phase rare markers using HMM state probability interpolation
        // Now returns state probabilities for the next overlap region if requested

        let stage1_p_recomb: Vec<f32> = std::iter::once(0.0f32)
            .chain(stage1_gen_dists.iter().map(|&d| self.params.p_recomb(d)))
            .collect();

        let next_overlap_handoff = if !rare_markers.is_empty() && hi_freq_markers.len() >= 2 {
            eprintln!(
                "Stage 2: Phasing {} rare markers using HMM interpolation...",
                rare_markers.len()
            );
            if let Some(bb) = &self.telemetry {
                bb.set_stage(Stage::PhasingStage2);
                bb.set_op("Phasing stage2: HMM interpolation");
                bb.set_total_iterations(0);
                bb.set_current_iteration(0);
                bb.set_total_markers(rare_markers.len() as u64);
                bb.set_markers_processed(0);
                bb.set_samples_processed(0);
            }
            let handoff = self.phase_rare_markers_with_hmm(
                target_gt,
                &mut geno,
                &hi_freq_markers,
                &gen_positions_vec,
                &hi_freq_gen_positions,
                &stage1_p_recomb,
                &mut sample_phases,
                &maf,
                rare_threshold,
                phased_overlap,
                next_overlap_start,
            );
            if let Some(bb) = &self.telemetry {
                bb.set_markers_processed(rare_markers.len() as u64);
                bb.set_samples_processed(n_samples as u64);
            }

            // Sync again after Stage 2
            self.sync_sample_phases_to_geno(&sample_phases, &mut geno);
            handoff
        } else {
            None
        };

        Ok((
            self.build_final_matrix(target_gt, &geno, &sample_phases),
            next_overlap_handoff,
        ))
    }

    /// Apply overlap constraint from previous window's phased output
    ///
    /// This sets the alleles in the overlap region to match the previous window's
    /// phased output, ensuring phase continuity.
    fn apply_overlap_constraint(
        &self,
        geno: &mut MutableGenotypes,
        overlap: &PhasedOverlap,
        missing_mask: &[BitBox<u8, Lsb0>],
    ) {
        let n_overlap = overlap.n_markers.min(geno.n_markers());
        let n_haps = overlap.n_haps.min(geno.n_haps());

        for h in 0..n_haps {
            let h_idx = HapIdx::new(h as u32);
            for m in 0..n_overlap {
                if missing_mask[h][m] {
                    continue;
                }
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
        overlap: Option<&PhasedOverlap>,
        confidence_by_sample: &[Vec<f32>],
        phase_mask: Option<&crate::data::storage::matrix::BitMatrix>,
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
                    .filter(|&m| {
                        missing_mask[hap1.as_usize()][m] || missing_mask[hap2.as_usize()][m]
                    })
                    .collect();

                // Overlap is treated as a hard phase lock only when the current input
                // actually observed both haplotypes and overlap provided concrete alleles.
                let unphased: Vec<usize> = (0..n_markers)
                    .filter(|&m| {
                        let a1 = alleles1[m];
                        let a2 = alleles2[m];
                        if a1 == a2 {
                            return false;
                        }
                        if missing_mask[hap1.as_usize()][m] || missing_mask[hap2.as_usize()][m] {
                            return false;
                        }
                        if m < overlap_markers {
                            if let Some(ov) = overlap {
                                if m < ov.n_markers
                                    && ov.allele(m, hap1.as_usize()) != 255
                                    && ov.allele(m, hap2.as_usize()) != 255
                                {
                                    return false;
                                }
                            }
                        }
                        match phase_mask.and_then(|mask| mask.get(m, s)) {
                            Some(0) => true,
                            Some(_) => false,
                            None => true,
                        }
                    })
                    .collect();

                let conf = &confidence_by_sample[s];
                SamplePhase::new(n_markers, &alleles1, &alleles2, conf, &unphased, &missing)
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
        phase_mask: Option<&crate::data::storage::matrix::BitMatrix>,
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

                // Unphased hets: only those explicitly unphased in the input phase mask
                let unphased: Vec<usize> = (0..n_markers)
                    .filter(|&m| {
                        let a1 = alleles1[m];
                        let a2 = alleles2[m];
                        if a1 == a2 || a1 == 255 || a2 == 255 {
                            return false;
                        }
                        match phase_mask.and_then(|mask| mask.get(m, s)) {
                            Some(0) => true,
                            Some(_) => false,
                            None => true,
                        }
                    })
                    .collect();

                let conf = &confidence_by_sample[s];
                SamplePhase::new(n_markers, &alleles1, &alleles2, conf, &unphased, &missing)
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
        let mut alleles_flat = Vec::with_capacity(n_subset.saturating_mul(n_haps));
        for &orig_m in marker_indices {
            let marker_slice = geno.marker_alleles(orig_m);
            alleles_flat.extend_from_slice(&marker_slice[..n_haps]);
        }

        BidirectionalPhaseIbs::build_for_subset_flat(alleles_flat, n_haps, n_subset, marker_indices)
    }

    /// Build bidirectional PBWT for a subset of markers using target+reference (composite).
    fn build_bidirectional_pbwt_subset_with_ref<RefPanelSpace>(
        &self,
        target_geno: &MutableGenotypes,
        ref_gt: &GenotypeMatrix<crate::data::storage::phase_state::Phased, RefPanelSpace>,
        alignment: &MarkerAlignment<crate::data::AnyMarkerSpace, RefPanelSpace>,
        marker_indices: &[usize],
    ) -> BidirectionalPhaseIbs {
        let n_subset = marker_indices.len();
        let n_target_haps = target_geno.n_haps();
        let n_ref_haps = ref_gt.n_haplotypes();
        let n_total_haps = n_target_haps + n_ref_haps;

        let view = GenotypeView::CompositeSubset {
            target: target_geno,
            reference: ref_gt,
            alignment,
            subset: marker_indices,
            n_target_haps,
        };
        let haps: Vec<HapIdx> = (0..n_total_haps).map(|h| HapIdx::new(h as u32)).collect();

        let mut alleles_flat = vec![255u8; n_subset.saturating_mul(n_total_haps)];
        for i in 0..n_subset {
            let row_start = i.saturating_mul(n_total_haps);
            let row_end = row_start.saturating_add(n_total_haps);
            view.fill_batch(
                MarkerIdx::new(i as u32),
                &haps,
                &mut alleles_flat[row_start..row_end],
            );
        }

        let mut pbwt = BidirectionalPhaseIbs::build_for_subset_flat(
            alleles_flat,
            n_total_haps,
            n_subset,
            marker_indices,
        );
        pbwt.set_reference_start_hap(n_target_haps as u32);
        pbwt
    }

    fn build_phasing_prescan_states<RefPanelSpace>(
        &self,
        target_gt: &GenotypeMatrix,
        target_geno: &MutableGenotypes,
        ref_gt: Option<&GenotypeMatrix<crate::data::storage::phase_state::Phased, RefPanelSpace>>,
        alignment: Option<&MarkerAlignment<crate::data::AnyMarkerSpace, RefPanelSpace>>,
        n_markers: usize,
        n_samples: usize,
        gen_positions: &[f64],
        step_cm: f32,
        marker_map: Option<&[usize]>,
    ) -> Result<Vec<crate::model::states::ThreadedHaps<CombinedHapSpace>>> {
        let telemetry_snapshot = self.telemetry.as_ref().map(|bb| bb.snapshot());
        let n_haps = target_geno.n_haps();
        let n_ref_haps = ref_gt.map(|r| r.n_haplotypes()).unwrap_or(n_haps).max(1);
        let step_cm = PBWT_SELECT_BLOCK_CM.max(step_cm as f64);

        let window_blocks = partition_markers_by_cm(gen_positions, stage1_block_cm(gen_positions));
        if window_blocks.is_empty() {
            return Err(crate::error::ReagleError::vcf(
                "Pre-scan produced no windows for phasing".to_string(),
            ));
        }
        let window_markers_est = window_blocks
            .iter()
            .map(|(s, e)| e.saturating_sub(*s))
            .max()
            .unwrap_or(STAGE1_BLOCK_MIN_MARKERS)
            .max(1);
        let mut n_threads = self
            .config
            .nthreads
            .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
            .unwrap_or(1);
        let mut avail_bytes = crate::utils::memory::available_memory_bytes().unwrap_or(0);
        if avail_bytes < MIN_AVAIL_BYTES_FOR_PLANNING {
            avail_bytes = 0;
        }
        let mut auto_budget =
            estimate_phase_state_budget(avail_bytes, n_threads, window_markers_est);
        while auto_budget < 64 && n_threads > 1 {
            n_threads = (n_threads / 2).max(1);
            auto_budget = estimate_phase_state_budget(avail_bytes, n_threads, window_markers_est);
        }
        let mut per_window_cap = if self.config.phase_states == 0 {
            if avail_bytes == 0 || auto_budget == 0 {
                n_ref_haps
            } else {
                auto_budget.max(1)
            }
        } else {
            self.config.phase_states
        };
        per_window_cap = if self.config.phase_states == 0 {
            let auto_target = self
                .params
                .n_states
                .max(1)
                .saturating_mul(PHASE_AUTO_PRESCAN_MULT)
                .clamp(PHASE_AUTO_PRESCAN_MIN, PHASE_AUTO_PRESCAN_MAX)
                .min(n_ref_haps.max(1));
            per_window_cap.min(auto_target).max(1)
        } else {
            per_window_cap.min(n_ref_haps).max(1)
        };
        if self.config.phase_states == 0 {
            eprintln!(
                "Phasing prescan: per_window_cap={} threads={} available_mb={} window_markers={}",
                per_window_cap,
                n_threads,
                avail_bytes / (1024 * 1024),
                window_markers_est
            );
        }
        let num_windows = window_blocks.len();

        let mut boundary_cm = Vec::with_capacity(num_windows.saturating_sub(1));
        for w in 0..num_windows.saturating_sub(1) {
            let (_, end) = window_blocks[w];
            let (next_start, _) = window_blocks[w + 1];
            let left = gen_positions[end
                .saturating_sub(1)
                .min(gen_positions.len().saturating_sub(1))];
            let right = gen_positions[next_start.min(gen_positions.len().saturating_sub(1))];
            let dist = (right - left).abs().max(1e-9);
            boundary_cm.push(dist);
        }

        let n_full_markers = target_gt.n_markers();
        let (ref_columns, ref_index_map): (Vec<GenotypeColumn>, Option<Vec<usize>>) =
            if let Some(ref_gt) = ref_gt {
                let n_ref_markers = ref_gt.n_markers();
                let mut ref_indices: Vec<usize> = Vec::with_capacity(n_markers);
                for m in 0..n_markers {
                    let orig_m = marker_map.map(|map| map[m]).unwrap_or(m);
                    if let Some(alignment) = alignment {
                        if let Some(ref_m) = alignment.target_to_ref(MarkerIdx::new(orig_m as u32))
                        {
                            ref_indices.push(ref_m.as_usize());
                        }
                    } else if orig_m < n_ref_markers {
                        ref_indices.push(orig_m);
                    }
                }
                ref_indices.sort_unstable();
                ref_indices.dedup();

                let mut map = vec![usize::MAX; n_ref_markers];
                let mut cols: Vec<GenotypeColumn> = Vec::with_capacity(ref_indices.len());
                for (i, ref_m) in ref_indices.iter().enumerate() {
                    map[*ref_m] = i;
                    cols.push(ref_gt.column(MarkerIdx::new(*ref_m as u32)).clone());
                }
                (cols, Some(map))
            } else {
                let mut map = vec![usize::MAX; n_full_markers];
                let mut cols: Vec<GenotypeColumn> = Vec::with_capacity(n_markers);
                for m in 0..n_markers {
                    let orig_m = marker_map.map(|map| map[m]).unwrap_or(m);
                    if orig_m >= n_full_markers {
                        continue;
                    }
                    let alleles = target_geno.marker_alleles(orig_m);
                    let n_alleles = target_gt
                        .markers()
                        .marker(MarkerIdx::new(orig_m as u32))
                        .n_alleles()
                        .max(1);
                    map[orig_m] = cols.len();
                    cols.push(GenotypeColumn::from_alleles(&alleles, n_alleles));
                }
                (cols, Some(map))
            };

        let freqs = compute_ref_freqs(
            target_gt,
            &ref_columns,
            alignment,
            marker_map,
            ref_index_map.as_deref(),
            n_markers,
        );
        let phase_mask = target_gt.phase_mask();
        let mut mask_unphased_hets = vec![false; n_haps / 2];
        let mut anchors_by_hap: Vec<Vec<(usize, u8, u8)>> = vec![Vec::new(); n_haps];
        let mut ref_col_for_marker = vec![usize::MAX; n_markers];
        if ref_gt.is_some() {
            if let Some(ref_gt) = ref_gt {
                let n_ref_markers = ref_gt.n_markers();
                for m in 0..n_markers {
                    let orig_m = marker_map.map(|map| map[m]).unwrap_or(m);
                    if let Some(alignment) = alignment {
                        if let Some(ref_m) = alignment.target_to_ref(MarkerIdx::new(orig_m as u32))
                        {
                            if let Some(map) = &ref_index_map {
                                let idx = map.get(ref_m.as_usize()).copied().unwrap_or(usize::MAX);
                                ref_col_for_marker[m] = idx;
                            }
                        }
                    } else if orig_m < n_ref_markers {
                        if let Some(map) = &ref_index_map {
                            let idx = map.get(orig_m).copied().unwrap_or(usize::MAX);
                            ref_col_for_marker[m] = idx;
                        }
                    }
                }
            }
            for s in 0..(n_haps / 2) {
                let hap1 = s * 2;
                let hap2 = s * 2 + 1;
                for m in 0..n_markers {
                    let orig_m = marker_map.map(|map| map[m]).unwrap_or(m);
                    let phased = phase_mask.and_then(|mask| mask.get(orig_m, s)).unwrap_or(0);
                    if phased == 0 {
                        continue;
                    }
                    let a1 = target_geno.get(orig_m, HapIdx::new(hap1 as u32));
                    let a2 = target_geno.get(orig_m, HapIdx::new(hap2 as u32));
                    if a1 == 255 || a2 == 255 || a1 == a2 {
                        continue;
                    }
                    anchors_by_hap[hap1].push((m, a1, a2));
                    anchors_by_hap[hap2].push((m, a2, a1));
                    mask_unphased_hets[s] = true;
                }
            }
        }
        if n_markers <= 60 {
            let sample = 0usize;
            let mut phased_count = 0usize;
            let mut unphased_hets = 0usize;
            for m in 0..n_markers {
                let phased = phase_mask.and_then(|mask| mask.get(m, sample)).unwrap_or(0);
                if phased != 0 {
                    phased_count += 1;
                }
                let a1 = target_geno.get(m, HapIdx::new((sample * 2) as u32));
                let a2 = target_geno.get(m, HapIdx::new((sample * 2 + 1) as u32));
                if phased == 0 && a1 != 255 && a2 != 255 && a1 != a2 {
                    unphased_hets += 1;
                }
            }
            eprintln!(
                "[prescan debug] mask_unphased_hets[0]={} phased={} unphased_hets={}",
                mask_unphased_hets.get(sample).copied().unwrap_or(false),
                phased_count,
                unphased_hets
            );
        }
        let avail = crate::utils::memory::available_memory_bytes().unwrap_or(0);
        let batch_size = estimate_scan_batch_size(avail, n_ref_haps, n_haps).max(1);
        let batches_per_window = (n_haps + batch_size - 1) / batch_size;
        let total_batches = num_windows.saturating_mul(batches_per_window).max(1);

        if let Some(bb) = &self.telemetry {
            bb.set_stage(Stage::PhasingPrescan);
            bb.set_producer_stage(Stage::PhasingPrescan);
            bb.set_op("Phasing prescan: PBWT scoring");
            bb.set_total_windows(num_windows as u64);
            bb.set_current_window(0);
            bb.set_total_markers(total_batches as u64);
            bb.set_markers_processed(0);
        }

        let mut scores_by_window_by_hap: Vec<Vec<Vec<(usize, f32)>>> =
            vec![Vec::with_capacity(num_windows); n_haps];

        for (window_idx, &(start, end)) in window_blocks.iter().enumerate() {
            if let Some(bb) = &self.telemetry {
                bb.set_current_window((window_idx + 1) as u64);
                let span_cm = if end > start && end <= gen_positions.len() {
                    (gen_positions[end - 1] - gen_positions[start]).abs()
                } else {
                    0.0
                };
                let span_markers = end.saturating_sub(start);
                bb.set_op(&format!(
                    "Phasing prescan: PBWT scoring (window {}/{}, span_cm={:.3}, markers={})",
                    window_idx + 1,
                    num_windows,
                    span_cm,
                    span_markers
                ));
            }
            let mut informative: Vec<bool> = vec![false; end.saturating_sub(start)];
            if !informative.is_empty() {
                for m in start..end {
                    let orig_m = marker_map.and_then(|map| map.get(m).copied()).unwrap_or(m);
                    let mut info = false;
                    if let Some(mask) = phase_mask {
                        if mask.row_has_any_set(orig_m) {
                            info = true;
                        }
                    }
                    if !info {
                        let alleles = target_geno.marker_alleles(orig_m);
                        if alleles.iter().any(|&a| a != 255) {
                            info = true;
                        }
                    }
                    informative[m - start] = info;
                }
            }
            let sampling = build_sampling_points(
                &gen_positions[start..end],
                step_cm,
                PBWT_MIN_MARKER_STEP,
                Some(&informative),
            );
            let k_per_hap = per_window_cap
                .saturating_mul(PBWT_PER_WINDOW_MULT)
                .max(PBWT_MIN_PER_HAP)
                .min(PBWT_MAX_PER_HAP)
                .max(1)
                .min(n_ref_haps.max(1));

            let mut batch_start = 0usize;
            let mut batch_haps_buf: Vec<usize> = Vec::with_capacity(batch_size);
            let mut window_scores_buf: Vec<Vec<f32>> = Vec::with_capacity(batch_size);
            while batch_start < n_haps {
                let batch_end = (batch_start + batch_size).min(n_haps);
                batch_haps_buf.clear();
                batch_haps_buf.extend(batch_start..batch_end);
                let batch_len = batch_haps_buf.len();

                if window_scores_buf.len() < batch_len {
                    let needed = batch_len - window_scores_buf.len();
                    window_scores_buf
                        .extend((0..needed).map(|_| vec![f32::NEG_INFINITY; n_ref_haps]));
                }
                for row in window_scores_buf.iter_mut().take(batch_len) {
                    if row.len() != n_ref_haps {
                        row.resize(n_ref_haps, f32::NEG_INFINITY);
                    } else {
                        row.fill(f32::NEG_INFINITY);
                    }
                }

                score_window_batch_pbwt_segment(
                    &batch_haps_buf,
                    target_gt,
                    target_geno,
                    &ref_columns,
                    phase_mask,
                    Some(&mask_unphased_hets),
                    alignment,
                    &freqs,
                    (start, end),
                    k_per_hap,
                    &sampling,
                    &mut window_scores_buf[..batch_len],
                    ref_gt.is_none(),
                    marker_map,
                    ref_index_map.as_deref(),
                );

                let base_top_m = per_window_cap
                    .saturating_mul(PBWT_PER_WINDOW_MULT)
                    .max(per_window_cap)
                    .min(n_ref_haps.max(1));
                for (i, &hap_idx) in batch_haps_buf.iter().enumerate() {
                    let top_m =
                        adaptive_prescan_top_m(&window_scores_buf[i], base_top_m, n_ref_haps);
                    let top = select_top_k(&window_scores_buf[i], top_m);
                    scores_by_window_by_hap[hap_idx].push(top);
                }

                batch_start = batch_end;
                if let Some(bb) = &self.telemetry {
                    bb.add_markers(1);
                }
            }
        }
        if n_markers <= 60 {
            if let Some(list) = scores_by_window_by_hap.get(0).and_then(|v| v.get(0)) {
                let mut hero_score = None;
                let has_hero = list.iter().any(|(h, _)| {
                    if *h == 98 {
                        hero_score = list.iter().find(|(hh, _)| *hh == 98).map(|(_, s)| *s);
                        true
                    } else {
                        false
                    }
                });
                eprintln!(
                    "[prescan debug] window0_top={:?} hero98_in_top={} hero98_score={:?}",
                    list.iter().take(10).collect::<Vec<_>>(),
                    has_hero,
                    hero_score
                );
            }
        }

        let per_window_caps = vec![per_window_cap; num_windows];
        let global_slot_budget = per_window_caps.iter().copied().sum::<usize>().max(1);
        let exclude_self = ref_gt.is_none();
        let debug_watch = vec![198usize, 199usize];
        let ref_has_panel = ref_gt.is_some();
        let out: Vec<crate::model::states::ThreadedHaps<CombinedHapSpace>> = (0..n_samples)
            .into_par_iter()
            .map(|s| {
                let hap1 = s * 2;
                let hap2 = s * 2 + 1;
                let mut dense_merge_buffer = vec![f32::NEG_INFINITY; n_ref_haps.max(1)];
                let mut touched_indices: Vec<usize> =
                    Vec::with_capacity(per_window_cap.saturating_mul(PBWT_PER_WINDOW_MULT).max(1));
                let mut window_scores: Vec<Vec<(usize, f32)>> = Vec::with_capacity(num_windows);
                let mut prev_window_scores: Vec<(usize, f32)> = Vec::new();
                for w in 0..num_windows {
                    for &idx in &touched_indices {
                        dense_merge_buffer[idx] = f32::NEG_INFINITY;
                    }
                    touched_indices.clear();

                    for &(h, score) in scores_by_window_by_hap[hap1][w]
                        .iter()
                        .chain(scores_by_window_by_hap[hap2][w].iter())
                    {
                        if h >= dense_merge_buffer.len() {
                            continue;
                        }
                        let current = &mut dense_merge_buffer[h];
                        if current.is_finite() {
                            if score > *current {
                                *current = score;
                            }
                        } else {
                            *current = score;
                            touched_indices.push(h);
                        }
                    }

                    touched_indices.sort_by(|&a, &b| {
                        dense_merge_buffer[b]
                            .partial_cmp(&dense_merge_buffer[a])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let cap = per_window_cap
                        .saturating_mul(PBWT_PER_WINDOW_MULT)
                        .max(per_window_cap)
                        .min(n_ref_haps.max(1));
                    let take = cap.min(touched_indices.len());
                    let mut list: Vec<(usize, f32)> = Vec::with_capacity(take);
                    for &h in touched_indices.iter().take(take) {
                        list.push((h, dense_merge_buffer[h]));
                    }
                    if list.is_empty() {
                        list.extend(prev_window_scores.iter().copied());
                    } else if !prev_window_scores.is_empty() {
                        let mut map: HashMap<usize, f32> = list.iter().copied().collect();
                        for (h, score) in prev_window_scores.iter().copied() {
                            map.entry(h).or_insert(score);
                        }
                        list = map.into_iter().collect();
                    }
                    prev_window_scores = list.clone();
                    window_scores.push(list);
                }

                let abyss = vec![false; n_ref_haps];
                let (mut candidate_haps, mut scores_by_hap) =
                    build_sparse_scores(&window_scores, &abyss);
                let mut candidate_present = vec![false; n_ref_haps.max(1)];
                for &h in &candidate_haps {
                    if h < candidate_present.len() {
                        candidate_present[h] = true;
                    }
                }
                if s == 0 && n_markers <= 60 {
                    let hero_in_candidates = candidate_haps.iter().any(|&h| h == 98);
                    eprintln!(
                        "[prescan debug] hero98_in_candidates={}",
                        hero_in_candidates
                    );
                    if hero_in_candidates {
                        if let Some(pos) = candidate_haps.iter().position(|&h| h == 98) {
                            eprintln!("[prescan debug] hero98_scores={:?}", scores_by_hap.get(pos));
                        }
                    }
                    if n_markers <= 12 && ref_has_panel {
                        let mut all_zero = Vec::new();
                        let mut all_one = Vec::new();
                        for h in 0..n_ref_haps {
                            let hap_idx = HapIdx::new(h as u32);
                            let mut ok0 = true;
                            let mut ok1 = true;
                            for m in 0..n_markers {
                                let ref_col_idx = ref_col_for_marker[m];
                                if ref_col_idx == usize::MAX {
                                    continue;
                                }
                                let ref_al = ref_columns
                                    .get(ref_col_idx)
                                    .map(|c| c.get(hap_idx))
                                    .unwrap_or(255);
                                if ref_al != 0 {
                                    ok0 = false;
                                }
                                if ref_al != 1 {
                                    ok1 = false;
                                }
                                if !ok0 && !ok1 {
                                    break;
                                }
                            }
                            if ok0 {
                                all_zero.push(h);
                            }
                            if ok1 {
                                all_one.push(h);
                            }
                        }
                        let all_zero_preview_len = all_zero.len().min(16);
                        let all_one_preview_len = all_one.len().min(16);
                        eprintln!(
                            "[prescan debug] all_zero_haps_count={} preview={:?} omitted={}",
                            all_zero.len(),
                            &all_zero[..all_zero_preview_len],
                            all_zero.len() - all_zero_preview_len
                        );
                        eprintln!(
                            "[prescan debug] all_one_haps_count={} preview={:?} omitted={}",
                            all_one.len(),
                            &all_one[..all_one_preview_len],
                            all_one.len() - all_one_preview_len
                        );
                        let zero_in_candidates = all_zero
                            .iter()
                            .any(|&h| h < candidate_present.len() && candidate_present[h]);
                        let one_in_candidates = all_one
                            .iter()
                            .any(|&h| h < candidate_present.len() && candidate_present[h]);
                        eprintln!(
                            "[prescan debug] all_zero_in_candidates={} all_one_in_candidates={}",
                            zero_in_candidates, one_in_candidates
                        );
                    }
                }
                if s == 0 {
                    eprintln!(
                        "[prescan] sample={} candidate_haps={} windows={}",
                        s,
                        candidate_haps.len(),
                        num_windows
                    );
                    for &h in &debug_watch {
                        let found = candidate_haps.iter().any(|&c| c == h);
                        eprintln!("[prescan] watch_hap={} present={}", h, found);
                    }
                }
                if ref_has_panel && PBWT_ANCHOR_TOP_HAPS > 0 {
                    let hap1 = s * 2;
                    let hap2 = s * 2 + 1;
                    let p_err_anchor = self.params.p_mismatch;
                    let p_no_err_anchor = 1.0 - p_err_anchor;
                    let mut anchor_scores = vec![0.0f32; n_ref_haps];
                    for &(start, end) in &window_blocks {
                        let mut window_anchors: Vec<(usize, u8, u8)> = Vec::new();
                        if let Some(list) = anchors_by_hap.get(hap1) {
                            for &(m, a1, a2) in list {
                                if m >= start && m < end {
                                    window_anchors.push((m, a1, a2));
                                }
                            }
                        }
                        if let Some(list) = anchors_by_hap.get(hap2) {
                            for &(m, a1, a2) in list {
                                if m >= start && m < end {
                                    window_anchors.push((m, a1, a2));
                                }
                            }
                        }
                        if window_anchors.is_empty() {
                            continue;
                        }
                        anchor_scores.fill(0.0);
                        for h in 0..n_ref_haps {
                            let hap_idx = HapIdx::new(h as u32);
                            let mut score = 0.0f32;
                            for (m, a1, _) in &window_anchors {
                                let ref_col_idx = ref_col_for_marker[*m];
                                if ref_col_idx == usize::MAX {
                                    continue;
                                }
                                let ref_al = ref_columns
                                    .get(ref_col_idx)
                                    .map(|c| c.get(hap_idx))
                                    .unwrap_or(255);
                                let emit =
                                    emit_prob(ref_al, *a1, 1.0, p_no_err_anchor, p_err_anchor);
                                score += emit.max(1e-30).ln();
                            }
                            anchor_scores[h] = score;
                        }
                        let mut idxs: Vec<usize> = (0..n_ref_haps).collect();
                        idxs.sort_by(|&a, &b| {
                            anchor_scores[b]
                                .partial_cmp(&anchor_scores[a])
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                        let take = PBWT_ANCHOR_TOP_HAPS.min(idxs.len());
                        for &h in idxs.iter().take(take) {
                            if h < candidate_present.len() && !candidate_present[h] {
                                candidate_haps.push(h);
                                scores_by_hap.push(Vec::new());
                                candidate_present[h] = true;
                            }
                        }
                    }
                }
                let allocation = allocate_lms_sparse(
                    &scores_by_hap,
                    &candidate_haps,
                    num_windows,
                    &boundary_cm,
                    &self.params,
                    n_ref_haps,
                    global_slot_budget,
                    &per_window_caps,
                );

                let mut selected: Vec<RefHapId> = allocation
                    .intervals_by_hap
                    .into_iter()
                    .map(|(h, _)| RefHapId::from(h))
                    .collect();
                selected.sort_unstable();
                selected.dedup();
                let mut selected_set: HashSet<RefHapId> = selected.iter().copied().collect();
                if PBWT_FORCE_TOP_HAPS > 0 && !window_scores.is_empty() {
                    for &idx in &touched_indices {
                        dense_merge_buffer[idx] = f32::NEG_INFINITY;
                    }
                    touched_indices.clear();
                    for list in &window_scores {
                        for &(h, score) in list {
                            if h >= dense_merge_buffer.len() {
                                continue;
                            }
                            let current = &mut dense_merge_buffer[h];
                            if current.is_finite() {
                                if score > *current {
                                    *current = score;
                                }
                            } else {
                                *current = score;
                                touched_indices.push(h);
                            }
                        }
                    }
                    touched_indices.sort_by(|&a, &b| {
                        dense_merge_buffer[b]
                            .partial_cmp(&dense_merge_buffer[a])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let take = PBWT_FORCE_TOP_HAPS.min(touched_indices.len());
                    for &h in touched_indices.iter().take(take) {
                        let hid = RefHapId::from(h);
                        if selected_set.insert(hid) {
                            selected.push(hid);
                        }
                    }
                }
                if exclude_self {
                    selected.retain(|h| (h.as_usize() / 2) != s);
                }
                if selected.is_empty() {
                    let fallback_cap = per_window_cap.min(n_ref_haps.max(1)).max(1);
                    let mut fallback: Vec<RefHapId> =
                        (0..fallback_cap).map(RefHapId::from).collect();
                    if exclude_self {
                        fallback.retain(|h| (h.as_usize() / 2) != s);
                    }
                    if fallback.is_empty() {
                        fallback.push(RefHapId::from(0usize));
                    }
                    selected = fallback;
                }

                // Order selected haplotypes by prescan score to seed beam initialization.
                let mut selected_set: HashSet<u32> = selected.iter().map(|h| h.as_u32()).collect();
                let mut ranked: Vec<(RefHapId, f32)> = Vec::with_capacity(candidate_haps.len());
                for (idx, &hap) in candidate_haps.iter().enumerate() {
                    if !selected_set.contains(&(hap as u32)) {
                        continue;
                    }
                    let scores = scores_by_hap.get(idx);
                    let mut sum = 0.0f32;
                    if let Some(list) = scores {
                        for &(_, v) in list.iter() {
                            if v.is_finite() {
                                sum += v;
                            }
                        }
                    }
                    ranked.push((RefHapId::from(hap), sum));
                }
                ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let mut ranked_selected: Vec<RefHapId> = Vec::with_capacity(selected.len());
                for (h, _) in ranked {
                    if selected_set.remove(&h.as_u32()) {
                        ranked_selected.push(h);
                    }
                }
                for h in selected {
                    if selected_set.remove(&h.as_u32()) {
                        ranked_selected.push(h);
                    }
                }
                selected = ranked_selected;

                if s == 0 && n_markers <= 60 {
                    let mut selected_dbg = selected.clone();
                    selected_dbg.sort_unstable();
                    eprintln!(
                        "[prescan debug] selected_ref_len={} first={:?}",
                        selected_dbg.len(),
                        selected_dbg.iter().take(10).collect::<Vec<_>>()
                    );
                }

                let offset = if ref_has_panel { n_haps } else { 0 };
                let mut th = crate::model::states::ThreadedHaps::<CombinedHapSpace>::new(
                    selected.len(),
                    selected.len(),
                    n_markers,
                );
                for h in selected {
                    let combined = combined_from_ref(h, offset as u32);
                    th.push_new(combined);
                }
                if s == 0 && n_markers <= 60 {
                    let mut buf = vec![CombinedHapId::from(0u32); th.n_states()];
                    th.materialize_at(0, &mut buf);
                    let mut selected_dbg: Vec<usize> =
                        buf.iter().map(|id| id.as_u32() as usize).collect();
                    selected_dbg.sort_unstable();
                    eprintln!(
                        "[prescan debug] sample=0 selected_combined_len={} first={:?}",
                        selected_dbg.len(),
                        selected_dbg.iter().take(10).collect::<Vec<_>>()
                    );
                }
                th
            })
            .collect();

        if let Some(bb) = &self.telemetry {
            if let Some(prev) = telemetry_snapshot {
                bb.set_stage(prev.stage);
                bb.set_producer_stage(prev.producer_stage);
                bb.set_consumer_stage(prev.consumer_stage);
                bb.set_current_window(prev.current_window);
                bb.set_total_windows(prev.total_windows);
                bb.set_current_iteration(prev.current_iteration);
                bb.set_total_iterations(prev.total_iterations);
                bb.set_samples_processed(prev.samples_processed);
                bb.set_total_samples(prev.total_samples);
                bb.set_markers_processed(prev.markers_processed);
                bb.set_total_markers(prev.total_markers);
                bb.set_op(&prev.current_op);
                bb.set_producer_op(&prev.producer_op);
                bb.set_consumer_op(&prev.consumer_op);
            }
        }

        Ok(out)
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
        sample_phases: &mut [SamplePhase],
        mcmc_paths: &mut [Option<GlobalMosaicPaths>],
        sample_p_mismatch: Option<&[f32]>,
        cohort_stats_out: Option<&mut [SampleCohortStats]>,
        atomic_estimates: Option<&crate::model::parameters::AtomicParamEstimates>,
        confidence_by_sample: &[Vec<f32>],
    ) -> Result<()> {
        let n_samples = geno.n_haps() / 2;
        let n_markers = geno.n_markers();
        let n_haps = geno.n_haps();
        let mut gen_positions = Vec::with_capacity(n_markers);
        gen_positions.push(0.0);
        for i in 1..n_markers {
            let dist = gen_dists.get(i - 1).copied().unwrap_or(0.0);
            gen_positions.push(gen_positions[i - 1] + dist);
        }

        tracing::Span::current().record("n_samples", n_samples);
        tracing::Span::current().record("n_markers", n_markers);

        // Compute total haplotype count (target + reference)
        let n_ref_haps = self
            .reference_gt
            .as_ref()
            .map(|r| r.n_haplotypes())
            .unwrap_or(0);
        let n_total_haps = n_haps + n_ref_haps;

        // No clone needed: the HMM phase is read-only; mutations happen after.
        // We use a scoped immutable borrow that ends before the swap phase.
        // Build composite haplotypes for all samples using streaming PBWT
        // This uses O(N) memory instead of O(M*N) for the PBWT index
        let final_states = self.params.n_states.min(n_total_haps).max(1);
        let ref_gt = self.reference_gt.as_ref().map(|v| v.as_ref());
        let threaded_haps_vec = tracing::info_span!("prescan_selection").in_scope(|| {
            self.build_phasing_prescan_states(
                target_gt,
                geno,
                ref_gt,
                self.alignment.as_ref(),
                n_markers,
                n_samples,
                &gen_positions,
                self.config.imp_step,
                None,
            )
        })?;
        let swap_results: Vec<(
            BitVec<u8, Lsb0>,
            Vec<(usize, f32)>,
            Vec<(usize, f32)>,
            Option<GlobalMosaicPaths>,
            Option<SampleCohortStats>,
        )> = info_span!("build_composite_view").in_scope(|| {
            // Immutable borrow of geno for the entire read phase
            let ref_geno: &MutableGenotypes = geno;

            // Use Composite view when reference panel is available
            let ref_view: GenotypeView<'_, crate::data::AnyMarkerSpace, RefSpace> =
                if let (Some(ref_gt), Some(alignment)) = (&self.reference_gt, &self.alignment) {
                    GenotypeView::Composite {
                        target: ref_geno,
                        reference: ref_gt,
                        alignment,
                        n_target_haps: n_haps,
                    }
                } else {
                    GenotypeView::Mutable(ref_geno)
                };
            let prior_paths = &mcmc_paths[..];
            let sample_phase_view: &[SamplePhase] = &*sample_phases;
            let collect_cohort_stats = cohort_stats_out.is_some();
            let mut swap_results: Vec<(
                BitVec<u8, Lsb0>,
                Vec<(usize, f32)>,
                Vec<(usize, f32)>,
                Option<GlobalMosaicPaths>,
                Option<SampleCohortStats>,
            )> = vec![
                (
                    BitVec::repeat(false, n_markers),
                    Vec::new(),
                    Vec::new(),
                    None,
                    None,
                );
                n_samples
            ];

            tracing::info_span!("hmm_samples").in_scope(|| {
                swap_results.par_iter_mut().enumerate().for_each(
                    |(s, (mask, het_lr_out, het_phase_out, paths_out, cohort_stats_out_one))| {
                        let sample_idx = SampleIdx::new(s as u32);
                        let hap1 = sample_idx.hap1();
                        let hap2 = sample_idx.hap2();
                        let sample_seed = (self.config.seed as u64)
                            .wrapping_add(s as u64)
                            .wrapping_add(0xA5A5_5A5A_D00Du64);

                        // Use pre-built composite haplotypes from streaming PBWT
                        let threaded_haps_full = &threaded_haps_vec[s];
                        let n_states_full = threaded_haps_full.n_states();
                        let mut threaded_haps = Cow::Borrowed(threaded_haps_full);
                        let mut n_states = n_states_full;
                        let mut selection_applied = false;

                        // Convert global prior paths to local paths for this iteration
                        let local_prior = prior_paths[s].as_ref().and_then(|gp| {
                            global_to_local_paths(gp, threaded_haps_full, n_markers)
                        });

                        // 2. Extract current alleles for H1 and H2
                        let seq1 = ref_geno.haplotype(hap1);
                        let seq2 = ref_geno.haplotype(hap2);
                        // Use pre-computed confidence instead of recomputing
                        let sample_conf = &confidence_by_sample[s];
                        let sample_p_err = MismatchProb::new_clamped(
                            sample_p_mismatch
                                .and_then(|v| v.get(s))
                                .copied()
                                .unwrap_or(self.params.p_mismatch),
                        )
                        .get();
                        let sample_p_no_err = 1.0 - sample_p_err;

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
                        if n_states_full > final_states {
                            let mut forced_counts: std::collections::HashMap<usize, u32> =
                                std::collections::HashMap::new();
                            if let Some(prior) = local_prior.as_ref() {
                                for &state in prior.path1.iter().chain(prior.path2.iter()) {
                                    let entry = forced_counts.entry(state as usize).or_insert(0);
                                    *entry += 1;
                                }
                            }

                            let mut forced: Vec<usize> = forced_counts.keys().copied().collect();
                            forced.sort_by(|&a, &b| {
                                let ca = forced_counts.get(&a).copied().unwrap_or(0);
                                let cb = forced_counts.get(&b).copied().unwrap_or(0);
                                cb.cmp(&ca).then_with(|| a.cmp(&b))
                            });

                            let mut can_reuse_prior_subset = false;
                            if let Some(prior) = local_prior.as_ref() {
                                if prior.path1.len() == n_markers && prior.path2.len() == n_markers
                                {
                                    let marker_pairs = n_markers.saturating_sub(1).max(1);
                                    let mut switches1 = 0usize;
                                    let mut switches2 = 0usize;
                                    for m in 1..n_markers {
                                        if prior.path1[m] != prior.path1[m - 1] {
                                            switches1 += 1;
                                        }
                                        if prior.path2[m] != prior.path2[m - 1] {
                                            switches2 += 1;
                                        }
                                    }
                                    let switch_rate =
                                        (switches1 + switches2) as f32 / (2 * marker_pairs) as f32;
                                    let mean_conf = if sample_conf.is_empty() {
                                        1.0
                                    } else {
                                        sample_conf.iter().copied().sum::<f32>()
                                            / sample_conf.len() as f32
                                    };
                                    let unique_states = forced_counts.len();
                                    can_reuse_prior_subset = unique_states > 0
                                        && unique_states <= final_states
                                        && switch_rate <= 0.12
                                        && mean_conf >= 0.90;
                                }
                            }

                            let mut selected: Vec<usize> = Vec::new();
                            let mut selected_set: HashSet<usize> = HashSet::new();
                            let cap = final_states.max(1);
                            for &state in forced.iter().take(cap) {
                                if state < n_states_full {
                                    if selected_set.insert(state) {
                                        selected.push(state);
                                    }
                                }
                            }
                            if !can_reuse_prior_subset {
                                let mut local_params = self.params.clone();
                                local_params.p_mismatch = sample_p_err;
                                let hmm_full = MosaicHmm::new(
                                    ref_view,
                                    &local_params,
                                    n_states_full,
                                    p_recomb,
                                );
                                let mut fwd1 = Vec::new();
                                let mut bwd1 = Vec::new();
                                let mut fwd2 = Vec::new();
                                let mut bwd2 = Vec::new();

                                let plp = PlProvider {
                                    gt: target_gt,
                                    sample: s,
                                    subset_to_orig: None,
                                };
                                hmm_full.conditioned_forward_backward(
                                    &seq1,
                                    &seq2,
                                    &seq2,
                                    Some(sample_conf),
                                    Some(&plp),
                                    None,
                                    None,
                                    threaded_haps_full,
                                    &mut fwd1,
                                    &mut bwd1,
                                );
                                hmm_full.conditioned_forward_backward(
                                    &seq2,
                                    &seq1,
                                    &seq1,
                                    Some(sample_conf),
                                    Some(&plp),
                                    None,
                                    None,
                                    threaded_haps_full,
                                    &mut fwd2,
                                    &mut bwd2,
                                );

                                let sparse_markers =
                                    select_reduction_sparse_markers(&seq1, &seq2, sample_conf);
                                let top_selected = select_top_k_by_mass_two_sparse_fb(
                                    &fwd1,
                                    &bwd1,
                                    &fwd2,
                                    &bwd2,
                                    n_states_full,
                                    &sparse_markers,
                                    final_states,
                                );
                                for &state in &top_selected {
                                    if selected.len() >= cap {
                                        break;
                                    }
                                    if selected_set.insert(state) {
                                        selected.push(state);
                                    }
                                }
                                if selected.is_empty() {
                                    selected = top_selected;
                                }
                            }
                            if selected.is_empty() {
                                for idx in 0..cap.min(n_states_full) {
                                    selected.push(idx);
                                }
                            }

                            let subset = threaded_haps_full.subset_states(&selected);
                            n_states = subset.n_states();
                            threaded_haps = Cow::Owned(subset);
                            selection_applied = true;
                        }

                        let local_estimates: Option<crate::model::parameters::ParamEstimates> =
                            if atomic_estimates.is_some() || collect_cohort_stats {
                                let mut local_params = self.params.clone();
                                local_params.p_mismatch = sample_p_err;
                                let hmm =
                                    MosaicHmm::new(ref_view, &local_params, n_states, p_recomb);
                                let mut local_est = crate::model::parameters::ParamEstimates::new();
                                hmm.collect_stats(&seq1, &threaded_haps, gen_dists, &mut local_est);
                                hmm.collect_stats(&seq2, &threaded_haps, gen_dists, &mut local_est);
                                if let Some(atomic) = atomic_estimates {
                                    atomic.add_estimation_data(&local_est);
                                }
                                Some(local_est)
                            } else {
                                None
                            };

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
                                a1 != 255
                                    && a2 != 255
                                    && a1 != a2
                                    && sample_phase_view[s].is_unphased(m)
                            })
                            .collect();

                        let (swap_bits, swap_lr, swap_probs, swap_probs_conf, new_paths) =
                            THREAD_WORKSPACE.with(|ws| {
                                let mut workspace = ws.borrow_mut();
                                if workspace.is_none() {
                                    *workspace =
                                        Some(crate::utils::workspace::ThreadWorkspace::new(64, 0));
                                }
                                let ws = workspace.as_mut().unwrap();
                                ws.clear(); // Explicit reset between samples to prevent state contamination
                                let ref_provider = RefAlleleProvider::new(ref_view, &threaded_haps);
                                let (anchor_h1, anchor_h2) =
                                    build_anchor_constraints(&sample_phase_view[s]);

                                let donor_blocks = partition_markers_by_cm(
                                    &gen_positions,
                                    stage1_block_cm(&gen_positions),
                                );
                                let block_starts: Arc<[usize]> =
                                    blocks_to_starts(&donor_blocks, n_markers)
                                        .into_boxed_slice()
                                        .into();
                                let result = sample_swap_bits_mosaic(
                                    n_markers,
                                    n_states,
                                    p_recomb,
                                    &seq1,
                                    &seq2,
                                    &sample_conf,
                                    ref_provider,
                                    Some(PlProvider {
                                        gt: target_gt,
                                        sample: s,
                                        subset_to_orig: None,
                                    }),
                                    block_starts,
                                    &het_positions,
                                    if selection_applied {
                                        None
                                    } else {
                                        local_prior.as_ref()
                                    },
                                    Some(&anchor_h1),
                                    Some(&anchor_h2),
                                    sample_seed,
                                    self.config.mcmc_burnin,
                                    self.config.mcmc_lr_samples,
                                    sample_p_no_err,
                                    sample_p_err,
                                    ws,
                                );
                                result
                            });
                        if new_paths.path1.is_empty() {
                            *paths_out = None;
                        } else {
                            *paths_out =
                                Some(local_to_global_paths(&new_paths, &threaded_haps, n_markers));
                        }
                        *het_lr_out = het_positions
                            .iter()
                            .copied()
                            .zip(swap_lr.iter().copied())
                            .collect();
                        *het_phase_out = het_positions
                            .iter()
                            .enumerate()
                            .map(|(idx, &m)| {
                                let p_swap = swap_probs_conf.get(idx).copied().unwrap_or(0.5);
                                let swap_bit = swap_bits.get(idx).copied().unwrap_or(0);
                                let p_orient = if swap_bit == 1 { p_swap } else { 1.0 - p_swap };
                                (m, p_orient)
                            })
                            .collect();
                        let swap_probs_mask_sum: f32 = swap_probs.iter().sum();
                        assert!(swap_probs_mask_sum.is_finite());
                        assert!(swap_lr.len() <= n_markers);
                        assert!(het_phase_out.len() <= het_positions.len());
                        if collect_cohort_stats {
                            let (mismatch_mass, emission_mass, expected_switches, genetic_dist_cm) =
                                if let Some(est) = local_estimates.as_ref() {
                                    (
                                        est.mismatch_mass(),
                                        est.emission_mass(),
                                        est.expected_switches(),
                                        est.genetic_distance_cm(),
                                    )
                                } else {
                                    (0.0, 0.0, 0.0, 0.0)
                                };
                            let mut uncertainty_sum = 0.0f64;
                            let mut uncertainty_count = 0usize;
                            for &m in &het_positions {
                                uncertainty_sum += (1.0
                                    - sample_phase_view[s].phase_confidence(m).clamp(0.0, 1.0))
                                    as f64;
                                uncertainty_count += 1;
                            }
                            *cohort_stats_out_one = Some(SampleCohortStats {
                                mismatch_mass,
                                emission_mass,
                                expected_switches,
                                genetic_dist_morgans: genetic_dist_cm / 100.0,
                                phase_uncertainty_sum: uncertainty_sum,
                                phase_uncertainty_count: uncertainty_count,
                            });
                        } else {
                            *cohort_stats_out_one = None;
                        }
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
                    },
                )
            });

            swap_results
        }); // ref_geno borrow ends here

        // Apply Swaps
        // After computing swap masks for all samples, apply them sequentially.
        // This is done sequentially because swap_haplotypes requires mutable access.
        info_span!("apply_swaps").in_scope(|| {
            let mut cohort_stats_out = cohort_stats_out;
            for (s, (mask, het_lr_values, het_phase_values, paths, cohort_stats_one)) in
                swap_results.into_iter().enumerate()
            {
                if let Some(out) = cohort_stats_out.as_deref_mut() {
                    if s < out.len() {
                        out[s] = cohort_stats_one.unwrap_or_default();
                    }
                }
                let sample_idx = SampleIdx::new(s as u32);
                let hap1 = sample_idx.hap1();
                let hap2 = sample_idx.hap2();
                geno.swap_haplotypes(hap1, hap2, &mask);

                if s < sample_phases.len() {
                    let sp = &mut sample_phases[s];

                    for m in mask.iter_ones() {
                        sp.swap_alleles(m);
                    }

                    for (m, p_orient) in het_phase_values {
                        sp.set_phase_confidence(m, p_orient);
                    }

                    let lr_threshold = self.params.lr_threshold;
                    for (m, lr) in het_lr_values {
                        if lr >= lr_threshold {
                            sp.mark_phased(m);
                        }
                    }
                }
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
        target_gt: &GenotypeMatrix,
        geno: &mut MutableGenotypes,
        threaded_haps_vec: &[crate::model::states::ThreadedHaps<CombinedHapSpace>],
        stage1_p_recomb: &[f32],
        stage1_gen_dists: &[f64],
        hi_freq_to_orig: &[usize],
        stage1_blocks: &[(usize, usize)],
        ibs2: &Ibs2,
        sample_phases: &mut [SamplePhase],
        mcmc_paths: &mut [Option<GlobalMosaicPaths>],
        frozen_samples: Option<&[bool]>,
        sample_p_mismatch: Option<&[f32]>,
        cohort_stats_out: Option<&mut [SampleCohortStats]>,
        atomic_estimates: Option<&crate::model::parameters::AtomicParamEstimates>,
        iteration: usize,
    ) -> Result<(usize, usize, Vec<bool>)> {
        let n_stage1_blocks = stage1_blocks.len();
        if n_stage1_blocks == 0 {
            return Ok((0, 0, vec![false; sample_phases.len()]));
        }
        let n_haps = geno.n_haps();

        let n_samples = sample_phases.len();
        let n_hi_freq = hi_freq_to_orig.len();
        let sample_has_unresolved_stage1_het: Vec<bool> = sample_phases
            .iter()
            .map(|sp| {
                hi_freq_to_orig.iter().any(|&m| {
                    if !sp.is_unphased(m) {
                        return false;
                    }
                    let a1 = sp.allele1(m);
                    let a2 = sp.allele2(m);
                    a1 != 255 && a2 != 255 && a1 != a2
                })
            })
            .collect();
        let active_sample_indices: Vec<usize> = (0..n_samples)
            .filter(|&s| {
                let frozen = frozen_samples
                    .and_then(|frozen| frozen.get(s))
                    .copied()
                    .unwrap_or(false);
                !frozen && sample_has_unresolved_stage1_het[s]
            })
            .collect();
        if active_sample_indices.is_empty() {
            return Ok((0, 0, vec![false; n_samples]));
        }
        let timing = Arc::new(Stage1Timing::default());
        let timing_start = Instant::now();
        const LOG_EVERY_NS: u64 = 60_000_000_000;
        if let Some(bb) = &self.telemetry {
            bb.set_op("Stage 1: sampling mosaics");
        }

        // No clone needed: the HMM phase is read-only; mutations happen after.
        // We use a scoped immutable borrow that ends before the apply phase.
        #[derive(Clone, Copy)]
        struct HiFreqMarkerIdx(usize);

        #[derive(Clone, Copy)]
        struct RelativeSwapBit(bool);

        #[derive(Clone, Copy)]
        struct AbsoluteHap1Allele(u8);

        #[derive(Clone, Copy)]
        struct PhaseLogOdds(f32);

        #[derive(Clone, Copy)]
        struct PhaseConfidence(f32);

        enum Stage1OrientationUpdate {
            NoChange,
            RelativeSwapMask(Vec<RelativeSwapBit>),
            AbsoluteHap1(Vec<(HiFreqMarkerIdx, AbsoluteHap1Allele)>),
        }

        struct Stage1HetUpdate {
            marker: HiFreqMarkerIdx,
            lr: PhaseLogOdds,
            confidence: PhaseConfidence,
        }

        struct Stage1PhaseDecision {
            orientation: Stage1OrientationUpdate,
            het_updates: Vec<Stage1HetUpdate>,
            paths: Option<GlobalMosaicPaths>,
            cohort_stats: Option<SampleCohortStats>,
        }
        let phase_decisions: Vec<Stage1PhaseDecision> = {
            // Immutable borrow of geno for the entire read phase
            let ref_geno: &MutableGenotypes = geno;

            // 1. Create Subset View for Stage 1 markers
            // Use CompositeSubset when reference panel is available
            let subset_view =
                if let (Some(ref_gt), Some(alignment)) = (&self.reference_gt, &self.alignment) {
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
            let use_dynamic_mcmc = self.config.dynamic_mcmc;
            eprintln!(
                "[stage1 mode] dynamic_mcmc={} n_hi_freq={} samples={}",
                use_dynamic_mcmc, n_hi_freq, n_samples
            );
            let phase_ibs = if use_dynamic_mcmc {
                if let (Some(ref_gt), Some(alignment)) = (&self.reference_gt, &self.alignment) {
                    Some(self.build_bidirectional_pbwt_subset_with_ref(
                        ref_geno,
                        ref_gt.as_ref(),
                        alignment,
                        hi_freq_to_orig,
                    ))
                } else {
                    Some(self.build_bidirectional_pbwt_subset(ref_geno, hi_freq_to_orig, n_haps))
                }
            } else {
                None
            };

            // Collect typed phase decisions per sample. Orientation updates are explicit:
            // either relative swaps (static MCMC) or absolute hap1 alignment (dynamic MCMC).
            let prior_paths = &mcmc_paths[..];
            let telemetry = self.telemetry.clone();
            let block_starts: Arc<[usize]> = if use_dynamic_mcmc {
                Arc::from([])
            } else {
                blocks_to_starts(stage1_blocks, n_hi_freq)
                    .into_boxed_slice()
                    .into()
            };
            if let Some(bb) = telemetry.as_ref() {
                let k = if self.config.dynamic_mcmc {
                    self.config.dynamic_k.max(1)
                } else {
                    0
                };
                bb.set_dynamic_mcmc(self.config.dynamic_mcmc, k);
                bb.reset_dyn_neighbors();
            }
            let sample_phase_stability: Vec<f32> = sample_phases
                .iter()
                .map(|sp| stage1_sample_phase_stability(sp, hi_freq_to_orig))
                .collect();
            let emission_conf_scales = Arc::new(build_marker_emission_conf_scales(
                target_gt,
                sample_phases,
                hi_freq_to_orig,
                self.params.p_mismatch,
            ));
            if !emission_conf_scales.is_empty() {
                let mut sum_scale = 0.0f32;
                let mut min_scale = 1.0f32;
                for &scale in emission_conf_scales.iter() {
                    sum_scale += scale;
                    min_scale = min_scale.min(scale);
                }
                let mean_scale = sum_scale / emission_conf_scales.len() as f32;
                eprintln!(
                    "[stage1 emission] markers={} mean_conf_scale={:.4} min_conf_scale={:.4}",
                    emission_conf_scales.len(),
                    mean_scale,
                    min_scale
                );
            }
            let collect_cohort_stats = cohort_stats_out.is_some();
            let sample_iter = || {
                active_sample_indices.par_iter().map(|&s| {
                    let sp = &sample_phases[s];
                    THREAD_WORKSPACE.with(|ws| {
                        let mut workspace = ws.borrow_mut();
                        if workspace.is_none() {
                            *workspace =
                                Some(crate::utils::workspace::ThreadWorkspace::new(64, 0));
                        }
                        let ws = workspace.as_mut().unwrap();
                        ws.clear();

                        let n_hi_freq = hi_freq_to_orig.len();
                        let mut threaded_haps = Cow::Borrowed(&threaded_haps_vec[s]);

                        let t0 = Instant::now();
                        // Extract alleles/confidence for SUBSET of markers using reused buffers
                        ws.seq1.clear();
                        ws.seq2.clear();
                        ws.sample_conf.clear();
                        ws.sample_phase_conf.clear();
                        ws.seq1.reserve(n_hi_freq);
                        ws.seq2.reserve(n_hi_freq);
                        ws.sample_conf.reserve(n_hi_freq);
                        ws.sample_phase_conf.reserve(n_hi_freq);
                        for (i, &m) in hi_freq_to_orig.iter().enumerate() {
                            ws.seq1.push(sp.allele1(m));
                            ws.seq2.push(sp.allele2(m));
                            let conf_scale = emission_conf_scales[i];
                            ws.sample_conf
                                .push((sp.confidence(m) * conf_scale).clamp(0.0, 1.0));
                            ws.sample_phase_conf.push(sp.phase_confidence(m));
                        }
                        let seq1 = std::mem::take(&mut ws.seq1);
                        let seq2 = std::mem::take(&mut ws.seq2);
                        let sample_conf = std::mem::take(&mut ws.sample_conf);
                        let sample_phase_conf = std::mem::take(&mut ws.sample_phase_conf);
                        let t_seq = t0.elapsed();

                        let sample_seed = (self.config.seed as u64)
                            .wrapping_add(s as u64)
                            .wrapping_add((iteration as u64) << 32)
                            .wrapping_add(0xFEED_FACE_1234u64);
                        let sample_p_err = MismatchProb::new_clamped(
                            sample_p_mismatch
                                .and_then(|v| v.get(s))
                                .copied()
                                .unwrap_or(self.params.p_mismatch),
                        )
                        .get();
                        let sample_p_no_err = 1.0 - sample_p_err;

                        // Identify unresolved heterozygotes in hi-freq space.
                        let mut het_positions: Vec<usize> = Vec::new();
                        for i in 0..n_hi_freq {
                            let m = hi_freq_to_orig[i];
                            if !sp.is_unphased(m) {
                                continue;
                            }
                            let a1 = seq1[i];
                            let a2 = seq2[i];
                            if a1 != 255 && a2 != 255 && a1 != a2 {
                                het_positions.push(i);
                            }
                        }

                        if het_positions.is_empty() {
                            // No unresolved hets to phase for this sample.
                            ws.seq1 = seq1;
                            ws.seq2 = seq2;
                            ws.sample_conf = sample_conf;
                            ws.sample_phase_conf = sample_phase_conf;
                            ws.het_positions = het_positions;
                            return Stage1PhaseDecision {
                                orientation: Stage1OrientationUpdate::NoChange,
                                het_updates: Vec::new(),
                                paths: None,
                                cohort_stats: None,
                            };
                        }

                        let has_phase_anchors = sp.has_input_phase_anchor();

                        let t_anchor_start = Instant::now();
                        if self.reference_gt.is_some() && has_phase_anchors {
                            let threaded_haps = threaded_haps.to_mut();
                            let direct_ref = self
                                .reference_gt
                                .as_ref()
                                .zip(self.alignment.as_ref())
                                .map(|(ref_gt, alignment)| (ref_gt.as_ref(), alignment));
                            let mut anchors: Vec<(usize, u8, u8)> = Vec::new();
                            for (i, &m) in hi_freq_to_orig.iter().enumerate() {
                                if !sp.is_input_phased_het(m) {
                                    continue;
                                }
                                let a1 = sp.allele1(m);
                                let a2 = sp.allele2(m);
                                if a1 == 255 || a2 == 255 || a1 == a2 {
                                    continue;
                                }
                                anchors.push((i, a1, a2));
                            }
                            if !anchors.is_empty() {
                                let n_ref_haps = self
                                    .reference_gt
                                    .as_ref()
                                    .map(|r| r.n_haplotypes())
                                    .unwrap_or(0);
                                let offset = n_haps;
                                let mut best_hap1 = (0usize, f32::NEG_INFINITY);
                                let mut best_hap2 = (0usize, f32::NEG_INFINITY);
                                for h in 0..n_ref_haps {
                                    let hap_idx = HapIdx::new((offset + h) as u32);
                                    let mut score1 = 0.0f32;
                                    let mut score2 = 0.0f32;
                                    for &(i, a1, a2) in &anchors {
                                        let ref_al = if let Some((ref_gt, alignment)) = direct_ref {
                                            let orig_marker = hi_freq_to_orig[i];
                                            if let Some(ref_m) =
                                                alignment.target_to_ref(MarkerIdx::new(orig_marker as u32))
                                            {
                                                let ra = ref_gt.allele(ref_m, HapIdx::new(h as u32));
                                                alignment.reverse_map_allele(orig_marker, ra)
                                            } else {
                                                255
                                            }
                                        } else {
                                            subset_view.allele(MarkerIdx::new(i as u32), hap_idx)
                                        };
                                        let conf_m = sample_conf[i].clamp(0.0, 1.0);
                                        score1 += emit_prob(
                                            ref_al,
                                            a1,
                                            conf_m,
                                            sample_p_no_err,
                                            sample_p_err,
                                        )
                                            .max(1e-30)
                                            .ln();
                                        score2 += emit_prob(
                                            ref_al,
                                            a2,
                                            conf_m,
                                            sample_p_no_err,
                                            sample_p_err,
                                        )
                                            .max(1e-30)
                                            .ln();
                                    }
                                    if score1 > best_hap1.1 {
                                        best_hap1 = (h, score1);
                                    }
                                    if score2 > best_hap2.1 {
                                        best_hap2 = (h, score2);
                                    }
                                }
                                if best_hap1.1.is_finite() || best_hap2.1.is_finite() {
                                    let mut existing = vec![CombinedHapId::from(0u32); threaded_haps.n_states()];
                                    threaded_haps.materialize_at(0, &mut existing);
                                    let has_hap = |hap: u32| existing.iter().any(|g| g.as_u32() == hap);
                                    let hap1_id = (offset + best_hap1.0) as u32;
                                    if best_hap1.1.is_finite() && !has_hap(hap1_id) {
                                        threaded_haps.push_new(CombinedHapId::new(hap1_id));
                                    }
                                    let hap2_id = (offset + best_hap2.0) as u32;
                                    if best_hap2.1.is_finite() && !has_hap(hap2_id) {
                                        threaded_haps.push_new(CombinedHapId::new(hap2_id));
                                    }
                                }

                                let max_scan = 10_000_000usize;
                                if n_ref_haps.saturating_mul(n_hi_freq) <= max_scan && n_ref_haps > 0 {
                                    let mut scores: Vec<f32> = vec![0.0; n_ref_haps];
                                    for h in 0..n_ref_haps {
                                        let hap_idx = HapIdx::new((offset + h) as u32);
                                        let mut score = 0.0f32;
                                        for i in 0..n_hi_freq {
                                            let a1 = seq1[i];
                                            let a2 = seq2[i];
                                            if a1 == 255 && a2 == 255 {
                                                continue;
                                            }
                                            let conf_m = sample_conf[i].clamp(0.0, 1.0);
                                            let ref_al = if let Some((ref_gt, alignment)) = direct_ref {
                                                let orig_marker = hi_freq_to_orig[i];
                                                if let Some(ref_m) =
                                                    alignment.target_to_ref(MarkerIdx::new(orig_marker as u32))
                                                {
                                                    let ra =
                                                        ref_gt.allele(ref_m, HapIdx::new(h as u32));
                                                    alignment.reverse_map_allele(orig_marker, ra)
                                                } else {
                                                    255
                                                }
                                            } else {
                                                subset_view.allele(MarkerIdx::new(i as u32), hap_idx)
                                            };
                                            let emit = if a1 == a2 {
                                                emit_prob(
                                                    ref_al,
                                                    a1,
                                                    conf_m,
                                                    sample_p_no_err,
                                                    sample_p_err,
                                                )
                                            } else {
                                                let keep = emit_prob(
                                                    ref_al,
                                                    a1,
                                                    conf_m,
                                                    sample_p_no_err,
                                                    sample_p_err,
                                                );
                                                let swap = emit_prob(
                                                    ref_al,
                                                    a2,
                                                    conf_m,
                                                    sample_p_no_err,
                                                    sample_p_err,
                                                );
                                                0.5 * (keep + swap)
                                            };
                                            score += emit.max(1e-30).ln();
                                        }
                                        scores[h] = score;
                                    }
                                    let mut idxs: Vec<usize> = (0..n_ref_haps).collect();
                                    idxs.sort_by(|&a, &b| {
                                        scores[b]
                                            .partial_cmp(&scores[a])
                                            .unwrap_or(std::cmp::Ordering::Equal)
                                    });
                                    let take = PBWT_FORCE_TOP_HAPS.min(idxs.len());
                                    if take > 0 {
                                        let mut existing =
                                            vec![CombinedHapId::from(0u32); threaded_haps.n_states()];
                                        threaded_haps.materialize_at(0, &mut existing);
                                        let has_hap =
                                            |hap: u32| existing.iter().any(|g| g.as_u32() == hap);
                                        for &h in idxs.iter().take(take) {
                                            let hap_id = (offset + h) as u32;
                                            if !has_hap(hap_id) {
                                                threaded_haps.push_new(CombinedHapId::new(hap_id));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        let t_anchor = t_anchor_start.elapsed();

                        let n_states = threaded_haps.as_ref().n_states();
                        if s == 0 && n_hi_freq <= 600 {
                            let mut state_ids = vec![CombinedHapId::from(0u32); n_states];
                            threaded_haps.as_ref().materialize_at(0, &mut state_ids);
                            let has_200 = state_ids.iter().any(|id| id.as_u32() == 200);
                            let has_201 = state_ids.iter().any(|id| id.as_u32() == 201);
                            eprintln!(
                                "[stage1 states] sample=0 n_states={} has_combined200={} has_combined201={}",
                                n_states, has_200, has_201
                            );
                        }

                        // Collect EM statistics if requested
                        let local_estimates: Option<crate::model::parameters::ParamEstimates> =
                            if atomic_estimates.is_some() || collect_cohort_stats {
                            let mut local_params = self.params.clone();
                            local_params.p_mismatch = sample_p_err;
                            let hmm = MosaicHmm::new(
                                subset_view,
                                &local_params,
                                n_states,
                                stage1_p_recomb,
                            );
                            let mut local_est = crate::model::parameters::ParamEstimates::new();
                            hmm.collect_stats(
                                &seq1,
                                threaded_haps.as_ref(),
                                stage1_gen_dists,
                                &mut local_est,
                            );
                            hmm.collect_stats(
                                &seq2,
                                threaded_haps.as_ref(),
                                stage1_gen_dists,
                                &mut local_est,
                            );
                            if let Some(atomic) = atomic_estimates {
                                atomic.add_estimation_data(&local_est);
                            }
                            Some(local_est)
                        } else {
                            None
                        };

                        let t_mcmc_start = Instant::now();
                        let (swap_bits, swap_lr, swap_probs, swap_probs_conf, new_paths) = if use_dynamic_mcmc {
                            let dyn_k_max = self.config.dynamic_k.max(1).min(n_states.max(1));
                            let dyn_k_min = dyn_k_max.min(PBWT_ADAPTIVE_K_FLOOR).max(1);
                            let sample_uncertainty =
                                1.0f32 - sample_phase_stability[s].clamp(0.0, 1.0);
                            let dyn_k = if dyn_k_max > dyn_k_min {
                                let span = (dyn_k_max - dyn_k_min) as f32;
                                dyn_k_min
                                    + (sample_uncertainty * span).round() as usize
                            } else {
                                dyn_k_max
                            };
                            let dyn_steps_max = self.config.mcmc_steps.max(1);
                            let dyn_steps_min = if sample_phase_stability[s] >= 0.995 {
                                1
                            } else {
                                dyn_steps_max.min(2).max(1)
                            };
                            let dyn_steps = if dyn_steps_max > dyn_steps_min {
                                let step_span = (dyn_steps_max - dyn_steps_min) as f32;
                                dyn_steps_min
                                    + (sample_uncertainty * step_span).round() as usize
                            } else {
                                dyn_steps_max
                            };
                            // SHAPEIT5-style dynamic MCMC: re-select states each step
                            let mut prior_local = prior_paths[s].as_ref().map(|gp| MosaicPaths {
                                path1: gp.path1.iter().map(|id| id.as_u32()).collect(),
                                path2: gp.path2.iter().map(|id| id.as_u32()).collect(),
                            });
                            if prior_local.is_none() {
                                let mut rp = RefAlleleProvider::new(subset_view, threaded_haps.as_ref());
                                if let Some(local_best) = find_best_constant_pair_with_buffer(
                                    n_hi_freq,
                                    n_states,
                                    &seq1,
                                    &seq2,
                                    &sample_conf,
                                    sample_p_no_err,
                                    sample_p_err,
                                    &mut rp,
                                    None,
                                    &mut ws.scores,
                                    None,
                                ) {
                                    let global_best =
                                        local_to_global_paths(&local_best, threaded_haps.as_ref(), n_hi_freq);
                                    prior_local = Some(MosaicPaths {
                                        path1: global_best
                                            .path1
                                            .iter()
                                            .map(|id: &CombinedHapId| id.as_u32())
                                            .collect(),
                                        path2: global_best
                                            .path2
                                            .iter()
                                            .map(|id: &CombinedHapId| id.as_u32())
                                            .collect(),
                                    });
                                    if s == 0 && n_hi_freq <= 600 {
                                        let p1 = prior_local
                                            .as_ref()
                                            .and_then(|p| p.path1.first().copied())
                                            .unwrap_or(0);
                                        let p2 = prior_local
                                            .as_ref()
                                            .and_then(|p| p.path2.first().copied())
                                            .unwrap_or(0);
                                        eprintln!(
                                            "[dynamic init] sample=0 best_constant_pair_first=({}, {})",
                                            p1, p2
                                        );
                                    }
                                }
                            }
                            // Do not inject per-marker phase anchors into dynamic MCMC.
                            // In unanchored/symmetric regimes this can create circular
                            // self-conditioning against the current phase assignment.
                            if s == 0 && n_hi_freq <= 600 {
                                let p1 = prior_local
                                    .as_ref()
                                    .and_then(|p| p.path1.first().copied())
                                    .unwrap_or(0);
                                let p2 = prior_local
                                    .as_ref()
                                    .and_then(|p| p.path2.first().copied())
                                    .unwrap_or(0);
                                eprintln!(
                                    "[dynamic prior] sample=0 have_prior={} first_pair=({}, {})",
                                    prior_local.is_some(),
                                    p1,
                                    p2
                                );
                            }

                            let (swap_bits, swap_lr, swap_probs, swap_probs_conf, new_paths) = if self.config.profile {
                                info_span!("run_dynamic_mcmc", sample = s).in_scope(|| {
                                        sample_dynamic_mcmc(
                                        n_hi_freq,
                                        dyn_k,
                                        stage1_p_recomb,
                                        &seq1,
                                        &seq2,
                                        &sample_conf,
                                        &sample_phase_conf,
                                        phase_ibs.as_ref().expect("phase_ibs"),
                                        ibs2,
                                        hi_freq_to_orig,
                                        s as u32,
                                        &sample_phase_stability,
                                        &het_positions,
                                        sample_seed,
                                        dyn_steps,
                                        sample_p_no_err,
                                        sample_p_err,
                                        prior_local.as_ref(),
                                        None,
                                        None,
                                        telemetry.as_deref(),
                                        ws,
                                    )
                                })
                            } else {
                                sample_dynamic_mcmc(
                                    n_hi_freq,
                                    dyn_k,
                                    stage1_p_recomb,
                                    &seq1,
                                    &seq2,
                                    &sample_conf,
                                    &sample_phase_conf,
                                    phase_ibs.as_ref().expect("phase_ibs"),
                                    ibs2,
                                    hi_freq_to_orig,
                                    s as u32,
                                    &sample_phase_stability,
                                    &het_positions,
                                    sample_seed,
                                    dyn_steps,
                                    sample_p_no_err,
                                    sample_p_err,
                                    prior_local.as_ref(),
                                    None,
                                    None,
                                    telemetry.as_deref(),
                                    ws,
                                )
                            };
                            let global_paths = GlobalMosaicPaths {
                                path1: new_paths.path1.into_iter().map(CombinedHapId::from).collect(),
                                path2: new_paths.path2.into_iter().map(CombinedHapId::from).collect(),
                            };
                            (swap_bits, swap_lr, swap_probs, swap_probs_conf, Some(global_paths))
                        } else {
                            // Classic Beagle-style: static state space MCMC with thread-local workspace
                            let ref_provider = if self.config.profile {
                                info_span!("prep_allele_provider", sample = s).in_scope(|| {
                                    RefAlleleProvider::new(subset_view, threaded_haps.as_ref())
                                })
                            } else {
                                RefAlleleProvider::new(subset_view, threaded_haps.as_ref())
                            };

                            let local_prior_raw = prior_paths[s]
                                .as_ref()
                                .and_then(|gp| {
                                    global_to_local_paths(gp, threaded_haps.as_ref(), n_hi_freq)
                                });
                            let (anchor_h1_full, anchor_h2_full) = build_anchor_constraints(sp);
                            let has_anchors = anchor_h1_full.iter().any(|&a| a != 255)
                                || anchor_h2_full.iter().any(|&a| a != 255);
                            let local_prior = if has_anchors {
                                None
                            } else {
                                local_prior_raw.as_ref()
                            };
                            let mut anchor_h1 = Vec::with_capacity(n_hi_freq);
                            let mut anchor_h2 = Vec::with_capacity(n_hi_freq);
                            for &m in hi_freq_to_orig {
                                anchor_h1.push(anchor_h1_full[m]);
                                anchor_h2.push(anchor_h2_full[m]);
                            }

                            let block_starts = block_starts.clone();
                            let result = if self.config.profile {
                                info_span!("run_mcmc_math", sample = s).in_scope(|| {
                                    sample_swap_bits_mosaic(
                                        n_hi_freq,
                                        n_states,
                                        stage1_p_recomb,
                                        &seq1,
                                        &seq2,
                                        &sample_conf,
                                        ref_provider,
                                        Some(PlProvider {
                                            gt: target_gt,
                                            sample: s,
                                            subset_to_orig: Some(hi_freq_to_orig),
                                        }),
                                        block_starts,
                                        &het_positions,
                                        local_prior,
                                        Some(&anchor_h1),
                                        Some(&anchor_h2),
                                        sample_seed,
                                        self.config.mcmc_burnin,
                                        self.config.mcmc_lr_samples,
                                        sample_p_no_err,
                                        sample_p_err,
                                        ws,
                                    )
                                })
                            } else {
                                sample_swap_bits_mosaic(
                                    n_hi_freq,
                                    n_states,
                                    stage1_p_recomb,
                                    &seq1,
                                    &seq2,
                                    &sample_conf,
                                    ref_provider,
                                    Some(PlProvider {
                                        gt: target_gt,
                                        sample: s,
                                        subset_to_orig: Some(hi_freq_to_orig),
                                    }),
                                    block_starts,
                                    &het_positions,
                                    local_prior,
                                    Some(&anchor_h1),
                                    Some(&anchor_h2),
                                    sample_seed,
                                    self.config.mcmc_burnin,
                                    self.config.mcmc_lr_samples,
                                    sample_p_no_err,
                                    sample_p_err,
                                    ws,
                                )
                            };
                            let global_paths =
                                local_to_global_paths(&result.4, threaded_haps.as_ref(), n_hi_freq);
                            (result.0, result.1, result.2, result.3, Some(global_paths))
                        };

                        let t_mcmc = t_mcmc_start.elapsed();
                        let swap_probs_sum: f32 = swap_probs.iter().sum();
                        assert!(swap_probs_sum.is_finite());
                        let orientation = if use_dynamic_mcmc {
                            if has_phase_anchors {
                                let mut desired_hap1: Vec<(HiFreqMarkerIdx, AbsoluteHap1Allele)> =
                                    Vec::with_capacity(het_positions.len());
                                if let Some(paths) = new_paths.as_ref() {
                                    for &idx in &het_positions {
                                        let a1 = seq1[idx];
                                        let a2 = seq2[idx];
                                        if a1 == 255 || a2 == 255 || a1 == a2 {
                                            continue;
                                        }
                                        let h1 = paths
                                            .path1
                                            .get(idx)
                                            .copied()
                                            .map(|h| h.as_u32())
                                            .unwrap_or(u32::MAX);
                                        let h2 = paths
                                            .path2
                                            .get(idx)
                                            .copied()
                                            .map(|h| h.as_u32())
                                            .unwrap_or(u32::MAX);
                                        let desired = if h1 == u32::MAX || h2 == u32::MAX {
                                            // Preserve current orientation if path is unavailable at this marker.
                                            a1
                                        } else {
                                            let marker_idx = MarkerIdx::new(idx as u32);
                                            let haps = [HapIdx::new(h1), HapIdx::new(h2)];
                                            let mut row = [255u8; 2];
                                            subset_view.fill_batch(marker_idx, &haps, &mut row);
                                            let r1 = row[0];
                                            let r2 = row[1];
                                            if r1 == a1 && r2 == a2 {
                                                a1
                                            } else if r1 == a2 && r2 == a1 {
                                                a2
                                            } else if r1 == a1 || r1 == a2 {
                                                r1
                                            } else if r2 == a1 {
                                                a2
                                            } else if r2 == a2 {
                                                a1
                                            } else {
                                                a1
                                            }
                                        };
                                        desired_hap1
                                            .push((HiFreqMarkerIdx(idx), AbsoluteHap1Allele(desired)));
                                    }
                                }
                                Stage1OrientationUpdate::AbsoluteHap1(desired_hap1)
                            } else {
                                let mut swap_mask = vec![RelativeSwapBit(false); n_hi_freq];
                                let lr_threshold = self.params.lr_threshold;
                                for (idx, &pos) in het_positions.iter().enumerate() {
                                    let swap_bit = swap_bits.get(idx).copied().unwrap_or(0);
                                    let lr = *swap_lr.get(idx).unwrap_or(&1.0);
                                    if lr >= lr_threshold && swap_bit == 1 {
                                        swap_mask[pos] = RelativeSwapBit(true);
                                    }
                                }
                                Stage1OrientationUpdate::RelativeSwapMask(swap_mask)
                            }
                        } else {
                            let mut swap_mask = vec![RelativeSwapBit(false); n_hi_freq];
                            for (idx, &pos) in het_positions.iter().enumerate() {
                                let swap_bit = swap_bits.get(idx).copied().unwrap_or(0);
                                swap_mask[pos] = RelativeSwapBit(swap_bit == 1);
                            }
                            Stage1OrientationUpdate::RelativeSwapMask(swap_mask)
                        };
                        let mut anchor_resets = 0usize;
                        for i in 0..n_hi_freq {
                            let m = hi_freq_to_orig[i];
                            let a1 = seq1[i];
                            let a2 = seq2[i];
                            let is_het = a1 != 255 && a2 != 255 && a1 != a2;
                            if is_het && !sp.is_unphased(m) {
                                anchor_resets += 1;
                            }
                        }

                        // Phase confidence should reflect absolute label certainty.
                        // If there are no anchored/phased markers yet, labels are symmetric,
                        // so confidence should remain ~0.5 even if a single chain picks a side.
                        let mut het_updates = Vec::with_capacity(het_positions.len());
                        for (map_idx, &idx) in het_positions.iter().enumerate() {
                            let lr = PhaseLogOdds(*swap_lr.get(map_idx).unwrap_or(&1.0));
                            let confidence = {
                                let p_swap = *swap_probs_conf.get(map_idx).unwrap_or(&0.5);
                                let swap_bit = *swap_bits.get(map_idx).unwrap_or(&0);
                                let p_orient = if swap_bit == 1 { p_swap } else { 1.0 - p_swap };
                                PhaseConfidence(p_orient.clamp(0.0, 1.0))
                            };
                            het_updates.push(Stage1HetUpdate {
                                marker: HiFreqMarkerIdx(idx),
                                lr,
                                confidence,
                            });
                        }

                        let cohort_stats = if collect_cohort_stats {
                            let (mismatch_mass, emission_mass, expected_switches, genetic_dist_cm) =
                                if let Some(est) = local_estimates.as_ref() {
                                    (
                                        est.mismatch_mass(),
                                        est.emission_mass(),
                                        est.expected_switches(),
                                        est.genetic_distance_cm(),
                                    )
                                } else {
                                    (0.0, 0.0, 0.0, 0.0)
                                };
                            let mut uncertainty_sum = 0.0f64;
                            let mut uncertainty_count = 0usize;
                            for i in 0..n_hi_freq {
                                let a1 = seq1[i];
                                let a2 = seq2[i];
                                if a1 != 255 && a2 != 255 && a1 != a2 {
                                    uncertainty_sum +=
                                        (1.0 - sample_phase_conf[i].clamp(0.0, 1.0)) as f64;
                                    uncertainty_count += 1;
                                }
                            }
                            Some(SampleCohortStats {
                                mismatch_mass,
                                emission_mass,
                                expected_switches,
                                genetic_dist_morgans: genetic_dist_cm / 100.0,
                                phase_uncertainty_sum: uncertainty_sum,
                                phase_uncertainty_count: uncertainty_count,
                            })
                        } else {
                            None
                        };

                        let het_count = het_positions.len() as u64;
                        ws.seq1 = seq1;
                        ws.seq2 = seq2;
                        ws.sample_conf = sample_conf;
                        ws.sample_phase_conf = sample_phase_conf;
                        ws.het_positions = het_positions;

                        let sample_total = t0.elapsed();
                        timing.add_sample(
                            t_seq.as_nanos() as u64,
                            t_anchor.as_nanos() as u64,
                            t_mcmc.as_nanos() as u64,
                            sample_total.as_nanos() as u64,
                            n_states as u64,
                            het_count,
                            anchor_resets as u64,
                        );
                        let elapsed_ns = timing_start.elapsed().as_nanos() as u64;
                        if elapsed_ns >= LOG_EVERY_NS && timing.should_log(elapsed_ns) {
                            let samples = timing.samples.load(Ordering::Relaxed).max(1);
                            let total_ns =
                                timing.total_sample_ns.load(Ordering::Relaxed).max(1);
                            let mcmc_ns = timing.mcmc_ns.load(Ordering::Relaxed);
                            let seq_ns = timing.seq_extract_ns.load(Ordering::Relaxed);
                            let anchor_ns = timing.anchor_ns.load(Ordering::Relaxed);
                            let avg_total_ms = total_ns as f64 / samples as f64 / 1e6;
                            let avg_mcmc_ms = mcmc_ns as f64 / samples as f64 / 1e6;
                            let avg_seq_ms = seq_ns as f64 / samples as f64 / 1e6;
                            let avg_anchor_ms = anchor_ns as f64 / samples as f64 / 1e6;
                            let avg_states = timing.n_states_sum.load(Ordering::Relaxed) as f64
                                / samples as f64;
                            let avg_hets =
                                timing.hets_sum.load(Ordering::Relaxed) as f64 / samples as f64;
                            let avg_anchors = timing.anchors_sum.load(Ordering::Relaxed) as f64
                                / samples as f64;
                            let elapsed_s = timing_start.elapsed().as_secs_f64().max(1e-9);
                            let samp_per_s = samples as f64 / elapsed_s;
                            eprintln!(
                                "[stage1] samples_done={} avg_total_ms={:.2} avg_mcmc_ms={:.2} avg_seq_ms={:.2} avg_anchor_ms={:.2} avg_states={:.1} avg_hets={:.1} avg_anchors={:.1} rate={:.2}/s",
                                samples,
                                avg_total_ms,
                                avg_mcmc_ms,
                                avg_seq_ms,
                                avg_anchor_ms,
                                avg_states,
                                avg_hets,
                                avg_anchors,
                                samp_per_s
                            );
                        }

                        if let Some(bb) = telemetry.as_ref() {
                            bb.add_samples(1);
                        }

                        Stage1PhaseDecision {
                            orientation,
                            het_updates,
                            paths: new_paths,
                            cohort_stats,
                        }
                    })
                })
            };

            if self.config.profile {
                info_span!("phase_sample_all", samples = n_samples)
                    .in_scope(|| sample_iter().collect())
            } else {
                sample_iter().collect()
            }
        }; // ref_geno borrow ends here

        // Apply phase decisions to SamplePhase
        let mut total_switches = 0;
        let mut total_phased = 0;
        let mut sample_changed = vec![false; n_samples];

        // Determine if we're in burn-in (don't mark as phased during burn-in)
        let is_burnin = iteration < self.config.burnin;
        let lr_threshold = self.params.lr_threshold;
        let mut cohort_stats_out = cohort_stats_out;

        for (s, decision) in phase_decisions.into_iter().enumerate() {
            if let Some(out) = cohort_stats_out.as_deref_mut() {
                if s < out.len() {
                    out[s] = decision.cohort_stats.unwrap_or_default();
                }
            }
            let sp = &mut sample_phases[s];

            match decision.orientation {
                Stage1OrientationUpdate::NoChange => {}
                Stage1OrientationUpdate::AbsoluteHap1(desired_hap1) => {
                    for (HiFreqMarkerIdx(hi_freq_idx), AbsoluteHap1Allele(desired_h1)) in
                        desired_hap1
                    {
                        let m = hi_freq_to_orig[hi_freq_idx];
                        let cur1 = sp.allele1(m);
                        let cur2 = sp.allele2(m);
                        if cur1 == desired_h1 {
                            continue;
                        }
                        if cur2 == desired_h1 {
                            sp.swap_alleles(m);
                            total_switches += 1;
                            sample_changed[s] = true;
                        }
                    }
                }
                Stage1OrientationUpdate::RelativeSwapMask(swap_mask) => {
                    for (hi_freq_idx, RelativeSwapBit(should_swap)) in
                        swap_mask.into_iter().enumerate()
                    {
                        if should_swap {
                            let m = hi_freq_to_orig[hi_freq_idx];
                            sp.swap_alleles(m);
                            total_switches += 1;
                            sample_changed[s] = true;
                        }
                    }
                }
            }

            // Mark hets as phased if LR exceeds threshold (independent of swap decision)
            if !is_burnin {
                for Stage1HetUpdate {
                    marker: HiFreqMarkerIdx(hi_freq_idx),
                    lr: PhaseLogOdds(lr),
                    ..
                } in &decision.het_updates
                {
                    if *lr >= lr_threshold {
                        let m = hi_freq_to_orig[*hi_freq_idx];
                        if sp.is_unphased(m) {
                            sp.mark_phased(m);
                            total_phased += 1;
                            sample_changed[s] = true;
                        }
                    }
                }
            }

            for Stage1HetUpdate {
                marker: HiFreqMarkerIdx(hi_freq_idx),
                confidence: PhaseConfidence(p_orient),
                ..
            } in &decision.het_updates
            {
                let m = hi_freq_to_orig[*hi_freq_idx];
                sp.set_phase_confidence(m, *p_orient);
            }

            if let Some(paths) = decision.paths {
                if let Some(slot) = mcmc_paths.get_mut(s) {
                    *slot = Some(paths);
                }
            }
        }

        // Also update MutableGenotypes to keep in sync for next iteration's PBWT
        self.sync_sample_phases_to_geno(sample_phases, geno);

        // Calculate cumulative phasing progress across all samples.
        let mut total_locked = 0usize;
        let mut total_unphased = 0usize;
        for sp in sample_phases.iter() {
            total_locked += sp.phased_count();
            total_unphased += sp.unphased_count();
        }
        let total_phasable = total_locked + total_unphased;
        let pct_locked = if total_phasable > 0 {
            (total_locked as f64 / total_phasable as f64) * 100.0
        } else {
            100.0
        };

        eprintln!(
            "Applied {} phase switches, {} new markers locked (Stage 1 FB)",
            total_switches, total_phased
        );
        eprintln!(
            "  Completion: {:.1}% ({}/{} heterozygous markers locked)",
            pct_locked, total_locked, total_phasable
        );
        Ok((total_switches, total_phased, sample_changed))
    }

    /// Build final GenotypeMatrix from mutable genotypes
    fn build_final_matrix(
        &self,
        original: &GenotypeMatrix,
        geno: &MutableGenotypes,
        sample_phases: &[SamplePhase],
    ) -> GenotypeMatrix<crate::data::storage::phase_state::Phased> {
        let markers = original.markers().clone();
        let samples = original.samples_arc();
        let n_markers = geno.n_markers();
        let n_samples = samples.len();

        let columns: Vec<GenotypeColumn> = (0..n_markers)
            .map(|m| {
                let alleles = geno.marker_alleles(m);
                let n_alleles = original
                    .markers()
                    .marker(MarkerIdx::new(m as u32))
                    .n_alleles()
                    .max(1);
                GenotypeColumn::from_alleles(&alleles, n_alleles)
            })
            .collect();

        let mut phase_confidence = vec![vec![255u8; n_samples]; n_markers];
        for (s, sp) in sample_phases.iter().enumerate() {
            for m in 0..n_markers {
                let p = sp.phase_confidence(m).clamp(0.0, 1.0);
                phase_confidence[m][s] = (p * 255.0).round() as u8;
            }
        }

        let confidence = original.confidence_clone();
        let pl = original.likelihoods_pl_arc();
        GenotypeMatrix::new_phased_with_confidence_and_likelihoods(
            markers, columns, samples, confidence, pl,
        )
        .with_phase_confidence(Some(phase_confidence))
    }

    fn count_unphased_hets(sample_phases: &[SamplePhase], hi_freq_to_orig: &[usize]) -> usize {
        let mut count = 0usize;
        for sp in sample_phases {
            for &m in hi_freq_to_orig {
                let a1 = sp.allele1(m);
                let a2 = sp.allele2(m);
                if a1 != 255 && a2 != 255 && a1 != a2 && sp.is_unphased(m) {
                    count += 1;
                }
            }
        }
        count
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
    /// from the previous window with current ones, ensuring continuity. The returned handoff
    /// includes identity-aware haplotype priors for the *next* overlap region.
    fn phase_rare_markers_with_hmm(
        &self,
        target_gt: &GenotypeMatrix,
        geno: &mut MutableGenotypes,
        hi_freq_markers: &[usize],
        gen_positions: &[f64],
        hi_freq_gen_positions: &[f64],
        stage1_p_recomb: &[f32],
        sample_phases: &mut [SamplePhase],
        maf: &[f32],
        rare_threshold: f32,
        previous_overlap: Option<&PhasedOverlap>,
        next_overlap_start: Option<usize>,
    ) -> Option<Stage2OverlapHandoff> {
        let n_markers = geno.n_markers();
        let n_haps = geno.n_haps();
        let n_ref_haps = self
            .reference_gt
            .as_ref()
            .map(|r| r.n_haplotypes())
            .unwrap_or(0);
        let n_total_haps = n_haps + n_ref_haps;
        let n_stage1 = hi_freq_markers.len();
        let n_haps_f = target_gt.n_haplotypes() as f32;
        let alt_freqs: Vec<f32> = if let (Some(ref_gt), Some(alignment)) =
            (&self.reference_gt, &self.alignment)
        {
            let n_ref_haps = ref_gt.n_haplotypes() as f32;
            let prior_alpha = 1.0f32;
            let prior_beta = 1.0f32;
            let mut freqs = vec![0.0f32; n_markers];
            for m in 0..n_markers {
                let fallback = if n_haps_f > 0.0 {
                    let alt = target_gt.column(MarkerIdx::new(m as u32)).alt_count() as f32;
                    ((alt + prior_alpha) / (n_haps_f + prior_alpha + prior_beta)).clamp(0.0, 1.0)
                } else {
                    0.5
                };
                let mapping = alignment.allele_mappings.get(m).and_then(|m| m.as_ref());
                if let Some(mapping) = mapping {
                    if let Some(ref_idx) = alignment.target_to_ref(MarkerIdx::new(m as u32)) {
                        if n_ref_haps > 0.0 {
                            let ref_col = ref_gt.column(ref_idx);
                            let ref_alt = (ref_col.alt_count() as f32 + prior_alpha)
                                / (n_ref_haps + prior_alpha + prior_beta);
                            if let Some(&targ_to_ref_alt) = mapping.targ_to_ref.get(1) {
                                if targ_to_ref_alt == 1 {
                                    freqs[m] = ref_alt.clamp(0.0, 1.0);
                                    continue;
                                }
                                if targ_to_ref_alt == 0 {
                                    freqs[m] = (1.0 - ref_alt).clamp(0.0, 1.0);
                                    continue;
                                }
                            }
                        }
                    }
                }
                freqs[m] = fallback;
            }
            freqs
        } else if n_haps_f > 0.0 {
            let prior_alpha = 1.0f32;
            let prior_beta = 1.0f32;
            (0..n_markers)
                .map(|m| {
                    let alt = target_gt.column(MarkerIdx::new(m as u32)).alt_count() as f32;
                    ((alt + prior_alpha) / (n_haps_f + prior_alpha + prior_beta)).clamp(0.0, 1.0)
                })
                .collect()
        } else {
            vec![0.5f32; n_markers]
        };

        if n_stage1 < 2 {
            return None;
        }
        let hi_freq_emit_conf_scale = build_marker_emission_conf_scales(
            target_gt,
            sample_phases,
            hi_freq_markers,
            self.params.p_mismatch,
        );
        if !hi_freq_emit_conf_scale.is_empty() {
            let mut sum_scale = 0.0f32;
            let mut min_scale = 1.0f32;
            for &scale in &hi_freq_emit_conf_scale {
                sum_scale += scale;
                min_scale = min_scale.min(scale);
            }
            let mean_scale = sum_scale / hi_freq_emit_conf_scale.len() as f32;
            eprintln!(
                "[stage2 scaffold emission] markers={} mean_conf_scale={:.4} min_conf_scale={:.4}",
                hi_freq_emit_conf_scale.len(),
                mean_scale,
                min_scale
            );
        }

        // Compute total haplotype count (target + reference)

        // Determine Stage 1 markers involved in the NEXT overlap region (for export)
        let next_overlap_indices = if let Some(start) = next_overlap_start {
            // Find first Stage 1 marker >= start
            let start_stage1 = hi_freq_markers
                .iter()
                .position(|&m| m >= start)
                .unwrap_or(n_stage1);
            (start_stage1..n_stage1).collect()
        } else {
            Vec::new()
        };

        // Determine Stage 1 markers involved in the PREVIOUS overlap region (for import/merge)
        let n_stage1_in_prev_overlap = if let Some(overlap) = previous_overlap {
            // Overlap markers are 0..overlap.n_markers
            hi_freq_markers
                .iter()
                .take_while(|&&m| m < overlap.n_markers)
                .count()
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
        type PhaseResult = (
            Vec<Stage2Decision>,
            Option<Vec<Vec<Vec<f32>>>>,
            Option<[HaplotypePriors; 2]>,
            Option<usize>,
            Option<f64>,
        );

        let n_samples = n_haps / 2;
        let ref_gt = self.reference_gt.as_ref().map(|v| v.as_ref());
        let threaded_haps_vec = match self.build_phasing_prescan_states(
            target_gt,
            geno,
            ref_gt,
            self.alignment.as_ref(),
            n_stage1,
            n_samples,
            hi_freq_gen_positions,
            self.config.imp_step,
            Some(hi_freq_markers),
        ) {
            Ok(states) => states,
            Err(err) => {
                eprintln!("Stage 2 prescan failed: {err}");
                return None;
            }
        };

        // No clone needed: we only read geno during computation; local rephase
        // happens during threaded hap construction above.
        // We use a scoped immutable borrow for the entire computation phase.
        let phase_results: Vec<PhaseResult> = {
            // Immutable borrow of geno for the entire read phase
            let ref_geno: &MutableGenotypes = geno;

            let has_missing = |m: usize| -> bool {
                let m_idx = MarkerIdx::new(m as u32);
                (0..n_haps).any(|h| target_gt.allele(m_idx, HapIdx::new(h as u32)) == 255)
            };
            let rare_markers: Vec<usize> = (0..n_markers)
                .filter(|&m| (maf[m] < rare_threshold && maf[m] > 0.0) || has_missing(m))
                .collect();
            let mut marker_emit_conf_scale = vec![1.0f32; n_markers];
            if !rare_markers.is_empty() {
                let rare_scales = build_marker_emission_conf_scales(
                    target_gt,
                    sample_phases,
                    &rare_markers,
                    self.params.p_mismatch,
                );
                for (idx, &m) in rare_markers.iter().enumerate() {
                    marker_emit_conf_scale[m] = rare_scales[idx];
                }
                let mut sum_scale = 0.0f32;
                let mut min_scale = 1.0f32;
                for &m in &rare_markers {
                    let scale = marker_emit_conf_scale[m];
                    sum_scale += scale;
                    min_scale = min_scale.min(scale);
                }
                let mean_scale = sum_scale / rare_markers.len() as f32;
                eprintln!(
                    "[stage2 emission] rare_markers={} mean_conf_scale={:.4} min_conf_scale={:.4}",
                    rare_markers.len(),
                    mean_scale,
                    min_scale
                );
            }

            // Use CompositeSubset view when reference panel is available
            let subset_view =
                if let (Some(ref_gt), Some(alignment)) = (&self.reference_gt, &self.alignment) {
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
                        if let Some(ref_m) = alignment.target_to_ref(MarkerIdx::new(marker as u32))
                        {
                            let ref_allele = ref_gt.allele(ref_m, HapIdx::new(ref_h as u32));
                            alignment.reverse_map_allele(marker, ref_allele)
                        } else {
                            255
                        }
                    } else {
                        255
                    }
                }
            };

            let carrier_panel_haps = if n_ref_haps > 0 {
                n_ref_haps
            } else {
                n_total_haps
            };
            let mut carrier_haps: Vec<Vec<u32>> = vec![Vec::new(); n_markers];
            let mut carrier_context_markers: Vec<Vec<usize>> = vec![Vec::new(); n_markers];
            let mut total_conditioned_rare = 0usize;
            let mut rare_with_carriers = 0usize;
            let mut rare_with_carrier_context = 0usize;
            for &m in &rare_markers {
                if !(maf[m] < rare_threshold && maf[m] > 0.0) {
                    continue;
                }
                total_conditioned_rare += 1;
                let mut carriers = Vec::new();
                if n_ref_haps > 0 {
                    // Reference-guided Stage 2: only reference carriers should
                    // define injected rare-donor support.
                    for h in n_haps..n_total_haps {
                        let allele = get_allele_global(m, h);
                        if allele > 0 && allele != 255 {
                            carriers.push(h as u32);
                        }
                    }
                } else {
                    for h in 0..n_total_haps {
                        let allele = get_allele_global(m, h);
                        if allele > 0 && allele != 255 {
                            carriers.push(h as u32);
                        }
                    }
                }
                carriers.sort_unstable();
                if carriers.is_empty() {
                    carrier_haps[m] = carriers;
                    continue;
                }
                rare_with_carriers += 1;
                let context = stage2_phaser.carrier_context_markers(m);
                if !context.is_empty() {
                    rare_with_carrier_context += 1;
                }
                carrier_context_markers[m] = context;
                carrier_haps[m] = carriers;
            }
            eprintln!(
                "Stage 2: rare-carrier conditioning markers with carriers={} with_context={} total_rare={}",
                rare_with_carriers, rare_with_carrier_context, total_conditioned_rare
            );

            // Process samples in parallel - collect results: Stage2Decision
            // Note: This is called after all iterations, so we use iteration=0 for deterministic state selection
            sample_phases
                .par_iter()
                .enumerate()
                .map(|(s, sp)| {
                    let threaded_haps = &threaded_haps_vec[s];
                    let n_states = threaded_haps.n_states();

                    // Extract Stage 1 alleles from SamplePhase
                    let seq1: Vec<u8> = hi_freq_markers.iter().map(|&m| sp.allele1(m)).collect();
                    let seq2: Vec<u8> = hi_freq_markers.iter().map(|&m| sp.allele2(m)).collect();
                    let seq_conf: Vec<f32> = hi_freq_markers
                        .iter()
                        .enumerate()
                        .map(|(i, &m)| {
                            (sp.confidence(m) * hi_freq_emit_conf_scale[i]).clamp(0.0, 1.0)
                        })
                        .collect();
                    let hmm = MosaicHmm::new(subset_view, &self.params, n_states, stage1_p_recomb);
                    let plp = PlProvider {
                        gt: target_gt,
                        sample: s,
                        subset_to_orig: Some(hi_freq_markers),
                    };

                    let mut fwd1 = Vec::new();
                    let mut bwd1 = Vec::new();
                    let (init_prior1_storage, init_prior2_storage) =
                        if let Some(overlap) = previous_overlap {
                            let h1_idx = s * 2;
                            let h2_idx = s * 2 + 1;
                            let mut prior_stage1_idx = n_stage1_in_prev_overlap
                                .saturating_sub(1)
                                .min(n_stage1.saturating_sub(1));
                            let mut anchor_source = "tail";
                            if let Some(prior_marker) = overlap.prior_stage1_global_marker() {
                                if let Some(idx) =
                                    hi_freq_markers.iter().position(|&m| m == prior_marker)
                                {
                                    prior_stage1_idx = idx;
                                    anchor_source = "marker";
                                }
                            }
                            if anchor_source == "tail" {
                                if let Some(prior_gen_pos) = overlap.prior_stage1_gen_pos() {
                                    if let Some((idx, _)) = hi_freq_gen_positions
                                        .iter()
                                        .enumerate()
                                        .min_by(|(_, a), (_, b)| {
                                            let da = (*a - prior_gen_pos).abs();
                                            let db = (*b - prior_gen_pos).abs();
                                            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                                        })
                                    {
                                        prior_stage1_idx = idx;
                                        anchor_source = "gen_pos";
                                    }
                                }
                            }
                            if s == 0 && n_stage1 <= 600 {
                                eprintln!(
                                    "[stage2 prior anchor] sample=0 source={} stage1_idx={}",
                                    anchor_source, prior_stage1_idx
                                );
                            }

                            // Identity-aware handoff: project haplotype priors onto the
                            // current window's local state set using state->hap mapping.
                            if let Some(hap_priors) = overlap.hap_priors() {
                                if prior_stage1_idx < n_stage1
                                    && h1_idx < hap_priors.len()
                                    && h2_idx < hap_priors.len()
                                    && n_states > 0
                                {
                                    let mut state_haps = vec![CombinedHapId::new(0); n_states];
                                    threaded_haps.materialize_at(prior_stage1_idx, &mut state_haps);

                                    (
                                        Some(project_haplotype_priors_to_states(
                                            &hap_priors[h1_idx],
                                            &state_haps,
                                        )),
                                        Some(project_haplotype_priors_to_states(
                                            &hap_priors[h2_idx],
                                            &state_haps,
                                        )),
                                    )
                                } else {
                                    (None, None)
                                }
                            } else {
                                (None, None)
                            }
                        } else {
                            (None, None)
                        };
                    let init_prior1: Option<&[f32]> = init_prior1_storage.as_deref();
                    let init_prior2: Option<&[f32]> = init_prior2_storage.as_deref();
                    let allele_freqs_stage1: Vec<f32> =
                        hi_freq_markers.iter().map(|&m| alt_freqs[m]).collect();
                    hmm.conditioned_forward_backward(
                        &seq1,
                        &seq2,
                        &seq2,
                        Some(&seq_conf),
                        Some(&plp),
                        Some(&allele_freqs_stage1),
                        init_prior1,
                        &threaded_haps,
                        &mut fwd1,
                        &mut bwd1,
                    );

                    let mut fwd2 = Vec::new();
                    let mut bwd2 = Vec::new();
                    hmm.conditioned_forward_backward(
                        &seq1,
                        &seq2,
                        &seq1,
                        Some(&seq_conf),
                        Some(&plp),
                        Some(&allele_freqs_stage1),
                        init_prior2,
                        &threaded_haps,
                        &mut fwd2,
                        &mut bwd2,
                    );

                    // Compute posterior state probabilities at each Stage 1 marker
                    let probs1 = compute_state_posteriors(&fwd1, &bwd1, n_stage1, n_states);
                    let probs2 = compute_state_posteriors(&fwd2, &bwd2, n_stage1, n_states);

                    // Do NOT merge previous window probabilities by state index. State
                    // identity is not preserved across windows, and blending by index
                    // corrupts the posterior.

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

                    // Export identity-aware haplotype priors for the next window.
                    let (next_hap_priors, next_prior_global_marker, next_prior_gen_pos) =
                        if !next_overlap_indices.is_empty() {
                            let stage1_idx = next_overlap_indices[0];
                            if stage1_idx < probs1.len() && n_states > 0 {
                                let mut state_haps = vec![CombinedHapId::new(0); n_states];
                                threaded_haps.materialize_at(stage1_idx, &mut state_haps);

                                // Use fwd probabilities only (unidirectional priors) to avoid leakage
                                // from future data in the overlap region.
                                let row_start = stage1_idx * n_states;
                                let fwd1_slice = &fwd1[row_start..row_start + n_states];
                                let fwd1_sum: f32 = fwd1_slice.iter().sum();
                                let norm_fwd1: Vec<f32> = if fwd1_sum > 0.0 {
                                    fwd1_slice.iter().map(|&p| p / fwd1_sum).collect()
                                } else {
                                    vec![1.0 / n_states as f32; n_states]
                                };

                                let fwd2_slice = &fwd2[row_start..row_start + n_states];
                                let fwd2_sum: f32 = fwd2_slice.iter().sum();
                                let norm_fwd2: Vec<f32> = if fwd2_sum > 0.0 {
                                    fwd2_slice.iter().map(|&p| p / fwd2_sum).collect()
                                } else {
                                    vec![1.0 / n_states as f32; n_states]
                                };

                                let prior1 = build_haplotype_priors_from_state_probs(
                                    &norm_fwd1,
                                    &state_haps,
                                    PRIOR_EXPORT_MIN_PROB,
                                );
                                let prior2 = build_haplotype_priors_from_state_probs(
                                    &norm_fwd2,
                                    &state_haps,
                                    PRIOR_EXPORT_MIN_PROB,
                                );
                                let marker = hi_freq_markers.get(stage1_idx).copied();
                                let gen_pos = hi_freq_gen_positions.get(stage1_idx).copied();
                                (Some([prior1, prior2]), marker, gen_pos)
                            } else {
                                (None, None, None)
                            }
                        } else {
                            (None, None, None)
                        };

                    // Lazy cache for state->hap mapping.
                    let mut hap_cache: Vec<Option<Vec<CombinedHapId>>> = vec![None; n_markers];

                    // Closure to get allele for any haplotype (target or reference)
                    let get_allele = |marker: usize, hap: usize| -> u8 {
                        if hap < n_haps {
                            // Target haplotype
                            ref_geno.get(marker, HapIdx::new(hap as u32))
                        } else {
                            // Reference haplotype
                            let ref_h = hap - n_haps;
                            if let (Some(ref_gt), Some(alignment)) =
                                (&self.reference_gt, &self.alignment)
                            {
                                if let Some(ref_m) =
                                    alignment.target_to_ref(MarkerIdx::new(marker as u32))
                                {
                                    let ref_allele =
                                        ref_gt.allele(ref_m, HapIdx::new(ref_h as u32));
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
                    let mut phase_evidence: Vec<PhaseEvidence> = Vec::new();

                    // Inline helper macro for imputing a single allele
                    // Matches Java Stage2Baum.imputeAllele()
                    macro_rules! impute_allele {
                        ($m:expr, $probs:expr) => {{
                            let m = $m;
                            let probs = $probs;
                            let n_alleles = target_gt
                                .markers()
                                .marker(MarkerIdx::new(m as u32))
                                .n_alleles()
                                .max(1);
                            let mut al_probs = vec![0.0f32; n_alleles];

                            let mkr_a = stage2_phaser.prev_stage1_marker[m];
                            let mkr_b = (mkr_a + 1).min(n_stage1.saturating_sub(1));
                            if hap_cache[mkr_a].is_none() {
                                let mut haps = vec![CombinedHapId::new(0); n_states];
                                threaded_haps.materialize_at(mkr_a, &mut haps);
                                hap_cache[mkr_a] = Some(haps);
                            }
                            if hap_cache[mkr_b].is_none() {
                                let mut haps = vec![CombinedHapId::new(0); n_states];
                                threaded_haps.materialize_at(mkr_b, &mut haps);
                                hap_cache[mkr_b] = Some(haps);
                            }
                            let state_haps_a = hap_cache[mkr_a].as_deref().unwrap_or(&[]);
                            let state_haps_b = hap_cache[mkr_b].as_deref().unwrap_or(&[]);
                            let bridge_probs = stage2_phaser.bridge_hap_probs(
                                m,
                                probs,
                                state_haps_a,
                                state_haps_b,
                            );

                            for (&hap, &prob_state) in bridge_probs.iter() {
                                let hap_allele = get_allele(m, hap as usize);

                                if hap_allele != 255 {
                                    let idx = hap_allele as usize;
                                    if idx < al_probs.len() {
                                        al_probs[idx] += prob_state;
                                    }
                                }
                            }

                            al_probs
                                .iter()
                                .enumerate()
                                .max_by(|(_, a), (_, b)| {
                                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                                })
                                .map(|(idx, _)| idx as u8)
                                .unwrap_or(0)
                        }};
                    }

                    for &m in &rare_markers {
                        let a1 = sp.allele1(m);
                        let a2 = sp.allele2(m);
                        let mkr_a = stage2_phaser.prev_stage1_marker[m];
                        let mkr_b = (mkr_a + 1).min(n_stage1.saturating_sub(1));
                        if hap_cache[mkr_a].is_none() {
                            let mut haps = vec![CombinedHapId::new(0); n_states];
                            threaded_haps.materialize_at(mkr_a, &mut haps);
                            hap_cache[mkr_a] = Some(haps);
                        }
                        if hap_cache[mkr_b].is_none() {
                            let mut haps = vec![CombinedHapId::new(0); n_states];
                            threaded_haps.materialize_at(mkr_b, &mut haps);
                            hap_cache[mkr_b] = Some(haps);
                        }
                        let state_haps_for_interp_a = hap_cache[mkr_a].as_deref().unwrap_or(&[]);
                        let state_haps_for_interp_b = hap_cache[mkr_b].as_deref().unwrap_or(&[]);
                        let marker_maf = maf[m];
                        let is_rare_marker = marker_maf < rare_threshold;
                        let carriers = &carrier_haps[m];
                        let context_markers = &carrier_context_markers[m];
                        let obs_conf =
                            (sp.confidence(m) * marker_emit_conf_scale[m]).clamp(0.05, 1.0);
                        let bridge_probs1 = if is_rare_marker && !carriers.is_empty() {
                            stage2_phaser.carrier_injected_bridge_hap_probs(
                                m,
                                &probs1,
                                state_haps_for_interp_a,
                                state_haps_for_interp_b,
                                carriers,
                                context_markers,
                                carrier_panel_haps.max(1),
                                s * 2,
                                obs_conf,
                                self.params.p_mismatch,
                                &get_allele,
                            )
                        } else {
                            stage2_phaser.bridge_hap_probs(
                                m,
                                &probs1,
                                state_haps_for_interp_a,
                                state_haps_for_interp_b,
                            )
                        };
                        let bridge_probs2 = if is_rare_marker && !carriers.is_empty() {
                            stage2_phaser.carrier_injected_bridge_hap_probs(
                                m,
                                &probs2,
                                state_haps_for_interp_a,
                                state_haps_for_interp_b,
                                carriers,
                                context_markers,
                                carrier_panel_haps.max(1),
                                s * 2 + 1,
                                obs_conf,
                                self.params.p_mismatch,
                                &get_allele,
                            )
                        } else {
                            stage2_phaser.bridge_hap_probs(
                                m,
                                &probs2,
                                state_haps_for_interp_a,
                                state_haps_for_interp_b,
                            )
                        };

                        // Handle missing genotypes by imputation
                        if sp.is_missing(m) || a1 == 255 || a2 == 255 {
                            let imp_a1 = if let Some((h, top, second)) =
                                top_bridge_haplotype(&bridge_probs1)
                            {
                                let al = get_allele(m, h as usize);
                                if donor_log_odds_pass(
                                    top,
                                    second,
                                    self.params.p_mismatch,
                                    obs_conf,
                                ) && al != 255
                                {
                                    al
                                } else {
                                    impute_allele!(m, &probs1)
                                }
                            } else {
                                impute_allele!(m, &probs1)
                            };
                            let imp_a2 = if let Some((h, top, second)) =
                                top_bridge_haplotype(&bridge_probs2)
                            {
                                let al = get_allele(m, h as usize);
                                if donor_log_odds_pass(
                                    top,
                                    second,
                                    self.params.p_mismatch,
                                    obs_conf,
                                ) && al != 255
                                {
                                    al
                                } else {
                                    impute_allele!(m, &probs2)
                                }
                            } else {
                                impute_allele!(m, &probs2)
                            };
                            decisions.push(Stage2Decision::Impute {
                                marker: m,
                                a1: imp_a1,
                                a2: imp_a2,
                            });
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

                        let mut log_same = 0.0f32;
                        let mut log_swap = 0.0f32;

                        if is_rare_marker && !carriers.is_empty() {
                            let mut score1 = 0.0f32;
                            let mut score2 = 0.0f32;
                            for (&hap, &prob) in &bridge_probs1 {
                                if carriers.binary_search(&hap).is_ok() {
                                    score1 += prob;
                                }
                            }
                            for (&hap, &prob) in &bridge_probs2 {
                                if carriers.binary_search(&hap).is_ok() {
                                    score2 += prob;
                                }
                            }
                            log_same += obs_conf * score1.max(1e-30).ln();
                            log_swap += obs_conf * score2.max(1e-30).ln();
                        }

                        let al_probs1 = stage2_phaser.interpolated_allele_probs_from_bridge(
                            m,
                            &bridge_probs1,
                            &get_allele,
                            a1,
                            a2,
                        );
                        let al_probs2 = stage2_phaser.interpolated_allele_probs_from_bridge(
                            m,
                            &bridge_probs2,
                            &get_allele,
                            a1,
                            a2,
                        );

                        let p1 = al_probs1[0] * al_probs2[1];
                        let p2 = al_probs1[1] * al_probs2[0];
                        log_same += obs_conf * p1.max(1e-30).ln();
                        log_swap += obs_conf * p2.max(1e-30).ln();

                        // Scaffold orientation evidence from dominant copied donors.
                        if let (Some((h1, top1, second1)), Some((h2, top2, second2))) = (
                            top_bridge_haplotype(&bridge_probs1),
                            top_bridge_haplotype(&bridge_probs2),
                        ) {
                            if donor_log_odds_pass(top1, second1, self.params.p_mismatch, obs_conf)
                                && donor_log_odds_pass(
                                    top2,
                                    second2,
                                    self.params.p_mismatch,
                                    obs_conf,
                                )
                            {
                                let d1 = get_allele(m, h1 as usize);
                                let d2 = get_allele(m, h2 as usize);
                                let w1 = top1.max(1e-6).ln();
                                let w2 = top2.max(1e-6).ln();
                                if d1 == a1 {
                                    log_same += obs_conf * w1;
                                } else if d1 == a2 {
                                    log_swap += obs_conf * w1;
                                }
                                if d2 == a2 {
                                    log_same += obs_conf * w2;
                                } else if d2 == a1 {
                                    log_swap += obs_conf * w2;
                                }
                            }
                        }

                        // Private/singleton-like sites are weakly identifiable under
                        // pure copying. Apply a low-confidence coalescent heuristic:
                        // place the non-reference allele on the haplotype with weaker
                        // local panel match (less copied from panel backbone).
                        let singleton_private = is_rare_marker
                            && carriers.is_empty()
                            && ((a1 == 0 && a2 > 0) || (a2 == 0 && a1 > 0));
                        if singleton_private {
                            let match1 = top_bridge_haplotype(&bridge_probs1)
                                .map(|(_, p, _)| p)
                                .unwrap_or(0.5);
                            let match2 = top_bridge_haplotype(&bridge_probs2)
                                .map(|(_, p, _)| p)
                                .unwrap_or(0.5);
                            let alt_on_h1 = a1 > 0 && a2 == 0;
                            let prefer_h1 = match1 <= match2;
                            let support_same = if alt_on_h1 { prefer_h1 } else { !prefer_h1 };
                            let bias = (0.02 + 0.10 * (match1 - match2).abs()).clamp(0.02, 0.12);
                            if support_same {
                                log_same += bias;
                            } else {
                                log_swap += bias;
                            }
                        }

                        phase_evidence.push(PhaseEvidence {
                            marker: m,
                            log_same,
                            log_swap,
                        });
                    }

                    for (marker, should_swap, lr) in decode_phase_evidence_path(
                        &phase_evidence,
                        gen_positions,
                        self.params.recomb_intensity,
                    ) {
                        decisions.push(Stage2Decision::Phase {
                            marker,
                            should_swap,
                            lr,
                        });
                    }

                    (
                        decisions,
                        next_probs,
                        next_hap_priors,
                        next_prior_global_marker,
                        next_prior_gen_pos,
                    )
                })
                .collect::<Vec<_>>()
        }; // ref_geno borrow ends here

        let mut all_next_hap_priors = if next_overlap_start.is_some() {
            Some(Vec::with_capacity(n_haps))
        } else {
            None
        };
        let mut next_prior_global_marker: Option<usize> = None;
        let mut next_prior_gen_pos: Option<f64> = None;

        // Apply phase changes and imputations to SamplePhase
        let mut total_switches = 0;
        let mut total_phased = 0;
        let mut total_imputed = 0;

        // Stage 2 runs after all iterations, so lr_threshold is typically 1.0
        // (all decisions pass). We still check for consistency with Stage 1.
        let lr_threshold = self.params.lr_threshold;

        for (s, (decisions, _, next_hap_priors, prior_marker, prior_gen_pos)) in
            phase_results.into_iter().enumerate()
        {
            if let Some(all) = all_next_hap_priors.as_mut() {
                if let Some(priors_pair) = next_hap_priors {
                    all.push(priors_pair[0].clone());
                    all.push(priors_pair[1].clone());
                    if next_prior_global_marker.is_none() {
                        next_prior_global_marker = prior_marker;
                    }
                    if next_prior_gen_pos.is_none() {
                        next_prior_gen_pos = prior_gen_pos;
                    }
                } else {
                    all.push(HaplotypePriors::empty());
                    all.push(HaplotypePriors::empty());
                }
            }

            let sp = &mut sample_phases[s];

            for decision in decisions {
                match decision {
                    Stage2Decision::Phase {
                        marker: m,
                        should_swap,
                        lr,
                    } => {
                        // Double-check still unphased (should always be true)
                        if !sp.is_unphased(m) {
                            continue;
                        }

                        let confident = lr >= lr_threshold;
                        if should_swap {
                            sp.swap_haps(m, m + 1);
                            if confident {
                                total_switches += 1;
                            }
                        }

                        let phase_conf = if sp.has_input_phase_anchor() {
                            lr / (1.0 + lr)
                        } else {
                            0.5
                        };
                        sp.set_phase_confidence(m, phase_conf);

                        // Only mark as phased if likelihood ratio exceeds threshold
                        // (Stage 2 runs after iterations, so threshold is typically 1.0)
                        if confident {
                            sp.mark_phased(m);
                            total_phased += 1;
                        }
                    }
                    Stage2Decision::Impute { marker: m, a1, a2 } => {
                        // Set imputed alleles for missing marker
                        sp.set_imputed(m, a1, a2);
                        if !sp.has_input_phase_anchor() {
                            sp.set_phase_confidence(m, 0.5);
                        }
                        total_imputed += 1;
                    }
                }
            }
        }

        eprintln!(
            "Stage 2: Applied {} phase switches, {} markers phased, {} markers imputed (HMM interpolation)",
            total_switches, total_phased, total_imputed
        );

        let next_state_probs = None;

        let next_hap_priors = all_next_hap_priors.and_then(|priors| {
            if priors.len() == n_haps && priors.iter().any(|p| !p.is_empty()) {
                Some(priors)
            } else {
                None
            }
        });

        if next_state_probs.is_some() || next_hap_priors.is_some() {
            return Some(Stage2OverlapHandoff {
                state_probs: next_state_probs,
                hap_priors: next_hap_priors,
                prior_stage1_global_marker: next_prior_global_marker,
                prior_stage1_gen_pos: next_prior_gen_pos,
            });
        }

        None
    }
}

const PRIOR_EXPORT_MIN_PROB: f32 = 1e-5;

/// Project haplotype-identity priors onto the current window's local state set.
fn project_haplotype_priors_to_states(
    priors: &HaplotypePriors,
    state_haps: &[CombinedHapId],
) -> Vec<f32> {
    let n_states = state_haps.len();
    if n_states == 0 {
        return Vec::new();
    }

    let mut out = vec![0.0f32; n_states];
    let mut covered_mass = 0.0f32;

    for (k, &hap) in state_haps.iter().enumerate() {
        let p = priors.prob_of(GlobalHapId(hap.as_u32())).unwrap_or(0.0);
        out[k] = p;
        covered_mass += p;
    }

    // Conditional projection onto the active set: renormalize covered mass.
    // This is the exact projection under P(h | h in active_set).
    if covered_mass > 1e-6 {
        for p in &mut out {
            *p /= covered_mass;
        }
    } else {
        let uniform = 1.0 / n_states as f32;
        out.fill(uniform);
    }

    out
}

/// Build haplotype priors from state posteriors.
fn build_haplotype_priors_from_state_probs(
    state_probs: &[f32],
    state_haps: &[CombinedHapId],
    min_prob: f32,
) -> HaplotypePriors {
    let mut mass_by_hap: std::collections::HashMap<u32, f32> =
        std::collections::HashMap::with_capacity(state_haps.len());

    for (k, &hap) in state_haps.iter().enumerate() {
        let p: f32 = state_probs.get(k).copied().unwrap_or(0.0);
        if p.is_finite() && p > 0.0 {
            *mass_by_hap.entry(hap.as_u32()).or_insert(0.0) += p;
        }
    }

    let mut entries: Vec<(u32, f32)> = mass_by_hap
        .into_iter()
        .filter(|&(_, p)| p >= min_prob)
        .collect();

    if entries.is_empty() {
        return HaplotypePriors::empty();
    }

    entries.sort_unstable_by_key(|(hap, _)| *hap);
    let mut hap_ids: Vec<GlobalHapId> = Vec::with_capacity(entries.len());
    let mut probs: Vec<f32> = Vec::with_capacity(entries.len());
    for (hap, p) in entries {
        hap_ids.push(GlobalHapId(hap));
        probs.push(p);
    }

    HaplotypePriors::new(hap_ids, probs)
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

fn select_reduction_sparse_markers(seq1: &[u8], seq2: &[u8], sample_conf: &[f32]) -> Vec<usize> {
    let n_markers = seq1.len().min(seq2.len());
    if n_markers == 0 {
        return Vec::new();
    }
    let budget = REDUCTION_SPARSE_MAX_MARKERS.min(n_markers);
    if n_markers <= budget {
        return (0..n_markers).collect();
    }

    let mut het_markers: Vec<usize> = Vec::new();
    let mut informative_markers: Vec<usize> = Vec::new();
    for m in 0..n_markers {
        let a1 = seq1[m];
        let a2 = seq2[m];
        if a1 == 255 || a2 == 255 {
            continue;
        }
        informative_markers.push(m);
        if a1 != a2 {
            het_markers.push(m);
        }
    }

    let mut selected: Vec<usize> = Vec::with_capacity(budget + 2);
    selected.push(0);
    selected.push(n_markers - 1);

    let take_evenly = |src: &[usize], take: usize, dst: &mut Vec<usize>| {
        if src.is_empty() || take == 0 {
            return;
        }
        if src.len() <= take {
            dst.extend_from_slice(src);
            return;
        }
        for i in 0..take {
            let idx = i * src.len() / take;
            dst.push(src[idx]);
        }
    };

    let het_quota = budget * REDUCTION_SPARSE_HET_FRAC_NUM / REDUCTION_SPARSE_HET_FRAC_DEN;
    take_evenly(
        &het_markers,
        het_quota.min(het_markers.len()),
        &mut selected,
    );

    let remaining = budget.saturating_sub(selected.len());
    if remaining > 0 {
        if !informative_markers.is_empty() {
            take_evenly(
                &informative_markers,
                remaining.min(informative_markers.len()),
                &mut selected,
            );
        } else {
            let stride = n_markers.div_ceil(remaining.max(1)).max(1);
            let mut m = 0usize;
            while m < n_markers && selected.len() < budget {
                selected.push(m);
                m = m.saturating_add(stride);
            }
        }
    }

    selected.sort_unstable();
    selected.dedup();
    if selected.len() > budget {
        let mut conf_ranked: Vec<(usize, f32)> = selected
            .into_iter()
            .map(|m| {
                let c = sample_conf.get(m).copied().unwrap_or(1.0).clamp(0.0, 1.0);
                let a1 = seq1[m];
                let a2 = seq2[m];
                let het_bonus = if a1 != 255 && a2 != 255 && a1 != a2 {
                    0.25f32
                } else {
                    0.0
                };
                (m, c + het_bonus)
            })
            .collect();
        conf_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        conf_ranked.truncate(budget);
        let mut out: Vec<usize> = conf_ranked.into_iter().map(|(m, _)| m).collect();
        out.sort_unstable();
        out
    } else {
        selected
    }
}

fn select_top_k_by_mass_two_sparse_fb(
    fwd1: &[f32],
    bwd1: &[f32],
    fwd2: &[f32],
    bwd2: &[f32],
    n_states: usize,
    marker_indices: &[usize],
    k: usize,
) -> Vec<usize> {
    let mut mass = vec![0.0f32; n_states];
    for &m in marker_indices {
        let row_start = m * n_states;
        let mut sum1 = 0.0f32;
        let mut sum2 = 0.0f32;
        for i in 0..n_states {
            sum1 += fwd1[row_start + i] * bwd1[row_start + i];
            sum2 += fwd2[row_start + i] * bwd2[row_start + i];
        }
        let inv1 = if sum1 > 0.0 { 1.0 / sum1 } else { 0.0 };
        let inv2 = if sum2 > 0.0 { 1.0 / sum2 } else { 0.0 };
        for i in 0..n_states {
            mass[i] += (fwd1[row_start + i] * bwd1[row_start + i]) * inv1;
            mass[i] += (fwd2[row_start + i] * bwd2[row_start + i]) * inv2;
        }
    }
    let mut idx: Vec<usize> = (0..n_states).collect();
    idx.sort_by(|&a, &b| {
        mass[b]
            .partial_cmp(&mass[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(k.min(n_states));
    idx
}

fn build_anchor_constraints(sample_phase: &SamplePhase) -> (Vec<u8>, Vec<u8>) {
    let n_markers = sample_phase.len();
    let mut hap1 = vec![255u8; n_markers];
    let mut hap2 = vec![255u8; n_markers];
    for m in 0..n_markers {
        if sample_phase.is_unphased(m) {
            continue;
        }
        let a1 = sample_phase.allele1(m);
        let a2 = sample_phase.allele2(m);
        if a1 == 255 || a2 == 255 {
            continue;
        }
        if a1 == a2 {
            continue;
        }
        hap1[m] = a1;
        hap2[m] = a2;
    }
    (hap1, hap2)
}

fn build_sample_confidence(target_gt: &GenotypeMatrix) -> Vec<Vec<f32>> {
    let n_samples = target_gt.n_samples();
    let n_markers = target_gt.n_markers();

    (0..n_samples)
        .map(|s| {
            (0..n_markers)
                .map(|m| {
                    let m_idx = MarkerIdx::new(m as u32);
                    if let Some(pl) = target_gt.sample_pl(m_idx, s) {
                        if pl.is_empty() {
                            return target_gt.sample_confidence_f32(m_idx, s);
                        }
                        pl_confidence_from_posterior(pl)
                    } else {
                        target_gt.sample_confidence_f32(m_idx, s)
                    }
                })
                .collect()
        })
        .collect()
}

fn build_marker_emission_conf_scales(
    target_gt: &GenotypeMatrix,
    sample_phases: &[SamplePhase],
    markers: &[usize],
    base_error_rate: f32,
) -> Vec<f32> {
    if markers.is_empty() {
        return Vec::new();
    }
    let n_samples = sample_phases.len();
    if n_samples == 0 {
        return vec![1.0; markers.len()];
    }

    let base_error = base_error_rate.clamp(1e-6, 0.45);
    let probe_target = n_samples.min(EMIT_PROFILE_MAX_PROBE_SAMPLES).max(1);
    let stride = n_samples.div_ceil(probe_target).max(1);
    let mut probe_samples = Vec::with_capacity(probe_target);
    for s in (0..n_samples).step_by(stride) {
        probe_samples.push(s);
        if probe_samples.len() >= probe_target {
            break;
        }
    }
    if probe_samples.is_empty() {
        probe_samples.push(0);
    }

    let mut marker_errors = vec![base_error; markers.len()];
    for (idx, &m) in markers.iter().enumerate() {
        let n_alleles = target_gt
            .markers()
            .marker(MarkerIdx::new(m as u32))
            .n_alleles()
            .max(2);
        let type_scale = if n_alleles > 2 { 1.6 } else { 1.0 };
        let prior_mean = (base_error * type_scale).clamp(base_error, 0.45);
        let alpha = (prior_mean * EMIT_PROFILE_PRIOR_STRENGTH).max(1e-6);
        let beta = ((1.0 - prior_mean) * EMIT_PROFILE_PRIOR_STRENGTH).max(1e-6);

        let mut weighted_residual_sum = 0.0f32;
        let mut weight_sum = 0.0f32;
        let mut missing_count = 0usize;
        for &s in &probe_samples {
            let sp = &sample_phases[s];
            let a1 = sp.allele1(m);
            let a2 = sp.allele2(m);
            if a1 == 255 || a2 == 255 {
                missing_count += 1;
                continue;
            }
            let conf = sp.confidence(m).clamp(0.0, 1.0);
            let heterozygous = a1 != a2;
            let gate = if !heterozygous {
                1.0
            } else {
                let phase_conf = sp.phase_confidence(m).clamp(0.0, 1.0);
                if phase_conf.min(1.0 - phase_conf) <= (1.0 - EMIT_PROFILE_HET_CONFIDENCE_GATE) {
                    0.35
                } else {
                    0.0
                }
            };
            if gate <= 0.0 {
                continue;
            }
            let weight = gate * (0.15 + 0.85 * conf);
            let residual = (1.0 - conf).clamp(0.0, 1.0);
            weighted_residual_sum += weight * residual;
            weight_sum += weight;
        }

        let probe_n = probe_samples.len().max(1) as f32;
        let missing_rate = missing_count as f32 / probe_n;
        let missing_weight = missing_rate * 2.0;
        weighted_residual_sum += missing_weight * 0.25;
        weight_sum += missing_weight;

        let posterior =
            (alpha + weighted_residual_sum) / (alpha + beta + weight_sum).max(alpha + beta + 1e-6);
        marker_errors[idx] = posterior.clamp(base_error, 0.45);
    }

    if marker_errors.len() >= 3 {
        let mut smoothed = marker_errors.clone();
        for i in 1..(marker_errors.len() - 1) {
            smoothed[i] =
                0.25 * marker_errors[i - 1] + 0.5 * marker_errors[i] + 0.25 * marker_errors[i + 1];
        }
        marker_errors = smoothed;
    }

    let denom = (0.5 - base_error).max(1e-6);
    marker_errors
        .into_iter()
        .map(|err| ((0.5 - err) / denom).clamp(EMIT_PROFILE_MIN_CONF_SCALE, 1.0))
        .collect()
}

fn stage1_sample_phase_stability(sp: &SamplePhase, hi_freq_to_orig: &[usize]) -> f32 {
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for &m in hi_freq_to_orig {
        let a1 = sp.allele1(m);
        let a2 = sp.allele2(m);
        if a1 == 255 || a2 == 255 || a1 == a2 {
            continue;
        }
        sum += sp.phase_confidence(m).clamp(0.0, 1.0);
        count += 1;
    }
    if count == 0 { 0.5 } else { sum / count as f32 }
}

#[inline]
fn pl_confidence_from_posterior(pl: &[u16]) -> f32 {
    if pl.is_empty() {
        return 0.5;
    }
    let min_pl = pl.iter().copied().min().unwrap_or(0) as f64;
    let mut denom = 0.0f64;
    let mut best = 0.0f64;
    for &v in pl {
        let rel = (v as f64) - min_pl;
        let p = 10f64.powf(-rel / 10.0);
        denom += p;
        if p > best {
            best = p;
        }
    }
    if denom <= 0.0 || !denom.is_finite() {
        return 0.5;
    }
    let w = (best / denom).clamp(0.0, 1.0);
    if w <= 0.5 {
        0.0
    } else {
        (2.0 * w - 1.0) as f32
    }
}

#[inline(always)]
fn emit_prob(ref_al: u8, targ_al: u8, conf: f32, p_no_err: f32, p_err: f32) -> f32 {
    let neutral = 0.5 * (p_no_err + p_err);
    let base = if ref_al == 255 || targ_al == 255 {
        neutral
    } else if ref_al == targ_al {
        p_no_err
    } else {
        p_err
    };
    base * conf + 0.5 * (1.0 - conf)
}

#[inline(always)]
fn emit_prob_hard(
    ref_al: u8,
    targ_al: u8,
    conf: f32,
    p_no_err: f32,
    p_err: f32,
    hard: bool,
) -> f32 {
    if hard && targ_al != 255 {
        if targ_al == INVALID_ALLELE {
            return p_err.max(1e-8);
        }
        if ref_al == 255 {
            let neutral = 0.5 * (p_no_err + p_err);
            return neutral * conf + 0.5 * (1.0 - conf);
        }
        return if ref_al == targ_al {
            p_no_err
        } else {
            p_err.max(1e-8)
        };
    }
    emit_prob(ref_al, targ_al, conf, p_no_err, p_err)
}

#[inline(always)]
fn subset_transition_params(r: f32, n_states: usize, panel_haps: usize) -> (f32, f32, f32) {
    if n_states == 0 {
        return (0.0, 0.0, 1.0);
    }
    let r = r.clamp(0.0, 1.0);
    let k_raw = n_states as f32;
    let n_total = panel_haps.max(1) as f32;
    // Subset-space approximation: preserve recombination mass inside the tractable
    // active set instead of suppressing switching by K/N_total when K << N_total.
    let k = k_raw.clamp(1.0, n_total);
    let switch_full = r / k.max(1.0);
    let z = ((1.0 - r) + k * switch_full).max(1e-12);
    let shift = switch_full / z;
    let stay_gap = (1.0 - r) / z;
    let stay = ((1.0 - r) + switch_full) / z;
    (stay_gap, shift, stay)
}

/// Emission mode for combined diploid genotype
#[derive(Clone, Copy)]
enum CombinedEmitMode {
    AllMissing,             // a1==255 && a2==255: always p_no_err
    Het { a1: u8, a2: u8 }, // a1!=a2: match if ref in {a1,a2,255}
    HomOrHemi { obs: u8 },  // hom or one missing: match if ref==obs or missing
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
fn emit_combined_fast(
    ref_al: u8,
    mode: CombinedEmitMode,
    conf: f32,
    p_no_err: f32,
    p_err: f32,
) -> f32 {
    let neutral = 0.5 * (p_no_err + p_err);
    if ref_al == 255 {
        return neutral * conf + 0.5 * (1.0 - conf);
    }
    let base = match mode {
        CombinedEmitMode::AllMissing => neutral,
        CombinedEmitMode::Het { a1, a2 } => {
            if ref_al == a1 || ref_al == a2 {
                p_no_err
            } else {
                p_err
            }
        }
        CombinedEmitMode::HomOrHemi { obs } => {
            if obs == 255 {
                neutral
            } else if ref_al == obs {
                p_no_err
            } else {
                p_err
            }
        }
    };
    base * conf + 0.5 * (1.0 - conf)
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

    let neutral_emit = 0.5 * (p_no_err + p_err);
    if ref_al == 255 {
        return conf * neutral_emit + (1.0 - conf) * 0.5;
    }

    // Unphased heterozygote with no fixed partner: allow either allele.
    if geno_a1 != geno_a2 && fixed_allele == 255 {
        let matches = ref_al == geno_a1 || ref_al == geno_a2;
        let raw_emit = if matches { p_no_err } else { p_err };
        return conf * raw_emit + (1.0 - conf) * 0.5;
    }

    // Inconsistent fixed allele must not impose a hard opposite-allele constraint.
    if geno_a1 != geno_a2 && fixed_allele != geno_a1 && fixed_allele != geno_a2 {
        let matches = ref_al == geno_a1 || ref_al == geno_a2;
        let raw_emit = if matches { p_no_err } else { p_err };
        return conf * raw_emit + (1.0 - conf) * 0.5;
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
    let raw_emit = if ref_al == required_allele {
        p_no_err
    } else {
        p_err
    };

    // Blend with uniform based on confidence
    conf * raw_emit + (1.0 - conf) * 0.5
}

#[inline]
fn constrained_emission_confidence(
    geno_a1: u8,
    geno_a2: u8,
    fixed_allele: u8,
    geno_conf: f32,
    phase_conf: f32,
) -> f32 {
    let g = geno_conf.clamp(0.0, 1.0);
    if geno_a1 != geno_a2 && fixed_allele != 255 {
        // At constrained heterozygotes, partner orientation uncertainty must
        // soften the emission even when genotype confidence is high.
        (g * phase_conf.clamp(0.0, 1.0)).clamp(0.0, 1.0)
    } else {
        g
    }
}

#[derive(Clone, Copy)]
struct HapEmissionInputs<'a> {
    /// Allele this haplotype is constrained to emit (non-PL emission path).
    target_constraint: &'a [u8],
    /// Allele carried by the partner haplotype (PL conditioning path).
    partner_allele: &'a [u8],
    /// Per-marker flag controlling combined (unconditioned) emissions.
    use_combined: &'a [bool],
    /// Per-marker flag enforcing hard match to target constraint.
    hard_match: &'a [bool],
}

#[inline]
fn compute_pl_allele_probs(
    pl: Option<&[u16]>,
    use_combined: bool,
    partner_allele: u8,
    allele_probs: &mut Vec<f32>,
) -> Option<usize> {
    let pl = pl.filter(|v| !v.is_empty())?;
    if use_combined {
        allele_probs_uncond_from_pl(pl, None, allele_probs)
    } else {
        allele_probs_cond_from_pl(pl, partner_allele, None, allele_probs)
            .or_else(|| allele_probs_uncond_from_pl(pl, None, allele_probs))
    }
}

#[inline]
fn build_pl_emit_lut(lut: &mut [f32; 256], allele_probs: &[f32], p_no_err: f32, p_err_other: f32) {
    let default_emit = p_err_other.max(1e-30);
    lut.fill(default_emit);
    for (a, &p_true) in allele_probs.iter().take(255).enumerate() {
        lut[a] = (p_no_err * p_true + p_err_other * (1.0 - p_true)).max(1e-30);
    }
    lut[255] = emit_from_allele_probs(255, allele_probs, p_no_err, p_err_other);
}

#[inline]
fn refresh_path_ref_from_states(path_ref: &mut [u32], path_idx: &[u32], neighbors: &[u32]) {
    for (m, &state_u32) in path_idx.iter().enumerate() {
        let state = state_u32 as usize;
        if state < neighbors.len() {
            path_ref[m] = neighbors[state];
        }
    }
}

fn compute_label_switch_transition_logs(
    p_recomb: &[f32],
    het_positions: &[usize],
) -> Vec<(f32, f32)> {
    let n_hets = het_positions.len();
    let mut out = vec![(0.0f32, 0.0f32); n_hets];
    if n_hets < 2 {
        return out;
    }

    let mut prefix_log_no_recomb = vec![0.0f64; p_recomb.len() + 1];
    for (i, &r_raw) in p_recomb.iter().enumerate() {
        let r = r_raw.clamp(0.0, 1.0 - 1e-12) as f64;
        prefix_log_no_recomb[i + 1] = prefix_log_no_recomb[i] + (1.0 - r).ln();
    }

    let last_recomb = p_recomb.len().saturating_sub(1);
    for i in 1..n_hets {
        let prev = het_positions[i - 1];
        let curr = het_positions[i];
        let switch_p = if curr <= prev {
            0.01f32
        } else {
            let start = (prev + 1).min(p_recomb.len());
            let end = curr.min(last_recomb);
            let log_stay = if start <= end {
                prefix_log_no_recomb[end + 1] - prefix_log_no_recomb[start]
            } else {
                0.0
            };
            let stay = log_stay.exp().clamp(0.0, 1.0);
            (1.0 - stay as f32).clamp(1e-6, 1.0 - 1e-6)
        };
        out[i] = ((1.0 - switch_p).ln(), switch_p.ln());
    }
    out
}

fn build_fwd_checkpoints<RefSpace>(
    checkpoints: &mut FwdCheckpoints,
    n_markers: usize,
    n_states: usize,
    panel_haps: usize,
    p_recomb: &[f32],
    seq1: &[u8],
    seq2: &[u8],
    conf: &[f32],
    inputs: HapEmissionInputs<'_>,
    ref_provider: &mut RefAlleleProvider<'_, AnyMarkerSpace, RefSpace>,
    ref_alleles_flat: Option<&[u8]>,
    pl_provider: Option<&PlProvider>,
    allele_probs: &mut Vec<f32>,
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
    let init = 1.0f32 / n_states as f32;
    let mut fwd_sum = 1.0f32;

    let mut next_block_idx = 0usize;
    let mut next_block_start = checkpoints
        .block_starts
        .get(next_block_idx)
        .copied()
        .unwrap_or(0);
    let mut pl_emit_lut = [0.0f32; 256];

    for m in 0..n_markers {
        if m > 0 {
            let r = p_recomb.get(m).copied().unwrap_or(0.0);
            let params = subset_transition_params(r, n_states, panel_haps);
            let stay_gap = params.0;
            let shift = params.1;
            let scale = stay_gap / fwd_sum.max(1e-30);

            // SIMD-optimized fwd_prior = scale * fwd + shift
            let shift_vec = f32x8::splat(shift);
            let scale_vec = f32x8::splat(scale);
            let mut k = 0;
            while k + 8 <= n_states {
                let fwd_arr: [f32; 8] = fwd[k..k + 8].try_into().unwrap();
                let fwd_chunk = f32x8::from(fwd_arr);
                let res = scale_vec * fwd_chunk + shift_vec;
                let res_arr: [f32; 8] = res.into();
                fwd_prior[k..k + 8].copy_from_slice(&res_arr);
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
        let ref_row = if let Some(flat) = ref_alleles_flat {
            let offset = m * n_states;
            &flat[offset..offset + n_states]
        } else {
            ref_provider.fill_ref_alleles(m, ref_alleles);
            &ref_alleles[..n_states]
        };

        let use_combined = matches!(mode, EmissionMode::Combined) || inputs.use_combined[m];
        let hard_match = inputs.hard_match[m];

        let pl = pl_provider.and_then(|p| p.pl(m));
        let pl_n_alleles =
            compute_pl_allele_probs(pl, use_combined, inputs.partner_allele[m], allele_probs);
        let p_no_err_pl = p_no_err;
        let p_err_pl = if let Some(n) = pl_n_alleles {
            if n > 2 {
                p_err / (n as f32 - 1.0)
            } else {
                p_err
            }
        } else {
            p_err
        };
        let has_pl = pl_n_alleles.is_some();
        if has_pl {
            build_pl_emit_lut(&mut pl_emit_lut, allele_probs, p_no_err_pl, p_err_pl);
        }

        // Compute fwd[k] = fwd_prior[k] * emit and accumulate sum
        // SIMD-optimized accumulation
        let mut sum_vec = f32x8::splat(0.0);
        let mut k = 0;

        if use_combined {
            let emit_mode = classify_combined(a1, a2);
            // Vectorized loop
            while k + 8 <= n_states {
                let prior_arr: [f32; 8] = fwd_prior[k..k + 8].try_into().unwrap();
                let prior_vec = f32x8::from(prior_arr);

                // Compute emissions for 8 states
                let emit_arr = if has_pl {
                    [
                        pl_emit_lut[ref_row[k] as usize],
                        pl_emit_lut[ref_row[k + 1] as usize],
                        pl_emit_lut[ref_row[k + 2] as usize],
                        pl_emit_lut[ref_row[k + 3] as usize],
                        pl_emit_lut[ref_row[k + 4] as usize],
                        pl_emit_lut[ref_row[k + 5] as usize],
                        pl_emit_lut[ref_row[k + 6] as usize],
                        pl_emit_lut[ref_row[k + 7] as usize],
                    ]
                } else {
                    [
                        emit_combined_fast(ref_row[k], emit_mode, conf_m, p_no_err, p_err),
                        emit_combined_fast(ref_row[k + 1], emit_mode, conf_m, p_no_err, p_err),
                        emit_combined_fast(ref_row[k + 2], emit_mode, conf_m, p_no_err, p_err),
                        emit_combined_fast(ref_row[k + 3], emit_mode, conf_m, p_no_err, p_err),
                        emit_combined_fast(ref_row[k + 4], emit_mode, conf_m, p_no_err, p_err),
                        emit_combined_fast(ref_row[k + 5], emit_mode, conf_m, p_no_err, p_err),
                        emit_combined_fast(ref_row[k + 6], emit_mode, conf_m, p_no_err, p_err),
                        emit_combined_fast(ref_row[k + 7], emit_mode, conf_m, p_no_err, p_err),
                    ]
                };
                let emit_vec = f32x8::from(emit_arr);

                let res = prior_vec * emit_vec;
                let res_arr: [f32; 8] = res.into();
                fwd[k..k + 8].copy_from_slice(&res_arr);
                sum_vec += res;
                k += 8;
            }
            // Scalar tail
            fwd_sum = sum_vec.reduce_add();
            for i in k..n_states {
                let emit = if has_pl {
                    pl_emit_lut[ref_row[i] as usize]
                } else {
                    emit_combined_fast(ref_row[i], emit_mode, conf_m, p_no_err, p_err)
                };
                fwd[i] = fwd_prior[i] * emit;
                fwd_sum += fwd[i];
            }
        } else {
            let target_al = inputs.target_constraint[m];
            if hard_match {
                fwd_sum = 0.0;
                for i in 0..n_states {
                    let emit = emit_prob_hard(ref_row[i], target_al, conf_m, p_no_err, p_err, true);
                    fwd[i] = fwd_prior[i] * emit;
                    fwd_sum += fwd[i];
                }
            } else {
                // Vectorized loop
                while k + 8 <= n_states {
                    let prior_arr: [f32; 8] = fwd_prior[k..k + 8].try_into().unwrap();
                    let prior_vec = f32x8::from(prior_arr);

                    let emit_arr = if has_pl {
                        [
                            pl_emit_lut[ref_row[k] as usize],
                            pl_emit_lut[ref_row[k + 1] as usize],
                            pl_emit_lut[ref_row[k + 2] as usize],
                            pl_emit_lut[ref_row[k + 3] as usize],
                            pl_emit_lut[ref_row[k + 4] as usize],
                            pl_emit_lut[ref_row[k + 5] as usize],
                            pl_emit_lut[ref_row[k + 6] as usize],
                            pl_emit_lut[ref_row[k + 7] as usize],
                        ]
                    } else {
                        [
                            emit_prob(ref_row[k], target_al, conf_m, p_no_err, p_err),
                            emit_prob(ref_row[k + 1], target_al, conf_m, p_no_err, p_err),
                            emit_prob(ref_row[k + 2], target_al, conf_m, p_no_err, p_err),
                            emit_prob(ref_row[k + 3], target_al, conf_m, p_no_err, p_err),
                            emit_prob(ref_row[k + 4], target_al, conf_m, p_no_err, p_err),
                            emit_prob(ref_row[k + 5], target_al, conf_m, p_no_err, p_err),
                            emit_prob(ref_row[k + 6], target_al, conf_m, p_no_err, p_err),
                            emit_prob(ref_row[k + 7], target_al, conf_m, p_no_err, p_err),
                        ]
                    };
                    let emit_vec = f32x8::from(emit_arr);

                    let res = prior_vec * emit_vec;
                    let res_arr: [f32; 8] = res.into();
                    fwd[k..k + 8].copy_from_slice(&res_arr);
                    sum_vec += res;
                    k += 8;
                }
                // Scalar tail
                fwd_sum = sum_vec.reduce_add();
                for i in k..n_states {
                    let emit = if has_pl {
                        pl_emit_lut[ref_row[i] as usize]
                    } else {
                        emit_prob(ref_row[i], target_al, conf_m, p_no_err, p_err)
                    };
                    fwd[i] = fwd_prior[i] * emit;
                    fwd_sum += fwd[i];
                }
            }
        }
        fwd_sum = fwd_sum.max(1e-30);

        if m == next_block_start {
            let dst = checkpoints.block_slice_mut(next_block_idx);
            dst.copy_from_slice(&fwd);
            next_block_idx += 1;
            next_block_start = checkpoints
                .block_starts
                .get(next_block_idx)
                .copied()
                .unwrap_or(usize::MAX);
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

fn sample_path_from_checkpoints<RefSpace>(
    path: &mut [u32],
    checkpoints: &FwdCheckpoints,
    n_markers: usize,
    n_states: usize,
    panel_haps: usize,
    p_recomb: &[f32],
    seq1: &[u8],
    seq2: &[u8],
    conf: &[f32],
    inputs: HapEmissionInputs<'_>,
    ref_provider: &mut RefAlleleProvider<'_, AnyMarkerSpace, RefSpace>,
    ref_alleles_flat: Option<&[u8]>,
    pl_provider: Option<&PlProvider>,
    p_no_err: f32,
    p_err: f32,
    rng: &mut rand::rngs::SmallRng,
    fwd_block: &mut [f32],
    weights: &mut [f32],
    ref_alleles: &mut [u8],
    allele_probs: &mut Vec<f32>,
    mode: EmissionMode,
) {
    use wide::f32x8;

    if n_markers == 0 || n_states == 0 {
        return;
    }

    let starts = checkpoints.block_starts.as_ref();
    let n_blocks = starts.len().max(1);

    let weights = &mut weights[..n_states];
    let ref_alleles = &mut ref_alleles[..n_states];
    let mut pl_emit_lut = [0.0f32; 256];

    for block_idx in (0..n_blocks).rev() {
        let start = starts.get(block_idx).copied().unwrap_or(0).min(n_markers);
        let end = starts
            .get(block_idx + 1)
            .copied()
            .unwrap_or(n_markers)
            .min(n_markers);
        if end <= start {
            continue;
        }
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
            let params = subset_transition_params(r, n_states, panel_haps);
            let stay_gap = params.0;
            let shift = params.1;
            let scale = stay_gap / prev_sum;

            let a1 = seq1[m];
            let a2 = seq2[m];
            let conf_m = conf[m];
            let row_idx = (m - start) * row_stride;
            let (prev_part, curr_part) = fwd_buf.split_at_mut(row_idx);
            let prev_row = &prev_part[row_idx - row_stride..];

            // Batch lookup ref alleles
            let ref_row = if let Some(flat) = ref_alleles_flat {
                let offset = m * n_states;
                &flat[offset..offset + n_states]
            } else {
                ref_provider.fill_ref_alleles(m, ref_alleles);
                &ref_alleles[..n_states]
            };

            // SIMD-optimized forward update
            let shift_vec = f32x8::splat(shift);
            let scale_vec = f32x8::splat(scale);
            let mut sum_vec = f32x8::splat(0.0);
            let mut k = 0;

            let use_combined = matches!(mode, EmissionMode::Combined) || inputs.use_combined[m];
            let hard_match = inputs.hard_match[m];

            let pl = pl_provider.and_then(|p| p.pl(m));
            let pl_n_alleles =
                compute_pl_allele_probs(pl, use_combined, inputs.partner_allele[m], allele_probs);
            let p_no_err_pl = p_no_err;
            let p_err_pl = if let Some(n) = pl_n_alleles {
                if n > 2 {
                    p_err / (n as f32 - 1.0)
                } else {
                    p_err
                }
            } else {
                p_err
            };
            let has_pl = pl_n_alleles.is_some();
            if has_pl {
                build_pl_emit_lut(&mut pl_emit_lut, allele_probs, p_no_err_pl, p_err_pl);
            }

            if use_combined {
                let emit_mode = classify_combined(a1, a2);
                while k + 8 <= n_states {
                    let prev_arr: [f32; 8] = prev_row[k..k + 8].try_into().unwrap();
                    let prev_vec = f32x8::from(prev_arr);
                    let prior_vec = scale_vec * prev_vec + shift_vec;

                    let emit_arr = if has_pl {
                        [
                            pl_emit_lut[ref_row[k] as usize],
                            pl_emit_lut[ref_row[k + 1] as usize],
                            pl_emit_lut[ref_row[k + 2] as usize],
                            pl_emit_lut[ref_row[k + 3] as usize],
                            pl_emit_lut[ref_row[k + 4] as usize],
                            pl_emit_lut[ref_row[k + 5] as usize],
                            pl_emit_lut[ref_row[k + 6] as usize],
                            pl_emit_lut[ref_row[k + 7] as usize],
                        ]
                    } else {
                        [
                            emit_combined_fast(ref_row[k], emit_mode, conf_m, p_no_err, p_err),
                            emit_combined_fast(ref_row[k + 1], emit_mode, conf_m, p_no_err, p_err),
                            emit_combined_fast(ref_row[k + 2], emit_mode, conf_m, p_no_err, p_err),
                            emit_combined_fast(ref_row[k + 3], emit_mode, conf_m, p_no_err, p_err),
                            emit_combined_fast(ref_row[k + 4], emit_mode, conf_m, p_no_err, p_err),
                            emit_combined_fast(ref_row[k + 5], emit_mode, conf_m, p_no_err, p_err),
                            emit_combined_fast(ref_row[k + 6], emit_mode, conf_m, p_no_err, p_err),
                            emit_combined_fast(ref_row[k + 7], emit_mode, conf_m, p_no_err, p_err),
                        ]
                    };
                    let emit_vec = f32x8::from(emit_arr);

                    let res = prior_vec * emit_vec;
                    let res_arr: [f32; 8] = res.into();
                    curr_part[k..k + 8].copy_from_slice(&res_arr);
                    sum_vec += res;
                    k += 8;
                }
                prev_sum = sum_vec.reduce_add();
                for i in k..n_states {
                    let prior = scale * prev_row[i] + shift;
                    let emit = if has_pl {
                        pl_emit_lut[ref_row[i] as usize]
                    } else {
                        emit_combined_fast(ref_row[i], emit_mode, conf_m, p_no_err, p_err)
                    };
                    curr_part[i] = prior * emit;
                    prev_sum += curr_part[i];
                }
            } else {
                let target_al = inputs.target_constraint[m];
                if hard_match {
                    prev_sum = 0.0;
                    for i in 0..n_states {
                        let prior = scale * prev_row[i] + shift;
                        let emit =
                            emit_prob_hard(ref_row[i], target_al, conf_m, p_no_err, p_err, true);
                        curr_part[i] = prior * emit;
                        prev_sum += curr_part[i];
                    }
                } else {
                    while k + 8 <= n_states {
                        let prev_arr: [f32; 8] = prev_row[k..k + 8].try_into().unwrap();
                        let prev_vec = f32x8::from(prev_arr);
                        let prior_vec = scale_vec * prev_vec + shift_vec;

                        let emit_arr = if has_pl {
                            [
                                pl_emit_lut[ref_row[k] as usize],
                                pl_emit_lut[ref_row[k + 1] as usize],
                                pl_emit_lut[ref_row[k + 2] as usize],
                                pl_emit_lut[ref_row[k + 3] as usize],
                                pl_emit_lut[ref_row[k + 4] as usize],
                                pl_emit_lut[ref_row[k + 5] as usize],
                                pl_emit_lut[ref_row[k + 6] as usize],
                                pl_emit_lut[ref_row[k + 7] as usize],
                            ]
                        } else {
                            [
                                emit_prob(ref_row[k], target_al, conf_m, p_no_err, p_err),
                                emit_prob(ref_row[k + 1], target_al, conf_m, p_no_err, p_err),
                                emit_prob(ref_row[k + 2], target_al, conf_m, p_no_err, p_err),
                                emit_prob(ref_row[k + 3], target_al, conf_m, p_no_err, p_err),
                                emit_prob(ref_row[k + 4], target_al, conf_m, p_no_err, p_err),
                                emit_prob(ref_row[k + 5], target_al, conf_m, p_no_err, p_err),
                                emit_prob(ref_row[k + 6], target_al, conf_m, p_no_err, p_err),
                                emit_prob(ref_row[k + 7], target_al, conf_m, p_no_err, p_err),
                            ]
                        };
                        let emit_vec = f32x8::from(emit_arr);

                        let res = prior_vec * emit_vec;
                        let res_arr: [f32; 8] = res.into();
                        curr_part[k..k + 8].copy_from_slice(&res_arr);
                        sum_vec += res;
                        k += 8;
                    }
                    prev_sum = sum_vec.reduce_add();
                    for i in k..n_states {
                        let prior = scale * prev_row[i] + shift;
                        let emit = if has_pl {
                            pl_emit_lut[ref_row[i] as usize]
                        } else {
                            emit_prob(ref_row[i], target_al, conf_m, p_no_err, p_err)
                        };
                        curr_part[i] = prior * emit;
                        prev_sum += curr_part[i];
                    }
                }
            }
            prev_sum = prev_sum.max(1e-30);
        }

        // Sample the last marker in this block conditional on the first state in the next block.
        // This is the explicit boundary projection that was missing from the previous checkpoint sampler.
        let next_state = if end < n_markers {
            Some(path[end] as usize)
        } else {
            None
        };
        let last_row = &fwd_buf[(block_len - 1) * row_stride..block_len * row_stride];
        if let Some(ns) = next_state {
            let r = p_recomb.get(end).copied().unwrap_or(0.0);
            let params = subset_transition_params(r, n_states, panel_haps);
            let shift = params.1;
            let stay = params.2;
            for i in 0..n_states {
                let t = if i == ns { stay } else { shift };
                weights[i] = last_row[i] * t;
            }
            let sampled = sample_from_weights(&weights, rng);
            path[end - 1] = sampled as u32;
        } else {
            let sampled = sample_from_weights(last_row, rng);
            path[end - 1] = sampled as u32;
        }

        for m in (start + 1..end).rev() {
            let next_state = path[m] as usize;
            let r = p_recomb.get(m).copied().unwrap_or(0.0);
            let params = subset_transition_params(r, n_states, panel_haps);
            let shift = params.1;
            let stay = params.2;
            let row_idx = (m - 1 - start) * row_stride;
            let prev_row = &fwd_buf[row_idx..row_idx + row_stride];

            // SIMD-optimized weight computation
            let shift_vec = f32x8::splat(shift);
            let mut k = 0;
            while k + 8 <= n_states {
                let prev_arr: [f32; 8] = prev_row[k..k + 8].try_into().unwrap();
                let prev_vec = f32x8::from(prev_arr);
                // Most states get shift transition
                let res = prev_vec * shift_vec;
                let res_arr: [f32; 8] = res.into();
                weights[k..k + 8].copy_from_slice(&res_arr);
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
    phase_conf: &[f32],
    fixed_allele: &[u8], // Allele assigned to OTHER haplotype (255 = no constraint)
    neighbors: &[u32],   // Selected neighbor haplotype indices
    phase_ibs: &BidirectionalPhaseIbs,
    p_no_err: f32,
    p_err: f32,
    rng: &mut rand::rngs::SmallRng,
    workspace: &mut crate::utils::workspace::ThreadWorkspace,
) {
    use wide::f32x8;

    if n_markers == 0
        || n_states == 0
        || neighbors.is_empty()
        || conf.len() < n_markers
        || phase_conf.len() < n_markers
        || geno_a1.len() < n_markers
        || geno_a2.len() < n_markers
        || fixed_allele.len() < n_markers
    {
        return;
    }

    let actual_n_states = neighbors.len().min(n_states);

    workspace.ensure_ffbs(n_markers, actual_n_states);
    let fwd_curr = &mut workspace.ffbs_fwd_curr;
    let fwd_prev = &mut workspace.ffbs_fwd_prev;
    let fwd_at_marker = &mut workspace.ffbs_fwd_at_marker;
    let weights = &mut workspace.ffbs_weights;
    let mut neighbor_alleles = vec![255u8; actual_n_states];
    fwd_curr[..actual_n_states].fill(0.0);
    fwd_prev[..actual_n_states].fill(0.0);

    // Initialize at marker 0
    let init = 1.0f32 / actual_n_states as f32;
    let conf0 = constrained_emission_confidence(
        geno_a1[0],
        geno_a2[0],
        fixed_allele[0],
        conf[0],
        phase_conf[0],
    );
    phase_ibs.fill_alleles_for_haps(0, &neighbors[..actual_n_states], &mut neighbor_alleles);
    for k in 0..actual_n_states {
        let ref_al = neighbor_alleles[k];
        let emit = emit_haploid_constrained(
            ref_al,
            geno_a1[0],
            geno_a2[0],
            fixed_allele[0],
            conf0,
            p_no_err,
            p_err,
        );
        fwd_curr[k] = init * emit;
    }
    let mut fwd_sum: f32 = fwd_curr.iter().sum();
    fwd_sum = fwd_sum.max(1e-30);
    fwd_at_marker[0..actual_n_states].copy_from_slice(&fwd_curr[..actual_n_states]);

    // Forward pass
    let panel_haps = phase_ibs.n_haps().max(actual_n_states);
    for m in 1..n_markers {
        std::mem::swap(fwd_prev, fwd_curr);

        let r = p_recomb.get(m).copied().unwrap_or(0.0);
        let params = subset_transition_params(r, actual_n_states, panel_haps);
        let stay_gap = params.0;
        let shift = params.1;
        let scale = stay_gap / fwd_sum;
        let conf_m = constrained_emission_confidence(
            geno_a1[m],
            geno_a2[m],
            fixed_allele[m],
            conf[m],
            phase_conf[m],
        );
        phase_ibs.fill_alleles_for_haps(m, &neighbors[..actual_n_states], &mut neighbor_alleles);

        // SIMD-optimized transition + emission
        let shift_vec = f32x8::splat(shift);
        let scale_vec = f32x8::splat(scale);
        let mut sum_vec = f32x8::splat(0.0);
        let mut k = 0;

        while k + 8 <= actual_n_states {
            let prev_arr: [f32; 8] = fwd_prev[k..k + 8].try_into().unwrap();
            let prev_vec = f32x8::from(prev_arr);
            let prior_vec = scale_vec * prev_vec + shift_vec;

            // Compute emissions
            let emit_arr = [
                emit_haploid_constrained(
                    neighbor_alleles[k],
                    geno_a1[m],
                    geno_a2[m],
                    fixed_allele[m],
                    conf_m,
                    p_no_err,
                    p_err,
                ),
                emit_haploid_constrained(
                    neighbor_alleles[k + 1],
                    geno_a1[m],
                    geno_a2[m],
                    fixed_allele[m],
                    conf_m,
                    p_no_err,
                    p_err,
                ),
                emit_haploid_constrained(
                    neighbor_alleles[k + 2],
                    geno_a1[m],
                    geno_a2[m],
                    fixed_allele[m],
                    conf_m,
                    p_no_err,
                    p_err,
                ),
                emit_haploid_constrained(
                    neighbor_alleles[k + 3],
                    geno_a1[m],
                    geno_a2[m],
                    fixed_allele[m],
                    conf_m,
                    p_no_err,
                    p_err,
                ),
                emit_haploid_constrained(
                    neighbor_alleles[k + 4],
                    geno_a1[m],
                    geno_a2[m],
                    fixed_allele[m],
                    conf_m,
                    p_no_err,
                    p_err,
                ),
                emit_haploid_constrained(
                    neighbor_alleles[k + 5],
                    geno_a1[m],
                    geno_a2[m],
                    fixed_allele[m],
                    conf_m,
                    p_no_err,
                    p_err,
                ),
                emit_haploid_constrained(
                    neighbor_alleles[k + 6],
                    geno_a1[m],
                    geno_a2[m],
                    fixed_allele[m],
                    conf_m,
                    p_no_err,
                    p_err,
                ),
                emit_haploid_constrained(
                    neighbor_alleles[k + 7],
                    geno_a1[m],
                    geno_a2[m],
                    fixed_allele[m],
                    conf_m,
                    p_no_err,
                    p_err,
                ),
            ];
            let emit_vec = f32x8::from(emit_arr);

            let res = prior_vec * emit_vec;
            let res_arr: [f32; 8] = res.into();
            fwd_curr[k..k + 8].copy_from_slice(&res_arr);
            sum_vec += res;
            k += 8;
        }

        // Scalar tail
        fwd_sum = sum_vec.reduce_add();
        for i in k..actual_n_states {
            let prior = scale * fwd_prev[i] + shift;
            let emit = emit_haploid_constrained(
                neighbor_alleles[i],
                geno_a1[m],
                geno_a2[m],
                fixed_allele[m],
                conf_m,
                p_no_err,
                p_err,
            );
            fwd_curr[i] = prior * emit;
            fwd_sum += fwd_curr[i];
        }
        fwd_sum = fwd_sum.max(1e-30);

        let start = m * actual_n_states;
        fwd_at_marker[start..start + actual_n_states].copy_from_slice(&fwd_curr[..actual_n_states]);
    }

    // Backward sampling
    let last_start = (n_markers - 1) * actual_n_states;
    let last_fwd = &fwd_at_marker[last_start..last_start + actual_n_states];
    path[n_markers - 1] = sample_from_weights(last_fwd, rng) as u32;
    weights[..actual_n_states].fill(0.0);
    for m in (1..n_markers).rev() {
        let next_state = path[m] as usize;
        let r = p_recomb.get(m).copied().unwrap_or(0.0);
        let params = subset_transition_params(r, actual_n_states, panel_haps);
        let shift = params.1;
        let stay = params.2;

        let prev_start = (m - 1) * actual_n_states;
        let prev_fwd = &fwd_at_marker[prev_start..prev_start + actual_n_states];

        for k in 0..actual_n_states {
            weights[k] = prev_fwd[k] * shift;
        }
        if next_state < actual_n_states {
            weights[next_state] = prev_fwd[next_state] * stay;
        }

        path[m - 1] = sample_from_weights(&weights[..actual_n_states], rng) as u32;
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
    phase_conf: &[f32],
    phase_ibs: &BidirectionalPhaseIbs,
    ibs2: &Ibs2,
    hi_freq_to_orig: &[usize],
    sample_idx: u32,
    sample_phase_stability: &[f32],
    het_positions: &[usize],
    seed: u64,
    n_mcmc_steps: usize,
    p_no_err: f32,
    p_err: f32,
    initial_paths: Option<&MosaicPaths>,
    anchor_hap1: Option<&[u8]>,
    anchor_hap2: Option<&[u8]>,
    telemetry: Option<&crate::utils::telemetry::TelemetryBlackboard>,
    workspace: &mut crate::utils::workspace::ThreadWorkspace,
) -> (Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>, MosaicPaths) {
    use rand::SeedableRng;

    #[derive(Clone, Copy)]
    struct LocalMarkerIdx(usize);

    #[inline]
    fn adaptive_dynamic_state_target(
        max_states: usize,
        n_haps: u32,
        local_phase_conf: f32,
        sample_stability: f32,
    ) -> usize {
        if max_states == 0 || n_haps == 0 {
            return 0;
        }
        let cap = max_states.min(n_haps.saturating_sub(2) as usize).max(1);
        let min_states = (cap / 3).max(PBWT_ADAPTIVE_K_FLOOR).min(cap);
        if min_states >= cap {
            return cap;
        }
        let local_uncertainty = 1.0 - local_phase_conf.clamp(0.0, 1.0);
        let sample_uncertainty = 1.0 - sample_stability.clamp(0.0, 1.0);
        let uncertainty = (0.7 * local_uncertainty + 0.3 * sample_uncertainty).clamp(0.0, 1.0);
        let span = (cap - min_states) as f32;
        (min_states as f32 + span * uncertainty)
            .round()
            .clamp(min_states as f32, cap as f32) as usize
    }

    if het_positions.is_empty()
        || n_markers == 0
        || n_states == 0
        || phase_conf.len() != n_markers
        || hi_freq_to_orig.len() != n_markers
    {
        return (
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            MosaicPaths {
                path1: Vec::new(),
                path2: Vec::new(),
            },
        );
    }
    if (sample_idx as usize) >= sample_phase_stability.len() {
        return (
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            MosaicPaths {
                path1: Vec::new(),
                path2: Vec::new(),
            },
        );
    }

    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    let hap1_idx = sample_idx * 2;
    let anchor_h1 = anchor_hap1.unwrap_or(&[]);
    let anchor_h2 = anchor_hap2.unwrap_or(&[]);
    let has_anchor = (0..n_markers).any(|m| {
        let a1 = anchor_h1.get(m).copied().unwrap_or(255);
        let a2 = anchor_h2.get(m).copied().unwrap_or(255);
        a1 != 255 || a2 != 255
    });
    let recipient_sample = SampleIdx::new(sample_idx);
    let recipient_stability = sample_phase_stability[sample_idx as usize].clamp(0.0, 1.0);
    let mut ibs2_by_other: HashMap<u32, Vec<(usize, usize)>> = HashMap::new();
    for seg in ibs2.segments(recipient_sample) {
        ibs2_by_other
            .entry(seg.other_sample.0)
            .or_default()
            .push((seg.start, seg.incl_end));
    }
    const IBD2_DIRECTION_MARGIN: f32 = 0.05;
    const IBD2_HIGH_CONF: f32 = 0.985;

    let allow_donor_at_marker = |donor_hap: u32, marker_idx: LocalMarkerIdx| -> bool {
        if marker_idx.0 >= n_markers {
            return false;
        }
        let donor_sample = donor_hap / 2;
        if donor_sample == sample_idx {
            return false;
        }
        let Some(global_m) = hi_freq_to_orig.get(marker_idx.0).copied() else {
            return false;
        };
        let Some(segments) = ibs2_by_other.get(&donor_sample) else {
            return true;
        };
        let mut in_ibs2 = false;
        for &(start, incl_end) in segments {
            if global_m >= start && global_m <= incl_end {
                in_ibs2 = true;
                break;
            }
        }
        if !in_ibs2 {
            return true;
        }
        let donor_stability = sample_phase_stability
            .get(donor_sample as usize)
            .copied()
            .unwrap_or(0.5)
            .clamp(0.0, 1.0);
        let local_recipient = phase_conf[marker_idx.0].clamp(0.0, 1.0);
        if donor_stability >= IBD2_HIGH_CONF && local_recipient >= IBD2_HIGH_CONF {
            return true;
        }
        donor_stability >= recipient_stability.max(local_recipient) + IBD2_DIRECTION_MARGIN
    };

    // Initialize H1, H2 alleles from genotype (random phase at hets)
    let mut h1_alleles = vec![0u8; n_markers];
    let mut h2_alleles = vec![0u8; n_markers];
    for m in 0..n_markers {
        let a1 = seq1[m];
        let a2 = seq2[m];
        let anchor_a1 = anchor_h1.get(m).copied().unwrap_or(255);
        let anchor_a2 = anchor_h2.get(m).copied().unwrap_or(255);
        if anchor_a1 != 255 || anchor_a2 != 255 {
            h1_alleles[m] = anchor_a1;
            h2_alleles[m] = anchor_a2;
            continue;
        }
        if a1 == 255 && a2 == 255 {
            h1_alleles[m] = 255;
            h2_alleles[m] = 255;
        } else if a1 == a2 {
            h1_alleles[m] = a1;
            h2_alleles[m] = a1;
        } else {
            // Het: if unanchored, keep input orientation to avoid label drift.
            if has_anchor {
                if rng.random::<bool>() {
                    h1_alleles[m] = a1;
                    h2_alleles[m] = a2;
                } else {
                    h1_alleles[m] = a2;
                    h2_alleles[m] = a1;
                }
            } else {
                h1_alleles[m] = a1;
                h2_alleles[m] = a2;
            }
        }
    }

    // Seed alleles from initial paths if available (from heuristic)
    // This ensures MCMC starts in a high-probability region rather than drifting
    // from a random start.
    if let Some(paths) = initial_paths {
        if paths.path1.len() == n_markers && paths.path2.len() == n_markers {
            let mut pair_haps = [0u32; 2];
            let mut pair_alleles = [255u8; 2];
            for m in 0..n_markers {
                let a1 = seq1[m];
                let a2 = seq2[m];
                if a1 == 255 || a2 == 255 || a1 == a2 {
                    continue;
                }

                let h1_idx = paths.path1[m] as usize;
                let h2_idx = paths.path2[m] as usize;

                if h1_idx < phase_ibs.n_haps() && h2_idx < phase_ibs.n_haps() {
                    pair_haps[0] = h1_idx as u32;
                    pair_haps[1] = h2_idx as u32;
                    phase_ibs.fill_alleles_for_haps(m, &pair_haps, &mut pair_alleles);
                    let ref1 = pair_alleles[0];
                    let ref2 = pair_alleles[1];

                    let matches_orient1 = ref1 == a1 && ref2 == a2;
                    let matches_orient2 = ref1 == a2 && ref2 == a1;

                    if matches_orient1 && !matches_orient2 {
                        h1_alleles[m] = a1;
                        h2_alleles[m] = a2;
                    } else if matches_orient2 && !matches_orient1 {
                        h1_alleles[m] = a2;
                        h2_alleles[m] = a1;
                    }
                }
            }
        }
    }

    // Initialize path with starting states from standard neighbor finding
    // This gives the first iteration something to work with
    let center_init = n_markers / 2;
    let initial_state_target = adaptive_dynamic_state_target(
        n_states,
        phase_ibs.n_haps() as u32,
        phase_conf[center_init],
        recipient_stability,
    );
    let mut initial_neighbors =
        phase_ibs.find_neighbors(hap1_idx, center_init, ibs2, initial_state_target);
    if !initial_neighbors.is_empty() {
        let mut filtered = Vec::with_capacity(initial_neighbors.len());
        for &h in &initial_neighbors {
            if allow_donor_at_marker(h, LocalMarkerIdx(center_init)) {
                filtered.push(h);
            }
        }
        if filtered.is_empty() {
            initial_neighbors.sort_unstable_by(|a, b| {
                let a_score = sample_phase_stability
                    .get((a / 2) as usize)
                    .copied()
                    .unwrap_or(0.5);
                let b_score = sample_phase_stability
                    .get((b / 2) as usize)
                    .copied()
                    .unwrap_or(0.5);
                b_score
                    .partial_cmp(&a_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            filtered.extend(
                initial_neighbors
                    .iter()
                    .take(initial_state_target.max(1))
                    .copied(),
            );
        }
        initial_neighbors = filtered;
    }
    if initial_neighbors.is_empty() {
        return (
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            MosaicPaths {
                path1: Vec::new(),
                path2: Vec::new(),
            },
        );
    }

    // Separate paths for H1 and H2 to avoid cross-talk in Gibbs sampling
    // Store reference hap IDs (for persistence) and local state indices (per step)
    let mut path1_ref = vec![0u32; n_markers];
    let mut path2_ref = vec![0u32; n_markers];
    let mut path1_idx = vec![0u32; n_markers];
    let mut path2_idx = vec![0u32; n_markers];
    let mut fixed_allele = vec![255u8; n_markers];
    let mut het_index = vec![usize::MAX; n_markers];
    for (i, &m) in het_positions.iter().enumerate() {
        if m < n_markers {
            het_index[m] = i;
        }
    }
    let mut swap_counts = vec![0f32; het_positions.len()];
    let mut swap_obs = vec![0f32; het_positions.len()];

    // Current set of neighbors (reused across markers within an MCMC step)
    let mut neighbors = initial_neighbors;
    let n_haps = phase_ibs.n_haps() as u32;

    let mut seeded_from_heuristic = false;
    if initial_paths.is_none() && !neighbors.is_empty() {
        let limit = neighbors.len().min(16);
        if limit >= 2 {
            // Keep this seed-search bounded on long windows.
            const MAX_INIT_EVAL_MARKERS: usize = 2000;
            let marker_stride = n_markers
                .saturating_add(MAX_INIT_EVAL_MARKERS - 1)
                .checked_div(MAX_INIT_EVAL_MARKERS)
                .unwrap_or(1)
                .max(1);
            let mut scores = vec![0.0f32; limit * limit];
            let mut neighbor_alleles = vec![255u8; limit];
            let mut informative = 0usize;
            for m in (0..n_markers).step_by(marker_stride) {
                let a1 = seq1[m];
                let a2 = seq2[m];
                if a1 == 255 && a2 == 255 {
                    continue;
                }
                informative += 1;
                let conf_m = conf[m].clamp(0.0, 1.0);
                let is_het = a1 != a2 && a1 != 255 && a2 != 255;
                phase_ibs.fill_alleles_for_haps(m, &neighbors[..limit], &mut neighbor_alleles);
                for i in 0..limit {
                    let r1 = neighbor_alleles[i];
                    for j in 0..i {
                        let r2 = neighbor_alleles[j];
                        let prob = if is_het {
                            let keep = emit_prob(r1, a1, conf_m, p_no_err, p_err)
                                * emit_prob(r2, a2, conf_m, p_no_err, p_err);
                            let swap = emit_prob(r1, a2, conf_m, p_no_err, p_err)
                                * emit_prob(r2, a1, conf_m, p_no_err, p_err);
                            0.5 * (keep + swap)
                        } else {
                            let obs = if a1 != 255 { a1 } else { a2 };
                            emit_prob(r1, obs, conf_m, p_no_err, p_err)
                                * emit_prob(r2, obs, conf_m, p_no_err, p_err)
                        };
                        scores[i * limit + j] += prob.max(1e-30).ln();
                    }
                }
            }

            if informative > 0 {
                let mut best_score = f32::NEG_INFINITY;
                let mut best_pair = (0usize, 1usize);
                for i in 0..limit {
                    for j in 0..i {
                        let s = scores[i * limit + j];
                        if s > best_score {
                            best_score = s;
                            best_pair = (i, j);
                        }
                    }
                }

                let h1_best = neighbors[best_pair.0];
                let h2_best = neighbors[best_pair.1];
                path1_ref.fill(h1_best);
                path2_ref.fill(h2_best);
                let mut pair_haps = [h1_best, h2_best];
                let mut pair_alleles = [255u8; 2];
                for m in 0..n_markers {
                    let a1 = seq1[m];
                    let a2 = seq2[m];
                    if a1 == 255 || a2 == 255 || a1 == a2 {
                        continue;
                    }
                    pair_haps[0] = h1_best;
                    pair_haps[1] = h2_best;
                    phase_ibs.fill_alleles_for_haps(m, &pair_haps, &mut pair_alleles);
                    let r1 = pair_alleles[0];
                    let r2 = pair_alleles[1];
                    let m1 = r1 == a1 && r2 == a2;
                    let m2 = r1 == a2 && r2 == a1;
                    if m1 && !m2 {
                        h1_alleles[m] = a1;
                        h2_alleles[m] = a2;
                    } else if m2 && !m1 {
                        h1_alleles[m] = a2;
                        h2_alleles[m] = a1;
                    }
                }
                seeded_from_heuristic = true;
            }
        }
    }

    if let Some(paths) = initial_paths {
        if paths.path1.len() == n_markers && paths.path2.len() == n_markers {
            path1_ref.copy_from_slice(&paths.path1);
            path2_ref.copy_from_slice(&paths.path2);
        }
    } else if !seeded_from_heuristic {
        if neighbors.len() >= 2 {
            let h1_seed = neighbors[0];
            let h2_seed = neighbors[1];
            path1_ref.fill(h1_seed);
            path2_ref.fill(h2_seed);
            let mut pair_haps = [h1_seed, h2_seed];
            let mut pair_alleles = [255u8; 2];
            for m in 0..n_markers {
                let a1 = seq1[m];
                let a2 = seq2[m];
                if a1 == 255 || a2 == 255 || a1 == a2 {
                    continue;
                }
                pair_haps[0] = h1_seed;
                pair_haps[1] = h2_seed;
                phase_ibs.fill_alleles_for_haps(m, &pair_haps, &mut pair_alleles);
                let r1 = pair_alleles[0];
                let r2 = pair_alleles[1];
                let m1 = r1 == a1 && r2 == a2;
                let m2 = r1 == a2 && r2 == a1;
                if m1 && !m2 {
                    h1_alleles[m] = a1;
                    h2_alleles[m] = a2;
                } else if m2 && !m1 {
                    h1_alleles[m] = a2;
                    h2_alleles[m] = a1;
                }
            }
        } else if let Some(&seed_hap) = neighbors.first() {
            path1_ref.fill(seed_hap);
            path2_ref.fill(seed_hap);
        }
    }

    fn mix_neighbors(
        neighbors: &mut Vec<u32>,
        n_states: usize,
        n_haps: u32,
        hap1_idx: u32,
        sample_uncertainty: f32,
        rng: &mut impl rand::Rng,
    ) {
        if n_haps == 0 {
            neighbors.clear();
            return;
        }

        // If there are no other haplotypes to sample from, fall back to self so we can proceed.
        // This avoids an infinite loop for single-sample or haploid inputs.
        if n_haps <= 2 {
            neighbors.clear();
            let self_hap = hap1_idx.min(n_haps.saturating_sub(1));
            neighbors.push(self_hap);
            return;
        }

        let target = n_states.min((n_haps.saturating_sub(2)) as usize).max(1);
        let mut seen: HashSet<u32> = HashSet::with_capacity(target.saturating_mul(2).max(8));
        neighbors.retain(|&h| {
            h != hap1_idx && h != hap1_idx + 1 && (h as usize) < n_haps as usize && seen.insert(h)
        });
        if neighbors.len() > target {
            neighbors.truncate(target);
            seen.clear();
            for &h in neighbors.iter() {
                seen.insert(h);
            }
        }

        while neighbors.len() < target {
            let h = rng.random_range(0..n_haps);
            if h == hap1_idx || h == hap1_idx + 1 {
                continue;
            }
            if seen.insert(h) {
                neighbors.push(h);
            }
        }

        if neighbors.is_empty() {
            return;
        }

        let mix_count = if sample_uncertainty >= 0.66 {
            (target / 5).max(6).min(target)
        } else if sample_uncertainty >= 0.33 {
            (target / 8).max(4).min(target)
        } else {
            (target / 16).max(2).min(target)
        };
        for _ in 0..mix_count {
            let h = rng.random_range(0..n_haps);
            if h == hap1_idx || h == hap1_idx + 1 {
                continue;
            }
            if !seen.insert(h) {
                continue;
            }
            let replace_idx = rng.random_range(0..neighbors.len());
            let old = neighbors[replace_idx];
            seen.remove(&old);
            neighbors[replace_idx] = h;
        }
    }

    fn refill_neighbors_for_marker(
        neighbors: &mut Vec<u32>,
        n_states: usize,
        n_haps: u32,
        hap1_idx: u32,
        marker_idx: LocalMarkerIdx,
        sample_uncertainty: f32,
        rng: &mut impl rand::Rng,
        allow_donor_at_marker: &impl Fn(u32, LocalMarkerIdx) -> bool,
    ) {
        if n_haps <= 2 {
            return;
        }
        let target = n_states.min((n_haps.saturating_sub(2)) as usize).max(1);
        let mut seen: HashSet<u32> = HashSet::with_capacity(target.saturating_mul(2).max(8));
        neighbors.retain(|&h| {
            h != hap1_idx
                && h != hap1_idx + 1
                && (h as usize) < n_haps as usize
                && allow_donor_at_marker(h, marker_idx)
                && seen.insert(h)
        });
        if neighbors.len() >= target {
            neighbors.truncate(target);
            return;
        }

        let max_attempts = if sample_uncertainty >= 0.66 {
            target.saturating_mul(32).max(64)
        } else if sample_uncertainty >= 0.33 {
            target.saturating_mul(24).max(48)
        } else {
            target.saturating_mul(16).max(32)
        };
        let mut attempts = 0usize;
        while neighbors.len() < target && attempts < max_attempts {
            attempts += 1;
            let h = rng.random_range(0..n_haps);
            if h == hap1_idx || h == hap1_idx + 1 {
                continue;
            }
            if !allow_donor_at_marker(h, marker_idx) {
                continue;
            }
            if seen.insert(h) {
                neighbors.push(h);
            }
        }
    }

    let record_neighbors = |neighbors: &Vec<u32>| {
        if let Some(bb) = telemetry {
            bb.add_dyn_neighbors(neighbors.len() as u64);
        }
    };

    record_neighbors(&neighbors);

    let sample_uncertainty = 1.0f32 - recipient_stability;
    let anchor_bins = (6.0 + 10.0 * sample_uncertainty).round() as usize;
    let stride = (n_markers / anchor_bins.max(1)).max(1);
    let mut anchors_static: Vec<usize> = Vec::new();
    // Prefer anchors with highest expected information under current emissions.
    let mut start = 0usize;
    while start < n_markers {
        let end = (start + stride).min(n_markers);
        let mut best_m = start;
        let mut best_score = f32::NEG_INFINITY;
        for m in start..end {
            let a1 = seq1[m];
            let a2 = seq2[m];
            if a1 == 255 && a2 == 255 {
                continue;
            }
            let conf_score = conf[m].clamp(0.0, 1.0);
            let emit_match = (p_no_err * conf_score + 0.5 * (1.0 - conf_score)).max(1e-30);
            let emit_mismatch = (p_err * conf_score + 0.5 * (1.0 - conf_score)).max(1e-30);
            let score = (emit_match / emit_mismatch).ln().abs();
            if score > best_score {
                best_score = score;
                best_m = m;
            }
        }
        anchors_static.push(best_m);
        start = end;
    }
    if anchors_static.last().copied() != Some(n_markers.saturating_sub(1)) {
        anchors_static.push(n_markers.saturating_sub(1));
    }
    anchors_static.sort_unstable();
    anchors_static.dedup();

    let base_radius = (n_markers / 40).clamp(12, 192);
    let radius = ((base_radius as f32) * (1.0 + 1.5 * sample_uncertainty)).round() as usize;
    let radius = radius.clamp(12, 320);
    let score_step = if sample_uncertainty >= 0.66 {
        1usize
    } else if sample_uncertainty >= 0.33 {
        2usize
    } else {
        3usize
    };
    let mut score_markers_static: Vec<usize> = Vec::new();
    for &anchor in &anchors_static {
        let start = anchor.saturating_sub(radius);
        let end = (anchor + radius + 1).min(n_markers);
        for m in start..end {
            // Light subsampling in dense windows for speed; always keep anchor.
            if m == anchor || ((m - start) % score_step == 0) {
                score_markers_static.push(m);
            }
        }
    }
    score_markers_static.sort_unstable();
    score_markers_static.dedup();
    let mut score_markers_sparse: Vec<usize> =
        Vec::with_capacity(score_markers_static.len() / 2 + 1);
    for (i, &m) in score_markers_static.iter().enumerate() {
        if i % 2 == 0 {
            score_markers_sparse.push(m);
        }
    }
    if score_markers_sparse.is_empty() && !score_markers_static.is_empty() {
        score_markers_sparse.push(score_markers_static[0]);
    }

    // Cache per-marker log emissions so donor scoring avoids repeated floating-point transforms.
    let mut ll_match: Vec<f32> = vec![0.0; n_markers];
    let mut ll_mismatch: Vec<f32> = vec![0.0; n_markers];
    for m in 0..n_markers {
        let conf_m = conf[m].clamp(0.0, 1.0);
        let emit_match = (p_no_err * conf_m + 0.5 * (1.0 - conf_m)).max(1e-30);
        let emit_mismatch = (p_err * conf_m + 0.5 * (1.0 - conf_m)).max(1e-30);
        ll_match[m] = emit_match.ln();
        ll_mismatch[m] = emit_mismatch.ln();
    }

    let mut candidates_buf: Vec<u32> = Vec::new();
    let mut scored_buf: Vec<(u32, f32)> = Vec::new();
    let mut donor_scores_buf: Vec<f32> = Vec::new();
    let mut donor_alleles_buf: Vec<u8> = Vec::new();
    let mut seen_buf: std::collections::HashSet<u32> = std::collections::HashSet::new();
    let mut collect_dynamic_neighbors =
        |path_ref: &[u32],
         query_hap: &[u8],
         target_states: usize,
         mcmc_step: usize,
         out: &mut Vec<u32>| {
            seen_buf.clear();
            candidates_buf.clear();
            let burnin_steps = (n_mcmc_steps / 2).max(1);
            let inject_target = if mcmc_step < burnin_steps {
                target_states.saturating_div(2).max(2)
            } else {
                target_states.saturating_div(8).max(1)
            };

            for &m in &anchors_static {
                let ref_hap = path_ref.get(m).copied().unwrap_or(0);
                if (ref_hap as usize) < phase_ibs.n_haps() {
                    let search_states = if recipient_stability < 0.95 {
                        target_states
                            .saturating_add((target_states / 2).max(1))
                            .min(phase_ibs.n_haps().saturating_sub(2).max(1))
                    } else {
                        target_states
                    };
                    let mut local =
                        phase_ibs.find_neighbors_of_state(ref_hap, m, sample_idx, search_states);
                    local.push(ref_hap);
                    for h in local {
                        if h == hap1_idx || h == hap1_idx + 1 {
                            continue;
                        }
                        if !allow_donor_at_marker(h, LocalMarkerIdx(m)) {
                            continue;
                        }
                        if seen_buf.insert(h) {
                            candidates_buf.push(h);
                        }
                    }
                }
                // Inject target-driven neighbors so the state-space can recover
                // from poor latent trajectories during burn-in/mixing.
                let mut genotype_neighbors =
                    phase_ibs.find_neighbors(hap1_idx, m, ibs2, inject_target);
                for h in genotype_neighbors.drain(..) {
                    if h == hap1_idx || h == hap1_idx + 1 {
                        continue;
                    }
                    if !allow_donor_at_marker(h, LocalMarkerIdx(m)) {
                        continue;
                    }
                    if seen_buf.insert(h) {
                        candidates_buf.push(h);
                    }
                }
            }

            if candidates_buf.is_empty() {
                out.clear();
                return;
            }

            // Phase-conditioned scoring:
            // score each donor haplotype by local log-likelihood under current sampled
            // haplotype sequence (query_hap), using cached confidence-weighted mismatch terms.
            scored_buf.clear();
            scored_buf.reserve(candidates_buf.len().saturating_sub(scored_buf.len()));
            donor_scores_buf.clear();
            donor_scores_buf.resize(candidates_buf.len(), 0.0);
            donor_alleles_buf.clear();
            donor_alleles_buf.resize(candidates_buf.len(), 255);
            let score_markers = if recipient_stability >= 0.95 && target_states <= n_states / 2 {
                &score_markers_sparse
            } else {
                &score_markers_static
            };
            for &m in score_markers {
                let obs = query_hap[m];
                if obs == 255 {
                    continue;
                }
                phase_ibs.fill_alleles_for_haps(m, &candidates_buf, &mut donor_alleles_buf);
                for i in 0..candidates_buf.len() {
                    let ref_al = donor_alleles_buf[i];
                    if ref_al == 255 {
                        continue;
                    }
                    donor_scores_buf[i] += if ref_al == obs {
                        ll_match[m]
                    } else {
                        ll_mismatch[m]
                    };
                }
            }
            for (i, &h) in candidates_buf.iter().enumerate() {
                scored_buf.push((h, donor_scores_buf[i]));
            }

            let keep = target_states.min(scored_buf.len()).max(1);
            if scored_buf.len() > keep {
                scored_buf.select_nth_unstable_by(keep - 1, |a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                scored_buf[..keep].sort_unstable_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                out.clear();
                out.reserve(keep.saturating_sub(out.capacity()));
                for (h, _) in scored_buf.iter().take(keep) {
                    out.push(*h);
                }
            } else {
                out.clear();
                out.reserve(scored_buf.len().saturating_sub(out.capacity()));
                for (h, _) in &scored_buf {
                    out.push(*h);
                }
            }
        };

    // MCMC loop: Gibbs sampling alternating between H1 and H2
    for step in 0..n_mcmc_steps {
        let prev_path1_ref = path1_ref.clone();
        let prev_path2_ref = path2_ref.clone();

        // === Sample H1 | (G, H2_fixed) ===

        // 1. Select neighbors with phase-conditioned scoring:
        //    candidates come from latent-state PBWT neighborhoods, then we rank
        //    by local likelihood under the current sampled H1 allele sequence.
        let center_marker = if n_mcmc_steps > 1 {
            n_markers / 4 + step * n_markers / (2 * n_mcmc_steps)
        } else {
            n_markers / 2
        };
        let state_target = adaptive_dynamic_state_target(
            n_states,
            n_haps,
            phase_conf[center_marker],
            recipient_stability,
        );
        collect_dynamic_neighbors(&path1_ref, &h1_alleles, state_target, step, &mut neighbors);
        let ref_hap = path1_ref.get(center_marker).copied().unwrap_or(0);
        if (ref_hap as usize) < phase_ibs.n_haps()
            && allow_donor_at_marker(ref_hap, LocalMarkerIdx(center_marker))
            && !neighbors.contains(&ref_hap)
        {
            neighbors.push(ref_hap);
        }
        if neighbors.is_empty() {
            continue;
        }
        mix_neighbors(
            &mut neighbors,
            state_target,
            n_haps,
            hap1_idx,
            sample_uncertainty,
            &mut rng,
        );
        neighbors.retain(|&h| allow_donor_at_marker(h, LocalMarkerIdx(center_marker)));
        if neighbors.is_empty() {
            refill_neighbors_for_marker(
                &mut neighbors,
                state_target,
                n_haps,
                hap1_idx,
                LocalMarkerIdx(center_marker),
                sample_uncertainty,
                &mut rng,
                &allow_donor_at_marker,
            );
            if neighbors.is_empty() {
                continue;
            }
        }
        record_neighbors(&neighbors);

        // 2. Build constraint: at hets, H1 must produce genotype with H2
        for m in 0..n_markers {
            let a1 = seq1[m];
            let a2 = seq2[m];
            let anchor_a1 = anchor_h1.get(m).copied().unwrap_or(255);
            let anchor_a2 = anchor_h2.get(m).copied().unwrap_or(255);
            let is_anchor = anchor_a1 != 255 || anchor_a2 != 255;
            if a1 == 255 || a2 == 255 || a1 == a2 {
                fixed_allele[m] = 255; // No constraint (hom/missing)
            } else if is_anchor {
                fixed_allele[m] = anchor_a2;
            } else {
                fixed_allele[m] = h2_alleles[m];
            }
        }

        // 3. Run haploid FFBS for H1
        ffbs_haploid_constrained(
            &mut path1_idx,
            n_markers,
            neighbors.len(),
            p_recomb,
            seq1,
            seq2,
            conf,
            phase_conf,
            &fixed_allele,
            &neighbors,
            phase_ibs,
            p_no_err,
            p_err,
            &mut rng,
            workspace,
        );

        // Refresh the latent reference path at all markers for the next iteration.
        refresh_path_ref_from_states(&mut path1_ref, &path1_idx, &neighbors);

        // 4. Update H1 based on sampled reference alleles at hets
        //    GIBBS SAMPLING: only update H1, leave H2 fixed
        //    At hets, set H1 to match the reference's allele (if compatible).
        for m in 0..n_markers {
            let state = path1_idx[m] as usize;
            let a1 = seq1[m];
            let a2 = seq2[m];
            let anchor_a1 = anchor_h1.get(m).copied().unwrap_or(255);
            let anchor_a2 = anchor_h2.get(m).copied().unwrap_or(255);
            let is_anchor = anchor_a1 != 255 || anchor_a2 != 255;
            if is_anchor {
                h1_alleles[m] = anchor_a1;
                h2_alleles[m] = anchor_a2;
                continue;
            }

            if a1 == 255 && a2 == 255 {
                h1_alleles[m] = 255;
            } else if a1 == a2 {
                h1_alleles[m] = a1;
            } else if state < neighbors.len() {
                // Het: use reference allele to determine H1
                let ref_hap = neighbors[state];
                let ref_al = phase_ibs.allele(m, ref_hap);
                if ref_al == a1 || ref_al == a2 {
                    // Set H1 to ref_al, and H2 must be the other allele
                    h1_alleles[m] = ref_al;
                    h2_alleles[m] = if ref_al == a1 { a2 } else { a1 };
                }
                // If ref_al is missing/different, keep current phase
            }
        }

        // === Sample H2 | (G, H1_new) ===

        // 1. Select neighbors for H2 with H2-conditioned scoring (not H1's sequence).
        collect_dynamic_neighbors(&path2_ref, &h2_alleles, state_target, step, &mut neighbors);
        let ref_hap = path2_ref.get(center_marker).copied().unwrap_or(0);
        if (ref_hap as usize) < phase_ibs.n_haps()
            && allow_donor_at_marker(ref_hap, LocalMarkerIdx(center_marker))
            && !neighbors.contains(&ref_hap)
        {
            neighbors.push(ref_hap);
        }
        if neighbors.is_empty() {
            continue;
        }
        mix_neighbors(
            &mut neighbors,
            state_target,
            n_haps,
            hap1_idx,
            sample_uncertainty,
            &mut rng,
        );
        neighbors.retain(|&h| allow_donor_at_marker(h, LocalMarkerIdx(center_marker)));
        if neighbors.is_empty() {
            refill_neighbors_for_marker(
                &mut neighbors,
                state_target,
                n_haps,
                hap1_idx,
                LocalMarkerIdx(center_marker),
                sample_uncertainty,
                &mut rng,
                &allow_donor_at_marker,
            );
            if neighbors.is_empty() {
                continue;
            }
        }
        record_neighbors(&neighbors);

        // 2. Build constraint: at hets, H2 must produce genotype with H1
        for m in 0..n_markers {
            let a1 = seq1[m];
            let a2 = seq2[m];
            let anchor_a1 = anchor_h1.get(m).copied().unwrap_or(255);
            let anchor_a2 = anchor_h2.get(m).copied().unwrap_or(255);
            let is_anchor = anchor_a1 != 255 || anchor_a2 != 255;
            if a1 == 255 || a2 == 255 || a1 == a2 {
                fixed_allele[m] = 255;
            } else if is_anchor {
                fixed_allele[m] = anchor_a1;
            } else {
                fixed_allele[m] = h1_alleles[m];
            }
        }

        // 3. Run haploid FFBS for H2
        ffbs_haploid_constrained(
            &mut path2_idx,
            n_markers,
            neighbors.len(),
            p_recomb,
            seq1,
            seq2,
            conf,
            phase_conf,
            &fixed_allele,
            &neighbors,
            phase_ibs,
            p_no_err,
            p_err,
            &mut rng,
            workspace,
        );

        // Refresh the latent reference path at all markers for the next iteration.
        refresh_path_ref_from_states(&mut path2_ref, &path2_idx, &neighbors);

        // 4. Update H2 based on sampled reference alleles
        //    GIBBS SAMPLING: only update H2, leave H1 fixed
        //    At hets, H2 is constrained to be opposite of H1, so just verify consistency.
        for m in 0..n_markers {
            let a1 = seq1[m];
            let a2 = seq2[m];
            let anchor_a1 = anchor_h1.get(m).copied().unwrap_or(255);
            let anchor_a2 = anchor_h2.get(m).copied().unwrap_or(255);
            let is_anchor = anchor_a1 != 255 || anchor_a2 != 255;
            if is_anchor {
                h1_alleles[m] = anchor_a1;
                h2_alleles[m] = anchor_a2;
                continue;
            }

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

        // Keep label orientation consistent across iterations to avoid H1/H2 drift.
        let mut keep_score = 0.0f32;
        let mut swap_score = 0.0f32;
        let progress = if n_mcmc_steps > 1 {
            step as f32 / (n_mcmc_steps - 1) as f32
        } else {
            1.0
        };
        let burnin_steps = (n_mcmc_steps / 2).max(1);
        let burnin_ratio = if step < burnin_steps {
            step as f32 / burnin_steps as f32
        } else {
            1.0
        };
        // Low inertia early for exploration; stronger later for stability.
        let inertia_weight = (0.15 + 0.85 * progress)
            * (0.4 + 0.6 * burnin_ratio)
            * (0.25 + 0.75 * (1.0 - sample_uncertainty));
        let anchor_weight = 1.0 + (1.0 - sample_uncertainty) * progress;
        for &m in &anchors_static {
            let w = phase_conf
                .get(m)
                .copied()
                .unwrap_or(1.0)
                .clamp(0.0, 1.0)
                .max(0.1);
            if prev_path1_ref.get(m).copied().unwrap_or(0) == path1_ref.get(m).copied().unwrap_or(0)
            {
                keep_score += inertia_weight * w;
            }
            if prev_path2_ref.get(m).copied().unwrap_or(0) == path2_ref.get(m).copied().unwrap_or(0)
            {
                keep_score += inertia_weight * w;
            }
            if prev_path1_ref.get(m).copied().unwrap_or(0) == path2_ref.get(m).copied().unwrap_or(0)
            {
                swap_score += inertia_weight * w;
            }
            if prev_path2_ref.get(m).copied().unwrap_or(0) == path1_ref.get(m).copied().unwrap_or(0)
            {
                swap_score += inertia_weight * w;
            }
            let anchor_a1 = anchor_h1.get(m).copied().unwrap_or(255);
            let anchor_a2 = anchor_h2.get(m).copied().unwrap_or(255);
            if anchor_a1 != 255 && anchor_a2 != 255 {
                let mut anchor_keep = 0.0f32;
                let mut anchor_swap = 0.0f32;
                if h1_alleles[m] == anchor_a1 {
                    anchor_keep += 1.0;
                }
                if h2_alleles[m] == anchor_a2 {
                    anchor_keep += 1.0;
                }
                if h2_alleles[m] == anchor_a1 {
                    anchor_swap += 1.0;
                }
                if h1_alleles[m] == anchor_a2 {
                    anchor_swap += 1.0;
                }
                keep_score += anchor_weight * anchor_keep;
                swap_score += anchor_weight * anchor_swap;
            }
        }
        if swap_score > keep_score {
            std::mem::swap(&mut path1_ref, &mut path2_ref);
            std::mem::swap(&mut h1_alleles, &mut h2_alleles);
        }

        // After first step, we have a valid path to use for latent state lookup
        // in subsequent iterations
        for &m in het_positions {
            let idx = het_index[m];
            if idx == usize::MAX {
                continue;
            }
            let a1 = seq1[m];
            let a2 = seq2[m];
            if a1 == 255 || a2 == 255 || a1 == a2 {
                continue;
            }
            let swap = h1_alleles[m] != a1;
            swap_counts[idx] += if swap { 1.0 } else { 0.0 };
            swap_obs[idx] += 1.0;
        }
    }

    // Determine swap decisions from posterior orientation mass across MCMC steps.
    // Using only the final draw is too unstable under symmetric evidence.
    let mut swap_bits = Vec::with_capacity(het_positions.len());
    let mut swap_lr = Vec::with_capacity(het_positions.len());
    let mut swap_probs = Vec::with_capacity(het_positions.len());
    for (i, &m) in het_positions.iter().enumerate() {
        let a1 = seq1[m];
        let a2 = seq2[m];

        if a1 == 255 || a2 == 255 || a1 == a2 {
            swap_bits.push(0);
            swap_lr.push(1.0);
            swap_probs.push(0.5);
            continue;
        }

        let p_swap = if swap_obs[i] > 0.0 {
            (swap_counts[i] + 0.5) / (swap_obs[i] + 1.0)
        } else {
            0.5
        };
        let swap = p_swap > 0.5;
        swap_bits.push(if swap { 1 } else { 0 });
        let p_keep = 1.0 - p_swap;
        let (max_p, min_p) = if p_swap >= p_keep {
            (p_swap, p_keep)
        } else {
            (p_keep, p_swap)
        };
        let lr = if min_p < 1e-30 {
            1e6
        } else {
            (max_p / min_p).min(1e6)
        };
        swap_lr.push(lr);
        swap_probs.push(p_swap.clamp(0.0, 1.0));
    }
    if !swap_probs.is_empty() {
        let transition_logs = compute_label_switch_transition_logs(p_recomb, het_positions);
        let mut dp0 = vec![f32::NEG_INFINITY; swap_probs.len()];
        let mut dp1 = vec![f32::NEG_INFINITY; swap_probs.len()];
        let mut prev_state = vec![0u8; swap_probs.len()];
        for i in 0..swap_probs.len() {
            let p1 = swap_probs[i].clamp(1e-6, 1.0 - 1e-6);
            let p0 = (1.0 - p1).clamp(1e-6, 1.0 - 1e-6);
            let e0 = p0.ln();
            let e1 = p1.ln();
            if i == 0 {
                dp0[i] = e0;
                dp1[i] = e1;
                continue;
            }
            let (stay, sw) = transition_logs
                .get(i)
                .copied()
                .unwrap_or_else(|| ((1.0 - 0.01f32).ln(), 0.01f32.ln()));
            let from0_to0 = dp0[i - 1] + stay;
            let from1_to0 = dp1[i - 1] + sw;
            if from0_to0 >= from1_to0 {
                dp0[i] = from0_to0 + e0;
                prev_state[i] = 0;
            } else {
                dp0[i] = from1_to0 + e0;
                prev_state[i] = 1;
            }
            let from0_to1 = dp0[i - 1] + sw;
            let from1_to1 = dp1[i - 1] + stay;
            if from0_to1 >= from1_to1 {
                dp1[i] = from0_to1 + e1;
            } else {
                dp1[i] = from1_to1 + e1;
                prev_state[i] |= 2;
            }
        }
        let mut state = if dp1[swap_probs.len() - 1] > dp0[swap_probs.len() - 1] {
            1u8
        } else {
            0u8
        };
        for i in (0..swap_probs.len()).rev() {
            swap_bits[i] = state;
            if i == 0 {
                break;
            }
            let prev = prev_state[i];
            if state == 0 {
                state = prev & 1;
            } else {
                state = if (prev & 2) != 0 { 1 } else { 0 };
            }
        }
    }

    let swap_probs_conf = swap_probs.clone();
    {
        let mut path1_switches = 0usize;
        let mut path2_switches = 0usize;
        let mut h1_transitions = 0usize;
        let mut path1_counts: std::collections::BTreeMap<u32, usize> =
            std::collections::BTreeMap::new();
        let mut path2_counts: std::collections::BTreeMap<u32, usize> =
            std::collections::BTreeMap::new();
        for m in 1..n_markers {
            if path1_ref[m] != path1_ref[m - 1] {
                path1_switches += 1;
            }
            if path2_ref[m] != path2_ref[m - 1] {
                path2_switches += 1;
            }
            let prev_is_het =
                seq1[m - 1] != 255 && seq2[m - 1] != 255 && seq1[m - 1] != seq2[m - 1];
            let curr_is_het = seq1[m] != 255 && seq2[m] != 255 && seq1[m] != seq2[m];
            if prev_is_het && curr_is_het && h1_alleles[m] != h1_alleles[m - 1] {
                h1_transitions += 1;
            }
        }
        for &h in &path1_ref {
            *path1_counts.entry(h).or_insert(0) += 1;
        }
        for &h in &path2_ref {
            *path2_counts.entry(h).or_insert(0) += 1;
        }
        let mut top1: Vec<(u32, usize)> = path1_counts.into_iter().collect();
        top1.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        top1.truncate(3);
        let mut top2: Vec<(u32, usize)> = path2_counts.into_iter().collect();
        top2.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        top2.truncate(3);
        let mut swap_transitions = 0usize;
        for i in 1..swap_bits.len() {
            if swap_bits[i] != swap_bits[i - 1] {
                swap_transitions += 1;
            }
        }
        let mut mean_prob = 0.0f32;
        let mut near_mid = 0usize;
        for &p in &swap_probs {
            mean_prob += p;
            if (p - 0.5).abs() < 0.05 {
                near_mid += 1;
            }
        }
        if !swap_probs.is_empty() {
            mean_prob /= swap_probs.len() as f32;
        }
        let suspicious_swap_entropy =
            swap_bits.len() >= 8 && swap_transitions * 3 > swap_bits.len();
        let suspicious_overconfidence = near_mid == 0 && mean_prob > 0.95;
        let is_anomaly = suspicious_swap_entropy || suspicious_overconfidence;
        let periodic_sample = sample_idx % 128 == 0;
        if het_positions.len() >= 20 && (is_anomaly || periodic_sample) {
            if is_anomaly {
                let preview = swap_bits.len().min(16);
                eprintln!(
                    "[mosaic dynamic summary] seed={} sample={} markers={} hets={} steps={} has_anchor={} anomaly=true path_switches=({},{}) h1_transitions={} swap_transitions={} swap_mean={:.3} near_mid={} top_path1={:?} top_path2={:?} preview_bits={:?} preview_probs={:?}",
                    seed,
                    sample_idx,
                    n_markers,
                    het_positions.len(),
                    n_mcmc_steps,
                    has_anchor,
                    path1_switches,
                    path2_switches,
                    h1_transitions,
                    swap_transitions,
                    mean_prob,
                    near_mid,
                    top1,
                    top2,
                    &swap_bits[..preview],
                    &swap_probs[..preview]
                );
            } else {
                eprintln!(
                    "[mosaic dynamic summary] seed={} sample={} markers={} hets={} steps={} has_anchor={} anomaly=false path_switches=({},{}) h1_transitions={} swap_transitions={} swap_mean={:.3}",
                    seed,
                    sample_idx,
                    n_markers,
                    het_positions.len(),
                    n_mcmc_steps,
                    has_anchor,
                    path1_switches,
                    path2_switches,
                    h1_transitions,
                    swap_transitions,
                    mean_prob
                );
            }
        }
    }
    (
        swap_bits,
        swap_lr,
        swap_probs,
        swap_probs_conf,
        MosaicPaths {
            path1: path1_ref,
            path2: path2_ref,
        },
    )
}

/// Find the best constant pair of states that explains the target genotype.
///
/// This initialization heuristic performs a pairwise scan of all HMM states (which
/// correspond to reference haplotypes in ThreadedHaps) to find the pair (i, j)
/// that maximizes consistency with the target genotype.
///
/// This breaks the symmetry of the Combined HMM initialization (which cannot distinguish
/// between phasing configurations at 0/1 sites) and helps the Gibbs sampler escape
/// "Mosaic Traps" where H1 and H2 lock each other into high-switching local optima.
fn find_best_constant_pair_with_buffer<RefSpace>(
    n_markers: usize,
    n_states: usize,
    seq1: &[u8],
    seq2: &[u8],
    conf: &[f32],
    p_no_err: f32,
    p_err: f32,
    ref_provider: &mut RefAlleleProvider<'_, AnyMarkerSpace, RefSpace>,
    predecoded_ref_flat: Option<&[u8]>,
    scores: &mut Vec<f32>,
    hint: Option<&MosaicPaths>,
) -> Option<MosaicPaths> {
    if n_states < 2 {
        return None;
    }
    // Bound compute on long windows by evaluating a sparse marker grid.
    const MAX_EVAL_MARKERS: usize = 2000;
    let marker_stride = n_markers
        .saturating_add(MAX_EVAL_MARKERS - 1)
        .checked_div(MAX_EVAL_MARKERS)
        .unwrap_or(1)
        .max(1);

    // Compute hint bonuses to break ties (prefer states present in prior paths).
    let mut state_bonus = vec![0.0f32; n_states];
    if let Some(h) = hint {
        for &s in h.path1.iter().chain(h.path2.iter()) {
            if (s as usize) < n_states {
                state_bonus[s as usize] += 0.001;
            }
        }
    }

    let sparse_markers: Vec<usize> = (0..n_markers)
        .step_by(marker_stride)
        .filter(|&m| {
            let a1 = seq1[m];
            let a2 = seq2[m];
            !(a1 == 255 && a2 == 255)
        })
        .collect();
    if sparse_markers.is_empty() {
        return None;
    }
    let mut sparse_ref_rows = vec![255u8; sparse_markers.len() * n_states];
    for (row_i, &m) in sparse_markers.iter().enumerate() {
        let dst = &mut sparse_ref_rows[row_i * n_states..(row_i + 1) * n_states];
        if let Some(flat) = predecoded_ref_flat {
            let src_start = m * n_states;
            let src_end = src_start + n_states;
            if src_end <= flat.len() {
                dst.copy_from_slice(&flat[src_start..src_end]);
            } else {
                ref_provider.fill_ref_alleles(m, dst);
            }
        } else {
            ref_provider.fill_ref_alleles(m, dst);
        }
    }

    // Stage 1: O(M*K) unary compatibility to prefilter candidate states.
    let mut state_unary = vec![0.0f32; n_states];
    let informative = sparse_markers.len();
    for (row_i, &m) in sparse_markers.iter().enumerate() {
        let a1 = seq1[m];
        let a2 = seq2[m];
        let conf_m = conf.get(m).copied().unwrap_or(1.0).clamp(0.0, 1.0);
        let is_het = a1 != 255 && a2 != 255 && a1 != a2;
        let obs = if a1 != 255 { a1 } else { a2 };
        let ref_row = &sparse_ref_rows[row_i * n_states..(row_i + 1) * n_states];
        for i in 0..n_states {
            let r = ref_row[i];
            let p = if is_het {
                let e1 = emit_prob(r, a1, conf_m, p_no_err, p_err);
                let e2 = emit_prob(r, a2, conf_m, p_no_err, p_err);
                0.5 * (e1 + e2)
            } else {
                emit_prob(r, obs, conf_m, p_no_err, p_err)
            };
            state_unary[i] += p.max(1e-30).ln();
        }
    }

    if informative == 0 {
        return None;
    }

    // For small panels or fully heterozygous targets, unary scores may be
    // uniform or noisy. Ensure we scan a sufficient number of candidates.
    let target_k = if n_states <= 256 {
        n_states
    } else {
        n_states.min((n_states / 3).max(64).min(256))
    };
    let mut ranked_states: Vec<usize> = (0..n_states).collect();
    ranked_states.sort_by(|&a, &b| {
        let sa = state_unary[a] + state_bonus[a];
        let sb = state_unary[b] + state_bonus[b];
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked_states.truncate(target_k.max(2));

    // Stage 2: Greedy beam pair scoring over reduced candidate set.
    // Use top unary states as first-haplotype seeds, then score all partners per seed.
    // For small sets, perform exhaustive pairwise scoring to handle uniform unary scores.
    let cand_k = ranked_states.len();
    let n_seeds = if cand_k <= 256 { cand_k } else { 8.min(cand_k) };

    if scores.len() < cand_k {
        scores.resize(cand_k, 0.0);
    }

    let mut seed_states = ranked_states.clone();
    seed_states.sort_by(|&a, &b| {
        state_unary[b]
            .partial_cmp(&state_unary[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    seed_states.truncate(n_seeds);

    let mut best_score = f32::NEG_INFINITY;
    let mut best_pair = (ranked_states[0], ranked_states[1]);
    for &si in seed_states.iter() {
        scores[..cand_k].fill(0.0);
        let bonus_i = state_bonus[si];

        for (row_i, &m) in sparse_markers.iter().enumerate() {
            let a1 = seq1[m];
            let a2 = seq2[m];
            let conf_m = conf.get(m).copied().unwrap_or(1.0).clamp(0.0, 1.0);
            let is_het = a1 != 255 && a2 != 255 && a1 != a2;
            let obs = if a1 != 255 { a1 } else { a2 };
            let ref_row = &sparse_ref_rows[row_i * n_states..(row_i + 1) * n_states];
            let r_seed = ref_row[si];

            if is_het {
                let seed_a1 = emit_prob(r_seed, a1, conf_m, p_no_err, p_err);
                let seed_a2 = emit_prob(r_seed, a2, conf_m, p_no_err, p_err);
                for (cj, &sj) in ranked_states.iter().enumerate() {
                    if sj == si {
                        continue;
                    }
                    let rj = ref_row[sj];
                    let e1j = emit_prob(rj, a1, conf_m, p_no_err, p_err);
                    let e2j = emit_prob(rj, a2, conf_m, p_no_err, p_err);
                    let prob = 0.5 * (seed_a1 * e2j + seed_a2 * e1j);
                    scores[cj] += prob.max(1e-30).ln();
                }
            } else {
                let seed_obs = emit_prob(r_seed, obs, conf_m, p_no_err, p_err);
                for (cj, &sj) in ranked_states.iter().enumerate() {
                    if sj == si {
                        continue;
                    }
                    let rj = ref_row[sj];
                    let obs_j = emit_prob(rj, obs, conf_m, p_no_err, p_err);
                    let prob = seed_obs * obs_j;
                    scores[cj] += prob.max(1e-30).ln();
                }
            }
        }

        for (cj, &sj) in ranked_states.iter().enumerate() {
            if sj == si {
                continue;
            }
            let s = scores[cj] + bonus_i + state_bonus[sj];
            if s > best_score {
                best_score = s;
                best_pair = (si, sj);
            }
        }
    }

    // If best score is too low (worse than random), maybe don't use it?
    // But random initialization is also bad. This is likely the "least bad" start.
    // So we return it.
    let path1 = vec![best_pair.0 as u32; n_markers];
    let path2 = vec![best_pair.1 as u32; n_markers];

    Some(MosaicPaths { path1, path2 })
}

/// Sample phase swap decisions using Stochastic EM (single chain MCMC).
///
/// This implements Forward-Filtering Backward-Sampling (FFBS) with a single
/// Markov chain, which is the mathematically correct approach for phasing.
/// Multiple chains would require phase alignment to avoid symmetric mode
/// cancellation, so we use exactly one chain (Stochastic EM).
///
/// The algorithm:
/// 1. Initialize H1/H2 using pairwise compatibility search (breaks symmetry)
///    OR fall back to Combined HMM checkpoint sampling
/// 2. Run burn-in steps to let the chain mix via Gibbs sampling
/// 3. Take samples from the posterior
/// 4. Return swap decisions based on average posterior
fn sample_swap_bits_mosaic<RefSpace>(
    n_markers: usize,
    n_states: usize,
    p_recomb: &[f32],
    seq1: &[u8],
    seq2: &[u8],
    conf: &[f32],
    mut ref_provider: RefAlleleProvider<'_, AnyMarkerSpace, RefSpace>,
    pl_provider: Option<PlProvider>,
    block_starts: Arc<[usize]>,
    het_positions: &[usize],
    initial_paths: Option<&MosaicPaths>,
    anchor_hap1: Option<&[u8]>,
    anchor_hap2: Option<&[u8]>,
    seed: u64,
    burnin: usize,
    lr_samples_param: usize,
    p_no_err: f32,
    p_err: f32,
    workspace: &mut crate::utils::workspace::ThreadWorkspace,
) -> (Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>, MosaicPaths) {
    if het_positions.is_empty() || n_markers == 0 || n_states == 0 {
        return (
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            MosaicPaths {
                path1: Vec::new(),
                path2: Vec::new(),
            },
        );
    }

    let n_states = unsafe { StateCount::new_unchecked(n_states) };
    let n_states_usize = n_states.get();
    let max_block_len = max_block_len_from_starts(&block_starts, n_markers).max(1);
    let n_blocks = block_starts.len().max(1);
    workspace.ensure_for_window(n_markers, n_states_usize, max_block_len, n_blocks);

    let ref_flat_len = n_markers.saturating_mul(n_states_usize);
    if ref_flat_len > 0 {
        ref_provider.materialize_into(n_markers, &mut workspace.ref_alleles_flat[..ref_flat_len]);
    }
    let shared_ref = &workspace.ref_alleles_flat[..ref_flat_len];

    let combined_data = std::mem::take(&mut workspace.combined_checkpoint_data);
    let mut combined_checkpoints =
        FwdCheckpoints::from_buffer(block_starts.clone(), n_states_usize, combined_data);

    if workspace.dummy_target.len() < n_markers {
        workspace.dummy_target.resize(n_markers, 255);
        workspace.dummy_partner.resize(n_markers, 255);
        workspace.dummy_combined.resize(n_markers, true);
        workspace.dummy_hard_match.resize(n_markers, false);
    } else {
        workspace.dummy_target[..n_markers].fill(255);
        workspace.dummy_partner[..n_markers].fill(255);
        workspace.dummy_combined[..n_markers].fill(true);
        workspace.dummy_hard_match[..n_markers].fill(false);
    }
    let dummy_target = &workspace.dummy_target[..n_markers];
    let dummy_partner = &workspace.dummy_partner[..n_markers];
    let dummy_combined = &workspace.dummy_combined[..n_markers];
    let dummy_hard_match = &workspace.dummy_hard_match[..n_markers];
    let fwd = &mut workspace.fwd[..n_states_usize];
    let fwd_prior = &mut workspace.fwd_prior[..n_states_usize];
    let ref_alleles = &mut workspace.ref_alleles[..n_states_usize];
    build_fwd_checkpoints(
        &mut combined_checkpoints,
        n_markers,
        n_states_usize,
        ref_provider.ref_gt.n_haps(),
        p_recomb,
        seq1,
        seq2,
        conf,
        HapEmissionInputs {
            target_constraint: dummy_target,
            partner_allele: dummy_partner,
            use_combined: dummy_combined,
            hard_match: dummy_hard_match,
        },
        &mut ref_provider,
        (!workspace.ref_alleles_flat.is_empty()).then_some(shared_ref),
        pl_provider.as_ref(),
        &mut workspace.allele_probs,
        fwd,
        fwd_prior,
        ref_alleles,
        p_no_err,
        p_err,
        EmissionMode::Combined,
    );

    let anchor_h1 = anchor_hap1.unwrap_or(&[]);
    let anchor_h2 = anchor_hap2.unwrap_or(&[]);
    let has_anchor = anchor_h1.iter().any(|&a| a != 255) || anchor_h2.iter().any(|&a| a != 255);

    let heuristic_paths = find_best_constant_pair_with_buffer(
        n_markers,
        n_states_usize,
        seq1,
        seq2,
        conf,
        p_no_err,
        p_err,
        &mut ref_provider,
        Some(shared_ref),
        &mut workspace.scores,
        initial_paths,
    );

    let ref_view = ref_provider.ref_gt;
    let threaded_haps = ref_provider.threaded_haps;

    let fwd_prior_store = if workspace.fwd_prior.len() < n_states_usize {
        StateAVec32::new(n_states, 0.0).into_avec()
    } else {
        std::mem::replace(&mut workspace.fwd_prior, aligned_vec::AVec::new(32))
    };
    let ref_alleles_store = if workspace.ref_alleles.len() < n_states_usize {
        StateVec::new(n_states, 0).into_vec()
    } else {
        std::mem::take(&mut workspace.ref_alleles)
    };

    let mut buffers = MosaicBuffers {
        n_states,
        fwd: StateAVec32::from_avec(
            n_states,
            std::mem::replace(&mut workspace.fwd, aligned_vec::AVec::new(32)),
            0.0,
        ),
        fwd_prior: StateAVec32::from_avec(n_states, fwd_prior_store, 0.0),
        ref_alleles: StateVec::from_vec(n_states, ref_alleles_store, 0),
        ref_alleles_flat: Vec::new(),
        weights: StateVec::from_vec(n_states, std::mem::take(&mut workspace.weights), 0.0),
        allele_probs: std::mem::take(&mut workspace.allele_probs),
        hap1_checkpoints: FwdCheckpoints::from_buffer(
            block_starts.clone(),
            n_states_usize,
            std::mem::take(&mut workspace.hap1_checkpoint_data),
        ),
        hap2_checkpoints: FwdCheckpoints::from_buffer(
            block_starts.clone(),
            n_states_usize,
            std::mem::take(&mut workspace.hap2_checkpoint_data),
        ),
        hap1_allele: std::mem::take(&mut workspace.hap1_allele),
        hap1_partner_allele: std::mem::take(&mut workspace.hap1_partner_allele),
        hap1_use_combined: std::mem::take(&mut workspace.hap1_use_combined),
        hap1_hard_match: std::mem::take(&mut workspace.hap1_hard_match),
        hap2_allele: std::mem::take(&mut workspace.hap2_allele),
        hap2_partner_allele: std::mem::take(&mut workspace.hap2_partner_allele),
        hap2_use_combined: std::mem::take(&mut workspace.hap2_use_combined),
        hap2_hard_match: std::mem::take(&mut workspace.hap2_hard_match),
        path1: std::mem::take(&mut workspace.path1),
        path2: std::mem::take(&mut workspace.path2),
        fwd_block: std::mem::take(&mut workspace.fwd_block),
    };

    let mut chain = MosaicChain::new_with_buffers(
        seed,
        n_markers,
        p_recomb,
        seq1,
        seq2,
        conf,
        RefAlleleProvider::new(ref_view, threaded_haps),
        &combined_checkpoints,
        buffers,
        Some(shared_ref),
        p_no_err,
        p_err,
        pl_provider,
        anchor_h1.to_vec(),
        anchor_h2.to_vec(),
    );
    if has_anchor {
        chain.anchor_drop_prob = 0.0;
    }

    let start_paths = initial_paths.cloned().or(heuristic_paths);
    if let Some(paths) = start_paths {
        if paths.path1.len() == n_markers
            && paths.path2.len() == n_markers
            && paths.path1.iter().all(|&p| (p as usize) < n_states_usize)
            && paths.path2.iter().all(|&p| (p as usize) < n_states_usize)
        {
            chain.path1 = paths.path1;
            chain.path2 = paths.path2;
            chain.first_iteration = false;
        }
    }

    let complexity_steps = (het_positions.len() / 64).min(4);
    let burnin_steps = burnin.saturating_add(complexity_steps).clamp(2, 6);
    for _ in 0..burnin_steps {
        chain.step();
    }

    let lr_samples = lr_samples_param.max(12).min(32);
    let mut swap_counts = vec![0.0f32; het_positions.len()];
    let mut obs_counts = vec![0.0f32; het_positions.len()];

    let anchor_indices: Vec<usize> = if has_anchor {
        (0..n_markers)
            .filter(|&m| {
                anchor_h1.get(m).copied().unwrap_or(255) != 255
                    || anchor_h2.get(m).copied().unwrap_or(255) != 255
            })
            .collect()
    } else {
        Vec::new()
    };

    for _ in 0..lr_samples {
        chain.step();

        let mut sample_flip = false;
        if !anchor_indices.is_empty() {
            let mut direct = 0.0f32;
            let mut flipped = 0.0f32;
            let mut evidence = 0usize;
            for &m in &anchor_indices {
                let a1 = anchor_h1.get(m).copied().unwrap_or(255);
                let a2 = anchor_h2.get(m).copied().unwrap_or(255);
                if a1 == 255 && a2 == 255 {
                    continue;
                }
                let p1 = chain.path1[m] as usize;
                let p2 = chain.path2[m] as usize;
                if p1 >= n_states_usize || p2 >= n_states_usize {
                    continue;
                }
                let ref_row = chain.ref_row(m);
                let r1 = ref_row[p1];
                let r2 = ref_row[p2];
                let conf_m = conf[m].clamp(0.0, 1.0);
                if a1 != 255 {
                    direct += emit_prob(r1, a1, conf_m, p_no_err, p_err).max(1e-30).ln();
                    flipped += emit_prob(r2, a1, conf_m, p_no_err, p_err).max(1e-30).ln();
                    evidence += 1;
                }
                if a2 != 255 {
                    direct += emit_prob(r2, a2, conf_m, p_no_err, p_err).max(1e-30).ln();
                    flipped += emit_prob(r1, a2, conf_m, p_no_err, p_err).max(1e-30).ln();
                    evidence += 1;
                }
            }
            sample_flip = evidence > 0 && flipped > direct;
        }

        for (i, &m) in het_positions.iter().enumerate() {
            let a1 = seq1[m];
            let a2 = seq2[m];
            if a1 == 255 || a2 == 255 || a1 == a2 {
                continue;
            }
            let p1 = chain.path1[m] as usize;
            let p2 = chain.path2[m] as usize;
            if p1 >= n_states_usize || p2 >= n_states_usize {
                continue;
            }
            let ref_row = chain.ref_row(m);
            let ref1 = ref_row[p1];
            let ref2 = ref_row[p2];

            let mut orient = if ref1 == a1 && ref2 == a2 {
                Some(0u8)
            } else if ref1 == a2 && ref2 == a1 {
                Some(1u8)
            } else {
                None
            };

            if orient.is_none() {
                let c = conf[m].clamp(0.0, 1.0);
                let keep = emit_prob(ref1, a1, c, p_no_err, p_err)
                    * emit_prob(ref2, a2, c, p_no_err, p_err);
                let swap = emit_prob(ref1, a2, c, p_no_err, p_err)
                    * emit_prob(ref2, a1, c, p_no_err, p_err);
                orient = Some(if swap > keep { 1 } else { 0 });
            }

            if let Some(mut bit) = orient {
                if sample_flip {
                    bit ^= 1;
                }
                if bit == 1 {
                    swap_counts[i] += 1.0;
                }
                obs_counts[i] += 1.0;
            }
        }
    }

    let new_paths = MosaicPaths {
        path1: chain.path1.clone(),
        path2: chain.path2.clone(),
    };
    buffers = chain.into_buffers();

    let mut swap_bits = Vec::with_capacity(het_positions.len());
    let mut swap_lr = Vec::with_capacity(het_positions.len());
    let mut swap_probs = Vec::with_capacity(het_positions.len());
    let mut swap_probs_conf = Vec::with_capacity(het_positions.len());

    for (i, &m) in het_positions.iter().enumerate() {
        let a1 = seq1[m];
        let a2 = seq2[m];
        if a1 == 255 || a2 == 255 || a1 == a2 || obs_counts[i] < 0.5 {
            swap_bits.push(0);
            swap_lr.push(1.0);
            swap_probs.push(0.5);
            swap_probs_conf.push(0.5);
            continue;
        }

        let p_swap = (swap_counts[i] + 0.5) / (obs_counts[i] + 1.0);
        let p_keep = 1.0 - p_swap;
        let chosen_swap = p_swap > 0.5;
        swap_bits.push(chosen_swap as u8);
        let (max_p, min_p) = if p_swap >= p_keep {
            (p_swap, p_keep)
        } else {
            (p_keep, p_swap)
        };
        let lr = if min_p < 1e-30 {
            1e6_f32
        } else {
            (max_p / min_p).min(1e6_f32)
        };
        swap_lr.push(lr);
        swap_probs.push(p_swap.clamp(0.0, 1.0));
        swap_probs_conf.push(p_swap.clamp(0.0, 1.0));
    }

    // Removed redundant and unstable overwrite loop (using single sample "new_paths").
    // The MCMC loop above already accumulates robust posterior probabilities in `swap_probs`.

    if !het_positions.is_empty() {
        let transition_logs = compute_label_switch_transition_logs(p_recomb, het_positions);
        let mut dp0 = vec![f32::NEG_INFINITY; het_positions.len()];
        let mut dp1 = vec![f32::NEG_INFINITY; het_positions.len()];
        let mut prev_state = vec![0u8; het_positions.len()];

        for i in 0..het_positions.len() {
            let p_swap = swap_probs[i].clamp(1e-6, 1.0 - 1e-6);
            let p_keep = (1.0 - p_swap).clamp(1e-6, 1.0 - 1e-6);
            let emit0 = p_keep.ln();
            let emit1 = p_swap.ln();

            if i == 0 {
                dp0[i] = emit0;
                dp1[i] = emit1;
            } else {
                let (stay, sw) = transition_logs
                    .get(i)
                    .copied()
                    .unwrap_or_else(|| ((1.0 - 0.01f32).ln(), 0.01f32.ln()));
                let from0_to0 = dp0[i - 1] + stay;
                let from1_to0 = dp1[i - 1] + sw;
                if from0_to0 >= from1_to0 {
                    dp0[i] = from0_to0 + emit0;
                    prev_state[i] = 0;
                } else {
                    dp0[i] = from1_to0 + emit0;
                    prev_state[i] = 1;
                }

                let from0_to1 = dp0[i - 1] + sw;
                let from1_to1 = dp1[i - 1] + stay;
                if from0_to1 >= from1_to1 {
                    dp1[i] = from0_to1 + emit1;
                } else {
                    dp1[i] = from1_to1 + emit1;
                    prev_state[i] |= 2;
                }
            }
        }

        let mut state = if dp1[het_positions.len() - 1] > dp0[het_positions.len() - 1] {
            1u8
        } else {
            0u8
        };
        for idx in (0..het_positions.len()).rev() {
            swap_bits[idx] = state;
            if idx == 0 {
                break;
            }
            let prev = prev_state[idx];
            if state == 0 {
                state = prev & 1;
            } else {
                state = if (prev & 2) != 0 { 1 } else { 0 };
            }
        }
    }

    workspace.fwd = buffers.fwd.into_avec();
    workspace.fwd_prior = buffers.fwd_prior.into_avec();
    workspace.ref_alleles = buffers.ref_alleles.into_vec();
    if !buffers.ref_alleles_flat.is_empty() {
        workspace.ref_alleles_flat = buffers.ref_alleles_flat;
    }
    workspace.weights = buffers.weights.into_vec();
    workspace.allele_probs = buffers.allele_probs;
    workspace.hap1_checkpoint_data = buffers.hap1_checkpoints.into_buffer();
    workspace.hap2_checkpoint_data = buffers.hap2_checkpoints.into_buffer();
    workspace.hap1_allele = buffers.hap1_allele;
    workspace.hap1_partner_allele = buffers.hap1_partner_allele;
    workspace.hap1_use_combined = buffers.hap1_use_combined;
    workspace.hap1_hard_match = buffers.hap1_hard_match;
    workspace.hap2_allele = buffers.hap2_allele;
    workspace.hap2_partner_allele = buffers.hap2_partner_allele;
    workspace.hap2_use_combined = buffers.hap2_use_combined;
    workspace.hap2_hard_match = buffers.hap2_hard_match;
    workspace.path1 = buffers.path1;
    workspace.path2 = buffers.path2;
    workspace.fwd_block = buffers.fwd_block;
    workspace.combined_checkpoint_data = combined_checkpoints.into_buffer();

    (swap_bits, swap_lr, swap_probs, swap_probs_conf, new_paths)
}

/// Decision type for Stage 2 marker processing
#[derive(Debug, Clone)]
enum Stage2Decision {
    /// Phase an unphased heterozygote
    Phase {
        marker: usize,
        should_swap: bool,
        lr: f32,
    },
    /// Impute a missing genotype
    Impute { marker: usize, a1: u8, a2: u8 },
}

#[derive(Debug, Clone, Copy)]
struct PhaseEvidence {
    marker: usize,
    log_same: f32,
    log_swap: f32,
}

fn decode_phase_evidence_path(
    evidence: &[PhaseEvidence],
    gen_positions: &[f64],
    recomb_intensity: f32,
) -> Vec<(usize, bool, f32)> {
    if evidence.is_empty() {
        return Vec::new();
    }
    let n = evidence.len();
    let mut dp_same = vec![f32::NEG_INFINITY; n];
    let mut dp_swap = vec![f32::NEG_INFINITY; n];
    let mut prev_same = vec![0u8; n];
    let mut prev_swap = vec![0u8; n];

    dp_same[0] = evidence[0].log_same;
    dp_swap[0] = evidence[0].log_swap;
    prev_same[0] = 0;
    prev_swap[0] = 1;

    for i in 1..n {
        let m_prev = evidence[i - 1].marker;
        let m_cur = evidence[i].marker;
        let pos_prev = *gen_positions.get(m_prev).unwrap_or(&0.0);
        let pos_cur = *gen_positions.get(m_cur).unwrap_or(&pos_prev);
        let d_cm = (pos_cur - pos_prev).max(0.0);
        let d_m = d_cm / 100.0;
        let r = (-f64::exp_m1(-(recomb_intensity as f64) * d_m) as f32).clamp(1e-5, 0.25);
        let stay = (1.0 - r).ln();
        let sw = r.ln();

        let same_from_same = dp_same[i - 1] + stay;
        let same_from_swap = dp_swap[i - 1] + sw;
        if same_from_same >= same_from_swap {
            dp_same[i] = same_from_same + evidence[i].log_same;
            prev_same[i] = 0;
        } else {
            dp_same[i] = same_from_swap + evidence[i].log_same;
            prev_same[i] = 1;
        }

        let swap_from_swap = dp_swap[i - 1] + stay;
        let swap_from_same = dp_same[i - 1] + sw;
        if swap_from_swap >= swap_from_same {
            dp_swap[i] = swap_from_swap + evidence[i].log_swap;
            prev_swap[i] = 1;
        } else {
            dp_swap[i] = swap_from_same + evidence[i].log_swap;
            prev_swap[i] = 0;
        }
    }

    let mut states = vec![0u8; n];
    let mut st = if dp_swap[n - 1] > dp_same[n - 1] {
        1u8
    } else {
        0u8
    };
    for i in (0..n).rev() {
        states[i] = st;
        st = if st == 0 { prev_same[i] } else { prev_swap[i] };
    }

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let log_lr = (evidence[i].log_swap - evidence[i].log_same).abs();
        let lr = log_lr.exp().clamp(1.0, 1.0e6);
        out.push((evidence[i].marker, states[i] == 1, lr));
    }
    out
}

fn top_bridge_haplotype(
    bridge_probs: &std::collections::HashMap<u32, f32>,
) -> Option<(u32, f32, f32)> {
    let mut top_h = 0u32;
    let mut top_p = -1.0f32;
    let mut second_p = 0.0f32;
    for (&h, &p) in bridge_probs {
        if p > top_p {
            second_p = top_p.max(0.0);
            top_p = p;
            top_h = h;
        } else if p > second_p {
            second_p = p;
        }
    }
    if top_p > 0.0 {
        Some((top_h, top_p, second_p.max(0.0)))
    } else {
        None
    }
}

#[inline]
fn donor_log_odds_pass(top: f32, second: f32, p_mismatch: f32, obs_conf: f32) -> bool {
    if top <= 0.0 {
        return false;
    }
    let second_floor = second.max(1e-8);
    let log_odds = (top / second_floor).ln() * obs_conf.max(0.05);
    let target_err = p_mismatch.clamp(1e-6, 0.25);
    let tau = ((1.0 - target_err) / target_err).ln();
    log_odds >= tau
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
    const MAX_CARRIER_CONTEXT_MARKERS: usize = 24;
    const CARRIER_CONTEXT_RADIUS_CM: f64 = 0.75;

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

    fn interpolated_allele_probs_from_bridge<F>(
        &self,
        marker: usize,
        bridge_probs: &std::collections::HashMap<u32, f32>,
        get_allele: &F,
        a1: u8,
        a2: u8,
    ) -> [f32; 2]
    where
        F: Fn(usize, usize) -> u8,
    {
        let mut al_probs = [0.0f32; 2];
        for (&hap, &prob) in bridge_probs.iter() {
            let ref_allele = get_allele(marker, hap as usize);
            if ref_allele == 255 {
                continue;
            }
            if ref_allele == a1 {
                al_probs[0] += prob;
            } else if ref_allele == a2 {
                al_probs[1] += prob;
            }
        }
        al_probs
    }

    fn p_recomb(&self, gen_dist_cm: f64) -> f32 {
        let c = -(self.recomb_intensity as f64);
        let gen_dist_m = gen_dist_cm / 100.0;
        (-f64::exp_m1(c * gen_dist_m)) as f32
    }

    fn bridge_hap_probs(
        &self,
        marker: usize,
        state_probs: &[Vec<f32>],
        haps_at_mkr_a: &[CombinedHapId],
        haps_at_mkr_b: &[CombinedHapId],
    ) -> std::collections::HashMap<u32, f32> {
        let mkr_a = self.prev_stage1_marker[marker];
        let mkr_b = (mkr_a + 1).min(self.n_stage1 - 1);

        let probs_a = &state_probs[mkr_a];
        let probs_b = &state_probs[mkr_b];
        let n_states_a = probs_a.len().min(haps_at_mkr_a.len());
        let n_states_b = probs_b.len().min(haps_at_mkr_b.len());
        let mut hap_probs: std::collections::HashMap<u32, f32> =
            std::collections::HashMap::with_capacity((n_states_a + n_states_b).max(1));

        if mkr_a == mkr_b || self.stage1_markers.is_empty() {
            for (k, hap) in haps_at_mkr_a.iter().take(n_states_a).enumerate() {
                *hap_probs.entry(hap.as_u32()).or_insert(0.0) += probs_a[k];
            }
            let sum: f32 = hap_probs.values().copied().sum();
            if sum > 0.0 {
                let inv = 1.0 / sum;
                for p in hap_probs.values_mut() {
                    *p *= inv;
                }
            }
            return hap_probs;
        }

        let pos_a_idx = self.stage1_markers[mkr_a];
        let pos_b_idx = self.stage1_markers[mkr_b];

        let pos_a = *self.gen_positions.get(pos_a_idx).unwrap_or(&0.0);
        let pos_b = *self.gen_positions.get(pos_b_idx).unwrap_or(&pos_a);
        let pos_m = *self.gen_positions.get(marker).unwrap_or(&pos_a);

        if pos_b <= pos_a || pos_m <= pos_a {
            for (k, hap) in haps_at_mkr_a.iter().take(n_states_a).enumerate() {
                *hap_probs.entry(hap.as_u32()).or_insert(0.0) += probs_a[k];
            }
            let sum: f32 = hap_probs.values().copied().sum();
            if sum > 0.0 {
                let inv = 1.0 / sum;
                for p in hap_probs.values_mut() {
                    *p *= inv;
                }
            }
            return hap_probs;
        }
        if pos_m >= pos_b {
            for (k, hap) in haps_at_mkr_b.iter().take(n_states_b).enumerate() {
                *hap_probs.entry(hap.as_u32()).or_insert(0.0) += probs_b[k];
            }
            let sum: f32 = hap_probs.values().copied().sum();
            if sum > 0.0 {
                let inv = 1.0 / sum;
                for p in hap_probs.values_mut() {
                    *p *= inv;
                }
            }
            return hap_probs;
        }

        let d1 = (pos_m - pos_a).max(0.0);
        let d2 = (pos_b - pos_m).max(0.0);
        let r1 = self.p_recomb(d1);
        let r2 = self.p_recomb(d2);

        let shift1 = r1 / n_states_a.max(1) as f32;
        let shift2 = r2 / n_states_b.max(1) as f32;
        let scale1 = 1.0 - r1;
        let scale2 = 1.0 - r2;

        let denom = d1 + d2;
        let weight_a = if denom > 0.0 {
            (d2 / denom) as f32
        } else {
            0.5
        };
        let weight_b = 1.0 - weight_a;

        let mut sum = 0.0f32;
        for k in 0..n_states_a {
            let hap = haps_at_mkr_a[k].as_u32();
            let w = weight_a * (scale1 * probs_a[k] + shift1);
            *hap_probs.entry(hap).or_insert(0.0) += w;
            sum += w;
        }
        for k in 0..n_states_b {
            let hap = haps_at_mkr_b[k].as_u32();
            let w = weight_b * (scale2 * probs_b[k] + shift2);
            *hap_probs.entry(hap).or_insert(0.0) += w;
            sum += w;
        }

        if sum > 0.0 {
            let inv = 1.0 / sum;
            for p in hap_probs.values_mut() {
                *p *= inv;
            }
        } else {
            hap_probs.clear();
            for (k, hap) in haps_at_mkr_a.iter().take(n_states_a).enumerate() {
                *hap_probs.entry(hap.as_u32()).or_insert(0.0) += probs_a[k];
            }
            let fallback_sum: f32 = hap_probs.values().copied().sum();
            if fallback_sum > 0.0 {
                let inv = 1.0 / fallback_sum;
                for p in hap_probs.values_mut() {
                    *p *= inv;
                }
            }
        }
        hap_probs
    }

    fn blend_base_with_carriers(
        mut base: std::collections::HashMap<u32, f32>,
        carriers: &[u32],
        panel_haps: usize,
        obs_conf: f32,
    ) -> std::collections::HashMap<u32, f32> {
        if carriers.is_empty() || panel_haps == 0 {
            return base;
        }

        let carrier_frac = (carriers.len() as f32 / panel_haps as f32).clamp(0.0, 1.0);
        let rarity_boost = (1.0 - carrier_frac.sqrt()).clamp(0.0, 1.0);
        let lambda = (0.12 + 0.38 * rarity_boost + 0.10 * obs_conf).clamp(0.10, 0.60);
        let keep = 1.0 - lambda;

        for p in base.values_mut() {
            *p *= keep;
        }

        let mut carrier_prior_sum = 0.0f32;
        for &hap in carriers {
            carrier_prior_sum += base.get(&hap).copied().unwrap_or(0.0);
        }

        if carrier_prior_sum > 0.0 {
            let inv = 1.0 / carrier_prior_sum;
            for &hap in carriers {
                let prior = base.get(&hap).copied().unwrap_or(0.0);
                let add = lambda * prior * inv;
                if add > 0.0 {
                    *base.entry(hap).or_insert(0.0) += add;
                }
            }
        } else {
            let add = lambda / carriers.len() as f32;
            for &hap in carriers {
                *base.entry(hap).or_insert(0.0) += add;
            }
        }

        let sum: f32 = base.values().copied().sum();
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for p in base.values_mut() {
                *p *= inv;
            }
        }
        base
    }

    fn carrier_injected_bridge_hap_probs<F>(
        &self,
        marker: usize,
        state_probs: &[Vec<f32>],
        haps_at_mkr_a: &[CombinedHapId],
        haps_at_mkr_b: &[CombinedHapId],
        carriers: &[u32],
        context_markers: &[usize],
        panel_haps: usize,
        target_hap: usize,
        obs_conf: f32,
        p_mismatch: f32,
        get_allele: &F,
    ) -> std::collections::HashMap<u32, f32>
    where
        F: Fn(usize, usize) -> u8,
    {
        let mut base = self.bridge_hap_probs(marker, state_probs, haps_at_mkr_a, haps_at_mkr_b);
        if carriers.is_empty() || panel_haps == 0 {
            return base;
        }
        if context_markers.is_empty() {
            return Self::blend_base_with_carriers(base, carriers, panel_haps, obs_conf);
        }

        let mkr_a = self.prev_stage1_marker[marker];
        let mkr_b = (mkr_a + 1).min(self.n_stage1.saturating_sub(1));
        if mkr_a == mkr_b || self.stage1_markers.is_empty() {
            return base;
        }

        let pos_a_idx = self.stage1_markers[mkr_a];
        let pos_b_idx = self.stage1_markers[mkr_b];
        let pos_a = *self.gen_positions.get(pos_a_idx).unwrap_or(&0.0);
        let pos_b = *self.gen_positions.get(pos_b_idx).unwrap_or(&pos_a);
        let pos_m = *self.gen_positions.get(marker).unwrap_or(&pos_a);
        if !(pos_b > pos_a && pos_m >= pos_a && pos_m <= pos_b) {
            return base;
        }

        let err = p_mismatch.clamp(1e-4, 0.25);
        let ln_match = (1.0 - err).ln();
        let ln_mismatch = err.ln();
        let prior_floor = (1.0 / panel_haps as f32).max(1e-9);
        const MIN_CONTEXT_INFO_MARKERS: usize = 3;

        let mut max_log_weight = f32::NEG_INFINITY;
        let mut carrier_log_weights: Vec<(u32, f32)> = Vec::with_capacity(carriers.len());
        for &hap in carriers {
            let mut n_info = 0usize;
            let mut ll = 0.0f32;
            for &ctx_m in context_markers {
                let ta = get_allele(ctx_m, target_hap);
                let ha = get_allele(ctx_m, hap as usize);
                if ta == 255 || ha == 255 {
                    continue;
                }
                n_info += 1;
                if ta == ha {
                    ll += ln_match;
                } else {
                    ll += ln_mismatch;
                }
            }
            if n_info < MIN_CONTEXT_INFO_MARKERS {
                continue;
            }
            let prior = base.get(&hap).copied().unwrap_or(0.0).max(prior_floor);
            let log_weight = prior.ln() + ll;
            if log_weight > max_log_weight {
                max_log_weight = log_weight;
            }
            carrier_log_weights.push((hap, log_weight));
        }
        if carrier_log_weights.is_empty() {
            return Self::blend_base_with_carriers(base, carriers, panel_haps, obs_conf);
        }

        let mut carrier_post_sum = 0.0f32;
        for i in 0..carrier_log_weights.len() {
            let log_weight = carrier_log_weights[i].1;
            let w = (log_weight - max_log_weight).exp();
            carrier_post_sum += w;
        }
        if carrier_post_sum <= 0.0 {
            return base;
        }
        let inv_post = 1.0 / carrier_post_sum;

        // Adaptive carrier-conditioning strength:
        // - stronger for very rare carrier sets and confident observations
        // - weaker when many carriers dilute informativeness.
        let carrier_frac = (carriers.len() as f32 / panel_haps as f32).clamp(0.0, 1.0);
        let rarity_boost = (1.0 - carrier_frac.sqrt()).clamp(0.0, 1.0);
        let context_strength =
            (carrier_log_weights.len() as f32 / context_markers.len() as f32).clamp(0.0, 1.0);
        let lambda = (0.25 + 0.50 * rarity_boost + 0.15 * context_strength + 0.10 * obs_conf)
            .clamp(0.20, 0.90);
        let keep = 1.0 - lambda;

        for p in base.values_mut() {
            *p *= keep;
        }
        for (hap, log_weight) in carrier_log_weights {
            let p_carrier = (log_weight - max_log_weight).exp() * inv_post;
            let add = lambda * p_carrier;
            if add > 0.0 {
                *base.entry(hap).or_insert(0.0) += add;
            }
        }

        let sum: f32 = base.values().copied().sum();
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for p in base.values_mut() {
                *p *= inv;
            }
        }
        base
    }

    fn carrier_context_markers(&self, marker: usize) -> Vec<usize> {
        if self.n_stage1 == 0
            || self.stage1_markers.is_empty()
            || marker >= self.gen_positions.len()
        {
            return Vec::new();
        }

        let mkr_a = self.prev_stage1_marker[marker];
        let mkr_b = (mkr_a + 1).min(self.n_stage1.saturating_sub(1));
        let pos_m = *self.gen_positions.get(marker).unwrap_or(&0.0);
        let mut out: Vec<usize> = Vec::with_capacity(Self::MAX_CARRIER_CONTEXT_MARKERS + 2);
        let mut left_idx = mkr_a as isize;
        let mut right_idx = mkr_b as isize;

        while out.len() < Self::MAX_CARRIER_CONTEXT_MARKERS
            && (left_idx >= 0 || right_idx < self.n_stage1 as isize)
        {
            let left_candidate = if left_idx >= 0 {
                let idx = left_idx as usize;
                let marker_idx = self.stage1_markers[idx];
                let d = (pos_m - *self.gen_positions.get(marker_idx).unwrap_or(&pos_m)).abs();
                if d <= Self::CARRIER_CONTEXT_RADIUS_CM {
                    Some((marker_idx, d))
                } else {
                    None
                }
            } else {
                None
            };
            let right_candidate = if right_idx < self.n_stage1 as isize {
                let idx = right_idx as usize;
                let marker_idx = self.stage1_markers[idx];
                let d = (*self.gen_positions.get(marker_idx).unwrap_or(&pos_m) - pos_m).abs();
                if d <= Self::CARRIER_CONTEXT_RADIUS_CM {
                    Some((marker_idx, d))
                } else {
                    None
                }
            } else {
                None
            };

            match (left_candidate, right_candidate) {
                (Some((lm, dl)), Some((rm, dr))) => {
                    if dl <= dr {
                        out.push(lm);
                        left_idx -= 1;
                    } else {
                        out.push(rm);
                        right_idx += 1;
                    }
                }
                (Some((lm, _)), None) => {
                    out.push(lm);
                    left_idx -= 1;
                }
                (None, Some((rm, _))) => {
                    out.push(rm);
                    right_idx += 1;
                }
                (None, None) => break,
            }
        }

        if out.len() > 1 {
            out.sort_unstable();
            out.dedup();
        }
        out
    }
}

impl<RefSpace: Send + Sync> PhasingPipeline<RefSpace> {
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
                        if current_phased.allele(MarkerIdx::new(m as u32), HapIdx::new(h as u32))
                            == 1
                        {
                            alt_count += 1;
                        }
                    }
                    let maf = (alt_count as f32 / n_haps as f32)
                        .min(1.0 - alt_count as f32 / n_haps as f32);
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

        Ok(GenotypeMatrix::new_phased_with_confidence_and_likelihoods(
            markers,
            columns,
            samples,
            current_phased.confidence_clone(),
            current_phased.likelihoods_pl_arc(),
        )
        .with_phase_confidence(current_phased.phase_confidence_clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn build_test_markers(
        n_markers: usize,
        step_bp: u32,
    ) -> crate::data::marker::Markers<crate::data::AnyMarkerSpace> {
        use crate::data::ChromIdx;
        use crate::data::marker::{Allele, Marker, Markers, Nucleotide};

        let mut markers = Markers::<crate::data::AnyMarkerSpace>::new();
        markers.add_chrom("chr1");
        for i in 0..n_markers {
            let m = Marker::new(
                ChromIdx::new(0),
                i as u32 * step_bp,
                Some(format!("m{}", i).into()),
                Allele::Base(Nucleotide::A),
                vec![Allele::Base(Nucleotide::T)],
            );
            markers.push(m);
        }
        markers
    }

    fn build_ref_panel_with_hero(
        n_markers: usize,
        n_ref_samples: usize,
        hero_sample_idx: usize,
        hero_pattern: &[u8],
        seed: u64,
    ) -> (
        GenotypeMatrix<crate::data::storage::phase_state::Phased, crate::data::AnyMarkerSpace>,
        usize,
    ) {
        use crate::data::haplotype::Samples;
        use crate::data::storage::GenotypeColumn;
        use rand::{Rng, SeedableRng};
        use std::sync::Arc;

        let n_ref_haps = n_ref_samples * 2;
        let mut ref_haps = vec![vec![0u8; n_markers]; n_ref_haps];
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        for h in 0..n_ref_haps {
            for m in 0..n_markers {
                ref_haps[h][m] = if rng.random_bool(0.5) { 1 } else { 0 };
            }
        }

        let hero_hap_idx = hero_sample_idx * 2;
        let anti_hap_idx = hero_hap_idx + 1;
        for m in 0..n_markers {
            let a = hero_pattern[m];
            ref_haps[hero_hap_idx][m] = a;
            ref_haps[anti_hap_idx][m] = 1 - a;
        }

        let markers = build_test_markers(n_markers, 1_000);
        let samples = Arc::new(Samples::from_ids(
            (0..n_ref_samples).map(|i| format!("r{}", i)).collect(),
        ));
        let mut columns = Vec::with_capacity(n_markers);
        for m in 0..n_markers {
            let mut alleles = Vec::with_capacity(n_ref_haps);
            for h in 0..n_ref_haps {
                alleles.push(ref_haps[h][m]);
            }
            columns.push(GenotypeColumn::from_alleles(&alleles, 2));
        }
        let ref_gt = GenotypeMatrix::new_phased(markers, columns, samples);
        (ref_gt, hero_hap_idx)
    }

    fn build_target_with_sparse_anchors(
        n_markers: usize,
        hero_pattern: &[u8],
        anchor_every: usize,
    ) -> GenotypeMatrix<crate::data::storage::phase_state::Unphased, crate::data::AnyMarkerSpace>
    {
        use crate::data::haplotype::Samples;
        use crate::data::storage::GenotypeColumn;
        use std::sync::Arc;

        let markers = build_test_markers(n_markers, 1_000);
        let samples = Arc::new(Samples::from_ids(vec!["target".to_string()]));
        let mut columns = Vec::with_capacity(n_markers);
        let mut phase_mask = vec![vec![0u8; 1]; n_markers];

        for m in 0..n_markers {
            let hero = hero_pattern[m];
            let anti = 1 - hero;
            let (a1, a2) = if anchor_every > 0 && m % anchor_every == 0 {
                phase_mask[m][0] = 1;
                (hero, anti)
            } else {
                (0, 1)
            };
            columns.push(GenotypeColumn::from_alleles(&[a1, a2], 2));
        }

        GenotypeMatrix::new_unphased(markers, columns, samples).with_phase_mask(Some(phase_mask))
    }

    #[test]
    fn test_pipeline_creation() {
        let config = Config {
            target: PathBuf::from("test.vcf"),
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
            dynamic_k: 32,
            mcmc_steps: 3,
            mcmc_lr_samples: 32,
            phase_states: 280,
            rare: 0.002,
            impute: true,
            imp_states: 1600,
            imp_segment: 6.0,
            imp_step: 0.1,
            imp_nsteps: 7,
            cluster: 0.005,
            pbwt_batch_mb: 256,
            ap: false,
            gp: false,
            ne: 100000.0,
            err: None,
            em: false, // Disable EM for unit test to avoid complexity
            window: 40.0,
            window_markers: 100000,
            overlap: 2.0,
            seed: 12345,
            nthreads: None,
            profile: false,
        };

        let pipeline = PhasingPipeline::<crate::data::AnyMarkerSpace>::new(config, None);
        assert_eq!(pipeline.params.n_states, 280);
    }

    #[test]
    fn test_run_phase() {
        // Create a small pipeline and run phase_in_memory
        use crate::data::ChromIdx;
        use crate::data::genetic_map::GeneticMaps;
        use crate::data::haplotype::Samples;
        use crate::data::marker::{Allele, Marker, Markers};
        use crate::data::storage::GenotypeColumn;
        use crate::data::storage::matrix::GenotypeMatrix;
        use std::sync::Arc;

        let n_markers = 100;
        let n_samples = 10;
        use crate::data::marker::Nucleotide;

        // Mock Markers
        let mut markers = Markers::<crate::data::AnyMarkerSpace>::new();
        markers.add_chrom("chr1");

        for i in 0..n_markers {
            let m = Marker::new(
                ChromIdx::new(0),
                i as u32 * 1000,
                Some(format!("m{}", i).into()),
                Allele::Base(Nucleotide::A),
                vec![Allele::Base(Nucleotide::T)],
            );
            markers.push(m);
        }

        // Mock Samples
        let samples = Arc::new(Samples::from_ids(
            (0..n_samples).map(|i| format!("s{}", i)).collect(),
        ));

        // Mock Genotypes (Random)
        let columns: Vec<GenotypeColumn> = (0..n_markers)
            .map(|_| {
                let bytes: Vec<u8> = (0..n_samples * 2).map(|i| (i % 3) as u8).collect();
                GenotypeColumn::from_alleles(&bytes, 2)
            })
            .collect();

        let gt = GenotypeMatrix::new_unphased(markers, columns, samples);

        // Mock Genetic Map (Empty uses default linear rate)
        let gen_maps = GeneticMaps::new();

        let config = Config {
            target: PathBuf::from("test.vcf"),
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
            dynamic_k: 32,
            mcmc_steps: 3,
            mcmc_lr_samples: 32,
            phase_states: 10,
            rare: 0.002,
            impute: true,
            imp_states: 10,
            imp_segment: 6.0,
            imp_step: 0.1,
            imp_nsteps: 7,
            cluster: 0.005,
            pbwt_batch_mb: 256,
            ap: false,
            gp: false,
            ne: 10000.0,
            err: None,
            em: false,
            window: 40.0,
            window_markers: 100000,
            overlap: 2.0,
            seed: 12345,
            nthreads: Some(2),
            profile: false,
        };

        let mut pipeline = PhasingPipeline::<crate::data::AnyMarkerSpace>::new(config, None);

        // Run phasing (with no overlap from previous window)
        let result = pipeline.phase_in_memory_with_overlap(&gt, &gen_maps, None, None);

        assert!(result.is_ok());
        let (phased, _) = result.unwrap();
        assert_eq!(phased.n_markers(), n_markers);
        assert_eq!(phased.n_haplotypes(), n_samples * 2);

        // Check phase confidence values
        let mut total_hets = 0;
        let mut sum_conf = 0.0;
        let mut count_conf = 0;

        let mut any_conf_oob = false;
        let mut any_conf_nan = false;
        let mut het_per_marker: Vec<usize> = vec![0; n_markers];
        let mut conf_min = 1.0f32;
        let mut conf_max = 0.0f32;
        let mut first_bad_conf: Option<(usize, usize, f32)> = None;

        for m in 0..n_markers {
            let marker_idx = MarkerIdx::new(m as u32);
            let column = phased.column(marker_idx);

            for s in 0..n_samples {
                let sample_idx = crate::data::SampleIdx::new(s as u32);
                let hap1 = column.get(sample_idx.hap1());
                let hap2 = column.get(sample_idx.hap2());

                // Get phase confidence
                let conf = phased.sample_phase_confidence_f32(marker_idx, s);

                if conf.is_nan() {
                    any_conf_nan = true;
                    if first_bad_conf.is_none() {
                        first_bad_conf = Some((m, s, conf));
                    }
                }
                if conf < 0.0 || conf > 1.0 {
                    any_conf_oob = true;
                    if first_bad_conf.is_none() {
                        first_bad_conf = Some((m, s, conf));
                    }
                }
                if conf.is_finite() {
                    conf_min = conf_min.min(conf);
                    conf_max = conf_max.max(conf);
                }

                // Track heterozygous sites
                if hap1 != hap2 {
                    total_hets += 1;
                    het_per_marker[m] += 1;
                    sum_conf += conf;
                    count_conf += 1;
                }
            }
        }

        // Assert that most heterozygous sites have reasonable confidence
        if total_hets > 0 {
            let mean_conf = sum_conf / count_conf as f32;

            // For this unit test with random data and minimal iterations,
            // we just verify confidence values are computed and in valid range.
            // Real integration tests with actual data should have mean_conf > 0.8
            assert!(
                mean_conf >= 0.0 && mean_conf <= 1.0,
                "Mean phase confidence out of range: {:.3}",
                mean_conf
            );

            assert!(
                !any_conf_nan && !any_conf_oob,
                "Invalid confidence encountered: min={:.6} max={:.6} first_bad={:?}",
                conf_min,
                conf_max,
                first_bad_conf
            );
        }
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
        assert!(
            emit_match > 0.9,
            "Expected high emission when ref matches required allele, got {}",
            emit_match
        );

        // H2 = 0, so H1 must = 1. Reference has 0 -> low emission
        let emit_mismatch = emit_haploid_constrained(0, 0, 1, 0, conf, p_no_err, p_err);
        assert!(
            emit_mismatch < 0.1,
            "Expected low emission when ref doesn't match, got {}",
            emit_mismatch
        );

        // At homozygous site (fixed_allele = 255), H1 must match genotype
        let emit_hom = emit_haploid_constrained(0, 0, 0, 255, conf, p_no_err, p_err);
        assert!(
            emit_hom > 0.9,
            "Expected high emission at hom site when ref matches, got {}",
            emit_hom
        );

        let emit_hom_mismatch = emit_haploid_constrained(1, 0, 0, 255, conf, p_no_err, p_err);
        assert!(
            emit_hom_mismatch < 0.1,
            "Expected low emission at hom when ref doesn't match, got {}",
            emit_hom_mismatch
        );
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
        assert!(
            (emit_zero_conf - 0.5).abs() < 0.01,
            "Expected 0.5 with zero confidence, got {}",
            emit_zero_conf
        );

        // Half confidence: emission should be blend
        let emit_half_conf = emit_haploid_constrained(1, 0, 1, 0, 0.5, p_no_err, p_err);
        let expected = 0.5 * p_no_err + 0.5 * 0.5;
        assert!(
            (emit_half_conf - expected).abs() < 0.01,
            "Expected {}, got {}",
            expected,
            emit_half_conf
        );
    }

    #[test]
    fn test_emit_haploid_constrained_missing_ref_is_neutral() {
        let p_no_err = 0.999;
        let p_err = 0.001;
        let conf = 1.0;
        let emit = emit_haploid_constrained(255, 0, 1, 0, conf, p_no_err, p_err);
        assert!(
            (emit - 0.5).abs() < 1e-6,
            "Missing reference allele should be neutral, got {}",
            emit
        );
    }

    #[test]
    fn test_emit_haploid_constrained_inconsistent_fixed_relaxes_constraint() {
        let p_no_err = 0.999;
        let p_err = 0.001;
        let conf = 1.0;
        // fixed_allele=2 is inconsistent with genotype {0,1}, so either allele should match.
        let emit_ref0 = emit_haploid_constrained(0, 0, 1, 2, conf, p_no_err, p_err);
        let emit_ref1 = emit_haploid_constrained(1, 0, 1, 2, conf, p_no_err, p_err);
        let emit_ref2 = emit_haploid_constrained(2, 0, 1, 2, conf, p_no_err, p_err);
        assert!(
            emit_ref0 > 0.9,
            "Expected relaxed-match for ref=0, got {}",
            emit_ref0
        );
        assert!(
            emit_ref1 > 0.9,
            "Expected relaxed-match for ref=1, got {}",
            emit_ref1
        );
        assert!(
            emit_ref2 < 0.1,
            "Expected mismatch for ref=2, got {}",
            emit_ref2
        );
    }

    #[test]
    fn test_compute_pl_allele_probs_partner_polarity() {
        // Strong heterozygous PL: 0/1 is overwhelmingly likely.
        // PL ordering for biallelic sites is (0/0, 0/1, 1/1).
        let pl = [100u16, 0u16, 100u16];
        let mut allele_probs = Vec::new();

        let n = compute_pl_allele_probs(Some(&pl), false, 0, &mut allele_probs)
            .expect("expected biallelic PL to be parsed");
        assert_eq!(n, 2);
        assert!(
            allele_probs[1] > allele_probs[0],
            "partner=0 should favor target allele 1, got {:?}",
            allele_probs
        );

        let n = compute_pl_allele_probs(Some(&pl), false, 1, &mut allele_probs)
            .expect("expected biallelic PL to be parsed");
        assert_eq!(n, 2);
        assert!(
            allele_probs[0] > allele_probs[1],
            "partner=1 should favor target allele 0, got {:?}",
            allele_probs
        );

        // In combined mode, conditioning on partner should have no effect.
        compute_pl_allele_probs(Some(&pl), true, 0, &mut allele_probs)
            .expect("expected biallelic PL to be parsed");
        let probs_partner0 = allele_probs.clone();
        compute_pl_allele_probs(Some(&pl), true, 1, &mut allele_probs)
            .expect("expected biallelic PL to be parsed");
        let probs_partner1 = allele_probs.clone();
        assert!(
            probs_partner0
                .iter()
                .zip(probs_partner1.iter())
                .all(|(a, b)| (a - b).abs() < 1e-6),
            "combined emissions should be partner-invariant: {:?} vs {:?}",
            probs_partner0,
            probs_partner1
        );
    }

    #[test]
    fn test_prescan_hero_survives_across_sparse_anchor_windows() {
        let n_markers = 50;
        let hero_pattern: Vec<u8> = (0..n_markers).map(|m| (m % 2) as u8).collect();
        let (ref_gt, hero_hap_idx) =
            build_ref_panel_with_hero(n_markers, 50, 49, &hero_pattern, 42);
        let target_gt = build_target_with_sparse_anchors(n_markers, &hero_pattern, 10);

        let target_geno = MutableGenotypes::from_fn(n_markers, 2, |m, h| {
            target_gt.allele(MarkerIdx::new(m as u32), HapIdx::new(h as u32))
        });
        let alignment = MarkerAlignment::new(&target_gt, &ref_gt);
        let gen_positions: Vec<f64> = (0..n_markers).map(|i| i as f64 * 0.02).collect();

        let mut config = Config::default();
        config.phase_states = ref_gt.n_haplotypes();
        config.ne = 10000.0;
        config.err = Some(0.0001);
        config.nthreads = Some(1);
        let mut pipeline = PhasingPipeline::<crate::data::AnyMarkerSpace>::new(config, None);
        pipeline.params =
            ModelParams::for_phasing(ref_gt.n_haplotypes() + 2, 10000.0, Some(0.0001));
        pipeline
            .params
            .set_n_states(pipeline.config.phase_states.min(ref_gt.n_haplotypes()));

        let threaded = pipeline
            .build_phasing_prescan_states(
                &target_gt,
                &target_geno,
                Some(&ref_gt),
                Some(&alignment),
                n_markers,
                1,
                &gen_positions,
                0.1,
                None,
            )
            .expect("prescan");

        let th = &threaded[0];
        let expected_states = pipeline
            .config
            .phase_states
            .min(ref_gt.n_haplotypes())
            .max(1);
        assert_eq!(
            th.n_states(),
            expected_states,
            "ThreadedHaps state count should match phase_states cap"
        );
        let mut state_buf = vec![CombinedHapId::from(0u32); th.n_states()];
        let offset = target_geno.n_haps() as u32;
        let mut expected: Vec<u32> = (0..ref_gt.n_haplotypes())
            .map(|h| offset + h as u32)
            .collect();
        expected.sort_unstable();

        let markers = [0usize, n_markers / 2, n_markers - 1];
        for &m in &markers {
            th.materialize_at(m, &mut state_buf);
            let mut actual: Vec<u32> = state_buf.iter().map(|id| id.as_u32()).collect();
            actual.sort_unstable();
            let unique: std::collections::HashSet<u32> = actual.iter().copied().collect();
            assert_eq!(
                actual.len(),
                unique.len(),
                "ThreadedHaps contains duplicate ids at marker {}",
                m
            );
            assert_eq!(
                actual.len(),
                th.n_states(),
                "ThreadedHaps materialize_at size mismatch at marker {}",
                m
            );
            assert_eq!(
                actual, expected,
                "ThreadedHaps states differ from expected full ref set at marker {}",
                m
            );
        }
        let hero_combined = (offset as usize + hero_hap_idx) as u32;
        let contains_hero = expected.iter().any(|&id| id == hero_combined);
        println!(
            "[prescan test] states={} hero_present={}",
            th.n_states(),
            contains_hero
        );
    }

    #[test]
    fn test_sample_swap_bits_mosaic_two_state_anchor_stability() {
        let n_markers = 40;
        let hero_pattern: Vec<u8> = (0..n_markers).map(|m| (m % 2) as u8).collect();
        let (ref_gt, hero_hap_idx) = build_ref_panel_with_hero(n_markers, 1, 0, &hero_pattern, 99);

        println!("[mosaic anchor test] hero_hap_idx={}", hero_hap_idx);
        let mut th = ThreadedHaps::<CombinedHapSpace>::new(2, 2, n_markers);
        th.push_new(CombinedHapId::new(0));
        th.push_new(CombinedHapId::new(1));

        let seq1: Vec<u8> = vec![0; n_markers];
        let seq2: Vec<u8> = vec![1; n_markers];
        let conf: Vec<f32> = vec![1.0; n_markers];
        let mut anchor_h1 = vec![255u8; n_markers];
        let mut anchor_h2 = vec![255u8; n_markers];
        for m in (0..n_markers).step_by(10) {
            anchor_h1[m] = hero_pattern[m];
            anchor_h2[m] = 1 - hero_pattern[m];
        }
        // Anchor the last marker to prevent tail drift
        if n_markers > 0 {
            let last = n_markers - 1;
            anchor_h1[last] = hero_pattern[last];
            anchor_h2[last] = 1 - hero_pattern[last];
        }

        let p_recomb = vec![0.0001f32; n_markers];
        let block_starts: Arc<[usize]> = blocks_to_starts(&[(0, n_markers)], n_markers)
            .into_boxed_slice()
            .into();
        let het_positions: Vec<usize> = (0..n_markers).collect();
        let p_no_err = 0.999;
        let p_err = 1.0 - p_no_err;
        let mut workspace = crate::utils::workspace::ThreadWorkspace::new(8, 0);
        let ref_provider = RefAlleleProvider::new(GenotypeView::from(&ref_gt), &th);

        let (swap_bits, swap_lr, swap_probs, swap_probs_conf, paths) = sample_swap_bits_mosaic(
            n_markers,
            2,
            &p_recomb,
            &seq1,
            &seq2,
            &conf,
            ref_provider,
            None,
            block_starts,
            &het_positions,
            None,
            Some(&anchor_h1),
            Some(&anchor_h2),
            123,
            0,
            32,
            p_no_err,
            p_err,
            &mut workspace,
        );
        let swap_probs_conf_sum: f32 = swap_probs_conf.iter().sum();
        assert!(swap_probs_conf_sum.is_finite());

        let mut switches1 = 0usize;
        let mut switches2 = 0usize;
        for m in 1..n_markers {
            if paths.path1[m] != paths.path1[m - 1] {
                switches1 += 1;
            }
            if paths.path2[m] != paths.path2[m - 1] {
                switches2 += 1;
            }
        }
        let p_min = swap_probs.iter().cloned().fold(1.0f32, |a, b| a.min(b));
        let p_max = swap_probs.iter().cloned().fold(0.0f32, |a, b| a.max(b));
        let p_mean = swap_probs.iter().sum::<f32>() / swap_probs.len().max(1) as f32;
        println!(
            "[mosaic anchor test] switches1={} switches2={} p_min={:.3} p_mean={:.3} p_max={:.3} swap_bits={} swap_lr={}",
            switches1,
            switches2,
            p_min,
            p_mean,
            p_max,
            swap_bits.len(),
            swap_lr.len()
        );
        assert_eq!(
            switches1 + switches2,
            0,
            "Unexpected path switching with anchors in two-state model"
        );
    }

    #[test]
    fn test_prescan_scores_hero_in_anchor_window() {
        let n_markers = 50;
        let hero_pattern: Vec<u8> = (0..n_markers).map(|m| (m % 2) as u8).collect();
        let (ref_gt, hero_hap_idx) = build_ref_panel_with_hero(n_markers, 50, 49, &hero_pattern, 7);
        let target_gt = build_target_with_sparse_anchors(n_markers, &hero_pattern, 10);
        let target_geno = MutableGenotypes::from_fn(n_markers, 2, |m, h| {
            target_gt.allele(MarkerIdx::new(m as u32), HapIdx::new(h as u32))
        });
        let alignment = MarkerAlignment::new(&target_gt, &ref_gt);

        let gen_positions: Vec<f64> = (0..n_markers).map(|i| i as f64 * 0.02).collect();
        let windows = partition_markers_by_cm(&gen_positions, stage1_block_cm(&gen_positions));
        let window = windows[0];
        let per_window_cap = 20usize;
        let n_ref_haps = ref_gt.n_haplotypes();
        let sampling = build_sampling_points(
            &gen_positions[window.0..window.1],
            0.1,
            PBWT_MIN_MARKER_STEP,
            None,
        );
        let k_per_hap = per_window_cap
            .saturating_mul(PBWT_PER_WINDOW_MULT)
            .max(PBWT_MIN_PER_HAP)
            .min(PBWT_MAX_PER_HAP)
            .max(1)
            .min(n_ref_haps.max(1));
        let sampled = sampling.iter().filter(|&&b| b).count();
        println!(
            "[prescan score test] window={:?} sampled={}",
            window, sampled
        );

        let ref_columns: Vec<GenotypeColumn> = (0..n_markers)
            .map(|m| ref_gt.column(MarkerIdx::new(m as u32)).clone())
            .collect();
        let freqs = compute_ref_freqs(
            &target_gt,
            &ref_columns,
            Some(&alignment),
            None,
            None,
            n_markers,
        );
        let mut window_scores = vec![vec![f32::NEG_INFINITY; ref_gt.n_haplotypes()]; 2];
        score_window_batch_pbwt_segment(
            &[0, 1],
            &target_gt,
            &target_geno,
            &ref_columns,
            target_gt.phase_mask(),
            Some(&[true]),
            Some(&alignment),
            &freqs,
            window,
            k_per_hap,
            &sampling,
            &mut window_scores,
            false,
            None,
            None,
        );

        let hero_score = window_scores[0][hero_hap_idx];
        let top = select_top_k(&window_scores[0], 15);
        let in_top = top.iter().any(|(h, _)| *h == hero_hap_idx);
        println!(
            "[prescan score test] top={:?}",
            top.iter()
                .map(|(h, s)| (*h, (*s * 1000.0).round() / 1000.0))
                .collect::<Vec<_>>()
        );
        println!(
            "[prescan score test] hero_hap={} score={:.3} in_top10={}",
            hero_hap_idx, hero_score, in_top
        );
        assert!(
            in_top,
            "Hero hap {} not in top-10 prescan scores for anchor window",
            hero_hap_idx
        );
    }

    #[test]
    fn test_allocator_keeps_hero_from_prescan_scores() {
        let n_markers = 50;
        let hero_pattern: Vec<u8> = (0..n_markers).map(|m| (m % 2) as u8).collect();
        let (ref_gt, hero_hap_idx) =
            build_ref_panel_with_hero(n_markers, 50, 49, &hero_pattern, 11);
        let target_gt = build_target_with_sparse_anchors(n_markers, &hero_pattern, 10);
        let target_geno = MutableGenotypes::from_fn(n_markers, 2, |m, h| {
            target_gt.allele(MarkerIdx::new(m as u32), HapIdx::new(h as u32))
        });
        let alignment = MarkerAlignment::new(&target_gt, &ref_gt);
        let gen_positions: Vec<f64> = (0..n_markers).map(|i| i as f64 * 0.02).collect();
        let windows = partition_markers_by_cm(&gen_positions, stage1_block_cm(&gen_positions));
        let num_windows = windows.len();
        let per_window_cap = 20usize;

        let n_ref_haps = ref_gt.n_haplotypes();
        let ref_columns: Vec<GenotypeColumn> = (0..n_markers)
            .map(|m| ref_gt.column(MarkerIdx::new(m as u32)).clone())
            .collect();
        let freqs = compute_ref_freqs(
            &target_gt,
            &ref_columns,
            Some(&alignment),
            None,
            None,
            n_markers,
        );

        let mut scores_by_window_by_hap: Vec<Vec<Vec<(usize, f32)>>> =
            vec![Vec::with_capacity(num_windows); 2];
        let phase_mask = target_gt.phase_mask();
        let mut informative: Vec<bool> = Vec::new();
        for &(start, end) in &windows {
            informative.clear();
            informative.resize(end.saturating_sub(start), false);
            if !informative.is_empty() {
                for m in start..end {
                    let mut info = false;
                    if let Some(mask) = phase_mask {
                        if mask.row_has_any_set(m) {
                            info = true;
                        }
                    }
                    if !info {
                        let alleles = target_geno.marker_alleles(m);
                        if alleles.iter().any(|&a| a != 255) {
                            info = true;
                        }
                    }
                    informative[m - start] = info;
                }
            }
            let sampling = build_sampling_points(
                &gen_positions[start..end],
                0.1,
                PBWT_MIN_MARKER_STEP,
                Some(&informative),
            );
            let k_per_hap = per_window_cap
                .saturating_mul(PBWT_PER_WINDOW_MULT)
                .max(PBWT_MIN_PER_HAP)
                .min(PBWT_MAX_PER_HAP)
                .max(1)
                .min(n_ref_haps.max(1));
            let mut window_scores = vec![vec![f32::NEG_INFINITY; n_ref_haps]; 2];
            score_window_batch_pbwt_segment(
                &[0, 1],
                &target_gt,
                &target_geno,
                &ref_columns,
                phase_mask,
                Some(&[true]),
                Some(&alignment),
                &freqs,
                (start, end),
                k_per_hap,
                &sampling,
                &mut window_scores,
                false,
                None,
                None,
            );
            for hap_idx in 0..2 {
                let top = select_top_k(&window_scores[hap_idx], 160.min(n_ref_haps.max(1)));
                scores_by_window_by_hap[hap_idx].push(top);
            }
        }

        let mut dense_merge_buffer = vec![f32::NEG_INFINITY; n_ref_haps.max(1)];
        let mut touched_indices: Vec<usize> = Vec::new();
        let mut window_scores: Vec<Vec<(usize, f32)>> = Vec::with_capacity(num_windows);
        let mut prev_window_scores: Vec<(usize, f32)> = Vec::new();
        for w in 0..num_windows {
            for &idx in &touched_indices {
                dense_merge_buffer[idx] = f32::NEG_INFINITY;
            }
            touched_indices.clear();

            for &(h, score) in scores_by_window_by_hap[0][w]
                .iter()
                .chain(scores_by_window_by_hap[1][w].iter())
            {
                if h >= dense_merge_buffer.len() {
                    continue;
                }
                let current = &mut dense_merge_buffer[h];
                if current.is_finite() {
                    if score > *current {
                        *current = score;
                    }
                } else {
                    *current = score;
                    touched_indices.push(h);
                }
            }

            touched_indices.sort_by(|&a, &b| {
                dense_merge_buffer[b]
                    .partial_cmp(&dense_merge_buffer[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let cap = 160.min(n_ref_haps.max(1));
            let take = cap.min(touched_indices.len());
            let mut list: Vec<(usize, f32)> = Vec::with_capacity(take);
            for &h in touched_indices.iter().take(take) {
                list.push((h, dense_merge_buffer[h]));
            }
            if list.is_empty() {
                list.extend(prev_window_scores.iter().copied());
            } else if !prev_window_scores.is_empty() {
                let mut map: HashMap<usize, f32> = list.iter().copied().collect();
                for (h, score) in prev_window_scores.iter().copied() {
                    map.entry(h).or_insert(score);
                }
                list = map.into_iter().collect();
            }
            prev_window_scores = list.clone();
            window_scores.push(list);
        }

        let abyss = vec![false; n_ref_haps];
        let (candidate_haps, scores_by_hap) = build_sparse_scores(&window_scores, &abyss);
        let hero_in_candidates = candidate_haps.iter().any(|&h| h == hero_hap_idx);
        println!(
            "[allocator test] candidates={} hero_in_candidates={}",
            candidate_haps.len(),
            hero_in_candidates
        );
        assert!(
            hero_in_candidates,
            "Hero hap {} missing from candidate set before allocation",
            hero_hap_idx
        );

        let per_window_cap = 20usize;
        let per_window_caps = vec![per_window_cap; num_windows];
        let global_slot_budget = per_window_caps.iter().copied().sum::<usize>().max(1);
        let mut boundary_cm = Vec::with_capacity(num_windows.saturating_sub(1));
        for w in 0..num_windows.saturating_sub(1) {
            let (_, end) = windows[w];
            let (next_start, _) = windows[w + 1];
            let left = gen_positions[end.saturating_sub(1).min(gen_positions.len() - 1)];
            let right = gen_positions[next_start.min(gen_positions.len() - 1)];
            boundary_cm.push((right - left).abs().max(0.1));
        }
        let mut config = Config::default();
        config.phase_states = 20;
        config.ne = 10000.0;
        config.err = Some(0.0001);
        config.nthreads = Some(1);
        let mut pipeline = PhasingPipeline::<crate::data::AnyMarkerSpace>::new(config, None);
        pipeline.params = ModelParams::for_phasing(n_ref_haps + 2, 10000.0, Some(0.0001));
        let params = pipeline.params.clone();
        let allocation = allocate_lms_sparse(
            &scores_by_hap,
            &candidate_haps,
            num_windows,
            &boundary_cm,
            &params,
            n_ref_haps,
            global_slot_budget,
            &per_window_caps,
        );
        let mut selected: Vec<usize> = allocation
            .intervals_by_hap
            .into_iter()
            .map(|(h, _)| h)
            .collect();
        selected.sort_unstable();
        selected.dedup();
        eprintln!(
            "[threaded test] expected_ref_len={} first={:?}",
            selected.len(),
            selected.iter().take(10).collect::<Vec<_>>()
        );
        eprintln!(
            "[threaded test] expected_ref_len={} first={:?}",
            selected.len(),
            selected.iter().take(10).collect::<Vec<_>>()
        );

        if PBWT_FORCE_TOP_HAPS > 0 && !window_scores.is_empty() {
            let mut dense_merge_buffer = vec![f32::NEG_INFINITY; n_ref_haps.max(1)];
            let mut touched_indices: Vec<usize> = Vec::new();
            for list in &window_scores {
                for &(h, score) in list {
                    if h >= dense_merge_buffer.len() {
                        continue;
                    }
                    let current = &mut dense_merge_buffer[h];
                    if current.is_finite() {
                        if score > *current {
                            *current = score;
                        }
                    } else {
                        *current = score;
                        touched_indices.push(h);
                    }
                }
            }
            touched_indices.sort_by(|&a, &b| {
                dense_merge_buffer[b]
                    .partial_cmp(&dense_merge_buffer[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let take = PBWT_FORCE_TOP_HAPS.min(touched_indices.len());
            for &h in touched_indices.iter().take(take) {
                if !selected.contains(&h) {
                    selected.push(h);
                }
            }
        }

        if PBWT_ANCHOR_TOP_HAPS > 0 {
            let mut anchors_by_hap: Vec<Vec<(usize, u8, u8)>> = vec![Vec::new(); 2];
            let phase_mask = target_gt.phase_mask();
            for s in 0..1usize {
                let hap1 = s * 2;
                let hap2 = hap1 + 1;
                for m in 0..n_markers {
                    let phased = phase_mask.and_then(|mask| mask.get(m, s)).unwrap_or(0);
                    if phased == 0 {
                        continue;
                    }
                    let a1 = target_geno.get(m, HapIdx::new(hap1 as u32));
                    let a2 = target_geno.get(m, HapIdx::new(hap2 as u32));
                    if a1 == 255 || a2 == 255 || a1 == a2 {
                        continue;
                    }
                    anchors_by_hap[hap1].push((m, a1, a2));
                    anchors_by_hap[hap2].push((m, a2, a1));
                }
            }

            let mut anchor_scores = vec![0i32; n_ref_haps];
            for &(start, end) in &windows {
                let mut window_anchors: Vec<(usize, u8, u8)> = Vec::new();
                for &(m, a1, a2) in anchors_by_hap[0].iter().chain(anchors_by_hap[1].iter()) {
                    if m >= start && m < end {
                        window_anchors.push((m, a1, a2));
                    }
                }
                if window_anchors.is_empty() {
                    continue;
                }
                anchor_scores.fill(0);
                for h in 0..n_ref_haps {
                    let hap_idx = HapIdx::new(h as u32);
                    let mut score = 0i32;
                    for (m, a1, _) in &window_anchors {
                        let ref_al = ref_columns.get(*m).map(|c| c.get(hap_idx)).unwrap_or(255);
                        if ref_al == 255 {
                            continue;
                        }
                        if ref_al == *a1 {
                            score += 1;
                        } else {
                            score -= 1;
                        }
                    }
                    anchor_scores[h] = score;
                }
                let mut idxs: Vec<usize> = (0..n_ref_haps).collect();
                idxs.sort_by(|&a, &b| anchor_scores[b].cmp(&anchor_scores[a]));
                let take = PBWT_ANCHOR_TOP_HAPS.min(idxs.len());
                for &h in idxs.iter().take(take) {
                    if !selected.contains(&h) {
                        selected.push(h);
                    }
                }
            }
        }
        let hero_selected = selected.iter().any(|&h| h == hero_hap_idx);
        println!(
            "[allocator test] selected={} hero_selected={}",
            selected.len(),
            hero_selected
        );
        assert!(
            hero_selected,
            "Hero hap {} dropped by allocator despite positive scores",
            hero_hap_idx
        );
    }

    #[test]
    fn test_threaded_haps_matches_selected_indices() {
        let n_markers = 50;
        let hero_pattern: Vec<u8> = (0..n_markers).map(|m| (m % 2) as u8).collect();
        let (ref_gt, hero_hap_idx) =
            build_ref_panel_with_hero(n_markers, 50, 49, &hero_pattern, 13);
        let target_gt = build_target_with_sparse_anchors(n_markers, &hero_pattern, 10);
        let target_geno = MutableGenotypes::from_fn(n_markers, 2, |m, h| {
            target_gt.allele(MarkerIdx::new(m as u32), HapIdx::new(h as u32))
        });
        let alignment = MarkerAlignment::new(&target_gt, &ref_gt);
        let gen_positions: Vec<f64> = (0..n_markers).map(|i| i as f64 * 0.02).collect();

        let mut config = Config::default();
        config.phase_states = ref_gt.n_haplotypes();
        config.ne = 10000.0;
        config.err = Some(0.0001);
        config.nthreads = Some(1);
        let mut pipeline = PhasingPipeline::<crate::data::AnyMarkerSpace>::new(config, None);
        pipeline.params =
            ModelParams::for_phasing(ref_gt.n_haplotypes() + 2, 10000.0, Some(0.0001));
        pipeline
            .params
            .set_n_states(pipeline.config.phase_states.min(ref_gt.n_haplotypes()));

        let threaded = pipeline
            .build_phasing_prescan_states(
                &target_gt,
                &target_geno,
                Some(&ref_gt),
                Some(&alignment),
                n_markers,
                1,
                &gen_positions,
                0.1,
                None,
            )
            .expect("prescan");
        let th = &threaded[0];
        let expected_states = pipeline
            .config
            .phase_states
            .min(ref_gt.n_haplotypes())
            .max(1);
        assert_eq!(
            th.n_states(),
            expected_states,
            "ThreadedHaps state count should match phase_states cap"
        );
        let mut state_buf = vec![CombinedHapId::from(0u32); th.n_states()];
        let offset = target_geno.n_haps() as u32;
        let mut expected: Vec<u32> = (0..ref_gt.n_haplotypes())
            .map(|h| offset + h as u32)
            .collect();
        expected.sort_unstable();
        let markers = [0usize, n_markers / 2, n_markers - 1];
        for &m in &markers {
            th.materialize_at(m, &mut state_buf);
            let mut actual: Vec<u32> = state_buf.iter().map(|id| id.as_u32()).collect();
            actual.sort_unstable();
            let unique: std::collections::HashSet<u32> = actual.iter().copied().collect();
            assert_eq!(
                actual.len(),
                unique.len(),
                "ThreadedHaps contains duplicate ids at marker {}",
                m
            );
            assert_eq!(
                actual.len(),
                th.n_states(),
                "ThreadedHaps materialize_at size mismatch at marker {}",
                m
            );
            assert_eq!(
                actual, expected,
                "ThreadedHaps states differ from expected full ref set at marker {}",
                m
            );
        }
        let hero_combined = (offset as usize + hero_hap_idx) as u32;
        let contains_hero = expected.iter().any(|&id| id == hero_combined);
        println!(
            "[threaded test] states={} hero_present={}",
            th.n_states(),
            contains_hero
        );
    }

    #[test]
    fn test_refresh_path_ref_from_states_updates_all_valid_markers() {
        let mut path_ref = vec![0u32, 0u32, 0u32, 0u32];
        let path_idx = vec![0u32, 1u32, 2u32, 1u32];
        let neighbors = vec![10u32, 11u32];

        refresh_path_ref_from_states(&mut path_ref, &path_idx, &neighbors);

        assert_eq!(path_ref[0], 10);
        assert_eq!(path_ref[1], 11);
        // Invalid state index should leave the previous value intact.
        assert_eq!(path_ref[2], 0);
        assert_eq!(path_ref[3], 11);
    }

    #[test]
    fn test_dynamic_mcmc_deterministic_phase() {
        // Create a scenario where the correct phase is deterministic:
        // Target sample (haps 0-1) with het genotype {0, 1}
        // Reference haplotypes (haps 2-9) all have allele 0
        // The HMM should set H1 = 0 (matching reference majority)
        use crate::model::ibs2::Ibs2;
        use crate::model::phase_ibs::BidirectionalPhaseIbs;

        let n_markers = 10;
        let n_target_haps = 2; // Sample 0: haplotypes 0 and 1
        let n_ref_haps = 8; // Reference: haplotypes 2-9
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
        let subset_to_global: Vec<usize> = (0..n_markers).collect();
        let alleles_flat: Vec<u8> = alleles.into_iter().flatten().collect();
        let phase_ibs = BidirectionalPhaseIbs::build_for_subset_flat(
            alleles_flat,
            n_total_haps,
            n_markers,
            &subset_to_global,
        );

        // Empty IBS2 - need at least 1 sample for the structure
        let ibs2 = Ibs2::empty(1);

        // Genotype: het at all sites (0/1)
        let seq1 = vec![0u8; n_markers];
        let seq2 = vec![1u8; n_markers];
        let conf = vec![1.0f32; n_markers];
        let phase_conf = vec![1.0f32; n_markers];
        let sample_phase_stability = vec![0.5f32];

        // p_recomb: low recombination
        let p_recomb = vec![0.01f32; n_markers];

        let het_positions: Vec<usize> = (0..n_markers).collect();

        // Sample 0: haplotypes 0 and 1
        let mut workspace = crate::utils::workspace::ThreadWorkspace::new(64, 0);
        let (swap_bits, swap_lr, swap_probs, swap_probs_conf, paths) = sample_dynamic_mcmc(
            n_markers,
            n_total_haps,
            &p_recomb,
            &seq1,
            &seq2,
            &conf,
            &phase_conf,
            &phase_ibs,
            &ibs2,
            &subset_to_global,
            0, // sample_idx = 0 (haplotypes 0 and 1)
            &sample_phase_stability,
            &het_positions,
            12345, // seed
            5,     // n_mcmc_steps
            0.999,
            0.001,
            None,
            None,
            None,
            None,
            &mut workspace,
        );
        let swap_probs_conf_sum: f32 = swap_probs_conf.iter().sum();
        assert!(swap_probs_conf_sum.is_finite());
        assert_eq!(paths.path1.len(), n_markers);
        assert_eq!(paths.path2.len(), n_markers);

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
        assert!(swap_probs.len() <= het_positions.len());
    }

    #[test]
    fn test_dynamic_mcmc_opposite_phase() {
        // Target sample (haps 0-1) with het genotype {0, 1}
        // Reference haplotypes (haps 2-9) all have allele 1
        // The HMM should set H1 = 1 (matching reference) -> swap needed
        use crate::model::ibs2::Ibs2;
        use crate::model::phase_ibs::BidirectionalPhaseIbs;

        let n_markers = 10;
        let n_target_haps = 2; // Sample 0: haplotypes 0 and 1
        let n_ref_haps = 8; // Reference: haplotypes 2-9
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
        let subset_to_global: Vec<usize> = (0..n_markers).collect();
        let alleles_flat: Vec<u8> = alleles.into_iter().flatten().collect();
        let phase_ibs = BidirectionalPhaseIbs::build_for_subset_flat(
            alleles_flat,
            n_total_haps,
            n_markers,
            &subset_to_global,
        );

        let ibs2 = Ibs2::empty(1);

        // Genotype: het at all sites (0/1)
        let seq1 = vec![0u8; n_markers];
        let seq2 = vec![1u8; n_markers];
        let conf = vec![1.0f32; n_markers];
        let phase_conf = vec![1.0f32; n_markers];
        let sample_phase_stability = vec![0.5f32];
        let p_recomb = vec![0.01f32; n_markers];
        let het_positions: Vec<usize> = (0..n_markers).collect();

        let mut workspace = crate::utils::workspace::ThreadWorkspace::new(64, 0);
        let (swap_bits, swap_lr, swap_probs, swap_probs_conf, paths) = sample_dynamic_mcmc(
            n_markers,
            n_total_haps,
            &p_recomb,
            &seq1,
            &seq2,
            &conf,
            &phase_conf,
            &phase_ibs,
            &ibs2,
            &subset_to_global,
            0, // sample_idx = 0 (haplotypes 0 and 1)
            &sample_phase_stability,
            &het_positions,
            12345,
            5,
            0.999,
            0.001,
            None,
            None,
            None,
            None,
            &mut workspace,
        );
        let swap_probs_conf_sum: f32 = swap_probs_conf.iter().sum();
        assert!(swap_probs_conf_sum.is_finite());
        assert_eq!(paths.path1.len(), n_markers);
        assert_eq!(paths.path2.len(), n_markers);

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
        assert!(swap_probs.len() <= het_positions.len());
    }

    #[test]
    fn test_find_best_constant_pair() {
        use crate::data::storage::MutableGenotypes;
        use crate::model::states::ThreadedHaps;

        let n_markers = 3;
        let n_states = 4;

        // Mock lookup
        // State 0: 0, 0, 0 (Matches Hero)
        // State 1: 1, 1, 1 (Matches Anti-Hero)
        // State 2: 0, 1, 0
        // State 3: 1, 0, 1

        // Target: 0/1 (Het) everywhere.
        // Seq1: 0, 0, 0
        // Seq2: 1, 1, 1
        // (This is one possible phasing of 0/1)

        let mut data = Vec::new();
        // M0
        data.extend_from_slice(&[0, 1, 0, 1]);
        // M1
        data.extend_from_slice(&[0, 1, 1, 0]);
        // M2
        data.extend_from_slice(&[0, 1, 0, 1]);

        let geno = MutableGenotypes::from_fn(n_markers, n_states, |m, h| data[m * n_states + h]);
        let mut threaded = ThreadedHaps::<CombinedHapSpace>::new(n_states, n_states, n_markers);
        for h in 0..n_states {
            threaded.push_new(CombinedHapId::new(h as u32));
        }
        let mut ref_provider: RefAlleleProvider<'_, AnyMarkerSpace, AnyMarkerSpace> =
            RefAlleleProvider::new(GenotypeView::Mutable(&geno), &threaded);

        let seq1 = vec![0, 0, 0];
        let seq2 = vec![1, 1, 1];
        let conf = vec![1.0; n_markers];

        let mut scores = Vec::new();
        let paths = find_best_constant_pair_with_buffer(
            n_markers,
            n_states,
            &seq1,
            &seq2,
            &conf,
            0.999,
            0.001,
            &mut ref_provider,
            None,
            &mut scores,
            None,
        )
        .unwrap();

        // Best pair should be (0, 1) or (1, 0) - Score 3.
        // Or (2, 3) / (3, 2).

        println!("Selected pair: ({}, {})", paths.path1[0], paths.path2[0]);

        assert!(
            (paths.path1[0] == 1 && paths.path2[0] == 0)
                || (paths.path1[0] == 0 && paths.path2[0] == 1)
                || (paths.path1[0] == 3 && paths.path2[0] == 2)
                || (paths.path1[0] == 2 && paths.path2[0] == 3)
        );
    }

    #[test]
    fn test_find_best_constant_pair_long_window_uses_sparse_eval() {
        use crate::data::storage::MutableGenotypes;
        use crate::model::states::ThreadedHaps;

        let n_markers = 4001;
        let n_states = 4;

        // Build two complementary "hero" haplotypes and two distractors.
        let geno = MutableGenotypes::from_fn(n_markers, n_states, |m, h| match h {
            0 => (m % 2) as u8,
            1 => 1u8 - (m % 2) as u8,
            2 => 0,
            _ => 1,
        });
        let mut threaded = ThreadedHaps::<CombinedHapSpace>::new(n_states, n_states, n_markers);
        for h in 0..n_states {
            threaded.push_new(CombinedHapId::new(h as u32));
        }
        let mut ref_provider: RefAlleleProvider<'_, AnyMarkerSpace, AnyMarkerSpace> =
            RefAlleleProvider::new(GenotypeView::Mutable(&geno), &threaded);

        // Fully heterozygous target.
        let seq1 = vec![0u8; n_markers];
        let seq2 = vec![1u8; n_markers];
        let conf = vec![1.0f32; n_markers];
        let mut scores = Vec::new();

        let paths = find_best_constant_pair_with_buffer(
            n_markers,
            n_states,
            &seq1,
            &seq2,
            &conf,
            0.999,
            0.001,
            &mut ref_provider,
            None,
            &mut scores,
            None,
        );
        assert!(
            paths.is_some(),
            "long-window heuristic should not be disabled"
        );
    }

    #[test]
    fn test_fit_cohort_calibration_small_n_skips() {
        let stats = vec![SampleCohortStats::default(); 10];
        let out = fit_cohort_calibration(&stats, 0.001);
        assert!(out.is_none());
    }

    #[test]
    fn test_fit_cohort_calibration_produces_per_sample_rates() {
        let mut stats = Vec::new();
        for _ in 0..70 {
            stats.push(SampleCohortStats {
                mismatch_mass: 5.0,
                emission_mass: 5000.0,
                expected_switches: 15.0,
                genetic_dist_morgans: 1.2,
                phase_uncertainty_sum: 80.0,
                phase_uncertainty_count: 200,
            });
        }
        for _ in 0..70 {
            stats.push(SampleCohortStats {
                mismatch_mass: 200.0,
                emission_mass: 5000.0,
                expected_switches: 60.0,
                genetic_dist_morgans: 1.2,
                phase_uncertainty_sum: 120.0,
                phase_uncertainty_count: 200,
            });
        }

        let out = fit_cohort_calibration(&stats, 0.001).expect("expected calibration model");
        assert_eq!(out.sample_p_mismatch.len(), stats.len());
        assert!(!out.cohort_p_mismatch.is_empty());
        let min_p = out
            .sample_p_mismatch
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        let max_p = out
            .sample_p_mismatch
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(min_p.is_finite() && max_p.is_finite());
        assert!(max_p > min_p);
    }
}
