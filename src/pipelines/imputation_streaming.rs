//! Streaming Imputation Pipeline
//!
//! Implements memory-efficient streaming imputation through overlapping windows.
//! Uses a producer-consumer model with MPSC channel to pipe phased matrices
//! directly to imputation in-memory.

use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::io::{BufRead, Write};
#[cfg(unix)]
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use bitvec::prelude::*;
use memmap2::{Mmap, MmapOptions};
use rayon::prelude::*;
use tracing::{info_span, instrument, warn};

use crate::Config;
use crate::data::alignment::MarkerAlignment;
use crate::data::genetic_map::GeneticMaps;
use crate::data::marker::{AnyMarkerSpace, Markers, RefWindowSpace};
use crate::data::storage::phase_state::{PhaseState, Phased};
use crate::data::storage::{GenotypeColumn, GenotypeMatrix};
use crate::data::{ChromIdx, HapIdx, HapSide, MarkerIdx, SampleIdx};
use crate::error::ReagleError;
use crate::error::Result;
use crate::io::bref3::{RefPanelReader, RefWindow, TargetMarkerIndex, convert_ref_vcf_to_bref3};
use crate::io::prescan_cache::{
    PackedRefColumn, PrescanCacheReader, PrescanCacheWriter, create_temp_cache_path,
    pack_ref_columns,
};
use crate::io::streaming::{
    GlobalHapId, GlobalMarkerIdx, HaplotypePriors, PhasedOverlap, StreamingConfig,
    StreamingVcfReader,
};
use crate::io::vcf::{ImputationQuality, VcfWriter};
use crate::model::impute_hmm::{
    ImputeHmmContext, ImputeWorkspace, RefAlleleFreqs, TargetAlleleProbs,
    compute_nearest_observed_lambda, run_impute_hmm, state_posteriors_to_priors,
};
use crate::model::parameters::ModelParams;
use crate::model::phase_query::{
    build_peer_indices, pbwt_beam_uncertainty, phase_best_orientation_error,
    phase_orientation_weight, phase_query_orientation_error_limit,
    uncertain_orientation_wildcard_info_weight,
};
use crate::model::pl_emission::{
    allele_probs_cond_from_pl, allele_probs_uncond_from_pl, genotype_probs_from_pl,
    infer_n_alleles_from_pl_len,
};
use crate::model::reference_pbwt::{
    DonorPick, PbwtQueryAllele, PbwtStrictAllele, RankBeam, ReferencePbwt,
};
use crate::model::transition_matrix::TransitionMatrix;
use crate::model::types::RefHapId;
use crate::pipelines::imputation::AllelePosteriors;
use crate::utils::telemetry::TelemetryBlackboard;

/// Retain only the `k` highest-weight donors, discarding the rest.
///
/// Uses `select_nth_unstable_by` for O(n) partitioning followed by a
/// local O(k log k) sort of the survivors, instead of a full O(n log n)
/// sort over all donors.
///
/// **Accuracy note**: Truncating low-weight donors also acts as a
/// denoising filter for the downstream HMM. By concentrating the
/// posterior probability budget on haplotypes with high continuity-
/// weighted evidence mass, the HMM avoids diluting its
/// allele posteriors across hundreds of weakly-matching reference
/// haplotypes. Empirically this improved overall R² by +0.0057 and
/// SEN by +0.00086 on the Kat benchmark (IQA run #1371 vs base).
#[inline]
fn donor_weight_cmp_desc(a: &(RefHapId, f32), b: &(RefHapId, f32)) -> std::cmp::Ordering {
    b.1.total_cmp(&a.1)
        .then_with(|| a.0.as_u32().cmp(&b.0.as_u32()))
}

#[inline]
fn keep_top_k_donors_by_weight(donors: &mut Vec<(RefHapId, f32)>, k: usize) {
    if donors.len() <= k {
        donors.sort_unstable_by(donor_weight_cmp_desc);
        return;
    }
    let split = k.max(1).min(donors.len());
    let (top, _, _) = donors.select_nth_unstable_by(split - 1, donor_weight_cmp_desc);
    top.sort_unstable_by(donor_weight_cmp_desc);
    donors.truncate(split);
}

/// Retain only the `k` highest-probability prior haplotype candidates.
///
/// Same O(n + k log k) strategy as [`keep_top_k_donors_by_weight`].
/// Discarding low-probability prior candidates prevents the HMM state
/// builder from wasting capacity on haplotypes that contribute negligible
/// posterior mass, sharpening the resulting allele posteriors.
#[inline]
fn keep_top_k_haps_by_prob(weighted: &mut Vec<(RefHapId, f32)>, k: usize) {
    if weighted.len() <= k {
        weighted
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        return;
    }
    let split = k.max(1).min(weighted.len());
    let (top, _, _) = weighted.select_nth_unstable_by(split - 1, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    top.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    weighted.truncate(split);
}

fn push_unique(dst: &mut Vec<String>, value: String) {
    if !dst.iter().any(|v| v == &value) {
        dst.push(value);
    }
}

fn chrom_variants(chrom: &str) -> Vec<String> {
    let mut candidates = Vec::new();
    push_unique(&mut candidates, chrom.to_string());
    let lower = chrom.to_ascii_lowercase();
    if lower.starts_with("chr") && chrom.len() >= 3 {
        let stripped = chrom[3..].to_string();
        if !stripped.is_empty() {
            push_unique(&mut candidates, stripped.clone());
            push_unique(&mut candidates, format!("chr{}", stripped));
            push_unique(&mut candidates, format!("CHR{}", stripped));
        }
    } else {
        push_unique(&mut candidates, format!("chr{}", chrom));
        push_unique(&mut candidates, format!("CHR{}", chrom));
    }
    candidates
}

#[inline]
fn fill_ref_alleles(col: &GenotypeColumn, out: &mut [u8]) {
    col.fill_all(out);
}

#[inline]
fn is_represented_in_states(state_haps: &[RefHapId], col: &GenotypeColumn, allele: u8) -> bool {
    for &hap in state_haps {
        if col.get(HapIdx::new(hap.as_u32())) == allele {
            return true;
        }
    }
    false
}

fn collect_carriers_for_allele(
    col: &GenotypeColumn,
    allele: u8,
    limit: usize,
    out: &mut Vec<RefHapId>,
) {
    out.clear();
    if limit == 0 {
        return;
    }

    let missing_raw = crate::data::storage::AlleleCode::MISSING.raw();

    match col {
        GenotypeColumn::Dense(d) if d.bits_per_allele() == 1 && allele <= 1 => {
            let n = d.n_haplotypes();
            let bits = d.bits_raw();
            let missing = d.missing_raw();
            let words = (n + 63) / 64;
            let want_alt = allele == 1;
            for w in 0..words {
                let bit_word = bits.get(w).copied().unwrap_or(0);
                let miss_word = missing.get(w).copied().unwrap_or(0);
                let mut active = if want_alt {
                    bit_word & !miss_word
                } else {
                    !bit_word & !miss_word
                };
                if w + 1 == words {
                    let tail = n % 64;
                    if tail != 0 {
                        let mask = (1u64 << tail) - 1;
                        active &= mask;
                    }
                }
                while active != 0 {
                    let bit = active.trailing_zeros() as usize;
                    if out.len() < limit {
                        out.push(RefHapId::new((w * 64 + bit) as u32));
                    }
                    if out.len() >= limit {
                        return;
                    }
                    active &= active - 1;
                }
            }
        }
        GenotypeColumn::Sparse(s) if allele <= 1 => {
            let carriers = s.carriers();
            if (!s.is_inverted() && allele == 1) || (s.is_inverted() && allele == 0) {
                for &hap in carriers.iter().take(limit) {
                    if out.len() < limit {
                        out.push(RefHapId::new(hap.as_usize() as u32));
                    }
                }
                return;
            }
            let n = s.n_haplotypes();
            for h in 0..n {
                if col.get(HapIdx::new(h as u32)) == allele {
                    if out.len() < limit {
                        out.push(RefHapId::new(h as u32));
                    }
                    if out.len() >= limit {
                        return;
                    }
                }
            }
        }
        _ => {
            let n = col.n_haplotypes();
            for h in 0..n {
                let a = col.get(HapIdx::new(h as u32));
                if a == missing_raw {
                    continue;
                }
                if a == allele {
                    if out.len() < limit {
                        out.push(RefHapId::new(h as u32));
                    }
                    if out.len() >= limit {
                        return;
                    }
                }
            }
        }
    }
}

const PBWT_SELECT_BLOCK_CM: f64 = 0.1;
const PBWT_PER_WINDOW_MULT: usize = 8;
const PBWT_MIN_PER_HAP: usize = 64;
const PBWT_MAX_PER_HAP: usize = 256;
const PBWT_MIN_MARKER_STEP: usize = 50;
const PBWT_MIN_SAMPLE_POINTS: usize = 10;
const PBWT_TYPED_ANCHORS_PER_BIN: usize = 1;
const PRESCAN_TOPM_WEAK_MULT_NUM: usize = 3;
const PRESCAN_TOPM_WEAK_MULT_DEN: usize = 2;
const PRESCAN_TOPM_STRONG_MULT_NUM: usize = 3;
const PRESCAN_TOPM_STRONG_MULT_DEN: usize = 4;
const IMPUTE_RAM_FRACTION: f64 = 0.4;
const STATE_BUDGET_SAFETY: f64 = 0.75;
// Keep donor-set expansion modest. PR #825 doubled the donor cap and raised the
// minimum donor count aggressively while loosening other guards; the broader,
// less selective state set regressed raw chr21 accuracy.
const SM_MATCH_DONORS: usize = 16;
const ADAPTIVE_REFINE_WINDOW_MARKERS: usize = 16;
const ADAPTIVE_REFINE_STEP_MARKERS: usize = 8;
const ADAPTIVE_REFINE_U_THRESHOLD: f32 = 0.38;
const ADAPTIVE_REFINE_MAX_DOSAGE_DELTA: f32 = 0.005;
const ADAPTIVE_REFINE_MAX_KL: f32 = 1e-3;
const SM_MATCH_LOW_CONF_FRAC: f32 = 0.02;
const SM_MATCH_MIN_DONORS: usize = 2;
// Keep the "small panel => use full panel" threshold generous. PR #808 cut this
// to 64 for speed, which prematurely truncated affordable panels and materially
// lowered chr21 R².
const SMALL_PANEL_FULL_CAP_HAPS: usize = 512;
const FULL_PANEL_RAM_FRACTION: f64 = 0.9;
const SCAN_RAM_FRACTION: f64 = 0.10;
const TARGET_CACHE_RAM_FRACTION: f64 = 0.10;
const REF_PANEL_RAM_FRACTION: f64 = 0.75;
const EXACT_PRESCAN_MAX_OPS: u128 = 250_000_000;
const MIN_AVAIL_BYTES_FOR_PLANNING: u64 = 64 * 1024 * 1024;
const ORIENTATION_HANDOFF_MIN_MARGIN: f64 = 0.05;
const ORIENTATION_ETA_MAX: f64 = 0.05;
const ORIENTATION_ETA_MIN: f64 = 1e-8;
const ORIENTATION_ETA_EM_ITERS: usize = 2;
const ORIENTATION_DECISION_MARGIN: f64 = 0.02;
// When memory detection fails, use a conservative fallback budget for prescan
// batching/caching to avoid pathological re-reads of the target VCF.
const PRESCAN_FALLBACK_AVAIL_BYTES: u64 = 256 * 1024 * 1024;
// Planning-grid target in hazard space. We target ~1% recombination/switch
// probability per planning window, then split each I/O interval so the
// cumulative Li-Stephens hazard per planning segment stays near this budget.
const PLANNING_TARGET_SWITCH_PROB: f64 = 0.01;
// Overlap/handoff retention target: include the suffix where expected retained
// copy signal to the window end is at least epsilon.
const HANDOFF_RETAIN_EPS: f64 = 1e-3;

/// Extra markers appended past each piecewise segment's core boundary so the
/// backward pass warms up before reaching the core region.  Zero RAM cost
/// (ref_columns are already loaded for the full I/O window) and negligible
/// speed cost (~10 extra forward-backward markers per segment boundary).
const BACKWARD_HALO_MARKERS: usize = 10;

/// Describes a piecewise HMM segment with an optional backward halo.
///
/// Layout within the I/O window's marker array:
///
/// ```text
///   [-------- core --------][-- halo --]
///    ^core_start             ^core_end  ^extended_end
/// ```
///
/// The HMM processes `[core_start..extended_end)`.
/// Only `[core_start..core_end)` posteriors are emitted.
///
/// All index arithmetic is encapsulated here so callers never perform raw
/// `seg_start`/`seg_end` math — preventing off-by-one and range-confusion bugs
/// at compile time via API design (zero runtime cost).
#[derive(Clone, Debug)]
struct SegmentExtent {
    /// First marker in the core output range (I/O window index, inclusive).
    core_start: usize,
    /// Last marker in the core output range (I/O window index, exclusive).
    core_end: usize,
    /// Last marker in the extended range (I/O window index, exclusive).
    /// `extended_end >= core_end`; the surplus is backward halo.
    extended_end: usize,
    /// Planning window index for state selection (inclusive).
    plan_start: usize,
    /// Planning window index for state selection (exclusive).
    plan_end: usize,
}

impl SegmentExtent {
    /// Build a segment extent, clamping the backward halo to the window size.
    #[inline]
    fn new(
        core_start: usize,
        core_end: usize,
        plan_start: usize,
        plan_end: usize,
        n_window_markers: usize,
    ) -> Self {
        assert!(core_start <= core_end);
        assert!(core_end <= n_window_markers);
        let extended_end = (core_end + BACKWARD_HALO_MARKERS).min(n_window_markers);
        Self {
            core_start,
            core_end,
            extended_end,
            plan_start,
            plan_end,
        }
    }

    /// Total markers the HMM will process (core + halo).
    #[inline]
    fn hmm_len(&self) -> usize {
        self.extended_end - self.core_start
    }

    /// Number of core output markers.
    #[inline]
    fn core_len(&self) -> usize {
        self.core_end - self.core_start
    }

    /// Planning window range for state selection.
    #[inline]
    fn plan_range(&self) -> (usize, usize) {
        (self.plan_start, self.plan_end)
    }

    /// Slice `ref_columns` for HMM input (core + halo).
    #[inline]
    fn slice_ref_columns<'a>(&self, cols: &'a [GenotypeColumn]) -> &'a [GenotypeColumn] {
        &cols[self.core_start..self.extended_end]
    }

    /// Slice `nearest_obs_retain` for HMM input (core + halo).
    #[inline]
    fn slice_retain<'a>(&self, retain: &'a [f32]) -> &'a [f32] {
        &retain[self.core_start..self.extended_end]
    }

    /// Build `p_recomb` for HMM input (core + halo), zeroing the first entry
    /// when the segment received boundary-mapped priors.
    #[inline]
    fn build_p_recomb(&self, full_p_recomb: &[f32], boundary_mapped: bool) -> Vec<f32> {
        let mut out = full_p_recomb[self.core_start..self.extended_end].to_vec();
        if boundary_mapped && !out.is_empty() {
            out[0] = 0.0;
        }
        out
    }

    /// Build `TargetAlleleProbs` for the extended range (core + halo).
    fn build_target_probs(&self, input_probs: &TargetAlleleProbs) -> TargetAlleleProbs {
        let hmm_len = self.hmm_len();
        let mut offsets = Vec::with_capacity(hmm_len + 1);
        let mut probs = Vec::new();
        let mut observed = Vec::with_capacity(hmm_len);
        let mut marker_errors = Vec::with_capacity(hmm_len);
        offsets.push(0);
        for m in self.core_start..self.extended_end {
            probs.extend_from_slice(input_probs.probs_for_marker(m));
            offsets.push(probs.len());
            observed.push(input_probs.is_observed_marker(m));
            marker_errors.push(input_probs.marker_error_rate(m).unwrap_or(0.0));
        }
        let panel_priors = input_probs.panel_priors().map(|panel| {
            let mut local = Vec::with_capacity(hmm_len);
            for m in self.core_start..self.extended_end {
                local.push(panel[m].clone());
            }
            std::sync::Arc::new(local)
        });
        let mut out = TargetAlleleProbs::new(
            offsets,
            probs,
            observed,
            panel_priors,
            input_probs.min_untyped_prior_mix(),
        );
        if marker_errors.iter().any(|&v| v > 0.0) {
            out.set_marker_error_rates(marker_errors);
        }
        out
    }

    /// HMM-local index for the prior/handoff marker (last core marker).
    ///
    /// State posteriors at this index are used to build `chained_priors` for
    /// the next segment.  This deliberately points to the last *core* marker,
    /// not the last halo marker, so the handoff reflects the core region's
    /// posterior rather than the less-constrained halo tail.
    #[inline]
    fn handoff_hmm_idx(&self) -> usize {
        self.core_len().saturating_sub(1)
    }

    /// The boundary recombination rate for boundary mapping (at core_start).
    #[inline]
    fn boundary_recomb(&self, full_p_recomb: &[f32]) -> f32 {
        full_p_recomb
            .get(self.core_start)
            .copied()
            .unwrap_or(0.0)
            .clamp(0.0, 1.0)
    }

    /// Extract core-range posteriors from the full HMM output, clipped to the
    /// output window `[output_start..output_end)`.
    #[inline]
    fn extract_output_posteriors<'a>(
        &'a self,
        seg_posteriors: &'a [AllelePosteriors],
        output_start: usize,
        output_end: usize,
    ) -> impl Iterator<Item = AllelePosteriors> + 'a {
        let take_start = self.core_start.max(output_start);
        let take_end = self.core_end.min(output_end);
        (take_start..take_end).map(move |gm| seg_posteriors[gm - self.core_start].clone())
    }
}

#[inline]
fn recomb_lambda_from_p(p: f32) -> f64 {
    let p = p as f64;
    if p <= 0.0 {
        0.0
    } else if p >= 1.0 {
        f64::INFINITY
    } else {
        -(1.0 - p).ln()
    }
}

#[inline]
fn overlap_start_from_hazard(output_start: usize, output_end: usize, p_recomb: &[f32]) -> usize {
    if output_end <= output_start {
        return output_start;
    }
    let eps = HANDOFF_RETAIN_EPS.clamp(1e-12, 0.5);
    let target_lambda = -eps.ln();
    let mut acc = 0.0f64;
    for m in ((output_start + 1)..output_end).rev() {
        acc += recomb_lambda_from_p(p_recomb.get(m).copied().unwrap_or(0.0));
        if acc >= target_lambda {
            return m.saturating_sub(1).max(output_start);
        }
    }
    output_start
}

#[inline]
fn compute_abyss_rank_cutoff(n_ref_haps: usize, window_top_k: usize) -> usize {
    if n_ref_haps == 0 {
        return 1;
    }
    // Keep this as a direct knob (clamped to panel size). We intentionally do
    // not reintroduce a hidden min-per-window floor here; that previously made
    // wide top-k sweeps look "flat" by collapsing many settings to one cutoff.
    window_top_k.max(1).min(n_ref_haps)
}

fn estimate_state_budget(
    available_bytes: u64,
    n_threads: usize,
    window_markers: usize,
    typed_markers: usize,
) -> usize {
    if available_bytes == 0 || n_threads == 0 || window_markers == 0 {
        return 0;
    }
    let per_state_bytes = estimate_per_state_bytes(window_markers, typed_markers);
    if per_state_bytes == 0 {
        return 0;
    }
    let budget = (available_bytes as f64 * IMPUTE_RAM_FRACTION) as u64;
    let per_thread = budget / n_threads.max(1) as u64;
    let safe_bytes = (per_thread as f64 * STATE_BUDGET_SAFETY) as u64;
    (safe_bytes as usize) / per_state_bytes
}

#[inline]
fn estimate_per_state_bytes(window_markers: usize, typed_markers: usize) -> usize {
    // Approximate active HMM memory from ImputeWorkspace:
    // - per-state vectors are O(states): fwd, bwd, emissions, weights, alleles, patterns.
    // - marker-scaled per-state storage is dominated by typed checkpoints (f32 each).
    // We apply a 1.5x safety factor to cover allocator overhead and temporary scratch.
    let typed = typed_markers.min(window_markers).max(1);
    let base = 40usize.saturating_add(typed.saturating_mul(4));
    let with_safety = base.saturating_add(base / 2);
    with_safety.max(64)
}

#[inline]
fn estimate_hmm_job_bytes(n_states: usize, window_markers: usize, typed_markers: usize) -> u64 {
    (estimate_per_state_bytes(window_markers, typed_markers) as u64)
        .saturating_mul(n_states.max(1) as u64)
}

#[inline]
fn calibrated_emission_error(input_probs: &TargetAlleleProbs, base_error_rate: f32) -> f32 {
    // Empirical-Bayes calibration:
    // Treat residual r = 1 - max_a p(a) at informative typed markers as a
    // noisy observation of per-marker emission error epsilon.
    // Posterior mean under Beta(alpha, beta) prior:
    //   eps_post = (alpha + sum(w*r)) / (alpha + beta + sum(w))
    // with prior mean anchored to base_error_rate and prior strength m0.
    // We weight each marker by normalized information content:
    //   w = 1 - H(p)/log(K), K = number of alleles
    // so near-uniform markers contribute little evidence.
    // Do not disable this calibration outright. PR #791 replaced it with a
    // base-error passthrough while simultaneously increasing other HMM
    // diffuseness knobs, and the combination worsened chr21 R²/SER instead of
    // improving calibration.
    const PRIOR_STRENGTH_MARKERS: f32 = 16.0;
    let mut weighted_residual_sum = 0.0f32;
    let mut weight_sum = 0.0f32;
    for m in 0..input_probs.n_markers() {
        if !input_probs.is_observed_marker(m) || input_probs.is_uniform_marker(m) {
            continue;
        }
        let probs = input_probs.probs_for_marker(m);
        if probs.is_empty() {
            continue;
        }
        let mut max_prob = 0.0f32;
        let mut any_finite = false;
        let mut entropy = 0.0f32;
        let mut n_alleles = 0usize;
        for &p in probs {
            if p.is_finite() {
                any_finite = true;
                if p > 0.0 {
                    entropy -= p * p.ln();
                }
                n_alleles += 1;
                if p > max_prob {
                    max_prob = p;
                }
            }
        }
        if !any_finite {
            continue;
        }
        let max_entropy = (n_alleles.max(2) as f32).ln();
        let info_weight = if max_entropy > 0.0 {
            (1.0 - (entropy / max_entropy)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        if info_weight <= 0.0 {
            continue;
        }
        let residual = (1.0 - max_prob.clamp(0.0, 1.0)).max(0.0);
        weighted_residual_sum += info_weight * residual;
        weight_sum += info_weight;
    }
    if weight_sum <= 0.0 {
        return base_error_rate.clamp(1e-6, 0.5);
    }
    let base = base_error_rate.clamp(1e-6, 0.5);
    let alpha = (base * PRIOR_STRENGTH_MARKERS).max(1e-6);
    let beta = ((1.0 - base) * PRIOR_STRENGTH_MARKERS).max(1e-6);
    let posterior = (alpha + weighted_residual_sum) / (alpha + beta + weight_sum);
    // Allow sharpening below base when typed evidence is strong, but keep it
    // bounded to avoid sparse-array collapse. PR #791 removed sharpening
    // entirely; PR #825 relaxed adjacent regularizers at the same time. Neither
    // produced better chr21 accuracy.
    let min_error = (base * 0.1).max(1e-6).min(base);
    posterior.clamp(min_error, 0.5)
}

#[inline]
fn marker_emission_error_from_probs(probs: &[f32], observed: bool, base_error_rate: f32) -> f32 {
    let base = base_error_rate.clamp(1e-6, 0.5);
    if !observed || probs.is_empty() {
        return base;
    }
    let mut max_prob = 0.0f32;
    let mut entropy = 0.0f32;
    let mut n_alleles = 0usize;
    for &p in probs {
        if !p.is_finite() || p <= 0.0 {
            continue;
        }
        n_alleles += 1;
        if p > max_prob {
            max_prob = p;
        }
        entropy -= p * p.ln();
    }
    if n_alleles == 0 {
        return base;
    }
    let max_entropy = (n_alleles.max(2) as f32).ln();
    let entropy_norm = if max_entropy > 0.0 {
        (entropy / max_entropy).clamp(0.0, 1.0)
    } else {
        1.0
    };
    let confidence = ((1.0 - entropy_norm) * max_prob.clamp(0.0, 1.0)).clamp(0.0, 1.0);
    let scaled = base * (1.6 - 1.2 * confidence);
    let residual = (1.0 - max_prob.clamp(0.0, 1.0)).max(0.0);
    let blended = 0.7 * scaled + 0.3 * residual;
    // Do not clamp this at `base` or `0.5 * base`. PR #829 and PR #801 both
    // raised the emission floor in this path and both regressed chr21 R².
    blended.clamp((base * 0.15).max(1e-6), 0.5)
}

// WARNING: Do NOT use aggressive scaling factors here. PR #740 tried
// base=0.08 with cluster/err factors clamped to [0.5, 10.0] (up to 100x
// combined), which pushed min_untyped_prior_mix to the 0.5 cap — replacing
// half the posterior with panel frequencies. Result: R² -0.000726, HOMALT
// -0.000730. The cubic ramp + mild factors below (clamped to [0.8, 1.6])
// are intentionally conservative: local LD should dominate when data exists.
// PR #808 also raised this floor for calibration/speed, while PR #825 lowered
// it during a larger donor-selection retune. Both moved the prior floor in the
// wrong direction once the full system was measured on chr21.
#[inline]
fn adaptive_untyped_prior_mix(
    observed_ratio: f32,
    cluster_cm: f32,
    p_mismatch: f32,
    phase_confidence_unavailable: bool,
) -> f32 {
    // Global panel-frequency floor for completely untyped sites.
    //
    // This floor should be small and primarily prevent degenerate collapse.
    // Local HMM evidence should remain dominant whenever available.
    let typed = observed_ratio.clamp(0.0, 1.0);
    let missing = 1.0 - typed;

    // Cubic ramp keeps the floor almost zero until missingness becomes
    // meaningful, then increases sharply in sparse windows.
    let missing_ramp = missing.powi(3);

    // Slightly stronger floor when cluster distance is wide (weaker local
    // linkage signal) or the model error rate is elevated.
    let cluster_factor = (cluster_cm / 0.04f32).clamp(0.8, 1.6);
    let err_factor = (p_mismatch / 4e-4f32).clamp(0.8, 1.6);

    // Unphased-target imputation has additional uncertainty from phase
    // transfer, so apply a mild boost.
    let phase_factor = if phase_confidence_unavailable {
        1.25
    } else {
        1.0
    };

    let floor = 0.002 + 0.08 * missing_ramp;
    (floor * cluster_factor * err_factor * phase_factor).clamp(0.001, 0.12)
}

#[inline]
fn adaptive_sm_donor_k(beam: &RankBeam, n_ref_haps: usize, query: PbwtQueryAllele) -> usize {
    if n_ref_haps == 0 {
        return 1;
    }
    let min_k = SM_MATCH_MIN_DONORS.max(1);
    let max_k = SM_MATCH_DONORS.saturating_mul(2).max(min_k).min(n_ref_haps);
    if max_k <= min_k {
        return min_k.min(n_ref_haps).max(1);
    }
    let u = pbwt_beam_uncertainty(beam, n_ref_haps, query);
    let span = (max_k - min_k) as f32;
    (min_k as f32 + span * u)
        .round()
        .clamp(min_k as f32, max_k as f32) as usize
}

#[inline]
fn prescan_match_weight(freq: f32, min_freq: f32) -> f32 {
    let p = freq.clamp(min_freq, 1.0 - min_freq);
    ((1.0 - p) / p).ln().max(0.0)
}

#[inline]
fn blend_haplotype_priors(
    p_keep: &HaplotypePriors,
    p_swap: &HaplotypePriors,
    swap_prob: f32,
) -> HaplotypePriors {
    let w_swap = swap_prob.clamp(0.0, 1.0);
    let w_keep = 1.0 - w_swap;
    if w_swap <= 0.0 {
        return p_keep.clone();
    }
    if w_keep <= 0.0 {
        return p_swap.clone();
    }
    let ids_keep = p_keep.ids();
    let probs_keep = p_keep.probs();
    let ids_swap = p_swap.ids();
    let probs_swap = p_swap.probs();
    let mut out_ids: Vec<GlobalHapId> = Vec::with_capacity(ids_keep.len().max(ids_swap.len()));
    let mut out_probs: Vec<f32> = Vec::with_capacity(out_ids.capacity());
    let mut i = 0usize;
    let mut j = 0usize;
    while i < ids_keep.len() || j < ids_swap.len() {
        if i < ids_keep.len() && (j >= ids_swap.len() || ids_keep[i] < ids_swap[j]) {
            out_ids.push(ids_keep[i]);
            out_probs.push(w_keep * probs_keep[i]);
            i += 1;
        } else if j < ids_swap.len() && (i >= ids_keep.len() || ids_swap[j] < ids_keep[i]) {
            out_ids.push(ids_swap[j]);
            out_probs.push(w_swap * probs_swap[j]);
            j += 1;
        } else {
            out_ids.push(ids_keep[i]);
            out_probs.push(w_keep * probs_keep[i] + w_swap * probs_swap[j]);
            i += 1;
            j += 1;
        }
    }
    HaplotypePriors::new(out_ids, out_probs)
}

#[inline]
fn orientation_eta_from_expected_flips(expected_flips: f64, n_boundaries: usize) -> f64 {
    if n_boundaries == 0 {
        return ORIENTATION_ETA_MIN;
    }
    (expected_flips.max(0.0) / n_boundaries as f64).clamp(ORIENTATION_ETA_MIN, ORIENTATION_ETA_MAX)
}

#[inline]
fn orientation_weights_from_posterior_swap(p_swap: f32) -> (f32, f32) {
    let w_swap = p_swap.clamp(0.0, 1.0);
    (1.0 - w_swap, w_swap)
}

#[inline]
fn compose_boundary_message(
    priors_id: &HaplotypePriors,
    priors_swap: &HaplotypePriors,
    weight_swap: f32,
) -> HaplotypePriors {
    let w_swap = weight_swap.clamp(0.0, 1.0);
    blend_haplotype_priors(priors_id, priors_swap, w_swap)
}

#[inline]
fn sample_idx_from_usize(sample_idx: usize) -> SampleIdx {
    SampleIdx::new(
        u32::try_from(sample_idx)
            .expect("sample index conversion overflow: usize does not fit in u32"),
    )
}

#[derive(Clone, Debug)]
struct ImputationPlan {
    n_ref_haps: usize,
    core_states: Vec<Vec<RefHapId>>, // per target hap (derived)
    window_intervals: Vec<Vec<HapIntervals>>, // per target hap (sparse)
    abyss_mask: Vec<BitVec<u64, Lsb0>>, // per target hap
    per_window_cap: usize,
    per_window_caps: Vec<usize>, // per I/O window (global, same for all target haps)
    io_to_planning_ranges: Vec<(usize, usize)>, // per I/O window -> planning window range [start, end)
    planning_num_windows: usize,
    planning_handoff: Vec<(f64, f64)>, // per planning window: (start_cm, end_cm)
    full_panel: bool,
    stats: ImputationPlanStats,
}

#[derive(Clone, Debug)]
struct HapIntervals {
    hap: RefHapId,
    intervals: Vec<crate::model::state_allocator::WindowSpan>,
}

#[inline]
fn interval_support_over_range(
    intervals: &HapIntervals,
    range_start: usize,
    range_end: usize,
) -> Option<u32> {
    if range_start >= range_end {
        return None;
    }
    let mut total = 0u32;
    for w in range_start..range_end {
        let mut covered = false;
        for &span in intervals.intervals.iter() {
            if span.contains(w) {
                covered = true;
                break;
            }
        }
        if covered {
            total = total.saturating_add(1);
        }
    }
    if total > 0 { Some(total) } else { None }
}

#[derive(Clone, Debug, Default)]
struct ImputationPlanStats {
    haps: usize,
    core_min: usize,
    core_max: usize,
    core_sum: usize,
    dynamic_min: usize,
    dynamic_max: usize,
    dynamic_sum: usize,
    abyss_min: usize,
    abyss_max: usize,
    abyss_sum: usize,
}

impl ImputationPlanStats {
    fn update(&mut self, core: usize, dynamic: usize, abyss: usize) {
        if self.haps == 0 {
            self.core_min = core;
            self.core_max = core;
            self.dynamic_min = dynamic;
            self.dynamic_max = dynamic;
            self.abyss_min = abyss;
            self.abyss_max = abyss;
        } else {
            self.core_min = self.core_min.min(core);
            self.core_max = self.core_max.max(core);
            self.dynamic_min = self.dynamic_min.min(dynamic);
            self.dynamic_max = self.dynamic_max.max(dynamic);
            self.abyss_min = self.abyss_min.min(abyss);
            self.abyss_max = self.abyss_max.max(abyss);
        }

        self.core_sum += core;
        self.dynamic_sum += dynamic;
        self.abyss_sum += abyss;
        self.haps += 1;
    }
}

fn log_imputation_plan_summary(plan: &ImputationPlan) {
    let n_target_haps = plan.core_states.len();
    if n_target_haps == 0 {
        eprintln!("Imputation plan: no target haplotypes");
        return;
    }
    let n_io_windows = plan.per_window_caps.len();
    let n_windows = plan.planning_num_windows.max(1);
    let (
        core_min,
        core_avg,
        core_max,
        dynamic_min,
        dynamic_avg,
        dynamic_max,
        abyss_min,
        abyss_avg,
        abyss_max,
    ) = if plan.stats.haps == n_target_haps && plan.stats.haps > 0 {
        let denom = plan.stats.haps as f64;
        (
            plan.stats.core_min,
            plan.stats.core_sum as f64 / denom,
            plan.stats.core_max,
            plan.stats.dynamic_min,
            plan.stats.dynamic_sum as f64 / denom,
            plan.stats.dynamic_max,
            plan.stats.abyss_min,
            plan.stats.abyss_sum as f64 / denom,
            plan.stats.abyss_max,
        )
    } else {
        let mut core_min = usize::MAX;
        let mut core_max = 0usize;
        let mut core_sum = 0usize;
        let mut dynamic_min = usize::MAX;
        let mut dynamic_max = 0usize;
        let mut dynamic_sum = 0usize;
        let mut abyss_min = usize::MAX;
        let mut abyss_max = 0usize;
        let mut abyss_sum = 0usize;

        for hap_idx in 0..n_target_haps {
            let core = plan.core_states.get(hap_idx).map(|v| v.len()).unwrap_or(0);
            let intervals = plan
                .window_intervals
                .get(hap_idx)
                .map(|v| v.len())
                .unwrap_or(0);
            let dynamic = intervals.saturating_sub(core);
            let abyss = plan
                .abyss_mask
                .get(hap_idx)
                .map(|v| v.count_ones())
                .unwrap_or(0);

            core_min = core_min.min(core);
            core_max = core_max.max(core);
            core_sum += core;

            dynamic_min = dynamic_min.min(dynamic);
            dynamic_max = dynamic_max.max(dynamic);
            dynamic_sum += dynamic;

            abyss_min = abyss_min.min(abyss);
            abyss_max = abyss_max.max(abyss);
            abyss_sum += abyss;
        }

        let denom = n_target_haps as f64;
        (
            core_min,
            core_sum as f64 / denom,
            core_max,
            dynamic_min,
            dynamic_sum as f64 / denom,
            dynamic_max,
            abyss_min,
            abyss_sum as f64 / denom,
            abyss_max,
        )
    };

    eprintln!(
        "Imputation plan hap counts (target_haps={}, io_windows={}, planning_windows={}): core_global[min/avg/max]={}/{:.1}/{}, dynamic_window[min/avg/max]={}/{:.1}/{}, abyss[min/avg/max]={}/{:.1}/{}",
        n_target_haps,
        n_io_windows,
        n_windows,
        core_min,
        core_avg,
        core_max,
        dynamic_min,
        dynamic_avg,
        dynamic_max,
        abyss_min,
        abyss_avg,
        abyss_max
    );
}

fn estimate_scan_batch_size(
    available_bytes: u64,
    n_ref_haps: usize,
    n_target_haps: usize,
    per_window_caps: &[usize],
) -> usize {
    if available_bytes == 0 || n_ref_haps == 0 || n_target_haps == 0 {
        return 1;
    }
    // Dense score vectors: global_scores + window_scores + best_window_scores + window_rank_hits.
    let dense_per_hap_bytes = (n_ref_haps as u64).saturating_mul(16);
    // Sparse per-window score store used by LMS allocator (`scores_by_window`).
    // Each stored score is `(usize, f32)` from `select_top_k`.
    let pair_bytes = std::mem::size_of::<(usize, f32)>() as u64;
    let sparse_pairs_per_hap = per_window_caps.iter().fold(0u64, |acc, &cap| {
        let base_top_m = cap
            .saturating_mul(PBWT_PER_WINDOW_MULT)
            .max(cap)
            .min(n_ref_haps.max(1));
        // Budget batch memory for worst-case adaptive top-M expansion in weak windows.
        let top_m =
            base_top_m.saturating_mul(PRESCAN_TOPM_WEAK_MULT_NUM) / PRESCAN_TOPM_WEAK_MULT_DEN;
        acc.saturating_add(top_m.min(n_ref_haps.max(1)) as u64)
    });
    // `Vec<Vec<(usize, f32)>>` header bytes per window for one hap entry.
    let vec_header_bytes = (per_window_caps.len() as u64)
        .saturating_mul(std::mem::size_of::<Vec<(usize, f32)>>() as u64);
    let per_hap_bytes = dense_per_hap_bytes
        .saturating_add(sparse_pairs_per_hap.saturating_mul(pair_bytes))
        .saturating_add(vec_header_bytes);
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

fn build_sampling_points(gen_positions: &[f64], step_cm: f64, min_marker_step: usize) -> Vec<bool> {
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
    }
    sampling[n - 1] = true;
    sampling
}

#[derive(Clone, Copy, Debug, Default)]
struct PbwtPrescanWindowDiag {
    sampled_markers: usize,
    total_markers: usize,
    min_gen_pos: f64,
    max_gen_pos: f64,
    distinct_gen_pos: usize,
    non_finite_gen_pos: usize,
}

fn add_typed_anchor_sampling_resolved(
    sampling: &mut [bool],
    gen_positions: &[f64],
    typed_resolution: &[Option<TypedMarkerResolution>],
    step_cm: f64,
) {
    let n = sampling
        .len()
        .min(gen_positions.len())
        .min(typed_resolution.len());
    if n == 0 {
        return;
    }
    let step = step_cm.max(1e-6);
    let first = gen_positions[0];
    let mut anchors_per_bin: HashMap<i64, usize> = HashMap::new();

    for m in 0..n {
        if !sampling[m] || typed_resolution[m].is_none() {
            continue;
        }
        let bin = ((gen_positions[m] - first) / step).floor() as i64;
        let entry = anchors_per_bin.entry(bin).or_insert(0);
        *entry = entry.saturating_add(1);
    }

    for m in 0..n {
        if typed_resolution[m].is_none() || sampling[m] {
            continue;
        }
        let bin = ((gen_positions[m] - first) / step).floor() as i64;
        let used = anchors_per_bin.get(&bin).copied().unwrap_or(0);
        if used < PBWT_TYPED_ANCHORS_PER_BIN {
            sampling[m] = true;
            anchors_per_bin.insert(bin, used + 1);
        }
    }
}

fn window_boundaries_from_handoff(handoff: &[(f64, f64)]) -> Vec<f64> {
    if handoff.len() < 2 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(handoff.len() - 1);
    for i in 0..handoff.len() - 1 {
        let (prev_start, _) = handoff[i];
        let (next_start, _) = handoff[i + 1];
        // Preserve true planning-grid geometry for continuity penalties.
        // Only apply epsilon flooring for numerical stability.
        let dist = (next_start - prev_start).abs().max(1e-12);
        out.push(dist);
    }
    out
}

#[inline]
fn posterior_entropy_norm(post: &AllelePosteriors) -> f32 {
    let mut entropy = 0.0f32;
    let n_alleles = match post {
        AllelePosteriors::Biallelic(p_alt) => {
            let p1 = p_alt.clamp(1e-8, 1.0 - 1e-8);
            let p0 = (1.0 - p1).clamp(1e-8, 1.0 - 1e-8);
            entropy = -(p0 * p0.ln() + p1 * p1.ln());
            2usize
        }
        AllelePosteriors::Multiallelic(probs) => {
            for &p in probs.iter() {
                if p > 1e-8 {
                    entropy -= p * p.ln();
                }
            }
            probs.len().max(2)
        }
    };
    (entropy / (n_alleles as f32).ln().max(1e-8)).clamp(0.0, 1.0)
}

#[inline]
fn top1_top2_gap(post: &AllelePosteriors) -> f32 {
    match post {
        AllelePosteriors::Biallelic(p_alt) => (1.0 - 2.0 * (*p_alt - 0.5).abs()).clamp(0.0, 1.0),
        AllelePosteriors::Multiallelic(probs) => {
            let mut top1 = 0.0f32;
            let mut top2 = 0.0f32;
            for &p in probs.iter() {
                if p >= top1 {
                    top2 = top1;
                    top1 = p;
                } else if p > top2 {
                    top2 = p;
                }
            }
            (1.0 - (top1 - top2).clamp(0.0, 1.0)).clamp(0.0, 1.0)
        }
    }
}

fn uncertainty_score_window(
    posteriors: &[AllelePosteriors],
    input_probs: &TargetAlleleProbs,
    output_start: usize,
    start: usize,
    end: usize,
) -> f32 {
    let mut entropy_sum = 0.0f32;
    let mut gap_sum = 0.0f32;
    let mut discord = 0.0f32;
    let mut n = 0usize;
    let mut typed_n = 0usize;
    for m in start..end.min(posteriors.len()) {
        let ref_m = output_start + m;
        entropy_sum += posterior_entropy_norm(&posteriors[m]);
        gap_sum += top1_top2_gap(&posteriors[m]);
        n += 1;
        if input_probs.is_observed_marker(ref_m) {
            let probs = input_probs.probs_for_marker(ref_m);
            if !probs.is_empty() {
                let mut best = 0usize;
                let mut best_p = f32::MIN;
                for (i, &p) in probs.iter().enumerate() {
                    if p > best_p {
                        best_p = p;
                        best = i;
                    }
                }
                let pred = match &posteriors[m] {
                    AllelePosteriors::Biallelic(p) => {
                        if *p >= 0.5 {
                            1usize
                        } else {
                            0usize
                        }
                    }
                    AllelePosteriors::Multiallelic(pmf) => pmf
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .map(|(i, _)| i)
                        .unwrap_or(0),
                };
                if pred != best {
                    discord += 1.0;
                }
                typed_n += 1;
            }
        }
    }
    if n == 0 {
        return 0.0;
    }
    let h_bar = entropy_sum / n as f32;
    let delta_term = gap_sum / n as f32;
    let d = if typed_n > 0 {
        (discord / typed_n as f32).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (0.4 * h_bar + 0.3 * delta_term + 0.3 * d).clamp(0.0, 1.0)
}

fn posterior_delta_and_kl(prev: &[AllelePosteriors], next: &[AllelePosteriors]) -> (f32, f32) {
    let n = prev.len().min(next.len());
    if n == 0 {
        return (0.0, 0.0);
    }
    let mut max_delta = 0.0f32;
    let mut kl_sum = 0.0f32;
    for i in 0..n {
        match (&prev[i], &next[i]) {
            (AllelePosteriors::Biallelic(a), AllelePosteriors::Biallelic(b)) => {
                max_delta = max_delta.max((a - b).abs());
                let p1 = (*a).clamp(1e-8, 1.0 - 1e-8);
                let q1 = (*b).clamp(1e-8, 1.0 - 1e-8);
                let p0 = 1.0 - p1;
                let q0 = 1.0 - q1;
                kl_sum += p0 * (p0 / q0).ln() + p1 * (p1 / q1).ln();
            }
            _ => {
                let pa: Vec<f32> = match &prev[i] {
                    AllelePosteriors::Biallelic(v) => vec![1.0 - *v, *v],
                    AllelePosteriors::Multiallelic(p) => p.to_vec(),
                };
                let pb: Vec<f32> = match &next[i] {
                    AllelePosteriors::Biallelic(v) => vec![1.0 - *v, *v],
                    AllelePosteriors::Multiallelic(p) => p.to_vec(),
                };
                let m = pa.len().max(pb.len());
                for aidx in 0..m {
                    let p = pa.get(aidx).copied().unwrap_or(0.0).clamp(1e-8, 1.0);
                    let q = pb.get(aidx).copied().unwrap_or(0.0).clamp(1e-8, 1.0);
                    kl_sum += p * (p / q).ln();
                }
                let d_a = pa.get(1).copied().unwrap_or(0.0);
                let d_b = pb.get(1).copied().unwrap_or(0.0);
                max_delta = max_delta.max((d_a - d_b).abs());
            }
        }
    }
    (max_delta, kl_sum / n as f32)
}
fn build_planning_grid_from_handoff(
    io_handoff: &[(f64, f64)],
    params: &ModelParams,
) -> (Vec<(f64, f64)>, Vec<(usize, usize)>) {
    if io_handoff.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut min_start = f64::INFINITY;
    let mut max_end = f64::NEG_INFINITY;
    for &(s, e) in io_handoff {
        if s.is_finite() && e.is_finite() {
            min_start = min_start.min(s.min(e));
            max_end = max_end.max(s.max(e));
        }
    }
    if !min_start.is_finite() || !max_end.is_finite() || max_end <= min_start {
        let mut fallback_windows = Vec::with_capacity(io_handoff.len());
        let mut fallback_ranges = Vec::with_capacity(io_handoff.len());
        for (i, &(s, e)) in io_handoff.iter().enumerate() {
            fallback_windows.push((s, e));
            fallback_ranges.push((i, i + 1));
        }
        return (fallback_windows, fallback_ranges);
    }

    let mut planning = Vec::new();
    let mut io_to_plan = Vec::with_capacity(io_handoff.len());
    let lambda_target = recomb_lambda_from_p(PLANNING_TARGET_SWITCH_PROB as f32).max(1e-9);
    let intensity = (params.recomb_intensity as f64).max(1e-12);
    // lambda = intensity * dist_morgans = intensity * (dist_cm / 100)
    let step_cm = (lambda_target * 100.0 / intensity).max(1e-9);
    let mut cur = min_start;
    while cur < max_end {
        let next = (cur + step_cm).min(max_end);
        planning.push((cur, next));
        cur = next;
    }
    if planning.is_empty() {
        planning.push((min_start, max_end));
    }

    for &(start_raw, end_raw) in io_handoff {
        let s = start_raw.min(end_raw);
        let e = start_raw.max(end_raw);
        if !s.is_finite() || !e.is_finite() || e <= s {
            let fallback_idx = 0usize;
            io_to_plan.push((fallback_idx, (fallback_idx + 1).min(planning.len())));
            continue;
        }
        let mut first = usize::MAX;
        let mut last_excl = 0usize;
        for (pidx, &(ps, pe)) in planning.iter().enumerate() {
            if pe <= s || ps >= e {
                continue;
            }
            if first == usize::MAX {
                first = pidx;
            }
            last_excl = pidx + 1;
        }
        if first == usize::MAX || last_excl <= first {
            let io_mid = 0.5 * (s + e);
            let mut best_idx = 0usize;
            let mut best_dist = f64::INFINITY;
            for (pidx, &(ps, pe)) in planning.iter().enumerate() {
                let mid = 0.5 * (ps + pe);
                let d = (mid - io_mid).abs();
                if d < best_dist {
                    best_dist = d;
                    best_idx = pidx;
                }
            }
            io_to_plan.push((best_idx, best_idx + 1));
        } else {
            io_to_plan.push((first, last_excl));
        }
    }

    (planning, io_to_plan)
}

fn distribute_scores_to_planning_bins(
    per_io_scores: &[(usize, f32)],
    io_start_raw: f64,
    io_end_raw: f64,
    plan_start: usize,
    plan_end: usize,
    planning_handoff: &[(f64, f64)],
    planning_scores: &mut [Vec<(usize, f32)>],
) {
    if plan_start >= plan_end || plan_end > planning_scores.len() || per_io_scores.is_empty() {
        return;
    }
    if plan_end > planning_handoff.len() {
        return;
    }
    let io_start = io_start_raw.min(io_end_raw);
    let io_end = io_start_raw.max(io_end_raw);
    let mut overlap_weights = vec![0.0f32; plan_end - plan_start];
    let mut total_overlap = 0.0f32;
    if io_start.is_finite() && io_end.is_finite() && io_end > io_start {
        for p in plan_start..plan_end {
            let (ps, pe) = planning_handoff[p];
            let s = ps.min(pe);
            let e = ps.max(pe);
            let overlap = (io_end.min(e) - io_start.max(s)).max(0.0);
            let w = overlap as f32;
            overlap_weights[p - plan_start] = w;
            total_overlap += w;
        }
    }
    if total_overlap <= 0.0 {
        let uniform = 1.0f32 / (plan_end - plan_start) as f32;
        overlap_weights.fill(uniform);
    } else {
        let inv = 1.0 / total_overlap;
        for w in overlap_weights.iter_mut() {
            *w *= inv;
        }
    }
    for p in plan_start..plan_end {
        let scale = overlap_weights[p - plan_start];
        if scale <= 0.0 || !scale.is_finite() {
            continue;
        }
        for &(hap, score) in per_io_scores {
            let v = score * scale;
            if v.is_finite() && v > 0.0 {
                planning_scores[p].push((hap, v));
            }
        }
    }
}

fn aggregate_window_sparse_scores(window_scores: &mut [Vec<(usize, f32)>]) {
    for ws in window_scores.iter_mut() {
        if ws.len() <= 1 {
            continue;
        }
        let mut acc: HashMap<usize, f32> = HashMap::with_capacity(ws.len() * 2);
        for &(hap, score) in ws.iter() {
            if score.is_finite() && score > 0.0 {
                acc.entry(hap).and_modify(|s| *s += score).or_insert(score);
            }
        }
        ws.clear();
        ws.extend(acc.into_iter());
    }
}

fn build_sparse_scores(
    window_scores: &[Vec<(usize, f32)>],
    abyss: &BitSlice<u64, Lsb0>,
) -> (Vec<usize>, Vec<Vec<(usize, f32)>>) {
    let mut map: HashMap<usize, usize> = HashMap::new();
    let mut candidate_haps: Vec<usize> = Vec::new();
    let mut scores_by_hap: Vec<Vec<(usize, f32)>> = Vec::new();

    for (w, list) in window_scores.iter().enumerate() {
        for &(hap, score) in list.iter() {
            if score <= 0.0 || !score.is_finite() {
                continue;
            }
            // BitSlice::get returns Option<BitRef>, deref to bool
            if hap < abyss.len() && *abyss.get(hap).unwrap() {
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

#[inline]
fn adaptive_top_m_window_from_support(
    support: usize,
    base_top_m: usize,
    per_window_cap_window: usize,
    n_ref_haps: usize,
) -> usize {
    if base_top_m == 0 {
        return per_window_cap_window.max(1).min(n_ref_haps.max(1));
    }
    let support = support.max(1);
    let denom = n_ref_haps.max(1) as f64;
    let support_frac = support as f64 / denom;
    let mut out = base_top_m;
    if support_frac <= 0.08 {
        out = out.saturating_mul(PRESCAN_TOPM_WEAK_MULT_NUM) / PRESCAN_TOPM_WEAK_MULT_DEN;
    } else if support_frac >= 0.65 {
        out = out.saturating_mul(PRESCAN_TOPM_STRONG_MULT_NUM) / PRESCAN_TOPM_STRONG_MULT_DEN;
    }
    out.max(per_window_cap_window.max(1)).min(n_ref_haps.max(1))
}

#[inline]
fn adaptive_top_m_upper_bound(
    base_top_m: usize,
    per_window_cap_window: usize,
    n_ref_haps: usize,
) -> usize {
    if base_top_m == 0 {
        return per_window_cap_window.max(1).min(n_ref_haps.max(1));
    }
    let boosted =
        base_top_m.saturating_mul(PRESCAN_TOPM_WEAK_MULT_NUM) / PRESCAN_TOPM_WEAK_MULT_DEN;
    boosted
        .max(per_window_cap_window.max(1))
        .min(n_ref_haps.max(1))
}

fn select_top_k_adaptive_with_support(
    scores: &[f32],
    base_top_m: usize,
    per_window_cap_window: usize,
    n_ref_haps: usize,
) -> (usize, Vec<(usize, f32)>) {
    if scores.is_empty() {
        let top_m = per_window_cap_window.max(1).min(n_ref_haps.max(1));
        return (top_m, Vec::new());
    }
    let upper_k = adaptive_top_m_upper_bound(base_top_m, per_window_cap_window, n_ref_haps)
        .max(1)
        .min(scores.len().max(1));
    let mut heap: BinaryHeap<Reverse<RankedScore>> =
        BinaryHeap::with_capacity(upper_k.saturating_add(1));
    let mut support = 0usize;
    for (idx, &score) in scores.iter().enumerate() {
        if !score.is_finite() || score <= 0.0 {
            continue;
        }
        support = support.saturating_add(1);
        let candidate = RankedScore { idx, score };
        if heap.len() < upper_k {
            heap.push(Reverse(candidate));
            continue;
        }
        let keep = heap
            .peek()
            .map(|lowest| candidate > lowest.0)
            .unwrap_or(true);
        if keep {
            heap.pop();
            heap.push(Reverse(candidate));
        }
    }
    let top_m =
        adaptive_top_m_window_from_support(support, base_top_m, per_window_cap_window, n_ref_haps);
    let mut ranked: Vec<(usize, f32)> = heap
        .into_iter()
        .map(|Reverse(r)| (r.idx, r.score))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if ranked.len() > top_m {
        ranked.truncate(top_m);
    }
    (top_m, ranked)
}

fn select_top_k(scores: &[f32], k: usize) -> Vec<(usize, f32)> {
    if k == 0 || scores.is_empty() {
        return Vec::new();
    }
    select_top_k_heap(scores, k, true)
}

fn select_top_k_allow_zero(scores: &[f32], k: usize) -> Vec<(usize, f32)> {
    if k == 0 || scores.is_empty() {
        return Vec::new();
    }
    select_top_k_heap(scores, k, false)
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct RankedScore {
    idx: usize,
    score: f32,
}

impl Eq for RankedScore {}

impl Ord for RankedScore {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // All callers filter to finite scores, so this ordering is total.
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.idx.cmp(&other.idx))
    }
}

impl PartialOrd for RankedScore {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

fn select_top_k_heap(scores: &[f32], k: usize, require_positive: bool) -> Vec<(usize, f32)> {
    let mut heap: BinaryHeap<Reverse<RankedScore>> = BinaryHeap::with_capacity(k.saturating_add(1));
    for (idx, &score) in scores.iter().enumerate() {
        if !score.is_finite() || (require_positive && score <= 0.0) {
            continue;
        }

        let candidate = RankedScore { idx, score };
        if heap.len() < k {
            heap.push(Reverse(candidate));
            continue;
        }

        let keep = heap
            .peek()
            .map(|lowest| candidate > lowest.0)
            .unwrap_or(true);
        if keep {
            heap.pop();
            heap.push(Reverse(candidate));
        }
    }

    let mut ranked: Vec<(usize, f32)> = heap
        .into_iter()
        .map(|Reverse(r)| (r.idx, r.score))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked
}

fn should_use_exact_prescan(n_ref_haps: usize, batch_len: usize, n_markers: usize) -> bool {
    let ops = n_ref_haps as u128 * batch_len as u128 * n_markers as u128;
    ops <= EXACT_PRESCAN_MAX_OPS
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TypedMarkerMapKind {
    Alignment,
    PositionalBiallelicMatch,
    PositionalBiallelicSwap,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TypedMarkerResolution {
    target_idx: usize,
    map_kind: TypedMarkerMapKind,
}

impl TypedMarkerResolution {
    #[inline]
    fn is_positional_fallback(self) -> bool {
        !matches!(self.map_kind, TypedMarkerMapKind::Alignment)
    }
}

fn build_target_marker_position_index<TargetSpace>(
    target_markers: &Markers<TargetSpace>,
) -> HashMap<String, HashMap<u32, Vec<usize>>> {
    let mut out: HashMap<String, HashMap<u32, Vec<usize>>> = HashMap::new();
    for t_idx in 0..target_markers.len() {
        let marker = target_markers.marker(MarkerIdx::new(t_idx as u32));
        let chrom = target_markers.chrom_name(marker.chrom).unwrap_or("");
        let chrom_key = normalize_chrom_local(chrom).to_string();
        out.entry(chrom_key)
            .or_default()
            .entry(marker.pos)
            .or_default()
            .push(t_idx);
    }
    out
}

fn build_ref_typed_marker_resolutions<TargetSpace, RefSpace>(
    target_markers: &Markers<TargetSpace>,
    ref_markers: &Markers<RefSpace>,
    alignment: &MarkerAlignment<TargetSpace, RefSpace>,
) -> Vec<Option<TypedMarkerResolution>> {
    let target_pos_index = build_target_marker_position_index(target_markers);
    let mut out = vec![None; ref_markers.len()];
    for (ref_m, slot) in out.iter_mut().enumerate() {
        if let Some(target_m) = alignment.ref_to_target.get(ref_m).and_then(|v| *v) {
            *slot = Some(TypedMarkerResolution {
                target_idx: target_m.as_usize(),
                map_kind: TypedMarkerMapKind::Alignment,
            });
            continue;
        }

        let ref_marker = ref_markers.marker(MarkerIdx::new(ref_m as u32));
        if ref_marker.n_alleles() != 2 {
            continue;
        }
        let ref_chrom = ref_markers.chrom_name(ref_marker.chrom).unwrap_or("");
        let candidates = target_pos_index
            .get(normalize_chrom_local(ref_chrom))
            .and_then(|m| m.get(&ref_marker.pos));
        let Some(candidates) = candidates else {
            continue;
        };
        let Some(ref_alt) = ref_marker.alt_alleles.first() else {
            continue;
        };
        let mut candidate: Option<TypedMarkerResolution> = None;
        for &target_idx in candidates {
            let target_marker = target_markers.marker(MarkerIdx::new(target_idx as u32));
            if target_marker.n_alleles() != 2 {
                continue;
            }
            let Some(target_alt) = target_marker.alt_alleles.first() else {
                continue;
            };
            let same = ref_marker.ref_allele == target_marker.ref_allele && ref_alt == target_alt;
            let swapped =
                ref_marker.ref_allele == *target_alt && *ref_alt == target_marker.ref_allele;
            if !same && !swapped {
                continue;
            }
            let map_kind = if swapped {
                TypedMarkerMapKind::PositionalBiallelicSwap
            } else {
                TypedMarkerMapKind::PositionalBiallelicMatch
            };
            if candidate.is_some() {
                candidate = None;
                break;
            }
            candidate = Some(TypedMarkerResolution {
                target_idx,
                map_kind,
            });
        }
        *slot = candidate;
    }
    out
}

#[inline]
fn map_target_allele_to_ref<TargetSpace, RefSpace>(
    alignment: &MarkerAlignment<TargetSpace, RefSpace>,
    resolution: TypedMarkerResolution,
    raw: u8,
) -> Option<u8> {
    if raw == crate::data::storage::AlleleCode::MISSING.raw() {
        return None;
    }
    match resolution.map_kind {
        TypedMarkerMapKind::Alignment => {
            let mapping = alignment
                .allele_mappings
                .get(resolution.target_idx)
                .and_then(|m| m.as_ref());
            if let Some(mapping) = mapping {
                if (raw as usize) < mapping.targ_to_ref.len() {
                    let mapped = mapping.targ_to_ref[raw as usize];
                    u8::try_from(mapped).ok()
                } else {
                    None
                }
            } else {
                Some(raw)
            }
        }
        TypedMarkerMapKind::PositionalBiallelicMatch => match raw {
            0 | 1 => Some(raw),
            _ => None,
        },
        TypedMarkerMapKind::PositionalBiallelicSwap => match raw {
            0 => Some(1),
            1 => Some(0),
            _ => None,
        },
    }
}

#[inline]
fn map_target_probs_to_ref<TargetSpace, RefSpace>(
    alignment: &MarkerAlignment<TargetSpace, RefSpace>,
    resolution: TypedMarkerResolution,
    target_probs: &[f32],
    n_ref_alleles: usize,
    out: &mut Vec<f32>,
) -> bool {
    out.clear();
    out.resize(n_ref_alleles.max(1), 0.0);
    match resolution.map_kind {
        TypedMarkerMapKind::Alignment => {
            let mapping = alignment
                .allele_mappings
                .get(resolution.target_idx)
                .and_then(|m| m.as_ref());
            if let Some(mapping) = mapping {
                for (t_idx, &p) in target_probs.iter().enumerate() {
                    if t_idx >= mapping.targ_to_ref.len() || !p.is_finite() || p <= 0.0 {
                        continue;
                    }
                    let mapped = mapping.targ_to_ref[t_idx];
                    if mapped >= 0 && (mapped as usize) < out.len() {
                        out[mapped as usize] += p;
                    }
                }
            } else if target_probs.len() == out.len() {
                out.copy_from_slice(target_probs);
            } else {
                return false;
            }
        }
        TypedMarkerMapKind::PositionalBiallelicMatch => {
            if out.len() != 2 || target_probs.len() < 2 {
                return false;
            }
            out[0] = target_probs[0];
            out[1] = target_probs[1];
        }
        TypedMarkerMapKind::PositionalBiallelicSwap => {
            if out.len() != 2 || target_probs.len() < 2 {
                return false;
            }
            out[0] = target_probs[1];
            out[1] = target_probs[0];
        }
    }
    let mut sum = 0.0f32;
    for p in out.iter_mut() {
        if !p.is_finite() || *p < 0.0 {
            *p = 0.0;
        }
        sum += *p;
    }
    if sum <= 0.0 {
        return false;
    }
    let inv = 1.0 / sum;
    for p in out.iter_mut() {
        *p *= inv;
    }
    true
}

fn score_window_batch_exact_packed<TargetSpace, RefSpace>(
    batch_haps: &[usize],
    target_gt: &GenotypeMatrix<Phased, TargetSpace>,
    ref_markers: &Markers<RefSpace>,
    ref_columns: &[PackedRefColumn],
    n_ref_haps: usize,
    alignment: &MarkerAlignment<TargetSpace, RefSpace>,
    global_scores: &mut [Vec<f32>],
    window_scores: &mut [Vec<f32>],
) {
    let n_markers = ref_columns.len().min(ref_markers.len());
    if n_markers == 0 || n_ref_haps == 0 || batch_haps.is_empty() {
        return;
    }
    let resolutions =
        build_ref_typed_marker_resolutions(target_gt.markers(), ref_markers, alignment);
    let min_freq = 1.0 / (n_ref_haps.max(1) as f32);

    let mut query_alleles = vec![crate::data::storage::AlleleCode::MISSING.raw(); batch_haps.len()];
    let mut ref_alleles = vec![crate::data::storage::AlleleCode::MISSING.raw(); n_ref_haps];
    let mut ref_bins: Vec<Vec<u32>> = Vec::new();
    let mut query_rows_by_allele: Vec<Vec<usize>> = Vec::new();
    let mut pending_updates: Vec<Vec<(u32, f32)>> = vec![Vec::new(); batch_haps.len()];

    for m in 0..n_markers {
        let Some(resolution) = resolutions.get(m).copied().flatten() else {
            continue;
        };
        let target_marker_idx = MarkerIdx::new(resolution.target_idx as u32);
        for (i, &hap_idx) in batch_haps.iter().enumerate() {
            let raw = target_gt.allele(target_marker_idx, HapIdx::new(hap_idx as u32));
            query_alleles[i] = map_target_allele_to_ref(alignment, resolution, raw)
                .unwrap_or(crate::data::storage::AlleleCode::MISSING.raw());
        }

        let n_alleles = ref_markers
            .marker(MarkerIdx::new(m as u32))
            .n_alleles()
            .max(1);
        if query_rows_by_allele.len() < n_alleles {
            query_rows_by_allele.resize_with(n_alleles, Vec::new);
        }
        for rows in query_rows_by_allele.iter_mut().take(n_alleles) {
            rows.clear();
        }
        for (i, &a) in query_alleles.iter().enumerate() {
            if a == crate::data::storage::AlleleCode::MISSING.raw() {
                continue;
            }
            let idx = a as usize;
            if idx < n_alleles {
                query_rows_by_allele[idx].push(i);
            }
        }

        if ref_bins.len() < n_alleles {
            ref_bins.resize_with(n_alleles, Vec::new);
        }
        for bins in ref_bins.iter_mut().take(n_alleles) {
            bins.clear();
        }

        let col = &ref_columns[m];
        col.fill_alleles(&mut ref_alleles);
        let mut ref_non_missing = 0usize;
        for (rh, &ref_a) in ref_alleles.iter().enumerate() {
            if ref_a == crate::data::storage::AlleleCode::MISSING.raw() {
                continue;
            }
            ref_non_missing += 1;
            let idx = ref_a as usize;
            if idx >= ref_bins.len() {
                ref_bins.resize_with(idx + 1, Vec::new);
            }
            ref_bins[idx].push(rh as u32);
        }
        if ref_non_missing == 0 {
            continue;
        }

        // Exact grouped update with deferred sparse accumulation:
        // Original per-row score for row i, hap h at marker m:
        //   S_i[h] += w_{m,a_i} * 1{a_h == a_i}
        // where a_i is target allele for row i and
        // w_{m,a} = prescan_match_weight(freq_{m,a}, min_freq).
        //
        // Grouping rows by a_i is algebraically identical:
        // for each allele a, all rows in group G_a share the same w_{m,a},
        // and all h in bin B_a satisfy 1{a_h == a}. Applying the same delta
        // to every (i in G_a, h in B_a) yields exactly the same sum.
        for targ_idx in 0..n_alleles {
            let rows = &query_rows_by_allele[targ_idx];
            if rows.is_empty() {
                continue;
            }
            let freq = ref_bins
                .get(targ_idx)
                .map(|bins| bins.len() as f32 / ref_non_missing as f32)
                .unwrap_or(0.0);
            if freq <= 0.0 {
                continue;
            }
            let weight = prescan_match_weight(freq, min_freq);
            if weight <= 0.0 {
                continue;
            }
            let bins = ref_bins.get(targ_idx);
            let Some(bins) = bins else { continue };
            for &i in rows {
                let row_pending = &mut pending_updates[i];
                row_pending.reserve(bins.len());
                for &rh in bins {
                    row_pending.push((rh, weight));
                }
            }
        }
    }

    // Flush sparse updates into dense score rows once. This preserves exact
    // scoring formula while reducing repeated random writes in the marker loop.
    for i in 0..batch_haps.len() {
        let row_pending = &mut pending_updates[i];
        if row_pending.is_empty() {
            continue;
        }
        row_pending.sort_unstable_by_key(|(idx, _)| *idx);
        let row_global = &mut global_scores[i];
        let row_window = &mut window_scores[i];
        let mut j = 0usize;
        while j < row_pending.len() {
            let idx = row_pending[j].0 as usize;
            let mut delta = 0.0f32;
            while j < row_pending.len() && row_pending[j].0 as usize == idx {
                delta += row_pending[j].1;
                j += 1;
            }
            row_global[idx] += delta;
            let w = &mut row_window[idx];
            if w.is_finite() {
                *w += delta;
            } else {
                *w = delta;
            }
        }
        row_pending.clear();
    }
}

#[inline]
#[cfg(test)]
fn assert_score_mats_close(a: &[Vec<f32>], b: &[Vec<f32>], tol: f32) {
    assert_eq!(a.len(), b.len(), "row count mismatch");
    for (r, (ar, br)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(ar.len(), br.len(), "col count mismatch at row {}", r);
        for (c, (&x, &y)) in ar.iter().zip(br.iter()).enumerate() {
            if x.is_finite() && y.is_finite() {
                let diff = (x - y).abs();
                assert!(
                    diff <= tol,
                    "score mismatch at row={} col={} x={} y={} diff={} tol={}",
                    r,
                    c,
                    x,
                    y,
                    diff,
                    tol
                );
            } else {
                assert!(
                    x.is_finite() == y.is_finite(),
                    "finiteness mismatch at row={} col={} x={} y={}",
                    r,
                    c,
                    x,
                    y
                );
            }
        }
    }
}

fn score_window_batch_pbwt_packed<TargetSpace, RefSpace>(
    batch_haps: &[usize],
    target_gt: &GenotypeMatrix<Phased, TargetSpace>,
    ref_markers: &Markers<RefSpace>,
    ref_columns: &[PackedRefColumn],
    n_ref_haps: usize,
    alignment: &MarkerAlignment<TargetSpace, RefSpace>,
    gen_maps: &GeneticMaps,
    k_per_hap: usize,
    step_cm: f64,
    global_scores: &mut [Vec<f32>],
    window_scores: &mut [Vec<f32>],
) -> PbwtPrescanWindowDiag {
    let n_markers = ref_columns.len().min(ref_markers.len());
    if n_markers == 0 || n_ref_haps == 0 || batch_haps.is_empty() {
        return PbwtPrescanWindowDiag::default();
    }
    let resolutions =
        build_ref_typed_marker_resolutions(target_gt.markers(), ref_markers, alignment);

    let mut gen_positions = Vec::with_capacity(n_markers);
    for m in 0..n_markers {
        let marker = ref_markers.marker(MarkerIdx::new(m as u32));
        let chrom_name = ref_markers.chrom_name(marker.chrom).unwrap_or("");
        let gen_pos = gen_maps.gen_pos_by_name(chrom_name, marker.pos);
        gen_positions.push(gen_pos);
    }
    let mut sampling = build_sampling_points(&gen_positions, step_cm, PBWT_MIN_MARKER_STEP);
    add_typed_anchor_sampling_resolved(&mut sampling, &gen_positions, &resolutions, step_cm);
    let sampled_markers = sampling.iter().filter(|&&b| b).count();
    let mut min_gen_pos = f64::INFINITY;
    let mut max_gen_pos = f64::NEG_INFINITY;
    let mut distinct_gen_pos = 0usize;
    let mut prev_bits: Option<u64> = None;
    let mut non_finite_gen_pos = 0usize;
    for &cm in &gen_positions {
        if !cm.is_finite() {
            non_finite_gen_pos += 1;
            continue;
        }
        min_gen_pos = min_gen_pos.min(cm);
        max_gen_pos = max_gen_pos.max(cm);
        let bits = cm.to_bits();
        if prev_bits != Some(bits) {
            distinct_gen_pos += 1;
            prev_bits = Some(bits);
        }
    }
    if !min_gen_pos.is_finite() {
        min_gen_pos = 0.0;
    }
    if !max_gen_pos.is_finite() {
        max_gen_pos = 0.0;
    }
    let mut pbwt_fwd = ReferencePbwt::new(n_ref_haps);
    let mut beams_fwd: Vec<RankBeam> = (0..batch_haps.len())
        .map(|_| RankBeam::full(n_ref_haps as u32))
        .collect();
    let mut ref_alleles = vec![0u8; n_ref_haps];
    let mut query_alleles = vec![PbwtStrictAllele::missing(); batch_haps.len()];
    let mut donors_buf: Vec<u32> = Vec::new();
    let mut allele_counts: Vec<u32> = Vec::new();

    let min_freq = 1.0 / (n_ref_haps.max(1) as f32);

    for m in 0..n_markers {
        let resolution = resolutions.get(m).copied().flatten();
        for (i, &hap_idx) in batch_haps.iter().enumerate() {
            let qa = if let Some(resolution) = resolution {
                let raw = target_gt.allele(
                    MarkerIdx::new(resolution.target_idx as u32),
                    HapIdx::new(hap_idx as u32),
                );
                map_target_allele_to_ref(alignment, resolution, raw)
                    .unwrap_or(crate::data::storage::AlleleCode::MISSING.raw())
            } else {
                crate::data::storage::AlleleCode::MISSING.raw()
            };
            query_alleles[i] =
                PbwtStrictAllele::allele(qa).unwrap_or_else(PbwtStrictAllele::missing);
        }
        let col = &ref_columns[m];
        col.fill_alleles(&mut ref_alleles);

        let mut max_allele = 1u8;
        for &a in ref_alleles.iter() {
            if a != crate::data::storage::AlleleCode::MISSING.raw() && a > max_allele {
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
        let n_alleles = (max_allele as usize).saturating_add(1).max(2);

        pbwt_fwd.advance_with_beams_strict(
            &ref_alleles,
            n_alleles,
            m,
            &query_alleles,
            &mut beams_fwd,
        );

        allele_counts.clear();
        allele_counts.resize(n_alleles, 0);
        let mut present = 0u32;
        for &a in ref_alleles.iter() {
            if a == crate::data::storage::AlleleCode::MISSING.raw() {
                continue;
            }
            let idx = a as usize;
            if idx < allele_counts.len() {
                allele_counts[idx] += 1;
                present += 1;
            }
        }

        if sampling[m] {
            for (i, _) in batch_haps.iter().enumerate() {
                if query_alleles[i].is_missing() {
                    continue;
                }
                let targ = query_alleles[i]
                    .as_allele()
                    .expect("non-missing strict query allele should be concrete");
                let freq = if present > 0 {
                    allele_counts.get(targ as usize).copied().unwrap_or(0) as f32 / present as f32
                } else {
                    0.0
                };
                if freq <= 0.0 {
                    continue;
                }
                let weight = prescan_match_weight(freq, min_freq);
                if weight <= 0.0 {
                    continue;
                }
                pbwt_fwd.select_donors_into(&beams_fwd[i], k_per_hap, &mut donors_buf);
                for &d in donors_buf.iter() {
                    let idx = d as usize;
                    if idx < n_ref_haps {
                        let ref_a = ref_alleles[idx];
                        if ref_a == crate::data::storage::AlleleCode::MISSING.raw() || ref_a != targ
                        {
                            continue;
                        }
                        global_scores[i][idx] += weight;
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
    for (rev_step, m) in (0..n_markers).rev().enumerate() {
        let resolution = resolutions.get(m).copied().flatten();
        for (i, &hap_idx) in batch_haps.iter().enumerate() {
            let qa = if let Some(resolution) = resolution {
                let raw = target_gt.allele(
                    MarkerIdx::new(resolution.target_idx as u32),
                    HapIdx::new(hap_idx as u32),
                );
                map_target_allele_to_ref(alignment, resolution, raw)
                    .unwrap_or(crate::data::storage::AlleleCode::MISSING.raw())
            } else {
                crate::data::storage::AlleleCode::MISSING.raw()
            };
            query_alleles[i] =
                PbwtStrictAllele::allele(qa).unwrap_or_else(PbwtStrictAllele::missing);
        }
        let col = &ref_columns[m];
        col.fill_alleles(&mut ref_alleles);

        let mut max_allele = 1u8;
        for &a in ref_alleles.iter() {
            if a != crate::data::storage::AlleleCode::MISSING.raw() && a > max_allele {
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
        let n_alleles = (max_allele as usize).saturating_add(1).max(2);

        pbwt_bwd.advance_with_beams_strict(
            &ref_alleles,
            n_alleles,
            rev_step,
            &query_alleles,
            &mut beams_bwd,
        );

        allele_counts.clear();
        allele_counts.resize(n_alleles, 0);
        let mut present = 0u32;
        for &a in ref_alleles.iter() {
            if a == crate::data::storage::AlleleCode::MISSING.raw() {
                continue;
            }
            let idx = a as usize;
            if idx < allele_counts.len() {
                allele_counts[idx] += 1;
                present += 1;
            }
        }

        if sampling[m] {
            for (i, _) in batch_haps.iter().enumerate() {
                if query_alleles[i].is_missing() {
                    continue;
                }
                let targ = query_alleles[i]
                    .as_allele()
                    .expect("non-missing strict query allele should be concrete");
                let freq = if present > 0 {
                    allele_counts.get(targ as usize).copied().unwrap_or(0) as f32 / present as f32
                } else {
                    0.0
                };
                if freq <= 0.0 {
                    continue;
                }
                let weight = prescan_match_weight(freq, min_freq);
                if weight <= 0.0 {
                    continue;
                }
                pbwt_bwd.select_donors_into(&beams_bwd[i], k_per_hap, &mut donors_buf);
                for &d in donors_buf.iter() {
                    let idx = d as usize;
                    if idx < n_ref_haps {
                        let ref_a = ref_alleles[idx];
                        if ref_a == crate::data::storage::AlleleCode::MISSING.raw() || ref_a != targ
                        {
                            continue;
                        }
                        global_scores[i][idx] += weight;
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
    PbwtPrescanWindowDiag {
        sampled_markers,
        total_markers: n_markers,
        min_gen_pos,
        max_gen_pos,
        distinct_gen_pos,
        non_finite_gen_pos,
    }
}

fn normalize_chrom_local(name: &str) -> &str {
    if name.len() >= 3 && name[..3].eq_ignore_ascii_case("chr") {
        &name[3..]
    } else {
        name
    }
}

#[inline]
fn push_needed_allele(
    needed: &mut Vec<(u8, f32)>,
    seen_alleles: &mut std::collections::HashSet<u8>,
    allele_idx: usize,
    weight: f32,
) {
    if allele_idx > u8::MAX as usize {
        return;
    }
    let allele = allele_idx as u8;
    if seen_alleles.insert(allele) {
        needed.push((allele, weight.max(0.0)));
    }
}

fn collect_target_positions(path: &Path) -> Result<(TargetMarkerIndex, usize)> {
    let (vcf_reader, mut reader) = crate::io::vcf::VcfReader::open(path)?;
    vcf_reader.samples_arc();
    let mut positions: TargetMarkerIndex = std::collections::HashMap::new();
    let mut total = 0usize;
    let mut line = String::new();

    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split('\t');
        let chrom = match parts.next() {
            Some(c) => c,
            None => continue,
        };
        let pos_str = match parts.next() {
            Some(p) => p,
            None => continue,
        };
        let pos: u32 = match pos_str.parse() {
            Ok(p) => p,
            Err(_) => continue,
        };
        let norm = normalize_chrom_local(chrom);
        positions
            .entry(norm.to_string())
            .or_insert_with(std::collections::HashSet::new)
            .insert(pos);
        total += 1;
    }

    Ok((positions, total))
}

fn open_ref_reader(path: &Path) -> Result<RefPanelReader> {
    let is_bref3 = path.extension().and_then(|e| e.to_str()) == Some("bref3");
    if is_bref3 {
        let stream_reader = crate::io::bref3::StreamingBref3Reader::open(path)?;
        let windowed = crate::io::bref3::StreamingBref3WindowReader::new(stream_reader);
        Ok(RefPanelReader::Bref3(windowed))
    } else {
        // Streaming reference VCF path: markers are parsed incrementally into
        // GenotypeColumn::from_alleles (Dense/Sparse selection), not the
        // batch dictionary compression path used by io/vcf.rs::VcfReader::read_all.
        let reader = crate::io::bref3::StreamingRefVcfReader::open(path)?;
        Ok(RefPanelReader::StreamingVcf(reader))
    }
}

const BREF3_CONVERT_MIN_BYTES: u64 = 500 * 1024 * 1024;
const IMPUTE_WINDOW_LOG_INTERVAL: usize = 250;

#[inline]
fn should_log_impute_window(window_idx: usize) -> bool {
    window_idx < 3 || window_idx % IMPUTE_WINDOW_LOG_INTERVAL == 0
}

fn should_convert_ref_to_bref3(config: &Config, ref_path: &Path) -> bool {
    if ref_path.extension().and_then(|e| e.to_str()) == Some("bref3") {
        return false;
    }
    if let Some(region) = config.chrom.as_deref() {
        if region.contains(':') && region.contains('-') {
            return false;
        }
    }
    let Ok(meta) = std::fs::metadata(ref_path) else {
        return false;
    };
    meta.len() >= BREF3_CONVERT_MIN_BYTES
}

fn cache_is_fresh(cache_path: &Path, ref_path: &Path) -> bool {
    let Ok(cache_meta) = std::fs::metadata(cache_path) else {
        return false;
    };
    if cache_meta.len() == 0 {
        return false;
    }
    let Ok(ref_meta) = std::fs::metadata(ref_path) else {
        return false;
    };
    match (cache_meta.modified(), ref_meta.modified()) {
        (Ok(cache_time), Ok(ref_time)) => cache_time >= ref_time,
        _ => true,
    }
}

fn ensure_binary_reference(ref_path: &Path, config: &Config) -> Result<PathBuf> {
    if ref_path.extension().and_then(|e| e.to_str()) == Some("bref3") {
        return Ok(ref_path.to_path_buf());
    }
    if !should_convert_ref_to_bref3(config, ref_path) {
        return Ok(ref_path.to_path_buf());
    }

    let mut candidates: Vec<PathBuf> = Vec::new();
    candidates.push(ref_path.with_extension("bref3"));
    candidates.push(config.out.with_extension("ref.bref3"));
    let stem = ref_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("reference");
    candidates.push(std::env::temp_dir().join(format!("reagle_ref_cache_{}.bref3", stem)));

    for path in candidates {
        if path.exists() && cache_is_fresh(&path, ref_path) {
            eprintln!("Using cached BREF3 reference at {:?}", path);
            return Ok(path);
        }
        let tmp_path = path.with_extension("bref3.tmp");
        match convert_ref_vcf_to_bref3(ref_path, &tmp_path) {
            Ok(()) => {
                // NOTE: current conversion writes ALLELE_CODED BREF3 records.
                // It improves startup/IO and avoids repeated VCF parsing, but
                // does not by itself guarantee SeqCoded HMM dispatch.
                if let Err(err) = std::fs::rename(&tmp_path, &path) {
                    eprintln!(
                        "Reference conversion rename failed at {:?}: {}. Trying next location...",
                        path, err
                    );
                    std::fs::remove_file(&tmp_path).ok();
                    continue;
                }
                eprintln!("Converted reference VCF to BREF3 at {:?}", path);
                return Ok(path);
            }
            Err(err) => {
                std::fs::remove_file(&tmp_path).ok();
                eprintln!(
                    "Reference conversion failed at {:?}: {}. Trying next location...",
                    path, err
                );
            }
        }
    }

    Ok(ref_path.to_path_buf())
}

#[derive(Debug)]
struct PrescanCacheMeta {
    path: PathBuf,
    n_ref_haps: usize,
    per_window_caps: Vec<usize>,
    window_handoff: Vec<(f64, f64)>,
}

struct PrescanCacheGuard {
    path: PathBuf,
}

impl PrescanCacheGuard {
    fn touch(&self) {}
}

impl Drop for PrescanCacheGuard {
    fn drop(&mut self) {
        std::fs::remove_file(&self.path).ok();
    }
}

enum ReferenceData {
    InMemory {
        windows: Vec<RefWindow>,
        packed_columns: Vec<Vec<PackedRefColumn>>,
        n_ref_haps: usize,
        per_window_caps: Vec<usize>,
        window_handoff: Vec<(f64, f64)>,
    },
    OnDisk {
        cache_meta: PrescanCacheMeta,
        guard: PrescanCacheGuard,
    },
}

impl ReferenceData {
    fn n_ref_haps(&self) -> usize {
        match self {
            Self::InMemory { n_ref_haps, .. } => *n_ref_haps,
            Self::OnDisk { cache_meta, .. } => cache_meta.n_ref_haps,
        }
    }

    fn per_window_caps(&self) -> &[usize] {
        match self {
            Self::InMemory {
                per_window_caps, ..
            } => per_window_caps.as_slice(),
            Self::OnDisk { cache_meta, .. } => cache_meta.per_window_caps.as_slice(),
        }
    }

    fn window_handoff(&self) -> &[(f64, f64)] {
        match self {
            Self::InMemory { window_handoff, .. } => window_handoff.as_slice(),
            Self::OnDisk { cache_meta, .. } => cache_meta.window_handoff.as_slice(),
        }
    }
}

fn packed_column_size_bytes(col: &PackedRefColumn) -> u64 {
    match col {
        PackedRefColumn::Bits { words, missing, .. } => {
            (words.len() as u64 * 8) + (missing.len() as u64 * 8)
        }
        PackedRefColumn::Bytes { alleles } => alleles.len() as u64,
    }
}

fn estimate_ref_window_bytes(ref_window: &RefWindow, packed_cols: &[PackedRefColumn]) -> u64 {
    let mut bytes = 0u64;
    for col in ref_window.ref_columns.iter() {
        bytes = bytes.saturating_add(col.size_bytes() as u64);
    }
    for col in packed_cols.iter() {
        bytes = bytes.saturating_add(packed_column_size_bytes(col));
    }
    let marker_overhead = ref_window.markers.len().saturating_mul(64) as u64;
    bytes.saturating_add(marker_overhead)
}

struct PrescanTargetEntry {
    phased_target: GenotypeMatrix<Phased, AnyMarkerSpace>,
    alignment: MarkerAlignment<AnyMarkerSpace, RefWindowSpace>,
}

fn trim_alignment_for_prescan_cache(
    mut alignment: MarkerAlignment<AnyMarkerSpace, RefWindowSpace>,
) -> MarkerAlignment<AnyMarkerSpace, RefWindowSpace> {
    // Prescan scoring only uses target_to_ref + reverse_map_allele (allele_mappings).
    // Dropping ref_to_target avoids storing a second large mapping that is unused
    // during prescan and materially increases cache memory.
    alignment.ref_to_target.clear();
    alignment.ref_to_target.shrink_to_fit();
    alignment
}

fn estimate_target_entry_bytes(entry: &PrescanTargetEntry) -> u64 {
    let target_bytes = entry.phased_target.size_bytes() as u64;
    let align_bytes = (entry.alignment.target_to_ref.len().saturating_mul(16) as u64)
        + entry.alignment.allele_mappings.len().saturating_mul(32) as u64;
    target_bytes.saturating_add(align_bytes)
}

fn compute_per_window_cap(
    n_ref_haps: usize,
    n_ref_markers: usize,
    n_target_markers: usize,
    available_bytes: u64,
    n_threads: usize,
    safe_bytes_per_thread: u64,
    force_full_panel: bool,
) -> usize {
    if n_ref_haps <= SMALL_PANEL_FULL_CAP_HAPS {
        return n_ref_haps.max(1);
    }
    let mut per_window_cap_window = if force_full_panel {
        n_ref_haps.max(1)
    } else {
        let per_state_bytes = estimate_per_state_bytes(n_ref_markers, n_target_markers);
        let mut cap = if per_state_bytes == 0 {
            0
        } else {
            let full_panel_bytes = per_state_bytes
                .saturating_mul(n_ref_haps)
                .saturating_mul(n_threads.max(1));
            let can_fit_full_panel = available_bytes > 0
                && full_panel_bytes as f64 <= (available_bytes as f64 * FULL_PANEL_RAM_FRACTION);
            if can_fit_full_panel {
                n_ref_haps
            } else {
                (safe_bytes_per_thread as usize) / per_state_bytes
            }
        };
        if cap == 0 {
            cap = 1;
        }
        cap
    };
    let cap = n_ref_haps.max(1);
    per_window_cap_window = per_window_cap_window.min(cap).max(1);
    per_window_cap_window
}

fn count_target_markers_in_ref_window<Space>(
    markers: &Markers<Space>,
    target_positions: &TargetMarkerIndex,
) -> usize {
    let mut count = 0usize;
    for m in 0..markers.len() {
        let marker = markers.marker(MarkerIdx::new(m as u32));
        let chrom = markers.chrom_name(marker.chrom).unwrap_or("");
        let norm = normalize_chrom_local(chrom);
        if target_positions
            .get(norm)
            .is_some_and(|positions| positions.contains(&marker.pos))
        {
            count += 1;
        }
    }
    count
}

fn prepare_reference_data(
    ref_path: &Path,
    streaming_config: &StreamingConfig,
    gen_maps: &GeneticMaps,
    target_positions: &TargetMarkerIndex,
    available_bytes: u64,
    n_threads: usize,
    safe_bytes_per_thread: u64,
    force_full_panel: bool,
) -> Result<ReferenceData> {
    let memory_budget = if available_bytes == 0 {
        0u64
    } else {
        (available_bytes as f64 * REF_PANEL_RAM_FRACTION) as u64
    };
    let mut use_in_memory = memory_budget > 0;

    let (tx, rx) = std::sync::mpsc::sync_channel::<Result<Option<RefWindow>>>(2);
    let ref_path = ref_path.to_path_buf();
    let streaming_config = streaming_config.clone();
    let gen_maps_thread = gen_maps.clone();
    let target_positions = target_positions.clone();
    let target_positions_reader = target_positions.clone();
    let reader_handle = std::thread::spawn(move || {
        let mut ref_reader = match open_ref_reader(&ref_path) {
            Ok(reader) => reader,
            Err(err) => {
                tx.send(Err(err.into())).ok();
                return Ok(());
            }
        };
        loop {
            let result = ref_reader.next_window(
                &streaming_config,
                &gen_maps_thread,
                Some(&target_positions_reader),
            );
            match result {
                Ok(Some(window)) => {
                    if tx.send(Ok(Some(window))).is_err() {
                        break;
                    }
                }
                Ok(None) => {
                    tx.send(Ok(None)).ok();
                    break;
                }
                Err(err) => {
                    tx.send(Err(err.into())).ok();
                    break;
                }
            }
        }
        Ok::<(), ReagleError>(())
    });
    let mut n_ref_haps = 0usize;
    let mut per_window_caps: Vec<usize> = Vec::new();
    let mut window_handoff: Vec<(f64, f64)> = Vec::new();
    let mut windows: Vec<RefWindow> = Vec::new();
    let mut packed_columns: Vec<Vec<PackedRefColumn>> = Vec::new();
    let mut total_bytes: u64 = 0;

    let mut cache_path: Option<PathBuf> = None;
    let mut cache_writer: Option<PrescanCacheWriter> = None;

    let read_result: Result<()> = (|| {
        for msg in rx {
            let ref_window = match msg? {
                Some(window) => window,
                None => break,
            };
            let n_ref_markers = ref_window.markers.len();
            if n_ref_markers == 0 {
                continue;
            }

            if n_ref_haps == 0 {
                n_ref_haps = ref_window
                    .ref_columns
                    .first()
                    .map(|c| c.n_haplotypes())
                    .unwrap_or(0);
                if n_ref_haps == 0 {
                    continue;
                }
            }

            let n_target_markers_window =
                count_target_markers_in_ref_window(&ref_window.markers, &target_positions);
            let per_window_cap_window = compute_per_window_cap(
                n_ref_haps,
                n_ref_markers,
                n_target_markers_window,
                available_bytes,
                n_threads,
                safe_bytes_per_thread,
                force_full_panel,
            );
            per_window_caps.push(per_window_cap_window);

            let output_start = ref_window.output_start.min(n_ref_markers.saturating_sub(1));
            let output_end = ref_window.output_end.min(n_ref_markers).max(1);
            let start_idx = output_start.min(n_ref_markers.saturating_sub(1));
            let end_idx = output_end.saturating_sub(1);
            let start_marker = ref_window.markers.marker(MarkerIdx::new(start_idx as u32));
            let end_marker = ref_window.markers.marker(MarkerIdx::new(end_idx as u32));
            let start_chrom = ref_window
                .markers
                .chrom_name(start_marker.chrom)
                .unwrap_or("");
            let end_chrom = ref_window
                .markers
                .chrom_name(end_marker.chrom)
                .unwrap_or("");
            let start_gen = gen_maps.gen_pos_by_name(start_chrom, start_marker.pos);
            let end_gen = gen_maps.gen_pos_by_name(end_chrom, end_marker.pos);
            window_handoff.push((start_gen, end_gen));

            if use_in_memory {
                let packed = pack_ref_columns(&ref_window.markers, &ref_window.ref_columns)?;
                total_bytes =
                    total_bytes.saturating_add(estimate_ref_window_bytes(&ref_window, &packed));
                if total_bytes <= memory_budget {
                    packed_columns.push(packed);
                    windows.push(ref_window);
                    continue;
                }

                use_in_memory = false;
                let path = create_temp_cache_path();
                let mut writer = PrescanCacheWriter::create(&path)?;
                writer.set_n_ref_haps(n_ref_haps);
                writer.write_header()?;
                for win in windows.iter() {
                    writer.write_window(win)?;
                }
                writer.write_window(&ref_window)?;
                windows.clear();
                packed_columns.clear();
                cache_path = Some(path);
                cache_writer = Some(writer);
            } else {
                if cache_writer.is_none() {
                    let path = cache_path.get_or_insert_with(create_temp_cache_path);
                    let mut writer = PrescanCacheWriter::create(path)?;
                    writer.set_n_ref_haps(n_ref_haps);
                    writer.write_header()?;
                    cache_writer = Some(writer);
                }
                if let Some(writer) = cache_writer.as_mut() {
                    writer.write_window(&ref_window)?;
                }
            }
        }
        Ok(())
    })();

    reader_handle
        .join()
        .map_err(|_| ReagleError::vcf("Reference reader thread panicked".to_string()))??;
    read_result?;

    if n_ref_haps == 0 {
        return Err(ReagleError::vcf(
            "Reference window scanning found no haplotypes".to_string(),
        ));
    }

    if use_in_memory {
        if windows.is_empty() {
            return Err(ReagleError::vcf(
                "Reference window scanning found no haplotypes".to_string(),
            ));
        }
        Ok(ReferenceData::InMemory {
            windows,
            packed_columns,
            n_ref_haps,
            per_window_caps,
            window_handoff,
        })
    } else {
        let path = cache_path
            .ok_or_else(|| ReagleError::vcf("Prescan cache path missing after scan".to_string()))?;
        let writer = cache_writer.ok_or_else(|| {
            ReagleError::vcf("Prescan cache writer missing after scan".to_string())
        })?;
        writer.finish()?;
        let meta = PrescanCacheMeta {
            path: path.clone(),
            n_ref_haps,
            per_window_caps,
            window_handoff,
        };
        let guard = PrescanCacheGuard { path };
        Ok(ReferenceData::OnDisk {
            cache_meta: meta,
            guard,
        })
    }
}

fn is_vcf_fully_phased(path: &Path) -> Result<bool> {
    let (_, mut reader) = crate::io::vcf::VcfReader::open(path)?;
    let mut line = String::new();
    while reader.read_line(&mut line)? != 0 {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            line.clear();
            continue;
        }
        let mut parts = trimmed.split('\t');
        if parts.next().is_none() {
            line.clear();
            continue;
        }
        if parts.next().is_none() {
            line.clear();
            continue;
        }
        let mut missing_fields = false;
        for _ in 0..6 {
            if parts.next().is_none() {
                missing_fields = true;
                break;
            }
        }
        if missing_fields {
            line.clear();
            continue;
        }
        let format = match parts.next() {
            Some(f) => f,
            None => continue,
        };
        let gt_idx = format.split(':').position(|f| f == "GT");
        if gt_idx.is_none() {
            continue;
        }
        let gt_idx = gt_idx.unwrap();
        for sample in parts {
            let mut fields = sample.split(':');
            let gt = fields.nth(gt_idx).unwrap_or("");
            if gt.contains('/') {
                line.clear();
                return Ok(false);
            }
            // Missing phased genotypes such as ".|." are allowed here.
        }
        line.clear();
    }
    Ok(true)
}

fn build_imputation_plan(
    target_path: &Path,
    streaming_config: &StreamingConfig,
    gen_maps: &GeneticMaps,
    per_window_cap: usize,
    window_top_k: usize,
    available_bytes: u64,
    imp_step_cm: f64,
    params: &crate::model::parameters::ModelParams,
    ref_data: &ReferenceData,
    telemetry: Option<&Arc<TelemetryBlackboard>>,
) -> Result<ImputationPlan> {
    let target_reader =
        StreamingVcfReader::open(target_path, gen_maps.clone(), streaming_config.clone())?;
    let n_target_haps = target_reader.samples_arc().len() * 2;
    if n_target_haps == 0 {
        return Err(ReagleError::vcf(
            "No target samples for pre-scan".to_string(),
        ));
    }

    let mut plan = ImputationPlan {
        n_ref_haps: 0,
        core_states: vec![Vec::new(); n_target_haps],
        window_intervals: vec![Vec::new(); n_target_haps],
        abyss_mask: vec![BitVec::new(); n_target_haps],
        per_window_cap: per_window_cap.max(1),
        per_window_caps: Vec::new(),
        io_to_planning_ranges: Vec::new(),
        planning_num_windows: 0,
        planning_handoff: Vec::new(),
        full_panel: false,
        stats: ImputationPlanStats::default(),
    };

    let avail = available_bytes;
    let prescan_avail = if avail == 0 {
        PRESCAN_FALLBACK_AVAIL_BYTES
    } else {
        avail
    };
    let n_ref_haps = ref_data.n_ref_haps();
    if n_ref_haps == 0 {
        return Err(ReagleError::vcf(
            "Reference window scanning found no haplotypes".to_string(),
        ));
    }
    plan.n_ref_haps = n_ref_haps;
    let window_handoff = ref_data.window_handoff();
    let per_window_caps = ref_data.per_window_caps();
    let (planning_handoff, io_to_planning_ranges) =
        build_planning_grid_from_handoff(window_handoff, params);
    plan.planning_handoff = planning_handoff.clone();
    let planning_num_windows = planning_handoff.len();
    if planning_num_windows == 0 {
        return Err(ReagleError::vcf(
            "Pre-scan produced no planning windows for LMS allocation".to_string(),
        ));
    }
    plan.io_to_planning_ranges = io_to_planning_ranges;
    plan.planning_num_windows = planning_num_windows;
    if avail == 0 {
        eprintln!(
            "Pre-scan: available memory unknown; using fallback={} MB for batching/cache",
            prescan_avail / (1024 * 1024)
        );
    }
    let batch_size =
        estimate_scan_batch_size(prescan_avail, n_ref_haps, n_target_haps, per_window_caps);
    let mut batch_start = 0usize;
    let batches_total = (n_target_haps + batch_size - 1) / batch_size;
    let prescan_start = std::time::Instant::now();
    let abyss_fallback_log_counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

    // Keep LMS allocation path; full-panel mode can degrade calibration
    // when donor-guided and overlap handoff logic are active.
    let can_full_panel = false;
    if can_full_panel {
        let num_windows = per_window_caps.len();
        if num_windows == 0 {
            return Err(ReagleError::vcf(
                "Pre-scan produced no windows for allocation".to_string(),
            ));
        }
        let plan_start = std::time::Instant::now();
        eprintln!(
            "Pre-scan: skipped (full panel global mode); ref_haps={}, windows={}",
            n_ref_haps, num_windows
        );
        if let Some(bb) = telemetry {
            bb.set_stage(crate::utils::telemetry::Stage::ImputationPlanning);
            bb.set_producer_stage(crate::utils::telemetry::Stage::ImputationPlanning);
            bb.set_op("Imputation prescan: skipped (full panel)");
        }
        plan.per_window_cap = n_ref_haps.max(1);
        plan.per_window_caps = per_window_caps.to_vec();
        plan.full_panel = true;
        for _ in 0..n_target_haps {
            plan.stats.update(n_ref_haps, 0, 0);
        }
        eprintln!(
            "Pre-scan summary: skipped full panel in {:.1}s",
            plan_start.elapsed().as_secs_f32()
        );
        return Ok(plan);
    }

    if window_handoff.is_empty() || per_window_caps.is_empty() {
        return Err(ReagleError::vcf(
            "Pre-scan produced no windows for LMS allocation".to_string(),
        ));
    }

    plan.per_window_caps = per_window_caps.to_vec();
    let mut planning_window_caps = vec![per_window_cap.max(1); planning_num_windows];
    for (io_idx, &(ps, pe)) in plan.io_to_planning_ranges.iter().enumerate() {
        let io_cap = per_window_caps
            .get(io_idx)
            .copied()
            .unwrap_or(per_window_cap.max(1))
            .max(1);
        let end = pe.min(planning_window_caps.len());
        let start = ps.min(end);
        for p in start..end {
            planning_window_caps[p] = planning_window_caps[p].min(io_cap);
        }
    }

    eprintln!(
        "Pre-scan: enabled (LMS allocation); target_haps={}, ref_haps={}, batch_size={}",
        n_target_haps, n_ref_haps, batch_size
    );
    if let Some(bb) = telemetry {
        let num_windows = window_handoff.len().max(1);
        let total_batches = batches_total.saturating_mul(num_windows).max(1);
        bb.set_stage(crate::utils::telemetry::Stage::ImputationPrescan);
        bb.set_producer_stage(crate::utils::telemetry::Stage::ImputationPrescan);
        bb.set_op("Imputation prescan: PBWT scoring");
        bb.set_total_windows(num_windows as u64);
        bb.set_current_window(0);
        bb.set_total_markers(total_batches as u64);
        bb.set_markers_processed(0);
    }

    let target_cache_budget = if avail == 0 {
        (prescan_avail as f64 * TARGET_CACHE_RAM_FRACTION) as u64
    } else {
        (avail as f64 * TARGET_CACHE_RAM_FRACTION) as u64
    };
    let mut target_cache: Option<Vec<Option<PrescanTargetEntry>>> = None;
    if target_cache_budget > 0 {
        let mut entries: Vec<Option<PrescanTargetEntry>> = Vec::new();
        let mut target_bytes = 0u64;
        let mut cache_ok = true;
        let mut target_reader =
            StreamingVcfReader::open(target_path, gen_maps.clone(), streaming_config.clone())?;

        match ref_data {
            ReferenceData::InMemory { windows, .. } => {
                for ref_window in windows.iter() {
                    let n_ref_markers = ref_window.markers.len();
                    if n_ref_markers == 0 {
                        entries.push(None);
                        continue;
                    }
                    let ref_chrom_idx = ref_window.markers.marker(MarkerIdx::new(0)).chrom;
                    let ref_chrom = ref_window
                        .markers
                        .chrom_name(ref_chrom_idx)
                        .unwrap_or("UNKNOWN");
                    let start_pos = ref_window.markers.marker(MarkerIdx::new(0)).pos;
                    let end_pos = ref_window
                        .markers
                        .marker(MarkerIdx::new((n_ref_markers - 1) as u32))
                        .pos;
                    let chrom_candidates = chrom_variants(ref_chrom);
                    let target_window = target_reader.load_window_for_region(
                        &chrom_candidates,
                        start_pos,
                        end_pos,
                    )?;
                    if let Some(target_window) = target_window {
                        let alignment = MarkerAlignment::new_with_ref_markers(
                            &target_window.genotypes,
                            &ref_window.markers,
                        );
                        let phased_target = target_window.genotypes.into_phased();
                        let entry = PrescanTargetEntry {
                            phased_target,
                            alignment: trim_alignment_for_prescan_cache(alignment),
                        };
                        target_bytes =
                            target_bytes.saturating_add(estimate_target_entry_bytes(&entry));
                        if target_bytes > target_cache_budget {
                            cache_ok = false;
                            break;
                        }
                        entries.push(Some(entry));
                    } else {
                        entries.push(None);
                    }
                }
            }
            ReferenceData::OnDisk { cache_meta, .. } => {
                let mut ref_reader = PrescanCacheReader::open(&cache_meta.path)?;
                loop {
                    let ref_window = ref_reader.next_window()?;
                    let Some(ref_window) = ref_window else { break };
                    let n_ref_markers = ref_window.markers.len();
                    if n_ref_markers == 0 {
                        entries.push(None);
                        continue;
                    }
                    let ref_chrom_idx = ref_window.markers.marker(MarkerIdx::new(0)).chrom;
                    let ref_chrom = ref_window
                        .markers
                        .chrom_name(ref_chrom_idx)
                        .unwrap_or("UNKNOWN");
                    let start_pos = ref_window.markers.marker(MarkerIdx::new(0)).pos;
                    let end_pos = ref_window
                        .markers
                        .marker(MarkerIdx::new((n_ref_markers - 1) as u32))
                        .pos;
                    let chrom_candidates = chrom_variants(ref_chrom);
                    let target_window = target_reader.load_window_for_region(
                        &chrom_candidates,
                        start_pos,
                        end_pos,
                    )?;
                    if let Some(target_window) = target_window {
                        let alignment = MarkerAlignment::new_with_ref_markers(
                            &target_window.genotypes,
                            &ref_window.markers,
                        );
                        let phased_target = target_window.genotypes.into_phased();
                        let entry = PrescanTargetEntry {
                            phased_target,
                            alignment: trim_alignment_for_prescan_cache(alignment),
                        };
                        target_bytes =
                            target_bytes.saturating_add(estimate_target_entry_bytes(&entry));
                        if target_bytes > target_cache_budget {
                            cache_ok = false;
                            break;
                        }
                        entries.push(Some(entry));
                    } else {
                        entries.push(None);
                    }
                }
            }
        }

        if cache_ok && !entries.is_empty() {
            eprintln!(
                "Pre-scan: cached target windows (~{} MB)",
                target_bytes / (1024 * 1024)
            );
            target_cache = Some(entries);
        } else if !entries.is_empty() {
            eprintln!("Pre-scan: target cache disabled (budget exceeded)");
        }
    }

    let mut global_scores: Vec<Vec<f32>> = Vec::new();
    let mut window_scores: Vec<Vec<f32>> = Vec::new();
    let mut best_window_scores: Vec<Vec<f32>> = Vec::new();
    let mut window_rank_hits: Vec<Vec<u32>> = Vec::new();
    let mut scores_by_window: Vec<Vec<Vec<(usize, f32)>>> = Vec::new();

    let mut on_disk_reader = match ref_data {
        ReferenceData::OnDisk { cache_meta, .. } => {
            Some(PrescanCacheReader::open(&cache_meta.path)?)
        }
        _ => None,
    };

    let cache_ready = target_cache
        .as_ref()
        .map(|c| c.iter().all(|e| e.is_some()))
        .unwrap_or(false);

    let mut batch_idx = 0usize;
    let mut window_span_cm: Option<f64> = None;
    let mut window_span_bp: Option<u64> = None;
    let mut exact_windows_total = 0usize;
    let mut pbwt_windows_total = 0usize;
    let mut pbwt_sampled_sum = 0usize;
    let mut pbwt_markers_sum = 0usize;
    let mut pbwt_sampled_min = usize::MAX;
    let mut pbwt_sampled_max = 0usize;
    let mut pbwt_flat_windows = 0usize;
    let mut pbwt_low_sample_windows = 0usize;
    let mut pbwt_non_finite_windows = 0usize;
    let mut adaptive_top_m_calls = 0usize;
    let mut adaptive_top_m_sum = 0usize;
    let mut adaptive_top_m_min = usize::MAX;
    let mut adaptive_top_m_max = 0usize;
    let mut adaptive_top_m_boosted = 0usize;
    let mut adaptive_top_m_reduced = 0usize;
    while batch_start < n_target_haps {
        batch_idx += 1;
        let batch_end = (batch_start + batch_size).min(n_target_haps);
        let batch_haps: Vec<usize> = (batch_start..batch_end).collect();
        let batch_len = batch_haps.len();
        if let Some(bb) = telemetry {
            let mut op = format!(
                "Imputation prescan: scoring batch {}/{} (batch_size={})",
                batch_idx,
                batches_total.max(1),
                batch_len
            );
            if let (Some(cm), Some(bp)) = (window_span_cm, window_span_bp) {
                op.push_str(&format!(", span_cm={:.3}, span_bp={}", cm, bp));
            }
            bb.set_op(&op);
        }

        let mut target_reader: Option<StreamingVcfReader> = if cache_ready {
            None
        } else {
            Some(StreamingVcfReader::open(
                target_path,
                gen_maps.clone(),
                streaming_config.clone(),
            )?)
        };

        if global_scores.len() > batch_len {
            global_scores.truncate(batch_len);
            window_scores.truncate(batch_len);
            best_window_scores.truncate(batch_len);
            window_rank_hits.truncate(batch_len);
            scores_by_window.truncate(batch_len);
        }
        while global_scores.len() < batch_len {
            global_scores.push(Vec::new());
            window_scores.push(Vec::new());
            best_window_scores.push(Vec::new());
            window_rank_hits.push(Vec::new());
            scores_by_window.push(Vec::new());
        }
        for list in scores_by_window.iter_mut() {
            list.clear();
            list.resize_with(planning_num_windows, Vec::new);
        }
        for i in 0..batch_len {
            if global_scores[i].len() != n_ref_haps {
                global_scores[i] = vec![0.0f32; n_ref_haps];
                window_scores[i] = vec![f32::NEG_INFINITY; n_ref_haps];
                best_window_scores[i] = vec![f32::NEG_INFINITY; n_ref_haps];
                window_rank_hits[i] = vec![0u32; n_ref_haps];
            } else {
                global_scores[i].fill(0.0);
                best_window_scores[i].fill(f32::NEG_INFINITY);
                window_rank_hits[i].fill(0);
            }
        }

        let mut window_idx = 0usize;
        match ref_data {
            ReferenceData::InMemory {
                windows,
                packed_columns,
                ..
            } => {
                for (idx, (ref_window, ref_columns)) in
                    windows.iter().zip(packed_columns.iter()).enumerate()
                {
                    let n_ref_markers = ref_window.markers.len();
                    if n_ref_markers == 0 {
                        continue;
                    }
                    if window_span_cm.is_none() || window_span_bp.is_none() {
                        let start_pos = ref_window.markers.marker(MarkerIdx::new(0)).pos;
                        let end_pos = ref_window
                            .markers
                            .marker(MarkerIdx::new((n_ref_markers - 1) as u32))
                            .pos;
                        let span_bp = end_pos.saturating_sub(start_pos);
                        let span_chrom = ref_window
                            .markers
                            .chrom_name(ref_window.markers.marker(MarkerIdx::new(0)).chrom)
                            .unwrap_or("");
                        let start_cm = gen_maps.gen_pos_by_name(span_chrom, start_pos);
                        let end_cm = gen_maps.gen_pos_by_name(span_chrom, end_pos);
                        window_span_bp = Some(span_bp.into());
                        window_span_cm = Some((end_cm - start_cm).abs());
                    }
                    // Derive per-window cap from the observed marker count to match
                    // the real workspace footprint (fwd/bwd/history scale with markers).
                    let per_window_cap_window = per_window_caps
                        .get(window_idx)
                        .copied()
                        .unwrap_or(per_window_cap.max(1));

                    let (alignment_cow, phased_target_cow) = if let Some(cache) =
                        target_cache.as_ref()
                    {
                        if let Some(Some(entry)) = cache.get(idx) {
                            (
                                Cow::Borrowed(&entry.alignment),
                                Cow::Borrowed(&entry.phased_target),
                            )
                        } else {
                            let ref_chrom_idx = ref_window.markers.marker(MarkerIdx::new(0)).chrom;
                            let ref_chrom = ref_window
                                .markers
                                .chrom_name(ref_chrom_idx)
                                .unwrap_or("UNKNOWN");
                            let start_pos = ref_window.markers.marker(MarkerIdx::new(0)).pos;
                            let end_pos = ref_window
                                .markers
                                .marker(MarkerIdx::new((n_ref_markers - 1) as u32))
                                .pos;
                            let chrom_candidates = chrom_variants(ref_chrom);
                            let reader = target_reader.as_mut().ok_or_else(|| {
                                ReagleError::vcf("Target reader missing in prescan".to_string())
                            })?;
                            let target_window = reader.load_window_for_region(
                                &chrom_candidates,
                                start_pos,
                                end_pos,
                            )?;
                            let Some(target_window) = target_window else {
                                continue;
                            };
                            let alignment = MarkerAlignment::new_with_ref_markers(
                                &target_window.genotypes,
                                &ref_window.markers,
                            );
                            let phased_target = target_window.genotypes.into_phased();
                            (Cow::Owned(alignment), Cow::Owned(phased_target))
                        }
                    } else {
                        let ref_chrom_idx = ref_window.markers.marker(MarkerIdx::new(0)).chrom;
                        let ref_chrom = ref_window
                            .markers
                            .chrom_name(ref_chrom_idx)
                            .unwrap_or("UNKNOWN");
                        let start_pos = ref_window.markers.marker(MarkerIdx::new(0)).pos;
                        let end_pos = ref_window
                            .markers
                            .marker(MarkerIdx::new((n_ref_markers - 1) as u32))
                            .pos;
                        let chrom_candidates = chrom_variants(ref_chrom);
                        let reader = target_reader.as_mut().ok_or_else(|| {
                            ReagleError::vcf("Target reader missing in prescan".to_string())
                        })?;
                        let target_window =
                            reader.load_window_for_region(&chrom_candidates, start_pos, end_pos)?;
                        let Some(target_window) = target_window else {
                            continue;
                        };
                        let alignment = MarkerAlignment::new_with_ref_markers(
                            &target_window.genotypes,
                            &ref_window.markers,
                        );
                        let phased_target = target_window.genotypes.into_phased();
                        (Cow::Owned(alignment), Cow::Owned(phased_target))
                    };
                    let alignment = alignment_cow.as_ref();
                    let phased_target = phased_target_cow.as_ref();

                    for w in window_scores.iter_mut() {
                        w.fill(f32::NEG_INFINITY);
                    }

                    let k_per_hap = per_window_cap_window
                        .saturating_mul(PBWT_PER_WINDOW_MULT)
                        .max(PBWT_MIN_PER_HAP)
                        .min(PBWT_MAX_PER_HAP)
                        .max(1);

                    let step_cm = PBWT_SELECT_BLOCK_CM.max(imp_step_cm);
                    let use_exact = should_use_exact_prescan(
                        n_ref_haps,
                        batch_haps.len(),
                        phased_target.n_markers(),
                    );
                    if use_exact {
                        exact_windows_total = exact_windows_total.saturating_add(1);
                        score_window_batch_exact_packed(
                            &batch_haps,
                            &phased_target,
                            &ref_window.markers,
                            ref_columns,
                            n_ref_haps,
                            &alignment,
                            &mut global_scores,
                            &mut window_scores,
                        );
                    } else {
                        pbwt_windows_total = pbwt_windows_total.saturating_add(1);
                        let diag = score_window_batch_pbwt_packed(
                            &batch_haps,
                            &phased_target,
                            &ref_window.markers,
                            ref_columns,
                            n_ref_haps,
                            &alignment,
                            gen_maps,
                            k_per_hap,
                            step_cm,
                            &mut global_scores,
                            &mut window_scores,
                        );
                        pbwt_sampled_sum = pbwt_sampled_sum.saturating_add(diag.sampled_markers);
                        pbwt_markers_sum = pbwt_markers_sum.saturating_add(diag.total_markers);
                        pbwt_sampled_min = pbwt_sampled_min.min(diag.sampled_markers);
                        pbwt_sampled_max = pbwt_sampled_max.max(diag.sampled_markers);
                        if diag.distinct_gen_pos <= 1
                            || (diag.max_gen_pos - diag.min_gen_pos).abs() <= 1e-12
                        {
                            pbwt_flat_windows = pbwt_flat_windows.saturating_add(1);
                        }
                        if diag.sampled_markers <= 2 {
                            pbwt_low_sample_windows = pbwt_low_sample_windows.saturating_add(1);
                        }
                        if diag.non_finite_gen_pos > 0 {
                            pbwt_non_finite_windows = pbwt_non_finite_windows.saturating_add(1);
                        }
                    }

                    let abyss_rank_cutoff =
                        compute_abyss_rank_cutoff(n_ref_haps, window_top_k.max(1));
                    for (i, _) in batch_haps.iter().enumerate() {
                        for (h, score) in window_scores[i].iter().copied().enumerate() {
                            if score > best_window_scores[i][h] {
                                best_window_scores[i][h] = score;
                            }
                        }

                        let abyss_top = select_top_k(&window_scores[i], abyss_rank_cutoff);
                        for (ref_idx, _) in abyss_top {
                            if ref_idx < window_rank_hits[i].len() {
                                window_rank_hits[i][ref_idx] =
                                    window_rank_hits[i][ref_idx].saturating_add(1);
                            }
                        }
                    }

                    // Persist per-window sparse scores for LMS allocator (top-M per window).
                    let (plan_start, plan_end) = plan
                        .io_to_planning_ranges
                        .get(window_idx)
                        .copied()
                        .unwrap_or((0, planning_num_windows.min(1)));
                    let (io_s, io_e) = window_handoff
                        .get(window_idx)
                        .copied()
                        .unwrap_or((f64::NAN, f64::NAN));
                    for i in 0..batch_len {
                        let base_top_m = per_window_cap_window
                            .saturating_mul(PBWT_PER_WINDOW_MULT)
                            .max(per_window_cap_window)
                            .min(n_ref_haps.max(1));
                        let (top_m, top) = select_top_k_adaptive_with_support(
                            &window_scores[i],
                            base_top_m,
                            per_window_cap_window,
                            n_ref_haps,
                        );
                        adaptive_top_m_calls = adaptive_top_m_calls.saturating_add(1);
                        adaptive_top_m_sum = adaptive_top_m_sum.saturating_add(top_m);
                        adaptive_top_m_min = adaptive_top_m_min.min(top_m);
                        adaptive_top_m_max = adaptive_top_m_max.max(top_m);
                        if top_m > base_top_m {
                            adaptive_top_m_boosted = adaptive_top_m_boosted.saturating_add(1);
                        } else if top_m < base_top_m {
                            adaptive_top_m_reduced = adaptive_top_m_reduced.saturating_add(1);
                        }
                        distribute_scores_to_planning_bins(
                            &top,
                            io_s,
                            io_e,
                            plan_start,
                            plan_end,
                            &planning_handoff,
                            &mut scores_by_window[i],
                        );
                    }

                    window_idx += 1;
                    if let Some(bb) = telemetry {
                        bb.set_current_window(window_idx as u64);
                        bb.add_markers(1);
                    }
                }
            }
            ReferenceData::OnDisk { .. } => {
                let Some(ref mut ref_reader) = on_disk_reader else {
                    return Err(ReagleError::vcf(
                        "Prescan cache reader unavailable".to_string(),
                    ));
                };
                ref_reader.rewind()?;
                loop {
                    let ref_window = ref_reader.next_window()?;
                    let Some(ref_window) = ref_window else { break };

                    let idx = window_idx;
                    let n_ref_markers = ref_window.markers.len();
                    if n_ref_markers == 0 {
                        continue;
                    }
                    if window_span_cm.is_none() || window_span_bp.is_none() {
                        let start_pos = ref_window.markers.marker(MarkerIdx::new(0)).pos;
                        let end_pos = ref_window
                            .markers
                            .marker(MarkerIdx::new((n_ref_markers - 1) as u32))
                            .pos;
                        let span_bp = end_pos.saturating_sub(start_pos);
                        let span_chrom = ref_window
                            .markers
                            .chrom_name(ref_window.markers.marker(MarkerIdx::new(0)).chrom)
                            .unwrap_or("");
                        let start_cm = gen_maps.gen_pos_by_name(span_chrom, start_pos);
                        let end_cm = gen_maps.gen_pos_by_name(span_chrom, end_pos);
                        window_span_bp = Some(span_bp.into());
                        window_span_cm = Some((end_cm - start_cm).abs());
                    }
                    // Derive per-window cap from the observed marker count to match
                    // the real workspace footprint (fwd/bwd/history scale with markers).
                    let per_window_cap_window = per_window_caps
                        .get(window_idx)
                        .copied()
                        .unwrap_or(per_window_cap.max(1));

                    let (alignment_cow, phased_target_cow) = if let Some(cache) =
                        target_cache.as_ref()
                    {
                        if let Some(Some(entry)) = cache.get(idx) {
                            (
                                Cow::Borrowed(&entry.alignment),
                                Cow::Borrowed(&entry.phased_target),
                            )
                        } else {
                            let ref_chrom_idx = ref_window.markers.marker(MarkerIdx::new(0)).chrom;
                            let ref_chrom = ref_window
                                .markers
                                .chrom_name(ref_chrom_idx)
                                .unwrap_or("UNKNOWN");
                            let start_pos = ref_window.markers.marker(MarkerIdx::new(0)).pos;
                            let end_pos = ref_window
                                .markers
                                .marker(MarkerIdx::new((n_ref_markers - 1) as u32))
                                .pos;

                            let chrom_candidates = chrom_variants(ref_chrom);
                            let reader = target_reader.as_mut().ok_or_else(|| {
                                ReagleError::vcf("Target reader missing in prescan".to_string())
                            })?;
                            let target_window = reader.load_window_for_region(
                                &chrom_candidates,
                                start_pos,
                                end_pos,
                            )?;
                            let Some(target_window) = target_window else {
                                continue;
                            };

                            let alignment = MarkerAlignment::new_with_ref_markers(
                                &target_window.genotypes,
                                &ref_window.markers,
                            );
                            let phased_target = target_window.genotypes.into_phased();
                            (Cow::Owned(alignment), Cow::Owned(phased_target))
                        }
                    } else {
                        let ref_chrom_idx = ref_window.markers.marker(MarkerIdx::new(0)).chrom;
                        let ref_chrom = ref_window
                            .markers
                            .chrom_name(ref_chrom_idx)
                            .unwrap_or("UNKNOWN");
                        let start_pos = ref_window.markers.marker(MarkerIdx::new(0)).pos;
                        let end_pos = ref_window
                            .markers
                            .marker(MarkerIdx::new((n_ref_markers - 1) as u32))
                            .pos;

                        let chrom_candidates = chrom_variants(ref_chrom);
                        let reader = target_reader.as_mut().ok_or_else(|| {
                            ReagleError::vcf("Target reader missing in prescan".to_string())
                        })?;
                        let target_window =
                            reader.load_window_for_region(&chrom_candidates, start_pos, end_pos)?;
                        let Some(target_window) = target_window else {
                            continue;
                        };

                        let alignment = MarkerAlignment::new_with_ref_markers(
                            &target_window.genotypes,
                            &ref_window.markers,
                        );
                        let phased_target = target_window.genotypes.into_phased();
                        (Cow::Owned(alignment), Cow::Owned(phased_target))
                    };
                    let alignment = alignment_cow.as_ref();
                    let phased_target = phased_target_cow.as_ref();

                    for w in window_scores.iter_mut() {
                        w.fill(f32::NEG_INFINITY);
                    }

                    let k_per_hap = per_window_cap_window
                        .saturating_mul(PBWT_PER_WINDOW_MULT)
                        .max(PBWT_MIN_PER_HAP)
                        .min(PBWT_MAX_PER_HAP)
                        .max(1);

                    let step_cm = PBWT_SELECT_BLOCK_CM.max(imp_step_cm);
                    let use_exact = should_use_exact_prescan(
                        n_ref_haps,
                        batch_haps.len(),
                        phased_target.n_markers(),
                    );
                    if use_exact {
                        exact_windows_total = exact_windows_total.saturating_add(1);
                        score_window_batch_exact_packed(
                            &batch_haps,
                            &phased_target,
                            &ref_window.markers,
                            &ref_window.columns,
                            n_ref_haps,
                            &alignment,
                            &mut global_scores,
                            &mut window_scores,
                        );
                    } else {
                        pbwt_windows_total = pbwt_windows_total.saturating_add(1);
                        let diag = score_window_batch_pbwt_packed(
                            &batch_haps,
                            &phased_target,
                            &ref_window.markers,
                            &ref_window.columns,
                            n_ref_haps,
                            &alignment,
                            gen_maps,
                            k_per_hap,
                            step_cm,
                            &mut global_scores,
                            &mut window_scores,
                        );
                        pbwt_sampled_sum = pbwt_sampled_sum.saturating_add(diag.sampled_markers);
                        pbwt_markers_sum = pbwt_markers_sum.saturating_add(diag.total_markers);
                        pbwt_sampled_min = pbwt_sampled_min.min(diag.sampled_markers);
                        pbwt_sampled_max = pbwt_sampled_max.max(diag.sampled_markers);
                        if diag.distinct_gen_pos <= 1
                            || (diag.max_gen_pos - diag.min_gen_pos).abs() <= 1e-12
                        {
                            pbwt_flat_windows = pbwt_flat_windows.saturating_add(1);
                        }
                        if diag.sampled_markers <= 2 {
                            pbwt_low_sample_windows = pbwt_low_sample_windows.saturating_add(1);
                        }
                        if diag.non_finite_gen_pos > 0 {
                            pbwt_non_finite_windows = pbwt_non_finite_windows.saturating_add(1);
                        }
                    }

                    let abyss_rank_cutoff =
                        compute_abyss_rank_cutoff(n_ref_haps, window_top_k.max(1));
                    for (i, _) in batch_haps.iter().enumerate() {
                        for (h, score) in window_scores[i].iter().copied().enumerate() {
                            if score > best_window_scores[i][h] {
                                best_window_scores[i][h] = score;
                            }
                        }

                        let abyss_top = select_top_k(&window_scores[i], abyss_rank_cutoff);
                        for (ref_idx, _) in abyss_top {
                            if ref_idx < window_rank_hits[i].len() {
                                window_rank_hits[i][ref_idx] =
                                    window_rank_hits[i][ref_idx].saturating_add(1);
                            }
                        }
                    }

                    // Persist per-window sparse scores for LMS allocator (top-M per window).
                    let (plan_start, plan_end) = plan
                        .io_to_planning_ranges
                        .get(window_idx)
                        .copied()
                        .unwrap_or((0, planning_num_windows.min(1)));
                    let (io_s, io_e) = window_handoff
                        .get(window_idx)
                        .copied()
                        .unwrap_or((f64::NAN, f64::NAN));
                    for i in 0..batch_len {
                        let base_top_m = per_window_cap_window
                            .saturating_mul(PBWT_PER_WINDOW_MULT)
                            .max(per_window_cap_window)
                            .min(n_ref_haps.max(1));
                        let (top_m, top) = select_top_k_adaptive_with_support(
                            &window_scores[i],
                            base_top_m,
                            per_window_cap_window,
                            n_ref_haps,
                        );
                        adaptive_top_m_calls = adaptive_top_m_calls.saturating_add(1);
                        adaptive_top_m_sum = adaptive_top_m_sum.saturating_add(top_m);
                        adaptive_top_m_min = adaptive_top_m_min.min(top_m);
                        adaptive_top_m_max = adaptive_top_m_max.max(top_m);
                        if top_m > base_top_m {
                            adaptive_top_m_boosted = adaptive_top_m_boosted.saturating_add(1);
                        } else if top_m < base_top_m {
                            adaptive_top_m_reduced = adaptive_top_m_reduced.saturating_add(1);
                        }
                        distribute_scores_to_planning_bins(
                            &top,
                            io_s,
                            io_e,
                            plan_start,
                            plan_end,
                            &planning_handoff,
                            &mut scores_by_window[i],
                        );
                    }

                    window_idx += 1;
                    if let Some(bb) = telemetry {
                        bb.set_current_window(window_idx as u64);
                        bb.add_markers(1);
                    }
                }
            }
        }

        for ws in scores_by_window.iter_mut() {
            aggregate_window_sparse_scores(ws);
        }

        let boundary_cm = window_boundaries_from_handoff(&planning_handoff);
        if !per_window_caps.is_empty() && per_window_caps.len() != window_handoff.len() {
            return Err(ReagleError::vcf(format!(
                "Per-window cap length mismatch (caps={}, bounds={})",
                per_window_caps.len(),
                window_handoff.len()
            )));
        }
        if window_idx != window_handoff.len() {
            return Err(ReagleError::vcf(format!(
                "Prescan window count mismatch (seen={}, bounds={})",
                window_idx,
                window_handoff.len()
            )));
        }

        let num_windows = planning_handoff.len();
        let per_window_caps_used = planning_window_caps.as_slice();
        let batch_results: Vec<_> = batch_haps
            .par_iter()
            .enumerate()
            .map(|(i, &hap_idx)| {
                let abyss_fallback_log_counter =
                    std::sync::Arc::clone(&abyss_fallback_log_counter);
                let mut abyss = bitvec![u64, Lsb0; 0; n_ref_haps];
                let mut abyss_count = 0usize;
                for h in 0..n_ref_haps {
                    let score = best_window_scores[i][h];
                    if window_rank_hits[i][h] == 0 || !score.is_finite() || score <= 0.0 {
                        abyss.set(h, true);
                        abyss_count += 1;
                    }
                }
                let window_scores_matrix = &scores_by_window[i];
                if window_scores_matrix.len() != planning_handoff.len() {
                    return Err(ReagleError::vcf(format!(
                        "Pre-scan window count mismatch for hap {} (scores={}, bounds={})",
                        hap_idx,
                        window_scores_matrix.len(),
                        planning_handoff.len()
                    )));
                }
                let per_window_cap_min = per_window_caps_used
                    .iter()
                    .copied()
                    .min()
                    .expect("planning_window_caps must be non-empty");
                // Safety floor: prevent abyss pruning from collapsing donor diversity.
                // Empirical note from chr21 top-k sweeps:
                // - runtime was mostly insensitive across top-k in [20, 120]
                // - lowering survivors affected metric tradeoffs (phase/IQS/Hellinger vs R²)
                // This floor is therefore a quality/robustness control, not a speed control.
                let min_survivors = 25usize.min(n_ref_haps);
                let mut survivors = n_ref_haps.saturating_sub(abyss_count);
                if survivors < min_survivors {
                    if survivors == 0
                        && abyss_fallback_log_counter
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                            < 5
                    {
                        eprintln!(
                            "Pre-scan warning: abyss masked all reference haplotypes for target hap {} (batch_idx={}); forcing donor floor={}",
                            hap_idx, i, min_survivors
                        );
                    }
                    let ranked = select_top_k_allow_zero(&global_scores[i], n_ref_haps);
                    for (h, _) in ranked {
                        if survivors >= min_survivors {
                            break;
                        }
                        if abyss[h] {
                            abyss.set(h, false);
                            abyss_count = abyss_count.saturating_sub(1);
                            survivors += 1;
                        }
                    }
                    if survivors < min_survivors {
                        for h in 0..n_ref_haps {
                            if survivors >= min_survivors {
                                break;
                            }
                            if abyss[h] {
                                abyss.set(h, false);
                                abyss_count = abyss_count.saturating_sub(1);
                                survivors += 1;
                            }
                        }
                    }
                }
                let (intervals, core) = if per_window_cap_min >= n_ref_haps {
                    // Keep abyss active even when we can fit the full panel:
                    // here abyss is a denoising prior over candidate donors,
                    // not only a memory-pruning mechanism.
                    let mut intervals = Vec::new();
                    let mut core = Vec::new();
                    let end = num_windows as u32;
                    for h in 0..n_ref_haps {
                        if abyss[h] {
                            continue;
                        }
                        let hap = RefHapId::new(h as u32);
                        intervals.push(HapIntervals {
                            hap,
                            intervals: vec![crate::model::state_allocator::WindowSpan::new(0, end)],
                        });
                        core.push(hap);
                    }
                    (intervals, core)
                } else {
                    let (candidate_haps, scores_by_hap) =
                        build_sparse_scores(window_scores_matrix, &abyss);
                    let global_slot_budget =
                        per_window_caps_used.iter().copied().sum::<usize>().max(1);
                    let allocation = crate::model::state_allocator::allocate_lms_sparse(
                        &scores_by_hap,
                        &candidate_haps,
                        num_windows,
                        &boundary_cm,
                        params,
                        n_ref_haps,
                        global_slot_budget,
                        per_window_caps_used,
                    );
                    let mut intervals = Vec::new();
                    for (hap, spans) in allocation.intervals_by_hap.into_iter() {
                        intervals.push(HapIntervals {
                            hap: RefHapId::new(hap as u32),
                            intervals: spans,
                        });
                    }
                    intervals.sort_unstable_by_key(|hi| hi.hap.as_u32());
                    let mut core = Vec::new();
                    let need_end = window_scores_matrix.len();
                    for hi in intervals.iter() {
                        if hi.intervals.len() == 1 && hi.intervals[0].is_full(need_end) {
                            core.push(hi.hap);
                        }
                    }
                    (intervals, core)
                };
                let core_len = core.len();
                let intervals_len = intervals.len();
                Ok((
                    hap_idx,
                    abyss,
                    intervals,
                    core,
                    core_len,
                    intervals_len,
                    abyss_count,
                ))
            })
            .collect::<Result<Vec<_>>>()?;

        for (hap_idx, abyss, intervals, core, core_len, intervals_len, abyss_count) in batch_results
        {
            plan.abyss_mask[hap_idx] = abyss;
            plan.window_intervals[hap_idx] = intervals;
            plan.core_states[hap_idx] = core;
            plan.stats.update(
                core_len,
                intervals_len.saturating_sub(core_len),
                abyss_count,
            );
        }

        batch_start = batch_end;
    }
    let elapsed = prescan_start.elapsed().as_secs_f32();
    let cache_hit = target_cache
        .as_ref()
        .map(|c| c.iter().filter(|e| e.is_some()).count())
        .unwrap_or(0);
    let cache_total = target_cache.as_ref().map(|c| c.len()).unwrap_or(0);
    eprintln!(
        "Pre-scan summary: batches={} io_windows={} planning_windows={} cache_hits={}/{} elapsed={:.1}s",
        batches_total.max(1),
        window_handoff.len(),
        planning_handoff.len(),
        cache_hit,
        cache_total,
        elapsed
    );
    if pbwt_windows_total > 0 {
        let sampled_avg = pbwt_sampled_sum as f64 / pbwt_windows_total as f64;
        let sampled_frac = if pbwt_markers_sum > 0 {
            pbwt_sampled_sum as f64 / pbwt_markers_sum as f64
        } else {
            0.0
        };
        eprintln!(
            "Pre-scan diagnostics: exact_windows={} pbwt_windows={} pbwt_sampled[min/avg/max]={}/{:.1}/{} pbwt_sampled_frac={:.4} pbwt_flat_windows={} pbwt_low_sample_windows={} pbwt_non_finite_windows={}",
            exact_windows_total,
            pbwt_windows_total,
            pbwt_sampled_min.min(pbwt_sampled_max),
            sampled_avg,
            pbwt_sampled_max,
            sampled_frac,
            pbwt_flat_windows,
            pbwt_low_sample_windows,
            pbwt_non_finite_windows
        );
    } else {
        eprintln!(
            "Pre-scan diagnostics: exact_windows={} pbwt_windows=0",
            exact_windows_total
        );
    }
    if adaptive_top_m_calls > 0 {
        let top_m_avg = adaptive_top_m_sum as f64 / adaptive_top_m_calls as f64;
        eprintln!(
            "Pre-scan adaptive top-M: calls={} min/avg/max={}/{:.1}/{} boosted={} reduced={}",
            adaptive_top_m_calls,
            adaptive_top_m_min.min(adaptive_top_m_max),
            top_m_avg,
            adaptive_top_m_max,
            adaptive_top_m_boosted,
            adaptive_top_m_reduced
        );
    }
    Ok(plan)
}

struct SampleImputationResult {
    sample_idx: usize,
    hap_alt_probs: (Option<Vec<f32>>, Option<Vec<f32>>),
    hap_posteriors: (Option<SparseHapPosteriors>, Option<SparseHapPosteriors>),
}

#[derive(Clone, Debug)]
struct SparseHapPosteriors {
    local_marker_indices: Vec<usize>,
    values: Vec<AllelePosteriors>,
}

impl SparseHapPosteriors {
    #[inline]
    fn get(&self, local_m: usize) -> Option<&AllelePosteriors> {
        let pos = self.local_marker_indices.binary_search(&local_m).ok()?;
        self.values.get(pos)
    }

    #[inline]
    fn take(&mut self, local_m: usize) -> Option<AllelePosteriors> {
        let pos = self.local_marker_indices.binary_search(&local_m).ok()?;
        self.local_marker_indices.remove(pos);
        Some(self.values.remove(pos))
    }

    #[inline]
    fn put(&mut self, local_m: usize, value: AllelePosteriors) {
        match self.local_marker_indices.binary_search(&local_m) {
            Ok(pos) => self.values[pos] = value,
            Err(pos) => {
                self.local_marker_indices.insert(pos, local_m);
                self.values.insert(pos, value);
            }
        }
    }
}

impl SampleImputationResult {
    #[inline]
    fn hap_posterior(&self, hap: usize, local_m: usize) -> Option<&AllelePosteriors> {
        let post = if hap == 0 {
            self.hap_posteriors.0.as_ref()
        } else {
            self.hap_posteriors.1.as_ref()
        };
        post.and_then(|p| p.get(local_m))
    }

    #[inline]
    fn hap_alt_prob(&self, hap: usize, local_m: usize) -> Option<f32> {
        let alt = if hap == 0 {
            self.hap_alt_probs.0.as_ref()
        } else {
            self.hap_alt_probs.1.as_ref()
        };
        alt.and_then(|v| v.get(local_m)).copied()
    }

    #[inline]
    fn hap_prob(&self, hap: usize, local_m: usize, allele: u8) -> Option<f32> {
        if allele == crate::data::storage::AlleleCode::MISSING.raw() {
            return None;
        }
        if let Some(post) = self.hap_posterior(hap, local_m) {
            let q = post.prob(allele as usize);
            if q.is_finite() {
                return Some(q.clamp(0.0, 1.0));
            }
            return None;
        }
        if allele > 1 {
            return None;
        }
        self.hap_alt_prob(hap, local_m).map(|p_alt| {
            let p_alt = if p_alt.is_finite() {
                p_alt.clamp(0.0, 1.0)
            } else {
                0.5
            };
            if allele == 1 { p_alt } else { 1.0 - p_alt }
        })
    }

    fn swap_hap_posteriors_at(&mut self, local_m: usize) {
        let left = self.hap_posteriors.0.as_mut().and_then(|s| s.take(local_m));
        let right = self.hap_posteriors.1.as_mut().and_then(|s| s.take(local_m));
        if let Some(v) = left {
            let slot = self
                .hap_posteriors
                .1
                .get_or_insert_with(|| SparseHapPosteriors {
                    local_marker_indices: Vec::new(),
                    values: Vec::new(),
                });
            slot.put(local_m, v);
        }
        if let Some(v) = right {
            let slot = self
                .hap_posteriors
                .0
                .get_or_insert_with(|| SparseHapPosteriors {
                    local_marker_indices: Vec::new(),
                    values: Vec::new(),
                });
            slot.put(local_m, v);
        }
    }
}

struct ImputationHandoff {
    priors_id: Vec<HaplotypePriors>,
    orientation_weight_swap: Vec<f32>,
    prior_global_idx: Option<usize>,
    prior_gen_pos: Option<f64>,
}

struct ImputationWindowResults {
    all_results: Vec<SampleImputationResult>,
    ref_is_biallelic: Vec<bool>,
    overlap_start_idx: usize,
    handoff: Option<ImputationHandoff>,
    alt_prob_store: Option<AltProbDiskStoreView>,
}

struct AltProbDiskStoreBuilder {
    file: tempfile::NamedTempFile,
    output_markers: usize,
    n_samples: usize,
    writer: Arc<std::fs::File>,
}

struct AltProbDiskStoreView {
    map: Mmap,
    output_markers: usize,
    n_samples: usize,
}

impl AltProbDiskStoreBuilder {
    fn new(n_samples: usize, output_markers: usize) -> Result<Self> {
        let mut file = tempfile::NamedTempFile::new()?;
        let total_floats = n_samples
            .checked_mul(2)
            .and_then(|v| v.checked_mul(output_markers))
            .ok_or_else(|| ReagleError::vcf("alt-prob store size overflow".to_string()))?;
        let total_bytes = total_floats
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| ReagleError::vcf("alt-prob store byte-size overflow".to_string()))?;
        file.as_file().set_len(total_bytes as u64)?;
        if total_floats > 0 {
            let nan_chunk = vec![f32::NAN; 16_384];
            let mut remaining = total_floats;
            while remaining > 0 {
                let n = remaining.min(nan_chunk.len());
                // SAFETY: f32 slice is POD; this preserves exact NaN sentinels.
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        nan_chunk.as_ptr() as *const u8,
                        n * std::mem::size_of::<f32>(),
                    )
                };
                file.as_file_mut().write_all(bytes)?;
                remaining -= n;
            }
            file.as_file_mut().flush()?;
        }
        let writer = Arc::new(file.as_file().try_clone()?);
        Ok(Self {
            file,
            output_markers,
            n_samples,
            writer,
        })
    }

    fn writer(&self) -> Arc<std::fs::File> {
        self.writer.clone()
    }

    fn write_hap_probs_at(
        file: &std::fs::File,
        n_samples: usize,
        output_markers: usize,
        sample_idx: usize,
        hap: usize,
        probs: &[f32],
    ) -> Result<()> {
        if sample_idx >= n_samples || hap > 1 {
            return Err(ReagleError::vcf(format!(
                "alt-prob store index out of range: sample={} hap={}",
                sample_idx, hap
            )));
        }
        let sample = sample_idx_from_usize(sample_idx);
        let side = if hap == 0 { HapSide::H1 } else { HapSide::H2 };
        let offset_floats = sample
            .hap(side)
            .as_usize()
            .checked_mul(output_markers)
            .ok_or_else(|| ReagleError::vcf("alt-prob offset overflow".to_string()))?;
        let offset_bytes = offset_floats
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| ReagleError::vcf("alt-prob offset byte overflow".to_string()))?;
        if probs.len() != output_markers {
            return Err(ReagleError::vcf(format!(
                "alt-prob length mismatch: got={} expected={}",
                probs.len(),
                output_markers
            )));
        }

        #[cfg(unix)]
        {
            // SAFETY: f32 slice is POD; writing raw bytes preserves exact values.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    probs.as_ptr() as *const u8,
                    probs.len() * std::mem::size_of::<f32>(),
                )
            };
            let mut written = 0usize;
            while written < bytes.len() {
                let n = file.write_at(&bytes[written..], offset_bytes as u64 + written as u64)?;
                if n == 0 {
                    return Err(ReagleError::vcf(
                        "alt-prob write_at wrote zero bytes".to_string(),
                    ));
                }
                written += n;
            }
            Ok(())
        }
        #[cfg(not(unix))]
        {
            let _ = (file, offset_bytes, probs);
            Err(ReagleError::vcf(
                "alt-prob spill requires unix file write_at support".to_string(),
            ))
        }
    }

    fn finalize(mut self) -> Result<AltProbDiskStoreView> {
        self.file.as_file_mut().flush()?;
        let map = {
            // SAFETY: mapping a stable temp file for read-only lookups.
            unsafe { MmapOptions::new().map(self.file.as_file()) }?
        };
        Ok(AltProbDiskStoreView {
            map,
            output_markers: self.output_markers,
            n_samples: self.n_samples,
        })
    }
}

impl AltProbDiskStoreView {
    #[inline]
    fn get(&self, sample_idx: usize, hap: usize, local_m: usize) -> Option<f32> {
        if sample_idx >= self.n_samples || hap > 1 || local_m >= self.output_markers {
            return None;
        }
        let sample = sample_idx_from_usize(sample_idx);
        let side = if hap == 0 { HapSide::H1 } else { HapSide::H2 };
        let idx = sample
            .hap(side)
            .as_usize()
            .checked_mul(self.output_markers)?
            .checked_add(local_m)?;
        // SAFETY: mmap is page-aligned and sized to an integral count of f32 values.
        let values = unsafe {
            std::slice::from_raw_parts(
                self.map.as_ptr() as *const f32,
                self.map.len() / std::mem::size_of::<f32>(),
            )
        };
        let v = values.get(idx).copied()?;
        if v.is_nan() { None } else { Some(v) }
    }
}

fn split_hap_posteriors(
    posts: &mut Option<Vec<AllelePosteriors>>,
) -> (Option<Vec<f32>>, Option<SparseHapPosteriors>) {
    let Some(values) = posts.take() else {
        return (None, None);
    };
    let mut alt_probs = Vec::with_capacity(values.len());
    let mut sparse_idx: Vec<usize> = Vec::new();
    let mut sparse_vals: Vec<AllelePosteriors> = Vec::new();
    for (local_m, post) in values.into_iter().enumerate() {
        match post {
            AllelePosteriors::Biallelic(p_alt) => alt_probs.push(p_alt),
            other => {
                alt_probs.push(f32::NAN);
                sparse_idx.push(local_m);
                sparse_vals.push(other);
            }
        }
    }
    let sparse = if sparse_idx.is_empty() {
        None
    } else {
        Some(SparseHapPosteriors {
            local_marker_indices: sparse_idx,
            values: sparse_vals,
        })
    };
    (Some(alt_probs), sparse)
}

impl crate::pipelines::ImputationPipeline {
    /// Run streaming imputation pipeline
    #[instrument(name = "imputation_streaming", skip(self))]
    pub fn run_streaming(&mut self) -> Result<()> {
        // Imputation benefits from shorter windows than phasing defaults.
        // Use a segment-coupled window so each HMM run remains locally coherent
        // when target density is sparse but reference density is very high.
        let derived_window_cm = (self.config.imp_segment * 2.0).max(self.config.overlap * 2.0);
        let effective_window_cm = self.config.window.min(derived_window_cm.max(1.0));
        let streaming_config = StreamingConfig {
            window_cm: effective_window_cm,
            overlap_cm: self.config.overlap,
            buffer_cm: 1.0,
        };
        if (effective_window_cm - self.config.window).abs() > f32::EPSILON {
            eprintln!(
                "Imputation window adjusted: configured={:.3}cM effective={:.3}cM (segment-coupled)",
                self.config.window, effective_window_cm
            );
        }

        if let Some(bb) = &self.telemetry {
            bb.set_stage(crate::utils::telemetry::Stage::LoadingData);
            bb.set_producer_stage(crate::utils::telemetry::Stage::LoadingData);
            bb.set_consumer_stage(crate::utils::telemetry::Stage::LoadingData);
            bb.set_op("Preparing input");
            bb.set_producer_op("Preparing input");
        }

        let (target_positions_map, target_marker_count) =
            collect_target_positions(&self.config.target)?;
        let target_positions = if target_marker_count == 0 {
            None
        } else {
            Some(Arc::new(target_positions_map.clone()))
        };
        if target_positions.is_none() {
            return Err(ReagleError::vcf(
                "No target markers found while building marker index".to_string(),
            ));
        }
        eprintln!("Target marker index: {} positions", target_marker_count);

        let gen_maps = if let Some(ref map_path) = self.config.map {
            let mut chrom_name_buf: Vec<String> = Vec::new();

            for chrom in target_positions_map.keys() {
                for variant in chrom_variants(chrom) {
                    if !chrom_name_buf.iter().any(|c| c == &variant) {
                        chrom_name_buf.push(variant);
                    }
                }
            }
            if let Some(chrom) = self.config.chrom.as_deref() {
                for variant in chrom_variants(chrom) {
                    if !chrom_name_buf.iter().any(|c| c == &variant) {
                        chrom_name_buf.push(variant);
                    }
                }
            }

            let chrom_name_refs: Vec<&str> = chrom_name_buf.iter().map(String::as_str).collect();
            GeneticMaps::from_plink_file(map_path, &chrom_name_refs)?
        } else {
            GeneticMaps::new()
        };

        let ref_path = self
            .config
            .r#ref
            .as_ref()
            .ok_or_else(|| ReagleError::config("Reference panel required for imputation"))?;
        let ref_path = ensure_binary_reference(ref_path, &self.config)?;

        let mut input_target_path = self.config.target.clone();
        let mut input_tmp: Option<tempfile::TempDir> = None;
        if input_target_path.as_os_str() == "-" {
            let tmpdir = tempfile::tempdir()?;
            let tmp_path = tmpdir.path().join("stdin_target.vcf");
            let stdin = std::io::stdin();
            let mut reader = stdin.lock();
            let mut out = std::fs::File::create(&tmp_path)?;
            std::io::copy(&mut reader, &mut out)?;
            input_target_path = tmp_path;
            input_tmp = Some(tmpdir);
        } else if !input_target_path.exists() {
            return Err(ReagleError::config(format!(
                "Target VCF not found: {:?}",
                input_target_path
            )));
        }

        let mut phased_target_path = input_target_path.clone();
        let mut phased_tmp: Option<tempfile::TempDir> = None;
        // NOTE: imputation uses its own mismatch/recombination priors.
        if !is_vcf_fully_phased(&phased_target_path)? {
            eprintln!("Target is unphased; running phasing before pre-scan...");
            let phased_prefix = if self.config.out.as_os_str() != "-" {
                let out = &self.config.out;
                let parent = out.parent().unwrap_or_else(|| std::path::Path::new("."));
                let stem = out
                    .file_name()
                    .unwrap_or_else(|| std::ffi::OsStr::new("reagle_out"));
                parent.join(format!("{}_phased_target", stem.to_string_lossy()))
            } else {
                let tmpdir = tempfile::tempdir()?;
                let prefix = tmpdir.path().join("phased_target");
                phased_tmp = Some(tmpdir);
                prefix
            };
            let mut phase_config = self.config.clone();
            phase_config.target = input_target_path.clone();
            phase_config.r#ref = Some(ref_path.to_path_buf());
            phase_config.out = phased_prefix.clone();
            let mut phasing = crate::pipelines::phasing::PhasingPipeline::new(
                phase_config,
                self.telemetry.clone(),
            );
            phasing.run()?;
            phased_target_path = phased_prefix.with_extension("vcf.gz");
            if phased_tmp.is_none() {
                eprintln!("Phased target saved at {:?}", phased_target_path);
            }
        } else {
            eprintln!("Target already phased; skipping phasing before pre-scan.");
        }

        let mut n_threads = self
            .config
            .nthreads
            .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
            .unwrap_or(1);
        let mut avail_bytes = crate::utils::memory::available_memory_bytes().unwrap_or(0);
        if avail_bytes < MIN_AVAIL_BYTES_FOR_PLANNING {
            // Treat unknown/low memory as "planning disabled" to avoid
            // tiny caps in CI/small test runs.
            avail_bytes = 0;
        }
        let min_states = 64usize;
        let mut raw_budget = estimate_state_budget(
            avail_bytes,
            n_threads,
            self.config.window_markers,
            target_marker_count,
        );
        loop {
            let total_budget = raw_budget.max(1);
            if total_budget >= min_states || n_threads <= 1 {
                break;
            }
            n_threads = (n_threads / 2).max(1);
            raw_budget = estimate_state_budget(
                avail_bytes,
                n_threads,
                self.config.window_markers,
                target_marker_count,
            );
        }
        let total_budget = raw_budget.max(1);
        let force_full_panel = avail_bytes == 0 || raw_budget == 0;
        let per_window_cap = if force_full_panel {
            usize::MAX
        } else {
            total_budget
        };
        let per_window_cap = per_window_cap.max(1);

        eprintln!(
            "Imputation plan: per_window_cap={}, threads={}, available_mb={}",
            per_window_cap,
            n_threads,
            avail_bytes / (1024 * 1024)
        );

        let safe_bytes_per_thread = if n_threads == 0 {
            0u64
        } else {
            let budget = (avail_bytes as f64 * IMPUTE_RAM_FRACTION) as u64;
            let per_thread = budget / n_threads as u64;
            (per_thread as f64 * STATE_BUDGET_SAFETY) as u64
        };
        let prescan_force_full_panel = avail_bytes < MIN_AVAIL_BYTES_FOR_PLANNING;
        if let Some(bb) = &self.telemetry {
            bb.set_stage(crate::utils::telemetry::Stage::ImputationPrescan);
            bb.set_producer_stage(crate::utils::telemetry::Stage::ImputationPrescan);
            bb.set_op("Imputation prescan: reference prep");
        }
        let ref_data = prepare_reference_data(
            &ref_path,
            &streaming_config,
            &gen_maps,
            &target_positions_map,
            if force_full_panel { 0 } else { avail_bytes },
            n_threads,
            safe_bytes_per_thread,
            prescan_force_full_panel,
        )?;

        match &ref_data {
            ReferenceData::InMemory { .. } => {
                eprintln!("Reference mode: in-memory (single-pass)");
            }
            ReferenceData::OnDisk { guard, .. } => {
                eprintln!("Reference mode: prescan cache (double-pass)");
                guard.touch();
            }
        }

        if let Some(bb) = &self.telemetry {
            bb.set_stage(crate::utils::telemetry::Stage::ImputationPlanning);
            bb.set_producer_stage(crate::utils::telemetry::Stage::ImputationPlanning);
            bb.set_op("Imputation planning: window caps");
        }
        let plan = build_imputation_plan(
            &phased_target_path,
            &streaming_config,
            &gen_maps,
            per_window_cap,
            self.config.window_top_k.max(1),
            if force_full_panel { 0 } else { avail_bytes },
            self.config.imp_step as f64,
            &self.params,
            &ref_data,
            self.telemetry.as_ref(),
        )?;

        let full_panel_cap = plan
            .per_window_caps
            .iter()
            .copied()
            .min()
            .unwrap_or(plan.per_window_cap);
        let full_panel = full_panel_cap >= plan.n_ref_haps;
        if full_panel {
            eprintln!("Imputation mode: full panel per window (abyss still active)");
        } else {
            eprintln!(
                "Imputation mode: LMS allocation (per_window_cap={})",
                plan.per_window_cap
            );
        }
        log_imputation_plan_summary(&plan);

        if let Some(bb) = &self.telemetry {
            bb.set_total_windows(plan.per_window_caps.len() as u64);
            bb.set_current_window(0);
        }

        let mut ref_data = ref_data;
        let target_was_unphased_for_impute = !is_vcf_fully_phased(&input_target_path)?;
        // Always use phased target haplotypes for emissions to preserve LD signal.
        // If the input was unphased, we will still treat phase as uncertain in
        // the emission model (heterozygotes → 0.5/0.5 per haplotype).
        let use_phased_for_impute = true;
        let target_path_for_impute = phased_target_path.clone();
        if target_was_unphased_for_impute {
            eprintln!(
                "Target was unphased at input; using phased target for imputation (phase-uncertain emissions)."
            );
        } else {
            eprintln!("Target already phased; using phased target directly for imputation.");
        }
        let mut target_reader = StreamingVcfReader::open(
            &target_path_for_impute,
            gen_maps.clone(),
            streaming_config.clone(),
        )?;
        let mut target_reader_pl = if use_phased_for_impute {
            // Pull PL/GL from the original target VCF (if present).
            Some(StreamingVcfReader::open(
                &input_target_path,
                gen_maps.clone(),
                streaming_config.clone(),
            )?)
        } else {
            None
        };
        let target_samples = target_reader.samples_arc();
        let n_target_samples = target_samples.len();
        if n_target_samples == 0 {
            return Err(ReagleError::vcf(
                "No target samples found in input VCF".to_string(),
            ));
        }

        let n_ref_pool = plan.n_ref_haps.max(1);
        let n_target_haps = n_target_samples.saturating_mul(2);
        self.params = crate::model::parameters::ModelParams::for_imputation(
            n_ref_pool,
            self.config.ne,
            self.config.err,
        );
        // Imputation transitions copy from the reference panel only; target batch
        // size must not alter Li-Stephens transition physics.
        let impute_recomb_intensity = (0.04 * self.config.ne / n_ref_pool as f32)
            .min(ModelParams::MAX_RECOMB_INTENSITY)
            .max(1e-6);
        self.params.recomb_intensity = impute_recomb_intensity;
        eprintln!(
            "Imputation recomb_intensity: {:.6} (source=config-ne, n_ref_haps={}, n_target_haps={}, n_transition_haps={})",
            self.params.recomb_intensity, n_ref_pool, n_target_haps, n_ref_pool,
        );
        // Do not inherit phasing mismatch estimates for imputation. Imputation
        // should use the Li-Stephens mismatch prior (or user override) tied to
        // the reference panel, not phasing-specific error rates.
        self.params
            .set_n_states(n_ref_pool.saturating_sub(2).max(1));

        let output_path = self.config.out.with_extension("vcf.gz");
        eprintln!("Writing output to {:?}", output_path);
        let mut writer = VcfWriter::create(&output_path, target_samples.clone())?;

        let mut imp_overlap: Option<PhasedOverlap> = None;
        let mut warned_no_overlap = false;
        let mut header_written = false;
        let mut total_markers = 0usize;
        let mut window_idx = 0usize;
        let mut sample_error_rates =
            vec![self.params.p_mismatch.clamp(1e-6, 0.5); n_target_samples];

        match &mut ref_data {
            ReferenceData::InMemory { windows, .. } => {
                for ref_window in windows.iter_mut() {
                    if let Some(bb) = &self.telemetry {
                        bb.set_producer_stage(crate::utils::telemetry::Stage::LoadingData);
                        bb.set_producer_op(&format!("Loading ref window {}", window_idx + 1));
                        bb.set_op(&format!("Loading ref window {}", window_idx + 1));
                        bb.set_current_window((window_idx + 1) as u64);
                    }

                    let n_ref_markers = ref_window.markers.len();
                    if n_ref_markers == 0 {
                        window_idx += 1;
                        continue;
                    }

                    let ref_chrom_idx = ref_window.markers.marker(MarkerIdx::new(0)).chrom;
                    let ref_chrom = ref_window
                        .markers
                        .chrom_name(ref_chrom_idx)
                        .unwrap_or("UNKNOWN");
                    let start_pos = ref_window.markers.marker(MarkerIdx::new(0)).pos;
                    let end_pos = ref_window
                        .markers
                        .marker(MarkerIdx::new((n_ref_markers - 1) as u32))
                        .pos;

                    let chrom_candidates = chrom_variants(ref_chrom);
                    let target_window = target_reader.load_window_for_region(
                        &chrom_candidates,
                        start_pos,
                        end_pos,
                    )?;
                    let Some(target_window) = target_window else {
                        window_idx += 1;
                        continue;
                    };
                    let target_window_source = if use_phased_for_impute {
                        let reader_pl = target_reader_pl.as_mut().ok_or_else(|| {
                            ReagleError::vcf(
                                "Internal error: missing original target reader for imputation"
                                    .to_string(),
                            )
                        })?;
                        match reader_pl.load_window_for_region(
                            &chrom_candidates,
                            start_pos,
                            end_pos,
                        )? {
                            Some(window) => window,
                            None => {
                                return Err(ReagleError::vcf(format!(
                                    "Original target window missing while phased window exists: chrom={} start={} end={}",
                                    ref_chrom, start_pos, end_pos
                                )));
                            }
                        }
                    } else {
                        target_window.clone()
                    };

                    // Align using the phased target markers; PL/GL lookups share the same
                    // marker set, so indices should remain consistent.
                    let alignment = MarkerAlignment::new_with_ref_markers(
                        &target_window.genotypes,
                        &ref_window.markers,
                    );

                    let phased_target = target_window.genotypes.clone().into_phased();
                    let phased_target_pl =
                        Some(target_window_source.genotypes.clone().into_phased());
                    let target_missing = Some(&target_window_source.genotypes);
                    if !header_written {
                        writer.write_header_extended(
                            &ref_window.markers,
                            true,
                            self.config.gp,
                            self.config.ap,
                        )?;
                        header_written = true;
                    }

                    let should_log = should_log_impute_window(window_idx);
                    if should_log {
                        eprintln!(
                            "  Imputing Window {} ({} markers, ref global {}..{}, output {}..{})",
                            window_idx,
                            phased_target.n_markers(),
                            ref_window.global_start,
                            ref_window.global_end,
                            ref_window.output_start,
                            ref_window.output_end
                        );
                    }

                    let n_alleles_per_marker: Vec<usize> = (0..ref_window.markers.len())
                        .map(|m| {
                            let marker = ref_window.markers.marker(MarkerIdx::new(m as u32));
                            1 + marker.alt_alleles.len()
                        })
                        .collect();
                    let mut window_quality = ImputationQuality::new(&n_alleles_per_marker);
                    let resolved_ref_targets = build_ref_typed_marker_resolutions(
                        target_window.genotypes.markers(),
                        &ref_window.markers,
                        &alignment,
                    );
                    let mut dbg_pos_present = 0usize;
                    let mut dbg_aligned_present = 0usize;
                    let mut dbg_pos_not_aligned = 0usize;
                    let mut dbg_pos_not_aligned_sites: Vec<(String, u64)> = Vec::new();
                    for (ref_m, resolved) in resolved_ref_targets.iter().copied().enumerate() {
                        let ref_marker = ref_window.markers.marker(MarkerIdx::new(ref_m as u32));
                        let ref_chrom = ref_window
                            .markers
                            .chrom_name(ref_marker.chrom)
                            .unwrap_or("");
                        let is_present = resolved.is_some();
                        window_quality.set_imputed(ref_m, !is_present);
                        if is_present {
                            dbg_pos_present += 1;
                            if matches!(
                                resolved.map(|r| r.map_kind),
                                Some(TypedMarkerMapKind::Alignment)
                            ) {
                                dbg_aligned_present += 1;
                            } else {
                                dbg_pos_not_aligned += 1;
                                if dbg_pos_not_aligned_sites.len() < 5 {
                                    dbg_pos_not_aligned_sites
                                        .push((ref_chrom.to_string(), ref_marker.pos as u64));
                                }
                            }
                        }
                    }
                    if dbg_pos_not_aligned > 0 && should_log {
                        eprintln!(
                            "    [alignment] genotyped-by-position={} aligned={} pos_not_aligned={}",
                            dbg_pos_present, dbg_aligned_present, dbg_pos_not_aligned
                        );
                        if !dbg_pos_not_aligned_sites.is_empty() {
                            let sites: Vec<String> = dbg_pos_not_aligned_sites
                                .iter()
                                .map(|(c, p)| format!("{}:{}", c, p))
                                .collect();
                            eprintln!(
                                "    [alignment] pos_not_aligned_sites(first{})={}",
                                sites.len(),
                                sites.join(",")
                            );
                        }
                    }

                    let window_results = self.run_imputation_window_streaming(
                        &phased_target,
                        phased_target_pl.as_ref(),
                        target_missing,
                        &ref_window.markers,
                        &ref_window.ref_columns,
                        &alignment,
                        &gen_maps,
                        imp_overlap.as_ref(),
                        &plan,
                        window_idx,
                        ref_window.global_start,
                        ref_window.output_start,
                        ref_window.output_end,
                        true,
                        &mut sample_error_rates,
                    )?;

                    let mut next_handoff = None;
                    let mut next_overlap_opt: Option<PhasedOverlap> = None;
                    if let Some(window_results) = window_results {
                        let ImputationWindowResults {
                            all_results,
                            ref_is_biallelic,
                            overlap_start_idx,
                            handoff,
                            alt_prob_store,
                        } = window_results;
                        next_handoff = handoff;
                        next_overlap_opt = Some(self.extract_imputed_overlap_streaming(
                            &ref_window.markers,
                            &phased_target,
                            &alignment,
                            ref_window.output_start,
                            ref_window.output_end,
                            overlap_start_idx,
                            &all_results,
                            alt_prob_store.as_ref(),
                        ));
                        // Drop heavy reference data before writing to reduce peak RSS.
                        // Drop reference genotypes/columns to free large buffers before write.
                        std::mem::take(&mut ref_window.ref_columns);
                        ref_window.ref_genotypes = None;
                        if let Some(bb) = &self.telemetry {
                            let output_markers = ref_window
                                .output_end
                                .saturating_sub(ref_window.output_start);
                            bb.set_stage(crate::utils::telemetry::Stage::WritingOutput);
                            bb.set_consumer_stage(crate::utils::telemetry::Stage::WritingOutput);
                            bb.set_total_markers(output_markers as u64);
                            bb.set_markers_processed(0);
                            bb.set_total_samples(phased_target.n_samples() as u64);
                            bb.set_samples_processed(0);
                            bb.set_op(&format!(
                                "Writing window {} ({} markers)",
                                window_idx, output_markers
                            ));
                            bb.set_consumer_op(&format!(
                                "Writing window {} ({} markers)",
                                window_idx, output_markers
                            ));
                        }

                        self.write_imputed_window_streaming(
                            &ref_window.markers,
                            &phased_target,
                            phased_target_pl.as_ref(),
                            target_missing,
                            &alignment,
                            &mut writer,
                            &mut window_quality,
                            &ref_is_biallelic,
                            ref_window.output_start,
                            ref_window.output_end,
                            ref_window.output_start,
                            &all_results,
                            alt_prob_store.as_ref(),
                            self.config.gp,
                            self.config.ap,
                            self.config.err.is_some(),
                        )?;

                        if let Some(bb) = &self.telemetry {
                            let output_markers = ref_window
                                .output_end
                                .saturating_sub(ref_window.output_start);
                            bb.set_markers_processed(output_markers as u64);
                            bb.set_samples_processed(phased_target.n_samples() as u64);
                            bb.set_stage(crate::utils::telemetry::Stage::Imputation);
                            bb.set_consumer_stage(crate::utils::telemetry::Stage::Imputation);
                        }
                    }

                    total_markers += ref_window
                        .output_end
                        .saturating_sub(ref_window.output_start);

                    let mut next_overlap = next_overlap_opt.unwrap_or_else(|| {
                        self.extract_imputed_overlap_streaming(
                            &ref_window.markers,
                            &phased_target,
                            &alignment,
                            ref_window.output_start,
                            ref_window.output_end,
                            ref_window.output_start,
                            &[],
                            None,
                        )
                    });
                    if let Some(handoff) = next_handoff {
                        // Imputation handoff consumes orientation-conditioned hap priors.
                        // Clear legacy state_probs to prevent mixed-frame overlap payloads.
                        next_overlap.state_probs = None;
                        next_overlap.set_orientation_hap_priors(handoff.priors_id);
                        next_overlap.set_orientation_weights(handoff.orientation_weight_swap);
                        if let Some(idx) = handoff.prior_global_idx {
                            next_overlap.set_prior_stage1_global_marker(GlobalMarkerIdx::new(idx));
                        }
                        if let Some(gen_pos) = handoff.prior_gen_pos {
                            next_overlap.set_prior_stage1_gen_pos(gen_pos);
                        }
                    } else if !warned_no_overlap {
                        warn!(
                            "No overlap handoff for window {} (empty/short window or region boundary)",
                            window_idx
                        );
                        warned_no_overlap = true;
                    }
                    imp_overlap = Some(next_overlap);

                    window_idx += 1;
                }
            }
            ReferenceData::OnDisk { .. } => {
                let mut ref_reader = open_ref_reader(&ref_path)?;
                loop {
                    if let Some(bb) = &self.telemetry {
                        bb.set_producer_stage(crate::utils::telemetry::Stage::LoadingData);
                        bb.set_producer_op(&format!("Loading ref window {}", window_idx + 1));
                        bb.set_op(&format!("Loading ref window {}", window_idx + 1));
                        bb.set_current_window((window_idx + 1) as u64);
                    }
                    let ref_window = ref_reader.next_window(
                        &streaming_config,
                        &gen_maps,
                        Some(&target_positions_map),
                    )?;
                    let Some(mut ref_window) = ref_window else {
                        break;
                    };

                    let n_ref_markers = ref_window.markers.len();
                    if n_ref_markers == 0 {
                        window_idx += 1;
                        continue;
                    }

                    let ref_chrom_idx = ref_window.markers.marker(MarkerIdx::new(0)).chrom;
                    let ref_chrom = ref_window
                        .markers
                        .chrom_name(ref_chrom_idx)
                        .unwrap_or("UNKNOWN");
                    let start_pos = ref_window.markers.marker(MarkerIdx::new(0)).pos;
                    let end_pos = ref_window
                        .markers
                        .marker(MarkerIdx::new((n_ref_markers - 1) as u32))
                        .pos;

                    let chrom_candidates = chrom_variants(ref_chrom);
                    let target_window = target_reader.load_window_for_region(
                        &chrom_candidates,
                        start_pos,
                        end_pos,
                    )?;
                    let Some(target_window) = target_window else {
                        window_idx += 1;
                        continue;
                    };
                    let target_window_source = if use_phased_for_impute {
                        let reader_pl = target_reader_pl.as_mut().ok_or_else(|| {
                            ReagleError::vcf(
                                "Internal error: missing original target reader for imputation"
                                    .to_string(),
                            )
                        })?;
                        match reader_pl.load_window_for_region(
                            &chrom_candidates,
                            start_pos,
                            end_pos,
                        )? {
                            Some(window) => window,
                            None => {
                                return Err(ReagleError::vcf(format!(
                                    "Original target window missing while phased window exists: chrom={} start={} end={}",
                                    ref_chrom, start_pos, end_pos
                                )));
                            }
                        }
                    } else {
                        target_window.clone()
                    };

                    // Align using the phased target markers; PL/GL lookups share the same
                    // marker set, so indices should remain consistent.
                    let alignment = MarkerAlignment::new_with_ref_markers(
                        &target_window.genotypes,
                        &ref_window.markers,
                    );

                    let phased_target = target_window.genotypes.clone().into_phased();
                    let phased_target_pl =
                        Some(target_window_source.genotypes.clone().into_phased());
                    let target_missing = Some(&target_window_source.genotypes);
                    if !header_written {
                        writer.write_header_extended(
                            &ref_window.markers,
                            true,
                            self.config.gp,
                            self.config.ap,
                        )?;
                        header_written = true;
                    }

                    let should_log = should_log_impute_window(window_idx);
                    if should_log {
                        eprintln!(
                            "  Imputing Window {} ({} markers, ref global {}..{}, output {}..{})",
                            window_idx,
                            phased_target.n_markers(),
                            ref_window.global_start,
                            ref_window.global_end,
                            ref_window.output_start,
                            ref_window.output_end
                        );
                    }

                    let n_alleles_per_marker: Vec<usize> = (0..ref_window.markers.len())
                        .map(|m| {
                            let marker = ref_window.markers.marker(MarkerIdx::new(m as u32));
                            1 + marker.alt_alleles.len()
                        })
                        .collect();
                    let mut window_quality = ImputationQuality::new(&n_alleles_per_marker);
                    let resolved_ref_targets = build_ref_typed_marker_resolutions(
                        target_window.genotypes.markers(),
                        &ref_window.markers,
                        &alignment,
                    );
                    for (ref_m, resolved) in resolved_ref_targets.iter().enumerate() {
                        let is_present = resolved.is_some();
                        window_quality.set_imputed(ref_m, !is_present);
                    }

                    let window_results = self.run_imputation_window_streaming(
                        &phased_target,
                        phased_target_pl.as_ref(),
                        target_missing,
                        &ref_window.markers,
                        &ref_window.ref_columns,
                        &alignment,
                        &gen_maps,
                        imp_overlap.as_ref(),
                        &plan,
                        window_idx,
                        ref_window.global_start,
                        ref_window.output_start,
                        ref_window.output_end,
                        true,
                        &mut sample_error_rates,
                    )?;

                    let mut next_handoff = None;
                    let mut next_overlap_opt: Option<PhasedOverlap> = None;
                    if let Some(window_results) = window_results {
                        let ImputationWindowResults {
                            all_results,
                            ref_is_biallelic,
                            overlap_start_idx,
                            handoff,
                            alt_prob_store,
                        } = window_results;
                        next_handoff = handoff;
                        next_overlap_opt = Some(self.extract_imputed_overlap_streaming(
                            &ref_window.markers,
                            &phased_target,
                            &alignment,
                            ref_window.output_start,
                            ref_window.output_end,
                            overlap_start_idx,
                            &all_results,
                            alt_prob_store.as_ref(),
                        ));
                        // Drop heavy reference data before writing to reduce peak RSS.
                        // Drop reference genotypes/columns to free large buffers before write.
                        std::mem::take(&mut ref_window.ref_columns);
                        ref_window.ref_genotypes = None;
                        if let Some(bb) = &self.telemetry {
                            let output_markers = ref_window
                                .output_end
                                .saturating_sub(ref_window.output_start);
                            bb.set_stage(crate::utils::telemetry::Stage::WritingOutput);
                            bb.set_consumer_stage(crate::utils::telemetry::Stage::WritingOutput);
                            bb.set_total_markers(output_markers as u64);
                            bb.set_markers_processed(0);
                            bb.set_total_samples(phased_target.n_samples() as u64);
                            bb.set_samples_processed(0);
                            bb.set_op(&format!(
                                "Writing window {} ({} markers)",
                                window_idx, output_markers
                            ));
                            bb.set_consumer_op(&format!(
                                "Writing window {} ({} markers)",
                                window_idx, output_markers
                            ));
                        }

                        self.write_imputed_window_streaming(
                            &ref_window.markers,
                            &phased_target,
                            phased_target_pl.as_ref(),
                            target_missing,
                            &alignment,
                            &mut writer,
                            &mut window_quality,
                            &ref_is_biallelic,
                            ref_window.output_start,
                            ref_window.output_end,
                            ref_window.output_start,
                            &all_results,
                            alt_prob_store.as_ref(),
                            self.config.gp,
                            self.config.ap,
                            self.config.err.is_some(),
                        )?;

                        if let Some(bb) = &self.telemetry {
                            let output_markers = ref_window
                                .output_end
                                .saturating_sub(ref_window.output_start);
                            bb.set_markers_processed(output_markers as u64);
                            bb.set_samples_processed(phased_target.n_samples() as u64);
                            bb.set_stage(crate::utils::telemetry::Stage::Imputation);
                            bb.set_consumer_stage(crate::utils::telemetry::Stage::Imputation);
                        }
                    }

                    total_markers += ref_window
                        .output_end
                        .saturating_sub(ref_window.output_start);

                    let mut next_overlap = next_overlap_opt.unwrap_or_else(|| {
                        self.extract_imputed_overlap_streaming(
                            &ref_window.markers,
                            &phased_target,
                            &alignment,
                            ref_window.output_start,
                            ref_window.output_end,
                            ref_window.output_start,
                            &[],
                            None,
                        )
                    });
                    if let Some(handoff) = next_handoff {
                        // Imputation handoff consumes orientation-conditioned hap priors.
                        // Clear legacy state_probs to prevent mixed-frame overlap payloads.
                        next_overlap.state_probs = None;
                        next_overlap.set_orientation_hap_priors(handoff.priors_id);
                        next_overlap.set_orientation_weights(handoff.orientation_weight_swap);
                        if let Some(idx) = handoff.prior_global_idx {
                            next_overlap.set_prior_stage1_global_marker(GlobalMarkerIdx::new(idx));
                        }
                        if let Some(gen_pos) = handoff.prior_gen_pos {
                            next_overlap.set_prior_stage1_gen_pos(gen_pos);
                        }
                    } else if !warned_no_overlap {
                        warn!(
                            "No overlap handoff for window {} (empty/short window or region boundary)",
                            window_idx
                        );
                        warned_no_overlap = true;
                    }
                    imp_overlap = Some(next_overlap);

                    window_idx += 1;
                }
            }
        }

        if let Some(tmpdir) = input_tmp.as_ref() {
            tracing::trace!(path = ?tmpdir.path(), "Using temp dir for stdin target");
        }
        if let Some(tmpdir) = phased_tmp.as_ref() {
            tracing::trace!(path = ?tmpdir.path(), "Using temp dir for phased target");
        }

        if total_markers == 0 {
            return Err(ReagleError::vcf(
                "No markers imputed; check reference/target overlap and region selection.",
            ));
        }

        eprintln!("Streaming imputation complete: {} markers", total_markers);
        Ok(())
    }
    fn run_imputation_window_streaming<
        TargetSpace: Sync,
        RefMarkerSpace: Sync,
        TargetMissingState: PhaseState + Sync,
    >(
        &self,
        target_win: &GenotypeMatrix<Phased, TargetSpace>,
        target_pl: Option<&GenotypeMatrix<Phased, TargetSpace>>,
        target_missing: Option<&GenotypeMatrix<TargetMissingState, TargetSpace>>,
        ref_markers: &crate::data::marker::Markers<RefMarkerSpace>,
        ref_columns: &[GenotypeColumn],
        alignment: &MarkerAlignment<TargetSpace, RefMarkerSpace>,
        gen_maps: &GeneticMaps,
        imp_overlap: Option<&PhasedOverlap>,
        plan: &ImputationPlan,
        window_idx: usize,
        global_start: usize,
        output_start: usize,
        output_end: usize,
        phase_conf_valid: bool,
        sample_error_rates: &mut [f32],
    ) -> Result<Option<ImputationWindowResults>> {
        let window_span = if self.config.profile {
            Some(
                info_span!(
                    "imputation_window_compute",
                    ref_markers = ref_markers.len(),
                    target_markers = target_win.n_markers(),
                    output_start,
                    output_end
                )
                .entered(),
            )
        } else {
            None
        };
        if let Some(span) = &window_span {
            tracing::trace!(id = ?span.id(), "Entered imputation window span");
        }

        let n_ref_markers = ref_markers.len();
        let n_target_samples = target_win.n_samples();
        // Transition panel size for imputation is the reference panel only.
        let n_transition_haps = plan.n_ref_haps.max(1);
        let n_transition_haps_f32 = n_transition_haps as f32;
        let should_log = should_log_impute_window(window_idx);
        let use_abyss = true;
        let output_markers = output_end.saturating_sub(output_start);

        if output_start >= output_end || n_ref_markers == 0 {
            return Ok(None);
        }
        if let Some(bb) = &self.telemetry {
            bb.set_stage(crate::utils::telemetry::Stage::Imputation);
            bb.set_consumer_stage(crate::utils::telemetry::Stage::Imputation);
            bb.set_producer_stage(crate::utils::telemetry::Stage::Imputation);
            bb.set_total_windows(plan.per_window_caps.len() as u64);
            bb.set_current_window((window_idx + 1) as u64);
            bb.set_total_markers(output_markers as u64);
            bb.set_markers_processed(0);
            bb.set_total_samples(n_target_samples as u64);
            bb.set_samples_processed(0);
            bb.set_op(&format!(
                "Imputing window {} ({} markers)",
                window_idx + 1,
                output_markers
            ));
            bb.set_producer_op(&format!(
                "Imputing window {} ({} markers)",
                window_idx + 1,
                output_markers
            ));
            bb.set_consumer_op(&format!(
                "Imputing window {} ({} markers)",
                window_idx + 1,
                output_markers
            ));
        }

        let ref_is_biallelic: Vec<bool> = (0..n_ref_markers)
            .map(|m| {
                ref_markers
                    .marker(MarkerIdx::new(m as u32))
                    .alt_alleles
                    .len()
                    == 1
            })
            .collect();
        if should_log {
            let n_multiallelic = ref_is_biallelic.iter().filter(|&&is_bi| !is_bi).count();
            let dense_post_per_sample_bytes = 2u64
                .saturating_mul(output_markers as u64)
                .saturating_mul(std::mem::size_of::<AllelePosteriors>() as u64);
            let dense_alt_per_sample_bytes = 2u64
                .saturating_mul(output_markers as u64)
                .saturating_mul(std::mem::size_of::<f32>() as u64);
            eprintln!(
                "    [window diag] output_markers={} biallelic={} multiallelic={} dense_post_per_sample_mb={} compact_alt_per_sample_mb={}",
                output_markers,
                output_markers.saturating_sub(n_multiallelic),
                n_multiallelic,
                dense_post_per_sample_bytes / (1024 * 1024),
                dense_alt_per_sample_bytes / (1024 * 1024)
            );
        }
        let ref_allele_freqs = RefAlleleFreqs::new(ref_columns);

        let gen_positions: Vec<f64> = {
            let chrom_idx = ref_markers.marker(MarkerIdx::new(0)).chrom;
            let chrom_name = ref_markers.chrom_name(chrom_idx).unwrap_or("");
            if let Some(gen_map) = gen_maps.get_by_name(chrom_name) {
                crate::data::genetic_map::MarkerMap::create(ref_markers, gen_map)
                    .gen_positions()
                    .to_vec()
            } else {
                crate::data::genetic_map::MarkerMap::from_positions(ref_markers)
                    .gen_positions()
                    .to_vec()
            }
        };
        if should_log {
            if let (Some(first), Some(last)) = (gen_positions.first(), gen_positions.last()) {
                let total_cm = (last - first).abs();
                eprintln!(
                    "    genetic span: {:.6} cM across {} markers",
                    total_cm,
                    gen_positions.len()
                );
            }
        }
        // Li-Stephens recombination probabilities between adjacent reference
        // markers. These are biological copy-path transition priors (donor
        // template switches), parameterized by genetic-map distance.
        let mut p_recomb: Vec<f32> = Vec::with_capacity(n_ref_markers);
        p_recomb.push(0.0f32);
        for m in 1..n_ref_markers {
            let dist_cm = (gen_positions[m] - gen_positions[m - 1]).abs();
            p_recomb.push(self.params.p_recomb(dist_cm));
        }

        if should_log {
            if let Some(min) = p_recomb.iter().copied().reduce(f32::min) {
                let max = p_recomb.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mean = p_recomb.iter().copied().sum::<f32>() / p_recomb.len().max(1) as f32;
                eprintln!(
                    "    p_recomb stats: min={:.6} mean={:.6} max={:.6}",
                    min, mean, max
                );
            }
        }

        if let Some(overlap) = imp_overlap {
            if let Some(prev_gen_pos) = overlap.prior_stage1_gen_pos() {
                if let Some(current_gen_pos) = gen_positions.first().copied() {
                    let dist_cm = (current_gen_pos - prev_gen_pos).abs();
                    if dist_cm > 0.0 && !dist_cm.is_nan() {
                        p_recomb[0] = self.params.p_recomb(dist_cm);
                    }
                }
            }
        }
        let handoff_recomb_rate = p_recomb.get(0).copied().unwrap_or(0.0).clamp(0.0, 1.0);

        // Only consume overlap priors when their anchor marker is physically
        // compatible with the current window. This prevents seam drift from
        // stale/misaligned priors being projected into the wrong window.
        let overlap_priors_usable = if let Some(overlap) = imp_overlap {
            if let Some(prior_marker) = overlap.prior_stage1_global_marker() {
                let prior_marker = prior_marker.as_usize();
                let window_start = global_start;
                let window_end = global_start.saturating_add(n_ref_markers);
                let valid_range = prior_marker >= window_start && prior_marker < window_end;
                let adjacent_prev = prior_marker.saturating_add(1) == window_start;
                if !(valid_range || adjacent_prev) {
                    warn!(
                        "Skipping overlap priors: marker anchor {} incompatible with window [{}..{})",
                        prior_marker, window_start, window_end
                    );
                }
                valid_range || adjacent_prev
            } else {
                false
            }
        } else {
            false
        };

        let per_window_cap_local = plan
            .per_window_caps
            .get(window_idx)
            .copied()
            .unwrap_or(plan.per_window_cap)
            .max(1);
        let (plan_range_start, plan_range_end) = plan
            .io_to_planning_ranges
            .get(window_idx)
            .copied()
            .unwrap_or((window_idx, window_idx.saturating_add(1)));
        // Even when full-panel memory is available, keep sample/window-specific
        // state sets from prescan/LMS. This preserves ancestry-local donor sets
        // and avoids diluting sparse-target inference with globally irrelevant
        // haplotypes.
        let full_states: Option<Vec<RefHapId>> = if plan.full_panel || n_target_samples <= 2 {
            Some(
                (0..plan.n_ref_haps)
                    .map(|h| RefHapId::new(h as u32))
                    .collect(),
            )
        } else {
            None
        };

        let mut state_haps_by_hap: Vec<Vec<RefHapId>> = Vec::with_capacity(n_target_samples * 2);
        if full_states.is_none() {
            for hap_idx in 0..(n_target_samples * 2) {
                let mut state_haps: Vec<RefHapId> = Vec::new();
                if hap_idx < plan.window_intervals.len() {
                    let mut ranked: Vec<(RefHapId, u32)> = Vec::new();
                    for hi in plan.window_intervals[hap_idx].iter() {
                        if let Some(span) =
                            interval_support_over_range(hi, plan_range_start, plan_range_end)
                        {
                            ranked.push((hi.hap, span));
                        }
                    }
                    ranked.sort_unstable_by(|a, b| {
                        b.1.cmp(&a.1).then_with(|| a.0.as_u32().cmp(&b.0.as_u32()))
                    });
                    for (hap, _) in ranked {
                        state_haps.push(hap);
                    }
                }
                if state_haps.is_empty() && hap_idx < plan.core_states.len() {
                    state_haps.extend(plan.core_states[hap_idx].iter().copied());
                }
                state_haps.sort_unstable_by_key(|g| g.as_u32());
                state_haps.dedup();
                if use_abyss && hap_idx < plan.abyss_mask.len() {
                    let abyss = &plan.abyss_mask[hap_idx];
                    state_haps.retain(|g| !abyss[g.as_usize()]);
                }
                if state_haps.len() > per_window_cap_local {
                    let core_for_hap = plan
                        .core_states
                        .get(hap_idx)
                        .map(|v| v.as_slice())
                        .unwrap_or(&[]);
                    let core_set: std::collections::HashSet<u32> =
                        core_for_hap.iter().map(|h| h.as_u32()).collect();
                    let mut prioritized: Vec<RefHapId> = state_haps
                        .iter()
                        .copied()
                        .filter(|h| core_set.contains(&h.as_u32()))
                        .collect();
                    if prioritized.len() < per_window_cap_local {
                        for hap in state_haps.iter().copied() {
                            if !core_set.contains(&hap.as_u32()) {
                                prioritized.push(hap);
                                if prioritized.len() >= per_window_cap_local {
                                    break;
                                }
                            }
                        }
                    } else {
                        prioritized.truncate(per_window_cap_local);
                    }
                    state_haps = prioritized;
                }
                // Keep empty here if no scored candidates survived; per-sample state
                // assembly later combines priors/donors/core and applies final checks.
                state_haps_by_hap.push(state_haps);
            }
        }

        let overlap_hap_priors_id = if overlap_priors_usable {
            imp_overlap.and_then(|o| o.hap_priors_id())
        } else {
            None
        };
        let overlap_orientation_weight_swap = if overlap_priors_usable {
            imp_overlap.and_then(|o| o.orientation_swap_weights())
        } else {
            None
        };

        let normalize_probs = |probs: &mut [f32]| -> bool {
            let mut sum = 0.0f32;
            for p in probs.iter_mut() {
                if *p < 0.0 {
                    *p = 0.0;
                }
                sum += *p;
            }
            if sum > 0.0 {
                for p in probs.iter_mut() {
                    *p /= sum;
                }
                true
            } else {
                false
            }
        };

        let target_samples = target_win.samples_arc();
        let target_pl_matrix = target_pl.unwrap_or(target_win);
        let err_floor = 0.0001f32;
        let err_rate = self.params.p_mismatch.max(err_floor).clamp(1e-6, 0.5);
        let smoothing_cluster_cm = self.config.cluster.max(1e-6);
        let overlap_start = overlap_start_from_hazard(output_start, output_end, &p_recomb);
        let resolved_ref_targets =
            build_ref_typed_marker_resolutions(target_win.markers(), ref_markers, alignment);
        if should_log {
            let positional_fallback = resolved_ref_targets
                .iter()
                .filter_map(|v| *v)
                .filter(|r| r.is_positional_fallback())
                .count();
            if positional_fallback > 0 {
                eprintln!(
                    "    [alignment fallback] hmm/prescan positional_typed_markers={}",
                    positional_fallback
                );
            }
        }
        let build_input_probs_pair = |hap1: HapIdx,
                                      hap2: HapIdx,
                                      sample_idx: usize|
         -> (
            TargetAlleleProbs,
            TargetAlleleProbs,
            Option<usize>,
            Option<usize>,
        ) {
            let mut offsets1 = Vec::with_capacity(n_ref_markers + 1);
            let mut offsets2 = Vec::with_capacity(n_ref_markers + 1);
            let mut probs1: Vec<f32> = Vec::new();
            let mut probs2: Vec<f32> = Vec::new();
            let mut observed1: Vec<bool> = Vec::with_capacity(n_ref_markers);
            let mut observed2: Vec<bool> = Vec::with_capacity(n_ref_markers);
            let mut marker_errors1: Vec<f32> = Vec::with_capacity(n_ref_markers);
            let mut marker_errors2: Vec<f32> = Vec::with_capacity(n_ref_markers);
            // Reuse scratch buffers across markers to avoid allocator churn in the hot loop.
            let mut aligned1: Vec<f32> = Vec::new();
            let mut aligned2: Vec<f32> = Vec::new();
            let mut pl_probs_buf: Vec<f32> = Vec::new();
            let mut cond_probs_buf: Vec<f32> = Vec::new();
            let mut mapped_buf: Vec<f32> = Vec::new();
            let mut weights_buf: Vec<f32> = Vec::new();
            let mut target_priors_buf: Vec<f32> = Vec::new();
            offsets1.push(0);
            offsets2.push(0);
            let mut last_info1: Option<usize> = None;
            let mut last_info2: Option<usize> = None;
            let mut diag_typed_hets = 0usize;
            let mut diag_typed_hets_phase_valid = 0usize;
            let mut diag_phase_conf_sum = 0.0f32;
            let is_uniform = |vals: &[f32]| -> bool {
                if vals.len() <= 1 {
                    return true;
                }
                let mut min = vals[0];
                let mut max = vals[0];
                for &v in vals.iter().skip(1) {
                    if v < min {
                        min = v;
                    }
                    if v > max {
                        max = v;
                    }
                }
                (max - min) <= 1e-6
            };

            for (ref_m, resolved) in resolved_ref_targets.iter().copied().enumerate() {
                let n_alleles = ref_markers.marker(MarkerIdx::new(ref_m as u32)).n_alleles();
                let mut observed1_marker = false;
                let mut observed2_marker = false;
                aligned1.clear();
                aligned2.clear();
                let mut use1 = false;
                let mut use2 = false;

                if let Some(resolution) = resolved {
                    let target_m = resolution.target_idx;
                    let conf_base = target_pl_matrix
                        .sample_confidence_f32(MarkerIdx::new(target_m as u32), sample_idx);
                    let mut conf1 = conf_base;
                    let mut conf2 = conf_base;
                    let raw_allele1 = target_win.allele(MarkerIdx::new(target_m as u32), hap1);
                    let raw_allele2 = target_win.allele(MarkerIdx::new(target_m as u32), hap2);
                    let mut allele1 = raw_allele1;
                    let mut allele2 = raw_allele2;
                    if let Some(missing) = target_missing {
                        if missing.allele(MarkerIdx::new(target_m as u32), hap1)
                            == crate::data::storage::AlleleCode::MISSING.raw()
                        {
                            allele1 = crate::data::storage::AlleleCode::MISSING.raw();
                        }
                        if missing.allele(MarkerIdx::new(target_m as u32), hap2)
                            == crate::data::storage::AlleleCode::MISSING.raw()
                        {
                            allele2 = crate::data::storage::AlleleCode::MISSING.raw();
                        }
                    }

                    let mapped1 = map_target_allele_to_ref(alignment, resolution, allele1)
                        .unwrap_or(crate::data::storage::AlleleCode::MISSING.raw());
                    let mapped2 = map_target_allele_to_ref(alignment, resolution, allele2)
                        .unwrap_or(crate::data::storage::AlleleCode::MISSING.raw());

                    let is_diploid = target_samples.is_diploid(SampleIdx::new(sample_idx as u32));
                    let has_hard = mapped1 != crate::data::storage::AlleleCode::MISSING.raw()
                        && (mapped1 as usize) < n_alleles
                        && (!is_diploid
                            || (mapped2 != crate::data::storage::AlleleCode::MISSING.raw()
                                && (mapped2 as usize) < n_alleles));
                    let input_phased = target_win
                        .phase_mask()
                        .and_then(|mask| mask.get(target_m, sample_idx))
                        .map(|v| v != 0)
                        .unwrap_or(true);
                    let local_phase_conf_valid = phase_conf_valid && input_phased;
                    if is_diploid
                        && has_hard
                        && mapped1 != crate::data::storage::AlleleCode::MISSING.raw()
                        && mapped2 != crate::data::storage::AlleleCode::MISSING.raw()
                        && mapped1 != mapped2
                    {
                        diag_typed_hets += 1;
                        if local_phase_conf_valid {
                            let phase_conf = target_win
                                .sample_phase_confidence_f32(
                                    MarkerIdx::new(target_m as u32),
                                    sample_idx,
                                )
                                .clamp(0.0, 1.0);
                            diag_typed_hets_phase_valid += 1;
                            diag_phase_conf_sum += phase_conf.max(1.0 - phase_conf);
                        }
                    }

                    // If phase confidence is unavailable (unphased input), we still
                    // use hard genotype information but avoid committing to a phase:
                    // heterozygotes are represented as 0.5/0.5 per haplotype.

                    let pl =
                        target_pl_matrix.sample_pl(MarkerIdx::new(target_m as u32), sample_idx);
                    let has_pl = pl.map_or(false, |vals| !vals.is_empty());

                    // If the input is unphased, avoid conditioning on a partner
                    // allele. Use unconditional allele probabilities from PL
                    // (if present) for both haplotypes.
                    if !local_phase_conf_valid {
                        if let Some(pl) = pl {
                            if !pl.is_empty() {
                                pl_probs_buf.clear();
                                if allele_probs_uncond_from_pl(pl, None, &mut pl_probs_buf)
                                    .is_some()
                                {
                                    if map_target_probs_to_ref(
                                        alignment,
                                        resolution,
                                        &pl_probs_buf,
                                        n_alleles,
                                        &mut mapped_buf,
                                    ) {
                                        aligned1.clear();
                                        aligned2.clear();
                                        aligned1.extend_from_slice(&mapped_buf);
                                        aligned2.extend_from_slice(&mapped_buf);
                                        use1 = true;
                                        use2 = true;
                                    }
                                }
                            }
                        }
                    }

                    let mut compute_from_pl =
                        |partner_allele: u8, out: &mut Vec<f32>, used: &mut bool| {
                            pl_probs_buf.clear();
                            if let Some(pl) = pl {
                                if !pl.is_empty() {
                                    let n_pl_alleles =
                                        infer_n_alleles_from_pl_len(pl.len()).unwrap_or(0);
                                    if n_pl_alleles > 0 {
                                        let mut used_conditional = false;
                                        if local_phase_conf_valid
                                            && partner_allele
                                                != crate::data::storage::AlleleCode::MISSING.raw()
                                            && (partner_allele as usize) < n_pl_alleles
                                        {
                                            let phase_conf = target_win
                                                .sample_phase_confidence_f32(
                                                    MarkerIdx::new(target_m as u32),
                                                    sample_idx,
                                                )
                                                .clamp(0.0, 1.0);
                                            weights_buf.clear();
                                            weights_buf.resize(n_pl_alleles, 0.0);
                                            let mut denom = 0.0f32;
                                            for i in 0..n_pl_alleles {
                                                if i != partner_allele as usize {
                                                    denom += 1.0;
                                                }
                                            }
                                            weights_buf[partner_allele as usize] = phase_conf;
                                            if denom > 0.0 {
                                                let scale = (1.0 - phase_conf) / denom;
                                                for i in 0..n_pl_alleles {
                                                    if i != partner_allele as usize {
                                                        weights_buf[i] = scale;
                                                    }
                                                }
                                            }

                                            cond_probs_buf.clear();
                                            pl_probs_buf.resize(n_pl_alleles, 0.0);
                                            let mut weight_sum = 0.0f32;
                                            for b in 0..n_pl_alleles {
                                                let w = weights_buf[b];
                                                if w <= 0.0 {
                                                    continue;
                                                }
                                                if allele_probs_cond_from_pl(
                                                    pl,
                                                    b as u8,
                                                    None,
                                                    &mut cond_probs_buf,
                                                )
                                                .is_some()
                                                {
                                                    for (a, &p) in cond_probs_buf.iter().enumerate()
                                                    {
                                                        if a < pl_probs_buf.len() {
                                                            pl_probs_buf[a] += w * p;
                                                        }
                                                    }
                                                    weight_sum += w;
                                                }
                                            }
                                            if weight_sum > 0.0 {
                                                normalize_probs(&mut pl_probs_buf);
                                                if map_target_probs_to_ref(
                                                    alignment,
                                                    resolution,
                                                    &pl_probs_buf,
                                                    n_alleles,
                                                    &mut mapped_buf,
                                                ) {
                                                    out.clear();
                                                    out.extend_from_slice(&mapped_buf);
                                                    *used = true;
                                                    used_conditional = true;
                                                }
                                            }
                                        }

                                        if !used_conditional
                                            && allele_probs_uncond_from_pl(
                                                pl,
                                                None,
                                                &mut pl_probs_buf,
                                            )
                                            .is_some()
                                        {
                                            if map_target_probs_to_ref(
                                                alignment,
                                                resolution,
                                                &pl_probs_buf,
                                                n_alleles,
                                                &mut mapped_buf,
                                            ) {
                                                out.clear();
                                                out.extend_from_slice(&mapped_buf);
                                                *used = true;
                                            }
                                        } else if !used_conditional {
                                            let uniform = 1.0 / n_pl_alleles as f32;
                                            target_priors_buf.clear();
                                            target_priors_buf.resize(n_pl_alleles, uniform);

                                            let conf = conf_base.clamp(0.0, 1.0);
                                            weights_buf.clear();
                                            weights_buf.resize(n_pl_alleles, 0.0);
                                            if partner_allele
                                                != crate::data::storage::AlleleCode::MISSING.raw()
                                                && (partner_allele as usize) < n_pl_alleles
                                            {
                                                let partner_idx = partner_allele as usize;
                                                let mut denom = 0.0f32;
                                                for (i, &p) in target_priors_buf.iter().enumerate()
                                                {
                                                    if i != partner_idx {
                                                        denom += p;
                                                    }
                                                }
                                                weights_buf[partner_idx] = conf;
                                                if denom > 0.0 {
                                                    let scale = (1.0 - conf) / denom;
                                                    for i in 0..n_pl_alleles {
                                                        if i != partner_idx {
                                                            weights_buf[i] =
                                                                target_priors_buf[i] * scale;
                                                        }
                                                    }
                                                } else if n_pl_alleles > 1 {
                                                    let uniform =
                                                        (1.0 - conf) / (n_pl_alleles as f32 - 1.0);
                                                    for i in 0..n_pl_alleles {
                                                        if i != partner_idx {
                                                            weights_buf[i] = uniform;
                                                        }
                                                    }
                                                }
                                            } else {
                                                weights_buf.copy_from_slice(&target_priors_buf);
                                            }

                                            cond_probs_buf.clear();
                                            pl_probs_buf.resize(n_pl_alleles, 0.0);
                                            let mut weight_sum = 0.0f32;
                                            for b in 0..n_pl_alleles {
                                                let w = weights_buf[b];
                                                if w <= 0.0 {
                                                    continue;
                                                }
                                                if allele_probs_cond_from_pl(
                                                    pl,
                                                    b as u8,
                                                    None,
                                                    &mut cond_probs_buf,
                                                )
                                                .is_some()
                                                {
                                                    for (a, &p) in cond_probs_buf.iter().enumerate()
                                                    {
                                                        if a < pl_probs_buf.len() {
                                                            pl_probs_buf[a] += w * p;
                                                        }
                                                    }
                                                    weight_sum += w;
                                                }
                                            }
                                            if weight_sum > 0.0 {
                                                normalize_probs(&mut pl_probs_buf);
                                                if map_target_probs_to_ref(
                                                    alignment,
                                                    resolution,
                                                    &pl_probs_buf,
                                                    n_alleles,
                                                    &mut mapped_buf,
                                                ) {
                                                    out.clear();
                                                    out.extend_from_slice(&mapped_buf);
                                                    *used = true;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        };

                    if !has_hard {
                        compute_from_pl(allele2, &mut aligned1, &mut use1);
                        compute_from_pl(allele1, &mut aligned2, &mut use2);
                    }

                    if !use1 && has_hard {
                        if pl.is_none() || pl.as_ref().map_or(true, |v| v.is_empty()) {
                            // GT-only input (no PL/GL): cap confidence by the emission error
                            // floor to avoid over-concentrating the HMM on sparse arrays.
                            conf1 = conf1.min(1.0 - err_rate);
                        }
                        aligned1.resize(n_alleles, 0.0);
                        if is_diploid
                            && mapped2 != crate::data::storage::AlleleCode::MISSING.raw()
                            && mapped2 != mapped1
                        {
                            if local_phase_conf_valid {
                                let phase_conf = target_win
                                    .sample_phase_confidence_f32(
                                        MarkerIdx::new(target_m as u32),
                                        sample_idx,
                                    )
                                    .clamp(0.0, 1.0);
                                let c = phase_conf;
                                let g = conf1.clamp(0.5, 1.0);
                                // Preserve orientation direction: c<0.5 means the phased
                                // hap order is likely flipped at this marker.
                                let p_primary = (0.5 + (c - 0.5) * (2.0 * g - 1.0)).clamp(0.0, 1.0);
                                aligned1[mapped1 as usize] = p_primary;
                                aligned1[mapped2 as usize] = 1.0 - p_primary;
                            } else {
                                aligned1[mapped1 as usize] = 0.5;
                                aligned1[mapped2 as usize] = 0.5;
                            }
                        } else {
                            aligned1[mapped1 as usize] = conf1.clamp(0.0, 1.0);
                        }
                        use1 = true;
                    }

                    if !use2 && has_hard {
                        if pl.is_none() || pl.as_ref().map_or(true, |v| v.is_empty()) {
                            // GT-only input (no PL/GL): cap confidence by the emission error
                            // floor to avoid over-concentrating the HMM on sparse arrays.
                            conf2 = conf2.min(1.0 - err_rate);
                        }
                        aligned2.resize(n_alleles, 0.0);
                        if is_diploid
                            && mapped1 != crate::data::storage::AlleleCode::MISSING.raw()
                            && mapped1 != mapped2
                        {
                            if local_phase_conf_valid {
                                let phase_conf = target_win
                                    .sample_phase_confidence_f32(
                                        MarkerIdx::new(target_m as u32),
                                        sample_idx,
                                    )
                                    .clamp(0.0, 1.0);
                                let c = phase_conf;
                                let g = conf2.clamp(0.5, 1.0);
                                let p_primary = (0.5 + (c - 0.5) * (2.0 * g - 1.0)).clamp(0.0, 1.0);
                                aligned2[mapped2 as usize] = p_primary;
                                aligned2[mapped1 as usize] = 1.0 - p_primary;
                            } else {
                                aligned2[mapped2 as usize] = 0.5;
                                aligned2[mapped1 as usize] = 0.5;
                            }
                        } else {
                            aligned2[mapped2 as usize] = conf2.clamp(0.0, 1.0);
                        }
                        use2 = true;
                    }

                    if has_pl {
                        observed1_marker = true;
                        observed2_marker = true;
                    } else if has_hard {
                        observed1_marker =
                            mapped1 != crate::data::storage::AlleleCode::MISSING.raw();
                        observed2_marker = if is_diploid {
                            mapped2 != crate::data::storage::AlleleCode::MISSING.raw()
                        } else {
                            mapped1 != crate::data::storage::AlleleCode::MISSING.raw()
                        };
                    }
                }

                if !use1 || !use2 {
                    // Untyped target marker: no direct evidence at this marker.
                    // Keep emissions neutral and let state-path evidence drive posteriors.
                    if !use1 {
                        aligned1.resize(n_alleles.max(1), 1.0);
                    }
                    if !use2 {
                        aligned2.resize(n_alleles.max(1), 1.0);
                    }
                }

                normalize_probs(&mut aligned1);
                normalize_probs(&mut aligned2);
                if observed1_marker && !is_uniform(&aligned1) && ref_m >= overlap_start {
                    last_info1 = Some(ref_m);
                }
                if observed2_marker && !is_uniform(&aligned2) && ref_m >= overlap_start {
                    last_info2 = Some(ref_m);
                }

                probs1.extend_from_slice(&aligned1);
                probs2.extend_from_slice(&aligned2);
                observed1.push(observed1_marker);
                observed2.push(observed2_marker);
                marker_errors1.push(marker_emission_error_from_probs(
                    &aligned1,
                    observed1_marker,
                    err_rate,
                ));
                marker_errors2.push(marker_emission_error_from_probs(
                    &aligned2,
                    observed2_marker,
                    err_rate,
                ));
                offsets1.push(probs1.len());
                offsets2.push(probs2.len());
            }
            let observed_ratio1 = if observed1.is_empty() {
                0.0
            } else {
                observed1.iter().filter(|&&v| v).count() as f32 / observed1.len() as f32
            };
            let observed_ratio2 = if observed2.is_empty() {
                0.0
            } else {
                observed2.iter().filter(|&&v| v).count() as f32 / observed2.len() as f32
            };
            let min_untyped_prior_mix1 = adaptive_untyped_prior_mix(
                observed_ratio1,
                smoothing_cluster_cm,
                self.params.p_mismatch,
                !phase_conf_valid,
            );
            let min_untyped_prior_mix2 = adaptive_untyped_prior_mix(
                observed_ratio2,
                smoothing_cluster_cm,
                self.params.p_mismatch,
                !phase_conf_valid,
            );
            if should_log && sample_idx == 0 {
                let mean_conf = if diag_typed_hets_phase_valid > 0 {
                    diag_phase_conf_sum / diag_typed_hets_phase_valid as f32
                } else {
                    0.0
                };
                eprintln!(
                    "    [diag phase-conf] typed_hets={} phase_valid={} mean_orientation_conf={:.3}",
                    diag_typed_hets, diag_typed_hets_phase_valid, mean_conf
                );
            }
            let mut input1 =
                TargetAlleleProbs::new(offsets1, probs1, observed1, None, min_untyped_prior_mix1);
            input1.set_marker_error_rates(marker_errors1);
            let mut input2 =
                TargetAlleleProbs::new(offsets2, probs2, observed2, None, min_untyped_prior_mix2);
            input2.set_marker_error_rates(marker_errors2);
            (input1, input2, last_info1, last_info2)
        };

        let n_target_haps = n_target_samples * 2;
        let min_info_nats = (plan.n_ref_haps as f32).ln() * 1.5;
        // Information-weighted confusion: normalized entropy of the PBWT match set
        // scaled by the emission-model LLR to keep confidence in probabilistic units.
        let mut sm_low_conf_weighted: Vec<f32> = vec![0.0; n_target_haps];
        let mut sm_total_info: Vec<f32> = vec![0.0; n_target_haps];
        let mut sm_donor_counts: Vec<HashMap<RefHapId, f32>> = vec![HashMap::new(); n_target_haps];
        let sm_needed: Vec<AtomicBool> =
            (0..n_target_haps).map(|_| AtomicBool::new(false)).collect();

        {
            let mut pbwt = ReferencePbwt::new(plan.n_ref_haps);
            let mut ref_alleles: Vec<u8> = vec![0u8; plan.n_ref_haps];
            let phase_mask = target_win.phase_mask();
            let batch_size = 4096usize;
            let mut batches: Vec<(
                Vec<usize>,
                Vec<RankBeam>,
                Vec<PbwtQueryAllele>,
                Vec<f32>,
                Vec<u32>,
                Vec<Option<usize>>,
                Vec<(u32, u32, u64)>,
            )> = Vec::new();
            let mut start = 0usize;
            while start < n_target_haps {
                let end = (start + batch_size).min(n_target_haps);
                let haps: Vec<usize> = (start..end).collect();
                let beams = vec![RankBeam::full(plan.n_ref_haps as u32); haps.len()];
                let query_alleles = vec![PbwtQueryAllele::wildcard(); haps.len()];
                let query_info_weight = vec![0.0f32; haps.len()];
                let current_donor = vec![0u32; haps.len()];
                let peer_idx_by_hap = build_peer_indices(&haps);
                let scratch = Vec::new();
                batches.push((
                    haps,
                    beams,
                    query_alleles,
                    query_info_weight,
                    current_donor,
                    peer_idx_by_hap,
                    scratch,
                ));
                start = end;
            }

            let push_donor_weight =
                |counts: &mut HashMap<RefHapId, f32>, hap: RefHapId, delta: f32| {
                    if !delta.is_finite() || delta <= 0.0 {
                        return;
                    }
                    let entry = counts.entry(hap).or_insert(0.0);
                    *entry += delta;
                };
            // Information weight in natural log space for one informative allele observation.
            let theta = self.params.p_mismatch.max(1e-9).min(1.0 - 1e-9) as f32;
            let info_llr = ((1.0 - theta) / theta).ln();
            let mut prefix_log_stay_full = vec![0.0f32; n_ref_markers.saturating_add(1)];
            if n_ref_markers > 0 {
                for m in 0..n_ref_markers {
                    let r = p_recomb.get(m).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                    // Match Li-Stephens convention used by subset transition math:
                    // full-panel stay term is (1-r), with switch mass handled separately.
                    // Let s_t = 1-r_t. For donor tract start a and marker b, continuity is:
                    //   cont(a->b) = product_{t=a+1..b} s_t
                    // Taking logs gives a fast O(1) span query via prefix sums:
                    //   log cont(a->b) = sum_{t=a+1..b} log s_t
                    let stay = (1.0 - r).clamp(1e-12, 1.0);
                    prefix_log_stay_full[m + 1] = prefix_log_stay_full[m] + stay.ln();
                }
            }

            for ref_m in 0..n_ref_markers {
                let col = &ref_columns[ref_m];
                fill_ref_alleles(col, &mut ref_alleles);
                let n_alleles = ref_markers
                    .marker(MarkerIdx::new(ref_m as u32))
                    .n_alleles()
                    .max(1);
                let target_m_opt = resolved_ref_targets.get(ref_m).copied().flatten();

                pbwt.prepare_step(&ref_alleles, n_alleles);
                for (haps, beams, query_alleles, query_info_weight, _, peer_idx_by_hap, scratch) in
                    batches.iter_mut()
                {
                    if let Some(resolution) = target_m_opt {
                        let target_idx = resolution.target_idx;
                        let target_marker = MarkerIdx::new(target_idx as u32);

                        let mut cached_sample_idx = usize::MAX;
                        let mut cached_query_pair =
                            [crate::data::storage::AlleleCode::MISSING.raw(); 2];
                        let mut cached_wildcard_weight = 0.0f32;
                        let mut cached_allele_weight = 0.0f32;
                        for (i, &hap_idx) in haps.iter().enumerate() {
                            let sample_idx = hap_idx / 2;
                            let local = hap_idx % 2;
                            if sample_idx != cached_sample_idx {
                                cached_sample_idx = sample_idx;
                                cached_wildcard_weight = 0.0;
                                cached_allele_weight = 0.0;
                                let sample = sample_idx_from_usize(sample_idx);
                                let h1 = sample.hap(HapSide::H1);
                                let h2 = sample.hap(HapSide::H2);
                                let mut a1 = target_win.allele(target_marker, h1);
                                let mut a2 = target_win.allele(target_marker, h2);
                                if let Some(missing) = target_missing {
                                    if missing.allele(target_marker, h1)
                                        == crate::data::storage::AlleleCode::MISSING.raw()
                                    {
                                        a1 = crate::data::storage::AlleleCode::MISSING.raw();
                                    }
                                    if missing.allele(target_marker, h2)
                                        == crate::data::storage::AlleleCode::MISSING.raw()
                                    {
                                        a2 = crate::data::storage::AlleleCode::MISSING.raw();
                                    }
                                }
                                let mapped1 = map_target_allele_to_ref(alignment, resolution, a1)
                                    .unwrap_or(crate::data::storage::AlleleCode::MISSING.raw());
                                let mapped2 = map_target_allele_to_ref(alignment, resolution, a2)
                                    .unwrap_or(crate::data::storage::AlleleCode::MISSING.raw());
                                let is_het = mapped1
                                    != crate::data::storage::AlleleCode::MISSING.raw()
                                    && mapped2 != crate::data::storage::AlleleCode::MISSING.raw()
                                    && mapped1 != mapped2;
                                let input_phased = phase_mask
                                    .and_then(|mask| mask.get(target_idx, sample_idx))
                                    .map(|v| v != 0)
                                    .unwrap_or(true);
                                if is_het && !input_phased {
                                    cached_query_pair = [
                                        crate::data::storage::AlleleCode::MISSING.raw(),
                                        crate::data::storage::AlleleCode::MISSING.raw(),
                                    ];
                                } else if is_het && input_phased {
                                    let qa1 = PbwtQueryAllele::allele(mapped1)
                                        .unwrap_or_else(PbwtQueryAllele::missing);
                                    let qa2 = PbwtQueryAllele::allele(mapped2)
                                        .unwrap_or_else(PbwtQueryAllele::missing);
                                    let phase_conf = target_win
                                        .sample_phase_confidence_f32(target_marker, sample_idx)
                                        .clamp(0.0, 1.0);
                                    let oriented_pair = if phase_conf < 0.5 {
                                        [qa2, qa1]
                                    } else {
                                        [qa1, qa2]
                                    };
                                    let self_query = oriented_pair[local];
                                    let mut beam_uncertainty = pbwt_beam_uncertainty(
                                        &beams[i],
                                        plan.n_ref_haps,
                                        self_query,
                                    );
                                    let peer_idx = peer_idx_by_hap[i];
                                    if let Some(peer_i) = peer_idx {
                                        let peer_local = haps[peer_i] % 2;
                                        let peer_query = oriented_pair[peer_local];
                                        let peer_uncertainty = pbwt_beam_uncertainty(
                                            &beams[peer_i],
                                            plan.n_ref_haps,
                                            peer_query,
                                        );
                                        beam_uncertainty =
                                            0.5 * (beam_uncertainty + peer_uncertainty);
                                    }
                                    let geno_conf = target_win
                                        .sample_confidence_f32(target_marker, sample_idx)
                                        .clamp(0.0, 1.0);
                                    let err_limit = phase_query_orientation_error_limit(
                                        geno_conf,
                                        beam_uncertainty,
                                    )
                                    .max(1e-6);
                                    let orientation_weight =
                                        phase_orientation_weight(phase_conf, err_limit);
                                    cached_allele_weight = orientation_weight;
                                    if phase_best_orientation_error(phase_conf) > err_limit {
                                        cached_query_pair = [
                                            crate::data::storage::AlleleCode::MISSING.raw(),
                                            crate::data::storage::AlleleCode::MISSING.raw(),
                                        ];
                                        cached_wildcard_weight =
                                            uncertain_orientation_wildcard_info_weight();
                                    } else if phase_conf < 0.5 {
                                        cached_query_pair = [mapped2, mapped1];
                                    } else {
                                        cached_query_pair = [mapped1, mapped2];
                                    }
                                } else {
                                    cached_query_pair = [mapped1, mapped2];
                                    cached_allele_weight = 1.0;
                                }
                            }
                            let query = PbwtQueryAllele::allele(cached_query_pair[local])
                                .unwrap_or_else(PbwtQueryAllele::wildcard);
                            let target_allele = query
                                .as_allele()
                                .unwrap_or(crate::data::storage::AlleleCode::MISSING.raw());
                            query_info_weight[i] = if target_allele
                                != crate::data::storage::AlleleCode::MISSING.raw()
                                && (target_allele as usize) < n_alleles
                            {
                                info_llr * cached_allele_weight
                            } else if query.is_wildcard() {
                                cached_wildcard_weight
                            } else {
                                0.0
                            };
                            query_alleles[i] = query;
                        }
                    } else {
                        for (qa, iw) in query_alleles.iter_mut().zip(query_info_weight.iter_mut()) {
                            *qa = PbwtQueryAllele::wildcard();
                            *iw = 0.0;
                        }
                    }

                    pbwt.update_beams_with_scratch_query(
                        beams,
                        query_alleles,
                        None,
                        n_alleles,
                        scratch,
                    );
                }
                pbwt.finalize_step(&ref_alleles, n_alleles, ref_m);

                // Donor/confusion evidence should be collected where the target
                // is informative in this window, and across the overlap tail for
                // handoff continuity.
                let aligned_here = target_m_opt.is_some();
                let in_overlap = ref_m >= overlap_start && ref_m < output_end;
                let store = aligned_here || in_overlap;

                if store {
                    for (haps, beams, query_alleles, query_info_weight, current_donor, _, _) in
                        batches.iter_mut()
                    {
                        let mut donor_picks: Vec<DonorPick> =
                            Vec::with_capacity(SM_MATCH_DONORS.saturating_mul(2));
                        for (i, &hap_idx) in haps.iter().enumerate() {
                            let beam = &beams[i];
                            let donor_k =
                                adaptive_sm_donor_k(beam, plan.n_ref_haps, query_alleles[i]);
                            donor_picks.clear();
                            pbwt.select_donor_picks_into(beam, donor_k, &mut donor_picks);
                            let donor = donor_picks
                                .first()
                                .map(|p| p.hap)
                                .unwrap_or(current_donor[i]);
                            current_donor[i] = donor;

                            let info_weight = query_info_weight[i];
                            if info_weight > 0.0 {
                                sm_total_info[hap_idx] += info_weight;
                            }

                            if info_weight > 0.0 {
                                let mut match_count: u32 = 0;
                                for &(l, r) in beam.intervals() {
                                    match_count = match_count.saturating_add(r.saturating_sub(l));
                                }
                                let n_matches = (match_count.max(1) as f32).ln();
                                let max_entropy = (plan.n_ref_haps as f32).ln().max(1e-6);
                                let normalized_entropy = (n_matches / max_entropy).clamp(0.0, 1.0);
                                sm_low_conf_weighted[hap_idx] += info_weight * normalized_entropy;
                            }
                            if info_weight <= 0.0 {
                                continue;
                            }
                            if donor_picks.is_empty() {
                                continue;
                            }
                            // Marker evidence mass (in nats) from typed-query emission info,
                            // tempered by candidate coverage:
                            //   alpha_m = info_weight(m) * c_m
                            //   c_m = |C_m| / k_m in [0,1]
                            // This avoids assigning full mass when PBWT returns a small
                            // candidate set (low recall / brittle beam context).
                            let coverage =
                                (donor_picks.len() as f32 / donor_k.max(1) as f32).clamp(0.0, 1.0);
                            let marker_mass = info_weight * coverage;
                            if marker_mass <= 0.0 || !marker_mass.is_finite() {
                                continue;
                            }
                            let mut log_mass: Vec<(RefHapId, f32)> =
                                Vec::with_capacity(donor_picks.len());
                            let mut max_log = f32::NEG_INFINITY;
                            for pick in donor_picks.iter() {
                                let start = pick.start.max(0) as usize;
                                let start = start.min(ref_m);
                                let log_cont = if start >= ref_m {
                                    0.0
                                } else {
                                    prefix_log_stay_full[ref_m + 1]
                                        - prefix_log_stay_full[start + 1]
                                };
                                // q_m(d) proportional to continuity mass:
                                //   q_m(d) proportional to cont_d(m) = exp(log_cont_d(m))
                                if log_cont > max_log {
                                    max_log = log_cont;
                                }
                                log_mass.push((RefHapId::new(pick.hap), log_cont));
                            }
                            let mut denom = 0.0f32;
                            for (_, v) in log_mass.iter() {
                                denom += (*v - max_log).exp();
                            }
                            if denom <= 0.0 || !denom.is_finite() {
                                let uniform = marker_mass / donor_picks.len().max(1) as f32;
                                for pick in donor_picks.iter() {
                                    push_donor_weight(
                                        &mut sm_donor_counts[hap_idx],
                                        RefHapId::new(pick.hap),
                                        uniform,
                                    );
                                }
                                continue;
                            }
                            let inv = 1.0 / denom;
                            for (hap, lv) in log_mass.into_iter() {
                                // Numerically stable softmax:
                                //   q_m(d) = exp(l_d - l_max) / sum_c exp(l_c - l_max)
                                // Donor accumulation objective:
                                //   W(d) += alpha_m * q_m(d)
                                let p = (lv - max_log).exp() * inv;
                                push_donor_weight(
                                    &mut sm_donor_counts[hap_idx],
                                    hap,
                                    marker_mass * p,
                                );
                            }
                        }
                    }
                }
            }
        }

        // Diagnostics: donor set size distribution across haplotypes.
        if should_log && n_target_haps > 0 {
            let mut min_donors = usize::MAX;
            let mut max_donors = 0usize;
            let mut sum_donors = 0usize;
            for counts in &sm_donor_counts {
                let len = counts.len();
                if len < min_donors {
                    min_donors = len;
                }
                if len > max_donors {
                    max_donors = len;
                }
                sum_donors += len;
            }
            let avg_donors = sum_donors as f64 / n_target_haps as f64;
            eprintln!(
                "    [debug donors] hap_donor_counts min={} avg={:.2} max={}",
                min_donors, avg_donors, max_donors
            );
        }

        thread_local! {
            static LOCAL_WORKSPACE: std::cell::RefCell<Option<ImputeWorkspace>> =
                std::cell::RefCell::new(None);
        }

        let prior_marker_idx = if output_end > 0 {
            Some(output_end.saturating_sub(1))
        } else {
            None
        };

        struct ImputeResult {
            result: SampleImputationResult,
            priors: Option<(HaplotypePriors, HaplotypePriors)>,
            last_info_idx: Option<usize>,
        }

        let telemetry = self.telemetry.clone();
        use std::sync::atomic::{AtomicUsize, Ordering};
        let dbg_use_hmm = AtomicUsize::new(0);
        let dbg_no_hmm = AtomicUsize::new(0);
        let dbg_no_info = AtomicUsize::new(0);
        let dbg_insufficient = AtomicUsize::new(0);
        let dbg_low_conf = AtomicUsize::new(0);
        let dbg_few_donors = AtomicUsize::new(0);
        let dbg_has_priors = AtomicUsize::new(0);
        let dbg_fallback_selected_priors = AtomicUsize::new(0);
        let output_markers = output_end.saturating_sub(output_start);
        let alt_prob_store_builder = if output_markers > 0 {
            Some(AltProbDiskStoreBuilder::new(
                n_target_samples,
                output_markers,
            )?)
        } else {
            None
        };
        let alt_prob_store_writer = alt_prob_store_builder.as_ref().map(|b| b.writer());

        let requested_threads = self
            .config
            .nthreads
            .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
            .unwrap_or(1)
            .max(1);
        let mut hmm_threads = requested_threads;
        let avail_bytes = crate::utils::memory::available_memory_bytes().unwrap_or(0);
        if avail_bytes >= MIN_AVAIL_BYTES_FOR_PLANNING {
            let hmm_budget_bytes =
                (avail_bytes as f64 * IMPUTE_RAM_FRACTION * STATE_BUDGET_SAFETY) as u64;
            let per_job_bytes =
                estimate_hmm_job_bytes(per_window_cap_local, n_ref_markers, target_win.n_markers());
            if per_job_bytes > 0 {
                let cap = (hmm_budget_bytes / per_job_bytes).max(1) as usize;
                hmm_threads = hmm_threads.min(cap.max(1));
            }
        }
        if should_log && hmm_threads < requested_threads {
            eprintln!(
                "    HMM parallelism capped by memory: threads {} -> {} (states={}, ref_markers={}, avail_mb={})",
                requested_threads,
                hmm_threads,
                per_window_cap_local,
                n_ref_markers,
                avail_bytes / (1024 * 1024)
            );
        }
        let hmm_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(hmm_threads.max(1))
            .build()
            .map_err(|e| {
                ReagleError::vcf(format!("Failed to build HMM worker thread pool: {}", e))
            })?;
        let (result_tx, result_rx) = std::sync::mpsc::channel::<Result<ImputeResult>>();
        hmm_pool.install(|| {
            (0..n_target_samples)
                .into_par_iter()
                .for_each_with(result_tx, |tx, s| {
                    let prior_error_rate = sample_error_rates[s].clamp(1e-6, 0.5);
                    let item: Result<ImputeResult> = (|| {
                let sample = sample_idx_from_usize(s);
                let h1_idx = sample.hap(HapSide::H1);
                let h2_idx = sample.hap(HapSide::H2);
                let compose_overlap_prior = |sample_idx: usize, hap_idx: usize| -> Option<std::borrow::Cow<'_, HaplotypePriors>> {
                    let priors_id = overlap_hap_priors_id?;
                    if hap_idx >= priors_id.len() {
                        return None;
                    }
                    let mate_idx = if hap_idx % 2 == 0 {
                        hap_idx.saturating_add(1)
                    } else {
                        hap_idx.saturating_sub(1)
                    };
                    if mate_idx >= priors_id.len() {
                        return None;
                    }
                    let w_swap = overlap_orientation_weight_swap
                        .and_then(|w| w.get(sample_idx))
                        .copied()
                        .unwrap_or(0.5);
                    let pri_keep = &priors_id[hap_idx];
                    let pri_swap = &priors_id[mate_idx];
                    if (w_swap as f64 - 0.5).abs() < ORIENTATION_HANDOFF_MIN_MARGIN {
                        // In non-identifiable orientation regions, avoid blending id/swap priors:
                        // averaging branches collapses biological support and can induce boundary drift.
                        return Some(std::borrow::Cow::Borrowed(pri_keep));
                    }
                    if w_swap <= 1e-6 {
                        Some(std::borrow::Cow::Borrowed(pri_keep))
                    } else if w_swap >= 1.0 - 1e-6 {
                        Some(std::borrow::Cow::Borrowed(pri_swap))
                    } else {
                        Some(std::borrow::Cow::Owned(compose_boundary_message(
                            pri_keep, pri_swap, w_swap,
                        )))
                    }
                };
                let prior_h1_composed = compose_overlap_prior(s, h1_idx.as_usize());
                let prior_h2_composed = compose_overlap_prior(s, h2_idx.as_usize());
                let priors_h1 = prior_h1_composed.as_deref();
                let priors_h2 = prior_h2_composed.as_deref();

                    let (mut input_probs_h1, mut input_probs_h2, last_info_h1, last_info_h2) =
                        build_input_probs_pair(h1_idx, h2_idx, s);
                let handoff_capture_idx_h1 = last_info_h1.or(prior_marker_idx);
                let handoff_capture_idx_h2 = last_info_h2.or(prior_marker_idx);
                // Information-weighted fallback decision: ratio of confused info to total info.
                // Missing targets provide no information, so treat missingness as low confidence.
        let total_info_h1 = sm_total_info[h1_idx.as_usize()].max(1e-9);
        let total_info_h2 = sm_total_info[h2_idx.as_usize()].max(1e-9);
        let conf_ratio_h1 = sm_low_conf_weighted[h1_idx.as_usize()] / total_info_h1;
        let conf_ratio_h2 = sm_low_conf_weighted[h2_idx.as_usize()] / total_info_h2;
        let insufficient_info_h1 = sm_total_info[h1_idx.as_usize()] < min_info_nats;
        let insufficient_info_h2 = sm_total_info[h2_idx.as_usize()] < min_info_nats;
        let no_info_h1 = sm_total_info[h1_idx.as_usize()] <= 0.0;
        let no_info_h2 = sm_total_info[h2_idx.as_usize()] <= 0.0;
                let has_priors_h1 = priors_h1.map(|p| !p.is_empty()).unwrap_or(false);
                let has_priors_h2 = priors_h2.map(|p| !p.is_empty()).unwrap_or(false);
                // Cap donor lists to the top-k by match weight. This is both
                // a speed optimization (avoid sorting thousands of donors) and
                // an accuracy improvement: concentrating the HMM on high-weight
                // donors produces sharper posteriors. See doc on
                // keep_top_k_donors_by_weight for empirical IQA results.
                let max_fast_donors = per_window_cap_local.saturating_mul(2).max(64);
                let build_donor_pool = |hap_usize: usize| -> Vec<(RefHapId, f32)> {
                    let mut combined: Vec<(RefHapId, f32)> = sm_donor_counts[hap_usize]
                        .iter()
                        .map(|(h, c)| (*h, *c))
                        .collect();
                    if !combined.is_empty() {
                        // Primary path: state proposal uses donor evidence W(d) learned from
                        // informative-marker mass and Li-Stephens continuity.
                        return combined;
                    }
                    // No donor evidence: fall back to structural priors only.
                    let mut fallback: HashMap<RefHapId, f32> = HashMap::new();
                    if let Some(core) = plan.core_states.get(hap_usize) {
                        for &h in core {
                            fallback.entry(h).or_insert(1.0);
                        }
                    }
                    if let Some(intervals) = plan.window_intervals.get(hap_usize) {
                        for interval in intervals {
                            if interval_support_over_range(interval, plan_range_start, plan_range_end)
                                .is_some()
                            {
                                fallback.entry(interval.hap).or_insert(1.0);
                            }
                        }
                    }
                    combined = fallback.into_iter().collect();
                    combined
                };
                let mut donors_h1 = build_donor_pool(h1_idx.as_usize());
                let mut donors_h2 = build_donor_pool(h2_idx.as_usize());
                let full_donors_h1 = donors_h1.clone();
                let full_donors_h2 = donors_h2.clone();
                if use_abyss {
                    if let Some(mask) = plan.abyss_mask.get(h1_idx.as_usize()) {
                        let before = donors_h1.len();
                        donors_h1.retain(|(h, _)| !mask[h.as_usize()]);
                        if donors_h1.is_empty() && before > 0 {
                            donors_h1 = sm_donor_counts[h1_idx.as_usize()]
                                .iter()
                                .map(|(h, c)| (*h, *c))
                                .collect();
                        }
                    }
                    if let Some(mask) = plan.abyss_mask.get(h2_idx.as_usize()) {
                        let before = donors_h2.len();
                        donors_h2.retain(|(h, _)| !mask[h.as_usize()]);
                        if donors_h2.is_empty() && before > 0 {
                            donors_h2 = sm_donor_counts[h2_idx.as_usize()]
                                .iter()
                                .map(|(h, c)| (*h, *c))
                                .collect();
                        }
                    }
                }
                keep_top_k_donors_by_weight(&mut donors_h1, max_fast_donors);
                keep_top_k_donors_by_weight(&mut donors_h2, max_fast_donors);
                let tiny_panel = plan.n_ref_haps <= 32;
                // Keep the fast-path/HMM gate conditional. PR #809 forced HMM
                // on for every haplotype here and lost to Beagle on chr21.
                let use_hmm_h1 = if tiny_panel {
                    true
                } else if has_priors_h1 {
                    true
                } else if no_info_h1 || insufficient_info_h1 {
                    true
                } else {
                    conf_ratio_h1 > SM_MATCH_LOW_CONF_FRAC
                        || donors_h1.len() < SM_MATCH_MIN_DONORS
                };
                let use_hmm_h2 = if tiny_panel {
                    true
                } else if has_priors_h2 {
                    true
                } else if no_info_h2 || insufficient_info_h2 {
                    true
                } else {
                    conf_ratio_h2 > SM_MATCH_LOW_CONF_FRAC
                        || donors_h2.len() < SM_MATCH_MIN_DONORS
                };

                let track_hmm = |use_hmm: bool,
                                 no_info: bool,
                                 insufficient: bool,
                                 conf_ratio: f32,
                                 donors_len: usize,
                                 has_priors: bool| {
                    if use_hmm {
                        dbg_use_hmm.fetch_add(1, Ordering::Relaxed);
                    } else {
                        dbg_no_hmm.fetch_add(1, Ordering::Relaxed);
                    }
                    if has_priors {
                        dbg_has_priors.fetch_add(1, Ordering::Relaxed);
                    }
                    if no_info {
                        dbg_no_info.fetch_add(1, Ordering::Relaxed);
                    }
                    if insufficient {
                        dbg_insufficient.fetch_add(1, Ordering::Relaxed);
                    }
                    if conf_ratio > SM_MATCH_LOW_CONF_FRAC {
                        dbg_low_conf.fetch_add(1, Ordering::Relaxed);
                    }
                    if donors_len < SM_MATCH_MIN_DONORS {
                        dbg_few_donors.fetch_add(1, Ordering::Relaxed);
                    }
                };

                track_hmm(
                    use_hmm_h1,
                    no_info_h1,
                    insufficient_info_h1,
                    conf_ratio_h1,
                    donors_h1.len(),
                    has_priors_h1,
                );
                track_hmm(
                    use_hmm_h2,
                    no_info_h2,
                    insufficient_info_h2,
                    conf_ratio_h2,
                    donors_h2.len(),
                    has_priors_h2,
                );

                let mut warned_no_priors = false;
                let mut warned_empty_map = false;
                let mut posts_probs_buf: Vec<f32> = Vec::new();
                let mix_prior_frac = self.config.state_mix_prior_frac.clamp(0.0, 1.0);
                let mix_window_frac = self.config.state_mix_window_frac.clamp(0.0, 1.0);
                let mix_donor_frac = self.config.state_mix_donor_frac.clamp(0.0, 1.0);
                let mix_core_frac = self.config.state_mix_core_frac.clamp(0.0, 1.0);
                let mix_prior_boost_min_frac = self
                    .config
                    .state_mix_prior_boost_min_frac
                    .clamp(0.0, 1.0);
                let mix_prior_boost_donor_min_frac = self
                    .config
                    .state_mix_prior_boost_donor_min_frac
                    .clamp(0.0, 1.0);
                let mix_prior_boost_core_max_frac = self
                    .config
                    .state_mix_prior_boost_core_max_frac
                    .clamp(0.0, 1.0);
                let mix_prior_floor_frac = self.config.state_mix_prior_floor_frac.clamp(0.0, 1.0);
                let mix_weak_signal_threshold = self
                    .config
                    .state_mix_weak_signal_threshold
                    .clamp(0.0, 1.0);
                let mix_weak_prior_frac = self.config.state_mix_weak_prior_frac.clamp(0.0, 1.0);
                let mix_weak_window_frac = self.config.state_mix_weak_window_frac.clamp(0.0, 1.0);
                let mix_weak_donor_frac = self.config.state_mix_weak_donor_frac.clamp(0.0, 1.0);
                let mix_weak_core_frac = self.config.state_mix_weak_core_frac.clamp(0.0, 1.0);
                let posts_from_donors =
                    |donors: &[(RefHapId, f32)], probs_buf: &mut Vec<f32>| -> Result<Vec<AllelePosteriors>> {
                    let mut out: Vec<AllelePosteriors> =
                        Vec::with_capacity(output_end.saturating_sub(output_start));
                    let total: f32 = donors
                        .iter()
                        .map(|(_, c)| if c.is_finite() && *c > 0.0 { *c } else { 0.0 })
                        .sum();
                    if total <= 0.0 {
                        return Err(ReagleError::vcf(format!(
                            "Empty donor subset for posterior construction: window={} sample={}",
                            window_idx, s
                        )));
                    }
                    let inv_total = 1.0f32 / total;
                    for ref_m in output_start..output_end {
                        let n_alleles = ref_markers
                            .marker(MarkerIdx::new(ref_m as u32))
                            .n_alleles()
                            .max(1);
                        probs_buf.clear();
                        probs_buf.resize(n_alleles, 0.0);
                        for (hap, c) in donors.iter() {
                            let allele = ref_columns
                                .get(ref_m)
                                .map(|col| col.get(HapIdx::new(hap.as_u32())))
                                .unwrap_or(crate::data::storage::AlleleCode::MISSING.raw());
                            if allele == crate::data::storage::AlleleCode::MISSING.raw() {
                                continue;
                            }
                            let idx = allele as usize;
                            if idx < probs_buf.len() {
                                probs_buf[idx] += *c * inv_total;
                            }
                        }
                        let sum: f32 = probs_buf.iter().sum();
                        if sum <= 0.0 {
                            return Err(ReagleError::vcf(format!(
                                "Donor-based posterior collapsed at marker: window={} sample={} marker={} donors={}",
                                window_idx,
                                s,
                                ref_m,
                                donors.len()
                            )));
                        }
                        let inv = 1.0 / sum;
                        for v in probs_buf.iter_mut() {
                            *v *= inv;
                        }
                        if n_alleles == 2 {
                            out.push(AllelePosteriors::Biallelic(
                                probs_buf.get(1).copied().unwrap_or(0.0),
                            ));
                        } else {
                            out.push(AllelePosteriors::Multiallelic(
                                std::sync::Arc::<[f32]>::from(probs_buf.clone()),
                            ));
                        }
                    }
                    Ok(out)
                };

                let build_state_haps = |hap_idx: HapIdx,
                                        priors: Option<&HaplotypePriors>,
                                        donors: &[(RefHapId, f32)],
                                        informative_ratio: f32,
                                        planning_range: Option<(usize, usize)>|
                 -> Vec<RefHapId> {
                    let has_nonempty_priors = priors.map(|p| !p.is_empty()).unwrap_or(false);
                    if plan.full_panel {
                        // In explicit full-panel mode, keep the exact LS state universe.
                        // This avoids donor truncation and preserves rare-carrier support.
                        return (0..plan.n_ref_haps)
                            .map(|h| RefHapId::new(h as u32))
                            .collect();
                    }
                    if let Some(full) = full_states.as_ref() {
                        return full.clone();
                    }
                    let k = per_window_cap_local.max(1).min(plan.n_ref_haps.max(1));
                    let mut out: Vec<RefHapId> = Vec::with_capacity(k);
                    let mut seen: std::collections::HashSet<RefHapId> =
                        std::collections::HashSet::with_capacity(k * 2);

                    let mut prior_haps: Vec<RefHapId> = Vec::new();
                    if let Some(p) = priors {
                        let mut weighted: Vec<(RefHapId, f32)> = p
                            .ids()
                            .iter()
                            .zip(p.probs().iter())
                            .map(|(id, prob)| (RefHapId::new(id.0), *prob))
                            .collect();
                        // Truncate to top-k priors so the state builder
                        // focuses capacity on high-posterior candidates.
                        keep_top_k_haps_by_prob(&mut weighted, k.saturating_mul(2).max(64));
                        prior_haps.extend(weighted.into_iter().map(|(hap, _)| hap));
                    }

                    let mut local_window_haps: Vec<RefHapId> = Vec::new();
                    if let Some((seg_plan_start, seg_plan_end)) = planning_range {
                        if hap_idx.as_usize() < plan.window_intervals.len() {
                            let mut ranked: Vec<(RefHapId, u32)> = Vec::new();
                            for hi in plan.window_intervals[hap_idx.as_usize()].iter() {
                                if let Some(span) =
                                    interval_support_over_range(hi, seg_plan_start, seg_plan_end)
                                {
                                    ranked.push((hi.hap, span));
                                }
                            }
                            ranked.sort_unstable_by(|a, b| {
                                b.1.cmp(&a.1).then_with(|| a.0.as_u32().cmp(&b.0.as_u32()))
                            });
                            local_window_haps.extend(ranked.into_iter().map(|(hap, _)| hap));
                        }
                    }
                    let empty_haps: &[RefHapId] = &[];
                    let window_haps: &[RefHapId] = if local_window_haps.is_empty() {
                        state_haps_by_hap
                            .get(hap_idx.as_usize())
                            .map(|v| v.as_slice())
                            .unwrap_or(empty_haps)
                    } else {
                        local_window_haps.as_slice()
                    };
                    let core_haps: &[RefHapId] = plan
                        .core_states
                        .get(hap_idx.as_usize())
                        .map(|v| v.as_slice())
                        .unwrap_or(empty_haps);

                    let fill_from = |out: &mut Vec<RefHapId>,
                                     seen: &mut std::collections::HashSet<RefHapId>,
                                     source: &[RefHapId],
                                     want: usize,
                                     k: usize|
                     -> usize {
                        if want == 0 || out.len() >= k {
                            return 0;
                        }
                        let mut added = 0usize;
                        for &hap in source {
                            if out.len() >= k || added >= want {
                                break;
                            }
                            if seen.insert(hap) {
                                out.push(hap);
                                added += 1;
                            }
                        }
                        added
                    };
                    let fill_from_donors =
                        |out: &mut Vec<RefHapId>,
                         seen: &mut std::collections::HashSet<RefHapId>,
                         source: &[(RefHapId, f32)],
                         want: usize,
                         k: usize|
                         -> usize {
                            if want == 0 || out.len() >= k {
                                return 0;
                            }
                            let mut added = 0usize;
                            for &(hap, _) in source {
                                if out.len() >= k || added >= want {
                                    break;
                                }
                                if seen.insert(hap) {
                                    out.push(hap);
                                    added += 1;
                                }
                            }
                            added
                        };

                    let quota4 = |k: usize, a: f32, b: f32, c: f32, d: f32| -> [usize; 4] {
                        if k == 0 {
                            return [0, 0, 0, 0];
                        }
                        let mut w = [a.max(0.0), b.max(0.0), c.max(0.0), d.max(0.0)];
                        let sum = w[0] + w[1] + w[2] + w[3];
                        if !sum.is_finite() || sum <= 0.0 {
                            w = [1.0, 1.0, 1.0, 1.0];
                        }
                        let sum = w[0] + w[1] + w[2] + w[3];
                        let mut q = [0usize; 4];
                        let mut frac = [0.0f32; 4];
                        let mut used = 0usize;
                        for i in 0..4 {
                            let exact = (k as f32) * (w[i] / sum);
                            let base = exact.floor() as usize;
                            q[i] = base;
                            frac[i] = exact - base as f32;
                            used += base;
                        }
                        while used < k {
                            let mut best = 0usize;
                            for i in 1..4 {
                                if frac[i] > frac[best] {
                                    best = i;
                                }
                            }
                            q[best] += 1;
                            frac[best] = -1.0;
                            used += 1;
                        }
                        q
                    };

                    let [mut q_prior, mut q_window, mut q_donor, mut q_core] = quota4(
                        k,
                        mix_prior_frac,
                        mix_window_frac,
                        mix_donor_frac,
                        mix_core_frac,
                    );
                    if donors.is_empty() {
                        q_donor = 0;
                    }
                    if has_nonempty_priors {
                        // Keep continuity strong across windows, but do not fully disable
                        // local donors (they remain useful at rare/local mismatches).
                        q_prior = q_prior.max((k as f32 * mix_prior_boost_min_frac).floor() as usize);
                        if !donors.is_empty() {
                            q_donor = q_donor.max(
                                (k as f32 * mix_prior_boost_donor_min_frac).floor() as usize
                            );
                        }
                        q_core = q_core.min((k as f32 * mix_prior_boost_core_max_frac).ceil() as usize);
                    }
                    if informative_ratio <= mix_weak_signal_threshold && !has_nonempty_priors {
                        // In weak local signal, broaden the state set with global core
                        // diversity. PBWT donors can be brittle on sparse arrays and
                        // excluding rare carriers collapses AF.
                        let weak_q = quota4(
                            k,
                            mix_weak_prior_frac,
                            mix_weak_window_frac,
                            mix_weak_donor_frac,
                            mix_weak_core_frac,
                        );
                        q_prior = weak_q[0];
                        q_window = weak_q[1];
                        q_donor = if donors.is_empty() { 0 } else { weak_q[2] };
                        q_core = weak_q[3];
                    }
                    let mut used_q = q_prior + q_window + q_donor + q_core;
                    while used_q < k {
                        q_prior += 1;
                        used_q += 1;
                        if used_q >= k {
                            break;
                        }
                        q_window += 1;
                        used_q += 1;
                        if used_q >= k {
                            break;
                        }
                        q_donor += 1;
                        used_q += 1;
                        if used_q >= k {
                            break;
                        }
                        q_core += 1;
                        used_q += 1;
                    }

                    let prior_floor = if has_nonempty_priors {
                        ((k as f32 * mix_prior_floor_frac).floor() as usize).min(prior_haps.len())
                    } else {
                        0
                    };
                    fill_from(&mut out, &mut seen, &prior_haps, prior_floor, k);
                    fill_from(&mut out, &mut seen, &prior_haps, q_prior, k);
                    fill_from(&mut out, &mut seen, window_haps, q_window, k);
                    fill_from_donors(&mut out, &mut seen, donors, q_donor, k);
                    fill_from(&mut out, &mut seen, core_haps, q_core, k);

                    while out.len() < k {
                        let before = out.len();
                        let remaining = k - out.len();
                        fill_from(&mut out, &mut seen, &prior_haps, remaining, k);
                        let remaining = k - out.len();
                        fill_from(&mut out, &mut seen, window_haps, remaining, k);
                        let remaining = k - out.len();
                        fill_from_donors(&mut out, &mut seen, donors, remaining, k);
                        let remaining = k - out.len();
                        fill_from(&mut out, &mut seen, core_haps, remaining, k);
                        if out.len() == before {
                            break;
                        }
                    }

                    // Keep active state width stable across segments/windows for
                    // SIMD/cache behavior and consistent subset-transition math.
                    // If donor/core/prior pools are exhausted, fill deterministically
                    // from the global panel universe.
                    if out.len() < k {
                        for h in 0..plan.n_ref_haps {
                            if out.len() >= k {
                                break;
                            }
                            let hap = RefHapId::new(h as u32);
                            if seen.insert(hap) {
                                out.push(hap);
                            }
                        }
                    }

                    out
                };

                let project_states_min_replacements =
                    |state_haps: &mut Vec<RefHapId>,
                     input_probs: &TargetAlleleProbs,
                     local_ref_columns: &[GenotypeColumn]| {
                        if state_haps.is_empty() || local_ref_columns.is_empty() {
                            return;
                        }
                        let k = state_haps.len();
                        if k >= n_transition_haps {
                            return;
                        }
                        let n_markers = input_probs.n_markers().min(local_ref_columns.len());
                        if n_markers == 0 {
                            return;
                        }

                        #[derive(Clone)]
                        struct Constraint {
                            carriers: Vec<RefHapId>,
                            weight: f32,
                        }

                        let mut constraints: Vec<Constraint> = Vec::new();
                        let mut carriers_buf: Vec<RefHapId> = Vec::new();
                        for m in 0..n_markers {
                            if input_probs.is_uniform_marker(m) {
                                continue;
                            }
                            let probs = input_probs.probs_for_marker(m);
                            if probs.is_empty() {
                                continue;
                            }
                            let col = &local_ref_columns[m];
                            let mut ranked: Vec<(usize, f32)> = probs
                                .iter()
                                .enumerate()
                                .filter_map(|(idx, &p)| {
                                    if p.is_finite() && p > 0.0 {
                                        Some((idx, p))
                                    } else {
                                        None
                                    }
                                })
                                .collect();
                            if ranked.is_empty() {
                                continue;
                            }
                            ranked.sort_by(|a, b| {
                                b.1.partial_cmp(&a.1)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                                    .then_with(|| a.0.cmp(&b.0))
                            });
                            let mut needed: Vec<(u8, f32)> = Vec::new();
                            let mut seen_alleles: std::collections::HashSet<u8> =
                                std::collections::HashSet::new();

                            if probs.len() == 2 {
                                let p0 = probs[0].max(0.0);
                                let p1 = probs[1].max(0.0);
                                if p0 >= 0.2 && p1 >= 0.2 {
                                    push_needed_allele(&mut needed, &mut seen_alleles, 0, p0);
                                    push_needed_allele(&mut needed, &mut seen_alleles, 1, p1);
                                } else {
                                    let top = if p1 > p0 { 1usize } else { 0usize };
                                    push_needed_allele(
                                        &mut needed,
                                        &mut seen_alleles,
                                        top,
                                        probs[top],
                                    );
                                }
                                let alt_freq = if col.n_haplotypes() > 0 {
                                    col.alt_count() as f32 / col.n_haplotypes() as f32
                                } else {
                                    0.0
                                };
                                let alt_supported = probs[1].max(0.0);
                                if alt_freq <= 0.02 && alt_supported >= 0.05 {
                                    push_needed_allele(
                                        &mut needed,
                                        &mut seen_alleles,
                                        1,
                                        alt_supported.max(0.25),
                                    );
                                }
                            } else {
                                let mut cumulative = 0.0f32;
                                for &(allele_idx, p) in &ranked {
                                    if p < 0.2 {
                                        continue;
                                    }
                                    push_needed_allele(
                                        &mut needed,
                                        &mut seen_alleles,
                                        allele_idx,
                                        p,
                                    );
                                    cumulative += p;
                                    if cumulative >= 0.9 {
                                        break;
                                    }
                                }
                                if needed.is_empty() {
                                    let (allele_idx, p) = ranked[0];
                                    push_needed_allele(
                                        &mut needed,
                                        &mut seen_alleles,
                                        allele_idx,
                                        p,
                                    );
                                    cumulative = p;
                                }
                                for &(allele_idx, p) in &ranked {
                                    if cumulative >= 0.9 || needed.len() >= 3 {
                                        break;
                                    }
                                    if allele_idx > u8::MAX as usize {
                                        continue;
                                    }
                                    let allele = allele_idx as u8;
                                    if seen_alleles.contains(&allele) {
                                        continue;
                                    }
                                    push_needed_allele(
                                        &mut needed,
                                        &mut seen_alleles,
                                        allele_idx,
                                        p,
                                    );
                                    cumulative += p;
                                }
                            }

                            for (allele, weight) in needed {
                                if is_represented_in_states(state_haps, col, allele) {
                                    continue;
                                }
                                collect_carriers_for_allele(col, allele, k, &mut carriers_buf);
                                if carriers_buf.is_empty() {
                                    continue;
                                }
                                constraints.push(Constraint {
                                    carriers: carriers_buf.clone(),
                                    weight,
                                });
                            }
                        }

                        if constraints.is_empty() {
                            return;
                        }

                        #[derive(Clone)]
                        struct Candidate {
                            hap: RefHapId,
                            is_base: bool,
                            constraints: Vec<usize>,
                        }

                        let mut candidates: Vec<Candidate> = state_haps
                            .iter()
                            .copied()
                            .map(|hap| Candidate {
                                hap,
                                is_base: true,
                                constraints: Vec::new(),
                            })
                            .collect();
                        let mut cand_index: std::collections::HashMap<RefHapId, usize> =
                            std::collections::HashMap::with_capacity(k.saturating_mul(4));
                        for (i, c) in candidates.iter().enumerate() {
                            cand_index.insert(c.hap, i);
                        }

                        for (j, constraint) in constraints.iter().enumerate() {
                            for &hap in &constraint.carriers {
                                let idx = if let Some(&idx) = cand_index.get(&hap) {
                                    idx
                                } else {
                                    let idx = candidates.len();
                                    candidates.push(Candidate {
                                        hap,
                                        is_base: false,
                                        constraints: Vec::new(),
                                    });
                                    cand_index.insert(hap, idx);
                                    idx
                                };
                                candidates[idx].constraints.push(j);
                            }
                        }

                        if candidates.len() <= k {
                            return;
                        }

                        let r = constraints.len();
                        let mut lambdas = vec![0.0f32; r];
                        let mut selected: Vec<usize> = Vec::with_capacity(k);
                        let mut covered = vec![false; r];
                        let mut best_selected: Option<Vec<usize>> = None;
                        let mut best_covered_weight = 0.0f32;
                        let mut best_base = 0usize;
                        let mut scored: Vec<(usize, f32)> = Vec::with_capacity(candidates.len());
                        const MAX_DUAL_ITERS: usize = 32;

                        for iter in 0..MAX_DUAL_ITERS {
                            scored.clear();
                            for (idx, cand) in candidates.iter().enumerate() {
                                let mut score = if cand.is_base { 1.0 } else { 0.0 };
                                for &j in &cand.constraints {
                                    score += lambdas[j];
                                }
                                scored.push((idx, score));
                            }
                            let rank_cmp = |a: &(usize, f32), b: &(usize, f32)| {
                                b.1.partial_cmp(&a.1)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                                    .then_with(|| {
                                        candidates[b.0]
                                            .is_base
                                            .cmp(&candidates[a.0].is_base)
                                    })
                                    .then_with(|| {
                                        candidates[a.0]
                                            .hap
                                            .as_u32()
                                            .cmp(&candidates[b.0].hap.as_u32())
                                    })
                            };
                            if scored.len() > k {
                                scored.select_nth_unstable_by(k - 1, rank_cmp);
                            }

                            selected.clear();
                            covered.fill(false);
                            let mut covered_count = 0usize;
                            let mut covered_weight = 0.0f32;
                            for &(idx, _) in scored.iter().take(k) {
                                selected.push(idx);
                                for &j in &candidates[idx].constraints {
                                    if !covered[j] {
                                        covered[j] = true;
                                        covered_count += 1;
                                        covered_weight += constraints[j].weight;
                                    }
                                }
                            }
                            let base_count = selected
                                .iter()
                                .filter(|&&idx| candidates[idx].is_base)
                                .count();
                            if covered_weight > best_covered_weight + f32::EPSILON
                                || ((covered_weight - best_covered_weight).abs() <= f32::EPSILON
                                    && base_count > best_base)
                            {
                                best_covered_weight = covered_weight;
                                best_base = base_count;
                                best_selected = Some(selected.clone());
                            }
                            if covered_count == r {
                                best_selected = Some(selected.clone());
                                break;
                            }

                            let eta = 1.0f32 / ((iter + 1) as f32).sqrt();
                            for j in 0..r {
                                let v = if covered[j] {
                                    0.0
                                } else {
                                    constraints[j].weight
                                };
                                lambdas[j] = (lambdas[j] + eta * v).max(0.0);
                            }
                        }

                        let Some(chosen) = best_selected else {
                            return;
                        };
                        let state_set: std::collections::HashSet<RefHapId> =
                            state_haps.iter().copied().collect();
                        let chosen_set: std::collections::HashSet<RefHapId> = chosen
                            .iter()
                            .map(|&idx| candidates[idx].hap)
                            .collect();

                        let mut out: Vec<RefHapId> = Vec::with_capacity(k);
                        for &hap in state_haps.iter() {
                            if chosen_set.contains(&hap) {
                                out.push(hap);
                            }
                        }
                        let mut extras: Vec<RefHapId> = chosen
                            .iter()
                            .map(|&idx| candidates[idx].hap)
                            .filter(|hap| !state_set.contains(hap))
                            .collect();
                        extras.sort_unstable_by_key(|h| h.as_u32());
                        for hap in extras {
                            if out.len() >= k {
                                break;
                            }
                            out.push(hap);
                        }
                        if out.len() == k {
                            *state_haps = out;
                        }
                    };

                let mut mapped_priors_buf: Vec<f32> = Vec::new();
                let mut mapped_entry_pi_buf: Vec<f32> = Vec::new();
                let mut prev_states_buf: Vec<RefHapId> = Vec::new();
                let build_entry_pi =
                    |state_haps: &[RefHapId],
                     donors: &[(RefHapId, f32)],
                     out: &mut Vec<f32>|
                     -> bool {
                        if state_haps.is_empty() {
                            return false;
                        }
                        out.clear();
                        out.resize(state_haps.len(), 0.0);
                        let mut donor_weight: std::collections::HashMap<RefHapId, f32> =
                            std::collections::HashMap::with_capacity(donors.len() * 2);
                        for &(hap, c) in donors {
                            let w = c;
                            if w > 0.0 && w.is_finite() {
                                donor_weight
                                    .entry(hap)
                                    .and_modify(|v| *v += w)
                                    .or_insert(w);
                            }
                        }
                        let mut sum = 0.0f32;
                        for (i, &hap) in state_haps.iter().enumerate() {
                            let w = donor_weight.get(&hap).copied().unwrap_or(0.0);
                            out[i] = w;
                            sum += w;
                        }
                        if sum > 0.0 {
                            let inv = 1.0 / sum;
                            for v in out.iter_mut() {
                                *v *= inv;
                            }
                            true
                        } else {
                            false
                        }
                    };
                let compute_transition_lambda =
                    |state_haps: &[RefHapId], donors: &[(RefHapId, f32)]| -> f32 {
                        const LAMBDA_MAX: f32 = 0.94;
                        if state_haps.is_empty() {
                            return 0.0;
                        }
                        if state_haps.len() == 1 {
                            return LAMBDA_MAX;
                        }
                        let mut donor_weight: std::collections::HashMap<RefHapId, f32> =
                            std::collections::HashMap::with_capacity(donors.len().max(1) * 2);
                        for &(hap, c) in donors {
                            let w = c;
                            if w.is_finite() && w > 0.0 {
                                donor_weight
                                    .entry(hap)
                                    .and_modify(|v| *v += w)
                                    .or_insert(w);
                            }
                        }
                        let mut sum_w = 0.0f32;
                        let mut sum_w2 = 0.0f32;
                        let mut k_ess = 0usize;
                        for &hap in state_haps {
                            let w = donor_weight.get(&hap).copied().unwrap_or(0.0);
                            if w > 0.0 && w.is_finite() {
                                sum_w += w;
                                sum_w2 += w * w;
                                k_ess += 1;
                            }
                        }
                        if sum_w <= 0.0 || sum_w2 <= 0.0 || !sum_w.is_finite() || !sum_w2.is_finite()
                        {
                            // No donor support on the selected state set -> use canonical subset
                            // transition (lambda=0) rather than injecting sticky behavior.
                            return 0.0;
                        }
                        if k_ess <= 1 {
                            return LAMBDA_MAX;
                        }
                        // Effective sample size over donor weights on the FINAL selected state set:
                        //   Neff = (sum w)^2 / sum(w^2), with Neff in [1, K_ess].
                        // (By Cauchy-Schwarz: (sum w)^2 <= K_ess * sum(w^2).)
                        //
                        // Map ESS to stickiness:
                        //   ess_norm = (Neff - 1) / (K_ess - 1) in [0, 1]
                        //   lambda   = LAMBDA_MAX * (1 - ess_norm)
                        // so concentrated donor support (Neff -> 1) gives high lambda,
                        // diffuse support (Neff -> K_ess) gives lambda -> 0 (canonical).
                        //
                        // We normalize by K_ess (supported states), not raw K, to avoid
                        // artificial stickiness when many selected states have zero donor weight.
                        let k_ess_f = k_ess as f32;
                        let neff = ((sum_w * sum_w) / sum_w2).clamp(1.0, k_ess_f);
                        let ess_norm = ((neff - 1.0) / (k_ess_f - 1.0)).clamp(0.0, 1.0);
                        (LAMBDA_MAX * (1.0 - ess_norm)).clamp(0.0, LAMBDA_MAX)
                    };
                let exact_no_info_posteriors =
                    |hap_idx: HapIdx,
                     donors: &[(RefHapId, f32)],
                     probs_buf: &mut Vec<f32>|
                     -> Result<(Vec<AllelePosteriors>, HaplotypePriors)> {
                        let state_haps = build_state_haps(hap_idx, None, donors, 0.0, None);
                        if state_haps.is_empty() {
                            return Err(ReagleError::vcf(format!(
                                "State selection produced empty subset in no-info fast path: window={} sample={} hap={} donors={}",
                                window_idx,
                                s,
                                hap_idx.as_usize(),
                                donors.len()
                            )));
                        }

                        let k = state_haps.len() as f32;
                        let uniform_weight = 1.0 / k;
                        let mut out: Vec<AllelePosteriors> =
                            Vec::with_capacity(output_end.saturating_sub(output_start));
                        for ref_m in output_start..output_end {
                            let n_alleles = ref_markers
                                .marker(MarkerIdx::new(ref_m as u32))
                                .n_alleles()
                                .max(1);
                            probs_buf.clear();
                            probs_buf.resize(n_alleles, 0.0);
                            for &hap in &state_haps {
                                let allele = ref_columns
                                    .get(ref_m)
                                    .map(|col| col.get(HapIdx::new(hap.as_u32())))
                                    .unwrap_or(crate::data::storage::AlleleCode::MISSING.raw());
                                if allele == crate::data::storage::AlleleCode::MISSING.raw() {
                                    continue;
                                }
                                let idx = allele as usize;
                                if idx < probs_buf.len() {
                                    probs_buf[idx] += uniform_weight;
                                }
                            }
                            let sum: f32 = probs_buf.iter().sum();
                            if sum <= 0.0 {
                                return Err(ReagleError::vcf(format!(
                                    "No-info fast path posterior collapsed at marker: window={} sample={} hap={} marker={} states={}",
                                    window_idx,
                                    s,
                                    hap_idx.as_usize(),
                                    ref_m,
                                    state_haps.len()
                                )));
                            }
                            let inv = 1.0 / sum;
                            for v in probs_buf.iter_mut() {
                                *v *= inv;
                            }
                            if n_alleles == 2 {
                                out.push(AllelePosteriors::Biallelic(
                                    probs_buf.get(1).copied().unwrap_or(0.0),
                                ));
                            } else {
                                out.push(AllelePosteriors::Multiallelic(
                                    std::sync::Arc::<[f32]>::from(probs_buf.clone()),
                                ));
                            }
                        }

                        let ids: Vec<GlobalHapId> =
                            state_haps.iter().map(|h| GlobalHapId(h.as_u32())).collect();
                        let probs = vec![uniform_weight; state_haps.len()];
                        Ok((out, HaplotypePriors::new(ids, probs)))
                    };
                let exact_transition_only_from_priors =
                    |hap_idx: HapIdx,
                     priors: &HaplotypePriors,
                     donors: &[(RefHapId, f32)],
                     capture_idx: Option<usize>,
                     probs_buf: &mut Vec<f32>|
                     -> Result<(Vec<AllelePosteriors>, HaplotypePriors)> {
                        if priors.is_empty() {
                            return Err(ReagleError::vcf(format!(
                                "Cannot run transition-only propagation with empty priors: window={} sample={} hap={}",
                                window_idx,
                                s,
                                hap_idx.as_usize()
                            )));
                        }

                        let k = priors.ids().len();
                        let mut state_probs = priors.probs().to_vec();
                        let mut state_sum = 0.0f32;
                        for v in &mut state_probs {
                            if !v.is_finite() || *v < 0.0 {
                                *v = 0.0;
                            }
                            state_sum += *v;
                        }
                        if state_sum <= 0.0 {
                            return Err(ReagleError::vcf(format!(
                                "Transition-only propagation got zero prior mass: window={} sample={} hap={} states={}",
                                window_idx,
                                s,
                                hap_idx.as_usize(),
                                k
                            )));
                        }
                        let inv0 = 1.0 / state_sum;
                        for v in &mut state_probs {
                            *v *= inv0;
                        }
                        state_sum = 1.0;

                        let mut captured_probs: Option<Vec<f32>> = None;
                        let mut out: Vec<AllelePosteriors> =
                            Vec::with_capacity(output_end.saturating_sub(output_start));
                        let prior_state_haps: Vec<RefHapId> = priors
                            .ids()
                            .iter()
                            .map(|id| RefHapId::new(id.0))
                            .collect();
                        let transition_lambda = compute_transition_lambda(&prior_state_haps, donors);

                        for ref_m in 0..output_end {
                            let recomb_rate = p_recomb.get(ref_m).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                            if recomb_rate > 0.0 && !state_probs.is_empty() {
                                let k_f = k as f32;
                                let n_f = n_transition_haps_f32.max(1.0);
                                // Same fixed-lambda transition family used by run_impute_hmm:
                                // equivalent to canonical subset transition at r_eff=r*(1-lambda).
                                let rho =
                                    transition_lambda + (1.0 - transition_lambda) * (k_f / n_f);
                                let z = ((1.0 - recomb_rate) + recomb_rate * rho).max(1e-30);
                                let stay_gap =
                                    ((1.0 - recomb_rate) + recomb_rate * transition_lambda) / z;
                                let shift =
                                    (recomb_rate * (1.0 - transition_lambda) / n_f) / z;
                                let scale = stay_gap / state_sum.max(1e-30);
                                let mut new_sum = 0.0f32;
                                for v in &mut state_probs {
                                    let t = scale.mul_add(*v, shift);
                                    *v = t;
                                    new_sum += t;
                                }
                                state_sum = new_sum.max(1e-30);
                            }

                            if capture_idx == Some(ref_m) {
                                let inv = 1.0 / state_sum.max(1e-30);
                                captured_probs = Some(state_probs.iter().map(|v| v * inv).collect());
                            }

                            if ref_m < output_start {
                                continue;
                            }

                            let n_alleles = ref_markers
                                .marker(MarkerIdx::new(ref_m as u32))
                                .n_alleles()
                                .max(1);
                            probs_buf.clear();
                            probs_buf.resize(n_alleles, 0.0);
                            for (id, p) in priors.ids().iter().zip(state_probs.iter()) {
                                let hap = HapIdx::new(id.0);
                                let allele = ref_columns
                                    .get(ref_m)
                                    .map(|col| col.get(hap))
                                    .unwrap_or(crate::data::storage::AlleleCode::MISSING.raw());
                                if allele == crate::data::storage::AlleleCode::MISSING.raw() {
                                    continue;
                                }
                                let idx = allele as usize;
                                if idx < probs_buf.len() {
                                    let pn = *p / state_sum.max(1e-30);
                                    probs_buf[idx] += pn.max(0.0);
                                }
                            }
                            let sum: f32 = probs_buf.iter().sum();
                            if sum <= 0.0 {
                                return Err(ReagleError::vcf(format!(
                                    "Transition-only posterior collapsed: window={} sample={} hap={} marker={} states={}",
                                    window_idx,
                                    s,
                                    hap_idx.as_usize(),
                                    ref_m,
                                    k
                                )));
                            }
                            let inv = 1.0 / sum;
                            for v in probs_buf.iter_mut() {
                                *v *= inv;
                            }
                            if n_alleles == 2 {
                                out.push(AllelePosteriors::Biallelic(
                                    probs_buf.get(1).copied().unwrap_or(0.0),
                                ));
                            } else {
                                out.push(AllelePosteriors::Multiallelic(
                                    std::sync::Arc::<[f32]>::from(probs_buf.clone()),
                                ));
                            }
                        }

                        let probs = captured_probs.unwrap_or_else(|| {
                            let inv = 1.0 / state_sum.max(1e-30);
                            state_probs.iter().map(|v| v * inv).collect()
                        });
                        Ok((out, HaplotypePriors::new(priors.ids().to_vec(), probs)))
                    };
                    let mut process_haplotype = |hap_idx: HapIdx,
                                                 priors: Option<&HaplotypePriors>,
                                                 input_probs: &mut TargetAlleleProbs,
                                                 error_rate: f32,
                                                 prior_marker_idx: Option<usize>,
                                                 donors: &[(RefHapId, f32)],
                                                 donor_pool_full: &[(RefHapId, f32)]|
                 -> Result<(
                    Vec<AllelePosteriors>,
                    HaplotypePriors,
                    bool,
                    f32,
                )> {
                    let n_markers = input_probs.n_markers();
                    let informative_n = (0..n_markers)
                        .filter(|&m| !input_probs.is_uniform_marker(m))
                        .count();
                    let informative_ratio = if n_markers == 0 {
                        0.0
                    } else {
                        informative_n as f32 / n_markers as f32
                    };
                    let use_piecewise_segments = n_markers > 0
                        && plan_range_end > plan_range_start + 1
                        && plan_range_end <= plan.planning_handoff.len();
                    if use_piecewise_segments {
                        let mut marker_plan_idx = vec![plan_range_start; n_markers];
                        let mut pidx = plan_range_start;
                        for (m, slot) in marker_plan_idx.iter_mut().enumerate() {
                            let gp = gen_positions.get(m).copied().unwrap_or(f64::NAN);
                            if !gp.is_finite() {
                                *slot = pidx;
                                continue;
                            }
                            while pidx + 1 < plan_range_end && gp >= plan.planning_handoff[pidx].1 {
                                pidx += 1;
                            }
                            while pidx > plan_range_start && gp < plan.planning_handoff[pidx].0 {
                                pidx -= 1;
                            }
                            *slot = pidx;
                        }

                        let mut segments: Vec<SegmentExtent> = Vec::new();
                        let mut seg_start = 0usize;
                        while seg_start < n_markers {
                            let seg_plan = marker_plan_idx[seg_start];
                            let mut seg_end = seg_start + 1;
                            while seg_end < n_markers && marker_plan_idx[seg_end] == seg_plan {
                                seg_end += 1;
                            }
                            let seg_plan_start = seg_plan.min(plan_range_end.saturating_sub(1));
                            let seg_plan_end = (seg_plan_start + 1).min(plan_range_end);
                            segments.push(SegmentExtent::new(
                                seg_start,
                                seg_end,
                                seg_plan_start,
                                seg_plan_end,
                                n_markers,
                            ));
                            seg_start = seg_end;
                        }

                        let mut chained_priors = priors.cloned();
                        let mut out_posts: Vec<AllelePosteriors> =
                            Vec::with_capacity(output_markers);
                        let mut subsetted_any = false;

                        // Pre-compute nearest-observed-marker retain over the
                        // full I/O window so per-segment HMM smoothing can see
                        // typed anchors in adjacent segments.
                        let full_window_retain: Vec<f32> = {
                            let mut tmp_ws = ImputeWorkspace::new(1, n_markers);
                            compute_nearest_observed_lambda(
                                &mut tmp_ws,
                                input_probs,
                                &p_recomb,
                                smoothing_cluster_cm,
                            );
                            tmp_ws.nearest_obs_retain[..n_markers].to_vec()
                        };

                        for extent in &segments {
                            if extent.core_len() == 0 {
                                continue;
                            }
                            let (seg_plan_start, seg_plan_end) = extent.plan_range();
                            let mut state_haps = build_state_haps(
                                hap_idx,
                                chained_priors.as_ref(),
                                donors,
                                informative_ratio,
                                Some((seg_plan_start, seg_plan_end)),
                            );
                            let seg_input_probs = extent.build_target_probs(input_probs);
                            let seg_ref_columns = extent.slice_ref_columns(ref_columns);
                            project_states_min_replacements(
                                &mut state_haps,
                                &seg_input_probs,
                                seg_ref_columns,
                            );
                            if state_haps.is_empty() {
                                return Err(ReagleError::vcf(format!(
                                    "Piecewise state selection produced empty subset: window={} sample={} hap={} segment=[{}..{})",
                                    window_idx,
                                    s,
                                    hap_idx.as_usize(),
                                    extent.core_start,
                                    extent.core_end
                                )));
                            }
                            if state_haps.len() < plan.n_ref_haps {
                                subsetted_any = true;
                            }
                            let transition_lambda = compute_transition_lambda(&state_haps, donors);

                            let mut seg_state_priors: Option<Vec<f32>> = None;
                            let mut seg_boundary_mapped = false;
                            let mut seg_had_nonempty_prior = false;
                            if let Some(p) = chained_priors.as_ref() {
                                if !p.is_empty() {
                                    seg_had_nonempty_prior = true;
                                    let has_entry_pi =
                                        build_entry_pi(&state_haps, donors, &mut mapped_entry_pi_buf);
                                    let entry_pi = if has_entry_pi {
                                        Some(mapped_entry_pi_buf.as_slice())
                                    } else {
                                        None
                                    };
                                    prev_states_buf.clear();
                                    prev_states_buf.reserve(p.ids().len());
                                    for id in p.ids() {
                                        prev_states_buf.push(RefHapId::new(id.0));
                                    }
                                    let recomb_boundary = extent.boundary_recomb(&p_recomb);
                                    let mapper = TransitionMatrix::build(
                                        &prev_states_buf,
                                        &state_haps,
                                        recomb_boundary,
                                        n_transition_haps,
                                        transition_lambda,
                                    );
                                    mapper.map_into_with_pi(
                                        p.probs(),
                                        entry_pi,
                                        &mut mapped_priors_buf,
                                    );
                                    let mut sum = 0.0f32;
                                    for v in mapped_priors_buf.iter() {
                                        if v.is_finite() && *v > 0.0 {
                                            sum += *v;
                                        }
                                    }
                                    if sum > 0.0 {
                                        let inv = 1.0 / sum;
                                        for v in mapped_priors_buf.iter_mut() {
                                            *v = if v.is_finite() && *v > 0.0 {
                                                *v * inv
                                            } else {
                                                0.0
                                            };
                                        }
                                        seg_state_priors = Some(mapped_priors_buf.clone());
                                        seg_boundary_mapped = true;
                                    } else {
                                        return Err(ReagleError::vcf(format!(
                                            "Piecewise boundary mapping collapsed: window={} sample={} hap={} segment=[{}..{}) next_states={}",
                                            window_idx,
                                            s,
                                            hap_idx.as_usize(),
                                            extent.core_start,
                                            extent.core_end,
                                            state_haps.len()
                                        )));
                                    }
                                }
                            }
                            if seg_had_nonempty_prior && seg_state_priors.is_none() {
                                return Err(ReagleError::vcf(format!(
                                    "Piecewise boundary mapping missing prior mass: window={} sample={} hap={} segment=[{}..{})",
                                    window_idx,
                                    s,
                                    hap_idx.as_usize(),
                                    extent.core_start,
                                    extent.core_end
                                )));
                            }

                            let seg_p_recomb = extent.build_p_recomb(&p_recomb, seg_boundary_mapped);
                            let hmm_len = extent.hmm_len();
                            let (seg_posteriors, seg_state_post) = LOCAL_WORKSPACE.with(|cell| {
                                let mut ws_opt = cell.borrow_mut();
                                *ws_opt = Some(ImputeWorkspace::new(state_haps.len(), hmm_len));
                                let ws = ws_opt.as_mut().unwrap();
                                let effective_error_rate =
                                    calibrated_emission_error(&seg_input_probs, error_rate);
                                run_impute_hmm(
                                    &state_haps,
                                    seg_ref_columns,
                                    &seg_input_probs,
                                    &seg_p_recomb,
                                    effective_error_rate,
                                    Some(extent.handoff_hmm_idx()),
                                    seg_state_priors.as_deref(),
                                    &ref_allele_freqs,
                                    n_transition_haps,
                                    transition_lambda,
                                    ImputeHmmContext {
                                        window_idx,
                                        sample_idx: s,
                                        hap_idx: hap_idx.as_usize(),
                                    },
                                    smoothing_cluster_cm,
                                    Some(extent.slice_retain(&full_window_retain)),
                                    ws,
                                )
                            })?;

                            out_posts.extend(extent.extract_output_posteriors(
                                &seg_posteriors,
                                output_start,
                                output_end,
                            ));

                            let mut next_priors_local = HaplotypePriors::empty();
                            if let Some(state_post) = seg_state_post.as_ref() {
                                if state_post.len() != state_haps.len() {
                                    return Err(ReagleError::vcf(format!(
                                        "Piecewise state posterior/state mismatch: window={} sample={} hap={} segment=[{}..{}) post_len={} states={}",
                                        window_idx,
                                        s,
                                        hap_idx.as_usize(),
                                        extent.core_start,
                                        extent.core_end,
                                        state_post.len(),
                                        state_haps.len()
                                    )));
                                }
                                let pairs = state_posteriors_to_priors(&state_haps, state_post, 0.0);
                                if !pairs.is_empty() {
                                    let (ids, probs): (Vec<GlobalHapId>, Vec<f32>) = pairs
                                        .into_iter()
                                        .map(|(g, p)| (GlobalHapId(g.as_u32()), p))
                                        .unzip();
                                    next_priors_local = HaplotypePriors::new(ids, probs);
                                }
                            }
                            chained_priors = if next_priors_local.is_empty() {
                                None
                            } else {
                                Some(next_priors_local)
                            };
                        }

                        if out_posts.len() != output_markers {
                            return Err(ReagleError::vcf(format!(
                                "Piecewise posterior length mismatch: window={} sample={} hap={} got={} expected={}",
                                window_idx,
                                s,
                                hap_idx.as_usize(),
                                out_posts.len(),
                                output_markers
                            )));
                        }
                        if self.config.err.is_none() {
                            for (out_idx, ref_m) in (output_start..output_end).enumerate() {
                                if !input_probs.is_observed_marker(ref_m) {
                                    continue;
                                }
                                let probs = input_probs.probs_for_marker(ref_m);
                                if probs.is_empty() {
                                    continue;
                                }
                                out_posts[out_idx] = if probs.len() == 2 {
                                    AllelePosteriors::Biallelic(
                                        probs.get(1).copied().unwrap_or(0.0),
                                    )
                                } else {
                                    AllelePosteriors::Multiallelic(std::sync::Arc::<[f32]>::from(
                                        probs.to_vec(),
                                    ))
                                };
                            }
                        }
                        let next_priors = chained_priors.unwrap_or_else(HaplotypePriors::empty);
                        return Ok((out_posts, next_priors, subsetted_any, informative_ratio));
                    }
                    let mut state_haps =
                        build_state_haps(hap_idx, priors, donors, informative_ratio, None);
                    project_states_min_replacements(&mut state_haps, input_probs, ref_columns);
                    if state_haps.is_empty() {
                        return Err(ReagleError::vcf(format!(
                            "State selection produced empty subset: window={} sample={} hap={} donors={} has_priors={}",
                            window_idx,
                            s,
                            hap_idx.as_usize(),
                            donors.len(),
                            priors.is_some()
                        )));
                    }
                    let transition_lambda = compute_transition_lambda(&state_haps, donors);

                    let mut state_priors_slice: Option<&[f32]> = if let Some(p) = priors {
                        if p.is_empty() {
                            if !warned_no_priors {
                                warn!(
                                    "Handoff priors missing for window {} (no markers or no posterior)",
                                    window_idx
                                );
                                warned_no_priors = true;
                            }
                            None
                        } else {
                            let has_entry_pi =
                                build_entry_pi(&state_haps, donors, &mut mapped_entry_pi_buf);
                            let entry_pi = if has_entry_pi {
                                Some(mapped_entry_pi_buf.as_slice())
                            } else {
                                None
                            };
                            prev_states_buf.clear();
                            prev_states_buf.reserve(p.ids().len());
                            for id in p.ids() {
                                prev_states_buf.push(RefHapId::new(id.0));
                            }
                            let mapper = TransitionMatrix::build(
                                &prev_states_buf,
                                &state_haps,
                                handoff_recomb_rate,
                                n_transition_haps,
                                transition_lambda,
                            );
                            mapper.map_into_with_pi(
                                p.probs(),
                                entry_pi,
                                &mut mapped_priors_buf,
                            );
                            let mut sum = 0.0f32;
                            for v in mapped_priors_buf.iter() {
                                if v.is_finite() && *v > 0.0 {
                                    sum += *v;
                                }
                            }
                            if sum <= 0.0 {
                                if !warned_empty_map {
                                    warn!(
                                        "State handoff mapped to empty priors for window {} (state set mismatch)",
                                        window_idx
                                    );
                                    warned_empty_map = true;
                                }
                                return Err(ReagleError::vcf(format!(
                                    "Mapped handoff priors collapsed: window={} sample={} hap={} state_count={}",
                                    window_idx,
                                    s,
                                    hap_idx.as_usize(),
                                    state_haps.len()
                                )));
                            }
                            let inv = 1.0 / sum;
                            for v in mapped_priors_buf.iter_mut() {
                                *v = if v.is_finite() && *v > 0.0 {
                                    *v * inv
                                } else {
                                    0.0
                                };
                            }
                            Some(mapped_priors_buf.as_slice())
                        }
                    } else {
                        // No handoff prior: start from the exact Li-Stephens uniform
                        // initial distribution over active states. A donor-count
                        // initialized prior can over-constrain sparse-marker windows
                        // and collapse rare-allele posteriors.
                        None
                    };

                    // WARNING: Do NOT add a second-pass HMM re-run here that feeds
                    // first-pass posteriors back as priors. This creates circular
                    // self-reinforcement: errors in pass 1 get amplified in pass 2,
                    // causing posterior collapse on rare alleles. Tightening the
                    // emission error for the second pass makes it worse by increasing
                    // confidence in wrong states. Tested in PR #755: R² -0.009,
                    // SEN -0.0018 — catastrophic accuracy regression.
                    let (mut posteriors, mut state_post_opt) = LOCAL_WORKSPACE.with(|cell| {
                        let mut ws_opt = cell.borrow_mut();
                        if ws_opt.is_none() {
                            *ws_opt = Some(ImputeWorkspace::new(state_haps.len(), n_ref_markers));
                        }
                        let ws = ws_opt.as_mut().unwrap();
                        let effective_error_rate = calibrated_emission_error(input_probs, error_rate);
                        run_impute_hmm(
                            &state_haps,
                            ref_columns,
                            input_probs,
                            &p_recomb,
                            effective_error_rate,
                            prior_marker_idx,
                            state_priors_slice.take(),
                            &ref_allele_freqs,
                            n_transition_haps,
                            transition_lambda,
                            ImputeHmmContext {
                                window_idx,
                                sample_idx: s,
                                hap_idx: hap_idx.as_usize(),
                            },
                            smoothing_cluster_cm,
                            None,
                            ws,
                        )
                    })?;

                    let donor_pool = if donor_pool_full.is_empty() {
                        donors
                    } else {
                        donor_pool_full
                    };
                    if donor_pool.len() > donors.len() && !posteriors.is_empty() {
                        let win = ADAPTIVE_REFINE_WINDOW_MARKERS.min(posteriors.len()).max(1);
                        let step = ADAPTIVE_REFINE_STEP_MARKERS.min(win).max(1);
                        let mut needs_refine = false;
                        let mut w_start = 0usize;
                        while w_start < posteriors.len() {
                            let w_end = (w_start + win).min(posteriors.len());
                            let u = uncertainty_score_window(&posteriors, input_probs, output_start, w_start, w_end);
                            if u > ADAPTIVE_REFINE_U_THRESHOLD {
                                needs_refine = true;
                                break;
                            }
                            w_start = w_start.saturating_add(step);
                        }

                        if needs_refine {
                            let mut l = donors.len().max(16);
                            while l < donor_pool.len() {
                                let next_l = (l * 2).min(128).min(donor_pool.len());
                                if next_l <= l {
                                    break;
                                }
                                let mut expanded = donor_pool.to_vec();
                                keep_top_k_donors_by_weight(&mut expanded, next_l);
                                let mut refined_states =
                                    build_state_haps(hap_idx, priors, &expanded, informative_ratio, None);
                                project_states_min_replacements(&mut refined_states, input_probs, ref_columns);
                                if refined_states.is_empty() {
                                    break;
                                }
                                let transition_lambda_refined =
                                    compute_transition_lambda(&refined_states, &expanded);
                                let (candidate_posts, candidate_state_post) = LOCAL_WORKSPACE.with(|cell| {
                                    let mut ws_opt = cell.borrow_mut();
                                    *ws_opt = Some(ImputeWorkspace::new(refined_states.len(), n_ref_markers));
                                    let ws = ws_opt.as_mut().unwrap();
                                    let effective_error_rate =
                                        calibrated_emission_error(input_probs, error_rate);
                                    run_impute_hmm(
                                        &refined_states,
                                        ref_columns,
                                        input_probs,
                                        &p_recomb,
                                        effective_error_rate,
                                        prior_marker_idx,
                                        None,
                                        &ref_allele_freqs,
                                        n_transition_haps,
                                        transition_lambda_refined,
                                        ImputeHmmContext {
                                            window_idx,
                                            sample_idx: s,
                                            hap_idx: hap_idx.as_usize(),
                                        },
                                        smoothing_cluster_cm,
                                        None,
                                        ws,
                                    )
                                })?;
                                let (max_delta, mean_kl) = posterior_delta_and_kl(&posteriors, &candidate_posts);
                                posteriors = candidate_posts;
                                state_post_opt = candidate_state_post;
                                state_haps = refined_states;
                                l = next_l;
                                if max_delta < ADAPTIVE_REFINE_MAX_DOSAGE_DELTA
                                    || mean_kl < ADAPTIVE_REFINE_MAX_KL
                                {
                                    break;
                                }
                            }
                        }
                    }

                    if self.config.err.is_none() {
                        // Preserve direct target evidence at observed markers. Imputation
                        // should not overwrite measured genotype probabilities at typed
                        // sites, otherwise dosage correlation is artificially degraded.
                        for (out_idx, ref_m) in (output_start..output_end).enumerate() {
                            if !input_probs.is_observed_marker(ref_m) {
                                continue;
                            }
                            let probs = input_probs.probs_for_marker(ref_m);
                            if probs.is_empty() {
                                continue;
                            }
                            posteriors[out_idx] = if probs.len() == 2 {
                                AllelePosteriors::Biallelic(probs.get(1).copied().unwrap_or(0.0))
                            } else {
                                AllelePosteriors::Multiallelic(std::sync::Arc::<[f32]>::from(
                                    probs.to_vec(),
                                ))
                            };
                        }
                    }

                    let mut next_priors = HaplotypePriors::empty();
                    if let Some(state_post) = state_post_opt.as_ref() {
                        let pairs = state_posteriors_to_priors(&state_haps, state_post, 0.0);
                        if !pairs.is_empty() {
                            let (ids, probs): (Vec<GlobalHapId>, Vec<f32>) = pairs
                                .into_iter()
                                .map(|(g, p)| (GlobalHapId(g.as_u32()), p))
                                .unzip();
                            next_priors = HaplotypePriors::new(ids, probs);
                        }
                    }

                    let subsetted_states = state_haps.len() < plan.n_ref_haps;
                    Ok((posteriors, next_priors, subsetted_states, informative_ratio))
                };

                let mut hap1_posts: Option<Vec<AllelePosteriors>> = None;
                let mut hap2_posts: Option<Vec<AllelePosteriors>> = None;
                let mut p1_out = HaplotypePriors::empty();
                let mut p2_out = HaplotypePriors::empty();

                if no_info_h1 && has_priors_h1 {
                    if let Some(p) = priors_h1 {
                        let (posts, propagated) = exact_transition_only_from_priors(
                            h1_idx,
                            p,
                            &donors_h1,
                            handoff_capture_idx_h1,
                            &mut posts_probs_buf,
                        )?;
                        hap1_posts = Some(posts);
                        p1_out = propagated;
                    }
                } else if use_hmm_h1 {
                    let (posts, out, subsetted_states, informative_ratio) = process_haplotype(
                        h1_idx,
                        priors_h1,
                        &mut input_probs_h1,
                        prior_error_rate.clamp(1e-6, 0.5),
                        handoff_capture_idx_h1,
                        &donors_h1,
                        &full_donors_h1,
                    )?;
                    let _ = (subsetted_states, informative_ratio);
                    hap1_posts = Some(posts);
                    p1_out = out;
                } else if has_priors_h1 {
                    if let Some(p) = priors_h1 {
                        let (posts, propagated) = exact_transition_only_from_priors(
                            h1_idx,
                            p,
                            &donors_h1,
                            handoff_capture_idx_h1,
                            &mut posts_probs_buf,
                        )?;
                        hap1_posts = Some(posts);
                        p1_out = propagated;
                    }
                } else if no_info_h1 {
                    let (posts, priors) =
                        exact_no_info_posteriors(h1_idx, &donors_h1, &mut posts_probs_buf)?;
                    hap1_posts = Some(posts);
                    p1_out = priors;
                    dbg_fallback_selected_priors.fetch_add(1, Ordering::Relaxed);
                } else {
                    let total: f32 = donors_h1
                        .iter()
                        .map(|(_, c)| if c.is_finite() && *c > 0.0 { *c } else { 0.0 })
                        .sum();
                    if total > 0.0 {
                        let (ids, probs): (Vec<GlobalHapId>, Vec<f32>) = donors_h1
                            .iter()
                            .map(|(h, c)| (GlobalHapId(h.as_u32()), *c / total))
                            .unzip();
                        p1_out = HaplotypePriors::new(ids, probs);
                        hap1_posts = Some(posts_from_donors(&donors_h1, &mut posts_probs_buf)?);
                    } else {
                        return Err(ReagleError::vcf(format!(
                            "No subset priors or donors for haplotype: window={} sample={} hap={}",
                            window_idx,
                            s,
                            h1_idx.as_usize()
                        )));
                    }
                }

                if no_info_h2 && has_priors_h2 {
                    if let Some(p) = priors_h2 {
                        let (posts, propagated) = exact_transition_only_from_priors(
                            h2_idx,
                            p,
                            &donors_h2,
                            handoff_capture_idx_h2,
                            &mut posts_probs_buf,
                        )?;
                        hap2_posts = Some(posts);
                        p2_out = propagated;
                    }
                } else if use_hmm_h2 {
                    let (posts, out, subsetted_states, informative_ratio) = process_haplotype(
                        h2_idx,
                        priors_h2,
                        &mut input_probs_h2,
                        prior_error_rate.clamp(1e-6, 0.5),
                        handoff_capture_idx_h2,
                        &donors_h2,
                        &full_donors_h2,
                    )?;
                    let _ = (subsetted_states, informative_ratio);
                    hap2_posts = Some(posts);
                    p2_out = out;
                } else if has_priors_h2 {
                    if let Some(p) = priors_h2 {
                        let (posts, propagated) = exact_transition_only_from_priors(
                            h2_idx,
                            p,
                            &donors_h2,
                            handoff_capture_idx_h2,
                            &mut posts_probs_buf,
                        )?;
                        hap2_posts = Some(posts);
                        p2_out = propagated;
                    }
                } else if no_info_h2 {
                    let (posts, priors) =
                        exact_no_info_posteriors(h2_idx, &donors_h2, &mut posts_probs_buf)?;
                    hap2_posts = Some(posts);
                    p2_out = priors;
                    dbg_fallback_selected_priors.fetch_add(1, Ordering::Relaxed);
                } else {
                    let total: f32 = donors_h2
                        .iter()
                        .map(|(_, c)| if c.is_finite() && *c > 0.0 { *c } else { 0.0 })
                        .sum();
                    if total > 0.0 {
                        let (ids, probs): (Vec<GlobalHapId>, Vec<f32>) = donors_h2
                            .iter()
                            .map(|(h, c)| (GlobalHapId(h.as_u32()), *c / total))
                            .unzip();
                        p2_out = HaplotypePriors::new(ids, probs);
                        hap2_posts = Some(posts_from_donors(&donors_h2, &mut posts_probs_buf)?);
                    } else {
                        return Err(ReagleError::vcf(format!(
                            "No subset priors or donors for haplotype: window={} sample={} hap={}",
                            window_idx,
                            s,
                            h2_idx.as_usize()
                        )));
                    }
                }

                if let Some(bb) = telemetry.as_ref() {
                    bb.add_samples(1);
                }

                let normalize_posteriors_to_output =
                    |posts: &mut Option<Vec<AllelePosteriors>>, hap_idx: HapIdx| -> Result<()> {
                        let Some(values) = posts.take() else {
                            return Ok(());
                        };
                        let len = values.len();
                        if len == output_markers {
                            *posts = Some(values);
                            return Ok(());
                        }
                        if output_end <= len && output_start <= output_end {
                            let trimmed: Vec<AllelePosteriors> = values
                                .into_iter()
                                .skip(output_start)
                                .take(output_markers)
                                .collect();
                            *posts = Some(trimmed);
                            return Ok(());
                        }
                        Err(ReagleError::vcf(format!(
                            "Posterior length incompatible with output span: window={} sample={} hap={} len={} output_start={} output_end={} output_markers={}",
                            window_idx,
                            s,
                            hap_idx.as_usize(),
                            len,
                            output_start,
                            output_end,
                            output_markers
                        )))
                    };
                normalize_posteriors_to_output(&mut hap1_posts, h1_idx)?;
                normalize_posteriors_to_output(&mut hap2_posts, h2_idx)?;

                let need_sm_h1 = hap1_posts.is_none();
                let need_sm_h2 = hap2_posts.is_none();
                if need_sm_h1 {
                    sm_needed[h1_idx.as_usize()].store(true, Ordering::Relaxed);
                }
                if need_sm_h2 {
                    sm_needed[h2_idx.as_usize()].store(true, Ordering::Relaxed);
                }

                let (hap1_alt_probs, hap1_sparse_posts) = split_hap_posteriors(&mut hap1_posts);
                let (hap2_alt_probs, hap2_sparse_posts) = split_hap_posteriors(&mut hap2_posts);
                let mut hap_alt_probs = (hap1_alt_probs, hap2_alt_probs);
                if let Some(writer) = alt_prob_store_writer.as_ref() {
                    if let Some(values) = hap_alt_probs.0.as_ref() {
                        AltProbDiskStoreBuilder::write_hap_probs_at(
                            writer.as_ref(),
                            n_target_samples,
                            output_markers,
                            s,
                            0,
                            values,
                        )?;
                        hap_alt_probs.0 = None;
                    }
                    if let Some(values) = hap_alt_probs.1.as_ref() {
                        AltProbDiskStoreBuilder::write_hap_probs_at(
                            writer.as_ref(),
                            n_target_samples,
                            output_markers,
                            s,
                            1,
                            values,
                        )?;
                        hap_alt_probs.1 = None;
                    }
                }

                    Ok(ImputeResult {
                        result: SampleImputationResult {
                            sample_idx: s,
                            hap_alt_probs,
                            hap_posteriors: (hap1_sparse_posts, hap2_sparse_posts),
                        },
                        priors: Some((p1_out, p2_out)),
                        last_info_idx: match (handoff_capture_idx_h1, handoff_capture_idx_h2) {
                            (Some(a), Some(b)) => Some(a.max(b)),
                            (Some(a), None) => Some(a),
                            (None, Some(b)) => Some(b),
                            (None, None) => None,
                        },
                    })
                    })();
                    let _ = tx.send(item);
                });
        });

        if should_log {
            eprintln!(
                "    [debug hmm] use_hmm={} no_hmm={} has_priors={} no_info={} insufficient={} low_conf={} few_donors={} fallback_ref_freq={}",
                dbg_use_hmm.load(Ordering::Relaxed),
                dbg_no_hmm.load(Ordering::Relaxed),
                dbg_has_priors.load(Ordering::Relaxed),
                dbg_no_info.load(Ordering::Relaxed),
                dbg_insufficient.load(Ordering::Relaxed),
                dbg_low_conf.load(Ordering::Relaxed),
                dbg_few_donors.load(Ordering::Relaxed),
                dbg_fallback_selected_priors.load(Ordering::Relaxed)
            );
        }

        let output_markers = output_end.saturating_sub(output_start);
        let mut sm_alt_probs_by_hap: Vec<Option<Vec<f32>>> = vec![None; n_target_haps];
        // WARNING: Do NOT expand sm_haps to all target haplotypes (e.g. by
        // adding `|| output_markers > 0`). Computing SM donor traces for every
        // haplotype wastes ~97s per run and the downstream PBWT blending it
        // enables (see dosage warning below) harms accuracy. Only haplotypes
        // that actually need the SM fallback path should be included.
        // Tested in PR #758: R² -0.0012, +97s slower.
        let sm_haps: Vec<usize> = sm_needed
            .iter()
            .enumerate()
            .filter_map(|(i, f)| {
                if f.load(Ordering::Relaxed) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        if !sm_haps.is_empty() && output_markers > 0 {
            let mut pbwt = ReferencePbwt::new(plan.n_ref_haps);
            let mut ref_alleles: Vec<u8> = vec![0u8; plan.n_ref_haps];
            let phase_mask = target_win.phase_mask();
            let batch_size = 4096usize;
            let mut batches: Vec<(
                Vec<usize>,
                Vec<RankBeam>,
                Vec<PbwtQueryAllele>,
                Vec<f32>,
                Vec<u32>,
                Vec<Option<usize>>,
                Vec<(u32, u32, u64)>,
            )> = Vec::new();
            let mut start = 0usize;
            while start < sm_haps.len() {
                let end = (start + batch_size).min(sm_haps.len());
                let haps: Vec<usize> = sm_haps[start..end].to_vec();
                let beams = vec![RankBeam::full(plan.n_ref_haps as u32); haps.len()];
                let query_alleles = vec![PbwtQueryAllele::wildcard(); haps.len()];
                let query_allele_weight = vec![1.0f32; haps.len()];
                let current_donor = vec![0u32; haps.len()];
                let peer_idx_by_hap = build_peer_indices(&haps);
                let scratch = Vec::new();
                for &hap in &haps {
                    sm_alt_probs_by_hap[hap] = Some(Vec::with_capacity(output_markers));
                }
                batches.push((
                    haps,
                    beams,
                    query_alleles,
                    query_allele_weight,
                    current_donor,
                    peer_idx_by_hap,
                    scratch,
                ));
                start = end;
            }

            for ref_m in 0..n_ref_markers {
                let col = &ref_columns[ref_m];
                fill_ref_alleles(col, &mut ref_alleles);
                let n_alleles = ref_markers
                    .marker(MarkerIdx::new(ref_m as u32))
                    .n_alleles()
                    .max(1);

                pbwt.prepare_step(&ref_alleles, n_alleles);
                for (
                    haps,
                    beams,
                    query_alleles,
                    query_allele_weight,
                    _,
                    peer_idx_by_hap,
                    scratch,
                ) in batches.iter_mut()
                {
                    if let Some(resolution) = resolved_ref_targets.get(ref_m).copied().flatten() {
                        let target_idx = resolution.target_idx;
                        let target_marker = MarkerIdx::new(target_idx as u32);

                        let mut cached_sample_idx = usize::MAX;
                        let mut cached_query_pair =
                            [crate::data::storage::AlleleCode::MISSING.raw(); 2];
                        let mut cached_wildcard_weight = 0.0f32;
                        let mut cached_allele_weight = 1.0f32;
                        for (i, &hap_idx) in haps.iter().enumerate() {
                            let sample_idx = hap_idx / 2;
                            let local = hap_idx % 2;
                            if sample_idx != cached_sample_idx {
                                cached_sample_idx = sample_idx;
                                cached_wildcard_weight = 0.0;
                                let sample = sample_idx_from_usize(sample_idx);
                                let h1 = sample.hap(HapSide::H1);
                                let h2 = sample.hap(HapSide::H2);
                                let mut a1 = target_win.allele(target_marker, h1);
                                let mut a2 = target_win.allele(target_marker, h2);
                                if let Some(missing) = target_missing {
                                    if missing.allele(target_marker, h1)
                                        == crate::data::storage::AlleleCode::MISSING.raw()
                                    {
                                        a1 = crate::data::storage::AlleleCode::MISSING.raw();
                                    }
                                    if missing.allele(target_marker, h2)
                                        == crate::data::storage::AlleleCode::MISSING.raw()
                                    {
                                        a2 = crate::data::storage::AlleleCode::MISSING.raw();
                                    }
                                }
                                let mapped1 = map_target_allele_to_ref(alignment, resolution, a1)
                                    .unwrap_or(crate::data::storage::AlleleCode::MISSING.raw());
                                let mapped2 = map_target_allele_to_ref(alignment, resolution, a2)
                                    .unwrap_or(crate::data::storage::AlleleCode::MISSING.raw());
                                let is_het = mapped1
                                    != crate::data::storage::AlleleCode::MISSING.raw()
                                    && mapped2 != crate::data::storage::AlleleCode::MISSING.raw()
                                    && mapped1 != mapped2;
                                let input_phased = phase_mask
                                    .and_then(|mask| mask.get(target_idx, sample_idx))
                                    .map(|v| v != 0)
                                    .unwrap_or(true);
                                if is_het && !input_phased {
                                    cached_query_pair = [
                                        crate::data::storage::AlleleCode::MISSING.raw(),
                                        crate::data::storage::AlleleCode::MISSING.raw(),
                                    ];
                                    cached_allele_weight = 0.0;
                                } else if is_het && input_phased {
                                    let qa1 = PbwtQueryAllele::allele(mapped1)
                                        .unwrap_or_else(PbwtQueryAllele::missing);
                                    let qa2 = PbwtQueryAllele::allele(mapped2)
                                        .unwrap_or_else(PbwtQueryAllele::missing);
                                    let phase_conf = target_win
                                        .sample_phase_confidence_f32(target_marker, sample_idx)
                                        .clamp(0.0, 1.0);
                                    let oriented_pair = if phase_conf < 0.5 {
                                        [qa2, qa1]
                                    } else {
                                        [qa1, qa2]
                                    };
                                    let self_query = oriented_pair[local];
                                    let mut beam_uncertainty = pbwt_beam_uncertainty(
                                        &beams[i],
                                        plan.n_ref_haps,
                                        self_query,
                                    );
                                    let peer_idx = peer_idx_by_hap[i];
                                    if let Some(peer_i) = peer_idx {
                                        let peer_local = haps[peer_i] % 2;
                                        let peer_query = oriented_pair[peer_local];
                                        let peer_uncertainty = pbwt_beam_uncertainty(
                                            &beams[peer_i],
                                            plan.n_ref_haps,
                                            peer_query,
                                        );
                                        beam_uncertainty =
                                            0.5 * (beam_uncertainty + peer_uncertainty);
                                    }
                                    let geno_conf = target_win
                                        .sample_confidence_f32(target_marker, sample_idx)
                                        .clamp(0.0, 1.0);
                                    let err_limit = phase_query_orientation_error_limit(
                                        geno_conf,
                                        beam_uncertainty,
                                    )
                                    .max(1e-6);
                                    cached_allele_weight =
                                        phase_orientation_weight(phase_conf, err_limit);
                                    if phase_best_orientation_error(phase_conf) > err_limit {
                                        cached_query_pair = [
                                            crate::data::storage::AlleleCode::MISSING.raw(),
                                            crate::data::storage::AlleleCode::MISSING.raw(),
                                        ];
                                        cached_wildcard_weight =
                                            uncertain_orientation_wildcard_info_weight();
                                    } else if phase_conf < 0.5 {
                                        cached_query_pair = [mapped2, mapped1];
                                    } else {
                                        cached_query_pair = [mapped1, mapped2];
                                    }
                                } else {
                                    cached_query_pair = [mapped1, mapped2];
                                    cached_allele_weight = 1.0;
                                }
                            }
                            query_alleles[i] = PbwtQueryAllele::allele(cached_query_pair[local])
                                .unwrap_or_else(PbwtQueryAllele::wildcard);
                            query_allele_weight[i] = if query_alleles[i].is_wildcard() {
                                cached_wildcard_weight
                            } else {
                                cached_allele_weight
                            };
                        }
                    } else {
                        for (qa, qw) in query_alleles.iter_mut().zip(query_allele_weight.iter_mut())
                        {
                            *qa = PbwtQueryAllele::wildcard();
                            *qw = 0.0;
                        }
                    }

                    pbwt.update_beams_with_scratch_query(
                        beams,
                        query_alleles,
                        None,
                        n_alleles,
                        scratch,
                    );
                }
                pbwt.finalize_step(&ref_alleles, n_alleles, ref_m);

                if ref_m < output_start || ref_m >= output_end {
                    continue;
                }
                for (haps, beams, query_alleles, query_allele_weight, current_donor, _, _) in
                    batches.iter_mut()
                {
                    let mut donor_candidates: Vec<u32> =
                        Vec::with_capacity(SM_MATCH_DONORS.saturating_mul(2));
                    for (i, &hap_idx) in haps.iter().enumerate() {
                        let beam = &beams[i];
                        let donor_k = adaptive_sm_donor_k(beam, plan.n_ref_haps, query_alleles[i]);
                        donor_candidates.clear();
                        pbwt.select_donors_into(beam, donor_k, &mut donor_candidates);
                        let mut donor = current_donor[i];
                        let mut found = false;
                        for &cand in &donor_candidates {
                            if cand == donor {
                                found = true;
                                break;
                            }
                        }
                        if !found {
                            if let Some(&cand) = donor_candidates.first() {
                                donor = cand;
                                current_donor[i] = donor;
                            }
                        }
                        let target_allele = query_alleles
                            .get(i)
                            .and_then(|qa| qa.as_allele())
                            .unwrap_or(crate::data::storage::AlleleCode::MISSING.raw());
                        let orient_weight = query_allele_weight[i].clamp(0.0, 1.0);
                        let p_alt =
                            if target_allele == crate::data::storage::AlleleCode::MISSING.raw() {
                                let donor_alt = if donor_candidates.is_empty() {
                                    0.5
                                } else {
                                    let mut alt_sum = 0u32;
                                    for &cand in &donor_candidates {
                                        let allele = col.get(HapIdx::new(cand));
                                        if allele == 1 {
                                            alt_sum += 1;
                                        }
                                    }
                                    (alt_sum as f32 / donor_candidates.len() as f32).clamp(0.0, 1.0)
                                };
                                (orient_weight * donor_alt + (1.0 - orient_weight) * 0.5)
                                    .clamp(1e-6, 1.0 - 1e-6)
                            } else {
                                let hard = if donor_candidates.is_empty() {
                                    if target_allele == 1 { 1.0 } else { 0.0 }
                                } else {
                                    let allele = col.get(HapIdx::new(donor));
                                    if allele == 1 { 1.0 } else { 0.0 }
                                };
                                (orient_weight * hard + (1.0 - orient_weight) * 0.5)
                                    .clamp(1e-6, 1.0 - 1e-6)
                            };
                        if let Some(buf) = sm_alt_probs_by_hap[hap_idx].as_mut() {
                            buf.push(p_alt);
                        }
                    }
                }
            }
        }

        let mut all_results = Vec::with_capacity(n_target_samples);
        let mut next_priors_id_vec = vec![HaplotypePriors::empty(); n_target_samples * 2];
        let mut next_orientation_weight_swap = vec![0.5f32; n_target_samples];
        let mut handoff_marker_idx: Option<usize> = None;

        let phase_mask = target_win.phase_mask();
        let mut scratch_emit0: Vec<f64> = vec![0.0; output_markers];
        let mut scratch_emit1: Vec<f64> = vec![0.0; output_markers];
        let mut scratch_log_stay: Vec<f64> = vec![0.0; output_markers];
        let mut scratch_log_flip: Vec<f64> = vec![f64::NEG_INFINITY; output_markers];
        let mut scratch_fwd0: Vec<f64> = vec![0.0; output_markers];
        let mut scratch_fwd1: Vec<f64> = vec![0.0; output_markers];
        let mut scratch_bwd0: Vec<f64> = vec![0.0; output_markers];
        let mut scratch_bwd1: Vec<f64> = vec![0.0; output_markers];
        let mut scratch_swapped: Vec<bool> = vec![false; output_markers];
        let get_result_prob = |result: &SampleImputationResult,
                               hap: usize,
                               local_m: usize,
                               allele: u8|
         -> Option<f32> { result.hap_prob(hap, local_m, allele) };
        let get_anchor_obs = |sample_idx: usize, ref_m: usize| -> Option<(u8, u8, f32)> {
            let resolution = resolved_ref_targets.get(ref_m).copied().flatten()?;
            let target_m = MarkerIdx::new(resolution.target_idx as u32);
            let target_idx = resolution.target_idx;
            let input_phased = phase_mask
                .and_then(|mask| mask.get(target_idx, sample_idx))
                .map(|v| v != 0)
                .unwrap_or(true);
            if !input_phased {
                return None;
            }
            let sample = sample_idx_from_usize(sample_idx);
            let h1 = sample.hap(HapSide::H1);
            let h2 = sample.hap(HapSide::H2);
            if let Some(missing) = target_missing {
                if missing.allele(target_m, h1) == crate::data::storage::AlleleCode::MISSING.raw()
                    || missing.allele(target_m, h2)
                        == crate::data::storage::AlleleCode::MISSING.raw()
                {
                    return None;
                }
            }
            let raw_a1 = target_win.allele(target_m, h1);
            let raw_a2 = target_win.allele(target_m, h2);
            if raw_a1 == crate::data::storage::AlleleCode::MISSING.raw()
                || raw_a2 == crate::data::storage::AlleleCode::MISSING.raw()
            {
                return None;
            }
            let mut a1 = map_target_allele_to_ref(alignment, resolution, raw_a1)
                .unwrap_or(crate::data::storage::AlleleCode::MISSING.raw());
            let mut a2 = map_target_allele_to_ref(alignment, resolution, raw_a2)
                .unwrap_or(crate::data::storage::AlleleCode::MISSING.raw());
            if a1 == crate::data::storage::AlleleCode::MISSING.raw()
                || a2 == crate::data::storage::AlleleCode::MISSING.raw()
                || a1 == a2
            {
                return None;
            }
            let mut conf = if phase_conf_valid {
                target_win
                    .sample_phase_confidence_f32(target_m, sample_idx)
                    .clamp(0.0, 1.0)
            } else {
                0.5
            };
            if conf < 0.5 {
                std::mem::swap(&mut a1, &mut a2);
                conf = 1.0 - conf;
            }
            Some((a1, a2, conf.clamp(0.5, 1.0)))
        };
        let mut smooth_sample_orientation = |result: &mut SampleImputationResult,
                                             capture_ref_idx: Option<usize>,
                                             prior_swap_prob: f32|
         -> Option<(f32, f64, f64, f64, usize)> {
            let n = output_end.saturating_sub(output_start);
            if n == 0 {
                return None;
            }
            let emit0 = &mut scratch_emit0;
            let emit1 = &mut scratch_emit1;
            let log_stay = &mut scratch_log_stay;
            let log_flip = &mut scratch_log_flip;
            let fwd0 = &mut scratch_fwd0;
            let fwd1 = &mut scratch_fwd1;
            let bwd0 = &mut scratch_bwd0;
            let bwd1 = &mut scratch_bwd1;
            let swapped = &mut scratch_swapped;
            emit0.resize(n, 0.0);
            emit1.resize(n, 0.0);
            log_stay.resize(n, 0.0);
            log_flip.resize(n, f64::NEG_INFINITY);
            fwd0.resize(n, 0.0);
            fwd1.resize(n, 0.0);
            bwd0.resize(n, 0.0);
            bwd1.resize(n, 0.0);
            swapped.resize(n, false);

            let eps = 1e-30f64;

            for local_m in 0..n {
                let ref_m = output_start + local_m;
                let (e0, e1) = if let Some((a_left, a_right, phase_conf)) =
                    get_anchor_obs(result.sample_idx, ref_m)
                {
                    let p1_left = get_result_prob(result, 0, local_m, a_left).unwrap_or(0.5);
                    let p1_right = get_result_prob(result, 0, local_m, a_right).unwrap_or(0.5);
                    let p2_left = get_result_prob(result, 1, local_m, a_left).unwrap_or(0.5);
                    let p2_right = get_result_prob(result, 1, local_m, a_right).unwrap_or(0.5);
                    let same = if (p1_left as f64).is_finite() && (p2_right as f64).is_finite() {
                        (p1_left * p2_right).clamp(0.0, 1.0)
                    } else {
                        0.5
                    };
                    let swapped_prob =
                        if (p1_right as f64).is_finite() && (p2_left as f64).is_finite() {
                            (p1_right * p2_left).clamp(0.0, 1.0)
                        } else {
                            0.5
                        };
                    let c = phase_conf as f64;
                    let same_mix = (c * same as f64 + (1.0 - c) * swapped_prob as f64)
                        .max(eps)
                        .ln();
                    let swap_mix = (c * swapped_prob as f64 + (1.0 - c) * same as f64)
                        .max(eps)
                        .ln();
                    (same_mix, swap_mix)
                } else {
                    // Without anchors, id/swap orientation is not identifiable from
                    // independent haplotype posteriors; keep emission neutral.
                    (0.0, 0.0)
                };
                emit0[local_m] = e0;
                emit1[local_m] = e1;
            }

            let logsumexp2 = |a: f64, b: f64| -> f64 {
                let m = a.max(b);
                if !m.is_finite() {
                    return m;
                }
                m + ((a - m).exp() + (b - m).exp()).ln()
            };
            let p_swap0 = (prior_swap_prob as f64).clamp(1e-6, 1.0 - 1e-6);
            let p_keep0 = (1.0 - p_swap0).clamp(1e-6, 1.0 - 1e-6);
            let mut run_orientation_fb = |eta: f64| -> Option<(f64, f64)> {
                let eta_clamped = eta.clamp(ORIENTATION_ETA_MIN, ORIENTATION_ETA_MAX);
                log_stay[0] = 0.0;
                log_flip[0] = f64::NEG_INFINITY;
                for local_m in 1..n {
                    log_stay[local_m] = (1.0 - eta_clamped).ln();
                    log_flip[local_m] = eta_clamped.ln();
                }
                fwd0[0] = p_keep0.ln() + emit0[0];
                fwd1[0] = p_swap0.ln() + emit1[0];
                for local_m in 1..n {
                    let prev0 = fwd0[local_m - 1];
                    let prev1 = fwd1[local_m - 1];
                    fwd0[local_m] = emit0[local_m]
                        + logsumexp2(prev0 + log_stay[local_m], prev1 + log_flip[local_m]);
                    fwd1[local_m] = emit1[local_m]
                        + logsumexp2(prev1 + log_stay[local_m], prev0 + log_flip[local_m]);
                }

                bwd0[n - 1] = 0.0;
                bwd1[n - 1] = 0.0;
                for local_m in (0..n - 1).rev() {
                    let next = local_m + 1;
                    bwd0[local_m] = logsumexp2(
                        log_stay[next] + emit0[next] + bwd0[next],
                        log_flip[next] + emit1[next] + bwd1[next],
                    );
                    bwd1[local_m] = logsumexp2(
                        log_flip[next] + emit0[next] + bwd0[next],
                        log_stay[next] + emit1[next] + bwd1[next],
                    );
                }

                let log_z = logsumexp2(fwd0[n - 1], fwd1[n - 1]);
                if !log_z.is_finite() {
                    return None;
                }
                let mut expected_flips = 0.0f64;
                for local_m in 1..n {
                    let from0_to1 =
                        fwd0[local_m - 1] + log_flip[local_m] + emit1[local_m] + bwd1[local_m];
                    let from1_to0 =
                        fwd1[local_m - 1] + log_flip[local_m] + emit0[local_m] + bwd0[local_m];
                    let log_p_flip = logsumexp2(from0_to1, from1_to0) - log_z;
                    let p_flip = if log_p_flip.is_finite() {
                        log_p_flip.exp().clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    expected_flips += p_flip;
                }
                Some((log_z, expected_flips))
            };
            let mut eta = orientation_eta_from_expected_flips(0.0, n.saturating_sub(1));
            for _ in 0..ORIENTATION_ETA_EM_ITERS {
                let (_, expected_flips) = run_orientation_fb(eta)?;
                eta = orientation_eta_from_expected_flips(expected_flips, n.saturating_sub(1));
            }
            let (log_z, _) = run_orientation_fb(eta)?;
            if !log_z.is_finite() {
                return Some((0.5, 0.0, 0.0, 0.0, 0));
            }
            for local_m in 0..n {
                let log_p1 = fwd1[local_m] + bwd1[local_m] - log_z;
                let p1 = if log_p1.is_finite() {
                    log_p1.exp().clamp(0.0, 1.0)
                } else {
                    0.5
                };
                let upper = (0.5 + ORIENTATION_DECISION_MARGIN).clamp(0.5, 1.0);
                let lower = (0.5 - ORIENTATION_DECISION_MARGIN).clamp(0.0, 0.5);
                swapped[local_m] = if p1 > upper {
                    true
                } else if p1 < lower {
                    false
                } else if local_m > 0 {
                    swapped[local_m - 1]
                } else {
                    prior_swap_prob > 0.5
                };
            }

            let mut orientation_flip_events = 0usize;
            for local_m in 1..n {
                if swapped[local_m] != swapped[local_m - 1] {
                    orientation_flip_events += 1;
                }
            }
            for (local_m, &is_swapped) in swapped.iter().enumerate() {
                if !is_swapped {
                    continue;
                }
                // Invariant: every hap-ordered field in SampleImputationResult must be
                // swapped consistently if orientation is swapped.
                result.swap_hap_posteriors_at(local_m);
                if let (Some(p1), Some(p2)) = (
                    result.hap_alt_probs.0.as_mut(),
                    result.hap_alt_probs.1.as_mut(),
                ) {
                    if local_m < p1.len() && local_m < p2.len() {
                        std::mem::swap(&mut p1[local_m], &mut p2[local_m]);
                    }
                }
            }

            let handoff_local = capture_ref_idx
                .and_then(|idx| {
                    if idx >= output_start && idx < output_end {
                        Some(idx - output_start)
                    } else {
                        None
                    }
                })
                .unwrap_or(n.saturating_sub(1));
            let log_p1_handoff = fwd1[handoff_local] + bwd1[handoff_local] - log_z;
            let p1_handoff = if log_p1_handoff.is_finite() {
                log_p1_handoff.exp().clamp(0.0, 1.0)
            } else {
                0.5
            };
            let eta_mean = eta;
            let eta_min = eta;
            let eta_max = eta;
            if (p1_handoff - 0.5).abs() < ORIENTATION_HANDOFF_MIN_MARGIN {
                return Some((0.5, eta_mean, eta_min, eta_max, orientation_flip_events));
            }
            Some((
                p1_handoff as f32,
                eta_mean,
                eta_min,
                eta_max,
                orientation_flip_events,
            ))
        };

        let mut buffered_alt_values: u64 = 0;
        let mut buffered_sparse_entries: u64 = 0;
        let mut buffered_sparse_haps: u64 = 0;
        let mut orientation_entropy_sum = 0.0f64;
        let mut orientation_eta_sum = 0.0f64;
        let mut orientation_eta_min = f64::INFINITY;
        let mut orientation_eta_max = 0.0f64;
        let mut orientation_flip_events_total = 0usize;
        let mem_diag_interval = (n_target_samples / 8).clamp(1, 128);
        for _ in 0..n_target_samples {
            let mut item = result_rx.recv().map_err(|e| {
                ReagleError::vcf(format!("Failed to receive sample imputation result: {}", e))
            })??;
            let sample_idx = item.result.sample_idx;
            let sample = sample_idx_from_usize(sample_idx);
            let h1 = sample.hap(HapSide::H1).as_usize();
            let h2 = sample.hap(HapSide::H2).as_usize();
            let need1 = sm_alt_probs_by_hap
                .get(h1)
                .and_then(|v| v.as_ref())
                .is_some();
            let need2 = sm_alt_probs_by_hap
                .get(h2)
                .and_then(|v| v.as_ref())
                .is_some();
            if (need1 || need2) && output_markers > 0 {
                let p1 = sm_alt_probs_by_hap
                    .get_mut(h1)
                    .and_then(|v| v.take())
                    .unwrap_or_default();
                let p2 = sm_alt_probs_by_hap
                    .get_mut(h2)
                    .and_then(|v| v.take())
                    .unwrap_or_default();
                if need1 {
                    item.result.hap_alt_probs.0 = Some(p1);
                }
                if need2 {
                    item.result.hap_alt_probs.1 = Some(p2);
                }
            }
            let capture_ref_idx = item.last_info_idx.or(prior_marker_idx);
            let prior_swap_prob = overlap_orientation_weight_swap
                .and_then(|w| w.get(sample_idx))
                .copied()
                .unwrap_or(0.5);
            let (handoff_swap_prob, eta_mean, eta_min, eta_max, orientation_flip_events) =
                smooth_sample_orientation(&mut item.result, capture_ref_idx, prior_swap_prob)
                    .unwrap_or((0.5, 0.0, 0.0, 0.0, 0));
            let (_, w_swap) =
                orientation_weights_from_posterior_swap(handoff_swap_prob.clamp(0.0, 1.0));
            orientation_eta_sum += eta_mean;
            orientation_eta_min = orientation_eta_min.min(eta_min);
            orientation_eta_max = orientation_eta_max.max(eta_max);
            orientation_flip_events_total =
                orientation_flip_events_total.saturating_add(orientation_flip_events);
            let p0 = (1.0f64 - w_swap as f64).clamp(1e-12, 1.0);
            let p1 = (w_swap as f64).clamp(1e-12, 1.0);
            orientation_entropy_sum += -(p0 * p0.ln() + p1 * p1.ln());

            if let Some(writer) = alt_prob_store_writer.as_ref() {
                if let Some(values) = item.result.hap_alt_probs.0.as_ref() {
                    AltProbDiskStoreBuilder::write_hap_probs_at(
                        writer.as_ref(),
                        n_target_samples,
                        output_markers,
                        sample_idx,
                        0,
                        values,
                    )?;
                    item.result.hap_alt_probs.0 = None;
                }
                if let Some(values) = item.result.hap_alt_probs.1.as_ref() {
                    AltProbDiskStoreBuilder::write_hap_probs_at(
                        writer.as_ref(),
                        n_target_samples,
                        output_markers,
                        sample_idx,
                        1,
                        values,
                    )?;
                    item.result.hap_alt_probs.1 = None;
                }
            }

            if let Some(v) = item.result.hap_alt_probs.0.as_ref() {
                buffered_alt_values = buffered_alt_values.saturating_add(v.len() as u64);
            }
            if let Some(v) = item.result.hap_alt_probs.1.as_ref() {
                buffered_alt_values = buffered_alt_values.saturating_add(v.len() as u64);
            }
            if let Some(v) = item.result.hap_posteriors.0.as_ref() {
                buffered_sparse_entries =
                    buffered_sparse_entries.saturating_add(v.values.len() as u64);
                buffered_sparse_haps = buffered_sparse_haps.saturating_add(1);
            }
            if let Some(v) = item.result.hap_posteriors.1.as_ref() {
                buffered_sparse_entries =
                    buffered_sparse_entries.saturating_add(v.values.len() as u64);
                buffered_sparse_haps = buffered_sparse_haps.saturating_add(1);
            }
            all_results.push(item.result);
            if should_log
                && (all_results.len() == 1
                    || all_results.len() == n_target_samples
                    || all_results.len() % mem_diag_interval == 0)
            {
                let alt_bytes =
                    buffered_alt_values.saturating_mul(std::mem::size_of::<f32>() as u64);
                let sparse_bytes = buffered_sparse_entries.saturating_mul(
                    (std::mem::size_of::<usize>() + std::mem::size_of::<AllelePosteriors>()) as u64,
                );
                let result_struct_bytes = (all_results.len() as u64)
                    .saturating_mul(std::mem::size_of::<SampleImputationResult>() as u64);
                let est_total_mb = (alt_bytes
                    .saturating_add(sparse_bytes)
                    .saturating_add(result_struct_bytes))
                    / (1024 * 1024);
                eprintln!(
                    "    [mem diag] buffered_samples={}/{} alt_mb={} sparse_sites={} sparse_haps={} est_results_mb={} note=\"result buffers only\"",
                    all_results.len(),
                    n_target_samples,
                    alt_bytes / (1024 * 1024),
                    buffered_sparse_entries,
                    buffered_sparse_haps,
                    est_total_mb
                );
            }
            if let Some((p1, p2)) = item.priors {
                let base = sample_idx_from_usize(sample_idx)
                    .hap(HapSide::H1)
                    .as_usize();
                if base + 1 < next_priors_id_vec.len() {
                    next_priors_id_vec[base] = p1.clone();
                    next_priors_id_vec[base + 1] = p2.clone();
                    if sample_idx < next_orientation_weight_swap.len() {
                        next_orientation_weight_swap[sample_idx] = w_swap;
                    }
                }
            }
            if let Some(idx) = item.last_info_idx {
                handoff_marker_idx = Some(match handoff_marker_idx {
                    Some(prev) => prev.max(idx),
                    None => idx,
                });
            }
        }

        all_results.sort_by_key(|result| result.sample_idx);
        let alt_prob_store = alt_prob_store_builder.map(|b| b.finalize()).transpose()?;

        if let Some(bb) = &self.telemetry {
            let output_markers = output_end.saturating_sub(output_start);
            bb.set_stage(crate::utils::telemetry::Stage::WritingOutput);
            bb.set_consumer_stage(crate::utils::telemetry::Stage::WritingOutput);
            bb.set_total_markers(output_markers as u64);
            bb.set_markers_processed(0);
            bb.set_total_samples(target_win.n_samples() as u64);
            bb.set_samples_processed(0);
            bb.set_op(&format!(
                "Writing window {} ({} markers)",
                window_idx, output_markers
            ));
            bb.set_consumer_op(&format!(
                "Writing window {} ({} markers)",
                window_idx, output_markers
            ));
        }
        if should_log && n_target_samples > 0 {
            let mean_eta = orientation_eta_sum / n_target_samples as f64;
            let mean_entropy = orientation_entropy_sum / n_target_samples as f64;
            let eta_min = if orientation_eta_min.is_finite() {
                orientation_eta_min
            } else {
                0.0
            };
            let expected_copy_switches_per_hap = (output_start + 1..output_end)
                .map(|m| p_recomb.get(m).copied().unwrap_or(0.0).clamp(0.0, 1.0) as f64)
                .sum::<f64>();
            eprintln!(
                "    [diag orientation] eta[min/mean/max]={:.6}/{:.6}/{:.6} mean_entropy={:.6} label_flip_events={} expected_copy_switches_per_hap={:.6}",
                eta_min,
                mean_eta,
                orientation_eta_max,
                mean_entropy,
                orientation_flip_events_total,
                expected_copy_switches_per_hap
            );
        }
        let handoff_marker_idx = handoff_marker_idx.or(prior_marker_idx);
        let handoff_global_idx = handoff_marker_idx.map(|idx| idx + global_start);
        let handoff_gen_pos = handoff_marker_idx.and_then(|idx| gen_positions.get(idx).copied());
        Ok(Some(ImputationWindowResults {
            all_results,
            ref_is_biallelic,
            overlap_start_idx: overlap_start,
            alt_prob_store,
            handoff: Some(ImputationHandoff {
                priors_id: next_priors_id_vec,
                orientation_weight_swap: next_orientation_weight_swap,
                prior_global_idx: handoff_global_idx,
                prior_gen_pos: handoff_gen_pos,
            }),
        }))
    }
    fn extract_imputed_overlap_streaming<TargetSpace, RefSpace>(
        &self,
        ref_markers: &crate::data::marker::Markers<RefSpace>,
        target_win: &GenotypeMatrix<Phased, TargetSpace>,
        alignment: &MarkerAlignment<TargetSpace, RefSpace>,
        output_start: usize,
        output_end: usize,
        overlap_start: usize,
        all_results: &[SampleImputationResult],
        alt_prob_store: Option<&AltProbDiskStoreView>,
    ) -> PhasedOverlap {
        let mut start = overlap_start.clamp(output_start, output_end);
        if start >= output_end && output_end > output_start {
            start = output_end - 1;
        }
        let end = output_end;
        let overlap_size = end.saturating_sub(start);
        let n_haps = target_win.n_haplotypes();
        let mut alleles =
            vec![crate::data::storage::AlleleCode::MISSING.raw(); overlap_size * n_haps];
        let n_samples = target_win.n_samples();
        let mut result_by_sample: Vec<Option<&SampleImputationResult>> = vec![None; n_samples];
        for result in all_results {
            if result.sample_idx < n_samples {
                result_by_sample[result.sample_idx] = Some(result);
            }
        }
        let get_result_alt_prob =
            |result: &SampleImputationResult, hap: usize, local_m: usize| -> Option<f32> {
                result.hap_alt_prob(hap, local_m).or_else(|| {
                    alt_prob_store.and_then(|store| store.get(result.sample_idx, hap, local_m))
                })
            };
        let resolved_ref_targets =
            build_ref_typed_marker_resolutions(target_win.markers(), ref_markers, alignment);
        for h in 0..n_haps {
            let sample_idx = h / 2;
            let hap_idx = h % 2;
            for (local_m, ref_m) in (start..end).enumerate() {
                let out_local = ref_m.saturating_sub(output_start);
                if let Some(result) = result_by_sample.get(sample_idx).and_then(|r| *r) {
                    if let Some(ap) = result.hap_posterior(hap_idx, out_local) {
                        let allele = match ap {
                            AllelePosteriors::Biallelic(p_alt) => {
                                if *p_alt >= 0.5 {
                                    1u8
                                } else {
                                    0u8
                                }
                            }
                            AllelePosteriors::Multiallelic(probs) => probs
                                .iter()
                                .enumerate()
                                .max_by(|a, b| {
                                    a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)
                                })
                                .map(|(i, _)| i as u8)
                                .unwrap_or(crate::data::storage::AlleleCode::MISSING.raw()),
                        };
                        alleles[h * overlap_size + local_m] = allele;
                        continue;
                    }
                }
                if let Some(result) = result_by_sample.get(sample_idx).and_then(|r| *r) {
                    let p_alt = get_result_alt_prob(result, hap_idx, out_local);
                    if let Some(p_alt) = p_alt {
                        alleles[h * overlap_size + local_m] = if p_alt >= 0.5 { 1 } else { 0 };
                        continue;
                    }
                }
                if let Some(resolution) = resolved_ref_targets.get(ref_m).copied().flatten() {
                    let target_m = MarkerIdx::new(resolution.target_idx as u32);
                    alleles[h * overlap_size + local_m] =
                        target_win.allele(target_m, HapIdx::new(h as u32));
                }
            }
        }
        PhasedOverlap::new(overlap_size, n_haps, alleles)
    }

    /// Write imputed window results to VCF
    #[allow(clippy::too_many_arguments)]
    fn write_imputed_window_streaming<
        TargetSpace: Sync,
        RefMarkerSpace: Sync,
        TargetMissingState: PhaseState + Sync,
    >(
        &self,
        ref_markers: &crate::data::marker::Markers<RefMarkerSpace>,
        target_win: &GenotypeMatrix<Phased, TargetSpace>,
        target_pl: Option<&GenotypeMatrix<Phased, TargetSpace>>,
        target_missing: Option<&GenotypeMatrix<TargetMissingState, TargetSpace>>,
        alignment: &MarkerAlignment<TargetSpace, RefMarkerSpace>,
        writer: &mut VcfWriter,
        quality: &mut ImputationQuality,
        ref_is_biallelic: &[bool],
        output_start: usize,
        output_end: usize,
        markers_to_process_start: usize,
        all_results: &[SampleImputationResult],
        alt_prob_store: Option<&AltProbDiskStoreView>,
        include_gp: bool,
        include_ap: bool,
        correct_errors: bool,
    ) -> Result<()> {
        let markers_range = output_start..output_end;
        let n_markers = markers_range.len();

        if n_markers == 0 || all_results.is_empty() {
            return Ok(());
        }

        let write_span = if self.config.profile {
            Some(
                info_span!(
                    "io_write_output",
                    markers = n_markers,
                    samples = target_win.n_samples()
                )
                .entered(),
            )
        } else {
            None
        };
        if let Some(span) = &write_span {
            tracing::trace!(id = ?span.id(), "Entered write output span");
        }

        let include_posteriors = include_gp || include_ap;
        let n_samples = target_win.n_samples();
        let mut result_by_sample: Vec<Option<&SampleImputationResult>> = vec![None; n_samples];
        for result in all_results {
            if result.sample_idx < n_samples {
                result_by_sample[result.sample_idx] = Some(result);
            }
        }
        let get_result_alt_prob =
            |result: &SampleImputationResult, hap: usize, local_m: usize| -> Option<f32> {
                result.hap_alt_prob(hap, local_m).or_else(|| {
                    alt_prob_store.and_then(|store| store.get(result.sample_idx, hap, local_m))
                })
            };

        let default_posteriors = |marker_idx: usize| -> (AllelePosteriors, AllelePosteriors) {
            let marker = ref_markers.marker(MarkerIdx::new(marker_idx as u32));
            let n_alleles = 1 + marker.alt_alleles.len();
            if n_alleles == 2 {
                (
                    AllelePosteriors::Biallelic(0.0),
                    AllelePosteriors::Biallelic(0.0),
                )
            } else {
                let zeros = std::sync::Arc::<[f32]>::from(vec![0.0f32; n_alleles]);
                (
                    AllelePosteriors::Multiallelic(zeros.clone()),
                    AllelePosteriors::Multiallelic(zeros),
                )
            }
        };

        let normalize_probs = |probs: &mut [f32]| -> bool {
            let mut sum = 0.0f32;
            for p in probs.iter_mut() {
                if *p < 0.0 {
                    *p = 0.0;
                }
                sum += *p;
            }
            if sum > 0.0 {
                for p in probs.iter_mut() {
                    *p /= sum;
                }
                true
            } else {
                false
            }
        };

        let samples = target_win.samples_arc();
        let target_pl = target_pl.unwrap_or(target_win);
        let pos_fallback_used = std::sync::atomic::AtomicUsize::new(0);
        let pos_fallback_map_fail = std::sync::atomic::AtomicUsize::new(0);
        let resolved_ref_targets =
            build_ref_typed_marker_resolutions(target_win.markers(), ref_markers, alignment);
        let marker_is_imputed: Vec<bool> =
            quality.marker_stats.iter().map(|s| s.is_imputed).collect();

        let get_genotyped_alleles = |marker_idx: usize, sample_idx: usize| -> Option<(u8, u8)> {
            let sample = sample_idx_from_usize(sample_idx);
            let h1 = sample.hap(HapSide::H1);
            let h2 = sample.hap(HapSide::H2);
            let resolution = resolved_ref_targets.get(marker_idx).copied().flatten()?;
            if resolution.is_positional_fallback() {
                pos_fallback_used.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            let target_m = MarkerIdx::new(resolution.target_idx as u32);
            if let Some(missing) = target_missing {
                let miss_a1 = missing.allele(target_m, h1);
                let miss_a2 = missing.allele(target_m, h2);
                if miss_a1 == crate::data::storage::AlleleCode::MISSING.raw()
                    || miss_a2 == crate::data::storage::AlleleCode::MISSING.raw()
                {
                    return None;
                }
            }
            let raw_a1 = target_win.allele(target_m, h1);
            let raw_a2 = target_win.allele(target_m, h2);
            if raw_a1 == crate::data::storage::AlleleCode::MISSING.raw()
                || raw_a2 == crate::data::storage::AlleleCode::MISSING.raw()
            {
                return None;
            }
            let a1 = match map_target_allele_to_ref(alignment, resolution, raw_a1) {
                Some(v) => v,
                None => {
                    pos_fallback_map_fail.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return None;
                }
            };
            let a2 = match map_target_allele_to_ref(alignment, resolution, raw_a2) {
                Some(v) => v,
                None => {
                    pos_fallback_map_fail.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return None;
                }
            };
            Some((a1, a2))
        };

        let get_target_raw_dosage = |marker_idx: usize, sample_idx: usize| -> Option<f32> {
            let sample = sample_idx_from_usize(sample_idx);
            let h1 = sample.hap(HapSide::H1);
            let h2 = sample.hap(HapSide::H2);
            let ref_marker = ref_markers.marker(MarkerIdx::new(marker_idx as u32));
            if ref_marker.n_alleles() != 2 {
                return None;
            }
            let resolution = resolved_ref_targets.get(marker_idx).copied().flatten()?;
            if resolution.is_positional_fallback() {
                pos_fallback_used.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            let target_m = MarkerIdx::new(resolution.target_idx as u32);

            if let Some(missing) = target_missing {
                if missing.allele(target_m, h1) == crate::data::storage::AlleleCode::MISSING.raw()
                    || missing.allele(target_m, h2)
                        == crate::data::storage::AlleleCode::MISSING.raw()
                {
                    return None;
                }
            }

            let a1 = target_win.allele(target_m, h1);
            let a2 = target_win.allele(target_m, h2);
            if a1 == crate::data::storage::AlleleCode::MISSING.raw()
                || a2 == crate::data::storage::AlleleCode::MISSING.raw()
            {
                return None;
            }
            let ra1 = match map_target_allele_to_ref(alignment, resolution, a1) {
                Some(v) => v,
                None => {
                    pos_fallback_map_fail.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return None;
                }
            };
            let ra2 = match map_target_allele_to_ref(alignment, resolution, a2) {
                Some(v) => v,
                None => {
                    pos_fallback_map_fail.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return None;
                }
            };
            let d = (ra1 + ra2) as f32;
            if samples.is_diploid(SampleIdx::new(sample_idx as u32)) {
                Some(d)
            } else {
                Some(d * 0.5)
            }
        };

        let get_posteriors_for_writer = if include_posteriors {
            Some(|marker_idx: usize, sample_idx: usize| {
                let local_m = marker_idx.saturating_sub(output_start);
                if let Some(result) = result_by_sample.get(sample_idx).and_then(|r| *r) {
                    let biallelic = ref_is_biallelic.get(marker_idx).copied().unwrap_or(false);
                    let post1 = result
                        .hap_posterior(0, local_m)
                        .cloned()
                        .or_else(|| {
                            if biallelic {
                                get_result_alt_prob(result, 0, local_m)
                                    .map(AllelePosteriors::Biallelic)
                            } else {
                                None
                            }
                        })
                        .unwrap_or_else(|| default_posteriors(marker_idx).0);
                    let post2 = result
                        .hap_posterior(1, local_m)
                        .cloned()
                        .or_else(|| {
                            if biallelic {
                                get_result_alt_prob(result, 1, local_m)
                                    .map(AllelePosteriors::Biallelic)
                            } else {
                                None
                            }
                        })
                        .unwrap_or_else(|| default_posteriors(marker_idx).1);
                    return (post1, post2);
                }
                default_posteriors(marker_idx)
            })
        } else {
            None
        };

        let genotype_index = |a: usize, b: usize| -> usize {
            let (i, j) = if a <= b { (a, b) } else { (b, a) };
            j * (j + 1) / 2 + i
        };

        let best_gt_from_gp = |n_alleles: usize, gp: &[f32]| -> (u8, u8) {
            let mut best = (0u8, 0u8);
            let mut best_prob = -1.0f32;
            let mut idx = 0usize;
            for j in 0..n_alleles {
                for i in 0..=j {
                    let p = gp.get(idx).copied().unwrap_or(0.0);
                    if p > best_prob {
                        best_prob = p;
                        if i == j {
                            best = (i as u8, i as u8);
                        } else {
                            best = (i as u8, j as u8);
                        }
                    }
                    idx += 1;
                }
            }
            best
        };

        let dosage_from_gp = |n_alleles: usize, gp: &[f32]| -> f32 {
            let mut dosage = 0.0f32;
            let mut idx = 0usize;
            for j in 0..n_alleles {
                for i in 0..=j {
                    let p = gp.get(idx).copied().unwrap_or(0.0);
                    let alt_count = (i > 0) as u8 + (j > 0) as u8;
                    dosage += p * (alt_count as f32);
                    idx += 1;
                }
            }
            dosage
        };

        let error_rate = self.params.p_mismatch;
        let use_hard_call_fallback = !correct_errors;
        let get_genotype_posteriors = |marker_idx: usize, sample_idx: usize| -> Option<Vec<f32>> {
            let resolution = resolved_ref_targets.get(marker_idx).copied().flatten()?;
            let target_m = MarkerIdx::new(resolution.target_idx as u32);
            // If the target genotype is missing at this marker, defer to imputation
            // posteriors instead of GL/PL-derived genotype probabilities.
            if get_genotyped_alleles(marker_idx, sample_idx).is_none() {
                return None;
            }
            let pl_opt = target_pl.sample_pl(target_m, sample_idx);

            if let Some(pl) = pl_opt {
                if !pl.is_empty() {
                    let n_pl_alleles = infer_n_alleles_from_pl_len(pl.len())?;
                    if n_pl_alleles == 0 {
                        return None;
                    }
                    let n_ref_alleles = ref_markers
                        .marker(MarkerIdx::new(marker_idx as u32))
                        .n_alleles();
                    let mut target_gp: Vec<f32> = Vec::new();
                    let n = genotype_probs_from_pl(pl, None, &mut target_gp)?;
                    if n != n_pl_alleles {
                        return None;
                    }

                    let mut ref_gp = vec![0.0f32; n_ref_alleles * (n_ref_alleles + 1) / 2];
                    let mut idx = 0usize;
                    for j in 0..n_pl_alleles {
                        for i in 0..=j {
                            let p = target_gp.get(idx).copied().unwrap_or(0.0);
                            idx += 1;
                            let ri = if i <= u8::MAX as usize {
                                map_target_allele_to_ref(alignment, resolution, i as u8)
                            } else {
                                None
                            };
                            let rj = if j <= u8::MAX as usize {
                                map_target_allele_to_ref(alignment, resolution, j as u8)
                            } else {
                                None
                            };
                            let (ri, rj) = match (ri, rj) {
                                (Some(ri), Some(rj)) => (ri as usize, rj as usize),
                                _ => continue,
                            };
                            if ri >= n_ref_alleles || rj >= n_ref_alleles {
                                continue;
                            }
                            let ref_idx = genotype_index(ri, rj);
                            if ref_idx < ref_gp.len() {
                                ref_gp[ref_idx] += p;
                            }
                        }
                    }
                    if !normalize_probs(&mut ref_gp) {
                        return None;
                    }
                    return Some(ref_gp);
                }
            }
            // Soft fallback using hard GT with an error rate: avoids hard-calling
            // genotyped markers when PLs are missing or uninformative.
            let n_ref_alleles = ref_markers
                .marker(MarkerIdx::new(marker_idx as u32))
                .n_alleles();
            if let Some((a1, a2)) = get_genotyped_alleles(marker_idx, sample_idx) {
                let n_genotypes = n_ref_alleles * (n_ref_alleles + 1) / 2;
                if n_genotypes == 0 {
                    return None;
                }
                let mut gp = vec![0.0f32; n_genotypes];
                let idx = genotype_index(a1 as usize, a2 as usize);
                let err = if use_hard_call_fallback {
                    0.0
                } else {
                    error_rate.clamp(1e-6, 0.5)
                };
                let main = 1.0 - err;
                if idx < gp.len() {
                    gp[idx] = main;
                }
                let spill = if n_genotypes > 1 {
                    err / (n_genotypes as f32 - 1.0)
                } else {
                    0.0
                };
                for (i, v) in gp.iter_mut().enumerate() {
                    if i != idx {
                        *v = spill;
                    }
                }
                return Some(gp);
            }
            None
        };

        // Closure to get dosage: marker_idx is window-local ref marker index from VCF writer
        // Dosages array is indexed from 0 for markers starting at output_start
        let get_dosage = |marker_idx: usize, sample_idx: usize| -> f32 {
            let hard_call = get_genotyped_alleles(marker_idx, sample_idx);
            let is_imputed = marker_is_imputed.get(marker_idx).copied().unwrap_or(true);

            if !is_imputed && !correct_errors {
                if let Some(d) = get_target_raw_dosage(marker_idx, sample_idx) {
                    return d;
                }
                let n_alleles = ref_markers
                    .marker(MarkerIdx::new(marker_idx as u32))
                    .n_alleles()
                    .max(1);
                if let Some(gp) = get_genotype_posteriors(marker_idx, sample_idx) {
                    let d = dosage_from_gp(n_alleles, &gp);
                    return if samples.is_diploid(SampleIdx::new(sample_idx as u32)) {
                        d
                    } else {
                        d * 0.5
                    };
                }
                if let Some((a1, a2)) = hard_call {
                    let d = (a1 + a2) as f32;
                    return if samples.is_diploid(SampleIdx::new(sample_idx as u32)) {
                        d
                    } else {
                        d * 0.5
                    };
                }
            }

            // Prefer hard calls if error correction is disabled
            if !correct_errors {
                if let Some(d) = get_target_raw_dosage(marker_idx, sample_idx) {
                    return d;
                }
                if let Some((a1, a2)) = hard_call {
                    let d = (a1 + a2) as f32;
                    return if samples.is_diploid(SampleIdx::new(sample_idx as u32)) {
                        d
                    } else {
                        d * 0.5
                    };
                }
            }

            let n_alleles = ref_markers
                .marker(MarkerIdx::new(marker_idx as u32))
                .n_alleles()
                .max(1);
            let local_m = marker_idx.saturating_sub(output_start);

            let dosage = if let Some(result) = result_by_sample.get(sample_idx).and_then(|r| *r) {
                let hap_dosage = |hap: usize| -> f32 {
                    if let Some(p) = result.hap_posterior(hap, local_m) {
                        return match p {
                            AllelePosteriors::Biallelic(p_alt) => *p_alt,
                            AllelePosteriors::Multiallelic(probs) => {
                                probs.iter().enumerate().map(|(i, p)| i as f32 * p).sum()
                            }
                        };
                    }
                    if n_alleles <= 2 {
                        if let Some(alt) = get_result_alt_prob(result, hap, local_m) {
                            return alt;
                        }
                    }
                    0.0
                };
                // WARNING: Do NOT blend PBWT donor alt-probs into these
                // dosages (e.g. `0.55 * HMM + 0.45 * PBWT`). The PBWT donor
                // trace is a heuristic match signal, not a calibrated posterior.
                // A 45% weight injects massive wrong signal when the best PBWT
                // donor carries the wrong allele (common at low-MAF sites),
                // corrupting dosage calibration. Tested in PR #758: R² -0.0012,
                // Hellinger +0.001, +97s slower.
                let d1 = hap_dosage(0);
                let d2 = hap_dosage(1);
                d1 + d2
            } else if !correct_errors {
                if let Some(gp) = get_genotype_posteriors(marker_idx, sample_idx) {
                    dosage_from_gp(n_alleles, &gp)
                } else {
                    0.0
                }
            } else if let Some((a1, a2)) = hard_call {
                (a1 + a2) as f32
            } else {
                0.0
            };

            if samples.is_diploid(SampleIdx::new(sample_idx as u32)) {
                dosage
            } else {
                dosage * 0.5
            }
        };

        // Closure to get best genotype
        let get_best_gt = |marker_idx: usize, sample_idx: usize| -> (u8, u8) {
            let hard_call = get_genotyped_alleles(marker_idx, sample_idx);
            let is_imputed = marker_is_imputed.get(marker_idx).copied().unwrap_or(true);

            if !is_imputed && !correct_errors {
                if let Some(gt) = hard_call {
                    return gt;
                }
                if let Some(gp) = get_genotype_posteriors(marker_idx, sample_idx) {
                    let n_alleles = ref_markers
                        .marker(MarkerIdx::new(marker_idx as u32))
                        .n_alleles();
                    return best_gt_from_gp(n_alleles, &gp);
                }
            }

            // Prefer hard calls if error correction is disabled
            if !correct_errors {
                if let Some(gt) = hard_call {
                    return gt;
                }
            }

            let local_m = marker_idx.saturating_sub(output_start);

            if let Some(result) = result_by_sample.get(sample_idx).and_then(|r| *r) {
                let n_alleles = ref_markers
                    .marker(MarkerIdx::new(marker_idx as u32))
                    .n_alleles()
                    .max(1);
                let p1_alt = result
                    .hap_posterior(0, local_m)
                    .map(|p| p.prob(1))
                    .or_else(|| get_result_alt_prob(result, 0, local_m));
                let p2_alt = result
                    .hap_posterior(1, local_m)
                    .map(|p| p.prob(1))
                    .or_else(|| get_result_alt_prob(result, 1, local_m));
                if n_alleles <= 2 {
                    let p1_alt = p1_alt.unwrap_or(0.0);
                    let p2_alt = p2_alt.unwrap_or(0.0);
                    let gp00 = (1.0 - p1_alt) * (1.0 - p2_alt);
                    let gp01 = p1_alt * (1.0 - p2_alt) + (1.0 - p1_alt) * p2_alt;
                    let gp11 = p1_alt * p2_alt;
                    if gp01 >= gp00 && gp01 >= gp11 {
                        let p10 = p1_alt * (1.0 - p2_alt);
                        let p01 = (1.0 - p1_alt) * p2_alt;
                        if p10 >= p01 { (1, 0) } else { (0, 1) }
                    } else if gp11 >= gp00 {
                        (1, 1)
                    } else {
                        (0, 0)
                    }
                } else if result.hap_posteriors.0.is_some() || result.hap_posteriors.1.is_some() {
                    let mut best = (0u8, 0u8);
                    let mut best_prob = -1.0f32;
                    for i in 0..n_alleles {
                        for j in i..n_alleles {
                            let p_i1 = result.hap_prob(0, local_m, i as u8).unwrap_or(0.0);
                            let p_i2 = result.hap_prob(1, local_m, i as u8).unwrap_or(0.0);
                            let p_j1 = result.hap_prob(0, local_m, j as u8).unwrap_or(0.0);
                            let p_j2 = result.hap_prob(1, local_m, j as u8).unwrap_or(0.0);
                            let prob = if i == j {
                                p_i1 * p_i2
                            } else {
                                p_i1 * p_j2 + p_j1 * p_i2
                            };
                            if prob > best_prob {
                                best_prob = prob;
                                if i == j {
                                    best = (i as u8, i as u8);
                                } else {
                                    let p_ij = p_i1 * p_j2;
                                    let p_ji = p_j1 * p_i2;
                                    if p_ij >= p_ji {
                                        best = (i as u8, j as u8);
                                    } else {
                                        best = (j as u8, i as u8);
                                    }
                                }
                            }
                        }
                    }
                    best
                } else {
                    (0, 0)
                }
            } else if !correct_errors {
                if let Some(gp) = get_genotype_posteriors(marker_idx, sample_idx) {
                    let n_alleles = ref_markers
                        .marker(MarkerIdx::new(marker_idx as u32))
                        .n_alleles();
                    best_gt_from_gp(n_alleles, &gp)
                } else {
                    (0, 0)
                }
            } else if let Some(gt) = hard_call {
                gt
            } else {
                (0, 0)
            }
        };

        let get_hap_probs = |marker_idx: usize, sample_idx: usize| -> (f32, f32) {
            let is_imputed = marker_is_imputed.get(marker_idx).copied().unwrap_or(true);
            if !is_imputed {
                if let Some((a1, a2)) = get_genotyped_alleles(marker_idx, sample_idx) {
                    return (a1 as f32, a2 as f32);
                }
                let n_alleles = ref_markers
                    .marker(MarkerIdx::new(marker_idx as u32))
                    .n_alleles()
                    .max(1);
                if let Some(gp) = get_genotype_posteriors(marker_idx, sample_idx) {
                    let dosage = dosage_from_gp(n_alleles, &gp);
                    let p_alt = (dosage * 0.5).clamp(0.0, 1.0);
                    return (p_alt, p_alt);
                }
            }
            let local_m = marker_idx.saturating_sub(output_start);
            if let Some(result) = result_by_sample.get(sample_idx).and_then(|r| *r) {
                let v1 = result
                    .hap_posterior(0, local_m)
                    .map(|p| p.prob(1))
                    .or_else(|| get_result_alt_prob(result, 0, local_m))
                    .unwrap_or(0.0);
                let v2 = result
                    .hap_posterior(1, local_m)
                    .map(|p| p.prob(1))
                    .or_else(|| get_result_alt_prob(result, 1, local_m))
                    .unwrap_or(0.0);
                return (v1, v2);
            }
            (0.0, 0.0)
        };

        let mut p1_buf: Vec<f32> = Vec::new();
        let mut p2_buf: Vec<f32> = Vec::new();
        let mut allele_counts: Vec<f32> = Vec::new();

        let mut update_multiallelic = |stats: &mut ImputationQuality,
                                       marker_idx: usize,
                                       sample_idx: usize,
                                       use_hard: bool| {
            let n_alleles = ref_markers
                .marker(MarkerIdx::new(marker_idx as u32))
                .n_alleles()
                .max(1);
            p1_buf.resize(n_alleles, 0.0);
            p2_buf.resize(n_alleles, 0.0);
            p1_buf.fill(0.0);
            p2_buf.fill(0.0);

            if use_hard {
                if let Some((a1, a2)) = get_genotyped_alleles(marker_idx, sample_idx) {
                    if (a1 as usize) < n_alleles {
                        p1_buf[a1 as usize] = 1.0;
                    }
                    if (a2 as usize) < n_alleles {
                        p2_buf[a2 as usize] = 1.0;
                    }
                    if let Some(stats) = stats.get_mut(marker_idx) {
                        stats.add_sample_multiallelic(&p1_buf, &p2_buf);
                        return;
                    }
                }
            }

            if let Some(gp) = get_genotype_posteriors(marker_idx, sample_idx) {
                allele_counts.resize(n_alleles, 0.0);
                allele_counts.fill(0.0);
                let mut idx = 0usize;
                for j in 0..n_alleles {
                    for i in 0..=j {
                        let p = gp.get(idx).copied().unwrap_or(0.0).max(0.0);
                        if i == j {
                            allele_counts[i] += 2.0 * p;
                        } else {
                            allele_counts[i] += p;
                            allele_counts[j] += p;
                        }
                        idx += 1;
                    }
                }
                for a in 0..n_alleles {
                    let v = (allele_counts[a] * 0.5).clamp(0.0, 1.0);
                    p1_buf[a] = v;
                    p2_buf[a] = v;
                }
            } else if let Some(result) = result_by_sample.get(sample_idx).and_then(|r| *r) {
                let local_m = marker_idx.saturating_sub(output_start);
                if let Some(p1) = result.hap_posterior(0, local_m) {
                    for a in 0..n_alleles {
                        p1_buf[a] = p1.prob(a).clamp(0.0, 1.0);
                    }
                }
                if let Some(p2) = result.hap_posterior(1, local_m) {
                    for a in 0..n_alleles {
                        p2_buf[a] = p2.prob(a).clamp(0.0, 1.0);
                    }
                }
            }

            if let Some(stats) = stats.get_mut(marker_idx) {
                stats.add_sample_multiallelic(&p1_buf, &p2_buf);
            }
        };

        if include_posteriors {
            for marker_idx in markers_to_process_start..output_end {
                let n_alleles = ref_markers
                    .marker(MarkerIdx::new(marker_idx as u32))
                    .n_alleles()
                    .max(1);
                if n_alleles == 2 {
                    if let Some(stats) = quality.get_mut(marker_idx) {
                        for s in 0..n_samples {
                            let (v1, v2) = get_hap_probs(marker_idx, s);
                            let (v1, v2) = if !stats.is_imputed && !correct_errors {
                                if let Some((a1, a2)) = get_genotyped_alleles(marker_idx, s) {
                                    (a1 as f32, a2 as f32)
                                } else if let Some(gp) = get_genotype_posteriors(marker_idx, s) {
                                    let dosage = dosage_from_gp(n_alleles, &gp);
                                    let p_alt = (dosage * 0.5).clamp(0.0, 1.0);
                                    (p_alt, p_alt)
                                } else {
                                    (v1, v2)
                                }
                            } else {
                                (v1, v2)
                            };
                            stats.add_sample_biallelic(v1, v2);
                        }
                    }
                } else if let Some(is_imputed) =
                    quality.get(marker_idx).map(|stats| stats.is_imputed)
                {
                    for s in 0..n_samples {
                        update_multiallelic(quality, marker_idx, s, !is_imputed && !correct_errors);
                    }
                }
            }
        } else {
            for marker_idx in markers_to_process_start..output_end {
                let n_alleles = ref_markers
                    .marker(MarkerIdx::new(marker_idx as u32))
                    .n_alleles()
                    .max(1);
                if n_alleles == 2 {
                    if let Some(stats) = quality.get_mut(marker_idx) {
                        for s in 0..n_samples {
                            let (v1, v2) = get_hap_probs(marker_idx, s);
                            let (v1, v2) = if !stats.is_imputed && !correct_errors {
                                if let Some((a1, a2)) = get_genotyped_alleles(marker_idx, s) {
                                    (a1 as f32, a2 as f32)
                                } else if let Some(gp) = get_genotype_posteriors(marker_idx, s) {
                                    let dosage = dosage_from_gp(n_alleles, &gp);
                                    let p_alt = (dosage * 0.5).clamp(0.0, 1.0);
                                    (p_alt, p_alt)
                                } else {
                                    (v1, v2)
                                }
                            } else {
                                (v1, v2)
                            };
                            stats.add_sample_biallelic(v1, v2);
                        }
                    }
                } else if let Some(is_imputed) =
                    quality.get(marker_idx).map(|stats| stats.is_imputed)
                {
                    for s in 0..n_samples {
                        update_multiallelic(quality, marker_idx, s, !is_imputed && !correct_errors);
                    }
                }
            }
        }

        let get_genotype_posteriors_for_writer = if include_gp && !correct_errors {
            Some(|m, s| get_genotype_posteriors(m, s))
        } else {
            None
        };

        // Preserve all target-only markers in this target window.
        let mut target_only_by_pos: std::collections::HashMap<
            String,
            std::collections::BTreeMap<u32, Vec<usize>>,
        > = std::collections::HashMap::new();
        for t_idx in 0..target_win.n_markers() {
            let target_m = MarkerIdx::new(t_idx as u32);
            if alignment.target_to_ref(target_m).is_some() {
                continue;
            }
            let t_marker = target_win.marker(target_m);
            let t_chrom = target_win
                .markers()
                .chrom_name(t_marker.chrom)
                .unwrap_or("");
            let chrom_key = normalize_chrom_local(t_chrom).to_string();
            target_only_by_pos
                .entry(chrom_key)
                .or_default()
                .entry(t_marker.pos)
                .or_default()
                .push(t_idx);
        }

        if target_only_by_pos.is_empty() {
            return writer.write_imputed_streaming(
                ref_markers,
                get_dosage,
                get_best_gt,
                get_posteriors_for_writer,
                get_genotype_posteriors_for_writer,
                quality,
                output_start,
                output_end,
                include_gp,
                include_ap,
                self.telemetry.as_ref(),
            );
        }

        #[derive(Clone, Copy)]
        enum OutputMarker {
            Ref(usize),
            Target(usize),
        }

        let mut output_markers: Vec<OutputMarker> = Vec::new();
        let mut emitted_target = vec![false; target_win.n_markers()];
        let mut chrom_rank: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for (idx, name) in ref_markers.chrom_names().iter().enumerate() {
            chrom_rank.insert(normalize_chrom_local(name.as_ref()).to_string(), idx);
        }
        let mut target_only_linear: std::collections::HashMap<String, Vec<usize>> =
            std::collections::HashMap::new();
        let mut target_only_cursor: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for (chrom_key, by_pos) in &target_only_by_pos {
            let mut v: Vec<usize> = Vec::new();
            for targets in by_pos.values() {
                v.extend(targets.iter().copied());
            }
            target_only_cursor.insert(chrom_key.clone(), 0);
            target_only_linear.insert(chrom_key.clone(), v);
        }

        for ref_m in output_start..output_end {
            let ref_marker = ref_markers.marker(MarkerIdx::new(ref_m as u32));
            let ref_chrom = ref_markers.chrom_name(ref_marker.chrom).unwrap_or("");
            let chrom_key = normalize_chrom_local(ref_chrom).to_string();
            if let (Some(list), Some(cursor)) = (
                target_only_linear.get(&chrom_key),
                target_only_cursor.get_mut(&chrom_key),
            ) {
                while *cursor < list.len() {
                    let t_idx = list[*cursor];
                    let t_pos = target_win.marker(MarkerIdx::new(t_idx as u32)).pos;
                    if t_pos > ref_marker.pos {
                        break;
                    }
                    if !emitted_target[t_idx] {
                        emitted_target[t_idx] = true;
                        output_markers.push(OutputMarker::Target(t_idx));
                    }
                    *cursor += 1;
                }
            }
            output_markers.push(OutputMarker::Ref(ref_m));
        }

        let mut chrom_keys: Vec<String> = target_only_linear.keys().cloned().collect();
        chrom_keys.sort_by(|a, b| {
            let ra = chrom_rank.get(a).copied().unwrap_or(usize::MAX);
            let rb = chrom_rank.get(b).copied().unwrap_or(usize::MAX);
            ra.cmp(&rb).then_with(|| a.cmp(b))
        });
        for chrom_key in chrom_keys {
            let Some(list) = target_only_linear.get(&chrom_key) else {
                continue;
            };
            let cursor = target_only_cursor.get(&chrom_key).copied().unwrap_or(0);
            for &t_idx in list.iter().skip(cursor) {
                if !emitted_target[t_idx] {
                    emitted_target[t_idx] = true;
                    output_markers.push(OutputMarker::Target(t_idx));
                }
            }
        }

        let mut out_markers = crate::data::marker::Markers::<RefMarkerSpace>::new();
        let mut out_chroms: std::collections::HashMap<String, ChromIdx> =
            std::collections::HashMap::new();
        for name in ref_markers.chrom_names() {
            out_markers.add_chrom(name.as_ref());
            let idx = ChromIdx::new((out_markers.chrom_names().len() - 1) as u16);
            out_chroms.insert(normalize_chrom_local(name.as_ref()).to_string(), idx);
        }

        let mut n_alleles_per_marker: Vec<usize> = Vec::with_capacity(output_markers.len());
        for om in &output_markers {
            match *om {
                OutputMarker::Ref(ref_m) => {
                    let marker = ref_markers.marker(MarkerIdx::new(ref_m as u32)).clone();
                    n_alleles_per_marker.push(marker.n_alleles());
                    out_markers.push(marker);
                }
                OutputMarker::Target(t_idx) => {
                    let t_marker = target_win.marker(MarkerIdx::new(t_idx as u32)).clone();
                    let t_chrom = target_win
                        .markers()
                        .chrom_name(t_marker.chrom)
                        .unwrap_or("");
                    let key = normalize_chrom_local(t_chrom).to_string();
                    let out_idx = if let Some(idx) = out_chroms.get(&key).copied() {
                        idx
                    } else {
                        out_markers.add_chrom(t_chrom);
                        let idx = ChromIdx::new((out_markers.chrom_names().len() - 1) as u16);
                        out_chroms.insert(key, idx);
                        idx
                    };
                    let mut marker = t_marker;
                    marker.chrom = out_idx;
                    n_alleles_per_marker.push(marker.n_alleles());
                    out_markers.push(marker);
                }
            }
        }

        let mut merged_quality = ImputationQuality::new(&n_alleles_per_marker);
        let target_samples = target_win.samples_arc();
        let get_target_alleles = |t_idx: usize, s: usize| -> Option<(u8, u8)> {
            let sample = sample_idx_from_usize(s);
            let h1 = sample.hap(HapSide::H1);
            let h2 = sample.hap(HapSide::H2);
            let m = MarkerIdx::new(t_idx as u32);
            if let Some(missing) = target_missing {
                if missing.allele(m, h1) == crate::data::storage::AlleleCode::MISSING.raw()
                    || missing.allele(m, h2) == crate::data::storage::AlleleCode::MISSING.raw()
                {
                    return None;
                }
            }
            let a1 = target_win.allele(m, h1);
            let a2 = target_win.allele(m, h2);
            if a1 == crate::data::storage::AlleleCode::MISSING.raw()
                || a2 == crate::data::storage::AlleleCode::MISSING.raw()
            {
                None
            } else {
                Some((a1, a2))
            }
        };

        for (out_idx, om) in output_markers.iter().enumerate() {
            match *om {
                OutputMarker::Ref(ref_m) => {
                    if let Some(stats) = quality.get(ref_m) {
                        merged_quality.marker_stats[out_idx] = stats.clone();
                    }
                }
                OutputMarker::Target(t_idx) => {
                    let n_alleles = n_alleles_per_marker[out_idx];
                    let mut stats = crate::io::vcf::MarkerImputationStats::new(n_alleles);
                    stats.is_imputed = false;
                    if n_alleles == 2 {
                        for s in 0..target_win.n_samples() {
                            if let Some((a1, a2)) = get_target_alleles(t_idx, s) {
                                let p1 = if a1 == 1 { 1.0 } else { 0.0 };
                                let p2 = if a2 == 1 { 1.0 } else { 0.0 };
                                stats.add_sample_biallelic(p1, p2);
                            }
                        }
                    }
                    merged_quality.marker_stats[out_idx] = stats;
                }
            }
        }

        let output_markers = std::sync::Arc::new(output_markers);
        let n_alleles_per_marker = std::sync::Arc::new(n_alleles_per_marker);

        let get_dosage_merged = {
            let output_markers = output_markers.clone();
            move |out_idx: usize, s: usize| -> f32 {
                match output_markers[out_idx] {
                    OutputMarker::Ref(ref_m) => get_dosage(ref_m, s),
                    OutputMarker::Target(t_idx) => {
                        if let Some((a1, a2)) = get_target_alleles(t_idx, s) {
                            let is_diploid = target_samples.is_diploid(SampleIdx::new(s as u32));
                            let d = (a1 > 0) as u8 + (a2 > 0) as u8;
                            if is_diploid {
                                d as f32
                            } else {
                                (d as f32) * 0.5
                            }
                        } else {
                            0.0
                        }
                    }
                }
            }
        };

        let get_best_gt_merged = {
            let output_markers = output_markers.clone();
            move |out_idx: usize, s: usize| -> (u8, u8) {
                match output_markers[out_idx] {
                    OutputMarker::Ref(ref_m) => get_best_gt(ref_m, s),
                    OutputMarker::Target(t_idx) => get_target_alleles(t_idx, s).unwrap_or((
                        crate::data::storage::AlleleCode::MISSING.raw(),
                        crate::data::storage::AlleleCode::MISSING.raw(),
                    )),
                }
            }
        };

        let get_posteriors_merged = get_posteriors_for_writer.as_ref().map(|base| {
            let output_markers = output_markers.clone();
            let n_alleles_per_marker = n_alleles_per_marker.clone();
            move |out_idx: usize, s: usize| match output_markers[out_idx] {
                OutputMarker::Ref(ref_m) => base(ref_m, s),
                OutputMarker::Target(_) => {
                    let n_alleles = n_alleles_per_marker[out_idx].max(1);
                    if n_alleles == 2 {
                        (
                            AllelePosteriors::Biallelic(0.0),
                            AllelePosteriors::Biallelic(0.0),
                        )
                    } else {
                        let zeros = std::sync::Arc::<[f32]>::from(vec![0.0f32; n_alleles]);
                        (
                            AllelePosteriors::Multiallelic(zeros.clone()),
                            AllelePosteriors::Multiallelic(zeros),
                        )
                    }
                }
            }
        });

        let genotype_index = |a: usize, b: usize| -> usize {
            let (i, j) = if a <= b { (a, b) } else { (b, a) };
            j * (j + 1) / 2 + i
        };

        let get_genotype_posteriors_merged = if include_gp && !correct_errors {
            let output_markers = output_markers.clone();
            let n_alleles_per_marker = n_alleles_per_marker.clone();
            Some(
                move |out_idx: usize, s: usize| match output_markers[out_idx] {
                    OutputMarker::Ref(ref_m) => get_genotype_posteriors(ref_m, s),
                    OutputMarker::Target(t_idx) => {
                        let n_alleles = n_alleles_per_marker[out_idx].max(1);
                        let n_genotypes = n_alleles * (n_alleles + 1) / 2;
                        let mut gp = vec![0.0f32; n_genotypes];
                        if let Some((a1, a2)) = get_target_alleles(t_idx, s) {
                            let idx = genotype_index(a1 as usize, a2 as usize);
                            if idx < gp.len() {
                                gp[idx] = 1.0;
                            }
                        }
                        Some(gp)
                    }
                },
            )
        } else {
            None
        };

        let fallback_used = pos_fallback_used.load(std::sync::atomic::Ordering::Relaxed);
        let fallback_fail = pos_fallback_map_fail.load(std::sync::atomic::Ordering::Relaxed);
        if fallback_used > 0 {
            if fallback_fail > 0 {
                return Err(ReagleError::vcf(format!(
                    "Position-only allele fallback failed: output={}..{} used={} failures={}. Refusing lossy fallback.",
                    output_start, output_end, fallback_used, fallback_fail
                )));
            }
            eprintln!(
                "    [alignment fallback] output={}..{} position_only_mapped={}",
                output_start, output_end, fallback_used
            );
        }

        writer.write_imputed_streaming(
            &out_markers,
            get_dosage_merged,
            get_best_gt_merged,
            get_posteriors_merged,
            get_genotype_posteriors_merged,
            &merged_quality,
            0,
            out_markers.len(),
            include_gp,
            include_ap,
            self.telemetry.as_ref(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::data::ChromIdx;
    use crate::data::alignment::MarkerAlignment;
    use crate::data::haplotype::Samples;
    use crate::data::marker::{Allele, Marker, Markers, Nucleotide};
    use crate::data::storage::GenotypeColumn;
    use crate::data::storage::phase_state::{Phased, Unphased};
    use crate::io::bref3::StreamingRefVcfReader;
    use crate::io::vcf::{ImputationQuality, VcfWriter};
    use crate::pipelines::ImputationPipeline;
    use std::io::{BufReader, Cursor};
    use tempfile::NamedTempFile;

    fn build_markers(chrom: ChromIdx, positions: &[u32]) -> Markers {
        let mut markers = Markers::<crate::data::AnyMarkerSpace>::new();
        markers.add_chrom("chr1");
        for (idx, &pos) in positions.iter().enumerate() {
            let marker = Marker::new(
                chrom,
                pos,
                Some(format!("m{idx}").into()),
                Allele::Base(Nucleotide::A),
                vec![Allele::Base(Nucleotide::C)],
            );
            markers.push(marker);
        }
        markers
    }

    fn build_unphased_matrix(markers: Markers, n_samples: usize) -> GenotypeMatrix<Unphased> {
        let samples = Arc::new(Samples::from_ids(
            (0..n_samples).map(|i| format!("s{i}")).collect(),
        ));
        let n_haps = n_samples * 2;
        let columns: Vec<GenotypeColumn> = (0..markers.len())
            .map(|_| {
                let bytes: Vec<u8> = vec![0u8; n_haps];
                GenotypeColumn::from_alleles(&bytes, 2)
            })
            .collect();
        GenotypeMatrix::new_unphased(markers, columns, samples)
    }

    fn build_phased_matrix_from_columns(
        markers: Markers,
        n_samples: usize,
        cols: Vec<Vec<u8>>,
        n_alleles_per_marker: &[usize],
    ) -> GenotypeMatrix<Phased> {
        let samples = Arc::new(Samples::from_ids(
            (0..n_samples).map(|i| format!("s{i}")).collect(),
        ));
        let mut columns: Vec<GenotypeColumn> = Vec::with_capacity(cols.len());
        for (m, alleles) in cols.into_iter().enumerate() {
            let n_alleles = n_alleles_per_marker.get(m).copied().unwrap_or(2).max(1);
            columns.push(GenotypeColumn::from_alleles(&alleles, n_alleles));
        }
        GenotypeMatrix::new_phased(markers, columns, samples)
    }

    fn score_window_batch_exact_packed_naive<TargetSpace, RefSpace>(
        batch_haps: &[usize],
        target_gt: &GenotypeMatrix<Phased, TargetSpace>,
        ref_markers: &Markers<RefSpace>,
        ref_columns: &[PackedRefColumn],
        n_ref_haps: usize,
        alignment: &MarkerAlignment<TargetSpace, RefSpace>,
        global_scores: &mut [Vec<f32>],
        window_scores: &mut [Vec<f32>],
    ) {
        let n_markers = ref_columns.len().min(ref_markers.len());
        if n_markers == 0 || n_ref_haps == 0 || batch_haps.is_empty() {
            return;
        }
        let resolutions =
            build_ref_typed_marker_resolutions(target_gt.markers(), ref_markers, alignment);
        let min_freq = 1.0 / (n_ref_haps.max(1) as f32);

        let mut query_alleles =
            vec![crate::data::storage::AlleleCode::MISSING.raw(); batch_haps.len()];
        let mut ref_bins: Vec<Vec<u32>> = Vec::new();

        for m in 0..n_markers {
            let Some(resolution) = resolutions.get(m).copied().flatten() else {
                continue;
            };
            for (i, &hap_idx) in batch_haps.iter().enumerate() {
                let raw = target_gt.allele(
                    MarkerIdx::new(resolution.target_idx as u32),
                    HapIdx::new(hap_idx as u32),
                );
                query_alleles[i] = map_target_allele_to_ref(alignment, resolution, raw)
                    .unwrap_or(crate::data::storage::AlleleCode::MISSING.raw());
            }

            let n_alleles = ref_markers
                .marker(MarkerIdx::new(m as u32))
                .n_alleles()
                .max(1);
            if ref_bins.len() < n_alleles {
                ref_bins.resize_with(n_alleles, Vec::new);
            }
            for bins in ref_bins.iter_mut().take(n_alleles) {
                bins.clear();
            }

            let col = &ref_columns[m];
            let mut present = 0usize;
            for rh in 0..n_ref_haps {
                let ref_a = col.allele(rh);
                if ref_a == crate::data::storage::AlleleCode::MISSING.raw() {
                    continue;
                }
                present += 1;
                let idx = ref_a as usize;
                if idx >= ref_bins.len() {
                    ref_bins.resize_with(idx + 1, Vec::new);
                }
                ref_bins[idx].push(rh as u32);
            }
            if present == 0 {
                continue;
            }

            for (i, _) in batch_haps.iter().enumerate() {
                let targ = query_alleles[i];
                if targ == crate::data::storage::AlleleCode::MISSING.raw() {
                    continue;
                }
                let freq = ref_bins
                    .get(targ as usize)
                    .map(|bins| bins.len() as f32 / present as f32)
                    .unwrap_or(0.0);
                if freq <= 0.0 {
                    continue;
                }
                let weight = prescan_match_weight(freq, min_freq);
                if weight <= 0.0 {
                    continue;
                }
                let bins = ref_bins.get(targ as usize);
                let Some(bins) = bins else { continue };
                for &rh in bins {
                    let idx = rh as usize;
                    global_scores[i][idx] += weight;
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

    #[test]
    fn test_calibrated_emission_error_sharp_observations_clamped_to_base() {
        // 3 observed, informative markers: near-certain hard calls.
        let offsets = vec![0, 2, 4, 6];
        let probs = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
        let observed = vec![true, true, true];
        let input = TargetAlleleProbs::new(offsets, probs, observed, None, 0.0);

        let base = 0.01;
        let out = calibrated_emission_error(&input, base);
        // Sharpening is allowed down to 10% of base or 1e-6.
        let limit = base * 0.1;
        assert!(
            out >= limit,
            "expected calibrated error clamped to limit {}, got {}",
            limit,
            out
        );
        // With sharp observations, it should be below base.
        assert!(
            out < base,
            "expected calibrated error to sharpen below base {}, got {}",
            base,
            out
        );
    }

    #[test]
    fn test_calibrated_emission_error_can_exceed_base_with_noisy_observations() {
        // Informative noisy markers (far from uniform) should raise posterior
        // error above a too-optimistic base rate under Beta shrinkage.
        let offsets = vec![0, 2, 4];
        let probs = vec![0.9, 0.1, 0.88, 0.12];
        let observed = vec![true, true];
        let input = TargetAlleleProbs::new(offsets, probs, observed, None, 0.0);

        let base = 0.02;
        let out = calibrated_emission_error(&input, base);
        assert!(
            out > base,
            "expected calibration to raise error above base {}, got {}",
            base,
            out
        );
        // For Beta-Binomial style shrinkage, posterior mean should lie between
        // prior mean (base) and empirical weighted residual mean.
        let empirical_weighted_residual = 0.10940167f32;
        assert!(
            out < empirical_weighted_residual,
            "expected shrinkage below empirical weighted residual {}, got {}",
            empirical_weighted_residual,
            out
        );
        assert!(
            out <= 0.5,
            "expected calibrated emission error in valid probability range, got {}",
            out
        );
    }

    #[test]
    fn test_orientation_eta_from_expected_flips_bounded_and_monotone() {
        let n_boundaries = 1000usize;
        let low = orientation_eta_from_expected_flips(0.0, n_boundaries);
        let mid = orientation_eta_from_expected_flips(10.0, n_boundaries);
        let high = orientation_eta_from_expected_flips(100.0, n_boundaries);
        assert!(low >= ORIENTATION_ETA_MIN);
        assert!(high <= ORIENTATION_ETA_MAX);
        assert!(mid >= low);
        assert!(high >= mid);
    }

    #[test]
    fn test_orientation_weights_from_posterior_swap_normalizes() {
        let (w_id, w_swap) = orientation_weights_from_posterior_swap(0.3);
        let sum = w_id + w_swap;
        assert!(w_id.is_finite() && w_swap.is_finite());
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(w_id > w_swap);
    }

    #[test]
    fn test_compose_boundary_message_id_swap_symmetry() {
        let p1 = HaplotypePriors::new(vec![GlobalHapId(1), GlobalHapId(3)], vec![0.75, 0.25]);
        let p2 = HaplotypePriors::new(vec![GlobalHapId(2), GlobalHapId(3)], vec![0.60, 0.40]);
        let m12 = compose_boundary_message(&p1, &p2, 0.3);
        let m21 = compose_boundary_message(&p2, &p1, 0.7);
        assert_eq!(m12.ids(), m21.ids());
        assert_eq!(m12.probs().len(), m21.probs().len());
        for (a, b) in m12.probs().iter().zip(m21.probs().iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_compute_abyss_rank_cutoff_uses_window_top_k() {
        let n_ref_haps = 6546usize;
        let cutoff = compute_abyss_rank_cutoff(n_ref_haps, 60);
        assert_eq!(cutoff, 60);
    }

    #[test]
    fn test_compute_abyss_rank_cutoff_clamps_to_panel_size() {
        let n_ref_haps = 200usize;
        let cutoff = compute_abyss_rank_cutoff(n_ref_haps, 1000);
        assert_eq!(cutoff, n_ref_haps);
    }

    #[test]
    fn test_sparse_target_should_not_truncate_reference_region() {
        let chrom = ChromIdx::new(0);
        let ref_positions: Vec<u32> = (0..3000).collect();
        let target_positions: Vec<u32> = vec![1500, 1501, 1502];

        let target_markers = build_markers(chrom, &target_positions);
        let target_gt = build_unphased_matrix(target_markers, 2);

        let mut vcf_data = String::new();
        vcf_data.push_str("##fileformat=VCFv4.2\n");
        vcf_data.push_str("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts0\ts1\n");
        for pos in &ref_positions {
            vcf_data.push_str(&format!(
                "chr1\t{}\t.\tA\tC\t.\tPASS\t.\tGT\t0|0\t0|0\n",
                pos
            ));
        }
        let reader = Box::new(BufReader::new(Cursor::new(vcf_data.into_bytes())));
        let ref_reader = StreamingRefVcfReader::from_reader(reader).expect("ref reader");
        let mut ref_reader = RefPanelReader::StreamingVcf(ref_reader);
        let config = StreamingConfig::default();
        let gen_maps = GeneticMaps::default();
        let ref_window = ref_reader
            .next_window(&config, &gen_maps, None)
            .expect("ref window load failed")
            .expect("no ref window found");

        // Desired behavior: sparse target data should not truncate the reference region.
        assert_eq!(target_gt.n_markers(), target_positions.len());
        assert_eq!(ref_window.global_start, 0);
        assert_eq!(ref_window.global_end, ref_positions.len());
    }

    #[test]
    fn test_write_imputed_window_ignores_short_ref_genotypes() {
        let chrom = ChromIdx::new(0);
        let ref_markers = build_markers(chrom, &[10, 20, 30]);
        let target_markers = build_markers(chrom, &[10]);

        let target_win = build_unphased_matrix(target_markers, 1).into_phased();

        let alignment = MarkerAlignment {
            ref_to_target: vec![None; ref_markers.len()],
            target_to_ref: vec![None; target_win.n_markers()],
            allele_mappings: vec![None; target_win.n_markers()],
        };

        let n_alleles_per_marker: Vec<usize> = (0..ref_markers.len())
            .map(|m| {
                let marker = ref_markers.marker(MarkerIdx::new(m as u32));
                1 + marker.alt_alleles.len()
            })
            .collect();
        let mut quality = ImputationQuality::new(&n_alleles_per_marker);

        let output_start = 0;
        let output_end = ref_markers.len();
        let all_results = vec![SampleImputationResult {
            sample_idx: 0,
            hap_alt_probs: (
                Some(vec![0.0; output_end - output_start]),
                Some(vec![0.0; output_end - output_start]),
            ),
            hap_posteriors: (None, None),
        }];

        let tmp = NamedTempFile::new().expect("temp vcf");
        let mut writer = VcfWriter::create(tmp.path(), target_win.samples_arc()).expect("writer");

        let pipeline = ImputationPipeline::new(Config::default(), None);
        let ref_is_biallelic = vec![true; ref_markers.len()];
        let result = pipeline.write_imputed_window_streaming(
            &ref_markers,
            &target_win,
            None,
            None::<&GenotypeMatrix<Phased, crate::data::AnyMarkerSpace>>,
            &alignment,
            &mut writer,
            &mut quality,
            &ref_is_biallelic,
            output_start,
            output_end,
            output_start,
            &all_results,
            None,
            false,
            false,
            false,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_write_imputed_window_preserves_target_only_marker_with_unique_position() {
        let chrom = ChromIdx::new(0);
        let ref_markers = build_markers(chrom, &[10, 30]);
        let target_markers = build_markers(chrom, &[20]);

        // One sample (2 haps): marker 20 => 0|1.
        let target_cols = vec![vec![0u8, 1u8]];
        let target_win = build_phased_matrix_from_columns(target_markers, 1, target_cols, &[2]);

        let alignment = MarkerAlignment {
            ref_to_target: vec![None; ref_markers.len()],
            target_to_ref: vec![None; target_win.n_markers()],
            allele_mappings: vec![None; target_win.n_markers()],
        };

        let n_alleles_per_marker: Vec<usize> = (0..ref_markers.len())
            .map(|m| {
                let marker = ref_markers.marker(MarkerIdx::new(m as u32));
                1 + marker.alt_alleles.len()
            })
            .collect();
        let mut quality = ImputationQuality::new(&n_alleles_per_marker);

        let output_start = 0;
        let output_end = ref_markers.len();
        let all_results = vec![SampleImputationResult {
            sample_idx: 0,
            hap_alt_probs: (
                Some(vec![0.0; output_end - output_start]),
                Some(vec![0.0; output_end - output_start]),
            ),
            hap_posteriors: (None, None),
        }];

        let tmp = NamedTempFile::new().expect("temp vcf");
        {
            let mut writer =
                VcfWriter::create(tmp.path(), target_win.samples_arc()).expect("writer");
            writer
                .write_header_extended(&ref_markers, true, false, false)
                .expect("write header");

            let pipeline = ImputationPipeline::new(Config::default(), None);
            let ref_is_biallelic = vec![true; ref_markers.len()];
            pipeline
                .write_imputed_window_streaming(
                    &ref_markers,
                    &target_win,
                    None,
                    None::<&GenotypeMatrix<Phased, crate::data::AnyMarkerSpace>>,
                    &alignment,
                    &mut writer,
                    &mut quality,
                    &ref_is_biallelic,
                    output_start,
                    output_end,
                    output_start,
                    &all_results,
                    None,
                    false,
                    false,
                    false,
                )
                .expect("write window");
        }

        let text = std::fs::read_to_string(tmp.path()).expect("read output");
        let positions: Vec<u32> = text
            .lines()
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .filter_map(|line| line.split('\t').nth(1))
            .map(|p| p.parse::<u32>().expect("POS parse"))
            .collect();
        assert_eq!(positions, vec![10, 20, 30]);
    }

    #[test]
    fn test_exact_prescan_grouped_matches_naive_multiallelic_missing() {
        let chrom = ChromIdx::new(0);
        let positions = [101u32, 202u32, 303u32, 404u32];
        let n_alleles_per_marker = [2usize, 3, 2, 3];
        let build_markers_with_arity = |positions: &[u32], n_alleles: &[usize]| {
            let mut markers = Markers::<crate::data::AnyMarkerSpace>::new();
            markers.add_chrom("chr1");
            for (idx, (&pos, &n_al)) in positions.iter().zip(n_alleles.iter()).enumerate() {
                let mut alts = Vec::new();
                if n_al >= 2 {
                    alts.push(Allele::Base(Nucleotide::C));
                }
                if n_al >= 3 {
                    alts.push(Allele::Base(Nucleotide::G));
                }
                if n_al >= 4 {
                    alts.push(Allele::Base(Nucleotide::T));
                }
                let marker = Marker::new(
                    chrom,
                    pos,
                    Some(format!("m{idx}").into()),
                    Allele::Base(Nucleotide::A),
                    alts,
                );
                markers.push(marker);
            }
            markers
        };
        let target_markers = build_markers_with_arity(&positions, &n_alleles_per_marker);
        let ref_markers = build_markers_with_arity(&positions, &n_alleles_per_marker);

        let n_target_samples = 2usize;
        let n_ref_samples = 3usize;
        let n_target_haps = n_target_samples * 2;
        let n_ref_haps = n_ref_samples * 2;

        // 4 target haplotypes across 4 markers; include missing + multi-allelic.
        let target_cols = vec![
            vec![0, 1, 1, 0],
            vec![2, 2, 0, 1],
            vec![1, u8::MAX, 1, 0],
            vec![0, 2, 2, 1],
        ];
        let target_n_alleles = n_alleles_per_marker.to_vec();
        let target_gt = build_phased_matrix_from_columns(
            target_markers,
            n_target_samples,
            target_cols,
            &target_n_alleles,
        );
        assert_eq!(target_gt.n_haplotypes(), n_target_haps);

        // 6 ref haplotypes across 4 markers; include missing + multi-allelic.
        let ref_cols_raw = vec![
            vec![0, 1, 1, 0, 0, 1],
            vec![2, 1, 0, 2, u8::MAX, 1],
            vec![1, 1, 0, 0, 1, u8::MAX],
            vec![0, 2, 2, 1, 1, 0],
        ];
        let ref_n_alleles = n_alleles_per_marker.to_vec();
        let ref_matrix = build_phased_matrix_from_columns(
            ref_markers.clone(),
            n_ref_samples,
            ref_cols_raw,
            &ref_n_alleles,
        );
        assert_eq!(ref_matrix.n_haplotypes(), n_ref_haps);

        let alignment = MarkerAlignment::new_with_ref_markers(&target_gt, &ref_markers);
        let mut packed_cols: Vec<PackedRefColumn> = Vec::with_capacity(ref_matrix.n_markers());
        for m in 0..ref_matrix.n_markers() {
            let packed = PackedRefColumn::pack_from_column(
                MarkerIdx::new(m as u32),
                ref_matrix.markers(),
                ref_matrix.column(MarkerIdx::new(m as u32)),
            )
            .expect("pack reference column");
            packed_cols.push(packed);
        }

        let batch_haps = vec![0usize, 1, 2, 3];
        let mut global_a = vec![vec![0.0f32; n_ref_haps]; batch_haps.len()];
        let mut window_a = vec![vec![f32::NEG_INFINITY; n_ref_haps]; batch_haps.len()];
        let mut global_b = vec![vec![0.0f32; n_ref_haps]; batch_haps.len()];
        let mut window_b = vec![vec![f32::NEG_INFINITY; n_ref_haps]; batch_haps.len()];

        score_window_batch_exact_packed_naive(
            &batch_haps,
            &target_gt,
            &ref_markers,
            &packed_cols,
            n_ref_haps,
            &alignment,
            &mut global_a,
            &mut window_a,
        );
        score_window_batch_exact_packed(
            &batch_haps,
            &target_gt,
            &ref_markers,
            &packed_cols,
            n_ref_haps,
            &alignment,
            &mut global_b,
            &mut window_b,
        );

        assert_score_mats_close(&global_a, &global_b, 1e-6);
        assert_score_mats_close(&window_a, &window_b, 1e-6);
    }

    #[test]
    fn test_typed_marker_resolution_positional_swap_maps_alleles() {
        let chrom = ChromIdx::new(0);
        let mut target_markers = Markers::<crate::data::AnyMarkerSpace>::new();
        target_markers.add_chrom("chr1");
        target_markers.push(Marker::new(
            chrom,
            10,
            Some("t0".into()),
            Allele::Base(Nucleotide::A),
            vec![Allele::Base(Nucleotide::C)],
        ));
        let target_gt = build_phased_matrix_from_columns(target_markers, 1, vec![vec![0, 1]], &[2]);

        let mut ref_markers = Markers::<crate::data::AnyMarkerSpace>::new();
        ref_markers.add_chrom("chr1");
        ref_markers.push(Marker::new(
            chrom,
            10,
            Some("r0".into()),
            Allele::Base(Nucleotide::C),
            vec![Allele::Base(Nucleotide::A)],
        ));
        let alignment = MarkerAlignment {
            ref_to_target: vec![None],
            target_to_ref: vec![None],
            allele_mappings: vec![None],
        };

        let resolved =
            build_ref_typed_marker_resolutions(target_gt.markers(), &ref_markers, &alignment);
        let r = resolved[0].expect("expected positional fallback resolution");
        assert_eq!(r.target_idx, 0);
        assert_eq!(r.map_kind, TypedMarkerMapKind::PositionalBiallelicSwap);
        assert_eq!(map_target_allele_to_ref(&alignment, r, 0), Some(1));
        assert_eq!(map_target_allele_to_ref(&alignment, r, 1), Some(0));
    }

    #[test]
    fn test_typed_marker_resolution_rejects_ambiguous_position_candidates() {
        let chrom = ChromIdx::new(0);
        let mut target_markers = Markers::<crate::data::AnyMarkerSpace>::new();
        target_markers.add_chrom("chr1");
        target_markers.push(Marker::new(
            chrom,
            10,
            Some("t0".into()),
            Allele::Base(Nucleotide::A),
            vec![Allele::Base(Nucleotide::C)],
        ));
        target_markers.push(Marker::new(
            chrom,
            10,
            Some("t1".into()),
            Allele::Base(Nucleotide::A),
            vec![Allele::Base(Nucleotide::C)],
        ));
        let target_gt = build_phased_matrix_from_columns(
            target_markers,
            1,
            vec![vec![0, 1], vec![0, 1]],
            &[2, 2],
        );

        let mut ref_markers = Markers::<crate::data::AnyMarkerSpace>::new();
        ref_markers.add_chrom("chr1");
        ref_markers.push(Marker::new(
            chrom,
            10,
            Some("r0".into()),
            Allele::Base(Nucleotide::C),
            vec![Allele::Base(Nucleotide::A)],
        ));
        let alignment = MarkerAlignment {
            ref_to_target: vec![None],
            target_to_ref: vec![None; target_gt.n_markers()],
            allele_mappings: vec![None; target_gt.n_markers()],
        };

        let resolved =
            build_ref_typed_marker_resolutions(target_gt.markers(), &ref_markers, &alignment);
        assert!(
            resolved[0].is_none(),
            "ambiguous positional candidates must not resolve"
        );
    }

    #[test]
    fn test_segment_build_target_probs_preserves_marker_error_rates() {
        let offsets = vec![0usize, 2, 4];
        let probs = vec![0.9f32, 0.1, 0.2, 0.8];
        let observed = vec![true, true];
        let mut input = TargetAlleleProbs::new(offsets, probs, observed, None, 0.0);
        input.set_marker_error_rates(vec![0.01, 0.2]);

        let extent = SegmentExtent::new(0, 1, 0, 1, 2);
        let seg = extent.build_target_probs(&input);
        assert!((seg.marker_error_rate(0).unwrap_or(0.0) - 0.01).abs() < 1e-6);
        assert!((seg.marker_error_rate(1).unwrap_or(0.0) - 0.2).abs() < 1e-6);
    }
}
