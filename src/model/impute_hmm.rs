//! HMM kernel for imputation using explicit haplotype states.
//!
//! This implements a Li-Stephens forward-backward pass over a selected set of
//! reference haplotypes (state set). Emissions are computed using per-haplotype
//! allele probabilities from the target, and reference alleles are read on demand.

use crate::data::HapIdx;
use crate::data::storage::{
    AlleleCode, DenseColumn, DictionaryColumn, GenotypeColumn, SeqCodedColumn, SparseColumn,
};
use crate::error::{ReagleError, Result};
use crate::model::li_stephens::subset_linear_exact_k;
use crate::model::types::RefHapId;
use crate::model::weighted_kernel::{EmissionProbs, PatternCounts, WeightedHmmUpdater};
use crate::pipelines::imputation::AllelePosteriors;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Default)]
pub struct ImputeHmmContext {
    pub window_idx: usize,
    pub sample_idx: usize,
    pub hap_idx: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
struct MarkerIx(usize);

impl MarkerIx {
    #[inline]
    fn new(idx: usize) -> Self {
        Self(idx)
    }

    #[inline]
    fn as_usize(self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
struct CheckpointIx(usize);

impl CheckpointIx {
    #[inline]
    fn new(idx: usize) -> Self {
        Self(idx)
    }

    #[inline]
    fn as_usize(self) -> usize {
        self.0
    }

    #[inline]
    fn fwd_offset(self, active_states: usize) -> usize {
        self.0 * active_states
    }
}

struct CheckpointGrid {
    markers: Vec<MarkerIx>,
}

impl CheckpointGrid {
    #[inline]
    fn len(&self) -> usize {
        self.markers.len()
    }

    #[inline]
    fn iter_forward(&self) -> impl Iterator<Item = (CheckpointIx, MarkerIx)> + '_ {
        self.markers
            .iter()
            .copied()
            .enumerate()
            .map(|(i, m)| (CheckpointIx::new(i), m))
    }

    #[inline]
    fn rev_indices(&self) -> impl Iterator<Item = CheckpointIx> {
        (0..self.markers.len()).rev().map(CheckpointIx::new)
    }

    #[inline]
    fn marker_at(&self, cp: CheckpointIx) -> MarkerIx {
        self.markers[cp.as_usize()]
    }

    #[inline]
    fn next(&self, cp: CheckpointIx) -> Option<CheckpointIx> {
        let next = cp.as_usize() + 1;
        if next < self.markers.len() {
            Some(CheckpointIx::new(next))
        } else {
            None
        }
    }

    #[inline]
    fn block_view(&self, cp: CheckpointIx, n_markers: usize) -> BlockView {
        let start = self.marker_at(cp);
        let end = self
            .next(cp)
            .map(|ncp| self.marker_at(ncp))
            .unwrap_or(MarkerIx::new(n_markers));
        BlockView { start, end }
    }
}

#[derive(Clone, Copy, Debug)]
struct BlockView {
    start: MarkerIx,
    end: MarkerIx,
}

impl BlockView {
    #[inline]
    fn len(self) -> usize {
        self.end.as_usize().saturating_sub(self.start.as_usize())
    }

    #[inline]
    fn start_usize(self) -> usize {
        self.start.as_usize()
    }

    #[inline]
    fn end_usize(self) -> usize {
        self.end.as_usize()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
struct UniformMarkerIx(MarkerIx);

impl UniformMarkerIx {
    #[inline]
    fn from_trusted(marker: MarkerIx) -> Self {
        Self(marker)
    }

    #[inline]
    fn as_usize(self) -> usize {
        self.0.as_usize()
    }
}

#[derive(Clone, Copy, Debug)]
struct UniformInteriorRange {
    start_exclusive: MarkerIx,
    end_exclusive: MarkerIx,
}

impl UniformInteriorRange {
    #[inline]
    fn from_block_checked(
        block: BlockView,
        uniform_mask: &MarkerMask<bool>,
        context: ImputeHmmContext,
        kernel: &str,
    ) -> Result<Option<Self>> {
        let start = block.start_usize();
        let end = block.end_usize();
        if start + 1 >= end {
            return Ok(None);
        }
        for m in start + 1..end {
            if !uniform_mask[MarkerIx::new(m)] {
                return Err(ReagleError::vcf(format!(
                    "Checkpoint interval contains non-uniform marker in imputation HMM ({}): window={} sample={} hap={} marker={}",
                    kernel, context.window_idx, context.sample_idx, context.hap_idx, m
                )));
            }
        }
        Ok(Some(Self {
            start_exclusive: block.start,
            end_exclusive: block.end,
        }))
    }

    #[inline]
    fn iter(self) -> impl Iterator<Item = UniformMarkerIx> {
        let start = self.start_exclusive.as_usize();
        let end = self.end_exclusive.as_usize();
        (start + 1..end).map(|m| UniformMarkerIx::from_trusted(MarkerIx::new(m)))
    }
}

#[repr(transparent)]
struct MarkerMask<T>(Vec<T>);

impl<T> MarkerMask<T> {
    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<T> std::ops::Index<MarkerIx> for MarkerMask<T> {
    type Output = T;
    #[inline]
    fn index(&self, index: MarkerIx) -> &Self::Output {
        &self.0[index.as_usize()]
    }
}

#[inline]
fn validate_target_probs_nonempty(
    target_probs: &TargetAlleleProbs,
    context: ImputeHmmContext,
    kernel: &str,
) -> Result<()> {
    for m in 0..target_probs.n_markers() {
        if target_probs.probs_for_marker(m).is_empty() {
            return Err(ReagleError::vcf(format!(
                "Empty target allele probabilities in imputation HMM ({}): window={} sample={} hap={} marker={}",
                kernel, context.window_idx, context.sample_idx, context.hap_idx, m
            )));
        }
    }
    Ok(())
}

#[inline]
fn validate_reference_marker_count(
    ref_marker_count: usize,
    target_probs: &TargetAlleleProbs,
    context: ImputeHmmContext,
    kernel: &str,
) -> Result<()> {
    let target_markers = target_probs.n_markers();
    if ref_marker_count != target_markers {
        return Err(ReagleError::vcf(format!(
            "Reference/target marker count mismatch in imputation HMM ({}): window={} sample={} hap={} ref_markers={} target_markers={}",
            kernel,
            context.window_idx,
            context.sample_idx,
            context.hap_idx,
            ref_marker_count,
            target_markers
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        AlleleProbsView, BackwardAffine, ForwardAffine, RefAlleles, effective_recomb_rate,
        fill_emissions, missing_emission_prob, subset_transition_params,
        subset_transition_params_adaptive_q, transition_only_forward_update,
    };
    use super::{RefColumnLike, RefHapId};
    use crate::data::storage::AlleleCode;
    use crate::data::storage::{DenseColumn, GenotypeColumn};

    #[test]
    fn test_recomb_mass_subset_sums_to_one() {
        let transition_haps = 50usize;
        let recomb_rate = 0.02f32;

        let mut fwd = vec![1.0 / transition_haps as f32; transition_haps];

        let fwd_sum: f32 = fwd.iter().sum();
        transition_only_forward_update(&mut fwd, fwd_sum, recomb_rate, transition_haps, 0.0);

        let sum: f32 = fwd.iter().sum();
        println!(
            "[subset mass] n_states={} r={:.4} sum={:.6}",
            transition_haps, recomb_rate, sum
        );

        assert!(
            (sum - 1.0).abs() < 1e-4,
            "Expected subset mass 1.0, got {:.6}",
            sum
        );
    }

    #[test]
    fn test_recomb_mass_full_panel_sums_to_one() {
        let n_total = 200usize;
        let n_states = 200usize;
        let recomb_rate = 0.02f32;

        let mut fwd = vec![1.0 / n_states as f32; n_states];
        let fwd_sum: f32 = fwd.iter().sum();
        transition_only_forward_update(&mut fwd, fwd_sum, recomb_rate, n_states, 0.0);

        let sum: f32 = fwd.iter().sum();
        println!(
            "[full-panel mass] n_states={} n_total={} r={:.4} sum={:.6}",
            n_states, n_total, recomb_rate, sum
        );

        assert!(
            (sum - 1.0).abs() < 1e-4,
            "Expected full-panel mass 1.0, got {:.6}",
            sum
        );
    }

    #[test]
    fn test_subset_transition_panel_aware_shift_small_vs_k_scaled() {
        let recomb_rate = 0.02f32;
        let active_states = 60usize;
        let n_ref_haps = 6_546usize;
        let (stay, shift) = subset_transition_params(recomb_rate, active_states, n_ref_haps);

        // panel-aware expectation
        let k = active_states as f32;
        let n = n_ref_haps as f32;
        let expected = (recomb_rate / n) / ((1.0 - recomb_rate) + k * (recomb_rate / n));
        assert!((shift - expected).abs() < 1e-8);
        assert!(stay.is_finite() && stay > 0.0);

        // Ensure we are not effectively using r/K scaling.
        let k_scaled = (recomb_rate / k) / ((1.0 - recomb_rate) + k * (recomb_rate / k));
        assert!(shift < k_scaled * 0.1);
    }

    #[test]
    fn test_fixed_lambda_transition_equals_canonical_with_r_eff() {
        let cases = [
            (0.0f32, 0.0f32, 16usize, 1000usize),
            (0.02f32, 0.0f32, 64usize, 10_000usize),
            (0.02f32, 0.5f32, 64usize, 10_000usize),
            (0.12f32, 0.98f32, 31usize, 6546usize),
            (1.0f32, 0.25f32, 3usize, 500usize),
        ];
        for (r, lambda, k, n) in cases {
            let (stay_l, shift_l) = subset_transition_params_adaptive_q(r, k, n, lambda);
            let r_eff = effective_recomb_rate(r, lambda);
            let (stay_e, shift_e) = subset_transition_params(r_eff, k, n);
            assert!(
                (stay_l - stay_e).abs() < 1e-6,
                "stay mismatch r={} lambda={} k={} n={} got={} expected={}",
                r,
                lambda,
                k,
                n,
                stay_l,
                stay_e
            );
            assert!(
                (shift_l - shift_e).abs() < 1e-8,
                "shift mismatch r={} lambda={} k={} n={} got={} expected={}",
                r,
                lambda,
                k,
                n,
                shift_l,
                shift_e
            );
        }
    }

    #[test]
    fn test_affine_group_mass_matches_direct_sum() {
        let fwd = [0.11f32, 0.27, 0.05, 0.33, 0.24];
        let bwd = [0.31f32, 0.07, 0.29, 0.13, 0.20];
        let groups = [0usize, 1, 1, 0, 2];
        let forward = ForwardAffine { a: 0.83, b: 0.017 };
        let backward = BackwardAffine::new(0.91, 0.012, 1.37);

        for group in 0..=2usize {
            let mut sum_f = 0.0f32;
            let mut sum_b = 0.0f32;
            let mut sum_fb = 0.0f32;
            let mut count = 0.0f32;
            let mut direct = 0.0f32;
            for i in 0..fwd.len() {
                if groups[i] != group {
                    continue;
                }
                let f = fwd[i];
                let b = bwd[i];
                sum_f += f;
                sum_b += b;
                sum_fb += f * b;
                count += 1.0;
                let f_aff = forward.a.mul_add(f, forward.b);
                let b_aff = backward.a.mul_add(b, backward.add);
                direct += f_aff * b_aff;
            }
            let fast = (forward.a * backward.a) * sum_fb
                + (forward.a * backward.add) * sum_f
                + (forward.b * backward.a) * sum_b
                + (forward.b * backward.add) * count;
            assert!(
                (fast - direct).abs() < 1e-6,
                "group={} fast={} direct={}",
                group,
                fast,
                direct
            );
        }
    }

    #[test]
    fn test_dense_fill_ref_alleles_oob_maps_to_missing_biallelic() {
        let col = DenseColumn::from_alleles([0u8, 1, 0, 1].into_iter(), 2);
        let state_haps = vec![RefHapId::new(0), RefHapId::new(3), RefHapId::new(9)];
        let mut out = vec![0u8; state_haps.len()];
        let mut dict_pattern_alleles = Vec::new();

        col.fill_ref_alleles(&state_haps, &mut out, &mut dict_pattern_alleles);

        assert_eq!(out[0], 0);
        assert_eq!(out[1], 1);
        assert_eq!(out[2], GenotypeColumn::MISSING_ALLELE);
    }

    #[test]
    fn test_dense_fill_ref_alleles_oob_maps_to_missing_multiallelic() {
        let col = DenseColumn::from_alleles([0u8, 2, 1].into_iter(), 3);
        let state_haps = vec![RefHapId::new(1), RefHapId::new(7)];
        let mut out = vec![0u8; state_haps.len()];
        let mut dict_pattern_alleles = Vec::new();

        col.fill_ref_alleles(&state_haps, &mut out, &mut dict_pattern_alleles);

        assert_eq!(out[0], 2);
        assert_eq!(out[1], GenotypeColumn::MISSING_ALLELE);
    }

    #[test]
    fn test_missing_emission_prob_tracks_target_concentration() {
        let match_prob = 0.99f32;
        let mismatch_prob = 0.01f32;

        let concentrated = AlleleProbsView::from_trusted(&[1.0, 0.0]);
        let diffuse = AlleleProbsView::from_trusted(&[0.5, 0.5]);

        let c = missing_emission_prob(concentrated, match_prob, mismatch_prob);
        let d = missing_emission_prob(diffuse, match_prob, mismatch_prob);

        assert!((c - match_prob).abs() < 1e-6);
        assert!(d < c);
        assert!(d > mismatch_prob);
    }

    #[test]
    fn test_fill_emissions_missing_not_superior_to_best_called() {
        let ref_raw = [AlleleCode::MISSING.raw(), 0, 1];
        let ref_alleles = RefAlleles { slice: &ref_raw };
        let target_probs = AlleleProbsView::from_trusted(&[0.5, 0.5]);
        let mut emission_by_allele = Vec::new();
        let mut emissions = vec![0.0f32; ref_raw.len()];

        fill_emissions(
            &ref_alleles,
            target_probs,
            0.01,
            &mut emission_by_allele,
            &mut emissions,
        );

        let best_called = emissions[1].max(emissions[2]);
        assert!(emissions[0] <= best_called + 1e-6);
    }

    #[test]
    fn test_impute_missing_emission_not_above_best_called_property_grid() {
        let prob_sets: [&[f32]; 5] = [
            &[1.0, 0.0],
            &[0.5, 0.5],
            &[0.8, 0.2],
            &[0.34, 0.33, 0.33],
            &[0.7, 0.2, 0.1],
        ];
        let error_rates = [0.001f32, 0.01, 0.05];

        for probs in prob_sets {
            for &error_rate in &error_rates {
                let target_probs = AlleleProbsView::from_trusted(probs);
                let n_alleles = probs.len();
                let mismatch_prob = if n_alleles > 1 {
                    error_rate / (n_alleles as f32 - 1.0)
                } else {
                    error_rate
                };
                let match_prob = 1.0 - error_rate;
                let miss = missing_emission_prob(target_probs, match_prob, mismatch_prob);
                let mut best_called = mismatch_prob;
                for &p in probs {
                    let called = mismatch_prob + (match_prob - mismatch_prob) * p;
                    if called > best_called {
                        best_called = called;
                    }
                }
                assert!(miss <= best_called + 1e-6);
            }
        }
    }
}

/// Per-marker allele probability distributions for a single target haplotype.
pub struct TargetAlleleProbs {
    offsets: Vec<usize>,
    probs: Vec<f32>,
    uniform: Vec<bool>,
    observed: Vec<bool>,
    marker_error_rates: Option<Vec<f32>>,
    panel_priors: Option<Arc<Vec<AllelePosteriors>>>,
    min_untyped_prior_mix: f32,
}

impl TargetAlleleProbs {
    pub fn new(
        offsets: Vec<usize>,
        probs: Vec<f32>,
        observed: Vec<bool>,
        panel_priors: Option<Arc<Vec<AllelePosteriors>>>,
        min_untyped_prior_mix: f32,
    ) -> Self {
        let mut uniform = Vec::new();
        if offsets.len() >= 2 {
            uniform.reserve(offsets.len() - 1);
            for m in 0..(offsets.len() - 1) {
                let start = offsets[m];
                let end = offsets[m + 1];
                let slice = probs.get(start..end).unwrap_or(&[]);
                let is_observed = observed.get(m).copied().unwrap_or(false);
                if is_observed {
                    uniform.push(is_uniform_probs(slice));
                } else {
                    // Untyped markers are structurally transition-only regardless of
                    // floating-point representation of allele probabilities.
                    uniform.push(true);
                }
            }
        }
        Self {
            offsets,
            probs,
            uniform,
            observed,
            marker_error_rates: None,
            panel_priors,
            min_untyped_prior_mix: min_untyped_prior_mix.clamp(0.0, 0.9),
        }
    }

    #[inline]
    pub fn probs_for_marker(&self, marker_idx: usize) -> &[f32] {
        let start = self.offsets[marker_idx];
        let end = self.offsets[marker_idx + 1];
        &self.probs[start..end]
    }

    /// Returns raw per-marker probabilities wrapped as trusted input.
    /// This does not normalize.
    #[inline]
    fn probs_for_marker_trusted(&self, marker_idx: usize) -> AlleleProbsView<'_> {
        AlleleProbsView::from_trusted(self.probs_for_marker(marker_idx))
    }

    #[inline]
    pub fn n_markers(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    #[inline]
    pub fn is_uniform_marker(&self, marker_idx: usize) -> bool {
        self.uniform.get(marker_idx).copied().unwrap_or(true)
    }

    #[inline]
    pub fn is_observed_marker(&self, marker_idx: usize) -> bool {
        self.observed.get(marker_idx).copied().unwrap_or(false)
    }

    #[inline]
    pub fn is_untyped_uniform_marker(&self, marker_idx: usize) -> bool {
        self.is_uniform_marker(marker_idx) && !self.is_observed_marker(marker_idx)
    }

    #[inline]
    pub fn has_untyped_markers(&self) -> bool {
        self.observed.iter().any(|&obs| !obs)
    }

    #[inline]
    pub fn panel_priors(&self) -> Option<&[AllelePosteriors]> {
        self.panel_priors.as_deref().map(|v| v.as_slice())
    }

    #[inline]
    pub fn set_marker_error_rates(&mut self, marker_error_rates: Vec<f32>) {
        if marker_error_rates.len() == self.n_markers() {
            self.marker_error_rates = Some(marker_error_rates);
        } else {
            self.marker_error_rates = None;
        }
    }

    #[inline]
    pub fn marker_error_rate(&self, marker_idx: usize) -> Option<f32> {
        self.marker_error_rates
            .as_ref()
            .and_then(|v| v.get(marker_idx).copied())
            .map(|v| v.clamp(1e-6, 0.5))
    }

    #[inline]
    pub fn min_untyped_prior_mix(&self) -> f32 {
        self.min_untyped_prior_mix
    }
}

/// Workspace for per-haplotype imputation HMM.
pub struct ImputeWorkspace {
    pub fwd: Vec<f32>,
    pub bwd: Vec<f32>,
    pub emissions: Vec<f32>,
    pub fwd_checkpoints: Vec<f32>,
    pub fwd_scales: Vec<f32>,
    pub weights: Vec<f32>,
    pub state_alleles: Vec<u8>,
    pub state_patterns: Vec<u16>,
    pub pattern_emissions: Vec<f32>,
    pub allele_probs: Vec<f32>,
    smoothing_prior_counts: Vec<f32>,
    state_posterior_scratch: Vec<f32>,
    boundary_fb_products: Vec<f32>,
    allele_prior_scratch: Vec<f32>,
    dict_pattern_alleles: Vec<u8>,
    emission_by_allele: Vec<f32>,
    pattern_id_cache: Vec<PatternIdCacheEntry>,
    pattern_id_cache_capacity: usize,
    nearest_obs_fwd: Vec<f32>,
    nearest_obs_bwd: Vec<f32>,
    pub(crate) nearest_obs_retain: Vec<f32>,
    affine_window_cache: Option<AffineWindowCache>,
    pattern_sum_f: Vec<f32>,
    pattern_sum_b: Vec<f32>,
    pattern_sum_fb: Vec<f32>,
    pattern_state_count: Vec<f32>,
    active_states: usize,
    active_markers: usize,
}

struct PatternIdCacheEntry {
    marker_key: usize,
    state_ptr: usize,
    active_states: usize,
    patterns: Vec<u16>,
}

struct PatternCacheView {
    ptr: *const u16,
    len: usize,
}

#[derive(Clone)]
struct AffineBlockCoeffs {
    block_start: usize,
    block_end: usize,
    fwd_a: Vec<f64>,
    fwd_b: Vec<f64>,
    bwd_a: Vec<f64>,
    bwd_b_coeff: Vec<f64>,
}

#[derive(Clone)]
struct AffineWindowCache {
    active_states: usize,
    transition_haps: usize,
    transition_lambda: f32,
    active_markers: usize,
    recomb_hash: u64,
    checkpoint_markers: Vec<usize>,
    by_checkpoint: Vec<Option<Arc<AffineBlockCoeffs>>>,
}

struct RefAlleles<'a> {
    slice: &'a [u8],
}

impl<'a> RefAlleles<'a> {
    #[inline]
    fn get(&self, idx: usize) -> u8 {
        self.slice[idx]
    }
}

struct SeqPatternAlleles<'a> {
    seq_alleles: &'a [u8],
    state_patterns: &'a [u16],
}

struct DictPatternAlleles<'a> {
    pattern_alleles: &'a [u8],
    state_patterns: &'a [u16],
}

/// Reference haplotype count metadata for imputation HMM transitions.
pub struct RefAlleleFreqs {
    n_ref_haps: usize,
}

impl RefAlleleFreqs {
    pub fn new(ref_columns: &[GenotypeColumn]) -> Self {
        let n_ref_haps = ref_columns.first().map(|c| c.n_haplotypes()).unwrap_or(0);
        Self { n_ref_haps }
    }

    #[inline]
    pub fn n_ref_haps(&self) -> usize {
        self.n_ref_haps
    }
}

#[derive(Clone, Copy)]
struct AlleleProbsView<'a> {
    probs: &'a [f32],
}

impl<'a> AlleleProbsView<'a> {
    #[inline]
    fn as_slice(self) -> &'a [f32] {
        self.probs
    }

    #[inline]
    fn len(self) -> usize {
        self.probs.len()
    }

    #[inline]
    fn get(self, idx: usize) -> Option<f32> {
        self.probs.get(idx).copied()
    }

    #[inline]
    fn is_empty(self) -> bool {
        self.probs.is_empty()
    }

    #[inline]
    fn from_trusted(slice: &'a [f32]) -> Self {
        Self { probs: slice }
    }
}

#[inline]
fn hash_recomb_slice(p_recomb: &[f32]) -> u64 {
    // FNV-1a over exact float bit patterns.
    let mut h: u64 = 0xcbf29ce484222325;
    for &x in p_recomb {
        h ^= x.to_bits() as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

impl ImputeWorkspace {
    pub fn new(n_states: usize, n_markers: usize) -> Self {
        Self {
            fwd: vec![0.0; n_states],
            bwd: vec![1.0; n_states],
            emissions: vec![1.0; n_states],
            fwd_checkpoints: Vec::new(),
            fwd_scales: vec![1.0; n_markers],
            weights: vec![1.0; n_states],
            state_alleles: vec![AlleleCode::MISSING.raw(); n_states],
            state_patterns: vec![0u16; n_states],
            pattern_emissions: Vec::new(),
            allele_probs: Vec::new(),
            smoothing_prior_counts: Vec::new(),
            state_posterior_scratch: Vec::new(),
            boundary_fb_products: Vec::new(),
            allele_prior_scratch: Vec::new(),
            dict_pattern_alleles: Vec::new(),
            emission_by_allele: Vec::new(),
            pattern_id_cache: Vec::new(),
            pattern_id_cache_capacity: 8,
            nearest_obs_fwd: Vec::new(),
            nearest_obs_bwd: Vec::new(),
            nearest_obs_retain: Vec::new(),
            affine_window_cache: None,
            pattern_sum_f: Vec::new(),
            pattern_sum_b: Vec::new(),
            pattern_sum_fb: Vec::new(),
            pattern_state_count: Vec::new(),
            active_states: n_states,
            active_markers: n_markers,
        }
    }

    pub fn resize(&mut self, n_states: usize, n_markers: usize) {
        if self.fwd.len() < n_states {
            self.fwd.resize(n_states, 0.0);
            self.bwd.resize(n_states, 1.0);
            self.emissions.resize(n_states, 1.0);
            self.weights.resize(n_states, 1.0);
        }
        if self.weights.len() < n_states {
            self.weights.resize(n_states, 1.0);
        }
        if self.state_alleles.len() < n_states {
            self.state_alleles
                .resize(n_states, AlleleCode::MISSING.raw());
        }
        if self.state_patterns.len() < n_states {
            self.state_patterns.resize(n_states, 0);
        }
        if self.fwd_scales.len() < n_markers {
            self.fwd_scales.resize(n_markers, 1.0);
        }
        if self.pattern_id_cache_capacity == 0 {
            self.pattern_id_cache_capacity = 8;
        }
        self.active_states = n_states;
        self.active_markers = n_markers;
    }

    pub fn ensure_typed_checkpoints(&mut self, n_states: usize, n_checkpoints: usize) {
        let want = n_states.max(1).saturating_mul(n_checkpoints.max(1));
        if self.fwd_checkpoints.len() < want {
            self.fwd_checkpoints.resize(want, 0.0);
        }
    }

    #[inline]
    fn store_checkpoint(&mut self, cp: CheckpointIx, n_states: usize) {
        let off = cp.fwd_offset(n_states);
        self.fwd_checkpoints[off..off + n_states].copy_from_slice(&self.fwd[..n_states]);
    }

    #[inline]
    fn load_checkpoint(&mut self, cp: CheckpointIx, n_states: usize) {
        let off = cp.fwd_offset(n_states);
        self.fwd[..n_states].copy_from_slice(&self.fwd_checkpoints[off..off + n_states]);
    }

    fn ensure_affine_window_cache(
        &mut self,
        p_recomb: &[f32],
        checkpoint_grid: &CheckpointGrid,
        active_states: usize,
        transition_haps: usize,
        transition_lambda: f32,
        active_markers: usize,
    ) {
        let recomb_hash = hash_recomb_slice(p_recomb);
        let checkpoint_markers: Vec<usize> = checkpoint_grid
            .markers
            .iter()
            .map(|m| m.as_usize())
            .collect();
        let valid = self
            .affine_window_cache
            .as_ref()
            .map(|cache| {
                cache.active_states == active_states
                    && cache.transition_haps == transition_haps
                    && cache.transition_lambda.to_bits() == transition_lambda.to_bits()
                    && cache.active_markers == active_markers
                    && cache.recomb_hash == recomb_hash
                    && cache.checkpoint_markers == checkpoint_markers
            })
            .unwrap_or(false);
        if valid {
            return;
        }

        let mut by_checkpoint: Vec<Option<Arc<AffineBlockCoeffs>>> =
            vec![None; checkpoint_grid.len()];
        for cp_idx in checkpoint_grid.rev_indices() {
            let block = checkpoint_grid.block_view(cp_idx, active_markers);
            let block_len = block.len();
            if block_len <= 1 {
                continue;
            }
            let block_start = block.start_usize();
            let block_end = block.end_usize();
            let mut fwd_a = vec![0.0f64; block_len];
            let mut fwd_b = vec![0.0f64; block_len];
            let mut bwd_a = vec![0.0f64; block_len];
            let mut bwd_b_coeff = vec![0.0f64; block_len];
            fill_fwd_affine_coeffs(
                &mut fwd_a,
                &mut fwd_b,
                p_recomb,
                block_start,
                block_end,
                active_states,
                transition_haps,
                transition_lambda,
            );
            fill_bwd_affine_coeffs(
                &mut bwd_a,
                &mut bwd_b_coeff,
                p_recomb,
                block_start,
                block_end,
                active_states,
                transition_haps,
                transition_lambda,
            );
            by_checkpoint[cp_idx.as_usize()] = Some(Arc::new(AffineBlockCoeffs {
                block_start,
                block_end,
                fwd_a,
                fwd_b,
                bwd_a,
                bwd_b_coeff,
            }));
        }
        self.affine_window_cache = Some(AffineWindowCache {
            active_states,
            transition_haps,
            transition_lambda,
            active_markers,
            recomb_hash,
            checkpoint_markers,
            by_checkpoint,
        });
    }

    #[inline]
    pub fn ensure_pattern_sums(&mut self, n_patterns: usize) {
        if self.pattern_sum_f.len() < n_patterns {
            self.pattern_sum_f.resize(n_patterns, 0.0);
        }
        if self.pattern_sum_b.len() < n_patterns {
            self.pattern_sum_b.resize(n_patterns, 0.0);
        }
        if self.pattern_sum_fb.len() < n_patterns {
            self.pattern_sum_fb.resize(n_patterns, 0.0);
        }
        if self.pattern_state_count.len() < n_patterns {
            self.pattern_state_count.resize(n_patterns, 0.0);
        }
    }

    #[inline]
    pub fn active_states(&self) -> usize {
        self.active_states
    }

    #[inline]
    pub fn active_markers(&self) -> usize {
        self.active_markers
    }

    #[inline]
    fn ensure_state_posterior_scratch(&mut self, n_states: usize) {
        if self.state_posterior_scratch.len() < n_states {
            self.state_posterior_scratch.resize(n_states, 0.0);
        }
    }

    #[inline]
    fn ensure_boundary_fb_products(&mut self, n_states: usize) {
        if self.boundary_fb_products.len() < n_states {
            self.boundary_fb_products.resize(n_states, 0.0);
        }
    }

    #[inline]
    fn ensure_smoothing_prior_counts(&mut self, n_alleles: usize) {
        if self.smoothing_prior_counts.len() < n_alleles {
            self.smoothing_prior_counts.resize(n_alleles, 0.0);
        }
    }

    #[inline]
    fn pattern_cache_lookup(
        &self,
        marker_key: usize,
        state_haps_ptr: *const RefHapId,
        active_states: usize,
    ) -> Option<PatternCacheView> {
        let state_ptr = state_haps_ptr as usize;
        self.pattern_id_cache
            .iter()
            .find(|entry| {
                entry.marker_key == marker_key
                    && entry.state_ptr == state_ptr
                    && entry.active_states == active_states
            })
            .map(|entry| PatternCacheView {
                ptr: entry.patterns.as_ptr(),
                len: entry.patterns.len(),
            })
    }

    #[inline]
    fn pattern_cache_insert(
        &mut self,
        marker_key: usize,
        state_haps_ptr: *const RefHapId,
        active_states: usize,
        patterns: &[u16],
    ) {
        let state_ptr = state_haps_ptr as usize;
        if let Some(entry) = self.pattern_id_cache.iter_mut().find(|entry| {
            entry.marker_key == marker_key
                && entry.state_ptr == state_ptr
                && entry.active_states == active_states
        }) {
            entry.patterns.clear();
            entry.patterns.extend_from_slice(patterns);
            return;
        }
        if self.pattern_id_cache.len() >= self.pattern_id_cache_capacity {
            self.pattern_id_cache.rotate_left(1);
            self.pattern_id_cache.pop();
        }
        self.pattern_id_cache.push(PatternIdCacheEntry {
            marker_key,
            state_ptr,
            active_states,
            patterns: patterns.to_vec(),
        });
    }

    #[inline]
    fn pattern_cache_clear(&mut self) {
        self.pattern_id_cache.clear();
    }
}

trait RefColumnLike {
    fn fill_ref_alleles(
        &self,
        state_haps: &[RefHapId],
        out: &mut [u8],
        dict_pattern_alleles: &mut Vec<u8>,
    );
}

impl<T: RefColumnLike + ?Sized> RefColumnLike for &T {
    #[inline]
    fn fill_ref_alleles(
        &self,
        state_haps: &[RefHapId],
        out: &mut [u8],
        dict_pattern_alleles: &mut Vec<u8>,
    ) {
        (*self).fill_ref_alleles(state_haps, out, dict_pattern_alleles);
    }
}

impl RefColumnLike for DenseColumn {
    #[inline]
    fn fill_ref_alleles(
        &self,
        state_haps: &[RefHapId],
        out: &mut [u8],
        dict_pattern_alleles: &mut Vec<u8>,
    ) {
        if dict_pattern_alleles.is_empty() {}
        if self.bits_per_allele() == 1 {
            let n_haps = self.n_haplotypes();
            let bits = self.bits_raw();
            let missing = self.missing_raw();
            let mut cached_word_idx = usize::MAX;
            let mut cached_bits_word = 0u64;
            let mut cached_missing_word = 0u64;
            for (i, hap) in state_haps.iter().enumerate() {
                let idx = hap.as_usize();
                if idx >= n_haps {
                    out[i] = GenotypeColumn::MISSING_ALLELE;
                    continue;
                }
                let word_idx = idx >> 6;
                let bit_idx = idx & 63;
                if word_idx != cached_word_idx {
                    cached_word_idx = word_idx;
                    cached_bits_word = if word_idx < bits.len() {
                        bits[word_idx]
                    } else {
                        0
                    };
                    cached_missing_word = if word_idx < missing.len() {
                        missing[word_idx]
                    } else {
                        0
                    };
                }
                if ((cached_missing_word >> bit_idx) & 1) != 0 {
                    out[i] = AlleleCode::MISSING.raw();
                    continue;
                }
                out[i] = ((cached_bits_word >> bit_idx) & 1) as u8;
            }
        } else {
            let n_haps = self.n_haplotypes();
            for (i, hap) in state_haps.iter().enumerate() {
                let idx = hap.as_usize();
                if idx >= n_haps {
                    out[i] = GenotypeColumn::MISSING_ALLELE;
                    continue;
                }
                out[i] = self.get_ref(*hap);
            }
        }
    }
}

impl RefColumnLike for SparseColumn {
    #[inline]
    fn fill_ref_alleles(
        &self,
        state_haps: &[RefHapId],
        out: &mut [u8],
        dict_pattern_alleles: &mut Vec<u8>,
    ) {
        if dict_pattern_alleles.is_empty() {}
        for (i, hap) in state_haps.iter().enumerate() {
            out[i] = self.get(HapIdx::new(hap.as_u32()));
        }
    }
}

impl RefColumnLike for SeqCodedColumn {
    #[inline]
    fn fill_ref_alleles(
        &self,
        state_haps: &[RefHapId],
        out: &mut [u8],
        dict_pattern_alleles: &mut Vec<u8>,
    ) {
        if dict_pattern_alleles.is_empty() {}
        let hap_to_seq = self.hap_to_seq();
        let seq_alleles = self.seq_alleles();
        for (i, hap) in state_haps.iter().enumerate() {
            let seq_idx = hap_to_seq[hap.as_usize()] as usize;
            out[i] = seq_alleles[seq_idx];
        }
    }
}

struct DictColRef<'a> {
    col: &'a DictionaryColumn,
    offset: usize,
}

#[inline]
fn seqcoded_col(col: &GenotypeColumn) -> &SeqCodedColumn {
    match col {
        GenotypeColumn::SeqCoded(c) => c,
        _ => unreachable!("SeqCoded-only dispatch violated"),
    }
}

#[inline]
fn dict_col_ref(col: &GenotypeColumn) -> DictColRef<'_> {
    match col {
        GenotypeColumn::Dictionary(c, offset) => DictColRef {
            col: c.as_ref(),
            offset: *offset,
        },
        _ => unreachable!("Dictionary-only dispatch violated"),
    }
}

impl RefColumnLike for DictColRef<'_> {
    #[inline]
    fn fill_ref_alleles(
        &self,
        state_haps: &[RefHapId],
        out: &mut [u8],
        dict_pattern_alleles: &mut Vec<u8>,
    ) {
        let n_patterns = self.col.n_patterns();
        if dict_pattern_alleles.len() < n_patterns {
            dict_pattern_alleles.resize(n_patterns, 0);
        }
        for pattern_idx in 0..n_patterns {
            dict_pattern_alleles[pattern_idx] = self.col.pattern_allele(self.offset, pattern_idx);
        }
        for (i, hap) in state_haps.iter().enumerate() {
            let pattern_idx = self.col.hap_pattern_idx(*hap);
            out[i] = dict_pattern_alleles[pattern_idx];
        }
    }
}

impl RefColumnLike for GenotypeColumn {
    #[inline]
    fn fill_ref_alleles(
        &self,
        state_haps: &[RefHapId],
        out: &mut [u8],
        dict_pattern_alleles: &mut Vec<u8>,
    ) {
        match self {
            GenotypeColumn::Dense(c) => {
                c.fill_ref_alleles(state_haps, out, dict_pattern_alleles);
            }
            GenotypeColumn::Sparse(c) => {
                c.fill_ref_alleles(state_haps, out, dict_pattern_alleles);
            }
            GenotypeColumn::Dictionary(c, offset) => {
                let col = DictColRef {
                    col: c.as_ref(),
                    offset: *offset,
                };
                col.fill_ref_alleles(state_haps, out, dict_pattern_alleles);
            }
            GenotypeColumn::SeqCoded(c) => {
                c.fill_ref_alleles(state_haps, out, dict_pattern_alleles);
            }
        }
    }
}

#[inline]
fn refresh_ref_alleles<'a, C: RefColumnLike + ?Sized>(
    col: &C,
    state_haps: &[RefHapId],
    state_alleles: &'a mut [u8],
    dict_pattern_alleles: &mut Vec<u8>,
) -> RefAlleles<'a> {
    col.fill_ref_alleles(state_haps, state_alleles, dict_pattern_alleles);
    // Invariant: RefAlleles always views state_alleles (never dict_pattern_alleles),
    // so callers may repurpose dict_pattern_alleles after this returns.
    RefAlleles {
        slice: state_alleles,
    }
}

#[inline]
fn refresh_seq_patterns<'a>(
    col: &'a SeqCodedColumn,
    last_hap_ptr: &mut *const u16,
    state_haps: &[RefHapId],
    state_patterns: &'a mut [u16],
) -> SeqPatternAlleles<'a> {
    let hap_to_seq = col.hap_to_seq();
    let hap_ptr = hap_to_seq.as_ptr();
    if hap_ptr != *last_hap_ptr {
        fill_state_patterns_seqcoded(hap_to_seq, state_haps, state_patterns);
        *last_hap_ptr = hap_ptr;
    }
    let seq_alleles = col.seq_alleles();
    SeqPatternAlleles {
        seq_alleles,
        state_patterns,
    }
}

#[inline]
fn refresh_dict_patterns<'a>(
    col: &DictColRef<'_>,
    last_dict_ptr: &mut *const DictionaryColumn,
    state_haps: &[RefHapId],
    state_patterns: &'a mut [u16],
    dict_pattern_alleles: &'a mut Vec<u8>,
) -> DictPatternAlleles<'a> {
    let dict_ptr = col.col as *const DictionaryColumn;
    if dict_ptr != *last_dict_ptr {
        fill_state_patterns_dict(col.col, state_haps, state_patterns);
        *last_dict_ptr = dict_ptr;
    }
    let n_patterns = col.col.n_patterns();
    if dict_pattern_alleles.len() < n_patterns {
        dict_pattern_alleles.resize(n_patterns, 0);
    }
    for pattern_idx in 0..n_patterns {
        dict_pattern_alleles[pattern_idx] = col.col.pattern_allele(col.offset, pattern_idx);
    }
    DictPatternAlleles {
        pattern_alleles: &dict_pattern_alleles[..n_patterns],
        state_patterns,
    }
}

#[inline]
fn fill_emissions(
    ref_alleles: &RefAlleles<'_>,
    target_probs: AlleleProbsView<'_>,
    error_rate: f32,
    emission_by_allele: &mut Vec<f32>,
    emissions: &mut [f32],
) {
    if target_probs.is_empty() {
        emissions.fill(1.0);
        return;
    }

    let n_alleles = target_probs.len();
    let mismatch_prob = if n_alleles > 1 {
        error_rate / (n_alleles as f32 - 1.0)
    } else {
        error_rate
    };
    let match_prob = 1.0 - error_rate;
    let missing_prob = missing_emission_prob(target_probs, match_prob, mismatch_prob);

    if emission_by_allele.len() < n_alleles {
        emission_by_allele.resize(n_alleles, 1.0);
    }
    for i in 0..n_alleles {
        let p_match = target_probs.get(i).unwrap_or(0.0);
        emission_by_allele[i] = mismatch_prob + (match_prob - mismatch_prob) * p_match;
    }

    for (i, &ref_allele) in ref_alleles.slice.iter().enumerate() {
        if AlleleCode::from_raw(ref_allele).is_missing() {
            emissions[i] = missing_prob;
            continue;
        }
        let idx = ref_allele as usize;
        if idx < n_alleles {
            emissions[i] = emission_by_allele[idx];
        } else {
            // Out-of-domain allele index must be treated as a mismatch under
            // the current emission model, not as neutral missingness.
            emissions[i] = mismatch_prob.max(1e-30);
        }
    }
}

#[inline]
fn fill_pattern_emissions(
    pattern_alleles: &[u8],
    target_probs: AlleleProbsView<'_>,
    error_rate: f32,
    emission_by_allele: &mut Vec<f32>,
    pattern_emissions: &mut Vec<f32>,
) -> f32 {
    if target_probs.is_empty() {
        pattern_emissions.resize(pattern_alleles.len(), 1.0);
        pattern_emissions.fill(1.0);
        return 0.0;
    }
    let n_alleles = target_probs.len();
    let mismatch_prob = if n_alleles > 1 {
        error_rate / (n_alleles as f32 - 1.0)
    } else {
        error_rate
    };
    let match_prob = 1.0 - error_rate;
    let missing_prob = missing_emission_prob(target_probs, match_prob, mismatch_prob);
    if emission_by_allele.len() < n_alleles {
        emission_by_allele.resize(n_alleles, 1.0);
    }
    for i in 0..n_alleles {
        let p_match = target_probs.get(i).unwrap_or(0.0);
        emission_by_allele[i] = mismatch_prob + (match_prob - mismatch_prob) * p_match;
    }
    if pattern_emissions.len() < pattern_alleles.len() {
        pattern_emissions.resize(pattern_alleles.len(), 1.0);
    }
    for (i, &allele) in pattern_alleles.iter().enumerate() {
        if AlleleCode::from_raw(allele).is_missing() {
            pattern_emissions[i] = missing_prob;
        } else {
            let idx = allele as usize;
            if idx < emission_by_allele.len() {
                pattern_emissions[i] = emission_by_allele[idx];
            } else {
                // Unknown allele index must be treated as mismatch likelihood.
                pattern_emissions[i] = mismatch_prob.max(1e-30);
            }
        }
    }
    mismatch_prob
}

#[inline]
fn missing_emission_prob(
    target_probs: AlleleProbsView<'_>,
    match_prob: f32,
    mismatch_prob: f32,
) -> f32 {
    let n = target_probs.len();
    if n == 0 {
        return 1.0;
    }
    let mut sum = 0.0f32;
    let mut sum_sq = 0.0f32;
    for i in 0..n {
        let p = target_probs.get(i).unwrap_or(0.0).max(0.0);
        sum += p;
        sum_sq += p * p;
    }
    let concentration = if sum > 0.0 {
        (sum_sq / (sum * sum)).clamp(0.0, 1.0)
    } else {
        1.0 / n as f32
    };
    mismatch_prob + (match_prob - mismatch_prob) * concentration
}

#[inline]
fn normalize_probs(probs: &mut [f32]) {
    let mut sum = 0.0f32;
    for p in probs.iter() {
        if p.is_finite() && *p > 0.0 {
            sum += *p;
        }
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for p in probs.iter_mut() {
            *p = (*p * inv).max(0.0);
        }
    }
}

#[inline]
fn adjusted_recomb_rate(recomb_rate: f32) -> f32 {
    recomb_rate.clamp(0.0, 1.0)
}

#[inline]
fn marker_recomb_rate(p_recomb: &[f32], marker_idx: usize) -> f32 {
    let rate = p_recomb.get(marker_idx).copied().unwrap_or(0.0);
    if marker_idx == 0 {
        adjusted_recomb_rate(rate)
    } else {
        adjusted_recomb_rate(rate)
    }
}

#[inline]
fn recomb_lambda_from_p(p: f32) -> f32 {
    // p = 1 - exp(-lambda) => lambda = -ln(1-p)
    // Clamp for numerical stability.
    let q = (1.0 - p.clamp(0.0, 1.0 - 1e-12)).max(1e-12);
    -q.ln()
}

pub(crate) fn compute_nearest_observed_lambda(
    ws: &mut ImputeWorkspace,
    target_probs: &TargetAlleleProbs,
    p_recomb: &[f32],
    smoothing_cluster_cm: f32,
) {
    const BASE_CLUSTER_CM: f32 = 0.005;
    let n = target_probs.n_markers();
    if ws.nearest_obs_fwd.len() < n {
        ws.nearest_obs_fwd.resize(n, f32::INFINITY);
    }
    if ws.nearest_obs_bwd.len() < n {
        ws.nearest_obs_bwd.resize(n, f32::INFINITY);
    }
    if ws.nearest_obs_retain.len() < n {
        ws.nearest_obs_retain.resize(n, 0.0);
    }
    ws.nearest_obs_fwd[..n].fill(f32::INFINITY);
    ws.nearest_obs_bwd[..n].fill(f32::INFINITY);
    if n == 0 {
        return;
    }

    let mut dist = f32::INFINITY;
    for m in 0..n {
        if target_probs.is_observed_marker(m) {
            dist = 0.0;
        } else if m > 0 && dist.is_finite() {
            dist += recomb_lambda_from_p(marker_recomb_rate(p_recomb, m));
        }
        ws.nearest_obs_fwd[m] = dist;
    }

    dist = f32::INFINITY;
    for m_rev in (0..n).rev() {
        if target_probs.is_observed_marker(m_rev) {
            dist = 0.0;
        } else if m_rev + 1 < n && dist.is_finite() {
            let next = m_rev + 1;
            dist += recomb_lambda_from_p(marker_recomb_rate(p_recomb, next));
        }
        ws.nearest_obs_bwd[m_rev] = dist;
    }

    // Map-distance retain used for output-regularization decisions.
    //
    // Interpretation:
    // - nearest_obs_fwd/bwd accumulate recombination hazard lambda from typed
    //   anchors to marker m using lambda = -ln(1-p_recomb).
    // - side retain is r = exp(-lambda), i.e. probability-like survival of
    //   anchor influence under a Poisson recombination intuition.
    //
    // This is not "approximation error"; it is physical map-distance decay.
    // approximation/truncation diagnostics are handled separately downstream.
    //
    // cluster_scale is a global calibration term for map-distance hazard.
    let cluster_scale = (BASE_CLUSTER_CM / smoothing_cluster_cm.max(1e-6)).max(0.0);
    const MIN_SAME_POS_LAMBDA: f32 = 0.05;
    for m in 0..n {
        let left = ws.nearest_obs_fwd[m];
        let right = ws.nearest_obs_bwd[m];
        let observed = target_probs.is_observed_marker(m);
        let mut left_lambda = if left.is_finite() {
            (left * cluster_scale).max(0.0)
        } else {
            f32::INFINITY
        };
        let mut right_lambda = if right.is_finite() {
            (right * cluster_scale).max(0.0)
        } else {
            f32::INFINITY
        };

        // Same-position untyped edge case:
        // avoid literal retain=1.0 from zero hazard at untyped markers that are
        // colocated with typed anchors.
        if !observed {
            if left_lambda == 0.0 {
                left_lambda = MIN_SAME_POS_LAMBDA;
            }
            if right_lambda == 0.0 {
                right_lambda = MIN_SAME_POS_LAMBDA;
            }
        }

        let left_retain = if left_lambda.is_finite() {
            (-left_lambda).exp()
        } else {
            0.0
        };
        let right_retain = if right_lambda.is_finite() {
            (-right_lambda).exp()
        } else {
            0.0
        };
        // Two-sided anchor union model:
        //   rL = exp(-lambdaL), rR = exp(-lambdaR)
        //   retain = P(left survives OR right survives)
        //          = 1 - (1-rL)(1-rR)
        //          = rL + rR - rL*rR
        //
        // Boundary behavior:
        // - at a typed anchor on one side, corresponding r=1 -> retain=1.
        // - if one side has no anchor (r=0), retain reduces to one-sided retain.
        ws.nearest_obs_retain[m] =
            (left_retain + right_retain - left_retain * right_retain).clamp(0.0, 1.0);
    }
}

#[inline]
fn smooth_allele_posteriors_subset(
    allele_probs: &mut [f32],
    subset_prior_probs: AlleleProbsView<'_>,
    nearest_obs_retain: f32,
    approximation_error: f32,
    untyped_uniform_marker: bool,
) {
    const MIN_RETAIN: f32 = 1e-4;
    if allele_probs.is_empty() {
        return;
    }
    let subset_prior_probs = subset_prior_probs.as_slice();
    if subset_prior_probs.len() != allele_probs.len() {
        return;
    }

    // Apply local-prior smoothing only on truly untyped/uniform markers.
    // Informative markers should be driven by likelihood, not AF pull.
    if !untyped_uniform_marker {
        return;
    }
    // Bayesian shrinkage toward a local prior pi on untyped/uniform markers.
    //
    // We estimate effective support from pi (not from current posterior p),
    // because p can be artificially collapsed by state truncation. Using p here
    // would suppress regularization exactly when it is most needed.
    let mut prior_sq_sum = 0.0f32;
    for &p in subset_prior_probs.iter() {
        let q = p.max(0.0);
        prior_sq_sum += q * q;
    }
    if prior_sq_sum <= 0.0 {
        return;
    }
    let max_effective = allele_probs.len().max(1) as f32;
    let effective_alleles = (1.0 / prior_sq_sum).clamp(1.0, max_effective);
    // Retain decomposition:
    //   dist_retain   = map-distance retain from typed anchors
    //   approx_retain = 1 - approximation_error
    //   retain        = dist_retain * approx_retain
    //
    // Multiplication encodes "both conditions must hold to trust p strongly":
    // even if distance retain is high, truncation/missing can still reduce trust.
    // even if approximation diagnostics look good, deep map distance still reduces trust.
    let dist_retain = nearest_obs_retain.clamp(MIN_RETAIN, 1.0);
    let approx_retain = (1.0 - approximation_error.clamp(0.0, 0.9999)).clamp(MIN_RETAIN, 1.0);
    let retain = (dist_retain * approx_retain).clamp(MIN_RETAIN, 1.0);
    // Entropy-aware confidence gating: when posterior entropy is much lower
    // than the local-prior entropy, the state subset is likely overconfident.
    // Increase smoothing in that regime to reduce sparse-subset collapse.
    let mut post_entropy = 0.0f32;
    let mut prior_entropy = 0.0f32;
    for (&post, &prior) in allele_probs.iter().zip(subset_prior_probs.iter()) {
        let p = post.clamp(0.0, 1.0);
        let q = prior.clamp(0.0, 1.0);
        if p > 0.0 {
            post_entropy -= p * p.ln();
        }
        if q > 0.0 {
            prior_entropy -= q * q.ln();
        }
    }
    let entropy_gap = (prior_entropy - post_entropy).max(0.0);
    let max_entropy = (allele_probs.len().max(2) as f32).ln().max(1e-6);
    let confidence_boost = (entropy_gap / max_entropy).clamp(0.0, 1.0);

    // Dirichlet-style pseudocount update:
    //   p'_a = (p_a + alpha * pi_a) / (1 + alpha)
    // where pi is the local donor-conditional prior.
    //
    // If alpha=(1-retain)/retain, posterior mass weight would be exactly retain:
    //   posterior coefficient = 1/(1+alpha) = retain.
    //
    // Here alpha is further scaled by:
    // - effective_alleles (pi diffuseness proxy),
    // - entropy-derived confidence_boost,
    // to prevent brittle overconfidence under sparse subset support.
    //
    // Deviation form (p = pi + delta):
    //   delta' = delta / (1 + alpha)
    // so this step is explicit shrinkage of deviation from pi.
    let base_mass = (effective_alleles * (1.0 - retain) / retain).max(0.0);
    // Keep entropy boost modest; large multipliers over-shrink rare ALT signal
    // when subset support is sparse.
    let prior_mass = base_mass * (1.0 + 0.5 * confidence_boost);
    if prior_mass <= 0.0 {
        return;
    }
    let denom = 1.0 + prior_mass;
    for (i, p) in allele_probs.iter_mut().enumerate() {
        let local_prior = subset_prior_probs.get(i).copied().unwrap_or(0.0).max(0.0);
        *p = (*p + prior_mass * local_prior) / denom;
    }
    normalize_probs(allele_probs);
}

#[inline]
fn apply_marker_prior_smoothing(
    allele_probs: &mut [f32],
    panel_priors: Option<&[AllelePosteriors]>,
    marker_idx: usize,
    smoothing_prior_counts: &mut [f32],
    smoothing_prior_total: f32,
    allele_prior_scratch: &mut Vec<f32>,
    probs: &[f32],
    nearest_obs_retain: f32,
    untyped_uniform_marker: bool,
    subset_total: f32,
    missing_ref_mass: f32,
    missing_ood_mass: f32,
    active_states: usize,
    panel_haps: usize,
    min_prior_mix: f32,
    warned_af_fallback: &mut bool,
    context: ImputeHmmContext,
) {
    // WARNING: Do NOT extend this function to non-uniform untyped markers.
    // PR #745 tried removing this guard and applying mild smoothing (0.25x mix,
    // cap 0.2) to all untyped markers, plus adding an anti-collapse regularizer
    // (35% panel blend) and 4x epsilon inflation for sparse windows. The triple
    // stacking caused Hellinger +0.002166 with zero R² gain — same over-
    // regularization failure mode as PR #746. Keep this guard: only uniform
    // untyped markers need panel-frequency smoothing.
    if !untyped_uniform_marker {
        return;
    }

    // Two-stage correction on untyped+uniform markers:
    // 1) optional panel blend (calibration floor + adaptive convex pull)
    // 2) local-prior pseudocount shrink (Dirichlet-style)
    //
    // This is intentionally NOT a single global convex mix:
    // stage (1) addresses panel calibration under missing/truncation;
    // stage (2) shrinks local posterior deviations using donor-conditional pi.

    // Prefer marker-local missing mass measured from the structural posterior
    // decomposition (represented vs missing-ref vs out-of-domain). Fall back to
    // the legacy active/panel sparsity proxy only if local mass accounting is
    // unavailable at this call site.
    let observed_total =
        subset_total.max(0.0) + missing_ref_mass.max(0.0) + missing_ood_mass.max(0.0);
    let observed_missing_mass = if observed_total > 0.0 {
        ((missing_ref_mass.max(0.0) + missing_ood_mass.max(0.0)) / observed_total).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let fallback_missing_mass = if panel_haps > 0 && active_states < panel_haps {
        let raw_ratio = ((panel_haps - active_states) as f32 / panel_haps as f32).clamp(0.0, 1.0);
        raw_ratio
    } else {
        0.0
    };
    let missing_mass = if observed_total > 0.0 {
        observed_missing_mass.max(fallback_missing_mass)
    } else {
        fallback_missing_mass
    };
    let floor_mix = min_prior_mix.clamp(0.0, 0.9);
    let dist_retain = nearest_obs_retain.clamp(0.0, 1.0);
    let dist_error = 1.0 - dist_retain;
    let active_ratio = if panel_haps > 0 {
        (active_states as f32 / panel_haps as f32).clamp(0.0, 1.0)
    } else {
        1.0
    };
    let sparsity_boost = (1.0 - active_ratio).powi(2);
    let truncation_error = (1.0 - active_ratio).clamp(0.0, 1.0);
    // Approximation error combines structural-missingness and truncation:
    //   approx_error = 1 - (1-missing_mass)*(1-truncation_error)
    // This term captures model/subset limitations, not map distance.
    let approximation_error =
        (1.0 - (1.0 - missing_mass) * (1.0 - truncation_error)).clamp(0.0, 0.9999);
    // Combine map and approximation uncertainty via independent-failure union:
    //   combined_error = 1 - (1-dist_error)*(1-approx_error)
    // so either axis can activate extra regularization.
    //
    // This keeps the decomposition explicit:
    // - distance governs physical information decay
    // - diagnostics govern approximation-risk inflation
    let combined_error =
        (1.0 - (1.0 - dist_error) * (1.0 - approximation_error)).clamp(0.0, 0.9999);
    // Conservative adaptive blend: panel priors should stabilize pathological
    // cases, not dominate local HMM evidence.
    let adaptive_panel_mix =
        (0.35 * missing_mass * combined_error * (1.0 + 0.75 * sparsity_boost)).clamp(0.0, 0.35);

    if let Some(panel) = panel_priors.and_then(|p| p.get(marker_idx)) {
        match panel {
            AllelePosteriors::Biallelic(p_alt) if allele_probs.len() == 2 => {
                let p_alt = p_alt.clamp(0.0, 1.0);
                let panel_probs = [1.0 - p_alt, p_alt];
                apply_adaptive_panel_blend(
                    allele_probs,
                    &panel_probs,
                    floor_mix,
                    adaptive_panel_mix,
                );
            }
            AllelePosteriors::Multiallelic(p) if p.len() == allele_probs.len() => {
                apply_adaptive_panel_blend(allele_probs, p, floor_mix, adaptive_panel_mix);
            }
            _ => {}
        }
    }

    let prior_probs = if smoothing_prior_total > 0.0 {
        let inv = 1.0 / smoothing_prior_total;
        for v in smoothing_prior_counts.iter_mut() {
            *v *= inv;
        }
        AlleleProbsView::from_trusted(smoothing_prior_counts)
    } else {
        if !*warned_af_fallback {
            eprintln!(
                "[warn] AF fallback in impute_hmm smoothing (no state prior): window={} sample={} hap={} marker={}",
                context.window_idx, context.sample_idx, context.hap_idx, marker_idx
            );
            *warned_af_fallback = true;
        }
        normalized_allele_prior(allele_prior_scratch, AlleleProbsView::from_trusted(probs))
    };
    smooth_allele_posteriors_subset(
        allele_probs,
        prior_probs,
        nearest_obs_retain,
        approximation_error,
        true,
    );
}

#[inline]
fn apply_adaptive_panel_blend(
    allele_probs: &mut [f32],
    panel_probs: &[f32],
    floor_mix: f32,
    adaptive_panel_mix: f32,
) {
    if allele_probs.len() != panel_probs.len() {
        return;
    }

    if floor_mix > 0.0 {
        // Floor step: enforce a small allele floor from panel probabilities so
        // extremely sparse subsets cannot assign hard zeros too early.
        let scaled_floor = floor_mix.clamp(0.0, 0.15);
        if scaled_floor > 0.0 {
            for (i, prob) in allele_probs.iter_mut().enumerate() {
                let panel_p = panel_probs.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                let floor = scaled_floor * panel_p;
                if *prob < floor {
                    *prob = floor;
                }
            }
            normalize_probs(allele_probs);
        }
    }

    if adaptive_panel_mix > 0.0 {
        // Jensen-Shannon disagreement quantifies mismatch between local subset
        // and panel prior. Larger mismatch increases blend strength, but blend
        // remains bounded by adaptive_panel_mix cap.
        let mut m_entropy = 0.0f32;
        let mut p_entropy = 0.0f32;
        let mut q_entropy = 0.0f32;
        for (i, &p_raw) in allele_probs.iter().enumerate() {
            let p = p_raw.clamp(0.0, 1.0);
            let q = panel_probs.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0);
            let m = 0.5 * (p + q);
            if p > 0.0 {
                p_entropy -= p * p.ln();
            }
            if q > 0.0 {
                q_entropy -= q * q.ln();
            }
            if m > 0.0 {
                m_entropy -= m * m.ln();
            }
        }
        let js_div = (m_entropy - 0.5 * (p_entropy + q_entropy)).max(0.0);
        let max_js = (2.0f32).ln();
        let disagreement = (js_div / max_js).clamp(0.0, 1.0);

        // Symmetric convex blend:
        //   p <- (1-w) * p + w * panel
        // applied after flooring to preserve calibration and normalization.
        let w = (adaptive_panel_mix * (1.0 + 0.5 * disagreement)).clamp(0.0, 0.45);
        let one_minus_w = 1.0 - w;
        for (i, prob) in allele_probs.iter_mut().enumerate() {
            let panel_p = panel_probs.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0);
            *prob = one_minus_w * *prob + w * panel_p;
        }
        normalize_probs(allele_probs);
    }
}

#[inline]
fn is_uniform_probs(probs: &[f32]) -> bool {
    if probs.len() <= 1 {
        return true;
    }
    let mut min = probs[0];
    let mut max = probs[0];
    if !min.is_finite() {
        return false;
    }
    for &v in probs.iter().skip(1) {
        if !v.is_finite() {
            return false;
        }
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    (max - min) <= 1e-6
}

#[inline]
fn build_uniform_mask(target_probs: &TargetAlleleProbs, n_markers: usize) -> MarkerMask<bool> {
    MarkerMask(
        (0..n_markers)
            .map(|m| target_probs.is_uniform_marker(m))
            .collect(),
    )
}

#[inline]
fn build_skip_untyped_mask(
    target_probs: &TargetAlleleProbs,
    nearest_obs_retain: &[f32],
    uniform_mask: &MarkerMask<bool>,
    use_prior_smoothing: bool,
) -> MarkerMask<bool> {
    // Skip mask behavior:
    // - applies only on final output emission for untyped+uniform markers.
    // - does NOT alter forward/backward recursion; state propagation is unchanged.
    //
    // So this is an output substitution optimization/calibration step:
    //   if retain < threshold, write panel prior for that marker.
    // It can change reported posterior at m, but does not break transition flow.
    let panel_priors = target_probs.panel_priors();
    let Some(panel) = panel_priors else {
        return MarkerMask(vec![false; uniform_mask.len()]);
    };
    MarkerMask(
        (0..uniform_mask.len())
            .map(|m| {
                let mx = MarkerIx::new(m);
                if !use_prior_smoothing || !uniform_mask[mx] || target_probs.is_observed_marker(m) {
                    return false;
                }
                if m >= panel.len() {
                    return false;
                }
                let retain = nearest_obs_retain
                    .get(m)
                    .copied()
                    .unwrap_or(0.0)
                    .clamp(0.0, 1.0);
                // Retain threshold for panel substitution on deep-untyped sites.
                // Keep this strict: substitution is a last resort when transition
                // information is essentially absent.
                retain < 0.0005
            })
            .collect(),
    )
}

#[inline]
fn write_panel_freq_posterior(
    dst: &mut AllelePosteriors,
    panel_priors: Option<&[AllelePosteriors]>,
    marker_idx: usize,
) {
    let Some(src) = panel_priors.and_then(|p| p.get(marker_idx)) else {
        *dst = AllelePosteriors::Biallelic(0.0);
        return;
    };
    match src {
        AllelePosteriors::Biallelic(p_alt) => {
            *dst = AllelePosteriors::Biallelic(*p_alt);
        }
        AllelePosteriors::Multiallelic(probs) => {
            // Arc clone is O(1) and avoids reallocating the PMF.
            *dst = AllelePosteriors::Multiallelic(std::sync::Arc::clone(probs));
        }
    }
}

#[inline]
fn build_checkpoint_markers(
    uniform_mask: &MarkerMask<bool>,
    prior_marker_idx: Option<usize>,
    n_markers: usize,
) -> CheckpointGrid {
    if n_markers == 0 {
        return CheckpointGrid {
            markers: Vec::new(),
        };
    }
    let mut markers = Vec::with_capacity(n_markers.min(4096));
    markers.push(MarkerIx::new(0));
    for m in 1..uniform_mask.len() {
        let uniform = uniform_mask[MarkerIx::new(m)];
        if !uniform {
            markers.push(MarkerIx::new(m));
        }
    }
    if let Some(pm) = prior_marker_idx {
        if pm < n_markers {
            let pm = MarkerIx::new(pm);
            match markers.binary_search(&pm) {
                Ok(_) => {}
                Err(ins) => markers.insert(ins, pm),
            }
        }
    }
    CheckpointGrid { markers }
}

#[inline]
fn normalized_allele_prior<'a>(
    out: &'a mut Vec<f32>,
    target_probs: AlleleProbsView<'_>,
) -> AlleleProbsView<'a> {
    let n = target_probs.len();
    if out.len() < n {
        out.resize(n, 0.0);
    }
    let prior = &mut out[..n];
    let mut sum = 0.0f32;
    for i in 0..n {
        let mut v = target_probs.get(i).unwrap_or(0.0);
        if !v.is_finite() || v < 0.0 {
            v = 0.0;
        }
        prior[i] = v;
        sum += v;
    }
    if sum <= 0.0 {
        let uniform = 1.0 / n.max(1) as f32;
        prior.fill(uniform);
        return AlleleProbsView { probs: prior };
    }
    let inv = 1.0 / sum;
    for p in prior.iter_mut() {
        *p *= inv;
    }
    AlleleProbsView { probs: prior }
}

#[inline]
fn structural_ood_dirichlet_alpha(prior_probs: &[f32]) -> f32 {
    if prior_probs.is_empty() {
        return 1.0;
    }
    // Structural concentration proxy: effective allele count n_eff = 1/sum(pi^2).
    // This keeps alpha small when prior is concentrated and larger when diffuse.
    let mut sum_sq = 0.0f64;
    for &p in prior_probs {
        let q = p.max(0.0) as f64;
        sum_sq += q * q;
    }
    if sum_sq <= 0.0 || !sum_sq.is_finite() {
        return 1.0;
    }
    let n_eff = (1.0 / sum_sq) as f32;
    n_eff.clamp(1.0, prior_probs.len().max(1) as f32)
}

#[inline]
fn scale_slice_in_place(values: &mut [f32], scale: f32) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if values.len() >= 8 && std::is_x86_feature_detected!("avx2") {
            unsafe {
                #[cfg(target_arch = "x86")]
                use std::arch::x86::*;
                #[cfg(target_arch = "x86_64")]
                use std::arch::x86_64::*;
                let scale_v = _mm256_set1_ps(scale);
                let mut i = 0usize;
                while i + 8 <= values.len() {
                    let v = _mm256_loadu_ps(values.as_ptr().add(i));
                    let out = _mm256_mul_ps(v, scale_v);
                    _mm256_storeu_ps(values.as_mut_ptr().add(i), out);
                    i += 8;
                }
                for x in &mut values[i..] {
                    *x *= scale;
                }
                return;
            }
        }
    }
    for x in values {
        *x *= scale;
    }
}

#[inline]
fn affine_blend_with_prior_in_place(
    values: &mut [f32],
    prior: &[f32],
    q_coeff: f32,
    pi_coeff: f32,
) {
    let n = values.len().min(prior.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if n >= 8 && std::is_x86_feature_detected!("avx2") {
            unsafe {
                #[cfg(target_arch = "x86")]
                use std::arch::x86::*;
                #[cfg(target_arch = "x86_64")]
                use std::arch::x86_64::*;
                let qv = _mm256_set1_ps(q_coeff);
                let piv = _mm256_set1_ps(pi_coeff);
                let mut i = 0usize;
                while i + 8 <= n {
                    let q = _mm256_loadu_ps(values.as_ptr().add(i));
                    let p = _mm256_loadu_ps(prior.as_ptr().add(i));
                    let out = _mm256_add_ps(_mm256_mul_ps(q, qv), _mm256_mul_ps(p, piv));
                    _mm256_storeu_ps(values.as_mut_ptr().add(i), out);
                    i += 8;
                }
                for i in i..n {
                    values[i] = values[i] * q_coeff + prior[i] * pi_coeff;
                }
                return;
            }
        }
    }
    for i in 0..n {
        values[i] = values[i] * q_coeff + prior[i] * pi_coeff;
    }
}

#[inline]
fn normalize_allele_posterior_structural_missing(
    allele_probs: &mut [f32],
    subset_total: f32,
    missing_ref_mass: f32,
    missing_ood_mass: f32,
    prior_scratch: &mut Vec<f32>,
    target_probs: AlleleProbsView<'_>,
) {
    // Structural posterior decomposition at one marker:
    //   p_i = (q_i + M_ref * rho_ref_i + M_ood * rho_ood_i) / (Q + M_ref + M_ood)
    // where:
    //   q_i      = represented-state mass on allele i
    //   Q        = sum_i q_i
    //   M_ref    = mass from reference-missing states (allele code u8::MAX)
    //   M_ood    = mass from out-of-domain allele codes
    //
    // Assumptions:
    // 1) Reference-missing states are MAR w.r.t. allele identity:
    //      rho_ref_i = q_i / Q
    // 2) Out-of-domain states use Dirichlet posterior predictive centered at prior pi:
    //      rho_ood_i = (q_i + alpha*pi_i) / (Q + alpha), alpha = 1
    //
    // Substituting yields an affine form used below:
    //   p_i = q_i * q_coeff + pi_i * pi_coeff
    //   q_coeff  = (1 + M_ref/Q + M_ood/(Q+alpha)) / (Q+M_ref+M_ood)
    //   pi_coeff = (M_ood*alpha/(Q+alpha)) / (Q+M_ref+M_ood)
    let subset = subset_total.max(0.0);
    let missing_ref = missing_ref_mass.max(0.0);
    let missing_ood = missing_ood_mass.max(0.0);
    let total = subset + missing_ref + missing_ood;
    if total <= 0.0 {
        let prior = normalized_allele_prior(prior_scratch, target_probs);
        allele_probs.copy_from_slice(prior.as_slice());
        return;
    }

    if subset <= 0.0 {
        // No represented-state evidence: unknown mass cannot inherit local shape.
        // Fall back to the target prior to avoid undefined q/Q terms.
        let prior = normalized_allele_prior(prior_scratch, target_probs);
        allele_probs.copy_from_slice(prior.as_slice());
        return;
    }

    let inv_total = 1.0 / total;
    if missing_ood > 0.0 {
        // Structural Dirichlet concentration alpha is derived from prior
        // concentration (effective allele count), not hand-tuned thresholds.
        // rho_ood = (q + alpha*pi)/(Q + alpha).
        let prior = normalized_allele_prior(prior_scratch, target_probs);
        let ood_dirichlet_alpha = structural_ood_dirichlet_alpha(prior.as_slice());
        let inv_subset = 1.0 / subset;
        let inv_rho = 1.0 / (subset + ood_dirichlet_alpha);
        let q_coeff = (1.0 + missing_ref * inv_subset + missing_ood * inv_rho) * inv_total;
        let pi_coeff = (missing_ood * ood_dirichlet_alpha * inv_rho) * inv_total;
        affine_blend_with_prior_in_place(allele_probs, prior.as_slice(), q_coeff, pi_coeff);
    } else {
        // No OOD mass: exact MAR redistribution for reference-missing states.
        let q_coeff = (1.0 + missing_ref / subset) * inv_total;
        scale_slice_in_place(allele_probs, q_coeff);
    }
}

#[inline]
fn subset_transition_params(
    recomb_rate: f32,
    active_states: usize,
    n_ref_haps: usize,
) -> (f32, f32) {
    // Transition model used in this imputation HMM interior:
    //
    // For a state vector x over K active states, one transition step is
    //   x'_i = stay * x_i + shift * sum_j x_j
    //
    // where (stay, shift) are the subset-conditioned Li-Stephens parameters.
    // This is an affine rank-1 update:
    //   x' = stay * x + shift * 1 * (sum x)
    //
    // The helper returns (stay, shift) with "exact K" subset semantics (no
    // clamping of K to panel size in this caller).
    if active_states == 0 {
        return (0.0, 0.0);
    }
    // Preserve historical imputation behavior: use exact active subset size
    // (no clamping of k). Recombination is still clamped in li_stephens.
    let (stay, shift) = subset_linear_exact_k(recomb_rate, active_states as f32, n_ref_haps);
    // Invariant for mass-preserving affine transition on K active states:
    //   stay + K*shift == 1
    // We keep this as a hard assert so any upstream transition regression
    // fails fast rather than silently biasing posteriors.
    assert!(
        ((stay + shift * active_states as f32) - 1.0).abs() < 1e-4
            || !stay.is_finite()
            || !shift.is_finite(),
        "subset transition mass drift: stay={} shift={} K={} stay+K*shift={}",
        stay,
        shift,
        active_states,
        stay + shift * active_states as f32
    );
    (stay, shift)
}

#[inline]
fn effective_recomb_rate(recomb_rate: f32, lambda: f32) -> f32 {
    let r = recomb_rate.clamp(0.0, 1.0);
    let lam = lambda.clamp(0.0, 1.0);
    (r * (1.0 - lam)).clamp(0.0, 1.0)
}

#[inline]
fn subset_transition_params_adaptive_q(
    recomb_rate: f32,
    active_states: usize,
    n_ref_haps: usize,
    lambda: f32,
) -> (f32, f32) {
    if active_states == 0 {
        return (0.0, 0.0);
    }
    // Fixed-lambda subset transition family used throughout imputation.
    //
    // Affine update over represented states j=1..K:
    //   x'_j = stay*x_j + shift*sum_i x_i
    //
    // Coefficients:
    //   rho   = lambda + (1-lambda)*K/N
    //   d     = (1-r) + r*rho
    //   stay  = ((1-r) + r*lambda) / d
    //   shift = (r*(1-lambda)/N) / d
    //
    // Derivation check (represented-state mass conservation):
    //   sum_j x'_j = (stay + K*shift) * sum_i x_i
    // and
    //   stay + K*shift
    //     = ((1-r)+r*lambda + r*(1-lambda)*K/N)/d
    //     = ((1-r)+r*rho)/d
    //     = 1.
    //
    // Note: diagonal/self transition probability in the implied matrix is
    // (stay + shift), not stay alone, because the rank-1 additive term also
    // contributes to the diagonal.
    //
    // Equivalent parameterization:
    //   r_eff = r*(1-lambda)
    // and then canonical subset coefficients at r_eff.
    let r_eff = effective_recomb_rate(recomb_rate, lambda);
    let (stay, shift) = subset_transition_params(r_eff, active_states, n_ref_haps);
    let k = active_states as f32;
    if cfg!(debug_assertions) {
        assert!(
            (stay + k * shift - 1.0).abs() < 1e-3 || !stay.is_finite() || !shift.is_finite(),
            "adaptive subset transition mass drift: stay={} shift={} K={} stay+K*shift={}",
            stay,
            shift,
            active_states,
            stay + k * shift
        );
    }
    (stay, shift)
}

#[inline]
fn transition_only_forward_update(
    fwd: &mut [f32],
    fwd_sum: f32,
    recomb_rate: f32,
    transition_haps: usize,
    transition_lambda: f32,
) -> f32 {
    if fwd.is_empty() {
        return 0.0;
    }
    if recomb_rate <= 0.0 {
        return fwd_sum;
    }
    let denom = fwd_sum.max(1e-30);
    let (stay_gap, shift) = subset_transition_params_adaptive_q(
        recomb_rate,
        fwd.len(),
        transition_haps,
        transition_lambda,
    );
    let scale = stay_gap / denom;
    let mut new_sum = 0.0f32;
    for v in fwd.iter_mut() {
        let t = scale.mul_add(*v, shift);
        *v = t;
        new_sum += t;
    }
    new_sum
}

#[inline]
fn transition_only_backward_update(
    bwd: &mut [f32],
    recomb_rate: f32,
    transition_haps: usize,
    bwd_sum: f32,
    transition_lambda: f32,
) -> f32 {
    if bwd.is_empty() || recomb_rate <= 0.0 {
        return bwd_sum;
    }
    let (stay_gap, shift_base) = subset_transition_params_adaptive_q(
        recomb_rate,
        bwd.len(),
        transition_haps,
        transition_lambda,
    );
    let shift = shift_base * bwd_sum;
    let scale = stay_gap;
    for v in bwd.iter_mut() {
        *v = scale.mul_add(*v, shift);
    }
    let n = bwd.len() as f32;
    (scale + shift_base * n).mul_add(bwd_sum, 0.0).max(1e-30)
}

#[inline]
fn batched_transition_forward(
    fwd: &mut [f32],
    p_recomb: &[f32],
    start: usize,
    end: usize,
    active_states: usize,
    transition_haps: usize,
    transition_lambda: f32,
) {
    if active_states == 0 || start >= end {
        return;
    }
    // Closed-form composition for a transition-only marker interval [start, end):
    //
    // Per marker t:
    //   x <- stay_t * x + shift_t
    //
    // Repeated composition keeps the same affine form:
    //   x <- a * x + b
    // with recurrence
    //   a <- a * stay_t
    //   b <- stay_t * b + shift_t
    //
    // Therefore, we can skip per-marker state updates and apply one affine map
    // to every state in the active set at the interval end.
    //
    // Note on normalization:
    // This driver keeps forward vectors normalized in the HMM recursion, so the
    // composed update is applied to a probability vector and re-normalized for
    // floating-point drift if needed.
    let mut a = 1.0f64;
    let mut b = 0.0f64;
    let mut touched = false;
    for m in start..end {
        let recomb_rate = marker_recomb_rate(p_recomb, m);
        if recomb_rate <= 0.0 {
            continue;
        }
        touched = true;
        let (stay, shift) = subset_transition_params_adaptive_q(
            recomb_rate,
            active_states,
            transition_haps,
            transition_lambda,
        );
        let stay = stay as f64;
        let shift = shift as f64;
        a *= stay;
        b = stay.mul_add(b, shift);
    }
    if !touched {
        return;
    }
    // Keep the true composed additive coefficient from the recurrence:
    //   x' = stay * x + shift
    // composed over markers as:
    //   a <- a * stay
    //   b <- stay * b + shift
    // Do not replace with (1-a)/k shortcut.
    let a = a as f32;
    let b = b as f32;
    let mut sum = 0.0f32;
    for v in fwd.iter_mut().take(active_states) {
        *v = a.mul_add(*v, b);
        sum += *v;
    }
    let sum = sum.max(1e-30);
    if (sum - 1.0).abs() > 1e-4 {
        let inv = 1.0 / sum;
        for v in fwd.iter_mut().take(active_states) {
            *v *= inv;
        }
    }
}

#[inline]
fn batched_transition_backward(
    bwd: &mut [f32],
    bwd_sum: f32,
    p_recomb: &[f32],
    start: usize,
    end: usize,
    active_states: usize,
    transition_haps: usize,
    transition_lambda: f32,
) -> f32 {
    if active_states == 0 || start >= end {
        return bwd_sum;
    }
    // Backward counterpart of forward affine composition on [start, end):
    //
    // Per reverse step:
    //   beta <- stay_t * beta + shift_t * sum(beta_right)
    //
    // With bwd_sum passed explicitly, we apply the composed map:
    //   beta <- a * beta + add
    // where
    //   a   = product_t stay_t
    //   add = b_coeff * bwd_sum
    // and b_coeff follows the same affine recurrence as forward composition.
    let mut a = 1.0f64;
    let mut b_coeff = 0.0f64;
    let mut touched = false;
    for m in (start..end).rev() {
        let recomb_rate = marker_recomb_rate(p_recomb, m);
        if recomb_rate <= 0.0 {
            continue;
        }
        touched = true;
        let (stay, shift) = subset_transition_params_adaptive_q(
            recomb_rate,
            active_states,
            transition_haps,
            transition_lambda,
        );
        let stay = stay as f64;
        let shift = shift as f64;
        a *= stay;
        b_coeff = stay.mul_add(b_coeff, shift);
    }
    if !touched {
        return bwd_sum;
    }
    // Keep the true composed additive coefficient from the recurrence:
    //   x' = stay * x + shift
    // composed over markers as:
    //   a <- a * stay
    //   b <- stay * b + shift
    // Do not replace with (1-a)/k shortcut.
    let a = a as f32;
    let add = b_coeff as f32 * bwd_sum;
    for v in bwd.iter_mut().take(active_states) {
        *v = a.mul_add(*v, add);
    }
    bwd_sum
}

#[inline]
fn fill_bwd_affine_coeffs(
    a_out: &mut [f64],
    b_out: &mut [f64],
    p_recomb: &[f32],
    block_start: usize,
    block_end: usize,
    active_states: usize,
    transition_haps: usize,
    transition_lambda: f32,
) {
    if block_start + 1 >= block_end {
        return;
    }
    // Derive per-marker backward affine coefficients for one checkpoint block.
    //
    // For interior marker m in (block_start, block_end), write
    //   B_m(i) = a_m * B_right(i) + c_m * sum(B_right)
    // where B_right is the backward vector at right boundary (block_end - 1).
    //
    // Recurrence (walking right->left):
    //   a_new = stay * a
    //   c_new = stay * c + shift * s
    //   s_new = (stay + K*shift) * s
    //
    // s tracks sum(B_m) / sum(B_right), needed because the additive term
    // depends on the running backward mass.
    let right = block_end - 1;
    let mut a = 1.0f64;
    let mut c = 0.0f64;
    let mut s = 1.0f64; // sum-factor: sum(b_m) / sum(b_right)
    a_out[right - block_start] = a;
    b_out[right - block_start] = c;
    let k = active_states as f64;
    for m in (block_start + 1..right).rev() {
        let recomb_rate = marker_recomb_rate(p_recomb, m + 1);
        if recomb_rate > 0.0 {
            let (stay, shift) = subset_transition_params_adaptive_q(
                recomb_rate,
                active_states,
                transition_haps,
                transition_lambda,
            );
            let stay = stay as f64;
            let shift = shift as f64;
            let a_new = stay * a;
            let c_new = stay.mul_add(c, shift * s);
            let s_new = (stay + shift * k) * s;
            a = a_new;
            c = c_new;
            s = s_new;
        }
        a_out[m - block_start] = a;
        b_out[m - block_start] = c;
    }
}

#[inline]
fn fill_fwd_affine_coeffs(
    a_out: &mut [f64],
    b_out: &mut [f64],
    p_recomb: &[f32],
    block_start: usize,
    block_end: usize,
    active_states: usize,
    transition_haps: usize,
    transition_lambda: f32,
) {
    if block_start + 1 >= block_end {
        return;
    }
    // Forward interior coefficients with explicit left-boundary mass handling:
    //
    // Let u = F_left and S_u = sum(u). Represent interior forward as:
    //   F_m(i) = a_m * u_i + b_m * S_u
    //
    // For one transition step x' = stay*x + shift*sum(x), the exact recurrence is:
    //   a' = stay * a
    //   b' = stay * b + shift * (a + K*b)
    //
    // where K is active state count. This form remains exact even if S_u != 1.
    //
    // Slot 0 corresponds to the block boundary marker itself and is initialized
    // to identity for robustness (`a=1, b=0`). Interior processing should only
    // consume slots for m in (block_start, block_end).
    let mut a = 1.0f64;
    let mut b = 0.0f64;
    a_out[0] = a;
    b_out[0] = b;
    let k = active_states as f64;
    for m in block_start + 1..block_end {
        let recomb_rate = marker_recomb_rate(p_recomb, m);
        if recomb_rate > 0.0 {
            let (stay, shift) = subset_transition_params_adaptive_q(
                recomb_rate,
                active_states,
                transition_haps,
                transition_lambda,
            );
            let stay = stay as f64;
            let shift = shift as f64;
            let a_new = stay * a;
            let b_new = stay.mul_add(b, shift * (a + k * b));
            a = a_new;
            b = b_new;
        }
        a_out[m - block_start] = a;
        b_out[m - block_start] = b;
    }
}

#[inline]
fn fill_pattern_sums(
    state_patterns: &[u16],
    active_states: usize,
    fwd_base: &[f32],
    bwd_base: &[f32],
    sum_f: &mut [f32],
    sum_b: &mut [f32],
    sum_fb: &mut [f32],
    state_count: &mut [f32],
) {
    // Pattern sufficient statistics at checkpoint boundary vectors:
    //
    // For each pattern p, accumulate over states i with pattern(i)=p:
    //   sum_f[p]      = sum_i f_i
    //   sum_b[p]      = sum_i b_i
    //   sum_fb[p]     = sum_i f_i * b_i
    //   state_count[p]= sum_i 1
    //
    // Interior marker masses can then be evaluated from these four arrays with
    // affine coefficients, avoiding a per-marker loop over all states.
    sum_f.fill(0.0);
    sum_b.fill(0.0);
    sum_fb.fill(0.0);
    state_count.fill(0.0);
    for i in 0..active_states {
        let pid = state_patterns[i] as usize;
        if pid >= sum_f.len() {
            continue;
        }
        let f = fwd_base[i];
        let b = bwd_base[i];
        sum_f[pid] += f;
        sum_b[pid] += b;
        sum_fb[pid] += f * b;
        state_count[pid] += 1.0;
    }
}

#[inline]
fn fill_pattern_sums_with_products(
    state_patterns: &[u16],
    active_states: usize,
    fwd_base: &[f32],
    bwd_base: &[f32],
    fb_products: &[f32],
    sum_f: &mut [f32],
    sum_b: &mut [f32],
    sum_fb: &mut [f32],
    state_count: &mut [f32],
) {
    sum_f.fill(0.0);
    sum_b.fill(0.0);
    sum_fb.fill(0.0);
    state_count.fill(0.0);
    for i in 0..active_states {
        let pid = state_patterns[i] as usize;
        if pid >= sum_f.len() {
            continue;
        }
        let f = fwd_base[i];
        let b = bwd_base[i];
        sum_f[pid] += f;
        sum_b[pid] += b;
        sum_fb[pid] += fb_products[i];
        state_count[pid] += 1.0;
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn bitmask16_at(words: &[u64], word_idx: usize, bit_off: usize) -> u16 {
    let lo = words.get(word_idx).copied().unwrap_or(0);
    if bit_off <= 48 {
        ((lo >> bit_off) & 0xFFFF) as u16
    } else {
        let hi = words.get(word_idx + 1).copied().unwrap_or(0);
        let left = lo >> bit_off;
        let right = hi << (64 - bit_off);
        ((left | right) & 0xFFFF) as u16
    }
}

#[inline]
fn dense_identity_biallelic_sums_scalar(
    limit: usize,
    bits: &[u64],
    missing: &[u64],
    fwd: &[f32],
    bwd: &[f32],
    fb_products: Option<&[f32]>,
) -> (f32, f32, f32, usize, f32, f32, f32, usize) {
    let mut alt_sum_f = 0.0f32;
    let mut alt_sum_b = 0.0f32;
    let mut alt_sum_fb = 0.0f32;
    let mut alt_count = 0usize;
    let mut miss_sum_f = 0.0f32;
    let mut miss_sum_b = 0.0f32;
    let mut miss_sum_fb = 0.0f32;
    let mut miss_count = 0usize;
    let max_words = (limit.saturating_add(63)) >> 6;
    if let Some(fb) = fb_products {
        for word_idx in 0..max_words {
            let base = word_idx << 6;
            let miss_bits = missing.get(word_idx).copied().unwrap_or(0);
            let mut miss_word = miss_bits;
            while miss_word != 0 {
                let bit = miss_word.trailing_zeros() as usize;
                let h = base + bit;
                if h >= limit {
                    break;
                }
                miss_sum_f += fwd[h];
                miss_sum_b += bwd[h];
                miss_sum_fb += fb[h];
                miss_count += 1;
                miss_word &= miss_word - 1;
            }
            let mut alt_word = bits.get(word_idx).copied().unwrap_or(0) & !miss_bits;
            while alt_word != 0 {
                let bit = alt_word.trailing_zeros() as usize;
                let h = base + bit;
                if h >= limit {
                    break;
                }
                alt_sum_f += fwd[h];
                alt_sum_b += bwd[h];
                alt_sum_fb += fb[h];
                alt_count += 1;
                alt_word &= alt_word - 1;
            }
        }
    } else {
        for word_idx in 0..max_words {
            let base = word_idx << 6;
            let miss_bits = missing.get(word_idx).copied().unwrap_or(0);
            let mut miss_word = miss_bits;
            while miss_word != 0 {
                let bit = miss_word.trailing_zeros() as usize;
                let h = base + bit;
                if h >= limit {
                    break;
                }
                let f = fwd[h];
                let b = bwd[h];
                miss_sum_f += f;
                miss_sum_b += b;
                miss_sum_fb += f * b;
                miss_count += 1;
                miss_word &= miss_word - 1;
            }
            let mut alt_word = bits.get(word_idx).copied().unwrap_or(0) & !miss_bits;
            while alt_word != 0 {
                let bit = alt_word.trailing_zeros() as usize;
                let h = base + bit;
                if h >= limit {
                    break;
                }
                let f = fwd[h];
                let b = bwd[h];
                alt_sum_f += f;
                alt_sum_b += b;
                alt_sum_fb += f * b;
                alt_count += 1;
                alt_word &= alt_word - 1;
            }
        }
    }
    (
        alt_sum_f,
        alt_sum_b,
        alt_sum_fb,
        alt_count,
        miss_sum_f,
        miss_sum_b,
        miss_sum_fb,
        miss_count,
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dense_identity_biallelic_sums_avx512(
    limit: usize,
    bits: &[u64],
    missing: &[u64],
    fwd: &[f32],
    bwd: &[f32],
    fb_products: Option<&[f32]>,
) -> (f32, f32, f32, usize, f32, f32, f32, usize) {
    use std::arch::x86_64::*;
    let mut alt_sum_f_vec = _mm512_setzero_ps();
    let mut alt_sum_b_vec = _mm512_setzero_ps();
    let mut alt_sum_fb_vec = _mm512_setzero_ps();
    let mut miss_sum_f_vec = _mm512_setzero_ps();
    let mut miss_sum_b_vec = _mm512_setzero_ps();
    let mut miss_sum_fb_vec = _mm512_setzero_ps();
    let mut alt_count = 0usize;
    let mut miss_count = 0usize;
    let mut k = 0usize;
    while k < limit {
        let remaining = limit - k;
        let valid_mask: __mmask16 = if remaining >= 16 {
            0xFFFF
        } else {
            ((1u32 << remaining) - 1) as __mmask16
        };
        let word_idx = k >> 6;
        let bit_off = k & 63;
        let miss16 = bitmask16_at(missing, word_idx, bit_off);
        let bits16 = bitmask16_at(bits, word_idx, bit_off);
        let miss_mask: __mmask16 = (miss16 as __mmask16) & valid_mask;
        let alt_mask: __mmask16 = (bits16 as __mmask16) & (!miss_mask) & valid_mask;
        alt_count += alt_mask.count_ones() as usize;
        miss_count += miss_mask.count_ones() as usize;

        let f_ptr = unsafe { fwd.as_ptr().add(k) };
        let b_ptr = unsafe { bwd.as_ptr().add(k) };
        let f_vec = unsafe { _mm512_maskz_loadu_ps(valid_mask, f_ptr) };
        let b_vec = unsafe { _mm512_maskz_loadu_ps(valid_mask, b_ptr) };
        let fb_vec = if let Some(fb) = fb_products {
            let fb_ptr = unsafe { fb.as_ptr().add(k) };
            unsafe { _mm512_maskz_loadu_ps(valid_mask, fb_ptr) }
        } else {
            _mm512_mul_ps(f_vec, b_vec)
        };

        // Dedicated masked accumulation: avoids materializing intermediate masked vectors.
        alt_sum_f_vec = _mm512_mask_add_ps(alt_sum_f_vec, alt_mask, alt_sum_f_vec, f_vec);
        alt_sum_b_vec = _mm512_mask_add_ps(alt_sum_b_vec, alt_mask, alt_sum_b_vec, b_vec);
        alt_sum_fb_vec = _mm512_mask_add_ps(alt_sum_fb_vec, alt_mask, alt_sum_fb_vec, fb_vec);
        miss_sum_f_vec = _mm512_mask_add_ps(miss_sum_f_vec, miss_mask, miss_sum_f_vec, f_vec);
        miss_sum_b_vec = _mm512_mask_add_ps(miss_sum_b_vec, miss_mask, miss_sum_b_vec, b_vec);
        miss_sum_fb_vec = _mm512_mask_add_ps(miss_sum_fb_vec, miss_mask, miss_sum_fb_vec, fb_vec);

        k += 16;
    }

    let mut tmp = [0f32; 16];
    unsafe { _mm512_storeu_ps(tmp.as_mut_ptr(), alt_sum_f_vec) };
    let alt_sum_f = tmp.iter().sum::<f32>();
    unsafe { _mm512_storeu_ps(tmp.as_mut_ptr(), alt_sum_b_vec) };
    let alt_sum_b = tmp.iter().sum::<f32>();
    unsafe { _mm512_storeu_ps(tmp.as_mut_ptr(), alt_sum_fb_vec) };
    let alt_sum_fb = tmp.iter().sum::<f32>();
    unsafe { _mm512_storeu_ps(tmp.as_mut_ptr(), miss_sum_f_vec) };
    let miss_sum_f = tmp.iter().sum::<f32>();
    unsafe { _mm512_storeu_ps(tmp.as_mut_ptr(), miss_sum_b_vec) };
    let miss_sum_b = tmp.iter().sum::<f32>();
    unsafe { _mm512_storeu_ps(tmp.as_mut_ptr(), miss_sum_fb_vec) };
    let miss_sum_fb = tmp.iter().sum::<f32>();
    (
        alt_sum_f,
        alt_sum_b,
        alt_sum_fb,
        alt_count,
        miss_sum_f,
        miss_sum_b,
        miss_sum_fb,
        miss_count,
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dense_identity_biallelic_sums_avx2(
    limit: usize,
    bits: &[u64],
    missing: &[u64],
    fwd: &[f32],
    bwd: &[f32],
    fb_products: Option<&[f32]>,
) -> (f32, f32, f32, usize, f32, f32, f32, usize) {
    use std::arch::x86_64::*;
    let mut alt_sum_f_vec = _mm256_setzero_ps();
    let mut alt_sum_b_vec = _mm256_setzero_ps();
    let mut alt_sum_fb_vec = _mm256_setzero_ps();
    let mut miss_sum_f_vec = _mm256_setzero_ps();
    let mut miss_sum_b_vec = _mm256_setzero_ps();
    let mut miss_sum_fb_vec = _mm256_setzero_ps();
    let mut alt_count = 0usize;
    let mut miss_count = 0usize;
    let mut k = 0usize;
    while k < limit {
        let remaining = limit - k;
        let valid_bits: u8 = if remaining >= 8 {
            0xFF
        } else {
            ((1u16 << remaining) - 1) as u8
        };
        let word_idx = k >> 6;
        let bit_off = k & 63;
        let miss8 = bitmask16_at(missing, word_idx, bit_off) as u8;
        let bits8 = bitmask16_at(bits, word_idx, bit_off) as u8;
        let miss_mask_bits = miss8 & valid_bits;
        let alt_mask_bits = bits8 & (!miss_mask_bits) & valid_bits;
        alt_count += alt_mask_bits.count_ones() as usize;
        miss_count += miss_mask_bits.count_ones() as usize;

        let lane_mask = |bits: u8| -> __m256i {
            _mm256_set_epi32(
                if (bits & (1 << 7)) != 0 { -1 } else { 0 },
                if (bits & (1 << 6)) != 0 { -1 } else { 0 },
                if (bits & (1 << 5)) != 0 { -1 } else { 0 },
                if (bits & (1 << 4)) != 0 { -1 } else { 0 },
                if (bits & (1 << 3)) != 0 { -1 } else { 0 },
                if (bits & (1 << 2)) != 0 { -1 } else { 0 },
                if (bits & (1 << 1)) != 0 { -1 } else { 0 },
                if (bits & (1 << 0)) != 0 { -1 } else { 0 },
            )
        };
        let valid_mask_i = lane_mask(valid_bits);
        let alt_mask_i = lane_mask(alt_mask_bits);
        let miss_mask_i = lane_mask(miss_mask_bits);

        let f_ptr = unsafe { fwd.as_ptr().add(k) };
        let b_ptr = unsafe { bwd.as_ptr().add(k) };
        let f_vec = unsafe { _mm256_maskload_ps(f_ptr, valid_mask_i) };
        let b_vec = unsafe { _mm256_maskload_ps(b_ptr, valid_mask_i) };
        let fb_vec = if let Some(fb) = fb_products {
            let fb_ptr = unsafe { fb.as_ptr().add(k) };
            unsafe { _mm256_maskload_ps(fb_ptr, valid_mask_i) }
        } else {
            _mm256_mul_ps(f_vec, b_vec)
        };

        alt_sum_f_vec = _mm256_add_ps(
            alt_sum_f_vec,
            _mm256_and_ps(f_vec, _mm256_castsi256_ps(alt_mask_i)),
        );
        alt_sum_b_vec = _mm256_add_ps(
            alt_sum_b_vec,
            _mm256_and_ps(b_vec, _mm256_castsi256_ps(alt_mask_i)),
        );
        alt_sum_fb_vec = _mm256_add_ps(
            alt_sum_fb_vec,
            _mm256_and_ps(fb_vec, _mm256_castsi256_ps(alt_mask_i)),
        );
        miss_sum_f_vec = _mm256_add_ps(
            miss_sum_f_vec,
            _mm256_and_ps(f_vec, _mm256_castsi256_ps(miss_mask_i)),
        );
        miss_sum_b_vec = _mm256_add_ps(
            miss_sum_b_vec,
            _mm256_and_ps(b_vec, _mm256_castsi256_ps(miss_mask_i)),
        );
        miss_sum_fb_vec = _mm256_add_ps(
            miss_sum_fb_vec,
            _mm256_and_ps(fb_vec, _mm256_castsi256_ps(miss_mask_i)),
        );

        k += 8;
    }

    let mut tmp = [0f32; 8];
    unsafe { _mm256_storeu_ps(tmp.as_mut_ptr(), alt_sum_f_vec) };
    let alt_sum_f = tmp.iter().sum::<f32>();
    unsafe { _mm256_storeu_ps(tmp.as_mut_ptr(), alt_sum_b_vec) };
    let alt_sum_b = tmp.iter().sum::<f32>();
    unsafe { _mm256_storeu_ps(tmp.as_mut_ptr(), alt_sum_fb_vec) };
    let alt_sum_fb = tmp.iter().sum::<f32>();
    unsafe { _mm256_storeu_ps(tmp.as_mut_ptr(), miss_sum_f_vec) };
    let miss_sum_f = tmp.iter().sum::<f32>();
    unsafe { _mm256_storeu_ps(tmp.as_mut_ptr(), miss_sum_b_vec) };
    let miss_sum_b = tmp.iter().sum::<f32>();
    unsafe { _mm256_storeu_ps(tmp.as_mut_ptr(), miss_sum_fb_vec) };
    let miss_sum_fb = tmp.iter().sum::<f32>();
    (
        alt_sum_f,
        alt_sum_b,
        alt_sum_fb,
        alt_count,
        miss_sum_f,
        miss_sum_b,
        miss_sum_fb,
        miss_count,
    )
}

#[derive(Clone, Copy, Debug)]
struct ForwardAffine {
    a: f32,
    b: f32,
}

impl ForwardAffine {}

#[derive(Clone, Copy, Debug)]
struct BackwardAffine {
    a: f32,
    add: f32,
}

impl BackwardAffine {
    #[inline]
    fn new(a: f32, b_coeff: f32, bwd_sum_right: f32) -> Self {
        Self {
            a,
            add: b_coeff * bwd_sum_right,
        }
    }
}

#[inline]
fn fill_state_patterns_seqcoded(hap_to_seq: &[u16], state_haps: &[RefHapId], out: &mut [u16]) {
    for (i, hap) in state_haps.iter().enumerate() {
        out[i] = hap_to_seq[hap.as_usize()];
    }
}

#[inline]
fn fill_state_patterns_dict(col: &DictionaryColumn, state_haps: &[RefHapId], out: &mut [u16]) {
    for (i, hap) in state_haps.iter().enumerate() {
        out[i] = col.hap_pattern_idx(*hap) as u16;
    }
}

#[inline]
fn forward_update_impl<C: RefColumnLike>(
    ws: &mut ImputeWorkspace,
    m: usize,
    state_haps: &[RefHapId],
    ref_columns: &[C],
    target_probs: &TargetAlleleProbs,
    p_recomb: &[f32],
    base_error_rate: f32,
    active_states: usize,
    transition_haps: usize,
    transition_lambda: f32,
) -> f32 {
    let probs = target_probs.probs_for_marker_trusted(m);
    let uniform = target_probs.is_uniform_marker(m);
    let recomb_rate = marker_recomb_rate(p_recomb, m);
    let marker_error = target_probs.marker_error_rate(m).unwrap_or(base_error_rate);

    let mut next_sum = if uniform {
        transition_only_forward_update(
            &mut ws.fwd[..active_states],
            1.0,
            recomb_rate,
            transition_haps,
            transition_lambda,
        )
    } else {
        let ref_alleles = refresh_ref_alleles(
            &ref_columns[m],
            state_haps,
            &mut ws.state_alleles[..active_states],
            &mut ws.dict_pattern_alleles,
        );
        fill_emissions(
            &ref_alleles,
            probs,
            marker_error,
            &mut ws.emission_by_allele,
            &mut ws.emissions[..active_states],
        );
        if recomb_rate > 0.0 {
            let recomb_rate_eff = effective_recomb_rate(recomb_rate, transition_lambda);
            WeightedHmmUpdater::fwd_update_weighted(
                &mut ws.fwd,
                1.0,
                recomb_rate_eff,
                transition_haps,
                PatternCounts::new(&ws.weights[..active_states]),
                EmissionProbs::new(&ws.emissions[..active_states]),
                active_states,
            )
        } else {
            for i in 0..active_states {
                ws.fwd[i] *= ws.emissions[i];
            }
            ws.fwd[..active_states].iter().sum::<f32>().max(1e-30)
        }
    };

    if next_sum <= 0.0 {
        next_sum = 1e-30;
    }
    ws.fwd_scales[m] = if uniform { 1.0 } else { next_sum };
    let inv = 1.0 / next_sum;
    for i in 0..active_states {
        ws.fwd[i] *= inv;
    }
    1.0
}

#[inline]
fn forward_update_seqcoded(
    ws: &mut ImputeWorkspace,
    m: usize,
    state_haps: &[RefHapId],
    ref_columns: &[GenotypeColumn],
    target_probs: &TargetAlleleProbs,
    p_recomb: &[f32],
    base_error_rate: f32,
    active_states: usize,
    transition_haps: usize,
    transition_lambda: f32,
    last_hap_ptr: &mut *const u16,
) -> f32 {
    let probs = target_probs.probs_for_marker_trusted(m);
    let uniform = target_probs.is_uniform_marker(m);
    let recomb_rate = marker_recomb_rate(p_recomb, m);
    let marker_error = target_probs.marker_error_rate(m).unwrap_or(base_error_rate);

    let col = seqcoded_col(&ref_columns[m]);
    let seq_patterns = refresh_seq_patterns(col, last_hap_ptr, state_haps, &mut ws.state_patterns);

    let mut next_sum = if uniform {
        transition_only_forward_update(
            &mut ws.fwd[..active_states],
            1.0,
            recomb_rate,
            transition_haps,
            transition_lambda,
        )
    } else {
        fill_pattern_emissions(
            seq_patterns.seq_alleles,
            probs,
            marker_error,
            &mut ws.emission_by_allele,
            &mut ws.pattern_emissions,
        );
        for i in 0..active_states {
            let pid = seq_patterns.state_patterns[i] as usize;
            ws.emissions[i] = ws.pattern_emissions.get(pid).copied().unwrap_or(1.0);
        }
        if recomb_rate > 0.0 {
            let recomb_rate_eff = effective_recomb_rate(recomb_rate, transition_lambda);
            WeightedHmmUpdater::fwd_update_weighted(
                &mut ws.fwd,
                1.0,
                recomb_rate_eff,
                transition_haps,
                PatternCounts::new(&ws.weights[..active_states]),
                EmissionProbs::new(&ws.emissions[..active_states]),
                active_states,
            )
        } else {
            for i in 0..active_states {
                ws.fwd[i] *= ws.emissions[i];
            }
            ws.fwd[..active_states].iter().sum::<f32>().max(1e-30)
        }
    };

    if next_sum <= 0.0 {
        next_sum = 1e-30;
    }
    ws.fwd_scales[m] = if uniform { 1.0 } else { next_sum };
    let inv = 1.0 / next_sum;
    for i in 0..active_states {
        ws.fwd[i] *= inv;
    }
    1.0
}

#[inline]
fn forward_update_dict(
    ws: &mut ImputeWorkspace,
    m: usize,
    state_haps: &[RefHapId],
    ref_columns: &[GenotypeColumn],
    target_probs: &TargetAlleleProbs,
    p_recomb: &[f32],
    base_error_rate: f32,
    active_states: usize,
    transition_haps: usize,
    transition_lambda: f32,
    last_dict_ptr: &mut *const DictionaryColumn,
) -> f32 {
    let probs = target_probs.probs_for_marker_trusted(m);
    let uniform = target_probs.is_uniform_marker(m);
    let recomb_rate = marker_recomb_rate(p_recomb, m);
    let marker_error = target_probs.marker_error_rate(m).unwrap_or(base_error_rate);

    let mut next_sum = if uniform {
        transition_only_forward_update(
            &mut ws.fwd[..active_states],
            1.0,
            recomb_rate,
            transition_haps,
            transition_lambda,
        )
    } else {
        let col = dict_col_ref(&ref_columns[m]);
        let dict_patterns = refresh_dict_patterns(
            &col,
            last_dict_ptr,
            state_haps,
            &mut ws.state_patterns,
            &mut ws.dict_pattern_alleles,
        );
        fill_pattern_emissions(
            dict_patterns.pattern_alleles,
            probs,
            marker_error,
            &mut ws.emission_by_allele,
            &mut ws.pattern_emissions,
        );
        for i in 0..active_states {
            let pid = dict_patterns.state_patterns[i] as usize;
            ws.emissions[i] = ws.pattern_emissions.get(pid).copied().unwrap_or(1.0);
        }
        if recomb_rate > 0.0 {
            let recomb_rate_eff = effective_recomb_rate(recomb_rate, transition_lambda);
            WeightedHmmUpdater::fwd_update_weighted(
                &mut ws.fwd,
                1.0,
                recomb_rate_eff,
                transition_haps,
                PatternCounts::new(&ws.weights[..active_states]),
                EmissionProbs::new(&ws.emissions[..active_states]),
                active_states,
            )
        } else {
            for i in 0..active_states {
                ws.fwd[i] *= ws.emissions[i];
            }
            ws.fwd[..active_states].iter().sum::<f32>().max(1e-30)
        }
    };

    if next_sum <= 0.0 {
        next_sum = 1e-30;
    }
    ws.fwd_scales[m] = if uniform { 1.0 } else { next_sum };
    let inv = 1.0 / next_sum;
    for i in 0..active_states {
        ws.fwd[i] *= inv;
    }
    1.0
}

/// Monomorphized HMM core over a concrete genotype column type.
#[derive(Clone, Copy)]
struct PreparedGroups {
    key: usize,
    n_groups: usize,
    direct_alleles_ptr: *const u8,
    direct_alleles_len: usize,
}

trait ImputeKernel {
    type MarkerCtx;
    const LABEL: &'static str;
    const CLAMP_AFFINE_GROUP_MASS: bool;
    const MAX_NON_MISSING_ALLELES: usize;

    fn reset_forward(&mut self);
    fn reset_backward(&mut self);

    fn forward_update(
        &mut self,
        ws: &mut ImputeWorkspace,
        m: usize,
        state_haps: &[RefHapId],
        ref_columns: &[GenotypeColumn],
        target_probs: &TargetAlleleProbs,
        p_recomb: &[f32],
        current_error: f32,
        active_states: usize,
        transition_haps: usize,
        transition_lambda: f32,
    ) -> f32;

    fn prepare_marker(
        &mut self,
        ws: &mut ImputeWorkspace,
        m: usize,
        state_haps: &[RefHapId],
        ref_columns: &[GenotypeColumn],
        n_alleles: usize,
        active_states: usize,
    ) -> Self::MarkerCtx;

    fn marker_key(ctx: &Self::MarkerCtx) -> usize;
    fn marker_group_count(ctx: &Self::MarkerCtx) -> usize;
    fn group_alleles<'a>(ctx: &Self::MarkerCtx, dict_pattern_alleles: &'a [u8]) -> &'a [u8];

    #[inline]
    fn prepare_marker_with_pattern_sums(
        &mut self,
        ws: &mut ImputeWorkspace,
        m: usize,
        state_haps: &[RefHapId],
        ref_columns: &[GenotypeColumn],
        n_alleles: usize,
        active_states: usize,
        checkpoint_totals: Option<[f32; 3]>,
        use_boundary_fb_products: bool,
        pattern_sum_key: &mut usize,
    ) -> Self::MarkerCtx {
        let _ = checkpoint_totals;
        let _ = use_boundary_fb_products;
        let prepared =
            self.prepare_marker(ws, m, state_haps, ref_columns, n_alleles, active_states);
        let key = Self::marker_key(&prepared);
        if key != *pattern_sum_key {
            let n_groups = Self::marker_group_count(&prepared);
            ws.ensure_pattern_sums(n_groups);
            if use_boundary_fb_products {
                fill_pattern_sums_with_products(
                    &ws.state_patterns[..active_states],
                    active_states,
                    &ws.fwd[..active_states],
                    &ws.bwd[..active_states],
                    &ws.boundary_fb_products[..active_states],
                    &mut ws.pattern_sum_f[..n_groups],
                    &mut ws.pattern_sum_b[..n_groups],
                    &mut ws.pattern_sum_fb[..n_groups],
                    &mut ws.pattern_state_count[..n_groups],
                );
            } else {
                fill_pattern_sums(
                    &ws.state_patterns[..active_states],
                    active_states,
                    &ws.fwd[..active_states],
                    &ws.bwd[..active_states],
                    &mut ws.pattern_sum_f[..n_groups],
                    &mut ws.pattern_sum_b[..n_groups],
                    &mut ws.pattern_sum_fb[..n_groups],
                    &mut ws.pattern_state_count[..n_groups],
                );
            }
            *pattern_sum_key = key;
        }
        prepared
    }
}

#[derive(Default)]
struct DenseKernel {
    sparse_state_index: Vec<i32>,
    sparse_state_index_ptr: *const RefHapId,
    sparse_state_index_len: usize,
    sparse_identity_order: bool,
    pattern_sig_valid: bool,
    pattern_sig_key: usize,
    pattern_sig_state_ptr: *const RefHapId,
    pattern_sig_active_states: usize,
    dict_allele_sig_valid: bool,
    dict_allele_sig_key: usize,
    dict_allele_sig_offset: usize,
    dict_allele_sig_n_patterns: usize,
    dense_id_sig_valid: bool,
    dense_id_sig_key: u64,
    dense_id_sig_words: usize,
    dense_id_sig_tail_bits: usize,
    dense_id_bits: Vec<u64>,
    dense_id_missing: Vec<u64>,
}

impl DenseKernel {
    #[inline]
    fn allele_to_pid(ref_allele: u8, n_alleles: usize, missing_pid: u16) -> usize {
        if AlleleCode::from_raw(ref_allele).is_missing() {
            missing_pid as usize
        } else {
            let idx = ref_allele as usize;
            if idx < n_alleles {
                idx
            } else {
                missing_pid as usize
            }
        }
    }

    #[inline]
    fn ensure_sparse_state_index(&mut self, state_haps: &[RefHapId], active_states: usize) {
        let ptr = state_haps.as_ptr();
        if self.sparse_state_index_ptr == ptr && self.sparse_state_index_len == active_states {
            return;
        }
        // Fast-path: full-panel identity ordering (state i == hap i).
        // In this case we can index fwd/bwd directly by hap index and avoid
        // materializing a hap->state map.
        self.sparse_identity_order = state_haps
            .iter()
            .take(active_states)
            .enumerate()
            .all(|(state_idx, hap)| hap.as_usize() == state_idx);
        if self.sparse_identity_order {
            self.sparse_state_index.clear();
            self.sparse_state_index_ptr = ptr;
            self.sparse_state_index_len = active_states;
            return;
        }

        let mut max_hap = 0usize;
        for hap in state_haps.iter().take(active_states) {
            max_hap = max_hap.max(hap.as_usize());
        }
        self.sparse_state_index.clear();
        self.sparse_state_index.resize(max_hap + 1, -1);
        for (state_idx, hap) in state_haps.iter().take(active_states).enumerate() {
            self.sparse_state_index[hap.as_usize()] = state_idx as i32;
        }
        self.sparse_state_index_ptr = ptr;
        self.sparse_state_index_len = active_states;
    }

    #[inline]
    fn pattern_signature_matches(
        &self,
        key: usize,
        state_haps: &[RefHapId],
        active_states: usize,
    ) -> bool {
        self.pattern_sig_valid
            && self.pattern_sig_key == key
            && self.pattern_sig_state_ptr == state_haps.as_ptr()
            && self.pattern_sig_active_states == active_states
    }

    #[inline]
    fn pattern_signature_set(&mut self, key: usize, state_haps: &[RefHapId], active_states: usize) {
        self.pattern_sig_valid = true;
        self.pattern_sig_key = key;
        self.pattern_sig_state_ptr = state_haps.as_ptr();
        self.pattern_sig_active_states = active_states;
    }

    #[inline]
    fn pattern_signature_invalidate(&mut self) {
        self.pattern_sig_valid = false;
        self.pattern_sig_key = 0;
        self.pattern_sig_state_ptr = std::ptr::null();
        self.pattern_sig_active_states = 0;
    }

    #[inline]
    fn dict_allele_signature_matches(&self, key: usize, offset: usize, n_patterns: usize) -> bool {
        self.dict_allele_sig_valid
            && self.dict_allele_sig_key == key
            && self.dict_allele_sig_offset == offset
            && self.dict_allele_sig_n_patterns == n_patterns
    }

    #[inline]
    fn dict_allele_signature_set(&mut self, key: usize, offset: usize, n_patterns: usize) {
        self.dict_allele_sig_valid = true;
        self.dict_allele_sig_key = key;
        self.dict_allele_sig_offset = offset;
        self.dict_allele_sig_n_patterns = n_patterns;
    }

    #[inline]
    fn dict_allele_signature_invalidate(&mut self) {
        self.dict_allele_sig_valid = false;
        self.dict_allele_sig_key = 0;
        self.dict_allele_sig_offset = 0;
        self.dict_allele_sig_n_patterns = 0;
    }

    #[inline]
    fn dense_identity_key(col: &DenseColumn, active_states: usize, n_alleles: usize) -> u64 {
        let mut h = col.fingerprint();
        h ^= (active_states as u64).wrapping_mul(0x9e3779b97f4a7c15);
        h ^= (n_alleles as u64).wrapping_mul(0x517cc1b727220a95);
        h
    }

    #[inline]
    fn dense_identity_match(&self, col: &DenseColumn, active_states: usize, key: u64) -> bool {
        if !self.dense_id_sig_valid || self.dense_id_sig_key != key {
            return false;
        }
        let words = (active_states.saturating_add(63)) >> 6;
        if words != self.dense_id_sig_words {
            return false;
        }
        let tail_bits = active_states & 63;
        if tail_bits != self.dense_id_sig_tail_bits {
            return false;
        }
        let bits = col.bits_raw();
        let missing = col.missing_raw();
        if words == 0 {
            return true;
        }
        for i in 0..words.saturating_sub(1) {
            let b = bits.get(i).copied().unwrap_or(0);
            let m = missing.get(i).copied().unwrap_or(0);
            if self.dense_id_bits[i] != b || self.dense_id_missing[i] != m {
                return false;
            }
        }
        let last = words - 1;
        let mask = if tail_bits == 0 {
            u64::MAX
        } else {
            (1u64 << tail_bits) - 1
        };
        let b_last = bits.get(last).copied().unwrap_or(0) & mask;
        let m_last = missing.get(last).copied().unwrap_or(0) & mask;
        self.dense_id_bits[last] == b_last && self.dense_id_missing[last] == m_last
    }

    #[inline]
    fn dense_identity_capture(&mut self, col: &DenseColumn, active_states: usize, key: u64) {
        let words = (active_states.saturating_add(63)) >> 6;
        let tail_bits = active_states & 63;
        if self.dense_id_bits.len() < words {
            self.dense_id_bits.resize(words, 0);
        }
        if self.dense_id_missing.len() < words {
            self.dense_id_missing.resize(words, 0);
        }
        let bits = col.bits_raw();
        let missing = col.missing_raw();
        if words > 0 {
            for i in 0..words.saturating_sub(1) {
                self.dense_id_bits[i] = bits.get(i).copied().unwrap_or(0);
                self.dense_id_missing[i] = missing.get(i).copied().unwrap_or(0);
            }
            let last = words - 1;
            let mask = if tail_bits == 0 {
                u64::MAX
            } else {
                (1u64 << tail_bits) - 1
            };
            self.dense_id_bits[last] = bits.get(last).copied().unwrap_or(0) & mask;
            self.dense_id_missing[last] = missing.get(last).copied().unwrap_or(0) & mask;
        }
        self.dense_id_sig_valid = true;
        self.dense_id_sig_key = key;
        self.dense_id_sig_words = words;
        self.dense_id_sig_tail_bits = tail_bits;
    }

    #[inline]
    fn dense_identity_invalidate(&mut self) {
        self.dense_id_sig_valid = false;
        self.dense_id_sig_key = 0;
        self.dense_id_sig_words = 0;
        self.dense_id_sig_tail_bits = 0;
    }

    #[inline]
    fn prepare_dense_groups(
        ws: &mut ImputeWorkspace,
        m: usize,
        state_haps: &[RefHapId],
        ref_columns: &[GenotypeColumn],
        n_alleles: usize,
        active_states: usize,
    ) -> PreparedGroups {
        let ref_alleles = refresh_ref_alleles(
            &ref_columns[m],
            state_haps,
            &mut ws.state_alleles[..active_states],
            &mut ws.dict_pattern_alleles,
        );

        let n_groups = if n_alleles == 0 { 1 } else { n_alleles + 1 };
        if ws.dict_pattern_alleles.len() < n_groups {
            ws.dict_pattern_alleles
                .resize(n_groups, AlleleCode::MISSING.raw());
        }
        for i in 0..n_alleles {
            ws.dict_pattern_alleles[i] = i as u8;
        }
        ws.dict_pattern_alleles[n_groups - 1] = AlleleCode::MISSING.raw();

        let missing_pid = (n_groups - 1) as u16;
        for i in 0..active_states {
            let ref_allele = ref_alleles.get(i);
            let pid = if AlleleCode::from_raw(ref_allele).is_missing() {
                missing_pid
            } else {
                let idx = ref_allele as usize;
                if idx < n_alleles {
                    idx as u16
                } else {
                    missing_pid
                }
            };
            ws.state_patterns[i] = pid;
        }

        PreparedGroups {
            key: m.wrapping_add(1),
            n_groups,
            direct_alleles_ptr: std::ptr::null(),
            direct_alleles_len: 0,
        }
    }
}

impl ImputeKernel for DenseKernel {
    type MarkerCtx = PreparedGroups;
    const LABEL: &'static str = "dense/sparse";
    const CLAMP_AFFINE_GROUP_MASS: bool = true;
    const MAX_NON_MISSING_ALLELES: usize = AlleleCode::MISSING.raw() as usize;

    #[inline]
    fn reset_forward(&mut self) {
        self.sparse_state_index_ptr = std::ptr::null();
        self.sparse_state_index_len = 0;
        self.sparse_identity_order = false;
        self.sparse_state_index.clear();
        self.pattern_signature_invalidate();
        self.dict_allele_signature_invalidate();
        self.dense_identity_invalidate();
    }

    #[inline]
    fn reset_backward(&mut self) {
        self.sparse_state_index_ptr = std::ptr::null();
        self.sparse_state_index_len = 0;
        self.sparse_identity_order = false;
        self.sparse_state_index.clear();
        self.pattern_signature_invalidate();
        self.dict_allele_signature_invalidate();
        self.dense_identity_invalidate();
    }

    #[inline]
    fn forward_update(
        &mut self,
        ws: &mut ImputeWorkspace,
        m: usize,
        state_haps: &[RefHapId],
        ref_columns: &[GenotypeColumn],
        target_probs: &TargetAlleleProbs,
        p_recomb: &[f32],
        current_error: f32,
        active_states: usize,
        transition_haps: usize,
        transition_lambda: f32,
    ) -> f32 {
        forward_update_impl(
            ws,
            m,
            state_haps,
            ref_columns,
            target_probs,
            p_recomb,
            current_error,
            active_states,
            transition_haps,
            transition_lambda,
        )
    }

    #[inline]
    fn prepare_marker(
        &mut self,
        ws: &mut ImputeWorkspace,
        m: usize,
        state_haps: &[RefHapId],
        ref_columns: &[GenotypeColumn],
        n_alleles: usize,
        active_states: usize,
    ) -> Self::MarkerCtx {
        Self::prepare_dense_groups(ws, m, state_haps, ref_columns, n_alleles, active_states)
    }

    #[inline]
    fn marker_key(ctx: &Self::MarkerCtx) -> usize {
        ctx.key
    }

    #[inline]
    fn marker_group_count(ctx: &Self::MarkerCtx) -> usize {
        ctx.n_groups
    }

    #[inline]
    fn group_alleles<'a>(ctx: &Self::MarkerCtx, dict_pattern_alleles: &'a [u8]) -> &'a [u8] {
        &dict_pattern_alleles[..ctx.n_groups]
    }

    #[inline]
    fn prepare_marker_with_pattern_sums(
        &mut self,
        ws: &mut ImputeWorkspace,
        m: usize,
        state_haps: &[RefHapId],
        ref_columns: &[GenotypeColumn],
        n_alleles: usize,
        active_states: usize,
        checkpoint_totals: Option<[f32; 3]>,
        use_boundary_fb_products: bool,
        pattern_sum_key: &mut usize,
    ) -> Self::MarkerCtx {
        let n_groups = if n_alleles == 0 { 1 } else { n_alleles + 1 };
        if ws.dict_pattern_alleles.len() < n_groups {
            ws.dict_pattern_alleles
                .resize(n_groups, AlleleCode::MISSING.raw());
        }
        for i in 0..n_alleles {
            ws.dict_pattern_alleles[i] = i as u8;
        }
        ws.dict_pattern_alleles[n_groups - 1] = AlleleCode::MISSING.raw();
        let missing_pid = (n_groups - 1) as u16;
        ws.ensure_pattern_sums(n_groups);

        match &ref_columns[m] {
            GenotypeColumn::SeqCoded(col) => {
                let hap_to_seq = col.hap_to_seq();
                let key = hap_to_seq.as_ptr() as usize;
                if !self.pattern_signature_matches(key, state_haps, active_states) {
                    if let Some(cached) =
                        ws.pattern_cache_lookup(key, state_haps.as_ptr(), active_states)
                    {
                        if cached.len == active_states {
                            // Safety:
                            // - cache entry was created from a `Vec<u16>` of this exact length.
                            // - destination slice is valid and non-overlapping with cache storage.
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    cached.ptr,
                                    ws.state_patterns.as_mut_ptr(),
                                    active_states,
                                );
                            }
                        } else {
                            fill_state_patterns_seqcoded(
                                hap_to_seq,
                                &state_haps[..active_states],
                                &mut ws.state_patterns[..active_states],
                            );
                            let cached_patterns = ws.state_patterns[..active_states].to_vec();
                            ws.pattern_cache_insert(
                                key,
                                state_haps.as_ptr(),
                                active_states,
                                &cached_patterns,
                            );
                        }
                    } else {
                        fill_state_patterns_seqcoded(
                            hap_to_seq,
                            &state_haps[..active_states],
                            &mut ws.state_patterns[..active_states],
                        );
                        let cached_patterns = ws.state_patterns[..active_states].to_vec();
                        ws.pattern_cache_insert(
                            key,
                            state_haps.as_ptr(),
                            active_states,
                            &cached_patterns,
                        );
                    }
                    self.pattern_signature_set(key, state_haps, active_states);
                }
                if key != *pattern_sum_key {
                    let n_patterns = col.seq_alleles().len();
                    ws.ensure_pattern_sums(n_patterns);
                    if use_boundary_fb_products {
                        fill_pattern_sums_with_products(
                            &ws.state_patterns[..active_states],
                            active_states,
                            &ws.fwd[..active_states],
                            &ws.bwd[..active_states],
                            &ws.boundary_fb_products[..active_states],
                            &mut ws.pattern_sum_f[..n_patterns],
                            &mut ws.pattern_sum_b[..n_patterns],
                            &mut ws.pattern_sum_fb[..n_patterns],
                            &mut ws.pattern_state_count[..n_patterns],
                        );
                    } else {
                        fill_pattern_sums(
                            &ws.state_patterns[..active_states],
                            active_states,
                            &ws.fwd[..active_states],
                            &ws.bwd[..active_states],
                            &mut ws.pattern_sum_f[..n_patterns],
                            &mut ws.pattern_sum_b[..n_patterns],
                            &mut ws.pattern_sum_fb[..n_patterns],
                            &mut ws.pattern_state_count[..n_patterns],
                        );
                    }
                    *pattern_sum_key = key;
                }
                let n_patterns = col.seq_alleles().len();
                return PreparedGroups {
                    key,
                    n_groups: n_patterns,
                    direct_alleles_ptr: col.seq_alleles().as_ptr(),
                    direct_alleles_len: n_patterns,
                };
            }
            GenotypeColumn::Dictionary(col, offset) => {
                let key = col.as_ref() as *const DictionaryColumn as usize;
                let n_patterns = col.n_patterns();
                if !self.pattern_signature_matches(key, state_haps, active_states) {
                    if let Some(cached) =
                        ws.pattern_cache_lookup(key, state_haps.as_ptr(), active_states)
                    {
                        if cached.len == active_states {
                            // Safety:
                            // - cache entry was created from a `Vec<u16>` of this exact length.
                            // - destination slice is valid and non-overlapping with cache storage.
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    cached.ptr,
                                    ws.state_patterns.as_mut_ptr(),
                                    active_states,
                                );
                            }
                        } else {
                            fill_state_patterns_dict(
                                col.as_ref(),
                                &state_haps[..active_states],
                                &mut ws.state_patterns[..active_states],
                            );
                            let cached_patterns = ws.state_patterns[..active_states].to_vec();
                            ws.pattern_cache_insert(
                                key,
                                state_haps.as_ptr(),
                                active_states,
                                &cached_patterns,
                            );
                        }
                    } else {
                        fill_state_patterns_dict(
                            col.as_ref(),
                            &state_haps[..active_states],
                            &mut ws.state_patterns[..active_states],
                        );
                        let cached_patterns = ws.state_patterns[..active_states].to_vec();
                        ws.pattern_cache_insert(
                            key,
                            state_haps.as_ptr(),
                            active_states,
                            &cached_patterns,
                        );
                    }
                    self.pattern_signature_set(key, state_haps, active_states);
                }
                if ws.dict_pattern_alleles.len() < n_patterns {
                    ws.dict_pattern_alleles.resize(n_patterns, 0);
                }
                let dict_offset = *offset as usize;
                if !self.dict_allele_signature_matches(key, dict_offset, n_patterns) {
                    for pattern_idx in 0..n_patterns {
                        ws.dict_pattern_alleles[pattern_idx] =
                            col.pattern_allele(*offset, pattern_idx);
                    }
                    self.dict_allele_signature_set(key, dict_offset, n_patterns);
                }
                if key != *pattern_sum_key {
                    ws.ensure_pattern_sums(n_patterns);
                    if use_boundary_fb_products {
                        fill_pattern_sums_with_products(
                            &ws.state_patterns[..active_states],
                            active_states,
                            &ws.fwd[..active_states],
                            &ws.bwd[..active_states],
                            &ws.boundary_fb_products[..active_states],
                            &mut ws.pattern_sum_f[..n_patterns],
                            &mut ws.pattern_sum_b[..n_patterns],
                            &mut ws.pattern_sum_fb[..n_patterns],
                            &mut ws.pattern_state_count[..n_patterns],
                        );
                    } else {
                        fill_pattern_sums(
                            &ws.state_patterns[..active_states],
                            active_states,
                            &ws.fwd[..active_states],
                            &ws.bwd[..active_states],
                            &mut ws.pattern_sum_f[..n_patterns],
                            &mut ws.pattern_sum_b[..n_patterns],
                            &mut ws.pattern_sum_fb[..n_patterns],
                            &mut ws.pattern_state_count[..n_patterns],
                        );
                    }
                    *pattern_sum_key = key;
                }
                return PreparedGroups {
                    key,
                    n_groups: n_patterns,
                    direct_alleles_ptr: std::ptr::null(),
                    direct_alleles_len: 0,
                };
            }
            GenotypeColumn::Dense(col) if col.bits_per_allele() == 1 && n_alleles <= 2 => {
                self.pattern_signature_invalidate();
                if let Some([sum_f_all, sum_b_all, sum_fb_all]) = checkpoint_totals {
                    self.ensure_sparse_state_index(state_haps, active_states);
                    if self.sparse_identity_order {
                        let dense_key_u64 = Self::dense_identity_key(col, active_states, n_alleles);
                        let dense_key = dense_key_u64 as usize;
                        if *pattern_sum_key == dense_key
                            && self.dense_identity_match(col, active_states, dense_key_u64)
                        {
                            return PreparedGroups {
                                key: dense_key,
                                n_groups,
                                direct_alleles_ptr: std::ptr::null(),
                                direct_alleles_len: 0,
                            };
                        }
                        ws.pattern_sum_f[..n_groups].fill(0.0);
                        ws.pattern_sum_b[..n_groups].fill(0.0);
                        ws.pattern_sum_fb[..n_groups].fill(0.0);
                        ws.pattern_state_count[..n_groups].fill(0.0);
                        let bits = col.bits_raw();
                        let missing = col.missing_raw();
                        let fb_products = if use_boundary_fb_products {
                            Some(&ws.boundary_fb_products[..active_states])
                        } else {
                            None
                        };
                        let (
                            alt_sum_f,
                            alt_sum_b,
                            alt_sum_fb,
                            alt_count,
                            miss_sum_f,
                            miss_sum_b,
                            miss_sum_fb,
                            miss_count,
                        ) = {
                            #[cfg(target_arch = "x86_64")]
                            {
                                if active_states >= 16 && std::is_x86_feature_detected!("avx512f") {
                                    // Safety:
                                    // - guarded by runtime feature check.
                                    // - slices are valid for [0, active_states).
                                    unsafe {
                                        dense_identity_biallelic_sums_avx512(
                                            active_states,
                                            bits,
                                            missing,
                                            &ws.fwd[..active_states],
                                            &ws.bwd[..active_states],
                                            fb_products,
                                        )
                                    }
                                } else if active_states >= 8
                                    && std::is_x86_feature_detected!("avx2")
                                {
                                    // Safety:
                                    // - guarded by runtime feature check.
                                    // - slices are valid for [0, active_states).
                                    unsafe {
                                        dense_identity_biallelic_sums_avx2(
                                            active_states,
                                            bits,
                                            missing,
                                            &ws.fwd[..active_states],
                                            &ws.bwd[..active_states],
                                            fb_products,
                                        )
                                    }
                                } else {
                                    dense_identity_biallelic_sums_scalar(
                                        active_states,
                                        bits,
                                        missing,
                                        &ws.fwd[..active_states],
                                        &ws.bwd[..active_states],
                                        fb_products,
                                    )
                                }
                            }
                            #[cfg(not(target_arch = "x86_64"))]
                            {
                                dense_identity_biallelic_sums_scalar(
                                    active_states,
                                    bits,
                                    missing,
                                    &ws.fwd[..active_states],
                                    &ws.bwd[..active_states],
                                    fb_products,
                                )
                            }
                        };
                        let ref_sum_f = (sum_f_all - alt_sum_f - miss_sum_f).max(0.0);
                        let ref_sum_b = (sum_b_all - alt_sum_b - miss_sum_b).max(0.0);
                        let ref_sum_fb = (sum_fb_all - alt_sum_fb - miss_sum_fb).max(0.0);
                        let ref_count = active_states.saturating_sub(alt_count + miss_count);
                        let ref_pid = Self::allele_to_pid(0, n_alleles, missing_pid);
                        ws.pattern_sum_f[ref_pid] = ref_sum_f;
                        ws.pattern_sum_b[ref_pid] = ref_sum_b;
                        ws.pattern_sum_fb[ref_pid] = ref_sum_fb;
                        ws.pattern_state_count[ref_pid] = ref_count as f32;
                        let alt_pid = Self::allele_to_pid(1, n_alleles, missing_pid);
                        ws.pattern_sum_f[alt_pid] = alt_sum_f;
                        ws.pattern_sum_b[alt_pid] = alt_sum_b;
                        ws.pattern_sum_fb[alt_pid] = alt_sum_fb;
                        ws.pattern_state_count[alt_pid] = alt_count as f32;
                        let miss_pid = missing_pid as usize;
                        ws.pattern_sum_f[miss_pid] = miss_sum_f;
                        ws.pattern_sum_b[miss_pid] = miss_sum_b;
                        ws.pattern_sum_fb[miss_pid] = miss_sum_fb;
                        ws.pattern_state_count[miss_pid] = miss_count as f32;
                        *pattern_sum_key = dense_key;
                        self.dense_identity_capture(col, active_states, dense_key_u64);
                    } else {
                        self.dense_identity_invalidate();
                        ws.pattern_sum_f[..n_groups].fill(0.0);
                        ws.pattern_sum_b[..n_groups].fill(0.0);
                        ws.pattern_sum_fb[..n_groups].fill(0.0);
                        ws.pattern_state_count[..n_groups].fill(0.0);
                        // For subsetted/non-identity states, dense bitset scans can
                        // over-touch panel space; stick to exact state-loop path.
                        let n_haps = col.n_haplotypes();
                        let bits = col.bits_raw();
                        let missing = col.missing_raw();
                        let mut cached_word_idx = usize::MAX;
                        let mut cached_bits_word = 0u64;
                        let mut cached_missing_word = 0u64;
                        for (i, hap) in state_haps.iter().take(active_states).enumerate() {
                            let idx = hap.as_usize();
                            let ref_allele = if idx >= n_haps {
                                AlleleCode::MISSING.raw()
                            } else {
                                let word_idx = idx >> 6;
                                let bit_idx = idx & 63;
                                if word_idx != cached_word_idx {
                                    cached_word_idx = word_idx;
                                    cached_bits_word = bits.get(word_idx).copied().unwrap_or(0);
                                    cached_missing_word =
                                        missing.get(word_idx).copied().unwrap_or(0);
                                }
                                if ((cached_missing_word >> bit_idx) & 1) != 0 {
                                    AlleleCode::MISSING.raw()
                                } else {
                                    ((cached_bits_word >> bit_idx) & 1) as u8
                                }
                            };
                            ws.state_alleles[i] = ref_allele;
                            let pid = Self::allele_to_pid(ref_allele, n_alleles, missing_pid);
                            ws.state_patterns[i] = pid as u16;
                            let f = ws.fwd[i];
                            let b = ws.bwd[i];
                            ws.pattern_sum_f[pid] += f;
                            ws.pattern_sum_b[pid] += b;
                            ws.pattern_sum_fb[pid] += if use_boundary_fb_products {
                                ws.boundary_fb_products[i]
                            } else {
                                f * b
                            };
                            ws.pattern_state_count[pid] += 1.0;
                        }
                        *pattern_sum_key = m.wrapping_add(1);
                    }
                } else {
                    self.ensure_sparse_state_index(state_haps, active_states);
                    if self.sparse_identity_order {
                        let dense_key_u64 = Self::dense_identity_key(col, active_states, n_alleles);
                        let dense_key = dense_key_u64 as usize;
                        if *pattern_sum_key == dense_key
                            && self.dense_identity_match(col, active_states, dense_key_u64)
                        {
                            return PreparedGroups {
                                key: dense_key,
                                n_groups,
                                direct_alleles_ptr: std::ptr::null(),
                                direct_alleles_len: 0,
                            };
                        }
                        ws.pattern_sum_f[..n_groups].fill(0.0);
                        ws.pattern_sum_b[..n_groups].fill(0.0);
                        ws.pattern_sum_fb[..n_groups].fill(0.0);
                        ws.pattern_state_count[..n_groups].fill(0.0);
                        let bits = col.bits_raw();
                        let missing = col.missing_raw();
                        let fb_products = if use_boundary_fb_products {
                            Some(&ws.boundary_fb_products[..active_states])
                        } else {
                            None
                        };
                        let (
                            alt_sum_f,
                            alt_sum_b,
                            alt_sum_fb,
                            alt_count,
                            miss_sum_f,
                            miss_sum_b,
                            miss_sum_fb,
                            miss_count,
                        ) = {
                            #[cfg(target_arch = "x86_64")]
                            {
                                if active_states >= 16 && std::is_x86_feature_detected!("avx512f") {
                                    // Safety:
                                    // - guarded by runtime feature check.
                                    // - slices are valid for [0, active_states).
                                    unsafe {
                                        dense_identity_biallelic_sums_avx512(
                                            active_states,
                                            bits,
                                            missing,
                                            &ws.fwd[..active_states],
                                            &ws.bwd[..active_states],
                                            fb_products,
                                        )
                                    }
                                } else if active_states >= 8
                                    && std::is_x86_feature_detected!("avx2")
                                {
                                    // Safety:
                                    // - guarded by runtime feature check.
                                    // - slices are valid for [0, active_states).
                                    unsafe {
                                        dense_identity_biallelic_sums_avx2(
                                            active_states,
                                            bits,
                                            missing,
                                            &ws.fwd[..active_states],
                                            &ws.bwd[..active_states],
                                            fb_products,
                                        )
                                    }
                                } else {
                                    dense_identity_biallelic_sums_scalar(
                                        active_states,
                                        bits,
                                        missing,
                                        &ws.fwd[..active_states],
                                        &ws.bwd[..active_states],
                                        fb_products,
                                    )
                                }
                            }
                            #[cfg(not(target_arch = "x86_64"))]
                            {
                                dense_identity_biallelic_sums_scalar(
                                    active_states,
                                    bits,
                                    missing,
                                    &ws.fwd[..active_states],
                                    &ws.bwd[..active_states],
                                    fb_products,
                                )
                            }
                        };

                        let mut sum_f_all = 0.0f32;
                        let mut sum_b_all = 0.0f32;
                        let mut sum_fb_all = 0.0f32;
                        if let Some(fb) = fb_products {
                            for i in 0..active_states {
                                sum_f_all += ws.fwd[i];
                                sum_b_all += ws.bwd[i];
                                sum_fb_all += fb[i];
                            }
                        } else {
                            for i in 0..active_states {
                                let f = ws.fwd[i];
                                let b = ws.bwd[i];
                                sum_f_all += f;
                                sum_b_all += b;
                                sum_fb_all += f * b;
                            }
                        }
                        let ref_sum_f = (sum_f_all - alt_sum_f - miss_sum_f).max(0.0);
                        let ref_sum_b = (sum_b_all - alt_sum_b - miss_sum_b).max(0.0);
                        let ref_sum_fb = (sum_fb_all - alt_sum_fb - miss_sum_fb).max(0.0);
                        let ref_count = active_states.saturating_sub(alt_count + miss_count);
                        let ref_pid = Self::allele_to_pid(0, n_alleles, missing_pid);
                        ws.pattern_sum_f[ref_pid] = ref_sum_f;
                        ws.pattern_sum_b[ref_pid] = ref_sum_b;
                        ws.pattern_sum_fb[ref_pid] = ref_sum_fb;
                        ws.pattern_state_count[ref_pid] = ref_count as f32;
                        let alt_pid = Self::allele_to_pid(1, n_alleles, missing_pid);
                        ws.pattern_sum_f[alt_pid] = alt_sum_f;
                        ws.pattern_sum_b[alt_pid] = alt_sum_b;
                        ws.pattern_sum_fb[alt_pid] = alt_sum_fb;
                        ws.pattern_state_count[alt_pid] = alt_count as f32;
                        let miss_pid = missing_pid as usize;
                        ws.pattern_sum_f[miss_pid] = miss_sum_f;
                        ws.pattern_sum_b[miss_pid] = miss_sum_b;
                        ws.pattern_sum_fb[miss_pid] = miss_sum_fb;
                        ws.pattern_state_count[miss_pid] = miss_count as f32;
                        *pattern_sum_key = dense_key;
                        self.dense_identity_capture(col, active_states, dense_key_u64);
                    } else {
                        self.dense_identity_invalidate();
                        ws.pattern_sum_f[..n_groups].fill(0.0);
                        ws.pattern_sum_b[..n_groups].fill(0.0);
                        ws.pattern_sum_fb[..n_groups].fill(0.0);
                        ws.pattern_state_count[..n_groups].fill(0.0);
                        let n_haps = col.n_haplotypes();
                        let bits = col.bits_raw();
                        let missing = col.missing_raw();
                        let mut cached_word_idx = usize::MAX;
                        let mut cached_bits_word = 0u64;
                        let mut cached_missing_word = 0u64;
                        for (i, hap) in state_haps.iter().take(active_states).enumerate() {
                            let idx = hap.as_usize();
                            let ref_allele = if idx >= n_haps {
                                AlleleCode::MISSING.raw()
                            } else {
                                let word_idx = idx >> 6;
                                let bit_idx = idx & 63;
                                if word_idx != cached_word_idx {
                                    cached_word_idx = word_idx;
                                    cached_bits_word = bits.get(word_idx).copied().unwrap_or(0);
                                    cached_missing_word =
                                        missing.get(word_idx).copied().unwrap_or(0);
                                }
                                if ((cached_missing_word >> bit_idx) & 1) != 0 {
                                    AlleleCode::MISSING.raw()
                                } else {
                                    ((cached_bits_word >> bit_idx) & 1) as u8
                                }
                            };
                            ws.state_alleles[i] = ref_allele;
                            let pid = Self::allele_to_pid(ref_allele, n_alleles, missing_pid);
                            ws.state_patterns[i] = pid as u16;
                            let f = ws.fwd[i];
                            let b = ws.bwd[i];
                            ws.pattern_sum_f[pid] += f;
                            ws.pattern_sum_b[pid] += b;
                            ws.pattern_sum_fb[pid] += if use_boundary_fb_products {
                                ws.boundary_fb_products[i]
                            } else {
                                f * b
                            };
                            ws.pattern_state_count[pid] += 1.0;
                        }
                        *pattern_sum_key = m.wrapping_add(1);
                    }
                }
                return PreparedGroups {
                    key: *pattern_sum_key,
                    n_groups,
                    direct_alleles_ptr: std::ptr::null(),
                    direct_alleles_len: 0,
                };
            }
            GenotypeColumn::Sparse(col) if n_alleles <= 2 => {
                self.pattern_signature_invalidate();
                self.dense_identity_invalidate();
                self.ensure_sparse_state_index(state_haps, active_states);
                ws.pattern_sum_f[..n_groups].fill(0.0);
                ws.pattern_sum_b[..n_groups].fill(0.0);
                ws.pattern_sum_fb[..n_groups].fill(0.0);
                ws.pattern_state_count[..n_groups].fill(0.0);
                let fb_products = if use_boundary_fb_products {
                    Some(&ws.boundary_fb_products[..active_states])
                } else {
                    None
                };
                let carriers = col.carriers();
                let mut carrier_sum_f = 0.0f32;
                let mut carrier_sum_b = 0.0f32;
                let mut carrier_sum_fb = 0.0f32;
                let mut carrier_count = 0usize;
                if let Some(fb) = fb_products {
                    for carrier in carriers {
                        let h = carrier.as_usize();
                        let state_idx = if self.sparse_identity_order {
                            if h >= active_states {
                                continue;
                            }
                            Some(h)
                        } else if h < self.sparse_state_index.len() {
                            let mapped = self.sparse_state_index[h];
                            if mapped >= 0 {
                                Some(mapped as usize)
                            } else {
                                None
                            }
                        } else {
                            None
                        };
                        if let Some(state_idx) = state_idx {
                            carrier_sum_f += ws.fwd[state_idx];
                            carrier_sum_b += ws.bwd[state_idx];
                            carrier_sum_fb += fb[state_idx];
                            carrier_count += 1;
                        }
                    }
                } else {
                    for carrier in carriers {
                        let h = carrier.as_usize();
                        let state_idx = if self.sparse_identity_order {
                            if h >= active_states {
                                continue;
                            }
                            Some(h)
                        } else if h < self.sparse_state_index.len() {
                            let mapped = self.sparse_state_index[h];
                            if mapped >= 0 {
                                Some(mapped as usize)
                            } else {
                                None
                            }
                        } else {
                            None
                        };
                        if let Some(state_idx) = state_idx {
                            let f = ws.fwd[state_idx];
                            let b = ws.bwd[state_idx];
                            carrier_sum_f += f;
                            carrier_sum_b += b;
                            carrier_sum_fb += f * b;
                            carrier_count += 1;
                        }
                    }
                }

                let (sum_f_all, sum_b_all, sum_fb_all) = if let Some(totals) = checkpoint_totals {
                    (totals[0], totals[1], totals[2])
                } else {
                    let mut total_f = 0.0f32;
                    let mut total_b = 0.0f32;
                    let mut total_fb = 0.0f32;
                    if let Some(fb) = fb_products {
                        for i in 0..active_states {
                            total_f += ws.fwd[i];
                            total_b += ws.bwd[i];
                            total_fb += fb[i];
                        }
                    } else {
                        for i in 0..active_states {
                            let f = ws.fwd[i];
                            let b = ws.bwd[i];
                            total_f += f;
                            total_b += b;
                            total_fb += f * b;
                        }
                    }
                    (total_f, total_b, total_fb)
                };

                let (
                    ref_sum_f,
                    ref_sum_b,
                    ref_sum_fb,
                    ref_count,
                    alt_sum_f,
                    alt_sum_b,
                    alt_sum_fb,
                    alt_count,
                ) = if col.is_inverted() {
                    (
                        carrier_sum_f,
                        carrier_sum_b,
                        carrier_sum_fb,
                        carrier_count,
                        (sum_f_all - carrier_sum_f).max(0.0),
                        (sum_b_all - carrier_sum_b).max(0.0),
                        (sum_fb_all - carrier_sum_fb).max(0.0),
                        active_states.saturating_sub(carrier_count),
                    )
                } else {
                    (
                        (sum_f_all - carrier_sum_f).max(0.0),
                        (sum_b_all - carrier_sum_b).max(0.0),
                        (sum_fb_all - carrier_sum_fb).max(0.0),
                        active_states.saturating_sub(carrier_count),
                        carrier_sum_f,
                        carrier_sum_b,
                        carrier_sum_fb,
                        carrier_count,
                    )
                };
                let ref_pid = Self::allele_to_pid(0, n_alleles, missing_pid);
                ws.pattern_sum_f[ref_pid] = ref_sum_f;
                ws.pattern_sum_b[ref_pid] = ref_sum_b;
                ws.pattern_sum_fb[ref_pid] = ref_sum_fb;
                ws.pattern_state_count[ref_pid] = ref_count as f32;
                let alt_pid = Self::allele_to_pid(1, n_alleles, missing_pid);
                ws.pattern_sum_f[alt_pid] = alt_sum_f;
                ws.pattern_sum_b[alt_pid] = alt_sum_b;
                ws.pattern_sum_fb[alt_pid] = alt_sum_fb;
                ws.pattern_state_count[alt_pid] = alt_count as f32;
            }
            _ => {
                self.pattern_signature_invalidate();
                // Full-support fallback (multiallelic, dictionary-backed, etc.).
                let prepared = Self::prepare_dense_groups(
                    ws,
                    m,
                    state_haps,
                    ref_columns,
                    n_alleles,
                    active_states,
                );
                if use_boundary_fb_products {
                    fill_pattern_sums_with_products(
                        &ws.state_patterns[..active_states],
                        active_states,
                        &ws.fwd[..active_states],
                        &ws.bwd[..active_states],
                        &ws.boundary_fb_products[..active_states],
                        &mut ws.pattern_sum_f[..n_groups],
                        &mut ws.pattern_sum_b[..n_groups],
                        &mut ws.pattern_sum_fb[..n_groups],
                        &mut ws.pattern_state_count[..n_groups],
                    );
                } else {
                    fill_pattern_sums(
                        &ws.state_patterns[..active_states],
                        active_states,
                        &ws.fwd[..active_states],
                        &ws.bwd[..active_states],
                        &mut ws.pattern_sum_f[..n_groups],
                        &mut ws.pattern_sum_b[..n_groups],
                        &mut ws.pattern_sum_fb[..n_groups],
                        &mut ws.pattern_state_count[..n_groups],
                    );
                }
                *pattern_sum_key = prepared.key;
                return prepared;
            }
        }

        *pattern_sum_key = m.wrapping_add(1);
        PreparedGroups {
            key: m.wrapping_add(1),
            n_groups,
            direct_alleles_ptr: std::ptr::null(),
            direct_alleles_len: 0,
        }
    }
}

trait PatternSource {
    type Stamp: Copy + PartialEq;
    const LABEL: &'static str;

    fn null_stamp() -> Self::Stamp;

    fn forward_update(
        stamp: &mut Self::Stamp,
        ws: &mut ImputeWorkspace,
        m: usize,
        state_haps: &[RefHapId],
        ref_columns: &[GenotypeColumn],
        target_probs: &TargetAlleleProbs,
        p_recomb: &[f32],
        current_error: f32,
        active_states: usize,
        transition_haps: usize,
        transition_lambda: f32,
    ) -> f32;

    fn prepare_patterns(
        stamp: &mut Self::Stamp,
        ws: &mut ImputeWorkspace,
        m: usize,
        state_haps: &[RefHapId],
        ref_columns: &[GenotypeColumn],
    ) -> PreparedGroups;
}

#[derive(Default)]
struct SeqcodedSource;

impl PatternSource for SeqcodedSource {
    type Stamp = *const u16;
    const LABEL: &'static str = "seqcoded";

    #[inline]
    fn null_stamp() -> Self::Stamp {
        std::ptr::null()
    }

    #[inline]
    fn forward_update(
        stamp: &mut Self::Stamp,
        ws: &mut ImputeWorkspace,
        m: usize,
        state_haps: &[RefHapId],
        ref_columns: &[GenotypeColumn],
        target_probs: &TargetAlleleProbs,
        p_recomb: &[f32],
        current_error: f32,
        active_states: usize,
        transition_haps: usize,
        transition_lambda: f32,
    ) -> f32 {
        forward_update_seqcoded(
            ws,
            m,
            state_haps,
            ref_columns,
            target_probs,
            p_recomb,
            current_error,
            active_states,
            transition_haps,
            transition_lambda,
            stamp,
        )
    }

    #[inline]
    fn prepare_patterns(
        stamp: &mut Self::Stamp,
        ws: &mut ImputeWorkspace,
        m: usize,
        state_haps: &[RefHapId],
        ref_columns: &[GenotypeColumn],
    ) -> PreparedGroups {
        let col = seqcoded_col(&ref_columns[m]);
        let key_ptr = col.hap_to_seq().as_ptr();
        let key = key_ptr as usize;
        refresh_seq_patterns(col, stamp, state_haps, &mut ws.state_patterns);
        let seq_alleles = col.seq_alleles();
        let n_patterns = seq_alleles.len();
        PreparedGroups {
            key,
            n_groups: n_patterns,
            direct_alleles_ptr: seq_alleles.as_ptr(),
            direct_alleles_len: n_patterns,
        }
    }
}

#[derive(Default)]
struct DictionarySource;

impl PatternSource for DictionarySource {
    type Stamp = *const DictionaryColumn;
    const LABEL: &'static str = "dictionary";

    #[inline]
    fn null_stamp() -> Self::Stamp {
        std::ptr::null()
    }

    #[inline]
    fn forward_update(
        stamp: &mut Self::Stamp,
        ws: &mut ImputeWorkspace,
        m: usize,
        state_haps: &[RefHapId],
        ref_columns: &[GenotypeColumn],
        target_probs: &TargetAlleleProbs,
        p_recomb: &[f32],
        current_error: f32,
        active_states: usize,
        transition_haps: usize,
        transition_lambda: f32,
    ) -> f32 {
        forward_update_dict(
            ws,
            m,
            state_haps,
            ref_columns,
            target_probs,
            p_recomb,
            current_error,
            active_states,
            transition_haps,
            transition_lambda,
            stamp,
        )
    }

    #[inline]
    fn prepare_patterns(
        stamp: &mut Self::Stamp,
        ws: &mut ImputeWorkspace,
        m: usize,
        state_haps: &[RefHapId],
        ref_columns: &[GenotypeColumn],
    ) -> PreparedGroups {
        let col = dict_col_ref(&ref_columns[m]);
        let key = col.col as *const DictionaryColumn as usize;
        let n_patterns = col.col.n_patterns();
        refresh_dict_patterns(
            &col,
            stamp,
            state_haps,
            &mut ws.state_patterns,
            &mut ws.dict_pattern_alleles,
        );
        PreparedGroups {
            key,
            n_groups: n_patterns,
            direct_alleles_ptr: std::ptr::null(),
            direct_alleles_len: 0,
        }
    }
}

struct PatternKernel<S: PatternSource> {
    forward_stamp: S::Stamp,
    backward_stamp: S::Stamp,
    source_marker: std::marker::PhantomData<S>,
}

impl<S: PatternSource> Default for PatternKernel<S> {
    fn default() -> Self {
        Self {
            forward_stamp: S::null_stamp(),
            backward_stamp: S::null_stamp(),
            source_marker: std::marker::PhantomData,
        }
    }
}

impl<S: PatternSource> ImputeKernel for PatternKernel<S> {
    type MarkerCtx = PreparedGroups;
    const LABEL: &'static str = S::LABEL;
    const CLAMP_AFFINE_GROUP_MASS: bool = false;
    const MAX_NON_MISSING_ALLELES: usize = AlleleCode::MISSING.raw() as usize;

    #[inline]
    fn reset_forward(&mut self) {
        self.forward_stamp = S::null_stamp();
    }

    #[inline]
    fn reset_backward(&mut self) {
        self.backward_stamp = S::null_stamp();
    }

    #[inline]
    fn forward_update(
        &mut self,
        ws: &mut ImputeWorkspace,
        m: usize,
        state_haps: &[RefHapId],
        ref_columns: &[GenotypeColumn],
        target_probs: &TargetAlleleProbs,
        p_recomb: &[f32],
        current_error: f32,
        active_states: usize,
        transition_haps: usize,
        transition_lambda: f32,
    ) -> f32 {
        S::forward_update(
            &mut self.forward_stamp,
            ws,
            m,
            state_haps,
            ref_columns,
            target_probs,
            p_recomb,
            current_error,
            active_states,
            transition_haps,
            transition_lambda,
        )
    }

    #[inline]
    fn prepare_marker(
        &mut self,
        ws: &mut ImputeWorkspace,
        m: usize,
        state_haps: &[RefHapId],
        ref_columns: &[GenotypeColumn],
        n_alleles: usize,
        active_states: usize,
    ) -> Self::MarkerCtx {
        assert!(active_states <= state_haps.len());
        assert!(n_alleles <= Self::MAX_NON_MISSING_ALLELES);
        S::prepare_patterns(
            &mut self.backward_stamp,
            ws,
            m,
            &state_haps[..active_states],
            ref_columns,
        )
    }

    #[inline]
    fn marker_key(ctx: &Self::MarkerCtx) -> usize {
        ctx.key
    }

    #[inline]
    fn marker_group_count(ctx: &Self::MarkerCtx) -> usize {
        ctx.n_groups
    }

    #[inline]
    fn group_alleles<'a>(ctx: &Self::MarkerCtx, dict_pattern_alleles: &'a [u8]) -> &'a [u8] {
        if ctx.direct_alleles_ptr.is_null() {
            &dict_pattern_alleles[..ctx.n_groups]
        } else {
            // SeqCoded context points directly at stable per-marker pattern alleles.
            unsafe { std::slice::from_raw_parts(ctx.direct_alleles_ptr, ctx.direct_alleles_len) }
        }
    }
}

#[inline]
fn write_posterior_from_probs(dst: &mut AllelePosteriors, probs: &[f32]) {
    if probs.len() == 2 {
        *dst = AllelePosteriors::Biallelic(probs[1]);
    } else {
        let mut out = Vec::with_capacity(probs.len());
        out.extend_from_slice(probs);
        *dst = AllelePosteriors::Multiallelic(std::sync::Arc::from(out));
    }
}

fn run_hmm_with_kernel<K: ImputeKernel>(
    mut kernel: K,
    state_haps: &[RefHapId],
    ref_columns: &[GenotypeColumn],
    target_probs: &TargetAlleleProbs,
    p_recomb: &[f32],
    error_rate: f32,
    prior_marker_idx: Option<usize>,
    state_priors: Option<&[f32]>,
    ref_allele_freqs: &RefAlleleFreqs,
    transition_haps: usize,
    transition_lambda: f32,
    context: ImputeHmmContext,
    smoothing_cluster_cm: f32,
    external_nearest_obs_retain: Option<&[f32]>,
    ws: &mut ImputeWorkspace,
) -> Result<(Vec<AllelePosteriors>, Option<Vec<f32>>)> {
    validate_target_probs_nonempty(target_probs, context, K::LABEL)?;
    validate_reference_marker_count(ref_columns.len(), target_probs, context, K::LABEL)?;
    if !smoothing_cluster_cm.is_finite() || smoothing_cluster_cm <= 0.0 {
        return Err(ReagleError::vcf(format!(
            "Invalid smoothing_cluster_cm in imputation HMM ({}): window={} sample={} hap={} value={}",
            K::LABEL,
            context.window_idx,
            context.sample_idx,
            context.hap_idx,
            smoothing_cluster_cm
        )));
    }

    let n_states = state_haps.len();
    let n_markers = target_probs.n_markers();
    ws.resize(n_states, n_markers);
    ws.pattern_cache_clear();
    let active_states = ws.active_states();
    assert!(active_states <= state_haps.len());
    let active_markers = ws.active_markers();
    if active_states > 0 {
        ws.weights[..active_states].fill(1.0);
    }

    let n_ref_haps = ref_allele_freqs.n_ref_haps().max(1);
    assert!(
        transition_haps == n_ref_haps,
        "imputation transition_haps must equal n_ref_haps: got transition_haps={} n_ref_haps={} window={} sample={} hap={} kernel={}",
        transition_haps,
        n_ref_haps,
        context.window_idx,
        context.sample_idx,
        context.hap_idx,
        K::LABEL
    );
    let transition_haps = n_ref_haps;
    let use_prior_smoothing = target_probs.has_untyped_markers();
    if use_prior_smoothing {
        if let Some(ext_retain) = external_nearest_obs_retain {
            // Use pre-computed retain from the full I/O window to avoid
            // edge-biased smoothing at piecewise segment boundaries.
            let n = target_probs.n_markers();
            ws.nearest_obs_retain.resize(n, 0.0);
            let copy_len = ext_retain.len().min(n);
            ws.nearest_obs_retain[..copy_len].copy_from_slice(&ext_retain[..copy_len]);
            // If external retain is shorter (shouldn't happen), fill remainder
            // with 0.0 to trigger maximum smoothing as a safe fallback.
            if copy_len < n {
                ws.nearest_obs_retain[copy_len..n].fill(0.0);
            }
        } else {
            compute_nearest_observed_lambda(ws, target_probs, p_recomb, smoothing_cluster_cm);
        }
    } else {
        ws.nearest_obs_retain.clear();
    }
    let uniform_mask = build_uniform_mask(target_probs, active_markers);
    let skip_untyped_mask = build_skip_untyped_mask(
        target_probs,
        &ws.nearest_obs_retain,
        &uniform_mask,
        use_prior_smoothing,
    );
    let panel_priors = target_probs.panel_priors();
    let checkpoint_grid = build_checkpoint_markers(&uniform_mask, prior_marker_idx, active_markers);
    ws.ensure_typed_checkpoints(active_states, checkpoint_grid.len());
    ws.ensure_affine_window_cache(
        p_recomb,
        &checkpoint_grid,
        active_states,
        transition_haps,
        transition_lambda,
        active_markers,
    );

    let mut final_posteriors: Vec<AllelePosteriors> = Vec::new();
    let mut final_prior_state_post: Option<Vec<f32>> = None;
    let mut warned_af_fallback = false;
    let mut warned_structural_invariant = false;
    let mut structural_invariant_violations = 0usize;
    let current_error = error_rate;

    let final_pass = 0usize;
    for pass in 0..1 {
        let is_final = pass == final_pass;
        if let Some(priors) = state_priors {
            let len = priors.len().min(active_states);
            ws.fwd[..len].copy_from_slice(&priors[..len]);
            if len < active_states {
                ws.fwd[len..active_states].fill(0.0);
            }
            normalize_probs(&mut ws.fwd[..active_states]);
        } else {
            let uniform = 1.0 / active_states.max(1) as f32;
            ws.fwd[..active_states].fill(uniform);
        }

        kernel.reset_forward();
        let mut prev_marker = 0usize;
        for (cp_idx, marker) in checkpoint_grid.iter_forward() {
            let m = marker.as_usize();
            if m > prev_marker {
                batched_transition_forward(
                    &mut ws.fwd[..active_states],
                    p_recomb,
                    prev_marker,
                    m,
                    active_states,
                    transition_haps,
                    transition_lambda,
                );
            }
            kernel.forward_update(
                ws,
                m,
                state_haps,
                ref_columns,
                target_probs,
                p_recomb,
                current_error,
                active_states,
                transition_haps,
                transition_lambda,
            );
            ws.store_checkpoint(cp_idx, active_states);
            prev_marker = m + 1;
        }
        if prev_marker < active_markers {
            batched_transition_forward(
                &mut ws.fwd[..active_states],
                p_recomb,
                prev_marker,
                active_markers,
                active_states,
                transition_haps,
                transition_lambda,
            );
        }

        let mut posteriors: Vec<AllelePosteriors> = Vec::new();
        if is_final {
            posteriors.reserve(n_markers);
            posteriors.resize_with(n_markers, || AllelePosteriors::Biallelic(0.0));
        }

        ws.bwd.fill(1.0);
        let mut bwd_sum = active_states as f32;
        if active_markers > 0 {
            kernel.reset_backward();
            for cp_idx in checkpoint_grid.rev_indices() {
                let mut pattern_sum_key: usize = usize::MAX;
                let block = checkpoint_grid.block_view(cp_idx, active_markers);
                let block_start = block.start_usize();
                let block_end = block.end_usize();
                let block_len = block.len();
                let cached_affine = ws
                    .affine_window_cache
                    .as_ref()
                    .and_then(|cache| cache.by_checkpoint.get(cp_idx.as_usize()))
                    .and_then(|entry| entry.as_ref().map(Arc::clone));
                ws.load_checkpoint(cp_idx, active_states);
                let mut fwd_sum_left = 0.0f32;
                for v in &ws.fwd[..active_states] {
                    fwd_sum_left += *v;
                }
                let mut bwd_sum_left = 0.0f32;
                let mut fb_sum_left = 0.0f32;
                ws.ensure_boundary_fb_products(active_states);
                for i in 0..active_states {
                    let b = ws.bwd[i];
                    bwd_sum_left += b;
                    let fb = ws.fwd[i] * b;
                    ws.boundary_fb_products[i] = fb;
                    fb_sum_left += fb;
                }
                let checkpoint_totals = Some([fwd_sum_left, bwd_sum_left, fb_sum_left]);
                assert!(
                    { (fwd_sum_left - 1.0).abs() < 1e-3 || active_states == 0 },
                    "checkpoint forward mass drift before interior affine block (S_u assumption)"
                );
                if let Some(interior) = UniformInteriorRange::from_block_checked(
                    block,
                    &uniform_mask,
                    context,
                    K::LABEL,
                )? {
                    // Affine interior path:
                    // - Enabled by marker kind (uniform/untyped interior), not by distance.
                    // - We keep transition propagation in affine form across the block.
                    // - Final-pass posterior emission may still short-circuit to panel priors
                    //   at deep untyped markers based on distance-to-anchor lambda.
                    let coeffs = cached_affine.ok_or_else(|| {
                        ReagleError::vcf(format!(
                            "Missing affine cache block in imputation HMM ({}): window={} sample={} hap={} checkpoint={} block=[{}, {})",
                            K::LABEL,
                            context.window_idx,
                            context.sample_idx,
                            context.hap_idx,
                            cp_idx.as_usize(),
                            block_start,
                            block_end
                        ))
                    })?;
                    if coeffs.block_start != block_start
                        || coeffs.block_end != block_end
                        || coeffs.fwd_a.len() != block_len
                    {
                        return Err(ReagleError::vcf(format!(
                            "Affine cache block mismatch in imputation HMM ({}): window={} sample={} hap={} checkpoint={} expected=[{}, {}) got=[{}, {})",
                            K::LABEL,
                            context.window_idx,
                            context.sample_idx,
                            context.hap_idx,
                            cp_idx.as_usize(),
                            block_start,
                            block_end,
                            coeffs.block_start,
                            coeffs.block_end
                        )));
                    }
                    let bwd_sum_right = bwd_sum;
                    for m_ix in interior.iter() {
                        let m = m_ix.as_usize();
                        // Interior marker math summary:
                        //
                        // 1) Forward affine at marker m from left checkpoint:
                        //      F_m(i) = a_fwd * F_left(i) + b_fwd
                        // 2) Backward affine at marker m from right boundary:
                        //      B_m(i) = a_bwd_m * B_right(i) + b_bwd_m * sum(B_right)
                        // 3) Pattern/group posterior mass is evaluated by combining
                        //    these affine maps with precomputed boundary sufficient stats:
                        //      mass_g(m) = sum_{i in g} F_m(i) * B_m(i)
                        //                = alpha*sum_fb + beta*sum_f + gamma*sum_b + delta*count
                        //
                        // This preserves exact Li-Stephens transition-only inference
                        // inside the block while avoiding per-marker/state recomputation.
                        let fwd_slot = m - block_start;
                        let a_fwd = coeffs.fwd_a[fwd_slot] as f32;
                        let b_fwd = coeffs.fwd_b[fwd_slot] as f32;
                        let fwd_add = b_fwd * fwd_sum_left;

                        if is_final
                            && prior_marker_idx != Some(m)
                            && skip_untyped_mask[MarkerIx::new(m)]
                        {
                            write_panel_freq_posterior(&mut posteriors[m], panel_priors, m);
                            continue;
                        }

                        let probs = target_probs.probs_for_marker_trusted(m);
                        let n_alleles = probs.len();
                        if n_alleles > K::MAX_NON_MISSING_ALLELES {
                            return Err(ReagleError::vcf(format!(
                                "Marker allele count exceeds {} kernel capacity in imputation HMM ({}): window={} sample={} hap={} marker={} n_alleles={}",
                                K::MAX_NON_MISSING_ALLELES,
                                K::LABEL,
                                context.window_idx,
                                context.sample_idx,
                                context.hap_idx,
                                m,
                                n_alleles
                            )));
                        }
                        if prior_marker_idx == Some(m) {
                            ws.ensure_state_posterior_scratch(active_states);
                        }
                        if is_final && n_alleles > 0 {
                            ws.ensure_smoothing_prior_counts(n_alleles);
                        }

                        let prepared = kernel.prepare_marker_with_pattern_sums(
                            ws,
                            m,
                            state_haps,
                            ref_columns,
                            n_alleles,
                            active_states,
                            checkpoint_totals,
                            true,
                            &mut pattern_sum_key,
                        );
                        let n_groups = K::marker_group_count(&prepared);

                        let a_bwd_m = coeffs.bwd_a[m - block_start] as f32;
                        let b_bwd_m = coeffs.bwd_b_coeff[m - block_start] as f32;
                        let forward_affine = ForwardAffine {
                            a: a_fwd,
                            b: fwd_add,
                        };
                        let backward_affine = BackwardAffine::new(a_bwd_m, b_bwd_m, bwd_sum_right);
                        let alpha_coeff = forward_affine.a * backward_affine.a;
                        let beta_coeff = forward_affine.a * backward_affine.add;
                        let gamma_coeff = forward_affine.b * backward_affine.a;
                        let delta_coeff = forward_affine.b * backward_affine.add;

                        if prior_marker_idx == Some(m) {
                            let gamma = &mut ws.state_posterior_scratch[..active_states];
                            let mut sum = 0.0f32;
                            for i in 0..active_states {
                                let u = ws.fwd[i];
                                let v = ws.bwd[i];
                                let g = (alpha_coeff * (u * v)
                                    + beta_coeff * u
                                    + gamma_coeff * v
                                    + delta_coeff)
                                    .max(0.0);
                                gamma[i] = g;
                                sum += g;
                            }
                            if sum > 0.0 {
                                let inv = 1.0f32 / sum;
                                for g in gamma.iter_mut() {
                                    *g *= inv;
                                }
                                final_prior_state_post = Some(gamma.to_vec());
                            }
                        }

                        if is_final {
                            ws.allele_probs.clear();
                            if n_alleles > 0 {
                                ws.allele_probs.resize(n_alleles, 0.0f32);
                                let smoothing_prior_counts =
                                    &mut ws.smoothing_prior_counts[..n_alleles];
                                smoothing_prior_counts.fill(0.0);
                                let mut subset_total = 0.0f64;
                                let mut smoothing_prior_total = 0.0f32;
                                let mut total = 0.0f64;
                                let mut missing_ref_mass = 0.0f64;
                                let mut missing_ood_mass = 0.0f64;
                                let group_alleles =
                                    K::group_alleles(&prepared, &ws.dict_pattern_alleles);
                                let missing_raw = AlleleCode::MISSING.raw();
                                let allele_len = ws.allele_probs.len();
                                let groups_len = group_alleles.len();
                                let group_limit = n_groups.min(groups_len);
                                if n_groups > groups_len {
                                    structural_invariant_violations =
                                        structural_invariant_violations
                                            .saturating_add(n_groups - groups_len);
                                    if !warned_structural_invariant {
                                        eprintln!(
                                            "[warn] impute_hmm structural group mismatch: window={} sample={} hap={} marker={} kernel={} groups={} alleles={}",
                                            context.window_idx,
                                            context.sample_idx,
                                            context.hap_idx,
                                            m,
                                            K::LABEL,
                                            n_groups,
                                            groups_len
                                        );
                                        warned_structural_invariant = true;
                                    }
                                }
                                // Partition marker mass into:
                                //   subset_total    = represented allele mass Q
                                //   missing_ref_mass= M_ref (ref allele code u8::MAX)
                                //   missing_ood_mass= M_ood (allele index outside represented set)
                                // These are fed to the structural posterior update above.
                                for pid in 0..group_limit {
                                    let mut state_prob = alpha_coeff * ws.pattern_sum_fb[pid]
                                        + beta_coeff * ws.pattern_sum_f[pid]
                                        + gamma_coeff * ws.pattern_sum_b[pid]
                                        + delta_coeff * ws.pattern_state_count[pid];
                                    if K::CLAMP_AFFINE_GROUP_MASS {
                                        state_prob = if state_prob.is_finite() {
                                            state_prob.max(0.0)
                                        } else {
                                            0.0
                                        };
                                    }
                                    total += state_prob as f64;
                                    let ref_allele = unsafe { *group_alleles.get_unchecked(pid) };
                                    if ref_allele == missing_raw {
                                        missing_ref_mass += state_prob as f64;
                                        continue;
                                    }
                                    let idx = ref_allele as usize;
                                    if idx < allele_len {
                                        ws.allele_probs[idx] += state_prob;
                                        subset_total += state_prob as f64;
                                        smoothing_prior_counts[idx] += state_prob.max(0.0);
                                        smoothing_prior_total += state_prob.max(0.0);
                                    } else {
                                        // Out-of-domain allele mass uses prior-shrunk redistribution.
                                        missing_ood_mass += state_prob as f64;
                                    }
                                }
                                for pid in group_limit..n_groups {
                                    let mut state_prob = alpha_coeff * ws.pattern_sum_fb[pid]
                                        + beta_coeff * ws.pattern_sum_f[pid]
                                        + gamma_coeff * ws.pattern_sum_b[pid]
                                        + delta_coeff * ws.pattern_state_count[pid];
                                    if K::CLAMP_AFFINE_GROUP_MASS {
                                        state_prob = if state_prob.is_finite() {
                                            state_prob.max(0.0)
                                        } else {
                                            0.0
                                        };
                                    }
                                    total += state_prob as f64;
                                    missing_ref_mass += state_prob as f64;
                                }
                                if total > 0.0 {
                                    let subset_total_f32 = subset_total as f32;
                                    let missing_ref_mass_f32 = missing_ref_mass as f32;
                                    let missing_ood_mass_f32 = missing_ood_mass as f32;
                                    if subset_total_f32 > 0.0
                                        || missing_ref_mass_f32 > 0.0
                                        || missing_ood_mass_f32 > 0.0
                                    {
                                        normalize_allele_posterior_structural_missing(
                                            &mut ws.allele_probs,
                                            subset_total_f32,
                                            missing_ref_mass_f32,
                                            missing_ood_mass_f32,
                                            &mut ws.allele_prior_scratch,
                                            probs,
                                        );
                                    } else {
                                        // No subset support reached the represented allele space
                                        // (e.g., out-of-domain reference alleles). Fall back to
                                        // the target prior and keep downstream smoothing behavior.
                                        if !warned_af_fallback {
                                            eprintln!(
                                                "[warn] AF fallback in impute_hmm (no represented alleles): window={} sample={} hap={} marker={}",
                                                context.window_idx,
                                                context.sample_idx,
                                                context.hap_idx,
                                                m
                                            );
                                            warned_af_fallback = true;
                                        }
                                        let prior = normalized_allele_prior(
                                            &mut ws.allele_prior_scratch,
                                            probs,
                                        );
                                        ws.allele_probs.copy_from_slice(prior.as_slice());
                                    }
                                    if use_prior_smoothing {
                                        apply_marker_prior_smoothing(
                                            &mut ws.allele_probs,
                                            panel_priors,
                                            m,
                                            smoothing_prior_counts,
                                            smoothing_prior_total,
                                            &mut ws.allele_prior_scratch,
                                            probs.as_slice(),
                                            ws.nearest_obs_retain.get(m).copied().unwrap_or(0.0),
                                            target_probs.is_untyped_uniform_marker(m),
                                            subset_total_f32,
                                            missing_ref_mass_f32,
                                            missing_ood_mass_f32,
                                            active_states,
                                            transition_haps,
                                            target_probs.min_untyped_prior_mix(),
                                            &mut warned_af_fallback,
                                            context,
                                        );
                                    }
                                    write_posterior_from_probs(
                                        &mut posteriors[m],
                                        &ws.allele_probs,
                                    );
                                } else {
                                    if !warned_af_fallback {
                                        eprintln!(
                                            "[warn] posterior-mass fallback in impute_hmm ({}): window={} sample={} hap={} marker={} active_states={}",
                                            K::LABEL,
                                            context.window_idx,
                                            context.sample_idx,
                                            context.hap_idx,
                                            m,
                                            active_states
                                        );
                                        warned_af_fallback = true;
                                    }
                                    write_panel_freq_posterior(&mut posteriors[m], panel_priors, m);
                                }
                            } else {
                                return Err(ReagleError::vcf(format!(
                                    "No allele space available in imputation HMM ({}): window={} sample={} hap={} marker={} active_states={}",
                                    K::LABEL,
                                    context.window_idx,
                                    context.sample_idx,
                                    context.hap_idx,
                                    m,
                                    active_states
                                )));
                            }
                        }
                    }
                    bwd_sum = batched_transition_backward(
                        &mut ws.bwd[..active_states],
                        bwd_sum,
                        p_recomb,
                        block_start + 1,
                        block_end,
                        active_states,
                        transition_haps,
                        transition_lambda,
                    );
                }

                let m_rev = block_start;
                let probs = target_probs.probs_for_marker_trusted(m_rev);
                let recomb_rate = marker_recomb_rate(p_recomb, m_rev);
                let n_alleles = probs.len();
                if n_alleles > K::MAX_NON_MISSING_ALLELES {
                    return Err(ReagleError::vcf(format!(
                        "Marker allele count exceeds {} kernel capacity in imputation HMM ({}): window={} sample={} hap={} marker={} n_alleles={}",
                        K::MAX_NON_MISSING_ALLELES,
                        K::LABEL,
                        context.window_idx,
                        context.sample_idx,
                        context.hap_idx,
                        m_rev,
                        n_alleles
                    )));
                }
                if prior_marker_idx == Some(m_rev) {
                    ws.ensure_state_posterior_scratch(active_states);
                }
                if is_final && n_alleles > 0 {
                    ws.ensure_smoothing_prior_counts(n_alleles);
                }

                let prepared = kernel.prepare_marker(
                    ws,
                    m_rev,
                    state_haps,
                    ref_columns,
                    n_alleles,
                    active_states,
                );
                let group_alleles = K::group_alleles(&prepared, &ws.dict_pattern_alleles);

                if is_final
                    && prior_marker_idx != Some(m_rev)
                    && skip_untyped_mask[MarkerIx::new(m_rev)]
                {
                    write_panel_freq_posterior(&mut posteriors[m_rev], panel_priors, m_rev);
                } else {
                    let fwd_slice = &ws.fwd[..active_states];
                    if prior_marker_idx == Some(m_rev) {
                        let gamma = &mut ws.state_posterior_scratch[..active_states];
                        let mut sum = 0.0f32;
                        for i in 0..active_states {
                            let g = (fwd_slice[i] * ws.bwd[i]).max(0.0);
                            gamma[i] = g;
                            sum += g;
                        }
                        if sum > 0.0 {
                            let inv = 1.0f32 / sum;
                            for g in gamma.iter_mut() {
                                *g *= inv;
                            }
                            final_prior_state_post = Some(gamma.to_vec());
                        }
                    }

                    if is_final {
                        ws.allele_probs.clear();
                        if n_alleles > 0 {
                            ws.allele_probs.resize(n_alleles, 0.0f32);
                            let smoothing_prior_counts =
                                &mut ws.smoothing_prior_counts[..n_alleles];
                            smoothing_prior_counts.fill(0.0);
                            let mut subset_total = 0.0f64;
                            let mut smoothing_prior_total = 0.0f32;
                            let mut total = 0.0f64;
                            let mut missing_ref_mass = 0.0f64;
                            let mut missing_ood_mass = 0.0f64;
                            let missing_raw = AlleleCode::MISSING.raw();
                            let allele_len = ws.allele_probs.len();
                            let state_count = active_states.min(ws.state_patterns.len());
                            let groups_len = group_alleles.len();
                            let mut pid_oob = false;
                            if state_count > 0 {
                                for &pid_raw in &ws.state_patterns[..state_count] {
                                    if (pid_raw as usize) >= groups_len {
                                        pid_oob = true;
                                        break;
                                    }
                                }
                            }
                            if active_states > state_count {
                                structural_invariant_violations = structural_invariant_violations
                                    .saturating_add(active_states - state_count);
                                if !warned_structural_invariant {
                                    eprintln!(
                                        "[warn] impute_hmm structural state-pattern shortfall: window={} sample={} hap={} marker={} kernel={} active_states={} state_patterns={}",
                                        context.window_idx,
                                        context.sample_idx,
                                        context.hap_idx,
                                        m_rev,
                                        K::LABEL,
                                        active_states,
                                        ws.state_patterns.len()
                                    );
                                    warned_structural_invariant = true;
                                }
                            }
                            if pid_oob {
                                structural_invariant_violations =
                                    structural_invariant_violations.saturating_add(1);
                                if !warned_structural_invariant {
                                    eprintln!(
                                        "[warn] impute_hmm structural group mismatch: window={} sample={} hap={} marker={} kernel={} active_states={} alleles={}",
                                        context.window_idx,
                                        context.sample_idx,
                                        context.hap_idx,
                                        m_rev,
                                        K::LABEL,
                                        active_states,
                                        groups_len
                                    );
                                    warned_structural_invariant = true;
                                }
                            }
                            // Same Q/M_ref/M_ood partition as interior markers, but on the
                            // explicit per-state boundary path (fwd*bwd state masses).
                            if !pid_oob {
                                for i in 0..state_count {
                                    let state_prob = fwd_slice[i] * ws.bwd[i];
                                    total += state_prob as f64;
                                    let pid = ws.state_patterns[i] as usize;
                                    let ref_allele = unsafe { *group_alleles.get_unchecked(pid) };
                                    if ref_allele == missing_raw {
                                        missing_ref_mass += state_prob as f64;
                                        continue;
                                    }
                                    let idx = ref_allele as usize;
                                    if idx < allele_len {
                                        ws.allele_probs[idx] += state_prob;
                                        subset_total += state_prob as f64;
                                        smoothing_prior_counts[idx] += state_prob.max(0.0);
                                        smoothing_prior_total += state_prob.max(0.0);
                                    } else {
                                        missing_ood_mass += state_prob as f64;
                                    }
                                }
                            } else {
                                for i in 0..state_count {
                                    let state_prob = fwd_slice[i] * ws.bwd[i];
                                    total += state_prob as f64;
                                    let pid = ws.state_patterns[i] as usize;
                                    let ref_allele = if pid < groups_len {
                                        unsafe { *group_alleles.get_unchecked(pid) }
                                    } else {
                                        missing_raw
                                    };
                                    if ref_allele == missing_raw {
                                        missing_ref_mass += state_prob as f64;
                                        continue;
                                    }
                                    let idx = ref_allele as usize;
                                    if idx < allele_len {
                                        ws.allele_probs[idx] += state_prob;
                                        subset_total += state_prob as f64;
                                        smoothing_prior_counts[idx] += state_prob.max(0.0);
                                        smoothing_prior_total += state_prob.max(0.0);
                                    } else {
                                        // Keep mass accounting consistent with interior path:
                                        // out-of-domain mass is tracked separately.
                                        missing_ood_mass += state_prob as f64;
                                    }
                                }
                            }
                            for i in state_count..active_states {
                                let state_prob = fwd_slice[i] * ws.bwd[i];
                                total += state_prob as f64;
                                missing_ref_mass += state_prob as f64;
                            }
                            if total > 0.0 {
                                let subset_total_f32 = subset_total as f32;
                                let missing_ref_mass_f32 = missing_ref_mass as f32;
                                let missing_ood_mass_f32 = missing_ood_mass as f32;
                                if subset_total_f32 > 0.0
                                    || missing_ref_mass_f32 > 0.0
                                    || missing_ood_mass_f32 > 0.0
                                {
                                    normalize_allele_posterior_structural_missing(
                                        &mut ws.allele_probs,
                                        subset_total_f32,
                                        missing_ref_mass_f32,
                                        missing_ood_mass_f32,
                                        &mut ws.allele_prior_scratch,
                                        probs,
                                    );
                                } else {
                                    if !warned_af_fallback {
                                        eprintln!(
                                            "[warn] AF fallback in impute_hmm (no represented alleles): window={} sample={} hap={} marker={}",
                                            context.window_idx,
                                            context.sample_idx,
                                            context.hap_idx,
                                            m_rev
                                        );
                                        warned_af_fallback = true;
                                    }
                                    let prior = normalized_allele_prior(
                                        &mut ws.allele_prior_scratch,
                                        probs,
                                    );
                                    ws.allele_probs.copy_from_slice(prior.as_slice());
                                }
                                if use_prior_smoothing {
                                    apply_marker_prior_smoothing(
                                        &mut ws.allele_probs,
                                        panel_priors,
                                        m_rev,
                                        smoothing_prior_counts,
                                        smoothing_prior_total,
                                        &mut ws.allele_prior_scratch,
                                        probs.as_slice(),
                                        ws.nearest_obs_retain.get(m_rev).copied().unwrap_or(0.0),
                                        target_probs.is_untyped_uniform_marker(m_rev),
                                        subset_total_f32,
                                        missing_ref_mass_f32,
                                        missing_ood_mass_f32,
                                        active_states,
                                        transition_haps,
                                        target_probs.min_untyped_prior_mix(),
                                        &mut warned_af_fallback,
                                        context,
                                    );
                                }
                                write_posterior_from_probs(
                                    &mut posteriors[m_rev],
                                    &ws.allele_probs,
                                );
                            } else {
                                if !warned_af_fallback {
                                    eprintln!(
                                        "[warn] posterior-mass fallback in impute_hmm ({}): window={} sample={} hap={} marker={} active_states={}",
                                        K::LABEL,
                                        context.window_idx,
                                        context.sample_idx,
                                        context.hap_idx,
                                        m_rev,
                                        active_states
                                    );
                                    warned_af_fallback = true;
                                }
                                write_panel_freq_posterior(
                                    &mut posteriors[m_rev],
                                    panel_priors,
                                    m_rev,
                                );
                            }
                        } else {
                            return Err(ReagleError::vcf(format!(
                                "No allele space available in imputation HMM ({}): window={} sample={} hap={} marker={} active_states={}",
                                K::LABEL,
                                context.window_idx,
                                context.sample_idx,
                                context.hap_idx,
                                m_rev,
                                active_states
                            )));
                        }
                    }
                }

                if uniform_mask[MarkerIx::new(m_rev)] {
                    bwd_sum = transition_only_backward_update(
                        &mut ws.bwd[..active_states],
                        recomb_rate,
                        transition_haps,
                        bwd_sum,
                        transition_lambda,
                    );
                } else {
                    let marker_error = target_probs
                        .marker_error_rate(m_rev)
                        .unwrap_or(current_error);
                    fill_pattern_emissions(
                        group_alleles,
                        probs,
                        marker_error,
                        &mut ws.emission_by_allele,
                        &mut ws.pattern_emissions,
                    );
                    let mut emit_beta_sum = 0.0f32;
                    for i in 0..active_states {
                        let pid = ws.state_patterns[i] as usize;
                        let emit = ws.pattern_emissions.get(pid).copied().unwrap_or(1.0);
                        ws.emissions[i] = emit;
                        emit_beta_sum += emit * ws.bwd[i];
                    }
                    let c_t = ws.fwd_scales.get(m_rev).copied().unwrap_or(1.0).max(1e-30);
                    let (stay_gap, shift_base) = subset_transition_params_adaptive_q(
                        recomb_rate,
                        active_states,
                        transition_haps,
                        transition_lambda,
                    );
                    let scale = stay_gap / c_t;
                    let shift = shift_base * (emit_beta_sum / c_t);
                    let mut new_sum = 0.0f32;
                    for i in 0..active_states {
                        ws.bwd[i] = scale * ws.emissions[i] * ws.bwd[i] + shift;
                        new_sum += ws.bwd[i];
                    }
                    bwd_sum = new_sum.max(1e-30);
                }
            }
        }

        if is_final {
            final_posteriors = posteriors;
        }
    }
    if structural_invariant_violations > 0 && !warned_structural_invariant {
        eprintln!(
            "[warn] impute_hmm structural invariant violations: window={} sample={} hap={} count={}",
            context.window_idx,
            context.sample_idx,
            context.hap_idx,
            structural_invariant_violations
        );
    }
    Ok((final_posteriors, final_prior_state_post))
}

/// Monomorphized HMM core over dense/sparse storage.
fn run_hmm_generic(
    state_haps: &[RefHapId],
    ref_columns: &[GenotypeColumn],
    target_probs: &TargetAlleleProbs,
    p_recomb: &[f32],
    error_rate: f32,
    prior_marker_idx: Option<usize>,
    state_priors: Option<&[f32]>,
    ref_allele_freqs: &RefAlleleFreqs,
    transition_haps: usize,
    transition_lambda: f32,
    context: ImputeHmmContext,
    smoothing_cluster_cm: f32,
    external_nearest_obs_retain: Option<&[f32]>,
    ws: &mut ImputeWorkspace,
) -> Result<(Vec<AllelePosteriors>, Option<Vec<f32>>)> {
    run_hmm_with_kernel(
        DenseKernel::default(),
        state_haps,
        ref_columns,
        target_probs,
        p_recomb,
        error_rate,
        prior_marker_idx,
        state_priors,
        ref_allele_freqs,
        transition_haps,
        transition_lambda,
        context,
        smoothing_cluster_cm,
        external_nearest_obs_retain,
        ws,
    )
}

fn run_hmm_seqcoded(
    state_haps: &[RefHapId],
    ref_columns: &[GenotypeColumn],
    target_probs: &TargetAlleleProbs,
    p_recomb: &[f32],
    error_rate: f32,
    prior_marker_idx: Option<usize>,
    state_priors: Option<&[f32]>,
    ref_allele_freqs: &RefAlleleFreqs,
    transition_haps: usize,
    transition_lambda: f32,
    context: ImputeHmmContext,
    smoothing_cluster_cm: f32,
    external_nearest_obs_retain: Option<&[f32]>,
    ws: &mut ImputeWorkspace,
) -> Result<(Vec<AllelePosteriors>, Option<Vec<f32>>)> {
    run_hmm_with_kernel(
        PatternKernel::<SeqcodedSource>::default(),
        state_haps,
        ref_columns,
        target_probs,
        p_recomb,
        error_rate,
        prior_marker_idx,
        state_priors,
        ref_allele_freqs,
        transition_haps,
        transition_lambda,
        context,
        smoothing_cluster_cm,
        external_nearest_obs_retain,
        ws,
    )
}

/// Dictionary-backed forward-backward kernel for imputation.
///
/// This delegates to the unified HMM driver with a dictionary-specific kernel.
fn run_hmm_dictionary(
    state_haps: &[RefHapId],
    ref_columns: &[GenotypeColumn],
    target_probs: &TargetAlleleProbs,
    p_recomb: &[f32],
    error_rate: f32,
    prior_marker_idx: Option<usize>,
    state_priors: Option<&[f32]>,
    ref_allele_freqs: &RefAlleleFreqs,
    transition_haps: usize,
    transition_lambda: f32,
    context: ImputeHmmContext,
    smoothing_cluster_cm: f32,
    external_nearest_obs_retain: Option<&[f32]>,
    ws: &mut ImputeWorkspace,
) -> Result<(Vec<AllelePosteriors>, Option<Vec<f32>>)> {
    run_hmm_with_kernel(
        PatternKernel::<DictionarySource>::default(),
        state_haps,
        ref_columns,
        target_probs,
        p_recomb,
        error_rate,
        prior_marker_idx,
        state_priors,
        ref_allele_freqs,
        transition_haps,
        transition_lambda,
        context,
        smoothing_cluster_cm,
        external_nearest_obs_retain,
        ws,
    )
}
/// Run forward-backward HMM and emit allele posteriors.
///
/// Returns (posteriors, optional state posterior at prior marker).
///
/// `external_nearest_obs_retain`: when `Some`, provides pre-computed
/// nearest-observed-marker retain values computed over the full I/O window.
/// This avoids edge-biased smoothing at piecewise segment boundaries where
/// the per-segment computation cannot see typed anchors in adjacent segments.
pub fn run_impute_hmm(
    state_haps: &[RefHapId],
    ref_columns: &[GenotypeColumn],
    target_probs: &TargetAlleleProbs,
    p_recomb: &[f32],
    error_rate: f32,
    prior_marker_idx: Option<usize>,
    state_priors: Option<&[f32]>,
    ref_allele_freqs: &RefAlleleFreqs,
    transition_haps: usize,
    transition_lambda: f32,
    context: ImputeHmmContext,
    smoothing_cluster_cm: f32,
    external_nearest_obs_retain: Option<&[f32]>,
    ws: &mut ImputeWorkspace,
) -> Result<(Vec<AllelePosteriors>, Option<Vec<f32>>)> {
    validate_reference_marker_count(ref_columns.len(), target_probs, context, "dispatch")?;
    if ref_columns.is_empty() {
        if target_probs.n_markers() > 0 {
            return Err(ReagleError::vcf(format!(
                "No reference columns available in imputation HMM: window={} sample={} hap={} markers={}",
                context.window_idx,
                context.sample_idx,
                context.hap_idx,
                target_probs.n_markers()
            )));
        }
        return Ok((Vec::new(), None));
    }

    if ref_columns
        .iter()
        .all(|col| matches!(col, GenotypeColumn::Dense(_)))
    {
        return run_hmm_generic(
            state_haps,
            ref_columns,
            target_probs,
            p_recomb,
            error_rate,
            prior_marker_idx,
            state_priors,
            ref_allele_freqs,
            transition_haps,
            transition_lambda,
            context,
            smoothing_cluster_cm,
            external_nearest_obs_retain,
            ws,
        );
    }

    if ref_columns
        .iter()
        .all(|col| matches!(col, GenotypeColumn::Sparse(_)))
    {
        return run_hmm_generic(
            state_haps,
            ref_columns,
            target_probs,
            p_recomb,
            error_rate,
            prior_marker_idx,
            state_priors,
            ref_allele_freqs,
            transition_haps,
            transition_lambda,
            context,
            smoothing_cluster_cm,
            external_nearest_obs_retain,
            ws,
        );
    }

    if ref_columns
        .iter()
        .all(|col| matches!(col, GenotypeColumn::SeqCoded(_)))
    {
        return run_hmm_seqcoded(
            state_haps,
            ref_columns,
            target_probs,
            p_recomb,
            error_rate,
            prior_marker_idx,
            state_priors,
            ref_allele_freqs,
            transition_haps,
            transition_lambda,
            context,
            smoothing_cluster_cm,
            external_nearest_obs_retain,
            ws,
        );
    }

    if ref_columns
        .iter()
        .all(|col| matches!(col, GenotypeColumn::Dictionary(_, _)))
    {
        // Dictionary-specialized kernel is used only when the current HMM call
        // sees a fully dictionary-backed window. With current streaming VCF
        // reference loading this is uncommon, because that path emits
        // Dense/Sparse columns rather than dictionary batches.
        return run_hmm_dictionary(
            state_haps,
            ref_columns,
            target_probs,
            p_recomb,
            error_rate,
            prior_marker_idx,
            state_priors,
            ref_allele_freqs,
            transition_haps,
            transition_lambda,
            context,
            smoothing_cluster_cm,
            external_nearest_obs_retain,
            ws,
        );
    }

    run_hmm_generic(
        state_haps,
        ref_columns,
        target_probs,
        p_recomb,
        error_rate,
        prior_marker_idx,
        state_priors,
        ref_allele_freqs,
        transition_haps,
        transition_lambda,
        context,
        smoothing_cluster_cm,
        external_nearest_obs_retain,
        ws,
    )
}

/// Convert dense state posteriors into sparse global priors (sorted by RefHapId).
pub fn state_posteriors_to_priors(
    state_haps: &[RefHapId],
    state_post: &[f32],
    threshold: f32,
) -> Vec<(RefHapId, f32)> {
    let mut out: Vec<(RefHapId, f32)> = state_haps
        .iter()
        .zip(state_post.iter())
        .filter(|(_, p)| p.is_finite() && **p > threshold)
        .map(|(h, &p)| (*h, p))
        .collect();
    out.sort_unstable_by_key(|(h, _)| h.as_u32());
    out
}
