//! HMM kernel for imputation using explicit haplotype states.
//!
//! This implements a Li-Stephens forward-backward pass over a selected set of
//! reference haplotypes (state set). Emissions are computed using per-haplotype
//! allele probabilities from the target, and reference alleles are read on demand.

use crate::data::HapIdx;
use crate::data::storage::{
    DenseColumn, DictionaryColumn, GenotypeColumn, SeqCodedColumn, SparseColumn,
};
use crate::error::{ReagleError, Result};
use crate::model::types::RefHapId;
use crate::model::weighted_kernel::{EmissionProbs, WeightedHmmUpdater};
use crate::pipelines::imputation::AllelePosteriors;
use crate::utils::fast_math::fast_ln;
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
        BackwardAffine, ForwardAffine, GroupSufficientStats, affine_group_mass,
        subset_transition_params, transition_only_forward_update,
    };

    #[test]
    fn test_recomb_mass_subset_sums_to_one() {
        let transition_haps = 50usize;
        let recomb_rate = 0.02f32;

        let mut fwd = vec![1.0 / transition_haps as f32; transition_haps];

        let fwd_sum: f32 = fwd.iter().sum();
        transition_only_forward_update(&mut fwd, fwd_sum, recomb_rate, transition_haps);

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
        transition_only_forward_update(&mut fwd, fwd_sum, recomb_rate, n_states);

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
    fn test_affine_group_mass_matches_direct_sum() {
        let fwd = [0.11f32, 0.27, 0.05, 0.33, 0.24];
        let bwd = [0.31f32, 0.07, 0.29, 0.13, 0.20];
        let groups = [0usize, 1, 1, 0, 2];
        let forward = ForwardAffine::from_f64(0.83, 0.017);
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
                direct += forward.apply(f) * backward.apply(b);
            }
            let stats = GroupSufficientStats {
                sum_f,
                sum_b,
                sum_fb,
                state_count: count,
            };
            let fast = affine_group_mass(forward, backward, stats);
            assert!(
                (fast - direct).abs() < 1e-6,
                "group={} fast={} direct={}",
                group,
                fast,
                direct
            );
        }
    }
}

/// Per-marker allele probability distributions for a single target haplotype.
pub struct TargetAlleleProbs {
    offsets: Vec<usize>,
    probs: Vec<f32>,
    uniform: Vec<bool>,
    observed: Vec<bool>,
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

    #[inline]
    fn probs_for_marker_normalized(&self, marker_idx: usize) -> NormalizedAlleleProbs<'_> {
        NormalizedAlleleProbs::from_trusted(self.probs_for_marker(marker_idx))
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
    subset_counts: Vec<f32>,
    smoothing_prior_counts: Vec<f32>,
    state_posterior_scratch: Vec<f32>,
    allele_prior_scratch: Vec<f32>,
    dict_pattern_alleles: Vec<u8>,
    emission_by_allele: Vec<f32>,
    nearest_obs_fwd: Vec<f32>,
    nearest_obs_bwd: Vec<f32>,
    nearest_obs_lambda: Vec<f32>,
    bwd_affine_a: Vec<f64>,
    bwd_affine_b_coeff: Vec<f64>,
    pattern_sum_f: Vec<f32>,
    pattern_sum_b: Vec<f32>,
    pattern_sum_fb: Vec<f32>,
    pattern_state_count: Vec<f32>,
    active_states: usize,
    active_markers: usize,
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

impl<'a> SeqPatternAlleles<'a> {
    #[inline]
    fn allele_for_state(&self, state_idx: usize) -> u8 {
        let pid = self.state_patterns[state_idx] as usize;
        *self.seq_alleles.get(pid).unwrap_or(&255)
    }
}

struct DictPatternAlleles<'a> {
    pattern_alleles: &'a [u8],
    state_patterns: &'a [u16],
}

impl<'a> DictPatternAlleles<'a> {
    #[inline]
    fn allele_for_state(&self, state_idx: usize) -> u8 {
        let pid = self.state_patterns[state_idx] as usize;
        *self.pattern_alleles.get(pid).unwrap_or(&255)
    }
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
struct NormalizedAlleleProbs<'a> {
    probs: &'a [f32],
}

impl<'a> NormalizedAlleleProbs<'a> {
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

const SKIP_RETAIN_THRESHOLD: f32 = 0.005;

impl ImputeWorkspace {
    pub fn new(n_states: usize, n_markers: usize) -> Self {
        Self {
            fwd: vec![0.0; n_states],
            bwd: vec![1.0; n_states],
            emissions: vec![1.0; n_states],
            fwd_checkpoints: Vec::new(),
            fwd_scales: vec![1.0; n_markers],
            weights: vec![1.0; n_states],
            state_alleles: vec![255u8; n_states],
            state_patterns: vec![0u16; n_states],
            pattern_emissions: Vec::new(),
            allele_probs: Vec::new(),
            subset_counts: Vec::new(),
            smoothing_prior_counts: Vec::new(),
            state_posterior_scratch: Vec::new(),
            allele_prior_scratch: Vec::new(),
            dict_pattern_alleles: Vec::new(),
            emission_by_allele: Vec::new(),
            nearest_obs_fwd: Vec::new(),
            nearest_obs_bwd: Vec::new(),
            nearest_obs_lambda: Vec::new(),
            bwd_affine_a: Vec::new(),
            bwd_affine_b_coeff: Vec::new(),
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
            self.state_alleles.resize(n_states, 255);
        }
        if self.state_patterns.len() < n_states {
            self.state_patterns.resize(n_states, 0);
        }
        if self.fwd_scales.len() < n_markers {
            self.fwd_scales.resize(n_markers, 1.0);
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

    pub fn ensure_bwd_affine_scratch(&mut self, block_len: usize) {
        if self.bwd_affine_a.len() < block_len {
            self.bwd_affine_a.resize(block_len, 0.0);
        }
        if self.bwd_affine_b_coeff.len() < block_len {
            self.bwd_affine_b_coeff.resize(block_len, 0.0);
        }
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
    fn ensure_subset_counts(&mut self, n_alleles: usize) {
        if self.subset_counts.len() < n_alleles {
            self.subset_counts.resize(n_alleles, 0.0);
        }
    }

    #[inline]
    fn ensure_smoothing_prior_counts(&mut self, n_alleles: usize) {
        if self.smoothing_prior_counts.len() < n_alleles {
            self.smoothing_prior_counts.resize(n_alleles, 0.0);
        }
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
                    out[i] = 0;
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
                    out[i] = 255;
                    continue;
                }
                out[i] = ((cached_bits_word >> bit_idx) & 1) as u8;
            }
        } else {
            for (i, hap) in state_haps.iter().enumerate() {
                out[i] = self.get(HapIdx::new(hap.as_u32()));
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
    target_probs: NormalizedAlleleProbs<'_>,
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

    if emission_by_allele.len() < n_alleles {
        emission_by_allele.resize(n_alleles, 1.0);
    }
    for i in 0..n_alleles {
        let p_match = target_probs.get(i).unwrap_or(0.0);
        emission_by_allele[i] = mismatch_prob + (match_prob - mismatch_prob) * p_match;
    }

    for (i, &ref_allele) in ref_alleles.slice.iter().enumerate() {
        if ref_allele == 255 {
            emissions[i] = 1.0;
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
    target_probs: NormalizedAlleleProbs<'_>,
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
        if allele == 255 {
            pattern_emissions[i] = 1.0;
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

fn compute_nearest_observed_lambda(
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
    if ws.nearest_obs_lambda.len() < n {
        ws.nearest_obs_lambda.resize(n, 0.0);
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

    // Larger cluster distance means slower information decay, so lambda should
    // shrink (less smoothing pressure) as cluster grows.
    let cluster_scale = (BASE_CLUSTER_CM / smoothing_cluster_cm.max(1e-6)).max(0.0);
    const MIN_SAME_POS_LAMBDA: f32 = 0.05;
    for m in 0..n {
        let left = ws.nearest_obs_fwd[m];
        let right = ws.nearest_obs_bwd[m];
        let mut raw_lambda = if left.is_finite() && right.is_finite() {
            // In an untyped interval bracketed by typed anchors, uncertainty is
            // governed by the full bracket span rather than the nearest side.
            left + right
        } else {
            left.min(right)
        };
        if raw_lambda == 0.0 && !target_probs.is_observed_marker(m) {
            raw_lambda = MIN_SAME_POS_LAMBDA;
        }
        ws.nearest_obs_lambda[m] = raw_lambda * cluster_scale;
        // When no typed marker is reachable in either direction, lambda stays
        // Infinity. smooth_allele_posteriors_subset handles this correctly:
        // exp(-Inf) = 0 -> maximum smoothing, which is
        // the right behavior for markers with zero LD anchor.
    }
}

#[inline]
fn smooth_allele_posteriors_subset(
    allele_probs: &mut [f32],
    subset_prior_probs: NormalizedAlleleProbs<'_>,
    nearest_obs_lambda: f32,
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
    // Bayesian shrinkage based on genetic-distance information decay.
    // Estimate effective support from the smoothing prior, not the current
    // posterior. Using the posterior here underweights smoothing precisely in
    // collapse cases where the posterior has already degenerated.
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
    // WARNING: Do NOT multiply nearest_obs_lambda by a gain factor here
    // (e.g. `* 8.0`). Amplifying lambda causes exp(-lambda*8) to decay to
    // near-zero even for short gaps, replacing posteriors with population
    // frequencies and destroying local LD signal. Tested in PR #746:
    // Hellinger +0.0037 (worst of all PRs), HET accuracy -0.0025.
    let retain = (-nearest_obs_lambda.max(0.0)).exp().clamp(MIN_RETAIN, 1.0);
    // Entropy-aware confidence gating: when posterior entropy is much lower
    // than the local-prior entropy, the state subset is likely overconfident.
    // Increase smoothing in that regime to reduce sparse-subset collapse.
    let mut post_entropy = 0.0f32;
    let mut prior_entropy = 0.0f32;
    for (&post, &prior) in allele_probs.iter().zip(subset_prior_probs.iter()) {
        let p = post.clamp(0.0, 1.0);
        let q = prior.clamp(0.0, 1.0);
        if p > 0.0 {
            post_entropy -= p * fast_ln(p);
        }
        if q > 0.0 {
            prior_entropy -= q * fast_ln(q);
        }
    }
    let entropy_gap = (prior_entropy - post_entropy).max(0.0);
    let max_entropy = fast_ln(allele_probs.len().max(2) as f32).max(1e-6);
    let confidence_boost = (entropy_gap / max_entropy).clamp(0.0, 1.0);

    let base_mass = (effective_alleles * (1.0 - retain) / retain).max(0.0);
    let prior_mass = base_mass * (1.0 + 1.75 * confidence_boost);
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
    nearest_obs_lambda: f32,
    untyped_uniform_marker: bool,
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

    // Heuristic: If we are using a subset of the panel (active_states < panel_haps),
    // we should mix in the global prior to account for the possibility that the
    // true donor is outside the selected subset.
    //
    // PBWT selection is highly effective, so we assume the "missing" mass is much
    // less than the random-selection baseline of (1 - active/total).
    // We use a power law to penalize sparse subsets while keeping dense subsets
    // mostly untouched.
    let missing_mass = if panel_haps > 0 && active_states < panel_haps {
        let raw_ratio = ((panel_haps - active_states) as f32 / panel_haps as f32).clamp(0.0, 1.0);
        raw_ratio.powi(3)
    } else {
        0.0
    };
    let floor_mix = min_prior_mix.clamp(0.0, 0.9);
    let retain = (-nearest_obs_lambda.max(0.0)).exp().clamp(0.0, 1.0);
    // Only inject panel-frequency information when distance from typed anchors
    // is sufficiently large. Near observed markers, let local LD dominate.
    let active_ratio = if panel_haps > 0 {
        (active_states as f32 / panel_haps as f32).clamp(0.0, 1.0)
    } else {
        1.0
    };
    let sparsity_boost = (1.0 - active_ratio).powi(2);
    let adaptive_panel_mix =
        (missing_mass * (1.0 - retain) * (1.0 + 1.5 * sparsity_boost)).clamp(0.0, 0.85);

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
        NormalizedAlleleProbs::from_trusted(smoothing_prior_counts)
    } else {
        if !*warned_af_fallback {
            eprintln!(
                "[warn] AF fallback in impute_hmm smoothing (no state prior): window={} sample={} hap={} marker={}",
                context.window_idx, context.sample_idx, context.hap_idx, marker_idx
            );
            *warned_af_fallback = true;
        }
        normalized_allele_prior(
            allele_prior_scratch,
            NormalizedAlleleProbs::from_trusted(probs),
        )
    };
    smooth_allele_posteriors_subset(allele_probs, prior_probs, nearest_obs_lambda, true);
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
        // Keep a small non-zero panel floor only in deep untyped regions.
        let scaled_floor = floor_mix.clamp(0.0, 0.6);
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
        // Jensen-Shannon disagreement between local subset posterior and
        // panel prior controls how strongly to apply panel blending.
        let mut m_entropy = 0.0f32;
        let mut p_entropy = 0.0f32;
        let mut q_entropy = 0.0f32;
        for (i, &p_raw) in allele_probs.iter().enumerate() {
            let p = p_raw.clamp(0.0, 1.0);
            let q = panel_probs.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0);
            let m = 0.5 * (p + q);
            if p > 0.0 {
                p_entropy -= p * fast_ln(p);
            }
            if q > 0.0 {
                q_entropy -= q * fast_ln(q);
            }
            if m > 0.0 {
                m_entropy -= m * fast_ln(m);
            }
        }
        let js_div = (m_entropy - 0.5 * (p_entropy + q_entropy)).max(0.0);
        let max_js = 0.69314718; // ln(2)
        let disagreement = (js_div / max_js).clamp(0.0, 1.0);

        // Symmetric blend toward panel frequencies. This avoids one-sided ALT
        // inflation and improves calibration for high-missingness windows.
        let w = (adaptive_panel_mix * (1.0 + 1.25 * disagreement)).clamp(0.0, 0.9);
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
    nearest_obs_lambda: &[f32],
    uniform_mask: &MarkerMask<bool>,
    use_prior_smoothing: bool,
) -> MarkerMask<bool> {
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
                let lambda = nearest_obs_lambda
                    .get(m)
                    .copied()
                    .unwrap_or(f32::INFINITY)
                    .max(0.0);
                (-lambda).exp() < SKIP_RETAIN_THRESHOLD
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
    target_probs: NormalizedAlleleProbs<'_>,
) -> NormalizedAlleleProbs<'a> {
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
        return NormalizedAlleleProbs { probs: prior };
    }
    let inv = 1.0 / sum;
    for p in prior.iter_mut() {
        *p *= inv;
    }
    NormalizedAlleleProbs { probs: prior }
}

#[inline]
fn subset_transition_params(
    recomb_rate: f32,
    active_states: usize,
    n_ref_haps: usize,
) -> (f32, f32) {
    if active_states == 0 {
        return (0.0, 0.0);
    }
    let r = recomb_rate.clamp(0.0, 1.0);
    let k = active_states as f32;
    let n = n_ref_haps.max(1) as f32;
    // Panel-aware subset conditioning:
    //   full-panel switch-to-specific-haplotype mass is r / N
    //   active subset retains k/N of recombination mass and we renormalize
    //   onto the tracked subset support.
    let switch_full = r / n;
    let z = ((1.0 - r) + k * switch_full).max(1e-30);
    let stay_gap = (1.0 - r) / z;
    let shift = switch_full / z;
    (stay_gap, shift)
}

#[inline]
fn transition_only_forward_update(
    fwd: &mut [f32],
    fwd_sum: f32,
    recomb_rate: f32,
    transition_haps: usize,
) -> f32 {
    if fwd.is_empty() {
        return 0.0;
    }
    if recomb_rate <= 0.0 {
        return fwd_sum;
    }
    let denom = fwd_sum.max(1e-30);
    let (stay_gap, shift) = subset_transition_params(recomb_rate, fwd.len(), transition_haps);
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
) -> f32 {
    if bwd.is_empty() || recomb_rate <= 0.0 {
        return bwd_sum;
    }
    let (stay_gap, shift_base) = subset_transition_params(recomb_rate, bwd.len(), transition_haps);
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
) {
    if active_states == 0 || start >= end {
        return;
    }
    let mut a = 1.0f64;
    let mut b = 0.0f64;
    let mut touched = false;
    for m in start..end {
        let recomb_rate = marker_recomb_rate(p_recomb, m);
        if recomb_rate <= 0.0 {
            continue;
        }
        touched = true;
        let (stay, shift) = subset_transition_params(recomb_rate, active_states, transition_haps);
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
) -> f32 {
    if active_states == 0 || start >= end {
        return bwd_sum;
    }
    let mut a = 1.0f64;
    let mut b_coeff = 0.0f64;
    let mut touched = false;
    for m in (start..end).rev() {
        let recomb_rate = marker_recomb_rate(p_recomb, m);
        if recomb_rate <= 0.0 {
            continue;
        }
        touched = true;
        let (stay, shift) = subset_transition_params(recomb_rate, active_states, transition_haps);
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
) {
    if block_start + 1 >= block_end {
        return;
    }
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
            let (stay, shift) =
                subset_transition_params(recomb_rate, active_states, transition_haps);
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

#[derive(Clone, Copy, Debug)]
struct ForwardAffine {
    a: f32,
    b: f32,
}

impl ForwardAffine {
    #[inline]
    fn from_f64(a: f64, b: f64) -> Self {
        Self {
            a: a as f32,
            b: b as f32,
        }
    }

    #[inline]
    fn apply(self, x: f32) -> f32 {
        self.a.mul_add(x, self.b)
    }
}

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

    #[inline]
    fn apply(self, x: f32) -> f32 {
        self.a.mul_add(x, self.add)
    }
}

#[derive(Clone, Copy, Debug)]
struct GroupSufficientStats {
    sum_f: f32,
    sum_b: f32,
    sum_fb: f32,
    state_count: f32,
}

impl GroupSufficientStats {
    #[inline]
    fn from_arrays(
        pid: usize,
        sum_f: &[f32],
        sum_b: &[f32],
        sum_fb: &[f32],
        state_count: &[f32],
    ) -> Self {
        Self {
            sum_f: sum_f[pid],
            sum_b: sum_b[pid],
            sum_fb: sum_fb[pid],
            state_count: state_count[pid],
        }
    }
}

#[inline]
fn affine_group_mass(
    forward: ForwardAffine,
    backward: BackwardAffine,
    stats: GroupSufficientStats,
) -> f32 {
    (forward.a * backward.a) * stats.sum_fb
        + (forward.a * backward.add) * stats.sum_f
        + (forward.b * backward.a) * stats.sum_b
        + (forward.b * backward.add) * stats.state_count
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
    current_error: f32,
    active_states: usize,
    transition_haps: usize,
) -> f32 {
    let probs = target_probs.probs_for_marker_normalized(m);
    let uniform = target_probs.is_uniform_marker(m);
    let recomb_rate = marker_recomb_rate(p_recomb, m);

    let mut next_sum = if uniform {
        transition_only_forward_update(
            &mut ws.fwd[..active_states],
            1.0,
            recomb_rate,
            transition_haps,
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
            current_error,
            &mut ws.emission_by_allele,
            &mut ws.emissions[..active_states],
        );
        if recomb_rate > 0.0 {
            // Reagle always uses uniform (1.0) weights for imputation states.
            // Using the specialized uniform kernel saves memory bandwidth and computation.
            WeightedHmmUpdater::fwd_update_uniform(
                &mut ws.fwd,
                1.0,
                recomb_rate,
                transition_haps,
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
    current_error: f32,
    active_states: usize,
    transition_haps: usize,
    last_hap_ptr: &mut *const u16,
) -> f32 {
    let probs = target_probs.probs_for_marker_normalized(m);
    let uniform = target_probs.is_uniform_marker(m);
    let recomb_rate = marker_recomb_rate(p_recomb, m);

    let col = seqcoded_col(&ref_columns[m]);
    let seq_patterns = refresh_seq_patterns(col, last_hap_ptr, state_haps, &mut ws.state_patterns);

    let mut next_sum = if uniform {
        transition_only_forward_update(
            &mut ws.fwd[..active_states],
            1.0,
            recomb_rate,
            transition_haps,
        )
    } else {
        fill_pattern_emissions(
            seq_patterns.seq_alleles,
            probs,
            current_error,
            &mut ws.emission_by_allele,
            &mut ws.pattern_emissions,
        );
        for i in 0..active_states {
            let pid = seq_patterns.state_patterns[i] as usize;
            ws.emissions[i] = ws.pattern_emissions.get(pid).copied().unwrap_or(1.0);
        }
        if recomb_rate > 0.0 {
            WeightedHmmUpdater::fwd_update_uniform(
                &mut ws.fwd,
                1.0,
                recomb_rate,
                transition_haps,
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
    current_error: f32,
    active_states: usize,
    transition_haps: usize,
    last_dict_ptr: &mut *const DictionaryColumn,
) -> f32 {
    let probs = target_probs.probs_for_marker_normalized(m);
    let uniform = target_probs.is_uniform_marker(m);
    let recomb_rate = marker_recomb_rate(p_recomb, m);

    let mut next_sum = if uniform {
        transition_only_forward_update(
            &mut ws.fwd[..active_states],
            1.0,
            recomb_rate,
            transition_haps,
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
            current_error,
            &mut ws.emission_by_allele,
            &mut ws.pattern_emissions,
        );
        for i in 0..active_states {
            let pid = dict_patterns.state_patterns[i] as usize;
            ws.emissions[i] = ws.pattern_emissions.get(pid).copied().unwrap_or(1.0);
        }
        if recomb_rate > 0.0 {
            WeightedHmmUpdater::fwd_update_uniform(
                &mut ws.fwd,
                1.0,
                recomb_rate,
                transition_haps,
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
fn run_impute_hmm_impl(
    state_haps: &[RefHapId],
    ref_columns: &[GenotypeColumn],
    target_probs: &TargetAlleleProbs,
    p_recomb: &[f32],
    error_rate: f32,
    prior_marker_idx: Option<usize>,
    state_priors: Option<&[f32]>,
    ref_allele_freqs: &RefAlleleFreqs,
    context: ImputeHmmContext,
    smoothing_cluster_cm: f32,
    ws: &mut ImputeWorkspace,
) -> Result<(Vec<AllelePosteriors>, Option<Vec<f32>>)> {
    validate_target_probs_nonempty(target_probs, context, "dense/sparse")?;
    validate_reference_marker_count(ref_columns.len(), target_probs, context, "dense/sparse")?;
    let n_states = state_haps.len();
    let n_markers = target_probs.n_markers();
    ws.resize(n_states, n_markers);
    let active_states = ws.active_states();
    let active_markers = ws.active_markers();
    if active_states > 0 {
        ws.weights[..active_states].fill(1.0);
    }
    let panel_haps = ref_allele_freqs.n_ref_haps().max(1);
    let transition_haps = panel_haps;
    // Compute distance-based shrinkage only when untyped markers exist.
    let use_prior_smoothing = target_probs.has_untyped_markers();
    if use_prior_smoothing {
        compute_nearest_observed_lambda(ws, target_probs, p_recomb, smoothing_cluster_cm);
    } else {
        ws.nearest_obs_lambda.clear();
    }
    let uniform_mask = build_uniform_mask(target_probs, active_markers);
    let skip_untyped_mask = build_skip_untyped_mask(
        target_probs,
        &ws.nearest_obs_lambda,
        &uniform_mask,
        use_prior_smoothing,
    );
    let panel_priors = target_probs.panel_priors();
    let checkpoint_grid = build_checkpoint_markers(&uniform_mask, prior_marker_idx, active_markers);
    ws.ensure_typed_checkpoints(active_states, checkpoint_grid.len());

    let mut final_posteriors: Vec<AllelePosteriors> = Vec::new();
    let mut final_prior_state_post: Option<Vec<f32>> = None;
    let mut warned_af_fallback = false;
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
                );
            }
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
            for cp_idx in checkpoint_grid.rev_indices() {
                let block = checkpoint_grid.block_view(cp_idx, active_markers);
                let block_start = block.start_usize();
                let block_end = block.end_usize();
                let block_len = block.len();
                ws.load_checkpoint(cp_idx, active_states);
                if let Some(interior) = UniformInteriorRange::from_block_checked(
                    block,
                    &uniform_mask,
                    context,
                    "dense/sparse",
                )? {
                    ws.ensure_bwd_affine_scratch(block_len);
                    {
                        let a_bwd = &mut ws.bwd_affine_a[..block_len];
                        let b_bwd = &mut ws.bwd_affine_b_coeff[..block_len];
                        fill_bwd_affine_coeffs(
                            a_bwd,
                            b_bwd,
                            p_recomb,
                            block_start,
                            block_end,
                            active_states,
                            transition_haps,
                        );
                    }
                    let mut a_fwd = 1.0f64;
                    let mut b_fwd = 0.0f64;
                    let bwd_sum_right = bwd_sum;
                    for m_ix in interior.iter() {
                        let m = m_ix.as_usize();
                        let recomb_rate = marker_recomb_rate(p_recomb, m);
                        if recomb_rate > 0.0 {
                            let (stay, shift) = subset_transition_params(
                                recomb_rate,
                                active_states,
                                transition_haps,
                            );
                            let stay = stay as f64;
                            let shift = shift as f64;
                            a_fwd *= stay;
                            b_fwd = stay.mul_add(b_fwd, shift);
                        }
                        if is_final
                            && prior_marker_idx != Some(m)
                            && skip_untyped_mask[MarkerIx::new(m)]
                        {
                            write_panel_freq_posterior(&mut posteriors[m], panel_priors, m);
                            continue;
                        }
                        let probs = target_probs.probs_for_marker_normalized(m);
                        let n_alleles = probs.len();
                        if prior_marker_idx == Some(m) {
                            ws.ensure_state_posterior_scratch(active_states);
                        }
                        if is_final && n_alleles > 0 {
                            ws.ensure_subset_counts(n_alleles);
                            ws.ensure_smoothing_prior_counts(n_alleles);
                        }
                        let a_bwd_m = ws.bwd_affine_a[m - block_start] as f32;
                        let b_bwd_m = ws.bwd_affine_b_coeff[m - block_start] as f32;
                        if prior_marker_idx == Some(m) {
                            let gamma = &mut ws.state_posterior_scratch[..active_states];
                            let mut sum = 0.0f32;
                            for i in 0..active_states {
                                let fwd_i = (a_fwd as f32).mul_add(ws.fwd[i], b_fwd as f32);
                                let bwd_i = a_bwd_m.mul_add(ws.bwd[i], b_bwd_m * bwd_sum_right);
                                let g = (fwd_i * bwd_i).max(0.0);
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
                                ws.ensure_pattern_sums(n_alleles);
                                let subset_counts = &mut ws.subset_counts[..n_alleles];
                                let smoothing_prior_counts =
                                    &mut ws.smoothing_prior_counts[..n_alleles];
                                subset_counts.fill(0.0);
                                smoothing_prior_counts.fill(0.0);
                                let mut subset_total = 0.0f32;
                                let mut smoothing_prior_total = 0.0f32;
                                let mut total = 0.0f32;
                                let missing_mass: f32;
                                let ref_alleles = refresh_ref_alleles(
                                    &ref_columns[m],
                                    state_haps,
                                    &mut ws.state_alleles[..active_states],
                                    &mut ws.dict_pattern_alleles,
                                );
                                let sum_f = &mut ws.pattern_sum_f[..n_alleles];
                                let sum_b = &mut ws.pattern_sum_b[..n_alleles];
                                let sum_fb = &mut ws.pattern_sum_fb[..n_alleles];
                                let state_count = &mut ws.pattern_state_count[..n_alleles];
                                sum_f.fill(0.0);
                                sum_b.fill(0.0);
                                sum_fb.fill(0.0);
                                state_count.fill(0.0);
                                let mut missing_stats = GroupSufficientStats {
                                    sum_f: 0.0,
                                    sum_b: 0.0,
                                    sum_fb: 0.0,
                                    state_count: 0.0,
                                };
                                for i in 0..active_states {
                                    let f = ws.fwd[i];
                                    let b = ws.bwd[i];
                                    let ref_allele = ref_alleles.get(i);
                                    if ref_allele == 255 {
                                        missing_stats.sum_f += f;
                                        missing_stats.sum_b += b;
                                        missing_stats.sum_fb += f * b;
                                        missing_stats.state_count += 1.0;
                                        continue;
                                    }
                                    let idx = ref_allele as usize;
                                    if idx < n_alleles {
                                        sum_f[idx] += f;
                                        sum_b[idx] += b;
                                        sum_fb[idx] += f * b;
                                        state_count[idx] += 1.0;
                                    }
                                }
                                let forward_affine = ForwardAffine::from_f64(a_fwd, b_fwd);
                                let backward_affine =
                                    BackwardAffine::new(a_bwd_m, b_bwd_m, bwd_sum_right);
                                for idx in 0..n_alleles {
                                    let stats = GroupSufficientStats::from_arrays(
                                        idx,
                                        sum_f,
                                        sum_b,
                                        sum_fb,
                                        state_count,
                                    );
                                    let state_prob =
                                        affine_group_mass(forward_affine, backward_affine, stats);
                                    let state_prob = if state_prob.is_finite() {
                                        state_prob.max(0.0)
                                    } else {
                                        0.0
                                    };
                                    ws.allele_probs[idx] += state_prob;
                                    subset_counts[idx] += state_prob;
                                    subset_total += state_prob;
                                    smoothing_prior_counts[idx] += state_count[idx];
                                    smoothing_prior_total += state_count[idx];
                                    total += state_prob;
                                }
                                missing_mass = affine_group_mass(
                                    forward_affine,
                                    backward_affine,
                                    missing_stats,
                                );
                                let missing_mass = if missing_mass.is_finite() {
                                    missing_mass.max(0.0)
                                } else {
                                    0.0
                                };
                                total += missing_mass;
                                if total > 0.0 {
                                    if missing_mass > 0.0 {
                                        let prior = if subset_total > 0.0 {
                                            let inv = 1.0 / subset_total;
                                            for v in subset_counts.iter_mut() {
                                                *v *= inv;
                                            }
                                            NormalizedAlleleProbs::from_trusted(subset_counts)
                                        } else {
                                            if !warned_af_fallback {
                                                eprintln!(
                                                    "[warn] AF fallback in impute_hmm (no state info): window={} sample={} hap={} marker={}",
                                                    context.window_idx,
                                                    context.sample_idx,
                                                    context.hap_idx,
                                                    m
                                                );
                                                warned_af_fallback = true;
                                            }
                                            normalized_allele_prior(
                                                &mut ws.allele_prior_scratch,
                                                probs,
                                            )
                                        };
                                        for (i, p) in ws.allele_probs.iter_mut().enumerate() {
                                            *p += missing_mass * prior.as_slice()[i];
                                        }
                                    }
                                    for p in ws.allele_probs.iter_mut() {
                                        *p /= total;
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
                                            ws.nearest_obs_lambda
                                                .get(m)
                                                .copied()
                                                .unwrap_or(f32::INFINITY),
                                            target_probs.is_untyped_uniform_marker(m),
                                            active_states,
                                            panel_haps,
                                            target_probs.min_untyped_prior_mix(),
                                            &mut warned_af_fallback,
                                            context,
                                        );
                                    }
                                } else {
                                    if !warned_af_fallback {
                                        eprintln!(
                                            "[warn] posterior-mass fallback in impute_hmm (dense/sparse): window={} sample={} hap={} marker={} active_states={}",
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
                                if ws.allele_probs.len() == 2 {
                                    posteriors[m] = AllelePosteriors::Biallelic(ws.allele_probs[1]);
                                } else {
                                    let mut out = Vec::with_capacity(ws.allele_probs.len());
                                    out.extend_from_slice(&ws.allele_probs);
                                    posteriors[m] =
                                        AllelePosteriors::Multiallelic(std::sync::Arc::from(out));
                                }
                            } else {
                                return Err(ReagleError::vcf(format!(
                                    "No allele space available in imputation HMM (dense/sparse): window={} sample={} hap={} marker={} active_states={}",
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
                    );
                }

                let m_rev = block_start;
                if is_final
                    && prior_marker_idx != Some(m_rev)
                    && skip_untyped_mask[MarkerIx::new(m_rev)]
                {
                    write_panel_freq_posterior(&mut posteriors[m_rev], panel_priors, m_rev);
                } else {
                    let probs = target_probs.probs_for_marker_normalized(m_rev);
                    let n_alleles = probs.len();
                    if prior_marker_idx == Some(m_rev) {
                        ws.ensure_state_posterior_scratch(active_states);
                    }
                    if is_final && n_alleles > 0 {
                        ws.ensure_subset_counts(n_alleles);
                        ws.ensure_smoothing_prior_counts(n_alleles);
                    }
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
                    let ref_alleles = refresh_ref_alleles(
                        &ref_columns[m_rev],
                        state_haps,
                        &mut ws.state_alleles[..active_states],
                        &mut ws.dict_pattern_alleles,
                    );

                    if is_final {
                        ws.allele_probs.clear();
                        if n_alleles > 0 {
                            ws.allele_probs.resize(n_alleles, 0.0f32);
                            let subset_counts = &mut ws.subset_counts[..n_alleles];
                            let smoothing_prior_counts =
                                &mut ws.smoothing_prior_counts[..n_alleles];
                            subset_counts.fill(0.0);
                            smoothing_prior_counts.fill(0.0);
                            let mut subset_total = 0.0f32;
                            let mut smoothing_prior_total = 0.0f32;
                            let mut total = 0.0f32;
                            let mut missing_mass = 0.0f32;
                            for i in 0..active_states {
                                let state_prob = fwd_slice[i] * ws.bwd[i];
                                total += state_prob;
                                let ref_allele = ref_alleles.get(i);
                                if ref_allele == 255 {
                                    missing_mass += state_prob;
                                    continue;
                                }
                                let idx = ref_allele as usize;
                                if idx < ws.allele_probs.len() {
                                    ws.allele_probs[idx] += state_prob;
                                    subset_counts[idx] += state_prob;
                                    subset_total += state_prob;
                                    smoothing_prior_counts[idx] += 1.0;
                                    smoothing_prior_total += 1.0;
                                }
                            }
                            if total > 0.0 {
                                if missing_mass > 0.0 {
                                    let prior = if subset_total > 0.0 {
                                        let inv = 1.0 / subset_total;
                                        for v in subset_counts.iter_mut() {
                                            *v *= inv;
                                        }
                                        NormalizedAlleleProbs::from_trusted(subset_counts)
                                    } else {
                                        if !warned_af_fallback {
                                            eprintln!(
                                                "[warn] AF fallback in impute_hmm (no state info): window={} sample={} hap={} marker={}",
                                                context.window_idx,
                                                context.sample_idx,
                                                context.hap_idx,
                                                m_rev
                                            );
                                            warned_af_fallback = true;
                                        }
                                        normalized_allele_prior(&mut ws.allele_prior_scratch, probs)
                                    };
                                    for (i, p) in ws.allele_probs.iter_mut().enumerate() {
                                        *p += missing_mass * prior.as_slice()[i];
                                    }
                                }
                                for p in ws.allele_probs.iter_mut() {
                                    *p /= total;
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
                                        ws.nearest_obs_lambda
                                            .get(m_rev)
                                            .copied()
                                            .unwrap_or(f32::INFINITY),
                                        target_probs.is_untyped_uniform_marker(m_rev),
                                        active_states,
                                        panel_haps,
                                        target_probs.min_untyped_prior_mix(),
                                        &mut warned_af_fallback,
                                        context,
                                    );
                                }
                            } else {
                                if !warned_af_fallback {
                                    eprintln!(
                                        "[warn] posterior-mass fallback in impute_hmm (dense/sparse): window={} sample={} hap={} marker={} active_states={}",
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
                            if ws.allele_probs.len() == 2 {
                                posteriors[m_rev] = AllelePosteriors::Biallelic(ws.allele_probs[1]);
                            } else {
                                let mut out = Vec::with_capacity(ws.allele_probs.len());
                                out.extend_from_slice(&ws.allele_probs);
                                posteriors[m_rev] =
                                    AllelePosteriors::Multiallelic(std::sync::Arc::from(out));
                            }
                        } else {
                            return Err(ReagleError::vcf(format!(
                                "No allele space available in imputation HMM (dense/sparse): window={} sample={} hap={} marker={} active_states={}",
                                context.window_idx,
                                context.sample_idx,
                                context.hap_idx,
                                m_rev,
                                active_states
                            )));
                        }
                    }
                }

                let probs = target_probs.probs_for_marker_normalized(m_rev);
                let recomb_rate = marker_recomb_rate(p_recomb, m_rev);
                if uniform_mask[MarkerIx::new(m_rev)] {
                    bwd_sum = transition_only_backward_update(
                        &mut ws.bwd[..active_states],
                        recomb_rate,
                        transition_haps,
                        bwd_sum,
                    );
                } else {
                    let ref_alleles = refresh_ref_alleles(
                        &ref_columns[m_rev],
                        state_haps,
                        &mut ws.state_alleles[..active_states],
                        &mut ws.dict_pattern_alleles,
                    );
                    fill_emissions(
                        &ref_alleles,
                        probs,
                        current_error,
                        &mut ws.emission_by_allele,
                        &mut ws.emissions[..active_states],
                    );
                    let mut emit_beta_sum = 0.0f32;
                    for i in 0..active_states {
                        emit_beta_sum += ws.emissions[i] * ws.bwd[i];
                    }
                    let c_t = ws.fwd_scales.get(m_rev).copied().unwrap_or(1.0).max(1e-30);
                    let (stay_gap, shift_base) =
                        subset_transition_params(recomb_rate, active_states, transition_haps);
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
            // WARNING: Do NOT add a post-HMM posterior interpolation pass here
            // (e.g. LD-bridge that overwrites untyped marker posteriors with
            // distance-weighted averages of typed anchor posteriors). The HMM
            // forward-backward posteriors already encode full LD structure
            // through the state space; replacing them with naive anchor
            // interpolation destroys this signal. Deep-untyped regions get
            // bridge_strength approaching 0.95, overwriting nearly the entire
            // posterior. Panel-prior mixing up to 35% further dilutes toward
            // population frequencies. Tested in PR #741: R² -0.008,
            // SEN -0.0013 — second worst accuracy regression observed.
            final_posteriors = posteriors;
        }
    }
    Ok((final_posteriors, final_prior_state_post))
}

fn run_impute_hmm_seqcoded(
    state_haps: &[RefHapId],
    ref_columns: &[GenotypeColumn],
    target_probs: &TargetAlleleProbs,
    p_recomb: &[f32],
    error_rate: f32,
    prior_marker_idx: Option<usize>,
    state_priors: Option<&[f32]>,
    ref_allele_freqs: &RefAlleleFreqs,
    context: ImputeHmmContext,
    smoothing_cluster_cm: f32,
    ws: &mut ImputeWorkspace,
) -> Result<(Vec<AllelePosteriors>, Option<Vec<f32>>)> {
    validate_target_probs_nonempty(target_probs, context, "seqcoded")?;
    validate_reference_marker_count(ref_columns.len(), target_probs, context, "seqcoded")?;
    let n_states = state_haps.len();
    let n_markers = target_probs.n_markers();
    ws.resize(n_states, n_markers);
    let active_states = ws.active_states();
    let active_markers = ws.active_markers();
    if active_states > 0 {
        ws.weights[..active_states].fill(1.0);
    }
    let panel_haps = ref_allele_freqs.n_ref_haps().max(1);
    let transition_haps = panel_haps;
    // Compute distance-based shrinkage only when untyped markers exist.
    let use_prior_smoothing = target_probs.has_untyped_markers();
    if use_prior_smoothing {
        compute_nearest_observed_lambda(ws, target_probs, p_recomb, smoothing_cluster_cm);
    } else {
        ws.nearest_obs_lambda.clear();
    }
    let uniform_mask = build_uniform_mask(target_probs, active_markers);
    let skip_untyped_mask = build_skip_untyped_mask(
        target_probs,
        &ws.nearest_obs_lambda,
        &uniform_mask,
        use_prior_smoothing,
    );
    let panel_priors = target_probs.panel_priors();
    let checkpoint_grid = build_checkpoint_markers(&uniform_mask, prior_marker_idx, active_markers);
    ws.ensure_typed_checkpoints(active_states, checkpoint_grid.len());

    let mut final_posteriors: Vec<AllelePosteriors> = Vec::new();
    let mut final_prior_state_post: Option<Vec<f32>> = None;
    let mut warned_af_fallback = false;
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

        let mut last_hap_ptr: *const u16 = std::ptr::null();
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
                );
            }
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
                &mut last_hap_ptr,
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
            let mut last_hap_ptr: *const u16 = std::ptr::null();
            for cp_idx in checkpoint_grid.rev_indices() {
                let mut pattern_sum_key_ptr: *const u16 = std::ptr::null();
                let block = checkpoint_grid.block_view(cp_idx, active_markers);
                let block_start = block.start_usize();
                let block_end = block.end_usize();
                let block_len = block.len();
                ws.load_checkpoint(cp_idx, active_states);
                if let Some(interior) = UniformInteriorRange::from_block_checked(
                    block,
                    &uniform_mask,
                    context,
                    "seqcoded",
                )? {
                    ws.ensure_bwd_affine_scratch(block_len);
                    fill_bwd_affine_coeffs(
                        &mut ws.bwd_affine_a[..block_len],
                        &mut ws.bwd_affine_b_coeff[..block_len],
                        p_recomb,
                        block_start,
                        block_end,
                        active_states,
                        transition_haps,
                    );
                    let mut a_fwd = 1.0f64;
                    let mut b_fwd = 0.0f64;
                    let bwd_sum_right = bwd_sum;
                    for m_ix in interior.iter() {
                        let m = m_ix.as_usize();
                        let recomb_rate = marker_recomb_rate(p_recomb, m);
                        if recomb_rate > 0.0 {
                            let (stay, shift) = subset_transition_params(
                                recomb_rate,
                                active_states,
                                transition_haps,
                            );
                            let stay = stay as f64;
                            let shift = shift as f64;
                            a_fwd *= stay;
                            b_fwd = stay.mul_add(b_fwd, shift);
                        }
                        if is_final
                            && prior_marker_idx != Some(m)
                            && skip_untyped_mask[MarkerIx::new(m)]
                        {
                            write_panel_freq_posterior(&mut posteriors[m], panel_priors, m);
                            continue;
                        }
                        let probs = target_probs.probs_for_marker_normalized(m);
                        let n_alleles = probs.len();
                        if prior_marker_idx == Some(m) {
                            ws.ensure_state_posterior_scratch(active_states);
                        }
                        if is_final && n_alleles > 0 {
                            ws.ensure_subset_counts(n_alleles);
                            ws.ensure_smoothing_prior_counts(n_alleles);
                        }
                        let col = seqcoded_col(&ref_columns[m]);
                        let hap_ptr = col.hap_to_seq().as_ptr();
                        refresh_seq_patterns(
                            col,
                            &mut last_hap_ptr,
                            state_haps,
                            &mut ws.state_patterns,
                        );
                        let seq_alleles = col.seq_alleles();
                        if hap_ptr != pattern_sum_key_ptr {
                            let n_patterns = seq_alleles.len();
                            ws.ensure_pattern_sums(n_patterns);
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
                            pattern_sum_key_ptr = hap_ptr;
                        }
                        let a_bwd_m = ws.bwd_affine_a[m - block_start] as f32;
                        let b_bwd_m = ws.bwd_affine_b_coeff[m - block_start] as f32;
                        let forward_affine = ForwardAffine::from_f64(a_fwd, b_fwd);
                        let backward_affine = BackwardAffine::new(a_bwd_m, b_bwd_m, bwd_sum_right);
                        if prior_marker_idx == Some(m) {
                            let gamma = &mut ws.state_posterior_scratch[..active_states];
                            let mut sum = 0.0f32;
                            for i in 0..active_states {
                                let fwd_i = forward_affine.apply(ws.fwd[i]);
                                let bwd_i = backward_affine.apply(ws.bwd[i]);
                                let g = (fwd_i * bwd_i).max(0.0);
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
                                let subset_counts = &mut ws.subset_counts[..n_alleles];
                                let smoothing_prior_counts =
                                    &mut ws.smoothing_prior_counts[..n_alleles];
                                subset_counts.fill(0.0);
                                smoothing_prior_counts.fill(0.0);
                                let mut subset_total = 0.0f32;
                                let mut smoothing_prior_total = 0.0f32;
                                let mut total = 0.0f32;
                                let mut missing_mass = 0.0f32;
                                let n_patterns = seq_alleles.len();
                                for pid in 0..n_patterns {
                                    let stats = GroupSufficientStats::from_arrays(
                                        pid,
                                        &ws.pattern_sum_f,
                                        &ws.pattern_sum_b,
                                        &ws.pattern_sum_fb,
                                        &ws.pattern_state_count,
                                    );
                                    let state_prob =
                                        affine_group_mass(forward_affine, backward_affine, stats);
                                    total += state_prob;
                                    let ref_allele = *seq_alleles.get(pid).unwrap_or(&255);
                                    if ref_allele == 255 {
                                        missing_mass += state_prob;
                                        continue;
                                    }
                                    let idx = ref_allele as usize;
                                    if idx < ws.allele_probs.len() {
                                        ws.allele_probs[idx] += state_prob;
                                        subset_counts[idx] += state_prob;
                                        subset_total += state_prob;
                                        smoothing_prior_counts[idx] += ws.pattern_state_count[pid];
                                        smoothing_prior_total += ws.pattern_state_count[pid];
                                    }
                                }
                                if total > 0.0 {
                                    if missing_mass > 0.0 {
                                        let prior = if subset_total > 0.0 {
                                            let inv = 1.0 / subset_total;
                                            for v in subset_counts.iter_mut() {
                                                *v *= inv;
                                            }
                                            NormalizedAlleleProbs::from_trusted(subset_counts)
                                        } else {
                                            if !warned_af_fallback {
                                                eprintln!(
                                                    "[warn] AF fallback in impute_hmm (no state info): window={} sample={} hap={} marker={}",
                                                    context.window_idx,
                                                    context.sample_idx,
                                                    context.hap_idx,
                                                    m
                                                );
                                                warned_af_fallback = true;
                                            }
                                            normalized_allele_prior(
                                                &mut ws.allele_prior_scratch,
                                                probs,
                                            )
                                        };
                                        for (i, p) in ws.allele_probs.iter_mut().enumerate() {
                                            *p += missing_mass * prior.as_slice()[i];
                                        }
                                    }
                                    for p in ws.allele_probs.iter_mut() {
                                        *p /= total;
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
                                            ws.nearest_obs_lambda
                                                .get(m)
                                                .copied()
                                                .unwrap_or(f32::INFINITY),
                                            target_probs.is_untyped_uniform_marker(m),
                                            active_states,
                                            panel_haps,
                                            target_probs.min_untyped_prior_mix(),
                                            &mut warned_af_fallback,
                                            context,
                                        );
                                    }
                                } else {
                                    if !warned_af_fallback {
                                        eprintln!(
                                            "[warn] posterior-mass fallback in impute_hmm (seqcoded): window={} sample={} hap={} marker={} active_states={}",
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
                                if ws.allele_probs.len() == 2 {
                                    posteriors[m] = AllelePosteriors::Biallelic(ws.allele_probs[1]);
                                } else {
                                    let mut out = Vec::with_capacity(ws.allele_probs.len());
                                    out.extend_from_slice(&ws.allele_probs);
                                    posteriors[m] =
                                        AllelePosteriors::Multiallelic(std::sync::Arc::from(out));
                                }
                            } else {
                                return Err(ReagleError::vcf(format!(
                                    "No allele space available in imputation HMM (seqcoded): window={} sample={} hap={} marker={} active_states={}",
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
                    );
                }

                let m_rev = block_start;
                let probs = target_probs.probs_for_marker_normalized(m_rev);
                let recomb_rate = marker_recomb_rate(p_recomb, m_rev);
                let n_alleles = probs.len();
                if prior_marker_idx == Some(m_rev) {
                    ws.ensure_state_posterior_scratch(active_states);
                }
                if is_final && n_alleles > 0 {
                    ws.ensure_subset_counts(n_alleles);
                    ws.ensure_smoothing_prior_counts(n_alleles);
                }
                let col = seqcoded_col(&ref_columns[m_rev]);
                let seq_patterns = refresh_seq_patterns(
                    col,
                    &mut last_hap_ptr,
                    state_haps,
                    &mut ws.state_patterns,
                );

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
                            let subset_counts = &mut ws.subset_counts[..n_alleles];
                            let smoothing_prior_counts =
                                &mut ws.smoothing_prior_counts[..n_alleles];
                            subset_counts.fill(0.0);
                            smoothing_prior_counts.fill(0.0);
                            let mut subset_total = 0.0f32;
                            let mut smoothing_prior_total = 0.0f32;
                            let mut total = 0.0f32;
                            let mut missing_mass = 0.0f32;
                            for i in 0..active_states {
                                let state_prob = fwd_slice[i] * ws.bwd[i];
                                total += state_prob;
                                let ref_allele = seq_patterns.allele_for_state(i);
                                if ref_allele == 255 {
                                    missing_mass += state_prob;
                                    continue;
                                }
                                let idx = ref_allele as usize;
                                if idx < ws.allele_probs.len() {
                                    ws.allele_probs[idx] += state_prob;
                                    subset_counts[idx] += state_prob;
                                    subset_total += state_prob;
                                    smoothing_prior_counts[idx] += 1.0;
                                    smoothing_prior_total += 1.0;
                                }
                            }
                            if total > 0.0 {
                                if missing_mass > 0.0 {
                                    let prior = if subset_total > 0.0 {
                                        let inv = 1.0 / subset_total;
                                        for v in subset_counts.iter_mut() {
                                            *v *= inv;
                                        }
                                        NormalizedAlleleProbs::from_trusted(subset_counts)
                                    } else {
                                        if !warned_af_fallback {
                                            eprintln!(
                                                "[warn] AF fallback in impute_hmm (no state info): window={} sample={} hap={} marker={}",
                                                context.window_idx,
                                                context.sample_idx,
                                                context.hap_idx,
                                                m_rev
                                            );
                                            warned_af_fallback = true;
                                        }
                                        normalized_allele_prior(&mut ws.allele_prior_scratch, probs)
                                    };
                                    for (i, p) in ws.allele_probs.iter_mut().enumerate() {
                                        *p += missing_mass * prior.as_slice()[i];
                                    }
                                }
                                for p in ws.allele_probs.iter_mut() {
                                    *p /= total;
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
                                        ws.nearest_obs_lambda
                                            .get(m_rev)
                                            .copied()
                                            .unwrap_or(f32::INFINITY),
                                        target_probs.is_untyped_uniform_marker(m_rev),
                                        active_states,
                                        panel_haps,
                                        target_probs.min_untyped_prior_mix(),
                                        &mut warned_af_fallback,
                                        context,
                                    );
                                }
                            } else {
                                if !warned_af_fallback {
                                    eprintln!(
                                        "[warn] posterior-mass fallback in impute_hmm (seqcoded): window={} sample={} hap={} marker={} active_states={}",
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
                            if ws.allele_probs.len() == 2 {
                                posteriors[m_rev] = AllelePosteriors::Biallelic(ws.allele_probs[1]);
                            } else {
                                let mut out = Vec::with_capacity(ws.allele_probs.len());
                                out.extend_from_slice(&ws.allele_probs);
                                posteriors[m_rev] =
                                    AllelePosteriors::Multiallelic(std::sync::Arc::from(out));
                            }
                        } else {
                            return Err(ReagleError::vcf(format!(
                                "No allele space available in imputation HMM (seqcoded): window={} sample={} hap={} marker={} active_states={}",
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
                    );
                } else {
                    fill_pattern_emissions(
                        seq_patterns.seq_alleles,
                        probs,
                        current_error,
                        &mut ws.emission_by_allele,
                        &mut ws.pattern_emissions,
                    );
                    let mut emit_beta_sum = 0.0f32;
                    for i in 0..active_states {
                        let pid = seq_patterns.state_patterns[i] as usize;
                        let emit = ws.pattern_emissions.get(pid).copied().unwrap_or(1.0);
                        ws.emissions[i] = emit;
                        emit_beta_sum += emit * ws.bwd[i];
                    }
                    let c_t = ws.fwd_scales.get(m_rev).copied().unwrap_or(1.0).max(1e-30);
                    let (stay_gap, shift_base) =
                        subset_transition_params(recomb_rate, active_states, transition_haps);
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
    Ok((final_posteriors, final_prior_state_post))
}

fn run_impute_hmm_dict(
    state_haps: &[RefHapId],
    ref_columns: &[GenotypeColumn],
    target_probs: &TargetAlleleProbs,
    p_recomb: &[f32],
    error_rate: f32,
    prior_marker_idx: Option<usize>,
    state_priors: Option<&[f32]>,
    ref_allele_freqs: &RefAlleleFreqs,
    context: ImputeHmmContext,
    smoothing_cluster_cm: f32,
    ws: &mut ImputeWorkspace,
) -> Result<(Vec<AllelePosteriors>, Option<Vec<f32>>)> {
    validate_target_probs_nonempty(target_probs, context, "dictionary")?;
    validate_reference_marker_count(ref_columns.len(), target_probs, context, "dictionary")?;
    let n_states = state_haps.len();
    let n_markers = target_probs.n_markers();
    ws.resize(n_states, n_markers);
    let active_states = ws.active_states();
    let active_markers = ws.active_markers();
    if active_states > 0 {
        ws.weights[..active_states].fill(1.0);
    }
    let panel_haps = ref_allele_freqs.n_ref_haps().max(1);
    let transition_haps = panel_haps;
    // Compute distance-based shrinkage only when untyped markers exist.
    let use_prior_smoothing = target_probs.has_untyped_markers();
    if use_prior_smoothing {
        compute_nearest_observed_lambda(ws, target_probs, p_recomb, smoothing_cluster_cm);
    } else {
        ws.nearest_obs_lambda.clear();
    }
    let uniform_mask = build_uniform_mask(target_probs, active_markers);
    let skip_untyped_mask = build_skip_untyped_mask(
        target_probs,
        &ws.nearest_obs_lambda,
        &uniform_mask,
        use_prior_smoothing,
    );
    let panel_priors = target_probs.panel_priors();
    let checkpoint_grid = build_checkpoint_markers(&uniform_mask, prior_marker_idx, active_markers);
    ws.ensure_typed_checkpoints(active_states, checkpoint_grid.len());

    let mut final_posteriors: Vec<AllelePosteriors> = Vec::new();
    let mut final_prior_state_post: Option<Vec<f32>> = None;
    let mut warned_af_fallback = false;
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

        let mut last_dict_ptr: *const DictionaryColumn = std::ptr::null();
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
                );
            }
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
                &mut last_dict_ptr,
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
            let mut last_dict_ptr: *const DictionaryColumn = std::ptr::null();
            for cp_idx in checkpoint_grid.rev_indices() {
                let mut pattern_sum_key_ptr: *const DictionaryColumn = std::ptr::null();
                let block = checkpoint_grid.block_view(cp_idx, active_markers);
                let block_start = block.start_usize();
                let block_end = block.end_usize();
                let block_len = block.len();
                ws.load_checkpoint(cp_idx, active_states);
                if let Some(interior) = UniformInteriorRange::from_block_checked(
                    block,
                    &uniform_mask,
                    context,
                    "dictionary",
                )? {
                    ws.ensure_bwd_affine_scratch(block_len);
                    fill_bwd_affine_coeffs(
                        &mut ws.bwd_affine_a[..block_len],
                        &mut ws.bwd_affine_b_coeff[..block_len],
                        p_recomb,
                        block_start,
                        block_end,
                        active_states,
                        transition_haps,
                    );
                    let mut a_fwd = 1.0f64;
                    let mut b_fwd = 0.0f64;
                    let bwd_sum_right = bwd_sum;
                    for m_ix in interior.iter() {
                        let m = m_ix.as_usize();
                        let recomb_rate = marker_recomb_rate(p_recomb, m);
                        if recomb_rate > 0.0 {
                            let (stay, shift) = subset_transition_params(
                                recomb_rate,
                                active_states,
                                transition_haps,
                            );
                            let stay = stay as f64;
                            let shift = shift as f64;
                            a_fwd *= stay;
                            b_fwd = stay.mul_add(b_fwd, shift);
                        }
                        if is_final
                            && prior_marker_idx != Some(m)
                            && skip_untyped_mask[MarkerIx::new(m)]
                        {
                            write_panel_freq_posterior(&mut posteriors[m], panel_priors, m);
                            continue;
                        }
                        let probs = target_probs.probs_for_marker_normalized(m);
                        let n_alleles = probs.len();
                        if prior_marker_idx == Some(m) {
                            ws.ensure_state_posterior_scratch(active_states);
                        }
                        if is_final && n_alleles > 0 {
                            ws.ensure_subset_counts(n_alleles);
                            ws.ensure_smoothing_prior_counts(n_alleles);
                        }
                        let col = dict_col_ref(&ref_columns[m]);
                        let dict_ptr = col.col as *const DictionaryColumn;
                        refresh_dict_patterns(
                            &col,
                            &mut last_dict_ptr,
                            state_haps,
                            &mut ws.state_patterns,
                            &mut ws.dict_pattern_alleles,
                        );
                        if dict_ptr != pattern_sum_key_ptr {
                            let n_patterns = col.col.n_patterns();
                            ws.ensure_pattern_sums(n_patterns);
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
                            pattern_sum_key_ptr = dict_ptr;
                        }
                        let a_bwd_m = ws.bwd_affine_a[m - block_start] as f32;
                        let b_bwd_m = ws.bwd_affine_b_coeff[m - block_start] as f32;
                        let forward_affine = ForwardAffine::from_f64(a_fwd, b_fwd);
                        let backward_affine = BackwardAffine::new(a_bwd_m, b_bwd_m, bwd_sum_right);
                        if prior_marker_idx == Some(m) {
                            let gamma = &mut ws.state_posterior_scratch[..active_states];
                            let mut sum = 0.0f32;
                            for i in 0..active_states {
                                let fwd_i = forward_affine.apply(ws.fwd[i]);
                                let bwd_i = backward_affine.apply(ws.bwd[i]);
                                let g = (fwd_i * bwd_i).max(0.0);
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
                                let subset_counts = &mut ws.subset_counts[..n_alleles];
                                let smoothing_prior_counts =
                                    &mut ws.smoothing_prior_counts[..n_alleles];
                                subset_counts.fill(0.0);
                                smoothing_prior_counts.fill(0.0);
                                let mut subset_total = 0.0f32;
                                let mut smoothing_prior_total = 0.0f32;
                                let mut total = 0.0f32;
                                let mut missing_mass = 0.0f32;
                                let n_patterns = col.col.n_patterns();
                                let pattern_alleles = &ws.dict_pattern_alleles[..n_patterns];
                                for pid in 0..n_patterns {
                                    let stats = GroupSufficientStats::from_arrays(
                                        pid,
                                        &ws.pattern_sum_f,
                                        &ws.pattern_sum_b,
                                        &ws.pattern_sum_fb,
                                        &ws.pattern_state_count,
                                    );
                                    let state_prob =
                                        affine_group_mass(forward_affine, backward_affine, stats);
                                    total += state_prob;
                                    let ref_allele = *pattern_alleles.get(pid).unwrap_or(&255);
                                    if ref_allele == 255 {
                                        missing_mass += state_prob;
                                        continue;
                                    }
                                    let idx = ref_allele as usize;
                                    if idx < ws.allele_probs.len() {
                                        ws.allele_probs[idx] += state_prob;
                                        subset_counts[idx] += state_prob;
                                        subset_total += state_prob;
                                        smoothing_prior_counts[idx] += ws.pattern_state_count[pid];
                                        smoothing_prior_total += ws.pattern_state_count[pid];
                                    }
                                }
                                if total > 0.0 {
                                    if missing_mass > 0.0 {
                                        let prior = if subset_total > 0.0 {
                                            let inv = 1.0 / subset_total;
                                            for v in subset_counts.iter_mut() {
                                                *v *= inv;
                                            }
                                            NormalizedAlleleProbs::from_trusted(subset_counts)
                                        } else {
                                            if !warned_af_fallback {
                                                eprintln!(
                                                    "[warn] AF fallback in impute_hmm (no state info): window={} sample={} hap={} marker={}",
                                                    context.window_idx,
                                                    context.sample_idx,
                                                    context.hap_idx,
                                                    m
                                                );
                                                warned_af_fallback = true;
                                            }
                                            normalized_allele_prior(
                                                &mut ws.allele_prior_scratch,
                                                probs,
                                            )
                                        };
                                        for (i, p) in ws.allele_probs.iter_mut().enumerate() {
                                            *p += missing_mass * prior.as_slice()[i];
                                        }
                                    }
                                    for p in ws.allele_probs.iter_mut() {
                                        *p /= total;
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
                                            ws.nearest_obs_lambda
                                                .get(m)
                                                .copied()
                                                .unwrap_or(f32::INFINITY),
                                            target_probs.is_untyped_uniform_marker(m),
                                            active_states,
                                            panel_haps,
                                            target_probs.min_untyped_prior_mix(),
                                            &mut warned_af_fallback,
                                            context,
                                        );
                                    }
                                } else {
                                    if !warned_af_fallback {
                                        eprintln!(
                                            "[warn] posterior-mass fallback in impute_hmm (dictionary): window={} sample={} hap={} marker={} active_states={}",
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
                                if ws.allele_probs.len() == 2 {
                                    posteriors[m] = AllelePosteriors::Biallelic(ws.allele_probs[1]);
                                } else {
                                    let mut out = Vec::with_capacity(ws.allele_probs.len());
                                    out.extend_from_slice(&ws.allele_probs);
                                    posteriors[m] =
                                        AllelePosteriors::Multiallelic(std::sync::Arc::from(out));
                                }
                            } else {
                                return Err(ReagleError::vcf(format!(
                                    "No allele space available in imputation HMM (dictionary): window={} sample={} hap={} marker={} active_states={}",
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
                    );
                }

                let m_rev = block_start;
                let probs = target_probs.probs_for_marker_normalized(m_rev);
                let recomb_rate = marker_recomb_rate(p_recomb, m_rev);
                let n_alleles = probs.len();
                if prior_marker_idx == Some(m_rev) {
                    ws.ensure_state_posterior_scratch(active_states);
                }
                if is_final && n_alleles > 0 {
                    ws.ensure_subset_counts(n_alleles);
                    ws.ensure_smoothing_prior_counts(n_alleles);
                }
                let col = dict_col_ref(&ref_columns[m_rev]);
                let dict_patterns = refresh_dict_patterns(
                    &col,
                    &mut last_dict_ptr,
                    state_haps,
                    &mut ws.state_patterns,
                    &mut ws.dict_pattern_alleles,
                );
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
                            let subset_counts = &mut ws.subset_counts[..n_alleles];
                            let smoothing_prior_counts =
                                &mut ws.smoothing_prior_counts[..n_alleles];
                            subset_counts.fill(0.0);
                            smoothing_prior_counts.fill(0.0);
                            let mut subset_total = 0.0f32;
                            let mut smoothing_prior_total = 0.0f32;
                            let mut total = 0.0f32;
                            let mut missing_mass = 0.0f32;
                            for i in 0..active_states {
                                let state_prob = fwd_slice[i] * ws.bwd[i];
                                total += state_prob;
                                let ref_allele = dict_patterns.allele_for_state(i);
                                if ref_allele == 255 {
                                    missing_mass += state_prob;
                                    continue;
                                }
                                let idx = ref_allele as usize;
                                if idx < ws.allele_probs.len() {
                                    ws.allele_probs[idx] += state_prob;
                                    subset_counts[idx] += state_prob;
                                    subset_total += state_prob;
                                    smoothing_prior_counts[idx] += 1.0;
                                    smoothing_prior_total += 1.0;
                                }
                            }
                            if total > 0.0 {
                                if missing_mass > 0.0 {
                                    let prior = if subset_total > 0.0 {
                                        let inv = 1.0 / subset_total;
                                        for v in subset_counts.iter_mut() {
                                            *v *= inv;
                                        }
                                        NormalizedAlleleProbs::from_trusted(subset_counts)
                                    } else {
                                        if !warned_af_fallback {
                                            eprintln!(
                                                "[warn] AF fallback in impute_hmm (no state info): window={} sample={} hap={} marker={}",
                                                context.window_idx,
                                                context.sample_idx,
                                                context.hap_idx,
                                                m_rev
                                            );
                                            warned_af_fallback = true;
                                        }
                                        normalized_allele_prior(&mut ws.allele_prior_scratch, probs)
                                    };
                                    for (i, p) in ws.allele_probs.iter_mut().enumerate() {
                                        *p += missing_mass * prior.as_slice()[i];
                                    }
                                }
                                for p in ws.allele_probs.iter_mut() {
                                    *p /= total;
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
                                        ws.nearest_obs_lambda
                                            .get(m_rev)
                                            .copied()
                                            .unwrap_or(f32::INFINITY),
                                        target_probs.is_untyped_uniform_marker(m_rev),
                                        active_states,
                                        panel_haps,
                                        target_probs.min_untyped_prior_mix(),
                                        &mut warned_af_fallback,
                                        context,
                                    );
                                }
                            } else {
                                if !warned_af_fallback {
                                    eprintln!(
                                        "[warn] posterior-mass fallback in impute_hmm (dictionary): window={} sample={} hap={} marker={} active_states={}",
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
                            if ws.allele_probs.len() == 2 {
                                posteriors[m_rev] = AllelePosteriors::Biallelic(ws.allele_probs[1]);
                            } else {
                                let mut out = Vec::with_capacity(ws.allele_probs.len());
                                out.extend_from_slice(&ws.allele_probs);
                                posteriors[m_rev] =
                                    AllelePosteriors::Multiallelic(std::sync::Arc::from(out));
                            }
                        } else {
                            return Err(ReagleError::vcf(format!(
                                "No allele space available in imputation HMM (dictionary): window={} sample={} hap={} marker={} active_states={}",
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
                    );
                } else {
                    fill_pattern_emissions(
                        dict_patterns.pattern_alleles,
                        probs,
                        current_error,
                        &mut ws.emission_by_allele,
                        &mut ws.pattern_emissions,
                    );
                    let mut emit_beta_sum = 0.0f32;
                    for i in 0..active_states {
                        let pid = dict_patterns.state_patterns[i] as usize;
                        let emit = ws.pattern_emissions.get(pid).copied().unwrap_or(1.0);
                        ws.emissions[i] = emit;
                        emit_beta_sum += emit * ws.bwd[i];
                    }
                    let c_t = ws.fwd_scales.get(m_rev).copied().unwrap_or(1.0).max(1e-30);
                    let (stay_gap, shift_base) =
                        subset_transition_params(recomb_rate, active_states, transition_haps);
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
    Ok((final_posteriors, final_prior_state_post))
}

/// Run forward-backward HMM and emit allele posteriors.
///
/// Returns (posteriors, optional state posterior at prior marker).
pub fn run_impute_hmm(
    state_haps: &[RefHapId],
    ref_columns: &[GenotypeColumn],
    target_probs: &TargetAlleleProbs,
    p_recomb: &[f32],
    error_rate: f32,
    prior_marker_idx: Option<usize>,
    state_priors: Option<&[f32]>,
    ref_allele_freqs: &RefAlleleFreqs,
    context: ImputeHmmContext,
    smoothing_cluster_cm: f32,
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
        return run_impute_hmm_impl(
            state_haps,
            ref_columns,
            target_probs,
            p_recomb,
            error_rate,
            prior_marker_idx,
            state_priors,
            ref_allele_freqs,
            context,
            smoothing_cluster_cm,
            ws,
        );
    }

    if ref_columns
        .iter()
        .all(|col| matches!(col, GenotypeColumn::Sparse(_)))
    {
        return run_impute_hmm_impl(
            state_haps,
            ref_columns,
            target_probs,
            p_recomb,
            error_rate,
            prior_marker_idx,
            state_priors,
            ref_allele_freqs,
            context,
            smoothing_cluster_cm,
            ws,
        );
    }

    if ref_columns
        .iter()
        .all(|col| matches!(col, GenotypeColumn::SeqCoded(_)))
    {
        return run_impute_hmm_seqcoded(
            state_haps,
            ref_columns,
            target_probs,
            p_recomb,
            error_rate,
            prior_marker_idx,
            state_priors,
            ref_allele_freqs,
            context,
            smoothing_cluster_cm,
            ws,
        );
    }

    if ref_columns
        .iter()
        .all(|col| matches!(col, GenotypeColumn::Dictionary(_, _)))
    {
        return run_impute_hmm_dict(
            state_haps,
            ref_columns,
            target_probs,
            p_recomb,
            error_rate,
            prior_marker_idx,
            state_priors,
            ref_allele_freqs,
            context,
            smoothing_cluster_cm,
            ws,
        );
    }

    run_impute_hmm_impl(
        state_haps,
        ref_columns,
        target_probs,
        p_recomb,
        error_rate,
        prior_marker_idx,
        state_priors,
        ref_allele_freqs,
        context,
        smoothing_cluster_cm,
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
