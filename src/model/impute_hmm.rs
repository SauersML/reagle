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
use crate::model::weighted_kernel::WeightedHmmUpdater;
use crate::pipelines::imputation::AllelePosteriors;

#[derive(Clone, Copy, Debug, Default)]
pub struct EmStats {
    pub expected_mismatches: f64,
    pub informative_sites: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct ImputeHmmContext {
    pub window_idx: usize,
    pub sample_idx: usize,
    pub hap_idx: usize,
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

#[cfg(test)]
mod tests {
    use super::transition_only_forward_update;

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
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};

#[inline(always)]
fn prefetch_read(ptr: *const u8) {
    std::hint::black_box(ptr);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe {
        _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
    }
}

/// Per-marker allele probability distributions for a single target haplotype.
pub struct TargetAlleleProbs {
    offsets: Vec<usize>,
    probs: Vec<f32>,
    uniform: Vec<bool>,
}

impl TargetAlleleProbs {
    pub fn new(offsets: Vec<usize>, probs: Vec<f32>) -> Self {
        let mut uniform = Vec::new();
        if offsets.len() >= 2 {
            uniform.reserve(offsets.len() - 1);
            for m in 0..(offsets.len() - 1) {
                let start = offsets[m];
                let end = offsets[m + 1];
                let slice = probs.get(start..end).unwrap_or(&[]);
                uniform.push(is_uniform_probs(slice));
            }
        }
        Self {
            offsets,
            probs,
            uniform,
        }
    }

    #[inline]
    pub fn probs_for_marker(&self, marker_idx: usize) -> &[f32] {
        let start = self.offsets[marker_idx];
        let end = self.offsets[marker_idx + 1];
        &self.probs[start..end]
    }

    #[inline]
    pub fn n_markers(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    #[inline]
    pub fn is_uniform_marker(&self, marker_idx: usize) -> bool {
        self.uniform.get(marker_idx).copied().unwrap_or(true)
    }
}

/// Workspace for per-haplotype imputation HMM.
pub struct ImputeWorkspace {
    pub fwd: Vec<f32>,
    pub bwd: Vec<f32>,
    pub emissions: Vec<f32>,
    pub fwd_history: Vec<f32>,
    pub fwd_checkpoints: Vec<f32>,
    pub checkpoint_stride: usize,
    pub fwd_scales: Vec<f32>,
    pub weights: Vec<f32>,
    pub state_alleles: Vec<u8>,
    pub state_patterns: Vec<u16>,
    pub pattern_emissions: Vec<f32>,
    pub allele_probs: Vec<f32>,
    dict_pattern_alleles: Vec<u8>,
    emission_by_allele: Vec<f32>,
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

impl ImputeWorkspace {
    pub fn new(n_states: usize, n_markers: usize) -> Self {
        Self {
            fwd: vec![0.0; n_states],
            bwd: vec![1.0; n_states],
            emissions: vec![1.0; n_states],
            fwd_history: Vec::new(),
            fwd_checkpoints: Vec::new(),
            checkpoint_stride: 1,
            fwd_scales: vec![1.0; n_markers],
            weights: vec![1.0; n_states],
            state_alleles: vec![255u8; n_states],
            state_patterns: vec![0u16; n_states],
            pattern_emissions: Vec::new(),
            allele_probs: Vec::new(),
            dict_pattern_alleles: Vec::new(),
            emission_by_allele: Vec::new(),
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

    pub fn configure_checkpoints(&mut self, n_states: usize, n_markers: usize) -> usize {
        // Keep checkpoint memory bounded while staying exact.
        const MAX_CHECKPOINT_BYTES: usize = 64 * 1024 * 1024;
        let bytes_per_cp = n_states.max(1).saturating_mul(std::mem::size_of::<f32>());
        let max_checkpoints = (MAX_CHECKPOINT_BYTES / bytes_per_cp).max(1);
        let stride = (n_markers.max(1) + max_checkpoints - 1) / max_checkpoints;
        let stride = stride.max(1);
        let n_checkpoints = (n_markers + stride - 1) / stride + 1;
        let want = n_checkpoints.saturating_mul(n_states.max(1));
        if self.fwd_checkpoints.len() < want {
            self.fwd_checkpoints.resize(want, 0.0);
        }
        self.checkpoint_stride = stride;
        stride
    }

    pub fn ensure_block_history(&mut self, n_states: usize, block_len: usize) {
        let want = n_states.saturating_mul(block_len.max(1));
        if self.fwd_history.len() < want {
            self.fwd_history.resize(want, 0.0);
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
        std::hint::black_box(dict_pattern_alleles.len());
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
        std::hint::black_box(dict_pattern_alleles.len());
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
        std::hint::black_box(dict_pattern_alleles.len());
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
    target_probs: &[f32],
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
        let p_match = target_probs[i];
        emission_by_allele[i] = mismatch_prob + (match_prob - mismatch_prob) * p_match;
    }

    let prefetch_stride = 64usize;
    for (i, &ref_allele) in ref_alleles.slice.iter().enumerate() {
        if i + prefetch_stride < ref_alleles.slice.len() {
            unsafe {
                prefetch_read(ref_alleles.slice.as_ptr().add(i + prefetch_stride));
            }
        }
        if ref_allele == 255 {
            emissions[i] = 1.0;
            continue;
        }
        let idx = ref_allele as usize;
        if idx < n_alleles {
            emissions[i] = emission_by_allele[idx];
        } else {
            emissions[i] = mismatch_prob;
        }
    }
}

#[inline]
fn fill_pattern_emissions(
    pattern_alleles: &[u8],
    target_probs: &[f32],
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
        let p_match = target_probs[i];
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
                pattern_emissions[i] = mismatch_prob;
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
fn smooth_allele_posteriors_subset(
    allele_probs: &mut [f32],
    subset_prior_probs: &[f32],
    total_mass: f32,
    state_prob_sq_sum: f32,
) {
    if allele_probs.is_empty() || total_mass <= 0.0 || state_prob_sq_sum <= 0.0 {
        return;
    }
    if subset_prior_probs.len() != allele_probs.len() {
        return;
    }

    // Weak Bayesian shrinkage toward the active-state subset prior.
    // As effective support grows, this prior becomes negligible.
    let effective_states = (total_mass * total_mass / state_prob_sq_sum).max(1.0);
    let prior_mass = (0.25f32 / effective_states).clamp(0.0, 0.5);
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
fn subset_transition_params(recomb_rate: f32, active_states: usize, n_ref_haps: usize) -> (f32, f32, f32) {
    if active_states == 0 {
        return (0.0, 0.0, 1.0);
    }
    let r = recomb_rate.clamp(0.0, 1.0);
    let k = active_states as f32;
    let n = n_ref_haps.max(1) as f32;
    let switch_full = r / n;
    let z = ((1.0 - r) + k * switch_full).max(1e-30);
    let stay_gap = (1.0 - r) / z;
    let shift = switch_full / z;
    let stay = ((1.0 - r) + switch_full) / z;
    (stay_gap, shift, stay)
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
    let params = subset_transition_params(recomb_rate, fwd.len(), transition_haps);
    let scale = params.0 / denom;
    let shift = params.1;
    let mut new_sum = 0.0f32;
    for v in fwd.iter_mut() {
        let t = scale.mul_add(*v, shift);
        *v = t;
        new_sum += t;
    }
    new_sum
}

#[inline]
fn transition_only_backward_update(bwd: &mut [f32], recomb_rate: f32, transition_haps: usize) {
    if bwd.is_empty() || recomb_rate <= 0.0 {
        return;
    }
    let mut sum = 0.0f32;
    for v in bwd.iter() {
        sum += *v;
    }
    let params = subset_transition_params(recomb_rate, bwd.len(), transition_haps);
    let shift = params.1 * sum;
    let scale = params.0;
    for v in bwd.iter_mut() {
        *v = scale.mul_add(*v, shift);
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
    use_prior_weighting: bool,
    fwd_sum: f32,
    state_haps: &[RefHapId],
    ref_columns: &[C],
    target_probs: &TargetAlleleProbs,
    p_recomb: &[f32],
    current_error: f32,
    active_states: usize,
    transition_haps: usize,
) -> f32 {
    let probs = target_probs.probs_for_marker(m);
    let recomb_rate = p_recomb.get(m).copied().unwrap_or(0.0);
    let uniform = target_probs.is_uniform_marker(m);

    let mut next_sum = if uniform {
        transition_only_forward_update(
            &mut ws.fwd[..active_states],
            fwd_sum,
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

        if use_prior_weighting {
            if recomb_rate > 0.0 {
                WeightedHmmUpdater::fwd_update_weighted(
                    &mut ws.fwd,
                    fwd_sum,
                    recomb_rate,
                    transition_haps,
                    &ws.weights,
                    &ws.emissions,
                    active_states,
                )
            } else {
                for i in 0..active_states {
                    ws.fwd[i] *= ws.emissions[i];
                }
                ws.fwd[..active_states].iter().sum::<f32>().max(1e-30)
            }
        } else {
            WeightedHmmUpdater::fwd_update_weighted(
                &mut ws.fwd,
                fwd_sum,
                recomb_rate,
                transition_haps,
                &ws.weights,
                &ws.emissions,
                active_states,
            )
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
    use_prior_weighting: bool,
    fwd_sum: f32,
    state_haps: &[RefHapId],
    ref_columns: &[&SeqCodedColumn],
    target_probs: &TargetAlleleProbs,
    p_recomb: &[f32],
    current_error: f32,
    active_states: usize,
    transition_haps: usize,
    last_hap_ptr: &mut *const u16,
) -> f32 {
    let probs = target_probs.probs_for_marker(m);
    let recomb_rate = p_recomb.get(m).copied().unwrap_or(0.0);
    let uniform = target_probs.is_uniform_marker(m);

    let col = ref_columns[m];
    let seq_patterns = refresh_seq_patterns(col, last_hap_ptr, state_haps, &mut ws.state_patterns);

    let mut next_sum = if uniform {
        transition_only_forward_update(
            &mut ws.fwd[..active_states],
            fwd_sum,
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

        if use_prior_weighting {
            if recomb_rate > 0.0 {
                WeightedHmmUpdater::fwd_update_weighted(
                    &mut ws.fwd,
                    fwd_sum,
                    recomb_rate,
                    transition_haps,
                    &ws.weights,
                    &ws.emissions,
                    active_states,
                )
            } else {
                for i in 0..active_states {
                    ws.fwd[i] *= ws.emissions[i];
                }
                ws.fwd[..active_states].iter().sum::<f32>().max(1e-30)
            }
        } else {
            WeightedHmmUpdater::fwd_update_weighted(
                &mut ws.fwd,
                fwd_sum,
                recomb_rate,
                transition_haps,
                &ws.weights,
                &ws.emissions,
                active_states,
            )
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
    use_prior_weighting: bool,
    fwd_sum: f32,
    state_haps: &[RefHapId],
    ref_columns: &[DictColRef<'_>],
    target_probs: &TargetAlleleProbs,
    p_recomb: &[f32],
    current_error: f32,
    active_states: usize,
    transition_haps: usize,
    last_dict_ptr: &mut *const DictionaryColumn,
) -> f32 {
    let probs = target_probs.probs_for_marker(m);
    let recomb_rate = p_recomb.get(m).copied().unwrap_or(0.0);
    let uniform = target_probs.is_uniform_marker(m);

    let mut next_sum = if uniform {
        transition_only_forward_update(
            &mut ws.fwd[..active_states],
            fwd_sum,
            recomb_rate,
            transition_haps,
        )
    } else {
        let col = &ref_columns[m];
        let dict_patterns = refresh_dict_patterns(
            col,
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

        if use_prior_weighting {
            if recomb_rate > 0.0 {
                WeightedHmmUpdater::fwd_update_weighted(
                    &mut ws.fwd,
                    fwd_sum,
                    recomb_rate,
                    transition_haps,
                    &ws.weights,
                    &ws.emissions,
                    active_states,
                )
            } else {
                for i in 0..active_states {
                    ws.fwd[i] *= ws.emissions[i];
                }
                ws.fwd[..active_states].iter().sum::<f32>().max(1e-30)
            }
        } else {
            WeightedHmmUpdater::fwd_update_weighted(
                &mut ws.fwd,
                fwd_sum,
                recomb_rate,
                transition_haps,
                &ws.weights,
                &ws.emissions,
                active_states,
            )
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
fn run_impute_hmm_impl<C: RefColumnLike>(
    state_haps: &[RefHapId],
    ref_columns: &[C],
    target_probs: &TargetAlleleProbs,
    p_recomb: &[f32],
    error_rate: f32,
    prior_marker_idx: Option<usize>,
    state_priors: Option<&[f32]>,
    ref_allele_freqs: &RefAlleleFreqs,
    context: ImputeHmmContext,
    ws: &mut ImputeWorkspace,
) -> Result<(Vec<AllelePosteriors>, Option<Vec<f32>>, EmStats)> {
    validate_target_probs_nonempty(target_probs, context, "dense/sparse")?;
    let n_states = state_haps.len();
    let n_markers = target_probs.n_markers();
    ws.resize(n_states, n_markers);
    let active_states = ws.active_states();
    let active_markers = ws.active_markers();
    let checkpoint_stride = ws.configure_checkpoints(active_states, active_markers);
    let transition_haps = ref_allele_freqs.n_ref_haps().max(1);
    if active_states > 0 {
        // Li-Stephens transition denominator should be full reference-panel size.
        // Subsequent normalization conditions probabilities onto active states.
        ws.weights.fill(1.0);
    }

    // --- Bayesian MAP update for p_mismatch (one EM step, minimal prior) ---
    //
    // We interpret `error_rate` as the copied-allele -> true-allele mismatch
    // probability (Li-Stephens "miscopy" / mutation). PL/GL already model
    // observation noise, so this parameter should *not* absorb genotyping error.
    //
    // We do a single MAP update with a Beta prior centered at the LS prior:
    //   e_ls = li_stephens_p_mismatch(N_ref)
    //   alpha = 1 + e_ls, beta = 1 + (1-e_ls)  (one pseudo-observation total)
    //
    // The MAP update for this conjugate model is:
    //   e_new = (sum_{m,k} gamma_{m,k} * eta_{m,k} + (alpha-1))
    //           / (M_eff + (alpha+beta-2))
    // where:
    //   gamma_{m,k} = posterior state weight at marker m and state k,
    //   eta_{m,k}   = posterior mismatch probability given state k at marker m,
    //   M_eff       = number of informative markers.
    //
    // With alpha+beta-2 = 1, this is a minimal Bayesian shrinkage toward e_ls.
    let mut final_posteriors: Vec<AllelePosteriors> = Vec::new();
    let mut final_prior_state_post: Option<Vec<f32>> = None;
    let current_error = error_rate;
    let mut mismatch_sum = 0.0f64;
    let mut mismatch_markers = 0.0f64;

    let final_pass = 0usize;
    for pass in 0..1 {
        let is_final = pass == final_pass;
        let mut fwd_sum: f32;
        if let Some(priors) = state_priors {
            let len = priors.len().min(active_states);
            ws.fwd[..len].copy_from_slice(&priors[..len]);
            if len < active_states {
                ws.fwd[len..active_states].fill(0.0);
            }
            normalize_probs(&mut ws.fwd[..active_states]);
            fwd_sum = ws.fwd[..active_states].iter().sum::<f32>().max(1e-30);
        } else {
            let uniform = 1.0 / active_states.max(1) as f32;
            ws.fwd[..active_states].fill(uniform);
            fwd_sum = 1.0;
        }

        for m in 0..active_markers {
            let use_prior_weighting = m == 0 && state_priors.is_some();
            fwd_sum = forward_update_impl(
                ws,
                m,
                use_prior_weighting,
                fwd_sum,
                state_haps,
                ref_columns,
                target_probs,
                p_recomb,
                current_error,
                active_states,
                transition_haps,
            );
            if m % checkpoint_stride == 0 {
                let cp = (m / checkpoint_stride) * active_states;
                ws.fwd_checkpoints[cp..cp + active_states]
                    .copy_from_slice(&ws.fwd[..active_states]);
            }
            if prior_marker_idx == Some(m) {
                final_prior_state_post = Some(ws.fwd[..active_states].to_vec());
            }
        }

        let mut posteriors: Vec<AllelePosteriors> = Vec::new();
        if is_final {
            posteriors.reserve(n_markers);
            posteriors.resize_with(n_markers, || AllelePosteriors::Biallelic(0.0));
        }

        ws.bwd.fill(1.0);
        if active_markers > 0 {
            let stride = checkpoint_stride.max(1);
            let mut block_start = ((active_markers - 1) / stride) * stride;
            loop {
                let block_end = (block_start + stride).min(active_markers);
                let block_len = block_end.saturating_sub(block_start);
                ws.ensure_block_history(active_states, block_len);

                let cp = (block_start / stride) * active_states;
                ws.fwd[..active_states]
                    .copy_from_slice(&ws.fwd_checkpoints[cp..cp + active_states]);
                ws.fwd_history[..active_states].copy_from_slice(&ws.fwd[..active_states]);

                if block_start + 1 < block_end {
                    let mut local_sum = 1.0f32;
                    for m in (block_start + 1)..block_end {
                        local_sum = forward_update_impl(
                            ws,
                            m,
                            false,
                            local_sum,
                            state_haps,
                            ref_columns,
                            target_probs,
                            p_recomb,
                            current_error,
                            active_states,
                            transition_haps,
                        );
                        let local_idx = (m - block_start) * active_states;
                        ws.fwd_history[local_idx..local_idx + active_states]
                            .copy_from_slice(&ws.fwd[..active_states]);
                    }
                }

                for m_rev in (block_start..block_end).rev() {
                    let probs = target_probs.probs_for_marker(m_rev);
                    let recomb_rate = p_recomb.get(m_rev).copied().unwrap_or(0.0);
                    let uniform = target_probs.is_uniform_marker(m_rev);

                    let start = (m_rev - block_start) * active_states;
                    let fwd_slice = &ws.fwd_history[start..start + active_states];
                    // Always refresh ref alleles for posterior calculation, even if emissions are uniform.
                    let ref_alleles = refresh_ref_alleles(
                        &ref_columns[m_rev],
                        state_haps,
                        &mut ws.state_alleles[..active_states],
                        &mut ws.dict_pattern_alleles,
                    );

                    if is_final {
                        // Compute posteriors using beta at time t (current ws.bwd), before updating
                        // beta for time t-1. This aligns alpha_t * beta_t for marker-level posteriors.
                        ws.allele_probs.clear();
                        let n_alleles = probs.len();
                        if n_alleles > 0 {
                            ws.allele_probs.resize(n_alleles, 0.0f32);
                            let mut subset_counts = vec![0.0f32; n_alleles];
                            let mut subset_total = 0.0f32;
                            let mut total = 0.0f32;
                            let mut sq_sum = 0.0f32;
                            for i in 0..active_states {
                                let ref_allele = ref_alleles.get(i);
                                if ref_allele == 255 {
                                    continue;
                                }
                                let state_prob = fwd_slice[i] * ws.bwd[i];
                                total += state_prob;
                                sq_sum += state_prob * state_prob;
                                let idx = ref_allele as usize;
                                if idx < ws.allele_probs.len() {
                                    ws.allele_probs[idx] += state_prob;
                                    subset_counts[idx] += 1.0;
                                    subset_total += 1.0;
                                }
                            }
                            if total > 0.0 {
                                for p in ws.allele_probs.iter_mut() {
                                    *p /= total;
                                }
                                if uniform && recomb_rate > 0.0 && subset_total > 0.0 {
                                    let inv_subset = 1.0 / subset_total;
                                    for p in subset_counts.iter_mut() {
                                        *p *= inv_subset;
                                    }
                                    smooth_allele_posteriors_subset(
                                        &mut ws.allele_probs,
                                        &subset_counts,
                                        total,
                                        sq_sum,
                                    );
                                }
                            } else {
                                return Err(ReagleError::vcf(format!(
                                    "Posterior mass collapse in imputation HMM (dense/sparse): window={} sample={} hap={} marker={} active_states={}",
                                    context.window_idx,
                                    context.sample_idx,
                                    context.hap_idx,
                                    m_rev,
                                    active_states
                                )));
                            }
                            if ws.allele_probs.len() == 2 {
                                posteriors[m_rev] = AllelePosteriors::Biallelic(ws.allele_probs[1]);
                            } else {
                                let mut out = Vec::with_capacity(ws.allele_probs.len());
                                out.extend_from_slice(&ws.allele_probs);
                                posteriors[m_rev] = AllelePosteriors::Multiallelic(out);
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

                    // Update beta for the previous marker.
                    // Scaled backward recursion:
                    //   beta_{t-1}(i) = ( (1-r) * b_t(i) * beta_t(i) + (r/N) * S_t ) / c_t
                    // where S_t = sum_j b_t(j) * beta_t(j) and c_t is the forward scale
                    // at marker t (sum of unnormalized alpha_t).
                    if uniform {
                        transition_only_backward_update(
                            &mut ws.bwd[..active_states],
                            recomb_rate,
                            transition_haps,
                        );
                    } else {
                        fill_emissions(
                            &ref_alleles,
                            probs,
                            current_error,
                            &mut ws.emission_by_allele,
                            &mut ws.emissions[..active_states],
                        );
                        if !probs.is_empty() && probs.len() > 1 {
                            let mismatch_prob = current_error / (probs.len() as f32 - 1.0);
                            let mut total_gamma = 0.0f64;
                            let mut mismatch_expect = 0.0f64;
                            for i in 0..active_states {
                                let gamma = (fwd_slice[i] * ws.bwd[i]) as f64;
                                let ref_allele = ref_alleles.get(i);
                                if ref_allele == 255 {
                                    continue;
                                }
                                total_gamma += gamma;
                                let idx = ref_allele as usize;
                                let p_match = probs.get(idx).copied().unwrap_or(0.0);
                                let emission = if idx < ws.emission_by_allele.len() {
                                    ws.emission_by_allele[idx]
                                } else {
                                    mismatch_prob
                                };
                                if emission > 0.0 {
                                    let eta = (mismatch_prob * (1.0 - p_match) / emission) as f64;
                                    mismatch_expect += gamma * eta;
                                }
                            }
                            if total_gamma > 0.0 {
                                mismatch_sum += mismatch_expect / total_gamma;
                                mismatch_markers += 1.0;
                            }
                        }
                        let mut emit_beta_sum = 0.0f32;
                        for i in 0..active_states {
                            emit_beta_sum += ws.emissions[i] * ws.bwd[i];
                        }
                        let c_t = ws.fwd_scales.get(m_rev).copied().unwrap_or(1.0).max(1e-30);
                        let params = subset_transition_params(recomb_rate, active_states, transition_haps);
                        let scale = params.0 / c_t;
                        let shift = params.1 * (emit_beta_sum / c_t);
                        for i in 0..active_states {
                            ws.bwd[i] = scale * ws.emissions[i] * ws.bwd[i] + shift;
                        }
                    }
                }
                if block_start == 0 {
                    break;
                }
                block_start = block_start.saturating_sub(stride);
            }
        }

        if is_final {
            final_posteriors = posteriors;
        }
    }

    let stats = EmStats {
        expected_mismatches: mismatch_sum,
        informative_sites: mismatch_markers,
    };

    Ok((final_posteriors, final_prior_state_post, stats))
}

fn run_impute_hmm_seqcoded(
    state_haps: &[RefHapId],
    ref_columns: &[&SeqCodedColumn],
    target_probs: &TargetAlleleProbs,
    p_recomb: &[f32],
    error_rate: f32,
    prior_marker_idx: Option<usize>,
    state_priors: Option<&[f32]>,
    ref_allele_freqs: &RefAlleleFreqs,
    context: ImputeHmmContext,
    ws: &mut ImputeWorkspace,
) -> Result<(Vec<AllelePosteriors>, Option<Vec<f32>>, EmStats)> {
    validate_target_probs_nonempty(target_probs, context, "seqcoded")?;
    let n_states = state_haps.len();
    let n_markers = target_probs.n_markers();
    ws.resize(n_states, n_markers);
    let active_states = ws.active_states();
    let active_markers = ws.active_markers();
    let checkpoint_stride = ws.configure_checkpoints(active_states, active_markers);
    let transition_haps = ref_allele_freqs.n_ref_haps().max(1);
    if active_states > 0 {
        ws.weights.fill(1.0);
    }

    let mut final_posteriors: Vec<AllelePosteriors> = Vec::new();
    let mut final_prior_state_post: Option<Vec<f32>> = None;
    let current_error = error_rate;
    let mut mismatch_sum = 0.0f64;
    let mut mismatch_markers = 0.0f64;

    let final_pass = 0usize;
    for pass in 0..1 {
        let is_final = pass == final_pass;
        let mut fwd_sum: f32;
        if let Some(priors) = state_priors {
            let len = priors.len().min(active_states);
            ws.fwd[..len].copy_from_slice(&priors[..len]);
            if len < active_states {
                ws.fwd[len..active_states].fill(0.0);
            }
            normalize_probs(&mut ws.fwd[..active_states]);
            fwd_sum = ws.fwd[..active_states].iter().sum::<f32>().max(1e-30);
        } else {
            let uniform = 1.0 / active_states.max(1) as f32;
            ws.fwd[..active_states].fill(uniform);
            fwd_sum = 1.0;
        }

        let mut last_hap_ptr: *const u16 = std::ptr::null();
        for m in 0..active_markers {
            let use_prior_weighting = m == 0 && state_priors.is_some();
            fwd_sum = forward_update_seqcoded(
                ws,
                m,
                use_prior_weighting,
                fwd_sum,
                state_haps,
                ref_columns,
                target_probs,
                p_recomb,
                current_error,
                active_states,
                transition_haps,
                &mut last_hap_ptr,
            );
            if m % checkpoint_stride == 0 {
                let cp = (m / checkpoint_stride) * active_states;
                ws.fwd_checkpoints[cp..cp + active_states]
                    .copy_from_slice(&ws.fwd[..active_states]);
            }
            if prior_marker_idx == Some(m) {
                final_prior_state_post = Some(ws.fwd[..active_states].to_vec());
            }
        }

        let mut posteriors: Vec<AllelePosteriors> = Vec::new();
        if is_final {
            posteriors.reserve(n_markers);
            posteriors.resize_with(n_markers, || AllelePosteriors::Biallelic(0.0));
        }

        ws.bwd.fill(1.0);
        if active_markers > 0 {
            let stride = checkpoint_stride.max(1);
            let mut block_start = ((active_markers - 1) / stride) * stride;
            let mut last_hap_ptr: *const u16 = std::ptr::null();
            loop {
                let block_end = (block_start + stride).min(active_markers);
                let block_len = block_end.saturating_sub(block_start);
                ws.ensure_block_history(active_states, block_len);

                let cp = (block_start / stride) * active_states;
                ws.fwd[..active_states]
                    .copy_from_slice(&ws.fwd_checkpoints[cp..cp + active_states]);
                ws.fwd_history[..active_states].copy_from_slice(&ws.fwd[..active_states]);

                if block_start + 1 < block_end {
                    let mut local_sum = 1.0f32;
                    let mut last_hap_ptr_re: *const u16 = std::ptr::null();
                    for m in (block_start + 1)..block_end {
                        local_sum = forward_update_seqcoded(
                            ws,
                            m,
                            false,
                            local_sum,
                            state_haps,
                            ref_columns,
                            target_probs,
                            p_recomb,
                            current_error,
                            active_states,
                            transition_haps,
                            &mut last_hap_ptr_re,
                        );
                        let local_idx = (m - block_start) * active_states;
                        ws.fwd_history[local_idx..local_idx + active_states]
                            .copy_from_slice(&ws.fwd[..active_states]);
                    }
                }

                for m_rev in (block_start..block_end).rev() {
                    let probs = target_probs.probs_for_marker(m_rev);
                    let recomb_rate = p_recomb.get(m_rev).copied().unwrap_or(0.0);
                    let uniform = target_probs.is_uniform_marker(m_rev);

                    let col = ref_columns[m_rev];
                    let seq_patterns = refresh_seq_patterns(
                        col,
                        &mut last_hap_ptr,
                        state_haps,
                        &mut ws.state_patterns,
                    );

                    let start = (m_rev - block_start) * active_states;
                    let fwd_slice = &ws.fwd_history[start..start + active_states];

                    if is_final {
                        ws.allele_probs.clear();
                        let n_alleles = probs.len();
                        if n_alleles > 0 {
                            ws.allele_probs.resize(n_alleles, 0.0f32);
                            let mut subset_counts = vec![0.0f32; n_alleles];
                            let mut subset_total = 0.0f32;
                            let mut total = 0.0f32;
                            let mut sq_sum = 0.0f32;
                            for i in 0..active_states {
                                let ref_allele = seq_patterns.allele_for_state(i);
                                if ref_allele == 255 {
                                    continue;
                                }
                                let state_prob = fwd_slice[i] * ws.bwd[i];
                                total += state_prob;
                                sq_sum += state_prob * state_prob;
                                let idx = ref_allele as usize;
                                if idx < ws.allele_probs.len() {
                                    ws.allele_probs[idx] += state_prob;
                                    subset_counts[idx] += 1.0;
                                    subset_total += 1.0;
                                }
                            }
                            if total > 0.0 {
                                for p in ws.allele_probs.iter_mut() {
                                    *p /= total;
                                }
                                if uniform && recomb_rate > 0.0 && subset_total > 0.0 {
                                    let inv_subset = 1.0 / subset_total;
                                    for p in subset_counts.iter_mut() {
                                        *p *= inv_subset;
                                    }
                                    smooth_allele_posteriors_subset(
                                        &mut ws.allele_probs,
                                        &subset_counts,
                                        total,
                                        sq_sum,
                                    );
                                }
                            } else {
                                return Err(ReagleError::vcf(format!(
                                    "Posterior mass collapse in imputation HMM (seqcoded): window={} sample={} hap={} marker={} active_states={}",
                                    context.window_idx,
                                    context.sample_idx,
                                    context.hap_idx,
                                    m_rev,
                                    active_states
                                )));
                            }
                            if ws.allele_probs.len() == 2 {
                                posteriors[m_rev] = AllelePosteriors::Biallelic(ws.allele_probs[1]);
                            } else {
                                let mut out = Vec::with_capacity(ws.allele_probs.len());
                                out.extend_from_slice(&ws.allele_probs);
                                posteriors[m_rev] = AllelePosteriors::Multiallelic(out);
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

                    if uniform {
                        transition_only_backward_update(
                            &mut ws.bwd[..active_states],
                            recomb_rate,
                            transition_haps,
                        );
                    } else {
                        let mismatch_prob = fill_pattern_emissions(
                            seq_patterns.seq_alleles,
                            probs,
                            current_error,
                            &mut ws.emission_by_allele,
                            &mut ws.pattern_emissions,
                        );
                        if !probs.is_empty() && probs.len() > 1 {
                            let mut total_gamma = 0.0f64;
                            let mut mismatch_expect = 0.0f64;
                            for i in 0..active_states {
                                let gamma = (fwd_slice[i] * ws.bwd[i]) as f64;
                                let ref_allele = seq_patterns.allele_for_state(i);
                                if ref_allele == 255 {
                                    continue;
                                }
                                total_gamma += gamma;
                                let idx = ref_allele as usize;
                                let p_match = probs.get(idx).copied().unwrap_or(0.0);
                                let emission = if idx < ws.emission_by_allele.len() {
                                    ws.emission_by_allele[idx]
                                } else {
                                    mismatch_prob
                                };
                                if emission > 0.0 {
                                    let eta = (mismatch_prob * (1.0 - p_match) / emission) as f64;
                                    mismatch_expect += gamma * eta;
                                }
                            }
                            if total_gamma > 0.0 {
                                mismatch_sum += mismatch_expect / total_gamma;
                                mismatch_markers += 1.0;
                            }
                        }
                        let mut emit_beta_sum = 0.0f32;
                        for i in 0..active_states {
                            let pid = seq_patterns.state_patterns[i] as usize;
                            let emit = ws.pattern_emissions.get(pid).copied().unwrap_or(1.0);
                            ws.emissions[i] = emit;
                            emit_beta_sum += emit * ws.bwd[i];
                        }
                        let c_t = ws.fwd_scales.get(m_rev).copied().unwrap_or(1.0).max(1e-30);
                        let params = subset_transition_params(recomb_rate, active_states, transition_haps);
                        let scale = params.0 / c_t;
                        let shift = params.1 * (emit_beta_sum / c_t);
                        for i in 0..active_states {
                            ws.bwd[i] = scale * ws.emissions[i] * ws.bwd[i] + shift;
                        }
                    }
                }
                if block_start == 0 {
                    break;
                }
                block_start = block_start.saturating_sub(stride);
            }
        }

        if is_final {
            final_posteriors = posteriors;
        }
    }

    let stats = EmStats {
        expected_mismatches: mismatch_sum,
        informative_sites: mismatch_markers,
    };

    Ok((final_posteriors, final_prior_state_post, stats))
}

fn run_impute_hmm_dict(
    state_haps: &[RefHapId],
    ref_columns: &[DictColRef<'_>],
    target_probs: &TargetAlleleProbs,
    p_recomb: &[f32],
    error_rate: f32,
    prior_marker_idx: Option<usize>,
    state_priors: Option<&[f32]>,
    ref_allele_freqs: &RefAlleleFreqs,
    context: ImputeHmmContext,
    ws: &mut ImputeWorkspace,
) -> Result<(Vec<AllelePosteriors>, Option<Vec<f32>>, EmStats)> {
    validate_target_probs_nonempty(target_probs, context, "dictionary")?;
    let n_states = state_haps.len();
    let n_markers = target_probs.n_markers();
    ws.resize(n_states, n_markers);
    let active_states = ws.active_states();
    let active_markers = ws.active_markers();
    let checkpoint_stride = ws.configure_checkpoints(active_states, active_markers);
    let transition_haps = ref_allele_freqs.n_ref_haps().max(1);
    if active_states > 0 {
        ws.weights.fill(1.0);
    }

    let mut final_posteriors: Vec<AllelePosteriors> = Vec::new();
    let mut final_prior_state_post: Option<Vec<f32>> = None;
    let current_error = error_rate;
    let mut mismatch_sum = 0.0f64;
    let mut mismatch_markers = 0.0f64;

    let final_pass = 0usize;
    for pass in 0..1 {
        let is_final = pass == final_pass;
        let mut fwd_sum: f32;
        if let Some(priors) = state_priors {
            let len = priors.len().min(active_states);
            ws.fwd[..len].copy_from_slice(&priors[..len]);
            if len < active_states {
                ws.fwd[len..active_states].fill(0.0);
            }
            normalize_probs(&mut ws.fwd[..active_states]);
            fwd_sum = ws.fwd[..active_states].iter().sum::<f32>().max(1e-30);
        } else {
            let uniform = 1.0 / active_states.max(1) as f32;
            ws.fwd[..active_states].fill(uniform);
            fwd_sum = 1.0;
        }

        let mut last_dict_ptr: *const DictionaryColumn = std::ptr::null();
        for m in 0..active_markers {
            let use_prior_weighting = m == 0 && state_priors.is_some();
            fwd_sum = forward_update_dict(
                ws,
                m,
                use_prior_weighting,
                fwd_sum,
                state_haps,
                ref_columns,
                target_probs,
                p_recomb,
                current_error,
                active_states,
                transition_haps,
                &mut last_dict_ptr,
            );
            if m % checkpoint_stride == 0 {
                let cp = (m / checkpoint_stride) * active_states;
                ws.fwd_checkpoints[cp..cp + active_states]
                    .copy_from_slice(&ws.fwd[..active_states]);
            }
            if prior_marker_idx == Some(m) {
                final_prior_state_post = Some(ws.fwd[..active_states].to_vec());
            }
        }

        let mut posteriors: Vec<AllelePosteriors> = Vec::new();
        if is_final {
            posteriors.reserve(n_markers);
            posteriors.resize_with(n_markers, || AllelePosteriors::Biallelic(0.0));
        }

        ws.bwd.fill(1.0);
        if active_markers > 0 {
            let stride = checkpoint_stride.max(1);
            let mut block_start = ((active_markers - 1) / stride) * stride;
            let mut last_dict_ptr: *const DictionaryColumn = std::ptr::null();
            loop {
                let block_end = (block_start + stride).min(active_markers);
                let block_len = block_end.saturating_sub(block_start);
                ws.ensure_block_history(active_states, block_len);

                let cp = (block_start / stride) * active_states;
                ws.fwd[..active_states]
                    .copy_from_slice(&ws.fwd_checkpoints[cp..cp + active_states]);
                ws.fwd_history[..active_states].copy_from_slice(&ws.fwd[..active_states]);

                if block_start + 1 < block_end {
                    let mut local_sum = 1.0f32;
                    let mut last_dict_ptr_re: *const DictionaryColumn = std::ptr::null();
                    for m in (block_start + 1)..block_end {
                        local_sum = forward_update_dict(
                            ws,
                            m,
                            false,
                            local_sum,
                            state_haps,
                            ref_columns,
                            target_probs,
                            p_recomb,
                            current_error,
                            active_states,
                            transition_haps,
                            &mut last_dict_ptr_re,
                        );
                        let local_idx = (m - block_start) * active_states;
                        ws.fwd_history[local_idx..local_idx + active_states]
                            .copy_from_slice(&ws.fwd[..active_states]);
                    }
                }

                for m_rev in (block_start..block_end).rev() {
                    let probs = target_probs.probs_for_marker(m_rev);
                    let recomb_rate = p_recomb.get(m_rev).copied().unwrap_or(0.0);
                    let uniform = target_probs.is_uniform_marker(m_rev);

                    let col = &ref_columns[m_rev];
                    let dict_patterns = refresh_dict_patterns(
                        col,
                        &mut last_dict_ptr,
                        state_haps,
                        &mut ws.state_patterns,
                        &mut ws.dict_pattern_alleles,
                    );

                    let start = (m_rev - block_start) * active_states;
                    let fwd_slice = &ws.fwd_history[start..start + active_states];

                    if is_final {
                        ws.allele_probs.clear();
                        let n_alleles = probs.len();
                        if n_alleles > 0 {
                            ws.allele_probs.resize(n_alleles, 0.0f32);
                            let mut subset_counts = vec![0.0f32; n_alleles];
                            let mut subset_total = 0.0f32;
                            let mut total = 0.0f32;
                            let mut sq_sum = 0.0f32;
                            for i in 0..active_states {
                                let ref_allele = dict_patterns.allele_for_state(i);
                                if ref_allele == 255 {
                                    continue;
                                }
                                let state_prob = fwd_slice[i] * ws.bwd[i];
                                total += state_prob;
                                sq_sum += state_prob * state_prob;
                                let idx = ref_allele as usize;
                                if idx < ws.allele_probs.len() {
                                    ws.allele_probs[idx] += state_prob;
                                    subset_counts[idx] += 1.0;
                                    subset_total += 1.0;
                                }
                            }
                            if total > 0.0 {
                                for p in ws.allele_probs.iter_mut() {
                                    *p /= total;
                                }
                                if uniform && recomb_rate > 0.0 && subset_total > 0.0 {
                                    let inv_subset = 1.0 / subset_total;
                                    for p in subset_counts.iter_mut() {
                                        *p *= inv_subset;
                                    }
                                    smooth_allele_posteriors_subset(
                                        &mut ws.allele_probs,
                                        &subset_counts,
                                        total,
                                        sq_sum,
                                    );
                                }
                            } else {
                                return Err(ReagleError::vcf(format!(
                                    "Posterior mass collapse in imputation HMM (dictionary): window={} sample={} hap={} marker={} active_states={}",
                                    context.window_idx,
                                    context.sample_idx,
                                    context.hap_idx,
                                    m_rev,
                                    active_states
                                )));
                            }
                            if ws.allele_probs.len() == 2 {
                                posteriors[m_rev] = AllelePosteriors::Biallelic(ws.allele_probs[1]);
                            } else {
                                let mut out = Vec::with_capacity(ws.allele_probs.len());
                                out.extend_from_slice(&ws.allele_probs);
                                posteriors[m_rev] = AllelePosteriors::Multiallelic(out);
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

                    if uniform {
                        transition_only_backward_update(
                            &mut ws.bwd[..active_states],
                            recomb_rate,
                            transition_haps,
                        );
                    } else {
                        let mismatch_prob = fill_pattern_emissions(
                            dict_patterns.pattern_alleles,
                            probs,
                            current_error,
                            &mut ws.emission_by_allele,
                            &mut ws.pattern_emissions,
                        );
                        if !probs.is_empty() && probs.len() > 1 {
                            let mut total_gamma = 0.0f64;
                            let mut mismatch_expect = 0.0f64;
                            for i in 0..active_states {
                                let gamma = (fwd_slice[i] * ws.bwd[i]) as f64;
                                let ref_allele = dict_patterns.allele_for_state(i);
                                if ref_allele == 255 {
                                    continue;
                                }
                                total_gamma += gamma;
                                let idx = ref_allele as usize;
                                let p_match = probs.get(idx).copied().unwrap_or(0.0);
                                let emission = if idx < ws.emission_by_allele.len() {
                                    ws.emission_by_allele[idx]
                                } else {
                                    mismatch_prob
                                };
                                if emission > 0.0 {
                                    let eta = (mismatch_prob * (1.0 - p_match) / emission) as f64;
                                    mismatch_expect += gamma * eta;
                                }
                            }
                            if total_gamma > 0.0 {
                                mismatch_sum += mismatch_expect / total_gamma;
                                mismatch_markers += 1.0;
                            }
                        }
                        let mut emit_beta_sum = 0.0f32;
                        for i in 0..active_states {
                            let pid = dict_patterns.state_patterns[i] as usize;
                            let emit = ws.pattern_emissions.get(pid).copied().unwrap_or(1.0);
                            ws.emissions[i] = emit;
                            emit_beta_sum += emit * ws.bwd[i];
                        }
                        let c_t = ws.fwd_scales.get(m_rev).copied().unwrap_or(1.0).max(1e-30);
                        let params = subset_transition_params(recomb_rate, active_states, transition_haps);
                        let scale = params.0 / c_t;
                        let shift = params.1 * (emit_beta_sum / c_t);
                        for i in 0..active_states {
                            ws.bwd[i] = scale * ws.emissions[i] * ws.bwd[i] + shift;
                        }
                    }
                }
                if block_start == 0 {
                    break;
                }
                block_start = block_start.saturating_sub(stride);
            }
        }

        if is_final {
            final_posteriors = posteriors;
        }
    }

    let stats = EmStats {
        expected_mismatches: mismatch_sum,
        informative_sites: mismatch_markers,
    };

    Ok((final_posteriors, final_prior_state_post, stats))
}

/// Run forward-backward HMM and emit allele posteriors.
///
/// Returns (posteriors, optional state posterior at prior marker, EM stats).
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
    ws: &mut ImputeWorkspace,
) -> Result<(Vec<AllelePosteriors>, Option<Vec<f32>>, EmStats)> {
    if ref_columns.is_empty() {
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
            ws,
        );
    }

    if ref_columns
        .iter()
        .all(|col| matches!(col, GenotypeColumn::Dense(_)))
    {
        let dense_refs: Vec<&DenseColumn> = ref_columns
            .iter()
            .map(|col| match col {
                GenotypeColumn::Dense(c) => c,
                _ => unreachable!("Dense-only check failed"),
            })
            .collect();
        return run_impute_hmm_impl(
            state_haps,
            &dense_refs,
            target_probs,
            p_recomb,
            error_rate,
            prior_marker_idx,
            state_priors,
            ref_allele_freqs,
            context,
            ws,
        );
    }

    if ref_columns
        .iter()
        .all(|col| matches!(col, GenotypeColumn::Sparse(_)))
    {
        let sparse_refs: Vec<&SparseColumn> = ref_columns
            .iter()
            .map(|col| match col {
                GenotypeColumn::Sparse(c) => c,
                _ => unreachable!("Sparse-only check failed"),
            })
            .collect();
        return run_impute_hmm_impl(
            state_haps,
            &sparse_refs,
            target_probs,
            p_recomb,
            error_rate,
            prior_marker_idx,
            state_priors,
            ref_allele_freqs,
            context,
            ws,
        );
    }

    if ref_columns
        .iter()
        .all(|col| matches!(col, GenotypeColumn::SeqCoded(_)))
    {
        let seq_refs: Vec<&SeqCodedColumn> = ref_columns
            .iter()
            .map(|col| match col {
                GenotypeColumn::SeqCoded(c) => c,
                _ => unreachable!("SeqCoded-only check failed"),
            })
            .collect();
        return run_impute_hmm_seqcoded(
            state_haps,
            &seq_refs,
            target_probs,
            p_recomb,
            error_rate,
            prior_marker_idx,
            state_priors,
            ref_allele_freqs,
            context,
            ws,
        );
    }

    if ref_columns
        .iter()
        .all(|col| matches!(col, GenotypeColumn::Dictionary(_, _)))
    {
        let dict_refs: Vec<DictColRef<'_>> = ref_columns
            .iter()
            .map(|col| match col {
                GenotypeColumn::Dictionary(c, offset) => DictColRef {
                    col: c.as_ref(),
                    offset: *offset,
                },
                _ => unreachable!("Dictionary-only check failed"),
            })
            .collect();
        return run_impute_hmm_dict(
            state_haps,
            &dict_refs,
            target_probs,
            p_recomb,
            error_rate,
            prior_marker_idx,
            state_priors,
            ref_allele_freqs,
            context,
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
