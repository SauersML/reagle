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

/// Per-marker allele probability distributions for a single target haplotype.
pub struct TargetAlleleProbs {
    offsets: Vec<usize>,
    probs: Vec<f32>,
    uniform: Vec<bool>,
    observed: Vec<bool>,
}

impl TargetAlleleProbs {
    pub fn new(offsets: Vec<usize>, probs: Vec<f32>, observed: Vec<bool>) -> Self {
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
    subset_counts: Vec<f32>,
    state_posterior_scratch: Vec<f32>,
    allele_prior_scratch: Vec<f32>,
    dict_pattern_alleles: Vec<u8>,
    emission_by_allele: Vec<f32>,
    nearest_obs_fwd: Vec<f32>,
    nearest_obs_bwd: Vec<f32>,
    nearest_obs_lambda: Vec<f32>,
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
        Self {
            n_ref_haps,
        }
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
            subset_counts: Vec::new(),
            state_posterior_scratch: Vec::new(),
            allele_prior_scratch: Vec::new(),
            dict_pattern_alleles: Vec::new(),
            emission_by_allele: Vec::new(),
            nearest_obs_fwd: Vec::new(),
            nearest_obs_bwd: Vec::new(),
            nearest_obs_lambda: Vec::new(),
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
) {
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

    for m in 0..n {
        ws.nearest_obs_lambda[m] = ws.nearest_obs_fwd[m].min(ws.nearest_obs_bwd[m]);
        // When no typed marker is reachable in either direction, lambda stays
        // Infinity.  smooth_allele_posteriors_subset handles this correctly:
        // exp(-Inf) = 0 → clamped to MIN_RETAIN → maximum smoothing, which is
        // the right behavior for markers with zero LD anchor.
    }
}

#[inline]
fn smooth_allele_posteriors_subset(
    allele_probs: &mut [f32],
    subset_prior_probs: &[f32],
    nearest_obs_lambda: f32,
    total_mass: f32,
    state_prob_sq_sum: f32,
    untyped_uniform_marker: bool,
) {
    const MIN_RETAIN: f32 = 1e-4;
    if allele_probs.is_empty() || total_mass <= 0.0 || state_prob_sq_sum <= 0.0 {
        return;
    }
    if subset_prior_probs.len() != allele_probs.len() {
        return;
    }

    // Apply local-prior smoothing only on truly untyped/uniform markers.
    // Informative markers should be driven by likelihood, not AF pull.
    if !untyped_uniform_marker {
        return;
    }

    // Bayesian shrinkage based on genetic-distance information decay.
    // When far from observed markers, local copy evidence should carry less
    // confidence and panel prior should carry more.
    let effective_states = (total_mass * total_mass / state_prob_sq_sum).max(2.0);
    let retain = (-nearest_obs_lambda.max(0.0)).exp().clamp(MIN_RETAIN, 1.0);
    let prior_mass = (effective_states * (1.0 - retain) / retain).max(0.0);
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
fn normalized_allele_prior<'a>(
    out: &'a mut Vec<f32>,
    target_probs: &[f32],
) -> &'a [f32] {
    let n = target_probs.len();
    if out.len() < n {
        out.resize(n, 0.0);
    }
    let prior = &mut out[..n];
    let mut sum = 0.0f32;
    for i in 0..n {
        let mut v = target_probs[i];
        if !v.is_finite() || v < 0.0 {
            v = 0.0;
        }
        prior[i] = v;
        sum += v;
    }
    if sum <= 0.0 {
        let uniform = 1.0 / n.max(1) as f32;
        prior.fill(uniform);
        return prior;
    }
    let inv = 1.0 / sum;
    for p in prior.iter_mut() {
        *p *= inv;
    }
    prior
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
    let k_raw = active_states as f32;
    let n = n_ref_haps.max(1) as f32;
    // Condition Li-Stephens transitions on the active-state support.
    // In subset HMM mode the active support is the tractable state space;
    // scaling by k prevents under-switching when k << n.
    let k = k_raw.clamp(1.0, n);
    let switch_full = r / k.max(1.0);
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
    state_haps: &[RefHapId],
    ref_columns: &[C],
    target_probs: &TargetAlleleProbs,
    p_recomb: &[f32],
    current_error: f32,
    active_states: usize,
    transition_haps: usize,
) -> f32 {
    let probs = target_probs.probs_for_marker(m);
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

        if use_prior_weighting {
            if recomb_rate > 0.0 {
                WeightedHmmUpdater::fwd_update_weighted(
                    &mut ws.fwd,
                    1.0,
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
                1.0,
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
    state_haps: &[RefHapId],
    ref_columns: &[GenotypeColumn],
    target_probs: &TargetAlleleProbs,
    p_recomb: &[f32],
    current_error: f32,
    active_states: usize,
    transition_haps: usize,
    last_hap_ptr: &mut *const u16,
) -> f32 {
    let probs = target_probs.probs_for_marker(m);
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

        if use_prior_weighting {
            if recomb_rate > 0.0 {
                WeightedHmmUpdater::fwd_update_weighted(
                    &mut ws.fwd,
                    1.0,
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
                1.0,
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
    state_haps: &[RefHapId],
    ref_columns: &[GenotypeColumn],
    target_probs: &TargetAlleleProbs,
    p_recomb: &[f32],
    current_error: f32,
    active_states: usize,
    transition_haps: usize,
    last_dict_ptr: &mut *const DictionaryColumn,
) -> f32 {
    let probs = target_probs.probs_for_marker(m);
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

        if use_prior_weighting {
            if recomb_rate > 0.0 {
                WeightedHmmUpdater::fwd_update_weighted(
                    &mut ws.fwd,
                    1.0,
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
                1.0,
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
) -> Result<(Vec<AllelePosteriors>, Option<Vec<f32>>)> {
    validate_target_probs_nonempty(target_probs, context, "dense/sparse")?;
    validate_reference_marker_count(ref_columns.len(), target_probs, context, "dense/sparse")?;
    let n_states = state_haps.len();
    let n_markers = target_probs.n_markers();
    ws.resize(n_states, n_markers);
    let active_states = ws.active_states();
    let active_markers = ws.active_markers();
    let checkpoint_stride = ws.configure_checkpoints(active_states, active_markers);
    let panel_haps = ref_allele_freqs.n_ref_haps().max(1);
    let transition_haps = active_states.max(1).min(panel_haps);
    // Compute distance-based shrinkage only when untyped markers exist.
    let use_prior_smoothing = target_probs.has_untyped_markers();
    if use_prior_smoothing {
        compute_nearest_observed_lambda(ws, target_probs, p_recomb);
    } else {
        ws.nearest_obs_lambda.clear();
    }
    if active_states > 0 {
        // The active subset is the imputation state space for this haplotype/window.
        // Scale transitions to that subset to avoid suppressing switch mass by K/N_ref.
        ws.weights.fill(1.0);
    }

    let mut final_posteriors: Vec<AllelePosteriors> = Vec::new();
    let mut final_prior_state_post: Option<Vec<f32>> = None;
    let mut forward_prior_state_post: Option<Vec<f32>> = None;
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

        for m in 0..active_markers {
            let use_prior_weighting = m == 0 && state_priors.is_some();
            forward_update_impl(
                ws,
                m,
                use_prior_weighting,
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
                let mut snapshot = ws.fwd[..active_states].to_vec();
                normalize_probs(&mut snapshot);
                forward_prior_state_post = Some(snapshot);
            }
        }

        let mut posteriors: Vec<AllelePosteriors> = Vec::new();
        if is_final {
            posteriors.reserve(n_markers);
            posteriors.resize_with(n_markers, || AllelePosteriors::Biallelic(0.0));
        }

        ws.bwd.fill(1.0);
        let mut bwd_sum = active_states as f32;
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
                    for m in (block_start + 1)..block_end {
                        forward_update_impl(
                            ws,
                            m,
                            false,
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
                    let uniform = target_probs.is_uniform_marker(m_rev);
                    let recomb_rate = marker_recomb_rate(p_recomb, m_rev);
                    let n_alleles = probs.len();
                    if prior_marker_idx == Some(m_rev) {
                        ws.ensure_state_posterior_scratch(active_states);
                    }
                    if is_final && n_alleles > 0 {
                        ws.ensure_subset_counts(n_alleles);
                    }

                    let start = (m_rev - block_start) * active_states;
                    let fwd_slice = &ws.fwd_history[start..start + active_states];
                    if prior_marker_idx == Some(m_rev) && forward_prior_state_post.is_none() {
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
                        if n_alleles > 0 {
                            ws.allele_probs.resize(n_alleles, 0.0f32);
                            let subset_counts = &mut ws.subset_counts[..n_alleles];
                            subset_counts.fill(0.0);
                            let mut subset_total = 0.0f32;
                            let mut total = 0.0f32;
                            let mut sq_sum = 0.0f32;
                            let mut missing_mass = 0.0f32;
                            for i in 0..active_states {
                                let state_prob = fwd_slice[i] * ws.bwd[i];
                                total += state_prob;
                                sq_sum += state_prob * state_prob;
                                let ref_allele = ref_alleles.get(i);
                                if ref_allele == 255 {
                                    missing_mass += state_prob;
                                    continue;
                                }
                                let idx = ref_allele as usize;
                                if idx < ws.allele_probs.len() {
                                    ws.allele_probs[idx] += state_prob;
                                    subset_counts[idx] += 1.0;
                                    subset_total += 1.0;
                                }
                            }
                            if total > 0.0 {
                                if missing_mass > 0.0 {
                                    let prior = normalized_allele_prior(
                                        &mut ws.allele_prior_scratch,
                                        probs,
                                    );
                                    for (i, p) in ws.allele_probs.iter_mut().enumerate() {
                                        *p += missing_mass * prior[i];
                                    }
                                }
                                for p in ws.allele_probs.iter_mut() {
                                    *p /= total;
                                }
                                if use_prior_smoothing
                                    && uniform
                                    && recomb_rate > 0.0
                                    && subset_total > 0.0
                                {
                                    // Use full-panel allele frequency (stored in
                                    // TargetAlleleProbs) as the smoothing prior instead
                                    // of subset AF, so that abyss haplotypes' allele
                                    // distribution is properly represented.
                                    let prior_counts = &mut ws.subset_counts[..n_alleles];
                                    for idx in 0..n_alleles {
                                        prior_counts[idx] = probs.get(idx).copied().unwrap_or(0.0)
                                            * active_states as f32;
                                    }
                                    smooth_allele_posteriors_subset(
                                        &mut ws.allele_probs,
                                        prior_counts,
                                        ws.nearest_obs_lambda.get(m_rev).copied().unwrap_or(f32::INFINITY),
                                        total,
                                        sq_sum,
                                        target_probs.is_untyped_uniform_marker(m_rev),
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
                        bwd_sum = transition_only_backward_update(
                            &mut ws.bwd[..active_states],
                            recomb_rate,
                            transition_haps,
                            bwd_sum,
                        );
                    } else {
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

    if forward_prior_state_post.is_some() {
        final_prior_state_post = forward_prior_state_post;
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
    ws: &mut ImputeWorkspace,
) -> Result<(Vec<AllelePosteriors>, Option<Vec<f32>>)> {
    validate_target_probs_nonempty(target_probs, context, "seqcoded")?;
    validate_reference_marker_count(ref_columns.len(), target_probs, context, "seqcoded")?;
    let n_states = state_haps.len();
    let n_markers = target_probs.n_markers();
    ws.resize(n_states, n_markers);
    let active_states = ws.active_states();
    let active_markers = ws.active_markers();
    let checkpoint_stride = ws.configure_checkpoints(active_states, active_markers);
    let panel_haps = ref_allele_freqs.n_ref_haps().max(1);
    let transition_haps = active_states.max(1).min(panel_haps);
    // Compute distance-based shrinkage only when untyped markers exist.
    let use_prior_smoothing = target_probs.has_untyped_markers();
    if use_prior_smoothing {
        compute_nearest_observed_lambda(ws, target_probs, p_recomb);
    } else {
        ws.nearest_obs_lambda.clear();
    }
    if active_states > 0 {
        ws.weights.fill(1.0);
    }

    let mut final_posteriors: Vec<AllelePosteriors> = Vec::new();
    let mut final_prior_state_post: Option<Vec<f32>> = None;
    let mut forward_prior_state_post: Option<Vec<f32>> = None;
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
        for m in 0..active_markers {
            let use_prior_weighting = m == 0 && state_priors.is_some();
            forward_update_seqcoded(
                ws,
                m,
                use_prior_weighting,
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
                let mut snapshot = ws.fwd[..active_states].to_vec();
                normalize_probs(&mut snapshot);
                forward_prior_state_post = Some(snapshot);
            }
        }

        let mut posteriors: Vec<AllelePosteriors> = Vec::new();
        if is_final {
            posteriors.reserve(n_markers);
            posteriors.resize_with(n_markers, || AllelePosteriors::Biallelic(0.0));
        }

        ws.bwd.fill(1.0);
        let mut bwd_sum = active_states as f32;
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
                    let mut last_hap_ptr_re: *const u16 = std::ptr::null();
                    for m in (block_start + 1)..block_end {
                        forward_update_seqcoded(
                            ws,
                            m,
                            false,
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
                    let uniform = target_probs.is_uniform_marker(m_rev);
                    let recomb_rate = marker_recomb_rate(p_recomb, m_rev);
                    let n_alleles = probs.len();
                    if prior_marker_idx == Some(m_rev) {
                        ws.ensure_state_posterior_scratch(active_states);
                    }
                    if is_final && n_alleles > 0 {
                        ws.ensure_subset_counts(n_alleles);
                    }

                    let col = seqcoded_col(&ref_columns[m_rev]);
                    let seq_patterns = refresh_seq_patterns(
                        col,
                        &mut last_hap_ptr,
                        state_haps,
                        &mut ws.state_patterns,
                    );

                    let start = (m_rev - block_start) * active_states;
                    let fwd_slice = &ws.fwd_history[start..start + active_states];
                    if prior_marker_idx == Some(m_rev) && forward_prior_state_post.is_none() {
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
                            subset_counts.fill(0.0);
                            let mut subset_total = 0.0f32;
                            let mut total = 0.0f32;
                            let mut sq_sum = 0.0f32;
                            let mut missing_mass = 0.0f32;
                            for i in 0..active_states {
                                let state_prob = fwd_slice[i] * ws.bwd[i];
                                total += state_prob;
                                sq_sum += state_prob * state_prob;
                                let ref_allele = seq_patterns.allele_for_state(i);
                                if ref_allele == 255 {
                                    missing_mass += state_prob;
                                    continue;
                                }
                                let idx = ref_allele as usize;
                                if idx < ws.allele_probs.len() {
                                    ws.allele_probs[idx] += state_prob;
                                    subset_counts[idx] += 1.0;
                                    subset_total += 1.0;
                                }
                            }
                            if total > 0.0 {
                                if missing_mass > 0.0 {
                                    let prior = normalized_allele_prior(
                                        &mut ws.allele_prior_scratch,
                                        probs,
                                    );
                                    for (i, p) in ws.allele_probs.iter_mut().enumerate() {
                                        *p += missing_mass * prior[i];
                                    }
                                }
                                for p in ws.allele_probs.iter_mut() {
                                    *p /= total;
                                }
                                if use_prior_smoothing
                                    && uniform
                                    && recomb_rate > 0.0
                                    && subset_total > 0.0
                                {
                                    let prior_counts = &mut ws.subset_counts[..n_alleles];
                                    for idx in 0..n_alleles {
                                        prior_counts[idx] = probs.get(idx).copied().unwrap_or(0.0)
                                            * active_states as f32;
                                    }
                                    smooth_allele_posteriors_subset(
                                        &mut ws.allele_probs,
                                        prior_counts,
                                        ws.nearest_obs_lambda.get(m_rev).copied().unwrap_or(f32::INFINITY),
                                        total,
                                        sq_sum,
                                        target_probs.is_untyped_uniform_marker(m_rev),
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

    if forward_prior_state_post.is_some() {
        final_prior_state_post = forward_prior_state_post;
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
    ws: &mut ImputeWorkspace,
) -> Result<(Vec<AllelePosteriors>, Option<Vec<f32>>)> {
    validate_target_probs_nonempty(target_probs, context, "dictionary")?;
    validate_reference_marker_count(ref_columns.len(), target_probs, context, "dictionary")?;
    let n_states = state_haps.len();
    let n_markers = target_probs.n_markers();
    ws.resize(n_states, n_markers);
    let active_states = ws.active_states();
    let active_markers = ws.active_markers();
    let checkpoint_stride = ws.configure_checkpoints(active_states, active_markers);
    let panel_haps = ref_allele_freqs.n_ref_haps().max(1);
    let transition_haps = active_states.max(1).min(panel_haps);
    // Compute distance-based shrinkage only when untyped markers exist.
    let use_prior_smoothing = target_probs.has_untyped_markers();
    if use_prior_smoothing {
        compute_nearest_observed_lambda(ws, target_probs, p_recomb);
    } else {
        ws.nearest_obs_lambda.clear();
    }
    if active_states > 0 {
        ws.weights.fill(1.0);
    }

    let mut final_posteriors: Vec<AllelePosteriors> = Vec::new();
    let mut final_prior_state_post: Option<Vec<f32>> = None;
    let mut forward_prior_state_post: Option<Vec<f32>> = None;
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
        for m in 0..active_markers {
            let use_prior_weighting = m == 0 && state_priors.is_some();
            forward_update_dict(
                ws,
                m,
                use_prior_weighting,
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
                let mut snapshot = ws.fwd[..active_states].to_vec();
                normalize_probs(&mut snapshot);
                forward_prior_state_post = Some(snapshot);
            }
        }

        let mut posteriors: Vec<AllelePosteriors> = Vec::new();
        if is_final {
            posteriors.reserve(n_markers);
            posteriors.resize_with(n_markers, || AllelePosteriors::Biallelic(0.0));
        }

        ws.bwd.fill(1.0);
        let mut bwd_sum = active_states as f32;
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
                    let mut last_dict_ptr_re: *const DictionaryColumn = std::ptr::null();
                    for m in (block_start + 1)..block_end {
                        forward_update_dict(
                            ws,
                            m,
                            false,
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
                    let uniform = target_probs.is_uniform_marker(m_rev);
                    let recomb_rate = marker_recomb_rate(p_recomb, m_rev);
                    let n_alleles = probs.len();
                    if prior_marker_idx == Some(m_rev) {
                        ws.ensure_state_posterior_scratch(active_states);
                    }
                    if is_final && n_alleles > 0 {
                        ws.ensure_subset_counts(n_alleles);
                    }

                    let col = dict_col_ref(&ref_columns[m_rev]);
                    let dict_patterns = refresh_dict_patterns(
                        &col,
                        &mut last_dict_ptr,
                        state_haps,
                        &mut ws.state_patterns,
                        &mut ws.dict_pattern_alleles,
                    );

                    let start = (m_rev - block_start) * active_states;
                    let fwd_slice = &ws.fwd_history[start..start + active_states];
                    if prior_marker_idx == Some(m_rev) && forward_prior_state_post.is_none() {
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
                            subset_counts.fill(0.0);
                            let mut subset_total = 0.0f32;
                            let mut total = 0.0f32;
                            let mut sq_sum = 0.0f32;
                            let mut missing_mass = 0.0f32;
                            for i in 0..active_states {
                                let state_prob = fwd_slice[i] * ws.bwd[i];
                                total += state_prob;
                                sq_sum += state_prob * state_prob;
                                let ref_allele = dict_patterns.allele_for_state(i);
                                if ref_allele == 255 {
                                    missing_mass += state_prob;
                                    continue;
                                }
                                let idx = ref_allele as usize;
                                if idx < ws.allele_probs.len() {
                                    ws.allele_probs[idx] += state_prob;
                                    subset_counts[idx] += 1.0;
                                    subset_total += 1.0;
                                }
                            }
                            if total > 0.0 {
                                if missing_mass > 0.0 {
                                    let prior = normalized_allele_prior(
                                        &mut ws.allele_prior_scratch,
                                        probs,
                                    );
                                    for (i, p) in ws.allele_probs.iter_mut().enumerate() {
                                        *p += missing_mass * prior[i];
                                    }
                                }
                                for p in ws.allele_probs.iter_mut() {
                                    *p /= total;
                                }
                                if use_prior_smoothing
                                    && uniform
                                    && recomb_rate > 0.0
                                    && subset_total > 0.0
                                {
                                    // Use full-panel allele frequency (stored in
                                    // TargetAlleleProbs) as the smoothing prior instead
                                    // of subset AF, so that abyss haplotypes' allele
                                    // distribution is properly represented.
                                    let prior_counts = &mut ws.subset_counts[..n_alleles];
                                    for idx in 0..n_alleles {
                                        prior_counts[idx] = probs.get(idx).copied().unwrap_or(0.0)
                                            * active_states as f32;
                                    }
                                    smooth_allele_posteriors_subset(
                                        &mut ws.allele_probs,
                                        prior_counts,
                                        ws.nearest_obs_lambda.get(m_rev).copied().unwrap_or(f32::INFINITY),
                                        total,
                                        sq_sum,
                                        target_probs.is_untyped_uniform_marker(m_rev),
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

    if forward_prior_state_post.is_some() {
        final_prior_state_post = forward_prior_state_post;
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
