//! HMM kernel for imputation using explicit haplotype states.
//!
//! This implements a Li-Stephens forward-backward pass over a selected set of
//! reference haplotypes (state set). Emissions are computed using per-haplotype
//! allele probabilities from the target, and reference alleles are read on demand.

use crate::data::marker::{MarkerIdx, Markers};
use crate::data::storage::{
    DenseColumn, DictionaryColumn, GenotypeColumn, SeqCodedColumn, SparseColumn,
};
use crate::data::HapIdx;
use crate::model::weighted_kernel::WeightedHmmUpdater;
use crate::model::types::GlobalId;
use crate::pipelines::imputation::AllelePosteriors;
use std::sync::OnceLock;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};

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

/// Lazily computed reference allele frequencies per marker.
pub struct RefAlleleFreqs<'a, Space = crate::data::marker::AnyMarkerSpace> {
    ref_columns: &'a [GenotypeColumn],
    ref_markers: &'a Markers<Space>,
    n_ref_haps: usize,
    cache: Vec<OnceLock<Vec<f32>>>,
}

impl<'a, Space> RefAlleleFreqs<'a, Space> {
    pub fn new(ref_columns: &'a [GenotypeColumn], ref_markers: &'a Markers<Space>) -> Self {
        let n_ref_haps = ref_columns
            .first()
            .map(|c| c.n_haplotypes())
            .unwrap_or(0);
        let cache = std::iter::repeat_with(OnceLock::new)
            .take(ref_markers.len())
            .collect();
        Self {
            ref_columns,
            ref_markers,
            n_ref_haps,
            cache,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    #[inline]
    pub fn n_alleles(&self, marker_idx: usize) -> usize {
        self.ref_markers
            .marker(MarkerIdx::new(marker_idx as u32))
            .n_alleles()
            .max(1)
    }

    pub fn get(&self, marker_idx: usize) -> Option<&[f32]> {
        if marker_idx >= self.cache.len() {
            return None;
        }
        if self.n_ref_haps == 0 {
            return None;
        }
        let freqs = self.cache[marker_idx].get_or_init(|| {
            let n_alleles = self.n_alleles(marker_idx);
            let mut counts = vec![0u32; n_alleles];
            let mut total = 0u32;
            match &self.ref_columns[marker_idx] {
                GenotypeColumn::Dense(col) => {
                    for h in 0..self.n_ref_haps {
                        let a = col.get(HapIdx::new(h as u32));
                        if a == 255 {
                            continue;
                        }
                        let idx = a as usize;
                        if idx < counts.len() {
                            counts[idx] += 1;
                            total += 1;
                        }
                    }
                }
                GenotypeColumn::Sparse(col) => {
                    for h in 0..self.n_ref_haps {
                        let a = col.get(HapIdx::new(h as u32));
                        if a == 255 {
                            continue;
                        }
                        let idx = a as usize;
                        if idx < counts.len() {
                            counts[idx] += 1;
                            total += 1;
                        }
                    }
                }
                GenotypeColumn::Dictionary(col, offset) => {
                    for h in 0..self.n_ref_haps {
                        let a = col.get(*offset, HapIdx::new(h as u32));
                        if a == 255 {
                            continue;
                        }
                        let idx = a as usize;
                        if idx < counts.len() {
                            counts[idx] += 1;
                            total += 1;
                        }
                    }
                }
                GenotypeColumn::SeqCoded(col) => {
                    for h in 0..self.n_ref_haps {
                        let a = col.get(HapIdx::new(h as u32));
                        if a == 255 {
                            continue;
                        }
                        let idx = a as usize;
                        if idx < counts.len() {
                            counts[idx] += 1;
                            total += 1;
                        }
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
            out
        });
        Some(freqs.as_slice())
    }
}

impl ImputeWorkspace {
    pub fn new(n_states: usize, n_markers: usize) -> Self {
        Self {
            fwd: vec![0.0; n_states],
            bwd: vec![1.0; n_states],
            emissions: vec![1.0; n_states],
            fwd_history: vec![0.0; n_states * n_markers],
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
        let want = n_states.saturating_mul(n_markers);
        if self.fwd_history.len() < want {
            self.fwd_history.resize(want, 0.0);
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
        state_haps: &[HapIdx],
        out: &mut [u8],
        dict_pattern_alleles: &mut Vec<u8>,
    );
}

impl<T: RefColumnLike + ?Sized> RefColumnLike for &T {
    #[inline]
    fn fill_ref_alleles(
        &self,
        state_haps: &[HapIdx],
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
        state_haps: &[HapIdx],
        out: &mut [u8],
        dict_pattern_alleles: &mut Vec<u8>,
    ) {
        std::hint::black_box(dict_pattern_alleles.len());
        if self.bits_per_allele() == 1 {
            let n_haps = self.n_haplotypes();
            let bits = self.bits_raw();
            let missing = self.missing_raw();
            for (i, hap) in state_haps.iter().enumerate() {
                let idx = hap.as_usize();
                if idx >= n_haps {
                    out[i] = 0;
                    continue;
                }
                let word_idx = idx >> 6;
                let bit_idx = idx & 63;
                if word_idx < missing.len()
                    && ((missing[word_idx] >> bit_idx) & 1) != 0
                {
                    out[i] = 255;
                    continue;
                }
                let bit = if word_idx < bits.len() {
                    (bits[word_idx] >> bit_idx) & 1
                } else {
                    0
                };
                out[i] = bit as u8;
            }
        } else {
            for (i, hap) in state_haps.iter().enumerate() {
                out[i] = self.get(*hap);
            }
        }
    }
}

impl RefColumnLike for SparseColumn {
    #[inline]
    fn fill_ref_alleles(
        &self,
        state_haps: &[HapIdx],
        out: &mut [u8],
        dict_pattern_alleles: &mut Vec<u8>,
    ) {
        std::hint::black_box(dict_pattern_alleles.len());
        for (i, hap) in state_haps.iter().enumerate() {
            out[i] = self.get(*hap);
        }
    }
}

impl RefColumnLike for SeqCodedColumn {
    #[inline]
    fn fill_ref_alleles(
        &self,
        state_haps: &[HapIdx],
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
        state_haps: &[HapIdx],
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
        state_haps: &[HapIdx],
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
fn fill_emissions(
    ref_alleles: &[u8],
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
    for (i, &ref_allele) in ref_alleles.iter().enumerate() {
        if i + prefetch_stride < ref_alleles.len() {
            unsafe {
                prefetch_read(ref_alleles.as_ptr().add(i + prefetch_stride));
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
fn transition_only_forward_update(
    fwd: &mut [f32],
    fwd_sum: f32,
    recomb_rate: f32,
    total_ref_haps: usize,
) -> f32 {
    if fwd.is_empty() {
        return 0.0;
    }
    if recomb_rate <= 0.0 {
        return fwd_sum;
    }
    let denom = fwd_sum.max(1e-30);
    let scale = (1.0 - recomb_rate) / denom;
    let shift = if total_ref_haps > 0 {
        recomb_rate / total_ref_haps as f32
    } else {
        0.0
    };
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
    total_ref_haps: usize,
) {
    if bwd.is_empty() || recomb_rate <= 0.0 {
        return;
    }
    let mut sum = 0.0f32;
    for v in bwd.iter() {
        sum += *v;
    }
    let shift = if total_ref_haps > 0 {
        (recomb_rate / total_ref_haps as f32) * sum
    } else {
        0.0
    };
    let scale = 1.0 - recomb_rate;
    for v in bwd.iter_mut() {
        *v = scale.mul_add(*v, shift);
    }
}

#[inline]
fn fill_state_patterns_seqcoded(
    hap_to_seq: &[u16],
    state_haps: &[HapIdx],
    out: &mut [u16],
) {
    for (i, hap) in state_haps.iter().enumerate() {
        out[i] = hap_to_seq[hap.as_usize()];
    }
}

#[inline]
fn fill_state_patterns_dict(
    col: &DictionaryColumn,
    state_haps: &[HapIdx],
    out: &mut [u16],
) {
    for (i, hap) in state_haps.iter().enumerate() {
        out[i] = col.hap_pattern_idx(*hap) as u16;
    }
}

/// Monomorphized HMM core over a concrete genotype column type.
fn run_impute_hmm_impl<Space, C: RefColumnLike>(
    state_haps: &[GlobalId],
    ref_columns: &[C],
    target_probs: &TargetAlleleProbs,
    p_recomb: &[f32],
    error_rate: f32,
    prior_marker_idx: Option<usize>,
    state_priors: Option<&[f32]>,
    total_ref_haps: usize,
    ref_allele_freqs: &RefAlleleFreqs<'_, Space>,
    ws: &mut ImputeWorkspace,
) -> (Vec<AllelePosteriors>, Option<Vec<f32>>) {
    let n_states = state_haps.len();
    let n_markers = target_probs.n_markers();
    ws.resize(n_states, n_markers);
    let active_states = ws.active_states();
    let active_markers = ws.active_markers();
    let state_hap_idx: Vec<HapIdx> = state_haps
        .iter()
        .map(|h| HapIdx::new(h.as_u32()))
        .collect();
    if active_states > 0 {
        // Li-Stephens transition for full panel:
        //   P(switch to h_j) = r / N
        // For subset K (with abyss), modeled switch mass is:
        //   sum_{j in K} r/N = r * K/N
        // Remaining mass r * (N-K)/N represents recombination to unmodeled haplotypes (abyss).
        // Therefore we keep weights=1.0 (no scaling). Scaling by N/K would force a
        // closed-universe model that consumes all recombination mass inside K.
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
    let e_ls = crate::model::parameters::ModelParams::li_stephens_p_mismatch(
        total_ref_haps.max(2),
    );
    let prior_alpha = 1.0 + e_ls;
    let prior_beta = 1.0 + (1.0 - e_ls);
    let prior_denom = (prior_alpha + prior_beta - 2.0).max(0.0);

    let mut final_posteriors: Vec<AllelePosteriors> = Vec::new();
    let mut final_prior_state_post: Option<Vec<f32>> = None;
    let mut current_error = error_rate;

    for pass in 0..2 {
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

        const TILE_SIZE: usize = 256;
        let mut tile_start = 0usize;
        while tile_start < active_markers {
            let tile_end = (tile_start + TILE_SIZE).min(active_markers);
            for m in tile_start..tile_end {
                let probs = target_probs.probs_for_marker(m);
                let recomb_rate = p_recomb.get(m).copied().unwrap_or(0.0);
                let uniform = target_probs.is_uniform_marker(m);

                let ref_slice = &mut ws.state_alleles[..active_states];
                if !(pass == 0 && uniform) {
                    ref_columns[m].fill_ref_alleles(
                        &state_hap_idx,
                        ref_slice,
                        &mut ws.dict_pattern_alleles,
                    );
                }
                if uniform {
                    fwd_sum = transition_only_forward_update(
                        &mut ws.fwd[..active_states],
                        fwd_sum,
                        recomb_rate,
                        total_ref_haps.max(1),
                    );
                } else {
                    fill_emissions(
                        ref_slice,
                        probs,
                        current_error,
                        &mut ws.emission_by_allele,
                        &mut ws.emissions[..active_states],
                    );

                    if m == 0 && state_priors.is_some() {
                        if recomb_rate > 0.0 {
                            fwd_sum = WeightedHmmUpdater::fwd_update_weighted(
                                &mut ws.fwd,
                                fwd_sum,
                                recomb_rate,
                                total_ref_haps.max(1),
                                &ws.weights,
                                &ws.emissions,
                                active_states,
                            );
                        } else {
                            for i in 0..active_states {
                                ws.fwd[i] *= ws.emissions[i];
                            }
                            fwd_sum = ws.fwd[..active_states].iter().sum::<f32>().max(1e-30);
                        }
                    } else {
                        fwd_sum = WeightedHmmUpdater::fwd_update_weighted(
                            &mut ws.fwd,
                            fwd_sum,
                            recomb_rate,
                            total_ref_haps.max(1),
                            &ws.weights,
                            &ws.emissions,
                            active_states,
                        );
                    }
                }

                if fwd_sum <= 0.0 {
                    fwd_sum = 1e-30;
                }
                ws.fwd_scales[m] = if uniform { 1.0 } else { fwd_sum };
                let inv = 1.0 / fwd_sum;
                for i in 0..active_states {
                    ws.fwd[i] *= inv;
                }
                // After scaling, sum(fwd) == 1.0
                fwd_sum = 1.0;
                let start = m * active_states;
                ws.fwd_history[start..start + active_states]
                    .copy_from_slice(&ws.fwd[..active_states]);
            }
            tile_start = tile_end;
        }

        let mut posteriors: Vec<AllelePosteriors> = Vec::new();
        if pass == 1 {
            posteriors.reserve(n_markers);
            posteriors.resize_with(n_markers, || AllelePosteriors::Biallelic(0.0));
        }

        ws.bwd.fill(1.0);
        let mut prior_state_post: Option<Vec<f32>> = None;

        // Accumulate expected mismatches for MAP update.
        let mut mismatch_sum = 0.0f64;
        let mut mismatch_markers = 0.0f64;

        let mut tile_end = active_markers;
        while tile_end > 0 {
            let tile_start = tile_end.saturating_sub(TILE_SIZE);
            for m_rev in (tile_start..tile_end).rev() {
                let probs = target_probs.probs_for_marker(m_rev);
                let recomb_rate = p_recomb.get(m_rev).copied().unwrap_or(0.0);
                let uniform = target_probs.is_uniform_marker(m_rev);

            let start = m_rev * active_states;
            let ref_slice = &mut ws.state_alleles[..active_states];
            let fwd_slice = &ws.fwd_history[start..start + active_states];
            if !(pass == 0 && uniform) {
                ref_columns[m_rev].fill_ref_alleles(
                    &state_hap_idx,
                    ref_slice,
                    &mut ws.dict_pattern_alleles,
                );
            }

            if pass == 1 {
                // Compute posteriors using beta at time t (current ws.bwd), before updating
                // beta for time t-1. This aligns alpha_t * beta_t for marker-level posteriors.
                ws.allele_probs.clear();
                let n_alleles = if !probs.is_empty() {
                    probs.len()
                } else if m_rev < ref_allele_freqs.len() {
                    ref_allele_freqs.n_alleles(m_rev)
                } else {
                    0
                };
                if n_alleles > 0 {
                    ws.allele_probs.resize(n_alleles, 0.0f32);
                    let mut total = 0.0f32;
                    for i in 0..active_states {
                        let ref_allele = ref_slice[i];
                        if ref_allele == 255 {
                            continue;
                        }
                        let state_prob = fwd_slice[i] * ws.bwd[i];
                        total += state_prob;
                        let idx = ref_allele as usize;
                        if idx < ws.allele_probs.len() {
                            ws.allele_probs[idx] += state_prob;
                        }
                    }
                    if total > 0.0 {
                        for p in ws.allele_probs.iter_mut() {
                            *p /= total;
                        }
                    } else if let Some(freqs) = ref_allele_freqs.get(m_rev) {
                        let mut sum = 0.0f32;
                        for (i, p) in ws.allele_probs.iter_mut().enumerate() {
                            let f = freqs.get(i).copied().unwrap_or(0.0).max(0.0);
                            *p = f;
                            sum += f;
                        }
                        if sum > 0.0 {
                            let inv = 1.0 / sum;
                            for p in ws.allele_probs.iter_mut() {
                                *p *= inv;
                            }
                        } else {
                            let uniform = 1.0 / ws.allele_probs.len().max(1) as f32;
                            for p in ws.allele_probs.iter_mut() {
                                *p = uniform;
                            }
                        }
                    } else {
                        let uniform = 1.0 / ws.allele_probs.len().max(1) as f32;
                        for p in ws.allele_probs.iter_mut() {
                            *p = uniform;
                        }
                    }
                    if ws.allele_probs.len() == 2 {
                        posteriors[m_rev] = AllelePosteriors::Biallelic(ws.allele_probs[1]);
                    } else {
                        let mut out = Vec::with_capacity(ws.allele_probs.len());
                        out.extend_from_slice(&ws.allele_probs);
                        posteriors[m_rev] = AllelePosteriors::Multiallelic(out);
                    }
                } else {
                    posteriors[m_rev] = AllelePosteriors::Biallelic(0.0);
                }

                if prior_marker_idx == Some(m_rev) {
                    let mut state_post = vec![0.0f32; active_states];
                    let mut total = 0.0f32;
                    for i in 0..active_states {
                        let v = fwd_slice[i] * ws.bwd[i];
                        state_post[i] = v;
                        total += v;
                    }
                    if total > 0.0 {
                        let inv = 1.0 / total;
                        for v in state_post.iter_mut() {
                            *v *= inv;
                        }
                    }
                    prior_state_post = Some(state_post);
                }
            }

            // Expected mismatch accumulation for MAP update.
            // Only use informative markers (non-empty allele probs and >1 allele).
            if pass == 0 && !uniform && !probs.is_empty() && probs.len() > 1 {
                fill_emissions(
                    ref_slice,
                    probs,
                    current_error,
                    &mut ws.emission_by_allele,
                    &mut ws.emissions[..active_states],
                );
                let mismatch_prob = current_error / (probs.len() as f32 - 1.0);
                let mut total_gamma = 0.0f64;
                let mut mismatch_expect = 0.0f64;
                for i in 0..active_states {
                    let gamma = (fwd_slice[i] * ws.bwd[i]) as f64;
                    let ref_allele = ref_slice[i];
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

            // Update beta for the previous marker.
            // Scaled backward recursion:
            //   beta_{t-1}(i) = ( (1-r) * b_t(i) * beta_t(i) + (r/N) * S_t ) / c_t
            // where S_t = sum_j b_t(j) * beta_t(j) and c_t is the forward scale
            // at marker t (sum of unnormalized alpha_t).
                if uniform {
                    transition_only_backward_update(
                        &mut ws.bwd[..active_states],
                        recomb_rate,
                        total_ref_haps.max(1),
                    );
                } else {
                    fill_emissions(
                        ref_slice,
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
                    let scale = (1.0 - recomb_rate) / c_t;
                    let shift = if total_ref_haps > 0 {
                        (recomb_rate / total_ref_haps as f32) * (emit_beta_sum / c_t)
                    } else {
                        0.0
                    };
                    for i in 0..active_states {
                        ws.bwd[i] = scale * ws.emissions[i] * ws.bwd[i] + shift;
                    }
                }
            }
            tile_end = tile_start;
        }

        if pass == 0 {
            // MAP update (minimal Beta prior) and run one more pass.
            if mismatch_markers > 0.0 {
                let numerator = mismatch_sum + (prior_alpha - 1.0) as f64;
                let denom = mismatch_markers + prior_denom as f64;
                let e_map = (numerator / denom).clamp(1e-8, 0.5 - 1e-8) as f32;
                current_error = e_map;
            }
        } else {
            final_posteriors = posteriors;
            final_prior_state_post = prior_state_post;
        }
    }

    (final_posteriors, final_prior_state_post)
}

fn run_impute_hmm_seqcoded<Space>(
    state_haps: &[GlobalId],
    ref_columns: &[&SeqCodedColumn],
    target_probs: &TargetAlleleProbs,
    p_recomb: &[f32],
    error_rate: f32,
    prior_marker_idx: Option<usize>,
    state_priors: Option<&[f32]>,
    total_ref_haps: usize,
    ref_allele_freqs: &RefAlleleFreqs<'_, Space>,
    ws: &mut ImputeWorkspace,
) -> (Vec<AllelePosteriors>, Option<Vec<f32>>) {
    let n_states = state_haps.len();
    let n_markers = target_probs.n_markers();
    ws.resize(n_states, n_markers);
    let active_states = ws.active_states();
    let active_markers = ws.active_markers();
    let state_hap_idx: Vec<HapIdx> = state_haps
        .iter()
        .map(|h| HapIdx::new(h.as_u32()))
        .collect();
    if active_states > 0 {
        ws.weights.fill(1.0);
    }

    let e_ls = crate::model::parameters::ModelParams::li_stephens_p_mismatch(
        total_ref_haps.max(2),
    );
    let prior_alpha = 1.0 + e_ls;
    let prior_beta = 1.0 + (1.0 - e_ls);
    let prior_denom = (prior_alpha + prior_beta - 2.0).max(0.0);

    let mut final_posteriors: Vec<AllelePosteriors> = Vec::new();
    let mut final_prior_state_post: Option<Vec<f32>> = None;
    let mut current_error = error_rate;

    for pass in 0..2 {
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
        const TILE_SIZE: usize = 256;
        let mut tile_start = 0usize;
        while tile_start < active_markers {
            let tile_end = (tile_start + TILE_SIZE).min(active_markers);
            for m in tile_start..tile_end {
                let probs = target_probs.probs_for_marker(m);
                let recomb_rate = p_recomb.get(m).copied().unwrap_or(0.0);
                let uniform = target_probs.is_uniform_marker(m);

                let col = ref_columns[m];
                let hap_to_seq = col.hap_to_seq();
                let hap_ptr = hap_to_seq.as_ptr();
                if hap_ptr != last_hap_ptr {
                    fill_state_patterns_seqcoded(
                        hap_to_seq,
                        &state_hap_idx,
                        &mut ws.state_patterns,
                    );
                    last_hap_ptr = hap_ptr;
                }
                let seq_alleles = col.seq_alleles();

                if uniform {
                    fwd_sum = transition_only_forward_update(
                        &mut ws.fwd[..active_states],
                        fwd_sum,
                        recomb_rate,
                        total_ref_haps.max(1),
                    );
                } else {
                    let _ = fill_pattern_emissions(
                        seq_alleles,
                        probs,
                        current_error,
                        &mut ws.emission_by_allele,
                        &mut ws.pattern_emissions,
                    );
                    for i in 0..active_states {
                        let pid = ws.state_patterns[i] as usize;
                        ws.emissions[i] = ws.pattern_emissions.get(pid).copied().unwrap_or(1.0);
                    }

                    if m == 0 && state_priors.is_some() {
                        if recomb_rate > 0.0 {
                            fwd_sum = WeightedHmmUpdater::fwd_update_weighted(
                                &mut ws.fwd,
                                fwd_sum,
                                recomb_rate,
                                total_ref_haps.max(1),
                                &ws.weights,
                                &ws.emissions,
                                active_states,
                            );
                        } else {
                            for i in 0..active_states {
                                ws.fwd[i] *= ws.emissions[i];
                            }
                            fwd_sum = ws.fwd[..active_states].iter().sum::<f32>().max(1e-30);
                        }
                    } else {
                        fwd_sum = WeightedHmmUpdater::fwd_update_weighted(
                            &mut ws.fwd,
                            fwd_sum,
                            recomb_rate,
                            total_ref_haps.max(1),
                            &ws.weights,
                            &ws.emissions,
                            active_states,
                        );
                    }
                }

                if fwd_sum <= 0.0 {
                    fwd_sum = 1e-30;
                }
                ws.fwd_scales[m] = if uniform { 1.0 } else { fwd_sum };
                let inv = 1.0 / fwd_sum;
                for i in 0..active_states {
                    ws.fwd[i] *= inv;
                }
                fwd_sum = 1.0;
                let start = m * active_states;
                ws.fwd_history[start..start + active_states]
                    .copy_from_slice(&ws.fwd[..active_states]);
            }
            tile_start = tile_end;
        }

        let mut posteriors: Vec<AllelePosteriors> = Vec::new();
        if pass == 1 {
            posteriors.reserve(n_markers);
            posteriors.resize_with(n_markers, || AllelePosteriors::Biallelic(0.0));
        }

        ws.bwd.fill(1.0);
        let mut prior_state_post: Option<Vec<f32>> = None;

        let mut mismatch_sum = 0.0f64;
        let mut mismatch_markers = 0.0f64;

        let mut last_hap_ptr: *const u16 = std::ptr::null();
        let mut tile_end = active_markers;
        while tile_end > 0 {
            let tile_start = tile_end.saturating_sub(TILE_SIZE);
            for m_rev in (tile_start..tile_end).rev() {
                let probs = target_probs.probs_for_marker(m_rev);
                let recomb_rate = p_recomb.get(m_rev).copied().unwrap_or(0.0);
                let uniform = target_probs.is_uniform_marker(m_rev);

                let col = ref_columns[m_rev];
                let hap_to_seq = col.hap_to_seq();
                let hap_ptr = hap_to_seq.as_ptr();
                if hap_ptr != last_hap_ptr {
                    fill_state_patterns_seqcoded(
                        hap_to_seq,
                        &state_hap_idx,
                        &mut ws.state_patterns,
                    );
                    last_hap_ptr = hap_ptr;
                }
                let seq_alleles = col.seq_alleles();

                let start = m_rev * active_states;
                let fwd_slice = &ws.fwd_history[start..start + active_states];

            if pass == 1 {
                ws.allele_probs.clear();
                let n_alleles = if !probs.is_empty() {
                    probs.len()
                } else if m_rev < ref_allele_freqs.len() {
                    ref_allele_freqs.n_alleles(m_rev)
                } else {
                    0
                };
                if n_alleles > 0 {
                    ws.allele_probs.resize(n_alleles, 0.0f32);
                    let mut total = 0.0f32;
                    for i in 0..active_states {
                        let pid = ws.state_patterns[i] as usize;
                        let ref_allele = *seq_alleles.get(pid).unwrap_or(&255);
                        if ref_allele == 255 {
                            continue;
                        }
                        let state_prob = fwd_slice[i] * ws.bwd[i];
                        total += state_prob;
                        let idx = ref_allele as usize;
                        if idx < ws.allele_probs.len() {
                            ws.allele_probs[idx] += state_prob;
                        }
                    }
                    if total > 0.0 {
                        for p in ws.allele_probs.iter_mut() {
                            *p /= total;
                        }
                    } else if let Some(freqs) = ref_allele_freqs.get(m_rev) {
                        let mut sum = 0.0f32;
                        for (i, p) in ws.allele_probs.iter_mut().enumerate() {
                            let f = freqs.get(i).copied().unwrap_or(0.0).max(0.0);
                            *p = f;
                            sum += f;
                        }
                        if sum > 0.0 {
                            let inv = 1.0 / sum;
                            for p in ws.allele_probs.iter_mut() {
                                *p *= inv;
                            }
                        } else {
                            let uniform = 1.0 / ws.allele_probs.len().max(1) as f32;
                            for p in ws.allele_probs.iter_mut() {
                                *p = uniform;
                            }
                        }
                    } else {
                        let uniform = 1.0 / ws.allele_probs.len().max(1) as f32;
                        for p in ws.allele_probs.iter_mut() {
                            *p = uniform;
                        }
                    }
                    if ws.allele_probs.len() == 2 {
                        posteriors[m_rev] = AllelePosteriors::Biallelic(ws.allele_probs[1]);
                    } else {
                        let mut out = Vec::with_capacity(ws.allele_probs.len());
                        out.extend_from_slice(&ws.allele_probs);
                        posteriors[m_rev] = AllelePosteriors::Multiallelic(out);
                    }
                } else {
                    posteriors[m_rev] = AllelePosteriors::Biallelic(0.0);
                }

                if prior_marker_idx == Some(m_rev) {
                    let mut state_post = vec![0.0f32; active_states];
                    let mut total = 0.0f32;
                    for i in 0..active_states {
                        let v = fwd_slice[i] * ws.bwd[i];
                        state_post[i] = v;
                        total += v;
                    }
                    if total > 0.0 {
                        let inv = 1.0 / total;
                        for v in state_post.iter_mut() {
                            *v *= inv;
                        }
                    }
                    prior_state_post = Some(state_post);
                }
            }

            if pass == 0 && !uniform && !probs.is_empty() && probs.len() > 1 {
                let mismatch_prob = fill_pattern_emissions(
                    seq_alleles,
                    probs,
                    current_error,
                    &mut ws.emission_by_allele,
                    &mut ws.pattern_emissions,
                );
                let mut total_gamma = 0.0f64;
                let mut mismatch_expect = 0.0f64;
                for i in 0..active_states {
                    let gamma = (fwd_slice[i] * ws.bwd[i]) as f64;
                    let pid = ws.state_patterns[i] as usize;
                    let ref_allele = *seq_alleles.get(pid).unwrap_or(&255);
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

                if uniform {
                    transition_only_backward_update(
                        &mut ws.bwd[..active_states],
                        recomb_rate,
                        total_ref_haps.max(1),
                    );
                } else {
                    let _ = fill_pattern_emissions(
                        seq_alleles,
                        probs,
                        current_error,
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
                    let scale = (1.0 - recomb_rate) / c_t;
                    let shift = if total_ref_haps > 0 {
                        (recomb_rate / total_ref_haps as f32) * (emit_beta_sum / c_t)
                    } else {
                        0.0
                    };
                    for i in 0..active_states {
                        ws.bwd[i] = scale * ws.emissions[i] * ws.bwd[i] + shift;
                    }
                }
            }
            tile_end = tile_start;
        }

        if pass == 0 {
            if mismatch_markers > 0.0 {
                let numerator = mismatch_sum + (prior_alpha - 1.0) as f64;
                let denom = mismatch_markers + prior_denom as f64;
                let e_map = (numerator / denom).clamp(1e-8, 0.5 - 1e-8) as f32;
                current_error = e_map;
            }
        } else {
            final_posteriors = posteriors;
            final_prior_state_post = prior_state_post;
        }
    }

    (final_posteriors, final_prior_state_post)
}

fn run_impute_hmm_dict<Space>(
    state_haps: &[GlobalId],
    ref_columns: &[DictColRef<'_>],
    target_probs: &TargetAlleleProbs,
    p_recomb: &[f32],
    error_rate: f32,
    prior_marker_idx: Option<usize>,
    state_priors: Option<&[f32]>,
    total_ref_haps: usize,
    ref_allele_freqs: &RefAlleleFreqs<'_, Space>,
    ws: &mut ImputeWorkspace,
) -> (Vec<AllelePosteriors>, Option<Vec<f32>>) {
    let n_states = state_haps.len();
    let n_markers = target_probs.n_markers();
    ws.resize(n_states, n_markers);
    let active_states = ws.active_states();
    let active_markers = ws.active_markers();
    let state_hap_idx: Vec<HapIdx> = state_haps
        .iter()
        .map(|h| HapIdx::new(h.as_u32()))
        .collect();
    if active_states > 0 {
        ws.weights.fill(1.0);
    }

    let e_ls = crate::model::parameters::ModelParams::li_stephens_p_mismatch(
        total_ref_haps.max(2),
    );
    let prior_alpha = 1.0 + e_ls;
    let prior_beta = 1.0 + (1.0 - e_ls);
    let prior_denom = (prior_alpha + prior_beta - 2.0).max(0.0);

    let mut final_posteriors: Vec<AllelePosteriors> = Vec::new();
    let mut final_prior_state_post: Option<Vec<f32>> = None;
    let mut current_error = error_rate;

    for pass in 0..2 {
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

        const TILE_SIZE: usize = 256;
        let mut last_dict_ptr: *const DictionaryColumn = std::ptr::null();
        let mut tile_start = 0usize;
        while tile_start < active_markers {
            let tile_end = (tile_start + TILE_SIZE).min(active_markers);
            for m in tile_start..tile_end {
                let probs = target_probs.probs_for_marker(m);
                let recomb_rate = p_recomb.get(m).copied().unwrap_or(0.0);
                let uniform = target_probs.is_uniform_marker(m);

                let col = &ref_columns[m];
                let dict_ptr = col.col as *const DictionaryColumn;
                if dict_ptr != last_dict_ptr {
                    fill_state_patterns_dict(col.col, &state_hap_idx, &mut ws.state_patterns);
                    last_dict_ptr = dict_ptr;
                }

                let n_patterns = col.col.n_patterns();
                if ws.dict_pattern_alleles.len() < n_patterns {
                    ws.dict_pattern_alleles.resize(n_patterns, 0);
                }
                for pattern_idx in 0..n_patterns {
                    ws.dict_pattern_alleles[pattern_idx] =
                        col.col.pattern_allele(col.offset, pattern_idx);
                }

                if uniform {
                    fwd_sum = transition_only_forward_update(
                        &mut ws.fwd[..active_states],
                        fwd_sum,
                        recomb_rate,
                        total_ref_haps.max(1),
                    );
                } else {
                    let _ = fill_pattern_emissions(
                        &ws.dict_pattern_alleles[..n_patterns],
                        probs,
                        current_error,
                        &mut ws.emission_by_allele,
                        &mut ws.pattern_emissions,
                    );
                    for i in 0..active_states {
                        let pid = ws.state_patterns[i] as usize;
                        ws.emissions[i] = ws.pattern_emissions.get(pid).copied().unwrap_or(1.0);
                    }

                    if m == 0 && state_priors.is_some() {
                        if recomb_rate > 0.0 {
                            fwd_sum = WeightedHmmUpdater::fwd_update_weighted(
                                &mut ws.fwd,
                                fwd_sum,
                                recomb_rate,
                                total_ref_haps.max(1),
                                &ws.weights,
                                &ws.emissions,
                                active_states,
                            );
                        } else {
                            for i in 0..active_states {
                                ws.fwd[i] *= ws.emissions[i];
                            }
                            fwd_sum = ws.fwd[..active_states].iter().sum::<f32>().max(1e-30);
                        }
                    } else {
                        fwd_sum = WeightedHmmUpdater::fwd_update_weighted(
                            &mut ws.fwd,
                            fwd_sum,
                            recomb_rate,
                            total_ref_haps.max(1),
                            &ws.weights,
                            &ws.emissions,
                            active_states,
                        );
                    }
                }

                if fwd_sum <= 0.0 {
                    fwd_sum = 1e-30;
                }
                ws.fwd_scales[m] = if uniform { 1.0 } else { fwd_sum };
                let inv = 1.0 / fwd_sum;
                for i in 0..active_states {
                    ws.fwd[i] *= inv;
                }
                fwd_sum = 1.0;
                let start = m * active_states;
                ws.fwd_history[start..start + active_states]
                    .copy_from_slice(&ws.fwd[..active_states]);
            }
            tile_start = tile_end;
        }

        let mut posteriors: Vec<AllelePosteriors> = Vec::new();
        if pass == 1 {
            posteriors.reserve(n_markers);
            posteriors.resize_with(n_markers, || AllelePosteriors::Biallelic(0.0));
        }

        ws.bwd.fill(1.0);
        let mut prior_state_post: Option<Vec<f32>> = None;

        let mut mismatch_sum = 0.0f64;
        let mut mismatch_markers = 0.0f64;

        let mut last_dict_ptr: *const DictionaryColumn = std::ptr::null();
        let mut tile_end = active_markers;
        while tile_end > 0 {
            let tile_start = tile_end.saturating_sub(TILE_SIZE);
            for m_rev in (tile_start..tile_end).rev() {
                let probs = target_probs.probs_for_marker(m_rev);
                let recomb_rate = p_recomb.get(m_rev).copied().unwrap_or(0.0);
                let uniform = target_probs.is_uniform_marker(m_rev);

                let col = &ref_columns[m_rev];
                let dict_ptr = col.col as *const DictionaryColumn;
                if dict_ptr != last_dict_ptr {
                    fill_state_patterns_dict(col.col, &state_hap_idx, &mut ws.state_patterns);
                    last_dict_ptr = dict_ptr;
                }
                let n_patterns = col.col.n_patterns();
                if ws.dict_pattern_alleles.len() < n_patterns {
                    ws.dict_pattern_alleles.resize(n_patterns, 0);
                }
                for pattern_idx in 0..n_patterns {
                    ws.dict_pattern_alleles[pattern_idx] =
                        col.col.pattern_allele(col.offset, pattern_idx);
                }

                let start = m_rev * active_states;
                let fwd_slice = &ws.fwd_history[start..start + active_states];

                if pass == 1 {
                    ws.allele_probs.clear();
                    let n_alleles = if !probs.is_empty() {
                        probs.len()
                    } else if m_rev < ref_allele_freqs.len() {
                        ref_allele_freqs.n_alleles(m_rev)
                    } else {
                        0
                    };
                    if n_alleles > 0 {
                        ws.allele_probs.resize(n_alleles, 0.0f32);
                        let mut total = 0.0f32;
                        for i in 0..active_states {
                            let pid = ws.state_patterns[i] as usize;
                            let ref_allele =
                                *ws.dict_pattern_alleles.get(pid).unwrap_or(&255);
                            if ref_allele == 255 {
                                continue;
                            }
                            let state_prob = fwd_slice[i] * ws.bwd[i];
                            total += state_prob;
                            let idx = ref_allele as usize;
                            if idx < ws.allele_probs.len() {
                                ws.allele_probs[idx] += state_prob;
                            }
                        }
                        if total > 0.0 {
                            for p in ws.allele_probs.iter_mut() {
                                *p /= total;
                            }
                        } else if let Some(freqs) = ref_allele_freqs.get(m_rev) {
                            let mut sum = 0.0f32;
                            for (i, p) in ws.allele_probs.iter_mut().enumerate() {
                                let f = freqs.get(i).copied().unwrap_or(0.0).max(0.0);
                                *p = f;
                                sum += f;
                            }
                            if sum > 0.0 {
                                let inv = 1.0 / sum;
                                for p in ws.allele_probs.iter_mut() {
                                    *p *= inv;
                                }
                            } else {
                                let uniform = 1.0 / ws.allele_probs.len().max(1) as f32;
                                for p in ws.allele_probs.iter_mut() {
                                    *p = uniform;
                                }
                            }
                        } else {
                            let uniform = 1.0 / ws.allele_probs.len().max(1) as f32;
                            for p in ws.allele_probs.iter_mut() {
                                *p = uniform;
                            }
                        }
                        if ws.allele_probs.len() == 2 {
                            posteriors[m_rev] = AllelePosteriors::Biallelic(ws.allele_probs[1]);
                        } else {
                            let mut out = Vec::with_capacity(ws.allele_probs.len());
                            out.extend_from_slice(&ws.allele_probs);
                            posteriors[m_rev] = AllelePosteriors::Multiallelic(out);
                        }
                    } else {
                        posteriors[m_rev] = AllelePosteriors::Biallelic(0.0);
                    }

                    if prior_marker_idx == Some(m_rev) {
                        let mut state_post = vec![0.0f32; active_states];
                        let mut total = 0.0f32;
                        for i in 0..active_states {
                            let v = fwd_slice[i] * ws.bwd[i];
                            state_post[i] = v;
                            total += v;
                        }
                        if total > 0.0 {
                            let inv = 1.0 / total;
                            for v in state_post.iter_mut() {
                                *v *= inv;
                            }
                        }
                        prior_state_post = Some(state_post);
                    }
                }

                if pass == 0 && !uniform && !probs.is_empty() && probs.len() > 1 {
                    let mismatch_prob = fill_pattern_emissions(
                        &ws.dict_pattern_alleles[..n_patterns],
                        probs,
                        current_error,
                        &mut ws.emission_by_allele,
                        &mut ws.pattern_emissions,
                    );
                    let mut total_gamma = 0.0f64;
                    let mut mismatch_expect = 0.0f64;
                    for i in 0..active_states {
                        let gamma = (fwd_slice[i] * ws.bwd[i]) as f64;
                        let pid = ws.state_patterns[i] as usize;
                        let ref_allele = *ws.dict_pattern_alleles.get(pid).unwrap_or(&255);
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

                if uniform {
                    transition_only_backward_update(
                        &mut ws.bwd[..active_states],
                        recomb_rate,
                        total_ref_haps.max(1),
                    );
                } else {
                    let _ = fill_pattern_emissions(
                        &ws.dict_pattern_alleles[..n_patterns],
                        probs,
                        current_error,
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
                    let scale = (1.0 - recomb_rate) / c_t;
                    let shift = if total_ref_haps > 0 {
                        (recomb_rate / total_ref_haps as f32) * (emit_beta_sum / c_t)
                    } else {
                        0.0
                    };
                    for i in 0..active_states {
                        ws.bwd[i] = scale * ws.emissions[i] * ws.bwd[i] + shift;
                    }
                }
            }
            tile_end = tile_start;
        }

        if pass == 0 {
            if mismatch_markers > 0.0 {
                let numerator = mismatch_sum + (prior_alpha - 1.0) as f64;
                let denom = mismatch_markers + prior_denom as f64;
                let e_map = (numerator / denom).clamp(1e-8, 0.5 - 1e-8) as f32;
                current_error = e_map;
            }
        } else {
            final_posteriors = posteriors;
            final_prior_state_post = prior_state_post;
        }
    }

    (final_posteriors, final_prior_state_post)
}

/// Run forward-backward HMM and emit allele posteriors.
///
/// Returns (posteriors, optional state posterior at prior marker).
pub fn run_impute_hmm<Space>(
    state_haps: &[GlobalId],
    ref_columns: &[GenotypeColumn],
    target_probs: &TargetAlleleProbs,
    p_recomb: &[f32],
    error_rate: f32,
    prior_marker_idx: Option<usize>,
    state_priors: Option<&[f32]>,
    total_ref_haps: usize,
    ref_allele_freqs: &RefAlleleFreqs<'_, Space>,
    ws: &mut ImputeWorkspace,
) -> (Vec<AllelePosteriors>, Option<Vec<f32>>) {
    if ref_columns.is_empty() {
        return run_impute_hmm_impl(
            state_haps,
            ref_columns,
            target_probs,
            p_recomb,
            error_rate,
            prior_marker_idx,
            state_priors,
            total_ref_haps,
            ref_allele_freqs,
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
            total_ref_haps,
            ref_allele_freqs,
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
            total_ref_haps,
            ref_allele_freqs,
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
            total_ref_haps,
            ref_allele_freqs,
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
            total_ref_haps,
            ref_allele_freqs,
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
        total_ref_haps,
        ref_allele_freqs,
        ws,
    )
}

/// Convert dense state posteriors into sparse global priors (sorted by GlobalId).
pub fn state_posteriors_to_priors(
    state_haps: &[GlobalId],
    state_post: &[f32],
    threshold: f32,
) -> Vec<(GlobalId, f32)> {
    let mut out: Vec<(GlobalId, f32)> = state_haps
        .iter()
        .zip(state_post.iter())
        .filter(|(_, p)| p.is_finite() && **p > threshold)
        .map(|(h, &p)| (*h, p))
        .collect();
    out.sort_unstable_by_key(|(h, _)| h.as_u32());
    out
}
