//! HMM kernel for imputation using explicit haplotype states.
//!
//! This implements a Li-Stephens forward-backward pass over a selected set of
//! reference haplotypes (state set). Emissions are computed using per-haplotype
//! allele probabilities from the target, and reference alleles are read on demand.

use crate::data::storage::GenotypeColumn;
use crate::data::HapIdx;
use crate::model::hmm::HmmUpdater;
use crate::model::weighted_kernel::WeightedHmmUpdater;
use crate::model::types::GlobalId;
use crate::pipelines::imputation::AllelePosteriors;

/// Per-marker allele probability distributions for a single target haplotype.
pub struct TargetAlleleProbs {
    offsets: Vec<usize>,
    probs: Vec<f32>,
}

impl TargetAlleleProbs {
    pub fn new(offsets: Vec<usize>, probs: Vec<f32>) -> Self {
        Self { offsets, probs }
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
}

/// Workspace for per-haplotype imputation HMM.
pub struct ImputeWorkspace {
    pub fwd: Vec<f32>,
    pub bwd: Vec<f32>,
    pub emissions: Vec<f32>,
    pub fwd_history: Vec<f32>,
    pub weights: Vec<f32>,
}

impl ImputeWorkspace {
    pub fn new(n_states: usize, n_markers: usize) -> Self {
        Self {
            fwd: vec![0.0; n_states],
            bwd: vec![1.0; n_states],
            emissions: vec![1.0; n_states],
            fwd_history: vec![0.0; n_states * n_markers],
            weights: vec![1.0; n_states],
        }
    }

    pub fn resize(&mut self, n_states: usize, n_markers: usize) {
        if self.fwd.len() != n_states {
            self.fwd.resize(n_states, 0.0);
            self.bwd.resize(n_states, 1.0);
            self.emissions.resize(n_states, 1.0);
            self.weights.resize(n_states, 1.0);
        }
        if self.weights.len() != n_states {
            self.weights.resize(n_states, 1.0);
        }
        let want = n_states.saturating_mul(n_markers);
        if self.fwd_history.len() != want {
            self.fwd_history.resize(want, 0.0);
        }
    }
}

#[inline]
fn emission_prob_soft(ref_allele: u8, target_probs: &[f32], error_rate: f32) -> f32 {
    if target_probs.is_empty() {
        return 1.0;
    }
    if ref_allele == 255 {
        return 1.0;
    }
    let n_alleles = target_probs.len();
    if n_alleles == 0 {
        return 1.0;
    }
    let mismatch_prob = if n_alleles > 1 {
        error_rate / (n_alleles as f32 - 1.0)
    } else {
        error_rate
    };
    let match_prob = 1.0 - error_rate;
    let p_match = target_probs
        .get(ref_allele as usize)
        .copied()
        .unwrap_or(0.0);
    mismatch_prob + (match_prob - mismatch_prob) * p_match
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

/// Run forward-backward HMM and emit allele posteriors.
///
/// Returns (posteriors, optional state posterior at prior marker).
pub fn run_impute_hmm(
    state_haps: &[GlobalId],
    ref_columns: &[GenotypeColumn],
    target_probs: &TargetAlleleProbs,
    p_recomb: &[f32],
    error_rate: f32,
    prior_marker_idx: Option<usize>,
    state_priors: Option<&[f32]>,
    ws: &mut ImputeWorkspace,
) -> (Vec<AllelePosteriors>, Option<Vec<f32>>) {
    let n_states = state_haps.len();
    let n_markers = target_probs.n_markers();
    ws.resize(n_states, n_markers);
    if n_states > 0 {
        ws.weights.fill(1.0);
    }

    let mut fwd_sum: f32;
    if let Some(priors) = state_priors {
        let len = priors.len().min(n_states);
        ws.fwd[..len].copy_from_slice(&priors[..len]);
        if len < n_states {
            ws.fwd[len..].fill(0.0);
        }
        normalize_probs(&mut ws.fwd);
        fwd_sum = ws.fwd.iter().sum::<f32>().max(1e-30);
    } else {
        let uniform = 1.0 / n_states.max(1) as f32;
        ws.fwd.fill(uniform);
        fwd_sum = 1.0;
    }

    for m in 0..n_markers {
        let probs = target_probs.probs_for_marker(m);
        let recomb_rate = p_recomb.get(m).copied().unwrap_or(0.0);

        for (i, hap) in state_haps.iter().enumerate() {
            let ref_allele = ref_columns[m].get(HapIdx::new(hap.as_u32()));
            ws.emissions[i] = emission_prob_soft(ref_allele, probs, error_rate);
        }

        if m == 0 && state_priors.is_some() {
            if recomb_rate > 0.0 {
                fwd_sum = WeightedHmmUpdater::fwd_update_weighted(
                    &mut ws.fwd,
                    fwd_sum,
                    recomb_rate,
                    n_states.max(1),
                    &ws.weights,
                    &ws.emissions,
                    n_states,
                );
            } else {
                for i in 0..n_states {
                    ws.fwd[i] *= ws.emissions[i];
                }
                fwd_sum = ws.fwd.iter().sum::<f32>().max(1e-30);
            }
        } else {
            fwd_sum = WeightedHmmUpdater::fwd_update_weighted(
                &mut ws.fwd,
                fwd_sum,
                recomb_rate,
                n_states.max(1),
                &ws.weights,
                &ws.emissions,
                n_states,
            );
        }

        if fwd_sum <= 0.0 {
            fwd_sum = 1e-30;
        }
        let start = m * n_states;
        ws.fwd_history[start..start + n_states].copy_from_slice(&ws.fwd);
    }

    let mut posteriors: Vec<AllelePosteriors> = Vec::with_capacity(n_markers);
    posteriors.resize_with(n_markers, || AllelePosteriors::Biallelic(0.0));

    ws.bwd.fill(1.0);
    let mut prior_state_post: Option<Vec<f32>> = None;

    for m_rev in (0..n_markers).rev() {
        let probs = target_probs.probs_for_marker(m_rev);
        let recomb_rate = p_recomb.get(m_rev).copied().unwrap_or(0.0);

        for (i, hap) in state_haps.iter().enumerate() {
            let ref_allele = ref_columns[m_rev].get(HapIdx::new(hap.as_u32()));
            ws.emissions[i] = emission_prob_soft(ref_allele, probs, error_rate);
        }

        let mut constant_term = 0.0f32;
        for i in 0..n_states {
            constant_term += ws.bwd[i] * ws.emissions[i];
        }
        HmmUpdater::bwd_update_constant(
            &mut ws.bwd,
            recomb_rate,
            &ws.emissions,
            constant_term.max(1e-30),
            n_states,
        );

        let start = m_rev * n_states;
        let fwd_slice = &ws.fwd_history[start..start + n_states];

        let mut allele_probs: Vec<f32> = Vec::new();
        let n_alleles = probs.len();
        if n_alleles > 0 {
            allele_probs.resize(n_alleles, 0.0f32);
            let mut total = 0.0f32;
            for (i, hap) in state_haps.iter().enumerate() {
                let ref_allele = ref_columns[m_rev].get(HapIdx::new(hap.as_u32()));
                if ref_allele == 255 {
                    continue;
                }
                let state_prob = fwd_slice[i] * ws.bwd[i];
                total += state_prob;
                let idx = ref_allele as usize;
                if idx < allele_probs.len() {
                    allele_probs[idx] += state_prob;
                }
            }
            if total > 0.0 {
                for p in allele_probs.iter_mut() {
                    *p /= total;
                }
            }
            if allele_probs.len() == 2 {
                posteriors[m_rev] = AllelePosteriors::Biallelic(allele_probs[1]);
            } else {
                posteriors[m_rev] = AllelePosteriors::Multiallelic(allele_probs);
            }
        } else {
            posteriors[m_rev] = AllelePosteriors::Biallelic(0.0);
        }

        if prior_marker_idx == Some(m_rev) {
            let mut state_post = vec![0.0f32; n_states];
            let mut total = 0.0f32;
            for i in 0..n_states {
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

    (posteriors, prior_state_post)
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
