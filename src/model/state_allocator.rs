//! Lagrangian Marginal Stickiness allocator for prescan state selection.
//!
//! This module implements a *selection-layer* optimization aligned with the
//! Li–Stephens transition physics. It is not the exact HMM marginal
//! likelihood; it is a surrogate objective that is fast, stable, and preserves
//! identity continuity across windows under a hard RAM budget.
//!
//! Surrogate objective (math derivation, selection-layer):
//!
//! Let S[h,w] be a log-score for haplotype h in window w (from PBWT/Shannon).
//! We interpret exp(S[h,w]) as a nonnegative *evidence weight* (not a calibrated
//! probability unless S is a true emission LLR).
//!
//! Per window, define a log-sum evidence:
//!   Z_w = 1 + sum_h exp(S[h,w])
//!   logZ_w = log(Z_w)
//!
//! This induces diminishing returns: adding another haplotype to a window that
//! is already well explained yields less incremental gain.
//!
//! The marginal gain of activating haplotype h in window w is:
//!   u_{h,w} = log( Z_w + exp(S[h,w]) ) - logZ_w - mu
//! where mu is the Lagrangian price per window-slot (RAM constraint).
//!
//! We maximize a continuity-aware sum of these marginal gains across windows:
//!   sum_w u_{h,w} y_w + sum_w b_w * y_w * y_{w+1}
//! where y_w ∈ {0,1} indicates whether haplotype h is active in window w.
//!
//! Continuity (Li–Stephens physics, 2-state surrogate):
//!
//! Let r_w = p_recomb(d_w) be the HMM’s recombination probability at boundary w.
//! Let a_w = 1 - r_w be the no-switch weight.
//! For a donor pool size n_pool, the Li–Stephens copying chain has:
//!   p11 = P(ON->ON) = a_w + (1-a_w)/n_pool
//!   p10 = P(ON->OFF) = (1-a_w)*(n_pool-1)/n_pool
//!   p01 = P(OFF->ON) = (1-a_w)/n_pool
//!   p00 = P(OFF->OFF) = 1 - p01
//!
//! We score schedules using transition log-odds relative to OFF->OFF:
//!   t11 = log(p11/p00), t10 = log(p10/p00), t01 = log(p01/p00)
//! This is the exact 2-state (ON/OFF) collapse of the Li–Stephens chain for a
//! single haplotype vs “any other donor”, and is the continuity prior used by
//! the DP below. This preserves the HMM’s stay/switch physics in the selection
//! layer without claiming to optimize the exact HMM marginal likelihood.
//!
//! Optimization:
//!
//! For each haplotype, we solve a 2-state DP over windows (ON/OFF) using the
//! u_{h,w} and (t11,t10,t01) terms. This yields the optimal disjoint intervals
//! for that haplotype under the surrogate objective. The outer loop performs
//! greedy coordinate ascent: pick the haplotype with highest DP gain, activate
//! its intervals, update logZ_w, and repeat until the global slot budget is
//! exhausted or gains become nonpositive.
//!
//! A coarse grid search over mu is used to hit the budget without assuming
//! monotonicity under the greedy outer loop.

use crate::model::parameters::ModelParams;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

const NEG_INF: f32 = -1.0e30;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WindowSpan {
    start: u32,
    end_exclusive: u32,
}

impl WindowSpan {
    pub fn new(start: u32, end_exclusive: u32) -> Self {
        assert!(
            start < end_exclusive,
            "invalid WindowSpan: start {} must be < end_exclusive {}",
            start,
            end_exclusive
        );
        Self {
            start,
            end_exclusive,
        }
    }

    pub fn full(num_windows: usize) -> Self {
        let end = num_windows as u32;
        assert!(end > 0, "invalid full WindowSpan for zero windows");
        Self {
            start: 0,
            end_exclusive: end,
        }
    }

    #[inline]
    pub fn contains(self, window_idx: usize) -> bool {
        let idx = window_idx as u32;
        idx >= self.start && idx < self.end_exclusive
    }

    #[inline]
    pub fn len(self) -> u32 {
        self.end_exclusive - self.start
    }

    #[inline]
    pub fn is_full(self, num_windows: usize) -> bool {
        self.start == 0 && self.end_exclusive == num_windows as u32
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DonorPoolSize(usize);

impl DonorPoolSize {
    fn new(raw: usize) -> Self {
        Self(raw.max(2))
    }

    fn min(self, other: Self) -> Self {
        Self(self.0.min(other.0))
    }

    fn as_f32(self) -> f32 {
        self.0 as f32
    }
}

#[inline]
fn logaddexp(a: f32, b: f32) -> f32 {
    if a <= NEG_INF {
        return b;
    }
    if b <= NEG_INF {
        return a;
    }
    let m = a.max(b);
    let d = (a - b).abs();
    if d > 12.0 {
        m
    } else {
        m + (-(d)).exp().ln_1p()
    }
}

#[derive(Default)]
struct DpScratch {
    dp0: Vec<f32>,
    dp1: Vec<f32>,
    prev0: Vec<u8>,
    prev1: Vec<u8>,
    active: Vec<bool>,
}

impl DpScratch {
    fn ensure_len(&mut self, w: usize) {
        if self.dp0.len() < w {
            self.dp0.resize(w, 0.0);
        }
        if self.dp1.len() < w {
            self.dp1.resize(w, NEG_INF);
        }
        if self.prev0.len() < w {
            self.prev0.resize(w, 0);
        }
        if self.prev1.len() < w {
            self.prev1.resize(w, 0);
        }
        if self.active.len() < w {
            self.active.resize(w, false);
        }
    }
}

/// Allocation result for a single target haplotype:
/// intervals per selected haplotype (reference panel indices).
#[derive(Clone, Debug)]
pub struct WindowAllocation {
    pub intervals_by_hap: Vec<(usize, Vec<WindowSpan>)>,
}

/// Compute continuity transition terms per boundary.
///
/// We must be consistent with the HMM parameterization. The recombination
/// probability used by the HMM is:
///
///     r_w = p_recomb(d_w)
///     a_w = 1 - r_w
///
/// Then, with boundary-specific donor pool size n_pool[w]:
///     p_stay = a_w + (1-a_w)/n_pool
///     p_switch_diff = (1-a_w) * (n_pool-1)/n_pool
///
/// and
///     t11 = log(p11/p00), t10 = log(p10/p00), t01 = log(p01/p00)
///
/// These are the exact 2-state (ON/OFF) transition log-odds relative to OFF->OFF
/// under a lumped ON=“hap h” / OFF=“not h” surrogate chain. This preserves the
/// Li–Stephens stay/switch physics in the selection layer without claiming to be
/// the full HMM marginal likelihood.
fn continuity_terms(
    boundary_cm: &[f64],
    params: &ModelParams,
    n_pool_by_boundary: &[DonorPoolSize],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut t11 = Vec::with_capacity(boundary_cm.len());
    let mut t10 = Vec::with_capacity(boundary_cm.len());
    let mut t01 = Vec::with_capacity(boundary_cm.len());
    for (i, &dist_cm) in boundary_cm.iter().enumerate() {
        // `n_pool_f` is boundary-local effective donor diversity.
        // When LD collapses donor uncertainty, N_eff decreases and the
        // ON/OFF transition odds become less switch-averse for equivalent evidence.
        let n_pool_f = n_pool_by_boundary
            .get(i)
            .copied()
            .unwrap_or(DonorPoolSize::new(2))
            .as_f32();
        let r_w = params.p_recomb(dist_cm);
        let a_w = (1.0 - r_w).max(0.0).min(1.0);
        let p11 = a_w + (1.0 - a_w) / n_pool_f;
        let p10 = (1.0 - a_w) * (n_pool_f - 1.0) / n_pool_f;
        let p01 = (1.0 - a_w) / n_pool_f;
        let p00 = 1.0 - p01;
        let t11_v = if p11 > 0.0 && p00 > 0.0 {
            (p11 / p00).ln()
        } else {
            NEG_INF
        };
        let t10_v = if p10 > 0.0 && p00 > 0.0 {
            (p10 / p00).ln()
        } else {
            NEG_INF
        };
        let t01_v = if p01 > 0.0 && p00 > 0.0 {
            (p01 / p00).ln()
        } else {
            NEG_INF
        };
        t11.push(t11_v);
        t10.push(t10_v);
        t01.push(t01_v);
    }
    (t11, t10, t01)
}

/// Run the 2-state DP for a single haplotype from sparse scores.
///
/// Inputs:
/// - scores: sparse (win, score) list for this haplotype (sorted by win).
/// - logZ: current log-evidence baseline per window.
/// - mu: per-window slot price.
/// - t11/t10/t01: 2-state transition log-odds vs OFF->OFF.
///
fn dp_intervals_sparse_scratch(
    scores: &[(usize, f32)],
    logz: &[f32],
    mu: f32,
    blocked: &[bool],
    t11: &[f32],
    t10: &[f32],
    t01: &[f32],
    scratch: &mut DpScratch,
) -> (f32, usize) {
    let w = logz.len();
    if w == 0 {
        return (0.0, 0);
    }
    scratch.ensure_len(w);
    let dp0 = &mut scratch.dp0[..w];
    let dp1 = &mut scratch.dp1[..w];
    let prev0 = &mut scratch.prev0[..w];
    let prev1 = &mut scratch.prev1[..w];
    let active = &mut scratch.active[..w];
    active.fill(false);

    let mut s_idx = 0usize;
    let mut score0 = NEG_INF;
    if !scores.is_empty() && scores[0].0 == 0 {
        score0 = scores[0].1;
        s_idx = 1;
    }
    dp0[0] = 0.0;
    let mut u0 = logaddexp(logz[0], score0) - logz[0] - mu;
    if blocked.get(0).copied().unwrap_or(false) {
        u0 = NEG_INF;
    }
    dp1[0] = u0;

    for i in 1..w {
        let mut score = NEG_INF;
        if s_idx < scores.len() && scores[s_idx].0 == i {
            score = scores[s_idx].1;
            s_idx += 1;
        }
        let mut u_w = logaddexp(logz[i], score) - logz[i] - mu;
        if blocked.get(i).copied().unwrap_or(false) {
            u_w = NEG_INF;
        }
        // OFF state: max(OFF->OFF, ON->OFF)
        let from0 = dp0[i - 1];
        let from1 = dp1[i - 1] + t10[i - 1];
        if from0 >= from1 {
            dp0[i] = dp0[i - 1];
            prev0[i] = 0;
        } else {
            dp0[i] = from1;
            prev0[i] = 1;
        }

        // ON state: max(OFF->ON, ON->ON)
        let from_off = dp0[i - 1] + t01[i - 1];
        let from_on = dp1[i - 1] + t11[i - 1];
        if from_off >= from_on {
            dp1[i] = u_w + from_off;
            prev1[i] = 0;
        } else {
            dp1[i] = u_w + from_on;
            prev1[i] = 1;
        }
    }

    let mut active_len = 0usize;
    let mut state = if dp1[w - 1] >= dp0[w - 1] { 1 } else { 0 };
    let mut i = w - 1;
    loop {
        if state == 1 {
            active[i] = true;
            active_len += 1;
            state = prev1[i] as usize;
        } else {
            state = prev0[i] as usize;
        }
        if i == 0 {
            break;
        }
        i -= 1;
    }

    let gain = dp0[w - 1].max(dp1[w - 1]);
    (gain, active_len)
}

fn active_to_intervals(active: &[bool]) -> Vec<WindowSpan> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < active.len() {
        if !active[i] {
            i += 1;
            continue;
        }
        let start = i;
        let mut end = i;
        while end + 1 < active.len() && active[end + 1] {
            end += 1;
        }
        out.push(WindowSpan::new(start as u32, (end + 1) as u32));
        i = end + 1;
    }
    out
}

/// Estimate per-window effective donor pool size from sparse donor scores.
///
/// Coalescent/Li-Stephens motivation:
/// - The ON/OFF transition collapse uses a donor-pool denominator `N`.
/// - In practice, donor evidence is highly concentrated in strong-LD regions.
/// - We estimate a local effective `N` via ESS:
///     N_eff = (sum w_i)^2 / sum(w_i^2),
///   where w_i is a positive donor weight in a window.
/// - Scores are in log-space, so we use exp(score - max_score_window) for
///   numerically stable relative weights (scale cancels in ESS).
fn window_effective_pool_sizes(scores_by_hap: &[Vec<(usize, f32)>], num_windows: usize) -> Vec<f32> {
    if num_windows == 0 {
        return Vec::new();
    }
    let mut max_score_by_window = vec![NEG_INF; num_windows];
    let mut support_by_window = vec![0usize; num_windows];
    for scores in scores_by_hap {
        for &(win, score) in scores {
            if win < num_windows && score.is_finite() {
                support_by_window[win] += 1;
                if score > max_score_by_window[win] {
                    max_score_by_window[win] = score;
                }
            }
        }
    }

    let mut sum_w = vec![0.0f64; num_windows];
    let mut sum_w2 = vec![0.0f64; num_windows];
    for scores in scores_by_hap {
        for &(win, score) in scores {
            if win >= num_windows || !score.is_finite() {
                continue;
            }
            let max_s = max_score_by_window[win];
            if !max_s.is_finite() {
                continue;
            }
            let centered = (score - max_s).clamp(-80.0, 0.0) as f64;
            let w_i = centered.exp();
            sum_w[win] += w_i;
            sum_w2[win] += w_i * w_i;
        }
    }

    let mut out = vec![1.0f32; num_windows];
    let mut win = 0usize;
    while win < num_windows {
        let s = sum_w[win];
        let s2 = sum_w2[win];
        if s > 0.0 && s2 > 0.0 {
            let neff = ((s * s) / s2) as f32;
            // Empirical-Bayes shrinkage toward observed support count when
            // score evidence is sparse to avoid overreacting to a few donors.
            let support = support_by_window[win] as f32;
            let lambda = support / (support + 8.0);
            out[win] = (lambda * neff + (1.0 - lambda) * support.max(1.0)).max(1.0);
        }
        win += 1;
    }

    // Smooth local ESS to avoid boundary-to-boundary jitter from sparse/noisy
    // score support. This keeps transition penalties stable while preserving
    // broad LD-driven variation in effective donor diversity.
    if num_windows >= 3 {
        let mut smoothed = out.clone();
        let mut i = 1usize;
        while i + 1 < num_windows {
            smoothed[i] = (0.25 * out[i - 1] + 0.5 * out[i] + 0.25 * out[i + 1]).max(1.0);
            i += 1;
        }
        out = smoothed;
    }
    out
}

/// Build boundary-specific donor-pool sizes for continuity terms.
///
/// We map per-window N_eff values to boundaries using the geometric mean
/// between adjacent windows, then clamp by hard resource limits.
fn boundary_pool_sizes_from_scores(
    scores_by_hap: &[Vec<(usize, f32)>],
    num_windows: usize,
    per_window_caps: &[usize],
    n_pool: usize,
) -> Vec<DonorPoolSize> {
    let mut out = Vec::with_capacity(num_windows.saturating_sub(1));
    if num_windows < 2 {
        return out;
    }
    let neff_by_window = window_effective_pool_sizes(scores_by_hap, num_windows);
    let panel_pool = DonorPoolSize::new(n_pool);

    let mut b = 0usize;
    while b + 1 < num_windows {
        let c0 = DonorPoolSize::new(per_window_caps.get(b).copied().unwrap_or(n_pool));
        let c1 = DonorPoolSize::new(per_window_caps.get(b + 1).copied().unwrap_or(n_pool));
        let cap_pool = c0.min(c1).min(panel_pool);

        let left = neff_by_window[b].max(1.0);
        let right = neff_by_window[b + 1].max(1.0);
        let boundary_neff = (left * right).sqrt();
        // Final shrinkage toward cap-based pool size for safety in sparse/noisy windows.
        let cap_f = cap_pool.as_f32();
        let lambda = (boundary_neff / (boundary_neff + 8.0)).clamp(0.0, 1.0);
        let blended = lambda * boundary_neff + (1.0 - lambda) * cap_f;
        let local_pool = DonorPoolSize::new(blended.round().max(2.0) as usize);
        out.push(local_pool.min(cap_pool));
        b += 1;
    }
    out
}

#[derive(Clone)]
struct AllocationState {
    active: Vec<bool>,
    len: usize,
}

fn recompute_counts_logz_used(
    selected_states: &[Option<AllocationState>],
    scores_by_hap: &[Vec<(usize, f32)>],
    num_windows: usize,
    per_window_caps: &[usize],
) -> (Vec<usize>, Vec<f32>, usize) {
    let mut counts = vec![0usize; num_windows];
    let mut z_w = vec![0.0f32; num_windows];
    let mut used = 0usize;

    let mut idx = 0usize;
    while idx < selected_states.len() {
        if let Some(state) = selected_states[idx].as_ref() {
            used += state.len;
            let mut win = 0usize;
            while win < num_windows {
                if state.active[win] && counts[win] < per_window_caps[win] {
                    counts[win] += 1;
                }
                win += 1;
            }
            for &(win, score) in &scores_by_hap[idx] {
                if win < num_windows && state.active[win] && score.is_finite() {
                    z_w[win] = logaddexp(z_w[win], score);
                }
            }
        }
        idx += 1;
    }

    (counts, z_w, used)
}

/// Allocate active haplotypes per window for a single target haplotype.
///
/// Inputs:
/// - scores_by_hap: sparse S[h,w] lists per candidate hap (log-score weights).
/// - boundary_cm: window boundary distances (len W-1).
/// - params: model params for recombination mapping.
/// - total_budget: total slots allowed across all windows (sum of per-window caps).
/// - per_window_caps: per-window max states (global, same for all target haps).
pub fn allocate_lms_sparse(
    scores_by_hap: &[Vec<(usize, f32)>],
    candidate_haps: &[usize],
    num_windows: usize,
    boundary_cm: &[f64],
    params: &ModelParams,
    n_pool: usize,
    total_budget: usize,
    per_window_caps: &[usize],
) -> WindowAllocation {
    let w = num_windows;
    if w == 0 || total_budget == 0 {
        return WindowAllocation {
            intervals_by_hap: Vec::new(),
        };
    }
    if per_window_caps.len() != w {
        return WindowAllocation {
            intervals_by_hap: Vec::new(),
        };
    }
    if !boundary_cm.is_empty() && boundary_cm.len() + 1 != w {
        return WindowAllocation {
            intervals_by_hap: Vec::new(),
        };
    }
    let n = candidate_haps.len();
    if n == 0 {
        return WindowAllocation {
            intervals_by_hap: Vec::new(),
        };
    }
    let per_window_min = per_window_caps.iter().copied().min().unwrap_or(0);
    if per_window_min >= n && total_budget >= n.saturating_mul(w) {
        let intervals_by_hap = candidate_haps
            .iter()
            .map(|&h| (h, vec![WindowSpan::full(w)]))
            .collect();
        return WindowAllocation { intervals_by_hap };
    }
    let mut counts = vec![0usize; w];
    let mut z_w = vec![0.0f32; w]; // log(1)
    let mut selected_states: Vec<Option<AllocationState>> = vec![None; n];
    let mut remaining = total_budget;

    let n_pool_by_boundary = boundary_pool_sizes_from_scores(scores_by_hap, w, per_window_caps, n_pool);
    let (t11, t10, t01) = continuity_terms(boundary_cm, params, &n_pool_by_boundary);

    // Determine mu by bracket + coarse search + local refinement.
    // Larger mu reduces activations under the same surrogate objective.
    let mut mu_low = -10.0f32;
    let mut mu_high = 10.0f32;

    // Ensure bounds bracket a feasible range.
    let mut low_iter = 0usize;
    while low_iter < 5 {
        let (used, _) =
            simulate_allocation(scores_by_hap, w, &t11, &t10, &t01, mu_low, per_window_caps);
        if used >= total_budget {
            break;
        }
        mu_low -= 5.0;
        low_iter += 1;
    }
    let mut high_iter = 0usize;
    while high_iter < 5 {
        let (used, _) =
            simulate_allocation(scores_by_hap, w, &t11, &t10, &t01, mu_high, per_window_caps);
        if used <= total_budget {
            break;
        }
        mu_high += 5.0;
        high_iter += 1;
    }

    // Coarse grid search over mu to avoid non-monotone behavior in the greedy outer loop.
    let mut mu_best = mu_high;
    let mut best_used = 0usize;
    let mut best_gain = NEG_INF;
    let mut best_k = 0usize;
    let mut found_feasible = false;
    let mut coarse_samples: Vec<(f32, usize, f32)> = Vec::with_capacity(17);
    let mut k = 0usize;
    while k < 17 {
        let t = k as f32 / 16.0;
        let mu = mu_low + t * (mu_high - mu_low);
        let (used, gain) =
            simulate_allocation(scores_by_hap, w, &t11, &t10, &t01, mu, per_window_caps);
        coarse_samples.push((mu, used, gain));
        if used <= total_budget
            && (!found_feasible
                || gain > best_gain
                || (gain == best_gain && used > best_used))
        {
            found_feasible = true;
            mu_best = mu;
            best_used = used;
            best_gain = gain;
            best_k = k;
        }
        k += 1;
    }

    if found_feasible {
        // Refine around the best coarse sample.
        let mut left = if best_k > 0 {
            coarse_samples[best_k - 1].0
        } else {
            mu_low
        };
        let mut right = if best_k + 1 < coarse_samples.len() {
            coarse_samples[best_k + 1].0
        } else {
            mu_high
        };
        let mut refine_iter = 0usize;
        while refine_iter < 3 {
            let span = right - left;
            if span <= 1e-3 {
                break;
            }
            let step = span / 8.0;
            let mut j = 0usize;
            while j < 9 {
                let mu = left + step * j as f32;
                let (used, gain) =
                    simulate_allocation(scores_by_hap, w, &t11, &t10, &t01, mu, per_window_caps);
                if used <= total_budget && (gain > best_gain || (gain == best_gain && used > best_used)) {
                    mu_best = mu;
                    best_used = used;
                    best_gain = gain;
                }
                j += 1;
            }
            left = (mu_best - step).max(mu_low);
            right = (mu_best + step).min(mu_high);
            refine_iter += 1;
        }
    }
    let mu = mu_best;

    #[derive(Clone)]
    struct HeapEntry {
        gain: f32,
        idx: usize,
        active: Vec<bool>,
        len: usize,
    }

    impl Eq for HeapEntry {}

    impl PartialEq for HeapEntry {
        fn eq(&self, other: &Self) -> bool {
            self.gain == other.gain && self.idx == other.idx
        }
    }

    impl Ord for HeapEntry {
        fn cmp(&self, other: &Self) -> Ordering {
            self.partial_cmp(other).unwrap_or(Ordering::Equal)
        }
    }

    impl PartialOrd for HeapEntry {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            self.gain
                .partial_cmp(&other.gain)
                .or_else(|| Some(self.idx.cmp(&other.idx)))
        }
    }

    // Lazy-greedy allocation using tuned mu.
    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();
    let mut dp_scratch = DpScratch::default();
    {
        let blocked: Vec<bool> = counts
            .iter()
            .enumerate()
            .map(|(w_i, &c)| c >= per_window_caps[w_i])
            .collect();
        for h in 0..n {
            let scores = &scores_by_hap[h];
            let (gain, len) = dp_intervals_sparse_scratch(
                scores,
                &z_w,
                mu,
                &blocked,
                &t11,
                &t10,
                &t01,
                &mut dp_scratch,
            );
            let active = dp_scratch.active[..w].to_vec();
            if gain > 0.0 && len > 0 && len <= remaining {
                heap.push(HeapEntry {
                    gain,
                    idx: h,
                    active,
                    len,
                });
            }
        }
    }

    while remaining > 0 {
        let Some(mut entry) = heap.pop() else { break };
        if selected_states[entry.idx].is_some() {
            continue;
        }
        let blocked: Vec<bool> = counts
            .iter()
            .enumerate()
            .map(|(w_i, &c)| c >= per_window_caps[w_i])
            .collect();
        let (gain, len) = dp_intervals_sparse_scratch(
            &scores_by_hap[entry.idx],
            &z_w,
            mu,
            &blocked,
            &t11,
            &t10,
            &t01,
            &mut dp_scratch,
        );
        let active = dp_scratch.active[..w].to_vec();
        if gain <= 0.0 || len == 0 || len > remaining {
            continue;
        }
        let next_gain = heap.peek().map(|e| e.gain).unwrap_or(NEG_INF);
        if gain >= next_gain {
            selected_states[entry.idx] = Some(AllocationState {
                active: active.clone(),
                len,
            });
            for win in 0..w {
                if active[win] && counts[win] < per_window_caps[win] {
                    counts[win] += 1;
                }
            }
            for &(win, score) in scores_by_hap[entry.idx].iter() {
                if win < w && active[win] && score.is_finite() {
                    z_w[win] = logaddexp(z_w[win], score);
                }
            }
            remaining = remaining.saturating_sub(len);
        } else {
            entry.gain = gain;
            entry.active = active;
            entry.len = len;
            heap.push(entry);
        }
    }

    // Residual-capacity pass: once Lagrangian filtering stops, continue to fill
    // positive-gain states under hard caps/budget using mu=0 (true objective).
    while remaining > 0 {
        let blocked: Vec<bool> = counts
            .iter()
            .enumerate()
            .map(|(w_i, &c)| c >= per_window_caps[w_i])
            .collect();
        let mut best_idx: Option<usize> = None;
        let mut best_gain = NEG_INF;
        let mut best_gain_per_slot = NEG_INF;
        let mut best_len = 0usize;

        let mut h = 0usize;
        while h < n {
            if selected_states[h].is_some() {
                h += 1;
                continue;
            }
            let (gain, len) = dp_intervals_sparse_scratch(
                &scores_by_hap[h],
                &z_w,
                0.0,
                &blocked,
                &t11,
                &t10,
                &t01,
                &mut dp_scratch,
            );
            if gain > 0.0 && len > 0 && len <= remaining {
                let gain_per_slot = gain / len as f32;
                if gain_per_slot > best_gain_per_slot
                    || (gain_per_slot == best_gain_per_slot
                        && (gain > best_gain || (gain == best_gain && len > best_len)))
                {
                    best_idx = Some(h);
                    best_gain = gain;
                    best_gain_per_slot = gain_per_slot;
                    best_len = len;
                }
            }
            h += 1;
        }

        let Some(chosen_idx) = best_idx else {
            break;
        };

        let blocked: Vec<bool> = counts
            .iter()
            .enumerate()
            .map(|(w_i, &c)| c >= per_window_caps[w_i])
            .collect();
        let (gain_chk, len_chk) = dp_intervals_sparse_scratch(
            &scores_by_hap[chosen_idx],
            &z_w,
            0.0,
            &blocked,
            &t11,
            &t10,
            &t01,
            &mut dp_scratch,
        );
        if gain_chk <= 0.0 || len_chk == 0 || len_chk > remaining {
            break;
        }
        let chosen_active = dp_scratch.active[..w].to_vec();

        selected_states[chosen_idx] = Some(AllocationState {
            active: chosen_active.clone(),
            len: len_chk,
        });
        for win in 0..w {
            if chosen_active[win] && counts[win] < per_window_caps[win] {
                counts[win] += 1;
            }
        }
        for &(win, score) in &scores_by_hap[chosen_idx] {
            if win < w && chosen_active[win] && score.is_finite() {
                z_w[win] = logaddexp(z_w[win], score);
            }
        }
        remaining = remaining.saturating_sub(len_chk);
    }

    // One coordinate-ascent polish sweep over selected donors.
    let mut polish_idx = 0usize;
    while polish_idx < n {
        let Some(old_state) = selected_states[polish_idx].take() else {
            polish_idx += 1;
            continue;
        };

        let (re_counts, re_logz, used_without_current) =
            recompute_counts_logz_used(&selected_states, scores_by_hap, w, per_window_caps);
        counts = re_counts;
        z_w = re_logz;
        remaining = total_budget.saturating_sub(used_without_current);

        let blocked: Vec<bool> = counts
            .iter()
            .enumerate()
            .map(|(w_i, &c)| c >= per_window_caps[w_i])
            .collect();
        let (gain, len) = dp_intervals_sparse_scratch(
            &scores_by_hap[polish_idx],
            &z_w,
            mu,
            &blocked,
            &t11,
            &t10,
            &t01,
            &mut dp_scratch,
        );

        if gain > 0.0 && len > 0 && len <= remaining {
            let active = dp_scratch.active[..w].to_vec();
            selected_states[polish_idx] = Some(AllocationState {
                active: active.clone(),
                len,
            });
            for win in 0..w {
                if active[win] && counts[win] < per_window_caps[win] {
                    counts[win] += 1;
                }
            }
            for &(win, score) in &scores_by_hap[polish_idx] {
                if win < w && active[win] && score.is_finite() {
                    z_w[win] = logaddexp(z_w[win], score);
                }
            }
            remaining = remaining.saturating_sub(len);
        } else {
            selected_states[polish_idx] = Some(old_state);
            let restored = selected_states[polish_idx]
                .as_ref()
                .expect("just restored allocation state");
            for win in 0..w {
                if restored.active[win] && counts[win] < per_window_caps[win] {
                    counts[win] += 1;
                }
            }
            for &(win, score) in &scores_by_hap[polish_idx] {
                if win < w && restored.active[win] && score.is_finite() {
                    z_w[win] = logaddexp(z_w[win], score);
                }
            }
            remaining = remaining.saturating_sub(restored.len);
        }

        polish_idx += 1;
    }

    let mut intervals_by_hap: Vec<(usize, Vec<WindowSpan>)> = Vec::new();
    let mut idx = 0usize;
    while idx < n {
        if let Some(state) = selected_states[idx].as_ref() {
            let intervals = active_to_intervals(&state.active);
            if !intervals.is_empty() {
                intervals_by_hap.push((candidate_haps[idx], intervals));
            }
        }
        idx += 1;
    }

    WindowAllocation { intervals_by_hap }
}

fn simulate_allocation(
    scores_by_hap: &[Vec<(usize, f32)>],
    num_windows: usize,
    t11: &[f32],
    t10: &[f32],
    t01: &[f32],
    mu: f32,
    per_window_caps: &[usize],
) -> (usize, f32) {
    let w = num_windows;
    if w == 0 {
        return (0, 0.0);
    }
    if per_window_caps.len() != w {
        return (0, 0.0);
    }
    let n = scores_by_hap.len();
    let mut counts = vec![0usize; w];
    let mut z_w = vec![0.0f32; w];
    let mut used = 0usize;
    let mut total_gain = 0.0f32;
    let mut selected = vec![false; n];

    #[derive(Clone)]
    struct HeapEntry {
        gain: f32,
        idx: usize,
        active: Vec<bool>,
        len: usize,
    }

    impl Eq for HeapEntry {}

    impl PartialEq for HeapEntry {
        fn eq(&self, other: &Self) -> bool {
            self.gain == other.gain && self.idx == other.idx
        }
    }

    impl Ord for HeapEntry {
        fn cmp(&self, other: &Self) -> Ordering {
            self.partial_cmp(other).unwrap_or(Ordering::Equal)
        }
    }

    impl PartialOrd for HeapEntry {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            self.gain
                .partial_cmp(&other.gain)
                .or_else(|| Some(self.idx.cmp(&other.idx)))
        }
    }

    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();
    let mut dp_scratch = DpScratch::default();
    {
        let blocked: Vec<bool> = counts
            .iter()
            .enumerate()
            .map(|(w_i, &c)| c >= per_window_caps[w_i])
            .collect();
        for h in 0..n {
            let scores = &scores_by_hap[h];
            let (gain, len) = dp_intervals_sparse_scratch(
                scores,
                &z_w,
                mu,
                &blocked,
                t11,
                t10,
                t01,
                &mut dp_scratch,
            );
            let active = dp_scratch.active[..w].to_vec();
            if gain > 0.0 && len > 0 {
                heap.push(HeapEntry {
                    gain,
                    idx: h,
                    active,
                    len,
                });
            }
        }
    }

    loop {
        let Some(mut entry) = heap.pop() else { break };
        if selected[entry.idx] {
            continue;
        }
        let blocked: Vec<bool> = counts
            .iter()
            .enumerate()
            .map(|(w_i, &c)| c >= per_window_caps[w_i])
            .collect();
        let (gain, len) = dp_intervals_sparse_scratch(
            &scores_by_hap[entry.idx],
            &z_w,
            mu,
            &blocked,
            t11,
            t10,
            t01,
            &mut dp_scratch,
        );
        let active = dp_scratch.active[..w].to_vec();
        if gain <= 0.0 || len == 0 {
            continue;
        }
        let next_gain = heap.peek().map(|e| e.gain).unwrap_or(NEG_INF);
        if gain >= next_gain {
            selected[entry.idx] = true;
            total_gain += gain;
            used += len;
            for win in 0..w {
                if active[win] && counts[win] < per_window_caps[win] {
                    counts[win] += 1;
                }
            }
            for &(win, score) in scores_by_hap[entry.idx].iter() {
                if win < w && active[win] && score.is_finite() {
                    z_w[win] = logaddexp(z_w[win], score);
                }
            }
        } else {
            entry.gain = gain;
            entry.active = active;
            entry.len = len;
            heap.push(entry);
        }
    }

    (used, total_gain)
}
