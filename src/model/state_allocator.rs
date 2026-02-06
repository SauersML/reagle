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

/// Allocation result for a single target haplotype:
/// intervals per selected haplotype (reference panel indices).
#[derive(Clone, Debug)]
pub struct WindowAllocation {
    pub intervals_by_hap: Vec<(usize, Vec<(u32, u32)>)>,
}

/// Compute continuity transition terms per boundary.
///
/// We must be consistent with the HMM parameterization. The recombination
/// probability used by the HMM is:
///
///     r_w = p_recomb(d_w)
///     a_w = 1 - r_w
///
/// Then, with donor pool size n_pool:
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
    n_pool: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut t11 = Vec::with_capacity(boundary_cm.len());
    let mut t10 = Vec::with_capacity(boundary_cm.len());
    let mut t01 = Vec::with_capacity(boundary_cm.len());
    let n_pool_f = n_pool.max(2) as f32;
    for &dist_cm in boundary_cm {
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
/// Returns: (total_gain, active_flags)
fn dp_intervals_sparse(
    scores: &[(usize, f32)],
    logz: &[f32],
    mu: f32,
    blocked: &[bool],
    t11: &[f32],
    t10: &[f32],
    t01: &[f32],
) -> (f32, Vec<bool>) {
    let w = logz.len();
    if w == 0 {
        return (0.0, Vec::new());
    }
    let mut dp0 = vec![0.0f32; w];
    let mut dp1 = vec![NEG_INF; w];
    let mut prev0 = vec![0u8; w];
    let mut prev1 = vec![0u8; w];

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

    let mut active = vec![false; w];
    let mut state = if dp1[w - 1] >= dp0[w - 1] { 1 } else { 0 };
    let mut i = w - 1;
    loop {
        if state == 1 {
            active[i] = true;
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
    (gain, active)
}

fn active_to_intervals(active: &[bool]) -> Vec<(u32, u32)> {
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
        out.push((start as u32, end as u32));
        i = end + 1;
    }
    out
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
            .map(|&h| (h, vec![(0u32, w as u32)]))
            .collect();
        return WindowAllocation { intervals_by_hap };
    }
    let mut intervals_by_hap: Vec<(usize, Vec<(u32, u32)>)> = Vec::new();
    let mut counts = vec![0usize; w];
    let mut z_w = vec![0.0f32; w]; // log(1)
    let mut selected = vec![false; n];
    let mut remaining = total_budget;

    // Use a fixed donor pool size for separability. We default to the full
    // reference panel size so continuity odds align with the actual HMM physics.
    let (t11, t10, t01) = continuity_terms(boundary_cm, params, n_pool);

    // Determine mu by binary search to approximately meet budget.
    // We tune mu on the fly: larger mu -> fewer activations.
    let mut mu_low = -10.0f32;
    let mut mu_high = 10.0f32;

    // Ensure bounds bracket a feasible range.
    for _ in 0..5 {
        let (used, _) =
            simulate_allocation(scores_by_hap, w, &t11, &t10, &t01, mu_low, per_window_caps);
        if used >= total_budget {
            break;
        }
        mu_low -= 5.0;
    }
    for _ in 0..5 {
        let (used, _) =
            simulate_allocation(scores_by_hap, w, &t11, &t10, &t01, mu_high, per_window_caps);
        if used <= total_budget {
            break;
        }
        mu_high += 5.0;
    }

    // Coarse grid search over mu to avoid non-monotone behavior in greedy outer loop.
    let mut mu_best = mu_high;
    let mut best_used = 0usize;
    let mut best_gain = NEG_INF;
    for k in 0..17 {
        let t = k as f32 / 16.0;
        let mu = mu_low + t * (mu_high - mu_low);
        let (used, gain) =
            simulate_allocation(scores_by_hap, w, &t11, &t10, &t01, mu, per_window_caps);
        if used <= total_budget && (gain > best_gain || (gain == best_gain && used > best_used)) {
            mu_best = mu;
            best_used = used;
            best_gain = gain;
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
    {
        let blocked: Vec<bool> = counts
            .iter()
            .enumerate()
            .map(|(w_i, &c)| c >= per_window_caps[w_i])
            .collect();
        for h in 0..n {
            let scores = &scores_by_hap[h];
            let (gain, active) = dp_intervals_sparse(scores, &z_w, mu, &blocked, &t11, &t10, &t01);
            let len = active.iter().filter(|v| **v).count();
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
        if selected[entry.idx] {
            continue;
        }
        let blocked: Vec<bool> = counts
            .iter()
            .enumerate()
            .map(|(w_i, &c)| c >= per_window_caps[w_i])
            .collect();
        let (gain, active) = dp_intervals_sparse(
            &scores_by_hap[entry.idx],
            &z_w,
            mu,
            &blocked,
            &t11,
            &t10,
            &t01,
        );
        let len = active.iter().filter(|v| **v).count();
        if gain <= 0.0 || len == 0 || len > remaining {
            continue;
        }
        let next_gain = heap.peek().map(|e| e.gain).unwrap_or(NEG_INF);
        if gain >= next_gain {
            selected[entry.idx] = true;
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
            let intervals = active_to_intervals(&active);
            if !intervals.is_empty() {
                intervals_by_hap.push((candidate_haps[entry.idx], intervals));
            }
            remaining = remaining.saturating_sub(len);
        } else {
            entry.gain = gain;
            entry.active = active;
            entry.len = len;
            heap.push(entry);
        }
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
    {
        let blocked: Vec<bool> = counts
            .iter()
            .enumerate()
            .map(|(w_i, &c)| c >= per_window_caps[w_i])
            .collect();
        for h in 0..n {
            let scores = &scores_by_hap[h];
            let (gain, active) = dp_intervals_sparse(scores, &z_w, mu, &blocked, t11, t10, t01);
            let len = active.iter().filter(|v| **v).count();
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
        let (gain, active) =
            dp_intervals_sparse(&scores_by_hap[entry.idx], &z_w, mu, &blocked, t11, t10, t01);
        let len = active.iter().filter(|v| **v).count();
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
