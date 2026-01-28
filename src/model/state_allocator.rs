//! Lagrangian Marginal Stickiness allocator for prescan state selection.
//!
//! This module implements a *selection-layer* optimization aligned with the
//! Li–Stephens transition physics, without claiming to be the exact HMM marginal
//! likelihood. The goal is to allocate haplotype identities across windows under
//! a hard RAM budget.
//!
//! Key ideas (math in brief):
//! - Window evidence for haplotype h at window w is a log-evidence score S[h,w].
//! - Convert to nonnegative evidence L[h,w] = exp(S[h,w]).
//!   (We clamp S to a finite range to avoid overflow while preserving ordering.)
//! - Maintain per-window explained mass Z_w = 1 + sum_{h in A_w} L[h,w].
//! - Marginal gain of adding h at w is:
//!     u_{h,w} = log(Z_w + L[h,w]) - log(Z_w) - mu
//!   where mu is the Lagrangian price per window-slot.
//! - Continuity bonus is derived from Li–Stephens stay-vs-switch odds:
//!     a_w = exp(-rho * d_w)  (consistent with model parameters)
//!     p_stay = a_w + (1-a_w)/n_pool
//!     p_switch_diff = (1-a_w) * (n_pool-1)/n_pool
//!     b_w = log(p_stay / p_switch_diff)
//! - For each haplotype, the optimal disjoint ON-intervals are found by a
//!   2-state DP over windows (exact for this surrogate objective).
//!
//! The outer loop is coordinate ascent: repeatedly pick the haplotype with the
//! highest DP gain, activate its interval set, update Z_w, and continue until
//! the global slot budget is exhausted or no positive gain remains. A binary
//! search over mu tunes the total slot usage.

use crate::model::parameters::ModelParams;

const NEG_INF: f32 = -1.0e30;
const MAX_LOG_EVIDENCE: f32 = 50.0;

#[inline]
fn exp_score(score: f32) -> f32 {
    score.clamp(-MAX_LOG_EVIDENCE, MAX_LOG_EVIDENCE).exp()
}

/// Allocation result for a single target haplotype:
/// active haplotype ids per window (indices into reference panel).
#[derive(Clone, Debug)]
pub struct WindowAllocation {
    pub active_by_window: Vec<Vec<usize>>,
}

/// Compute Li–Stephens continuity bonus per boundary.
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
///     b_w = log(p_stay / p_switch_diff)
///
/// This is the exact stay-vs-switch-to-different odds for the Li–Stephens chain.
fn continuity_bonus(
    boundary_cm: &[f64],
    params: &ModelParams,
    n_pool: usize,
) -> Vec<f32> {
    let mut b = Vec::with_capacity(boundary_cm.len());
    let n_pool_f = n_pool.max(2) as f32;
    for &dist_cm in boundary_cm {
        let r_w = params.p_recomb(dist_cm);
        let a_w = (1.0 - r_w).max(0.0).min(1.0);
        let p_stay = a_w + (1.0 - a_w) / n_pool_f;
        let p_switch_diff = (1.0 - a_w) * (n_pool_f - 1.0) / n_pool_f;
        let bonus = if p_stay > 0.0 && p_switch_diff > 0.0 {
            (p_stay / p_switch_diff).ln()
        } else {
            0.0
        };
        b.push(bonus);
    }
    b
}

/// Run the 2-state DP for a single haplotype.
///
/// Inputs:
/// - u_w: marginal gain per window (already includes mu penalty)
/// - b_w: continuity bonus per boundary
///
/// Returns: (total_gain, active_flags)
fn dp_intervals(u_w: &[f32], b_w: &[f32]) -> (f32, Vec<bool>) {
    let w = u_w.len();
    if w == 0 {
        return (0.0, Vec::new());
    }
    let mut dp0 = vec![0.0f32; w];
    let mut dp1 = vec![NEG_INF; w];
    let mut prev0 = vec![0u8; w];
    let mut prev1 = vec![0u8; w];

    dp0[0] = 0.0;
    dp1[0] = u_w[0];

    for i in 1..w {
        // OFF state
        if dp0[i - 1] >= dp1[i - 1] {
            dp0[i] = dp0[i - 1];
            prev0[i] = 0;
        } else {
            dp0[i] = dp1[i - 1];
            prev0[i] = 1;
        }

        // ON state
        let from_off = dp0[i - 1];
        let from_on = dp1[i - 1] + b_w[i - 1];
        if from_off >= from_on {
            dp1[i] = u_w[i] + from_off;
            prev1[i] = 0;
        } else {
            dp1[i] = u_w[i] + from_on;
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

/// Allocate active haplotypes per window for a single target haplotype.
///
/// Inputs:
/// - scores_by_window: S[h,w] as Vec[W][N], log-evidence scores.
/// - boundary_cm: window boundary distances (len W-1).
/// - params: model params for recombination mapping.
/// - total_budget: total slots allowed across all windows (K_total * W).
/// - per_window_cap: max states per window (K_total).
/// - abyss: optional mask of haplotypes to exclude.
pub fn allocate_lms(
    scores_by_window: &[Vec<f32>],
    boundary_cm: &[f64],
    params: &ModelParams,
    total_budget: usize,
    per_window_cap: usize,
    abyss: Option<&[bool]>,
) -> WindowAllocation {
    let w = scores_by_window.len();
    if w == 0 || total_budget == 0 {
        return WindowAllocation {
            active_by_window: vec![Vec::new(); w],
        };
    }
    let n = scores_by_window[0].len();
    let mut active_by_window: Vec<Vec<usize>> = vec![Vec::new(); w];
    let mut counts = vec![0usize; w];
    let mut z_w = vec![1.0f32; w];
    let mut remaining = total_budget;

    // Use pool size based on per-window cap + background.
    let n_pool = 1 + per_window_cap.max(1);
    let b_w = continuity_bonus(boundary_cm, params, n_pool);

    // Determine mu by binary search to approximately meet budget.
    // We tune mu on the fly: larger mu -> fewer activations.
    let mut mu_low = -10.0f32;
    let mut mu_high = 10.0f32;

    // Ensure bounds bracket a feasible range.
    let mut used_low = 0usize;
    for _ in 0..5 {
        let (used, _) = simulate_allocation(
            scores_by_window,
            &b_w,
            mu_low,
            per_window_cap,
            abyss,
        );
        used_low = used;
        if used_low >= total_budget {
            break;
        }
        mu_low -= 5.0;
    }
    let mut used_high = 0usize;
    for _ in 0..5 {
        let (used, _) = simulate_allocation(
            scores_by_window,
            &b_w,
            mu_high,
            per_window_cap,
            abyss,
        );
        used_high = used;
        if used_high <= total_budget {
            break;
        }
        mu_high += 5.0;
    }

    let mut mu = mu_high;
    for _ in 0..16 {
        let (used, _) = simulate_allocation(
            scores_by_window,
            &b_w,
            mu,
            per_window_cap,
            abyss,
        );
        if used > total_budget {
            mu_low = mu;
        } else {
            mu_high = mu;
        }
        mu = 0.5 * (mu_low + mu_high);
    }

    // Main coordinate-ascent allocation using tuned mu.
    while remaining > 0 {
        let mut best_gain = 0.0f32;
        let mut best_h = None;
        let mut best_active: Vec<bool> = Vec::new();
        let mut best_len = 0usize;

        for h in 0..n {
            if let Some(mask) = abyss {
                if h < mask.len() && mask[h] {
                    continue;
                }
            }
            let mut u_w = vec![NEG_INF; w];
            for win in 0..w {
                if counts[win] >= per_window_cap {
                    continue;
                }
                let score = scores_by_window[win][h];
                if !score.is_finite() {
                    continue;
                }
                let l = exp_score(score);
                let z = z_w[win];
                let gain = (z + l).ln() - z.ln() - mu;
                u_w[win] = gain;
            }
            let (gain, active) = dp_intervals(&u_w, &b_w);
            if gain > best_gain {
                let len = active.iter().filter(|v| **v).count();
                if len == 0 || len > remaining {
                    continue;
                }
                best_gain = gain;
                best_h = Some(h);
                best_active = active;
                best_len = len;
            }
        }

        if best_gain <= 0.0 {
            break;
        }
        let Some(h) = best_h else { break };

        for win in 0..w {
            if best_active[win] && counts[win] < per_window_cap {
                active_by_window[win].push(h);
                counts[win] += 1;
                let score = scores_by_window[win][h];
                if score.is_finite() {
                    z_w[win] += exp_score(score);
                }
            }
        }
        if best_len > remaining {
            break;
        }
        remaining -= best_len;
    }

    WindowAllocation { active_by_window }
}

fn simulate_allocation(
    scores_by_window: &[Vec<f32>],
    b_w: &[f32],
    mu: f32,
    per_window_cap: usize,
    abyss: Option<&[bool]>,
) -> (usize, f32) {
    let w = scores_by_window.len();
    if w == 0 {
        return (0, 0.0);
    }
    let n = scores_by_window[0].len();
    let mut counts = vec![0usize; w];
    let mut z_w = vec![1.0f32; w];
    let mut used = 0usize;
    let mut total_gain = 0.0f32;

    loop {
        let mut best_gain = 0.0f32;
        let mut best_h = None;
        let mut best_active: Vec<bool> = Vec::new();
        let mut best_len = 0usize;

        for h in 0..n {
            if let Some(mask) = abyss {
                if h < mask.len() && mask[h] {
                    continue;
                }
            }
            let mut u_w = vec![NEG_INF; w];
            for win in 0..w {
                if counts[win] >= per_window_cap {
                    continue;
                }
                let score = scores_by_window[win][h];
                if !score.is_finite() {
                    continue;
                }
                let l = exp_score(score);
                let z = z_w[win];
                let gain = (z + l).ln() - z.ln() - mu;
                u_w[win] = gain;
            }
            let (gain, active) = dp_intervals(&u_w, b_w);
            if gain > best_gain {
                let len = active.iter().filter(|v| **v).count();
                if len == 0 {
                    continue;
                }
                best_gain = gain;
                best_h = Some(h);
                best_active = active;
                best_len = len;
            }
        }

        if best_gain <= 0.0 {
            break;
        }
        let Some(h) = best_h else { break };
        total_gain += best_gain;
        used += best_len;
        for win in 0..w {
            if best_active[win] && counts[win] < per_window_cap {
                counts[win] += 1;
                let score = scores_by_window[win][h];
                if score.is_finite() {
                    z_w[win] += exp_score(score);
                }
            }
        }
    }

    (used, total_gain)
}
