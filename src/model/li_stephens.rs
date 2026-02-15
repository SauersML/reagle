//! Canonical Li-Stephens transition math.
//!
//! This is the single source of truth for panel-conditioned transition
//! projections used by phasing, imputation, and selection-layer allocators.

#[inline]
pub fn subset_linear_exact_k(recomb_rate: f32, k_subset: f32, n_total: usize) -> (f32, f32) {
    let r = recomb_rate.clamp(0.0, 1.0);
    let n = n_total.max(1) as f32;
    // exact_k preserves caller-provided subset size (no clamping to [1, n]),
    // but still rejects nonphysical values to avoid NaN/invalid transitions.
    if !k_subset.is_finite() || k_subset <= 0.0 {
        return (0.0, 0.0);
    }
    let k = k_subset;
    let switch_full = r / n;
    let z = ((1.0 - r) + k * switch_full).max(1e-30);
    let stay = (1.0 - r) / z;
    let shift = switch_full / z;
    (stay, shift)
}

#[inline]
pub fn subset_linear_clamped_k(recomb_rate: f32, k_subset: f32, n_total: usize) -> (f32, f32) {
    let r = recomb_rate.clamp(0.0, 1.0);
    let n = n_total.max(1) as f32;
    let k = if k_subset.is_finite() {
        k_subset.clamp(1.0, n)
    } else {
        1.0
    };
    let switch_full = r / n;
    let z = ((1.0 - r) + k * switch_full).max(1e-30);
    let stay = (1.0 - r) / z;
    let shift = switch_full / z;
    (stay, shift)
}

#[inline]
pub fn normalized_switch_scale_shift(
    p_switch: f32,
    n_states: usize,
    sum: f32,
    min_sum: f32,
) -> (f32, f32) {
    let n = n_states.max(1) as f32;
    // Clamp to a valid probability range to keep affine transition updates
    // numerically physical even if upstream noise pushes p_switch slightly out of range.
    let r = p_switch.clamp(0.0, 1.0);
    let shift = r / n;
    let scale = (1.0 - r) / sum.max(min_sum);
    (scale, shift)
}

#[inline]
pub fn on_off_transition_probs(recomb_rate: f32, donor_pool: f32) -> (f32, f32, f32, f32) {
    let n_pool = donor_pool.max(1.0);
    let a_w = (1.0 - recomb_rate).clamp(0.0, 1.0);
    let p11 = a_w + (1.0 - a_w) / n_pool;
    let p10 = (1.0 - a_w) * (n_pool - 1.0) / n_pool;
    let p01 = (1.0 - a_w) / n_pool;
    let p00 = 1.0 - p01;
    (p11, p10, p01, p00)
}

#[inline]
pub fn on_off_transition_log_odds(recomb_rate: f32, donor_pool: f32) -> (f32, f32, f32) {
    let (p11, p10, p01, p00) = on_off_transition_probs(recomb_rate, donor_pool);
    let t11 = if p11 > 0.0 && p00 > 0.0 {
        (p11 / p00).ln()
    } else {
        f32::NEG_INFINITY
    };
    let t10 = if p10 > 0.0 && p00 > 0.0 {
        (p10 / p00).ln()
    } else {
        f32::NEG_INFINITY
    };
    let t01 = if p01 > 0.0 && p00 > 0.0 {
        (p01 / p00).ln()
    } else {
        f32::NEG_INFINITY
    };
    (t11, t10, t01)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subset_linear_exact_k_preserves_probability_mass() {
        let cases = [
            (0.0f32, 1.0f32, 1usize),
            (0.001, 3.0, 100),
            (0.05, 16.0, 1000),
            (0.25, 64.0, 5000),
            (0.75, 32.0, 40),
            (1.0, 10.0, 10),
        ];
        for (r, k, n) in cases {
            let (stay, shift) = subset_linear_exact_k(r, k, n);
            assert!(
                stay.is_finite() && stay >= 0.0,
                "stay invalid for r={r}, k={k}, n={n}"
            );
            assert!(
                shift.is_finite() && shift >= 0.0,
                "shift invalid for r={r}, k={k}, n={n}"
            );
            let mass = stay + k * shift;
            assert!(
                (mass - 1.0).abs() < 1e-6,
                "mass not conserved: r={r}, k={k}, n={n}, mass={mass}"
            );
        }
    }

    #[test]
    fn subset_linear_exact_k_matches_explicit_transition_update() {
        let fwd = [0.05f32, 0.25, 0.15, 0.10, 0.45];
        let sum: f32 = fwd.iter().sum();
        let (stay, shift) = subset_linear_exact_k(0.07, fwd.len() as f32, 5000);

        let mut direct = vec![0.0f32; fwd.len()];
        for i in 0..fwd.len() {
            let mut s = 0.0f32;
            for (j, &p_j) in fwd.iter().enumerate() {
                let t = if i == j { stay + shift } else { shift };
                s += p_j * t;
            }
            direct[i] = s;
        }

        for i in 0..fwd.len() {
            let affine = stay * fwd[i] + shift * sum;
            assert!(
                (affine - direct[i]).abs() < 1e-6,
                "mismatch at i={i}: affine={affine}, direct={}",
                direct[i]
            );
        }
    }
}
