use std::collections::HashMap;

use crate::model::reference_pbwt::{PbwtQueryAllele, RankBeam};

#[derive(Debug, Clone, Copy)]
pub struct OrientationDecision {
    pub use_wildcard: bool,
    pub flip_orientation: bool,
    pub allele_weight: f32,
    pub wildcard_info_weight: f32,
}

#[inline]
pub fn phase_query_orientation_error_limit(genotype_conf: f32, beam_uncertainty: f32) -> f32 {
    let base = 0.08 + 0.17 * genotype_conf.clamp(0.0, 1.0);
    let scale = 0.55 + 0.9 * beam_uncertainty.clamp(0.0, 1.0);
    (base * scale).clamp(0.04, 0.30)
}

#[inline]
pub fn phase_best_orientation_error(phase_conf: f32) -> f32 {
    let p = phase_conf.clamp(0.0, 1.0);
    p.min(1.0 - p)
}

/// Robust orientation decision for heterozygous query markers.
///
/// This calibration intentionally downweights optimistic phase confidence when
/// either genotype confidence is poor or local PBWT beams are diffuse.
#[inline]
pub fn decide_phase_query_orientation(
    phase_conf: f32,
    genotype_conf: f32,
    beam_uncertainty: f32,
) -> OrientationDecision {
    let p = phase_conf.clamp(0.0, 1.0);
    let g = genotype_conf.clamp(0.0, 1.0);
    let u = beam_uncertainty.clamp(0.0, 1.0);

    // Signed margin around 0.5 (orientation boundary).
    let margin = 2.0 * p - 1.0;
    let abs_margin = margin.abs();

    // Evidence quality increases with genotype confidence and decreases with
    // beam uncertainty. This controls both confidence shrinkage and gating.
    let evidence_quality = (g.sqrt() * (1.0 - 0.65 * u)).clamp(0.0, 1.0);

    // Non-linear shrinkage:
    //  - small margins are penalized aggressively,
    //  - large margins survive if evidence quality is good.
    let sharpened_margin = abs_margin.powf(1.35);
    let calibrated_margin = (sharpened_margin * evidence_quality).clamp(0.0, 1.0);

    let calibrated_conf = if margin >= 0.0 {
        0.5 + 0.5 * calibrated_margin
    } else {
        0.5 - 0.5 * calibrated_margin
    };

    let err_limit = phase_query_orientation_error_limit(g, u).max(1e-6);
    let calibrated_err = phase_best_orientation_error(calibrated_conf);

    // Weight is smooth and stricter near the boundary than the previous
    // piecewise heuristic.
    let relative_headroom = ((err_limit - calibrated_err) / err_limit).clamp(0.0, 1.0);
    let allele_weight = relative_headroom * relative_headroom;
    let use_wildcard = calibrated_err > err_limit || allele_weight <= 0.0;

    OrientationDecision {
        use_wildcard,
        flip_orientation: calibrated_conf < 0.5,
        allele_weight: if use_wildcard { 0.0 } else { allele_weight },
        wildcard_info_weight: if use_wildcard {
            uncertain_orientation_wildcard_info_weight()
        } else {
            0.0
        },
    }
}

#[inline]
pub fn pbwt_beam_uncertainty(beam: &RankBeam, n_ref_haps: usize, query: PbwtQueryAllele) -> f32 {
    if n_ref_haps == 0 {
        return 0.0;
    }
    let mut total = 0.0f32;
    let mut sq_sum = 0.0f32;
    let mut n_intervals = 0usize;
    for &(l, r) in beam.intervals() {
        if r <= l {
            continue;
        }
        let len = (r - l) as f32;
        total += len;
        sq_sum += len * len;
        n_intervals += 1;
    }
    if total <= 0.0 {
        return 0.0;
    }
    let coverage = (total / n_ref_haps as f32).clamp(0.0, 1.0);
    let spread = if n_intervals > 1 && sq_sum > 0.0 {
        let neff = (total * total / sq_sum).clamp(1.0, n_intervals as f32);
        (neff - 1.0) / (n_intervals as f32 - 1.0)
    } else {
        0.0
    };
    let mut uncertainty = (0.7 * coverage + 0.3 * spread).clamp(0.0, 1.0);
    if query.is_wildcard() {
        uncertainty = uncertainty.max(0.85);
    }
    uncertainty
}

#[inline]
pub fn build_peer_indices(haps: &[usize]) -> Vec<Option<usize>> {
    let mut peers = vec![None; haps.len()];
    let mut slots_by_sample: HashMap<usize, (Option<usize>, Option<usize>)> =
        HashMap::with_capacity(haps.len() / 2 + 1);
    for (i, &hap_idx) in haps.iter().enumerate() {
        let sample_idx = hap_idx / 2;
        let local = hap_idx % 2;
        let slots = slots_by_sample.entry(sample_idx).or_insert((None, None));
        if local == 0 {
            if slots.0.is_none() {
                slots.0 = Some(i);
            }
        } else if slots.1.is_none() {
            slots.1 = Some(i);
        }
    }
    for (i, &hap_idx) in haps.iter().enumerate() {
        let sample_idx = hap_idx / 2;
        let local = hap_idx % 2;
        if let Some(&(even_idx, odd_idx)) = slots_by_sample.get(&sample_idx) {
            peers[i] = if local == 0 { odd_idx } else { even_idx };
        }
    }
    peers
}

#[inline]
pub const fn uncertain_orientation_wildcard_info_weight() -> f32 {
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peer_indices_are_order_independent_and_parity_correct() {
        let haps = vec![5, 0, 2, 7, 3, 4, 1, 6];
        let peers = build_peer_indices(&haps);
        for (i, &h) in haps.iter().enumerate() {
            if let Some(j) = peers[i] {
                assert_eq!(h / 2, haps[j] / 2);
                assert_ne!(h % 2, haps[j] % 2);
                assert_eq!(peers[j], Some(i));
            }
        }
    }

    #[test]
    fn uncertain_orientation_wildcard_info_is_zero() {
        let w = uncertain_orientation_wildcard_info_weight();
        assert_eq!(w, 0.0);
    }

    #[test]
    fn orientation_decision_keeps_strong_clean_signal() {
        let d = decide_phase_query_orientation(0.995, 0.99, 0.02);
        assert!(!d.use_wildcard);
        assert!(!d.flip_orientation);
        assert!(d.allele_weight > 0.7);
    }

    #[test]
    fn orientation_decision_wildcards_ambiguous_noisy_signal() {
        let d = decide_phase_query_orientation(0.57, 0.38, 0.95);
        assert!(d.use_wildcard);
        assert_eq!(d.allele_weight, 0.0);
    }

    #[test]
    fn orientation_decision_preserves_flip_direction() {
        let d = decide_phase_query_orientation(0.04, 0.98, 0.05);
        assert!(d.flip_orientation);
        assert!(!d.use_wildcard);
    }
}
