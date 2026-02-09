use std::collections::HashMap;

use crate::model::reference_pbwt::{PbwtQueryAllele, RankBeam};

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

#[inline]
pub fn phase_orientation_weight(phase_conf: f32, err_limit: f32) -> f32 {
    let best_orient_err = phase_best_orientation_error(phase_conf);
    let limit = err_limit.max(1e-6);
    (limit / best_orient_err.max(limit)).clamp(0.0, 1.0)
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
}
