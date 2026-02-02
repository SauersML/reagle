//! Condensed target representation for fast phasing.
//!
//! Collapses homozygous stretches into segments with reference consistency masks.

use crate::data::marker::AnyMarkerSpace;
use crate::data::storage::sample_phase::SamplePhase;
use crate::data::{MarkerIdx};
use crate::data::ref_packed::{PackedRefView, mask_all_ones, mask_and_inplace};
use crate::model::parameters::ModelParams;

#[derive(Clone, Debug)]
pub struct CondensedSegment {
    pub mask: Vec<u64>,
}

#[derive(Clone, Debug)]
pub struct CallSite {
    pub marker: MarkerIdx<AnyMarkerSpace>,
    pub hi_idx: usize,
    pub a1: u8,
    pub a2: u8,
    pub switch_cost: i32,
}

#[derive(Clone, Debug)]
pub struct CondensedTarget {
    pub segments: Vec<CondensedSegment>, // len = call_sites.len() + 1
    pub call_sites: Vec<CallSite>,
}

impl CondensedTarget {
    /// Build condensed target for a single sample on hi-freq marker subset.
    pub fn build<RefSpace>(
        sample_phase: &SamplePhase,
        hi_freq_to_orig: &[usize],
        hi_freq_gen_positions: &[f64],
        packed_ref: &PackedRefView<RefSpace>,
        params: &ModelParams,
    ) -> Self {
        let n_ref = packed_ref.n_ref_haps();
        let n_words = (n_ref + 63) / 64;

        let mut call_sites: Vec<CallSite> = Vec::new();
        let mut segments: Vec<CondensedSegment> = Vec::new();

        let mut last_call_pos: Option<f64> = None;

        // Iterate hi-freq markers to identify call sites.
        for (hi_idx, &orig_m) in hi_freq_to_orig.iter().enumerate() {
            let a1 = sample_phase.allele1(orig_m);
            let a2 = sample_phase.allele2(orig_m);
            if a1 == 255 || a2 == 255 {
                continue;
            }
            if a1 != a2 && a1 <= 1 && a2 <= 1 && sample_phase.is_unphased(orig_m) {
                let marker = MarkerIdx::new(orig_m as u32);
                let pos = hi_freq_gen_positions.get(hi_idx).copied().unwrap_or(0.0);
                let switch_cost = if let Some(prev_pos) = last_call_pos {
                    let dist = (pos - prev_pos).max(0.0);
                    let p_switch = params.p_recomb(dist).max(1e-12);
                    // fixed-point cost
                    (-p_switch.ln() * 1_000_000.0).round() as i32
                } else {
                    0
                };

                call_sites.push(CallSite {
                    marker,
                    hi_idx,
                    a1,
                    a2,
                    switch_cost,
                });
                last_call_pos = Some(pos);
            }
        }

        // Build segments between call sites (including leading/trailing).
        let mut prev_hi = 0usize;
        for cs in call_sites.iter() {
            let seg = build_segment_mask(
                sample_phase,
                hi_freq_to_orig,
                prev_hi,
                cs.hi_idx,
                packed_ref,
                n_words,
            );
            segments.push(seg);
            prev_hi = cs.hi_idx + 1;
        }
        // trailing segment after last call site
        let trailing = build_segment_mask(
            sample_phase,
            hi_freq_to_orig,
            prev_hi,
            hi_freq_to_orig.len(),
            packed_ref,
            n_words,
        );
        segments.push(trailing);

        if call_sites.is_empty() {
            // No hets: still create one segment covering all.
            if segments.is_empty() {
                let seg = build_segment_mask(
                    sample_phase,
                    hi_freq_to_orig,
                    0,
                    hi_freq_to_orig.len(),
                    packed_ref,
                    n_words,
                );
                segments.push(seg);
            }
        }

        Self { segments, call_sites }
    }
}

fn build_segment_mask<RefSpace>(
    sample_phase: &SamplePhase,
    hi_freq_to_orig: &[usize],
    start_hi: usize,
    end_hi: usize,
    packed_ref: &PackedRefView<RefSpace>,
    n_words: usize,
) -> CondensedSegment {
    let mut mask: Vec<u64> = vec![0u64; n_words];
    mask_all_ones(&mut mask, packed_ref.n_ref_haps());

    let mut tmp: Vec<u64> = vec![0u64; n_words];
    let mut any_constraint = false;

    for hi_idx in start_hi..end_hi {
        let orig_m = hi_freq_to_orig[hi_idx];
        let a1 = sample_phase.allele1(orig_m);
        let a2 = sample_phase.allele2(orig_m);
        if a1 == 255 || a2 == 255 {
            continue;
        }
        if a1 == a2 {
            if packed_ref.fill_match_mask(orig_m, a1, &mut tmp) {
                mask_and_inplace(&mut mask, &tmp);
                any_constraint = true;
            }
        }
    }

    if !any_constraint {
        mask_all_ones(&mut mask, packed_ref.n_ref_haps());
    }

    CondensedSegment { mask }
}
