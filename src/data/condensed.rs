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
    pub any_constraint: bool,
}

#[derive(Clone, Debug)]
pub struct CallSite {
    pub marker: MarkerIdx<AnyMarkerSpace>,
    pub hi_idx: usize,
    pub a1: u8,
    pub a2: u8,
    pub switch_cost: i32,
    pub flip_cost: i32,
    pub fixed: bool,
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
            if a1 != a2 && a1 <= 1 && a2 <= 1 {
                let marker = MarkerIdx::new(orig_m as u32);
                let pos = hi_freq_gen_positions.get(hi_idx).copied().unwrap_or(0.0);
                let switch_cost = if let Some(prev_pos) = last_call_pos {
                    let dist = (pos - prev_pos).max(0.0);
                    // Use log-odds cost so "stay" is the implicit baseline.
                    let p_switch = params.p_recomb(dist).clamp(1e-12, 0.5 - 1e-12);
                    let odds = p_switch / (1.0 - p_switch);
                    (-odds.ln() * 1_000_000.0).round() as i32
                } else {
                    0
                };
                let fixed = !sample_phase.is_unphased(orig_m);
                let flip_cost = if fixed {
                    if switch_cost == 0 {
                        250_000
                    } else {
                        switch_cost.max(250_000)
                    }
                } else if switch_cost == 0 {
                    1_000_000
                } else {
                    switch_cost.max(1_000_000)
                };

                call_sites.push(CallSite {
                    marker,
                    hi_idx,
                    a1,
                    a2,
                    switch_cost,
                    flip_cost,
                    fixed,
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


        Self { segments, call_sites }
    }

    pub fn reversed(
        &self,
        hi_freq_gen_positions: &[f64],
        params: &ModelParams,
    ) -> Self {
        let mut call_sites_rev: Vec<CallSite> = Vec::with_capacity(self.call_sites.len());
        let mut last_pos: Option<f64> = None;
        for cs in self.call_sites.iter().rev() {
            let pos = hi_freq_gen_positions.get(cs.hi_idx).copied().unwrap_or(0.0);
            let switch_cost = if let Some(prev_pos) = last_pos {
                let dist = (pos - prev_pos).abs();
                let p_switch = params.p_recomb(dist).clamp(1e-12, 0.5 - 1e-12);
                let odds = p_switch / (1.0 - p_switch);
                (-odds.ln() * 1_000_000.0).round() as i32
            } else {
                0
            };
            let flip_cost = if cs.fixed {
                if switch_cost == 0 {
                    250_000
                } else {
                    switch_cost.max(250_000)
                }
            } else if switch_cost == 0 {
                1_000_000
            } else {
                switch_cost.max(1_000_000)
            };
            call_sites_rev.push(CallSite {
                marker: cs.marker,
                hi_idx: cs.hi_idx,
                a1: cs.a1,
                a2: cs.a2,
                switch_cost,
                flip_cost,
                fixed: cs.fixed,
            });
            last_pos = Some(pos);
        }
        let segments_rev: Vec<CondensedSegment> = self.segments.iter().rev().cloned().collect();
        CondensedTarget {
            segments: segments_rev,
            call_sites: call_sites_rev,
        }
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
    let mut tmp2: Vec<u64> = vec![0u64; n_words];
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
        } else if a1 <= 1 && a2 <= 1 {
            // If heterozygous and phased, constrain segment to match either allele.
            // This anchors local consistency across phased hets.
            if !sample_phase.is_unphased(orig_m) {
                let mut ok = false;
                if packed_ref.fill_match_mask(orig_m, a1, &mut tmp) {
                    ok = true;
                }
                if packed_ref.fill_match_mask(orig_m, a2, &mut tmp2) {
                    ok = true;
                }
                if ok {
                    for i in 0..mask.len().min(tmp.len()).min(tmp2.len()) {
                        tmp[i] |= tmp2[i];
                    }
                    mask_and_inplace(&mut mask, &tmp);
                    any_constraint = true;
                }
            }
        }
    }

    if !any_constraint {
        mask_all_ones(&mut mask, packed_ref.n_ref_haps());
    }

    CondensedSegment { mask, any_constraint }
}
