//! Condensed target representation for fast phasing.
//!
//! Collapses homozygous stretches into segments with compact constraints.

use crate::data::marker::AnyMarkerSpace;
use crate::data::storage::sample_phase::SamplePhase;
use crate::data::{MarkerIdx};
use crate::data::ref_packed::PackedRefView;
use crate::model::parameters::ModelParams;

#[derive(Clone, Debug)]
pub struct CondensedSegment {
    pub constraints: Vec<SegmentConstraint>,
    pub any_constraint: bool,
    pub len_morgans: f32,
}

#[derive(Clone, Debug)]
pub struct SegmentConstraint {
    pub marker: MarkerIdx<AnyMarkerSpace>,
    pub alleles: [u8; 2],
    pub n_alleles: u8,
}

#[derive(Clone, Debug)]
pub struct CallSite {
    pub marker: MarkerIdx<AnyMarkerSpace>,
    pub hi_idx: usize,
    pub a1: u8,
    pub a2: u8,
    /// Frequency of allele a1 in the reference panel (for TMRCA-aware scoring).
    pub a1_freq: f32,
    /// Frequency of allele a2 in the reference panel (for TMRCA-aware scoring).
    pub a2_freq: f32,
    /// PBWT-derived match length proxy for allele a1 (in marker steps).
    pub pbwt_len_a1: f32,
    /// PBWT-derived match length proxy for allele a2 (in marker steps).
    pub pbwt_len_a2: f32,
    /// PBWT-derived density proxy for allele a1 (count of candidate haps).
    pub pbwt_density_a1: f32,
    /// PBWT-derived density proxy for allele a2 (count of candidate haps).
    pub pbwt_density_a2: f32,
    /// Genetic distance to previous call site (Morgans).
    pub dist_morgans: f32,
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
    ///
    /// `allele_freqs` contains (freq_allele0, freq_allele1) for each hi-freq marker.
    pub fn build<RefSpace>(
        sample_phase: &SamplePhase,
        hi_freq_to_orig: &[usize],
        hi_freq_gen_positions: &[f64],
        allele_freqs: Option<&[(f32, f32)]>,
        pbwt_stats: Option<&[(f32, f32, f32, f32)]>,
        packed_ref: &PackedRefView<RefSpace>,
        params: &ModelParams,
    ) -> Self {
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
                let dist = if let Some(prev_pos) = last_call_pos {
                    (pos - prev_pos).max(0.0)
                } else {
                    0.0
                };
                let switch_cost = 0;
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

                // Get allele frequencies (default to 0.5 if not available).
                let (a1_freq, a2_freq) = allele_freqs
                    .and_then(|af| af.get(hi_idx))
                    .map(|&(f0, f1)| {
                        // Map target allele to reference frequency.
                        // a1/a2 are 0 or 1, so f0 is freq of allele 0, f1 is freq of allele 1.
                        let fa1 = if a1 == 0 { f0 } else { f1 };
                        let fa2 = if a2 == 0 { f0 } else { f1 };
                        (fa1.max(1e-6), fa2.max(1e-6))
                    })
                    .unwrap_or((0.5, 0.5));
                let (pbwt_len_a1, pbwt_len_a2, pbwt_density_a1, pbwt_density_a2) = pbwt_stats
                    .and_then(|stats| stats.get(hi_idx).copied())
                    .map(|(len0, len1, den0, den1)| {
                        let (l1, d1) = if a1 == 0 {
                            (len0, den0)
                        } else if a1 == 1 {
                            (len1, den1)
                        } else {
                            (0.0, 0.0)
                        };
                        let (l2, d2) = if a2 == 0 {
                            (len0, den0)
                        } else if a2 == 1 {
                            (len1, den1)
                        } else {
                            (0.0, 0.0)
                        };
                        (l1, l2, d1, d2)
                    })
                    .unwrap_or((0.0, 0.0, 0.0, 0.0));
                call_sites.push(CallSite {
                    marker,
                    hi_idx,
                    a1,
                    a2,
                    a1_freq,
                    a2_freq,
                    pbwt_len_a1,
                    pbwt_len_a2,
                    pbwt_density_a1,
                    pbwt_density_a2,
                    dist_morgans: dist as f32,
                    switch_cost,
                    flip_cost,
                    fixed,
                });
                last_call_pos = Some(pos);
            }
        }

        let theta = params.p_mismatch.max(1e-9) as f64;
        let hard_threshold = theta.ln().neg() + 6.0;
        // Build segments between call sites (including leading/trailing).
        let mut prev_hi = 0usize;
        for cs in call_sites.iter() {
            let end_hi = (cs.hi_idx + 1).min(hi_freq_to_orig.len());
            let seg = build_segment_mask(
                sample_phase,
                hi_freq_to_orig,
                prev_hi,
                end_hi,
                hi_freq_gen_positions,
                allele_freqs,
                hard_threshold,
                theta,
                packed_ref,
            );
            segments.push(seg);
            prev_hi = end_hi;
        }
        // trailing segment after last call site
        let trailing = build_segment_mask(
            sample_phase,
            hi_freq_to_orig,
            prev_hi,
            hi_freq_to_orig.len(),
            hi_freq_gen_positions,
            allele_freqs,
            hard_threshold,
            theta,
            packed_ref,
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
            let dist = if let Some(prev_pos) = last_pos {
                (pos - prev_pos).abs()
            } else {
                0.0
            };
            let switch_cost = 0;
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
            let hi_idx = cs.hi_idx;
            call_sites_rev.push(CallSite {
                marker: cs.marker,
                hi_idx: cs.hi_idx,
                a1: cs.a1,
                a2: cs.a2,
                a1_freq: cs.a1_freq,
                a2_freq: cs.a2_freq,
                pbwt_len_a1: cs.pbwt_len_a1,
                pbwt_len_a2: cs.pbwt_len_a2,
                pbwt_density_a1: cs.pbwt_density_a1,
                pbwt_density_a2: cs.pbwt_density_a2,
                dist_morgans: dist as f32,
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

/// Segment constraints are stored compactly as per-marker allowed alleles.
///
/// We add homozygous constraints for any mapped allele (strong signal), and
/// add phased heterozygous constraints as an allowed-allele set. This avoids
/// materializing full reference bitmasks while preserving the same filtering
/// semantics for candidate haplotypes.

fn build_segment_mask<RefSpace>(
    sample_phase: &SamplePhase,
    hi_freq_to_orig: &[usize],
    start_hi: usize,
    end_hi: usize,
    hi_freq_gen_positions: &[f64],
    allele_freqs: Option<&[(f32, f32)]>,
    hard_threshold: f64,
    theta: f64,
    packed_ref: &PackedRefView<RefSpace>,
) -> CondensedSegment {
    let mut constraints: Vec<SegmentConstraint> = Vec::new();
    let mut any_constraint = false;
    let len_morgans = if end_hi > start_hi && hi_freq_gen_positions.len() > 1 {
        let start_pos = hi_freq_gen_positions
            .get(start_hi)
            .copied()
            .unwrap_or(0.0);
        let end_pos = hi_freq_gen_positions
            .get(end_hi.saturating_sub(1))
            .copied()
            .unwrap_or(start_pos);
        (end_pos - start_pos).abs() as f32
    } else {
        0.0
    };

    for hi_idx in start_hi..end_hi {
        let orig_m = hi_freq_to_orig[hi_idx];
        let a1 = sample_phase.allele1(orig_m);
        let a2 = sample_phase.allele2(orig_m);
        if a1 == 255 || a2 == 255 {
            continue;
        }
        if a1 == a2 {
            let pi = allele_freqs
                .and_then(|af| af.get(hi_idx))
                .map(|&(f0, f1)| if a1 == 0 { f0 } else { f1 })
                .unwrap_or(0.5)
                .max(1e-9) as f64;
            let mismatch_cost = -((theta * pi).max(1e-12)).ln();
            if mismatch_cost >= hard_threshold && packed_ref.can_map_targ_allele(orig_m, a1) {
                constraints.push(SegmentConstraint {
                    marker: MarkerIdx::new(orig_m as u32),
                    alleles: [a1, a1],
                    n_alleles: 1,
                });
                any_constraint = true;
            }
        }
    }

    CondensedSegment {
        constraints,
        any_constraint,
        len_morgans,
    }
}
