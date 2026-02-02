//! Beam search phaser for condensed targets.

use crate::data::condensed::{CondensedTarget, CallSite};
use crate::data::ref_packed::{PackedRefView, mask_bit_is_set};
use crate::data::storage::sample_phase::SamplePhase;
use crate::data::MarkerIdx;
use crate::model::parameters::ModelParams;
use crate::model::reference_pbwt::{ReferencePbwt, RankBeam, PbwtStrictAllele};
use crate::data::alignment::MarkerAlignment;
use crate::data::storage::GenotypeMatrix;
use crate::data::storage::phase_state::Phased;
use crate::data::marker::AnyMarkerSpace;

#[derive(Clone, Copy, Debug)]
pub struct BeamConfig {
    pub beam_width: usize,
    pub switch_candidates: usize,
    pub inject_interval: usize,
    pub inject_k: usize,
    pub collapse_gap: i32,
    pub prune_tolerance: i32,
}

impl Default for BeamConfig {
    fn default() -> Self {
        Self {
            beam_width: 64,
            switch_candidates: 8,
            inject_interval: 8,
            inject_k: 16,
            collapse_gap: 5_000_000, // in fixed-point cost units
            prune_tolerance: 30_000_000,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BeamCosts {
    pub p_err: f64,
    pub recomb_intensity: f64,
}

impl BeamCosts {
    pub fn from_params(params: &ModelParams) -> Self {
        let p_err = params.p_mismatch.max(1e-9).min(1.0 - 1e-9);
        Self {
            p_err: p_err as f64,
            recomb_intensity: params.recomb_intensity.max(1e-12) as f64,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BackPtr {
    pub prev: u32,
    pub swapped: bool,
}

#[derive(Clone, Debug)]
pub struct BeamPath {
    pub hap1: usize,
    pub hap2: usize,
    pub cluster1: u16,
    pub cluster2: u16,
    pub score: i32,
    pub last_swapped: bool,
    pub history_bits: u64,
    pub history_len: u8,
    pub prev_idx: u32,
    pub prev_swapped: bool,
}

#[derive(Clone, Debug)]
pub struct BeamPosteriors {
    pub decisions: Vec<bool>,
    pub p_swapped: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct ActivePool {
    n_ref: usize,
    list: Vec<usize>,
    bitset: Vec<u64>,
    pbwt_cluster0: Vec<u16>,
    pbwt_cluster1: Vec<u16>,
    pbwt_cluster_size0: Vec<f32>,
    pbwt_cluster_size1: Vec<f32>,
    pbwt_match_len0: Vec<f32>,
    pbwt_match_len1: Vec<f32>,
    pbwt_version0: Vec<u32>,
    pbwt_version1: Vec<u32>,
}

impl ActivePool {
    pub fn new(n_ref: usize) -> Self {
        let n_words = (n_ref + 63) / 64;
        Self {
            n_ref,
            list: Vec::new(),
            bitset: vec![0u64; n_words],
            pbwt_cluster0: vec![u16::MAX; n_ref],
            pbwt_cluster1: vec![u16::MAX; n_ref],
            pbwt_cluster_size0: vec![0.0; n_ref],
            pbwt_cluster_size1: vec![0.0; n_ref],
            pbwt_match_len0: vec![0.0; n_ref],
            pbwt_match_len1: vec![0.0; n_ref],
            pbwt_version0: vec![u32::MAX; n_ref],
            pbwt_version1: vec![u32::MAX; n_ref],
        }
    }

    pub fn add(&mut self, hap: usize) {
        if hap >= self.n_ref {
            return;
        }
        let w = hap / 64;
        let b = hap % 64;
        if ((self.bitset[w] >> b) & 1) != 0 {
            return;
        }
        self.bitset[w] |= 1u64 << b;
        self.list.push(hap);
    }

    pub fn list(&self) -> &[usize] {
        &self.list
    }

    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }

    pub fn promote(&mut self, hap: usize) {
        if hap >= self.n_ref {
            return;
        }
        let w = hap / 64;
        let b = hap % 64;
        if ((self.bitset[w] >> b) & 1) == 0 {
            self.add(hap);
            return;
        }
        if let Some(pos) = self.list.iter().position(|&h| h == hap) {
            if pos + 1 != self.list.len() {
                let h = self.list.remove(pos);
                self.list.push(h);
            }
        }
    }

    #[inline]
    pub fn set_pbwt_meta(
        &mut self,
        allele: u8,
        hap: usize,
        cluster_id: u16,
        cluster_size: f32,
        match_len_markers: f32,
        version: u32,
    ) {
        if hap >= self.n_ref {
            return;
        }
        if allele == 0 {
            self.pbwt_cluster0[hap] = cluster_id;
            self.pbwt_cluster_size0[hap] = cluster_size;
            self.pbwt_match_len0[hap] = match_len_markers;
            self.pbwt_version0[hap] = version;
        } else if allele == 1 {
            self.pbwt_cluster1[hap] = cluster_id;
            self.pbwt_cluster_size1[hap] = cluster_size;
            self.pbwt_match_len1[hap] = match_len_markers;
            self.pbwt_version1[hap] = version;
        }
    }

    #[inline]
    pub fn pbwt_meta(&self, allele: u8, hap: usize, version: u32) -> Option<PbwtMeta> {
        if hap >= self.n_ref {
            return None;
        }
        if allele == 0 {
            if self.pbwt_version0[hap] != version {
                return None;
            }
            let cluster_id = self.pbwt_cluster0[hap];
            if cluster_id == u16::MAX {
                return None;
            }
            return Some(PbwtMeta {
                cluster_id,
                cluster_size: self.pbwt_cluster_size0[hap],
                match_len_markers: self.pbwt_match_len0[hap],
            });
        }
        if allele == 1 {
            if self.pbwt_version1[hap] != version {
                return None;
            }
            let cluster_id = self.pbwt_cluster1[hap];
            if cluster_id == u16::MAX {
                return None;
            }
            return Some(PbwtMeta {
                cluster_id,
                cluster_size: self.pbwt_cluster_size1[hap],
                match_len_markers: self.pbwt_match_len1[hap],
            });
        }
        None
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PbwtMeta {
    pub cluster_id: u16,
    pub cluster_size: f32,
    pub match_len_markers: f32,
}

pub trait BeamInjector {
    fn maybe_inject(&mut self, call_site_idx: usize, hi_idx: usize, marker: MarkerIdx, active_pool: &mut ActivePool);
}


/// PBWT-based dynamic injection (per marker).
pub struct PbwtBeamIndex {
    pub donor_meta0: Vec<Option<Vec<PbwtDonorMeta>>>, // hi-freq marker idx -> donor meta for allele 0
    pub donor_meta1: Vec<Option<Vec<PbwtDonorMeta>>>, // hi-freq marker idx -> donor meta for allele 1
    pub inject_interval: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct PbwtDonorMeta {
    pub hap: u32,
    pub cluster_id: u16,
    pub cluster_size: f32,
    pub match_len_markers: f32,
}

impl PbwtBeamIndex {
    pub fn build<RefSpace>(
        ref_gt: &GenotypeMatrix<Phased, RefSpace>,
        alignment: &MarkerAlignment<AnyMarkerSpace, RefSpace>,
        hi_freq_to_orig: &[usize],
        k: usize,
        inject_interval: usize,
    ) -> Self {
        let n_ref = ref_gt.n_haplotypes();
        let mut pbwt = ReferencePbwt::new(n_ref);
        let mut donor_meta0: Vec<Option<Vec<PbwtDonorMeta>>> = Vec::with_capacity(hi_freq_to_orig.len());
        let mut donor_meta1: Vec<Option<Vec<PbwtDonorMeta>>> = Vec::with_capacity(hi_freq_to_orig.len());

        let mut ref_alleles: Vec<u8> = vec![0u8; n_ref];
        for (hi_idx, &orig_m) in hi_freq_to_orig.iter().enumerate() {
            if inject_interval == 0 || (hi_idx % inject_interval) != 0 {
                donor_meta0.push(None);
                donor_meta1.push(None);
                continue;
            }
            let r_idx = match alignment.target_to_ref.get(orig_m).and_then(|v| *v) {
                Some(r) => r,
                None => {
                    donor_meta0.push(None);
                    donor_meta1.push(None);
                    continue;
                }
            };
            let marker = ref_gt.marker(r_idx);
            let n_alleles = marker.n_alleles();
            if n_alleles != 2 {
                donor_meta0.push(None);
                donor_meta1.push(None);
                continue;
            }

            let col = ref_gt.column(r_idx);
            for h in 0..n_ref {
                ref_alleles[h] = col.get(crate::data::haplotype::HapIdx::new(h as u32));
            }

            let mapping = alignment.allele_mappings.get(orig_m).and_then(|v| v.clone());
            let (qa0, qa1) = if let Some(map) = mapping {
                let m0 = map.targ_to_ref.get(0).copied().unwrap_or(-1);
                let m1 = map.targ_to_ref.get(1).copied().unwrap_or(-1);
                let qa0 = if m0 >= 0 && m0 <= 1 {
                    PbwtStrictAllele::allele(m0 as u8).unwrap_or(PbwtStrictAllele::missing())
                } else {
                    PbwtStrictAllele::missing()
                };
                let qa1 = if m1 >= 0 && m1 <= 1 {
                    PbwtStrictAllele::allele(m1 as u8).unwrap_or(PbwtStrictAllele::missing())
                } else {
                    PbwtStrictAllele::missing()
                };
                (qa0, qa1)
            } else {
                (PbwtStrictAllele::missing(), PbwtStrictAllele::missing())
            };

            let mut beams = [RankBeam::full(n_ref as u32), RankBeam::full(n_ref as u32)];
            pbwt.advance_with_beams_strict(
                &ref_alleles,
                n_alleles,
                hi_idx,
                &[qa0, qa1],
                &mut beams,
            );
            let mut d0: Vec<u32> = Vec::new();
            let mut d1: Vec<u32> = Vec::new();
            pbwt.select_donors_into(&beams[0], k, &mut d0);
            pbwt.select_donors_into(&beams[1], k, &mut d1);
            let mut union: Vec<u32> = Vec::with_capacity(d0.len() + d1.len());
            union.extend_from_slice(&d0);
            for &h in &d1 {
                if !union.contains(&h) {
                    union.push(h);
                }
            }
            let mut pos_lens: Vec<(u32, usize, f32)> = Vec::new();
            pbwt.collect_positions_and_lens(hi_idx, &union, &mut pos_lens);
            let select_longest = |donors: &mut Vec<u32>| {
                donors.sort_unstable_by(|a, b| {
                    let la = pos_lens
                        .iter()
                        .find(|(h, _, _)| h == a)
                        .map(|(_, _, l)| *l)
                        .unwrap_or(0.0);
                    let lb = pos_lens
                        .iter()
                        .find(|(h, _, _)| h == b)
                        .map(|(_, _, l)| *l)
                        .unwrap_or(0.0);
                    lb.partial_cmp(&la).unwrap_or(std::cmp::Ordering::Equal)
                });
                if donors.len() > k {
                    donors.truncate(k);
                }
            };
            select_longest(&mut d0);
            select_longest(&mut d1);
            let meta0 = build_donor_meta(&d0, &beams[0], &pos_lens);
            let meta1 = build_donor_meta(&d1, &beams[1], &pos_lens);
            donor_meta0.push(Some(meta0));
            donor_meta1.push(Some(meta1));
        }

        Self {
            donor_meta0,
            donor_meta1,
            inject_interval,
        }
    }

    pub fn stats_for_hi(&self, hi_idx: usize) -> (f32, f32, f32, f32) {
        let mut idx = hi_idx;
        if idx >= self.donor_meta0.len() {
            return (0.0, 0.0, 0.0, 0.0);
        }
        if self.donor_meta0[idx].is_none() && self.inject_interval > 0 {
            idx = hi_idx - (hi_idx % self.inject_interval);
        }
        let (len0, den0) = mean_meta(self.donor_meta0.get(idx).and_then(|v| v.as_ref()));
        let (len1, den1) = mean_meta(self.donor_meta1.get(idx).and_then(|v| v.as_ref()));
        (len0, len1, den0, den1)
    }
}

fn find_cluster(beam: &RankBeam, pos: usize) -> (u16, f32) {
    for (idx, &(l, r)) in beam.intervals().iter().enumerate() {
        let l = l as usize;
        let r = r as usize;
        if pos >= l && pos < r {
            let size = (r - l) as f32;
            return (idx as u16, size);
        }
    }
    (u16::MAX, 0.0)
}

fn build_donor_meta(
    donors: &[u32],
    beam: &RankBeam,
    pos_lens: &[(u32, usize, f32)],
) -> Vec<PbwtDonorMeta> {
    let mut out = Vec::with_capacity(donors.len());
    for &hap in donors {
        if let Some((_, pos, len)) = pos_lens.iter().find(|(h, _, _)| *h == hap).copied() {
            let (cluster_id, cluster_size) = find_cluster(beam, pos);
            out.push(PbwtDonorMeta {
                hap,
                cluster_id,
                cluster_size,
                match_len_markers: len,
            });
        } else {
            out.push(PbwtDonorMeta {
                hap,
                cluster_id: u16::MAX,
                cluster_size: 0.0,
                match_len_markers: 0.0,
            });
        }
    }
    out
}

fn mean_meta(meta: Option<&Vec<PbwtDonorMeta>>) -> (f32, f32) {
    let Some(list) = meta else {
        return (0.0, 0.0);
    };
    if list.is_empty() {
        return (0.0, 0.0);
    }
    let mut sum_len = 0.0f32;
    let mut sum_den = 0.0f32;
    for m in list {
        sum_len += m.match_len_markers;
        sum_den += m.cluster_size;
    }
    let n = list.len() as f32;
    (sum_len / n, sum_den / n)
}

pub struct PbwtInjector<'a> {
    pub index: &'a PbwtBeamIndex,
    pub k: usize,
}

impl<'a> PbwtInjector<'a> {
    pub fn new(index: &'a PbwtBeamIndex, n_ref: usize, k: usize) -> Self {
        let _ = n_ref;
        Self { index, k }
    }
}

impl<'a> BeamInjector for PbwtInjector<'a> {
    fn maybe_inject(&mut self, call_site_idx: usize, hi_idx: usize, marker: MarkerIdx, active_pool: &mut ActivePool) {
        let _ = (call_site_idx, marker);
        if hi_idx >= self.index.donor_meta0.len() {
            return;
        }
        let mut idx = hi_idx;
        if self.index.donor_meta0[idx].is_none() && self.index.inject_interval > 0 {
            idx = hi_idx - (hi_idx % self.index.inject_interval);
        }
        let version = idx as u32;
        if let Some(list) = self.index.donor_meta0[idx].as_ref() {
            for m in list.iter().take(self.k) {
                let hap = m.hap as usize;
                active_pool.add(hap);
                active_pool.set_pbwt_meta(0, hap, m.cluster_id, m.cluster_size, m.match_len_markers, version);
            }
        }
        if let Some(list) = self.index.donor_meta1[idx].as_ref() {
            for m in list.iter().take(self.k) {
                let hap = m.hap as usize;
                active_pool.add(hap);
                active_pool.set_pbwt_meta(1, hap, m.cluster_id, m.cluster_size, m.match_len_markers, version);
            }
        }
    }
}

pub struct BeamPhaser<'a, RefSpace = AnyMarkerSpace> {
    config: BeamConfig,
    costs: BeamCosts,
    packed_ref: &'a PackedRefView<RefSpace>,
}

struct BeamScratch {
    hap1_candidates: Vec<(usize, i32)>,
    hap2_candidates: Vec<(usize, i32)>,
    hap1_allele: Vec<(usize, i32, i32)>,
    hap2_allele: Vec<(usize, i32, i32)>,
    spread: Vec<usize>,
}

impl BeamScratch {
    fn new(cap: usize) -> Self {
        Self {
            hap1_candidates: Vec::with_capacity(cap),
            hap2_candidates: Vec::with_capacity(cap),
            hap1_allele: Vec::with_capacity(cap),
            hap2_allele: Vec::with_capacity(cap),
            spread: Vec::with_capacity(cap),
        }
    }
}

impl<'a, RefSpace> BeamPhaser<'a, RefSpace> {
    pub fn new(packed_ref: &'a PackedRefView<RefSpace>, params: &ModelParams, config: BeamConfig) -> Self {
        Self {
            config,
            costs: BeamCosts::from_params(params),
            packed_ref,
        }
    }

    pub fn phase_sample<I: BeamInjector>(
        &self,
        condensed: &CondensedTarget,
        sample_phase: &mut SamplePhase,
        active_pool: &mut ActivePool,
        injector: &mut I,
    ) -> BeamPosteriors {
        if self.packed_ref.n_ref_haps() == 0 {
            return BeamPosteriors {
                decisions: Vec::new(),
                p_swapped: Vec::new(),
            };
        }
        if active_pool.is_empty() {
            active_pool.add(0);
        }

        let mut scratch = BeamScratch::new(self.config.switch_candidates.max(4));

        // Prefer an unfixed call site for initialization so both orientations are informative.
        let init_call = condensed
            .call_sites
            .iter()
            .find(|call| !call.fixed)
            .or_else(|| condensed.call_sites.first());
        let mut beam: Vec<BeamPath> = if let Some(call) = init_call {
            self.init_beam_with_alleles(active_pool, call.marker, call.a1, call.a2)
        } else {
            self.init_beam(active_pool)
        };

        let n_calls = condensed.call_sites.len();
        let mut backptrs: Vec<Vec<BackPtr>> = Vec::with_capacity(n_calls);
        let mut logsum_unswapped: Vec<f64> = vec![f64::NEG_INFINITY; n_calls];
        let mut logsum_swapped: Vec<f64> = vec![f64::NEG_INFINITY; n_calls];
        for i in 0..n_calls {
            let segment = &condensed.segments[i];
            let call = &condensed.call_sites[i];

            // Segment consistency repair.
            beam = self.apply_segment_constraints(
                &beam,
                segment,
                active_pool,
                call.switch_cost,
                &mut scratch,
            );
            if beam.is_empty() {
                beam = self.init_beam_with_alleles(active_pool, call.marker, call.a1, call.a2);
                if beam.is_empty() {
                    beam = self.init_beam(active_pool);
                }
            }

            // Dynamic injection on collapse or interval.
            let inject_interval = self.config.inject_interval;
            if inject_interval > 0 && (call.hi_idx % inject_interval) == 0 {
                injector.maybe_inject(i, call.hi_idx, call.marker, active_pool);
            } else if self.beam_collapsed(&beam) {
                injector.maybe_inject(i, call.hi_idx, call.marker, active_pool);
            }

            // Call site branching (phase decisions)
            let mut next: Vec<BeamPath> = Vec::with_capacity(self.config.beam_width * 2);
            let mut best_score = i32::MAX;
            for p in &beam {
                if p.score < best_score {
                    best_score = p.score;
                }
            }
            let cutoff = if self.config.prune_tolerance > 0 {
                best_score.saturating_add(self.config.prune_tolerance)
            } else {
                i32::MAX
            };
        for (path_idx, path) in beam.iter().enumerate() {
            self.expand_call_site(
                path,
                path_idx as u32,
                call,
                active_pool,
                &mut next,
                &mut logsum_unswapped,
                &mut logsum_swapped,
                i,
                cutoff,
                &mut scratch,
            );
        }
            self.prune_and_collapse(&mut next);
            let mut step_ptrs: Vec<BackPtr> = Vec::with_capacity(next.len());
            for p in &next {
                step_ptrs.push(BackPtr {
                    prev: p.prev_idx,
                    swapped: p.prev_swapped,
                });
            }
            backptrs.push(step_ptrs);
            beam = next;
        }

        // Apply trailing segment constraints
        if let Some(last_seg) = condensed.segments.get(n_calls) {
            beam = self.apply_segment_constraints(
                &beam,
                last_seg,
                active_pool,
                0,
                &mut scratch,
            );
            if beam.is_empty() {
                beam = self.init_beam(active_pool);
            }
        }

        // Pick best path
        if let Some((best_idx, _)) = beam.iter().enumerate().min_by_key(|(_, p)| p.score) {
            let mut phases = Vec::with_capacity(n_calls);
            let mut idx = best_idx;
            for step in (0..n_calls).rev() {
                if let Some(ptrs) = backptrs.get(step) {
                    if let Some(bp) = ptrs.get(idx) {
                        phases.push(bp.swapped);
                        idx = bp.prev as usize;
                        continue;
                    }
                }
                phases.push(false);
                idx = 0;
            }
            phases.reverse();
            let p_swapped = compute_swap_posteriors(&logsum_swapped, &logsum_unswapped);
            for (i, phase_swapped) in phases.iter().enumerate() {
                let call = &condensed.call_sites[i];
                let m = call.marker.as_usize();
                if *phase_swapped {
                    sample_phase.swap_alleles(m);
                }
                sample_phase.mark_phased(m);
                let p = p_swapped.get(i).copied().unwrap_or(0.5);
                let conf = if *phase_swapped { p } else { 1.0 - p };
                sample_phase.set_phase_confidence(m, conf);
            }
            return BeamPosteriors { decisions: phases, p_swapped };
        }
        BeamPosteriors { decisions: Vec::new(), p_swapped: vec![0.5; n_calls] }
    }

    fn init_beam(&self, active_pool: &ActivePool) -> Vec<BeamPath> {
        let mut beam = Vec::new();
        let list = active_pool.list();
        if list.is_empty() {
            return beam;
        }
        let k = (self.config.beam_width as f32).sqrt().ceil() as usize;
        let n = k.min(list.len()).max(1);
        let mut picks: Vec<usize> = Vec::with_capacity(n);
        let tail_n = n.min(list.len());
        let tail_start = list.len().saturating_sub(tail_n);
        picks.extend_from_slice(&list[tail_start..]);
        if picks.len() < n {
            let spread = sample_even(list, n);
            for h in spread {
                if !picks.contains(&h) {
                    picks.push(h);
                    if picks.len() >= n {
                        break;
                    }
                }
            }
        }
        for i in 0..picks.len() {
            for j in 0..picks.len() {
                let hap1 = picks[i];
                let hap2 = picks[j];
                beam.push(BeamPath {
                    hap1,
                    hap2,
                    cluster1: u16::MAX,
                    cluster2: u16::MAX,
                    score: 0,
                    last_swapped: false,
                    history_bits: 0,
                    history_len: 0,
                    prev_idx: 0,
                    prev_swapped: false,
                });
                if beam.len() >= self.config.beam_width {
                    return beam;
                }
            }
        }
        beam
    }

    /// Principled beam initialization: seed with haplotypes matching both alleles.
    /// This ensures the prior over orientations is uniform when evidence is symmetric.
    fn init_beam_with_alleles(
        &self,
        active_pool: &ActivePool,
        marker: MarkerIdx,
        a1: u8,
        a2: u8,
    ) -> Vec<BeamPath> {
        let list = active_pool.list();
        if list.is_empty() {
            return Vec::new();
        }

        let marker_idx = marker.as_usize();
        let mut match_a1: Vec<usize> = Vec::new();
        let mut match_a2: Vec<usize> = Vec::new();

        // Partition haplotypes by which allele they carry.
        for &h in list {
            if self.ref_allele_matches(marker_idx, h, a1) {
                match_a1.push(h);
            }
            if self.ref_allele_matches(marker_idx, h, a2) {
                match_a2.push(h);
            }
        }

        // Fallback: if no matches for an allele, use the full pool.
        if match_a1.is_empty() {
            match_a1 = list.to_vec();
        }
        if match_a2.is_empty() {
            match_a2 = list.to_vec();
        }

        let k = (self.config.beam_width as f32).sqrt().ceil() as usize;
        let n1 = k.min(match_a1.len()).max(1);
        let n2 = k.min(match_a2.len()).max(1);

        let picks_a1 = sample_even(&match_a1, n1);
        let picks_a2 = sample_even(&match_a2, n2);

        let mut beam = Vec::with_capacity(self.config.beam_width);
        let half_width = self.config.beam_width / 2;

        // Orientation 0|1: hap1 carries a1, hap2 carries a2.
        for &h1 in &picks_a1 {
            for &h2 in &picks_a2 {
                beam.push(BeamPath {
                    hap1: h1,
                    hap2: h2,
                    cluster1: u16::MAX,
                    cluster2: u16::MAX,
                    score: 0,
                    last_swapped: false,
                    history_bits: 0,
                    history_len: 0,
                    prev_idx: 0,
                    prev_swapped: false,
                });
                if beam.len() >= half_width {
                    break;
                }
            }
            if beam.len() >= half_width {
                break;
            }
        }

        // Orientation 1|0: hap1 carries a2, hap2 carries a1.
        for &h1 in &picks_a2 {
            for &h2 in &picks_a1 {
                beam.push(BeamPath {
                    hap1: h1,
                    hap2: h2,
                    cluster1: u16::MAX,
                    cluster2: u16::MAX,
                    score: 0,
                    last_swapped: true,
                    history_bits: 0,
                    history_len: 0,
                    prev_idx: 0,
                    prev_swapped: true,
                });
                if beam.len() >= self.config.beam_width {
                    break;
                }
            }
            if beam.len() >= self.config.beam_width {
                break;
            }
        }

        beam
    }

    fn apply_segment_constraints(
        &self,
        beam: &[BeamPath],
        segment: &crate::data::condensed::CondensedSegment,
        active_pool: &ActivePool,
        switch_cost: i32,
        scratch: &mut BeamScratch,
    ) -> Vec<BeamPath> {
        if !segment.any_constraint {
            return beam.to_vec();
        }
        let soft_segment = segment.len_morgans >= 0.001;
        let soft_penalty = switch_cost;
        let mut out: Vec<BeamPath> = Vec::with_capacity(beam.len());
        for path in beam {
            self.repair_hap_into(
                path.hap1,
                &segment.mask,
                active_pool,
                switch_cost,
                soft_segment,
                soft_penalty,
                &mut scratch.hap1_candidates,
            );
            self.repair_hap_into(
                path.hap2,
                &segment.mask,
                active_pool,
                switch_cost,
                soft_segment,
                soft_penalty,
                &mut scratch.hap2_candidates,
            );
            for (h1, c1) in scratch.hap1_candidates.iter() {
                for (h2, c2) in scratch.hap2_candidates.iter() {
                    out.push(BeamPath {
                        hap1: *h1,
                        hap2: *h2,
                        cluster1: u16::MAX,
                        cluster2: u16::MAX,
                        score: path.score + *c1 + *c2,
                        last_swapped: path.last_swapped,
                        history_bits: path.history_bits,
                        history_len: path.history_len,
                        prev_idx: path.prev_idx,
                        prev_swapped: path.prev_swapped,
                    });
                }
            }
        }
        self.prune_inplace(out)
    }

    fn repair_hap_into(
        &self,
        hap: usize,
        mask: &[u64],
        active_pool: &ActivePool,
        switch_cost: i32,
        soft_segment: bool,
        soft_penalty: i32,
        out: &mut Vec<(usize, i32)>,
    ) {
        out.clear();
        let hap_ok = mask_bit_is_set(mask, hap);
        if hap_ok {
            out.push((hap, 0));
        } else if soft_segment {
            out.push((hap, soft_penalty));
        }
        // switch to most recently injected candidates first
        for &h in active_pool.list().iter().rev().take(self.config.switch_candidates) {
            if mask_bit_is_set(mask, h) {
                out.push((h, switch_cost));
            }
            if out.len() >= self.config.switch_candidates {
                break;
            }
        }
        if hap_ok && out.len() < 2 {
            // allow a limited exploratory switch even if current hap is consistent
            for &h in active_pool.list().iter().rev().take(2) {
                if h != hap && mask_bit_is_set(mask, h) {
                    out.push((h, switch_cost));
                }
                if out.len() >= self.config.switch_candidates {
                    break;
                }
            }
        }
        if out.is_empty() {
            // allow staying with penalty
            out.push((hap, switch_cost * 2));
        }
    }

    fn expand_call_site(
        &self,
        path: &BeamPath,
        parent_idx: u32,
        call: &CallSite,
        active_pool: &ActivePool,
        out: &mut Vec<BeamPath>,
        logsum_unswapped: &mut [f64],
        logsum_swapped: &mut [f64],
        call_idx: usize,
        cutoff: i32,
        scratch: &mut BeamScratch,
    ) {
        let a1 = call.a1;
        let a2 = call.a2;

        if call.fixed {
            self.expand_orientation(
                path,
                parent_idx,
                call,
                a1,
                a2,
                false,
                active_pool,
                out,
                logsum_unswapped,
                logsum_swapped,
                call_idx,
                cutoff,
                scratch,
            );
        } else {
            self.expand_orientation(
                path,
                parent_idx,
                call,
                a1,
                a2,
                false,
                active_pool,
                out,
                logsum_unswapped,
                logsum_swapped,
                call_idx,
                cutoff,
                scratch,
            );
            self.expand_orientation(
                path,
                parent_idx,
                call,
                a2,
                a1,
                true,
                active_pool,
                out,
                logsum_unswapped,
                logsum_swapped,
                call_idx,
                cutoff,
                scratch,
            );
        }
    }

    fn expand_orientation(
        &self,
        path: &BeamPath,
        parent_idx: u32,
        call: &CallSite,
        hap1_al: u8,
        hap2_al: u8,
        swapped: bool,
        active_pool: &ActivePool,
        out: &mut Vec<BeamPath>,
        logsum_unswapped: &mut [f64],
        logsum_swapped: &mut [f64],
        call_idx: usize,
        cutoff: i32,
        scratch: &mut BeamScratch,
    ) {
        // Get allele frequencies for TMRCA-aware scoring.
        let (hap1_freq, hap2_freq) = if swapped {
            (call.a2_freq, call.a1_freq)
        } else {
            (call.a1_freq, call.a2_freq)
        };
        let pbwt_version = if self.config.inject_interval > 0 {
            call.hi_idx - (call.hi_idx % self.config.inject_interval)
        } else {
            call.hi_idx
        };

        self.repair_hap_for_allele_into(
            path.hap1,
            call.marker,
            hap1_al,
            hap1_freq,
            if swapped { call.pbwt_len_a2 } else { call.pbwt_len_a1 },
            if swapped { call.pbwt_density_a2 } else { call.pbwt_density_a1 },
            call.dist_morgans,
            call.pbwt_step_morgans,
            pbwt_version as u32,
            call.fixed,
            active_pool,
            &mut scratch.hap1_allele,
            &mut scratch.spread,
        );
        self.repair_hap_for_allele_into(
            path.hap2,
            call.marker,
            hap2_al,
            hap2_freq,
            if swapped { call.pbwt_len_a1 } else { call.pbwt_len_a2 },
            if swapped { call.pbwt_density_a1 } else { call.pbwt_density_a2 },
            call.dist_morgans,
            call.pbwt_step_morgans,
            pbwt_version as u32,
            call.fixed,
            active_pool,
            &mut scratch.hap2_allele,
            &mut scratch.spread,
        );
        for (h1, c1, e1) in scratch.hap1_allele.iter() {
            for (h2, c2, e2) in scratch.hap2_allele.iter() {
                let score_no_flip = path.score + *c1 + *c2 + *e1 + *e2;
                let flip_penalty = if call_idx == 0 {
                    0
                } else if swapped != path.last_swapped {
                    call.flip_cost
                } else {
                    0
                };
                let score = score_no_flip + flip_penalty;
                let logp = -(score as f64) / 1_000_000.0;
                if swapped {
                    logsum_swapped[call_idx] = logaddexp(logsum_swapped[call_idx], logp);
                } else {
                    logsum_unswapped[call_idx] = logaddexp(logsum_unswapped[call_idx], logp);
                }
                let (history_bits, history_len) = push_history_bits(path.history_bits, path.history_len, swapped);
                if score <= cutoff {
                    let c1 = active_pool
                        .pbwt_meta(hap1_al, *h1, pbwt_version as u32)
                        .map(|m| m.cluster_id)
                        .unwrap_or(u16::MAX);
                    let c2 = active_pool
                        .pbwt_meta(hap2_al, *h2, pbwt_version as u32)
                        .map(|m| m.cluster_id)
                        .unwrap_or(u16::MAX);
                    out.push(BeamPath {
                        hap1: *h1,
                        hap2: *h2,
                        cluster1: c1,
                        cluster2: c2,
                        score,
                        last_swapped: swapped,
                        history_bits,
                        history_len,
                        prev_idx: parent_idx,
                        prev_swapped: swapped,
                    });
                }
            }
        }
    }

    fn repair_hap_for_allele_into(
        &self,
        hap: usize,
        marker: MarkerIdx,
        targ_allele: u8,
        targ_freq: f32,
        pbwt_match_len: f32,
        pbwt_density: f32,
        dist_morgans: f32,
        pbwt_step_morgans: f32,
        pbwt_version: u32,
        fixed: bool,
        active_pool: &ActivePool,
        out: &mut Vec<(usize, i32, i32)>,
        spread: &mut Vec<usize>,
    ) {
        out.clear();
        let marker_idx = marker.as_usize();

        // Emission costs (Li–Stephens mixture, explicit rare-allele behavior):
        //   P(X|match) = (1-θ) + θ·π(X)
        //   P(X|mismatch) = θ·π(X)
        // So as π(X) → 0, mismatches become exponentially costly while matches
        // retain the (1-θ) mass.
        let pi = targ_freq.max(1e-9) as f64;
        let theta = self.costs.p_err.max(1e-9).min(1.0 - 1e-9);
        let p_match = (1.0 - theta) + theta * pi;
        let p_mismatch = theta * pi;
        let effective_match_cost = (-(p_match.ln()) * 1_000_000.0)
            .round()
            .clamp(i32::MIN as f64, i32::MAX as f64) as i32;
        let effective_mismatch_cost = (-(p_mismatch.ln()) * 1_000_000.0)
            .round()
            .clamp(i32::MIN as f64, i32::MAX as f64) as i32;

        let allele = targ_allele;
        let meta_curr = active_pool.pbwt_meta(allele, hap, pbwt_version);
        let (match_len_morgans, cluster_size, cluster_id) = if let Some(meta) = meta_curr {
            (
                (meta.match_len_markers * pbwt_step_morgans).max(0.0),
                meta.cluster_size.max(0.0),
                meta.cluster_id,
            )
        } else {
            (pbwt_match_len.max(0.0), pbwt_density.max(0.0), u16::MAX)
        };
        let pbwt_switch_cost = self.pbwt_switch_cost(match_len_morgans, cluster_size, dist_morgans, false);
        // Note: pbwt_switch_cost already incorporates genetic distance via the
        // coalescent stay/switch model, so we do not add the global switch_cost
        // here to avoid double-counting recombination distance.

        let matches = self.ref_allele_matches(marker_idx, hap, targ_allele);
        if matches {
            out.push((hap, 0, effective_match_cost));
            // also allow a limited switch to a strong candidate for future-proofing
            for &h in active_pool.list().iter().rev().take(1) {
                if h != hap && self.ref_allele_matches(marker_idx, h, targ_allele) {
                    let same_cluster = active_pool
                        .pbwt_meta(allele, h, pbwt_version)
                        .map(|m| m.cluster_id == cluster_id)
                        .unwrap_or(false);
                    let effective_switch_cost = if same_cluster {
                        0
                    } else {
                        pbwt_switch_cost
                    };
                    out.push((h, effective_switch_cost, effective_match_cost));
                }
            }
            return;
        }
        if fixed {
            // For phased anchors, scan the full pool to ensure we honor the fixed allele.
            for &h in active_pool.list().iter().rev() {
                if self.ref_allele_matches(marker_idx, h, targ_allele) {
                    let same_cluster = active_pool
                        .pbwt_meta(allele, h, pbwt_version)
                        .map(|m| m.cluster_id == cluster_id)
                        .unwrap_or(false);
                    let effective_switch_cost = if same_cluster {
                        0
                    } else {
                        pbwt_switch_cost
                    };
                    out.push((h, effective_switch_cost, effective_match_cost));
                    if out.len() >= self.config.switch_candidates {
                        break;
                    }
                }
            }
            if out.is_empty() {
                out.push((hap, 0, effective_mismatch_cost));
            }
            return;
        }
        // Try switching to matching haps
        for &h in active_pool.list().iter().rev().take(self.config.switch_candidates) {
            if self.ref_allele_matches(marker_idx, h, targ_allele) {
                let same_cluster = active_pool
                    .pbwt_meta(allele, h, pbwt_version)
                    .map(|m| m.cluster_id == cluster_id)
                    .unwrap_or(false);
                let effective_switch_cost = if same_cluster {
                    0
                } else {
                    pbwt_switch_cost
                };
                out.push((h, effective_switch_cost, effective_match_cost));
            }
            if out.len() >= self.config.switch_candidates {
                break;
            }
        }
        if out.len() < self.config.switch_candidates {
            sample_even_into(active_pool.list(), self.config.switch_candidates, spread);
            for &h in spread.iter() {
                if self.ref_allele_matches(marker_idx, h, targ_allele) {
                    let same_cluster = active_pool
                        .pbwt_meta(allele, h, pbwt_version)
                        .map(|m| m.cluster_id == cluster_id)
                        .unwrap_or(false);
                    let effective_switch_cost = if same_cluster {
                        0
                    } else {
                        pbwt_switch_cost
                    };
                    out.push((h, effective_switch_cost, effective_match_cost));
                }
                if out.len() >= self.config.switch_candidates {
                    break;
                }
            }
        }
        // Always allow staying with a mismatch cost to avoid forced switching.
        // Mismatch cost is MAF-dependent: rare allele mismatch is more surprising.
        out.push((hap, 0, effective_mismatch_cost));
    }

    #[inline]
    fn pbwt_switch_cost(
        &self,
        match_len_morgans: f32,
        density: f32,
        dist_morgans: f32,
        same_cluster: bool,
    ) -> i32 {
        // Coalescent-based stay probability from PBWT match length with
        // explicit recombination intensity scaling (rho = 4Ne in Morgans):
        //   prior: t ~ Gamma(α=2, β=1) over TMRCA (coalescent units)
        //   L | t ~ Exp(rate = rho * t)
        //   t | L ~ Gamma(α=2, β=1 + rho * L)
        //   P(stay | L, d) = E[exp(-rho * t * d)] = (β / (β + rho * d))^α
        // Switch odds = P(switch)/P(stay), cost = -ln(odds).
        let l = match_len_morgans.max(0.0) as f64;
        let d = dist_morgans.max(0.0) as f64;
        let rho = self.costs.recomb_intensity;
        let beta = 1.0 + rho * l;
        let denom = beta + rho * d;
        let p_stay = if denom > 0.0 {
            (beta / denom).powi(2)
        } else {
            0.0
        };
        let p_stay = p_stay.clamp(1e-12, 1.0 - 1e-12);
        let p_switch = (1.0 - p_stay).clamp(1e-12, 1.0 - 1e-12);
        let switch_odds = p_switch / p_stay;
        if same_cluster {
            return 0;
        }
        let mut cost = -switch_odds.ln();
        // Density prior (cluster multiplicity): picking one donor from K
        // equally plausible candidates adds +log K to the cost.
        let k = density.max(0.0) as f64;
        if k > 0.0 {
            cost += (1.0 + k).ln();
        }
        (cost * 1_000_000.0)
            .round()
            .clamp(i32::MIN as f64, i32::MAX as f64) as i32
    }

    #[inline]
    fn ref_allele_matches(&self, marker: usize, hap: usize, targ_allele: u8) -> bool {
        match self.packed_ref.ref_allele_targ(marker, hap) {
            Some(a) => a == targ_allele,
            None => false,
        }
    }

    fn prune_inplace(&self, mut beam: Vec<BeamPath>) -> Vec<BeamPath> {
        self.prune_and_collapse(&mut beam);
        beam
    }

    fn prune_and_collapse(&self, beam: &mut Vec<BeamPath>) {
        if beam.is_empty() {
            return;
        }
        // Prune by tolerance relative to best score.
        let mut best = i32::MAX;
        for p in beam.iter() {
            if p.score < best {
                best = p.score;
            }
        }
        let cutoff = if self.config.prune_tolerance > 0 {
            best.saturating_add(self.config.prune_tolerance)
        } else {
            i32::MAX
        };
        beam.retain(|p| p.score <= cutoff);
        if beam.is_empty() {
            return;
        }

        // Collapse identical states (hap1, hap2, history fingerprint, last_swapped).
        let cluster_key = |cluster: u16, hap: usize| -> u32 {
            if cluster != u16::MAX {
                cluster as u32
            } else {
                0x8000_0000u32 | (hap as u32 & 0x7FFF_FFFF)
            }
        };
        beam.sort_unstable_by(|a, b| {
            let a1 = cluster_key(a.cluster1, a.hap1);
            let b1 = cluster_key(b.cluster1, b.hap1);
            let a2 = cluster_key(a.cluster2, a.hap2);
            let b2 = cluster_key(b.cluster2, b.hap2);
            a1.cmp(&b1)
                .then(a2.cmp(&b2))
                .then(a.history_bits.cmp(&b.history_bits))
                .then(a.history_len.cmp(&b.history_len))
                .then(a.last_swapped.cmp(&b.last_swapped))
                .then(a.score.cmp(&b.score))
        });
        let mut write = 1usize;
        for i in 1..beam.len() {
            let prev = &beam[write - 1];
            let curr = &beam[i];
            let same = cluster_key(prev.cluster1, prev.hap1) == cluster_key(curr.cluster1, curr.hap1)
                && cluster_key(prev.cluster2, prev.hap2) == cluster_key(curr.cluster2, curr.hap2)
                && prev.history_bits == curr.history_bits
                && prev.history_len == curr.history_len
                && prev.last_swapped == curr.last_swapped;
            if !same {
                if write != i {
                    beam[write] = curr.clone();
                }
                write += 1;
            }
        }
        beam.truncate(write);

        if beam.len() > self.config.beam_width {
            let k = self.config.beam_width;
            beam.select_nth_unstable_by(k, |a, b| a.score.cmp(&b.score));
            beam.truncate(k);
        }
    }

    fn beam_collapsed(&self, beam: &[BeamPath]) -> bool {
        if beam.len() < 2 {
            return true;
        }
        let mut best = i32::MAX;
        let mut worst = i32::MIN;
        for p in beam {
            if p.score < best { best = p.score; }
            if p.score > worst { worst = p.score; }
        }
        (worst - best) < self.config.collapse_gap
    }
}

fn sample_even(list: &[usize], n: usize) -> Vec<usize> {
    if list.is_empty() || n == 0 {
        return Vec::new();
    }
    let n = n.min(list.len());
    if n == list.len() {
        return list.to_vec();
    }
    let step = list.len() as f64 / n as f64;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let idx = (i as f64 * step).floor() as usize;
        out.push(list[idx.min(list.len() - 1)]);
    }
    out
}

fn sample_even_into(list: &[usize], n: usize, out: &mut Vec<usize>) {
    out.clear();
    if list.is_empty() || n == 0 {
        return;
    }
    let n = n.min(list.len());
    if n == list.len() {
        out.extend_from_slice(list);
        return;
    }
    let step = list.len() as f64 / n as f64;
    out.reserve(n);
    for i in 0..n {
        let idx = (i as f64 * step).floor() as usize;
        out.push(list[idx.min(list.len() - 1)]);
    }
}

fn push_history_bits(prev_bits: u64, prev_len: u8, swapped: bool) -> (u64, u8) {
    const HISTORY_BITS: u8 = 32;
    let bit = if swapped { 1u64 } else { 0u64 };
    let bits = ((prev_bits << 1) | bit) & ((1u64 << HISTORY_BITS) - 1);
    let len = if prev_len < HISTORY_BITS { prev_len + 1 } else { HISTORY_BITS };
    (bits, len)
}

#[inline]
fn logaddexp(a: f64, b: f64) -> f64 {
    if a.is_infinite() && a.is_sign_negative() {
        return b;
    }
    if b.is_infinite() && b.is_sign_negative() {
        return a;
    }
    let m = if a > b { a } else { b };
    m + ((a - m).exp() + (b - m).exp()).ln()
}

fn compute_swap_posteriors(logsum_swapped: &[f64], logsum_unswapped: &[f64]) -> Vec<f32> {
    let mut out = Vec::with_capacity(logsum_swapped.len());
    for i in 0..logsum_swapped.len() {
        let ls = logsum_swapped[i];
        let lu = logsum_unswapped.get(i).copied().unwrap_or(f64::NEG_INFINITY);
        if ls.is_infinite() && ls.is_sign_negative() {
            if lu.is_infinite() && lu.is_sign_negative() {
                out.push(0.5);
            } else {
                out.push(0.0);
            }
            continue;
        }
        if lu.is_infinite() && lu.is_sign_negative() {
            out.push(1.0);
            continue;
        }
        let m = if ls > lu { ls } else { lu };
        let ps = (ls - m).exp();
        let pu = (lu - m).exp();
        let denom = ps + pu;
        if denom <= 0.0 {
            out.push(0.5);
        } else {
            out.push((ps / denom).clamp(0.0, 1.0) as f32);
        }
    }
    out
}
