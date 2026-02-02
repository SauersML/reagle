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
    pub pbwt_len_weight: i32,
    pub pbwt_density_weight: i32,
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
            // Fixed-point weights for PBWT-derived terms (ln(1+x) scaled by 1e6).
            pbwt_len_weight: 200_000,
            pbwt_density_weight: 100_000,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BeamCosts {
    pub match_cost: i32,
    pub mismatch_cost: i32,
}

impl BeamCosts {
    pub fn from_params(params: &ModelParams) -> Self {
        let p_err = params.p_mismatch.max(1e-9).min(1.0 - 1e-9);
        let p_ok = (1.0 - p_err).max(1e-9);
        let match_cost = (-p_ok.ln() * 1_000_000.0).round() as i32;
        let mismatch_cost = (-p_err.ln() * 1_000_000.0).round() as i32;
        Self {
            match_cost,
            mismatch_cost,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HistoryIdx(pub u32);

#[derive(Clone, Copy, Debug)]
pub struct HistoryNode {
    pub parent: HistoryIdx,
    pub phase: bool,
}

#[derive(Clone, Debug)]
pub struct HistoryTrie {
    nodes: Vec<HistoryNode>,
}

impl HistoryTrie {
    pub fn new() -> Self {
        Self {
            nodes: vec![HistoryNode { parent: HistoryIdx(0), phase: false }],
        }
    }

    pub fn push(&mut self, parent: HistoryIdx, phase: bool) -> HistoryIdx {
        let idx = self.nodes.len() as u32;
        self.nodes.push(HistoryNode { parent, phase });
        HistoryIdx(idx)
    }

    pub fn reconstruct(&self, mut idx: HistoryIdx, out: &mut Vec<bool>) {
        out.clear();
        while idx.0 != 0 {
            let node = self.nodes[idx.0 as usize];
            out.push(node.phase);
            idx = node.parent;
        }
        out.reverse();
    }
}

#[derive(Clone, Debug)]
pub struct BeamPath {
    pub hap1: usize,
    pub hap2: usize,
    pub score: i32,
    pub history: HistoryIdx,
    pub last_swapped: bool,
    pub history_bits: u64,
    pub history_len: u8,
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
}

impl ActivePool {
    pub fn new(n_ref: usize) -> Self {
        let n_words = (n_ref + 63) / 64;
        Self {
            n_ref,
            list: Vec::new(),
            bitset: vec![0u64; n_words],
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
}

pub trait BeamInjector {
    fn maybe_inject(&mut self, call_site_idx: usize, hi_idx: usize, marker: MarkerIdx, active_pool: &mut ActivePool);
}


/// PBWT-based dynamic injection (per marker).
pub struct PbwtBeamIndex {
    pub donors0: Vec<Option<Vec<u32>>>, // hi-freq marker idx -> donor hap ids for allele 0
    pub donors1: Vec<Option<Vec<u32>>>, // hi-freq marker idx -> donor hap ids for allele 1
    pub match_len0: Vec<Option<f32>>, // hi-freq marker idx -> mean match length for allele 0
    pub match_len1: Vec<Option<f32>>, // hi-freq marker idx -> mean match length for allele 1
    pub density0: Vec<Option<f32>>,   // hi-freq marker idx -> density proxy for allele 0
    pub density1: Vec<Option<f32>>,   // hi-freq marker idx -> density proxy for allele 1
    pub inject_interval: usize,
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
        let mut donors0: Vec<Option<Vec<u32>>> = Vec::with_capacity(hi_freq_to_orig.len());
        let mut donors1: Vec<Option<Vec<u32>>> = Vec::with_capacity(hi_freq_to_orig.len());
        let mut match_len0: Vec<Option<f32>> = Vec::with_capacity(hi_freq_to_orig.len());
        let mut match_len1: Vec<Option<f32>> = Vec::with_capacity(hi_freq_to_orig.len());
        let mut density0: Vec<Option<f32>> = Vec::with_capacity(hi_freq_to_orig.len());
        let mut density1: Vec<Option<f32>> = Vec::with_capacity(hi_freq_to_orig.len());

        let mut ref_alleles: Vec<u8> = vec![0u8; n_ref];
        for (hi_idx, &orig_m) in hi_freq_to_orig.iter().enumerate() {
            if inject_interval == 0 || (hi_idx % inject_interval) != 0 {
                donors0.push(None);
                donors1.push(None);
                match_len0.push(None);
                match_len1.push(None);
                density0.push(None);
                density1.push(None);
                continue;
            }
            let r_idx = match alignment.target_to_ref.get(orig_m).and_then(|v| *v) {
                Some(r) => r,
                None => {
                    donors0.push(None);
                    donors1.push(None);
                    match_len0.push(None);
                    match_len1.push(None);
                    density0.push(None);
                    density1.push(None);
                    continue;
                }
            };
            let marker = ref_gt.marker(r_idx);
            let n_alleles = marker.n_alleles();
            if n_alleles != 2 {
                donors0.push(None);
                donors1.push(None);
                match_len0.push(None);
                match_len1.push(None);
                density0.push(None);
                density1.push(None);
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
            donors0.push(Some(d0));
            donors1.push(Some(d1));

            let (ml0, ml1) = pbwt.mean_match_len_by_allele(&ref_alleles, hi_idx);
            match_len0.push(Some(ml0));
            match_len1.push(Some(ml1));
            let dens0 = beam_span(&beams[0]) as f32;
            let dens1 = beam_span(&beams[1]) as f32;
            density0.push(Some(dens0));
            density1.push(Some(dens1));
        }

        Self {
            donors0,
            donors1,
            match_len0,
            match_len1,
            density0,
            density1,
            inject_interval,
        }
    }

    pub fn stats_for_hi(&self, hi_idx: usize) -> (f32, f32, f32, f32) {
        let mut idx = hi_idx;
        if idx >= self.match_len0.len() {
            return (0.0, 0.0, 0.0, 0.0);
        }
        if self.match_len0[idx].is_none() && self.inject_interval > 0 {
            idx = hi_idx - (hi_idx % self.inject_interval);
        }
        let len0 = self.match_len0.get(idx).and_then(|v| *v).unwrap_or(0.0);
        let len1 = self.match_len1.get(idx).and_then(|v| *v).unwrap_or(0.0);
        let den0 = self.density0.get(idx).and_then(|v| *v).unwrap_or(0.0);
        let den1 = self.density1.get(idx).and_then(|v| *v).unwrap_or(0.0);
        (len0, len1, den0, den1)
    }
}

fn beam_span(beam: &RankBeam) -> u32 {
    let mut total = 0u32;
    for &(l, r) in beam.intervals() {
        if r > l {
            total = total.saturating_add(r - l);
        }
    }
    total
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
        if hi_idx >= self.index.donors0.len() {
            return;
        }
        let mut idx = hi_idx;
        if self.index.donors0[idx].is_none() && self.index.inject_interval > 0 {
            idx = hi_idx - (hi_idx % self.index.inject_interval);
        }
        if let Some(list) = self.index.donors0[idx].as_ref() {
            for &d in list.iter().take(self.k) {
                active_pool.add(d as usize);
            }
        }
        if let Some(list) = self.index.donors1[idx].as_ref() {
            for &d in list.iter().take(self.k) {
                active_pool.add(d as usize);
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

        let mut trie = HistoryTrie::new();
        let mut scratch = BeamScratch::new(self.config.switch_candidates.max(4));

        // Principled initialization: seed beam with both orientations at first call site.
        let mut beam: Vec<BeamPath> = if let Some(first_call) = condensed.call_sites.first() {
            self.init_beam_with_alleles(active_pool, first_call.marker, first_call.a1, first_call.a2)
        } else {
            self.init_beam(active_pool)
        };

        let n_calls = condensed.call_sites.len();
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
            for path in &beam {
                self.expand_call_site(
                    path,
                    call,
                    active_pool,
                    &mut trie,
                    &mut next,
                    &mut logsum_unswapped,
                    &mut logsum_swapped,
                    i,
                    &mut scratch,
                );
            }
            self.prune_and_collapse(&mut next);
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
        if let Some(best) = beam.iter().min_by_key(|p| p.score).cloned() {
            let mut phases = Vec::new();
            trie.reconstruct(best.history, &mut phases);
            if phases.len() < n_calls {
                phases.resize(n_calls, false);
            }
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
                    score: 0,
                    history: HistoryIdx(0),
                    last_swapped: false,
                    history_bits: 0,
                    history_len: 0,
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
                    score: 0,
                    history: HistoryIdx(0),
                    last_swapped: false,
                    history_bits: 0,
                    history_len: 0,
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
                    score: 0,
                    history: HistoryIdx(0),
                    last_swapped: true,
                    history_bits: 0,
                    history_len: 0,
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
        let mut out: Vec<BeamPath> = Vec::with_capacity(beam.len());
        for path in beam {
            self.repair_hap_into(
                path.hap1,
                &segment.mask,
                active_pool,
                switch_cost,
                &mut scratch.hap1_candidates,
            );
            self.repair_hap_into(
                path.hap2,
                &segment.mask,
                active_pool,
                switch_cost,
                &mut scratch.hap2_candidates,
            );
            for (h1, c1) in scratch.hap1_candidates.iter() {
                for (h2, c2) in scratch.hap2_candidates.iter() {
                    out.push(BeamPath {
                        hap1: *h1,
                        hap2: *h2,
                        score: path.score + *c1 + *c2,
                        history: path.history,
                        last_swapped: path.last_swapped,
                        history_bits: path.history_bits,
                        history_len: path.history_len,
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
        out: &mut Vec<(usize, i32)>,
    ) {
        out.clear();
        let hap_ok = mask_bit_is_set(mask, hap);
        if hap_ok {
            out.push((hap, 0));
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
        call: &CallSite,
        active_pool: &ActivePool,
        trie: &mut HistoryTrie,
        out: &mut Vec<BeamPath>,
        logsum_unswapped: &mut [f64],
        logsum_swapped: &mut [f64],
        call_idx: usize,
        scratch: &mut BeamScratch,
    ) {
        let a1 = call.a1;
        let a2 = call.a2;

        if call.fixed {
            self.expand_orientation(path, call, a1, a2, false, active_pool, trie, out, logsum_unswapped, logsum_swapped, call_idx, scratch);
        } else {
            self.expand_orientation(path, call, a1, a2, false, active_pool, trie, out, logsum_unswapped, logsum_swapped, call_idx, scratch);
            self.expand_orientation(path, call, a2, a1, true, active_pool, trie, out, logsum_unswapped, logsum_swapped, call_idx, scratch);
        }
    }

    fn expand_orientation(
        &self,
        path: &BeamPath,
        call: &CallSite,
        hap1_al: u8,
        hap2_al: u8,
        swapped: bool,
        active_pool: &ActivePool,
        trie: &mut HistoryTrie,
        out: &mut Vec<BeamPath>,
        logsum_unswapped: &mut [f64],
        logsum_swapped: &mut [f64],
        call_idx: usize,
        scratch: &mut BeamScratch,
    ) {
        // Get allele frequencies for TMRCA-aware scoring.
        let (hap1_freq, hap2_freq) = if swapped {
            (call.a2_freq, call.a1_freq)
        } else {
            (call.a1_freq, call.a2_freq)
        };

        self.repair_hap_for_allele_into(
            path.hap1,
            call.marker,
            hap1_al,
            hap1_freq,
            if swapped { call.pbwt_len_a2 } else { call.pbwt_len_a1 },
            if swapped { call.pbwt_density_a2 } else { call.pbwt_density_a1 },
            call.fixed,
            active_pool,
            call.switch_cost,
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
            call.fixed,
            active_pool,
            call.switch_cost,
            &mut scratch.hap2_allele,
            &mut scratch.spread,
        );
        for (h1, c1, e1) in scratch.hap1_allele.iter() {
            for (h2, c2, e2) in scratch.hap2_allele.iter() {
                let history = trie.push(path.history, swapped);
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
                out.push(BeamPath {
                    hap1: *h1,
                    hap2: *h2,
                    score,
                    history,
                    last_swapped: swapped,
                    history_bits,
                    history_len,
                });
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
        fixed: bool,
        active_pool: &ActivePool,
        switch_cost: i32,
        out: &mut Vec<(usize, i32, i32)>,
        spread: &mut Vec<usize>,
    ) {
        out.clear();
        let marker_idx = marker.as_usize();

        // LLR-based costs (optimal under Li-Stephens copying model):
        //
        // The log-likelihood ratio compares P(X|donor) vs P(X|null=population).
        // For emission at marker m with target allele X:
        //   LLR = log(P(X|donor)) - log(π(X))
        //
        // As costs (negative log-likelihood), we ADD the information content -log(π(X)):
        //   match_cost(m) = base_match + (-log(π(X)))
        //   mismatch_cost(m) = base_mismatch + (-log(π(X)))
        //
        // For rare alleles (small π): -log(π(X)) is large and positive, so:
        //   - Match cost increases slightly, but matches become more valuable relative to mismatches
        //   - Mismatch cost increases significantly, making rare allele errors very costly
        //
        // This is exactly what coalescent theory predicts: rare alleles are more genealogically
        // informative, so getting them wrong should be penalized more heavily.
        //
        // Use f64 to avoid overflow, then saturate to i32.
        let info_content = -(targ_freq.max(1e-9) as f64).ln() * 1_000_000.0;
        let effective_match_cost = ((self.costs.match_cost as f64) + info_content)
            .round()
            .clamp(i32::MIN as f64, i32::MAX as f64) as i32;
        let effective_mismatch_cost = ((self.costs.mismatch_cost as f64) + info_content)
            .round()
            .clamp(i32::MIN as f64, i32::MAX as f64) as i32;
        let pbwt_cost = self.pbwt_emission_cost(pbwt_match_len, pbwt_density);
        let effective_match_cost = effective_match_cost.saturating_add(pbwt_cost);
        let effective_mismatch_cost = effective_mismatch_cost.saturating_add(pbwt_cost);

        // Switch cost: recombination penalty is already encoded in switch_cost.
        // The optimal switch criterion compares LLR gain vs recombination penalty.
        // No additional MAF term needed here - it's in the emission costs.
        let effective_switch_cost = switch_cost;

        let matches = self.ref_allele_matches(marker_idx, hap, targ_allele);
        if matches {
            out.push((hap, 0, effective_match_cost));
            // also allow a limited switch to a strong candidate for future-proofing
            for &h in active_pool.list().iter().rev().take(1) {
                if h != hap && self.ref_allele_matches(marker_idx, h, targ_allele) {
                    out.push((h, effective_switch_cost, effective_match_cost));
                }
            }
            return;
        }
        if fixed {
            // For phased anchors, scan the full pool to ensure we honor the fixed allele.
            for &h in active_pool.list().iter().rev() {
                if self.ref_allele_matches(marker_idx, h, targ_allele) {
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
    fn pbwt_emission_cost(&self, match_len: f32, density: f32) -> i32 {
        let len_term = (1.0 + match_len.max(0.0) as f64).ln();
        let density_term = (1.0 + density.max(0.0) as f64).ln();
        let cost = (self.config.pbwt_density_weight as f64 * density_term)
            - (self.config.pbwt_len_weight as f64 * len_term);
        cost.round()
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
        beam.sort_unstable_by(|a, b| {
            a.hap1
                .cmp(&b.hap1)
                .then(a.hap2.cmp(&b.hap2))
                .then(a.history_bits.cmp(&b.history_bits))
                .then(a.history_len.cmp(&b.history_len))
                .then(a.last_swapped.cmp(&b.last_swapped))
                .then(a.score.cmp(&b.score))
        });
        let mut write = 1usize;
        for i in 1..beam.len() {
            let prev = &beam[write - 1];
            let curr = &beam[i];
            let same = prev.hap1 == curr.hap1
                && prev.hap2 == curr.hap2
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
