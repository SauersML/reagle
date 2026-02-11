//! Beam search phaser for condensed targets.

use crate::data::MarkerIdx;
use crate::data::alignment::MarkerAlignment;
use crate::data::condensed::{CallSite, CondensedTarget};
use crate::data::marker::AnyMarkerSpace;
use crate::data::ref_packed::PackedRefView;
use crate::data::storage::GenotypeMatrix;
use crate::data::storage::phase_state::Phased;
use crate::data::storage::sample_phase::SamplePhase;
use crate::model::parameters::ModelParams;
use crate::model::reference_pbwt::{PbwtStrictAllele, RankBeam, ReferencePbwt};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
pub struct BeamConfig {
    pub beam_width: usize,
    pub switch_candidates: usize,
    pub inject_interval: usize,
    pub inject_k: usize,
    pub active_pool_ttl: u32,
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
            active_pool_ttl: 0,
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

#[inline]
fn pack_backptr(prev: u32, swapped: bool) -> u16 {
    let p = (prev as u16) & 0x7FFF;
    p | ((swapped as u16) << 15)
}

#[inline]
fn unpack_backptr(packed: u16) -> BackPtr {
    BackPtr {
        prev: (packed & 0x7FFF) as u32,
        swapped: (packed & 0x8000) != 0,
    }
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
    last_seen: Vec<u32>,
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
            last_seen: vec![0u32; n_ref],
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

    pub fn touch(&mut self, hap: usize, version: u32) {
        if hap >= self.n_ref {
            return;
        }
        self.last_seen[hap] = version;
        self.add(hap);
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

    pub fn sweep(&mut self, version: u32, max_age: u32) {
        if self.list.is_empty() {
            return;
        }
        let mut write = 0usize;
        for i in 0..self.list.len() {
            let hap = self.list[i];
            let age = version.saturating_sub(self.last_seen[hap]);
            if age <= max_age {
                if write != i {
                    self.list[write] = hap;
                }
                write += 1;
            } else {
                let w = hap / 64;
                let b = hap % 64;
                self.bitset[w] &= !(1u64 << b);
            }
        }
        self.list.truncate(write);
    }

    #[inline]
    pub fn set_pbwt_meta(
        &mut self,
        allele: u8,
        hap: usize,
        cluster_id: u16,
        cluster_size: f32,
        match_len_morgans: f32,
        version: u32,
    ) {
        if hap >= self.n_ref {
            return;
        }
        if allele == 0 {
            self.pbwt_cluster0[hap] = cluster_id;
            self.pbwt_cluster_size0[hap] = cluster_size;
            self.pbwt_match_len0[hap] = match_len_morgans;
            self.pbwt_version0[hap] = version;
        } else if allele == 1 {
            self.pbwt_cluster1[hap] = cluster_id;
            self.pbwt_cluster_size1[hap] = cluster_size;
            self.pbwt_match_len1[hap] = match_len_morgans;
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
                match_len_morgans: self.pbwt_match_len0[hap],
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
                match_len_morgans: self.pbwt_match_len1[hap],
            });
        }
        None
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PbwtMeta {
    pub cluster_id: u16,
    pub cluster_size: f32,
    pub match_len_morgans: f32,
}

pub trait BeamInjector {
    fn maybe_inject(
        &mut self,
        call_site_idx: usize,
        hi_idx: usize,
        marker: MarkerIdx,
        active_pool: &mut ActivePool,
    );
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
    pub match_len_morgans: f32,
}

impl PbwtBeamIndex {
    pub fn build<RefSpace>(
        ref_gt: &GenotypeMatrix<Phased, RefSpace>,
        alignment: &MarkerAlignment<AnyMarkerSpace, RefSpace>,
        hi_freq_to_orig: &[usize],
        hi_freq_gen_positions: &[f64],
        k: usize,
        inject_interval: usize,
        recomb_intensity: f32,
    ) -> Self {
        let n_ref = ref_gt.n_haplotypes();
        let mut pbwt = ReferencePbwt::new(n_ref);
        let mut donor_meta0: Vec<Option<Vec<PbwtDonorMeta>>> =
            Vec::with_capacity(hi_freq_to_orig.len());
        let mut donor_meta1: Vec<Option<Vec<PbwtDonorMeta>>> =
            Vec::with_capacity(hi_freq_to_orig.len());

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

            let mapping = alignment
                .allele_mappings
                .get(orig_m)
                .and_then(|v| v.clone());
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
            let mut pos_lens: Vec<(u32, usize, i32)> = Vec::new();
            pbwt.collect_positions_and_lens(hi_idx, &union, &mut pos_lens);
            let gen_pos = hi_freq_gen_positions.get(hi_idx).copied().unwrap_or(0.0);
            let step_morgans = if hi_idx > 0 {
                (gen_pos - hi_freq_gen_positions[hi_idx - 1]).abs() / 100.0
            } else if hi_idx + 1 < hi_freq_gen_positions.len() {
                (hi_freq_gen_positions[hi_idx + 1] - gen_pos).abs() / 100.0
            } else {
                0.0
            };
            let select_best_by_transition = |donors: &mut Vec<u32>, beam: &RankBeam| {
                const EULER_MASCHERONI: f64 = 0.5772156649015329;
                let rho = recomb_intensity.max(1e-12) as f64;
                let d = (step_morgans.max(1e-12)) as f64;
                let donor_score = |hap: u32| -> f64 {
                    let Some((_, pos, start)) =
                        pos_lens.iter().find(|(h, _, _)| *h == hap).copied()
                    else {
                        return 0.0;
                    };
                    let start_idx = start.max(0) as usize;
                    let start_pos = hi_freq_gen_positions
                        .get(start_idx)
                        .copied()
                        .unwrap_or(gen_pos);
                    let len_morgans = ((gen_pos - start_pos).abs() / 100.0) as f64;
                    let (_, cluster_size) = find_cluster(beam, pos);
                    let k = cluster_size.max(0.0) as f64;
                    let h_k = if k >= 2.0 {
                        k.ln() + EULER_MASCHERONI
                    } else {
                        1.0
                    };
                    let l_eff = len_morgans / h_k.max(1.0);
                    let beta = 1.0 + rho * l_eff;
                    let denom = beta + rho * d;
                    if denom > 0.0 {
                        (beta / denom).powi(2)
                    } else {
                        0.0
                    }
                };
                donors.sort_unstable_by(|a, b| {
                    let sa = donor_score(*a);
                    let sb = donor_score(*b);
                    sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
                });
                if donors.len() > k {
                    donors.truncate(k);
                }
            };
            select_best_by_transition(&mut d0, &beams[0]);
            select_best_by_transition(&mut d1, &beams[1]);
            let meta0 = build_donor_meta(&d0, &beams[0], &pos_lens, gen_pos, hi_freq_gen_positions);
            let meta1 = build_donor_meta(&d1, &beams[1], &pos_lens, gen_pos, hi_freq_gen_positions);
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
    pos_lens: &[(u32, usize, i32)],
    gen_pos: f64,
    hi_freq_gen_positions: &[f64],
) -> Vec<PbwtDonorMeta> {
    let mut out = Vec::with_capacity(donors.len());
    for &hap in donors {
        if let Some((_, pos, start)) = pos_lens.iter().find(|(h, _, _)| *h == hap).copied() {
            let start_idx = start.max(0) as usize;
            let start_pos = hi_freq_gen_positions
                .get(start_idx)
                .copied()
                .unwrap_or(gen_pos);
            let len_morgans = ((gen_pos - start_pos).abs() / 100.0) as f32;
            let (cluster_id, cluster_size) = find_cluster(beam, pos);
            out.push(PbwtDonorMeta {
                hap,
                cluster_id,
                cluster_size,
                match_len_morgans: len_morgans,
            });
        } else {
            out.push(PbwtDonorMeta {
                hap,
                cluster_id: u16::MAX,
                cluster_size: 0.0,
                match_len_morgans: 0.0,
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
        sum_len += m.match_len_morgans;
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
    fn maybe_inject(
        &mut self,
        call_site_idx: usize,
        hi_idx: usize,
        marker: MarkerIdx,
        active_pool: &mut ActivePool,
    ) {
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
                active_pool.touch(hap, version);
                active_pool.set_pbwt_meta(
                    0,
                    hap,
                    m.cluster_id,
                    m.cluster_size,
                    m.match_len_morgans,
                    version,
                );
            }
        }
        if let Some(list) = self.index.donor_meta1[idx].as_ref() {
            for m in list.iter().take(self.k) {
                let hap = m.hap as usize;
                active_pool.touch(hap, version);
                active_pool.set_pbwt_meta(
                    1,
                    hap,
                    m.cluster_id,
                    m.cluster_size,
                    m.match_len_morgans,
                    version,
                );
            }
        }
    }
}

pub struct BeamPhaser<'a, RefSpace = AnyMarkerSpace> {
    config: BeamConfig,
    costs: BeamCosts,
    packed_ref: &'a PackedRefView<RefSpace>,
    lr_threshold: f32,
}

struct BeamScratch {
    hap1_candidates: Vec<(usize, i32)>,
    hap2_candidates: Vec<(usize, i32)>,
    hap1_allele: Vec<(usize, i32, i32)>,
    hap2_allele: Vec<(usize, i32, i32)>,
    spread: Vec<usize>,
    pool_alleles: Vec<u8>,
    switch_support: SwitchSupportCache,
}

struct SwitchSupportCache {
    marker_idx: usize,
    pbwt_version: u32,
    initialized: bool,
    global_match_counts: [usize; 2],
    cluster_match_counts0: HashMap<u16, usize>,
    cluster_match_counts1: HashMap<u16, usize>,
}

impl BeamScratch {
    fn new(cap: usize) -> Self {
        Self {
            hap1_candidates: Vec::with_capacity(cap),
            hap2_candidates: Vec::with_capacity(cap),
            hap1_allele: Vec::with_capacity(cap),
            hap2_allele: Vec::with_capacity(cap),
            spread: Vec::with_capacity(cap),
            pool_alleles: Vec::with_capacity(cap),
            switch_support: SwitchSupportCache::new(),
        }
    }
}

impl SwitchSupportCache {
    fn new() -> Self {
        Self {
            marker_idx: usize::MAX,
            pbwt_version: u32::MAX,
            initialized: false,
            global_match_counts: [1, 1],
            cluster_match_counts0: HashMap::new(),
            cluster_match_counts1: HashMap::new(),
        }
    }

    fn ensure_initialized(
        &mut self,
        marker_idx: usize,
        pbwt_version: u32,
        active_pool: &ActivePool,
        pool_alleles: &[u8],
    ) {
        if self.initialized && self.marker_idx == marker_idx && self.pbwt_version == pbwt_version {
            return;
        }
        self.initialized = true;
        self.marker_idx = marker_idx;
        self.pbwt_version = pbwt_version;
        self.global_match_counts = [0, 0];
        self.cluster_match_counts0.clear();
        self.cluster_match_counts1.clear();

        for (idx, &hap) in active_pool.list().iter().enumerate() {
            let allele = pool_alleles.get(idx).copied().unwrap_or(255);
            if allele == 0 {
                self.global_match_counts[0] = self.global_match_counts[0].saturating_add(1);
                if let Some(meta) = active_pool.pbwt_meta(0, hap, pbwt_version) {
                    let e = self
                        .cluster_match_counts0
                        .entry(meta.cluster_id)
                        .or_insert(0);
                    *e = e.saturating_add(1);
                }
            } else if allele == 1 {
                self.global_match_counts[1] = self.global_match_counts[1].saturating_add(1);
                if let Some(meta) = active_pool.pbwt_meta(1, hap, pbwt_version) {
                    let e = self
                        .cluster_match_counts1
                        .entry(meta.cluster_id)
                        .or_insert(0);
                    *e = e.saturating_add(1);
                }
            }
        }
        self.global_match_counts[0] = self.global_match_counts[0].max(1);
        self.global_match_counts[1] = self.global_match_counts[1].max(1);
    }

    #[inline]
    fn global_match_count(&self, allele: u8) -> usize {
        if allele == 0 {
            self.global_match_counts[0]
        } else {
            self.global_match_counts[1]
        }
    }

    #[inline]
    fn cluster_match_count(&self, allele: u8, cluster_id: u16) -> usize {
        if cluster_id == u16::MAX {
            return 1;
        }
        let count = if allele == 0 {
            self.cluster_match_counts0
                .get(&cluster_id)
                .copied()
                .unwrap_or(0)
        } else {
            self.cluster_match_counts1
                .get(&cluster_id)
                .copied()
                .unwrap_or(0)
        };
        count.max(1)
    }
}

impl<'a, RefSpace> BeamPhaser<'a, RefSpace> {
    pub fn new(
        packed_ref: &'a PackedRefView<RefSpace>,
        params: &ModelParams,
        config: BeamConfig,
    ) -> Self {
        Self {
            config,
            costs: BeamCosts::from_params(params),
            packed_ref,
            lr_threshold: params.initial_lr,
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
        let mut backptrs: Vec<Vec<u16>> = Vec::with_capacity(n_calls);
        let mut segment_ptrs: Vec<Vec<u32>> = Vec::with_capacity(n_calls);
        let mut logsum_unswapped: Vec<f64> = vec![f64::NEG_INFINITY; n_calls];
        let mut logsum_swapped: Vec<f64> = vec![f64::NEG_INFINITY; n_calls];
        for i in 0..n_calls {
            let segment = &condensed.segments[i];
            let call = &condensed.call_sites[i];

            // Segment consistency repair.
            let (mut constrained, mut seg_ptrs) =
                self.apply_segment_constraints_with_ptrs(&beam, segment, active_pool, &mut scratch);
            if constrained.is_empty() {
                constrained =
                    self.init_beam_with_alleles(active_pool, call.marker, call.a1, call.a2);
                if constrained.is_empty() {
                    constrained = self.init_beam(active_pool);
                }
                seg_ptrs = vec![0u32; constrained.len()];
            }
            beam = constrained;
            segment_ptrs.push(seg_ptrs);

            // Dynamic injection on collapse or interval.
            let inject_interval = self.config.inject_interval;
            if inject_interval > 0 && (call.hi_idx % inject_interval) == 0 {
                injector.maybe_inject(i, call.hi_idx, call.marker, active_pool);
            } else if self.beam_collapsed(&beam) {
                injector.maybe_inject(i, call.hi_idx, call.marker, active_pool);
            }
            let ttl = if self.config.active_pool_ttl > 0 {
                self.config.active_pool_ttl
            } else if inject_interval > 0 {
                (inject_interval.saturating_mul(4)) as u32
            } else {
                16
            };
            active_pool.sweep(call.hi_idx as u32, ttl);

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
            let mut step_ptrs: Vec<u16> = Vec::with_capacity(next.len());
            for p in &next {
                assert!(p.prev_idx <= 0x7FFF, "beam backptr overflow");
                step_ptrs.push(pack_backptr(p.prev_idx, p.prev_swapped));
            }
            backptrs.push(step_ptrs);
            beam = next;
        }

        // Apply trailing segment constraints
        let mut trailing_ptrs: Option<Vec<u32>> = None;
        if let Some(last_seg) = condensed.segments.get(n_calls) {
            let (mut constrained, mut seg_ptrs) = self.apply_segment_constraints_with_ptrs(
                &beam,
                last_seg,
                active_pool,
                &mut scratch,
            );
            if constrained.is_empty() {
                constrained = self.init_beam(active_pool);
                seg_ptrs = vec![0u32; constrained.len()];
            }
            beam = constrained;
            trailing_ptrs = Some(seg_ptrs);
        }

        // Pick best path
        if let Some((best_idx, _)) = beam.iter().enumerate().min_by_key(|(_, p)| p.score) {
            let mut phases = Vec::with_capacity(n_calls);
            let mut idx = best_idx;
            if let Some(ptrs) = trailing_ptrs.as_ref() {
                idx = ptrs.get(idx).copied().unwrap_or(0) as usize;
            }
            for step in (0..n_calls).rev() {
                if let Some(ptrs) = backptrs.get(step) {
                    if let Some(packed) = ptrs.get(idx) {
                        let bp = unpack_backptr(*packed);
                        phases.push(bp.swapped);
                        let prev_idx = bp.prev as usize;
                        let mapped_prev = segment_ptrs
                            .get(step)
                            .and_then(|m| m.get(prev_idx))
                            .copied()
                            .unwrap_or(0) as usize;
                        idx = mapped_prev;
                        continue;
                    }
                }
                phases.push(false);
                idx = 0;
            }
            phases.reverse();
            let p_swapped = self.compute_swap_posteriors_smoothed(
                &logsum_swapped,
                &logsum_unswapped,
                &condensed.call_sites,
            );
            let posterior_decisions =
                self.decode_posterior_phase_path(&p_swapped, &condensed.call_sites);
            if posterior_decisions.len() == phases.len() {
                phases = posterior_decisions;
            }
            let has_input_anchor = sample_phase.has_input_phase_anchor();
            for (i, phase_swapped) in phases.iter().enumerate() {
                let call = &condensed.call_sites[i];
                let m = call.marker.as_usize();
                let p = p_swapped.get(i).copied().unwrap_or(0.5);
                let conf = if *phase_swapped { p } else { 1.0 - p };
                if !has_input_anchor {
                    sample_phase.set_phase_confidence(m, 0.5);
                    continue;
                }
                let lr = if conf <= 0.0 {
                    0.0
                } else if conf >= 1.0 {
                    f32::INFINITY
                } else {
                    conf / (1.0 - conf)
                };
                if lr >= self.lr_threshold {
                    if *phase_swapped {
                        sample_phase.swap_alleles(m);
                    }
                    sample_phase.mark_phased(m);
                    sample_phase.set_phase_confidence(m, conf);
                } else {
                    sample_phase.set_phase_confidence(m, 0.5);
                }
            }
            return BeamPosteriors {
                decisions: phases,
                p_swapped,
            };
        }
        BeamPosteriors {
            decisions: Vec::new(),
            p_swapped: vec![0.5; n_calls],
        }
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

    fn apply_segment_constraints_with_ptrs(
        &self,
        beam: &[BeamPath],
        segment: &crate::data::condensed::CondensedSegment,
        active_pool: &ActivePool,
        scratch: &mut BeamScratch,
    ) -> (Vec<BeamPath>, Vec<u32>) {
        if !segment.any_constraint {
            let mut ptrs: Vec<u32> = Vec::with_capacity(beam.len());
            for i in 0..beam.len() {
                ptrs.push(i as u32);
            }
            return (beam.to_vec(), ptrs);
        }
        let soft_segment = segment.len_morgans >= 0.001;
        let switch_cost = self.segment_switch_cost(segment.len_morgans);
        let mut out: Vec<BeamPath> = Vec::with_capacity(beam.len());
        let mut ptrs: Vec<u32> = Vec::with_capacity(beam.len());
        for (src_idx, path) in beam.iter().enumerate() {
            self.repair_hap_into(
                path.hap1,
                &segment.constraints,
                active_pool,
                switch_cost,
                soft_segment,
                &mut scratch.hap1_candidates,
            );
            self.repair_hap_into(
                path.hap2,
                &segment.constraints,
                active_pool,
                switch_cost,
                soft_segment,
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
                    ptrs.push(src_idx as u32);
                }
            }
        }
        self.prune_inplace_with_ptrs(out, ptrs)
    }

    fn repair_hap_into(
        &self,
        hap: usize,
        constraints: &[crate::data::condensed::SegmentConstraint],
        active_pool: &ActivePool,
        switch_cost: i32,
        soft_segment: bool,
        out: &mut Vec<(usize, i32)>,
    ) {
        out.clear();
        let (hap_penalty, hap_ok) = hap_constraints_penalty(self.packed_ref, hap, constraints);
        if hap_ok {
            out.push((hap, 0));
        } else if soft_segment {
            out.push((hap, hap_penalty));
        }
        // switch to most recently injected candidates first
        for &h in active_pool
            .list()
            .iter()
            .rev()
            .take(self.config.switch_candidates)
        {
            if hap_constraints_penalty(self.packed_ref, h, constraints).1 {
                out.push((h, switch_cost));
            }
            if out.len() >= self.config.switch_candidates {
                break;
            }
        }
        if hap_ok && out.len() < 2 {
            // allow a limited exploratory switch even if current hap is consistent
            for &h in active_pool.list().iter().rev().take(2) {
                if h != hap && hap_constraints_penalty(self.packed_ref, h, constraints).1 {
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
        let marker_idx = call.marker.as_usize();
        let pbwt_version = if self.config.inject_interval > 0 {
            call.hi_idx - (call.hi_idx % self.config.inject_interval)
        } else {
            call.hi_idx
        } as u32;
        let mut pool_alleles = std::mem::take(&mut scratch.pool_alleles);
        self.fill_pool_alleles(marker_idx, active_pool, &mut pool_alleles);
        scratch.switch_support.ensure_initialized(
            marker_idx,
            pbwt_version,
            active_pool,
            &pool_alleles,
        );

        if call.fixed {
            self.expand_orientation(
                path,
                parent_idx,
                call,
                a1,
                a2,
                false,
                pbwt_version,
                active_pool,
                &pool_alleles,
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
                pbwt_version,
                active_pool,
                &pool_alleles,
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
                pbwt_version,
                active_pool,
                &pool_alleles,
                out,
                logsum_unswapped,
                logsum_swapped,
                call_idx,
                cutoff,
                scratch,
            );
        }
        scratch.pool_alleles = pool_alleles;
    }

    fn expand_orientation(
        &self,
        path: &BeamPath,
        parent_idx: u32,
        call: &CallSite,
        hap1_al: u8,
        hap2_al: u8,
        swapped: bool,
        pbwt_version: u32,
        active_pool: &ActivePool,
        pool_alleles: &[u8],
        out: &mut Vec<BeamPath>,
        logsum_unswapped: &mut [f64],
        logsum_swapped: &mut [f64],
        call_idx: usize,
        cutoff: i32,
        scratch: &mut BeamScratch,
    ) {
        self.repair_hap_for_allele_into(
            path.hap1,
            call.marker,
            hap1_al,
            if swapped {
                call.pbwt_len_morgans_a2
            } else {
                call.pbwt_len_morgans_a1
            },
            if swapped {
                call.pbwt_density_a2
            } else {
                call.pbwt_density_a1
            },
            call.dist_morgans,
            pbwt_version as u32,
            call.fixed,
            active_pool,
            pool_alleles,
            &scratch.switch_support,
            &mut scratch.hap1_allele,
            &mut scratch.spread,
        );
        self.repair_hap_for_allele_into(
            path.hap2,
            call.marker,
            hap2_al,
            if swapped {
                call.pbwt_len_morgans_a1
            } else {
                call.pbwt_len_morgans_a2
            },
            if swapped {
                call.pbwt_density_a1
            } else {
                call.pbwt_density_a2
            },
            call.dist_morgans,
            pbwt_version as u32,
            call.fixed,
            active_pool,
            pool_alleles,
            &scratch.switch_support,
            &mut scratch.hap2_allele,
            &mut scratch.spread,
        );
        for (h1, c1, e1) in scratch.hap1_allele.iter() {
            for (h2, c2, e2) in scratch.hap2_allele.iter() {
                let score_no_flip = path
                    .score
                    .saturating_add(*c1)
                    .saturating_add(*c2)
                    .saturating_add(*e1)
                    .saturating_add(*e2);
                let flip_penalty = if call_idx == 0 {
                    0
                } else if swapped != path.last_swapped {
                    call.flip_cost
                } else {
                    0
                };
                let score = score_no_flip.saturating_add(flip_penalty);
                let logp = -(score as f64) / 1_000_000.0;
                if swapped {
                    logsum_swapped[call_idx] = logaddexp(logsum_swapped[call_idx], logp);
                } else {
                    logsum_unswapped[call_idx] = logaddexp(logsum_unswapped[call_idx], logp);
                }
                let (history_bits, history_len) =
                    push_history_bits(path.history_bits, path.history_len, swapped);
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
        pbwt_match_len: f32,
        pbwt_density: f32,
        dist_morgans: f32,
        pbwt_version: u32,
        fixed: bool,
        active_pool: &ActivePool,
        pool_alleles: &[u8],
        switch_support: &SwitchSupportCache,
        out: &mut Vec<(usize, i32, i32)>,
        spread: &mut Vec<usize>,
    ) {
        out.clear();
        let marker_idx = marker.as_usize();

        // Emission costs (uniform genotyping error model):
        //   P(match) = 1 - θ
        //   P(mismatch) = θ
        let theta = self.costs.p_err.max(1e-9).min(1.0 - 1e-9);
        let p_match = 1.0 - theta;
        let p_mismatch = theta;
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
                meta.match_len_morgans.max(0.0),
                meta.cluster_size.max(0.0),
                meta.cluster_id,
            )
        } else {
            (pbwt_match_len.max(0.0), pbwt_density.max(0.0), u16::MAX)
        };
        let n_match_global = switch_support.global_match_count(targ_allele);
        let n_match_cluster = switch_support.cluster_match_count(targ_allele, cluster_id);
        let (pbwt_stay_cost, pbwt_switch_event_cost) =
            self.pbwt_switch_cost(match_len_morgans, cluster_size, dist_morgans);
        let selection_cost_global = self.selection_cost(n_match_global);
        let selection_cost_cluster = self.selection_cost(n_match_cluster);

        let matches = self.ref_allele_matches(marker_idx, hap, targ_allele);
        if fixed {
            // For phased anchors, always scan the full pool to ensure we honor the fixed allele
            // while still exploring better-matching donors.
            if matches {
                out.push((hap, pbwt_stay_cost, effective_match_cost));
            }
            for (idx, &h) in active_pool.list().iter().rev().enumerate() {
                let pool_idx = active_pool.list().len().saturating_sub(1) - idx;
                let pooled = pool_alleles.get(pool_idx).copied().unwrap_or(255);
                if h == hap && matches {
                    continue;
                }
                if pooled == targ_allele {
                    let same_cluster = active_pool
                        .pbwt_meta(allele, h, pbwt_version)
                        .map(|m| m.cluster_id == cluster_id)
                        .unwrap_or(false);
                    let selection_cost = if same_cluster {
                        selection_cost_cluster
                    } else {
                        selection_cost_global
                    };
                    let effective_switch_cost =
                        pbwt_switch_event_cost.saturating_add(selection_cost);
                    out.push((h, effective_switch_cost, effective_match_cost));
                    if out.len() >= self.config.switch_candidates {
                        break;
                    }
                }
            }
            if out.is_empty() {
                out.push((hap, pbwt_stay_cost, effective_mismatch_cost));
            }
            return;
        }
        if matches {
            out.push((hap, pbwt_stay_cost, effective_match_cost));
            // also allow a limited switch to a strong candidate for future-proofing
            for (idx, &h) in active_pool.list().iter().rev().take(1).enumerate() {
                let pool_idx = active_pool.list().len().saturating_sub(1) - idx;
                let pooled = pool_alleles.get(pool_idx).copied().unwrap_or(255);
                if h != hap && pooled == targ_allele {
                    let same_cluster = active_pool
                        .pbwt_meta(allele, h, pbwt_version)
                        .map(|m| m.cluster_id == cluster_id)
                        .unwrap_or(false);
                    let selection_cost = if same_cluster {
                        selection_cost_cluster
                    } else {
                        selection_cost_global
                    };
                    let effective_switch_cost =
                        pbwt_switch_event_cost.saturating_add(selection_cost);
                    out.push((h, effective_switch_cost, effective_match_cost));
                }
            }
            return;
        }
        // Try switching to matching haps
        for (idx, &h) in active_pool
            .list()
            .iter()
            .rev()
            .take(self.config.switch_candidates)
            .enumerate()
        {
            let pool_idx = active_pool.list().len().saturating_sub(1) - idx;
            let pooled = pool_alleles.get(pool_idx).copied().unwrap_or(255);
            if pooled == targ_allele {
                let same_cluster = active_pool
                    .pbwt_meta(allele, h, pbwt_version)
                    .map(|m| m.cluster_id == cluster_id)
                    .unwrap_or(false);
                let selection_cost = if same_cluster {
                    selection_cost_cluster
                } else {
                    selection_cost_global
                };
                let effective_switch_cost = pbwt_switch_event_cost.saturating_add(selection_cost);
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
                    let selection_cost = if same_cluster {
                        selection_cost_cluster
                    } else {
                        selection_cost_global
                    };
                    let effective_switch_cost =
                        pbwt_switch_event_cost.saturating_add(selection_cost);
                    out.push((h, effective_switch_cost, effective_match_cost));
                }
                if out.len() >= self.config.switch_candidates {
                    break;
                }
            }
        }
        // Always allow staying with a mismatch cost to avoid forced switching.
        // Mismatch cost is error-rate dependent under the uniform error model.
        out.push((hap, pbwt_stay_cost, effective_mismatch_cost));
    }

    #[inline]
    fn pbwt_switch_cost(
        &self,
        match_len_morgans: f32,
        density: f32,
        dist_morgans: f32,
    ) -> (i32, i32) {
        // Coalescent-based stay probability from PBWT one-sided match length with
        // explicit recombination intensity scaling (rho = 4Ne in Morgans):
        //   prior: t ~ Exp(1) over TMRCA (coalescent units)
        //   L | t ~ Exp(rate = rho * t)
        //   t | L ~ Gamma(α=2, β=1 + rho * L) for a single exponential draw
        //   P(stay | L, d) = E[exp(-rho * t * d)] = (β / (β + rho * d))^2
        // The PBWT uses a max over candidates; we map the observed max length to
        // an effective single-draw length using an order-statistic correction.
        // Use true negative log transition probabilities (no log-odds).
        const EULER_MASCHERONI: f64 = 0.5772156649015329;
        let l_raw = match_len_morgans.max(0.0) as f64;
        let k = density.max(0.0) as f64;
        let l = if k > 0.0 {
            // Order-statistic correction: E[L_max] ~= H_k / (rho * t),
            // where H_k is the harmonic number (or its continuous extension).
            let h_k = Self::harmonic_number_approx(k, EULER_MASCHERONI);
            l_raw / h_k.max(1.0)
        } else {
            l_raw
        };
        let d = dist_morgans.max(0.0) as f64;
        let rho = self.costs.recomb_intensity;
        let beta = 1.0 + rho * l;
        let denom = beta + rho * d;
        let p_no_recomb = if denom > 0.0 {
            (beta / denom).powi(2)
        } else {
            0.0
        };
        let p_stay = p_no_recomb.clamp(1e-12, 1.0 - 1e-12);
        let p_switch_event = (1.0 - p_no_recomb).clamp(1e-12, 1.0 - 1e-12);
        let stay_cost = (-(p_stay.ln()) * 1_000_000.0)
            .round()
            .clamp(i32::MIN as f64, i32::MAX as f64) as i32;
        let switch_event_cost = (-(p_switch_event.ln()) * 1_000_000.0)
            .round()
            .clamp(i32::MIN as f64, i32::MAX as f64) as i32;
        (stay_cost, switch_event_cost)
    }

    #[inline]
    fn selection_cost(&self, n_eff: usize) -> i32 {
        let n_eff = (n_eff as f64).max(1.0);
        (n_eff.ln() * 1_000_000.0)
            .round()
            .clamp(i32::MIN as f64, i32::MAX as f64) as i32
    }

    #[inline]
    fn segment_switch_cost(&self, dist_morgans: f32) -> i32 {
        let d = dist_morgans.max(0.0) as f64;
        let rho = self.costs.recomb_intensity;
        let denom = 1.0 + rho * d;
        let p_no_recomb = if denom > 0.0 { 1.0 / denom } else { 0.0 };
        let p_switch = (1.0 - p_no_recomb).clamp(1e-12, 1.0 - 1e-12);
        (-(p_switch.ln()) * 1_000_000.0)
            .round()
            .clamp(i32::MIN as f64, i32::MAX as f64) as i32
    }

    #[inline]
    fn harmonic_number_approx(k: f64, euler_mascheroni: f64) -> f64 {
        if !k.is_finite() || k <= 1.0 {
            return 1.0;
        }
        let k_round = k.round();
        if (k - k_round).abs() <= 1e-9 && k_round <= 128.0 {
            let n = k_round as usize;
            let mut h = 0.0;
            for i in 1..=n {
                h += 1.0 / (i as f64);
            }
            return h;
        }
        let inv = 1.0 / k;
        let inv2 = inv * inv;
        let inv4 = inv2 * inv2;
        k.ln() + euler_mascheroni + 0.5 * inv - (inv2 / 12.0) + (inv4 / 120.0)
    }

    #[inline]
    fn ref_allele_matches(&self, marker: usize, hap: usize, targ_allele: u8) -> bool {
        match self.packed_ref.ref_allele_targ(marker, hap) {
            Some(a) => a == targ_allele,
            None => false,
        }
    }

    #[inline]
    fn fill_pool_alleles(&self, marker: usize, active_pool: &ActivePool, out: &mut Vec<u8>) {
        let list = active_pool.list();
        if out.len() < list.len() {
            out.resize(list.len(), 255);
        }
        for (i, &h) in list.iter().enumerate() {
            out[i] = self.packed_ref.ref_allele_targ(marker, h).unwrap_or(255);
        }
    }

    fn prune_inplace_with_ptrs(
        &self,
        mut beam: Vec<BeamPath>,
        mut ptrs: Vec<u32>,
    ) -> (Vec<BeamPath>, Vec<u32>) {
        self.prune_and_collapse_with_ptrs(&mut beam, &mut ptrs);
        (beam, ptrs)
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
            let same = cluster_key(prev.cluster1, prev.hap1)
                == cluster_key(curr.cluster1, curr.hap1)
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

    fn prune_and_collapse_with_ptrs(&self, beam: &mut Vec<BeamPath>, ptrs: &mut Vec<u32>) {
        if beam.is_empty() {
            return;
        }
        if beam.len() != ptrs.len() {
            ptrs.clear();
            ptrs.resize(beam.len(), 0);
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
        let mut write = 0usize;
        for i in 0..beam.len() {
            if beam[i].score <= cutoff {
                if write != i {
                    beam[write] = beam[i].clone();
                    ptrs[write] = ptrs[i];
                }
                write += 1;
            }
        }
        beam.truncate(write);
        ptrs.truncate(write);
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
        let mut zipped: Vec<(BeamPath, u32)> =
            beam.iter().cloned().zip(ptrs.iter().copied()).collect();
        zipped.sort_unstable_by(|(a, _), (b, _)| {
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
        for i in 1..zipped.len() {
            let (ref prev, _) = zipped[write - 1];
            let (ref curr, _) = zipped[i];
            let same = cluster_key(prev.cluster1, prev.hap1)
                == cluster_key(curr.cluster1, curr.hap1)
                && cluster_key(prev.cluster2, prev.hap2) == cluster_key(curr.cluster2, curr.hap2)
                && prev.history_bits == curr.history_bits
                && prev.history_len == curr.history_len
                && prev.last_swapped == curr.last_swapped;
            if !same {
                if write != i {
                    zipped[write] = zipped[i].clone();
                }
                write += 1;
            }
        }
        zipped.truncate(write);

        if zipped.len() > self.config.beam_width {
            let k = self.config.beam_width;
            zipped.select_nth_unstable_by(k, |a, b| a.0.score.cmp(&b.0.score));
            zipped.truncate(k);
        }

        beam.clear();
        ptrs.clear();
        beam.reserve(zipped.len());
        ptrs.reserve(zipped.len());
        for (p, ptr) in zipped {
            beam.push(p);
            ptrs.push(ptr);
        }
    }

    fn beam_collapsed(&self, beam: &[BeamPath]) -> bool {
        if beam.len() < 2 {
            return true;
        }
        let mut best = i32::MAX;
        let mut worst = i32::MIN;
        for p in beam {
            if p.score < best {
                best = p.score;
            }
            if p.score > worst {
                worst = p.score;
            }
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
    const HISTORY_BITS: u8 = 64;
    let bit = if swapped { 1u64 } else { 0u64 };
    let bits = if HISTORY_BITS >= 64 {
        (prev_bits << 1) | bit
    } else {
        ((prev_bits << 1) | bit) & ((1u64 << HISTORY_BITS) - 1)
    };
    let len = if prev_len < HISTORY_BITS {
        prev_len + 1
    } else {
        HISTORY_BITS
    };
    (bits, len)
}

fn hap_constraints_penalty<RefSpace>(
    packed_ref: &PackedRefView<RefSpace>,
    hap: usize,
    constraints: &[crate::data::condensed::SegmentConstraint],
) -> (i32, bool) {
    let mut penalty: i32 = 0;
    for c in constraints {
        let marker = c.marker.as_usize();
        let ref_al = match packed_ref.ref_allele_targ(marker, hap) {
            Some(a) => a,
            None => {
                penalty = penalty.saturating_add(c.mismatch_cost);
                continue;
            }
        };
        let matches = if c.n_alleles <= 1 {
            ref_al == c.alleles[0]
        } else {
            ref_al == c.alleles[0] || ref_al == c.alleles[1]
        };
        if !matches {
            penalty = penalty.saturating_add(c.mismatch_cost);
        }
    }
    (penalty, penalty == 0)
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
        let lu = logsum_unswapped
            .get(i)
            .copied()
            .unwrap_or(f64::NEG_INFINITY);
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

impl<'a, RefSpace> BeamPhaser<'a, RefSpace> {
    fn compute_swap_posteriors_smoothed(
        &self,
        logsum_swapped: &[f64],
        logsum_unswapped: &[f64],
        call_sites: &[CallSite],
    ) -> Vec<f32> {
        let base = compute_swap_posteriors(logsum_swapped, logsum_unswapped);
        if base.is_empty() {
            return base;
        }

        let n = base.len();
        let mut emit_unswapped = vec![0.0f64; n];
        let mut emit_swapped = vec![0.0f64; n];
        for i in 0..n {
            let p = base[i].clamp(1e-6, 1.0 - 1e-6) as f64;
            emit_swapped[i] = p.ln();
            emit_unswapped[i] = (1.0 - p).ln();
            if let Some(cs) = call_sites.get(i)
                && cs.fixed
                && cs.flip_cost > 0
            {
                // Preserve hard/soft external anchors: fixed phase input should
                // dominate unless sequencing evidence is overwhelming.
                emit_swapped[i] -= (cs.flip_cost as f64) / 1_000_000.0;
            }
        }

        let mut trans_same = vec![0.0f64; n.saturating_sub(1)];
        let mut trans_flip = vec![0.0f64; n.saturating_sub(1)];
        for i in 1..n {
            let d = call_sites
                .get(i)
                .map(|c| c.dist_morgans.max(0.0))
                .unwrap_or(0.0);
            let p_flip = self.orientation_flip_prob(d);
            trans_flip[i - 1] = p_flip.ln();
            trans_same[i - 1] = (1.0 - p_flip).ln();
        }

        let mut alpha_u = vec![f64::NEG_INFINITY; n];
        let mut alpha_s = vec![f64::NEG_INFINITY; n];
        alpha_u[0] = (-0.5f64.ln()) + emit_unswapped[0];
        alpha_s[0] = (-0.5f64.ln()) + emit_swapped[0];
        for i in 1..n {
            alpha_u[i] = logaddexp(
                alpha_u[i - 1] + trans_same[i - 1],
                alpha_s[i - 1] + trans_flip[i - 1],
            ) + emit_unswapped[i];
            alpha_s[i] = logaddexp(
                alpha_s[i - 1] + trans_same[i - 1],
                alpha_u[i - 1] + trans_flip[i - 1],
            ) + emit_swapped[i];
        }

        let mut beta_u = vec![0.0f64; n];
        let mut beta_s = vec![0.0f64; n];
        for i in (0..n.saturating_sub(1)).rev() {
            beta_u[i] = logaddexp(
                trans_same[i] + emit_unswapped[i + 1] + beta_u[i + 1],
                trans_flip[i] + emit_swapped[i + 1] + beta_s[i + 1],
            );
            beta_s[i] = logaddexp(
                trans_same[i] + emit_swapped[i + 1] + beta_s[i + 1],
                trans_flip[i] + emit_unswapped[i + 1] + beta_u[i + 1],
            );
        }

        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let lu = alpha_u[i] + beta_u[i];
            let ls = alpha_s[i] + beta_s[i];
            let z = logaddexp(lu, ls);
            if !z.is_finite() {
                out.push(base[i]);
            } else {
                out.push(((ls - z).exp() as f32).clamp(0.0, 1.0));
            }
        }
        out
    }

    fn decode_posterior_phase_path(&self, p_swapped: &[f32], call_sites: &[CallSite]) -> Vec<bool> {
        if p_swapped.is_empty() {
            return Vec::new();
        }
        let n = p_swapped.len();
        let mut v_u = vec![f64::NEG_INFINITY; n];
        let mut v_s = vec![f64::NEG_INFINITY; n];
        let mut ptr_u = vec![false; n];
        let mut ptr_s = vec![false; n];
        let p0 = p_swapped[0].clamp(1e-6, 1.0 - 1e-6) as f64;
        v_u[0] = (-0.5f64.ln()) + (1.0 - p0).ln();
        v_s[0] = (-0.5f64.ln()) + p0.ln();
        for i in 1..n {
            let p = p_swapped[i].clamp(1e-6, 1.0 - 1e-6) as f64;
            let emit_u = (1.0 - p).ln();
            let emit_s = p.ln();
            let d = call_sites
                .get(i)
                .map(|c| c.dist_morgans.max(0.0))
                .unwrap_or(0.0);
            let p_flip = self.orientation_flip_prob(d);
            let log_same = (1.0 - p_flip).ln();
            let log_flip = p_flip.ln();

            let uu = v_u[i - 1] + log_same;
            let su = v_s[i - 1] + log_flip;
            if uu >= su {
                v_u[i] = uu + emit_u;
                ptr_u[i] = false;
            } else {
                v_u[i] = su + emit_u;
                ptr_u[i] = true;
            }
            let ss = v_s[i - 1] + log_same;
            let us = v_u[i - 1] + log_flip;
            if ss >= us {
                v_s[i] = ss + emit_s;
                ptr_s[i] = true;
            } else {
                v_s[i] = us + emit_s;
                ptr_s[i] = false;
            }
        }
        let mut out = vec![false; n];
        out[n - 1] = v_s[n - 1] > v_u[n - 1];
        for i in (1..n).rev() {
            out[i - 1] = if out[i] { ptr_s[i] } else { ptr_u[i] };
        }
        out
    }

    #[inline]
    fn orientation_flip_prob(&self, dist_morgans: f32) -> f64 {
        let d = dist_morgans.max(0.0) as f64;
        let rho = self.costs.recomb_intensity.max(1e-12);
        let p = (rho * d) / (1.0 + rho * d);
        p.clamp(1e-6, 0.35)
    }
}
