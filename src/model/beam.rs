//! Beam search phaser for condensed targets.

use crate::data::MarkerIdx;
use crate::data::alignment::MarkerAlignment;
use crate::data::condensed::{CallSite, CondensedTarget};
use crate::data::marker::AnyMarkerSpace;
use crate::data::ref_packed::PackedRefView;
use crate::data::storage::GenotypeMatrix;
use crate::data::storage::phase_state::Phased;
use crate::data::storage::sample_phase::SamplePhase;
use crate::model::li_stephens::subset_linear_exact_k;
use crate::model::parameters::ModelParams;
use crate::model::reference_pbwt::{
    DonorPick, PbwtBiallelicQueryProb, PbwtQueryAllele, RankBeam, ReferencePbwt,
};

const BACKPTR_INDEX_BITS: usize = 15;
const MAX_BACKPTR_PREV: u32 = (1u32 << BACKPTR_INDEX_BITS) - 1;
const MAX_BEAM_WIDTH_FOR_PACKED_BACKPTR: usize = (MAX_BACKPTR_PREV as usize) + 1;
const MAX_PBWT_CLUSTER_INTERVALS: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
struct PbwtClusterIx(u8);

impl PbwtClusterIx {
    const CAP: usize = MAX_PBWT_CLUSTER_INTERVALS;

    #[inline]
    fn from_u16(cluster_id: u16) -> Option<Self> {
        if cluster_id == u16::MAX {
            return None;
        }
        if (cluster_id as usize) < Self::CAP {
            Some(Self(cluster_id as u8))
        } else {
            None
        }
    }

    #[inline]
    fn as_usize(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BeamConfig {
    pub beam_width: usize,
    pub switch_candidates: usize,
    pub inject_interval: usize,
    pub inject_k: usize,
    pub active_pool_ttl: u32,
    pub collapse_gap: i64,
    pub prune_tolerance: i64,
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
    pub match_emit_cost: i32,
    pub mismatch_emit_cost: i32,
}

impl BeamCosts {
    pub fn from_params(params: &ModelParams) -> Self {
        let p_err = params.p_mismatch.max(1e-9).min(1.0 - 1e-9);
        let p_match = (1.0 - p_err) as f64;
        let p_mismatch = p_err as f64;
        let match_emit_cost = (-(p_match.ln()) * 1_000_000.0)
            .round()
            .clamp(i32::MIN as f64, i32::MAX as f64) as i32;
        let mismatch_emit_cost = (-(p_mismatch.ln()) * 1_000_000.0)
            .round()
            .clamp(i32::MIN as f64, i32::MAX as f64) as i32;
        Self {
            p_err: p_err as f64,
            recomb_intensity: params.recomb_intensity.max(1e-12) as f64,
            match_emit_cost,
            mismatch_emit_cost,
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
    assert!(prev <= MAX_BACKPTR_PREV);
    let p = (prev as u16) & (MAX_BACKPTR_PREV as u16);
    p | ((swapped as u16) << 15)
}

#[inline]
fn unpack_backptr(packed: u16) -> BackPtr {
    BackPtr {
        prev: (packed & (MAX_BACKPTR_PREV as u16)) as u32,
        swapped: (packed & 0x8000) != 0,
    }
}

#[derive(Clone, Debug)]
pub struct BeamPath {
    pub hap1: usize,
    pub hap2: usize,
    pub cluster1: u16,
    pub cluster2: u16,
    pub score: i64,
    pub model_score: i64,
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
    pub donor_mass: Vec<(usize, f32)>,
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

#[derive(Clone, Copy, Debug)]
struct RankedDonorPick {
    pick: DonorPick,
    cluster_id: u16,
    cluster_size: f32,
    score: f64,
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
            let ((qa0, qp0), (qa1, qp1)) = if let Some(map) = mapping {
                let m0 = map.targ_to_ref.get(0).copied().unwrap_or(-1);
                let m1 = map.targ_to_ref.get(1).copied().unwrap_or(-1);
                let q0 = if m0 >= 0 && m0 <= 1 {
                    (
                        PbwtQueryAllele::allele(m0 as u8).unwrap_or_else(PbwtQueryAllele::wildcard),
                        PbwtBiallelicQueryProb::deterministic(m0 as u8),
                    )
                } else {
                    (
                        PbwtQueryAllele::wildcard(),
                        PbwtBiallelicQueryProb::uniform(),
                    )
                };
                let q1 = if m1 >= 0 && m1 <= 1 {
                    (
                        PbwtQueryAllele::allele(m1 as u8).unwrap_or_else(PbwtQueryAllele::wildcard),
                        PbwtBiallelicQueryProb::deterministic(m1 as u8),
                    )
                } else {
                    (
                        PbwtQueryAllele::wildcard(),
                        PbwtBiallelicQueryProb::uniform(),
                    )
                };
                (q0, q1)
            } else {
                (
                    (
                        PbwtQueryAllele::wildcard(),
                        PbwtBiallelicQueryProb::uniform(),
                    ),
                    (
                        PbwtQueryAllele::wildcard(),
                        PbwtBiallelicQueryProb::uniform(),
                    ),
                )
            };

            let mut beams = [RankBeam::full(n_ref as u32), RankBeam::full(n_ref as u32)];
            pbwt.advance_with_beams_query_probs(
                &ref_alleles,
                n_alleles,
                hi_idx,
                &[qa0, qa1],
                Some(&[qp0, qp1]),
                &mut beams,
            );
            let mut d0: Vec<DonorPick> = Vec::new();
            let mut d1: Vec<DonorPick> = Vec::new();
            pbwt.select_donor_picks_into(&beams[0], k, &mut d0);
            pbwt.select_donor_picks_into(&beams[1], k, &mut d1);
            let gen_pos = hi_freq_gen_positions.get(hi_idx).copied().unwrap_or(0.0);
            let step_morgans = if hi_idx > 0 {
                (gen_pos - hi_freq_gen_positions[hi_idx - 1]).abs() / 100.0
            } else if hi_idx + 1 < hi_freq_gen_positions.len() {
                (hi_freq_gen_positions[hi_idx + 1] - gen_pos).abs() / 100.0
            } else {
                0.0
            };
            let select_best_by_transition = |donors: &[DonorPick], beam: &RankBeam| {
                const EULER_MASCHERONI: f64 = 0.5772156649015329;
                let rho = recomb_intensity.max(1e-12) as f64;
                let d = (step_morgans.max(1e-12)) as f64;
                let mut ranked: Vec<RankedDonorPick> = Vec::with_capacity(donors.len());
                for &pick in donors {
                    let (cluster_id, cluster_size) = find_cluster(beam, pick.pos as usize);
                    let start_idx = pick.start.max(0) as usize;
                    let start_pos = hi_freq_gen_positions
                        .get(start_idx)
                        .copied()
                        .unwrap_or(gen_pos);
                    let len_morgans = ((gen_pos - start_pos).abs() / 100.0) as f64;
                    let k = cluster_size.max(0.0) as f64;
                    let h_k = if k >= 2.0 {
                        k.ln() + EULER_MASCHERONI
                    } else {
                        1.0
                    };
                    let l_eff = len_morgans / h_k.max(1.0);
                    let beta = 1.0 + rho * l_eff;
                    let denom = beta + rho * d;
                    let score = if denom > 0.0 {
                        (beta / denom).powi(2)
                    } else {
                        0.0
                    };
                    ranked.push(RankedDonorPick {
                        pick,
                        cluster_id,
                        cluster_size,
                        score,
                    });
                }
                ranked.sort_unstable_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                ranked.truncate(k);
                ranked
            };
            let ranked0 = select_best_by_transition(&d0, &beams[0]);
            let ranked1 = select_best_by_transition(&d1, &beams[1]);
            let meta0 = build_ranked_donor_meta(&ranked0, gen_pos, hi_freq_gen_positions);
            let meta1 = build_ranked_donor_meta(&ranked1, gen_pos, hi_freq_gen_positions);
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

fn build_ranked_donor_meta(
    donors: &[RankedDonorPick],
    gen_pos: f64,
    hi_freq_gen_positions: &[f64],
) -> Vec<PbwtDonorMeta> {
    let mut out = Vec::with_capacity(donors.len());
    for ranked in donors {
        let pick = ranked.pick;
        let start_idx = pick.start.max(0) as usize;
        let start_pos = hi_freq_gen_positions
            .get(start_idx)
            .copied()
            .unwrap_or(gen_pos);
        let len_morgans = ((gen_pos - start_pos).abs() / 100.0) as f32;
        out.push(PbwtDonorMeta {
            hap: pick.hap,
            cluster_id: ranked.cluster_id,
            cluster_size: ranked.cluster_size,
            match_len_morgans: len_morgans,
        });
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
    hap1_allele: Vec<(usize, i32, i32, i32)>,
    hap2_allele: Vec<(usize, i32, i32, i32)>,
    repair_cand: Vec<(usize, i32, bool)>,
    spread: Vec<usize>,
    pool_alleles: Vec<u8>,
    switch_support: SwitchSupportCache,
}

struct SwitchSupportCache {
    marker_idx: usize,
    pbwt_version: u32,
    initialized: bool,
    global_match_counts: [usize; 2],
    cluster_match_counts0: [usize; MAX_PBWT_CLUSTER_INTERVALS],
    cluster_match_counts1: [usize; MAX_PBWT_CLUSTER_INTERVALS],
}

struct AlignedPoolAlleles<'a> {
    active_pool: &'a ActivePool,
    haps: &'a [usize],
    alleles: &'a [u8],
}

impl<'a> AlignedPoolAlleles<'a> {
    #[inline]
    fn alleles(&self) -> &'a [u8] {
        self.alleles
    }

    #[inline]
    fn iter(&self) -> impl Iterator<Item = (usize, u8)> + 'a {
        self.haps.iter().copied().zip(self.alleles.iter().copied())
    }
}

impl BeamScratch {
    fn new(cap: usize) -> Self {
        Self {
            hap1_candidates: Vec::with_capacity(cap),
            hap2_candidates: Vec::with_capacity(cap),
            hap1_allele: Vec::with_capacity(cap),
            hap2_allele: Vec::with_capacity(cap),
            repair_cand: Vec::with_capacity(cap.saturating_add(2)),
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
            cluster_match_counts0: [0; MAX_PBWT_CLUSTER_INTERVALS],
            cluster_match_counts1: [0; MAX_PBWT_CLUSTER_INTERVALS],
        }
    }

    fn ensure_initialized(
        &mut self,
        marker_idx: usize,
        pbwt_version: u32,
        pool_alleles: &AlignedPoolAlleles<'_>,
    ) {
        if self.initialized && self.marker_idx == marker_idx && self.pbwt_version == pbwt_version {
            return;
        }
        self.initialized = true;
        self.marker_idx = marker_idx;
        self.pbwt_version = pbwt_version;
        self.global_match_counts = [0, 0];
        self.cluster_match_counts0.fill(0);
        self.cluster_match_counts1.fill(0);

        for (hap, allele) in pool_alleles.iter() {
            if allele == 0 {
                self.global_match_counts[0] = self.global_match_counts[0].saturating_add(1);
                if let Some(meta) = pool_alleles.active_pool.pbwt_meta(0, hap, pbwt_version) {
                    if let Some(cid) = PbwtClusterIx::from_u16(meta.cluster_id) {
                        let idx = cid.as_usize();
                        self.cluster_match_counts0[idx] =
                            self.cluster_match_counts0[idx].saturating_add(1);
                    }
                }
            } else if allele == 1 {
                self.global_match_counts[1] = self.global_match_counts[1].saturating_add(1);
                if let Some(meta) = pool_alleles.active_pool.pbwt_meta(1, hap, pbwt_version) {
                    if let Some(cid) = PbwtClusterIx::from_u16(meta.cluster_id) {
                        let idx = cid.as_usize();
                        self.cluster_match_counts1[idx] =
                            self.cluster_match_counts1[idx].saturating_add(1);
                    }
                }
            }
        }
        self.global_match_counts[0] = self.global_match_counts[0].max(1);
        self.global_match_counts[1] = self.global_match_counts[1].max(1);
    }
}

impl<'a, RefSpace> BeamPhaser<'a, RefSpace> {
    pub fn new(
        packed_ref: &'a PackedRefView<RefSpace>,
        params: &ModelParams,
        config: BeamConfig,
    ) -> Self {
        assert!(
            config.beam_width > 0,
            "beam_width must be > 0 for BeamPhaser"
        );
        assert!(
            config.beam_width <= MAX_BEAM_WIDTH_FOR_PACKED_BACKPTR,
            "beam_width={} exceeds packed-backpointer capacity {}; increase backpointer width or lower beam_width",
            config.beam_width,
            MAX_BEAM_WIDTH_FOR_PACKED_BACKPTR
        );
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
                donor_mass: Vec::new(),
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
        let mut step_haps: Vec<Vec<(usize, usize)>> = Vec::with_capacity(n_calls);
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
            // Subset stickiness: keep donors currently used by active paths in the pool.
            // This prevents artificial forced switches when dynamic injection jitter
            // would otherwise evict a donor that still carries high posterior support.
            let sticky_version = call.hi_idx as u32;
            for p in &beam {
                active_pool.touch(p.hap1, sticky_version);
                active_pool.touch(p.hap2, sticky_version);
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
            let mut best_score = i64::MAX;
            for p in &beam {
                if p.score < best_score {
                    best_score = p.score;
                }
            }
            let cutoff = if self.config.prune_tolerance > 0 {
                best_score + self.config.prune_tolerance
            } else {
                i64::MAX
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
            let mut step_pairs: Vec<(usize, usize)> = Vec::with_capacity(next.len());
            for p in &next {
                assert!(p.prev_idx <= MAX_BACKPTR_PREV, "beam backptr overflow");
                step_ptrs.push(pack_backptr(p.prev_idx, p.prev_swapped));
                step_pairs.push((p.hap1, p.hap2));
            }
            backptrs.push(step_ptrs);
            step_haps.push(step_pairs);
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

        let donor_mass = donor_posterior_mass(
            &beam,
            &step_haps,
            &backptrs,
            &segment_ptrs,
            trailing_ptrs.as_deref(),
            self.packed_ref.n_ref_haps(),
            self.config.beam_width.saturating_mul(4),
        );

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
            let p_swapped = compute_swap_posteriors(&logsum_swapped, &logsum_unswapped);
            let phases = self.decode_swap_track(
                &logsum_swapped,
                &logsum_unswapped,
                &condensed.call_sites,
                &phases,
            );
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
                donor_mass,
            };
        }
        BeamPosteriors {
            decisions: Vec::new(),
            p_swapped: vec![0.5; n_calls],
            donor_mass,
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
                    model_score: 0,
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
        let missing = crate::data::storage::AlleleCode::MISSING.raw();

        // Partition haplotypes by which allele they carry.
        for &h in list {
            match self.packed_ref.ref_allele_targ(marker_idx, h) {
                Some(ref_al) if ref_al == a1 => match_a1.push(h),
                Some(ref_al) if ref_al == a2 => match_a2.push(h),
                Some(ref_al) if ref_al == missing || ref_al > 1 => {
                    // Keep unknown/out-of-domain reference alleles neutral at initialization.
                    match_a1.push(h);
                    match_a2.push(h);
                }
                None => {
                    match_a1.push(h);
                    match_a2.push(h);
                }
                _ => {}
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
                    model_score: 0,
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
                    model_score: 0,
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
                        score: path.score + i64::from(*c1) + i64::from(*c2),
                        model_score: path.model_score + i64::from(*c1) + i64::from(*c2),
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
        cutoff: i64,
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
        {
            let aligned_pool = self.fill_pool_alleles(marker_idx, active_pool, &mut pool_alleles);
            scratch
                .switch_support
                .ensure_initialized(marker_idx, pbwt_version, &aligned_pool);
            let flip_event_cost = self.orientation_flip_event_cost(call.dist_morgans);

            if call.fixed {
                self.expand_orientation(
                    path,
                    parent_idx,
                    call,
                    call_idx,
                    flip_event_cost,
                    a1,
                    a2,
                    false,
                    pbwt_version,
                    active_pool,
                    aligned_pool.alleles(),
                    out,
                    logsum_unswapped,
                    logsum_swapped,
                    cutoff,
                    scratch,
                );
            } else {
                self.expand_orientation(
                    path,
                    parent_idx,
                    call,
                    call_idx,
                    flip_event_cost,
                    a1,
                    a2,
                    false,
                    pbwt_version,
                    active_pool,
                    aligned_pool.alleles(),
                    out,
                    logsum_unswapped,
                    logsum_swapped,
                    cutoff,
                    scratch,
                );
                self.expand_orientation(
                    path,
                    parent_idx,
                    call,
                    call_idx,
                    flip_event_cost,
                    a2,
                    a1,
                    true,
                    pbwt_version,
                    active_pool,
                    aligned_pool.alleles(),
                    out,
                    logsum_unswapped,
                    logsum_swapped,
                    cutoff,
                    scratch,
                );
            }
        }
        scratch.pool_alleles = pool_alleles;
    }

    fn expand_orientation(
        &self,
        path: &BeamPath,
        parent_idx: u32,
        call: &CallSite,
        call_idx: usize,
        flip_event_cost: i32,
        hap1_al: u8,
        hap2_al: u8,
        swapped: bool,
        pbwt_version: u32,
        active_pool: &ActivePool,
        pool_alleles: &[u8],
        out: &mut Vec<BeamPath>,
        logsum_unswapped: &mut [f64],
        logsum_swapped: &mut [f64],
        cutoff: i64,
        scratch: &mut BeamScratch,
    ) {
        let recomb_prob = self.recomb_prob_from_dist(call.dist_morgans);
        self.repair_hap_for_allele_into(
            path.hap1,
            call.marker,
            hap1_al,
            recomb_prob,
            pbwt_version,
            active_pool,
            pool_alleles,
            &mut scratch.hap1_allele,
            &mut scratch.repair_cand,
            &mut scratch.spread,
        );
        self.repair_hap_for_allele_into(
            path.hap2,
            call.marker,
            hap2_al,
            recomb_prob,
            pbwt_version,
            active_pool,
            pool_alleles,
            &mut scratch.hap2_allele,
            &mut scratch.repair_cand,
            &mut scratch.spread,
        );
        for (h1, c1_total, c1_model, e1) in scratch.hap1_allele.iter() {
            for (h2, c2_total, c2_model, e2) in scratch.hap2_allele.iter() {
                let score_no_flip = path.score
                    + i64::from(*c1_total)
                    + i64::from(*c2_total)
                    + i64::from(*e1)
                    + i64::from(*e2);
                let model_score_no_flip = path.model_score
                    + i64::from(*c1_model)
                    + i64::from(*c2_model)
                    + i64::from(*e1)
                    + i64::from(*e2);
                let flip_penalty = if call_idx == 0 {
                    0
                } else if swapped != path.last_swapped {
                    i64::from(flip_event_cost) + i64::from(call.flip_cost)
                } else {
                    0
                };
                let orient_search_cost =
                    i64::from(self.orientation_search_prior_cost(swapped, call.phase_conf));
                let score = score_no_flip + flip_penalty + orient_search_cost;
                let model_score = model_score_no_flip + flip_penalty + orient_search_cost;
                let logp = -(model_score as f64) / 1_000_000.0;
                // Posterior orientation mass at a marker should include the local
                // orientation confidence prior c:
                //   P(keep) ∝ c * w_keep, P(swap) ∝ (1-c) * w_swap.
                // We only apply this in the reporting accumulator (`p_swapped`)
                // so search/ranking remains unchanged.
                let c = call.phase_conf.clamp(1e-6, 1.0 - 1e-6) as f64;
                let orient_prior = if swapped { (1.0 - c).ln() } else { c.ln() };
                let logp_post = logp + orient_prior;
                if swapped {
                    logsum_swapped[call_idx] = logaddexp(logsum_swapped[call_idx], logp_post);
                } else {
                    logsum_unswapped[call_idx] = logaddexp(logsum_unswapped[call_idx], logp_post);
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
                        model_score,
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
        recomb_prob: f32,
        pbwt_version: u32,
        active_pool: &ActivePool,
        pool_alleles: &[u8],
        out: &mut Vec<(usize, i32, i32, i32)>,
        cand: &mut Vec<(usize, i32, bool)>,
        spread: &mut Vec<usize>,
    ) {
        out.clear();
        cand.clear();
        let marker_idx = marker.as_usize();
        // Build candidate support C first, then normalize exact transition probabilities on C.
        // Each candidate stores (hap, emission_cost, is_stay_candidate).
        #[inline]
        fn push_unique(cand: &mut Vec<(usize, i32, bool)>, h: usize, emit: i32, is_stay: bool) {
            if cand.iter().any(|(x, _, _)| *x == h) {
                return;
            }
            cand.push((h, emit, is_stay));
        }

        // Always keep stay branch so truncation never forces a recombination.
        let stay_emit = self.emission_cost_for_hap(marker_idx, hap, targ_allele);
        push_unique(cand, hap, stay_emit, true);

        // Add recent switch candidates without hard allele gating. Under non-zero
        // emission error, non-matching donors must remain reachable.
        let cap = self.config.switch_candidates.max(1);
        for (idx, &h) in active_pool.list().iter().rev().enumerate() {
            if h == hap {
                continue;
            }
            let pool_idx = active_pool.list().len().saturating_sub(1) - idx;
            let pooled = pool_alleles[pool_idx];
            let emit = self.emission_cost_for_ref_allele(pooled, targ_allele);
            push_unique(cand, h, emit, false);
            if cand.len() >= cap.saturating_add(1) {
                break;
            }
        }

        // Diversity backstop from spread sampling over the active pool.
        if cand.len() < cap.saturating_add(1) {
            sample_even_into(active_pool.list(), cap, spread);
            for &h in spread.iter() {
                if h == hap {
                    continue;
                }
                let emit = self.emission_cost_for_hap(marker_idx, h, targ_allele);
                push_unique(cand, h, emit, false);
                if cand.len() >= cap.saturating_add(1) {
                    break;
                }
            }
        }

        let n_total = self.packed_ref.n_ref_haps().max(1);
        let switch_prob = self
            .adjust_switch_prob_from_pbwt(recomb_prob, active_pool, targ_allele, hap, pbwt_version)
            .clamp(1e-9, 1.0 - 1e-9);
        let k_subset = cand.len().max(1) as f32;
        let (stay_gap, shift) = subset_linear_exact_k(switch_prob, k_subset, n_total);
        let stay_prob = (stay_gap + shift).clamp(1e-12, 1.0 - 1e-12);
        let switch_prob = shift.clamp(1e-12, 1.0 - 1e-12);
        let stay_cost = (-(f64::from(stay_prob).ln()) * 1_000_000.0)
            .round()
            .clamp(i32::MIN as f64, i32::MAX as f64) as i32;
        let switch_cost = (-(f64::from(switch_prob).ln()) * 1_000_000.0)
            .round()
            .clamp(i32::MIN as f64, i32::MAX as f64) as i32;

        out.reserve(cand.len());
        for (h, emit_cost, is_stay) in cand.drain(..) {
            let t_cost = if is_stay && h == hap {
                stay_cost
            } else {
                switch_cost
            };
            out.push((h, t_cost, t_cost, emit_cost));
        }
    }

    #[inline]
    fn orientation_flip_event_cost(&self, dist_morgans: f32) -> i32 {
        let d = dist_morgans.max(0.0) as f64;
        let rho = self.costs.recomb_intensity.max(1e-12);
        let p_stay = (-rho * d).exp().clamp(1e-6, 1.0 - 1e-6);
        ((p_stay / (1.0 - p_stay)).ln() * 1_000_000.0)
            .round()
            .clamp(i32::MIN as f64, i32::MAX as f64) as i32
    }

    #[inline]
    fn segment_switch_cost(&self, dist_morgans: f32) -> i32 {
        let d = dist_morgans.max(0.0) as f64;
        let lambda = self.costs.recomb_intensity;
        let p_switch = (-f64::exp_m1(-lambda * d)).clamp(1e-12, 1.0 - 1e-12);
        (-(p_switch.ln()) * 1_000_000.0)
            .round()
            .clamp(i32::MIN as f64, i32::MAX as f64) as i32
    }

    #[inline]
    fn recomb_prob_from_dist(&self, dist_morgans: f32) -> f32 {
        let d = dist_morgans.max(0.0) as f64;
        let lambda = self.costs.recomb_intensity.max(1e-12);
        (-f64::exp_m1(-lambda * d)).clamp(0.0, 1.0) as f32
    }

    #[inline]
    fn emission_cost_for_ref_allele(&self, ref_allele: u8, targ_allele: u8) -> i32 {
        let missing = crate::data::storage::AlleleCode::MISSING.raw();
        if targ_allele == missing || targ_allele > 1 {
            return 0;
        }
        if ref_allele == missing || ref_allele > 1 {
            return 0;
        }
        if ref_allele == targ_allele {
            self.costs.match_emit_cost
        } else {
            self.costs.mismatch_emit_cost
        }
    }

    #[inline]
    fn emission_cost_for_hap(&self, marker: usize, hap: usize, targ_allele: u8) -> i32 {
        let ref_allele = self
            .packed_ref
            .ref_allele_targ(marker, hap)
            .unwrap_or(crate::data::storage::AlleleCode::MISSING.raw());
        self.emission_cost_for_ref_allele(ref_allele, targ_allele)
    }

    #[inline]
    fn adjust_switch_prob_from_pbwt(
        &self,
        base_switch_prob: f32,
        active_pool: &ActivePool,
        allele: u8,
        current_hap: usize,
        pbwt_version: u32,
    ) -> f32 {
        let base = base_switch_prob.clamp(1e-9, 1.0 - 1e-9) as f64;
        let Some(meta) = active_pool.pbwt_meta(allele, current_hap, pbwt_version) else {
            return base_switch_prob;
        };
        let len = meta.match_len_morgans.max(0.0) as f64;
        let cluster = meta.cluster_size.max(1.0) as f64;

        // Positive signal => stronger evidence to stay with current donor.
        let len_signal = (len / 0.0025).ln_1p();
        let density_penalty = 0.35 * (cluster / 4.0).ln_1p();
        let stay_signal = (len_signal - density_penalty).clamp(-4.0, 4.0);

        // Move switching log-odds modestly to avoid brittle over-anchoring.
        let base_logit = (base / (1.0 - base)).ln();
        let logit_bias = (-0.9 * stay_signal).clamp(-2.5, 2.5);
        let adj_logit = base_logit + logit_bias;
        let adj = 1.0 / (1.0 + (-adj_logit).exp());
        adj.clamp(1e-9, 1.0 - 1e-9) as f32
    }

    #[inline]
    fn orientation_search_prior_cost(&self, swapped: bool, phase_conf: f32) -> i32 {
        let c = if phase_conf.is_finite() {
            phase_conf.clamp(1e-6, 1.0 - 1e-6)
        } else {
            0.5
        };
        let informativeness = ((c - 0.5).abs() * 2.0).clamp(0.0, 1.0);
        if informativeness <= 0.5 {
            return 0;
        }
        let prior = if swapped { 1.0 - c } else { c } as f64;
        let weight = ((informativeness - 0.5) / 0.5) as f64;
        let scaled_nats = (-prior.ln()) * (0.05 * weight * weight);
        (scaled_nats * 1_000_000.0)
            .round()
            .clamp(i32::MIN as f64, i32::MAX as f64) as i32
    }

    #[inline]
    fn fill_pool_alleles<'b>(
        &self,
        marker: usize,
        active_pool: &'b ActivePool,
        out: &'b mut Vec<u8>,
    ) -> AlignedPoolAlleles<'b> {
        let list = active_pool.list();
        out.clear();
        out.reserve(list.len());
        for &h in list {
            out.push(
                self.packed_ref
                    .ref_allele_targ(marker, h)
                    .unwrap_or(crate::data::storage::AlleleCode::MISSING.raw()),
            );
        }
        AlignedPoolAlleles {
            active_pool,
            haps: list,
            alleles: out.as_slice(),
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
        let mut best = i64::MAX;
        for p in beam.iter() {
            if p.score < best {
                best = p.score;
            }
        }
        let cutoff = if self.config.prune_tolerance > 0 {
            best + self.config.prune_tolerance
        } else {
            i64::MAX
        };
        beam.retain(|p| p.score <= cutoff);
        if beam.is_empty() {
            return;
        }

        // Collapse identical states by concrete hap ids (not PBWT cluster ids).
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

    fn prune_and_collapse_with_ptrs(&self, beam: &mut Vec<BeamPath>, ptrs: &mut Vec<u32>) {
        if beam.is_empty() {
            return;
        }
        if beam.len() != ptrs.len() {
            ptrs.clear();
            ptrs.resize(beam.len(), 0);
        }
        // Prune by tolerance relative to best score.
        let mut best = i64::MAX;
        for p in beam.iter() {
            if p.score < best {
                best = p.score;
            }
        }
        let cutoff = if self.config.prune_tolerance > 0 {
            best + self.config.prune_tolerance
        } else {
            i64::MAX
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

        // Collapse identical states by concrete hap ids (not PBWT cluster ids).
        let mut zipped: Vec<(BeamPath, u32)> =
            beam.iter().cloned().zip(ptrs.iter().copied()).collect();
        zipped.sort_unstable_by(|(a, _), (b, _)| {
            a.hap1
                .cmp(&b.hap1)
                .then(a.hap2.cmp(&b.hap2))
                .then(a.history_bits.cmp(&b.history_bits))
                .then(a.history_len.cmp(&b.history_len))
                .then(a.last_swapped.cmp(&b.last_swapped))
                .then(a.score.cmp(&b.score))
        });
        let mut write = 1usize;
        for i in 1..zipped.len() {
            let (ref prev, _) = zipped[write - 1];
            let (ref curr, _) = zipped[i];
            let same = prev.hap1 == curr.hap1
                && prev.hap2 == curr.hap2
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
        let mut best = i64::MAX;
        let mut worst = i64::MIN;
        for p in beam {
            if p.score < best {
                best = p.score;
            }
            if p.score > worst {
                worst = p.score;
            }
        }
        worst.saturating_sub(best) < self.config.collapse_gap
    }

    fn decode_swap_track(
        &self,
        logsum_swapped: &[f64],
        logsum_unswapped: &[f64],
        calls: &[CallSite],
        fallback: &[bool],
    ) -> Vec<bool> {
        let n = calls
            .len()
            .min(logsum_swapped.len())
            .min(logsum_unswapped.len());
        if n == 0 {
            return Vec::new();
        }

        let mut dp_unswapped = vec![f64::NEG_INFINITY; n];
        let mut dp_swapped = vec![f64::NEG_INFINITY; n];
        let mut prev_unswapped = vec![false; n];
        let mut prev_swapped = vec![false; n];

        let (e0u, e0s) = swap_emission_pair(logsum_unswapped[0], logsum_swapped[0]);
        dp_unswapped[0] = e0u;
        dp_swapped[0] = e0s;

        for i in 1..n {
            let flip_cost = i64::from(self.orientation_flip_event_cost(calls[i].dist_morgans))
                + i64::from(calls[i].flip_cost);
            let flip_log_penalty = -(flip_cost as f64) / 1_000_000.0;
            let (emit_unswapped, emit_swapped) =
                swap_emission_pair(logsum_unswapped[i], logsum_swapped[i]);

            let keep_unswapped = dp_unswapped[i - 1];
            let flip_to_unswapped = dp_swapped[i - 1] + flip_log_penalty;
            if keep_unswapped >= flip_to_unswapped {
                dp_unswapped[i] = keep_unswapped + emit_unswapped;
                prev_unswapped[i] = false;
            } else {
                dp_unswapped[i] = flip_to_unswapped + emit_unswapped;
                prev_unswapped[i] = true;
            }

            let keep_swapped = dp_swapped[i - 1];
            let flip_to_swapped = dp_unswapped[i - 1] + flip_log_penalty;
            if keep_swapped >= flip_to_swapped {
                dp_swapped[i] = keep_swapped + emit_swapped;
                prev_swapped[i] = true;
            } else {
                dp_swapped[i] = flip_to_swapped + emit_swapped;
                prev_swapped[i] = false;
            }
        }

        let mut out = vec![false; n];
        out[n - 1] = if dp_swapped[n - 1] > dp_unswapped[n - 1] {
            true
        } else if dp_swapped[n - 1] < dp_unswapped[n - 1] {
            false
        } else {
            fallback.get(n - 1).copied().unwrap_or(false)
        };

        for i in (1..n).rev() {
            out[i - 1] = if out[i] {
                prev_swapped[i]
            } else {
                prev_unswapped[i]
            };
        }
        out
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

#[inline]
fn swap_emission_pair(log_unswapped: f64, log_swapped: f64) -> (f64, f64) {
    if !log_unswapped.is_finite() && !log_swapped.is_finite() {
        return (0.0, 0.0);
    }
    let m = if log_unswapped > log_swapped {
        log_unswapped
    } else {
        log_swapped
    };
    let pu = (log_unswapped - m).exp();
    let ps = (log_swapped - m).exp();
    let z = pu + ps;
    if z <= f64::MIN_POSITIVE {
        return (0.0, 0.0);
    }
    ((pu / z).ln(), (ps / z).ln())
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

fn posterior_path_weights(beam: &[BeamPath]) -> (Vec<f64>, f64) {
    let mut best = i64::MAX;
    for p in beam {
        if p.model_score < best {
            best = p.model_score;
        }
    }
    let mut weights: Vec<f64> = Vec::with_capacity(beam.len());
    let mut z = 0.0f64;
    for p in beam {
        // Scores are in scaled nats (1e6). Clip to avoid denorm/underflow churn.
        let delta_nats = ((p.model_score - best) as f64 / 1_000_000.0).max(0.0);
        let w = (-delta_nats.min(80.0)).exp();
        weights.push(w);
        z += w;
    }
    (weights, z)
}

fn rank_donor_mass(
    mass_by_hap: std::collections::HashMap<usize, f64>,
    max_out: usize,
) -> Vec<(usize, f32)> {
    if max_out == 0 {
        return Vec::new();
    }
    let mut ranked: Vec<(usize, f32)> = mass_by_hap
        .into_iter()
        .filter_map(|(h, m)| if m > 0.0 { Some((h, m as f32)) } else { None })
        .collect();
    ranked.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if ranked.len() > max_out {
        ranked.truncate(max_out);
    }
    ranked
}

fn rank_donor_mass_dense(
    mass_by_hap: &[f64],
    touched: &[usize],
    max_out: usize,
) -> Vec<(usize, f32)> {
    if max_out == 0 {
        return Vec::new();
    }
    let mut ranked: Vec<(usize, f32)> = touched
        .iter()
        .filter_map(|&h| {
            let m = mass_by_hap.get(h).copied().unwrap_or(0.0);
            if m > 0.0 { Some((h, m as f32)) } else { None }
        })
        .collect();
    ranked.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if ranked.len() > max_out {
        ranked.truncate(max_out);
    }
    ranked
}

fn donor_posterior_mass_terminal(
    beam: &[BeamPath],
    weights: &[f64],
    z: f64,
    n_ref_haps: usize,
    max_out: usize,
) -> Vec<(usize, f32)> {
    if beam.is_empty() || max_out == 0 || z <= f64::MIN_POSITIVE || n_ref_haps == 0 {
        return Vec::new();
    }
    const DENSE_ACCUM_MAX_HAPS: usize = 32 * 1024;
    if n_ref_haps <= DENSE_ACCUM_MAX_HAPS {
        let mut mass_by_hap = vec![0.0f64; n_ref_haps];
        let mut touched: Vec<usize> =
            Vec::with_capacity(beam.len().saturating_mul(2).min(n_ref_haps));
        for (p, &w) in beam.iter().zip(weights.iter()) {
            let post_half = 0.5 * (w / z);
            if p.hap1 < n_ref_haps {
                if mass_by_hap[p.hap1] == 0.0 {
                    touched.push(p.hap1);
                }
                mass_by_hap[p.hap1] += post_half;
            }
            if p.hap2 < n_ref_haps {
                if mass_by_hap[p.hap2] == 0.0 {
                    touched.push(p.hap2);
                }
                mass_by_hap[p.hap2] += post_half;
            }
        }
        return rank_donor_mass_dense(&mass_by_hap, &touched, max_out);
    }
    let mut mass_by_hap: std::collections::HashMap<usize, f64> =
        std::collections::HashMap::with_capacity(beam.len().saturating_mul(2));
    for (p, &w) in beam.iter().zip(weights.iter()) {
        let post_half = 0.5 * (w / z);
        if p.hap1 < n_ref_haps {
            *mass_by_hap.entry(p.hap1).or_insert(0.0) += post_half;
        }
        if p.hap2 < n_ref_haps {
            *mass_by_hap.entry(p.hap2).or_insert(0.0) += post_half;
        }
    }
    rank_donor_mass(mass_by_hap, max_out)
}

fn donor_posterior_mass(
    beam: &[BeamPath],
    step_haps: &[Vec<(usize, usize)>],
    backptrs: &[Vec<u16>],
    segment_ptrs: &[Vec<u32>],
    trailing_ptrs: Option<&[u32]>,
    n_ref_haps: usize,
    max_out: usize,
) -> Vec<(usize, f32)> {
    if beam.is_empty() || max_out == 0 {
        return Vec::new();
    }
    let (weights, z) = posterior_path_weights(beam);
    if z <= f64::MIN_POSITIVE {
        return Vec::new();
    }

    let n_steps = step_haps.len();
    if n_steps == 0 || backptrs.len() != n_steps || segment_ptrs.len() != n_steps {
        return donor_posterior_mass_terminal(beam, &weights, z, n_ref_haps, max_out);
    }

    const DENSE_ACCUM_MAX_HAPS: usize = 32 * 1024;
    let use_dense = n_ref_haps <= DENSE_ACCUM_MAX_HAPS;
    let mut mass_dense: Vec<f64> = if use_dense {
        vec![0.0f64; n_ref_haps]
    } else {
        Vec::new()
    };
    let mut touched: Vec<usize> = if use_dense {
        Vec::with_capacity(beam.len().saturating_mul(n_steps).min(n_ref_haps))
    } else {
        Vec::new()
    };
    let mut mass_sparse: std::collections::HashMap<usize, f64> = if use_dense {
        std::collections::HashMap::new()
    } else {
        std::collections::HashMap::with_capacity(beam.len().saturating_mul(n_steps).max(16))
    };
    let step_scale = 1.0f64 / n_steps as f64;
    let mut broken_chain = false;

    if let Some(ptrs) = trailing_ptrs {
        for (terminal_idx, &w) in weights.iter().enumerate() {
            let post_step_half = 0.5 * (w / z) * step_scale;
            let mut idx = ptrs.get(terminal_idx).copied().unwrap_or(0) as usize;
            for step in (0..n_steps).rev() {
                if idx >= step_haps[step].len() || idx >= backptrs[step].len() {
                    broken_chain = true;
                    break;
                }
                let (h1, h2) = step_haps[step][idx];
                if h1 < n_ref_haps {
                    if use_dense {
                        if mass_dense[h1] == 0.0 {
                            touched.push(h1);
                        }
                        mass_dense[h1] += post_step_half;
                    } else {
                        *mass_sparse.entry(h1).or_insert(0.0) += post_step_half;
                    }
                }
                if h2 < n_ref_haps {
                    if use_dense {
                        if mass_dense[h2] == 0.0 {
                            touched.push(h2);
                        }
                        mass_dense[h2] += post_step_half;
                    } else {
                        *mass_sparse.entry(h2).or_insert(0.0) += post_step_half;
                    }
                }
                let bp = unpack_backptr(backptrs[step][idx]);
                let prev_idx = bp.prev as usize;
                idx = segment_ptrs[step].get(prev_idx).copied().unwrap_or(0) as usize;
            }
            if broken_chain {
                break;
            }
        }
    } else {
        for (terminal_idx, &w) in weights.iter().enumerate() {
            let post_step_half = 0.5 * (w / z) * step_scale;
            let mut idx = terminal_idx;
            for step in (0..n_steps).rev() {
                if idx >= step_haps[step].len() || idx >= backptrs[step].len() {
                    broken_chain = true;
                    break;
                }
                let (h1, h2) = step_haps[step][idx];
                if h1 < n_ref_haps {
                    if use_dense {
                        if mass_dense[h1] == 0.0 {
                            touched.push(h1);
                        }
                        mass_dense[h1] += post_step_half;
                    } else {
                        *mass_sparse.entry(h1).or_insert(0.0) += post_step_half;
                    }
                }
                if h2 < n_ref_haps {
                    if use_dense {
                        if mass_dense[h2] == 0.0 {
                            touched.push(h2);
                        }
                        mass_dense[h2] += post_step_half;
                    } else {
                        *mass_sparse.entry(h2).or_insert(0.0) += post_step_half;
                    }
                }
                let bp = unpack_backptr(backptrs[step][idx]);
                let prev_idx = bp.prev as usize;
                idx = segment_ptrs[step].get(prev_idx).copied().unwrap_or(0) as usize;
            }
            if broken_chain {
                break;
            }
        }
    }

    if broken_chain {
        return donor_posterior_mass_terminal(beam, &weights, z, n_ref_haps, max_out);
    }

    if use_dense {
        rank_donor_mass_dense(&mass_dense, &touched, max_out)
    } else {
        rank_donor_mass(mass_sparse, max_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::ChromIdx;
    use crate::data::alignment::MarkerAlignment;
    use crate::data::haplotype::Samples;
    use crate::data::marker::{Allele, Marker, Markers, Nucleotide};
    use crate::data::ref_packed::PackedRefView;
    use crate::data::storage::phase_state::{Phased, Unphased};
    use crate::data::storage::{GenotypeColumn, GenotypeMatrix};
    use crate::model::parameters::ModelParams;
    use std::sync::Arc;

    #[test]
    fn beam_score_fields_are_i64() {
        fn assert_i64(_: i64) {}

        let cfg = BeamConfig::default();
        assert_i64(cfg.collapse_gap);
        assert_i64(cfg.prune_tolerance);

        let path = BeamPath {
            hap1: 0,
            hap2: 0,
            cluster1: u16::MAX,
            cluster2: u16::MAX,
            score: i64::MAX - 1,
            model_score: i64::MAX - 1,
            last_swapped: false,
            history_bits: 0,
            history_len: 0,
            prev_idx: 0,
            prev_swapped: false,
        };
        assert_i64(path.score);
        assert_i64(path.model_score);
    }

    #[test]
    fn swap_posteriors_keep_discrimination_beyond_i32_scale() {
        let logsum_unswapped = vec![-3_000.0];
        let logsum_swapped = vec![-3_150.0];
        let p = compute_swap_posteriors(&logsum_swapped, &logsum_unswapped);
        assert_eq!(p.len(), 1);
        // 150 nats separation should produce near-certain mass on the better (unswapped) branch.
        assert!(
            p[0] < 1e-6,
            "expected strong posterior discrimination, got {}",
            p[0]
        );
    }

    fn make_markers() -> Markers<AnyMarkerSpace> {
        let mut markers = Markers::<AnyMarkerSpace>::new();
        markers.add_chrom("chr1");
        markers.push(Marker::new(
            ChromIdx::new(0),
            100,
            None,
            Allele::Base(Nucleotide::A),
            vec![Allele::Base(Nucleotide::C)],
        ));
        markers.push(Marker::new(
            ChromIdx::new(0),
            200,
            None,
            Allele::Base(Nucleotide::A),
            vec![Allele::Base(Nucleotide::C)],
        ));
        markers
    }

    fn make_target_gt() -> GenotypeMatrix<Unphased, AnyMarkerSpace> {
        let markers = make_markers();
        let samples = Arc::new(Samples::from_ids(vec!["T0".to_string()]));
        let col0 = GenotypeColumn::from_alleles(&[0, 1], 2);
        let col1 = GenotypeColumn::from_alleles(&[1, 0], 2);
        GenotypeMatrix::new_unphased(markers, vec![col0, col1], samples)
    }

    fn make_ref_gt() -> GenotypeMatrix<Phased, AnyMarkerSpace> {
        let markers = make_markers();
        let samples = Arc::new(Samples::from_ids(vec![
            "R0".to_string(),
            "R1".to_string(),
            "R2".to_string(),
            "R3".to_string(),
        ]));
        let col0 = GenotypeColumn::from_alleles(&[0, 0, 1, 1, 0, 1, 0, 1], 2);
        let col1 = GenotypeColumn::from_alleles(&[1, 0, 1, 0, 1, 1, 0, 0], 2);
        GenotypeMatrix::new_phased(markers, vec![col0, col1], samples)
    }

    #[test]
    fn fill_pool_alleles_shrinks_with_active_pool() {
        let target_gt = make_target_gt();
        let ref_gt = make_ref_gt();
        let alignment = MarkerAlignment::new(&target_gt, &ref_gt);
        let packed_ref =
            PackedRefView::build_sparse(&target_gt, &ref_gt, &alignment, &[0usize, 1usize])
                .expect("packed ref build should succeed");
        let phaser = BeamPhaser::new(&packed_ref, &ModelParams::default(), BeamConfig::default());
        let mut active_pool = ActivePool::new(ref_gt.n_haplotypes());

        for hap in 0..ref_gt.n_haplotypes() {
            active_pool.touch(hap, 10);
        }

        let mut scratch_pool_alleles = Vec::new();

        {
            let aligned = phaser.fill_pool_alleles(0usize, &active_pool, &mut scratch_pool_alleles);
            assert_eq!(aligned.alleles().len(), active_pool.list().len());
            let mut switch_support = SwitchSupportCache::new();
            switch_support.ensure_initialized(0usize, 10, &aligned);
        }

        active_pool.touch(0, 100);
        active_pool.touch(1, 100);
        active_pool.sweep(100, 0);
        assert_eq!(active_pool.list().len(), 2);

        {
            let aligned = phaser.fill_pool_alleles(1usize, &active_pool, &mut scratch_pool_alleles);
            assert_eq!(aligned.alleles().len(), active_pool.list().len());
            let mut switch_support = SwitchSupportCache::new();
            switch_support.ensure_initialized(1usize, 100, &aligned);
        }
        assert_eq!(scratch_pool_alleles.len(), active_pool.list().len());
    }

    #[test]
    fn donor_mass_uses_trajectory_occupancy_not_terminal_only() {
        let beam = vec![BeamPath {
            hap1: 1,
            hap2: 2,
            cluster1: u16::MAX,
            cluster2: u16::MAX,
            score: 0,
            model_score: 0,
            last_swapped: false,
            history_bits: 0,
            history_len: 0,
            prev_idx: 0,
            prev_swapped: false,
        }];
        let step_haps = vec![vec![(0usize, 0usize)], vec![(1usize, 2usize)]];
        let backptrs = vec![vec![pack_backptr(0, false)], vec![pack_backptr(0, false)]];
        let segment_ptrs = vec![vec![0u32], vec![0u32]];

        let ranked = donor_posterior_mass(&beam, &step_haps, &backptrs, &segment_ptrs, None, 3, 3);
        assert_eq!(ranked.first().map(|(h, _)| *h), Some(0));
        assert!(ranked.iter().any(|(h, _)| *h == 1));
        assert!(ranked.iter().any(|(h, _)| *h == 2));
    }

    #[test]
    fn donor_mass_falls_back_to_terminal_when_chain_is_inconsistent() {
        let beam = vec![BeamPath {
            hap1: 1,
            hap2: 2,
            cluster1: u16::MAX,
            cluster2: u16::MAX,
            score: 0,
            model_score: 0,
            last_swapped: false,
            history_bits: 0,
            history_len: 0,
            prev_idx: 0,
            prev_swapped: false,
        }];
        // Broken chain: missing step_haps entry for the terminal step index.
        let step_haps = vec![vec![(0usize, 0usize)], Vec::new()];
        let backptrs = vec![vec![pack_backptr(0, false)], vec![pack_backptr(0, false)]];
        let segment_ptrs = vec![vec![0u32], vec![0u32]];

        let ranked = donor_posterior_mass(&beam, &step_haps, &backptrs, &segment_ptrs, None, 3, 3);
        assert!(!ranked.is_empty());
        assert!(ranked.iter().any(|(h, _)| *h == 1));
        assert!(ranked.iter().any(|(h, _)| *h == 2));
        assert!(!ranked.iter().any(|(h, _)| *h == 0));
    }

    #[test]
    fn emission_cost_treats_missing_and_out_of_domain_as_neutral() {
        let target_gt = make_target_gt();
        let ref_gt = make_ref_gt();
        let alignment = MarkerAlignment::new(&target_gt, &ref_gt);
        let packed_ref =
            PackedRefView::build_sparse(&target_gt, &ref_gt, &alignment, &[0usize, 1usize])
                .expect("packed ref build should succeed");
        let phaser = BeamPhaser::new(&packed_ref, &ModelParams::default(), BeamConfig::default());
        let missing = crate::data::storage::AlleleCode::MISSING.raw();

        assert_eq!(phaser.emission_cost_for_ref_allele(missing, 0), 0);
        assert_eq!(phaser.emission_cost_for_ref_allele(2, 0), 0);
        assert_eq!(
            phaser.emission_cost_for_ref_allele(0, 0),
            phaser.costs.match_emit_cost
        );
        assert_eq!(
            phaser.emission_cost_for_ref_allele(1, 0),
            phaser.costs.mismatch_emit_cost
        );
    }

    #[test]
    fn prune_collapse_keeps_distinct_haps_even_if_cluster_matches() {
        let target_gt = make_target_gt();
        let ref_gt = make_ref_gt();
        let alignment = MarkerAlignment::new(&target_gt, &ref_gt);
        let packed_ref =
            PackedRefView::build_sparse(&target_gt, &ref_gt, &alignment, &[0usize, 1usize])
                .expect("packed ref build should succeed");
        let phaser = BeamPhaser::new(&packed_ref, &ModelParams::default(), BeamConfig::default());

        let mut beam = vec![
            BeamPath {
                hap1: 0,
                hap2: 1,
                cluster1: 3,
                cluster2: 5,
                score: 10,
                model_score: 10,
                last_swapped: false,
                history_bits: 0,
                history_len: 0,
                prev_idx: 0,
                prev_swapped: false,
            },
            BeamPath {
                hap1: 2,
                hap2: 3,
                cluster1: 3,
                cluster2: 5,
                score: 10,
                model_score: 10,
                last_swapped: false,
                history_bits: 0,
                history_len: 0,
                prev_idx: 0,
                prev_swapped: false,
            },
        ];
        phaser.prune_and_collapse(&mut beam);
        assert_eq!(beam.len(), 2);
    }

    #[test]
    fn decode_swap_track_smooths_isolated_flip_spike() {
        let target_gt = make_target_gt();
        let ref_gt = make_ref_gt();
        let alignment = MarkerAlignment::new(&target_gt, &ref_gt);
        let packed_ref =
            PackedRefView::build_sparse(&target_gt, &ref_gt, &alignment, &[0usize, 1usize])
                .expect("packed ref build should succeed");
        let phaser = BeamPhaser::new(&packed_ref, &ModelParams::default(), BeamConfig::default());

        let calls: Vec<CallSite> = (0..5)
            .map(|i| CallSite {
                marker: MarkerIdx::new(i as u32),
                hi_idx: i,
                a1: 0,
                a2: 1,
                phase_conf: 0.5,
                a1_freq: 0.5,
                a2_freq: 0.5,
                pbwt_len_morgans_a1: 0.0,
                pbwt_len_morgans_a2: 0.0,
                pbwt_density_a1: 0.0,
                pbwt_density_a2: 0.0,
                dist_morgans: 0.001,
                flip_cost: 0,
                fixed: false,
            })
            .collect();

        let logsum_unswapped = vec![0.0, 0.0, -1.0, 0.0, 0.0];
        let logsum_swapped = vec![-2.0, -2.0, 0.0, -2.0, -2.0];
        let fallback = vec![false; calls.len()];
        let decoded =
            phaser.decode_swap_track(&logsum_swapped, &logsum_unswapped, &calls, &fallback);

        assert_eq!(decoded, vec![false, false, false, false, false]);
    }

    #[test]
    fn pbwt_adjusted_switch_prob_discourages_switch_for_strong_match() {
        let target_gt = make_target_gt();
        let ref_gt = make_ref_gt();
        let alignment = MarkerAlignment::new(&target_gt, &ref_gt);
        let packed_ref =
            PackedRefView::build_sparse(&target_gt, &ref_gt, &alignment, &[0usize, 1usize])
                .expect("packed ref build should succeed");
        let phaser = BeamPhaser::new(&packed_ref, &ModelParams::default(), BeamConfig::default());

        let mut pool = ActivePool::new(ref_gt.n_haplotypes());
        let hap = 1usize;
        let version = 42u32;
        let base = 0.02f32;
        let no_meta = phaser.adjust_switch_prob_from_pbwt(base, &pool, 0, hap, version);

        pool.set_pbwt_meta(0, hap, 0, 2.0, 0.03, version);
        let strong_meta = phaser.adjust_switch_prob_from_pbwt(base, &pool, 0, hap, version);
        assert!(strong_meta < no_meta, "expected stronger stay preference");

        pool.set_pbwt_meta(0, hap, 0, 64.0, 0.0001, version);
        let weak_meta = phaser.adjust_switch_prob_from_pbwt(base, &pool, 0, hap, version);
        assert!(
            weak_meta > strong_meta,
            "expected weaker match to allow more switching"
        );
    }

    #[test]
    fn orientation_search_prior_cost_is_small_and_confidence_sensitive() {
        let target_gt = make_target_gt();
        let ref_gt = make_ref_gt();
        let alignment = MarkerAlignment::new(&target_gt, &ref_gt);
        let packed_ref =
            PackedRefView::build_sparse(&target_gt, &ref_gt, &alignment, &[0usize, 1usize])
                .expect("packed ref build should succeed");
        let phaser = BeamPhaser::new(&packed_ref, &ModelParams::default(), BeamConfig::default());

        let neutral = phaser.orientation_search_prior_cost(false, 0.5);
        let high_conf_correct = phaser.orientation_search_prior_cost(false, 0.99);
        let high_conf_wrong = phaser.orientation_search_prior_cost(true, 0.99);
        assert_eq!(neutral, 0);
        assert!(high_conf_correct >= 0);
        assert!(high_conf_wrong > high_conf_correct);
        assert!(
            high_conf_wrong < 300_000,
            "search prior should remain small relative to main model terms"
        );
    }
}
