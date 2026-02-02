//! # Phasing Pipeline
//!
//! Orchestrates the phasing workflow:
//! 1. Load target VCF
//! 2. Classify markers into Stage 1 (high-frequency) and Stage 2 (low-frequency/rare)
//! 3. Build PBWT for haplotype matching
//! 4. Run PBWT-accelerated Li-Stephens HMM (PhasingHmm) on Stage 1 markers
//! 5. Update phase and iterate
//! 6. Collect EM parameter estimates and update
//! 7. Run Stage 2 phasing: interpolate state probabilities to phase rare variants
//! 8. Write phased output
//!
//! This implements Beagle's two-stage phasing algorithm for handling rare variants.

use std::collections::HashMap;
use std::sync::Arc;

use bitvec::prelude::*;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use tracing::{info_span, instrument};

use crate::config::Config;
use crate::data::genetic_map::{GeneticMaps, MarkerMap};
use crate::data::haplotype::{HapIdx, SampleIdx};
use crate::data::marker::{AnyMarkerSpace, MarkerIdx};
use crate::data::storage::phase_state::Phased;
use crate::data::storage::sample_phase::SamplePhase;
use crate::data::storage::{GenotypeColumn, GenotypeMatrix, GenotypeView, MutableGenotypes};
use crate::error::Result;
use crate::io::bref3::Bref3Reader;
use crate::io::streaming::{
    GlobalHapId, HaplotypePriors, PhasedOverlap, StateProbs, StreamWindow, StreamingConfig,
    StreamingVcfReader,
};
use crate::io::vcf::{VcfReader, VcfWriter};
use crate::model::ibs2::Ibs2;
use crate::model::pl_emission::{
    PlProvider, allele_probs_cond_from_pl, allele_probs_uncond_from_pl, emit_from_allele_probs,
};

thread_local! {
    static THREAD_WORKSPACE: std::cell::RefCell<Option<crate::utils::workspace::ThreadWorkspace>> =
        std::cell::RefCell::new(None);
}

/// Helper struct for double-buffered window processing
struct StreamWindowWithResult {
    window: StreamWindow,
    phased_result: Option<GenotypeMatrix<Phased>>,
}

impl std::ops::Deref for StreamWindowWithResult {
    type Target = StreamWindow;
    fn deref(&self) -> &Self::Target {
        &self.window
    }
}
use crate::data::alignment::{AlignmentStats, MarkerAlignment};
use crate::model::types::{combined_from_ref, CombinedHapId, CombinedHapSpace, RefHapId};
use crate::model::states::ThreadedHaps;
use crate::model::hmm::MosaicHmm;
use crate::model::parameters::ModelParams;
use crate::model::beam::{BeamConfig, BeamPhaser, ActivePool, PbwtBeamIndex, PbwtInjector};
use crate::data::condensed::CondensedTarget;
use crate::data::ref_packed::PackedRefView;
use crate::model::phase_ibs::BidirectionalPhaseIbs;
use crate::model::reference_pbwt::{PbwtQueryAllele, RankBeam, ReferencePbwt};
use crate::model::state_allocator::allocate_lms_sparse;
use crate::utils::telemetry::{Stage, TelemetryBlackboard};
use mini_mcmc::core::{MarkovChain, Trace};
use sysinfo::System;

const STAGE1_BLOCK_MIN_CM: f64 = 0.01;
const STAGE1_BLOCK_MAX_CM: f64 = 0.2;
const STAGE1_BLOCK_TARGET_MARKERS: usize = 200;
const STAGE1_BLOCK_MIN_MARKERS: usize = 10;
const PBWT_SELECT_BLOCK_CM: f64 = 0.1;
const PBWT_MIN_MARKER_STEP: usize = 50;
const PBWT_MIN_SAMPLE_POINTS: usize = 10;
const PBWT_PER_WINDOW_MULT: usize = 8;
const PBWT_MIN_PER_HAP: usize = 64;
const PBWT_MAX_PER_HAP: usize = 256;
const PBWT_FORCE_TOP_HAPS: usize = 8;
const PBWT_ANCHOR_TOP_HAPS: usize = 32;
const SCAN_RAM_FRACTION: f64 = 0.10;
const PHASE_RAM_FRACTION: f64 = 0.15;
const PHASE_STATE_BUDGET_SAFETY: f64 = 0.6;
const MIN_AVAIL_BYTES_FOR_PLANNING: u64 = 64 * 1024 * 1024;
const INVALID_ALLELE: u8 = 254;

struct RefAlleleProvider<'a, TargetSpace = AnyMarkerSpace, RefSpace = AnyMarkerSpace> {
    ref_gt: GenotypeView<'a, TargetSpace, RefSpace>,
    threaded_haps: &'a ThreadedHaps<CombinedHapSpace>,
    state_buf: Vec<CombinedHapId>,
}

impl<'a, TargetSpace, RefSpace> RefAlleleProvider<'a, TargetSpace, RefSpace> {
    fn new(
        ref_gt: GenotypeView<'a, TargetSpace, RefSpace>,
        threaded_haps: &'a ThreadedHaps<CombinedHapSpace>,
    ) -> Self {
        let n_states = threaded_haps.n_states();
        Self {
            ref_gt,
            threaded_haps,
            state_buf: vec![CombinedHapId::from(0u32); n_states],
        }
    }

    #[inline]
    fn fill_ref_alleles(&mut self, marker: usize, out: &mut [u8]) {
        let n_states = self.threaded_haps.n_states().min(out.len());
        if self.state_buf.len() < n_states {
            self.state_buf.resize(n_states, CombinedHapId::from(0u32));
        }
        self.threaded_haps.materialize_at(marker, &mut self.state_buf);
        let marker_idx = MarkerIdx::new(marker as u32);
        for i in 0..n_states {
            let hap = HapIdx::new(self.state_buf[i].as_u32());
            out[i] = self.ref_gt.allele(marker_idx, hap);
        }
    }
}

fn partition_markers_by_cm(gen_positions: &[f64], block_cm: f64) -> Vec<(usize, usize)> {
    if gen_positions.is_empty() {
        return Vec::new();
    }
    let mut blocks = Vec::new();
    let mut start = 0usize;
    while start < gen_positions.len() {
        let start_pos = gen_positions[start];
        let mut end = start + 1;
        let limit = start_pos + block_cm;
        while end < gen_positions.len() && gen_positions[end] < limit {
            end += 1;
        }
        let min_end = (start + STAGE1_BLOCK_MIN_MARKERS).min(gen_positions.len());
        if end < min_end {
            end = min_end;
        }
        if end <= start {
            end = start.saturating_add(1).min(gen_positions.len());
        }
        blocks.push((start, end));
        start = end;
    }
    blocks
}

fn stage1_block_cm(gen_positions: &[f64]) -> f64 {
    if gen_positions.len() < 2 {
        return STAGE1_BLOCK_MIN_CM;
    }
    let span = (gen_positions[gen_positions.len() - 1] - gen_positions[0]).abs();
    let avg = span / (gen_positions.len().saturating_sub(1).max(1) as f64);
    let block = avg * STAGE1_BLOCK_TARGET_MARKERS as f64;
    block.clamp(STAGE1_BLOCK_MIN_CM, STAGE1_BLOCK_MAX_CM)
}

fn available_memory_bytes() -> Option<u64> {
    fn read_cgroup_limit_bytes() -> Option<u64> {
        let v2 = std::fs::read_to_string("/sys/fs/cgroup/memory.max").ok();
        if let Some(s) = v2 {
            let t = s.trim();
            if t != "max" {
                if let Ok(v) = t.parse::<u64>() {
                    if v > 0 && v < (1u64 << 60) {
                        return Some(v);
                    }
                }
            }
        }
        let v1 = std::fs::read_to_string("/sys/fs/cgroup/memory/memory.limit_in_bytes").ok();
        if let Some(s) = v1 {
            let t = s.trim();
            if let Ok(v) = t.parse::<u64>() {
                if v > 0 && v < (1u64 << 60) {
                    return Some(v);
                }
            }
        }
        None
    }

    fn read_cgroup_available_bytes(limit: u64) -> Option<u64> {
        let v2 = std::fs::read_to_string("/sys/fs/cgroup/memory.current").ok();
        if let Some(s) = v2 {
            if let Ok(cur) = s.trim().parse::<u64>() {
                return Some(limit.saturating_sub(cur));
            }
        }
        let v1 = std::fs::read_to_string("/sys/fs/cgroup/memory/memory.usage_in_bytes").ok();
        if let Some(s) = v1 {
            if let Ok(cur) = s.trim().parse::<u64>() {
                return Some(limit.saturating_sub(cur));
            }
        }
        None
    }

    let mut sys = System::new();
    sys.refresh_memory();
    let mut avail_bytes = sys.available_memory();
    let mut total_bytes = sys.total_memory();
    if total_bytes > 0 {
        let scaled_total = total_bytes.saturating_mul(1024);
        let looks_like_kib = total_bytes < 1_073_741_824
            && scaled_total >= 1_073_741_824
            && scaled_total <= (1u64 << 50);
        if looks_like_kib {
            avail_bytes = avail_bytes.saturating_mul(1024);
            total_bytes = scaled_total;
        }
    }
    if let Some(limit) = read_cgroup_limit_bytes() {
        if limit > 0 {
            total_bytes = total_bytes.min(limit);
            if let Some(avail) = read_cgroup_available_bytes(limit) {
                avail_bytes = avail_bytes.min(avail);
            } else {
                avail_bytes = avail_bytes.min(limit);
            }
        }
    }

    if avail_bytes >= MIN_AVAIL_BYTES_FOR_PLANNING {
        return Some(avail_bytes);
    }
    if total_bytes > 0 {
        Some(total_bytes)
    } else {
        None
    }
}

fn estimate_phase_state_budget(
    available_bytes: u64,
    n_threads: usize,
    window_markers: usize,
) -> usize {
    if available_bytes == 0 || n_threads == 0 || window_markers == 0 {
        return 0;
    }
    let per_state_bytes = 16usize.saturating_add(window_markers.saturating_mul(5));
    if per_state_bytes == 0 {
        return 0;
    }
    let budget = (available_bytes as f64 * PHASE_RAM_FRACTION) as u64;
    let per_thread = budget / n_threads.max(1) as u64;
    let safe_bytes = (per_thread as f64 * PHASE_STATE_BUDGET_SAFETY) as u64;
    (safe_bytes as usize) / per_state_bytes
}

fn estimate_scan_batch_size(available_bytes: u64, n_ref_haps: usize, n_target_haps: usize) -> usize {
    if available_bytes == 0 || n_ref_haps == 0 || n_target_haps == 0 {
        return 1;
    }
    let per_hap_bytes = (n_ref_haps as u64).saturating_mul(20);
    if per_hap_bytes == 0 {
        return 1;
    }
    let budget = (available_bytes as f64 * SCAN_RAM_FRACTION) as u64;
    let mut batch = (budget / per_hap_bytes) as usize;
    if batch == 0 {
        batch = 1;
    }
    batch.min(n_target_haps)
}

fn build_sampling_points(
    gen_positions: &[f64],
    step_cm: f64,
    min_marker_step: usize,
    informative: Option<&[bool]>,
) -> Vec<bool> {
    let n = gen_positions.len();
    let mut sampling = vec![false; n];
    if n == 0 {
        return sampling;
    }
    let min_step = min_marker_step.min((n / PBWT_MIN_SAMPLE_POINTS).max(1));
    let step = step_cm.max(1e-6);
    let mut next_cm = gen_positions[0];
    let mut next_marker = 0usize;
    for m in 0..n {
        let cm = gen_positions[m];
        if cm >= next_cm || m >= next_marker {
            sampling[m] = true;
            next_cm = cm + step;
            next_marker = m.saturating_add(min_step);
        }
        if informative
            .and_then(|flags| flags.get(m))
            .copied()
            .unwrap_or(false)
        {
            sampling[m] = true;
        }
    }
    sampling[n - 1] = true;
    let count = sampling.iter().filter(|&&b| b).count();
    eprintln!(
        "[pbwt sampling] markers={} step_cm={:.6} min_step={} sampled={} first_cm={:.6} last_cm={:.6}",
        n,
        step,
        min_step,
        count,
        gen_positions.first().copied().unwrap_or(0.0),
        gen_positions.last().copied().unwrap_or(0.0)
    );
    sampling
}

fn select_top_k(scores: &[f32], k: usize) -> Vec<(usize, f32)> {
    if k == 0 || scores.is_empty() {
        return Vec::new();
    }
    let mut ranked: Vec<(usize, f32)> = scores
        .iter()
        .enumerate()
        .filter(|&(_, &s)| s.is_finite() && s > 0.0)
        .map(|(i, &s)| (i, s))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if ranked.len() > k {
        ranked.truncate(k);
    }
    ranked
}

fn build_sparse_scores(
    window_scores: &[Vec<(usize, f32)>],
    abyss: &[bool],
) -> (Vec<usize>, Vec<Vec<(usize, f32)>>) {
    let mut map: HashMap<usize, usize> = HashMap::new();
    let mut candidate_haps: Vec<usize> = Vec::new();
    let mut scores_by_hap: Vec<Vec<(usize, f32)>> = Vec::new();

    for (w, list) in window_scores.iter().enumerate() {
        for &(hap, score) in list.iter() {
            if score <= 0.0 || !score.is_finite() {
                continue;
            }
            if hap < abyss.len() && abyss[hap] {
                continue;
            }
            let idx = *map.entry(hap).or_insert_with(|| {
                candidate_haps.push(hap);
                scores_by_hap.push(Vec::new());
                candidate_haps.len() - 1
            });
            scores_by_hap[idx].push((w, score));
        }
    }
    for scores in scores_by_hap.iter_mut() {
        scores.sort_by_key(|(w, _)| *w);
    }
    (candidate_haps, scores_by_hap)
}

fn compute_ref_freqs<TargetSpace, RefSpace>(
    target_gt: &GenotypeMatrix<impl crate::data::storage::phase_state::PhaseState, TargetSpace>,
    ref_columns: &[GenotypeColumn],
    alignment: Option<&MarkerAlignment<TargetSpace, RefSpace>>,
    marker_map: Option<&[usize]>,
    ref_index_map: Option<&[usize]>,
    n_markers: usize,
) -> Vec<Vec<f32>> {
    let n_ref_haps = ref_columns
        .first()
        .map(|c| c.n_haplotypes())
        .unwrap_or(0);
    let mut freqs: Vec<Vec<f32>> = Vec::with_capacity(n_markers);
    for m in 0..n_markers {
        let orig_m = marker_map.map(|map| map[m]).unwrap_or(m);
        let n_alleles = target_gt
            .markers()
            .marker(MarkerIdx::new(orig_m as u32))
            .n_alleles();
        let mut counts = vec![0u32; n_alleles.max(1)];
        let mut total = 0u32;
        if let Some(alignment) = alignment {
            if let Some(ref_m) = alignment.target_to_ref(MarkerIdx::new(orig_m as u32)) {
                let ref_idx = if let Some(map) = ref_index_map {
                    let idx = map[ref_m.as_usize()];
                    if idx == usize::MAX {
                        freqs.push(vec![0.0f32; counts.len()]);
                        continue;
                    }
                    idx
                } else {
                    ref_m.as_usize()
                };
                for rh in 0..n_ref_haps {
                    let ref_a = ref_columns[ref_idx].get(HapIdx::new(rh as u32));
                    let mapped = alignment.reverse_map_allele(orig_m, ref_a);
                    if mapped == 255 {
                        continue;
                    }
                    let idx = mapped as usize;
                    if idx < counts.len() {
                        counts[idx] += 1;
                        total += 1;
                    }
                }
            }
        } else {
            let ref_idx = if let Some(map) = ref_index_map {
                if orig_m >= map.len() {
                    freqs.push(vec![0.0f32; counts.len()]);
                    continue;
                }
                let idx = map[orig_m];
                if idx == usize::MAX {
                    freqs.push(vec![0.0f32; counts.len()]);
                    continue;
                }
                idx
            } else {
                if orig_m >= ref_columns.len() {
                    freqs.push(vec![0.0f32; counts.len()]);
                    continue;
                }
                orig_m
            };
            for rh in 0..n_ref_haps {
                let ref_a = ref_columns[ref_idx].get(HapIdx::new(rh as u32));
                if ref_a == 255 {
                    continue;
                }
                let idx = ref_a as usize;
                if idx < counts.len() {
                    counts[idx] += 1;
                    total += 1;
                }
            }
        }
        let mut out = vec![0.0f32; counts.len()];
        if total > 0 {
            let inv = 1.0 / total as f32;
            for (i, c) in counts.into_iter().enumerate() {
                out[i] = c as f32 * inv;
            }
        }
        freqs.push(out);
    }
    freqs
}

fn score_window_batch_pbwt_segment<TargetSpace, RefSpace>(
    batch_haps: &[usize],
    geno: &MutableGenotypes,
    ref_columns: &[GenotypeColumn],
    phase_mask: Option<&Vec<Vec<u8>>>,
    mask_unphased_hets: Option<&[bool]>,
    alignment: Option<&MarkerAlignment<TargetSpace, RefSpace>>,
    freqs: &[Vec<f32>],
    window: (usize, usize),
    k_per_hap: usize,
    sampling: &[bool],
    window_scores: &mut [Vec<f32>],
    exclude_self: bool,
    marker_map: Option<&[usize]>,
    ref_index_map: Option<&[usize]>,
) {
    let n_ref_haps = ref_columns
        .first()
        .map(|c| c.n_haplotypes())
        .unwrap_or(0);
    if batch_haps.is_empty() || n_ref_haps == 0 {
        return;
    }
    let (start, end) = window;
    if start >= end {
        return;
    }

    let mut pbwt_fwd = ReferencePbwt::new(n_ref_haps);
    let mut beams_fwd: Vec<RankBeam> = (0..batch_haps.len())
        .map(|_| RankBeam::full(n_ref_haps as u32))
        .collect();
    let mut ref_alleles = vec![0u8; n_ref_haps];
    let mut query_alleles = vec![PbwtQueryAllele::missing(); batch_haps.len()];
    let mut donors_buf: Vec<u32> = Vec::new();

    let min_freq = 1.0 / (2.0 * n_ref_haps.max(1) as f32);

    for m in start..end {
        let local_idx = m - start;
        let orig_m = marker_map
            .and_then(|map| map.get(m).copied())
            .unwrap_or(m);
        for (i, &hap_idx) in batch_haps.iter().enumerate() {
            let sample_idx = hap_idx / 2;
            let phased = phase_mask
                .and_then(|mask| mask.get(orig_m).and_then(|row| row.get(sample_idx)))
                .copied()
                .unwrap_or(0);
            if phased == 0
                && mask_unphased_hets
                    .and_then(|flags| flags.get(sample_idx))
                    .copied()
                    .unwrap_or(false)
            {
                let hap1 = sample_idx * 2;
                let hap2 = hap1 + 1;
                let a1 = geno.get(orig_m, HapIdx::new(hap1 as u32));
                let a2 = geno.get(orig_m, HapIdx::new(hap2 as u32));
                if a1 != 255 && a1 == a2 {
                    query_alleles[i] =
                        PbwtQueryAllele::allele(a1).unwrap_or_else(PbwtQueryAllele::missing);
                } else {
                    query_alleles[i] = PbwtQueryAllele::wildcard();
                }
            } else {
                let qa = geno.get(orig_m, HapIdx::new(hap_idx as u32));
                query_alleles[i] =
                    PbwtQueryAllele::allele(qa).unwrap_or_else(PbwtQueryAllele::missing);
            }
        }

        if let Some(alignment) = alignment {
            if let Some(ref_m) = alignment.target_to_ref(MarkerIdx::new(orig_m as u32)) {
                let ref_idx = if let Some(map) = ref_index_map {
                    let idx = map[ref_m.as_usize()];
                    if idx == usize::MAX {
                        ref_alleles.fill(255);
                        continue;
                    }
                    idx
                } else {
                    ref_m.as_usize()
                };
                for rh in 0..n_ref_haps {
                    let ref_a = ref_columns[ref_idx].get(HapIdx::new(rh as u32));
                    ref_alleles[rh] = alignment.reverse_map_allele(orig_m, ref_a);
                }
            } else {
                ref_alleles.fill(255);
            }
        } else {
            let ref_idx = if let Some(map) = ref_index_map {
                if orig_m >= map.len() {
                    ref_alleles.fill(255);
                    continue;
                }
                let idx = map[orig_m];
                if idx == usize::MAX {
                    ref_alleles.fill(255);
                    continue;
                }
                idx
            } else {
                if orig_m >= ref_columns.len() {
                    ref_alleles.fill(255);
                    continue;
                }
                orig_m
            };
            for rh in 0..n_ref_haps {
                ref_alleles[rh] = ref_columns[ref_idx].get(HapIdx::new(rh as u32));
            }
        }

        let mut is_biallelic = true;
        for &a in ref_alleles.iter() {
            if a >= 2 && a != 255 {
                is_biallelic = false;
                break;
            }
        }
        if is_biallelic {
            for &q in &query_alleles {
                let a = q.value();
                if a == PbwtQueryAllele::WILDCARD_VALUE {
                    continue;
                }
                if a >= 2 && a != 255 {
                    is_biallelic = false;
                    break;
                }
            }
        }
        let n_alleles = if is_biallelic { 2 } else { 256 };

        pbwt_fwd.advance_with_beams_query(
            &ref_alleles,
            n_alleles,
            local_idx,
            &query_alleles,
            &mut beams_fwd,
        );

        if sampling.get(local_idx).copied().unwrap_or(false) {
            for (i, &hap_idx) in batch_haps.iter().enumerate() {
                let targ = query_alleles[i].value();
                if targ == 255 {
                    continue;
                }
                if targ == PbwtQueryAllele::WILDCARD_VALUE {
                    pbwt_fwd.select_donors_into(&beams_fwd[i], k_per_hap, &mut donors_buf);
                    for &d in donors_buf.iter() {
                        let idx = d as usize;
                        if idx >= n_ref_haps {
                            continue;
                        }
                        if exclude_self && idx / 2 == hap_idx / 2 {
                            continue;
                        }
                        let ref_allele = ref_alleles[idx];
                        if ref_allele == 255 {
                            continue;
                        }
                        if ref_allele != 0 && ref_allele != 1 {
                            continue;
                        }
                        let freq = freqs
                            .get(m)
                            .and_then(|f| f.get(ref_allele as usize))
                            .copied()
                            .unwrap_or(0.0);
                        if freq <= 0.0 {
                            continue;
                        }
                        let weight = -(freq.max(min_freq)).ln();
                        let w = &mut window_scores[i][idx];
                        if w.is_finite() {
                            *w += weight;
                        } else {
                            *w = weight;
                        }
                    }
                    continue;
                }
                let freq = freqs
                    .get(m)
                    .and_then(|f| f.get(targ as usize))
                    .copied()
                    .unwrap_or(0.0);
                if freq <= 0.0 {
                    continue;
                }
                let weight = -(freq.max(min_freq)).ln();
                pbwt_fwd.select_donors_into(&beams_fwd[i], k_per_hap, &mut donors_buf);
                for &d in donors_buf.iter() {
                    let idx = d as usize;
                    if idx >= n_ref_haps {
                        continue;
                    }
                    if exclude_self && idx / 2 == hap_idx / 2 {
                        continue;
                    }
                    let ref_a = ref_alleles[idx];
                    if ref_a == 255 || ref_a != targ {
                        continue;
                    }
                    let w = &mut window_scores[i][idx];
                    if w.is_finite() {
                        *w += weight;
                    } else {
                        *w = weight;
                    }
                }
            }
        }
    }

    let mut pbwt_bwd = ReferencePbwt::new(n_ref_haps);
    let mut beams_bwd: Vec<RankBeam> = (0..batch_haps.len())
        .map(|_| RankBeam::full(n_ref_haps as u32))
        .collect();
    for (rev_step, m) in (start..end).rev().enumerate() {
        let local_idx = end - start - 1 - rev_step;
        let orig_m = marker_map
            .and_then(|map| map.get(m).copied())
            .unwrap_or(m);
        for (i, &hap_idx) in batch_haps.iter().enumerate() {
            let sample_idx = hap_idx / 2;
            let phased = phase_mask
                .and_then(|mask| mask.get(orig_m).and_then(|row| row.get(sample_idx)))
                .copied()
                .unwrap_or(0);
            if phased == 0
                && mask_unphased_hets
                    .and_then(|flags| flags.get(sample_idx))
                    .copied()
                    .unwrap_or(false)
            {
                let hap1 = sample_idx * 2;
                let hap2 = hap1 + 1;
                let a1 = geno.get(orig_m, HapIdx::new(hap1 as u32));
                let a2 = geno.get(orig_m, HapIdx::new(hap2 as u32));
                if a1 != 255 && a1 == a2 {
                    query_alleles[i] =
                        PbwtQueryAllele::allele(a1).unwrap_or_else(PbwtQueryAllele::missing);
                } else {
                    query_alleles[i] = PbwtQueryAllele::wildcard();
                }
            } else {
                let qa = geno.get(orig_m, HapIdx::new(hap_idx as u32));
                query_alleles[i] =
                    PbwtQueryAllele::allele(qa).unwrap_or_else(PbwtQueryAllele::missing);
            }
        }

        if let Some(alignment) = alignment {
            if let Some(ref_m) = alignment.target_to_ref(MarkerIdx::new(orig_m as u32)) {
                let ref_idx = if let Some(map) = ref_index_map {
                    let idx = map[ref_m.as_usize()];
                    if idx == usize::MAX {
                        ref_alleles.fill(255);
                        continue;
                    }
                    idx
                } else {
                    ref_m.as_usize()
                };
                for rh in 0..n_ref_haps {
                    let ref_a = ref_columns[ref_idx].get(HapIdx::new(rh as u32));
                    ref_alleles[rh] = alignment.reverse_map_allele(orig_m, ref_a);
                }
            } else {
                ref_alleles.fill(255);
            }
        } else {
            let ref_idx = if let Some(map) = ref_index_map {
                if orig_m >= map.len() {
                    ref_alleles.fill(255);
                    continue;
                }
                let idx = map[orig_m];
                if idx == usize::MAX {
                    ref_alleles.fill(255);
                    continue;
                }
                idx
            } else {
                if orig_m >= ref_columns.len() {
                    ref_alleles.fill(255);
                    continue;
                }
                orig_m
            };
            for rh in 0..n_ref_haps {
                ref_alleles[rh] = ref_columns[ref_idx].get(HapIdx::new(rh as u32));
            }
        }

        let mut is_biallelic = true;
        for &a in ref_alleles.iter() {
            if a >= 2 && a != 255 {
                is_biallelic = false;
                break;
            }
        }
        if is_biallelic {
            for &q in &query_alleles {
                let a = q.value();
                if a == PbwtQueryAllele::WILDCARD_VALUE {
                    continue;
                }
                if a >= 2 && a != 255 {
                    is_biallelic = false;
                    break;
                }
            }
        }
        let n_alleles = if is_biallelic { 2 } else { 256 };

        pbwt_bwd.advance_with_beams_query(
            &ref_alleles,
            n_alleles,
            rev_step,
            &query_alleles,
            &mut beams_bwd,
        );

        if sampling.get(local_idx).copied().unwrap_or(false) {
            for (i, &hap_idx) in batch_haps.iter().enumerate() {
                let targ = query_alleles[i].value();
                if targ == 255 {
                    continue;
                }
                if targ == PbwtQueryAllele::WILDCARD_VALUE {
                    pbwt_bwd.select_donors_into(&beams_bwd[i], k_per_hap, &mut donors_buf);
                    for &d in donors_buf.iter() {
                        let idx = d as usize;
                        if idx >= n_ref_haps {
                            continue;
                        }
                        if exclude_self && idx / 2 == hap_idx / 2 {
                            continue;
                        }
                        let ref_allele = ref_alleles[idx];
                        if ref_allele == 255 {
                            continue;
                        }
                        if ref_allele != 0 && ref_allele != 1 {
                            continue;
                        }
                        let freq = freqs
                            .get(m)
                            .and_then(|f| f.get(ref_allele as usize))
                            .copied()
                            .unwrap_or(0.0);
                        if freq <= 0.0 {
                            continue;
                        }
                        let weight = -(freq.max(min_freq)).ln();
                        let w = &mut window_scores[i][idx];
                        if w.is_finite() {
                            *w += weight;
                        } else {
                            *w = weight;
                        }
                    }
                    continue;
                }
                let freq = freqs
                    .get(m)
                    .and_then(|f| f.get(targ as usize))
                    .copied()
                    .unwrap_or(0.0);
                if freq <= 0.0 {
                    continue;
                }
                let weight = -(freq.max(min_freq)).ln();
                pbwt_bwd.select_donors_into(&beams_bwd[i], k_per_hap, &mut donors_buf);
                for &d in donors_buf.iter() {
                    let idx = d as usize;
                    if idx >= n_ref_haps {
                        continue;
                    }
                    if exclude_self && idx / 2 == hap_idx / 2 {
                        continue;
                    }
                    let ref_a = ref_alleles[idx];
                    if ref_a == 255 || ref_a != targ {
                        continue;
                    }
                    let w = &mut window_scores[i][idx];
                    if w.is_finite() {
                        *w += weight;
                    } else {
                        *w = weight;
                    }
                }
            }
        }
    }
}

/// Phasing pipeline
pub struct PhasingPipeline<RefSpace = crate::data::AnyMarkerSpace> {
    config: Config,
    params: ModelParams,
    /// Reference panel for reference-guided phasing (optional)
    /// Uses Arc for shared ownership to avoid cloning the large reference panel
    reference_gt: Option<Arc<GenotypeMatrix<Phased, RefSpace>>>,
    /// Marker alignment between target and reference
    alignment: Option<MarkerAlignment<crate::data::AnyMarkerSpace, RefSpace>>,
    telemetry: Option<Arc<TelemetryBlackboard>>,
}

struct FwdCheckpoints {
    block_starts: Arc<[usize]>,
    n_states: usize,
    data: Vec<f32>,
}

impl FwdCheckpoints {
    fn from_buffer(block_starts: Arc<[usize]>, n_states: usize, mut data: Vec<f32>) -> Self {
        let n_blocks = block_starts.len().max(1);
        let required = n_blocks * n_states;
        if data.len() < required {
            data.resize(required, 0.0);
        } else {
            data[..required].fill(0.0);
        }
        Self {
            n_states,
            block_starts,
            data,
        }
    }

    fn into_buffer(self) -> Vec<f32> {
        self.data
    }

    fn block_slice(&self, block_idx: usize) -> &[f32] {
        let start = block_idx * self.n_states;
        &self.data[start..start + self.n_states]
    }

    fn block_slice_mut(&mut self, block_idx: usize) -> &mut [f32] {
        let start = block_idx * self.n_states;
        &mut self.data[start..start + self.n_states]
    }
}

fn blocks_to_starts(blocks: &[(usize, usize)], n_markers: usize) -> Vec<usize> {
    if n_markers == 0 {
        return Vec::new();
    }
    if blocks.is_empty() {
        return vec![0];
    }
    let mut out = Vec::with_capacity(blocks.len());
    for &(s, _) in blocks {
        if s < n_markers {
            out.push(s);
        }
    }
    if out.first().copied().unwrap_or(usize::MAX) != 0 {
        out.insert(0, 0);
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn max_block_len_from_starts(block_starts: &[usize], n_markers: usize) -> usize {
    if n_markers == 0 {
        return 0;
    }
    if block_starts.is_empty() {
        return n_markers.max(1);
    }
    let mut max_len = 1usize;
    for (i, &s) in block_starts.iter().enumerate() {
        let e = block_starts.get(i + 1).copied().unwrap_or(n_markers);
        if e > s {
            max_len = max_len.max(e - s);
        }
    }
    max_len
}

/// Overlap handoff payload for streaming windows.
///
/// `state_probs` is kept for compatibility and intra-window diagnostics, but
/// cross-window continuity should use `hap_priors`, which is identity-aware.
#[derive(Clone, Debug)]
pub struct Stage2OverlapHandoff {
    state_probs: Option<StateProbs>,
    hap_priors: Option<Vec<HaplotypePriors>>,
    prior_stage1_global_marker: Option<usize>,
}

#[derive(Debug, Clone)]
struct MosaicTrace {
    mean_state: f64,
    switch_rate: f64,
    log_likelihood: f64,
}

impl Trace for MosaicTrace {
    fn trace(&self) -> Vec<f64> {
        vec![self.mean_state, self.switch_rate, self.log_likelihood]
    }
}

struct MosaicBuffers {
    fwd: aligned_vec::AVec<f32, aligned_vec::ConstAlign<32>>,
    fwd_prior: aligned_vec::AVec<f32, aligned_vec::ConstAlign<32>>,
    ref_alleles: Vec<u8>,
    weights: Vec<f32>,
    allele_probs: Vec<f32>,
    hap1_checkpoints: FwdCheckpoints,
    hap2_checkpoints: FwdCheckpoints,
    hap1_allele: Vec<u8>,
    hap1_partner_allele: Vec<u8>,
    hap1_use_combined: Vec<bool>,
    hap1_hard_match: Vec<bool>,
    hap2_allele: Vec<u8>,
    hap2_partner_allele: Vec<u8>,
    hap2_use_combined: Vec<bool>,
    hap2_hard_match: Vec<bool>,
    path1: Vec<u32>,
    path2: Vec<u32>,
    fwd_block: Vec<f32>,
}

#[derive(Clone, Debug)]
struct MosaicPaths {
    path1: Vec<u32>,
    path2: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct GlobalMosaicPaths {
    pub path1: Vec<CombinedHapId>,
    pub path2: Vec<CombinedHapId>,
}

fn local_to_global_paths(
    local: &MosaicPaths,
    threaded: &crate::model::states::ThreadedHaps<CombinedHapSpace>,
    n_markers: usize,
) -> GlobalMosaicPaths {
    let n_states = threaded.n_states();
    let mut buffer = vec![CombinedHapId::from(0u32); n_states];
    let mut path1 = Vec::with_capacity(n_markers);
    let mut path2 = Vec::with_capacity(n_markers);

    for m in 0..n_markers {
        threaded.materialize_at(m, &mut buffer);
        let s1 = local.path1[m] as usize;
        let s2 = local.path2[m] as usize;

        path1.push(if s1 < n_states {
            buffer[s1]
        } else {
            CombinedHapId::from(0u32)
        });
        path2.push(if s2 < n_states {
            buffer[s2]
        } else {
            CombinedHapId::from(0u32)
        });
    }

    GlobalMosaicPaths { path1, path2 }
}

fn global_to_local_paths(
    global: &GlobalMosaicPaths,
    threaded: &crate::model::states::ThreadedHaps<CombinedHapSpace>,
    n_markers: usize,
) -> Option<MosaicPaths> {
    let n_states = threaded.n_states();
    let mut buffer = vec![CombinedHapId::from(0u32); n_states];
    let mut path1 = Vec::with_capacity(n_markers);
    let mut path2 = Vec::with_capacity(n_markers);

    for m in 0..n_markers {
        threaded.materialize_at(m, &mut buffer);

        let g1 = global.path1[m];
        let mut s1 = None;
        for (i, &gid) in buffer.iter().enumerate() {
            if gid == g1 {
                s1 = Some(i as u32);
                break;
            }
        }

        let g2 = global.path2[m];
        let mut s2 = None;
        for (i, &gid) in buffer.iter().enumerate() {
            if gid == g2 {
                s2 = Some(i as u32);
                break;
            }
        }

        if let (Some(idx1), Some(idx2)) = (s1, s2) {
            path1.push(idx1);
            path2.push(idx2);
        } else {
            return None;
        }
    }

    Some(MosaicPaths { path1, path2 })
}

struct MosaicChain<'a, RefSpace = crate::data::AnyMarkerSpace> {
    rng: rand::rngs::SmallRng,
    n_markers: usize,
    n_states: usize,
    p_recomb: &'a [f32],
    seq1: &'a [u8],
    seq2: &'a [u8],
    conf: &'a [f32],
    ref_provider: RefAlleleProvider<'a, AnyMarkerSpace, RefSpace>,
    combined_checkpoints: &'a FwdCheckpoints,
    fwd: aligned_vec::AVec<f32, aligned_vec::ConstAlign<32>>,
    fwd_prior: aligned_vec::AVec<f32, aligned_vec::ConstAlign<32>>,
    ref_alleles: Vec<u8>,
    weights: Vec<f32>,
    allele_probs: Vec<f32>,
    hap1_checkpoints: FwdCheckpoints,
    hap1_allele: Vec<u8>,
    hap1_partner_allele: Vec<u8>,
    hap1_use_combined: Vec<bool>,
    hap1_hard_match: Vec<bool>,
    hap2_checkpoints: FwdCheckpoints,
    hap2_allele: Vec<u8>,
    hap2_partner_allele: Vec<u8>,
    hap2_use_combined: Vec<bool>,
    hap2_hard_match: Vec<bool>,
    path1: Vec<u32>, // u32 saves 50% memory vs usize
    path2: Vec<u32>,
    fwd_block: Vec<f32>,
    trace: MosaicTrace,
    p_no_err: f32,
    p_err: f32,
    first_iteration: bool,
    pl_provider: Option<PlProvider<'a>>,
    anchor_hap1: Vec<u8>,
    anchor_hap2: Vec<u8>,
}

impl<'a, RefSpace> MosaicChain<'a, RefSpace> {
    fn new_with_buffers(
        seed: u64,
        n_markers: usize,
        n_states: usize,
        p_recomb: &'a [f32],
        seq1: &'a [u8],
        seq2: &'a [u8],
        conf: &'a [f32],
        ref_provider: RefAlleleProvider<'a, AnyMarkerSpace, RefSpace>,
        combined_checkpoints: &'a FwdCheckpoints,
        buffers: MosaicBuffers,
        p_no_err: f32,
        p_err: f32,
        pl_provider: Option<PlProvider<'a>>,
        anchor_hap1: Vec<u8>,
        anchor_hap2: Vec<u8>,
    ) -> Self {
        let out = Self {
            rng: rand::rngs::SmallRng::seed_from_u64(seed),
            n_markers,
            n_states,
            p_recomb,
            seq1,
            seq2,
            conf,
            ref_provider,
            combined_checkpoints,
            fwd: buffers.fwd,
            fwd_prior: buffers.fwd_prior,
            ref_alleles: buffers.ref_alleles,
            weights: buffers.weights,
            allele_probs: buffers.allele_probs,
            hap1_checkpoints: buffers.hap1_checkpoints,
            hap1_allele: buffers.hap1_allele,
            hap1_partner_allele: buffers.hap1_partner_allele,
            hap1_use_combined: buffers.hap1_use_combined,
            hap1_hard_match: buffers.hap1_hard_match,
            hap2_checkpoints: buffers.hap2_checkpoints,
            hap2_allele: buffers.hap2_allele,
            hap2_partner_allele: buffers.hap2_partner_allele,
            hap2_use_combined: buffers.hap2_use_combined,
            hap2_hard_match: buffers.hap2_hard_match,
            path1: buffers.path1,
            path2: buffers.path2,
            fwd_block: buffers.fwd_block,
            trace: MosaicTrace {
                mean_state: 0.0,
                switch_rate: 0.0,
                log_likelihood: 0.0,
            },
            p_no_err,
            p_err,
            first_iteration: true,
            pl_provider,
            anchor_hap1,
            anchor_hap2,
        };
        out
    }

    fn into_buffers(self) -> MosaicBuffers {
        MosaicBuffers {
            fwd: self.fwd,
            fwd_prior: self.fwd_prior,
            ref_alleles: self.ref_alleles,
            weights: self.weights,
            allele_probs: self.allele_probs,
            hap1_checkpoints: self.hap1_checkpoints,
            hap2_checkpoints: self.hap2_checkpoints,
            hap1_allele: self.hap1_allele,
            hap1_partner_allele: self.hap1_partner_allele,
            hap1_use_combined: self.hap1_use_combined,
            hap1_hard_match: self.hap1_hard_match,
            hap2_allele: self.hap2_allele,
            hap2_partner_allele: self.hap2_partner_allele,
            hap2_use_combined: self.hap2_use_combined,
            hap2_hard_match: self.hap2_hard_match,
            path1: self.path1,
            path2: self.path2,
            fwd_block: self.fwd_block,
        }
    }

    fn update_trace(&mut self) {
        if self.n_markers == 0 {
            self.trace.mean_state = 0.0;
            self.trace.switch_rate = 0.0;
            self.trace.log_likelihood = 0.0;
            return;
        }

        let mut sum = 0.0f64;
        let mut switches = 0usize;
        let mut logp = 0.0f64;
        for i in 0..self.n_markers {
            let s1 = self.path1[i] as f64;
            let s2 = self.path2[i] as f64;
            sum += s1 + s2;
            if i > 0 {
                if self.path1[i] != self.path1[i - 1] {
                    switches += 1;
                }
                if self.path2[i] != self.path2[i - 1] {
                    switches += 1;
                }
            }
            logp += (self.path1[i] as f64 + 1.0).ln();
        }

        let denom = (self.n_markers * 2) as f64;
        self.trace.mean_state = sum / denom;
        self.trace.switch_rate = if self.n_markers > 1 {
            switches as f64 / ((self.n_markers - 1) as f64 * 2.0)
        } else {
            0.0
        };
        self.trace.log_likelihood = logp;
    }

    fn build_hap2_inputs(&mut self) {
        for m in 0..self.n_markers {
            let a1 = self.seq1[m];
            let a2 = self.seq2[m];
            self.ref_provider
                .fill_ref_alleles(m, &mut self.ref_alleles);
            let ref_al = self.ref_alleles[self.path1[m] as usize];
            // Partner allele is always the other haplotype's current reference allele.
            self.hap2_partner_allele[m] = ref_al;
            self.hap2_hard_match[m] = false;
            if a1 == 255 && a2 == 255 {
                self.hap2_use_combined[m] = true;
                self.hap2_allele[m] = 255;
                continue;
            }
            if a1 == a2 {
                self.hap2_use_combined[m] = false;
                self.hap2_allele[m] = a1;
                continue;
            }

            if ref_al == a1 {
                self.hap2_use_combined[m] = false;
                self.hap2_allele[m] = a2;
            } else if ref_al == a2 {
                self.hap2_use_combined[m] = false;
                self.hap2_allele[m] = a1;
            } else {
                // Partner allele incompatible with genotype: impossible under the model.
                self.hap2_use_combined[m] = false;
                self.hap2_allele[m] = INVALID_ALLELE;
                self.hap2_hard_match[m] = true;
            }
        }
        if !self.anchor_hap2.is_empty() {
            for m in 0..self.n_markers {
                let a2 = self.anchor_hap2[m];
                if a2 == 255 {
                    continue;
                }
                self.hap2_use_combined[m] = false;
                self.hap2_allele[m] = a2;
                self.hap2_hard_match[m] = true;
                let a1 = self.anchor_hap1.get(m).copied().unwrap_or(255);
                if a1 != 255 {
                    self.hap2_partner_allele[m] = a1;
                }
            }
        }
    }

    /// Build hap1 inputs based on current path2 (for proper Gibbs sampling).
    /// This determines what allele H1 must carry given H2's sampled path.
    fn build_hap1_inputs(&mut self) {
        for m in 0..self.n_markers {
            let a1 = self.seq1[m];
            let a2 = self.seq2[m];
            self.ref_provider
                .fill_ref_alleles(m, &mut self.ref_alleles);
            let ref_al = self.ref_alleles[self.path2[m] as usize];
            // Partner allele is always the other haplotype's current reference allele.
            self.hap1_partner_allele[m] = ref_al;
            self.hap1_hard_match[m] = false;
            if a1 == 255 && a2 == 255 {
                self.hap1_use_combined[m] = true;
                self.hap1_allele[m] = 255;
                continue;
            }
            if a1 == a2 {
                self.hap1_use_combined[m] = false;
                self.hap1_allele[m] = a1;
                continue;
            }

            // Given path2's reference allele, determine what H1 must be
            if ref_al == a1 {
                // H2 carries a1, so H1 must carry a2
                self.hap1_use_combined[m] = false;
                self.hap1_allele[m] = a2;
            } else if ref_al == a2 {
                // H2 carries a2, so H1 must carry a1
                self.hap1_use_combined[m] = false;
                self.hap1_allele[m] = a1;
            } else {
                // Partner allele incompatible with genotype: impossible under the model.
                self.hap1_use_combined[m] = false;
                self.hap1_allele[m] = INVALID_ALLELE;
                self.hap1_hard_match[m] = true;
            }
        }
        if !self.anchor_hap1.is_empty() {
            for m in 0..self.n_markers {
                let a1 = self.anchor_hap1[m];
                if a1 == 255 {
                    continue;
                }
                self.hap1_use_combined[m] = false;
                self.hap1_allele[m] = a1;
                self.hap1_hard_match[m] = true;
                let a2 = self.anchor_hap2.get(m).copied().unwrap_or(255);
                if a2 != 255 {
                    self.hap1_partner_allele[m] = a2;
                }
            }
        }
    }
}

impl<RefSpace> MarkovChain<MosaicTrace> for MosaicChain<'_, RefSpace> {
    fn step(&mut self) -> &MosaicTrace {
        // Proper Gibbs sampling: H1 and H2 must each condition on the other.
        //
        // First iteration: use combined_checkpoints (marginal) to initialize path1.
        // Subsequent iterations: rebuild hap1_checkpoints based on current path2,
        // then sample path1 conditioned on H2's state.
        //
        // This creates the feedback loop required for convergence to P(H1,H2|G).

        if self.first_iteration {
            // Initialize: sample path1 from combined (marginal) distribution
            let dummy_target = vec![255u8; self.n_markers];
            let dummy_partner = vec![255u8; self.n_markers];
            let dummy_combined = vec![true; self.n_markers];
            let dummy_hard = vec![false; self.n_markers];

            sample_path_from_checkpoints(
                &mut self.path1,
                &self.combined_checkpoints,
                self.n_markers,
                self.n_states,
                self.p_recomb,
                self.seq1,
                self.seq2,
                self.conf,
                HapEmissionInputs {
                    target_constraint: &dummy_target,
                    partner_allele: &dummy_partner,
                    use_combined: &dummy_combined,
                    hard_match: &dummy_hard,
                },
                &mut self.ref_provider,
                self.pl_provider.as_ref(),
                self.p_no_err,
                self.p_err,
                &mut self.rng,
                &mut self.fwd_block,
                &mut self.weights,
                &mut self.ref_alleles,
                &mut self.allele_probs,
                EmissionMode::Combined,
            );
            self.first_iteration = false;
        } else {
            // Gibbs step: sample H1 | H2
            // Build hap1 constraints based on current path2
            self.build_hap1_inputs();
            let fwd = &mut self.fwd[..self.n_states];
            let fwd_prior = &mut self.fwd_prior[..self.n_states];
            let ref_alleles = &mut self.ref_alleles[..self.n_states];
            build_fwd_checkpoints(
                &mut self.hap1_checkpoints,
                self.n_markers,
                self.n_states,
                self.p_recomb,
                self.seq1,
                self.seq2,
                self.conf,
                HapEmissionInputs {
                    target_constraint: &self.hap1_allele,
                    partner_allele: &self.hap1_partner_allele,
                    use_combined: &self.hap1_use_combined,
                    hard_match: &self.hap1_hard_match,
                },
                &mut self.ref_provider,
                self.pl_provider.as_ref(),
                &mut self.allele_probs,
                fwd,
                fwd_prior,
                ref_alleles,
                self.p_no_err,
                self.p_err,
                EmissionMode::Hap,
            );
            sample_path_from_checkpoints(
                &mut self.path1,
                &self.hap1_checkpoints,
                self.n_markers,
                self.n_states,
                self.p_recomb,
                self.seq1,
                self.seq2,
                self.conf,
                HapEmissionInputs {
                    target_constraint: &self.hap1_allele,
                    partner_allele: &self.hap1_partner_allele,
                    use_combined: &self.hap1_use_combined,
                    hard_match: &self.hap1_hard_match,
                },
                &mut self.ref_provider,
                self.pl_provider.as_ref(),
                self.p_no_err,
                self.p_err,
                &mut self.rng,
                &mut self.fwd_block,
                &mut self.weights,
                &mut self.ref_alleles,
                &mut self.allele_probs,
                EmissionMode::Hap,
            );
        }

        // Gibbs step: sample H2 | H1
        self.build_hap2_inputs();
        let fwd = &mut self.fwd[..self.n_states];
        let fwd_prior = &mut self.fwd_prior[..self.n_states];
        let ref_alleles = &mut self.ref_alleles[..self.n_states];
        build_fwd_checkpoints(
            &mut self.hap2_checkpoints,
            self.n_markers,
            self.n_states,
            self.p_recomb,
            self.seq1,
            self.seq2,
            self.conf,
            HapEmissionInputs {
                target_constraint: &self.hap2_allele,
                partner_allele: &self.hap2_partner_allele,
                use_combined: &self.hap2_use_combined,
                hard_match: &self.hap2_hard_match,
            },
            &mut self.ref_provider,
            self.pl_provider.as_ref(),
            &mut self.allele_probs,
            fwd,
            fwd_prior,
            ref_alleles,
            self.p_no_err,
            self.p_err,
            EmissionMode::Hap,
        );
        sample_path_from_checkpoints(
            &mut self.path2,
            &self.hap2_checkpoints,
            self.n_markers,
            self.n_states,
            self.p_recomb,
            self.seq1,
            self.seq2,
            self.conf,
            HapEmissionInputs {
                target_constraint: &self.hap2_allele,
                partner_allele: &self.hap2_partner_allele,
                use_combined: &self.hap2_use_combined,
                hard_match: &self.hap2_hard_match,
            },
            &mut self.ref_provider,
            self.pl_provider.as_ref(),
            self.p_no_err,
            self.p_err,
            &mut self.rng,
            &mut self.fwd_block,
            &mut self.weights,
            &mut self.ref_alleles,
            &mut self.allele_probs,
            EmissionMode::Hap,
        );
        self.update_trace();
        &self.trace
    }

    fn current_state(&self) -> &MosaicTrace {
        &self.trace
    }
}

impl<RefSpace: Send + Sync> PhasingPipeline<RefSpace> {
    /// Create a new phasing pipeline
    pub fn new(config: Config, telemetry: Option<Arc<TelemetryBlackboard>>) -> Self {
        let params = ModelParams::new();
        Self {
            config,
            params,
            reference_gt: None,
            alignment: None,
            telemetry,
        }
    }

    /// Access current model parameters (after EM updates).
    pub fn params(&self) -> &ModelParams {
        &self.params
    }

    /// Set reference panel for reference-guided phasing
    ///
    /// When a reference panel is provided, the phasing algorithm uses it to:
    /// 1. Improve state selection (PBWT neighbors from reference)
    /// 2. Guide phase decisions with reference haplotypes
    ///
    /// Uses Arc for shared ownership to avoid cloning the large reference panel.
    pub fn set_reference(
        &mut self,
        reference: Arc<GenotypeMatrix<Phased, RefSpace>>,
        alignment: MarkerAlignment<crate::data::AnyMarkerSpace, RefSpace>,
    ) {
        self.reference_gt = Some(reference);
        self.alignment = Some(alignment);
    }
}

impl PhasingPipeline<crate::data::AnyMarkerSpace> {

    /// Run the phasing pipeline
    pub fn run(&mut self) -> Result<()> {
        eprintln!("Loading VCF...");

        // Load exclusion lists
        let exclude_samples = self.config.load_exclude_samples()?;
        let exclude_markers = self.config.load_exclude_markers()?;

        if !exclude_samples.is_empty() {
            eprintln!("Excluding {} samples", exclude_samples.len());
        }
        if !exclude_markers.is_empty() {
            eprintln!("Excluding {} markers", exclude_markers.len());
        }

        // Load target VCF with filtering
        let (mut reader, file_reader) = VcfReader::open(&self.config.gt)?;
        reader.set_exclude_samples(&exclude_samples);
        reader.set_exclude_markers(exclude_markers);
        let target_gt = reader.read_all(file_reader)?;

        if target_gt.n_markers() == 0 {
            eprintln!("No markers found in input VCF");
            return Ok(());
        }

        let n_markers = target_gt.n_markers();
        let n_samples = target_gt.n_samples();
        let n_haps = target_gt.n_haplotypes();

        if let Some(bb) = &self.telemetry {
            bb.set_total_samples(n_samples as u64);
            bb.set_samples_processed(0);
            bb.set_total_markers(n_markers as u64);
            bb.set_markers_processed(0);
        }

        eprintln!(
            "Loaded {} markers, {} samples ({} haplotypes), {:.2} MB",
            n_markers,
            n_samples,
            n_haps,
            target_gt.size_bytes() as f64 / 1024.0 / 1024.0
        );

        // Load reference panel if provided (for reference-guided phasing)
        if let Some(ref_path) = &self.config.r#ref {
            eprintln!("Loading reference panel for phasing...");
            let ref_gt: GenotypeMatrix<Phased> =
                if ref_path.extension().map(|e| e == "bref3").unwrap_or(false) {
                    eprintln!("  Detected BREF3 format");
                    let reader = Bref3Reader::open(ref_path)?;
                    reader.read_all()?
                } else {
                    eprintln!("  Detected VCF format");
                    let (mut ref_reader, ref_file) = VcfReader::open(ref_path)?;
                    ref_reader.read_all(ref_file)?.into_phased()
                };
            eprintln!(
                "  Reference: {} markers, {} haplotypes",
                ref_gt.n_markers(),
                ref_gt.n_haplotypes()
            );

            // Create marker alignment between target and reference
            let alignment = MarkerAlignment::new(&target_gt, &ref_gt);
            eprintln!(
                "  Aligned {} reference markers to target",
                alignment.n_aligned()
            );

            // Store in pipeline struct for use during phasing iterations
            self.set_reference(Arc::new(ref_gt), alignment);
        }

        // Compute combined haplotype count
        let n_ref_haps = self
            .reference_gt
            .as_ref()
            .map(|r| r.n_haplotypes())
            .unwrap_or(0);
        let n_total_haps = n_haps + n_ref_haps;

        if n_ref_haps > 0 {
            eprintln!(
                "Combined haplotype space: {} target + {} reference = {} total",
                n_haps, n_ref_haps, n_total_haps
            );
        }

        // Initialize parameters based on TOTAL haplotype count (target + ref)
        self.params = ModelParams::for_phasing(n_total_haps, self.config.ne, self.config.err);
        self.params
            .set_n_states(self.config.phase_states.min(n_total_haps.saturating_sub(2)));

        // Load genetic map if provided
        let gen_maps = if let Some(ref map_path) = self.config.map {
            let chrom_names: Vec<&str> = target_gt
                .markers()
                .chrom_names()
                .iter()
                .map(|s| s.as_ref())
                .collect();
            GeneticMaps::from_plink_file(map_path, &chrom_names)?
        } else {
            GeneticMaps::new()
        };

        // Create mutable genotype storage for phasing
        // MutableGenotypes now internally tracks missing data (allele = 255)
        // so we can use from_fn to initialize all values including missing
        let mut geno = MutableGenotypes::from_fn(n_markers, n_haps, |m, h| {
            target_gt.allele(MarkerIdx::new(m as u32), HapIdx::new(h as u32))
        });

        // Compute genetic distances and recombination probabilities using MarkerMap
        // This handles map interpolation and minimum distance enforcement
        let chrom = target_gt.marker(MarkerIdx::new(0)).chrom;
        let marker_map = if let Some(map) = gen_maps.get(chrom) {
            MarkerMap::create(target_gt.markers(), map)
        } else {
            MarkerMap::from_positions(target_gt.markers())
        };

        let gen_positions: Vec<f64> = marker_map.gen_positions().to_vec();

        // Compute MAF for each marker (used by IBS2 and two-stage phasing)
        // Includes reference panel allele counts if available, ensuring homozygous
        // markers in small target panels are correctly classified as high-frequency.
        let maf: Vec<f32> =
            if let (Some(ref_gt), Some(alignment)) = (&self.reference_gt, &self.alignment) {
                (0..n_markers)
                    .map(|m| {
                        let m_idx = MarkerIdx::new(m as u32);
                        let mut target_alt = target_gt.column(m_idx).alt_count() as usize;
                        let target_total = target_gt.n_haplotypes();
                        if let Some(ref_m) = alignment.target_to_ref(m_idx) {
                            let ref_col = ref_gt.column(ref_m);
                            let ref_total = ref_gt.n_haplotypes();
                            let ref_alt = ref_col.alt_count() as usize;

                            if let Some(Some(mapping)) = alignment.allele_mappings.get(m) {
                                let ref_equiv_of_alt =
                                    mapping.targ_to_ref.get(1).copied().unwrap_or(-1);
                                if ref_equiv_of_alt == 1 {
                                    target_alt += ref_alt;
                                } else if ref_equiv_of_alt == 0 {
                                    target_alt += ref_total - ref_alt;
                                }
                            }

                            let total = target_total + ref_total;
                            if total == 0 {
                                0.0
                            } else {
                                let freq = target_alt as f32 / total as f32;
                                freq.min(1.0 - freq)
                            }
                        } else {
                            let freq = target_alt as f32 / target_total as f32;
                            freq.min(1.0 - freq)
                        }
                    })
                    .collect()
            } else {
                (0..n_markers)
                    .map(|m| target_gt.column(MarkerIdx::new(m as u32)).maf() as f32)
                    .collect()
            };

        // TWO-STAGE PHASING: Classify markers by frequency
        // Stage 1 (high-frequency): Run full HMM - these markers provide phasing signal
        // Stage 2 (rare): Interpolate from flanking high-frequency markers
        let rare_threshold = self.config.rare;
        let hi_freq_markers: Vec<usize> = (0..n_markers)
            .filter(|&m| maf[m] >= rare_threshold)
            .collect();
        let rare_markers: Vec<usize> = (0..n_markers)
            .filter(|&m| maf[m] < rare_threshold && maf[m] > 0.0) // Exclude monomorphic
            .collect();

        let n_hi_freq = hi_freq_markers.len();
        eprintln!(
            "Two-stage phasing: {} high-frequency markers (MAF >= {}), {} rare markers",
            n_hi_freq,
            rare_threshold,
            rare_markers.len()
        );

        // Create mapping from hi-freq index to original index
        let hi_freq_to_orig: Vec<usize> = hi_freq_markers.clone();
        let hi_freq_gen_positions: Vec<f64> =
            hi_freq_to_orig.iter().map(|&m| gen_positions[m]).collect();

        let stage1_blocks =
            partition_markers_by_cm(&hi_freq_gen_positions, stage1_block_cm(&hi_freq_gen_positions));
        eprintln!("Stage 1 blocks: {}", stage1_blocks.len());

        // Compute genetic distances only for HIGH-FREQUENCY markers
        // This is critical: recombination probabilities must be computed for the
        // inter-marker distances between consecutive hi-freq markers, not all markers
        let stage1_gen_dists: Vec<f64> = if hi_freq_markers.len() > 1 {
            hi_freq_markers
                .windows(2)
                .map(|w| gen_positions[w[1]] - gen_positions[w[0]])
                .collect()
        } else {
            Vec::new()
        };

        // Log ploidy information from detected samples
        let samples = target_gt.samples_arc();
        let n_haploid = (0..n_samples)
            .filter(|&s| !samples.is_diploid(SampleIdx::new(s as u32)))
            .count();
        if n_haploid > 0 {
            return Err(crate::error::ReagleError::vcf(format!(
                "Detected {} haploid samples. Reagle currently supports diploid samples only.",
                n_haploid
            )));
        }

        // Stage 1 (condensed beam) - single pass
        if let Some(bb) = &self.telemetry {
            bb.set_stage(Stage::PhasingMain);
            bb.set_producer_stage(Stage::PhasingMain);
            bb.set_total_samples(n_samples as u64);
            bb.set_samples_processed(0);
            bb.set_total_markers(hi_freq_markers.len() as u64);
            bb.set_markers_processed(0);
            bb.set_total_iterations(1);
            bb.set_current_iteration(1);
        }

        let confidence_by_sample = build_sample_confidence(&target_gt);
        let phase_mask = target_gt.phase_mask();
        let mut sample_phases = self.create_sample_phases(&geno, &confidence_by_sample, phase_mask);
        let mut stage1_p_recomb: Vec<f32> = std::iter::once(0.0f32)
            .chain(stage1_gen_dists.iter().map(|&d| self.params.p_recomb(d)))
            .collect();

        if self.reference_gt.is_none() {
            // Build IBS2 segments for phase consistency (target-only)
            if let Some(bb) = &self.telemetry {
                bb.set_stage(Stage::PhasingPrescan);
                bb.set_producer_stage(Stage::PhasingPrescan);
                bb.set_op("Phasing prescan: IBS2 segments");
            }
            eprintln!("Building IBS2 segments...");
            let ibs2 = Ibs2::new(&target_gt, &gen_maps, chrom, &maf);
            let n_with_ibs2 = (0..n_samples)
                .filter(|&s| ibs2.n_segments(crate::data::haplotype::SampleIdx::new(s as u32)) > 0)
                .count();
            eprintln!(
                "Found {} samples with IBS2 segments, {} total",
                n_with_ibs2,
                ibs2.n_samples()
            );

            // Fallback: iterative HMM phasing without reference
            let n_burnin = self.config.burnin;
            let n_iterations = self.config.iterations;
            let total_iterations = n_burnin + n_iterations;
            if let Some(bb) = &self.telemetry {
                bb.set_total_samples(n_samples as u64);
                bb.set_samples_processed(0);
                bb.set_total_markers(hi_freq_markers.len() as u64);
                bb.set_markers_processed(0);
                bb.set_total_iterations(total_iterations as u64);
                bb.set_current_iteration(0);
            }

            let ref_gt = self.reference_gt.as_ref().map(|v| v.as_ref());
            let threaded_haps_vec = self.build_phasing_prescan_states(
                &target_gt,
                &geno,
                ref_gt,
                self.alignment.as_ref(),
                n_hi_freq,
                n_samples,
                &hi_freq_gen_positions,
                self.config.imp_step,
                Some(&hi_freq_to_orig),
            )?;
            let mut mcmc_paths: Vec<Option<GlobalMosaicPaths>> = vec![None; n_samples];
            let mut stable_main_iters = 0usize;

            for it in 0..total_iterations {
                let is_burnin = it < n_burnin;
                let iter_type = if is_burnin { "burnin" } else { "main" };
                eprintln!("Iteration {}/{} ({})", it + 1, total_iterations, iter_type);
                if let Some(bb) = &self.telemetry {
                    let stage = if is_burnin {
                        Stage::PhasingBurnin
                    } else {
                        Stage::PhasingMain
                    };
                    bb.set_stage(stage);
                    bb.set_producer_stage(stage);
                    bb.set_current_iteration((it + 1) as u64);
                    bb.set_samples_processed(0);
                    bb.set_markers_processed(0);
                }

                self.params.lr_threshold = self.params.lr_threshold_for_iteration(it);
                let atomic_estimates = if is_burnin && self.config.em {
                    Some(crate::model::parameters::AtomicParamEstimates::new())
                } else {
                    None
                };

                let (total_switches, total_phased) = self.run_phase_baum_iteration_stage1(
                    &target_gt,
                    &mut geno,
                    &threaded_haps_vec,
                    &stage1_p_recomb,
                    &stage1_gen_dists,
                    &hi_freq_to_orig,
                    &stage1_blocks,
                    &ibs2,
                    &mut sample_phases,
                    &mut mcmc_paths,
                    atomic_estimates.as_ref(),
                    it,
                )?;
                if let Some(bb) = &self.telemetry {
                    bb.set_samples_processed(n_samples as u64);
                    bb.set_markers_processed(hi_freq_markers.len() as u64);
                }

                if let Some(ref atomic) = atomic_estimates {
                    let est = atomic.to_estimates();
                    let mut params_updated = false;
                    if est.n_emit_obs() > 0 {
                        self.params.update_p_mismatch(est.p_mismatch());
                        params_updated = true;
                    }
                    if est.n_switch_obs() > 0 {
                        self.params.update_recomb_intensity(est.recomb_intensity());
                        params_updated = true;
                    }
                    if params_updated {
                        stage1_p_recomb = std::iter::once(0.0f32)
                            .chain(stage1_gen_dists.iter().map(|&d| self.params.p_recomb(d)))
                            .collect();
                    }
                }

                if !is_burnin {
                    let remaining_hets = Self::count_unphased_hets(&sample_phases, &hi_freq_to_orig);
                    let change = total_switches + total_phased;
                    let threshold = (remaining_hets / 100).max(1);
                    if change <= threshold {
                        stable_main_iters += 1;
                        if stable_main_iters >= 2 {
                            eprintln!(
                                "Phasing converged (changes {} <= threshold {} for 2 main iterations); stopping early.",
                                change, threshold
                            );
                            break;
                        }
                    } else {
                        stable_main_iters = 0;
                    }
                }
            }
        } else {
            let ref_gt = match self.reference_gt.as_ref() {
                Some(r) => r.clone(),
                None => unreachable!(),
            };
            let alignment = self.alignment.as_ref().ok_or_else(|| {
                crate::error::ReagleError::config("Reference alignment missing for beam phasing")
            })?;

            let packed_ref = PackedRefView::build(&target_gt, &ref_gt, alignment);

            // Compute allele frequencies for TMRCA-aware beam scoring.
            // For each hi-freq marker, compute (freq_allele0, freq_allele1) from reference.
            let hi_freq_allele_freqs: Option<Vec<(f32, f32)>> = {
                let n_ref_haps = packed_ref.n_ref_haps();
                if n_ref_haps > 0 {
                    Some(hi_freq_to_orig.iter().map(|&orig_m| {
                        let mut count0 = 0u32;
                        let mut count1 = 0u32;
                        for h in 0..n_ref_haps {
                            match packed_ref.ref_allele_targ(orig_m, h) {
                                Some(0) => count0 += 1,
                                Some(1) => count1 += 1,
                                _ => {}
                            }
                        }
                        let total = (count0 + count1).max(1) as f32;
                        (count0 as f32 / total, count1 as f32 / total)
                    }).collect())
                } else {
                    None
                }
            };

            let beam_config = BeamConfig::default();
        let beam_index = PbwtBeamIndex::build(
            &ref_gt,
            alignment,
            &hi_freq_to_orig,
            beam_config.inject_k,
            beam_config.inject_interval,
        );
            let phaser = BeamPhaser::new(&packed_ref, &self.params, beam_config);

            let ibs2 = Ibs2::new(&target_gt, &gen_maps, chrom, &maf);

            let threaded_haps_vec = self.build_phasing_prescan_states(
                &target_gt,
                &geno,
                Some(ref_gt.as_ref()),
                self.alignment.as_ref(),
                n_hi_freq,
                n_samples,
                &hi_freq_gen_positions,
                self.config.imp_step,
                Some(&hi_freq_to_orig),
            )?;

            let original_unphased: Vec<Vec<usize>> = sample_phases
                .iter()
                .map(|sp| {
                    hi_freq_to_orig
                        .iter()
                        .copied()
                        .filter(|&m| sp.is_unphased(m))
                        .collect()
                })
                .collect();

            let beam_confidence: Vec<std::sync::Mutex<Vec<(usize, u8, u8, f32)>>> =
                (0..n_samples).map(|_| std::sync::Mutex::new(Vec::new())).collect();

            let n_target_haps = target_gt.n_haplotypes();
            sample_phases
                .par_iter_mut()
                .enumerate()
                .for_each(|(s, sp)| {
                    let mut active_pool = ActivePool::new(packed_ref.n_ref_haps());
                    let mut tmp = vec![crate::model::types::CombinedHapId::from(0u32); threaded_haps_vec[s].n_states()];
                    let th = &threaded_haps_vec[s];
                    th.materialize_at(0, &mut tmp);
                for id in tmp.iter().copied() {
                    let hid = id.as_u32() as usize;
                    if hid >= n_target_haps {
                        let ref_id = hid - n_target_haps;
                        if ref_id < packed_ref.n_ref_haps() {
                            active_pool.add(ref_id);
                        }
                    }
                }

                // Re-rank: push best matching reference haps to the end so they are prioritized.
                let mut scored: Vec<(i32, usize)> = Vec::with_capacity(active_pool.list().len());
                for &h in active_pool.list() {
                    let mut score = 0i32;
                    for &m in hi_freq_to_orig.iter() {
                        let a1 = sp.allele1(m);
                        let a2 = sp.allele2(m);
                        if a1 == 255 || a2 == 255 {
                            continue;
                        }
                        if let Some(r) = packed_ref.ref_allele_targ(m, h) {
                            if a1 == a2 {
                                score += if r == a1 { 2 } else { -2 };
                            } else {
                                score += if r == a1 || r == a2 { 1 } else { -1 };
                            }
                        }
                    }
                    scored.push((score, h));
                }
                scored.sort_by(|a, b| b.0.cmp(&a.0));
                for &(_, h) in scored.iter().take(4) {
                    active_pool.promote(h);
                }

                    let condensed = CondensedTarget::build(
                        sp,
                        &hi_freq_to_orig,
                        &hi_freq_gen_positions,
                        hi_freq_allele_freqs.as_deref(),
                        &packed_ref,
                        &self.params,
                    );

                    let mut injector = PbwtInjector::new(&beam_index, packed_ref.n_ref_haps(), beam_config.inject_k);
                    let fwd = phaser.phase_sample(&condensed, sp, &mut active_pool, &mut injector);

                    // Use forward pass posteriors only (avoid heuristic bidirectional combine).
                    if let Ok(mut slot) = beam_confidence[s].lock() {
                        slot.clear();
                        for (i, &p) in fwd.p_swapped.iter().enumerate() {
                            let call = &condensed.call_sites[i];
                            slot.push((call.marker.as_usize(), call.a1, call.a2, p));
                        }
                    }
                });

            for (s, sp) in sample_phases.iter_mut().enumerate() {
                for &m in original_unphased[s].iter() {
                    sp.mark_unphased(m);
                }
            }

            // Micro-HMM refinement on hi-frequency markers (single pass).
            let mut mcmc_paths: Vec<Option<GlobalMosaicPaths>> = vec![None; n_samples];
            let _ = self.run_phase_baum_iteration_stage1(
                &target_gt,
                &mut geno,
                &threaded_haps_vec,
                &stage1_p_recomb,
                &stage1_gen_dists,
                &hi_freq_to_orig,
                &stage1_blocks,
                &ibs2,
                &mut sample_phases,
                &mut mcmc_paths,
                None,
                0,
            )?;

            for (s, sp) in sample_phases.iter_mut().enumerate() {
                if let Ok(slot) = beam_confidence[s].lock() {
                    for &(m, a1, a2, p) in slot.iter() {
                        let swapped = sp.allele1(m) == a2 && sp.allele2(m) == a1;
                        let conf = if swapped { p } else { 1.0 - p };
                        sp.set_phase_confidence(m, conf);
                    }
                }
            }
        }

        // Sync final phase state from SamplePhase to MutableGenotypes
        self.sync_sample_phases_to_geno(&sample_phases, &mut geno);

        // STAGE 2: Phase rare markers using HMM state probability interpolation
        // This implements the proper algorithm from Java Beagle's Stage2Baum.java
        if !rare_markers.is_empty() && hi_freq_markers.len() >= 2 {
            eprintln!(
                "Stage 2: Phasing {} rare markers using HMM interpolation...",
                rare_markers.len()
            );
            if let Some(bb) = &self.telemetry {
                bb.set_stage(Stage::PhasingStage2);
                bb.set_producer_stage(Stage::PhasingStage2);
                bb.set_op("Phasing stage2: HMM interpolation");
                bb.set_total_iterations(0);
                bb.set_current_iteration(0);
                bb.set_total_markers(rare_markers.len() as u64);
                bb.set_markers_processed(0);
                bb.set_samples_processed(0);
            }
            let stage2_handoff = self.phase_rare_markers_with_hmm(
                &target_gt,
                &mut geno,
                &hi_freq_markers,
                &gen_positions,
                &hi_freq_gen_positions,
                &stage1_p_recomb,
                &mut sample_phases,
                &maf,
                rare_threshold,
                None,
                None,
            );
            tracing::trace!(
                has_handoff = stage2_handoff.is_some(),
                "Stage 2 HMM overlap handoff computed"
            );
            if let Some(bb) = &self.telemetry {
                bb.set_markers_processed(rare_markers.len() as u64);
                bb.set_samples_processed(n_samples as u64);
            }

            // Sync again after Stage 2
            self.sync_sample_phases_to_geno(&sample_phases, &mut geno);
        }

        // Build final GenotypeMatrix from mutable genotypes
        let final_gt = self.build_final_matrix(&target_gt, &geno, &sample_phases);

        // Write output
        let output_path = self.config.out.with_extension("vcf.gz");
        eprintln!("Writing output to {:?}", output_path);

        let mut writer = VcfWriter::create(&output_path, samples)?;
        writer.write_header(final_gt.markers())?;
        writer.write_phased(&final_gt, 0, final_gt.n_markers())?;
        writer.flush()?;

        eprintln!("Phasing complete!");
        Ok(())
    }

    /// Run the phasing pipeline in streaming mode for large datasets
    pub fn run_streaming(&mut self) -> Result<()> {
        eprintln!("Opening VCF for streaming...");

        // Configure streaming (genetic maps loaded lazily by StreamingVcfReader)
        let streaming_config = StreamingConfig {
            window_cm: self.config.window,
            overlap_cm: self.config.overlap,
            max_markers: self.config.window_markers,
            ..Default::default()
        };

        // Load genetic maps - use empty maps if no map file provided
        let gen_maps = if let Some(ref map_path) = self.config.map {
            GeneticMaps::from_plink_file(
                map_path,
                &[
                    "chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9",
                    "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17",
                    "chr18", "chr19", "chr20", "chr21", "chr22", "chrX", "1", "2", "3", "4", "5",
                    "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
                    "20", "21", "22", "X",
                ],
            )?
        } else {
            GeneticMaps::new()
        };

        // Load reference panel if provided (for reference-guided phasing)
        let mut ref_pos_map: Option<HashMap<(String, u32), Vec<usize>>> = None;
        if let Some(ref_path) = &self.config.r#ref {
            eprintln!("Loading reference panel for streaming phasing...");
            let ref_gt: GenotypeMatrix<Phased> =
                if ref_path.extension().map(|e| e == "bref3").unwrap_or(false) {
                    eprintln!("  Detected BREF3 format");
                    let reader = Bref3Reader::open(ref_path)?;
                    reader.read_all()?
                } else {
                    eprintln!("  Detected VCF format");
                    let (mut ref_reader, ref_file) = VcfReader::open(ref_path)?;
                    ref_reader.read_all(ref_file)?.into_phased()
                };
            eprintln!(
                "  Reference: {} markers, {} haplotypes",
                ref_gt.n_markers(),
                ref_gt.n_haplotypes()
            );
            ref_pos_map = Some(MarkerAlignment::<crate::data::AnyMarkerSpace, _>::build_ref_pos_index(ref_gt.markers()));
            self.reference_gt = Some(Arc::new(ref_gt));
            self.alignment = None;
        }

        // Open streaming reader
        let mut reader =
            StreamingVcfReader::open(&self.config.gt, gen_maps.clone(), streaming_config)?;
        let samples = reader.samples_arc();

        // Check for haploid samples
        let n_samples = samples.len();
        let n_haploid = (0..n_samples)
            .filter(|&s| !samples.is_diploid(SampleIdx::new(s as u32)))
            .count();
        if n_haploid > 0 {
            return Err(crate::error::ReagleError::vcf(format!(
                "Detected {} haploid samples. Reagle currently supports diploid samples only.",
                n_haploid
            )));
        }

        // Create output writer
        let output_path = self.config.out.with_extension("vcf.gz");
        eprintln!("Writing output to {:?}", output_path);
        let mut writer = VcfWriter::create(&output_path, samples)?;

        let mut window_count = 0;
        let mut total_markers = 0;
        let mut wrote_header = false;
        let mut align_stats = AlignmentStats::default();

        // Track phased overlap from previous window for phase continuity
        // PhasedOverlap contains state probabilities used for PBWT state handoff
        let mut phased_overlap: Option<PhasedOverlap> = None;

        // Double-buffered windows
        let mut current_window: Option<StreamWindowWithResult> = None;
        let mut next_window_opt = reader.next_window()?;

        // Process windows with double-buffering
        while let Some(mut window) = next_window_opt {
            window_count += 1;

            let n_markers = window.genotypes.n_markers();

            eprintln!(
                "Loading window {} ({} markers, global {}..{}, output {}..{})",
                window_count,
                n_markers,
                window.global_start,
                window.global_end,
                window.output_start,
                window.output_end
            );

            // Load next window
            next_window_opt = reader.next_window()?;

            // Set the phased overlap from previous window
            window.phased_overlap = phased_overlap.take();

            if let (Some(ref_gt), Some(ref_pos_map)) =
                (self.reference_gt.as_ref(), ref_pos_map.as_ref())
            {
                let (alignment, stats) = MarkerAlignment::new_with_ref_index(
                    &window.genotypes,
                    ref_gt.markers(),
                    ref_pos_map,
                );
                align_stats.aligned += stats.aligned;
                align_stats.strand_flipped += stats.strand_flipped;
                align_stats.allele_swapped += stats.allele_swapped;
                self.alignment = Some(alignment);
            } else {
                self.alignment = None;
            }

            // Phase this window with overlap constraint
            let (phased, next_overlap_handoff) = self
                .phase_in_memory_with_overlap(
                    &window.genotypes,
                    &gen_maps,
                    window.phased_overlap.as_ref(),
                    Some(window.output_end),
                )?;

            // Extract overlap for next window (contains identity-aware priors for handoff)
            if !window.is_last() {
                phased_overlap = Some(self.extract_overlap(
                    &phased,
                    window.output_end,
                    n_markers,
                    next_overlap_handoff,
                ));
            }

            // If we have a current window to finalize Stage 2
            if let Some(current) = current_window.take() {
                // Perform Stage 2 finalization using phased markers from next window
                let finalized = info_span!("finalize_stage2").in_scope(|| {
                    self.finalize_stage2_with_forward_context(
                        &current.phased_result.as_ref().unwrap(),
                        &phased,
                    )
                })?;

                // Write output region
                if current.window.is_first && !wrote_header {
                    writer.write_header(finalized.markers())?;
                    wrote_header = true;
                }
                writer.write_phased(
                    &finalized,
                    current.window.output_start,
                    current.window.output_end,
                )?;
                total_markers += current.window.output_end - current.window.output_start;
            }

            // Move to next window
            current_window = Some(StreamWindowWithResult {
                window,
                phased_result: Some(phased),
            });
        }

        // Finalize last window (no next window for Stage 2 context)
        if let Some(ref current) = current_window {
            info_span!("finalize_last_window").in_scope(|| -> Result<()> {
                let finalized = current.phased_result.as_ref().unwrap().clone(); // No additional context
                if current.window.is_first && !wrote_header {
                    writer.write_header(finalized.markers())?;
                }
                writer.write_phased(&finalized, current.output_start, current.output_end)?;
                total_markers += current.output_end - current.output_start;
                Ok(())
            })?;
        }

        writer.flush()?;
        eprintln!(
            "Streaming phasing complete: {} windows, {} markers",
            window_count, total_markers
        );
        if align_stats.aligned > 0 && (align_stats.strand_flipped > 0 || align_stats.allele_swapped > 0)
        {
            eprintln!(
                "  Alignment summary (streaming): {} strand-flipped, {} allele-swapped markers",
                align_stats.strand_flipped, align_stats.allele_swapped
            );
        }
        Ok(())
    }

    /// Extract phased overlap region from a phased genotype matrix
    ///
    /// This extracts the overlap region (markers from `start` to `end`) to be used
    /// as a constraint for the next window's phasing, ensuring phase continuity.
    fn extract_overlap(
        &self,
        phased: &GenotypeMatrix<crate::data::storage::phase_state::Phased>,
        start: usize,
        end: usize,
        handoff: Option<Stage2OverlapHandoff>,
    ) -> PhasedOverlap {
        let n_overlap = end - start;
        let n_haps = phased.n_haplotypes();

        let mut alleles = Vec::with_capacity(n_overlap * n_haps);

        // Layout: alleles[hap * n_markers + marker]
        for h in 0..n_haps {
            let h_idx = HapIdx::new(h as u32);
            for m in start..end {
                let m_idx = MarkerIdx::new(m as u32);
                alleles.push(phased.allele(m_idx, h_idx));
            }
        }

        let mut overlap = PhasedOverlap::new(n_overlap, n_haps, alleles);

        // Attach soft-information handoff payloads if available.
        if let Some(handoff) = handoff {
            if let Some(probs) = handoff.state_probs {
                let state_meta = (probs.n_states, probs.marker_indices.len(), probs.data.len());
                tracing::trace!(
                    n_states = state_meta.0,
                    marker_indices = state_meta.1,
                    hap_entries = state_meta.2,
                    "Attaching legacy state_probs handoff"
                );
                overlap.set_state_probs(probs);
            }
            if let Some(priors) = handoff.hap_priors {
                overlap.set_hap_priors(priors);
            }
            if let Some(marker) = handoff.prior_stage1_global_marker {
                overlap.set_prior_stage1_global_marker(marker);
            }
        }

        overlap
    }

    /// Automatically select between in-memory and streaming mode based on data size
    pub fn run_auto(&mut self) -> Result<()> {
        let file_size = std::fs::metadata(&self.config.gt)
            .map(|m| m.len())
            .unwrap_or(0);
        let estimated_markers = file_size / 100;
        let use_streaming = estimated_markers > self.config.window_markers as u64;

        if use_streaming {
            eprintln!(
                "Auto-detected large dataset (~{} markers), using streaming mode",
                estimated_markers
            );
            self.run_streaming()
        } else {
            self.run()
        }
    }
}

impl<RefSpace: Send + Sync> PhasingPipeline<RefSpace> {
    /// Phase a GenotypeMatrix in-memory with overlap constraint from previous window
    ///
    /// This is like `phase_in_memory` but seeds the phasing with alleles from the
    /// overlap region of the previous window, ensuring phase continuity at window
    /// boundaries. Based on Java's FixedPhaseData and SplicedGT.
    pub fn phase_in_memory_with_overlap(
        &mut self,
        target_gt: &GenotypeMatrix,
        gen_maps: &GeneticMaps,
        phased_overlap: Option<&PhasedOverlap>,
        next_overlap_start: Option<usize>,
    ) -> Result<(
        GenotypeMatrix<crate::data::storage::phase_state::Phased>,
        Option<Stage2OverlapHandoff>,
    )> {
        let n_markers = target_gt.n_markers();
        let n_haps = target_gt.n_haplotypes();
        let n_samples = n_haps / 2;
        let n_ref_haps = self
            .reference_gt
            .as_ref()
            .map(|r| r.n_haplotypes())
            .unwrap_or(0);
        let n_total_haps = n_haps + n_ref_haps;
        let samples = target_gt.samples_arc();

        // Check for haploid samples
        let n_haploid = (0..n_samples)
            .filter(|&s| !samples.is_diploid(SampleIdx::new(s as u32)))
            .count();
        if n_haploid > 0 {
            return Err(crate::error::ReagleError::vcf(format!(
                "Detected {} haploid samples. Reagle currently supports diploid samples only.",
                n_haploid
            )));
        }

        if n_markers == 0 {
            return Ok((target_gt.clone().into_phased(), None));
        }

        self.params = ModelParams::for_phasing(n_total_haps, self.config.ne, self.config.err);
        self.params
            .set_n_states(self.config.phase_states.min(n_total_haps.saturating_sub(2)));

        // Initialize genotypes preserving actual allele values including missing (255)
        let mut geno = MutableGenotypes::from_fn(n_markers, n_haps, |m, h| {
            target_gt.allele(MarkerIdx::new(m as u32), HapIdx::new(h as u32))
        });

        // Build missing mask for overlap constraint handling
        let missing_mask: Vec<BitBox<u8, Lsb0>> = (0..n_haps)
            .map(|h| {
                let bits: BitVec<u8, Lsb0> = (0..n_markers)
                    .map(|m| {
                        target_gt.allele(MarkerIdx::new(m as u32), HapIdx::new(h as u32)) == 255
                    })
                    .collect();
                bits.into_boxed_bitslice()
            })
            .collect();

        // Apply overlap constraint: set alleles from previous window's phased overlap
        // This seeds the phasing with the known phase from the overlap region
        let overlap_markers = if let Some(overlap) = phased_overlap {
            self.apply_overlap_constraint(&mut geno, overlap);
            overlap.n_markers.min(n_markers)
        } else {
            0
        };

        let chrom = target_gt.marker(MarkerIdx::new(0)).chrom;

        // Compute MAF for each marker (used by IBS2 and two-stage phasing)
        // Includes reference panel allele counts if available, ensuring homozygous
        // markers in small target panels are correctly classified as high-frequency.
        let maf: Vec<f32> =
            if let (Some(ref_gt), Some(alignment)) = (&self.reference_gt, &self.alignment) {
                (0..n_markers)
                    .map(|m| {
                        let m_idx = MarkerIdx::new(m as u32);
                        let mut target_alt = target_gt.column(m_idx).alt_count() as usize;
                        let target_total = target_gt.n_haplotypes();
                        if let Some(ref_m) = alignment.target_to_ref(m_idx) {
                            let ref_col = ref_gt.column(ref_m);
                            let ref_total = ref_gt.n_haplotypes();
                            let ref_alt = ref_col.alt_count() as usize;

                            if let Some(Some(mapping)) = alignment.allele_mappings.get(m) {
                                let ref_equiv_of_alt =
                                    mapping.targ_to_ref.get(1).copied().unwrap_or(-1);
                                if ref_equiv_of_alt == 1 {
                                    target_alt += ref_alt;
                                } else if ref_equiv_of_alt == 0 {
                                    target_alt += ref_total - ref_alt;
                                }
                            }

                            let total = target_total + ref_total;
                            if total == 0 {
                                0.0
                            } else {
                                let freq = target_alt as f32 / total as f32;
                                freq.min(1.0 - freq)
                            }
                        } else {
                            let freq = target_alt as f32 / target_total as f32;
                            freq.min(1.0 - freq)
                        }
                    })
                    .collect()
            } else {
                (0..n_markers)
                    .map(|m| target_gt.column(MarkerIdx::new(m as u32)).maf() as f32)
                    .collect()
            };

        // Build hi-frequency / rare marker sets for Stage 1/2
        let rare_threshold = self.config.rare;
        let hi_freq_markers: Vec<usize> = (0..n_markers)
            .filter(|&m| maf[m] >= rare_threshold)
            .collect();
        let rare_markers: Vec<usize> = (0..n_markers)
            .filter(|&m| maf[m] < rare_threshold && maf[m] > 0.0)
            .collect();
        let hi_freq_to_orig: Vec<usize> = hi_freq_markers.clone();

        // Genetic map for this window
        let marker_map = if let Some(map) = gen_maps.get(chrom) {
            MarkerMap::create(target_gt.markers(), map)
        } else {
            MarkerMap::from_positions(target_gt.markers())
        };
        let gen_positions_vec = marker_map.gen_positions().to_vec();
        let hi_freq_gen_positions: Vec<f64> = hi_freq_markers
            .iter()
            .map(|&m| gen_positions_vec[m])
            .collect();

        let stage1_gen_dists: Vec<f64> = if hi_freq_markers.len() > 1 {
            hi_freq_markers
                .windows(2)
                .map(|w| gen_positions_vec[w[1]] - gen_positions_vec[w[0]])
                .collect()
        } else {
            Vec::new()
        };

        if let Some(bb) = &self.telemetry {
            bb.set_stage(Stage::PhasingMain);
            bb.set_producer_stage(Stage::PhasingMain);
            bb.set_total_samples(n_samples as u64);
            bb.set_samples_processed(0);
            bb.set_total_markers(hi_freq_markers.len() as u64);
            bb.set_markers_processed(0);
            bb.set_total_iterations(1);
            bb.set_current_iteration(1);
        }

        // Create sample phases with overlap markers pre-phased
        let confidence_by_sample = build_sample_confidence(&target_gt);
        let phase_mask = target_gt.phase_mask();
        let mut sample_phases = self.create_sample_phases_with_overlap(
            &geno,
            &missing_mask,
            overlap_markers,
            &confidence_by_sample,
            phase_mask,
        );

        if self.reference_gt.is_none() {
            // Fallback: iterative HMM phasing without reference (overlap-aware)
            let n_burnin = self.config.burnin.min(3);
            let n_iterations = self.config.iterations.min(6);
            let total_iterations = n_burnin + n_iterations;
            if let Some(bb) = &self.telemetry {
                bb.set_total_samples(n_samples as u64);
                bb.set_samples_processed(0);
                bb.set_total_markers(n_markers as u64);
                bb.set_markers_processed(0);
                bb.set_total_iterations(total_iterations as u64);
                bb.set_current_iteration(0);
            }

            let gen_dists: Vec<f64> = (0..n_markers.saturating_sub(1))
                .map(|m| {
                    let pos1 = target_gt.marker(MarkerIdx::new(m as u32)).pos;
                    let pos2 = target_gt.marker(MarkerIdx::new((m + 1) as u32)).pos;
                    gen_maps.gen_dist(chrom, pos1, pos2)
                })
                .collect();
            let mut p_recomb: Vec<f32> = std::iter::once(0.0f32)
                .chain(gen_dists.iter().map(|&d| self.params.p_recomb(d)))
                .collect();

            let mut mcmc_paths: Vec<Option<GlobalMosaicPaths>> = vec![None; n_samples];
            for it in 0..total_iterations {
                let is_burnin = it < n_burnin;
                self.params.lr_threshold = self.params.lr_threshold_for_iteration(it);
                if let Some(bb) = &self.telemetry {
                    let stage = if is_burnin {
                        Stage::PhasingBurnin
                    } else {
                        Stage::PhasingMain
                    };
                    bb.set_stage(stage);
                    bb.set_producer_stage(stage);
                    bb.set_current_iteration((it + 1) as u64);
                    bb.set_samples_processed(0);
                    bb.set_markers_processed(0);
                }

                let atomic_estimates = if is_burnin && self.config.em {
                    Some(crate::model::parameters::AtomicParamEstimates::new())
                } else {
                    None
                };

                self.run_phase_baum_iteration(
                    target_gt,
                    &mut geno,
                    &p_recomb,
                    &gen_dists,
                    &mut sample_phases,
                    &mut mcmc_paths,
                    atomic_estimates.as_ref(),
                    &confidence_by_sample,
                )?;
                if let Some(bb) = &self.telemetry {
                    bb.set_samples_processed(n_samples as u64);
                    bb.set_markers_processed(n_markers as u64);
                }

                if let Some(ref atomic) = atomic_estimates {
                    let est = atomic.to_estimates();
                    let mut params_updated = false;
                    if est.n_emit_obs() > 0 {
                        self.params.update_p_mismatch(est.p_mismatch());
                        params_updated = true;
                    }
                    if est.n_switch_obs() > 0 {
                        self.params.update_recomb_intensity(est.recomb_intensity());
                        params_updated = true;
                    }
                    if params_updated {
                        p_recomb = std::iter::once(0.0f32)
                            .chain(gen_dists.iter().map(|&d| self.params.p_recomb(d)))
                            .collect();
                    }
                }
            }
        } else {
            let ref_gt = match self.reference_gt.as_ref() {
                Some(r) => r.clone(),
                None => unreachable!(),
            };
            let alignment = self.alignment.as_ref().ok_or_else(|| {
                crate::error::ReagleError::config("Reference alignment missing for beam phasing")
            })?;

            let packed_ref = PackedRefView::build(&target_gt, &ref_gt, alignment);
            let beam_config = BeamConfig::default();
            let beam_index = PbwtBeamIndex::build(
                &ref_gt,
                alignment,
                &hi_freq_to_orig,
                beam_config.inject_k,
                beam_config.inject_interval,
            );
            let phaser = BeamPhaser::new(&packed_ref, &self.params, beam_config);

            // Compute allele frequencies for TMRCA-aware switch costs
            let hi_freq_allele_freqs: Option<Vec<(f32, f32)>> = {
                let n_ref_haps = packed_ref.n_ref_haps();
                if n_ref_haps > 0 {
                    Some(hi_freq_to_orig.iter().map(|&orig_m| {
                        let mut count0 = 0u32;
                        let mut count1 = 0u32;
                        for h in 0..n_ref_haps {
                            match packed_ref.ref_allele_targ(orig_m, h) {
                                Some(0) => count0 += 1,
                                Some(1) => count1 += 1,
                                _ => {}
                            }
                        }
                        let total = (count0 + count1).max(1) as f32;
                        (count0 as f32 / total, count1 as f32 / total)
                    }).collect())
                } else {
                    None
                }
            };

            let threaded_haps_vec = self.build_phasing_prescan_states(
                target_gt,
                &geno,
                Some(ref_gt.as_ref()),
                self.alignment.as_ref(),
                hi_freq_markers.len(),
                n_samples,
                &hi_freq_gen_positions,
                self.config.imp_step,
                Some(&hi_freq_to_orig),
            )?;

            let n_target_haps = target_gt.n_haplotypes();
            sample_phases
                .par_iter_mut()
                .enumerate()
                .for_each(|(s, sp)| {
                    let mut active_pool = ActivePool::new(packed_ref.n_ref_haps());
                    let mut tmp = vec![crate::model::types::CombinedHapId::from(0u32); threaded_haps_vec[s].n_states()];
                    let th = &threaded_haps_vec[s];
                    th.materialize_at(0, &mut tmp);
                    for id in tmp.iter().copied() {
                        let hid = id.as_u32() as usize;
                        if hid >= n_target_haps {
                            let ref_id = hid - n_target_haps;
                            if ref_id < packed_ref.n_ref_haps() {
                                active_pool.add(ref_id);
                            }
                        }
                    }

                    let condensed = CondensedTarget::build(
                        sp,
                        &hi_freq_to_orig,
                        &hi_freq_gen_positions,
                        hi_freq_allele_freqs.as_deref(),
                        &packed_ref,
                        &self.params,
                    );
                    let mut injector =
                        PbwtInjector::new(&beam_index, packed_ref.n_ref_haps(), beam_config.inject_k);
                    let fwd = phaser.phase_sample(&condensed, sp, &mut active_pool, &mut injector);

                    if !condensed.call_sites.is_empty() {
                        let mut active_pool_rev = ActivePool::new(packed_ref.n_ref_haps());
                        th.materialize_at(0, &mut tmp);
                        for id in tmp.iter().copied() {
                            let hid = id.as_u32() as usize;
                            if hid >= n_target_haps {
                                let ref_id = hid - n_target_haps;
                                if ref_id < packed_ref.n_ref_haps() {
                                    active_pool_rev.add(ref_id);
                                }
                            }
                        }
                        let condensed_rev = condensed.reversed(&hi_freq_gen_positions, &self.params);
                        let mut injector_rev =
                            PbwtInjector::new(&beam_index, packed_ref.n_ref_haps(), beam_config.inject_k);
                        let mut sp_rev = sp.clone();
                        let bwd = phaser.phase_sample(&condensed_rev, &mut sp_rev, &mut active_pool_rev, &mut injector_rev);
                        let mut p_swapped_bwd = bwd.p_swapped;
                        p_swapped_bwd.reverse();
                        let mut combined = Vec::with_capacity(fwd.p_swapped.len());
                        for i in 0..fwd.p_swapped.len() {
                            let pf = fwd.p_swapped[i].clamp(1e-6, 1.0 - 1e-6);
                            let pb = p_swapped_bwd.get(i).copied().unwrap_or(0.5).clamp(1e-6, 1.0 - 1e-6);
                            let lf = (pf / (1.0 - pf)).ln();
                            let lb = (pb / (1.0 - pb)).ln();
                            let logit = lf + lb;
                            let p = 1.0 / (1.0 + (-logit).exp());
                            let p = p.clamp(1e-6, 1.0 - 1e-6);
                            combined.push(p as f32);
                        }
                        for (i, &swapped) in fwd.decisions.iter().enumerate() {
                            let m = condensed.call_sites[i].marker.as_usize();
                            let p = combined.get(i).copied().unwrap_or(0.5);
                            let conf = if swapped { p } else { 1.0 - p };
                            sp.set_phase_confidence(m, conf);
                        }
                    }
                });
        }

        // Sync final phase state from SamplePhase to MutableGenotypes
        self.sync_sample_phases_to_geno(&sample_phases, &mut geno);

        // STAGE 2: Phase rare markers using HMM state probability interpolation
        // Now returns state probabilities for the next overlap region if requested

        let stage1_p_recomb: Vec<f32> = std::iter::once(0.0f32)
            .chain(stage1_gen_dists.iter().map(|&d| self.params.p_recomb(d)))
            .collect();

        let next_overlap_handoff = if !rare_markers.is_empty() && hi_freq_markers.len() >= 2 {
            eprintln!(
                "Stage 2: Phasing {} rare markers using HMM interpolation...",
                rare_markers.len()
            );
            if let Some(bb) = &self.telemetry {
                bb.set_stage(Stage::PhasingStage2);
                bb.set_op("Phasing stage2: HMM interpolation");
                bb.set_total_iterations(0);
                bb.set_current_iteration(0);
                bb.set_total_markers(rare_markers.len() as u64);
                bb.set_markers_processed(0);
                bb.set_samples_processed(0);
            }
            let handoff = self.phase_rare_markers_with_hmm(
                target_gt,
                &mut geno,
                &hi_freq_markers,
                &gen_positions_vec,
                &hi_freq_gen_positions,
                &stage1_p_recomb,
                &mut sample_phases,
                &maf,
                rare_threshold,
                phased_overlap,
                next_overlap_start,
            );
            if let Some(bb) = &self.telemetry {
                bb.set_markers_processed(rare_markers.len() as u64);
                bb.set_samples_processed(n_samples as u64);
            }

            // Sync again after Stage 2
            self.sync_sample_phases_to_geno(&sample_phases, &mut geno);
            handoff
        } else {
            None
        };

        Ok((
            self.build_final_matrix(target_gt, &geno, &sample_phases),
            next_overlap_handoff,
        ))
    }

    /// Apply overlap constraint from previous window's phased output
    ///
    /// This sets the alleles in the overlap region to match the previous window's
    /// phased output, ensuring phase continuity.
    fn apply_overlap_constraint(&self, geno: &mut MutableGenotypes, overlap: &PhasedOverlap) {
        let n_overlap = overlap.n_markers.min(geno.n_markers());
        let n_haps = overlap.n_haps.min(geno.n_haps());

        for h in 0..n_haps {
            let h_idx = HapIdx::new(h as u32);
            for m in 0..n_overlap {
                let allele = overlap.allele(m, h);
                if allele != 255 {
                    geno.set(m, h_idx, allele);
                }
            }
        }
    }

    /// Create SamplePhase instances with overlap markers pre-phased
    ///
    /// Markers in the overlap region (0..overlap_markers) are marked as already
    /// phased since their phase comes from the previous window.
    fn create_sample_phases_with_overlap(
        &self,
        geno: &MutableGenotypes,
        missing_mask: &[BitBox<u8, Lsb0>],
        overlap_markers: usize,
        confidence_by_sample: &[Vec<f32>],
        phase_mask: Option<&Vec<Vec<u8>>>,
    ) -> Vec<SamplePhase> {
        let n_samples = geno.n_haps() / 2;
        let n_markers = geno.n_markers();

        (0..n_samples)
            .map(|s| {
                let hap1 = HapIdx::new((s * 2) as u32);
                let hap2 = HapIdx::new((s * 2 + 1) as u32);

                // Use bulk haplotype access instead of per-marker get()
                let alleles1 = geno.haplotype(hap1);
                let alleles2 = geno.haplotype(hap2);

                // Identify missing markers
                let missing: Vec<usize> = (0..n_markers)
                    .filter(|&m| {
                        missing_mask[hap1.as_usize()][m] || missing_mask[hap2.as_usize()][m]
                    })
                    .collect();

                // Hets in the overlap region are already phased (from previous window)
                // Only hets AFTER the overlap region start as unphased (per input phase mask)
                let unphased: Vec<usize> = (overlap_markers..n_markers)
                    .filter(|&m| {
                        let a1 = alleles1[m];
                        let a2 = alleles2[m];
                        if a1 == a2 {
                            return false;
                        }
                        if missing_mask[hap1.as_usize()][m] || missing_mask[hap2.as_usize()][m] {
                            return false;
                        }
                        match phase_mask.and_then(|mask| mask.get(m).and_then(|row| row.get(s))) {
                            Some(&0) => true,
                            Some(&_) => false,
                            None => true,
                        }
                    })
                    .collect();

                let conf = &confidence_by_sample[s];
                SamplePhase::new(n_markers, &alleles1, &alleles2, conf, &unphased, &missing)
            })
            .collect()
    }

    /// Create SamplePhase instances for all samples
    ///
    /// This initializes phase tracking state from the current genotype data.
    fn create_sample_phases(
        &self,
        geno: &MutableGenotypes,
        confidence_by_sample: &[Vec<f32>],
        phase_mask: Option<&Vec<Vec<u8>>>,
    ) -> Vec<SamplePhase> {
        let n_samples = geno.n_haps() / 2;
        let n_markers = geno.n_markers();

        (0..n_samples)
            .map(|s| {
                let hap1 = HapIdx::new((s * 2) as u32);
                let hap2 = HapIdx::new((s * 2 + 1) as u32);

                // Use bulk haplotype access instead of per-marker get()
                // geno.haplotype() returns 255 for missing positions
                let alleles1 = geno.haplotype(hap1);
                let alleles2 = geno.haplotype(hap2);

                // Identify missing markers using the internal missing tracking
                let missing: Vec<usize> = (0..n_markers)
                    .filter(|&m| geno.is_missing(m, hap1) || geno.is_missing(m, hap2))
                    .collect();

                // Unphased hets: only those explicitly unphased in the input phase mask
                let unphased: Vec<usize> = (0..n_markers)
                    .filter(|&m| {
                        let a1 = alleles1[m];
                        let a2 = alleles2[m];
                        if a1 == a2 || a1 == 255 || a2 == 255 {
                            return false;
                        }
                        match phase_mask.and_then(|mask| mask.get(m).and_then(|row| row.get(s))) {
                            Some(&0) => true,
                            Some(&_) => false,
                            None => true,
                        }
                    })
                    .collect();

                let conf = &confidence_by_sample[s];
                SamplePhase::new(n_markers, &alleles1, &alleles2, conf, &unphased, &missing)
            })
            .collect()
    }

    /// Sync SamplePhase alleles back to MutableGenotypes
    fn sync_sample_phases_to_geno(
        &self,
        sample_phases: &[SamplePhase],
        geno: &mut MutableGenotypes,
    ) {
        let n_markers = geno.n_markers();

        for (s, sp) in sample_phases.iter().enumerate() {
            let hap1 = HapIdx::new((s * 2) as u32);
            let hap2 = HapIdx::new((s * 2 + 1) as u32);

            for m in 0..n_markers {
                let a1 = sp.allele1(m);
                let a2 = sp.allele2(m);
                geno.set(m, hap1, a1);
                geno.set(m, hap2, a2);
            }
        }
    }

    /// Build bidirectional PBWT for a subset of markers (e.g., high-frequency only)
    fn build_bidirectional_pbwt_subset(
        &self,
        geno: &MutableGenotypes,
        marker_indices: &[usize],
        n_haps: usize,
    ) -> BidirectionalPhaseIbs {
        let n_subset = marker_indices.len();
        // Use bulk slice access instead of per-haplotype get() calls
        let mut alleles_by_marker: Vec<Vec<u8>> = Vec::with_capacity(n_subset);

        for &orig_m in marker_indices {
            let marker_slice = geno.marker_alleles(orig_m);
            alleles_by_marker.push(marker_slice[..n_haps].to_vec());
        }

        BidirectionalPhaseIbs::build_for_subset(alleles_by_marker, n_haps, n_subset, marker_indices)
    }

    fn build_phasing_prescan_states<RefPanelSpace>(
        &self,
        target_gt: &GenotypeMatrix,
        target_geno: &MutableGenotypes,
        ref_gt: Option<&GenotypeMatrix<crate::data::storage::phase_state::Phased, RefPanelSpace>>,
        alignment: Option<&MarkerAlignment<crate::data::AnyMarkerSpace, RefPanelSpace>>,
        n_markers: usize,
        n_samples: usize,
        gen_positions: &[f64],
        step_cm: f32,
        marker_map: Option<&[usize]>,
    ) -> Result<Vec<crate::model::states::ThreadedHaps<CombinedHapSpace>>> {
        let telemetry_snapshot = self.telemetry.as_ref().map(|bb| bb.snapshot());
        let n_haps = target_geno.n_haps();
        let n_ref_haps = ref_gt.map(|r| r.n_haplotypes()).unwrap_or(n_haps).max(1);
        let step_cm = PBWT_SELECT_BLOCK_CM.max(step_cm as f64);

        let window_blocks = partition_markers_by_cm(gen_positions, stage1_block_cm(gen_positions));
        if window_blocks.is_empty() {
            return Err(crate::error::ReagleError::vcf(
                "Pre-scan produced no windows for phasing".to_string(),
            ));
        }
        let window_markers_est = window_blocks
            .iter()
            .map(|(s, e)| e.saturating_sub(*s))
            .max()
            .unwrap_or(STAGE1_BLOCK_MIN_MARKERS)
            .max(1);
        let mut n_threads = self
            .config
            .nthreads
            .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
            .unwrap_or(1);
        let mut avail_bytes = available_memory_bytes().unwrap_or(0);
        if avail_bytes < MIN_AVAIL_BYTES_FOR_PLANNING {
            avail_bytes = 0;
        }
        let mut auto_budget = estimate_phase_state_budget(avail_bytes, n_threads, window_markers_est);
        while auto_budget < 64 && n_threads > 1 {
            n_threads = (n_threads / 2).max(1);
            auto_budget = estimate_phase_state_budget(avail_bytes, n_threads, window_markers_est);
        }
        let mut per_window_cap = if self.config.phase_states == 0 {
            if avail_bytes == 0 || auto_budget == 0 {
                n_ref_haps
            } else {
                auto_budget.max(1)
            }
        } else {
            self.config.phase_states
        };
        per_window_cap = per_window_cap.min(n_ref_haps).max(1);
        if self.config.phase_states == 0 {
            eprintln!(
                "Phasing prescan: per_window_cap={} threads={} available_mb={} window_markers={}",
                per_window_cap,
                n_threads,
                avail_bytes / (1024 * 1024),
                window_markers_est
            );
        }
        let num_windows = window_blocks.len();

        let mut boundary_cm = Vec::with_capacity(num_windows.saturating_sub(1));
        for w in 0..num_windows.saturating_sub(1) {
            let (_, end) = window_blocks[w];
            let (next_start, _) = window_blocks[w + 1];
            let left = gen_positions[end.saturating_sub(1).min(gen_positions.len().saturating_sub(1))];
            let right = gen_positions[next_start.min(gen_positions.len().saturating_sub(1))];
            let dist = (right - left).abs().max(step_cm);
            boundary_cm.push(dist);
        }

        let n_full_markers = target_gt.n_markers();
        let (ref_columns, ref_index_map): (Vec<GenotypeColumn>, Option<Vec<usize>>) =
            if let Some(ref_gt) = ref_gt {
                let n_ref_markers = ref_gt.n_markers();
                let mut ref_indices: Vec<usize> = Vec::with_capacity(n_markers);
                for m in 0..n_markers {
                    let orig_m = marker_map.map(|map| map[m]).unwrap_or(m);
                    if let Some(alignment) = alignment {
                        if let Some(ref_m) = alignment.target_to_ref(MarkerIdx::new(orig_m as u32)) {
                            ref_indices.push(ref_m.as_usize());
                        }
                    } else if orig_m < n_ref_markers {
                        ref_indices.push(orig_m);
                    }
                }
                ref_indices.sort_unstable();
                ref_indices.dedup();

                let mut map = vec![usize::MAX; n_ref_markers];
                let mut cols: Vec<GenotypeColumn> = Vec::with_capacity(ref_indices.len());
                for (i, ref_m) in ref_indices.iter().enumerate() {
                    map[*ref_m] = i;
                    cols.push(ref_gt.column(MarkerIdx::new(*ref_m as u32)).clone());
                }
                (cols, Some(map))
            } else {
                let mut map = vec![usize::MAX; n_full_markers];
                let mut cols: Vec<GenotypeColumn> = Vec::with_capacity(n_markers);
                for m in 0..n_markers {
                    let orig_m = marker_map.map(|map| map[m]).unwrap_or(m);
                    if orig_m >= n_full_markers {
                        continue;
                    }
                    let alleles = target_geno.marker_alleles(orig_m);
                    let n_alleles = target_gt
                        .markers()
                        .marker(MarkerIdx::new(orig_m as u32))
                        .n_alleles()
                        .max(1);
                    map[orig_m] = cols.len();
                    cols.push(GenotypeColumn::from_alleles(&alleles, n_alleles));
                }
                (cols, Some(map))
            };

        let freqs = compute_ref_freqs(
            target_gt,
            &ref_columns,
            alignment,
            marker_map,
            ref_index_map.as_deref(),
            n_markers,
        );
        let phase_mask = target_gt.phase_mask();
        let mut mask_unphased_hets = vec![false; n_haps / 2];
        let mut anchors_by_hap: Vec<Vec<(usize, u8, u8)>> = vec![Vec::new(); n_haps];
        let mut ref_col_for_marker = vec![usize::MAX; n_markers];
        if ref_gt.is_some() {
            if let Some(ref_gt) = ref_gt {
                let n_ref_markers = ref_gt.n_markers();
                for m in 0..n_markers {
                    let orig_m = marker_map.map(|map| map[m]).unwrap_or(m);
                    if let Some(alignment) = alignment {
                        if let Some(ref_m) = alignment.target_to_ref(MarkerIdx::new(orig_m as u32)) {
                            if let Some(map) = &ref_index_map {
                                let idx = map.get(ref_m.as_usize()).copied().unwrap_or(usize::MAX);
                                ref_col_for_marker[m] = idx;
                            }
                        }
                    } else if orig_m < n_ref_markers {
                        if let Some(map) = &ref_index_map {
                            let idx = map.get(orig_m).copied().unwrap_or(usize::MAX);
                            ref_col_for_marker[m] = idx;
                        }
                    }
                }
            }
            for s in 0..(n_haps / 2) {
                let hap1 = s * 2;
                let hap2 = s * 2 + 1;
                for m in 0..n_markers {
                    let orig_m = marker_map.map(|map| map[m]).unwrap_or(m);
                    let phased = phase_mask
                        .and_then(|mask| mask.get(orig_m).and_then(|row| row.get(s)))
                        .copied()
                        .unwrap_or(0);
                    if phased == 0 {
                        continue;
                    }
                    let a1 = target_geno.get(orig_m, HapIdx::new(hap1 as u32));
                    let a2 = target_geno.get(orig_m, HapIdx::new(hap2 as u32));
                    if a1 == 255 || a2 == 255 || a1 == a2 {
                        continue;
                    }
                    anchors_by_hap[hap1].push((m, a1, a2));
                    anchors_by_hap[hap2].push((m, a2, a1));
                    mask_unphased_hets[s] = true;
                }
            }
        }
        if n_markers <= 60 {
            let sample = 0usize;
            let mut phased_count = 0usize;
            let mut unphased_hets = 0usize;
            for m in 0..n_markers {
                let phased = phase_mask
                    .and_then(|mask| mask.get(m).and_then(|row| row.get(sample)))
                    .copied()
                    .unwrap_or(0);
                if phased != 0 {
                    phased_count += 1;
                }
                let a1 = target_geno.get(m, HapIdx::new((sample * 2) as u32));
                let a2 = target_geno.get(m, HapIdx::new((sample * 2 + 1) as u32));
                if phased == 0 && a1 != 255 && a2 != 255 && a1 != a2 {
                    unphased_hets += 1;
                }
            }
            eprintln!(
                "[prescan debug] mask_unphased_hets[0]={} phased={} unphased_hets={}",
                mask_unphased_hets.get(sample).copied().unwrap_or(false),
                phased_count,
                unphased_hets
            );
        }
        let avail = available_memory_bytes().unwrap_or(0);
        let batch_size = estimate_scan_batch_size(avail, n_ref_haps, n_haps).max(1);
        let batches_per_window = (n_haps + batch_size - 1) / batch_size;
        let total_batches = num_windows.saturating_mul(batches_per_window).max(1);

        if let Some(bb) = &self.telemetry {
            bb.set_stage(Stage::PhasingPrescan);
            bb.set_producer_stage(Stage::PhasingPrescan);
            bb.set_op("Phasing prescan: PBWT scoring");
            bb.set_total_windows(num_windows as u64);
            bb.set_current_window(0);
            bb.set_total_markers(total_batches as u64);
            bb.set_markers_processed(0);
        }

        let mut scores_by_window_by_hap: Vec<Vec<Vec<(usize, f32)>>> =
            vec![Vec::with_capacity(num_windows); n_haps];

        for (window_idx, &(start, end)) in window_blocks.iter().enumerate() {
            if let Some(bb) = &self.telemetry {
                bb.set_current_window((window_idx + 1) as u64);
                let span_cm = if end > start && end <= gen_positions.len() {
                    (gen_positions[end - 1] - gen_positions[start]).abs()
                } else {
                    0.0
                };
                let span_markers = end.saturating_sub(start);
                bb.set_op(&format!(
                    "Phasing prescan: PBWT scoring (window {}/{}, span_cm={:.3}, markers={})",
                    window_idx + 1,
                    num_windows,
                    span_cm,
                    span_markers
                ));
            }
            let mut informative: Vec<bool> = vec![false; end.saturating_sub(start)];
            if !informative.is_empty() {
                for m in start..end {
                    let orig_m = marker_map
                        .and_then(|map| map.get(m).copied())
                        .unwrap_or(m);
                    let mut info = false;
                    if let Some(mask) = phase_mask {
                        if let Some(row) = mask.get(orig_m) {
                            if row.iter().any(|&v| v != 0) {
                                info = true;
                            }
                        }
                    }
                    if !info {
                        let alleles = target_geno.marker_alleles(orig_m);
                        if alleles.iter().any(|&a| a != 255) {
                            info = true;
                        }
                    }
                    informative[m - start] = info;
                }
            }
            let sampling = build_sampling_points(
                &gen_positions[start..end],
                step_cm,
                PBWT_MIN_MARKER_STEP,
                Some(&informative),
            );
            let k_per_hap = per_window_cap
                .saturating_mul(PBWT_PER_WINDOW_MULT)
                .max(PBWT_MIN_PER_HAP)
                .min(PBWT_MAX_PER_HAP)
                .max(1)
                .min(n_ref_haps.max(1));

            let mut batch_start = 0usize;
            let mut batch_haps_buf: Vec<usize> = Vec::with_capacity(batch_size);
            let mut window_scores_buf: Vec<Vec<f32>> = Vec::with_capacity(batch_size);
            while batch_start < n_haps {
                let batch_end = (batch_start + batch_size).min(n_haps);
                batch_haps_buf.clear();
                batch_haps_buf.extend(batch_start..batch_end);
                let batch_len = batch_haps_buf.len();

                if window_scores_buf.len() < batch_len {
                    let needed = batch_len - window_scores_buf.len();
                    window_scores_buf.extend(
                        (0..needed).map(|_| vec![f32::NEG_INFINITY; n_ref_haps]),
                    );
                }
                for row in window_scores_buf.iter_mut().take(batch_len) {
                    if row.len() != n_ref_haps {
                        row.resize(n_ref_haps, f32::NEG_INFINITY);
                    } else {
                        row.fill(f32::NEG_INFINITY);
                    }
                }

                score_window_batch_pbwt_segment(
                    &batch_haps_buf,
                    target_geno,
                    &ref_columns,
                    phase_mask,
                    Some(&mask_unphased_hets),
                    alignment,
                    &freqs,
                    (start, end),
                    k_per_hap,
                    &sampling,
                    &mut window_scores_buf[..batch_len],
                    ref_gt.is_none(),
                    marker_map,
                    ref_index_map.as_deref(),
                );

                let top_m = per_window_cap
                    .saturating_mul(PBWT_PER_WINDOW_MULT)
                    .max(per_window_cap)
                    .min(n_ref_haps.max(1));
                for (i, &hap_idx) in batch_haps_buf.iter().enumerate() {
                    let top = select_top_k(&window_scores_buf[i], top_m);
                    scores_by_window_by_hap[hap_idx].push(top);
                }

                batch_start = batch_end;
                if let Some(bb) = &self.telemetry {
                    bb.add_markers(1);
                }
            }
        }
        if n_markers <= 60 {
            if let Some(list) = scores_by_window_by_hap.get(0).and_then(|v| v.get(0)) {
                let mut hero_score = None;
                let has_hero = list.iter().any(|(h, _)| {
                    if *h == 98 {
                        hero_score = list.iter().find(|(hh, _)| *hh == 98).map(|(_, s)| *s);
                        true
                    } else {
                        false
                    }
                });
                eprintln!(
                    "[prescan debug] window0_top={:?} hero98_in_top={} hero98_score={:?}",
                    list.iter().take(10).collect::<Vec<_>>(),
                    has_hero,
                    hero_score
                );
            }
        }

        let per_window_caps = vec![per_window_cap; num_windows];
        let global_slot_budget = per_window_caps.iter().copied().sum::<usize>().max(1);
        let exclude_self = ref_gt.is_none();
        let debug_watch = vec![198usize, 199usize];
        let ref_has_panel = ref_gt.is_some();
        let out: Vec<crate::model::states::ThreadedHaps<CombinedHapSpace>> = (0..n_samples)
            .into_par_iter()
            .map(|s| {
            let hap1 = s * 2;
            let hap2 = s * 2 + 1;
            let mut dense_merge_buffer = vec![f32::NEG_INFINITY; n_ref_haps.max(1)];
            let mut touched_indices: Vec<usize> =
                Vec::with_capacity(per_window_cap.saturating_mul(PBWT_PER_WINDOW_MULT).max(1));
            let mut window_scores: Vec<Vec<(usize, f32)>> = Vec::with_capacity(num_windows);
            let mut prev_window_scores: Vec<(usize, f32)> = Vec::new();
            for w in 0..num_windows {
                for &idx in &touched_indices {
                    dense_merge_buffer[idx] = f32::NEG_INFINITY;
                }
                touched_indices.clear();

                for &(h, score) in scores_by_window_by_hap[hap1][w]
                    .iter()
                    .chain(scores_by_window_by_hap[hap2][w].iter())
                {
                    if h >= dense_merge_buffer.len() {
                        continue;
                    }
                    let current = &mut dense_merge_buffer[h];
                    if current.is_finite() {
                        if score > *current {
                            *current = score;
                        }
                    } else {
                        *current = score;
                        touched_indices.push(h);
                    }
                }

                touched_indices.sort_by(|&a, &b| {
                    dense_merge_buffer[b]
                        .partial_cmp(&dense_merge_buffer[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let cap = per_window_cap
                    .saturating_mul(PBWT_PER_WINDOW_MULT)
                    .max(per_window_cap)
                    .min(n_ref_haps.max(1));
                let take = cap.min(touched_indices.len());
                let mut list: Vec<(usize, f32)> = Vec::with_capacity(take);
                for &h in touched_indices.iter().take(take) {
                    list.push((h, dense_merge_buffer[h]));
                }
                if list.is_empty() {
                    list.extend(prev_window_scores.iter().copied());
                } else if !prev_window_scores.is_empty() {
                    let mut map: HashMap<usize, f32> = list.iter().copied().collect();
                    for (h, score) in prev_window_scores.iter().copied() {
                        map.entry(h).or_insert(score);
                    }
                    list = map.into_iter().collect();
                }
                prev_window_scores = list.clone();
                window_scores.push(list);
            }

            let abyss = vec![false; n_ref_haps];
            let (mut candidate_haps, mut scores_by_hap) = build_sparse_scores(&window_scores, &abyss);
            if s == 0 && n_markers <= 60 {
                let hero_in_candidates = candidate_haps.iter().any(|&h| h == 98);
                eprintln!(
                    "[prescan debug] hero98_in_candidates={}",
                    hero_in_candidates
                );
                if hero_in_candidates {
                    if let Some(pos) = candidate_haps.iter().position(|&h| h == 98) {
                        eprintln!(
                            "[prescan debug] hero98_scores={:?}",
                            scores_by_hap.get(pos)
                        );
                    }
                }
                if n_markers <= 12 && ref_has_panel {
                    let mut all_zero = Vec::new();
                    let mut all_one = Vec::new();
                    for h in 0..n_ref_haps {
                        let hap_idx = HapIdx::new(h as u32);
                        let mut ok0 = true;
                        let mut ok1 = true;
                        for m in 0..n_markers {
                            let ref_col_idx = ref_col_for_marker[m];
                            if ref_col_idx == usize::MAX {
                                continue;
                            }
                            let ref_al = ref_columns
                                .get(ref_col_idx)
                                .map(|c| c.get(hap_idx))
                                .unwrap_or(255);
                            if ref_al != 0 {
                                ok0 = false;
                            }
                            if ref_al != 1 {
                                ok1 = false;
                            }
                            if !ok0 && !ok1 {
                                break;
                            }
                        }
                        if ok0 {
                            all_zero.push(h);
                        }
                        if ok1 {
                            all_one.push(h);
                        }
                    }
                    eprintln!("[prescan debug] all_zero_haps={:?}", all_zero);
                    eprintln!("[prescan debug] all_one_haps={:?}", all_one);
                    let zero_in_candidates = all_zero.iter().any(|h| candidate_haps.contains(h));
                    let one_in_candidates = all_one.iter().any(|h| candidate_haps.contains(h));
                    eprintln!(
                        "[prescan debug] all_zero_in_candidates={} all_one_in_candidates={}",
                        zero_in_candidates, one_in_candidates
                    );
                }
            }
            if s == 0 {
                eprintln!(
                    "[prescan] sample={} candidate_haps={} windows={}",
                    s,
                    candidate_haps.len(),
                    num_windows
                );
                for &h in &debug_watch {
                    let found = candidate_haps.iter().any(|&c| c == h);
                    eprintln!("[prescan] watch_hap={} present={}", h, found);
                }
            }
            if ref_has_panel && PBWT_ANCHOR_TOP_HAPS > 0 {
                let hap1 = s * 2;
                let hap2 = s * 2 + 1;
                let mut anchor_scores = vec![0i32; n_ref_haps];
                for &(start, end) in &window_blocks {
                    let mut window_anchors: Vec<(usize, u8, u8)> = Vec::new();
                    if let Some(list) = anchors_by_hap.get(hap1) {
                        for &(m, a1, a2) in list {
                            if m >= start && m < end {
                                window_anchors.push((m, a1, a2));
                            }
                        }
                    }
                    if let Some(list) = anchors_by_hap.get(hap2) {
                        for &(m, a1, a2) in list {
                            if m >= start && m < end {
                                window_anchors.push((m, a1, a2));
                            }
                        }
                    }
                    if window_anchors.is_empty() {
                        continue;
                    }
                    anchor_scores.fill(0);
                    for h in 0..n_ref_haps {
                        let hap_idx = HapIdx::new(h as u32);
                        let mut score = 0i32;
                        for (m, a1, _) in &window_anchors {
                            let ref_col_idx = ref_col_for_marker[*m];
                            if ref_col_idx == usize::MAX {
                                continue;
                            }
                            let ref_al = ref_columns
                                .get(ref_col_idx)
                                .map(|c| c.get(hap_idx))
                                .unwrap_or(255);
                            if ref_al == 255 {
                                continue;
                            }
                            if ref_al == *a1 {
                                score += 1;
                            } else {
                                score -= 1;
                            }
                        }
                        anchor_scores[h] = score;
                    }
                    let mut idxs: Vec<usize> = (0..n_ref_haps).collect();
                    idxs.sort_by(|&a, &b| anchor_scores[b].cmp(&anchor_scores[a]));
                    let take = PBWT_ANCHOR_TOP_HAPS.min(idxs.len());
                    for &h in idxs.iter().take(take) {
                        if !candidate_haps.contains(&h) {
                            candidate_haps.push(h);
                            scores_by_hap.push(Vec::new());
                        }
                    }
                }
            }
            let allocation = allocate_lms_sparse(
                &scores_by_hap,
                &candidate_haps,
                num_windows,
                &boundary_cm,
                &self.params,
                n_ref_haps,
                global_slot_budget,
                &per_window_caps,
            );

            let mut selected: Vec<RefHapId> =
                allocation.intervals_by_hap.into_iter().map(|(h, _)| RefHapId::from(h)).collect();
            selected.sort_unstable();
            selected.dedup();
            if PBWT_FORCE_TOP_HAPS > 0 && !window_scores.is_empty() {
                for &idx in &touched_indices {
                    dense_merge_buffer[idx] = f32::NEG_INFINITY;
                }
                touched_indices.clear();
                for list in &window_scores {
                    for &(h, score) in list {
                        if h >= dense_merge_buffer.len() {
                            continue;
                        }
                        let current = &mut dense_merge_buffer[h];
                        if current.is_finite() {
                            if score > *current {
                                *current = score;
                            }
                        } else {
                            *current = score;
                            touched_indices.push(h);
                        }
                    }
                }
                touched_indices.sort_by(|&a, &b| {
                    dense_merge_buffer[b]
                        .partial_cmp(&dense_merge_buffer[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let take = PBWT_FORCE_TOP_HAPS.min(touched_indices.len());
                for &h in touched_indices.iter().take(take) {
                    let hid = RefHapId::from(h);
                    if !selected.contains(&hid) {
                        selected.push(hid);
                    }
                }
            }
            if exclude_self {
                selected.retain(|h| (h.as_usize() / 2) != s);
            }
            if selected.is_empty() {
                let fallback_cap = per_window_cap.min(n_ref_haps.max(1)).max(1);
                let mut fallback: Vec<RefHapId> =
                    (0..fallback_cap).map(RefHapId::from).collect();
                if exclude_self {
                    fallback.retain(|h| (h.as_usize() / 2) != s);
                }
                if fallback.is_empty() {
                    fallback.push(RefHapId::from(0usize));
                }
                selected = fallback;
            }

            if s == 0 && n_markers <= 60 {
                let mut selected_dbg = selected.clone();
                selected_dbg.sort_unstable();
                eprintln!(
                    "[prescan debug] selected_ref_len={} first={:?}",
                    selected_dbg.len(),
                    selected_dbg.iter().take(10).collect::<Vec<_>>()
                );
            }

            let offset = if ref_has_panel { n_haps } else { 0 };
            let mut th = crate::model::states::ThreadedHaps::<CombinedHapSpace>::new(
                selected.len(),
                selected.len(),
                n_markers,
            );
            for h in selected {
                let combined = combined_from_ref(h, offset as u32);
                th.push_new(combined);
            }
            if s == 0 && n_markers <= 60 {
                let mut buf = vec![CombinedHapId::from(0u32); th.n_states()];
                th.materialize_at(0, &mut buf);
                let mut selected_dbg: Vec<usize> =
                    buf.iter().map(|id| id.as_u32() as usize).collect();
                selected_dbg.sort_unstable();
                eprintln!(
                    "[prescan debug] sample=0 selected_combined_len={} first={:?}",
                    selected_dbg.len(),
                    selected_dbg.iter().take(10).collect::<Vec<_>>()
                );
            }
            th
        })
        .collect();

        if let Some(bb) = &self.telemetry {
            if let Some(prev) = telemetry_snapshot {
                bb.set_stage(prev.stage);
                bb.set_producer_stage(prev.producer_stage);
                bb.set_consumer_stage(prev.consumer_stage);
                bb.set_current_window(prev.current_window);
                bb.set_total_windows(prev.total_windows);
                bb.set_current_iteration(prev.current_iteration);
                bb.set_total_iterations(prev.total_iterations);
                bb.set_samples_processed(prev.samples_processed);
                bb.set_total_samples(prev.total_samples);
                bb.set_markers_processed(prev.markers_processed);
                bb.set_total_markers(prev.total_markers);
                bb.set_op(&prev.current_op);
                bb.set_producer_op(&prev.producer_op);
                bb.set_consumer_op(&prev.consumer_op);
            }
        }

        Ok(out)
    }

    /// Run a single phasing iteration using Forward-Backward Li-Stephens HMM
    ///
    /// This uses the full Forward-Backward algorithm to compute posterior probabilities
    /// of the phase, ensuring that phasing decisions are informed by both upstream
    /// and downstream data.
    #[instrument(skip_all, fields(n_samples, n_markers))]
    fn run_phase_baum_iteration(
        &mut self,
        target_gt: &GenotypeMatrix,
        geno: &mut MutableGenotypes,
        p_recomb: &[f32],
        gen_dists: &[f64],
        sample_phases: &mut [SamplePhase],
        mcmc_paths: &mut [Option<GlobalMosaicPaths>],
        atomic_estimates: Option<&crate::model::parameters::AtomicParamEstimates>,
        confidence_by_sample: &[Vec<f32>],
    ) -> Result<()> {
        let n_samples = geno.n_haps() / 2;
        let n_markers = geno.n_markers();
        let n_haps = geno.n_haps();
        let mut gen_positions = Vec::with_capacity(n_markers);
        gen_positions.push(0.0);
        for i in 1..n_markers {
            let dist = gen_dists.get(i - 1).copied().unwrap_or(0.0);
            gen_positions.push(gen_positions[i - 1] + dist);
        }

        tracing::Span::current().record("n_samples", n_samples);
        tracing::Span::current().record("n_markers", n_markers);

        // Compute total haplotype count (target + reference)
        let n_ref_haps = self
            .reference_gt
            .as_ref()
            .map(|r| r.n_haplotypes())
            .unwrap_or(0);
        let n_total_haps = n_haps + n_ref_haps;

        // No clone needed: the HMM phase is read-only; mutations happen after.
        // We use a scoped immutable borrow that ends before the swap phase.
        // Build composite haplotypes for all samples using streaming PBWT
        // This uses O(N) memory instead of O(M*N) for the PBWT index
        let final_states = self.params.n_states.min(n_total_haps).max(1);
        let ref_gt = self.reference_gt.as_ref().map(|v| v.as_ref());
        let threaded_haps_vec =
            tracing::info_span!("prescan_selection").in_scope(|| {
                self.build_phasing_prescan_states(
                    target_gt,
                    geno,
                    ref_gt,
                    self.alignment.as_ref(),
                    n_markers,
                    n_samples,
                    &gen_positions,
                    self.config.imp_step,
                    None,
                )
            })?;
        let swap_results: Vec<(
            BitVec<u8, Lsb0>,
            Vec<(usize, f32)>,
            Vec<(usize, f32)>,
            Option<GlobalMosaicPaths>,
        )> =
            info_span!("build_composite_view").in_scope(|| {
                // Immutable borrow of geno for the entire read phase
                let ref_geno: &MutableGenotypes = geno;

                // Use Composite view when reference panel is available
                let ref_view: GenotypeView<'_, crate::data::AnyMarkerSpace, RefSpace> =
                    if let (Some(ref_gt), Some(alignment)) = (&self.reference_gt, &self.alignment) {
                        GenotypeView::Composite {
                            target: ref_geno,
                            reference: ref_gt,
                            alignment,
                            n_target_haps: n_haps,
                        }
                    } else {
                        GenotypeView::Mutable(ref_geno)
                    };
                let prior_paths = &mcmc_paths[..];
                let sample_phase_view: &[SamplePhase] = &*sample_phases;
                let mut swap_results: Vec<(
                    BitVec<u8, Lsb0>,
                    Vec<(usize, f32)>,
                    Vec<(usize, f32)>,
                    Option<GlobalMosaicPaths>,
                )> = vec![
                    (
                        BitVec::repeat(false, n_markers),
                        Vec::new(),
                        Vec::new(),
                        None
                    );
                    n_samples
                ];

                tracing::info_span!("hmm_samples").in_scope(|| {
                    swap_results
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(s, (mask, het_lr_out, het_phase_out, paths_out))| {
                            let sample_idx = SampleIdx::new(s as u32);
                            let hap1 = sample_idx.hap1();
                            let hap2 = sample_idx.hap2();
                            let sample_seed = (self.config.seed as u64)
                                .wrapping_add(s as u64)
                                .wrapping_add(0xA5A5_5A5A_D00Du64);

                            // Use pre-built composite haplotypes from streaming PBWT
                            let threaded_haps_full = threaded_haps_vec[s].clone();
                            let n_states_full = threaded_haps_full.n_states();
                            let mut threaded_haps = threaded_haps_full.clone();
                            let mut n_states = n_states_full;
                            let mut selection_applied = false;

                            // Convert global prior paths to local paths for this iteration
                            let local_prior = prior_paths[s].as_ref().and_then(|gp| {
                                global_to_local_paths(gp, &threaded_haps_full, n_markers)
                            });

                            // 2. Extract current alleles for H1 and H2
                            let seq1 = ref_geno.haplotype(hap1);
                            let seq2 = ref_geno.haplotype(hap2);
                            // Use pre-computed confidence instead of recomputing
                            let sample_conf = &confidence_by_sample[s];

                            // 3. Run HMM with per-heterozygote swap probabilities
                            // Following Java PhaseBaum2.java: interleave phase decisions in the forward pass.
                            //
                            // Key Algorithm (3-Track HMM):
                            // 1. Run backward pass for BOTH haplotypes first, storing backward values
                            // 2. Run forward pass marker-by-marker for BOTH haplotypes
                            // 3. At each het, compute swap probability using fwd and stored bwd
                            // 4. After the forward pass, sample a swap mask via MCMC
                            // 5. Apply the sampled mask to update phase
                            //
                            // Collect EM statistics if requested (using original sequences)
                            // Only create HMM when needed to avoid unnecessary p_recomb.clone()
                            if n_states_full > final_states {
                                let hmm_full = MosaicHmm::new(
                                    ref_view,
                                    &self.params,
                                    n_states_full,
                                    p_recomb.to_vec(),
                                );
                                let mut fwd1 = Vec::new();
                                let mut bwd1 = Vec::new();
                                let mut fwd2 = Vec::new();
                                let mut bwd2 = Vec::new();

                                let plp = PlProvider {
                                    gt: target_gt,
                                    sample: s,
                                    subset_to_orig: None,
                                };
                                hmm_full.conditioned_forward_backward(
                                    &seq1,
                                    &seq2,
                                    &seq2,
                                    Some(sample_conf),
                                    Some(&plp),
                                    None,
                                    None,
                                    &threaded_haps_full,
                                    &mut fwd1,
                                    &mut bwd1,
                                );
                                hmm_full.conditioned_forward_backward(
                                    &seq2,
                                    &seq1,
                                    &seq1,
                                    Some(sample_conf),
                                    Some(&plp),
                                    None,
                                    None,
                                    &threaded_haps_full,
                                    &mut fwd2,
                                    &mut bwd2,
                                );

                                let probs1 = compute_state_posteriors(&fwd1, &bwd1, n_markers, n_states_full);
                                let probs2 = compute_state_posteriors(&fwd2, &bwd2, n_markers, n_states_full);
                                let top_selected =
                                    select_top_k_by_mass_two(&probs1, &probs2, n_states_full, final_states);
                                let mut forced_counts: std::collections::HashMap<usize, u32> =
                                    std::collections::HashMap::new();
                                if let Some(prior) = local_prior.as_ref() {
                                    for &state in prior.path1.iter().chain(prior.path2.iter()) {
                                        let entry = forced_counts
                                            .entry(state as usize)
                                            .or_insert(0);
                                        *entry += 1;
                                    }
                                }

                                let mut forced: Vec<usize> = forced_counts.keys().copied().collect();
                                forced.sort_by(|&a, &b| {
                                    let ca = forced_counts.get(&a).copied().unwrap_or(0);
                                    let cb = forced_counts.get(&b).copied().unwrap_or(0);
                                    cb.cmp(&ca).then_with(|| a.cmp(&b))
                                });

                                let mut selected: Vec<usize> = Vec::new();
                                let cap = final_states.max(1);
                                for &state in forced.iter().take(cap) {
                                    if state < n_states_full {
                                        selected.push(state);
                                    }
                                }
                                for &state in &top_selected {
                                    if selected.len() >= cap {
                                        break;
                                    }
                                    if !selected.contains(&state) {
                                        selected.push(state);
                                    }
                                }
                                if selected.is_empty() {
                                    selected = top_selected;
                                }

                                threaded_haps = threaded_haps_full.subset_states(&selected);
                                n_states = threaded_haps.n_states();
                                selection_applied = true;
                            }

                            if let Some(atomic) = atomic_estimates {
                                let hmm = MosaicHmm::new(
                                    ref_view,
                                    &self.params,
                                    n_states,
                                    p_recomb.to_vec(),
                                );
                                let mut local_est = crate::model::parameters::ParamEstimates::new();
                                hmm.collect_stats(&seq1, &threaded_haps, gen_dists, &mut local_est);
                                hmm.collect_stats(&seq2, &threaded_haps, gen_dists, &mut local_est);
                                atomic.add_estimation_data(&local_est);
                            }

                            // 3-Track HMM with Prior-First Approach
                            //
                            // This implementation avoids the numerically unstable division workaround.
                            // Instead, we:
                            // 1. Run sparse backward passes, storing only at het positions
                            // 2. Run forward with prior-first: compute transition before emission
                            // 3. At hets: use prior (no emission) to evaluate both hypotheses
                            // 4. Apply combined emission after decision for numerical stability

                            // Identify heterozygote positions first
                            let het_positions: Vec<usize> = (0..n_markers)
                                .filter(|&m| {
                                    let a1 = seq1[m];
                                    let a2 = seq2[m];
                                    a1 != 255
                                        && a2 != 255
                                        && a1 != a2
                                        && sample_phase_view[s].is_unphased(m)
                                })
                                .collect();

                            let p_err = self.params.p_mismatch;
                            let p_no_err = 1.0 - p_err;

                            let (swap_bits, swap_lr, swap_probs, new_paths) = THREAD_WORKSPACE
                                .with(|ws| {
                                    let mut workspace = ws.borrow_mut();
                                    if workspace.is_none() {
                                        *workspace = Some(
                                            crate::utils::workspace::ThreadWorkspace::new(64, 0),
                                        );
                                    }
                                    let ws = workspace.as_mut().unwrap();
                                    ws.clear(); // Explicit reset between samples to prevent state contamination
                                    let ref_provider =
                                        RefAlleleProvider::new(ref_view, &threaded_haps);
                                    let (anchor_h1, anchor_h2) =
                                        build_anchor_constraints(&sample_phase_view[s]);

                                    let donor_blocks =
                                        partition_markers_by_cm(&gen_positions, stage1_block_cm(&gen_positions));
                                    let block_starts: Arc<[usize]> =
                                        blocks_to_starts(&donor_blocks, n_markers)
                                            .into_boxed_slice()
                                            .into();
                                    let result = sample_swap_bits_mosaic(
                                        n_markers,
                                        n_states,
                                        p_recomb,
                                        &seq1,
                                        &seq2,
                                        &sample_conf,
                                        ref_provider,
                                        Some(PlProvider {
                                            gt: target_gt,
                                            sample: s,
                                            subset_to_orig: None,
                                        }),
                                        block_starts,
                                        &het_positions,
                                        if selection_applied {
                                            None
                                        } else {
                                            local_prior.as_ref()
                                        },
                                        Some(&anchor_h1),
                                        Some(&anchor_h2),
                                        sample_seed,
                                        self.config.mcmc_burnin,
                                        self.config.mcmc_lr_samples,
                                        p_no_err,
                                        p_err,
                                        ws,
                                    );
                                    result
                                });
                            if new_paths.path1.is_empty() {
                                *paths_out = None;
                            } else {
                                *paths_out = Some(local_to_global_paths(
                                    &new_paths,
                                    &threaded_haps,
                                    n_markers,
                                ));
                            }
                            *het_lr_out = het_positions
                                .iter()
                                .copied()
                                .zip(swap_lr.iter().copied())
                                .collect();
                            *het_phase_out = het_positions
                                .iter()
                                .copied()
                                .zip(swap_probs.into_iter())
                                .collect();
                            assert!(swap_lr.len() <= n_markers);
                            assert!(het_phase_out.len() <= het_positions.len());
                            let mut swapped = false;
                            let mut swap_idx = 0usize;
                            for m in 0..n_markers {
                                if swap_idx < het_positions.len() && het_positions[swap_idx] == m {
                                    swapped = swap_bits.get(swap_idx).copied().unwrap_or(0) == 1;
                                    swap_idx += 1;
                                }
                                if swapped {
                                    mask.set(m, true);
                                }
                            }
                        })
                });

                swap_results
            }); // ref_geno borrow ends here

        // Apply Swaps
        // After computing swap masks for all samples, apply them sequentially.
        // This is done sequentially because swap_haplotypes requires mutable access.
        info_span!("apply_swaps").in_scope(|| {
            for (s, (mask, het_lr_values, het_phase_values, paths)) in
                swap_results.into_iter().enumerate()
            {
                let sample_idx = SampleIdx::new(s as u32);
                let hap1 = sample_idx.hap1();
                let hap2 = sample_idx.hap2();
                geno.swap_haplotypes(hap1, hap2, &mask);

                if s < sample_phases.len() {
                    let sp = &mut sample_phases[s];

                    for m in mask.iter_ones() {
                        sp.swap_alleles(m);
                    }

                    for (m, p_orient) in het_phase_values {
                        sp.set_phase_confidence(m, p_orient);
                    }

                    let lr_threshold = self.params.lr_threshold;
                    for (m, lr) in het_lr_values {
                        if lr >= lr_threshold {
                            sp.mark_phased(m);
                        }
                    }
                }
                if let Some(paths) = paths {
                    if let Some(slot) = mcmc_paths.get_mut(s) {
                        *slot = Some(paths);
                    }
                }
            }
        });

        Ok(())
    }

    /// Run Stage 1 phasing iteration on HIGH-FREQUENCY markers only using FB HMM
    ///
    /// Uses SamplePhase to track phase state and only phases unphased markers.
    fn run_phase_baum_iteration_stage1(
        &mut self,
        target_gt: &GenotypeMatrix,
        geno: &mut MutableGenotypes,
        threaded_haps_vec: &[crate::model::states::ThreadedHaps<CombinedHapSpace>],
        stage1_p_recomb: &[f32],
        stage1_gen_dists: &[f64],
        hi_freq_to_orig: &[usize],
        stage1_blocks: &[(usize, usize)],
        ibs2: &Ibs2,
        sample_phases: &mut [SamplePhase],
        mcmc_paths: &mut [Option<GlobalMosaicPaths>],
        atomic_estimates: Option<&crate::model::parameters::AtomicParamEstimates>,
        iteration: usize,
    ) -> Result<(usize, usize)> {
        let n_stage1_blocks = stage1_blocks.len();
        if n_stage1_blocks == 0 {
            return Ok((0, 0));
        }
        let n_haps = geno.n_haps();

        let n_samples = sample_phases.len();
        let n_hi_freq = hi_freq_to_orig.len();


        // No clone needed: the HMM phase is read-only; mutations happen after.
        // We use a scoped immutable borrow that ends before the apply phase.
        type PhaseDecision = (
            Vec<bool>,
            Vec<(usize, f32)>,
            Vec<(usize, f32)>,
            Option<GlobalMosaicPaths>,
        );
        let phase_decisions: Vec<PhaseDecision> = {
            // Immutable borrow of geno for the entire read phase
            let ref_geno: &MutableGenotypes = geno;

            // 1. Create Subset View for Stage 1 markers
            // Use CompositeSubset when reference panel is available
            let subset_view =
                if let (Some(ref_gt), Some(alignment)) = (&self.reference_gt, &self.alignment) {
                    GenotypeView::CompositeSubset {
                        target: ref_geno,
                        reference: ref_gt,
                        alignment,
                        subset: hi_freq_to_orig,
                        n_target_haps: n_haps,
                    }
                } else {
                    GenotypeView::MutableSubset {
                        geno: ref_geno,
                        subset: hi_freq_to_orig,
                    }
                };

            // 2. Build bidirectional PBWT on high-frequency markers only
            let use_dynamic_mcmc = self.config.dynamic_mcmc && self.reference_gt.is_none();
            let phase_ibs = if use_dynamic_mcmc {
                Some(self.build_bidirectional_pbwt_subset(ref_geno, hi_freq_to_orig, n_haps))
            } else {
                None
            };

            // Collect phase decisions per sample using correct per-het algorithm.
            // Returns: (swap_mask, het_lr_values) per sample where:
            //   - swap_mask[i] = true if the sampled phase orientation at marker i is swapped
            //   - het_lr_values = (hi_freq_idx, lr) for each het, used for phased marking threshold
            let prior_paths = &mcmc_paths[..];
            let telemetry = self.telemetry.clone();
            let block_starts: Arc<[usize]> = if use_dynamic_mcmc {
                Arc::from([])
            } else {
                blocks_to_starts(stage1_blocks, n_hi_freq)
                    .into_boxed_slice()
                    .into()
            };
            let sample_iter = || {
                sample_phases.par_iter().enumerate().map(|(s, sp)| {
                    THREAD_WORKSPACE.with(|ws| {
                        let mut workspace = ws.borrow_mut();
                        if workspace.is_none() {
                            *workspace =
                                Some(crate::utils::workspace::ThreadWorkspace::new(64, 0));
                        }
                        let ws = workspace.as_mut().unwrap();
                        ws.clear();

                        let n_hi_freq = hi_freq_to_orig.len();
                        let threaded_haps = &threaded_haps_vec[s];
                        let mut threaded_haps = threaded_haps.clone();

                        // Extract alleles/confidence for SUBSET of markers using reused buffers
                        ws.seq1.clear();
                        ws.seq2.clear();
                        ws.sample_conf.clear();
                        ws.seq1.reserve(n_hi_freq);
                        ws.seq2.reserve(n_hi_freq);
                        ws.sample_conf.reserve(n_hi_freq);
                        for &m in hi_freq_to_orig {
                            ws.seq1.push(sp.allele1(m));
                            ws.seq2.push(sp.allele2(m));
                            ws.sample_conf.push(sp.confidence(m));
                        }
                        let seq1 = std::mem::take(&mut ws.seq1);
                        let seq2 = std::mem::take(&mut ws.seq2);
                        let sample_conf = std::mem::take(&mut ws.sample_conf);

                        let sample_seed = (self.config.seed as u64)
                            .wrapping_add(s as u64)
                            .wrapping_add((iteration as u64) << 32)
                            .wrapping_add(0xFEED_FACE_1234u64);

                        // Identify UNPHASED heterozygote positions in hi-freq marker space
                        ws.het_positions.clear();
                        for i in 0..n_hi_freq {
                            let m = hi_freq_to_orig[i];
                            let a1 = seq1[i];
                            let a2 = seq2[i];
                            if a1 != 255 && a2 != 255 && a1 != a2 && sp.is_unphased(m) {
                                ws.het_positions.push(i);
                            }
                        }
                        let het_positions = std::mem::take(&mut ws.het_positions);

                        if het_positions.is_empty() {
                            // No hets to phase: no swaps needed, no LR values
                            ws.seq1 = seq1;
                            ws.seq2 = seq2;
                            ws.sample_conf = sample_conf;
                            ws.het_positions = het_positions;
                            return (vec![false; n_hi_freq], Vec::new(), Vec::new(), None);
                        }

                        let p_err = self.params.p_mismatch;
                        let p_no_err = 1.0 - p_err;

                        if self.reference_gt.is_some() {
                            let mut anchors: Vec<(usize, u8, u8)> = Vec::new();
                            for (i, &m) in hi_freq_to_orig.iter().enumerate() {
                                if sp.is_unphased(m) {
                                    continue;
                                }
                                let a1 = sp.allele1(m);
                                let a2 = sp.allele2(m);
                                if a1 == 255 || a2 == 255 || a1 == a2 {
                                    continue;
                                }
                                anchors.push((i, a1, a2));
                            }
                            if !anchors.is_empty() {
                                let n_ref_haps = self
                                    .reference_gt
                                    .as_ref()
                                    .map(|r| r.n_haplotypes())
                                    .unwrap_or(0);
                                let offset = n_haps;
                                let mut best_hap1 = (0usize, 0u32);
                                let mut best_hap2 = (0usize, 0u32);
                                for h in 0..n_ref_haps {
                                    let hap_idx = HapIdx::new((offset + h) as u32);
                                    let mut score1 = 0u32;
                                    let mut score2 = 0u32;
                                    for &(i, a1, a2) in &anchors {
                                        let ref_al = subset_view.allele(MarkerIdx::new(i as u32), hap_idx);
                                        if ref_al == a1 {
                                            score1 += 1;
                                        }
                                        if ref_al == a2 {
                                            score2 += 1;
                                        }
                                    }
                                    if score1 > best_hap1.1 {
                                        best_hap1 = (h, score1);
                                    }
                                    if score2 > best_hap2.1 {
                                        best_hap2 = (h, score2);
                                    }
                                }
                                if best_hap1.1 > 0 || best_hap2.1 > 0 {
                                    let mut existing = vec![CombinedHapId::from(0u32); threaded_haps.n_states()];
                                    threaded_haps.materialize_at(0, &mut existing);
                                    let has_hap = |hap: u32| existing.iter().any(|g| g.as_u32() == hap);
                                    let hap1_id = (offset + best_hap1.0) as u32;
                                    if best_hap1.1 > 0 && !has_hap(hap1_id) {
                                        threaded_haps.push_new(CombinedHapId::new(hap1_id));
                                    }
                                    let hap2_id = (offset + best_hap2.0) as u32;
                                    if best_hap2.1 > 0 && !has_hap(hap2_id) {
                                        threaded_haps.push_new(CombinedHapId::new(hap2_id));
                                    }
                                }

                                let max_scan = 10_000_000usize;
                                if n_ref_haps.saturating_mul(n_hi_freq) <= max_scan && n_ref_haps > 0 {
                                    let mut scores: Vec<i32> = vec![0; n_ref_haps];
                                    for h in 0..n_ref_haps {
                                        let hap_idx = HapIdx::new((offset + h) as u32);
                                        let mut score = 0i32;
                                        for i in 0..n_hi_freq {
                                            let a1 = seq1[i];
                                            let a2 = seq2[i];
                                            if a1 == 255 && a2 == 255 {
                                                continue;
                                            }
                                            let ref_al =
                                                subset_view.allele(MarkerIdx::new(i as u32), hap_idx);
                                            if a1 == a2 {
                                                if ref_al == a1 {
                                                    score += 1;
                                                } else {
                                                    score -= 1;
                                                }
                                            } else if ref_al == a1 || ref_al == a2 {
                                                score += 1;
                                            } else {
                                                score -= 1;
                                            }
                                        }
                                        scores[h] = score;
                                    }
                                    let mut idxs: Vec<usize> = (0..n_ref_haps).collect();
                                    idxs.sort_by(|&a, &b| scores[b].cmp(&scores[a]));
                                    let take = PBWT_FORCE_TOP_HAPS.min(idxs.len());
                                    if take > 0 {
                                        let mut existing =
                                            vec![CombinedHapId::from(0u32); threaded_haps.n_states()];
                                        threaded_haps.materialize_at(0, &mut existing);
                                        let has_hap =
                                            |hap: u32| existing.iter().any(|g| g.as_u32() == hap);
                                        for &h in idxs.iter().take(take) {
                                            let hap_id = (offset + h) as u32;
                                            if !has_hap(hap_id) {
                                                threaded_haps.push_new(CombinedHapId::new(hap_id));
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        let n_states = threaded_haps.n_states();

                        // Collect EM statistics if requested
                        if let Some(atomic) = atomic_estimates {
                            let hmm = MosaicHmm::new(
                                subset_view,
                                &self.params,
                                n_states,
                                stage1_p_recomb.to_vec(),
                            );
                            let mut local_est = crate::model::parameters::ParamEstimates::new();
                            hmm.collect_stats(&seq1, &threaded_haps, stage1_gen_dists, &mut local_est);
                            hmm.collect_stats(&seq2, &threaded_haps, stage1_gen_dists, &mut local_est);
                            atomic.add_estimation_data(&local_est);
                        }

                        let (swap_bits, swap_lr, swap_probs, new_paths) = if use_dynamic_mcmc {
                            // SHAPEIT5-style dynamic MCMC: re-select states each step
                            let prior_local = prior_paths[s].as_ref().map(|gp| MosaicPaths {
                                path1: gp.path1.iter().map(|id| id.as_u32()).collect(),
                                path2: gp.path2.iter().map(|id| id.as_u32()).collect(),
                            });
                            let (anchor_h1_full, anchor_h2_full) = build_anchor_constraints(sp);
                            let mut anchor_h1 = Vec::with_capacity(n_hi_freq);
                            let mut anchor_h2 = Vec::with_capacity(n_hi_freq);
                            for &m in hi_freq_to_orig {
                                anchor_h1.push(anchor_h1_full[m]);
                                anchor_h2.push(anchor_h2_full[m]);
                            }

                            let (swap_bits, swap_lr, swap_probs, new_paths) = if self.config.profile {
                                info_span!("run_dynamic_mcmc", sample = s).in_scope(|| {
                                    sample_dynamic_mcmc(
                                        n_hi_freq,
                                        n_states,
                                        stage1_p_recomb,
                                        &seq1,
                                        &seq2,
                                        &sample_conf,
                                        phase_ibs.as_ref().expect("phase_ibs"),
                                        ibs2,
                                        s as u32,
                                        &het_positions,
                                        sample_seed,
                                        self.config.mcmc_steps,
                                        p_no_err,
                                        p_err,
                                        prior_local.as_ref(),
                                        Some(&anchor_h1),
                                        Some(&anchor_h2),
                                        ws,
                                    )
                                })
                            } else {
                                sample_dynamic_mcmc(
                                    n_hi_freq,
                                    n_states,
                                    stage1_p_recomb,
                                    &seq1,
                                    &seq2,
                                    &sample_conf,
                                    phase_ibs.as_ref().expect("phase_ibs"),
                                    ibs2,
                                    s as u32,
                                    &het_positions,
                                    sample_seed,
                                    self.config.mcmc_steps,
                                    p_no_err,
                                    p_err,
                                    prior_local.as_ref(),
                                    Some(&anchor_h1),
                                    Some(&anchor_h2),
                                    ws,
                                )
                            };
                            let global_paths = GlobalMosaicPaths {
                                path1: new_paths.path1.into_iter().map(CombinedHapId::from).collect(),
                                path2: new_paths.path2.into_iter().map(CombinedHapId::from).collect(),
                            };
                            (swap_bits, swap_lr, swap_probs, Some(global_paths))
                        } else {
                            // Classic Beagle-style: static state space MCMC with thread-local workspace
                            let ref_provider = if self.config.profile {
                                info_span!("prep_allele_provider", sample = s).in_scope(|| {
                                    RefAlleleProvider::new(subset_view, &threaded_haps)
                                })
                            } else {
                                RefAlleleProvider::new(subset_view, &threaded_haps)
                            };

                            let local_prior_raw = prior_paths[s]
                                .as_ref()
                                .and_then(|gp| global_to_local_paths(gp, &threaded_haps, n_hi_freq));
                            let (anchor_h1_full, anchor_h2_full) = build_anchor_constraints(sp);
                            let has_anchors = anchor_h1_full.iter().any(|&a| a != 255)
                                || anchor_h2_full.iter().any(|&a| a != 255);
                            let local_prior = if has_anchors {
                                None
                            } else {
                                local_prior_raw.as_ref()
                            };
                            let mut anchor_h1 = Vec::with_capacity(n_hi_freq);
                            let mut anchor_h2 = Vec::with_capacity(n_hi_freq);
                            for &m in hi_freq_to_orig {
                                anchor_h1.push(anchor_h1_full[m]);
                                anchor_h2.push(anchor_h2_full[m]);
                            }

                            let block_starts = block_starts.clone();
                            let result = if self.config.profile {
                                info_span!("run_mcmc_math", sample = s).in_scope(|| {
                                    sample_swap_bits_mosaic(
                                        n_hi_freq,
                                        n_states,
                                        stage1_p_recomb,
                                        &seq1,
                                        &seq2,
                                        &sample_conf,
                                        ref_provider,
                                        Some(PlProvider {
                                            gt: target_gt,
                                            sample: s,
                                            subset_to_orig: Some(hi_freq_to_orig),
                                        }),
                                        block_starts,
                                        &het_positions,
                                        local_prior,
                                        Some(&anchor_h1),
                                        Some(&anchor_h2),
                                        sample_seed,
                                        self.config.mcmc_burnin,
                                        self.config.mcmc_lr_samples,
                                        p_no_err,
                                        p_err,
                                        ws,
                                    )
                                })
                            } else {
                                sample_swap_bits_mosaic(
                                    n_hi_freq,
                                    n_states,
                                    stage1_p_recomb,
                                    &seq1,
                                    &seq2,
                                    &sample_conf,
                                    ref_provider,
                                    Some(PlProvider {
                                        gt: target_gt,
                                        sample: s,
                                        subset_to_orig: Some(hi_freq_to_orig),
                                    }),
                                    block_starts,
                                    &het_positions,
                                    local_prior,
                                    Some(&anchor_h1),
                                    Some(&anchor_h2),
                                    sample_seed,
                                    self.config.mcmc_burnin,
                                    self.config.mcmc_lr_samples,
                                    p_no_err,
                                    p_err,
                                    ws,
                                )
                            };
                            let global_paths =
                                local_to_global_paths(&result.3, &threaded_haps, n_hi_freq);
                            (result.0, result.1, result.2, Some(global_paths))
                        };

                        let mut swap_mask = vec![false; n_hi_freq];
                        let mut anchor_resets = 0usize;
                        let mut p_swap = vec![0.5f32; n_hi_freq];
                        for (idx, &pos) in het_positions.iter().enumerate() {
                            let p_orient = swap_probs.get(idx).copied().unwrap_or(0.5);
                            let swap_bit = swap_bits.get(idx).copied().unwrap_or(0);
                            let p = if swap_bit == 1 { p_orient } else { 1.0 - p_orient };
                            p_swap[pos] = p.clamp(0.0, 1.0);
                            swap_mask[pos] = p_swap[pos] > 0.5;
                        }
                        for i in 0..n_hi_freq {
                            let m = hi_freq_to_orig[i];
                            let a1 = seq1[i];
                            let a2 = seq2[i];
                            let is_het = a1 != 255 && a2 != 255 && a1 != a2;
                            let is_phased_het = is_het && !sp.is_unphased(m);
                            if is_phased_het {
                                let a1_anchor = sp.allele1(m);
                                let a2_anchor = sp.allele2(m);
                                if a1 == a1_anchor && a2 == a2_anchor {
                                    swap_mask[i] = false;
                                } else if a1 == a2_anchor && a2 == a1_anchor {
                                    swap_mask[i] = true;
                                }
                                anchor_resets += 1;
                            }
                        }

                        eprintln!(
                            "[phase anchors] sample={} anchors={} hets={}",
                            s,
                            anchor_resets,
                            het_positions.len()
                        );

                        if let Some(ref paths) = new_paths {
                            if paths.path1.len() == n_hi_freq && paths.path2.len() == n_hi_freq {
                                for i in 0..n_hi_freq {
                                    let a1 = seq1[i];
                                    let a2 = seq2[i];
                                    if a1 == 255 || a2 == 255 || a1 == a2 {
                                        continue;
                                    }
                                    let h1 = paths.path1[i].as_u32();
                                    let h2 = paths.path2[i].as_u32();
                                    let ref1 = subset_view.allele(
                                        MarkerIdx::new(i as u32),
                                        HapIdx::new(h1),
                                    );
                                    let ref2 = subset_view.allele(
                                        MarkerIdx::new(i as u32),
                                        HapIdx::new(h2),
                                    );
                                    if ref1 == a1 && ref2 == a2 {
                                        swap_mask[i] = false;
                                    } else if ref1 == a2 && ref2 == a1 {
                                        swap_mask[i] = true;
                                    }
                                }
                            }
                        }
                        if s == 0 && n_hi_freq <= 12 {
                            let mut rows = Vec::with_capacity(n_hi_freq);
                            let mut p1_ids = Vec::with_capacity(n_hi_freq);
                            let mut p2_ids = Vec::with_capacity(n_hi_freq);
                            for i in 0..n_hi_freq {
                                let a1 = seq1[i];
                                let a2 = seq2[i];
                                let (ref1, ref2) = if let Some(ref paths) = new_paths {
                                    if paths.path1.len() == n_hi_freq && paths.path2.len() == n_hi_freq
                                    {
                                        p1_ids.push(paths.path1[i].as_u32());
                                        p2_ids.push(paths.path2[i].as_u32());
                                        let h1 = paths.path1[i].as_u32();
                                        let h2 = paths.path2[i].as_u32();
                                        (
                                            subset_view.allele(MarkerIdx::new(i as u32), HapIdx::new(h1)),
                                            subset_view.allele(MarkerIdx::new(i as u32), HapIdx::new(h2)),
                                        )
                                    } else {
                                        (255, 255)
                                    }
                                } else {
                                    (255, 255)
                                };
                                rows.push((i, a1, a2, ref1, ref2, swap_mask[i]));
                            }
                            if !p1_ids.is_empty() {
                                eprintln!("[swap debug] path1_ids={:?}", p1_ids);
                                eprintln!("[swap debug] path2_ids={:?}", p2_ids);
                            }
                            eprintln!("[swap debug] i a1 a2 ref1 ref2 swap={:?}", rows);
                        }

                        let het_lr_values: Vec<(usize, f32)> = het_positions
                            .iter()
                            .copied()
                            .zip(swap_lr.into_iter())
                            .collect();
                        let het_phase_values: Vec<(usize, f32)> = het_positions
                            .iter()
                            .copied()
                            .zip(swap_probs.iter().copied())
                            .map(|(idx, p_swap)| {
                                let conf = p_swap.max(1.0 - p_swap);
                                (idx, conf)
                            })
                            .collect();

                        ws.seq1 = seq1;
                        ws.seq2 = seq2;
                        ws.sample_conf = sample_conf;
                        ws.het_positions = het_positions;

                        if let Some(ref paths) = new_paths {
                            let mut p1_switches = 0usize;
                            let mut p2_switches = 0usize;
                            for i in 1..paths.path1.len() {
                                if paths.path1[i] != paths.path1[i - 1] {
                                    p1_switches += 1;
                                }
                                if paths.path2[i] != paths.path2[i - 1] {
                                    p2_switches += 1;
                                }
                            }
                            eprintln!(
                                "[mosaic paths] sample={} path1_switches={} path2_switches={} markers={}",
                                s,
                                p1_switches,
                                p2_switches,
                                paths.path1.len()
                            );
                        }

                        if let Some(bb) = telemetry.as_ref() {
                            bb.add_samples(1);
                        }

                        (swap_mask, het_lr_values, het_phase_values, new_paths)
                    })
                })
            };

            if self.config.profile {
                info_span!("phase_sample_all", samples = n_samples)
                    .in_scope(|| sample_iter().collect())
            } else {
                sample_iter().collect()
            }
        }; // ref_geno borrow ends here

        // Apply phase decisions to SamplePhase
        let mut total_switches = 0;
        let mut total_phased = 0;

        // Determine if we're in burn-in (don't mark as phased during burn-in)
        let is_burnin = iteration < self.config.burnin;
        let lr_threshold = self.params.lr_threshold;

        for (s, (swap_mask, het_lr_values, het_phase_values, new_paths)) in
            phase_decisions.into_iter().enumerate()
        {
            let sp = &mut sample_phases[s];

            // Apply swaps using the mask (correctly handles cumulative swap propagation)
            for (hi_freq_idx, should_swap) in swap_mask.into_iter().enumerate() {
                if should_swap {
                    let m = hi_freq_to_orig[hi_freq_idx];
                    sp.swap_alleles(m);
                    total_switches += 1;
                }
            }

            // Mark hets as phased if LR exceeds threshold (independent of swap decision)
            if !is_burnin {
                for (hi_freq_idx, lr) in het_lr_values {
                    if lr >= lr_threshold {
                        let m = hi_freq_to_orig[hi_freq_idx];
                        sp.mark_phased(m);
                        total_phased += 1;
                    }
                }
            }

            for (hi_freq_idx, p_orient) in het_phase_values {
                let m = hi_freq_to_orig[hi_freq_idx];
                sp.set_phase_confidence(m, p_orient);
            }

            if let Some(paths) = new_paths {
                if let Some(slot) = mcmc_paths.get_mut(s) {
                    *slot = Some(paths);
                }
            }
        }

        // Also update MutableGenotypes to keep in sync for next iteration's PBWT
        self.sync_sample_phases_to_geno(sample_phases, geno);

        // Calculate cumulative phasing progress across all samples.
        let mut total_locked = 0usize;
        let mut total_unphased = 0usize;
        for sp in sample_phases.iter() {
            total_locked += sp.phased_count();
            total_unphased += sp.unphased_count();
        }
        let total_phasable = total_locked + total_unphased;
        let pct_locked = if total_phasable > 0 {
            (total_locked as f64 / total_phasable as f64) * 100.0
        } else {
            100.0
        };

        eprintln!(
            "Applied {} phase switches, {} new markers locked (Stage 1 FB)",
            total_switches, total_phased
        );
        eprintln!(
            "  Completion: {:.1}% ({}/{} heterozygous markers locked)",
            pct_locked, total_locked, total_phasable
        );
        Ok((total_switches, total_phased))
    }

    /// Build final GenotypeMatrix from mutable genotypes
    fn build_final_matrix(
        &self,
        original: &GenotypeMatrix,
        geno: &MutableGenotypes,
        sample_phases: &[SamplePhase],
    ) -> GenotypeMatrix<crate::data::storage::phase_state::Phased> {
        let markers = original.markers().clone();
        let samples = original.samples_arc();
        let n_markers = geno.n_markers();
        let n_samples = samples.len();

        let columns: Vec<GenotypeColumn> = (0..n_markers)
            .map(|m| {
                let alleles = geno.marker_alleles(m);
                let bytes: Vec<u8> = alleles.to_vec();
                GenotypeColumn::from_alleles(&bytes, 2)
            })
            .collect();

        let mut phase_confidence = vec![vec![255u8; n_samples]; n_markers];
        for (s, sp) in sample_phases.iter().enumerate() {
            for m in 0..n_markers {
                let p = sp.phase_confidence(m).clamp(0.0, 1.0);
                phase_confidence[m][s] = (p * 255.0).round() as u8;
            }
        }

        let confidence = original.confidence_clone();
        let pl = original.likelihoods_pl_arc();
        GenotypeMatrix::new_phased_with_confidence_and_likelihoods(
            markers, columns, samples, confidence, pl,
        )
        .with_phase_confidence(Some(phase_confidence))
    }

    fn count_unphased_hets(sample_phases: &[SamplePhase], hi_freq_to_orig: &[usize]) -> usize {
        let mut count = 0usize;
        for sp in sample_phases {
            for &m in hi_freq_to_orig {
                let a1 = sp.allele1(m);
                let a2 = sp.allele2(m);
                if a1 != 255 && a2 != 255 && a1 != a2 && sp.is_unphased(m) {
                    count += 1;
                }
            }
        }
        count
    }

    /// Stage 2: Phase rare markers using HMM state probability interpolation
    ///
    /// This implements the proper algorithm from Java Beagle's Stage2Baum.java:
    ///
    /// 1. Run HMM on high-frequency markers to get state probabilities for each haplotype
    /// 2. For each rare heterozygote:
    ///    - Find flanking high-frequency markers (mkrA, mkrB)
    ///    - Interpolate state probabilities: prob = wt*probsA[j] + (1-wt)*probsB[j]
    ///    - Accumulate allele probabilities from reference haplotypes
    /// 3. Decide phase: p1 = alProbs1[a1] * alProbs2[a2], p2 = alProbs1[a2] * alProbs2[a1]
    ///    Switch if p2 > p1
    ///
    /// **Key fix**: Only phases markers that are currently UNPHASED in SamplePhase.
    ///
    /// **Streaming Soft-Handoff**: Accepts optional `previous_overlap` to combine state probabilities
    /// from the previous window with current ones, ensuring continuity. The returned handoff
    /// includes identity-aware haplotype priors for the *next* overlap region.
    fn phase_rare_markers_with_hmm(
        &self,
        target_gt: &GenotypeMatrix,
        geno: &mut MutableGenotypes,
        hi_freq_markers: &[usize],
        gen_positions: &[f64],
        hi_freq_gen_positions: &[f64],
        stage1_p_recomb: &[f32],
        sample_phases: &mut [SamplePhase],
        maf: &[f32],
        rare_threshold: f32,
        previous_overlap: Option<&PhasedOverlap>,
        next_overlap_start: Option<usize>,
    ) -> Option<Stage2OverlapHandoff> {
        let n_markers = geno.n_markers();
        let n_haps = geno.n_haps();
        let n_ref_haps = self
            .reference_gt
            .as_ref()
            .map(|r| r.n_haplotypes())
            .unwrap_or(0);
        let n_total_haps = n_haps + n_ref_haps;
        let n_stage1 = hi_freq_markers.len();
        let seed = self.config.seed;
        let n_haps_f = target_gt.n_haplotypes() as f32;
        let has_ref = self.reference_gt.is_some() && self.alignment.is_some();
        let alt_freqs: Vec<f32> = if let (Some(ref_gt), Some(alignment)) =
            (&self.reference_gt, &self.alignment)
        {
            let n_ref_haps = ref_gt.n_haplotypes() as f32;
            let prior_alpha = 1.0f32;
            let prior_beta = 1.0f32;
            let mut freqs = vec![0.0f32; n_markers];
            for m in 0..n_markers {
                let fallback = if n_haps_f > 0.0 {
                    let alt = target_gt.column(MarkerIdx::new(m as u32)).alt_count() as f32;
                    ((alt + prior_alpha) / (n_haps_f + prior_alpha + prior_beta)).clamp(0.0, 1.0)
                } else {
                    0.5
                };
                let mapping = alignment
                    .allele_mappings
                    .get(m)
                    .and_then(|m| m.as_ref());
                if let Some(mapping) = mapping {
                    if let Some(ref_idx) = alignment.target_to_ref(MarkerIdx::new(m as u32)) {
                        if n_ref_haps > 0.0 {
                            let ref_col = ref_gt.column(ref_idx);
                            let ref_alt = (ref_col.alt_count() as f32 + prior_alpha)
                                / (n_ref_haps + prior_alpha + prior_beta);
                            if let Some(&targ_to_ref_alt) = mapping.targ_to_ref.get(1) {
                                if targ_to_ref_alt == 1 {
                                    freqs[m] = ref_alt.clamp(0.0, 1.0);
                                    continue;
                                }
                                if targ_to_ref_alt == 0 {
                                    freqs[m] = (1.0 - ref_alt).clamp(0.0, 1.0);
                                    continue;
                                }
                            }
                        }
                    }
                }
                freqs[m] = fallback;
            }
            freqs
        } else if n_haps_f > 0.0 {
            let prior_alpha = 1.0f32;
            let prior_beta = 1.0f32;
            (0..n_markers)
                .map(|m| {
                    let alt = target_gt.column(MarkerIdx::new(m as u32)).alt_count() as f32;
                    ((alt + prior_alpha) / (n_haps_f + prior_alpha + prior_beta))
                        .clamp(0.0, 1.0)
                })
                .collect()
        } else {
            vec![0.5f32; n_markers]
        };

        if n_stage1 < 2 {
            return None;
        }

        // Compute total haplotype count (target + reference)

        // Determine Stage 1 markers involved in the NEXT overlap region (for export)
        let next_overlap_indices = if let Some(start) = next_overlap_start {
            // Find first Stage 1 marker >= start
            let start_stage1 = hi_freq_markers
                .iter()
                .position(|&m| m >= start)
                .unwrap_or(n_stage1);
            (start_stage1..n_stage1).collect()
        } else {
            Vec::new()
        };

        // Determine Stage 1 markers involved in the PREVIOUS overlap region (for import/merge)
        let n_stage1_in_prev_overlap = if let Some(overlap) = previous_overlap {
            // Overlap markers are 0..overlap.n_markers
            hi_freq_markers
                .iter()
                .take_while(|&&m| m < overlap.n_markers)
                .count()
        } else {
            0
        };

        // Build Stage 2 interpolation mappings
        let stage2_phaser = Stage2Phaser::new(
            hi_freq_markers,
            gen_positions,
            n_markers,
            self.params.recomb_intensity,
        );

        // Result container for next window's state probs
        // We will collect this from parallel iteration.
        // It needs to be ordered by haplotype.

        // Return type from parallel map
        type PhaseResult = (
            Vec<Stage2Decision>,
            Option<Vec<Vec<Vec<f32>>>>,
            Option<[HaplotypePriors; 2]>,
            Option<usize>,
        );

        let n_samples = n_haps / 2;
        let ref_gt = self.reference_gt.as_ref().map(|v| v.as_ref());
        let threaded_haps_vec = match self.build_phasing_prescan_states(
            target_gt,
            geno,
            ref_gt,
            self.alignment.as_ref(),
            n_stage1,
            n_samples,
            hi_freq_gen_positions,
            self.config.imp_step,
            Some(hi_freq_markers),
        ) {
            Ok(states) => states,
            Err(err) => {
                eprintln!("Stage 2 prescan failed: {err}");
                return None;
            }
        };

        // No clone needed: we only read geno during computation; local rephase
        // happens during threaded hap construction above.
        // We use a scoped immutable borrow for the entire computation phase.
        let phase_results: Vec<PhaseResult> = {
            // Immutable borrow of geno for the entire read phase
            let ref_geno: &MutableGenotypes = geno;
            let phase_ibs = if has_ref {
                None
            } else {
                Some(self.build_bidirectional_pbwt_subset(ref_geno, hi_freq_markers, n_haps))
            };

            let rare_markers: Vec<usize> = (0..n_markers)
                .filter(|&m| maf[m] < rare_threshold && maf[m] > 0.0)
                .collect();

            // Use CompositeSubset view when reference panel is available
            let subset_view =
                if let (Some(ref_gt), Some(alignment)) = (&self.reference_gt, &self.alignment) {
                    GenotypeView::CompositeSubset {
                        target: ref_geno,
                        reference: ref_gt,
                        alignment,
                        subset: hi_freq_markers,
                        n_target_haps: n_haps,
                    }
                } else {
                    GenotypeView::MutableSubset {
                        geno: ref_geno,
                        subset: hi_freq_markers,
                    }
                };

            let get_allele_global = |marker: usize, hap: usize| -> u8 {
                if hap < n_haps {
                    ref_geno.get(marker, HapIdx::new(hap as u32))
                } else {
                    let ref_h = hap - n_haps;
                    if let (Some(ref_gt), Some(alignment)) = (&self.reference_gt, &self.alignment) {
                        if let Some(ref_m) = alignment.target_to_ref(MarkerIdx::new(marker as u32))
                        {
                            let ref_allele = ref_gt.allele(ref_m, HapIdx::new(ref_h as u32));
                            alignment.reverse_map_allele(marker, ref_allele)
                        } else {
                            255
                        }
                    } else {
                        255
                    }
                }
            };

            let mut carrier_haps: Vec<Vec<u32>> = vec![Vec::new(); n_markers];
            for &m in &rare_markers {
                let mut carriers = Vec::new();
                for h in 0..n_total_haps {
                    let allele = get_allele_global(m, h);
                    if allele > 0 && allele != 255 {
                        carriers.push(h as u32);
                    }
                }
                carrier_haps[m] = carriers;
            }

            // Process samples in parallel - collect results: Stage2Decision
            // Note: This is called after all iterations, so we use iteration=0 for deterministic state selection
            sample_phases
                .par_iter()
                .enumerate()
                .map(|(s, sp)| {
                    // Create deterministic RNG for this sample for random tie-breaking
                    // Seed combines global seed + sample index + constant for Stage 2 distinction
                    use rand::{Rng, SeedableRng};
                    let sample_seed = (seed as u64)
                        .wrapping_add(s as u64)
                        .wrapping_add(0xDEAD_BEEF_CAFE_u64); // Stage 2 distinction constant
                    let mut rng = rand::rngs::StdRng::seed_from_u64(sample_seed);

                    let threaded_haps = &threaded_haps_vec[s];
                    let n_states = threaded_haps.n_states();

                    // Extract Stage 1 alleles from SamplePhase
                    let seq1: Vec<u8> = hi_freq_markers.iter().map(|&m| sp.allele1(m)).collect();
                    let seq2: Vec<u8> = hi_freq_markers.iter().map(|&m| sp.allele2(m)).collect();
                    let seq_conf: Vec<f32> =
                        hi_freq_markers.iter().map(|&m| sp.confidence(m)).collect();
                    let hmm = MosaicHmm::new(
                        subset_view,
                        &self.params,
                        n_states,
                        stage1_p_recomb.to_vec(),
                    );
                    let plp = PlProvider {
                        gt: target_gt,
                        sample: s,
                        subset_to_orig: Some(hi_freq_markers),
                    };

                    let mut fwd1 = Vec::new();
                    let mut bwd1 = Vec::new();
                    let (init_prior1_storage, init_prior2_storage) = if let Some(overlap) =
                        previous_overlap
                    {
                        let h1_idx = s * 2;
                        let h2_idx = s * 2 + 1;
                        let mut prior_stage1_idx = n_stage1_in_prev_overlap
                            .saturating_sub(1)
                            .min(n_stage1.saturating_sub(1));
                        if let Some(prior_marker) = overlap.prior_stage1_global_marker() {
                            if let Some(idx) = hi_freq_markers.iter().position(|&m| m == prior_marker)
                            {
                                prior_stage1_idx = idx;
                            }
                        }
                        let current_global_marker = hi_freq_markers.get(prior_stage1_idx).copied();
                        if let (Some(expected), Some(current)) =
                            (overlap.prior_stage1_global_marker(), current_global_marker)
                        {
                            if expected != current {
                                panic!(
                                    "Stage2 hap prior marker mismatch: expected={}, current={}, sample={}",
                                    expected, current, s
                                );
                            }
                        }

                        // Identity-aware handoff: project haplotype priors onto the
                        // current window's local state set using state->hap mapping.
                        if let Some(hap_priors) = overlap.hap_priors() {
                            if prior_stage1_idx < n_stage1
                                && h1_idx < hap_priors.len()
                                && h2_idx < hap_priors.len()
                                && n_states > 0
                            {
                                let mut state_haps = vec![CombinedHapId::new(0); n_states];
                                threaded_haps.materialize_at(prior_stage1_idx, &mut state_haps);

                                (
                                    Some(project_haplotype_priors_to_states(
                                        &hap_priors[h1_idx],
                                        &state_haps,
                                    )),
                                    Some(project_haplotype_priors_to_states(
                                        &hap_priors[h2_idx],
                                        &state_haps,
                                    )),
                                )
                            } else {
                                (None, None)
                            }
                        } else {
                            (None, None)
                        }
                    } else {
                        (None, None)
                    };
                    let init_prior1: Option<&[f32]> = init_prior1_storage.as_deref();
                    let init_prior2: Option<&[f32]> = init_prior2_storage.as_deref();
                    let allele_freqs_stage1: Vec<f32> =
                        hi_freq_markers.iter().map(|&m| alt_freqs[m]).collect();
                    hmm.conditioned_forward_backward(
                        &seq1,
                        &seq2,
                        &seq2,
                        Some(&seq_conf),
                        Some(&plp),
                        Some(&allele_freqs_stage1),
                        init_prior1,
                        &threaded_haps,
                        &mut fwd1,
                        &mut bwd1,
                    );

                    let mut fwd2 = Vec::new();
                    let mut bwd2 = Vec::new();
                    hmm.conditioned_forward_backward(
                        &seq1,
                        &seq2,
                        &seq1,
                        Some(&seq_conf),
                        Some(&plp),
                        Some(&allele_freqs_stage1),
                        init_prior2,
                        &threaded_haps,
                        &mut fwd2,
                        &mut bwd2,
                    );

                    // Compute posterior state probabilities at each Stage 1 marker
                    let probs1 = compute_state_posteriors(&fwd1, &bwd1, n_stage1, n_states);
                    let probs2 = compute_state_posteriors(&fwd2, &bwd2, n_stage1, n_states);

                    // Do NOT merge previous window probabilities by state index. State
                    // identity is not preserved across windows, and blending by index
                    // corrupts the posterior.

                    // Extract state probs for next window (if needed)
                    let next_probs = if !next_overlap_indices.is_empty() {
                        let mut p1_tail = Vec::with_capacity(next_overlap_indices.len());
                        let mut p2_tail = Vec::with_capacity(next_overlap_indices.len());

                        for &i in &next_overlap_indices {
                            if i < probs1.len() {
                                p1_tail.push(probs1[i].clone());
                            } else {
                                p1_tail.push(vec![0.0; n_states]);
                            }
                            if i < probs2.len() {
                                p2_tail.push(probs2[i].clone());
                            } else {
                                p2_tail.push(vec![0.0; n_states]);
                            }
                        }
                        Some(vec![p1_tail, p2_tail])
                    } else {
                        None
                    };

                    // Export identity-aware haplotype priors for the next window.
                    let (next_hap_priors, next_prior_global_marker) = if !next_overlap_indices
                        .is_empty()
                    {
                        let stage1_idx = next_overlap_indices[0];
                        if stage1_idx < probs1.len() && n_states > 0 {
                            let mut state_haps = vec![CombinedHapId::new(0); n_states];
                            threaded_haps.materialize_at(stage1_idx, &mut state_haps);

                            let prior1 = build_haplotype_priors_from_state_probs(
                                &probs1[stage1_idx],
                                &state_haps,
                                PRIOR_EXPORT_MIN_PROB,
                            );
                            let prior2 = build_haplotype_priors_from_state_probs(
                                &probs2[stage1_idx],
                                &state_haps,
                                PRIOR_EXPORT_MIN_PROB,
                            );
                            let marker = hi_freq_markers.get(stage1_idx).copied();
                            (Some([prior1, prior2]), marker)
                        } else {
                            (None, None)
                        }
                    } else {
                        (None, None)
                    };

                    // Lazy cache for state->hap mapping - O(1) indexing with Option<Vec>
                    // Uses immutable materialize_at() to avoid clone() overhead
                    let mut hap_cache: Vec<Option<Vec<CombinedHapId>>> = vec![None; n_markers];

                    macro_rules! get_haps {
                        ($marker:expr) => {{
                            let m = $marker;
                            if hap_cache[m].is_none() {
                                let mut haps = vec![CombinedHapId::new(0); n_states];
                                threaded_haps.materialize_at(m, &mut haps);
                                hap_cache[m] = Some(haps);
                            }
                            hap_cache[m].as_ref().unwrap()
                        }};
                    }

                    // Closure to get allele for any haplotype (target or reference)
                    let get_allele = |marker: usize, hap: usize| -> u8 {
                        if hap < n_haps {
                            // Target haplotype
                            ref_geno.get(marker, HapIdx::new(hap as u32))
                        } else {
                            // Reference haplotype
                            let ref_h = hap - n_haps;
                            if let (Some(ref_gt), Some(alignment)) =
                                (&self.reference_gt, &self.alignment)
                            {
                                if let Some(ref_m) =
                                    alignment.target_to_ref(MarkerIdx::new(marker as u32))
                                {
                                    let ref_allele = ref_gt.allele(ref_m, HapIdx::new(ref_h as u32));
                                    alignment.reverse_map_allele(marker, ref_allele)
                                } else {
                                    255 // Missing - marker not in reference
                                }
                            } else {
                                255 // No reference panel
                            }
                        }
                    };

                    let mut decisions: Vec<Stage2Decision> = Vec::new();

                    // Inline helper macro for imputing a single allele
                    // Matches Java Stage2Baum.imputeAllele()
                    macro_rules! impute_allele {
                        ($m:expr, $probs:expr) => {{
                            let m = $m;
                            let probs = $probs;
                            let n_alleles = target_gt
                                .markers()
                                .marker(MarkerIdx::new(m as u32))
                                .n_alleles()
                                .max(1);
                            let mut al_probs = vec![0.0f32; n_alleles];

                            let mkr_a = stage2_phaser.prev_stage1_marker[m];
                            let state_haps = get_haps!(mkr_a);
                            let n_states = state_haps.len();
                            let bridge_probs = stage2_phaser.bridge_state_probs(m, probs, n_states);

                            for (j, &hap) in state_haps.iter().enumerate() {
                                let prob_state = bridge_probs.get(j).copied().unwrap_or(0.0);
                                let hap_allele = get_allele(m, hap.as_u32() as usize);

                                if hap_allele != 255 {
                                    let idx = hap_allele as usize;
                                    if idx < al_probs.len() {
                                        al_probs[idx] += prob_state;
                                    }
                                }
                            }

                            al_probs
                                .iter()
                                .enumerate()
                                .max_by(|(_, a), (_, b)| {
                                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                                })
                                .map(|(idx, _)| idx as u8)
                                .unwrap_or(0)
                        }};
                    }

                    // Inline helper macro for carrier score calculation
                    macro_rules! carrier_score {
                        ($m:expr, $probs:expr, $carrier_set:expr) => {{
                            let m = $m;
                            let probs = $probs;
                            let carrier_set = $carrier_set;
                            let state_haps = get_haps!(m);
                            let n_states = state_haps.len();
                            let bridge_probs = stage2_phaser.bridge_state_probs(m, probs, n_states);
                            let mut score = 0.0f32;
                            for (j, &hap) in state_haps.iter().enumerate() {
                                let prob = bridge_probs.get(j).copied().unwrap_or(0.0);
                                if carrier_set.contains(&hap.as_u32()) {
                                    score += prob;
                                }
                            }
                            score
                        }};
                    }

                    for &m in &rare_markers {
                        let a1 = sp.allele1(m);
                        let a2 = sp.allele2(m);

                        // Handle missing genotypes by imputation
                        if sp.is_missing(m) || a1 == 255 || a2 == 255 {
                            let imp_a1 = impute_allele!(m, &probs1);
                            let imp_a2 = impute_allele!(m, &probs2);
                            decisions.push(Stage2Decision::Impute {
                                marker: m,
                                a1: imp_a1,
                                a2: imp_a2,
                            });
                            continue;
                        }

                        // Skip if not unphased heterozygote
                        if !sp.is_unphased(m) {
                            continue;
                        }

                        // Skip homozygotes
                        if a1 == a2 {
                            continue;
                        }

                        let marker_maf = maf[m];
                        let is_rare_marker = marker_maf < rare_threshold;
                        let carriers = &carrier_haps[m];

                        if is_rare_marker && !carriers.is_empty() {
                            let carrier_set: std::collections::HashSet<u32> =
                                carriers.iter().copied().collect();
                            let score1 = carrier_score!(m, &probs1, &carrier_set);
                            let score2 = carrier_score!(m, &probs2, &carrier_set);

                            if carriers.len() == 1 || (score1 == 0.0 && score2 == 0.0) {
                                let stage1_idx = stage2_phaser.prev_stage1_marker[m];
                                let hap1_idx = (s * 2) as u32;
                                let hap2_idx = (s * 2 + 1) as u32;
                                let shorter_is_hap1 = if has_ref {
                                    let max1 = probs1
                                        .get(stage1_idx)
                                        .and_then(|v| v.iter().copied().reduce(f32::max))
                                        .unwrap_or(0.0);
                                    let max2 = probs2
                                        .get(stage1_idx)
                                        .and_then(|v| v.iter().copied().reduce(f32::max))
                                        .unwrap_or(0.0);
                                    max1 < max2
                                } else if let Some(phase_ibs) = phase_ibs.as_ref() {
                                    let span1 = phase_ibs.best_match_span(hap1_idx, stage1_idx);
                                    let span2 = phase_ibs.best_match_span(hap2_idx, stage1_idx);
                                    span1 < span2
                                } else {
                                    rng.random_bool(0.5)
                                };
                                let alt_on_hap1 = a1 > 0 && a1 != 255;
                                let alt_on_hap2 = a2 > 0 && a2 != 255;
                                if alt_on_hap1 ^ alt_on_hap2 {
                                    let should_swap = if alt_on_hap1 {
                                        !shorter_is_hap1
                                    } else {
                                        shorter_is_hap1
                                    };
                                    decisions.push(Stage2Decision::Phase {
                                        marker: m,
                                        should_swap,
                                        lr: 1.0,
                                    });
                                    continue;
                                }
                            }

                            let mut lr = if score2 > score1 {
                                (score2 / score1.max(1e-30)) as f32
                            } else {
                                (score1 / score2.max(1e-30)) as f32
                            };
                            let eps = 1e-6f64;
                            let s1 = (score1 as f64 + eps).max(eps);
                            let s2 = (score2 as f64 + eps).max(eps);
                            let denom = s1 + s2;
                            let mut p_swap = if denom > 0.0 {
                                (s2 / denom).clamp(0.0, 1.0)
                            } else {
                                0.5
                            };
                            let mut p_conf = (lr / (1.0 + lr)).clamp(0.0, 1.0);
                            p_conf = 0.5 + (p_conf - 0.5) * 0.5;
                            lr = (p_conf / (1.0 - p_conf)).max(1e-6);
                            let alpha = ((lr - 1.0) / (lr + 1.0)).clamp(0.0, 1.0) as f64;
                            p_swap = 0.5 * (1.0 - alpha) + alpha * p_swap;
                            let should_swap = rng.random_bool(p_swap as f64);
                            decisions.push(Stage2Decision::Phase {
                                marker: m,
                                should_swap,
                                lr,
                            });
                            continue;
                        }

                        // Fallback to interpolated allele probabilities
                        let mkr_a = stage2_phaser.prev_stage1_marker[m];
                        let state_haps_for_interp = get_haps!(mkr_a);
                        let al_probs1 = stage2_phaser.interpolated_allele_probs(
                            m,
                            &probs1,
                            state_haps_for_interp,
                            &get_allele,
                            a1,
                            a2,
                        );
                        let al_probs2 = stage2_phaser.interpolated_allele_probs(
                            m,
                            &probs2,
                            state_haps_for_interp,
                            &get_allele,
                            a1,
                            a2,
                        );

                        let p1 = al_probs1[0] * al_probs2[1];
                        let p2 = al_probs1[1] * al_probs2[0];

                        let mut lr = if p2 > p1 {
                            (p2 / p1.max(1e-30)) as f32
                        } else {
                            (p1 / p2.max(1e-30)) as f32
                        };
                        let eps = 1e-6f64;
                        let pp1 = (p1 as f64 + eps).max(eps);
                        let pp2 = (p2 as f64 + eps).max(eps);
                        let denom = pp1 + pp2;
                        let mut p_swap = if denom > 0.0 {
                            (pp2 / denom).clamp(0.0, 1.0)
                        } else {
                            0.5
                        };
                        let mut p_conf = (lr / (1.0 + lr)).clamp(0.0, 1.0);
                        p_conf = 0.5 + (p_conf - 0.5) * 0.5;
                        lr = (p_conf / (1.0 - p_conf)).max(1e-6);
                        let alpha = ((lr - 1.0) / (lr + 1.0)).clamp(0.0, 1.0) as f64;
                        p_swap = 0.5 * (1.0 - alpha) + alpha * p_swap;
                        let should_swap = rng.random_bool(p_swap as f64);
                        decisions.push(Stage2Decision::Phase {
                            marker: m,
                            should_swap,
                            lr,
                        });
                    }

                    (
                        decisions,
                        next_probs,
                        next_hap_priors,
                        next_prior_global_marker,
                    )
                })
                .collect::<Vec<_>>()
        }; // ref_geno borrow ends here

        let mut all_next_hap_priors = if next_overlap_start.is_some() {
            Some(Vec::with_capacity(n_haps))
        } else {
            None
        };
        let mut next_prior_global_marker: Option<usize> = None;

        // Apply phase changes and imputations to SamplePhase
        let mut total_switches = 0;
        let mut total_phased = 0;
        let mut total_imputed = 0;

        // Stage 2 runs after all iterations, so lr_threshold is typically 1.0
        // (all decisions pass). We still check for consistency with Stage 1.
        let lr_threshold = self.params.lr_threshold;

        for (s, (decisions, _, next_hap_priors, prior_marker)) in
            phase_results.into_iter().enumerate()
        {
            if let Some(all) = all_next_hap_priors.as_mut() {
                if let Some(priors_pair) = next_hap_priors {
                    all.push(priors_pair[0].clone());
                    all.push(priors_pair[1].clone());
                    if next_prior_global_marker.is_none() {
                        next_prior_global_marker = prior_marker;
                    }
                } else {
                    all.push(HaplotypePriors::empty());
                    all.push(HaplotypePriors::empty());
                }
            }

            let sp = &mut sample_phases[s];

            for decision in decisions {
                match decision {
                    Stage2Decision::Phase {
                        marker: m,
                        should_swap,
                        lr,
                    } => {
                        // Double-check still unphased (should always be true)
                        if !sp.is_unphased(m) {
                            continue;
                        }

                        let confident = lr >= lr_threshold;
                        if should_swap {
                            sp.swap_haps(m, m + 1);
                            if confident {
                                total_switches += 1;
                            }
                        }

                        sp.set_phase_confidence(m, lr / (1.0 + lr));

                        // Only mark as phased if likelihood ratio exceeds threshold
                        // (Stage 2 runs after iterations, so threshold is typically 1.0)
                        if confident {
                            sp.mark_phased(m);
                            total_phased += 1;
                        }
                    }
                    Stage2Decision::Impute { marker: m, a1, a2 } => {
                        // Set imputed alleles for missing marker
                        sp.set_imputed(m, a1, a2);
                        total_imputed += 1;
                    }
                }
            }
        }

        eprintln!(
            "Stage 2: Applied {} phase switches, {} markers phased, {} markers imputed (HMM interpolation)",
            total_switches, total_phased, total_imputed
        );

        let next_state_probs = None;

        let next_hap_priors = all_next_hap_priors.and_then(|priors| {
            if priors.len() == n_haps && priors.iter().any(|p| !p.is_empty()) {
                Some(priors)
            } else {
                None
            }
        });

        if next_state_probs.is_some() || next_hap_priors.is_some() {
            return Some(Stage2OverlapHandoff {
                state_probs: next_state_probs,
                hap_priors: next_hap_priors,
                prior_stage1_global_marker: next_prior_global_marker,
            });
        }

        None
    }
}

const PRIOR_EXPORT_MIN_PROB: f32 = 1e-5;

/// Project haplotype-identity priors onto the current window's local state set.
fn project_haplotype_priors_to_states(
    priors: &HaplotypePriors,
    state_haps: &[CombinedHapId],
) -> Vec<f32> {
    let n_states = state_haps.len();
    if n_states == 0 {
        return Vec::new();
    }

    let mut out = vec![0.0f32; n_states];
    let mut covered_mass = 0.0f32;

    for (k, &hap) in state_haps.iter().enumerate() {
        let p = priors.prob_of(GlobalHapId(hap.as_u32())).unwrap_or(0.0);
        out[k] = p;
        covered_mass += p;
    }

    // Any prior mass that is not represented in the new state set becomes
    // background uncertainty rather than being silently dropped.
    let leftover = (1.0 - covered_mass).max(0.0);
    if leftover > 0.0 {
        let background = leftover / n_states as f32;
        for p in &mut out {
            *p += background;
        }
    }

    let total: f32 = out.iter().sum();
    // Use a small epsilon
    if total > 1e-6 {
        for p in &mut out {
            *p /= total;
        }
    } else {
        let uniform = 1.0 / n_states as f32;
        out.fill(uniform);
    }

    out
}

/// Build haplotype priors from state posteriors.
fn build_haplotype_priors_from_state_probs(
    state_probs: &[f32],
    state_haps: &[CombinedHapId],
    min_prob: f32,
) -> HaplotypePriors {
    let mut mass_by_hap: std::collections::HashMap<u32, f32> =
        std::collections::HashMap::with_capacity(state_haps.len());

    for (k, &hap) in state_haps.iter().enumerate() {
        let p: f32 = state_probs.get(k).copied().unwrap_or(0.0);
        if p.is_finite() && p > 0.0 {
            *mass_by_hap.entry(hap.as_u32()).or_insert(0.0) += p;
        }
    }

    let mut entries: Vec<(u32, f32)> = mass_by_hap
        .into_iter()
        .filter(|&(_, p)| p >= min_prob)
        .collect();

    if entries.is_empty() {
        return HaplotypePriors::empty();
    }

    entries.sort_unstable_by_key(|(hap, _)| *hap);
    let mut hap_ids: Vec<GlobalHapId> = Vec::with_capacity(entries.len());
    let mut probs: Vec<f32> = Vec::with_capacity(entries.len());
    for (hap, p) in entries {
        hap_ids.push(GlobalHapId(hap));
        probs.push(p);
    }

    HaplotypePriors::new(hap_ids, probs)
}

/// Compute normalized posterior state probabilities from forward-backward arrays
fn compute_state_posteriors(
    fwd: &[f32],
    bwd: &[f32],
    n_markers: usize,
    n_states: usize,
) -> Vec<Vec<f32>> {
    let mut probs = vec![vec![0.0f32; n_states]; n_markers];

    for m in 0..n_markers {
        let row_start = m * n_states;
        let mut sum = 0.0f32;

        for (k, p) in probs[m].iter_mut().enumerate().take(n_states) {
            *p = fwd[row_start + k] * bwd[row_start + k];
            sum += *p;
        }

        // Normalize
        if sum > 0.0 {
            for p in probs[m].iter_mut().take(n_states) {
                *p /= sum;
            }
        }
    }

    probs
}

fn select_top_k_by_mass_two(
    probs1: &[Vec<f32>],
    probs2: &[Vec<f32>],
    n_states: usize,
    k: usize,
) -> Vec<usize> {
    let mut mass = vec![0.0f32; n_states];
    for row in probs1.iter() {
        for (i, &p) in row.iter().enumerate().take(n_states) {
            mass[i] += p;
        }
    }
    for row in probs2.iter() {
        for (i, &p) in row.iter().enumerate().take(n_states) {
            mass[i] += p;
        }
    }
    let mut idx: Vec<usize> = (0..n_states).collect();
    idx.sort_by(|&a, &b| mass[b].partial_cmp(&mass[a]).unwrap_or(std::cmp::Ordering::Equal));
    idx.truncate(k.min(n_states));
    idx
}


fn build_anchor_constraints(sample_phase: &SamplePhase) -> (Vec<u8>, Vec<u8>) {
    let n_markers = sample_phase.len();
    let mut hap1 = vec![255u8; n_markers];
    let mut hap2 = vec![255u8; n_markers];
    for m in 0..n_markers {
        if sample_phase.is_unphased(m) {
            continue;
        }
        let a1 = sample_phase.allele1(m);
        let a2 = sample_phase.allele2(m);
        if a1 == 255 || a2 == 255 {
            continue;
        }
        if a1 == a2 {
            continue;
        }
        hap1[m] = a1;
        hap2[m] = a2;
    }
    (hap1, hap2)
}

fn build_sample_confidence(target_gt: &GenotypeMatrix) -> Vec<Vec<f32>> {
    let n_samples = target_gt.n_samples();
    let n_markers = target_gt.n_markers();

    (0..n_samples)
        .map(|s| {
            (0..n_markers)
                .map(|m| {
                    let m_idx = MarkerIdx::new(m as u32);
                    if let Some(pl) = target_gt.sample_pl(m_idx, s) {
                        if pl.is_empty() {
                            return target_gt.sample_confidence_f32(m_idx, s);
                        }
                        let mut best = u16::MAX;
                        let mut second = u16::MAX;
                        for &v in pl {
                            if v < best {
                                second = best;
                                best = v;
                            } else if v < second {
                                second = v;
                            }
                        }
                        if second == u16::MAX {
                            return 1.0;
                        }
                        let delta = (second - best) as f32;
                        (delta / 60.0).clamp(0.0, 1.0)
                    } else {
                        target_gt.sample_confidence_f32(m_idx, s)
                    }
                })
                .collect()
        })
        .collect()
}

#[inline(always)]
fn emit_prob(ref_al: u8, targ_al: u8, conf: f32, p_no_err: f32, p_err: f32) -> f32 {
    let base = if ref_al == targ_al || ref_al == 255 || targ_al == 255 {
        p_no_err
    } else {
        p_err
    };
    base * conf + 0.5 * (1.0 - conf)
}

#[inline(always)]
fn emit_prob_hard(ref_al: u8, targ_al: u8, conf: f32, p_no_err: f32, p_err: f32, hard: bool) -> f32 {
    if hard && targ_al != 255 {
        if targ_al == INVALID_ALLELE {
            return 0.0;
        }
        return if ref_al == targ_al { p_no_err } else { 0.0 };
    }
    emit_prob(ref_al, targ_al, conf, p_no_err, p_err)
}

/// Emission mode for combined diploid genotype
#[derive(Clone, Copy)]
enum CombinedEmitMode {
    AllMissing,             // a1==255 && a2==255: always p_no_err
    Het { a1: u8, a2: u8 }, // a1!=a2: match if ref in {a1,a2,255}
    HomOrHemi { obs: u8 },  // hom or one missing: match if ref==obs or missing
}

#[inline]
fn classify_combined(a1: u8, a2: u8) -> CombinedEmitMode {
    if a1 == 255 && a2 == 255 {
        CombinedEmitMode::AllMissing
    } else if a1 != 255 && a2 != 255 && a1 != a2 {
        CombinedEmitMode::Het { a1, a2 }
    } else {
        let obs = if a1 != 255 { a1 } else { a2 };
        CombinedEmitMode::HomOrHemi { obs }
    }
}

/// Fast emit - assumes conf is already clamped to [0,1]
#[inline(always)]
fn emit_combined_fast(
    ref_al: u8,
    mode: CombinedEmitMode,
    conf: f32,
    p_no_err: f32,
    p_err: f32,
) -> f32 {
    let base = match mode {
        CombinedEmitMode::AllMissing => p_no_err,
        CombinedEmitMode::Het { a1, a2 } => {
            if ref_al == a1 || ref_al == a2 || ref_al == 255 {
                p_no_err
            } else {
                p_err
            }
        }
        CombinedEmitMode::HomOrHemi { obs } => {
            if ref_al == obs || ref_al == 255 || obs == 255 {
                p_no_err
            } else {
                p_err
            }
        }
    };
    base * conf + 0.5 * (1.0 - conf)
}

/// Compute the likelihood ratio for a phase decision with a single reference.
///
/// Used when only one reference haplotype path is available (e.g., in Gibbs sampling).
/// The LR is computed based on whether the reference supports the chosen allele.
#[inline]
fn compute_phase_lr_single(
    chosen_allele: u8,
    other_allele: u8,
    ref_allele: u8,
    conf: f32,
    p_no_err: f32,
    p_err: f32,
) -> f32 {
    if ref_allele == 255 {
        // Missing reference - no information
        return 1.0;
    }

    // Emission probability if chosen allele is correct
    let p_chosen = emit_prob(ref_allele, chosen_allele, conf, p_no_err, p_err);
    // Emission probability if other allele is correct
    let p_other = emit_prob(ref_allele, other_allele, conf, p_no_err, p_err);

    // LR = P(chosen) / P(other)
    if p_other < 1e-30 {
        if p_chosen < 1e-30 { 1.0 } else { 1e6 }
    } else {
        (p_chosen / p_other).min(1e6)
    }
}

#[derive(Clone, Copy, Debug)]
enum EmissionMode {
    Combined,
    Hap,
}

/// Compute haploid emission probability with heterozygote constraint.
///
/// At heterozygous sites, the target haplotype (H1) must emit the allele that,
/// when combined with the fixed haplotype (H2), produces the observed genotype.
/// This is the core of SHAPEIT5-style constrained Gibbs sampling.
///
/// # Arguments
/// * `ref_al` - Reference haplotype allele at this marker
/// * `geno_a1` - First allele of genotype
/// * `geno_a2` - Second allele of genotype
/// * `fixed_allele` - The allele of the fixed haplotype (H2), or 255 if homozygous
/// * `conf` - Genotype confidence (0..1)
/// * `p_no_err` - Probability of no error (e.g., 0.999)
/// * `p_err` - Probability of error (e.g., 0.001)
///
/// # Returns
/// Emission probability for this state
#[inline]
fn emit_haploid_constrained(
    ref_al: u8,
    geno_a1: u8,
    geno_a2: u8,
    fixed_allele: u8,
    conf: f32,
    p_no_err: f32,
    p_err: f32,
) -> f32 {
    // Missing data: return neutral emission (no information)
    if geno_a1 == 255 || geno_a2 == 255 {
        return 1.0;
    }

    // Unphased heterozygote with no fixed partner: allow either allele.
    if geno_a1 != geno_a2 && fixed_allele == 255 {
        let matches = ref_al == geno_a1 || ref_al == geno_a2 || ref_al == 255;
        let raw_emit = if matches { p_no_err } else { p_err };
        return conf * raw_emit + (1.0 - conf) * 0.5;
    }

    // At homozygous sites (fixed_allele == 255), both alleles are same
    // so H1 must emit geno_a1
    // At heterozygous sites, H1 must emit the allele opposite to fixed_allele
    let required_allele = if fixed_allele == 255 {
        geno_a1 // Homozygous: H1 must emit the homozygous allele
    } else if fixed_allele == geno_a1 {
        geno_a2 // H2 has a1, so H1 must have a2
    } else {
        geno_a1 // H2 has a2, so H1 must have a1
    };

    // Emission: does ref_al match the required allele?
    let matches = (ref_al == required_allele) as u8 as f32;
    let raw_emit = matches * p_no_err + (1.0 - matches) * p_err;

    // Blend with uniform based on confidence
    conf * raw_emit + (1.0 - conf) * 0.5
}

#[derive(Clone, Copy)]
struct HapEmissionInputs<'a> {
    /// Allele this haplotype is constrained to emit (non-PL emission path).
    target_constraint: &'a [u8],
    /// Allele carried by the partner haplotype (PL conditioning path).
    partner_allele: &'a [u8],
    /// Per-marker flag controlling combined (unconditioned) emissions.
    use_combined: &'a [bool],
    /// Per-marker flag enforcing hard match to target constraint.
    hard_match: &'a [bool],
}

#[inline]
fn compute_pl_allele_probs(
    pl: Option<&[u16]>,
    use_combined: bool,
    partner_allele: u8,
    allele_probs: &mut Vec<f32>,
) -> Option<usize> {
    let pl = pl.filter(|v| !v.is_empty())?;
    if use_combined {
        allele_probs_uncond_from_pl(pl, None, allele_probs)
    } else {
        allele_probs_cond_from_pl(pl, partner_allele, None, allele_probs)
            .or_else(|| allele_probs_uncond_from_pl(pl, None, allele_probs))
    }
}

#[inline]
fn refresh_path_ref_from_states(path_ref: &mut [u32], path_idx: &[u32], neighbors: &[u32]) {
    for (m, &state_u32) in path_idx.iter().enumerate() {
        let state = state_u32 as usize;
        if state < neighbors.len() {
            path_ref[m] = neighbors[state];
        }
    }
}

fn build_fwd_checkpoints<RefSpace>(
    checkpoints: &mut FwdCheckpoints,
    n_markers: usize,
    n_states: usize,
    p_recomb: &[f32],
    seq1: &[u8],
    seq2: &[u8],
    conf: &[f32],
    inputs: HapEmissionInputs<'_>,
    ref_provider: &mut RefAlleleProvider<'_, AnyMarkerSpace, RefSpace>,
    pl_provider: Option<&PlProvider>,
    allele_probs: &mut Vec<f32>,
    fwd: &mut [f32],
    fwd_prior: &mut [f32],
    ref_alleles: &mut [u8],
    p_no_err: f32,
    p_err: f32,
    mode: EmissionMode,
) {
    use wide::f32x8;

    if n_markers == 0 || n_states == 0 {
        return;
    }

    let init = 1.0f32 / n_states as f32;
    fwd[..n_states].fill(init);
    fwd_prior[..n_states].fill(0.0);
    let init = 1.0f32 / n_states as f32;
    let mut fwd_sum = 1.0f32;

    let mut next_block_idx = 0usize;
    let mut next_block_start = checkpoints
        .block_starts
        .get(next_block_idx)
        .copied()
        .unwrap_or(0);

    for m in 0..n_markers {
        if m > 0 {
            let r = p_recomb.get(m).copied().unwrap_or(0.0);
            let shift = r / n_states as f32;
            let scale = (1.0 - r) / fwd_sum.max(1e-30);

            // SIMD-optimized fwd_prior = scale * fwd + shift
            let shift_vec = f32x8::splat(shift);
            let scale_vec = f32x8::splat(scale);
            let mut k = 0;
            while k + 8 <= n_states {
                let fwd_arr: [f32; 8] = fwd[k..k + 8].try_into().unwrap();
                let fwd_chunk = f32x8::from(fwd_arr);
                let res = scale_vec * fwd_chunk + shift_vec;
                let res_arr: [f32; 8] = res.into();
                fwd_prior[k..k + 8].copy_from_slice(&res_arr);
                k += 8;
            }
            // Scalar tail
            for i in k..n_states {
                fwd_prior[i] = scale * fwd[i] + shift;
            }
        } else {
            fwd_prior.fill(init);
        }

        let a1 = seq1[m];
        let a2 = seq2[m];
        let conf_m = conf[m].clamp(0.0, 1.0);

        // Batch lookup: get all ref alleles for this marker at once
        ref_provider.fill_ref_alleles(m, ref_alleles);

        let use_combined = matches!(mode, EmissionMode::Combined) || inputs.use_combined[m];
        let hard_match = inputs.hard_match[m];

        let pl = pl_provider.and_then(|p| p.pl(m));
        let pl_n_alleles =
            compute_pl_allele_probs(pl, use_combined, inputs.partner_allele[m], allele_probs);
        let p_no_err_pl = 1.0 - p_err;
        let p_err_pl = if let Some(n) = pl_n_alleles {
            if n > 2 {
                p_err / (n as f32 - 1.0)
            } else {
                p_err
            }
        } else {
            p_err
        };

        // Compute fwd[k] = fwd_prior[k] * emit and accumulate sum
        // SIMD-optimized accumulation
        let mut sum_vec = f32x8::splat(0.0);
        let mut k = 0;

        if use_combined {
            let emit_mode = classify_combined(a1, a2);
            // Vectorized loop
            while k + 8 <= n_states {
                let prior_arr: [f32; 8] = fwd_prior[k..k + 8].try_into().unwrap();
                let prior_vec = f32x8::from(prior_arr);

                // Compute emissions for 8 states
                let emit_arr = if pl_n_alleles.is_some() {
                    [
                        emit_from_allele_probs(
                            ref_alleles[k],
                            &allele_probs,
                            p_no_err_pl,
                            p_err_pl,
                        ),
                        emit_from_allele_probs(
                            ref_alleles[k + 1],
                            &allele_probs,
                            p_no_err_pl,
                            p_err_pl,
                        ),
                        emit_from_allele_probs(
                            ref_alleles[k + 2],
                            &allele_probs,
                            p_no_err_pl,
                            p_err_pl,
                        ),
                        emit_from_allele_probs(
                            ref_alleles[k + 3],
                            &allele_probs,
                            p_no_err_pl,
                            p_err_pl,
                        ),
                        emit_from_allele_probs(
                            ref_alleles[k + 4],
                            &allele_probs,
                            p_no_err_pl,
                            p_err_pl,
                        ),
                        emit_from_allele_probs(
                            ref_alleles[k + 5],
                            &allele_probs,
                            p_no_err_pl,
                            p_err_pl,
                        ),
                        emit_from_allele_probs(
                            ref_alleles[k + 6],
                            &allele_probs,
                            p_no_err_pl,
                            p_err_pl,
                        ),
                        emit_from_allele_probs(
                            ref_alleles[k + 7],
                            &allele_probs,
                            p_no_err_pl,
                            p_err_pl,
                        ),
                    ]
                } else {
                    [
                        emit_combined_fast(ref_alleles[k], emit_mode, conf_m, p_no_err, p_err),
                        emit_combined_fast(ref_alleles[k + 1], emit_mode, conf_m, p_no_err, p_err),
                        emit_combined_fast(ref_alleles[k + 2], emit_mode, conf_m, p_no_err, p_err),
                        emit_combined_fast(ref_alleles[k + 3], emit_mode, conf_m, p_no_err, p_err),
                        emit_combined_fast(ref_alleles[k + 4], emit_mode, conf_m, p_no_err, p_err),
                        emit_combined_fast(ref_alleles[k + 5], emit_mode, conf_m, p_no_err, p_err),
                        emit_combined_fast(ref_alleles[k + 6], emit_mode, conf_m, p_no_err, p_err),
                        emit_combined_fast(ref_alleles[k + 7], emit_mode, conf_m, p_no_err, p_err),
                    ]
                };
                let emit_vec = f32x8::from(emit_arr);

                let res = prior_vec * emit_vec;
                let res_arr: [f32; 8] = res.into();
                fwd[k..k + 8].copy_from_slice(&res_arr);
                sum_vec += res;
                k += 8;
            }
            // Scalar tail
            fwd_sum = sum_vec.reduce_add();
            for i in k..n_states {
                let emit = if pl_n_alleles.is_some() {
                    emit_from_allele_probs(ref_alleles[i], &allele_probs, p_no_err_pl, p_err_pl)
                } else {
                    emit_combined_fast(ref_alleles[i], emit_mode, conf_m, p_no_err, p_err)
                };
                fwd[i] = fwd_prior[i] * emit;
                fwd_sum += fwd[i];
            }
        } else {
            let target_al = inputs.target_constraint[m];
            if hard_match {
                fwd_sum = 0.0;
                for i in 0..n_states {
                    let emit = emit_prob_hard(ref_alleles[i], target_al, conf_m, p_no_err, p_err, true);
                    fwd[i] = fwd_prior[i] * emit;
                    fwd_sum += fwd[i];
                }
            } else {
                // Vectorized loop
                while k + 8 <= n_states {
                    let prior_arr: [f32; 8] = fwd_prior[k..k + 8].try_into().unwrap();
                    let prior_vec = f32x8::from(prior_arr);

                    let emit_arr = if pl_n_alleles.is_some() {
                        [
                            emit_from_allele_probs(
                                ref_alleles[k],
                                &allele_probs,
                                p_no_err_pl,
                                p_err_pl,
                            ),
                            emit_from_allele_probs(
                                ref_alleles[k + 1],
                                &allele_probs,
                                p_no_err_pl,
                                p_err_pl,
                            ),
                            emit_from_allele_probs(
                                ref_alleles[k + 2],
                                &allele_probs,
                                p_no_err_pl,
                                p_err_pl,
                            ),
                            emit_from_allele_probs(
                                ref_alleles[k + 3],
                                &allele_probs,
                                p_no_err_pl,
                                p_err_pl,
                            ),
                            emit_from_allele_probs(
                                ref_alleles[k + 4],
                                &allele_probs,
                                p_no_err_pl,
                                p_err_pl,
                            ),
                            emit_from_allele_probs(
                                ref_alleles[k + 5],
                                &allele_probs,
                                p_no_err_pl,
                                p_err_pl,
                            ),
                            emit_from_allele_probs(
                                ref_alleles[k + 6],
                                &allele_probs,
                                p_no_err_pl,
                                p_err_pl,
                            ),
                            emit_from_allele_probs(
                                ref_alleles[k + 7],
                                &allele_probs,
                                p_no_err_pl,
                                p_err_pl,
                            ),
                        ]
                    } else {
                        [
                            emit_prob(ref_alleles[k], target_al, conf_m, p_no_err, p_err),
                            emit_prob(ref_alleles[k + 1], target_al, conf_m, p_no_err, p_err),
                            emit_prob(ref_alleles[k + 2], target_al, conf_m, p_no_err, p_err),
                            emit_prob(ref_alleles[k + 3], target_al, conf_m, p_no_err, p_err),
                            emit_prob(ref_alleles[k + 4], target_al, conf_m, p_no_err, p_err),
                            emit_prob(ref_alleles[k + 5], target_al, conf_m, p_no_err, p_err),
                            emit_prob(ref_alleles[k + 6], target_al, conf_m, p_no_err, p_err),
                            emit_prob(ref_alleles[k + 7], target_al, conf_m, p_no_err, p_err),
                        ]
                    };
                    let emit_vec = f32x8::from(emit_arr);

                    let res = prior_vec * emit_vec;
                    let res_arr: [f32; 8] = res.into();
                    fwd[k..k + 8].copy_from_slice(&res_arr);
                    sum_vec += res;
                    k += 8;
                }
                // Scalar tail
                fwd_sum = sum_vec.reduce_add();
                for i in k..n_states {
                    let emit = if pl_n_alleles.is_some() {
                        emit_from_allele_probs(ref_alleles[i], &allele_probs, p_no_err_pl, p_err_pl)
                    } else {
                        emit_prob(ref_alleles[i], target_al, conf_m, p_no_err, p_err)
                    };
                    fwd[i] = fwd_prior[i] * emit;
                    fwd_sum += fwd[i];
                }
            }
        }
        fwd_sum = fwd_sum.max(1e-30);

        if m == next_block_start {
            let dst = checkpoints.block_slice_mut(next_block_idx);
            dst.copy_from_slice(&fwd);
            next_block_idx += 1;
            next_block_start = checkpoints
                .block_starts
                .get(next_block_idx)
                .copied()
                .unwrap_or(usize::MAX);
        }
    }
}

fn sample_from_weights(weights: &[f32], rng: &mut rand::rngs::SmallRng) -> usize {
    let total: f32 = weights.iter().sum();
    if total <= 0.0 {
        let idx = rng.random::<u32>() as usize % weights.len().max(1);
        return idx.min(weights.len().saturating_sub(1));
    }

    let mut threshold = rng.random::<f32>() * total;
    for (i, w) in weights.iter().enumerate() {
        threshold -= *w;
        if threshold <= 0.0 {
            return i;
        }
    }
    weights.len().saturating_sub(1)
}

fn sample_path_from_checkpoints<RefSpace>(
    path: &mut [u32],
    checkpoints: &FwdCheckpoints,
    n_markers: usize,
    n_states: usize,
    p_recomb: &[f32],
    seq1: &[u8],
    seq2: &[u8],
    conf: &[f32],
    inputs: HapEmissionInputs<'_>,
    ref_provider: &mut RefAlleleProvider<'_, AnyMarkerSpace, RefSpace>,
    pl_provider: Option<&PlProvider>,
    p_no_err: f32,
    p_err: f32,
    rng: &mut rand::rngs::SmallRng,
    fwd_block: &mut [f32],
    weights: &mut [f32],
    ref_alleles: &mut [u8],
    allele_probs: &mut Vec<f32>,
    mode: EmissionMode,
) {
    use wide::f32x8;

    if n_markers == 0 || n_states == 0 {
        return;
    }

    let starts = checkpoints.block_starts.as_ref();
    let n_blocks = starts.len().max(1);

    let weights = &mut weights[..n_states];
    let ref_alleles = &mut ref_alleles[..n_states];

    for block_idx in (0..n_blocks).rev() {
        let start = starts.get(block_idx).copied().unwrap_or(0).min(n_markers);
        let end = starts
            .get(block_idx + 1)
            .copied()
            .unwrap_or(n_markers)
            .min(n_markers);
        if end <= start {
            continue;
        }
        let block_len = end - start;
        let row_stride = n_states;
        let buf_len = block_len * row_stride;
        let fwd_buf = &mut fwd_block[..buf_len];

        // Seed forward values at block start from checkpoint.
        let seed = checkpoints.block_slice(block_idx);
        fwd_buf[..row_stride].copy_from_slice(seed);
        let mut prev_sum: f32 = seed.iter().sum();
        prev_sum = prev_sum.max(1e-30);

        for m in (start + 1)..end {
            let r = p_recomb.get(m).copied().unwrap_or(0.0);
            let shift = r / n_states as f32;
            let scale = (1.0 - r) / prev_sum;

            let a1 = seq1[m];
            let a2 = seq2[m];
            let conf_m = conf[m];
            let row_idx = (m - start) * row_stride;
            let (prev_part, curr_part) = fwd_buf.split_at_mut(row_idx);
            let prev_row = &prev_part[row_idx - row_stride..];

            // Batch lookup ref alleles
            ref_provider.fill_ref_alleles(m, ref_alleles);

            // SIMD-optimized forward update
            let shift_vec = f32x8::splat(shift);
            let scale_vec = f32x8::splat(scale);
            let mut sum_vec = f32x8::splat(0.0);
            let mut k = 0;

            let use_combined = matches!(mode, EmissionMode::Combined) || inputs.use_combined[m];
            let hard_match = inputs.hard_match[m];

            let pl = pl_provider.and_then(|p| p.pl(m));
            let pl_n_alleles =
                compute_pl_allele_probs(pl, use_combined, inputs.partner_allele[m], allele_probs);
            let p_no_err_pl = 1.0 - p_err;
            let p_err_pl = if let Some(n) = pl_n_alleles {
                if n > 2 {
                    p_err / (n as f32 - 1.0)
                } else {
                    p_err
                }
            } else {
                p_err
            };

            if use_combined {
                let emit_mode = classify_combined(a1, a2);
                while k + 8 <= n_states {
                    let prev_arr: [f32; 8] = prev_row[k..k + 8].try_into().unwrap();
                    let prev_vec = f32x8::from(prev_arr);
                    let prior_vec = scale_vec * prev_vec + shift_vec;

                    let emit_arr = if pl_n_alleles.is_some() {
                        [
                            emit_from_allele_probs(
                                ref_alleles[k],
                                &allele_probs,
                                p_no_err_pl,
                                p_err_pl,
                            ),
                            emit_from_allele_probs(
                                ref_alleles[k + 1],
                                &allele_probs,
                                p_no_err_pl,
                                p_err_pl,
                            ),
                            emit_from_allele_probs(
                                ref_alleles[k + 2],
                                &allele_probs,
                                p_no_err_pl,
                                p_err_pl,
                            ),
                            emit_from_allele_probs(
                                ref_alleles[k + 3],
                                &allele_probs,
                                p_no_err_pl,
                                p_err_pl,
                            ),
                            emit_from_allele_probs(
                                ref_alleles[k + 4],
                                &allele_probs,
                                p_no_err_pl,
                                p_err_pl,
                            ),
                            emit_from_allele_probs(
                                ref_alleles[k + 5],
                                &allele_probs,
                                p_no_err_pl,
                                p_err_pl,
                            ),
                            emit_from_allele_probs(
                                ref_alleles[k + 6],
                                &allele_probs,
                                p_no_err_pl,
                                p_err_pl,
                            ),
                            emit_from_allele_probs(
                                ref_alleles[k + 7],
                                &allele_probs,
                                p_no_err_pl,
                                p_err_pl,
                            ),
                        ]
                    } else {
                        [
                            emit_combined_fast(ref_alleles[k], emit_mode, conf_m, p_no_err, p_err),
                            emit_combined_fast(
                                ref_alleles[k + 1],
                                emit_mode,
                                conf_m,
                                p_no_err,
                                p_err,
                            ),
                            emit_combined_fast(
                                ref_alleles[k + 2],
                                emit_mode,
                                conf_m,
                                p_no_err,
                                p_err,
                            ),
                            emit_combined_fast(
                                ref_alleles[k + 3],
                                emit_mode,
                                conf_m,
                                p_no_err,
                                p_err,
                            ),
                            emit_combined_fast(
                                ref_alleles[k + 4],
                                emit_mode,
                                conf_m,
                                p_no_err,
                                p_err,
                            ),
                            emit_combined_fast(
                                ref_alleles[k + 5],
                                emit_mode,
                                conf_m,
                                p_no_err,
                                p_err,
                            ),
                            emit_combined_fast(
                                ref_alleles[k + 6],
                                emit_mode,
                                conf_m,
                                p_no_err,
                                p_err,
                            ),
                            emit_combined_fast(
                                ref_alleles[k + 7],
                                emit_mode,
                                conf_m,
                                p_no_err,
                                p_err,
                            ),
                        ]
                    };
                    let emit_vec = f32x8::from(emit_arr);

                    let res = prior_vec * emit_vec;
                    let res_arr: [f32; 8] = res.into();
                    curr_part[k..k + 8].copy_from_slice(&res_arr);
                    sum_vec += res;
                    k += 8;
                }
                prev_sum = sum_vec.reduce_add();
                for i in k..n_states {
                    let prior = scale * prev_row[i] + shift;
                    let emit = if pl_n_alleles.is_some() {
                        emit_from_allele_probs(ref_alleles[i], &allele_probs, p_no_err_pl, p_err_pl)
                    } else {
                        emit_combined_fast(ref_alleles[i], emit_mode, conf_m, p_no_err, p_err)
                    };
                    curr_part[i] = prior * emit;
                    prev_sum += curr_part[i];
                }
            } else {
                let target_al = inputs.target_constraint[m];
                if hard_match {
                    prev_sum = 0.0;
                    for i in 0..n_states {
                        let prior = scale * prev_row[i] + shift;
                        let emit = emit_prob_hard(ref_alleles[i], target_al, conf_m, p_no_err, p_err, true);
                        curr_part[i] = prior * emit;
                        prev_sum += curr_part[i];
                    }
                } else {
                    while k + 8 <= n_states {
                        let prev_arr: [f32; 8] = prev_row[k..k + 8].try_into().unwrap();
                        let prev_vec = f32x8::from(prev_arr);
                        let prior_vec = scale_vec * prev_vec + shift_vec;

                        let emit_arr = if pl_n_alleles.is_some() {
                            [
                                emit_from_allele_probs(
                                    ref_alleles[k],
                                    &allele_probs,
                                    p_no_err_pl,
                                    p_err_pl,
                                ),
                                emit_from_allele_probs(
                                    ref_alleles[k + 1],
                                    &allele_probs,
                                    p_no_err_pl,
                                    p_err_pl,
                                ),
                                emit_from_allele_probs(
                                    ref_alleles[k + 2],
                                    &allele_probs,
                                    p_no_err_pl,
                                    p_err_pl,
                                ),
                                emit_from_allele_probs(
                                    ref_alleles[k + 3],
                                    &allele_probs,
                                    p_no_err_pl,
                                    p_err_pl,
                                ),
                                emit_from_allele_probs(
                                    ref_alleles[k + 4],
                                    &allele_probs,
                                    p_no_err_pl,
                                    p_err_pl,
                                ),
                                emit_from_allele_probs(
                                    ref_alleles[k + 5],
                                    &allele_probs,
                                    p_no_err_pl,
                                    p_err_pl,
                                ),
                                emit_from_allele_probs(
                                    ref_alleles[k + 6],
                                    &allele_probs,
                                    p_no_err_pl,
                                    p_err_pl,
                                ),
                                emit_from_allele_probs(
                                    ref_alleles[k + 7],
                                    &allele_probs,
                                    p_no_err_pl,
                                    p_err_pl,
                                ),
                            ]
                        } else {
                            [
                                emit_prob(ref_alleles[k], target_al, conf_m, p_no_err, p_err),
                                emit_prob(
                                    ref_alleles[k + 1],
                                    target_al,
                                    conf_m,
                                    p_no_err,
                                    p_err,
                                ),
                                emit_prob(
                                    ref_alleles[k + 2],
                                    target_al,
                                    conf_m,
                                    p_no_err,
                                    p_err,
                                ),
                                emit_prob(
                                    ref_alleles[k + 3],
                                    target_al,
                                    conf_m,
                                    p_no_err,
                                    p_err,
                                ),
                                emit_prob(
                                    ref_alleles[k + 4],
                                    target_al,
                                    conf_m,
                                    p_no_err,
                                    p_err,
                                ),
                                emit_prob(
                                    ref_alleles[k + 5],
                                    target_al,
                                    conf_m,
                                    p_no_err,
                                    p_err,
                                ),
                                emit_prob(
                                    ref_alleles[k + 6],
                                    target_al,
                                    conf_m,
                                    p_no_err,
                                    p_err,
                                ),
                                emit_prob(
                                    ref_alleles[k + 7],
                                    target_al,
                                    conf_m,
                                    p_no_err,
                                    p_err,
                                ),
                            ]
                        };
                        let emit_vec = f32x8::from(emit_arr);

                        let res = prior_vec * emit_vec;
                        let res_arr: [f32; 8] = res.into();
                        curr_part[k..k + 8].copy_from_slice(&res_arr);
                        sum_vec += res;
                        k += 8;
                    }
                    prev_sum = sum_vec.reduce_add();
                    for i in k..n_states {
                        let prior = scale * prev_row[i] + shift;
                        let emit = if pl_n_alleles.is_some() {
                            emit_from_allele_probs(
                                ref_alleles[i],
                                &allele_probs,
                                p_no_err_pl,
                                p_err_pl,
                            )
                        } else {
                            emit_prob(ref_alleles[i], target_al, conf_m, p_no_err, p_err)
                        };
                        curr_part[i] = prior * emit;
                        prev_sum += curr_part[i];
                    }
                }
            }
            prev_sum = prev_sum.max(1e-30);
        }

        // Sample the last marker in this block conditional on the first state in the next block.
        // This is the explicit boundary projection that was missing from the previous checkpoint sampler.
        let next_state = if end < n_markers {
            Some(path[end] as usize)
        } else {
            None
        };
        let last_row = &fwd_buf[(block_len - 1) * row_stride..block_len * row_stride];
        if let Some(ns) = next_state {
            let r = p_recomb.get(end).copied().unwrap_or(0.0);
            let shift = r / n_states as f32;
            let stay = (1.0 - r) + shift;
            for i in 0..n_states {
                let t = if i == ns { stay } else { shift };
                weights[i] = last_row[i] * t;
            }
            let sampled = sample_from_weights(&weights, rng);
            path[end - 1] = sampled as u32;
        } else {
            let sampled = sample_from_weights(last_row, rng);
            path[end - 1] = sampled as u32;
        }

        for m in (start + 1..end).rev() {
            let next_state = path[m] as usize;
            let r = p_recomb.get(m).copied().unwrap_or(0.0);
            let shift = r / n_states as f32;
            // Li-Stephens: P(stay) = (1-r) + r/K, P(switch) = r/K
            let stay = (1.0 - r) + shift;
            let row_idx = (m - 1 - start) * row_stride;
            let prev_row = &fwd_buf[row_idx..row_idx + row_stride];

            // SIMD-optimized weight computation
            let shift_vec = f32x8::splat(shift);
            let mut k = 0;
            while k + 8 <= n_states {
                let prev_arr: [f32; 8] = prev_row[k..k + 8].try_into().unwrap();
                let prev_vec = f32x8::from(prev_arr);
                // Most states get shift transition
                let res = prev_vec * shift_vec;
                let res_arr: [f32; 8] = res.into();
                weights[k..k + 8].copy_from_slice(&res_arr);
                k += 8;
            }
            for i in k..n_states {
                weights[i] = prev_row[i] * shift;
            }
            // Fix up the stay state
            if next_state < n_states {
                weights[next_state] = prev_row[next_state] * stay;
            }

            let sampled = sample_from_weights(&weights, rng);
            path[m - 1] = sampled as u32;
        }
    }
}

/// Forward-Filtering Backward-Sampling for haploid HMM with constraint.
///
/// This is the core of SHAPEIT5-style Gibbs sampling. It samples a haplotype
/// path through K reference states, with emissions constrained at heterozygous
/// sites to be opposite of the fixed other haplotype.
///
/// Returns the sampled state path in `path`.
fn ffbs_haploid_constrained(
    path: &mut [u32],
    n_markers: usize,
    n_states: usize,
    p_recomb: &[f32],
    geno_a1: &[u8],
    geno_a2: &[u8],
    conf: &[f32],
    fixed_allele: &[u8], // Allele assigned to OTHER haplotype (255 = no constraint)
    neighbors: &[u32],   // Selected neighbor haplotype indices
    phase_ibs: &BidirectionalPhaseIbs,
    p_no_err: f32,
    p_err: f32,
    rng: &mut rand::rngs::SmallRng,
    workspace: &mut crate::utils::workspace::ThreadWorkspace,
) {
    use wide::f32x8;

    if n_markers == 0 || n_states == 0 || neighbors.is_empty() {
        return;
    }

    let actual_n_states = neighbors.len().min(n_states);

    workspace.ensure_ffbs(n_markers, actual_n_states);
    let fwd_curr = &mut workspace.ffbs_fwd_curr;
    let fwd_prev = &mut workspace.ffbs_fwd_prev;
    let fwd_at_marker = &mut workspace.ffbs_fwd_at_marker;
    let weights = &mut workspace.ffbs_weights;
    fwd_curr[..actual_n_states].fill(0.0);
    fwd_prev[..actual_n_states].fill(0.0);

    // Initialize at marker 0
    let init = 1.0f32 / actual_n_states as f32;
    for k in 0..actual_n_states {
        let ref_al = phase_ibs.allele(0, neighbors[k]);
        let emit = emit_haploid_constrained(
            ref_al,
            geno_a1[0],
            geno_a2[0],
            fixed_allele[0],
            conf[0],
            p_no_err,
            p_err,
        );
        fwd_curr[k] = init * emit;
    }
    let mut fwd_sum: f32 = fwd_curr.iter().sum();
    fwd_sum = fwd_sum.max(1e-30);
    fwd_at_marker[0..actual_n_states].copy_from_slice(&fwd_curr[..actual_n_states]);

    // Forward pass
    for m in 1..n_markers {
        std::mem::swap(fwd_prev, fwd_curr);

        let r = p_recomb.get(m).copied().unwrap_or(0.0);
        let shift = r / actual_n_states as f32;
        let scale = (1.0 - r) / fwd_sum;

        // SIMD-optimized transition + emission
        let shift_vec = f32x8::splat(shift);
        let scale_vec = f32x8::splat(scale);
        let mut sum_vec = f32x8::splat(0.0);
        let mut k = 0;

        while k + 8 <= actual_n_states {
            let prev_arr: [f32; 8] = fwd_prev[k..k + 8].try_into().unwrap();
            let prev_vec = f32x8::from(prev_arr);
            let prior_vec = scale_vec * prev_vec + shift_vec;

            // Compute emissions
            let emit_arr = [
                emit_haploid_constrained(
                    phase_ibs.allele(m, neighbors[k]),
                    geno_a1[m],
                    geno_a2[m],
                    fixed_allele[m],
                    conf[m],
                    p_no_err,
                    p_err,
                ),
                emit_haploid_constrained(
                    phase_ibs.allele(m, neighbors[k + 1]),
                    geno_a1[m],
                    geno_a2[m],
                    fixed_allele[m],
                    conf[m],
                    p_no_err,
                    p_err,
                ),
                emit_haploid_constrained(
                    phase_ibs.allele(m, neighbors[k + 2]),
                    geno_a1[m],
                    geno_a2[m],
                    fixed_allele[m],
                    conf[m],
                    p_no_err,
                    p_err,
                ),
                emit_haploid_constrained(
                    phase_ibs.allele(m, neighbors[k + 3]),
                    geno_a1[m],
                    geno_a2[m],
                    fixed_allele[m],
                    conf[m],
                    p_no_err,
                    p_err,
                ),
                emit_haploid_constrained(
                    phase_ibs.allele(m, neighbors[k + 4]),
                    geno_a1[m],
                    geno_a2[m],
                    fixed_allele[m],
                    conf[m],
                    p_no_err,
                    p_err,
                ),
                emit_haploid_constrained(
                    phase_ibs.allele(m, neighbors[k + 5]),
                    geno_a1[m],
                    geno_a2[m],
                    fixed_allele[m],
                    conf[m],
                    p_no_err,
                    p_err,
                ),
                emit_haploid_constrained(
                    phase_ibs.allele(m, neighbors[k + 6]),
                    geno_a1[m],
                    geno_a2[m],
                    fixed_allele[m],
                    conf[m],
                    p_no_err,
                    p_err,
                ),
                emit_haploid_constrained(
                    phase_ibs.allele(m, neighbors[k + 7]),
                    geno_a1[m],
                    geno_a2[m],
                    fixed_allele[m],
                    conf[m],
                    p_no_err,
                    p_err,
                ),
            ];
            let emit_vec = f32x8::from(emit_arr);

            let res = prior_vec * emit_vec;
            let res_arr: [f32; 8] = res.into();
            fwd_curr[k..k + 8].copy_from_slice(&res_arr);
            sum_vec += res;
            k += 8;
        }

        // Scalar tail
        fwd_sum = sum_vec.reduce_add();
        for i in k..actual_n_states {
            let prior = scale * fwd_prev[i] + shift;
            let emit = emit_haploid_constrained(
                phase_ibs.allele(m, neighbors[i]),
                geno_a1[m],
                geno_a2[m],
                fixed_allele[m],
                conf[m],
                p_no_err,
                p_err,
            );
            fwd_curr[i] = prior * emit;
            fwd_sum += fwd_curr[i];
        }
        fwd_sum = fwd_sum.max(1e-30);

        let start = m * actual_n_states;
        fwd_at_marker[start..start + actual_n_states]
            .copy_from_slice(&fwd_curr[..actual_n_states]);
    }

    // Backward sampling
    let last_start = (n_markers - 1) * actual_n_states;
    let last_fwd = &fwd_at_marker[last_start..last_start + actual_n_states];
    path[n_markers - 1] = sample_from_weights(last_fwd, rng) as u32;
    weights[..actual_n_states].fill(0.0);
    for m in (1..n_markers).rev() {
        let next_state = path[m] as usize;
        let r = p_recomb.get(m).copied().unwrap_or(0.0);
        let shift = r / actual_n_states as f32;
        let stay = (1.0 - r) + shift;

        let prev_start = (m - 1) * actual_n_states;
        let prev_fwd = &fwd_at_marker[prev_start..prev_start + actual_n_states];

        for k in 0..actual_n_states {
            weights[k] = prev_fwd[k] * shift;
        }
        if next_state < actual_n_states {
            weights[next_state] = prev_fwd[next_state] * stay;
        }

        path[m - 1] = sample_from_weights(&weights[..actual_n_states], rng) as u32;
    }
}

/// Dynamic MCMC phasing using SHAPEIT5-style Gibbs sampling.
///
/// This implements the correct MCMC approach with implicit anchoring:
/// 1. At each MCMC step, select K neighbors by threading current H1/H2 through PBWT
/// 2. Sample H1 | (G, H2_fixed) using haploid constrained HMM
/// 3. Sample H2 | (G, H1_new) using haploid constrained HMM
/// 4. Repeat for n_steps
///
/// The "implicit anchoring" comes from state selection being biased toward
/// haplotypes that match the current phase estimate via the "Latent State" approach:
/// neighbors are found by looking up the position of the PREVIOUSLY SAMPLED reference
/// state in the PBWT, giving O(1) lookup and preserving phase inertia.
fn sample_dynamic_mcmc(
    n_markers: usize,
    n_states: usize,
    p_recomb: &[f32],
    seq1: &[u8],
    seq2: &[u8],
    conf: &[f32],
    phase_ibs: &BidirectionalPhaseIbs,
    ibs2: &Ibs2,
    sample_idx: u32,
    het_positions: &[usize],
    seed: u64,
    n_mcmc_steps: usize,
    p_no_err: f32,
    p_err: f32,
    initial_paths: Option<&MosaicPaths>,
    anchor_hap1: Option<&[u8]>,
    anchor_hap2: Option<&[u8]>,
    workspace: &mut crate::utils::workspace::ThreadWorkspace,
) -> (Vec<u8>, Vec<f32>, Vec<f32>, MosaicPaths) {
    use rand::SeedableRng;

    if het_positions.is_empty() || n_markers == 0 || n_states == 0 {
        return (
            Vec::new(),
            Vec::new(),
            Vec::new(),
            MosaicPaths {
                path1: Vec::new(),
                path2: Vec::new(),
            },
        );
    }

    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    let hap1_idx = sample_idx * 2;
    let anchor_h1 = anchor_hap1.unwrap_or(&[]);
    let anchor_h2 = anchor_hap2.unwrap_or(&[]);

    // Initialize H1, H2 alleles from genotype (random phase at hets)
    let mut h1_alleles = vec![0u8; n_markers];
    let mut h2_alleles = vec![0u8; n_markers];
    for m in 0..n_markers {
        let a1 = seq1[m];
        let a2 = seq2[m];
        let anchor_a1 = anchor_h1.get(m).copied().unwrap_or(255);
        let anchor_a2 = anchor_h2.get(m).copied().unwrap_or(255);
        if anchor_a1 != 255 || anchor_a2 != 255 {
            h1_alleles[m] = anchor_a1;
            h2_alleles[m] = anchor_a2;
            continue;
        }
        if a1 == 255 && a2 == 255 {
            h1_alleles[m] = 255;
            h2_alleles[m] = 255;
        } else if a1 == a2 {
            h1_alleles[m] = a1;
            h2_alleles[m] = a1;
        } else {
            // Het: random initial phase
            if rng.random::<bool>() {
                h1_alleles[m] = a1;
                h2_alleles[m] = a2;
            } else {
                h1_alleles[m] = a2;
                h2_alleles[m] = a1;
            }
        }
    }

    // Seed alleles from initial paths if available (from heuristic)
    // This ensures MCMC starts in a high-probability region rather than drifting
    // from a random start.
    if let Some(paths) = initial_paths {
        if paths.path1.len() == n_markers && paths.path2.len() == n_markers {
            for m in 0..n_markers {
                let a1 = seq1[m];
                let a2 = seq2[m];
                if a1 == 255 || a2 == 255 || a1 == a2 {
                    continue;
                }

                let h1_idx = paths.path1[m] as usize;
                let h2_idx = paths.path2[m] as usize;

                if h1_idx < phase_ibs.n_haps() && h2_idx < phase_ibs.n_haps() {
                    let ref1 = phase_ibs.allele(m, h1_idx as u32);
                    let ref2 = phase_ibs.allele(m, h2_idx as u32);

                    let matches_orient1 = ref1 == a1 && ref2 == a2;
                    let matches_orient2 = ref1 == a2 && ref2 == a1;

                    if matches_orient1 && !matches_orient2 {
                        h1_alleles[m] = a1;
                        h2_alleles[m] = a2;
                    } else if matches_orient2 && !matches_orient1 {
                        h1_alleles[m] = a2;
                        h2_alleles[m] = a1;
                    }
                }
            }
        }
    }

    // Initialize path with starting states from standard neighbor finding
    // This gives the first iteration something to work with
    let initial_neighbors = phase_ibs.find_neighbors(hap1_idx, n_markers / 2, ibs2, n_states);
    if initial_neighbors.is_empty() {
        return (
            Vec::new(),
            Vec::new(),
            Vec::new(),
            MosaicPaths {
                path1: Vec::new(),
                path2: Vec::new(),
            },
        );
    }

    // Separate paths for H1 and H2 to avoid cross-talk in Gibbs sampling
    // Store reference hap IDs (for persistence) and local state indices (per step)
    let mut path1_ref = vec![0u32; n_markers];
    let mut path2_ref = vec![0u32; n_markers];
    let mut path1_idx = vec![0u32; n_markers];
    let mut path2_idx = vec![0u32; n_markers];
    let mut fixed_allele = vec![255u8; n_markers];

    // Current set of neighbors (reused across markers within an MCMC step)
    let mut neighbors = initial_neighbors;
    let n_haps = phase_ibs.n_haps() as u32;

    if let Some(paths) = initial_paths {
        if paths.path1.len() == n_markers && paths.path2.len() == n_markers {
            path1_ref.copy_from_slice(&paths.path1);
            path2_ref.copy_from_slice(&paths.path2);
        }
    } else if let Some(&seed_hap) = neighbors.first() {
        path1_ref.fill(seed_hap);
        path2_ref.fill(seed_hap);
    }

    fn mix_neighbors(
        neighbors: &mut Vec<u32>,
        n_states: usize,
        n_haps: u32,
        hap1_idx: u32,
        rng: &mut impl rand::Rng,
    ) {
        // If there are no other haplotypes to sample from, fall back to self so we can proceed.
        // This avoids an infinite loop for single-sample or haploid inputs.
        if n_haps <= 2 {
            neighbors.clear();
            let self_hap = hap1_idx.min(n_haps.saturating_sub(1));
            neighbors.push(self_hap);
            return;
        }

        let target = n_states.min((n_haps.saturating_sub(2)) as usize).max(1);
        if neighbors.len() > target {
            neighbors.truncate(target);
        }

        while neighbors.len() < target {
            let h = rng.random_range(0..n_haps);
            if h == hap1_idx || h == hap1_idx + 1 {
                continue;
            }
            if !neighbors.contains(&h) {
                neighbors.push(h);
            }
        }

        if neighbors.is_empty() {
            return;
        }

        let mix_count = (target / 10).max(4).min(target);
        for _ in 0..mix_count {
            let h = rng.random_range(0..n_haps);
            if h == hap1_idx || h == hap1_idx + 1 {
                continue;
            }
            if neighbors.contains(&h) {
                continue;
            }
            let replace_idx = rng.random_range(0..neighbors.len());
            neighbors[replace_idx] = h;
        }
    }

    mix_neighbors(&mut neighbors, n_states, n_haps, hap1_idx, &mut rng);

    let collect_dynamic_neighbors = |path_ref: &[u32], sample_idx: u32| -> Vec<u32> {
        let stride = (n_markers / 8).max(1);
        // Prefer informative anchors: within each stride window, choose the best marker.
        let anchor_score = |m: usize| -> f32 {
            let a1 = seq1[m];
            let a2 = seq2[m];
            let non_missing = a1 != 255 && a2 != 255;
            let is_het = non_missing && a1 != a2;
            let conf_score = conf[m].clamp(0.0, 1.0);
            // Non-missing anchors dominate, then confidence, then a small het bonus.
            (if non_missing { 4.0 } else { 0.0 }) + conf_score + if is_het { 0.25 } else { 0.0 }
        };

        let mut anchors: Vec<usize> = Vec::new();
        let mut start = 0usize;
        while start < n_markers {
            let end = (start + stride).min(n_markers);
            let mut best_m = start;
            let mut best_score = f32::NEG_INFINITY;
            for m in start..end {
                let score = anchor_score(m);
                if score > best_score {
                    best_score = score;
                    best_m = m;
                }
            }
            anchors.push(best_m);
            start = end;
        }
        if anchors.last().copied() != Some(n_markers.saturating_sub(1)) {
            anchors.push(n_markers.saturating_sub(1));
        }
        let mut seen = std::collections::HashSet::new();
        let mut out: Vec<u32> = Vec::new();

        for &m in &anchors {
            let ref_hap = path_ref.get(m).copied().unwrap_or(0);
            if (ref_hap as usize) < phase_ibs.n_haps() {
                let mut local = phase_ibs.find_neighbors_of_state(ref_hap, m, sample_idx, n_states);
                local.push(ref_hap);
                for h in local {
                    if h == hap1_idx || h == hap1_idx + 1 {
                        continue;
                    }
                    if seen.insert(h) {
                        out.push(h);
                    }
                }
            }
        }
        out
    };

    // MCMC loop: Gibbs sampling alternating between H1 and H2
    for step in 0..n_mcmc_steps {
        // === Sample H1 | (G, H2_fixed) ===

        // 1. Select neighbors using "Latent State" approach:
        //    Use H1's previously sampled state at a marker to find neighbors
        //    Vary the marker position across steps for robustness
        let center_marker = if n_mcmc_steps > 1 {
            n_markers / 4 + step * n_markers / (2 * n_mcmc_steps)
        } else {
            n_markers / 2
        };
        neighbors = collect_dynamic_neighbors(&path1_ref, sample_idx);
        let ref_hap = path1_ref.get(center_marker).copied().unwrap_or(0);
        if (ref_hap as usize) < phase_ibs.n_haps() && !neighbors.contains(&ref_hap) {
            neighbors.push(ref_hap);
        }
        if neighbors.is_empty() {
            continue;
        }
        mix_neighbors(&mut neighbors, n_states, n_haps, hap1_idx, &mut rng);

        // 2. Build constraint: at hets, H1 must produce genotype with H2
        for m in 0..n_markers {
            let a1 = seq1[m];
            let a2 = seq2[m];
            let anchor_a1 = anchor_h1.get(m).copied().unwrap_or(255);
            let anchor_a2 = anchor_h2.get(m).copied().unwrap_or(255);
            let is_anchor = anchor_a1 != 255 || anchor_a2 != 255;
            if a1 == 255 || a2 == 255 || a1 == a2 {
                fixed_allele[m] = 255; // No constraint (hom/missing)
            } else if is_anchor {
                fixed_allele[m] = anchor_a2;
            } else {
                fixed_allele[m] = 255; // Unphased het: no orientation constraint
            }
        }

        // 3. Run haploid FFBS for H1
        ffbs_haploid_constrained(
            &mut path1_idx,
            n_markers,
            neighbors.len(),
            p_recomb,
            seq1,
            seq2,
            conf,
            &fixed_allele,
            &neighbors,
            phase_ibs,
            p_no_err,
            p_err,
            &mut rng,
            workspace,
        );

        // Refresh the latent reference path at all markers for the next iteration.
        refresh_path_ref_from_states(&mut path1_ref, &path1_idx, &neighbors);

        // 4. Update H1 based on sampled reference alleles at hets
        //    GIBBS SAMPLING: only update H1, leave H2 fixed
        //    At hets, set H1 to match the reference's allele (if compatible).
        for m in 0..n_markers {
            let state = path1_idx[m] as usize;
            let a1 = seq1[m];
            let a2 = seq2[m];
            let anchor_a1 = anchor_h1.get(m).copied().unwrap_or(255);
            let anchor_a2 = anchor_h2.get(m).copied().unwrap_or(255);
            let is_anchor = anchor_a1 != 255 || anchor_a2 != 255;
            if is_anchor {
                h1_alleles[m] = anchor_a1;
                h2_alleles[m] = anchor_a2;
                continue;
            }

            if a1 == 255 && a2 == 255 {
                h1_alleles[m] = 255;
            } else if a1 == a2 {
                h1_alleles[m] = a1;
            } else if state < neighbors.len() {
                // Het: use reference allele to determine H1
                let ref_hap = neighbors[state];
                let ref_al = phase_ibs.allele(m, ref_hap);
                if ref_al == a1 || ref_al == a2 {
                    // Set H1 to ref_al, and H2 must be the other allele
                    h1_alleles[m] = ref_al;
                    h2_alleles[m] = if ref_al == a1 { a2 } else { a1 };
                }
                // If ref_al is missing/different, keep current phase
            }
        }

        // === Sample H2 | (G, H1_new) ===

        // 1. Select neighbors for H2 using H2's own latent state (not H1's!)
        neighbors = collect_dynamic_neighbors(&path2_ref, sample_idx);
        let ref_hap = path2_ref.get(center_marker).copied().unwrap_or(0);
        if (ref_hap as usize) < phase_ibs.n_haps() && !neighbors.contains(&ref_hap) {
            neighbors.push(ref_hap);
        }
        if neighbors.is_empty() {
            continue;
        }
        mix_neighbors(&mut neighbors, n_states, n_haps, hap1_idx, &mut rng);

        // 2. Build constraint: at hets, H2 must produce genotype with H1
        for m in 0..n_markers {
            let a1 = seq1[m];
            let a2 = seq2[m];
            let anchor_a1 = anchor_h1.get(m).copied().unwrap_or(255);
            let anchor_a2 = anchor_h2.get(m).copied().unwrap_or(255);
            let is_anchor = anchor_a1 != 255 || anchor_a2 != 255;
            if a1 == 255 || a2 == 255 || a1 == a2 {
                fixed_allele[m] = 255;
            } else if is_anchor {
                fixed_allele[m] = anchor_a1;
            } else {
                fixed_allele[m] = 255; // Unphased het: no orientation constraint
            }
        }

        // 3. Run haploid FFBS for H2
        ffbs_haploid_constrained(
            &mut path2_idx,
            n_markers,
            neighbors.len(),
            p_recomb,
            seq1,
            seq2,
            conf,
            &fixed_allele,
            &neighbors,
            phase_ibs,
            p_no_err,
            p_err,
            &mut rng,
            workspace,
        );

        // Refresh the latent reference path at all markers for the next iteration.
        refresh_path_ref_from_states(&mut path2_ref, &path2_idx, &neighbors);

        // 4. Update H2 based on sampled reference alleles
        //    GIBBS SAMPLING: only update H2, leave H1 fixed
        //    At hets, H2 is constrained to be opposite of H1, so just verify consistency.
        for m in 0..n_markers {
            let a1 = seq1[m];
            let a2 = seq2[m];
            let anchor_a1 = anchor_h1.get(m).copied().unwrap_or(255);
            let anchor_a2 = anchor_h2.get(m).copied().unwrap_or(255);
            let is_anchor = anchor_a1 != 255 || anchor_a2 != 255;
            if is_anchor {
                h1_alleles[m] = anchor_a1;
                h2_alleles[m] = anchor_a2;
                continue;
            }

            if a1 == 255 && a2 == 255 {
                h2_alleles[m] = 255;
            } else if a1 == a2 {
                h2_alleles[m] = a2;
            } else {
                // Het: H2 must be opposite of H1 (already determined in H1 step)
                // The constraint in emit_haploid_constrained enforced this.
                // Just ensure consistency - H2 is the allele NOT assigned to H1.
                h2_alleles[m] = if h1_alleles[m] == a1 { a2 } else { a1 };
            }
        }

        // After first step, we have a valid path to use for latent state lookup
        // in subsequent iterations
    }

    // Determine swap decisions from final H1, H2 vs original seq1, seq2
    let mut swap_bits = Vec::with_capacity(het_positions.len());
    let mut swap_lr = Vec::with_capacity(het_positions.len());
    let mut swap_probs = Vec::with_capacity(het_positions.len());

    for &m in het_positions {
        let a1 = seq1[m];
        let a2 = seq2[m];

        if a1 == 255 || a2 == 255 || a1 == a2 {
            swap_bits.push(0);
            swap_lr.push(1.0);
            swap_probs.push(0.5);
            continue;
        }

        // Original phase: seq1[m] on H1, seq2[m] on H2
        // Swap if final H1 allele differs from original seq1
        let swap = h1_alleles[m] != a1;
        swap_bits.push(if swap { 1 } else { 0 });

        // Compute LR from the reference allele at this position (use H1's path)
        let ref_al = if (path1_ref[m] as usize) < phase_ibs.n_haps() {
            phase_ibs.allele(m, path1_ref[m])
        } else {
            255
        };
        let lr = compute_phase_lr_single(
            h1_alleles[m], // chosen allele for H1
            h2_alleles[m], // other allele (H2)
            ref_al,
            conf[m],
            p_no_err,
            p_err,
        );
        swap_lr.push(lr);
        swap_probs.push(lr / (1.0 + lr));
    }

    (
        swap_bits,
        swap_lr,
        swap_probs,
        MosaicPaths {
            path1: path1_ref,
            path2: path2_ref,
        },
    )
}

/// Find the best constant pair of states that explains the target genotype.
///
/// This initialization heuristic performs a pairwise scan of all HMM states (which
/// correspond to reference haplotypes in ThreadedHaps) to find the pair (i, j)
/// that maximizes consistency with the target genotype.
///
/// This breaks the symmetry of the Combined HMM initialization (which cannot distinguish
/// between phasing configurations at 0/1 sites) and helps the Gibbs sampler escape
/// "Mosaic Traps" where H1 and H2 lock each other into high-switching local optima.
fn find_best_constant_pair_with_buffer<RefSpace>(
    n_markers: usize,
    n_states: usize,
    seq1: &[u8],
    seq2: &[u8],
    ref_provider: &mut RefAlleleProvider<'_, AnyMarkerSpace, RefSpace>,
    scores: &mut Vec<f32>,
) -> Option<MosaicPaths> {
    if n_states < 2 {
        return None;
    }

    let need = n_states * n_states;
    if scores.len() < need {
        scores.resize(need, 0.0);
    } else {
        scores[..need].fill(0.0);
    }

    let mut ref_alleles = vec![255u8; n_states];
    let mut informative = 0usize;
    for m in 0..n_markers {
        let a1 = seq1[m];
        let a2 = seq2[m];
        if a1 == 255 && a2 == 255 {
            continue;
        }
        informative += 1;

        let is_het = a1 != a2 && a1 != 255 && a2 != 255;

        ref_provider.fill_ref_alleles(m, &mut ref_alleles);
        for i in 0..n_states {
            let r1 = ref_alleles[i];
            if r1 == 255 {
                continue;
            }

            // Symmetric scan: only check j < i (lower triangle)
            // We can infer upper triangle or just pick best from lower.
            for j in 0..i {
                let r2 = ref_alleles[j];
                if r2 == 255 {
                    continue;
                }

                let compatible = if is_het {
                    // Het: need (r1=a1, r2=a2) OR (r1=a2, r2=a1)
                    (r1 == a1 && r2 == a2) || (r1 == a2 && r2 == a1)
                } else {
                    // Hom (or one missing): need r1=obs and r2=obs
                    // If a1 or a2 is missing, we match the present one.
                    let obs = if a1 != 255 { a1 } else { a2 };
                    r1 == obs && r2 == obs
                };

                if compatible {
                    scores[i * n_states + j] += 1.0;
                } else {
                    scores[i * n_states + j] -= 1.0;
                }
            }
        }
    }

    // Find best pair
    let mut best_score = f32::NEG_INFINITY;
    let mut best_pair = (0, 1);

    for i in 0..n_states {
        for j in 0..i {
            let s = scores[i * n_states + j];
            if s > best_score {
                best_score = s;
                best_pair = (i, j);
            }
        }
    }

    // If best score is too low (worse than random), maybe don't use it?
    // But random initialization is also bad. This is likely the "least bad" start.
    // So we return it.
    if informative == 0 {
        return None;
    }
    let threshold = 0.5 * (informative as f32);
    if best_score < threshold || n_markers > 500 {
        return None;
    }

    let path1 = vec![best_pair.0 as u32; n_markers];
    let path2 = vec![best_pair.1 as u32; n_markers];

    Some(MosaicPaths { path1, path2 })
}

/// Sample phase swap decisions using Stochastic EM (single chain MCMC).
///
/// This implements Forward-Filtering Backward-Sampling (FFBS) with a single
/// Markov chain, which is the mathematically correct approach for phasing.
/// Multiple chains would require phase alignment to avoid symmetric mode
/// cancellation, so we use exactly one chain (Stochastic EM).
///
/// The algorithm:
/// 1. Initialize H1/H2 using pairwise compatibility search (breaks symmetry)
///    OR fall back to Combined HMM checkpoint sampling
/// 2. Run burn-in steps to let the chain mix via Gibbs sampling
/// 3. Take samples from the posterior
/// 4. Return swap decisions based on average posterior
fn sample_swap_bits_mosaic<RefSpace>(
    n_markers: usize,
    n_states: usize,
    p_recomb: &[f32],
    seq1: &[u8],
    seq2: &[u8],
    conf: &[f32],
    mut ref_provider: RefAlleleProvider<'_, AnyMarkerSpace, RefSpace>,
    pl_provider: Option<PlProvider>,
    block_starts: Arc<[usize]>,
    het_positions: &[usize],
    initial_paths: Option<&MosaicPaths>,
    anchor_hap1: Option<&[u8]>,
    anchor_hap2: Option<&[u8]>,
    seed: u64,
    burnin: usize,
    lr_samples_param: usize,
    p_no_err: f32,
    p_err: f32,
    workspace: &mut crate::utils::workspace::ThreadWorkspace,
) -> (Vec<u8>, Vec<f32>, Vec<f32>, MosaicPaths) {
    if het_positions.is_empty() || n_markers == 0 || n_states == 0 {
        return (
            Vec::new(),
            Vec::new(),
            Vec::new(),
            MosaicPaths {
                path1: Vec::new(),
                path2: Vec::new(),
            },
        );
    }

    let max_block_len = max_block_len_from_starts(&block_starts, n_markers).max(1);
    let n_blocks = block_starts.len().max(1);
    // Resize workspace if needed for this window
    workspace.ensure_for_window(n_markers, n_states, max_block_len, n_blocks);
    let anchor_h1 = anchor_hap1.unwrap_or(&[]);
    let anchor_h2 = anchor_hap2.unwrap_or(&[]);
    let has_anchor = anchor_h1.iter().any(|&a| a != 255) || anchor_h2.iter().any(|&a| a != 255);
    let combined_data = std::mem::take(&mut workspace.combined_checkpoint_data);
    // Attempt pairwise initialization if no initial paths provided
    let mut heuristic_paths = if initial_paths.is_none() {
        find_best_constant_pair_with_buffer(
            n_markers,
            n_states,
            seq1,
            seq2,
            &mut ref_provider,
            &mut workspace.scores,
        )
    } else {
        None
    };

    // Align heuristic orientation to anchors when present.
    if has_anchor {
        if let Some(paths) = heuristic_paths.as_mut() {
            let mut score_direct: i32 = 0;
            let mut score_flip: i32 = 0;
            let ref_alleles = &mut workspace.ref_alleles;
            for m in 0..n_markers {
                let a1 = anchor_h1.get(m).copied().unwrap_or(255);
                let a2 = anchor_h2.get(m).copied().unwrap_or(255);
                if a1 == 255 && a2 == 255 {
                    continue;
                }
                ref_provider.fill_ref_alleles(m, ref_alleles);
                let p1 = paths.path1[m] as usize;
                let p2 = paths.path2[m] as usize;
                if p1 >= n_states || p2 >= n_states {
                    continue;
                }
                let r1 = ref_alleles[p1];
                let r2 = ref_alleles[p2];
                if a1 != 255 {
                    if r1 == a1 {
                        score_direct += 1;
                    } else {
                        score_direct -= 1;
                    }
                    if r2 == a1 {
                        score_flip += 1;
                    } else {
                        score_flip -= 1;
                    }
                }
                if a2 != 255 {
                    if r2 == a2 {
                        score_direct += 1;
                    } else {
                        score_direct -= 1;
                    }
                    if r1 == a2 {
                        score_flip += 1;
                    } else {
                        score_flip -= 1;
                    }
                }
            }
            if score_flip > score_direct {
                std::mem::swap(&mut paths.path1, &mut paths.path2);
                eprintln!(
                    "[anchor init] flipped heuristic paths (direct={} flip={})",
                    score_direct, score_flip
                );
            } else {
                eprintln!(
                    "[anchor init] kept heuristic paths (direct={} flip={})",
                    score_direct, score_flip
                );
            }
        }
    }

    let mut start_paths = initial_paths.or(heuristic_paths.as_ref());
    let mut start_paths_owned: Option<MosaicPaths> = None;
    if has_anchor {
        if let Some(paths) = start_paths {
            let mut score_direct: i32 = 0;
            let mut score_flip: i32 = 0;
            let ref_alleles = &mut workspace.ref_alleles;
            for m in 0..n_markers {
                let a1 = anchor_h1.get(m).copied().unwrap_or(255);
                let a2 = anchor_h2.get(m).copied().unwrap_or(255);
                if a1 == 255 && a2 == 255 {
                    continue;
                }
                ref_provider.fill_ref_alleles(m, ref_alleles);
                let p1 = paths.path1[m] as usize;
                let p2 = paths.path2[m] as usize;
                if p1 >= n_states || p2 >= n_states {
                    continue;
                }
                let r1 = ref_alleles[p1];
                let r2 = ref_alleles[p2];
                if a1 != 255 {
                    if r1 == a1 {
                        score_direct += 1;
                    } else {
                        score_direct -= 1;
                    }
                    if r2 == a1 {
                        score_flip += 1;
                    } else {
                        score_flip -= 1;
                    }
                }
                if a2 != 255 {
                    if r2 == a2 {
                        score_direct += 1;
                    } else {
                        score_direct -= 1;
                    }
                    if r1 == a2 {
                        score_flip += 1;
                    } else {
                        score_flip -= 1;
                    }
                }
            }
            if score_flip > score_direct {
                let mut owned = paths.clone();
                std::mem::swap(&mut owned.path1, &mut owned.path2);
                start_paths_owned = Some(owned);
                eprintln!(
                    "[anchor init] flipped start paths (direct={} flip={})",
                    score_direct, score_flip
                );
            }
        }
    }
    if start_paths_owned.is_some() {
        start_paths = start_paths_owned.as_ref();
    }

    // Only build combined checkpoints if we don't have a start path
    // This optimization avoids the expensive Combined HMM step when we have a good guess
    let mut combined_checkpoints =
        FwdCheckpoints::from_buffer(block_starts.clone(), n_states, combined_data);

    if start_paths.is_none() {
        if workspace.dummy_target.len() < n_markers {
            workspace.dummy_target.resize(n_markers, 255);
            workspace.dummy_partner.resize(n_markers, 255);
            workspace.dummy_combined.resize(n_markers, true);
            workspace.dummy_hard_match.resize(n_markers, false);
        } else {
            workspace.dummy_target[..n_markers].fill(255);
            workspace.dummy_partner[..n_markers].fill(255);
            workspace.dummy_combined[..n_markers].fill(true);
            workspace.dummy_hard_match[..n_markers].fill(false);
        }
        let dummy_target = &workspace.dummy_target[..n_markers];
        let dummy_partner = &workspace.dummy_partner[..n_markers];
        let dummy_combined = &workspace.dummy_combined[..n_markers];
        let dummy_hard_match = &workspace.dummy_hard_match[..n_markers];
        let fwd = &mut workspace.fwd[..n_states];
        let fwd_prior = &mut workspace.fwd_prior[..n_states];
        let ref_alleles = &mut workspace.ref_alleles[..n_states];
        build_fwd_checkpoints(
            &mut combined_checkpoints,
            n_markers,
            n_states,
            p_recomb,
            seq1,
            seq2,
            conf,
            HapEmissionInputs {
                target_constraint: &dummy_target,
                partner_allele: &dummy_partner,
                use_combined: &dummy_combined,
                hard_match: &dummy_hard_match,
            },
            &mut ref_provider,
            pl_provider.as_ref(),
            &mut workspace.allele_probs,
            fwd,
            fwd_prior,
            ref_alleles,
            p_no_err,
            p_err,
            EmissionMode::Combined,
        );
    }

    let combined_checkpoints_ref = &combined_checkpoints;

    // Pure Stochastic EM: single chain
    let chain_anchor_hap1 = anchor_hap1.unwrap_or(&[]).to_vec();
    let chain_anchor_hap2 = anchor_hap2.unwrap_or(&[]).to_vec();
    let mut anchor_indices: Vec<usize> = Vec::new();
    if !chain_anchor_hap1.is_empty() || !chain_anchor_hap2.is_empty() {
        for m in 0..n_markers {
            let a1 = chain_anchor_hap1.get(m).copied().unwrap_or(255);
            let a2 = chain_anchor_hap2.get(m).copied().unwrap_or(255);
            if a1 != 255 || a2 != 255 {
                anchor_indices.push(m);
            }
        }
    }
    let lr_samples = lr_samples_param.max(1);

    let run_chain = |seed: u64,
                     init_paths: Option<&MosaicPaths>,
                     buffers: MosaicBuffers,
                     ref_provider: RefAlleleProvider<'_, AnyMarkerSpace, RefSpace>|
     -> (Vec<u32>, Vec<u32>, MosaicPaths, MosaicBuffers) {
        let mut chain = MosaicChain::new_with_buffers(
            seed,
            n_markers,
            n_states,
            p_recomb,
            seq1,
            seq2,
            conf,
            ref_provider,
            combined_checkpoints_ref,
            buffers,
            p_no_err,
            p_err,
            pl_provider,
            chain_anchor_hap1.clone(),
            chain_anchor_hap2.clone(),
        );

        if let Some(paths) = init_paths {
            let has_valid_lengths = paths.path1.len() == n_markers && paths.path2.len() == n_markers;
            let has_valid_states = has_valid_lengths
                && paths.path1.iter().all(|&p| (p as usize) < n_states)
                && paths.path2.iter().all(|&p| (p as usize) < n_states);
            if has_valid_states {
                chain.path1.resize(n_markers, 0);
                chain.path2.resize(n_markers, 0);
                chain.path1.copy_from_slice(&paths.path1);
                chain.path2.copy_from_slice(&paths.path2);
                chain.first_iteration = false;
            }
        }

        for _ in 0..burnin {
            chain.step();
        }

        let mut swap_counts = vec![0u32; het_positions.len()];
        let mut obs_counts = vec![0u32; het_positions.len()];
        let mut new_paths = MosaicPaths {
            path1: Vec::new(),
            path2: Vec::new(),
        };

        for sample_idx in 0..lr_samples {
            chain.step();
            let is_last = sample_idx + 1 == lr_samples;

            let mut sample_flip: Option<bool> = None;
            if !anchor_indices.is_empty() {
                let mut direct = 0u32;
                let mut flipped = 0u32;
                for &m in &anchor_indices {
                    let a1 = chain_anchor_hap1.get(m).copied().unwrap_or(255);
                    let a2 = chain_anchor_hap2.get(m).copied().unwrap_or(255);
                    if a1 == 255 && a2 == 255 {
                        continue;
                    }
                    chain.ref_provider.fill_ref_alleles(m, &mut chain.ref_alleles);
                    let p1 = chain.path1[m] as usize;
                    let p2 = chain.path2[m] as usize;
                    if p1 >= n_states || p2 >= n_states {
                        continue;
                    }
                    let r1 = chain.ref_alleles[p1];
                    let r2 = chain.ref_alleles[p2];
                    if a1 != 255 && a2 != 255 {
                        if r1 == a1 && r2 == a2 {
                            direct += 1;
                        } else if r1 == a2 && r2 == a1 {
                            flipped += 1;
                        }
                    } else if a1 != 255 {
                        if r1 == a1 {
                            direct += 1;
                        } else if r2 == a1 {
                            flipped += 1;
                        }
                    } else if a2 != 255 {
                        if r2 == a2 {
                            direct += 1;
                        } else if r1 == a2 {
                            flipped += 1;
                        }
                    }
                }
                if direct > 0 || flipped > 0 {
                    sample_flip = Some(flipped > direct);
                }
            }

            for (i, &m) in het_positions.iter().enumerate() {
                let a1 = seq1[m];
                let a2 = seq2[m];
                if a1 == 255 || a2 == 255 || a1 == a2 {
                    continue;
                }

                let p1 = chain.path1[m] as usize;
                let p2 = chain.path2[m] as usize;
                if p1 >= n_states || p2 >= n_states {
                    continue;
                }
                chain.ref_provider.fill_ref_alleles(m, &mut chain.ref_alleles);
                let ref1 = chain.ref_alleles[p1];
                let ref2 = chain.ref_alleles[p2];

                let mut orient = if ref1 == a1 && ref2 == a2 {
                    Some(0u8)
                } else if ref1 == a2 && ref2 == a1 {
                    Some(1u8)
                } else if ref1 == 255 && ref2 == a2 {
                    Some(0u8)
                } else if ref1 == 255 && ref2 == a1 {
                    Some(1u8)
                } else if ref2 == 255 && ref1 == a1 {
                    Some(0u8)
                } else if ref2 == 255 && ref1 == a2 {
                    Some(1u8)
                } else {
                    None
                };
                if let (Some(flip), Some(val)) = (sample_flip, orient) {
                    if flip {
                        orient = Some(1 - val);
                    }
                }

                if let Some(orient) = orient {
                    swap_counts[i] += orient as u32;
                    obs_counts[i] += 1;
                }
            }

            if is_last {
                new_paths = MosaicPaths {
                    path1: chain.path1.clone(),
                    path2: chain.path2.clone(),
                };
            }
        }

        let returned = chain.into_buffers();
        (swap_counts, obs_counts, new_paths, returned)
    };

    let ref_view = ref_provider.ref_gt;
    let threaded_haps = ref_provider.threaded_haps;

    let buffers = MosaicBuffers {
        fwd: std::mem::replace(&mut workspace.fwd, aligned_vec::AVec::new(32)),
        fwd_prior: std::mem::replace(&mut workspace.fwd_prior, aligned_vec::AVec::new(32)),
        ref_alleles: std::mem::take(&mut workspace.ref_alleles),
        weights: std::mem::take(&mut workspace.weights),
        allele_probs: std::mem::take(&mut workspace.allele_probs),
        hap1_checkpoints: FwdCheckpoints::from_buffer(
            block_starts.clone(),
            n_states,
            std::mem::take(&mut workspace.hap1_checkpoint_data),
        ),
        hap2_checkpoints: FwdCheckpoints::from_buffer(
            block_starts.clone(),
            n_states,
            std::mem::take(&mut workspace.hap2_checkpoint_data),
        ),
        hap1_allele: std::mem::take(&mut workspace.hap1_allele),
        hap1_partner_allele: std::mem::take(&mut workspace.hap1_partner_allele),
        hap1_use_combined: std::mem::take(&mut workspace.hap1_use_combined),
        hap1_hard_match: std::mem::take(&mut workspace.hap1_hard_match),
        hap2_allele: std::mem::take(&mut workspace.hap2_allele),
        hap2_partner_allele: std::mem::take(&mut workspace.hap2_partner_allele),
        hap2_use_combined: std::mem::take(&mut workspace.hap2_use_combined),
        hap2_hard_match: std::mem::take(&mut workspace.hap2_hard_match),
        path1: std::mem::take(&mut workspace.path1),
        path2: std::mem::take(&mut workspace.path2),
        fwd_block: std::mem::take(&mut workspace.fwd_block),
    };

    let chain_seed = seed.wrapping_add(0xC0FFEE_BAAD_F00Du64);
    let (swap_counts1, obs_counts1, new_paths, mut buffers) =
        run_chain(chain_seed, start_paths, buffers, ref_provider);

    let mut swap_counts = swap_counts1;
    let mut obs_counts = obs_counts1;

    if !has_anchor {
        let flipped_paths =
            if new_paths.path1.len() == n_markers && new_paths.path2.len() == n_markers {
                Some(MosaicPaths {
                    path1: new_paths.path2.clone(),
                    path2: new_paths.path1.clone(),
                })
            } else {
                None
            };

        let chain_seed_2 = seed.wrapping_add(0xBAD_CAFE_F00Du64);
        let ref_provider_2 = RefAlleleProvider::new(ref_view, threaded_haps);
        let (swap_counts2, obs_counts2, paths2, buffers2) = run_chain(
            chain_seed_2,
            flipped_paths.as_ref(),
            buffers,
            ref_provider_2,
        );
        buffers = buffers2;
        if paths2.path1.len() != n_markers || paths2.path2.len() != n_markers {
            eprintln!(
                "[mosaic paths] secondary chain path lengths: path1={} path2={}",
                paths2.path1.len(),
                paths2.path2.len()
            );
        }

        for i in 0..het_positions.len() {
            swap_counts[i] = swap_counts[i].saturating_add(swap_counts2[i]);
            obs_counts[i] = obs_counts[i].saturating_add(obs_counts2[i]);
        }
    }

    let mut swap_bits = Vec::with_capacity(het_positions.len());
    let mut swap_lr = Vec::with_capacity(het_positions.len());
    let mut swap_probs = Vec::with_capacity(het_positions.len());
    let mut obs_zero = 0usize;
    let mut p_min = 1.0f32;
    let mut p_max = 0.0f32;
    let mut p_sum = 0.0f32;
    for (i, &m) in het_positions.iter().enumerate() {
        let a1 = seq1[m];
        let a2 = seq2[m];
        if a1 == 255 || a2 == 255 || a1 == a2 || obs_counts[i] == 0 {
            swap_bits.push(0);
            swap_lr.push(1.0);
            swap_probs.push(0.5);
            if obs_counts[i] == 0 {
                obs_zero += 1;
            }
            continue;
        }

        let p_swap = (swap_counts[i] as f32 + 0.5) / (obs_counts[i] as f32 + 1.0);
        let p_keep = 1.0 - p_swap;
        let chosen_swap = p_swap > 0.5;
        swap_bits.push(chosen_swap as u8);
        let (max_p, min_p) = if p_swap >= p_keep {
            (p_swap, p_keep)
        } else {
            (p_keep, p_swap)
        };
        let lr = if min_p < 1e-30 {
            1e6
        } else {
            (max_p / min_p).min(1e6)
        };
        swap_lr.push(lr);
        swap_probs.push(p_swap.clamp(0.0, 1.0));
        let p = p_swap.clamp(0.0, 1.0);
        p_min = p_min.min(p);
        p_max = p_max.max(p);
        p_sum += p;
    }
    if has_anchor && !het_positions.is_empty() {
        let mut dp0 = vec![f32::NEG_INFINITY; het_positions.len()];
        let mut dp1 = vec![f32::NEG_INFINITY; het_positions.len()];
        let mut prev_state = vec![0u8; het_positions.len()];

        for (i, &m) in het_positions.iter().enumerate() {
            let a1 = seq1[m];
            let a2 = seq2[m];
            let p_swap = swap_probs[i].clamp(1e-6, 1.0 - 1e-6);
            let p_keep = (1.0 - p_swap).clamp(1e-6, 1.0 - 1e-6);
            let emit0 = p_keep.ln();
            let emit1 = p_swap.ln();
            let anchor_a1 = anchor_h1.get(m).copied().unwrap_or(255);
            let anchor_a2 = anchor_h2.get(m).copied().unwrap_or(255);
            let mut force: Option<u8> = None;
            if anchor_a1 != 255 || anchor_a2 != 255 {
                if a1 == anchor_a1 && a2 == anchor_a2 {
                    force = Some(0);
                } else if a1 == anchor_a2 && a2 == anchor_a1 {
                    force = Some(1);
                }
            }

            let r = p_recomb.get(m).copied().unwrap_or(0.0).clamp(1e-9, 1.0 - 1e-9);
            let stay = (1.0 - r).ln();
            let sw = r.ln();

            if i == 0 {
                dp0[i] = emit0;
                dp1[i] = emit1;
            } else {
                let from0_to0 = dp0[i - 1] + stay;
                let from1_to0 = dp1[i - 1] + sw;
                if from0_to0 >= from1_to0 {
                    dp0[i] = from0_to0 + emit0;
                    prev_state[i] = 0;
                } else {
                    dp0[i] = from1_to0 + emit0;
                    prev_state[i] = 1;
                }

                let from0_to1 = dp0[i - 1] + sw;
                let from1_to1 = dp1[i - 1] + stay;
                if from0_to1 >= from1_to1 {
                    dp1[i] = from0_to1 + emit1;
                } else {
                    dp1[i] = from1_to1 + emit1;
                    prev_state[i] |= 2;
                }
            }

            if let Some(state) = force {
                if state == 0 {
                    dp1[i] = f32::NEG_INFINITY;
                } else {
                    dp0[i] = f32::NEG_INFINITY;
                }
            }
        }

        let mut state = if dp1[het_positions.len() - 1] > dp0[het_positions.len() - 1] {
            1u8
        } else {
            0u8
        };
        for idx in (0..het_positions.len()).rev() {
            swap_bits[idx] = state;
            if idx == 0 {
                break;
            }
            let prev = prev_state[idx];
            if state == 0 {
                state = prev & 1;
            } else {
                state = if (prev & 2) != 0 { 1 } else { 0 };
            }
        }
    }
    if !het_positions.is_empty() {
        let denom = (het_positions.len().saturating_sub(obs_zero)).max(1) as f32;
        let mean = p_sum / denom;
        eprintln!(
            "[swap stats] hets={} obs0={} p_min={:.3} p_mean={:.3} p_max={:.3} anchors={}",
            het_positions.len(),
            obs_zero,
            p_min,
            mean,
            p_max,
            chain_anchor_hap1.iter().filter(|&&a| a != 255).count()
        );
        if n_markers <= 60 {
            let limit = het_positions.len().min(12);
            for i in 0..limit {
                let m = het_positions[i];
                eprintln!(
                    "[swap stats] m={} obs={} swap_ct={} p={:.3}",
                    m,
                    obs_counts[i],
                    swap_counts[i],
                    swap_probs[i]
                );
            }
        }
    }

    // Return buffers to workspace for reuse
    workspace.fwd = buffers.fwd;
    workspace.fwd_prior = buffers.fwd_prior;
    workspace.ref_alleles = buffers.ref_alleles;
    workspace.weights = buffers.weights;
    workspace.allele_probs = buffers.allele_probs;
    workspace.hap1_checkpoint_data = buffers.hap1_checkpoints.into_buffer();
    workspace.hap2_checkpoint_data = buffers.hap2_checkpoints.into_buffer();
    workspace.hap1_allele = buffers.hap1_allele;
    workspace.hap1_partner_allele = buffers.hap1_partner_allele;
    workspace.hap1_use_combined = buffers.hap1_use_combined;
    workspace.hap1_hard_match = buffers.hap1_hard_match;
    workspace.hap2_allele = buffers.hap2_allele;
    workspace.hap2_partner_allele = buffers.hap2_partner_allele;
    workspace.hap2_use_combined = buffers.hap2_use_combined;
    workspace.hap2_hard_match = buffers.hap2_hard_match;
    workspace.path1 = buffers.path1;
    workspace.path2 = buffers.path2;
    workspace.fwd_block = buffers.fwd_block;
    workspace.combined_checkpoint_data = combined_checkpoints.into_buffer();

    (swap_bits, swap_lr, swap_probs, new_paths)
}

/// Decision type for Stage 2 marker processing
#[derive(Debug, Clone)]
enum Stage2Decision {
    /// Phase an unphased heterozygote
    Phase {
        marker: usize,
        should_swap: bool,
        lr: f32,
    },
    /// Impute a missing genotype
    Impute { marker: usize, a1: u8, a2: u8 },
}

/// Stage 2 phaser with HMM state probability interpolation
///
/// Implements the algorithm from Java Beagle's Stage2Baum.java for phasing
/// rare variants using interpolated HMM state probabilities.
struct Stage2Phaser {
    /// For each Stage 2 marker, the index of the preceding Stage 1 marker
    prev_stage1_marker: Vec<usize>,
    /// Number of Stage 1 markers
    n_stage1: usize,
    /// Stage 1 marker indices in original marker space
    stage1_markers: Vec<usize>,
    /// Genetic positions (cM) for all markers
    gen_positions: Vec<f64>,
    /// Recombination intensity for bridge interpolation
    recomb_intensity: f32,
}

impl Stage2Phaser {
    /// Create a new Stage2Phaser
    ///
    /// # Arguments
    /// * `hi_freq_markers` - Indices of high-frequency (Stage 1) markers in original space
    /// * `gen_positions` - Genetic positions (cM) for all markers
    /// * `n_total_markers` - Total number of markers
    fn new(
        hi_freq_markers: &[usize],
        gen_positions: &[f64],
        n_total_markers: usize,
        recomb_intensity: f32,
    ) -> Self {
        let n_stage1 = hi_freq_markers.len();

        // Build prevStage1Marker: for each marker, which Stage 1 marker precedes it
        let mut prev_stage1_marker = vec![0usize; n_total_markers];

        if n_stage1 >= 2 {
            // Fill markers before first Stage 1 marker with 0
            let first_hf = hi_freq_markers[0];
            prev_stage1_marker[..=first_hf].fill(0);

            // Fill between Stage 1 markers
            for j in 1..n_stage1 {
                let prev_hf = hi_freq_markers[j - 1];
                let curr_hf = hi_freq_markers[j];
                prev_stage1_marker[(prev_hf + 1)..=curr_hf].fill(j - 1);
            }

            // Fill after last Stage 1 marker
            let last_hf = hi_freq_markers[n_stage1 - 1];
            prev_stage1_marker[(last_hf + 1)..].fill(n_stage1 - 1);
        }

        Self {
            prev_stage1_marker,
            n_stage1,
            stage1_markers: hi_freq_markers.to_vec(),
            gen_positions: gen_positions.to_vec(),
            recomb_intensity,
        }
    }

    /// Compute interpolated allele probabilities for a rare marker
    ///
    /// Following Java Stage2Baum.unscaledAlProbs:
    /// - For each HMM state, interpolate probability from flanking Stage 1 markers
    /// - Accumulate allele probabilities based on reference haplotype alleles
    /// Compute allele probabilities using haploid Li-Stephens emission model.
    ///
    /// Each HMM state corresponds to a specific reference haplotype. The emission
    /// probability depends ONLY on that haplotype's allele - checking paired
    /// haplotypes would violate the haploid model assumption.
    fn interpolated_allele_probs<F>(
        &self,
        marker: usize,
        state_probs: &[Vec<f32>],   // [stage1_marker][state]
        haps_at_mkr_a: &[CombinedHapId], // haplotypes at flanking Stage 1 marker
        get_allele: &F,             // Closure to get allele for any haplotype
        a1: u8,
        a2: u8,
    ) -> [f32; 2]
    where
        F: Fn(usize, usize) -> u8, // (marker, hap_index) -> allele
    {
        let mut al_probs = [0.0f32; 2];

        let n_states = haps_at_mkr_a.len();
        let bridge_probs = self.bridge_state_probs(marker, state_probs, n_states);

        for j in 0..n_states {
            let hap = haps_at_mkr_a[j].as_u32() as usize;

            // Get allele from this specific haplotype at the rare marker.
            // Li-Stephens HMM models haploid copying: state k means we're copying
            // haplotype k, so emission depends ONLY on haplotype k's allele.
            // The paired haplotype (hap ^ 1) is irrelevant - checking it would
            // introduce "free switching" and wash out the phasing signal.
            let ref_allele = get_allele(marker, hap);

            if ref_allele == 255 {
                continue;
            }

            let prob = bridge_probs.get(j).copied().unwrap_or(0.0);

            // Simple haploid emission: if this reference haplotype carries a1, add
            // probability to a1; if it carries a2, add to a2.
            if ref_allele == a1 {
                al_probs[0] += prob;
            } else if ref_allele == a2 {
                al_probs[1] += prob;
            }
            // If ref_allele matches neither (e.g., multiallelic), no contribution
        }

        al_probs
    }

    fn p_recomb(&self, gen_dist_cm: f64) -> f32 {
        let c = -(self.recomb_intensity as f64);
        let gen_dist_m = gen_dist_cm / 100.0;
        (-f64::exp_m1(c * gen_dist_m)) as f32
    }

    fn bridge_state_probs(
        &self,
        marker: usize,
        state_probs: &[Vec<f32>],
        n_states: usize,
    ) -> Vec<f32> {
        let mkr_a = self.prev_stage1_marker[marker];
        let mkr_b = (mkr_a + 1).min(self.n_stage1 - 1);

        let probs_a = &state_probs[mkr_a];
        let probs_b = &state_probs[mkr_b];

        if mkr_a == mkr_b || self.stage1_markers.is_empty() {
            return probs_a.clone();
        }

        let pos_a_idx = self.stage1_markers[mkr_a];
        let pos_b_idx = self.stage1_markers[mkr_b];

        let pos_a = *self.gen_positions.get(pos_a_idx).unwrap_or(&0.0);
        let pos_b = *self.gen_positions.get(pos_b_idx).unwrap_or(&pos_a);
        let pos_m = *self.gen_positions.get(marker).unwrap_or(&pos_a);

        if pos_b <= pos_a {
            return probs_a.clone();
        }
        if pos_m <= pos_a {
            return probs_a.clone();
        }
        if pos_m >= pos_b {
            return probs_b.clone();
        }

        let d1 = (pos_m - pos_a).max(0.0);
        let d2 = (pos_b - pos_m).max(0.0);
        let r1 = self.p_recomb(d1);
        let r2 = self.p_recomb(d2);

        let shift1 = r1 / n_states.max(1) as f32;
        let shift2 = r2 / n_states.max(1) as f32;
        let scale1 = 1.0 - r1;
        let scale2 = 1.0 - r2;

        let denom = d1 + d2;
        let weight_a = if denom > 0.0 { (d2 / denom) as f32 } else { 0.5 };
        let weight_b = 1.0 - weight_a;

        let mut weights = vec![0.0f32; n_states];
        let mut sum = 0.0f32;
        for k in 0..n_states {
            let a = scale1 * probs_a.get(k).copied().unwrap_or(0.0) + shift1;
            let b = scale2 * probs_b.get(k).copied().unwrap_or(0.0) + shift2;
            let w = weight_a * a + weight_b * b;
            weights[k] = w;
            sum += w;
        }

        if sum > 0.0 {
            for w in &mut weights {
                *w /= sum;
            }
            weights
        } else {
            probs_a.clone()
        }
    }
}

impl<RefSpace: Send + Sync> PhasingPipeline<RefSpace> {
    /// Finalize Stage 2 phasing using context from next window
    ///
    /// Finalize Stage 2 phasing with forward context from the next window.
    ///
    /// Stage 1 phasing handles common variants in-window. Stage 2 handles rare variants
    /// using HMM state probabilities interpolated between Stage 1 markers.
    ///
    /// Cross-window context enables better rare variant phasing at window boundaries
    /// by providing forward context from the next window's phased markers. However,
    /// since GenotypeMatrix is immutable by design, the actual rare variant phasing
    /// is performed in-window by phase_rare_markers_with_hmm. This function validates
    /// the cross-window boundary continuity.
    ///
    /// The next_phased parameter provides forward context - markers from the next
    /// window that help verify phasing consistency at window boundaries.
    fn finalize_stage2_with_forward_context(
        &self,
        current_phased: &GenotypeMatrix<Phased>,
        next_phased: &GenotypeMatrix<Phased>,
    ) -> Result<GenotypeMatrix<Phased>> {
        let current_markers = current_phased.n_markers();
        let next_markers = next_phased.n_markers();
        let n_samples = current_phased.n_samples();

        if current_markers == 0 || next_markers == 0 || n_samples == 0 {
            return Ok(current_phased.clone());
        }

        // Find rare markers in the overlap region (last ~2cM or last 1000 markers)
        let overlap_start = current_markers.saturating_sub(1000);
        let rare_threshold = self.config.rare;

        // Collect markers that need re-phasing: rare hets in overlap that exist in next window
        let mut markers_to_fix: Vec<(usize, usize)> = Vec::new(); // (current_idx, next_idx)

        let mut next_idx = 0usize;
        for m in overlap_start..current_markers {
            let marker = current_phased.marker(MarkerIdx::new(m as u32));
            let key = (marker.chrom.0, marker.pos);

            // Advance next_idx until we reach or pass current marker (linear merge on sorted markers)
            while next_idx < next_markers {
                let next_marker = next_phased.marker(MarkerIdx::new(next_idx as u32));
                let next_key = (next_marker.chrom.0, next_marker.pos);
                if next_key < key {
                    next_idx += 1;
                } else {
                    break;
                }
            }

            // Check if this marker exists in next window
            if next_idx < next_markers {
                let next_marker = next_phased.marker(MarkerIdx::new(next_idx as u32));
                if (next_marker.chrom.0, next_marker.pos) != key {
                    continue;
                }
                let next_m = next_idx;
                // Check if it's a rare variant (simplified: check if any sample has het)
                let n_alleles = 1 + marker.alt_alleles.len();
                if n_alleles == 2 {
                    // For biallelic, check MAF
                    let mut alt_count = 0u32;
                    let n_haps = current_phased.n_haplotypes();
                    for h in 0..n_haps {
                        if current_phased.allele(MarkerIdx::new(m as u32), HapIdx::new(h as u32))
                            == 1
                        {
                            alt_count += 1;
                        }
                    }
                    let maf = (alt_count as f32 / n_haps as f32)
                        .min(1.0 - alt_count as f32 / n_haps as f32);
                    if maf < rare_threshold && maf > 0.0 {
                        markers_to_fix.push((m, next_m));
                    }
                }
            }
        }

        if markers_to_fix.is_empty() {
            tracing::debug!("Stage 2 finalization: no rare markers in overlap need fixing");
            return Ok(current_phased.clone());
        }

        tracing::debug!(
            "Stage 2 finalization: checking {} rare markers in overlap region",
            markers_to_fix.len()
        );

        let mut mismatches = 0usize;
        let mut matches = 0usize;

        // For each rare marker, check if next window has different phasing.
        // We avoid swapping single markers to preserve local LD structure.
        for (current_m, next_m) in markers_to_fix {
            for s in 0..n_samples {
                let hap1 = HapIdx::new((s * 2) as u32);
                let hap2 = HapIdx::new((s * 2 + 1) as u32);

                let curr_a1 = current_phased.allele(MarkerIdx::new(current_m as u32), hap1);
                let curr_a2 = current_phased.allele(MarkerIdx::new(current_m as u32), hap2);

                // Only fix heterozygotes
                if curr_a1 == curr_a2 || curr_a1 == 255 || curr_a2 == 255 {
                    continue;
                }

                let next_a1 = next_phased.allele(MarkerIdx::new(next_m as u32), hap1);
                let next_a2 = next_phased.allele(MarkerIdx::new(next_m as u32), hap2);

                // Check if next window has opposite phasing
                if next_a1 == curr_a2 && next_a2 == curr_a1 {
                    mismatches += 1;
                } else if next_a1 == curr_a1 && next_a2 == curr_a2 {
                    matches += 1;
                }
            }
        }

        if mismatches == 0 || mismatches <= matches {
            if mismatches > 0 {
                tracing::debug!(
                    "Stage 2 finalization: detected {} phase mismatches from forward context",
                    mismatches
                );
            }
            return Ok(current_phased.clone());
        }

        tracing::debug!(
            "Stage 2 finalization: applying phase flip ({} mismatches vs {} matches)",
            mismatches,
            matches
        );

        let n_markers = current_phased.n_markers();
        let n_haps = current_phased.n_haplotypes();
        let mut geno = MutableGenotypes::from_fn(n_markers, n_haps, |m, h| {
            current_phased.allele(MarkerIdx::new(m as u32), HapIdx::new(h as u32))
        });

        for s in 0..n_samples {
            let hap1 = HapIdx::new((s * 2) as u32);
            let hap2 = HapIdx::new((s * 2 + 1) as u32);

            let mut mask = BitVec::repeat(false, n_markers);
            for m in 0..n_markers {
                let a1 = current_phased.allele(MarkerIdx::new(m as u32), hap1);
                let a2 = current_phased.allele(MarkerIdx::new(m as u32), hap2);
                if a1 != a2 && a1 != 255 && a2 != 255 {
                    mask.set(m, true);
                }
            }
            geno.swap_haplotypes(hap1, hap2, &mask);
        }

        let markers = current_phased.markers().clone();
        let samples = current_phased.samples_arc();
        let columns: Vec<GenotypeColumn> = (0..n_markers)
            .map(|m| {
                let alleles = geno.marker_alleles(m);
                let bytes: Vec<u8> = alleles.to_vec();
                GenotypeColumn::from_alleles(&bytes, 2)
            })
            .collect();

        Ok(GenotypeMatrix::new_phased_with_confidence_and_likelihoods(
            markers,
            columns,
            samples,
            current_phased.confidence_clone(),
            current_phased.likelihoods_pl_arc(),
        )
        .with_phase_confidence(current_phased.phase_confidence_clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn build_test_markers(
        n_markers: usize,
        step_bp: u32,
    ) -> crate::data::marker::Markers<crate::data::AnyMarkerSpace> {
        use crate::data::marker::{Allele, Marker, Markers, Nucleotide};
        use crate::data::ChromIdx;

        let mut markers = Markers::<crate::data::AnyMarkerSpace>::new();
        markers.add_chrom("chr1");
        for i in 0..n_markers {
            let m = Marker::new(
                ChromIdx::new(0),
                i as u32 * step_bp,
                Some(format!("m{}", i).into()),
                Allele::Base(Nucleotide::A),
                vec![Allele::Base(Nucleotide::T)],
            );
            markers.push(m);
        }
        markers
    }

    fn build_ref_panel_with_hero(
        n_markers: usize,
        n_ref_samples: usize,
        hero_sample_idx: usize,
        hero_pattern: &[u8],
        seed: u64,
    ) -> (
        GenotypeMatrix<crate::data::storage::phase_state::Phased, crate::data::AnyMarkerSpace>,
        usize,
    ) {
        use crate::data::haplotype::Samples;
        use crate::data::storage::GenotypeColumn;
        use rand::{Rng, SeedableRng};
        use std::sync::Arc;

        let n_ref_haps = n_ref_samples * 2;
        let mut ref_haps = vec![vec![0u8; n_markers]; n_ref_haps];
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        for h in 0..n_ref_haps {
            for m in 0..n_markers {
                ref_haps[h][m] = if rng.random_bool(0.5) { 1 } else { 0 };
            }
        }

        let hero_hap_idx = hero_sample_idx * 2;
        let anti_hap_idx = hero_hap_idx + 1;
        for m in 0..n_markers {
            let a = hero_pattern[m];
            ref_haps[hero_hap_idx][m] = a;
            ref_haps[anti_hap_idx][m] = 1 - a;
        }

        let markers = build_test_markers(n_markers, 1_000);
        let samples = Arc::new(Samples::from_ids(
            (0..n_ref_samples).map(|i| format!("r{}", i)).collect(),
        ));
        let mut columns = Vec::with_capacity(n_markers);
        for m in 0..n_markers {
            let mut alleles = Vec::with_capacity(n_ref_haps);
            for h in 0..n_ref_haps {
                alleles.push(ref_haps[h][m]);
            }
            columns.push(GenotypeColumn::from_alleles(&alleles, 2));
        }
        let ref_gt = GenotypeMatrix::new_phased(markers, columns, samples);
        (ref_gt, hero_hap_idx)
    }

    fn build_target_with_sparse_anchors(
        n_markers: usize,
        hero_pattern: &[u8],
        anchor_every: usize,
    ) -> GenotypeMatrix<crate::data::storage::phase_state::Unphased, crate::data::AnyMarkerSpace>
    {
        use crate::data::haplotype::Samples;
        use crate::data::storage::GenotypeColumn;
        use std::sync::Arc;

        let markers = build_test_markers(n_markers, 1_000);
        let samples = Arc::new(Samples::from_ids(vec!["target".to_string()]));
        let mut columns = Vec::with_capacity(n_markers);
        let mut phase_mask = vec![vec![0u8; 1]; n_markers];

        for m in 0..n_markers {
            let hero = hero_pattern[m];
            let anti = 1 - hero;
            let (a1, a2) = if anchor_every > 0 && m % anchor_every == 0 {
                phase_mask[m][0] = 1;
                (hero, anti)
            } else {
                (0, 1)
            };
            columns.push(GenotypeColumn::from_alleles(&[a1, a2], 2));
        }

        GenotypeMatrix::new_unphased(markers, columns, samples).with_phase_mask(Some(phase_mask))
    }

    #[test]
    fn test_pipeline_creation() {
        let config = Config {
            gt: PathBuf::from("test.vcf"),
            r#ref: None,
            out: PathBuf::from("out"),
            map: None,
            chrom: None,
            excludesamples: None,
            excludemarkers: None,
            burnin: 3,
            iterations: 12,
            mcmc_burnin: 1,
            dynamic_mcmc: false,
            mcmc_steps: 3,
            mcmc_lr_samples: 32,
            phase_states: 280,
            rare: 0.002,
            impute: true,
            imp_states: 1600,
            imp_segment: 6.0,
            imp_step: 0.1,
            imp_nsteps: 7,
            cluster: 0.005,
            pbwt_batch_mb: 256,
            ap: false,
            gp: false,
            ne: 100000.0,
            err: None,
            em: false, // Disable EM for unit test to avoid complexity
            window: 40.0,
            window_markers: 100000,
            overlap: 2.0,
            seed: 12345,
            nthreads: None,
            profile: false,
        };

        let pipeline = PhasingPipeline::<crate::data::AnyMarkerSpace>::new(config, None);
        assert_eq!(pipeline.params.n_states, 280);
    }

    #[test]
    fn test_run_phase() {
        // Create a small pipeline and run phase_in_memory
        use crate::data::ChromIdx;
        use crate::data::genetic_map::GeneticMaps;
        use crate::data::haplotype::Samples;
        use crate::data::marker::{Allele, Marker, Markers};
        use crate::data::storage::GenotypeColumn;
        use crate::data::storage::matrix::GenotypeMatrix;
        use std::sync::Arc;

        let n_markers = 50;
        let n_samples = 10;
        use crate::data::marker::Nucleotide;

        // Mock Markers
        let mut markers = Markers::<crate::data::AnyMarkerSpace>::new();
        markers.add_chrom("chr1");

        for i in 0..n_markers {
            let m = Marker::new(
                ChromIdx::new(0),
                i as u32 * 1000,
                Some(format!("m{}", i).into()),
                Allele::Base(Nucleotide::A),
                vec![Allele::Base(Nucleotide::T)],
            );
            markers.push(m);
        }

        // Mock Samples
        let samples = Arc::new(Samples::from_ids(
            (0..n_samples).map(|i| format!("s{}", i)).collect(),
        ));

        // Mock Genotypes (Random)
        let columns: Vec<GenotypeColumn> = (0..n_markers)
            .map(|_| {
                let bytes: Vec<u8> = (0..n_samples * 2).map(|i| (i % 3) as u8).collect();
                GenotypeColumn::from_alleles(&bytes, 2)
            })
            .collect();

        let gt = GenotypeMatrix::new_unphased(markers, columns, samples);

        // Mock Genetic Map (Empty uses default linear rate)
        let gen_maps = GeneticMaps::new();

        let config = Config {
            gt: PathBuf::from("test.vcf"),
            r#ref: None,
            out: PathBuf::from("out"),
            map: None,
            chrom: None,
            excludesamples: None,
            excludemarkers: None,
            burnin: 2,
            iterations: 2,
            mcmc_burnin: 1,
            dynamic_mcmc: false,
            mcmc_steps: 3,
            mcmc_lr_samples: 32,
            phase_states: 10,
            rare: 0.002,
            impute: true,
            imp_states: 10,
            imp_segment: 6.0,
            imp_step: 0.1,
            imp_nsteps: 7,
            cluster: 0.005,
            pbwt_batch_mb: 256,
            ap: false,
            gp: false,
            ne: 10000.0,
            err: None,
            em: false,
            window: 40.0,
            window_markers: 100000,
            overlap: 2.0,
            seed: 12345,
            nthreads: Some(2),
            profile: false,
        };

        let mut pipeline = PhasingPipeline::<crate::data::AnyMarkerSpace>::new(config, None);

        // Run phasing (with no overlap from previous window)
        let result = pipeline.phase_in_memory_with_overlap(&gt, &gen_maps, None, None);

        assert!(result.is_ok());
        let (phased, _) = result.unwrap();
        assert_eq!(phased.n_markers(), n_markers);
        assert_eq!(phased.n_haplotypes(), n_samples * 2);

        // Check phase confidence values
        let mut total_hets = 0;
        let mut high_conf_hets = 0;
        let mut sum_conf = 0.0;
        let mut count_conf = 0;

        for m in 0..n_markers {
            let marker_idx = MarkerIdx::new(m as u32);
            let column = phased.column(marker_idx);

            for s in 0..n_samples {
                let sample_idx = crate::data::SampleIdx::new(s as u32);
                let hap1 = column.get(sample_idx.hap1());
                let hap2 = column.get(sample_idx.hap2());

                // Get phase confidence
                let conf = phased.sample_phase_confidence_f32(marker_idx, s);

                // Confidence must be in valid range [0.0, 1.0]
                assert!(
                    conf >= 0.0 && conf <= 1.0,
                    "Phase confidence out of range: {} at marker {} sample {}",
                    conf,
                    m,
                    s
                );

                // Track heterozygous sites
                if hap1 != hap2 {
                    total_hets += 1;
                    sum_conf += conf;
                    count_conf += 1;

                    // Count hets with high confidence (>0.7)
                    if conf > 0.7 {
                        high_conf_hets += 1;
                    }
                }
            }
        }

        // Assert that most heterozygous sites have reasonable confidence
        if total_hets > 0 {
            let mean_conf = sum_conf / count_conf as f32;
            let high_conf_ratio = high_conf_hets as f32 / total_hets as f32;

            // For this unit test with random data and minimal iterations,
            // we just verify confidence values are computed and in valid range.
            // Real integration tests with actual data should have mean_conf > 0.8
            assert!(
                mean_conf >= 0.0 && mean_conf <= 1.0,
                "Mean phase confidence out of range: {:.3}",
                mean_conf
            );

            println!(
                "Phase confidence stats: mean={:.3}, high_conf_ratio={:.1}%, n_hets={}",
                mean_conf,
                high_conf_ratio * 100.0,
                total_hets
            );
        }
    }

    #[test]
    fn test_emit_haploid_constrained_at_het() {
        // At a het site with genotype {0, 1}, if H2 is fixed to 0,
        // H1 must be 1. Emission should be high if reference has 1, low if 0.
        let p_no_err = 0.999;
        let p_err = 0.001;
        let conf = 1.0;

        // H2 = 0, so H1 must = 1. Reference has 1 -> high emission
        let emit_match = emit_haploid_constrained(1, 0, 1, 0, conf, p_no_err, p_err);
        assert!(
            emit_match > 0.9,
            "Expected high emission when ref matches required allele, got {}",
            emit_match
        );

        // H2 = 0, so H1 must = 1. Reference has 0 -> low emission
        let emit_mismatch = emit_haploid_constrained(0, 0, 1, 0, conf, p_no_err, p_err);
        assert!(
            emit_mismatch < 0.1,
            "Expected low emission when ref doesn't match, got {}",
            emit_mismatch
        );

        // At homozygous site (fixed_allele = 255), H1 must match genotype
        let emit_hom = emit_haploid_constrained(0, 0, 0, 255, conf, p_no_err, p_err);
        assert!(
            emit_hom > 0.9,
            "Expected high emission at hom site when ref matches, got {}",
            emit_hom
        );

        let emit_hom_mismatch = emit_haploid_constrained(1, 0, 0, 255, conf, p_no_err, p_err);
        assert!(
            emit_hom_mismatch < 0.1,
            "Expected low emission at hom when ref doesn't match, got {}",
            emit_hom_mismatch
        );
    }

    #[test]
    fn test_emit_haploid_constrained_confidence_blending() {
        // With low confidence, emission should be closer to 0.5
        let p_no_err = 0.999;
        let p_err = 0.001;

        // Full confidence: emission should be ~p_no_err
        let emit_full_conf = emit_haploid_constrained(1, 0, 1, 0, 1.0, p_no_err, p_err);
        assert!((emit_full_conf - p_no_err).abs() < 0.01);

        // Zero confidence: emission should be 0.5
        let emit_zero_conf = emit_haploid_constrained(1, 0, 1, 0, 0.0, p_no_err, p_err);
        assert!(
            (emit_zero_conf - 0.5).abs() < 0.01,
            "Expected 0.5 with zero confidence, got {}",
            emit_zero_conf
        );

        // Half confidence: emission should be blend
        let emit_half_conf = emit_haploid_constrained(1, 0, 1, 0, 0.5, p_no_err, p_err);
        let expected = 0.5 * p_no_err + 0.5 * 0.5;
        assert!(
            (emit_half_conf - expected).abs() < 0.01,
            "Expected {}, got {}",
            expected,
            emit_half_conf
        );
    }

    #[test]
    fn test_compute_pl_allele_probs_partner_polarity() {
        // Strong heterozygous PL: 0/1 is overwhelmingly likely.
        // PL ordering for biallelic sites is (0/0, 0/1, 1/1).
        let pl = [100u16, 0u16, 100u16];
        let mut allele_probs = Vec::new();

        let n = compute_pl_allele_probs(Some(&pl), false, 0, &mut allele_probs)
            .expect("expected biallelic PL to be parsed");
        assert_eq!(n, 2);
        assert!(
            allele_probs[1] > allele_probs[0],
            "partner=0 should favor target allele 1, got {:?}",
            allele_probs
        );

        let n = compute_pl_allele_probs(Some(&pl), false, 1, &mut allele_probs)
            .expect("expected biallelic PL to be parsed");
        assert_eq!(n, 2);
        assert!(
            allele_probs[0] > allele_probs[1],
            "partner=1 should favor target allele 0, got {:?}",
            allele_probs
        );

        // In combined mode, conditioning on partner should have no effect.
        compute_pl_allele_probs(Some(&pl), true, 0, &mut allele_probs)
            .expect("expected biallelic PL to be parsed");
        let probs_partner0 = allele_probs.clone();
        compute_pl_allele_probs(Some(&pl), true, 1, &mut allele_probs)
            .expect("expected biallelic PL to be parsed");
        let probs_partner1 = allele_probs.clone();
        assert!(
            probs_partner0
                .iter()
                .zip(probs_partner1.iter())
                .all(|(a, b)| (a - b).abs() < 1e-6),
            "combined emissions should be partner-invariant: {:?} vs {:?}",
            probs_partner0,
            probs_partner1
        );
    }

    #[test]
    fn test_prescan_hero_survives_across_sparse_anchor_windows() {
        let n_markers = 50;
        let hero_pattern: Vec<u8> = (0..n_markers).map(|m| (m % 2) as u8).collect();
        let (ref_gt, hero_hap_idx) =
            build_ref_panel_with_hero(n_markers, 50, 49, &hero_pattern, 42);
        let target_gt = build_target_with_sparse_anchors(n_markers, &hero_pattern, 10);

        let target_geno = MutableGenotypes::from_fn(n_markers, 2, |m, h| {
            target_gt.allele(MarkerIdx::new(m as u32), HapIdx::new(h as u32))
        });
        let alignment = MarkerAlignment::new(&target_gt, &ref_gt);
        let gen_positions: Vec<f64> = (0..n_markers).map(|i| i as f64 * 0.02).collect();

        let mut config = Config::default();
        config.phase_states = ref_gt.n_haplotypes();
        config.ne = 10000.0;
        config.err = Some(0.0001);
        config.nthreads = Some(1);
        let mut pipeline = PhasingPipeline::<crate::data::AnyMarkerSpace>::new(config, None);
        pipeline.params =
            ModelParams::for_phasing(ref_gt.n_haplotypes() + 2, 10000.0, Some(0.0001));
        pipeline
            .params
            .set_n_states(pipeline.config.phase_states.min(ref_gt.n_haplotypes()));

        let threaded = pipeline
            .build_phasing_prescan_states(
                &target_gt,
                &target_geno,
                Some(&ref_gt),
                Some(&alignment),
                n_markers,
                1,
                &gen_positions,
                0.1,
                None,
            )
            .expect("prescan");

        let th = &threaded[0];
        let expected_states = pipeline
            .config
            .phase_states
            .min(ref_gt.n_haplotypes())
            .max(1);
        assert_eq!(
            th.n_states(),
            expected_states,
            "ThreadedHaps state count should match phase_states cap"
        );
        let mut state_buf = vec![CombinedHapId::from(0u32); th.n_states()];
        let offset = target_geno.n_haps() as u32;
        let mut expected: Vec<u32> = (0..ref_gt.n_haplotypes())
            .map(|h| offset + h as u32)
            .collect();
        expected.sort_unstable();

        let markers = [0usize, n_markers / 2, n_markers - 1];
        for &m in &markers {
            th.materialize_at(m, &mut state_buf);
            let mut actual: Vec<u32> = state_buf.iter().map(|id| id.as_u32()).collect();
            actual.sort_unstable();
            let unique: std::collections::HashSet<u32> = actual.iter().copied().collect();
            assert_eq!(
                actual.len(),
                unique.len(),
                "ThreadedHaps contains duplicate ids at marker {}",
                m
            );
            assert_eq!(
                actual.len(),
                th.n_states(),
                "ThreadedHaps materialize_at size mismatch at marker {}",
                m
            );
            assert_eq!(
                actual, expected,
                "ThreadedHaps states differ from expected full ref set at marker {}",
                m
            );
        }
        let hero_combined = (offset as usize + hero_hap_idx) as u32;
        let contains_hero = expected.iter().any(|&id| id == hero_combined);
        println!(
            "[prescan test] states={} hero_present={}",
            th.n_states(),
            contains_hero
        );
    }

    #[test]
    fn test_sample_swap_bits_mosaic_two_state_anchor_stability() {
        let n_markers = 40;
        let hero_pattern: Vec<u8> = (0..n_markers).map(|m| (m % 2) as u8).collect();
        let (ref_gt, hero_hap_idx) = build_ref_panel_with_hero(n_markers, 1, 0, &hero_pattern, 99);

        println!("[mosaic anchor test] hero_hap_idx={}", hero_hap_idx);
        let mut th = ThreadedHaps::<CombinedHapSpace>::new(2, 2, n_markers);
        th.push_new(CombinedHapId::new(0));
        th.push_new(CombinedHapId::new(1));

        let seq1: Vec<u8> = vec![0; n_markers];
        let seq2: Vec<u8> = vec![1; n_markers];
        let conf: Vec<f32> = vec![1.0; n_markers];
        let mut anchor_h1 = vec![255u8; n_markers];
        let mut anchor_h2 = vec![255u8; n_markers];
        for m in (0..n_markers).step_by(10) {
            anchor_h1[m] = hero_pattern[m];
            anchor_h2[m] = 1 - hero_pattern[m];
        }

        let p_recomb = vec![0.001f32; n_markers];
        let block_starts: Arc<[usize]> =
            blocks_to_starts(&[(0, n_markers)], n_markers).into_boxed_slice().into();
        let het_positions: Vec<usize> = (0..n_markers).collect();
        let p_no_err = 0.999;
        let p_err = 1.0 - p_no_err;
        let mut workspace = crate::utils::workspace::ThreadWorkspace::new(8, 0);
        let ref_provider = RefAlleleProvider::new(GenotypeView::from(&ref_gt), &th);

        let (swap_bits, swap_lr, swap_probs, paths) = sample_swap_bits_mosaic(
            n_markers,
            2,
            &p_recomb,
            &seq1,
            &seq2,
            &conf,
            ref_provider,
            None,
            block_starts,
            &het_positions,
            None,
            Some(&anchor_h1),
            Some(&anchor_h2),
            123,
            0,
            32,
            p_no_err,
            p_err,
            &mut workspace,
        );

        let mut switches1 = 0usize;
        let mut switches2 = 0usize;
        for m in 1..n_markers {
            if paths.path1[m] != paths.path1[m - 1] {
                switches1 += 1;
            }
            if paths.path2[m] != paths.path2[m - 1] {
                switches2 += 1;
            }
        }
        let p_min = swap_probs
            .iter()
            .cloned()
            .fold(1.0f32, |a, b| a.min(b));
        let p_max = swap_probs
            .iter()
            .cloned()
            .fold(0.0f32, |a, b| a.max(b));
        let p_mean = swap_probs.iter().sum::<f32>() / swap_probs.len().max(1) as f32;
        println!(
            "[mosaic anchor test] switches1={} switches2={} p_min={:.3} p_mean={:.3} p_max={:.3} swap_bits={} swap_lr={}",
            switches1,
            switches2,
            p_min,
            p_mean,
            p_max,
            swap_bits.len(),
            swap_lr.len()
        );
        assert_eq!(
            switches1 + switches2,
            0,
            "Unexpected path switching with anchors in two-state model"
        );
    }

    #[test]
    fn test_prescan_scores_hero_in_anchor_window() {
        let n_markers = 50;
        let hero_pattern: Vec<u8> = (0..n_markers).map(|m| (m % 2) as u8).collect();
        let (ref_gt, hero_hap_idx) =
            build_ref_panel_with_hero(n_markers, 50, 49, &hero_pattern, 7);
        let target_gt = build_target_with_sparse_anchors(n_markers, &hero_pattern, 10);
        let target_geno = MutableGenotypes::from_fn(n_markers, 2, |m, h| {
            target_gt.allele(MarkerIdx::new(m as u32), HapIdx::new(h as u32))
        });
        let alignment = MarkerAlignment::new(&target_gt, &ref_gt);

        let gen_positions: Vec<f64> = (0..n_markers).map(|i| i as f64 * 0.02).collect();
        let windows = partition_markers_by_cm(&gen_positions, stage1_block_cm(&gen_positions));
        let window = windows[0];
        let per_window_cap = 20usize;
        let n_ref_haps = ref_gt.n_haplotypes();
        let sampling = build_sampling_points(
            &gen_positions[window.0..window.1],
            0.1,
            PBWT_MIN_MARKER_STEP,
            None,
        );
        let k_per_hap = per_window_cap
            .saturating_mul(PBWT_PER_WINDOW_MULT)
            .max(PBWT_MIN_PER_HAP)
            .min(PBWT_MAX_PER_HAP)
            .max(1)
            .min(n_ref_haps.max(1));
        let sampled = sampling.iter().filter(|&&b| b).count();
        println!(
            "[prescan score test] window={:?} sampled={}",
            window, sampled
        );

        let ref_columns: Vec<GenotypeColumn> = (0..n_markers)
            .map(|m| ref_gt.column(MarkerIdx::new(m as u32)).clone())
            .collect();
        let freqs = compute_ref_freqs(
            &target_gt,
            &ref_columns,
            Some(&alignment),
            None,
            None,
            n_markers,
        );
        let mut window_scores = vec![vec![f32::NEG_INFINITY; ref_gt.n_haplotypes()]; 2];
        score_window_batch_pbwt_segment(
            &[0, 1],
            &target_geno,
            &ref_columns,
            target_gt.phase_mask(),
            Some(&[true]),
            Some(&alignment),
            &freqs,
            window,
            k_per_hap,
            &sampling,
            &mut window_scores,
            false,
            None,
            None,
        );

        let hero_score = window_scores[0][hero_hap_idx];
        let top = select_top_k(&window_scores[0], 15);
        let in_top = top.iter().any(|(h, _)| *h == hero_hap_idx);
        println!(
            "[prescan score test] top={:?}",
            top.iter()
                .map(|(h, s)| (*h, (*s * 1000.0).round() / 1000.0))
                .collect::<Vec<_>>()
        );
        println!(
            "[prescan score test] hero_hap={} score={:.3} in_top10={}",
            hero_hap_idx, hero_score, in_top
        );
        assert!(
            in_top,
            "Hero hap {} not in top-10 prescan scores for anchor window",
            hero_hap_idx
        );
    }

    #[test]
    fn test_allocator_keeps_hero_from_prescan_scores() {
        let n_markers = 50;
        let hero_pattern: Vec<u8> = (0..n_markers).map(|m| (m % 2) as u8).collect();
        let (ref_gt, hero_hap_idx) =
            build_ref_panel_with_hero(n_markers, 50, 49, &hero_pattern, 11);
        let target_gt = build_target_with_sparse_anchors(n_markers, &hero_pattern, 10);
        let target_geno = MutableGenotypes::from_fn(n_markers, 2, |m, h| {
            target_gt.allele(MarkerIdx::new(m as u32), HapIdx::new(h as u32))
        });
        let alignment = MarkerAlignment::new(&target_gt, &ref_gt);
        let gen_positions: Vec<f64> = (0..n_markers).map(|i| i as f64 * 0.02).collect();
        let windows = partition_markers_by_cm(&gen_positions, stage1_block_cm(&gen_positions));
        let num_windows = windows.len();
        let per_window_cap = 20usize;

        let n_ref_haps = ref_gt.n_haplotypes();
        let ref_columns: Vec<GenotypeColumn> = (0..n_markers)
            .map(|m| ref_gt.column(MarkerIdx::new(m as u32)).clone())
            .collect();
        let freqs = compute_ref_freqs(
            &target_gt,
            &ref_columns,
            Some(&alignment),
            None,
            None,
            n_markers,
        );

        let mut scores_by_window_by_hap: Vec<Vec<Vec<(usize, f32)>>> =
            vec![Vec::with_capacity(num_windows); 2];
        let phase_mask = target_gt.phase_mask();
        let mut informative: Vec<bool> = Vec::new();
        for &(start, end) in &windows {
            informative.clear();
            informative.resize(end.saturating_sub(start), false);
            if !informative.is_empty() {
                for m in start..end {
                    let mut info = false;
                    if let Some(mask) = phase_mask {
                        if let Some(row) = mask.get(m) {
                            if row.iter().any(|&v| v != 0) {
                                info = true;
                            }
                        }
                    }
                    if !info {
                        let alleles = target_geno.marker_alleles(m);
                        if alleles.iter().any(|&a| a != 255) {
                            info = true;
                        }
                    }
                    informative[m - start] = info;
                }
            }
            let sampling = build_sampling_points(
                &gen_positions[start..end],
                0.1,
                PBWT_MIN_MARKER_STEP,
                Some(&informative),
            );
            let k_per_hap = per_window_cap
                .saturating_mul(PBWT_PER_WINDOW_MULT)
                .max(PBWT_MIN_PER_HAP)
                .min(PBWT_MAX_PER_HAP)
                .max(1)
                .min(n_ref_haps.max(1));
            let mut window_scores = vec![vec![f32::NEG_INFINITY; n_ref_haps]; 2];
            score_window_batch_pbwt_segment(
                &[0, 1],
                &target_geno,
                &ref_columns,
                phase_mask,
                Some(&[true]),
                Some(&alignment),
                &freqs,
                (start, end),
                k_per_hap,
                &sampling,
                &mut window_scores,
                false,
                None,
                None,
            );
            for hap_idx in 0..2 {
                let top = select_top_k(&window_scores[hap_idx], 160.min(n_ref_haps.max(1)));
                scores_by_window_by_hap[hap_idx].push(top);
            }
        }

        let mut dense_merge_buffer = vec![f32::NEG_INFINITY; n_ref_haps.max(1)];
        let mut touched_indices: Vec<usize> = Vec::new();
        let mut window_scores: Vec<Vec<(usize, f32)>> = Vec::with_capacity(num_windows);
        let mut prev_window_scores: Vec<(usize, f32)> = Vec::new();
        for w in 0..num_windows {
            for &idx in &touched_indices {
                dense_merge_buffer[idx] = f32::NEG_INFINITY;
            }
            touched_indices.clear();

            for &(h, score) in scores_by_window_by_hap[0][w]
                .iter()
                .chain(scores_by_window_by_hap[1][w].iter())
            {
                if h >= dense_merge_buffer.len() {
                    continue;
                }
                let current = &mut dense_merge_buffer[h];
                if current.is_finite() {
                    if score > *current {
                        *current = score;
                    }
                } else {
                    *current = score;
                    touched_indices.push(h);
                }
            }

            touched_indices.sort_by(|&a, &b| {
                dense_merge_buffer[b]
                    .partial_cmp(&dense_merge_buffer[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let cap = 160.min(n_ref_haps.max(1));
            let take = cap.min(touched_indices.len());
            let mut list: Vec<(usize, f32)> = Vec::with_capacity(take);
            for &h in touched_indices.iter().take(take) {
                list.push((h, dense_merge_buffer[h]));
            }
            if list.is_empty() {
                list.extend(prev_window_scores.iter().copied());
            } else if !prev_window_scores.is_empty() {
                let mut map: HashMap<usize, f32> = list.iter().copied().collect();
                for (h, score) in prev_window_scores.iter().copied() {
                    map.entry(h).or_insert(score);
                }
                list = map.into_iter().collect();
            }
            prev_window_scores = list.clone();
            window_scores.push(list);
        }

        let abyss = vec![false; n_ref_haps];
        let (candidate_haps, scores_by_hap) = build_sparse_scores(&window_scores, &abyss);
        let hero_in_candidates = candidate_haps.iter().any(|&h| h == hero_hap_idx);
        println!(
            "[allocator test] candidates={} hero_in_candidates={}",
            candidate_haps.len(),
            hero_in_candidates
        );
        assert!(
            hero_in_candidates,
            "Hero hap {} missing from candidate set before allocation",
            hero_hap_idx
        );

        let per_window_cap = 20usize;
        let per_window_caps = vec![per_window_cap; num_windows];
        let global_slot_budget = per_window_caps.iter().copied().sum::<usize>().max(1);
        let mut boundary_cm = Vec::with_capacity(num_windows.saturating_sub(1));
        for w in 0..num_windows.saturating_sub(1) {
            let (_, end) = windows[w];
            let (next_start, _) = windows[w + 1];
            let left = gen_positions[end.saturating_sub(1).min(gen_positions.len() - 1)];
            let right = gen_positions[next_start.min(gen_positions.len() - 1)];
            boundary_cm.push((right - left).abs().max(0.1));
        }
        let mut config = Config::default();
        config.phase_states = 20;
        config.ne = 10000.0;
        config.err = Some(0.0001);
        config.nthreads = Some(1);
        let mut pipeline = PhasingPipeline::<crate::data::AnyMarkerSpace>::new(config, None);
        pipeline.params = ModelParams::for_phasing(n_ref_haps + 2, 10000.0, Some(0.0001));
        let params = pipeline.params.clone();
        let allocation = allocate_lms_sparse(
            &scores_by_hap,
            &candidate_haps,
            num_windows,
            &boundary_cm,
            &params,
            n_ref_haps,
            global_slot_budget,
            &per_window_caps,
        );
        let mut selected: Vec<usize> =
            allocation.intervals_by_hap.into_iter().map(|(h, _)| h).collect();
        selected.sort_unstable();
        selected.dedup();
        eprintln!(
            "[threaded test] expected_ref_len={} first={:?}",
            selected.len(),
            selected.iter().take(10).collect::<Vec<_>>()
        );
        eprintln!(
            "[threaded test] expected_ref_len={} first={:?}",
            selected.len(),
            selected.iter().take(10).collect::<Vec<_>>()
        );

        if PBWT_FORCE_TOP_HAPS > 0 && !window_scores.is_empty() {
            let mut dense_merge_buffer = vec![f32::NEG_INFINITY; n_ref_haps.max(1)];
            let mut touched_indices: Vec<usize> = Vec::new();
            for list in &window_scores {
                for &(h, score) in list {
                    if h >= dense_merge_buffer.len() {
                        continue;
                    }
                    let current = &mut dense_merge_buffer[h];
                    if current.is_finite() {
                        if score > *current {
                            *current = score;
                        }
                    } else {
                        *current = score;
                        touched_indices.push(h);
                    }
                }
            }
            touched_indices.sort_by(|&a, &b| {
                dense_merge_buffer[b]
                    .partial_cmp(&dense_merge_buffer[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let take = PBWT_FORCE_TOP_HAPS.min(touched_indices.len());
            for &h in touched_indices.iter().take(take) {
                if !selected.contains(&h) {
                    selected.push(h);
                }
            }
        }

        if PBWT_ANCHOR_TOP_HAPS > 0 {
            let mut anchors_by_hap: Vec<Vec<(usize, u8, u8)>> = vec![Vec::new(); 2];
            let phase_mask = target_gt.phase_mask();
            for s in 0..1usize {
                let hap1 = s * 2;
                let hap2 = hap1 + 1;
                for m in 0..n_markers {
                    let phased = phase_mask
                        .and_then(|mask| mask.get(m).and_then(|row| row.get(s)))
                        .copied()
                        .unwrap_or(0);
                    if phased == 0 {
                        continue;
                    }
                    let a1 = target_geno.get(m, HapIdx::new(hap1 as u32));
                    let a2 = target_geno.get(m, HapIdx::new(hap2 as u32));
                    if a1 == 255 || a2 == 255 || a1 == a2 {
                        continue;
                    }
                    anchors_by_hap[hap1].push((m, a1, a2));
                    anchors_by_hap[hap2].push((m, a2, a1));
                }
            }

            let mut anchor_scores = vec![0i32; n_ref_haps];
            for &(start, end) in &windows {
                let mut window_anchors: Vec<(usize, u8, u8)> = Vec::new();
                for &(m, a1, a2) in anchors_by_hap[0]
                    .iter()
                    .chain(anchors_by_hap[1].iter())
                {
                    if m >= start && m < end {
                        window_anchors.push((m, a1, a2));
                    }
                }
                if window_anchors.is_empty() {
                    continue;
                }
                anchor_scores.fill(0);
                for h in 0..n_ref_haps {
                    let hap_idx = HapIdx::new(h as u32);
                    let mut score = 0i32;
                    for (m, a1, _) in &window_anchors {
                        let ref_al = ref_columns
                            .get(*m)
                            .map(|c| c.get(hap_idx))
                            .unwrap_or(255);
                        if ref_al == 255 {
                            continue;
                        }
                        if ref_al == *a1 {
                            score += 1;
                        } else {
                            score -= 1;
                        }
                    }
                    anchor_scores[h] = score;
                }
                let mut idxs: Vec<usize> = (0..n_ref_haps).collect();
                idxs.sort_by(|&a, &b| anchor_scores[b].cmp(&anchor_scores[a]));
                let take = PBWT_ANCHOR_TOP_HAPS.min(idxs.len());
                for &h in idxs.iter().take(take) {
                    if !selected.contains(&h) {
                        selected.push(h);
                    }
                }
            }
        }
        let hero_selected = selected.iter().any(|&h| h == hero_hap_idx);
        println!(
            "[allocator test] selected={} hero_selected={}",
            selected.len(),
            hero_selected
        );
        assert!(
            hero_selected,
            "Hero hap {} dropped by allocator despite positive scores",
            hero_hap_idx
        );
    }

    #[test]
    fn test_threaded_haps_matches_selected_indices() {
        let n_markers = 50;
        let hero_pattern: Vec<u8> = (0..n_markers).map(|m| (m % 2) as u8).collect();
        let (ref_gt, hero_hap_idx) =
            build_ref_panel_with_hero(n_markers, 50, 49, &hero_pattern, 13);
        let target_gt = build_target_with_sparse_anchors(n_markers, &hero_pattern, 10);
        let target_geno = MutableGenotypes::from_fn(n_markers, 2, |m, h| {
            target_gt.allele(MarkerIdx::new(m as u32), HapIdx::new(h as u32))
        });
        let alignment = MarkerAlignment::new(&target_gt, &ref_gt);
        let gen_positions: Vec<f64> = (0..n_markers).map(|i| i as f64 * 0.02).collect();

        let mut config = Config::default();
        config.phase_states = ref_gt.n_haplotypes();
        config.ne = 10000.0;
        config.err = Some(0.0001);
        config.nthreads = Some(1);
        let mut pipeline = PhasingPipeline::<crate::data::AnyMarkerSpace>::new(config, None);
        pipeline.params =
            ModelParams::for_phasing(ref_gt.n_haplotypes() + 2, 10000.0, Some(0.0001));
        pipeline
            .params
            .set_n_states(pipeline.config.phase_states.min(ref_gt.n_haplotypes()));

        let threaded = pipeline
            .build_phasing_prescan_states(
                &target_gt,
                &target_geno,
                Some(&ref_gt),
                Some(&alignment),
                n_markers,
                1,
                &gen_positions,
                0.1,
                None,
            )
            .expect("prescan");
        let th = &threaded[0];
        let expected_states = pipeline
            .config
            .phase_states
            .min(ref_gt.n_haplotypes())
            .max(1);
        assert_eq!(
            th.n_states(),
            expected_states,
            "ThreadedHaps state count should match phase_states cap"
        );
        let mut state_buf = vec![CombinedHapId::from(0u32); th.n_states()];
        let offset = target_geno.n_haps() as u32;
        let mut expected: Vec<u32> = (0..ref_gt.n_haplotypes())
            .map(|h| offset + h as u32)
            .collect();
        expected.sort_unstable();
        let markers = [0usize, n_markers / 2, n_markers - 1];
        for &m in &markers {
            th.materialize_at(m, &mut state_buf);
            let mut actual: Vec<u32> = state_buf.iter().map(|id| id.as_u32()).collect();
            actual.sort_unstable();
            let unique: std::collections::HashSet<u32> = actual.iter().copied().collect();
            assert_eq!(
                actual.len(),
                unique.len(),
                "ThreadedHaps contains duplicate ids at marker {}",
                m
            );
            assert_eq!(
                actual.len(),
                th.n_states(),
                "ThreadedHaps materialize_at size mismatch at marker {}",
                m
            );
            assert_eq!(
                actual, expected,
                "ThreadedHaps states differ from expected full ref set at marker {}",
                m
            );
        }
        let hero_combined = (offset as usize + hero_hap_idx) as u32;
        let contains_hero = expected.iter().any(|&id| id == hero_combined);
        println!(
            "[threaded test] states={} hero_present={}",
            th.n_states(),
            contains_hero
        );
    }

    #[test]
    fn test_refresh_path_ref_from_states_updates_all_valid_markers() {
        let mut path_ref = vec![0u32, 0u32, 0u32, 0u32];
        let path_idx = vec![0u32, 1u32, 2u32, 1u32];
        let neighbors = vec![10u32, 11u32];

        refresh_path_ref_from_states(&mut path_ref, &path_idx, &neighbors);

        assert_eq!(path_ref[0], 10);
        assert_eq!(path_ref[1], 11);
        // Invalid state index should leave the previous value intact.
        assert_eq!(path_ref[2], 0);
        assert_eq!(path_ref[3], 11);
    }

    #[test]
    fn test_dynamic_mcmc_deterministic_phase() {
        // Create a scenario where the correct phase is deterministic:
        // Target sample (haps 0-1) with het genotype {0, 1}
        // Reference haplotypes (haps 2-9) all have allele 0
        // The HMM should set H1 = 0 (matching reference majority)
        use crate::model::ibs2::Ibs2;
        use crate::model::phase_ibs::BidirectionalPhaseIbs;

        let n_markers = 10;
        let n_target_haps = 2; // Sample 0: haplotypes 0 and 1
        let n_ref_haps = 8; // Reference: haplotypes 2-9
        let n_total_haps = n_target_haps + n_ref_haps;

        // Build PBWT with target + reference
        // Target haps (0, 1): missing (255) - we're phasing these
        // Reference haps (2-9): all have allele 0
        let alleles: Vec<Vec<u8>> = (0..n_markers)
            .map(|_| {
                let mut haps = vec![255u8; n_total_haps]; // Start with missing
                for h in n_target_haps..n_total_haps {
                    haps[h] = 0; // Reference haplotypes have allele 0
                }
                haps
            })
            .collect();
        let subset_to_global: Vec<usize> = (0..n_markers).collect();
        let phase_ibs = BidirectionalPhaseIbs::build_for_subset(
            alleles,
            n_total_haps,
            n_markers,
            &subset_to_global,
        );

        // Empty IBS2 - need at least 1 sample for the structure
        let ibs2 = Ibs2::empty(1);

        // Genotype: het at all sites (0/1)
        let seq1 = vec![0u8; n_markers];
        let seq2 = vec![1u8; n_markers];
        let conf = vec![1.0f32; n_markers];

        // p_recomb: low recombination
        let p_recomb = vec![0.01f32; n_markers];

        let het_positions: Vec<usize> = (0..n_markers).collect();

        // Sample 0: haplotypes 0 and 1
        let mut workspace = crate::utils::workspace::ThreadWorkspace::new(64, 0);
        let (swap_bits, swap_lr, swap_probs, paths) = sample_dynamic_mcmc(
            n_markers,
            n_total_haps,
            &p_recomb,
            &seq1,
            &seq2,
            &conf,
            &phase_ibs,
            &ibs2,
            0, // sample_idx = 0 (haplotypes 0 and 1)
            &het_positions,
            12345, // seed
            5,     // n_mcmc_steps
            0.999,
            0.001,
            None,
            None,
            None,
            &mut workspace,
        );
        assert_eq!(paths.path1.len(), n_markers);
        assert_eq!(paths.path2.len(), n_markers);

        // With all reference having allele 0, H1 should be set to 0 at all hets.
        // Since seq1 = 0, this means no swap (swap_bit = 0).
        let n_swaps: usize = swap_bits.iter().map(|&b| b as usize).sum();

        // We expect very few or no swaps since reference strongly supports H1 = 0
        assert!(
            n_swaps <= 2,
            "Expected <=2 swaps with consistent reference, got {} swaps out of {} hets",
            n_swaps,
            het_positions.len()
        );

        // LR should be high confidence
        assert_eq!(swap_lr.len(), het_positions.len());
        assert!(swap_probs.len() <= het_positions.len());
    }

    #[test]
    fn test_dynamic_mcmc_opposite_phase() {
        // Target sample (haps 0-1) with het genotype {0, 1}
        // Reference haplotypes (haps 2-9) all have allele 1
        // The HMM should set H1 = 1 (matching reference) -> swap needed
        use crate::model::ibs2::Ibs2;
        use crate::model::phase_ibs::BidirectionalPhaseIbs;

        let n_markers = 10;
        let n_target_haps = 2; // Sample 0: haplotypes 0 and 1
        let n_ref_haps = 8; // Reference: haplotypes 2-9
        let n_total_haps = n_target_haps + n_ref_haps;

        // Build PBWT with target + reference
        // Target haps (0, 1): missing (255)
        // Reference haps (2-9): all have allele 1
        let alleles: Vec<Vec<u8>> = (0..n_markers)
            .map(|_| {
                let mut haps = vec![255u8; n_total_haps];
                for h in n_target_haps..n_total_haps {
                    haps[h] = 1; // Reference haplotypes have allele 1
                }
                haps
            })
            .collect();
        let subset_to_global: Vec<usize> = (0..n_markers).collect();
        let phase_ibs = BidirectionalPhaseIbs::build_for_subset(
            alleles,
            n_total_haps,
            n_markers,
            &subset_to_global,
        );

        let ibs2 = Ibs2::empty(1);

        // Genotype: het at all sites (0/1)
        let seq1 = vec![0u8; n_markers];
        let seq2 = vec![1u8; n_markers];
        let conf = vec![1.0f32; n_markers];
        let p_recomb = vec![0.01f32; n_markers];
        let het_positions: Vec<usize> = (0..n_markers).collect();

        let mut workspace = crate::utils::workspace::ThreadWorkspace::new(64, 0);
        let (swap_bits, swap_lr, swap_probs, paths) = sample_dynamic_mcmc(
            n_markers,
            n_total_haps,
            &p_recomb,
            &seq1,
            &seq2,
            &conf,
            &phase_ibs,
            &ibs2,
            0, // sample_idx = 0 (haplotypes 0 and 1)
            &het_positions,
            12345,
            5,
            0.999,
            0.001,
            None,
            None,
            None,
            &mut workspace,
        );
        assert_eq!(paths.path1.len(), n_markers);
        assert_eq!(paths.path2.len(), n_markers);

        // With all reference having allele 1, H1 should be set to 1 at all hets.
        // Since seq1 = 0, this means swap (swap_bit = 1).
        let n_swaps: usize = swap_bits.iter().map(|&b| b as usize).sum();

        // We expect most or all to swap since reference strongly supports H1 = 1
        assert!(
            n_swaps >= n_markers - 2,
            "Expected >={} swaps with opposite reference, got {} swaps",
            n_markers - 2,
            n_swaps
        );

        // Verify LR values exist
        assert_eq!(swap_lr.len(), het_positions.len());
        assert!(swap_probs.len() <= het_positions.len());
    }

    #[test]
    fn test_find_best_constant_pair() {
        use crate::data::storage::MutableGenotypes;
        use crate::model::states::ThreadedHaps;

        let n_markers = 3;
        let n_states = 4;

        // Mock lookup
        // State 0: 0, 0, 0 (Matches Hero)
        // State 1: 1, 1, 1 (Matches Anti-Hero)
        // State 2: 0, 1, 0
        // State 3: 1, 0, 1

        // Target: 0/1 (Het) everywhere.
        // Seq1: 0, 0, 0
        // Seq2: 1, 1, 1
        // (This is one possible phasing of 0/1)

        let mut data = Vec::new();
        // M0
        data.extend_from_slice(&[0, 1, 0, 1]);
        // M1
        data.extend_from_slice(&[0, 1, 1, 0]);
        // M2
        data.extend_from_slice(&[0, 1, 0, 1]);

        let geno = MutableGenotypes::from_fn(n_markers, n_states, |m, h| {
            data[m * n_states + h]
        });
        let mut threaded = ThreadedHaps::<CombinedHapSpace>::new(n_states, n_states, n_markers);
        for h in 0..n_states {
            threaded.push_new(CombinedHapId::new(h as u32));
        }
        let mut ref_provider: RefAlleleProvider<'_, AnyMarkerSpace, AnyMarkerSpace> =
            RefAlleleProvider::new(GenotypeView::Mutable(&geno), &threaded);

        let seq1 = vec![0, 0, 0];
        let seq2 = vec![1, 1, 1];

        let mut scores = Vec::new();
        let paths = find_best_constant_pair_with_buffer(
            n_markers,
            n_states,
            &seq1,
            &seq2,
            &mut ref_provider,
            &mut scores,
        )
        .unwrap();

        // Best pair should be (0, 1) or (1, 0) - Score 3.
        // Or (2, 3) / (3, 2).

        println!("Selected pair: ({}, {})", paths.path1[0], paths.path2[0]);

        assert!(
            (paths.path1[0] == 1 && paths.path2[0] == 0)
                || (paths.path1[0] == 0 && paths.path2[0] == 1)
                || (paths.path1[0] == 3 && paths.path2[0] == 2)
                || (paths.path1[0] == 2 && paths.path2[0] == 3)
        );
    }
}
