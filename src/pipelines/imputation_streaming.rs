//! Streaming Imputation Pipeline
//!
//! Implements memory-efficient streaming imputation through overlapping windows.
//! Uses a producer-consumer model with MPSC channel to pipe phased matrices
//! directly to imputation in-memory.

use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::collections::HashMap;

use rayon::prelude::*;
use sysinfo::System;
use tracing::{info_span, instrument, warn};

use crate::data::alignment::MarkerAlignment;
use crate::data::genetic_map::GeneticMaps;
use crate::data::storage::{GenotypeColumn, GenotypeMatrix};
use crate::data::storage::phase_state::{Phased, PhaseState};
use crate::data::{ChromIdx, HapIdx, MarkerIdx, SampleIdx};
use crate::data::marker::{AnyMarkerSpace, RefWindowSpace};
use crate::error::ReagleError;
use crate::error::Result;
use crate::Config;
use crate::io::bref3::{RefPanelReader, RefWindow, TargetMarkerIndex, convert_ref_vcf_to_bref3};
use crate::io::prescan_cache::{
    create_temp_cache_path, pack_ref_columns, PackedRefColumn, PrescanCacheReader,
    PrescanCacheWriter,
};
use crate::io::streaming::{
    GlobalHapId, HaplotypePriors, PhasedOverlap, StreamingConfig, StreamingVcfReader,
};
use crate::io::vcf::{ImputationQuality, VcfWriter};
use crate::model::parameters::ModelParams;
use crate::model::pl_emission::{
    allele_probs_cond_from_pl, allele_probs_uncond_from_pl, genotype_probs_from_pl,
    infer_n_alleles_from_pl_len,
};
use crate::model::reference_pbwt::{PbwtQueryAllele, PbwtStrictAllele, RankBeam, ReferencePbwt};
use crate::model::types::RefHapId;
use crate::model::impute_hmm::{
    ImputeWorkspace, RefAlleleFreqs, TargetAlleleProbs, run_impute_hmm, state_posteriors_to_priors,
};
use crate::model::transition_matrix::TransitionMatrix;
use crate::pipelines::imputation::AllelePosteriors;
use crate::utils::telemetry::TelemetryBlackboard;


fn push_unique(dst: &mut Vec<String>, value: String) {
    if !dst.iter().any(|v| v == &value) {
        dst.push(value);
    }
}


fn chrom_variants(chrom: &str) -> Vec<String> {
    let mut candidates = Vec::new();
    push_unique(&mut candidates, chrom.to_string());
    let lower = chrom.to_ascii_lowercase();
    if lower.starts_with("chr") && chrom.len() >= 3 {
        let stripped = chrom[3..].to_string();
        if !stripped.is_empty() {
            push_unique(&mut candidates, stripped.clone());
            push_unique(&mut candidates, format!("chr{}", stripped));
            push_unique(&mut candidates, format!("CHR{}", stripped));
        }
    } else {
        push_unique(&mut candidates, format!("chr{}", chrom));
        push_unique(&mut candidates, format!("CHR{}", chrom));
    }
    candidates
}

#[inline]
fn fill_ref_alleles(col: &GenotypeColumn, out: &mut [u8]) {
    let n = out.len();
    for (i, v) in out.iter_mut().enumerate().take(n) {
        *v = col.get(HapIdx::new(i as u32));
    }
}

#[inline]
fn allele_to_posterior(n_alleles: usize, allele: u8) -> AllelePosteriors {
    if n_alleles <= 2 {
        let p_alt = if allele == 1 { 1.0 } else if allele == 0 { 0.0 } else { 0.5 };
        AllelePosteriors::Biallelic(p_alt)
    } else {
        let mut probs = vec![0.0f32; n_alleles.max(1)];
        if (allele as usize) < n_alleles {
            probs[allele as usize] = 1.0;
        } else {
            let uniform = 1.0 / n_alleles.max(1) as f32;
            for p in &mut probs {
                *p = uniform;
            }
        }
        AllelePosteriors::Multiallelic(probs)
    }
}

const PBWT_SELECT_BLOCK_CM: f64 = 0.1;
const PBWT_PER_WINDOW_MULT: usize = 8;
const PBWT_MIN_PER_HAP: usize = 64;
const PBWT_MAX_PER_HAP: usize = 256;
const ABYSS_RANK_BASE: usize = 60;
const IMPUTE_RAM_FRACTION: f64 = 0.25;
const STATE_BUDGET_SAFETY: f64 = 0.6;
const SM_MATCH_DONORS: usize = 16;
const SM_MATCH_LOW_CONF_FRAC: f32 = 0.02;
const SM_MATCH_MIN_DONORS: usize = 2;
const FULL_PANEL_RAM_FRACTION: f64 = 0.9;
const SCAN_RAM_FRACTION: f64 = 0.10;
const TARGET_CACHE_RAM_FRACTION: f64 = 0.10;
const REF_PANEL_RAM_FRACTION: f64 = 0.75;
const EXACT_PRESCAN_MAX_OPS: u128 = 250_000_000;
const MIN_AVAIL_BYTES_FOR_PLANNING: u64 = 64 * 1024 * 1024;

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
    // sysinfo reports memory values in bytes.
    let mut avail_bytes = sys.available_memory();
    let mut total_bytes = sys.total_memory();
    // Some sysinfo versions report memory in KiB. Detect and normalize.
    // Heuristic: if total < 1 GiB but total*1024 looks like a plausible RAM size,
    // treat the values as KiB. This avoids collapsing available memory to ~0.
    if total_bytes > 0 {
        let scaled_total = total_bytes.saturating_mul(1024);
        let looks_like_kib =
            total_bytes < 1_073_741_824 && scaled_total >= 1_073_741_824 && scaled_total <= (1u64 << 50);
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

fn estimate_state_budget(
    available_bytes: u64,
    n_threads: usize,
    window_markers: usize,
) -> usize {
    if available_bytes == 0 || n_threads == 0 || window_markers == 0 {
        return 0;
    }
    // Per-state memory: fwd + bwd + emissions + weights (4 * f32),
    // plus per-marker fwd_history (f32) and ref_alleles (u8).
    let per_state_bytes = 16usize.saturating_add(window_markers.saturating_mul(5));
    if per_state_bytes == 0 {
        return 0;
    }
    let budget = (available_bytes as f64 * IMPUTE_RAM_FRACTION) as u64;
    let per_thread = budget / n_threads.max(1) as u64;
    let safe_bytes = (per_thread as f64 * STATE_BUDGET_SAFETY) as u64;
    (safe_bytes as usize) / per_state_bytes
}


#[derive(Clone, Debug)]
struct ImputationPlan {
    n_ref_haps: usize,
    core_states: Vec<Vec<RefHapId>>,          // per target hap (derived)
    window_intervals: Vec<Vec<HapIntervals>>, // per target hap (sparse)
    abyss_mask: Vec<Vec<bool>>,               // per target hap
    per_window_cap: usize,
    per_window_caps: Vec<usize>, // per window (global, same for all target haps)
    full_panel: bool,
    stats: ImputationPlanStats,
}

#[derive(Clone, Debug)]
struct HapIntervals {
    hap: RefHapId,
    intervals: Vec<(u32, u32)>,
}

impl HapIntervals {
    fn contains(&self, window_idx: usize) -> bool {
        let idx = window_idx as u32;
        for &(start, end) in self.intervals.iter() {
            if idx >= start && idx <= end {
                return true;
            }
        }
        false
    }
}

#[derive(Clone, Debug, Default)]
struct ImputationPlanStats {
    haps: usize,
    core_min: usize,
    core_max: usize,
    core_sum: usize,
    dynamic_min: usize,
    dynamic_max: usize,
    dynamic_sum: usize,
    abyss_min: usize,
    abyss_max: usize,
    abyss_sum: usize,
}

impl ImputationPlanStats {
    fn update(&mut self, core: usize, dynamic: usize, abyss: usize) {
        if self.haps == 0 {
            self.core_min = core;
            self.core_max = core;
            self.dynamic_min = dynamic;
            self.dynamic_max = dynamic;
            self.abyss_min = abyss;
            self.abyss_max = abyss;
        } else {
            self.core_min = self.core_min.min(core);
            self.core_max = self.core_max.max(core);
            self.dynamic_min = self.dynamic_min.min(dynamic);
            self.dynamic_max = self.dynamic_max.max(dynamic);
            self.abyss_min = self.abyss_min.min(abyss);
            self.abyss_max = self.abyss_max.max(abyss);
        }

        self.core_sum += core;
        self.dynamic_sum += dynamic;
        self.abyss_sum += abyss;
        self.haps += 1;
    }
}

fn log_imputation_plan_summary(plan: &ImputationPlan) {
    let n_target_haps = plan.core_states.len();
    if n_target_haps == 0 {
        eprintln!("Imputation plan: no target haplotypes");
        return;
    }
    let n_windows = plan.per_window_caps.len();
    let (core_min, core_avg, core_max, dynamic_min, dynamic_avg, dynamic_max, abyss_min, abyss_avg, abyss_max) =
        if plan.stats.haps == n_target_haps && plan.stats.haps > 0 {
            let denom = plan.stats.haps as f64;
            (
                plan.stats.core_min,
                plan.stats.core_sum as f64 / denom,
                plan.stats.core_max,
                plan.stats.dynamic_min,
                plan.stats.dynamic_sum as f64 / denom,
                plan.stats.dynamic_max,
                plan.stats.abyss_min,
                plan.stats.abyss_sum as f64 / denom,
                plan.stats.abyss_max,
            )
        } else {
            let mut core_min = usize::MAX;
            let mut core_max = 0usize;
            let mut core_sum = 0usize;
            let mut dynamic_min = usize::MAX;
            let mut dynamic_max = 0usize;
            let mut dynamic_sum = 0usize;
            let mut abyss_min = usize::MAX;
            let mut abyss_max = 0usize;
            let mut abyss_sum = 0usize;

            for hap_idx in 0..n_target_haps {
                let core = plan.core_states.get(hap_idx).map(|v| v.len()).unwrap_or(0);
                let intervals = plan
                    .window_intervals
                    .get(hap_idx)
                    .map(|v| v.len())
                    .unwrap_or(0);
                let dynamic = intervals.saturating_sub(core);
                let abyss = plan
                    .abyss_mask
                    .get(hap_idx)
                    .map(|v| v.iter().filter(|&&b| b).count())
                    .unwrap_or(0);

                core_min = core_min.min(core);
                core_max = core_max.max(core);
                core_sum += core;

                dynamic_min = dynamic_min.min(dynamic);
                dynamic_max = dynamic_max.max(dynamic);
                dynamic_sum += dynamic;

                abyss_min = abyss_min.min(abyss);
                abyss_max = abyss_max.max(abyss);
                abyss_sum += abyss;
            }

            let denom = n_target_haps as f64;
            (
                core_min,
                core_sum as f64 / denom,
                core_max,
                dynamic_min,
                dynamic_sum as f64 / denom,
                dynamic_max,
                abyss_min,
                abyss_sum as f64 / denom,
                abyss_max,
            )
        };

    eprintln!(
        "Imputation plan hap counts (target_haps={}, windows={}): core_global[min/avg/max]={}/{:.1}/{}, dynamic_window[min/avg/max]={}/{:.1}/{}, abyss[min/avg/max]={}/{:.1}/{}",
        n_target_haps,
        n_windows,
        core_min,
        core_avg,
        core_max,
        dynamic_min,
        dynamic_avg,
        dynamic_max,
        abyss_min,
        abyss_avg,
        abyss_max
    );
}

fn estimate_scan_batch_size(
    available_bytes: u64,
    n_ref_haps: usize,
    n_target_haps: usize,
) -> usize {
    if available_bytes == 0 || n_ref_haps == 0 || n_target_haps == 0 {
        return 1;
    }
    // global_scores + window_scores + best_window_scores + window_rank_hits
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

fn build_sampling_points(gen_positions: &[f64], step_cm: f64) -> Vec<bool> {
    let n = gen_positions.len();
    let mut sampling = vec![false; n];
    if n == 0 {
        return sampling;
    }
    let step = step_cm.max(1e-6);
    let mut next_cm = gen_positions[0];
    for m in 0..n {
        let cm = gen_positions[m];
        if cm >= next_cm {
            sampling[m] = true;
            next_cm = cm + step;
        }
    }
    sampling[n - 1] = true;
    sampling
}

fn window_boundaries_from_handoff(handoff: &[(f64, f64)], min_step_cm: f64) -> Vec<f64> {
    if handoff.len() < 2 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(handoff.len() - 1);
    for i in 0..handoff.len() - 1 {
        let (prev_left, _) = handoff[i];
        let (_, next_right) = handoff[i + 1];
        // Use handoff anchor distance in cM (gen_pos is cM here); enforce
        // a small minimum to avoid
        // zero-distance degeneracy across overlapping windows.
        let dist = (next_right - prev_left).max(min_step_cm);
        out.push(dist);
    }
    out
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

fn compute_target_freqs_packed<TargetSpace, RefSpace>(
    target_gt: &GenotypeMatrix<Phased, TargetSpace>,
    ref_columns: &[PackedRefColumn],
    n_ref_haps: usize,
    alignment: &MarkerAlignment<TargetSpace, RefSpace>,
) -> Vec<Vec<f32>> {
    let n_markers = target_gt.n_markers();
    let mut freqs: Vec<Vec<f32>> = Vec::with_capacity(n_markers);
    for m in 0..n_markers {
        let n_alleles = target_gt
            .markers()
            .marker(MarkerIdx::new(m as u32))
            .n_alleles();
        let mut counts = vec![0u32; n_alleles.max(1)];
        let mut total = 0u32;
        if let Some(ref_m) = alignment.target_to_ref(MarkerIdx::new(m as u32)) {
            let col = &ref_columns[ref_m.as_usize()];
            if let Some((zeros, ones, missing)) = col.counts_biallelic() {
                let map0 = alignment.reverse_map_allele(m, 0);
                let map1 = alignment.reverse_map_allele(m, 1);
                if map0 != 255 {
                    let idx = map0 as usize;
                    if idx < counts.len() {
                        counts[idx] += zeros as u32;
                    }
                }
                if map1 != 255 {
                    let idx = map1 as usize;
                    if idx < counts.len() {
                        counts[idx] += ones as u32;
                    }
                }
                total = (n_ref_haps.saturating_sub(missing)) as u32;
            } else {
                for rh in 0..n_ref_haps {
                    let ref_a = col.allele(rh);
                    let mapped = alignment.reverse_map_allele(m, ref_a);
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

fn select_top_k_allow_zero(scores: &[f32], k: usize) -> Vec<(usize, f32)> {
    if k == 0 || scores.is_empty() {
        return Vec::new();
    }
    let mut ranked: Vec<(usize, f32)> = scores
        .iter()
        .enumerate()
        .filter(|&(_, &s)| s.is_finite())
        .map(|(i, &s)| (i, s))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if ranked.len() > k {
        ranked.truncate(k);
    }
    ranked
}

fn should_use_exact_prescan(n_ref_haps: usize, batch_len: usize, n_markers: usize) -> bool {
    let ops = n_ref_haps as u128 * batch_len as u128 * n_markers as u128;
    ops <= EXACT_PRESCAN_MAX_OPS
}

fn score_window_batch_exact_packed<TargetSpace, RefSpace>(
    batch_haps: &[usize],
    target_gt: &GenotypeMatrix<Phased, TargetSpace>,
    ref_columns: &[PackedRefColumn],
    n_ref_haps: usize,
    alignment: &MarkerAlignment<TargetSpace, RefSpace>,
    global_scores: &mut [Vec<f32>],
    window_scores: &mut [Vec<f32>],
) {
    let n_markers = target_gt.n_markers();
    if n_markers == 0 || n_ref_haps == 0 || batch_haps.is_empty() {
        return;
    }

    let freqs = compute_target_freqs_packed(target_gt, ref_columns, n_ref_haps, alignment);
    let min_freq = 1.0 / (2.0 * n_ref_haps.max(1) as f32);

    let mut query_alleles = vec![255u8; batch_haps.len()];
    let mut ref_bins: Vec<Vec<u32>> = Vec::new();

    for m in 0..n_markers {
        for (i, &hap_idx) in batch_haps.iter().enumerate() {
            query_alleles[i] =
                target_gt.allele(MarkerIdx::new(m as u32), HapIdx::new(hap_idx as u32));
        }

        let Some(ref_m) = alignment.target_to_ref(MarkerIdx::new(m as u32)) else {
            continue;
        };

        let n_alleles = target_gt
            .markers()
            .marker(MarkerIdx::new(m as u32))
            .n_alleles()
            .max(1);
        if ref_bins.len() < n_alleles {
            ref_bins.resize_with(n_alleles, Vec::new);
        }
        for bins in ref_bins.iter_mut().take(n_alleles) {
            bins.clear();
        }

        let col = &ref_columns[ref_m.as_usize()];
        for rh in 0..n_ref_haps {
            let ref_a = col.allele(rh);
            if ref_a == 255 {
                continue;
            }
            let mapped = alignment.reverse_map_allele(m, ref_a);
            if mapped == 255 {
                continue;
            }
            let idx = mapped as usize;
            if idx >= ref_bins.len() {
                ref_bins.resize_with(idx + 1, Vec::new);
            }
            ref_bins[idx].push(rh as u32);
        }

        for (i, _) in batch_haps.iter().enumerate() {
            let targ = query_alleles[i];
            if targ == 255 {
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
            let bins = ref_bins.get(targ as usize);
            let Some(bins) = bins else { continue };
            for &rh in bins {
                let idx = rh as usize;
                global_scores[i][idx] += weight;
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

fn score_window_batch_pbwt_packed<TargetSpace, RefSpace>(
    batch_haps: &[usize],
    target_gt: &GenotypeMatrix<Phased, TargetSpace>,
    ref_columns: &[PackedRefColumn],
    n_ref_haps: usize,
    alignment: &MarkerAlignment<TargetSpace, RefSpace>,
    gen_maps: &GeneticMaps,
    k_per_hap: usize,
    step_cm: f64,
    global_scores: &mut [Vec<f32>],
    window_scores: &mut [Vec<f32>],
) {
    let n_markers = target_gt.n_markers();
    if n_markers == 0 || n_ref_haps == 0 || batch_haps.is_empty() {
        return;
    }

    let mut gen_positions = Vec::with_capacity(n_markers);
    for m in 0..n_markers {
        let marker = target_gt.markers().marker(MarkerIdx::new(m as u32));
        let gen_pos = gen_maps.gen_pos(marker.chrom, marker.pos);
        gen_positions.push(gen_pos);
    }
    let mut sampling = build_sampling_points(&gen_positions, step_cm);
    for m in 0..n_markers {
        if alignment.target_to_ref(MarkerIdx::new(m as u32)).is_some() {
            sampling[m] = true;
        }
    }
    let freqs = compute_target_freqs_packed(target_gt, ref_columns, n_ref_haps, alignment);

    let mut pbwt_fwd = ReferencePbwt::new(n_ref_haps);
    let mut beams_fwd: Vec<RankBeam> = (0..batch_haps.len())
        .map(|_| RankBeam::full(n_ref_haps as u32))
        .collect();
    let mut ref_alleles = vec![0u8; n_ref_haps];
    let mut query_alleles = vec![PbwtStrictAllele::missing(); batch_haps.len()];
    let mut donors_buf: Vec<u32> = Vec::new();

    let min_freq = 1.0 / (2.0 * n_ref_haps.max(1) as f32);

    for m in 0..n_markers {
        for (i, &hap_idx) in batch_haps.iter().enumerate() {
            let qa = target_gt.allele(MarkerIdx::new(m as u32), HapIdx::new(hap_idx as u32));
            query_alleles[i] =
                PbwtStrictAllele::allele(qa).unwrap_or_else(PbwtStrictAllele::missing);
        }
        if let Some(ref_m) = alignment.target_to_ref(MarkerIdx::new(m as u32)) {
            let col = &ref_columns[ref_m.as_usize()];
            col.fill_alleles(&mut ref_alleles);
            for rh in 0..n_ref_haps {
                let ref_a = ref_alleles[rh];
                ref_alleles[rh] = alignment.reverse_map_allele(m, ref_a);
            }
        } else {
            ref_alleles.fill(255);
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
                if a >= 2 && a != 255 {
                    is_biallelic = false;
                    break;
                }
            }
        }
        let n_alleles = if is_biallelic { 2 } else { 256 };

        pbwt_fwd.advance_with_beams_strict(
            &ref_alleles,
            n_alleles,
            m,
            &query_alleles,
            &mut beams_fwd,
        );

        if sampling[m] {
            for (i, _) in batch_haps.iter().enumerate() {
                let targ = query_alleles[i].value();
                if targ == 255 {
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
                    if idx < n_ref_haps {
                        let ref_a = ref_alleles[idx];
                        if ref_a == 255 || ref_a != targ {
                            continue;
                        }
                        global_scores[i][idx] += weight;
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

    let mut pbwt_bwd = ReferencePbwt::new(n_ref_haps);
    let mut beams_bwd: Vec<RankBeam> = (0..batch_haps.len())
        .map(|_| RankBeam::full(n_ref_haps as u32))
        .collect();
    for (rev_step, m) in (0..n_markers).rev().enumerate() {
        for (i, &hap_idx) in batch_haps.iter().enumerate() {
            let qa = target_gt.allele(MarkerIdx::new(m as u32), HapIdx::new(hap_idx as u32));
            query_alleles[i] =
                PbwtStrictAllele::allele(qa).unwrap_or_else(PbwtStrictAllele::missing);
        }
        if let Some(ref_m) = alignment.target_to_ref(MarkerIdx::new(m as u32)) {
            let col = &ref_columns[ref_m.as_usize()];
            col.fill_alleles(&mut ref_alleles);
            for rh in 0..n_ref_haps {
                let ref_a = ref_alleles[rh];
                ref_alleles[rh] = alignment.reverse_map_allele(m, ref_a);
            }
        } else {
            ref_alleles.fill(255);
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
                if a >= 2 && a != 255 {
                    is_biallelic = false;
                    break;
                }
            }
        }
        let n_alleles = if is_biallelic { 2 } else { 256 };

        pbwt_bwd.advance_with_beams_strict(
            &ref_alleles,
            n_alleles,
            rev_step,
            &query_alleles,
            &mut beams_bwd,
        );

        if sampling[m] {
            for (i, _) in batch_haps.iter().enumerate() {
                let targ = query_alleles[i].value();
                if targ == 255 {
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
                    if idx < n_ref_haps {
                        let ref_a = ref_alleles[idx];
                        if ref_a == 255 || ref_a != targ {
                            continue;
                        }
                        global_scores[i][idx] += weight;
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
}

fn normalize_chrom_local(name: &str) -> &str {
    if name.len() >= 3 && name[..3].eq_ignore_ascii_case("chr") {
        &name[3..]
    } else {
        name
    }
}

fn collect_target_positions(path: &Path) -> Result<(TargetMarkerIndex, usize)> {
    let (vcf_reader, mut reader) = crate::io::vcf::VcfReader::open(path)?;
    vcf_reader.samples_arc();
    let mut positions: TargetMarkerIndex = std::collections::HashMap::new();
    let mut total = 0usize;
    let mut line = String::new();

    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split('\t');
        let chrom = match parts.next() {
            Some(c) => c,
            None => continue,
        };
        let pos_str = match parts.next() {
            Some(p) => p,
            None => continue,
        };
        let pos: u32 = match pos_str.parse() {
            Ok(p) => p,
            Err(_) => continue,
        };
        let norm = normalize_chrom_local(chrom);
        positions
            .entry(norm.to_string())
            .or_insert_with(std::collections::HashSet::new)
            .insert(pos);
        total += 1;
    }

    Ok((positions, total))
}

fn open_ref_reader(path: &Path) -> Result<RefPanelReader> {
    let is_bref3 = path.extension().and_then(|e| e.to_str()) == Some("bref3");
    if is_bref3 {
        let stream_reader = crate::io::bref3::StreamingBref3Reader::open(path)?;
        let windowed = crate::io::bref3::StreamingBref3WindowReader::new(stream_reader);
        Ok(RefPanelReader::Bref3(windowed))
    } else {
        let reader = crate::io::bref3::StreamingRefVcfReader::open(path)?;
        Ok(RefPanelReader::StreamingVcf(reader))
    }
}

const BREF3_CONVERT_MIN_BYTES: u64 = 500 * 1024 * 1024;

fn should_convert_ref_to_bref3(config: &Config, ref_path: &Path) -> bool {
    if ref_path.extension().and_then(|e| e.to_str()) == Some("bref3") {
        return false;
    }
    if let Some(region) = config.chrom.as_deref() {
        if region.contains(':') && region.contains('-') {
            return false;
        }
    }
    let Ok(meta) = std::fs::metadata(ref_path) else {
        return false;
    };
    meta.len() >= BREF3_CONVERT_MIN_BYTES
}

fn cache_is_fresh(cache_path: &Path, ref_path: &Path) -> bool {
    let Ok(cache_meta) = std::fs::metadata(cache_path) else {
        return false;
    };
    if cache_meta.len() == 0 {
        return false;
    }
    let Ok(ref_meta) = std::fs::metadata(ref_path) else {
        return false;
    };
    match (cache_meta.modified(), ref_meta.modified()) {
        (Ok(cache_time), Ok(ref_time)) => cache_time >= ref_time,
        _ => true,
    }
}

fn ensure_binary_reference(ref_path: &Path, config: &Config) -> Result<PathBuf> {
    if ref_path.extension().and_then(|e| e.to_str()) == Some("bref3") {
        return Ok(ref_path.to_path_buf());
    }
    if !should_convert_ref_to_bref3(config, ref_path) {
        return Ok(ref_path.to_path_buf());
    }

    let mut candidates: Vec<PathBuf> = Vec::new();
    candidates.push(ref_path.with_extension("bref3"));
    candidates.push(config.out.with_extension("ref.bref3"));
    let stem = ref_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("reference");
    candidates.push(std::env::temp_dir().join(format!(
        "reagle_ref_cache_{}.bref3",
        stem
    )));

    for path in candidates {
        if path.exists() && cache_is_fresh(&path, ref_path) {
            eprintln!("Using cached BREF3 reference at {:?}", path);
            return Ok(path);
        }
        let tmp_path = path.with_extension("bref3.tmp");
        match convert_ref_vcf_to_bref3(ref_path, &tmp_path) {
            Ok(()) => {
                if let Err(err) = std::fs::rename(&tmp_path, &path) {
                    eprintln!(
                        "Reference conversion rename failed at {:?}: {}. Trying next location...",
                        path, err
                    );
                    let _ = std::fs::remove_file(&tmp_path);
                    continue;
                }
                eprintln!("Converted reference VCF to BREF3 at {:?}", path);
                return Ok(path);
            }
            Err(err) => {
                let _ = std::fs::remove_file(&tmp_path);
                eprintln!(
                    "Reference conversion failed at {:?}: {}. Trying next location...",
                    path, err
                );
            }
        }
    }

    Ok(ref_path.to_path_buf())
}

#[derive(Debug)]
struct PrescanCacheMeta {
    path: PathBuf,
    n_ref_haps: usize,
    per_window_caps: Vec<usize>,
    window_handoff: Vec<(f64, f64)>,
}

struct PrescanCacheGuard {
    path: PathBuf,
}

impl Drop for PrescanCacheGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

enum ReferenceData {
    InMemory {
        windows: Vec<RefWindow>,
        packed_columns: Vec<Vec<PackedRefColumn>>,
        n_ref_haps: usize,
        per_window_caps: Vec<usize>,
        window_handoff: Vec<(f64, f64)>,
    },
    OnDisk {
        cache_meta: PrescanCacheMeta,
        guard: PrescanCacheGuard,
    },
}

impl ReferenceData {
    fn n_ref_haps(&self) -> usize {
        match self {
            Self::InMemory { n_ref_haps, .. } => *n_ref_haps,
            Self::OnDisk { cache_meta, .. } => cache_meta.n_ref_haps,
        }
    }

    fn per_window_caps(&self) -> &[usize] {
        match self {
            Self::InMemory { per_window_caps, .. } => per_window_caps.as_slice(),
            Self::OnDisk { cache_meta, .. } => cache_meta.per_window_caps.as_slice(),
        }
    }

    fn window_handoff(&self) -> &[(f64, f64)] {
        match self {
            Self::InMemory { window_handoff, .. } => window_handoff.as_slice(),
            Self::OnDisk { cache_meta, .. } => cache_meta.window_handoff.as_slice(),
        }
    }
}

fn packed_column_size_bytes(col: &PackedRefColumn) -> u64 {
    match col {
        PackedRefColumn::Bits { words, missing, .. } => {
            (words.len() as u64 * 8) + (missing.len() as u64 * 8)
        }
        PackedRefColumn::Bytes { alleles } => alleles.len() as u64,
    }
}

fn estimate_ref_window_bytes(ref_window: &RefWindow, packed_cols: &[PackedRefColumn]) -> u64 {
    let mut bytes = 0u64;
    for col in ref_window.ref_columns.iter() {
        bytes = bytes.saturating_add(col.size_bytes() as u64);
    }
    for col in packed_cols.iter() {
        bytes = bytes.saturating_add(packed_column_size_bytes(col));
    }
    let marker_overhead = ref_window.markers.len().saturating_mul(64) as u64;
    bytes.saturating_add(marker_overhead)
}

struct PrescanTargetEntry {
    phased_target: GenotypeMatrix<Phased, AnyMarkerSpace>,
    alignment: MarkerAlignment<AnyMarkerSpace, RefWindowSpace>,
}

fn estimate_target_entry_bytes(entry: &PrescanTargetEntry) -> u64 {
    let target_bytes = entry.phased_target.size_bytes() as u64;
    let align_markers =
        entry.alignment.ref_to_target.len().saturating_add(entry.alignment.target_to_ref.len());
    let align_bytes = align_markers.saturating_mul(16) as u64
        + entry.alignment.allele_mappings.len().saturating_mul(32) as u64;
    target_bytes.saturating_add(align_bytes)
}

fn compute_per_window_cap(
    n_ref_haps: usize,
    n_ref_markers: usize,
    available_bytes: u64,
    n_threads: usize,
    safe_bytes_per_thread: u64,
    force_full_panel: bool,
    desired_cap: Option<usize>,
) -> usize {
    let mut per_window_cap_window = if force_full_panel {
        n_ref_haps.max(1)
    } else {
        let per_state_bytes = 4usize.saturating_mul(4 + n_ref_markers);
        let mut cap = if per_state_bytes == 0 {
            0
        } else {
            let full_panel_bytes = per_state_bytes
                .saturating_mul(n_ref_haps)
                .saturating_mul(n_threads.max(1));
            let can_fit_full_panel = available_bytes > 0
                && full_panel_bytes as f64 <= (available_bytes as f64 * FULL_PANEL_RAM_FRACTION);
            if can_fit_full_panel {
                n_ref_haps
            } else {
                (safe_bytes_per_thread as usize) / per_state_bytes
            }
        };
        if cap == 0 {
            cap = 1;
        }
        cap
    };
    let mut cap = n_ref_haps.max(1);
    if let Some(requested) = desired_cap {
        cap = requested.max(1).min(n_ref_haps.max(1));
        per_window_cap_window = cap;
    } else {
        per_window_cap_window = per_window_cap_window.min(cap).max(1);
    }
    per_window_cap_window
}

fn prepare_reference_data(
    ref_path: &Path,
    streaming_config: &StreamingConfig,
    gen_maps: &GeneticMaps,
    target_positions: &TargetMarkerIndex,
    available_bytes: u64,
    n_threads: usize,
    safe_bytes_per_thread: u64,
    force_full_panel: bool,
    desired_cap: Option<usize>,
) -> Result<ReferenceData> {
    let memory_budget = if available_bytes == 0 {
        0u64
    } else {
        (available_bytes as f64 * REF_PANEL_RAM_FRACTION) as u64
    };
    let mut use_in_memory = memory_budget > 0;

    let (tx, rx) = std::sync::mpsc::sync_channel::<Result<Option<RefWindow>>>(2);
    let ref_path = ref_path.to_path_buf();
    let streaming_config = streaming_config.clone();
    let gen_maps_thread = gen_maps.clone();
    let target_positions = target_positions.clone();
    let reader_handle = std::thread::spawn(move || {
        let mut ref_reader = match open_ref_reader(&ref_path) {
            Ok(reader) => reader,
            Err(err) => {
                let _ = tx.send(Err(err.into()));
                return Ok(());
            }
        };
        loop {
            let result = ref_reader.next_window(
                &streaming_config,
                &gen_maps_thread,
                Some(&target_positions),
            );
            match result {
                Ok(Some(window)) => {
                    if tx.send(Ok(Some(window))).is_err() {
                        break;
                    }
                }
                Ok(None) => {
                    let _ = tx.send(Ok(None));
                    break;
                }
                Err(err) => {
                    let _ = tx.send(Err(err.into()));
                    break;
                }
            }
        }
        Ok::<(), ReagleError>(())
    });
    let mut n_ref_haps = 0usize;
    let mut per_window_caps: Vec<usize> = Vec::new();
    let mut window_handoff: Vec<(f64, f64)> = Vec::new();
    let mut windows: Vec<RefWindow> = Vec::new();
    let mut packed_columns: Vec<Vec<PackedRefColumn>> = Vec::new();
    let mut total_bytes: u64 = 0;

    let mut cache_path: Option<PathBuf> = None;
    let mut cache_writer: Option<PrescanCacheWriter> = None;

    let read_result: Result<()> = (|| {
        for msg in rx {
            let ref_window = match msg? {
                Some(window) => window,
                None => break,
            };
            let n_ref_markers = ref_window.markers.len();
            if n_ref_markers == 0 {
                continue;
            }

            if n_ref_haps == 0 {
                n_ref_haps = ref_window
                    .ref_columns
                    .first()
                    .map(|c| c.n_haplotypes())
                    .unwrap_or(0);
                if n_ref_haps == 0 {
                    continue;
                }
            }

            let per_window_cap_window = compute_per_window_cap(
                n_ref_haps,
                n_ref_markers,
                available_bytes,
                n_threads,
                safe_bytes_per_thread,
                force_full_panel,
                desired_cap,
            );
            per_window_caps.push(per_window_cap_window);

            let output_start = ref_window.output_start.min(n_ref_markers.saturating_sub(1));
            let output_end = ref_window.output_end.min(n_ref_markers).max(1);
            let left_idx = output_end.saturating_sub(1);
            let right_idx = output_start.min(n_ref_markers.saturating_sub(1));
            let left_marker = ref_window.markers.marker(MarkerIdx::new(left_idx as u32));
            let right_marker = ref_window.markers.marker(MarkerIdx::new(right_idx as u32));
            let left_gen = gen_maps.gen_pos(left_marker.chrom, left_marker.pos);
            let right_gen = gen_maps.gen_pos(right_marker.chrom, right_marker.pos);
            window_handoff.push((left_gen, right_gen));

            if use_in_memory {
                let packed = pack_ref_columns(&ref_window.markers, &ref_window.ref_columns);
                total_bytes =
                    total_bytes.saturating_add(estimate_ref_window_bytes(&ref_window, &packed));
                if total_bytes <= memory_budget {
                    packed_columns.push(packed);
                    windows.push(ref_window);
                    continue;
                }

                use_in_memory = false;
                let path = create_temp_cache_path();
                let mut writer = PrescanCacheWriter::create(&path)?;
                writer.set_n_ref_haps(n_ref_haps);
                writer.write_header()?;
                for win in windows.iter() {
                    writer.write_window(win)?;
                }
                writer.write_window(&ref_window)?;
                windows.clear();
                packed_columns.clear();
                cache_path = Some(path);
                cache_writer = Some(writer);
            } else {
                if cache_writer.is_none() {
                    let path = cache_path.get_or_insert_with(create_temp_cache_path);
                    let mut writer = PrescanCacheWriter::create(path)?;
                    writer.set_n_ref_haps(n_ref_haps);
                    writer.write_header()?;
                    cache_writer = Some(writer);
                }
                if let Some(writer) = cache_writer.as_mut() {
                    writer.write_window(&ref_window)?;
                }
            }
        }
        Ok(())
    })();

    let _ = reader_handle
        .join()
        .map_err(|_| ReagleError::vcf("Reference reader thread panicked".to_string()))??;
    read_result?;

    if n_ref_haps == 0 {
        return Err(ReagleError::vcf(
            "Reference window scanning found no haplotypes".to_string(),
        ));
    }

    if use_in_memory {
        if windows.is_empty() {
            return Err(ReagleError::vcf(
                "Reference window scanning found no haplotypes".to_string(),
            ));
        }
        Ok(ReferenceData::InMemory {
            windows,
            packed_columns,
            n_ref_haps,
            per_window_caps,
            window_handoff,
        })
    } else {
        let path = cache_path.ok_or_else(|| {
            ReagleError::vcf("Prescan cache path missing after scan".to_string())
        })?;
        let writer = cache_writer.ok_or_else(|| {
            ReagleError::vcf("Prescan cache writer missing after scan".to_string())
        })?;
        let _ = writer.finish()?;
        let meta = PrescanCacheMeta {
            path: path.clone(),
            n_ref_haps,
            per_window_caps,
            window_handoff,
        };
        let guard = PrescanCacheGuard { path };
        Ok(ReferenceData::OnDisk {
            cache_meta: meta,
            guard,
        })
    }
}

fn is_vcf_fully_phased(path: &Path) -> Result<bool> {
    let (_, mut reader) = crate::io::vcf::VcfReader::open(path)?;
    let mut line = String::new();
    while reader.read_line(&mut line)? != 0 {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            line.clear();
            continue;
        }
        let mut parts = trimmed.split('\t');
        if parts.next().is_none() {
            line.clear();
            continue;
        }
        if parts.next().is_none() {
            line.clear();
            continue;
        }
        let mut missing_fields = false;
        for _ in 0..6 {
            if parts.next().is_none() {
                missing_fields = true;
                break;
            }
        }
        if missing_fields {
            line.clear();
            continue;
        }
        let format = match parts.next() {
            Some(f) => f,
            None => continue,
        };
        let gt_idx = format.split(':').position(|f| f == "GT");
        if gt_idx.is_none() {
            continue;
        }
        let gt_idx = gt_idx.unwrap();
        for sample in parts {
            let mut fields = sample.split(':');
            let gt = fields.nth(gt_idx).unwrap_or("");
            // Missing genotypes (e.g. "./.", ".|.", "0/.") should not force
            // the entire file to be treated as unphased.
            if gt.contains('.') {
                continue;
            }
            if gt.contains('/') {
                line.clear();
                return Ok(false);
            }
        }
        line.clear();
    }
    Ok(true)
}

fn build_imputation_plan(
    target_path: &Path,
    streaming_config: &StreamingConfig,
    gen_maps: &GeneticMaps,
    per_window_cap: usize,
    available_bytes: u64,
    imp_step_cm: f64,
    params: &crate::model::parameters::ModelParams,
    ref_data: &ReferenceData,
    telemetry: Option<&Arc<TelemetryBlackboard>>,
) -> Result<ImputationPlan> {
    let target_reader =
        StreamingVcfReader::open(target_path, gen_maps.clone(), streaming_config.clone())?;
    let n_target_haps = target_reader.samples_arc().len() * 2;
    if n_target_haps == 0 {
        return Err(ReagleError::vcf("No target samples for pre-scan".to_string()));
    }

    let mut plan = ImputationPlan {
        n_ref_haps: 0,
        core_states: vec![Vec::new(); n_target_haps],
        window_intervals: vec![Vec::new(); n_target_haps],
        abyss_mask: vec![Vec::new(); n_target_haps],
        per_window_cap: per_window_cap.max(1),
        per_window_caps: Vec::new(),
        full_panel: false,
        stats: ImputationPlanStats::default(),
    };

    let avail = available_bytes;
    let n_ref_haps = ref_data.n_ref_haps();
    if n_ref_haps == 0 {
        return Err(ReagleError::vcf(
            "Reference window scanning found no haplotypes".to_string(),
        ));
    }
    plan.n_ref_haps = n_ref_haps;
    let window_handoff = ref_data.window_handoff().to_vec();
    let per_window_caps = ref_data.per_window_caps().to_vec();
    let batch_size = estimate_scan_batch_size(avail, n_ref_haps, n_target_haps);
    let mut batch_start = 0usize;
    let batches_total = (n_target_haps + batch_size - 1) / batch_size;
    let prescan_start = std::time::Instant::now();

    let can_full_panel =
        !per_window_caps.is_empty() && per_window_caps.iter().all(|&c| c >= n_ref_haps);
    if can_full_panel {
        let num_windows = per_window_caps.len();
        if num_windows == 0 {
            return Err(ReagleError::vcf(
                "Pre-scan produced no windows for allocation".to_string(),
            ));
        }
        let plan_start = std::time::Instant::now();
        eprintln!(
            "Pre-scan: skipped (full panel global mode); ref_haps={}, windows={}",
            n_ref_haps,
            num_windows
        );
        if let Some(bb) = telemetry {
            bb.set_stage(crate::utils::telemetry::Stage::ImputationPlanning);
            bb.set_producer_stage(crate::utils::telemetry::Stage::ImputationPlanning);
            bb.set_op("Imputation prescan: skipped (full panel)");
        }
        plan.per_window_cap = n_ref_haps.max(1);
        plan.per_window_caps = per_window_caps;
        plan.full_panel = true;
        for _ in 0..n_target_haps {
            plan.stats.update(n_ref_haps, 0, 0);
        }
        eprintln!(
            "Pre-scan summary: skipped full panel in {:.1}s",
            plan_start.elapsed().as_secs_f32()
        );
        return Ok(plan);
    }

    if window_handoff.is_empty() || per_window_caps.is_empty() {
        return Err(ReagleError::vcf(
            "Pre-scan produced no windows for LMS allocation".to_string(),
        ));
    }

    plan.per_window_caps = per_window_caps.clone();

    eprintln!(
        "Pre-scan: enabled (LMS allocation); target_haps={}, ref_haps={}, batch_size={}",
        n_target_haps,
        n_ref_haps,
        batch_size
    );
    if let Some(bb) = telemetry {
        let num_windows = window_handoff.len().max(1);
        let total_batches = batches_total.saturating_mul(num_windows).max(1);
        bb.set_stage(crate::utils::telemetry::Stage::ImputationPrescan);
        bb.set_producer_stage(crate::utils::telemetry::Stage::ImputationPrescan);
        bb.set_op("Imputation prescan: PBWT scoring");
        bb.set_total_windows(num_windows as u64);
        bb.set_current_window(0);
        bb.set_total_markers(total_batches as u64);
        bb.set_markers_processed(0);
    }

    let target_cache_budget = if avail == 0 {
        0u64
    } else {
        (avail as f64 * TARGET_CACHE_RAM_FRACTION) as u64
    };
    let mut target_cache: Option<Vec<Option<PrescanTargetEntry>>> = None;
    if target_cache_budget > 0 {
        let mut entries: Vec<Option<PrescanTargetEntry>> = Vec::new();
        let mut target_bytes = 0u64;
        let mut cache_ok = true;
        let mut target_reader =
            StreamingVcfReader::open(target_path, gen_maps.clone(), streaming_config.clone())?;

        match ref_data {
            ReferenceData::InMemory { windows, .. } => {
                for ref_window in windows.iter() {
                    let n_ref_markers = ref_window.markers.len();
                    if n_ref_markers == 0 {
                        entries.push(None);
                        continue;
                    }
                    let ref_chrom_idx = ref_window.markers.marker(MarkerIdx::new(0)).chrom;
                    let ref_chrom = ref_window
                        .markers
                        .chrom_name(ref_chrom_idx)
                        .unwrap_or("UNKNOWN");
                    let start_pos = ref_window.markers.marker(MarkerIdx::new(0)).pos;
                    let end_pos = ref_window
                        .markers
                        .marker(MarkerIdx::new((n_ref_markers - 1) as u32))
                        .pos;
                    let chrom_candidates = chrom_variants(ref_chrom);
                    let target_window = target_reader.load_window_for_region(
                        &chrom_candidates,
                        start_pos,
                        end_pos,
                    )?;
                    if let Some(target_window) = target_window {
                        let alignment = MarkerAlignment::new_with_ref_markers(
                            &target_window.genotypes,
                            &ref_window.markers,
                        );
                        let phased_target = target_window.genotypes.clone().into_phased();
                        let entry = PrescanTargetEntry {
                            phased_target,
                            alignment,
                        };
                        target_bytes = target_bytes.saturating_add(estimate_target_entry_bytes(&entry));
                        if target_bytes > target_cache_budget {
                            cache_ok = false;
                            break;
                        }
                        entries.push(Some(entry));
                    } else {
                        entries.push(None);
                    }
                }
            }
            ReferenceData::OnDisk { cache_meta, .. } => {
                let mut ref_reader = PrescanCacheReader::open(&cache_meta.path)?;
                loop {
                    let ref_window = ref_reader.next_window()?;
                    let Some(ref_window) = ref_window else { break };
                    let n_ref_markers = ref_window.markers.len();
                    if n_ref_markers == 0 {
                        entries.push(None);
                        continue;
                    }
                    let ref_chrom_idx = ref_window.markers.marker(MarkerIdx::new(0)).chrom;
                    let ref_chrom = ref_window
                        .markers
                        .chrom_name(ref_chrom_idx)
                        .unwrap_or("UNKNOWN");
                    let start_pos = ref_window.markers.marker(MarkerIdx::new(0)).pos;
                    let end_pos = ref_window
                        .markers
                        .marker(MarkerIdx::new((n_ref_markers - 1) as u32))
                        .pos;
                    let chrom_candidates = chrom_variants(ref_chrom);
                    let target_window = target_reader.load_window_for_region(
                        &chrom_candidates,
                        start_pos,
                        end_pos,
                    )?;
                    if let Some(target_window) = target_window {
                        let alignment = MarkerAlignment::new_with_ref_markers(
                            &target_window.genotypes,
                            &ref_window.markers,
                        );
                        let phased_target = target_window.genotypes.clone().into_phased();
                        let entry = PrescanTargetEntry {
                            phased_target,
                            alignment,
                        };
                        target_bytes = target_bytes.saturating_add(estimate_target_entry_bytes(&entry));
                        if target_bytes > target_cache_budget {
                            cache_ok = false;
                            break;
                        }
                        entries.push(Some(entry));
                    } else {
                        entries.push(None);
                    }
                }
            }
        }

        if cache_ok && !entries.is_empty() {
            eprintln!(
                "Pre-scan: cached target windows (~{} MB)",
                target_bytes / (1024 * 1024)
            );
            target_cache = Some(entries);
        } else if !entries.is_empty() {
            eprintln!("Pre-scan: target cache disabled (budget exceeded)");
        }
    }

    let mut global_scores: Vec<Vec<f32>> = Vec::new();
    let mut window_scores: Vec<Vec<f32>> = Vec::new();
    let mut best_window_scores: Vec<Vec<f32>> = Vec::new();
    let mut window_rank_hits: Vec<Vec<u32>> = Vec::new();
    let mut scores_by_window: Vec<Vec<Vec<(usize, f32)>>> = Vec::new();

    let mut on_disk_reader = match ref_data {
        ReferenceData::OnDisk { cache_meta, .. } => {
            Some(PrescanCacheReader::open(&cache_meta.path)?)
        }
        _ => None,
    };

    let cache_ready = target_cache
        .as_ref()
        .map(|c| c.iter().all(|e| e.is_some()))
        .unwrap_or(false);

    let mut batch_idx = 0usize;
    let mut window_span_cm: Option<f64> = None;
    let mut window_span_bp: Option<u64> = None;
    while batch_start < n_target_haps {
        batch_idx += 1;
        let batch_end = (batch_start + batch_size).min(n_target_haps);
        let batch_haps: Vec<usize> = (batch_start..batch_end).collect();
        let batch_len = batch_haps.len();
        if let Some(bb) = telemetry {
            let mut op = format!(
                "Imputation prescan: scoring batch {}/{} (batch_size={})",
                batch_idx,
                batches_total.max(1),
                batch_len
            );
            if let (Some(cm), Some(bp)) = (window_span_cm, window_span_bp) {
                op.push_str(&format!(", span_cm={:.3}, span_bp={}", cm, bp));
            }
            bb.set_op(&op);
        }

        let mut target_reader: Option<StreamingVcfReader> = if cache_ready {
            None
        } else {
            Some(StreamingVcfReader::open(
                target_path,
                gen_maps.clone(),
                streaming_config.clone(),
            )?)
        };

        if global_scores.len() > batch_len {
            global_scores.truncate(batch_len);
            window_scores.truncate(batch_len);
            best_window_scores.truncate(batch_len);
            window_rank_hits.truncate(batch_len);
            scores_by_window.truncate(batch_len);
        }
        while global_scores.len() < batch_len {
            global_scores.push(Vec::new());
            window_scores.push(Vec::new());
            best_window_scores.push(Vec::new());
            window_rank_hits.push(Vec::new());
            scores_by_window.push(Vec::new());
        }
        for list in scores_by_window.iter_mut() {
            list.clear();
        }

        let mut window_idx = 0usize;
        match ref_data {
            ReferenceData::InMemory {
                windows,
                packed_columns,
                ..
            } => {
                for (idx, (ref_window, ref_columns)) in
                    windows.iter().zip(packed_columns.iter()).enumerate()
                {
                    let n_ref_markers = ref_window.markers.len();
                    if n_ref_markers == 0 {
                        continue;
                    }
                    if window_span_cm.is_none() || window_span_bp.is_none() {
                        let start_pos = ref_window.markers.marker(MarkerIdx::new(0)).pos;
                        let end_pos = ref_window
                            .markers
                            .marker(MarkerIdx::new((n_ref_markers - 1) as u32))
                            .pos;
                        let span_bp = end_pos.saturating_sub(start_pos);
                        let start_cm = gen_maps.gen_pos(
                            ref_window.markers.marker(MarkerIdx::new(0)).chrom,
                            start_pos,
                        );
                        let end_cm = gen_maps.gen_pos(
                            ref_window.markers.marker(MarkerIdx::new(0)).chrom,
                            end_pos,
                        );
                        window_span_bp = Some(span_bp.into());
                        window_span_cm = Some((end_cm - start_cm).abs());
                    }
                    // Derive per-window cap from the observed marker count to match
                    // the real workspace footprint (fwd/bwd/history scale with markers).
                    let per_window_cap_window = per_window_caps
                        .get(window_idx)
                        .copied()
                        .unwrap_or(per_window_cap.max(1));

                    let (alignment, phased_target) = if let Some(cache) = target_cache.as_ref() {
                        if let Some(Some(entry)) = cache.get(idx) {
                            (
                                entry.alignment.clone(),
                                entry.phased_target.clone(),
                            )
                        } else {
                            let ref_chrom_idx =
                                ref_window.markers.marker(MarkerIdx::new(0)).chrom;
                            let ref_chrom = ref_window
                                .markers
                                .chrom_name(ref_chrom_idx)
                                .unwrap_or("UNKNOWN");
                            let start_pos = ref_window.markers.marker(MarkerIdx::new(0)).pos;
                            let end_pos = ref_window
                                .markers
                                .marker(MarkerIdx::new((n_ref_markers - 1) as u32))
                                .pos;
                            let chrom_candidates = chrom_variants(ref_chrom);
                            let reader = target_reader.as_mut().ok_or_else(|| {
                                ReagleError::vcf("Target reader missing in prescan".to_string())
                            })?;
                            let target_window = reader.load_window_for_region(
                                &chrom_candidates,
                                start_pos,
                                end_pos,
                            )?;
                            let Some(target_window) = target_window else {
                                continue;
                            };
                            let alignment = MarkerAlignment::new_with_ref_markers(
                                &target_window.genotypes,
                                &ref_window.markers,
                            );
                            let phased_target = target_window.genotypes.clone().into_phased();
                            (alignment, phased_target)
                        }
                    } else {
                        let ref_chrom_idx = ref_window.markers.marker(MarkerIdx::new(0)).chrom;
                        let ref_chrom = ref_window
                            .markers
                            .chrom_name(ref_chrom_idx)
                            .unwrap_or("UNKNOWN");
                        let start_pos = ref_window.markers.marker(MarkerIdx::new(0)).pos;
                        let end_pos = ref_window
                            .markers
                            .marker(MarkerIdx::new((n_ref_markers - 1) as u32))
                            .pos;
                        let chrom_candidates = chrom_variants(ref_chrom);
                        let reader = target_reader.as_mut().ok_or_else(|| {
                            ReagleError::vcf("Target reader missing in prescan".to_string())
                        })?;
                        let target_window = reader.load_window_for_region(
                            &chrom_candidates,
                            start_pos,
                            end_pos,
                        )?;
                        let Some(target_window) = target_window else {
                            continue;
                        };
                        let alignment = MarkerAlignment::new_with_ref_markers(
                            &target_window.genotypes,
                            &ref_window.markers,
                        );
                        let phased_target = target_window.genotypes.clone().into_phased();
                        (alignment, phased_target)
                    };

                    for i in 0..batch_len {
                        if global_scores[i].len() != n_ref_haps {
                            global_scores[i] = vec![0.0f32; n_ref_haps];
                            window_scores[i] = vec![f32::NEG_INFINITY; n_ref_haps];
                            best_window_scores[i] = vec![f32::NEG_INFINITY; n_ref_haps];
                            window_rank_hits[i] = vec![0u32; n_ref_haps];
                        }
                    }

                    for w in window_scores.iter_mut() {
                        w.fill(f32::NEG_INFINITY);
                    }

                    let k_per_hap = per_window_cap_window
                        .saturating_mul(PBWT_PER_WINDOW_MULT)
                        .max(PBWT_MIN_PER_HAP)
                        .min(PBWT_MAX_PER_HAP)
                        .max(1);

                    let step_cm = PBWT_SELECT_BLOCK_CM.max(imp_step_cm);
                    let use_exact = should_use_exact_prescan(
                        n_ref_haps,
                        batch_haps.len(),
                        phased_target.n_markers(),
                    );
                    if use_exact {
                        score_window_batch_exact_packed(
                            &batch_haps,
                            &phased_target,
                            ref_columns,
                            n_ref_haps,
                            &alignment,
                            &mut global_scores,
                            &mut window_scores,
                        );
                    } else {
                        score_window_batch_pbwt_packed(
                            &batch_haps,
                            &phased_target,
                            ref_columns,
                            n_ref_haps,
                            &alignment,
                            gen_maps,
                            k_per_hap,
                            step_cm,
                            &mut global_scores,
                            &mut window_scores,
                        );
                    }

                    let abyss_rank_cutoff = ((n_ref_haps / 1000).max(ABYSS_RANK_BASE))
                        .min(n_ref_haps)
                        .max(1);
                    for (i, _) in batch_haps.iter().enumerate() {
                        for (h, score) in window_scores[i].iter().copied().enumerate() {
                            if score > best_window_scores[i][h] {
                                best_window_scores[i][h] = score;
                            }
                        }

                        let abyss_top = select_top_k(&window_scores[i], abyss_rank_cutoff);
                        for (ref_idx, _) in abyss_top {
                            if ref_idx < window_rank_hits[i].len() {
                                window_rank_hits[i][ref_idx] =
                                    window_rank_hits[i][ref_idx].saturating_add(1);
                            }
                        }
                    }

                    // Persist per-window sparse scores for LMS allocator (top-M per window).
                    for i in 0..batch_len {
                        let top_m = per_window_cap_window
                            .saturating_mul(PBWT_PER_WINDOW_MULT)
                            .max(per_window_cap_window)
                            .min(n_ref_haps.max(1));
                        let top = select_top_k(&window_scores[i], top_m);
                        scores_by_window[i].push(top);
                    }

                    window_idx += 1;
                    if let Some(bb) = telemetry {
                        bb.set_current_window(window_idx as u64);
                        bb.add_markers(1);
                    }
                }
            }
            ReferenceData::OnDisk { .. } => {
                let Some(ref mut ref_reader) = on_disk_reader else {
                    return Err(ReagleError::vcf(
                        "Prescan cache reader unavailable".to_string(),
                    ));
                };
                ref_reader.rewind()?;
                loop {
                    let ref_window = ref_reader.next_window()?;
                    let Some(ref_window) = ref_window else { break };

                    let idx = window_idx;
                    let n_ref_markers = ref_window.markers.len();
                    if n_ref_markers == 0 {
                        continue;
                    }
                    if window_span_cm.is_none() || window_span_bp.is_none() {
                        let start_pos = ref_window.markers.marker(MarkerIdx::new(0)).pos;
                        let end_pos = ref_window
                            .markers
                            .marker(MarkerIdx::new((n_ref_markers - 1) as u32))
                            .pos;
                        let span_bp = end_pos.saturating_sub(start_pos);
                        let start_cm = gen_maps.gen_pos(
                            ref_window.markers.marker(MarkerIdx::new(0)).chrom,
                            start_pos,
                        );
                        let end_cm = gen_maps.gen_pos(
                            ref_window.markers.marker(MarkerIdx::new(0)).chrom,
                            end_pos,
                        );
                        window_span_bp = Some(span_bp.into());
                        window_span_cm = Some((end_cm - start_cm).abs());
                    }
                    // Derive per-window cap from the observed marker count to match
                    // the real workspace footprint (fwd/bwd/history scale with markers).
                    let per_window_cap_window = per_window_caps
                        .get(window_idx)
                        .copied()
                        .unwrap_or(per_window_cap.max(1));

                    let (alignment, phased_target) = if let Some(cache) = target_cache.as_ref() {
                        if let Some(Some(entry)) = cache.get(idx) {
                            (
                                entry.alignment.clone(),
                                entry.phased_target.clone(),
                            )
                        } else {
                            let ref_chrom_idx =
                                ref_window.markers.marker(MarkerIdx::new(0)).chrom;
                            let ref_chrom = ref_window
                                .markers
                                .chrom_name(ref_chrom_idx)
                                .unwrap_or("UNKNOWN");
                            let start_pos = ref_window.markers.marker(MarkerIdx::new(0)).pos;
                            let end_pos = ref_window
                                .markers
                                .marker(MarkerIdx::new((n_ref_markers - 1) as u32))
                                .pos;

                            let chrom_candidates = chrom_variants(ref_chrom);
                            let reader = target_reader.as_mut().ok_or_else(|| {
                                ReagleError::vcf("Target reader missing in prescan".to_string())
                            })?;
                            let target_window = reader.load_window_for_region(
                                &chrom_candidates,
                                start_pos,
                                end_pos,
                            )?;
                            let Some(target_window) = target_window else {
                                continue;
                            };

                            let alignment = MarkerAlignment::new_with_ref_markers(
                                &target_window.genotypes,
                                &ref_window.markers,
                            );
                            let phased_target = target_window.genotypes.clone().into_phased();
                            (alignment, phased_target)
                        }
                    } else {
                        let ref_chrom_idx = ref_window.markers.marker(MarkerIdx::new(0)).chrom;
                        let ref_chrom = ref_window
                            .markers
                            .chrom_name(ref_chrom_idx)
                            .unwrap_or("UNKNOWN");
                        let start_pos = ref_window.markers.marker(MarkerIdx::new(0)).pos;
                        let end_pos = ref_window
                            .markers
                            .marker(MarkerIdx::new((n_ref_markers - 1) as u32))
                            .pos;

                        let chrom_candidates = chrom_variants(ref_chrom);
                        let reader = target_reader.as_mut().ok_or_else(|| {
                            ReagleError::vcf("Target reader missing in prescan".to_string())
                        })?;
                        let target_window = reader.load_window_for_region(
                            &chrom_candidates,
                            start_pos,
                            end_pos,
                        )?;
                        let Some(target_window) = target_window else {
                            continue;
                        };

                        let alignment = MarkerAlignment::new_with_ref_markers(
                            &target_window.genotypes,
                            &ref_window.markers,
                        );
                        let phased_target = target_window.genotypes.clone().into_phased();
                        (alignment, phased_target)
                    };

                    for i in 0..batch_len {
                        if global_scores[i].len() != n_ref_haps {
                            global_scores[i] = vec![0.0f32; n_ref_haps];
                            window_scores[i] = vec![f32::NEG_INFINITY; n_ref_haps];
                            best_window_scores[i] = vec![f32::NEG_INFINITY; n_ref_haps];
                            window_rank_hits[i] = vec![0u32; n_ref_haps];
                        }
                    }

                    for w in window_scores.iter_mut() {
                        w.fill(f32::NEG_INFINITY);
                    }

                    let k_per_hap = per_window_cap_window
                        .saturating_mul(PBWT_PER_WINDOW_MULT)
                        .max(PBWT_MIN_PER_HAP)
                        .min(PBWT_MAX_PER_HAP)
                        .max(1);

                    let step_cm = PBWT_SELECT_BLOCK_CM.max(imp_step_cm);
                    let use_exact = should_use_exact_prescan(
                        n_ref_haps,
                        batch_haps.len(),
                        phased_target.n_markers(),
                    );
                    if use_exact {
                        score_window_batch_exact_packed(
                            &batch_haps,
                            &phased_target,
                            &ref_window.columns,
                            n_ref_haps,
                            &alignment,
                            &mut global_scores,
                            &mut window_scores,
                        );
                    } else {
                        score_window_batch_pbwt_packed(
                            &batch_haps,
                            &phased_target,
                            &ref_window.columns,
                            n_ref_haps,
                            &alignment,
                            gen_maps,
                            k_per_hap,
                            step_cm,
                            &mut global_scores,
                            &mut window_scores,
                        );
                    }

                    let abyss_rank_cutoff = ((n_ref_haps / 1000).max(ABYSS_RANK_BASE))
                        .min(n_ref_haps)
                        .max(1);
                    for (i, _) in batch_haps.iter().enumerate() {
                        for (h, score) in window_scores[i].iter().copied().enumerate() {
                            if score > best_window_scores[i][h] {
                                best_window_scores[i][h] = score;
                            }
                        }

                        let abyss_top = select_top_k(&window_scores[i], abyss_rank_cutoff);
                        for (ref_idx, _) in abyss_top {
                            if ref_idx < window_rank_hits[i].len() {
                                window_rank_hits[i][ref_idx] =
                                    window_rank_hits[i][ref_idx].saturating_add(1);
                            }
                        }
                    }

                    // Persist per-window sparse scores for LMS allocator (top-M per window).
                    for i in 0..batch_len {
                        let top_m = per_window_cap_window
                            .saturating_mul(PBWT_PER_WINDOW_MULT)
                            .max(per_window_cap_window)
                            .min(n_ref_haps.max(1));
                        let top = select_top_k(&window_scores[i], top_m);
                        scores_by_window[i].push(top);
                    }

                    window_idx += 1;
                    if let Some(bb) = telemetry {
                        bb.set_current_window(window_idx as u64);
                        bb.add_markers(1);
                    }
                }
            }
        }

        let min_step_cm = (streaming_config.overlap_cm as f64)
            .max(imp_step_cm)
            .max(1e-6);
        let boundary_cm = window_boundaries_from_handoff(&window_handoff, min_step_cm);
        if !per_window_caps.is_empty() && per_window_caps.len() != window_handoff.len() {
            return Err(ReagleError::vcf(format!(
                "Per-window cap length mismatch (caps={}, bounds={})",
                per_window_caps.len(),
                window_handoff.len()
            )));
        }
        if window_idx != window_handoff.len() {
            return Err(ReagleError::vcf(format!(
                "Prescan window count mismatch (seen={}, bounds={})",
                window_idx,
                window_handoff.len()
            )));
        }

        let num_windows = window_handoff.len();
        let use_plan_caps = !plan.per_window_caps.is_empty() && plan.per_window_caps.len() == num_windows;
        let fallback_caps = if use_plan_caps {
            None
        } else {
            Some(vec![per_window_cap.max(1); num_windows])
        };
        let plan_caps = plan.per_window_caps.clone();
        let batch_results: Vec<_> = batch_haps
            .par_iter()
            .enumerate()
            .map(|(i, &hap_idx)| {
                let mut abyss = vec![false; n_ref_haps];
                let mut abyss_count = 0usize;
                for h in 0..n_ref_haps {
                    let score = best_window_scores[i][h];
                    if window_rank_hits[i][h] == 0 || !score.is_finite() || score <= 0.0 {
                        abyss[h] = true;
                        abyss_count += 1;
                    }
                }
                if abyss_count == n_ref_haps {
                    let keep = ((n_ref_haps / 1000).max(ABYSS_RANK_BASE))
                        .min(n_ref_haps)
                        .max(1);
                    let top = select_top_k_allow_zero(&global_scores[i], keep);
                    if top.is_empty() {
                        for h in 0..keep {
                            abyss[h] = false;
                        }
                        abyss_count = n_ref_haps.saturating_sub(keep);
                    } else {
                        for (h, _) in top {
                            if abyss[h] {
                                abyss[h] = false;
                                abyss_count = abyss_count.saturating_sub(1);
                            }
                        }
                    }
                }
                let window_scores_matrix = &scores_by_window[i];
                if window_scores_matrix.len() != window_handoff.len() {
                    return Err(ReagleError::vcf(format!(
                        "Pre-scan window count mismatch for hap {} (scores={}, bounds={})",
                        hap_idx,
                        window_scores_matrix.len(),
                        window_handoff.len()
                    )));
                }
                let per_window_caps_used = if use_plan_caps {
                    plan_caps.as_slice()
                } else {
                    fallback_caps.as_ref().unwrap().as_slice()
                };
                let per_window_cap_min = per_window_caps_used
                    .iter()
                    .copied()
                    .min()
                    .unwrap_or(per_window_cap.max(1));
                let (intervals, core) = if per_window_cap_min >= n_ref_haps {
                    let mut intervals = Vec::new();
                    let mut core = Vec::new();
                    let end = num_windows.saturating_sub(1) as u32;
                    for h in 0..n_ref_haps {
                        if abyss[h] {
                            continue;
                        }
                        let hap = RefHapId::new(h as u32);
                        intervals.push(HapIntervals {
                            hap,
                            intervals: vec![(0, end)],
                        });
                        core.push(hap);
                    }
                    (intervals, core)
                } else {
                    let (candidate_haps, scores_by_hap) =
                        build_sparse_scores(window_scores_matrix, &abyss);
                    let global_slot_budget =
                        per_window_caps_used.iter().copied().sum::<usize>().max(1);
                    let allocation = crate::model::state_allocator::allocate_lms_sparse(
                        &scores_by_hap,
                        &candidate_haps,
                        num_windows,
                        &boundary_cm,
                        params,
                        n_ref_haps,
                        global_slot_budget,
                        per_window_caps_used,
                    );
                    let mut intervals = Vec::new();
                    for (hap, spans) in allocation.intervals_by_hap.into_iter() {
                        intervals.push(HapIntervals {
                            hap: RefHapId::new(hap as u32),
                            intervals: spans,
                        });
                    }
                    let mut core = Vec::new();
                    let need_end = window_scores_matrix.len().saturating_sub(1) as u32;
                    for hi in intervals.iter() {
                        if hi.intervals.len() == 1 && hi.intervals[0] == (0, need_end) {
                            core.push(hi.hap);
                        }
                    }
                    (intervals, core)
                };
                let core_len = core.len();
                let intervals_len = intervals.len();
                Ok((hap_idx, abyss, intervals, core, core_len, intervals_len, abyss_count))
            })
            .collect::<Result<Vec<_>>>()?;

        for (hap_idx, abyss, intervals, core, core_len, intervals_len, abyss_count) in batch_results {
            plan.abyss_mask[hap_idx] = abyss;
            plan.window_intervals[hap_idx] = intervals;
            plan.core_states[hap_idx] = core;
            plan
                .stats
                .update(core_len, intervals_len.saturating_sub(core_len), abyss_count);
        }

        batch_start = batch_end;
    }
    let elapsed = prescan_start.elapsed().as_secs_f32();
    let cache_hit = target_cache
        .as_ref()
        .map(|c| c.iter().filter(|e| e.is_some()).count())
        .unwrap_or(0);
    let cache_total = target_cache.as_ref().map(|c| c.len()).unwrap_or(0);
    eprintln!(
        "Pre-scan summary: batches={} windows={} cache_hits={}/{} elapsed={:.1}s",
        batches_total.max(1),
        window_handoff.len(),
        cache_hit,
        cache_total,
        elapsed
    );

    Ok(plan)
}

struct SampleImputationResult {
    sample_idx: usize,
    hap_alt_probs: Option<(Vec<f32>, Vec<f32>)>,
    hap_posteriors: Option<(Vec<AllelePosteriors>, Vec<AllelePosteriors>)>,
}

struct ImputationHandoff {
    priors: Vec<HaplotypePriors>,
    prior_global_idx: Option<usize>,
    prior_gen_pos: Option<f64>,
}

struct ImputationWindowResults {
    all_results: Vec<SampleImputationResult>,
    ref_is_biallelic: Vec<bool>,
    handoff: Option<ImputationHandoff>,
}

impl crate::pipelines::ImputationPipeline {
    /// Run streaming imputation pipeline
    #[instrument(name = "imputation_streaming", skip(self))]
    pub fn run_streaming(&mut self) -> Result<()> {
        let streaming_config = StreamingConfig {
            window_cm: self.config.window,
            overlap_cm: self.config.overlap,
            buffer_cm: 1.0,
            max_markers: self.config.window_markers,
        };

        if let Some(bb) = &self.telemetry {
            bb.set_stage(crate::utils::telemetry::Stage::LoadingData);
            bb.set_producer_stage(crate::utils::telemetry::Stage::LoadingData);
            bb.set_consumer_stage(crate::utils::telemetry::Stage::LoadingData);
            bb.set_op("Preparing input");
            bb.set_producer_op("Preparing input");
        }

        let (target_positions_map, target_marker_count) =
            collect_target_positions(&self.config.gt)?;
        let target_positions = if target_marker_count == 0 {
            None
        } else {
            Some(Arc::new(target_positions_map.clone()))
        };
        if target_positions.is_none() {
            return Err(ReagleError::vcf(
                "No target markers found while building marker index".to_string(),
            ));
        }
        eprintln!("Target marker index: {} positions", target_marker_count);

        let gen_maps = if let Some(ref map_path) = self.config.map {
            let mut chrom_name_buf: Vec<String> = Vec::new();

            for chrom in target_positions_map.keys() {
                for variant in chrom_variants(chrom) {
                    if !chrom_name_buf.iter().any(|c| c == &variant) {
                        chrom_name_buf.push(variant);
                    }
                }
            }
            if let Some(chrom) = self.config.chrom.as_deref() {
                for variant in chrom_variants(chrom) {
                    if !chrom_name_buf.iter().any(|c| c == &variant) {
                        chrom_name_buf.push(variant);
                    }
                }
            }

            let chrom_name_refs: Vec<&str> = chrom_name_buf.iter().map(String::as_str).collect();
            GeneticMaps::from_plink_file(map_path, &chrom_name_refs)?
        } else {
            GeneticMaps::new()
        };

        let ref_path = self
            .config
            .r#ref
            .as_ref()
            .ok_or_else(|| ReagleError::config("Reference panel required for imputation"))?;
        let ref_path = ensure_binary_reference(ref_path, &self.config)?;

        let mut input_target_path = self.config.gt.clone();
        let mut input_tmp: Option<tempfile::TempDir> = None;
        if input_target_path.as_os_str() == "-" {
            let tmpdir = tempfile::tempdir()?;
            let tmp_path = tmpdir.path().join("stdin_target.vcf");
            let stdin = std::io::stdin();
            let mut reader = stdin.lock();
            let mut out = std::fs::File::create(&tmp_path)?;
            std::io::copy(&mut reader, &mut out)?;
            input_target_path = tmp_path;
            input_tmp = Some(tmpdir);
        } else if !input_target_path.exists() {
            return Err(ReagleError::config(format!(
                "Target VCF not found: {:?}",
                input_target_path
            )));
        }

    let mut phased_target_path = input_target_path.clone();
    let mut phased_tmp: Option<tempfile::TempDir> = None;
    // NOTE: imputation uses its own mismatch prior; we do not carry phasing error rates.
    let mut phased_recomb_intensity: Option<f32> = None;
    if !is_vcf_fully_phased(&phased_target_path)? {
        eprintln!("Target is unphased; running phasing before pre-scan...");
        let phased_prefix = if self.config.out.as_os_str() != "-" {
            let out = &self.config.out;
            let parent = out.parent().unwrap_or_else(|| std::path::Path::new("."));
            let stem = out
                .file_name()
                .unwrap_or_else(|| std::ffi::OsStr::new("reagle_out"));
            parent.join(format!("{}_phased_target", stem.to_string_lossy()))
        } else {
            let tmpdir = tempfile::tempdir()?;
            let prefix = tmpdir.path().join("phased_target");
            phased_tmp = Some(tmpdir);
            prefix
        };
        let mut phase_config = self.config.clone();
        phase_config.gt = input_target_path.clone();
        phase_config.r#ref = Some(ref_path.to_path_buf());
        phase_config.out = phased_prefix.clone();
        let mut phasing = crate::pipelines::phasing::PhasingPipeline::new(
            phase_config,
            self.telemetry.clone(),
        );
        phasing.run()?;
        phased_recomb_intensity = Some(phasing.params().recomb_intensity);
        phased_target_path = phased_prefix.with_extension("vcf.gz");
        if phased_tmp.is_none() {
            eprintln!("Phased target saved at {:?}", phased_target_path);
        }
    } else {
        eprintln!("Target already phased; skipping phasing before pre-scan.");
    }

        let mut n_threads = self
            .config
            .nthreads
            .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
            .unwrap_or(1);
        let mut avail_bytes = available_memory_bytes().unwrap_or(0);
        if avail_bytes < MIN_AVAIL_BYTES_FOR_PLANNING {
            // Treat unknown/low memory as "planning disabled" to avoid
            // tiny caps in CI/small test runs.
            avail_bytes = 0;
        }
        let min_states = 64usize;
        let mut raw_budget = estimate_state_budget(avail_bytes, n_threads, self.config.window_markers);
        loop {
            let total_budget = raw_budget.max(1);
            if total_budget >= min_states || n_threads <= 1 {
                break;
            }
            n_threads = (n_threads / 2).max(1);
            raw_budget = estimate_state_budget(avail_bytes, n_threads, self.config.window_markers);
        }
        let total_budget = raw_budget.max(1);
        let force_full_panel = avail_bytes == 0 || raw_budget == 0;
        let per_window_cap = if force_full_panel {
            usize::MAX
        } else {
            total_budget
        };
        let per_window_cap = per_window_cap.max(1);

        eprintln!(
            "Imputation plan: per_window_cap={}, threads={}, available_mb={}",
            per_window_cap,
            n_threads,
            avail_bytes / (1024 * 1024)
        );

        let safe_bytes_per_thread = if n_threads == 0 {
            0u64
        } else {
            let budget = (avail_bytes as f64 * IMPUTE_RAM_FRACTION) as u64;
            let per_thread = budget / n_threads as u64;
            (per_thread as f64 * STATE_BUDGET_SAFETY) as u64
        };
        let prescan_force_full_panel = avail_bytes < MIN_AVAIL_BYTES_FOR_PLANNING;
        if let Some(bb) = &self.telemetry {
            bb.set_stage(crate::utils::telemetry::Stage::ImputationPrescan);
            bb.set_producer_stage(crate::utils::telemetry::Stage::ImputationPrescan);
            bb.set_op("Imputation prescan: reference prep");
        }
        let desired_cap = if self.config.imp_states > 0 {
            Some(self.config.imp_states)
        } else {
            None
        };
        let ref_data = prepare_reference_data(
            &ref_path,
            &streaming_config,
            &gen_maps,
            &target_positions_map,
            if force_full_panel { 0 } else { avail_bytes },
            n_threads,
            safe_bytes_per_thread,
            prescan_force_full_panel,
            desired_cap,
        )?;

        match &ref_data {
            ReferenceData::InMemory { .. } => {
                eprintln!("Reference mode: in-memory (single-pass)");
            }
            ReferenceData::OnDisk { guard, .. } => {
                eprintln!("Reference mode: prescan cache (double-pass)");
                let _ = guard;
            }
        }

        if let Some(bb) = &self.telemetry {
            bb.set_stage(crate::utils::telemetry::Stage::ImputationPlanning);
            bb.set_producer_stage(crate::utils::telemetry::Stage::ImputationPlanning);
            bb.set_op("Imputation planning: window caps");
        }
        let plan = build_imputation_plan(
            &phased_target_path,
            &streaming_config,
            &gen_maps,
            per_window_cap,
            if force_full_panel { 0 } else { avail_bytes },
            self.config.imp_step as f64,
            &self.params,
            &ref_data,
            self.telemetry.as_ref(),
        )?;

        self.params.recomb_intensity = (0.04 * self.config.ne / plan.n_ref_haps as f32)
            .min(ModelParams::MAX_RECOMB_INTENSITY);

        let full_panel_cap = plan
            .per_window_caps
            .iter()
            .copied()
            .min()
            .unwrap_or(plan.per_window_cap);
        let full_panel = full_panel_cap >= plan.n_ref_haps;
        if full_panel {
            eprintln!("Imputation mode: full panel per window (abyss still active)");
        } else {
            eprintln!(
                "Imputation mode: LMS allocation (per_window_cap={})",
                plan.per_window_cap
            );
        }
        log_imputation_plan_summary(&plan);

        if let Some(bb) = &self.telemetry {
            bb.set_total_windows(plan.per_window_caps.len() as u64);
            bb.set_current_window(0);
        }

        let mut ref_data = ref_data;
        let target_was_unphased_for_impute = !is_vcf_fully_phased(&input_target_path)?;
        // Always use phased target haplotypes for emissions to preserve LD signal.
        // If the input was unphased, we will still treat phase as uncertain in
        // the emission model (heterozygotes → 0.5/0.5 per haplotype).
        let use_phased_for_impute = true;
        let target_path_for_impute = phased_target_path.clone();
        if target_was_unphased_for_impute {
            eprintln!("Target was unphased at input; using phased target for imputation (phase-uncertain emissions).");
        } else {
            eprintln!("Target already phased; using phased target directly for imputation.");
        }
        let mut target_reader = StreamingVcfReader::open(
            &target_path_for_impute,
            gen_maps.clone(),
            streaming_config.clone(),
        )?;
        let mut target_reader_pl = if use_phased_for_impute {
            // Pull PL/GL from the original target VCF (if present).
            Some(StreamingVcfReader::open(
                &input_target_path,
                gen_maps.clone(),
                streaming_config.clone(),
            )?)
        } else {
            None
        };
        // Imputation HMM copies from the reference panel; align LS parameters
        // to the donor pool size (reference haplotypes), not target+reference.
        let n_ref_pool = plan.n_ref_haps.max(1);
        self.params = crate::model::parameters::ModelParams::for_phasing(
            n_ref_pool,
            self.config.ne,
            self.config.err,
        );
        self.params.recomb_intensity = self
            .params
            .recomb_intensity
            .min(crate::model::parameters::ModelParams::MAX_RECOMB_INTENSITY);
        if let Some(recomb_intensity) = phased_recomb_intensity {
            if recomb_intensity.is_finite() && recomb_intensity > 0.0 {
                self.params.recomb_intensity = recomb_intensity.min(
                    crate::model::parameters::ModelParams::MAX_RECOMB_INTENSITY,
                );
            }
        }
        // Do not inherit phasing mismatch estimates for imputation. Imputation
        // should use the Li-Stephens mismatch prior (or user override) tied to
        // the reference panel, not phasing-specific error rates.
        self.params
            .set_n_states(n_ref_pool.saturating_sub(2).max(1));

        let target_samples = target_reader.samples_arc();
        let n_target_samples = target_samples.len();
        if n_target_samples == 0 {
            return Err(ReagleError::vcf(
                "No target samples found in input VCF".to_string(),
            ));
        }

        let output_path = self.config.out.with_extension("vcf.gz");
        eprintln!("Writing output to {:?}", output_path);
        let mut writer = VcfWriter::create(&output_path, target_samples.clone())?;

        let mut imp_overlap: Option<PhasedOverlap> = None;
        let mut warned_no_overlap = false;
        let mut header_written = false;
        let mut total_markers = 0usize;
        let mut window_idx = 0usize;
        let mut sample_error_rates = vec![self.params.p_mismatch.clamp(1e-6, 0.5); n_target_samples];

        match &mut ref_data {
            ReferenceData::InMemory { windows, .. } => {
                for ref_window in windows.iter_mut() {
                    if let Some(bb) = &self.telemetry {
                        bb.set_producer_stage(crate::utils::telemetry::Stage::LoadingData);
                        bb.set_producer_op(&format!("Loading ref window {}", window_idx + 1));
                        bb.set_op(&format!("Loading ref window {}", window_idx + 1));
                        bb.set_current_window((window_idx + 1) as u64);
                    }

                    let n_ref_markers = ref_window.markers.len();
                    if n_ref_markers == 0 {
                        window_idx += 1;
                        continue;
                    }

                    let ref_chrom_idx = ref_window.markers.marker(MarkerIdx::new(0)).chrom;
                    let ref_chrom = ref_window
                        .markers
                        .chrom_name(ref_chrom_idx)
                        .unwrap_or("UNKNOWN");
                    let start_pos = ref_window.markers.marker(MarkerIdx::new(0)).pos;
                    let end_pos = ref_window
                        .markers
                        .marker(MarkerIdx::new((n_ref_markers - 1) as u32))
                        .pos;

                    let chrom_candidates = chrom_variants(ref_chrom);
                    let target_window = target_reader.load_window_for_region(
                        &chrom_candidates,
                        start_pos,
                        end_pos,
                    )?;
                    let Some(target_window) = target_window else {
                        window_idx += 1;
                        continue;
                    };
                    let target_window_pl = if let Some(reader_pl) = target_reader_pl.as_mut() {
                        reader_pl.load_window_for_region(&chrom_candidates, start_pos, end_pos)?
                    } else {
                        None
                    };

                    // Align using the phased target markers; PL/GL lookups share the same
                    // marker set, so indices should remain consistent.
                    let alignment = MarkerAlignment::new_with_ref_markers(
                        &target_window.genotypes,
                        &ref_window.markers,
                    );

                    let phased_target = target_window.genotypes.clone().into_phased();
                    let phased_target_pl = target_window_pl
                        .as_ref()
                        .map(|w| w.genotypes.clone().into_phased());
                    let target_missing = if target_was_unphased_for_impute {
                        target_window_pl.as_ref().map(|w| &w.genotypes)
                    } else {
                        None
                    };
                    if !header_written {
                        writer.write_header_extended(
                            &ref_window.markers,
                            true,
                            self.config.gp,
                            self.config.ap,
                        )?;
                        header_written = true;
                    }

                    let should_log = phased_target.n_markers() >= 100 || window_idx % 1000 == 0;
                    if should_log {
                        eprintln!(
                            "  Imputing Window {} ({} markers, ref global {}..{}, output {}..{})",
                            window_idx,
                            phased_target.n_markers(),
                            ref_window.global_start,
                            ref_window.global_end,
                            ref_window.output_start,
                            ref_window.output_end
                        );
                    }

                    let n_alleles_per_marker: Vec<usize> = (0..ref_window.markers.len())
                        .map(|m| {
                            let marker = ref_window.markers.marker(MarkerIdx::new(m as u32));
                            1 + marker.alt_alleles.len()
                        })
                        .collect();
                    let mut window_quality = ImputationQuality::new(&n_alleles_per_marker);
                    #[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
                    struct TargetChromIdx(ChromIdx);
                    #[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
                    struct TargetChromPos {
                        chrom: TargetChromIdx,
                        pos: u32,
                    }

                    let mut target_chroms: std::collections::HashMap<String, TargetChromIdx> =
                        std::collections::HashMap::new();
                    for c in 0..target_window.genotypes.markers().chrom_names().len() {
                        let idx = ChromIdx::new(c as u16);
                        let name = target_window
                            .genotypes
                            .markers()
                            .chrom_name(idx)
                            .unwrap_or("");
                        target_chroms.insert(
                            normalize_chrom_local(name).to_string(),
                            TargetChromIdx(idx),
                        );
                    }

                    let mut target_positions: std::collections::HashSet<TargetChromPos> =
                        std::collections::HashSet::with_capacity(
                            target_window.genotypes.n_markers(),
                        );
                    for m in 0..target_window.genotypes.n_markers() {
                        let marker = target_window
                            .genotypes
                            .markers()
                            .marker(MarkerIdx::new(m as u32));
                        target_positions
                            .insert(TargetChromPos { chrom: TargetChromIdx(marker.chrom), pos: marker.pos });
                    }
                    for (ref_m, target_idx) in alignment.ref_to_target.iter().enumerate() {
                        let ref_marker =
                            ref_window.markers.marker(MarkerIdx::new(ref_m as u32));
                        let ref_chrom = ref_window
                            .markers
                            .chrom_name(ref_marker.chrom)
                            .unwrap_or("");
                        let ref_key = target_chroms
                            .get(normalize_chrom_local(ref_chrom))
                            .copied()
                            .map(|chrom| TargetChromPos { chrom, pos: ref_marker.pos });
                        let is_present = target_idx.is_some()
                            || ref_key
                                .map(|key| target_positions.contains(&key))
                                .unwrap_or(false);
                        window_quality.set_imputed(ref_m, !is_present);
                    }

                    let window_results = self.run_imputation_window_streaming(
                        &phased_target,
                        phased_target_pl.as_ref(),
                        target_missing,
                        &ref_window.markers,
                        &ref_window.ref_columns,
                        &alignment,
                        &gen_maps,
                        imp_overlap.as_ref(),
                        &plan,
                        window_idx,
                        ref_window.global_start,
                        ref_window.output_start,
                        ref_window.output_end,
                        // Use phase confidence from the phased target haplotypes. If the
                        // input was unphased, the phasing pipeline provides calibrated
                        // phase confidence for heterozygotes, which we should leverage
                        // to preserve LD signal in imputation emissions.
                        true,
                        &mut sample_error_rates,
                    )?;

                    let mut next_handoff = None;
                    if let Some(window_results) = window_results {
                        let ImputationWindowResults {
                            all_results,
                            ref_is_biallelic,
                            handoff,
                        } = window_results;
                        next_handoff = handoff;
                        // Drop heavy reference data before writing to reduce peak RSS.
                        // Drop reference genotypes/columns to free large buffers before write.
                        let _ = std::mem::take(&mut ref_window.ref_columns);
                        ref_window.ref_genotypes = None;

                        if let Some(bb) = &self.telemetry {
                            let output_markers =
                                ref_window.output_end.saturating_sub(ref_window.output_start);
                            bb.set_stage(crate::utils::telemetry::Stage::WritingOutput);
                            bb.set_consumer_stage(crate::utils::telemetry::Stage::WritingOutput);
                            bb.set_total_markers(output_markers as u64);
                            bb.set_markers_processed(0);
                            bb.set_total_samples(phased_target.n_samples() as u64);
                            bb.set_samples_processed(0);
                            bb.set_op(&format!(
                                "Writing window {} ({} markers)",
                                window_idx, output_markers
                            ));
                            bb.set_consumer_op(&format!(
                                "Writing window {} ({} markers)",
                                window_idx, output_markers
                            ));
                        }

                        self.write_imputed_window_streaming(
                            &ref_window.markers,
                            &phased_target,
                            phased_target_pl.as_ref(),
                            &alignment,
                            &mut writer,
                            &mut window_quality,
                            &ref_is_biallelic,
                            ref_window.output_start,
                            ref_window.output_end,
                            ref_window.output_start,
                            &all_results,
                            self.config.gp,
                            self.config.ap,
                            self.config.err.is_some(),
                        )?;

                        if let Some(bb) = &self.telemetry {
                            let output_markers =
                                ref_window.output_end.saturating_sub(ref_window.output_start);
                            bb.set_markers_processed(output_markers as u64);
                            bb.set_samples_processed(phased_target.n_samples() as u64);
                            bb.set_stage(crate::utils::telemetry::Stage::Imputation);
                            bb.set_consumer_stage(crate::utils::telemetry::Stage::Imputation);
                        }

                    }

                    total_markers += ref_window.output_end.saturating_sub(ref_window.output_start);

                    let mut next_overlap = self.extract_imputed_overlap_streaming(
                        &phased_target,
                        &ref_window.markers,
                        &alignment,
                        ref_window.output_end,
                    );
                    if let Some(handoff) = next_handoff {
                        next_overlap.set_hap_priors(handoff.priors);
                        if let Some(idx) = handoff.prior_global_idx {
                            next_overlap.set_prior_stage1_global_marker(idx);
                        }
                        if let Some(gen_pos) = handoff.prior_gen_pos {
                            next_overlap.set_prior_stage1_gen_pos(gen_pos);
                        }
                    } else if !warned_no_overlap {
                        warn!(
                            "No overlap handoff for window {} (empty/short window or region boundary)",
                            window_idx
                        );
                        warned_no_overlap = true;
                    }
                    imp_overlap = Some(next_overlap);

                    window_idx += 1;
                }
            }
            ReferenceData::OnDisk { .. } => {
                let mut ref_reader = open_ref_reader(&ref_path)?;
                loop {
                    if let Some(bb) = &self.telemetry {
                        bb.set_producer_stage(crate::utils::telemetry::Stage::LoadingData);
                        bb.set_producer_op(&format!("Loading ref window {}", window_idx + 1));
                        bb.set_op(&format!("Loading ref window {}", window_idx + 1));
                        bb.set_current_window((window_idx + 1) as u64);
                    }
                    let ref_window = ref_reader.next_window(
                        &streaming_config,
                        &gen_maps,
                        Some(&target_positions_map),
                    )?;
                    let Some(mut ref_window) = ref_window else { break };

                    let n_ref_markers = ref_window.markers.len();
                    if n_ref_markers == 0 {
                        window_idx += 1;
                        continue;
                    }

                    let ref_chrom_idx = ref_window.markers.marker(MarkerIdx::new(0)).chrom;
                    let ref_chrom = ref_window
                        .markers
                        .chrom_name(ref_chrom_idx)
                        .unwrap_or("UNKNOWN");
                    let start_pos = ref_window.markers.marker(MarkerIdx::new(0)).pos;
                    let end_pos = ref_window
                        .markers
                        .marker(MarkerIdx::new((n_ref_markers - 1) as u32))
                        .pos;

                    let chrom_candidates = chrom_variants(ref_chrom);
                    let target_window = target_reader.load_window_for_region(
                        &chrom_candidates,
                        start_pos,
                        end_pos,
                    )?;
                    let Some(target_window) = target_window else {
                        window_idx += 1;
                        continue;
                    };
                    let target_window_pl = if let Some(reader_pl) = target_reader_pl.as_mut() {
                        reader_pl.load_window_for_region(&chrom_candidates, start_pos, end_pos)?
                    } else {
                        None
                    };

                    // Align using the phased target markers; PL/GL lookups share the same
                    // marker set, so indices should remain consistent.
                    let alignment = MarkerAlignment::new_with_ref_markers(
                        &target_window.genotypes,
                        &ref_window.markers,
                    );

                    let phased_target = target_window.genotypes.clone().into_phased();
                    let phased_target_pl = target_window_pl
                        .as_ref()
                        .map(|w| w.genotypes.clone().into_phased());
                    let target_missing = if target_was_unphased_for_impute {
                        target_window_pl.as_ref().map(|w| &w.genotypes)
                    } else {
                        None
                    };
                    if !header_written {
                        writer.write_header_extended(
                            &ref_window.markers,
                            true,
                            self.config.gp,
                            self.config.ap,
                        )?;
                        header_written = true;
                    }

                    let should_log = phased_target.n_markers() >= 100 || window_idx % 1000 == 0;
                    if should_log {
                        eprintln!(
                            "  Imputing Window {} ({} markers, ref global {}..{}, output {}..{})",
                            window_idx,
                            phased_target.n_markers(),
                            ref_window.global_start,
                            ref_window.global_end,
                            ref_window.output_start,
                            ref_window.output_end
                        );
                    }

                    let n_alleles_per_marker: Vec<usize> = (0..ref_window.markers.len())
                        .map(|m| {
                            let marker = ref_window.markers.marker(MarkerIdx::new(m as u32));
                            1 + marker.alt_alleles.len()
                        })
                        .collect();
                    let mut window_quality = ImputationQuality::new(&n_alleles_per_marker);
                    #[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
                    struct TargetChromIdx(ChromIdx);
                    #[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
                    struct TargetChromPos {
                        chrom: TargetChromIdx,
                        pos: u32,
                    }

                    let mut target_chroms: std::collections::HashMap<String, TargetChromIdx> =
                        std::collections::HashMap::new();
                    for c in 0..target_window.genotypes.markers().chrom_names().len() {
                        let idx = ChromIdx::new(c as u16);
                        let name = target_window
                            .genotypes
                            .markers()
                            .chrom_name(idx)
                            .unwrap_or("");
                        target_chroms.insert(
                            normalize_chrom_local(name).to_string(),
                            TargetChromIdx(idx),
                        );
                    }

                    let mut target_positions: std::collections::HashSet<TargetChromPos> =
                        std::collections::HashSet::with_capacity(
                            target_window.genotypes.n_markers(),
                        );
                    for m in 0..target_window.genotypes.n_markers() {
                        let marker = target_window
                            .genotypes
                            .markers()
                            .marker(MarkerIdx::new(m as u32));
                        target_positions
                            .insert(TargetChromPos { chrom: TargetChromIdx(marker.chrom), pos: marker.pos });
                    }
                    for (ref_m, target_idx) in alignment.ref_to_target.iter().enumerate() {
                        let ref_marker =
                            ref_window.markers.marker(MarkerIdx::new(ref_m as u32));
                        let ref_chrom = ref_window
                            .markers
                            .chrom_name(ref_marker.chrom)
                            .unwrap_or("");
                        let ref_key = target_chroms
                            .get(normalize_chrom_local(ref_chrom))
                            .copied()
                            .map(|chrom| TargetChromPos { chrom, pos: ref_marker.pos });
                        let is_present = target_idx.is_some()
                            || ref_key
                                .map(|key| target_positions.contains(&key))
                                .unwrap_or(false);
                        window_quality.set_imputed(ref_m, !is_present);
                    }

                    let window_results = self.run_imputation_window_streaming(
                        &phased_target,
                        phased_target_pl.as_ref(),
                        target_missing,
                        &ref_window.markers,
                        &ref_window.ref_columns,
                        &alignment,
                        &gen_maps,
                        imp_overlap.as_ref(),
                        &plan,
                        window_idx,
                        ref_window.global_start,
                        ref_window.output_start,
                        ref_window.output_end,
                        // Use phase confidence from the phased target haplotypes. If the
                        // input was unphased, the phasing pipeline provides calibrated
                        // phase confidence for heterozygotes, which we should leverage
                        // to preserve LD signal in imputation emissions.
                        true,
                        &mut sample_error_rates,
                    )?;

                    let mut next_handoff = None;
                    if let Some(window_results) = window_results {
                        let ImputationWindowResults {
                            all_results,
                            ref_is_biallelic,
                            handoff,
                        } = window_results;
                        next_handoff = handoff;
                        // Drop heavy reference data before writing to reduce peak RSS.
                        // Drop reference genotypes/columns to free large buffers before write.
                        let _ = std::mem::take(&mut ref_window.ref_columns);
                        ref_window.ref_genotypes = None;

                        if let Some(bb) = &self.telemetry {
                            let output_markers =
                                ref_window.output_end.saturating_sub(ref_window.output_start);
                            bb.set_stage(crate::utils::telemetry::Stage::WritingOutput);
                            bb.set_consumer_stage(crate::utils::telemetry::Stage::WritingOutput);
                            bb.set_total_markers(output_markers as u64);
                            bb.set_markers_processed(0);
                            bb.set_total_samples(phased_target.n_samples() as u64);
                            bb.set_samples_processed(0);
                            bb.set_op(&format!(
                                "Writing window {} ({} markers)",
                                window_idx, output_markers
                            ));
                            bb.set_consumer_op(&format!(
                                "Writing window {} ({} markers)",
                                window_idx, output_markers
                            ));
                        }

                        self.write_imputed_window_streaming(
                            &ref_window.markers,
                            &phased_target,
                            phased_target_pl.as_ref(),
                            &alignment,
                            &mut writer,
                            &mut window_quality,
                            &ref_is_biallelic,
                            ref_window.output_start,
                            ref_window.output_end,
                            ref_window.output_start,
                            &all_results,
                            self.config.gp,
                            self.config.ap,
                            self.config.err.is_some(),
                        )?;

                        if let Some(bb) = &self.telemetry {
                            let output_markers =
                                ref_window.output_end.saturating_sub(ref_window.output_start);
                            bb.set_markers_processed(output_markers as u64);
                            bb.set_samples_processed(phased_target.n_samples() as u64);
                            bb.set_stage(crate::utils::telemetry::Stage::Imputation);
                            bb.set_consumer_stage(crate::utils::telemetry::Stage::Imputation);
                        }

                    }

                    total_markers += ref_window.output_end.saturating_sub(ref_window.output_start);

                    let mut next_overlap = self.extract_imputed_overlap_streaming(
                        &phased_target,
                        &ref_window.markers,
                        &alignment,
                        ref_window.output_end,
                    );
                    if let Some(handoff) = next_handoff {
                        next_overlap.set_hap_priors(handoff.priors);
                        if let Some(idx) = handoff.prior_global_idx {
                            next_overlap.set_prior_stage1_global_marker(idx);
                        }
                        if let Some(gen_pos) = handoff.prior_gen_pos {
                            next_overlap.set_prior_stage1_gen_pos(gen_pos);
                        }
                    } else if !warned_no_overlap {
                        warn!(
                            "No overlap handoff for window {} (empty/short window or region boundary)",
                            window_idx
                        );
                        warned_no_overlap = true;
                    }
                    imp_overlap = Some(next_overlap);

                    window_idx += 1;
                }
            }
        }

        if let Some(tmpdir) = input_tmp.as_ref() {
            tracing::trace!(path = ?tmpdir.path(), "Using temp dir for stdin target");
        }
        if let Some(tmpdir) = phased_tmp.as_ref() {
            tracing::trace!(path = ?tmpdir.path(), "Using temp dir for phased target");
        }

        if total_markers == 0 {
            return Err(ReagleError::vcf(
                "No markers imputed; check reference/target overlap and region selection.",
            ));
        }

        eprintln!("Streaming imputation complete: {} markers", total_markers);
        Ok(())
    }
    fn run_imputation_window_streaming<
        TargetSpace: Sync,
        RefMarkerSpace: Sync,
        TargetMissingState: PhaseState + Sync,
    >(
        &self,
        target_win: &GenotypeMatrix<Phased, TargetSpace>,
        target_pl: Option<&GenotypeMatrix<Phased, TargetSpace>>,
        target_missing: Option<&GenotypeMatrix<TargetMissingState, TargetSpace>>,
        ref_markers: &crate::data::marker::Markers<RefMarkerSpace>,
        ref_columns: &[GenotypeColumn],
        alignment: &MarkerAlignment<TargetSpace, RefMarkerSpace>,
        gen_maps: &GeneticMaps,
        imp_overlap: Option<&PhasedOverlap>,
        plan: &ImputationPlan,
        window_idx: usize,
        global_start: usize,
        output_start: usize,
        output_end: usize,
        phase_conf_valid: bool,
        sample_error_rates: &mut [f32],
    ) -> Result<Option<ImputationWindowResults>> {
        let window_span = if self.config.profile {
            Some(
                info_span!(
                    "imputation_window_compute",
                    ref_markers = ref_markers.len(),
                    target_markers = target_win.n_markers(),
                    output_start,
                    output_end
                )
                .entered(),
            )
        } else {
            None
        };
        if let Some(span) = &window_span {
            tracing::trace!(id = ?span.id(), "Entered imputation window span");
        }

        let n_ref_markers = ref_markers.len();
        let n_target_samples = target_win.n_samples();
        let output_markers = output_end.saturating_sub(output_start);

        if output_start >= output_end || n_ref_markers == 0 {
            return Ok(None);
        }
        if let Some(bb) = &self.telemetry {
            bb.set_stage(crate::utils::telemetry::Stage::Imputation);
            bb.set_consumer_stage(crate::utils::telemetry::Stage::Imputation);
            bb.set_producer_stage(crate::utils::telemetry::Stage::Imputation);
            bb.set_total_windows(plan.per_window_caps.len() as u64);
            bb.set_current_window((window_idx + 1) as u64);
            bb.set_total_markers(output_markers as u64);
            bb.set_markers_processed(0);
            bb.set_total_samples(n_target_samples as u64);
            bb.set_samples_processed(0);
            bb.set_op(&format!(
                "Imputing window {} ({} markers)",
                window_idx + 1,
                output_markers
            ));
            bb.set_producer_op(&format!(
                "Imputing window {} ({} markers)",
                window_idx + 1,
                output_markers
            ));
            bb.set_consumer_op(&format!(
                "Imputing window {} ({} markers)",
                window_idx + 1,
                output_markers
            ));
        }

        let ref_is_biallelic: Vec<bool> = (0..n_ref_markers)
            .map(|m| {
                ref_markers
                    .marker(MarkerIdx::new(m as u32))
                    .alt_alleles
                    .len()
                    == 1
            })
            .collect();

        let ref_allele_freqs = RefAlleleFreqs::new(ref_columns, ref_markers);

        let marker_map = {
            let chrom = ref_markers
                .marker(MarkerIdx::new(0))
                .chrom;
            let min_cluster_cm = (self.config.cluster as f64).max(1e-8);
            if let Some(gen_map) = gen_maps.get(chrom) {
                // External map: enforce a minimum distance based on the cluster
                // parameter to avoid near-zero recombination in dense regions.
                crate::data::genetic_map::MarkerMap::from_gen_map_with_min_dist(
                    ref_markers,
                    gen_map,
                    min_cluster_cm,
                )
            } else {
                // No external map: use linear map (1 cM per Mb) with a minimum
                // distance derived from the cluster parameter.
                crate::data::genetic_map::MarkerMap::from_positions_with_min_dist(
                    ref_markers,
                    min_cluster_cm,
                )
            }
        };
        let gen_positions: Vec<f64> = marker_map.gen_positions().to_vec();
        if let (Some(first), Some(last)) = (gen_positions.first(), gen_positions.last()) {
            let total_cm = (last - first).abs();
            eprintln!(
                "    genetic span: {:.6} cM across {} markers",
                total_cm,
                gen_positions.len()
            );
        }
        let mut p_recomb: Vec<f32> = Vec::with_capacity(n_ref_markers);
        p_recomb.push(0.0f32);
        for m in 1..n_ref_markers {
            let dist_cm = (gen_positions[m] - gen_positions[m - 1]).abs();
            p_recomb.push(self.params.p_recomb(dist_cm));
        }

        if let Some(min) = p_recomb.iter().copied().reduce(f32::min) {
            let max = p_recomb
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let mean = p_recomb.iter().copied().sum::<f32>() / p_recomb.len().max(1) as f32;
            eprintln!(
                "    p_recomb stats: min={:.6} mean={:.6} max={:.6}",
                min, mean, max
            );
        }

        if let Some(overlap) = imp_overlap {
            if let Some(prev_gen_pos) = overlap.prior_stage1_gen_pos() {
                if let Some(current_gen_pos) = gen_positions.first().copied() {
                    let dist_cm = (current_gen_pos - prev_gen_pos).abs();
                    if dist_cm > 0.0 && !dist_cm.is_nan() {
                        p_recomb[0] = self.params.p_recomb(dist_cm);
                    }
                }
            }
        }

        let per_window_cap_local = plan
            .per_window_caps
            .get(window_idx)
            .copied()
            .unwrap_or(plan.per_window_cap)
            .max(1);
        let full_states = if plan.full_panel {
            let mut full: Vec<RefHapId> = Vec::with_capacity(plan.n_ref_haps);
            for h in 0..plan.n_ref_haps {
                full.push(RefHapId::new(h as u32));
            }
            Some(full)
        } else {
            None
        };

        let mut state_haps_by_hap: Vec<Vec<RefHapId>> = Vec::with_capacity(n_target_samples * 2);
        if full_states.is_none() {
            for hap_idx in 0..(n_target_samples * 2) {
                let mut state_haps: Vec<RefHapId> = Vec::new();
                if hap_idx < plan.window_intervals.len() {
                    for hi in plan.window_intervals[hap_idx].iter() {
                        if hi.contains(window_idx) {
                            state_haps.push(hi.hap);
                            if state_haps.len() >= per_window_cap_local {
                                break;
                            }
                        }
                    }
                }
                if state_haps.is_empty() && hap_idx < plan.core_states.len() {
                    state_haps.extend(plan.core_states[hap_idx].iter().copied());
                }
                state_haps.sort_unstable_by_key(|g| g.as_u32());
                state_haps.dedup();
                if hap_idx < plan.abyss_mask.len() {
                    let abyss = &plan.abyss_mask[hap_idx];
                    state_haps.retain(|g| !abyss[g.as_usize()]);
                }
                if state_haps.is_empty() {
                    // Hard fallback: pick the first non-abyss haplotypes.
                    if hap_idx < plan.abyss_mask.len() {
                        let abyss = &plan.abyss_mask[hap_idx];
                        for h in 0..plan.n_ref_haps {
                            if !abyss.get(h).copied().unwrap_or(true) {
                                state_haps.push(RefHapId::new(h as u32));
                                if state_haps.len() >= per_window_cap_local {
                                    break;
                                }
                            }
                        }
                    }
                }
                assert!(
                    !state_haps.is_empty(),
                    "State selection produced empty haplotype set"
                );
                state_haps_by_hap.push(state_haps);
            }
        }

        let prior_mappers_by_hap: Option<Vec<Option<TransitionMatrix>>> =
            imp_overlap.and_then(|o| o.hap_priors()).map(|hp| {
                let mut out: Vec<Option<TransitionMatrix>> = Vec::with_capacity(n_target_samples * 2);
                for hap_idx in 0..(n_target_samples * 2) {
                    let priors = hp.get(hap_idx);
                    if let Some(p) = priors {
                        if p.is_empty() {
                            out.push(None);
                            continue;
                        }
                        let prev_states: Vec<RefHapId> =
                            p.ids().iter().map(|id| RefHapId::new(id.0)).collect();
                        let next_states = if let Some(full) = full_states.as_ref() {
                            full.as_slice()
                        } else {
                            state_haps_by_hap
                                .get(hap_idx)
                                .map(|v| v.as_slice())
                                .unwrap_or(&[])
                        };
                        if next_states.is_empty() {
                            out.push(None);
                        } else {
                            out.push(Some(TransitionMatrix::build(&prev_states, next_states)));
                        }
                    } else {
                        out.push(None);
                    }
                }
                out
            });

        let normalize_probs = |probs: &mut [f32]| -> bool {
            let mut sum = 0.0f32;
            for p in probs.iter_mut() {
                if *p < 0.0 {
                    *p = 0.0;
                }
                sum += *p;
            }
            if sum > 0.0 {
                for p in probs.iter_mut() {
                    *p /= sum;
                }
                true
            } else {
                false
            }
        };


        let target_samples = target_win.samples_arc();
        let target_pl_matrix = target_pl.unwrap_or(target_win);
        let genotyped_fraction = (alignment
            .ref_to_target
            .iter()
            .filter(|v| v.is_some())
            .count() as f32
            / n_ref_markers.max(1) as f32)
            .clamp(0.0, 1.0);
        let err_floor = if genotyped_fraction < 0.01 {
            0.005f32
        } else {
            self.params.p_mismatch
        };
        let err_rate = self.params.p_mismatch.max(err_floor).clamp(1e-6, 0.5);
        let build_input_probs_pair = |hap1: HapIdx, hap2: HapIdx, sample_idx: usize| -> (TargetAlleleProbs, TargetAlleleProbs) {
            let mut offsets1 = Vec::with_capacity(n_ref_markers + 1);
            let mut offsets2 = Vec::with_capacity(n_ref_markers + 1);
            let mut probs1: Vec<f32> = Vec::new();
            let mut probs2: Vec<f32> = Vec::new();
            offsets1.push(0);
            offsets2.push(0);
            let mut dbg_genotyped = 0usize;
            let mut dbg_hard_used = 0usize;
            let mut dbg_pl_present_with_hard = 0usize;
            let mut dbg_pl_uniform_with_hard = 0usize;
            let is_uniform = |vals: &[f32]| -> bool {
                if vals.len() <= 1 {
                    return true;
                }
                let mut min = vals[0];
                let mut max = vals[0];
                for &v in vals.iter().skip(1) {
                    if v < min {
                        min = v;
                    }
                    if v > max {
                        max = v;
                    }
                }
                (max - min) <= 1e-6
            };

            for (ref_m, target_m_idx) in alignment.ref_to_target.iter().enumerate() {
                let n_alleles = ref_markers.marker(MarkerIdx::new(ref_m as u32)).n_alleles();
                let mut aligned1: Vec<f32> = Vec::new();
                let mut aligned2: Vec<f32> = Vec::new();
                let mut use1 = false;
                let mut use2 = false;

                if let Some(target_m_idx) = target_m_idx {
                    let target_m = target_m_idx.as_usize();
                    let conf_base = target_pl_matrix
                        .sample_confidence_f32(MarkerIdx::new(target_m as u32), sample_idx);
                    let mut conf1 = conf_base;
                    let mut conf2 = conf_base;
                    let mut allele1 = target_win.allele(MarkerIdx::new(target_m as u32), hap1);
                    let mut allele2 = target_win.allele(MarkerIdx::new(target_m as u32), hap2);
                    if let Some(missing) = target_missing {
                        if missing.allele(MarkerIdx::new(target_m as u32), hap1) == 255 {
                            allele1 = 255;
                        }
                        if missing.allele(MarkerIdx::new(target_m as u32), hap2) == 255 {
                            allele2 = 255;
                        }
                    }

                    let (mapped1, mapped2) = if let Some(mapping) = alignment
                        .allele_mappings
                        .get(target_m)
                        .and_then(|m| m.as_ref())
                    {
                        let mapped1 = if (allele1 as usize) < mapping.targ_to_ref.len() {
                            let r = mapping.targ_to_ref[allele1 as usize];
                            if r >= 0 { r as u8 } else { 255 }
                        } else {
                            255
                        };
                        let mapped2 = if (allele2 as usize) < mapping.targ_to_ref.len() {
                            let r = mapping.targ_to_ref[allele2 as usize];
                            if r >= 0 { r as u8 } else { 255 }
                        } else {
                            255
                        };
                        (mapped1, mapped2)
                    } else {
                        (allele1, allele2)
                    };

                    let is_diploid =
                        target_samples.is_diploid(SampleIdx::new(sample_idx as u32));
                    let has_hard = mapped1 != 255
                        && (mapped1 as usize) < n_alleles
                        && (!is_diploid
                            || (mapped2 != 255 && (mapped2 as usize) < n_alleles));
                    let input_phased = target_win
                        .phase_mask()
                        .and_then(|mask| mask.get(target_m).and_then(|row| row.get(sample_idx)).copied())
                        .map(|v| v != 0)
                        .unwrap_or(true);
                    let local_phase_conf_valid = phase_conf_valid && input_phased;

                    // If phase confidence is unavailable (unphased input), we still
                    // use hard genotype information but avoid committing to a phase:
                    // heterozygotes are represented as 0.5/0.5 per haplotype.

                    let pl = target_pl_matrix.sample_pl(MarkerIdx::new(target_m as u32), sample_idx);
                    let mapping = alignment
                        .allele_mappings
                        .get(target_m)
                        .and_then(|m| m.as_ref());

                    // If the input is unphased, avoid conditioning on a partner
                    // allele. Use unconditional allele probabilities from PL
                    // (if present) for both haplotypes.
                    if !local_phase_conf_valid {
                        if let Some(pl) = pl {
                            if !pl.is_empty() {
                                let mut pl_probs: Vec<f32> = Vec::new();
                                if allele_probs_uncond_from_pl(pl, None, &mut pl_probs)
                                    .is_some()
                                {
                                    if pl_probs.len() == n_alleles {
                                        aligned1 = pl_probs.clone();
                                        aligned2 = pl_probs;
                                        use1 = true;
                                        use2 = true;
                                    }
                                }
                            }
                        }
                    }

                    let compute_from_pl = |partner_allele: u8, out: &mut Vec<f32>, used: &mut bool| {
                        let mut pl_probs: Vec<f32> = Vec::new();
                        if let Some(pl) = pl {
                            if !pl.is_empty() {
                                let n_pl_alleles = infer_n_alleles_from_pl_len(pl.len()).unwrap_or(0);
                                if n_pl_alleles > 0 {
                                    let mut used_conditional = false;
                                    if local_phase_conf_valid
                                        && partner_allele != 255
                                        && (partner_allele as usize) < n_pl_alleles
                                    {
                                        let phase_conf = target_win
                                            .sample_phase_confidence_f32(
                                                MarkerIdx::new(target_m as u32),
                                                sample_idx,
                                            )
                                            .clamp(0.0, 1.0);
                                        let mut weights = vec![0.0f32; n_pl_alleles];
                                        let mut denom = 0.0f32;
                                        for i in 0..n_pl_alleles {
                                            if i != partner_allele as usize {
                                                denom += 1.0;
                                            }
                                        }
                                        weights[partner_allele as usize] = phase_conf;
                                        if denom > 0.0 {
                                            let scale = (1.0 - phase_conf) / denom;
                                            for i in 0..n_pl_alleles {
                                                if i != partner_allele as usize {
                                                    weights[i] = scale;
                                                }
                                            }
                                        }

                                        let mut cond_probs: Vec<f32> = Vec::new();
                                        pl_probs.resize(n_pl_alleles, 0.0);
                                        let mut weight_sum = 0.0f32;
                                        for b in 0..n_pl_alleles {
                                            let w = weights[b];
                                            if w <= 0.0 {
                                                continue;
                                            }
                                            if allele_probs_cond_from_pl(pl, b as u8, None, &mut cond_probs)
                                                .is_some()
                                            {
                                                for (a, &p) in cond_probs.iter().enumerate() {
                                                    if a < pl_probs.len() {
                                                        pl_probs[a] += w * p;
                                                    }
                                                }
                                                weight_sum += w;
                                            }
                                        }
                                        if weight_sum > 0.0 {
                                            normalize_probs(&mut pl_probs);
                                            if let Some(mapping) = mapping {
                                                let mut mapped = vec![0.0f32; n_alleles];
                                                for (t_idx, &p) in pl_probs.iter().enumerate() {
                                                    if t_idx < mapping.targ_to_ref.len() {
                                                        let r = mapping.targ_to_ref[t_idx];
                                                        if r >= 0 && (r as usize) < n_alleles {
                                                            mapped[r as usize] += p;
                                                        }
                                                    }
                                                }
                                                if normalize_probs(&mut mapped) {
                                                    *out = mapped;
                                                    *used = true;
                                                    used_conditional = true;
                                                }
                                            } else if pl_probs.len() == n_alleles {
                                                *out = pl_probs.clone();
                                                *used = true;
                                                used_conditional = true;
                                            }
                                        }
                                    }

                                    if !used_conditional
                                        && allele_probs_uncond_from_pl(pl, None, &mut pl_probs).is_some()
                                    {
                                        if let Some(mapping) = mapping {
                                            let mut mapped = vec![0.0f32; n_alleles];
                                            for (t_idx, &p) in pl_probs.iter().enumerate() {
                                                if t_idx < mapping.targ_to_ref.len() {
                                                    let r = mapping.targ_to_ref[t_idx];
                                                    if r >= 0 && (r as usize) < n_alleles {
                                                        mapped[r as usize] += p;
                                                    }
                                                }
                                            }
                                            if normalize_probs(&mut mapped) {
                                                *out = mapped;
                                                *used = true;
                                            }
                                        } else if pl_probs.len() == n_alleles {
                                            *out = pl_probs.clone();
                                            *used = true;
                                        }
                                    } else if !used_conditional {
                                        let uniform = 1.0 / n_pl_alleles as f32;
                                        let target_priors = vec![uniform; n_pl_alleles];

                                        let conf = conf_base.clamp(0.0, 1.0);
                                        let mut weights = vec![0.0f32; n_pl_alleles];
                                        if partner_allele != 255 && (partner_allele as usize) < n_pl_alleles {
                                            let partner_idx = partner_allele as usize;
                                            let mut denom = 0.0f32;
                                            for (i, &p) in target_priors.iter().enumerate() {
                                                if i != partner_idx {
                                                    denom += p;
                                                }
                                            }
                                            weights[partner_idx] = conf;
                                            if denom > 0.0 {
                                                let scale = (1.0 - conf) / denom;
                                                for i in 0..n_pl_alleles {
                                                    if i != partner_idx {
                                                        weights[i] = target_priors[i] * scale;
                                                    }
                                                }
                                            } else if n_pl_alleles > 1 {
                                                let uniform =
                                                    (1.0 - conf) / (n_pl_alleles as f32 - 1.0);
                                                for i in 0..n_pl_alleles {
                                                    if i != partner_idx {
                                                        weights[i] = uniform;
                                                    }
                                                }
                                            }
                                        } else {
                                            weights.copy_from_slice(&target_priors);
                                        }

                                        let mut cond_probs: Vec<f32> = Vec::new();
                                        pl_probs.resize(n_pl_alleles, 0.0);
                                        let mut weight_sum = 0.0f32;
                                        for b in 0..n_pl_alleles {
                                            let w = weights[b];
                                            if w <= 0.0 {
                                                continue;
                                            }
                                            if allele_probs_cond_from_pl(pl, b as u8, None, &mut cond_probs)
                                                .is_some()
                                            {
                                                for (a, &p) in cond_probs.iter().enumerate() {
                                                    if a < pl_probs.len() {
                                                        pl_probs[a] += w * p;
                                                    }
                                                }
                                                weight_sum += w;
                                            }
                                        }
                                        if weight_sum > 0.0 {
                                            normalize_probs(&mut pl_probs);
                                            if let Some(mapping) = mapping {
                                                let mut mapped = vec![0.0f32; n_alleles];
                                                for (t_idx, &p) in pl_probs.iter().enumerate() {
                                                    if t_idx < mapping.targ_to_ref.len() {
                                                        let r = mapping.targ_to_ref[t_idx];
                                                        if r >= 0 && (r as usize) < n_alleles {
                                                            mapped[r as usize] += p;
                                                        }
                                                    }
                                                }
                                                if normalize_probs(&mut mapped) {
                                                    *out = mapped;
                                                    *used = true;
                                                }
                                            } else if pl_probs.len() == n_alleles {
                                                *out = pl_probs.clone();
                                                *used = true;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    };

                    if !has_hard {
                        compute_from_pl(allele2, &mut aligned1, &mut use1);
                        compute_from_pl(allele1, &mut aligned2, &mut use2);
                    }

                    if !use1 && has_hard {
                        if pl.is_none() || pl.as_ref().map_or(true, |v| v.is_empty()) {
                            conf1 = conf1.min(1.0 - err_rate);
                        }
                        aligned1.resize(n_alleles, 0.0);
                        if is_diploid && mapped2 != 255 && mapped2 != mapped1 {
                            if local_phase_conf_valid {
                                let phase_conf = target_win
                                    .sample_phase_confidence_f32(
                                        MarkerIdx::new(target_m as u32),
                                        sample_idx,
                                    )
                                    .clamp(0.0, 1.0);
                                aligned1[mapped1 as usize] = phase_conf;
                                aligned1[mapped2 as usize] = 1.0 - phase_conf;
                            } else {
                                aligned1[mapped1 as usize] = 0.5;
                                aligned1[mapped2 as usize] = 0.5;
                            }
                        } else {
                            aligned1[mapped1 as usize] = conf1.clamp(0.0, 1.0);
                        }
                        use1 = true;
                    }

                    if !use2 && has_hard {
                        if pl.is_none() || pl.as_ref().map_or(true, |v| v.is_empty()) {
                            conf2 = conf2.min(1.0 - err_rate);
                        }
                        aligned2.resize(n_alleles, 0.0);
                        if is_diploid && mapped1 != 255 && mapped1 != mapped2 {
                            if local_phase_conf_valid {
                                let phase_conf = target_win
                                    .sample_phase_confidence_f32(
                                        MarkerIdx::new(target_m as u32),
                                        sample_idx,
                                    )
                                    .clamp(0.0, 1.0);
                                aligned2[mapped2 as usize] = phase_conf;
                                aligned2[mapped1 as usize] = 1.0 - phase_conf;
                            } else {
                                aligned2[mapped2 as usize] = 0.5;
                                aligned2[mapped1 as usize] = 0.5;
                            }
                        } else {
                            aligned2[mapped2 as usize] = conf2.clamp(0.0, 1.0);
                        }
                        use2 = true;
                    }

                    if sample_idx == 0 {
                        dbg_genotyped += 1;
                        if has_hard && use1 && use2 {
                            dbg_hard_used += 1;
                        }
                        if has_hard {
                            if let Some(pl_vals) = pl {
                                if !pl_vals.is_empty() {
                                    dbg_pl_present_with_hard += 1;
                                    let mut pl_probs: Vec<f32> = Vec::new();
                                    if allele_probs_uncond_from_pl(pl_vals, None, &mut pl_probs)
                                        .is_some()
                                        && is_uniform(&pl_probs)
                                    {
                                        dbg_pl_uniform_with_hard += 1;
                                    }
                                }
                            }
                        }
                    }
                }

                if !use1 {
                    aligned1.resize(n_alleles.max(1), 1.0);
                }
                if !use2 {
                    aligned2.resize(n_alleles.max(1), 1.0);
                }

                normalize_probs(&mut aligned1);
                normalize_probs(&mut aligned2);

                probs1.extend_from_slice(&aligned1);
                probs2.extend_from_slice(&aligned2);
                offsets1.push(probs1.len());
                offsets2.push(probs2.len());
            }
            if sample_idx == 0 && dbg_genotyped > 0 {
                eprintln!(
                    "    [debug] genotyped markers={} hard_used={} pl_present_with_hard={} pl_uniform_with_hard={}",
                    dbg_genotyped,
                    dbg_hard_used,
                    dbg_pl_present_with_hard,
                    dbg_pl_uniform_with_hard
                );
            }
            (
                TargetAlleleProbs::new(offsets1, probs1),
                TargetAlleleProbs::new(offsets2, probs2),
            )
        };

        let n_target_haps = n_target_samples * 2;
        let mut sm_posts_by_hap: Vec<Vec<AllelePosteriors>> =
            vec![Vec::with_capacity(output_markers); n_target_haps];
        // Information-weighted confusion: weight donor instability by -log(π(X))
        // This is optimal Li-Stephens scoring: rare allele switches are more informative
        let mut sm_low_conf_weighted: Vec<f32> = vec![0.0; n_target_haps];
        let mut sm_total_info: Vec<f32> = vec![0.0; n_target_haps];
        let mut sm_donor_counts: Vec<Vec<(RefHapId, u32)>> =
            vec![Vec::new(); n_target_haps];

        let mut pbwt = ReferencePbwt::new(plan.n_ref_haps);
        let mut beams: Vec<RankBeam> = vec![RankBeam::full(plan.n_ref_haps as u32); n_target_haps];
        let mut current_donor: Vec<u32> = vec![0u32; n_target_haps];
        let mut donor_candidates: Vec<u32> = Vec::with_capacity(SM_MATCH_DONORS);
        let mut ref_alleles: Vec<u8> = vec![0u8; plan.n_ref_haps];
        let mut query_alleles: Vec<PbwtQueryAllele> =
            vec![PbwtQueryAllele::wildcard(); n_target_haps];

        let push_donor_count = |counts: &mut Vec<(RefHapId, u32)>, hap: RefHapId| {
            if let Some(entry) = counts.iter_mut().find(|(h, _)| *h == hap) {
                entry.1 = entry.1.saturating_add(1);
            } else {
                counts.push((hap, 1));
            }
        };

        for ref_m in 0..n_ref_markers {
            let col = &ref_columns[ref_m];
            fill_ref_alleles(col, &mut ref_alleles);
            let n_alleles = ref_markers.marker(MarkerIdx::new(ref_m as u32)).n_alleles().max(1);

            if let Some(target_m) = alignment.ref_to_target.get(ref_m).and_then(|v| *v) {
                let target_idx = target_m.as_usize();
                let mapping = alignment
                    .allele_mappings
                    .get(target_idx)
                    .and_then(|m| m.as_ref());

                for sample_idx in 0..n_target_samples {
                    let h1 = HapIdx::new((sample_idx * 2) as u32);
                    let h2 = HapIdx::new((sample_idx * 2 + 1) as u32);
                    let mut a1 = target_win.allele(MarkerIdx::new(target_idx as u32), h1);
                    let mut a2 = target_win.allele(MarkerIdx::new(target_idx as u32), h2);
                    if let Some(missing) = target_missing {
                        if missing.allele(MarkerIdx::new(target_idx as u32), h1) == 255 {
                            a1 = 255;
                        }
                        if missing.allele(MarkerIdx::new(target_idx as u32), h2) == 255 {
                            a2 = 255;
                        }
                    }
                    let map_allele = |a: u8| -> u8 {
                        if a == 255 {
                            return 255;
                        }
                        if let Some(mapping) = mapping {
                            if (a as usize) < mapping.targ_to_ref.len() {
                                let r = mapping.targ_to_ref[a as usize];
                                if r >= 0 { r as u8 } else { 255 }
                            } else {
                                255
                            }
                        } else {
                            a
                        }
                    };
                    let a1 = map_allele(a1);
                    let a2 = map_allele(a2);
                    let idx1 = sample_idx * 2;
                    let idx2 = idx1 + 1;
                    query_alleles[idx1] = PbwtQueryAllele::allele(a1).unwrap_or_else(|| {
                        if a1 == 255 {
                            PbwtQueryAllele::wildcard()
                        } else {
                            PbwtQueryAllele::wildcard()
                        }
                    });
                    query_alleles[idx2] = PbwtQueryAllele::allele(a2).unwrap_or_else(|| {
                        if a2 == 255 {
                            PbwtQueryAllele::wildcard()
                        } else {
                            PbwtQueryAllele::wildcard()
                        }
                    });
                }
            } else {
                for qa in query_alleles.iter_mut() {
                    *qa = PbwtQueryAllele::wildcard();
                }
            }

            pbwt.advance_with_beams_query(
                &ref_alleles,
                n_alleles,
                ref_m,
                &query_alleles,
                &mut beams,
            );

            let store = ref_m >= output_start && ref_m < output_end;
            
            // Compute per-allele information weights: info = -ln(π(allele))
            // This is the optimal Li-Stephens LLR-based scoring
            let allele_info_weights: Vec<f32> = ref_allele_freqs
                .get(ref_m)
                .map(|freqs| {
                    freqs.iter().map(|&p| -(p.max(1e-9).ln())).collect()
                })
                .unwrap_or_default();
            
            // Precompute allele-frequency prior for missing targets at this marker.
            let freq_prior = ref_allele_freqs
                .get(ref_m)
                .and_then(|freqs| freqs.get(1).copied())
                .unwrap_or(0.5)
                .clamp(1e-6, 1.0 - 1e-6);

            for hap_idx in 0..n_target_haps {
                let beam = &beams[hap_idx];
                pbwt.select_donors_into(beam, SM_MATCH_DONORS, &mut donor_candidates);
                let mut donor = current_donor[hap_idx];
                let mut found = false;
                for &cand in &donor_candidates {
                    if cand == donor {
                        found = true;
                        break;
                    }
                }
                
                // Get the target allele for this haplotype to compute info weight
                let target_allele = query_alleles.get(hap_idx)
                    .and_then(|qa| qa.as_allele())
                    .unwrap_or(255);
                let info_weight = if target_allele == 255 {
                    0.0 // Missing data provides no information
                } else {
                    allele_info_weights.get(target_allele as usize).copied().unwrap_or(0.0)
                };
                
                // Track total information for this haplotype
                if target_allele != 255 {
                    sm_total_info[hap_idx] += info_weight;
                }
                
                // Information-weighted donor instability tracking
                if !found {
                    if let Some(&cand) = donor_candidates.first() {
                        donor = cand;
                        current_donor[hap_idx] = donor;
                    } else {
                        // No candidates at all - add full information weight as confusion
                        sm_low_conf_weighted[hap_idx] += info_weight;
                    }
                }
                if let Some(&cand) = donor_candidates.first() {
                    if cand != donor {
                        // Top candidate differs from current - add information-weighted confusion
                        // Common alleles (low info) contribute little; rare alleles (high info) contribute more
                        sm_low_conf_weighted[hap_idx] += info_weight;
                    }
                }
                push_donor_count(
                    &mut sm_donor_counts[hap_idx],
                    RefHapId::new(donor as u32),
                );
                if store {
                    let ap = if target_allele == 255 {
                        // Missing target: use donor ensemble (soft) to avoid hard-calling
                        if n_alleles <= 2 {
                            if donor_candidates.is_empty() {
                                AllelePosteriors::Biallelic(freq_prior)
                            } else {
                                let mut alt_sum = 0u32;
                                for &cand in &donor_candidates {
                                    let allele = col.get(HapIdx::new(cand));
                                    if allele == 1 {
                                        alt_sum += 1;
                                    }
                                }
                                let p_alt = alt_sum as f32 / donor_candidates.len() as f32;
                                AllelePosteriors::Biallelic(p_alt.clamp(1e-6, 1.0 - 1e-6))
                            }
                        } else {
                            let mut probs = vec![0.0f32; n_alleles.max(1)];
                            if donor_candidates.is_empty() {
                                let uniform = 1.0 / n_alleles.max(1) as f32;
                                for p in &mut probs {
                                    *p = uniform;
                                }
                            } else {
                                for &cand in &donor_candidates {
                                    let allele = col.get(HapIdx::new(cand));
                                    if (allele as usize) < n_alleles {
                                        probs[allele as usize] += 1.0;
                                    }
                                }
                                let denom = donor_candidates.len() as f32;
                                if denom > 0.0 {
                                    for p in &mut probs {
                                        *p /= denom;
                                    }
                                }
                            }
                            AllelePosteriors::Multiallelic(probs)
                        }
                    } else {
                        let allele = col.get(HapIdx::new(donor));
                        allele_to_posterior(n_alleles, allele)
                    };
                    sm_posts_by_hap[hap_idx].push(ap);
                }
            }
        }

        thread_local! {
            static LOCAL_WORKSPACE: std::cell::RefCell<Option<ImputeWorkspace>> =
                std::cell::RefCell::new(None);
        }

        let prior_marker_idx = if output_end > 0 {
            Some(output_end.saturating_sub(1))
        } else {
            None
        };
        let prior_global_idx = prior_marker_idx.map(|idx| idx + global_start);
        let prior_gen_pos = prior_marker_idx.and_then(|idx| gen_positions.get(idx).copied());
        struct ImputeResult {
            result: SampleImputationResult,
            priors: Option<(HaplotypePriors, HaplotypePriors)>,
        }

        let telemetry = self.telemetry.clone();
        let sample_results: Vec<ImputeResult> = sample_error_rates
            .par_iter_mut()
            .enumerate()
            .map(|(s, prior_error_rate)| {
                let h1_idx = HapIdx::new((s * 2) as u32);
                let h2_idx = HapIdx::new((s * 2 + 1) as u32);

                let priors_h1 = imp_overlap
                    .and_then(|o| o.hap_priors())
                    .and_then(|p| p.get(h1_idx.as_usize()));
                let priors_h2 = imp_overlap
                    .and_then(|o| o.hap_priors())
                    .and_then(|p| p.get(h2_idx.as_usize()));

                let (input_probs_h1, input_probs_h2) = build_input_probs_pair(h1_idx, h2_idx, s);
                // Information-weighted fallback decision: ratio of confused info to total info.
                // Missing targets provide no information, so treat missingness as low confidence.
                let total_info_h1 = sm_total_info[h1_idx.as_usize()].max(1e-9);
                let total_info_h2 = sm_total_info[h2_idx.as_usize()].max(1e-9);
                let conf_ratio_h1 = sm_low_conf_weighted[h1_idx.as_usize()] / total_info_h1;
                let conf_ratio_h2 = sm_low_conf_weighted[h2_idx.as_usize()] / total_info_h2;
                let min_info_nats = (plan.n_ref_haps as f32).ln() * 1.5;
                let insufficient_info_h1 = sm_total_info[h1_idx.as_usize()] < min_info_nats;
                let insufficient_info_h2 = sm_total_info[h2_idx.as_usize()] < min_info_nats;
                let donors_h1 = &sm_donor_counts[h1_idx.as_usize()];
                let donors_h2 = &sm_donor_counts[h2_idx.as_usize()];
                // SM_MATCH_LOW_CONF_FRAC now means: fraction of *information* that was confused
                let use_hmm_h1 = conf_ratio_h1 > SM_MATCH_LOW_CONF_FRAC
                    || insufficient_info_h1
                    || donors_h1.len() < SM_MATCH_MIN_DONORS;
                let use_hmm_h2 = conf_ratio_h2 > SM_MATCH_LOW_CONF_FRAC
                    || insufficient_info_h2
                    || donors_h2.len() < SM_MATCH_MIN_DONORS;

                let mut warned_no_priors = false;
                let mut warned_empty_map = false;
                let update_error_rate = |stats: &crate::model::impute_hmm::EmStats,
                                         prior_error_rate: &mut f32| {
                    if stats.informative_sites > 0.0 {
                        let ratio = (stats.expected_mismatches / stats.informative_sites)
                            .clamp(1e-6, 0.5) as f32;
                        *prior_error_rate =
                            (*prior_error_rate * 0.8 + ratio * 0.2).clamp(err_floor, 0.5);
                    }
                };

                let build_state_haps = |hap_idx: HapIdx,
                                        donors: &[(RefHapId, u32)]|
                 -> Vec<RefHapId> {
                    let mut out: Vec<RefHapId> = Vec::new();
                    let mut sorted = donors.to_vec();
                    sorted.sort_unstable_by(|a, b| b.1.cmp(&a.1));
                    for (hap, _) in sorted.iter().take(per_window_cap_local) {
                        out.push(*hap);
                    }
                    if let Some(core) = plan.core_states.get(hap_idx.as_usize()) {
                        for &hap in core {
                            out.push(hap);
                            if out.len() >= per_window_cap_local {
                                break;
                            }
                        }
                    }
                    out.sort_unstable_by_key(|h| h.as_u32());
                    out.dedup();
                    if out.is_empty() {
                        if let Some(full) = full_states.as_ref() {
                            out.extend_from_slice(full);
                        } else if let Some(states) = state_haps_by_hap.get(hap_idx.as_usize()) {
                            out.extend_from_slice(states);
                        }
                    }
                    if out.len() > per_window_cap_local {
                        out.truncate(per_window_cap_local);
                    }
                    out
                };

                let mut process_haplotype = |hap_idx: HapIdx,
                                             priors: Option<&HaplotypePriors>,
                                             input_probs: &TargetAlleleProbs,
                                             error_rate: f32,
                                             donors: &[(RefHapId, u32)]|
                 -> (Vec<AllelePosteriors>, HaplotypePriors, crate::model::impute_hmm::EmStats) {
                    let state_haps = build_state_haps(hap_idx, donors);
                    assert!(
                        !state_haps.is_empty(),
                        "State selection produced empty haplotype set"
                    );

                    let state_priors = priors.and_then(|p| {
                        if p.is_empty() {
                            if !warned_no_priors {
                                warn!(
                                    "Handoff priors missing for window {} (no markers or no posterior)",
                                    window_idx
                                );
                                warned_no_priors = true;
                            }
                            return None;
                        }
                        let mapper = prior_mappers_by_hap
                            .as_ref()
                            .and_then(|v| v.get(hap_idx.as_usize()))
                            .and_then(|m| m.as_ref());
                        let mapped = if let Some(mapper) = mapper {
                            mapper.map(p.probs()).into_vec()
                        } else {
                            let prev_states: Vec<RefHapId> =
                                p.ids().iter().map(|id| RefHapId::new(id.0)).collect();
                            let mapper = TransitionMatrix::build(&prev_states, &state_haps);
                            mapper.map(p.probs()).into_vec()
                        };
                        if mapped.iter().all(|v| !v.is_finite() || *v <= 0.0) && !warned_empty_map {
                            warn!(
                                "State handoff mapped to empty priors for window {} (state set mismatch)",
                                window_idx
                            );
                            warned_empty_map = true;
                        }
                        Some(mapped)
                    });

                    let (posteriors, state_post, stats) = LOCAL_WORKSPACE.with(|cell| {
                        let mut ws_opt = cell.borrow_mut();
                        if ws_opt.is_none() {
                            *ws_opt = Some(ImputeWorkspace::new(state_haps.len(), n_ref_markers));
                        }
                        let ws = ws_opt.as_mut().unwrap();
                        run_impute_hmm(
                            &state_haps,
                            ref_columns,
                            input_probs,
                            &p_recomb,
                            error_rate,
                            prior_marker_idx,
                            state_priors.as_deref(),
                            &ref_allele_freqs,
                            ws,
                        )
                    });

                    let mut next_priors = HaplotypePriors::empty();
                    if let Some(state_post) = state_post {
                        let pairs = state_posteriors_to_priors(&state_haps, &state_post, 0.0);
                        if !pairs.is_empty() {
                            let (ids, probs): (Vec<GlobalHapId>, Vec<f32>) = pairs
                                .into_iter()
                                .map(|(g, p)| (GlobalHapId(g.as_u32()), p))
                                .unzip();
                            next_priors = HaplotypePriors::new(ids, probs);
                        }
                    }

                    (posteriors, next_priors, stats)
                };

                let mut hap1_posts = sm_posts_by_hap[h1_idx.as_usize()].clone();
                let mut hap2_posts = sm_posts_by_hap[h2_idx.as_usize()].clone();
                let mut p1_out = HaplotypePriors::empty();
                let mut p2_out = HaplotypePriors::empty();

                if use_hmm_h1 {
                    let (posts, out, stats) = process_haplotype(
                        h1_idx,
                        priors_h1,
                        &input_probs_h1,
                        (*prior_error_rate).max(err_floor).clamp(1e-6, 0.5),
                        donors_h1,
                    );
                    hap1_posts = posts;
                    p1_out = out;
                    update_error_rate(&stats, prior_error_rate);
                } else {
                    let total: u32 = donors_h1.iter().map(|(_, c)| *c).sum();
                    if total > 0 {
                        let (ids, probs): (Vec<GlobalHapId>, Vec<f32>) = donors_h1
                            .iter()
                            .map(|(h, c)| (GlobalHapId(h.as_u32()), *c as f32 / total as f32))
                            .unzip();
                        p1_out = HaplotypePriors::new(ids, probs);
                    }
                }

                if use_hmm_h2 {
                    let (posts, out, stats) = process_haplotype(
                        h2_idx,
                        priors_h2,
                        &input_probs_h2,
                        (*prior_error_rate).max(err_floor).clamp(1e-6, 0.5),
                        donors_h2,
                    );
                    hap2_posts = posts;
                    p2_out = out;
                    update_error_rate(&stats, prior_error_rate);
                } else {
                    let total: u32 = donors_h2.iter().map(|(_, c)| *c).sum();
                    if total > 0 {
                        let (ids, probs): (Vec<GlobalHapId>, Vec<f32>) = donors_h2
                            .iter()
                            .map(|(h, c)| (GlobalHapId(h.as_u32()), *c as f32 / total as f32))
                            .unzip();
                        p2_out = HaplotypePriors::new(ids, probs);
                    }
                }

                if let Some(bb) = telemetry.as_ref() {
                    bb.add_samples(1);
                }

                ImputeResult {
                    result: SampleImputationResult {
                        sample_idx: s,
                        hap_alt_probs: None,
                        hap_posteriors: Some((hap1_posts, hap2_posts)),
                    },
                    priors: Some((p1_out, p2_out)),
                }
            })
            .collect();

        let mut all_results = Vec::with_capacity(n_target_samples);
        let mut next_priors_vec = vec![HaplotypePriors::empty(); n_target_samples * 2];

        for item in sample_results {
            let sample_idx = item.result.sample_idx;
            all_results.push(item.result);
            if let Some((p1, p2)) = item.priors {
                let base = sample_idx * 2;
                if base + 1 < next_priors_vec.len() {
                    next_priors_vec[base] = p1;
                    next_priors_vec[base + 1] = p2;
                }
            }
        }

        all_results.sort_by_key(|result| result.sample_idx);

        if let Some(bb) = &self.telemetry {
            let output_markers = output_end.saturating_sub(output_start);
            bb.set_stage(crate::utils::telemetry::Stage::WritingOutput);
            bb.set_consumer_stage(crate::utils::telemetry::Stage::WritingOutput);
            bb.set_total_markers(output_markers as u64);
            bb.set_markers_processed(0);
            bb.set_total_samples(target_win.n_samples() as u64);
            bb.set_samples_processed(0);
            bb.set_op(&format!(
                "Writing window {} ({} markers)",
                window_idx, output_markers
            ));
            bb.set_consumer_op(&format!(
                "Writing window {} ({} markers)",
                window_idx, output_markers
            ));
        }
        Ok(Some(ImputationWindowResults {
            all_results,
            ref_is_biallelic,
            handoff: Some(ImputationHandoff {
                priors: next_priors_vec,
                prior_global_idx,
                prior_gen_pos,
            }),
        }))
    }
    fn extract_imputed_overlap_streaming<TargetSpace, RefSpace>(
        &self,
        target_win: &GenotypeMatrix<Phased, TargetSpace>,
        ref_markers: &crate::data::marker::Markers<RefSpace>,
        alignment: &MarkerAlignment<TargetSpace, RefSpace>,
        output_end: usize,
    ) -> PhasedOverlap {
        let overlap_size = 1000.min(ref_markers.len().saturating_sub(output_end));
        let start = output_end;
        let end = output_end.saturating_add(overlap_size);
        let n_haps = target_win.n_haplotypes();
        let mut alleles = vec![255u8; overlap_size * n_haps];
        for h in 0..n_haps {
            for (local_m, ref_m) in (start..end).enumerate() {
                if let Some(target_m) =
                    alignment.target_marker(MarkerIdx::new(ref_m as u32))
                {
                    alleles[h * overlap_size + local_m] =
                        target_win.allele(target_m, HapIdx::new(h as u32));
                }
            }
        }
        PhasedOverlap::new(overlap_size, n_haps, alleles)
    }

    /// Write imputed window results to VCF
    #[allow(clippy::too_many_arguments)]
    fn write_imputed_window_streaming<TargetSpace: Sync, RefMarkerSpace: Sync>(
        &self,
        ref_markers: &crate::data::marker::Markers<RefMarkerSpace>,
        target_win: &GenotypeMatrix<Phased, TargetSpace>,
        target_pl: Option<&GenotypeMatrix<Phased, TargetSpace>>,
        alignment: &MarkerAlignment<TargetSpace, RefMarkerSpace>,
        writer: &mut VcfWriter,
        quality: &mut ImputationQuality,
        ref_is_biallelic: &[bool],
        output_start: usize,
        output_end: usize,
        markers_to_process_start: usize,
        all_results: &[SampleImputationResult],
        include_gp: bool,
        include_ap: bool,
        correct_errors: bool,
    ) -> Result<()> {
        let markers_range = output_start..output_end;
        let n_markers = markers_range.len();

        if n_markers == 0 || all_results.is_empty() {
            return Ok(());
        }

        let write_span = if self.config.profile {
            Some(
                info_span!(
                    "io_write_output",
                    markers = n_markers,
                    samples = target_win.n_samples()
                )
                .entered(),
            )
        } else {
            None
        };
        if let Some(span) = &write_span {
            tracing::trace!(id = ?span.id(), "Entered write output span");
        }

        let include_posteriors = include_gp || include_ap;
        let n_samples = target_win.n_samples();
        let mut result_by_sample: Vec<Option<&SampleImputationResult>> = vec![None; n_samples];
        for result in all_results {
            if result.sample_idx < n_samples {
                result_by_sample[result.sample_idx] = Some(result);
            }
        }

        let default_posteriors = |marker_idx: usize| -> (AllelePosteriors, AllelePosteriors) {
            let marker = ref_markers.marker(MarkerIdx::new(marker_idx as u32));
            let n_alleles = 1 + marker.alt_alleles.len();
            if n_alleles == 2 {
                (
                    AllelePosteriors::Biallelic(0.0),
                    AllelePosteriors::Biallelic(0.0),
                )
            } else {
                let zeros = vec![0.0f32; n_alleles];
                (
                    AllelePosteriors::Multiallelic(zeros.clone()),
                    AllelePosteriors::Multiallelic(zeros),
                )
            }
        };

        let normalize_probs = |probs: &mut [f32]| -> bool {
            let mut sum = 0.0f32;
            for p in probs.iter_mut() {
                if *p < 0.0 {
                    *p = 0.0;
                }
                sum += *p;
            }
            if sum > 0.0 {
                for p in probs.iter_mut() {
                    *p /= sum;
                }
                true
            } else {
                false
            }
        };

        let get_posteriors_for_writer = if include_posteriors {
            Some(|marker_idx: usize, sample_idx: usize| {
                let local_m = marker_idx.saturating_sub(output_start);
                if let Some(result) = result_by_sample.get(sample_idx).and_then(|r| *r) {
                    if let Some((p1, p2)) = result.hap_posteriors.as_ref() {
                        let post1 = p1
                            .get(local_m)
                            .cloned()
                            .unwrap_or_else(|| default_posteriors(marker_idx).0);
                        let post2 = p2
                            .get(local_m)
                            .cloned()
                            .unwrap_or_else(|| default_posteriors(marker_idx).1);
                        return (post1, post2);
                    }
                }
                default_posteriors(marker_idx)
            })
        } else {
            None
        };

        let samples = target_win.samples_arc();
        let target_pl = target_pl.unwrap_or(target_win);

        let get_genotyped_alleles = |marker_idx: usize, sample_idx: usize| -> Option<(u8, u8)> {
            let target_m = alignment.target_marker(MarkerIdx::new(marker_idx as u32))?;
            let h1 = HapIdx::new((sample_idx * 2) as u32);
            let h2 = HapIdx::new((sample_idx * 2 + 1) as u32);
            let raw_a1 = target_win.allele(target_m, h1);
            let raw_a2 = target_win.allele(target_m, h2);
            if raw_a1 == 255 || raw_a2 == 255 {
                return None;
            }
            let mapping = alignment
                .allele_mappings
                .get(target_m.as_usize())
                .and_then(|m| m.as_ref());
            let map_allele = |a: u8| -> u8 {
                if a == 255 {
                    return 255;
                }
                if let Some(m) = mapping {
                    if (a as usize) < m.targ_to_ref.len() {
                        let r = m.targ_to_ref[a as usize];
                        if r >= 0 { r as u8 } else { 255 }
                    } else {
                        255
                    }
                } else {
                    a
                }
            };
            let a1 = map_allele(raw_a1);
            let a2 = map_allele(raw_a2);
            if a1 == 255 || a2 == 255 {
                None
            } else {
                Some((a1, a2))
            }
        };

        let genotype_index = |a: usize, b: usize| -> usize {
            let (i, j) = if a <= b { (a, b) } else { (b, a) };
            j * (j + 1) / 2 + i
        };

        let best_gt_from_gp = |n_alleles: usize, gp: &[f32]| -> (u8, u8) {
            let mut best = (0u8, 0u8);
            let mut best_prob = -1.0f32;
            let mut idx = 0usize;
            for j in 0..n_alleles {
                for i in 0..=j {
                    let p = gp.get(idx).copied().unwrap_or(0.0);
                    if p > best_prob {
                        best_prob = p;
                        if i == j {
                            best = (i as u8, i as u8);
                        } else {
                            best = (i as u8, j as u8);
                        }
                    }
                    idx += 1;
                }
            }
            best
        };

        let dosage_from_gp = |n_alleles: usize, gp: &[f32]| -> f32 {
            let mut dosage = 0.0f32;
            let mut idx = 0usize;
            for j in 0..n_alleles {
                for i in 0..=j {
                    let p = gp.get(idx).copied().unwrap_or(0.0);
                    let alt_count = (i > 0) as u8 + (j > 0) as u8;
                    dosage += p * (alt_count as f32);
                    idx += 1;
                }
            }
            dosage
        };

        let error_rate = self.params.p_mismatch;
        let use_hard_call_fallback = !correct_errors;
        let get_genotype_posteriors = |marker_idx: usize, sample_idx: usize| -> Option<Vec<f32>> {
            let target_m = alignment.target_marker(MarkerIdx::new(marker_idx as u32))?;
            // If the target genotype is missing at this marker, defer to imputation
            // posteriors instead of GL/PL-derived genotype probabilities.
            if get_genotyped_alleles(marker_idx, sample_idx).is_none() {
                return None;
            }
            let pl_opt = target_pl.sample_pl(target_m, sample_idx);

            if let Some(pl) = pl_opt {
                if !pl.is_empty() {
                    let n_pl_alleles = infer_n_alleles_from_pl_len(pl.len())?;
                    if n_pl_alleles == 0 {
                        return None;
                    }
                    let n_ref_alleles = ref_markers
                        .marker(MarkerIdx::new(marker_idx as u32))
                        .n_alleles();
                    let mapping = alignment
                        .allele_mappings
                        .get(target_m.as_usize())
                        .and_then(|m| m.as_ref());
                    let mut target_gp: Vec<f32> = Vec::new();
                    let n = genotype_probs_from_pl(pl, None, &mut target_gp)?;
                    if n != n_pl_alleles {
                        return None;
                    }

                    let mut ref_gp = vec![0.0f32; n_ref_alleles * (n_ref_alleles + 1) / 2];
                    let mut idx = 0usize;
                    for j in 0..n_pl_alleles {
                        for i in 0..=j {
                            let p = target_gp.get(idx).copied().unwrap_or(0.0);
                            idx += 1;
                            let ri: i8 = if let Some(mapping) = mapping {
                                mapping.targ_to_ref.get(i).copied().unwrap_or(-1)
                            } else if i <= i8::MAX as usize {
                                i as i8
                            } else {
                                -1
                            };
                            let rj: i8 = if let Some(mapping) = mapping {
                                mapping.targ_to_ref.get(j).copied().unwrap_or(-1)
                            } else if j <= i8::MAX as usize {
                                j as i8
                            } else {
                                -1
                            };
                            if ri < 0 || rj < 0 {
                                continue;
                            }
                            let (ri, rj) = (ri as usize, rj as usize);
                            if ri >= n_ref_alleles || rj >= n_ref_alleles {
                                continue;
                            }
                            let ref_idx = genotype_index(ri, rj);
                            if ref_idx < ref_gp.len() {
                                ref_gp[ref_idx] += p;
                            }
                        }
                    }
                    if !normalize_probs(&mut ref_gp) {
                        return None;
                    }
                    return Some(ref_gp);
                }
            }
            // Soft fallback using hard GT with an error rate: avoids hard-calling
            // genotyped markers when PLs are missing or uninformative.
            let n_ref_alleles = ref_markers
                .marker(MarkerIdx::new(marker_idx as u32))
                .n_alleles();
            if let Some((a1, a2)) = get_genotyped_alleles(marker_idx, sample_idx) {
                let n_genotypes = n_ref_alleles * (n_ref_alleles + 1) / 2;
                if n_genotypes == 0 {
                    return None;
                }
                let mut gp = vec![0.0f32; n_genotypes];
                let idx = genotype_index(a1 as usize, a2 as usize);
                let err = if use_hard_call_fallback {
                    0.0
                } else {
                    error_rate.clamp(1e-6, 0.5)
                };
                let main = 1.0 - err;
                if idx < gp.len() {
                    gp[idx] = main;
                }
                let spill = if n_genotypes > 1 {
                    err / (n_genotypes as f32 - 1.0)
                } else {
                    0.0
                };
                for (i, v) in gp.iter_mut().enumerate() {
                    if i != idx {
                        *v = spill;
                    }
                }
                return Some(gp);
            }
            None
        };

        // Closure to get dosage: marker_idx is window-local ref marker index from VCF writer
        // Dosages array is indexed from 0 for markers starting at output_start
        let get_dosage = |marker_idx: usize, sample_idx: usize| -> f32 {
            let local_m = marker_idx.saturating_sub(output_start);
            let hard_call = get_genotyped_alleles(marker_idx, sample_idx);
            let n_alleles =
                ref_markers.marker(MarkerIdx::new(marker_idx as u32)).n_alleles().max(1);

            // If error correction is enabled, prioritize imputed dosages.
            // Otherwise, prefer hard calls from input when available.
            let use_hard_call = !correct_errors && hard_call.is_some();

            let dosage = if use_hard_call {
                let (a1, a2) = hard_call.unwrap();
                (a1 + a2) as f32
            } else if let Some(result) = result_by_sample.get(sample_idx).and_then(|r| *r) {
                if let Some((p1, p2)) = result.hap_posteriors.as_ref() {
                    let d1 = p1
                        .get(local_m)
                        .map(|p| match p {
                            AllelePosteriors::Biallelic(p_alt) => *p_alt,
                            AllelePosteriors::Multiallelic(probs) => probs
                                .iter()
                                .enumerate()
                                .map(|(i, p)| i as f32 * p)
                                .sum(),
                        })
                        .unwrap_or(0.0);
                    let d2 = p2
                        .get(local_m)
                        .map(|p| match p {
                            AllelePosteriors::Biallelic(p_alt) => *p_alt,
                            AllelePosteriors::Multiallelic(probs) => probs
                                .iter()
                                .enumerate()
                                .map(|(i, p)| i as f32 * p)
                                .sum(),
                        })
                        .unwrap_or(0.0);
                    d1 + d2
                } else if n_alleles <= 2 {
                    if let Some((p1, p2)) = result.hap_alt_probs.as_ref() {
                        let d1 = p1.get(local_m).copied().unwrap_or(0.0);
                        let d2 = p2.get(local_m).copied().unwrap_or(0.0);
                        d1 + d2
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            } else if !correct_errors {
                if let Some(gp) = get_genotype_posteriors(marker_idx, sample_idx) {
                    dosage_from_gp(n_alleles, &gp)
                } else if let Some((a1, a2)) = hard_call {
                    // Fallback to hard call if imputation result is missing
                    (a1 + a2) as f32
                } else {
                    0.0
                }
            } else if let Some((a1, a2)) = hard_call {
                // Fallback to hard call if imputation result is missing
                (a1 + a2) as f32
            } else {
                0.0
            };

            if samples.is_diploid(SampleIdx::new(sample_idx as u32)) {
                dosage
            } else {
                dosage * 0.5
            }
        };

        // Closure to get best genotype
        let get_best_gt = |marker_idx: usize, sample_idx: usize| -> (u8, u8) {
            let local_m = marker_idx.saturating_sub(output_start);
            let hard_call = get_genotyped_alleles(marker_idx, sample_idx);

            let use_hard_call = !correct_errors && hard_call.is_some();

            if use_hard_call {
                hard_call.unwrap()
            } else if let Some(result) = result_by_sample.get(sample_idx).and_then(|r| *r) {
                let n_alleles =
                    ref_markers.marker(MarkerIdx::new(marker_idx as u32)).n_alleles().max(1);
                if let Some((p1, p2)) = result.hap_posteriors.as_ref() {
                    if n_alleles <= 2 {
                        let p1_alt = p1.get(local_m).map(|p| p.prob(1)).unwrap_or(0.0);
                        let p2_alt = p2.get(local_m).map(|p| p.prob(1)).unwrap_or(0.0);
                        let gp00 = (1.0 - p1_alt) * (1.0 - p2_alt);
                        let gp01 =
                            p1_alt * (1.0 - p2_alt) + (1.0 - p1_alt) * p2_alt;
                        let gp11 = p1_alt * p2_alt;
                        if gp01 >= gp00 && gp01 >= gp11 {
                            let p10 = p1_alt * (1.0 - p2_alt);
                            let p01 = (1.0 - p1_alt) * p2_alt;
                            if p10 >= p01 {
                                (1, 0)
                            } else {
                                (0, 1)
                            }
                        } else if gp11 >= gp00 {
                            (1, 1)
                        } else {
                            (0, 0)
                        }
                    } else {
                        let mut best = (0u8, 0u8);
                        let mut best_prob = -1.0f32;
                        for i in 0..n_alleles {
                            for j in i..n_alleles {
                                let p_i1 = p1.get(local_m).map(|p| p.prob(i)).unwrap_or(0.0);
                                let p_i2 = p2.get(local_m).map(|p| p.prob(i)).unwrap_or(0.0);
                                let p_j1 = p1.get(local_m).map(|p| p.prob(j)).unwrap_or(0.0);
                                let p_j2 = p2.get(local_m).map(|p| p.prob(j)).unwrap_or(0.0);
                                let prob = if i == j {
                                    p_i1 * p_i2
                                } else {
                                    p_i1 * p_j2 + p_j1 * p_i2
                                };
                                if prob > best_prob {
                                    best_prob = prob;
                                    if i == j {
                                        best = (i as u8, i as u8);
                                    } else {
                                        let p_ij = p_i1 * p_j2;
                                        let p_ji = p_j1 * p_i2;
                                        if p_ij >= p_ji {
                                            best = (i as u8, j as u8);
                                        } else {
                                            best = (j as u8, i as u8);
                                        }
                                    }
                                }
                            }
                        }
                        best
                    }
                } else if n_alleles <= 2 {
                    if let Some((p1, p2)) = result.hap_alt_probs.as_ref() {
                        let p1_alt = p1.get(local_m).copied().unwrap_or(0.0);
                        let p2_alt = p2.get(local_m).copied().unwrap_or(0.0);
                        let gp00 = (1.0 - p1_alt) * (1.0 - p2_alt);
                        let gp01 = p1_alt * (1.0 - p2_alt) + (1.0 - p1_alt) * p2_alt;
                        let gp11 = p1_alt * p2_alt;
                        if gp01 >= gp00 && gp01 >= gp11 {
                            let p10 = p1_alt * (1.0 - p2_alt);
                            let p01 = (1.0 - p1_alt) * p2_alt;
                            if p10 >= p01 {
                                (1, 0)
                            } else {
                                (0, 1)
                            }
                        } else if gp11 >= gp00 {
                            (1, 1)
                        } else {
                            (0, 0)
                        }
                    } else {
                        (0, 0)
                    }
                } else {
                    (0, 0)
                }
            } else if !correct_errors {
                if let Some(gp) = get_genotype_posteriors(marker_idx, sample_idx) {
                    let n_alleles =
                        ref_markers.marker(MarkerIdx::new(marker_idx as u32)).n_alleles();
                    best_gt_from_gp(n_alleles, &gp)
                } else if let Some(gt) = hard_call {
                    gt
                } else {
                    (0, 0)
                }
            } else if let Some(gt) = hard_call {
                gt
            } else {
                (0, 0)
            }
        };

        let get_hap_probs = |marker_idx: usize, sample_idx: usize| -> (f32, f32) {
            let local_m = marker_idx.saturating_sub(output_start);
            if let Some(result) = result_by_sample.get(sample_idx).and_then(|r| *r) {
                if let Some((p1, p2)) = result.hap_posteriors.as_ref() {
                    let v1 = p1.get(local_m).map(|p| p.prob(1)).unwrap_or(0.0);
                    let v2 = p2.get(local_m).map(|p| p.prob(1)).unwrap_or(0.0);
                    return (v1, v2);
                }
                if let Some((p1, p2)) = result.hap_alt_probs.as_ref() {
                    let v1 = p1.get(local_m).copied().unwrap_or(0.0);
                    let v2 = p2.get(local_m).copied().unwrap_or(0.0);
                    return (v1, v2);
                }
            }
            (0.0, 0.0)
        };

        if include_posteriors {
            for marker_idx in markers_to_process_start..output_end {
                if marker_idx >= ref_is_biallelic.len() || !ref_is_biallelic[marker_idx] {
                    continue;
                }
                if let Some(stats) = quality.get_mut(marker_idx) {
                    for s in 0..n_samples {
                        let (v1, v2) = get_hap_probs(marker_idx, s);
                        let (v1, v2) = if !stats.is_imputed {
                            if let Some((a1, a2)) = get_genotyped_alleles(marker_idx, s) {
                                (a1 as f32, a2 as f32)
                            } else if let Some(gp) = get_genotype_posteriors(marker_idx, s) {
                                let n_alleles = ref_markers
                                    .marker(MarkerIdx::new(marker_idx as u32))
                                    .n_alleles();
                                let dosage = dosage_from_gp(n_alleles, &gp);
                                let p_alt = (dosage * 0.5).clamp(0.0, 1.0);
                                (p_alt, p_alt)
                            } else {
                                (v1, v2)
                            }
                        } else {
                            (v1, v2)
                        };
                        stats.add_sample_biallelic(v1, v2);
                    }
                }
            }
        } else {
            for marker_idx in markers_to_process_start..output_end {
                if marker_idx >= ref_is_biallelic.len() || !ref_is_biallelic[marker_idx] {
                    continue;
                }
                if let Some(stats) = quality.get_mut(marker_idx) {
                    for s in 0..n_samples {
                        let (v1, v2) = get_hap_probs(marker_idx, s);
                        let (v1, v2) = if !stats.is_imputed {
                            if let Some((a1, a2)) = get_genotyped_alleles(marker_idx, s) {
                                (a1 as f32, a2 as f32)
                            } else if let Some(gp) = get_genotype_posteriors(marker_idx, s) {
                                let n_alleles = ref_markers
                                    .marker(MarkerIdx::new(marker_idx as u32))
                                    .n_alleles();
                                let dosage = dosage_from_gp(n_alleles, &gp);
                                let p_alt = (dosage * 0.5).clamp(0.0, 1.0);
                                (p_alt, p_alt)
                            } else {
                                (v1, v2)
                            }
                        } else {
                            (v1, v2)
                        };
                        stats.add_sample_biallelic(v1, v2);
                    }
                }
            }
        }

        let get_genotype_posteriors_for_writer = if include_gp && !correct_errors {
            Some(|m, s| get_genotype_posteriors(m, s))
        } else {
            None
        };

        writer.write_imputed_streaming(
            ref_markers,
            get_dosage,
            get_best_gt,
            get_posteriors_for_writer,
            get_genotype_posteriors_for_writer,
            quality,
            output_start,
            output_end,
            include_gp,
            include_ap,
            self.telemetry.as_ref(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
use crate::data::ChromIdx;
    use crate::data::alignment::MarkerAlignment;
    use crate::data::haplotype::Samples;
    use crate::data::marker::{Allele, Marker, Markers, Nucleotide};
    use crate::data::storage::GenotypeColumn;
    use crate::data::storage::phase_state::Unphased;
    use crate::io::bref3::StreamingRefVcfReader;
    use crate::io::vcf::{ImputationQuality, VcfWriter};
    use crate::pipelines::ImputationPipeline;
    use std::io::{BufReader, Cursor};
    use tempfile::NamedTempFile;

    fn build_markers(chrom: ChromIdx, positions: &[u32]) -> Markers {
        let mut markers = Markers::<crate::data::AnyMarkerSpace>::new();
        markers.add_chrom("chr1");
        for (idx, &pos) in positions.iter().enumerate() {
            let marker = Marker::new(
                chrom,
                pos,
                Some(format!("m{idx}").into()),
                Allele::Base(Nucleotide::A),
                vec![Allele::Base(Nucleotide::C)],
            );
            markers.push(marker);
        }
        markers
    }

    fn build_unphased_matrix(markers: Markers, n_samples: usize) -> GenotypeMatrix<Unphased> {
        let samples = Arc::new(Samples::from_ids(
            (0..n_samples).map(|i| format!("s{i}")).collect(),
        ));
        let n_haps = n_samples * 2;
        let columns: Vec<GenotypeColumn> = (0..markers.len())
            .map(|_| {
                let bytes: Vec<u8> = vec![0u8; n_haps];
                GenotypeColumn::from_alleles(&bytes, 2)
            })
            .collect();
        GenotypeMatrix::new_unphased(markers, columns, samples)
    }

    #[test]
    fn test_sparse_target_should_not_truncate_reference_region() {
        let chrom = ChromIdx::new(0);
        let ref_positions: Vec<u32> = (0..3000).collect();
        let target_positions: Vec<u32> = vec![1500, 1501, 1502];

        let target_markers = build_markers(chrom, &target_positions);
        let target_gt = build_unphased_matrix(target_markers, 2);

        let mut vcf_data = String::new();
        vcf_data.push_str("##fileformat=VCFv4.2\n");
        vcf_data.push_str("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts0\ts1\n");
        for pos in &ref_positions {
            vcf_data.push_str(&format!(
                "chr1\t{}\t.\tA\tC\t.\tPASS\t.\tGT\t0|0\t0|0\n",
                pos
            ));
        }
        let reader = Box::new(BufReader::new(Cursor::new(vcf_data.into_bytes())));
        let ref_reader = StreamingRefVcfReader::from_reader(reader).expect("ref reader");
        let mut ref_reader = RefPanelReader::StreamingVcf(ref_reader);
        let config = StreamingConfig::default();
        let gen_maps = GeneticMaps::default();
        let ref_window = ref_reader
            .next_window(&config, &gen_maps, None)
            .expect("ref window load failed")
            .expect("no ref window found");

        // Desired behavior: sparse target data should not truncate the reference region.
        assert_eq!(target_gt.n_markers(), target_positions.len());
        assert_eq!(ref_window.global_start, 0);
        assert_eq!(ref_window.global_end, ref_positions.len());
    }

    #[test]
    fn test_write_imputed_window_ignores_short_ref_genotypes() {
        let chrom = ChromIdx::new(0);
        let ref_markers = build_markers(chrom, &[10, 20, 30]);
        let target_markers = build_markers(chrom, &[10]);

        let target_win = build_unphased_matrix(target_markers, 1).into_phased();

        let alignment = MarkerAlignment {
            ref_to_target: vec![None; ref_markers.len()],
            target_to_ref: vec![None; target_win.n_markers()],
            allele_mappings: vec![None; target_win.n_markers()],
        };

        let n_alleles_per_marker: Vec<usize> = (0..ref_markers.len())
            .map(|m| {
                let marker = ref_markers.marker(MarkerIdx::new(m as u32));
                1 + marker.alt_alleles.len()
            })
            .collect();
        let mut quality = ImputationQuality::new(&n_alleles_per_marker);

        let output_start = 0;
        let output_end = ref_markers.len();
        let all_results = vec![SampleImputationResult {
            sample_idx: 0,
            hap_alt_probs: Some((
                vec![0.0; output_end - output_start],
                vec![0.0; output_end - output_start],
            )),
            hap_posteriors: None,
        }];

        let tmp = NamedTempFile::new().expect("temp vcf");
        let mut writer = VcfWriter::create(tmp.path(), target_win.samples_arc()).expect("writer");

        let pipeline = ImputationPipeline::new(Config::default(), None);
        let ref_is_biallelic = vec![true; ref_markers.len()];
        let result = pipeline.write_imputed_window_streaming(
            &ref_markers,
            &target_win,
            None,
            &alignment,
            &mut writer,
            &mut quality,
            &ref_is_biallelic,
            output_start,
            output_end,
            output_start,
            &all_results,
            false,
            false,
            false,
        );
        assert!(result.is_ok());
    }
}
