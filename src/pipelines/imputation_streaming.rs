//! Streaming Imputation Pipeline
//!
//! Implements memory-efficient streaming imputation through overlapping windows.
//! Uses a producer-consumer model with MPSC channel to pipe phased matrices
//! directly to imputation in-memory.

use std::collections::HashMap;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use rayon::prelude::*;
use tracing::{info_span, instrument, warn};

use crate::Config;
use crate::data::alignment::MarkerAlignment;
use crate::data::genetic_map::GeneticMaps;
use crate::data::marker::{AnyMarkerSpace, RefWindowSpace};
use crate::data::storage::phase_state::{PhaseState, Phased};
use crate::data::storage::{GenotypeColumn, GenotypeMatrix};
use crate::data::{ChromIdx, HapIdx, MarkerIdx, SampleIdx};
use crate::error::ReagleError;
use crate::error::Result;
use crate::io::bref3::{RefPanelReader, RefWindow, TargetMarkerIndex, convert_ref_vcf_to_bref3};
use crate::io::prescan_cache::{
    PackedRefColumn, PrescanCacheReader, PrescanCacheWriter, create_temp_cache_path,
    pack_ref_columns,
};
use crate::io::streaming::{
    GlobalHapId, HaplotypePriors, PhasedOverlap, StreamingConfig, StreamingVcfReader,
};
use crate::io::vcf::{ImputationQuality, VcfWriter};
use crate::model::impute_hmm::{
    ImputeHmmContext, ImputeWorkspace, RefAlleleFreqs, TargetAlleleProbs, run_impute_hmm,
    state_posteriors_to_priors,
};
use crate::model::parameters::ModelParams;
use crate::model::pl_emission::{
    allele_probs_cond_from_pl, allele_probs_uncond_from_pl, genotype_probs_from_pl,
    infer_n_alleles_from_pl_len,
};
use crate::model::reference_pbwt::{PbwtQueryAllele, PbwtStrictAllele, RankBeam, ReferencePbwt};
use crate::model::transition_matrix::TransitionMatrix;
use crate::model::types::RefHapId;
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
const STATE_MIX_PRIOR_FRAC_NUM: usize = 20;
const STATE_MIX_WINDOW_FRAC_NUM: usize = 35;
const STATE_MIX_DONOR_FRAC_NUM: usize = 25;
const STATE_MIX_CORE_FRAC_NUM: usize = 20;
const STATE_MIX_FRAC_DEN: usize = 100;
const SMALL_PANEL_FULL_CAP_HAPS: usize = 512;
const FULL_PANEL_RAM_FRACTION: f64 = 0.9;
const SCAN_RAM_FRACTION: f64 = 0.10;
const TARGET_CACHE_RAM_FRACTION: f64 = 0.10;
const REF_PANEL_RAM_FRACTION: f64 = 0.75;
const EXACT_PRESCAN_MAX_OPS: u128 = 250_000_000;
const MIN_AVAIL_BYTES_FOR_PLANNING: u64 = 64 * 1024 * 1024;
// When memory detection fails, use a conservative fallback budget for prescan
// batching/caching to avoid pathological re-reads of the target VCF.
const PRESCAN_FALLBACK_AVAIL_BYTES: u64 = 256 * 1024 * 1024;

fn estimate_state_budget(available_bytes: u64, n_threads: usize, window_markers: usize) -> usize {
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
    core_states: Vec<Vec<RefHapId>>, // per target hap (derived)
    window_intervals: Vec<Vec<HapIntervals>>, // per target hap (sparse)
    abyss_mask: Vec<Vec<bool>>,      // per target hap
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
    let (
        core_min,
        core_avg,
        core_max,
        dynamic_min,
        dynamic_avg,
        dynamic_max,
        abyss_min,
        abyss_avg,
        abyss_max,
    ) = if plan.stats.haps == n_target_haps && plan.stats.haps > 0 {
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
    candidates.push(std::env::temp_dir().join(format!("reagle_ref_cache_{}.bref3", stem)));

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
                    std::fs::remove_file(&tmp_path).ok();
                    continue;
                }
                eprintln!("Converted reference VCF to BREF3 at {:?}", path);
                return Ok(path);
            }
            Err(err) => {
                std::fs::remove_file(&tmp_path).ok();
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

impl PrescanCacheGuard {
    fn touch(&self) {}
}

impl Drop for PrescanCacheGuard {
    fn drop(&mut self) {
        std::fs::remove_file(&self.path).ok();
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
            Self::InMemory {
                per_window_caps, ..
            } => per_window_caps.as_slice(),
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
    let align_markers = entry
        .alignment
        .ref_to_target
        .len()
        .saturating_add(entry.alignment.target_to_ref.len());
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
    if n_ref_haps <= SMALL_PANEL_FULL_CAP_HAPS {
        return n_ref_haps.max(1);
    }
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
                tx.send(Err(err.into())).ok();
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
                    tx.send(Ok(None)).ok();
                    break;
                }
                Err(err) => {
                    tx.send(Err(err.into())).ok();
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

    reader_handle
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
        let path = cache_path
            .ok_or_else(|| ReagleError::vcf("Prescan cache path missing after scan".to_string()))?;
        let writer = cache_writer.ok_or_else(|| {
            ReagleError::vcf("Prescan cache writer missing after scan".to_string())
        })?;
        writer.finish()?;
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
        return Err(ReagleError::vcf(
            "No target samples for pre-scan".to_string(),
        ));
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
    let prescan_avail = if avail == 0 {
        PRESCAN_FALLBACK_AVAIL_BYTES
    } else {
        avail
    };
    let n_ref_haps = ref_data.n_ref_haps();
    if n_ref_haps == 0 {
        return Err(ReagleError::vcf(
            "Reference window scanning found no haplotypes".to_string(),
        ));
    }
    plan.n_ref_haps = n_ref_haps;
    let window_handoff = ref_data.window_handoff().to_vec();
    let per_window_caps = ref_data.per_window_caps().to_vec();
    if avail == 0 {
        eprintln!(
            "Pre-scan: available memory unknown; using fallback={} MB for batching/cache",
            prescan_avail / (1024 * 1024)
        );
    }
    let batch_size = estimate_scan_batch_size(prescan_avail, n_ref_haps, n_target_haps);
    let mut batch_start = 0usize;
    let batches_total = (n_target_haps + batch_size - 1) / batch_size;
    let prescan_start = std::time::Instant::now();

    // Always run LMS prescan allocation for imputation.
    // Even when full panel fits in RAM, sparse-target scenarios benefit from
    // ancestry/locality-constrained state selection instead of uniform full-panel HMM.
    let can_full_panel = false;
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
            n_ref_haps, num_windows
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
        n_target_haps, n_ref_haps, batch_size
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
        (prescan_avail as f64 * TARGET_CACHE_RAM_FRACTION) as u64
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
                        target_bytes =
                            target_bytes.saturating_add(estimate_target_entry_bytes(&entry));
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
                        target_bytes =
                            target_bytes.saturating_add(estimate_target_entry_bytes(&entry));
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
                        let end_cm = gen_maps
                            .gen_pos(ref_window.markers.marker(MarkerIdx::new(0)).chrom, end_pos);
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
                            (entry.alignment.clone(), entry.phased_target.clone())
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
                        let target_window =
                            reader.load_window_for_region(&chrom_candidates, start_pos, end_pos)?;
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
                        let end_cm = gen_maps
                            .gen_pos(ref_window.markers.marker(MarkerIdx::new(0)).chrom, end_pos);
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
                            (entry.alignment.clone(), entry.phased_target.clone())
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
                        let target_window =
                            reader.load_window_for_region(&chrom_candidates, start_pos, end_pos)?;
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
        let use_plan_caps =
            !plan.per_window_caps.is_empty() && plan.per_window_caps.len() == num_windows;
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
                Ok((
                    hap_idx,
                    abyss,
                    intervals,
                    core,
                    core_len,
                    intervals_len,
                    abyss_count,
                ))
            })
            .collect::<Result<Vec<_>>>()?;

        for (hap_idx, abyss, intervals, core, core_len, intervals_len, abyss_count) in batch_results
        {
            plan.abyss_mask[hap_idx] = abyss;
            plan.window_intervals[hap_idx] = intervals;
            plan.core_states[hap_idx] = core;
            plan.stats.update(
                core_len,
                intervals_len.saturating_sub(core_len),
                abyss_count,
            );
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
            collect_target_positions(&self.config.target)?;
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

        let mut input_target_path = self.config.target.clone();
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
            phase_config.target = input_target_path.clone();
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
        let mut avail_bytes = crate::utils::memory::available_memory_bytes().unwrap_or(0);
        if avail_bytes < MIN_AVAIL_BYTES_FOR_PLANNING {
            // Treat unknown/low memory as "planning disabled" to avoid
            // tiny caps in CI/small test runs.
            avail_bytes = 0;
        }
        let min_states = 64usize;
        let mut raw_budget =
            estimate_state_budget(avail_bytes, n_threads, self.config.window_markers);
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
                guard.touch();
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
            eprintln!(
                "Target was unphased at input; using phased target for imputation (phase-uncertain emissions)."
            );
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
        let mut impute_recomb_intensity = (0.04 * self.config.ne / n_ref_pool as f32)
            .min(ModelParams::MAX_RECOMB_INTENSITY)
            .max(1e-6);
        if let Some(phased_rho) = phased_recomb_intensity {
            if phased_rho.is_finite() && phased_rho > 0.0 {
                impute_recomb_intensity =
                    phased_rho.clamp(1e-6, ModelParams::MAX_RECOMB_INTENSITY);
            }
        }
        self.params.recomb_intensity = impute_recomb_intensity;
        eprintln!(
            "Imputation recomb_intensity: {:.6} (source={})",
            self.params.recomb_intensity,
            if phased_recomb_intensity
                .map(|v| v.is_finite() && v > 0.0)
                .unwrap_or(false)
            {
                "phasing-estimated"
            } else {
                "config-ne"
            }
        );
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
        let mut sample_error_rates =
            vec![self.params.p_mismatch.clamp(1e-6, 0.5); n_target_samples];

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
                    let target_missing = target_window_pl.as_ref().map(|w| &w.genotypes);
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
                    let mut target_pos_to_indices: std::collections::HashMap<
                        (String, u32),
                        Vec<usize>,
                    > = std::collections::HashMap::new();
                    for t_idx in 0..target_window.genotypes.n_markers() {
                        let t_marker = target_window
                            .genotypes
                            .markers()
                            .marker(MarkerIdx::new(t_idx as u32));
                        let t_chrom = target_window
                            .genotypes
                            .markers()
                            .chrom_name(t_marker.chrom)
                            .unwrap_or("");
                        let key = (normalize_chrom_local(t_chrom).to_string(), t_marker.pos);
                        target_pos_to_indices.entry(key).or_default().push(t_idx);
                    }
                    let mut dbg_pos_present = 0usize;
                    let mut dbg_aligned_present = 0usize;
                    let mut dbg_pos_not_aligned = 0usize;
                    let mut dbg_pos_not_aligned_sites: Vec<(String, u64)> = Vec::new();
                    for (ref_m, target_idx) in alignment.ref_to_target.iter().enumerate() {
                        let ref_marker = ref_window.markers.marker(MarkerIdx::new(ref_m as u32));
                        let ref_chrom = ref_window
                            .markers
                            .chrom_name(ref_marker.chrom)
                            .unwrap_or("");
                        let has_biallelic_swap_or_match =
                            if target_idx.is_none() && ref_marker.n_alleles() == 2 {
                                let key =
                                    (normalize_chrom_local(ref_chrom).to_string(), ref_marker.pos);
                                if let Some(candidates) = target_pos_to_indices.get(&key) {
                                    let ref0 = ref_marker.ref_allele.to_string();
                                    let ref1 = ref_marker
                                        .alt_alleles
                                        .first()
                                        .map(|a| a.to_string())
                                        .unwrap_or_default();
                                    let n_matches = candidates
                                        .iter()
                                        .filter(|&&t_idx| {
                                            let t_marker = target_window
                                                .genotypes
                                                .markers()
                                                .marker(MarkerIdx::new(t_idx as u32));
                                            if t_marker.n_alleles() != 2 {
                                                return false;
                                            }
                                            let t0 = t_marker.ref_allele.to_string();
                                            let t1 = t_marker
                                                .alt_alleles
                                                .first()
                                                .map(|a| a.to_string())
                                                .unwrap_or_default();
                                            (ref0 == t0 && ref1 == t1) || (ref0 == t1 && ref1 == t0)
                                        })
                                        .count();
                                    n_matches > 0
                                } else {
                                    false
                                }
                            } else {
                                false
                            };
                        let is_present = target_idx.is_some() || has_biallelic_swap_or_match;
                        window_quality.set_imputed(ref_m, !is_present);
                        if is_present {
                            dbg_pos_present += 1;
                            if target_idx.is_some() {
                                dbg_aligned_present += 1;
                            } else {
                                dbg_pos_not_aligned += 1;
                                if dbg_pos_not_aligned_sites.len() < 5 {
                                    dbg_pos_not_aligned_sites
                                        .push((ref_chrom.to_string(), ref_marker.pos as u64));
                                }
                            }
                        }
                    }
                    if dbg_pos_not_aligned > 0 {
                        eprintln!(
                            "    [alignment] genotyped-by-position={} aligned={} pos_not_aligned={}",
                            dbg_pos_present, dbg_aligned_present, dbg_pos_not_aligned
                        );
                        if !dbg_pos_not_aligned_sites.is_empty() {
                            let sites: Vec<String> = dbg_pos_not_aligned_sites
                                .iter()
                                .map(|(c, p)| format!("{}:{}", c, p))
                                .collect();
                            eprintln!(
                                "    [alignment] pos_not_aligned_sites(first{})={}",
                                sites.len(),
                                sites.join(",")
                            );
                        }
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
                    let mut next_overlap_opt: Option<PhasedOverlap> = None;
                    if let Some(window_results) = window_results {
                        let ImputationWindowResults {
                            all_results,
                            ref_is_biallelic,
                            handoff,
                        } = window_results;
                        next_handoff = handoff;
                        next_overlap_opt = Some(self.extract_imputed_overlap_streaming(
                            &phased_target,
                            &alignment,
                            ref_window.output_start,
                            ref_window.output_end,
                            &all_results,
                        ));
                        // Drop heavy reference data before writing to reduce peak RSS.
                        // Drop reference genotypes/columns to free large buffers before write.
                        std::mem::take(&mut ref_window.ref_columns);
                        ref_window.ref_genotypes = None;
                        if let Some(bb) = &self.telemetry {
                            let output_markers = ref_window
                                .output_end
                                .saturating_sub(ref_window.output_start);
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
                            target_missing,
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

                        // Drop heavy reference data after writing to reduce peak RSS.
                        let _ = std::mem::take(&mut ref_window.ref_columns);
                        ref_window.ref_genotypes = None;

                        if let Some(bb) = &self.telemetry {
                            let output_markers = ref_window
                                .output_end
                                .saturating_sub(ref_window.output_start);
                            bb.set_markers_processed(output_markers as u64);
                            bb.set_samples_processed(phased_target.n_samples() as u64);
                            bb.set_stage(crate::utils::telemetry::Stage::Imputation);
                            bb.set_consumer_stage(crate::utils::telemetry::Stage::Imputation);
                        }
                    }

                    total_markers += ref_window
                        .output_end
                        .saturating_sub(ref_window.output_start);

                    let mut next_overlap = next_overlap_opt.unwrap_or_else(|| {
                        self.extract_imputed_overlap_streaming(
                            &phased_target,
                            &alignment,
                            ref_window.output_start,
                            ref_window.output_end,
                            &[],
                        )
                    });
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
                    let Some(mut ref_window) = ref_window else {
                        break;
                    };

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
                    let target_missing = target_window_pl.as_ref().map(|w| &w.genotypes);
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
                    let mut target_pos_to_indices: std::collections::HashMap<
                        (String, u32),
                        Vec<usize>,
                    > = std::collections::HashMap::new();
                    for t_idx in 0..target_window.genotypes.n_markers() {
                        let t_marker = target_window
                            .genotypes
                            .markers()
                            .marker(MarkerIdx::new(t_idx as u32));
                        let t_chrom = target_window
                            .genotypes
                            .markers()
                            .chrom_name(t_marker.chrom)
                            .unwrap_or("");
                        let key = (normalize_chrom_local(t_chrom).to_string(), t_marker.pos);
                        target_pos_to_indices.entry(key).or_default().push(t_idx);
                    }
                    for (ref_m, target_idx) in alignment.ref_to_target.iter().enumerate() {
                        let ref_marker = ref_window.markers.marker(MarkerIdx::new(ref_m as u32));
                        let ref_chrom = ref_window
                            .markers
                            .chrom_name(ref_marker.chrom)
                            .unwrap_or("");
                        let has_biallelic_swap_or_match =
                            if target_idx.is_none() && ref_marker.n_alleles() == 2 {
                                let key =
                                    (normalize_chrom_local(ref_chrom).to_string(), ref_marker.pos);
                                if let Some(candidates) = target_pos_to_indices.get(&key) {
                                    let ref0 = ref_marker.ref_allele.to_string();
                                    let ref1 = ref_marker
                                        .alt_alleles
                                        .first()
                                        .map(|a| a.to_string())
                                        .unwrap_or_default();
                                    let n_matches = candidates
                                        .iter()
                                        .filter(|&&t_idx| {
                                            let t_marker = target_window
                                                .genotypes
                                                .markers()
                                                .marker(MarkerIdx::new(t_idx as u32));
                                            if t_marker.n_alleles() != 2 {
                                                return false;
                                            }
                                            let t0 = t_marker.ref_allele.to_string();
                                            let t1 = t_marker
                                                .alt_alleles
                                                .first()
                                                .map(|a| a.to_string())
                                                .unwrap_or_default();
                                            (ref0 == t0 && ref1 == t1) || (ref0 == t1 && ref1 == t0)
                                        })
                                        .count();
                                    n_matches > 0
                                } else {
                                    false
                                }
                            } else {
                                false
                            };
                        let is_present = target_idx.is_some() || has_biallelic_swap_or_match;
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
                    let mut next_overlap_opt: Option<PhasedOverlap> = None;
                    if let Some(window_results) = window_results {
                        let ImputationWindowResults {
                            all_results,
                            ref_is_biallelic,
                            handoff,
                        } = window_results;
                        next_handoff = handoff;
                        next_overlap_opt = Some(self.extract_imputed_overlap_streaming(
                            &phased_target,
                            &alignment,
                            ref_window.output_start,
                            ref_window.output_end,
                            &all_results,
                        ));
                        // Drop heavy reference data before writing to reduce peak RSS.
                        // Drop reference genotypes/columns to free large buffers before write.
                        std::mem::take(&mut ref_window.ref_columns);
                        ref_window.ref_genotypes = None;
                        if let Some(bb) = &self.telemetry {
                            let output_markers = ref_window
                                .output_end
                                .saturating_sub(ref_window.output_start);
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
                            target_missing,
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

                        // Drop heavy reference data after writing to reduce peak RSS.
                        let _ = std::mem::take(&mut ref_window.ref_columns);
                        ref_window.ref_genotypes = None;

                        if let Some(bb) = &self.telemetry {
                            let output_markers = ref_window
                                .output_end
                                .saturating_sub(ref_window.output_start);
                            bb.set_markers_processed(output_markers as u64);
                            bb.set_samples_processed(phased_target.n_samples() as u64);
                            bb.set_stage(crate::utils::telemetry::Stage::Imputation);
                            bb.set_consumer_stage(crate::utils::telemetry::Stage::Imputation);
                        }
                    }

                    total_markers += ref_window
                        .output_end
                        .saturating_sub(ref_window.output_start);

                    let mut next_overlap = next_overlap_opt.unwrap_or_else(|| {
                        self.extract_imputed_overlap_streaming(
                            &phased_target,
                            &alignment,
                            ref_window.output_start,
                            ref_window.output_end,
                            &[],
                        )
                    });
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

        let ref_allele_freqs = RefAlleleFreqs::new(ref_columns);

        let gen_positions: Vec<f64> = {
            let chrom = ref_markers.marker(MarkerIdx::new(0)).chrom;
            if let Some(gen_map) = gen_maps.get(chrom) {
                crate::data::genetic_map::MarkerMap::create(ref_markers, gen_map)
                    .gen_positions()
                    .to_vec()
            } else {
                crate::data::genetic_map::MarkerMap::from_positions(ref_markers)
                    .gen_positions()
                    .to_vec()
            }
        };
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
            let max = p_recomb.iter().copied().fold(f32::NEG_INFINITY, f32::max);
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
            if n_ref_markers > 1 {
                // Handoff priors can become over-concentrated after overlap
                // compression. Apply a light first-step transition inflation to
                // recover single-window-equivalent boundary behavior.
                let base = p_recomb[1].max(1e-8);
                let handoff_scale = if window_idx <= 1 { 1.6 } else { 1.475 };
                p_recomb[0] = p_recomb[0].max(base) * handoff_scale;
                p_recomb[0] = p_recomb[0].min(0.25);
            }
        }

        let per_window_cap_local = plan
            .per_window_caps
            .get(window_idx)
            .copied()
            .unwrap_or(plan.per_window_cap)
            .max(1);
        // Even when full-panel memory is available, keep sample/window-specific
        // state sets from prescan/LMS. This preserves ancestry-local donor sets
        // and avoids diluting sparse-target inference with globally irrelevant
        // haplotypes.
        let full_states: Option<Vec<RefHapId>> = if plan.full_panel
            && plan.n_ref_haps > 32
            && plan.n_ref_haps <= 1024
        {
            Some(
                (0..plan.n_ref_haps)
                    .map(|h| RefHapId::new(h as u32))
                    .collect(),
            )
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
                let mut out: Vec<Option<TransitionMatrix>> =
                    Vec::with_capacity(n_target_samples * 2);
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
        let overlap_size = 1000.min(output_end);
        let overlap_start = output_end.saturating_sub(overlap_size);
        let build_input_probs_pair = |hap1: HapIdx,
                                      hap2: HapIdx,
                                      sample_idx: usize|
         -> (
            TargetAlleleProbs,
            TargetAlleleProbs,
            Option<usize>,
            Option<usize>,
        ) {
            let mut offsets1 = Vec::with_capacity(n_ref_markers + 1);
            let mut offsets2 = Vec::with_capacity(n_ref_markers + 1);
            let mut probs1: Vec<f32> = Vec::new();
            let mut probs2: Vec<f32> = Vec::new();
            offsets1.push(0);
            offsets2.push(0);
            let mut last_info1: Option<usize> = None;
            let mut last_info2: Option<usize> = None;
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
                    let raw_allele1 = target_win.allele(MarkerIdx::new(target_m as u32), hap1);
                    let raw_allele2 = target_win.allele(MarkerIdx::new(target_m as u32), hap2);
                    let mut allele1 = raw_allele1;
                    let mut allele2 = raw_allele2;
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

                    let is_diploid = target_samples.is_diploid(SampleIdx::new(sample_idx as u32));
                    let has_hard = mapped1 != 255
                        && (mapped1 as usize) < n_alleles
                        && (!is_diploid || (mapped2 != 255 && (mapped2 as usize) < n_alleles));
                    let input_phased = target_win
                        .phase_mask()
                        .and_then(|mask| {
                            mask.get(target_m)
                                .and_then(|row| row.get(sample_idx))
                                .copied()
                        })
                        .map(|v| v != 0)
                        .unwrap_or(true);
                    let local_phase_conf_valid = phase_conf_valid && input_phased;

                    // If phase confidence is unavailable (unphased input), we still
                    // use hard genotype information but avoid committing to a phase:
                    // heterozygotes are represented as 0.5/0.5 per haplotype.

                    let pl =
                        target_pl_matrix.sample_pl(MarkerIdx::new(target_m as u32), sample_idx);
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
                                if allele_probs_uncond_from_pl(pl, None, &mut pl_probs).is_some() {
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

                    let compute_from_pl =
                        |partner_allele: u8, out: &mut Vec<f32>, used: &mut bool| {
                            let mut pl_probs: Vec<f32> = Vec::new();
                            if let Some(pl) = pl {
                                if !pl.is_empty() {
                                    let n_pl_alleles =
                                        infer_n_alleles_from_pl_len(pl.len()).unwrap_or(0);
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
                                                if allele_probs_cond_from_pl(
                                                    pl,
                                                    b as u8,
                                                    None,
                                                    &mut cond_probs,
                                                )
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
                                            && allele_probs_uncond_from_pl(pl, None, &mut pl_probs)
                                                .is_some()
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
                                            if partner_allele != 255
                                                && (partner_allele as usize) < n_pl_alleles
                                            {
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
                                                if allele_probs_cond_from_pl(
                                                    pl,
                                                    b as u8,
                                                    None,
                                                    &mut cond_probs,
                                                )
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
                            // GT-only input (no PL/GL): enforce a high-confidence floor for
                            // observed typed alleles so sparse chips still anchor the HMM.
                            conf1 = conf1.max(1.0 - err_rate);
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
                            // GT-only input (no PL/GL): enforce a high-confidence floor for
                            // observed typed alleles so sparse chips still anchor the HMM.
                            conf2 = conf2.max(1.0 - err_rate);
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
                }

                if !use1 {
                    aligned1.resize(n_alleles.max(1), 1.0);
                }
                if !use2 {
                    aligned2.resize(n_alleles.max(1), 1.0);
                }

                normalize_probs(&mut aligned1);
                normalize_probs(&mut aligned2);
                if !is_uniform(&aligned1) && ref_m >= overlap_start {
                    last_info1 = Some(ref_m);
                }
                if !is_uniform(&aligned2) && ref_m >= overlap_start {
                    last_info2 = Some(ref_m);
                }

                probs1.extend_from_slice(&aligned1);
                probs2.extend_from_slice(&aligned2);
                offsets1.push(probs1.len());
                offsets2.push(probs2.len());
            }
            (
                TargetAlleleProbs::new(offsets1, probs1),
                TargetAlleleProbs::new(offsets2, probs2),
                last_info1,
                last_info2,
            )
        };

        let n_target_haps = n_target_samples * 2;
        let min_info_nats = (plan.n_ref_haps as f32).ln() * 1.5;
        // Information-weighted confusion: normalized entropy of the PBWT match set
        // scaled by the emission-model LLR to keep confidence in probabilistic units.
        let mut sm_low_conf_weighted: Vec<f32> = vec![0.0; n_target_haps];
        let mut sm_total_info: Vec<f32> = vec![0.0; n_target_haps];
        let mut sm_donor_counts: Vec<HashMap<RefHapId, u32>> = vec![HashMap::new(); n_target_haps];
        let sm_needed: Vec<AtomicBool> =
            (0..n_target_haps).map(|_| AtomicBool::new(false)).collect();

        {
            let mut pbwt = ReferencePbwt::new(plan.n_ref_haps);
            let mut ref_alleles: Vec<u8> = vec![0u8; plan.n_ref_haps];
            let batch_size = 4096usize;
            let mut batches: Vec<(
                Vec<usize>,
                Vec<RankBeam>,
                Vec<PbwtQueryAllele>,
                Vec<u32>,
                Vec<(u32, u32, u32)>,
            )> = Vec::new();
            let mut start = 0usize;
            while start < n_target_haps {
                let end = (start + batch_size).min(n_target_haps);
                let haps: Vec<usize> = (start..end).collect();
                let beams = vec![RankBeam::full(plan.n_ref_haps as u32); haps.len()];
                let query_alleles = vec![PbwtQueryAllele::wildcard(); haps.len()];
                let current_donor = vec![0u32; haps.len()];
                let scratch = Vec::new();
                batches.push((haps, beams, query_alleles, current_donor, scratch));
                start = end;
            }

            let push_donor_count = |counts: &mut HashMap<RefHapId, u32>, hap: RefHapId| {
                let entry = counts.entry(hap).or_insert(0);
                *entry = entry.saturating_add(1);
            };

            for ref_m in 0..n_ref_markers {
                let col = &ref_columns[ref_m];
                fill_ref_alleles(col, &mut ref_alleles);
                let n_alleles = ref_markers
                    .marker(MarkerIdx::new(ref_m as u32))
                    .n_alleles()
                    .max(1);

                pbwt.prepare_step(&ref_alleles, n_alleles);
                for (haps, beams, query_alleles, _, scratch) in batches.iter_mut() {
                    if let Some(target_m) = alignment.ref_to_target.get(ref_m).and_then(|v| *v) {
                        let target_idx = target_m.as_usize();
                        let mapping = alignment
                            .allele_mappings
                            .get(target_idx)
                            .and_then(|m| m.as_ref());

                        for (i, &hap_idx) in haps.iter().enumerate() {
                            let sample_idx = hap_idx / 2;
                            let local = hap_idx % 2;
                            let h = HapIdx::new((sample_idx * 2 + local) as u32);
                            let mut a = target_win.allele(MarkerIdx::new(target_idx as u32), h);
                            if let Some(missing) = target_missing {
                               if missing.allele(MarkerIdx::new(target_idx as u32), h) == 255 {
                                   a = 255;
                               }
                           }
                            let a = if a == 255 {
                                255
                            } else if let Some(mapping) = mapping {
                                if (a as usize) < mapping.targ_to_ref.len() {
                                    let r = mapping.targ_to_ref[a as usize];
                                    if r >= 0 { r as u8 } else { 255 }
                                } else {
                                    255
                                }
                            } else {
                                a
                            };
                            query_alleles[i] = PbwtQueryAllele::allele(a)
                                .unwrap_or_else(PbwtQueryAllele::wildcard);
                        }
                    } else {
                        for qa in query_alleles.iter_mut() {
                            *qa = PbwtQueryAllele::wildcard();
                        }
                    }

                    pbwt.update_beams_with_scratch_query(beams, query_alleles, n_alleles, scratch);
                }
                pbwt.finalize_step(&ref_alleles, n_alleles, ref_m);

                // Donor/confusion evidence should be collected where the target
                // is informative in this window, not only at the tail overlap.
                // Overlap markers are still included to preserve handoff behavior.
                let aligned_here = alignment.ref_to_target.get(ref_m).and_then(|v| *v).is_some();
                let in_overlap = ref_m >= overlap_start && ref_m < output_end;
                let store = aligned_here || in_overlap;

                // Information weight in natural log space for one informative allele observation.
                let theta = self.params.p_mismatch.max(1e-9).min(1.0 - 1e-9) as f32;
                let info_llr = ((1.0 - theta) / theta).ln();

                if store {
                    for (haps, beams, query_alleles, current_donor, _) in batches.iter_mut() {
                        let mut donor_candidates: Vec<u32> = Vec::with_capacity(SM_MATCH_DONORS);
                        for (i, &hap_idx) in haps.iter().enumerate() {
                            let beam = &beams[i];
                            donor_candidates.clear();
                            pbwt.select_donors_into(beam, SM_MATCH_DONORS, &mut donor_candidates);
                            let donor = donor_candidates
                                .first()
                                .copied()
                                .unwrap_or(current_donor[i]);
                            current_donor[i] = donor;

                            let target_allele = query_alleles
                                .get(i)
                                .and_then(|qa| qa.as_allele())
                                .unwrap_or(255);
                            let info_weight = if target_allele == 255
                                || (target_allele as usize) >= n_alleles
                            {
                                0.0
                            } else {
                                info_llr
                            };
                            if target_allele != 255 {
                                sm_total_info[hap_idx] += info_weight;
                            }

                            if info_weight > 0.0 {
                                let mut match_count: u32 = 0;
                                for &(l, r) in beam.intervals() {
                                    match_count = match_count.saturating_add(r.saturating_sub(l));
                                }
                                let n_matches = (match_count.max(1) as f32).ln();
                                let max_entropy = (plan.n_ref_haps as f32).ln().max(1e-6);
                                let normalized_entropy = (n_matches / max_entropy).clamp(0.0, 1.0);
                                sm_low_conf_weighted[hap_idx] += info_weight * normalized_entropy;
                            }
                            if donor_candidates.is_empty() {
                                push_donor_count(
                                    &mut sm_donor_counts[hap_idx],
                                    RefHapId::new(donor as u32),
                                );
                            } else {
                                for &cand in &donor_candidates {
                                    push_donor_count(
                                        &mut sm_donor_counts[hap_idx],
                                        RefHapId::new(cand as u32),
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        // Diagnostics: donor set size distribution across haplotypes.
        if n_target_haps > 0 {
            let mut min_donors = usize::MAX;
            let mut max_donors = 0usize;
            let mut sum_donors = 0usize;
            for counts in &sm_donor_counts {
                let len = counts.len();
                if len < min_donors {
                    min_donors = len;
                }
                if len > max_donors {
                    max_donors = len;
                }
                sum_donors += len;
            }
            let avg_donors = sum_donors as f64 / n_target_haps as f64;
            eprintln!(
                "    [debug donors] hap_donor_counts min={} avg={:.2} max={}",
                min_donors, avg_donors, max_donors
            );
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
        struct ImputeResult {
            result: SampleImputationResult,
            priors: Option<(HaplotypePriors, HaplotypePriors)>,
            last_info_idx: Option<usize>,
        }

        let telemetry = self.telemetry.clone();
        use std::sync::atomic::{AtomicUsize, Ordering};
        let dbg_use_hmm = AtomicUsize::new(0);
        let dbg_no_hmm = AtomicUsize::new(0);
        let dbg_no_info = AtomicUsize::new(0);
        let dbg_insufficient = AtomicUsize::new(0);
        let dbg_low_conf = AtomicUsize::new(0);
        let dbg_few_donors = AtomicUsize::new(0);
        let dbg_has_priors = AtomicUsize::new(0);
        let dbg_fallback_ref_freq = AtomicUsize::new(0);
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

                let (input_probs_h1, input_probs_h2, last_info_h1, last_info_h2) =
                    build_input_probs_pair(h1_idx, h2_idx, s);
                let handoff_capture_idx_h1 = prior_marker_idx;
                let handoff_capture_idx_h2 = prior_marker_idx;
                // Information-weighted fallback decision: ratio of confused info to total info.
                // Missing targets provide no information, so treat missingness as low confidence.
        let total_info_h1 = sm_total_info[h1_idx.as_usize()].max(1e-9);
        let total_info_h2 = sm_total_info[h2_idx.as_usize()].max(1e-9);
        let conf_ratio_h1 = sm_low_conf_weighted[h1_idx.as_usize()] / total_info_h1;
        let conf_ratio_h2 = sm_low_conf_weighted[h2_idx.as_usize()] / total_info_h2;
        let insufficient_info_h1 = sm_total_info[h1_idx.as_usize()] < min_info_nats;
        let insufficient_info_h2 = sm_total_info[h2_idx.as_usize()] < min_info_nats;
        let no_info_h1 = sm_total_info[h1_idx.as_usize()] <= 0.0;
        let no_info_h2 = sm_total_info[h2_idx.as_usize()] <= 0.0;
                let has_priors_h1 = priors_h1.map(|p| !p.is_empty()).unwrap_or(false);
                let has_priors_h2 = priors_h2.map(|p| !p.is_empty()).unwrap_or(false);
                let mut donors_h1: Vec<(RefHapId, u32)> = sm_donor_counts[h1_idx.as_usize()]
                    .iter()
                    .map(|(h, c)| (*h, *c))
                    .collect();
                let mut donors_h2: Vec<(RefHapId, u32)> = sm_donor_counts[h2_idx.as_usize()]
                    .iter()
                    .map(|(h, c)| (*h, *c))
                    .collect();
                donors_h1.sort_unstable_by(|a, b| b.1.cmp(&a.1));
                donors_h2.sort_unstable_by(|a, b| b.1.cmp(&a.1));
                let tiny_panel = plan.n_ref_haps <= 32;
                let use_hmm_h1 = if tiny_panel {
                    true
                } else if has_priors_h1 || no_info_h1 {
                    true
                } else {
                    conf_ratio_h1 > SM_MATCH_LOW_CONF_FRAC
                        || insufficient_info_h1
                        || donors_h1.len() < SM_MATCH_MIN_DONORS
                };
                let use_hmm_h2 = if tiny_panel {
                    true
                } else if has_priors_h2 || no_info_h2 {
                    true
                } else {
                    conf_ratio_h2 > SM_MATCH_LOW_CONF_FRAC
                        || insufficient_info_h2
                        || donors_h2.len() < SM_MATCH_MIN_DONORS
                };

                let track_hmm = |use_hmm: bool,
                                 no_info: bool,
                                 insufficient: bool,
                                 conf_ratio: f32,
                                 donors_len: usize,
                                 has_priors: bool| {
                    if use_hmm {
                        dbg_use_hmm.fetch_add(1, Ordering::Relaxed);
                    } else {
                        dbg_no_hmm.fetch_add(1, Ordering::Relaxed);
                    }
                    if has_priors {
                        dbg_has_priors.fetch_add(1, Ordering::Relaxed);
                    }
                    if no_info {
                        dbg_no_info.fetch_add(1, Ordering::Relaxed);
                    }
                    if insufficient {
                        dbg_insufficient.fetch_add(1, Ordering::Relaxed);
                    }
                    if conf_ratio > SM_MATCH_LOW_CONF_FRAC {
                        dbg_low_conf.fetch_add(1, Ordering::Relaxed);
                    }
                    if donors_len < SM_MATCH_MIN_DONORS {
                        dbg_few_donors.fetch_add(1, Ordering::Relaxed);
                    }
                };

                track_hmm(
                    use_hmm_h1,
                    no_info_h1,
                    insufficient_info_h1,
                    conf_ratio_h1,
                    donors_h1.len(),
                    has_priors_h1,
                );
                track_hmm(
                    use_hmm_h2,
                    no_info_h2,
                    insufficient_info_h2,
                    conf_ratio_h2,
                    donors_h2.len(),
                    has_priors_h2,
                );

                let mut warned_no_priors = false;
                let mut warned_empty_map = false;
                let posts_from_priors =
                    |priors: &HaplotypePriors| -> Result<Vec<AllelePosteriors>> {
                    let mut out: Vec<AllelePosteriors> =
                        Vec::with_capacity(output_end.saturating_sub(output_start));
                    for ref_m in output_start..output_end {
                        let n_alleles = ref_markers
                            .marker(MarkerIdx::new(ref_m as u32))
                            .n_alleles()
                            .max(1);
                        let mut probs = vec![0.0f32; n_alleles];
                        for (id, p) in priors.ids().iter().zip(priors.probs().iter()) {
                            let hap = HapIdx::new(id.0);
                            let allele = ref_columns
                                .get(ref_m)
                                .map(|c| c.get(hap))
                                .unwrap_or(255);
                            if allele == 255 {
                                continue;
                            }
                            let idx = allele as usize;
                            if idx < probs.len() {
                                probs[idx] += *p;
                            }
                        }
                        let sum: f32 = probs.iter().sum();
                        if sum <= 0.0 {
                            return Err(ReagleError::vcf(format!(
                                "Subset prior collapsed while building allele posteriors: window={} sample={} marker={} source=handoff_priors",
                                window_idx, s, ref_m
                            )));
                        }
                        if sum > 0.0 {
                            let inv = 1.0 / sum;
                            for v in probs.iter_mut() {
                                *v *= inv;
                            }
                        }
                        if n_alleles == 2 {
                            out.push(AllelePosteriors::Biallelic(
                                probs.get(1).copied().unwrap_or(0.0),
                            ));
                        } else {
                            out.push(AllelePosteriors::Multiallelic(probs));
                        }
                    }
                    Ok(out)
                };
                let posts_from_donors =
                    |donors: &[(RefHapId, u32)]| -> Result<Vec<AllelePosteriors>> {
                    let mut out: Vec<AllelePosteriors> =
                        Vec::with_capacity(output_end.saturating_sub(output_start));
                    let total: u32 = donors.iter().map(|(_, c)| *c).sum();
                    if total == 0 {
                        return Err(ReagleError::vcf(format!(
                            "Empty donor subset for posterior construction: window={} sample={}",
                            window_idx, s
                        )));
                    }
                    let inv_total = 1.0f32 / total as f32;
                    for ref_m in output_start..output_end {
                        let n_alleles = ref_markers
                            .marker(MarkerIdx::new(ref_m as u32))
                            .n_alleles()
                            .max(1);
                        let mut probs = vec![0.0f32; n_alleles];
                        for (hap, c) in donors.iter() {
                            let allele = ref_columns
                                .get(ref_m)
                                .map(|col| col.get(HapIdx::new(hap.as_u32())))
                                .unwrap_or(255);
                            if allele == 255 {
                                continue;
                            }
                            let idx = allele as usize;
                            if idx < probs.len() {
                                probs[idx] += *c as f32 * inv_total;
                            }
                        }
                        let sum: f32 = probs.iter().sum();
                        if sum <= 0.0 {
                            return Err(ReagleError::vcf(format!(
                                "Donor-based posterior collapsed at marker: window={} sample={} marker={} donors={}",
                                window_idx,
                                s,
                                ref_m,
                                donors.len()
                            )));
                        }
                        let inv = 1.0 / sum;
                        for v in probs.iter_mut() {
                            *v *= inv;
                        }
                        if n_alleles == 2 {
                            out.push(AllelePosteriors::Biallelic(
                                probs.get(1).copied().unwrap_or(0.0),
                            ));
                        } else {
                            out.push(AllelePosteriors::Multiallelic(probs));
                        }
                    }
                    Ok(out)
                };
                let blend_posts = |base: &mut [AllelePosteriors],
                                   donor: &[AllelePosteriors],
                                   w: f32| {
                    if w <= 0.0 {
                        return;
                    }
                    let ww = w.clamp(0.0, 1.0);
                    for (b, d) in base.iter_mut().zip(donor.iter()) {
                        match (b, d) {
                            (AllelePosteriors::Biallelic(pb), AllelePosteriors::Biallelic(pd)) => {
                                *pb = ((1.0 - ww) * *pb + ww * *pd).clamp(0.0, 1.0);
                            }
                            (
                                AllelePosteriors::Multiallelic(pb),
                                AllelePosteriors::Multiallelic(pd),
                            ) => {
                                if pb.len() == pd.len() && !pb.is_empty() {
                                    let mut sum = 0.0f32;
                                    for i in 0..pb.len() {
                                        pb[i] = (1.0 - ww) * pb[i] + ww * pd[i];
                                        if pb[i] < 0.0 {
                                            pb[i] = 0.0;
                                        }
                                        sum += pb[i];
                                    }
                                    if sum > 0.0 {
                                        let inv = 1.0 / sum;
                                        for v in pb.iter_mut() {
                                            *v *= inv;
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                };
                let temper_posts = |base: &mut [AllelePosteriors], tau: f32| {
                    // Temperature smoothing in weak-signal regimes to reduce
                    // overconfident posteriors before donor fusion.
                    let tau = tau.max(1.0);
                    if tau <= 1.000_001 {
                        return;
                    }
                    let inv_tau = 1.0 / tau;
                    for b in base.iter_mut() {
                        match b {
                            AllelePosteriors::Biallelic(p_alt) => {
                                let p = (*p_alt).clamp(1e-9, 1.0 - 1e-9);
                                let logit = (p / (1.0 - p)).ln();
                                let scaled = logit * inv_tau;
                                *p_alt = 1.0 / (1.0 + (-scaled).exp());
                            }
                            AllelePosteriors::Multiallelic(probs) => {
                                if probs.is_empty() {
                                    continue;
                                }
                                let mut sum = 0.0f32;
                                for v in probs.iter_mut() {
                                    let p = (*v).max(1e-12);
                                    *v = p.powf(inv_tau);
                                    sum += *v;
                                }
                                if sum > 0.0 {
                                    let inv = 1.0 / sum;
                                    for v in probs.iter_mut() {
                                        *v *= inv;
                                    }
                                }
                            }
                        }
                    }
                };

                let build_state_haps = |hap_idx: HapIdx,
                                        priors: Option<&HaplotypePriors>,
                                        donors: &[(RefHapId, u32)]|
                 -> Vec<RefHapId> {
                    let panel_haps = ref_allele_freqs.n_ref_haps();
                    if panel_haps > 0 && panel_haps <= 256 {
                        // For small reference panels, exact full-state LS is
                        // cheap and avoids accuracy loss from unnecessary
                        // per-haplotype state truncation.
                        return (0..panel_haps).map(|h| RefHapId::new(h as u32)).collect();
                    }
                    let full_panel_window = per_window_cap_local >= plan.n_ref_haps;
                    if full_panel_window {
                        // In full-panel windows, run the exact Li-Stephens state
                        // space for this haplotype instead of a mixed subset.
                        return (0..plan.n_ref_haps)
                            .map(|h| RefHapId::new(h as u32))
                            .collect();
                    }
                    if let Some(full) = full_states.as_ref() {
                        return full.clone();
                    }
                    let k = if plan.n_ref_haps <= 512 {
                        // For small/medium panels, avoid over-pruning states:
                        // preserve at least 80% of panel states (bounded by panel size).
                        let floor = (plan.n_ref_haps * 8 + 9) / 10;
                        per_window_cap_local.max(floor).min(plan.n_ref_haps).max(1)
                    } else {
                        per_window_cap_local.max(1)
                    };
                    let mut out: Vec<RefHapId> = Vec::with_capacity(k);
                    let mut seen: std::collections::HashSet<RefHapId> =
                        std::collections::HashSet::with_capacity(k * 2);

                    let mut prior_haps: Vec<RefHapId> = Vec::new();
                    if let Some(p) = priors {
                        let mut weighted: Vec<(RefHapId, f32)> = p
                            .ids()
                            .iter()
                            .zip(p.probs().iter())
                            .map(|(id, prob)| (RefHapId::new(id.0), *prob))
                            .collect();
                        weighted.sort_by(|a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        prior_haps.extend(weighted.into_iter().map(|(hap, _)| hap));
                    }

                    let window_haps: Vec<RefHapId> = state_haps_by_hap
                        .get(hap_idx.as_usize())
                        .cloned()
                        .unwrap_or_default();
                    // When stitched handoff priors are available, avoid injecting fresh
                    // PBWT donor IDs into the state set. Handoff priors already carry the
                    // posterior continuity signal; adding local donor IDs at the seam can
                    // create boundary-only state churn that is absent in single-window runs.
                    let donor_haps: Vec<RefHapId> = if priors.is_some() {
                        Vec::new()
                    } else {
                        donors.iter().map(|(hap, _)| *hap).collect()
                    };
                    let core_haps: Vec<RefHapId> = plan
                        .core_states
                        .get(hap_idx.as_usize())
                        .cloned()
                        .unwrap_or_default();

                    let fill_from = |out: &mut Vec<RefHapId>,
                                     seen: &mut std::collections::HashSet<RefHapId>,
                                     source: &[RefHapId],
                                     want: usize,
                                     k: usize|
                     -> usize {
                        if want == 0 || out.len() >= k {
                            return 0;
                        }
                        let mut added = 0usize;
                        for &hap in source {
                            if out.len() >= k || added >= want {
                                break;
                            }
                            if seen.insert(hap) {
                                out.push(hap);
                                added += 1;
                            }
                        }
                        added
                    };

                    let mut q_prior = k * STATE_MIX_PRIOR_FRAC_NUM / STATE_MIX_FRAC_DEN;
                    let mut q_window = k * STATE_MIX_WINDOW_FRAC_NUM / STATE_MIX_FRAC_DEN;
                    let mut q_donor = if donor_haps.is_empty() {
                        0
                    } else {
                        k * STATE_MIX_DONOR_FRAC_NUM / STATE_MIX_FRAC_DEN
                    };
                    let mut q_core = k * STATE_MIX_CORE_FRAC_NUM / STATE_MIX_FRAC_DEN;
                    let mut used_q = q_prior + q_window + q_donor + q_core;
                    while used_q < k {
                        q_prior += 1;
                        used_q += 1;
                        if used_q >= k {
                            break;
                        }
                        q_window += 1;
                        used_q += 1;
                        if used_q >= k {
                            break;
                        }
                        q_donor += 1;
                        used_q += 1;
                        if used_q >= k {
                            break;
                        }
                        q_core += 1;
                        used_q += 1;
                    }

                    fill_from(&mut out, &mut seen, &prior_haps, q_prior, k);
                    fill_from(&mut out, &mut seen, &window_haps, q_window, k);
                    fill_from(&mut out, &mut seen, &donor_haps, q_donor, k);
                    fill_from(&mut out, &mut seen, &core_haps, q_core, k);

                    while out.len() < k {
                        let before = out.len();
                        let remaining = k - out.len();
                        fill_from(&mut out, &mut seen, &prior_haps, remaining, k);
                        let remaining = k - out.len();
                        fill_from(&mut out, &mut seen, &window_haps, remaining, k);
                        let remaining = k - out.len();
                        fill_from(&mut out, &mut seen, &donor_haps, remaining, k);
                        let remaining = k - out.len();
                        fill_from(&mut out, &mut seen, &core_haps, remaining, k);
                        if out.len() == before {
                            break;
                        }
                    }

                    if out.len() < k {
                        // Deterministically complete the state set from the
                        // reference panel. This prevents silent under-filled
                        // state spaces when mixed sources do not cover `k`.
                        for h in 0..plan.n_ref_haps {
                            if out.len() >= k {
                                break;
                            }
                            let hap = RefHapId::new(h as u32);
                            if seen.insert(hap) {
                                out.push(hap);
                            }
                        }
                    }

                    out
                };

                let mut process_haplotype = |hap_idx: HapIdx,
                                             priors: Option<&HaplotypePriors>,
                                             input_probs: &TargetAlleleProbs,
                                             error_rate: f32,
                                             prior_marker_idx: Option<usize>,
                                             donors: &[(RefHapId, u32)]|
                 -> Result<(Vec<AllelePosteriors>, HaplotypePriors, crate::model::impute_hmm::EmStats)> {
                    let state_haps = build_state_haps(hap_idx, priors, donors);
                    if state_haps.is_empty() {
                        return Err(ReagleError::vcf(format!(
                            "State selection produced empty subset: window={} sample={} hap={} donors={} has_priors={}",
                            window_idx,
                            s,
                            hap_idx.as_usize(),
                            donors.len(),
                            priors.is_some()
                        )));
                    }

                    let state_priors: Option<Vec<f32>> = if let Some(p) = priors {
                        if p.is_empty() {
                            if !warned_no_priors {
                                warn!(
                                    "Handoff priors missing for window {} (no markers or no posterior)",
                                    window_idx
                                );
                                warned_no_priors = true;
                            }
                            None
                        } else {
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
                            let mut sum = 0.0f32;
                            for v in mapped.iter() {
                                if v.is_finite() && *v > 0.0 {
                                    sum += *v;
                                }
                            }
                            if sum <= 0.0 {
                                if !warned_empty_map {
                                    warn!(
                                        "State handoff mapped to empty priors for window {} (state set mismatch)",
                                        window_idx
                                    );
                                    warned_empty_map = true;
                                }
                                return Err(ReagleError::vcf(format!(
                                    "Mapped handoff priors collapsed: window={} sample={} hap={} state_count={}",
                                    window_idx,
                                    s,
                                    hap_idx.as_usize(),
                                    state_haps.len()
                                )));
                            }
                            let inv = 1.0 / sum;
                            let norm: Vec<f32> = mapped
                                .into_iter()
                                .map(|v| if v.is_finite() && v > 0.0 { v * inv } else { 0.0 })
                                .collect();
                            Some(norm)
                        }
                    } else if state_haps.len() < plan.n_ref_haps
                        && plan.n_ref_haps > 16
                        && !donors.is_empty()
                    {
                        let mut mapped = vec![0.0f32; state_haps.len()];
                        let mut donor_total = 0.0f32;
                        for (hap, count) in donors.iter() {
                            donor_total += *count as f32;
                            if let Ok(idx) = state_haps.binary_search(hap) {
                                mapped[idx] += *count as f32;
                            }
                        }
                        if donor_total > 0.0 && !mapped.is_empty() {
                            let inv = 1.0f32 / donor_total;
                            for v in mapped.iter_mut() {
                                *v *= inv;
                            }
                            // Donor-guided initialization for first-window/no-handoff cases.
                            // Blend with uniform to preserve coverage while anchoring to local matches.
                            let lambda = 0.20f32;
                            let uniform = 1.0f32 / mapped.len() as f32;
                            for v in mapped.iter_mut() {
                                *v = (1.0 - lambda) * uniform + lambda * *v;
                            }
                            Some(mapped)
                        } else {
                            None
                        }
                    } else {
                        None
                    };

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
                            ImputeHmmContext {
                                window_idx,
                                sample_idx: s,
                                hap_idx: hap_idx.as_usize(),
                            },
                            ws,
                        )
                    })?;

                    let mut next_priors = HaplotypePriors::empty();
                    if let Some(state_post) = state_post.as_ref() {
                        let pairs = state_posteriors_to_priors(&state_haps, state_post, 0.0);
                        if !pairs.is_empty() {
                            let (ids, probs): (Vec<GlobalHapId>, Vec<f32>) = pairs
                                .into_iter()
                                .map(|(g, p)| (GlobalHapId(g.as_u32()), p))
                                .unzip();
                            next_priors = HaplotypePriors::new(ids, probs);
                        }
                    }

                    Ok((posteriors, next_priors, stats))
                };

                let mut hap1_posts: Option<Vec<AllelePosteriors>> = None;
                let mut hap2_posts: Option<Vec<AllelePosteriors>> = None;
                let mut p1_out = HaplotypePriors::empty();
                let mut p2_out = HaplotypePriors::empty();

                if no_info_h1 && has_priors_h1 {
                    if let Some(p) = priors_h1 {
                        hap1_posts = Some(posts_from_priors(p)?);
                        p1_out = p.clone();
                    }
                } else if use_hmm_h1 {
                    let (posts, out, stats) = process_haplotype(
                        h1_idx,
                        priors_h1,
                        &input_probs_h1,
                        (*prior_error_rate).max(err_floor).clamp(1e-6, 0.5),
                        handoff_capture_idx_h1,
                        &donors_h1,
                    )?;
                    let mut posts = posts;
                    if !has_priors_h1
                        && plan.n_ref_haps > 32
                        && !donors_h1.is_empty()
                        && false && conf_ratio_h1 > SM_MATCH_LOW_CONF_FRAC
                    {
                        let donor_posts = posts_from_donors(&donors_h1)?;
                        let t = ((conf_ratio_h1 - SM_MATCH_LOW_CONF_FRAC)
                            / (1.0 - SM_MATCH_LOW_CONF_FRAC).max(1e-6))
                        .clamp(0.0, 1.0);
                        let tau = 1.0 + 1.5 * t;
                        temper_posts(&mut posts, tau);
                        let w = 1.00f32 * t;
                        blend_posts(&mut posts, &donor_posts, w);
                    }
                    hap1_posts = Some(posts);
                    p1_out = out;
                    // Keep imputation emissions stationary across windows.
                    // Window-order-dependent error adaptation introduces
                    // path dependence and boundary drift.
                    let _ = (stats.expected_mismatches, stats.informative_sites);
                } else if has_priors_h1 {
                    if let Some(p) = priors_h1 {
                        hap1_posts = Some(posts_from_priors(p)?);
                        p1_out = p.clone();
                    }
                } else {
                    let total: u32 = donors_h1.iter().map(|(_, c)| *c).sum();
                    if total > 0 {
                        let (ids, probs): (Vec<GlobalHapId>, Vec<f32>) = donors_h1
                            .iter()
                            .map(|(h, c)| (GlobalHapId(h.as_u32()), *c as f32 / total as f32))
                            .unzip();
                        p1_out = HaplotypePriors::new(ids, probs);
                        hap1_posts = Some(posts_from_donors(&donors_h1)?);
                    } else {
                        return Err(ReagleError::vcf(format!(
                            "No subset priors or donors for haplotype: window={} sample={} hap={}",
                            window_idx,
                            s,
                            h1_idx.as_usize()
                        )));
                    }
                }

                if no_info_h2 && has_priors_h2 {
                    if let Some(p) = priors_h2 {
                        hap2_posts = Some(posts_from_priors(p)?);
                        p2_out = p.clone();
                    }
                } else if use_hmm_h2 {
                    let (posts, out, stats) = process_haplotype(
                        h2_idx,
                        priors_h2,
                        &input_probs_h2,
                        (*prior_error_rate).max(err_floor).clamp(1e-6, 0.5),
                        handoff_capture_idx_h2,
                        &donors_h2,
                    )?;
                    let mut posts = posts;
                    if !has_priors_h2
                        && plan.n_ref_haps > 32
                        && !donors_h2.is_empty()
                        && false && conf_ratio_h2 > SM_MATCH_LOW_CONF_FRAC
                    {
                        let donor_posts = posts_from_donors(&donors_h2)?;
                        let t = ((conf_ratio_h2 - SM_MATCH_LOW_CONF_FRAC)
                            / (1.0 - SM_MATCH_LOW_CONF_FRAC).max(1e-6))
                        .clamp(0.0, 1.0);
                        let tau = 1.0 + 1.5 * t;
                        temper_posts(&mut posts, tau);
                        let w = 1.00f32 * t;
                        blend_posts(&mut posts, &donor_posts, w);
                    }
                    hap2_posts = Some(posts);
                    p2_out = out;
                    let _ = (stats.expected_mismatches, stats.informative_sites);
                } else if has_priors_h2 {
                    if let Some(p) = priors_h2 {
                        hap2_posts = Some(posts_from_priors(p)?);
                        p2_out = p.clone();
                    }
                } else {
                    let total: u32 = donors_h2.iter().map(|(_, c)| *c).sum();
                    if total > 0 {
                        let (ids, probs): (Vec<GlobalHapId>, Vec<f32>) = donors_h2
                            .iter()
                            .map(|(h, c)| (GlobalHapId(h.as_u32()), *c as f32 / total as f32))
                            .unzip();
                        p2_out = HaplotypePriors::new(ids, probs);
                        hap2_posts = Some(posts_from_donors(&donors_h2)?);
                    } else {
                        return Err(ReagleError::vcf(format!(
                            "No subset priors or donors for haplotype: window={} sample={} hap={}",
                            window_idx,
                            s,
                            h2_idx.as_usize()
                        )));
                    }
                }

                if let Some(bb) = telemetry.as_ref() {
                    bb.add_samples(1);
                }

                let need_sm_h1 = hap1_posts.is_none();
                let need_sm_h2 = hap2_posts.is_none();
                if need_sm_h1 {
                    sm_needed[h1_idx.as_usize()].store(true, Ordering::Relaxed);
                }
                if need_sm_h2 {
                    sm_needed[h2_idx.as_usize()].store(true, Ordering::Relaxed);
                }

                Ok(ImputeResult {
                    result: SampleImputationResult {
                        sample_idx: s,
                        hap_alt_probs: None,
                        hap_posteriors: match (hap1_posts, hap2_posts) {
                            (Some(p1), Some(p2)) => Some((p1, p2)),
                            _ => None,
                        },
                    },
                    priors: Some((p1_out, p2_out)),
                    last_info_idx: match (last_info_h1, last_info_h2) {
                        (Some(a), Some(b)) => Some(a.max(b)),
                        (Some(a), None) => Some(a),
                        (None, Some(b)) => Some(b),
                        (None, None) => None,
                    },
                }
                )
            })
            .collect::<Result<Vec<_>>>()?;

        eprintln!(
            "    [debug hmm] use_hmm={} no_hmm={} has_priors={} no_info={} insufficient={} low_conf={} few_donors={} fallback_ref_freq={}",
            dbg_use_hmm.load(Ordering::Relaxed),
            dbg_no_hmm.load(Ordering::Relaxed),
            dbg_has_priors.load(Ordering::Relaxed),
            dbg_no_info.load(Ordering::Relaxed),
            dbg_insufficient.load(Ordering::Relaxed),
            dbg_low_conf.load(Ordering::Relaxed),
            dbg_few_donors.load(Ordering::Relaxed),
            dbg_fallback_ref_freq.load(Ordering::Relaxed)
        );

        let output_markers = output_end.saturating_sub(output_start);
        let mut sm_alt_probs_by_hap: Vec<Option<Vec<f32>>> = vec![None; n_target_haps];
        let sm_haps: Vec<usize> = sm_needed
            .iter()
            .enumerate()
            .filter_map(|(i, f)| {
                if f.load(Ordering::Relaxed) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        if !sm_haps.is_empty() && output_markers > 0 {
            let mut pbwt = ReferencePbwt::new(plan.n_ref_haps);
            let mut ref_alleles: Vec<u8> = vec![0u8; plan.n_ref_haps];
            let batch_size = 4096usize;
            let mut batches: Vec<(
                Vec<usize>,
                Vec<RankBeam>,
                Vec<PbwtQueryAllele>,
                Vec<u32>,
                Vec<(u32, u32, u32)>,
            )> = Vec::new();
            let mut start = 0usize;
            while start < sm_haps.len() {
                let end = (start + batch_size).min(sm_haps.len());
                let haps: Vec<usize> = sm_haps[start..end].to_vec();
                let beams = vec![RankBeam::full(plan.n_ref_haps as u32); haps.len()];
                let query_alleles = vec![PbwtQueryAllele::wildcard(); haps.len()];
                let current_donor = vec![0u32; haps.len()];
                let scratch = Vec::new();
                for &hap in &haps {
                    sm_alt_probs_by_hap[hap] = Some(Vec::with_capacity(output_markers));
                }
                batches.push((haps, beams, query_alleles, current_donor, scratch));
                start = end;
            }

            for ref_m in 0..n_ref_markers {
                let col = &ref_columns[ref_m];
                fill_ref_alleles(col, &mut ref_alleles);
                let n_alleles = ref_markers
                    .marker(MarkerIdx::new(ref_m as u32))
                    .n_alleles()
                    .max(1);

                pbwt.prepare_step(&ref_alleles, n_alleles);
                for (haps, beams, query_alleles, _, scratch) in batches.iter_mut() {
                    if let Some(target_m) = alignment.ref_to_target.get(ref_m).and_then(|v| *v) {
                        let target_idx = target_m.as_usize();
                        let mapping = alignment
                            .allele_mappings
                            .get(target_idx)
                            .and_then(|m| m.as_ref());

                        for (i, &hap_idx) in haps.iter().enumerate() {
                            let sample_idx = hap_idx / 2;
                            let local = hap_idx % 2;
                            let h = HapIdx::new((sample_idx * 2 + local) as u32);
                            let mut a = target_win.allele(MarkerIdx::new(target_idx as u32), h);
                            if let Some(missing) = target_missing {
                               if missing.allele(MarkerIdx::new(target_idx as u32), h) == 255 {
                                   a = 255;
                               }
                           }
                            let a = if a == 255 {
                                255
                            } else if let Some(mapping) = mapping {
                                if (a as usize) < mapping.targ_to_ref.len() {
                                    let r = mapping.targ_to_ref[a as usize];
                                    if r >= 0 { r as u8 } else { 255 }
                                } else {
                                    255
                                }
                            } else {
                                a
                            };
                            query_alleles[i] = PbwtQueryAllele::allele(a)
                                .unwrap_or_else(PbwtQueryAllele::wildcard);
                        }
                    } else {
                        for qa in query_alleles.iter_mut() {
                            *qa = PbwtQueryAllele::wildcard();
                        }
                    }

                    pbwt.update_beams_with_scratch_query(beams, query_alleles, n_alleles, scratch);
                }
                pbwt.finalize_step(&ref_alleles, n_alleles, ref_m);

                if ref_m < output_start || ref_m >= output_end {
                    continue;
                }
                for (haps, beams, query_alleles, current_donor, _) in batches.iter_mut() {
                    let mut donor_candidates: Vec<u32> = Vec::with_capacity(SM_MATCH_DONORS);
                    for (i, &hap_idx) in haps.iter().enumerate() {
                        let beam = &beams[i];
                        pbwt.select_donors_into(beam, SM_MATCH_DONORS, &mut donor_candidates);
                        let mut donor = current_donor[i];
                        let mut found = false;
                        for &cand in &donor_candidates {
                            if cand == donor {
                                found = true;
                                break;
                            }
                        }
                        if !found {
                            if let Some(&cand) = donor_candidates.first() {
                                donor = cand;
                                current_donor[i] = donor;
                            }
                        }
                        let target_allele = query_alleles
                            .get(i)
                            .and_then(|qa| qa.as_allele())
                            .unwrap_or(255);
                        let p_alt = if n_alleles <= 2 {
                            if target_allele == 255 {
                                if donor_candidates.is_empty() {
                                    panic!(
                                        "SM-match donor set empty at ref_m={} for hap_idx={} (cannot impute missing site)",
                                        ref_m, hap_idx
                                    );
                                }
                                let mut alt_sum = 0u32;
                                for &cand in &donor_candidates {
                                    let allele = col.get(HapIdx::new(cand));
                                    if allele == 1 {
                                        alt_sum += 1;
                                    }
                                }
                                (alt_sum as f32 / donor_candidates.len() as f32)
                                    .clamp(1e-6, 1.0 - 1e-6)
                            } else {
                                let allele = col.get(HapIdx::new(donor));
                                if allele == 1 { 1.0 } else { 0.0 }
                            }
                        } else {
                            0.0
                        };
                        if let Some(buf) = sm_alt_probs_by_hap[hap_idx].as_mut() {
                            buf.push(p_alt);
                        }
                    }
                }
            }
        }

        let mut all_results = Vec::with_capacity(n_target_samples);
        let mut next_priors_vec = vec![HaplotypePriors::empty(); n_target_samples * 2];
        let mut handoff_marker_idx: Option<usize> = None;

        for mut item in sample_results {
            let sample_idx = item.result.sample_idx;
            let h1 = sample_idx * 2;
            let h2 = h1 + 1;
            let need1 = sm_alt_probs_by_hap
                .get(h1)
                .and_then(|v| v.as_ref())
                .is_some();
            let need2 = sm_alt_probs_by_hap
                .get(h2)
                .and_then(|v| v.as_ref())
                .is_some();
            if (need1 || need2) && output_markers > 0 {
                let p1 = sm_alt_probs_by_hap
                    .get_mut(h1)
                    .and_then(|v| v.take())
                    .unwrap_or_default();
                let p2 = sm_alt_probs_by_hap
                    .get_mut(h2)
                    .and_then(|v| v.take())
                    .unwrap_or_default();
                item.result.hap_alt_probs = Some((p1, p2));
            }
            all_results.push(item.result);
            if let Some((p1, p2)) = item.priors {
                let base = sample_idx * 2;
                if base + 1 < next_priors_vec.len() {
                    next_priors_vec[base] = p1;
                    next_priors_vec[base + 1] = p2;
                }
            }
            if let Some(idx) = item.last_info_idx {
                handoff_marker_idx = Some(match handoff_marker_idx {
                    Some(prev) => prev.max(idx),
                    None => idx,
                });
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
        let handoff_marker_idx = handoff_marker_idx.or(prior_marker_idx);
        let handoff_global_idx = handoff_marker_idx.map(|idx| idx + global_start);
        let handoff_gen_pos = handoff_marker_idx.and_then(|idx| gen_positions.get(idx).copied());
        Ok(Some(ImputationWindowResults {
            all_results,
            ref_is_biallelic,
            handoff: Some(ImputationHandoff {
                priors: next_priors_vec,
                prior_global_idx: handoff_global_idx,
                prior_gen_pos: handoff_gen_pos,
            }),
        }))
    }
    fn extract_imputed_overlap_streaming<TargetSpace, RefSpace>(
        &self,
        target_win: &GenotypeMatrix<Phased, TargetSpace>,
        alignment: &MarkerAlignment<TargetSpace, RefSpace>,
        output_start: usize,
        output_end: usize,
        all_results: &[SampleImputationResult],
    ) -> PhasedOverlap {
        let overlap_size = 1000.min(output_end);
        let start = output_end.saturating_sub(overlap_size);
        let end = output_end;
        let n_haps = target_win.n_haplotypes();
        let mut alleles = vec![255u8; overlap_size * n_haps];
        let n_samples = target_win.n_samples();
        let mut result_by_sample: Vec<Option<&SampleImputationResult>> = vec![None; n_samples];
        for result in all_results {
            if result.sample_idx < n_samples {
                result_by_sample[result.sample_idx] = Some(result);
            }
        }
        for h in 0..n_haps {
            let sample_idx = h / 2;
            let hap_idx = h % 2;
            let posteriors = result_by_sample
                .get(sample_idx)
                .and_then(|r| *r)
                .and_then(|r| r.hap_posteriors.as_ref());
            for (local_m, ref_m) in (start..end).enumerate() {
                let out_local = ref_m.saturating_sub(output_start);
                if let Some((p1, p2)) = posteriors {
                    let post = if hap_idx == 0 { p1 } else { p2 };
                    if let Some(ap) = post.get(out_local) {
                        let allele = match ap {
                            AllelePosteriors::Biallelic(p_alt) => {
                                if *p_alt >= 0.5 {
                                    1u8
                                } else {
                                    0u8
                                }
                            }
                            AllelePosteriors::Multiallelic(probs) => probs
                                .iter()
                                .enumerate()
                                .max_by(|a, b| {
                                    a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)
                                })
                                .map(|(i, _)| i as u8)
                                .unwrap_or(255),
                        };
                        alleles[h * overlap_size + local_m] = allele;
                        continue;
                    }
                }
                if let Some(target_m) = alignment.target_marker(MarkerIdx::new(ref_m as u32)) {
                    alleles[h * overlap_size + local_m] =
                        target_win.allele(target_m, HapIdx::new(h as u32));
                }
            }
        }
        PhasedOverlap::new(overlap_size, n_haps, alleles)
    }

    /// Write imputed window results to VCF
    #[allow(clippy::too_many_arguments)]
    fn write_imputed_window_streaming<
        TargetSpace: Sync,
        RefMarkerSpace: Sync,
        TargetMissingState: PhaseState + Sync,
    >(
        &self,
        ref_markers: &crate::data::marker::Markers<RefMarkerSpace>,
        target_win: &GenotypeMatrix<Phased, TargetSpace>,
        target_pl: Option<&GenotypeMatrix<Phased, TargetSpace>>,
        target_missing: Option<&GenotypeMatrix<TargetMissingState, TargetSpace>>,
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

        let samples = target_win.samples_arc();
        let target_pl = target_pl.unwrap_or(target_win);
        let mut target_pos_to_indices: std::collections::HashMap<(String, u32), Vec<usize>> =
            std::collections::HashMap::new();
        for t_idx in 0..target_win.n_markers() {
            let t_marker = target_win.marker(MarkerIdx::new(t_idx as u32));
            let t_chrom = target_win
                .markers()
                .chrom_name(t_marker.chrom)
                .unwrap_or("");
            let key = (normalize_chrom_local(t_chrom).to_string(), t_marker.pos);
            target_pos_to_indices.entry(key).or_default().push(t_idx);
        }

        let pick_target_marker_by_alleles = |ref_marker_idx: usize| -> Option<usize> {
            let ref_marker = ref_markers.marker(MarkerIdx::new(ref_marker_idx as u32));
            if ref_marker.n_alleles() != 2 {
                return None;
            }
            let ref_chrom = ref_markers.chrom_name(ref_marker.chrom).unwrap_or("");
            let key = (normalize_chrom_local(ref_chrom).to_string(), ref_marker.pos);
            let candidates = target_pos_to_indices.get(&key)?;

            let ref0 = ref_marker.ref_allele.to_string();
            let ref1 = ref_marker.alt_alleles.first()?.to_string();
            let mut matches: Vec<usize> = Vec::new();
            for &t_idx in candidates {
                let t_marker = target_win.marker(MarkerIdx::new(t_idx as u32));
                if t_marker.n_alleles() != 2 {
                    continue;
                }
                let t0 = t_marker.ref_allele.to_string();
                let t1 = t_marker.alt_alleles.first().map(|a| a.to_string());
                if let Some(t1) = t1 {
                    let same = ref0 == t0 && ref1 == t1;
                    let swapped = ref0 == t1 && ref1 == t0;
                    if same || swapped {
                        matches.push(t_idx);
                    }
                }
            }
            if matches.len() == 1 {
                Some(matches[0])
            } else {
                None
            }
        };

        let get_genotyped_alleles = |marker_idx: usize, sample_idx: usize| -> Option<(u8, u8)> {
            let h1 = HapIdx::new((sample_idx * 2) as u32);
            let h2 = HapIdx::new((sample_idx * 2 + 1) as u32);
            if let Some(target_m) = alignment.target_marker(MarkerIdx::new(marker_idx as u32)) {
                if let Some(missing) = target_missing {
                    let miss_a1 = missing.allele(target_m, h1);
                    let miss_a2 = missing.allele(target_m, h2);
                    if miss_a1 == 255 || miss_a2 == 255 {
                        return None;
                    }
                }
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
                    return None;
                }
                return Some((a1, a2));
            }

            // Position-only fallback for unaligned markers: preserve target hard calls
            // when there is a unique marker at the same position.
            let target_idx = pick_target_marker_by_alleles(marker_idx)?;
            let target_m = MarkerIdx::new(target_idx as u32);
            if let Some(missing) = target_missing {
                let miss_a1 = missing.allele(target_m, h1);
                let miss_a2 = missing.allele(target_m, h2);
                if miss_a1 == 255 || miss_a2 == 255 {
                    return None;
                }
            }
            let raw_a1 = target_win.allele(target_m, h1);
            let raw_a2 = target_win.allele(target_m, h2);
            if raw_a1 == 255 || raw_a2 == 255 {
                return None;
            }
            let a1 = if raw_a1 == 0 { 0 } else { 1 };
            let a2 = if raw_a2 == 0 { 0 } else { 1 };
            Some((a1, a2))
        };

        let get_target_raw_dosage = |marker_idx: usize, sample_idx: usize| -> Option<f32> {
            let h1 = HapIdx::new((sample_idx * 2) as u32);
            let h2 = HapIdx::new((sample_idx * 2 + 1) as u32);
            let ref_marker = ref_markers.marker(MarkerIdx::new(marker_idx as u32));
            if ref_marker.n_alleles() != 2 {
                return None;
            }

            let aligned_target_m = alignment.target_marker(MarkerIdx::new(marker_idx as u32));
            let target_m = if let Some(tm) = aligned_target_m {
                tm
            } else {
                let target_idx = pick_target_marker_by_alleles(marker_idx)?;
                MarkerIdx::new(target_idx as u32)
            };

            if let Some(missing) = target_missing {
                if missing.allele(target_m, h1) == 255 || missing.allele(target_m, h2) == 255 {
                    return None;
                }
            }

            let a1 = target_win.allele(target_m, h1);
            let a2 = target_win.allele(target_m, h2);
            if a1 == 255 || a2 == 255 {
                return None;
            }

            let d = if let Some(tm) = aligned_target_m {
                let mapping = alignment
                    .allele_mappings
                    .get(tm.as_usize())
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
                let ra1 = map_allele(a1);
                let ra2 = map_allele(a2);
                if ra1 == 255 || ra2 == 255 {
                    return None;
                }
                ((ra1 > 0) as u8 + (ra2 > 0) as u8) as f32
            } else {
                ((a1 > 0) as u8 + (a2 > 0) as u8) as f32
            };
            if samples.is_diploid(SampleIdx::new(sample_idx as u32)) {
                Some(d)
            } else {
                Some(d * 0.5)
            }
        };

        let get_posteriors_for_writer = if include_posteriors {
            Some(|marker_idx: usize, sample_idx: usize| {
                let _ = ref_markers;
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
            let hard_call = get_genotyped_alleles(marker_idx, sample_idx);

            // Prefer hard calls if error correction is disabled
            if !correct_errors {
                if let Some(d) = get_target_raw_dosage(marker_idx, sample_idx) {
                    return d;
                }
                if let Some((a1, a2)) = hard_call {
                    let d = (a1 + a2) as f32;
                    return if samples.is_diploid(SampleIdx::new(sample_idx as u32)) {
                        d
                    } else {
                        d * 0.5
                    };
                }
            }

            let n_alleles = ref_markers
                .marker(MarkerIdx::new(marker_idx as u32))
                .n_alleles()
                .max(1);
            let local_m = marker_idx.saturating_sub(output_start);

            let dosage = if let Some(result) = result_by_sample.get(sample_idx).and_then(|r| *r) {
                if let Some((p1, p2)) = result.hap_posteriors.as_ref() {
                    let d1 = p1
                        .get(local_m)
                        .map(|p| match p {
                            AllelePosteriors::Biallelic(p_alt) => *p_alt,
                            AllelePosteriors::Multiallelic(probs) => {
                                probs.iter().enumerate().map(|(i, p)| i as f32 * p).sum()
                            }
                        })
                        .unwrap_or(0.0);
                    let d2 = p2
                        .get(local_m)
                        .map(|p| match p {
                            AllelePosteriors::Biallelic(p_alt) => *p_alt,
                            AllelePosteriors::Multiallelic(probs) => {
                                probs.iter().enumerate().map(|(i, p)| i as f32 * p).sum()
                            }
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
                } else {
                    0.0
                }
            } else if let Some((a1, a2)) = hard_call {
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
            let hard_call = get_genotyped_alleles(marker_idx, sample_idx);

            // Prefer hard calls if error correction is disabled
            if !correct_errors {
                if let Some(gt) = hard_call {
                    return gt;
                }
            }

            let local_m = marker_idx.saturating_sub(output_start);

            if let Some(result) = result_by_sample.get(sample_idx).and_then(|r| *r) {
                let n_alleles = ref_markers
                    .marker(MarkerIdx::new(marker_idx as u32))
                    .n_alleles()
                    .max(1);
                if let Some((p1, p2)) = result.hap_posteriors.as_ref() {
                    if n_alleles <= 2 {
                        let p1_alt = p1.get(local_m).map(|p| p.prob(1)).unwrap_or(0.0);
                        let p2_alt = p2.get(local_m).map(|p| p.prob(1)).unwrap_or(0.0);
                        let gp00 = (1.0 - p1_alt) * (1.0 - p2_alt);
                        let gp01 = p1_alt * (1.0 - p2_alt) + (1.0 - p1_alt) * p2_alt;
                        let gp11 = p1_alt * p2_alt;
                        if gp01 >= gp00 && gp01 >= gp11 {
                            let p10 = p1_alt * (1.0 - p2_alt);
                            let p01 = (1.0 - p1_alt) * p2_alt;
                            if p10 >= p01 { (1, 0) } else { (0, 1) }
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
                            if p10 >= p01 { (1, 0) } else { (0, 1) }
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
                    let n_alleles = ref_markers
                        .marker(MarkerIdx::new(marker_idx as u32))
                        .n_alleles();
                    best_gt_from_gp(n_alleles, &gp)
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

        // Preserve target-only markers that share positions with reference markers
        // but are not allele-aligned. These are still genotyped target variants.
        let mut ref_pos_set: std::collections::HashSet<(String, u32)> =
            std::collections::HashSet::new();
        for ref_m in output_start..output_end {
            let ref_marker = ref_markers.marker(MarkerIdx::new(ref_m as u32));
            let ref_chrom = ref_markers.chrom_name(ref_marker.chrom).unwrap_or("");
            ref_pos_set.insert((normalize_chrom_local(ref_chrom).to_string(), ref_marker.pos));
        }

        let mut target_only_by_pos: std::collections::HashMap<(String, u32), Vec<usize>> =
            std::collections::HashMap::new();
        for t_idx in 0..target_win.n_markers() {
            let target_m = MarkerIdx::new(t_idx as u32);
            if alignment.target_to_ref(target_m).is_some() {
                continue;
            }
            let t_marker = target_win.marker(target_m);
            let t_chrom = target_win
                .markers()
                .chrom_name(t_marker.chrom)
                .unwrap_or("");
            let key = (normalize_chrom_local(t_chrom).to_string(), t_marker.pos);
            if ref_pos_set.contains(&key) {
                target_only_by_pos.entry(key).or_default().push(t_idx);
            }
        }

        if target_only_by_pos.is_empty() {
            return writer.write_imputed_streaming(
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
            );
        }

        #[derive(Clone, Copy)]
        enum OutputMarker {
            Ref(usize),
            Target(usize),
        }

        let mut output_markers: Vec<OutputMarker> = Vec::new();
        for ref_m in output_start..output_end {
            let ref_marker = ref_markers.marker(MarkerIdx::new(ref_m as u32));
            let ref_chrom = ref_markers.chrom_name(ref_marker.chrom).unwrap_or("");
            let key = (normalize_chrom_local(ref_chrom).to_string(), ref_marker.pos);
            if let Some(targets) = target_only_by_pos.get(&key) {
                for &t_idx in targets {
                    output_markers.push(OutputMarker::Target(t_idx));
                }
            }
            output_markers.push(OutputMarker::Ref(ref_m));
        }

        let mut out_markers = crate::data::marker::Markers::<RefMarkerSpace>::new();
        let mut out_chroms: std::collections::HashMap<String, ChromIdx> =
            std::collections::HashMap::new();
        for name in ref_markers.chrom_names() {
            out_markers.add_chrom(name.as_ref());
            let idx = ChromIdx::new((out_markers.chrom_names().len() - 1) as u16);
            out_chroms.insert(normalize_chrom_local(name.as_ref()).to_string(), idx);
        }

        let mut n_alleles_per_marker: Vec<usize> = Vec::with_capacity(output_markers.len());
        for om in &output_markers {
            match *om {
                OutputMarker::Ref(ref_m) => {
                    let marker = ref_markers.marker(MarkerIdx::new(ref_m as u32)).clone();
                    n_alleles_per_marker.push(marker.n_alleles());
                    out_markers.push(marker);
                }
                OutputMarker::Target(t_idx) => {
                    let t_marker = target_win.marker(MarkerIdx::new(t_idx as u32)).clone();
                    let t_chrom = target_win
                        .markers()
                        .chrom_name(t_marker.chrom)
                        .unwrap_or("");
                    let key = normalize_chrom_local(t_chrom).to_string();
                    let out_idx = if let Some(idx) = out_chroms.get(&key).copied() {
                        idx
                    } else {
                        out_markers.add_chrom(t_chrom);
                        let idx = ChromIdx::new((out_markers.chrom_names().len() - 1) as u16);
                        out_chroms.insert(key, idx);
                        idx
                    };
                    let mut marker = t_marker;
                    marker.chrom = out_idx;
                    n_alleles_per_marker.push(marker.n_alleles());
                    out_markers.push(marker);
                }
            }
        }

        let mut merged_quality = ImputationQuality::new(&n_alleles_per_marker);
        let target_samples = target_win.samples_arc();
        let get_target_alleles = |t_idx: usize, s: usize| -> Option<(u8, u8)> {
            let h1 = HapIdx::new((s * 2) as u32);
            let h2 = HapIdx::new((s * 2 + 1) as u32);
            let m = MarkerIdx::new(t_idx as u32);
            if let Some(missing) = target_missing {
                if missing.allele(m, h1) == 255 || missing.allele(m, h2) == 255 {
                    return None;
                }
            }
            let a1 = target_win.allele(m, h1);
            let a2 = target_win.allele(m, h2);
            if a1 == 255 || a2 == 255 {
                None
            } else {
                Some((a1, a2))
            }
        };

        for (out_idx, om) in output_markers.iter().enumerate() {
            match *om {
                OutputMarker::Ref(ref_m) => {
                    if let Some(stats) = quality.get(ref_m) {
                        merged_quality.marker_stats[out_idx] = stats.clone();
                    }
                }
                OutputMarker::Target(t_idx) => {
                    let n_alleles = n_alleles_per_marker[out_idx];
                    let mut stats = crate::io::vcf::MarkerImputationStats::new(n_alleles);
                    stats.is_imputed = false;
                    if n_alleles == 2 {
                        for s in 0..target_win.n_samples() {
                            if let Some((a1, a2)) = get_target_alleles(t_idx, s) {
                                let p1 = if a1 == 1 { 1.0 } else { 0.0 };
                                let p2 = if a2 == 1 { 1.0 } else { 0.0 };
                                stats.add_sample_biallelic(p1, p2);
                            }
                        }
                    }
                    merged_quality.marker_stats[out_idx] = stats;
                }
            }
        }

        let output_markers = std::sync::Arc::new(output_markers);
        let n_alleles_per_marker = std::sync::Arc::new(n_alleles_per_marker);

        let get_dosage_merged = {
            let output_markers = output_markers.clone();
            move |out_idx: usize, s: usize| -> f32 {
                match output_markers[out_idx] {
                    OutputMarker::Ref(ref_m) => get_dosage(ref_m, s),
                    OutputMarker::Target(t_idx) => {
                        if let Some((a1, a2)) = get_target_alleles(t_idx, s) {
                            let is_diploid = target_samples.is_diploid(SampleIdx::new(s as u32));
                            let d = (a1 > 0) as u8 + (a2 > 0) as u8;
                            if is_diploid {
                                d as f32
                            } else {
                                (d as f32) * 0.5
                            }
                        } else {
                            0.0
                        }
                    }
                }
            }
        };

        let get_best_gt_merged = {
            let output_markers = output_markers.clone();
            move |out_idx: usize, s: usize| -> (u8, u8) {
                match output_markers[out_idx] {
                    OutputMarker::Ref(ref_m) => get_best_gt(ref_m, s),
                    OutputMarker::Target(t_idx) => {
                        get_target_alleles(t_idx, s).unwrap_or((255, 255))
                    }
                }
            }
        };

        let get_posteriors_merged = get_posteriors_for_writer.as_ref().map(|base| {
            let output_markers = output_markers.clone();
            let n_alleles_per_marker = n_alleles_per_marker.clone();
            move |out_idx: usize, s: usize| match output_markers[out_idx] {
                OutputMarker::Ref(ref_m) => base(ref_m, s),
                OutputMarker::Target(_) => {
                    let n_alleles = n_alleles_per_marker[out_idx].max(1);
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
                }
            }
        });

        let genotype_index = |a: usize, b: usize| -> usize {
            let (i, j) = if a <= b { (a, b) } else { (b, a) };
            j * (j + 1) / 2 + i
        };

        let get_genotype_posteriors_merged = if include_gp && !correct_errors {
            let output_markers = output_markers.clone();
            let n_alleles_per_marker = n_alleles_per_marker.clone();
            Some(
                move |out_idx: usize, s: usize| match output_markers[out_idx] {
                    OutputMarker::Ref(ref_m) => get_genotype_posteriors(ref_m, s),
                    OutputMarker::Target(t_idx) => {
                        let n_alleles = n_alleles_per_marker[out_idx].max(1);
                        let n_genotypes = n_alleles * (n_alleles + 1) / 2;
                        let mut gp = vec![0.0f32; n_genotypes];
                        if let Some((a1, a2)) = get_target_alleles(t_idx, s) {
                            let idx = genotype_index(a1 as usize, a2 as usize);
                            if idx < gp.len() {
                                gp[idx] = 1.0;
                            }
                        }
                        Some(gp)
                    }
                },
            )
        } else {
            None
        };

        writer.write_imputed_streaming(
            &out_markers,
            get_dosage_merged,
            get_best_gt_merged,
            get_posteriors_merged,
            get_genotype_posteriors_merged,
            &merged_quality,
            0,
            out_markers.len(),
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
    use crate::data::storage::phase_state::{Phased, Unphased};
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
            None::<&GenotypeMatrix<Phased, crate::data::AnyMarkerSpace>>,
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
