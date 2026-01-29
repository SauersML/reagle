//! Streaming Imputation Pipeline
//!
//! Implements memory-efficient streaming imputation through overlapping windows.
//! Uses a producer-consumer model with MPSC channel to pipe phased matrices
//! directly to imputation in-memory.

use std::io::BufRead;
use std::path::Path;
use std::sync::Arc;
use std::collections::HashMap;

use rayon::prelude::*;
use sysinfo::System;
use tracing::{info_span, instrument, warn};

use crate::data::alignment::MarkerAlignment;
use crate::data::genetic_map::GeneticMaps;
use crate::data::storage::{GenotypeColumn, GenotypeMatrix};
use crate::data::storage::phase_state::Phased;
use crate::data::{HapIdx, MarkerIdx, SampleIdx};
use crate::error::ReagleError;
use crate::error::Result;
use crate::io::bref3::{RefPanelReader, TargetMarkerIndex};
use crate::io::streaming::{
    GlobalHapId, HaplotypePriors, PhasedOverlap, StreamingConfig, StreamingVcfReader,
};
use crate::io::vcf::{ImputationQuality, VcfWriter};
use crate::model::pl_emission::{
    allele_probs_cond_from_pl, allele_probs_uncond_from_pl, genotype_probs_from_pl,
    infer_n_alleles_from_pl_len,
};
use crate::model::reference_pbwt::{RankBeam, ReferencePbwt};
use crate::model::types::GlobalId;
use crate::model::impute_hmm::{
    ImputeWorkspace, TargetAlleleProbs, run_impute_hmm, state_posteriors_to_priors,
};
use crate::model::transition_matrix::TransitionMatrix;
use crate::pipelines::imputation::AllelePosteriors;


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

const PBWT_SELECT_BLOCK_CM: f64 = 0.1;
const PBWT_PER_WINDOW_MULT: usize = 8;
const PBWT_MIN_PER_HAP: usize = 64;
const PBWT_MAX_PER_HAP: usize = 256;
const ABYSS_RANK_BASE: usize = 60;
const IMPUTE_RAM_FRACTION: f64 = 0.25;
const STATE_BUDGET_SAFETY: f64 = 0.6;
const SCAN_RAM_FRACTION: f64 = 0.10;
const EXACT_PRESCAN_MAX_OPS: u128 = 250_000_000;
const MIN_AVAIL_BYTES_FOR_PLANNING: u64 = 64 * 1024 * 1024;

fn available_memory_bytes() -> Option<u64> {
    let mut sys = System::new();
    sys.refresh_memory();
    // sysinfo reports memory values in bytes.
    let avail_bytes = sys.available_memory();
    if avail_bytes >= MIN_AVAIL_BYTES_FOR_PLANNING {
        return Some(avail_bytes);
    }
    // Fallback: if available memory is unavailable (some CI/containers),
    // use total memory rather than collapsing to an unusable cap.
    let total_bytes = sys.total_memory();
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
    // Per-state memory: fwd + bwd + emissions + weights + fwd_history per marker.
    let per_state_bytes = 4usize.saturating_mul(4 + window_markers);
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
    core_states: Vec<Vec<GlobalId>>,          // per target hap (derived)
    window_intervals: Vec<Vec<HapIntervals>>, // per target hap (sparse)
    abyss_mask: Vec<Vec<bool>>,               // per target hap
    per_window_cap: usize,
    per_window_caps: Vec<usize>, // per window (global, same for all target haps)
}

#[derive(Clone, Debug)]
struct HapIntervals {
    hap: GlobalId,
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

fn compute_target_freqs<TargetSpace, RefSpace>(
    target_gt: &GenotypeMatrix<Phased, TargetSpace>,
    ref_columns: &[GenotypeColumn],
    alignment: &MarkerAlignment<TargetSpace, RefSpace>,
) -> Vec<Vec<f32>> {
    let n_markers = target_gt.n_markers();
    let n_ref_haps = ref_columns
        .first()
        .map(|c| c.n_haplotypes())
        .unwrap_or(0);
    let mut freqs: Vec<Vec<f32>> = Vec::with_capacity(n_markers);
    for m in 0..n_markers {
        let n_alleles = target_gt
            .markers()
            .marker(MarkerIdx::new(m as u32))
            .n_alleles();
        let mut counts = vec![0u32; n_alleles.max(1)];
        let mut total = 0u32;
        if let Some(ref_m) = alignment.target_to_ref(MarkerIdx::new(m as u32)) {
            for rh in 0..n_ref_haps {
                let ref_a = ref_columns[ref_m.as_usize()].get(HapIdx::new(rh as u32));
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

fn score_window_batch_exact<TargetSpace, RefSpace>(
    batch_haps: &[usize],
    target_gt: &GenotypeMatrix<Phased, TargetSpace>,
    ref_columns: &[GenotypeColumn],
    alignment: &MarkerAlignment<TargetSpace, RefSpace>,
    global_scores: &mut [Vec<f32>],
    window_scores: &mut [Vec<f32>],
) {
    let n_markers = target_gt.n_markers();
    let n_ref_haps = ref_columns
        .first()
        .map(|c| c.n_haplotypes())
        .unwrap_or(0);
    if n_markers == 0 || n_ref_haps == 0 || batch_haps.is_empty() {
        return;
    }

    let freqs = compute_target_freqs(target_gt, ref_columns, alignment);
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

        for rh in 0..n_ref_haps {
            let ref_a = ref_columns[ref_m.as_usize()].get(HapIdx::new(rh as u32));
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
                window_scores[i][idx] += weight;
            }
        }
    }
}

fn score_window_batch_pbwt<TargetSpace, RefSpace>(
    batch_haps: &[usize],
    target_gt: &GenotypeMatrix<Phased, TargetSpace>,
    ref_columns: &[GenotypeColumn],
    alignment: &MarkerAlignment<TargetSpace, RefSpace>,
    gen_maps: &GeneticMaps,
    k_per_hap: usize,
    step_cm: f64,
    global_scores: &mut [Vec<f32>],
    window_scores: &mut [Vec<f32>],
) {
    let n_markers = target_gt.n_markers();
    let n_ref_haps = ref_columns
        .first()
        .map(|c| c.n_haplotypes())
        .unwrap_or(0);
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
    // Always sample genotyped target markers so prescan state selection
    // captures IBS signal from sparse arrays.
    for m in 0..n_markers {
        if alignment.target_to_ref(MarkerIdx::new(m as u32)).is_some() {
            sampling[m] = true;
        }
    }
    let freqs = compute_target_freqs(target_gt, ref_columns, alignment);

    let mut pbwt_fwd = ReferencePbwt::new(n_ref_haps);
    let mut beams_fwd: Vec<RankBeam> = (0..batch_haps.len())
        .map(|_| RankBeam::full(n_ref_haps as u32))
        .collect();
    let mut ref_alleles = vec![0u8; n_ref_haps];
    let mut query_alleles = vec![0u8; batch_haps.len()];

    let min_freq = 1.0 / (2.0 * n_ref_haps.max(1) as f32);

    for m in 0..n_markers {
        for (i, &hap_idx) in batch_haps.iter().enumerate() {
            query_alleles[i] =
                target_gt.allele(MarkerIdx::new(m as u32), HapIdx::new(hap_idx as u32));
        }
        if let Some(ref_m) = alignment.target_to_ref(MarkerIdx::new(m as u32)) {
            for rh in 0..n_ref_haps {
                let ref_a = ref_columns[ref_m.as_usize()].get(HapIdx::new(rh as u32));
                ref_alleles[rh] = alignment.reverse_map_allele(m, ref_a);
            }
        } else {
            ref_alleles.fill(255);
        }

        let mut is_biallelic = true;
        for &a in ref_alleles.iter().chain(query_alleles.iter()) {
            if a >= 2 && a != 255 {
                is_biallelic = false;
                break;
            }
        }
        let n_alleles = if is_biallelic { 2 } else { 256 };

        pbwt_fwd.advance_with_beams(&ref_alleles, n_alleles, m, &query_alleles, &mut beams_fwd);

        if sampling[m] {
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
                let donors = pbwt_fwd.select_donors(&beams_fwd[i], k_per_hap);
                for d in donors {
                    let idx = d as usize;
                    if idx < n_ref_haps {
                        let ref_a = ref_alleles[idx];
                        if ref_a == 255 || ref_a != targ {
                            continue;
                        }
                        global_scores[i][idx] += weight;
                        window_scores[i][idx] += weight;
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
            query_alleles[i] =
                target_gt.allele(MarkerIdx::new(m as u32), HapIdx::new(hap_idx as u32));
        }
        if let Some(ref_m) = alignment.target_to_ref(MarkerIdx::new(m as u32)) {
            for rh in 0..n_ref_haps {
                let ref_a = ref_columns[ref_m.as_usize()].get(HapIdx::new(rh as u32));
                ref_alleles[rh] = alignment.reverse_map_allele(m, ref_a);
            }
        } else {
            ref_alleles.fill(255);
        }

        let mut is_biallelic = true;
        for &a in ref_alleles.iter().chain(query_alleles.iter()) {
            if a >= 2 && a != 255 {
                is_biallelic = false;
                break;
            }
        }
        let n_alleles = if is_biallelic { 2 } else { 256 };

        pbwt_bwd.advance_with_beams(
            &ref_alleles,
            n_alleles,
            rev_step,
            &query_alleles,
            &mut beams_bwd,
        );

        if sampling[m] {
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
                let donors = pbwt_bwd.select_donors(&beams_bwd[i], k_per_hap);
                for d in donors {
                    let idx = d as usize;
                    if idx < n_ref_haps {
                        let ref_a = ref_alleles[idx];
                        if ref_a == 255 || ref_a != targ {
                            continue;
                        }
                        global_scores[i][idx] += weight;
                        window_scores[i][idx] += weight;
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
    let _ = vcf_reader.samples_arc();
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
        let _ = parts.next();
        let _ = parts.next();
        for _ in 0..6 {
            parts.next();
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
    ref_path: &Path,
    streaming_config: &StreamingConfig,
    gen_maps: &GeneticMaps,
    target_positions: &TargetMarkerIndex,
    per_window_cap: usize,
    available_bytes: u64,
    n_threads: usize,
    imp_step_cm: f64,
    params: &crate::model::parameters::ModelParams,
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
        per_window_cap,
        per_window_caps: Vec::new(),
    };

    let avail = available_bytes;
    let safe_bytes_per_thread = if n_threads == 0 {
        0u64
    } else {
        let budget = (avail as f64 * IMPUTE_RAM_FRACTION) as u64;
        let per_thread = budget / n_threads as u64;
        (per_thread as f64 * STATE_BUDGET_SAFETY) as u64
    };
    let force_full_panel = available_bytes == 0;
    let mut ref_reader = open_ref_reader(ref_path)?;
    let mut n_ref_haps = 0usize;
    loop {
        let ref_window = ref_reader.next_window(
            streaming_config,
            gen_maps,
            Some(target_positions),
        )?;
        let Some(ref_window) = ref_window else { break };
        n_ref_haps = ref_window
            .ref_columns
            .first()
            .map(|c| c.n_haplotypes())
            .unwrap_or(0);
        if n_ref_haps > 0 {
            break;
        }
    }
    if n_ref_haps == 0 {
        return Err(ReagleError::vcf(
            "Reference window scanning found no haplotypes".to_string(),
        ));
    }
    plan.n_ref_haps = n_ref_haps;
    let batch_size = estimate_scan_batch_size(avail, n_ref_haps, n_target_haps);
    let mut batch_start = 0usize;

    // Handoff anchor: (prev_output_end_gen_pos, next_output_start_gen_pos)
    let mut window_handoff: Vec<(f64, f64)> = Vec::new();
    let mut per_window_caps: Vec<usize> = Vec::new();

    // Fast path: if every window can hold the full panel, skip prescan and
    // select all haplotypes globally. If memory is unknown (available_bytes=0),
    // default to full-panel for small/CI runs to avoid degrading accuracy.
    {
        let mut ref_reader = open_ref_reader(ref_path)?;
        loop {
            let ref_window = ref_reader.next_window(
                streaming_config,
                gen_maps,
                Some(target_positions),
            )?;
            let Some(ref_window) = ref_window else { break };
            let n_ref_markers = ref_window.markers.len();
            if n_ref_markers == 0 {
                continue;
            }
            let mut per_window_cap_window = if force_full_panel {
                n_ref_haps.max(1)
            } else {
                let per_state_bytes = 4usize.saturating_mul(4 + n_ref_markers);
                let mut per_window_cap_window = if per_state_bytes == 0 {
                    0
                } else {
                    (safe_bytes_per_thread as usize) / per_state_bytes
                };
                if per_window_cap_window == 0 {
                    per_window_cap_window = 1;
                }
                per_window_cap_window
            };
            per_window_cap_window = per_window_cap_window.min(n_ref_haps).max(1);
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
        }
    }

    if !per_window_caps.is_empty() && per_window_caps.iter().all(|&c| c >= n_ref_haps) {
        let num_windows = per_window_caps.len();
        if num_windows == 0 {
            return Err(ReagleError::vcf(
                "Pre-scan produced no windows for allocation".to_string(),
            ));
        }
        plan.per_window_cap = n_ref_haps.max(1);
        plan.per_window_caps = per_window_caps;
        for hap_idx in 0..n_target_haps {
            let mut intervals = Vec::new();
            let mut core = Vec::new();
            let end = num_windows.saturating_sub(1) as u32;
            for h in 0..n_ref_haps {
                let hap = GlobalId::new(h as u32);
                intervals.push(HapIntervals {
                    hap,
                    intervals: vec![(0, end)],
                });
                core.push(hap);
            }
            plan.window_intervals[hap_idx] = intervals;
            plan.core_states[hap_idx] = core;
            plan.abyss_mask[hap_idx] = vec![false; n_ref_haps];
        }
        return Ok(plan);
    } else {
        // Reset for prescan path (we rebuilt these in the fast-path probe).
        window_handoff.clear();
        per_window_caps.clear();
    }

    while batch_start < n_target_haps {
        let batch_end = (batch_start + batch_size).min(n_target_haps);
        let batch_haps: Vec<usize> = (batch_start..batch_end).collect();
        let batch_len = batch_haps.len();

        let mut ref_reader = open_ref_reader(ref_path)?;
        let mut target_reader =
            StreamingVcfReader::open(target_path, gen_maps.clone(), streaming_config.clone())?;

        let mut global_scores: Vec<Vec<f32>> = Vec::with_capacity(batch_len);
        let mut window_scores: Vec<Vec<f32>> = Vec::with_capacity(batch_len);
        let mut best_window_scores: Vec<Vec<f32>> = Vec::with_capacity(batch_len);
        let mut window_rank_hits: Vec<Vec<u32>> = Vec::with_capacity(batch_len);
        let mut scores_by_window: Vec<Vec<Vec<(usize, f32)>>> = Vec::with_capacity(batch_len);

        for _ in 0..batch_len {
            global_scores.push(Vec::new());
            window_scores.push(Vec::new());
            best_window_scores.push(Vec::new());
            window_rank_hits.push(Vec::new());
            scores_by_window.push(Vec::new());
        }

        let mut window_idx = 0usize;
        loop {
            let ref_window = ref_reader.next_window(
                streaming_config,
                gen_maps,
                Some(target_positions),
            )?;
            let Some(ref_window) = ref_window else { break };

            let n_ref_markers = ref_window.markers.len();
            if n_ref_markers == 0 {
                continue;
            }
            // Derive per-window cap from the observed marker count to match
            // the real workspace footprint (fwd/bwd/history scale with markers).
            let per_state_bytes = 4usize.saturating_mul(4 + n_ref_markers);
            let mut per_window_cap_window = if per_state_bytes == 0 {
                0
            } else {
                (safe_bytes_per_thread as usize) / per_state_bytes
            };
            if per_window_cap_window == 0 {
                per_window_cap_window = 1;
            }
            per_window_cap_window = per_window_cap_window.min(n_ref_haps).max(1);
            if batch_start == 0 {
                per_window_caps.push(per_window_cap_window);
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
                continue;
            };

            let alignment = MarkerAlignment::new_with_ref_markers(
                &target_window.genotypes,
                &ref_window.markers,
            );

            for i in 0..batch_len {
                if global_scores[i].len() != n_ref_haps {
                    global_scores[i] = vec![0.0f32; n_ref_haps];
                    window_scores[i] = vec![0.0f32; n_ref_haps];
                    best_window_scores[i] = vec![0.0f32; n_ref_haps];
                    window_rank_hits[i] = vec![0u32; n_ref_haps];
                }
            }

            for w in window_scores.iter_mut() {
                w.fill(0.0);
            }

            let k_per_hap = per_window_cap_window
                .saturating_mul(PBWT_PER_WINDOW_MULT)
                .max(PBWT_MIN_PER_HAP)
                .min(PBWT_MAX_PER_HAP)
                .max(1);

            let phased_target = target_window.genotypes.clone().into_phased();
            let step_cm = PBWT_SELECT_BLOCK_CM.max(imp_step_cm);
            let use_exact = should_use_exact_prescan(
                n_ref_haps,
                batch_haps.len(),
                phased_target.n_markers(),
            );
            if use_exact {
                score_window_batch_exact(
                    &batch_haps,
                    &phased_target,
                    &ref_window.ref_columns,
                    &alignment,
                    &mut global_scores,
                    &mut window_scores,
                );
            } else {
                score_window_batch_pbwt(
                    &batch_haps,
                    &phased_target,
                    &ref_window.ref_columns,
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

            // Record handoff anchor points once (batch 0), reuse for all batches.
            if batch_start == 0 {
                let output_start = ref_window.output_start.min(n_ref_markers.saturating_sub(1));
                let output_end = ref_window.output_end.min(n_ref_markers).max(1);
                let left_idx = output_end.saturating_sub(1);
                let right_idx = output_start.min(n_ref_markers.saturating_sub(1));
                let left_marker = ref_window.markers.marker(MarkerIdx::new(left_idx as u32));
                let right_marker = ref_window.markers.marker(MarkerIdx::new(right_idx as u32));
                let left_gen = gen_maps.gen_pos(left_marker.chrom, left_marker.pos);
                let right_gen = gen_maps.gen_pos(right_marker.chrom, right_marker.pos);
                if window_handoff.len() == window_idx {
                    window_handoff.push((left_gen, right_gen));
                }
            }
            window_idx += 1;

        }

        let min_step_cm = (streaming_config.overlap_cm as f64)
            .max(imp_step_cm)
            .max(1e-6);
        let boundary_cm = window_boundaries_from_handoff(&window_handoff, min_step_cm);
        if window_handoff.is_empty() {
            return Err(ReagleError::vcf(
                "Pre-scan produced no windows for LMS allocation".to_string(),
            ));
        }
        if !per_window_caps.is_empty() && per_window_caps.len() != window_handoff.len() {
            return Err(ReagleError::vcf(format!(
                "Per-window cap length mismatch (caps={}, bounds={})",
                per_window_caps.len(),
                window_handoff.len()
            )));
        }
        if batch_start == 0 {
            plan.per_window_caps = per_window_caps.clone();
        }

        for (i, &hap_idx) in batch_haps.iter().enumerate() {
            let mut abyss = vec![false; n_ref_haps];
            for h in 0..n_ref_haps {
                let score = best_window_scores[i][h];
                if window_rank_hits[i][h] == 0 || !score.is_finite() || score <= 0.0 {
                    abyss[h] = true;
                }
            }
            if abyss.iter().all(|v| *v) {
                let keep = ((n_ref_haps / 1000).max(ABYSS_RANK_BASE))
                    .min(n_ref_haps)
                    .max(1);
                let top = select_top_k_allow_zero(&global_scores[i], keep);
                if top.is_empty() {
                    for h in 0..keep {
                        abyss[h] = false;
                    }
                } else {
                    for (h, _) in top {
                        abyss[h] = false;
                    }
                }
            }
            // Run LMS allocator for this target haplotype.
            let window_scores_matrix = &scores_by_window[i];
            if window_scores_matrix.len() != window_handoff.len() {
                return Err(ReagleError::vcf(format!(
                    "Pre-scan window count mismatch for hap {} (scores={}, bounds={})",
                    hap_idx,
                    window_scores_matrix.len(),
                    window_handoff.len()
                )));
            }
            // LMS allocation: select a per-window active set that maximizes a
            // Li–Stephens-aligned surrogate objective under a slot budget.
            // This is a prescan-only selection layer; the actual HMM still uses
            // explicit GlobalId states and identity-preserving transitions.
            let num_windows = window_scores_matrix.len();
            let per_window_caps_used = if !plan.per_window_caps.is_empty()
                && plan.per_window_caps.len() == num_windows
            {
                plan.per_window_caps.clone()
            } else {
                vec![per_window_cap.max(1); num_windows]
            };
            let per_window_cap_min = per_window_caps_used
                .iter()
                .copied()
                .min()
                .unwrap_or(per_window_cap.max(1));
            if per_window_cap_min >= n_ref_haps {
                // Full-panel mode: do not exclude abyss in selection, since we
                // can afford to keep all haplotypes and avoid false negatives
                // for rare-variant carriers.
                abyss.fill(false);
            }
            plan.abyss_mask[hap_idx] = abyss.clone();
            let (intervals, core) = if per_window_cap_min >= n_ref_haps {
                // Full panel per window (minus abyss). Store as 1 interval per hap.
                let mut intervals = Vec::new();
                let mut core = Vec::new();
                let end = num_windows.saturating_sub(1) as u32;
                for h in 0..n_ref_haps {
                    if abyss[h] {
                        continue;
                    }
                    let hap = GlobalId::new(h as u32);
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
                    &per_window_caps_used,
                );
                let mut intervals = Vec::new();
                for (hap, spans) in allocation.intervals_by_hap.into_iter() {
                    intervals.push(HapIntervals {
                        hap: GlobalId::new(hap as u32),
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
            plan.window_intervals[hap_idx] = intervals;
            plan.core_states[hap_idx] = core;
        }

        batch_start = batch_end;
    }

    Ok(plan)
}

struct SampleImputationResult {
    sample_idx: usize,
    dosages: Vec<f32>,
    best_gt: Vec<(u8, u8)>,
    hap_alt_probs: Option<(Vec<f32>, Vec<f32>)>,
    hap_posteriors: Option<(Vec<AllelePosteriors>, Vec<AllelePosteriors>)>,
}

struct ImputationHandoff {
    priors: Vec<HaplotypePriors>,
    prior_global_idx: Option<usize>,
    prior_gen_pos: Option<f64>,
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
        let mut phased_p_mismatch: Option<f32> = None;
        if !is_vcf_fully_phased(&phased_target_path)? {
            eprintln!("Target is unphased; running phasing before pre-scan...");
            let tmpdir = tempfile::tempdir()?;
            let phased_prefix = tmpdir.path().join("phased_target");
            let mut phase_config = self.config.clone();
            phase_config.gt = input_target_path.clone();
            phase_config.r#ref = Some(ref_path.to_path_buf());
            phase_config.out = phased_prefix.clone();
            let mut phasing = crate::pipelines::phasing::PhasingPipeline::new(
                phase_config,
                self.telemetry.clone(),
            );
            phasing.run()?;
            phased_p_mismatch = Some(phasing.params().p_mismatch);
            phased_target_path = phased_prefix.with_extension("vcf.gz");
            phased_tmp = Some(tmpdir);
        }

        let mut n_threads = self
            .config
            .nthreads
            .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
            .unwrap_or(1);
        let avail_bytes = available_memory_bytes().unwrap_or(0);
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

        eprintln!(
            "Imputation plan: per_window_cap={}, threads={}, available_mb={}",
            per_window_cap,
            n_threads,
            avail_bytes / (1024 * 1024)
        );

        let plan = build_imputation_plan(
            &phased_target_path,
            ref_path,
            &streaming_config,
            &gen_maps,
            &target_positions_map,
            per_window_cap,
            if force_full_panel { 0 } else { avail_bytes },
            n_threads,
            self.config.imp_step as f64,
            &self.params,
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

        let mut ref_reader = open_ref_reader(ref_path)?;
        let target_was_unphased_for_impute = !is_vcf_fully_phased(&input_target_path)?;
        let target_path_for_impute = if target_was_unphased_for_impute {
            // Use the phased output for imputation to provide haplotype-resolved
            // emissions; this preserves long-range LD signal for HMM inference.
            phased_target_path.clone()
        } else {
            phased_target_path.clone()
        };
        let mut target_reader = StreamingVcfReader::open(
            &target_path_for_impute,
            gen_maps.clone(),
            streaming_config.clone(),
        )?;
        let mut target_reader_pl = if target_was_unphased_for_impute {
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
        if let Some(p_mismatch) = phased_p_mismatch {
            if p_mismatch.is_finite() && p_mismatch > 0.0 {
                self.params.p_mismatch = p_mismatch;
            }
        }
        self.params
            .set_n_states(self.config.phase_states.min(n_ref_pool.saturating_sub(2)));

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

        loop {
            let ref_window = ref_reader.next_window(
                &streaming_config,
                &gen_maps,
                Some(&target_positions_map),
            )?;
            let Some(ref_window) = ref_window else { break };

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

            if !header_written {
                writer.write_header_extended(&ref_window.markers, true, self.config.gp, self.config.ap)?;
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
            for (ref_m, target_idx) in alignment.ref_to_target.iter().enumerate() {
                window_quality.set_imputed(ref_m, target_idx.is_none());
            }

            let next_handoff = self.run_imputation_window_streaming(
                &phased_target,
                phased_target_pl.as_ref(),
                &ref_window.markers,
                &ref_window.ref_columns,
                ref_window.ref_genotypes.as_ref(),
                &alignment,
                &gen_maps,
                imp_overlap.as_ref(),
                &plan,
                window_idx,
                &mut window_quality,
                &mut writer,
                ref_window.global_start,
                ref_window.output_start,
                ref_window.output_end,
                !target_was_unphased_for_impute,
            )?;

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

        let _ = phased_tmp.as_ref();
        let _ = input_tmp.as_ref();

        if total_markers == 0 {
            return Err(ReagleError::vcf(
                "No markers imputed; check reference/target overlap and region selection.",
            ));
        }

        eprintln!("Streaming imputation complete: {} markers", total_markers);
        Ok(())
    }
    fn run_imputation_window_streaming<TargetSpace: Sync, RefMarkerSpace: Sync, RefPhaseSpace>(
        &self,
        target_win: &GenotypeMatrix<Phased, TargetSpace>,
        target_pl: Option<&GenotypeMatrix<Phased, TargetSpace>>,
        ref_markers: &crate::data::marker::Markers<RefMarkerSpace>,
        ref_columns: &[GenotypeColumn],
        ref_genotypes: Option<&GenotypeMatrix<Phased, RefPhaseSpace>>,
        alignment: &MarkerAlignment<TargetSpace, RefMarkerSpace>,
        gen_maps: &GeneticMaps,
        imp_overlap: Option<&PhasedOverlap>,
        plan: &ImputationPlan,
        window_idx: usize,
        window_quality: &mut ImputationQuality,
        final_writer: &mut VcfWriter,
        global_start: usize,
        output_start: usize,
        output_end: usize,
        phase_conf_valid: bool,
    ) -> Result<Option<ImputationHandoff>> {
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
        let _ = &window_span;

        let n_ref_markers = ref_markers.len();
        let n_target_samples = target_win.n_samples();

        if output_start >= output_end || n_ref_markers == 0 {
            return Ok(None);
        }
        if let Some(bb) = &self.telemetry {
            bb.set_total_markers((output_end - output_start) as u64);
            bb.set_markers_processed(0);
            bb.set_total_samples(n_target_samples as u64);
            bb.set_samples_processed(0);
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

        let mut ref_allele_freqs: Vec<Vec<f32>> = vec![Vec::new(); n_ref_markers];
        let n_ref_haps = ref_columns
            .first()
            .map(|c| c.n_haplotypes())
            .unwrap_or(0);
        for m in 0..n_ref_markers {
            let n_alleles = ref_markers.marker(MarkerIdx::new(m as u32)).n_alleles();
            let mut counts = vec![0u32; n_alleles.max(1)];
            let mut total = 0u32;
            for h in 0..n_ref_haps {
                let a = ref_columns[m].get(HapIdx::new(h as u32));
                if a == 255 {
                    continue;
                }
                let idx = a as usize;
                if idx < counts.len() {
                    counts[idx] += 1;
                    total += 1;
                }
            }
            let mut freqs = vec![0.0f32; counts.len()];
            if total > 0 {
                let inv = 1.0 / total as f32;
                for (i, c) in counts.into_iter().enumerate() {
                    freqs[i] = c as f32 * inv;
                }
            }
            ref_allele_freqs[m] = freqs;
        }

        let marker_map = {
            let chrom = ref_markers
                .marker(MarkerIdx::new(0))
                .chrom;
            if let Some(gen_map) = gen_maps.get(chrom) {
                crate::data::genetic_map::MarkerMap::create(ref_markers, gen_map)
            } else {
                crate::data::genetic_map::MarkerMap::from_positions(ref_markers)
            }
        };
        let gen_positions: Vec<f64> = marker_map.gen_positions().to_vec();
        let mut p_recomb: Vec<f32> = Vec::with_capacity(n_ref_markers);
        p_recomb.push(0.0f32);
        for m in 1..n_ref_markers {
            let dist_cm = (gen_positions[m] - gen_positions[m - 1]).abs();
            p_recomb.push(self.params.p_recomb(dist_cm));
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
        let build_input_probs = |hap_idx: HapIdx, sample_idx: usize| -> TargetAlleleProbs {
            let mut offsets = Vec::with_capacity(n_ref_markers + 1);
            let mut probs: Vec<f32> = Vec::new();
            offsets.push(0);

            for (ref_m, target_m_idx) in alignment.ref_to_target.iter().enumerate() {
                let n_alleles = ref_markers.marker(MarkerIdx::new(ref_m as u32)).n_alleles();
                let mut aligned_probs: Vec<f32> = Vec::new();
                let mut use_probs = false;

                if let Some(target_m_idx) = target_m_idx {
                    let target_m = target_m_idx.as_usize();
                    let conf = target_pl_matrix
                        .sample_confidence_f32(MarkerIdx::new(target_m as u32), sample_idx);
                    let allele = target_win.allele(MarkerIdx::new(target_m as u32), hap_idx);
                    let partner_allele =
                        target_win.allele(MarkerIdx::new(target_m as u32), hap_idx.other());

                    let (mapped_allele, mapped_partner) = if let Some(mapping) = alignment
                        .allele_mappings
                        .get(target_m)
                        .and_then(|m| m.as_ref())
                    {
                        if (allele as usize) < mapping.targ_to_ref.len() {
                            let r = mapping.targ_to_ref[allele as usize];
                            let mapped_allele = if r >= 0 { r as u8 } else { 255 };
                            let mapped_partner =
                                if (partner_allele as usize) < mapping.targ_to_ref.len() {
                                    let rp = mapping.targ_to_ref[partner_allele as usize];
                                    if rp >= 0 { rp as u8 } else { 255 }
                                } else {
                                    255
                                };
                            (mapped_allele, mapped_partner)
                        } else {
                            (255, 255)
                        }
                    } else {
                        (allele, partner_allele)
                    };

                    let is_diploid =
                        target_samples.is_diploid(SampleIdx::new(sample_idx as u32));
                    let has_hard = mapped_allele != 255
                        && (mapped_allele as usize) < n_alleles
                        && (!is_diploid || (mapped_partner != 255 && (mapped_partner as usize) < n_alleles));
                    let mut pl_probs: Vec<f32> = Vec::new();
                    let pl = target_pl_matrix.sample_pl(MarkerIdx::new(target_m as u32), sample_idx);
                    if let Some(pl) = pl {
                        if !pl.is_empty() {
                            let n_pl_alleles = infer_n_alleles_from_pl_len(pl.len()).unwrap_or(0);
                            if n_pl_alleles > 0 {
                                let mapping = alignment
                                    .allele_mappings
                                    .get(target_m)
                                    .and_then(|m| m.as_ref());
                                // Prefer haplotype-conditional allele probabilities when we have
                                // a phased partner allele. If phase is uncertain, mix conditional
                                // posteriors using phase confidence.
                                let mut used_conditional = false;
                                if phase_conf_valid
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
                                                aligned_probs = mapped;
                                                use_probs = true;
                                                used_conditional = true;
                                            }
                                        } else if pl_probs.len() == n_alleles {
                                            aligned_probs = pl_probs.clone();
                                            use_probs = true;
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
                                            aligned_probs = mapped;
                                            use_probs = true;
                                        }
                                    } else if pl_probs.len() == n_alleles {
                                        aligned_probs = pl_probs;
                                        use_probs = true;
                                    }
                                } else if !used_conditional {
                                    let uniform = 1.0 / n_pl_alleles as f32;
                                    let target_priors = vec![uniform; n_pl_alleles];

                                    let partner = target_win
                                        .allele(MarkerIdx::new(target_m as u32), hap_idx.other());
                                    let conf = conf.clamp(0.0, 1.0);
                                    let mut weights = vec![0.0f32; n_pl_alleles];
                                    if partner != 255 && (partner as usize) < n_pl_alleles {
                                        let partner_idx = partner as usize;
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
                                                aligned_probs = mapped;
                                                use_probs = true;
                                            }
                                        } else if pl_probs.len() == n_alleles {
                                            aligned_probs = pl_probs.clone();
                                            use_probs = true;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if !use_probs && has_hard {
                        aligned_probs.resize(n_alleles, 0.0);
                        if is_diploid && mapped_partner != 255 && mapped_partner != mapped_allele {
                            if phase_conf_valid {
                                let phase_conf = target_win
                                    .sample_phase_confidence_f32(
                                        MarkerIdx::new(target_m as u32),
                                        sample_idx,
                                    )
                                    .clamp(0.0, 1.0);
                                aligned_probs[mapped_allele as usize] = phase_conf;
                                aligned_probs[mapped_partner as usize] = 1.0 - phase_conf;
                            } else {
                                aligned_probs[mapped_allele as usize] = 0.5;
                                aligned_probs[mapped_partner as usize] = 0.5;
                            }
                        } else {
                            aligned_probs[mapped_allele as usize] = 1.0;
                        }
                        if conf < 1.0 {
                            let uniform = 1.0 / n_alleles as f32;
                            for p in aligned_probs.iter_mut() {
                                *p = conf * *p + (1.0 - conf) * uniform;
                            }
                        }
                        use_probs = true;
                    }
                }

                if use_probs {
                    probs.extend_from_slice(&aligned_probs);
                } else {
                    let uniform = 1.0 / n_alleles as f32;
                    for _ in 0..n_alleles {
                        probs.push(uniform);
                    }
                }

                offsets.push(probs.len());
            }

            TargetAlleleProbs::new(offsets, probs)
        };

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

        let sample_results: Vec<ImputeResult> = (0..n_target_samples)
            .into_par_iter()
            .map(|s| {
                let h1_idx = HapIdx::new((s * 2) as u32);
                let h2_idx = HapIdx::new((s * 2 + 1) as u32);

                let priors_h1 = imp_overlap
                    .and_then(|o| o.hap_priors())
                    .and_then(|p| p.get(h1_idx.as_usize()));
                let priors_h2 = imp_overlap
                    .and_then(|o| o.hap_priors())
                    .and_then(|p| p.get(h2_idx.as_usize()));

                let mut warned_no_priors = false;
                let mut warned_empty_map = false;
                let mut process_haplotype = |hap_idx: HapIdx,
                                         priors: Option<&HaplotypePriors>|
                 -> (Vec<AllelePosteriors>, HaplotypePriors) {
                    let input_probs = build_input_probs(hap_idx, s);

                    let plan_idx = hap_idx.as_usize();
                    let per_window_cap_local = plan
                        .per_window_caps
                        .get(window_idx)
                        .copied()
                        .unwrap_or(plan.per_window_cap)
                        .max(1);
                    let mut state_haps: Vec<GlobalId> = Vec::new();
                    if plan_idx < plan.window_intervals.len() {
                        for hi in plan.window_intervals[plan_idx].iter() {
                            if hi.contains(window_idx) {
                                state_haps.push(hi.hap);
                                if state_haps.len() >= per_window_cap_local {
                                    break;
                                }
                            }
                        }
                    }
                    if state_haps.is_empty() && plan_idx < plan.core_states.len() {
                        state_haps.extend(plan.core_states[plan_idx].iter().copied());
                    }
                    state_haps.sort_unstable_by_key(|g| g.as_u32());
                    state_haps.dedup();
                    if plan_idx < plan.abyss_mask.len() {
                        let abyss = &plan.abyss_mask[plan_idx];
                        state_haps.retain(|g| !abyss[g.as_usize()]);
                    }
                    if state_haps.is_empty() {
                        // Hard fallback: pick the first non-abyss haplotypes.
                        if plan_idx < plan.abyss_mask.len() {
                            let abyss = &plan.abyss_mask[plan_idx];
                            for h in 0..plan.n_ref_haps {
                                if !abyss.get(h).copied().unwrap_or(true) {
                                    state_haps.push(GlobalId::new(h as u32));
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
                        let prev_states: Vec<GlobalId> =
                            p.ids().iter().map(|id| GlobalId::new(id.0)).collect();
                        let mapper = TransitionMatrix::build(&prev_states, &state_haps);
                        let mapped = mapper.map(p.probs()).into_vec();
                        if mapped.iter().all(|v| !v.is_finite() || *v <= 0.0) && !warned_empty_map {
                            warn!(
                                "State handoff mapped to empty priors for window {} (state set mismatch)",
                                window_idx
                            );
                            warned_empty_map = true;
                        }
                        Some(mapped)
                    });

                    let (posteriors, state_post) = LOCAL_WORKSPACE.with(|cell| {
                        let mut ws_opt = cell.borrow_mut();
                        if ws_opt.is_none() {
                            *ws_opt = Some(ImputeWorkspace::new(state_haps.len(), n_ref_markers));
                        }
                        let ws = ws_opt.as_mut().unwrap();
                        run_impute_hmm(
                            &state_haps,
                            ref_columns,
                            &input_probs,
                            &p_recomb,
                            self.params.p_mismatch,
                            prior_marker_idx,
                            state_priors.as_deref(),
                            plan.n_ref_haps,
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

                    (posteriors, next_priors)
                };

                let (post1_full, p1_out) = process_haplotype(h1_idx, priors_h1);
                let (post2_full, p2_out) = process_haplotype(h2_idx, priors_h2);

                let output_len = output_end.saturating_sub(output_start);
                let mut dosages = Vec::with_capacity(output_len);
                let mut best_gt = Vec::with_capacity(output_len);

                let include_posteriors = self.config.gp || self.config.ap;
                let mut hap1_alt = if !include_posteriors {
                    Some(Vec::with_capacity(output_len))
                } else {
                    None
                };
                let mut hap2_alt = if !include_posteriors {
                    Some(Vec::with_capacity(output_len))
                } else {
                    None
                };
                let mut hap1_posts = if include_posteriors {
                    Some(Vec::with_capacity(output_len))
                } else {
                    None
                };
                let mut hap2_posts = if include_posteriors {
                    Some(Vec::with_capacity(output_len))
                } else {
                    None
                };

                for m in output_start..output_end {
                    let p1 = &post1_full[m];
                    let p2 = &post2_full[m];

                    let (d1, prob1) = match p1 {
                        AllelePosteriors::Biallelic(p) => (*p, *p),
                        AllelePosteriors::Multiallelic(probs) => {
                            let dosage = probs
                                .iter()
                                .enumerate()
                                .map(|(i, p)| i as f32 * p)
                                .sum();
                            let p_alt = if probs.len() > 1 { probs[1] } else { 0.0 };
                            (dosage, p_alt)
                        }
                    };

                    let (d2, prob2) = match p2 {
                        AllelePosteriors::Biallelic(p) => (*p, *p),
                        AllelePosteriors::Multiallelic(probs) => {
                            let dosage = probs
                                .iter()
                                .enumerate()
                                .map(|(i, p)| i as f32 * p)
                                .sum();
                            let p_alt = if probs.len() > 1 { probs[1] } else { 0.0 };
                            (dosage, p_alt)
                        }
                    };

                    let n_alleles = ref_markers
                        .marker(MarkerIdx::new(m as u32))
                        .n_alleles()
                        .max(1);
                    let (gt1, gt2) = if n_alleles <= 2 {
                        let p1_alt = p1.prob(1);
                        let p2_alt = p2.prob(1);
                        let gp00 = (1.0 - p1_alt) * (1.0 - p2_alt);
                        let gp01 = p1_alt * (1.0 - p2_alt) + (1.0 - p1_alt) * p2_alt;
                        let gp11 = p1_alt * p2_alt;
                        if gp01 >= gp00 && gp01 >= gp11 {
                            let p10 = p1_alt * (1.0 - p2_alt);
                            let p01 = (1.0 - p1_alt) * p2_alt;
                            if p10 >= p01 {
                                (1u8, 0u8)
                            } else {
                                (0u8, 1u8)
                            }
                        } else if gp11 >= gp00 {
                            (1u8, 1u8)
                        } else {
                            (0u8, 0u8)
                        }
                    } else {
                        let mut best = (0u8, 0u8);
                        let mut best_prob = -1.0f32;
                        for i in 0..n_alleles {
                            for j in i..n_alleles {
                                let p_i1 = p1.prob(i);
                                let p_i2 = p2.prob(i);
                                let p_j1 = p1.prob(j);
                                let p_j2 = p2.prob(j);
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
                    };

                    best_gt.push((gt1, gt2));
                    dosages.push(d1 + d2);

                    if let Some(v) = hap1_alt.as_mut() {
                        v.push(prob1);
                    }
                    if let Some(v) = hap2_alt.as_mut() {
                        v.push(prob2);
                    }
                    if let Some(v) = hap1_posts.as_mut() {
                        v.push(p1.clone());
                    }
                    if let Some(v) = hap2_posts.as_mut() {
                        v.push(p2.clone());
                    }
                }

                let hap_alt_probs = match (hap1_alt, hap2_alt) {
                    (Some(h1), Some(h2)) => Some((h1, h2)),
                    _ => None,
                };

                let hap_posteriors = match (hap1_posts, hap2_posts) {
                    (Some(h1), Some(h2)) => Some((h1, h2)),
                    _ => None,
                };

                ImputeResult {
                    result: SampleImputationResult {
                        sample_idx: s,
                        dosages,
                        best_gt,
                        hap_alt_probs,
                        hap_posteriors,
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
        self.write_imputed_window_streaming(
            ref_markers,
            ref_genotypes,
            &ref_allele_freqs,
            target_win,
            target_pl,
            alignment,
            final_writer,
            window_quality,
            &ref_is_biallelic,
            output_start,
            output_end,
            output_start,
            &all_results,
            self.config.gp,
            self.config.ap,
        )?;
        if let Some(bb) = &self.telemetry {
            let output_markers = output_end.saturating_sub(output_start);
            bb.set_markers_processed(output_markers as u64);
            bb.set_samples_processed(target_win.n_samples() as u64);
            bb.set_stage(crate::utils::telemetry::Stage::Imputation);
            bb.set_consumer_stage(crate::utils::telemetry::Stage::Imputation);
        }

        Ok(Some(ImputationHandoff {
            priors: next_priors_vec,
            prior_global_idx,
            prior_gen_pos,
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
    fn write_imputed_window_streaming<TargetSpace, RefMarkerSpace, RefPhaseSpace>(
        &self,
        ref_markers: &crate::data::marker::Markers<RefMarkerSpace>,
        ref_genotypes: Option<&GenotypeMatrix<Phased, RefPhaseSpace>>,
        ref_allele_freqs: &[Vec<f32>],
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
        let _ = &write_span;

        let include_posteriors = include_gp || include_ap;
        let _ = ref_allele_freqs.len();
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
        let get_genotype_posteriors = |marker_idx: usize, sample_idx: usize| -> Option<Vec<f32>> {
            let target_m = alignment.target_marker(MarkerIdx::new(marker_idx as u32))?;
            let pl = target_pl.sample_pl(target_m, sample_idx)?;
            if !pl.is_empty() {
                let n_pl_alleles = infer_n_alleles_from_pl_len(pl.len())?;
                if n_pl_alleles == 0 {
                    return None;
                }
                let n_ref_alleles = ref_markers.marker(MarkerIdx::new(marker_idx as u32)).n_alleles();
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
            // Soft fallback using hard GT with an error rate: avoids hard-calling
            // genotyped markers when PLs are missing or uninformative.
            let n_ref_alleles = ref_markers.marker(MarkerIdx::new(marker_idx as u32)).n_alleles();
            if let Some((a1, a2)) = get_genotyped_alleles(marker_idx, sample_idx) {
                let n_genotypes = n_ref_alleles * (n_ref_alleles + 1) / 2;
                if n_genotypes == 0 {
                    return None;
                }
                let mut gp = vec![0.0f32; n_genotypes];
                let idx = genotype_index(a1 as usize, a2 as usize);
                let err = error_rate.clamp(1e-6, 0.5);
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
            let dosage = if self.config.err.is_none() {
                if let Some((a1, a2)) = get_genotyped_alleles(marker_idx, sample_idx) {
                    (a1 + a2) as f32
                } else if let Some(gp) = get_genotype_posteriors(marker_idx, sample_idx) {
                    let n_alleles = ref_markers.marker(MarkerIdx::new(marker_idx as u32)).n_alleles();
                    dosage_from_gp(n_alleles, &gp)
                } else if let Some(result) = result_by_sample.get(sample_idx).and_then(|r| *r) {
                    result.dosages.get(local_m).copied().unwrap_or(0.0)
                } else {
                    0.0
                }
            } else {
                if let Some(gp) = get_genotype_posteriors(marker_idx, sample_idx) {
                    let n_alleles = ref_markers.marker(MarkerIdx::new(marker_idx as u32)).n_alleles();
                    dosage_from_gp(n_alleles, &gp)
                } else if let Some(result) = result_by_sample.get(sample_idx).and_then(|r| *r) {
                    result.dosages.get(local_m).copied().unwrap_or(0.0)
                } else if let Some((a1, a2)) = get_genotyped_alleles(marker_idx, sample_idx) {
                    (a1 + a2) as f32
                } else {
                    0.0
                }
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
            if self.config.err.is_none() {
                if let Some((a1, a2)) = get_genotyped_alleles(marker_idx, sample_idx) {
                    (a1, a2)
                } else if let Some(gp) = get_genotype_posteriors(marker_idx, sample_idx) {
                    let n_alleles = ref_markers.marker(MarkerIdx::new(marker_idx as u32)).n_alleles();
                    best_gt_from_gp(n_alleles, &gp)
                } else if let Some(result) = result_by_sample.get(sample_idx).and_then(|r| *r) {
                    result.best_gt.get(local_m).copied().unwrap_or((0, 0))
                } else {
                    (0, 0)
                }
            } else {
                if let Some(gp) = get_genotype_posteriors(marker_idx, sample_idx) {
                    let n_alleles = ref_markers.marker(MarkerIdx::new(marker_idx as u32)).n_alleles();
                    best_gt_from_gp(n_alleles, &gp)
                } else if let Some(result) = result_by_sample.get(sample_idx).and_then(|r| *r) {
                    result.best_gt.get(local_m).copied().unwrap_or((0, 0))
                } else if let Some((a1, a2)) = get_genotyped_alleles(marker_idx, sample_idx) {
                    (a1, a2)
                } else {
                    (0, 0)
                }
            }
        };

        let get_hap_probs = |marker_idx: usize, sample_idx: usize| -> (f32, f32) {
            let local_m = marker_idx.saturating_sub(output_start);
            let mut result_probs = None;
            if let Some(result) = result_by_sample.get(sample_idx).and_then(|r| *r) {
                if let Some((p1, p2)) = result.hap_posteriors.as_ref() {
                    let v1 = p1.get(local_m).map(|p| p.prob(1)).unwrap_or(0.0);
                    let v2 = p2.get(local_m).map(|p| p.prob(1)).unwrap_or(0.0);
                    result_probs = Some((v1, v2));
                } else if let Some((p1, p2)) = result.hap_alt_probs.as_ref() {
                    let v1 = p1.get(local_m).copied().unwrap_or(0.0);
                    let v2 = p2.get(local_m).copied().unwrap_or(0.0);
                    result_probs = Some((v1, v2));
                }
            }

            if self.config.err.is_none() {
                if let Some((a1, a2)) = get_genotyped_alleles(marker_idx, sample_idx) {
                    // Map allele to prob: if allele > 0, it's 1.0 (alt). If 0, it's 0.0 (ref).
                    // Wait, hap_probs are for the alt allele (1).
                    // Multiallelic? Hap probs usually track "alt mass".
                    let v1 = if a1 > 0 && a1 != 255 { 1.0 } else { 0.0 };
                    let v2 = if a2 > 0 && a2 != 255 { 1.0 } else { 0.0 };
                    (v1, v2)
                } else if let Some((v1, v2)) = result_probs {
                    (v1, v2)
                } else {
                    (0.0, 0.0)
                }
            } else {
                if let Some((v1, v2)) = result_probs {
                    (v1, v2)
                } else {
                    (0.0, 0.0)
                }
            }
        };

        if include_posteriors {
            for marker_idx in markers_to_process_start..output_end {
                if marker_idx >= ref_is_biallelic.len() || !ref_is_biallelic[marker_idx] {
                    continue;
                }
                if let Some(stats) = quality.get_mut(marker_idx) {
                    for s in 0..n_samples {
                        let (mut v1, mut v2) = get_hap_probs(marker_idx, s);
                        if !stats.is_imputed {
                            if self.config.err.is_none() {
                                // If error correction is OFF, force hard calls for stats too
                                if let Some((a1, a2)) = get_genotyped_alleles(marker_idx, s) {
                                    v1 = if a1 > 0 && a1 != 255 { 1.0 } else { 0.0 };
                                    v2 = if a2 > 0 && a2 != 255 { 1.0 } else { 0.0 };
                                }
                            }
                        }
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
                        stats.add_sample_biallelic(v1, v2);
                    }
                }
            }
        }

        let marker_matrix;
        let marker_matrix_ref = {
            if let Some(ref_gt) = ref_genotypes {
                if ref_gt.n_markers() != ref_markers.len() {
                    warn!(
                        ref_markers = ref_markers.len(),
                        ref_genotypes = ref_gt.n_markers(),
                        "Reference genotypes length mismatch; using marker-only matrix for output"
                    );
                }
            }
            let samples = target_win.samples_arc();
            let columns: Vec<crate::data::storage::GenotypeColumn> = (0..ref_markers.len())
                .map(|_| crate::data::storage::GenotypeColumn::default())
                .collect();
            marker_matrix = GenotypeMatrix::new_phased(ref_markers.clone(), columns, samples);
            &marker_matrix
        };

        let get_genotype_posteriors_for_writer =
            if include_gp { Some(|m, s| get_genotype_posteriors(m, s)) } else { None };

        writer.write_imputed_streaming(
            marker_matrix_ref,
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
        let short_ref_markers = build_markers(chrom, &[10]);
        let target_markers = build_markers(chrom, &[10]);

        let ref_genotypes = build_unphased_matrix(short_ref_markers, 1).into_phased();
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
        let dosages = vec![0.0; output_end - output_start];
        let best_gt = vec![(0, 0); output_end - output_start];
        let all_results = vec![SampleImputationResult {
            sample_idx: 0,
            dosages,
            best_gt,
            hap_alt_probs: None,
            hap_posteriors: None,
        }];

        let tmp = NamedTempFile::new().expect("temp vcf");
        let mut writer = VcfWriter::create(tmp.path(), target_win.samples_arc()).expect("writer");

        let pipeline = ImputationPipeline::new(Config::default(), None);
        let ref_is_biallelic = vec![true; ref_markers.len()];
        let ref_allele_freqs = vec![vec![0.5, 0.5]; ref_markers.len()];

        let result = pipeline.write_imputed_window_streaming(
            &ref_markers,
            Some(&ref_genotypes),
            &ref_allele_freqs,
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
        );
        assert!(result.is_ok());
    }
}
