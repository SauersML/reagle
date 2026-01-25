//! Streaming Imputation Pipeline
//!
//! Implements memory-efficient streaming imputation through overlapping windows.
//! Uses a producer-consumer model with MPSC channel to pipe phased matrices
//! directly to imputation in-memory.

use std::io::BufRead;
use std::path::Path;
use std::sync::{Arc, mpsc};
use std::thread;

use rayon::prelude::*;
use sysinfo::System;
use tracing::{info_span, instrument, warn};

use crate::data::alignment::MarkerAlignment;
use crate::data::genetic_map::GeneticMaps;
use crate::data::storage::GenotypeMatrix;
use crate::data::storage::phase_state::{Phased, Unphased};
use crate::data::{HapIdx, MarkerIdx, SampleIdx};
use crate::error::ReagleError;
use crate::error::Result;
use crate::io::bref3::{RefPanelReader, TargetMarkerIndex};
use crate::io::streaming::{
    GlobalHapId, HaplotypePriors, PhasedOverlap, StreamWindow, StreamingConfig, StreamingVcfReader,
};
use crate::io::vcf::{ImputationQuality, VcfWriter};
use crate::model::block_hash::ReferenceMap;
use crate::model::block_hash::hmm::TargetAlleleProbs;
use crate::model::parameters::ModelParams;
use crate::model::pbwt::PbwtState;
use crate::model::pbwt_streaming::PbwtWavefront;
use crate::model::pl_emission::{
    allele_probs_cond_from_pl, allele_probs_uncond_from_pl, infer_n_alleles_from_pl_len,
};
use crate::model::reference_pbwt::{RankBeam, ReferencePbwt};
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

#[derive(Clone, Copy, Debug)]
enum ImputeRefMode {
    Lossless,
    PbwtSubset,
    Truncated,
}

#[derive(Clone, Copy, Debug)]
struct AutoImputePlan {
    mode: ImputeRefMode,
    max_states: usize,
}

const PBWT_SELECT_BLOCK_CM: f64 = 0.05;
const PBWT_MIN_PER_HAP: usize = 8;
const PBWT_MAX_PER_HAP: usize = 256;
const IMPUTE_RAM_FRACTION: f64 = 0.25;

fn available_memory_bytes() -> Option<u64> {
    let mut sys = System::new();
    sys.refresh_memory();
    let avail_kb = sys.available_memory();
    if avail_kb == 0 {
        None
    } else {
        Some(avail_kb.saturating_mul(1024))
    }
}

fn estimate_max_states(
    available_bytes: u64,
    n_threads: usize,
    window_markers: usize,
    block_size: usize,
) -> usize {
    if available_bytes == 0 || n_threads == 0 || window_markers == 0 || block_size == 0 {
        return 0;
    }
    let n_blocks = (window_markers + block_size - 1) / block_size;
    let per_state_bytes = 4usize.saturating_mul(3 + n_blocks + window_markers);
    if per_state_bytes == 0 {
        return 0;
    }
    let budget = (available_bytes as f64 * IMPUTE_RAM_FRACTION) as u64;
    let per_thread = budget / n_threads.max(1) as u64;
    (per_thread as usize) / per_state_bytes
}

fn partition_markers_by_cm(gen_positions: &[f64], block_cm: f64) -> Vec<(usize, usize)> {
    if gen_positions.is_empty() {
        return Vec::new();
    }
    let mut blocks = Vec::new();
    let mut start = 0usize;
    let mut start_pos = gen_positions[0];
    for (i, &pos) in gen_positions.iter().enumerate() {
        if pos - start_pos >= block_cm {
            blocks.push((start, i));
            start = i;
            start_pos = pos;
        }
    }
    if start < gen_positions.len() {
        blocks.push((start, gen_positions.len()));
    }
    blocks
}

fn pbwt_select_keep_mask(
    target_gt: &GenotypeMatrix<Phased>,
    ref_gt: &GenotypeMatrix<Phased>,
    alignment: &MarkerAlignment,
    gen_maps: &GeneticMaps,
    max_states: usize,
) -> Option<Vec<bool>> {
    let n_markers = target_gt.n_markers();
    let n_ref_haps = ref_gt.n_haplotypes();
    let n_target_haps = target_gt.n_haplotypes();
    if n_markers == 0 || n_ref_haps == 0 || n_target_haps == 0 {
        return None;
    }
    if max_states == 0 || max_states >= n_ref_haps {
        return None;
    }

    let k_per_hap = (max_states / n_target_haps.max(1))
        .clamp(PBWT_MIN_PER_HAP, PBWT_MAX_PER_HAP)
        .min(max_states)
        .max(1);

    let markers = target_gt.markers();
    let mut gen_positions = Vec::with_capacity(n_markers);
    for m in 0..n_markers {
        let marker = markers.marker(MarkerIdx::new(m as u32));
        let gen_pos = gen_maps.gen_pos(marker.chrom, marker.pos);
        gen_positions.push(gen_pos);
    }

    let donor_blocks = partition_markers_by_cm(&gen_positions, PBWT_SELECT_BLOCK_CM);
    let mut sampling_points = vec![false; n_markers];
    if donor_blocks.is_empty() {
        sampling_points.fill(true);
    } else {
        for &(s, e) in &donor_blocks {
            if s < n_markers {
                sampling_points[s] = true;
            }
            if e > 0 {
                sampling_points[e - 1] = true;
            }
        }
        sampling_points[n_markers - 1] = true;
    }

    let mut counts = vec![0u32; n_ref_haps];
    let mut pbwt_fwd = ReferencePbwt::new(n_ref_haps);
    let mut beams_fwd: Vec<RankBeam> = (0..n_target_haps)
        .map(|_| RankBeam::full(n_ref_haps as u32))
        .collect();
    let mut ref_alleles = vec![0u8; n_ref_haps];
    let mut query_alleles = vec![0u8; n_target_haps];

    for m in 0..n_markers {
        for h in 0..n_target_haps {
            query_alleles[h] = target_gt.allele(MarkerIdx::new(m as u32), HapIdx::new(h as u32));
        }
        if let Some(ref_m) = alignment.target_to_ref(m) {
            for rh in 0..n_ref_haps {
                let ref_a = ref_gt.allele(MarkerIdx::new(ref_m as u32), HapIdx::new(rh as u32));
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

        if sampling_points[m] {
            for h in 0..n_target_haps {
                let donors = pbwt_fwd.select_donors(&beams_fwd[h], k_per_hap);
                for d in donors {
                    let idx = d as usize;
                    if idx < counts.len() {
                        counts[idx] = counts[idx].saturating_add(1);
                    }
                }
            }
        }
    }

    let mut pbwt_bwd = ReferencePbwt::new(n_ref_haps);
    let mut beams_bwd: Vec<RankBeam> = (0..n_target_haps)
        .map(|_| RankBeam::full(n_ref_haps as u32))
        .collect();
    for (rev_step, m) in (0..n_markers).rev().enumerate() {
        for h in 0..n_target_haps {
            query_alleles[h] = target_gt.allele(MarkerIdx::new(m as u32), HapIdx::new(h as u32));
        }
        if let Some(ref_m) = alignment.target_to_ref(m) {
            for rh in 0..n_ref_haps {
                let ref_a = ref_gt.allele(MarkerIdx::new(ref_m as u32), HapIdx::new(rh as u32));
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

        if sampling_points[m] {
            for h in 0..n_target_haps {
                let donors = pbwt_bwd.select_donors(&beams_bwd[h], k_per_hap);
                for d in donors {
                    let idx = d as usize;
                    if idx < counts.len() {
                        counts[idx] = counts[idx].saturating_add(1);
                    }
                }
            }
        }
    }

    let mut ranked: Vec<(usize, u32)> = counts
        .iter()
        .enumerate()
        .filter(|&(_, &c)| c > 0)
        .map(|(i, &c)| (i, c))
        .collect();
    if ranked.is_empty() {
        return None;
    }
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let keep_n = max_states.min(ranked.len());
    let mut keep_mask = vec![false; n_ref_haps];
    for i in 0..keep_n {
        keep_mask[ranked[i].0] = true;
    }
    Some(keep_mask)
}

fn build_reference_map_with_mask(
    ref_gt: &GenotypeMatrix<Phased>,
    markers: &crate::data::marker::Markers,
    gen_maps: &GeneticMaps,
    params: &ModelParams,
    block_size: usize,
    max_states: usize,
    keep_mask: Option<&[bool]>,
) -> Arc<ReferenceMap> {
    let n_markers = markers.len();
    let mut gen_positions = Vec::with_capacity(n_markers);
    for m in 0..n_markers {
        let marker = markers.marker(MarkerIdx::new(m as u32));
        let gen_pos = gen_maps.gen_pos(marker.chrom, marker.pos);
        gen_positions.push(gen_pos);
    }

    let mut blocks: Vec<Arc<crate::model::block_hash::CompressedBlock>> = Vec::new();
    let mut start = 0usize;
    while start < n_markers {
        let end = (start + block_size).min(n_markers);
        let mut block_markers: Vec<crate::data::marker::Marker> = Vec::with_capacity(end - start);
        let mut block_columns: Vec<crate::data::storage::GenotypeColumn> =
            Vec::with_capacity(end - start);
        for m in start..end {
            block_markers.push(markers.marker(MarkerIdx::new(m as u32)).clone());
            block_columns.push(ref_gt.column(MarkerIdx::new(m as u32)).clone());
        }
        let mut recomb_rates = Vec::with_capacity(end.saturating_sub(start + 1));
        for i in start..end.saturating_sub(1) {
            let dist_cm = (gen_positions[i + 1] - gen_positions[i]).abs();
            recomb_rates.push(params.p_recomb(dist_cm));
        }
        let block =
            crate::model::block_hash::compression::build_compressed_block_from_columns_with_mask(
                &block_markers,
                &block_columns,
                start,
                max_states,
                &recomb_rates,
                keep_mask,
            );
        blocks.push(Arc::new(block));
        start = end;
    }

    let mut boundary_rates = Vec::with_capacity(blocks.len().saturating_sub(1));
    for i in 0..blocks.len().saturating_sub(1) {
        let left_idx = blocks[i].end_marker - 1;
        let right_idx = blocks[i + 1].start_marker;
        let dist_cm = (gen_positions[right_idx] - gen_positions[left_idx]).abs();
        boundary_rates.push(params.p_recomb(dist_cm));
    }

    ReferenceMap::build_from_blocks(blocks, &boundary_rates, block_size)
}

fn should_stream_ref_vcf(path: &Path) -> Option<u64> {
    let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    if file_size == 0 {
        return None;
    }
    Some(file_size / 100)
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

/// Payload passed from Phasing (Producer) to Imputation (Consumer)
struct StreamingPayload {
    phased_target: GenotypeMatrix<Phased>,
    ref_markers: crate::data::marker::Markers,
    ref_map: Arc<ReferenceMap>,
    ref_genotypes: Option<GenotypeMatrix<Phased>>,
    alignment: MarkerAlignment,
    output_start: usize,
    output_end: usize,
    window_idx: usize,
    /// Reference window global marker offset (for coordinate translation)
    ref_global_start: usize,
    /// Reference window output range start (where to start output)
    ref_output_start: usize,
    /// Reference window output range end
    ref_output_end: usize,
}

struct SampleImputationResult {
    sample_idx: usize,
    dosages: Vec<f32>,
    best_gt: Vec<(u8, u8)>,
    hap_alt_probs: Option<(Vec<f32>, Vec<f32>)>,
    hap_posteriors: Option<(Vec<AllelePosteriors>, Vec<AllelePosteriors>)>,
}

impl crate::pipelines::ImputationPipeline {
    /// Run streaming imputation pipeline
    #[instrument(name = "imputation_streaming", skip(self))]
    pub fn run_streaming(&mut self) -> Result<()> {
        // Configure streaming windows
        let streaming_config = StreamingConfig {
            window_cm: self.config.window,
            overlap_cm: self.config.overlap,
            buffer_cm: 1.0,
            max_markers: self.config.window_markers,
        };

        // Collect target positions first so we can load the correct chromosomes from the map.
        let (target_positions_map, target_marker_count) = collect_target_positions(&self.config.gt)?;
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

        // Load genetic maps for the chromosomes actually present in the target.
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

        // Open streaming target reader
        let target_reader =
            StreamingVcfReader::open(&self.config.gt, gen_maps.clone(), streaming_config.clone())?;
        let target_bytes = std::fs::metadata(&self.config.gt)
            .map(|m| m.len())
            .unwrap_or(0);
        let target_samples = target_reader.samples_arc();
        let n_target_samples = target_samples.len();
        let n_target_haps = n_target_samples * 2;
        if let Some(bb) = &self.telemetry {
            bb.set_total_samples(n_target_samples as u64);
            bb.set_samples_processed(0);
        }

        // Load reference panel
        let ref_path = self
            .config
            .r#ref
            .as_ref()
            .ok_or_else(|| ReagleError::config("Reference panel required for imputation"))?;

        let is_bref3 = ref_path.extension().map(|e| e == "bref3").unwrap_or(false);
        // Note: RefPanelReader is not cloneable, so we load it inside the producer thread
        // We need to pass the path
        let ref_path_clone = ref_path.clone();

        // Initialize parameters
        // We load reference size estimate or just guess?
        // Ideally we need n_ref_haps for params.
        // We can open it briefly or just trust config.
        // Let's open it briefly to get N.
        let n_ref_haps = if is_bref3 {
            crate::io::bref3::StreamingBref3Reader::open(&ref_path)?.n_haps()
        } else {
            let (reader, _) = crate::io::vcf::VcfReader::open(&ref_path)?;
            reader.samples_arc().len() * 2
        };

        let n_total_haps = n_ref_haps + n_target_haps;
        self.params =
            ModelParams::for_imputation(n_ref_haps, n_total_haps, self.config.ne, self.config.err);

        let n_threads = self.config.nthreads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        });
        let available_bytes = available_memory_bytes().unwrap_or(0);
        let window_markers = self.config.window_markers.max(1);
        let block_size = 64usize;
        let max_states_budget = if available_bytes > 0 {
            estimate_max_states(available_bytes, n_threads, window_markers, block_size)
        } else {
            self.config.imp_states.max(1)
        };
        let mode = if max_states_budget >= n_ref_haps {
            ImputeRefMode::Lossless
        } else if is_bref3 {
            ImputeRefMode::Truncated
        } else {
            ImputeRefMode::PbwtSubset
        };
        let max_states = if matches!(mode, ImputeRefMode::Lossless) {
            n_ref_haps
        } else {
            max_states_budget.min(n_ref_haps).max(1)
        };
        let auto_plan = AutoImputePlan { mode, max_states };
        self.params.set_n_states(max_states);

        eprintln!(
            "Streaming imputation: {} ref haplotypes, {} target samples (target_bytes={})",
            n_ref_haps, n_target_samples, target_bytes
        );
        let n_blocks = (window_markers + block_size - 1) / block_size;
        let per_state_bytes = 4usize.saturating_mul(3 + n_blocks + window_markers);
        let per_thread_budget = if n_threads > 0 {
            (available_bytes as usize / n_threads).saturating_div(1024 * 1024)
        } else {
            0
        };
        eprintln!(
            "Imputation state plan: mode={:?}, max_states={}, ref_haps={}, threads={}, available_mb={}, per_thread_mb={}, est_bytes_per_state={}",
            auto_plan.mode,
            auto_plan.max_states,
            n_ref_haps,
            n_threads,
            (available_bytes / (1024 * 1024)),
            per_thread_budget,
            per_state_bytes
        );

        let ref_block_size = block_size;
        let ref_max_states = match auto_plan.mode {
            ImputeRefMode::Lossless => 0,
            ImputeRefMode::PbwtSubset | ImputeRefMode::Truncated => auto_plan.max_states,
        };

        // Create output writer
        let output_path = self.config.out.with_extension("vcf.gz");
        eprintln!("Writing output to {:?}", output_path);
        let mut writer = VcfWriter::create(&output_path, target_samples.clone())?;

        // Channel for streaming data
        // Keep the buffer small to avoid holding multiple large windows in memory.
        let (tx, rx) = mpsc::sync_channel::<StreamingPayload>(2);
        if let Some(bb) = &self.telemetry {
            bb.set_channel_capacity(2);
        }

        // Clone config/params for producer
        let producer_config = self.config.clone();
        let producer_params = self.params.clone();
        let producer_maps = gen_maps.clone();
        let producer_telemetry = self.telemetry.clone();
        let producer_target_positions = target_positions.clone();

        let auto_plan_for_producer = auto_plan;

        // Spawn Producer (Phasing)
        let producer_handle = thread::spawn(move || -> Result<()> {
            let pipeline = crate::pipelines::ImputationPipeline {
                config: producer_config,
                params: producer_params,
                telemetry: producer_telemetry,
            };

            // Re-open readers in thread
            let mut target_reader = StreamingVcfReader::open(
                &pipeline.config.gt,
                producer_maps.clone(),
                streaming_config.clone(),
            )?;

            let mut ref_reader: RefPanelReader = if is_bref3 {
                let stream_reader = crate::io::bref3::StreamingBref3Reader::open(&ref_path_clone)?;
                let windowed = crate::io::bref3::StreamingBref3WindowReader::new(stream_reader);
                RefPanelReader::Bref3(windowed)
            } else {
                if let Some(estimated_markers) = should_stream_ref_vcf(&ref_path_clone) {
                    eprintln!(
                        "Using streaming VCF reference reader (~{} markers)",
                        estimated_markers
                    );
                }
                RefPanelReader::StreamingVcf(crate::io::bref3::StreamingRefVcfReader::open(
                    &ref_path_clone,
                )?)
            };

            let mut window_count = 0;
            let mut phased_overlap: Option<PhasedOverlap> = None;
            let mut pbwt_state: Option<PbwtState> = None;

            eprintln!("Phase 1: Streaming phasing of target data...");

            loop {
                let ref_window = if pipeline.config.profile {
                    let span_guard = info_span!("io_read_ref_window").entered();
                    let _ = &span_guard;
                    ref_reader.next_window(
                        &streaming_config,
                        &producer_maps,
                        &pipeline.params,
                        ref_block_size,
                        ref_max_states,
                        producer_target_positions.as_deref(),
                    )?
                } else {
                    ref_reader.next_window(
                        &streaming_config,
                        &producer_maps,
                        &pipeline.params,
                        ref_block_size,
                        ref_max_states,
                        producer_target_positions.as_deref(),
                    )?
                };
                let ref_window = match ref_window {
                    Some(window) => window,
                    None => break,
                };
                window_count += 1;
                let n_ref_markers = ref_window.markers.len();
                let ref_chrom_idx = ref_window.markers.marker(MarkerIdx::new(0)).chrom;
                let ref_chrom = ref_window
                    .markers
                    .chrom_name(ref_chrom_idx)
                    .ok_or_else(|| anyhow::anyhow!("Invalid reference chromosome index"))?;
                let chrom_candidates = chrom_variants(ref_chrom);
                let start_pos = ref_window.markers.marker(MarkerIdx::new(0)).pos;
                let end_pos = ref_window
                    .markers
                    .marker(MarkerIdx::new((n_ref_markers - 1) as u32))
                    .pos;

                let target_window = if pipeline.config.profile {
                    let span_guard = info_span!("io_read_target_region").entered();
                    let _ = &span_guard;
                    target_reader.load_window_for_region(&chrom_candidates, start_pos, end_pos)?
                } else {
                    target_reader.load_window_for_region(&chrom_candidates, start_pos, end_pos)?
                };

                let target_window = if let Some(window) = target_window {
                    window
                } else {
                    let samples = target_reader.samples_arc();
                    let markers = crate::data::marker::Markers::new();
                    let columns: Vec<crate::data::storage::GenotypeColumn> = Vec::new();
                    let genotypes = GenotypeMatrix::new_unphased(markers, columns, samples);
                    StreamWindow {
                        genotypes,
                        global_start: 0,
                        global_end: 0,
                        output_start: 0,
                        output_end: 0,
                        is_first: window_count == 1,
                        phased_overlap: None,
                    }
                };
                if let Some(bb) = &pipeline.telemetry {
                    bb.set_current_window(window_count as u64);
                    if ref_window.is_last {
                        bb.set_total_windows(window_count as u64);
                    }
                    bb.set_total_samples(target_window.genotypes.n_samples() as u64);
                    bb.set_samples_processed(0);
                    bb.set_total_markers(n_ref_markers as u64);
                    bb.set_markers_processed(0);
                    bb.set_total_iterations(0);
                    bb.set_current_iteration(0);
                }
                let phase_span = if pipeline.config.profile {
                    Some(
                        info_span!(
                            "phasing_window",
                            window = window_count,
                            markers = target_window.genotypes.n_markers(),
                            start_pos = start_pos,
                            end_pos = end_pos
                        )
                        .entered(),
                    )
                } else {
                    None
                };
                let _ = &phase_span;

                // Only log major windows to reduce spam (100+ markers, first/last, or every 1000th)
                let should_log = target_window.genotypes.n_markers() >= 100
                    || ref_window.is_first
                    || ref_window.is_last
                    || window_count % 1000 == 0;

                if should_log {
                    eprintln!(
                        "  Phasing Window {} ({} markers, pos {}..{})",
                        window_count,
                        target_window.genotypes.n_markers(),
                        start_pos,
                        end_pos
                    );
                }

                // Use RefWindow metadata for coordinate tracking and boundary handling
                if ref_window.is_first {
                    eprintln!("    (First reference window)");
                }
                if ref_window.is_last {
                    eprintln!("    (Last reference window)");
                }
                let ref_global_start = ref_window.global_start;
                // ref_global_end used implicitly via ref markers length
                let ref_output_start = ref_window.output_start;
                let ref_output_end = ref_window.output_end;
                // Only log ref markers for significant windows
                if should_log {
                    eprintln!(
                        "    Ref markers: {} (global {}..{})",
                        ref_window.markers.len(),
                        ref_global_start,
                        ref_window.global_end
                    );
                }
                if should_log && window_count == 1 {
                    let mut min_patterns = usize::MAX;
                    let mut max_patterns = 0usize;
                    let mut sum_patterns = 0usize;
                    let n_blocks = ref_window.ref_map.blocks.len();
                    for block in ref_window.ref_map.blocks.iter() {
                        let n = block.n_patterns();
                        min_patterns = min_patterns.min(n);
                        max_patterns = max_patterns.max(n);
                        sum_patterns += n;
                    }
                    let avg_patterns = if n_blocks > 0 {
                        sum_patterns as f64 / n_blocks as f64
                    } else {
                        0.0
                    };
                    if min_patterns == usize::MAX {
                        min_patterns = 0;
                    }
                    eprintln!(
                        "    Compression: blocks={}, patterns min/avg/max={}/{:.1}/{}",
                        n_blocks, min_patterns, avg_patterns, max_patterns
                    );
                }
                let alignment = MarkerAlignment::new_with_ref_markers(
                    &target_window.genotypes,
                    &ref_window.markers,
                );
                let phasing_alignment = ref_window
                    .ref_genotypes
                    .as_ref()
                    .map(|ref_gt| MarkerAlignment::new(&target_window.genotypes, ref_gt));

                if window_count == 1 {
                    let mut counts = [0usize; 5];
                    for (t_idx, mapping_opt) in alignment.allele_mappings.iter().enumerate() {
                        if let Some(mapping) = mapping_opt {
                            let is_partial = mapping.targ_to_ref.iter().any(|&x| x < 0);
                            if is_partial {
                                counts[4] += 1;
                            } else if mapping.alleles_swapped && mapping.strand_flipped {
                                counts[3] += 1;
                            } else if mapping.strand_flipped {
                                counts[2] += 1;
                            } else if mapping.alleles_swapped {
                                counts[1] += 1;
                            } else {
                                counts[0] += 1;
                            }

                            if is_partial && counts[4] <= 5 {
                                let t_marker = target_window
                                    .genotypes
                                    .marker(crate::data::MarkerIdx::new(t_idx as u32));
                                let r_idx = alignment.target_to_ref[t_idx];
                                let r_marker = ref_window
                                    .markers
                                    .marker(crate::data::MarkerIdx::new(r_idx as u32));
                                eprintln!(
                                    "[ALIGN-WARN] Partial match at pos {}: Target={:?}/{:?} Ref={:?}/{:?}",
                                    t_marker.pos,
                                    t_marker.ref_allele,
                                    t_marker.alt_alleles,
                                    r_marker.ref_allele,
                                    r_marker.alt_alleles
                                );
                            }
                        }
                    }
                    eprintln!("\n[ALIGNMENT DIAGNOSTICS] Window 1");
                    eprintln!("  Direct Exact Matches: {}", counts[0]);
                    eprintln!("  Swapped (Ref/Alt):    {}", counts[1]);
                    eprintln!("  Strand Flipped:       {}", counts[2]);
                    eprintln!("  Both:                 {}", counts[3]);
                    eprintln!("  Partial/Mismatching:  {}", counts[4]);
                }

                if let Some(bb) = &pipeline.telemetry {
                    bb.set_op(&format!("Phasing window {}", window_count));
                }
                let n_target_markers = target_window.genotypes.n_markers();
                let phased = if n_target_markers == 0 {
                    target_window.genotypes.clone().into_phased()
                } else if target_reader.was_all_phased() {
                    target_window.genotypes.clone().into_phased()
                } else {
                    let phase_guard = if pipeline.config.profile {
                        Some(info_span!("compute_phasing").entered())
                    } else {
                        None
                    };
                    let _ = &phase_guard;
                    match ref_window.ref_genotypes.as_ref() {
                        Some(ref_gt) => {
                            let phase_alignment = phasing_alignment.as_ref().unwrap_or(&alignment);
                            pipeline.phase_window_streaming(
                                &target_window.genotypes,
                                Some(ref_gt),
                                phase_alignment,
                                &producer_maps,
                                phased_overlap.as_ref(),
                                pbwt_state.as_ref(),
                            )?
                        }
                        None => {
                            if window_count == 1 {
                                eprintln!(
                                    "  Warning: Reference-guided phasing disabled (no overlapping ref markers)"
                                );
                            }
                            pipeline.phase_window_streaming(
                                &target_window.genotypes,
                                None,
                                &alignment,
                                &producer_maps,
                                phased_overlap.as_ref(),
                                pbwt_state.as_ref(),
                            )?
                        }
                    }
                };
                if let Some(bb) = &pipeline.telemetry {
                    bb.set_samples_processed(target_window.genotypes.n_samples() as u64);
                    bb.set_markers_processed(target_window.genotypes.n_markers() as u64);
                }

                // Extract state for next window BEFORE moving phased to channel
                phased_overlap = Some(pipeline.extract_overlap_streaming(
                    &phased,
                    n_target_markers,
                    target_window.output_end,
                ));
                pbwt_state = Some(pipeline.extractpbwt_state_streaming(&phased, n_target_markers));

                let mut ref_map = ref_window.ref_map;
                if matches!(auto_plan_for_producer.mode, ImputeRefMode::PbwtSubset) {
                    if let Some(ref_gt) = ref_window.ref_genotypes.as_ref() {
                        if let Some(keep_mask) = pbwt_select_keep_mask(
                            &phased,
                            ref_gt,
                            &alignment,
                            &producer_maps,
                            auto_plan_for_producer.max_states,
                        ) {
                            let kept = keep_mask.iter().filter(|&&k| k).count();
                            let total = keep_mask.len();
                            eprintln!(
                                "  PBWT selection: kept {} / {} ref haplotypes (max_states={})",
                                kept, total, auto_plan_for_producer.max_states
                            );
                            ref_map = build_reference_map_with_mask(
                                ref_gt,
                                &ref_window.markers,
                                &producer_maps,
                                &pipeline.params,
                                ref_block_size,
                                0,
                                Some(&keep_mask),
                            );
                        } else {
                            ref_map = build_reference_map_with_mask(
                                ref_gt,
                                &ref_window.markers,
                                &producer_maps,
                                &pipeline.params,
                                ref_block_size,
                                ref_max_states,
                                None,
                            );
                        }
                    } else if window_count == 1 {
                        eprintln!(
                            "  Warning: PBWT selection unavailable without reference genotypes; falling back to truncation"
                        );
                    }
                }

                // Send to consumer
                if let Some(bb) = &pipeline.telemetry {
                    bb.set_op("Producer waiting on channel");
                }
                let send_result = if pipeline.config.profile {
                    let span_guard = info_span!("channel_send_wait").entered();
                    let _ = &span_guard;
                    tx.send(StreamingPayload {
                        phased_target: phased,
                        ref_markers: ref_window.markers,
                        ref_map,
                        ref_genotypes: ref_window.ref_genotypes,
                        alignment,
                        output_start: target_window.output_start,
                        output_end: target_window.output_end,
                        window_idx: window_count,
                        ref_global_start,
                        ref_output_start,
                        ref_output_end,
                    })
                } else {
                    tx.send(StreamingPayload {
                        phased_target: phased,
                        ref_markers: ref_window.markers,
                        ref_map,
                        ref_genotypes: ref_window.ref_genotypes,
                        alignment,
                        output_start: target_window.output_start,
                        output_end: target_window.output_end,
                        window_idx: window_count,
                        ref_global_start,
                        ref_output_start,
                        ref_output_end,
                    })
                };
                if let Ok(()) = send_result {
                    if let Some(bb) = &pipeline.telemetry {
                        bb.inc_channel_depth();
                        bb.set_op("Producer processing");
                    }
                } else {
                    break; // Consumer hung up
                }
            }
            if window_count == 0 {
                let target_samples = target_reader.samples_arc().len();
                let target_size = std::fs::metadata(&pipeline.config.gt)
                    .map(|m| m.len())
                    .unwrap_or(0);
                return Err(ReagleError::vcf(format!(
                    "No target markers read; check input VCF GT field and chromosome naming. \
target_samples={} target_bytes={}",
                    target_samples, target_size
                )));
            }

            Ok(())
        });

        // Consumer (Imputation)
        eprintln!("Phase 2: Streaming imputation...");

        let mut imp_overlap: Option<PhasedOverlap> = None;
        let mut total_markers = 0;
        let mut header_written = false;

        for payload in rx {
            if let Some(bb) = &self.telemetry {
                bb.dec_channel_depth();
                bb.set_stage(crate::utils::telemetry::Stage::Imputation);
                bb.set_current_window(payload.window_idx as u64);
                bb.set_total_samples(payload.phased_target.n_samples() as u64);
                bb.set_samples_processed(0);
                bb.set_markers_processed(0);
                // Clear stale iteration data from phasing phase
                bb.set_total_iterations(0);
                bb.set_current_iteration(0);
                bb.set_op(&format!("Imputing window {}", payload.window_idx));
            }
            let StreamingPayload {
                phased_target,
                ref_markers,
                ref_map,
                ref_genotypes,
                alignment,
                output_start,
                output_end,
                window_idx,
                ref_global_start,
                ref_output_start,
                ref_output_end,
            } = payload;
            let _ = (output_start, output_end);

            if !header_written {
                writer.write_header_extended(&ref_markers, true, self.config.gp, self.config.ap)?;
                header_written = true;
            }
            // Only log major windows to reduce spam (100+ markers or every 1000th)
            let should_log = phased_target.n_markers() >= 100 || window_idx % 1000 == 0;

            if should_log {
                eprintln!(
                    "  Imputing Window {} ({} markers, ref global {}..{}, output {}..{})",
                    window_idx,
                    phased_target.n_markers(),
                    ref_global_start,
                    ref_global_start + ref_markers.len(),
                    ref_output_start,
                    ref_output_end
                );
            }

            // Initialize quality for this window
            let n_alleles_per_marker: Vec<usize> = (0..ref_markers.len())
                .map(|m| {
                    let marker = ref_markers.marker(MarkerIdx::new(m as u32));
                    1 + marker.alt_alleles.len()
                })
                .collect();
            let mut window_quality = ImputationQuality::new(&n_alleles_per_marker);

            // Mark imputed markers
            for (ref_m, &target_idx) in alignment.ref_to_target.iter().enumerate() {
                if target_idx < 0 {
                    window_quality.set_imputed(ref_m, true);
                } else {
                    window_quality.set_imputed(ref_m, false);
                }
            }

            // Check if we have haplotype priors from previous window for soft-information handoff
            if should_log {
                if let Some(ref overlap) = imp_overlap {
                    if let Some(priors) = overlap.hap_priors() {
                        let n_with_priors = priors.iter().filter(|p| !p.is_empty()).count();
                        if n_with_priors > 0 {
                            eprintln!(
                                "    Using {} haplotypes with soft-information priors",
                                n_with_priors
                            );
                        }
                    }
                }
            }

            let window_span = if self.config.profile {
                Some(
                    info_span!(
                        "imputation_window",
                        window = window_idx,
                        ref_markers = ref_markers.len(),
                        target_markers = phased_target.n_markers(),
                        output_start = ref_output_start,
                        output_end = ref_output_end,
                        n_states = self.params.n_states
                    )
                    .entered(),
                )
            } else {
                None
            };
            let _ = &window_span;

            let next_priors = if self.config.profile {
                let span_guard = info_span!("compute_imputation", window = window_idx).entered();
                let _ = &span_guard;
                self.run_imputation_window_streaming(
                    &phased_target,
                    &ref_markers,
                    &ref_map,
                    ref_genotypes.as_ref(),
                    &alignment,
                    imp_overlap.as_ref(),
                    &mut window_quality,
                    &mut writer,
                    window_idx,
                    ref_output_start,
                    ref_output_end,
                )?
            } else {
                self.run_imputation_window_streaming(
                    &phased_target,
                    &ref_markers,
                    &ref_map,
                    ref_genotypes.as_ref(),
                    &alignment,
                    imp_overlap.as_ref(),
                    &mut window_quality,
                    &mut writer,
                    window_idx,
                    ref_output_start,
                    ref_output_end,
                )?
            };

            total_markers += ref_output_end.saturating_sub(ref_output_start);

            let mut next_overlap = self.extract_imputed_overlap_streaming(
                &phased_target,
                &ref_markers,
                &alignment,
                ref_output_end,
            );
            if let Some(priors) = next_priors {
                next_overlap.set_hap_priors(priors);
            }
            let n_ref_markers = ref_markers.len();
            let overlap_size = 1000.min(n_ref_markers);
            let prior_marker_idx = ref_output_end
                .saturating_sub(overlap_size)
                .saturating_sub(1);
            let boundary_recomb_rate = if prior_marker_idx + 1 < n_ref_markers {
                let marker_idx = prior_marker_idx + 1;
                let block_idx = ref_map
                    .blocks
                    .partition_point(|b| b.end_marker <= marker_idx);
                if block_idx >= ref_map.blocks.len() {
                    0.0
                } else {
                    let block = &ref_map.blocks[block_idx];
                    let local_idx = marker_idx.saturating_sub(block.start_marker);
                    if local_idx == 0 {
                        if block_idx == 0 {
                            0.0
                        } else {
                            ref_map.boundary_rates[block_idx - 1]
                        }
                    } else {
                        block.local_recomb_rates[local_idx - 1]
                    }
                }
            } else {
                0.0
            };
            next_overlap.set_incoming_recomb_rate(boundary_recomb_rate);
            imp_overlap = Some(next_overlap);
        }

        writer.flush()?;

        // Check producer result
        match producer_handle.join() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(e),
            Err(e) => std::panic::resume_unwind(e),
        }

        if total_markers == 0 {
            return Err(ReagleError::vcf(
                "No markers imputed; check reference/target overlap and region selection.",
            ));
        }

        eprintln!("Streaming imputation complete: {} markers", total_markers);
        Ok(())
    }

    fn phase_window_streaming(
        &self,
        target_gt: &GenotypeMatrix<Unphased>,
        ref_gt: Option<&GenotypeMatrix<Phased>>,
        alignment: &MarkerAlignment,
        gen_maps: &GeneticMaps,
        phased_overlap: Option<&PhasedOverlap>,
        pbwt_state: Option<&PbwtState>,
    ) -> Result<GenotypeMatrix<Phased>> {
        let mut phasing =
            crate::pipelines::PhasingPipeline::new(self.config.clone(), self.telemetry.clone());
        if let Some(ref_gt) = ref_gt {
            let ref_gt_arc = Arc::new(ref_gt.clone());
            phasing.set_reference(ref_gt_arc, alignment.clone());
        }
        phasing.phase_window_with_pbwt_handoff(target_gt, gen_maps, phased_overlap, pbwt_state)
    }

    fn extract_overlap_streaming(
        &self,
        phased: &GenotypeMatrix<Phased>,
        n_markers: usize,
        output_end: usize,
    ) -> PhasedOverlap {
        use crate::io::streaming::HaplotypePriors;

        let overlap_size = 1000.min(n_markers);
        let start = output_end.saturating_sub(overlap_size);
        let end = output_end;
        let n_haps = phased.n_haplotypes();
        let mut alleles = vec![255u8; overlap_size * n_haps];
        for h in 0..n_haps {
            for (local_m, global_m) in (start..end).enumerate() {
                alleles[h * overlap_size + local_m] =
                    phased.allele(MarkerIdx::new(global_m as u32), HapIdx::new(h as u32));
            }
        }
        let mut overlap = PhasedOverlap::new(overlap_size, n_haps, alleles);

        // Initialize haplotype priors with empty maps
        // Each target haplotype gets its own priors map (populated by HMM when state probs are available)
        let n_target_haps = phased.n_haplotypes();
        let hap_priors: Vec<HaplotypePriors> = (0..n_target_haps)
            .map(|_| HaplotypePriors::empty())
            .collect();
        overlap.set_hap_priors(hap_priors);

        overlap
    }

    fn extractpbwt_state_streaming(
        &self,
        phased: &GenotypeMatrix<Phased>,
        n_markers: usize,
    ) -> PbwtState {
        let n_haps = phased.n_haplotypes();
        if n_markers == 0 || n_haps == 0 {
            return PbwtState::new(vec![], vec![], 0);
        }
        let mut wavefront = PbwtWavefront::new(n_haps, n_markers);
        for m in 0..n_markers {
            let alleles: Vec<u8> = (0..n_haps)
                .map(|h| phased.allele(MarkerIdx::new(m as u32), HapIdx::new(h as u32)))
                .collect();
            wavefront.advance_forward(&alleles, 2);
        }
        wavefront.get_state()
    }

    fn run_imputation_window_streaming(
        &self,
        target_win: &GenotypeMatrix<Phased>,
        ref_markers: &crate::data::marker::Markers,
        ref_map: &Arc<ReferenceMap>,
        ref_genotypes: Option<&GenotypeMatrix<Phased>>,
        alignment: &MarkerAlignment,
        imp_overlap: Option<&PhasedOverlap>,
        window_quality: &mut ImputationQuality,
        final_writer: &mut VcfWriter,
        window_idx: usize,
        output_start: usize,
        output_end: usize,
    ) -> Result<Option<Vec<HaplotypePriors>>> {
        use crate::model::block_hash::BlockHmmWorkspace;

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

        // Thread-local workspace - must be defined inside the parallel context
        thread_local! {
            static LOCAL_WORKSPACE: std::cell::RefCell<Option<BlockHmmWorkspace>> =
                std::cell::RefCell::new(None);
        }

        let n_ref_markers = ref_markers.len();
        let n_target_samples = target_win.n_samples();

        let markers_to_process = output_start..n_ref_markers;

        if markers_to_process.start >= markers_to_process.end {
            return Ok(None);
        }
        if let Some(bb) = &self.telemetry {
            bb.set_total_markers(markers_to_process.len() as u64);
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
        for block in ref_map.blocks.iter() {
            let window_size = block.window_size();
            for marker_in_window in 0..window_size {
                let ref_m = block.start_marker + marker_in_window;
                if ref_m >= n_ref_markers {
                    continue;
                }
                let n_alleles = ref_markers.marker(MarkerIdx::new(ref_m as u32)).n_alleles();
                let freqs = &mut ref_allele_freqs[ref_m];
                if freqs.is_empty() {
                    freqs.resize(n_alleles, 0.0);
                }
                for pattern_idx in 0..block.n_patterns() {
                    let ref_allele =
                        block.unpacked_alleles[pattern_idx * window_size + marker_in_window];
                    if ref_allele != 255 && (ref_allele as usize) < n_alleles {
                        freqs[ref_allele as usize] += block.pattern_counts[pattern_idx];
                    }
                }
                if block.reservoir_count > 0 {
                    for allele in 0..n_alleles {
                        freqs[allele] += block.reservoir_freq(marker_in_window, allele as u8)
                            * block.reservoir_count as f32;
                    }
                }
            }
        }

        for (m, freqs) in ref_allele_freqs.iter_mut().enumerate() {
            let n_alleles = ref_markers.marker(MarkerIdx::new(m as u32)).n_alleles();
            if freqs.is_empty() {
                freqs.resize(n_alleles, 1.0 / n_alleles as f32);
                continue;
            }
            let mut sum = 0.0f32;
            for f in freqs.iter_mut() {
                if *f < 0.0 {
                    *f = 0.0;
                }
                sum += *f;
            }
            if sum > 0.0 {
                for f in freqs.iter_mut() {
                    *f /= sum;
                }
            } else {
                freqs.fill(1.0 / n_alleles as f32);
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

        let build_input_probs = |hap_idx: HapIdx, sample_idx: usize| -> TargetAlleleProbs {
            let mut offsets = Vec::with_capacity(n_ref_markers + 1);
            let mut probs: Vec<f32> = Vec::new();
            offsets.push(0);

            for (ref_m, &target_m_idx) in alignment.ref_to_target.iter().enumerate() {
                let n_alleles = ref_markers.marker(MarkerIdx::new(ref_m as u32)).n_alleles();
                let mut aligned_probs: Vec<f32> = Vec::new();
                let mut use_probs = false;

                if target_m_idx >= 0 {
                    let target_m = target_m_idx as usize;
                    let conf = target_win
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

                    let mut pl_probs: Vec<f32> = Vec::new();
                    let pl = target_win.sample_pl(MarkerIdx::new(target_m as u32), sample_idx);
                    if let Some(pl) = pl {
                        if !pl.is_empty() {
                            let mut base_probs = ref_allele_freqs[ref_m].clone();
                            if base_probs.len() != n_alleles {
                                base_probs.resize(n_alleles, 1.0 / n_alleles as f32);
                            }
                            let n_pl_alleles = infer_n_alleles_from_pl_len(pl.len()).unwrap_or(0);
                            if n_pl_alleles > 0 {
                                let mapping = alignment
                                    .allele_mappings
                                    .get(target_m)
                                    .and_then(|m| m.as_ref());
                                let mut target_priors = vec![0.0f32; n_pl_alleles];
                                if let Some(mapping) = mapping {
                                    for t in 0..n_pl_alleles {
                                        if t < mapping.targ_to_ref.len() {
                                            let r = mapping.targ_to_ref[t];
                                            if r >= 0 && (r as usize) < base_probs.len() {
                                                target_priors[t] = base_probs[r as usize];
                                            }
                                        }
                                    }
                                } else if n_pl_alleles == base_probs.len() {
                                    target_priors.copy_from_slice(&base_probs);
                                }
                                if !normalize_probs(&mut target_priors) {
                                    let uniform = 1.0 / n_pl_alleles as f32;
                                    target_priors.fill(uniform);
                                }

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
                                        let uniform = (1.0 - conf) / (n_pl_alleles as f32 - 1.0);
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
                                        Some(&base_probs),
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
                                            aligned_probs = mapped;
                                            use_probs = true;
                                        }
                                    } else if pl_probs.len() == n_alleles {
                                        aligned_probs = pl_probs.clone();
                                        use_probs = true;
                                    }
                                } else if allele_probs_uncond_from_pl(
                                    pl,
                                    Some(&base_probs),
                                    &mut pl_probs,
                                )
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
                                            aligned_probs = mapped;
                                            use_probs = true;
                                        }
                                    } else if pl_probs.len() == n_alleles {
                                        aligned_probs = pl_probs;
                                        use_probs = true;
                                    }
                                }
                            }
                        }
                    }

                    if mapped_allele != 255 && (mapped_allele as usize) < n_alleles && !use_probs {
                        aligned_probs.resize(n_alleles, 1.0 / n_alleles as f32);
                        let is_het = mapped_partner != 255
                            && (mapped_partner as usize) < n_alleles
                            && mapped_partner != mapped_allele;
                        let phase_conf = if is_het {
                            target_win.sample_phase_confidence_f32(
                                MarkerIdx::new(target_m as u32),
                                sample_idx,
                            )
                        } else {
                            1.0
                        };
                        for a in 0..n_alleles {
                            let hard = if is_het {
                                if a == mapped_allele as usize {
                                    phase_conf
                                } else if a == mapped_partner as usize {
                                    1.0 - phase_conf
                                } else {
                                    0.0
                                }
                            } else if a == mapped_allele as usize {
                                1.0
                            } else {
                                0.0
                            };
                            aligned_probs[a] = conf * hard + (1.0 - conf) * aligned_probs[a];
                        }
                        normalize_probs(&mut aligned_probs);
                        use_probs = true;
                    }
                }

                if use_probs {
                    probs.extend_from_slice(&aligned_probs);
                }

                offsets.push(probs.len());
            }

            TargetAlleleProbs::new(offsets, probs)
        };

        // Container for next priors (populated in parallel)
        // Since we can't easily merge vectors from par_iter, we'll collect results including priors
        struct ImputeResult {
            result: SampleImputationResult,
            priors: Option<(HaplotypePriors, HaplotypePriors)>,
        }

        let overlap_size = 1000.min(n_ref_markers);
        let prior_marker_idx = output_end.saturating_sub(overlap_size).saturating_sub(1);
        let sample_results: Vec<ImputeResult> = (0..n_target_samples)
            .into_par_iter()
            .map(|s| {
                let h1_idx = HapIdx::new((s * 2) as u32);
                let h2_idx = HapIdx::new((s * 2 + 1) as u32);

                // Get incoming priors if available
                let priors_h1 = imp_overlap
                    .and_then(|o| o.hap_priors())
                    .and_then(|p| p.get(h1_idx.as_usize()));
                let priors_h2 = imp_overlap
                    .and_then(|o| o.hap_priors())
                    .and_then(|p| p.get(h2_idx.as_usize()));

                LOCAL_WORKSPACE.with(|cell| {
                    let mut ws_opt = cell.borrow_mut();

                    // Check if workspace needs resizing
                    // Use max_observed_states instead of configured max_states to prevent thrashing
                    let needs_resize = if let Some(ws) = ws_opt.as_ref() {
                        ws.checkpoints.len() < ref_map.blocks.len()
                            || ws.max_states < ref_map.max_observed_states
                            || ws.fwd_history.len()
                                < (ref_map.max_observed_states + 1) * ref_map.window_size
                    } else {
                        true
                    };

                    if needs_resize {
                        *ws_opt = Some(ref_map.create_workspace());
                    }
                    let ws = ws_opt.as_mut().unwrap();

                    let mut process_haplotype =
                        |hap_idx: HapIdx,
                         priors: Option<&HaplotypePriors>|
                         -> (Vec<AllelePosteriors>, HaplotypePriors) {
                            let input_probs = build_input_probs(hap_idx, s);

                            // Initialize workspace
                            if let Some(first_block) = ref_map.blocks.first() {
                                if let Some(p) = priors {
                                    // Initialize from priors
                                    ws.fwd.fill(0.0);
                                    ws.reservoir_prob_fwd = 0.0;
                                    let mut total_mass = 0.0;

                                    for (&global_id, &prob) in p.ids().iter().zip(p.probs().iter())
                                    {
                                        let pid = first_block.pattern_for_haplotype(
                                            crate::model::block_hash::types::GlobalId::new(
                                                global_id.0,
                                            ),
                                        );
                                        if pid.is_reservoir() {
                                            ws.reservoir_prob_fwd += prob;
                                        } else {
                                            ws.fwd[pid.as_usize()] += prob;
                                        }
                                        total_mass += prob;
                                    }

                                    // Fill remaining mass with uniform?
                                    if total_mass < 0.999 {
                                        let remaining = (1.0f32 - total_mass).max(0.0f32);
                                        let uniform = remaining / first_block.n_ref_haps() as f32;

                                        for i in 0..first_block.n_patterns() {
                                            ws.fwd[i] += uniform * first_block.pattern_counts[i];
                                        }
                                        ws.reservoir_prob_fwd +=
                                            uniform * first_block.reservoir_count as f32;
                                    }

                                    ws.normalize_forward(first_block.n_patterns());
                                } else {
                                    ws.reset_from_block(first_block);
                                }
                            }

                            // Run HMM

                            let initial_recomb_rate = imp_overlap
                                .and_then(|o| o.incoming_recomb_rate())
                                .unwrap_or(0.0);
                            ref_map.forward_pass_probs(
                                &input_probs,
                                self.params.p_mismatch,
                                ws,
                                initial_recomb_rate,
                            );

                            let posteriors = ref_map.backward_and_emit_posteriors_probs(
                                &input_probs,
                                self.params.p_mismatch,
                                ws,
                                initial_recomb_rate,
                            );

                            // Extract next priors

                            let mut next_priors = HaplotypePriors::empty();

                            // Determine marker index for priors (start of overlap region - 1)

                            // This corresponds to state before observing the first marker of next window's overlap

                            // We use state at overlap_start - 1 (approximate, misses one transition step but avoids double emission)

                            // Run forward pass up to prior_marker_idx
                            ref_map.forward_to_marker_probs(
                                &input_probs,
                                self.params.p_mismatch,
                                ws,
                                prior_marker_idx,
                                initial_recomb_rate,
                            );

                            // Extract state from ws.fwd (which is now at prior_marker_idx)

                            // Need to find which block this marker belongs to, to access pattern counts

                            let block_idx = ref_map
                                .blocks
                                .partition_point(|b| b.end_marker <= prior_marker_idx);

                            if block_idx < ref_map.blocks.len() {
                                let block = &ref_map.blocks[block_idx];

                                // ws.fwd now contains state after observing prior_marker_idx

                                let fwd = &ws.fwd;

                                let res_prob = ws.reservoir_prob_fwd;

                                let threshold = 1e-4;

                                let mut priors_list: Vec<(GlobalHapId, f32)> = Vec::new();

                                for (pat_idx, &prob) in
                                    fwd.iter().enumerate().take(block.n_patterns())
                                {
                                    if prob > threshold {
                                        let count = block.pattern_counts[pat_idx];

                                        let global_prob = prob / count;

                                        for &global_id in block.pattern_globals(pat_idx) {
                                            priors_list.push((
                                                GlobalHapId(global_id.as_u32()),
                                                global_prob,
                                            ));
                                        }
                                    }
                                }

                                if res_prob > threshold && block.reservoir_count > 0 {
                                    let global_prob = res_prob / block.reservoir_count as f32;

                                    for &global_id in &block.reservoir_globals {
                                        priors_list
                                            .push((GlobalHapId(global_id.as_u32()), global_prob));
                                    }
                                }

                                priors_list.sort_unstable_by_key(|(h, _)| *h);

                                let (hap_ids, probs): (Vec<GlobalHapId>, Vec<f32>) =
                                    priors_list.into_iter().unzip();
                                next_priors = HaplotypePriors::new(hap_ids, probs);
                            }

                            (posteriors, next_priors)
                        };

                    let (post1_full, p1_out) = process_haplotype(h1_idx, priors_h1);

                    let (post2_full, p2_out) = process_haplotype(h2_idx, priors_h2);

                    // Combine results

                    let output_len = output_end.saturating_sub(output_start);

                    let mut dosages = Vec::with_capacity(output_len);

                    let mut best_gt = Vec::with_capacity(output_len);

                    // Optional outputs

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

                        let (d1, g1, prob1) = match p1 {
                            AllelePosteriors::Biallelic(p) => {
                                (*p, if *p > 0.5 { 1 } else { 0 }, *p)
                            }

                            AllelePosteriors::Multiallelic(probs) => {
                                let dosage =
                                    probs.iter().enumerate().map(|(i, p)| i as f32 * p).sum();

                                let (best_allele, _) = probs
                                    .iter()
                                    .enumerate()
                                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                                    .unwrap_or((0, &0.0));

                                let p_alt = if probs.len() > 1 { probs[1] } else { 0.0 };

                                (dosage, best_allele as u8, p_alt)
                            }
                        };

                        let (d2, g2, prob2) = match p2 {
                            AllelePosteriors::Biallelic(p) => {
                                (*p, if *p > 0.5 { 1 } else { 0 }, *p)
                            }

                            AllelePosteriors::Multiallelic(probs) => {
                                let dosage =
                                    probs.iter().enumerate().map(|(i, p)| i as f32 * p).sum();

                                let (best_allele, _) = probs
                                    .iter()
                                    .enumerate()
                                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                                    .unwrap_or((0, &0.0));

                                let p_alt = if probs.len() > 1 { probs[1] } else { 0.0 };

                                (dosage, best_allele as u8, p_alt)
                            }
                        };

                        best_gt.push((g1, g2));

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

        // Sort all results by sample index for writing
        all_results.sort_by_key(|result| result.sample_idx);

        if let Some(bb) = &self.telemetry {
            let output_markers = output_end.saturating_sub(output_start);
            bb.set_stage(crate::utils::telemetry::Stage::WritingOutput);
            bb.set_total_markers(output_markers as u64);
            bb.set_markers_processed(0);
            bb.set_total_samples(target_win.n_samples() as u64);
            bb.set_samples_processed(0);
            bb.set_op(&format!(
                "Writing window {} ({} markers)",
                window_idx, output_markers
            ));
        }
        self.write_imputed_window_streaming(
            ref_markers,
            ref_genotypes,
            target_win,
            alignment,
            final_writer,
            window_quality,
            &ref_is_biallelic,
            output_start,
            output_end,
            markers_to_process.start,
            &all_results,
            self.config.gp,
            self.config.ap,
        )?;
        if let Some(bb) = &self.telemetry {
            let output_markers = output_end.saturating_sub(output_start);
            bb.set_markers_processed(output_markers as u64);
            bb.set_samples_processed(target_win.n_samples() as u64);
            bb.set_stage(crate::utils::telemetry::Stage::Imputation);
        }

        Ok(Some(next_priors_vec))
    }

    fn extract_imputed_overlap_streaming(
        &self,
        target_win: &GenotypeMatrix<Phased>,
        ref_markers: &crate::data::marker::Markers,
        alignment: &MarkerAlignment,
        output_end: usize,
    ) -> PhasedOverlap {
        let overlap_size = 1000.min(ref_markers.len());
        let start = output_end.saturating_sub(overlap_size);
        let end = output_end;
        let n_haps = target_win.n_haplotypes();
        let mut alleles = vec![255u8; overlap_size * n_haps];
        for h in 0..n_haps {
            for (local_m, ref_m) in (start..end).enumerate() {
                if let Some(target_m) = alignment.target_marker(ref_m) {
                    alleles[h * overlap_size + local_m] =
                        target_win.allele(MarkerIdx::new(target_m as u32), HapIdx::new(h as u32));
                }
            }
        }
        PhasedOverlap::new(overlap_size, n_haps, alleles)
    }

    /// Write imputed window results to VCF
    fn write_imputed_window_streaming(
        &self,
        ref_markers: &crate::data::marker::Markers,
        ref_genotypes: Option<&GenotypeMatrix<Phased>>,
        target_win: &GenotypeMatrix<Phased>,
        alignment: &MarkerAlignment,
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

        let get_genotyped_alleles = |marker_idx: usize, sample_idx: usize| -> Option<(u8, u8)> {
            let target_m = alignment.target_marker(marker_idx)?;
            let h1 = HapIdx::new((sample_idx * 2) as u32);
            let h2 = HapIdx::new((sample_idx * 2 + 1) as u32);
            let raw_a1 = target_win.allele(MarkerIdx::new(target_m as u32), h1);
            let raw_a2 = target_win.allele(MarkerIdx::new(target_m as u32), h2);
            if raw_a1 == 255 || raw_a2 == 255 {
                return None;
            }
            let mapping = alignment
                .allele_mappings
                .get(target_m)
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

        // Closure to get dosage: marker_idx is window-local ref marker index from VCF writer
        // Dosages array is indexed from 0 for markers starting at output_start
        let get_dosage = |marker_idx: usize, sample_idx: usize| -> f32 {
            let local_m = marker_idx.saturating_sub(output_start);
            let dosage = if let Some(result) = result_by_sample.get(sample_idx).and_then(|r| *r) {
                result.dosages.get(local_m).copied().unwrap_or(0.0)
            } else if let Some((a1, a2)) = get_genotyped_alleles(marker_idx, sample_idx) {
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
            if let Some((a1, a2)) = get_genotyped_alleles(marker_idx, sample_idx) {
                (a1, a2)
            } else if let Some(result) = result_by_sample.get(sample_idx).and_then(|r| *r) {
                result.best_gt.get(local_m).copied().unwrap_or((0, 0))
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
                        let (mut v1, mut v2) = get_hap_probs(marker_idx, s);
                        if !stats.is_imputed {
                            if let Some(target_m) = alignment.target_marker(marker_idx) {
                                let h1 = HapIdx::new((s * 2) as u32);
                                let h2 = HapIdx::new((s * 2 + 1) as u32);
                                let raw_a1 = target_win.allele(MarkerIdx::new(target_m as u32), h1);
                                let raw_a2 = target_win.allele(MarkerIdx::new(target_m as u32), h2);

                                let mapping = alignment
                                    .allele_mappings
                                    .get(target_m)
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
                                if a1 < 2 && a2 < 2 {
                                    v1 = a1 as f32;
                                    v2 = a2 as f32;
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
                        let (mut v1, mut v2) = get_hap_probs(marker_idx, s);
                        if !stats.is_imputed {
                            if let Some(target_m) = alignment.target_marker(marker_idx) {
                                let h1 = HapIdx::new((s * 2) as u32);
                                let h2 = HapIdx::new((s * 2 + 1) as u32);
                                let raw_a1 = target_win.allele(MarkerIdx::new(target_m as u32), h1);
                                let raw_a2 = target_win.allele(MarkerIdx::new(target_m as u32), h2);

                                let mapping = alignment
                                    .allele_mappings
                                    .get(target_m)
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
                                if a1 < 2 && a2 < 2 {
                                    v1 = a1 as f32;
                                    v2 = a2 as f32;
                                }
                            }
                        }
                        stats.add_sample_biallelic(v1, v2);
                    }
                }
            }
        }

        let marker_matrix;
        let marker_matrix_ref = if let Some(ref_gt) = ref_genotypes {
            if ref_gt.n_markers() == ref_markers.len() {
                ref_gt
            } else {
                warn!(
                    ref_markers = ref_markers.len(),
                    ref_genotypes = ref_gt.n_markers(),
                    "Reference genotypes length mismatch; falling back to marker-only matrix"
                );
                let samples = target_win.samples_arc();
                let columns: Vec<crate::data::storage::GenotypeColumn> = (0..ref_markers.len())
                    .map(|_| crate::data::storage::GenotypeColumn::default())
                    .collect();
                marker_matrix = GenotypeMatrix::new_phased(ref_markers.clone(), columns, samples);
                &marker_matrix
            }
        } else {
            let samples = target_win.samples_arc();
            let columns: Vec<crate::data::storage::GenotypeColumn> = (0..ref_markers.len())
                .map(|_| crate::data::storage::GenotypeColumn::default())
                .collect();
            marker_matrix = GenotypeMatrix::new_phased(ref_markers.clone(), columns, samples);
            &marker_matrix
        };

        writer.write_imputed_streaming(
            marker_matrix_ref,
            get_dosage,
            get_best_gt,
            get_posteriors_for_writer,
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
    use crate::data::marker::{Allele, Marker, Markers};
    use crate::data::storage::GenotypeColumn;
    use crate::io::bref3::StreamingRefVcfReader;
    use crate::io::vcf::{ImputationQuality, VcfWriter};
    use crate::model::parameters::ModelParams;
    use crate::pipelines::ImputationPipeline;
    use std::io::{BufReader, Cursor};
    use tempfile::NamedTempFile;

    fn build_markers(chrom: ChromIdx, positions: &[u32]) -> Markers {
        let mut markers = Markers::new();
        markers.add_chrom("chr1");
        for (idx, &pos) in positions.iter().enumerate() {
            let marker = Marker::new(
                chrom,
                pos,
                Some(format!("m{idx}").into()),
                Allele::Base(b'A'),
                vec![Allele::Base(b'C')],
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
        let params = ModelParams::new();
        let ref_window = ref_reader
            .next_window(&config, &gen_maps, &params, 64, 4096, None)
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
            ref_to_target: vec![-1; ref_markers.len()],
            target_to_ref: vec![0; target_win.n_markers()],
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

        let result = pipeline.write_imputed_window_streaming(
            &ref_markers,
            Some(&ref_genotypes),
            &target_win,
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
