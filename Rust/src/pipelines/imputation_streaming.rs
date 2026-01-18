//! Streaming Imputation Pipeline
//!
//! Implements memory-efficient streaming imputation through overlapping windows.
//! Uses a producer-consumer model with MPSC channel to pipe phased matrices
//! directly to imputation in-memory.

use std::collections::HashSet;
use std::path::Path;
use std::rc::Rc;
use std::sync::{Arc, mpsc};
use std::thread;

use rayon::prelude::*;
use tracing::{info_span, instrument};
use crate::data::genetic_map::GeneticMaps;
use crate::data::haplotype::HapIdx;
use crate::data::marker::MarkerIdx;
use crate::data::storage::phase_state::{Phased, Unphased};
use crate::data::storage::GenotypeMatrix;
use crate::data::alignment::MarkerAlignment;
use crate::error::{Result, ReagleError};
use crate::io::bref3::RefPanelReader;
use crate::io::streaming::{HaplotypePriors, PhasedOverlap, StreamingConfig, StreamingVcfReader, StreamWindow};
use crate::io::vcf::{VcfWriter, ImputationQuality};
use crate::pipelines::imputation::{AllelePosteriors, AllelePosteriorCache, ClusterStateProbs};
use crate::model::imp_utils::*;
use crate::model::parameters::ModelParams;
use crate::model::pbwt_streaming::PbwtWavefront;
use crate::model::pbwt::PbwtState;
use crate::utils::workspace::ImpWorkspace;

const PATTERN_BLOCK_SIZE: usize = 32;

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

#[derive(Clone, Debug)]
struct PatternBlock {
    start: usize,
    end: usize,
    hap_to_pattern: Vec<u32>,
    pattern_alleles: Vec<Vec<u8>>,
}

impl PatternBlock {
    fn block_id(&self) -> usize {
        self.start
    }

    fn hap_to_pattern(&self) -> &[u32] {
        &self.hap_to_pattern
    }

    fn pattern_alleles(&self, marker: usize) -> Option<&[u8]> {
        if marker < self.start || marker >= self.end {
            return None;
        }
        let offset = marker - self.start;
        self.pattern_alleles.get(offset).map(|v| v.as_slice())
    }
}

fn build_pattern_block(ref_win: &GenotypeMatrix<Phased>, start: usize, end: usize) -> PatternBlock {
    use std::collections::HashMap;

    let n_haps = ref_win.n_haplotypes();
    let block_len = end.saturating_sub(start).max(1);
    let mut hap_to_pattern = vec![0u32; n_haps];
    let mut patterns: Vec<Vec<u8>> = Vec::new();
    let mut pattern_map: HashMap<Vec<u8>, u32> = HashMap::new();

    for h in 0..n_haps {
        let hap = HapIdx::new(h as u32);
        let mut seq = Vec::with_capacity(block_len);
        for m in start..end {
            let allele = ref_win.allele(MarkerIdx::new(m as u32), hap);
            seq.push(allele);
        }
        let pat_idx = if let Some(&idx) = pattern_map.get(&seq) {
            idx
        } else {
            let idx = patterns.len() as u32;
            pattern_map.insert(seq.clone(), idx);
            patterns.push(seq);
            idx
        };
        hap_to_pattern[h] = pat_idx;
    }

    let n_patterns = patterns.len();
    let mut pattern_alleles: Vec<Vec<u8>> = Vec::with_capacity(block_len);
    for offset in 0..block_len {
        let mut alleles = Vec::with_capacity(n_patterns);
        for pat in &patterns {
            let allele = pat.get(offset).copied().unwrap_or(255);
            alleles.push(allele);
        }
        pattern_alleles.push(alleles);
    }

    PatternBlock {
        start,
        end,
        hap_to_pattern,
        pattern_alleles,
    }
}

fn should_stream_ref_vcf(path: &Path, window_markers: usize) -> Option<u64> {
    let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    if file_size == 0 {
        return None;
    }

    let estimated_markers = file_size / 100;
    let threshold = std::cmp::min(window_markers as u64, 500_000);
    if estimated_markers > threshold {
        Some(estimated_markers)
    } else {
        None
    }
}

/// Payload passed from Phasing (Producer) to Imputation (Consumer)
struct StreamingPayload {
    phased_target: GenotypeMatrix<Phased>,
    ref_window: GenotypeMatrix<Phased>,
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
    priors: Option<(HaplotypePriors, HaplotypePriors)>,
    state_probs: Option<(Arc<ClusterStateProbs>, Arc<ClusterStateProbs>)>,
    hap_alt_probs: Option<(Vec<f32>, Vec<f32>)>,
}

impl crate::pipelines::ImputationPipeline {
    fn build_pbwt_hap_indices_for_batch(
        &self,
        target_win: &GenotypeMatrix<Phased>,
        ref_win: &GenotypeMatrix<Phased>,
        alignment: &MarkerAlignment,
        ref_is_biallelic: &[bool],
        cluster_bounds: &[(usize, usize)],
        n_states: usize,
        batch_samples: &[usize],
    ) -> Vec<(Vec<Vec<u32>>, Vec<Vec<u32>>)> {
        if batch_samples.is_empty() {
            return Vec::new();
        }

        let n_ref_markers = ref_win.n_markers();
        let n_ref_haps = ref_win.n_haplotypes();
        let n_clusters = cluster_bounds.len();
        let n_batch_haps = batch_samples.len() * 2;
        let n_total_haps = n_ref_haps + n_batch_haps;

        thread_local! {
            static PBWT_WORKSPACE: std::cell::RefCell<Option<(PbwtWavefront, Vec<u8>)>> =
                std::cell::RefCell::new(None);
        }

        let mut hap1_neighbors: Vec<Vec<Vec<u32>>> = vec![vec![Vec::new(); n_clusters]; batch_samples.len()];
        let mut hap2_neighbors: Vec<Vec<Vec<u32>>> = vec![vec![Vec::new(); n_clusters]; batch_samples.len()];

        let add_neighbors = |neighbors: &mut Vec<Vec<Vec<u32>>>, batch_idx: usize, cluster_idx: usize, raw: Vec<u32>| {
            if cluster_idx >= neighbors.len() {
                return;
            }
            let list = &mut neighbors[batch_idx][cluster_idx];
            for h in raw {
                if (h as usize) < n_ref_haps {
                    list.push(h);
                }
            }
        };

        let n_candidates = n_states.min(n_ref_haps);
        let mut batch_haps: Vec<HapIdx> = Vec::with_capacity(n_batch_haps);
        for &s in batch_samples {
            batch_haps.push(HapIdx::new((s * 2) as u32));
            batch_haps.push(HapIdx::new((s * 2 + 1) as u32));
        }
        let batch_positions: Vec<(u32, u32)> = batch_samples
            .iter()
            .enumerate()
            .map(|(i, _)| ((n_ref_haps + i * 2) as u32, (n_ref_haps + i * 2 + 1) as u32))
            .collect();

        let mut pbwt_ref_markers: Vec<usize> = Vec::new();
        for ref_m in 0..n_ref_markers {
            let target_m_idx = alignment.ref_to_target.get(ref_m).copied().unwrap_or(-1);
            if target_m_idx < 0 {
                continue;
            }
            let target_m = target_m_idx as usize;
            let mut any_informative = false;
            for hap in &batch_haps {
                let allele = target_win.allele(MarkerIdx::new(target_m as u32), *hap);
                if allele < 2 {
                    any_informative = true;
                    break;
                }
            }
            if any_informative {
                pbwt_ref_markers.push(ref_m);
            }
        }

        // Multi-point union: query PBWT at left/mid/right typed markers per cluster.
        let mut cluster_query_at_ref = vec![usize::MAX; n_ref_markers];
        if !pbwt_ref_markers.is_empty() {
            for (c, &(start, end)) in cluster_bounds.iter().enumerate() {
                if start >= end || start >= n_ref_markers {
                    continue;
                }
                let mid = (start + end - 1) / 2;
                let start_idx = pbwt_ref_markers.partition_point(|&m| m < start);
                let end_idx = pbwt_ref_markers.partition_point(|&m| m < end);
                if start_idx >= end_idx {
                    continue;
                }

                let pos = pbwt_ref_markers[start_idx..end_idx].partition_point(|&m| m < mid);
                let left_idx = if pos == 0 { start_idx } else { start_idx + pos - 1 };
                let right_idx = if start_idx + pos < end_idx { start_idx + pos } else { left_idx };

                let left_m = pbwt_ref_markers[start_idx];
                let right_m = pbwt_ref_markers[end_idx - 1];
                let mid_m = {
                    let left_mid = pbwt_ref_markers[left_idx];
                    let right_mid = pbwt_ref_markers[right_idx];
                    if (mid as isize - left_mid as isize).abs() <= (right_mid as isize - mid as isize).abs() {
                        left_mid
                    } else {
                        right_mid
                    }
                };

                for &m in &[left_m, mid_m, right_m] {
                    if m < cluster_query_at_ref.len() {
                        cluster_query_at_ref[m] = c;
                    }
                }
            }
        }

        PBWT_WORKSPACE.with(|cell| {
            let mut ws_opt = cell.borrow_mut();
            if ws_opt.is_none() {
                *ws_opt = Some((PbwtWavefront::new(n_total_haps, pbwt_ref_markers.len()), vec![0u8; n_total_haps]));
            }
            let (wavefront, alleles) = ws_opt.as_mut().unwrap();
            if wavefront.n_haps() != n_total_haps || wavefront.n_markers() != pbwt_ref_markers.len() {
                *wavefront = PbwtWavefront::new(n_total_haps, pbwt_ref_markers.len());
                alleles.resize(n_total_haps, 0u8);
            }

            wavefront.reset_forward();
            for &ref_m in &pbwt_ref_markers {
                let ref_marker_idx = MarkerIdx::new(ref_m as u32);
                let target_m_idx = alignment.ref_to_target.get(ref_m).copied().unwrap_or(-1);
                if target_m_idx < 0 {
                    continue;
                }
                let target_m = target_m_idx as usize;

                for h in 0..n_ref_haps {
                    let mut allele = ref_win.allele(ref_marker_idx, HapIdx::new(h as u32));
                    if alignment.has_allele_mapping(target_m) {
                        allele = alignment.reverse_map_allele(target_m, allele);
                    }
                    alleles[h] = allele;
                }

                let mut is_biallelic = ref_is_biallelic
                    .get(ref_m)
                    .copied()
                    .unwrap_or(true);
                for (local_idx, hap_idx) in batch_haps.iter().enumerate() {
                    let allele = target_win.allele(MarkerIdx::new(target_m as u32), *hap_idx);
                    alleles[n_ref_haps + local_idx] = allele;
                    if allele >= 2 {
                        is_biallelic = false;
                    }
                }

                let n_alleles = if is_biallelic { 2 } else { 256 };

                wavefront.advance_forward(alleles, n_alleles);

                if ref_m < cluster_query_at_ref.len() {
                    let cluster_idx = cluster_query_at_ref[ref_m];
                    if cluster_idx != usize::MAX {
                        wavefront.prepare_fwd_queries();
                        for (batch_idx, &(hap1_pos, hap2_pos)) in batch_positions.iter().enumerate() {
                            let h1 = wavefront.find_fwd_neighbors_readonly(hap1_pos, n_candidates);
                            let h2 = wavefront.find_fwd_neighbors_readonly(hap2_pos, n_candidates);
                            add_neighbors(&mut hap1_neighbors, batch_idx, cluster_idx, h1);
                            add_neighbors(&mut hap2_neighbors, batch_idx, cluster_idx, h2);
                        }
                    }
                }
            }

            wavefront.reset_backward();
            for &ref_m in pbwt_ref_markers.iter().rev() {
                let ref_marker_idx = MarkerIdx::new(ref_m as u32);
                let target_m_idx = alignment.ref_to_target.get(ref_m).copied().unwrap_or(-1);
                if target_m_idx < 0 {
                    continue;
                }
                let target_m = target_m_idx as usize;

                for h in 0..n_ref_haps {
                    let mut allele = ref_win.allele(ref_marker_idx, HapIdx::new(h as u32));
                    if alignment.has_allele_mapping(target_m) {
                        allele = alignment.reverse_map_allele(target_m, allele);
                    }
                    alleles[h] = allele;
                }

                let mut is_biallelic = ref_is_biallelic
                    .get(ref_m)
                    .copied()
                    .unwrap_or(true);
                for (local_idx, hap_idx) in batch_haps.iter().enumerate() {
                    let allele = target_win.allele(MarkerIdx::new(target_m as u32), *hap_idx);
                    alleles[n_ref_haps + local_idx] = allele;
                    if allele >= 2 {
                        is_biallelic = false;
                    }
                }

                let n_alleles = if is_biallelic { 2 } else { 256 };

                wavefront.advance_backward(alleles, n_alleles);

                if ref_m < cluster_query_at_ref.len() {
                    let cluster_idx = cluster_query_at_ref[ref_m];
                    if cluster_idx != usize::MAX {
                        wavefront.prepare_bwd_queries();
                        for (batch_idx, &(hap1_pos, hap2_pos)) in batch_positions.iter().enumerate() {
                            let h1 = wavefront.find_bwd_neighbors_readonly(hap1_pos, n_candidates);
                            let h2 = wavefront.find_bwd_neighbors_readonly(hap2_pos, n_candidates);
                            add_neighbors(&mut hap1_neighbors, batch_idx, cluster_idx, h1);
                            add_neighbors(&mut hap2_neighbors, batch_idx, cluster_idx, h2);
                        }
                    }
                }
            }
        });

        let finalize = |neighbors: Vec<Vec<u32>>| -> Vec<Vec<u32>> {
            let mut out = Vec::with_capacity(n_clusters);
            for mut list in neighbors {
                list.sort_unstable();
                list.dedup();
                if list.len() < n_states {
                    let mut seen: HashSet<u32> = list.iter().copied().collect();
                    for h in 0..n_ref_haps as u32 {
                        if seen.insert(h) {
                            list.push(h);
                            if list.len() == n_states {
                                break;
                            }
                        }
                    }
                }
                if list.len() > n_states {
                    list.truncate(n_states);
                }
                out.push(list);
            }
            out
        };

        let mut out = Vec::with_capacity(batch_samples.len());
        for batch_idx in 0..batch_samples.len() {
            let h1 = std::mem::take(&mut hap1_neighbors[batch_idx]);
            let h2 = std::mem::take(&mut hap2_neighbors[batch_idx]);
            out.push((finalize(h1), finalize(h2)));
        }
        out
    }

    /// Run streaming imputation pipeline
    #[instrument(name = "imputation_streaming", skip(self))]
    pub fn run_streaming(&mut self) -> Result<()> {
        // Configure streaming windows
        let streaming_config = StreamingConfig {
            window_cm: self.config.window,
            overlap_cm: self.config.overlap,
            buffer_cm: 1.0,
            max_markers: 100_000,
        };

        // Load genetic maps
        let gen_maps = if let Some(ref map_path) = self.config.map {
            GeneticMaps::from_plink_file(map_path, &[])?
        } else {
            GeneticMaps::new()
        };

        // Open streaming target reader
        let target_reader = StreamingVcfReader::open(
            &self.config.gt,
            gen_maps.clone(),
            streaming_config.clone(),
        )?;
        let target_bytes = std::fs::metadata(&self.config.gt).map(|m| m.len()).unwrap_or(0);
        let target_samples = target_reader.samples_arc();
        let n_target_samples = target_samples.len();
        let n_target_haps = n_target_samples * 2;
        if let Some(bb) = &self.telemetry {
            bb.set_total_samples(n_target_samples as u64);
            bb.set_samples_processed(0);
        }

        // Load reference panel
        let ref_path = self.config.r#ref.as_ref().ok_or_else(|| {
            ReagleError::config("Reference panel required for imputation")
        })?;

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
        self.params = ModelParams::for_imputation(n_ref_haps, n_total_haps, self.config.ne, self.config.err);
        self.params.set_n_states(self.config.imp_states.min(n_ref_haps));

        eprintln!(
            "Streaming imputation: {} ref haplotypes, {} target samples (target_bytes={})",
            n_ref_haps, n_target_samples
            , target_bytes
        );

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
            
            let use_streaming_vcf = should_stream_ref_vcf(&ref_path_clone, pipeline.config.window_markers);
            let mut ref_reader: RefPanelReader = if is_bref3 {
                let stream_reader = crate::io::bref3::StreamingBref3Reader::open(&ref_path_clone)?;
                let windowed = crate::io::bref3::StreamingBref3WindowReader::new(stream_reader);
                RefPanelReader::Bref3(windowed)
            } else if let Some(estimated_markers) = use_streaming_vcf {
                eprintln!(
                    "Auto-detected large reference (~{} markers), using streaming VCF reader",
                    estimated_markers
                );
                // Streaming VCF for memory-constrained environments
                RefPanelReader::StreamingVcf(crate::io::bref3::StreamingRefVcfReader::open(&ref_path_clone)?)
            } else {
                // In-memory VCF (default, safer for correctness)
                let (mut vcf_reader, vcf_file) = crate::io::vcf::VcfReader::open(&ref_path_clone)?;
                let ref_gt = Arc::new(vcf_reader.read_all(vcf_file)?.into_phased());
                RefPanelReader::InMemory(crate::io::bref3::InMemoryRefReader::new(ref_gt))
            };

            let mut window_count = 0;
            let mut phased_overlap: Option<PhasedOverlap> = None;
            let mut pbwt_state: Option<PbwtState> = None;

            eprintln!("Phase 1: Streaming phasing of target data...");
            
            loop {
                let ref_window = if pipeline.config.profile {
                    let span_guard = info_span!("io_read_ref_window").entered();
                    let _ = &span_guard;
                    ref_reader.next_window(&streaming_config, &producer_maps)?
                } else {
                    ref_reader.next_window(&streaming_config, &producer_maps)?
                };
                let ref_window = match ref_window {
                    Some(window) => window,
                    None => break,
                };
                window_count += 1;
                let n_ref_markers = ref_window.genotypes.n_markers();
                let ref_chrom_idx = ref_window.genotypes.marker(MarkerIdx::new(0)).chrom;
                let ref_chrom = ref_window
                    .genotypes
                    .markers()
                    .chrom_name(ref_chrom_idx)
                    .ok_or_else(|| anyhow::anyhow!("Invalid reference chromosome index"))?;
                let chrom_candidates = chrom_variants(ref_chrom);
                let start_pos = ref_window.genotypes.marker(MarkerIdx::new(0)).pos;
                let end_pos = ref_window
                    .genotypes
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

                eprintln!(
                    "  Phasing Window {} ({} markers, pos {}..{})",
                    window_count,
                    target_window.genotypes.n_markers(),
                    start_pos,
                    end_pos
                );

                // Use RefWindow metadata for coordinate tracking and boundary handling
                if ref_window.is_first {
                    eprintln!("    (First reference window)");
                }
                if ref_window.is_last {
                    eprintln!("    (Last reference window)");
                }
                let ref_global_start = ref_window.global_start;
                // ref_global_end used implicitly via ref_window_gt.n_markers()
                let ref_output_start = ref_window.output_start;
                let ref_output_end = ref_window.output_end;
                // Consume global_end to avoid unused warning
                eprintln!("    Ref markers: {} (global {}..{})",
                    ref_window.genotypes.n_markers(), ref_global_start, ref_window.global_end);
                let ref_window_gt = ref_window.genotypes;

                let alignment = MarkerAlignment::new(
                    &target_window.genotypes,
                    &ref_window_gt,
                );

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
                    pipeline.phase_window_streaming(
                        &target_window.genotypes,
                        &ref_window_gt,
                        &alignment,
                        &producer_maps,
                        phased_overlap.as_ref(),
                        pbwt_state.as_ref(),
                    )?
                };
                if let Some(bb) = &pipeline.telemetry {
                    bb.set_samples_processed(target_window.genotypes.n_samples() as u64);
                    bb.set_markers_processed(target_window.genotypes.n_markers() as u64);
                }

                // Extract state for next window BEFORE moving phased to channel
                phased_overlap = Some(
                    pipeline.extract_overlap_streaming(&phased, n_target_markers, target_window.output_end),
                );
                pbwt_state = Some(pipeline.extractpbwt_state_streaming(&phased, n_target_markers));

                // Send to consumer
                if let Some(bb) = &pipeline.telemetry {
                    bb.set_op("Producer waiting on channel");
                }
                let send_result = if pipeline.config.profile {
                    let span_guard = info_span!("channel_send_wait").entered();
                    let _ = &span_guard;
                    tx.send(StreamingPayload {
                        phased_target: phased,
                        ref_window: ref_window_gt,
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
                        ref_window: ref_window_gt,
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
                ref_window,
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
                writer.write_header_extended(
                    ref_window.markers(),
                    true,
                    self.config.gp,
                    self.config.ap,
                )?;
                header_written = true;
            }

            eprintln!(
                "  Imputing Window {} ({} markers, ref global {}..{}, output {}..{})",
                window_idx, phased_target.n_markers(),
                ref_global_start, ref_global_start + ref_window.n_markers(),
                ref_output_start, ref_output_end
            );

            // Initialize quality for this window
            let n_alleles_per_marker: Vec<usize> = (0..ref_window.n_markers())
                .map(|m| {
                    let marker = ref_window.marker(MarkerIdx::new(m as u32));
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
            if let Some(ref overlap) = imp_overlap {
                if let Some(priors) = overlap.hap_priors() {
                    let n_with_priors = priors.iter().filter(|p| !p.is_empty()).count();
                    if n_with_priors > 0 {
                        eprintln!("    Using {} haplotypes with soft-information priors", n_with_priors);
                    }
                }
            }

        let window_span = if self.config.profile {
            Some(
                info_span!(
                    "imputation_window",
                    window = window_idx,
                    ref_markers = ref_window.n_markers(),
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
                    &ref_window,
                    &alignment,
                    &gen_maps,
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
                    &ref_window,
                    &alignment,
                    &gen_maps,
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
                &ref_window,
                &alignment,
                ref_output_end,
            );
            if let Some(priors) = next_priors {
                next_overlap.set_hap_priors(priors);
            }
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
        ref_gt: &GenotypeMatrix<Phased>,
        alignment: &MarkerAlignment,
        gen_maps: &GeneticMaps,
        phased_overlap: Option<&PhasedOverlap>,
        pbwt_state: Option<&PbwtState>,
    ) -> Result<GenotypeMatrix<Phased>> {
        let mut phasing = crate::pipelines::PhasingPipeline::new(
            self.config.clone(),
            self.telemetry.clone(),
        );
        let ref_gt_arc = Arc::new(ref_gt.clone());
        phasing.set_reference(ref_gt_arc, alignment.clone());
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
                alleles[h * overlap_size + local_m] = phased.allele(
                    MarkerIdx::new(global_m as u32),
                    HapIdx::new(h as u32),
                );
            }
        }
        let mut overlap = PhasedOverlap::new(overlap_size, n_haps, alleles);

        // Initialize haplotype priors with empty maps
        // Each target haplotype gets its own priors map (populated by HMM when state probs are available)
        let n_target_haps = phased.n_haplotypes();
        let hap_priors: Vec<HaplotypePriors> = (0..n_target_haps)
            .map(|_| HaplotypePriors::new())
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
        ref_win: &GenotypeMatrix<Phased>,
        alignment: &MarkerAlignment,
        gen_maps: &GeneticMaps,
        imp_overlap: Option<&PhasedOverlap>,
        window_quality: &mut ImputationQuality,
        final_writer: &mut VcfWriter,
        window_idx: usize,
        output_start: usize,
        output_end: usize,
    ) -> Result<Option<Vec<HaplotypePriors>>> {
        let window_span = if self.config.profile {
            Some(
                info_span!(
                    "imputation_window_compute",
                    ref_markers = ref_win.n_markers(),
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
            static LOCAL_WORKSPACE: std::cell::RefCell<Option<ImpWorkspace>> =
                std::cell::RefCell::new(None);
        }

        let n_ref_markers = ref_win.n_markers();
        let n_target_samples = target_win.n_samples();
        let n_ref_haps = ref_win.n_haplotypes();
        let markers_to_process = if let Some(overlap) = imp_overlap {
            let start = overlap.n_markers.max(output_start);
            start..n_ref_markers
        } else {
            output_start..n_ref_markers
        };

        if markers_to_process.start >= markers_to_process.end {
            return Ok(None);
        }
        if let Some(bb) = &self.telemetry {
            bb.set_total_markers(markers_to_process.len() as u64);
            bb.set_markers_processed(0);
            bb.set_total_samples(n_target_samples as u64);
            bb.set_samples_processed(0);
        }

        let chrom = ref_win.marker(MarkerIdx::new(0)).chrom;
        let ref_is_biallelic: Vec<bool> = (0..n_ref_markers)
            .map(|m| ref_win.marker(MarkerIdx::new(m as u32)).alt_alleles.len() == 1)
            .collect();
        let gen_positions: Vec<f64> = (0..n_ref_markers)
            .map(|m| {
                let pos = ref_win.marker(MarkerIdx::new(m as u32)).pos;
                gen_maps.gen_pos(chrom, pos)
            })
            .collect();

        let genotyped_markers: Vec<usize> = alignment
            .ref_to_target
            .iter()
            .enumerate()
            .filter_map(|(ref_m, &target_m)| if target_m >= 0 { Some(ref_m) } else { None })
            .collect();

        let sample_genotyped_vec: Vec<Vec<usize>> = (0..n_target_samples)
            .map(|s| {
                genotyped_markers
                    .iter()
                    .copied()
                    .filter(|&ref_m| {
                        let target_m = alignment.ref_to_target[ref_m] as usize;
                        let marker_idx = MarkerIdx::new(target_m as u32);
                        let a1 = target_win.allele(marker_idx, HapIdx::new((s * 2) as u32));
                        let a2 = target_win.allele(marker_idx, HapIdx::new((s * 2 + 1) as u32));
                        a1 != 255 || a2 != 255
                    })
                    .collect()
            })
            .collect();

        let clusters = compute_marker_clusters(
            &genotyped_markers,
            &gen_positions,
            self.config.cluster as f64,
        );
        let n_clusters = clusters.len();
        let cluster_bounds: Vec<(usize, usize)> = clusters.iter().map(|c| (c.start, c.end)).collect();
        let cluster_midpoints: Vec<usize> = clusters
            .iter()
            .map(|c| {
                let mid = if c.end > c.start {
                    (c.start + c.end - 1) / 2
                } else {
                    c.start
                };
                genotyped_markers.get(mid).copied().unwrap_or(0)
            })
            .collect();
        let cluster_midpoints_pos: Vec<f64> = cluster_midpoints
            .iter()
            .map(|&m| gen_positions[m])
            .collect();
        let cluster_p_recomb: Vec<f32> = if n_clusters == 0 {
            Vec::new()
        } else {
            std::iter::once(0.0f32)
                .chain((1..n_clusters).map(|c| {
                    let gen_dist = (cluster_midpoints_pos[c] - cluster_midpoints_pos[c - 1])
                        .abs()
                        .max(crate::model::imp_utils::MIN_CM_DIST);
                    self.params.p_recomb(gen_dist)
                }))
                .collect()
        };

        let (ref_cluster_start, ref_cluster_end) = compute_ref_cluster_bounds(&genotyped_markers, &clusters);
        let cluster_bounds_ref: Vec<(usize, usize)> = ref_cluster_start
            .iter()
            .zip(ref_cluster_end.iter())
            .map(|(&start, &end)| (start, end))
            .collect();
        let marker_cluster = Arc::new(build_marker_cluster_index(&ref_cluster_start, n_ref_markers));
        let ref_cluster_end: Arc<Vec<usize>> = Arc::new(ref_cluster_end);
        let gen_positions = Arc::new(gen_positions);
        let cluster_midpoints_pos = Arc::new(cluster_midpoints_pos);

        const PBWT_BYTES_PER_HAP: usize = 64;
        let max_haps = self
            .config
            .pbwt_batch_mb
            .saturating_mul(1024 * 1024)
            / PBWT_BYTES_PER_HAP;
        let max_batch_haps = max_haps.saturating_sub(n_ref_haps);
        let max_batch_samples = (max_batch_haps / 2).max(1);
        let batch_size = max_batch_samples.min(n_target_samples).max(1);
        let n_batches = (n_target_samples + batch_size - 1) / batch_size;

        let mut all_results: Vec<SampleImputationResult> = Vec::new();

        let telemetry = self.telemetry.clone();
        for batch_idx in 0..n_batches {
            let batch_start = batch_idx * batch_size;
            let batch_end = (batch_start + batch_size).min(n_target_samples);
            let batch_samples: Vec<usize> = (batch_start..batch_end).collect();

            let pbwt_states = self.params.n_states.min(n_ref_haps);
            let batch_neighbors = {
                let pbwt_span = if self.config.profile {
                    Some(
                        info_span!(
                            "pbwt_neighbor_batch",
                            batch = batch_idx,
                            batch_size = batch_samples.len(),
                            n_states = pbwt_states
                        )
                        .entered(),
                    )
                } else {
                    None
                };
                let _ = &pbwt_span;
                self.build_pbwt_hap_indices_for_batch(
                    target_win,
                    ref_win,
                    alignment,
                    &ref_is_biallelic,
                    &cluster_bounds_ref,
                    pbwt_states,
                    &batch_samples,
                )
            };

            let batch_results: Vec<SampleImputationResult> = batch_samples
                .par_iter()
                .enumerate()
                .map(|(local_idx, &s)| {
                    let lifecycle_span = if local_idx == 0 {
                        Some(info_span!("sample_lifecycle", sample_idx = s).entered())
                    } else {
                        None
                    };
                    let _ = &lifecycle_span;

                    let hap1_idx = HapIdx::new((s * 2) as u32);
                    let hap2_idx = HapIdx::new((s * 2 + 1) as u32);
                    let sample_genotyped = &sample_genotyped_vec[s];

                    if sample_genotyped.is_empty() {
                        let result = SampleImputationResult {
                            sample_idx: s,
                            dosages: Vec::new(),
                            best_gt: Vec::new(),
                            priors: None,
                            state_probs: None,
                            hap_alt_probs: None,
                        };
                        if let Some(bb) = telemetry.as_ref() {
                            bb.add_samples(1);
                        }
                        return result;
                    }

                    let (dosages, best_gt, priors, state_probs_pair, hap_alt_probs) =
                        LOCAL_WORKSPACE.with(|cell| {
                        let mut ws_opt = cell.borrow_mut();
                        if ws_opt.is_none() { *ws_opt = Some(ImpWorkspace::new(n_ref_haps)); }
                        let ws = ws_opt.as_mut().unwrap();
                        ws.clear();

                        let (hap1_indices, hap2_indices) = &batch_neighbors[local_idx];

                        let obs_hap1: Vec<u8> = (0..target_win.n_markers())
                            .map(|m| target_win.allele(MarkerIdx::new(m as u32), hap1_idx))
                            .collect();
                        let obs_hap2: Vec<u8> = (0..target_win.n_markers())
                            .map(|m| target_win.allele(MarkerIdx::new(m as u32), hap2_idx))
                            .collect();

                        let mut hap_priors: Vec<HaplotypePriors> = Vec::with_capacity(2);
                        let mut hap_state_probs: Vec<Arc<ClusterStateProbs>> = Vec::with_capacity(2);

                        let (hap1_probs, h1_locked, hap1_prior) = {
                            let hap_indices = &hap1_indices;
                            let actual_n_states = pbwt_states;

                            let prior_probs = imp_overlap
                                .and_then(|overlap| overlap.hap_priors())
                                .and_then(|priors| priors.get(hap1_idx.as_usize()))
                                .filter(|priors| !priors.is_empty())
                                .map(|priors| {
                                    let mut prior_probs = Vec::new();
                                    if let Some(first_cluster) = hap_indices.get(0) {
                                        for &hap in first_cluster.iter().take(actual_n_states) {
                                            prior_probs.push(priors.prior(hap, actual_n_states));
                                        }
                                    }
                                    prior_probs
                                });

                        let state_probs = compute_state_probs(
                            &hap_indices,
                            &cluster_bounds,
                            &genotyped_markers,
                            target_win,
                            ref_win,
                            alignment,
                            &obs_hap1,
                            &obs_hap2,
                            &obs_hap1,
                            Some(&obs_hap2),
                            s,
                            actual_n_states,
                            ws,
                            self.params.p_mismatch,
                                &cluster_p_recomb,
                                marker_cluster.clone(),
                                ref_cluster_end.clone(),
                                gen_positions.clone(),
                                cluster_midpoints_pos.clone(),
                                self.params.recomb_intensity,
                                prior_probs.as_deref(),
                                local_idx == 0,
                            );

                            let mut posterior_cache = AllelePosteriorCache::default();
                            let mut locked = obs_hap1.clone();
                            for &ref_m in sample_genotyped {
                                if let Some(target_m) = alignment.target_marker(ref_m) {
                                    let marker_idx = MarkerIdx::new(ref_m as u32);
                                    let column = ref_win.column(marker_idx);
                                    let map_ref_to_targ = alignment
                                        .target_marker(ref_m)
                                        .and_then(|target_m| {
                                            alignment
                                                .allele_mappings
                                                .get(target_m)
                                                .and_then(|m| m.as_ref())
                                                .map(|m| m.ref_to_targ.as_slice())
                                        });
                                    let p = state_probs.allele_posteriors_for_column_cached(
                                        ref_m,
                                        2,
                                        column,
                                        map_ref_to_targ,
                                        &mut posterior_cache,
                                    );
                                    locked[target_m] = p.max_allele();
                                }
                            }

                            let priors = if output_end > 0 && n_ref_markers > 0 {
                                let overlap_size = 1000.min(n_ref_markers);
                                let overlap_start = output_end.saturating_sub(overlap_size);
                                let prior_marker = overlap_start.min(n_ref_markers.saturating_sub(1));
                                let (hap_ids, probs) = state_probs.haplotype_priors_at(prior_marker);
                                let gen_pos = gen_maps.gen_pos(
                                    chrom,
                                    ref_win.marker(MarkerIdx::new(prior_marker as u32)).pos,
                                );
                                let mut priors = HaplotypePriors::new();
                                priors.set_from_posteriors(&hap_ids, &probs, gen_pos, window_idx);
                                priors
                            } else {
                                HaplotypePriors::new()
                            };

                            (state_probs, locked, priors)
                        };

                        let (hap2_probs, hap2_prior) = {
                            let hap_indices = &hap2_indices;
                            let actual_n_states = pbwt_states;

                            let prior_probs = imp_overlap
                                .and_then(|overlap| overlap.hap_priors())
                                .and_then(|priors| priors.get(hap2_idx.as_usize()))
                                .filter(|priors| !priors.is_empty())
                                .map(|priors| {
                                    let mut prior_probs = Vec::new();
                                    if let Some(first_cluster) = hap_indices.get(0) {
                                        for &hap in first_cluster.iter().take(actual_n_states) {
                                            prior_probs.push(priors.prior(hap, actual_n_states));
                                        }
                                    }
                                    prior_probs
                                });

                        let state_probs = compute_state_probs(
                            &hap_indices,
                            &cluster_bounds,
                            &genotyped_markers,
                            target_win,
                            ref_win,
                            alignment,
                            &obs_hap1,
                            &obs_hap2,
                            &obs_hap2,
                            Some(&h1_locked),
                            s,
                            actual_n_states,
                            ws,
                            self.params.p_mismatch,
                                &cluster_p_recomb,
                                marker_cluster.clone(),
                                ref_cluster_end.clone(),
                                gen_positions.clone(),
                                cluster_midpoints_pos.clone(),
                                self.params.recomb_intensity,
                                prior_probs.as_deref(),
                                local_idx == 0,
                            );

                            let priors = if output_end > 0 && n_ref_markers > 0 {
                                let overlap_size = 1000.min(n_ref_markers);
                                let overlap_start = output_end.saturating_sub(overlap_size);
                                let prior_marker = overlap_start.min(n_ref_markers.saturating_sub(1));
                                let (hap_ids, probs) = state_probs.haplotype_priors_at(prior_marker);
                                let gen_pos = gen_maps.gen_pos(
                                    chrom,
                                    ref_win.marker(MarkerIdx::new(prior_marker as u32)).pos,
                                );
                                let mut priors = HaplotypePriors::new();
                                priors.set_from_posteriors(&hap_ids, &probs, gen_pos, window_idx);
                                priors
                            } else {
                                HaplotypePriors::new()
                            };

                            (state_probs, priors)
                        };

                        hap_state_probs.push(Arc::clone(&hap1_probs));
                        hap_state_probs.push(Arc::clone(&hap2_probs));

                        hap_priors.push(hap1_prior);
                        hap_priors.push(hap2_prior);

                        let combined_dosages = Vec::new();
                        let combined_best_gt = Vec::new();

                        let priors = if hap_priors.len() == 2 {
                            Some((hap_priors[0].clone(), hap_priors[1].clone()))
                        } else {
                            None
                        };

                        let state_probs_pair = if hap_state_probs.len() == 2 {
                            Some((Arc::clone(&hap_state_probs[0]), Arc::clone(&hap_state_probs[1])))
                        } else {
                            None
                        };

                        let hap_alt_probs = None;

                        (combined_dosages, combined_best_gt, priors, state_probs_pair, hap_alt_probs)
                    });
                    let result = SampleImputationResult {
                        sample_idx: s,
                        dosages,
                        best_gt,
                        priors,
                        state_probs: state_probs_pair,
                        hap_alt_probs,
                    };
                    if let Some(bb) = telemetry.as_ref() {
                        bb.add_samples(1);
                    }
                    result
                }).collect();

            all_results.extend(batch_results);
        }
        
        // Sort all results by sample index for writing
        all_results.sort_by_key(|result| result.sample_idx);


        let mut next_priors = if output_end > 0 && n_ref_markers > 0 {
            Some(vec![HaplotypePriors::new(); n_target_samples * 2])
        } else {
            None
        };

        if let Some(priors) = next_priors.as_mut() {
            for result in &all_results {
                if let Some((p1, p2)) = &result.priors {
                    let base = result.sample_idx * 2;
                    if base + 1 < priors.len() {
                        priors[base] = p1.clone();
                        priors[base + 1] = p2.clone();
                    }
                }
            }
        }

        if let Some(bb) = &self.telemetry {
            let output_markers = output_end.saturating_sub(output_start);
            bb.set_stage(crate::utils::telemetry::Stage::WritingOutput);
            bb.set_total_markers(output_markers as u64);
            bb.set_markers_processed(0);
            bb.set_total_samples(target_win.n_samples() as u64);
            bb.set_samples_processed(0);
            bb.set_op(&format!("Writing window {} ({} markers)", window_idx, output_markers));
        }
        self.write_imputed_window_streaming(
            ref_win,
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
        Ok(next_priors)
    }
    
    fn extract_imputed_overlap_streaming(
        &self,
        target_win: &GenotypeMatrix<Phased>,
        ref_win: &GenotypeMatrix<Phased>,
        alignment: &MarkerAlignment,
        output_end: usize,
    ) -> PhasedOverlap {
        let overlap_size = 1000.min(ref_win.n_markers());
        let start = output_end.saturating_sub(overlap_size);
        let end = output_end;
        let n_haps = target_win.n_haplotypes();
        let mut alleles = vec![255u8; overlap_size * n_haps];
        for h in 0..n_haps {
            for (local_m, ref_m) in (start..end).enumerate() {
                if let Some(target_m) = alignment.target_marker(ref_m) {
                    alleles[h * overlap_size + local_m] = target_win.allele(
                        MarkerIdx::new(target_m as u32),
                        HapIdx::new(h as u32),
                    );
                }
            }
        }
        PhasedOverlap::new(overlap_size, n_haps, alleles)
    }

    /// Write imputed window results to VCF
    fn write_imputed_window_streaming(
        &self,
        ref_win: &GenotypeMatrix<Phased>,
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

        // Build lookup maps for sample -> (dosages, best_gt)
        let sample_data: std::collections::HashMap<usize, (&Vec<f32>, &Vec<(u8, u8)>)> =
            all_results
                .iter()
                .map(|result| (result.sample_idx, (&result.dosages, &result.best_gt)))
                .collect();
        let sample_hap_probs: std::collections::HashMap<usize, (&Vec<f32>, &Vec<f32>)> =
            all_results
                .iter()
                .filter_map(|result| {
                    result
                        .hap_alt_probs
                        .as_ref()
                        .map(|(p1, p2)| (result.sample_idx, (p1, p2)))
                })
                .collect();

        let sample_posteriors: std::collections::HashMap<
            usize,
            (&Arc<ClusterStateProbs>, &Arc<ClusterStateProbs>),
        > = all_results
            .iter()
            .filter_map(|result| {
                result
                    .state_probs
                    .as_ref()
                    .map(|(p1, p2)| (result.sample_idx, (p1, p2)))
            })
            .collect();
        let sample_posteriors = Arc::new(sample_posteriors);

        let include_posteriors = include_gp || include_ap;
        let n_samples = target_win.n_samples();
        struct PatternBlockCache {
            block_idx: usize,
            block: Option<Arc<PatternBlock>>,
        }
        let pattern_cache = std::cell::RefCell::new(PatternBlockCache {
            block_idx: usize::MAX,
            block: None,
        });
        let get_pattern_block = |marker_idx: usize| -> Option<Arc<PatternBlock>> {
            if marker_idx < markers_to_process_start {
                return None;
            }
            let block_idx = (marker_idx - markers_to_process_start) / PATTERN_BLOCK_SIZE;
            let mut cache = pattern_cache.borrow_mut();
            if cache.block_idx != block_idx {
                let block_start = markers_to_process_start + block_idx * PATTERN_BLOCK_SIZE;
                let block_end = (block_start + PATTERN_BLOCK_SIZE).min(output_end);
                cache.block = Some(Arc::new(build_pattern_block(ref_win, block_start, block_end)));
                cache.block_idx = block_idx;
            }
            cache.block.clone()
        };
        struct PosteriorCache {
            marker_idx: usize,
            data: Vec<(AllelePosteriors, AllelePosteriors)>,
        }
        let post_cache = std::cell::RefCell::new(PosteriorCache {
            marker_idx: usize::MAX,
            data: Vec::new(),
        });
        let get_posteriors: Option<
            Rc<dyn Fn(usize, usize) -> (AllelePosteriors, AllelePosteriors) + '_>,
        > = if include_posteriors {
            let sample_posteriors = Arc::clone(&sample_posteriors);
            Some(Rc::new(move |marker_idx, sample_idx| {
                let mut cache = post_cache.borrow_mut();
                if cache.marker_idx != marker_idx {
                    cache.marker_idx = marker_idx;
                    let marker = ref_win.marker(MarkerIdx::new(marker_idx as u32));
                    let n_alleles = 1 + marker.alt_alleles.len();
                    let default = if n_alleles == 2 {
                        AllelePosteriors::Biallelic(0.0)
                    } else {
                        AllelePosteriors::Multiallelic(vec![0.0f32; n_alleles])
                    };
                    let column = ref_win.column(MarkerIdx::new(marker_idx as u32));
                    let map_ref_to_targ = alignment
                        .target_marker(marker_idx)
                        .and_then(|target_m| {
                            alignment
                                .allele_mappings
                                .get(target_m)
                                .and_then(|m| m.as_ref())
                                .map(|m| m.ref_to_targ.as_slice())
                        });
                    let block = get_pattern_block(marker_idx);
                    cache.data = (0..n_samples)
                        .into_par_iter()
                        .map(|s| {
                            if let Some((p1, p2)) = sample_posteriors.get(&s) {
                                let mut local_cache1 = AllelePosteriorCache::default();
                                let mut local_cache2 = AllelePosteriorCache::default();
                                let (post1, post2) = match column {
                                    crate::data::storage::GenotypeColumn::Dense(_)
                                    | crate::data::storage::GenotypeColumn::Sparse(_) => {
                                        if let Some(block) = block.as_ref() {
                                            if let Some(pattern_alleles) = block.pattern_alleles(marker_idx) {
                                                (
                                                    p1.allele_posteriors_for_patterns_cached(
                                                        marker_idx,
                                                        n_alleles,
                                                        block.hap_to_pattern(),
                                                        pattern_alleles,
                                                        map_ref_to_targ,
                                                        &mut local_cache1,
                                                        block.block_id(),
                                                    ),
                                                    p2.allele_posteriors_for_patterns_cached(
                                                        marker_idx,
                                                        n_alleles,
                                                        block.hap_to_pattern(),
                                                        pattern_alleles,
                                                        map_ref_to_targ,
                                                        &mut local_cache2,
                                                        block.block_id(),
                                                    ),
                                                )
                                            } else {
                                                (
                                                    p1.allele_posteriors_for_column_cached(
                                                        marker_idx,
                                                        n_alleles,
                                                        column,
                                                        map_ref_to_targ,
                                                        &mut local_cache1,
                                                    ),
                                                    p2.allele_posteriors_for_column_cached(
                                                        marker_idx,
                                                        n_alleles,
                                                        column,
                                                        map_ref_to_targ,
                                                        &mut local_cache2,
                                                    ),
                                                )
                                            }
                                        } else {
                                            (
                                                p1.allele_posteriors_for_column_cached(
                                                    marker_idx,
                                                    n_alleles,
                                                    column,
                                                    map_ref_to_targ,
                                                    &mut local_cache1,
                                                ),
                                                p2.allele_posteriors_for_column_cached(
                                                    marker_idx,
                                                    n_alleles,
                                                    column,
                                                    map_ref_to_targ,
                                                    &mut local_cache2,
                                                ),
                                            )
                                        }
                                    }
                                    _ => (
                                        p1.allele_posteriors_for_column_cached(
                                            marker_idx,
                                            n_alleles,
                                            column,
                                            map_ref_to_targ,
                                            &mut local_cache1,
                                        ),
                                        p2.allele_posteriors_for_column_cached(
                                            marker_idx,
                                            n_alleles,
                                            column,
                                            map_ref_to_targ,
                                            &mut local_cache2,
                                        ),
                                    ),
                                };
                                (post1, post2)
                            } else {
                                (default.clone(), default.clone())
                            }
                        })
                        .collect();
                }
                cache
                    .data
                    .get(sample_idx)
                    .cloned()
                    .unwrap_or_else(|| (AllelePosteriors::Biallelic(0.0), AllelePosteriors::Biallelic(0.0)))
            }))
        } else {
            None
        };

        struct HapAltCache {
            marker_idx: usize,
            data: Vec<(f32, f32)>,
        }
        let hap_alt_cache = std::cell::RefCell::new(HapAltCache {
            marker_idx: usize::MAX,
            data: Vec::new(),
        });
        let get_hap_alt: Option<Box<dyn Fn(usize, usize) -> (f32, f32) + '_>> =
            if !sample_posteriors.is_empty() && !include_posteriors {
                let sample_posteriors = Arc::clone(&sample_posteriors);
                Some(Box::new(move |marker_idx, sample_idx| {
                    let mut cache = hap_alt_cache.borrow_mut();
                    if cache.marker_idx != marker_idx {
                        cache.marker_idx = marker_idx;
                        let marker = ref_win.marker(MarkerIdx::new(marker_idx as u32));
                        let n_alleles = 1 + marker.alt_alleles.len();
                        let column = ref_win.column(MarkerIdx::new(marker_idx as u32));
                        let map_ref_to_targ = alignment
                            .target_marker(marker_idx)
                            .and_then(|target_m| {
                                alignment
                                    .allele_mappings
                                    .get(target_m)
                                    .and_then(|m| m.as_ref())
                                    .map(|m| m.ref_to_targ.as_slice())
                            });
                        let block = get_pattern_block(marker_idx);
                        cache.data = (0..n_samples)
                            .into_par_iter()
                            .map(|s| {
                                if let Some((p1, p2)) = sample_posteriors.get(&s) {
                                    let mut local_cache1 = AllelePosteriorCache::default();
                                    let mut local_cache2 = AllelePosteriorCache::default();
                                    let (post1, post2) = match column {
                                        crate::data::storage::GenotypeColumn::Dense(_)
                                        | crate::data::storage::GenotypeColumn::Sparse(_) => {
                                            if let Some(block) = block.as_ref() {
                                                if let Some(pattern_alleles) = block.pattern_alleles(marker_idx) {
                                                    (
                                                        p1.allele_posteriors_for_patterns_cached(
                                                            marker_idx,
                                                            n_alleles,
                                                            block.hap_to_pattern(),
                                                            pattern_alleles,
                                                            map_ref_to_targ,
                                                            &mut local_cache1,
                                                            block.block_id(),
                                                        ),
                                                        p2.allele_posteriors_for_patterns_cached(
                                                            marker_idx,
                                                            n_alleles,
                                                            block.hap_to_pattern(),
                                                            pattern_alleles,
                                                            map_ref_to_targ,
                                                            &mut local_cache2,
                                                            block.block_id(),
                                                        ),
                                                    )
                                                } else {
                                                    (
                                                        p1.allele_posteriors_for_column_cached(
                                                            marker_idx,
                                                            n_alleles,
                                                            column,
                                                            map_ref_to_targ,
                                                            &mut local_cache1,
                                                        ),
                                                        p2.allele_posteriors_for_column_cached(
                                                            marker_idx,
                                                            n_alleles,
                                                            column,
                                                            map_ref_to_targ,
                                                            &mut local_cache2,
                                                        ),
                                                    )
                                                }
                                            } else {
                                                (
                                                    p1.allele_posteriors_for_column_cached(
                                                        marker_idx,
                                                        n_alleles,
                                                        column,
                                                        map_ref_to_targ,
                                                        &mut local_cache1,
                                                    ),
                                                    p2.allele_posteriors_for_column_cached(
                                                        marker_idx,
                                                        n_alleles,
                                                        column,
                                                        map_ref_to_targ,
                                                        &mut local_cache2,
                                                    ),
                                                )
                                            }
                                        }
                                        _ => (
                                            p1.allele_posteriors_for_column_cached(
                                                marker_idx,
                                                n_alleles,
                                                column,
                                                map_ref_to_targ,
                                                &mut local_cache1,
                                            ),
                                            p2.allele_posteriors_for_column_cached(
                                                marker_idx,
                                                n_alleles,
                                                column,
                                                map_ref_to_targ,
                                                &mut local_cache2,
                                            ),
                                        ),
                                    };
                                    (post1.prob(1), post2.prob(1))
                                } else {
                                    (0.0, 0.0)
                                }
                            })
                            .collect();
                    }
                    cache
                        .data
                        .get(sample_idx)
                        .copied()
                        .unwrap_or((0.0, 0.0))
                }))
            } else {
                None
            };

        let get_posteriors_for_writer = get_posteriors.as_ref().map(|get_post| {
            let get_post = Rc::clone(get_post);
            move |marker_idx: usize, sample_idx: usize| get_post(marker_idx, sample_idx)
        });

        // Closure to get dosage: marker_idx is window-local ref marker index from VCF writer
        // Dosages array is indexed from 0 for markers starting at markers_to_process_start
        let get_dosage = |marker_idx: usize, sample_idx: usize| -> f32 {
            let local_m = marker_idx.saturating_sub(markers_to_process_start);
            let (p1, p2) = if let Some((p1, p2)) = sample_hap_probs.get(&sample_idx) {
                (p1.get(local_m).copied().unwrap_or(0.0), p2.get(local_m).copied().unwrap_or(0.0))
            } else if let Some(ref get_post) = get_posteriors {
                let (post1, post2) = get_post(marker_idx, sample_idx);
                (post1.prob(1), post2.prob(1))
            } else if let Some(ref get_alt) = get_hap_alt {
                get_alt(marker_idx, sample_idx)
            } else {
                (0.0, 0.0)
            };

            if let Some(target_m) = alignment.target_marker(marker_idx) {
                let h1 = HapIdx::new((sample_idx * 2) as u32);
                let h2 = HapIdx::new((sample_idx * 2 + 1) as u32);
                let raw_a1 = target_win.allele(MarkerIdx::new(target_m as u32), h1);
                let raw_a2 = target_win.allele(MarkerIdx::new(target_m as u32), h2);

                // Map target alleles to reference allele space
                let mapping = alignment.allele_mappings.get(target_m).and_then(|m| m.as_ref());
                let map_allele = |a: u8| -> u8 {
                    if a == 255 {
                        return 255;
                    }
                    if let Some(m) = mapping {
                        if (a as usize) < m.targ_to_ref.len() {
                            let r = m.targ_to_ref[a as usize];
                            if r >= 0 {
                                r as u8
                            } else {
                                255
                            }
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
                    return (a1 as f32) + (a2 as f32);
                }

                let conf = target_win
                    .sample_confidence_f32(MarkerIdx::new(target_m as u32), sample_idx)
                    .clamp(0.0, 1.0);
                if a1 == 255 || a2 == 255 || a1 > 1 || a2 > 1 {
                    p1 + p2
                } else {
                    let is_het = a1 != a2;
                    let (l00, l01, l11) = if is_het {
                        (0.5 * (1.0 - conf), conf, 0.5 * (1.0 - conf))
                    } else if a1 == 1 {
                        (0.5 * (1.0 - conf), 0.5 * (1.0 - conf), conf)
                    } else {
                        (conf, 0.5 * (1.0 - conf), 0.5 * (1.0 - conf))
                    };
                    let p00 = (1.0 - p1) * (1.0 - p2);
                    let p01 = p1 * (1.0 - p2) + p2 * (1.0 - p1);
                    let p11 = p1 * p2;
                    let q00 = p00 * l00;
                    let q01 = p01 * l01;
                    let q11 = p11 * l11;
                    let sum = q00 + q01 + q11;
                    if sum > 0.0 {
                        let inv_sum = 1.0 / sum;
                        let q01n = q01 * inv_sum;
                        let q11n = q11 * inv_sum;
                        q01n + 2.0 * q11n
                    } else {
                        // If sum is 0, prior and likelihood are disjoint (e.g. prior says 0|0, data says 1|1 with high confidence)
                        // Trust the data (hard call) if we have it
                        let d1 = if a1 != 0 { 1.0 } else { 0.0 };
                        let d2 = if a2 != 0 { 1.0 } else { 0.0 };
                        d1 + d2
                    }
                }
            } else {
                if let Some((dosages, _)) = sample_data.get(&sample_idx) {
                    if !dosages.is_empty() {
                        dosages.get(local_m).copied().unwrap_or(p1 + p2)
                    } else {
                        p1 + p2
                    }
                } else {
                    p1 + p2
                }
            }
        };

        // Closure to get best genotype
        let get_best_gt = |marker_idx: usize, sample_idx: usize| -> (u8, u8) {
            let local_m = marker_idx.saturating_sub(markers_to_process_start);
            let (p1, p2) = if let Some((p1, p2)) = sample_hap_probs.get(&sample_idx) {
                (p1.get(local_m).copied().unwrap_or(0.0), p2.get(local_m).copied().unwrap_or(0.0))
            } else if let Some(ref get_alt) = get_hap_alt {
                get_alt(marker_idx, sample_idx)
            } else {
                (0.0, 0.0)
            };
            if let Some(target_m) = alignment.target_marker(marker_idx) {
                let h1 = HapIdx::new((sample_idx * 2) as u32);
                let h2 = HapIdx::new((sample_idx * 2 + 1) as u32);
                let raw_a1 = target_win.allele(MarkerIdx::new(target_m as u32), h1);
                let raw_a2 = target_win.allele(MarkerIdx::new(target_m as u32), h2);

                // Map target alleles to reference allele space
                let mapping = alignment.allele_mappings.get(target_m).and_then(|m| m.as_ref());
                let map_allele = |a: u8| -> u8 {
                    if a == 255 {
                        return 255;
                    }
                    if let Some(m) = mapping {
                        if (a as usize) < m.targ_to_ref.len() {
                            let r = m.targ_to_ref[a as usize];
                            if r >= 0 {
                                r as u8
                            } else {
                                255
                            }
                        } else {
                            255
                        }
                    } else {
                        a
                    }
                };
                let a1 = map_allele(raw_a1);
                let a2 = map_allele(raw_a2);

                let conf = target_win
                    .sample_confidence_f32(MarkerIdx::new(target_m as u32), sample_idx)
                    .clamp(0.0, 1.0);
                if a1 == 255 || a2 == 255 || a1 > 1 || a2 > 1 {
                    if p1 + p2 >= 1.5 {
                        (1, 1)
                    } else if p1 + p2 >= 0.5 {
                        (0, 1)
                    } else {
                        (0, 0)
                    }
                } else {
                    // Start with hard-call from mapped alleles
                    // For phasing preservation, if conf is high, we should respect a1/a2 order
                    if conf >= 0.99 {
                        (a1, a2)
                    } else {
                        let is_het = a1 != a2;
                        let (l00, l01, l11) = if is_het {
                            (0.5 * (1.0 - conf), conf, 0.5 * (1.0 - conf))
                        } else if a1 == 1 {
                            (0.5 * (1.0 - conf), 0.5 * (1.0 - conf), conf)
                        } else {
                            (conf, 0.5 * (1.0 - conf), 0.5 * (1.0 - conf))
                        };
                        let p00 = (1.0 - p1) * (1.0 - p2);
                        let p01 = p1 * (1.0 - p2) + p2 * (1.0 - p1);
                        let p11 = p1 * p2;
                        let q00 = p00 * l00;
                        let q01 = p01 * l01;
                        let q11 = p11 * l11;
                        if q11 >= q01 && q11 >= q00 {
                            (1, 1)
                        } else if q01 >= q00 {
                            (0, 1)
                        } else {
                            (0, 0)
                        }
                    }
                }
            } else {
                if let Some((_, best_gt)) = sample_data.get(&sample_idx) {
                    if !best_gt.is_empty() {
                        best_gt.get(local_m).copied().unwrap_or((0, 0))
                    } else {
                        if p1 + p2 >= 1.5 {
                            (1, 1)
                        } else if p1 + p2 >= 0.5 {
                            (0, 1)
                        } else {
                            (0, 0)
                        }
                    }
                } else {
                    if p1 + p2 >= 1.5 {
                        (1, 1)
                    } else if p1 + p2 >= 0.5 {
                        (0, 1)
                    } else {
                        (0, 0)
                    }
                }
            }
        };

        if let Some(ref get_post) = get_posteriors {
            for marker_idx in markers_to_process_start..output_end {
                if marker_idx >= ref_is_biallelic.len() || !ref_is_biallelic[marker_idx] {
                    continue;
                }
                if let Some(stats) = quality.get_mut(marker_idx) {
                    for s in 0..n_samples {
                        let (post1, post2) = get_post(marker_idx, s);
                        let (mut v1, mut v2) = (post1.prob(1), post2.prob(1));
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
        } else if let Some(ref get_alt) = get_hap_alt {
            for marker_idx in markers_to_process_start..output_end {
                if marker_idx >= ref_is_biallelic.len() || !ref_is_biallelic[marker_idx] {
                    continue;
                }
                if let Some(stats) = quality.get_mut(marker_idx) {
                    for s in 0..n_samples {
                        let (mut v1, mut v2) = get_alt(marker_idx, s);
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

        writer.write_imputed_streaming(
            ref_win,
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
    use crate::data::haplotype::Samples;
    use crate::data::marker::{Allele, Marker, Markers};
    use crate::data::storage::GenotypeColumn;
    use crate::data::ChromIdx;
    use crate::io::bref3::InMemoryRefReader;

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

    fn build_phased_matrix(markers: Markers, n_samples: usize) -> GenotypeMatrix<Phased> {
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
        GenotypeMatrix::new_phased(markers, columns, samples)
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

        let ref_markers = build_markers(chrom, &ref_positions);
        let target_markers = build_markers(chrom, &target_positions);

        let ref_gt = Arc::new(build_phased_matrix(ref_markers, 2));
        let target_gt = build_unphased_matrix(target_markers, 2);

        let mut ref_reader = RefPanelReader::InMemory(InMemoryRefReader::new(ref_gt.clone()));
        let config = StreamingConfig::default();
        let gen_maps = GeneticMaps::default();
        let ref_window = ref_reader
            .next_window(&config, &gen_maps)
            .expect("ref window load failed")
            .expect("no ref window found");

        // Desired behavior: sparse target data should not truncate the reference region.
        assert_eq!(target_gt.n_markers(), target_positions.len());
        assert_eq!(ref_window.global_start, 0);
        assert_eq!(ref_window.global_end, ref_gt.n_markers());
    }
}
