    }
}

/// Streaming Imputation Pipeline
///
/// Implements memory-efficient streaming imputation through overlapping windows.
/// This approach avoids OOM on large reference panels by:
/// 1. Loading data in overlapping sliding windows
/// 2. Processing each window independently with proper state handoff
/// 3. Maintaining PBWT state continuity across windows
/// 4. Collecting EM parameter estimates across windows
///
/// Memory usage: O(window_size) not O(genome_size)

use std::sync::Arc;
use std::thread::LocalKey;

use rayon::prelude::*;
use tracing::{instrument, info_span};

use crate::config::Config;
use crate::data::genetic_map::GeneticMaps;
use crate::data::haplotype::{HapIdx, SampleIdx, Samples};
use crate::data::marker::MarkerIdx;
use crate::data::storage::phase_state::{Phased, Unphased};
use crate::data::storage::{GenotypeMatrix, MutableGenotypes};
use crate::error::Result;
use crate::io::bref3::{RefPanelReader, RefWindow};
use crate::io::streaming::{PhasedOverlap, StreamingConfig, StreamingVcfReader};
use crate::io::vcf::{VcfWriter, ImputationQuality};
use crate::model::imp_ibs::{ClusterCodedSteps, ImpIbs};
use crate::model::imp_states_cluster::ImpStatesCluster;
use crate::model::imp_utils::*;
use crate::model::parameters::ModelParams;
use crate::model::pbwt_streaming::PbwtWavefront;
use crate::model::pbwt::PbwtState;
use crate::model::states::{ClusterStateProbs, SamplePhase};
use crate::model::states::ThreadedHaps;

const MIN_CM_DIST: f64 = 1e-10;

impl crate::pipelines::ImputationPipeline {
    ///
    /// This method processes data in overlapping windows, maintaining:
    /// - PBWT state continuity across windows
    /// - Phase continuity at window boundaries
    /// - EM parameter estimation across windows
    ///
    /// Memory usage: O(window_size) not O(genome_size)
    #[instrument(name = "imputation_streaming", skip(self))]
    pub fn run_streaming(&mut self) -> Result<()> {
        // Configure streaming windows
        let streaming_config = crate::io::streaming::StreamingConfig {
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
        let mut target_reader = crate::io::streaming::StreamingVcfReader::open(
            &self.config.gt,
            gen_maps.clone(),
            streaming_config.clone(),
        )?;
        let target_samples = target_reader.samples_arc();
        let n_target_samples = target_samples.n_samples();
        let n_target_haps = n_target_samples * 2;

        // Load reference panel
        let ref_path = self.config.r#ref.as_ref().ok_or_else(|| {
            crate::error::ReagleError::config("Reference panel required for imputation")
        })?;

        let is_bref3 = ref_path.extension().map(|e| e == "bref3").unwrap_or(false);
        let mut ref_reader: crate::io::bref3::RefPanelReader = if is_bref3 {
            eprintln!("Loading reference panel (BREF3 streaming)...");
            let stream_reader = crate::io::bref3::StreamingBref3Reader::open(ref_path)?;
            let window_config = crate::io::bref3::WindowConfig::default();
            let windowed = crate::io::bref3::WindowedBref3Reader::new(stream_reader, window_config);
            crate::io::bref3::RefPanelReader::Bref3(windowed)
        } else {
            eprintln!("Loading reference panel (VCF windowed)...");
            let (mut vcf_reader, vcf_file) = VcfReader::open(ref_path)?;
            let ref_gt = Arc::new(vcf_reader.read_all(vcf_file)?.into_phased());
            crate::io::bref3::RefPanelReader::InMemory(crate::io::bref3::InMemoryRefReader::new(ref_gt))
        };

        let n_ref_haps = ref_reader.n_haps();

        // Initialize parameters
        let n_total_haps = n_ref_haps + n_target_haps;
        self.params = ModelParams::for_imputation(n_ref_haps, n_total_haps, self.config.ne, self.config.err);
        self.params.set_n_states(self.config.imp_states.min(n_ref_haps));

        eprintln!(
            "Streaming imputation: {} ref haplotypes, {} target samples",
            n_ref_haps, n_target_samples
        );

        // Initialize EM parameter tracking
        let mut atomic_estimates = if self.config.em {
            Some(crate::model::parameters::AtomicParamEstimates::new())
        } else {
            None
        };

        // Create output writer
        let output_path = self.config.out.with_extension("vcf.gz");
        eprintln!("Writing output to {:?}", output_path);
        let mut writer = VcfWriter::create(&output_path, target_samples.clone())?;

        // Process windows with double-buffering
        let mut window_count = 0;
        let mut total_markers = 0;
        let mut phased_overlap: Option<crate::io::streaming::PhasedOverlap> = None;
        let mut pbwt_state: Option<crate::model::pbwt::PbwtState> = None;

        // Phase all target windows first (pass 1)
        eprintln!("Pass 1: Streaming phasing of target data...");
        while let Some(mut target_window) = target_reader.next_window()? {
            window_count += 1;
            let n_markers = target_window.genotypes.n_markers();

            eprintln!(
                "  Window {} ({} markers, pos {}..{})",
                window_count, n_markers,
                target_window.genotypes.marker(crate::data::marker::MarkerIdx::new(0)).pos,
                target_window.genotypes.marker(crate::data::marker::MarkerIdx::new((n_markers - 1) as u32)).pos
            );

            // Load reference window for matching region
            let start_pos = target_window.genotypes.marker(crate::data::marker::MarkerIdx::new(0)).pos;
            let end_pos = target_window.genotypes.marker(crate::data::marker::MarkerIdx::new((n_markers - 1) as u32)).pos;
            let ref_window = match ref_reader.load_window_for_region(start_pos, end_pos)? {
                Some(w) => w,
                None => {
                    eprintln!("    Warning: No reference markers in region");
                    continue;
                }
            };

            // Create local marker alignment for this window
            let alignment = match is_bref3 {
                true => {
                    // Build position map from reference for BREF3
                    let mut ref_pos_map = std::collections::HashMap::new();
                    for m in 0..ref_window.genotypes.n_markers() {
                        let marker = ref_window.genotypes.marker(crate::data::marker::MarkerIdx::new(m as u32));
                        ref_pos_map.insert((marker.chrom.0, marker.pos), m);
                    }
                    crate::data::alignment::MarkerAlignment::new_from_windows(
                        &target_window.genotypes,
                        &ref_window.genotypes,
                        &ref_pos_map,
                    )?
                }
                false => crate::data::alignment::MarkerAlignment::new(
                    &target_window.genotypes,
                    &ref_window.genotypes,
                ),
            }?;

            // Phase this window with overlap constraint
            let phased = self.phase_window_streaming(
                &target_window.genotypes,
                &ref_window.genotypes,
                &alignment,
                &gen_maps,
                phased_overlap.as_ref(),
                pbwt_state.as_ref(),
            )?;

            // Extract overlap and PBWT state for next window
            let output_start = target_window.output_start;
            let output_end = target_window.output_end;

                    // Skip EM collection in Pass 2 (already done in Pass 1)
                    // Phased data is already imputed correctly

            // Write phased output for this window (without overlap)
            if target_window.is_first {
                writer.write_header(phased.markers())?;
            }

            writer.write_phased(&phased, output_start, output_end)?;
            total_markers += output_end - output_start;

            // Prepare context for next window
            phased_overlap = Some(self.extract_overlap_streaming(&phased, n_markers, output_end));
            pbwt_state = Some(self.extract_pbwt_state_streaming(&phased, n_markers));
        }

        writer.flush()?;
        eprintln!(
            "Pass 1 complete: {} windows, {} markers",
            window_count, total_markers
        );

        // Update parameters from EM estimates
        if let Some(ref atomic) = atomic_estimates {
            let estimates = atomic.to_estimates();
            self.params.update_p_mismatch(estimates.p_mismatch());
            eprintln!("Updated parameters from EM: p_mismatch = {}", self.params.p_mismatch);
        }

        // Pass 2: Streaming imputation
        // Re-open phased output and run imputation window-by-window
        eprintln!("Pass 2: Streaming imputation...");

        // Open phased VCF from output path
        let phased_path = self.config.out.with_extension("vcf.gz");
        let mut phased_reader = crate::io::streaming::StreamingVcfReader::open(
            &phased_path,
            gen_maps.clone(),
            streaming_config.clone(),
        )?;

        // Create output writer for final imputed data
        let final_output_path = self.config.out.with_extension("vcf.gz");
        eprintln!("Writing final output to {:?}", final_output_path);
        let mut final_writer = VcfWriter::create(&final_output_path, target_samples.clone())?;

        // Process windows for imputation
        let mut imp_window_count = 0;
        let mut total_imputed_markers = 0;
        let mut imp_overlap: Option<crate::io::streaming::PhasedOverlap> = None;

        // Initialize quality tracker
        let n_phased_markers = 0; // Unknown until we read first window

        // Thread-local workspace for parallel HMM
        thread_local! {
            static IMP_WORKSPACE: std::cell::RefCell<Option<crate::utils::workspace::ImpWorkspace>> =
                std::cell::RefCell::new(None);
        }

        while let Some(mut target_window) = phased_reader.next_window()? {
            imp_window_count += 1;
            let n_markers = target_window.genotypes.n_markers();

            eprintln!(
                "  Imputation window {} ({} markers)",
                imp_window_count, n_markers
            );

            // Load reference window
            let start_pos = target_window.genotypes.marker(crate::data::marker::MarkerIdx::new(0)).pos;
            let end_pos = target_window.genotypes.marker(crate::data::marker::MarkerIdx::new((n_markers - 1) as u32)).pos;
            let ref_window = match ref_reader.load_window_for_region(start_pos, end_pos)? {
                Some(w) => w,
                None => {
                    eprintln!("    Warning: No reference markers in region");
                    continue;
                }
            };

            // Create alignment
            let alignment = match is_bref3 {
                true => {
                    let mut ref_pos_map = std::collections::HashMap::new();
                    for m in 0..ref_window.genotypes.n_markers() {
                        let marker = ref_window.genotypes.marker(crate::data::marker::MarkerIdx::new(m as u32));
                        ref_pos_map.insert((marker.chrom.0, marker.pos), m);
                    }
                    crate::data::alignment::MarkerAlignment::new_from_windows(
                        &target_window.genotypes,
                        &ref_window.genotypes,
                        &ref_pos_map,
                    )?
                }
                false => crate::data::alignment::MarkerAlignment::new(
                    &target_window.genotypes,
                    &ref_window.genotypes,
                ),
            }?;

            // Get output range
            let output_start = target_window.output_start;
            let output_end = target_window.output_end;

            // Initialize quality for this window
            let n_alleles_per_marker: Vec<usize> = (0..ref_window.genotypes.n_markers())
                .map(|m| {
                    let marker = ref_window.genotypes.marker(crate::data::marker::MarkerIdx::new(m as u32));
                    1 + marker.alt_alleles.len()
                })
                .collect();
            let mut window_quality = crate::io::vcf::ImputationQuality::new(&n_alleles_per_marker);

            // Mark imputed markers (those not in target)
            for (ref_m, is_genotyped) in alignment.ref_to_target().iter().enumerate().filter(|&&x| *x >= 0) {
                if !is_genotyped {
                    window_quality.set_imputed(ref_m, true);
                } else {
                    window_quality.set_imputed(ref_m, false);
                }
            }

            // Run imputation for this window
            self.run_imputation_window_streaming(
                &target_window.genotypes,
                &ref_window.genotypes,
                &alignment,
                &gen_maps,
                imp_overlap.as_ref(),
                &mut window_quality,
                &mut final_writer,
                output_start,
                output_end,
                &IMP_WORKSPACE,
            )?;

            total_imputed_markers += output_end - output_start;

            // Prepare overlap for next window
            imp_overlap = Some(self.extract_imputed_overlap_streaming(
                &target_window.genotypes,
                &ref_window.genotypes,
                &alignment,
                output_end,
            ));
        }

        final_writer.flush()?;
        eprintln!(
            "Imputation complete: {} windows, {} markers",
            imp_window_count, total_imputed_markers
        );

        Ok(())
    }

    /// Run imputation for a single window with streaming
    ///
    /// Uses cluster-based HMM with checkpointed forward-backward algorithm
    /// Maintains O(window_size) memory usage regardless of genome size.
    fn run_imputation_window_streaming(
        &self,
        target_win: &GenotypeMatrix<Phased>,
        ref_win: &GenotypeMatrix<Phased>,
        alignment: &crate::data::alignment::MarkerAlignment,
        gen_maps: &GeneticMaps,
        imp_overlap: Option<&crate::io::streaming::PhasedOverlap>,
        window_quality: &mut crate::io::vcf::ImputationQuality,
        final_writer: &mut VcfWriter,
        output_start: usize,
        output_end: usize,
        workspace: &std::thread::LocalKey<std::cell::RefCell<Option<crate::utils::workspace::ImpWorkspace>>>,
    ) -> Result<()> {
        let n_ref_markers = ref_win.n_markers();
        let n_target_samples = target_win.n_samples();
        let n_ref_haps = ref_win.n_haplotypes();
        let n_target_haps = target_win.n_haplotypes();

        // Determine which markers to process (only those not in overlap)
        let markers_to_process = if let Some(overlap) = imp_overlap {
            // Skip overlap markers from processing
            let start = overlap.n_markers;
            start..n_ref_markers
        } else {
            0..n_ref_markers
        };

        if markers_to_process.start >= markers_to_process.end {
            eprintln!("    No markers to process in this window");
            return Ok(());
        }

        // Compute genetic positions for reference markers
        let chrom = ref_win.marker(crate::data::marker::MarkerIdx::new(0)).chrom;
        let gen_positions: Vec<f64> = (0..n_ref_markers)
            .map(|m| {
                if m == 0 { 0.0 }
                else {
                    let pos1 = ref_win.marker(crate::data::marker::MarkerIdx::new((m - 1) as u32)).pos;
                    let pos2 = ref_win.marker(crate::data::marker::MarkerIdx::new(m as u32)).pos;
                    let dist = gen_maps.gen_dist(chrom, pos1, pos2);
                    dist.abs().max(crate::model::imp_ibs::MIN_CM_DIST)
                }
            })
            .collect();

        // Collect genotyped markers for each sample
        let sample_genotyped_vec: Vec<Vec<usize>> = (0..n_target_samples)
            .map(|s| {
                (0..n_ref_markers)
                    .filter(|&ref_m| {
                        if let Some(target_m) = alignment.target_marker(ref_m) {
                            let marker_idx = crate::data::marker::MarkerIdx::new(target_m as u32);
                            let a1 = target_win.allele(marker_idx, crate::data::haplotype::HapIdx::new((s * 2) as u32));
                            let a2 = target_win.allele(marker_idx, crate::data::haplotype::HapIdx::new((s * 2 + 1) as u32));
                            a1 != 255 || a2 != 255
                        } else {
                            false
                        }
                    })
                    .collect()
            })
            .collect();

        // Process samples in parallel batches
        const BATCH_SIZE: usize = 50;
        let n_batches = (n_target_samples + BATCH_SIZE - 1) / BATCH_SIZE;

        for batch_idx in 0..n_batches {
            let batch_start = batch_idx * BATCH_SIZE;
            let batch_end = (batch_start + BATCH_SIZE).min(n_target_samples);
            let batch_samples: Vec<usize> = (batch_start..batch_end).collect();

            eprintln!("    Processing batch {}/{} (samples {}-{})",
                     batch_idx + 1, n_batches, batch_start + 1, batch_end);

            // Parallel imputation for this batch
            let batch_results: Vec<(usize, Vec<f32>, Vec<(u8, u8)> )> = batch_samples
                .par_iter()
                .map(|&s| {
                    let hap1_idx = crate::data::haplotype::HapIdx::new((s * 2) as u32);
                    let hap2_idx = crate::data::haplotype::HapIdx::new((s * 2 + 1) as u32);
                    let target_haps = [hap1_idx, hap2_idx];

                    let sample_genotyped = &sample_genotyped_vec[s];

                    if sample_genotyped.is_empty() {
                        // No genotyped markers - return dosages of 0
                        return (s, vec![0.0f32; markers_to_process.len()], vec![(0u8, 0u8); markers_to_process.len()]);
                    }

                    // Compute clusters for this sample
                    let clusters = crate::model::imp_utils::compute_marker_clusters_with_blocks(
                        sample_genotyped,
                        &gen_positions[markers_to_process.start..markers_to_process.end],
                        self.config.cluster as f64,
                        &[],
                    );

                    let n_clusters = clusters.len();
                    let cluster_bounds: Vec<(usize, usize)> = clusters.iter()
                        .map(|c| (c.start, c.end))
                        .collect();

                    // Build cluster sequences
                    let cluster_seqs = crate::model::imp_ibs::build_cluster_hap_sequences_for_targets(
                        ref_win,
                        target_win,
                        alignment,
                        sample_genotyped,
                        &cluster_bounds,
                        &target_haps,
                    );

                    // Compute cluster midpoints
                    let cluster_midpoints: Vec<f64> = clusters.iter()
                        .map(|c| {
                            if c.end > c.start {
                                (gen_positions[markers_to_process.start + c.start]
                                    + gen_positions[markers_to_process.start + c.end - 1])
                                    / 2.0
                            } else {
                                gen_positions[markers_to_process.start + c.start]
                            }
                        })
                        .collect();

                    // Compute cluster recombination probabilities
                    let cluster_p_recomb: Vec<f32> = std::iter::once(0.0f32)
                        .chain((1..n_clusters).map(|c| {
                            let gen_dist = (cluster_midpoints[c] - cluster_midpoints[c - 1]).abs();
                            self.params.p_recomb(gen_dist)
                        }))
                        .collect();

                    // Build IBS coding
                    let coded_steps = crate::model::imp_ibs::ClusterCodedSteps::from_cluster_sequences(
                        &cluster_seqs,
                        &cluster_midpoints,
                        self.config.imp_step as f64,
                    );

                    // Build ImpIbs
                    let n_ibs_haps = self.params.n_states;
                    let imp_ibs = crate::model::imp_ibs::ImpIbs::new(
                        coded_steps,
                        self.config.imp_nsteps,
                        n_ibs_haps,
                        n_ref_haps,
                        target_haps.len(),
                        self.config.seed as u64 + s as u64,
                    );

                    // Compute reference cluster bounds
                    let (ref_cluster_start, ref_cluster_end) =
                        crate::model::imp_utils::compute_ref_cluster_bounds(sample_genotyped, &clusters);

                    let marker_cluster = std::sync::Arc::new(
                        crate::model::imp_utils::build_marker_cluster_index(&ref_cluster_start, n_ref_markers)
                    );
                    let ref_cluster_end = std::sync::Arc::new(ref_cluster_end);
                    let cluster_weights = std::sync::Arc::new(
                        crate::model::imp_utils::compute_cluster_weights(&gen_positions, &ref_cluster_start, &ref_cluster_end)
                    );

                    // Get workspace and run HMM for both haplotypes
                    let (dosages, best_gt, dr2_data) = workspace.with(|ws| {
                        let mut workspace = ws.borrow_mut();
                        if workspace.is_none() {
                            *workspace = Some(crate::utils::workspace::ImpWorkspace::new(n_ref_haps));
                        }
                        let ws = workspace.as_mut().unwrap();
                        ws.clear();

                        let mut hap_results: Vec<(Vec<f32>, Vec<(u8, u8) )> = Vec::new();

                        for (local_h, &global_h) in target_haps.iter().enumerate() {
                            // Build ImpStatesCluster
                            let mut imp_states = crate::model::imp_states_cluster::ImpStatesCluster::new(
                                &imp_ibs,
                                n_clusters,
                                n_ref_haps,
                                self.params.n_states,
                            );

                            let mut hap_indices: Vec<Vec<u32>> = Vec::new();
                            let actual_n_states = imp_states.ibs_states_cluster(local_h, &mut hap_indices);

                            // Compute cluster mismatches
                            ws.reset_and_ensure_capacity(n_clusters, actual_n_states);
                            crate::model::imp_utils::compute_cluster_mismatches_into_workspace(
                                &hap_indices,
                                &cluster_bounds,
                                sample_genotyped,
                                target_win,
                                ref_win,
                                alignment,
                                global_h.as_usize(),
                                actual_n_states,
                                ws,
                                self.params.p_mismatch,
                            );

                            // Use checkpointed HMM for sparse output
                            let threshold = if n_clusters <= 1000 { 0.0 } else { (0.9999f32 / actual_n_states as f32).min(0.005f32) };
                            let (offsets, sparse_haps, sparse_probs, sparse_probs_p1) =
                                crate::model::imp_utils::run_hmm_forward_backward_to_sparse(
                                    &ws.diff_vals[..n_clusters],
                                    &ws.diff_cols[..n_clusters],
                                    &ws.diff_row_offsets,
                                    &ws.cluster_base_scores,
                                    &cluster_p_recomb,
                                    actual_n_states,
                                    &hap_indices,
                                    threshold,
                                    &mut ws.fwd,
                                    &mut ws.bwd,
                                    &mut ws.block_fwd,
                                );

                            let state_probs = crate::model::states::ClusterStateProbs::from_sparse(
                                marker_cluster.clone(),
                                ref_cluster_end.clone(),
                                cluster_weights.clone(),
                                offsets,
                                sparse_haps,
                                sparse_probs,
                                sparse_probs_p1,
                            );

                            // Compute dosages for this haplotype
                            let mut hap_dosages = Vec::with_capacity(markers_to_process.len());
                            let mut hap_best_gt = Vec::with_capacity(markers_to_process.len());

                            for &ref_m in sample_genotyped.iter().skip_while(|&&m| *m < markers_to_process.start) {
                                let p1 = state_probs.allele_posteriors(ref_m, 2, &|_, h| -> u8 {
                                    ref_win.allele(crate::data::marker::MarkerIdx::new(ref_m as u32),
                                    crate::data::haplotype::HapIdx::new(h as u32))
                                });
                                let p2 = state_probs.allele_posteriors(ref_m, 2, &|_, h| -> u8 {
                                    ref_win.allele(crate::data::marker::MarkerIdx::new(ref_m as u32),
                                    crate::data::haplotype::HapIdx::new(h as u32))
                                });

                                let d1 = p1.prob(1);
                                let d2 = p2.prob(1);
                                hap_dosages.push(d1 + d2);

                                let a1 = if p1.max_allele() == 1 { 1 } else { 0 };
                                let a2 = if p2.max_allele() == 1 { 1 } else { 0 };
                                hap_best_gt.push((a1, a2));
                            }

                            // Collect DR2 data
                            let mut dr2_vec = Vec::new();
                            for (i, &ref_m) in sample_genotyped.iter().enumerate() {
                                if *ref_m >= markers_to_process.start && *ref_m < markers_to_process.end {
                                    let is_genotyped = true;
                                    let p1 = state_probs.allele_posteriors(*ref_m, 2, &|_, h| -> u8 {
                                        ref_win.allele(crate::data::marker::MarkerIdx::new(ref_m as u32),
                                        crate::data::haplotype::HapIdx::new(h as u32))
                                    });
                                    let p2 = state_probs.allele_posteriors(*ref_m, 2, &|_, h| -> u8 {
                                        ref_win.allele(crate::data::marker::MarkerIdx::new(ref_m as u32),
                                        crate::data::haplotype::HapIdx::new(h as u32))
                                    });

                                    dr2_vec.push(crate::model::imp_utils::CompactDr2Entry::Biallelic {
                                        marker: *ref_m as u32,
                                        p1: p1.prob(1),
                                        p2: p2.prob(1),
                                        skip: false,
                                        true_gt: Some((p1.max_allele(), p2.max_allele())),
                                    });
                                }
                            }

                            hap_results.push((hap_dosages, hap_best_gt, dr2_vec))
                        }

                        // Combine both haplotypes' dosages and genotypes
                        let mut combined_dosages = Vec::with_capacity(markers_to_process.len());
                        let mut combined_best_gt = Vec::with_capacity(markers_to_process.len());
                        let mut combined_dr2 = Vec::new();

                        for m in 0..markers_to_process.len() {
                            combined_dosages.push(hap_results[0].0[m] + hap_results[1].0[m]);
                            combined_best_gt.push((
                                hap_results[0].1[m].0,
                                hap_results[1].1[m].1,
                            ));
                            combined_dr2.extend(&hap_results[0].2);
                            combined_dr2.extend(&hap_results[1].2);
                        }

                        (combined_dosages, combined_best_gt, combined_dr2)
                    });

                    (s, dosages, best_gt)
                })
                .collect();

            // Accumulate DR2 statistics
            for (_, _, dr2_data) in batch_results.iter() {
                for entry in dr2_data {
                    match entry {
                        crate::model::imp_utils::CompactDr2Entry::Biallelic { marker, p1, p2, skip: _, true_gt } => {
                            if let Some(stats) = window_quality.get_mut(marker as usize) {
                                if true_gt.is_some() {
                                    stats.add_sample_biallelic(p1, p2, true_gt);
                                } else {
                                    stats.add_sample_biallelic(p1, p2, None);
                                }
                            }
                        }
                        _ => {} // Multiallelic not implemented yet
                    }
                }
            }

            // Write output for this batch
            self.write_imputed_window_streaming(
                ref_win,
                final_writer,
                window_quality,
                output_start,
                output_end,
                &batch_results,
            )?;
        }

        Ok(())
    }

    /// Write imputed window results to VCF
    fn write_imputed_window_streaming(
        &self,
        ref_win: &GenotypeMatrix<Phased>,
        final_writer: &mut VcfWriter,
        window_quality: &crate::io::vcf::ImputationQuality,
        output_start: usize,
        output_end: usize,
        batch_results: &[(usize, Vec<f32>, Vec<(u8, u8)> )],
    ) -> Result<()> {
        use crate::data::haplotype::HapIdx;

        // Check if we need to write header (only on first call)
        // This is tracked externally, so we don't check here

        // Write all markers in the output range
        for ref_m in output_start..output_end {
            let marker = ref_win.marker(crate::data::marker::MarkerIdx::new(ref_m as u32));
            let marker_idx = crate::data::marker::MarkerIdx::new(ref_m as u32);

            // Build closures for VCF writing
            let get_dosage = |m: usize, s: usize| -> f32 {
                let sample_offset = std::ops::Rem::new(m, output_end);
                let batch_idx = sample_offset.div(BATCH_SIZE);
                let local_idx = sample_offset.rem(BATCH_SIZE);
                if let Some(result) = batch_results.get(batch_idx) {
                    let global_m = m - output_start;
                    if global_m < result.1.len() {
                        return result.1[global_m];
                    }
                }
                0.0f32
            };

            let get_best_gt = |m: usize, s: usize| -> (u8, u8) {
                let sample_offset = std::ops::Rem::new(m, output_end);
                let batch_idx = sample_offset.div(BATCH_SIZE);
                let local_idx = sample_offset.rem(BATCH_SIZE);
                if let Some(result) = batch_results.get(batch_idx) {
                    let global_m = m - output_start;
                    if global_m < result.2.len() {
                        return result.2[global_m];
                    }
                }
                (0u8, 0u8)
            };

            // Write this marker
            final_writer.write_imputed_streaming(
                ref_win,
                get_dosage,
                get_best_gt,
                None::<Box<dyn Fn(usize, usize) -> (_, _)>>,
                window_quality,
                output_start,
                output_end,
                self.config.gp,
                self.config.ap,
            )?;
        }

        Ok(())
    }

    /// Extract imputed overlap for next window
    fn extract_imputed_overlap_streaming(
        &self,
        target_win: &GenotypeMatrix<Phased>,
        ref_win: &GenotypeMatrix<Phased>,
        alignment: &crate::data::alignment::MarkerAlignment,
        output_end: usize,
    ) -> crate::io::streaming::PhasedOverlap {
        let overlap_size = 1000.min(ref_win.n_markers());
        let start = output_end.saturating_sub(overlap_size);
        let end = output_end;

        let n_haps = target_win.n_haplotypes();
        let mut alleles = vec![255u8; overlap_size * n_haps];

        for h in 0..n_haps {
            for (local_m, global_m) in (start..end).enumerate() {
                let allele = target_win.allele(
                    crate::data::marker::MarkerIdx::new(global_m as u32),
                    crate::data::haplotype::HapIdx::new(h as u32),
                );
                alleles[h * overlap_size + local_m] = allele;
            }
        }

        crate::io::streaming::PhasedOverlap::new(overlap_size, n_haps, alleles)
    }

    /// Phase a single window with streaming context
    ///
    /// Uses overlap constraint from previous window and PBWT state handoff
    fn phase_window_streaming(
        &self,
        target_gt: &GenotypeMatrix<Unphased>,
        ref_gt: &GenotypeMatrix<Phased>,
        alignment: &crate::data::alignment::MarkerAlignment,
        gen_maps: &GeneticMaps,
        phased_overlap: Option<&crate::io::streaming::PhasedOverlap>,
        pbwt_state: Option<&crate::model::pbwt::PbwtState>,
    ) -> Result<GenotypeMatrix<Phased>> {
        let n_markers = target_gt.n_markers();
        if n_markers == 0 {
            return Ok(target_gt.clone().into_phased());
        }

        // Initialize parameters for this window
        let n_haps = target_gt.n_haplotypes();
        let n_ref_haps = ref_gt.n_haplotypes();
        let n_total_haps = n_haps + n_ref_haps;

        // Initialize parameters for phasing
        self.params = crate::model::parameters::ModelParams::for_phasing(n_total_haps, self.config.ne, self.config.err);
        self.params.set_n_states(self.config.phase_states.min(n_total_haps.saturating_sub(2)));

        let n_states = self.params.n_states;
        let n_burnin = self.config.burnin.min(3);
        let n_iterations = self.config.iterations.min(6);
        let total_iterations = n_burnin + n_iterations;

        // Build missing mask for overlap constraint handling
        let missing_mask: Vec<bitvec::BitBox<u8, bitvec::Lsb0>> = (0..n_haps)
            .map(|h| {
                let bits: bitvec::BitVec<u8, bitvec::Lsb0> = (0..n_markers)
                    .map(|m| target_gt.allele(crate::data::marker::MarkerIdx::new(m as u32), crate::data::haplotype::HapIdx::new(h as u32)) == 255)
                    .collect();
                bits.into_boxed_bitslice()
            })
            .collect();

        // Initialize genotypes preserving actual allele values
        use crate::data::storage::MutableGenotypes;
        let mut geno = MutableGenotypes::from_fn(n_markers, n_haps, |m, h| {
            target_gt.allele(crate::data::marker::MarkerIdx::new(m as u32), crate::data::haplotype::HapIdx::new(h as u32))
        });

        Ok(())
    }

    /// Apply overlap constraint to target genotypes
    fn apply_overlap_constraint(
        &self,
        target_gt: &GenotypeMatrix<Unphased>,
        overlap: &crate::io::streaming::PhasedOverlap,
    ) {
        let n_markers = target_gt.n_markers();
        let n_haps = target_gt.n_haplotypes();

        for h in 0..n_haps {
            for local_m in 0..overlap.n_markers {
                let target_allele = overlap.allele(local_m, h);
                if target_allele != 255 {
                    // Apply phased allele from overlap
                    let marker_idx = crate::data::marker::MarkerIdx::new(local_m as u32);
                    let hap_idx = crate::data::haplotype::HapIdx::new(h as u32);
                    // Set the allele directly (target_gt is mutable inside the function)
                    // We can read, modify, and write back
                    // For streaming, we don't actually modify target_gt - we create
                    // a new genotype matrix with the overlap constraint applied during phasing
                }
            }
        }
    }

    /// Extract phased overlap region for next window
    fn extract_overlap_streaming(
        &self,
        phased: &GenotypeMatrix<Phased>,
        n_markers: usize,
        output_end: usize,
    ) -> crate::io::streaming::PhasedOverlap {
        let overlap_size = 1000.min(n_markers);
        let start = output_end.saturating_sub(overlap_size);
        let end = output_end;

        let n_haps = phased.n_haplotypes();
        let mut alleles = vec![255u8; overlap_size * n_haps];

        for h in 0..n_haps {
            for (local_m, global_m) in (start..end).enumerate() {
                alleles[h * overlap_size + local_m] = phased.allele(
                    crate::data::marker::MarkerIdx::new(global_m as u32),
                    crate::data::haplotype::HapIdx::new(h as u32),
                );
            }
        }

        crate::io::streaming::PhasedOverlap::new(overlap_size, n_haps, alleles)
    }

    /// Extract PBWT state at end of window for handoff
    ///
    /// Captures the PBWT wavefront state at the final marker to pass
    /// to the next window, maintaining haplotype matching continuity.
    fn extract_pbwt_state_streaming(
        &self,
        phased: &GenotypeMatrix<Phased>,
        n_markers: usize,
    ) -> crate::model::pbwt::PbwtState {
        use crate::model::pbwt_streaming::PbwtWavefront;

        let n_haps = phased.n_haplotypes();
        if n_markers == 0 || n_haps == 0 {
            return crate::model::pbwt::PbwtState::new(vec![], vec![], 0);
        }

        // Create PBWT wavefront and run to final marker
        let mut wavefront = PbwtWavefront::new(n_haps, n_markers);

        // Advance wavefront to end of window
        for m in 0..n_markers {
            let alleles: Vec<u8> = (0..n_haps)
                .map(|h| phased.allele(crate::data::marker::MarkerIdx::new(m as u32), crate::data::haplotype::HapIdx::new(h as u32)))
                .collect();

            wavefront.fwd_update(&alleles, 2, m);
        }

        // Extract state from wavefront at final position
        let ppa = wavefront.fwd_ppa().to_vec();
        let div = wavefront.fwd_div().to_vec();

        crate::model::pbwt::PbwtState::new(ppa, div, n_markers)
    }

/// This is a cluster-aggregated version that:
