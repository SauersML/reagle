//! Streaming Imputation Pipeline
//!
//! Implements memory-efficient streaming imputation through overlapping windows.
//! Uses a producer-consumer model with MPSC channel to pipe phased matrices
//! directly to imputation in-memory.

use std::sync::{Arc, mpsc};
use std::thread;

use rayon::prelude::*;
use tracing::instrument;
use crate::data::genetic_map::GeneticMaps;
use crate::data::haplotype::HapIdx;
use crate::data::marker::MarkerIdx;
use crate::data::storage::phase_state::{Phased, Unphased};
use crate::data::storage::GenotypeMatrix;
use crate::data::alignment::MarkerAlignment;
use crate::error::{Result, ReagleError};
use crate::io::bref3::RefPanelReader;
use crate::io::streaming::{PhasedOverlap, StreamingConfig, StreamingVcfReader};
use crate::io::vcf::{VcfWriter, ImputationQuality};
use crate::model::imp_ibs::{ClusterCodedSteps, ImpIbs};
use crate::model::imp_states_cluster::ImpStatesCluster;
use crate::model::imp_utils::*;
use crate::model::parameters::ModelParams;
use crate::model::pbwt_streaming::PbwtWavefront;
use crate::model::pbwt::PbwtState;
use crate::utils::workspace::ImpWorkspace;

/// Payload passed from Phasing (Producer) to Imputation (Consumer)
struct StreamingPayload {
    phased_target: GenotypeMatrix<Phased>,
    ref_window: GenotypeMatrix<Phased>,
    alignment: MarkerAlignment,
    output_start: usize,
    output_end: usize,
    window_idx: usize,
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
        let target_samples = target_reader.samples_arc();
        let n_target_samples = target_samples.len();
        let n_target_haps = n_target_samples * 2;

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
            "Streaming imputation: {} ref haplotypes, {} target samples",
            n_ref_haps, n_target_samples
        );

        // Create output writer
        let output_path = self.config.out.with_extension("vcf.gz");
        eprintln!("Writing output to {:?}", output_path);
        let mut writer = VcfWriter::create(&output_path, target_samples.clone())?;

        // Channel for streaming data
        let (tx, rx) = mpsc::sync_channel::<StreamingPayload>(8);

        // Clone config/params for producer
        let producer_config = self.config.clone();
        let producer_params = self.params.clone();
        let producer_maps = gen_maps.clone();

        // Spawn Producer (Phasing)
        let producer_handle = thread::spawn(move || -> Result<()> {
            let pipeline = crate::pipelines::ImputationPipeline {
                config: producer_config,
                params: producer_params,
            };

            // Re-open readers in thread
            let mut target_reader = StreamingVcfReader::open(
                &pipeline.config.gt,
                producer_maps.clone(),
                streaming_config.clone(),
            )?;
            
            let mut ref_reader: RefPanelReader = if is_bref3 {
                let stream_reader = crate::io::bref3::StreamingBref3Reader::open(&ref_path_clone)?;
                let window_config = crate::io::bref3::WindowConfig::default();
                let windowed = crate::io::bref3::WindowedBref3Reader::new(stream_reader, window_config);
                RefPanelReader::Bref3(windowed)
            } else {
                let (mut vcf_reader, vcf_file) = crate::io::vcf::VcfReader::open(&ref_path_clone)?;
                let ref_gt = Arc::new(vcf_reader.read_all(vcf_file)?.into_phased());
                RefPanelReader::InMemory(crate::io::bref3::InMemoryRefReader::new(ref_gt))
            };

            let mut window_count = 0;
            let mut phased_overlap: Option<PhasedOverlap> = None;
            let mut pbwt_state: Option<PbwtState> = None;

            eprintln!("Phase 1: Streaming phasing of target data...");
            
            while let Some(target_window) = target_reader.next_window()? {
                window_count += 1;
                let n_markers = target_window.genotypes.n_markers();

                eprintln!(
                    "  Phasing Window {} ({} markers, pos {}..{})",
                    window_count, n_markers,
                    target_window.genotypes.marker(MarkerIdx::new(0)).pos,
                    target_window.genotypes.marker(MarkerIdx::new((n_markers - 1) as u32)).pos
                );

                let start_pos = target_window.genotypes.marker(MarkerIdx::new(0)).pos;
                let end_pos = target_window.genotypes.marker(MarkerIdx::new((n_markers - 1) as u32)).pos;
                
                let ref_window_wrapper = match ref_reader.load_window_for_region(start_pos, end_pos)? {
                    Some(w) => w,
                    None => {
                        eprintln!("    Warning: No reference markers in region");
                        continue;
                    }
                };
                
                let ref_window_gt = ref_window_wrapper.genotypes;

                let alignment = match is_bref3 {
                    true => {
                        let mut ref_pos_map = std::collections::HashMap::new();
                        for m in 0..ref_window_gt.n_markers() {
                            let marker = ref_window_gt.marker(MarkerIdx::new(m as u32));
                            ref_pos_map.insert((marker.chrom.0, marker.pos), m);
                        }
                        MarkerAlignment::new_from_windows(
                            &target_window.genotypes,
                            &ref_window_gt,
                        )?
                    }
                    false => MarkerAlignment::new(
                        &target_window.genotypes,
                        &ref_window_gt,
                    ),
                };

                let phased = pipeline.phase_window_streaming(
                    &target_window.genotypes,
                    &ref_window_gt,
                    &alignment,
                    &producer_maps,
                    phased_overlap.as_ref(),
                    pbwt_state.as_ref(),
                )?;

                // Extract state for next window BEFORE moving phased to channel
                phased_overlap = Some(pipeline.extract_overlap_streaming(&phased, n_markers, target_window.output_end));
                pbwt_state = Some(pipeline.extractpbwt_state_streaming(&phased, n_markers));

                // Send to consumer
                if let Err(_) = tx.send(StreamingPayload {
                    phased_target: phased,
                    ref_window: ref_window_gt,
                    alignment,
                    output_start: target_window.output_start,
                    output_end: target_window.output_end,
                    window_idx: window_count,
                }) {
                    break; // Consumer hung up
                }
            }
            Ok(())
        });

        // Consumer (Imputation)
        eprintln!("Phase 2: Streaming imputation...");
        
        let mut imp_overlap: Option<PhasedOverlap> = None;
        let mut total_markers = 0;

        for payload in rx {
            let StreamingPayload { 
                phased_target, 
                ref_window, 
                alignment, 
                output_start, 
                output_end,
                window_idx
            } = payload;

            eprintln!("  Imputing Window {} ({} markers)", window_idx, phased_target.n_markers());

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

            self.run_imputation_window_streaming(
                &phased_target,
                &ref_window,
                &alignment,
                &gen_maps,
                imp_overlap.as_ref(),
                &mut window_quality,
                &mut writer,
                output_start,
                output_end,
            )?;

            total_markers += output_end - output_start;

            imp_overlap = Some(self.extract_imputed_overlap_streaming(
                &phased_target,
                &ref_window,
                output_end,
            ));
        }

        writer.flush()?;
        
        // Check producer result
        if let Err(e) = producer_handle.join() {
            std::panic::resume_unwind(e);
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
        let mut phasing = crate::pipelines::PhasingPipeline::new(self.config.clone());
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
        PhasedOverlap::new(overlap_size, n_haps, alleles)
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
        output_start: usize,
        output_end: usize,
    ) -> Result<()> {
        // Thread-local workspace - must be defined inside the parallel context
        thread_local! {
            static LOCAL_WORKSPACE: std::cell::RefCell<Option<ImpWorkspace>> =
                std::cell::RefCell::new(None);
        }

        let n_ref_markers = ref_win.n_markers();
        let n_target_samples = target_win.n_samples();
        let n_ref_haps = ref_win.n_haplotypes();

        let markers_to_process = if let Some(overlap) = imp_overlap {
            let start = overlap.n_markers;
            start..n_ref_markers
        } else {
            0..n_ref_markers
        };

        if markers_to_process.start >= markers_to_process.end {
            return Ok(());
        }

        let chrom = ref_win.marker(MarkerIdx::new(0)).chrom;
        let gen_positions: Vec<f64> = (0..n_ref_markers)
            .map(|m| {
                if m == 0 { 0.0 }
                else {
                    let pos1 = ref_win.marker(MarkerIdx::new((m - 1) as u32)).pos;
                    let pos2 = ref_win.marker(MarkerIdx::new(m as u32)).pos;
                    let dist = gen_maps.gen_dist(chrom, pos1, pos2);
                    dist.abs().max(crate::model::imp_utils::MIN_CM_DIST)
                }
            })
            .collect();

        let sample_genotyped_vec: Vec<Vec<usize>> = (0..n_target_samples)
            .map(|s| {
                (0..n_ref_markers)
                    .filter(|&ref_m| {
                        if let Some(target_m) = alignment.target_marker(ref_m) {
                            let marker_idx = MarkerIdx::new(target_m as u32);
                            let a1 = target_win.allele(marker_idx, HapIdx::new((s * 2) as u32));
                            let a2 = target_win.allele(marker_idx, HapIdx::new((s * 2 + 1) as u32));
                            a1 != 255 || a2 != 255
                        } else {
                            false
                        }
                    })
                    .collect()
            })
            .collect();

        const BATCH_SIZE: usize = 50;
        let n_batches = (n_target_samples + BATCH_SIZE - 1) / BATCH_SIZE;
        
        let mut all_results = Vec::new();

        for batch_idx in 0..n_batches {
            let batch_start = batch_idx * BATCH_SIZE;
            let batch_end = (batch_start + BATCH_SIZE).min(n_target_samples);
            let batch_samples: Vec<usize> = (batch_start..batch_end).collect();

            let batch_results: Vec<(usize, Vec<f32>, Vec<(u8, u8)> )> = batch_samples
                .par_iter()
                .map(|&s| {
                    let hap1_idx = HapIdx::new((s * 2) as u32);
                    let hap2_idx = HapIdx::new((s * 2 + 1) as u32);
                    let target_haps = [hap1_idx, hap2_idx];
                    let sample_genotyped = &sample_genotyped_vec[s];

                    if sample_genotyped.is_empty() {
                        return (s, vec![0.0f32; markers_to_process.len()], vec![(0u8, 0u8); markers_to_process.len()]);
                    }

                    let clusters = compute_marker_clusters_with_blocks(
                        sample_genotyped,
                        &gen_positions[markers_to_process.start..markers_to_process.end],
                        self.config.cluster as f64,
                        &[],
                    );

                    let n_clusters = clusters.len();
                    let cluster_bounds: Vec<(usize, usize)> = clusters.iter().map(|c| (c.start, c.end)).collect();

                    let cluster_midpoints: Vec<f64> = clusters.iter().map(|c| {
                            if c.end > c.start {
                                (gen_positions[markers_to_process.start + c.start]
                                    + gen_positions[markers_to_process.start + c.end - 1]) / 2.0
                            } else {
                                gen_positions[markers_to_process.start + c.start]
                            }
                        }).collect();

                    let cluster_p_recomb: Vec<f32> = std::iter::once(0.0f32)
                        .chain((1..n_clusters).map(|c| {
                            let gen_dist = (cluster_midpoints[c] - cluster_midpoints[c - 1]).abs();
                            self.params.p_recomb(gen_dist)
                        })).collect();

                    let cluster_seqs = crate::model::imp_ibs::build_cluster_hap_sequences_for_targets(
                        ref_win, target_win, alignment, sample_genotyped, &cluster_bounds, &target_haps,
                    );
                    let coded_steps = ClusterCodedSteps::from_cluster_sequences(
                        &cluster_seqs, &cluster_midpoints, self.config.imp_step as f64,
                    );
                    let imp_ibs = ImpIbs::new(
                        coded_steps, self.config.imp_nsteps, self.params.n_states,
                        n_ref_haps, target_haps.len(), self.config.seed as u64 + s as u64,
                    );

                    let (ref_cluster_start, ref_cluster_end) = compute_ref_cluster_bounds(sample_genotyped, &clusters);
                    let marker_cluster = Arc::new(build_marker_cluster_index(&ref_cluster_start, n_ref_markers));
                    let ref_cluster_end: Arc<Vec<usize>> = Arc::new(ref_cluster_end);
                    let cluster_weights = Arc::new(compute_cluster_weights(&gen_positions, &ref_cluster_start, &ref_cluster_end));

                    let (dosages, best_gt) = LOCAL_WORKSPACE.with(|cell| {
                        let mut ws_opt = cell.borrow_mut();
                        if ws_opt.is_none() { *ws_opt = Some(ImpWorkspace::new(n_ref_haps)); }
                        let ws = ws_opt.as_mut().unwrap();
                        ws.clear();

                        let mut hap_results: Vec<(Vec<f32>, Vec<(u8, u8)>)> = Vec::new();

                        for (local_h, &global_h) in target_haps.iter().enumerate() {
                            let mut imp_states = ImpStatesCluster::new(&imp_ibs, n_clusters, n_ref_haps, self.params.n_states);
                            let mut hap_indices: Vec<Vec<u32>> = Vec::new();
                            let actual_n_states = imp_states.ibs_states_cluster(local_h, &mut hap_indices);

                            let state_probs = compute_state_probs(
                                &hap_indices, &cluster_bounds, sample_genotyped, target_win, ref_win,
                                alignment, global_h.as_usize(), actual_n_states, ws, self.params.p_mismatch,
                                &cluster_p_recomb, marker_cluster.clone(), ref_cluster_end.clone(), cluster_weights.clone()
                            );

                            let mut hap_dosages = Vec::with_capacity(markers_to_process.len());
                            let mut hap_best_gt = Vec::with_capacity(markers_to_process.len());
                            for &ref_m in sample_genotyped.iter().skip_while(|&m| *m < markers_to_process.start) {
                                let p = state_probs.allele_posteriors(ref_m, 2, &|_, h| ref_win.allele(MarkerIdx::new(ref_m as u32), HapIdx::new(h as u32)));
                                hap_dosages.push(p.prob(1));
                                hap_best_gt.push(if p.max_allele() == 1 { (1, 0) } else { (0, 0) });
                            }

                            hap_results.push((hap_dosages, hap_best_gt));
                        }

                        let mut combined_dosages = Vec::new();
                        let mut combined_best_gt = Vec::new();

                        for m in 0..markers_to_process.len() {
                            combined_dosages.push(hap_results[0].0[m] + hap_results[1].0[m]);
                            combined_best_gt.push((hap_results[0].1[m].0, hap_results[1].1[m].0));
                        }

                        (combined_dosages, combined_best_gt)
                    });
                    (s, dosages, best_gt)
                }).collect();
            
            all_results.extend(batch_results);
        }
        
        // Sort all results by sample index for writing
        all_results.sort_by_key(|(s, _, _)| *s);

        self.write_imputed_window_streaming(
            ref_win, final_writer, window_quality, output_start, output_end,
            markers_to_process.start, &all_results,
        )?;
        
        Ok(())
    }
    
    fn extract_imputed_overlap_streaming(
        &self,
        target_win: &GenotypeMatrix<Phased>,
        ref_win: &GenotypeMatrix<Phased>,
        output_end: usize,
    ) -> PhasedOverlap {
        let overlap_size = 1000.min(ref_win.n_markers());
        let start = output_end.saturating_sub(overlap_size);
        let end = output_end;
        let n_haps = target_win.n_haplotypes();
        let mut alleles = vec![255u8; overlap_size * n_haps];
        for h in 0..n_haps {
            for (local_m, global_m) in (start..end).enumerate() {
                alleles[h * overlap_size + local_m] = target_win.allele(MarkerIdx::new(global_m as u32), HapIdx::new(h as u32));
            }
        }
        PhasedOverlap::new(overlap_size, n_haps, alleles)
    }

    /// Write imputed window results to VCF
    fn write_imputed_window_streaming(
        &self,
        ref_win: &GenotypeMatrix<Phased>,
        writer: &mut VcfWriter,
        quality: &ImputationQuality,
        output_start: usize,
        output_end: usize,
        markers_to_process_start: usize,
        all_results: &[(usize, Vec<f32>, Vec<(u8, u8)>)],
    ) -> Result<()> {
        let markers_range = output_start..output_end;
        let n_markers = markers_range.len();

        if n_markers == 0 || all_results.is_empty() {
            return Ok(());
        }

        // Build lookup maps for sample -> (dosages, best_gt)
        let sample_data: std::collections::HashMap<usize, (&Vec<f32>, &Vec<(u8, u8)>)> =
            all_results.iter().map(|(s, d, g)| (*s, (d, g))).collect();

        // Closure to get dosage: marker_idx is window-local ref marker index from VCF writer
        // Dosages array is indexed from 0 for markers starting at markers_to_process_start
        let get_dosage = |marker_idx: usize, sample_idx: usize| -> f32 {
            let local_m = marker_idx.saturating_sub(markers_to_process_start);
            if let Some((dosages, _)) = sample_data.get(&sample_idx) {
                dosages.get(local_m).copied().unwrap_or(0.0)
            } else {
                0.0
            }
        };

        // Closure to get best genotype
        let get_best_gt = |marker_idx: usize, sample_idx: usize| -> (u8, u8) {
            let local_m = marker_idx.saturating_sub(markers_to_process_start);
            if let Some((_, best_gt)) = sample_data.get(&sample_idx) {
                best_gt.get(local_m).copied().unwrap_or((0, 0))
            } else {
                (0, 0)
            }
        };

        writer.write_imputed_streaming(
            ref_win,
            get_dosage,
            get_best_gt,
            None::<fn(usize, usize) -> (crate::pipelines::imputation::AllelePosteriors, crate::pipelines::imputation::AllelePosteriors)>,
            quality,
            output_start,
            output_end,
            false, // include_gp
            false, // include_ap
        )
    }
}