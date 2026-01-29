import re

def main():
    with open('src/pipelines/phasing.rs', 'r') as f:
        content = f.read()

    # 1. Update the run call
    # Find the call to run_phase_baum_iteration_stage1
    # It ends with ", it)?;"
    # We want to insert ", &confidence_by_sample" before ", it)?;"
    # But matching the whole call is safer.
    
    pattern_call = r'(self\.run_phase_baum_iteration_stage1\s*\(\s*&target_gt,\s*&mut geno,\s*samples\.as_ref\(\),\s*&stage1_p_recomb,\s*&stage1_gen_dists,\s*&hi_freq_to_orig,\s*&hi_freq_gen_positions,\s*&stage1_blocks,\s*&ibs2,\s*&mut sample_phases,\s*&mut mcmc_paths,\s*atomic_estimates\.as_ref\(\),)(\s*it,\s*\)\?;)'
    
    # We use replacement that adds &confidence_by_sample
    new_call = r'\1\n                &confidence_by_sample,\2'
    
    content, count = re.subn(pattern_call, new_call, content)
    if count == 0:
        print("Warning: Could not update run_phase_baum_iteration_stage1 call site")
    else:
        print("Updated run_phase_baum_iteration_stage1 call site")

    # 2. Replace run_phase_baum_iteration_stage1 function
    # Find the start
    start_match = re.search(r'fn run_phase_baum_iteration_stage1\s*\(', content)
    if not start_match:
        print("Could not find start of run_phase_baum_iteration_stage1")
        return

    start_idx = start_match.start()
    
    # Find the end by brace counting
    open_braces = 0
    end_idx = -1
    for i in range(start_idx, len(content)):
        if content[i] == '{':
            open_braces += 1
        elif content[i] == '}':
            open_braces -= 1
            if open_braces == 0:
                end_idx = i + 1
                break
    
    if end_idx == -1:
        print("Could not find end of function (brace mismatch in file?)")
        # Try to find the start of the next function "fn build_final_matrix"
        next_fn = re.search(r'\s*/// Build final GenotypeMatrix', content[start_idx:])
        if next_fn:
            end_idx = start_idx + next_fn.start()
            print(f"Found next function start at {end_idx}, using as boundary")
        else:
            print("Could not recover function boundary.")
            return

    # New function body
    new_func = r'''fn run_phase_baum_iteration_stage1(
        &mut self,
        target_gt: &GenotypeMatrix,
        geno: &mut MutableGenotypes,
        samples: &Samples,
        stage1_p_recomb: &[f32],
        stage1_gen_dists: &[f64],
        hi_freq_to_orig: &[usize],
        hi_freq_gen_positions: &[f64],
        stage1_blocks: &[(usize, usize)],
        ibs2: &Ibs2,
        sample_phases: &mut [SamplePhase],
        mcmc_paths: &mut [Option<MosaicPaths>],
        atomic_estimates: Option<&crate::model::parameters::AtomicParamEstimates>,
        confidence_by_sample: &[Vec<f32>],
        iteration: usize,
    ) -> Result<()> {
        let n_stage1_blocks = stage1_blocks.len();
        if n_stage1_blocks == 0 {
            return Ok(());
        }
        let n_haps = geno.n_haps();

        // Compute total haplotype count (target + reference)
        let n_ref_haps = self
            .reference_gt
            .as_ref()
            .map(|r| r.n_haplotypes())
            .unwrap_or(0);
        let n_total_haps = n_haps + n_ref_haps;
        let n_samples = sample_phases.len();
        let n_hi_freq = hi_freq_to_orig.len();

        let n_candidates = self.params.n_states.min(n_total_haps).max(1);

        // Build composite haplotypes
        let (threaded_haps_vec, _) =
            if let (Some(ref_gt), Some(alignment)) = (&self.reference_gt, &self.alignment) {
                 let func = || {
                    self.build_composite_haps_streaming(
                        geno,
                        Some(ref_gt),
                        Some(alignment),
                        n_hi_freq,
                        n_total_haps,
                        n_samples,
                        ibs2,
                        n_candidates,
                        self.params.n_states,
                        None,
                        Some(hi_freq_to_orig),
                        hi_freq_gen_positions,
                        self.config.imp_step,
                    )
                 };
                if self.config.profile {
                    info_span!("phase_pbwt_build", markers = n_hi_freq, samples = n_samples)
                        .in_scope(func)
                } else {
                    func()
                }
            } else {
                 let func = || {
                     self.build_composite_haps_streaming_direct(
                        geno,
                        samples,
                        n_hi_freq,
                        n_samples,
                        ibs2,
                        n_candidates,
                        self.params.n_states,
                        None,
                        None,
                        hi_freq_gen_positions,
                        self.config.imp_step,
                    )
                 };
                if self.config.profile {
                    info_span!("phase_pbwt_build", markers = n_hi_freq, samples = n_samples).in_scope(func)
                } else {
                    func()
                }
            };

        // Build phase IBS if dynamic MCMC is enabled
        let phase_ibs = if self.config.dynamic_mcmc {
            Some(self.build_bidirectional_pbwt_subset(geno, hi_freq_to_orig, n_total_haps))
        } else {
            None
        };

        // No clone needed: the HMM phase is read-only; mutations happen after.
        // We use a scoped immutable borrow that ends before the apply phase.
        type PhaseDecision = (
            Vec<bool>,
            Vec<(usize, f32)>,
            Vec<(usize, f32)>,
            Option<MosaicPaths>,
        );
        let swap_results: Vec<PhaseDecision> =
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
                // Initialize with empty/default values - will be overwritten by parallel iter
                let mut swap_results: Vec<PhaseDecision> =
                    vec![(Vec::new(), Vec::new(), Vec::new(), None); n_samples];

                tracing::info_span!("hmm_samples").in_scope(|| {
                    swap_results
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(s, decision_out)| {
                            let sample_idx = SampleIdx::new(s as u32);
                            let hap1 = sample_idx.hap1();
                            let hap2 = sample_idx.hap2();
                            let sample_seed = (self.config.seed as u64)
                                .wrapping_add(s as u64)
                                .wrapping_add((iteration as u64) << 32)
                                .wrapping_add(0xA5A5_5A5A_D00Du64);

                            // Use pre-built composite haplotypes from streaming PBWT
                            let threaded_haps_full = &threaded_haps_vec[s];
                            let n_states_full = threaded_haps_full.n_states();
                            let threaded_haps = threaded_haps_full;
                            let n_states = n_states_full;

                            // 2. Extract current alleles for H1 and H2
                            // ref_geno.haplotype returns slice of all markers
                            let _seq1_full = ref_geno.haplotype(hap1);
                            let _seq2_full = ref_geno.haplotype(hap2);
                            let sample_conf_full = &confidence_by_sample[s];
                            let sp = &sample_phases[s];

                            // Collect EM statistics if requested (using hi-freq markers only)
                            let mut seq1_subset = Vec::with_capacity(n_hi_freq);
                            let mut seq2_subset = Vec::with_capacity(n_hi_freq);
                            let mut conf_subset = Vec::with_capacity(n_hi_freq);

                            for &m in hi_freq_to_orig {
                                seq1_subset.push(ref_geno.get(m, hap1));
                                seq2_subset.push(ref_geno.get(m, hap2));
                                conf_subset.push(sample_conf_full[m]);
                            }

                            if let Some(atomic) = atomic_estimates {
                                let hmm = MosaicHmm::new(
                                    ref_view,
                                    &self.params,
                                    n_states,
                                    stage1_p_recomb.to_vec(),
                                );
                                let mut local_est = crate::model::parameters::ParamEstimates::new();
                                hmm.collect_stats(&seq1_subset, threaded_haps, stage1_gen_dists, &mut local_est);
                                hmm.collect_stats(&seq2_subset, threaded_haps, stage1_gen_dists, &mut local_est);
                                atomic.add_estimation_data(&local_est);
                            }

                            // Identify UNPHASED heterozygote positions in hi-freq marker space
                            let het_positions: Vec<usize> = (0..n_hi_freq)
                                .filter(|&i| {
                                    let m = hi_freq_to_orig[i];
                                    let a1 = ref_geno.get(m, hap1);
                                    let a2 = ref_geno.get(m, hap2);
                                    a1 != 255 && a2 != 255 && a1 != a2 && sp.is_unphased(m)
                                })
                                .collect();

                            if het_positions.is_empty() {
                                *decision_out = (vec![false; n_hi_freq], Vec::new(), Vec::new(), None);
                                return;
                            }

                            let p_err = self.params.p_mismatch;
                            let p_no_err = 1.0 - p_err;

                            let (swap_bits, swap_lr, swap_probs, new_paths) = if self.config.dynamic_mcmc {
                                THREAD_WORKSPACE.with(|ws| {
                                    let mut workspace = ws.borrow_mut();
                                    if workspace.is_none() {
                                        *workspace =
                                            Some(crate::utils::workspace::ThreadWorkspace::new(64, 0));
                                    }
                                    let ws = workspace.as_mut().unwrap();
                                    let func = || {
                                        sample_dynamic_mcmc(
                                            n_hi_freq,
                                            n_states,
                                            stage1_p_recomb,
                                            &seq1_subset,
                                            &seq2_subset,
                                            &conf_subset,
                                            phase_ibs.as_ref().expect("phase_ibs"),
                                            ibs2,
                                            s as u32,
                                            &het_positions,
                                            sample_seed,
                                            self.config.mcmc_steps,
                                            p_no_err,
                                            p_err,
                                            prior_paths.get(s).and_then(|p| p.as_ref()),
                                            ws,
                                        )
                                    };
                                    if self.config.profile {
                                        info_span!("run_dynamic_mcmc", sample = s).in_scope(func)
                                    } else {
                                        func()
                                    }
                                })
                            } else {
                                THREAD_WORKSPACE.with(|ws| {
                                    let mut workspace = ws.borrow_mut();
                                    if workspace.is_none() {
                                        *workspace =
                                            Some(crate::utils::workspace::ThreadWorkspace::new(64, 0));
                                    }
                                    let ws = workspace.as_mut().unwrap();
                                    ws.clear();
                                    let lookup = if self.config.profile {
                                        info_span!("prep_allele_lookup", sample = s).in_scope(|| {
                                            RefAlleleLookup::new_from_threaded_with_buffer(
                                                threaded_haps,
                                                n_hi_freq,
                                                n_states,
                                                n_haps,
                                                ref_geno,
                                                self.reference_gt.as_deref(),
                                                self.alignment.as_ref(),
                                                Some(hi_freq_to_orig),
                                                std::mem::replace(
                                                    &mut ws.lookup,
                                                    aligned_vec::AVec::new(32),
                                                ),
                                            )
                                        })
                                    } else {
                                        RefAlleleLookup::new_from_threaded_with_buffer(
                                            threaded_haps,
                                            n_hi_freq,
                                            n_states,
                                            n_haps,
                                            ref_geno,
                                            self.reference_gt.as_deref(),
                                            self.alignment.as_ref(),
                                            Some(hi_freq_to_orig),
                                            std::mem::replace(&mut ws.lookup, aligned_vec::AVec::new(32)),
                                        )
                                    };

                                    let block_starts: Arc<[usize]> =
                                        blocks_to_starts(stage1_blocks, n_hi_freq)
                                            .into_boxed_slice()
                                            .into();
                                    let result = if self.config.profile {
                                        info_span!("run_mcmc_math", sample = s).in_scope(|| {
                                            sample_swap_bits_mosaic(
                                                n_hi_freq,
                                                n_states,
                                                stage1_p_recomb,
                                                &seq1_subset,
                                                &seq2_subset,
                                                &conf_subset,
                                                &lookup,
                                                Some(PlProvider {
                                                    gt: target_gt,
                                                    sample: s,
                                                    subset_to_orig: Some(hi_freq_to_orig),
                                                }),
                                                block_starts.clone(),
                                                &het_positions,
                                                prior_paths.get(s).and_then(|p| p.as_ref()),
                                                sample_seed,
                                                self.config.mcmc_burnin,
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
                                            &seq1_subset,
                                            &seq2_subset,
                                            &conf_subset,
                                            &lookup,
                                            Some(PlProvider {
                                                gt: target_gt,
                                                sample: s,
                                                subset_to_orig: Some(hi_freq_to_orig),
                                            }),
                                            block_starts,
                                            &het_positions,
                                            prior_paths.get(s).and_then(|p| p.as_ref()),
                                            sample_seed,
                                            self.config.mcmc_burnin,
                                            p_no_err,
                                            p_err,
                                            ws,
                                        )
                                    };
                                    ws.lookup = lookup.into_buffer();
                                    (result.0, result.1, result.2, Some(result.3))
                                });
                            };

                            let mut swap_mask = vec![false; n_hi_freq];
                            let mut current_phase = 0u8;
                            let mut phase_idx = 0usize;
                            for i in 0..n_hi_freq {
                                if phase_idx < het_positions.len() && het_positions[phase_idx] == i {
                                    current_phase = swap_bits.get(phase_idx).copied().unwrap_or(0);
                                    phase_idx += 1;
                                }
                                swap_mask[i] = current_phase == 1;
                            }

                            let het_lr_values: Vec<(usize, f32)> = het_positions
                                .iter()
                                .copied()
                                .zip(swap_lr.into_iter())
                                .collect();
                            let het_phase_values: Vec<(usize, f32)> = het_positions
                                .iter()
                                .copied()
                                .zip(swap_probs.into_iter())
                                .collect();

                            if let Some(bb) = self.telemetry.as_ref() {
                                bb.add_samples(1);
                            }

                            *decision_out = (swap_mask, het_lr_values, het_phase_values, new_paths);
                        })
                });

                swap_results
            }); // ref_geno borrow ends here

        // Apply phase decisions to SamplePhase
        let mut total_switches = 0;
        let mut total_phased = 0;

        let is_burnin = iteration < self.config.burnin;
        let lr_threshold = self.params.lr_threshold;

        for (s, (swap_mask, het_lr_values, het_phase_values, new_paths)) in
            swap_results.into_iter().enumerate()
        {
            let sp = &mut sample_phases[s];

            for (hi_freq_idx, &should_swap) in swap_mask.iter().enumerate() {
                if should_swap {
                    let m = hi_freq_to_orig[hi_freq_idx];
                    sp.swap_alleles(m);
                    total_switches += 1;
                }
            }

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
                 if s == 0 && m == 0 {
                     println!("Setting confidence for s=0 m=0 to {}", p_orient);
                }
                sp.set_phase_confidence(m, p_orient);
            }

            if let Some(paths) = new_paths {
                if let Some(slot) = mcmc_paths.get_mut(s) {
                    *slot = Some(paths);
                }
            }
        }

        self.sync_sample_phases_to_geno(sample_phases, geno);

        eprintln!(
            "Applied {} phase switches, {} markers phased (Stage 1 FB)",
            total_switches, total_phased
        );
        Ok(())
    }'''
    
    # Replace
    content = content[:start_idx] + new_func + content[end_idx:]
    
    with open('src/pipelines/phasing.rs', 'w') as f:
        f.write(content)

    print("Replaced run_phase_baum_iteration_stage1")

main()
