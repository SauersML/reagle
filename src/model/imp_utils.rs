//! Imputation utilities and HMM logic.

use crate::data::alignment::MarkerAlignment;
use crate::data::storage::phase_state::Phased;
use crate::data::storage::{GenotypeColumn, GenotypeMatrix};
use crate::data::{HapIdx, MarkerIdx};
use crate::model::pl_emission::{
    PlProvider, allele_probs_cond_from_pl, allele_probs_uncond_from_pl,
};
use crate::pipelines::imputation::ClusterStateProbs;
use crate::utils::workspace::ImpWorkspace;
use aligned_vec::{AVec, ConstAlign};
use std::sync::Arc; // Assuming we keep it there or import it

/// Minimum genetic distance between markers
pub const MIN_CM_DIST: f64 = 1e-7;

#[derive(Clone, Debug)]
pub struct MarkerCluster {
    pub start: usize,
    pub end: usize,
}

pub fn compute_marker_clusters(
    genotyped_markers: &[usize],
    gen_positions: &[f64],
    cluster_dist: f64,
) -> Vec<MarkerCluster> {
    if genotyped_markers.is_empty() {
        return Vec::new();
    }

    let mut clusters = Vec::new();
    let mut cluster_start = 0;
    let mut start_pos = gen_positions[genotyped_markers[0]];

    for m in 1..genotyped_markers.len() {
        let pos = gen_positions[genotyped_markers[m]];
        if pos - start_pos > cluster_dist {
            clusters.push(MarkerCluster {
                start: cluster_start,
                end: m,
            });
            cluster_start = m;
            start_pos = pos;
        }
    }

    clusters.push(MarkerCluster {
        start: cluster_start,
        end: genotyped_markers.len(),
    });

    clusters
}

pub fn compute_ref_cluster_bounds(
    genotyped_markers: &[usize],
    clusters: &[MarkerCluster],
) -> (Vec<usize>, Vec<usize>) {
    let mut starts = Vec::with_capacity(clusters.len());
    let mut ends = Vec::with_capacity(clusters.len());
    for cluster in clusters {
        let start = genotyped_markers[cluster.start];
        let end = genotyped_markers[cluster.end - 1] + 1;
        starts.push(start);
        ends.push(end);
    }
    (starts, ends)
}

pub fn build_marker_cluster_index(ref_cluster_start: &[usize], n_ref_markers: usize) -> Vec<usize> {
    let mut marker_cluster = vec![0usize; n_ref_markers];
    if ref_cluster_start.is_empty() {
        return marker_cluster;
    }
    let mut c = 0usize;
    for m in 0..n_ref_markers {
        while c + 1 < ref_cluster_start.len() && m >= ref_cluster_start[c + 1] {
            c += 1;
        }
        marker_cluster[m] = c;
    }
    marker_cluster
}

#[inline]
pub fn get_log_probs(conf: f32, p_err: f32) -> (f32, f32) {
    let p_no_err = 1.0 - p_err;
    let half_compl = (1.0 - conf) * 0.5;
    let match_prob = conf * p_no_err + half_compl;
    let mismatch_prob = conf * p_err + half_compl;
    (match_prob.ln(), mismatch_prob.ln())
}

#[allow(clippy::too_many_lines)]
#[allow(clippy::too_many_lines)]
pub fn compute_cluster_mismatches_into_workspace(
    hap_indices: &[Vec<u32>],
    cluster_bounds: &[(usize, usize)],
    genotyped_markers: &[usize],
    target_gt: &GenotypeMatrix<Phased>,
    ref_gt: &GenotypeMatrix<Phased>,
    alignment: &MarkerAlignment,
    geno_a1: &[u8],
    geno_a2: &[u8],
    targ_alleles: &[u8],
    partner_alleles: Option<&[u8]>,
    pl_provider: Option<&PlProvider>,
    sample_idx: usize,
    n_states: usize,
    workspace: &mut ImpWorkspace,
    base_err_rate: f32,
    trace: bool,
) {
    // wide imports removed as we use scalar unrolled loop

    let span = if trace {
        Some(tracing::info_span!("mismatch_precalc").entered())
    } else {
        None
    };
    let _ = &span;
    workspace.reset_and_ensure_capacity(hap_indices.len(), n_states);

    let n_clusters = hap_indices.len();
    let p_err = base_err_rate.clamp(1e-8, 0.5);

    let mut printed_hmm_trace = false;

    for (c, &(start, end)) in cluster_bounds.iter().enumerate() {
        if c >= n_clusters {
            break;
        }

        let row_buffer = &mut workspace.row_buffer;
        // Optimization: No need to fill row_buffer with 0.0 if we overwrite it completely.
        // But we must handle the case where n_states is not a multiple of 8 carefully
        // or just fill the tail. To be safe/simple, we can keeping fill(0.0) or handle overwrite correctly.
        // Our optimized loop "stores" accumulated values, so it overwrites.
        // We just need to make sure we visit every state.

        let mut cluster_base_score = 0.0f32;

        // --- Pre-calculate marker constants for the whole cluster ---
        // This avoids repeated lookups inside the state loop.
        // We store them in a small aligned buffer on the stack.
        // Max cluster size is usually small (e.g. < 100 markers).
        // If it's huge, we might need a heap vec, but stack is fast.
        // Let's use a Vec for safety but reuse it if we wanted (workspace?)
        // Allocating a Vec per cluster is efficient for typical cluster sizes.

        struct MarkerProps {
            target_m: usize,
            geno1: u8,
            geno2: u8,
            targ_allele: u8,
            partner_allele: u8,
            log_diff: f32,
            hard_log_diff: f32,
            // log_match removed (unused)
            ref_marker_idx: MarkerIdx,
            map_alleles: bool,
        }

        // We can't allocate inside the hot loop easily without overhead,
        // but populating a vector "markers_in_cluster" O(N_markers) is much cheaper
        // than N_states * N_markers lookups.

        let mut active_markers = Vec::with_capacity(end - start);
        let mut allele_probs: Vec<f32> = Vec::new();

        for &ref_m in &genotyped_markers[start..end] {
            let target_m_idx = alignment.ref_to_target.get(ref_m).copied().unwrap_or(-1);
            if target_m_idx < 0 {
                continue;
            }
            let target_m = target_m_idx as usize;

            let geno1 = geno_a1[target_m];
            let geno2 = geno_a2[target_m];
            if geno1 == 255 || geno2 == 255 {
                continue;
            }

            let targ_allele = targ_alleles[target_m];
            let partner_allele = partner_alleles.map(|p| p[target_m]).unwrap_or(255);

            let target_marker_idx = MarkerIdx::new(target_m as u32);
            let confidence = target_gt.sample_confidence_f32(target_marker_idx, sample_idx);
            if confidence <= 0.0 {
                continue;
            }

            // Prefer PL-derived emission when available and biallelic.
            let log_match: f32;
            let log_mism: f32;
            if let Some(plp) = pl_provider {
                let pl = plp.pl(target_m).filter(|v| !v.is_empty());
                if let Some(pl) = pl {
                    // Only do exact PL-based emissions for biallelic (0/1).
                    // For multi-allelic, fall back to confidence-based approximation.
                    let partner = partner_allele;
                    let maybe_n = if partner != 255 {
                        allele_probs_cond_from_pl(pl, partner, &mut allele_probs)
                    } else {
                        allele_probs_uncond_from_pl(pl, &mut allele_probs)
                    };
                    if maybe_n == Some(2) {
                        let req = if partner != 255 {
                            if partner == geno1 {
                                geno2
                            } else if partner == geno2 {
                                geno1
                            } else {
                                255
                            }
                        } else {
                            targ_allele
                        };
                        if req < 2 {
                            let p_req = allele_probs
                                .get(req as usize)
                                .copied()
                                .unwrap_or(0.0)
                                .clamp(0.0, 1.0);
                            let p_no_err = 1.0 - p_err;
                            // For biallelic, the "other" probability is 1 - p_req.
                            let emit_match = (p_no_err * p_req + p_err * (1.0 - p_req)).max(1e-30);
                            let emit_mism = (p_no_err * (1.0 - p_req) + p_err * p_req).max(1e-30);
                            log_match = emit_match.ln();
                            log_mism = emit_mism.ln();
                        } else {
                            (log_match, log_mism) = get_log_probs(confidence, p_err);
                        }
                    } else {
                        (log_match, log_mism) = get_log_probs(confidence, p_err);
                    }
                } else {
                    (log_match, log_mism) = get_log_probs(confidence, p_err);
                }
            } else {
                (log_match, log_mism) = get_log_probs(confidence, p_err);
            }

            let log_diff = log_mism - log_match;
            let hard_log_mism = (1e-9f32).ln();
            let hard_log_diff = hard_log_mism - log_match;

            if !printed_hmm_trace
                && c == 0
                && geno1 != geno2
                && geno1 != 255
                && geno2 != 255
                && !workspace.cluster_base_scores.is_empty()
            {
                printed_hmm_trace = true;
                eprintln!(
                    "\n[HMM TRACE] ref_marker={} target_m={} target_het={}/{}",
                    ref_m, target_m, geno1, geno2
                );
                eprintln!(
                    "  Conf={:.4} LogMatch={:.4} LogMism={:.4} LogDiff={:.4}",
                    confidence, log_match, log_mism, log_diff
                );

                let ref_col = ref_gt.column(MarkerIdx::new(ref_m as u32));
                let mut ref_counts = [0usize; 3];
                for k in 0..100.min(n_states) {
                    let hap_idx = hap_indices[c][k];
                    let ref_raw = ref_col.get(HapIdx::new(hap_idx));
                    let mapped = if alignment.has_allele_mapping(target_m) {
                        alignment.reverse_map_allele(target_m, ref_raw)
                    } else {
                        ref_raw
                    };
                    if mapped == 0 {
                        ref_counts[0] += 1;
                    } else if mapped == 1 {
                        ref_counts[1] += 1;
                    } else {
                        ref_counts[2] += 1;
                    }
                }
                eprintln!(
                    "  Ref State Sample (first 100): 0s={}, 1s={}, other/missing={}",
                    ref_counts[0], ref_counts[1], ref_counts[2]
                );
                if ref_counts[1] == 0 {
                    eprintln!(
                        "  --> ALARM: Target is HET (has allele 1), but reference states have ZERO mapped 1s"
                    );
                }
            }

            cluster_base_score += log_match;

            active_markers.push(MarkerProps {
                target_m,
                geno1,
                geno2,
                targ_allele,
                partner_allele,
                log_diff,
                hard_log_diff,
                // log_match: 0.0,
                ref_marker_idx: MarkerIdx::new(ref_m as u32),
                map_alleles: alignment.has_allele_mapping(target_m),
            });
        }

        workspace.cluster_base_scores.push(cluster_base_score);

        // --- Inverted Loop: Iterate State Blocks (Chunks of 8) ---

        let mut j_base = 0;
        // chunk_size = 8 (implicit)

        // Helper to get 8 ref alleles efficiently
        // We have to inspect the column type.
        // Ideally we would inspect column type ONCE per marker, not per state block.
        // But the column type is attached to the marker.
        // So inside the marker loop (which is inner), we dispatch.
        // Dispatching enums 8 at a time is better than 1 at a time.

        while j_base < n_states {
            let remainder = n_states - j_base;

            if remainder >= 8 {
                let indices: [usize; 8] = [
                    hap_indices[c][j_base] as usize,
                    hap_indices[c][j_base + 1] as usize,
                    hap_indices[c][j_base + 2] as usize,
                    hap_indices[c][j_base + 3] as usize,
                    hap_indices[c][j_base + 4] as usize,
                    hap_indices[c][j_base + 5] as usize,
                    hap_indices[c][j_base + 6] as usize,
                    hap_indices[c][j_base + 7] as usize,
                ];

                let mut acc = [0.0f32; 8];

                for m_props in &active_markers {
                    let ref_column = ref_gt.column(m_props.ref_marker_idx);

                    let mut ref_alleles_arr = [0u8; 8];
                    match ref_column {
                        GenotypeColumn::Dense(col) => {
                            for k in 0..8 {
                                ref_alleles_arr[k] = col.get(HapIdx::new(indices[k] as u32));
                            }
                        }
                        GenotypeColumn::Sparse(col) => {
                            for k in 0..8 {
                                ref_alleles_arr[k] = col.get(HapIdx::new(indices[k] as u32));
                            }
                        }
                        GenotypeColumn::Dictionary(col, offset) => {
                            for k in 0..8 {
                                ref_alleles_arr[k] =
                                    col.get(*offset, HapIdx::new(indices[k] as u32));
                            }
                        }
                        GenotypeColumn::SeqCoded(col) => {
                            for k in 0..8 {
                                ref_alleles_arr[k] = col.get(HapIdx::new(indices[k] as u32));
                            }
                        }
                    }

                    for k in 0..8 {
                        let ref_allele = ref_alleles_arr[k];
                        let final_ref = if m_props.map_alleles {
                            alignment.reverse_map_allele(m_props.target_m, ref_allele)
                        } else {
                            ref_allele
                        };

                        // Logic identical to scalar, but unrolled
                        if final_ref == 255 {
                            if ref_allele != 255 {
                                acc[k] += m_props.log_diff;
                            }
                        } else if m_props.partner_allele != 255 {
                            let required = if m_props.partner_allele == m_props.geno1 {
                                m_props.geno2
                            } else if m_props.partner_allele == m_props.geno2 {
                                m_props.geno1
                            } else {
                                255
                            };
                            if required != 255 {
                                if final_ref != required {
                                    acc[k] += m_props.hard_log_diff;
                                }
                            } else if m_props.targ_allele != 255 && final_ref != m_props.targ_allele
                            {
                                acc[k] += m_props.log_diff;
                            }
                        } else if m_props.targ_allele != 255 && final_ref != m_props.targ_allele {
                            acc[k] += m_props.log_diff;
                        }
                    }
                }

                // Write accumulators to row_buffer
                row_buffer[j_base..j_base + 8].copy_from_slice(&acc);

                j_base += 8;
            } else {
                // Scalar tail loop
                for j in j_base..n_states {
                    let hap_idx = hap_indices[c][j];
                    let mut acc_penalty = 0.0;

                    for m_props in &active_markers {
                        let ref_col = ref_gt.column(m_props.ref_marker_idx);
                        let ref_allele = ref_col.get(HapIdx::new(hap_idx));

                        let final_ref = if m_props.map_alleles {
                            alignment.reverse_map_allele(m_props.target_m, ref_allele)
                        } else {
                            ref_allele
                        };

                        if final_ref == 255 {
                            if ref_allele != 255 {
                                acc_penalty += m_props.log_diff;
                            }
                        } else if m_props.partner_allele != 255 {
                            let required = if m_props.partner_allele == m_props.geno1 {
                                m_props.geno2
                            } else if m_props.partner_allele == m_props.geno2 {
                                m_props.geno1
                            } else {
                                255
                            };
                            if required != 255 {
                                if final_ref != required {
                                    acc_penalty += m_props.hard_log_diff;
                                }
                            } else if m_props.targ_allele != 255 && final_ref != m_props.targ_allele
                            {
                                acc_penalty += m_props.log_diff;
                            }
                        } else if m_props.targ_allele != 255 && final_ref != m_props.targ_allele {
                            acc_penalty += m_props.log_diff;
                        }
                    }
                    row_buffer[j] = acc_penalty;
                }
                j_base = n_states; // Done
            }
        }

        // --- Populate sparse outputs from row_buffer ---
        // Optimization: We could have written directly to sparse vectors above,
        // but populating row_buffer first is safer and keeps logic separate.
        // We can optimize this later if needed.

        for (j, &val) in row_buffer.iter().enumerate().take(n_states) {
            if val.abs() > 1e-9 {
                workspace.diff_vals.push(val);
                workspace.diff_cols.push(j as u16);
            }
        }
        workspace.diff_row_offsets.push(workspace.diff_vals.len());
    }
}

pub fn run_hmm_forward_backward_to_sparse(
    diff_vals: &[f32],
    diff_cols: &[u16],
    diff_row_offsets: &[usize],
    cluster_base_scores: &[f32],
    p_recomb: &[f32],
    n_states: usize,
    hap_indices_input: &[Vec<u32>],
    prior_probs: Option<&[f32]>,
    threshold: f32,
    fwd_buffer: &mut AVec<f32, ConstAlign<32>>,
    bwd_buffer: &mut AVec<f32, ConstAlign<32>>,
    block_fwd_buffer: &mut AVec<f32, ConstAlign<32>>,
    trace: bool,
) -> (Vec<usize>, Vec<u32>, Vec<f32>, Vec<f32>) {
    use wide::f32x8;

    let n_clusters = cluster_base_scores.len();
    if n_clusters == 0 {
        return (vec![0], Vec::new(), Vec::new(), Vec::new());
    }

    // Prevent exp underflow in long windows (matches legacy -80.0 log-floor)
    const LOG_EMIT_FLOOR: f32 = -80.0;

    const CHECKPOINT_INTERVAL: usize = 64;
    let n_checkpoints = (n_clusters + CHECKPOINT_INTERVAL - 1) / CHECKPOINT_INTERVAL;

    let fwd = fwd_buffer;
    fwd.resize(n_checkpoints * n_states + 2 * n_states, 0.0);
    let curr_base = n_checkpoints * n_states;
    let prev_base = curr_base + n_states;

    let mut fwd_sums = vec![1.0f32; n_clusters];
    let mut last_sum = 1.0f32;

    {
        let fwd_span = if trace {
            Some(tracing::info_span!("hmm_fwd_initial").entered())
        } else {
            None
        };
        let _ = &fwd_span;

        for m in 0..n_clusters {
            let p_rec = p_recomb.get(m).copied().unwrap_or(0.0);
            let shift = p_rec / n_states as f32;
            let scale = (1.0 - p_rec) / last_sum.max(1e-30);

            let (curr_off, _) = if m % 2 == 0 {
                (curr_base, prev_base)
            } else {
                (prev_base, curr_base)
            };

            let base_emit = cluster_base_scores[m].max(LOG_EMIT_FLOOR).exp();

            if m == 0 {
                if let Some(priors) = prior_probs {
                    let mut sum = 0.0f32;
                    for k in 0..n_states {
                        let prior = priors.get(k).copied().unwrap_or(1.0 / n_states as f32);
                        let val = base_emit * prior;
                        fwd[curr_off + k] = val;
                        sum += val;
                    }
                    if sum <= 0.0 {
                        let val = base_emit / n_states as f32;
                        fwd[curr_off..curr_off + n_states].fill(val);
                    }
                } else {
                    let val = base_emit / n_states as f32;
                    fwd[curr_off..curr_off + n_states].fill(val);
                }
            } else {
                let (lower, upper) = fwd.split_at_mut(prev_base);
                let (curr_slice, prev_slice) = if m % 2 == 0 {
                    (
                        &mut lower[curr_base..curr_base + n_states],
                        &upper[..n_states],
                    )
                } else {
                    (
                        &mut upper[..n_states],
                        &lower[curr_base..curr_base + n_states],
                    )
                };

                let shift_vec = f32x8::splat(shift);
                let scale_vec = f32x8::splat(scale);
                let emit_vec = f32x8::splat(base_emit);

                let mut k = 0;
                while k + 8 <= n_states {
                    let prev_chunk_arr: &[f32; 8] = prev_slice[k..k + 8].try_into().unwrap();
                    let prev_vec = f32x8::from(*prev_chunk_arr);
                    let trans = prev_vec.mul_add(scale_vec, shift_vec);
                    let res = trans * emit_vec;
                    let res_arr: [f32; 8] = res.into();
                    curr_slice[k..k + 8].copy_from_slice(&res_arr);
                    k += 8;
                }
                for i in k..n_states {
                    let p = prev_slice[i];
                    curr_slice[i] = base_emit * (scale * p + shift);
                }
            }

            let curr_slice = &mut fwd[curr_off..curr_off + n_states];
            let start = diff_row_offsets[m];
            let end = diff_row_offsets[m + 1];

            {
                let exp_span = if trace && m == 0 {
                    Some(tracing::info_span!("expensive_float_exp").entered())
                } else {
                    None
                };
                let _ = &exp_span;
                for i in start..end {
                    let col = diff_cols[i] as usize;
                    let val = diff_vals[i];
                    if col < n_states {
                        let penalty = val.exp();
                        curr_slice[col] = curr_slice[col] * penalty;
                    }
                }
            }

            let mut new_sum = 0.0f32;
            let mut k = 0;
            let mut sum_vec = f32x8::splat(0.0);
            while k + 8 <= n_states {
                let chunk_arr: &[f32; 8] = curr_slice[k..k + 8].try_into().unwrap();
                let chunk = f32x8::from(*chunk_arr);
                sum_vec += chunk;
                k += 8;
            }
            new_sum += sum_vec.reduce_add();
            for &x in &curr_slice[k..n_states] {
                new_sum += x;
            }

            fwd_sums[m] = new_sum;
            last_sum = new_sum;

            if (m + 1) % CHECKPOINT_INTERVAL == 0 {
                let cp_idx = ((m + 1) / CHECKPOINT_INTERVAL - 1) * n_states;
                let inv_sum = if new_sum > 1e-30 { 1.0 / new_sum } else { 0.0 };
                let (checkpoints, working) = fwd.split_at_mut(curr_base);
                let src_off = if m % 2 == 0 { 0 } else { n_states };
                let src = &working[src_off..src_off + n_states];
                for (i, &x) in src.iter().enumerate() {
                    checkpoints[cp_idx + i] = x * inv_sum;
                }
            }
        }
    } // End fwd_span block

    let block_fwd = block_fwd_buffer;
    block_fwd.resize((CHECKPOINT_INTERVAL + 1) * n_states, 0.0);

    let bwd = bwd_buffer;
    bwd.resize(n_states, 0.0);
    bwd.fill(1.0 / n_states as f32);

    let estimated_nnz = n_clusters * 50;
    let mut hap_indices = Vec::with_capacity(estimated_nnz);
    let mut probs = Vec::with_capacity(estimated_nnz);
    let mut probs_p1 = Vec::with_capacity(estimated_nnz);
    let mut entry_counts = Vec::with_capacity(n_clusters);
    let mut curr_posteriors = vec![0.0f32; n_states];
    let mut next_posteriors = vec![0.0f32; n_states];

    for block_idx in (0..n_checkpoints).rev() {
        let block_start = block_idx * CHECKPOINT_INTERVAL;
        let block_end = ((block_idx + 1) * CHECKPOINT_INTERVAL).min(n_clusters);

        if block_start >= n_clusters {
            continue;
        }

        let mut recomp_sum;
        let mut curr_off;

        {
            let recomp_span = if trace {
                Some(tracing::info_span!("hmm_recompute_block").entered())
            } else {
                None
            };
            let _ = &recomp_span;

            if block_idx == 0 {
                let base_emit = cluster_base_scores[0].max(LOG_EMIT_FLOOR).exp();
                let val = base_emit / n_states as f32;
                block_fwd[0..n_states].fill(val);

                let start = diff_row_offsets[0];
                let end = diff_row_offsets[1];
                for i in start..end {
                    let col = diff_cols[i] as usize;
                    let val = diff_vals[i];
                    if col < n_states {
                        let penalty = val.exp();
                        block_fwd[col] = block_fwd[col] * penalty;
                    }
                }

                let mut sum = 0.0f32;
                for &x in &block_fwd[0..n_states] {
                    sum += x;
                }
                recomp_sum = sum.max(1e-30);
                curr_off = 0;
            } else {
                let load_idx = block_idx - 1;
                let checkpoint_off = load_idx * n_states;
                block_fwd[0..n_states]
                    .copy_from_slice(&fwd[checkpoint_off..checkpoint_off + n_states]);
                recomp_sum = 1.0;
                curr_off = 0;
            }

            let loop_start = if block_idx == 0 {
                block_start + 1
            } else {
                block_start
            };
            for local_m in loop_start..block_end {
                let p_rec = p_recomb.get(local_m).copied().unwrap_or(0.0);
                let shift = p_rec / n_states as f32;
                let scale = (1.0 - p_rec) / recomp_sum.max(1e-30);
                let base_emit = cluster_base_scores[local_m].max(LOG_EMIT_FLOOR).exp();

                let next_off = curr_off + n_states;
                // Use split_at_mut to satisfy the borrow checker for non-overlapping slices
                let (before, after) = block_fwd.split_at_mut(next_off);
                let prev_slice = &before[curr_off..curr_off + n_states];
                let curr_slice = &mut after[0..n_states];

                let shift_vec = f32x8::splat(shift);
                let scale_vec = f32x8::splat(scale);
                let emit_vec = f32x8::splat(base_emit);

                let mut k = 0;
                while k + 8 <= n_states {
                    let prev_chunk_arr: &[f32; 8] = prev_slice[k..k + 8].try_into().unwrap();
                    let prev_vec = f32x8::from(*prev_chunk_arr);
                    let trans = prev_vec.mul_add(scale_vec, shift_vec);
                    let res = trans * emit_vec;
                    let res_arr: [f32; 8] = res.into();
                    curr_slice[k..k + 8].copy_from_slice(&res_arr);
                    k += 8;
                }

                for i in k..n_states {
                    let p = prev_slice[i];
                    curr_slice[i] = base_emit * (scale * p + shift);
                }

                let start = diff_row_offsets[local_m];
                let end = diff_row_offsets[local_m + 1];
                for i in start..end {
                    let col = diff_cols[i] as usize;
                    let val = diff_vals[i];
                    if col < n_states {
                        let penalty = val.exp();
                        curr_slice[col] = curr_slice[col] * penalty;
                    }
                }

                let mut new_sum = 0.0f32;
                for x in curr_slice.iter() {
                    new_sum += *x;
                }
                recomp_sum = new_sum.max(1e-30);
                curr_off = next_off;
            }
        } // End recomp_span block

        {
            let bwd_span = if trace {
                Some(tracing::info_span!("hmm_bwd_sparse").entered())
            } else {
                None
            };
            let _ = &bwd_span;

            for m in (block_start..block_end).rev() {
                if m + 1 < n_clusters {
                    let p_rec = p_recomb.get(m + 1).copied().unwrap_or(0.0);
                    let shift = p_rec / n_states as f32;
                    let base_emit = cluster_base_scores[m + 1].max(LOG_EMIT_FLOOR).exp();

                    let mut k = 0;
                    let base_emit_vec = f32x8::splat(base_emit);
                    while k + 8 <= n_states {
                        let initial_chunk_arr: &[f32; 8] = bwd[k..k + 8].try_into().unwrap();
                        let initial_chunk = f32x8::from(*initial_chunk_arr);
                        let res = initial_chunk * base_emit_vec;
                        let res_arr: [f32; 8] = res.into();
                        bwd[k..k + 8].copy_from_slice(&res_arr);
                        k += 8;
                    }
                    for x in bwd[k..].iter_mut() {
                        *x *= base_emit;
                    }

                    let start = diff_row_offsets[m + 1];
                    let end = diff_row_offsets[m + 2];
                    for i in start..end {
                        let col = diff_cols[i] as usize;
                        let val = diff_vals[i];
                        if col < n_states {
                            let penalty = val.exp();
                            bwd[col] = bwd[col] * penalty;
                        }
                    }

                    let mut emitted_sum = 0.0f32;
                    let mut sum_vec = f32x8::splat(0.0);
                    k = 0;
                    while k + 8 <= n_states {
                        let chunk_arr: &[f32; 8] = bwd[k..k + 8].try_into().unwrap();
                        let chunk = f32x8::from(*chunk_arr);
                        sum_vec += chunk;
                        k += 8;
                    }
                    emitted_sum += sum_vec.reduce_add();
                    for &x in bwd[k..].iter() {
                        emitted_sum += x;
                    }

                    if emitted_sum > 0.0 {
                        let scale_v = (1.0 - p_rec) / emitted_sum;
                        let scale_vec = f32x8::splat(scale_v);
                        let shift_vec = f32x8::splat(shift);
                        k = 0;
                        while k + 8 <= n_states {
                            let chunk_arr: &[f32; 8] = bwd[k..k + 8].try_into().unwrap();
                            let chunk = f32x8::from(*chunk_arr);
                            let res = chunk.mul_add(scale_vec, shift_vec);
                            let res_arr: [f32; 8] = res.into();
                            bwd[k..k + 8].copy_from_slice(&res_arr);
                            k += 8;
                        }
                        for x in bwd[k..].iter_mut() {
                            *x = scale_v * *x + shift;
                        }
                    } else {
                        bwd.fill(1.0 / n_states as f32);
                    }
                }

                let local_offset = if block_idx == 0 {
                    (m - block_start) * n_states
                } else {
                    (m - block_start + 1) * n_states
                };
                let fwd_row = &block_fwd[local_offset..local_offset + n_states];

                let mut state_sum = 0.0f32;
                for k in 0..n_states {
                    curr_posteriors[k] = fwd_row[k] * bwd[k];
                    state_sum += curr_posteriors[k];
                }
                if state_sum > 0.0 {
                    let inv = 1.0 / state_sum;
                    for k in 0..n_states {
                        curr_posteriors[k] *= inv;
                    }
                }

                let entries_before = hap_indices.len();
                if m == n_clusters - 1 {
                    for k in 0..n_states {
                        let prob = curr_posteriors[k];
                        if prob > threshold {
                            hap_indices.push(hap_indices_input[m][k]);
                            probs.push(prob);
                            probs_p1.push(prob);
                        }
                    }
                } else {
                    for k in 0..n_states {
                        let prob = curr_posteriors[k];
                        let prob_next = next_posteriors[k];
                        if prob > threshold || prob_next > threshold {
                            hap_indices.push(hap_indices_input[m][k]);
                            probs.push(prob);
                            probs_p1.push(prob_next);
                        }
                    }
                }

                entry_counts.push(hap_indices.len() - entries_before);
                std::mem::swap(&mut curr_posteriors, &mut next_posteriors);
            }
        } // End bwd_span block
    }

    entry_counts.reverse();
    hap_indices.reverse();
    probs.reverse();
    probs_p1.reverse();

    let mut offsets = Vec::with_capacity(n_clusters + 1);
    offsets.push(0);
    let mut cumsum = 0;
    for &count in &entry_counts {
        cumsum += count;
        offsets.push(cumsum);
    }

    (offsets, hap_indices, probs, probs_p1)
}

/// Computes state probabilities using HMM.
///
/// Replaces the constant stub with actual HMM logic.
pub fn compute_state_probs(
    hap_indices: &[Vec<u32>],
    cluster_bounds: &[(usize, usize)],
    genotyped_markers: &[usize],
    target_gt: &GenotypeMatrix<Phased>,
    ref_gt: &GenotypeMatrix<Phased>,
    alignment: &MarkerAlignment,
    geno_a1: &[u8],
    geno_a2: &[u8],
    targ_alleles: &[u8],
    partner_alleles: Option<&[u8]>,
    pl_provider: Option<&PlProvider>,
    sample_idx: usize,
    n_states: usize,
    workspace: &mut ImpWorkspace,
    base_err_rate: f32,
    cluster_p_recomb: &[f32],
    marker_cluster: Arc<Vec<usize>>,
    ref_cluster_end: Arc<Vec<usize>>,
    gen_positions: Arc<Vec<f64>>,
    cluster_midpoints_pos: Arc<Vec<f64>>,
    recomb_intensity: f32,
    prior_probs: Option<&[f32]>,
    trace: bool,
) -> Arc<ClusterStateProbs> {
    let n_clusters = cluster_bounds.len();
    workspace.reset_and_ensure_capacity(n_clusters, n_states);

    compute_cluster_mismatches_into_workspace(
        hap_indices,
        cluster_bounds,
        genotyped_markers,
        target_gt,
        ref_gt,
        alignment,
        geno_a1,
        geno_a2,
        targ_alleles,
        partner_alleles,
        pl_provider,
        sample_idx,
        n_states,
        workspace,
        base_err_rate,
        trace,
    );

    let threshold = 1e-5;

    let (offsets, sparse_haps, sparse_probs, sparse_probs_p1) = run_hmm_forward_backward_to_sparse(
        &workspace.diff_vals,
        &workspace.diff_cols,
        &workspace.diff_row_offsets,
        &workspace.cluster_base_scores,
        cluster_p_recomb,
        n_states,
        hap_indices,
        prior_probs,
        threshold,
        &mut workspace.fwd,
        &mut workspace.bwd,
        &mut workspace.block_fwd,
        trace,
    );

    Arc::new(ClusterStateProbs::from_sparse(
        marker_cluster,
        ref_cluster_end,
        gen_positions,
        cluster_midpoints_pos,
        recomb_intensity,
        n_states,
        offsets,
        sparse_haps,
        sparse_probs,
        sparse_probs_p1,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::ChromIdx;
    use crate::data::alignment::MarkerAlignment;
    use crate::data::haplotype::Samples;
    use crate::data::marker::Allele;
    use crate::data::marker::Marker;
    use crate::data::marker::Markers;
    use crate::data::storage::phase_state::Unphased;
    use crate::data::storage::{GenotypeColumn, GenotypeMatrix};
    use crate::model::pl_emission::PlProvider;
    use crate::utils::workspace::ImpWorkspace;
    use std::sync::Arc;

    #[test]
    fn test_compute_cluster_mismatches_accumulation() {
        // Regression test for cluster log-likelihood accumulation bug.
        // Ensures that penalties are summed (product of probabilities) rather than min-reduced.

        // Setup:
        // 1 cluster with 2 markers
        // Target sample has alleles (0, 0)
        // Reference haplotype 0 has (1, 1) -> mismatch at BOTH markers
        // Reference haplotype 1 has (1, 0) -> mismatch at FIRST marker only

        let n_states = 2;
        let mut workspace = ImpWorkspace::new(n_states);

        let samples = Arc::new(Samples::from_ids(vec!["S1".to_string()]));
        let mut markers = Markers::new();
        markers.add_chrom("chr1");

        for i in 0..2 {
            markers.push(Marker::new(
                ChromIdx::new(0),
                (i * 100) as u32,
                None,
                Allele::Base(0),
                vec![Allele::Base(1)],
            ));
        }

        // Target: (0, 0)
        let target_col0 = GenotypeColumn::from_alleles(&[0], 2);
        let target_col1 = GenotypeColumn::from_alleles(&[0], 2);
        let target_gt = GenotypeMatrix::new_phased(
            markers.clone(),
            vec![target_col0, target_col1],
            samples.clone(),
        );

        // Reference: Hap 0: (1, 1), Hap 1: (1, 0)
        let ref_col0 = GenotypeColumn::from_alleles(&[1, 1], 2);
        let ref_col1 = GenotypeColumn::from_alleles(&[1, 0], 2);
        let ref_samples = Arc::new(Samples::from_ids(vec!["Ref1".to_string()]));
        let ref_gt =
            GenotypeMatrix::new_phased(markers.clone(), vec![ref_col0, ref_col1], ref_samples);

        let cluster_bounds = vec![(0, 2)];
        let genotyped_markers = vec![0, 1];
        let hap_indices = vec![vec![0, 1]];

        let alignment = MarkerAlignment {
            ref_to_target: vec![0, 1],
            target_to_ref: vec![0, 1],
            allele_mappings: vec![None, None],
        };

        // Inputs
        let geno_a1 = vec![0, 0];
        let geno_a2 = vec![0, 0];
        let targ_alleles = vec![0, 0];

        compute_cluster_mismatches_into_workspace(
            &hap_indices,
            &cluster_bounds,
            &genotyped_markers,
            &target_gt,
            &ref_gt,
            &alignment,
            &geno_a1,
            &geno_a2,
            &targ_alleles,
            None,
            None,
            0, // sample_idx
            n_states,
            &mut workspace,
            0.01,
            false,
        );

        let err = 0.01f32;
        let match_prob = 1.0 - err;
        let mismatch_prob = err;
        let log_match = match_prob.ln();
        let log_mism = mismatch_prob.ln();
        let log_diff = log_mism - log_match;

        // Hap 1: 1 mismatch
        assert!(
            (workspace.row_buffer[1] - log_diff).abs() < 1e-4,
            "Hap 1 should confirm single mismatch"
        );

        // Hap 0: 2 mismatches (sum accumulation)
        assert!(
            (workspace.row_buffer[0] - 2.0 * log_diff).abs() < 1e-4,
            "Hap 0 should have double penalty"
        );
    }

    #[test]
    fn test_compute_cluster_mismatches_uses_pl_provider() {
        let n_states = 2;
        let mut markers = Markers::new();
        markers.add_chrom("chr1");
        markers.push(Marker::new(
            ChromIdx::new(0),
            100,
            None,
            Allele::Base(0),
            vec![Allele::Base(1)],
        ));

        let samples = Arc::new(Samples::from_ids(vec!["S1".to_string()]));
        let target_col = GenotypeColumn::from_alleles(&[0, 0], 2);
        let pl = Arc::new(crate::data::storage::matrix::PlMatrix::from_marker_blocks(
            1,
            vec![3u16],
            vec![vec![0u16, 0u16, 0u16]],
        ));
        let target_gt_unphased =
            GenotypeMatrix::<Unphased>::new_unphased_with_confidence_and_likelihoods(
                markers.clone(),
                vec![target_col],
                samples,
                None,
                pl,
            );
        let target_gt = target_gt_unphased.into_phased();

        let ref_samples = Arc::new(Samples::from_ids(vec!["R1".to_string()]));
        let ref_col = GenotypeColumn::from_alleles(&[0, 1], 2);
        let ref_gt = GenotypeMatrix::new_phased(markers, vec![ref_col], ref_samples);

        let cluster_bounds = vec![(0, 1)];
        let genotyped_markers = vec![0];
        let hap_indices = vec![vec![0, 1]];

        let alignment = MarkerAlignment {
            ref_to_target: vec![0],
            target_to_ref: vec![0],
            allele_mappings: vec![None],
        };

        let geno_a1 = vec![0];
        let geno_a2 = vec![0];
        let targ_alleles = vec![0];

        let mut workspace = ImpWorkspace::new(n_states);
        compute_cluster_mismatches_into_workspace(
            &hap_indices,
            &cluster_bounds,
            &genotyped_markers,
            &target_gt,
            &ref_gt,
            &alignment,
            &geno_a1,
            &geno_a2,
            &targ_alleles,
            None,
            None,
            0,
            n_states,
            &mut workspace,
            0.01,
            false,
        );
        let conf_penalty = workspace.row_buffer[1];

        let plp = PlProvider {
            gt: target_gt.as_unphased_ref(),
            sample: 0,
            subset_to_orig: None,
        };
        workspace.clear();
        compute_cluster_mismatches_into_workspace(
            &hap_indices,
            &cluster_bounds,
            &genotyped_markers,
            &target_gt,
            &ref_gt,
            &alignment,
            &geno_a1,
            &geno_a2,
            &targ_alleles,
            None,
            Some(&plp),
            0,
            n_states,
            &mut workspace,
            0.01,
            false,
        );
        let pl_penalty = workspace.row_buffer[1];

        assert!(conf_penalty < -1.0);
        assert!(pl_penalty.abs() < 1e-3);
    }
}
