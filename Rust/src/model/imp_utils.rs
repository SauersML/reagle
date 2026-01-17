//! Imputation utilities and HMM logic.

use std::sync::Arc;
use aligned_vec::{AVec, ConstAlign};
use crate::data::haplotype::HapIdx;
use crate::data::marker::MarkerIdx;
use crate::data::storage::{GenotypeColumn, GenotypeMatrix};
use crate::data::storage::phase_state::Phased;
use crate::data::alignment::MarkerAlignment;
use crate::utils::workspace::ImpWorkspace;
use crate::pipelines::imputation::ClusterStateProbs; // Assuming we keep it there or import it

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

pub fn compute_cluster_weights(
    gen_positions: &[f64],
    ref_cluster_start: &[usize],
    ref_cluster_end: &[usize],
) -> Vec<f32> {
    let n_ref_markers = gen_positions.len();
    let mut wts = vec![f32::NAN; n_ref_markers];
    if ref_cluster_start.is_empty() {
        return wts;
    }

    for c in 0..ref_cluster_start.len().saturating_sub(1) {
        let end = ref_cluster_end[c];
        let next_start = ref_cluster_start[c + 1];
        if end == 0 || next_start <= end {
            continue;
        }
        let next_start_pos = gen_positions[next_start];
        let end_pos = gen_positions[end - 1];
        let total_len = (next_start_pos - end_pos).max(1e-12);

        for m in end..next_start {
            let wt = (next_start_pos - gen_positions[m]) / total_len;
            wts[m] = wt as f32;
        }
    }
    wts
}

pub fn build_marker_cluster_index(
    ref_cluster_start: &[usize],
    n_ref_markers: usize,
) -> Vec<usize> {
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
    sample_idx: usize,
    n_states: usize,
    workspace: &mut ImpWorkspace,
    base_err_rate: f32,
    trace: bool,
) {
    let span = if trace {
        Some(tracing::info_span!("mismatch_precalc").entered())
    } else {
        None
    };
    let _ = &span;
    workspace.reset_and_ensure_capacity(hap_indices.len(), n_states);

    let n_clusters = hap_indices.len();
    let p_err = base_err_rate.clamp(1e-8, 0.5);

    for (c, &(start, end)) in cluster_bounds.iter().enumerate() {
        if c >= n_clusters {
            break;
        }

        let row_buffer = &mut workspace.row_buffer;
        row_buffer.fill(0.0);
        let mut cluster_base_score = 0.0f32;

        for &ref_m in &genotyped_markers[start..end] {
            // Use direct access via ref_to_target if possible, or fallback to method
            let target_m_idx = alignment.ref_to_target.get(ref_m).copied().unwrap_or(-1);
            if target_m_idx < 0 { continue; }
            let target_m = target_m_idx as usize;

            let target_marker_idx = MarkerIdx::new(target_m as u32);
            let geno1 = geno_a1[target_m];
            let geno2 = geno_a2[target_m];
            if geno1 == 255 || geno2 == 255 {
                continue;
            }
            let targ_allele = targ_alleles[target_m];
            let partner_allele = partner_alleles
                .map(|p| p[target_m])
                .unwrap_or(255);
            let confidence = target_gt.sample_confidence_f32(target_marker_idx, sample_idx);
            if confidence <= 0.0 {
                continue;
            }
            
            let (log_match, log_mism) = get_log_probs(confidence, p_err);
            let log_diff = log_mism - log_match;
            let hard_log_mism = (1e-12f32).ln();
            let hard_log_diff = hard_log_mism - log_match;
            
            cluster_base_score += log_match;

            let ref_marker_idx = MarkerIdx::new(ref_m as u32);
            let ref_column = ref_gt.column(ref_marker_idx);

            macro_rules! process_states {
                ($col:expr, $get_fn:expr) => {
                    for (j, &hap) in hap_indices[c].iter().enumerate().take(n_states) {
                        let ref_allele = $get_fn($col, HapIdx::new(hap));
                        let final_ref = if alignment.has_allele_mapping(target_m) {
                             alignment.reverse_map_allele(target_m, ref_allele)
                        } else {
                            ref_allele
                        };

                        if final_ref == 255 {
                            if ref_allele != 255 {
                                row_buffer[j] += log_diff;
                            }
                        } else if partner_allele != 255 {
                            let required = if partner_allele == geno1 {
                                geno2
                            } else if partner_allele == geno2 {
                                geno1
                            } else {
                                255
                            };
                            if required != 255 {
                                if final_ref != required {
                                    row_buffer[j] += hard_log_diff;
                                }
                            } else if targ_allele != 255 && final_ref != targ_allele {
                                row_buffer[j] += log_diff;
                            }
                        } else if targ_allele != 255 && final_ref != targ_allele {
                            row_buffer[j] += log_diff;
                        }
                    }
                }
            }

            match ref_column {
                GenotypeColumn::Dense(col) => {
                     process_states!(col, |c: &crate::data::storage::dense::DenseColumn, h| c.get(h));
                },
                GenotypeColumn::Sparse(col) => {
                     process_states!(col, |c: &crate::data::storage::sparse::SparseColumn, h| c.get(h));
                },
                GenotypeColumn::Dictionary(col, offset) => {
                     process_states!(col, |c: &std::sync::Arc<crate::data::storage::dictionary::DictionaryColumn>, h| c.get(*offset, h));
                },
                GenotypeColumn::SeqCoded(col) => {
                     process_states!(col, |c: &crate::data::storage::seq_coded::SeqCodedColumn, h| c.get(h));
                }
            }
        }
        
        workspace.cluster_base_scores.push(cluster_base_score);

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
                    fwd[curr_off..curr_off+n_states].fill(val);
                }
            } else {
                let val = base_emit / n_states as f32;
                fwd[curr_off..curr_off+n_states].fill(val);
            }
        } else {
            let (lower, upper) = fwd.split_at_mut(prev_base);
            let (curr_slice, prev_slice) = if m % 2 == 0 {
                (&mut lower[curr_base..curr_base+n_states], &upper[..n_states])
            } else {
                (&mut upper[..n_states], &lower[curr_base..curr_base+n_states])
            };

            let shift_vec = f32x8::splat(shift);
            let scale_vec = f32x8::splat(scale);
            let emit_vec = f32x8::splat(base_emit);
            
            let mut k = 0;
            while k + 8 <= n_states {
                let prev_chunk_arr: &[f32; 8] = prev_slice[k..k+8].try_into().unwrap();
                let prev_vec = f32x8::from(*prev_chunk_arr);
                let trans = prev_vec.mul_add(scale_vec, shift_vec);
                let res = trans * emit_vec;
                let res_arr: [f32; 8] = res.into();
                curr_slice[k..k+8].copy_from_slice(&res_arr);
                k += 8;
            }
            for i in k..n_states {
                let p = prev_slice[i];
                curr_slice[i] = base_emit * (scale * p + shift);
            }
        }
        
        let curr_slice = &mut fwd[curr_off..curr_off+n_states];
        let start = diff_row_offsets[m];
        let end = diff_row_offsets[m+1];

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
             let chunk_arr: &[f32; 8] = curr_slice[k..k+8].try_into().unwrap();
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
            let src = &working[src_off..src_off+n_states];
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

        if block_start >= n_clusters { continue; }

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
            for &x in &block_fwd[0..n_states] { sum += x; }
            recomp_sum = sum.max(1e-30);
            curr_off = 0;
        } else {
            let load_idx = block_idx - 1;
            let checkpoint_off = load_idx * n_states;
            block_fwd[0..n_states].copy_from_slice(&fwd[checkpoint_off..checkpoint_off + n_states]);
            recomp_sum = 1.0;
            curr_off = 0;
        }

        let loop_start = if block_idx == 0 { block_start + 1 } else { block_start };
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
                let prev_chunk_arr: &[f32; 8] = prev_slice[k..k+8].try_into().unwrap();
                let prev_vec = f32x8::from(*prev_chunk_arr);
                let trans = prev_vec.mul_add(scale_vec, shift_vec);
                let res = trans * emit_vec;
                let res_arr: [f32; 8] = res.into();
                curr_slice[k..k+8].copy_from_slice(&res_arr);
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
            for x in curr_slice.iter() { new_sum += *x; }
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
                    let initial_chunk_arr: &[f32; 8] = bwd[k..k+8].try_into().unwrap();
                    let initial_chunk = f32x8::from(*initial_chunk_arr);
                    let res = initial_chunk * base_emit_vec;
                    let res_arr: [f32; 8] = res.into();
                    bwd[k..k+8].copy_from_slice(&res_arr);
                    k += 8;
                }
                for x in bwd[k..].iter_mut() { *x *= base_emit; }

                let start = diff_row_offsets[m+1];
                let end = diff_row_offsets[m+2];
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
                    let chunk_arr: &[f32; 8] = bwd[k..k+8].try_into().unwrap();
                    let chunk = f32x8::from(*chunk_arr);
                    sum_vec += chunk;
                    k += 8;
                }
                emitted_sum += sum_vec.reduce_add();
                for &x in bwd[k..].iter() { emitted_sum += x; }

                if emitted_sum > 0.0 {
                    let scale_v = (1.0 - p_rec) / emitted_sum;
                    let scale_vec = f32x8::splat(scale_v);
                    let shift_vec = f32x8::splat(shift);
                    k = 0;
                    while k + 8 <= n_states {
                         let chunk_arr: &[f32; 8] = bwd[k..k+8].try_into().unwrap();
                         let chunk = f32x8::from(*chunk_arr);
                         let res = chunk.mul_add(scale_vec, shift_vec);
                         let res_arr: [f32; 8] = res.into();
                         bwd[k..k+8].copy_from_slice(&res_arr);
                         k += 8;
                    }
                    for x in bwd[k..].iter_mut() { *x = scale_v * *x + shift; }
                } else {
                    bwd.fill(1.0 / n_states as f32);
                }
            }

            let local_offset = if block_idx == 0 { (m - block_start) * n_states } else { (m - block_start + 1) * n_states };
            let fwd_row = &block_fwd[local_offset..local_offset + n_states];

            let mut state_sum = 0.0f32;
            for k in 0..n_states {
                curr_posteriors[k] = fwd_row[k] * bwd[k];
                state_sum += curr_posteriors[k];
            }
            if state_sum > 0.0 {
                let inv = 1.0 / state_sum;
                for k in 0..n_states { curr_posteriors[k] *= inv; }
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
    sample_idx: usize,
    n_states: usize,
    workspace: &mut ImpWorkspace,
    base_err_rate: f32,
    cluster_p_recomb: &[f32],
    marker_cluster: Arc<Vec<usize>>,
    ref_cluster_end: Arc<Vec<usize>>,
    cluster_weights: Arc<Vec<f32>>,
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
        sample_idx,
        n_states,
        workspace,
        base_err_rate,
        trace,
    );

    let threshold = if n_clusters <= 1000 {
        0.0
    } else {
        (0.9999f32 / n_states as f32).min(0.005f32)
    };

    let (offsets, sparse_haps, sparse_probs, sparse_probs_p1) =
        run_hmm_forward_backward_to_sparse(
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
        cluster_weights,
        offsets,
        sparse_haps,
        sparse_probs,
        sparse_probs_p1,
    ))
}
