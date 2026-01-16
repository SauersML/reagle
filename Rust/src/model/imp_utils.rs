//! Imputation utilities and HMM logic.

use std::sync::Arc;
use crate::data::haplotype::HapIdx;
use crate::data::marker::MarkerIdx;
use crate::data::storage::GenotypeMatrix;
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
) {
    workspace.reset_and_ensure_capacity(hap_indices.len(), n_states);

    let n_clusters = hap_indices.len();
    let p_err = base_err_rate.clamp(1e-8, 0.5);
    let p_no_err = 1.0 - p_err;

    for (c, &(start, end)) in cluster_bounds.iter().enumerate() {
        if c >= n_clusters {
            break;
        }

        let row_buffer = &mut workspace.row_buffer;
        row_buffer.fill(1.0);

        for &ref_m in &genotyped_markers[start..end] {
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

            let ref_marker_idx = MarkerIdx::new(ref_m as u32);
            let ref_column = ref_gt.column(ref_marker_idx);

            let mut process_states = |final_ref: u8, j: usize| {
                if final_ref == 255 {
                    row_buffer[j] *= 0.5;
                } else if partner_allele != 255 {
                    let required = if partner_allele == geno1 { geno2 } else { geno1 };
                    if final_ref != required {
                        row_buffer[j] = 0.0;
                    }
                } else if targ_allele != 255 {
                    let match_prob = if final_ref == targ_allele { p_no_err } else { p_err };
                    row_buffer[j] *= confidence * match_prob + (1.0 - confidence) * 0.5;
                }
            };

            for (j, &hap) in hap_indices[c].iter().enumerate().take(n_states) {
                let ref_allele = ref_column.get(HapIdx::new(hap));
                let final_ref = if alignment.has_allele_mapping(target_m) {
                    alignment.reverse_map_allele(target_m, ref_allele)
                } else {
                    ref_allele
                };
                process_states(final_ref, j);
            }
        }
        
        workspace.cluster_base_scores.push(1.0);

        for (j, &val) in row_buffer.iter().enumerate().take(n_states) {
            if val < 0.9999 {
                workspace.diff_vals.push(val.ln());
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
) -> (Vec<usize>, Vec<u32>, Vec<f32>, Vec<f32>) {
    let n_clusters = cluster_base_scores.len();
    if n_clusters == 0 {
        return (vec![0], Vec::new(), Vec::new(), Vec::new());
    }

    const LOG_EMIT_FLOOR: f32 = -80.0;

    // Simplified forward pass (store all probabilities)
    let mut fwd = vec![0.0f32; n_clusters * n_states];
    let mut fwd_sums = vec![0.0f32; n_clusters];

    for m in 0..n_clusters {
        let p_rec = p_recomb.get(m).copied().unwrap_or(0.0);
        let shift = p_rec / n_states as f32;
        let scale = if m == 0 {
            1.0 - p_rec
        } else {
            (1.0 - p_rec) / fwd_sums[m - 1].max(1e-30)
        };
        let base_emit = cluster_base_scores[m].max(LOG_EMIT_FLOOR).exp();

        let (fwd_prev_full, fwd_curr) = fwd.split_at_mut(m * n_states);
        let fwd_curr = &mut fwd_curr[..n_states];

        if m > 0 {
            let fwd_prev = &fwd_prev_full[(m - 1) * n_states..];
            for k in 0..n_states {
                fwd_curr[k] = base_emit * (scale * fwd_prev[k] + shift);
            }
        } else if let Some(priors) = prior_probs {
            for k in 0..n_states {
                fwd_curr[k] = base_emit * priors.get(k).copied().unwrap_or(1.0 / n_states as f32);
            }
        } else {
            fwd_curr.fill(base_emit / n_states as f32);
        }

        let start = diff_row_offsets[m];
        let end = diff_row_offsets[m + 1];
        for i in start..end {
            let col = diff_cols[i] as usize;
            if col < n_states {
                fwd_curr[col] *= diff_vals[i].exp();
            }
        }
        fwd_sums[m] = fwd_curr.iter().sum();
    }

    // Simplified backward pass
    let mut bwd = vec![0.0f32; n_clusters * n_states];
    bwd.chunks_exact_mut(n_states).last().unwrap().fill(1.0 / n_states as f32);

    for m in (0..n_clusters - 1).rev() {
        let p_rec = p_recomb.get(m + 1).copied().unwrap_or(0.0);
        let shift = p_rec / n_states as f32;
        let base_emit = cluster_base_scores[m + 1].max(LOG_EMIT_FLOOR).exp();

        let (bwd_curr_slice, bwd_next_slice) = bwd.split_at_mut((m + 1) * n_states);
        let bwd_curr = &mut bwd_curr_slice[m * n_states..];
        let bwd_next_mut = &mut bwd_next_slice[..n_states];

        let start = diff_row_offsets[m + 1];
        let end = diff_row_offsets[m + 2];
        for i in start..end {
            let col = diff_cols[i] as usize;
            if col < n_states {
                bwd_next_mut[col] *= diff_vals[i].exp();
            }
        }
        for k in 0..n_states {
            bwd_next_mut[k] *= base_emit;
        }

        let bwd_sum: f32 = bwd_next_mut.iter().sum();
        if bwd_sum > 0.0 {
            let scale = (1.0 - p_rec) / bwd_sum;
            for k in 0..n_states {
                bwd_curr[k] = scale * bwd_next_mut[k] + shift;
            }
        } else {
            bwd_curr.fill(1.0 / n_states as f32);
        }
    }

    // Combine and store sparse posteriors
    let mut offsets = Vec::with_capacity(n_clusters + 1);
    offsets.push(0);
    let mut hap_indices = Vec::new();
    let mut probs = Vec::new();
    let mut probs_p1 = Vec::new();

    for m in 0..n_clusters {
        let fwd_m = &fwd[m * n_states..(m + 1) * n_states];
        let bwd_m = &bwd[m * n_states..(m + 1) * n_states];
        let bwd_p1 = if m + 1 < n_clusters {
            &bwd[(m + 1) * n_states..(m + 2) * n_states]
        } else {
            bwd_m
        };

        let mut sum = 0.0;
        let mut posteriors = vec![0.0; n_states];
        for k in 0..n_states {
            let val = fwd_m[k] * bwd_m[k];
            posteriors[k] = val;
            sum += val;
        }
        if sum > 1e-30 {
            let inv_sum = 1.0 / sum;
            for p in &mut posteriors { *p *= inv_sum; }
        }

        let mut sum_p1 = 0.0;
        let mut posteriors_p1 = vec![0.0; n_states];
        for k in 0..n_states {
            let val = fwd_m[k] * bwd_p1[k];
            posteriors_p1[k] = val;
            sum_p1 += val;
        }
        if sum_p1 > 1e-30 {
            let inv_sum = 1.0 / sum_p1;
            for p in &mut posteriors_p1 { *p *= inv_sum; }
        }

        for k in 0..n_states {
            let p = posteriors[k];
            let p1 = posteriors_p1[k];
            if p > threshold || p1 > threshold {
                hap_indices.push(hap_indices_input[m][k]);
                probs.push(p);
                probs_p1.push(p1);
            }
        }
        offsets.push(hap_indices.len());
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
