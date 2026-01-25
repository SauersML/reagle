//! # Compression Wrapper for Building MicroWindows
//!
//! This module provides functions to build `DictionaryColumn` and `MicroWindow`
//! from ranges of markers in a `GenotypeMatrix`.

use super::compressed_block::CompressedBlock;
use super::types::{GlobalId, PatternId};
use crate::data::haplotype::HapIdx;
use crate::data::marker::Marker;
use crate::data::storage::GenotypeColumn;
use crate::data::storage::dictionary::DictionaryColumn;

use rayon::prelude::*;
use std::sync::Arc;

fn flatten_pattern_globals(pattern_to_globals: &[Vec<GlobalId>]) -> (Vec<GlobalId>, Vec<usize>) {
    let mut offsets = Vec::with_capacity(pattern_to_globals.len() + 1);
    let mut flat = Vec::new();
    let mut current = 0usize;

    for globals in pattern_to_globals {
        offsets.push(current);
        flat.extend(globals.iter().copied());
        current = flat.len();
    }
    offsets.push(current);

    (flat, offsets)
}

fn build_compressed_block_with_accessor<F>(
    start_marker: usize,
    n_markers: usize,
    n_haplotypes: usize,
    marker_n_alleles: Vec<u8>,
    max_states: usize,
    recomb_rates: &[f32],
    keep_haps: Option<&[bool]>,
    get_allele: F,
) -> CompressedBlock
where
    F: Fn(usize, HapIdx) -> u8 + Sync,
{
    assert_eq!(
        recomb_rates.len(),
        n_markers.saturating_sub(1),
        "Recombination rates must be interval rates (length = n_markers - 1)"
    );
    let local_recomb_rates = recomb_rates.to_vec();

    let mut max_allele_overall = 0u8;
    for &n_alleles in &marker_n_alleles {
        if n_alleles > 0 {
            max_allele_overall = max_allele_overall.max((n_alleles - 1) as u8);
        }
    }

    let bits_per_allele = if max_allele_overall < 2 {
        1
    } else if max_allele_overall < 4 {
        2
    } else if max_allele_overall < 16 {
        4
    } else {
        8
    };

    let dict_column =
        DictionaryColumn::compress(get_allele, n_markers, n_haplotypes, bits_per_allele);
    let storage = Arc::new(dict_column);

    let hap_to_pattern = storage.hap_to_pattern();
    let n_unique_patterns = storage.n_patterns();

    let use_keep = keep_haps.is_some();
    let keep_mask = keep_haps.unwrap_or(&[]);
    let mut initial_pattern_counts: Vec<f32> = vec![0.0; n_unique_patterns];
    let mut initial_pattern_to_globals: Vec<Vec<GlobalId>> = vec![Vec::new(); n_unique_patterns];
    let mut unkept_globals: Vec<GlobalId> = Vec::new();

    for (hap_idx, &pattern_idx) in hap_to_pattern.iter().enumerate() {
        let global_id = GlobalId::new(hap_idx as u32);
        if use_keep && !keep_mask.get(hap_idx).copied().unwrap_or(false) {
            unkept_globals.push(global_id);
            continue;
        }
        initial_pattern_counts[pattern_idx as usize] += 1.0;
        initial_pattern_to_globals[pattern_idx as usize].push(global_id);
    }

    let limit = if max_states == 0 {
        usize::MAX
    } else {
        max_states
    };

    let mut pattern_order: Vec<usize> = (0..n_unique_patterns)
        .filter(|&i| initial_pattern_counts[i] > 0.0 || !use_keep)
        .collect();
    pattern_order.sort_by(|&a, &b| {
        initial_pattern_counts[b]
            .partial_cmp(&initial_pattern_counts[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let kept_indices = if pattern_order.len() > limit {
        pattern_order[..limit].to_vec()
    } else {
        pattern_order
    };
    let kept_patterns_set: std::collections::HashSet<usize> =
        kept_indices.iter().copied().collect();

    let mut old_to_new = vec![PatternId::RESERVOIR; n_unique_patterns];
    for (new_idx, &old_idx) in kept_indices.iter().enumerate() {
        old_to_new[old_idx] = PatternId::new(new_idx as u32);
    }

    let hap_to_state: Vec<PatternId> = hap_to_pattern
        .iter()
        .enumerate()
        .map(|(hap_idx, &old_idx)| {
            if use_keep && !keep_mask.get(hap_idx).copied().unwrap_or(false) {
                PatternId::RESERVOIR
            } else {
                old_to_new[old_idx as usize]
            }
        })
        .collect();

    let mut reservoir_globals = if use_keep { unkept_globals } else { Vec::new() };
    for (pattern_idx, globals) in initial_pattern_to_globals.iter().enumerate() {
        if !kept_patterns_set.contains(&pattern_idx) {
            reservoir_globals.extend(globals.iter().copied());
        }
    }

    let reservoir_count = reservoir_globals.len() as u32;

    let (reservoir_freqs, reservoir_freq_offsets, reservoir_obs_fractions, reservoir_ld) =
        if reservoir_count > 0 {
            compute_reservoir_freqs(&storage, &reservoir_globals, n_markers, &marker_n_alleles)
        } else {
            (
                Vec::new(),
                vec![0; n_markers],
                vec![0.0; n_markers],
                Vec::new(),
            )
        };

    let pattern_counts: Vec<f32> = kept_indices
        .iter()
        .map(|&i| initial_pattern_counts[i])
        .collect();
    let pattern_to_globals: Vec<Vec<GlobalId>> = kept_indices
        .iter()
        .map(|&i| initial_pattern_to_globals[i].clone())
        .collect();

    let (
        pattern_counts,
        pattern_to_globals,
        reservoir_count,
        reservoir_globals,
        reservoir_allele_freqs,
        reservoir_freq_offsets,
        reservoir_obs_fractions,
        reservoir_ld,
        hap_to_state,
    ) = (
        pattern_counts,
        pattern_to_globals,
        reservoir_count,
        reservoir_globals,
        reservoir_freqs,
        reservoir_freq_offsets,
        reservoir_obs_fractions,
        reservoir_ld,
        hap_to_state,
    );

    let n_kept_patterns = pattern_counts.len();
    let mut unpacked_alleles = vec![0u8; n_kept_patterns * n_markers];

    for (kept_idx, globals) in pattern_to_globals.iter().enumerate() {
        if let Some(first_global) = globals.first() {
            let hap_u32: u32 = first_global.as_u32();
            let hap = HapIdx::new(hap_u32);
            for m in 0..n_markers {
                let allele = storage.get(m, hap);
                unpacked_alleles[kept_idx * n_markers + m] = allele;
            }
        }
    }

    let (pattern_globals, pattern_globals_offsets) = flatten_pattern_globals(&pattern_to_globals);

    CompressedBlock {
        start_marker,
        end_marker: start_marker + n_markers,
        hap_to_state,
        pattern_counts,
        pattern_globals,
        pattern_globals_offsets,
        reservoir_count,
        reservoir_globals,
        reservoir_freqs: reservoir_allele_freqs,
        reservoir_freq_offsets,
        reservoir_ld,
        reservoir_obs_fractions,
        unpacked_alleles,
        local_recomb_rates,
        marker_n_alleles,
    }
}

/// Build a CompressedBlock from a range of markers (Recommended API)
///
/// This creates immutable, Arc-shareable reference data.
/// Use this instead of build_micro_window for parallel processing.
///
/// # Arguments
/// * `ref_data` - Reference panel genotype matrix
/// * `marker_range` - Range of markers to include in this window
/// * `max_states` - Maximum number of unique patterns to track (0 = no limit)
/// * `recomb_rates` - Vector of interval recombination rates. Length = marker_range.len() - 1.
///                    Rate[i] is the recombination rate between marker i and i+1.
///
/// # Returns
/// CompressedBlock with only immutable reference data (no per-sample HMM state)
pub(crate) fn build_compressed_block_from_columns(
    markers: &[Marker],
    columns: &[GenotypeColumn],
    start_marker: usize,
    max_states: usize,
    recomb_rates: &[f32],
) -> CompressedBlock {
    let n_markers = markers.len();
    let n_haplotypes = columns.first().map(|c| c.n_haplotypes()).unwrap_or(0);
    let marker_n_alleles: Vec<u8> = markers
        .iter()
        .map(|m| (1 + m.alt_alleles.len()).min(u8::MAX as usize) as u8)
        .collect();

    build_compressed_block_with_accessor(
        start_marker,
        n_markers,
        n_haplotypes,
        marker_n_alleles,
        max_states,
        recomb_rates,
        None,
        |m, h| columns[m].get(h),
    )
}

/// Build a CompressedBlock from columns with an optional keep-mask over haplotypes.
///
/// Haplotypes with keep_mask=false are routed into the reservoir.
pub(crate) fn build_compressed_block_from_columns_with_mask(
    markers: &[Marker],
    columns: &[GenotypeColumn],
    start_marker: usize,
    max_states: usize,
    recomb_rates: &[f32],
    keep_mask: Option<&[bool]>,
) -> CompressedBlock {
    let n_markers = markers.len();
    let n_haplotypes = columns.first().map(|c| c.n_haplotypes()).unwrap_or(0);
    let marker_n_alleles: Vec<u8> = markers
        .iter()
        .map(|m| (1 + m.alt_alleles.len()).min(u8::MAX as usize) as u8)
        .collect();

    build_compressed_block_with_accessor(
        start_marker,
        n_markers,
        n_haplotypes,
        marker_n_alleles,
        max_states,
        recomb_rates,
        keep_mask,
        |m, h| columns[m].get(h),
    )
}

/// Compute allele frequencies for reservoir haplotypes (Multiallelic)
/// Parallelized: chunks of haplotypes are processed in parallel, then reduced.
fn compute_reservoir_freqs(
    storage: &Arc<DictionaryColumn>,
    reservoir_globals: &[GlobalId],
    window_size: usize,
    marker_n_alleles: &[u8],
) -> (Vec<f32>, Vec<usize>, Vec<f32>, Vec<[f32; 4]>) {
    let n_reservoir = reservoir_globals.len();
    if n_reservoir == 0 {
        return (
            Vec::new(),
            vec![0; window_size],
            vec![0.0; window_size],
            Vec::new(),
        );
    }

    // Calculate total size and offsets
    let mut offsets = Vec::with_capacity(window_size);
    let mut current_offset = 0;
    for &n in marker_n_alleles {
        offsets.push(current_offset);
        current_offset += n as usize;
    }
    let total_slots = current_offset;

    // Parallel map-reduce over chunks of reservoir haplotypes
    let chunk_size = (n_reservoir / rayon::current_num_threads().max(1)).max(64);
    let (counts, obs_counts) = reservoir_globals
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut local_counts = vec![0u32; total_slots];
            let mut local_obs = vec![0u32; window_size];
            for &global_id in chunk {
                let hap = HapIdx::new(global_id.as_u32());
                for marker_idx in 0..window_size {
                    let allele = storage.get(marker_idx, hap);
                    if allele != 255 {
                        let n_alleles = marker_n_alleles[marker_idx] as usize;
                        if (allele as usize) < n_alleles {
                            let offset = offsets[marker_idx];
                            local_counts[offset + allele as usize] += 1;
                            local_obs[marker_idx] += 1;
                        }
                    }
                }
            }
            (local_counts, local_obs)
        })
        .reduce(
            || (vec![0u32; total_slots], vec![0u32; window_size]),
            |(mut acc_counts, mut acc_obs), (counts, obs)| {
                for (a, c) in acc_counts.iter_mut().zip(counts.iter()) {
                    *a += c;
                }
                for (a, o) in acc_obs.iter_mut().zip(obs.iter()) {
                    *a += o;
                }
                (acc_counts, acc_obs)
            },
        );

    // frequencies
    let mut freqs = Vec::with_capacity(counts.len());
    let mut current_marker = 0;
    let mut current_end = offsets.get(1).copied().unwrap_or(counts.len());

    for (i, &c) in counts.iter().enumerate() {
        while i >= current_end {
            current_marker += 1;
            current_end = offsets
                .get(current_marker + 1)
                .copied()
                .unwrap_or(counts.len());
        }

        let n_obs = obs_counts[current_marker];
        if n_obs > 0 {
            freqs.push(c as f32 / n_obs as f32);
        } else {
            freqs.push(0.0);
        }
    }

    // obs fractions
    let obs_fractions: Vec<f32> = obs_counts
        .iter()
        .map(|&c| {
            if n_reservoir > 0 {
                c as f32 / n_reservoir as f32
            } else {
                0.0
            }
        })
        .collect();

    let reservoir_ld =
        compute_reservoir_ld(storage, reservoir_globals, window_size, marker_n_alleles);
    (freqs, offsets, obs_fractions, reservoir_ld)
}

/// Compute LD coherence factors for reservoir haplotypes (biallelic only).
/// Parallelized: each marker pair is computed independently.
fn compute_reservoir_ld(
    storage: &Arc<DictionaryColumn>,
    reservoir_globals: &[GlobalId],
    window_size: usize,
    marker_n_alleles: &[u8],
) -> Vec<[f32; 4]> {
    if reservoir_globals.is_empty() || window_size < 2 {
        return vec![[1.0f32; 4]; window_size.saturating_sub(1)];
    }

    // Parallel computation per marker pair
    (0..window_size.saturating_sub(1))
        .into_par_iter()
        .map(|m| {
            if marker_n_alleles[m] != 2 || marker_n_alleles[m + 1] != 2 {
                return [1.0f32; 4];
            }

            let mut counts = [[0u32; 2]; 2];
            let mut row = [0u32; 2];
            let mut col = [0u32; 2];
            let mut total = 0u32;

            for &global_id in reservoir_globals {
                let hap = HapIdx::new(global_id.as_u32());
                let a = storage.get(m, hap);
                let b = storage.get(m + 1, hap);
                if a == 255 || b == 255 || a > 1 || b > 1 {
                    continue;
                }
                let ai = a as usize;
                let bi = b as usize;
                counts[ai][bi] += 1;
                row[ai] += 1;
                col[bi] += 1;
                total += 1;
            }

            if total == 0 {
                return [1.0f32; 4];
            }

            let total_f = total as f32;
            let mut ld = [1.0f32; 4];
            for ai in 0..2 {
                for bi in 0..2 {
                    let p_uv = counts[ai][bi] as f32 / total_f;
                    let p_u = row[ai] as f32 / total_f;
                    let p_v = col[bi] as f32 / total_f;
                    let mut lambda = if p_u > 0.0 && p_v > 0.0 {
                        p_uv / (p_u * p_v)
                    } else {
                        1.0
                    };
                    if !lambda.is_finite() {
                        lambda = 1.0;
                    }
                    ld[ai * 2 + bi] = lambda.clamp(0.1, 10.0);
                }
            }
            ld
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::data::marker::Markers;
    use crate::data::storage::GenotypeColumn;

    #[test]
    fn test_build_compressed_block() {
        // Setup simple scenario: 4 haplotypes, 2 markers
        // Hap 0: (0, 0)
        // Hap 1: (0, 0)
        // Hap 2: (1, 1)
        // Hap 3: (1, 1)

        let col0 = GenotypeColumn::from_alleles(&[0, 0, 1, 1], 2);
        let col1 = GenotypeColumn::from_alleles(&[0, 0, 1, 1], 2);

        use crate::data::marker::{Allele, Marker};
        let mut m = Markers::new();
        let chr = m.add_chrom("chr1");
        m.push(Marker::new(
            chr,
            100,
            None,
            Allele::Base(Nucleotide::A),
            vec![Allele::Base(Nucleotide::C)],
        ));
        m.push(Marker::new(
            chr,
            200,
            None,
            Allele::Base(Nucleotide::A),
            vec![Allele::Base(Nucleotide::C)],
        ));

        let markers: Vec<Marker> = (0..m.len())
            .map(|i| m[crate::data::marker::MarkerIdx::new(i as u32)].clone())
            .collect();
        let columns = vec![col0, col1];

        let recomb_rates = vec![0.01];
        let block = build_compressed_block_from_columns(&markers, &columns, 0, 0, &recomb_rates);

        // Verify compression
        // Haps 0,1 should be Pattern A
        // Haps 2,3 should be Pattern B
        assert_eq!(block.n_patterns(), 2);
        assert_eq!(block.n_ref_haps(), 4);

        let p0 = block.pattern_for_haplotype(GlobalId::new(0));
        let p1 = block.pattern_for_haplotype(GlobalId::new(1));
        let p2 = block.pattern_for_haplotype(GlobalId::new(2));
        let p3 = block.pattern_for_haplotype(GlobalId::new(3));

        assert_eq!(p0, p1);
        assert_eq!(p2, p3);
        assert_ne!(p0, p2);

        // Check counts
        assert_eq!(block.pattern_counts[p0.as_usize()], 2.0);
        assert_eq!(block.pattern_counts[p2.as_usize()], 2.0);

        // Check unpacked alleles
        // Pattern 0: (0, 0)
        assert_eq!(block.get_pattern_allele(p0, 0), 0);
        assert_eq!(block.get_pattern_allele(p0, 1), 0);
        // Pattern 1: (1, 1)
        assert_eq!(block.get_pattern_allele(p2, 0), 1);
        assert_eq!(block.get_pattern_allele(p2, 1), 1);
    }
}
