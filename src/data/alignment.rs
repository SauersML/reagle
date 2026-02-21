//! Marker alignment between target and reference panels.

use crate::data::marker::{AlleleMapping, AnyMarkerSpace, MarkerIdx, Markers};
use crate::data::storage::GenotypeMatrix;
use crate::data::storage::phase_state::PhaseState;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ChromKey(u32);

impl ChromKey {
    fn from_name(name: &str) -> Self {
        Self(chrom_to_code(name))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PosKey {
    chrom: ChromKey,
    pos: u32,
}

impl PosKey {
    fn new(chrom_name: &str, pos: u32) -> Self {
        Self {
            chrom: ChromKey::from_name(chrom_name),
            pos,
        }
    }
}

/// Marker alignment between target and reference panels
#[derive(Debug)]
pub struct MarkerAlignment<TargetSpace = AnyMarkerSpace, RefSpace = AnyMarkerSpace> {
    /// For each reference marker, the index of the corresponding target marker (-1 if not in target)
    pub ref_to_target: Vec<Option<MarkerIdx<TargetSpace>>>,
    /// For each target marker, the index of the corresponding reference marker
    pub target_to_ref: Vec<Option<MarkerIdx<RefSpace>>>,

    /// Allele mapping for each aligned marker (indexed by target marker)
    /// Maps target allele indices to reference allele indices
    pub allele_mappings: Vec<Option<AlleleMapping>>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct AlignmentStats {
    pub aligned: usize,
    pub strand_flipped: usize,
    pub allele_swapped: usize,
}

impl<TargetSpace, RefSpace> Clone for MarkerAlignment<TargetSpace, RefSpace> {
    fn clone(&self) -> Self {
        Self {
            ref_to_target: self.ref_to_target.clone(),
            target_to_ref: self.target_to_ref.clone(),
            allele_mappings: self.allele_mappings.clone(),
        }
    }
}

impl<TargetSpace, RefSpace> MarkerAlignment<TargetSpace, RefSpace> {
    /// Create alignment by matching markers by position with allele mapping
    ///
    /// This handles strand flips (A/T vs T/A) and allele swaps automatically
    /// using `compute_allele_mapping`.
    pub fn new<S1: PhaseState, S2: PhaseState>(
        target_gt: &GenotypeMatrix<S1, TargetSpace>,
        ref_gt: &GenotypeMatrix<S2, RefSpace>,
    ) -> Self {
        let n_ref_markers = ref_gt.n_markers();
        let n_target_markers = target_gt.n_markers();

        let target_pos_map = build_target_pos_map(target_gt);
        let ref_pos_map = build_ref_pos_map_from_gt(ref_gt);

        let mut ref_to_target = vec![None; n_ref_markers];
        let mut target_to_ref = vec![None; n_target_markers];
        let mut allele_mappings: Vec<Option<AlleleMapping>> = vec![None; n_target_markers];

        let mut n_strand_flipped = 0usize;
        let mut n_allele_swapped = 0usize;
        let mut dropped_ambiguous_flips = 0usize;

        for (key, ref_candidates) in &ref_pos_map {
            let Some(target_candidates) = target_pos_map.get(key) else {
                continue;
            };
            let matches =
                best_group_matches(ref_candidates, target_candidates, |ref_idx, target_idx| {
                    let ref_marker = ref_gt.marker(MarkerIdx::new(ref_idx as u32));
                    let target_marker = target_gt.marker(MarkerIdx::new(target_idx as u32));
                    candidate_mapping(target_marker, ref_marker, &mut dropped_ambiguous_flips)
                });

            for (ref_idx, target_idx, mapping) in matches {
                let ref_marker = ref_gt.marker(MarkerIdx::new(ref_idx as u32));
                let target_marker = target_gt.marker(MarkerIdx::new(target_idx as u32));
                let strand_flipped = mapping.strand_flipped;
                let alleles_swapped = mapping.alleles_swapped;
                ref_to_target[ref_idx] = Some(MarkerIdx::new(target_idx as u32));
                target_to_ref[target_idx] = Some(MarkerIdx::new(ref_idx as u32));
                allele_mappings[target_idx] = Some(mapping);

                if strand_flipped {
                    n_strand_flipped += 1;
                }
                if alleles_swapped {
                    n_allele_swapped += 1;
                    eprintln!(
                        "  Allele swapped at pos {}: target {}>{}, ref {}>{}",
                        target_marker.pos,
                        target_marker.ref_allele,
                        target_marker
                            .alt_alleles
                            .first()
                            .map(|a| a.to_string())
                            .unwrap_or_else(|| "-".to_string()),
                        ref_marker.ref_allele,
                        ref_marker
                            .alt_alleles
                            .first()
                            .map(|a| a.to_string())
                            .unwrap_or_else(|| "-".to_string()),
                    );
                }
            }
        }

        if n_strand_flipped > 0 || n_allele_swapped > 0 {
            eprintln!(
                "  Allele alignment: {} strand-flipped, {} allele-swapped markers",
                n_strand_flipped, n_allele_swapped
            );
        }
        if dropped_ambiguous_flips > 0 {
            eprintln!(
                "  Dropped {} strand-ambiguous SNV candidates requiring strand flips",
                dropped_ambiguous_flips
            );
        }

        Self {
            ref_to_target,
            target_to_ref,
            allele_mappings,
        }
    }

    /// Create alignment against reference markers without reference genotypes.
    pub fn new_with_ref_markers<S: PhaseState>(
        target_gt: &GenotypeMatrix<S, TargetSpace>,
        ref_markers: &Markers<RefSpace>,
    ) -> Self {
        let n_ref_markers = ref_markers.len();
        let n_target_markers = target_gt.n_markers();

        let target_pos_map = build_target_pos_map(target_gt);
        let ref_pos_map = build_ref_pos_map(ref_markers);

        let mut ref_to_target = vec![None; n_ref_markers];
        let mut target_to_ref = vec![None; n_target_markers];
        let mut allele_mappings: Vec<Option<AlleleMapping>> = vec![None; n_target_markers];

        let mut n_strand_flipped = 0usize;
        let mut n_allele_swapped = 0usize;
        let mut dropped_ambiguous_flips = 0usize;
        let mut unaligned_with_candidates = 0usize;
        let mut unaligned_examples: Vec<String> = Vec::new();

        for (key, ref_candidates) in &ref_pos_map {
            let Some(target_candidates) = target_pos_map.get(key) else {
                continue;
            };
            let matches =
                best_group_matches(ref_candidates, target_candidates, |ref_idx, target_idx| {
                    let ref_marker = ref_markers.marker(MarkerIdx::new(ref_idx as u32));
                    let target_marker = target_gt.marker(MarkerIdx::new(target_idx as u32));
                    candidate_mapping(target_marker, ref_marker, &mut dropped_ambiguous_flips)
                });

            let mut aligned_refs = vec![false; ref_candidates.len()];
            for (ref_idx, target_idx, mapping) in matches {
                let Some(ref_local_idx) = ref_candidates.iter().position(|&v| v == ref_idx) else {
                    continue;
                };
                aligned_refs[ref_local_idx] = true;

                let ref_marker = ref_markers.marker(MarkerIdx::new(ref_idx as u32));
                let target_marker = target_gt.marker(MarkerIdx::new(target_idx as u32));
                let strand_flipped = mapping.strand_flipped;
                let alleles_swapped = mapping.alleles_swapped;
                ref_to_target[ref_idx] = Some(MarkerIdx::new(target_idx as u32));
                target_to_ref[target_idx] = Some(MarkerIdx::new(ref_idx as u32));
                allele_mappings[target_idx] = Some(mapping);

                if strand_flipped {
                    n_strand_flipped += 1;
                }
                if alleles_swapped {
                    n_allele_swapped += 1;
                    eprintln!(
                        "  Allele swapped at pos {}: target {}>{}, ref {}>{}",
                        target_marker.pos,
                        target_marker.ref_allele,
                        target_marker
                            .alt_alleles
                            .first()
                            .map(|a| a.to_string())
                            .unwrap_or_else(|| "-".to_string()),
                        ref_marker.ref_allele,
                        ref_marker
                            .alt_alleles
                            .first()
                            .map(|a| a.to_string())
                            .unwrap_or_else(|| "-".to_string()),
                    );
                }
            }

            for (i, &ref_idx) in ref_candidates.iter().enumerate() {
                if aligned_refs[i] {
                    continue;
                }
                unaligned_with_candidates += 1;
                if unaligned_examples.len() < 5 {
                    let ref_marker = ref_markers.marker(MarkerIdx::new(ref_idx as u32));
                    let ref_chrom = ref_markers.chrom_name(ref_marker.chrom).unwrap_or("");
                    let ref_alt = ref_marker
                        .alt_alleles
                        .iter()
                        .map(|a| a.to_string())
                        .collect::<Vec<_>>()
                        .join(",");
                    unaligned_examples.push(format!(
                        "{}:{} {}>{}",
                        ref_chrom, ref_marker.pos, ref_marker.ref_allele, ref_alt
                    ));
                }
            }
        }

        if n_strand_flipped > 0 || n_allele_swapped > 0 {
            eprintln!(
                "  Allele alignment: {} strand-flipped, {} allele-swapped markers",
                n_strand_flipped, n_allele_swapped
            );
        }
        if dropped_ambiguous_flips > 0 {
            eprintln!(
                "  Dropped {} strand-ambiguous SNV candidates requiring strand flips",
                dropped_ambiguous_flips
            );
        }
        if unaligned_with_candidates > 0 {
            if unaligned_examples.is_empty() {
                eprintln!(
                    "  Unaligned positions with candidates: {}",
                    unaligned_with_candidates
                );
            } else {
                eprintln!(
                    "  Unaligned positions with candidates: {} (first {})",
                    unaligned_with_candidates,
                    unaligned_examples.len()
                );
                eprintln!("  Examples: {}", unaligned_examples.join(", "));
            }
        }

        Self {
            ref_to_target,
            target_to_ref,
            allele_mappings,
        }
    }

    /// Build a reference position index for fast streaming alignment.
    pub fn build_ref_pos_index(ref_markers: &Markers<RefSpace>) -> HashMap<PosKey, Vec<usize>> {
        build_ref_pos_map(ref_markers)
    }

    /// Create alignment against a pre-built reference position index.
    ///
    /// This avoids scanning all reference markers for each target window. The returned
    /// alignment is sized to the target markers; `ref_to_target` is left empty because
    /// streaming phasing only queries target-to-ref mappings.
    pub fn new_with_ref_index<S: PhaseState>(
        target_gt: &GenotypeMatrix<S, TargetSpace>,
        ref_markers: &Markers<RefSpace>,
        ref_pos_map: &HashMap<PosKey, Vec<usize>>,
    ) -> (Self, AlignmentStats) {
        use crate::data::marker::compute_allele_mapping;

        let n_target_markers = target_gt.n_markers();
        let mut target_to_ref = vec![None; n_target_markers];
        let mut allele_mappings: Vec<Option<AlleleMapping>> = vec![None; n_target_markers];
        let mut stats = AlignmentStats::default();

        for m in 0..n_target_markers {
            let target_marker = target_gt.marker(MarkerIdx::new(m as u32));
            let chrom_name = target_gt
                .markers()
                .chrom_name(target_marker.chrom)
                .unwrap_or("");
            if let Some(ref_candidates) =
                ref_pos_map.get(&PosKey::new(chrom_name, target_marker.pos))
            {
                let mut best: Option<(usize, AlleleMapping, u32)> = None;
                for &ref_idx in ref_candidates {
                    let ref_marker = ref_markers.marker(MarkerIdx::new(ref_idx as u32));
                    if let Some(mapping) = compute_allele_mapping(target_marker, ref_marker) {
                        if !mapping.is_valid() {
                            continue;
                        }
                        if mapping.strand_flipped
                            && (crate::data::marker::is_strand_ambiguous(target_marker)
                                || crate::data::marker::is_strand_ambiguous(ref_marker))
                        {
                            continue;
                        }
                        let score = mapping_score(&mapping);
                        if best
                            .as_ref()
                            .map(|(_, _, best_score)| score > *best_score)
                            .unwrap_or(true)
                        {
                            best = Some((ref_idx, mapping, score));
                        }
                    }
                }
                if let Some((ref_idx, mapping, _)) = best {
                    let ref_marker = ref_markers.marker(MarkerIdx::new(ref_idx as u32));
                    let strand_flipped = mapping.strand_flipped;
                    let alleles_swapped = mapping.alleles_swapped;
                    target_to_ref[m] = Some(MarkerIdx::new(ref_idx as u32));
                    if strand_flipped {
                        stats.strand_flipped += 1;
                    }
                    if alleles_swapped {
                        stats.allele_swapped += 1;
                        eprintln!(
                            "  Allele swapped at pos {}: target {}>{}, ref {}>{}",
                            target_marker.pos,
                            target_marker.ref_allele,
                            target_marker
                                .alt_alleles
                                .first()
                                .map(|a| a.to_string())
                                .unwrap_or_else(|| "-".to_string()),
                            ref_marker.ref_allele,
                            ref_marker
                                .alt_alleles
                                .first()
                                .map(|a| a.to_string())
                                .unwrap_or_else(|| "-".to_string()),
                        );
                    }
                    allele_mappings[m] = Some(mapping);
                    stats.aligned += 1;
                }
            }
        }

        (
            Self {
                ref_to_target: Vec::new(),
                target_to_ref,
                allele_mappings,
            },
            stats,
        )
    }

    /// Get target marker index for a reference marker (returns None if not genotyped)
    #[inline]
    pub fn target_marker(&self, ref_marker: MarkerIdx<RefSpace>) -> Option<MarkerIdx<TargetSpace>> {
        self.ref_to_target
            .get(ref_marker.as_usize())
            .and_then(|idx| *idx)
    }

    /// Map a reference allele to target allele space (reverse mapping)
    ///
    /// Returns the target allele index for a given reference allele,
    /// handling strand flips and swaps automatically.
    /// Returns missing if no valid mapping exists.
    pub fn reverse_map_allele(&self, target_marker: usize, ref_allele: u8) -> u8 {
        if ref_allele == crate::data::storage::AlleleCode::MISSING.raw() {
            return crate::data::storage::AlleleCode::MISSING.raw(); // Missing stays missing
        }

        if let Some(Some(mapping)) = self.allele_mappings.get(target_marker) {
            mapping
                .reverse_map_allele(ref_allele)
                .unwrap_or(crate::data::storage::AlleleCode::MISSING.raw())
        } else {
            // Unaligned markers must not leak through as identity; treat as missing.
            crate::data::storage::AlleleCode::MISSING.raw()
        }
    }

    /// Get reference marker index for a target marker (returns None if not aligned)
    pub fn target_to_ref(
        &self,
        target_marker: MarkerIdx<TargetSpace>,
    ) -> Option<MarkerIdx<RefSpace>> {
        // Check allele_mappings to ensure the marker actually aligns.
        // `target_to_ref` stores Option and unaligned markers remain None.
        if self
            .allele_mappings
            .get(target_marker.as_usize())
            .and_then(|m| m.as_ref())
            .is_some()
        {
            self.target_to_ref
                .get(target_marker.as_usize())
                .and_then(|idx| *idx)
        } else {
            None
        }
    }

    /// Get the number of markers that were successfully aligned
    pub fn n_aligned(&self) -> usize {
        self.ref_to_target.iter().filter(|x| x.is_some()).count()
    }
}

fn build_target_pos_map<S: PhaseState, Space>(
    target_gt: &GenotypeMatrix<S, Space>,
) -> HashMap<PosKey, Vec<usize>> {
    let mut target_pos_map: HashMap<PosKey, Vec<usize>> = HashMap::new();
    for m in 0..target_gt.n_markers() {
        let marker = target_gt.marker(MarkerIdx::new(m as u32));
        let chrom_name = target_gt.markers().chrom_name(marker.chrom).unwrap_or("");
        target_pos_map
            .entry(PosKey::new(chrom_name, marker.pos))
            .or_default()
            .push(m);
    }
    target_pos_map
}

fn build_ref_pos_map<Space>(ref_markers: &Markers<Space>) -> HashMap<PosKey, Vec<usize>> {
    let mut ref_pos_map: HashMap<PosKey, Vec<usize>> = HashMap::new();
    for m in 0..ref_markers.len() {
        let marker = ref_markers.marker(MarkerIdx::new(m as u32));
        let chrom_name = ref_markers.chrom_name(marker.chrom).unwrap_or("");
        ref_pos_map
            .entry(PosKey::new(chrom_name, marker.pos))
            .or_default()
            .push(m);
    }
    ref_pos_map
}

fn build_ref_pos_map_from_gt<S: PhaseState, Space>(
    ref_gt: &GenotypeMatrix<S, Space>,
) -> HashMap<PosKey, Vec<usize>> {
    let mut ref_pos_map: HashMap<PosKey, Vec<usize>> = HashMap::new();
    for m in 0..ref_gt.n_markers() {
        let marker = ref_gt.marker(MarkerIdx::new(m as u32));
        let chrom_name = ref_gt.markers().chrom_name(marker.chrom).unwrap_or("");
        ref_pos_map
            .entry(PosKey::new(chrom_name, marker.pos))
            .or_default()
            .push(m);
    }
    ref_pos_map
}

fn candidate_mapping(
    target_marker: &crate::data::marker::Marker,
    ref_marker: &crate::data::marker::Marker,
    dropped_ambiguous_flips: &mut usize,
) -> Option<(u32, AlleleMapping)> {
    use crate::data::marker::compute_allele_mapping;

    compute_allele_mapping(target_marker, ref_marker).and_then(|mapping| {
        if !mapping.is_valid() {
            return None;
        }
        if mapping.strand_flipped
            && (crate::data::marker::is_strand_ambiguous(target_marker)
                || crate::data::marker::is_strand_ambiguous(ref_marker))
        {
            *dropped_ambiguous_flips += 1;
            return None;
        }
        Some((mapping_score(&mapping), mapping))
    })
}

#[inline]
fn normalize_chrom(name: &str) -> &str {
    if name.len() >= 3 && name[..3].eq_ignore_ascii_case("chr") {
        &name[3..]
    } else {
        name
    }
}

#[inline]
fn chrom_to_code(name: &str) -> u32 {
    let norm = normalize_chrom(name);
    if let Ok(parsed) = norm.parse::<u32>() {
        return parsed;
    }
    if norm.eq_ignore_ascii_case("X") {
        return 23;
    }
    if norm.eq_ignore_ascii_case("Y") {
        return 24;
    }
    if norm.eq_ignore_ascii_case("XY") {
        return 25;
    }
    if norm.eq_ignore_ascii_case("M") || norm.eq_ignore_ascii_case("MT") {
        return 26;
    }

    let mut hash = 2166136261u32;
    for b in norm.bytes() {
        hash ^= b.to_ascii_lowercase() as u32;
        hash = hash.wrapping_mul(16777619u32);
    }
    hash | (1u32 << 31)
}

#[inline]
fn mapping_score(mapping: &AlleleMapping) -> u32 {
    let mapped = mapping.targ_to_ref.iter().filter(|&&v| v >= 0).count() as u32;
    let ref_is_ref = u32::from(mapping.targ_to_ref.first().copied().unwrap_or(-1) == 0);
    let unswapped = u32::from(!mapping.alleles_swapped);
    let unflipped = u32::from(!mapping.strand_flipped);
    (mapped << 3) | (ref_is_ref << 2) | (unswapped << 1) | unflipped
}

#[derive(Clone, Copy)]
struct MatchScore {
    aligned: usize,
    weight: u64,
}

impl MatchScore {
    fn zero() -> Self {
        Self {
            aligned: 0,
            weight: 0,
        }
    }
}

#[derive(Clone)]
struct GroupEdge {
    target_local_idx: usize,
    score: u32,
    mapping: AlleleMapping,
}

fn score_better(left: MatchScore, right: MatchScore) -> bool {
    left.aligned > right.aligned || (left.aligned == right.aligned && left.weight > right.weight)
}

fn solve_group_dp(
    ref_local_idx: usize,
    used_target_mask: u64,
    edges_by_ref: &[Vec<GroupEdge>],
    memo: &mut HashMap<(usize, u64), (MatchScore, Option<usize>)>,
) -> MatchScore {
    if ref_local_idx == edges_by_ref.len() {
        return MatchScore::zero();
    }
    if let Some((score, _)) = memo.get(&(ref_local_idx, used_target_mask)) {
        return *score;
    }

    let mut best = solve_group_dp(ref_local_idx + 1, used_target_mask, edges_by_ref, memo);
    let mut best_choice = None;

    for edge in &edges_by_ref[ref_local_idx] {
        let bit = 1u64 << edge.target_local_idx;
        if (used_target_mask & bit) != 0 {
            continue;
        }
        let mut candidate = solve_group_dp(
            ref_local_idx + 1,
            used_target_mask | bit,
            edges_by_ref,
            memo,
        );
        candidate.aligned += 1;
        candidate.weight += edge.score as u64;
        if score_better(candidate, best) {
            best = candidate;
            best_choice = Some(edge.target_local_idx);
        }
    }

    memo.insert((ref_local_idx, used_target_mask), (best, best_choice));
    best
}

fn best_group_matches<F>(
    ref_indices: &[usize],
    target_indices: &[usize],
    mut build_edge: F,
) -> Vec<(usize, usize, AlleleMapping)>
where
    F: FnMut(usize, usize) -> Option<(u32, AlleleMapping)>,
{
    let mut edges_by_ref: Vec<Vec<GroupEdge>> = vec![Vec::new(); ref_indices.len()];
    for (ref_local_idx, &ref_idx) in ref_indices.iter().enumerate() {
        for (target_local_idx, &target_idx) in target_indices.iter().enumerate() {
            if let Some((score, mapping)) = build_edge(ref_idx, target_idx) {
                edges_by_ref[ref_local_idx].push(GroupEdge {
                    target_local_idx,
                    score,
                    mapping,
                });
            }
        }
    }

    if target_indices.len() <= 63 {
        let mut memo: HashMap<(usize, u64), (MatchScore, Option<usize>)> = HashMap::new();
        let _ = solve_group_dp(0, 0, &edges_by_ref, &mut memo);

        let mut out = Vec::new();
        let mut used_target_mask = 0u64;
        for ref_local_idx in 0..ref_indices.len() {
            if let Some((_, choice)) = memo.get(&(ref_local_idx, used_target_mask)) {
                if let Some(target_local_idx) = *choice {
                    if let Some(edge) = edges_by_ref[ref_local_idx]
                        .iter()
                        .find(|edge| edge.target_local_idx == target_local_idx)
                    {
                        out.push((
                            ref_indices[ref_local_idx],
                            target_indices[target_local_idx],
                            edge.mapping.clone(),
                        ));
                        used_target_mask |= 1u64 << target_local_idx;
                    }
                }
            }
        }
        return out;
    }

    let mut all_edges: Vec<(usize, usize, u32, AlleleMapping)> = Vec::new();
    for (ref_local_idx, ref_edges) in edges_by_ref.iter().enumerate() {
        for edge in ref_edges {
            all_edges.push((
                ref_local_idx,
                edge.target_local_idx,
                edge.score,
                edge.mapping.clone(),
            ));
        }
    }
    all_edges.sort_by(|a, b| b.2.cmp(&a.2));

    let mut used_refs = vec![false; ref_indices.len()];
    let mut used_targets = vec![false; target_indices.len()];
    let mut out = Vec::new();
    for (ref_local_idx, target_local_idx, _, mapping) in all_edges {
        if used_refs[ref_local_idx] || used_targets[target_local_idx] {
            continue;
        }
        used_refs[ref_local_idx] = true;
        used_targets[target_local_idx] = true;
        out.push((
            ref_indices[ref_local_idx],
            target_indices[target_local_idx],
            mapping,
        ));
    }
    out
}
