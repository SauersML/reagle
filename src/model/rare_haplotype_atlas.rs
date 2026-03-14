use crate::data::HapIdx;
use crate::data::storage::GenotypeColumn;
use crate::model::impute_hmm::TargetAlleleProbs;
use crate::model::types::RefHapId;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
pub struct RareHaplotypeAtlasConfig {
    pub max_anchor_markers: usize,
    pub rare_freq_max: f32,
    pub min_leaf_size: usize,
    pub max_representatives_per_leaf: usize,
    pub max_enriched_alleles_per_leaf: usize,
    pub min_enrichment: f32,
}

impl Default for RareHaplotypeAtlasConfig {
    fn default() -> Self {
        Self {
            max_anchor_markers: 12,
            rare_freq_max: 0.01,
            min_leaf_size: 2,
            max_representatives_per_leaf: 4,
            max_enriched_alleles_per_leaf: 8,
            min_enrichment: 2.0,
        }
    }
}

#[derive(Clone, Debug)]
struct EnrichedAllele {
    marker: usize,
    allele: u8,
    enrichment: f32,
}

#[derive(Clone, Debug)]
struct RareLeaf {
    signature: u64,
    members: Vec<RefHapId>,
    representatives: Vec<RefHapId>,
    enriched: Vec<EnrichedAllele>,
}

#[derive(Clone, Debug, Default)]
pub struct RareHaplotypeAtlas {
    anchor_markers: Vec<usize>,
    leaves: Vec<RareLeaf>,
}

impl RareHaplotypeAtlas {
    pub fn build(
        ref_columns: &[GenotypeColumn],
        target_probs: &TargetAlleleProbs,
        n_ref_haps: usize,
        cfg: RareHaplotypeAtlasConfig,
    ) -> Self {
        if ref_columns.is_empty() || n_ref_haps == 0 {
            return Self::default();
        }
        let anchor_markers = select_anchor_markers(target_probs, cfg.max_anchor_markers);
        if anchor_markers.is_empty() {
            return Self::default();
        }

        let mut leaves_map: HashMap<u64, Vec<RefHapId>> = HashMap::new();
        for h in 0..n_ref_haps {
            let hap = RefHapId::new(h as u32);
            let sig = hap_signature(hap, ref_columns, &anchor_markers);
            leaves_map.entry(sig).or_default().push(hap);
        }

        let mut leaves = Vec::new();
        for (signature, members) in leaves_map {
            if members.len() < cfg.min_leaf_size {
                continue;
            }
            let representatives = pick_representatives(&members, cfg.max_representatives_per_leaf);
            let enriched = compute_enriched_alleles(
                ref_columns,
                &members,
                n_ref_haps,
                cfg.rare_freq_max,
                cfg.min_enrichment,
                cfg.max_enriched_alleles_per_leaf,
            );
            if representatives.is_empty() {
                continue;
            }
            leaves.push(RareLeaf {
                signature,
                members,
                representatives,
                enriched,
            });
        }
        Self {
            anchor_markers,
            leaves,
        }
    }

    pub fn suggest_representatives(
        &self,
        target_probs: &TargetAlleleProbs,
        donors: &[(RefHapId, f32)],
        max_inject: usize,
    ) -> Vec<RefHapId> {
        if self.leaves.is_empty() || max_inject == 0 {
            return Vec::new();
        }
        let target_sig = target_signature(target_probs, &self.anchor_markers);
        let donor_set: std::collections::HashSet<RefHapId> =
            donors.iter().map(|(h, _)| *h).collect();

        let mut scored: Vec<(usize, f32, u32)> = Vec::new();
        for (idx, leaf) in self.leaves.iter().enumerate() {
            let dist = (leaf.signature ^ target_sig).count_ones() as f32;
            let mut score = 1.0 / (1.0 + dist);
            score += (leaf.members.len() as f32).ln_1p() * 0.05;
            for enr in &leaf.enriched {
                let p = target_probs
                    .probs_for_marker(enr.marker)
                    .get(enr.allele as usize)
                    .copied()
                    .unwrap_or(0.0)
                    .max(0.0);
                score += p * enr.enrichment.min(8.0) * 0.1;
            }
            let missing = leaf
                .representatives
                .iter()
                .filter(|h| !donor_set.contains(h))
                .count() as u32;
            if missing > 0 {
                scored.push((idx, score, missing));
            }
        }

        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| b.2.cmp(&a.2)));
        let mut out = Vec::new();
        for (leaf_idx, _, _) in scored {
            for &hap in &self.leaves[leaf_idx].representatives {
                if donor_set.contains(&hap) || out.contains(&hap) {
                    continue;
                }
                out.push(hap);
                if out.len() >= max_inject {
                    return out;
                }
            }
        }
        out
    }
}

fn select_anchor_markers(
    target_probs: &TargetAlleleProbs,
    max_anchor_markers: usize,
) -> Vec<usize> {
    let n = target_probs.n_markers();
    if n == 0 || max_anchor_markers == 0 {
        return Vec::new();
    }
    let mut informative = Vec::new();
    for m in 0..n {
        if !target_probs.is_uniform_marker(m) {
            informative.push(m);
        }
    }
    if informative.is_empty() {
        return Vec::new();
    }
    let want = max_anchor_markers.min(informative.len()).max(1);
    let mut out = Vec::with_capacity(want);
    for i in 0..want {
        let idx = i * informative.len() / want;
        out.push(informative[idx]);
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn hap_signature(hap: RefHapId, ref_columns: &[GenotypeColumn], anchors: &[usize]) -> u64 {
    let mut sig = 0u64;
    for (i, &m) in anchors.iter().enumerate().take(64) {
        let allele = ref_columns[m].get(HapIdx::new(hap.as_u32()));
        let bit = (allele as u64)
            .wrapping_mul(0x9e3779b97f4a7c15)
            .rotate_left((i % 63) as u32);
        sig ^= bit ^ (((m as u64) << (i % 16)) | 1);
    }
    sig
}

fn target_signature(target_probs: &TargetAlleleProbs, anchors: &[usize]) -> u64 {
    let mut sig = 0u64;
    for (i, &m) in anchors.iter().enumerate().take(64) {
        let probs = target_probs.probs_for_marker(m);
        let mut best_idx = 0usize;
        let mut best = f32::MIN;
        for (idx, &p) in probs.iter().enumerate() {
            if p > best {
                best = p;
                best_idx = idx;
            }
        }
        let bit = (best_idx as u64)
            .wrapping_mul(0x9e3779b97f4a7c15)
            .rotate_left((i % 63) as u32);
        sig ^= bit ^ (((m as u64) << (i % 16)) | 1);
    }
    sig
}

fn pick_representatives(members: &[RefHapId], max_representatives: usize) -> Vec<RefHapId> {
    if members.is_empty() || max_representatives == 0 {
        return Vec::new();
    }
    let want = members.len().min(max_representatives);
    let mut reps = Vec::with_capacity(want);
    for i in 0..want {
        let idx = i * members.len() / want;
        reps.push(members[idx]);
    }
    reps.sort_unstable_by_key(|h| h.as_u32());
    reps.dedup();
    reps
}

fn compute_enriched_alleles(
    ref_columns: &[GenotypeColumn],
    members: &[RefHapId],
    n_ref_haps: usize,
    rare_freq_max: f32,
    min_enrichment: f32,
    max_keep: usize,
) -> Vec<EnrichedAllele> {
    let mut enriched = Vec::new();
    for (m, col) in ref_columns.iter().enumerate() {
        let alt_freq = if n_ref_haps > 0 {
            col.alt_count() as f32 / n_ref_haps as f32
        } else {
            0.0
        };
        if alt_freq <= 0.0 || alt_freq > rare_freq_max {
            continue;
        }
        let mut leaf_alt = 0usize;
        for &hap in members {
            if col.get(HapIdx::new(hap.as_u32())) == 1 {
                leaf_alt += 1;
            }
        }
        if leaf_alt == 0 {
            continue;
        }
        let local_freq = leaf_alt as f32 / members.len() as f32;
        let enrichment = local_freq / alt_freq.max(1e-6);
        if enrichment >= min_enrichment {
            enriched.push(EnrichedAllele {
                marker: m,
                allele: 1,
                enrichment,
            });
        }
    }
    enriched.sort_by(|a, b| b.enrichment.total_cmp(&a.enrichment));
    enriched.truncate(max_keep);
    enriched
}

pub fn should_query_rare_atlas(
    target_probs: &TargetAlleleProbs,
    donors: &[(RefHapId, f32)],
) -> bool {
    let n = target_probs.n_markers();
    if n == 0 {
        return false;
    }
    let mut entropy_sum = 0.0f32;
    let mut informative = 0usize;
    let mut uncertain_rare_like = 0usize;
    for m in 0..n {
        if target_probs.is_uniform_marker(m) {
            continue;
        }
        let probs = target_probs.probs_for_marker(m);
        if probs.is_empty() {
            continue;
        }
        informative += 1;
        let mut entropy = 0.0f32;
        let mut max_p = 0.0f32;
        for &p in probs {
            if p > 0.0 {
                entropy -= p * p.ln();
                max_p = max_p.max(p);
            }
        }
        let max_ent = (probs.len().max(2) as f32).ln().max(1e-6);
        entropy_sum += (entropy / max_ent).clamp(0.0, 1.0);
        if probs.len() == 2 {
            let p_alt = probs[1].clamp(0.0, 1.0);
            if p_alt >= 0.02 && p_alt <= 0.35 && max_p <= 0.9 {
                uncertain_rare_like += 1;
            }
        }
    }
    if informative == 0 {
        return false;
    }
    let avg_entropy = entropy_sum / informative as f32;
    let uncertain_share = uncertain_rare_like as f32 / informative as f32;

    let mut donor_sum = 0.0f32;
    let mut top = 0.0f32;
    for &(_, w) in donors {
        if w.is_finite() && w > 0.0 {
            donor_sum += w;
            top = top.max(w);
        }
    }
    let top_share = if donor_sum > 0.0 {
        top / donor_sum
    } else {
        0.0
    };
    avg_entropy >= 0.45 || uncertain_share >= 0.10 || (donor_sum > 0.0 && top_share <= 0.25)
}

#[cfg(test)]
mod tests {
    use super::should_query_rare_atlas;
    use crate::model::impute_hmm::TargetAlleleProbs;

    #[test]
    fn detects_hard_region_from_entropy() {
        let n = 20usize;
        let mut offsets = Vec::with_capacity(n + 1);
        offsets.push(0);
        for i in 1..=n {
            offsets.push(i * 2);
        }
        let mut probs = Vec::with_capacity(n * 2);
        for _ in 0..n {
            probs.push(0.51);
            probs.push(0.49);
        }
        let observed = vec![true; n];
        let tp = TargetAlleleProbs::new(offsets, probs, observed, None, 0.0);
        assert!(should_query_rare_atlas(&tp, &[]));
    }
}
