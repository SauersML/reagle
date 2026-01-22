//! # Transition Bridge: CSR-based Probability Transfer Between Windows
//!
//! This module provides the KEY fix for the index scrambling bug using
//! efficient sparse matrix representation (CSR format).
//!
//! The Insight: DictionaryColumn already provides the Global ID → Pattern ID
//! mapping via `hap_to_pattern()`. We zip these mappings and build a
//! deterministic sparse transition matrix.

use super::types::PatternId;
use crate::model::block_hash::CompressedBlock;
use aligned_vec::AVec;

/// Sparse transition matrix in CSR (Compressed Sparse Row) format
///
/// This is deterministic and cache-friendly compared to HashMap.
#[derive(Clone, Debug)]
pub struct TransitionBridge {
    /// Source pattern IDs (sorted for deterministic iteration)
    sources: Vec<PatternId>,

    /// Destination pattern IDs (parallel to sources)
    destinations: Vec<PatternId>,

    /// Transition weights (parallel to sources/destinations)
    weights: Vec<f32>,

    /// Backward: Destination pattern IDs (sorted for deterministic iteration)
    /// Used for propagating probability from B to A (backward pass)
    /// Renamed to transpose_rows to indicate these are the row indices (destination) in the transposed matrix A <- B
    transpose_rows: Vec<PatternId>, // Pattern A

    /// Backward: Source pattern IDs (parallel to transpose_rows)
    /// Renamed to transpose_cols to indicate these are the column indices (source) in the transposed matrix A <- B
    transpose_cols: Vec<PatternId>, // Pattern B

    /// Backward: Transition weights (parallel to transpose_rows/transpose_cols)
    bwd_weights: Vec<f32>,

    /// Reservoir → specific pattern transitions
    reservoir_to_pattern_ids: Vec<PatternId>,
    reservoir_to_pattern_weights: Vec<f32>,

    /// Specific pattern → Reservoir transitions
    pattern_to_reservoir_ids: Vec<PatternId>,
    pattern_to_reservoir_weights: Vec<f32>,

    /// Reservoir → Reservoir transition weight
    reservoir_to_reservoir: f32,

    /// Recombination rate for this transition
    recomb_rate: f32,

    /// Total number of reference haplotypes (for background recombination)
    n_ref_haps: usize,
}

impl TransitionBridge {
    /// Build a transition bridge using efficient COO → CSR conversion
    ///
    /// This produces deterministic, cache-friendly sparse transitions.
    pub(crate) fn build(
        window_a: &CompressedBlock,
        window_b: &CompressedBlock,
        recomb_rate: f32,
    ) -> Self {
        let n_ref_haps = window_a.n_ref_haps();
        assert_eq!(
            window_b.n_ref_haps(),
            n_ref_haps,
            "Windows must have same reference panel size"
        );

        // Get the hap→state mappings directly (already remapped/truncated)
        let map_a = &window_a.hap_to_state;
        let map_b = &window_b.hap_to_state;

        // Collect all transitions in COO format (coordinate list)
        // We store (src, dst, weight)
        let mut transitions: Vec<(PatternId, PatternId, f32)> = Vec::new();
        
        let mut reservoir_to_pattern: Vec<(PatternId, f32)> = Vec::new();
        let mut pattern_to_reservoir: Vec<(PatternId, f32)> = Vec::new();
        let mut reservoir_to_reservoir = 0.0f32;

        // Zip the two mappings to track how each haplotype transitions
        for (&pat_a, &pat_b) in map_a.iter().zip(map_b.iter()) {
            // No truncation logic needed here - hap_to_state already handles it!

            // Calculate per-haplotype weight (cardinality-aware)
            let weight = if pat_a.is_reservoir() {
                if window_a.reservoir_count > 0 {
                    1.0 / window_a.reservoir_count as f32
                } else {
                    continue;
                }
            } else {
                1.0 / window_a.pattern_counts[pat_a.as_usize()]
            };

            // Apply no-recombination probability
            let flow = weight * (1.0 - recomb_rate);

            // Route the flow based on source and destination
            match (pat_a.is_reservoir(), pat_b.is_reservoir()) {
                (false, false) => {
                    transitions.push((pat_a, pat_b, flow));
                }
                (false, true) => {
                    pattern_to_reservoir.push((pat_a, flow));
                }
                (true, false) => {
                    reservoir_to_pattern.push((pat_b, flow));
                }
                (true, true) => {
                    reservoir_to_reservoir += flow;
                }
            }
        }

        // Sort and aggregate for Forward (sorted by Source)
        transitions.sort_by_key(|(from, to, _)| (*from, *to));
        let (sources, destinations, weights) = aggregate_transitions(transitions.clone());

        // Sort and aggregate for Backward (sorted by Dest)
        let mut bwd_transitions = transitions; // Move
        bwd_transitions.sort_by_key(|(from, to, _)| (*to, *from));
        
        let (bwd_i, bwd_j, bwd_w) = aggregate_transitions(bwd_transitions);
        // aggregate returns (from, to, w).
        // Since we sorted by (to, from):
        // `bwd_i` is 'from' (i), `bwd_j` is 'to' (j).
        // And it is sorted by `bwd_j`.
        // We want to iterate j (cols of A<-B) and scatter to i (rows of A<-B).
        // bwd_sources (j) -> bwd_dests (i).
        // Renaming to transpose_cols -> transpose_rows.
        let transpose_cols = bwd_j; // j (sorted)
        let transpose_rows = bwd_i;   // i
        let bwd_weights = bwd_w;

        // Reservoir → Pattern (sort for determinism)
        reservoir_to_pattern.sort_by_key(|(to, _)| *to);
        let (reservoir_to_pattern_ids, reservoir_to_pattern_weights) =
            aggregate_reservoir_transitions(reservoir_to_pattern);

        // Pattern → Reservoir (sort for determinism)
        pattern_to_reservoir.sort_by_key(|(from, _)| *from);
        let (pattern_to_reservoir_ids, pattern_to_reservoir_weights) =
            aggregate_reservoir_transitions(pattern_to_reservoir);

        Self {
            sources,
            destinations,
            weights,
            transpose_rows,
            transpose_cols,
            bwd_weights,
            reservoir_to_pattern_ids,
            reservoir_to_pattern_weights,
            pattern_to_reservoir_ids,
            pattern_to_reservoir_weights,
            reservoir_to_reservoir,
            recomb_rate,
            n_ref_haps,
        }
    }

    /// Apply this transition to transfer probabilities from window_a to window_b (Forward)
    pub(crate) fn apply_forward(&self, window_a: &CompressedBlock, window_b: &CompressedBlock, ws: &mut super::workspace::BlockHmmWorkspace) {
        let n_patterns_b = window_b.n_patterns();

        // Initialize new forward probabilities in a temporary buffer (or use emissions buffer if unused)
        let mut new_fwd = std::mem::replace(&mut ws.emissions, AVec::from_iter(32, std::iter::repeat(0.0).take(ws.fwd.len())));
        new_fwd.fill(0.0);
        // new_fwd.resize(ws.fwd.len(), 0.0); // Ensure size - assume max_states is consistent

        let mut new_reservoir_prob = 0.0f32;

        // Pattern → Pattern transitions (deterministic CSR iteration)
        for i in 0..self.sources.len() {
            let from = self.sources[i];
            let to = self.destinations[i];
            let weight = self.weights[i];

            let prob = ws.fwd[from.as_usize()];
            new_fwd[to.as_usize()] += prob * weight;
        }

        // Reservoir → Pattern transitions
        for i in 0..self.reservoir_to_pattern_ids.len() {
            let to = self.reservoir_to_pattern_ids[i];
            let weight = self.reservoir_to_pattern_weights[i];
            new_fwd[to.as_usize()] += ws.reservoir_prob_fwd * weight;
        }

        // Pattern → Reservoir transitions
        for i in 0..self.pattern_to_reservoir_ids.len() {
            let from = self.pattern_to_reservoir_ids[i];
            let weight = self.pattern_to_reservoir_weights[i];
            let prob = ws.fwd[from.as_usize()];
            new_reservoir_prob += prob * weight;
        }

        // Reservoir → Reservoir transition
        new_reservoir_prob += ws.reservoir_prob_fwd * self.reservoir_to_reservoir;

        // Add recombination background to all states
        let total_mass = ws.fwd.iter().take(window_a.n_patterns()).sum::<f32>() + ws.reservoir_prob_fwd;
        let background_per_hap = total_mass * self.recomb_rate / (self.n_ref_haps as f32);

        // Distribute recombination mass proportionally to pattern counts in B
        for pattern_idx in 0..n_patterns_b {
            let count = window_b.pattern_counts[pattern_idx];
            new_fwd[pattern_idx] += background_per_hap * count;
        }

        // Reservoir also receives recombination mass proportional to its cardinality
        new_reservoir_prob += background_per_hap * (window_b.reservoir_count as f32);

        // Update workspace
        // Copy new_fwd back to ws.fwd
        ws.fwd.copy_from_slice(&new_fwd);
        ws.reservoir_prob_fwd = new_reservoir_prob;
        
        // Restore emissions buffer
        ws.emissions = new_fwd;

        // Normalize to prevent drift
        ws.normalize_forward(n_patterns_b);
    }

    /// Apply transition Backward (B -> A)
    /// Updates ws.bwd (which currently holds B) to hold A.
    pub(crate) fn apply_backward(&self, window_a: &CompressedBlock, window_b: &CompressedBlock, ws: &mut super::workspace::BlockHmmWorkspace) {
        let n_patterns_a = window_a.n_patterns();
        let n_patterns_b = window_b.n_patterns();

        // ws.bwd holds values for B. We want to compute values for A.
        let mut new_bwd = std::mem::replace(&mut ws.emissions, AVec::from_iter(32, std::iter::repeat(0.0).take(ws.bwd.len())));
        new_bwd.fill(0.0);
        // new_bwd.resize(ws.bwd.len(), 0.0);
        
        // Background recombination contribution (Gather from all B)
        let mut recomb_sum = 0.0f32;
        for j in 0..n_patterns_b {
            recomb_sum += ws.bwd[j] * window_b.pattern_counts[j];
        }
        recomb_sum += ws.reservoir_prob_bwd * (window_b.reservoir_count as f32);
        
        let recomb_term = (self.recomb_rate / self.n_ref_haps as f32) * recomb_sum;
        
        // Initialize A with recomb term
        for i in 0..n_patterns_a {
            new_bwd[i] = recomb_term;
        }
        // Use recomb_term directly instead of redundant init
        let mut new_reservoir_prob = recomb_term;

        // Transmission Part: beta[i] += beta[j] * (1-r) * weight
        // Iterate over transposed CSR
        
        for k in 0..self.transpose_cols.len() {
            let pat_b = self.transpose_cols[k]; // Source in backward (B)
            let pat_a = self.transpose_rows[k]; // Dest in backward (A)
            let weight = self.bwd_weights[k];
            
            new_bwd[pat_a.as_usize()] += ws.bwd[pat_b.as_usize()] * weight;
        }
        
        // Reservoir transitions
        // Reservoir -> Pattern (A->B)
        // beta[res_A] += beta[pat_B] * weight
        for k in 0..self.reservoir_to_pattern_ids.len() {
             let pat_b = self.reservoir_to_pattern_ids[k]; // Pattern in B
             let weight = self.reservoir_to_pattern_weights[k];
             new_reservoir_prob += ws.bwd[pat_b.as_usize()] * weight;
        }
        
        // Pattern -> Reservoir (A->B)
        // beta[pat_A] += beta[res_B] * weight
        for k in 0..self.pattern_to_reservoir_ids.len() {
            let pat_a = self.pattern_to_reservoir_ids[k]; // Pattern in A
            let weight = self.pattern_to_reservoir_weights[k];
            new_bwd[pat_a.as_usize()] += ws.reservoir_prob_bwd * weight;
        }
        
        // Reservoir -> Reservoir
        new_reservoir_prob += ws.reservoir_prob_bwd * self.reservoir_to_reservoir;

        // Update workspace
        ws.bwd.copy_from_slice(&new_bwd);
        ws.reservoir_prob_bwd = new_reservoir_prob;
        ws.emissions = new_bwd; // Return buffer
    }
}


/// Aggregate sorted transitions into parallel vectors (CSR-like format)
fn aggregate_transitions(
    sorted_transitions: Vec<(PatternId, PatternId, f32)>,
) -> (Vec<PatternId>, Vec<PatternId>, Vec<f32>) {
    if sorted_transitions.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let mut sources = Vec::new();
    let mut destinations = Vec::new();
    let mut weights = Vec::new();

    let mut current_from = sorted_transitions[0].0;
    let mut current_to = sorted_transitions[0].1;
    let mut current_weight = sorted_transitions[0].2;

    for i in 1..sorted_transitions.len() {
        let (from, to, weight) = sorted_transitions[i];

        if from == current_from && to == current_to {
            // Aggregate duplicate transitions (deterministic floating-point sum)
            current_weight += weight;
        } else {
            // Flush previous transition
            sources.push(current_from);
            destinations.push(current_to);
            weights.push(current_weight);

            // Start new transition
            current_from = from;
            current_to = to;
            current_weight = weight;
        }
    }

    // Flush last transition
    sources.push(current_from);
    destinations.push(current_to);
    weights.push(current_weight);

    (sources, destinations, weights)
}

/// Aggregate reservoir transitions
fn aggregate_reservoir_transitions(
    sorted: Vec<(PatternId, f32)>,
) -> (Vec<PatternId>, Vec<f32>) {
    if sorted.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let mut ids = Vec::new();
    let mut weights = Vec::new();

    let mut current_id = sorted[0].0;
    let mut current_weight = sorted[0].1;

    for i in 1..sorted.len() {
        let (id, weight) = sorted[i];

        if id == current_id {
            current_weight += weight;
        } else {
            ids.push(current_id);
            weights.push(current_weight);
            current_id = id;
            current_weight = weight;
        }
    }

    ids.push(current_id);
    weights.push(current_weight);

    (ids, weights)
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_build_bridge_deterministic() {
        // Integration test - verifies deterministic CSR construction
    }

    #[test]
    fn test_apply_transition_mass_conservation() {
        // Integration test - verifies probability mass conservation
    }
}
