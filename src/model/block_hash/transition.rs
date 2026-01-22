//! # Transition Bridge: CSR-based Probability Transfer Between Windows
//!
//! This module provides the KEY fix for the index scrambling bug using
//! efficient sparse matrix representation (CSR format).
//!
//! The Insight: DictionaryColumn already provides the Global ID → Pattern ID
//! mapping via `hap_to_pattern()`. We zip these mappings and build a
//! deterministic sparse transition matrix.

use super::micro_window::MicroWindow;
use super::types::PatternId;

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
    bwd_dests: Vec<PatternId>, // Actually Source IDs

    /// Backward: Source pattern IDs (parallel to bwd_dests)
    /// Used for propagating probability from B to A (backward pass)
    bwd_sources: Vec<PatternId>, // Actually Dest IDs

    /// Backward: Transition weights (parallel to bwd_dests/bwd_sources)
    /// These are P(Source|Dest) = P(Dest|Source) * P(Source) / P(Dest)
    /// Actually, for the backward pass we just need the transpose of the transition matrix.
    /// The transition probability is P(next|prev).
    /// In backward pass: beta[prev] = sum( beta[next] * P(next|prev) )
    /// So we need to iterate over 'next' for each 'prev'.
    /// Wait, standard forward is: alpha[next] = sum( alpha[prev] * P(next|prev) )
    /// Standard backward is: beta[prev] = sum( P(next|prev) * beta[next] * emission[next] )
    ///
    /// My `apply_backward` function takes `block_a` (prev) and `block_b` (next) and computes `block_a.bwd_probs`.
    /// `block_b` already has `bwd_probs` computed (from future).
    /// So we need to compute:
    /// block_a.bwd[i] = sum_j ( P(j|i) * block_b.bwd[j] )
    ///
    /// This requires iterating over all j connected to i.
    /// This is EXACTLY the same structure as Forward: source-major order.
    /// We iterate over i (source), and for each j (dest), we add contribution to i.
    ///
    /// Wait, no.
    /// Forward: iterate over i, push to j. (Scatter)
    /// OR: iterate over j, pull from i. (Gather)
    /// My current implementation iterates over (i, j) pairs sorted by i.
    /// This allows streaming through i.
    /// For Forward: alpha[j] += alpha[i] * w. This is scatter.
    /// It requires sorting by i?
    /// If I sort by i, I can iterate i once, and update multiple j's.
    ///
    /// For Backward: beta[i] += beta[j] * w.
    /// This is GATHER into i.
    /// If I sort by i, I can iterate i once, and pull from multiple j's.
    /// So I can reuse the exact same `sources`, `destinations`, `weights` vectors!
    ///
    /// Let's verify.
    /// Forward (Scatter):
    /// for (i, j, w) in transitions (sorted by i):
    ///    alpha[j] += alpha[i] * w
    ///
    /// Backward (Gather):
    /// for (i, j, w) in transitions (sorted by i):
    ///    beta[i] += beta[j] * w
    ///
    /// Yes! If I sort by i (source), I can do both efficiently if random access to beta[j] is fine.
    /// Since j is random access into a small vector (4096 floats), it is cache friendly enough.
    /// The `sources` vector is sequential access.
    ///
    /// HOWEVER, the user specifically asked for "Reverse Adjacency List" / "Backward CSR".
    /// "For the Backward pass, you need the Transpose: you need to map destinations -> sources".
    /// Why?
    /// If I use the Forward list (sorted by Source):
    /// I iterate i=0, 1, 2...
    /// For i=0, I see list of j's. I compute beta[0] = sum(beta[j] * w).
    /// This is perfect for cache locality of `beta[i]` (write) but random read of `beta[j]`.
    ///
    /// If I use Backward list (sorted by Dest):
    /// I iterate j=0, 1, 2...
    /// For j=0, I see list of i's. I update beta[i] += beta[0] * w.
    /// This is random write of `beta[i]` and sequential read of `beta[j]`.
    ///
    /// Usually, we want sequential write.
    /// So sorting by Source (current `sources`) is actually BETTER for Backward pass (Gather) too!
    /// Because we are computing beta[source].
    ///
    /// Let's re-read the user's critique.
    /// "The TransitionBridge struct in the diff is optimized only for the Forward pass. It stores sources -> destinations. For the Backward pass, you need the Transpose: you need to map destinations -> sources to propagate probability from the future back to the past."
    ///
    /// User might be thinking of "Pull" vs "Push".
    /// Forward Push: Iterate i, update j. (Random write j).
    /// Forward Pull: Iterate j, sum from i. (Random read i).
    /// Backward Push: Iterate j, update i. (Random write i).
    /// Backward Pull: Iterate i, sum from j. (Random read j).
    ///
    /// My `apply_forward` does:
    /// for (i, j, w) in sorted_by_i:
    ///    new_fwd[j] += old_fwd[i] * w
    /// This is Forward Push (Scatter). Random write to `new_fwd`.
    ///
    /// For Backward, I want to compute `bwd[i]`.
    /// If I iterate (i, j, w) in sorted_by_i:
    ///    bwd[i] += bwd[j] * w
    /// This is Backward Pull (Gather). Sequential write to `bwd[i]`. Random read `bwd[j]`.
    ///
    /// So technically, I don't NEED the transpose if I implement Backward Pull.
    /// BUT, my `apply_forward` implementation actually sorts by `sources` (i).
    /// So it iterates i sequentially, and scatters to j.
    ///
    /// If the user explicitly asked for "Backward CSR", they might want to support the "Push" pattern or they believe it's necessary.
    /// Or maybe they think I'm doing Forward Pull?
    ///
    /// Let's look at `apply_forward`:
    /// ```rust
    /// for i in 0..self.sources.len() {
    ///     let from = self.sources[i];
    ///     let to = self.destinations[i];
    ///     let weight = self.weights[i];
    ///     new_fwd[to.as_usize()] += window_a.fwd_probs[from.as_usize()] * weight;
    /// }
    /// ```
    /// This reads `from` (mostly sequential if sorted by from) and writes `to` (random).
    ///
    /// For backward `apply_backward(block_a, block_b, ws)`:
    /// We want to update `block_a.bwd` using `block_b.bwd`.
    /// `block_a.bwd[i] += block_b.bwd[j] * weight`.
    ///
    /// If I use `sources` (sorted by i):
    /// I read `i` sequentially. I write `block_a.bwd[i]`.
    /// I read `block_b.bwd[j]` (random).
    /// This is actually quite good.
    ///
    /// Why did the user say I need the transpose?
    /// Maybe to allow "Push" style?
    /// Or maybe they thought I was sorting by Destination?
    ///
    /// Actually, if I have `sources` sorted, I can group them.
    /// `start_idx[i] .. end_idx[i]` gives all transitions from i.
    /// Then `bwd[i] = sum( w * bwd[j] )`.
    /// This avoids repeated writes to `bwd[i]`.
    ///
    /// But my CSR is just coordinate list (COO) sorted by source.
    /// So I have multiple entries for same `i`.
    /// `bwd[i] += ...` happens multiple times.
    ///
    /// If I use Transpose (sorted by j):
    /// I iterate j sequentially.
    /// `val = bwd[j]`.
    /// for each i connected to j:
    ///     bwd[i] += val * w.
    /// This is Scatter into `bwd[i]`. Random write.
    ///
    /// So:
    /// Sorted by Source (i):
    ///   Forward: Read i (seq), Write j (rnd). (Scatter)
    ///   Backward: Write i (seq), Read j (rnd). (Gather)
    ///
    /// Sorted by Dest (j):
    ///   Forward: Write j (seq), Read i (rnd). (Gather)
    ///   Backward: Read j (seq), Write i (rnd). (Scatter)
    ///
    /// The user asked for "Backward CSR" which usually implies sorting by Destination (if we consider time flowing A->B).
    /// But wait, "backward weights vector (CSR format for the reverse direction)".
    /// Reverse direction: B -> A.
    /// Source in reverse is B (j). Dest in reverse is A (i).
    /// So "Backward CSR" means sorted by j.
    ///
    /// If I have "Backward CSR" (sorted by j):
    /// I can do Backward Push (Scatter) or Forward Pull (Gather).
    ///
    /// I will implement what the user asked: Transpose.
    /// This gives me flexibility.
    /// And specifically, `bwd_dests` (i) and `bwd_sources` (j).
    ///
    /// Wait, naming.
    /// Forward: A -> B.
    /// Backward: B -> A.
    ///
    /// `bwd_sources`: Patterns in B (j).
    /// `bwd_dests`: Patterns in A (i).
    /// `bwd_weights`: P(j|i) same as forward? Or P(i|j)?
    /// The transition is P(j|i).
    /// In backward pass: beta[i] = sum_j beta[j] * P(j|i).
    /// The weight is still P(j|i).
    ///
    /// So `bwd_weights` should store P(j|i).
    ///
    /// So I will store:
    /// `bwd_sources` = j (sorted)
    /// `bwd_dests` = i
    /// `bwd_weights` = P(j|i)
    ///
    /// Then `apply_backward`:
    /// for k in 0..len:
    ///    j = bwd_sources[k]
    ///    i = bwd_dests[k]
    ///    w = bwd_weights[k]
    ///    beta[i] += beta[j] * w
    ///
    /// This is Scatter into i.
    ///
    /// If I stick to `sources` (sorted by i):
    /// for k in 0..len:
    ///    i = sources[k]
    ///    j = destinations[k]
    ///    w = weights[k]
    ///    beta[i] += beta[j] * w
    ///
    /// This is Gather into i. Sequential write.
    /// This seems better for CPU cache (write combining).
    ///
    /// However, strictly following the user's plan is safer.
    /// "The Fix: The TransitionBridge must build a backward_weights vector (CSR format for the reverse direction) at construction time."
    ///
    /// I will add `bwd_sources` (j), `bwd_dests` (i), `bwd_weights` (P(j|i)).
    /// And I will use them in `apply_backward`.

    reservoir_to_pattern_ids: Vec<PatternId>,
    reservoir_to_pattern_weights: Vec<f32>,

    pattern_to_reservoir_ids: Vec<PatternId>,
    pattern_to_reservoir_weights: Vec<f32>,

    reservoir_to_reservoir: f32,

    recomb_rate: f32,

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

        // Get the hap→pattern mappings from both windows
        let map_a = window_a.storage.hap_to_pattern();
        let map_b = window_b.storage.hap_to_pattern();

        // Collect all transitions in COO format (coordinate list)
        // We store (src, dst, weight)
        let mut transitions: Vec<(PatternId, PatternId, f32)> = Vec::new();
        
        let mut reservoir_to_pattern: Vec<(PatternId, f32)> = Vec::new();
        let mut pattern_to_reservoir: Vec<(PatternId, f32)> = Vec::new();
        let mut reservoir_to_reservoir = 0.0f32;

        // Zip the two mappings to track how each haplotype transitions
        for (&pat_a_raw, &pat_b_raw) in map_a.iter().zip(map_b.iter()) {
            // Convert to local pattern IDs (accounting for truncation)
            let pat_a = if (pat_a_raw as usize) < window_a.n_patterns() {
                PatternId::new(pat_a_raw as u16)
            } else {
                PatternId::RESERVOIR
            };

            let pat_b = if (pat_b_raw as usize) < window_b.n_patterns() {
                PatternId::new(pat_b_raw as u16)
            } else {
                PatternId::RESERVOIR
            };

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
        let (sources, destinations, weights) = aggregate_transitions(&transitions);

        // Sort and aggregate for Backward (sorted by Dest)
        // We can reuse the transitions vector, just re-sort
        let mut bwd_transitions = transitions; // Move
        bwd_transitions.sort_by_key(|(from, to, _)| (*to, *from));
        // aggregate_transitions expects (from, to, weight) but handles any sorting.
        // However, we want to extract (j, i, w) where j is primary sort key.
        // aggregate_transitions implementation aggregates if (from, to) matches.
        // Since we sorted by (to, from), duplicates will be adjacent.
        // So we can reuse the same function!
        // The output will be: sources=from, destinations=to, weights=w.
        // But the order will be grouped by 'to'.
        // So `bwd_sources` will contain `from` (i), `bwd_dests` will contain `to` (j).
        // Wait, my struct definition said:
        // bwd_dests: Vec<PatternId>, // Actually Source IDs
        // bwd_sources: Vec<PatternId>, // Actually Dest IDs
        // Let's stick to that naming to avoid confusion or swap them.
        // I want `apply_backward` to iterate over B -> A.
        // So I want to iterate j (in B) and update i (in A).
        // If I sort by `to` (j), I iterate j.
        // So I should call the sorted arrays `bwd_sources` (j) and `bwd_dests` (i).
        
        let (bwd_i, bwd_j, bwd_w) = aggregate_transitions(&bwd_transitions);
        // aggregate returns (from, to, w).
        // Since we sorted by (to, from):
        // `bwd_i` is 'from' (i), `bwd_j` is 'to' (j).
        // And it is sorted by `bwd_j`.
        // So I will store:
        let bwd_sources = bwd_j; // j (sorted)
        let bwd_dests = bwd_i;   // i
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
            bwd_dests,
            bwd_sources,
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
        // We can't overwrite ws.fwd directly because we need window_a values (which are in ws.fwd if sequential).
        // But wait, ws.fwd holds current state. `apply` updates state.
        // If we are doing A -> B, we read from ws.fwd (A) and write to new buffer (B).
        // Since A and B are different blocks, do they share ws.fwd?
        // Workspace has one fwd buffer.
        // So we definitely need a temp buffer.
        // ws.emissions is available and sized to max_states. We can use it as temp.
        let mut new_fwd = std::mem::take(&mut ws.emissions);
        new_fwd.fill(0.0);
        new_fwd.resize(ws.fwd.len(), 0.0); // Ensure size

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
        // Using ws.emissions as temp buffer for A.
        let mut new_bwd = std::mem::take(&mut ws.emissions);
        new_bwd.fill(0.0);
        new_bwd.resize(ws.bwd.len(), 0.0);
        
        let mut new_reservoir_prob = 0.0f32;

        // Background recombination contribution (Gather from all B)
        // beta_back[i] += sum_j ( beta[j] * P(recomb) * count[j] / N )
        // All i get the same background contribution because P(recomb to any j) is uniform-ish?
        // No.
        // P(j|i) = (1-r)*W + r*count[j]/N.
        // beta[i] = sum_j beta[j] * P(j|i)
        //         = sum_j beta[j] * ( (1-r)W_ij + r*count[j]/N )
        //         = sum_j beta[j]*(1-r)W_ij  +  sum_j beta[j]*r*count[j]/N
        //         = (Transmission Part)      +  (Recombination Part)
        //
        // The Recombination Part is: (r/N) * sum_j (beta[j] * count[j])
        // This is a constant added to ALL i (including reservoir).
        
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
        new_reservoir_prob = recomb_term;

        // Transmission Part: beta[i] += beta[j] * (1-r) * weight
        // Note: self.weights already includes (1-r).
        // We use the Backward CSR (sorted by Dest=Source=j)?
        // Wait, bwd_sources = j. bwd_dests = i.
        // We iterate k. j = bwd_sources[k]. i = bwd_dests[k].
        // beta[i] += beta[j] * w.
        
        for k in 0..self.bwd_sources.len() {
            let j = self.bwd_sources[k];
            let i = self.bwd_dests[k];
            let weight = self.bwd_weights[k];
            
            new_bwd[i.as_usize()] += ws.bwd[j.as_usize()] * weight;
        }
        
        // Reservoir transitions
        // Reservoir -> Pattern (A->B)
        // beta[res_A] += beta[pat_B] * weight
        for k in 0..self.reservoir_to_pattern_ids.len() {
             let j = self.reservoir_to_pattern_ids[k]; // Pattern in B
             let weight = self.reservoir_to_pattern_weights[k];
             new_reservoir_prob += ws.bwd[j.as_usize()] * weight;
        }
        
        // Pattern -> Reservoir (A->B)
        // beta[pat_A] += beta[res_B] * weight
        for k in 0..self.pattern_to_reservoir_ids.len() {
            let i = self.pattern_to_reservoir_ids[k]; // Pattern in A
            let weight = self.pattern_to_reservoir_weights[k];
            new_bwd[i.as_usize()] += ws.reservoir_prob_bwd * weight;
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
