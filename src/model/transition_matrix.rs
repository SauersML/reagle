//! Sparse mapping between window-specific state sets.
//!
//! This maps posterior probabilities from window A to priors for window B when
//! the active state set changes (core + dynamic). Uses a CSR-style structure to
//! avoid dense O(N^2) remapping while conserving probability mass.

use crate::model::types::RefHapId;

/// Sparse CSR transition matrix from previous to next window state sets.
pub struct TransitionMatrix {
    /// Row offsets into col_indices/weights (len = n_prev + 1)
    row_offsets: Vec<usize>,
    /// Column indices in next-state space
    col_indices: Vec<usize>,
    /// Weights for each mapping entry
    weights: Vec<f32>,
    /// Number of states in next window
    n_next: usize,
}

impl TransitionMatrix {
    pub fn build(prev_states: &[RefHapId], next_states: &[RefHapId]) -> Self {
        let n_prev = prev_states.len();
        let n_next = next_states.len();
        let mut row_offsets = Vec::with_capacity(n_prev + 1);
        let mut col_indices = Vec::new();
        let mut weights = Vec::new();

        let mut next_index = std::collections::HashMap::with_capacity(n_next * 2);
        for (j, gid) in next_states.iter().enumerate() {
            next_index.insert(*gid, j);
        }

        row_offsets.push(0);
        for gid in prev_states {
            if let Some(&j) = next_index.get(gid) {
                col_indices.push(j);
                weights.push(1.0);
            }
            row_offsets.push(col_indices.len());
        }

        Self {
            row_offsets,
            col_indices,
            weights,
            n_next,
        }
    }

    /// Map previous state probabilities into next-state space, reusing `out` storage.
    ///
    /// Dropped mass is redistributed proportionally to retained mass when possible.
    pub fn map_into(&self, prev_probs: &[f32], out: &mut Vec<f32>) {
        out.resize(self.n_next, 0.0);
        let next = &mut out[..];
        next.fill(0.0);
        if self.n_next == 0 || prev_probs.is_empty() {
            return;
        }

        let mut total_mass = 0.0f32;
        for p in prev_probs.iter() {
            if p.is_finite() && *p > 0.0 {
                total_mass += *p;
            }
        }
        if total_mass <= 0.0 {
            let uniform = 1.0 / self.n_next as f32;
            for v in next.iter_mut() {
                *v = uniform;
            }
            return;
        }
        let inv_total = 1.0 / total_mass;

        let mut kept_mass = 0.0f32;
        let mut sum_next = 0.0f32;
        for (i, p) in prev_probs.iter().enumerate() {
            if !p.is_finite() || *p <= 0.0 {
                continue;
            }
            let p_norm = *p * inv_total;
            let start = self.row_offsets[i];
            let end = self.row_offsets[i + 1];
            if start == end {
                continue;
            }
            for k in start..end {
                let j = self.col_indices[k];
                let w = self.weights[k];
                let add = p_norm * w;
                next[j] += add;
                kept_mass += add;
                sum_next += add;
            }
        }

        let dropped = (1.0 - kept_mass).max(0.0);
        if dropped > 0.0 {
            if sum_next > 0.0 {
                let scale = dropped / sum_next;
                for v in next.iter_mut() {
                    *v += *v * scale;
                }
            } else {
                let add = dropped / self.n_next as f32;
                for v in next.iter_mut() {
                    *v += add;
                }
            }
        }
    }
}
