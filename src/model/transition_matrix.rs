//! Sparse mapping between window-specific state sets.
//!
//! This maps posterior probabilities from window A to priors for window B when
//! the active state set changes (core + dynamic). Uses a CSR-style structure to
//! avoid dense O(N^2) remapping while conserving probability mass.

use crate::model::types::GlobalId;

#[derive(Debug, Clone, Copy)]
pub enum Redistribution {
    /// Do not spread dropped mass; keep only the overlapping state mass.
    None,
    /// Spread dropped mass uniformly across all next states.
    Uniform,
}

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
    pub fn build(prev_states: &[GlobalId], next_states: &[GlobalId]) -> Self {
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

    /// Map previous state probabilities into next-state space.
    pub fn map(&self, prev_probs: &[f32], redistribution: Redistribution) -> Vec<f32> {
        let mut next = vec![0.0f32; self.n_next];
        let mut kept_mass = 0.0f32;
        let mut total_mass = 0.0f32;

        for (i, p) in prev_probs.iter().enumerate() {
            if !p.is_finite() || *p <= 0.0 {
                continue;
            }
            total_mass += *p;
            let start = self.row_offsets[i];
            let end = self.row_offsets[i + 1];
            if start == end {
                continue;
            }
            for k in start..end {
                let j = self.col_indices[k];
                let w = self.weights[k];
                next[j] += *p * w;
                kept_mass += *p * w;
            }
        }

        if let Redistribution::Uniform = redistribution {
            let dropped = (total_mass - kept_mass).max(0.0);
            if dropped > 0.0 && self.n_next > 0 {
                let add = dropped / self.n_next as f32;
                for v in next.iter_mut() {
                    *v += add;
                }
            }
        }

        let mut sum = 0.0f32;
        for v in next.iter() {
            if v.is_finite() && *v > 0.0 {
                sum += *v;
            }
        }
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for v in next.iter_mut() {
                *v *= inv;
            }
        }
        next
    }
}
