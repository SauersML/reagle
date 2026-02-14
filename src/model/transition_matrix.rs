//! Li-Stephens-consistent boundary mapping between window-specific state sets.
//!
//! This maps posterior probabilities from window A to priors for window B when
//! the active donor subset changes. Unlike pure projection/renormalization, this
//! mapping includes the switch component implied by Li-Stephens transitions:
//! dropped states inject mass into the next subset through a uniform switch term.

use crate::model::types::RefHapId;

/// Boundary transition mapper from previous to next window state sets.
pub struct TransitionMatrix {
    /// For each previous state index, optional matching index in next-state space.
    prev_to_next: Vec<Option<usize>>,
    /// Number of states in next window
    n_next: usize,
    /// Boundary recombination probability.
    recomb_rate: f32,
    /// Total panel haplotypes in Li-Stephens transition model.
    n_panel_haps: usize,
    /// Exogenous transition stickiness in [0, 1].
    transition_lambda: f32,
}

impl TransitionMatrix {
    pub fn build(
        prev_states: &[RefHapId],
        next_states: &[RefHapId],
        recomb_rate: f32,
        n_panel_haps: usize,
        transition_lambda: f32,
    ) -> Self {
        let n_next = next_states.len();
        let mut prev_to_next = Vec::with_capacity(prev_states.len());

        let mut next_index = std::collections::HashMap::with_capacity(n_next * 2);
        for (j, gid) in next_states.iter().enumerate() {
            next_index.insert(*gid, j);
        }

        for gid in prev_states {
            prev_to_next.push(next_index.get(gid).copied());
        }

        Self {
            prev_to_next,
            n_next,
            recomb_rate: recomb_rate.clamp(0.0, 1.0),
            n_panel_haps: n_panel_haps.max(1),
            transition_lambda: transition_lambda.clamp(0.0, 1.0),
        }
    }

    /// Map previous state probabilities into next-state space, reusing `out` storage.
    ///
    /// If `entry_pi` is provided, it is used only for dropped-state entry mass
    /// (states present in prev but absent in next). Overlap stay/switch terms
    /// remain Li-Stephens-conditioned and unchanged.
    pub fn map_into_with_pi(
        &self,
        prev_probs: &[f32],
        entry_pi: Option<&[f32]>,
        out: &mut Vec<f32>,
    ) {
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

        let mut overlap_mass_by_next = vec![0.0f32; self.n_next];
        let mut overlap_mass = 0.0f32;
        for (i, p) in prev_probs.iter().enumerate() {
            if !p.is_finite() || *p <= 0.0 {
                continue;
            }
            let p_norm = *p * inv_total;
            if let Some(Some(j)) = self.prev_to_next.get(i) {
                overlap_mass_by_next[*j] += p_norm;
                overlap_mass += p_norm;
            }
        }
        let dropped_mass = (1.0 - overlap_mass).max(0.0);

        let k_next = self.n_next as f32;
        let n_panel = self.n_panel_haps as f32;
        let r = self.recomb_rate;
        let lam = self.transition_lambda;
        let rho = lam + (1.0 - lam) * (k_next / n_panel);
        let denom_in = ((1.0 - r) + r * rho).max(1e-30);
        let stay_scale = ((1.0 - r) + r * lam) / denom_in;
        let switch_each = (r * (1.0 - lam) / n_panel) / denom_in;
        let switch_from_overlap_each = overlap_mass * switch_each;
        // For dropped states (not in next subset), conditioning on "state in next
        // subset" yields entry mass over next states. Default is uniform; caller
        // can provide a window-local entry prior pi for safer state introduction.
        let mut pi_buf: Vec<f32> = Vec::new();
        let pi = if let Some(pi) = entry_pi {
            if pi.len() == self.n_next {
                let mut sum = 0.0f32;
                pi_buf.resize(self.n_next, 0.0);
                for (i, &v) in pi.iter().enumerate() {
                    let w = if v.is_finite() && v > 0.0 { v } else { 0.0 };
                    pi_buf[i] = w;
                    sum += w;
                }
                if sum > 0.0 {
                    let inv = 1.0 / sum;
                    for v in pi_buf.iter_mut() {
                        *v *= inv;
                    }
                    Some(pi_buf.as_slice())
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        for j in 0..self.n_next {
            let dropped_entry = if let Some(pi) = pi {
                dropped_mass * pi[j]
            } else {
                dropped_mass / k_next.max(1.0)
            };
            next[j] =
                stay_scale * overlap_mass_by_next[j] + switch_from_overlap_each + dropped_entry;
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
                *v = (*v * inv).max(0.0);
            }
        } else {
            let uniform = 1.0 / self.n_next as f32;
            for v in next.iter_mut() {
                *v = uniform;
            }
        }
    }

    /// Backward-compatible mapping with uniform dropped-state entry.
    pub fn map_into(&self, prev_probs: &[f32], out: &mut Vec<f32>) {
        self.map_into_with_pi(prev_probs, None, out);
    }
}
