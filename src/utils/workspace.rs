//! # Workspace Pattern for HMM Buffers
//!
//! Pre-allocated buffers for HMM computations to avoid repeated allocations
//! in hot loops. This pattern is essential for satisfying the Rust borrow
//! checker while maintaining performance.

use aligned_vec::{AVec, ConstAlign};

/// Workspace for phasing HMM computations
#[derive(Debug)]
pub struct ThreadWorkspace {
    /// Forward probabilities: fwd[m * n_states + k] = P(state k at marker m)
    pub fwd: AVec<f32, ConstAlign<32>>,
    /// Backward probabilities: bwd[m * n_states + k] = P(state k at marker m)
    pub bwd: AVec<f32, ConstAlign<32>>,
    /// Pre-computed alleles: alleles[m * n_states + k] = allele for state k at marker m
    pub lookup: AVec<u8, ConstAlign<32>>,
    /// Forward prior buffer (same length as fwd)
    pub fwd_prior: AVec<f32, ConstAlign<32>>,
    /// Per-marker ref alleles buffer (n_states)
    pub ref_alleles: Vec<u8>,
    /// Reusable weights buffer for backward sampling (n_states)
    pub weights: Vec<f32>,
    /// Reusable allele probability scratch (variable length; typically n_alleles)
    pub allele_probs: Vec<f32>,
    /// MCMC path buffers
    pub path1: Vec<u32>,
    pub path2: Vec<u32>,
    /// Per-marker haplotype alleles and flags
    pub hap1_allele: Vec<u8>,
    pub hap2_allele: Vec<u8>,
    pub hap1_partner_allele: Vec<u8>,
    pub hap2_partner_allele: Vec<u8>,
    pub hap1_use_combined: Vec<bool>,
    pub hap2_use_combined: Vec<bool>,
    /// Forward block buffer for checkpoint recompute
    pub fwd_block: Vec<f32>,
    /// FFBS forward buffers for dynamic MCMC (haploid constrained)
    pub ffbs_fwd_curr: Vec<f32>,
    pub ffbs_fwd_prev: Vec<f32>,
    pub ffbs_fwd_at_marker: Vec<f32>,
    pub ffbs_weights: Vec<f32>,
    /// Checkpoint storage buffers (reused between samples)
    pub combined_checkpoint_data: Vec<f32>,
    pub hap1_checkpoint_data: Vec<f32>,
    pub hap2_checkpoint_data: Vec<f32>,
    /// Cached marker count
    n_markers: usize,
    /// Number of states (cached for convenience)
    n_states: usize,
}

impl ThreadWorkspace {
    /// Create a new workspace with bounded memory usage
    ///
    /// Uses checkpoint-based approach: only stores active HMM state blocks,
    /// not the entire window. Memory usage is O(checkpoint_interval * n_states).
    pub fn new(checkpoint_interval: usize, n_states: usize) -> Self {
        const DEFAULT_CHECKPOINT_INTERVAL: usize = 64; // L2 cache friendly
        let interval = checkpoint_interval.max(1).min(DEFAULT_CHECKPOINT_INTERVAL);
        let size = interval * n_states;

        Self {
            fwd: AVec::from_iter(32, std::iter::repeat(0.0).take(size)),
            bwd: AVec::from_iter(32, std::iter::repeat(0.0).take(size)),
            lookup: AVec::from_iter(32, std::iter::repeat(0).take(size)),
            fwd_prior: AVec::from_iter(32, std::iter::repeat(0.0).take(size)),
            ref_alleles: Vec::new(),
            weights: Vec::new(),
            allele_probs: Vec::new(),
            path1: Vec::new(),
            path2: Vec::new(),
            hap1_allele: Vec::new(),
            hap2_allele: Vec::new(),
            hap1_partner_allele: Vec::new(),
            hap2_partner_allele: Vec::new(),
            hap1_use_combined: Vec::new(),
            hap2_use_combined: Vec::new(),
            fwd_block: Vec::new(),
            ffbs_fwd_curr: Vec::new(),
            ffbs_fwd_prev: Vec::new(),
            ffbs_fwd_at_marker: Vec::new(),
            ffbs_weights: Vec::new(),
            combined_checkpoint_data: Vec::new(),
            hap1_checkpoint_data: Vec::new(),
            hap2_checkpoint_data: Vec::new(),
            n_markers: 0,
            n_states,
        }
    }

    /// Resize workspace for a new number of states (keeps memory bounded)
    ///
    /// Only resizes if needed - doesn't allocate per window size.
    /// The workspace maintains constant memory regardless of window size.
    pub fn resize_for_states(&mut self, n_states: usize) {
        if n_states > self.n_states {
            // Only resize if we need more states, not for window size
            let current_interval = if self.n_states > 0 {
                self.fwd.len() / self.n_states
            } else {
                64
            };
            let new_size = current_interval * n_states;

            self.fwd = AVec::from_iter(32, std::iter::repeat(0.0).take(new_size));
            self.bwd = AVec::from_iter(32, std::iter::repeat(0.0).take(new_size));
            self.lookup = AVec::from_iter(32, std::iter::repeat(0).take(new_size));
            self.fwd_prior = AVec::from_iter(32, std::iter::repeat(0.0).take(new_size));
            self.n_states = n_states;
        }
    }

    /// Ensure buffers are sized for the current window and state count.
    pub fn ensure_for_window(
        &mut self,
        n_markers: usize,
        n_states: usize,
        max_block_len: usize,
        n_blocks: usize,
    ) {
        self.resize_for_states(n_states);
        self.n_markers = n_markers;

        if self.ref_alleles.len() < n_states {
            self.ref_alleles.resize(n_states, 0);
        }

        if self.weights.len() < n_states {
            self.weights.resize(n_states, 0.0);
        }

        if self.path1.len() < n_markers {
            self.path1.resize(n_markers, 0);
        }
        if self.path2.len() < n_markers {
            self.path2.resize(n_markers, 0);
        }
        if self.hap1_allele.len() < n_markers {
            self.hap1_allele.resize(n_markers, 255);
        }
        if self.hap2_allele.len() < n_markers {
            self.hap2_allele.resize(n_markers, 255);
        }
        if self.hap1_partner_allele.len() < n_markers {
            self.hap1_partner_allele.resize(n_markers, 255);
        }
        if self.hap2_partner_allele.len() < n_markers {
            self.hap2_partner_allele.resize(n_markers, 255);
        }
        if self.hap1_use_combined.len() < n_markers {
            self.hap1_use_combined.resize(n_markers, true);
        }
        if self.hap2_use_combined.len() < n_markers {
            self.hap2_use_combined.resize(n_markers, true);
        }

        let block_len = n_states * max_block_len;
        if self.fwd_block.len() < block_len {
            self.fwd_block.resize(block_len, 0.0);
        }

        let checkpoints_len = n_blocks * n_states;
        if self.combined_checkpoint_data.len() < checkpoints_len {
            self.combined_checkpoint_data.resize(checkpoints_len, 0.0);
        }
        if self.hap1_checkpoint_data.len() < checkpoints_len {
            self.hap1_checkpoint_data.resize(checkpoints_len, 0.0);
        }
        if self.hap2_checkpoint_data.len() < checkpoints_len {
            self.hap2_checkpoint_data.resize(checkpoints_len, 0.0);
        }
    }

    /// Ensure FFBS buffers are sized for haploid constrained sampling.
    pub fn ensure_ffbs(&mut self, n_markers: usize, n_states: usize) {
        if self.ffbs_fwd_curr.len() < n_states {
            self.ffbs_fwd_curr.resize(n_states, 0.0);
        }
        if self.ffbs_fwd_prev.len() < n_states {
            self.ffbs_fwd_prev.resize(n_states, 0.0);
        }
        if self.ffbs_weights.len() < n_states {
            self.ffbs_weights.resize(n_states, 0.0);
        }
        let needed = n_markers.saturating_mul(n_states);
        if self.ffbs_fwd_at_marker.len() < needed {
            self.ffbs_fwd_at_marker.resize(needed, 0.0);
        }
    }

    /// Clear workspace contents without deallocating
    pub fn clear(&mut self) {
        // No need to zero out, as we'll overwrite during fill
    }
}
