//! # Block HMM Workspace: Mutable Per-Sample State
//!
//! This module defines the mutable workspace for HMM probability calculations.
//! Each sample gets its own workspace (thread-local or pooled).

use aligned_vec::{AVec, ConstAlign};

/// Mutable workspace for block-hash HMM calculations
///
/// This is allocated per-sample and contains the dynamic HMM state.
/// Separated from CompressedBlock to enable parallel processing.
pub struct BlockHmmWorkspace {
    /// Forward probabilities (sized to max_states, e.g., 4096)
    pub fwd: AVec<f32, ConstAlign<32>>,

    /// Backward probabilities (sized to max_states)
    pub bwd: AVec<f32, ConstAlign<32>>,

    /// Emission probabilities (temporary buffer, sized to max_states)
    pub emissions: AVec<f32, ConstAlign<32>>,

    /// Reservoir probability (forward)
    pub reservoir_prob_fwd: f32,

    /// Reservoir probability (backward)
    pub reservoir_prob_bwd: f32,

    /// Checkpoints: Forward state at the START of each block
    /// Used for forward-backward combination during dosage emission
    /// checkpoints[block_idx] = (fwd_probs, reservoir_prob)
    pub checkpoints: Vec<(Vec<f32>, f32)>,

    /// History buffer for intra-block forward pass
    /// Flattened [marker_idx * (max_states + 1) + pattern_idx]
    /// The +1 is for the reservoir state.
    pub fwd_history: Vec<f32>,

    /// Maximum states capacity (excluding reservoir)
    pub max_states: usize,
}

impl BlockHmmWorkspace {
    /// Create a new workspace for a given maximum number of states
    pub fn new(required_patterns: usize, n_blocks: usize, window_size: usize) -> Self {
        // Ensure we at least have space for 1 pattern to avoid empty-vec panics
        // in edge cases, though logical flow should prevent usage if empty.
        let capacity = required_patterns.max(1);

        Self {
            fwd: AVec::from_iter(32, std::iter::repeat(0.0).take(capacity)),
            bwd: AVec::from_iter(32, std::iter::repeat(0.0).take(capacity)),
            emissions: AVec::from_iter(32, std::iter::repeat(0.0).take(capacity)),
            reservoir_prob_fwd: 0.0,
            reservoir_prob_bwd: 0.0,
            checkpoints: vec![(vec![0.0; capacity], 0.0); n_blocks],
            fwd_history: vec![0.0; (capacity + 1) * window_size],
            max_states: capacity,
        }
    }



    /// Reset workspace for a new sample using the first block's pattern counts
    ///
    /// This initializes the forward probabilities to the "Uniform Haplotype Prior".
    /// Probability of pattern i = count(i) / N_ref_haps.
    pub fn reset_from_block(&mut self, block: &super::compressed_block::CompressedBlock) {
        let n_patterns = block.n_patterns();
        let n_ref_haps = block.n_ref_haps() as f32;
        
        // Safety check to avoid division by zero (should not happen in valid blocks)
        if n_ref_haps <= 0.0 {
            self.fwd[..n_patterns].fill(0.0);
            self.reservoir_prob_fwd = 0.0;
        } else {
            let scale = 1.0 / n_ref_haps;
            for i in 0..n_patterns {
                self.fwd[i] = block.pattern_counts[i] * scale;
            }
            // Clear remaining slots
            if n_patterns < self.fwd.len() {
               self.fwd[n_patterns..].fill(0.0);
            }

            if block.reservoir_count > 0 {
                self.reservoir_prob_fwd = (block.reservoir_count as f32) * scale;
            } else {
                self.reservoir_prob_fwd = 0.0;
            }
        }

        // Initialize Backward to 0.0 (clearing garbage). 
        // Backward pass usually starts from end of chromosome or bridge from next block.
        self.bwd.fill(0.0);
        self.reservoir_prob_bwd = 0.0;
    }

    /// Save checkpoint at current forward state
    pub fn save_checkpoint(&mut self, block_idx: usize, n_patterns: usize) {
        let (checkpoint_fwd, checkpoint_res) = &mut self.checkpoints[block_idx];
        checkpoint_fwd[..n_patterns].copy_from_slice(&self.fwd[..n_patterns]);
        *checkpoint_res = self.reservoir_prob_fwd;
    }

    /// Restore checkpoint to forward state
    pub fn restore_checkpoint(&mut self, block_idx: usize, n_patterns: usize) {
        let (checkpoint_fwd, checkpoint_res) = &self.checkpoints[block_idx];
        self.fwd[..n_patterns].copy_from_slice(&checkpoint_fwd[..n_patterns]);
        self.reservoir_prob_fwd = *checkpoint_res;
    }

    /// Normalize forward probabilities
    pub fn normalize_forward(&mut self, n_patterns: usize) {
        let total: f32 = self.fwd[..n_patterns].iter().sum::<f32>() + self.reservoir_prob_fwd;

        if total > 1e-30 {
            let scale = 1.0 / total;
            for prob in &mut self.fwd[..n_patterns] {
                *prob *= scale;
            }
            self.reservoir_prob_fwd *= scale;
        } else {
            // Underflow - reinitialize to uniform
            let uniform = 1.0 / n_patterns as f32;
            self.fwd[..n_patterns].fill(uniform);
            self.reservoir_prob_fwd = 0.0;
        }
    }

    /// Normalize backward probabilities
    pub fn normalize_bwd(&mut self, n_patterns: usize) {
        let total: f32 = self.bwd[..n_patterns].iter().sum::<f32>() + self.reservoir_prob_bwd;

        if total > 1e-30 {
            let scale = 1.0 / total;
            for prob in &mut self.bwd[..n_patterns] {
                *prob *= scale;
            }
            self.reservoir_prob_bwd *= scale;
        } else {
            // Underflow - reinitialize to uniform
            let uniform = 1.0 / n_patterns as f32;
            self.bwd[..n_patterns].fill(uniform);
            self.reservoir_prob_bwd = 0.0;
        }
    }
}
