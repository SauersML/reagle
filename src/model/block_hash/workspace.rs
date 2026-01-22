//! # Block HMM Workspace: Mutable Per-Sample State
//!
//! This module defines the mutable workspace for HMM probability calculations.
//! Each sample gets its own workspace (thread-local or pooled).

/// Mutable workspace for block-hash HMM calculations
///
/// This is allocated per-sample and contains the dynamic HMM state.
/// Separated from CompressedBlock to enable parallel processing.
pub struct BlockHmmWorkspace {
    /// Forward probabilities (sized to max_states, e.g., 4096)
    pub fwd: Vec<f32>,

    /// Backward probabilities (sized to max_states)
    pub bwd: Vec<f32>,

    /// Emission probabilities (temporary buffer, sized to max_states)
    pub emissions: Vec<f32>,

    /// Reservoir probability (forward)
    pub reservoir_prob_fwd: f32,

    /// Reservoir probability (backward)
    pub reservoir_prob_bwd: f32,

    /// Checkpoints: Forward state at the START of each block
    /// Used for forward-backward combination during dosage emission
    /// checkpoints[block_idx] = (fwd_probs, reservoir_prob)
    pub checkpoints: Vec<(Vec<f32>, f32)>,
}

impl BlockHmmWorkspace {
    /// Create a new workspace for a given maximum number of states
    pub fn new(max_states: usize, n_blocks: usize) -> Self {
        Self {
            fwd: vec![0.0; max_states],
            bwd: vec![0.0; max_states],
            emissions: vec![0.0; max_states],
            reservoir_prob_fwd: 0.0,
            reservoir_prob_bwd: 0.0,
            checkpoints: vec![(vec![0.0; max_states], 0.0); n_blocks],
        }
    }

    /// Reset workspace for a new sample
    pub fn reset(&mut self, n_patterns: usize) {
        // Initialize uniform prior
        let uniform = 1.0 / n_patterns as f32;
        self.fwd[..n_patterns].fill(uniform);
        self.fwd[n_patterns..].fill(0.0);
        self.reservoir_prob_fwd = 0.0;

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
