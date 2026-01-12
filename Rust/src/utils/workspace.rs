//! # Workspace Pattern for HMM Buffers
//!
//! Pre-allocated buffers for HMM computations to avoid repeated allocations
//! in hot loops. This pattern is essential for satisfying the Rust borrow
//! checker while maintaining performance.

/// Workspace for imputation HMM computations
#[derive(Debug)]
pub struct ImpWorkspace {
    /// Forward probabilities
    pub fwd: Vec<f32>,
    /// Backward probabilities
    pub bwd: Vec<f32>,
    // --- CSR Storage for Mismatches ---
    // --- CSR Storage for Log-Likelihood Differences ---
    /// Non-zero difference values (log_mismatch - log_match) or (-log_match for missing ref)
    pub diff_vals: Vec<f32>,
    /// Column indices for diffs (state indices)
    pub diff_cols: Vec<u16>,
    /// Row offsets for diffs (indexes into vals/cols)
    /// Length = n_clusters + 1
    pub diff_row_offsets: Vec<usize>,

    /// Base Log-Score per cluster (Sum of log(match_prob) for all markers)
    pub cluster_base_scores: Vec<f32>,

    /// Reusable row buffer for accumulation
    pub row_buffer: Vec<f32>,
}

impl ImpWorkspace {
    /// Create a new imputation workspace
    pub fn new(n_states: usize) -> Self {
        Self {
            fwd: vec![0.0; n_states],
            bwd: vec![0.0; n_states],
            diff_vals: Vec::new(),
            diff_cols: Vec::new(),
            diff_row_offsets: vec![0],
            cluster_base_scores: Vec::new(),
            row_buffer: vec![0.0; n_states],
        }
    }

    /// Create workspace with reference panel size (kept for call-site compatibility)
    pub fn with_ref_size(n_states: usize) -> Self {
        Self::new(n_states)
    }

    /// Resize buffers (used by test HMM functions)
    #[cfg(test)]
    pub fn resize(&mut self, n_states: usize) {
        self.fwd.resize(n_states, 0.0);
        self.bwd.resize(n_states, 0.0);
        self.row_buffer.resize(n_states, 0.0);
    }

    /// Ensure cluster buffers are ready for accumulation
    pub fn reset_and_ensure_capacity(&mut self, n_clusters_hint: usize, n_states: usize) {
        // Clear CSR but keep capacity
        self.diff_vals.clear();
        self.diff_cols.clear();
        self.diff_row_offsets.clear();
        self.diff_row_offsets.reserve(n_clusters_hint + 1);
        self.diff_row_offsets.push(0);

        self.cluster_base_scores.clear();
        self.cluster_base_scores.reserve(n_clusters_hint);

        // Resize scratch buffers
        if self.row_buffer.len() < n_states {
            self.row_buffer.resize(n_states, 0.0);
        }
    }
}

impl Default for ImpWorkspace {
    fn default() -> Self {
        Self::new(0)
    }
}
