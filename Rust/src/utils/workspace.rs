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
    /// Non-zero mismatch values (weights)
    pub mismatch_vals: Vec<f32>,
    /// Column indices for mismatches (state indices)
    pub mismatch_cols: Vec<u16>,
    /// Row offsets for mismatches (indexes into vals/cols)
    /// Length = n_clusters + 1
    pub mismatch_row_offsets: Vec<usize>,

    // --- CSR Storage for Missing Reference (Non-Missing deviations) ---
    /// Values where reference is missing (penalty subtraction) or specific confidence
    /// Plan: `cluster_non_missing` -> `cluster_total_conf` - `missing_ref_vals`
    pub missing_ref_vals: Vec<f32>,
    pub missing_ref_cols: Vec<u16>,
    pub missing_ref_row_offsets: Vec<usize>,
    
    /// Total confidence per cluster (base value for non-missing)
    pub cluster_total_conf: Vec<f32>,

    /// Reusable row buffer for accumulation
    pub row_buffer: Vec<f32>,
    pub row_buffer_missing: Vec<f32>,
}

impl ImpWorkspace {
    /// Create a new imputation workspace
    pub fn new(n_states: usize) -> Self {
        Self {
            fwd: vec![0.0; n_states],
            bwd: vec![0.0; n_states],
            mismatch_vals: Vec::new(),
            mismatch_cols: Vec::new(),
            mismatch_row_offsets: vec![0],
            missing_ref_vals: Vec::new(),
            missing_ref_cols: Vec::new(),
            missing_ref_row_offsets: vec![0],
            cluster_total_conf: Vec::new(),
            row_buffer: vec![0.0; n_states],
            row_buffer_missing: vec![0.0; n_states],
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
        self.row_buffer_missing.resize(n_states, 0.0);
    }

    /// Ensure cluster buffers are ready for accumulation
    pub fn reset_and_ensure_capacity(&mut self, n_clusters_hint: usize, n_states: usize) {
        // Clear CSR but keep capacity
        self.mismatch_vals.clear();
        self.mismatch_cols.clear();
        self.mismatch_row_offsets.clear();
        self.mismatch_row_offsets.reserve(n_clusters_hint + 1);
        self.mismatch_row_offsets.push(0);

        self.missing_ref_vals.clear();
        self.missing_ref_cols.clear();
        self.missing_ref_row_offsets.clear();
        self.missing_ref_row_offsets.reserve(n_clusters_hint + 1);
        self.missing_ref_row_offsets.push(0);
        
        self.cluster_total_conf.clear();
        self.cluster_total_conf.reserve(n_clusters_hint);

        // Resize scratch buffers
        if self.row_buffer.len() < n_states {
            self.row_buffer.resize(n_states, 0.0);
            self.row_buffer_missing.resize(n_states, 0.0);
        }
    }
}

impl Default for ImpWorkspace {
    fn default() -> Self {
        Self::new(0)
    }
}
