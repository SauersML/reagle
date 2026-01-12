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
    /// Cluster mismatches buffer: mismatches[cluster][state]
    pub cluster_mismatches: Vec<Vec<f32>>,
    /// Cluster non-missing buffer: non_missing[cluster][state]
    pub cluster_non_missing: Vec<Vec<f32>>,
}

impl ImpWorkspace {
    /// Create a new imputation workspace
    pub fn new(n_states: usize) -> Self {
        Self {
            fwd: vec![0.0; n_states],
            bwd: vec![0.0; n_states],
            cluster_mismatches: Vec::new(),
            cluster_non_missing: Vec::new(),
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
    }

    /// Ensure cluster buffers are sized for given dimensions, clearing all values
    pub fn ensure_cluster_buffers(&mut self, n_clusters: usize, n_states: usize) {
        // Resize outer vec if needed
        if self.cluster_mismatches.len() < n_clusters {
            self.cluster_mismatches.resize_with(n_clusters, Vec::new);
            self.cluster_non_missing.resize_with(n_clusters, Vec::new);
        }
        // Resize and zero each inner vec
        for c in 0..n_clusters {
            let mism = &mut self.cluster_mismatches[c];
            if mism.len() < n_states {
                mism.resize(n_states, 0.0);
            } else {
                mism[..n_states].fill(0.0);
            }
            let nm = &mut self.cluster_non_missing[c];
            if nm.len() < n_states {
                nm.resize(n_states, 0.0);
            } else {
                nm[..n_states].fill(0.0);
            }
        }
    }
}

impl Default for ImpWorkspace {
    fn default() -> Self {
        Self::new(0)
    }
}
