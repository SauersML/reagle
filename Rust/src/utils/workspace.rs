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
}

impl ImpWorkspace {
    /// Create a new imputation workspace
    pub fn new(n_states: usize) -> Self {
        Self {
            fwd: vec![0.0; n_states],
            bwd: vec![0.0; n_states],
        }
    }

    /// Create workspace with reference panel size (kept for call-site compatibility)
    pub fn with_ref_size(n_states: usize) -> Self {
        Self::new(n_states)
    }

    /// Resize buffers
    #[cfg(test)]
    pub fn resize(&mut self, n_states: usize) {
        self.fwd.resize(n_states, 0.0);
        self.bwd.resize(n_states, 0.0);
    }
}

impl Default for ImpWorkspace {
    fn default() -> Self {
        Self::new(0)
    }
}
