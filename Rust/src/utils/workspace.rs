//! # Workspace Pattern for HMM Buffers
//!
//! Pre-allocated buffers for HMM computations to avoid repeated allocations
//! in hot loops. This pattern is essential for satisfying the Rust borrow
//! checker while maintaining performance.
//!
//! ## Design Philosophy
//! Instead of storing mutable buffers inside model structs (which causes
//! borrow checker issues), we create a separate Workspace that owns all
//! temporary buffers and pass `&mut Workspace` to computation functions.

use crate::data::HapIdx;

/// Workspace for phasing HMM computations
#[derive(Debug)]
pub struct Workspace {
    /// Forward probabilities (n_states)
    pub fwd: Vec<f32>,

    /// Backward probabilities (n_states)
    pub bwd: Vec<f32>,

    /// State probabilities at each marker (n_markers x n_states)
    pub state_probs: Vec<Vec<f32>>,

    /// Temporary buffer for state updates
    pub tmp: Vec<f32>,

    /// PBWT prefix array
    pub prefix: Vec<u32>,

    /// PBWT divergence array
    pub divergence: Vec<u32>,

    /// Selected reference haplotype indices
    pub ref_haps: Vec<HapIdx>,

    /// Allele buffer for current marker
    pub alleles: Vec<u8>,

    /// Switch probability buffer
    pub switch_probs: Vec<f32>,

    /// Emission probability buffer
    pub emit_probs: Vec<f32>,

    /// Random number generator state
    pub rng_state: u64,
}

impl Workspace {
    /// Create a new workspace with given capacities
    pub fn new(n_states: usize, n_markers: usize, n_haps: usize) -> Self {
        Self {
            fwd: vec![0.0; n_states],
            bwd: vec![0.0; n_states],
            state_probs: vec![vec![0.0; n_states]; n_markers],
            tmp: vec![0.0; n_states],
            prefix: (0..n_haps as u32).collect(),
            divergence: vec![0; n_haps],
            ref_haps: Vec::with_capacity(n_states),
            alleles: vec![0; n_haps],
            switch_probs: vec![0.0; n_markers],
            emit_probs: vec![0.0; n_states],
            rng_state: 0,
        }
    }

    /// Create a minimal workspace for testing
    pub fn minimal() -> Self {
        Self::new(0, 0, 0)
    }

    /// Resize buffers for new dimensions
    pub fn resize(&mut self, n_states: usize, n_markers: usize, n_haps: usize) {
        self.fwd.resize(n_states, 0.0);
        self.bwd.resize(n_states, 0.0);
        self.state_probs.resize(n_markers, vec![0.0; n_states]);
        for probs in &mut self.state_probs {
            probs.resize(n_states, 0.0);
        }
        self.tmp.resize(n_states, 0.0);

        if n_haps > self.prefix.len() {
            let old_len = self.prefix.len();
            self.prefix.resize(n_haps, 0);
            for i in old_len..n_haps {
                self.prefix[i] = i as u32;
            }
            self.divergence.resize(n_haps, 0);
            self.alleles.resize(n_haps, 0);
        }

        self.emit_probs.resize(n_states, 0.0);
    }

    /// Clear all buffers (set to zero)
    pub fn clear(&mut self) {
        self.fwd.fill(0.0);
        self.bwd.fill(0.0);
        for probs in &mut self.state_probs {
            probs.fill(0.0);
        }
        self.tmp.fill(0.0);
        self.ref_haps.clear();
        self.switch_probs.fill(0.0);
        self.emit_probs.fill(0.0);
    }

    /// Reset PBWT arrays to identity permutation
    pub fn reset_pbwt(&mut self, n_haps: usize) {
        self.prefix.truncate(n_haps);
        self.prefix.resize(n_haps, 0);
        for (i, p) in self.prefix.iter_mut().enumerate() {
            *p = i as u32;
        }
        self.divergence.truncate(n_haps);
        self.divergence.resize(n_haps, 0);
        self.divergence.fill(0);
    }

    /// Initialize forward probabilities uniformly
    pub fn init_fwd_uniform(&mut self, n_states: usize) {
        let prob = 1.0 / n_states as f32;
        self.fwd.truncate(n_states);
        self.fwd.resize(n_states, prob);
        self.fwd.fill(prob);
    }

    /// Initialize backward probabilities to 1.0
    pub fn init_bwd_ones(&mut self, n_states: usize) {
        self.bwd.truncate(n_states);
        self.bwd.resize(n_states, 1.0);
        self.bwd.fill(1.0);
    }

    /// Set random seed
    pub fn set_seed(&mut self, seed: u64) {
        self.rng_state = seed;
    }

    /// Generate next random u64 (xorshift64)
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }

    /// Generate random f32 in [0, 1)
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Sample an index from a probability distribution
    pub fn sample_index(&mut self, probs: &[f32]) -> usize {
        let sum: f32 = probs.iter().sum();
        if sum <= 0.0 {
            return 0;
        }

        let threshold = self.next_f32() * sum;
        let mut cumsum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if cumsum >= threshold {
                return i;
            }
        }
        probs.len() - 1
    }
}

impl Default for Workspace {
    fn default() -> Self {
        Self::minimal()
    }
}

/// Workspace for imputation HMM computations
#[derive(Debug)]
pub struct ImpWorkspace {
    /// Forward probabilities
    pub fwd: Vec<f32>,

    /// Backward probabilities
    pub bwd: Vec<f32>,

    /// Allele dosages at each marker
    pub dosages: Vec<f32>,

    /// Allele probabilities at each marker (for GP output)
    pub allele_probs: Vec<[f32; 3]>,

    /// Temporary buffer
    pub tmp: Vec<f32>,

    /// State to allele mapping
    pub state_alleles: Vec<u8>,
}

impl ImpWorkspace {
    /// Create a new imputation workspace
    pub fn new(n_states: usize, n_markers: usize) -> Self {
        Self {
            fwd: vec![0.0; n_states],
            bwd: vec![0.0; n_states],
            dosages: vec![0.0; n_markers],
            allele_probs: vec![[0.0; 3]; n_markers],
            tmp: vec![0.0; n_states],
            state_alleles: vec![0; n_states],
        }
    }

    /// Resize buffers
    pub fn resize(&mut self, n_states: usize, n_markers: usize) {
        self.fwd.resize(n_states, 0.0);
        self.bwd.resize(n_states, 0.0);
        self.dosages.resize(n_markers, 0.0);
        self.allele_probs.resize(n_markers, [0.0; 3]);
        self.tmp.resize(n_states, 0.0);
        self.state_alleles.resize(n_states, 0);
    }

    /// Clear buffers
    pub fn clear(&mut self) {
        self.fwd.fill(0.0);
        self.bwd.fill(0.0);
        self.dosages.fill(0.0);
        for p in &mut self.allele_probs {
            *p = [0.0; 3];
        }
        self.tmp.fill(0.0);
    }
}

impl Default for ImpWorkspace {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workspace_creation() {
        let ws = Workspace::new(100, 50, 1000);
        assert_eq!(ws.fwd.len(), 100);
        assert_eq!(ws.state_probs.len(), 50);
        assert_eq!(ws.prefix.len(), 1000);
    }

    #[test]
    fn test_workspace_resize() {
        let mut ws = Workspace::minimal();
        ws.resize(200, 100, 2000);
        assert_eq!(ws.fwd.len(), 200);
        assert_eq!(ws.state_probs.len(), 100);
        assert_eq!(ws.prefix.len(), 2000);
    }

    #[test]
    fn test_random_sampling() {
        let mut ws = Workspace::minimal();
        ws.set_seed(12345);

        let probs = vec![0.1, 0.2, 0.3, 0.4];
        let mut counts = vec![0usize; 4];

        for _ in 0..10000 {
            let idx = ws.sample_index(&probs);
            counts[idx] += 1;
        }

        // Check rough distribution
        assert!(counts[3] > counts[0]); // 0.4 should be sampled more than 0.1
    }
}