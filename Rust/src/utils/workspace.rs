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

    /// Forward pass combined (n_markers x n_states) - for phasing HMM
    pub fwd_combined: Vec<Vec<f32>>,

    /// Forward pass hap1 (n_markers x n_states) - for phasing HMM
    pub fwd1: Vec<Vec<f32>>,

    /// Forward pass hap2 (n_markers x n_states) - for phasing HMM
    pub fwd2: Vec<Vec<f32>>,

    /// Temporary buffer for state updates
    pub tmp: Vec<f32>,

    /// PBWT prefix array
    pub prefix: Vec<u32>,

    /// PBWT divergence array
    pub divergence: Vec<u32>,

    /// Allele buffer for current marker
    pub alleles: Vec<u8>,

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
            fwd_combined: vec![vec![0.0; n_states]; n_markers],
            fwd1: vec![vec![0.0; n_states]; n_markers],
            fwd2: vec![vec![0.0; n_states]; n_markers],
            tmp: vec![0.0; n_states],
            prefix: (0..n_haps as u32).collect(),
            divergence: vec![0; n_haps],
            alleles: vec![0; n_haps],
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
        self.fwd_combined.resize(n_markers, vec![0.0; n_states]);
        for probs in &mut self.fwd_combined {
            probs.resize(n_states, 0.0);
        }
        self.fwd1.resize(n_markers, vec![0.0; n_states]);
        for probs in &mut self.fwd1 {
            probs.resize(n_states, 0.0);
        }
        self.fwd2.resize(n_markers, vec![0.0; n_states]);
        for probs in &mut self.fwd2 {
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

    /// PBWT prefix array (for CodedPbwtView) - forward direction
    pub pbwt_prefix: Vec<u32>,

    /// PBWT divergence array (for CodedPbwtView) - forward direction
    pub pbwt_divergence: Vec<i32>,

    /// PBWT prefix array for backward direction
    pub pbwt_prefix_bwd: Vec<u32>,

    /// PBWT divergence array for backward direction
    pub pbwt_divergence_bwd: Vec<i32>,

    // === Counting sort scratch buffers for optimized PBWT updates ===
    /// Pattern counts for counting sort
    pub sort_counts: Vec<usize>,

    /// Cumulative offsets for counting sort
    pub sort_offsets: Vec<usize>,

    /// Scratch buffer for new prefix order
    pub sort_prefix_scratch: Vec<u32>,

    /// Scratch buffer for new divergence values
    pub sort_div_scratch: Vec<i32>,
}

impl ImpWorkspace {
    /// Create a new imputation workspace
    pub fn new(n_states: usize, n_markers: usize) -> Self {
        // Initialize PBWT arrays with identity permutation
        let pbwt_prefix = (0..n_states as u32).collect();
        let pbwt_divergence = vec![0; n_states + 1];
        let pbwt_prefix_bwd = (0..n_states as u32).collect();
        let pbwt_divergence_bwd = vec![0; n_states + 1];

        Self {
            fwd: vec![0.0; n_states],
            bwd: vec![0.0; n_states],
            dosages: vec![0.0; n_markers],
            allele_probs: vec![[0.0; 3]; n_markers],
            tmp: vec![0.0; n_states],
            state_alleles: vec![0; n_states],
            pbwt_prefix,
            pbwt_divergence,
            pbwt_prefix_bwd,
            pbwt_divergence_bwd,
            // Counting sort buffers - start small, will resize as needed
            sort_counts: Vec::new(),
            sort_offsets: Vec::new(),
            sort_prefix_scratch: vec![0; n_states],
            sort_div_scratch: vec![0; n_states + 1],
        }
    }

    /// Create workspace with reference panel size
    pub fn with_ref_size(n_states: usize, n_markers: usize, n_ref_haps: usize) -> Self {
        // PBWT arrays sized for reference panel
        let pbwt_prefix = (0..n_ref_haps as u32).collect();
        let pbwt_divergence = vec![0; n_ref_haps + 1];
        let pbwt_prefix_bwd = (0..n_ref_haps as u32).collect();
        let pbwt_divergence_bwd = vec![0; n_ref_haps + 1];

        Self {
            fwd: vec![0.0; n_states],
            bwd: vec![0.0; n_states],
            dosages: vec![0.0; n_markers],
            allele_probs: vec![[0.0; 3]; n_markers],
            tmp: vec![0.0; n_states],
            state_alleles: vec![0; n_states],
            pbwt_prefix,
            pbwt_divergence,
            pbwt_prefix_bwd,
            pbwt_divergence_bwd,
            // Counting sort buffers - sized for reference panel
            sort_counts: Vec::new(),
            sort_offsets: Vec::new(),
            sort_prefix_scratch: vec![0; n_ref_haps],
            sort_div_scratch: vec![0; n_ref_haps + 1],
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
        // PBWT arrays keep their size (based on reference panel)
    }

    /// Resize including PBWT buffers
    pub fn resize_with_ref(&mut self, n_states: usize, n_markers: usize, n_ref_haps: usize) {
        self.fwd.resize(n_states, 0.0);
        self.bwd.resize(n_states, 0.0);
        self.dosages.resize(n_markers, 0.0);
        self.allele_probs.resize(n_markers, [0.0; 3]);
        self.tmp.resize(n_states, 0.0);
        self.state_alleles.resize(n_states, 0);

        // Resize forward PBWT arrays if needed
        if self.pbwt_prefix.len() != n_ref_haps {
            self.pbwt_prefix.resize(n_ref_haps, 0);
            for (i, p) in self.pbwt_prefix.iter_mut().enumerate() {
                *p = i as u32;
            }
        }
        self.pbwt_divergence.resize(n_ref_haps + 1, 0);
        self.pbwt_divergence.fill(0);

        // Resize backward PBWT arrays if needed
        if self.pbwt_prefix_bwd.len() != n_ref_haps {
            self.pbwt_prefix_bwd.resize(n_ref_haps, 0);
            for (i, p) in self.pbwt_prefix_bwd.iter_mut().enumerate() {
                *p = i as u32;
            }
        }
        self.pbwt_divergence_bwd.resize(n_ref_haps + 1, 0);
        self.pbwt_divergence_bwd.fill(0);

        // Resize counting sort scratch buffers
        self.sort_prefix_scratch.resize(n_ref_haps, 0);
        self.sort_div_scratch.resize(n_ref_haps + 1, 0);
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
