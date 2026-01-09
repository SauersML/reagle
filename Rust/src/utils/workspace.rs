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
    /// Pattern counts for counting sort (forward)
    pub sort_counts: Vec<usize>,
    /// Cumulative offsets for counting sort (forward)
    pub sort_offsets: Vec<usize>,
    /// Scratch buffer for new prefix order (forward)
    pub sort_prefix_scratch: Vec<u32>,
    /// Scratch buffer for new divergence values (forward)
    pub sort_div_scratch: Vec<i32>,
    /// Pattern counts for counting sort (backward)
    pub sort_counts_bwd: Vec<usize>,
    /// Cumulative offsets for counting sort (backward)
    pub sort_offsets_bwd: Vec<usize>,
    /// Scratch buffer for new prefix order (backward)
    pub sort_prefix_scratch_bwd: Vec<u32>,
    /// Scratch buffer for new divergence values (backward)
    pub sort_div_scratch_bwd: Vec<i32>,
}

impl ImpWorkspace {
    /// Create a new imputation workspace
    pub fn new(n_states: usize, n_markers: usize) -> Self {
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
            sort_counts: Vec::new(),
            sort_offsets: Vec::new(),
            sort_prefix_scratch: vec![0; n_states],
            sort_div_scratch: vec![0; n_states + 1],
            sort_counts_bwd: Vec::new(),
            sort_offsets_bwd: Vec::new(),
            sort_prefix_scratch_bwd: vec![0; n_states],
            sort_div_scratch_bwd: vec![0; n_states + 1],
        }
    }

    /// Create workspace with reference panel size
    pub fn with_ref_size(n_states: usize, n_markers: usize, n_ref_haps: usize) -> Self {
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
            sort_counts: Vec::new(),
            sort_offsets: Vec::new(),
            sort_prefix_scratch: vec![0; n_ref_haps],
            sort_div_scratch: vec![0; n_ref_haps + 1],
            sort_counts_bwd: Vec::new(),
            sort_offsets_bwd: Vec::new(),
            sort_prefix_scratch_bwd: vec![0; n_ref_haps],
            sort_div_scratch_bwd: vec![0; n_ref_haps + 1],
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

    /// Resize including PBWT buffers
    pub fn resize_with_ref(&mut self, n_states: usize, n_markers: usize, n_ref_haps: usize) {
        self.fwd.resize(n_states, 0.0);
        self.bwd.resize(n_states, 0.0);
        self.dosages.resize(n_markers, 0.0);
        self.allele_probs.resize(n_markers, [0.0; 3]);
        self.tmp.resize(n_states, 0.0);
        self.state_alleles.resize(n_states, 0);
        if self.pbwt_prefix.len() != n_ref_haps {
            self.pbwt_prefix.resize(n_ref_haps, 0);
            for (i, p) in self.pbwt_prefix.iter_mut().enumerate() {
                *p = i as u32;
            }
        }
        self.pbwt_divergence.resize(n_ref_haps + 1, 0);
        self.pbwt_divergence.fill(0);
        if self.pbwt_prefix_bwd.len() != n_ref_haps {
            self.pbwt_prefix_bwd.resize(n_ref_haps, 0);
            for (i, p) in self.pbwt_prefix_bwd.iter_mut().enumerate() {
                *p = i as u32;
            }
        }
        self.pbwt_divergence_bwd.resize(n_ref_haps + 1, 0);
        self.pbwt_divergence_bwd.fill(0);
        self.sort_prefix_scratch.resize(n_ref_haps, 0);
        self.sort_div_scratch.resize(n_ref_haps + 1, 0);
        self.sort_prefix_scratch_bwd.resize(n_ref_haps, 0);
        self.sort_div_scratch_bwd.resize(n_ref_haps + 1, 0);
    }
}

impl Default for ImpWorkspace {
    fn default() -> Self {
        Self::new(0, 0)
    }
}
