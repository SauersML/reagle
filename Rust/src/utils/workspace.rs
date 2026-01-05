//! # Workspace Buffers (Critical Optimization)
//!
//! ## Role
//! Pre-allocated buffers for HMM calculations to achieve zero allocations
//! in hot loops. Replaces Java's reliance on GC for temporary arrays.
//!
//! ## Why This Matters
//! The HMM forward-backward algorithm runs millions of times.
//! Each run needs O(n_states * n_markers) temporary storage.
//! Allocating this fresh each time would be catastrophic for performance.
//!
//! ## Spec
//!
//! ### HmmWorkspace
//! ```rust,ignore
//! pub struct HmmWorkspace {
//!     /// Forward probabilities: forward[m][s] = P(observations_0..m, state_m = s)
//!     pub forward: Vec<f64>,
//!
//!     /// Backward probabilities: backward[m][s] = P(observations_m+1..M | state_m = s)
//!     pub backward: Vec<f64>,
//!
//!     /// Scale factors to prevent underflow: one per marker
//!     pub scales: Vec<f64>,
//!
//!     /// Temporary buffer for transition calculations
//!     pub temp: Vec<f64>,
//!
//!     /// Current capacity (n_states * n_markers for forward/backward)
//!     n_states: usize,
//!     n_markers: usize,
//! }
//! ```
//!
//! ### Methods
//! ```rust,ignore
//! impl HmmWorkspace {
//!     /// Create with initial capacity.
//!     pub fn new(n_states: usize, n_markers: usize) -> Self {
//!         Self {
//!             forward: vec![0.0; n_states * n_markers],
//!             backward: vec![0.0; n_states * n_markers],
//!             scales: vec![0.0; n_markers],
//!             temp: vec![0.0; n_states],
//!             n_states,
//!             n_markers,
//!         }
//!     }
//!
//!     /// Resize buffers if needed. Only allocates if current capacity is insufficient.
//!     pub fn ensure_capacity(&mut self, n_states: usize, n_markers: usize) {
//!         let needed = n_states * n_markers;
//!         if self.forward.len() < needed {
//!             self.forward.resize(needed, 0.0);
//!             self.backward.resize(needed, 0.0);
//!         }
//!         if self.scales.len() < n_markers {
//!             self.scales.resize(n_markers, 0.0);
//!         }
//!         if self.temp.len() < n_states {
//!             self.temp.resize(n_states, 0.0);
//!         }
//!         self.n_states = n_states;
//!         self.n_markers = n_markers;
//!     }
//!
//!     /// Clear buffers for reuse (fast: just sets len, doesn't deallocate).
//!     pub fn clear(&mut self) {
//!         // Note: we don't need to zero memory if we overwrite everything
//!         // But for safety in debug builds:
//!         #[cfg(debug_assertions)]
//!         {
//!             self.forward.fill(0.0);
//!             self.backward.fill(0.0);
//!         }
//!     }
//!
//!     /// Access forward probability at (marker, state).
//!     #[inline]
//!     pub fn fwd(&self, marker: usize, state: usize) -> f64 {
//!         self.forward[marker * self.n_states + state]
//!     }
//!
//!     /// Mutable access to forward probability.
//!     #[inline]
//!     pub fn fwd_mut(&mut self, marker: usize, state: usize) -> &mut f64 {
//!         &mut self.forward[marker * self.n_states + state]
//!     }
//! }
//! ```
//!
//! ### Usage Pattern
//! ```rust,ignore
//! // In pipeline: create workspace once per thread
//! let mut workspace = HmmWorkspace::new(1000, 10000);
//!
//! // In hot loop: reuse workspace
//! for target in targets {
//!     workspace.ensure_capacity(n_ref_haps, window.n_markers());
//!     let result = forward_backward(&target, &reference, &params, &mut workspace);
//!     // workspace is reused next iteration
//! }
//! ```
//!
//! ### Memory Estimate
//! For 10K reference haplotypes, 50K markers per window:
//! - forward: 10K * 50K * 8 bytes = 4 GB (too big!)
//!
//! **Optimization:** Don't store full forward matrix.
//! Use checkpointing: store every Kth column, recompute between checkpoints.
