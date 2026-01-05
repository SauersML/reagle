//! # Threading Configuration
//!
//! ## Role
//! Configure rayon thread pools and manage parallel execution.
//! Replaces `blbutil/MultiThreadUtils.java`.
//!
//! ## Spec
//!
//! ### Thread Pool Builder
//! ```rust,ignore
//! /// Create a configured thread pool.
//! pub fn build_thread_pool(n_threads: usize) -> Result<rayon::ThreadPool> {
//!     rayon::ThreadPoolBuilder::new()
//!         .num_threads(n_threads)
//!         .thread_name(|i| format!("reagle-worker-{}", i))
//!         .build()
//!         .map_err(|e| ReagleError::Config {
//!             msg: format!("Failed to create thread pool: {}", e),
//!         })
//! }
//! ```
//!
//! ### Thread-Local Storage
//! For per-thread workspaces (avoids mutex contention):
//! ```rust,ignore
//! use std::cell::RefCell;
//! use thread_local::ThreadLocal;
//!
//! pub struct ThreadLocalWorkspace {
//!     workspaces: ThreadLocal<RefCell<HmmWorkspace>>,
//! }
//!
//! impl ThreadLocalWorkspace {
//!     pub fn new() -> Self {
//!         Self {
//!             workspaces: ThreadLocal::new(),
//!         }
//!     }
//!
//!     /// Get or create workspace for current thread.
//!     pub fn get_or_create(&self, n_states: usize, n_markers: usize) -> RefMut<HmmWorkspace> {
//!         self.workspaces
//!             .get_or(|| RefCell::new(HmmWorkspace::new(n_states, n_markers)))
//!             .borrow_mut()
//!     }
//! }
//! ```
//!
//! ### Parallel Iteration Patterns
//! ```rust,ignore
//! use rayon::prelude::*;
//!
//! // Process samples in parallel
//! samples
//!     .par_iter()
//!     .map(|sample| {
//!         let mut workspace = thread_local_workspace.get_or_create(n_states, n_markers);
//!         phase_sample(sample, &mut workspace)
//!     })
//!     .collect()
//! ```
//!
//! ### Progress Reporting with Parallelism
//! ```rust,ignore
//! use std::sync::atomic::{AtomicUsize, Ordering};
//!
//! let progress = AtomicUsize::new(0);
//! let total = samples.len();
//!
//! samples.par_iter().for_each(|sample| {
//!     process(sample);
//!     let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
//!     if done % 100 == 0 {
//!         log::info!("Processed {}/{} samples", done, total);
//!     }
//! });
//! ```
