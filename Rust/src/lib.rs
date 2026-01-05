//! # Reagle Library Root
//!
//! ## Role
//! The crate root that declares all public modules and re-exports common types.
//!
//! ## Spec
//! - Declare all public modules (`pub mod data`, `pub mod model`, etc.).
//! - Re-export commonly used types for ergonomic access.
//! - This allows the code to be used as a library by other tools
//!   (e.g., a future Pangenome tool) or by the binary executable.
//!
//! ## Module Structure
//! ```text
//! reagle
//! ├── data        # In-memory representations (markers, genotypes)
//! │   └── storage # Storage backends (dense, sparse, dictionary)
//! ├── io          # File I/O (VCF reading/writing, windowing)
//! ├── model       # Algorithms (HMM, PBWT)
//! ├── pipelines   # High-level orchestration (phasing, imputation)
//! └── utils       # Helpers (threading, workspace buffers)
//! ```

pub mod config;
pub mod data;
pub mod error;
pub mod io;
pub mod model;
pub mod pipelines;
pub mod utils;
