//! # Pipeline Module
//!
//! High-level orchestration of phasing and imputation workflows.
//! Coordinates I/O, windowing, and algorithm execution.

pub mod imputation;
pub mod phasing;

pub use imputation::ImputationPipeline;
pub use phasing::PhasingPipeline;
