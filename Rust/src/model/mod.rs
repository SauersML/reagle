//! # Model Module
//!
//! Statistical model implementations for phasing and imputation.
//!
//! ## Core Algorithms
//! - `PBWT`: Positional Burrows-Wheeler Transform for efficient haplotype matching
//! - `HMM`: Li-Stephens Hidden Markov Model for phasing and imputation
//! - `States`: Composite/mosaic state management for HMM
//! - `Parameters`: Model hyperparameters (Ne, error rates)

pub mod hmm;
pub mod ibs2;
pub mod imp_states;
pub mod parameters;
pub mod pbwt;
pub mod phase_ibs;
pub mod recursive_ibs;
pub mod states;
