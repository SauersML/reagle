//! # Model Module
//!
//! Statistical model implementations for phasing and imputation.
//!
//! ## Core Algorithms
//! - `PBWT`: Positional Burrows-Wheeler Transform for efficient haplotype matching
//! - `HMM`: Li-Stephens Hidden Markov Model for phasing and imputation
//! - `Parameters`: Model hyperparameters (Ne, error rates)

pub mod hmm;
pub mod ibs2;
pub mod imp_states;
pub mod parameters;
pub mod pbwt;

pub use hmm::{LiStephensHmm, HmmResult};
pub use ibs2::{Ibs2, Ibs2Segment};
pub use imp_states::{ImpStates, CodedSteps, CodedStepsConfig};
pub use parameters::ModelParams;
pub use pbwt::{PbwtUpdater, PbwtDivUpdater};