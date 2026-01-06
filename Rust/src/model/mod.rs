//! # Model Module
//!
//! Statistical model implementations for phasing and imputation.
//!
//! ## Core Algorithms
//! - `PBWT`: Positional Burrows-Wheeler Transform for efficient haplotype matching
//! - `HMM`: Li-Stephens Hidden Markov Model for phasing and imputation
//! - `States`: Composite/mosaic state management for HMM
//! - `Parameters`: Model hyperparameters (Ne, error rates)
//!
//! ## Why PBWT Instead of Recursive IBS
//!
//! Java Beagle uses two different algorithms for selecting HMM states:
//! - Phasing: Uses PBWT (`PbwtPhaseIbs`)
//! - Imputation: Uses Recursive Partitioning (`ImpIbs`)
//!
//! This Rust implementation uses PBWT for both phasing and imputation because:
//!
//! ### 1. Accuracy
//! PBWT mathematically guarantees finding the longest common substring matches.
//! Recursive partitioning is a greedy top-down heuristic that relies on arbitrary
//! step boundaries. If a match is perfect but offset slightly from a step boundary,
//! or if the tree branches too early due to a single mismatch, valuable long-range
//! matches can be lost.
//!
//! ### 2. Performance
//! PBWT uses linear arrays and is highly vectorizable (SIMD-friendly). The sorting
//! and neighbor-finding operations access memory sequentially, which is optimal for
//! CPU cache utilization. Recursive IBS involves pointer chasing and recursion,
//! causing cache misses and branch mispredictions.
//!
//! ### 3. Modern Standard
//! PBWT is the algorithm used by state-of-the-art tools (SHAPEIT4, SHAPEIT5, Eagle).
//! It represents the mathematically correct way to solve the haplotype matching
//! problem. Using recursive partitioning would be a regression to older technology.
//!
//! ### 4. Best of Both Worlds
//! This implementation applies PBWT directly on compressed "Coded Steps" (dictionary
//! compression), combining the memory efficiency of Beagle's compression with the
//! superior search algorithm. Bidirectional PBWT (forward and backward) finds the
//! absolute best haplotypes matching on both sides, which is crucial for accurate
//! phasing and imputation.

pub mod hmm;
pub mod ibs2;
pub mod imp_states;
pub mod parameters;
pub mod pbwt;
pub mod phase_ibs;
pub mod states;
