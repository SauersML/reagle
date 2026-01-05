//! # Utilities Module
//!
//! ## Role
//! Cross-cutting helpers that don't belong in domain-specific modules.
//!
//! ## Sub-modules
//! - `threading`: Rayon thread pool configuration
//! - `workspace`: Pre-allocated buffers for zero-allocation hot paths
//!
//! ## Design Notes
//! These utilities exist to support performance-critical patterns:
//! - Avoid allocations in inner loops
//! - Configure parallelism appropriately
//! - Provide reusable infrastructure

pub mod threading;
pub mod workspace;
