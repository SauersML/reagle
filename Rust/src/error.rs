//! # Centralized Error Handling
//!
//! ## Role
//! Unified error types for the entire crate using `thiserror`.
//!
//! ## Spec
//! Define `enum ReagleError` with variants:
//!
//! - `Io(#[from] std::io::Error)`
//!   File missing, permission denied, read/write failures.
//!
//! - `Vcf(String)`
//!   Malformed VCF records, missing fields, parse errors.
//!   (Wrap noodles errors as String for simpler error chains)
//!
//! - `InvalidData { msg: String }`
//!   Inconsistent data: sample count mismatch between files,
//!   marker position out of order, invalid allele codes.
//!
//! - `Algorithm { msg: String }`
//!   Math errors: non-finite probabilities, matrix dimension mismatch,
//!   convergence failures.
//!
//! - `Config { msg: String }`
//!   Invalid CLI arguments caught after clap parsing.
//!
//! ## Type Alias
//! ```rust,ignore
//! pub type Result<T> = std::result::Result<T, ReagleError>;
//! ```
//!
//! ## Usage
//! ```rust,ignore
//! use crate::error::{ReagleError, Result};
//!
//! fn load_vcf(path: &Path) -> Result<GenotypeMatrix> {
//!     let file = File::open(path)?; // auto-converts io::Error
//!     // ...
//! }
//! ```
