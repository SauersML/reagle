//! # I/O Module
//!
//! ## Role
//! File reading/writing boundaries. Converts between disk formats and in-memory
//! `GenotypeMatrix` representation.
//!
//! ## Design Philosophy
//! - I/O is **separate** from data structures.
//! - The `data::storage` module doesn't know about VCF or file formats.
//! - This module handles parsing and immediately converts to optimal storage.
//!
//! ## Sub-modules
//! - `vcf`: VCF/BCF reading and writing using `noodles`
//! - `window`: Sliding window chunking for chromosome-scale data
//!
//! ## Future Extensions
//! - `bref3`: Binary reference format (deferred for v1.0)
//! - `pgen`: PLINK 2.0 format support

pub mod vcf;
pub mod window;
