//! # I/O Module
//!
//! File reading/writing boundaries. Converts between disk formats and in-memory
//! `GenotypeMatrix` representation.

pub mod bref3;
pub mod streaming;
pub mod vcf;
pub mod window;
