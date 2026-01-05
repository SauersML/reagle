//! # I/O Module
//!
//! File reading/writing boundaries. Converts between disk formats and in-memory
//! `GenotypeMatrix` representation.

pub mod vcf;
pub mod window;

pub use vcf::{VcfReader, VcfWriter};
pub use window::{Window, WindowBuilder, SlidingWindowIterator};