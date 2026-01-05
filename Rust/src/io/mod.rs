//! # I/O Module
//!
//! File reading/writing boundaries. Converts between disk formats and in-memory
//! `GenotypeMatrix` representation.

pub mod streaming;
pub mod vcf;
pub mod window;

pub use streaming::{StreamingVcfReader, StreamingConfig, StreamWindow};
pub use vcf::{VcfReader, VcfWriter, ImputationQuality, MarkerImputationStats};
pub use window::{Window, WindowBuilder, SlidingWindowIterator};