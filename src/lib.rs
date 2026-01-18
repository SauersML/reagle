//! # Reagle Library
//!
//! High-performance genotype phasing and imputation library.
//! A Rust reimplementation of Beagle.
//!
//! ## Modules
//! - `config`: CLI argument parsing and validation
//! - `data`: In-memory representations of genomic data
//! - `error`: Error types and result aliases
//! - `io`: File reading/writing (VCF, windows)
//! - `model`: Statistical models (PBWT, HMM)
//! - `pipelines`: High-level workflow orchestration
//! - `utils`: Shared utilities (workspace pattern)

pub mod config;
pub mod data;
pub mod error;
pub mod io;
pub mod model;
pub mod pipelines;
pub mod utils;

// Re-export commonly used types
pub use config::Config;
pub use data::haplotype::{HapIdx, SampleIdx, Samples};
pub use data::marker::{Marker, MarkerIdx, Markers};
pub use data::genetic_map::{GeneticMap, GeneticMaps, MarkerMap};
pub use data::storage::GenotypeMatrix;
pub use error::{ReagleError, Result};
pub use io::bref3::Bref3Reader;
pub use io::vcf::{VcfReader, VcfWriter};
pub use model::hmm::BeagleHmm;
pub use model::states::ThreadedHaps;
pub use model::parameters::ModelParams;

pub use pipelines::{ImputationPipeline, PhasingPipeline};
