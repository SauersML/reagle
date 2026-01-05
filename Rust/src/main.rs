//! # Reagle: High-Performance Genotype Phasing and Imputation
//!
//! A Rust reimplementation of Beagle, optimized for modern hardware.
//!
//! ## Usage
//! ```bash
//! # Phasing only
//! reagle --gt input.vcf.gz --out phased
//!
//! # Imputation with reference panel
//! reagle --gt input.vcf.gz --ref reference.vcf.gz --out imputed
//! ```

use std::time::Instant;

mod config;
mod data;
mod error;
mod io;
mod model;
mod pipelines;
mod utils;

use config::Config;
use error::Result;
use pipelines::{ImputationPipeline, PhasingPipeline};

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let start = Instant::now();

    // Parse and validate configuration
    let config = Config::parse_and_validate()?;

    // Configure thread pool
    let n_threads = config.nthreads();
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()
        .ok();

    eprintln!("Reagle v0.1.0");
    eprintln!("Threads: {}", n_threads);

    // Run appropriate pipeline
    if config.is_imputation_mode() {
        eprintln!("Mode: Imputation");
        eprintln!("Target: {:?}", config.gt);
        eprintln!("Reference: {:?}", config.r#ref.as_ref().unwrap());
        
        let mut pipeline = ImputationPipeline::new(config);
        pipeline.run()?;
    } else {
        eprintln!("Mode: Phasing");
        eprintln!("Input: {:?}", config.gt);
        
        let mut pipeline = PhasingPipeline::new(config);
        pipeline.run()?;
    }

    let elapsed = start.elapsed();
    eprintln!("Completed in {:.2}s", elapsed.as_secs_f64());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_imports() {
        // Verify all modules are accessible
        let _ = config::Config::parse_and_validate;
        let _ = error::ReagleError::vcf("test");
        let _ = data::marker::MarkerIdx::new;
        let _ = io::vcf::VcfReader::open;
        let _ = model::parameters::ModelParams::new;
        let _ = pipelines::PhasingPipeline::new;
        let _ = utils::workspace::Workspace::new;
    }
}