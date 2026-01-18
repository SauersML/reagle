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
//!
//! # With profiling output
//! reagle --gt input.vcf.gz --ref reference.vcf.gz --out imputed --profile
//! ```

use std::time::Instant;

use reagle::config::Config;
use reagle::Result;
use reagle::pipelines::{ImputationPipeline, PhasingPipeline};
use reagle::utils::telemetry::{HeartbeatConfig, HeartbeatHandle, Stage, TelemetryBlackboard};

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

/// Initialize tracing subscriber for hierarchical profiling output
fn init_profiling() {
    use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt};
    use tracing_subscriber::fmt::format::FmtSpan;

    tracing_subscriber::registry()
        .with(
            fmt::layer()
                .with_span_events(FmtSpan::CLOSE)
                .with_target(false)
                .with_timer(fmt::time::uptime())
        )
        .init();
}

fn run() -> Result<()> {
    let start = Instant::now();

    // Parse and validate configuration
    let config = Config::parse_and_validate()?;

    // Initialize profiling if requested
    if config.profile {
        init_profiling();
        eprintln!("=== Profiling enabled ===\n");
    }

    // Configure thread pool
    let n_threads = config.nthreads();
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()
        .ok();

    eprintln!("Reagle v0.1.0");
    eprintln!("Threads: {}", n_threads);

    // Initialize telemetry blackboard and heartbeat thread
    let telemetry = TelemetryBlackboard::new();
    let heartbeat = HeartbeatHandle::spawn(
        telemetry.clone(),
        HeartbeatConfig::default(),
    );

    telemetry.set_stage(Stage::LoadingData);

    // Run appropriate pipeline
    if config.is_imputation_mode() {
        eprintln!("Mode: Imputation");
        eprintln!("Target: {:?}", config.gt);
        eprintln!("Reference: {:?}", config.r#ref.as_ref().unwrap());

        let mut pipeline = ImputationPipeline::new(config, Some(telemetry.clone()));
        pipeline.run()?;
    } else {
        eprintln!("Mode: Phasing");
        eprintln!("Input: {:?}", config.gt);

        let mut pipeline = PhasingPipeline::new(config, Some(telemetry.clone()));
        pipeline.run_auto()?;
    }

    // Signal completion and shutdown heartbeat
    telemetry.set_stage(Stage::Complete);
    heartbeat.shutdown();

    let elapsed = start.elapsed();
    eprintln!("\nCompleted in {:.2}s", elapsed.as_secs_f64());

    Ok(())
}

#[cfg(test)]
mod tests {
    use reagle::{config, data, error, io, model, pipelines};

    #[test]
    fn test_module_imports() {
        // Verify all modules are accessible
        let _ = config::Config::parse_and_validate;
        let _ = error::ReagleError::vcf("test");
        let _ = data::marker::MarkerIdx::new;
        let _ = io::vcf::VcfReader::open;
        let _ = model::parameters::ModelParams::new;
        let _ = pipelines::PhasingPipeline::new;
    }
}
