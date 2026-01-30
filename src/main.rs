//! # Reagle: High-Performance Genotype Phasing and Imputation
//!
//! A Rust imputation/phasing engine inspired by Beagle, optimized for modern hardware.
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
use utils::telemetry::{HeartbeatConfig, HeartbeatHandle, Stage, TelemetryBlackboard};

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

/// Initialize tracing subscriber for hierarchical profiling output
fn init_profiling() {
    use tracing_subscriber::fmt::format::FmtSpan;
    use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt};

    tracing_subscriber::registry()
        .with(
            fmt::layer()
                .with_span_events(FmtSpan::CLOSE)
                .with_target(false)
                .with_timer(fmt::time::uptime()),
        )
        .init();
}

fn run() -> Result<()> {
    let start = Instant::now();

    // Parse and validate configuration
    let config = Config::parse_and_validate()?;

    #[cfg(feature = "pprof")]
    let pprof_guard = maybe_start_pprof();

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
        .map(|_| ())
        .unwrap_or(());

    eprintln!("Reagle v0.1.0");
    eprintln!("Threads: {}", n_threads);

    // Initialize telemetry blackboard and heartbeat thread
    let telemetry = TelemetryBlackboard::new();
    let heartbeat = HeartbeatHandle::spawn(telemetry.clone(), HeartbeatConfig::default());

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

    #[cfg(feature = "pprof")]
    if let Some(guard) = pprof_guard {
        maybe_write_pprof(guard);
    }

    Ok(())
}

#[cfg(test)]
mod tests {}

#[cfg(feature = "pprof")]
fn maybe_start_pprof() -> Option<pprof::ProfilerGuard<'static>> {
    if std::env::var("REAGLE_PPROF").is_ok() {
        match pprof::ProfilerGuard::new(100) {
            Ok(guard) => Some(guard),
            Err(err) => {
                eprintln!("pprof: failed to start profiler: {}", err);
                None
            }
        }
    } else {
        None
    }
}

#[cfg(feature = "pprof")]
fn maybe_write_pprof(guard: pprof::ProfilerGuard<'static>) {
    let report = match guard.report().build() {
        Ok(report) => report,
        Err(err) => {
            eprintln!("pprof: failed to build report: {}", err);
            return;
        }
    };
    let output_path =
        std::env::var("REAGLE_PPROF_OUTPUT").unwrap_or_else(|_| "reagle.pprof.svg".to_string());
    match std::fs::File::create(&output_path) {
        Ok(mut file) => {
            if let Err(err) = report.flamegraph(&mut file) {
                eprintln!("pprof: failed to write flamegraph: {}", err);
            } else {
                eprintln!("pprof flamegraph written to {}", output_path);
            }
        }
        Err(err) => {
            eprintln!(
                "pprof: failed to create output file {}: {}",
                output_path, err
            );
        }
    }
}
