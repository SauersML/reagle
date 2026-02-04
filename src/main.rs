//! # Reagle: High-Performance Genotype Phasing and Imputation
//!
//! A Rust imputation/phasing engine inspired by Beagle, optimized for modern hardware.
//!
//! ## Usage
//! ```bash
//! # Phasing only
//! reagle --target input.vcf.gz --out phased
//!
//! # Imputation with reference panel
//! reagle --target input.vcf.gz --ref reference.vcf.gz --out imputed
//!
//! # With profiling output
//! reagle --target input.vcf.gz --ref reference.vcf.gz --out imputed --profile
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
    use tracing_subscriber::filter::LevelFilter;
    use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("trace"));

    tracing_subscriber::registry()
        .with(filter.add_directive(LevelFilter::TRACE.into()))
        .with(
            fmt::layer()
                .with_span_events(FmtSpan::CLOSE)
                .with_ansi(false)
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
        eprintln!("Target: {:?}", config.target);
        eprintln!("Reference: {:?}", config.r#ref.as_ref().unwrap());

        let mut pipeline = ImputationPipeline::new(config, Some(telemetry.clone()));
        pipeline.run()?;
    } else {
        eprintln!("Mode: Phasing");
        eprintln!("Input: {:?}", config.target);

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
    let text_output_path = std::env::var("REAGLE_PPROF_TEXT").unwrap_or_else(|_| {
        if let Some(stem) = output_path.strip_suffix(".svg") {
            format!("{}.txt", stem)
        } else {
            format!("{}.txt", output_path)
        }
    });
    let folded_output_path = if let Some(stem) = text_output_path.strip_suffix(".txt") {
        format!("{}.folded.txt", stem)
    } else {
        format!("{}.folded.txt", text_output_path)
    };
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

    match std::fs::File::create(&text_output_path) {
        Ok(mut file) => {
            use std::collections::HashMap;
            use std::io::Write as _;

            let mut total_samples: i64 = 0;
            let mut per_thread: HashMap<String, i64> = HashMap::new();
            let mut per_symbol: HashMap<String, i64> = HashMap::new();

            for (key, value) in report.data.iter() {
                let count = *value as i64;
                total_samples += count;
                *per_thread.entry(key.thread_name_or_id()).or_insert(0) += count;

                for frame in key.frames.iter() {
                    for symbol in frame.iter() {
                        let name = format!("{}", symbol);
                        *per_symbol.entry(name).or_insert(0) += count;
                    }
                }
            }

            writeln!(file, "Reagle pprof summary").ok();
            writeln!(file, "Total samples: {}", total_samples).ok();
            writeln!(file).ok();

            let mut threads: Vec<(String, i64)> = per_thread.into_iter().collect();
            threads.sort_by(|a, b| b.1.cmp(&a.1));

            let mut symbols: Vec<(String, i64)> = per_symbol.into_iter().collect();
            symbols.sort_by(|a, b| b.1.cmp(&a.1));
            writeln!(file, "Top functions (inclusive samples):").ok();
            for (name, count) in symbols.iter().take(50) {
                let pct = if total_samples > 0 {
                    (count * 100) as f64 / total_samples as f64
                } else {
                    0.0
                };
                writeln!(file, "  {:>6.2}%  {:>8}  {}", pct, count, name).ok();
            }

            let mut stacks: Vec<(String, i64)> = Vec::new();
            for (key, value) in report.data.iter() {
                let mut stack = String::new();
                stack.push_str(&key.thread_name_or_id());
                stack.push_str(" :: ");
                let mut first = true;
                for frame in key.frames.iter().rev() {
                    for symbol in frame.iter().rev() {
                        if !first {
                            stack.push_str(" <- ");
                        }
                        first = false;
                        stack.push_str(&format!("{}", symbol));
                    }
                }
                stacks.push((stack, *value as i64));
            }
            stacks.sort_by(|a, b| b.1.cmp(&a.1));
            writeln!(file).ok();
            writeln!(file, "Top stacks (thread :: root <- leaf):").ok();
            for (stack, count) in stacks.iter().take(50) {
                let pct = if total_samples > 0 {
                    (count * 100) as f64 / total_samples as f64
                } else {
                    0.0
                };
                writeln!(file, "  {:>6.2}%  {:>8}  {}", pct, count, stack).ok();
            }

            eprintln!("pprof summary written to {}", text_output_path);
        }
        Err(err) => {
            eprintln!(
                "pprof: failed to create output file {}: {}",
                text_output_path, err
            );
        }
    }

    match std::fs::File::create(&folded_output_path) {
        Ok(mut file) => {
            use std::fmt::Write as _;
            use std::io::Write as _;
            for (key, value) in report.data.iter() {
                let mut line = key.thread_name_or_id();
                line.push(';');
                for frame in key.frames.iter().rev() {
                    for symbol in frame.iter().rev() {
                        write!(&mut line, "{};", symbol).ok();
                    }
                }
                line.pop();
                write!(&mut line, " {}", value).ok();
                writeln!(file, "{}", line).ok();
            }
            eprintln!("pprof folded stacks written to {}", folded_output_path);
        }
        Err(err) => {
            eprintln!(
                "pprof: failed to create output file {}: {}",
                folded_output_path, err
            );
        }
    }
}
