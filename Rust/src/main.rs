//! # Application Entry Point
//!
//! ## Role
//! The CLI entry point for Reagle.
//!
//! ## Spec
//! - Initialize the logger (`env_logger` or `tracing`).
//! - Use `clap` to parse CLI arguments into the `Config` struct.
//! - Determine mode (Phasing vs. Imputation) based on input flags.
//! - Initialize the global thread pool (`rayon::ThreadPoolBuilder`).
//! - Call the appropriate pipeline orchestrator:
//!   - `pipelines::phasing::run()` for phasing mode
//!   - `pipelines::imputation::run()` for imputation mode
//! - Handle top-level errors and exit gracefully with appropriate codes.
//!
//! ## Dependencies
//! - `config::Config` for CLI parsing
//! - `error::ReagleError` for error handling
//! - `pipelines::*` for execution
