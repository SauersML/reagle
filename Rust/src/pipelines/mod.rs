//! # Pipelines Module (Orchestration)
//!
//! ## Role
//! High-level workflow coordination. Connects I/O, data structures, and algorithms.
//! This is the "Controller" in MVC terms.
//!
//! ## Design Philosophy
//! - Pipelines **own** the execution flow.
//! - They call into `io` to load data, `model` to process, `io` to write.
//! - Parallelization (via rayon) is coordinated here.
//! - Progress reporting and logging happen here.
//!
//! ## Sub-modules
//! - `phasing`: Statistical phasing pipeline
//! - `imputation`: Genotype imputation pipeline
//!
//! ## Common Pattern
//! ```rust,ignore
//! pub fn run(config: Config) -> Result<()> {
//!     // 1. Setup
//!     let thread_pool = rayon::ThreadPoolBuilder::new()
//!         .num_threads(config.nthreads)
//!         .build()?;
//!
//!     // 2. Load data
//!     let reader = VcfReader::open(&config.gt)?;
//!     let windows = WindowIterator::new(reader, config.window_config());
//!
//!     // 3. Process windows in parallel
//!     let results: Vec<_> = thread_pool.install(|| {
//!         windows
//!             .par_bridge()
//!             .map(|window| process_window(window?, &config))
//!             .collect()
//!     })?;
//!
//!     // 4. Merge and write output
//!     let merged = merge_windows(&results);
//!     let mut writer = VcfWriter::create(&config.out)?;
//!     writer.write_phased(&merged)?;
//!
//!     Ok(())
//! }
//! ```

pub mod imputation;
pub mod phasing;
