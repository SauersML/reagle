//! # Run Statistics
//!
//! A module for tracking and reporting run statistics, similar to `main/RunStats.java`.

use std::time::Instant;

/// A struct for tracking and reporting run statistics.
pub struct RunStats {
    start: Instant,
}

impl RunStats {
    /// Creates a new `RunStats` instance.
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }
}
