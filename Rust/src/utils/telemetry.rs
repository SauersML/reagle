//! # Telemetry Blackboard
//!
//! Thread-safe progress tracking for phasing and imputation pipelines.
//! Uses atomic counters that can be cheaply updated from rayon parallel iterators.
//!
//! The blackboard pattern decouples work execution from progress reporting:
//! - Worker threads update atomic counters with minimal overhead
//! - A background heartbeat thread periodically reads and reports progress

use std::io::{self, IsTerminal, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Processing stage for high-level progress tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Stage {
    Initializing = 0,
    LoadingData = 1,
    PhasingBurnin = 2,
    PhasingMain = 3,
    PhasingStage2 = 4,
    Imputation = 5,
    WritingOutput = 6,
    Complete = 7,
}

impl Stage {
    pub fn as_str(self) -> &'static str {
        match self {
            Stage::Initializing => "Initializing",
            Stage::LoadingData => "Loading Data",
            Stage::PhasingBurnin => "Phasing (burn-in)",
            Stage::PhasingMain => "Phasing (main)",
            Stage::PhasingStage2 => "Phasing Stage 2",
            Stage::Imputation => "Imputation",
            Stage::WritingOutput => "Writing Output",
            Stage::Complete => "Complete",
        }
    }

    fn from_u64(val: u64) -> Self {
        match val {
            0 => Stage::Initializing,
            1 => Stage::LoadingData,
            2 => Stage::PhasingBurnin,
            3 => Stage::PhasingMain,
            4 => Stage::PhasingStage2,
            5 => Stage::Imputation,
            6 => Stage::WritingOutput,
            _ => Stage::Complete,
        }
    }
}

/// Global telemetry state - designed for cheap atomic updates from hot loops.
///
/// All fields use relaxed ordering since we only need eventual visibility,
/// not strict synchronization. The heartbeat thread reads approximate values.
pub struct TelemetryBlackboard {
    // --- Macro Progress (Stage) ---
    stage: AtomicU64,

    // --- Meso Progress (Window/Iteration) ---
    current_window: AtomicU64,
    total_windows: AtomicU64,
    current_iteration: AtomicU64,
    total_iterations: AtomicU64,

    // --- Micro Progress (within-window counters) ---
    samples_processed: AtomicU64,
    total_samples: AtomicU64,
    markers_processed: AtomicU64,
    total_markers: AtomicU64,

    // --- Timing ---
    start_time: Instant,
    last_progress_nanos: AtomicU64,

    // --- Control ---
    shutdown: AtomicBool,
}

impl TelemetryBlackboard {
    /// Create a new telemetry blackboard
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            stage: AtomicU64::new(Stage::Initializing as u64),
            current_window: AtomicU64::new(0),
            total_windows: AtomicU64::new(0),
            current_iteration: AtomicU64::new(0),
            total_iterations: AtomicU64::new(0),
            samples_processed: AtomicU64::new(0),
            total_samples: AtomicU64::new(0),
            markers_processed: AtomicU64::new(0),
            total_markers: AtomicU64::new(0),
            start_time: Instant::now(),
            last_progress_nanos: AtomicU64::new(0),
            shutdown: AtomicBool::new(false),
        })
    }

    // === Stage Updates ===

    #[inline]
    pub fn set_stage(&self, stage: Stage) {
        self.stage.store(stage as u64, Ordering::Relaxed);
        self.touch_progress();
    }

    #[inline]
    pub fn stage(&self) -> Stage {
        Stage::from_u64(self.stage.load(Ordering::Relaxed))
    }

    // === Timing ===

    #[inline]
    fn touch_progress(&self) {
        let elapsed = self.start_time.elapsed().as_nanos() as u64;
        self.last_progress_nanos.store(elapsed, Ordering::Relaxed);
    }

    pub fn elapsed_secs(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }

    // === Snapshot for Heartbeat ===

    fn snapshot(&self) -> TelemetrySnapshot {
        TelemetrySnapshot {
            stage: self.stage(),
            current_window: self.current_window.load(Ordering::Relaxed),
            total_windows: self.total_windows.load(Ordering::Relaxed),
            current_iteration: self.current_iteration.load(Ordering::Relaxed),
            total_iterations: self.total_iterations.load(Ordering::Relaxed),
            samples_processed: self.samples_processed.load(Ordering::Relaxed),
            total_samples: self.total_samples.load(Ordering::Relaxed),
            markers_processed: self.markers_processed.load(Ordering::Relaxed),
            total_markers: self.total_markers.load(Ordering::Relaxed),
            elapsed_secs: self.elapsed_secs(),
            last_progress_nanos: self.last_progress_nanos.load(Ordering::Relaxed),
            current_nanos: self.start_time.elapsed().as_nanos() as u64,
        }
    }

    fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::SeqCst)
    }

    fn signal_shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }
}

impl Default for TelemetryBlackboard {
    fn default() -> Self {
        // This is only used if someone needs a non-Arc version
        Self {
            stage: AtomicU64::new(Stage::Initializing as u64),
            current_window: AtomicU64::new(0),
            total_windows: AtomicU64::new(0),
            current_iteration: AtomicU64::new(0),
            total_iterations: AtomicU64::new(0),
            samples_processed: AtomicU64::new(0),
            total_samples: AtomicU64::new(0),
            markers_processed: AtomicU64::new(0),
            total_markers: AtomicU64::new(0),
            start_time: Instant::now(),
            last_progress_nanos: AtomicU64::new(0),
            shutdown: AtomicBool::new(false),
        }
    }
}

/// Snapshot of telemetry state at a point in time
struct TelemetrySnapshot {
    stage: Stage,
    current_window: u64,
    total_windows: u64,
    current_iteration: u64,
    total_iterations: u64,
    samples_processed: u64,
    total_samples: u64,
    markers_processed: u64,
    total_markers: u64,
    elapsed_secs: f64,
    last_progress_nanos: u64,
    current_nanos: u64,
}

/// Heartbeat output configuration
pub struct HeartbeatConfig {
    /// Interval between heartbeats (seconds)
    pub interval_secs: u64,
    /// Stall warning threshold (seconds with no progress)
    pub stall_threshold_secs: u64,
}

impl Default for HeartbeatConfig {
    fn default() -> Self {
        Self {
            interval_secs: 30,
            stall_threshold_secs: 300, // 5 minutes
        }
    }
}

/// Handle to the heartbeat thread
pub struct HeartbeatHandle {
    handle: Option<JoinHandle<()>>,
    blackboard: Arc<TelemetryBlackboard>,
}

impl HeartbeatHandle {
    /// Spawn the heartbeat thread
    pub fn spawn(blackboard: Arc<TelemetryBlackboard>, config: HeartbeatConfig) -> Self {
        let bb = blackboard.clone();
        let is_tty = io::stderr().is_terminal();

        let handle = thread::Builder::new()
            .name("heartbeat".to_string())
            .spawn(move || {
                heartbeat_loop(bb, config, is_tty);
            })
            .expect("Failed to spawn heartbeat thread");

        Self {
            handle: Some(handle),
            blackboard,
        }
    }

    /// Signal shutdown and wait for thread to finish
    pub fn shutdown(mut self) {
        self.blackboard.signal_shutdown();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for HeartbeatHandle {
    fn drop(&mut self) {
        self.blackboard.signal_shutdown();
        // Don't block in drop - just signal shutdown
    }
}

/// Get RSS memory usage in MB (Linux only)
fn get_rss_mb() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/self/statm")
            .ok()
            .and_then(|s| {
                let parts: Vec<&str> = s.split_whitespace().collect();
                // Second field is RSS in pages
                parts.get(1)?.parse::<u64>().ok()
            })
            .map(|pages| pages * 4096 / (1024 * 1024)) // pages to MB
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Format duration in human-readable form
fn format_duration(secs: f64) -> String {
    if secs < 60.0 {
        format!("{:.0}s", secs)
    } else if secs < 3600.0 {
        let mins = (secs / 60.0).floor();
        let remaining_secs = secs % 60.0;
        format!("{:.0}m{:.0}s", mins, remaining_secs)
    } else {
        format!("{:.1}h", secs / 3600.0)
    }
}

/// Main heartbeat loop
fn heartbeat_loop(bb: Arc<TelemetryBlackboard>, config: HeartbeatConfig, is_tty: bool) {
    let interval = Duration::from_secs(config.interval_secs);
    let mut last_markers = 0u64;
    let mut last_time = Instant::now();

    loop {
        thread::sleep(interval);

        if bb.is_shutdown() {
            break;
        }

        let snap = bb.snapshot();

        // Calculate velocity (markers per second)
        let now = Instant::now();
        let dt = now.duration_since(last_time).as_secs_f64();
        let markers_velocity = if dt > 0.1 {
            (snap.markers_processed.saturating_sub(last_markers)) as f64 / dt
        } else {
            0.0
        };
        last_markers = snap.markers_processed;
        last_time = now;

        // ETA calculation based on markers
        let eta_str = if markers_velocity > 0.0 && snap.total_markers > snap.markers_processed {
            let remaining = snap.total_markers - snap.markers_processed;
            let eta_secs = remaining as f64 / markers_velocity;
            format_duration(eta_secs)
        } else {
            "unknown".to_string()
        };

        // Stall detection
        let stall_secs =
            (snap.current_nanos.saturating_sub(snap.last_progress_nanos)) / 1_000_000_000;
        let is_stalled = stall_secs > config.stall_threshold_secs;

        let rss_mb = get_rss_mb();

        if is_tty {
            print_tty_progress(&snap, &eta_str, rss_mb, markers_velocity, is_stalled);
        } else {
            print_log_progress(&snap, &eta_str, rss_mb, markers_velocity, is_stalled);
        }
    }

    // Clear TTY line on shutdown
    if is_tty {
        eprint!("\r\x1b[K");
        let _ = io::stderr().flush();
    }
}

/// Print progress for TTY (rewriting single line)
fn print_tty_progress(
    snap: &TelemetrySnapshot,
    eta: &str,
    rss_mb: Option<u64>,
    velocity: f64,
    is_stalled: bool,
) {
    // Build window/iteration context
    let window_str = if snap.total_windows > 0 {
        format!("W{}/{}", snap.current_window, snap.total_windows)
    } else if snap.current_window > 0 {
        format!("W{}", snap.current_window)
    } else {
        String::new()
    };

    let iter_str = if snap.total_iterations > 0 {
        format!("I{}/{}", snap.current_iteration, snap.total_iterations)
    } else {
        String::new()
    };

    // Calculate progress percentage
    let progress_pct = if snap.total_markers > 0 {
        (snap.markers_processed as f64 / snap.total_markers as f64 * 100.0).min(100.0)
    } else if snap.total_windows > 0 {
        (snap.current_window as f64 / snap.total_windows as f64 * 100.0).min(100.0)
    } else {
        0.0
    };

    // Sample progress
    let sample_str = if snap.total_samples > 0 {
        format!("S{}/{}", snap.samples_processed, snap.total_samples)
    } else {
        String::new()
    };

    // Build progress bar (20 chars)
    let bar_width = 20;
    let filled = ((progress_pct / 100.0) * bar_width as f64) as usize;
    let bar: String = "=".repeat(filled.min(bar_width))
        + &" ".repeat(bar_width.saturating_sub(filled));

    let mem_str = rss_mb
        .map(|mb| format!(" {}MB", mb))
        .unwrap_or_default();
    let stall_str = if is_stalled { " [STALLED]" } else { "" };

    // Combine context strings
    let context_parts: Vec<&str> = [window_str.as_str(), iter_str.as_str(), sample_str.as_str()]
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect();
    let context = context_parts.join(" ");

    eprint!(
        "\r[{}] {:>5.1}% | {} {} | {:.0} mk/s | {} | ETA: {}{}{}    \x1b[K",
        bar,
        progress_pct,
        snap.stage.as_str(),
        context,
        velocity,
        format_duration(snap.elapsed_secs),
        eta,
        mem_str,
        stall_str
    );
    let _ = io::stderr().flush();
}

/// Print progress for non-TTY (structured log line)
fn print_log_progress(
    snap: &TelemetrySnapshot,
    eta: &str,
    rss_mb: Option<u64>,
    velocity: f64,
    is_stalled: bool,
) {
    eprintln!(
        "[HEARTBEAT] stage=\"{}\" window={}/{} iter={}/{} samples={}/{} markers={}/{} \
         velocity={:.0}/s elapsed={:.0}s eta={} rss_mb={} stalled={}",
        snap.stage.as_str(),
        snap.current_window,
        snap.total_windows,
        snap.current_iteration,
        snap.total_iterations,
        snap.samples_processed,
        snap.total_samples,
        snap.markers_processed,
        snap.total_markers,
        velocity,
        snap.elapsed_secs,
        eta,
        rss_mb
            .map(|m| m.to_string())
            .unwrap_or_else(|| "?".to_string()),
        is_stalled
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_roundtrip() {
        for stage in [
            Stage::Initializing,
            Stage::LoadingData,
            Stage::PhasingBurnin,
            Stage::PhasingMain,
            Stage::PhasingStage2,
            Stage::Imputation,
            Stage::WritingOutput,
            Stage::Complete,
        ] {
            assert_eq!(Stage::from_u64(stage as u64), stage);
        }
    }

    #[test]
    fn test_blackboard_updates() {
        let bb = TelemetryBlackboard::new();

        bb.set_stage(Stage::Imputation);
        assert_eq!(bb.stage(), Stage::Imputation);
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(30.0), "30s");
        assert_eq!(format_duration(90.0), "1m30s");
        assert_eq!(format_duration(3661.0), "1.0h");
    }
}
