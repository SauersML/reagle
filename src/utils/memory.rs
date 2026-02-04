//! # Memory Utilities
//!
//! Helper functions for memory detection and management.

use sysinfo::System;

const MIN_AVAIL_BYTES_FOR_PLANNING: u64 = 64 * 1024 * 1024;

/// Get the system page size in bytes.
#[cfg(target_os = "linux")]
pub fn get_page_size() -> u64 {
    let val = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if val > 0 {
        val as u64
    } else {
        4096
    }
}

/// Estimate available memory in bytes, accounting for cgroups.
pub fn available_memory_bytes() -> Option<u64> {
    fn read_cgroup_limit_bytes() -> Option<u64> {
        let v2 = std::fs::read_to_string("/sys/fs/cgroup/memory.max").ok();
        if let Some(s) = v2 {
            let t = s.trim();
            if t != "max" {
                if let Ok(v) = t.parse::<u64>() {
                    if v > 0 && v < (1u64 << 60) {
                        return Some(v);
                    }
                }
            }
        }
        let v1 = std::fs::read_to_string("/sys/fs/cgroup/memory/memory.limit_in_bytes").ok();
        if let Some(s) = v1 {
            let t = s.trim();
            if let Ok(v) = t.parse::<u64>() {
                if v > 0 && v < (1u64 << 60) {
                    return Some(v);
                }
            }
        }
        None
    }

    fn read_cgroup_available_bytes(limit: u64) -> Option<u64> {
        let v2 = std::fs::read_to_string("/sys/fs/cgroup/memory.current").ok();
        if let Some(s) = v2 {
            if let Ok(cur) = s.trim().parse::<u64>() {
                return Some(limit.saturating_sub(cur));
            }
        }
        let v1 = std::fs::read_to_string("/sys/fs/cgroup/memory/memory.usage_in_bytes").ok();
        if let Some(s) = v1 {
            if let Ok(cur) = s.trim().parse::<u64>() {
                return Some(limit.saturating_sub(cur));
            }
        }
        None
    }

    let mut sys = System::new();
    sys.refresh_memory();
    let mut avail_bytes = sys.available_memory();
    let mut total_bytes = sys.total_memory();
    if total_bytes > 0 {
        let scaled_total = total_bytes.saturating_mul(1024);
        let looks_like_kib = total_bytes < 1_073_741_824
            && scaled_total >= 1_073_741_824
            && scaled_total <= (1u64 << 50);
        if looks_like_kib {
            avail_bytes = avail_bytes.saturating_mul(1024);
            total_bytes = scaled_total;
        }
    }
    if let Some(limit) = read_cgroup_limit_bytes() {
        if limit > 0 {
            total_bytes = total_bytes.min(limit);
            if let Some(avail) = read_cgroup_available_bytes(limit) {
                avail_bytes = avail_bytes.min(avail);
            } else {
                avail_bytes = avail_bytes.min(limit);
            }
        }
    }

    if avail_bytes >= MIN_AVAIL_BYTES_FOR_PLANNING {
        return Some(avail_bytes);
    }
    if total_bytes > 0 {
        Some(total_bytes)
    } else {
        None
    }
}
