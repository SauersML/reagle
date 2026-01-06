//! # Binary (Zero-Copy) Dictionary Storage
//!
//! Provides a read-only, memory-mapped implementation of dictionary storage.
//! This allows instantaneous loading of massive reference panels.

use std::sync::Arc;
use memmap2::Mmap;
use bytemuck::pod_read_unaligned;

use crate::data::HapIdx;

/// A dictionary-compressed block backed by a memory map.
///
/// Layout in memory/file (Little Endian):
/// - n_markers (u32)
/// - n_haps (u32)
/// - n_patterns (u32)
/// - _padding (u32) - align to 8 bytes
/// - hap_to_pattern [u16; n_haps]
/// - _padding - align to 8 bytes
/// - patterns [u64; n_patterns * words_per_pattern]
#[derive(Clone, Debug)]
pub struct BinaryDictionaryColumn {
    /// Reference to the memory map (kept alive by Arc)
    mmap: Arc<Mmap>,
    
    // Computed offsets relative to block_start
    hap_to_pattern_start: usize,
    patterns_start: usize,
    words_per_pattern: usize,
    n_haps: usize,
}

impl BinaryDictionaryColumn {
    /// Parse a binary dictionary block from the mmap at the given offset.
    /// Returns the column and the total bytes consumed.
    pub fn new(mmap: Arc<Mmap>, block_offset: usize) -> Option<(Self, usize)> {
        let data = &mmap[block_offset..];
        
        if data.len() < 16 {
            return None;
        }

        // Read header
        let n_markers = pod_read_unaligned::<u32>(&data[0..4]) as usize;
        let n_haps = pod_read_unaligned::<u32>(&data[4..8]) as usize;
        let n_patterns = pod_read_unaligned::<u32>(&data[8..12]) as usize;
        // 4 bytes padding at 12..16

        let mut cursor = 16;

        // hap_to_pattern array (u16)
        let hap_to_pattern_start = cursor;
        let hap_bytes = n_haps * 2;
        cursor += hap_bytes;

        // Align to 8 bytes for patterns (u64)
        let padding = (8 - (cursor % 8)) % 8;
        cursor += padding;

        let patterns_start = cursor;
        
        // Calculate pattern size
        let words_per_pattern = (n_markers + 63) / 64;
        let patterns_bytes = n_patterns * words_per_pattern * 8;
        cursor += patterns_bytes;

        if cursor > data.len() {
            return None;
        }

        let col = Self {
            mmap,
            hap_to_pattern_start: block_offset + hap_to_pattern_start,
            patterns_start: block_offset + patterns_start,
            words_per_pattern,
            n_haps,
        };

        Some((col, cursor))
    }

    /// Get allele at (marker_offset, haplotype)
    #[inline]
    pub fn get(&self, marker_offset: usize, hap: HapIdx) -> u8 {
        let h = hap.as_usize();
        if h >= self.n_haps {
            return 0; // Out of bounds safety
        }

        // Read pattern index
        // Safety: We verified bounds in new()
        let pattern_idx = unsafe {
            let offset = self.hap_to_pattern_start + h * 2;
            let ptr = self.mmap.as_ptr().add(offset);
            std::ptr::read_unaligned(ptr as *const u16)
        } as usize;

        // Read bit from pattern
        // Pattern layout: flat array of u64s. 
        // Pattern P is at patterns_start + P * words_per_pattern * 8
        let pattern_offset = self.patterns_start + pattern_idx * self.words_per_pattern * 8;
        
        // Find which u64 word and which bit
        let word_idx = marker_offset / 64;
        let bit_idx = marker_offset % 64;

        if word_idx >= self.words_per_pattern {
            return 0;
        }

        unsafe {
            let ptr = self.mmap.as_ptr().add(pattern_offset + word_idx * 8);
            let word = std::ptr::read_unaligned(ptr as *const u64);
            ((word >> bit_idx) & 1) as u8
        }
    }

    pub fn n_haplotypes(&self) -> usize {
        self.n_haps
    }
}

