//! # Positional Burrows-Wheeler Transform (PBWT)
//!
//! Implementation of the PBWT algorithm for efficient haplotype matching.
//! Based on Durbin (2014) "Efficient haplotype matching and storage using
//! the positional Burrows-Wheeler transform (PBWT)".
//!
//! This implementation follows the Beagle Java code (PbwtUpdater.java and
//! PbwtDivUpdater.java) closely for correctness.
//!
//! ## Key Concepts
//! - `Prefix array (a)`: Permutation of haplotypes sorted by reverse prefixes
//! - `Divergence array (d)`: Position where each haplotype diverges from predecessor
//!
//! ## Reference
//! Durbin, Richard (2014) Efficient haplotype matching and storage using the
//! positional Burrows-Wheeler transform (PBWT).

use tracing::info_span;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};

#[inline(always)]
fn prefetch_read<T>(ptr: *const T) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe {
        _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        std::hint::black_box(ptr);
    }
}

#[inline(always)]
fn pbwt_bin_for_allele(allele: u8, alphabet: PbwtAlphabet) -> usize {
    PbwtAllele::from_raw(allele, alphabet).bin(alphabet)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PbwtAlphabet {
    n_alleles: u16,
}

impl PbwtAlphabet {
    pub fn new(n_alleles: usize) -> Option<Self> {
        if (2..=usize::from(u8::MAX)).contains(&n_alleles) {
            Some(Self {
                n_alleles: n_alleles as u16,
            })
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn n_alleles(self) -> usize {
        self.n_alleles as usize
    }

    #[inline(always)]
    pub fn n_bins(self) -> usize {
        self.n_alleles() + 1
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PbwtAllele {
    Ref,
    Alt(u8),
    Missing,
}

#[repr(usize)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PbwtBiallelicBin {
    Ref = 0,
    Missing = 1,
    Alt = 2,
}

impl PbwtBiallelicBin {
    pub const NON_MISSING: [Self; 2] = [Self::Ref, Self::Alt];

    #[inline(always)]
    pub const fn as_usize(self) -> usize {
        self as usize
    }
}

impl PbwtAllele {
    #[inline(always)]
    pub fn from_raw(allele: u8, alphabet: PbwtAlphabet) -> Self {
        if allele == 0 {
            Self::Ref
        } else if allele == crate::data::storage::AlleleCode::MISSING.raw() || (allele as usize) >= alphabet.n_alleles() {
            Self::Missing
        } else {
            Self::Alt(allele)
        }
    }

    #[inline(always)]
    pub fn bin(self, alphabet: PbwtAlphabet) -> usize {
        if alphabet.n_alleles() == 2 {
            self.biallelic_bin().as_usize()
        } else {
            match self {
                Self::Ref => 0,
                Self::Missing => 1,
                Self::Alt(a) => (a as usize) + 1,
            }
        }
    }

    #[inline(always)]
    pub fn biallelic_bin(self) -> PbwtBiallelicBin {
        match self {
            Self::Ref => PbwtBiallelicBin::Ref,
            Self::Alt(1) => PbwtBiallelicBin::Alt,
            Self::Alt(_) | Self::Missing => PbwtBiallelicBin::Missing,
        }
    }
}

pub trait PbwtIndex: Copy + Default {
    fn from_usize(v: usize) -> Self;
    fn to_usize(self) -> usize;
    fn to_u32(self) -> u32;
}

impl PbwtIndex for u32 {
    #[inline]
    fn from_usize(v: usize) -> Self {
        v as u32
    }

    #[inline]
    fn to_usize(self) -> usize {
        self as usize
    }

    #[inline]
    fn to_u32(self) -> u32 {
        self
    }
}

impl PbwtIndex for u16 {
    #[inline]
    fn from_usize(v: usize) -> Self {
        assert!(v <= u16::MAX as usize, "PBWT index overflow: {}", v);
        v as u16
    }

    #[inline]
    fn to_usize(self) -> usize {
        self as usize
    }

    #[inline]
    fn to_u32(self) -> u32 {
        self as u32
    }
}

/// PBWT updater with divergence array tracking
///
/// This optimized implementation uses flat arrays and Counting Sort
/// to avoid heap allocations during updates.
#[derive(Debug)]
pub struct PbwtDivUpdater<I: PbwtIndex = u32> {
    /// Number of haplotypes
    n_haps: usize,
    /// Current prefix array (flat)
    a: Vec<I>,
    /// Current divergence array (flat)
    d: Vec<i32>,
    /// Scratch prefix array for double buffering
    scratch_a: Vec<I>,
    /// Scratch divergence array for double buffering
    scratch_d: Vec<i32>,
    /// Pre-permuted alleles: permuted_alleles[i] = alleles[prefix[i]]
    /// Converts random-access gather to sequential access for counting and scatter
    permuted_alleles: Vec<u8>,
    /// Bit-packed permuted alleles for biallelic fast path (1 = ALT)
    permuted_bits: Vec<u64>,
    /// Bit-packed permuted missing mask for biallelic fast path (1 = missing)
    permuted_missing_bits: Vec<u64>,
    /// Word-level base offsets for biallelic scatter (prefix order -> bin order)
    word_base0: Vec<usize>,
    /// Word-level base offsets for allele-1 bin
    word_base1: Vec<usize>,
    /// Word-level base offsets for missing bin
    word_base_miss: Vec<usize>,
    /// Per-word ALT1 counts for biallelic packed words
    word_count1: Vec<u8>,
    /// Per-word missing counts for biallelic packed words
    word_count_miss: Vec<u8>,
    /// Propagation array for tracking max/min divergence across alleles
    p: Vec<i32>,
    /// Helper for counting sort: counts per allele
    counts: Vec<usize>,
    /// Helper for counting sort: starting offset per allele
    offsets: Vec<usize>,
}

impl<I: PbwtIndex> PbwtDivUpdater<I> {
    /// Create a new PBWT divergence updater
    pub fn new(n_haps: usize) -> Self {
        let max_alleles = 256; // Max u8 alleles
        Self {
            n_haps,
            a: Vec::new(), // Will be initialized on first use
            d: Vec::new(),
            scratch_a: Vec::new(),
            scratch_d: Vec::new(),
            permuted_alleles: Vec::new(),
            permuted_bits: Vec::new(),
            permuted_missing_bits: Vec::new(),
            word_base0: Vec::new(),
            word_base1: Vec::new(),
            word_base_miss: Vec::new(),
            word_count1: Vec::new(),
            word_count_miss: Vec::new(),
            p: vec![0; max_alleles],
            counts: vec![0; max_alleles],
            offsets: vec![0; max_alleles + 1],
        }
    }

    pub fn n_haps(&self) -> usize {
        self.n_haps
    }

    fn ensure_capacity(&mut self, n_alleles: usize) {
        if self.p.len() < n_alleles {
            self.p.resize(n_alleles, 0);
            self.counts.resize(n_alleles, 0);
            self.offsets.resize(n_alleles + 1, 0);
        }

        if self.scratch_a.len() < self.n_haps {
            self.scratch_a.resize(self.n_haps, I::from_usize(0));
            self.scratch_d.resize(self.n_haps, 0);
            self.permuted_alleles.resize(self.n_haps, 0);
            let n_words = (self.n_haps + 63) / 64;
            self.permuted_bits.resize(n_words, 0);
            self.permuted_missing_bits.resize(n_words, 0);
            self.word_base0.resize(n_words, 0);
            self.word_base1.resize(n_words, 0);
            self.word_base_miss.resize(n_words, 0);
            self.word_count1.resize(n_words, 0);
            self.word_count_miss.resize(n_words, 0);
            // Also initialize 'a' and 'd' if empty (lazy init)
            if self.a.is_empty() {
                self.a.resize(self.n_haps, I::from_usize(0));
                self.d.resize(self.n_haps, 0);
            }
        }
    }

    #[inline]
    fn pack_biallelic_bits(&mut self, alleles: &[u8], prefix: &[I]) -> (usize, usize) {
        let n_words = (self.n_haps + 63) / 64;
        if self.permuted_bits.len() < n_words {
            self.permuted_bits.resize(n_words, 0);
            self.permuted_missing_bits.resize(n_words, 0);
            self.word_base0.resize(n_words, 0);
            self.word_base1.resize(n_words, 0);
            self.word_base_miss.resize(n_words, 0);
            self.word_count1.resize(n_words, 0);
            self.word_count_miss.resize(n_words, 0);
        }

        let mut count1 = 0usize;
        let mut count_miss = 0usize;
        let mut idx = 0usize;

        for w in 0..n_words {
            let mut bits = 0u64;
            let mut miss = 0u64;
            let block_end = (idx + 64).min(self.n_haps);
            let mut bit = 0u64;
            while idx < block_end {
                let allele = alleles[prefix[idx].to_usize()];
                if allele == 1 {
                    bits |= 1u64 << bit;
                } else if allele > 1 {
                    miss |= 1u64 << bit;
                }
                idx += 1;
                bit += 1;
            }

            self.permuted_bits[w] = bits;
            self.permuted_missing_bits[w] = miss;
            let c1 = bits.count_ones() as usize;
            let cm = miss.count_ones() as usize;
            self.word_count1[w] = c1 as u8;
            self.word_count_miss[w] = cm as u8;
            count1 += c1;
            count_miss += cm;
        }

        (count1, count_miss)
    }

    #[inline]
    fn compute_biallelic_word_bases(&mut self, has_missing: bool) {
        let n_words = (self.n_haps + 63) / 64;
        if self.word_base0.len() < n_words {
            self.word_base0.resize(n_words, 0);
            self.word_base1.resize(n_words, 0);
            self.word_base_miss.resize(n_words, 0);
        }

        let mut base0 = self.offsets[0];
        let mut base1 = self.offsets[2];
        let mut base_miss = self.offsets[1];
        let mut idx = 0usize;

        for w in 0..n_words {
            let block_end = (idx + 64).min(self.n_haps);
            let block_len = block_end - idx;
            let count1 = self.word_count1[w] as usize;
            let count_m = if has_missing {
                self.word_count_miss[w] as usize
            } else {
                0
            };
            let count0 = block_len.saturating_sub(count1 + count_m);

            self.word_base0[w] = base0;
            self.word_base1[w] = base1;
            if has_missing {
                self.word_base_miss[w] = base_miss;
                base_miss += count_m;
            }

            base0 += count0;
            base1 += count1;
            idx = block_end;
        }
    }

    /// Forward update of prefix and divergence arrays
    ///
    /// Uses In-Place Counting Sort to remove allocations.
    ///
    /// # Forward PBWT Semantics (vs Backward)
    ///
    /// - **Forward PBWT** (markers 0 → M-1): divergence[i] = marker where match STARTS
    ///   - Small divergence = long match (started far back)
    ///   - Uses MAX propagation: latest divergence point limits the match
    ///   - Reset to MIN_VALUE after output
    ///
    /// - **Backward PBWT** (markers M-1 → 0): divergence[i] = marker where match ENDS
    ///   - Small divergence = short match (ends soon)
    ///   - Uses MIN propagation: earliest end point limits the match
    ///   - Reset to MAX_VALUE after output
    ///
    /// # Missing Data Handling
    ///
    /// Missing data (allele u8::MAX, or invalid allele >= n_alleles) is placed in
    /// its own bin at index 1. This keeps missing separate from both REF (0)
    /// and ALT bins.
    ///
    /// # Arguments
    /// * `alleles` - Allele for each haplotype
    /// * `n_alleles` - Number of distinct alleles
    /// * `marker` - Current marker index
    /// * `prefix` - Prefix array to update
    /// * `divergence` - Divergence array to update
    pub fn fwd_update(
        &mut self,
        alleles: &[u8],
        n_alleles: usize,
        marker: usize,
        prefix: &mut [I],
        divergence: &mut [i32],
    ) {
        info_span!(
            "pbwt_fwd_update",
            n_haps = self.n_haps,
            n_alleles = n_alleles
        )
        .in_scope(|| {
            assert_eq!(alleles.len(), self.n_haps);
            assert_eq!(prefix.len(), self.n_haps);
            assert!(divergence.len() >= self.n_haps);
            let alphabet = PbwtAlphabet::new(n_alleles).unwrap_or_else(|| {
                panic!(
                    "PBWT invalid n_alleles={} (expected 2..=u8::MAX with u8::MAX reserved for missing)",
                    n_alleles
                )
            });

            // Use n_alleles + 1 bins with Bin 1 reserved for missing/invalid.
            let n_bins = alphabet.n_bins();
            self.ensure_capacity(n_bins);

            // 1. Count frequencies of each allele (Counting Sort Phase 1) - now sequential access
            self.counts[..n_bins].fill(0);

            if n_alleles == 2 {
                let (count1, count_miss) = self.pack_biallelic_bits(alleles, prefix);
                self.counts[0] = self.n_haps - count1 - count_miss;
                self.counts[1] = count_miss; // Bin 1: Missing
                self.counts[2] = count1; // Bin 2: Allele 1
            } else {
                // General path for multiallelic
                // 0. Pre-permute alleles: gather alleles[prefix[i]] into contiguous buffer.
                // This converts subsequent random-access patterns to sequential access.
                // Single gather pass here enables two sequential passes below.
                for i in 0..self.n_haps {
                    self.permuted_alleles[i] = alleles[prefix[i].to_usize()];
                }

                for i in 0..self.n_haps {
                    let allele = self.permuted_alleles[i] as usize;
                    // Map missing/invalid alleles to dedicated Bin 1.
                    let bin = pbwt_bin_for_allele(allele as u8, alphabet);
                    self.counts[bin] += 1;
                }
            }

            // 2. Compute Offsets (Counting Sort Phase 2)
            let mut running = 0;
            for i in 0..n_bins {
                self.offsets[i] = running;
                running += self.counts[i];
            }

            // 3. Check if there's any missing data (before resetting counts)
            // This lets us use a faster 2-bin path when there's no missing data
            // Missing data is now in Bin 1
            let has_missing = self.counts[1] > 0;
            if n_alleles == 2 {
                self.compute_biallelic_word_bases(has_missing);
            }

            // 4. Initialize p array and reset counts for scatter pass
            let init_value = (marker + 1) as i32;
            self.counts[..n_bins].fill(0);
            self.p[..n_bins].fill(init_value);

            // 5. Scatter to scratch buffers with p propagation (Counting Sort Phase 3)
            // Now uses permuted_alleles for sequential access instead of random gather
            if n_alleles == 2 && !has_missing {
                // Fast biallelic path when no missing data - only 2 bins needed
                // This is the common case and avoids the 3rd comparison per haplotype
                let mut p0 = init_value;
                let mut p1 = init_value;

                let mut idx = 0usize;
                let n_words = (self.n_haps + 63) / 64;
                for w in 0..n_words {
                    let mut bits = self.permuted_bits[w];
                    let mut pos0 = self.word_base0[w];
                    let mut pos1 = self.word_base1[w];
                    let block_end = (idx + 64).min(self.n_haps);
                    while idx < block_end {
                        let hap = prefix[idx];
                        let div = divergence[idx];

                        // Propagate max to both bins
                        if div > p0 {
                            p0 = div;
                        }
                        if div > p1 {
                            p1 = div;
                        }

                        if (bits & 1) == 0 {
                            self.scratch_a[pos0] = hap;
                            self.scratch_d[pos0] = p0;
                            p0 = i32::MIN;
                            pos0 += 1;
                        } else {
                            self.scratch_a[pos1] = hap;
                            self.scratch_d[pos1] = p1;
                            p1 = i32::MIN;
                            pos1 += 1;
                        }

                        bits >>= 1;
                        idx += 1;
                    }
                }
            } else if n_alleles == 2 {
                // Biallelic path with missing data - needs 3 bins
                let mut p0 = init_value;
                let mut p1 = init_value;
                let mut p_miss = init_value;

                let mut idx = 0usize;
                let n_words = (self.n_haps + 63) / 64;
                for w in 0..n_words {
                    let mut bits = self.permuted_bits[w];
                    let mut miss = self.permuted_missing_bits[w];
                    let mut pos0 = self.word_base0[w];
                    let mut pos1 = self.word_base1[w];
                    let mut pos_miss = self.word_base_miss[w];
                    let block_end = (idx + 64).min(self.n_haps);
                    while idx < block_end {
                        let hap = prefix[idx];
                        let div = divergence[idx];

                        // Propagate max to all bins - this is essential for correctness
                        // The divergence must propagate through all allele bins
                        if div > p0 {
                            p0 = div;
                        }
                        if div > p1 {
                            p1 = div;
                        }
                        if div > p_miss {
                            p_miss = div;
                        }

                        if (miss & 1) != 0 {
                            // Missing or invalid allele
                            self.scratch_a[pos_miss] = hap;
                            self.scratch_d[pos_miss] = p_miss;
                            p_miss = i32::MIN;
                            pos_miss += 1;
                        } else if (bits & 1) == 0 {
                            self.scratch_a[pos0] = hap;
                            self.scratch_d[pos0] = p0;
                            p0 = i32::MIN;
                            pos0 += 1;
                        } else {
                            self.scratch_a[pos1] = hap;
                            self.scratch_d[pos1] = p1;
                            p1 = i32::MIN;
                            pos1 += 1;
                        }

                        bits >>= 1;
                        miss >>= 1;
                        idx += 1;
                    }
                }
            } else {
                // General multiallelic path
                let prefetch_stride = 64usize;
                for i in 0..self.n_haps {
                    if i + prefetch_stride < self.n_haps {
                        unsafe {
                            prefetch_read(self.permuted_alleles.as_ptr().add(i + prefetch_stride));
                            prefetch_read(prefix.as_ptr().add(i + prefetch_stride));
                            prefetch_read(divergence.as_ptr().add(i + prefetch_stride));
                        }
                    }
                    let hap = prefix[i];
                    let div = divergence[i];
                    let allele = self.permuted_alleles[i] as usize; // Sequential access
                    let bin = pbwt_bin_for_allele(allele as u8, alphabet);

                    // Update p (Max Divergence Propagation) for ALL bins
                    for j in 0..n_bins {
                        if div > self.p[j] {
                            self.p[j] = div;
                        }
                    }

                    let base = self.offsets[bin];
                    let offset = self.counts[bin];
                    let pos = base + offset;

                    self.scratch_a[pos] = hap;
                    self.scratch_d[pos] = self.p[bin];

                    // Reset p for this bin after output
                    self.p[bin] = i32::MIN;

                    self.counts[bin] += 1;
                }
            }

            // 6. Copy back
            prefix.copy_from_slice(&self.scratch_a[..self.n_haps]);
            divergence[..self.n_haps].copy_from_slice(&self.scratch_d[..self.n_haps]);
        })
    }

    /// Forward update for biallelic markers using precomputed prepared state.
    ///
    /// Exactness:
    /// - This runs the same counting-sort scatter and divergence propagation as
    ///   `fwd_update(..., n_alleles=2, ...)`.
    /// - The only difference is input source: precomputed permuted bits/counts
    ///   are provided by caller (`ReferencePbwt::prepare_step`) instead of
    ///   recomputing by scanning `alleles[prefix[i]]` again.
    pub fn fwd_update_biallelic_prepared(
        &mut self,
        marker: usize,
        prefix: &mut [I],
        divergence: &mut [i32],
        permuted_bits: &[u64],
        permuted_missing_bits: &[u64],
        binary_counts: [u32; 3],
    ) {
        assert_eq!(prefix.len(), self.n_haps);
        assert!(divergence.len() >= self.n_haps);
        let n_words = (self.n_haps + 63) / 64;
        assert!(permuted_bits.len() >= n_words);
        assert!(permuted_missing_bits.len() >= n_words);
        assert_eq!(
            binary_counts[0]
                .saturating_add(binary_counts[1])
                .saturating_add(binary_counts[2]) as usize,
            self.n_haps
        );

        let n_bins = 3usize;
        self.ensure_capacity(n_bins);
        self.counts[0] = binary_counts[0] as usize;
        self.counts[1] = binary_counts[1] as usize;
        self.counts[2] = binary_counts[2] as usize;

        let mut running = 0usize;
        for i in 0..n_bins {
            self.offsets[i] = running;
            running += self.counts[i];
        }

        let has_missing = self.counts[1] > 0;
        self.compute_biallelic_word_bases(has_missing);

        let init_value = (marker + 1) as i32;
        self.counts[..n_bins].fill(0);
        self.p[..n_bins].fill(init_value);

        if !has_missing {
            let mut p0 = init_value;
            let mut p1 = init_value;
            let mut idx = 0usize;
            for (w, bits_word) in permuted_bits.iter().enumerate().take(n_words) {
                let mut bits = *bits_word;
                let mut pos0 = self.word_base0[w];
                let mut pos1 = self.word_base1[w];
                let block_end = (idx + 64).min(self.n_haps);
                while idx < block_end {
                    let hap = prefix[idx];
                    let div = divergence[idx];
                    if div > p0 {
                        p0 = div;
                    }
                    if div > p1 {
                        p1 = div;
                    }
                    if (bits & 1) == 0 {
                        self.scratch_a[pos0] = hap;
                        self.scratch_d[pos0] = p0;
                        p0 = i32::MIN;
                        pos0 += 1;
                    } else {
                        self.scratch_a[pos1] = hap;
                        self.scratch_d[pos1] = p1;
                        p1 = i32::MIN;
                        pos1 += 1;
                    }
                    bits >>= 1;
                    idx += 1;
                }
            }
        } else {
            let mut p0 = init_value;
            let mut p1 = init_value;
            let mut p_miss = init_value;
            let mut idx = 0usize;
            for (w, bits_word) in permuted_bits.iter().enumerate().take(n_words) {
                let mut bits = *bits_word;
                let mut miss = permuted_missing_bits[w];
                let mut pos0 = self.word_base0[w];
                let mut pos1 = self.word_base1[w];
                let mut pos_miss = self.word_base_miss[w];
                let block_end = (idx + 64).min(self.n_haps);
                while idx < block_end {
                    let hap = prefix[idx];
                    let div = divergence[idx];
                    if div > p0 {
                        p0 = div;
                    }
                    if div > p1 {
                        p1 = div;
                    }
                    if div > p_miss {
                        p_miss = div;
                    }
                    if (miss & 1) != 0 {
                        self.scratch_a[pos_miss] = hap;
                        self.scratch_d[pos_miss] = p_miss;
                        p_miss = i32::MIN;
                        pos_miss += 1;
                    } else if (bits & 1) == 0 {
                        self.scratch_a[pos0] = hap;
                        self.scratch_d[pos0] = p0;
                        p0 = i32::MIN;
                        pos0 += 1;
                    } else {
                        self.scratch_a[pos1] = hap;
                        self.scratch_d[pos1] = p1;
                        p1 = i32::MIN;
                        pos1 += 1;
                    }
                    bits >>= 1;
                    miss >>= 1;
                    idx += 1;
                }
            }
        }

        prefix.copy_from_slice(&self.scratch_a[..self.n_haps]);
        divergence[..self.n_haps].copy_from_slice(&self.scratch_d[..self.n_haps]);
    }

    /// Backward update of prefix and divergence arrays
    ///
    /// Uses In-Place Counting Sort.
    ///
    /// # Backward PBWT Semantics (vs Forward)
    ///
    /// - **Forward PBWT** (markers 0 → M-1): divergence[i] = marker where match STARTS
    ///   - Small divergence = long match (started far back)
    ///   - Uses MAX propagation: latest divergence point limits the match
    ///   - Reset to MIN_VALUE after output
    ///
    /// - **Backward PBWT** (markers M-1 → 0): divergence[i] = marker where match ENDS
    ///   - Small divergence = short match (ends soon)
    ///   - Uses MIN propagation: earliest end point limits the match
    ///   - Reset to MAX_VALUE after output
    ///
    /// # Missing Data Handling
    ///
    /// Missing data (allele u8::MAX, or invalid allele >= n_alleles) is placed in
    /// its own bin at index 1. This keeps missing separate from both REF (0)
    /// and ALT bins.
    ///
    /// This matches the Java Beagle implementation in PbwtDivUpdater.bwdUpdate.
    pub fn bwd_update(
        &mut self,
        alleles: &[u8],
        n_alleles: usize,
        marker: usize,
        prefix: &mut [I],
        divergence: &mut [i32],
    ) {
        assert_eq!(alleles.len(), self.n_haps);
        assert_eq!(prefix.len(), self.n_haps);
        assert!(divergence.len() >= self.n_haps);
        let alphabet = PbwtAlphabet::new(n_alleles).unwrap_or_else(|| {
            panic!(
                "PBWT invalid n_alleles={} (expected 2..=u8::MAX with u8::MAX reserved for missing)",
                n_alleles
            )
        });

        // Use n_alleles + 1 bins with Bin 1 reserved for missing/invalid.
        let n_bins = alphabet.n_bins();
        self.ensure_capacity(n_bins);

        // 1. Initialize p array for backward PBWT
        //
        // Java uses marker-1 for initialization. This correctly handles allele boundaries:
        // when two adjacent haplotypes in sorted order have different alleles at marker m,
        // their match must end at m-1 (they differ at m). Using i32::MAX would incorrectly
        // suggest they match indefinitely.
        let init_value = (marker as i32) - 1;

        // 2. Count frequencies
        self.counts[..n_bins].fill(0);

        if n_alleles == 2 {
            let (count1, count_miss) = self.pack_biallelic_bits(alleles, prefix);
            self.counts[0] = self.n_haps - count1 - count_miss;
            self.counts[1] = count_miss; // Bin 1: Missing
            self.counts[2] = count1; // Bin 2: Allele 1
        } else {
            // General path for multiallelic
            // 0. Pre-permute alleles: gather alleles[prefix[i]] into contiguous buffer.
            // This converts subsequent random-access patterns to sequential access.
            for i in 0..self.n_haps {
                self.permuted_alleles[i] = alleles[prefix[i].to_usize()];
            }

            for i in 0..self.n_haps {
                let allele = self.permuted_alleles[i] as usize;
                let bin = pbwt_bin_for_allele(allele as u8, alphabet);
                self.counts[bin] += 1;
            }
        }

        // 3. Compute Offsets
        let mut running = 0;
        for i in 0..n_bins {
            self.offsets[i] = running;
            running += self.counts[i];
        }

        // 4. Check if there's any missing data (before resetting counts)
        let has_missing = self.counts[1] > 0;
        if n_alleles == 2 {
            self.compute_biallelic_word_bases(has_missing);
        }

        // 5. Scatter with MIN propagation for backward PBWT
        // p[j] tracks the minimum divergence seen since last output for allele j
        // Now uses permuted_alleles for sequential access
        self.counts[..n_bins].fill(0);
        self.p[..n_bins].fill(init_value);

        if n_alleles == 2 && !has_missing {
            // Fast biallelic path when no missing data - only 2 bins needed
            let mut p0 = init_value;
            let mut p1 = init_value;

            let mut idx = 0usize;
            let n_words = (self.n_haps + 63) / 64;
            for w in 0..n_words {
                let mut bits = self.permuted_bits[w];
                let mut pos0 = self.word_base0[w];
                let mut pos1 = self.word_base1[w];
                let block_end = (idx + 64).min(self.n_haps);
                while idx < block_end {
                    let hap = prefix[idx];
                    let div = divergence[idx];

                    // Propagate min to both bins (backward PBWT)
                    if div < p0 {
                        p0 = div;
                    }
                    if div < p1 {
                        p1 = div;
                    }

                    if (bits & 1) == 0 {
                        self.scratch_a[pos0] = hap;
                        self.scratch_d[pos0] = p0;
                        p0 = i32::MAX;
                        pos0 += 1;
                    } else {
                        self.scratch_a[pos1] = hap;
                        self.scratch_d[pos1] = p1;
                        p1 = i32::MAX;
                        pos1 += 1;
                    }

                    bits >>= 1;
                    idx += 1;
                }
            }
        } else if n_alleles == 2 {
            // Biallelic path with missing data - needs 3 bins
            let mut p0 = init_value;
            let mut p1 = init_value;
            let mut p_miss = init_value;

            let mut idx = 0usize;
            let n_words = (self.n_haps + 63) / 64;
            for w in 0..n_words {
                let mut bits = self.permuted_bits[w];
                let mut miss = self.permuted_missing_bits[w];
                let mut pos0 = self.word_base0[w];
                let mut pos1 = self.word_base1[w];
                let mut pos_miss = self.word_base_miss[w];
                let block_end = (idx + 64).min(self.n_haps);
                while idx < block_end {
                    let hap = prefix[idx];
                    let div = divergence[idx];

                    // Propagate min to all bins (backward PBWT)
                    if div < p0 {
                        p0 = div;
                    }
                    if div < p1 {
                        p1 = div;
                    }
                    if div < p_miss {
                        p_miss = div;
                    }

                    if (miss & 1) != 0 {
                        // Missing or invalid allele
                        self.scratch_a[pos_miss] = hap;
                        self.scratch_d[pos_miss] = p_miss;
                        p_miss = i32::MAX;
                        pos_miss += 1;
                    } else if (bits & 1) == 0 {
                        self.scratch_a[pos0] = hap;
                        self.scratch_d[pos0] = p0;
                        p0 = i32::MAX;
                        pos0 += 1;
                    } else {
                        self.scratch_a[pos1] = hap;
                        self.scratch_d[pos1] = p1;
                        p1 = i32::MAX;
                        pos1 += 1;
                    }

                    bits >>= 1;
                    miss >>= 1;
                    idx += 1;
                }
            }
        } else {
            // General multiallelic path
            let prefetch_stride = 64usize;
            for i in 0..self.n_haps {
                if i + prefetch_stride < self.n_haps {
                    unsafe {
                        prefetch_read(self.permuted_alleles.as_ptr().add(i + prefetch_stride));
                        prefetch_read(prefix.as_ptr().add(i + prefetch_stride));
                        prefetch_read(divergence.as_ptr().add(i + prefetch_stride));
                    }
                }
                let hap = prefix[i];
                let div = divergence[i];
                let allele = self.permuted_alleles[i] as usize; // Sequential access
                let bin = pbwt_bin_for_allele(allele as u8, alphabet);

                // Update p: min(p, div) for backward PBWT
                // Smaller divergence = earlier end point = shorter match
                // We propagate the minimum to find the "worst case" match length
                for j in 0..n_bins {
                    if div < self.p[j] {
                        self.p[j] = div;
                    }
                }

                let base = self.offsets[bin];
                let offset = self.counts[bin];
                let pos = base + offset;

                self.scratch_a[pos] = hap;
                self.scratch_d[pos] = self.p[bin];

                // Reset to MAX so next haplotype takes its own divergence
                self.p[bin] = i32::MAX;
                self.counts[bin] += 1;
            }
        }

        // 6. Copy back
        prefix.copy_from_slice(&self.scratch_a[..self.n_haps]);
        divergence[..self.n_haps].copy_from_slice(&self.scratch_d[..self.n_haps]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pbwt_div_fwd_update() {
        let mut updater = PbwtDivUpdater::new(4);
        let mut prefix: Vec<u32> = vec![0, 1, 2, 3];
        let mut divergence: Vec<i32> = vec![0, 0, 0, 0];

        let alleles = vec![0u8, 1, 0, 1];
        updater.fwd_update(&alleles, 2, 0, &mut prefix, &mut divergence);

        // Check grouping: haps with allele 0 first (0, 2), then allele 1 (1, 3)
        assert_eq!(prefix, vec![0, 2, 1, 3]);

        // For forward PBWT at marker 0 with initial div=0 for all:
        // - p initialized to marker+1=1
        // - Hap 0 (allele 0): div=0 <= p[0]=1, so p unchanged. Store d=p[0]=1, reset p[0]=MIN
        // - Hap 1 (allele 1): div=0 > MIN, so p[0]=0. div=0 <= p[1]=1. Store d=p[1]=1, reset p[1]=MIN
        // - Hap 2 (allele 0): div=0 > MIN for both. p=[0,0]. Store d=p[0]=0, reset p[0]=MIN
        // - Hap 3 (allele 1): div=0 > MIN for both. p=[0,0]. Store d=p[1]=0, reset p[1]=MIN
        // Result: d[allele 0]=[1,0], d[allele 1]=[1,0]
        // Final divergence = [1, 0, 1, 0]
        assert_eq!(divergence[0], 1); // Hap 0, first with allele 0
        assert_eq!(divergence[1], 0); // Hap 2, second with allele 0
        assert_eq!(divergence[2], 1); // Hap 1, first with allele 1
        assert_eq!(divergence[3], 0); // Hap 3, second with allele 1
    }

    /// Test PBWT divergence propagation across multiple markers.
    /// This tests the CRITIC's claim that "PBWT forgets history at every step".
    /// If that were true, divergence would reset to marker+1 at each step,
    /// and we couldn't find matches longer than 1 marker.
    #[test]
    fn test_pbwt_multi_marker_divergence_propagation() {
        let mut updater = PbwtDivUpdater::new(4);
        let mut prefix: Vec<u32> = vec![0, 1, 2, 3];
        let mut divergence: Vec<i32> = vec![0, 0, 0, 0]; // Match started at marker 0

        // Haplotypes:
        // Hap 0: [0, 0, 0]  - all allele 0
        // Hap 1: [0, 0, 0]  - all allele 0 (matches hap 0)
        // Hap 2: [1, 1, 1]  - all allele 1
        // Hap 3: [1, 1, 1]  - all allele 1 (matches hap 2)

        // Marker 0
        let alleles_m0 = vec![0u8, 0, 1, 1];
        updater.fwd_update(&alleles_m0, 2, 0, &mut prefix, &mut divergence);
        // After m0: prefix=[0,1,2,3], divergence=[1,0,1,0]
        // Haps 0,1 grouped (allele 0), 2,3 grouped (allele 1)

        // Marker 1
        let alleles_m1 = vec![0u8, 0, 1, 1];
        updater.fwd_update(&alleles_m1, 2, 1, &mut prefix, &mut divergence);
        // After m1: same grouping, divergence should propagate
        // If CRITIC were right, divergence would be [2,2,2,2] (marker+1)
        // But with correct propagation, hap 1's divergence carries forward from m0

        // Marker 2
        let alleles_m2 = vec![0u8, 0, 1, 1];
        updater.fwd_update(&alleles_m2, 2, 2, &mut prefix, &mut divergence);

        // Key assertion: If divergence propagates correctly, hap 1 should have
        // divergence value propagated from marker 0 (when it was first grouped with hap 0).
        // If CRITIC were right, all divergences would be marker+1 = 3.

        // The second hap in each allele group should have LOW divergence (0 or 1),
        // indicating a match that started early.
        assert!(
            divergence[1] < 3,
            "PBWT divergence NOT propagating! Second hap in group has div={}, expected < 3",
            divergence[1]
        );
        assert!(
            divergence[3] < 3,
            "PBWT divergence NOT propagating! Fourth hap in group has div={}, expected < 3",
            divergence[3]
        );

        // First hap in each group can have marker+1 (no predecessor with same allele yet)
        // That's expected behavior, not a bug.
    }

    /// Test that matches are detected when haplotypes share a long identical segment.
    /// This tests the core PBWT functionality for finding IBS matches.
    #[test]
    fn test_pbwt_long_match_detection() {
        let mut updater = PbwtDivUpdater::new(4);
        let mut prefix: Vec<u32> = vec![0, 1, 2, 3];
        let mut divergence: Vec<i32> = vec![0, 0, 0, 0];

        // Haplotypes sharing a 5-marker segment:
        // Hap 0: [0, 0, 0, 0, 0]
        // Hap 1: [0, 0, 0, 0, 0]  - matches hap 0 from start
        // Hap 2: [1, 0, 0, 0, 0]  - matches haps 0,1 from marker 1
        // Hap 3: [1, 1, 0, 0, 0]  - matches haps 0,1,2 from marker 2

        for marker in 0..5 {
            let alleles = vec![
                0u8,                            // hap 0: always 0
                0,                              // hap 1: always 0
                if marker < 1 { 1 } else { 0 }, // hap 2: 1 at m0
                if marker < 2 { 1 } else { 0 }, // hap 3: 1 at m0,m1
            ];
            updater.fwd_update(&alleles, 2, marker, &mut prefix, &mut divergence);
        }

        // After marker 4:
        // Haps 0,1 have been together since marker 0 -> one should have low divergence
        // Hap 2 joined allele-0 group at marker 1 -> divergence should be ~1
        // Hap 3 joined allele-0 group at marker 2 -> divergence should be ~2

        // Find hap 1 in the sorted prefix array
        let hap1_pos = prefix.iter().position(|&h| h == 1).unwrap();
        let hap1_div = divergence[hap1_pos];

        // Hap 1 should have a match starting early (div close to 0 or 1)
        assert!(
            hap1_div <= 2,
            "Hap 1 should have long match with hap 0, but divergence is {} (expected <= 2)",
            hap1_div
        );
    }

    /// Test that missing data (u8::MAX) is handled correctly without reference bias.
    ///
    /// This tests the fix for the CRITIC-identified bug where missing data was
    /// being mapped to REF (allele 0), creating systematic reference bias.
    /// Missing data should be placed in its own bin, not grouped with REF or ALT.
    #[test]
    fn test_pbwt_missing_data_no_reference_bias() {
        let mut updater = PbwtDivUpdater::new(4);
        let mut prefix: Vec<u32> = vec![0, 1, 2, 3];
        let mut divergence: Vec<i32> = vec![0, 0, 0, 0];

        // Haplotypes:
        // Hap 0: REF (0)
        // Hap 1: ALT (1)
        // Hap 2: MISSING (u8::MAX)
        // Hap 3: REF (0)
        let alleles = vec![0u8, 1, u8::MAX, 0];
        updater.fwd_update(&alleles, 2, 0, &mut prefix, &mut divergence);

        // With the fix: missing (u8::MAX) goes to Bin 1 (between REF and ALT)
        // Sorted order should be: [REF haps (0,3), MISSING hap (2), ALT hap (1)]
        // This ensures the missing haplotype has access to neighbors from BOTH Ref and Alt groups.

        let hap0_pos = prefix.iter().position(|&h| h == 0).unwrap();
        let hap1_pos = prefix.iter().position(|&h| h == 1).unwrap();
        let hap2_pos = prefix.iter().position(|&h| h == 2).unwrap();
        let hap3_pos = prefix.iter().position(|&h| h == 3).unwrap();

        // 1. REF haps (0 and 3) should be adjacent (grouped together)
        assert!(
            (hap0_pos as i32 - hap3_pos as i32).abs() == 1,
            "REF haps 0 and 3 should be adjacent in PBWT. Positions: hap0={}, hap3={}",
            hap0_pos,
            hap3_pos
        );

        // 2. MISSING hap (2) should be positioned BETWEEN Ref and Alt
        // Max Ref position < Missing Position < Alt Position
        let max_ref_pos = std::cmp::max(hap0_pos, hap3_pos);

        assert!(
            max_ref_pos < hap2_pos,
            "MISSING hap 2 should come AFTER Ref haps. Positions: max_ref={}, hap2={}",
            max_ref_pos,
            hap2_pos
        );

        assert!(
            hap2_pos < hap1_pos,
            "MISSING hap 2 should come BEFORE Alt hap 1. Positions: hap2={}, hap1={}",
            hap2_pos,
            hap1_pos
        );
    }

    /// Test backward PBWT also handles missing data correctly.
    #[test]
    fn test_pbwt_bwd_missing_data_no_reference_bias() {
        let mut updater = PbwtDivUpdater::new(4);
        let mut prefix: Vec<u32> = vec![0, 1, 2, 3];
        let mut divergence: Vec<i32> = vec![10, 10, 10, 10]; // High initial values for backward

        // Haplotypes:
        // Hap 0: REF (0)
        // Hap 1: ALT (1)
        // Hap 2: MISSING (u8::MAX)
        // Hap 3: REF (0)
        let alleles = vec![0u8, 1, u8::MAX, 0];
        let marker = 5; // Use marker 5 so init_value = 4
        updater.bwd_update(&alleles, 2, marker, &mut prefix, &mut divergence);

        // Same logic as forward: MISSING should be in Bin 1 (between REF and ALT)
        let hap0_pos = prefix.iter().position(|&h| h == 0).unwrap();
        let hap1_pos = prefix.iter().position(|&h| h == 1).unwrap();
        let hap2_pos = prefix.iter().position(|&h| h == 2).unwrap();
        let hap3_pos = prefix.iter().position(|&h| h == 3).unwrap();

        // REF haps (0 and 3) should be adjacent
        assert!(
            (hap0_pos as i32 - hap3_pos as i32).abs() == 1,
            "REF haps 0 and 3 should be adjacent in backward PBWT. Positions: hap0={}, hap3={}",
            hap0_pos,
            hap3_pos
        );

        // MISSING hap (2) should be positioned BETWEEN Ref and Alt
        let max_ref_pos = std::cmp::max(hap0_pos, hap3_pos);

        assert!(
            max_ref_pos < hap2_pos,
            "MISSING hap 2 should come AFTER Ref haps. Positions: max_ref={}, hap2={}",
            max_ref_pos,
            hap2_pos
        );

        assert!(
            hap2_pos < hap1_pos,
            "MISSING hap 2 should come BEFORE Alt hap 1. Positions: hap2={}, hap1={}",
            hap2_pos,
            hap1_pos
        );
    }
}
