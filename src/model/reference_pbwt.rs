use crate::model::pbwt::{PbwtDivUpdater, PbwtIndex};

const MAX_RANK_INTERVALS: usize = 8;

#[derive(Clone, Copy, Debug)]
pub struct RankBeam {
    intervals: [(u32, u32); MAX_RANK_INTERVALS],
    len: usize,
}

impl RankBeam {
    pub fn full(n_ref_haps: u32) -> Self {
        let mut beam = Self {
            intervals: [(0, 0); MAX_RANK_INTERVALS],
            len: 1,
        };
        beam.intervals[0] = (0, n_ref_haps);
        beam
    }

    pub fn intervals(&self) -> &[(u32, u32)] {
        &self.intervals[..self.len]
    }

    fn push_interval(&mut self, l: u32, r: u32) {
        if l >= r {
            return;
        }
        if self.len < MAX_RANK_INTERVALS {
            self.intervals[self.len] = (l, r);
            self.len += 1;
        } else {
            let last = self.intervals[MAX_RANK_INTERVALS - 1];
            self.intervals[MAX_RANK_INTERVALS - 1] = (last.0.min(l), last.1.max(r));
        }
    }

    fn normalize(&mut self) {
        if self.len <= 1 {
            return;
        }
        let mut tmp = self.intervals;
        tmp[..self.len].sort_unstable_by_key(|(l, _)| *l);
        let mut out_len = 0usize;
        for i in 0..self.len {
            let (l, r) = tmp[i];
            if out_len == 0 {
                tmp[out_len] = (l, r);
                out_len = 1;
                continue;
            }
            let prev = tmp[out_len - 1];
            if l <= prev.1 {
                tmp[out_len - 1] = (prev.0, prev.1.max(r));
            } else {
                tmp[out_len] = (l, r);
                out_len += 1;
                if out_len >= MAX_RANK_INTERVALS {
                    break;
                }
            }
        }
        self.intervals = tmp;
        self.len = out_len;
    }
}

pub struct ReferencePbwtImpl<I: PbwtIndex> {
    updater: PbwtDivUpdater<I>,
    ppa: Vec<I>,
    div: Vec<i32>,
    permuted_ref: Vec<u8>,
    permuted_bits: Vec<u64>,
    permuted_missing_bits: Vec<u64>,
    prefix_ones_words: Vec<u32>,
    prefix_missing_words: Vec<u32>,
    binary_counts: [u32; 3],
    binary_offsets: [u32; 3],
    prefix_counts: Vec<u32>,
    counts: Vec<u32>,
    offsets: Vec<u32>,
    intervals_buf: Vec<(usize, usize)>,
}

#[repr(transparent)]
#[derive(Copy, Clone, Eq, PartialEq)]
pub struct PbwtQueryAllele(u8);

#[repr(transparent)]
#[derive(Copy, Clone, Eq, PartialEq)]
pub struct PbwtStrictAllele(u8);

impl PbwtQueryAllele {
    pub const WILDCARD_VALUE: u8 = 128;

    pub fn allele(a: u8) -> Option<Self> {
        if a <= 1 {
            Some(Self(a))
        } else {
            None
        }
    }

    pub fn missing() -> Self {
        Self(255)
    }

    pub fn wildcard() -> Self {
        Self(Self::WILDCARD_VALUE)
    }

    #[inline]
    pub fn value(self) -> u8 {
        self.0
    }
}

impl PbwtStrictAllele {
    pub fn allele(a: u8) -> Option<Self> {
        if a <= 1 {
            Some(Self(a))
        } else {
            None
        }
    }

    pub fn missing() -> Self {
        Self(255)
    }

    #[inline]
    pub fn value(self) -> u8 {
        self.0
    }
}

impl<I: PbwtIndex> ReferencePbwtImpl<I> {
    pub fn new(n_ref_haps: usize) -> Self {
        Self {
            updater: PbwtDivUpdater::new(n_ref_haps),
            ppa: (0..n_ref_haps).map(I::from_usize).collect(),
            div: vec![0; n_ref_haps],
            permuted_ref: vec![0; n_ref_haps],
            permuted_bits: Vec::new(),
            permuted_missing_bits: Vec::new(),
            prefix_ones_words: Vec::new(),
            prefix_missing_words: Vec::new(),
            binary_counts: [0; 3],
            binary_offsets: [0; 3],
            prefix_counts: Vec::new(),
            counts: Vec::new(),
            offsets: Vec::new(),
            intervals_buf: Vec::new(),
        }
    }

    pub fn select_donors_into(&mut self, beam: &RankBeam, k: usize, out: &mut Vec<u32>) {
        out.clear();
        if k == 0 {
            return;
        }
        let n_ref = self.ppa.len();
        if n_ref == 0 {
            return;
        }

        // Uniform spaced sampling from all intervals to ensure coverage of the
        // entire PBWT space, avoiding bias towards the center or beginning.
        // We treat the intervals as a single contiguous range and sample uniformly.
        self.intervals_buf.clear();
        self.intervals_buf.reserve(beam.intervals().len());
        for &(l, r) in beam.intervals() {
            let l = l.min(n_ref as u32) as usize;
            let r = r.min(n_ref as u32) as usize;
            if l < r {
                self.intervals_buf.push((l, r));
            }
        }

        if self.intervals_buf.is_empty() {
            return;
        }

        let total_len: usize = self.intervals_buf.iter().map(|&(l, r)| r - l).sum();

        if total_len <= k {
            out.reserve(total_len);
            for &(l, r) in &self.intervals_buf {
                for i in l..r {
                    out.push(self.ppa[i].to_u32());
                }
            }
        } else {
            // Centered uniform spaced sampling
            // i-th sample at: (2*i + 1) * total_len / (2*k)
            let mut current_interval_idx = 0;
            let mut current_interval_start_offset = 0;

            out.reserve(k);
            for i in 0..k {
                let target = (2 * i + 1) * total_len / (2 * k);

                // Advance to interval containing target
                while current_interval_idx < self.intervals_buf.len() {
                    let (l, r) = self.intervals_buf[current_interval_idx];
                    let len = r - l;
                    if target < current_interval_start_offset + len {
                        let offset_in_interval = target - current_interval_start_offset;
                        out.push(self.ppa[l + offset_in_interval].to_u32());
                        break;
                    }
                    current_interval_start_offset += len;
                    current_interval_idx += 1;
                }
            }
        }
    }

    fn bin_for_allele(a: u8, n_alleles: usize) -> usize {
        if n_alleles == 2 {
            if a == 0 {
                0
            } else if a == 1 {
                2
            } else {
                1
            }
        } else if a == 0 {
            0
        } else if (a as usize) >= n_alleles {
            1
        } else {
            (a as usize) + 1
        }
    }

    fn ensure_buffers(&mut self, n_bins: usize) {
        let n_ref = self.ppa.len();
        let needed = n_bins * (n_ref + 1);
        if self.prefix_counts.len() < needed {
            self.prefix_counts.resize(needed, 0);
        }
        if self.counts.len() < n_bins {
            self.counts.resize(n_bins, 0);
        }
        if self.offsets.len() < n_bins {
            self.offsets.resize(n_bins, 0);
        }
    }

    fn ensure_bit_buffers(&mut self) {
        let n_ref = self.ppa.len();
        let n_words = (n_ref + 63) / 64;
        if self.permuted_bits.len() < n_words {
            self.permuted_bits.resize(n_words, 0);
            self.permuted_missing_bits.resize(n_words, 0);
        }
        if self.prefix_ones_words.len() < n_words + 1 {
            self.prefix_ones_words.resize(n_words + 1, 0);
            self.prefix_missing_words.resize(n_words + 1, 0);
        }
    }

    fn offset_for(&self, bin: usize, n_alleles: usize) -> u32 {
        if n_alleles == 2 {
            self.binary_offsets[bin]
        } else {
            self.offsets[bin]
        }
    }

    fn count_for(&self, bin: usize, n_alleles: usize) -> u32 {
        if n_alleles == 2 {
            self.binary_counts[bin]
        } else {
            self.counts[bin]
        }
    }

    fn prefix_idx(bin: usize, pos: usize, n_ref: usize) -> usize {
        bin * (n_ref + 1) + pos
    }

    fn rank(&self, bin: usize, pos: u32, n_ref: usize, n_alleles: usize) -> u32 {
        if n_alleles == 2 {
            let p = pos.min(n_ref as u32) as usize;
            let n_words = self.permuted_bits.len();
            let word = p / 64;
            let bit = p % 64;
            let mask = if bit == 0 { 0 } else { (1u64 << bit) - 1 };
            let base_word = word.min(n_words.saturating_sub(1));
            let ones_base = self.prefix_ones_words[word.min(n_words)];
            let missing_base = self.prefix_missing_words[word.min(n_words)];
            let ones = if word >= n_words {
                ones_base
            } else {
                ones_base + (self.permuted_bits[base_word] & mask).count_ones()
            };
            let missing = if word >= n_words {
                missing_base
            } else {
                missing_base + (self.permuted_missing_bits[base_word] & mask).count_ones()
            };
            let zeros = p as u32 - ones - missing;
            return match bin {
                0 => zeros,
                1 => missing,
                _ => ones,
            };
        }
        let p = pos.min(n_ref as u32) as usize;
        self.prefix_counts[Self::prefix_idx(bin, p, n_ref)]
    }

    pub fn advance_with_beams_query(
        &mut self,
        ref_alleles: &[u8],
        n_alleles: usize,
        marker: usize,
        query_alleles: &[PbwtQueryAllele],
        beams: &mut [RankBeam],
    ) {
        let mut scratch: Vec<(u32, u32, u32)> = Vec::new();
        self.advance_with_beams_query_scratch(
            ref_alleles,
            n_alleles,
            marker,
            query_alleles,
            beams,
            &mut scratch,
        );
    }

    pub fn advance_with_beams_strict(
        &mut self,
        ref_alleles: &[u8],
        n_alleles: usize,
        marker: usize,
        query_alleles: &[PbwtStrictAllele],
        beams: &mut [RankBeam],
    ) {
        let mut scratch: Vec<(u32, u32, u32)> = Vec::new();
        self.advance_with_beams_strict_scratch(
            ref_alleles,
            n_alleles,
            marker,
            query_alleles,
            beams,
            &mut scratch,
        );
    }


    pub fn advance_with_beams_query_scratch(
        &mut self,
        ref_alleles: &[u8],
        n_alleles: usize,
        marker: usize,
        query_alleles: &[PbwtQueryAllele],
        beams: &mut [RankBeam],
        scratch: &mut Vec<(u32, u32, u32)>,
    ) {
        self.prepare_step(ref_alleles, n_alleles);
        self.update_beams_with_scratch_query(beams, query_alleles, n_alleles, scratch);
        self.finalize_step(ref_alleles, n_alleles, marker);
    }

    pub fn advance_with_beams_strict_scratch(
        &mut self,
        ref_alleles: &[u8],
        n_alleles: usize,
        marker: usize,
        query_alleles: &[PbwtStrictAllele],
        beams: &mut [RankBeam],
        scratch: &mut Vec<(u32, u32, u32)>,
    ) {
        self.prepare_step(ref_alleles, n_alleles);
        self.update_beams_with_scratch_strict(beams, query_alleles, n_alleles, scratch);
        self.finalize_step(ref_alleles, n_alleles, marker);
    }

    pub fn prepare_step(&mut self, ref_alleles: &[u8], n_alleles: usize) {
        let n_ref = self.ppa.len();
        let n_bins = if n_alleles == 2 { 3 } else { n_alleles + 1 };
        if n_alleles == 2 {
            self.ensure_bit_buffers();
            let n_words = (n_ref + 63) / 64;
            let mut count1 = 0u32;
            let mut count_miss = 0u32;
            let mut idx = 0usize;

            for w in 0..n_words {
                let mut bits = 0u64;
                let mut miss = 0u64;
                let block_end = (idx + 64).min(n_ref);
                let mut bit = 0u64;
                while idx < block_end {
                    let allele = ref_alleles[self.ppa[idx].to_usize()];
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
                count1 += bits.count_ones();
                count_miss += miss.count_ones();
            }

            let count0 = n_ref as u32 - count1 - count_miss;
            self.binary_counts[0] = count0;
            self.binary_counts[1] = count_miss;
            self.binary_counts[2] = count1;

            let mut running = 0u32;
            for b in 0..n_bins {
                self.binary_offsets[b] = running;
                running += self.binary_counts[b];
            }

            self.prefix_ones_words[0] = 0;
            self.prefix_missing_words[0] = 0;
            for w in 0..n_words {
                self.prefix_ones_words[w + 1] =
                    self.prefix_ones_words[w] + self.permuted_bits[w].count_ones();
                self.prefix_missing_words[w + 1] =
                    self.prefix_missing_words[w] + self.permuted_missing_bits[w].count_ones();
            }
        } else {
            self.ensure_buffers(n_bins);

            for i in 0..n_ref {
                self.permuted_ref[i] = ref_alleles[self.ppa[i].to_usize()];
            }

            self.counts[..n_bins].fill(0);
            for &a in &self.permuted_ref {
                let b = Self::bin_for_allele(a, n_alleles);
                if b < n_bins {
                    self.counts[b] += 1;
                }
            }

            let mut running = 0u32;
            for b in 0..n_bins {
                self.offsets[b] = running;
                running += self.counts[b];
            }

            for b in 0..n_bins {
                let mut c = 0u32;
                let base = b * (n_ref + 1);
                self.prefix_counts[base] = 0;
                for i in 0..n_ref {
                    let a = self.permuted_ref[i];
                    let bin = Self::bin_for_allele(a, n_alleles);
                    if bin == b {
                        c += 1;
                    }
                    self.prefix_counts[base + i + 1] = c;
                }
            }
        }
    }


    pub fn update_beams_with_scratch_query(
        &self,
        beams: &mut [RankBeam],
        query_alleles: &[PbwtQueryAllele],
        n_alleles: usize,
        scratch: &mut Vec<(u32, u32, u32)>,
    ) {
        let n_ref = self.ppa.len();
        let n_bins = if n_alleles == 2 { 3 } else { n_alleles + 1 };

        for (q_idx, &qa) in query_alleles.iter().enumerate() {
            if q_idx >= beams.len() {
                break;
            }
            let qa = qa.value();
            let old = beams[q_idx];
            let mut next = RankBeam {
                intervals: [(0, 0); MAX_RANK_INTERVALS],
                len: 0,
            };

            if qa == PbwtQueryAllele::WILDCARD_VALUE {
                scratch.clear();
                for &(l, r) in old.intervals() {
                    for &b in &[0usize, 2usize] {
                        let nl = self.offset_for(b, n_alleles) + self.rank(b, l, n_ref, n_alleles);
                        let nr = self.offset_for(b, n_alleles) + self.rank(b, r, n_ref, n_alleles);
                        if nl < nr {
                            let len = nr - nl;
                            let score = len.saturating_mul(self.count_for(b, n_alleles));
                            scratch.push((nl, nr, score));
                        }
                    }
                }

                scratch.sort_unstable_by(|a, b| b.2.cmp(&a.2));
                let keep = scratch.len().min(MAX_RANK_INTERVALS);
                for i in 0..keep {
                    next.intervals[i] = (scratch[i].0, scratch[i].1);
                }
                next.len = keep;
            } else if qa == 255 {
                scratch.clear();
                for &(l, r) in old.intervals() {
                    for b in 0..n_bins {
                        let nl = self.offset_for(b, n_alleles) + self.rank(b, l, n_ref, n_alleles);
                        let nr = self.offset_for(b, n_alleles) + self.rank(b, r, n_ref, n_alleles);
                        if nl < nr {
                            let len = nr - nl;
                            let score = len.saturating_mul(self.count_for(b, n_alleles));
                            scratch.push((nl, nr, score));
                        }
                    }
                }

                scratch.sort_unstable_by(|a, b| b.2.cmp(&a.2));
                let keep = scratch.len().min(MAX_RANK_INTERVALS);
                for i in 0..keep {
                    next.intervals[i] = (scratch[i].0, scratch[i].1);
                }
                next.len = keep;
            } else {
                let b = Self::bin_for_allele(qa, n_alleles);
                if b < n_bins {
                    for &(l, r) in old.intervals() {
                        let nl = self.offset_for(b, n_alleles) + self.rank(b, l, n_ref, n_alleles);
                        let nr = self.offset_for(b, n_alleles) + self.rank(b, r, n_ref, n_alleles);
                        next.push_interval(nl, nr);
                    }
                } else {
                    next = RankBeam::full(n_ref as u32);
                }

                if next.len == 0 {
                    scratch.clear();
                    for &(l, r) in old.intervals() {
                        for b in 0..n_bins {
                            let nl =
                                self.offset_for(b, n_alleles) + self.rank(b, l, n_ref, n_alleles);
                            let nr =
                                self.offset_for(b, n_alleles) + self.rank(b, r, n_ref, n_alleles);
                            if nl < nr {
                                let len = nr - nl;
                                let score = len.saturating_mul(self.count_for(b, n_alleles));
                                scratch.push((nl, nr, score));
                            }
                        }
                    }

                    scratch.sort_unstable_by(|a, b| b.2.cmp(&a.2));
                    let keep = scratch.len().min(MAX_RANK_INTERVALS);
                    for i in 0..keep {
                        next.intervals[i] = (scratch[i].0, scratch[i].1);
                    }
                    next.len = keep;
                }
            }

            next.normalize();
            beams[q_idx] = next;
        }
    }

    pub fn update_beams_with_scratch_strict(
        &self,
        beams: &mut [RankBeam],
        query_alleles: &[PbwtStrictAllele],
        n_alleles: usize,
        scratch: &mut Vec<(u32, u32, u32)>,
    ) {
        let n_ref = self.ppa.len();
        let n_bins = if n_alleles == 2 { 3 } else { n_alleles + 1 };

        for (q_idx, &qa) in query_alleles.iter().enumerate() {
            if q_idx >= beams.len() {
                break;
            }
            let qa = qa.value();
            let old = beams[q_idx];
            let mut next = RankBeam {
                intervals: [(0, 0); MAX_RANK_INTERVALS],
                len: 0,
            };

            if qa == 255 {
                scratch.clear();
                for &(l, r) in old.intervals() {
                    for b in 0..n_bins {
                        let nl = self.offset_for(b, n_alleles) + self.rank(b, l, n_ref, n_alleles);
                        let nr = self.offset_for(b, n_alleles) + self.rank(b, r, n_ref, n_alleles);
                        if nl < nr {
                            let len = nr - nl;
                            let score = len.saturating_mul(self.count_for(b, n_alleles));
                            scratch.push((nl, nr, score));
                        }
                    }
                }

                scratch.sort_unstable_by(|a, b| b.2.cmp(&a.2));
                let keep = scratch.len().min(MAX_RANK_INTERVALS);
                for i in 0..keep {
                    next.intervals[i] = (scratch[i].0, scratch[i].1);
                }
                next.len = keep;
            } else {
                let b = Self::bin_for_allele(qa, n_alleles);
                if b < n_bins {
                    for &(l, r) in old.intervals() {
                        let nl = self.offset_for(b, n_alleles) + self.rank(b, l, n_ref, n_alleles);
                        let nr = self.offset_for(b, n_alleles) + self.rank(b, r, n_ref, n_alleles);
                        next.push_interval(nl, nr);
                    }
                } else {
                    next = RankBeam::full(n_ref as u32);
                }

                if next.len == 0 {
                    scratch.clear();
                    for &(l, r) in old.intervals() {
                        for b in 0..n_bins {
                            let nl =
                                self.offset_for(b, n_alleles) + self.rank(b, l, n_ref, n_alleles);
                            let nr =
                                self.offset_for(b, n_alleles) + self.rank(b, r, n_ref, n_alleles);
                            if nl < nr {
                                let len = nr - nl;
                                let score = len.saturating_mul(self.count_for(b, n_alleles));
                                scratch.push((nl, nr, score));
                            }
                        }
                    }

                    scratch.sort_unstable_by(|a, b| b.2.cmp(&a.2));
                    let keep = scratch.len().min(MAX_RANK_INTERVALS);
                    for i in 0..keep {
                        next.intervals[i] = (scratch[i].0, scratch[i].1);
                    }
                    next.len = keep;
                }
            }

            next.normalize();
            beams[q_idx] = next;
        }
    }

    pub fn finalize_step(&mut self, ref_alleles: &[u8], n_alleles: usize, marker: usize) {
        self.updater
            .fwd_update(ref_alleles, n_alleles, marker, &mut self.ppa, &mut self.div);
    }

}

pub enum ReferencePbwt {
    U16(ReferencePbwtImpl<u16>),
    U32(ReferencePbwtImpl<u32>),
}

impl ReferencePbwt {
    pub fn new(n_ref_haps: usize) -> Self {
        if n_ref_haps <= u16::MAX as usize {
            Self::U16(ReferencePbwtImpl::<u16>::new(n_ref_haps))
        } else {
            Self::U32(ReferencePbwtImpl::<u32>::new(n_ref_haps))
        }
    }

    pub fn select_donors_into(&mut self, beam: &RankBeam, k: usize, out: &mut Vec<u32>) {
        match self {
            Self::U16(inner) => inner.select_donors_into(beam, k, out),
            Self::U32(inner) => inner.select_donors_into(beam, k, out),
        }
    }

    pub fn advance_with_beams_query(
        &mut self,
        ref_alleles: &[u8],
        n_alleles: usize,
        marker: usize,
        query_alleles: &[PbwtQueryAllele],
        beams: &mut [RankBeam],
    ) {
        match self {
            Self::U16(inner) => inner.advance_with_beams_query(
                ref_alleles,
                n_alleles,
                marker,
                query_alleles,
                beams,
            ),
            Self::U32(inner) => inner.advance_with_beams_query(
                ref_alleles,
                n_alleles,
                marker,
                query_alleles,
                beams,
            ),
        }
    }

    pub fn advance_with_beams_strict(
        &mut self,
        ref_alleles: &[u8],
        n_alleles: usize,
        marker: usize,
        query_alleles: &[PbwtStrictAllele],
        beams: &mut [RankBeam],
    ) {
        match self {
            Self::U16(inner) => inner.advance_with_beams_strict(
                ref_alleles,
                n_alleles,
                marker,
                query_alleles,
                beams,
            ),
            Self::U32(inner) => inner.advance_with_beams_strict(
                ref_alleles,
                n_alleles,
                marker,
                query_alleles,
                beams,
            ),
        }
    }

}
