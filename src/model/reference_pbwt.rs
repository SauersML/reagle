use crate::model::pbwt::{PbwtAllele, PbwtAlphabet, PbwtBiallelicBin, PbwtDivUpdater, PbwtIndex};
use std::cmp::Ordering;
use std::collections::HashMap;

const MAX_RANK_INTERVALS: usize = 8;
const PBWT_SCORE_SCALE: u64 = 1_000_000;

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
    donor_candidate_pos: Vec<usize>,
    donor_seen_marks: Vec<u32>,
    donor_seen_tick: u32,
    step_scratch: Vec<(u32, u32, u64)>,
    wanted_map: HashMap<u32, usize>,
    found_pos_start: Vec<(usize, i32)>,
    found_mask: Vec<bool>,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum PbwtQueryAllele {
    Allele(u8),
    Missing,
    Wildcard,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum PbwtStrictAllele {
    Allele(u8),
    Missing,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PbwtBiallelicQueryProb {
    p0: f32,
    p1: f32,
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct DonorChoice {
    div: i32,
    pos: usize,
}

impl Ord for DonorChoice {
    fn cmp(&self, other: &Self) -> Ordering {
        self.div
            .cmp(&other.div)
            .then_with(|| self.pos.cmp(&other.pos))
    }
}

impl PartialOrd for DonorChoice {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PbwtQueryAllele {
    pub const WILDCARD_VALUE: u8 = 128;

    pub fn allele(a: u8) -> Option<Self> {
        if a == Self::WILDCARD_VALUE || a == 255 {
            None
        } else {
            Some(Self::Allele(a))
        }
    }

    pub fn missing() -> Self {
        Self::Missing
    }

    pub fn wildcard() -> Self {
        Self::Wildcard
    }

    /// Returns the concrete allele value, or None for wildcard/missing.
    #[inline]
    pub fn as_allele(self) -> Option<u8> {
        match self {
            Self::Allele(a) => Some(a),
            Self::Missing | Self::Wildcard => None,
        }
    }

    #[inline]
    pub fn is_missing(self) -> bool {
        matches!(self, Self::Missing)
    }

    #[inline]
    pub fn is_wildcard(self) -> bool {
        matches!(self, Self::Wildcard)
    }
}

impl PbwtStrictAllele {
    pub fn allele(a: u8) -> Option<Self> {
        if a == 255 {
            None
        } else {
            Some(Self::Allele(a))
        }
    }

    pub fn missing() -> Self {
        Self::Missing
    }

    #[inline]
    pub fn as_allele(self) -> Option<u8> {
        match self {
            Self::Allele(a) => Some(a),
            Self::Missing => None,
        }
    }

    #[inline]
    pub fn is_missing(self) -> bool {
        matches!(self, Self::Missing)
    }
}

impl PbwtBiallelicQueryProb {
    #[inline]
    pub fn new(p0: f32, p1: f32) -> Self {
        let a = p0.clamp(0.0, 1.0);
        let b = p1.clamp(0.0, 1.0);
        let s = a + b;
        if s <= f32::EPSILON {
            Self { p0: 0.5, p1: 0.5 }
        } else {
            Self {
                p0: a / s,
                p1: b / s,
            }
        }
    }

    #[inline]
    pub fn uniform() -> Self {
        Self { p0: 0.5, p1: 0.5 }
    }

    #[inline]
    pub fn deterministic(allele: u8) -> Self {
        if allele == 0 {
            Self { p0: 1.0, p1: 0.0 }
        } else if allele == 1 {
            Self { p0: 0.0, p1: 1.0 }
        } else {
            Self::uniform()
        }
    }

    #[inline]
    pub fn prob_for_allele(self, allele: u8) -> f32 {
        if allele == 0 {
            self.p0
        } else if allele == 1 {
            self.p1
        } else {
            0.0
        }
    }

    #[inline]
    fn prob_for_bin(self, b: PbwtBiallelicBin) -> f32 {
        if b == PbwtBiallelicBin::Ref {
            self.p0
        } else if b == PbwtBiallelicBin::Alt {
            self.p1
        } else {
            0.0
        }
    }
}

impl<I: PbwtIndex> ReferencePbwtImpl<I> {
    #[inline]
    fn push_top_k_choice(
        best: &mut std::collections::BinaryHeap<DonorChoice>,
        choice: DonorChoice,
        k: usize,
    ) {
        if best.len() < k {
            best.push(choice);
            return;
        }
        if let Some(top) = best.peek().copied() {
            if choice.div < top.div || (choice.div == top.div && choice.pos < top.pos) {
                best.pop();
                best.push(choice);
            }
        }
    }

    #[inline]
    fn flush_top_k_choices(&self, best: std::collections::BinaryHeap<DonorChoice>, out: &mut Vec<u32>) {
        let mut choices: Vec<DonorChoice> = best.into_vec();
        choices.sort_unstable_by(|a, b| a.div.cmp(&b.div).then_with(|| a.pos.cmp(&b.pos)));
        out.reserve(choices.len());
        for c in choices {
            out.push(self.ppa[c.pos].to_u32());
        }
    }

    #[inline]
    fn ensure_donor_seen_marks(&mut self, n_ref: usize) {
        if self.donor_seen_marks.len() < n_ref {
            self.donor_seen_marks.resize(n_ref, 0);
        }
    }

    #[inline]
    fn next_donor_seen_tick(&mut self) -> u32 {
        if self.donor_seen_tick == u32::MAX {
            self.donor_seen_marks.fill(0);
            self.donor_seen_tick = 1;
        } else {
            self.donor_seen_tick += 1;
        }
        self.donor_seen_tick
    }

    #[inline]
    fn load_top_intervals(
        scratch: &mut Vec<(u32, u32, u64)>,
        next: &mut RankBeam,
        keep_cap: usize,
    ) {
        let keep = scratch.len().min(keep_cap);
        if keep == 0 {
            next.len = 0;
            return;
        }
        if scratch.len() > keep {
            scratch.select_nth_unstable_by(keep - 1, |a, b| b.2.cmp(&a.2));
        }
        scratch[..keep].sort_unstable_by(|a, b| b.2.cmp(&a.2));
        for i in 0..keep {
            next.intervals[i] = (scratch[i].0, scratch[i].1);
        }
        next.len = keep;
    }

    #[inline]
    fn scaled_score(len: u32, prob: f32) -> u64 {
        let p = prob.clamp(0.0, 1.0) as f64;
        let weighted = (len as f64) * p * (PBWT_SCORE_SCALE as f64);
        if weighted <= 0.0 {
            0
        } else if weighted >= u64::MAX as f64 {
            u64::MAX
        } else {
            weighted.round() as u64
        }
    }

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
            donor_candidate_pos: Vec::new(),
            donor_seen_marks: Vec::new(),
            donor_seen_tick: 0,
            step_scratch: Vec::new(),
            wanted_map: HashMap::new(),
            found_pos_start: Vec::new(),
            found_mask: Vec::new(),
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
        self.ensure_donor_seen_marks(n_ref);

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

        // Defensive normalization: some callers may provide adjacent/overlapping ranges.
        // Merging here guarantees unique positional coverage for donor selection.
        self.intervals_buf.sort_unstable_by_key(|&(l, _)| l);
        let mut merged_len = 0usize;
        for i in 0..self.intervals_buf.len() {
            let (l, r) = self.intervals_buf[i];
            if merged_len == 0 {
                self.intervals_buf[merged_len] = (l, r);
                merged_len = 1;
                continue;
            }
            let (prev_l, prev_r) = self.intervals_buf[merged_len - 1];
            if l <= prev_r {
                self.intervals_buf[merged_len - 1] = (prev_l, prev_r.max(r));
            } else {
                self.intervals_buf[merged_len] = (l, r);
                merged_len += 1;
            }
        }
        self.intervals_buf.truncate(merged_len);

        let total_len: usize = self.intervals_buf.iter().map(|&(l, r)| r - l).sum();
        if total_len <= k {
            out.reserve(total_len);
            for &(l, r) in &self.intervals_buf {
                for i in l..r {
                    out.push(self.ppa[i].to_u32());
                }
            }
            return;
        }

        // For very wide beams, avoid full scans but still prioritize low-divergence
        // donors rather than uniform positional samples.
        const EXACT_DIV_SCAN_FACTOR: usize = 64;
        if total_len > k.saturating_mul(EXACT_DIV_SCAN_FACTOR) {
            const APPROX_SCAN_FACTOR: usize = 24;
            const MIN_APPROX_SCAN_POINTS: usize = 128;
            const LOCAL_REFINE_RADIUS: usize = 2;

            let n_scan_targets = total_len.min(
                k.saturating_mul(APPROX_SCAN_FACTOR)
                    .max(MIN_APPROX_SCAN_POINTS),
            );
            let candidate_tick = self.next_donor_seen_tick();
            self.donor_candidate_pos.clear();
            self.donor_candidate_pos.reserve(
                n_scan_targets
                    .saturating_mul(2 * LOCAL_REFINE_RADIUS + 1)
                    .saturating_add(self.intervals_buf.len() * 3),
            );

            let mut current_interval_idx = 0usize;
            let mut current_interval_start_offset = 0usize;
            for i in 0..n_scan_targets {
                let target = (2 * i + 1) * total_len / (2 * n_scan_targets);
                while current_interval_idx < self.intervals_buf.len() {
                    let (l, r) = self.intervals_buf[current_interval_idx];
                    let len = r - l;
                    if target < current_interval_start_offset + len {
                        let offset_in_interval = target - current_interval_start_offset;
                        let center = l + offset_in_interval;
                        let start = center.saturating_sub(LOCAL_REFINE_RADIUS).max(l);
                        let end = (center + LOCAL_REFINE_RADIUS + 1).min(r);
                        for pos in start..end {
                            if self.donor_seen_marks[pos] != candidate_tick {
                                self.donor_seen_marks[pos] = candidate_tick;
                                self.donor_candidate_pos.push(pos);
                            }
                        }
                        break;
                    }
                    current_interval_start_offset += len;
                    current_interval_idx += 1;
                }
            }

            // Ensure interval edges and centers are represented.
            for &(l, r) in &self.intervals_buf {
                if l < r {
                    if self.donor_seen_marks[l] != candidate_tick {
                        self.donor_seen_marks[l] = candidate_tick;
                        self.donor_candidate_pos.push(l);
                    }
                    let rr = r - 1;
                    if self.donor_seen_marks[rr] != candidate_tick {
                        self.donor_seen_marks[rr] = candidate_tick;
                        self.donor_candidate_pos.push(rr);
                    }
                    let mid = l + (r - l) / 2;
                    if self.donor_seen_marks[mid] != candidate_tick {
                        self.donor_seen_marks[mid] = candidate_tick;
                        self.donor_candidate_pos.push(mid);
                    }
                }
            }

            // If candidate generation was too sparse, add deterministic spread points.
            if self.donor_candidate_pos.len() < k {
                let mut spread_interval_idx = 0usize;
                let mut spread_interval_start_offset = 0usize;
                for i in 0..k {
                    let target = (2 * i + 1) * total_len / (2 * k);
                    while spread_interval_idx < self.intervals_buf.len() {
                        let (l, r) = self.intervals_buf[spread_interval_idx];
                        let len = r - l;
                        if target < spread_interval_start_offset + len {
                            let offset_in_interval = target - spread_interval_start_offset;
                            let pos = l + offset_in_interval;
                            if self.donor_seen_marks[pos] != candidate_tick {
                                self.donor_seen_marks[pos] = candidate_tick;
                                self.donor_candidate_pos.push(pos);
                            }
                            break;
                        }
                        spread_interval_start_offset += len;
                        spread_interval_idx += 1;
                    }
                }
            }

            let mut best: std::collections::BinaryHeap<DonorChoice> =
                std::collections::BinaryHeap::with_capacity(k + 1);
            for &pos in &self.donor_candidate_pos {
                if pos >= self.div.len() {
                    continue;
                }
                let choice = DonorChoice {
                    div: self.div[pos],
                    pos,
                };
                Self::push_top_k_choice(&mut best, choice, k);
            }

            // Safety net: if approximation produced too few unique candidates, backfill exactly.
            if best.len() < k {
                let chosen_tick = self.next_donor_seen_tick();
                for c in best.iter() {
                    self.donor_seen_marks[c.pos] = chosen_tick;
                }
                for &(l, r) in &self.intervals_buf {
                    for pos in l..r {
                        if self.donor_seen_marks[pos] == chosen_tick {
                            continue;
                        }
                        self.donor_seen_marks[pos] = chosen_tick;
                        if pos >= self.div.len() {
                            continue;
                        }
                        let choice = DonorChoice {
                            div: self.div[pos],
                            pos,
                        };
                        Self::push_top_k_choice(&mut best, choice, k);
                    }
                }
            }

            self.flush_top_k_choices(best, out);
            return;
        }

        let mut best: std::collections::BinaryHeap<DonorChoice> =
            std::collections::BinaryHeap::with_capacity(k + 1);
        for &(l, r) in &self.intervals_buf {
            for pos in l..r {
                if pos >= self.div.len() {
                    continue;
                }
                let choice = DonorChoice {
                    div: self.div[pos],
                    pos,
                };
                Self::push_top_k_choice(&mut best, choice, k);
            }
        }
        self.flush_top_k_choices(best, out);
    }

    fn bin_for_allele(a: u8, alphabet: PbwtAlphabet) -> usize {
        PbwtAllele::from_raw(a, alphabet).bin(alphabet)
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

    #[inline]
    fn count_for(&self, bin: usize, n_alleles: usize) -> u32 {
        if n_alleles == 2 {
            self.binary_counts.get(bin).copied().unwrap_or(0)
        } else {
            self.counts.get(bin).copied().unwrap_or(0)
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

    pub fn advance_with_beams_query_probs(
        &mut self,
        ref_alleles: &[u8],
        n_alleles: usize,
        marker: usize,
        query_alleles: &[PbwtQueryAllele],
        query_bin_probs: Option<&[PbwtBiallelicQueryProb]>,
        beams: &mut [RankBeam],
    ) {
        let mut scratch = std::mem::take(&mut self.step_scratch);
        self.advance_with_beams_query_scratch(
            ref_alleles,
            n_alleles,
            marker,
            query_alleles,
            query_bin_probs,
            beams,
            &mut scratch,
        );
        self.step_scratch = scratch;
    }

    pub fn advance_with_beams_strict(
        &mut self,
        ref_alleles: &[u8],
        n_alleles: usize,
        marker: usize,
        query_alleles: &[PbwtStrictAllele],
        beams: &mut [RankBeam],
    ) {
        let mut scratch = std::mem::take(&mut self.step_scratch);
        self.advance_with_beams_strict_scratch(
            ref_alleles,
            n_alleles,
            marker,
            query_alleles,
            beams,
            &mut scratch,
        );
        self.step_scratch = scratch;
    }

    pub fn advance_with_beams_query_scratch(
        &mut self,
        ref_alleles: &[u8],
        n_alleles: usize,
        marker: usize,
        query_alleles: &[PbwtQueryAllele],
        query_bin_probs: Option<&[PbwtBiallelicQueryProb]>,
        beams: &mut [RankBeam],
        scratch: &mut Vec<(u32, u32, u64)>,
    ) {
        self.prepare_step(ref_alleles, n_alleles);
        self.update_beams_with_scratch_query(
            beams,
            query_alleles,
            query_bin_probs,
            n_alleles,
            scratch,
        );
        self.finalize_step(ref_alleles, n_alleles, marker);
    }

    pub fn advance_with_beams_strict_scratch(
        &mut self,
        ref_alleles: &[u8],
        n_alleles: usize,
        marker: usize,
        query_alleles: &[PbwtStrictAllele],
        beams: &mut [RankBeam],
        scratch: &mut Vec<(u32, u32, u64)>,
    ) {
        self.prepare_step(ref_alleles, n_alleles);
        self.update_beams_with_scratch_strict(beams, query_alleles, n_alleles, scratch);
        self.finalize_step(ref_alleles, n_alleles, marker);
    }

    pub fn prepare_step(&mut self, ref_alleles: &[u8], n_alleles: usize) {
        let alphabet = PbwtAlphabet::new(n_alleles)
            .expect("invalid PBWT alphabet: n_alleles must be in 2..=255");
        let n_ref = self.ppa.len();
        let n_bins = if alphabet.n_alleles() == 2 {
            3
        } else {
            alphabet.n_bins()
        };
        if alphabet.n_alleles() == 2 {
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
                let b = Self::bin_for_allele(a, alphabet);
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
                    let bin = Self::bin_for_allele(a, alphabet);
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
        query_bin_probs: Option<&[PbwtBiallelicQueryProb]>,
        n_alleles: usize,
        scratch: &mut Vec<(u32, u32, u64)>,
    ) {
        let alphabet = PbwtAlphabet::new(n_alleles)
            .expect("invalid PBWT alphabet: n_alleles must be in 2..=255");
        let n_ref = self.ppa.len();
        let n_bins = if alphabet.n_alleles() == 2 {
            3
        } else {
            alphabet.n_bins()
        };

        for (q_idx, &qa) in query_alleles.iter().enumerate() {
            if q_idx >= beams.len() {
                break;
            }
            let old = beams[q_idx];
            let mut next = RankBeam {
                intervals: [(0, 0); MAX_RANK_INTERVALS],
                len: 0,
            };

            if qa.is_wildcard() {
                scratch.clear();
                for &(l, r) in old.intervals() {
                    if n_alleles == 2 {
                        for bb in PbwtBiallelicBin::NON_MISSING {
                            let b = bb.as_usize();
                            let nl =
                                self.offset_for(b, n_alleles) + self.rank(b, l, n_ref, n_alleles);
                            let nr =
                                self.offset_for(b, n_alleles) + self.rank(b, r, n_ref, n_alleles);
                            if nl < nr {
                                let len = nr - nl;
                                let p = query_bin_probs
                                    .and_then(|probs| probs.get(q_idx))
                                    .map(|prob| prob.prob_for_bin(bb))
                                    .unwrap_or(0.5);
                                let score = Self::scaled_score(len, p);
                                if score == 0 {
                                    continue;
                                }
                                scratch.push((nl, nr, score));
                            }
                        }
                    } else {
                        for b in 0..n_bins {
                            if b == 1 {
                                continue;
                            }
                            let nl =
                                self.offset_for(b, n_alleles) + self.rank(b, l, n_ref, n_alleles);
                            let nr =
                                self.offset_for(b, n_alleles) + self.rank(b, r, n_ref, n_alleles);
                            if nl < nr {
                                let score = (nr - nl) as u64;
                                scratch.push((nl, nr, score));
                            }
                        }
                    }
                }
                Self::load_top_intervals(scratch, &mut next, MAX_RANK_INTERVALS);
            } else if qa.is_missing() {
                scratch.clear();
                for &(l, r) in old.intervals() {
                    for b in 0..n_bins {
                        let nl = self.offset_for(b, n_alleles) + self.rank(b, l, n_ref, n_alleles);
                        let nr = self.offset_for(b, n_alleles) + self.rank(b, r, n_ref, n_alleles);
                        if nl < nr {
                            let score = (nr - nl) as u64;
                            scratch.push((nl, nr, score));
                        }
                    }
                }
                Self::load_top_intervals(scratch, &mut next, MAX_RANK_INTERVALS);
            } else {
                let queried_allele = qa
                    .as_allele()
                    .expect("non-wildcard/non-missing query allele should be concrete");
                let b = Self::bin_for_allele(queried_allele, alphabet);
                if b < n_bins && n_alleles == 2 && query_bin_probs.is_some() {
                    scratch.clear();
                    let (p0, p1) = query_bin_probs
                        .and_then(|probs| probs.get(q_idx))
                        .map(|prob| (prob.prob_for_allele(0), prob.prob_for_allele(1)))
                        .unwrap_or_else(|| {
                            match PbwtAllele::from_raw(queried_allele, alphabet).biallelic_bin() {
                                PbwtBiallelicBin::Ref => (1.0, 0.0),
                                PbwtBiallelicBin::Alt => (0.0, 1.0),
                                PbwtBiallelicBin::Missing => (0.5, 0.5),
                            }
                        });
                    for &(l, r) in old.intervals() {
                        for (bb, p) in [(PbwtBiallelicBin::Ref, p0), (PbwtBiallelicBin::Alt, p1)] {
                            let b_idx = bb.as_usize();
                            let nl = self.offset_for(b_idx, n_alleles)
                                + self.rank(b_idx, l, n_ref, n_alleles);
                            let nr = self.offset_for(b_idx, n_alleles)
                                + self.rank(b_idx, r, n_ref, n_alleles);
                            if nl < nr {
                                let score = Self::scaled_score(nr - nl, p);
                                if score == 0 {
                                    continue;
                                }
                                scratch.push((nl, nr, score));
                            }
                        }
                    }
                    Self::load_top_intervals(scratch, &mut next, MAX_RANK_INTERVALS);
                } else if b < n_bins {
                    for &(l, r) in old.intervals() {
                        let nl = self.offset_for(b, n_alleles) + self.rank(b, l, n_ref, n_alleles);
                        let nr = self.offset_for(b, n_alleles) + self.rank(b, r, n_ref, n_alleles);
                        next.push_interval(nl, nr);
                    }
                } else {
                    next = RankBeam::full(n_ref as u32);
                }

                if next.len == 0 {
                    // Preserve locality first: recover from the previous beam across bins.
                    // This avoids global hard-resets on single-site errors/mismatches.
                    scratch.clear();
                    for &(l, r) in old.intervals() {
                        for b in 0..n_bins {
                            let nl =
                                self.offset_for(b, n_alleles) + self.rank(b, l, n_ref, n_alleles);
                            let nr =
                                self.offset_for(b, n_alleles) + self.rank(b, r, n_ref, n_alleles);
                            if nl < nr {
                                let score = (nr - nl) as u64;
                                scratch.push((nl, nr, score));
                            }
                        }
                    }
                    Self::load_top_intervals(scratch, &mut next, MAX_RANK_INTERVALS);
                }

                if next.len == 0 {
                    let queried_bin = Self::bin_for_allele(queried_allele, alphabet);
                    let nl = self.offset_for(queried_bin, n_alleles);
                    let nr = nl + self.count_for(queried_bin, n_alleles);
                    if nl < nr {
                        next.intervals[0] = (nl, nr);
                        next.len = 1;
                    }
                }

                if next.len == 0 {
                    next = RankBeam::full(n_ref as u32);
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
        scratch: &mut Vec<(u32, u32, u64)>,
    ) {
        let alphabet = PbwtAlphabet::new(n_alleles)
            .expect("invalid PBWT alphabet: n_alleles must be in 2..=255");
        let n_ref = self.ppa.len();
        let n_bins = if alphabet.n_alleles() == 2 {
            3
        } else {
            alphabet.n_bins()
        };

        for (q_idx, &qa) in query_alleles.iter().enumerate() {
            if q_idx >= beams.len() {
                break;
            }
            let old = beams[q_idx];
            let mut next = RankBeam {
                intervals: [(0, 0); MAX_RANK_INTERVALS],
                len: 0,
            };

            if qa.is_missing() {
                scratch.clear();
                for &(l, r) in old.intervals() {
                    for b in 0..n_bins {
                        let nl = self.offset_for(b, n_alleles) + self.rank(b, l, n_ref, n_alleles);
                        let nr = self.offset_for(b, n_alleles) + self.rank(b, r, n_ref, n_alleles);
                        if nl < nr {
                            let score = (nr - nl) as u64;
                            scratch.push((nl, nr, score));
                        }
                    }
                }
                Self::load_top_intervals(scratch, &mut next, MAX_RANK_INTERVALS);
            } else {
                let queried_allele = qa
                    .as_allele()
                    .expect("non-missing strict query allele should be concrete");
                let b = Self::bin_for_allele(queried_allele, alphabet);
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
                    let queried_bin = Self::bin_for_allele(queried_allele, alphabet);
                    let nl = self.offset_for(queried_bin, n_alleles);
                    let nr = nl + self.count_for(queried_bin, n_alleles);
                    if nl < nr {
                        next.intervals[0] = (nl, nr);
                        next.len = 1;
                    }
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

    pub fn collect_positions_and_lens(
        &mut self,
        marker: usize,
        haps: &[u32],
        out: &mut Vec<(u32, usize, i32)>,
    ) {
        out.clear();
        if haps.is_empty() {
            return;
        }
        let m = marker as i32;
        self.wanted_map.clear();
        self.wanted_map.reserve(haps.len());
        for (i, &h) in haps.iter().enumerate() {
            self.wanted_map.insert(h, i);
        }
        if self.found_pos_start.len() < haps.len() {
            self.found_pos_start.resize(haps.len(), (0, m));
        }
        if self.found_mask.len() < haps.len() {
            self.found_mask.resize(haps.len(), false);
        }
        for i in 0..haps.len() {
            self.found_pos_start[i] = (0, m);
            self.found_mask[i] = false;
        }
        let mut remaining = self.wanted_map.len();
        for (pos, hap) in self.ppa.iter().enumerate() {
            if remaining == 0 {
                break;
            }
            let h = hap.to_usize() as u32;
            if let Some(&idx) = self.wanted_map.get(&h) {
                let start = self.div.get(pos).copied().unwrap_or(m);
                self.found_pos_start[idx] = (pos, start);
                self.found_mask[idx] = true;
                remaining -= 1;
            }
        }
        for (i, &h) in haps.iter().enumerate() {
            if self.found_mask[i] {
                let (pos, start) = self.found_pos_start[i];
                out.push((h, pos, start));
            }
        }
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

    pub fn advance_with_beams_query_probs(
        &mut self,
        ref_alleles: &[u8],
        n_alleles: usize,
        marker: usize,
        query_alleles: &[PbwtQueryAllele],
        query_bin_probs: Option<&[PbwtBiallelicQueryProb]>,
        beams: &mut [RankBeam],
    ) {
        match self {
            Self::U16(inner) => inner.advance_with_beams_query_probs(
                ref_alleles,
                n_alleles,
                marker,
                query_alleles,
                query_bin_probs,
                beams,
            ),
            Self::U32(inner) => inner.advance_with_beams_query_probs(
                ref_alleles,
                n_alleles,
                marker,
                query_alleles,
                query_bin_probs,
                beams,
            ),
        }
    }

    pub fn prepare_step(&mut self, ref_alleles: &[u8], n_alleles: usize) {
        match self {
            Self::U16(inner) => inner.prepare_step(ref_alleles, n_alleles),
            Self::U32(inner) => inner.prepare_step(ref_alleles, n_alleles),
        }
    }

    pub fn update_beams_with_scratch_query(
        &mut self,
        beams: &mut [RankBeam],
        query_alleles: &[PbwtQueryAllele],
        query_bin_probs: Option<&[PbwtBiallelicQueryProb]>,
        n_alleles: usize,
        scratch: &mut Vec<(u32, u32, u64)>,
    ) {
        match self {
            Self::U16(inner) => {
                inner.update_beams_with_scratch_query(
                    beams,
                    query_alleles,
                    query_bin_probs,
                    n_alleles,
                    scratch,
                )
            }
            Self::U32(inner) => {
                inner.update_beams_with_scratch_query(
                    beams,
                    query_alleles,
                    query_bin_probs,
                    n_alleles,
                    scratch,
                )
            }
        }
    }

    pub fn finalize_step(&mut self, ref_alleles: &[u8], n_alleles: usize, marker: usize) {
        match self {
            Self::U16(inner) => inner.finalize_step(ref_alleles, n_alleles, marker),
            Self::U32(inner) => inner.finalize_step(ref_alleles, n_alleles, marker),
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

    pub fn collect_positions_and_lens(
        &mut self,
        marker: usize,
        haps: &[u32],
        out: &mut Vec<(u32, usize, i32)>,
    ) {
        match self {
            Self::U16(inner) => inner.collect_positions_and_lens(marker, haps, out),
            Self::U32(inner) => inner.collect_positions_and_lens(marker, haps, out),
        }
    }
}
