use crate::model::pbwt::{PbwtDivUpdater, PbwtState};

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

pub struct ReferencePbwt {
    updater: PbwtDivUpdater,
    ppa: Vec<u32>,
    div: Vec<i32>,
    permuted_ref: Vec<u8>,
    prefix_counts: Vec<u32>,
    counts: Vec<u32>,
    offsets: Vec<u32>,
    inv_ppa: Vec<u32>,
}

impl ReferencePbwt {
    pub fn new(n_ref_haps: usize) -> Self {
        Self {
            updater: PbwtDivUpdater::new(n_ref_haps),
            ppa: (0..n_ref_haps as u32).collect(),
            div: vec![0; n_ref_haps],
            permuted_ref: vec![0; n_ref_haps],
            prefix_counts: Vec::new(),
            counts: Vec::new(),
            offsets: Vec::new(),
            inv_ppa: (0..n_ref_haps as u32).collect(),
        }
    }

    pub fn with_state(n_ref_haps: usize, state: Option<&PbwtState>) -> Self {
        let mut pbwt = Self::new(n_ref_haps);
        if let Some(state) = state {
            if state.ppa.len() == n_ref_haps && state.div.len() == n_ref_haps {
                pbwt.ppa = state.ppa.clone();
                pbwt.div = state.div.clone();
                pbwt.update_inv_ppa();
            }
        }
        pbwt
    }

    fn update_inv_ppa(&mut self) {
        if self.inv_ppa.len() != self.ppa.len() {
            self.inv_ppa.resize(self.ppa.len(), 0);
        }
        for (i, &hap) in self.ppa.iter().enumerate() {
            if (hap as usize) < self.inv_ppa.len() {
                self.inv_ppa[hap as usize] = i as u32;
            }
        }
    }

    pub fn get_state(&self, marker_pos: usize) -> PbwtState {
        PbwtState::new(self.ppa.clone(), self.div.clone(), marker_pos)
    }

    pub fn select_donors(&self, beam: &RankBeam, k: usize) -> Vec<u32> {
        if k == 0 {
            return Vec::new();
        }
        let n_ref = self.ppa.len();
        if n_ref == 0 {
            return Vec::new();
        }

        let mut out = Vec::with_capacity(k);

        for &(l, r) in beam.intervals() {
            if out.len() >= k {
                break;
            }
            let l = l.min(n_ref as u32) as usize;
            let r = r.min(n_ref as u32) as usize;
            if l >= r {
                continue;
            }
            let center = (l + r) / 2;

            let mut left = center;
            let mut right = center;
            while out.len() < k && (left > l || right < r) {
                if left > l {
                    left -= 1;
                    out.push(self.ppa[left]);
                    if out.len() >= k {
                        break;
                    }
                }
                if right < r {
                    out.push(self.ppa[right]);
                    right += 1;
                }
            }
        }

        out.sort_unstable();
        out.dedup();
        if out.len() > k {
            out.truncate(k);
        }
        out
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

    fn prefix_idx(bin: usize, pos: usize, n_ref: usize) -> usize {
        bin * (n_ref + 1) + pos
    }

    fn rank(&self, bin: usize, pos: u32, n_ref: usize) -> u32 {
        let p = pos.min(n_ref as u32) as usize;
        self.prefix_counts[Self::prefix_idx(bin, p, n_ref)]
    }

    pub fn advance_with_beams(
        &mut self,
        ref_alleles: &[u8],
        n_alleles: usize,
        marker: usize,
        query_alleles: &[u8],
        beams: &mut [RankBeam],
    ) {
        let mut scratch: Vec<(u32, u32, u32)> = Vec::new();
        self.advance_with_beams_scratch(
            ref_alleles,
            n_alleles,
            marker,
            query_alleles,
            beams,
            &mut scratch,
        );
    }

    pub fn advance_with_beams_scratch(
        &mut self,
        ref_alleles: &[u8],
        n_alleles: usize,
        marker: usize,
        query_alleles: &[u8],
        beams: &mut [RankBeam],
        scratch: &mut Vec<(u32, u32, u32)>,
    ) {
        self.prepare_step(ref_alleles, n_alleles);
        self.update_beams_with_scratch(beams, query_alleles, n_alleles, scratch);
        self.finalize_step(ref_alleles, n_alleles, marker);
    }

    pub fn prepare_step(&mut self, ref_alleles: &[u8], n_alleles: usize) {
        let n_ref = self.ppa.len();
        let n_bins = if n_alleles == 2 { 3 } else { n_alleles + 1 };
        self.ensure_buffers(n_bins);

        for i in 0..n_ref {
            self.permuted_ref[i] = ref_alleles[self.ppa[i] as usize];
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

    pub fn match_len(&self, beam: &RankBeam, allele: u8, n_alleles: usize) -> u32 {
        if allele == 255 {
            return 0;
        }
        let n_ref = self.ppa.len();
        let n_bins = if n_alleles == 2 { 3 } else { n_alleles + 1 };
        let b = Self::bin_for_allele(allele, n_alleles);

        if b >= n_bins {
            return 0;
        }

        let mut total = 0;
        for &(l, r) in beam.intervals() {
            let nl = self.offsets[b] + self.rank(b, l, n_ref);
            let nr = self.offsets[b] + self.rank(b, r, n_ref);
            if nr > nl {
                total += nr - nl;
            }
        }
        total
    }

    pub fn update_beams(&self, beams: &mut [RankBeam], query_alleles: &[u8], n_alleles: usize) {
        let mut scratch: Vec<(u32, u32, u32)> = Vec::new();
        self.update_beams_with_scratch(beams, query_alleles, n_alleles, &mut scratch);
    }

    pub fn update_beams_with_scratch(
        &self,
        beams: &mut [RankBeam],
        query_alleles: &[u8],
        n_alleles: usize,
        scratch: &mut Vec<(u32, u32, u32)>,
    ) {
        let n_ref = self.ppa.len();
        let n_bins = if n_alleles == 2 { 3 } else { n_alleles + 1 };

        for (q_idx, &qa) in query_alleles.iter().enumerate() {
            if q_idx >= beams.len() {
                break;
            }
            let old = beams[q_idx];
            let mut next = RankBeam {
                intervals: [(0, 0); MAX_RANK_INTERVALS],
                len: 0,
            };

            if qa == 255 {
                // Missing query allele: mapping to the union of all bins can cause an
                // interval explosion and (with overflow merging) degenerate to the full
                // reference range on sparse targets. Instead, keep only the most informative
                // mapped intervals.
                scratch.clear();
                for &(l, r) in old.intervals() {
                    for b in 0..n_bins {
                        let nl = self.offsets[b] + self.rank(b, l, n_ref);
                        let nr = self.offsets[b] + self.rank(b, r, n_ref);
                        if nl < nr {
                            let len = nr - nl;
                            let score = len.saturating_mul(self.counts[b]);
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
                        let nl = self.offsets[b] + self.rank(b, l, n_ref);
                        let nr = self.offsets[b] + self.rank(b, r, n_ref);
                        next.push_interval(nl, nr);
                    }
                } else {
                    next = RankBeam::full(n_ref as u32);
                }

                if next.len == 0 {
                    scratch.clear();
                    for &(l, r) in old.intervals() {
                        for b in 0..n_bins {
                            let nl = self.offsets[b] + self.rank(b, l, n_ref);
                            let nr = self.offsets[b] + self.rank(b, r, n_ref);
                            if nl < nr {
                                let len = nr - nl;
                                let score = len.saturating_mul(self.counts[b]);
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
        self.update_inv_ppa();
    }

    pub fn advance_with_rephase(
        &mut self,
        ref_alleles: &[u8],
        n_alleles: usize,
        marker: usize,
        query_alleles: &mut [u8],
        beams: &mut [RankBeam],
        swaps_out: &mut [bool],
        hints: Option<&[u32]>,
    ) {
        self.prepare_step(ref_alleles, n_alleles);

        let n_samples = query_alleles.len() / 2;
        // Greedy Local Rephasing
        for s in 0..n_samples {
            let h1 = s * 2;
            let h2 = h1 + 1;

            if h2 >= beams.len() {
                continue;
            }

            let a1 = query_alleles[h1];
            let a2 = query_alleles[h2];

            if a1 != a2 && a1 != 255 && a2 != 255 {
                let len_keep_h1 = self.match_len(&beams[h1], a1, n_alleles);
                let len_keep_h2 = self.match_len(&beams[h2], a2, n_alleles);

                let b1 = Self::bin_for_allele(a1, n_alleles);
                let count_a1 = self.counts[b1].max(1) as f32;
                let b2 = Self::bin_for_allele(a2, n_alleles);
                let count_a2 = self.counts[b2].max(1) as f32;

                // Smoothed Consistency Scoring: len / (count + 1)
                // This balances maximizing match length with preferring unique/rare haplotypes (high consistency),
                // but avoids over-penalizing common haplotypes by adding +1 smoothing.
                // Smoothed Consistency Scoring: len / (count + 1)
                // This balances maximizing match length with preferring unique/rare haplotypes (high consistency),
                // but avoids over-penalizing common haplotypes by adding +1 smoothing.
                let mut score_keep =
                    ((len_keep_h1 as f32) / (count_a1 + 1.0)) * ((len_keep_h2 as f32) / (count_a2 + 1.0));

                let len_swap_h1 = self.match_len(&beams[h1], a2, n_alleles);
                let len_swap_h2 = self.match_len(&beams[h2], a1, n_alleles);

                // For swap: h1 gets a2 (so we use count_a2), h2 gets a1 (so we use count_a1)
                let mut score_swap =
                    ((len_swap_h1 as f32) / (count_a2 + 1.0)) * ((len_swap_h2 as f32) / (count_a1 + 1.0));

                if let Some(hints_vec) = hints {
                    // Boost scores if the hint haplotype is compatible and present in the beam
                    let boost = 1000.0;
                    
                    // Check H1 hint
                    if h1 < hints_vec.len() {
                        let h_hint = hints_vec[h1] as usize;
                        if h_hint < self.inv_ppa.len() {
                            let rank = self.inv_ppa[h_hint];
                            let ref_al = self.permuted_ref[self.inv_ppa[h_hint] as usize]; 
                            // Note: permuted_ref stores alleles in PPA order. 
                            // ref_alleles passed to this function are in Hap order.
                            // Better use ref_alleles directly if available, but they are permuted inside prepare_step.
                            // Wait, ref_alleles passed to advance_with_rephase are NOT permuted yet?
                            // prepare_step fills self.permuted_ref from ref_alleles[ppa[i]].
                            // So self.permuted_ref[rank] is the allele for the haplotype at rank .
                            // Since rank = inv_ppa[h_hint], the haplotype at rank is .
                            // So self.permuted_ref[rank] IS the allele of .
                            
                            let hint_allele = self.permuted_ref[rank as usize];
                            
                            // Check if hint is in beam[h1]
                            // Beams are intervals of ranks.
                            let mut in_beam = false;
                            for &(l, r) in beams[h1].intervals() {
                                if rank >= l && rank < r {
                                    in_beam = true;
                                    break;
                                }
                            }
                            
                            if in_beam {
                                if hint_allele == a1 {
                                    score_keep += boost;
                                } else if hint_allele == a2 {
                                    score_swap += boost;
                                }
                            }
                        }
                    }
                    
                    // Check H2 hint
                    if h2 < hints_vec.len() {
                        let h_hint = hints_vec[h2] as usize;
                        if h_hint < self.inv_ppa.len() {
                            let rank = self.inv_ppa[h_hint];
                            let hint_allele = self.permuted_ref[rank as usize];
                            
                            let mut in_beam = false;
                            for &(l, r) in beams[h2].intervals() {
                                if rank >= l && rank < r {
                                    in_beam = true;
                                    break;
                                }
                            }
                            
                            if in_beam {
                                if hint_allele == a2 {
                                    score_keep += boost;
                                } else if hint_allele == a1 {
                                    score_swap += boost;
                                }
                            }
                        }
                    }
                }

                if score_swap > score_keep {
                if score_swap > score_keep {
                    query_alleles[h1] = a2;
                    query_alleles[h2] = a1;
                    swaps_out[s] = true;
                } else {
                    swaps_out[s] = false;
                }
            } else {
                swaps_out[s] = false;
            }
        }

        self.update_beams(beams, query_alleles, n_alleles);
        self.finalize_step(ref_alleles, n_alleles, marker);
    }
}
