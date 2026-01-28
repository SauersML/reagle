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
        }
    }

    pub fn with_state(n_ref_haps: usize, state: Option<&PbwtState>) -> Self {
        let mut pbwt = Self::new(n_ref_haps);
        if let Some(state) = state {
            if state.ppa.len() == n_ref_haps && state.div.len() == n_ref_haps {
                pbwt.ppa = state.ppa.clone();
                pbwt.div = state.div.clone();
            }
        }
        pbwt
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

        // Round-robin selection from all intervals to avoid excluding rare matches
        // when a large common interval comes first.
        struct IntervalState {
            l: usize,
            r: usize,
            left: usize,
            right: usize,
            exhausted: bool,
            pick_left: bool,
        }

        let mut states: Vec<IntervalState> = beam
            .intervals()
            .iter()
            .map(|&(l, r)| {
                let l = l.min(n_ref as u32) as usize;
                let r = r.min(n_ref as u32) as usize;
                let center = (l + r) / 2;
                IntervalState {
                    l,
                    r,
                    left: center,
                    right: center,
                    exhausted: l >= r,
                    pick_left: true,
                }
            })
            .collect();

        let mut active = true;
        while out.len() < k && active {
            active = false;
            for state in &mut states {
                if out.len() >= k {
                    break;
                }
                if state.exhausted {
                    continue;
                }

                let can_left = state.left > state.l;
                let can_right = state.right < state.r;

                if !can_left && !can_right {
                    state.exhausted = true;
                    continue;
                }

                if state.pick_left {
                    if can_left {
                        state.left -= 1;
                        out.push(self.ppa[state.left]);
                        active = true;
                        state.pick_left = false;
                    } else {
                        // Must pick right
                        out.push(self.ppa[state.right]);
                        state.right += 1;
                        active = true;
                        // Keep pick_left=true since we failed to pick left
                    }
                } else {
                    if can_right {
                        out.push(self.ppa[state.right]);
                        state.right += 1;
                        active = true;
                        state.pick_left = true;
                    } else {
                        // Must pick left
                        state.left -= 1;
                        out.push(self.ppa[state.left]);
                        active = true;
                        // Keep pick_left=false since we failed to pick right
                    }
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
        self.prepare_step(ref_alleles, n_alleles);
        self.update_beams(beams, query_alleles, n_alleles);
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
                let mut candidates: Vec<(u32, u32, u32)> = Vec::new();
                for &(l, r) in old.intervals() {
                    for b in 0..n_bins {
                        let nl = self.offsets[b] + self.rank(b, l, n_ref);
                        let nr = self.offsets[b] + self.rank(b, r, n_ref);
                        if nl < nr {
                            let len = nr - nl;
                            let score = len.saturating_mul(self.counts[b]);
                            candidates.push((nl, nr, score));
                        }
                    }
                }

                candidates.sort_unstable_by(|a, b| b.2.cmp(&a.2));
                let keep = candidates.len().min(MAX_RANK_INTERVALS);
                for i in 0..keep {
                    next.intervals[i] = (candidates[i].0, candidates[i].1);
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
                    let mut candidates: Vec<(u32, u32, u32)> = Vec::new();
                    for &(l, r) in old.intervals() {
                        for b in 0..n_bins {
                            let nl = self.offsets[b] + self.rank(b, l, n_ref);
                            let nr = self.offsets[b] + self.rank(b, r, n_ref);
                            if nl < nr {
                                let len = nr - nl;
                                let score = len.saturating_mul(self.counts[b]);
                                candidates.push((nl, nr, score));
                            }
                        }
                    }

                    candidates.sort_unstable_by(|a, b| b.2.cmp(&a.2));
                    let keep = candidates.len().min(MAX_RANK_INTERVALS);
                    for i in 0..keep {
                        next.intervals[i] = (candidates[i].0, candidates[i].1);
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

    pub fn advance_with_rephase(
        &mut self,
        ref_alleles: &[u8],
        n_alleles: usize,
        marker: usize,
        query_alleles: &mut [u8],
        beams: &mut [RankBeam],
        swaps_out: &mut [bool],
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
                let score_keep =
                    ((len_keep_h1 as f32) / (count_a1 + 1.0)) * ((len_keep_h2 as f32) / (count_a2 + 1.0));

                let len_swap_h1 = self.match_len(&beams[h1], a2, n_alleles);
                let len_swap_h2 = self.match_len(&beams[h2], a1, n_alleles);

                // For swap: h1 gets a2 (so we use count_a2), h2 gets a1 (so we use count_a1)
                let score_swap =
                    ((len_swap_h1 as f32) / (count_a2 + 1.0)) * ((len_swap_h2 as f32) / (count_a1 + 1.0));

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
