use crate::model::pbwt::PbwtDivUpdater;

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

        self.updater
            .fwd_update(ref_alleles, n_alleles, marker, &mut self.ppa, &mut self.div);
    }
}
