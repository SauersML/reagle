//! Cluster-space imputation state selection using recursive IBS partitioning.
//!
//! Mirrors Java's ImpStates over cluster indices (not dense markers).

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::model::imp_ibs::ImpIbs;
use crate::model::states::ThreadedHaps;
use crate::utils::workspace::ImpWorkspace;

const IBS_NIL: i32 = i32::MIN;
const JAVA_RNG_MULT: u64 = 0x5DEECE66D;
const JAVA_RNG_ADD: u64 = 0xB;
const JAVA_RNG_MASK: u64 = (1u64 << 48) - 1;

#[derive(Clone, Debug)]
struct CompHapEntry {
    comp_hap_idx: usize,
    hap: u32,
    last_ibs_step: i32,
}

impl PartialEq for CompHapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.last_ibs_step == other.last_ibs_step
    }
}

impl Eq for CompHapEntry {}

impl PartialOrd for CompHapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CompHapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other.last_ibs_step.cmp(&self.last_ibs_step)
    }
}

#[derive(Clone, Debug)]
struct JavaRng {
    seed: u64,
}

impl JavaRng {
    fn new(seed: u64) -> Self {
        let seed = (seed ^ JAVA_RNG_MULT) & JAVA_RNG_MASK;
        Self { seed }
    }

    fn next_bits(&mut self, bits: u32) -> u32 {
        self.seed = (self.seed.wrapping_mul(JAVA_RNG_MULT).wrapping_add(JAVA_RNG_ADD)) & JAVA_RNG_MASK;
        (self.seed >> (48 - bits)) as u32
    }

    fn next_int(&mut self, bound: usize) -> usize {
        if bound <= 1 {
            return 0;
        }
        if bound.is_power_of_two() {
            return (((bound as u64) * (self.next_bits(31) as u64)) >> 31) as usize;
        }
        loop {
            let bits = self.next_bits(31) as i64;
            let val = (bits % bound as i64) as i64;
            if bits - val + (bound as i64 - 1) >= 0 {
                return val as usize;
            }
        }
    }
}

pub struct ImpStatesCluster<'a> {
    max_states: usize,
    n_clusters: usize,
    n_ref_haps: usize,
    ibs: &'a ImpIbs,
    threaded_haps: ThreadedHaps,
    hap_to_last_ibs: Vec<i32>,
    queue: BinaryHeap<CompHapEntry>,
}

impl<'a> ImpStatesCluster<'a> {
    pub fn new(
        ibs: &'a ImpIbs,
        n_clusters: usize,
        n_ref_haps: usize,
        max_states: usize,
    ) -> Self {
        Self {
            max_states,
            n_clusters,
            n_ref_haps,
            ibs,
            threaded_haps: ThreadedHaps::new(max_states, max_states * 4, n_clusters),
            hap_to_last_ibs: vec![IBS_NIL; n_ref_haps],
            queue: BinaryHeap::with_capacity(max_states),
        }
    }

    pub fn ibs_states_cluster(
        &mut self,
        targ_hap: usize,
        hap_indices: &mut Vec<Vec<u32>>,
    ) -> usize {
        self.initialize();

        let n_steps = self.ibs.coded_steps.n_steps();
        for step_idx in 0..n_steps {
            let ibs_haps = self.ibs.ibs_haps(targ_hap, step_idx);
            for &hap in ibs_haps {
                self.update_with_ibs_hap(hap, step_idx as i32);
            }
        }

        if self.queue.is_empty() {
            self.fill_with_random(targ_hap as u32);
        }

        let n_states = self.queue.len().min(self.max_states);
        self.build_output(n_states, hap_indices);
        n_states
    }

    fn initialize(&mut self) {
        self.hap_to_last_ibs.fill(IBS_NIL);
        self.threaded_haps.clear();
        self.queue.clear();
    }

    fn update_with_ibs_hap(&mut self, hap: u32, step: i32) {
        let hap_idx = hap as usize;
        if self.hap_to_last_ibs[hap_idx] == IBS_NIL {
            self.update_queue_head();

            if self.queue.len() == self.max_states {
                if let Some(mut head) = self.queue.pop() {
                    let mid_step = (head.last_ibs_step + step) / 2;
                    let mid_step_idx = mid_step.max(0) as usize;
                    let start_cluster = if mid_step_idx < self.ibs.coded_steps.n_steps() {
                        self.ibs.coded_steps.step_start(mid_step_idx)
                    } else {
                        0
                    };

                    self.hap_to_last_ibs[head.hap as usize] = IBS_NIL;
                    if head.comp_hap_idx < self.threaded_haps.n_states() {
                        self.threaded_haps.add_segment(head.comp_hap_idx, hap, start_cluster);
                    }
                    head.hap = hap;
                    head.last_ibs_step = step;
                    self.queue.push(head);
                }
            } else {
                let comp_hap_idx = self.threaded_haps.push_new(hap);
                self.queue.push(CompHapEntry {
                    comp_hap_idx,
                    hap,
                    last_ibs_step: step,
                });
            }
        }
        self.hap_to_last_ibs[hap_idx] = step;
    }

    fn update_queue_head(&mut self) {
        while let Some(head) = self.queue.peek() {
            let last_ibs = self.hap_to_last_ibs[head.hap as usize];
            if head.last_ibs_step != last_ibs {
                let mut head = self.queue.pop().unwrap();
                head.last_ibs_step = last_ibs;
                self.queue.push(head);
            } else {
                break;
            }
        }
    }

    fn fill_with_random(&mut self, targ_hap_hash: u32) {
        let n_states = self.max_states.min(self.n_ref_haps);
        if self.queue.len() >= n_states {
            return;
        }

        let mut rng = JavaRng::new(targ_hap_hash as u64);
        let ibs_step = 0;
        let mut attempts = 0;
        while self.queue.len() < n_states && attempts < self.n_ref_haps * 2 {
            let h = rng.next_int(self.n_ref_haps) as u32;
            if self.hap_to_last_ibs[h as usize] == IBS_NIL {
                let comp_hap_idx = self.threaded_haps.push_new(h);
                self.queue.push(CompHapEntry {
                    comp_hap_idx,
                    hap: h,
                    last_ibs_step: ibs_step,
                });
                self.hap_to_last_ibs[h as usize] = ibs_step;
            }
            attempts += 1;
        }
    }

    fn build_output(&mut self, n_states: usize, hap_indices: &mut Vec<Vec<u32>>) {
        hap_indices.clear();
        hap_indices.resize(self.n_clusters, vec![0u32; n_states]);

        self.threaded_haps.reset_cursors();

        let mut entries: Vec<CompHapEntry> = self.queue.iter().cloned().take(n_states).collect();
        entries.sort_by_key(|e| e.comp_hap_idx);

        for c in 0..self.n_clusters {
            for (j, entry) in entries.iter().enumerate() {
                if entry.comp_hap_idx < self.threaded_haps.n_states() {
                    let hap = self.threaded_haps.hap_at_raw(entry.comp_hap_idx, c);
                    hap_indices[c][j] = hap;
                }
            }
        }
    }
}
