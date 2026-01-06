use crate::model::parameters::ModelParams;

/// Manages the HMM state for a single sample during online phasing.
/// Allows step-by-step updates as we sweep across markers.
#[derive(Debug, Clone)]
pub struct OnlineHmm {
    /// Active state indices (haplotype indices) from the previous step
    prev_states: Vec<u32>,
    
    /// Normalized forward probabilities for Combined (unphased-like) pass
    prev_fwd_combined: Vec<f32>,
    
    /// Normalized forward probabilities for Hap1
    prev_fwd1: Vec<f32>,
    
    /// Normalized forward probabilities for Hap2
    prev_fwd2: Vec<f32>,

    /// Generation counter for O(1) state lookup
    generation: u16,

    /// Global usage map: maps global_hap_idx -> (generation, local_idx_in_prev_states)
    /// Used to quickly find if a haplotype was present in the previous step and where.
    global_state_map: Vec<StateEntry>,
}

#[derive(Debug, Clone, Copy, Default)]
struct StateEntry {
    generation: u16,
    index: u16, // Assuming max K < 65536, which is extremely likely (default K=100-200)
}

impl OnlineHmm {
    /// Create a new OnlineHmm tracker
    pub fn new() -> Self {
        Self {
            prev_states: Vec::new(),
            prev_fwd_combined: Vec::new(),
            prev_fwd1: Vec::new(),
            prev_fwd2: Vec::new(),
            generation: 1, // Start at 1 because 0 is the default/empty state in the map
            global_state_map: Vec::new(),
        }
    }
    
    /// Initialize at the first marker with a set of states
    pub fn init(&mut self, initial_states: &[u32]) {
        self.prev_states = initial_states.to_vec();
        let n = initial_states.len();
        let init_prob = if n > 0 { 1.0 / n as f32 } else { 0.0 };
        
        // At start, everything is uniform
        self.prev_fwd_combined = vec![init_prob; n];
        self.prev_fwd1 = vec![init_prob; n];
        self.prev_fwd2 = vec![init_prob; n];

        // Initialize map
        self.generation = 1;
        // Ensure map is large enough for max haplotype index found
        if let Some(&max_hap) = initial_states.iter().max() {
            if self.global_state_map.len() <= max_hap as usize {
                self.global_state_map.resize(max_hap as usize + 1, StateEntry::default());
            }
        }
        
        for (i, &hap) in initial_states.iter().enumerate() {
             self.global_state_map[hap as usize] = StateEntry {
                 generation: self.generation,
                 index: i as u16,
             };
        }
    }
    
    /// Advance one step in the HMM
    /// 
    /// # Arguments
    /// * `new_states`: The set of active states (haplotype indices) for the current marker.
    /// * `emission_fn`: A closure `(hap_idx) -> (emit_c, emit1, emit2)`
    ///   - emit_c: Emission prob for combined
    ///   - emit1: Emission prob for hap1
    ///   - emit2: Emission prob for hap2
    /// * `p_recomb`: Recombination probability since last marker
    /// * `n_total_haps`: Total number of haplotypes (N) for transition smoothing
    pub fn step<F>(
        &mut self,
        new_states: &[u32],
        mut emission_fn: F,
        p_recomb: f32,
        n_total_haps: usize,
    ) 
    where F: FnMut(u32) -> (f32, f32, f32)
    {
        let n_new = new_states.len();
        // REMOVED early return: Must process even if empty to update generation and sync state
        // if n_new == 0 { ... }


        // Resize map if needed
        if self.global_state_map.len() < n_total_haps {
            self.global_state_map.resize(n_total_haps, StateEntry::default());
        }

        // Calculate transition terms
        // T(j -> k) = (1-rho) * delta(j,k) + rho/N
        // P(S_m = k) = sum_j P(S_{m-1}=j) * T(j->k)
        //            = sum_j P(j) * (rho/N) + P(k)*(1-rho) [if k was active]
        //            = (rho/N) * sum_j P(j) + (1-rho) * P_prev(k)
        // Since P(j) are normalized, sum_j P(j) = 1.0 (approximately)
        // So P_trans(k) = (rho/N) + (1-rho) * P_prev(k)
        
        let shift = p_recomb / n_total_haps as f32;
        let stay = 1.0 - p_recomb; // Probability of NOT recombining globally
        
        let mut new_fwd_c = Vec::with_capacity(n_new);
        let mut new_fwd1 = Vec::with_capacity(n_new);
        let mut new_fwd2 = Vec::with_capacity(n_new);
        
        let mut sum_c = 0.0f32;
        let mut sum1 = 0.0f32;
        let mut sum2 = 0.0f32;
        
        // Use current generation to identify previous states
        let prev_gen = self.generation;

        for &k in new_states {
            // O(1) Lookup
            let mut idx_prev = None;
            if (k as usize) < self.global_state_map.len() {
                let entry = self.global_state_map[k as usize];
                if entry.generation == prev_gen {
                    idx_prev = Some(entry.index as usize);
                }
            }
            
            let term_c = if let Some(idx) = idx_prev {
                self.prev_fwd_combined[idx] * stay + shift
            } else {
                shift 
            };
            
            let term1 = if let Some(idx) = idx_prev {
                self.prev_fwd1[idx] * stay + shift
            } else {
                shift
            };
            
            let term2 = if let Some(idx) = idx_prev {
                self.prev_fwd2[idx] * stay + shift
            } else {
                shift
            };
            
            // Multiply by emissions
            let (e_c, e_1, e_2) = emission_fn(k);
            
            let val_c = term_c * e_c;
            let val_1 = term1 * e_1;
            let val_2 = term2 * e_2;
            
            new_fwd_c.push(val_c);
            new_fwd1.push(val_1);
            new_fwd2.push(val_2);
            
            sum_c += val_c;
            sum1 += val_1;
            sum2 += val_2;
        }
        
        // Normalize
        let norm_c = if sum_c > 1e-30 { 1.0 / sum_c } else { 1.0 };
        let norm_1 = if sum1 > 1e-30 { 1.0 / sum1 } else { 1.0 };
        let norm_2 = if sum2 > 1e-30 { 1.0 / sum2 } else { 1.0 };
        
        for x in &mut new_fwd_c { *x *= norm_c; }
        for x in &mut new_fwd1 { *x *= norm_1; }
        for x in &mut new_fwd2 { *x *= norm_2; }
        
        // Update state
        self.prev_states = new_states.to_vec();
        self.prev_fwd_combined = new_fwd_c;
        self.prev_fwd1 = new_fwd1;
        self.prev_fwd2 = new_fwd2;

        // Advance generation and populate map for NEXT step
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            // Overflow handling: Clear the map to avoid collisions with old generation 0
            // Or just reset all to 0. Since 0 is now current, next check will fail unless we set it.
            // Wait, if we wrap to 0, anything with 0 (from long ago) might be considered valid.
            // But we actually only care about `prev_states`.
            // So we need to ensure that when we wrap, we invalidate all old entries.
            // Simplest way: if wrapped to 0, memset the whole array to (0,0) [invalid gen] or just skip 0?
            // Safer: Use generation > 0 always. 
            // If wrap to 0 -> set to 1, and clear the vector.
            // This happens once every 65k steps. Cheap enough.
            self.generation = 1;
            self.global_state_map.fill(StateEntry::default());
        }

        // Populate map for the NEW states which become the PREVIOUS states in the next call
        for (i, &hap) in self.prev_states.iter().enumerate() {
            // Safe because we resized above
            if (hap as usize) < self.global_state_map.len() {
                self.global_state_map[hap as usize] = StateEntry {
                    generation: self.generation,
                    index: i as u16,
                };
            }
        }
    }
    
    /// Calculate current phase likelihood ratio (log scale or probability)
    /// Returns magnitude of preference for current phase vs swap.
    /// Positive -> Keep current. Negative -> Swap.
    /// Based on HMM forward probabilities only (Online Filter).
    ///
    /// The forward variables represent P(X_1...m, S_m=s).
    /// To decide phase, we want P(Phase=0 | X) vs P(Phase=1 | X).
    /// With online filtering, we approximate this by marginalizing over current state S_m.
    /// L(Keep) = sum_s P(S_m=s for Hap1) * P(S_m=s for Hap2)? 
    /// No.
    /// We have fwd1[s] and fwd2[s].
    /// Hypothesis:
    ///   Phase 0 (Current): Hap1 generated by s1, Hap2 generated by s2 (consistent with current)
    ///   Phase 1 (Swap): Hap1 generated by s2, Hap2 generated by s1 (swapped assignment)
    /// BUT fwd1 track "Hap1 is constructed from reference s".
    /// If we swap alleles, we change the observation sequence going forward.
    /// This changes fwd1 and fwd2 values.
    /// So we can't just compare fwd1 and fwd2?
    ///
    /// Wait, fwd1[s] = P(Obs1 | State1=s).
    /// If we had swapped at previous steps, fwd1 would be different.
    /// The "Online Phase" decision implies we check if the *current marker* alleles should be swapped.
    /// Let a1, a2 be alleles at this marker.
    /// Option A (Current): Obs1=a1, Obs2=a2.
    /// Option B (Swap): Obs1=a2, Obs2=a1.
    ///
    /// We can compute the likelihood of this step given previous:
    /// L(OptA) = sum_{s1,s2} P(s1,s2 | prev) * P(a1|s1) * P(a2|s2)
    /// L(OptB) = sum_{s1,s2} P(s1,s2 | prev) * P(a2|s1) * P(a1|s2)
    ///
    /// Assuming independence of s1, s2 given prev (approximation):
    /// P(s1,s2 | prev) = P(s1|prev) * P(s2|prev)
    /// P(s|prev) = transition term calculated in `step` without emission.
    ///
    /// So we can compute this decision *before* committing the step?
    /// Yes, `step` currently fuses transition + emission.
    /// We should separate them or use a helper to query "what if".
    pub fn decide_phase_at_step(
        &self,
        states1: &[u32],
        states2: &[u32],
        a1: u8,
        a2: u8,
        get_ref_allele: impl Fn(u32) -> u8,
        p_recomb: f32,
        n_total_haps: usize,
        params: &ModelParams,
    ) -> bool {
        // Calculate transition priors for each new state
        // P(S_m = k | prev)
        let shift = p_recomb / n_total_haps as f32;
        let stay = 1.0 - p_recomb;
        
        // Compute L(A) and L(B)
        // L(A) = sum_s1 (P(s1)*E(a1|s1)) * sum_s2 (P(s2)*E(a2|s2))
        //      = TotalProb1(a1) * TotalProb2(a2)
        // L(B) = TotalProb1(a2) * TotalProb2(a1)
        
        let mut total_prob1_a1 = 0.0;
        let mut total_prob1_a2 = 0.0;
        let mut total_prob2_a1 = 0.0;
        let mut total_prob2_a2 = 0.0;
        
        let p_match = params.emit_match();
        let p_mismatch = params.emit_mismatch();
        
        let emit = |obs, ref_a| {
            if obs == 255 || ref_a == 255 { 1.0 }
            else if obs == ref_a { p_match }
            else { p_mismatch }
        };
        
        // Helper to process a state k
        let mut process_state = |k: u32| {
            let ref_a = get_ref_allele(k);
            let e_a1 = emit(a1, ref_a);
            let e_a2 = emit(a2, ref_a);
            
            // Calculate prior P(S_m=k | prev) using O(1) lookup
            let prev_gen = self.generation;
            let mut idx_prev = None;
            if (k as usize) < self.global_state_map.len() {
                // Safety: decide_phase relies on 'generation' but it assumes 'generation' corresponds to 'prev_states'.
                // 'step' updates generation at the END. So 'generation' field currently points to 'prev_states'.
                // Yes, initialized to 1 in init().
                // step() updates to new generation for NEXT step.
                // So current self.generation IS the generation for the current self.prev_states.
                
                let entry = self.global_state_map[k as usize];
                if entry.generation == prev_gen {
                    idx_prev = Some(entry.index as usize);
                }
            }
            
            let term1 = if let Some(idx) = idx_prev {
                self.prev_fwd1[idx] * stay + shift
            } else { shift };
            
            let term2 = if let Some(idx) = idx_prev {
                self.prev_fwd2[idx] * stay + shift
            } else { shift };
            
            total_prob1_a1 += term1 * e_a1;
            total_prob1_a2 += term1 * e_a2;
            total_prob2_a1 += term2 * e_a1;
            total_prob2_a2 += term2 * e_a2;
        };

        // Iterate union of states1 and states2
        for &k in states1 {
            process_state(k);
        }
        for &k in states2 {
            if !states1.contains(&k) {
                process_state(k);
            }
        }
        
        let likelihood_current = total_prob1_a1 * total_prob2_a2; // a1 on hap1, a2 on hap2
        let likelihood_swap = total_prob1_a2 * total_prob2_a1;    // a2 on hap1, a1 on hap2
        
        // Return true if swap is preferred
        likelihood_swap > likelihood_current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hmm_transition_correctness() {
        let mut hmm = OnlineHmm::new();
        let initial_states = vec![10, 20, 30];
        hmm.init(&initial_states);

        // Verify init
        assert_eq!(hmm.prev_states, initial_states);
        assert_eq!(hmm.global_state_map.len(), 31);
        assert_eq!(hmm.global_state_map[10].index, 0);
        assert_eq!(hmm.global_state_map[20].index, 1);
        assert_eq!(hmm.global_state_map[30].index, 2);
        assert_eq!(hmm.generation, 1);

        // Step 1: Some overlapping states, some new
        let new_states = vec![20, 30, 40];
        let p_recomb = 0.01;
        let n_total_haps = 100;

        hmm.step(
            &new_states,
            |_| (1.0, 1.0, 1.0), // All emissions 1.0 for simplicity
            p_recomb,
            n_total_haps,
        );

        // Check map expansion
        assert!(hmm.global_state_map.len() >= 100);
        
        // Verify previous states (which are now 'new_states') are mapped correctly in the NEW generation
        // After step(), generation is incremented (now 2).
        // The map should store indices for the states we just transitioned TO (which are now 'prev').
        assert_eq!(hmm.generation, 2);
        assert_eq!(hmm.prev_states, new_states);
        assert_eq!(hmm.global_state_map[20].generation, 2);
        assert_eq!(hmm.global_state_map[20].index, 0);
        assert_eq!(hmm.global_state_map[30].generation, 2);
        assert_eq!(hmm.global_state_map[30].index, 1);
        assert_eq!(hmm.global_state_map[40].generation, 2);
        assert_eq!(hmm.global_state_map[40].index, 2);
        
        // Old state 10 should still have old generation (1) or be untouched
        assert_eq!(hmm.global_state_map[10].generation, 1);
    }

    #[test]
    fn test_generation_overflow() {
        let mut hmm = OnlineHmm::new();
        let initial_states = vec![5];
        hmm.init(&initial_states);
        
        // Artificially set generation near overflow
        hmm.generation = 65535;
        
        // Update map manually to match this generation manually for state 5
        hmm.global_state_map[5] = StateEntry { generation: 65535, index: 0 };

        let new_states = vec![5, 6];
        hmm.step(
            &new_states,
            |_| (1.0, 1.0, 1.0),
            0.01,
            100
        );

        // Should have wrapped to 1
        assert_eq!(hmm.generation, 1);
        
        // Map should have been cleared/reset. 
        // State 5 is in new_states, so it should be present with new generation 1.
        assert_eq!(hmm.global_state_map[5].generation, 1);
        assert_eq!(hmm.global_state_map[5].index, 0);
        
        // State 6 is new, should be present
        assert_eq!(hmm.global_state_map[6].generation, 1);
        assert_eq!(hmm.global_state_map[6].index, 1);
        
        // Ensure some random other slot is empty/0
        assert_eq!(hmm.global_state_map[99].generation, 0);
    }

    #[test]
    fn test_step_without_init() {
        let mut hmm = OnlineHmm::new();
        // Should start with generation 1
        assert_eq!(hmm.generation, 1);
        
        let new_states = vec![10, 20];
        // This used to panic because generation 0 matched default map entries (0)
        hmm.step(
            &new_states,
            |_| (1.0, 1.0, 1.0),
            0.01,
            100
        );
        
        // After step, should be gen 2
        assert_eq!(hmm.generation, 2);
        assert_eq!(hmm.prev_states, new_states);
    }
}
