use crate::model::parameters::ModelParams;

/// Manages the HMM state for a single sample during online phasing.
/// Allows step-by-step updates as we sweep across markers.
#[derive(Debug, Clone)]
pub struct OnlineHmm {
    /// Active state indices (haplotype indices) from the previous step
    prev_states: Vec<u32>,
    
    /// Normalized forward probabilities for Combined (unphased-like) pass
    /// Corresponds to P(X | state)
    prev_fwd_combined: Vec<f32>,
    
    /// Normalized forward probabilities for Hap1
    prev_fwd1: Vec<f32>,
    
    /// Normalized forward probabilities for Hap2
    prev_fwd2: Vec<f32>,
}

impl OnlineHmm {
    /// Create a new OnlineHmm tracker
    pub fn new() -> Self {
        Self {
            prev_states: Vec::new(),
            prev_fwd_combined: Vec::new(),
            prev_fwd1: Vec::new(),
            prev_fwd2: Vec::new(),
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
        if n_new == 0 {
            self.prev_states.clear();
            self.prev_fwd_combined.clear();
            self.prev_fwd1.clear();
            self.prev_fwd2.clear();
            return;
        }

        // Calculate transition terms
        // T(j -> k) = (1-rho) * delta(j,k) + rho/N
        // P(S_m = k) = sum_j P(S_{m-1}=j) * T(j->k)
        //            = sum_j P(j) * (rho/N) + P(k)*(1-rho) [if k was active]
        //            = (rho/N) * sum_j P(j) + (1-rho) * P(prev=k)
        // Since P(j) are normalized, sum_j P(j) = 1.0 (approximately)
        // So P_trans(k) = (rho/N) + (1-rho) * P_prev(k)
        
        let shift = p_recomb / n_total_haps as f32;
        let stay = 1.0 - p_recomb; // Probability of NOT recombining globally
        // Note: Li-Stephens usually uses N_states in denominator. 
        // If we use N_total_haps, we account for transitions to non-active states implicitly?
        // Standard Li-Stephens: T(j->k) = r/N where N is number of *reference* haplotypes. 
        // Here n_total_haps is correct.
        
        // Build map of previous states to indices for O(1) matching
        // Since K is small (~100), linear scan or hash map is fine.
        // Actually, if we keep vectors sorted, we can merge.
        // But PBWT neighbors come sorted? Usually yes because PPA is sorted.
        // Let's assume sorted inputs to optimize matching in future.
        // For now, naive lookup or simple iterate.
        
        let mut new_fwd_c = Vec::with_capacity(n_new);
        let mut new_fwd1 = Vec::with_capacity(n_new);
        let mut new_fwd2 = Vec::with_capacity(n_new);
        
        let mut sum_c = 0.0f32;
        let mut sum1 = 0.0f32;
        let mut sum2 = 0.0f32;
        
        for &k in new_states {
            // Find prob of k in previous step
            // Use linear search for now (K is small)
            // Can optimize later.
            let idx_prev = self.prev_states.iter().position(|&p| p == k);
            
            let term_c = if let Some(idx) = idx_prev {
                self.prev_fwd_combined[idx] * stay + shift
            } else {
                shift // If not present, only recombination mass comes here
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
        // Avoid division by zero
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
            
            // Calculate prior P(S_m=k | prev) using same logic as step
            let idx_prev = self.prev_states.iter().position(|&p| p == k);
            
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
