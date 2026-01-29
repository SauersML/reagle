import sys

filepath = "src/model/reference_pbwt.rs"

with open(filepath, "r") as f:
    lines = f.readlines()

new_lines = []
in_struct = False
inv_ppa_added = False

for line in lines:
    new_lines.append(line)
    if "pub struct ReferencePbwt {" in line:
        in_struct = True
    if in_struct and "offsets: Vec<u32>," in line and not inv_ppa_added:
        new_lines.append("    inv_ppa: Vec<u32>,\n")
        inv_ppa_added = True
        in_struct = False

# Update new()
lines = new_lines
new_lines = []
in_new = False
inv_ppa_init_added = False

for line in lines:
    new_lines.append(line)
    if "pub fn new(n_ref_haps: usize) -> Self {" in line:
        in_new = True
    if in_new and "offsets: Vec::new()," in line and not inv_ppa_init_added:
        new_lines.append("            inv_ppa: (0..n_ref_haps as u32).collect(),\n")
        inv_ppa_init_added = True
        in_new = False

# Update with_state()
lines = new_lines
new_lines = []
in_with_state = False
inv_ppa_state_added = False

for line in lines:
    new_lines.append(line)
    if "pub fn with_state(n_ref_haps: usize, state: Option<&PbwtState>) -> Self {" in line:
        in_with_state = True
    if in_with_state and "pbwt.div = state.div.clone();" in line and not inv_ppa_state_added:
        new_lines.append("                pbwt.update_inv_ppa();\n")
        inv_ppa_state_added = True
        in_with_state = False

# Add update_inv_ppa method
lines = new_lines
new_lines = []
for line in lines:
    new_lines.append(line)
    if "pub fn get_state(&self, marker_pos: usize) -> PbwtState {" in line:
        new_lines.insert(len(new_lines) - 1, """    fn update_inv_ppa(&mut self) {
        if self.inv_ppa.len() != self.ppa.len() {
            self.inv_ppa.resize(self.ppa.len(), 0);
        }
        for (i, &hap) in self.ppa.iter().enumerate() {
            if (hap as usize) < self.inv_ppa.len() {
                self.inv_ppa[hap as usize] = i as u32;
            }
        }
    }

""")

# Update finalize_step to call update_inv_ppa
lines = new_lines
new_lines = []
in_finalize = False
for line in lines:
    new_lines.append(line)
    if "pub fn finalize_step(&mut self, ref_alleles: &[u8], n_alleles: usize, marker: usize) {" in line:
        in_finalize = True
    if in_finalize and "self.updater" in line:
        pass
    if in_finalize and ".fwd_update(ref_alleles, n_alleles, marker, &mut self.ppa, &mut self.div);" in line:
        new_lines.append("        self.update_inv_ppa();\n")
        in_finalize = False

# Update advance_with_rephase signature and logic
lines = new_lines
new_lines = []
in_advance = False
hints_arg_added = False
logic_replaced = False

i = 0
while i < len(lines):
    line = lines[i]
    if "pub fn advance_with_rephase" in line:
        in_advance = True
        new_lines.append(line)
        i += 1
        continue
    
    if in_advance and "swaps_out: &mut [bool]," in line and not hints_arg_added:
        new_lines.append(line)
        new_lines.append("        hints: Option<&[u32]>,\n")
        hints_arg_added = True
        i += 1
        continue

    if in_advance and "let score_keep =" in line:
        # Replace the scoring logic block
        new_lines.append("""                // Smoothed Consistency Scoring: len / (count + 1)
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
""")
        # Skip original lines until if score_swap > score_keep
        while "if score_swap > score_keep {" not in lines[i]:
            i += 1
        new_lines.append(lines[i]) # append the if line
        i += 1
        continue

    new_lines.append(line)
    i += 1

with open(filepath, "w") as f:
    f.writelines(new_lines)
