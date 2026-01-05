//! # Phasing Pipeline
//!
//! Orchestrates the phasing workflow:
//! 1. Load target VCF
//! 2. Build PBWT for haplotype matching
//! 3. Run PBWT-accelerated Li-Stephens HMM
//! 4. Update phase and iterate
//! 5. Write phased output
//!
//! This is the core of Beagle's phasing algorithm.


use rayon::prelude::*;

use crate::config::Config;
use crate::data::genetic_map::GeneticMaps;
use crate::data::haplotype::{HapIdx, SampleIdx};
use crate::data::marker::MarkerIdx;
use crate::data::storage::{GenotypeColumn, GenotypeMatrix, MutableGenotypes};
use crate::error::Result;
use crate::io::vcf::{VcfReader, VcfWriter};
use crate::model::parameters::ModelParams;
use crate::model::pbwt::{PbwtDivUpdater, PbwtIbs};
use crate::utils::Workspace;

/// Phasing pipeline
pub struct PhasingPipeline {
    config: Config,
    params: ModelParams,
}

impl PhasingPipeline {
    /// Create a new phasing pipeline
    pub fn new(config: Config) -> Self {
        let params = ModelParams::new();
        Self { config, params }
    }

    /// Run the phasing pipeline
    pub fn run(&mut self) -> Result<()> {
        eprintln!("Loading VCF...");
        
        // Load target VCF
        let (mut reader, file_reader) = VcfReader::open(&self.config.gt)?;
        let samples = reader.samples_arc();
        let target_gt = reader.read_all(file_reader)?;

        if target_gt.n_markers() == 0 {
            eprintln!("No markers found in input VCF");
            return Ok(());
        }

        let n_markers = target_gt.n_markers();
        let n_samples = target_gt.n_samples();
        let n_haps = target_gt.n_haplotypes();

        eprintln!("Loaded {} markers, {} samples ({} haplotypes)", n_markers, n_samples, n_haps);

        // Initialize parameters based on sample size
        self.params = ModelParams::for_phasing(n_haps);
        self.params.set_n_states(self.config.phase_states.min(n_haps.saturating_sub(2)));

        // Load genetic map if provided
        let gen_maps = if let Some(ref map_path) = self.config.map {
            let chrom_names: Vec<&str> = target_gt
                .markers()
                .chrom_names()
                .iter()
                .map(|s| s.as_ref())
                .collect();
            GeneticMaps::from_plink_file(map_path, &chrom_names)?
        } else {
            GeneticMaps::new()
        };

        // Create mutable genotype storage for phasing
        let mut geno = MutableGenotypes::from_fn(n_markers, n_haps, |m, h| {
            target_gt.allele(MarkerIdx::new(m as u32), HapIdx::new(h as u32))
        });

        // Compute genetic distances
        let chrom = target_gt.marker(MarkerIdx::new(0)).chrom;
        let gen_dists: Vec<f64> = (0..n_markers.saturating_sub(1))
            .map(|m| {
                let pos1 = target_gt.marker(MarkerIdx::new(m as u32)).pos;
                let pos2 = target_gt.marker(MarkerIdx::new((m + 1) as u32)).pos;
                gen_maps.gen_dist(chrom, pos1, pos2)
            })
            .collect();

        // Run phasing iterations
        let n_burnin = self.config.burnin;
        let n_iterations = self.config.iterations;
        let total_iterations = n_burnin + n_iterations;

        for it in 0..total_iterations {
            let is_burnin = it < n_burnin;
            let iter_type = if is_burnin { "burnin" } else { "main" };
            eprintln!("Iteration {}/{} ({})", it + 1, total_iterations, iter_type);
            
            self.run_iteration(&target_gt, &mut geno, &gen_dists, it)?;
        }

        // Build final GenotypeMatrix from mutable genotypes
        let final_gt = self.build_final_matrix(&target_gt, &geno);

        // Write output
        let output_path = self.config.out.with_extension("vcf.gz");
        eprintln!("Writing output to {:?}", output_path);
        
        let mut writer = VcfWriter::create(&output_path, samples)?;
        writer.write_header(final_gt.markers())?;
        writer.write_phased(&final_gt, 0, final_gt.n_markers())?;
        writer.flush()?;

        eprintln!("Phasing complete!");
        Ok(())
    }

    /// Run a single phasing iteration
    fn run_iteration(
        &self,
        target_gt: &GenotypeMatrix,
        geno: &mut MutableGenotypes,
        gen_dists: &[f64],
        iteration: usize,
    ) -> Result<()> {
        let n_samples = target_gt.n_samples();
        let n_markers = target_gt.n_markers();
        let n_haps = target_gt.n_haplotypes();
        let n_states = self.params.n_states;
        let seed = self.config.seed as u64 + iteration as u64;

        // Build PBWT for state selection using current phasing
        let pbwt = self.build_pbwt(geno, n_markers, n_haps);

        // Phase each sample in parallel
        let updates: Vec<(SampleIdx, Vec<usize>)> = (0..n_samples)
            .into_par_iter()
            .filter_map(|s| {
                let sample_idx = SampleIdx::new(s as u32);
                let hap1 = sample_idx.hap1();
                let hap2 = sample_idx.hap2();

                // Get current alleles for this sample
                let alleles1 = geno.haplotype(hap1);
                let alleles2 = geno.haplotype(hap2);

                // Find heterozygous markers
                let het_markers: Vec<usize> = (0..n_markers)
                    .filter(|&m| alleles1[m] != alleles2[m])
                    .collect();

                if het_markers.is_empty() {
                    return None; // Nothing to phase
                }

                // Select reference haplotypes using PBWT (excluding this sample's haps)
                let ref_haps = self.select_ref_haps(&pbwt, hap1, n_states, n_haps);

                // Create workspace for this thread
                let mut workspace = Workspace::new(n_states, n_markers, n_haps);
                workspace.set_seed(seed + s as u64);

                // Build reference panel view for HMM
                let ref_alleles: Vec<Vec<u8>> = ref_haps
                    .iter()
                    .map(|&h| geno.haplotype(h))
                    .collect();

                // Run phasing HMM
                let switch_markers = self.phase_sample(
                    &alleles1,
                    &alleles2,
                    &het_markers,
                    &ref_alleles,
                    gen_dists,
                    &mut workspace,
                );

                if switch_markers.is_empty() {
                    None
                } else {
                    Some((sample_idx, switch_markers))
                }
            })
            .collect();

        // Apply phase switches
        let mut n_switches = 0;
        for (sample_idx, switch_markers) in updates {
            let hap1 = sample_idx.hap1();
            let hap2 = sample_idx.hap2();
            
            for &m in &switch_markers {
                geno.swap(m, hap1, hap2);
            }
            n_switches += switch_markers.len();
        }

        eprintln!("  Applied {} phase switches", n_switches);
        Ok(())
    }

    /// Build PBWT from current genotypes
    fn build_pbwt(&self, geno: &MutableGenotypes, n_markers: usize, n_haps: usize) -> PbwtIbs {
        let mut pbwt = PbwtIbs::new(n_haps);
        let mut updater = PbwtDivUpdater::new(n_haps);

        for m in 0..n_markers {
            let alleles = geno.marker_alleles(m);
            
            // Determine number of alleles (usually 2 for biallelic)
            let n_alleles = alleles.iter().copied().max().unwrap_or(0) as usize + 1;

            let mut temp_prefix = pbwt.fwd_prefix().to_vec();
            let mut temp_div = pbwt.fwd_divergence().to_vec();

            updater.update(alleles, n_alleles.max(2), m, &mut temp_prefix, &mut temp_div);

            pbwt.fwd_prefix_mut().copy_from_slice(&temp_prefix);
            pbwt.fwd_divergence_mut().copy_from_slice(&temp_div);
        }

        pbwt
    }

    /// Select reference haplotypes using PBWT
    fn select_ref_haps(
        &self,
        pbwt: &PbwtIbs,
        target_hap: HapIdx,
        n_states: usize,
        n_haps: usize,
    ) -> Vec<HapIdx> {
        // Get the sample this haplotype belongs to
        let _target_sample = target_hap.0 / 2;
        let other_hap = if target_hap.0 % 2 == 0 {
            HapIdx::new(target_hap.0 + 1)
        } else {
            HapIdx::new(target_hap.0 - 1)
        };

        // Use PBWT to find nearby haplotypes
        let mut selected = pbwt.select_states(target_hap, n_states + 2, true);
        
        // Remove the other haplotype from the same sample
        selected.retain(|&h| h != other_hap);
        
        // Limit to n_states
        selected.truncate(n_states);

        // If we don't have enough, add random haplotypes
        if selected.len() < n_states {
            for h in 0..n_haps as u32 {
                let hap = HapIdx::new(h);
                if hap != target_hap && hap != other_hap && !selected.contains(&hap) {
                    selected.push(hap);
                    if selected.len() >= n_states {
                        break;
                    }
                }
            }
        }

        selected
    }

    /// Phase a single sample using HMM
    fn phase_sample(
        &self,
        alleles1: &[u8],
        alleles2: &[u8],
        het_markers: &[usize],
        ref_alleles: &[Vec<u8>],
        gen_dists: &[f64],
        workspace: &mut Workspace,
    ) -> Vec<usize> {
        let _n_markers = alleles1.len();
        let n_states = ref_alleles.len();

        if n_states == 0 || het_markers.is_empty() {
            return Vec::new();
        }

        // Compute forward probabilities for haplotype 1
        let fwd1 = self.forward_pass(alleles1, ref_alleles, gen_dists, workspace);
        
        // Compute forward probabilities for haplotype 2
        let fwd2 = self.forward_pass(alleles2, ref_alleles, gen_dists, workspace);

        // Decide phase switches based on likelihood comparison
        let mut switch_markers = Vec::new();
        
        // Simple phase decision: at each het site, check if swapping improves likelihood
        for &m in het_markers {
            // Current assignment likelihood (sum over states)
            let _curr_ll = self.log_sum_exp(&fwd1[m]) + self.log_sum_exp(&fwd2[m]);
            
            // For swap, we'd need to recompute with swapped alleles
            // Simplified: use emission probability comparison
            let a1 = alleles1[m];
            let a2 = alleles2[m];
            
            let mut curr_match = 0.0f32;
            let mut swap_match = 0.0f32;
            
            for (s, ref_hap) in ref_alleles.iter().enumerate() {
                let ref_a = ref_hap[m];
                let weight = fwd1[m][s].exp() + fwd2[m][s].exp();
                
                if a1 == ref_a {
                    curr_match += weight;
                }
                if a2 == ref_a {
                    swap_match += weight;
                }
            }
            
            // Random component for exploration (especially in burnin)
            let rand_val = workspace.next_f32();
            let threshold = 0.5 + 0.3 * (swap_match - curr_match) / (swap_match + curr_match + 1e-10);
            
            if rand_val > threshold && swap_match > curr_match {
                switch_markers.push(m);
            }
        }

        switch_markers
    }

    /// Forward pass of HMM
    fn forward_pass(
        &self,
        target_alleles: &[u8],
        ref_alleles: &[Vec<u8>],
        gen_dists: &[f64],
        _workspace: &mut Workspace,
    ) -> Vec<Vec<f32>> {
        let n_markers = target_alleles.len();
        let n_states = ref_alleles.len();

        let mut fwd = vec![vec![0.0f32; n_states]; n_markers];

        // Initialize
        let init_log_prob = -(n_states as f32).ln();
        for s in 0..n_states {
            let emit = self.emission_prob(target_alleles[0], ref_alleles[s][0]);
            fwd[0][s] = init_log_prob + emit.ln();
        }

        // Forward recursion
        for m in 1..n_markers {
            let gen_dist = gen_dists.get(m - 1).copied().unwrap_or(0.01);
            let p_switch = self.params.switch_prob(gen_dist);
            let p_stay = 1.0 - p_switch;
            let p_switch_to = p_switch / n_states as f32;

            let log_fwd_sum = self.log_sum_exp(&fwd[m - 1]);

            for s in 0..n_states {
                let emit = self.emission_prob(target_alleles[m], ref_alleles[s][m]);
                let term1 = p_stay.ln() + fwd[m - 1][s];
                let term2 = p_switch_to.ln() + log_fwd_sum;
                let trans = self.log_sum_exp(&[term1, term2]);
                fwd[m][s] = trans + emit.ln();
            }
        }

        fwd
    }

    /// Calculates log(sum(exp(a))) for a slice of log-probabilities.
    fn log_sum_exp(&self, a: &[f32]) -> f32 {
        let max_val = a.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if max_val.is_infinite() {
            return max_val;
        }
        let sum = a.iter().map(|&x| (x - max_val).exp()).sum::<f32>();
        max_val + sum.ln()
    }

    /// Emission probability
    #[inline]
    fn emission_prob(&self, target: u8, reference: u8) -> f32 {
        if target == reference {
            self.params.emit_match()
        } else {
            self.params.emit_mismatch()
        }
    }

    /// Build final GenotypeMatrix from mutable genotypes
    fn build_final_matrix(
        &self,
        original: &GenotypeMatrix,
        geno: &MutableGenotypes,
    ) -> GenotypeMatrix {
        let markers = original.markers().clone();
        let samples = original.samples_arc();
        let n_markers = geno.n_markers();

        let columns: Vec<GenotypeColumn> = (0..n_markers)
            .map(|m| {
                let alleles = geno.marker_alleles(m);
                GenotypeColumn::from_alleles(alleles, 2)
            })
            .collect();

        GenotypeMatrix::new(markers, columns, samples, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_pipeline_creation() {
        let config = Config {
            gt: PathBuf::from("test.vcf"),
            r#ref: None,
            out: PathBuf::from("out"),
            map: None,
            chrom: None,
            excludesamples: None,
            excludemarkers: None,
            burnin: 3,
            iterations: 12,
            phase_states: 280,
            rare: 0.002,
            impute: true,
            imp_states: 1600,
            imp_segment: 6.0,
            imp_step: 0.1,
            imp_nsteps: 7,
            cluster: 0.005,
            ap: false,
            gp: false,
            ne: 100000.0,
            err: None,
            em: true,
            window: 40.0,
            window_markers: 4000000,
            overlap: 2.0,
            seed: 12345,
            nthreads: None,
        };

        let pipeline = PhasingPipeline::new(config);
        assert_eq!(pipeline.params.n_states, 280);
    }

    #[test]
    fn test_emission_prob() {
        let config = Config {
            gt: PathBuf::from("test.vcf"),
            r#ref: None,
            out: PathBuf::from("out"),
            map: None,
            chrom: None,
            excludesamples: None,
            excludemarkers: None,
            burnin: 3,
            iterations: 12,
            phase_states: 280,
            rare: 0.002,
            impute: true,
            imp_states: 1600,
            imp_segment: 6.0,
            imp_step: 0.1,
            imp_nsteps: 7,
            cluster: 0.005,
            ap: false,
            gp: false,
            ne: 100000.0,
            err: None,
            em: true,
            window: 40.0,
            window_markers: 4000000,
            overlap: 2.0,
            seed: 12345,
            nthreads: None,
        };

        let pipeline = PhasingPipeline::new(config);
        
        // Match should have high probability
        let match_prob = pipeline.emission_prob(0, 0);
        assert!(match_prob > 0.99);
        
        // Mismatch should have low probability
        let mismatch_prob = pipeline.emission_prob(0, 1);
        assert!(mismatch_prob < 0.01);
    }
}