
use reagle::data::marker::{MarkerIdx, AnyMarkerSpace};
use reagle::data::haplotype::HapIdx;
use reagle::data::storage::GenotypeView;
use reagle::pipelines::phasing::{RefAlleleProvider, MosaicPaths};
use reagle::model::states::ThreadedHaps;
use reagle::model::types::{CombinedHapId, CombinedHapSpace};
use rand::{Rng, SeedableRng};

// Mock RefAlleleProvider
struct MockRef {
    alleles: Vec<Vec<u8>>, // [state][marker]
}

impl MockRef {
    fn fill(&self, m: usize, out: &mut [u8]) {
        for i in 0..self.alleles.len() {
            out[i] = self.alleles[i][m];
        }
    }
}

// Copy of Original Logic
fn original_logic(
    n_markers: usize,
    n_states: usize,
    seq1: &[u8],
    seq2: &[u8],
    mock_ref: &MockRef,
) -> (Vec<f32>, Option<MosaicPaths>) {
    let mut scores = vec![0.0f32; n_states * n_states];
    let mut ref_alleles = vec![255u8; n_states];
    let mut informative = 0usize;

    for m in 0..n_markers {
        let a1 = seq1[m];
        let a2 = seq2[m];
        if a1 == 255 && a2 == 255 {
            continue;
        }
        informative += 1;

        let is_het = a1 != a2 && a1 != 255 && a2 != 255;

        mock_ref.fill(m, &mut ref_alleles);
        for i in 0..n_states {
            let r1 = ref_alleles[i];
            if r1 == 255 {
                continue;
            }

            for j in 0..i {
                let r2 = ref_alleles[j];
                if r2 == 255 {
                    continue;
                }

                let compatible = if is_het {
                    (r1 == a1 && r2 == a2) || (r1 == a2 && r2 == a1)
                } else {
                    let obs = if a1 != 255 { a1 } else { a2 };
                    r1 == obs && r2 == obs
                };

                if compatible {
                    scores[i * n_states + j] += 1.0;
                } else {
                    scores[i * n_states + j] -= 1.0;
                }
            }
        }
    }

    let mut best_score = f32::NEG_INFINITY;
    let mut best_pair = (0, 1);

    for i in 0..n_states {
        for j in 0..i {
            let s = scores[i * n_states + j];
            if s > best_score {
                best_score = s;
                best_pair = (i, j);
            }
        }
    }

    if informative == 0 { return (scores, None); }
    let path1 = vec![best_pair.0 as u32; n_markers];
    let path2 = vec![best_pair.1 as u32; n_markers];
    (scores, Some(MosaicPaths { path1, path2 }))
}

// Copy of New Logic
fn new_logic(
    n_markers: usize,
    n_states: usize,
    seq1: &[u8],
    seq2: &[u8],
    mock_ref: &MockRef,
) -> (Vec<f32>, Option<MosaicPaths>) {
    let mut scores = vec![0.0f32; n_states * n_states];
    let mut ref_alleles = vec![255u8; n_states];
    let mut informative = 0usize;

    let n_words = (n_markers + 63) / 64;
    let mut match_a1 = vec![vec![0u64; n_words]; n_states];
    let mut match_a2 = vec![vec![0u64; n_words]; n_states];
    let mut valid_ref = vec![vec![0u64; n_words]; n_states];

    let mut is_het = vec![0u64; n_words];
    let mut is_valid = vec![0u64; n_words];

    for m in 0..n_markers {
        let a1 = seq1[m];
        let a2 = seq2[m];
        if a1 == 255 && a2 == 255 {
            continue;
        }
        informative += 1;

        let word_idx = m / 64;
        let bit_mask = 1u64 << (m % 64);

        is_valid[word_idx] |= bit_mask;
        if a1 != a2 && a1 != 255 && a2 != 255 {
            is_het[word_idx] |= bit_mask;
        }

        mock_ref.fill(m, &mut ref_alleles);
        for i in 0..n_states {
            let r = ref_alleles[i];
            if r != 255 {
                valid_ref[i][word_idx] |= bit_mask;

                let (t1, t2) = if a1 != 255 && a2 != 255 && a1 != a2 {
                    (a1, a2)
                } else {
                    let obs = if a1 != 255 { a1 } else { a2 };
                    (obs, obs)
                };

                if r == t1 {
                    match_a1[i][word_idx] |= bit_mask;
                }
                if r == t2 {
                    match_a2[i][word_idx] |= bit_mask;
                }
            }
        }
    }

    let mut best_score = f32::NEG_INFINITY;
    let mut best_pair = (0, 1);

    for i in 0..n_states {
        for j in 0..i {
            let mut score = 0i32;
            for w in 0..n_words {
                if is_valid[w] == 0 {
                    continue;
                }

                let het = is_het[w];
                let hom = is_valid[w] & !het;

                let val_i = valid_ref[i][w];
                let val_j = valid_ref[j][w];
                let val_pair = val_i & val_j;

                let valid_mask = val_pair & is_valid[w];
                if valid_mask == 0 {
                    continue;
                }

                let m1_i = match_a1[i][w];
                let m2_i = match_a2[i][w];
                let m1_j = match_a1[j][w];
                let m2_j = match_a2[j][w];

                let het_matches = ((m1_i & m2_j) | (m2_i & m1_j)) & het;
                let hom_matches = (m1_i & m1_j) & hom;

                let matches = (het_matches | hom_matches) & valid_mask;
                let match_count = matches.count_ones() as i32;

                let total_count = valid_mask.count_ones() as i32;
                score += 2 * match_count - total_count;
            }

            let s = score as f32;
            if s > best_score {
                best_score = s;
                best_pair = (i, j);
            }
            scores[i * n_states + j] = s;
        }
    }

    if informative == 0 { return (scores, None); }
    let path1 = vec![best_pair.0 as u32; n_markers];
    let path2 = vec![best_pair.1 as u32; n_markers];
    (scores, Some(MosaicPaths { path1, path2 }))
}

fn main() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let n_markers = 100;
    let n_states = 10;

    let mut seq1 = vec![0u8; n_markers];
    let mut seq2 = vec![0u8; n_markers];
    for i in 0..n_markers {
        let r = rng.gen_range(0..10);
        if r < 1 {
            seq1[i] = 255;
            seq2[i] = 255;
        } else if r < 3 {
            seq1[i] = 0;
            seq2[i] = 1;
        } else {
            seq1[i] = 0;
            seq2[i] = 0;
        }
    }

    let mut alleles = vec![vec![0u8; n_markers]; n_states];
    for s in 0..n_states {
        for m in 0..n_markers {
            let r = rng.gen_range(0..10);
            if r < 1 {
                alleles[s][m] = 255;
            } else if r < 5 {
                alleles[s][m] = 0;
            } else {
                alleles[s][m] = 1;
            }
        }
    }

    let mock_ref = MockRef { alleles };

    let (s1, p1) = original_logic(n_markers, n_states, &seq1, &seq2, &mock_ref);
    let (s2, p2) = new_logic(n_markers, n_states, &seq1, &seq2, &mock_ref);

    for i in 0..n_states {
        for j in 0..i {
            let diff = (s1[i*n_states+j] - s2[i*n_states+j]).abs();
            if diff > 1e-5 {
                println!("Mismatch at ({}, {}): orig={} new={}", i, j, s1[i*n_states+j], s2[i*n_states+j]);
                panic!("Scores mismatch!");
            }
        }
    }

    println!("Scores match!");

    if let (Some(pp1), Some(pp2)) = (p1, p2) {
        if pp1.path1[0] != pp2.path1[0] || pp1.path2[0] != pp2.path2[0] {
             panic!("Best pair mismatch!");
        }
    }
    println!("Best pair matches!");
}
