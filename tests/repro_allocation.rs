
use reagle::model::parameters::ModelParams;
use reagle::model::state_allocator::allocate_lms_sparse;

#[test]
fn repro_allocator_issue() {
    let n_ref_haps = 100;
    let num_windows = 5;
    let mut candidate_haps = Vec::new();
    let mut scores_by_hap = Vec::new();
    
    // Add "Hero" hap (98)
    candidate_haps.push(98);
    // Give it good scores in all windows
    scores_by_hap.push(vec![(0, 12.0), (1, 12.0), (2, 12.0), (3, 12.0), (4, 12.0)]);
    
    // Add "Noise" haps.
    // 15 haps with BETTER score (14.0)
    for i in 0..15 {
        if i == 98 { continue; }
        candidate_haps.push(i);
        let s = 14.0;
        scores_by_hap.push(vec![(0, s), (1, s), (2, s), (3, s), (4, s)]);
    }
    
    // 35 haps with WORSE score (10.0)
    for i in 15..50 {
        if i == 98 { continue; }
        candidate_haps.push(i);
        let s = 10.0;
        scores_by_hap.push(vec![(0, s), (1, s), (2, s), (3, s), (4, s)]);
    }
    
    // Hero is rank 16. With cap 20, it should be selected.
    
    let params = ModelParams::for_phasing(n_ref_haps + 2, 10000.0, Some(0.0001));
    let boundary_cm = vec![0.1; 4];
    let global_slot_budget = 100; // 20 * 5
    let per_window_caps = vec![20; 5];
    
    let alloc1 = allocate_lms_sparse(
        &scores_by_hap,
        &candidate_haps,
        num_windows,
        &boundary_cm,
        &params,
        n_ref_haps,
        global_slot_budget,
        &per_window_caps,
    );
    let selected1: Vec<usize> = alloc1.intervals_by_hap.iter().map(|(h, _)| *h).collect();
    println!("Selected1 (without anchors): len={} contains_hero={}", selected1.len(), selected1.contains(&98));
    assert!(selected1.contains(&98), "Hero should be selected initially");
    
    // Now simulate "Actual" state (with anchors)
    // Anchors add more candidates with EMPTY scores.
    // Add 30 anchors.
    
    let mut candidate_haps2 = candidate_haps.clone();
    let mut scores_by_hap2 = scores_by_hap.clone();
    
    for i in 50..80 {
        if !candidate_haps2.contains(&i) {
            candidate_haps2.push(i);
            scores_by_hap2.push(Vec::new()); // Empty scores!
        }
    }
    
    let alloc2 = allocate_lms_sparse(
        &scores_by_hap2,
        &candidate_haps2,
        num_windows,
        &boundary_cm,
        &params,
        n_ref_haps,
        global_slot_budget,
        &per_window_caps,
    );
    let selected2: Vec<usize> = alloc2.intervals_by_hap.iter().map(|(h, _)| *h).collect();
    println!("Selected2 (with anchors): len={} contains_hero={}", selected2.len(), selected2.contains(&98));
    
    assert!(selected2.contains(&98), "Hero should still be selected with anchors");
}
