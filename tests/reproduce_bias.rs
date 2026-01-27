
#[cfg(test)]
mod tests {
    use reagle::model::phase_states::PhaseStates;
    use reagle::model::block_hash::types::GlobalId;

    #[test]
    fn test_eviction_bias_repro() {
        // Setup distinct pools: small capacity
        let max_states = 10;
        let n_markers = 100;
        let mut ps = PhaseStates::new(max_states, n_markers);

        // Sets of "perfect match" reference haplotypes
        // Set A: 8 items
        let set_a: Vec<u32> = (0..8).collect(); 
        // Set B: 8 items
        let set_b: Vec<u32> = (10..18).collect(); 

        // Total 16 items > 10 capacity

        // Simulate Streaming
        for m in 0..n_markers {
            // Force "H2 after H1" insertion repeatedly
            ps.add_neighbors_at_marker(999, m, &set_a, &set_b);
        }

        // Inspect Result
        let th = ps.finalize_streaming(999, 1000);

        let mut count_a = 0;
        let mut count_b = 0;
        let mut buffer = vec![GlobalId::from(0u32); th.n_states()];

        // Check reference haplotype used at last marker
        th.materialize_at(n_markers - 1, &mut buffer);

        println!(
            "Capacity: {}, H1 Set size: {}, H2 Set size: {}",
            max_states,
            set_a.len(),
            set_b.len()
        );
        println!("State | Final Ref Hap ID | Origin");
        println!("---------------------------------");

        for (i, &hap_id) in buffer.iter().enumerate() {
            let id = hap_id.as_u32();
            let origin = if set_a.contains(&id) {
                count_a += 1;
                "Set A (H1)"
            } else if set_b.contains(&id) {
                count_b += 1;
                "Set B (H2)"
            } else {
                "Unknown"
            };
            println!("{:<5} | {:<16} | {}", i, id, origin);
        }

        println!(
            "Final Counts -> H1_Matches: {}, H2_Matches: {}",
            count_a, count_b
        );

        // Assert fairness
        assert!(
            count_a > 2,
            "Eviction bias detected! H1 matches (Set A) were evicted. count_a={}, count_b={}",
            count_a,
            count_b
        );
         assert!(
            count_b > 2,
            "Eviction bias detected! H2 matches (Set B) were evicted. count_a={}, count_b={}",
            count_a,
            count_b
        );
    }
}
