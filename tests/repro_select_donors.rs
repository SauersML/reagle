
#[cfg(test)]
mod tests {
    use reagle::model::reference_pbwt::{RankBeam, ReferencePbwt};

    #[test]
    fn test_select_donors_bias() {
        // Create a PBWT with 100 haplotypes
        let n_ref = 100;
        let pbwt = ReferencePbwt::new(n_ref);

        // Create a beam with one interval [0, 100)
        // i.e., all haplotypes are candidates.
        let beam = RankBeam::full(n_ref as u32);
        
        // We want to select k=10 donors.
        let k = 10;
        let donors = pbwt.select_donors(&beam, k);

        println!("Donors: {:?}", donors);

        // Current implementation expands from center (50).
        // Expect donors to be around 45-55.
        // Uniform sampling should be 0, 10, 20, ... 90 (approx).

        let has_low = donors.iter().any(|&x| x < 10);
        let has_high = donors.iter().any(|&x| x > 90);
        
        assert!(has_low, "Should select from low range (uniform). Got: {:?}", donors);
        assert!(has_high, "Should select from high range (uniform). Got: {:?}", donors);
    }
}
