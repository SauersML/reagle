
use reagle::model::pl_emission::allele_probs_uncond_from_pl;

#[test]
fn test_pl_0_0_0_uniformity() {
    let pl = vec![0u16, 0, 0];
    let mut probs = Vec::new();
    let n = allele_probs_uncond_from_pl(&pl, None, &mut probs).unwrap();
    assert_eq!(n, 2);
    println!("Probs: {:?}", probs);
    
    let is_uniform = {
        let mut min = probs[0];
        let mut max = probs[0];
        for &v in &probs {
            if v < min { min = v; }
            if v > max { max = v; }
        }
        (max - min) <= 1e-6
    };
    println!("Is uniform: {}", is_uniform);
    assert!(is_uniform);
}
