
#[cfg(test)]
mod reproduction_test {
    use super::*;
    use crate::model::hmm::HmmUpdater;

    #[test]
    fn test_bwd_update_constant_normalization() {
        let n_states = 2;
        let mut bwd = vec![1.0, 1.0];
        let p_switch = 0.1;
        let emissions = vec![1.0, 1.0];
        
        // Constant term C = sum(bwd * emissions) = 1*1 + 1*1 = 2
        let constant_term = 2.0;

        // Formula: bwd[i] = (1-r)*e[i]*bwd[i] + (r/N)*C
        // bwd[i] = (0.9)*1*1 + (0.1/2)*2 = 0.9 + 0.1 = 1.0

        HmmUpdater::bwd_update_constant(&mut bwd, p_switch, &emissions, constant_term, n_states);

        // Current buggy implementation:
        // bwd[i] = (1-r)*e[i]*bwd[i] + r*C
        // bwd[i] = 0.9*1*1 + 0.1*2 = 0.9 + 0.2 = 1.1

        assert!((bwd[0] - 1.0).abs() < 1e-6, "Expected 1.0, got {}", bwd[0]);
    }
}
