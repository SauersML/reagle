//! Minimal MCMC traits vendored to remove broken dependency.

/// A trace of the MCMC state.
pub trait Trace {
    fn trace(&self) -> Vec<f64>;
}

/// A Markov Chain Monte Carlo sampler.
pub trait MarkovChain<T: Trace> {
    fn step(&mut self) -> &T;
    fn current_state(&self) -> &T;
}
