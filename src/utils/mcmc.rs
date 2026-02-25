pub trait Trace {
    fn trace(&self) -> Vec<f64>;
}

pub trait MarkovChain<T: Trace> {
    fn step(&mut self) -> &T;
    fn current_state(&self) -> &T;
}
