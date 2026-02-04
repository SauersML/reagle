//! State-sized buffer helpers.
//!
//! These types encode the invariant that per-state buffers are never shorter
//! than the number of HMM states. Construction performs the only sizing checks.

use aligned_vec::{AVec, ConstAlign};
use std::num::NonZeroUsize;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StateCount(NonZeroUsize);

impl StateCount {
    #[inline]
    pub unsafe fn new_unchecked(n: usize) -> Self {
        // SAFETY: caller guarantees n > 0.
        Self(unsafe { NonZeroUsize::new_unchecked(n) })
    }

    #[inline]
    pub fn get(self) -> usize {
        self.0.get()
    }
}

#[derive(Clone, Debug)]
pub struct StateVec<T> {
    n: StateCount,
    data: Vec<T>,
}

impl<T: Clone> StateVec<T> {
    #[inline]
    pub fn new(n: StateCount, value: T) -> Self {
        let len = n.get();
        Self {
            n,
            data: vec![value; len],
        }
    }

    #[inline]
    pub fn from_vec(n: StateCount, mut data: Vec<T>, fill: T) -> Self {
        let len = n.get();
        if data.len() < len {
            data.resize(len, fill);
        }
        Self { n, data }
    }
}

impl<T> StateVec<T> {
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        let len = self.n.get();
        &self.data[..len]
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        let len = self.n.get();
        &mut self.data[..len]
    }

    #[inline]
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
}

#[derive(Clone, Debug)]
pub struct StateAVec32<T> {
    n: StateCount,
    data: AVec<T, ConstAlign<32>>,
}

impl<T: Clone> StateAVec32<T> {
    #[inline]
    pub fn new(n: StateCount, value: T) -> Self {
        let len = n.get();
        let data = AVec::from_iter(32, std::iter::repeat(value).take(len));
        Self { n, data }
    }

    #[inline]
    pub fn from_avec(n: StateCount, mut data: AVec<T, ConstAlign<32>>, fill: T) -> Self {
        let len = n.get();
        if data.len() < len {
            data = AVec::from_iter(32, std::iter::repeat(fill).take(len));
        }
        Self { n, data }
    }
}

impl<T> StateAVec32<T> {
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        let len = self.n.get();
        &mut self.data[..len]
    }

    #[inline]
    pub fn into_avec(self) -> AVec<T, ConstAlign<32>> {
        self.data
    }
}
