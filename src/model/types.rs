//! # Type-Safe Index Types for Block-Hash HMM
//!
//! This module defines strict newtype wrappers to prevent index confusion bugs.
//! Using distinct types for Global Haplotype IDs vs. Local Pattern IDs ensures
//! that the compiler prevents accidental misuse at compile time.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Global haplotype index (0..N_ref_haplotypes)
///
/// This identifies a specific haplotype in the reference panel across all windows.
/// It remains stable throughout the entire chromosome and enables tracking
/// probability flow through the correct physical DNA molecules.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct GlobalId(pub u32);

impl GlobalId {
    #[inline]
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    #[inline]
    pub fn as_usize(self) -> usize {
        self.0 as usize
    }

    #[inline]
    pub fn as_u32(self) -> u32 {
        self.0
    }
}

impl fmt::Debug for GlobalId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GlobalId({})", self.0)
    }
}

impl fmt::Display for GlobalId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "G{}", self.0)
    }
}

impl From<u32> for GlobalId {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl From<usize> for GlobalId {
    fn from(id: usize) -> Self {
        Self(id as u32)
    }
}

/// Local unique pattern index within a window (0..U_unique_patterns)
///
/// This identifies a unique haplotype pattern within a specific micro-window.
/// Patterns are local to each window and may have different meanings across windows.
/// The special RESERVOIR value indicates patterns that were truncated.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PatternId(pub u32);

impl PatternId {
    /// Sentinel value indicating a haplotype is in the reservoir (truncated)
    pub const RESERVOIR: Self = PatternId(u32::MAX);

    #[inline]
    pub fn new(id: u32) -> Self {
        assert!(
            id != u32::MAX,
            "Use PatternId::RESERVOIR for sentinel value"
        );
        Self(id)
    }

    #[inline]
    pub fn is_reservoir(self) -> bool {
        self == Self::RESERVOIR
    }

    #[inline]
    pub fn as_usize(self) -> usize {
        self.0 as usize
    }

    #[inline]
    pub fn as_u32(self) -> u32 {
        self.0
    }
}

impl fmt::Debug for PatternId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_reservoir() {
            write!(f, "PatternId(RESERVOIR)")
        } else {
            write!(f, "PatternId({})", self.0)
        }
    }
}

impl fmt::Display for PatternId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_reservoir() {
            write!(f, "R")
        } else {
            write!(f, "P{}", self.0)
        }
    }
}

impl From<u32> for PatternId {
    fn from(id: u32) -> Self {
        if id == u32::MAX {
            Self::RESERVOIR
        } else {
            Self(id)
        }
    }
}

impl From<usize> for PatternId {
    fn from(id: usize) -> Self {
        assert!(id <= u32::MAX as usize, "PatternId overflow");
        Self(id as u32)
    }
}

/// Type-safe indexing: Vec<PatternId> indexed by GlobalId
///
/// This prevents accidental misuse like:
///   pattern_counts[global_id]  // Compile error!
///   pattern_counts[pattern_id] // Compile error!
///
/// Only allows:
///   global_to_pattern[global_id] -> PatternId
impl std::ops::Index<GlobalId> for Vec<PatternId> {
    type Output = PatternId;

    #[inline]
    fn index(&self, id: GlobalId) -> &Self::Output {
        &self[id.as_usize()]
    }
}

impl std::ops::IndexMut<GlobalId> for Vec<PatternId> {
    #[inline]
    fn index_mut(&mut self, id: GlobalId) -> &mut Self::Output {
        &mut self[id.as_usize()]
    }
}

/// Type-safe indexing: Vec<T> indexed by PatternId (for pattern data)
impl<T> std::ops::Index<PatternId> for Vec<T> {
    type Output = T;

    #[inline]
    fn index(&self, id: PatternId) -> &Self::Output {
        assert!(!id.is_reservoir(), "Cannot index with RESERVOIR sentinel");
        &self[id.as_usize()]
    }
}

impl<T> std::ops::IndexMut<PatternId> for Vec<T> {
    #[inline]
    fn index_mut(&mut self, id: PatternId) -> &mut Self::Output {
        assert!(!id.is_reservoir(), "Cannot index with RESERVOIR sentinel");
        &mut self[id.as_usize()]
    }
}

/// HMM state index (0..n_states)
///
/// Distinct from GlobalId to avoid mixing reference hap IDs with local state indices.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct StateId(pub u32);

impl StateId {
    #[inline]
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    #[inline]
    pub fn as_usize(self) -> usize {
        self.0 as usize
    }

    #[inline]
    pub fn as_u32(self) -> u32 {
        self.0
    }
}

impl fmt::Debug for StateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StateId({})", self.0)
    }
}

impl fmt::Display for StateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "S{}", self.0)
    }
}

impl From<u32> for StateId {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl From<usize> for StateId {
    fn from(id: usize) -> Self {
        Self(id as u32)
    }
}

impl<T> std::ops::Index<StateId> for Vec<T> {
    type Output = T;

    #[inline]
    fn index(&self, id: StateId) -> &Self::Output {
        &self[id.as_usize()]
    }
}

impl<T> std::ops::IndexMut<StateId> for Vec<T> {
    #[inline]
    fn index_mut(&mut self, id: StateId) -> &mut Self::Output {
        &mut self[id.as_usize()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_id_basic() {
        let gid = GlobalId::new(42);
        assert_eq!(gid.as_u32(), 42);
        assert_eq!(gid.as_usize(), 42);
    }

    #[test]
    fn test_pattern_id_basic() {
        let pid = PatternId::new(10);
        assert_eq!(pid.as_u32(), 10);
        assert_eq!(pid.as_usize(), 10);
        assert!(!pid.is_reservoir());
    }

    #[test]
    fn test_reservoir_sentinel() {
        let reservoir = PatternId::RESERVOIR;
        assert!(reservoir.is_reservoir());
        assert_eq!(reservoir.as_u32(), u32::MAX);
    }

    #[test]
    fn test_type_safe_indexing() {
        let mut global_to_pattern = vec![PatternId::new(0); 100];
        let global_id = GlobalId::new(42);

        // This should compile and work
        global_to_pattern[global_id] = PatternId::new(5);
        assert_eq!(global_to_pattern[global_id], PatternId::new(5));

        // Test pattern-indexed vectors
        let mut pattern_counts = vec![1.0f32; 10];
        let pattern_id = PatternId::new(5);
        pattern_counts[pattern_id] = 42.0;
        assert_eq!(pattern_counts[pattern_id], 42.0);
    }

    #[test]
    fn test_display_formatting() {
        assert_eq!(format!("{}", GlobalId::new(42)), "G42");
        assert_eq!(format!("{}", PatternId::new(10)), "P10");
        assert_eq!(format!("{}", PatternId::RESERVOIR), "R");
    }

    #[test]
    #[should_panic(expected = "Use PatternId::RESERVOIR for sentinel value")]
    fn test_pattern_id_new_rejects_max() {
        PatternId::new(u32::MAX);
    }
}
