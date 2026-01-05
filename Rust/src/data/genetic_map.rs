//! # Genetic Map
//!
//! Physical-to-genetic distance interpolation.
//! Replaces `vcf/GeneticMap.java` and `vcf/PlinkGenMap.java`.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::data::ChromIdx;
use crate::error::{ReagleError, Result};

/// A genetic map for interpolating physical positions to genetic distances (cM)
#[derive(Clone, Debug)]
pub struct GeneticMap {
    /// Chromosome index
    chrom: ChromIdx,

    /// Physical positions (bp), sorted
    positions: Vec<u32>,

    /// Genetic positions (cM) corresponding to physical positions
    gen_positions: Vec<f64>,
}

impl GeneticMap {
    /// Create a new genetic map
    pub fn new(chrom: ChromIdx, positions: Vec<u32>, gen_positions: Vec<f64>) -> Self {
        debug_assert_eq!(positions.len(), gen_positions.len());
        debug_assert!(positions.windows(2).all(|w| w[0] <= w[1]));
        Self {
            chrom,
            positions,
            gen_positions,
        }
    }

    /// Create an empty genetic map (uses default rate)
    pub fn empty(chrom: ChromIdx) -> Self {
        Self {
            chrom,
            positions: Vec::new(),
            gen_positions: Vec::new(),
        }
    }

    /// Load from PLINK format map file
    /// Format: chrom position_bp rate_cM_per_Mb position_cM
    pub fn from_plink_file(path: &Path, target_chrom: &str) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut positions = Vec::new();
        let mut gen_positions = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 4 {
                return Err(ReagleError::parse(
                    line_num + 1,
                    format!("Expected 4 columns, got {}", parts.len()),
                ));
            }

            let chrom = parts[0];
            if chrom != target_chrom {
                continue;
            }

            let pos: u32 = parts[1].parse().map_err(|_| {
                ReagleError::parse(line_num + 1, "Invalid position")
            })?;

            // Column 3 is rate (ignored), column 4 is genetic position
            let gen_pos: f64 = parts[3].parse().map_err(|_| {
                ReagleError::parse(line_num + 1, "Invalid genetic position")
            })?;

            positions.push(pos);
            gen_positions.push(gen_pos);
        }

        Ok(Self {
            chrom: ChromIdx::new(0), // Will be set by caller
            positions,
            gen_positions,
        })
    }

    /// Interpolate genetic position (cM) from physical position (bp)
    pub fn gen_pos(&self, phys_pos: u32) -> f64 {
        if self.positions.is_empty() {
            // Default: 1 cM per Mb
            return phys_pos as f64 / 1_000_000.0;
        }

        // Binary search for position
        match self.positions.binary_search(&phys_pos) {
            Ok(idx) => self.gen_positions[idx],
            Err(idx) => {
                if idx == 0 {
                    // Before first position: extrapolate
                    let rate = if self.positions.len() > 1 {
                        (self.gen_positions[1] - self.gen_positions[0])
                            / (self.positions[1] - self.positions[0]) as f64
                    } else {
                        1.0 / 1_000_000.0 // Default rate
                    };
                    self.gen_positions[0] - rate * (self.positions[0] - phys_pos) as f64
                } else if idx == self.positions.len() {
                    // After last position: extrapolate
                    let last = self.positions.len() - 1;
                    let rate = if last > 0 {
                        (self.gen_positions[last] - self.gen_positions[last - 1])
                            / (self.positions[last] - self.positions[last - 1]) as f64
                    } else {
                        1.0 / 1_000_000.0
                    };
                    self.gen_positions[last] + rate * (phys_pos - self.positions[last]) as f64
                } else {
                    // Interpolate between idx-1 and idx
                    let p0 = self.positions[idx - 1] as f64;
                    let p1 = self.positions[idx] as f64;
                    let g0 = self.gen_positions[idx - 1];
                    let g1 = self.gen_positions[idx];
                    let t = (phys_pos as f64 - p0) / (p1 - p0);
                    g0 + t * (g1 - g0)
                }
            }
        }
    }

    /// Get genetic distance between two physical positions (cM)
    pub fn gen_dist(&self, pos1: u32, pos2: u32) -> f64 {
        (self.gen_pos(pos2) - self.gen_pos(pos1)).abs()
    }

    /// Number of map entries
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Check if map is empty
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Chromosome index
    pub fn chrom(&self) -> ChromIdx {
        self.chrom
    }

    /// Set chromosome index
    pub fn set_chrom(&mut self, chrom: ChromIdx) {
        self.chrom = chrom;
    }
}

/// A collection of genetic maps for multiple chromosomes
#[derive(Clone, Debug, Default)]
pub struct GeneticMaps {
    maps: Vec<Option<GeneticMap>>,
}

impl GeneticMaps {
    /// Create empty collection
    pub fn new() -> Self {
        Self::default()
    }

    /// Load all chromosomes from a PLINK map file
    pub fn from_plink_file(path: &Path, chrom_names: &[&str]) -> Result<Self> {
        let mut maps = Vec::with_capacity(chrom_names.len());
        for (i, &name) in chrom_names.iter().enumerate() {
            let mut map = GeneticMap::from_plink_file(path, name)?;
            map.set_chrom(ChromIdx::new(i as u16));
            maps.push(Some(map));
        }
        Ok(Self { maps })
    }

    /// Get genetic map for a chromosome
    pub fn get(&self, chrom: ChromIdx) -> Option<&GeneticMap> {
        self.maps.get(chrom.as_usize()).and_then(|m| m.as_ref())
    }

    /// Add a genetic map
    pub fn insert(&mut self, chrom: ChromIdx, map: GeneticMap) {
        let idx = chrom.as_usize();
        if idx >= self.maps.len() {
            self.maps.resize_with(idx + 1, || None);
        }
        self.maps[idx] = Some(map);
    }

    /// Get genetic position, falling back to default rate if no map
    pub fn gen_pos(&self, chrom: ChromIdx, phys_pos: u32) -> f64 {
        match self.get(chrom) {
            Some(map) => map.gen_pos(phys_pos),
            None => phys_pos as f64 / 1_000_000.0, // Default: 1 cM per Mb
        }
    }

    /// Get genetic distance between two positions
    pub fn gen_dist(&self, chrom: ChromIdx, pos1: u32, pos2: u32) -> f64 {
        match self.get(chrom) {
            Some(map) => map.gen_dist(pos1, pos2),
            None => (pos2 as f64 - pos1 as f64).abs() / 1_000_000.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolation() {
        let map = GeneticMap::new(
            ChromIdx::new(0),
            vec![1_000_000, 2_000_000, 3_000_000],
            vec![0.0, 1.0, 2.5],
        );

        // Exact positions
        assert!((map.gen_pos(1_000_000) - 0.0).abs() < 0.001);
        assert!((map.gen_pos(2_000_000) - 1.0).abs() < 0.001);
        assert!((map.gen_pos(3_000_000) - 2.5).abs() < 0.001);

        // Interpolated position
        assert!((map.gen_pos(1_500_000) - 0.5).abs() < 0.001);
        assert!((map.gen_pos(2_500_000) - 1.75).abs() < 0.001);
    }

    #[test]
    fn test_extrapolation() {
        let map = GeneticMap::new(
            ChromIdx::new(0),
            vec![1_000_000, 2_000_000],
            vec![1.0, 2.0],
        );

        // Before first position
        let before = map.gen_pos(500_000);
        assert!(before < 1.0);

        // After last position
        let after = map.gen_pos(2_500_000);
        assert!(after > 2.0);
    }

    #[test]
    fn test_empty_map() {
        let map = GeneticMap::empty(ChromIdx::new(0));
        // Should use default rate of 1 cM per Mb
        assert!((map.gen_pos(1_000_000) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_gen_dist() {
        let map = GeneticMap::new(
            ChromIdx::new(0),
            vec![1_000_000, 2_000_000],
            vec![0.0, 1.0],
        );

        assert!((map.gen_dist(1_000_000, 2_000_000) - 1.0).abs() < 0.001);
        assert!((map.gen_dist(1_000_000, 1_500_000) - 0.5).abs() < 0.001);
    }
}