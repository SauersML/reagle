//! # Genetic Map
//!
//! Physical-to-genetic distance interpolation.
//! This module provides:
//! - `GeneticMap`: Raw genetic map data from PLINK files
//! - `MarkerMap`: Pre-computed genetic positions and distances for markers
//!
//! Replaces Java `vcf/GeneticMap.java`, `vcf/PlinkGenMap.java`, and `vcf/MarkerMap.java`.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::data::ChromIdx;
use crate::data::marker::{MarkerIdx, Markers};
use crate::error::{ReagleError, Result};

/// Default scale factor: 1 cM per Mb (1e-6 cM per bp)
pub const DEFAULT_SCALE_FACTOR: f64 = 1e-6;

/// A genetic map for interpolating physical positions to genetic distances (cM)
///
/// This matches Java `vcf/GeneticMap.java`.
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
    /// Load from PLINK format map file
    ///
    /// Format: chrom position_bp rate_cM_per_Mb position_cM
    /// (Note: rate column is ignored, we use the cumulative position)
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

            let pos: u32 = parts[1]
                .parse()
                .map_err(|_| ReagleError::parse(line_num + 1, "Invalid position"))?;

            // Column 3 is rate (ignored), column 4 is genetic position
            let gen_pos: f64 = parts[3]
                .parse()
                .map_err(|_| ReagleError::parse(line_num + 1, "Invalid genetic position"))?;

            if !gen_pos.is_finite() {
                return Err(ReagleError::parse(
                    line_num + 1,
                    "Genetic position is not finite",
                ));
            }

            positions.push(pos);
            gen_positions.push(gen_pos);
        }

        // Verify sorted order
        for i in 1..positions.len() {
            if positions[i] <= positions[i - 1] {
                return Err(ReagleError::Config {
                    message: format!(
                        "Genetic map positions not in ascending order at position {}",
                        positions[i]
                    ),
                });
            }
        }

        Ok(Self {
            chrom: ChromIdx::new(0), // Will be set by caller
            positions,
            gen_positions,
        })
    }

    /// Interpolate genetic position (cM) from physical position (bp)
    ///
    /// From Java `GeneticMap.genPos(int chrom, int basePosition)`
    pub fn gen_pos(&self, phys_pos: u32) -> f64 {
        if self.positions.is_empty() {
            // Default: 1 cM per Mb
            return phys_pos as f64 * DEFAULT_SCALE_FACTOR;
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
                        DEFAULT_SCALE_FACTOR
                    };
                    self.gen_positions[0] - rate * (self.positions[0] - phys_pos) as f64
                } else if idx == self.positions.len() {
                    // After last position: extrapolate
                    let last = self.positions.len() - 1;
                    let rate = if last > 0 {
                        (self.gen_positions[last] - self.gen_positions[last - 1])
                            / (self.positions[last] - self.positions[last - 1]) as f64
                    } else {
                        DEFAULT_SCALE_FACTOR
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

    /// Set chromosome index
    pub fn set_chrom(&mut self, chrom: ChromIdx) {
        self.chrom = chrom;
    }
}

/// Position-based genetic map (no recombination rate map)
///
/// This matches Java `vcf/PositionMap.java`
#[derive(Clone, Debug)]
pub struct PositionMap {
    scale_factor: f64,
}

impl PositionMap {
    /// Create a new position map with default scale factor (1 cM per Mb)
    pub fn new() -> Self {
        Self {
            scale_factor: DEFAULT_SCALE_FACTOR,
        }
    }

    /// Get genetic position from physical position
    pub fn gen_pos(&self, phys_pos: u32) -> f64 {
        phys_pos as f64 * self.scale_factor
    }
}

/// Pre-computed genetic positions for a set of markers
///
/// This matches Java `vcf/MarkerMap.java`
#[derive(Clone, Debug)]
pub struct MarkerMap {
    /// Genetic positions (cM) for each marker
    gen_pos: Vec<f64>,
}

impl MarkerMap {
    /// Create a MarkerMap from markers and genetic map
    ///
    /// This matches Java `MarkerMap.create(GeneticMap genMap, Markers markers)`
    pub fn create(markers: &Markers, gen_map: &GeneticMap) -> Self {
        let n = markers.len();
        if n == 0 {
            return Self { gen_pos: Vec::new() };
        }

        // Calculate mean single-base genetic distance
        let mean_gen_dist = Self::mean_single_base_gen_dist(markers, gen_map);

        // Calculate genetic positions with minimum distance enforcement
        Self::from_gen_map_with_min_dist(markers, gen_map, mean_gen_dist)
    }

    /// Create from genetic map with minimum distance between markers
    ///
    /// From Java `GeneticMap.genPos(GeneticMap genMap, double minGenDist, Markers markers)`
    pub fn from_gen_map_with_min_dist(
        markers: &Markers,
        gen_map: &GeneticMap,
        min_gen_dist: f64,
    ) -> Self {
        let n = markers.len();
        if n == 0 {
            return Self { gen_pos: Vec::new() };
        }

        let mut gen_pos = Vec::with_capacity(n);

        // First marker
        let first_pos = markers
            .get(MarkerIdx::from(0usize))
            .map(|m| m.pos)
            .unwrap_or(0);
        gen_pos.push(gen_map.gen_pos(first_pos));

        let mut last_map_pos = gen_pos[0];

        // Subsequent markers
        for i in 1..n {
            let pos = markers.get(MarkerIdx::from(i)).map(|m| m.pos).unwrap_or(0);
            let map_pos = gen_map.gen_pos(pos);
            let dist = (map_pos - last_map_pos).max(min_gen_dist);
            gen_pos.push(gen_pos[i - 1] + dist);
            last_map_pos = map_pos;
        }

        Self { gen_pos }
    }

    /// Create using default position-based map (1 cM per Mb)
    pub fn from_positions(markers: &Markers) -> Self {
        let n = markers.len();
        if n == 0 {
            return Self { gen_pos: Vec::new() };
        }

        let pos_map = PositionMap::new();
        let mut gen_pos = Vec::with_capacity(n);

        for i in 0..n {
            let pos = markers.get(MarkerIdx::from(i)).map(|m| m.pos).unwrap_or(0);
            gen_pos.push(pos_map.gen_pos(pos));
        }

        Self { gen_pos }
    }

    /// Mean single-base genetic distance
    ///
    /// From Java `MarkerMap.meanSingleBaseGenDist`
    fn mean_single_base_gen_dist(markers: &Markers, gen_map: &GeneticMap) -> f64 {
        // Minimum genetic distance (~0.01 * mean human single base genetic distance)
        const MIN_GEN_DIST: f64 = 1e-8;

        let n = markers.len();
        if n < 2 {
            return MIN_GEN_DIST;
        }

        let first_pos = markers
            .get(MarkerIdx::from(0usize))
            .map(|m| m.pos)
            .unwrap_or(0);
        let last_pos = markers
            .get(MarkerIdx::from(n - 1))
            .map(|m| m.pos)
            .unwrap_or(0);

        if first_pos == last_pos {
            return MIN_GEN_DIST;
        }

        let first_gen = gen_map.gen_pos(first_pos);
        let last_gen = gen_map.gen_pos(last_pos);

        let mean = (last_gen - first_gen).abs() / (last_pos as f64 - first_pos as f64).abs();

        // Require mean to be at least 0.01 * mean human single base genetic distance
        mean.max(MIN_GEN_DIST)
    }

    /// Get all genetic positions
    pub fn gen_positions(&self) -> &[f64] {
        &self.gen_pos
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
        info_span!("genetic_maps_from_plink_file", path = ?path).in_scope(|| {
            let mut maps = Vec::with_capacity(chrom_names.len());
            for (i, &name) in chrom_names.iter().enumerate() {
                let mut map = GeneticMap::from_plink_file(path, name)?;
                map.set_chrom(ChromIdx::new(i as u16));
                maps.push(Some(map));
            }
            Ok(Self { maps })
        })
    }

    /// Get genetic map for a chromosome
    pub fn get(&self, chrom: ChromIdx) -> Option<&GeneticMap> {
        self.maps.get(chrom.as_usize()).and_then(|m| m.as_ref())
    }

    /// Get genetic position, falling back to default rate if no map
    pub fn gen_pos(&self, chrom: ChromIdx, phys_pos: u32) -> f64 {
        match self.get(chrom) {
            Some(map) => map.gen_pos(phys_pos),
            None => phys_pos as f64 * DEFAULT_SCALE_FACTOR,
        }
    }

    /// Get genetic distance between two positions
    pub fn gen_dist(&self, chrom: ChromIdx, pos1: u32, pos2: u32) -> f64 {
        match self.get(chrom) {
            Some(map) => map.gen_dist(pos1, pos2),
            None => (pos2 as f64 - pos1 as f64).abs() * DEFAULT_SCALE_FACTOR,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::marker::{Allele, Marker};

    /// Get the path to the PLINK genetic map fixture
    fn fixture_map_path() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("test_genetic_map.map")
    }

    fn make_test_markers() -> Markers {
        let mut markers = Markers::new();
        markers.add_chrom("chr1");

        // 5 markers at 1Mb intervals
        for i in 0..5 {
            let m = Marker::new(
                ChromIdx::new(0),
                (i + 1) * 1_000_000,
                None,
                Allele::Base(0),
                vec![Allele::Base(1)],
            );
            markers.push(m);
        }
        markers
    }

    #[test]
    fn test_interpolation() {
        // Use production from_plink_file() to load the map
        let map = GeneticMap::from_plink_file(fixture_map_path().as_path(), "chr1")
            .expect("Failed to load PLINK map fixture");

        // Exact positions (from fixture: 1Mb=0.0, 2Mb=1.0, 3Mb=2.5)
        assert!((map.gen_pos(1_000_000) - 0.0).abs() < 0.001);
        assert!((map.gen_pos(2_000_000) - 1.0).abs() < 0.001);
        assert!((map.gen_pos(3_000_000) - 2.5).abs() < 0.001);

        // Interpolated position
        assert!((map.gen_pos(1_500_000) - 0.5).abs() < 0.001);
        assert!((map.gen_pos(2_500_000) - 1.75).abs() < 0.001);
    }

    #[test]
    fn test_extrapolation() {
        let map = GeneticMap::from_plink_file(fixture_map_path().as_path(), "chr1")
            .expect("Failed to load PLINK map fixture");

        // Before first position (should extrapolate)
        let before = map.gen_pos(500_000);
        assert!(before < 0.0, "Position before first marker should extrapolate to < 0");

        // After last position (should extrapolate)
        let after = map.gen_pos(3_500_000);
        assert!(after > 2.5, "Position after last marker should extrapolate to > 2.5");
    }

    #[test]
    fn test_empty_map_for_missing_chrom() {
        // Load map for a chromosome that doesn't exist - should return empty map
        let map = GeneticMap::from_plink_file(fixture_map_path().as_path(), "chr99")
            .expect("Loading map for missing chrom should succeed with empty positions");
        
        // Empty map should use default rate of 1 cM per Mb
        assert!((map.gen_pos(1_000_000) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_gen_dist() {
        let map = GeneticMap::from_plink_file(fixture_map_path().as_path(), "chr1")
            .expect("Failed to load PLINK map fixture");

        // Distance from 1Mb (0.0 cM) to 2Mb (1.0 cM) = 1.0 cM
        assert!((map.gen_dist(1_000_000, 2_000_000) - 1.0).abs() < 0.001);
        
        // Distance from 1Mb to 1.5Mb (interpolated to 0.5 cM) = 0.5 cM
        assert!((map.gen_dist(1_000_000, 1_500_000) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_marker_map_from_positions() {
        let markers = make_test_markers();
        let mm = MarkerMap::from_positions(&markers);

        // Verify genetic positions are computed correctly
        let positions = mm.gen_positions();
        assert_eq!(positions.len(), 5);

        // First marker at 1 Mb should have gen_pos = 1 cM (default rate: 1 cM/Mb)
        assert!((positions[0] - 1.0).abs() < 0.001);

        // Other markers: ~1 cM apart (1 Mb * 1e-6 = 1 cM)
        for i in 1..5 {
            assert!((positions[i] - positions[i - 1] - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_position_map() {
        let pm = PositionMap::new();

        assert!((pm.gen_pos(1_000_000) - 1.0).abs() < 0.001);
        assert!((pm.gen_pos(2_000_000) - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_genetic_maps_collection() {
        // Test the GeneticMaps collection with from_plink_file
        let chrom_names = ["chr1"];
        let maps = GeneticMaps::from_plink_file(fixture_map_path().as_path(), &chrom_names)
            .expect("Failed to load maps");

        // Should have loaded chr1
        assert!(maps.get(ChromIdx::new(0)).is_some());
        
        // Test gen_pos through collection
        let pos = maps.gen_pos(ChromIdx::new(0), 1_500_000);
        assert!((pos - 0.5).abs() < 0.001);
    }
}
