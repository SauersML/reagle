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

use crate::data::marker::{MarkerIdx, Markers};
use crate::data::ChromIdx;
use crate::error::{ReagleError, Result};

/// Default scale factor: 1 cM per Mb (1e-6 cM per bp)
pub const DEFAULT_SCALE_FACTOR: f64 = 1e-6;

/// Minimum genetic distance between consecutive markers (prevents zero distances)
/// This is approximately 0.01 * mean human single base genetic distance
pub const MIN_GEN_DIST: f64 = 1e-8;

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

    /// Create a position-based map (no genetic map file)
    ///
    /// This matches Java `vcf/PositionMap.java`
    /// Converts genome coordinates to genetic units by multiplying by scale_factor.
    /// Default scale_factor = 1e-6 (1 cM per Mb)
    pub fn position_map(chrom: ChromIdx, scale_factor: f64) -> PositionMap {
        PositionMap { chrom, scale_factor }
    }

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

            let pos: u32 = parts[1].parse().map_err(|_| {
                ReagleError::parse(line_num + 1, "Invalid position")
            })?;

            // Column 3 is rate (ignored), column 4 is genetic position
            let gen_pos: f64 = parts[3].parse().map_err(|_| {
                ReagleError::parse(line_num + 1, "Invalid genetic position")
            })?;

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

    /// Slice the genetic map to cover a physical position range
    ///
    /// The slice includes all map points within `[min_bp, max_bp]`, plus
    /// immediately surrounding points to ensure correct interpolation/extrapolation.
    pub fn slice(&self, min_bp: u32, max_bp: u32) -> Self {
        if self.positions.is_empty() {
            return self.clone();
        }

        // Find start index (inclusive)
        // binary_search: Ok(i) -> i, Err(i) -> i-1 (if i>0)
        let start_idx = match self.positions.binary_search(&min_bp) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        };

        // Find end index (inclusive)
        // binary_search: Ok(j) -> j, Err(j) -> j (clamped)
        let mut end_idx = match self.positions.binary_search(&max_bp) {
            Ok(j) => j,
            Err(j) => j.min(self.positions.len() - 1),
        };

        // Ensure we preserve extrapolation slopes at the edges
        let mut final_start = start_idx;
        let mut final_end = end_idx;

        // If slice starts at 0, include index 1 to preserve initial slope
        if final_start == 0 && self.positions.len() > 1 {
            final_end = final_end.max(1);
        }

        // If slice ends at last, include second-to-last to preserve final slope
        if final_end == self.positions.len() - 1 && self.positions.len() > 1 {
            final_start = final_start.min(self.positions.len() - 2);
        }

        // Ensure start <= end (handling empty range or overlap)
        if final_start > final_end {
            final_end = final_start;
        }

        Self {
            chrom: self.chrom,
            positions: self.positions[final_start..=final_end].to_vec(),
            gen_positions: self.gen_positions[final_start..=final_end].to_vec(),
        }
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

/// Position-based genetic map (no recombination rate map)
///
/// This matches Java `vcf/PositionMap.java`
#[derive(Clone, Debug)]
pub struct PositionMap {
    chrom: ChromIdx,
    scale_factor: f64,
}

impl PositionMap {
    /// Create a new position map with default scale factor (1 cM per Mb)
    pub fn new(chrom: ChromIdx) -> Self {
        Self {
            chrom,
            scale_factor: DEFAULT_SCALE_FACTOR,
        }
    }

    /// Create with custom scale factor
    pub fn with_scale(chrom: ChromIdx, scale_factor: f64) -> Self {
        Self { chrom, scale_factor }
    }

    /// Get genetic position from physical position
    pub fn gen_pos(&self, phys_pos: u32) -> f64 {
        phys_pos as f64 * self.scale_factor
    }

    /// Get genetic distance between two positions
    pub fn gen_dist(&self, pos1: u32, pos2: u32) -> f64 {
        ((pos2 as i64 - pos1 as i64).abs() as f64) * self.scale_factor
    }
}

/// Pre-computed genetic positions and distances for a set of markers
///
/// This matches Java `vcf/MarkerMap.java`
#[derive(Clone, Debug)]
pub struct MarkerMap {
    /// Genetic positions (cM) for each marker
    gen_pos: Vec<f64>,

    /// Genetic distance from previous marker (first element is 0)
    gen_dist: Vec<f32>,
}

impl MarkerMap {
    /// Create a MarkerMap from markers and genetic map
    ///
    /// This matches Java `MarkerMap.create(GeneticMap genMap, Markers markers)`
    pub fn create(markers: &Markers, gen_map: &GeneticMap) -> Self {
        let n = markers.len();
        if n == 0 {
            return Self {
                gen_pos: Vec::new(),
                gen_dist: Vec::new(),
            };
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
            return Self {
                gen_pos: Vec::new(),
                gen_dist: Vec::new(),
            };
        }

        let mut gen_pos = Vec::with_capacity(n);
        let mut gen_dist = Vec::with_capacity(n);

        // First marker
        let first_pos = markers.get(MarkerIdx::from(0usize)).map(|m| m.pos).unwrap_or(0);
        gen_pos.push(gen_map.gen_pos(first_pos));
        gen_dist.push(0.0);

        let mut last_map_pos = gen_pos[0];

        // Subsequent markers
        for i in 1..n {
            let pos = markers.get(MarkerIdx::from(i)).map(|m| m.pos).unwrap_or(0);
            let map_pos = gen_map.gen_pos(pos);
            let dist = (map_pos - last_map_pos).max(min_gen_dist);
            gen_pos.push(gen_pos[i - 1] + dist);
            gen_dist.push(dist as f32);
            last_map_pos = map_pos;
        }

        Self { gen_pos, gen_dist }
    }

    /// Create from genetic map without minimum distance enforcement
    pub fn from_gen_map(markers: &Markers, gen_map: &GeneticMap) -> Self {
        let n = markers.len();
        if n == 0 {
            return Self {
                gen_pos: Vec::new(),
                gen_dist: Vec::new(),
            };
        }

        let mut gen_pos = Vec::with_capacity(n);
        let mut gen_dist = Vec::with_capacity(n);

        for i in 0..n {
            let pos = markers.get(MarkerIdx::from(i)).map(|m| m.pos).unwrap_or(0);
            gen_pos.push(gen_map.gen_pos(pos));
            if i == 0 {
                gen_dist.push(0.0);
            } else {
                gen_dist.push((gen_pos[i] - gen_pos[i - 1]) as f32);
            }
        }

        Self { gen_pos, gen_dist }
    }

    /// Create using default position-based map (1 cM per Mb)
    pub fn from_positions(markers: &Markers) -> Self {
        let n = markers.len();
        if n == 0 {
            return Self {
                gen_pos: Vec::new(),
                gen_dist: Vec::new(),
            };
        }

        let pos_map = PositionMap::new(ChromIdx::new(0));
        let mut gen_pos = Vec::with_capacity(n);
        let mut gen_dist = Vec::with_capacity(n);

        for i in 0..n {
            let pos = markers.get(MarkerIdx::from(i)).map(|m| m.pos).unwrap_or(0);
            gen_pos.push(pos_map.gen_pos(pos));
            if i == 0 {
                gen_dist.push(0.0);
            } else {
                gen_dist.push((gen_pos[i] - gen_pos[i - 1]) as f32);
            }
        }

        Self { gen_pos, gen_dist }
    }

    /// Mean single-base genetic distance
    ///
    /// From Java `MarkerMap.meanSingleBaseGenDist`
    fn mean_single_base_gen_dist(markers: &Markers, gen_map: &GeneticMap) -> f64 {
        let n = markers.len();
        if n < 2 {
            return MIN_GEN_DIST;
        }

        let first_pos = markers.get(MarkerIdx::from(0usize)).map(|m| m.pos).unwrap_or(0);
        let last_pos = markers.get(MarkerIdx::from(n - 1)).map(|m| m.pos).unwrap_or(0);

        if first_pos == last_pos {
            return MIN_GEN_DIST;
        }

        let first_gen = gen_map.gen_pos(first_pos);
        let last_gen = gen_map.gen_pos(last_pos);

        let mean = (last_gen - first_gen).abs() / (last_pos as f64 - first_pos as f64).abs();

        // Require mean to be at least 0.01 * mean human single base genetic distance
        mean.max(MIN_GEN_DIST)
    }

    /// Number of markers
    pub fn len(&self) -> usize {
        self.gen_pos.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.gen_pos.is_empty()
    }

    /// Get genetic position for marker at index
    pub fn gen_pos(&self, index: usize) -> f64 {
        self.gen_pos[index]
    }

    /// Get genetic distance from previous marker
    pub fn gen_dist(&self, index: usize) -> f32 {
        self.gen_dist[index]
    }

    /// Get all genetic positions
    pub fn gen_positions(&self) -> &[f64] {
        &self.gen_pos
    }

    /// Get all genetic distances
    pub fn gen_distances(&self) -> &[f32] {
        &self.gen_dist
    }

    /// Calculate recombination probabilities for given recombination intensity
    ///
    /// From Java `MarkerMap.pRecomb(float recombIntensity)`:
    /// ```java
    /// double c = -recombIntensity;
    /// pRecomb[m] = -Math.expm1(c * genDist.get(m))
    /// ```
    ///
    /// Note: -expm1(x) = 1 - exp(x), more numerically stable for small x
    pub fn p_recomb(&self, recomb_intensity: f32) -> Vec<f32> {
        if recomb_intensity <= 0.0 || !recomb_intensity.is_finite() {
            panic!(
                "recomb_intensity must be positive and finite, got {}",
                recomb_intensity
            );
        }

        let c = -(recomb_intensity as f64);

        self.gen_dist
            .iter()
            .map(|&d| (-f64::exp_m1(c * d as f64)) as f32)
            .collect()
    }

    /// Restrict to a subset of markers
    ///
    /// From Java `MarkerMap.restrict(int[] indices)`
    pub fn restrict(&self, indices: &[usize]) -> Self {
        if indices.is_empty() {
            return Self {
                gen_pos: Vec::new(),
                gen_dist: Vec::new(),
            };
        }

        // Verify indices are sorted and increasing
        for i in 1..indices.len() {
            assert!(
                indices[i] > indices[i - 1],
                "Indices must be strictly increasing"
            );
        }

        let mut gen_pos = Vec::with_capacity(indices.len());
        let mut gen_dist = Vec::with_capacity(indices.len());

        gen_pos.push(self.gen_pos[indices[0]]);
        gen_dist.push(0.0);

        for i in 1..indices.len() {
            gen_pos.push(self.gen_pos[indices[i]]);
            gen_dist.push((gen_pos[i] - gen_pos[i - 1]) as f32);
        }

        Self { gen_pos, gen_dist }
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

    #[test]
    fn test_marker_map_from_positions() {
        let markers = make_test_markers();
        let mm = MarkerMap::from_positions(&markers);

        assert_eq!(mm.len(), 5);

        // First marker: gen_dist = 0
        assert_eq!(mm.gen_dist(0), 0.0);

        // Other markers: ~1 cM apart (1 Mb * 1e-6 = 1 cM)
        for i in 1..5 {
            assert!((mm.gen_dist(i) - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_marker_map_p_recomb() {
        let markers = make_test_markers();
        let mm = MarkerMap::from_positions(&markers);

        let p_recomb = mm.p_recomb(1.0);
        assert_eq!(p_recomb.len(), 5);

        // First marker: pRecomb = 0 (no previous marker)
        assert!((p_recomb[0] - 0.0).abs() < 0.0001);

        // Other markers: pRecomb = 1 - exp(-1 * 1.0) ≈ 0.632
        for i in 1..5 {
            let expected = 1.0 - (-1.0f64).exp();
            assert!((p_recomb[i] as f64 - expected).abs() < 0.001);
        }
    }

    #[test]
    fn test_marker_map_restrict() {
        let markers = make_test_markers();
        let mm = MarkerMap::from_positions(&markers);

        let restricted = mm.restrict(&[0, 2, 4]);
        assert_eq!(restricted.len(), 3);

        // Check distances are correct
        assert_eq!(restricted.gen_dist(0), 0.0);
        assert!((restricted.gen_dist(1) - 2.0).abs() < 0.001); // 2 cM (skipped one marker)
        assert!((restricted.gen_dist(2) - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_position_map() {
        let pm = PositionMap::new(ChromIdx::new(0));

        assert!((pm.gen_pos(1_000_000) - 1.0).abs() < 0.001);
        assert!((pm.gen_pos(2_000_000) - 2.0).abs() < 0.001);
        assert!((pm.gen_dist(1_000_000, 2_000_000) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_slice() {
        let map = GeneticMap::new(
            ChromIdx::new(0),
            vec![100, 200, 300, 400, 500],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        );

        // 1. Slice containing points exactly
        let s1 = map.slice(200, 400);
        assert_eq!(s1.positions, vec![200, 300, 400]);

        // 2. Slice between points
        let s2 = map.slice(250, 350);
        // Should include 200 (start-1) and 400 (end+1/clamped)
        // 250 -> Err(2) -> start=1 (200)
        // 350 -> Err(3) -> end=3 (400)
        assert_eq!(s2.positions, vec![200, 300, 400]);

        // 3. Slice at start
        let s3 = map.slice(50, 150);
        // 50 -> Err(0) -> start=0
        // 150 -> Err(1) -> end=1
        // start=0 -> ensures end>=1.
        assert_eq!(s3.positions, vec![100, 200]);

        // 4. Slice at end
        let s4 = map.slice(450, 550);
        // 450 -> Err(4) -> start=3 (400)
        // 550 -> Err(5) -> end=4 (500)
        // end=4 -> ensures start<=3.
        assert_eq!(s4.positions, vec![400, 500]);

        // 5. Slice completely before
        let s5 = map.slice(10, 20);
        // start=0, end=0 -> force end=1
        assert_eq!(s5.positions, vec![100, 200]);

        // 6. Slice completely after
        let s6 = map.slice(600, 700);
        // start=4, end=4 -> force start=3
        assert_eq!(s6.positions, vec![400, 500]);

        // 7. Small slice in middle
        let s7 = map.slice(210, 220);
        // 210 -> Err(2) -> start=1 (200)
        // 220 -> Err(2) -> end=2 (300)
        assert_eq!(s7.positions, vec![200, 300]);

        // Verify interpolation/extrapolation works same as original
        // Before first
        assert!((s5.gen_pos(50) - map.gen_pos(50)).abs() < 1e-10);
        // After last
        assert!((s6.gen_pos(600) - map.gen_pos(600)).abs() < 1e-10);
        // Middle
        assert!((s7.gen_pos(215) - map.gen_pos(215)).abs() < 1e-10);
    }
}
