//! # Binary Reference Panel I/O
//!
//! Implements a custom binary format for "Instant Start" reference panels.
//!
//! Format:
//! - [Magic 8 bytes] "REAGLE01"
//! - [Header] (Version, counts, metadata offset)
//! - [Metadata Length u64]
//! - [Metadata JSON] (Markers, Samples)
//! - [Block Index] (Table of offsets to compressed blocks)
//! - [Block Data] (Compressed genotype blocks)

use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};

use crate::data::haplotype::Samples;
use crate::data::marker::Markers;
use crate::data::storage::{
    compress_block, BinaryDictionaryColumn, GenotypeColumn, GenotypeMatrix, PhaseState, Phased,
    Unphased,
};

const MAGIC: &[u8; 8] = b"REAGLE01";
const VERSION: u32 = 1;

/// Metadata stored in the JSON header
#[derive(Serialize, Deserialize)]
struct Metadata {
    markers: Markers,
    samples: Samples,
}

/// Entry in the block index table
#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct BlockIndexEntry {
    start_marker: u32,
    n_markers: u32,
    offset: u64,
}

// Ensure C layout is safe for simple casting (though we'll use manual write for safety)

/// Binary Writer
pub struct BinaryWriter {
    writer: BufWriter<File>,
    block_index: Vec<BlockIndexEntry>,
}

impl BinaryWriter {
    /// Create a new binary writer
    pub fn create(path: &Path) -> Result<Self> {
        let file = File::create(path).context("Failed to create binary file")?;
        Ok(Self {
            writer: BufWriter::new(file),
            block_index: Vec::new(),
        })
    }

    /// Write the genotype matrix to the binary file
    pub fn write_matrix<S: PhaseState>(&mut self, matrix: &GenotypeMatrix<S>) -> Result<()> {
        let n_markers = matrix.n_markers();
        let n_haps = matrix.n_haplotypes();
        
        // 1. Prepare Metadata
        let metadata = Metadata {
            markers: matrix.markers().clone(),
            samples: (*matrix.samples()).clone(),
        };
        let metadata_json = serde_json::to_vec(&metadata)?;
        let metadata_len = metadata_json.len() as u64;

        // 2. Write Magic and Header Placeholder
        self.writer.write_all(MAGIC)?;
        // Header: Version(4), n_haps(4), n_markers(4), n_blocks(4)
        self.writer.write_all(&VERSION.to_le_bytes())?;
        self.writer.write_all(&(n_haps as u32).to_le_bytes())?;
        self.writer.write_all(&(n_markers as u32).to_le_bytes())?;
        
        // Placeholder for n_blocks
        self.writer.write_all(&0u32.to_le_bytes())?;

        // 3. Write Metadata
        self.writer.write_all(&metadata_len.to_le_bytes())?;
        self.writer.write_all(&metadata_json)?;

        // 4. Align to 8 bytes for block index
        self.align_to(8)?;
        
        // Placeholder for Block Index
        // We don't know how many blocks yet, but we will write the index at the END 
        // or we need to reserve space. 
        // Strategy: Write blocks first, then write index at the end? 
        // No, reader expects structure. 
        // Let's write blocks first, then the index at the VERY END.
        // And the header will point to the index offset?
        // Or we can just append the index and put its offset in the header.
        // Let's modify the header concept:
        // [Magic] [Version] [n_haps] [n_markers] [Metadata_Len] [Metadata] [Index_Offset] [Block_Data...] [Index]
        
        // Let's rewrite header strategy.
        // Current file pos is after metadata.
        
        // Save position where Block Index Offset will be written
        let index_offset_pos = self.writer.stream_position()?;
        self.writer.write_all(&0u64.to_le_bytes())?; // Placeholder for index offset
        
        // 5. Write Genotype Blocks
        let mut start_marker = 0;
        let block_size = 64; // Markers per block

        while start_marker < n_markers {
            let end_marker = (start_marker + block_size).min(n_markers);
            let n_in_block = end_marker - start_marker;

            // Align start of block to 8 bytes
            self.align_to(8)?;
            let block_offset = self.writer.stream_position()?;

            // Compress block
            // Helper to access matrix alleles
            let get_allele = |m_offset: usize, h: crate::data::haplotype::HapIdx| {
                matrix.column(crate::data::marker::MarkerIdx::new((start_marker + m_offset) as u32)).get(h)
            };

            // Force compression or use raw if compression fails?
            // compress_block returns Option<DictionaryColumn>. 
            // If None, we still need to store it. 
            // But for this Binary Format, we ONLY support dictionary blocks for now 
            // because that's the "Instant Start" goal.
            // If compression is bad, we just store it as a dictionary with many patterns.
            // So we explicitly construct DictionaryColumn even if ratio is bad.
            
            // Re-implement simplified compression logic that always returns a DictionaryColumn
            // Or change compress_block API. 
            // For now, let's just use DictionaryColumn::compress directly.
            
            let columns: Vec<Box<dyn Fn(crate::data::haplotype::HapIdx) -> u8>> = (0..n_in_block)
                .map(|m| {
                    let get_allele_ref = &get_allele;
                    Box::new(move |h| get_allele_ref(m, h)) as Box<dyn Fn(crate::data::haplotype::HapIdx) -> u8>
                })
                .collect();
            
            let column_fns: Vec<_> = columns.iter().map(|f| |h| f(h)).collect();
            
            // Use 1 bit per allele (assuming biallelic for now as that's the main target)
            // TODO: Support multiallelic
            let dict = crate::data::storage::DictionaryColumn::compress(
                &column_fns,
                n_in_block,
                n_haps,
                1,
            );

            // Write block to file
            self.write_block(&dict)?;

            // Record index entry
            self.block_index.push(BlockIndexEntry {
                start_marker: start_marker as u32,
                n_markers: n_in_block as u32,
                offset: block_offset,
            });

            start_marker = end_marker;
        }

        // 6. Write Block Index at the end
        let index_pos = self.writer.stream_position()?;
        self.writer.write_all(&(self.block_index.len() as u32).to_le_bytes())?; // n_blocks
        
        for entry in &self.block_index {
            self.writer.write_all(&entry.start_marker.to_le_bytes())?;
            self.writer.write_all(&entry.n_markers.to_le_bytes())?;
            self.writer.write_all(&entry.offset.to_le_bytes())?;
        }

        // 7. Update offsets in header
        // n_blocks was at n_blocks_pos (Wait, I removed n_blocks from header in revised plan)
        // I put Index Offset at index_offset_pos
        
        self.writer.seek(SeekFrom::Start(index_offset_pos))?;
        self.writer.write_all(&index_pos.to_le_bytes())?;

        self.writer.flush()?;
        Ok(())
    }

    fn align_to(&mut self, align: u64) -> Result<()> {
        let pos = self.writer.stream_position()?;
        let remainder = pos % align;
        if remainder != 0 {
            let padding = align - remainder;
            self.writer.write_all(&vec![0u8; padding as usize])?;
        }
        Ok(())
    }

    fn write_block(&mut self, dict: &crate::data::storage::DictionaryColumn) -> Result<()> {
        // Layout:
        // n_markers (4)
        // n_haps (4)
        // n_patterns (4)
        // padding (4)
        // hap_to_pattern (n_haps * 2)
        // padding (align 8)
        // patterns (n_patterns * words * 8)

        let n_markers = dict.n_markers() as u32;
        let n_haps = dict.n_haplotypes() as u32;
        let n_patterns = dict.n_patterns() as u32;

        self.writer.write_all(&n_markers.to_le_bytes())?;
        self.writer.write_all(&n_haps.to_le_bytes())?;
        self.writer.write_all(&n_patterns.to_le_bytes())?;
        self.writer.write_all(&[0u8; 4])?; // Padding

        // Write hap_to_pattern (Vec<u16>)
        // We need to access private fields of DictionaryColumn or add accessors.
        // Currently DictionaryColumn fields are private.
        // I need to add accessors to DictionaryColumn in dictionary.rs
        // OR make fields crate-public.
        // For now, I'll rely on pattern_index() accessor.
        
        for h in 0..n_haps {
            let pat_idx = dict.pattern_index(crate::data::haplotype::HapIdx::new(h));
            self.writer.write_all(&pat_idx.to_le_bytes())?;
        }

        self.align_to(8)?;

        // Write patterns (Vec<BitVec>)
        // DictionaryColumn exposes raw patterns via patterns()
        
        let patterns = dict.patterns();
        for pattern in patterns {
            let slice = pattern.as_raw_slice();
            // Ensure we write exactly words_per_pattern words
            // BitVec might shrink the storage if trailing words are zero? 
            // Lsb0 BitVec usually keeps capacity.
            // But strict correctness: write fixed size.
            
            let words_to_write = (n_markers as usize + 63) / 64;
            for i in 0..words_to_write {
                let word = slice.get(i).copied().unwrap_or(0);
                self.writer.write_all(&word.to_le_bytes())?;
            }
        }

        Ok(())
    }
}

/// Binary Reader
pub struct BinaryReader {
    mmap: Arc<Mmap>,
    metadata: Metadata,
    block_index: Vec<BlockIndexEntry>,
}

impl BinaryReader {
    /// Open a binary reference panel
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).context("Failed to open binary file")?;
        let mmap = unsafe { Mmap::map(&file).context("Failed to mmap file")? };
        let mmap = Arc::new(mmap);

        if mmap.len() < 16 || &mmap[0..8] != MAGIC {
            anyhow::bail!("Invalid file format or magic");
        }

        let version = u32::from_le_bytes(mmap[8..12].try_into()?);
        if version != VERSION {
            anyhow::bail!("Unsupported version: {}", version);
        }

        // Read offsets
        // [Magic 8] [Ver 4] [nHaps 4] [nMarkers 4] [Index_Offset 8] (Wait, index offset was after n_markers? no.)
        
        // Writer layout recap:
        // Magic(8)
        // Version(4)
        // n_haps(4)
        // n_markers(4)
        // Placeholder n_blocks (4) - Logic check:
        // write Version, n_haps, n_markers.
        // Then I wrote 0u32 (placeholder for n_blocks) in step 2.
        // Then metadata len (8)
        // Then metadata
        // Then Index_Offset (8) -> Wrote 0u64 at index_offset_pos.
        
        // Reading:
        let n_haps = u32::from_le_bytes(mmap[12..16].try_into()?);
        let n_markers = u32::from_le_bytes(mmap[16..20].try_into()?);
        let _ = u32::from_le_bytes(mmap[20..24].try_into()?); // Skip placeholder
        
        let metadata_len = u64::from_le_bytes(mmap[24..32].try_into()?);
        let metadata_start = 32;
        let metadata_end = metadata_start + metadata_len as usize;
        
        // Read Index Offset (located after metadata)
        
        let index_offset_pos = metadata_end;
        let index_offset = u64::from_le_bytes(mmap[index_offset_pos..index_offset_pos+8].try_into()?);
        
        // Parse metadata
        let metadata_bytes = &mmap[metadata_start..metadata_end];
        let metadata: Metadata = serde_json::from_slice(metadata_bytes)?;

        // Verify header against metadata
        if n_haps as usize != metadata.samples.n_haps() {
            anyhow::bail!("Header n_haps ({}) does not match metadata ({})", n_haps, metadata.samples.n_haps());
        }
        if n_markers as usize != metadata.markers.len() {
             anyhow::bail!("Header n_markers ({}) does not match metadata ({})", n_markers, metadata.markers.len());
        }

        // Parse Block Index
        let index_start = index_offset as usize;
        
        // n_blocks follows immediately but logic above skipped the placeholder read?
        // Wait, index_start points to the start of the index block.
        // My writer logic: 
        // 6. Write Block Index at the end
        // [n_blocks (4)] [Entry...]
        // So index_offset points to n_blocks.
        
        let n_blocks = u32::from_le_bytes(mmap[index_start..index_start+4].try_into()?);
        
        let mut cursor = index_start + 4; 
        let mut block_index = Vec::with_capacity(n_blocks as usize);
        
        for _ in 0..n_blocks {
            let start_marker = u32::from_le_bytes(mmap[cursor..cursor+4].try_into()?);
            let n_markers = u32::from_le_bytes(mmap[cursor+4..cursor+8].try_into()?);
            let offset = u64::from_le_bytes(mmap[cursor+8..cursor+16].try_into()?);
            cursor += 16;
            
            block_index.push(BlockIndexEntry {
                start_marker,
                n_markers,
                offset,
            });
        }

        Ok(Self {
            mmap,
            metadata,
            block_index,
        })
    }

    /// Convert to GenotypeMatrix (Zero-Copy)
    pub fn into_matrix(self) -> GenotypeMatrix<Unphased> {
        let n_blocks = self.block_index.len();
        let mut columns = Vec::with_capacity(self.metadata.markers.len());

        // Fill columns with BinaryDictionaryColumn
        // Since blocks cover multiple columns, we create shared Arcs.
        
        for block in &self.block_index {
            let col = BinaryDictionaryColumn::new(
                Arc::clone(&self.mmap),
                block.offset as usize
            ).expect("Invalid block offset in file");
            
            let arc_col = Arc::new(col);
            
            for m in 0..block.n_markers {
                columns.push(GenotypeColumn::BinaryDictionary(Arc::clone(&arc_col), m as usize));
            }
        }

        GenotypeMatrix::new_unphased(
            self.metadata.markers,
            columns,
            Arc::new(self.metadata.samples),
        )
    }
}
