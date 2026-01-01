//! Memory-mapped neighbor indices for large datasets
//!
//! This module provides memory-mapped storage for neighbor indices, allowing
//! efficient storage and access of large neighbor index structures that don't
//! fit in memory.

use crate::{NeighborsError, NeighborsResult};
#[cfg(feature = "memmap")]
use bytemuck::{bytes_of, cast_slice, from_bytes};
use bytemuck::{Pod, Zeroable};
#[cfg(feature = "memmap")]
use memmap2::{Mmap, MmapOptions};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::types::{Features, Float};
#[cfg(feature = "memmap")]
use std::fs::{File, OpenOptions};
#[cfg(feature = "memmap")]
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Header structure for memory-mapped neighbor index files
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct MmapHeader {
    /// Magic number for file format validation
    magic: u64,
    /// Version of the file format
    version: u32,
    /// Number of samples in the dataset
    n_samples: u32,
    /// Number of features per sample
    n_features: u32,
    /// Number of neighbors stored per sample
    k_neighbors: u32,
    /// Offset to neighbor indices data
    indices_offset: u64,
    /// Offset to neighbor distances data  
    distances_offset: u64,
    /// Offset to sample data
    samples_offset: u64,
}

// Implement Pod and Zeroable traits manually
unsafe impl Pod for MmapHeader {}
unsafe impl Zeroable for MmapHeader {}

impl Default for MmapHeader {
    fn default() -> Self {
        Self {
            magic: 0x4E4E4D4D41505F53, // "NNMMAP_S" in ASCII
            version: 1,
            n_samples: 0,
            n_features: 0,
            k_neighbors: 0,
            indices_offset: 0,
            distances_offset: 0,
            samples_offset: 0,
        }
    }
}

/// Memory-mapped neighbor index for efficient storage and retrieval of
/// precomputed neighbor relationships for large datasets
#[cfg(feature = "memmap")]
pub struct MmapNeighborIndex {
    /// File path for the memory-mapped index
    file_path: PathBuf,
    /// Memory-mapped file
    mmap: Option<Mmap>,
    /// Header information
    header: MmapHeader,
    /// Whether the index is read-only
    read_only: bool,
}

#[cfg(not(feature = "memmap"))]
pub struct MmapNeighborIndex {
    /// Placeholder when memmap feature is not enabled
    _phantom: std::marker::PhantomData<()>,
}

impl MmapNeighborIndex {
    /// Create a new memory-mapped neighbor index
    ///
    /// # Arguments
    /// * `file_path` - Path to the memory-mapped file
    /// * `n_samples` - Number of samples in the dataset
    /// * `n_features` - Number of features per sample
    /// * `k_neighbors` - Number of neighbors to store per sample
    ///
    /// # Returns
    /// * `NeighborsResult<Self>` - New memory-mapped neighbor index
    #[cfg(feature = "memmap")]
    pub fn new<P: AsRef<Path>>(
        file_path: P,
        n_samples: usize,
        n_features: usize,
        k_neighbors: usize,
    ) -> NeighborsResult<Self> {
        let file_path = file_path.as_ref().to_path_buf();

        let header = MmapHeader {
            n_samples: n_samples as u32,
            n_features: n_features as u32,
            k_neighbors: k_neighbors as u32,
            indices_offset: std::mem::size_of::<MmapHeader>() as u64,
            distances_offset: std::mem::size_of::<MmapHeader>() as u64
                + (n_samples * k_neighbors * std::mem::size_of::<u32>()) as u64,
            samples_offset: std::mem::size_of::<MmapHeader>() as u64
                + (n_samples * k_neighbors * std::mem::size_of::<u32>()) as u64
                + (n_samples * k_neighbors * std::mem::size_of::<Float>()) as u64,
            ..Default::default()
        };

        Ok(Self {
            file_path,
            mmap: None,
            header,
            read_only: false,
        })
    }

    #[cfg(not(feature = "memmap"))]
    pub fn new<P: AsRef<Path>>(
        _file_path: P,
        _n_samples: usize,
        _n_features: usize,
        _k_neighbors: usize,
    ) -> NeighborsResult<Self> {
        Err(NeighborsError::InvalidInput(
            "Memory mapping feature not enabled. Enable 'memmap' feature to use this functionality.".to_string()
        ))
    }

    /// Create the memory-mapped file with proper size allocation
    #[cfg(feature = "memmap")]
    pub fn create(&mut self) -> NeighborsResult<()> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.file_path)
            .map_err(|e| NeighborsError::InvalidInput(format!("Failed to create file: {}", e)))?;

        // Calculate total file size needed
        let total_size = self.header.samples_offset
            + (self.header.n_samples as u64
                * self.header.n_features as u64
                * std::mem::size_of::<Float>() as u64);

        // Resize file to required size
        file.set_len(total_size)
            .map_err(|e| NeighborsError::InvalidInput(format!("Failed to resize file: {}", e)))?;

        // Write header
        let mut writer = BufWriter::new(file);
        writer
            .write_all(bytes_of(&self.header))
            .map_err(|e| NeighborsError::InvalidInput(format!("Failed to write header: {}", e)))?;

        writer
            .flush()
            .map_err(|e| NeighborsError::InvalidInput(format!("Failed to flush file: {}", e)))?;

        self.read_only = false;
        Ok(())
    }

    #[cfg(not(feature = "memmap"))]
    pub fn create(&mut self) -> NeighborsResult<()> {
        Err(NeighborsError::InvalidInput(
            "Memory mapping feature not enabled.".to_string(),
        ))
    }

    /// Open an existing memory-mapped neighbor index file
    #[cfg(feature = "memmap")]
    pub fn open<P: AsRef<Path>>(file_path: P, read_only: bool) -> NeighborsResult<Self> {
        let file_path = file_path.as_ref().to_path_buf();

        if !file_path.exists() {
            return Err(NeighborsError::InvalidInput(format!(
                "File does not exist: {}",
                file_path.display()
            )));
        }

        let file = if read_only {
            File::open(&file_path)
        } else {
            OpenOptions::new().read(true).write(true).open(&file_path)
        }
        .map_err(|e| NeighborsError::InvalidInput(format!("Failed to open file: {}", e)))?;

        // Read header
        let mut reader = BufReader::new(file);
        let mut header_bytes = vec![0u8; std::mem::size_of::<MmapHeader>()];
        reader
            .read_exact(&mut header_bytes)
            .map_err(|e| NeighborsError::InvalidInput(format!("Failed to read header: {}", e)))?;

        let header: MmapHeader = *from_bytes(&header_bytes);

        // Validate magic number
        if header.magic != MmapHeader::default().magic {
            return Err(NeighborsError::InvalidInput(
                "Invalid file format: magic number mismatch".to_string(),
            ));
        }

        Ok(Self {
            file_path,
            mmap: None,
            header,
            read_only,
        })
    }

    #[cfg(not(feature = "memmap"))]
    pub fn open<P: AsRef<Path>>(_file_path: P, _read_only: bool) -> NeighborsResult<Self> {
        Err(NeighborsError::InvalidInput(
            "Memory mapping feature not enabled.".to_string(),
        ))
    }

    /// Memory-map the file for efficient access
    #[cfg(feature = "memmap")]
    pub fn map(&mut self) -> NeighborsResult<()> {
        let file = File::open(&self.file_path).map_err(|e| {
            NeighborsError::InvalidInput(format!("Failed to open file for mapping: {}", e))
        })?;

        let mmap = unsafe {
            MmapOptions::new().map(&file).map_err(|e| {
                NeighborsError::InvalidInput(format!("Failed to memory map file: {}", e))
            })?
        };

        self.mmap = Some(mmap);
        Ok(())
    }

    #[cfg(not(feature = "memmap"))]
    pub fn map(&mut self) -> NeighborsResult<()> {
        Err(NeighborsError::InvalidInput(
            "Memory mapping feature not enabled.".to_string(),
        ))
    }

    /// Store neighbor indices for a batch of samples
    ///
    /// # Arguments
    /// * `sample_indices` - Indices of samples to store neighbors for
    /// * `neighbor_indices` - 2D array of neighbor indices [n_samples, k_neighbors]
    /// * `neighbor_distances` - 2D array of neighbor distances [n_samples, k_neighbors]
    ///
    /// # Returns
    /// * `NeighborsResult<()>` - Success or error
    #[cfg(feature = "memmap")]
    pub fn store_neighbors(
        &mut self,
        sample_indices: &[usize],
        neighbor_indices: &Array2<u32>,
        neighbor_distances: &Array2<Float>,
    ) -> NeighborsResult<()> {
        if self.read_only {
            return Err(NeighborsError::InvalidInput(
                "Cannot write to read-only memory-mapped index".to_string(),
            ));
        }

        if neighbor_indices.shape()[1] != self.header.k_neighbors as usize {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![sample_indices.len(), self.header.k_neighbors as usize],
                actual: neighbor_indices.shape().to_vec(),
            });
        }

        let file = OpenOptions::new()
            .write(true)
            .open(&self.file_path)
            .map_err(|e| {
                NeighborsError::InvalidInput(format!("Failed to open file for writing: {}", e))
            })?;

        let mut writer = BufWriter::new(file);

        for (i, &sample_idx) in sample_indices.iter().enumerate() {
            if sample_idx >= self.header.n_samples as usize {
                return Err(NeighborsError::InvalidInput(format!(
                    "Sample index {} out of bounds",
                    sample_idx
                )));
            }

            // Write neighbor indices
            let indices_offset = self.header.indices_offset
                + (sample_idx * self.header.k_neighbors as usize * std::mem::size_of::<u32>())
                    as u64;

            writer
                .seek(SeekFrom::Start(indices_offset))
                .map_err(|e| NeighborsError::InvalidInput(format!("Failed to seek: {}", e)))?;

            let row_indices = neighbor_indices.row(i);
            let indices_bytes = cast_slice(row_indices.as_slice().unwrap());
            writer.write_all(indices_bytes).map_err(|e| {
                NeighborsError::InvalidInput(format!("Failed to write indices: {}", e))
            })?;

            // Write neighbor distances
            let distances_offset = self.header.distances_offset
                + (sample_idx * self.header.k_neighbors as usize * std::mem::size_of::<Float>())
                    as u64;

            writer
                .seek(SeekFrom::Start(distances_offset))
                .map_err(|e| NeighborsError::InvalidInput(format!("Failed to seek: {}", e)))?;

            let row_distances = neighbor_distances.row(i);
            let distances_bytes = cast_slice(row_distances.as_slice().unwrap());
            writer.write_all(distances_bytes).map_err(|e| {
                NeighborsError::InvalidInput(format!("Failed to write distances: {}", e))
            })?;
        }

        writer
            .flush()
            .map_err(|e| NeighborsError::InvalidInput(format!("Failed to flush: {}", e)))?;

        Ok(())
    }

    #[cfg(not(feature = "memmap"))]
    pub fn store_neighbors(
        &mut self,
        _sample_indices: &[usize],
        _neighbor_indices: &Array2<u32>,
        _neighbor_distances: &Array2<Float>,
    ) -> NeighborsResult<()> {
        Err(NeighborsError::InvalidInput(
            "Memory mapping feature not enabled.".to_string(),
        ))
    }

    /// Retrieve neighbor indices for a sample
    ///
    /// # Arguments
    /// * `sample_idx` - Index of the sample to retrieve neighbors for
    ///
    /// # Returns
    /// * `NeighborsResult<(Array1<u32>, Array1`<Float>`)>` - Neighbor indices and distances
    #[cfg(feature = "memmap")]
    pub fn get_neighbors(
        &self,
        sample_idx: usize,
    ) -> NeighborsResult<(Array1<u32>, Array1<Float>)> {
        if sample_idx >= self.header.n_samples as usize {
            return Err(NeighborsError::InvalidInput(format!(
                "Sample index {} out of bounds",
                sample_idx
            )));
        }

        let mmap = self.mmap.as_ref().ok_or_else(|| {
            NeighborsError::InvalidInput("Index not memory-mapped. Call map() first.".to_string())
        })?;

        // Read neighbor indices
        let indices_offset = self.header.indices_offset as usize
            + sample_idx * self.header.k_neighbors as usize * std::mem::size_of::<u32>();
        let indices_end =
            indices_offset + self.header.k_neighbors as usize * std::mem::size_of::<u32>();

        if indices_end > mmap.len() {
            return Err(NeighborsError::InvalidInput(
                "Index data out of bounds".to_string(),
            ));
        }

        let indices_slice: &[u32] = cast_slice(&mmap[indices_offset..indices_end]);
        let neighbor_indices = Array1::from_vec(indices_slice.to_vec());

        // Read neighbor distances
        let distances_offset = self.header.distances_offset as usize
            + sample_idx * self.header.k_neighbors as usize * std::mem::size_of::<Float>();
        let distances_end =
            distances_offset + self.header.k_neighbors as usize * std::mem::size_of::<Float>();

        if distances_end > mmap.len() {
            return Err(NeighborsError::InvalidInput(
                "Distance data out of bounds".to_string(),
            ));
        }

        let distances_slice: &[Float] = cast_slice(&mmap[distances_offset..distances_end]);
        let neighbor_distances = Array1::from_vec(distances_slice.to_vec());

        Ok((neighbor_indices, neighbor_distances))
    }

    #[cfg(not(feature = "memmap"))]
    pub fn get_neighbors(
        &self,
        _sample_idx: usize,
    ) -> NeighborsResult<(Array1<u32>, Array1<Float>)> {
        Err(NeighborsError::InvalidInput(
            "Memory mapping feature not enabled.".to_string(),
        ))
    }

    /// Store the original sample data in the memory-mapped file
    ///
    /// # Arguments
    /// * `samples` - Sample data to store [n_samples, n_features]
    ///
    /// # Returns
    /// * `NeighborsResult<()>` - Success or error
    #[cfg(feature = "memmap")]
    pub fn store_samples(&mut self, samples: &Features) -> NeighborsResult<()> {
        if self.read_only {
            return Err(NeighborsError::InvalidInput(
                "Cannot write to read-only memory-mapped index".to_string(),
            ));
        }

        if samples.shape()
            != [
                self.header.n_samples as usize,
                self.header.n_features as usize,
            ]
        {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![
                    self.header.n_samples as usize,
                    self.header.n_features as usize,
                ],
                actual: samples.shape().to_vec(),
            });
        }

        let file = OpenOptions::new()
            .write(true)
            .open(&self.file_path)
            .map_err(|e| {
                NeighborsError::InvalidInput(format!("Failed to open file for writing: {}", e))
            })?;

        let mut writer = BufWriter::new(file);
        writer
            .seek(SeekFrom::Start(self.header.samples_offset))
            .map_err(|e| NeighborsError::InvalidInput(format!("Failed to seek: {}", e)))?;

        // Write sample data row by row
        for sample in samples.axis_iter(scirs2_core::ndarray::Axis(0)) {
            let sample_bytes = cast_slice(sample.as_slice().unwrap());
            writer.write_all(sample_bytes).map_err(|e| {
                NeighborsError::InvalidInput(format!("Failed to write sample data: {}", e))
            })?;
        }

        writer
            .flush()
            .map_err(|e| NeighborsError::InvalidInput(format!("Failed to flush: {}", e)))?;

        Ok(())
    }

    #[cfg(not(feature = "memmap"))]
    pub fn store_samples(&mut self, _samples: &Features) -> NeighborsResult<()> {
        Err(NeighborsError::InvalidInput(
            "Memory mapping feature not enabled.".to_string(),
        ))
    }

    /// Retrieve a specific sample from the memory-mapped file
    ///
    /// # Arguments
    /// * `sample_idx` - Index of the sample to retrieve
    ///
    /// # Returns
    /// * `NeighborsResult<Array1`<Float>`>` - Sample data
    #[cfg(feature = "memmap")]
    pub fn get_sample(&self, sample_idx: usize) -> NeighborsResult<Array1<Float>> {
        if sample_idx >= self.header.n_samples as usize {
            return Err(NeighborsError::InvalidInput(format!(
                "Sample index {} out of bounds",
                sample_idx
            )));
        }

        let mmap = self.mmap.as_ref().ok_or_else(|| {
            NeighborsError::InvalidInput("Index not memory-mapped. Call map() first.".to_string())
        })?;

        let sample_offset = self.header.samples_offset as usize
            + sample_idx * self.header.n_features as usize * std::mem::size_of::<Float>();
        let sample_end =
            sample_offset + self.header.n_features as usize * std::mem::size_of::<Float>();

        if sample_end > mmap.len() {
            return Err(NeighborsError::InvalidInput(
                "Sample data out of bounds".to_string(),
            ));
        }

        let sample_slice: &[Float] = cast_slice(&mmap[sample_offset..sample_end]);
        Ok(Array1::from_vec(sample_slice.to_vec()))
    }

    #[cfg(not(feature = "memmap"))]
    pub fn get_sample(&self, _sample_idx: usize) -> NeighborsResult<Array1<Float>> {
        Err(NeighborsError::InvalidInput(
            "Memory mapping feature not enabled.".to_string(),
        ))
    }

    /// Get metadata about the memory-mapped index
    #[cfg(feature = "memmap")]
    pub fn metadata(&self) -> (usize, usize, usize) {
        (
            self.header.n_samples as usize,
            self.header.n_features as usize,
            self.header.k_neighbors as usize,
        )
    }

    #[cfg(not(feature = "memmap"))]
    pub fn metadata(&self) -> (usize, usize, usize) {
        (0, 0, 0)
    }

    /// Get the file path of the memory-mapped index
    #[cfg(feature = "memmap")]
    pub fn file_path(&self) -> &Path {
        &self.file_path
    }

    #[cfg(not(feature = "memmap"))]
    pub fn file_path(&self) -> &Path {
        Path::new("")
    }

    /// Check if the index is read-only
    #[cfg(feature = "memmap")]
    pub fn is_read_only(&self) -> bool {
        self.read_only
    }

    #[cfg(not(feature = "memmap"))]
    pub fn is_read_only(&self) -> bool {
        true
    }

    /// Check if the index is currently memory-mapped
    #[cfg(feature = "memmap")]
    pub fn is_mapped(&self) -> bool {
        self.mmap.is_some()
    }

    #[cfg(not(feature = "memmap"))]
    pub fn is_mapped(&self) -> bool {
        false
    }

    /// Get the size of the memory-mapped file in bytes
    #[cfg(feature = "memmap")]
    pub fn file_size(&self) -> Option<usize> {
        self.mmap.as_ref().map(|mmap| mmap.len())
    }

    #[cfg(not(feature = "memmap"))]
    pub fn file_size(&self) -> Option<usize> {
        None
    }
}

/// Builder for creating memory-mapped neighbor indices with various configurations
pub struct MmapNeighborIndexBuilder {
    file_path: Option<PathBuf>,
    n_samples: Option<usize>,
    n_features: Option<usize>,
    k_neighbors: Option<usize>,
    read_only: bool,
}

impl Default for MmapNeighborIndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl MmapNeighborIndexBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            file_path: None,
            n_samples: None,
            n_features: None,
            k_neighbors: None,
            read_only: false,
        }
    }

    /// Set the file path for the memory-mapped index
    pub fn file_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.file_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Set the number of samples
    pub fn n_samples(mut self, n_samples: usize) -> Self {
        self.n_samples = Some(n_samples);
        self
    }

    /// Set the number of features
    pub fn n_features(mut self, n_features: usize) -> Self {
        self.n_features = Some(n_features);
        self
    }

    /// Set the number of neighbors to store per sample
    pub fn k_neighbors(mut self, k_neighbors: usize) -> Self {
        self.k_neighbors = Some(k_neighbors);
        self
    }

    /// Set whether the index should be read-only
    pub fn read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }

    /// Build the memory-mapped neighbor index
    pub fn build(self) -> NeighborsResult<MmapNeighborIndex> {
        let file_path = self
            .file_path
            .ok_or_else(|| NeighborsError::InvalidInput("File path not specified".to_string()))?;

        if file_path.exists() && file_path.metadata().map(|m| m.len()).unwrap_or(0) > 0 {
            // Open existing file if it's not empty
            MmapNeighborIndex::open(file_path, self.read_only)
        } else {
            // Create new file
            let n_samples = self.n_samples.ok_or_else(|| {
                NeighborsError::InvalidInput("Number of samples not specified".to_string())
            })?;
            let n_features = self.n_features.ok_or_else(|| {
                NeighborsError::InvalidInput("Number of features not specified".to_string())
            })?;
            let k_neighbors = self.k_neighbors.ok_or_else(|| {
                NeighborsError::InvalidInput("Number of neighbors not specified".to_string())
            })?;

            let mut index = MmapNeighborIndex::new(file_path, n_samples, n_features, k_neighbors)?;
            index.create()?;
            Ok(index)
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "memmap")]
    use scirs2_core::ndarray::arr2;

    use tempfile::NamedTempFile;

    #[test]
    #[cfg(feature = "memmap")]
    fn test_mmap_neighbor_index_creation() {
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path();

        let mut index = MmapNeighborIndex::new(file_path, 100, 10, 5).unwrap();
        assert!(index.create().is_ok());

        let (n_samples, n_features, k_neighbors) = index.metadata();
        assert_eq!(n_samples, 100);
        assert_eq!(n_features, 10);
        assert_eq!(k_neighbors, 5);
    }

    #[test]
    #[cfg(feature = "memmap")]
    fn test_mmap_neighbor_index_store_and_retrieve() {
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path();

        let mut index = MmapNeighborIndex::new(file_path, 3, 2, 2).unwrap();
        index.create().unwrap();
        index.map().unwrap();

        // Store some neighbor data
        let sample_indices = vec![0, 1];
        let neighbor_indices = arr2(&[[1, 2], [0, 2]]);
        let neighbor_distances = arr2(&[[0.5, 1.0], [0.3, 0.8]]);

        index
            .store_neighbors(&sample_indices, &neighbor_indices, &neighbor_distances)
            .unwrap();

        // Retrieve the data
        let (retrieved_indices, retrieved_distances) = index.get_neighbors(0).unwrap();
        assert_eq!(retrieved_indices.as_slice().unwrap(), &[1, 2]);
        assert!((retrieved_distances[0] - 0.5).abs() < 1e-6);
        assert!((retrieved_distances[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    #[cfg(feature = "memmap")]
    fn test_mmap_neighbor_index_builder() {
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path();

        let index = MmapNeighborIndexBuilder::new()
            .file_path(file_path)
            .n_samples(50)
            .n_features(4)
            .k_neighbors(3)
            .build()
            .unwrap();

        let (n_samples, n_features, k_neighbors) = index.metadata();
        assert_eq!(n_samples, 50);
        assert_eq!(n_features, 4);
        assert_eq!(k_neighbors, 3);
    }

    #[test]
    #[cfg(feature = "memmap")]
    fn test_mmap_neighbor_index_error_cases() {
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path();

        let mut index = MmapNeighborIndex::new(file_path, 2, 3, 2).unwrap();
        index.create().unwrap();

        // Test invalid sample index
        assert!(index.get_neighbors(5).is_err());

        // Test shape mismatch
        let sample_indices = vec![0];
        let wrong_neighbor_indices = arr2(&[[1, 2, 3]]); // Wrong k_neighbors
        let neighbor_distances = arr2(&[[0.5, 1.0]]);

        assert!(index
            .store_neighbors(
                &sample_indices,
                &wrong_neighbor_indices,
                &neighbor_distances
            )
            .is_err());
    }

    #[test]
    #[cfg(not(feature = "memmap"))]
    fn test_mmap_without_feature() {
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path();

        let result = MmapNeighborIndex::new(file_path, 100, 10, 5);
        assert!(result.is_err());
    }
}
