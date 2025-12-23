//! Memory-mapped distance matrix computation for large datasets
//!
//! This module provides memory-mapped distance matrix computation that allows
//! processing of datasets larger than available RAM by storing intermediate
//! results on disk and accessing them through memory mapping.

use memmap2::{MmapMut, MmapOptions};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use tempfile::TempDir;

use sklears_core::{
    error::{Result, SklearsError},
    types::{Array2, Float},
};

use crate::simd_distances::{simd_distance, SimdDistanceMetric};

/// Configuration for memory-mapped distance computation
#[derive(Debug, Clone)]
pub struct MemoryMappedConfig {
    /// Directory for temporary files (None for system temp)
    pub temp_dir: Option<PathBuf>,
    /// Chunk size for processing (number of samples per chunk)
    pub chunk_size: usize,
    /// Distance metric to use
    pub metric: SimdDistanceMetric,
    /// Whether to use compression for temporary files
    pub use_compression: bool,
    /// Whether to keep temporary files for debugging
    pub keep_temp_files: bool,
}

impl Default for MemoryMappedConfig {
    fn default() -> Self {
        Self {
            temp_dir: None,
            chunk_size: 1000,
            metric: SimdDistanceMetric::Euclidean,
            use_compression: false,
            keep_temp_files: false,
        }
    }
}

/// Memory-mapped distance matrix for large-scale distance computation
pub struct MemoryMappedDistanceMatrix {
    /// Configuration
    config: MemoryMappedConfig,
    /// Number of samples
    n_samples: usize,
    /// Temporary directory
    temp_dir: TempDir,
    /// Memory-mapped file for distance matrix
    distance_file: File,
    /// Memory map of the distance matrix
    distance_mmap: Option<MmapMut>,
    /// Size of each distance value in bytes
    value_size: usize,
}

impl MemoryMappedDistanceMatrix {
    /// Create a new memory-mapped distance matrix
    ///
    /// # Arguments
    /// * `n_samples` - Number of samples in the dataset
    /// * `config` - Configuration for memory mapping
    pub fn new(n_samples: usize, config: MemoryMappedConfig) -> Result<Self> {
        // Create temporary directory
        let temp_dir = if let Some(ref dir) = config.temp_dir {
            TempDir::new_in(dir)
        } else {
            TempDir::new()
        }
        .map_err(|e| SklearsError::Other(format!("Failed to create temp directory: {}", e)))?;

        // Calculate required file size for distance matrix
        // We store only the upper triangle since distance matrices are symmetric
        let n_pairs = (n_samples * (n_samples - 1)) / 2;
        let value_size = std::mem::size_of::<Float>();
        let file_size = n_pairs * value_size;

        // Create memory-mapped file
        let distance_file_path = temp_dir.path().join("distance_matrix.bin");
        let distance_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&distance_file_path)
            .map_err(|e| SklearsError::Other(format!("Failed to create distance file: {}", e)))?;

        // Set file size
        distance_file
            .set_len(file_size as u64)
            .map_err(|e| SklearsError::Other(format!("Failed to set file size: {}", e)))?;

        Ok(Self {
            config,
            n_samples,
            temp_dir,
            distance_file,
            distance_mmap: None,
            value_size,
        })
    }

    /// Initialize the memory map
    fn initialize_mmap(&mut self) -> Result<()> {
        if self.distance_mmap.is_none() {
            let mmap = unsafe {
                MmapOptions::new()
                    .map_mut(&self.distance_file)
                    .map_err(|e| {
                        SklearsError::Other(format!("Failed to create memory map: {}", e))
                    })?
            };
            self.distance_mmap = Some(mmap);
        }
        Ok(())
    }

    /// Compute distance matrix in chunks and store in memory-mapped file
    ///
    /// # Arguments
    /// * `data` - Input data matrix (n_samples × n_features)
    pub fn compute_distances(&mut self, data: &Array2<Float>) -> Result<()> {
        if data.nrows() != self.n_samples {
            return Err(SklearsError::InvalidInput(format!(
                "Data has {} samples but expected {}",
                data.nrows(),
                self.n_samples
            )));
        }

        self.initialize_mmap()?;

        let chunk_size = self.config.chunk_size;
        let n_chunks = (self.n_samples + chunk_size - 1) / chunk_size;

        // Process data in chunks to manage memory usage
        for i_chunk in 0..n_chunks {
            let i_start = i_chunk * chunk_size;
            let i_end = (i_start + chunk_size).min(self.n_samples);

            for j_chunk in i_chunk..n_chunks {
                let j_start = j_chunk * chunk_size;
                let j_end = (j_start + chunk_size).min(self.n_samples);

                // Compute distances between chunk i and chunk j
                self.compute_chunk_distances(data, i_start, i_end, j_start, j_end)?;
            }

            // Log progress
            if (i_chunk + 1) % 10 == 0 || i_chunk == n_chunks - 1 {
                eprintln!("Processed chunk {} of {}", i_chunk + 1, n_chunks);
            }
        }

        Ok(())
    }

    /// Compute distances between two chunks
    fn compute_chunk_distances(
        &mut self,
        data: &Array2<Float>,
        i_start: usize,
        i_end: usize,
        j_start: usize,
        j_end: usize,
    ) -> Result<()> {
        for i in i_start..i_end {
            let j_min = if i_start == j_start { i + 1 } else { j_start };
            for j in j_min..j_end {
                if i < j {
                    let row_i = data.row(i);
                    let row_j = data.row(j);

                    let distance =
                        simd_distance(&row_i, &row_j, self.config.metric).map_err(|e| {
                            SklearsError::NumericalError(format!(
                                "SIMD distance computation failed: {}",
                                e
                            ))
                        })?;

                    self.set_distance(i, j, distance)?;
                }
            }
        }
        Ok(())
    }

    /// Convert (i, j) indices to linear index in upper triangular storage
    fn indices_to_linear(&self, i: usize, j: usize) -> usize {
        assert!(i < j, "Only upper triangle is stored (i must be < j)");
        assert!(i < self.n_samples && j < self.n_samples);

        // Formula for upper triangular matrix indexing
        let n = self.n_samples;
        i * n - (i * (i + 1)) / 2 + j - i - 1
    }

    /// Set distance value at position (i, j)
    fn set_distance(&mut self, i: usize, j: usize, distance: Float) -> Result<()> {
        let linear_index = self.indices_to_linear(i, j);
        let byte_offset = linear_index * self.value_size;

        if let Some(ref mut mmap) = self.distance_mmap {
            let bytes = distance.to_le_bytes();
            let start = byte_offset;
            let end = start + self.value_size;

            if end <= mmap.len() {
                mmap[start..end].copy_from_slice(&bytes);
            } else {
                return Err(SklearsError::Other(format!(
                    "Index out of bounds: {} >= {}",
                    end,
                    mmap.len()
                )));
            }
        }
        Ok(())
    }

    /// Get distance value at position (i, j)
    pub fn get_distance(&self, i: usize, j: usize) -> Result<Float> {
        if i == j {
            return Ok(0.0);
        }

        let (min_idx, max_idx) = if i < j { (i, j) } else { (j, i) };
        let linear_index = self.indices_to_linear(min_idx, max_idx);
        let byte_offset = linear_index * self.value_size;

        if let Some(ref mmap) = self.distance_mmap {
            let start = byte_offset;
            let end = start + self.value_size;

            if end <= mmap.len() {
                let bytes: [u8; 8] = mmap[start..end].try_into().map_err(|_| {
                    SklearsError::Other("Failed to read distance bytes".to_string())
                })?;
                Ok(Float::from_le_bytes(bytes))
            } else {
                Err(SklearsError::Other(format!(
                    "Index out of bounds: {} >= {}",
                    end,
                    mmap.len()
                )))
            }
        } else {
            Err(SklearsError::Other(
                "Memory map not initialized".to_string(),
            ))
        }
    }

    /// Get k-nearest neighbors for a specific sample
    ///
    /// # Arguments
    /// * `sample_idx` - Index of the sample
    /// * `k` - Number of nearest neighbors to find
    ///
    /// # Returns
    /// Vector of (neighbor_index, distance) pairs sorted by distance
    pub fn get_k_nearest_neighbors(
        &self,
        sample_idx: usize,
        k: usize,
    ) -> Result<Vec<(usize, Float)>> {
        if sample_idx >= self.n_samples {
            return Err(SklearsError::InvalidInput(format!(
                "Sample index {} out of bounds (max: {})",
                sample_idx,
                self.n_samples - 1
            )));
        }

        let mut neighbors = Vec::new();

        // Collect all distances for this sample
        for other_idx in 0..self.n_samples {
            if other_idx != sample_idx {
                let distance = self.get_distance(sample_idx, other_idx)?;
                neighbors.push((other_idx, distance));
            }
        }

        // Sort by distance and take k nearest
        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        neighbors.truncate(k);

        Ok(neighbors)
    }

    /// Get all neighbors within a specific radius
    ///
    /// # Arguments
    /// * `sample_idx` - Index of the sample
    /// * `radius` - Maximum distance for neighbors
    ///
    /// # Returns
    /// Vector of (neighbor_index, distance) pairs within radius
    pub fn get_neighbors_within_radius(
        &self,
        sample_idx: usize,
        radius: Float,
    ) -> Result<Vec<(usize, Float)>> {
        if sample_idx >= self.n_samples {
            return Err(SklearsError::InvalidInput(format!(
                "Sample index {} out of bounds (max: {})",
                sample_idx,
                self.n_samples - 1
            )));
        }

        let mut neighbors = Vec::new();

        for other_idx in 0..self.n_samples {
            if other_idx != sample_idx {
                let distance = self.get_distance(sample_idx, other_idx)?;
                if distance <= radius {
                    neighbors.push((other_idx, distance));
                }
            }
        }

        // Sort by distance
        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        Ok(neighbors)
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> MemoryStats {
        let n_pairs = (self.n_samples * (self.n_samples - 1)) / 2;
        let matrix_size_bytes = n_pairs * self.value_size;
        let matrix_size_mb = matrix_size_bytes as f64 / (1024.0 * 1024.0);
        let matrix_size_gb = matrix_size_mb / 1024.0;

        MemoryStats {
            n_samples: self.n_samples,
            n_pairs,
            matrix_size_bytes,
            matrix_size_mb,
            matrix_size_gb,
            temp_dir_path: self.temp_dir.path().to_path_buf(),
        }
    }

    /// Export distance matrix to a standard format
    ///
    /// This method exports the distance matrix as a regular Array2 for smaller datasets
    /// or when you need to integrate with other algorithms that expect in-memory matrices.
    ///
    /// Warning: This will load the entire distance matrix into memory.
    pub fn to_array(&self) -> Result<Array2<Float>> {
        if self.n_samples > 10000 {
            eprintln!("Warning: Converting large distance matrix ({} samples) to Array2. This may use significant memory.", self.n_samples);
        }

        let mut matrix = Array2::zeros((self.n_samples, self.n_samples));

        for i in 0..self.n_samples {
            for j in i + 1..self.n_samples {
                let distance = self.get_distance(i, j)?;
                matrix[[i, j]] = distance;
                matrix[[j, i]] = distance; // Symmetric
            }
        }

        Ok(matrix)
    }

    /// Flush any pending writes to disk
    pub fn sync(&mut self) -> Result<()> {
        if let Some(ref mut mmap) = self.distance_mmap {
            mmap.flush()
                .map_err(|e| SklearsError::Other(format!("Failed to sync memory map: {}", e)))?;
        }
        Ok(())
    }
}

/// Memory usage statistics for the distance matrix
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Number of samples
    pub n_samples: usize,
    /// Number of distance pairs stored
    pub n_pairs: usize,
    /// Size of distance matrix in bytes
    pub matrix_size_bytes: usize,
    /// Size of distance matrix in MB
    pub matrix_size_mb: f64,
    /// Size of distance matrix in GB
    pub matrix_size_gb: f64,
    /// Path to temporary directory
    pub temp_dir_path: PathBuf,
}

impl std::fmt::Display for MemoryStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MemoryStats {{ samples: {}, pairs: {}, size: {:.2} MB ({:.2} GB), temp: {:?} }}",
            self.n_samples,
            self.n_pairs,
            self.matrix_size_mb,
            self.matrix_size_gb,
            self.temp_dir_path
        )
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_memory_mapped_small_dataset() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],];

        let config = MemoryMappedConfig {
            chunk_size: 2,
            ..Default::default()
        };

        let mut mmap_matrix = MemoryMappedDistanceMatrix::new(4, config).unwrap();
        mmap_matrix.compute_distances(&data).unwrap();

        // Test distance computation
        let dist_01 = mmap_matrix.get_distance(0, 1).unwrap();
        let dist_02 = mmap_matrix.get_distance(0, 2).unwrap();
        let dist_03 = mmap_matrix.get_distance(0, 3).unwrap();

        // Verify expected distances
        assert!((dist_01 - 1.0).abs() < 1e-6); // Distance between (0,0) and (1,0)
        assert!((dist_02 - 1.0).abs() < 1e-6); // Distance between (0,0) and (0,1)
        assert!((dist_03 - 2.0_f64.sqrt()).abs() < 1e-6); // Distance between (0,0) and (1,1)

        // Test symmetry
        let dist_10 = mmap_matrix.get_distance(1, 0).unwrap();
        assert!((dist_01 - dist_10).abs() < 1e-10);
    }

    #[test]
    fn test_k_nearest_neighbors() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [10.0, 10.0],];

        let config = MemoryMappedConfig::default();
        let mut mmap_matrix = MemoryMappedDistanceMatrix::new(4, config).unwrap();
        mmap_matrix.compute_distances(&data).unwrap();

        // Get 2 nearest neighbors of point 0
        let neighbors = mmap_matrix.get_k_nearest_neighbors(0, 2).unwrap();

        assert_eq!(neighbors.len(), 2);

        // Points 1 and 2 should be the nearest to point 0
        let neighbor_indices: Vec<usize> = neighbors.iter().map(|&(idx, _)| idx).collect();
        assert!(neighbor_indices.contains(&1));
        assert!(neighbor_indices.contains(&2));
        assert!(!neighbor_indices.contains(&3)); // Point 3 is farthest
    }

    #[test]
    fn test_neighbors_within_radius() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [10.0, 10.0],];

        let config = MemoryMappedConfig::default();
        let mut mmap_matrix = MemoryMappedDistanceMatrix::new(4, config).unwrap();
        mmap_matrix.compute_distances(&data).unwrap();

        // Get neighbors within radius 1.5 of point 0
        let neighbors = mmap_matrix.get_neighbors_within_radius(0, 1.5).unwrap();

        // Should include points 1 and 2 but not 3
        assert_eq!(neighbors.len(), 2);
        let neighbor_indices: Vec<usize> = neighbors.iter().map(|&(idx, _)| idx).collect();
        assert!(neighbor_indices.contains(&1));
        assert!(neighbor_indices.contains(&2));
        assert!(!neighbor_indices.contains(&3));
    }

    #[test]
    fn test_memory_stats() {
        let config = MemoryMappedConfig::default();
        let mmap_matrix = MemoryMappedDistanceMatrix::new(100, config).unwrap();

        let stats = mmap_matrix.memory_stats();
        assert_eq!(stats.n_samples, 100);
        assert_eq!(stats.n_pairs, (100 * 99) / 2); // Upper triangle
        assert!(stats.matrix_size_bytes > 0);
        assert!(stats.matrix_size_mb > 0.0);
    }

    #[test]
    fn test_indices_to_linear() {
        let config = MemoryMappedConfig::default();
        let mmap_matrix = MemoryMappedDistanceMatrix::new(5, config).unwrap();

        // Test a few specific index conversions
        // For n=5, upper triangle indices should be:
        // Row 0: (0,1)=0, (0,2)=1, (0,3)=2, (0,4)=3
        // Row 1: (1,2)=4, (1,3)=5, (1,4)=6
        // Row 2: (2,3)=7, (2,4)=8
        // Row 3: (3,4)=9
        assert_eq!(mmap_matrix.indices_to_linear(0, 1), 0);
        assert_eq!(mmap_matrix.indices_to_linear(0, 2), 1);
        assert_eq!(mmap_matrix.indices_to_linear(0, 3), 2);
        assert_eq!(mmap_matrix.indices_to_linear(1, 2), 4);
        assert_eq!(mmap_matrix.indices_to_linear(1, 3), 5);
        assert_eq!(mmap_matrix.indices_to_linear(2, 3), 7);
    }

    #[test]
    fn test_to_array_conversion() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0],];

        let config = MemoryMappedConfig::default();
        let mut mmap_matrix = MemoryMappedDistanceMatrix::new(3, config).unwrap();
        mmap_matrix.compute_distances(&data).unwrap();

        let array_matrix = mmap_matrix.to_array().unwrap();

        assert_eq!(array_matrix.shape(), &[3, 3]);

        // Check diagonal is zero
        for i in 0..3 {
            assert!((array_matrix[[i, i]] - 0.0).abs() < 1e-10);
        }

        // Check symmetry
        for i in 0..3 {
            for j in 0..3 {
                assert!((array_matrix[[i, j]] - array_matrix[[j, i]]).abs() < 1e-10);
            }
        }
    }
}
