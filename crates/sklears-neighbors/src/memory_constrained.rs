//! Memory-constrained algorithms for neighbor search with limited memory
//!
//! This module implements algorithms that can handle large datasets that don't fit
//! in memory by using external storage, disk-based operations, and streaming approaches.

use crate::distance::Distance;
use crate::{NeighborsError, NeighborsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::types::Float;
use std::collections::{BinaryHeap, HashMap};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Ordered float wrapper for use in BinaryHeap
#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedFloat(Float);

impl OrderedFloat {
    fn new(val: Float) -> Self {
        OrderedFloat(val)
    }
}

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// External memory k-nearest neighbors search
///
/// Uses disk-based storage to handle datasets larger than available RAM.
/// Implements a block-based approach where data is processed in chunks.
pub struct ExternalMemoryKNN {
    /// Number of neighbors to find
    k: usize,
    /// Distance metric
    distance: Distance,
    /// Block size (number of samples per block)
    block_size: usize,
    /// Directory for temporary files
    temp_dir: PathBuf,
    /// Training data file path
    data_file: Option<PathBuf>,
    /// Index file for fast access
    index_file: Option<PathBuf>,
    /// Number of samples in training data
    n_samples: usize,
    /// Number of features
    n_features: usize,
}

impl ExternalMemoryKNN {
    /// Create a new external memory KNN instance
    pub fn new(k: usize, temp_dir: impl AsRef<Path>) -> Self {
        Self {
            k,
            distance: Distance::Euclidean,
            block_size: 1000,
            temp_dir: temp_dir.as_ref().to_path_buf(),
            data_file: None,
            index_file: None,
            n_samples: 0,
            n_features: 0,
        }
    }

    /// Set distance metric
    pub fn with_distance(mut self, distance: Distance) -> Self {
        self.distance = distance;
        self
    }

    /// Set block size for processing
    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size;
        self
    }

    /// Fit the model to training data by storing it externally
    pub fn fit(&mut self, X: &ArrayView2<Float>) -> NeighborsResult<()> {
        if X.is_empty() {
            return Err(NeighborsError::EmptyInput);
        }

        self.n_samples = X.nrows();
        self.n_features = X.ncols();

        // Create temporary directory if it doesn't exist
        std::fs::create_dir_all(&self.temp_dir).map_err(|e| {
            NeighborsError::InvalidInput(format!("Failed to create temp dir: {}", e))
        })?;

        // Store data to disk
        let data_file = self.temp_dir.join("training_data.bin");
        let index_file = self.temp_dir.join("data_index.bin");

        self.store_data_to_disk(X, &data_file, &index_file)?;
        self.data_file = Some(data_file);
        self.index_file = Some(index_file);

        Ok(())
    }

    /// Store training data to disk in binary format
    fn store_data_to_disk(
        &self,
        X: &ArrayView2<Float>,
        data_file: &Path,
        index_file: &Path,
    ) -> NeighborsResult<()> {
        // Write data file
        let mut data_writer = BufWriter::new(File::create(data_file).map_err(|e| {
            NeighborsError::InvalidInput(format!("Failed to create data file: {}", e))
        })?);

        // Write header: n_samples, n_features
        data_writer
            .write_all(&self.n_samples.to_le_bytes())
            .map_err(|e| NeighborsError::InvalidInput(format!("Write error: {}", e)))?;
        data_writer
            .write_all(&self.n_features.to_le_bytes())
            .map_err(|e| NeighborsError::InvalidInput(format!("Write error: {}", e)))?;

        // Write data row by row
        for row in X.axis_iter(Axis(0)) {
            for &value in row.iter() {
                data_writer
                    .write_all(&value.to_le_bytes())
                    .map_err(|e| NeighborsError::InvalidInput(format!("Write error: {}", e)))?;
            }
        }

        data_writer
            .flush()
            .map_err(|e| NeighborsError::InvalidInput(format!("Flush error: {}", e)))?;

        // Write index file with block boundaries
        let mut index_writer = BufWriter::new(File::create(index_file).map_err(|e| {
            NeighborsError::InvalidInput(format!("Failed to create index file: {}", e))
        })?);

        let n_blocks = (self.n_samples + self.block_size - 1) / self.block_size;
        index_writer
            .write_all(&n_blocks.to_le_bytes())
            .map_err(|e| NeighborsError::InvalidInput(format!("Write error: {}", e)))?;

        let header_size = 2 * std::mem::size_of::<usize>();
        for block_idx in 0..n_blocks {
            let start_row = block_idx * self.block_size;
            let end_row = std::cmp::min(start_row + self.block_size, self.n_samples);
            let start_offset =
                header_size + start_row * self.n_features * std::mem::size_of::<Float>();
            let size = (end_row - start_row) * self.n_features * std::mem::size_of::<Float>();

            index_writer
                .write_all(&start_offset.to_le_bytes())
                .map_err(|e| NeighborsError::InvalidInput(format!("Write error: {}", e)))?;
            index_writer
                .write_all(&size.to_le_bytes())
                .map_err(|e| NeighborsError::InvalidInput(format!("Write error: {}", e)))?;
        }

        index_writer
            .flush()
            .map_err(|e| NeighborsError::InvalidInput(format!("Flush error: {}", e)))?;

        Ok(())
    }

    /// Load a block of data from disk
    fn load_block(&self, block_idx: usize) -> NeighborsResult<Array2<Float>> {
        let data_file = self
            .data_file
            .as_ref()
            .ok_or_else(|| NeighborsError::InvalidInput("Model not fitted".to_string()))?;
        let index_file = self
            .index_file
            .as_ref()
            .ok_or_else(|| NeighborsError::InvalidInput("Model not fitted".to_string()))?;

        // Read block info from index file
        let mut index_reader = BufReader::new(File::open(index_file).map_err(|e| {
            NeighborsError::InvalidInput(format!("Failed to open index file: {}", e))
        })?);

        let mut n_blocks_bytes = [0u8; std::mem::size_of::<usize>()];
        index_reader
            .read_exact(&mut n_blocks_bytes)
            .map_err(|e| NeighborsError::InvalidInput(format!("Read error: {}", e)))?;
        let n_blocks = usize::from_le_bytes(n_blocks_bytes);

        if block_idx >= n_blocks {
            return Err(NeighborsError::InvalidInput(
                "Block index out of range".to_string(),
            ));
        }

        // Seek to block info
        index_reader
            .seek(SeekFrom::Start(
                (std::mem::size_of::<usize>() + block_idx * 2 * std::mem::size_of::<usize>())
                    as u64,
            ))
            .map_err(|e| NeighborsError::InvalidInput(format!("Seek error: {}", e)))?;

        let mut offset_bytes = [0u8; std::mem::size_of::<usize>()];
        let mut size_bytes = [0u8; std::mem::size_of::<usize>()];
        index_reader
            .read_exact(&mut offset_bytes)
            .map_err(|e| NeighborsError::InvalidInput(format!("Read error: {}", e)))?;
        index_reader
            .read_exact(&mut size_bytes)
            .map_err(|e| NeighborsError::InvalidInput(format!("Read error: {}", e)))?;

        let offset = usize::from_le_bytes(offset_bytes);
        let size = usize::from_le_bytes(size_bytes);

        // Read block data
        let mut data_reader = BufReader::new(File::open(data_file).map_err(|e| {
            NeighborsError::InvalidInput(format!("Failed to open data file: {}", e))
        })?);

        data_reader
            .seek(SeekFrom::Start(offset as u64))
            .map_err(|e| NeighborsError::InvalidInput(format!("Seek error: {}", e)))?;

        let n_values = size / std::mem::size_of::<Float>();
        let n_rows = n_values / self.n_features;

        let mut buffer = vec![0u8; size];
        data_reader
            .read_exact(&mut buffer)
            .map_err(|e| NeighborsError::InvalidInput(format!("Read error: {}", e)))?;

        // Convert bytes to floats
        let mut data = Vec::with_capacity(n_values);
        for chunk in buffer.chunks_exact(std::mem::size_of::<Float>()) {
            let mut bytes = [0u8; std::mem::size_of::<Float>()];
            bytes.copy_from_slice(chunk);
            data.push(Float::from_le_bytes(bytes));
        }

        Array2::from_shape_vec((n_rows, self.n_features), data)
            .map_err(|e| NeighborsError::InvalidInput(format!("Array shape error: {}", e)))
    }

    /// Find k-nearest neighbors using external memory approach
    pub fn kneighbors(
        &self,
        query: &ArrayView1<Float>,
    ) -> NeighborsResult<(Array1<Float>, Array1<usize>)> {
        if query.len() != self.n_features {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![self.n_features],
                actual: vec![query.len()],
            });
        }

        let mut best_neighbors: BinaryHeap<(OrderedFloat, usize)> = BinaryHeap::new();
        let index_file = self
            .index_file
            .as_ref()
            .ok_or_else(|| NeighborsError::InvalidInput("Model not fitted".to_string()))?;

        // Read number of blocks
        let mut index_reader = BufReader::new(File::open(index_file).map_err(|e| {
            NeighborsError::InvalidInput(format!("Failed to open index file: {}", e))
        })?);

        let mut n_blocks_bytes = [0u8; std::mem::size_of::<usize>()];
        index_reader
            .read_exact(&mut n_blocks_bytes)
            .map_err(|e| NeighborsError::InvalidInput(format!("Read error: {}", e)))?;
        let n_blocks = usize::from_le_bytes(n_blocks_bytes);

        // Process each block
        for block_idx in 0..n_blocks {
            let block_data = self.load_block(block_idx)?;
            let start_idx = block_idx * self.block_size;

            // Compute distances for this block
            for (row_idx, row) in block_data.axis_iter(Axis(0)).enumerate() {
                let distance = self.distance.calculate(query, &row);
                let global_idx = start_idx + row_idx;

                if best_neighbors.len() < self.k {
                    best_neighbors.push((OrderedFloat::new(distance), global_idx));
                } else if let Some(&(max_distance, _)) = best_neighbors.peek() {
                    if distance < max_distance.0 {
                        best_neighbors.pop();
                        best_neighbors.push((OrderedFloat::new(distance), global_idx));
                    }
                }
            }
        }

        // Extract results
        let mut neighbors: Vec<(Float, usize)> = best_neighbors
            .into_vec()
            .into_iter()
            .map(|(d, i)| (d.0, i))
            .collect();
        neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let distances: Vec<Float> = neighbors.iter().map(|(d, _)| *d).collect();
        let indices: Vec<usize> = neighbors.iter().map(|(_, i)| *i).collect();

        Ok((Array1::from(distances), Array1::from(indices)))
    }

    /// Clean up temporary files
    pub fn cleanup(&self) -> NeighborsResult<()> {
        if let Some(ref data_file) = self.data_file {
            if data_file.exists() {
                std::fs::remove_file(data_file).map_err(|e| {
                    NeighborsError::InvalidInput(format!("Failed to remove data file: {}", e))
                })?;
            }
        }

        if let Some(ref index_file) = self.index_file {
            if index_file.exists() {
                std::fs::remove_file(index_file).map_err(|e| {
                    NeighborsError::InvalidInput(format!("Failed to remove index file: {}", e))
                })?;
            }
        }

        Ok(())
    }
}

impl Drop for ExternalMemoryKNN {
    fn drop(&mut self) {
        let _ = self.cleanup();
    }
}

/// Cache-oblivious algorithm for neighbor search
///
/// Uses a cache-friendly data layout and access patterns that work well
/// across different cache hierarchies without knowing cache parameters.
pub struct CacheObliviousNeighbors {
    /// Training data organized in cache-friendly layout
    data: Array2<Float>,
    /// Recursive subdivision of data space
    subdivision: Option<CacheObliviousTree>,
    /// Distance metric
    distance: Distance,
    /// Number of neighbors to find
    k: usize,
}

#[derive(Debug, Clone)]
struct CacheObliviousTree {
    /// Indices of points in this node
    point_indices: Vec<usize>,
    /// Bounding box of this node
    bbox_min: Array1<Float>,
    bbox_max: Array1<Float>,
    /// Left child
    left: Option<Box<CacheObliviousTree>>,
    /// Right child
    right: Option<Box<CacheObliviousTree>>,
    /// Split dimension
    split_dim: usize,
    /// Split value
    split_value: Float,
}

impl CacheObliviousNeighbors {
    /// Create a new cache-oblivious neighbors instance
    pub fn new(k: usize) -> Self {
        Self {
            data: Array2::zeros((0, 0)),
            subdivision: None,
            distance: Distance::Euclidean,
            k,
        }
    }

    /// Set distance metric
    pub fn with_distance(mut self, distance: Distance) -> Self {
        self.distance = distance;
        self
    }

    /// Fit the model to training data
    pub fn fit(&mut self, X: &ArrayView2<Float>) -> NeighborsResult<()> {
        if X.is_empty() {
            return Err(NeighborsError::EmptyInput);
        }

        self.data = X.to_owned();

        // Build cache-oblivious subdivision
        let indices: Vec<usize> = (0..X.nrows()).collect();
        let bbox_min = X
            .axis_iter(Axis(0))
            .fold(Array1::from_elem(X.ncols(), Float::INFINITY), |acc, row| {
                Array1::from_iter(acc.iter().zip(row.iter()).map(|(a, b)| a.min(*b)))
            });
        let bbox_max = X.axis_iter(Axis(0)).fold(
            Array1::from_elem(X.ncols(), Float::NEG_INFINITY),
            |acc, row| Array1::from_iter(acc.iter().zip(row.iter()).map(|(a, b)| a.max(*b))),
        );

        self.subdivision = Some(self.build_cache_oblivious_tree(indices, &bbox_min, &bbox_max));

        Ok(())
    }

    /// Build cache-oblivious tree recursively
    fn build_cache_oblivious_tree(
        &self,
        mut indices: Vec<usize>,
        bbox_min: &Array1<Float>,
        bbox_max: &Array1<Float>,
    ) -> CacheObliviousTree {
        // Base case: small number of points
        if indices.len() <= 8 {
            return CacheObliviousTree {
                point_indices: indices,
                bbox_min: bbox_min.clone(),
                bbox_max: bbox_max.clone(),
                left: None,
                right: None,
                split_dim: 0,
                split_value: 0.0,
            };
        }

        // Find the dimension with largest extent
        let mut max_extent = 0.0;
        let mut split_dim = 0;
        for d in 0..bbox_min.len() {
            let extent = bbox_max[d] - bbox_min[d];
            if extent > max_extent {
                max_extent = extent;
                split_dim = d;
            }
        }

        // Sort points by split dimension
        indices.sort_by(|&a, &b| {
            self.data[[a, split_dim]]
                .partial_cmp(&self.data[[b, split_dim]])
                .unwrap()
        });

        // Split at median
        let mid = indices.len() / 2;
        let split_value = self.data[[indices[mid], split_dim]];

        // Create left and right subsets
        let left_indices = indices[..mid].to_vec();
        let right_indices = indices[mid..].to_vec();

        // Update bounding boxes
        let mut left_bbox_max = bbox_max.clone();
        left_bbox_max[split_dim] = split_value;

        let mut right_bbox_min = bbox_min.clone();
        right_bbox_min[split_dim] = split_value;

        // Recursively build subtrees
        let left = Some(Box::new(self.build_cache_oblivious_tree(
            left_indices,
            bbox_min,
            &left_bbox_max,
        )));
        let right = Some(Box::new(self.build_cache_oblivious_tree(
            right_indices,
            &right_bbox_min,
            bbox_max,
        )));

        CacheObliviousTree {
            point_indices: indices,
            bbox_min: bbox_min.clone(),
            bbox_max: bbox_max.clone(),
            left,
            right,
            split_dim,
            split_value,
        }
    }

    /// Find k-nearest neighbors using cache-oblivious search
    pub fn kneighbors(
        &self,
        query: &ArrayView1<Float>,
    ) -> NeighborsResult<(Array1<Float>, Array1<usize>)> {
        if query.len() != self.data.ncols() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![self.data.ncols()],
                actual: vec![query.len()],
            });
        }

        let subdivision = self
            .subdivision
            .as_ref()
            .ok_or_else(|| NeighborsError::InvalidInput("Model not fitted".to_string()))?;

        let mut best_neighbors: BinaryHeap<(OrderedFloat, usize)> = BinaryHeap::new();
        self.search_cache_oblivious_tree(subdivision, query, &mut best_neighbors);

        // Extract results
        let mut neighbors: Vec<(Float, usize)> = best_neighbors
            .into_vec()
            .into_iter()
            .map(|(d, i)| (d.0, i))
            .collect();
        neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let distances: Vec<Float> = neighbors.iter().map(|(d, _)| *d).collect();
        let indices: Vec<usize> = neighbors.iter().map(|(_, i)| *i).collect();

        Ok((Array1::from(distances), Array1::from(indices)))
    }

    /// Search cache-oblivious tree recursively
    fn search_cache_oblivious_tree(
        &self,
        node: &CacheObliviousTree,
        query: &ArrayView1<Float>,
        best_neighbors: &mut BinaryHeap<(OrderedFloat, usize)>,
    ) {
        // If leaf node, check all points
        if node.left.is_none() && node.right.is_none() {
            for &idx in &node.point_indices {
                let distance = self.distance.calculate(query, &self.data.row(idx));

                if best_neighbors.len() < self.k {
                    best_neighbors.push((OrderedFloat::new(distance), idx));
                } else if let Some(&(max_distance, _)) = best_neighbors.peek() {
                    if distance < max_distance.0 {
                        best_neighbors.pop();
                        best_neighbors.push((OrderedFloat::new(distance), idx));
                    }
                }
            }
            return;
        }

        // Check which side of split the query is on
        let query_value = query[node.split_dim];
        let on_left = query_value <= node.split_value;

        // Search near side first
        if on_left {
            if let Some(ref left) = node.left {
                self.search_cache_oblivious_tree(left, query, best_neighbors);
            }
        } else if let Some(ref right) = node.right {
            self.search_cache_oblivious_tree(right, query, best_neighbors);
        }

        // Check if we need to search far side
        let worst_distance = best_neighbors
            .peek()
            .map(|(d, _)| d.0)
            .unwrap_or(Float::INFINITY);
        let distance_to_split = (query_value - node.split_value).abs();

        if best_neighbors.len() < self.k || distance_to_split < worst_distance {
            if on_left {
                if let Some(ref right) = node.right {
                    self.search_cache_oblivious_tree(right, query, best_neighbors);
                }
            } else if let Some(ref left) = node.left {
                self.search_cache_oblivious_tree(left, query, best_neighbors);
            }
        }
    }
}

/// Memory-bounded approximate neighbors using sketching
///
/// Uses random projections and sketching techniques to provide approximate
/// nearest neighbor search with bounded memory usage.
pub struct MemoryBoundedApproximateNeighbors {
    /// Number of neighbors to find
    k: usize,
    /// Number of random projections
    n_projections: usize,
    /// Memory budget in bytes
    memory_budget: usize,
    /// Random projection matrices
    projections: Vec<Array2<Float>>,
    /// Projected training data
    projected_data: Vec<Array2<Float>>,
    /// Original training data indices (for memory-bounded storage)
    stored_indices: Vec<usize>,
    /// Sampling rate for memory constraint
    sampling_rate: Float,
}

impl MemoryBoundedApproximateNeighbors {
    /// Create a new memory-bounded approximate neighbors instance
    pub fn new(k: usize, memory_budget_mb: usize) -> Self {
        Self {
            k,
            n_projections: 10,
            memory_budget: memory_budget_mb * 1024 * 1024, // Convert MB to bytes
            projections: Vec::new(),
            projected_data: Vec::new(),
            stored_indices: Vec::new(),
            sampling_rate: 1.0,
        }
    }

    /// Set number of random projections
    pub fn with_n_projections(mut self, n_projections: usize) -> Self {
        self.n_projections = n_projections;
        self
    }

    /// Fit the model with memory constraints
    pub fn fit(&mut self, X: &ArrayView2<Float>) -> NeighborsResult<()> {
        if X.is_empty() {
            return Err(NeighborsError::EmptyInput);
        }

        let n_samples = X.nrows();
        let n_features = X.ncols();

        // Estimate memory usage and determine sampling rate
        let data_size = n_samples * n_features * std::mem::size_of::<Float>();
        let projection_size = self.n_projections * n_features * 16 * std::mem::size_of::<Float>(); // 16D projections
        let total_estimated = data_size + projection_size;

        if total_estimated > self.memory_budget {
            self.sampling_rate = (self.memory_budget as Float * 0.8) / total_estimated as Float;
            self.sampling_rate = self.sampling_rate.max(0.1).min(1.0);
        }

        // Sample data points to fit memory budget
        let n_samples_to_keep = (n_samples as Float * self.sampling_rate) as usize;
        let step = if n_samples_to_keep > 0 {
            n_samples / n_samples_to_keep
        } else {
            1
        };

        self.stored_indices.clear();
        for i in (0..n_samples).step_by(step.max(1)) {
            if self.stored_indices.len() >= n_samples_to_keep {
                break;
            }
            self.stored_indices.push(i);
        }

        // Generate random projections
        let projection_dim = 16; // Fixed low dimension
        self.projections.clear();
        self.projected_data.clear();

        for _ in 0..self.n_projections {
            // Create random projection matrix
            let mut projection = Array2::zeros((projection_dim, n_features));
            let scale = 1.0 / (n_features as Float).sqrt();

            for i in 0..projection_dim {
                for j in 0..n_features {
                    // Simple random projection with ±1 values
                    projection[[i, j]] = if (i + j) % 2 == 0 { scale } else { -scale };
                }
            }

            // Project sampled data
            let mut projected = Array2::zeros((self.stored_indices.len(), projection_dim));
            for (new_idx, &orig_idx) in self.stored_indices.iter().enumerate() {
                let row = X.row(orig_idx);
                let projected_row = projection.dot(&row);
                projected.row_mut(new_idx).assign(&projected_row);
            }

            self.projections.push(projection);
            self.projected_data.push(projected);
        }

        Ok(())
    }

    /// Find approximate k-nearest neighbors
    pub fn kneighbors(
        &self,
        query: &ArrayView1<Float>,
    ) -> NeighborsResult<(Array1<Float>, Array1<usize>)> {
        if self.projections.is_empty() {
            return Err(NeighborsError::InvalidInput("Model not fitted".to_string()));
        }

        let mut candidate_scores: HashMap<usize, Float> = HashMap::new();

        // Search in each projection
        for (proj_idx, projection) in self.projections.iter().enumerate() {
            let projected_query = projection.dot(query);
            let projected_data = &self.projected_data[proj_idx];

            // Find closest points in this projection
            let mut distances: Vec<(Float, usize)> = Vec::new();
            for (data_idx, row) in projected_data.axis_iter(Axis(0)).enumerate() {
                let distance = projected_query
                    .iter()
                    .zip(row.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<Float>()
                    .sqrt();
                distances.push((distance, data_idx));
            }

            // Sort and take top candidates
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let n_candidates = std::cmp::min(self.k * 2, distances.len());

            // Accumulate scores for candidates
            for (rank, (_, data_idx)) in distances.iter().take(n_candidates).enumerate() {
                let orig_idx = self.stored_indices[*data_idx];
                let score = 1.0 / (rank as Float + 1.0); // Rank-based scoring
                *candidate_scores.entry(orig_idx).or_insert(0.0) += score;
            }
        }

        // Select top k candidates by accumulated scores
        let mut final_candidates: Vec<(Float, usize)> = candidate_scores
            .into_iter()
            .map(|(idx, score)| (score, idx))
            .collect();
        final_candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap()); // Sort by score descending

        let n_results = std::cmp::min(self.k, final_candidates.len());
        let distances: Vec<Float> = final_candidates
            .iter()
            .take(n_results)
            .map(|(score, _)| 1.0 / score)
            .collect();
        let indices: Vec<usize> = final_candidates
            .iter()
            .take(n_results)
            .map(|(_, idx)| *idx)
            .collect();

        Ok((Array1::from(distances), Array1::from(indices)))
    }

    /// Get memory usage statistics
    pub fn memory_usage(&self) -> (usize, Float) {
        let projections_size = self
            .projections
            .iter()
            .map(|p| p.len() * std::mem::size_of::<Float>())
            .sum::<usize>();
        let projected_data_size = self
            .projected_data
            .iter()
            .map(|p| p.len() * std::mem::size_of::<Float>())
            .sum::<usize>();
        let total_size = projections_size + projected_data_size;

        (total_size, self.sampling_rate)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;
    use tempfile::TempDir;

    #[test]
    #[allow(non_snake_case)]
    fn test_external_memory_knn() {
        let temp_dir = TempDir::new().unwrap();
        let mut em_knn = ExternalMemoryKNN::new(2, temp_dir.path()).with_block_size(2);

        let X = Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 10.0, 10.0, 11.0, 11.0])
            .unwrap();

        em_knn.fit(&X.view()).unwrap();

        let query = array![1.5, 1.5];
        let (distances, indices) = em_knn.kneighbors(&query.view()).unwrap();

        assert_eq!(distances.len(), 2);
        assert_eq!(indices.len(), 2);
        assert!(distances[0] <= distances[1]);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_cache_oblivious_neighbors() {
        let mut co_neighbors = CacheObliviousNeighbors::new(2);

        let X = Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 10.0, 10.0, 11.0, 11.0])
            .unwrap();

        co_neighbors.fit(&X.view()).unwrap();

        let query = array![1.5, 1.5];
        let (distances, indices) = co_neighbors.kneighbors(&query.view()).unwrap();

        assert_eq!(distances.len(), 2);
        assert_eq!(indices.len(), 2);
        assert!(distances[0] <= distances[1]);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_memory_bounded_approximate_neighbors() {
        let mut mb_neighbors = MemoryBoundedApproximateNeighbors::new(2, 1); // 1MB budget

        let X = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 1.0, 1.1, 1.1, 2.0, 2.0, 2.1, 2.1, 10.0, 10.0, 11.0, 11.0,
            ],
        )
        .unwrap();

        mb_neighbors.fit(&X.view()).unwrap();

        let query = array![1.5, 1.5];
        let (distances, indices) = mb_neighbors.kneighbors(&query.view()).unwrap();

        assert_eq!(distances.len(), 2);
        assert_eq!(indices.len(), 2);

        let (memory_usage, sampling_rate) = mb_neighbors.memory_usage();
        assert!(memory_usage > 0);
        assert!(sampling_rate > 0.0 && sampling_rate <= 1.0);
    }

    #[test]
    fn test_external_memory_error_cases() {
        let temp_dir = TempDir::new().unwrap();
        let mut em_knn = ExternalMemoryKNN::new(2, temp_dir.path());

        let empty_X = Array2::<Float>::zeros((0, 2));
        assert!(em_knn.fit(&empty_X.view()).is_err());

        // Test query without fitting
        let query = array![1.0, 2.0];
        assert!(em_knn.kneighbors(&query.view()).is_err());
    }

    #[test]
    fn test_cache_oblivious_error_cases() {
        let mut co_neighbors = CacheObliviousNeighbors::new(2);

        let empty_X = Array2::<Float>::zeros((0, 2));
        assert!(co_neighbors.fit(&empty_X.view()).is_err());

        // Test query without fitting
        let query = array![1.0, 2.0];
        assert!(co_neighbors.kneighbors(&query.view()).is_err());
    }
}
