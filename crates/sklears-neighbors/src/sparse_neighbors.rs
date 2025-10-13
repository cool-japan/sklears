//! Sparse neighbor representations for memory-efficient storage
//!
//! This module provides sparse data structures for storing neighbor relationships
//! when most neighbor distances are above a certain threshold or when only a
//! small number of neighbors per sample are relevant.

use crate::{NeighborsError, NeighborsResult};
use scirs2_core::ndarray::{Array2, Axis};
use sklears_core::types::Float;
use std::collections::{BTreeMap, HashMap};
use std::fmt;

/// Type of sparse index representation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseIndexType {
    /// HashMap-based storage (fast random access)
    HashMap,
    /// BTreeMap-based storage (ordered access)
    BTreeMap,
    /// Compressed Sparse Row (CSR) format
    CompressedSparseRow,
    /// Coordinate (COO) format
    Coordinate,
    /// Compressed Sparse Column (CSC) format
    CompressedSparseColumn,
}

/// Sparse storage formats for different use cases
#[derive(Debug, Clone)]
pub enum SparseStorage {
    HashMap(HashMap<(usize, usize), (u32, Float)>),
    BTreeMap(BTreeMap<(usize, usize), (u32, Float)>),
    Csr {
        row_ptr: Vec<usize>,
        col_indices: Vec<u32>,
        neighbor_indices: Vec<u32>,
        distances: Vec<Float>,
    },
    /// COO format: row_indices, col_indices, neighbor_indices, distances
    Coo {
        row_indices: Vec<u32>,
        col_indices: Vec<u32>,
        neighbor_indices: Vec<u32>,
        distances: Vec<Float>,
    },
    /// CSC format: col_ptr, row_indices, neighbor_indices, distances
    Csc {
        col_ptr: Vec<usize>,
        row_indices: Vec<u32>,
        neighbor_indices: Vec<u32>,
        distances: Vec<Float>,
    },
}

/// Statistics for sparse neighbor matrix
#[derive(Debug, Clone)]
pub struct SparseStats {
    pub total_elements: usize,
    pub stored_elements: usize,
    pub sparsity: Float,
    pub memory_usage: usize,
    pub avg_neighbors_per_sample: Float,
    pub max_neighbors_per_sample: usize,
    pub min_neighbors_per_sample: usize,
}

impl SparseStats {
    pub fn new(total_elements: usize, stored_elements: usize, memory_usage: usize) -> Self {
        let sparsity = if total_elements > 0 {
            1.0 - (stored_elements as Float / total_elements as Float)
        } else {
            0.0
        };

        Self {
            total_elements,
            stored_elements,
            sparsity,
            memory_usage,
            avg_neighbors_per_sample: 0.0,
            max_neighbors_per_sample: 0,
            min_neighbors_per_sample: 0,
        }
    }
}

impl fmt::Display for SparseStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SparseStats {{ sparsity: {:.2}%, stored: {}/{}, memory: {} KB, avg_neighbors: {:.1} }}",
            self.sparsity * 100.0,
            self.stored_elements,
            self.total_elements,
            self.memory_usage / 1024,
            self.avg_neighbors_per_sample
        )
    }
}

/// Sparse neighbor matrix for memory-efficient storage of neighbor relationships
pub struct SparseNeighborMatrix {
    storage: SparseStorage,
    shape: (usize, usize),
    index_type: SparseIndexType,
    threshold: Float,
    max_neighbors: Option<usize>,
    stats: SparseStats,
}

impl SparseNeighborMatrix {
    /// Create a new sparse neighbor matrix from dense neighbor data
    ///
    /// # Arguments
    /// * `neighbor_indices` - Dense neighbor indices matrix [n_samples, k_neighbors]
    /// * `neighbor_distances` - Dense neighbor distances matrix [n_samples, k_neighbors]
    /// * `index_type` - Type of sparse index to use
    /// * `threshold` - Distance threshold for sparsity (only store neighbors below this)
    /// * `max_neighbors` - Maximum neighbors to store per sample
    ///
    /// # Returns
    /// * `NeighborsResult<Self>` - Sparse neighbor matrix
    pub fn from_dense(
        neighbor_indices: &Array2<u32>,
        neighbor_distances: &Array2<Float>,
        index_type: SparseIndexType,
        threshold: Float,
        max_neighbors: Option<usize>,
    ) -> NeighborsResult<Self> {
        if neighbor_indices.shape() != neighbor_distances.shape() {
            return Err(NeighborsError::ShapeMismatch {
                expected: neighbor_indices.shape().to_vec(),
                actual: neighbor_distances.shape().to_vec(),
            });
        }

        let shape = neighbor_indices.dim();
        let total_elements = shape.0 * shape.1;

        let storage = match index_type {
            SparseIndexType::HashMap => Self::build_hashmap_storage(
                neighbor_indices,
                neighbor_distances,
                threshold,
                max_neighbors,
            )?,
            SparseIndexType::BTreeMap => Self::build_btreemap_storage(
                neighbor_indices,
                neighbor_distances,
                threshold,
                max_neighbors,
            )?,
            SparseIndexType::CompressedSparseRow => Self::build_csr_storage(
                neighbor_indices,
                neighbor_distances,
                threshold,
                max_neighbors,
            )?,
            SparseIndexType::Coordinate => Self::build_coo_storage(
                neighbor_indices,
                neighbor_distances,
                threshold,
                max_neighbors,
            )?,
            SparseIndexType::CompressedSparseColumn => Self::build_csc_storage(
                neighbor_indices,
                neighbor_distances,
                threshold,
                max_neighbors,
            )?,
        };

        let stored_elements = Self::count_stored_elements(&storage);
        let memory_usage = Self::estimate_memory_usage(&storage);
        let mut stats = SparseStats::new(total_elements, stored_elements, memory_usage);

        // Calculate neighbor statistics
        let mut neighbor_counts = vec![0; shape.0];
        match &storage {
            SparseStorage::HashMap(map) => {
                for &(row, _) in map.keys() {
                    neighbor_counts[row] += 1;
                }
            }
            SparseStorage::BTreeMap(map) => {
                for &(row, _) in map.keys() {
                    neighbor_counts[row] += 1;
                }
            }
            SparseStorage::Csr { row_ptr, .. } => {
                for i in 0..shape.0 {
                    neighbor_counts[i] = row_ptr[i + 1] - row_ptr[i];
                }
            }
            SparseStorage::Coo { row_indices, .. } => {
                for &row in row_indices {
                    neighbor_counts[row as usize] += 1;
                }
            }
            SparseStorage::Csc { row_indices, .. } => {
                for &row in row_indices {
                    neighbor_counts[row as usize] += 1;
                }
            }
        }

        if !neighbor_counts.is_empty() {
            stats.avg_neighbors_per_sample =
                neighbor_counts.iter().sum::<usize>() as Float / neighbor_counts.len() as Float;
            stats.max_neighbors_per_sample = *neighbor_counts.iter().max().unwrap_or(&0);
            stats.min_neighbors_per_sample = *neighbor_counts.iter().min().unwrap_or(&0);
        }

        Ok(Self {
            storage,
            shape,
            index_type,
            threshold,
            max_neighbors,
            stats,
        })
    }

    /// Get neighbors for a specific sample
    ///
    /// # Arguments
    /// * `sample_idx` - Index of the sample
    ///
    /// # Returns
    /// * `NeighborsResult<(Vec<u32>, Vec<Float>)>` - Neighbor indices and distances
    pub fn get_neighbors(&self, sample_idx: usize) -> NeighborsResult<(Vec<u32>, Vec<Float>)> {
        if sample_idx >= self.shape.0 {
            return Err(NeighborsError::InvalidInput(format!(
                "Sample index {} out of bounds",
                sample_idx
            )));
        }

        match &self.storage {
            SparseStorage::HashMap(map) => {
                let mut neighbors = Vec::new();
                let mut distances = Vec::new();

                for col in 0..self.shape.1 {
                    if let Some(&(neighbor_idx, distance)) = map.get(&(sample_idx, col)) {
                        neighbors.push(neighbor_idx);
                        distances.push(distance);
                    }
                }

                Ok((neighbors, distances))
            }
            SparseStorage::BTreeMap(map) => {
                let mut neighbors = Vec::new();
                let mut distances = Vec::new();

                for col in 0..self.shape.1 {
                    if let Some(&(neighbor_idx, distance)) = map.get(&(sample_idx, col)) {
                        neighbors.push(neighbor_idx);
                        distances.push(distance);
                    }
                }

                Ok((neighbors, distances))
            }
            SparseStorage::Csr {
                row_ptr,
                col_indices,
                neighbor_indices,
                distances,
            } => {
                let start = row_ptr[sample_idx];
                let end = row_ptr[sample_idx + 1];

                let neighbors = neighbor_indices[start..end].to_vec();
                let dists = distances[start..end].to_vec();

                Ok((neighbors, dists))
            }
            SparseStorage::Coo {
                row_indices,
                col_indices,
                neighbor_indices,
                distances,
            } => {
                let mut neighbors = Vec::new();
                let mut dists = Vec::new();

                for (i, &row) in row_indices.iter().enumerate() {
                    if row as usize == sample_idx {
                        neighbors.push(neighbor_indices[i]);
                        dists.push(distances[i]);
                    }
                }

                Ok((neighbors, dists))
            }
            SparseStorage::Csc {
                col_ptr,
                row_indices,
                neighbor_indices,
                distances,
            } => {
                let mut neighbors = Vec::new();
                let mut dists = Vec::new();

                for (i, &row) in row_indices.iter().enumerate() {
                    if row as usize == sample_idx {
                        neighbors.push(neighbor_indices[i]);
                        dists.push(distances[i]);
                    }
                }

                Ok((neighbors, dists))
            }
        }
    }

    /// Get k nearest neighbors for a sample
    ///
    /// # Arguments
    /// * `sample_idx` - Index of the sample
    /// * `k` - Number of nearest neighbors to return
    ///
    /// # Returns
    /// * `NeighborsResult<(Vec<u32>, Vec<Float>)>` - K nearest neighbor indices and distances
    pub fn get_k_neighbors(
        &self,
        sample_idx: usize,
        k: usize,
    ) -> NeighborsResult<(Vec<u32>, Vec<Float>)> {
        let (mut neighbors, mut distances) = self.get_neighbors(sample_idx)?;

        // Sort by distance and take k nearest
        let mut pairs: Vec<(u32, Float)> = neighbors.into_iter().zip(distances).collect();
        pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs.truncate(k);

        neighbors = pairs.iter().map(|(idx, _)| *idx).collect();
        distances = pairs.iter().map(|(_, dist)| *dist).collect();

        Ok((neighbors, distances))
    }

    /// Check if a neighbor relationship exists
    ///
    /// # Arguments
    /// * `sample_idx` - Index of the sample
    /// * `neighbor_pos` - Position in the neighbor list
    ///
    /// # Returns
    /// * `bool` - True if the neighbor exists
    pub fn has_neighbor(&self, sample_idx: usize, neighbor_pos: usize) -> bool {
        match &self.storage {
            SparseStorage::HashMap(map) => map.contains_key(&(sample_idx, neighbor_pos)),
            SparseStorage::BTreeMap(map) => map.contains_key(&(sample_idx, neighbor_pos)),
            SparseStorage::Csr { row_ptr, .. } => {
                let start = row_ptr[sample_idx];
                let end = row_ptr[sample_idx + 1];
                neighbor_pos < (end - start)
            }
            SparseStorage::Coo {
                row_indices,
                col_indices,
                ..
            } => row_indices
                .iter()
                .zip(col_indices.iter())
                .any(|(&r, &c)| r as usize == sample_idx && c as usize == neighbor_pos),
            SparseStorage::Csc { row_indices, .. } => {
                row_indices.iter().any(|&r| r as usize == sample_idx)
            }
        }
    }

    /// Convert to dense neighbor matrices
    ///
    /// # Returns
    /// * `NeighborsResult<(Array2<u32>, Array2<Float>)>` - Dense neighbor indices and distances
    pub fn to_dense(&self) -> NeighborsResult<(Array2<u32>, Array2<Float>)> {
        let mut neighbor_indices = Array2::zeros(self.shape);
        let mut neighbor_distances = Array2::from_elem(self.shape, Float::INFINITY);

        match &self.storage {
            SparseStorage::HashMap(map) => {
                for (&(row, col), &(neighbor_idx, distance)) in map {
                    neighbor_indices[[row, col]] = neighbor_idx;
                    neighbor_distances[[row, col]] = distance;
                }
            }
            SparseStorage::BTreeMap(map) => {
                for (&(row, col), &(neighbor_idx, distance)) in map {
                    neighbor_indices[[row, col]] = neighbor_idx;
                    neighbor_distances[[row, col]] = distance;
                }
            }
            SparseStorage::Csr {
                row_ptr,
                col_indices,
                neighbor_indices: nei_idx,
                distances,
            } => {
                for row in 0..self.shape.0 {
                    let start = row_ptr[row];
                    let end = row_ptr[row + 1];

                    for (i, col_idx) in col_indices[start..end].iter().enumerate() {
                        let col = *col_idx as usize;
                        if col < self.shape.1 {
                            neighbor_indices[[row, col]] = nei_idx[start + i];
                            neighbor_distances[[row, col]] = distances[start + i];
                        }
                    }
                }
            }
            SparseStorage::Coo {
                row_indices,
                col_indices,
                neighbor_indices: nei_idx,
                distances,
            } => {
                for (i, (&row, &col)) in row_indices.iter().zip(col_indices.iter()).enumerate() {
                    let r = row as usize;
                    let c = col as usize;
                    if r < self.shape.0 && c < self.shape.1 {
                        neighbor_indices[[r, c]] = nei_idx[i];
                        neighbor_distances[[r, c]] = distances[i];
                    }
                }
            }
            SparseStorage::Csc {
                col_ptr,
                row_indices,
                neighbor_indices: nei_idx,
                distances,
            } => {
                for col in 0..self.shape.1 {
                    let start = col_ptr[col];
                    let end = col_ptr[col + 1];

                    for (i, row_idx) in row_indices[start..end].iter().enumerate() {
                        let row = *row_idx as usize;
                        if row < self.shape.0 {
                            neighbor_indices[[row, col]] = nei_idx[start + i];
                            neighbor_distances[[row, col]] = distances[start + i];
                        }
                    }
                }
            }
        }

        Ok((neighbor_indices, neighbor_distances))
    }

    /// Get statistics about the sparse matrix
    pub fn stats(&self) -> &SparseStats {
        &self.stats
    }

    /// Get the shape of the matrix
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Get the index type used
    pub fn index_type(&self) -> SparseIndexType {
        self.index_type
    }

    /// Get the sparsity threshold
    pub fn threshold(&self) -> Float {
        self.threshold
    }

    // Storage builders

    fn build_hashmap_storage(
        neighbor_indices: &Array2<u32>,
        neighbor_distances: &Array2<Float>,
        threshold: Float,
        max_neighbors: Option<usize>,
    ) -> NeighborsResult<SparseStorage> {
        let mut map = HashMap::new();

        for (i, (idx_row, dist_row)) in neighbor_indices
            .axis_iter(Axis(0))
            .zip(neighbor_distances.axis_iter(Axis(0)))
            .enumerate()
        {
            let mut neighbors: Vec<(usize, u32, Float)> = idx_row
                .iter()
                .zip(dist_row.iter())
                .enumerate()
                .filter(|(_, (_, &dist))| dist <= threshold)
                .map(|(j, (&idx, &dist))| (j, idx, dist))
                .collect();

            // Sort by distance and apply max_neighbors limit
            neighbors.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

            if let Some(max_k) = max_neighbors {
                neighbors.truncate(max_k);
            }

            for (j, neighbor_idx, distance) in neighbors {
                map.insert((i, j), (neighbor_idx, distance));
            }
        }

        Ok(SparseStorage::HashMap(map))
    }

    fn build_btreemap_storage(
        neighbor_indices: &Array2<u32>,
        neighbor_distances: &Array2<Float>,
        threshold: Float,
        max_neighbors: Option<usize>,
    ) -> NeighborsResult<SparseStorage> {
        let mut map = BTreeMap::new();

        for (i, (idx_row, dist_row)) in neighbor_indices
            .axis_iter(Axis(0))
            .zip(neighbor_distances.axis_iter(Axis(0)))
            .enumerate()
        {
            let mut neighbors: Vec<(usize, u32, Float)> = idx_row
                .iter()
                .zip(dist_row.iter())
                .enumerate()
                .filter(|(_, (_, &dist))| dist <= threshold)
                .map(|(j, (&idx, &dist))| (j, idx, dist))
                .collect();

            neighbors.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

            if let Some(max_k) = max_neighbors {
                neighbors.truncate(max_k);
            }

            for (j, neighbor_idx, distance) in neighbors {
                map.insert((i, j), (neighbor_idx, distance));
            }
        }

        Ok(SparseStorage::BTreeMap(map))
    }

    fn build_csr_storage(
        neighbor_indices: &Array2<u32>,
        neighbor_distances: &Array2<Float>,
        threshold: Float,
        max_neighbors: Option<usize>,
    ) -> NeighborsResult<SparseStorage> {
        let mut row_ptr = vec![0];
        let mut col_indices = Vec::new();
        let mut neighbor_idx_vec = Vec::new();
        let mut distances = Vec::new();

        for (idx_row, dist_row) in neighbor_indices
            .axis_iter(Axis(0))
            .zip(neighbor_distances.axis_iter(Axis(0)))
        {
            let mut neighbors: Vec<(usize, u32, Float)> = idx_row
                .iter()
                .zip(dist_row.iter())
                .enumerate()
                .filter(|(_, (_, &dist))| dist <= threshold)
                .map(|(j, (&idx, &dist))| (j, idx, dist))
                .collect();

            neighbors.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

            if let Some(max_k) = max_neighbors {
                neighbors.truncate(max_k);
            }

            for (j, neighbor_idx, distance) in neighbors {
                col_indices.push(j as u32);
                neighbor_idx_vec.push(neighbor_idx);
                distances.push(distance);
            }

            row_ptr.push(col_indices.len());
        }

        Ok(SparseStorage::Csr {
            row_ptr,
            col_indices,
            neighbor_indices: neighbor_idx_vec,
            distances,
        })
    }

    fn build_coo_storage(
        neighbor_indices: &Array2<u32>,
        neighbor_distances: &Array2<Float>,
        threshold: Float,
        max_neighbors: Option<usize>,
    ) -> NeighborsResult<SparseStorage> {
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut neighbor_idx_vec = Vec::new();
        let mut distances = Vec::new();

        for (i, (idx_row, dist_row)) in neighbor_indices
            .axis_iter(Axis(0))
            .zip(neighbor_distances.axis_iter(Axis(0)))
            .enumerate()
        {
            let mut neighbors: Vec<(usize, u32, Float)> = idx_row
                .iter()
                .zip(dist_row.iter())
                .enumerate()
                .filter(|(_, (_, &dist))| dist <= threshold)
                .map(|(j, (&idx, &dist))| (j, idx, dist))
                .collect();

            neighbors.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

            if let Some(max_k) = max_neighbors {
                neighbors.truncate(max_k);
            }

            for (j, neighbor_idx, distance) in neighbors {
                row_indices.push(i as u32);
                col_indices.push(j as u32);
                neighbor_idx_vec.push(neighbor_idx);
                distances.push(distance);
            }
        }

        Ok(SparseStorage::Coo {
            row_indices,
            col_indices,
            neighbor_indices: neighbor_idx_vec,
            distances,
        })
    }

    fn build_csc_storage(
        neighbor_indices: &Array2<u32>,
        neighbor_distances: &Array2<Float>,
        threshold: Float,
        max_neighbors: Option<usize>,
    ) -> NeighborsResult<SparseStorage> {
        // First collect all entries
        let mut entries: Vec<(usize, usize, u32, Float)> = Vec::new();

        for (i, (idx_row, dist_row)) in neighbor_indices
            .axis_iter(Axis(0))
            .zip(neighbor_distances.axis_iter(Axis(0)))
            .enumerate()
        {
            let mut neighbors: Vec<(usize, u32, Float)> = idx_row
                .iter()
                .zip(dist_row.iter())
                .enumerate()
                .filter(|(_, (_, &dist))| dist <= threshold)
                .map(|(j, (&idx, &dist))| (j, idx, dist))
                .collect();

            neighbors.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

            if let Some(max_k) = max_neighbors {
                neighbors.truncate(max_k);
            }

            for (j, neighbor_idx, distance) in neighbors {
                entries.push((i, j, neighbor_idx, distance));
            }
        }

        // Sort by column for CSC format
        entries.sort_by_key(|&(_, j, _, _)| j);

        let mut col_ptr = vec![0];
        let mut row_indices = Vec::new();
        let mut neighbor_idx_vec = Vec::new();
        let mut distances = Vec::new();

        let mut current_col = 0;
        for (i, j, neighbor_idx, distance) in entries {
            // Add empty columns if needed
            while current_col < j {
                col_ptr.push(row_indices.len());
                current_col += 1;
            }

            row_indices.push(i as u32);
            neighbor_idx_vec.push(neighbor_idx);
            distances.push(distance);

            if current_col == j {
                current_col = j + 1;
            }
        }

        // Fill remaining columns
        let n_cols = neighbor_indices.shape()[1];
        while col_ptr.len() <= n_cols {
            col_ptr.push(row_indices.len());
        }

        Ok(SparseStorage::Csc {
            col_ptr,
            row_indices,
            neighbor_indices: neighbor_idx_vec,
            distances,
        })
    }

    fn count_stored_elements(storage: &SparseStorage) -> usize {
        match storage {
            SparseStorage::HashMap(map) => map.len(),
            SparseStorage::BTreeMap(map) => map.len(),
            SparseStorage::Csr {
                neighbor_indices, ..
            } => neighbor_indices.len(),
            SparseStorage::Coo {
                neighbor_indices, ..
            } => neighbor_indices.len(),
            SparseStorage::Csc {
                neighbor_indices, ..
            } => neighbor_indices.len(),
        }
    }

    fn estimate_memory_usage(storage: &SparseStorage) -> usize {
        match storage {
            SparseStorage::HashMap(map) => {
                map.len()
                    * (std::mem::size_of::<(usize, usize)>() + std::mem::size_of::<(u32, Float)>())
            }
            SparseStorage::BTreeMap(map) => {
                map.len()
                    * (std::mem::size_of::<(usize, usize)>() + std::mem::size_of::<(u32, Float)>())
            }
            SparseStorage::Csr {
                row_ptr,
                col_indices,
                neighbor_indices,
                distances,
            } => {
                row_ptr.len() * std::mem::size_of::<usize>()
                    + col_indices.len() * std::mem::size_of::<u32>()
                    + neighbor_indices.len() * std::mem::size_of::<u32>()
                    + distances.len() * std::mem::size_of::<Float>()
            }
            SparseStorage::Coo {
                row_indices,
                col_indices,
                neighbor_indices,
                distances,
            } => {
                row_indices.len() * std::mem::size_of::<u32>()
                    + col_indices.len() * std::mem::size_of::<u32>()
                    + neighbor_indices.len() * std::mem::size_of::<u32>()
                    + distances.len() * std::mem::size_of::<Float>()
            }
            SparseStorage::Csc {
                col_ptr,
                row_indices,
                neighbor_indices,
                distances,
            } => {
                col_ptr.len() * std::mem::size_of::<usize>()
                    + row_indices.len() * std::mem::size_of::<u32>()
                    + neighbor_indices.len() * std::mem::size_of::<u32>()
                    + distances.len() * std::mem::size_of::<Float>()
            }
        }
    }
}

/// Builder for creating sparse neighbor matrices with various configurations
pub struct SparseNeighborBuilder {
    index_type: SparseIndexType,
    threshold: Option<Float>,
    max_neighbors: Option<usize>,
}

impl Default for SparseNeighborBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseNeighborBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            index_type: SparseIndexType::CompressedSparseRow,
            threshold: None,
            max_neighbors: None,
        }
    }

    /// Set the index type
    pub fn index_type(mut self, index_type: SparseIndexType) -> Self {
        self.index_type = index_type;
        self
    }

    /// Set the distance threshold for sparsity
    pub fn threshold(mut self, threshold: Float) -> Self {
        self.threshold = Some(threshold);
        self
    }

    /// Set the maximum number of neighbors per sample
    pub fn max_neighbors(mut self, max_neighbors: usize) -> Self {
        self.max_neighbors = Some(max_neighbors);
        self
    }

    /// Build the sparse neighbor matrix
    pub fn build(
        self,
        neighbor_indices: &Array2<u32>,
        neighbor_distances: &Array2<Float>,
    ) -> NeighborsResult<SparseNeighborMatrix> {
        let threshold = self.threshold.unwrap_or(Float::INFINITY);

        SparseNeighborMatrix::from_dense(
            neighbor_indices,
            neighbor_distances,
            self.index_type,
            threshold,
            self.max_neighbors,
        )
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::arr2;

    #[test]
    fn test_sparse_neighbor_matrix_hashmap() {
        let neighbor_indices = arr2(&[[1, 2, 3], [0, 2, 3], [0, 1, 3]]);
        let neighbor_distances = arr2(&[[0.5, 1.5, 2.5], [0.3, 1.2, 2.8], [0.8, 1.1, 2.2]]);

        let sparse = SparseNeighborMatrix::from_dense(
            &neighbor_indices,
            &neighbor_distances,
            SparseIndexType::HashMap,
            2.0, // threshold
            None,
        )
        .unwrap();

        let (neighbors, distances) = sparse.get_neighbors(0).unwrap();
        assert_eq!(neighbors.len(), 2); // Only 2 neighbors under threshold
        assert!(distances[0] <= 2.0);
        assert!(distances[1] <= 2.0);

        let stats = sparse.stats();
        assert!(stats.sparsity > 0.0);
        assert!(stats.stored_elements < 9); // Less than full 3x3 matrix
    }

    #[test]
    fn test_sparse_neighbor_matrix_csr() {
        let neighbor_indices = arr2(&[[1, 2, 3, 4], [0, 2, 3, 4]]);
        let neighbor_distances = arr2(&[[0.5, 1.5, 10.0, 20.0], [0.3, 1.2, 15.0, 25.0]]);

        let sparse = SparseNeighborMatrix::from_dense(
            &neighbor_indices,
            &neighbor_distances,
            SparseIndexType::CompressedSparseRow,
            2.0,     // threshold
            Some(3), // max neighbors
        )
        .unwrap();

        let (neighbors, distances) = sparse.get_neighbors(0).unwrap();
        assert!(neighbors.len() <= 3);
        assert!(neighbors.len() >= 2); // At least 2 neighbors under threshold

        // Test k-neighbors
        let (k_neighbors, k_distances) = sparse.get_k_neighbors(0, 2).unwrap();
        assert_eq!(k_neighbors.len(), 2);
        assert_eq!(k_distances.len(), 2);

        // Should be sorted by distance
        if k_distances.len() > 1 {
            assert!(k_distances[0] <= k_distances[1]);
        }
    }

    #[test]
    fn test_sparse_neighbor_matrix_to_dense() {
        let neighbor_indices = arr2(&[[1, 2], [0, 2]]);
        let neighbor_distances = arr2(&[[0.5, 1.5], [0.3, 1.2]]);

        let sparse = SparseNeighborMatrix::from_dense(
            &neighbor_indices,
            &neighbor_distances,
            SparseIndexType::CompressedSparseRow,
            2.0,
            None,
        )
        .unwrap();

        let (dense_indices, dense_distances) = sparse.to_dense().unwrap();

        // Check that original data is preserved for values under threshold
        assert_eq!(dense_indices[[0, 0]], 1);
        assert_eq!(dense_indices[[0, 1]], 2);
        assert_abs_diff_eq!(dense_distances[[0, 0]], 0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(dense_distances[[0, 1]], 1.5, epsilon = 1e-6);
    }

    #[test]
    fn test_sparse_neighbor_builder() {
        let neighbor_indices = arr2(&[[1, 2, 3], [0, 2, 3]]);
        let neighbor_distances = arr2(&[[0.5, 1.5, 10.0], [0.3, 1.2, 15.0]]);

        let sparse = SparseNeighborBuilder::new()
            .index_type(SparseIndexType::BTreeMap)
            .threshold(2.0)
            .max_neighbors(2)
            .build(&neighbor_indices, &neighbor_distances)
            .unwrap();

        assert_eq!(sparse.index_type(), SparseIndexType::BTreeMap);
        assert_eq!(sparse.threshold(), 2.0);

        let (neighbors, _) = sparse.get_neighbors(0).unwrap();
        assert!(neighbors.len() <= 2);
    }

    #[test]
    fn test_sparse_statistics() {
        let neighbor_indices = arr2(&[[1, 2, 3, 4], [0, 2, 3, 4], [0, 1, 3, 4]]);
        let neighbor_distances = arr2(&[
            [0.1, 0.2, 10.0, 20.0],
            [0.1, 0.2, 15.0, 25.0],
            [0.1, 0.2, 12.0, 22.0],
        ]);

        let sparse = SparseNeighborMatrix::from_dense(
            &neighbor_indices,
            &neighbor_distances,
            SparseIndexType::CompressedSparseRow,
            1.0, // Only keep very small distances
            None,
        )
        .unwrap();

        let stats = sparse.stats();
        assert!(stats.sparsity > 0.0); // Should be somewhat sparse
        assert!(stats.avg_neighbors_per_sample > 0.0);
        assert!(stats.max_neighbors_per_sample >= stats.min_neighbors_per_sample);

        // Test stats display
        let stats_str = format!("{}", stats);
        assert!(stats_str.contains("sparsity"));
        assert!(stats_str.contains("stored"));
    }

    #[test]
    fn test_different_index_types() {
        let neighbor_indices = arr2(&[[1, 2], [0, 2], [0, 1]]);
        let neighbor_distances = arr2(&[[0.5, 1.5], [0.3, 1.2], [0.8, 1.1]]);

        let index_types = vec![
            SparseIndexType::HashMap,
            SparseIndexType::BTreeMap,
            SparseIndexType::CompressedSparseRow,
            SparseIndexType::Coordinate,
            SparseIndexType::CompressedSparseColumn,
        ];

        for index_type in index_types {
            let sparse = SparseNeighborMatrix::from_dense(
                &neighbor_indices,
                &neighbor_distances,
                index_type,
                2.0,
                None,
            )
            .unwrap();

            // All should work and preserve data
            let (neighbors, distances) = sparse.get_neighbors(0).unwrap();
            assert!(!neighbors.is_empty());
            assert_eq!(neighbors.len(), distances.len());

            // Should be able to convert back to dense
            let (dense_indices, dense_distances) = sparse.to_dense().unwrap();
            assert_eq!(dense_indices.shape(), [3, 2]);
            assert_eq!(dense_distances.shape(), [3, 2]);
        }
    }
}
