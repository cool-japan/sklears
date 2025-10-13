//! Sparse matrix representations for large-scale clustering
//!
//! This module provides sparse matrix data structures and algorithms optimized
//! for clustering applications where many distances are zero or above a threshold.
//! This is particularly useful for high-dimensional data or when using distance
//! thresholds to create neighborhood graphs.

use std::collections::{HashMap, VecDeque};

use sklears_core::{
    error::{Result, SklearsError},
    types::{Array2, Float},
};

use crate::simd_distances::{simd_distance, SimdDistanceMetric};

/// Sparse matrix entry with row, column, and value
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SparseEntry {
    pub row: usize,
    pub col: usize,
    pub value: Float,
}

/// Compressed Sparse Row (CSR) matrix for efficient sparse distance storage
#[derive(Debug, Clone)]
pub struct SparseDistanceMatrix {
    /// Values of non-zero entries
    values: Vec<Float>,
    /// Column indices for each value
    col_indices: Vec<usize>,
    /// Row pointers (cumulative count of entries per row)
    row_pointers: Vec<usize>,
    /// Matrix dimensions
    n_rows: usize,
    n_cols: usize,
    /// Number of non-zero entries
    nnz: usize,
    /// Whether the matrix is symmetric
    symmetric: bool,
}

/// Configuration for sparse matrix creation
#[derive(Debug, Clone)]
pub struct SparseMatrixConfig {
    /// Distance threshold - distances above this are not stored
    pub distance_threshold: Float,
    /// Distance metric to use
    pub metric: SimdDistanceMetric,
    /// Initial capacity hint for sparse entries
    pub initial_capacity: Option<usize>,
    /// Whether to enforce symmetry
    pub symmetric: bool,
    /// Sparsity threshold (fraction of entries that must be zero to use sparse)
    pub sparsity_threshold: Float,
}

impl Default for SparseMatrixConfig {
    fn default() -> Self {
        Self {
            distance_threshold: Float::INFINITY,
            metric: SimdDistanceMetric::Euclidean,
            initial_capacity: None,
            symmetric: true,
            sparsity_threshold: 0.5, // 50% sparse to justify sparse storage
        }
    }
}

impl SparseDistanceMatrix {
    /// Create a new empty sparse distance matrix
    ///
    /// # Arguments
    /// * `n_rows` - Number of rows
    /// * `n_cols` - Number of columns
    /// * `symmetric` - Whether the matrix should be symmetric
    pub fn new(n_rows: usize, n_cols: usize, symmetric: bool) -> Self {
        let row_pointers = vec![0; n_rows + 1];

        Self {
            values: Vec::new(),
            col_indices: Vec::new(),
            row_pointers,
            n_rows,
            n_cols,
            nnz: 0,
            symmetric,
        }
    }

    /// Create sparse distance matrix from dense data
    ///
    /// # Arguments
    /// * `data` - Input data matrix (n_samples × n_features)
    /// * `config` - Configuration for sparse matrix creation
    pub fn from_data(data: &Array2<Float>, config: SparseMatrixConfig) -> Result<Self> {
        let n_samples = data.nrows();

        // First pass: count non-zero entries to estimate sparsity
        let entry_count = 0;
        let total_entries = if config.symmetric {
            (n_samples * (n_samples - 1)) / 2
        } else {
            n_samples * n_samples
        };

        // Sample a subset to estimate sparsity
        let sample_size = (n_samples / 10).max(10).min(100);
        let mut sampled_entries = 0;
        let mut sampled_nonzero = 0;

        let step_size = (n_samples / sample_size).max(1);
        for i in (0..n_samples).step_by(step_size) {
            let j_start = if config.symmetric { i + 1 } else { 0 };
            for j in (j_start..n_samples).step_by(step_size) {
                if i != j {
                    let row_i = data.row(i);
                    let row_j = data.row(j);
                    let distance = simd_distance(&row_i, &row_j, config.metric).map_err(|e| {
                        SklearsError::NumericalError(format!(
                            "SIMD distance computation failed: {}",
                            e
                        ))
                    })?;

                    sampled_entries += 1;
                    if distance <= config.distance_threshold && distance > 0.0 {
                        sampled_nonzero += 1;
                    }
                }
            }
        }

        // Estimate sparsity
        let estimated_sparsity = if sampled_entries > 0 {
            1.0 - (sampled_nonzero as Float / sampled_entries as Float)
        } else {
            0.0
        };

        // Check if sparse representation is beneficial
        if estimated_sparsity < config.sparsity_threshold {
            return Err(SklearsError::InvalidInput(format!(
                "Data is not sparse enough ({:.2}% sparse) for sparse representation. Consider using dense matrix.",
                estimated_sparsity * 100.0
            )));
        }

        // Create sparse matrix
        let mut sparse_matrix = Self::new(n_samples, n_samples, config.symmetric);

        // Pre-allocate based on estimated non-zero count
        let estimated_nnz = (total_entries as Float * (1.0 - estimated_sparsity)) as usize;
        let capacity = config.initial_capacity.unwrap_or(estimated_nnz);
        sparse_matrix.values.reserve(capacity);
        sparse_matrix.col_indices.reserve(capacity);

        // Build sparse matrix row by row
        let mut entries_buffer = Vec::new();

        for i in 0..n_samples {
            let j_start = if config.symmetric { i + 1 } else { 0 };

            entries_buffer.clear();

            for j in j_start..n_samples {
                if i != j {
                    let row_i = data.row(i);
                    let row_j = data.row(j);
                    let distance = simd_distance(&row_i, &row_j, config.metric).map_err(|e| {
                        SklearsError::NumericalError(format!(
                            "SIMD distance computation failed: {}",
                            e
                        ))
                    })?;

                    if distance <= config.distance_threshold && distance > 0.0 {
                        entries_buffer.push((j, distance));
                    }
                }
            }

            // Sort entries by column index for CSR format
            entries_buffer.sort_by_key(|&(col, _)| col);

            // Add entries to sparse matrix
            for (col, value) in entries_buffer.iter() {
                sparse_matrix.values.push(*value);
                sparse_matrix.col_indices.push(*col);
                sparse_matrix.nnz += 1;
            }

            sparse_matrix.row_pointers[i + 1] = sparse_matrix.nnz;

            // Log progress for large datasets
            if (i + 1) % 1000 == 0 || i == n_samples - 1 {
                let progress = (i + 1) as f64 / n_samples as f64 * 100.0;
                eprintln!("Building sparse matrix: {:.1}% complete", progress);
            }
        }

        eprintln!(
            "Created sparse matrix: {} non-zero entries out of {} total ({:.2}% sparse)",
            sparse_matrix.nnz,
            total_entries,
            (1.0 - sparse_matrix.nnz as f64 / total_entries as f64) * 100.0
        );

        Ok(sparse_matrix)
    }

    /// Get the value at position (row, col)
    pub fn get(&self, row: usize, col: usize) -> Float {
        if row >= self.n_rows || col >= self.n_cols {
            return 0.0;
        }

        if row == col {
            return 0.0; // Diagonal is always zero for distance matrices
        }

        // Handle symmetry
        let (search_row, search_col) = if self.symmetric && row > col {
            (col, row)
        } else {
            (row, col)
        };

        // If symmetric and searching in lower triangle, return 0
        if self.symmetric && search_row > search_col {
            return 0.0;
        }

        // Binary search in the row
        let start = self.row_pointers[search_row];
        let end = self.row_pointers[search_row + 1];

        match self.col_indices[start..end].binary_search(&search_col) {
            Ok(idx) => self.values[start + idx],
            Err(_) => 0.0,
        }
    }

    /// Set the value at position (row, col)
    ///
    /// Note: This is an expensive operation for CSR matrices as it may require
    /// rebuilding the entire structure. Use during construction only.
    pub fn set(&mut self, row: usize, col: usize, value: Float) -> Result<()> {
        if row >= self.n_rows || col >= self.n_cols {
            return Err(SklearsError::InvalidInput(format!(
                "Index ({}, {}) out of bounds for matrix {}×{}",
                row, col, self.n_rows, self.n_cols
            )));
        }

        if row == col && value != 0.0 {
            return Err(SklearsError::InvalidInput(
                "Cannot set non-zero diagonal element in distance matrix".to_string(),
            ));
        }

        // For now, we'll rebuild the matrix with the new value
        // This is inefficient but correct
        self.set_and_rebuild(row, col, value)
    }

    /// Set value and rebuild matrix structure (expensive operation)
    fn set_and_rebuild(&mut self, row: usize, col: usize, value: Float) -> Result<()> {
        // Convert to COO format, modify, and convert back
        let mut entries = Vec::new();

        // Extract existing entries
        for r in 0..self.n_rows {
            let start = self.row_pointers[r];
            let end = self.row_pointers[r + 1];

            for idx in start..end {
                let c = self.col_indices[idx];
                let v = self.values[idx];

                if !(r == row && c == col) {
                    entries.push((r, c, v));
                }
            }
        }

        // Add new entry if non-zero
        if value != 0.0 {
            entries.push((row, col, value));
        }

        // Handle symmetry
        if self.symmetric && value != 0.0 && row != col {
            entries.push((col, row, value));
        }

        // Sort entries by (row, col)
        entries.sort_by_key(|&(r, c, _)| (r, c));

        // Rebuild CSR structure
        self.values.clear();
        self.col_indices.clear();
        self.row_pointers.fill(0);
        self.nnz = 0;

        let mut current_row = 0;
        for (r, c, v) in entries {
            // Update row pointers
            while current_row <= r {
                self.row_pointers[current_row] = self.nnz;
                current_row += 1;
            }

            self.values.push(v);
            self.col_indices.push(c);
            self.nnz += 1;
        }

        // Fill remaining row pointers
        while current_row <= self.n_rows {
            self.row_pointers[current_row] = self.nnz;
            current_row += 1;
        }

        Ok(())
    }

    /// Get all non-zero entries in a row
    pub fn row_entries(&self, row: usize) -> Vec<(usize, Float)> {
        if row >= self.n_rows {
            return Vec::new();
        }

        let start = self.row_pointers[row];
        let end = self.row_pointers[row + 1];

        let mut entries = Vec::new();
        for idx in start..end {
            entries.push((self.col_indices[idx], self.values[idx]));
        }

        entries
    }

    /// Get k-nearest neighbors for a specific row
    ///
    /// # Arguments
    /// * `row` - Row index to find neighbors for
    /// * `k` - Number of nearest neighbors to find
    ///
    /// # Returns
    /// Vector of (column_index, distance) pairs sorted by distance
    pub fn k_nearest_neighbors(&self, row: usize, k: usize) -> Vec<(usize, Float)> {
        if row >= self.n_rows {
            return Vec::new();
        }

        let mut neighbors = self.row_entries(row);

        // For symmetric matrices, also check the column (avoiding duplicates)
        if self.symmetric {
            for other_row in 0..self.n_rows {
                if other_row != row {
                    let value = self.get(other_row, row);
                    if value > 0.0 && !neighbors.iter().any(|(idx, _)| *idx == other_row) {
                        neighbors.push((other_row, value));
                    }
                }
            }
        }

        // Sort by distance and take k nearest
        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        neighbors.truncate(k);

        neighbors
    }

    /// Get all neighbors within a specific radius
    pub fn neighbors_within_radius(&self, row: usize, radius: Float) -> Vec<(usize, Float)> {
        if row >= self.n_rows {
            return Vec::new();
        }

        let mut neighbors = Vec::new();

        // Get entries from the row
        for (col, distance) in self.row_entries(row) {
            if distance <= radius {
                neighbors.push((col, distance));
            }
        }

        // For symmetric matrices, also check the column (avoiding duplicates)
        if self.symmetric {
            for other_row in 0..self.n_rows {
                if other_row != row {
                    let distance = self.get(other_row, row);
                    if distance > 0.0
                        && distance <= radius
                        && !neighbors.iter().any(|(idx, _)| *idx == other_row)
                    {
                        neighbors.push((other_row, distance));
                    }
                }
            }
        }

        // Sort by distance
        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        neighbors
    }

    /// Convert to dense matrix (for small matrices or debugging)
    pub fn to_dense(&self) -> Array2<Float> {
        let mut dense = Array2::zeros((self.n_rows, self.n_cols));

        for row in 0..self.n_rows {
            let start = self.row_pointers[row];
            let end = self.row_pointers[row + 1];

            for idx in start..end {
                let col = self.col_indices[idx];
                let value = self.values[idx];
                dense[[row, col]] = value;

                // Handle symmetry
                if self.symmetric && row != col {
                    dense[[col, row]] = value;
                }
            }
        }

        dense
    }

    /// Get matrix statistics
    pub fn stats(&self) -> SparseMatrixStats {
        let total_entries = self.n_rows * self.n_cols;
        let sparsity = 1.0 - (self.nnz as f64 / total_entries as f64);

        let dense_memory = total_entries * std::mem::size_of::<Float>();
        let sparse_memory = self.values.len() * std::mem::size_of::<Float>()
            + self.col_indices.len() * std::mem::size_of::<usize>()
            + self.row_pointers.len() * std::mem::size_of::<usize>();

        let memory_savings = 1.0 - (sparse_memory as f64 / dense_memory as f64);

        SparseMatrixStats {
            n_rows: self.n_rows,
            n_cols: self.n_cols,
            nnz: self.nnz,
            total_entries,
            sparsity,
            dense_memory_bytes: dense_memory,
            sparse_memory_bytes: sparse_memory,
            memory_savings,
            symmetric: self.symmetric,
        }
    }

    /// Matrix dimensions
    pub fn shape(&self) -> (usize, usize) {
        (self.n_rows, self.n_cols)
    }

    /// Number of non-zero entries
    pub fn nnz(&self) -> usize {
        self.nnz
    }

    /// Check if matrix is symmetric
    pub fn is_symmetric(&self) -> bool {
        self.symmetric
    }

    /// Advanced: Get connected components using union-find algorithm
    /// Returns vector where each element is the component ID for that vertex
    pub fn connected_components(&self) -> Vec<usize> {
        let mut parent: Vec<usize> = (0..self.n_rows).collect();
        let mut rank = vec![0; self.n_rows];

        // Union-Find find with path compression
        fn find(parent: &mut [usize], x: usize) -> usize {
            if parent[x] != x {
                parent[x] = find(parent, parent[x]);
            }
            parent[x]
        }

        // Union-Find union by rank
        fn union(parent: &mut [usize], rank: &mut [usize], x: usize, y: usize) {
            let root_x = find(parent, x);
            let root_y = find(parent, y);

            if root_x != root_y {
                if rank[root_x] < rank[root_y] {
                    parent[root_x] = root_y;
                } else if rank[root_x] > rank[root_y] {
                    parent[root_y] = root_x;
                } else {
                    parent[root_y] = root_x;
                    rank[root_x] += 1;
                }
            }
        }

        // Process all edges
        for i in 0..self.n_rows {
            let start = self.row_pointers[i];
            let end = self.row_pointers[i + 1];

            for idx in start..end {
                let j = self.col_indices[idx];
                if i < j {
                    // Process each edge only once
                    union(&mut parent, &mut rank, i, j);
                }
            }
        }

        // Assign component IDs
        let mut component_id = HashMap::new();
        let mut next_id = 0;
        let mut result = vec![0; self.n_rows];

        for i in 0..self.n_rows {
            let root = find(&mut parent, i);
            let id = *component_id.entry(root).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            result[i] = id;
        }

        result
    }

    /// Advanced: Compute graph diameter (longest shortest path)
    /// Uses BFS from multiple starting points for efficiency
    pub fn graph_diameter(&self) -> Option<usize> {
        if self.n_rows == 0 {
            return None;
        }

        let mut max_distance = 0;
        let n_samples = (self.n_rows as f64).sqrt() as usize + 1;

        // Sample starting vertices for diameter approximation
        for start in (0..self.n_rows).step_by(self.n_rows / n_samples.max(1)) {
            let distances = self.bfs_distances(start);
            // Find max distance that is not usize::MAX (unreachable)
            if let Some(max_dist) = distances.iter().filter(|&&d| d != usize::MAX).max() {
                max_distance = max_distance.max(*max_dist);
            }
        }

        Some(max_distance)
    }

    /// BFS to compute distances from a source vertex
    fn bfs_distances(&self, source: usize) -> Vec<usize> {
        let mut distances = vec![usize::MAX; self.n_rows];
        let mut queue = VecDeque::new();

        distances[source] = 0;
        queue.push_back(source);

        while let Some(vertex) = queue.pop_front() {
            let start = self.row_pointers[vertex];
            let end = self.row_pointers[vertex + 1];

            for idx in start..end {
                let neighbor = self.col_indices[idx];
                if distances[neighbor] == usize::MAX {
                    distances[neighbor] = distances[vertex] + 1;
                    queue.push_back(neighbor);
                }
            }
        }

        distances
    }

    /// Advanced: Compute clustering coefficient for a vertex
    /// Measures how connected a vertex's neighbors are to each other
    pub fn clustering_coefficient(&self, vertex: usize) -> Float {
        if vertex >= self.n_rows {
            return 0.0;
        }

        let neighbors = self.row_entries(vertex);
        let degree = neighbors.len();

        if degree < 2 {
            return 0.0;
        }

        let mut triangles = 0;

        // Count triangles: edges between neighbors
        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                let neighbor1 = neighbors[i].0;
                let neighbor2 = neighbors[j].0;

                // Check if neighbor1 and neighbor2 are connected
                if self.get(neighbor1, neighbor2) > 0.0 {
                    triangles += 1;
                }
            }
        }

        // Clustering coefficient = 2 * triangles / (degree * (degree - 1))
        2.0 * triangles as Float / (degree * (degree - 1)) as Float
    }

    /// Advanced: Compute average clustering coefficient for the entire graph
    pub fn average_clustering_coefficient(&self) -> Float {
        let mut total_coefficient = 0.0;
        let mut valid_vertices = 0;

        for vertex in 0..self.n_rows {
            let coefficient = self.clustering_coefficient(vertex);
            if coefficient.is_finite() {
                total_coefficient += coefficient;
                valid_vertices += 1;
            }
        }

        if valid_vertices > 0 {
            total_coefficient / valid_vertices as Float
        } else {
            0.0
        }
    }

    /// Advanced: Approximate betweenness centrality using sampling
    /// Measures how often a vertex lies on shortest paths between other vertices
    pub fn approximate_betweenness_centrality(&self, sample_size: usize) -> Vec<Float> {
        let mut centrality = vec![0.0; self.n_rows];
        let sample_vertices: Vec<usize> = (0..self.n_rows)
            .step_by((self.n_rows / sample_size.max(1)).max(1))
            .collect();

        for &source in &sample_vertices {
            let (distances, predecessors) = self.single_source_shortest_paths(source);

            // Count paths through each vertex
            let mut path_counts = vec![0.0; self.n_rows];
            let mut dependency = vec![0.0; self.n_rows];

            // Initialize path counts
            for i in 0..self.n_rows {
                if distances[i] != usize::MAX {
                    path_counts[i] = 1.0;
                }
            }

            // Process vertices in order of decreasing distance
            let mut vertices_by_distance: Vec<usize> = (0..self.n_rows).collect();
            vertices_by_distance.sort_by_key(|&v| std::cmp::Reverse(distances[v]));

            for &vertex in &vertices_by_distance {
                if distances[vertex] == usize::MAX {
                    continue;
                }

                for &pred in &predecessors[vertex] {
                    let contrib =
                        path_counts[pred] * (1.0 + dependency[vertex]) / path_counts[vertex];
                    dependency[pred] += contrib;
                }

                if vertex != source {
                    centrality[vertex] += dependency[vertex];
                }
            }
        }

        // Normalize by sample size
        let scale = sample_vertices.len() as Float;
        for centrality_val in &mut centrality {
            *centrality_val /= scale;
        }

        centrality
    }

    /// Single-source shortest paths with predecessor tracking
    fn single_source_shortest_paths(&self, source: usize) -> (Vec<usize>, Vec<Vec<usize>>) {
        let mut distances = vec![usize::MAX; self.n_rows];
        let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); self.n_rows];
        let mut queue = VecDeque::new();

        distances[source] = 0;
        queue.push_back(source);

        while let Some(vertex) = queue.pop_front() {
            let start = self.row_pointers[vertex];
            let end = self.row_pointers[vertex + 1];

            for idx in start..end {
                let neighbor = self.col_indices[idx];
                let new_dist = distances[vertex] + 1;

                if new_dist < distances[neighbor] {
                    distances[neighbor] = new_dist;
                    predecessors[neighbor].clear();
                    predecessors[neighbor].push(vertex);
                    queue.push_back(neighbor);
                } else if new_dist == distances[neighbor] {
                    predecessors[neighbor].push(vertex);
                }
            }
        }

        (distances, predecessors)
    }
}

/// Statistics for sparse matrix
#[derive(Debug, Clone)]
pub struct SparseMatrixStats {
    pub n_rows: usize,
    pub n_cols: usize,
    pub nnz: usize,
    pub total_entries: usize,
    pub sparsity: f64,
    pub dense_memory_bytes: usize,
    pub sparse_memory_bytes: usize,
    pub memory_savings: f64,
    pub symmetric: bool,
}

impl std::fmt::Display for SparseMatrixStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SparseMatrix {{ {}×{}, {:.2}% sparse, {} nnz, {:.1}% memory savings }}",
            self.n_rows,
            self.n_cols,
            self.sparsity * 100.0,
            self.nnz,
            self.memory_savings * 100.0
        )
    }
}

/// Sparse neighborhood graph for clustering algorithms
pub struct SparseNeighborhoodGraph {
    /// Sparse adjacency matrix
    adjacency: SparseDistanceMatrix,
    /// Vertex degrees
    degrees: Vec<usize>,
}

impl SparseNeighborhoodGraph {
    /// Create neighborhood graph from sparse distance matrix
    pub fn from_sparse_matrix(matrix: SparseDistanceMatrix) -> Self {
        let n_vertices = matrix.n_rows;
        let mut degrees = vec![0; n_vertices];

        // Calculate degrees
        for i in 0..n_vertices {
            degrees[i] = matrix.row_entries(i).len();
        }

        Self {
            adjacency: matrix,
            degrees,
        }
    }

    /// Get neighbors of a vertex
    pub fn neighbors(&self, vertex: usize) -> Vec<(usize, Float)> {
        self.adjacency.row_entries(vertex)
    }

    /// Get degree of a vertex
    pub fn degree(&self, vertex: usize) -> usize {
        self.degrees.get(vertex).copied().unwrap_or(0)
    }

    /// Get all vertices
    pub fn vertices(&self) -> std::ops::Range<usize> {
        0..self.adjacency.n_rows
    }

    /// Number of vertices
    pub fn n_vertices(&self) -> usize {
        self.adjacency.n_rows
    }

    /// Number of edges
    pub fn n_edges(&self) -> usize {
        if self.adjacency.symmetric {
            self.adjacency.nnz / 2
        } else {
            self.adjacency.nnz
        }
    }

    /// Graph statistics
    pub fn graph_stats(&self) -> GraphStats {
        let total_degree: usize = self.degrees.iter().sum();
        let avg_degree = total_degree as f64 / self.n_vertices() as f64;
        let max_degree = *self.degrees.iter().max().unwrap_or(&0);
        let min_degree = *self.degrees.iter().min().unwrap_or(&0);

        GraphStats {
            n_vertices: self.n_vertices(),
            n_edges: self.n_edges(),
            avg_degree,
            max_degree,
            min_degree,
            matrix_stats: self.adjacency.stats(),
        }
    }
}

/// Statistics for sparse neighborhood graph
#[derive(Debug, Clone)]
pub struct GraphStats {
    pub n_vertices: usize,
    pub n_edges: usize,
    pub avg_degree: f64,
    pub max_degree: usize,
    pub min_degree: usize,
    pub matrix_stats: SparseMatrixStats,
}

impl std::fmt::Display for GraphStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Graph {{ {} vertices, {} edges, avg degree: {:.1}, degree range: {}-{} }}",
            self.n_vertices, self.n_edges, self.avg_degree, self.min_degree, self.max_degree
        )
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_sparse_matrix_creation() {
        let matrix = SparseDistanceMatrix::new(3, 3, true);

        assert_eq!(matrix.shape(), (3, 3));
        assert_eq!(matrix.nnz(), 0);
        assert!(matrix.is_symmetric());

        // All entries should be zero initially
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(matrix.get(i, j), 0.0);
            }
        }
    }

    #[test]
    fn test_sparse_matrix_set_get() {
        let mut matrix = SparseDistanceMatrix::new(3, 3, true);

        // Set some values
        matrix.set(0, 1, 2.5).unwrap();
        matrix.set(1, 2, 3.0).unwrap();

        // Check values
        assert_eq!(matrix.get(0, 1), 2.5);
        assert_eq!(matrix.get(1, 0), 2.5); // Symmetric
        assert_eq!(matrix.get(1, 2), 3.0);
        assert_eq!(matrix.get(2, 1), 3.0); // Symmetric
        assert_eq!(matrix.get(0, 2), 0.0); // Not set

        // Diagonal should remain zero
        for i in 0..3 {
            assert_eq!(matrix.get(i, i), 0.0);
        }
    }

    #[test]
    fn test_sparse_matrix_from_data() {
        // Create data with clear clusters (sparse distances)
        let data = array![
            [0.0, 0.0],
            [0.1, 0.0],  // Close to first point
            [10.0, 0.0], // Far from others
            [10.1, 0.0], // Close to third point
        ];

        let config = SparseMatrixConfig {
            distance_threshold: 1.0, // Only store distances <= 1.0
            sparsity_threshold: 0.3, // Allow 30% sparse
            ..Default::default()
        };

        let sparse_matrix = SparseDistanceMatrix::from_data(&data, config).unwrap();

        // Should have captured close pairs only
        assert!(sparse_matrix.get(0, 1) > 0.0); // Points 0 and 1 are close
        assert!(sparse_matrix.get(2, 3) > 0.0); // Points 2 and 3 are close
        assert_eq!(sparse_matrix.get(0, 2), 0.0); // Points 0 and 2 are far
        assert_eq!(sparse_matrix.get(1, 3), 0.0); // Points 1 and 3 are far
    }

    #[test]
    fn test_k_nearest_neighbors() {
        let mut matrix = SparseDistanceMatrix::new(4, 4, true);

        // Set up distances: point 0 connected to 1 (distance 1.0) and 2 (distance 2.0)
        matrix.set(0, 1, 1.0).unwrap();
        matrix.set(0, 2, 2.0).unwrap();
        matrix.set(0, 3, 5.0).unwrap();

        let neighbors = matrix.k_nearest_neighbors(0, 2);

        assert_eq!(neighbors.len(), 2);
        assert_eq!(neighbors[0], (1, 1.0)); // Closest
        assert_eq!(neighbors[1], (2, 2.0)); // Second closest
    }

    #[test]
    fn test_neighbors_within_radius() {
        let mut matrix = SparseDistanceMatrix::new(4, 4, true);

        matrix.set(0, 1, 1.0).unwrap();
        matrix.set(0, 2, 2.0).unwrap();
        matrix.set(0, 3, 5.0).unwrap();

        let neighbors = matrix.neighbors_within_radius(0, 2.5);

        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.iter().any(|&(idx, _)| idx == 1));
        assert!(neighbors.iter().any(|&(idx, _)| idx == 2));
        assert!(!neighbors.iter().any(|&(idx, _)| idx == 3)); // Distance 5.0 > 2.5
    }

    #[test]
    fn test_sparse_matrix_stats() {
        let mut matrix = SparseDistanceMatrix::new(4, 4, true);

        matrix.set(0, 1, 1.0).unwrap();
        matrix.set(1, 2, 2.0).unwrap();

        let stats = matrix.stats();

        assert_eq!(stats.n_rows, 4);
        assert_eq!(stats.n_cols, 4);
        assert_eq!(stats.nnz, 4); // 2 entries + 2 symmetric entries
        assert!(stats.sparsity > 0.5); // Should be quite sparse
        assert!(stats.memory_savings > 0.0);
    }

    #[test]
    fn test_to_dense_conversion() {
        let mut matrix = SparseDistanceMatrix::new(3, 3, true);

        matrix.set(0, 1, 1.5).unwrap();
        matrix.set(1, 2, 2.5).unwrap();

        let dense = matrix.to_dense();

        assert_eq!(dense.shape(), &[3, 3]);
        assert_eq!(dense[[0, 1]], 1.5);
        assert_eq!(dense[[1, 0]], 1.5); // Symmetric
        assert_eq!(dense[[1, 2]], 2.5);
        assert_eq!(dense[[2, 1]], 2.5); // Symmetric
        assert_eq!(dense[[0, 2]], 0.0); // Not connected

        // Diagonal should be zero
        for i in 0..3 {
            assert_eq!(dense[[i, i]], 0.0);
        }
    }

    #[test]
    fn test_neighborhood_graph() {
        let mut matrix = SparseDistanceMatrix::new(4, 4, true);

        matrix.set(0, 1, 1.0).unwrap();
        matrix.set(1, 2, 1.5).unwrap();
        matrix.set(2, 3, 2.0).unwrap();

        let graph = SparseNeighborhoodGraph::from_sparse_matrix(matrix);

        assert_eq!(graph.n_vertices(), 4);
        assert_eq!(graph.n_edges(), 3); // 3 undirected edges

        // Check degrees
        assert_eq!(graph.degree(0), 1); // Connected to 1
        assert_eq!(graph.degree(1), 2); // Connected to 0 and 2
        assert_eq!(graph.degree(2), 2); // Connected to 1 and 3
        assert_eq!(graph.degree(3), 1); // Connected to 2

        // Check neighbors
        let neighbors_1 = graph.neighbors(1);
        assert_eq!(neighbors_1.len(), 2);
    }

    #[test]
    fn test_graph_stats() {
        let mut matrix = SparseDistanceMatrix::new(3, 3, true);

        matrix.set(0, 1, 1.0).unwrap();
        matrix.set(1, 2, 1.5).unwrap();

        let graph = SparseNeighborhoodGraph::from_sparse_matrix(matrix);
        let stats = graph.graph_stats();

        assert_eq!(stats.n_vertices, 3);
        assert_eq!(stats.n_edges, 2);
        assert!((stats.avg_degree - 4.0 / 3.0).abs() < 1e-10); // Total degree 4, 3 vertices
        assert_eq!(stats.max_degree, 2); // Vertex 1 has degree 2
        assert_eq!(stats.min_degree, 1); // Vertices 0 and 2 have degree 1
    }

    #[test]
    fn test_connected_components() {
        let mut matrix = SparseDistanceMatrix::new(5, 5, true);

        // Create two disconnected components: {0,1,2} and {3,4}
        matrix.set(0, 1, 1.0).unwrap();
        matrix.set(1, 2, 1.0).unwrap();
        matrix.set(3, 4, 1.0).unwrap();

        let components = matrix.connected_components();

        // Vertices 0, 1, 2 should be in the same component
        assert_eq!(components[0], components[1]);
        assert_eq!(components[1], components[2]);

        // Vertices 3, 4 should be in the same component (different from above)
        assert_eq!(components[3], components[4]);
        assert_ne!(components[0], components[3]);

        // Each component should have at least one member
        let unique_components: std::collections::HashSet<_> = components.iter().collect();
        assert!(unique_components.len() >= 2);
    }

    #[test]
    fn test_graph_diameter() {
        let mut matrix = SparseDistanceMatrix::new(4, 4, true);

        // Create a linear graph: 0-1-2-3
        matrix.set(0, 1, 1.0).unwrap();
        matrix.set(1, 2, 1.0).unwrap();
        matrix.set(2, 3, 1.0).unwrap();

        let diameter = matrix.graph_diameter();
        assert_eq!(diameter, Some(3)); // Distance from 0 to 3
    }

    #[test]
    fn test_clustering_coefficient() {
        let mut matrix = SparseDistanceMatrix::new(4, 4, true);

        // Create a triangle plus one: 0-1-2-0, and 1-3
        matrix.set(0, 1, 1.0).unwrap();
        matrix.set(1, 2, 1.0).unwrap();
        matrix.set(2, 0, 1.0).unwrap(); // Triangle complete
        matrix.set(1, 3, 1.0).unwrap();

        // Vertex 0: neighbors are {1, 2}, they are connected → coefficient = 1.0
        let coeff_0 = matrix.clustering_coefficient(0);
        assert!((coeff_0 - 1.0).abs() < 1e-10);

        // Vertex 1: neighbors are {0, 2, 3}, 0-2 connected, others not → coefficient = 1/3
        let coeff_1 = matrix.clustering_coefficient(1);
        assert!((coeff_1 - 1.0 / 3.0).abs() < 1e-10);

        // Vertex 3: only one neighbor → coefficient = 0.0
        let coeff_3 = matrix.clustering_coefficient(3);
        assert_eq!(coeff_3, 0.0);
    }

    #[test]
    fn test_average_clustering_coefficient() {
        let mut matrix = SparseDistanceMatrix::new(3, 3, true);

        // Create a complete triangle
        matrix.set(0, 1, 1.0).unwrap();
        matrix.set(1, 2, 1.0).unwrap();
        matrix.set(2, 0, 1.0).unwrap();

        let avg_coeff = matrix.average_clustering_coefficient();
        // In a complete triangle, all vertices have clustering coefficient 1.0
        assert!((avg_coeff - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_betweenness_centrality() {
        let mut matrix = SparseDistanceMatrix::new(4, 4, true);

        // Create a linear graph: 0-1-2-3
        // Vertices 1 and 2 should have higher betweenness centrality
        matrix.set(0, 1, 1.0).unwrap();
        matrix.set(1, 2, 1.0).unwrap();
        matrix.set(2, 3, 1.0).unwrap();

        let centrality = matrix.approximate_betweenness_centrality(4);

        // End vertices (0, 3) should have lower centrality than middle vertices (1, 2)
        assert!(centrality[1] > centrality[0]);
        assert!(centrality[2] > centrality[3]);
        assert!((centrality[1] - centrality[2]).abs() < 0.1); // Should be similar
    }

    #[test]
    fn test_bfs_distances() {
        let mut matrix = SparseDistanceMatrix::new(4, 4, true);

        // Create a simple path: 0-1-2-3
        matrix.set(0, 1, 1.0).unwrap();
        matrix.set(1, 2, 1.0).unwrap();
        matrix.set(2, 3, 1.0).unwrap();

        let distances = matrix.bfs_distances(0);

        assert_eq!(distances[0], 0);
        assert_eq!(distances[1], 1);
        assert_eq!(distances[2], 2);
        assert_eq!(distances[3], 3);
    }

    #[test]
    fn test_advanced_algorithms_empty_graph() {
        let matrix = SparseDistanceMatrix::new(3, 3, true);

        // Test empty graph behavior
        let components = matrix.connected_components();
        assert_eq!(components.len(), 3);
        // Each vertex should be its own component
        assert_ne!(components[0], components[1]);
        assert_ne!(components[1], components[2]);

        let diameter = matrix.graph_diameter();
        assert_eq!(diameter, Some(0));

        let avg_coeff = matrix.average_clustering_coefficient();
        assert_eq!(avg_coeff, 0.0);
    }
}
