//! Locality-Sensitive Hashing (LSH) for approximate distance computations
//!
//! This module provides LSH implementations for fast approximate nearest neighbor
//! search and similarity estimation. LSH is particularly useful for high-dimensional
//! data where exact distance computation becomes prohibitively expensive.

use std::collections::{HashMap, HashSet};
use std::hash::Hasher;

use sklears_core::{
    error::{Result, SklearsError},
    types::{Array1, Array2, Float},
};

use scirs2_core::random::{thread_rng, CoreRandom, Rng};

/// LSH family type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LSHFamily {
    /// Random hyperplane hashing (for cosine similarity)
    RandomHyperplane,
    /// Random projection hashing (for Euclidean distance)
    RandomProjection,
    /// MinHash (for Jaccard similarity on binary/sparse data)
    MinHash,
    /// SimHash (for Hamming distance)
    SimHash,
}

/// Configuration for LSH
#[derive(Debug, Clone)]
pub struct LSHConfig {
    /// LSH family to use
    pub family: LSHFamily,
    /// Number of hash functions per table
    pub num_hash_functions: usize,
    /// Number of hash tables
    pub num_tables: usize,
    /// Input dimension (for hash function generation)
    pub input_dim: usize,
    /// Output dimension for projections
    pub projection_dim: Option<usize>,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Distance threshold for candidate pairs
    pub distance_threshold: Float,
}

impl Default for LSHConfig {
    fn default() -> Self {
        Self {
            family: LSHFamily::RandomHyperplane,
            num_hash_functions: 10,
            num_tables: 5,
            input_dim: 100,
            projection_dim: None,
            seed: 42,
            distance_threshold: 0.1,
        }
    }
}

/// Hash function for LSH
#[derive(Debug, Clone)]
pub struct LSHHashFunction {
    /// Random vector for projection
    projection_vector: Array1<Float>,
    /// Bias term (for some hash families)
    bias: Float,
    /// Hash family type
    family: LSHFamily,
}

impl LSHHashFunction {
    /// Create a new hash function
    pub fn new(family: LSHFamily, input_dim: usize, rng: &mut impl Rng) -> Self {
        let projection_vector = match family {
            LSHFamily::RandomHyperplane | LSHFamily::RandomProjection => {
                // Generate random Gaussian vector
                let mut vec = Array1::zeros(input_dim);
                for i in 0..input_dim {
                    vec[i] = rng.gen::<Float>() * 2.0 - 1.0; // Uniform [-1, 1] approximation
                }
                vec
            }
            LSHFamily::MinHash | LSHFamily::SimHash => {
                // Generate random permutation indices (simplified)
                let mut vec = Array1::zeros(input_dim);
                for i in 0..input_dim {
                    vec[i] = rng.gen::<Float>();
                }
                vec
            }
        };

        let bias = match family {
            LSHFamily::RandomProjection => rng.gen::<Float>() * 2.0 - 1.0,
            _ => 0.0,
        };

        Self {
            projection_vector,
            bias,
            family,
        }
    }

    /// Hash a vector
    pub fn hash(&self, vector: &Array1<Float>) -> u64 {
        match self.family {
            LSHFamily::RandomHyperplane => {
                let dot_product = vector.dot(&self.projection_vector);
                if dot_product >= 0.0 {
                    1
                } else {
                    0
                }
            }
            LSHFamily::RandomProjection => {
                let projection = vector.dot(&self.projection_vector) + self.bias;
                (projection.floor() as i64) as u64
            }
            LSHFamily::MinHash => {
                // Simplified MinHash: find minimum hash value
                let mut min_hash = Float::INFINITY;
                for (i, &val) in vector.iter().enumerate() {
                    if val > 0.0 {
                        let hash_val = self.projection_vector[i] + val;
                        min_hash = min_hash.min(hash_val);
                    }
                }
                min_hash.to_bits()
            }
            LSHFamily::SimHash => {
                // Simplified SimHash: weighted sum
                let mut hash_bits = 0u64;
                let sum = vector.dot(&self.projection_vector);
                if sum >= 0.0 {
                    hash_bits = 1;
                }
                hash_bits
            }
        }
    }
}

/// LSH table storing hash buckets
#[derive(Debug, Clone)]
pub struct LSHTable {
    /// Hash functions for this table
    hash_functions: Vec<LSHHashFunction>,
    /// Buckets mapping hash signatures to point indices
    buckets: HashMap<Vec<u64>, Vec<usize>>,
    /// Configuration
    config: LSHConfig,
}

impl LSHTable {
    /// Create a new LSH table
    pub fn new(config: LSHConfig, rng: &mut impl Rng) -> Self {
        let mut hash_functions = Vec::new();
        for _ in 0..config.num_hash_functions {
            hash_functions.push(LSHHashFunction::new(config.family, config.input_dim, rng));
        }

        Self {
            hash_functions,
            buckets: HashMap::new(),
            config,
        }
    }

    /// Insert a point into the table
    pub fn insert(&mut self, point_id: usize, vector: &Array1<Float>) {
        let signature = self.compute_signature(vector);
        self.buckets.entry(signature).or_default().push(point_id);
    }

    /// Compute hash signature for a vector
    fn compute_signature(&self, vector: &Array1<Float>) -> Vec<u64> {
        self.hash_functions
            .iter()
            .map(|hf| hf.hash(vector))
            .collect()
    }

    /// Query for candidate neighbors
    pub fn query(&self, vector: &Array1<Float>) -> Vec<usize> {
        let signature = self.compute_signature(vector);
        self.buckets.get(&signature).cloned().unwrap_or_default()
    }

    /// Get statistics about the table
    pub fn stats(&self) -> TableStats {
        let total_points: usize = self.buckets.values().map(|bucket| bucket.len()).sum();
        let num_buckets = self.buckets.len();
        let avg_bucket_size = if num_buckets > 0 {
            total_points as f64 / num_buckets as f64
        } else {
            0.0
        };
        let max_bucket_size = self
            .buckets
            .values()
            .map(|bucket| bucket.len())
            .max()
            .unwrap_or(0);

        TableStats {
            num_buckets,
            total_points,
            avg_bucket_size,
            max_bucket_size,
            num_hash_functions: self.hash_functions.len(),
        }
    }
}

/// Statistics for an LSH table
#[derive(Debug, Clone)]
pub struct TableStats {
    pub num_buckets: usize,
    pub total_points: usize,
    pub avg_bucket_size: f64,
    pub max_bucket_size: usize,
    pub num_hash_functions: usize,
}

/// Locality-Sensitive Hashing index for approximate nearest neighbor search
#[derive(Debug)]
pub struct LSHIndex {
    /// Hash tables
    tables: Vec<LSHTable>,
    /// Stored vectors
    vectors: Vec<Array1<Float>>,
    /// Configuration
    config: LSHConfig,
    /// Random number generator state
    rng: CoreRandom,
}

impl LSHIndex {
    /// Create a new LSH index
    pub fn new(config: LSHConfig) -> Self {
        let mut rng = thread_rng();
        let mut tables = Vec::new();

        for _ in 0..config.num_tables {
            tables.push(LSHTable::new(config.clone(), &mut rng));
        }

        Self {
            tables,
            vectors: Vec::new(),
            config,
            rng,
        }
    }

    /// Build index from data matrix
    ///
    /// # Arguments
    /// * `data` - Input data matrix (n_samples × n_features)
    pub fn build_from_data(&mut self, data: &Array2<Float>) -> Result<()> {
        if data.ncols() != self.config.input_dim {
            return Err(SklearsError::InvalidInput(format!(
                "Input dimension mismatch: expected {}, got {}",
                self.config.input_dim,
                data.ncols()
            )));
        }

        // Store vectors
        self.vectors.clear();
        for row in data.outer_iter() {
            self.vectors.push(row.to_owned());
        }

        // Insert into all tables
        for (point_id, vector) in self.vectors.iter().enumerate() {
            for table in &mut self.tables {
                table.insert(point_id, vector);
            }

            // Log progress for large datasets
            if (point_id + 1) % 10000 == 0 || point_id == self.vectors.len() - 1 {
                let progress = (point_id + 1) as f64 / self.vectors.len() as f64 * 100.0;
                eprintln!("Building LSH index: {:.1}% complete", progress);
            }
        }

        Ok(())
    }

    /// Query for approximate nearest neighbors
    ///
    /// # Arguments
    /// * `query_vector` - Vector to find neighbors for
    /// * `k` - Number of nearest neighbors to return
    ///
    /// # Returns
    /// Vector of (point_id, distance) pairs sorted by distance
    pub fn query_k_nearest(
        &self,
        query_vector: &Array1<Float>,
        k: usize,
    ) -> Result<Vec<(usize, Float)>> {
        if query_vector.len() != self.config.input_dim {
            return Err(SklearsError::InvalidInput(format!(
                "Query vector dimension mismatch: expected {}, got {}",
                self.config.input_dim,
                query_vector.len()
            )));
        }

        // Collect candidates from all tables
        let mut candidates = HashSet::new();
        for table in &self.tables {
            let table_candidates = table.query(query_vector);
            candidates.extend(table_candidates);
        }

        // Compute exact distances for candidates
        let mut neighbor_candidates = Vec::new();
        for &candidate_id in &candidates {
            if candidate_id < self.vectors.len() {
                let distance = self.compute_distance(query_vector, &self.vectors[candidate_id]);
                neighbor_candidates.push((candidate_id, distance));
            }
        }

        // Sort by distance and return top k
        neighbor_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        neighbor_candidates.truncate(k);

        Ok(neighbor_candidates)
    }

    /// Query for neighbors within a radius
    ///
    /// # Arguments
    /// * `query_vector` - Vector to find neighbors for
    /// * `radius` - Maximum distance for neighbors
    ///
    /// # Returns
    /// Vector of (point_id, distance) pairs within radius
    pub fn query_radius(
        &self,
        query_vector: &Array1<Float>,
        radius: Float,
    ) -> Result<Vec<(usize, Float)>> {
        if query_vector.len() != self.config.input_dim {
            return Err(SklearsError::InvalidInput(format!(
                "Query vector dimension mismatch: expected {}, got {}",
                self.config.input_dim,
                query_vector.len()
            )));
        }

        // Collect candidates from all tables
        let mut candidates = HashSet::new();
        for table in &self.tables {
            let table_candidates = table.query(query_vector);
            candidates.extend(table_candidates);
        }

        // Filter by radius and compute exact distances
        let mut neighbors = Vec::new();
        for &candidate_id in &candidates {
            if candidate_id < self.vectors.len() {
                let distance = self.compute_distance(query_vector, &self.vectors[candidate_id]);
                if distance <= radius {
                    neighbors.push((candidate_id, distance));
                }
            }
        }

        // Sort by distance
        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        Ok(neighbors)
    }

    /// Find all approximate nearest neighbor pairs within a threshold
    ///
    /// This is useful for clustering algorithms that need to find all pairs
    /// of points within a certain distance threshold.
    pub fn find_neighbor_pairs(&self) -> Result<Vec<(usize, usize, Float)>> {
        let mut pairs = Vec::new();
        let mut processed_pairs = HashSet::new();

        for i in 0..self.vectors.len() {
            let candidates = self.query_radius(&self.vectors[i], self.config.distance_threshold)?;

            for (j, distance) in candidates {
                if i < j {
                    let pair = (i, j);
                    if !processed_pairs.contains(&pair) {
                        pairs.push((i, j, distance));
                        processed_pairs.insert(pair);
                    }
                }
            }

            // Log progress
            if (i + 1) % 1000 == 0 || i == self.vectors.len() - 1 {
                let progress = (i + 1) as f64 / self.vectors.len() as f64 * 100.0;
                eprintln!("Finding neighbor pairs: {:.1}% complete", progress);
            }
        }

        Ok(pairs)
    }

    /// Compute distance between two vectors based on LSH family
    fn compute_distance(&self, v1: &Array1<Float>, v2: &Array1<Float>) -> Float {
        match self.config.family {
            LSHFamily::RandomHyperplane => {
                // Cosine distance
                let dot_product = v1.dot(v2);
                let norm1 = v1.dot(v1).sqrt();
                let norm2 = v2.dot(v2).sqrt();
                if norm1 > 0.0 && norm2 > 0.0 {
                    1.0 - (dot_product / (norm1 * norm2))
                } else {
                    1.0
                }
            }
            LSHFamily::RandomProjection => {
                // Euclidean distance
                let diff = v1 - v2;
                diff.dot(&diff).sqrt()
            }
            LSHFamily::MinHash => {
                // Jaccard distance (approximation for real-valued vectors)
                let mut intersection = 0.0;
                let mut union = 0.0;
                for i in 0..v1.len().min(v2.len()) {
                    let min_val = v1[i].min(v2[i]);
                    let max_val = v1[i].max(v2[i]);
                    intersection += min_val;
                    union += max_val;
                }
                if union > 0.0 {
                    1.0 - (intersection / union)
                } else {
                    0.0
                }
            }
            LSHFamily::SimHash => {
                // Hamming distance approximation
                let mut diff_count = 0;
                for i in 0..v1.len().min(v2.len()) {
                    if (v1[i] > 0.0) != (v2[i] > 0.0) {
                        diff_count += 1;
                    }
                }
                diff_count as Float / v1.len() as Float
            }
        }
    }

    /// Get index statistics
    pub fn stats(&self) -> LSHIndexStats {
        let table_stats: Vec<TableStats> = self.tables.iter().map(|table| table.stats()).collect();

        let total_buckets: usize = table_stats.iter().map(|stats| stats.num_buckets).sum();
        let avg_buckets_per_table = total_buckets as f64 / self.tables.len() as f64;

        let total_points: usize = table_stats.iter().map(|stats| stats.total_points).sum();
        let avg_points_per_table = total_points as f64 / self.tables.len() as f64;

        LSHIndexStats {
            num_tables: self.tables.len(),
            num_vectors: self.vectors.len(),
            total_buckets,
            avg_buckets_per_table,
            total_points,
            avg_points_per_table,
            table_stats,
            config: self.config.clone(),
        }
    }

    /// Estimate memory usage
    pub fn memory_usage(&self) -> MemoryUsage {
        let vector_memory =
            self.vectors.len() * self.config.input_dim * std::mem::size_of::<Float>();

        let table_memory: usize = self
            .tables
            .iter()
            .map(|table| {
                table.buckets.len()
                    * (std::mem::size_of::<Vec<u64>>() + std::mem::size_of::<Vec<usize>>())
                    + table.hash_functions.len()
                        * self.config.input_dim
                        * std::mem::size_of::<Float>()
            })
            .sum();

        let total_memory = vector_memory + table_memory;

        MemoryUsage {
            vector_memory_bytes: vector_memory,
            table_memory_bytes: table_memory,
            total_memory_bytes: total_memory,
            memory_mb: total_memory as f64 / (1024.0 * 1024.0),
            memory_gb: total_memory as f64 / (1024.0 * 1024.0 * 1024.0),
        }
    }
}

/// Statistics for LSH index
#[derive(Debug, Clone)]
pub struct LSHIndexStats {
    pub num_tables: usize,
    pub num_vectors: usize,
    pub total_buckets: usize,
    pub avg_buckets_per_table: f64,
    pub total_points: usize,
    pub avg_points_per_table: f64,
    pub table_stats: Vec<TableStats>,
    pub config: LSHConfig,
}

/// Memory usage information
#[derive(Debug, Clone)]
pub struct MemoryUsage {
    pub vector_memory_bytes: usize,
    pub table_memory_bytes: usize,
    pub total_memory_bytes: usize,
    pub memory_mb: f64,
    pub memory_gb: f64,
}

impl std::fmt::Display for LSHIndexStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LSHIndex {{ {} tables, {} vectors, avg {:.1} buckets/table, family: {:?} }}",
            self.num_tables, self.num_vectors, self.avg_buckets_per_table, self.config.family
        )
    }
}

impl std::fmt::Display for MemoryUsage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Memory {{ {:.1} MB total, {:.1} MB vectors, {:.1} MB tables }}",
            self.memory_mb,
            self.vector_memory_bytes as f64 / (1024.0 * 1024.0),
            self.table_memory_bytes as f64 / (1024.0 * 1024.0)
        )
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;
    use scirs2_core::random::Random;

    #[test]
    fn test_lsh_hash_function_creation() {
        let mut rng = Random::default();
        let hash_fn = LSHHashFunction::new(LSHFamily::RandomHyperplane, 5, &mut rng);

        assert_eq!(hash_fn.projection_vector.len(), 5);
        assert_eq!(hash_fn.family, LSHFamily::RandomHyperplane);
    }

    #[test]
    fn test_random_hyperplane_hashing() {
        let mut rng = Random::default();
        let hash_fn = LSHHashFunction::new(LSHFamily::RandomHyperplane, 3, &mut rng);

        let v1 = array![1.0, 2.0, 3.0];
        let v2 = array![-1.0, -2.0, -3.0];

        let h1 = hash_fn.hash(&v1);
        let h2 = hash_fn.hash(&v2);

        // Opposite vectors should likely have different hash values
        // (though not guaranteed due to randomness)
        assert!(h1 == 0 || h1 == 1);
        assert!(h2 == 0 || h2 == 1);
    }

    #[test]
    fn test_lsh_table_creation() {
        let config = LSHConfig {
            family: LSHFamily::RandomHyperplane,
            num_hash_functions: 3,
            input_dim: 5,
            ..Default::default()
        };

        let mut rng = Random::default();
        let table = LSHTable::new(config, &mut rng);

        assert_eq!(table.hash_functions.len(), 3);
        assert!(table.buckets.is_empty());
    }

    #[test]
    fn test_lsh_table_insert_and_query() {
        let config = LSHConfig {
            family: LSHFamily::RandomHyperplane,
            num_hash_functions: 3,
            input_dim: 3,
            ..Default::default()
        };

        let mut rng = Random::default();
        let mut table = LSHTable::new(config, &mut rng);

        let v1 = array![1.0, 2.0, 3.0];
        let v2 = array![1.1, 2.1, 3.1]; // Similar to v1
        let v3 = array![-10.0, -20.0, -30.0]; // Very different

        table.insert(0, &v1);
        table.insert(1, &v2);
        table.insert(2, &v3);

        // Query with a vector similar to v1 and v2
        let query = array![1.05, 2.05, 3.05];
        let candidates = table.query(&query);

        // Should return some candidates (exact behavior depends on hash functions)
        assert!(!candidates.is_empty());
    }

    #[test]
    fn test_lsh_index_creation() {
        let config = LSHConfig {
            family: LSHFamily::RandomHyperplane,
            num_tables: 2,
            num_hash_functions: 3,
            input_dim: 4,
            ..Default::default()
        };

        let index = LSHIndex::new(config);

        assert_eq!(index.tables.len(), 2);
        assert!(index.vectors.is_empty());
    }

    #[test]
    fn test_lsh_index_build_and_query() {
        let data = array![
            [1.0, 2.0, 3.0],
            [1.1, 2.1, 3.1],
            [1.2, 2.2, 3.2],
            [-10.0, -20.0, -30.0],
        ];

        let config = LSHConfig {
            family: LSHFamily::RandomHyperplane,
            num_tables: 2,
            num_hash_functions: 5,
            input_dim: 3,
            ..Default::default()
        };

        let mut index = LSHIndex::new(config);
        index.build_from_data(&data).unwrap();

        assert_eq!(index.vectors.len(), 4);

        // Query for neighbors of first point
        let query = array![1.05, 2.05, 3.05];
        let neighbors = index.query_k_nearest(&query, 2).unwrap();

        // Should find some neighbors
        assert!(!neighbors.is_empty());
        assert!(neighbors.len() <= 2);

        // Distances should be non-negative
        for &(_, distance) in &neighbors {
            assert!(distance >= 0.0);
        }
    }

    #[test]
    fn test_lsh_radius_query() {
        let data = array![[0.0, 0.0], [0.1, 0.1], [10.0, 10.0],];

        let config = LSHConfig {
            family: LSHFamily::RandomProjection,
            num_tables: 3,
            num_hash_functions: 5,
            input_dim: 2,
            distance_threshold: 1.0,
            ..Default::default()
        };

        let mut index = LSHIndex::new(config);
        index.build_from_data(&data).unwrap();

        let query = array![0.05, 0.05];
        let neighbors = index.query_radius(&query, 0.5).unwrap();

        // Should find close neighbors but not the far one
        assert!(!neighbors.is_empty());
        for &(_, distance) in &neighbors {
            assert!(distance <= 0.5);
        }
    }

    #[test]
    fn test_distance_computation() {
        let config = LSHConfig {
            family: LSHFamily::RandomProjection,
            input_dim: 2,
            ..Default::default()
        };

        let index = LSHIndex::new(config);

        let v1 = array![3.0, 4.0];
        let v2 = array![0.0, 0.0];

        let distance = index.compute_distance(&v1, &v2);

        // Should be Euclidean distance: sqrt(3^2 + 4^2) = 5.0
        assert!((distance - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_lsh_stats() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],];

        let config = LSHConfig {
            family: LSHFamily::RandomHyperplane,
            num_tables: 2,
            num_hash_functions: 3,
            input_dim: 2,
            ..Default::default()
        };

        let mut index = LSHIndex::new(config);
        index.build_from_data(&data).unwrap();

        let stats = index.stats();

        assert_eq!(stats.num_tables, 2);
        assert_eq!(stats.num_vectors, 3);
        assert!(stats.total_buckets > 0);
        assert_eq!(stats.table_stats.len(), 2);
    }

    #[test]
    fn test_memory_usage() {
        let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0],];

        let config = LSHConfig {
            family: LSHFamily::RandomHyperplane,
            num_tables: 1,
            num_hash_functions: 2,
            input_dim: 3,
            ..Default::default()
        };

        let mut index = LSHIndex::new(config);
        index.build_from_data(&data).unwrap();

        let memory = index.memory_usage();

        assert!(memory.total_memory_bytes > 0);
        assert!(memory.vector_memory_bytes > 0);
        assert!(memory.table_memory_bytes > 0);
        assert!(memory.memory_mb > 0.0);
    }
}
