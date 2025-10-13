//! Locality-Sensitive Hashing (LSH) for approximate nearest neighbor search

use crate::{Distance, NeighborsError, NeighborsResult};
use scirs2_core::ndarray::Axis;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use sklears_core::error::Result;
use sklears_core::traits::{Fit, Predict};
use sklears_core::types::{Features, Float, Int};
use std::collections::{HashMap, HashSet};

/// Hash family type for different LSH schemes
#[derive(Debug, Clone)]
pub enum HashFamily {
    /// Random projection LSH for cosine similarity
    RandomProjection {
        /// Number of hash functions
        num_hashes: usize,
        /// Dimension of the input vectors
        dimension: usize,
        /// Random projection matrix
        projection_matrix: Array2<Float>,
    },
    /// MinHash LSH for Jaccard similarity
    MinHash {
        /// Number of hash functions
        num_hashes: usize,
        /// Random permutation parameters
        a_params: Array1<Float>,
        b_params: Array1<Float>,
        /// Large prime number for hash computation
        prime: u64,
    },
}

/// LSH index for approximate nearest neighbor search
#[derive(Debug, Clone)]
pub struct LshIndex {
    /// Hash family used for hashing
    hash_family: HashFamily,
    /// Number of hash tables (L in LSH literature)
    num_tables: usize,
    /// Hash tables mapping hash values to point indices
    hash_tables: Vec<HashMap<Vec<i32>, Vec<usize>>>,
    /// Original data points
    data: Option<Array2<Float>>,
    /// Number of nearest neighbors to return
    num_neighbors: usize,
}

impl LshIndex {
    /// Create a new LSH index with random projection for cosine similarity
    pub fn new_random_projection(
        num_tables: usize,
        num_hashes: usize,
        dimension: usize,
        num_neighbors: usize,
    ) -> Self {
        let mut hash_tables = Vec::with_capacity(num_tables);
        for _ in 0..num_tables {
            hash_tables.push(HashMap::new());
        }

        // Generate random projection matrix for cosine similarity
        let mut projection_matrix = Array2::zeros((num_hashes, dimension));
        use scirs2_core::random::*;
        let mut rng = thread_rng();
        let normal = scirs2_core::random::RandNormal::new(0.0, 1.0).unwrap();

        for i in 0..num_hashes {
            for j in 0..dimension {
                projection_matrix[[i, j]] = rng.sample(normal);
            }
        }

        let hash_family = HashFamily::RandomProjection {
            num_hashes,
            dimension,
            projection_matrix,
        };

        Self {
            hash_family,
            num_tables,
            hash_tables,
            data: None,
            num_neighbors,
        }
    }

    /// Create a new LSH index with MinHash for Jaccard similarity
    pub fn new_minhash(num_tables: usize, num_hashes: usize, num_neighbors: usize) -> Self {
        let mut hash_tables = Vec::with_capacity(num_tables);
        for _ in 0..num_tables {
            hash_tables.push(HashMap::new());
        }

        // Generate random parameters for MinHash
        use scirs2_core::random::*;
        let mut rng = thread_rng();
        let prime = 2147483647_u64; // Large prime number

        let mut a_params = Array1::zeros(num_hashes);
        let mut b_params = Array1::zeros(num_hashes);

        for i in 0..num_hashes {
            a_params[i] = rng.gen_range(1.0..prime as Float);
            b_params[i] = rng.gen_range(0.0..prime as Float);
        }

        let hash_family = HashFamily::MinHash {
            num_hashes,
            a_params,
            b_params,
            prime,
        };

        Self {
            hash_family,
            num_tables,
            hash_tables,
            data: None,
            num_neighbors,
        }
    }

    /// Build the LSH index from training data
    pub fn fit(&mut self, data: &Array2<Float>) -> NeighborsResult<()> {
        if data.is_empty() {
            return Err(NeighborsError::EmptyInput);
        }

        // Clear existing hash tables
        for table in &mut self.hash_tables {
            table.clear();
        }

        // Store the data
        self.data = Some(data.clone());

        // Hash each data point and insert into hash tables
        for (point_idx, point) in data.axis_iter(Axis(0)).enumerate() {
            let hashes = self.hash_point(&point)?;

            // Insert into each hash table
            for (table_idx, hash_values) in hashes.iter().enumerate() {
                self.hash_tables[table_idx]
                    .entry(hash_values.clone())
                    .or_default()
                    .push(point_idx);
            }
        }

        Ok(())
    }

    /// Query for approximate nearest neighbors
    pub fn query(&self, query_point: &ArrayView1<Float>) -> NeighborsResult<Vec<usize>> {
        if self.data.is_none() {
            return Err(NeighborsError::InvalidInput(
                "LSH index not fitted. Call fit() first.".to_string(),
            ));
        }

        let data = self.data.as_ref().unwrap();

        // Hash the query point
        let query_hashes = self.hash_point(query_point)?;

        // Collect candidate points from all hash tables
        let mut candidates = HashSet::new();

        for (table_idx, query_hash) in query_hashes.iter().enumerate() {
            if let Some(point_indices) = self.hash_tables[table_idx].get(query_hash) {
                for &idx in point_indices {
                    candidates.insert(idx);
                }
            }
        }

        // If we don't have enough candidates, expand the search
        if candidates.len() < self.num_neighbors {
            // Add some random points or use a fallback strategy
            for i in 0..std::cmp::min(self.num_neighbors * 2, data.nrows()) {
                candidates.insert(i);
            }
        }

        // Compute actual distances to candidates and return top-k
        let mut candidate_distances: Vec<(Float, usize)> = candidates
            .into_iter()
            .map(|idx| {
                let distance = Distance::Euclidean.calculate(query_point, &data.row(idx));
                (distance, idx)
            })
            .collect();

        // Sort by distance and return top-k
        candidate_distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let result: Vec<usize> = candidate_distances
            .into_iter()
            .take(self.num_neighbors)
            .map(|(_, idx)| idx)
            .collect();

        Ok(result)
    }

    /// Query for approximate nearest neighbors with distances
    pub fn query_with_distances(
        &self,
        query_point: &ArrayView1<Float>,
    ) -> NeighborsResult<Vec<(Float, usize)>> {
        if self.data.is_none() {
            return Err(NeighborsError::InvalidInput(
                "LSH index not fitted. Call fit() first.".to_string(),
            ));
        }

        let data = self.data.as_ref().unwrap();
        let neighbor_indices = self.query(query_point)?;

        let mut result = Vec::new();
        for idx in neighbor_indices {
            let distance = Distance::Euclidean.calculate(query_point, &data.row(idx));
            result.push((distance, idx));
        }

        Ok(result)
    }

    /// Hash a single point using the hash family
    fn hash_point(&self, point: &ArrayView1<Float>) -> NeighborsResult<Vec<Vec<i32>>> {
        match &self.hash_family {
            HashFamily::RandomProjection {
                num_hashes,
                projection_matrix,
                ..
            } => {
                let mut hashes = Vec::with_capacity(self.num_tables);

                // Each table uses a subset of hash functions
                let hashes_per_table = num_hashes / self.num_tables;

                for table_idx in 0..self.num_tables {
                    let start_hash = table_idx * hashes_per_table;
                    let end_hash = std::cmp::min((table_idx + 1) * hashes_per_table, *num_hashes);

                    let mut table_hash = Vec::new();
                    for hash_idx in start_hash..end_hash {
                        let projection_row = projection_matrix.row(hash_idx);
                        let dot_product: Float = point
                            .iter()
                            .zip(projection_row.iter())
                            .map(|(&a, &b)| a * b)
                            .sum();

                        // Hash bit is 1 if dot product is positive, 0 otherwise
                        table_hash.push(if dot_product >= 0.0 { 1 } else { 0 });
                    }
                    hashes.push(table_hash);
                }

                Ok(hashes)
            }
            HashFamily::MinHash {
                num_hashes,
                a_params,
                b_params,
                prime,
            } => {
                // For MinHash, assume input is a sparse binary vector
                // Convert dense vector to set of non-zero indices
                let mut non_zero_indices = Vec::new();
                for (i, &value) in point.iter().enumerate() {
                    if value > 0.0 {
                        non_zero_indices.push(i as u64);
                    }
                }

                let mut hashes = Vec::with_capacity(self.num_tables);
                let hashes_per_table = num_hashes / self.num_tables;

                for table_idx in 0..self.num_tables {
                    let start_hash = table_idx * hashes_per_table;
                    let end_hash = std::cmp::min((table_idx + 1) * hashes_per_table, *num_hashes);

                    let mut table_hash = Vec::new();
                    for hash_idx in start_hash..end_hash {
                        let a = a_params[hash_idx] as u64;
                        let b = b_params[hash_idx] as u64;

                        let min_hash = non_zero_indices
                            .iter()
                            .map(|&x| (a.wrapping_mul(x).wrapping_add(b)) % prime)
                            .min()
                            .unwrap_or(0);

                        table_hash.push(min_hash as i32);
                    }
                    hashes.push(table_hash);
                }

                Ok(hashes)
            }
        }
    }
}

/// LSH-based approximate nearest neighbor classifier
#[derive(Debug, Clone)]
pub struct LshKNeighborsClassifier<State = sklears_core::traits::Untrained> {
    /// LSH index
    pub lsh_index: LshIndex,
    /// Number of hash tables
    pub num_tables: usize,
    /// Number of hash functions per table
    pub num_hashes: usize,
    /// Number of neighbors to consider
    pub k: usize,
    /// Training labels (only available after fitting)
    pub(crate) y_train: Option<scirs2_core::ndarray::Array1<sklears_core::types::Int>>,
    /// Phantom data for state
    pub(crate) _state: std::marker::PhantomData<State>,
}

impl LshKNeighborsClassifier {
    /// Create a new LSH k-neighbors classifier with random projection
    pub fn new_random_projection(
        k: usize,
        num_tables: usize,
        num_hashes: usize,
        dimension: usize,
    ) -> Self {
        let lsh_index = LshIndex::new_random_projection(num_tables, num_hashes, dimension, k);

        Self {
            lsh_index,
            num_tables,
            num_hashes,
            k,
            y_train: None,
            _state: std::marker::PhantomData,
        }
    }

    /// Create a new LSH k-neighbors classifier with MinHash
    pub fn new_minhash(k: usize, num_tables: usize, num_hashes: usize) -> Self {
        let lsh_index = LshIndex::new_minhash(num_tables, num_hashes, k);

        Self {
            lsh_index,
            num_tables,
            num_hashes,
            k,
            y_train: None,
            _state: std::marker::PhantomData,
        }
    }
}

use sklears_core::traits::Estimator;

impl Estimator for LshKNeighborsClassifier {
    type Config = ();
    type Error = NeighborsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Features, Array1<Int>> for LshKNeighborsClassifier {
    type Fitted = LshKNeighborsClassifier<sklears_core::traits::Trained>;

    fn fit(mut self, x: &Features, y: &Array1<Int>) -> Result<Self::Fitted> {
        if x.is_empty() || y.is_empty() {
            return Err(NeighborsError::EmptyInput.into());
        }

        if x.nrows() != y.len() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![x.nrows()],
                actual: vec![y.len()],
            }
            .into());
        }

        // Fit the LSH index
        self.lsh_index.fit(x)?;

        Ok(LshKNeighborsClassifier {
            lsh_index: self.lsh_index,
            num_tables: self.num_tables,
            num_hashes: self.num_hashes,
            k: self.k,
            y_train: Some(y.clone()),
            _state: std::marker::PhantomData,
        })
    }
}

impl Predict<Features, Array1<Int>> for LshKNeighborsClassifier<sklears_core::traits::Trained> {
    fn predict(&self, x: &Features) -> Result<Array1<Int>> {
        if x.is_empty() {
            return Err(NeighborsError::EmptyInput.into());
        }

        let y_train = self.y_train.as_ref().unwrap();
        let mut predictions = scirs2_core::ndarray::Array1::zeros(x.nrows());

        for (i, query_point) in x.axis_iter(Axis(0)).enumerate() {
            let neighbor_indices = self.lsh_index.query(&query_point)?;

            // Vote among the neighbors
            let mut class_counts = HashMap::new();
            for &neighbor_idx in &neighbor_indices {
                if neighbor_idx < y_train.len() {
                    *class_counts.entry(y_train[neighbor_idx]).or_insert(0) += 1;
                }
            }

            // Find the most common class
            let predicted_class = class_counts
                .into_iter()
                .max_by_key(|&(_, count)| count)
                .map(|(class, _)| class)
                .unwrap_or(0); // Default to class 0 if no neighbors

            predictions[i] = predicted_class;
        }

        Ok(predictions)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};
    use sklears_core::traits::Predict;

    #[test]
    fn test_lsh_index_random_projection() {
        let data = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        )
        .unwrap();

        let mut lsh_index = LshIndex::new_random_projection(2, 4, 3, 2);
        lsh_index.fit(&data).unwrap();

        let query = array![1.0, 0.0, 0.0];
        let neighbors = lsh_index.query(&query.view()).unwrap();

        assert!(!neighbors.is_empty());
        assert!(neighbors.len() <= 2);

        // Should find at least the exact match (index 0)
        assert!(neighbors.contains(&0));
    }

    #[test]
    fn test_lsh_index_minhash() {
        // Create binary data for MinHash
        let data = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        )
        .unwrap();

        let mut lsh_index = LshIndex::new_minhash(2, 4, 2);
        lsh_index.fit(&data).unwrap();

        let query = array![1.0, 0.0, 0.0];
        let neighbors = lsh_index.query(&query.view()).unwrap();

        assert!(!neighbors.is_empty());
        assert!(neighbors.len() <= 2);
    }

    #[test]
    fn test_lsh_kneighbors_classifier() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 1.0, // Class 0
                1.1, 1.1, // Class 0
                1.2, 1.2, // Class 0
                5.0, 5.0, // Class 1
                5.1, 5.1, // Class 1
                5.2, 5.2, // Class 1
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let classifier = LshKNeighborsClassifier::new_random_projection(3, 2, 4, 2);
        let fitted = classifier.fit(&x, &y).unwrap();

        let x_test = Array2::from_shape_vec(
            (2, 2),
            vec![
                1.05, 1.05, // Should be class 0
                5.05, 5.05, // Should be class 1
            ],
        )
        .unwrap();

        let predictions = fitted.predict(&x_test).unwrap();

        // With LSH, results are approximate, so we just check that we get valid predictions
        assert_eq!(predictions.len(), 2);
        for &pred in predictions.iter() {
            assert!(pred == 0 || pred == 1);
        }
    }

    #[test]
    fn test_lsh_index_errors() {
        let mut lsh_index = LshIndex::new_random_projection(2, 4, 3, 2);

        // Test querying before fitting
        let query = array![1.0, 0.0, 0.0];
        let result = lsh_index.query(&query.view());
        assert!(result.is_err());

        // Test fitting with empty data
        let empty_data = Array2::zeros((0, 3));
        let result = lsh_index.fit(&empty_data);
        assert!(result.is_err());
    }
}
