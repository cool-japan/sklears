//! Approximate distance computations for faster neighbor search
//!
//! This module provides various techniques for computing approximate distances
//! that are much faster than exact distances while maintaining reasonable accuracy.

use crate::{NeighborsError, NeighborsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use scirs2_core::rand_prelude::SliceRandom;
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::thread_rng;
use scirs2_core::random::{Rng, SeedableRng};
use sklears_core::types::Float;
use std::collections::HashMap;

/// Approximate distance computation methods
#[derive(Debug, Clone)]
pub enum ApproximateDistance {
    /// Random projection for approximate Euclidean distance
    RandomProjection {
        /// Number of random projections
        n_projections: usize,
        /// Random projection matrix
        projection_matrix: Array2<Float>,
        /// Original dimensionality
        original_dim: usize,
    },
    /// Product quantization for approximate distances
    ProductQuantization {
        /// Number of subspaces
        n_subspaces: usize,
        /// Number of centroids per subspace
        n_centroids: usize,
        /// Codebooks for each subspace
        codebooks: Vec<Array2<Float>>,
        /// Subspace assignments for dimensions
        subspace_dims: Vec<Vec<usize>>,
    },
    /// LSH families for approximate distance/similarity
    LSHFamily {
        /// Number of hash functions
        n_hash_functions: usize,
        /// Hash function parameters for random projections
        hash_functions: Vec<(Array1<Float>, Float)>,
        /// Hash table for quick lookup
        hash_table: HashMap<Vec<i32>, Vec<usize>>,
    },
}

impl ApproximateDistance {
    /// Create a random projection approximation
    pub fn random_projection(
        n_projections: usize,
        original_dim: usize,
        seed: Option<u64>,
    ) -> NeighborsResult<Self> {
        if n_projections == 0 || original_dim == 0 {
            return Err(NeighborsError::InvalidInput(
                "Dimensions must be positive".to_string(),
            ));
        }

        let rng_seed = seed.unwrap_or_else(|| thread_rng().gen_range(0..u64::MAX));
        let mut rng = StdRng::seed_from_u64(rng_seed);

        // Johnson-Lindenstrauss lemma: we can project to O(log n / ε²) dimensions
        // Create random Gaussian matrix for projection
        let mut projection_matrix = Array2::zeros((n_projections, original_dim));
        let normal = Normal::new(0.0, 1.0).unwrap();
        for i in 0..n_projections {
            for j in 0..original_dim {
                projection_matrix[[i, j]] = rng.sample(normal);
            }
        }

        // Normalize projection vectors
        for i in 0..n_projections {
            let norm = projection_matrix.row(i).mapv(|x: f64| x * x).sum().sqrt();
            if norm > 0.0 {
                for j in 0..original_dim {
                    projection_matrix[[i, j]] /= norm;
                }
            }
        }

        Ok(ApproximateDistance::RandomProjection {
            n_projections,
            projection_matrix,
            original_dim,
        })
    }

    /// Create a product quantization approximation
    pub fn product_quantization(
        n_subspaces: usize,
        n_centroids: usize,
        training_data: &Array2<Float>,
        seed: Option<u64>,
    ) -> NeighborsResult<Self> {
        if n_subspaces == 0 || n_centroids == 0 {
            return Err(NeighborsError::InvalidInput(
                "Number of subspaces and centroids must be positive".to_string(),
            ));
        }

        if training_data.is_empty() {
            return Err(NeighborsError::EmptyInput);
        }

        let original_dim = training_data.ncols();
        if original_dim < n_subspaces {
            return Err(NeighborsError::InvalidInput(
                "Number of subspaces cannot exceed dimensionality".to_string(),
            ));
        }

        let rng_seed = seed.unwrap_or_else(|| thread_rng().gen_range(0..u64::MAX));
        let mut rng = StdRng::seed_from_u64(rng_seed);

        // Divide dimensions into subspaces
        let subspace_size = original_dim / n_subspaces;
        let mut subspace_dims: Vec<Vec<usize>> = Vec::new();

        for i in 0..n_subspaces {
            let start = i * subspace_size;
            let end = if i == n_subspaces - 1 {
                original_dim // Last subspace gets remaining dimensions
            } else {
                (i + 1) * subspace_size
            };
            subspace_dims.push((start..end).collect::<Vec<usize>>());
        }

        // Train codebooks for each subspace using k-means
        let mut codebooks = Vec::new();
        for dims in &subspace_dims {
            let subspace_data = Self::extract_subspace(training_data, dims);
            let codebook = Self::kmeans_codebook(&subspace_data, n_centroids, &mut rng)?;
            codebooks.push(codebook);
        }

        Ok(ApproximateDistance::ProductQuantization {
            n_subspaces,
            n_centroids,
            codebooks,
            subspace_dims,
        })
    }

    /// Create an LSH family for approximate similarity search
    pub fn lsh_family(
        n_hash_functions: usize,
        original_dim: usize,
        seed: Option<u64>,
    ) -> NeighborsResult<Self> {
        if n_hash_functions == 0 || original_dim == 0 {
            return Err(NeighborsError::InvalidInput(
                "Dimensions must be positive".to_string(),
            ));
        }

        let rng_seed = seed.unwrap_or_else(|| thread_rng().gen_range(0..u64::MAX));
        let mut rng = StdRng::seed_from_u64(rng_seed);

        // Create random hash functions for LSH
        let mut hash_functions = Vec::new();
        let normal = Normal::new(0.0, 1.0).unwrap();
        for _ in 0..n_hash_functions {
            let mut random_vector = Array1::zeros(original_dim);
            for j in 0..original_dim {
                random_vector[j] = rng.sample(normal);
            }
            let bias = rng.gen_range(0.0..1.0);
            hash_functions.push((random_vector, bias));
        }

        Ok(ApproximateDistance::LSHFamily {
            n_hash_functions,
            hash_functions,
            hash_table: HashMap::new(),
        })
    }

    /// Compute approximate distance between two points
    pub fn approximate_distance(
        &self,
        p1: &ArrayView1<Float>,
        p2: &ArrayView1<Float>,
    ) -> NeighborsResult<Float> {
        match self {
            ApproximateDistance::RandomProjection {
                projection_matrix, ..
            } => {
                // Project both points and compute distance in projected space
                let proj1 = projection_matrix.dot(p1);
                let proj2 = projection_matrix.dot(p2);
                let diff = &proj1 - &proj2;
                Ok(diff.mapv(|x| x * x).sum().sqrt())
            }
            ApproximateDistance::ProductQuantization {
                codebooks,
                subspace_dims,
                ..
            } => {
                let mut total_distance = 0.0;

                for (i, dims) in subspace_dims.iter().enumerate() {
                    let sub1 = Self::extract_subvector(p1, dims);
                    let sub2 = Self::extract_subvector(p2, dims);

                    // Find nearest centroids in codebook
                    let centroid1_idx = Self::find_nearest_centroid(&sub1, &codebooks[i])?;
                    let centroid2_idx = Self::find_nearest_centroid(&sub2, &codebooks[i])?;

                    // Use precomputed distance between centroids (for now, compute on the fly)
                    let centroid1 = codebooks[i].row(centroid1_idx);
                    let centroid2 = codebooks[i].row(centroid2_idx);
                    let diff = &centroid1 - &centroid2;
                    total_distance += diff.mapv(|x| x * x).sum();
                }

                Ok(total_distance.sqrt())
            }
            ApproximateDistance::LSHFamily { .. } => {
                // LSH is used for finding similar items, not distance computation
                // Fall back to exact Euclidean distance
                let diff = p1 - p2;
                Ok(diff.mapv(|x| x * x).sum().sqrt())
            }
        }
    }

    /// Build hash table for LSH (for training data)
    pub fn build_hash_table(&mut self, training_data: &Array2<Float>) -> NeighborsResult<()> {
        if let ApproximateDistance::LSHFamily {
            hash_functions,
            hash_table,
            ..
        } = self
        {
            // Clone hash functions to avoid borrowing conflicts
            let hash_funcs = hash_functions.clone();
            hash_table.clear();

            for (point_idx, point) in training_data.axis_iter(Axis(0)).enumerate() {
                // Compute hash values using cloned hash functions
                let hash_values: Vec<i32> = hash_funcs
                    .iter()
                    .map(|func| {
                        let dot_product = point.dot(&func.0.view());
                        if dot_product >= 0.0 {
                            1
                        } else {
                            0
                        }
                    })
                    .collect();

                hash_table
                    .entry(hash_values)
                    .or_insert_with(Vec::new)
                    .push(point_idx);
            }
        }
        Ok(())
    }

    /// Compute LSH hash for a point
    pub fn compute_hash(&self, point: &ArrayView1<Float>) -> NeighborsResult<Vec<i32>> {
        if let ApproximateDistance::LSHFamily { hash_functions, .. } = self {
            let mut hash_values = Vec::new();

            for (random_vector, bias) in hash_functions {
                let dot_product = random_vector.dot(point);
                let hash_value = if dot_product + bias >= 0.0 { 1 } else { 0 };
                hash_values.push(hash_value);
            }

            Ok(hash_values)
        } else {
            Err(NeighborsError::InvalidInput(
                "Hash computation only available for LSH family".to_string(),
            ))
        }
    }

    /// Find approximate nearest neighbors using LSH
    pub fn lsh_candidates(&self, query: &ArrayView1<Float>) -> NeighborsResult<Vec<usize>> {
        if let ApproximateDistance::LSHFamily { hash_table, .. } = self {
            let hash_values = self.compute_hash(query)?;

            if let Some(candidates) = hash_table.get(&hash_values) {
                Ok(candidates.clone())
            } else {
                Ok(Vec::new())
            }
        } else {
            Err(NeighborsError::InvalidInput(
                "LSH candidates only available for LSH family".to_string(),
            ))
        }
    }

    // Helper methods
    fn extract_subspace(data: &Array2<Float>, dims: &[usize]) -> Array2<Float> {
        let n_samples = data.nrows();
        let n_dims = dims.len();
        let mut subspace = Array2::zeros((n_samples, n_dims));

        for (i, sample) in data.axis_iter(Axis(0)).enumerate() {
            for (j, &dim) in dims.iter().enumerate() {
                subspace[[i, j]] = sample[dim];
            }
        }

        subspace
    }

    fn extract_subvector(vector: &ArrayView1<Float>, dims: &[usize]) -> Array1<Float> {
        let mut subvector = Array1::zeros(dims.len());
        for (i, &dim) in dims.iter().enumerate() {
            subvector[i] = vector[dim];
        }
        subvector
    }

    fn kmeans_codebook(
        data: &Array2<Float>,
        k: usize,
        rng: &mut StdRng,
    ) -> NeighborsResult<Array2<Float>> {
        if data.is_empty() || k == 0 {
            return Err(NeighborsError::InvalidInput(
                "Invalid input for k-means".to_string(),
            ));
        }

        let n_samples = data.nrows();
        let n_dims = data.ncols();
        let k = k.min(n_samples); // Can't have more centroids than samples

        // Initialize centroids randomly
        let mut centroids = Array2::zeros((k, n_dims));
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(rng);

        for (i, &idx) in indices.iter().enumerate().take(k) {
            centroids.row_mut(i).assign(&data.row(idx));
        }

        // Simple k-means iterations
        for _ in 0..10 {
            // Max 10 iterations for efficiency
            let mut assignments = vec![0; n_samples];

            // Assign each point to nearest centroid
            for (i, sample) in data.axis_iter(Axis(0)).enumerate() {
                let mut min_dist = Float::INFINITY;
                let mut best_centroid = 0;

                for (j, centroid) in centroids.axis_iter(Axis(0)).enumerate() {
                    let diff = &sample - &centroid;
                    let dist = diff.mapv(|x| x * x).sum();
                    if dist < min_dist {
                        min_dist = dist;
                        best_centroid = j;
                    }
                }
                assignments[i] = best_centroid;
            }

            // Update centroids
            let mut new_centroids = Array2::zeros((k, n_dims));
            let mut counts = vec![0; k];

            for (i, sample) in data.axis_iter(Axis(0)).enumerate() {
                let cluster = assignments[i];
                new_centroids.row_mut(cluster).scaled_add(1.0, &sample);
                counts[cluster] += 1;
            }

            for (j, &count) in counts.iter().enumerate().take(k) {
                if count > 0 {
                    new_centroids
                        .row_mut(j)
                        .mapv_inplace(|x| x / count as Float);
                }
            }

            centroids = new_centroids;
        }

        Ok(centroids)
    }

    fn find_nearest_centroid(
        point: &Array1<Float>,
        codebook: &Array2<Float>,
    ) -> NeighborsResult<usize> {
        if codebook.is_empty() {
            return Err(NeighborsError::InvalidInput("Empty codebook".to_string()));
        }

        let mut min_dist = Float::INFINITY;
        let mut best_idx = 0;

        for (i, centroid) in codebook.axis_iter(Axis(0)).enumerate() {
            let diff = point - &centroid;
            let dist = diff.mapv(|x| x * x).sum();
            if dist < min_dist {
                min_dist = dist;
                best_idx = i;
            }
        }

        Ok(best_idx)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_random_projection_creation() {
        let approx_dist = ApproximateDistance::random_projection(10, 50, Some(42)).unwrap();

        if let ApproximateDistance::RandomProjection {
            n_projections,
            original_dim,
            ..
        } = approx_dist
        {
            assert_eq!(n_projections, 10);
            assert_eq!(original_dim, 50);
        } else {
            panic!("Expected RandomProjection variant");
        }
    }

    #[test]
    fn test_random_projection_distance() {
        let approx_dist = ApproximateDistance::random_projection(5, 3, Some(42)).unwrap();

        let p1 = array![1.0, 2.0, 3.0];
        let p2 = array![4.0, 5.0, 6.0];

        let approx_distance = approx_dist
            .approximate_distance(&p1.view(), &p2.view())
            .unwrap();
        assert!(approx_distance >= 0.0);
        assert!(approx_distance.is_finite());
    }

    #[test]
    fn test_product_quantization_creation() {
        let data = Array2::from_shape_vec((10, 4), (0..40).map(|x| x as Float).collect()).unwrap();
        let approx_dist = ApproximateDistance::product_quantization(2, 3, &data, Some(42)).unwrap();

        if let ApproximateDistance::ProductQuantization {
            n_subspaces,
            n_centroids,
            codebooks,
            ..
        } = approx_dist
        {
            assert_eq!(n_subspaces, 2);
            assert_eq!(n_centroids, 3);
            assert_eq!(codebooks.len(), 2);
        } else {
            panic!("Expected ProductQuantization variant");
        }
    }

    #[test]
    fn test_lsh_family_creation() {
        let approx_dist = ApproximateDistance::lsh_family(8, 10, Some(42)).unwrap();

        if let ApproximateDistance::LSHFamily {
            n_hash_functions,
            hash_functions,
            ..
        } = approx_dist
        {
            assert_eq!(n_hash_functions, 8);
            assert_eq!(hash_functions.len(), 8);
        } else {
            panic!("Expected LSHFamily variant");
        }
    }

    #[test]
    fn test_lsh_hash_computation() {
        let approx_dist = ApproximateDistance::lsh_family(4, 3, Some(42)).unwrap();

        let point = array![1.0, -2.0, 3.0];
        let hash_values = approx_dist.compute_hash(&point.view()).unwrap();

        assert_eq!(hash_values.len(), 4);
        for &hash_val in &hash_values {
            assert!(hash_val == 0 || hash_val == 1);
        }
    }

    #[test]
    fn test_lsh_hash_table() {
        let mut approx_dist = ApproximateDistance::lsh_family(4, 2, Some(42)).unwrap();

        let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        approx_dist.build_hash_table(&data).unwrap();

        let query = array![1.5, 2.5];
        let candidates = approx_dist.lsh_candidates(&query.view()).unwrap();

        // Should return some candidates (or empty if no hash collisions)
        assert!(candidates.len() <= 3);
    }

    #[test]
    fn test_invalid_inputs() {
        // Invalid random projection
        let result = ApproximateDistance::random_projection(0, 10, None);
        assert!(result.is_err());

        // Invalid product quantization
        let empty_data = Array2::zeros((0, 0));
        let result = ApproximateDistance::product_quantization(2, 3, &empty_data, None);
        assert!(result.is_err());

        // Invalid LSH family
        let result = ApproximateDistance::lsh_family(0, 10, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_approximate_vs_exact_distance() {
        // Test that approximate distance is reasonably close to exact distance
        let approx_dist = ApproximateDistance::random_projection(20, 5, Some(42)).unwrap();

        let p1 = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let p2 = array![2.0, 3.0, 4.0, 5.0, 6.0];

        let exact_dist = {
            let diff = &p1 - &p2;
            (diff.mapv(|x| x * x).sum() as Float).sqrt()
        };

        let approx_dist_val = approx_dist
            .approximate_distance(&p1.view(), &p2.view())
            .unwrap();

        // With enough projections, approximate distance should be reasonably close
        // This is a loose test - in practice you'd want more rigorous validation
        assert!(approx_dist_val > 0.0);
        assert!((approx_dist_val - exact_dist).abs() < exact_dist * 2.0); // Within 200%
    }
}
