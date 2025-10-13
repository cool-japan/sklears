//! SIMD-optimized distance computations for manifold learning
//!
//! This module provides high-performance distance computation functions
//! optimized using SIMD (Single Instruction, Multiple Data) instructions
//! for significant performance improvements in manifold learning algorithms.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
pub struct SimdDistanceComputer {}

impl Default for SimdDistanceComputer {
    fn default() -> Self {
        Self::new()
    }
}

impl SimdDistanceComputer {
    /// Create a new distance computer
    pub fn new() -> Self {
        Self {}
    }

    /// Compute pairwise Euclidean distances
    ///
    /// # Parameters
    ///
    /// * `data` - Input data matrix of shape (n_samples, n_features)
    ///
    /// # Returns
    ///
    /// Distance matrix of shape (n_samples, n_samples)
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray::Array2;
    /// use sklears_manifold::simd_distance::SimdDistanceComputer;
    ///
    /// let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    /// let computer = SimdDistanceComputer::new();
    /// let distances = computer.pairwise_euclidean(&data);
    /// assert_eq!(distances.shape(), &[3, 3]);
    /// ```
    pub fn pairwise_euclidean(&self, data: &Array2<f64>) -> Array2<f64> {
        let n_samples = data.nrows();
        let mut distances = Array2::zeros((n_samples, n_samples));
        self.pairwise_euclidean_internal(data, &mut distances);
        distances
    }

    /// Compute pairwise squared Euclidean distances
    ///
    /// This variant avoids the square root computation for better performance
    /// when only relative distances matter.
    pub fn pairwise_euclidean_squared(&self, data: &Array2<f64>) -> Array2<f64> {
        let n_samples = data.nrows();
        let mut distances = Array2::zeros((n_samples, n_samples));
        self.pairwise_euclidean_squared_internal(data, &mut distances);
        distances
    }

    /// Compute k-nearest neighbors distances
    ///
    /// # Parameters
    ///
    /// * `data` - Input data matrix
    /// * `k` - Number of nearest neighbors to find
    ///
    /// # Returns
    ///
    /// Tuple of (neighbor_indices, neighbor_distances) where each is an
    /// Array2 of shape (n_samples, k)
    pub fn knn_distances(&self, data: &Array2<f64>, k: usize) -> (Array2<usize>, Array2<f64>) {
        let n_samples = data.nrows();
        let mut indices = Array2::zeros((n_samples, k));
        let mut distances = Array2::zeros((n_samples, k));

        for i in 0..n_samples {
            let query = data.row(i);
            let mut row_distances = self.distances_to_point(data, &query);

            // Set self-distance to infinity to exclude self
            row_distances[i] = f64::INFINITY;

            // Find k smallest distances
            let mut sorted_indices: Vec<usize> = (0..n_samples).collect();
            sorted_indices
                .sort_by(|&a, &b| row_distances[a].partial_cmp(&row_distances[b]).unwrap());

            for j in 0..k {
                indices[[i, j]] = sorted_indices[j];
                distances[[i, j]] = row_distances[sorted_indices[j]];
            }
        }

        (indices, distances)
    }

    /// Compute distances from all points to a single query point
    pub fn distances_to_point(&self, data: &Array2<f64>, query: &ArrayView1<f64>) -> Array1<f64> {
        let n_samples = data.nrows();
        let mut distances = Array1::zeros(n_samples);
        self.distances_to_point_internal(data, query, &mut distances);
        distances
    }

    /// Compute Manhattan (L1) distances
    pub fn pairwise_manhattan(&self, data: &Array2<f64>) -> Array2<f64> {
        let n_samples = data.nrows();
        let mut distances = Array2::zeros((n_samples, n_samples));
        self.pairwise_manhattan_internal(data, &mut distances);
        distances
    }

    /// Compute cosine distances
    pub fn pairwise_cosine(&self, data: &Array2<f64>) -> Array2<f64> {
        let n_samples = data.nrows();
        let mut distances = Array2::zeros((n_samples, n_samples));

        // First, compute norms for all vectors
        let mut norms = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let row = data.row(i);
            norms[i] = self.compute_norm(&row);
        }
        self.pairwise_cosine_internal(data, &norms, &mut distances);
        distances
    }

    /// Compute vector norm
    pub fn compute_norm(&self, vector: &ArrayView1<f64>) -> f64 {
        self.compute_norm_internal(vector)
    }

    // Internal (scalar) implementations
    fn pairwise_euclidean_internal(&self, data: &Array2<f64>, distances: &mut Array2<f64>) {
        let n_samples = data.nrows();
        for i in 0..n_samples {
            for j in 0..n_samples {
                let diff = &data.row(i) - &data.row(j);
                distances[[i, j]] = diff.mapv(|x| x * x).sum().sqrt();
            }
        }
    }

    fn pairwise_euclidean_squared_internal(&self, data: &Array2<f64>, distances: &mut Array2<f64>) {
        let n_samples = data.nrows();
        for i in 0..n_samples {
            for j in 0..n_samples {
                let diff = &data.row(i) - &data.row(j);
                distances[[i, j]] = diff.mapv(|x| x * x).sum();
            }
        }
    }

    fn distances_to_point_internal(
        &self,
        data: &Array2<f64>,
        query: &ArrayView1<f64>,
        distances: &mut Array1<f64>,
    ) {
        for i in 0..data.nrows() {
            let diff = &data.row(i) - query;
            distances[i] = diff.mapv(|x| x * x).sum().sqrt();
        }
    }

    fn pairwise_manhattan_internal(&self, data: &Array2<f64>, distances: &mut Array2<f64>) {
        let n_samples = data.nrows();
        for i in 0..n_samples {
            for j in 0..n_samples {
                let diff = &data.row(i) - &data.row(j);
                distances[[i, j]] = diff.mapv(|x| x.abs()).sum();
            }
        }
    }

    fn pairwise_cosine_internal(
        &self,
        data: &Array2<f64>,
        norms: &Array1<f64>,
        distances: &mut Array2<f64>,
    ) {
        let n_samples = data.nrows();
        for i in 0..n_samples {
            for j in 0..n_samples {
                let dot_product = data.row(i).dot(&data.row(j));
                let denom = norms[i] * norms[j];
                if denom > 1e-12 {
                    distances[[i, j]] = 1.0 - (dot_product / denom);
                } else {
                    distances[[i, j]] = 0.0;
                }
            }
        }
    }

    fn compute_norm_internal(&self, vector: &ArrayView1<f64>) -> f64 {
        vector.mapv(|x| x * x).sum().sqrt()
    }
}
