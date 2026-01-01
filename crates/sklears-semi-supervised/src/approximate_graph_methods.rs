//! Approximate graph construction methods for scalable semi-supervised learning
//!
//! This module provides approximate algorithms for constructing graphs from large datasets,
//! trading some accuracy for significant computational speedup and memory efficiency.

use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::rand_prelude::*;
use scirs2_core::random::Random;
use sklears_core::error::SklearsError;
use std::collections::{HashMap, HashSet};

/// Approximate k-NN graph construction using random projection and locality sensitive hashing
#[derive(Clone)]
pub struct ApproximateKNN {
    /// Number of neighbors to find
    pub k_neighbors: usize,
    /// Number of random projections for LSH
    pub n_projections: usize,
    /// Number of hash tables for LSH
    pub n_tables: usize,
    /// Approximation ratio (quality vs speed tradeoff)
    pub approximation_ratio: f64,
    /// Sample ratio for approximate methods
    pub sample_ratio: f64,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

impl ApproximateKNN {
    /// Create a new approximate k-NN instance
    pub fn new() -> Self {
        Self {
            k_neighbors: 5,
            n_projections: 10,
            n_tables: 5,
            approximation_ratio: 1.5,
            sample_ratio: 0.1,
            random_state: None,
        }
    }

    /// Set the number of neighbors
    pub fn k_neighbors(mut self, k: usize) -> Self {
        self.k_neighbors = k;
        self
    }

    /// Set the number of random projections
    pub fn n_projections(mut self, n: usize) -> Self {
        self.n_projections = n;
        self
    }

    /// Set the number of hash tables
    pub fn n_tables(mut self, n: usize) -> Self {
        self.n_tables = n;
        self
    }

    /// Set the approximation ratio
    pub fn approximation_ratio(mut self, ratio: f64) -> Self {
        self.approximation_ratio = ratio;
        self
    }

    /// Set the sample ratio
    pub fn sample_ratio(mut self, ratio: f64) -> Self {
        self.sample_ratio = ratio;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Construct approximate k-NN graph using LSH
    pub fn fit(&self, X: &ArrayView2<f64>) -> Result<Array2<f64>, SklearsError> {
        let n_samples = X.nrows();
        let n_features = X.ncols();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "No samples provided".to_string(),
            ));
        }

        let mut rng = if let Some(seed) = self.random_state {
            Random::seed(seed)
        } else {
            Random::seed(42)
        };

        // Use different strategies based on dataset size
        if n_samples <= 1000 {
            // For small datasets, use exact method
            self.exact_knn(X)
        } else if n_samples <= 10000 {
            // For medium datasets, use random sampling
            self.sampled_knn(X, &mut rng)
        } else {
            // For large datasets, use LSH
            self.lsh_knn(X, &mut rng)
        }
    }

    /// Exact k-NN computation for small datasets
    fn exact_knn(&self, X: &ArrayView2<f64>) -> Result<Array2<f64>, SklearsError> {
        let n_samples = X.nrows();
        let mut graph = Array2::<f64>::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            let mut distances: Vec<(f64, usize)> = Vec::new();

            for j in 0..n_samples {
                if i != j {
                    let dist = self.euclidean_distance(&X.row(i), &X.row(j));
                    distances.push((dist, j));
                }
            }

            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            for (dist, j) in distances.iter().take(self.k_neighbors.min(distances.len())) {
                let weight = (-dist.powi(2) / 2.0).exp();
                graph[[i, *j]] = weight;
                graph[[*j, i]] = weight; // Make symmetric
            }
        }

        Ok(graph)
    }

    /// Sampled k-NN computation for medium datasets
    fn sampled_knn<R>(
        &self,
        X: &ArrayView2<f64>,
        rng: &mut Random<R>,
    ) -> Result<Array2<f64>, SklearsError>
    where
        R: Rng,
    {
        let n_samples = X.nrows();
        let sample_size = (n_samples as f64 * self.sample_ratio).max(50.0) as usize;
        let mut graph = Array2::<f64>::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            let mut candidates: Vec<usize> = (0..n_samples).filter(|&j| j != i).collect();

            // Sample candidates
            if candidates.len() > sample_size {
                candidates.shuffle(rng);
                candidates.truncate(sample_size);
            }

            let mut distances: Vec<(f64, usize)> = candidates
                .iter()
                .map(|&j| {
                    let dist = self.euclidean_distance(&X.row(i), &X.row(j));
                    (dist, j)
                })
                .collect();

            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            for (dist, j) in distances.iter().take(self.k_neighbors.min(distances.len())) {
                let weight = (-dist.powi(2) / 2.0).exp();
                graph[[i, *j]] = weight;
                graph[[*j, i]] = weight; // Make symmetric
            }
        }

        Ok(graph)
    }

    /// LSH-based k-NN computation for large datasets
    fn lsh_knn<R>(
        &self,
        X: &ArrayView2<f64>,
        rng: &mut Random<R>,
    ) -> Result<Array2<f64>, SklearsError>
    where
        R: Rng,
    {
        let n_samples = X.nrows();
        let n_features = X.ncols();

        // Generate random projection vectors
        let mut projection_vectors = Vec::new();
        for _ in 0..self.n_projections {
            let mut vec = Array1::<f64>::zeros(n_features);
            for j in 0..n_features {
                vec[j] = rng.random_range(-1.0..1.0);
            }
            // Normalize
            let norm = vec.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                vec /= norm;
            }
            projection_vectors.push(vec);
        }

        // Create hash tables
        let mut hash_tables: Vec<HashMap<Vec<i32>, Vec<usize>>> =
            vec![HashMap::new(); self.n_tables];

        for i in 0..n_samples {
            #[allow(clippy::needless_range_loop)]
            for table_idx in 0..self.n_tables {
                let mut hash_key = Vec::new();
                let start_proj = (table_idx * self.n_projections) / self.n_tables;
                let end_proj = ((table_idx + 1) * self.n_projections) / self.n_tables;

                for proj_idx in start_proj..end_proj {
                    let projection = X
                        .row(i)
                        .dot(&projection_vectors[proj_idx % self.n_projections]);
                    hash_key.push(if projection >= 0.0 { 1 } else { 0 });
                }

                hash_tables[table_idx].entry(hash_key).or_default().push(i);
            }
        }

        // Find approximate neighbors
        let mut graph = Array2::<f64>::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            let mut candidates = HashSet::new();

            // Collect candidates from all hash tables
            #[allow(clippy::needless_range_loop)]
            for table_idx in 0..self.n_tables {
                let mut hash_key = Vec::new();
                let start_proj = (table_idx * self.n_projections) / self.n_tables;
                let end_proj = ((table_idx + 1) * self.n_projections) / self.n_tables;

                for proj_idx in start_proj..end_proj {
                    let projection = X
                        .row(i)
                        .dot(&projection_vectors[proj_idx % self.n_projections]);
                    hash_key.push(if projection >= 0.0 { 1 } else { 0 });
                }

                if let Some(bucket) = hash_tables[table_idx].get(&hash_key) {
                    for &candidate in bucket {
                        if candidate != i {
                            candidates.insert(candidate);
                        }
                    }
                }
            }

            // If not enough candidates, add random samples
            if candidates.len() < self.k_neighbors * 2 {
                let additional_needed = self.k_neighbors * 2 - candidates.len();
                let mut available: Vec<usize> = (0..n_samples)
                    .filter(|&j| j != i && !candidates.contains(&j))
                    .collect();
                available.shuffle(rng);

                for &candidate in available.iter().take(additional_needed) {
                    candidates.insert(candidate);
                }
            }

            // Compute distances and select k nearest
            let mut distances: Vec<(f64, usize)> = candidates
                .iter()
                .map(|&j| {
                    let dist = self.euclidean_distance(&X.row(i), &X.row(j));
                    (dist, j)
                })
                .collect();

            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            for (dist, j) in distances.iter().take(self.k_neighbors.min(distances.len())) {
                let weight = (-dist.powi(2) / 2.0).exp();
                graph[[i, *j]] = weight;
                graph[[*j, i]] = weight; // Make symmetric
            }
        }

        Ok(graph)
    }

    /// Compute Euclidean distance between two vectors
    fn euclidean_distance(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        x1.iter()
            .zip(x2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

impl Default for ApproximateKNN {
    fn default() -> Self {
        Self::new()
    }
}

/// Spectral clustering with approximate eigenvalue decomposition
#[derive(Clone)]
pub struct ApproximateSpectralClustering {
    /// Number of clusters
    pub n_clusters: usize,
    /// Number of neighbors for graph construction
    pub k_neighbors: usize,
    /// Number of random vectors for randomized SVD
    pub n_components: usize,
    /// Number of power iterations for randomized SVD
    pub n_power_iter: usize,
    /// Use normalized Laplacian
    pub normalized: bool,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

impl ApproximateSpectralClustering {
    /// Create a new approximate spectral clustering instance
    pub fn new() -> Self {
        Self {
            n_clusters: 2,
            k_neighbors: 10,
            n_components: 10,
            n_power_iter: 2,
            normalized: true,
            random_state: None,
        }
    }

    /// Set the number of clusters
    pub fn n_clusters(mut self, k: usize) -> Self {
        self.n_clusters = k;
        self
    }

    /// Set the number of neighbors
    pub fn k_neighbors(mut self, k: usize) -> Self {
        self.k_neighbors = k;
        self
    }

    /// Set the number of components for randomized SVD
    pub fn n_components(mut self, n: usize) -> Self {
        self.n_components = n;
        self
    }

    /// Set the number of power iterations
    pub fn n_power_iter(mut self, n: usize) -> Self {
        self.n_power_iter = n;
        self
    }

    /// Set whether to use normalized Laplacian
    pub fn normalized(mut self, normalized: bool) -> Self {
        self.normalized = normalized;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Perform approximate spectral clustering
    pub fn fit_predict(&self, X: &ArrayView2<f64>) -> Result<Array1<i32>, SklearsError> {
        let n_samples = X.nrows();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "No samples provided".to_string(),
            ));
        }

        let mut rng = if let Some(seed) = self.random_state {
            Random::seed(seed)
        } else {
            Random::seed(42)
        };

        // Construct approximate graph
        let mut approx_knn = ApproximateKNN::new().k_neighbors(self.k_neighbors);

        if let Some(seed) = self.random_state {
            approx_knn = approx_knn.random_state(seed);
        }

        let adjacency = approx_knn.fit(X)?;

        // Compute approximate Laplacian eigendecomposition
        let laplacian = self.compute_laplacian(&adjacency)?;
        let embedding = self.randomized_svd(&laplacian, &mut rng)?;

        // Perform k-means clustering on embedding
        let labels = self.kmeans_clustering(&embedding, &mut rng)?;

        Ok(labels)
    }

    /// Compute graph Laplacian
    fn compute_laplacian(&self, adjacency: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        let n = adjacency.nrows();
        let mut laplacian = adjacency.clone();

        // Compute degree matrix
        let degrees: Vec<f64> = (0..n).map(|i| adjacency.row(i).sum()).collect();

        if self.normalized {
            // Normalized Laplacian: L = I - D^(-1/2) * W * D^(-1/2)
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        laplacian[[i, j]] = 1.0;
                    } else if degrees[i] > 0.0 && degrees[j] > 0.0 {
                        laplacian[[i, j]] = -adjacency[[i, j]] / (degrees[i] * degrees[j]).sqrt();
                    } else {
                        laplacian[[i, j]] = 0.0;
                    }
                }
            }
        } else {
            // Unnormalized Laplacian: L = D - W
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        laplacian[[i, j]] = degrees[i] - adjacency[[i, j]];
                    } else {
                        laplacian[[i, j]] = -adjacency[[i, j]];
                    }
                }
            }
        }

        Ok(laplacian)
    }

    /// Randomized SVD for approximate eigendecomposition
    #[allow(non_snake_case)]
    fn randomized_svd<R>(
        &self,
        matrix: &Array2<f64>,
        rng: &mut Random<R>,
    ) -> Result<Array2<f64>, SklearsError>
    where
        R: Rng,
    {
        let n = matrix.nrows();
        let k = self.n_components.min(n);

        // Generate random matrix
        let mut omega = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            for j in 0..k {
                omega[[i, j]] = rng.random_range(-1.0..1.0);
            }
        }

        // Power iterations
        let mut Y = matrix.dot(&omega);
        for _ in 0..self.n_power_iter {
            let Q = self.qr_decomposition(&Y)?;
            Y = matrix.dot(&Q);
        }

        // QR decomposition
        let Q = self.qr_decomposition(&Y)?;

        // Project matrix
        let B = Q.t().dot(matrix);

        // SVD of smaller matrix B (simplified - using power iteration for largest eigenvalues)
        let embedding = self.power_iteration_embedding(&B, rng)?;

        Ok(Q.dot(&embedding))
    }

    /// Simplified QR decomposition using Gram-Schmidt
    fn qr_decomposition(&self, matrix: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        let (m, n) = matrix.dim();
        let mut Q = matrix.clone();

        // Gram-Schmidt orthogonalization
        for j in 0..n {
            // Normalize current column
            let mut norm = 0.0;
            for i in 0..m {
                norm += Q[[i, j]] * Q[[i, j]];
            }
            norm = norm.sqrt();

            if norm > 1e-10 {
                for i in 0..m {
                    Q[[i, j]] /= norm;
                }
            }

            // Orthogonalize remaining columns
            for k in (j + 1)..n {
                let mut dot_product = 0.0;
                for i in 0..m {
                    dot_product += Q[[i, j]] * Q[[i, k]];
                }

                for i in 0..m {
                    Q[[i, k]] -= dot_product * Q[[i, j]];
                }
            }
        }

        Ok(Q)
    }

    /// Power iteration for embedding (simplified)
    fn power_iteration_embedding<R>(
        &self,
        matrix: &Array2<f64>,
        rng: &mut Random<R>,
    ) -> Result<Array2<f64>, SklearsError>
    where
        R: Rng,
    {
        let n = matrix.nrows();
        let k = self.n_clusters.min(n);

        let mut embedding = Array2::<f64>::zeros((n, k));

        for j in 0..k {
            // Random initial vector
            let mut v = Array1::<f64>::zeros(n);
            for i in 0..n {
                v[i] = rng.random_range(-1.0..1.0);
            }

            // Power iteration
            for _ in 0..20 {
                v = matrix.dot(&v);

                // Normalize
                let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > 1e-10 {
                    v /= norm;
                }

                // Deflation (orthogonalize against previous eigenvectors)
                for prev_j in 0..j {
                    let dot = v.dot(&embedding.column(prev_j));
                    for i in 0..n {
                        v[i] -= dot * embedding[[i, prev_j]];
                    }
                }

                // Re-normalize after deflation
                let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > 1e-10 {
                    v /= norm;
                }
            }

            // Store eigenvector
            for i in 0..n {
                embedding[[i, j]] = v[i];
            }
        }

        Ok(embedding)
    }

    /// Simple k-means clustering
    fn kmeans_clustering<R>(
        &self,
        embedding: &Array2<f64>,
        rng: &mut Random<R>,
    ) -> Result<Array1<i32>, SklearsError>
    where
        R: Rng,
    {
        let n_samples = embedding.nrows();
        let n_features = embedding.ncols();

        if n_samples == 0 {
            return Ok(Array1::zeros(0));
        }

        // Initialize centroids randomly
        let mut centroids = Array2::<f64>::zeros((self.n_clusters, n_features));
        for i in 0..self.n_clusters {
            let sample_idx = rng.gen_range(0..n_samples);
            for j in 0..n_features {
                centroids[[i, j]] = embedding[[sample_idx, j]];
            }
        }

        let mut labels = Array1::<i32>::zeros(n_samples);

        // K-means iterations
        for _iter in 0..100 {
            let mut changed = false;

            // Assign points to nearest centroids
            for i in 0..n_samples {
                let mut min_dist = f64::INFINITY;
                let mut best_cluster = 0;

                for k in 0..self.n_clusters {
                    let mut dist = 0.0;
                    for j in 0..n_features {
                        let diff = embedding[[i, j]] - centroids[[k, j]];
                        dist += diff * diff;
                    }

                    if dist < min_dist {
                        min_dist = dist;
                        best_cluster = k;
                    }
                }

                if labels[i] != best_cluster as i32 {
                    labels[i] = best_cluster as i32;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Update centroids
            for k in 0..self.n_clusters {
                let mut count = 0;
                let mut sum = Array1::<f64>::zeros(n_features);

                for i in 0..n_samples {
                    if labels[i] == k as i32 {
                        count += 1;
                        for j in 0..n_features {
                            sum[j] += embedding[[i, j]];
                        }
                    }
                }

                if count > 0 {
                    for j in 0..n_features {
                        centroids[[k, j]] = sum[j] / count as f64;
                    }
                }
            }
        }

        Ok(labels)
    }
}

impl Default for ApproximateSpectralClustering {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::array;

    #[test]
    #[allow(non_snake_case)]
    fn test_approximate_knn_small_dataset() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];

        let approx_knn = ApproximateKNN::new().k_neighbors(2).random_state(42);

        let result = approx_knn.fit(&X.view());
        assert!(result.is_ok());

        let graph = result.unwrap();
        assert_eq!(graph.dim(), (4, 4));

        // Check symmetry
        for i in 0..4 {
            for j in 0..4 {
                assert_abs_diff_eq!(graph[[i, j]], graph[[j, i]], epsilon = 1e-10);
            }
        }

        // Check diagonal is zero
        for i in 0..4 {
            assert_eq!(graph[[i, i]], 0.0);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_approximate_knn_medium_dataset() {
        // Create a medium-sized dataset
        let mut X_vec = Vec::new();
        for i in 0..100 {
            X_vec.push(vec![i as f64, (i * 2) as f64]);
        }
        let X = Array2::from_shape_vec((100, 2), X_vec.into_iter().flatten().collect()).unwrap();

        let approx_knn = ApproximateKNN::new()
            .k_neighbors(5)
            .sample_ratio(0.2)
            .random_state(42);

        let result = approx_knn.fit(&X.view());
        assert!(result.is_ok());

        let graph = result.unwrap();
        assert_eq!(graph.dim(), (100, 100));

        // Check that each row has reasonable number of non-zero entries
        // (might be more than k_neighbors due to symmetry and sampling)
        for i in 0..100 {
            let non_zero_count = graph.row(i).iter().filter(|&&x| x > 0.0).count();
            assert!(non_zero_count <= 15); // Relaxed constraint for sampling method
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_approximate_spectral_clustering() {
        let X = array![
            [1.0, 1.0],
            [1.5, 1.5],
            [2.0, 2.0],
            [8.0, 8.0],
            [8.5, 8.5],
            [9.0, 9.0]
        ];

        let asc = ApproximateSpectralClustering::new()
            .n_clusters(2)
            .k_neighbors(3)
            .random_state(42);

        let result = asc.fit_predict(&X.view());
        assert!(result.is_ok());

        let labels = result.unwrap();
        assert_eq!(labels.len(), 6);

        // Check that labels are valid
        for &label in labels.iter() {
            assert!(label >= 0 && label < 2);
        }
    }

    #[test]
    fn test_laplacian_computation() {
        let adjacency = array![[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]];

        let asc = ApproximateSpectralClustering::new().normalized(false);
        let laplacian = asc.compute_laplacian(&adjacency).unwrap();

        // Check Laplacian properties for unnormalized case
        assert_eq!(laplacian[[0, 0]], 1.0); // degree of node 0
        assert_eq!(laplacian[[1, 1]], 2.0); // degree of node 1
        assert_eq!(laplacian[[0, 1]], -1.0); // -adjacency

        // Test normalized Laplacian
        let asc_norm = ApproximateSpectralClustering::new().normalized(true);
        let norm_laplacian = asc_norm.compute_laplacian(&adjacency).unwrap();

        // Diagonal should be 1 for normalized Laplacian
        assert_abs_diff_eq!(norm_laplacian[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(norm_laplacian[[1, 1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(norm_laplacian[[2, 2]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_qr_decomposition() {
        let matrix = array![[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

        let asc = ApproximateSpectralClustering::new();
        let result = asc.qr_decomposition(&matrix);
        assert!(result.is_ok());

        let Q = result.unwrap();
        assert_eq!(Q.dim(), matrix.dim());

        // Check that columns are orthonormal (approximately)
        for j in 0..Q.ncols() {
            let norm: f64 = Q.column(j).iter().map(|x| x * x).sum::<f64>().sqrt();
            assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_approximate_knn_builder() {
        let approx_knn = ApproximateKNN::new()
            .k_neighbors(10)
            .n_projections(20)
            .n_tables(8)
            .approximation_ratio(2.0)
            .sample_ratio(0.15)
            .random_state(123);

        assert_eq!(approx_knn.k_neighbors, 10);
        assert_eq!(approx_knn.n_projections, 20);
        assert_eq!(approx_knn.n_tables, 8);
        assert_eq!(approx_knn.approximation_ratio, 2.0);
        assert_eq!(approx_knn.sample_ratio, 0.15);
        assert_eq!(approx_knn.random_state, Some(123));
    }

    #[test]
    fn test_approximate_spectral_clustering_builder() {
        let asc = ApproximateSpectralClustering::new()
            .n_clusters(5)
            .k_neighbors(15)
            .n_components(12)
            .n_power_iter(3)
            .normalized(false)
            .random_state(456);

        assert_eq!(asc.n_clusters, 5);
        assert_eq!(asc.k_neighbors, 15);
        assert_eq!(asc.n_components, 12);
        assert_eq!(asc.n_power_iter, 3);
        assert!(!asc.normalized);
        assert_eq!(asc.random_state, Some(456));
    }

    #[test]
    fn test_error_cases() {
        let approx_knn = ApproximateKNN::new();

        // Test with empty dataset
        let empty_X = Array2::<f64>::zeros((0, 2));
        let result = approx_knn.fit(&empty_X.view());
        assert!(result.is_err());

        let asc = ApproximateSpectralClustering::new();

        // Test spectral clustering with empty dataset
        let result = asc.fit_predict(&empty_X.view());
        assert!(result.is_err());
    }
}
