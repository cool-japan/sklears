//! Manifold Learning algorithms for non-linear dimensionality reduction
//!
//! This module provides various manifold learning techniques:
//! - Locally Linear Embedding (LLE)
//! - Isomap (Isometric Mapping)
//! - Laplacian Eigenmaps
//! - t-Distributed Stochastic Neighbor Embedding (t-SNE)
//! - Uniform Manifold Approximation and Projection (UMAP)

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use scirs2_core::rand_prelude::SliceRandom;
use scirs2_core::random::{Rng, thread_rng, Random};
use scirs2_linalg::compat::{ArrayLinalgExt, UPLO};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Transform, Untrained},
};

/// Type alias for manifold learning method results
/// Returns: (embedding, optional_weights, optional_distances, iterations)
type ManifoldResult = Result<(Array2<f64>, Option<Array2<f64>>, Option<Array2<f64>>, usize)>;

/// Manifold learning algorithm variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ManifoldAlgorithm {
    /// Locally Linear Embedding
    #[default]
    LLE,
    /// Isomap (Isometric Mapping)
    Isomap,
    /// Laplacian Eigenmaps
    LaplacianEigenmaps,
    /// t-Distributed Stochastic Neighbor Embedding
    TSNE,
    /// Uniform Manifold Approximation and Projection
    UMAP,
}

/// Distance metrics for manifold learning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DistanceMetric {
    /// Euclidean distance
    #[default]
    Euclidean,
    /// Manhattan distance
    Manhattan,
    /// Cosine distance
    Cosine,
}

/// Manifold Learning transformer
#[derive(Debug, Clone)]
pub struct ManifoldLearning<State = Untrained> {
    /// Algorithm to use
    pub algorithm: ManifoldAlgorithm,
    /// Target dimensionality
    pub n_components: usize,
    /// Number of neighbors for local methods
    pub n_neighbors: usize,
    /// Distance metric
    pub metric: DistanceMetric,
    /// Maximum number of iterations (for iterative methods)
    pub max_iter: usize,
    /// Learning rate (for gradient-based methods)
    pub learning_rate: f64,
    /// Perplexity (for t-SNE)
    pub perplexity: f64,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Early exaggeration factor (for t-SNE)
    pub early_exaggeration: f64,
    /// Minimum distance (for UMAP)
    pub min_dist: f64,
    /// Spread (for UMAP)
    pub spread: f64,

    /// Trained state
    state: State,
}

/// Trained manifold learning state
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TrainedManifoldLearning {
    pub embedding: Array2<f64>,
    pub training_data: Array2<f64>,
    pub neighbor_graph: Option<Array2<f64>>,
    pub distance_matrix: Option<Array2<f64>>,
    pub algorithm: ManifoldAlgorithm,
    pub metric: DistanceMetric,
    pub n_features_in: usize,
    pub n_components: usize,
    pub n_iter: usize,
}

impl ManifoldLearning<Untrained> {
    /// Create a new manifold learning transformer
    pub fn new(algorithm: ManifoldAlgorithm, n_components: usize) -> Self {
        Self {
            algorithm,
            n_components,
            n_neighbors: 5,
            metric: DistanceMetric::Euclidean,
            max_iter: 1000,
            learning_rate: 200.0,
            perplexity: 30.0,
            random_state: None,
            early_exaggeration: 12.0,
            min_dist: 0.1,
            spread: 1.0,
            state: Untrained,
        }
    }

    /// Set number of neighbors
    pub fn n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self
    }

    /// Set distance metric
    pub fn metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set perplexity (for t-SNE)
    pub fn perplexity(mut self, perplexity: f64) -> Self {
        self.perplexity = perplexity;
        self
    }

    /// Set random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Set early exaggeration (for t-SNE)
    pub fn early_exaggeration(mut self, early_exaggeration: f64) -> Self {
        self.early_exaggeration = early_exaggeration;
        self
    }

    /// Set minimum distance (for UMAP)
    pub fn min_dist(mut self, min_dist: f64) -> Self {
        self.min_dist = min_dist;
        self
    }

    /// Set spread (for UMAP)
    pub fn spread(mut self, spread: f64) -> Self {
        self.spread = spread;
        self
    }
}

impl Fit<Array2<f64>, ()> for ManifoldLearning<Untrained> {
    type Fitted = ManifoldLearning<TrainedManifoldLearning>;

    fn fit(self, x: &Array2<f64>, _y: &()) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples < self.n_components {
            return Err(SklearsError::InvalidInput(
                "Number of samples must be greater than n_components".to_string(),
            ));
        }

        if self.n_neighbors >= n_samples {
            return Err(SklearsError::InvalidInput(
                "n_neighbors must be less than number of samples".to_string(),
            ));
        }

        let (embedding, neighbor_graph, distance_matrix, n_iter) = match self.algorithm {
            ManifoldAlgorithm::LLE => self.locally_linear_embedding(x)?,
            ManifoldAlgorithm::Isomap => self.isomap(x)?,
            ManifoldAlgorithm::LaplacianEigenmaps => self.laplacian_eigenmaps(x)?,
            ManifoldAlgorithm::TSNE => self.tsne(x)?,
            ManifoldAlgorithm::UMAP => self.umap(x)?,
        };

        Ok(ManifoldLearning {
            algorithm: self.algorithm,
            n_components: self.n_components,
            n_neighbors: self.n_neighbors,
            metric: self.metric,
            max_iter: self.max_iter,
            learning_rate: self.learning_rate,
            perplexity: self.perplexity,
            random_state: self.random_state,
            early_exaggeration: self.early_exaggeration,
            min_dist: self.min_dist,
            spread: self.spread,
            state: TrainedManifoldLearning {
                embedding,
                training_data: x.clone(),
                neighbor_graph,
                distance_matrix,
                algorithm: self.algorithm,
                metric: self.metric,
                n_features_in: n_features,
                n_components: self.n_components,
                n_iter,
            },
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for ManifoldLearning<TrainedManifoldLearning> {
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let (_n_samples, n_features) = x.dim();

        if n_features != self.state.n_features_in {
            return Err(SklearsError::FeatureMismatch {
                expected: self.state.n_features_in,
                actual: n_features,
            });
        }

        // Out-of-sample extension (simplified approach)
        match self.state.algorithm {
            ManifoldAlgorithm::LLE => self.lle_transform(x),
            ManifoldAlgorithm::Isomap => self.isomap_transform(x),
            ManifoldAlgorithm::LaplacianEigenmaps => self.laplacian_transform(x),
            ManifoldAlgorithm::TSNE => {
                // t-SNE doesn't have a natural out-of-sample extension
                Err(SklearsError::InvalidInput(
                    "t-SNE does not support out-of-sample transformation".to_string(),
                ))
            }
            ManifoldAlgorithm::UMAP => self.umap_transform(x),
        }
    }
}

impl ManifoldLearning<Untrained> {
    /// Locally Linear Embedding (LLE)
    fn locally_linear_embedding(&self, x: &Array2<f64>) -> ManifoldResult {
        let (_n_samples, _) = x.dim();

        // Step 1: Find k-nearest neighbors
        let neighbors = self.find_knn(x, self.n_neighbors)?;

        // Step 2: Compute reconstruction weights
        let weights = self.compute_lle_weights(x, &neighbors)?;

        // Step 3: Find low-dimensional embedding
        let embedding = self.compute_lle_embedding(&weights)?;

        Ok((embedding, Some(weights), None, 1))
    }

    /// Isomap algorithm
    fn isomap(&self, x: &Array2<f64>) -> ManifoldResult {
        let (_n_samples, _) = x.dim();

        // Step 1: Build neighborhood graph
        let neighbors = self.find_knn(x, self.n_neighbors)?;
        let distance_matrix = self.compute_distance_matrix(x)?;

        // Step 2: Compute geodesic distances using Floyd-Warshall
        let geodesic_distances = self.compute_geodesic_distances(&distance_matrix, &neighbors)?;

        // Step 3: Apply classical MDS
        let embedding = self.classical_mds(&geodesic_distances)?;

        Ok((embedding, None, Some(geodesic_distances), 1))
    }

    /// Laplacian Eigenmaps
    fn laplacian_eigenmaps(&self, x: &Array2<f64>) -> ManifoldResult {
        let (_n_samples, _) = x.dim();

        // Step 1: Build neighborhood graph
        let neighbors = self.find_knn(x, self.n_neighbors)?;

        // Step 2: Compute weight matrix (using heat kernel)
        let weight_matrix = self.compute_laplacian_weights(x, &neighbors)?;

        // Step 3: Compute Laplacian and solve eigenvalue problem
        let embedding = self.solve_laplacian_eigenproblem(&weight_matrix)?;

        Ok((embedding, Some(weight_matrix), None, 1))
    }

    /// t-SNE algorithm (simplified version)
    fn tsne(&self, x: &Array2<f64>) -> ManifoldResult {
        let (n_samples, _) = x.dim();

        // Initialize random number generator
        // TODO: Support seeding for reproducibility
        let mut rng = thread_rng();

        // Step 1: Compute pairwise similarities in high-dimensional space
        let p_matrix = self.compute_tsne_similarities(x)?;

        // Step 2: Initialize low-dimensional embedding randomly
        let mut embedding = Array2::zeros((n_samples, self.n_components));
        for i in 0..n_samples {
            for j in 0..self.n_components {
                embedding[[i, j]] = rng.gen() * 1e-4;
            }
        }

        // Step 3: Optimize embedding using gradient descent
        let n_iter = self.optimize_tsne_embedding(&mut embedding, &p_matrix)?;

        Ok((embedding, Some(p_matrix), None, n_iter))
    }

    /// UMAP algorithm (simplified version)
    fn umap(&self, x: &Array2<f64>) -> ManifoldResult {
        let (n_samples, _) = x.dim();

        // Initialize random number generator
        // TODO: Support seeding for reproducibility
        let mut rng = thread_rng();

        // Step 1: Build fuzzy topological representation
        let neighbors = self.find_knn(x, self.n_neighbors)?;
        let fuzzy_graph = self.build_fuzzy_simplicial_set(x, &neighbors)?;

        // Step 2: Initialize low-dimensional embedding
        let mut embedding = Array2::zeros((n_samples, self.n_components));
        for i in 0..n_samples {
            for j in 0..self.n_components {
                embedding[[i, j]] = (rng.gen() - 0.5) * 20.0;
            }
        }

        // Step 3: Optimize embedding
        let n_iter = self.optimize_umap_embedding(&mut embedding, &fuzzy_graph)?;

        Ok((embedding, Some(fuzzy_graph), None, n_iter))
    }

    /// Find k-nearest neighbors
    fn find_knn(&self, x: &Array2<f64>, k: usize) -> Result<Array2<usize>> {
        let (n_samples, _) = x.dim();
        let mut neighbors = Array2::zeros((n_samples, k));

        for i in 0..n_samples {
            let mut distances: Vec<(f64, usize)> = Vec::new();

            for j in 0..n_samples {
                if i != j {
                    let distance = self.compute_distance(&x.row(i), &x.row(j));
                    distances.push((distance, j));
                }
            }

            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            for (idx, &(_, neighbor_idx)) in distances.iter().take(k).enumerate() {
                neighbors[[i, idx]] = neighbor_idx;
            }
        }

        Ok(neighbors)
    }

    /// Compute distance between two points
    fn compute_distance(
        &self,
        a: &scirs2_core::ndarray::ArrayView1<f64>,
        b: &scirs2_core::ndarray::ArrayView1<f64>,
    ) -> f64 {
        match self.metric {
            DistanceMetric::Euclidean => a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| (x - y) * (x - y))
                .sum::<f64>()
                .sqrt(),
            DistanceMetric::Manhattan => a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| (x - y).abs())
                .sum::<f64>(),
            DistanceMetric::Cosine => {
                let dot_product = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum::<f64>();
                let norm_a = a.iter().map(|&x| x * x).sum::<f64>().sqrt();
                let norm_b = b.iter().map(|&x| x * x).sum::<f64>().sqrt();

                if norm_a > 1e-12 && norm_b > 1e-12 {
                    1.0 - (dot_product / (norm_a * norm_b))
                } else {
                    0.0
                }
            }
        }
    }

    /// Compute distance matrix
    fn compute_distance_matrix(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let (n_samples, _) = x.dim();
        let mut distance_matrix = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let distance = self.compute_distance(&x.row(i), &x.row(j));
                distance_matrix[[i, j]] = distance;
                distance_matrix[[j, i]] = distance;
            }
        }

        Ok(distance_matrix)
    }

    /// Compute LLE reconstruction weights
    fn compute_lle_weights(
        &self,
        x: &Array2<f64>,
        neighbors: &Array2<usize>,
    ) -> Result<Array2<f64>> {
        let (n_samples, n_features) = x.dim();
        let k = neighbors.ncols();
        let mut weights = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            // Build local covariance matrix
            let mut local_cov = Array2::zeros((k, k));

            for a in 0..k {
                for b in 0..k {
                    let neighbor_a = neighbors[[i, a]];
                    let neighbor_b = neighbors[[i, b]];

                    let mut cov = 0.0;
                    for d in 0..n_features {
                        let diff_a = x[[neighbor_a, d]] - x[[i, d]];
                        let diff_b = x[[neighbor_b, d]] - x[[i, d]];
                        cov += diff_a * diff_b;
                    }
                    local_cov[[a, b]] = cov;
                }
            }

            // Add regularization
            for j in 0..k {
                local_cov[[j, j]] += 1e-3;
            }

            // Solve for weights (least squares solution)
            let ones = Array1::ones(k);
            let weights_local = self.solve_linear_system(&local_cov, &ones)?;

            // Normalize weights
            let weight_sum = weights_local.sum();
            if weight_sum > 1e-12 {
                for (idx, &neighbor_idx) in neighbors.row(i).iter().enumerate() {
                    weights[[i, neighbor_idx]] = weights_local[idx] / weight_sum;
                }
            }
        }

        Ok(weights)
    }

    /// Solve linear system Ax = b
    fn solve_linear_system(&self, a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
        // Use scirs2-linalg solve method
        let solution = a.solve(b).map_err(|e| {
            SklearsError::NumericalError(format!("Failed to solve linear system: {}", e))
        })?;

        Ok(solution)
    }

    /// Compute LLE embedding using eigendecomposition
    fn compute_lle_embedding(&self, weights: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = weights.nrows();

        // Compute M = (I - W)^T (I - W)
        let identity = Array2::eye(n_samples);
        let i_minus_w = &identity - weights;
        let m_matrix = i_minus_w.t().dot(&i_minus_w);

        // Find smallest eigenvalues and eigenvectors
        let eigenresult = self.compute_smallest_eigenvectors(&m_matrix)?;

        Ok(eigenresult)
    }

    /// Compute geodesic distances using Floyd-Warshall algorithm
    fn compute_geodesic_distances(
        &self,
        distance_matrix: &Array2<f64>,
        neighbors: &Array2<usize>,
    ) -> Result<Array2<f64>> {
        let n_samples = distance_matrix.nrows();
        let mut geodesic = Array2::from_elem((n_samples, n_samples), f64::INFINITY);

        // Initialize with direct neighbor distances
        for i in 0..n_samples {
            geodesic[[i, i]] = 0.0;
            for &neighbor in neighbors.row(i) {
                geodesic[[i, neighbor]] = distance_matrix[[i, neighbor]];
                geodesic[[neighbor, i]] = distance_matrix[[neighbor, i]];
            }
        }

        // Floyd-Warshall algorithm
        for k in 0..n_samples {
            for i in 0..n_samples {
                for j in 0..n_samples {
                    let through_k = geodesic[[i, k]] + geodesic[[k, j]];
                    if through_k < geodesic[[i, j]] {
                        geodesic[[i, j]] = through_k;
                    }
                }
            }
        }

        Ok(geodesic)
    }

    /// Classical Multidimensional Scaling (MDS)
    fn classical_mds(&self, distance_matrix: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = distance_matrix.nrows();

        // Convert distances to similarities using double centering
        let mut similarity = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            for j in 0..n_samples {
                similarity[[i, j]] = -0.5 * distance_matrix[[i, j]] * distance_matrix[[i, j]];
            }
        }

        // Double centering
        let row_means = similarity.mean_axis(Axis(1)).unwrap();
        let col_means = similarity.mean_axis(Axis(0)).unwrap();
        let grand_mean = row_means.mean().unwrap();

        for i in 0..n_samples {
            for j in 0..n_samples {
                similarity[[i, j]] = similarity[[i, j]] - row_means[i] - col_means[j] + grand_mean;
            }
        }

        // Eigendecomposition and take top components
        self.compute_largest_eigenvectors(&similarity)
    }

    /// Compute Laplacian weights using heat kernel
    fn compute_laplacian_weights(
        &self,
        x: &Array2<f64>,
        neighbors: &Array2<usize>,
    ) -> Result<Array2<f64>> {
        let (n_samples, _) = x.dim();
        let mut weight_matrix = Array2::zeros((n_samples, n_samples));

        // Estimate sigma parameter
        let sigma = self.estimate_sigma(x, neighbors)?;

        for i in 0..n_samples {
            for &j in neighbors.row(i) {
                if i != j {
                    let distance = self.compute_distance(&x.row(i), &x.row(j));
                    let weight = (-distance * distance / (2.0 * sigma * sigma)).exp();
                    weight_matrix[[i, j]] = weight;
                    weight_matrix[[j, i]] = weight;
                }
            }
        }

        Ok(weight_matrix)
    }

    /// Estimate sigma parameter for heat kernel
    fn estimate_sigma(&self, x: &Array2<f64>, neighbors: &Array2<usize>) -> Result<f64> {
        let (n_samples, _) = x.dim();
        let mut distances = Vec::new();

        for i in 0..n_samples {
            for &j in neighbors.row(i) {
                if i != j {
                    let distance = self.compute_distance(&x.row(i), &x.row(j));
                    distances.push(distance);
                }
            }
        }

        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_distance = distances[distances.len() / 2];

        Ok(median_distance)
    }

    /// Solve Laplacian eigenvalue problem
    fn solve_laplacian_eigenproblem(&self, weight_matrix: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = weight_matrix.nrows();

        // Compute degree matrix
        let mut degree_matrix = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            let degree: f64 = weight_matrix.row(i).sum();
            degree_matrix[[i, i]] = degree;
        }

        // Compute Laplacian L = D - W
        let laplacian = &degree_matrix - weight_matrix;

        // Find smallest eigenvalues (skip the first one which is zero)
        self.compute_smallest_eigenvectors(&laplacian)
    }

    /// Compute t-SNE similarities (P matrix)
    fn compute_tsne_similarities(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let (n_samples, _) = x.dim();
        let mut p_matrix = Array2::zeros((n_samples, n_samples));

        // Compute pairwise conditional probabilities
        for i in 0..n_samples {
            let sigma = self.find_optimal_sigma(x, i)?;

            let mut row_sum = 0.0;
            for j in 0..n_samples {
                if i != j {
                    let distance_sq = self.compute_distance(&x.row(i), &x.row(j)).powi(2);
                    let prob = (-distance_sq / (2.0 * sigma * sigma)).exp();
                    p_matrix[[i, j]] = prob;
                    row_sum += prob;
                }
            }

            // Normalize row
            if row_sum > 1e-12 {
                for j in 0..n_samples {
                    if i != j {
                        p_matrix[[i, j]] /= row_sum;
                    }
                }
            }
        }

        // Symmetrize: p_ij = (p_i|j + p_j|i) / (2n)
        let mut symmetric_p = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in 0..n_samples {
                symmetric_p[[i, j]] =
                    (p_matrix[[i, j]] + p_matrix[[j, i]]) / (2.0 * n_samples as f64);
            }
        }

        Ok(symmetric_p)
    }

    /// Find optimal sigma for t-SNE using binary search
    fn find_optimal_sigma(&self, x: &Array2<f64>, i: usize) -> Result<f64> {
        let target_perplexity = self.perplexity;
        let mut sigma_min = 1e-20;
        let mut sigma_max = 1e20;
        let tolerance = 1e-5;
        let max_iterations = 50;

        for _ in 0..max_iterations {
            let sigma = (sigma_min + sigma_max) / 2.0;
            let perplexity = self.compute_perplexity(x, i, sigma);

            if (perplexity - target_perplexity).abs() < tolerance {
                return Ok(sigma);
            }

            if perplexity > target_perplexity {
                sigma_max = sigma;
            } else {
                sigma_min = sigma;
            }
        }

        Ok((sigma_min + sigma_max) / 2.0)
    }

    /// Compute perplexity for given sigma
    fn compute_perplexity(&self, x: &Array2<f64>, i: usize, sigma: f64) -> f64 {
        let (n_samples, _) = x.dim();
        let mut probabilities = Vec::new();
        let mut sum_prob = 0.0;

        for j in 0..n_samples {
            if i != j {
                let distance_sq = self.compute_distance(&x.row(i), &x.row(j)).powi(2);
                let prob = (-distance_sq / (2.0 * sigma * sigma)).exp();
                probabilities.push(prob);
                sum_prob += prob;
            }
        }

        // Normalize probabilities
        for prob in &mut probabilities {
            *prob /= sum_prob;
        }

        // Compute entropy
        let entropy = probabilities
            .iter()
            .filter(|&&p| p > 1e-12)
            .map(|&p| -p * p.ln())
            .sum::<f64>();

        // Perplexity = 2^entropy
        2.0_f64.powf(entropy)
    }

    /// Optimize t-SNE embedding using gradient descent
    fn optimize_tsne_embedding(
        &self,
        embedding: &mut Array2<f64>,
        p_matrix: &Array2<f64>,
    ) -> Result<usize> {
        let (n_samples, n_components) = embedding.dim();
        let mut momentum = Array2::<f64>::zeros((n_samples, n_components));
        let momentum_factor = 0.8;

        for iter in 0..self.max_iter {
            // Compute Q matrix (similarities in low-dimensional space)
            let q_matrix = self.compute_tsne_q_matrix(embedding)?;

            // Compute gradient
            let gradient = self.compute_tsne_gradient(embedding, p_matrix, &q_matrix)?;

            // Apply gradient with momentum
            for i in 0..n_samples {
                for j in 0..n_components {
                    momentum[[i, j]] =
                        momentum_factor * momentum[[i, j]] - self.learning_rate * gradient[[i, j]];
                    embedding[[i, j]] += momentum[[i, j]];
                }
            }

            // Early stopping condition (simplified)
            if iter > 100 && iter % 100 == 0 {
                let gradient_norm = gradient.iter().map(|&x| x * x).sum::<f64>().sqrt();
                if gradient_norm < 1e-6 {
                    return Ok(iter + 1);
                }
            }
        }

        Ok(self.max_iter)
    }

    /// Compute Q matrix for t-SNE
    fn compute_tsne_q_matrix(&self, embedding: &Array2<f64>) -> Result<Array2<f64>> {
        let (n_samples, _) = embedding.dim();
        let mut q_matrix = Array2::zeros((n_samples, n_samples));
        let mut sum_q = 0.0;

        for i in 0..n_samples {
            for j in 0..n_samples {
                if i != j {
                    let mut distance_sq = 0.0;
                    for d in 0..embedding.ncols() {
                        let diff = embedding[[i, d]] - embedding[[j, d]];
                        distance_sq += diff * diff;
                    }

                    let q = 1.0 / (1.0 + distance_sq);
                    q_matrix[[i, j]] = q;
                    sum_q += q;
                }
            }
        }

        // Normalize Q matrix
        if sum_q > 1e-12 {
            for i in 0..n_samples {
                for j in 0..n_samples {
                    if i != j {
                        q_matrix[[i, j]] /= sum_q;
                    }
                }
            }
        }

        Ok(q_matrix)
    }

    /// Compute t-SNE gradient
    fn compute_tsne_gradient(
        &self,
        embedding: &Array2<f64>,
        p_matrix: &Array2<f64>,
        q_matrix: &Array2<f64>,
    ) -> Result<Array2<f64>> {
        let (n_samples, n_components) = embedding.dim();
        let mut gradient = Array2::zeros((n_samples, n_components));

        for i in 0..n_samples {
            for d in 0..n_components {
                let mut grad = 0.0;

                for j in 0..n_samples {
                    if i != j {
                        let p_ij = p_matrix[[i, j]];
                        let q_ij = q_matrix[[i, j]];

                        let mut distance_sq = 0.0;
                        for k in 0..n_components {
                            let diff = embedding[[i, k]] - embedding[[j, k]];
                            distance_sq += diff * diff;
                        }

                        let factor = (p_ij - q_ij) * (embedding[[i, d]] - embedding[[j, d]])
                            / (1.0 + distance_sq);
                        grad += 4.0 * factor;
                    }
                }

                gradient[[i, d]] = grad;
            }
        }

        Ok(gradient)
    }

    /// Build fuzzy simplicial set for UMAP
    fn build_fuzzy_simplicial_set(
        &self,
        x: &Array2<f64>,
        neighbors: &Array2<usize>,
    ) -> Result<Array2<f64>> {
        let (n_samples, _) = x.dim();
        let mut fuzzy_graph = Array2::zeros((n_samples, n_samples));

        // Compute local connectivity
        for i in 0..n_samples {
            let rho = self.compute_distance(&x.row(i), &x.row(neighbors[[i, 0]]));

            for &j in neighbors.row(i) {
                if i != j {
                    let distance = self.compute_distance(&x.row(i), &x.row(j));
                    let sigma = self.estimate_sigma_umap(x, i, neighbors)?;

                    let weight = if distance > rho {
                        (-(distance - rho) / sigma).exp()
                    } else {
                        1.0
                    };

                    fuzzy_graph[[i, j]] = weight;
                }
            }
        }

        // Symmetrize the graph
        let mut symmetric_graph = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in 0..n_samples {
                let prob_ij = fuzzy_graph[[i, j]];
                let prob_ji = fuzzy_graph[[j, i]];

                // Combine probabilities: a + b - ab
                symmetric_graph[[i, j]] = prob_ij + prob_ji - prob_ij * prob_ji;
            }
        }

        Ok(symmetric_graph)
    }

    /// Estimate sigma for UMAP
    fn estimate_sigma_umap(
        &self,
        x: &Array2<f64>,
        i: usize,
        neighbors: &Array2<usize>,
    ) -> Result<f64> {
        let mut distances = Vec::new();

        for &j in neighbors.row(i) {
            if i != j {
                let distance = self.compute_distance(&x.row(i), &x.row(j));
                distances.push(distance);
            }
        }

        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Use median distance as sigma estimate
        if !distances.is_empty() {
            Ok(distances[distances.len() / 2])
        } else {
            Ok(1.0)
        }
    }

    /// Optimize UMAP embedding
    fn optimize_umap_embedding(
        &self,
        embedding: &mut Array2<f64>,
        fuzzy_graph: &Array2<f64>,
    ) -> Result<usize> {
        let (n_samples, _n_components) = embedding.dim();
        // TODO: Support seeding for reproducibility
        let mut rng = thread_rng();

        for iter in 0..self.max_iter {
            // Sample edges from the fuzzy graph
            for i in 0..n_samples {
                for j in (i + 1)..n_samples {
                    let weight = fuzzy_graph[[i, j]];

                    if weight > rng.gen() {
                        // Attractive force
                        self.apply_umap_force(embedding, i, j, true)?;
                    } else {
                        // Repulsive force
                        self.apply_umap_force(embedding, i, j, false)?;
                    }
                }
            }

            // Simple convergence check
            if iter > 100 && iter % 50 == 0 {
                // Could add more sophisticated convergence criteria
            }
        }

        Ok(self.max_iter)
    }

    /// Apply UMAP force between two points
    fn apply_umap_force(
        &self,
        embedding: &mut Array2<f64>,
        i: usize,
        j: usize,
        attractive: bool,
    ) -> Result<()> {
        let n_components = embedding.ncols();

        // Compute distance
        let mut distance_sq = 0.0;
        for d in 0..n_components {
            let diff = embedding[[i, d]] - embedding[[j, d]];
            distance_sq += diff * diff;
        }

        let distance = distance_sq.sqrt().max(1e-12);

        // Compute force magnitude
        let force_magnitude = if attractive {
            // Attractive force
            1.0 / (1.0 + self.spread * distance_sq)
        } else {
            // Repulsive force
            self.spread / ((0.001 + distance_sq) * (1.0 + self.spread * distance_sq))
        };

        // Apply force
        let learning_rate = self.learning_rate * 0.01; // Scale down for stability

        for d in 0..n_components {
            let diff = embedding[[i, d]] - embedding[[j, d]];
            let force_component = force_magnitude * diff / distance;

            if attractive {
                embedding[[i, d]] -= learning_rate * force_component;
                embedding[[j, d]] += learning_rate * force_component;
            } else {
                embedding[[i, d]] += learning_rate * force_component;
                embedding[[j, d]] -= learning_rate * force_component;
            }
        }

        Ok(())
    }

    /// Compute smallest eigenvectors (for LLE and Laplacian Eigenmaps)
    fn compute_smallest_eigenvectors(&self, matrix: &Array2<f64>) -> Result<Array2<f64>> {
        let n = matrix.nrows();

        // Use scirs2-linalg for eigendecomposition
        let (eigenvalues, eigenvectors) = matrix.eigh(UPLO::Lower).map_err(|e| {
            SklearsError::NumericalError(format!("Eigendecomposition failed: {}", e))
        })?;

        // eigh returns eigenvalues in ascending order
        // Collect eigenvalue-eigenvector pairs
        let mut eigen_pairs: Vec<(f64, usize)> = eigenvalues
            .iter()
            .enumerate()
            .map(|(i, &val)| (val, i))
            .collect();

        // Sort by eigenvalue (already ascending from eigh, but sort for clarity)
        eigen_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Skip the first eigenvalue if it's close to zero (for Laplacian)
        let start_idx = if eigen_pairs[0].0.abs() < 1e-10 { 1 } else { 0 };
        let end_idx = (start_idx + self.n_components).min(n);

        let mut result = Array2::zeros((n, self.n_components));
        for (result_col, i) in (start_idx..end_idx).enumerate() {
            if result_col < self.n_components {
                let eigen_idx = eigen_pairs[i].1;
                for row in 0..n {
                    result[[row, result_col]] = eigenvectors[[row, eigen_idx]];
                }
            }
        }

        Ok(result)
    }

    /// Compute largest eigenvectors (for MDS)
    fn compute_largest_eigenvectors(&self, matrix: &Array2<f64>) -> Result<Array2<f64>> {
        let n = matrix.nrows();

        // Use scirs2-linalg for eigendecomposition
        let (eigenvalues, eigenvectors) = matrix.eigh(UPLO::Lower).map_err(|e| {
            SklearsError::NumericalError(format!("Eigendecomposition failed: {}", e))
        })?;

        // eigh returns eigenvalues in ascending order, we need descending
        let mut eigen_pairs: Vec<(f64, usize)> = eigenvalues
            .iter()
            .enumerate()
            .map(|(i, &val)| (val, i))
            .collect();

        // Sort in descending order for largest eigenvalues
        eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut result = Array2::zeros((n, self.n_components));
        for (result_col, i) in (0..self.n_components).enumerate() {
            if i < eigen_pairs.len() && eigen_pairs[i].0 > 0.0 {
                let eigen_idx = eigen_pairs[i].1;
                let sqrt_eigenval = eigen_pairs[i].0.sqrt();

                for row in 0..n {
                    result[[row, result_col]] = eigenvectors[[row, eigen_idx]] * sqrt_eigenval;
                }
            }
        }

        Ok(result)
    }
}

impl ManifoldLearning<TrainedManifoldLearning> {
    /// Get the embedding
    pub fn embedding(&self) -> &Array2<f64> {
        &self.state.embedding
    }

    /// Compute distance between two points using the trained metric
    fn compute_distance(
        &self,
        a: &scirs2_core::ndarray::ArrayView1<f64>,
        b: &scirs2_core::ndarray::ArrayView1<f64>,
    ) -> f64 {
        match self.state.metric {
            DistanceMetric::Euclidean => a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| (x - y) * (x - y))
                .sum::<f64>()
                .sqrt(),
            DistanceMetric::Manhattan => a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| (x - y).abs())
                .sum::<f64>(),
            DistanceMetric::Cosine => {
                let dot_product = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum::<f64>();
                let norm_a = a.iter().map(|&x| x * x).sum::<f64>().sqrt();
                let norm_b = b.iter().map(|&x| x * x).sum::<f64>().sqrt();

                if norm_a > 1e-12 && norm_b > 1e-12 {
                    1.0 - (dot_product / (norm_a * norm_b))
                } else {
                    0.0
                }
            }
        }
    }

    /// LLE out-of-sample transformation
    fn lle_transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        // Simplified out-of-sample extension using nearest neighbors
        let (n_new_samples, _) = x.dim();
        let mut transformed = Array2::zeros((n_new_samples, self.state.n_components));

        for i in 0..n_new_samples {
            // Find nearest neighbors in training data
            let mut distances: Vec<(f64, usize)> = Vec::new();

            for j in 0..self.state.training_data.nrows() {
                let distance = self.compute_distance(&x.row(i), &self.state.training_data.row(j));
                distances.push((distance, j));
            }

            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // Use weighted average of k nearest neighbors
            let k = self.n_neighbors.min(distances.len());
            let mut total_weight = 0.0;

            for &(distance, neighbor_idx) in distances.iter().take(k) {
                let weight = if distance > 1e-12 {
                    1.0 / distance
                } else {
                    1e12
                };
                total_weight += weight;

                for d in 0..self.state.n_components {
                    transformed[[i, d]] += weight * self.state.embedding[[neighbor_idx, d]];
                }
            }

            // Normalize by total weight
            if total_weight > 1e-12 {
                for d in 0..self.state.n_components {
                    transformed[[i, d]] /= total_weight;
                }
            }
        }

        Ok(transformed)
    }

    /// Isomap out-of-sample transformation
    fn isomap_transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        // Use same approach as LLE for simplicity
        self.lle_transform(x)
    }

    /// Laplacian Eigenmaps out-of-sample transformation
    fn laplacian_transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        // Use same approach as LLE for simplicity
        self.lle_transform(x)
    }

    /// UMAP out-of-sample transformation
    fn umap_transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        // Use same approach as LLE for simplicity
        self.lle_transform(x)
    }
}

impl Default for ManifoldLearning<Untrained> {
    fn default() -> Self {
        Self::new(ManifoldAlgorithm::LLE, 2)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_manifold_learning_creation() {
        let ml = ManifoldLearning::new(ManifoldAlgorithm::LLE, 2)
            .n_neighbors(10)
            .metric(DistanceMetric::Euclidean)
            .max_iter(100)
            .learning_rate(200.0)
            .perplexity(30.0)
            .random_state(42);

        assert_eq!(ml.algorithm, ManifoldAlgorithm::LLE);
        assert_eq!(ml.n_components, 2);
        assert_eq!(ml.n_neighbors, 10);
        assert_eq!(ml.metric, DistanceMetric::Euclidean);
        assert_eq!(ml.max_iter, 100);
        assert_eq!(ml.learning_rate, 200.0);
        assert_eq!(ml.perplexity, 30.0);
        assert_eq!(ml.random_state, Some(42));
    }

    #[test]
    fn test_manifold_learning_lle() {
        // Create a simple 3D dataset that lies on a 2D manifold
        let x = array![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ];

        let ml = ManifoldLearning::new(ManifoldAlgorithm::LLE, 2)
            .n_neighbors(3)
            .random_state(42);

        let trained_ml = ml.fit(&x, &()).unwrap();

        assert_eq!(trained_ml.embedding().dim(), (6, 2));
        assert_eq!(trained_ml.state.algorithm, ManifoldAlgorithm::LLE);

        // Test transformation
        let new_point = array![[0.5, 0.5, 0.0]];
        let transformed = trained_ml.transform(&new_point).unwrap();
        assert_eq!(transformed.dim(), (1, 2));
    }

    #[test]
    fn test_manifold_learning_different_metrics() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];

        let metrics = vec![
            DistanceMetric::Euclidean,
            DistanceMetric::Manhattan,
            DistanceMetric::Cosine,
        ];

        for metric in metrics {
            let ml = ManifoldLearning::new(ManifoldAlgorithm::LLE, 1)
                .n_neighbors(2)
                .metric(metric)
                .random_state(42);

            let trained_ml = ml.fit(&x, &()).unwrap();
            assert_eq!(trained_ml.embedding().dim(), (4, 1));
        }
    }

    #[test]
    fn test_manifold_learning_error_cases() {
        let x_small = array![[1.0, 2.0]]; // Only 1 sample
        let ml = ManifoldLearning::new(ManifoldAlgorithm::LLE, 2);
        let result = ml.fit(&x_small, &());
        assert!(result.is_err());

        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let ml_bad_neighbors = ManifoldLearning::new(ManifoldAlgorithm::LLE, 1).n_neighbors(5); // More neighbors than samples
        let result = ml_bad_neighbors.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_distance_computation() {
        let ml = ManifoldLearning::new(ManifoldAlgorithm::LLE, 2);

        let a = array![1.0, 2.0, 3.0];
        let b = array![4.0, 5.0, 6.0];

        let euclidean_dist = ml.compute_distance(&a.view(), &b.view());
        assert!((euclidean_dist - (3.0_f64 * 3.0_f64).sqrt() * 3.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_knn_computation() {
        let x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];

        let ml = ManifoldLearning::new(ManifoldAlgorithm::LLE, 2);
        let neighbors = ml.find_knn(&x, 2).unwrap();

        assert_eq!(neighbors.dim(), (4, 2));

        // Check that each point doesn't include itself as a neighbor
        for i in 0..4 {
            for j in 0..2 {
                assert_ne!(neighbors[[i, j]], i);
            }
        }
    }
}
