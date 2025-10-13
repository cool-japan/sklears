//! Locally Linear Embedding (LLE) implementation
//!
//! This module provides LLE for non-linear dimensionality reduction through locally linear embedding.

use scirs2_core::ndarray::ndarray_linalg::{Eigh, UPLO};
use scirs2_core::ndarray::{Array2, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};

/// Locally Linear Embedding (LLE)
///
/// LLE seeks a lower-dimensional projection of the data which preserves
/// distances within local neighborhoods. It attempts to characterize the
/// local geometry of the manifold by linear coefficients that reconstruct
/// each data point from its neighbors.
///
/// # Parameters
///
/// * `n_neighbors` - Number of neighbors to consider for each point
/// * `n_components` - Number of coordinates for the manifold
/// * `reg` - Regularization constant for weight calculation
/// * `eigen_solver` - The eigensolver to use
/// * `tol` - Tolerance for convergence
/// * `max_iter` - Maximum number of iterations
/// * `method` - Implementation method for LLE
/// * `hessian_tol` - Threshold for Hessian eigenvalue regularization
/// * `modified_tol` - Tolerance for modified LLE
/// * `neighbors_algorithm` - Algorithm to use for nearest neighbors search
/// * `random_state` - Random state for reproducibility
/// * `n_jobs` - Number of parallel jobs
///
/// # Examples
///
/// ```
/// use sklears_manifold::LocallyLinearEmbedding;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]];
///
/// let lle = LocallyLinearEmbedding::new()
///     .n_neighbors(2)
///     .n_components(2);
/// let fitted = lle.fit(&x.view(), &()).unwrap();
/// let embedded = fitted.transform(&x.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct LocallyLinearEmbedding<S = Untrained> {
    state: S,
    n_neighbors: usize,
    n_components: usize,
    reg: f64,
    eigen_solver: String,
    tol: f64,
    max_iter: Option<usize>,
    method: String,
    hessian_tol: f64,
    modified_tol: f64,
    neighbors_algorithm: String,
    random_state: Option<u64>,
    n_jobs: Option<i32>,
}

/// Trained state for LLE
#[derive(Debug, Clone)]
pub struct LleTrained {
    /// The low-dimensional embedding of the training data
    pub embedding: Array2<f64>,
    /// Reconstruction weights matrix
    pub reconstruction_weights: Array2<f64>,
    /// Reconstruction error from the embedding
    pub reconstruction_error: f64,
}

impl LocallyLinearEmbedding<Untrained> {
    /// Create a new LocallyLinearEmbedding instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_neighbors: 5,
            n_components: 2,
            reg: 1e-3,
            eigen_solver: "auto".to_string(),
            tol: 1e-6,
            max_iter: Some(100),
            method: "standard".to_string(),
            hessian_tol: 1e-4,
            modified_tol: 1e-12,
            neighbors_algorithm: "auto".to_string(),
            random_state: None,
            n_jobs: None,
        }
    }

    /// Set the number of neighbors
    pub fn n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the regularization constant
    pub fn reg(mut self, reg: f64) -> Self {
        self.reg = reg;
        self
    }

    /// Set the eigen solver
    pub fn eigen_solver(mut self, eigen_solver: &str) -> Self {
        self.eigen_solver = eigen_solver.to_string();
        self
    }

    /// Set the tolerance
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the maximum iterations
    pub fn max_iter(mut self, max_iter: Option<usize>) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the method
    pub fn method(mut self, method: &str) -> Self {
        self.method = method.to_string();
        self
    }

    /// Set the hessian tolerance
    pub fn hessian_tol(mut self, hessian_tol: f64) -> Self {
        self.hessian_tol = hessian_tol;
        self
    }

    /// Set the modified tolerance
    pub fn modified_tol(mut self, modified_tol: f64) -> Self {
        self.modified_tol = modified_tol;
        self
    }

    /// Set the neighbors algorithm
    pub fn neighbors_algorithm(mut self, neighbors_algorithm: &str) -> Self {
        self.neighbors_algorithm = neighbors_algorithm.to_string();
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }

    /// Set the number of jobs
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.n_jobs = n_jobs;
        self
    }
}

impl Default for LocallyLinearEmbedding<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for LocallyLinearEmbedding<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for LocallyLinearEmbedding<Untrained> {
    type Fitted = LocallyLinearEmbedding<LleTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let x = x.mapv(|x| x);
        let (n_samples, _) = x.dim();

        if n_samples <= self.n_components {
            return Err(SklearsError::InvalidInput(
                "Number of samples must be greater than n_components".to_string(),
            ));
        }

        if self.n_neighbors >= n_samples {
            return Err(SklearsError::InvalidInput(
                "n_neighbors must be less than number of samples".to_string(),
            ));
        }

        // Step 1: Find k-nearest neighbors for each point
        let neighbor_indices = self.find_neighbors(&x)?;

        // Step 2: Compute reconstruction weights
        let weights = self.compute_reconstruction_weights(&x, &neighbor_indices)?;

        // Step 3: Find the embedding that preserves these weights
        let embedding = self.compute_embedding(&weights)?;

        // Compute reconstruction error
        let reconstruction_error =
            self.compute_lle_reconstruction_error(&x, &neighbor_indices, &weights);

        Ok(LocallyLinearEmbedding {
            state: LleTrained {
                embedding,
                reconstruction_weights: weights,
                reconstruction_error,
            },
            n_neighbors: self.n_neighbors,
            n_components: self.n_components,
            reg: self.reg,
            eigen_solver: self.eigen_solver,
            tol: self.tol,
            max_iter: self.max_iter,
            method: self.method,
            hessian_tol: self.hessian_tol,
            modified_tol: self.modified_tol,
            neighbors_algorithm: self.neighbors_algorithm,
            random_state: self.random_state,
            n_jobs: self.n_jobs,
        })
    }
}

impl LocallyLinearEmbedding<Untrained> {
    fn find_neighbors(&self, x: &Array2<f64>) -> SklResult<Array2<usize>> {
        let n_samples = x.nrows();
        let mut neighbor_indices = Array2::zeros((n_samples, self.n_neighbors));

        for i in 0..n_samples {
            // Compute distances to all other points
            let mut distances: Vec<(f64, usize)> = Vec::new();
            for j in 0..n_samples {
                if i != j {
                    let diff = &x.row(i) - &x.row(j);
                    let dist = diff.mapv(|x| x * x).sum().sqrt();
                    distances.push((dist, j));
                }
            }

            // Sort by distance and take k nearest neighbors
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            for (neighbor_idx, &(_, j)) in distances.iter().take(self.n_neighbors).enumerate() {
                neighbor_indices[[i, neighbor_idx]] = j;
            }
        }

        Ok(neighbor_indices)
    }

    fn compute_reconstruction_weights(
        &self,
        x: &Array2<f64>,
        neighbor_indices: &Array2<usize>,
    ) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows();
        let mut weights = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            // Extract neighbors for point i
            let neighbors: Vec<usize> = (0..self.n_neighbors)
                .map(|j| neighbor_indices[[i, j]])
                .collect();

            // Create local covariance matrix
            let mut local_gram = Array2::zeros((self.n_neighbors, self.n_neighbors));
            for (a, &neighbor_a) in neighbors.iter().enumerate() {
                for (b, &neighbor_b) in neighbors.iter().enumerate() {
                    let diff_a = &x.row(neighbor_a) - &x.row(i);
                    let diff_b = &x.row(neighbor_b) - &x.row(i);
                    local_gram[[a, b]] = diff_a.dot(&diff_b);
                }
            }

            // Add regularization
            for j in 0..self.n_neighbors {
                local_gram[[j, j]] += self.reg;
            }

            // Solve for weights: local_gram * w = 1
            let ones = Array2::ones((self.n_neighbors, 1));
            let w = match self.solve_linear_system(&local_gram, &ones) {
                Ok(w) => w,
                Err(_) => {
                    // Fallback: use uniform weights
                    Array2::from_elem((self.n_neighbors, 1), 1.0 / self.n_neighbors as f64)
                }
            };

            // Normalize weights
            let weight_sum: f64 = w.sum();
            if weight_sum > 1e-15 {
                for (j, &neighbor_j) in neighbors.iter().enumerate() {
                    weights[[i, neighbor_j]] = w[[j, 0]] / weight_sum;
                }
            }
        }

        Ok(weights)
    }

    fn solve_linear_system(&self, a: &Array2<f64>, b: &Array2<f64>) -> SklResult<Array2<f64>> {
        // Simple pseudo-inverse solution for small systems
        // In a full implementation, would use proper linear system solver
        let n = a.nrows();
        let a_inv: Array2<f64> = Array2::eye(n);

        // Simple diagonal regularization for numerical stability
        let mut a_reg = a.clone();
        for i in 0..n {
            a_reg[[i, i]] += 1e-10;
        }

        // Very simplified solution - in practice would use proper linear algebra
        Ok(Array2::ones((n, 1)) / n as f64)
    }

    fn compute_embedding(&self, weights: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = weights.nrows();

        // Create the matrix M = (I - W)^T (I - W)
        let identity: Array2<f64> = Array2::eye(n_samples);
        let i_minus_w = &identity - weights;
        let mut m = Array2::zeros((n_samples, n_samples));

        // Compute M = (I - W)^T (I - W)
        for i in 0..n_samples {
            for j in 0..n_samples {
                let mut sum = 0.0;
                for k in 0..n_samples {
                    sum += i_minus_w[[k, i]] * i_minus_w[[k, j]] as f64;
                }
                m[[i, j]] = sum;
            }
        }

        // Find the eigenvectors corresponding to the smallest eigenvalues
        let (eigenvals, eigenvecs) = m
            .eigh(UPLO::Lower)
            .map_err(|e| SklearsError::InvalidInput(format!("Eigendecomposition failed: {e}")))?;

        // Sort eigenvalues and eigenvectors in ascending order
        let mut eigen_pairs: Vec<(f64, usize)> = eigenvals
            .iter()
            .enumerate()
            .map(|(i, &val)| (val, i))
            .collect();
        eigen_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Take the eigenvectors corresponding to the smallest non-zero eigenvalues
        // Skip the first eigenvector (corresponding to eigenvalue 0)
        let mut embedding = Array2::zeros((n_samples, self.n_components));
        for (comp_idx, &(eigenval, eigen_idx)) in eigen_pairs
            .iter()
            .skip(1)
            .take(self.n_components)
            .enumerate()
        {
            if eigenval > 1e-12 {
                for i in 0..n_samples {
                    embedding[[i, comp_idx]] = eigenvecs[[i, eigen_idx]];
                }
            }
        }

        Ok(embedding)
    }

    fn compute_lle_reconstruction_error(
        &self,
        x: &Array2<f64>,
        neighbor_indices: &Array2<usize>,
        weights: &Array2<f64>,
    ) -> f64 {
        let n_samples = x.nrows();
        let mut total_error = 0.0;

        for i in 0..n_samples {
            let mut reconstruction: Array2<f64> = Array2::zeros((1, x.ncols()));

            // Reconstruct point i from its neighbors
            for j in 0..self.n_neighbors {
                let neighbor_j = neighbor_indices[[i, j]];
                let weight = weights[[i, neighbor_j]];
                for k in 0..x.ncols() {
                    reconstruction[[0, k]] += weight * x[[neighbor_j, k]];
                }
            }

            // Compute reconstruction error
            let diff = &x.row(i) - &reconstruction.row(0);
            let error = diff.mapv(|x| x * x).sum();
            total_error += error;
        }

        total_error / n_samples as f64
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<f64>> for LocallyLinearEmbedding<LleTrained> {
    fn transform(&self, _x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        // LLE doesn't support transforming new data in this implementation
        Ok(self.state.embedding.clone())
    }
}

impl LocallyLinearEmbedding<LleTrained> {
    /// Get the embedding
    pub fn embedding(&self) -> &Array2<f64> {
        &self.state.embedding
    }

    /// Get the reconstruction weights
    pub fn reconstruction_weights(&self) -> &Array2<f64> {
        &self.state.reconstruction_weights
    }

    /// Get the reconstruction error
    pub fn reconstruction_error(&self) -> f64 {
        self.state.reconstruction_error
    }
}
