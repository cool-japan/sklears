//! Locally Linear Embedding (LLE) implementation
//!
//! This module provides LLE for non-linear dimensionality reduction through locally linear embedding.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_linalg::compat::{ArrayLinalgExt, UPLO};
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
    /// Training data retained for the out-of-sample `transform` extension.
    ///
    /// New points are projected by reconstructing them from their nearest
    /// training neighbors (barycentric weights) and applying those same weights
    /// to the neighbors' embedding coordinates, so the original training
    /// coordinates must be kept around.
    training_data: Array2<f64>,
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

        // Compute reconstruction error (last use of `x` as a borrow before it is moved).
        let reconstruction_error =
            self.compute_lle_reconstruction_error(&x, &neighbor_indices, &weights);

        Ok(LocallyLinearEmbedding {
            state: LleTrained {
                embedding,
                reconstruction_weights: weights,
                reconstruction_error,
                // `x` is not needed after the reconstruction-error computation, so it is
                // moved (not cloned) into the trained state for out-of-sample transforms.
                training_data: x,
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

impl<S> LocallyLinearEmbedding<S> {
    /// Solve the local LLE weight system `a x = b` for barycentric reconstruction weights.
    ///
    /// `a` is the (already `reg`-regularized) local Gram matrix and `b` is the all-ones
    /// right-hand side. Returns the raw, *un-normalized* weight vector. A tiny diagonal
    /// jitter is added on top of the caller's `self.reg` purely for numerical stability so
    /// that near-singular Gram matrices (e.g. duplicate neighbors) still solve cleanly.
    ///
    /// This lives on the generic `impl<S>` block so both `fit` (via `Untrained`) and the
    /// out-of-sample `transform` (via `LleTrained`) can reuse the exact same solver.
    fn solve_linear_system(&self, a: &Array2<f64>, b: &Array1<f64>) -> SklResult<Array1<f64>> {
        let n = a.nrows();
        let mut a_reg = a.clone();
        for i in 0..n {
            // Numerical stability, layered on top of the caller's `self.reg` diagonal term.
            a_reg[[i, i]] += 1e-10;
        }
        a_reg.solve(b).map_err(|e| {
            SklearsError::NumericalError(format!("Failed to solve local LLE weight system: {e}"))
        })
    }

    /// Compute normalized barycentric reconstruction weights for a single query point.
    ///
    /// Given a `query` point and the row indices of its `neighbors` within `data`
    /// (shape `(k, n_features)` once gathered), this builds the local Gram matrix
    /// `G[a, b] = (data[neighbor_a] - query) . (data[neighbor_b] - query)`, regularizes
    /// its diagonal with `self.reg`, solves `G w = 1`, and normalizes `w` to sum to 1.
    ///
    /// Both `fit` (query = the training point itself, neighbors = its k training neighbors)
    /// and `transform` (query = a new point, neighbors = its k nearest training points)
    /// call this identical routine — there is a single copy of the weight-solving logic.
    ///
    /// Falls back to uniform weights when the linear solve genuinely fails (singular system)
    /// or the resulting weights sum to ~0.
    fn barycentric_weights(
        &self,
        query: &ArrayView1<'_, f64>,
        data: &Array2<f64>,
        neighbors: &[usize],
    ) -> Array1<f64> {
        let k = neighbors.len();

        // Center each neighbor on the query point: diff_a = data[neighbor_a] - query.
        let diffs: Vec<Array1<f64>> = neighbors.iter().map(|&n| &data.row(n) - query).collect();

        // Local Gram matrix G[a, b] = diff_a . diff_b.
        let mut local_gram = Array2::zeros((k, k));
        for a in 0..k {
            for b in 0..k {
                local_gram[[a, b]] = diffs[a].dot(&diffs[b]);
            }
        }

        // Regularize the diagonal so the system is well-posed even for flat neighborhoods.
        for j in 0..k {
            local_gram[[j, j]] += self.reg;
        }

        // Solve G w = 1 for the (un-normalized) barycentric weights.
        let ones = Array1::ones(k);
        let w = match self.solve_linear_system(&local_gram, &ones) {
            Ok(w) => w,
            // Defensive fallback for a genuinely singular system: uniform weights.
            Err(_) => Array1::from_elem(k, 1.0 / k as f64),
        };

        // Normalize the weights to sum to 1 (the barycentric constraint).
        let weight_sum: f64 = w.sum();
        if weight_sum.abs() > 1e-15 {
            w / weight_sum
        } else {
            Array1::from_elem(k, 1.0 / k as f64)
        }
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
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("operation should succeed"));
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

            // Barycentric reconstruction weights of point i from its own neighbors.
            // Reuses the exact same weight-solving routine that `transform` uses for
            // out-of-sample points (query = the training point itself here).
            let w = self.barycentric_weights(&x.row(i), x, &neighbors);

            // Scatter the (already normalized) weights into the sparse weight matrix.
            for (j, &neighbor_j) in neighbors.iter().enumerate() {
                weights[[i, neighbor_j]] = w[j];
            }
        }

        Ok(weights)
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
        eigen_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("operation should succeed"));

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
    /// Project new (out-of-sample) points into the learned embedding space.
    ///
    /// This is the standard LLE out-of-sample extension (the same one implemented by
    /// scikit-learn's `LocallyLinearEmbedding.transform`): each new point is reconstructed
    /// from its nearest *training* neighbors using barycentric weights, and those weights
    /// are then applied to the neighbors' embedding coordinates.
    fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let training_data = &self.state.training_data;
        let embedding = &self.state.embedding;
        let n_train = training_data.nrows();
        let n_features = training_data.ncols();

        // Feature-count must match what the model was trained on.
        if x.ncols() != n_features {
            return Err(SklearsError::FeatureMismatch {
                expected: n_features,
                actual: x.ncols(),
            });
        }

        let n_new = x.nrows();
        // Use at most as many neighbors as there are training points, and at least one.
        let k = self.n_neighbors.min(n_train).max(1);

        let mut embedding_new = Array2::zeros((n_new, self.n_components));

        for p in 0..n_new {
            let query = x.row(p);

            // (a) Find the k nearest TRAINING points to the new query point by direct
            //     Euclidean distance (cross-set version of `find_neighbors`).
            let mut distances: Vec<(f64, usize)> = Vec::with_capacity(n_train);
            for t in 0..n_train {
                let diff = &training_data.row(t) - &query;
                let dist = diff.mapv(|v| v * v).sum().sqrt();
                distances.push((dist, t));
            }
            // `total_cmp` gives a panic-free total order over f64 (no unwrap/expect).
            distances.sort_by(|a, b| a.0.total_cmp(&b.0));
            let neighbors: Vec<usize> = distances.iter().take(k).map(|&(_, t)| t).collect();

            // (b) + (c) Build the local Gram matrix from differences to the NEW point,
            //     solve `G w = 1`, and normalize `w` to sum to 1 — the identical routine
            //     that `fit` uses for its training reconstruction weights.
            let w = self.barycentric_weights(&query, training_data, &neighbors);

            // (d) The new point's embedding is the weighted combination of its neighbors'
            //     training embedding coordinates.
            for (j, &neighbor) in neighbors.iter().enumerate() {
                for c in 0..self.n_components {
                    embedding_new[[p, c]] += w[j] * embedding[[neighbor, c]];
                }
            }
        }

        Ok(embedding_new)
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

#[cfg(test)]
mod tests {
    use super::{LleTrained, LocallyLinearEmbedding};
    use scirs2_core::ndarray::{array, Array2};
    use sklears_core::traits::{Fit, Transform};

    /// An 8-point 4x2 grid lying on a gently tilted plane (intrinsic dimension 2,
    /// ambient dimension 3) — a well-behaved manifold for LLE.
    fn training_data() -> Array2<f64> {
        array![
            [0.0, 0.0, 0.00],
            [1.0, 0.0, 0.10],
            [2.0, 0.0, 0.20],
            [3.0, 0.0, 0.30],
            [0.0, 1.0, 0.10],
            [1.0, 1.0, 0.20],
            [2.0, 1.0, 0.30],
            [3.0, 1.0, 0.40],
        ]
    }

    fn fitted() -> LocallyLinearEmbedding<LleTrained> {
        let x = training_data();
        LocallyLinearEmbedding::new()
            .n_neighbors(4)
            .n_components(2)
            .fit(&x.view(), &())
            .expect("fit should succeed")
    }

    /// (1) Fit succeeds and the embedding has the expected shape.
    #[test]
    fn test_fit_embedding_shape() {
        let model = fitted();
        assert_eq!(model.embedding().dim(), (8, 2));
    }

    /// (3) `transform` output shape matches `(new_data.nrows(), n_components)`.
    #[test]
    fn test_transform_output_shape() {
        let model = fitted();
        let new = array![[0.5, 0.5, 0.15], [2.5, 0.5, 0.35], [1.2, 0.3, 0.20]];
        let out = model
            .transform(&new.view())
            .expect("transform should succeed");
        assert_eq!(out.dim(), (3, 2));
    }

    /// (2) Core regression test: `transform` must actually depend on its input.
    ///
    /// The old bug returned the stale training-time embedding regardless of input.
    /// Here we assert (a) transforming genuinely new data is NOT value-identical to the
    /// stored embedding, and (b) two different new datasets produce two different outputs.
    #[test]
    fn test_transform_depends_on_input() {
        let model = fitted();
        let stored = model.embedding().clone();

        // Genuinely new data, same row count as the embedding but different coordinates.
        let new_diff = array![
            [0.5, 0.2, 0.12],
            [1.5, 0.1, 0.18],
            [2.5, 0.3, 0.28],
            [3.2, 0.4, 0.35],
            [0.2, 0.8, 0.15],
            [1.2, 0.7, 0.24],
            [2.2, 0.6, 0.33],
            [2.9, 0.5, 0.40],
        ];
        let out_diff = model
            .transform(&new_diff.view())
            .expect("transform new_diff should succeed");
        // Must NOT reproduce the stale training embedding (the old broken behavior).
        assert_ne!(
            out_diff, stored,
            "transform returned the stale training embedding instead of projecting the input"
        );

        // Two different new datasets must give two different outputs.
        let new_a = array![[0.5, 0.5, 0.15], [2.5, 0.5, 0.35]];
        let new_b = array![[2.8, 0.9, 0.42], [0.1, 0.2, 0.10]];
        let out_a = model
            .transform(&new_a.view())
            .expect("transform new_a should succeed");
        let out_b = model
            .transform(&new_b.view())
            .expect("transform new_b should succeed");
        assert_ne!(
            out_a, out_b,
            "two different inputs produced identical outputs — input is being ignored"
        );
    }

    /// (4) Near-duplicate sanity check (the standard LLE out-of-sample correctness test).
    ///
    /// Perturbing every training point by a tiny amount and transforming it should
    /// reproduce that point's original embedding row closely, because the barycentric
    /// weights collapse onto the (near-)coincident training neighbor.
    #[test]
    fn test_transform_near_duplicate_recovers_embedding() {
        let model = fitted();
        let stored = model.embedding().clone();
        let x = training_data();

        let perturbed = &x + 1e-4;
        let out = model
            .transform(&perturbed.view())
            .expect("transform perturbed should succeed");

        assert_eq!(out.dim(), stored.dim());
        for i in 0..x.nrows() {
            for c in 0..2 {
                let diff = (out[[i, c]] - stored[[i, c]]).abs();
                assert!(
                    diff < 1e-2,
                    "row {i} comp {c}: transformed {} vs stored {} (|diff| = {diff})",
                    out[[i, c]],
                    stored[[i, c]]
                );
            }
        }
    }

    /// (5) Guards against silently reintroducing the `solve_linear_system` stub.
    ///
    /// The old stub ignored `a` and `b` and always returned uniform `1/n` weights, which
    /// does NOT solve a general system. Here we solve a small known system and check both
    /// `A @ x == b` and that `x` equals the closed-form answer.
    #[test]
    fn test_solve_linear_system_actually_solves() {
        // A = [[4, 1], [1, 3]], b = [1, 2] => x = [1/11, 7/11].
        let a = array![[4.0, 1.0], [1.0, 3.0]];
        let b = array![1.0, 2.0];

        let model = LocallyLinearEmbedding::new();
        let x = model
            .solve_linear_system(&a, &b)
            .expect("solve_linear_system should succeed");

        // The stub's uniform [0.5, 0.5] would give A@x = [2.5, 2.0] != b and would fail here.
        let ax = a.dot(&x);
        assert!((ax[0] - b[0]).abs() < 1e-6, "A@x[0] = {} (expected 1.0)", ax[0]);
        assert!((ax[1] - b[1]).abs() < 1e-6, "A@x[1] = {} (expected 2.0)", ax[1]);

        assert!((x[0] - 1.0 / 11.0).abs() < 1e-6, "x[0] = {}", x[0]);
        assert!((x[1] - 7.0 / 11.0).abs() < 1e-6, "x[1] = {}", x[1]);
    }
}
