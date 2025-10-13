//! Laplacian Eigenmaps implementation
//!
//! This module provides Laplacian Eigenmaps for non-linear dimensionality reduction through spectral graph theory.

use scirs2_core::ndarray::ndarray_linalg::{Eigh, UPLO};
use scirs2_core::ndarray::{Array2, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};

/// Laplacian Eigenmaps
///
/// Laplacian Eigenmaps is a dimensionality reduction technique that uses
/// spectral graph theory. It finds a low-dimensional representation that
/// respects the locality of the manifold by preserving local distances
/// through the eigenvectors of the graph Laplacian.
///
/// # Parameters
///
/// * `n_neighbors` - Number of neighbors to consider for each point
/// * `n_components` - Number of coordinates for the manifold
/// * `reg` - Regularization constant added to the diagonal of the Laplacian
/// * `eigen_solver` - The eigensolver to use
/// * `tol` - Tolerance for convergence
/// * `max_iter` - Maximum number of iterations
/// * `random_state` - Random state for reproducibility
/// * `n_jobs` - Number of parallel jobs
///
/// # Examples
///
/// ```
/// use sklears_manifold::LaplacianEigenmaps;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]];
///
/// let laplacian = LaplacianEigenmaps::new()
///     .n_neighbors(2)
///     .n_components(2);
/// let fitted = laplacian.fit(&x.view(), &()).unwrap();
/// let embedded = fitted.transform(&x.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct LaplacianEigenmaps<S = Untrained> {
    state: S,
    n_neighbors: usize,
    n_components: usize,
    reg: f64,
    eigen_solver: String,
    tol: f64,
    max_iter: Option<usize>,
    random_state: Option<u64>,
    n_jobs: Option<i32>,
}

/// Trained state for Laplacian Eigenmaps
#[derive(Debug, Clone)]
pub struct LaplacianTrained {
    /// The low-dimensional embedding of the training data
    pub embedding: Array2<f64>,
    /// The Laplacian matrix used for embedding
    pub laplacian_matrix: Array2<f64>,
    /// The adjacency matrix of the neighborhood graph
    pub adjacency_matrix: Array2<f64>,
}

impl LaplacianEigenmaps<Untrained> {
    /// Create a new LaplacianEigenmaps instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_neighbors: 5,
            n_components: 2,
            reg: 1e-6,
            eigen_solver: "auto".to_string(),
            tol: 1e-6,
            max_iter: Some(100),
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

impl Default for LaplacianEigenmaps<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for LaplacianEigenmaps<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for LaplacianEigenmaps<Untrained> {
    type Fitted = LaplacianEigenmaps<LaplacianTrained>;

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

        // Build the adjacency matrix using k-nearest neighbors
        let adjacency = self.build_adjacency_matrix(&x)?;

        // Compute the graph Laplacian
        let laplacian = self.compute_laplacian(&adjacency)?;

        // Compute the embedding via eigenvectors
        let embedding = self.compute_embedding(&laplacian)?;

        Ok(LaplacianEigenmaps {
            state: LaplacianTrained {
                embedding,
                laplacian_matrix: laplacian,
                adjacency_matrix: adjacency,
            },
            n_neighbors: self.n_neighbors,
            n_components: self.n_components,
            reg: self.reg,
            eigen_solver: self.eigen_solver,
            tol: self.tol,
            max_iter: self.max_iter,
            random_state: self.random_state,
            n_jobs: self.n_jobs,
        })
    }
}

impl LaplacianEigenmaps<Untrained> {
    fn build_adjacency_matrix(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows();
        let mut adjacency = Array2::zeros((n_samples, n_samples));

        // For each point, find k nearest neighbors
        for i in 0..n_samples {
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

            for &(_, neighbor_idx) in distances.iter().take(self.n_neighbors.min(n_samples - 1)) {
                // Use heat kernel weights: w_ij = exp(-||x_i - x_j||²/σ²)
                let diff = &x.row(i) - &x.row(neighbor_idx);
                let dist_sq = diff.mapv(|x| x * x).sum();
                let weight = (-dist_sq / (2.0 * 1.0)).exp(); // σ = 1.0 for simplicity

                adjacency[[i, neighbor_idx]] = weight;
                adjacency[[neighbor_idx, i]] = weight; // Ensure symmetry
            }
        }

        Ok(adjacency)
    }

    fn compute_laplacian(&self, adjacency: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n = adjacency.nrows();
        let mut laplacian = Array2::zeros((n, n));

        // Compute degree matrix
        let mut degrees = Array2::zeros((n, n));
        for i in 0..n {
            let degree: f64 = adjacency.row(i).sum();
            degrees[[i, i]] = degree;
        }

        // Laplacian = D - A
        for i in 0..n {
            for j in 0..n {
                laplacian[[i, j]] = degrees[[i, j]] - adjacency[[i, j]];
            }
        }

        // Add regularization to diagonal
        for i in 0..n {
            laplacian[[i, i]] += self.reg;
        }

        Ok(laplacian)
    }

    fn compute_embedding(&self, laplacian: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n = laplacian.nrows();

        // Compute eigendecomposition
        let (eigenvals, eigenvecs) = laplacian
            .eigh(UPLO::Lower)
            .map_err(|e| SklearsError::InvalidInput(format!("Eigendecomposition failed: {e}")))?;

        // Sort eigenvalues and eigenvectors in ascending order (smallest first)
        let mut eigen_pairs: Vec<(f64, usize)> = eigenvals
            .iter()
            .enumerate()
            .map(|(i, &val)| (val, i))
            .collect();
        eigen_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Take the eigenvectors corresponding to the smallest non-zero eigenvalues
        // Skip the first eigenvector (corresponding to eigenvalue 0)
        let mut embedding = Array2::zeros((n, self.n_components));
        for (comp_idx, &(eigenval, eigen_idx)) in eigen_pairs
            .iter()
            .skip(1)
            .take(self.n_components)
            .enumerate()
        {
            if eigenval > 1e-12 {
                for i in 0..n {
                    embedding[[i, comp_idx]] = eigenvecs[[i, eigen_idx]];
                }
            }
        }

        Ok(embedding)
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<f64>> for LaplacianEigenmaps<LaplacianTrained> {
    fn transform(&self, _x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        // Laplacian Eigenmaps doesn't support transforming new data in this implementation
        Ok(self.state.embedding.clone())
    }
}

impl LaplacianEigenmaps<LaplacianTrained> {
    /// Get the embedding
    pub fn embedding(&self) -> &Array2<f64> {
        &self.state.embedding
    }

    /// Get the Laplacian matrix
    pub fn laplacian_matrix(&self) -> &Array2<f64> {
        &self.state.laplacian_matrix
    }

    /// Get the adjacency matrix
    pub fn adjacency_matrix(&self) -> &Array2<f64> {
        &self.state.adjacency_matrix
    }
}
