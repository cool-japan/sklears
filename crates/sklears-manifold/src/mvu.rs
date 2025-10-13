//! Maximum Variance Unfolding (MVU) implementation
//!
//! This module provides MVU for non-linear dimensionality reduction through variance maximization.

use scirs2_core::ndarray::ndarray_linalg::{Eigh, UPLO};
use scirs2_core::ndarray::{Array2, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};

/// Maximum Variance Unfolding (MVU)
///
/// MVU is a dimensionality reduction method that seeks to unfold the manifold
/// by maximizing the variance of the data while preserving local distances.
/// It formulates the problem as a semidefinite programming (SDP) optimization
/// that maximizes the trace of the kernel matrix subject to distance constraints.
///
/// # Parameters
///
/// * `n_components` - Number of dimensions in the embedded space
/// * `n_neighbors` - Number of nearest neighbors to consider for local structure
/// * `tol` - Tolerance for optimization convergence
/// * `max_iter` - Maximum number of iterations for optimization
/// * `regularization` - Regularization parameter for numerical stability
///
/// # Examples
///
/// ```
/// use sklears_manifold::MVU;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]];
///
/// let mvu = MVU::new()
///     .n_components(2)
///     .n_neighbors(2);
/// let fitted = mvu.fit(&x.view(), &()).unwrap();
/// let embedded = fitted.transform(&x.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MVU<S = Untrained> {
    state: S,
    n_components: usize,
    n_neighbors: usize,
    tol: f64,
    max_iter: usize,
    regularization: f64,
}

impl MVU<Untrained> {
    /// Create a new MVU instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 2,
            n_neighbors: 5,
            tol: 1e-6,
            max_iter: 100,
            regularization: 1e-12,
        }
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the number of neighbors
    pub fn n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self
    }

    /// Set the tolerance
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the regularization parameter
    pub fn regularization(mut self, regularization: f64) -> Self {
        self.regularization = regularization;
        self
    }
}

impl Default for MVU<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for MVU<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for MVU<Untrained> {
    type Fitted = MVU<MvuTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples < self.n_neighbors {
            return Err(SklearsError::InvalidParameter {
                name: "n_neighbors".to_string(),
                reason: format!(
                    "must be less than or equal to n_samples ({}), got {}",
                    n_samples, self.n_neighbors
                ),
            });
        }

        if self.n_components >= n_features {
            return Err(SklearsError::InvalidParameter {
                name: "n_components".to_string(),
                reason: format!(
                    "must be less than n_features ({}), got {}",
                    n_features, self.n_components
                ),
            });
        }

        // Convert to f64 for computation
        let x_f64 = x.mapv(|v| v);

        // Find k-nearest neighbors for each point
        let neighbors = self.find_k_nearest_neighbors(&x_f64)?;

        // Construct the kernel matrix using MVU optimization
        let kernel_matrix = self.construct_kernel_matrix(&x_f64, &neighbors)?;

        // Perform eigendecomposition to get embedding
        let embedding = self.compute_embedding(&kernel_matrix)?;

        Ok(MVU {
            state: MvuTrained {
                embedding: embedding.mapv(|v| v as Float),
                kernel_matrix,
                neighbors,
            },
            n_components: self.n_components,
            n_neighbors: self.n_neighbors,
            tol: self.tol,
            max_iter: self.max_iter,
            regularization: self.regularization,
        })
    }
}

impl MVU<Untrained> {
    fn find_k_nearest_neighbors(&self, x: &Array2<f64>) -> SklResult<Vec<Vec<usize>>> {
        let n_samples = x.nrows();
        let mut neighbors = vec![Vec::new(); n_samples];

        for i in 0..n_samples {
            let mut distances: Vec<(f64, usize)> = Vec::new();

            for j in 0..n_samples {
                if i != j {
                    let dist = (&x.row(i) - &x.row(j)).mapv(|v| v * v).sum().sqrt();
                    distances.push((dist, j));
                }
            }

            // Sort by distance and take k nearest
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            neighbors[i] = distances
                .iter()
                .take(self.n_neighbors)
                .map(|(_, idx)| *idx)
                .collect();
        }

        Ok(neighbors)
    }

    fn construct_kernel_matrix(
        &self,
        x: &Array2<f64>,
        neighbors: &[Vec<usize>],
    ) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows();
        let mut kernel = Array2::zeros((n_samples, n_samples));

        // Initialize kernel matrix with inner products
        for i in 0..n_samples {
            for j in 0..n_samples {
                kernel[[i, j]] = x.row(i).dot(&x.row(j));
            }
        }

        // Apply distance constraints using iterative optimization
        for iter in 0..self.max_iter {
            let mut kernel_new = kernel.clone();
            let mut max_change: f64 = 0.0;

            // Update kernel matrix to satisfy local distance constraints
            for i in 0..n_samples {
                for &j in &neighbors[i] {
                    if i != j {
                        // Compute original distance
                        let orig_dist_sq = (&x.row(i) - &x.row(j)).mapv(|v| v * v).sum();

                        // Current distance in kernel space
                        let curr_dist_sq = kernel[[i, i]] + kernel[[j, j]] - 2.0 * kernel[[i, j]];

                        // Adjust kernel entries to preserve local distances
                        let adjustment = (orig_dist_sq - curr_dist_sq) * 0.1;
                        kernel_new[[i, j]] += adjustment;
                        kernel_new[[j, i]] += adjustment;

                        max_change = max_change.max(adjustment.abs());
                    }
                }
            }

            // Ensure positive semidefiniteness by adding regularization
            for i in 0..n_samples {
                kernel_new[[i, i]] += self.regularization;
            }

            kernel = kernel_new;

            if max_change < self.tol {
                break;
            }
        }

        Ok(kernel)
    }

    fn compute_embedding(&self, kernel: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = kernel.nrows();

        // Center the kernel matrix
        let row_means = kernel.mean_axis(Axis(1)).unwrap();
        let col_means = kernel.mean_axis(Axis(0)).unwrap();
        let total_mean = kernel.mean().unwrap();

        let mut centered_kernel = kernel.clone();
        for i in 0..n_samples {
            for j in 0..n_samples {
                centered_kernel[[i, j]] = kernel[[i, j]] - row_means[i] - col_means[j] + total_mean;
            }
        }

        // Compute eigendecomposition
        let (eigenvalues, eigenvectors) = centered_kernel.eigh(UPLO::Lower).map_err(|e| {
            SklearsError::NumericalError(format!("Eigendecomposition failed: {}", e))
        })?;

        // Select the largest eigenvalues and corresponding eigenvectors
        let mut sorted_indices: Vec<usize> = (0..eigenvalues.len()).collect();
        sorted_indices.sort_by(|&i, &j| eigenvalues[j].partial_cmp(&eigenvalues[i]).unwrap());

        let mut embedding = Array2::zeros((n_samples, self.n_components));
        for (comp, &idx) in sorted_indices.iter().take(self.n_components).enumerate() {
            let eigenval = eigenvalues[idx];
            if eigenval > 0.0 {
                let sqrt_eigenval = eigenval.sqrt();
                for i in 0..n_samples {
                    embedding[[i, comp]] = eigenvectors[[i, idx]] * sqrt_eigenval;
                }
            }
        }

        Ok(embedding)
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>> for MVU<MvuTrained> {
    fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let (n_samples, _) = x.dim();

        // For MVU, we can only transform the original training data
        // Out-of-sample extension is not directly supported in standard MVU
        if n_samples != self.state.embedding.nrows() {
            return Err(SklearsError::InvalidParameter {
                name: "input_data".to_string(),
                reason: "MVU does not support out-of-sample extensions. Input must be the same as training data.".to_string()
            });
        }

        Ok(self.state.embedding.clone())
    }
}

impl MVU<MvuTrained> {
    /// Get the embedding
    pub fn embedding(&self) -> &Array2<Float> {
        &self.state.embedding
    }

    /// Get the kernel matrix
    pub fn kernel_matrix(&self) -> &Array2<f64> {
        &self.state.kernel_matrix
    }

    /// Get the neighbors
    pub fn neighbors(&self) -> &[Vec<usize>] {
        &self.state.neighbors
    }
}

/// Trained state for MVU
#[derive(Debug, Clone)]
pub struct MvuTrained {
    /// The low-dimensional embedding of the training data
    pub embedding: Array2<Float>,
    /// The kernel matrix constructed during training
    pub kernel_matrix: Array2<f64>,
    /// The k-nearest neighbors for each point
    pub neighbors: Vec<Vec<usize>>,
}
