//! Robust Principal Component Analysis (RPCA)
//!
//! This module implements Robust PCA using Principal Component Pursuit (PCP)
//! to decompose a matrix into low-rank and sparse components, useful for
//! outlier detection and noise removal in high-dimensional data.

use crate::utils::matrix_inverse;
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Untrained},
    types::Float,
};

/// Configuration for RobustPCA estimator
#[derive(Debug, Clone)]
pub struct RobustPCAConfig {
    /// Penalty parameter for low-rank component
    pub lambda: f64,
    /// Penalty parameter for sparse component  
    pub mu: f64,
    /// Whether to store the precision matrix
    pub store_precision: bool,
    /// Whether to assume the data is centered
    pub assume_centered: bool,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Whether to use automatic parameter selection
    pub auto_params: bool,
    /// Rank threshold for low-rank approximation
    pub rank_threshold: Option<usize>,
}

/// Robust Principal Component Analysis (RPCA) Estimator
///
/// Decomposes the input matrix into low-rank and sparse components using
/// Principal Component Pursuit (PCP). The method solves:
/// min ||L||_* + lambda * ||S||_1 subject to L + S = X
/// where L is low-rank and S is sparse.
///
/// # Parameters
///
/// * `lambda` - Penalty parameter for sparse component (default: auto)
/// * `mu` - Augmented Lagrangian parameter (default: auto)
/// * `store_precision` - Whether to store the precision matrix (default: true)
/// * `assume_centered` - Whether to assume the data is centered (default: false)
/// * `max_iter` - Maximum number of iterations (default: 1000)
/// * `tol` - Convergence tolerance (default: 1e-6)
/// * `auto_params` - Whether to automatically select parameters (default: true)
/// * `rank_threshold` - Rank threshold for low-rank component (default: None)
///
/// # Examples
///
/// ```
/// use sklears_covariance::RobustPCA;
///
/// let rpca = RobustPCA::new()
///     .lambda(0.1)
///     .max_iter(100)
///     .tol(1e-6);
/// ```
#[derive(Debug, Clone)]
pub struct RobustPCA<S = Untrained> {
    state: S,
    config: RobustPCAConfig,
}

/// Trained state for RobustPCA
#[derive(Debug, Clone)]
pub struct RobustPCATrained {
    /// The covariance matrix
    pub covariance: Array2<f64>,
    /// The precision matrix (inverse of covariance)
    pub precision: Option<Array2<f64>>,
    /// The location (mean) vector
    pub location: Array1<f64>,
    /// Low-rank component
    pub low_rank_component: Array2<f64>,
    /// Sparse component
    pub sparse_component: Array2<f64>,
    /// Singular values of low-rank component
    pub singular_values: Array1<f64>,
    /// Left singular vectors
    pub left_singular_vectors: Array2<f64>,
    /// Right singular vectors
    pub right_singular_vectors: Array2<f64>,
    /// Effective rank of low-rank component
    pub rank: usize,
    /// Number of iterations taken for convergence
    pub n_iter: usize,
}

impl RobustPCA<Untrained> {
    /// Create a new RobustPCA instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            config: RobustPCAConfig {
                lambda: 0.0, // Will be set automatically if auto_params = true
                mu: 0.0,     // Will be set automatically if auto_params = true
                store_precision: true,
                assume_centered: false,
                max_iter: 1000,
                tol: 1e-6,
                auto_params: true,
                rank_threshold: None,
            },
        }
    }

    /// Set the penalty parameter for sparse component
    pub fn lambda(mut self, lambda: f64) -> Self {
        self.config.lambda = lambda;
        self.config.auto_params = false;
        self
    }

    /// Set the augmented Lagrangian parameter
    pub fn mu(mut self, mu: f64) -> Self {
        self.config.mu = mu;
        self
    }

    /// Set whether to store the precision matrix
    pub fn store_precision(mut self, store_precision: bool) -> Self {
        self.config.store_precision = store_precision;
        self
    }

    /// Set whether to assume the data is centered
    pub fn assume_centered(mut self, assume_centered: bool) -> Self {
        self.config.assume_centered = assume_centered;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance
    pub fn tol(mut self, tol: f64) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set whether to automatically select parameters
    pub fn auto_params(mut self, auto_params: bool) -> Self {
        self.config.auto_params = auto_params;
        self
    }

    /// Set the rank threshold
    pub fn rank_threshold(mut self, rank_threshold: Option<usize>) -> Self {
        self.config.rank_threshold = rank_threshold;
        self
    }
}

impl Default for RobustPCA<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for RobustPCA<Untrained> {
    type Config = RobustPCAConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for RobustPCA<Untrained> {
    type Fitted = RobustPCA<RobustPCATrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let x = *x;
        let (n_samples, n_features) = x.dim();

        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 samples".to_string(),
            ));
        }

        if n_features < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 features".to_string(),
            ));
        }

        // Center the data
        let location = if self.config.assume_centered {
            Array1::<f64>::zeros(n_features)
        } else {
            x.mean_axis(Axis(0)).unwrap()
        };

        let mut x_centered = x.to_owned();
        if !self.config.assume_centered {
            for mut row in x_centered.axis_iter_mut(Axis(0)) {
                row -= &location;
            }
        }

        // Set automatic parameters if requested
        let lambda = if self.config.auto_params {
            1.0 / (n_samples.max(n_features) as f64).sqrt()
        } else {
            self.config.lambda
        };

        let mu = if self.config.auto_params {
            0.25 / self.compute_operator_norm(&x_centered)
        } else {
            self.config.mu
        };

        // Perform robust PCA decomposition
        let (low_rank, sparse, n_iter) =
            self.principal_component_pursuit(&x_centered, lambda, mu)?;

        // Compute SVD of low-rank component
        let (singular_values, left_vectors, right_vectors, rank) = self.compute_svd(&low_rank)?;

        // Compute covariance matrix from low-rank component
        let covariance = self.compute_covariance_from_low_rank(&low_rank, n_samples)?;

        // Compute precision matrix if requested
        let precision = if self.config.store_precision {
            Some(matrix_inverse(&covariance)?)
        } else {
            None
        };

        Ok(RobustPCA {
            state: RobustPCATrained {
                covariance,
                precision,
                location,
                low_rank_component: low_rank,
                sparse_component: sparse,
                singular_values,
                left_singular_vectors: left_vectors,
                right_singular_vectors: right_vectors,
                rank,
                n_iter,
            },
            config: RobustPCAConfig {
                lambda,
                mu,
                ..self.config
            },
        })
    }
}

impl RobustPCA<Untrained> {
    fn compute_operator_norm(&self, matrix: &Array2<f64>) -> f64 {
        // Compute the largest singular value (operator norm)
        // Using power iteration for simplicity
        let (m, n) = matrix.dim();
        let use_transpose = m < n;

        let target_matrix = if use_transpose {
            // Compute A^T * A
            let mut ata = Array2::<f64>::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    for k in 0..m {
                        ata[[i, j]] += matrix[[k, i]] * matrix[[k, j]];
                    }
                }
            }
            ata
        } else {
            // Compute A * A^T
            let mut aat = Array2::<f64>::zeros((m, m));
            for i in 0..m {
                for j in 0..m {
                    for k in 0..n {
                        aat[[i, j]] += matrix[[i, k]] * matrix[[j, k]];
                    }
                }
            }
            aat
        };

        let size = target_matrix.nrows();
        let mut x = Array1::ones(size);
        x /= (size as f64).sqrt();

        for _ in 0..50 {
            // Power iteration
            let mut new_x = Array1::<f64>::zeros(size);
            for i in 0..size {
                for j in 0..size {
                    new_x[i] += target_matrix[[i, j]] * x[j];
                }
            }

            let norm = new_x.iter().map(|&val| val * val).sum::<f64>().sqrt();
            if norm > 0.0 {
                x = new_x / norm;
            }
        }

        // Compute Rayleigh quotient
        let mut numerator: f64 = 0.0;
        for i in 0..size {
            for j in 0..size {
                numerator += x[i] * target_matrix[[i, j]] * x[j];
            }
        }

        numerator.sqrt()
    }

    fn principal_component_pursuit(
        &self,
        x: &Array2<f64>,
        lambda: f64,
        mu: f64,
    ) -> SklResult<(Array2<f64>, Array2<f64>, usize)> {
        let (m, n) = x.dim();
        let mut low_rank = Array2::<f64>::zeros((m, n));
        let mut sparse = Array2::<f64>::zeros((m, n));
        let mut lagrange_multiplier = Array2::<f64>::zeros((m, n));

        let rho = 1.1; // Penalty parameter update factor
        let mut current_mu = mu;

        let mut iter = 0;
        for _ in 0..self.config.max_iter {
            iter += 1;

            // Update low-rank component (SVD soft thresholding)
            let temp_l = x - &sparse + &lagrange_multiplier / current_mu;
            low_rank = self.svd_soft_threshold(&temp_l, 1.0 / current_mu)?;

            // Update sparse component (element-wise soft thresholding)
            let temp_s = x - &low_rank + &lagrange_multiplier / current_mu;
            sparse = self.soft_threshold(&temp_s, lambda / current_mu);

            // Update Lagrange multiplier
            let residual = x - &low_rank - &sparse;
            lagrange_multiplier = &lagrange_multiplier + current_mu * &residual;

            // Check convergence
            let residual_norm = residual.iter().map(|&val| val * val).sum::<f64>().sqrt();
            let data_norm = x.iter().map(|&val| val * val).sum::<f64>().sqrt();

            if residual_norm / data_norm < self.config.tol {
                break;
            }

            // Update penalty parameter
            current_mu *= rho;
        }

        Ok((low_rank, sparse, iter))
    }

    fn svd_soft_threshold(&self, matrix: &Array2<f64>, threshold: f64) -> SklResult<Array2<f64>> {
        let (u, sigma, vt) = self.compute_svd_full(matrix)?;

        // Apply soft thresholding to singular values
        let mut sigma_thresh = Array1::<f64>::zeros(sigma.len());
        for i in 0..sigma.len() {
            sigma_thresh[i] = (sigma[i] - threshold).max(0.0);
        }

        // Reconstruct matrix with thresholded singular values
        let (m, n) = matrix.dim();
        let mut result = Array2::<f64>::zeros((m, n));

        let rank = if let Some(r) = self.config.rank_threshold {
            r.min(sigma_thresh.len())
        } else {
            sigma_thresh.len()
        };

        for i in 0..rank {
            if sigma_thresh[i] > 0.0 {
                for j in 0..m {
                    for k in 0..n {
                        result[[j, k]] += u[[j, i]] * sigma_thresh[i] * vt[[i, k]];
                    }
                }
            }
        }

        Ok(result)
    }

    fn compute_svd_full(
        &self,
        matrix: &Array2<f64>,
    ) -> SklResult<(Array2<f64>, Array1<f64>, Array2<f64>)> {
        let (m, n) = matrix.dim();
        let min_dim = m.min(n);

        // Simplified SVD using eigendecomposition of A^T A or A A^T
        let use_small_side = m > n;

        if use_small_side {
            // Compute A^T A
            let mut ata = Array2::<f64>::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    for k in 0..m {
                        ata[[i, j]] += matrix[[k, i]] * matrix[[k, j]];
                    }
                }
            }

            // Find eigenvalues and eigenvectors of A^T A
            let (eigenvals, eigenvecs) = self.simple_eigendecomposition(&ata)?;

            // Singular values are sqrt of eigenvalues
            let mut sigma = Array1::<f64>::zeros(min_dim);
            for i in 0..min_dim {
                sigma[i] = eigenvals[i].max(0.0).sqrt();
            }

            // V = eigenvectors of A^T A
            let vt = eigenvecs.t().to_owned();

            // U = A * V / sigma
            let mut u = Array2::<f64>::zeros((m, min_dim));
            for i in 0..min_dim {
                if sigma[i] > 1e-10 {
                    for j in 0..m {
                        for k in 0..n {
                            u[[j, i]] += matrix[[j, k]] * eigenvecs[[k, i]] / sigma[i];
                        }
                    }
                }
            }

            Ok((u, sigma, vt))
        } else {
            // Compute A A^T
            let mut aat = Array2::<f64>::zeros((m, m));
            for i in 0..m {
                for j in 0..m {
                    for k in 0..n {
                        aat[[i, j]] += matrix[[i, k]] * matrix[[j, k]];
                    }
                }
            }

            // Find eigenvalues and eigenvectors of A A^T
            let (eigenvals, eigenvecs) = self.simple_eigendecomposition(&aat)?;

            // Singular values are sqrt of eigenvalues
            let mut sigma = Array1::<f64>::zeros(min_dim);
            for i in 0..min_dim {
                sigma[i] = eigenvals[i].max(0.0).sqrt();
            }

            // U = eigenvectors of A A^T
            let u = eigenvecs;

            // V = A^T * U / sigma
            let mut vt = Array2::<f64>::zeros((min_dim, n));
            for i in 0..min_dim {
                if sigma[i] > 1e-10 {
                    for j in 0..n {
                        for k in 0..m {
                            vt[[i, j]] += matrix[[k, j]] * u[[k, i]] / sigma[i];
                        }
                    }
                }
            }

            Ok((u, sigma, vt))
        }
    }

    fn simple_eigendecomposition(
        &self,
        matrix: &Array2<f64>,
    ) -> SklResult<(Array1<f64>, Array2<f64>)> {
        // Simplified eigendecomposition using power iteration for dominant eigenvalues
        let n = matrix.nrows();
        let mut eigenvalues = Array1::<f64>::zeros(n);
        let mut eigenvectors = Array2::eye(n);

        let mut remaining_matrix = matrix.clone();

        for i in 0..n {
            let (eigenval, eigenvec) = self.power_iteration_eigen(&remaining_matrix)?;
            eigenvalues[i] = eigenval;

            for j in 0..n {
                eigenvectors[[j, i]] = eigenvec[j];
            }

            // Deflate matrix
            for row in 0..n {
                for col in 0..n {
                    remaining_matrix[[row, col]] -= eigenval * eigenvec[row] * eigenvec[col];
                }
            }
        }

        Ok((eigenvalues, eigenvectors))
    }

    fn power_iteration_eigen(&self, matrix: &Array2<f64>) -> SklResult<(f64, Array1<f64>)> {
        let n = matrix.nrows();
        let mut x = Array1::ones(n);
        let mut lambda = 0.0;

        // Normalize
        let norm = x.iter().map(|&val| val * val).sum::<f64>().sqrt();
        if norm > 0.0 {
            x /= norm;
        }

        for _ in 0..100 {
            let mut ax = Array1::<f64>::zeros(n);
            for i in 0..n {
                for j in 0..n {
                    ax[i] += matrix[[i, j]] * x[j];
                }
            }

            let new_lambda: f64 = x.iter().zip(ax.iter()).map(|(&xi, &axi)| xi * axi).sum();

            let norm = ax.iter().map(|&val| val * val).sum::<f64>().sqrt();
            if norm < 1e-10 {
                break;
            }
            ax /= norm;

            if (new_lambda - lambda).abs() < 1e-8 {
                return Ok((new_lambda, ax));
            }

            lambda = new_lambda;
            x = ax;
        }

        Ok((lambda, x))
    }

    fn soft_threshold(&self, matrix: &Array2<f64>, threshold: f64) -> Array2<f64> {
        let mut result = matrix.clone();
        for val in result.iter_mut() {
            *val = if *val > threshold {
                *val - threshold
            } else if *val < -threshold {
                *val + threshold
            } else {
                0.0
            };
        }
        result
    }

    fn compute_svd(
        &self,
        matrix: &Array2<f64>,
    ) -> SklResult<(Array1<f64>, Array2<f64>, Array2<f64>, usize)> {
        let (u, sigma, vt) = self.compute_svd_full(matrix)?;

        // Count effective rank
        let rank = sigma.iter().filter(|&&s| s > 1e-10).count();

        Ok((sigma, u, vt, rank))
    }

    fn compute_covariance_from_low_rank(
        &self,
        low_rank: &Array2<f64>,
        n_samples: usize,
    ) -> SklResult<Array2<f64>> {
        let n_features = low_rank.ncols();
        let mut covariance = Array2::<f64>::zeros((n_features, n_features));

        // Compute sample covariance of low-rank component
        for i in 0..n_features {
            for j in 0..n_features {
                let mut cov_ij = 0.0;
                for k in 0..n_samples {
                    cov_ij += low_rank[[k, i]] * low_rank[[k, j]];
                }
                covariance[[i, j]] = cov_ij / (n_samples - 1) as f64;
            }
        }

        Ok(covariance)
    }
}

impl RobustPCA<RobustPCATrained> {
    /// Get the covariance matrix
    pub fn get_covariance(&self) -> &Array2<f64> {
        &self.state.covariance
    }

    /// Get the precision matrix (inverse covariance)
    pub fn get_precision(&self) -> Option<&Array2<f64>> {
        self.state.precision.as_ref()
    }

    /// Get the location (mean)
    pub fn get_location(&self) -> &Array1<f64> {
        &self.state.location
    }

    /// Get the low-rank component
    pub fn get_low_rank_component(&self) -> &Array2<f64> {
        &self.state.low_rank_component
    }

    /// Get the sparse component
    pub fn get_sparse_component(&self) -> &Array2<f64> {
        &self.state.sparse_component
    }

    /// Get the singular values
    pub fn get_singular_values(&self) -> &Array1<f64> {
        &self.state.singular_values
    }

    /// Get the left singular vectors
    pub fn get_left_singular_vectors(&self) -> &Array2<f64> {
        &self.state.left_singular_vectors
    }

    /// Get the right singular vectors
    pub fn get_right_singular_vectors(&self) -> &Array2<f64> {
        &self.state.right_singular_vectors
    }

    /// Get the effective rank
    pub fn get_rank(&self) -> usize {
        self.state.rank
    }

    /// Get the lambda parameter
    pub fn get_lambda(&self) -> f64 {
        self.config.lambda
    }

    /// Get the mu parameter
    pub fn get_mu(&self) -> f64 {
        self.config.mu
    }

    /// Get the number of iterations taken
    pub fn get_n_iter(&self) -> usize {
        self.state.n_iter
    }
}
