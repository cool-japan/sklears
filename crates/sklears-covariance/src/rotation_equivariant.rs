//! Rotation-Equivariant Shrinkage Covariance Estimator
//!
//! A covariance estimator that maintains rotation invariance properties
//! while applying eigenvalue-dependent shrinkage for robust estimation.

use crate::empirical::EmpiricalCovariance;
use crate::utils::matrix_inverse;
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Untrained},
    types::Float,
};

/// Configuration for RotationEquivariant estimator
#[derive(Debug, Clone)]
pub struct RotationEquivariantConfig {
    /// Whether to store the precision matrix
    pub store_precision: bool,
    /// Whether to assume the data is centered
    pub assume_centered: bool,
    /// Minimum shrinkage threshold
    pub min_shrinkage: f64,
    /// Maximum shrinkage threshold
    pub max_shrinkage: f64,
    /// Tolerance for eigenvalue computation
    pub tol: f64,
}

/// Rotation-Equivariant Shrinkage Covariance Estimator
///
/// This estimator maintains rotation invariance by applying eigenvalue-dependent
/// shrinkage that preserves the eigenvector structure while regularizing
/// eigenvalues based on their reliability.
///
/// # Parameters
///
/// * `store_precision` - Whether to store the precision matrix
/// * `assume_centered` - Whether to assume the data is centered
/// * `min_shrinkage` - Minimum shrinkage threshold (default: 0.0)
/// * `max_shrinkage` - Maximum shrinkage threshold (default: 1.0)
/// * `tol` - Tolerance for eigenvalue computation (default: 1e-8)
///
/// # Examples
///
/// ```
/// use sklears_covariance::RotationEquivariant;
/// use sklears_core::traits::Fit;
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [3.0, 1.0], [2.0, 3.0], [4.0, 2.0]];
///
/// let re = RotationEquivariant::new();
/// let fitted = re.fit(&X.view(), &()).unwrap();
/// let covariance = fitted.get_covariance();
/// let eigenvalues = fitted.get_eigenvalues();
/// ```
#[derive(Debug, Clone)]
pub struct RotationEquivariant<S = Untrained> {
    state: S,
    config: RotationEquivariantConfig,
}

/// Trained state for RotationEquivariant
#[derive(Debug, Clone)]
pub struct RotationEquivariantTrained {
    /// The covariance matrix
    pub covariance: Array2<f64>,
    /// The precision matrix (inverse of covariance)
    pub precision: Option<Array2<f64>>,
    /// The location (mean) vector
    pub location: Array1<f64>,
    /// The eigenvalues of the covariance matrix
    pub eigenvalues: Array1<f64>,
    /// The eigenvectors of the covariance matrix
    pub eigenvectors: Array2<f64>,
    /// The shrinkage parameters applied to each eigenvalue
    pub shrinkage_params: Array1<f64>,
}

impl RotationEquivariant<Untrained> {
    /// Create a new RotationEquivariant instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            config: RotationEquivariantConfig {
                store_precision: true,
                assume_centered: false,
                min_shrinkage: 0.0,
                max_shrinkage: 1.0,
                tol: 1e-8,
            },
        }
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

    /// Set the minimum shrinkage threshold
    pub fn min_shrinkage(mut self, min_shrinkage: f64) -> Self {
        self.config.min_shrinkage = min_shrinkage;
        self
    }

    /// Set the maximum shrinkage threshold
    pub fn max_shrinkage(mut self, max_shrinkage: f64) -> Self {
        self.config.max_shrinkage = max_shrinkage;
        self
    }

    /// Set the tolerance for eigenvalue computation
    pub fn tol(mut self, tol: f64) -> Self {
        self.config.tol = tol;
        self
    }
}

impl Default for RotationEquivariant<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for RotationEquivariant<Untrained> {
    type Config = RotationEquivariantConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for RotationEquivariant<Untrained> {
    type Fitted = RotationEquivariant<RotationEquivariantTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let x = *x;
        let (n_samples, n_features) = x.dim();

        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 samples".to_string(),
            ));
        }

        // Compute empirical covariance
        let emp_cov = EmpiricalCovariance::new()
            .assume_centered(self.config.assume_centered)
            .store_precision(false)
            .fit(&x, &())?;

        let covariance_emp = emp_cov.get_covariance().clone();
        let location = emp_cov.get_location().clone();

        // Eigenvalue decomposition of empirical covariance
        let (eigenvalues, eigenvectors) = self.eigenvalue_decomposition(&covariance_emp)?;

        // Compute rotation-equivariant shrinkage parameters
        let shrinkage_params = self.compute_rotation_equivariant_shrinkage(
            &x,
            &eigenvalues,
            &eigenvectors,
            &location,
            n_samples,
        )?;

        // Apply eigenvalue-dependent shrinkage
        let mut shrunken_eigenvalues = Array1::zeros(n_features);
        for i in 0..n_features {
            let shrink =
                shrinkage_params[i].clamp(self.config.min_shrinkage, self.config.max_shrinkage);
            let target_eigenvalue = eigenvalues.mean().unwrap_or(1.0);
            shrunken_eigenvalues[i] = (1.0 - shrink) * eigenvalues[i] + shrink * target_eigenvalue;
        }

        // Reconstruct covariance matrix: U * D * U^T
        let mut covariance = Array2::zeros((n_features, n_features));
        for i in 0..n_features {
            for j in 0..n_features {
                for k in 0..n_features {
                    covariance[[i, j]] +=
                        eigenvectors[[i, k]] * shrunken_eigenvalues[k] * eigenvectors[[j, k]];
                }
            }
        }

        // Compute precision matrix if requested
        let precision = if self.config.store_precision {
            Some(matrix_inverse(&covariance)?)
        } else {
            None
        };

        Ok(RotationEquivariant {
            state: RotationEquivariantTrained {
                covariance,
                precision,
                location,
                eigenvalues,
                eigenvectors,
                shrinkage_params,
            },
            config: self.config,
        })
    }
}

impl RotationEquivariant<Untrained> {
    fn eigenvalue_decomposition(
        &self,
        matrix: &Array2<f64>,
    ) -> SklResult<(Array1<f64>, Array2<f64>)> {
        let n = matrix.nrows();

        // Simplified eigenvalue decomposition using power iteration for dominant eigenvalues
        // This is a basic implementation - in practice, you'd use LAPACK/BLAS
        let mut eigenvalues = Array1::zeros(n);
        let mut eigenvectors = Array2::eye(n);

        // For each eigenvalue, use power iteration
        let mut remaining_matrix = matrix.clone();

        for i in 0..n {
            let (eigenval, eigenvec) = self.power_iteration(&remaining_matrix)?;
            eigenvalues[i] = eigenval;

            // Store eigenvector
            for j in 0..n {
                eigenvectors[[j, i]] = eigenvec[j];
            }

            // Deflate matrix by removing the found eigenvalue/eigenvector
            for row in 0..n {
                for col in 0..n {
                    remaining_matrix[[row, col]] -= eigenval * eigenvec[row] * eigenvec[col];
                }
            }
        }

        Ok((eigenvalues, eigenvectors))
    }

    fn power_iteration(&self, matrix: &Array2<f64>) -> SklResult<(f64, Array1<f64>)> {
        let n = matrix.nrows();
        let mut x = Array1::ones(n);
        let mut lambda = 0.0;

        // Normalize initial vector
        let norm = x.iter().map(|&val| val * val).sum::<f64>().sqrt();
        if norm > 0.0 {
            x /= norm;
        }

        for _ in 0..100 {
            // Maximum iterations
            // Compute A * x
            let mut ax = Array1::zeros(n);
            for i in 0..n {
                for j in 0..n {
                    ax[i] += matrix[[i, j]] * x[j];
                }
            }

            // Compute Rayleigh quotient: x^T * A * x
            let new_lambda: f64 = x.iter().zip(ax.iter()).map(|(&xi, &axi)| xi * axi).sum();

            // Normalize
            let norm = ax.iter().map(|&val| val * val).sum::<f64>().sqrt();
            if norm < self.config.tol {
                break;
            }
            ax /= norm;

            // Check convergence
            if (new_lambda - lambda).abs() < self.config.tol {
                return Ok((new_lambda, ax));
            }

            lambda = new_lambda;
            x = ax;
        }

        Ok((lambda, x))
    }

    fn compute_rotation_equivariant_shrinkage(
        &self,
        x: &ArrayView2<'_, Float>,
        eigenvalues: &Array1<f64>,
        eigenvectors: &Array2<f64>,
        location: &Array1<f64>,
        n_samples: usize,
    ) -> SklResult<Array1<f64>> {
        let n_features = eigenvalues.len();
        let mut shrinkage_params = Array1::zeros(n_features);

        // Center the data
        let mut x_centered = x.to_owned();
        if !self.config.assume_centered {
            for mut row in x_centered.axis_iter_mut(Axis(0)) {
                row -= location;
            }
        }

        // Transform data to eigenspace
        let mut x_eigen = Array2::zeros((n_samples, n_features));
        for i in 0..n_samples {
            for j in 0..n_features {
                for k in 0..n_features {
                    x_eigen[[i, j]] += x_centered[[i, k]] * eigenvectors[[k, j]];
                }
            }
        }

        // Compute shrinkage for each eigenvalue based on estimation quality
        for i in 0..n_features {
            let lambda_i = eigenvalues[i];

            // Compute sample variance of the i-th principal component
            let mut sample_var = 0.0;
            let mean_i = x_eigen.column(i).mean().unwrap_or(0.0);
            for j in 0..n_samples {
                let dev = x_eigen[[j, i]] - mean_i;
                sample_var += dev * dev;
            }
            sample_var /= (n_samples - 1) as f64;

            // Compute theoretical variance for this eigenvalue
            let theoretical_var = 2.0 * lambda_i * lambda_i / n_samples as f64;

            // Shrinkage based on variance ratio and eigenvalue magnitude
            let variance_ratio = if sample_var > 0.0 {
                theoretical_var / sample_var
            } else {
                1.0
            };

            // Additional shrinkage for small eigenvalues (less reliable)
            let eigenvalue_factor = if lambda_i > 0.0 {
                1.0 / (1.0 + lambda_i / eigenvalues.mean().unwrap_or(1.0))
            } else {
                1.0
            };

            shrinkage_params[i] = (variance_ratio * eigenvalue_factor).clamp(0.0, 1.0);
        }

        Ok(shrinkage_params)
    }
}

impl RotationEquivariant<RotationEquivariantTrained> {
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

    /// Get the eigenvalues
    pub fn get_eigenvalues(&self) -> &Array1<f64> {
        &self.state.eigenvalues
    }

    /// Get the eigenvectors
    pub fn get_eigenvectors(&self) -> &Array2<f64> {
        &self.state.eigenvectors
    }

    /// Get the shrinkage parameters
    pub fn get_shrinkage_params(&self) -> &Array1<f64> {
        &self.state.shrinkage_params
    }
}
