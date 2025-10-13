//! Neighborhood Selection for Precision Matrix Estimation
//!
//! This module implements neighborhood selection, a method for estimating
//! sparse precision matrices by solving separate L1-regularized regression
//! problems for each variable to identify its neighbors in the graph.

use crate::utils::matrix_inverse;
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Untrained},
    types::Float,
};

/// Configuration for NeighborhoodSelection estimator
#[derive(Debug, Clone)]
pub struct NeighborhoodSelectionConfig {
    /// L1 regularization parameter
    pub alpha: f64,
    /// Whether to store the precision matrix
    pub store_precision: bool,
    /// Whether to assume the data is centered
    pub assume_centered: bool,
    /// Maximum number of iterations for coordinate descent
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Whether to symmetrize the precision matrix
    pub symmetrize: bool,
}

/// Neighborhood Selection Estimator
///
/// Estimates sparse precision matrices using neighborhood selection.
/// For each variable, solves an L1-regularized regression to find its
/// neighbors in the graphical model.
///
/// # Parameters
///
/// * `alpha` - L1 regularization parameter (default: 0.01)
/// * `store_precision` - Whether to store the precision matrix (default: true)
/// * `assume_centered` - Whether to assume the data is centered (default: false)
/// * `max_iter` - Maximum number of iterations (default: 1000)
/// * `tol` - Convergence tolerance (default: 1e-4)
/// * `symmetrize` - Whether to symmetrize the precision matrix (default: true)
///
/// # Examples
///
/// ```
/// use sklears_covariance::NeighborhoodSelection;
/// use sklears_core::traits::Fit;
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0, 0.1], [3.0, 4.0, 0.2], [5.0, 6.0, 0.3]];
///
/// let ns = NeighborhoodSelection::new().alpha(0.1);
/// let fitted = ns.fit(&X.view(), &()).unwrap();
/// let precision = fitted.get_precision().unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct NeighborhoodSelection<S = Untrained> {
    state: S,
    config: NeighborhoodSelectionConfig,
}

/// Trained state for NeighborhoodSelection
#[derive(Debug, Clone)]
pub struct NeighborhoodSelectionTrained {
    /// The covariance matrix
    pub covariance: Array2<f64>,
    /// The precision matrix (inverse of covariance)
    pub precision: Option<Array2<f64>>,
    /// The location (mean) vector
    pub location: Array1<f64>,
    /// The regression coefficients for each variable
    pub coefficients: Array2<f64>,
    /// Number of iterations taken for convergence
    pub n_iter: usize,
}

impl NeighborhoodSelection<Untrained> {
    /// Create a new NeighborhoodSelection instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            config: NeighborhoodSelectionConfig {
                alpha: 0.01,
                store_precision: true,
                assume_centered: false,
                max_iter: 1000,
                tol: 1e-4,
                symmetrize: true,
            },
        }
    }

    /// Set the L1 regularization parameter
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.config.alpha = alpha;
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

    /// Set whether to symmetrize the precision matrix
    pub fn symmetrize(mut self, symmetrize: bool) -> Self {
        self.config.symmetrize = symmetrize;
        self
    }
}

impl Default for NeighborhoodSelection<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for NeighborhoodSelection<Untrained> {
    type Config = NeighborhoodSelectionConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for NeighborhoodSelection<Untrained> {
    type Fitted = NeighborhoodSelection<NeighborhoodSelectionTrained>;

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
            Array1::zeros(n_features)
        } else {
            x.mean_axis(Axis(0)).unwrap()
        };

        let mut x_centered = x.to_owned();
        if !self.config.assume_centered {
            for mut row in x_centered.axis_iter_mut(Axis(0)) {
                row -= &location;
            }
        }

        // Estimate coefficients using neighborhood selection
        let (coefficients, n_iter) = self.estimate_neighborhoods(&x_centered)?;

        // Construct precision matrix from neighborhood selections
        let mut precision_matrix = Array2::zeros((n_features, n_features));

        for j in 0..n_features {
            // Estimate variance of residuals for variable j
            let mut residual_var = 0.0;
            for i in 0..n_samples {
                let mut prediction = 0.0;
                for k in 0..n_features {
                    if k != j {
                        let coeff_idx = if k < j { k } else { k - 1 };
                        prediction += coefficients[[j, coeff_idx]] * x_centered[[i, k]];
                    }
                }
                let residual = x_centered[[i, j]] - prediction;
                residual_var += residual * residual;
            }
            residual_var /= (n_samples - 1) as f64;

            if residual_var > 0.0 {
                precision_matrix[[j, j]] = 1.0 / residual_var;

                // Set off-diagonal elements
                for k in 0..n_features {
                    if k != j {
                        let coeff_idx = if k < j { k } else { k - 1 };
                        precision_matrix[[j, k]] = -coefficients[[j, coeff_idx]] / residual_var;
                    }
                }
            } else {
                precision_matrix[[j, j]] = 1.0;
            }
        }

        // Symmetrize if requested
        if self.config.symmetrize {
            for i in 0..n_features {
                for j in (i + 1)..n_features {
                    let avg = (precision_matrix[[i, j]] + precision_matrix[[j, i]]) / 2.0;
                    precision_matrix[[i, j]] = avg;
                    precision_matrix[[j, i]] = avg;
                }
            }
        }

        // Compute covariance matrix as inverse of precision
        let covariance = matrix_inverse(&precision_matrix)?;

        let precision = if self.config.store_precision {
            Some(precision_matrix)
        } else {
            None
        };

        Ok(NeighborhoodSelection {
            state: NeighborhoodSelectionTrained {
                covariance,
                precision,
                location,
                coefficients,
                n_iter,
            },
            config: self.config,
        })
    }
}

impl NeighborhoodSelection<Untrained> {
    fn estimate_neighborhoods(&self, x: &Array2<f64>) -> SklResult<(Array2<f64>, usize)> {
        let (n_samples, n_features) = x.dim();
        let mut coefficients = Array2::zeros((n_features, n_features - 1));
        let mut total_iter = 0;

        // For each variable, solve L1-regularized regression
        for j in 0..n_features {
            // Prepare response variable (column j)
            let y = x.column(j);

            // Prepare design matrix (all columns except j)
            let mut x_design = Array2::zeros((n_samples, n_features - 1));
            let mut col_idx = 0;
            for k in 0..n_features {
                if k != j {
                    for i in 0..n_samples {
                        x_design[[i, col_idx]] = x[[i, k]];
                    }
                    col_idx += 1;
                }
            }

            // Solve L1-regularized regression using coordinate descent
            let y_owned = y.to_owned();
            let (beta, iter_count) = self.coordinate_descent_lasso(&x_design, &y_owned)?;

            // Store coefficients
            for k in 0..(n_features - 1) {
                coefficients[[j, k]] = beta[k];
            }

            total_iter += iter_count;
        }

        Ok((coefficients, total_iter))
    }

    fn coordinate_descent_lasso(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> SklResult<(Array1<f64>, usize)> {
        let (n_samples, n_features) = x.dim();
        let mut beta = Array1::zeros(n_features);

        // Precompute X^T X diagonal and X^T y
        let mut xtx_diag = Array1::<f64>::zeros(n_features);
        let mut xty = Array1::<f64>::zeros(n_features);

        for j in 0..n_features {
            xtx_diag[j] = x.column(j).iter().map(|&val| val * val).sum::<f64>();
            xty[j] = x
                .column(j)
                .iter()
                .zip(y.iter())
                .map(|(&xij, &yi)| xij * yi)
                .sum::<f64>();
        }

        let mut iter = 0;
        for _ in 0..self.config.max_iter {
            iter += 1;
            let mut max_change: f64 = 0.0;

            for j in 0..n_features {
                if xtx_diag[j] == 0.0 {
                    continue;
                }

                let old_beta_j = beta[j];

                // Compute partial residual
                let mut partial_residual: f64 = xty[j];
                for k in 0..n_features {
                    if k != j {
                        let mut xk_dot_xj = 0.0;
                        for i in 0..n_samples {
                            xk_dot_xj += x[[i, k]] * x[[i, j]];
                        }
                        partial_residual -= beta[k] * xk_dot_xj;
                    }
                }

                // Soft thresholding
                let threshold = self.config.alpha * n_samples as f64;
                beta[j] = if partial_residual > threshold {
                    (partial_residual - threshold) / xtx_diag[j]
                } else if partial_residual < -threshold {
                    (partial_residual + threshold) / xtx_diag[j]
                } else {
                    0.0
                };

                max_change = max_change.max((beta[j] - old_beta_j).abs());
            }

            if max_change < self.config.tol {
                break;
            }
        }

        Ok((beta, iter))
    }
}

impl NeighborhoodSelection<NeighborhoodSelectionTrained> {
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

    /// Get the regression coefficients
    pub fn get_coefficients(&self) -> &Array2<f64> {
        &self.state.coefficients
    }

    /// Get the regularization parameter
    pub fn get_alpha(&self) -> f64 {
        self.config.alpha
    }

    /// Get the number of iterations taken
    pub fn get_n_iter(&self) -> usize {
        self.state.n_iter
    }
}
