//! TIGER (Tuning-Insensitive Graph Estimation)
//!
//! This module implements TIGER, a method for graph estimation that is robust
//! to the choice of tuning parameters by using model averaging across multiple
//! penalty parameters, providing stable graph recovery.

use crate::empirical::EmpiricalCovariance;
use crate::utils::matrix_inverse;
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Untrained},
    types::Float,
};

/// Configuration for TIGER estimator
#[derive(Debug, Clone)]
pub struct TIGERConfig {
    /// Range of lambda values for model averaging
    pub lambda_range: (f64, f64),
    /// Number of lambda values to test
    pub n_lambdas: usize,
    /// Whether to store the precision matrix
    pub store_precision: bool,
    /// Whether to assume the data is centered
    pub assume_centered: bool,
    /// Maximum number of iterations for each lambda
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Stability threshold for edge selection
    pub stability_threshold: f64,
    /// Whether to use bootstrap for stability selection
    pub bootstrap: bool,
    /// Number of bootstrap samples
    pub n_bootstrap: usize,
}

/// TIGER (Tuning-Insensitive Graph Estimation) Estimator
///
/// Estimates sparse precision matrices using model averaging across multiple
/// penalty parameters, providing robust graph structure recovery that is
/// insensitive to the choice of tuning parameters.
///
/// # Parameters
///
/// * `lambda_range` - Range of penalty parameters (default: (0.01, 1.0))
/// * `n_lambdas` - Number of lambda values to test (default: 50)
/// * `store_precision` - Whether to store the precision matrix (default: true)
/// * `assume_centered` - Whether to assume the data is centered (default: false)
/// * `max_iter` - Maximum number of iterations per lambda (default: 1000)
/// * `tol` - Convergence tolerance (default: 1e-4)
/// * `stability_threshold` - Threshold for edge selection (default: 0.5)
/// * `bootstrap` - Whether to use bootstrap (default: true)
/// * `n_bootstrap` - Number of bootstrap samples (default: 100)
///
/// # Examples
///
/// ```
/// use sklears_covariance::TIGER;
/// use sklears_core::traits::Fit;
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [1.0, 2.0], [3.0, 1.0], [1.0, 3.0]];
///
/// let tiger = TIGER::new().stability_threshold(0.6);
/// let fitted = tiger.fit(&X.view(), &()).unwrap();
/// let precision = fitted.get_precision().unwrap();
/// let stability_matrix = fitted.get_stability_matrix();
/// ```
#[derive(Debug, Clone)]
pub struct TIGER<S = Untrained> {
    state: S,
    config: TIGERConfig,
}

/// Trained state for TIGER
#[derive(Debug, Clone)]
pub struct TIGERTrained {
    /// The covariance matrix
    pub covariance: Array2<f64>,
    /// The precision matrix (inverse of covariance)
    pub precision: Option<Array2<f64>>,
    /// The location (mean) vector
    pub location: Array1<f64>,
    /// Stability matrix (selection frequencies)
    pub stability_matrix: Array2<f64>,
    /// The selected graph structure (binary adjacency matrix)
    pub adjacency_matrix: Array2<f64>,
    /// Lambda values used
    pub lambda_values: Array1<f64>,
    /// Number of edges selected
    pub n_edges: usize,
}

impl TIGER<Untrained> {
    /// Create a new TIGER instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            config: TIGERConfig {
                lambda_range: (0.01, 1.0),
                n_lambdas: 50,
                store_precision: true,
                assume_centered: false,
                max_iter: 1000,
                tol: 1e-4,
                stability_threshold: 0.5,
                bootstrap: true,
                n_bootstrap: 100,
            },
        }
    }

    /// Set the range of lambda values
    pub fn lambda_range(mut self, range: (f64, f64)) -> Self {
        self.config.lambda_range = range;
        self
    }

    /// Set the number of lambda values
    pub fn n_lambdas(mut self, n_lambdas: usize) -> Self {
        self.config.n_lambdas = n_lambdas;
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

    /// Set the stability threshold
    pub fn stability_threshold(mut self, threshold: f64) -> Self {
        self.config.stability_threshold = threshold;
        self
    }

    /// Set whether to use bootstrap
    pub fn bootstrap(mut self, bootstrap: bool) -> Self {
        self.config.bootstrap = bootstrap;
        self
    }

    /// Set the number of bootstrap samples
    pub fn n_bootstrap(mut self, n_bootstrap: usize) -> Self {
        self.config.n_bootstrap = n_bootstrap;
        self
    }
}

impl Default for TIGER<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for TIGER<Untrained> {
    type Config = TIGERConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for TIGER<Untrained> {
    type Fitted = TIGER<TIGERTrained>;

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

        // Compute empirical covariance
        let emp_cov = EmpiricalCovariance::new()
            .assume_centered(self.config.assume_centered)
            .store_precision(false)
            .fit(&x, &())?;

        let covariance_emp = emp_cov.get_covariance().clone();
        let location = emp_cov.get_location().clone();

        // Generate lambda values
        let lambda_values = self.generate_lambda_values();

        // Compute stability matrix using model averaging
        let stability_matrix = if self.config.bootstrap {
            self.compute_stability_matrix_bootstrap(&x, &lambda_values)?
        } else {
            self.compute_stability_matrix_regular(&covariance_emp, &lambda_values)?
        };

        // Select edges based on stability threshold
        let adjacency_matrix = self.select_edges(&stability_matrix);

        // Estimate final precision matrix using selected edges
        let precision_matrix =
            self.estimate_precision_with_structure(&covariance_emp, &adjacency_matrix)?;

        // Compute final covariance matrix
        let covariance = matrix_inverse(&precision_matrix)?;

        // Count selected edges
        let n_edges = adjacency_matrix
            .iter()
            .enumerate()
            .filter(|(i, &val)| {
                let (row, col) = (i / n_features, i % n_features);
                row < col && val > 0.0
            })
            .count();

        let precision = if self.config.store_precision {
            Some(precision_matrix)
        } else {
            None
        };

        Ok(TIGER {
            state: TIGERTrained {
                covariance,
                precision,
                location,
                stability_matrix,
                adjacency_matrix,
                lambda_values,
                n_edges,
            },
            config: self.config,
        })
    }
}

impl TIGER<Untrained> {
    fn generate_lambda_values(&self) -> Array1<f64> {
        let (min_lambda, max_lambda) = self.config.lambda_range;
        let n = self.config.n_lambdas;

        let mut lambdas = Array1::zeros(n);
        for i in 0..n {
            let t = i as f64 / (n - 1) as f64;
            // Use log scale for better coverage
            let log_min = min_lambda.ln();
            let log_max = max_lambda.ln();
            lambdas[i] = (log_min + t * (log_max - log_min)).exp();
        }

        lambdas
    }

    fn compute_stability_matrix_bootstrap(
        &self,
        x: &ArrayView2<'_, Float>,
        lambda_values: &Array1<f64>,
    ) -> SklResult<Array2<f64>> {
        let (n_samples, n_features) = x.dim();
        let mut stability_matrix = Array2::zeros((n_features, n_features));

        for _boot in 0..self.config.n_bootstrap {
            // Generate bootstrap sample
            let boot_indices = self.generate_bootstrap_indices(n_samples);
            let mut x_boot = Array2::zeros((n_samples, n_features));

            for (i, &idx) in boot_indices.iter().enumerate() {
                for j in 0..n_features {
                    x_boot[[i, j]] = x[[idx, j]];
                }
            }

            // Compute empirical covariance on bootstrap sample
            let emp_cov = EmpiricalCovariance::new()
                .assume_centered(self.config.assume_centered)
                .store_precision(false)
                .fit(&x_boot.view(), &())?;

            let boot_cov = emp_cov.get_covariance();

            // Compute stability for this bootstrap sample
            let boot_stability = self.compute_stability_matrix_regular(boot_cov, lambda_values)?;

            // Accumulate stability scores
            for i in 0..n_features {
                for j in 0..n_features {
                    stability_matrix[[i, j]] += boot_stability[[i, j]];
                }
            }
        }

        // Average over bootstrap samples
        stability_matrix /= self.config.n_bootstrap as f64;

        Ok(stability_matrix)
    }

    fn generate_bootstrap_indices(&self, n_samples: usize) -> Vec<usize> {
        let mut indices = Vec::with_capacity(n_samples);

        // Simple pseudo-random generation for bootstrap indices
        let mut seed = 12345u64; // Fixed seed for reproducibility
        for _ in 0..n_samples {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let idx = (seed / 65536) % n_samples as u64;
            indices.push(idx as usize);
        }

        indices
    }

    fn compute_stability_matrix_regular(
        &self,
        covariance: &Array2<f64>,
        lambda_values: &Array1<f64>,
    ) -> SklResult<Array2<f64>> {
        let n_features = covariance.nrows();
        let mut stability_matrix = Array2::zeros((n_features, n_features));

        for &lambda in lambda_values {
            // Estimate precision matrix with this lambda
            let precision = self.estimate_precision_lasso(covariance, lambda)?;

            // Add to stability count (binary: edge present or not)
            for i in 0..n_features {
                for j in 0..n_features {
                    if i != j && precision[[i, j]].abs() > 1e-10 {
                        stability_matrix[[i, j]] += 1.0;
                    }
                }
            }
        }

        // Normalize by number of lambda values
        stability_matrix /= lambda_values.len() as f64;

        Ok(stability_matrix)
    }

    fn estimate_precision_lasso(
        &self,
        covariance: &Array2<f64>,
        lambda: f64,
    ) -> SklResult<Array2<f64>> {
        let n_features = covariance.nrows();
        let mut precision = matrix_inverse(covariance)?;

        // Simple graphical lasso-like iteration
        for _iter in 0..self.config.max_iter {
            let old_precision = precision.clone();

            for j in 0..n_features {
                // Update j-th row/column using coordinate descent
                for k in 0..n_features {
                    if j != k {
                        let mut sum_other = 0.0;
                        for l in 0..n_features {
                            if l != j && l != k {
                                sum_other += precision[[j, l]] * covariance[[l, k]];
                            }
                        }

                        let numerator = covariance[[j, k]] + sum_other;

                        // Soft thresholding
                        let threshold = lambda / covariance[[k, k]];
                        precision[[j, k]] = if numerator > threshold {
                            -(numerator - threshold) / covariance[[k, k]]
                        } else if numerator < -threshold {
                            -(numerator + threshold) / covariance[[k, k]]
                        } else {
                            0.0
                        };

                        // Maintain symmetry
                        precision[[k, j]] = precision[[j, k]];
                    }
                }
            }

            // Check convergence
            let mut max_change: f64 = 0.0;
            for i in 0..n_features {
                for j in 0..n_features {
                    max_change = max_change.max((precision[[i, j]] - old_precision[[i, j]]).abs());
                }
            }

            if max_change < self.config.tol {
                break;
            }
        }

        Ok(precision)
    }

    fn select_edges(&self, stability_matrix: &Array2<f64>) -> Array2<f64> {
        let n_features = stability_matrix.nrows();
        let mut adjacency = Array2::zeros((n_features, n_features));

        for i in 0..n_features {
            for j in 0..n_features {
                if i != j && stability_matrix[[i, j]] >= self.config.stability_threshold {
                    adjacency[[i, j]] = 1.0;
                }
            }
        }

        adjacency
    }

    fn estimate_precision_with_structure(
        &self,
        covariance: &Array2<f64>,
        adjacency: &Array2<f64>,
    ) -> SklResult<Array2<f64>> {
        let n_features = covariance.nrows();
        let mut precision = matrix_inverse(covariance)?;

        // Zero out elements not in the selected graph structure
        for i in 0..n_features {
            for j in 0..n_features {
                if i != j && adjacency[[i, j]] == 0.0 {
                    precision[[i, j]] = 0.0;
                }
            }
        }

        // Refine the precision matrix while maintaining sparsity structure
        for _iter in 0..50 {
            // Limited refinement iterations
            let old_precision = precision.clone();

            for i in 0..n_features {
                for j in 0..n_features {
                    if i != j && adjacency[[i, j]] > 0.0 {
                        // Update this element using first-order condition
                        let mut sum_term = 0.0;
                        for k in 0..n_features {
                            if k != j {
                                sum_term += precision[[i, k]] * covariance[[k, j]];
                            }
                        }

                        if covariance[[j, j]] > 0.0 {
                            precision[[i, j]] = -sum_term / covariance[[j, j]];
                        }
                    }
                }
            }

            // Check convergence
            let mut max_change: f64 = 0.0;
            for i in 0..n_features {
                for j in 0..n_features {
                    max_change = max_change.max((precision[[i, j]] - old_precision[[i, j]]).abs());
                }
            }

            if max_change < self.config.tol {
                break;
            }
        }

        Ok(precision)
    }
}

impl TIGER<TIGERTrained> {
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

    /// Get the stability matrix
    pub fn get_stability_matrix(&self) -> &Array2<f64> {
        &self.state.stability_matrix
    }

    /// Get the adjacency matrix (selected graph structure)
    pub fn get_adjacency_matrix(&self) -> &Array2<f64> {
        &self.state.adjacency_matrix
    }

    /// Get the lambda values used
    pub fn get_lambda_values(&self) -> &Array1<f64> {
        &self.state.lambda_values
    }

    /// Get the stability threshold
    pub fn get_stability_threshold(&self) -> f64 {
        self.config.stability_threshold
    }

    /// Get the number of selected edges
    pub fn get_n_edges(&self) -> usize {
        self.state.n_edges
    }
}
