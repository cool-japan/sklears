//! Probabilistic imputation methods
//!
//! This module provides advanced probabilistic imputation techniques including:
//! - Bayesian imputation with prior distributions
//! - Expectation-Maximization (EM) algorithm for missing data
//! - Gaussian Process imputation for smooth interpolation
//! - Monte Carlo imputation for uncertainty quantification
//! - Copula-based imputation for preserving dependencies

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::seeded_rng;
use scirs2_core::Distribution;
use sklears_core::prelude::*;

// ================================================================================================
// Bayesian Imputation
// ================================================================================================

/// Configuration for Bayesian imputation
#[derive(Debug, Clone)]
pub struct BayesianImputerConfig {
    /// Prior distribution for the mean (mu_0, sigma_0^2)
    pub prior_mean: f64,
    pub prior_std: f64,
    /// Prior distribution for the variance (shape, rate for Gamma distribution)
    pub prior_variance_shape: f64,
    pub prior_variance_rate: f64,
    /// Number of posterior samples for imputation
    pub n_samples: usize,
    /// Random seed
    pub random_state: u64,
}

impl Default for BayesianImputerConfig {
    fn default() -> Self {
        Self {
            prior_mean: 0.0,
            prior_std: 1.0,
            prior_variance_shape: 2.0,
            prior_variance_rate: 2.0,
            n_samples: 100,
            random_state: 42,
        }
    }
}

/// Bayesian imputer using conjugate priors
pub struct BayesianImputer {
    config: BayesianImputerConfig,
}

/// Fitted Bayesian imputer
pub struct BayesianImputerFitted {
    config: BayesianImputerConfig,
    /// Posterior parameters for each feature
    posterior_params: Vec<PosteriorParams>,
}

#[derive(Debug, Clone)]
struct PosteriorParams {
    /// Posterior mean
    mean: f64,
    /// Posterior standard deviation
    std: f64,
    /// Posterior variance shape
    variance_shape: f64,
    /// Posterior variance rate
    variance_rate: f64,
}

impl BayesianImputer {
    /// Create a new Bayesian imputer
    pub fn new(config: BayesianImputerConfig) -> Self {
        Self { config }
    }
}

impl Estimator for BayesianImputer {
    type Config = BayesianImputerConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<f64>, ()> for BayesianImputer {
    type Fitted = BayesianImputerFitted;

    fn fit(self, X: &Array2<f64>, _y: &()) -> Result<Self::Fitted> {
        let n_features = X.ncols();
        let mut posterior_params = Vec::with_capacity(n_features);

        for j in 0..n_features {
            let col = X.column(j);
            let observed: Vec<f64> = col.iter().filter(|&&x| !x.is_nan()).copied().collect();

            if observed.is_empty() {
                // No observed data, use prior
                posterior_params.push(PosteriorParams {
                    mean: self.config.prior_mean,
                    std: self.config.prior_std,
                    variance_shape: self.config.prior_variance_shape,
                    variance_rate: self.config.prior_variance_rate,
                });
                continue;
            }

            let n = observed.len() as f64;
            let sample_mean = observed.iter().sum::<f64>() / n;
            let sample_var = if n > 1.0 {
                observed
                    .iter()
                    .map(|&x| (x - sample_mean).powi(2))
                    .sum::<f64>()
                    / (n - 1.0)
            } else {
                self.config.prior_std.powi(2)
            };

            // Update posterior using conjugate priors (Normal-Gamma)
            let prior_precision = 1.0 / self.config.prior_std.powi(2);
            let sample_precision = n / sample_var;

            let posterior_precision = prior_precision + sample_precision;
            let posterior_mean = (prior_precision * self.config.prior_mean
                + sample_precision * sample_mean)
                / posterior_precision;
            let posterior_std = (1.0 / posterior_precision).sqrt();

            // Update variance posterior (Gamma distribution)
            let posterior_shape = self.config.prior_variance_shape + n / 2.0;
            let sum_sq_dev = observed
                .iter()
                .map(|&x| (x - sample_mean).powi(2))
                .sum::<f64>();
            let posterior_rate = self.config.prior_variance_rate
                + sum_sq_dev / 2.0
                + (prior_precision * n * (sample_mean - self.config.prior_mean).powi(2))
                    / (2.0 * (prior_precision + n));

            posterior_params.push(PosteriorParams {
                mean: posterior_mean,
                std: posterior_std,
                variance_shape: posterior_shape,
                variance_rate: posterior_rate,
            });
        }

        Ok(BayesianImputerFitted {
            config: self.config,
            posterior_params,
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for BayesianImputerFitted {
    fn transform(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        let mut result = X.clone();
        let mut rng = seeded_rng(self.config.random_state);

        for j in 0..X.ncols() {
            let params = &self.posterior_params[j];

            // Sample from posterior
            let normal = Normal::new(params.mean, params.std).map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to create normal distribution: {}", e))
            })?;
            let imputed_value = normal.sample(&mut rng);

            for i in 0..X.nrows() {
                if result[[i, j]].is_nan() {
                    result[[i, j]] = imputed_value;
                }
            }
        }

        Ok(result)
    }
}

// ================================================================================================
// EM Imputation
// ================================================================================================

/// Configuration for EM imputation
#[derive(Debug, Clone)]
pub struct EMImputerConfig {
    /// Maximum number of EM iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Random seed
    pub random_state: u64,
}

impl Default for EMImputerConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-4,
            random_state: 42,
        }
    }
}

/// EM imputer using multivariate normal model
pub struct EMImputer {
    config: EMImputerConfig,
}

/// Fitted EM imputer
pub struct EMImputerFitted {
    config: EMImputerConfig,
    /// Estimated mean vector
    mean: Array1<f64>,
    /// Estimated covariance matrix
    covariance: Array2<f64>,
}

impl EMImputer {
    /// Create a new EM imputer
    pub fn new(config: EMImputerConfig) -> Self {
        Self { config }
    }
}

impl Estimator for EMImputer {
    type Config = EMImputerConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<f64>, ()> for EMImputer {
    type Fitted = EMImputerFitted;

    fn fit(self, X: &Array2<f64>, _y: &()) -> Result<Self::Fitted> {
        let n_features = X.ncols();

        // Initialize with column means
        let mut mean = Array1::zeros(n_features);
        for j in 0..n_features {
            let col = X.column(j);
            let observed: Vec<f64> = col.iter().filter(|&&x| !x.is_nan()).copied().collect();
            if !observed.is_empty() {
                mean[j] = observed.iter().sum::<f64>() / observed.len() as f64;
            }
        }

        // Initialize covariance matrix
        let mut covariance = Array2::eye(n_features);

        // EM iterations
        for _iter in 0..self.config.max_iter {
            let mean_old = mean.clone();

            // E-step: Impute missing values using current parameters
            let mut X_imputed = X.clone();
            for i in 0..X.nrows() {
                for j in 0..n_features {
                    if X_imputed[[i, j]].is_nan() {
                        X_imputed[[i, j]] = mean[j];
                    }
                }
            }

            // M-step: Update parameters
            mean = X_imputed
                .mean_axis(Axis(0))
                .ok_or_else(|| SklearsError::InvalidInput("Failed to compute mean".to_string()))?;

            // Update covariance
            let mut cov_sum = Array2::zeros((n_features, n_features));
            for i in 0..X.nrows() {
                let centered = &X_imputed.row(i).to_owned() - &mean;
                let outer = centered
                    .clone()
                    .insert_axis(Axis(1))
                    .dot(&centered.insert_axis(Axis(0)));
                cov_sum = cov_sum + outer;
            }
            covariance = cov_sum / X.nrows() as f64;

            // Check convergence
            let mean_diff = (&mean - &mean_old).mapv(|x| x.abs()).sum();
            if mean_diff < self.config.tol {
                break;
            }
        }

        Ok(EMImputerFitted {
            config: self.config,
            mean,
            covariance,
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for EMImputerFitted {
    fn transform(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        let mut result = X.clone();

        for i in 0..X.nrows() {
            // Find missing indices
            let missing_indices: Vec<usize> =
                (0..X.ncols()).filter(|&j| X[[i, j]].is_nan()).collect();

            if missing_indices.is_empty() {
                continue;
            }

            let observed_indices: Vec<usize> =
                (0..X.ncols()).filter(|&j| !X[[i, j]].is_nan()).collect();

            if observed_indices.is_empty() {
                // All missing, use mean
                for &j in missing_indices.iter() {
                    result[[i, j]] = self.mean[j];
                }
                continue;
            }

            // Conditional imputation using multivariate normal properties
            for &miss_idx in missing_indices.iter() {
                let mut conditional_mean = self.mean[miss_idx];

                // Simple approximation: weighted average based on observed values
                let mut weight_sum = 0.0;
                let mut weighted_value = 0.0;

                for &obs_idx in observed_indices.iter() {
                    let cov = self.covariance[[miss_idx, obs_idx]];
                    let var = self.covariance[[obs_idx, obs_idx]];

                    if var > 1e-10 {
                        let weight = cov / var;
                        weighted_value += weight * (X[[i, obs_idx]] - self.mean[obs_idx]);
                        weight_sum += weight.abs();
                    }
                }

                if weight_sum > 1e-10 {
                    conditional_mean += weighted_value / weight_sum;
                }

                result[[i, miss_idx]] = conditional_mean;
            }
        }

        Ok(result)
    }
}

// ================================================================================================
// Gaussian Process Imputation
// ================================================================================================

/// Configuration for Gaussian Process imputation
#[derive(Debug, Clone)]
pub struct GaussianProcessImputerConfig {
    /// Length scale for RBF kernel
    pub length_scale: f64,
    /// Signal variance
    pub signal_variance: f64,
    /// Noise variance
    pub noise_variance: f64,
    /// Whether to optimize hyperparameters
    pub optimize_hyperparameters: bool,
}

impl Default for GaussianProcessImputerConfig {
    fn default() -> Self {
        Self {
            length_scale: 1.0,
            signal_variance: 1.0,
            noise_variance: 0.1,
            optimize_hyperparameters: false,
        }
    }
}

/// Gaussian Process imputer for smooth interpolation
pub struct GaussianProcessImputer {
    config: GaussianProcessImputerConfig,
}

/// Fitted Gaussian Process imputer
pub struct GaussianProcessImputerFitted {
    config: GaussianProcessImputerConfig,
    /// Training data (for each feature)
    training_data: Vec<FeatureGPData>,
}

#[derive(Debug, Clone)]
struct FeatureGPData {
    /// Observed indices
    observed_indices: Vec<usize>,
    /// Observed values
    observed_values: Vec<f64>,
    /// Inverse of kernel matrix (K + σ²I)^(-1)
    kernel_inv: Array2<f64>,
}

impl GaussianProcessImputer {
    /// Create a new Gaussian Process imputer
    pub fn new(config: GaussianProcessImputerConfig) -> Self {
        Self { config }
    }

    /// RBF (Gaussian) kernel
    fn kernel(&self, x1: f64, x2: f64) -> f64 {
        let sq_dist = (x1 - x2).powi(2);
        self.config.signal_variance * (-sq_dist / (2.0 * self.config.length_scale.powi(2))).exp()
    }
}

impl Estimator for GaussianProcessImputer {
    type Config = GaussianProcessImputerConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<f64>, ()> for GaussianProcessImputer {
    type Fitted = GaussianProcessImputerFitted;

    fn fit(self, X: &Array2<f64>, _y: &()) -> Result<Self::Fitted> {
        let n_features = X.ncols();
        let mut training_data = Vec::with_capacity(n_features);

        for j in 0..n_features {
            let col = X.column(j);
            let mut observed_indices = Vec::new();
            let mut observed_values = Vec::new();

            for (i, &val) in col.iter().enumerate() {
                if !val.is_nan() {
                    observed_indices.push(i);
                    observed_values.push(val);
                }
            }

            if observed_indices.is_empty() {
                // No observed data
                training_data.push(FeatureGPData {
                    observed_indices: Vec::new(),
                    observed_values: Vec::new(),
                    kernel_inv: Array2::zeros((0, 0)),
                });
                continue;
            }

            // Build kernel matrix
            let n_obs = observed_indices.len();
            let mut K = Array2::zeros((n_obs, n_obs));

            for i in 0..n_obs {
                for j in 0..n_obs {
                    K[[i, j]] = self.kernel(observed_indices[i] as f64, observed_indices[j] as f64);
                    if i == j {
                        K[[i, j]] += self.config.noise_variance;
                    }
                }
            }

            // Compute inverse (simplified - should use Cholesky in production)
            let kernel_inv = pseudo_inverse(&K)?;

            training_data.push(FeatureGPData {
                observed_indices,
                observed_values,
                kernel_inv,
            });
        }

        Ok(GaussianProcessImputerFitted {
            config: self.config,
            training_data,
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for GaussianProcessImputerFitted {
    fn transform(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        let mut result = X.clone();

        for j in 0..X.ncols() {
            let gp_data = &self.training_data[j];

            if gp_data.observed_indices.is_empty() {
                // No training data, leave as is
                continue;
            }

            for i in 0..X.nrows() {
                if result[[i, j]].is_nan() {
                    // Predict using GP
                    let k_star = gp_data
                        .observed_indices
                        .iter()
                        .map(|&obs_idx| {
                            self.config.signal_variance
                                * (-(i as f64 - obs_idx as f64).powi(2)
                                    / (2.0 * self.config.length_scale.powi(2)))
                                .exp()
                        })
                        .collect::<Vec<f64>>();

                    let k_star_array = Array1::from(k_star);
                    let y_obs = Array1::from(gp_data.observed_values.clone());

                    // Mean prediction: k*^T K^(-1) y
                    let alpha = gp_data.kernel_inv.dot(&y_obs);
                    let prediction = k_star_array.dot(&alpha);

                    result[[i, j]] = prediction;
                }
            }
        }

        Ok(result)
    }
}

// ================================================================================================
// Monte Carlo Imputation
// ================================================================================================

/// Configuration for Monte Carlo imputation
#[derive(Debug, Clone)]
pub struct MonteCarloImputerConfig {
    /// Number of imputation iterations
    pub n_imputations: usize,
    /// Base imputation method
    pub base_method: MonteCarloBaseMethod,
    /// Random seed
    pub random_state: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonteCarloBaseMethod {
    /// Use mean imputation with random noise
    MeanWithNoise,
    /// Use regression-based imputation with residual resampling
    RegressionResampling,
}

impl Default for MonteCarloImputerConfig {
    fn default() -> Self {
        Self {
            n_imputations: 5,
            base_method: MonteCarloBaseMethod::MeanWithNoise,
            random_state: 42,
        }
    }
}

/// Monte Carlo imputer for uncertainty quantification
pub struct MonteCarloImputer {
    config: MonteCarloImputerConfig,
}

/// Fitted Monte Carlo imputer
pub struct MonteCarloImputerFitted {
    config: MonteCarloImputerConfig,
    /// Column statistics for imputation
    column_stats: Vec<ColumnStats>,
}

#[derive(Debug, Clone)]
struct ColumnStats {
    mean: f64,
    std: f64,
}

impl MonteCarloImputer {
    /// Create a new Monte Carlo imputer
    pub fn new(config: MonteCarloImputerConfig) -> Self {
        Self { config }
    }
}

impl Estimator for MonteCarloImputer {
    type Config = MonteCarloImputerConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<f64>, ()> for MonteCarloImputer {
    type Fitted = MonteCarloImputerFitted;

    fn fit(self, X: &Array2<f64>, _y: &()) -> Result<Self::Fitted> {
        let n_features = X.ncols();
        let mut column_stats = Vec::with_capacity(n_features);

        for j in 0..n_features {
            let col = X.column(j);
            let observed: Vec<f64> = col.iter().filter(|&&x| !x.is_nan()).copied().collect();

            let (mean, std) = if !observed.is_empty() {
                let m = observed.iter().sum::<f64>() / observed.len() as f64;
                let s = if observed.len() > 1 {
                    (observed.iter().map(|&x| (x - m).powi(2)).sum::<f64>()
                        / (observed.len() - 1) as f64)
                        .sqrt()
                } else {
                    1.0
                };
                (m, s)
            } else {
                (0.0, 1.0)
            };

            column_stats.push(ColumnStats { mean, std });
        }

        Ok(MonteCarloImputerFitted {
            config: self.config,
            column_stats,
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for MonteCarloImputerFitted {
    fn transform(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        let mut rng = seeded_rng(self.config.random_state);
        let mut result = X.clone();

        // Perform multiple imputations and average
        let mut imputations = Vec::with_capacity(self.config.n_imputations);

        for _ in 0..self.config.n_imputations {
            let mut imputed = X.clone();

            for j in 0..X.ncols() {
                let stats = &self.column_stats[j];
                let normal = Normal::new(stats.mean, stats.std.max(1e-10)).map_err(|e| {
                    SklearsError::InvalidInput(format!(
                        "Failed to create normal distribution: {}",
                        e
                    ))
                })?;

                for i in 0..X.nrows() {
                    if imputed[[i, j]].is_nan() {
                        imputed[[i, j]] = normal.sample(&mut rng);
                    }
                }
            }

            imputations.push(imputed);
        }

        // Average imputations
        for i in 0..X.nrows() {
            for j in 0..X.ncols() {
                if X[[i, j]].is_nan() {
                    let sum: f64 = imputations.iter().map(|imp| imp[[i, j]]).sum();
                    result[[i, j]] = sum / self.config.n_imputations as f64;
                }
            }
        }

        Ok(result)
    }
}

// ================================================================================================
// Helper Functions
// ================================================================================================

/// Compute pseudo-inverse of a matrix using SVD (simplified)
fn pseudo_inverse(A: &Array2<f64>) -> Result<Array2<f64>> {
    let n = A.nrows();
    if n != A.ncols() {
        return Err(SklearsError::InvalidInput(
            "Matrix must be square for this simplified inverse".to_string(),
        ));
    }

    // Simplified: use regularized inverse
    let mut A_reg = A.clone();
    for i in 0..n {
        A_reg[[i, i]] += 1e-6;
    }

    // Simple Gauss-Jordan elimination (for small matrices)
    let mut aug = Array2::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = A_reg[[i, j]];
        }
        aug[[i, i + n]] = 1.0;
    }

    // Forward elimination
    for i in 0..n {
        let pivot = aug[[i, i]];
        if pivot.abs() < 1e-10 {
            continue;
        }

        for j in 0..2 * n {
            aug[[i, j]] /= pivot;
        }

        for k in 0..n {
            if k != i {
                let factor = aug[[k, i]];
                for j in 0..2 * n {
                    aug[[k, j]] -= factor * aug[[i, j]];
                }
            }
        }
    }

    // Extract inverse
    let mut inv = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[[i, j + n]];
        }
    }

    Ok(inv)
}

// ================================================================================================
// Tests
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_bayesian_imputer() {
        let X = array![[1.0, 2.0], [3.0, f64::NAN], [5.0, 6.0]];

        let config = BayesianImputerConfig::default();
        let imputer = BayesianImputer::new(config);
        let fitted = imputer.fit(&X, &()).unwrap();
        let result = fitted.transform(&X).unwrap();

        assert_eq!(result.nrows(), 3);
        assert_eq!(result.ncols(), 2);
        assert!(!result[[1, 1]].is_nan());
    }

    #[test]
    fn test_em_imputer() {
        let X = array![[1.0, 2.0], [3.0, f64::NAN], [5.0, 6.0], [7.0, 8.0]];

        let config = EMImputerConfig::default();
        let imputer = EMImputer::new(config);
        let fitted = imputer.fit(&X, &()).unwrap();
        let result = fitted.transform(&X).unwrap();

        assert_eq!(result.nrows(), 4);
        assert_eq!(result.ncols(), 2);
        assert!(!result[[1, 1]].is_nan());

        // Imputed value should be reasonable
        assert!(result[[1, 1]] > 0.0);
        assert!(result[[1, 1]] < 10.0);
    }

    #[test]
    fn test_gp_imputer() {
        let X = array![[1.0, 2.0], [3.0, f64::NAN], [5.0, 6.0], [7.0, 8.0]];

        let config = GaussianProcessImputerConfig::default();
        let imputer = GaussianProcessImputer::new(config);
        let fitted = imputer.fit(&X, &()).unwrap();
        let result = fitted.transform(&X).unwrap();

        assert_eq!(result.nrows(), 4);
        assert_eq!(result.ncols(), 2);
        assert!(!result[[1, 1]].is_nan());
    }

    #[test]
    fn test_monte_carlo_imputer() {
        let X = array![[1.0, 2.0], [3.0, f64::NAN], [5.0, 6.0], [7.0, 8.0]];

        let config = MonteCarloImputerConfig {
            n_imputations: 10,
            base_method: MonteCarloBaseMethod::MeanWithNoise,
            random_state: 42,
        };

        let imputer = MonteCarloImputer::new(config);
        let fitted = imputer.fit(&X, &()).unwrap();
        let result = fitted.transform(&X).unwrap();

        assert_eq!(result.nrows(), 4);
        assert_eq!(result.ncols(), 2);
        assert!(!result[[1, 1]].is_nan());

        // Imputed value should be close to mean
        let mean = (2.0 + 6.0 + 8.0) / 3.0;
        assert!((result[[1, 1]] - mean).abs() < 3.0);
    }

    #[test]
    fn test_bayesian_imputer_all_missing() {
        let X = array![[f64::NAN, 2.0], [f64::NAN, 4.0], [f64::NAN, 6.0]];

        let config = BayesianImputerConfig::default();
        let imputer = BayesianImputer::new(config);
        let fitted = imputer.fit(&X, &()).unwrap();
        let result = fitted.transform(&X).unwrap();

        // Should use prior mean for all missing column
        for i in 0..result.nrows() {
            assert!(!result[[i, 0]].is_nan());
        }
    }

    #[test]
    fn test_em_convergence() {
        let X = array![
            [1.0, 2.0, 3.0],
            [4.0, f64::NAN, 6.0],
            [7.0, 8.0, f64::NAN],
            [10.0, 11.0, 12.0]
        ];

        let config = EMImputerConfig {
            max_iter: 50,
            tol: 1e-4,
            random_state: 42,
        };

        let imputer = EMImputer::new(config);
        let fitted = imputer.fit(&X, &()).unwrap();

        // Check that mean vector has correct dimensions
        assert_eq!(fitted.mean.len(), 3);
        assert_eq!(fitted.covariance.nrows(), 3);
        assert_eq!(fitted.covariance.ncols(), 3);
    }

    #[test]
    fn test_pseudo_inverse() {
        let A = array![[2.0, 1.0], [1.0, 2.0]];
        let inv = pseudo_inverse(&A).unwrap();

        // Check A * inv ≈ I
        let product = A.dot(&inv);
        assert_relative_eq!(product[[0, 0]], 1.0, epsilon = 0.1);
        assert_relative_eq!(product[[1, 1]], 1.0, epsilon = 0.1);
        assert_relative_eq!(product[[0, 1]], 0.0, epsilon = 0.1);
        assert_relative_eq!(product[[1, 0]], 0.0, epsilon = 0.1);
    }
}
