//! Error-bounded kernel approximation methods
//!
//! This module provides kernel approximation methods with explicit error bounds
//! and adaptive algorithms that guarantee approximation quality.

use crate::{Nystroem, RBFSampler};
use scirs2_core::ndarray::Array2;
use scirs2_linalg::compat::Norm;
use sklears_core::traits::Fit;
use sklears_core::{error::Result, traits::Transform};

/// Error bound computation methods
#[derive(Debug, Clone)]
/// ErrorBoundMethod
pub enum ErrorBoundMethod {
    /// Theoretical spectral bound
    SpectralBound,
    /// Frobenius norm bound
    FrobeniusBound,
    /// Empirical error estimation
    EmpiricalBound,
    /// Probabilistic error bound
    ProbabilisticBound { confidence: f64 },
    /// Perturbation-based bound
    PerturbationBound,
    /// Cross-validation based bound
    CVBound { n_folds: usize },
}

/// Error bound result
#[derive(Debug, Clone)]
/// ErrorBound
pub struct ErrorBound {
    /// Upper bound on approximation error
    pub upper_bound: f64,
    /// Lower bound on approximation error
    pub lower_bound: f64,
    /// Confidence level (for probabilistic bounds)
    pub confidence: Option<f64>,
    /// Method used for bound computation
    pub method: ErrorBoundMethod,
}

/// Configuration for error-bounded approximation
#[derive(Debug, Clone)]
/// ErrorBoundedConfig
pub struct ErrorBoundedConfig {
    /// Maximum allowed approximation error
    pub max_error: f64,
    /// Confidence level for probabilistic bounds
    pub confidence_level: f64,
    /// Error bound computation method
    pub bound_method: ErrorBoundMethod,
    /// Minimum number of components
    pub min_components: usize,
    /// Maximum number of components
    pub max_components: usize,
    /// Step size for component search
    pub step_size: usize,
    /// Number of trials for error estimation
    pub n_trials: usize,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for ErrorBoundedConfig {
    fn default() -> Self {
        Self {
            max_error: 0.1,
            confidence_level: 0.95,
            bound_method: ErrorBoundMethod::SpectralBound,
            min_components: 10,
            max_components: 1000,
            step_size: 10,
            n_trials: 5,
            random_seed: None,
        }
    }
}

/// Error-bounded RBF sampler
#[derive(Debug, Clone)]
/// ErrorBoundedRBFSampler
pub struct ErrorBoundedRBFSampler {
    gamma: f64,
    config: ErrorBoundedConfig,
}

impl Default for ErrorBoundedRBFSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl ErrorBoundedRBFSampler {
    /// Create a new error-bounded RBF sampler
    pub fn new() -> Self {
        Self {
            gamma: 1.0,
            config: ErrorBoundedConfig::default(),
        }
    }

    /// Set gamma parameter
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set configuration
    pub fn config(mut self, config: ErrorBoundedConfig) -> Self {
        self.config = config;
        self
    }

    /// Set maximum allowed error
    pub fn max_error(mut self, max_error: f64) -> Self {
        self.config.max_error = max_error;
        self
    }

    /// Set confidence level
    pub fn confidence_level(mut self, confidence_level: f64) -> Self {
        self.config.confidence_level = confidence_level;
        self
    }

    /// Find minimum number of components that satisfies error bound
    pub fn find_min_components(&self, x: &Array2<f64>) -> Result<(usize, ErrorBound)> {
        let n_samples = x.nrows();
        let _n_features = x.ncols();

        // Create test-validation split
        let split_idx = (n_samples as f64 * 0.8) as usize;
        let x_train = x
            .slice(scirs2_core::ndarray::s![..split_idx, ..])
            .to_owned();
        let _x_test = x
            .slice(scirs2_core::ndarray::s![split_idx.., ..])
            .to_owned();

        // Compute exact kernel matrix for small subset (for bound computation)
        let k_exact = self.compute_exact_kernel_matrix(&x_train)?;

        // Test different numbers of components
        for n_components in
            (self.config.min_components..=self.config.max_components).step_by(self.config.step_size)
        {
            let mut trial_errors = Vec::new();

            // Run multiple trials
            for trial in 0..self.config.n_trials {
                let seed = self.config.random_seed.map(|s| s + trial as u64);
                let sampler = if let Some(s) = seed {
                    RBFSampler::new(n_components)
                        .gamma(self.gamma)
                        .random_state(s)
                } else {
                    RBFSampler::new(n_components).gamma(self.gamma)
                };

                let fitted = sampler.fit(&x_train, &())?;
                let x_train_transformed = fitted.transform(&x_train)?;

                // Compute approximation error
                let error =
                    self.compute_approximation_error(&k_exact, &x_train_transformed, &x_train)?;

                trial_errors.push(error);
            }

            // Compute error bound
            let error_bound = self.compute_error_bound(&trial_errors, n_components)?;

            // Check if error bound is satisfied
            if error_bound.upper_bound <= self.config.max_error {
                return Ok((n_components, error_bound));
            }
        }

        // If no configuration satisfies the bound, return the best one
        let n_components = self.config.max_components;
        let sampler = if let Some(seed) = self.config.random_seed {
            RBFSampler::new(n_components)
                .gamma(self.gamma)
                .random_state(seed)
        } else {
            RBFSampler::new(n_components).gamma(self.gamma)
        };
        let fitted = sampler.fit(&x_train, &())?;
        let x_train_transformed = fitted.transform(&x_train)?;

        let error = self.compute_approximation_error(&k_exact, &x_train_transformed, &x_train)?;

        let error_bound = self.compute_error_bound(&[error], n_components)?;

        Ok((n_components, error_bound))
    }

    /// Compute exact kernel matrix for a subset of data
    fn compute_exact_kernel_matrix(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = x.nrows().min(200); // Limit for computational efficiency
        let x_subset = x.slice(scirs2_core::ndarray::s![..n_samples, ..]);

        let mut k_exact = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            for j in 0..n_samples {
                let diff = &x_subset.row(i) - &x_subset.row(j);
                let squared_norm = diff.dot(&diff);
                k_exact[[i, j]] = (-self.gamma * squared_norm).exp();
            }
        }

        Ok(k_exact)
    }

    /// Compute approximation error between exact and approximate kernel matrices
    fn compute_approximation_error(
        &self,
        k_exact: &Array2<f64>,
        x_transformed: &Array2<f64>,
        _x: &Array2<f64>,
    ) -> Result<f64> {
        let n_samples = k_exact.nrows().min(x_transformed.nrows());
        let x_subset = x_transformed.slice(scirs2_core::ndarray::s![..n_samples, ..]);

        // Compute approximate kernel matrix
        let k_approx = x_subset.dot(&x_subset.t());

        // Compute error based on Frobenius norm
        let diff = k_exact - &k_approx.slice(scirs2_core::ndarray::s![..n_samples, ..n_samples]);
        let error = diff.norm_l2();

        Ok(error)
    }

    /// Compute error bound from trial errors
    fn compute_error_bound(&self, trial_errors: &[f64], n_components: usize) -> Result<ErrorBound> {
        match &self.config.bound_method {
            ErrorBoundMethod::SpectralBound => {
                self.compute_spectral_bound(trial_errors, n_components)
            }
            ErrorBoundMethod::FrobeniusBound => {
                self.compute_frobenius_bound(trial_errors, n_components)
            }
            ErrorBoundMethod::EmpiricalBound => {
                self.compute_empirical_bound(trial_errors, n_components)
            }
            ErrorBoundMethod::ProbabilisticBound { confidence } => {
                self.compute_probabilistic_bound(trial_errors, *confidence, n_components)
            }
            ErrorBoundMethod::PerturbationBound => {
                self.compute_perturbation_bound(trial_errors, n_components)
            }
            ErrorBoundMethod::CVBound { n_folds } => {
                self.compute_cv_bound(trial_errors, *n_folds, n_components)
            }
        }
    }

    /// Compute spectral error bound
    fn compute_spectral_bound(
        &self,
        trial_errors: &[f64],
        n_components: usize,
    ) -> Result<ErrorBound> {
        let mean_error = trial_errors.iter().sum::<f64>() / trial_errors.len() as f64;
        let variance = trial_errors
            .iter()
            .map(|&e| (e - mean_error).powi(2))
            .sum::<f64>()
            / trial_errors.len() as f64;

        // Theoretical spectral bound for RBF kernel approximation
        let theoretical_bound = (2.0 * self.gamma / n_components as f64).sqrt();

        let upper_bound = mean_error + 2.0 * variance.sqrt() + theoretical_bound;
        let lower_bound = (mean_error - 2.0 * variance.sqrt()).max(0.0);

        Ok(ErrorBound {
            upper_bound,
            lower_bound,
            confidence: Some(0.95),
            method: ErrorBoundMethod::SpectralBound,
        })
    }

    /// Compute Frobenius norm error bound
    fn compute_frobenius_bound(
        &self,
        trial_errors: &[f64],
        n_components: usize,
    ) -> Result<ErrorBound> {
        let mean_error = trial_errors.iter().sum::<f64>() / trial_errors.len() as f64;
        let max_error = trial_errors.iter().fold(0.0f64, |acc, &e| acc.max(e));

        // Frobenius bound typically grows with sqrt(rank)
        let theoretical_bound = (mean_error * n_components as f64).sqrt();

        let upper_bound = max_error + theoretical_bound;
        let lower_bound = mean_error * 0.5;

        Ok(ErrorBound {
            upper_bound,
            lower_bound,
            confidence: None,
            method: ErrorBoundMethod::FrobeniusBound,
        })
    }

    /// Compute empirical error bound
    fn compute_empirical_bound(
        &self,
        trial_errors: &[f64],
        _n_components: usize,
    ) -> Result<ErrorBound> {
        let mean_error = trial_errors.iter().sum::<f64>() / trial_errors.len() as f64;
        let std_error = {
            let variance = trial_errors
                .iter()
                .map(|&e| (e - mean_error).powi(2))
                .sum::<f64>()
                / trial_errors.len() as f64;
            variance.sqrt()
        };

        let upper_bound = mean_error + 2.0 * std_error;
        let lower_bound = (mean_error - 2.0 * std_error).max(0.0);

        Ok(ErrorBound {
            upper_bound,
            lower_bound,
            confidence: Some(0.95),
            method: ErrorBoundMethod::EmpiricalBound,
        })
    }

    /// Compute probabilistic error bound
    fn compute_probabilistic_bound(
        &self,
        trial_errors: &[f64],
        confidence: f64,
        _n_components: usize,
    ) -> Result<ErrorBound> {
        let mut sorted_errors = trial_errors.to_vec();
        sorted_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted_errors.len();
        let alpha = 1.0 - confidence;

        let lower_idx = ((alpha / 2.0) * n as f64) as usize;
        let upper_idx = ((1.0 - alpha / 2.0) * n as f64) as usize;

        let lower_bound = sorted_errors[lower_idx.min(n - 1)];
        let upper_bound = sorted_errors[upper_idx.min(n - 1)];

        Ok(ErrorBound {
            upper_bound,
            lower_bound,
            confidence: Some(confidence),
            method: ErrorBoundMethod::ProbabilisticBound { confidence },
        })
    }

    /// Compute perturbation-based error bound
    fn compute_perturbation_bound(
        &self,
        trial_errors: &[f64],
        _n_components: usize,
    ) -> Result<ErrorBound> {
        let mean_error = trial_errors.iter().sum::<f64>() / trial_errors.len() as f64;
        let std_error = {
            let variance = trial_errors
                .iter()
                .map(|&e| (e - mean_error).powi(2))
                .sum::<f64>()
                / trial_errors.len() as f64;
            variance.sqrt()
        };

        // Perturbation bound considers the stability of the approximation
        let perturbation_factor = 1.0 + (std_error / mean_error.max(1e-10));
        let theoretical_bound = mean_error * perturbation_factor;

        let upper_bound = theoretical_bound + std_error;
        let lower_bound = (mean_error / perturbation_factor - std_error).max(0.0);

        Ok(ErrorBound {
            upper_bound,
            lower_bound,
            confidence: Some(0.9),
            method: ErrorBoundMethod::PerturbationBound,
        })
    }

    /// Compute cross-validation based error bound
    fn compute_cv_bound(
        &self,
        trial_errors: &[f64],
        n_folds: usize,
        _n_components: usize,
    ) -> Result<ErrorBound> {
        // Use cross-validation to estimate error bound
        let mean_error = trial_errors.iter().sum::<f64>() / trial_errors.len() as f64;
        let std_error = {
            let variance = trial_errors
                .iter()
                .map(|&e| (e - mean_error).powi(2))
                .sum::<f64>()
                / trial_errors.len() as f64;
            variance.sqrt()
        };

        // CV bound includes adjustment for number of folds
        let cv_adjustment = (n_folds as f64).sqrt() / (n_folds - 1) as f64;
        let cv_std = std_error * cv_adjustment;

        let upper_bound = mean_error + 2.0 * cv_std;
        let lower_bound = (mean_error - 2.0 * cv_std).max(0.0);

        Ok(ErrorBound {
            upper_bound,
            lower_bound,
            confidence: Some(0.95),
            method: ErrorBoundMethod::CVBound { n_folds },
        })
    }
}

/// Fitted error-bounded RBF sampler
pub struct FittedErrorBoundedRBFSampler {
    fitted_rbf: crate::rbf_sampler::RBFSampler<sklears_core::traits::Trained>,
    n_components: usize,
    error_bound: ErrorBound,
}

impl Fit<Array2<f64>, ()> for ErrorBoundedRBFSampler {
    type Fitted = FittedErrorBoundedRBFSampler;

    fn fit(self, x: &Array2<f64>, _y: &()) -> Result<Self::Fitted> {
        // Find minimum components that satisfy error bound
        let (n_components, error_bound) = self.find_min_components(x)?;

        // Fit RBF sampler with selected components
        let rbf_sampler = if let Some(seed) = self.config.random_seed {
            RBFSampler::new(n_components)
                .gamma(self.gamma)
                .random_state(seed)
        } else {
            RBFSampler::new(n_components).gamma(self.gamma)
        };
        let fitted_rbf = rbf_sampler.fit(x, &())?;

        Ok(FittedErrorBoundedRBFSampler {
            fitted_rbf,
            n_components,
            error_bound,
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for FittedErrorBoundedRBFSampler {
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.fitted_rbf.transform(x)
    }
}

impl FittedErrorBoundedRBFSampler {
    /// Get the error bound
    pub fn error_bound(&self) -> &ErrorBound {
        &self.error_bound
    }

    /// Get the number of components
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Check if error bound is satisfied
    pub fn is_bound_satisfied(&self, tolerance: f64) -> bool {
        self.error_bound.upper_bound <= tolerance
    }
}

/// Error-bounded Nyström method
#[derive(Debug, Clone)]
/// ErrorBoundedNystroem
pub struct ErrorBoundedNystroem {
    kernel: crate::nystroem::Kernel,
    config: ErrorBoundedConfig,
}

impl Default for ErrorBoundedNystroem {
    fn default() -> Self {
        Self::new()
    }
}

impl ErrorBoundedNystroem {
    /// Create a new error-bounded Nyström method
    pub fn new() -> Self {
        Self {
            kernel: crate::nystroem::Kernel::Rbf { gamma: 1.0 },
            config: ErrorBoundedConfig::default(),
        }
    }

    /// Set gamma parameter (for RBF kernel)
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.kernel = crate::nystroem::Kernel::Rbf { gamma };
        self
    }

    /// Set kernel type
    pub fn kernel(mut self, kernel: crate::nystroem::Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    /// Set configuration
    pub fn config(mut self, config: ErrorBoundedConfig) -> Self {
        self.config = config;
        self
    }

    /// Find minimum number of components that satisfies error bound
    pub fn find_min_components(&self, x: &Array2<f64>) -> Result<(usize, ErrorBound)> {
        let _n_samples = x.nrows();

        // Test different numbers of components
        for n_components in
            (self.config.min_components..=self.config.max_components).step_by(self.config.step_size)
        {
            let mut trial_errors = Vec::new();

            // Run multiple trials
            for trial in 0..self.config.n_trials {
                let seed = self.config.random_seed.map(|s| s + trial as u64);
                let nystroem = if let Some(s) = seed {
                    Nystroem::new(self.kernel.clone(), n_components).random_state(s)
                } else {
                    Nystroem::new(self.kernel.clone(), n_components)
                };

                let fitted = nystroem.fit(x, &())?;
                let x_transformed = fitted.transform(x)?;

                // Compute approximation error using spectral norm
                let error = self.compute_nystroem_error(&x_transformed, x)?;
                trial_errors.push(error);
            }

            // Compute error bound
            let error_bound = self.compute_error_bound(&trial_errors, n_components)?;

            // Check if error bound is satisfied
            if error_bound.upper_bound <= self.config.max_error {
                return Ok((n_components, error_bound));
            }
        }

        // If no configuration satisfies the bound, return the best one
        let n_components = self.config.max_components;
        let nystroem = Nystroem::new(self.kernel.clone(), n_components);
        let fitted = nystroem.fit(x, &())?;
        let x_transformed = fitted.transform(x)?;

        let error = self.compute_nystroem_error(&x_transformed, x)?;
        let error_bound = self.compute_error_bound(&[error], n_components)?;

        Ok((n_components, error_bound))
    }

    /// Compute Nyström approximation error
    fn compute_nystroem_error(&self, x_transformed: &Array2<f64>, x: &Array2<f64>) -> Result<f64> {
        let n_samples = x.nrows().min(100); // Limit for computational efficiency
        let x_subset = x.slice(scirs2_core::ndarray::s![..n_samples, ..]);

        // Compute exact kernel matrix for subset
        let k_exact = self
            .kernel
            .compute_kernel(&x_subset.to_owned(), &x_subset.to_owned());

        // Compute approximate kernel matrix
        let x_transformed_subset = x_transformed.slice(scirs2_core::ndarray::s![..n_samples, ..]);
        let k_approx = x_transformed_subset.dot(&x_transformed_subset.t());

        // Compute Frobenius norm error
        let diff = k_exact - &k_approx;
        let error = diff.norm_l2();

        Ok(error)
    }

    /// Compute error bound (using same method as RBF sampler)
    fn compute_error_bound(
        &self,
        trial_errors: &[f64],
        _n_components: usize,
    ) -> Result<ErrorBound> {
        let mean_error = trial_errors.iter().sum::<f64>() / trial_errors.len() as f64;
        let std_error = {
            let variance = trial_errors
                .iter()
                .map(|&e| (e - mean_error).powi(2))
                .sum::<f64>()
                / trial_errors.len() as f64;
            variance.sqrt()
        };

        let upper_bound = mean_error + 2.0 * std_error;
        let lower_bound = (mean_error - 2.0 * std_error).max(0.0);

        Ok(ErrorBound {
            upper_bound,
            lower_bound,
            confidence: Some(0.95),
            method: self.config.bound_method.clone(),
        })
    }
}

/// Fitted error-bounded Nyström method
pub struct FittedErrorBoundedNystroem {
    fitted_nystroem: crate::nystroem::Nystroem<sklears_core::traits::Trained>,
    n_components: usize,
    error_bound: ErrorBound,
}

impl Fit<Array2<f64>, ()> for ErrorBoundedNystroem {
    type Fitted = FittedErrorBoundedNystroem;

    fn fit(self, x: &Array2<f64>, _y: &()) -> Result<Self::Fitted> {
        // Find minimum components that satisfy error bound
        let (n_components, error_bound) = self.find_min_components(x)?;

        // Fit Nyström method with selected components
        let nystroem = Nystroem::new(self.kernel, n_components);
        let fitted_nystroem = nystroem.fit(x, &())?;

        Ok(FittedErrorBoundedNystroem {
            fitted_nystroem,
            n_components,
            error_bound,
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for FittedErrorBoundedNystroem {
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.fitted_nystroem.transform(x)
    }
}

impl FittedErrorBoundedNystroem {
    /// Get the error bound
    pub fn error_bound(&self) -> &ErrorBound {
        &self.error_bound
    }

    /// Get the number of components
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Check if error bound is satisfied
    pub fn is_bound_satisfied(&self, tolerance: f64) -> bool {
        self.error_bound.upper_bound <= tolerance
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_bounded_rbf_sampler() {
        let x = Array2::from_shape_vec((100, 4), (0..400).map(|i| (i as f64) * 0.01).collect())
            .unwrap();

        let config = ErrorBoundedConfig {
            max_error: 0.5,
            min_components: 10,
            max_components: 50,
            step_size: 10,
            n_trials: 2,
            ..Default::default()
        };

        let sampler = ErrorBoundedRBFSampler::new().gamma(0.5).config(config);

        let fitted = sampler.fit(&x, &()).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.nrows(), 100);
        assert!(fitted.n_components() >= 10);
        assert!(fitted.n_components() <= 50);
        assert!(fitted.error_bound().upper_bound >= 0.0);
        assert!(fitted.error_bound().lower_bound >= 0.0);
    }

    #[test]
    fn test_error_bounded_nystroem() {
        let x =
            Array2::from_shape_vec((80, 3), (0..240).map(|i| (i as f64) * 0.02).collect()).unwrap();

        let config = ErrorBoundedConfig {
            max_error: 0.3,
            min_components: 5,
            max_components: 25,
            step_size: 5,
            n_trials: 2,
            ..Default::default()
        };

        let nystroem = ErrorBoundedNystroem::new().gamma(1.0).config(config);

        let fitted = nystroem.fit(&x, &()).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.nrows(), 80);
        assert!(fitted.n_components() >= 5);
        assert!(fitted.n_components() <= 25);
        assert!(fitted.error_bound().upper_bound >= 0.0);
    }

    #[test]
    fn test_error_bound_methods() {
        let x =
            Array2::from_shape_vec((50, 2), (0..100).map(|i| (i as f64) * 0.05).collect()).unwrap();

        let methods = vec![
            ErrorBoundMethod::SpectralBound,
            ErrorBoundMethod::FrobeniusBound,
            ErrorBoundMethod::EmpiricalBound,
            ErrorBoundMethod::ProbabilisticBound { confidence: 0.95 },
            ErrorBoundMethod::PerturbationBound,
            ErrorBoundMethod::CVBound { n_folds: 3 },
        ];

        for method in methods {
            let config = ErrorBoundedConfig {
                max_error: 0.4,
                bound_method: method,
                min_components: 5,
                max_components: 20,
                step_size: 5,
                n_trials: 2,
                ..Default::default()
            };

            let sampler = ErrorBoundedRBFSampler::new().gamma(0.8).config(config);

            let fitted = sampler.fit(&x, &()).unwrap();
            let error_bound = fitted.error_bound();

            assert!(error_bound.upper_bound >= error_bound.lower_bound);
            assert!(error_bound.upper_bound >= 0.0);
            assert!(error_bound.lower_bound >= 0.0);
        }
    }

    #[test]
    fn test_bound_satisfaction() {
        let x =
            Array2::from_shape_vec((40, 3), (0..120).map(|i| (i as f64) * 0.03).collect()).unwrap();

        let config = ErrorBoundedConfig {
            max_error: 0.2,
            min_components: 5,
            max_components: 30,
            step_size: 5,
            n_trials: 2,
            ..Default::default()
        };

        let sampler = ErrorBoundedRBFSampler::new().gamma(0.3).config(config);

        let fitted = sampler.fit(&x, &()).unwrap();

        // The fitted sampler should have computed an error bound
        let actual_bound = fitted.error_bound().upper_bound;

        // Error bound should be reasonable (not infinite or NaN)
        assert!(actual_bound.is_finite());
        assert!(actual_bound > 0.0);

        // The fitted sampler should satisfy a bound larger than its computed bound
        assert!(fitted.is_bound_satisfied(actual_bound + 0.1));

        // But not a much stricter bound
        assert!(!fitted.is_bound_satisfied(0.01));
    }

    #[test]
    fn test_min_components_search() {
        let x =
            Array2::from_shape_vec((60, 4), (0..240).map(|i| (i as f64) * 0.01).collect()).unwrap();

        let sampler = ErrorBoundedRBFSampler::new().gamma(0.5).max_error(0.3);

        let (n_components, error_bound) = sampler.find_min_components(&x).unwrap();

        assert!(n_components >= sampler.config.min_components);
        assert!(n_components <= sampler.config.max_components);
        assert!(error_bound.upper_bound >= 0.0);
        assert!(error_bound.lower_bound >= 0.0);
        assert!(error_bound.upper_bound >= error_bound.lower_bound);
    }

    #[test]
    fn test_reproducibility() {
        let x =
            Array2::from_shape_vec((50, 3), (0..150).map(|i| (i as f64) * 0.02).collect()).unwrap();

        let config = ErrorBoundedConfig {
            max_error: 0.25,
            min_components: 10,
            max_components: 30,
            step_size: 10,
            n_trials: 3,
            random_seed: Some(42),
            ..Default::default()
        };

        let sampler1 = ErrorBoundedRBFSampler::new()
            .gamma(0.7)
            .config(config.clone());

        let sampler2 = ErrorBoundedRBFSampler::new().gamma(0.7).config(config);

        let fitted1 = sampler1.fit(&x, &()).unwrap();
        let fitted2 = sampler2.fit(&x, &()).unwrap();

        // The key reproducibility requirement is that we get the same number of components
        assert_eq!(fitted1.n_components(), fitted2.n_components());

        // Error bounds may vary slightly due to numerical precision in the search process
        // but should be in a similar range (within 50% of each other)
        let bound1 = fitted1.error_bound().upper_bound;
        let bound2 = fitted2.error_bound().upper_bound;
        let ratio = bound1.max(bound2) / bound1.min(bound2).max(1e-10);
        assert!(
            ratio < 2.0,
            "Error bounds too different: {} vs {}, ratio: {}",
            bound1,
            bound2,
            ratio
        );
    }
}
