//! Bias-variance decomposition analysis for model performance understanding
//!
//! This module provides tools for performing bias-variance decomposition of model predictions,
//! which helps understand the sources of generalization error. The decomposition separates
//! the expected test error into three components:
//! - Bias²: Error due to overly simplistic assumptions
//! - Variance: Error due to sensitivity to small fluctuations in training data
//! - Noise: Irreducible error inherent in the problem

use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict},
};
use std::fmt::{self, Display, Formatter};

/// Results of bias-variance decomposition analysis
#[derive(Debug, Clone)]
pub struct BiasVarianceResult {
    /// Bias component (squared)
    pub bias_squared: f64,
    /// Variance component
    pub variance: f64,
    /// Noise component (irreducible error)
    pub noise: f64,
    /// Total expected error
    pub expected_error: f64,
    /// Standard error of bias estimate
    pub bias_std_error: f64,
    /// Standard error of variance estimate
    pub variance_std_error: f64,
    /// Number of bootstrap samples used
    pub n_bootstrap: usize,
    /// Sample-wise bias and variance estimates
    pub sample_wise_results: Vec<SampleBiasVariance>,
}

/// Bias-variance results for individual test samples
#[derive(Debug, Clone)]
pub struct SampleBiasVariance {
    /// Sample index
    pub sample_index: usize,
    /// True target value
    pub true_value: f64,
    /// Mean prediction across bootstrap samples
    pub mean_prediction: f64,
    /// Variance of predictions across bootstrap samples
    pub prediction_variance: f64,
    /// Squared bias for this sample
    pub squared_bias: f64,
    /// Individual predictions from each bootstrap sample
    pub predictions: Vec<f64>,
}

impl Display for BiasVarianceResult {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Bias-Variance Decomposition Results:\n\
             Expected Error: {:.6}\n\
             Bias²: {:.6} (SE: {:.6})\n\
             Variance: {:.6} (SE: {:.6})\n\
             Noise: {:.6}\n\
             Bootstrap Samples: {}",
            self.expected_error,
            self.bias_squared,
            self.bias_std_error,
            self.variance,
            self.variance_std_error,
            self.noise,
            self.n_bootstrap
        )
    }
}

/// Configuration for bias-variance decomposition
#[derive(Debug, Clone)]
pub struct BiasVarianceConfig {
    /// Number of bootstrap samples to generate
    pub n_bootstrap: usize,
    /// Fraction of original dataset to sample in each bootstrap
    pub sample_fraction: f64,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Whether to use sampling with replacement
    pub with_replacement: bool,
    /// Whether to compute sample-wise decomposition
    pub compute_sample_wise: bool,
}

impl Default for BiasVarianceConfig {
    fn default() -> Self {
        Self {
            n_bootstrap: 100,
            sample_fraction: 1.0,
            random_seed: None,
            with_replacement: true,
            compute_sample_wise: true,
        }
    }
}

/// Bias-variance decomposition analyzer
pub struct BiasVarianceAnalyzer {
    config: BiasVarianceConfig,
}

impl BiasVarianceAnalyzer {
    /// Create a new bias-variance analyzer with default configuration
    pub fn new() -> Self {
        Self {
            config: BiasVarianceConfig::default(),
        }
    }

    /// Create a new bias-variance analyzer with custom configuration
    pub fn with_config(config: BiasVarianceConfig) -> Self {
        Self { config }
    }

    /// Set the number of bootstrap samples
    pub fn n_bootstrap(mut self, n_bootstrap: usize) -> Self {
        self.config.n_bootstrap = n_bootstrap;
        self
    }

    /// Set the sample fraction for bootstrap sampling
    pub fn sample_fraction(mut self, fraction: f64) -> Self {
        self.config.sample_fraction = fraction;
        self
    }

    /// Set random seed for reproducibility
    pub fn random_seed(mut self, seed: u64) -> Self {
        self.config.random_seed = Some(seed);
        self
    }

    /// Enable or disable sampling with replacement
    pub fn with_replacement(mut self, with_replacement: bool) -> Self {
        self.config.with_replacement = with_replacement;
        self
    }

    /// Enable or disable sample-wise computation
    pub fn compute_sample_wise(mut self, compute: bool) -> Self {
        self.config.compute_sample_wise = compute;
        self
    }

    /// Perform bias-variance decomposition
    pub fn decompose<E, X, Y>(
        &self,
        estimator: &E,
        x_train: &[X],
        y_train: &[Y],
        x_test: &[X],
        y_test: &[Y],
    ) -> Result<BiasVarianceResult>
    where
        E: Estimator + Fit<Vec<X>, Vec<Y>> + Clone,
        E::Fitted: Predict<Vec<X>, Vec<f64>>,
        X: Clone,
        Y: Clone + Into<f64>,
    {
        if self.config.n_bootstrap == 0 {
            return Err(SklearsError::InvalidParameter {
                name: "n_bootstrap".to_string(),
                reason: "must be > 0".to_string(),
            });
        }

        if self.config.sample_fraction <= 0.0 || self.config.sample_fraction > 1.0 {
            return Err(SklearsError::InvalidParameter {
                name: "sample_fraction".to_string(),
                reason: "must be in (0, 1]".to_string(),
            });
        }

        let mut rng = self.get_rng();
        let n_train = x_train.len();
        let n_test = x_test.len();
        let sample_size = (n_train as f64 * self.config.sample_fraction) as usize;

        // Convert y_test to f64 values
        let y_test_f64: Vec<f64> = y_test.iter().map(|y| y.clone().into()).collect();

        // Store predictions from each bootstrap sample
        let mut all_predictions = Vec::with_capacity(self.config.n_bootstrap);

        // Generate bootstrap samples and train models
        for _ in 0..self.config.n_bootstrap {
            // Create bootstrap sample
            let (x_boot, y_boot) =
                self.bootstrap_sample(x_train, y_train, sample_size, &mut rng)?;

            // Train model on bootstrap sample
            let trained_model = estimator.clone().fit(&x_boot, &y_boot)?;

            // Make predictions on test set
            let x_test_vec: Vec<X> = x_test.to_vec();
            let predictions = trained_model.predict(&x_test_vec)?;
            all_predictions.push(predictions);
        }

        // Compute bias-variance decomposition
        self.compute_decomposition(&all_predictions, &y_test_f64)
    }

    /// Generate a bootstrap sample
    fn bootstrap_sample<X, Y>(
        &self,
        x_train: &[X],
        y_train: &[Y],
        sample_size: usize,
        rng: &mut impl scirs2_core::random::Rng,
    ) -> Result<(Vec<X>, Vec<Y>)>
    where
        X: Clone,
        Y: Clone,
    {
        let n_train = x_train.len();
        let mut x_boot = Vec::with_capacity(sample_size);
        let mut y_boot = Vec::with_capacity(sample_size);

        if self.config.with_replacement {
            // Sample with replacement
            for _ in 0..sample_size {
                let idx = rng.gen_range(0..n_train);
                x_boot.push(x_train[idx].clone());
                y_boot.push(y_train[idx].clone());
            }
        } else {
            // Sample without replacement
            let mut indices: Vec<usize> = (0..n_train).collect();
            indices.shuffle(rng);
            indices.truncate(sample_size);

            for &idx in &indices {
                x_boot.push(x_train[idx].clone());
                y_boot.push(y_train[idx].clone());
            }
        }

        Ok((x_boot, y_boot))
    }

    /// Compute bias-variance decomposition from predictions
    fn compute_decomposition(
        &self,
        all_predictions: &[Vec<f64>],
        y_test: &[f64],
    ) -> Result<BiasVarianceResult> {
        let n_test = y_test.len();
        let n_bootstrap = all_predictions.len();

        if n_bootstrap == 0 {
            return Err(SklearsError::InvalidParameter {
                name: "predictions".to_string(),
                reason: "no bootstrap predictions provided".to_string(),
            });
        }

        if all_predictions.iter().any(|p| p.len() != n_test) {
            return Err(SklearsError::InvalidParameter {
                name: "predictions".to_string(),
                reason: "all prediction arrays must have same length as test set".to_string(),
            });
        }

        let mut sample_wise_results = Vec::new();
        let mut total_bias_squared = 0.0;
        let mut total_variance = 0.0;
        let mut bias_estimates = Vec::new();
        let mut variance_estimates = Vec::new();

        // Compute bias and variance for each test sample
        for i in 0..n_test {
            let true_value = y_test[i];
            let predictions: Vec<f64> = all_predictions.iter().map(|p| p[i]).collect();

            // Mean prediction across bootstrap samples
            let mean_prediction = predictions.iter().sum::<f64>() / n_bootstrap as f64;

            // Variance of predictions
            let prediction_variance = predictions
                .iter()
                .map(|&p| (p - mean_prediction).powi(2))
                .sum::<f64>()
                / n_bootstrap as f64;

            // Squared bias
            let squared_bias = (mean_prediction - true_value).powi(2);

            total_bias_squared += squared_bias;
            total_variance += prediction_variance;

            bias_estimates.push(squared_bias);
            variance_estimates.push(prediction_variance);

            if self.config.compute_sample_wise {
                sample_wise_results.push(SampleBiasVariance {
                    sample_index: i,
                    true_value,
                    mean_prediction,
                    prediction_variance,
                    squared_bias,
                    predictions,
                });
            }
        }

        // Average across all test samples
        let bias_squared = total_bias_squared / n_test as f64;
        let variance = total_variance / n_test as f64;

        // Compute standard errors
        let bias_std_error = self.compute_standard_error(&bias_estimates);
        let variance_std_error = self.compute_standard_error(&variance_estimates);

        // Estimate noise (irreducible error) as the minimum achievable error
        // This is approximated as the variance in y_test if available, or set to 0
        let noise = self.estimate_noise(y_test);

        let expected_error = bias_squared + variance + noise;

        Ok(BiasVarianceResult {
            bias_squared,
            variance,
            noise,
            expected_error,
            bias_std_error,
            variance_std_error,
            n_bootstrap,
            sample_wise_results,
        })
    }

    /// Compute standard error of estimates
    fn compute_standard_error(&self, estimates: &[f64]) -> f64 {
        let n = estimates.len() as f64;
        let mean = estimates.iter().sum::<f64>() / n;
        let variance = estimates.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        (variance / n).sqrt()
    }

    /// Estimate noise component (irreducible error)
    fn estimate_noise(&self, _y_test: &[f64]) -> f64 {
        // In practice, noise estimation requires multiple observations of the same input
        // For simplicity, we assume noise is 0 here, but in real applications,
        // this could be estimated from repeated measurements or domain knowledge
        0.0
    }

    /// Get random number generator
    fn get_rng(&self) -> impl scirs2_core::random::Rng {
        use scirs2_core::random::rngs::StdRng;
        use scirs2_core::random::SeedableRng;

        match self.config.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                use scirs2_core::random::thread_rng;
                StdRng::from_rng(&mut thread_rng())
            }
        }
    }
}

impl Default for BiasVarianceAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function for performing bias-variance decomposition
pub fn bias_variance_decompose<E, X, Y>(
    estimator: &E,
    x_train: &[X],
    y_train: &[Y],
    x_test: &[X],
    y_test: &[Y],
    n_bootstrap: Option<usize>,
) -> Result<BiasVarianceResult>
where
    E: Estimator + Fit<Vec<X>, Vec<Y>> + Clone,
    E::Fitted: Predict<Vec<X>, Vec<f64>>,
    X: Clone,
    Y: Clone + Into<f64>,
{
    let mut analyzer = BiasVarianceAnalyzer::new();
    if let Some(n) = n_bootstrap {
        analyzer = analyzer.n_bootstrap(n);
    }
    analyzer.decompose(estimator, x_train, y_train, x_test, y_test)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    // Mock estimator for testing
    #[derive(Clone)]
    struct MockEstimator {
        noise_level: f64,
    }

    struct MockTrained {
        noise_level: f64,
    }

    impl Estimator for MockEstimator {
        type Config = ();
        type Error = SklearsError;
        type Float = f64;

        fn config(&self) -> &Self::Config {
            &()
        }
    }

    impl Fit<Vec<f64>, Vec<f64>> for MockEstimator {
        type Fitted = MockTrained;

        fn fit(self, _x: &Vec<f64>, _y: &Vec<f64>) -> Result<Self::Fitted> {
            Ok(MockTrained {
                noise_level: self.noise_level,
            })
        }
    }

    impl Predict<Vec<f64>, Vec<f64>> for MockTrained {
        fn predict(&self, x: &Vec<f64>) -> Result<Vec<f64>> {
            use scirs2_core::random::Rng;
            let mut rng = scirs2_core::random::thread_rng();
            Ok(x.iter()
                .map(|&xi| xi + rng.random_range(-self.noise_level..self.noise_level))
                .collect())
        }
    }

    #[test]
    fn test_bias_variance_analyzer_creation() {
        let analyzer = BiasVarianceAnalyzer::new();
        assert_eq!(analyzer.config.n_bootstrap, 100);
        assert_eq!(analyzer.config.sample_fraction, 1.0);
        assert!(analyzer.config.random_seed.is_none());
        assert!(analyzer.config.with_replacement);
        assert!(analyzer.config.compute_sample_wise);
    }

    #[test]
    fn test_bias_variance_configuration() {
        let analyzer = BiasVarianceAnalyzer::new()
            .n_bootstrap(50)
            .sample_fraction(0.8)
            .random_seed(42)
            .with_replacement(false)
            .compute_sample_wise(false);

        assert_eq!(analyzer.config.n_bootstrap, 50);
        assert_eq!(analyzer.config.sample_fraction, 0.8);
        assert_eq!(analyzer.config.random_seed, Some(42));
        assert!(!analyzer.config.with_replacement);
        assert!(!analyzer.config.compute_sample_wise);
    }

    #[test]
    fn test_bias_variance_decomposition() {
        let estimator = MockEstimator { noise_level: 0.1 };
        let x_train: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let y_train: Vec<f64> = x_train.iter().map(|&x| x * 2.0 + 1.0).collect();
        let x_test: Vec<f64> = (0..20).map(|i| i as f64 * 0.1 + 10.0).collect();
        let y_test: Vec<f64> = x_test.iter().map(|&x| x * 2.0 + 1.0).collect();

        let analyzer = BiasVarianceAnalyzer::new().n_bootstrap(10).random_seed(42);

        let result = analyzer.decompose(&estimator, &x_train, &y_train, &x_test, &y_test);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.n_bootstrap, 10);
        assert!(result.bias_squared >= 0.0);
        assert!(result.variance >= 0.0);
        assert_eq!(result.noise, 0.0); // Our mock noise estimation returns 0
        assert_eq!(
            result.expected_error,
            result.bias_squared + result.variance + result.noise
        );
        assert_eq!(result.sample_wise_results.len(), x_test.len());
    }

    #[test]
    fn test_invalid_parameters() {
        let analyzer = BiasVarianceAnalyzer::new().n_bootstrap(0);
        let estimator = MockEstimator { noise_level: 0.1 };
        let x_train = vec![1.0, 2.0, 3.0];
        let y_train = vec![1.0, 2.0, 3.0];
        let x_test = vec![4.0, 5.0];
        let y_test = vec![4.0, 5.0];

        let result = analyzer.decompose(&estimator, &x_train, &y_train, &x_test, &y_test);
        assert!(result.is_err());
    }

    #[test]
    fn test_convenience_function() {
        let estimator = MockEstimator { noise_level: 0.05 };
        let x_train: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
        let y_train: Vec<f64> = x_train.iter().map(|&x| x + 0.5).collect();
        let x_test: Vec<f64> = (0..10).map(|i| i as f64 * 0.1 + 5.0).collect();
        let y_test: Vec<f64> = x_test.iter().map(|&x| x + 0.5).collect();

        let result =
            bias_variance_decompose(&estimator, &x_train, &y_train, &x_test, &y_test, Some(20));
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.n_bootstrap, 20);
    }
}

// Add additional imports that might be needed
use scirs2_core::rand_prelude::SliceRandom;
use scirs2_core::random::{Rng, SeedableRng};
use scirs2_core::SliceRandomExt;
