//! Prediction Intervals for Multiclass Classification
//!
//! This module provides methods for computing prediction intervals for
//! class probabilities in multiclass classification problems.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{rngs::StdRng, seeded_rng, CoreRandom, Rng};
use sklears_core::error::{Result as SklResult, SklearsError};

/// Methods for interval estimation
#[derive(Debug, Clone, PartialEq)]
pub enum IntervalMethod {
    /// Bootstrap-based intervals
    Bootstrap {
        n_bootstrap: usize,

        random_state: Option<u64>,
    },
    /// Bayesian credible intervals
    Bayesian { prior_alpha: Vec<f64> },
    /// Normal approximation intervals
    NormalApproximation,
    /// Quantile-based intervals
    Quantile,
    /// Wilson score intervals
    Wilson,
}

impl Default for IntervalMethod {
    fn default() -> Self {
        Self::Bootstrap {
            n_bootstrap: 1000,
            random_state: None,
        }
    }
}

/// Prediction interval result
#[derive(Debug, Clone)]
pub struct PredictionInterval {
    /// Lower bounds for each class probability [n_samples, n_classes]
    pub lower_bounds: Array2<f64>,
    /// Upper bounds for each class probability [n_samples, n_classes]
    pub upper_bounds: Array2<f64>,
    /// Confidence level used
    pub confidence_level: f64,
    /// Method used for computation
    pub method: IntervalMethod,
    /// Width of intervals [n_samples, n_classes]
    pub widths: Array2<f64>,
}

/// Interval estimator for multiclass classification
#[derive(Debug, Clone)]
pub struct IntervalEstimator {
    method: IntervalMethod,
    confidence_level: f64,
}

impl IntervalEstimator {
    /// Create a new interval estimator
    pub fn new() -> Self {
        Self {
            method: IntervalMethod::default(),
            confidence_level: 0.95,
        }
    }

    /// Create a builder for interval estimator
    pub fn builder() -> IntervalEstimatorBuilder {
        IntervalEstimatorBuilder::new()
    }

    /// Set the interval method
    pub fn method(mut self, method: IntervalMethod) -> Self {
        self.method = method;
        self
    }

    /// Set confidence level
    pub fn confidence_level(mut self, confidence_level: f64) -> Self {
        if confidence_level > 0.0 && confidence_level < 1.0 {
            self.confidence_level = confidence_level;
        }
        self
    }

    /// Build the estimator
    pub fn build(self) -> Self {
        self
    }

    /// Estimate prediction intervals
    pub fn estimate_intervals(
        &self,
        probabilities: &Array2<f64>,
        calibration_scores: Option<&Array2<f64>>,
    ) -> SklResult<Array2<f64>> {
        let (_n_samples, _n_classes) = probabilities.dim();

        match &self.method {
            IntervalMethod::Bootstrap {
                n_bootstrap,
                random_state,
            } => self.bootstrap_intervals(probabilities, *n_bootstrap, *random_state),
            IntervalMethod::Bayesian { prior_alpha } => {
                self.bayesian_intervals(probabilities, prior_alpha)
            }
            IntervalMethod::NormalApproximation => {
                self.normal_approximation_intervals(probabilities)
            }
            IntervalMethod::Quantile => self.quantile_intervals(probabilities, calibration_scores),
            IntervalMethod::Wilson => self.wilson_intervals(probabilities),
        }
    }

    /// Estimate detailed prediction intervals
    pub fn estimate_detailed(
        &self,
        probabilities: &Array2<f64>,
        calibration_scores: Option<&Array2<f64>>,
    ) -> SklResult<PredictionInterval> {
        let intervals = self.estimate_intervals(probabilities, calibration_scores)?;
        let (n_samples, n_classes) = probabilities.dim();

        // Split intervals into lower and upper bounds
        let mut lower_bounds = Array2::zeros((n_samples, n_classes));
        let mut upper_bounds = Array2::zeros((n_samples, n_classes));
        let mut widths = Array2::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            for j in 0..n_classes {
                lower_bounds[[i, j]] = intervals[[i, j * 2]];
                upper_bounds[[i, j]] = intervals[[i, j * 2 + 1]];
                widths[[i, j]] = upper_bounds[[i, j]] - lower_bounds[[i, j]];
            }
        }

        Ok(PredictionInterval {
            lower_bounds,
            upper_bounds,
            confidence_level: self.confidence_level,
            method: self.method.clone(),
            widths,
        })
    }

    /// Bootstrap-based confidence intervals
    fn bootstrap_intervals(
        &self,
        probabilities: &Array2<f64>,
        n_bootstrap: usize,
        random_state: Option<u64>,
    ) -> SklResult<Array2<f64>> {
        let (n_samples, n_classes) = probabilities.dim();
        let mut intervals = Array2::zeros((n_samples, n_classes * 2));

        let mut rng: CoreRandom<StdRng> = match random_state {
            Some(seed) => seeded_rng(seed),
            None => seeded_rng(42),
        };

        for i in 0..n_samples {
            let sample_probs = probabilities.row(i).to_owned();

            // Generate bootstrap samples using Dirichlet distribution
            let mut bootstrap_samples = Vec::with_capacity(n_bootstrap);

            for _ in 0..n_bootstrap {
                let bootstrap_sample = self.generate_bootstrap_sample(&sample_probs, &mut rng)?;
                bootstrap_samples.push(bootstrap_sample);
            }

            // Compute confidence intervals for each class
            for j in 0..n_classes {
                let mut class_samples: Vec<f64> =
                    bootstrap_samples.iter().map(|sample| sample[j]).collect();
                class_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let alpha = 1.0 - self.confidence_level;
                let lower_idx = ((alpha / 2.0) * n_bootstrap as f64) as usize;
                let upper_idx = ((1.0 - alpha / 2.0) * n_bootstrap as f64) as usize;
                let lower_idx = std::cmp::min(lower_idx, n_bootstrap - 1);
                let upper_idx = std::cmp::min(upper_idx, n_bootstrap - 1);

                intervals[[i, j * 2]] = class_samples[lower_idx];
                intervals[[i, j * 2 + 1]] = class_samples[upper_idx];
            }
        }

        Ok(intervals)
    }

    /// Generate bootstrap sample using simple perturbation method
    fn generate_bootstrap_sample(
        &self,
        probabilities: &Array1<f64>,
        rng: &mut CoreRandom<StdRng>,
    ) -> SklResult<Array1<f64>> {
        let n_classes = probabilities.len();

        // Apply random perturbation to each probability
        let mut perturbed = Array1::zeros(n_classes);
        for (i, &p) in probabilities.iter().enumerate() {
            let perturbation: f64 = rng.gen::<f64>() * 0.4 + 0.8; // Perturbation range (0.8 to 1.2)
            perturbed[i] = (p * perturbation).max(1e-6);
        }

        // Normalize to ensure probabilities sum to 1
        let sum: f64 = perturbed.sum();
        if sum > 1e-15 {
            for sample in perturbed.iter_mut() {
                *sample /= sum;
            }
        } else {
            // Fallback to uniform distribution
            perturbed.fill(1.0 / n_classes as f64);
        }

        Ok(perturbed)
    }

    /// Bayesian credible intervals using Dirichlet posterior
    fn bayesian_intervals(
        &self,
        probabilities: &Array2<f64>,
        prior_alpha: &[f64],
    ) -> SklResult<Array2<f64>> {
        let (n_samples, n_classes) = probabilities.dim();

        if prior_alpha.len() != n_classes {
            return Err(SklearsError::InvalidInput(
                "Prior alpha length must match number of classes".to_string(),
            ));
        }

        let mut intervals = Array2::zeros((n_samples, n_classes * 2));

        for i in 0..n_samples {
            let sample_probs = probabilities.row(i);

            // Assume we have observed counts that led to these probabilities
            // Convert probabilities to pseudo-counts
            let total_counts = 100.0; // Assume 100 observations for this sample
            let observed_counts: Vec<f64> =
                sample_probs.iter().map(|&p| p * total_counts).collect();

            // Compute posterior parameters (Dirichlet)
            let posterior_alpha: Vec<f64> = prior_alpha
                .iter()
                .zip(observed_counts.iter())
                .map(|(&prior, &count)| prior + count)
                .collect();

            // Compute credible intervals using Beta approximation for each class
            for j in 0..n_classes {
                let alpha_j = posterior_alpha[j];
                let beta_j: f64 = posterior_alpha.iter().sum::<f64>() - alpha_j;

                let (lower, upper) = self.beta_credible_interval(alpha_j, beta_j)?;
                intervals[[i, j * 2]] = lower;
                intervals[[i, j * 2 + 1]] = upper;
            }
        }

        Ok(intervals)
    }

    /// Compute credible interval using normal approximation for Beta distribution
    fn beta_credible_interval(&self, alpha: f64, beta: f64) -> SklResult<(f64, f64)> {
        // Use normal approximation for Beta distribution
        let mean = alpha / (alpha + beta);
        let variance = (alpha * beta) / ((alpha + beta).powi(2) * (alpha + beta + 1.0));
        let std_dev = variance.sqrt();

        let conf_alpha = 1.0 - self.confidence_level;
        let z_score = self.normal_quantile(1.0 - conf_alpha / 2.0);

        let margin = z_score * std_dev;
        let lower = (mean - margin).max(0.0);
        let upper = (mean + margin).min(1.0);

        Ok((lower, upper))
    }

    /// Approximate normal quantile using Box-Muller transform
    fn normal_quantile(&self, p: f64) -> f64 {
        // Approximation for standard normal quantile
        if p <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if p >= 1.0 {
            return f64::INFINITY;
        }
        if p == 0.5 {
            return 0.0;
        }

        // Beasley-Springer-Moro algorithm approximation
        let a0 = 2.515517;
        let a1 = 0.802853;
        let a2 = 0.010328;
        let b1 = 1.432788;
        let b2 = 0.189269;
        let b3 = 0.001308;

        let t = if p > 0.5 {
            (-2.0 * (1.0 - p).ln()).sqrt()
        } else {
            (-2.0 * p.ln()).sqrt()
        };

        let numerator = a0 + a1 * t + a2 * t * t;
        let denominator = 1.0 + b1 * t + b2 * t * t + b3 * t * t * t;
        let z = t - numerator / denominator;

        if p > 0.5 {
            z
        } else {
            -z
        }
    }

    /// Normal approximation intervals
    fn normal_approximation_intervals(
        &self,
        probabilities: &Array2<f64>,
    ) -> SklResult<Array2<f64>> {
        let (n_samples, n_classes) = probabilities.dim();
        let mut intervals = Array2::zeros((n_samples, n_classes * 2));

        // Z-score for desired confidence level
        let alpha = 1.0 - self.confidence_level;
        let z_score = self.normal_quantile(1.0 - alpha / 2.0);

        for i in 0..n_samples {
            for j in 0..n_classes {
                let p = probabilities[[i, j]];

                // Assume we have n observations, estimate from probability
                let n = 100.0; // Assumed sample size
                let variance = p * (1.0 - p) / n;
                let std_error = variance.sqrt();

                let margin = z_score * std_error;
                let lower = (p - margin).max(0.0);
                let upper = (p + margin).min(1.0);

                intervals[[i, j * 2]] = lower;
                intervals[[i, j * 2 + 1]] = upper;
            }
        }

        Ok(intervals)
    }

    /// Quantile-based intervals using calibration data
    fn quantile_intervals(
        &self,
        probabilities: &Array2<f64>,
        calibration_scores: Option<&Array2<f64>>,
    ) -> SklResult<Array2<f64>> {
        let (n_samples, n_classes) = probabilities.dim();

        if let Some(cal_scores) = calibration_scores {
            // Use calibration data to estimate quantiles
            self.compute_empirical_quantile_intervals(probabilities, cal_scores)
        } else {
            // Fallback to simple quantile estimation
            let mut intervals = Array2::zeros((n_samples, n_classes * 2));

            let alpha = 1.0 - self.confidence_level;
            let margin = alpha / 2.0;

            for i in 0..n_samples {
                for j in 0..n_classes {
                    let p = probabilities[[i, j]];
                    intervals[[i, j * 2]] = (p - margin).max(0.0);
                    intervals[[i, j * 2 + 1]] = (p + margin).min(1.0);
                }
            }

            Ok(intervals)
        }
    }

    /// Compute empirical quantile intervals using calibration data
    fn compute_empirical_quantile_intervals(
        &self,
        probabilities: &Array2<f64>,
        calibration_scores: &Array2<f64>,
    ) -> SklResult<Array2<f64>> {
        let (n_samples, n_classes) = probabilities.dim();
        let mut intervals = Array2::zeros((n_samples, n_classes * 2));

        let alpha = 1.0 - self.confidence_level;

        // For each class, compute quantiles from calibration data
        for j in 0..n_classes {
            let cal_class_scores = calibration_scores.column(j);
            let mut sorted_scores = cal_class_scores.to_vec();
            sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let n_cal = sorted_scores.len();
            if n_cal == 0 {
                continue;
            }

            let lower_idx = ((alpha / 2.0) * n_cal as f64) as usize;
            let upper_idx = ((1.0 - alpha / 2.0) * n_cal as f64) as usize;
            let lower_idx = std::cmp::min(lower_idx, n_cal - 1);
            let upper_idx = std::cmp::min(upper_idx, n_cal - 1);

            let lower_quantile = sorted_scores[lower_idx];
            let upper_quantile = sorted_scores[upper_idx];

            // Apply quantiles to all samples for this class
            for i in 0..n_samples {
                let p = probabilities[[i, j]];
                intervals[[i, j * 2]] = (p + lower_quantile - 0.5).max(0.0);
                intervals[[i, j * 2 + 1]] = (p + upper_quantile - 0.5).min(1.0);
            }
        }

        Ok(intervals)
    }

    /// Wilson score intervals
    fn wilson_intervals(&self, probabilities: &Array2<f64>) -> SklResult<Array2<f64>> {
        let (n_samples, n_classes) = probabilities.dim();
        let mut intervals = Array2::zeros((n_samples, n_classes * 2));

        let alpha = 1.0 - self.confidence_level;
        let z = self.normal_quantile(1.0 - alpha / 2.0);
        let z_squared = z * z;

        for i in 0..n_samples {
            for j in 0..n_classes {
                let p = probabilities[[i, j]];
                let n = 100.0; // Assumed sample size

                let denominator = 1.0 + z_squared / n;
                let center = (p + z_squared / (2.0 * n)) / denominator;
                let margin =
                    z * (p * (1.0 - p) / n + z_squared / (4.0 * n * n)).sqrt() / denominator;

                let lower = (center - margin).max(0.0);
                let upper = (center + margin).min(1.0);

                intervals[[i, j * 2]] = lower;
                intervals[[i, j * 2 + 1]] = upper;
            }
        }

        Ok(intervals)
    }
}

impl Default for IntervalEstimator {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for interval estimator
#[derive(Debug)]
pub struct IntervalEstimatorBuilder {
    method: IntervalMethod,
    confidence_level: f64,
}

impl Default for IntervalEstimatorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl IntervalEstimatorBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            method: IntervalMethod::default(),
            confidence_level: 0.95,
        }
    }

    /// Set the interval method
    pub fn method(mut self, method: IntervalMethod) -> Self {
        self.method = method;
        self
    }

    /// Set confidence level
    pub fn confidence_level(mut self, confidence_level: f64) -> Self {
        if confidence_level > 0.0 && confidence_level < 1.0 {
            self.confidence_level = confidence_level;
        }
        self
    }

    /// Build the interval estimator
    pub fn build(self) -> IntervalEstimator {
        IntervalEstimator {
            method: self.method,
            confidence_level: self.confidence_level,
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_autograd::ndarray::array;

    #[test]
    fn test_interval_estimator_creation() {
        let estimator = IntervalEstimator::new();
        assert_eq!(estimator.confidence_level, 0.95);
    }

    #[test]
    fn test_interval_estimator_builder() {
        let estimator = IntervalEstimator::builder()
            .method(IntervalMethod::Wilson)
            .confidence_level(0.99)
            .build();

        assert_eq!(estimator.confidence_level, 0.99);
        assert_eq!(estimator.method, IntervalMethod::Wilson);
    }

    #[test]
    fn test_bootstrap_intervals() {
        let estimator = IntervalEstimator::new()
            .method(IntervalMethod::Bootstrap {
                n_bootstrap: 100,
                random_state: Some(42),
            })
            .confidence_level(0.9)
            .build();

        let probabilities = array![[0.8, 0.1, 0.1], [0.3, 0.4, 0.3]];

        let intervals = estimator.estimate_intervals(&probabilities, None).unwrap();

        // Check dimensions: [n_samples, n_classes * 2]
        assert_eq!(intervals.dim(), (2, 6));

        // Check that lower bounds <= upper bounds
        for i in 0..2 {
            for j in 0..3 {
                let lower = intervals[[i, j * 2]];
                let upper = intervals[[i, j * 2 + 1]];
                assert!(
                    lower <= upper,
                    "Lower bound {} should be <= upper bound {}",
                    lower,
                    upper
                );
                assert!(lower >= 0.0 && upper <= 1.0, "Bounds should be in [0, 1]");
            }
        }
    }

    #[test]
    fn test_normal_approximation_intervals() {
        let estimator = IntervalEstimator::new()
            .method(IntervalMethod::NormalApproximation)
            .confidence_level(0.95)
            .build();

        let probabilities = array![[0.7, 0.2, 0.1], [0.4, 0.4, 0.2]];

        let intervals = estimator.estimate_intervals(&probabilities, None).unwrap();

        assert_eq!(intervals.dim(), (2, 6));

        // Check that intervals are reasonable
        for i in 0..2 {
            for j in 0..3 {
                let lower = intervals[[i, j * 2]];
                let upper = intervals[[i, j * 2 + 1]];
                let original = probabilities[[i, j]];

                assert!(lower <= original && original <= upper);
                assert!(lower >= 0.0 && upper <= 1.0);
            }
        }
    }

    #[test]
    fn test_wilson_intervals() {
        let estimator = IntervalEstimator::new()
            .method(IntervalMethod::Wilson)
            .confidence_level(0.95)
            .build();

        let probabilities = array![[0.5, 0.3, 0.2]];

        let intervals = estimator.estimate_intervals(&probabilities, None).unwrap();

        assert_eq!(intervals.dim(), (1, 6));

        // Wilson intervals should be symmetric around 0.5 for p=0.5
        let lower_0 = intervals[[0, 0]];
        let upper_0 = intervals[[0, 1]];
        let p_0 = probabilities[[0, 0]];

        assert!(lower_0 < p_0 && p_0 < upper_0);

        // For p=0.5, the interval should be roughly symmetric
        if (p_0 - 0.5).abs() < 1e-10 {
            let margin_lower = p_0 - lower_0;
            let margin_upper = upper_0 - p_0;
            assert!((margin_lower - margin_upper).abs() < 0.1);
        }
    }

    #[test]
    fn test_bayesian_intervals() {
        let estimator = IntervalEstimator::new()
            .method(IntervalMethod::Bayesian {
                prior_alpha: vec![1.0, 1.0, 1.0], // Uniform prior
            })
            .confidence_level(0.9)
            .build();

        let probabilities = array![[0.6, 0.3, 0.1]];

        let intervals = estimator.estimate_intervals(&probabilities, None).unwrap();

        assert_eq!(intervals.dim(), (1, 6));

        // Check bounds are valid
        for j in 0..3 {
            let lower = intervals[[0, j * 2]];
            let upper = intervals[[0, j * 2 + 1]];
            assert!(lower <= upper);
            assert!(lower >= 0.0 && upper <= 1.0);
        }
    }

    #[test]
    fn test_detailed_intervals() {
        let estimator = IntervalEstimator::new()
            .method(IntervalMethod::NormalApproximation)
            .confidence_level(0.95)
            .build();

        let probabilities = array![[0.8, 0.1, 0.1], [0.3, 0.4, 0.3]];

        let detailed = estimator.estimate_detailed(&probabilities, None).unwrap();

        assert_eq!(detailed.lower_bounds.dim(), (2, 3));
        assert_eq!(detailed.upper_bounds.dim(), (2, 3));
        assert_eq!(detailed.widths.dim(), (2, 3));
        assert_eq!(detailed.confidence_level, 0.95);

        // Check that widths = upper - lower
        for i in 0..2 {
            for j in 0..3 {
                let expected_width = detailed.upper_bounds[[i, j]] - detailed.lower_bounds[[i, j]];
                assert!((detailed.widths[[i, j]] - expected_width).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_normal_quantile() {
        let estimator = IntervalEstimator::new();

        // Test known values
        assert!((estimator.normal_quantile(0.5) - 0.0).abs() < 1e-6);
        assert!(estimator.normal_quantile(0.975) > 1.9); // Should be close to 1.96
        assert!(estimator.normal_quantile(0.025) < -1.9); // Should be close to -1.96

        // Test edge cases
        assert!(
            estimator.normal_quantile(0.0).is_infinite() && estimator.normal_quantile(0.0) < 0.0
        );
        assert!(
            estimator.normal_quantile(1.0).is_infinite() && estimator.normal_quantile(1.0) > 0.0
        );
    }
}
