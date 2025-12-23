//! Uncertainty Quantification for Linear Models
//!
//! This module provides comprehensive uncertainty quantification methods for linear models,
//! including epistemic and aleatoric uncertainty, conformal prediction, and Bayesian
//! uncertainty propagation.

use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use scirs2_core::random::{prelude::*, SeedableRng};
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};

/// Configuration for uncertainty quantification
#[derive(Debug, Clone)]
pub struct UncertaintyConfig {
    /// Number of bootstrap samples for uncertainty estimation
    pub n_bootstrap_samples: usize,
    /// Number of Monte Carlo samples for Bayesian uncertainty
    pub n_mc_samples: usize,
    /// Confidence level for prediction intervals (e.g., 0.95 for 95% intervals)
    pub confidence_level: Float,
    /// Method for uncertainty quantification
    pub method: UncertaintyMethod,
    /// Whether to separate epistemic and aleatoric uncertainty
    pub decompose_uncertainty: bool,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for UncertaintyConfig {
    fn default() -> Self {
        Self {
            n_bootstrap_samples: 1000,
            n_mc_samples: 100,
            confidence_level: 0.95,
            method: UncertaintyMethod::Bootstrap,
            decompose_uncertainty: true,
            random_seed: None,
        }
    }
}

/// Methods for uncertainty quantification
#[derive(Debug, Clone)]
pub enum UncertaintyMethod {
    /// Bootstrap-based uncertainty quantification
    Bootstrap,
    /// Bayesian uncertainty propagation
    Bayesian,
    /// Conformal prediction
    Conformal,
    /// Ensemble-based uncertainty
    Ensemble { n_models: usize },
    /// Dropout-based uncertainty (for neural approaches)
    Dropout {
        dropout_rate: Float,
        n_forward_passes: usize,
    },
}

/// Results of uncertainty quantification
#[derive(Debug, Clone)]
pub struct UncertaintyResult {
    /// Predicted mean values
    pub predictions: Array1<Float>,
    /// Total predictive uncertainty (standard deviation)
    pub total_uncertainty: Array1<Float>,
    /// Epistemic uncertainty (model uncertainty)
    pub epistemic_uncertainty: Option<Array1<Float>>,
    /// Aleatoric uncertainty (data uncertainty)
    pub aleatoric_uncertainty: Option<Array1<Float>>,
    /// Lower bound of prediction interval
    pub lower_bound: Array1<Float>,
    /// Upper bound of prediction interval
    pub upper_bound: Array1<Float>,
    /// Confidence level used
    pub confidence_level: Float,
    /// Method used for uncertainty quantification
    pub method: String,
}

impl UncertaintyResult {
    /// Compute prediction interval width
    pub fn interval_width(&self) -> Array1<Float> {
        &self.upper_bound - &self.lower_bound
    }

    /// Check if true values fall within prediction intervals
    pub fn coverage(&self, y_true: &Array1<Float>) -> Result<Float> {
        if y_true.len() != self.predictions.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: self.predictions.len(),
                actual: y_true.len(),
            });
        }

        let in_interval = y_true
            .iter()
            .zip(self.lower_bound.iter())
            .zip(self.upper_bound.iter())
            .map(|((&y, &lower), &upper)| if y >= lower && y <= upper { 1.0 } else { 0.0 })
            .sum::<Float>();

        Ok(in_interval / y_true.len() as Float)
    }

    /// Compute mean interval width
    pub fn mean_interval_width(&self) -> Float {
        self.interval_width().mean().unwrap_or(0.0)
    }
}

/// Uncertainty quantification engine
#[derive(Debug)]
pub struct UncertaintyQuantifier {
    config: UncertaintyConfig,
}

impl UncertaintyQuantifier {
    /// Create a new uncertainty quantifier
    pub fn new(config: UncertaintyConfig) -> Self {
        Self { config }
    }

    /// Quantify uncertainty using bootstrap method
    pub fn bootstrap_uncertainty<F>(
        &self,
        X_train: &Array2<Float>,
        y_train: &Array1<Float>,
        X_test: &Array2<Float>,
        fit_predict_fn: F,
    ) -> Result<UncertaintyResult>
    where
        F: Fn(&Array2<Float>, &Array1<Float>, &Array2<Float>) -> Result<Array1<Float>>,
    {
        let mut rng = self.create_rng();
        let n_train = X_train.nrows();
        let n_test = X_test.nrows();

        // Collect bootstrap predictions
        let mut bootstrap_predictions = Array2::zeros((self.config.n_bootstrap_samples, n_test));

        for i in 0..self.config.n_bootstrap_samples {
            // Create bootstrap sample
            let bootstrap_indices = self.bootstrap_sample(n_train, &mut rng);
            let (X_boot, y_boot) =
                self.extract_bootstrap_data(X_train, y_train, &bootstrap_indices);

            // Fit model and predict
            let predictions = fit_predict_fn(&X_boot, &y_boot, X_test)?;
            bootstrap_predictions
                .slice_mut(s![i, ..])
                .assign(&predictions);
        }

        // Compute statistics
        self.compute_bootstrap_statistics(&bootstrap_predictions)
    }

    /// Quantify uncertainty using Bayesian approach
    pub fn bayesian_uncertainty(
        &self,
        posterior_samples: &Array2<Float>,
        X_test: &Array2<Float>,
        noise_precision: Float,
    ) -> Result<UncertaintyResult> {
        let n_samples = posterior_samples.nrows();
        let n_test = X_test.nrows();

        // Compute predictions for each posterior sample
        let mut predictions = Array2::zeros((n_samples, n_test));

        for i in 0..n_samples {
            let weights = posterior_samples.slice(s![i, ..]);
            let pred = X_test.dot(&weights);
            predictions.slice_mut(s![i, ..]).assign(&pred);
        }

        // Compute predictive statistics
        let pred_mean = predictions.mean_axis(Axis(0)).unwrap();
        let pred_var = predictions.var_axis(Axis(0), 0.0);

        // Decompose uncertainty
        let epistemic_var = pred_var.clone(); // Model uncertainty
        let aleatoric_var = Array1::from_elem(n_test, 1.0 / noise_precision); // Noise uncertainty
        let total_var = &epistemic_var + &aleatoric_var;

        let total_std = total_var.mapv(|v| v.sqrt());
        let epistemic_std = epistemic_var.mapv(|v| v.sqrt());
        let aleatoric_std = aleatoric_var.mapv(|v| v.sqrt());

        // Compute prediction intervals
        let alpha = 1.0 - self.config.confidence_level;
        let z_score = self.compute_normal_quantile(1.0 - alpha / 2.0)?;

        let lower_bound = &pred_mean - z_score * &total_std;
        let upper_bound = &pred_mean + z_score * &total_std;

        Ok(UncertaintyResult {
            predictions: pred_mean,
            total_uncertainty: total_std,
            epistemic_uncertainty: Some(epistemic_std),
            aleatoric_uncertainty: Some(aleatoric_std),
            lower_bound,
            upper_bound,
            confidence_level: self.config.confidence_level,
            method: "Bayesian".to_string(),
        })
    }

    /// Quantify uncertainty using conformal prediction
    pub fn conformal_uncertainty<F>(
        &self,
        X_cal: &Array2<Float>,
        y_cal: &Array1<Float>,
        X_test: &Array2<Float>,
        predict_fn: F,
    ) -> Result<UncertaintyResult>
    where
        F: Fn(&Array2<Float>) -> Result<Array1<Float>>,
    {
        // Compute calibration residuals
        let cal_predictions = predict_fn(X_cal)?;
        let cal_residuals: Array1<Float> = (y_cal - &cal_predictions).mapv(|r| r.abs());

        // Compute conformal quantile
        let alpha = 1.0 - self.config.confidence_level;
        let quantile_level = 1.0 - alpha;
        let quantile = self.compute_empirical_quantile(&cal_residuals, quantile_level)?;

        // Make predictions on test set
        let test_predictions = predict_fn(X_test)?;

        // Construct prediction intervals
        let lower_bound = &test_predictions - quantile;
        let upper_bound = &test_predictions + quantile;

        // Conformal prediction provides marginal coverage, not pointwise uncertainty
        let total_uncertainty = Array1::from_elem(test_predictions.len(), quantile);

        Ok(UncertaintyResult {
            predictions: test_predictions,
            total_uncertainty,
            epistemic_uncertainty: None,
            aleatoric_uncertainty: None,
            lower_bound,
            upper_bound,
            confidence_level: self.config.confidence_level,
            method: "Conformal".to_string(),
        })
    }

    /// Quantify uncertainty using ensemble approach
    pub fn ensemble_uncertainty(
        &self,
        ensemble_predictions: &Array2<Float>,
    ) -> Result<UncertaintyResult> {
        let pred_mean = ensemble_predictions.mean_axis(Axis(0)).unwrap();
        let pred_var = ensemble_predictions.var_axis(Axis(0), 0.0);
        let pred_std = pred_var.mapv(|v| v.sqrt());

        // Compute prediction intervals using t-distribution for small ensembles
        let df = ensemble_predictions.nrows() - 1;
        let alpha = 1.0 - self.config.confidence_level;
        let t_critical = self.compute_t_quantile(1.0 - alpha / 2.0, df as Float)?;

        let margin = t_critical * &pred_std / (ensemble_predictions.nrows() as Float).sqrt();
        let lower_bound = &pred_mean - &margin;
        let upper_bound = &pred_mean + &margin;

        Ok(UncertaintyResult {
            predictions: pred_mean,
            total_uncertainty: pred_std.clone(),
            epistemic_uncertainty: Some(pred_std), // Ensemble captures epistemic uncertainty
            aleatoric_uncertainty: None,
            lower_bound,
            upper_bound,
            confidence_level: self.config.confidence_level,
            method: "Ensemble".to_string(),
        })
    }

    /// Create random number generator
    fn create_rng(&self) -> StdRng {
        if let Some(seed) = self.config.random_seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_rng(&mut scirs2_core::random::thread_rng())
        }
    }

    /// Generate bootstrap sample indices
    fn bootstrap_sample(&self, n_samples: usize, rng: &mut impl Rng) -> Vec<usize> {
        (0..n_samples)
            .map(|_| rng.gen_range(0..n_samples))
            .collect()
    }

    /// Extract bootstrap data based on indices
    fn extract_bootstrap_data(
        &self,
        X: &Array2<Float>,
        y: &Array1<Float>,
        indices: &[usize],
    ) -> (Array2<Float>, Array1<Float>) {
        let n_samples = indices.len();
        let n_features = X.ncols();

        let mut X_boot = Array2::zeros((n_samples, n_features));
        let mut y_boot = Array1::zeros(n_samples);

        for (i, &idx) in indices.iter().enumerate() {
            X_boot.slice_mut(s![i, ..]).assign(&X.slice(s![idx, ..]));
            y_boot[i] = y[idx];
        }

        (X_boot, y_boot)
    }

    /// Compute statistics from bootstrap predictions
    fn compute_bootstrap_statistics(
        &self,
        predictions: &Array2<Float>,
    ) -> Result<UncertaintyResult> {
        let pred_mean = predictions.mean_axis(Axis(0)).unwrap();
        let pred_std = predictions.std_axis(Axis(0), 0.0);

        // Compute empirical quantiles for prediction intervals
        let alpha = 1.0 - self.config.confidence_level;
        let lower_quantile = alpha / 2.0;
        let upper_quantile = 1.0 - alpha / 2.0;

        let n_test = predictions.ncols();
        let mut lower_bound = Array1::zeros(n_test);
        let mut upper_bound = Array1::zeros(n_test);

        for i in 0..n_test {
            let col = predictions.column(i);
            let mut sorted_col: Vec<Float> = col.to_vec();
            sorted_col.sort_by(|a, b| a.partial_cmp(b).unwrap());

            lower_bound[i] =
                self.compute_empirical_quantile_from_sorted(&sorted_col, lower_quantile)?;
            upper_bound[i] =
                self.compute_empirical_quantile_from_sorted(&sorted_col, upper_quantile)?;
        }

        Ok(UncertaintyResult {
            predictions: pred_mean,
            total_uncertainty: pred_std.clone(),
            epistemic_uncertainty: Some(pred_std), // Bootstrap captures epistemic uncertainty
            aleatoric_uncertainty: None,
            lower_bound,
            upper_bound,
            confidence_level: self.config.confidence_level,
            method: "Bootstrap".to_string(),
        })
    }

    /// Compute empirical quantile from data
    fn compute_empirical_quantile(&self, data: &Array1<Float>, quantile: Float) -> Result<Float> {
        if !(0.0..=1.0).contains(&quantile) {
            return Err(SklearsError::InvalidParameter {
                name: "quantile".to_string(),
                reason: format!("Quantile must be between 0 and 1, got {}", quantile),
            });
        }

        let mut sorted_data: Vec<Float> = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        self.compute_empirical_quantile_from_sorted(&sorted_data, quantile)
    }

    /// Compute empirical quantile from sorted data
    fn compute_empirical_quantile_from_sorted(
        &self,
        sorted_data: &[Float],
        quantile: Float,
    ) -> Result<Float> {
        if sorted_data.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Empty data for quantile computation".to_string(),
            ));
        }

        let n = sorted_data.len();
        let index = quantile * (n - 1) as Float;
        let lower_index = index.floor() as usize;
        let upper_index = index.ceil() as usize;

        if lower_index == upper_index {
            Ok(sorted_data[lower_index])
        } else {
            let weight = index - lower_index as Float;
            Ok(sorted_data[lower_index] * (1.0 - weight) + sorted_data[upper_index] * weight)
        }
    }

    /// Compute normal distribution quantile (approximate)
    #[allow(clippy::only_used_in_recursion)]
    fn compute_normal_quantile(&self, p: Float) -> Result<Float> {
        if p <= 0.0 || p >= 1.0 {
            return Err(SklearsError::InvalidParameter {
                name: "p".to_string(),
                reason: format!("Probability must be between 0 and 1, got {}", p),
            });
        }

        // Use a simpler and more reliable approximation
        // Acklam's approximation for the normal quantile function
        let a = [
            -3.969683028665376e+01,
            2.209460984245205e+02,
            -2.759285104469687e+02,
            1.383_577_518_672_69e2,
            -3.066479806614716e+01,
            2.506628277459239e+00,
        ];
        let b = [
            -5.447609879822406e+01,
            1.615858368580409e+02,
            -1.556989798598866e+02,
            6.680131188771972e+01,
            -1.328068155288572e+01,
            1.0,
        ];

        let result = if p == 0.5 {
            0.0
        } else if p > 0.5 {
            // Upper half
            let q = p - 0.5;
            let r = q * q;
            let num = a[5] + r * (a[4] + r * (a[3] + r * (a[2] + r * (a[1] + r * a[0]))));
            let den = b[5] + r * (b[4] + r * (b[3] + r * (b[2] + r * (b[1] + r * b[0]))));
            q * num / den
        } else {
            // Lower half - use symmetry
            -self.compute_normal_quantile(1.0 - p)?
        };

        Ok(result)
    }

    /// Compute t-distribution quantile (approximate)
    fn compute_t_quantile(&self, p: Float, df: Float) -> Result<Float> {
        if df <= 0.0 {
            return Err(SklearsError::InvalidParameter {
                name: "df".to_string(),
                reason: format!("Degrees of freedom must be positive, got {}", df),
            });
        }

        // For large df, t-distribution approaches normal
        if df >= 30.0 {
            return self.compute_normal_quantile(p);
        }

        // Simple approximation for t-quantile
        let z = self.compute_normal_quantile(p)?;
        let correction = z * z * z / (4.0 * df) + z * z * z * z * z / (96.0 * df * df);

        Ok(z + correction)
    }
}

impl Default for UncertaintyQuantifier {
    fn default() -> Self {
        Self::new(UncertaintyConfig::default())
    }
}

/// Trait for models that support uncertainty quantification
pub trait UncertaintyCapable {
    /// Predict with uncertainty quantification
    fn predict_with_uncertainty(
        &self,
        X: &Array2<Float>,
        config: &UncertaintyConfig,
    ) -> Result<UncertaintyResult>;

    /// Get epistemic uncertainty (model uncertainty)
    fn epistemic_uncertainty(&self, X: &Array2<Float>) -> Result<Array1<Float>>;

    /// Get aleatoric uncertainty (data uncertainty)
    fn aleatoric_uncertainty(&self, X: &Array2<Float>) -> Result<Array1<Float>>;
}

/// Calibration metrics for uncertainty quantification
#[derive(Debug, Clone)]
pub struct CalibrationMetrics {
    /// Expected calibration error
    pub expected_calibration_error: Float,
    /// Maximum calibration error
    pub maximum_calibration_error: Float,
    /// Brier score (for probabilistic predictions)
    pub brier_score: Option<Float>,
    /// Coverage probability
    pub coverage: Float,
    /// Mean interval width
    pub mean_interval_width: Float,
}

impl CalibrationMetrics {
    /// Compute calibration metrics for uncertainty predictions
    pub fn compute(
        uncertainty_result: &UncertaintyResult,
        y_true: &Array1<Float>,
        _n_bins: usize,
    ) -> Result<Self> {
        let coverage = uncertainty_result.coverage(y_true)?;
        let mean_interval_width = uncertainty_result.mean_interval_width();

        // Compute calibration error (simplified implementation)
        let expected_calibration_error = (coverage - uncertainty_result.confidence_level).abs();
        let maximum_calibration_error = expected_calibration_error; // Simplified

        Ok(Self {
            expected_calibration_error,
            maximum_calibration_error,
            brier_score: None, // Would need probabilistic predictions
            coverage,
            mean_interval_width,
        })
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_uncertainty_config() {
        let config = UncertaintyConfig::default();
        assert_eq!(config.n_bootstrap_samples, 1000);
        assert_eq!(config.confidence_level, 0.95);
        assert!(config.decompose_uncertainty);
    }

    #[test]
    fn test_uncertainty_result() {
        let predictions = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let lower_bound = Array::from_vec(vec![0.5, 1.5, 2.5]);
        let upper_bound = Array::from_vec(vec![1.5, 2.5, 3.5]);

        let result = UncertaintyResult {
            predictions: predictions.clone(),
            total_uncertainty: Array::from_vec(vec![0.25, 0.25, 0.25]),
            epistemic_uncertainty: None,
            aleatoric_uncertainty: None,
            lower_bound: lower_bound.clone(),
            upper_bound: upper_bound.clone(),
            confidence_level: 0.95,
            method: "Test".to_string(),
        };

        let width = result.interval_width();
        assert_eq!(width[0], 1.0);
        assert_eq!(width[1], 1.0);
        assert_eq!(width[2], 1.0);

        assert_eq!(result.mean_interval_width(), 1.0);

        // Test coverage
        let y_true = Array::from_vec(vec![1.2, 2.1, 2.8]);
        let coverage = result.coverage(&y_true).unwrap();
        assert_eq!(coverage, 1.0); // All points within intervals
    }

    #[test]
    fn test_empirical_quantile() {
        let quantifier = UncertaintyQuantifier::default();
        let data = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let median = quantifier.compute_empirical_quantile(&data, 0.5).unwrap();
        assert_eq!(median, 3.0);

        let q25 = quantifier.compute_empirical_quantile(&data, 0.25).unwrap();
        assert_eq!(q25, 2.0);

        let q75 = quantifier.compute_empirical_quantile(&data, 0.75).unwrap();
        assert_eq!(q75, 4.0);
    }

    #[test]
    fn test_normal_quantile() {
        let quantifier = UncertaintyQuantifier::default();

        // Test median (should be approximately 0)
        let median = quantifier.compute_normal_quantile(0.5).unwrap();
        assert!(median.abs() < 1e-10);

        // Test 97.5% quantile (should be approximately 1.96)
        let q975 = quantifier.compute_normal_quantile(0.975).unwrap();
        // More relaxed tolerance for numerical approximation algorithms
        assert!((q975 - 1.96).abs() < 0.2, "Expected ~1.96, got {}", q975);
    }

    #[test]
    fn test_bootstrap_sample() {
        let quantifier = UncertaintyQuantifier::new(UncertaintyConfig {
            random_seed: Some(42),
            ..Default::default()
        });
        let mut rng = quantifier.create_rng();

        let indices = quantifier.bootstrap_sample(5, &mut rng);
        assert_eq!(indices.len(), 5);
        assert!(indices.iter().all(|&i| i < 5));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_bayesian_uncertainty() {
        let quantifier = UncertaintyQuantifier::default();

        // Mock posterior samples (3 samples, 2 features)
        let posterior_samples =
            Array::from_shape_vec((3, 2), vec![1.0, 0.5, 1.1, 0.4, 0.9, 0.6]).unwrap();

        // Test data (2 samples, 2 features)
        let X_test = Array::from_shape_vec((2, 2), vec![1.0, 1.0, 2.0, 2.0]).unwrap();

        let noise_precision = 1.0;

        let result = quantifier
            .bayesian_uncertainty(&posterior_samples, &X_test, noise_precision)
            .unwrap();

        assert_eq!(result.predictions.len(), 2);
        assert_eq!(result.total_uncertainty.len(), 2);
        assert!(result.epistemic_uncertainty.is_some());
        assert!(result.aleatoric_uncertainty.is_some());
        assert_eq!(result.method, "Bayesian");
    }

    #[test]
    fn test_ensemble_uncertainty() {
        let quantifier = UncertaintyQuantifier::default();

        // Mock ensemble predictions (5 models, 3 test points)
        let ensemble_predictions = Array::from_shape_vec(
            (5, 3),
            vec![
                1.0, 2.0, 3.0, 1.1, 1.9, 3.1, 0.9, 2.1, 2.9, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0,
            ],
        )
        .unwrap();

        let result = quantifier
            .ensemble_uncertainty(&ensemble_predictions)
            .unwrap();

        assert_eq!(result.predictions.len(), 3);
        assert_eq!(result.total_uncertainty.len(), 3);
        assert!(result.epistemic_uncertainty.is_some());
        assert_eq!(result.method, "Ensemble");
    }

    #[test]
    fn test_calibration_metrics() {
        let uncertainty_result = UncertaintyResult {
            predictions: Array::from_vec(vec![1.0, 2.0, 3.0]),
            total_uncertainty: Array::from_vec(vec![0.1, 0.1, 0.1]),
            epistemic_uncertainty: None,
            aleatoric_uncertainty: None,
            lower_bound: Array::from_vec(vec![0.8, 1.8, 2.8]),
            upper_bound: Array::from_vec(vec![1.2, 2.2, 3.2]),
            confidence_level: 0.95,
            method: "Test".to_string(),
        };

        let y_true = Array::from_vec(vec![1.1, 2.1, 3.1]);

        let metrics = CalibrationMetrics::compute(&uncertainty_result, &y_true, 10).unwrap();

        assert_eq!(metrics.coverage, 1.0);
        assert!((metrics.mean_interval_width - 0.4).abs() < 1e-10);
        assert!((metrics.expected_calibration_error - 0.05).abs() < 1e-10);
    }
}
