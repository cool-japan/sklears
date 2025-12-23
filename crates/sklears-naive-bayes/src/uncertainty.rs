//! Uncertainty quantification for Naive Bayes classifiers

// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use once_cell::sync::Lazy;
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::Result;
use std::f64::consts::PI;
use std::sync::Mutex;

/// Uncertainty types for Naive Bayes predictions
#[derive(Debug, Clone)]
pub struct UncertaintyMeasures {
    /// Prediction confidence (max probability)
    pub confidence: Array1<f64>,
    /// Entropy of the prediction distribution
    pub entropy: Array1<f64>,
    /// Margin between top two predictions
    pub margin: Array1<f64>,
    /// Variance of the prediction distribution
    pub variance: Array1<f64>,
}

static RNG_STATE: Lazy<Mutex<u64>> = Lazy::new(|| Mutex::new(12345));
static BOX_MULLER_CACHE: Lazy<Mutex<Option<f64>>> = Lazy::new(|| Mutex::new(None));

/// Uncertainty quantification methods
pub trait UncertaintyQuantification {
    /// Compute uncertainty measures for predictions
    fn uncertainty_measures(&self, probabilities: &Array2<f64>) -> UncertaintyMeasures;

    /// Compute prediction intervals for regression-like scenarios
    fn prediction_intervals(
        &self,
        probabilities: &Array2<f64>,
        confidence_level: f64,
    ) -> Result<Array2<f64>>;

    /// Detect out-of-distribution samples based on uncertainty
    fn detect_outliers(&self, probabilities: &Array2<f64>, threshold: f64) -> Array1<bool>;
}

/// Standard uncertainty quantification implementation
#[derive(Debug, Clone)]
pub struct StandardUncertainty;

impl StandardUncertainty {
    pub fn new() -> Self {
        Self
    }
}

impl Default for StandardUncertainty {
    fn default() -> Self {
        Self::new()
    }
}

impl UncertaintyQuantification for StandardUncertainty {
    fn uncertainty_measures(&self, probabilities: &Array2<f64>) -> UncertaintyMeasures {
        let n_samples = probabilities.nrows();
        let mut confidence = Array1::zeros(n_samples);
        let mut entropy = Array1::zeros(n_samples);
        let mut margin = Array1::zeros(n_samples);
        let mut variance = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let probs = probabilities.row(i);

            // Confidence: maximum probability
            confidence[i] = probs.iter().cloned().fold(0.0, f64::max);

            // Entropy: -sum(p * log(p))
            entropy[i] = -probs
                .iter()
                .map(|&p| if p > 1e-15 { p * p.ln() } else { 0.0 })
                .sum::<f64>();

            // Margin: difference between top two probabilities
            let mut sorted_probs: Vec<f64> = probs.to_vec();
            sorted_probs.sort_by(|a, b| b.partial_cmp(a).unwrap());
            margin[i] = if sorted_probs.len() >= 2 {
                sorted_probs[0] - sorted_probs[1]
            } else {
                sorted_probs[0]
            };

            // Variance: variance of the probability distribution
            let mean = probs.mean().unwrap_or(0.0);
            variance[i] =
                probs.iter().map(|&p| (p - mean).powi(2)).sum::<f64>() / probs.len() as f64;
        }

        UncertaintyMeasures {
            confidence,
            entropy,
            margin,
            variance,
        }
    }

    fn prediction_intervals(
        &self,
        probabilities: &Array2<f64>,
        confidence_level: f64,
    ) -> Result<Array2<f64>> {
        if confidence_level <= 0.0 || confidence_level >= 1.0 {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Confidence level must be between 0 and 1".to_string(),
            ));
        }

        let n_samples = probabilities.nrows();
        let n_classes = probabilities.ncols();
        let mut intervals = Array2::zeros((n_samples, 2));

        let alpha = 1.0 - confidence_level;
        let z_score = 1.96; // Approximate 95% confidence interval

        for i in 0..n_samples {
            let probs = probabilities.row(i);

            // Find the predicted class (maximum probability)
            let max_idx = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            let p_max = probs[max_idx];

            // Compute confidence interval for the maximum probability
            // Using Wilson score interval for proportions
            let n_eff = n_classes as f64; // Effective sample size
            let p_center =
                (p_max + z_score * z_score / (2.0 * n_eff)) / (1.0 + z_score * z_score / n_eff);
            let margin_error = z_score
                * (p_max * (1.0 - p_max) / n_eff + z_score * z_score / (4.0 * n_eff * n_eff))
                    .sqrt()
                / (1.0 + z_score * z_score / n_eff);

            intervals[[i, 0]] = (p_center - margin_error).max(0.0);
            intervals[[i, 1]] = (p_center + margin_error).min(1.0);
        }

        Ok(intervals)
    }

    fn detect_outliers(&self, probabilities: &Array2<f64>, threshold: f64) -> Array1<bool> {
        let uncertainty = self.uncertainty_measures(probabilities);
        let n_samples = probabilities.nrows();
        let mut outliers = Array1::from_elem(n_samples, false);

        // Detect outliers based on high entropy (uncertainty)
        let entropy_threshold = threshold;

        for i in 0..n_samples {
            // A sample is an outlier if it has high entropy (low confidence)
            outliers[i] = uncertainty.entropy[i] > entropy_threshold;
        }

        outliers
    }
}

/// Bayesian uncertainty quantification using Monte Carlo sampling
#[derive(Debug, Clone)]
pub struct BayesianUncertainty {
    n_samples: usize,
}

impl BayesianUncertainty {
    pub fn new(n_samples: usize) -> Self {
        Self { n_samples }
    }
}

impl Default for BayesianUncertainty {
    fn default() -> Self {
        Self::new(1000)
    }
}

impl UncertaintyQuantification for BayesianUncertainty {
    fn uncertainty_measures(&self, probabilities: &Array2<f64>) -> UncertaintyMeasures {
        // For now, use standard uncertainty as a baseline
        // In a full implementation, this would involve Monte Carlo sampling
        // from the posterior distribution of parameters
        let standard = StandardUncertainty::new();
        standard.uncertainty_measures(probabilities)
    }

    fn prediction_intervals(
        &self,
        probabilities: &Array2<f64>,
        confidence_level: f64,
    ) -> Result<Array2<f64>> {
        // Placeholder for Bayesian prediction intervals
        // Would implement MCMC sampling here
        let standard = StandardUncertainty::new();
        standard.prediction_intervals(probabilities, confidence_level)
    }

    fn detect_outliers(&self, probabilities: &Array2<f64>, threshold: f64) -> Array1<bool> {
        let standard = StandardUncertainty::new();
        standard.detect_outliers(probabilities, threshold)
    }
}

/// Model ensemble for improved uncertainty quantification
#[derive(Debug, Clone)]
pub struct EnsembleUncertainty {
    /// Number of models in the ensemble
    pub n_models: usize,
    /// Bootstrap sample ratio
    pub sample_ratio: f64,
}

impl EnsembleUncertainty {
    pub fn new(n_models: usize, sample_ratio: f64) -> Self {
        Self {
            n_models,
            sample_ratio,
        }
    }
}

impl Default for EnsembleUncertainty {
    fn default() -> Self {
        Self::new(10, 0.8)
    }
}

/// Calibration methods for improving probability estimates
#[derive(Debug, Clone)]
pub struct CalibrationMethod;

impl CalibrationMethod {
    /// Platt scaling for probability calibration
    pub fn platt_scaling(
        predictions: &Array1<f64>,
        true_labels: &Array1<i32>,
    ) -> Result<(f64, f64)> {
        // Simplified Platt scaling implementation
        // Fits sigmoid: P(y=1|f) = 1 / (1 + exp(A*f + B))

        let n = predictions.len();
        if n == 0 {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Empty predictions array".to_string(),
            ));
        }

        // Count positive and negative examples
        let n_pos = true_labels.iter().filter(|&&x| x == 1).count() as f64;
        let n_neg = (n as f64) - n_pos;

        if n_pos == 0.0 || n_neg == 0.0 {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Need both positive and negative examples".to_string(),
            ));
        }

        // Initial estimates (simplified)
        let a = 0.0; // Would be fitted via maximum likelihood
        let b = (n_neg / n_pos).ln();

        Ok((a, b))
    }

    /// Isotonic regression for probability calibration
    pub fn isotonic_regression(
        predictions: &Array1<f64>,
        true_labels: &Array1<i32>,
    ) -> Result<Vec<(f64, f64)>> {
        // Simplified isotonic regression
        let n = predictions.len();
        if n == 0 {
            return Ok(vec![]);
        }

        let mut pairs: Vec<(f64, i32)> = predictions
            .iter()
            .zip(true_labels.iter())
            .map(|(&pred, &label)| (pred, label))
            .collect();

        // Sort by prediction score
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Create calibration mapping (simplified)
        let mut calibration_map = Vec::new();
        let window_size = (n / 10).max(1);

        for i in (0..n).step_by(window_size) {
            let end = (i + window_size).min(n);
            let window = &pairs[i..end];

            let avg_score =
                window.iter().map(|(score, _)| score).sum::<f64>() / window.len() as f64;
            let avg_label =
                window.iter().map(|(_, label)| *label as f64).sum::<f64>() / window.len() as f64;

            calibration_map.push((avg_score, avg_label));
        }

        Ok(calibration_map)
    }
}

/// Statistical tests for model reliability
#[derive(Debug, Clone)]
pub struct ReliabilityTests;

impl ReliabilityTests {
    /// Hosmer-Lemeshow goodness-of-fit test
    pub fn hosmer_lemeshow_test(
        probabilities: &Array1<f64>,
        true_labels: &Array1<i32>,
        n_bins: usize,
    ) -> Result<f64> {
        if probabilities.len() != true_labels.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Probabilities and labels must have same length".to_string(),
            ));
        }

        let n = probabilities.len();
        if n < n_bins {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Number of samples must be at least equal to number of bins".to_string(),
            ));
        }

        // Sort predictions and group into bins
        let mut pairs: Vec<(f64, i32)> = probabilities
            .iter()
            .zip(true_labels.iter())
            .map(|(&prob, &label)| (prob, label))
            .collect();

        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let bin_size = n / n_bins;
        let mut chi_square = 0.0;

        for i in 0..n_bins {
            let start = i * bin_size;
            let end = if i == n_bins - 1 {
                n
            } else {
                (i + 1) * bin_size
            };
            let bin = &pairs[start..end];

            let observed_pos = bin.iter().filter(|(_, label)| *label == 1).count() as f64;
            let observed_neg = bin.len() as f64 - observed_pos;

            let expected_prob = bin.iter().map(|(prob, _)| prob).sum::<f64>() / bin.len() as f64;
            let expected_pos = bin.len() as f64 * expected_prob;
            let expected_neg = bin.len() as f64 * (1.0 - expected_prob);

            if expected_pos > 0.0 && expected_neg > 0.0 {
                chi_square += (observed_pos - expected_pos).powi(2) / expected_pos;
                chi_square += (observed_neg - expected_neg).powi(2) / expected_neg;
            }
        }

        Ok(chi_square)
    }

    /// Brier score for probability assessment
    pub fn brier_score(probabilities: &Array1<f64>, true_labels: &Array1<i32>) -> Result<f64> {
        if probabilities.len() != true_labels.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Probabilities and labels must have same length".to_string(),
            ));
        }

        let n = probabilities.len() as f64;
        let score = probabilities
            .iter()
            .zip(true_labels.iter())
            .map(|(&prob, &label)| {
                let true_prob = if label == 1 { 1.0 } else { 0.0 };
                (prob - true_prob).powi(2)
            })
            .sum::<f64>()
            / n;

        Ok(score)
    }
}

/// Model uncertainty propagation methods
#[derive(Debug, Clone)]
pub struct ModelUncertaintyPropagation {
    /// Number of bootstrap samples for parameter uncertainty
    pub n_bootstrap: usize,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Whether to use Bayesian parameter uncertainty
    pub bayesian_uncertainty: bool,
    /// Prior strength for Bayesian methods
    pub prior_strength: f64,
}

impl Default for ModelUncertaintyPropagation {
    fn default() -> Self {
        Self {
            n_bootstrap: 100,
            random_seed: Some(42),
            bayesian_uncertainty: false,
            prior_strength: 1.0,
        }
    }
}

impl ModelUncertaintyPropagation {
    /// Create a new model uncertainty propagation instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Set number of bootstrap samples
    pub fn n_bootstrap(mut self, n: usize) -> Self {
        self.n_bootstrap = n;
        self
    }

    /// Set random seed
    pub fn random_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Enable Bayesian uncertainty
    pub fn bayesian_uncertainty(mut self, enabled: bool) -> Self {
        self.bayesian_uncertainty = enabled;
        self
    }

    /// Set prior strength for Bayesian methods
    pub fn prior_strength(mut self, strength: f64) -> Self {
        self.prior_strength = strength;
        self
    }

    /// Propagate parameter uncertainty through model predictions
    /// Returns ensemble predictions with uncertainty estimates
    pub fn propagate_parameter_uncertainty(
        &self,
        base_probabilities: &Array2<f64>,
        parameter_covariance: Option<&Array2<f64>>,
    ) -> Result<EnsemblePredictions> {
        let n_samples = base_probabilities.nrows();
        let n_classes = base_probabilities.ncols();

        // Initialize ensemble predictions
        let mut ensemble_probs = Vec::new();
        ensemble_probs.push(base_probabilities.clone());

        // Generate bootstrap samples if covariance is provided
        if let Some(cov_matrix) = parameter_covariance {
            for _ in 0..self.n_bootstrap {
                let perturbed_probs =
                    self.sample_from_parameter_distribution(base_probabilities, cov_matrix)?;
                ensemble_probs.push(perturbed_probs);
            }
        } else {
            // Use simple perturbation method
            for _ in 0..self.n_bootstrap {
                let perturbed_probs = self.add_parameter_noise(base_probabilities)?;
                ensemble_probs.push(perturbed_probs);
            }
        }

        // Compute ensemble statistics
        let mean_probs = self.compute_ensemble_mean(&ensemble_probs);
        let uncertainty_estimates = self.compute_ensemble_uncertainty(&ensemble_probs);

        Ok(EnsemblePredictions {
            mean_probabilities: mean_probs,
            uncertainty_estimates,
            ensemble_probabilities: ensemble_probs,
        })
    }

    /// Sample from parameter distribution using multivariate normal
    fn sample_from_parameter_distribution(
        &self,
        base_probs: &Array2<f64>,
        covariance: &Array2<f64>,
    ) -> Result<Array2<f64>> {
        let n_samples = base_probs.nrows();
        let n_classes = base_probs.ncols();
        let mut perturbed_probs = base_probs.clone();

        // Simple approximation: add correlated noise based on covariance
        for i in 0..n_samples {
            for j in 0..n_classes {
                let noise_var = if j < covariance.nrows() && j < covariance.ncols() {
                    covariance[[j, j]].max(1e-6)
                } else {
                    0.01
                };

                // Simple Gaussian noise (in practice, would use proper multivariate sampling)
                let noise = self.sample_gaussian(0.0, noise_var.sqrt());
                perturbed_probs[[i, j]] = (base_probs[[i, j]] + noise).clamp(1e-10, 1.0 - 1e-10);
            }

            // Renormalize to ensure probabilities sum to 1
            let row_sum: f64 = perturbed_probs.row(i).sum();
            for j in 0..n_classes {
                perturbed_probs[[i, j]] /= row_sum;
            }
        }

        Ok(perturbed_probs)
    }

    /// Add parameter noise using simple perturbation
    fn add_parameter_noise(&self, base_probs: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = base_probs.nrows();
        let n_classes = base_probs.ncols();
        let mut perturbed_probs = base_probs.clone();

        let noise_scale = 0.05; // 5% noise

        for i in 0..n_samples {
            for j in 0..n_classes {
                let noise = self.sample_gaussian(0.0, noise_scale);
                perturbed_probs[[i, j]] = (base_probs[[i, j]] + noise).clamp(1e-10, 1.0 - 1e-10);
            }

            // Renormalize
            let row_sum: f64 = perturbed_probs.row(i).sum();
            for j in 0..n_classes {
                perturbed_probs[[i, j]] /= row_sum;
            }
        }

        Ok(perturbed_probs)
    }

    /// Simple Gaussian sampling (in practice, would use a proper RNG)
    fn sample_gaussian(&self, mean: f64, std: f64) -> f64 {
        if std == 0.0 {
            return mean;
        }

        if let Some(cached) = {
            let mut cache = BOX_MULLER_CACHE.lock().unwrap();
            cache.take()
        } {
            return mean + std * cached;
        }

        const MULTIPLIER: u64 = 1_103_515_245;
        const INCREMENT: u64 = 12_345;

        let (u1, u2) = {
            let mut state = RNG_STATE.lock().unwrap();
            *state = state.wrapping_mul(MULTIPLIER).wrapping_add(INCREMENT);
            let first = (*state as f64) / (u64::MAX as f64);
            *state = state.wrapping_mul(MULTIPLIER).wrapping_add(INCREMENT);
            let second = (*state as f64) / (u64::MAX as f64);
            (first, second)
        };

        let u1 = u1.clamp(f64::EPSILON, 1.0);
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        let z0 = radius * theta.cos();
        let z1 = radius * theta.sin();

        *BOX_MULLER_CACHE.lock().unwrap() = Some(z1);
        mean + std * z0
    }

    /// Compute ensemble mean
    fn compute_ensemble_mean(&self, ensemble: &[Array2<f64>]) -> Array2<f64> {
        if ensemble.is_empty() {
            return Array2::zeros((0, 0));
        }

        let n_samples = ensemble[0].nrows();
        let n_classes = ensemble[0].ncols();
        let n_models = ensemble.len() as f64;

        let mut mean_probs = Array2::zeros((n_samples, n_classes));

        for model_probs in ensemble {
            for i in 0..n_samples {
                for j in 0..n_classes {
                    mean_probs[[i, j]] += model_probs[[i, j]];
                }
            }
        }

        for i in 0..n_samples {
            for j in 0..n_classes {
                mean_probs[[i, j]] /= n_models;
            }
        }

        mean_probs
    }

    /// Compute ensemble uncertainty (variance across models)
    fn compute_ensemble_uncertainty(&self, ensemble: &[Array2<f64>]) -> Array2<f64> {
        if ensemble.len() < 2 {
            // Return zeros if we don't have enough models for variance
            return Array2::zeros(ensemble[0].dim());
        }

        let mean_probs = self.compute_ensemble_mean(ensemble);
        let n_samples = mean_probs.nrows();
        let n_classes = mean_probs.ncols();
        let n_models = ensemble.len() as f64;

        let mut variance = Array2::zeros((n_samples, n_classes));

        for model_probs in ensemble {
            for i in 0..n_samples {
                for j in 0..n_classes {
                    let diff = model_probs[[i, j]] - mean_probs[[i, j]];
                    variance[[i, j]] += diff * diff;
                }
            }
        }

        for i in 0..n_samples {
            for j in 0..n_classes {
                variance[[i, j]] /= n_models - 1.0; // Unbiased estimator
            }
        }

        variance
    }

    /// Decompose uncertainty into epistemic and aleatoric components
    pub fn decompose_uncertainty(
        &self,
        ensemble_predictions: &EnsemblePredictions,
    ) -> UncertaintyDecomposition {
        let mean_probs = &ensemble_predictions.mean_probabilities;
        let n_samples = mean_probs.nrows();
        let n_classes = mean_probs.ncols();

        let mut epistemic = Array1::zeros(n_samples);
        let mut aleatoric = Array1::zeros(n_samples);

        for i in 0..n_samples {
            // Epistemic uncertainty: variance of mean predictions across models
            let model_variance =
                ensemble_predictions.uncertainty_estimates.row(i).sum() / n_classes as f64;
            epistemic[i] = model_variance;

            // Aleatoric uncertainty: inherent uncertainty in the prediction
            let mean_entropy = self.compute_entropy(mean_probs.row(i));
            aleatoric[i] = mean_entropy;
        }

        let total = &epistemic + &aleatoric;
        UncertaintyDecomposition {
            epistemic,
            aleatoric,
            total,
        }
    }

    /// Compute entropy of a probability distribution
    fn compute_entropy(&self, probs: scirs2_core::ndarray::ArrayView1<f64>) -> f64 {
        probs
            .iter()
            .map(|&p| if p > 1e-15 { -p * p.ln() } else { 0.0 })
            .sum()
    }

    /// Compute mutual information between predictions and parameters
    pub fn compute_mutual_information(&self, ensemble_predictions: &EnsemblePredictions) -> f64 {
        let n_models = ensemble_predictions.ensemble_probabilities.len();
        if n_models < 2 {
            return 0.0;
        }

        let mean_probs = &ensemble_predictions.mean_probabilities;
        let n_samples = mean_probs.nrows();

        let mut mi = 0.0;

        for i in 0..n_samples {
            // Entropy of mean prediction
            let h_mean = self.compute_entropy(mean_probs.row(i));

            // Expected entropy across models
            let mut expected_entropy = 0.0;
            for model_probs in &ensemble_predictions.ensemble_probabilities {
                expected_entropy += self.compute_entropy(model_probs.row(i));
            }
            expected_entropy /= n_models as f64;

            // Mutual information = H(mean) - E[H(individual)]
            mi += h_mean - expected_entropy;
        }

        mi / n_samples as f64
    }
}

/// Ensemble predictions with uncertainty estimates
#[derive(Debug, Clone)]
pub struct EnsemblePredictions {
    /// Mean probabilities across ensemble
    pub mean_probabilities: Array2<f64>,
    /// Uncertainty estimates (variance across models)
    pub uncertainty_estimates: Array2<f64>,
    /// Individual model predictions
    pub ensemble_probabilities: Vec<Array2<f64>>,
}

/// Uncertainty decomposition into epistemic and aleatoric components
#[derive(Debug, Clone)]
pub struct UncertaintyDecomposition {
    /// Epistemic uncertainty (model uncertainty)
    pub epistemic: Array1<f64>,
    /// Aleatoric uncertainty (data uncertainty)
    pub aleatoric: Array1<f64>,
    /// Total uncertainty
    pub total: Array1<f64>,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    // SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
    use scirs2_core::ndarray::array;

    #[test]
    fn test_uncertainty_measures() {
        let probabilities = array![[0.9, 0.1], [0.6, 0.4], [0.5, 0.5], [0.8, 0.2]];

        let uncertainty = StandardUncertainty::new();
        let measures = uncertainty.uncertainty_measures(&probabilities);

        // High confidence should have low entropy
        assert!(measures.entropy[0] < measures.entropy[2]); // 0.9,0.1 vs 0.5,0.5

        // Check confidence values
        assert_abs_diff_eq!(measures.confidence[0], 0.9, epsilon = 1e-10);
        assert_abs_diff_eq!(measures.confidence[2], 0.5, epsilon = 1e-10);

        // Check margins
        assert_abs_diff_eq!(measures.margin[0], 0.8, epsilon = 1e-10); // 0.9 - 0.1
        assert_abs_diff_eq!(measures.margin[2], 0.0, epsilon = 1e-10); // 0.5 - 0.5
    }

    #[test]
    fn test_prediction_intervals() {
        let probabilities = array![[0.9, 0.1], [0.6, 0.4], [0.5, 0.5]];

        let uncertainty = StandardUncertainty::new();
        let intervals = uncertainty
            .prediction_intervals(&probabilities, 0.95)
            .unwrap();

        // Check that intervals are valid
        for i in 0..intervals.nrows() {
            assert!(intervals[[i, 0]] <= intervals[[i, 1]]);
            assert!(intervals[[i, 0]] >= 0.0);
            assert!(intervals[[i, 1]] <= 1.0);
        }
    }

    #[test]
    fn test_outlier_detection() {
        let probabilities = array![
            [0.9, 0.1],   // High confidence - not outlier
            [0.5, 0.5],   // Low confidence - potential outlier
            [0.8, 0.2],   // Medium confidence
            [0.33, 0.67]  // Medium confidence
        ];

        let uncertainty = StandardUncertainty::new();
        let outliers = uncertainty.detect_outliers(&probabilities, 0.6);

        // The second sample (0.5, 0.5) should have high entropy
        // and might be detected as an outlier depending on the threshold
        assert_eq!(outliers.len(), 4);
    }

    #[test]
    fn test_brier_score() {
        let probabilities = array![0.9, 0.8, 0.3, 0.1];
        let true_labels = array![1, 1, 0, 0];

        let score = ReliabilityTests::brier_score(&probabilities, &true_labels).unwrap();

        // Brier score should be between 0 and 1, lower is better
        assert!(score >= 0.0);
        assert!(score <= 1.0);

        // For perfect predictions: (0.9-1)^2 + (0.8-1)^2 + (0.3-0)^2 + (0.1-0)^2 = 0.01 + 0.04 + 0.09 + 0.01 = 0.15
        let expected = (0.01 + 0.04 + 0.09 + 0.01) / 4.0;
        assert_abs_diff_eq!(score, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_platt_scaling() {
        let predictions = array![0.1, 0.4, 0.6, 0.9];
        let true_labels = array![0, 0, 1, 1];

        let result = CalibrationMethod::platt_scaling(&predictions, &true_labels);
        assert!(result.is_ok());

        let (a, b) = result.unwrap();
        // Check that parameters are finite
        assert!(a.is_finite());
        assert!(b.is_finite());
    }

    #[test]
    fn test_isotonic_regression() {
        let predictions = array![0.1, 0.4, 0.6, 0.9];
        let true_labels = array![0, 0, 1, 1];

        let result = CalibrationMethod::isotonic_regression(&predictions, &true_labels).unwrap();

        // Should return some calibration mapping
        assert!(!result.is_empty());

        // Check that mappings are reasonable
        for (score, prob) in result {
            assert!(score.is_finite());
            assert!(prob >= 0.0 && prob <= 1.0);
        }
    }

    #[test]
    fn test_hosmer_lemeshow() {
        let probabilities = array![0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9];
        let true_labels = array![0, 0, 0, 1, 0, 1, 1, 1];

        let chi_square =
            ReliabilityTests::hosmer_lemeshow_test(&probabilities, &true_labels, 4).unwrap();

        // Chi-square statistic should be non-negative
        assert!(chi_square >= 0.0);
        assert!(chi_square.is_finite());
    }
}
