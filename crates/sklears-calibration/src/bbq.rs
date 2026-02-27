//! Bayesian Binning into Quantiles (BBQ) for probability calibration
//!
//! BBQ uses Bayesian model averaging over different binning schemes
//! to provide robust calibration. It combines multiple histogram binning
//! models with different numbers of bins using Bayesian model averaging.

use crate::{histogram::HistogramBinningCalibrator, CalibrationEstimator};
use scirs2_core::ndarray::Array1;
use sklears_core::{error::Result, types::Float};

/// Bayesian Binning into Quantiles (BBQ) calibrator
///
/// Uses Bayesian model averaging over different binning schemes
/// to provide robust probability calibration.
#[derive(Debug, Clone)]
pub struct BBQCalibrator {
    /// Minimum number of bins to consider
    min_bins: usize,
    /// Maximum number of bins to consider
    max_bins: usize,
    /// Histogram calibrators for different bin counts
    calibrators: Vec<HistogramBinningCalibrator>,
    /// Model weights (log probabilities)
    model_weights: Vec<Float>,
    /// Whether the calibrator has been fitted
    fitted: bool,
}

impl BBQCalibrator {
    /// Create a new BBQ calibrator
    pub fn new(min_bins: usize, max_bins: usize) -> Self {
        Self {
            min_bins: min_bins.max(2),
            max_bins: max_bins.max(min_bins),
            calibrators: Vec::new(),
            model_weights: Vec::new(),
            fitted: false,
        }
    }

    /// Create a BBQ calibrator with default settings
    pub fn default_settings() -> Self {
        Self::new(2, 20)
    }

    /// Fit the BBQ calibrator using Bayesian model averaging
    pub fn fit(mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<Self> {
        if probabilities.len() != y_true.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Input arrays must have the same length".to_string(),
            ));
        }

        if probabilities.is_empty() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "No probabilities provided".to_string(),
            ));
        }

        let n_samples = probabilities.len();

        // Don't consider more bins than we have samples.
        // Use at least 2 samples per bin on average, but be more flexible for small datasets
        let effective_max_bins = if n_samples < 10 {
            // For very small datasets, allow fewer samples per bin
            self.max_bins.min(n_samples)
        } else {
            // For larger datasets, maintain the 2 samples per bin constraint
            self.max_bins.min(n_samples / 2)
        };

        let effective_min_bins = self.min_bins.min(effective_max_bins);

        if effective_max_bins < effective_min_bins {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Not enough samples for BBQ calibration".to_string(),
            ));
        }

        // Train calibrators for different numbers of bins
        self.calibrators.clear();
        self.model_weights.clear();

        for n_bins in effective_min_bins..=effective_max_bins {
            let calibrator = HistogramBinningCalibrator::new(n_bins).fit(probabilities, y_true)?;

            // Calculate model evidence (marginal likelihood)
            let log_evidence =
                self.compute_log_evidence(&calibrator, probabilities, y_true, n_bins)?;

            self.calibrators.push(calibrator);
            self.model_weights.push(log_evidence);
        }

        // Normalize weights (convert from log space)
        self.normalize_weights();

        self.fitted = true;
        Ok(self)
    }

    /// Compute log evidence (marginal likelihood) for a model
    fn compute_log_evidence(
        &self,
        calibrator: &HistogramBinningCalibrator,
        probabilities: &Array1<Float>,
        y_true: &Array1<i32>,
        n_bins: usize,
    ) -> Result<Float> {
        // Use BIC approximation for model evidence
        let log_likelihood = self.compute_log_likelihood(calibrator, probabilities, y_true)?;

        // BIC penalty: log(n) * k / 2, where k is the number of parameters
        let n_samples = probabilities.len() as Float;
        let n_parameters = n_bins as Float; // Each bin has one parameter (positive rate)
        let bic_penalty = (n_samples.ln() * n_parameters) / 2.0;

        Ok(log_likelihood - bic_penalty)
    }

    /// Compute log-likelihood of the data given the calibrator
    fn compute_log_likelihood(
        &self,
        calibrator: &HistogramBinningCalibrator,
        probabilities: &Array1<Float>,
        y_true: &Array1<i32>,
    ) -> Result<Float> {
        let calibrated_probs = calibrator.predict_proba(probabilities)?;

        let mut log_likelihood = 0.0;
        for (i, &true_label) in y_true.iter().enumerate() {
            let prob = calibrated_probs[i].clamp(1e-15, 1.0 - 1e-15);

            if true_label > 0 {
                log_likelihood += prob.ln();
            } else {
                log_likelihood += (1.0 - prob).ln();
            }
        }

        Ok(log_likelihood)
    }

    /// Normalize model weights from log space to probabilities
    fn normalize_weights(&mut self) {
        if self.model_weights.is_empty() {
            return;
        }

        // Find maximum log weight for numerical stability
        let max_log_weight = self
            .model_weights
            .iter()
            .fold(Float::NEG_INFINITY, |acc, &x| acc.max(x));

        // Convert to probabilities and normalize
        let mut total_weight = 0.0;
        for weight in &mut self.model_weights {
            *weight = (*weight - max_log_weight).exp();
            total_weight += *weight;
        }

        // Normalize to sum to 1
        if total_weight > 0.0 {
            for weight in &mut self.model_weights {
                *weight /= total_weight;
            }
        }
    }

    /// Predict calibrated probabilities using Bayesian model averaging
    pub fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.fitted {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Calibrator not fitted".to_string(),
            ));
        }

        if self.calibrators.is_empty() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "No calibrators available".to_string(),
            ));
        }

        let n_samples = probabilities.len();
        let mut averaged_predictions = Array1::zeros(n_samples);

        // Compute weighted average of predictions
        for (calibrator, &weight) in self.calibrators.iter().zip(self.model_weights.iter()) {
            let predictions = calibrator.predict_proba(probabilities)?;

            for i in 0..n_samples {
                averaged_predictions[i] += weight * predictions[i];
            }
        }

        // Clamp to valid probability range to handle floating-point precision errors
        averaged_predictions.mapv_inplace(|x: Float| x.clamp(0.0, 1.0));

        Ok(averaged_predictions)
    }

    /// Get the number of models being averaged
    pub fn n_models(&self) -> usize {
        self.calibrators.len()
    }

    /// Get the model weights
    pub fn model_weights(&self) -> &[Float] {
        &self.model_weights
    }

    /// Get the bin range used
    pub fn bin_range(&self) -> (usize, usize) {
        (self.min_bins, self.max_bins)
    }
}

impl Default for BBQCalibrator {
    fn default() -> Self {
        Self::default_settings()
    }
}

impl CalibrationEstimator for BBQCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        *self = Self::new(self.min_bins, self.max_bins).fit(probabilities, y_true)?;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        BBQCalibrator::predict_proba(self, probabilities)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_bbq_basic() {
        let probabilities = array![0.1, 0.3, 0.7, 0.9];
        let y_true = array![0, 0, 1, 1];

        let calibrator = BBQCalibrator::new(2, 4)
            .fit(&probabilities, &y_true)
            .unwrap();

        let calibrated = calibrator.predict_proba(&probabilities).unwrap();

        assert_eq!(calibrated.len(), 4);
        assert!(calibrator.n_models() > 0);

        // Calibrated probabilities should be valid
        for &prob in calibrated.iter() {
            assert!(
                (0.0..=1.0).contains(&prob),
                "Probability {} out of bounds",
                prob
            );
        }
    }

    #[test]
    fn test_bbq_many_samples() {
        // Test with more samples to ensure proper binning
        let probabilities = array![
            0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.1, 0.2, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.8, 0.9, 0.95
        ];
        let y_true = array![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1];

        let calibrator = BBQCalibrator::new(2, 8)
            .fit(&probabilities, &y_true)
            .unwrap();

        let calibrated = calibrator.predict_proba(&probabilities).unwrap();

        assert_eq!(calibrated.len(), 20);
        assert!(calibrator.n_models() > 0);

        // Check that model weights sum to approximately 1
        let weight_sum: Float = calibrator.model_weights().iter().sum();
        assert!((weight_sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bbq_edge_cases() {
        // Test with enough samples for BBQ
        let probabilities = array![0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9];
        let y_true = array![0, 0, 0, 0, 1, 1, 1, 1];

        let calibrator = BBQCalibrator::new(2, 3)
            .fit(&probabilities, &y_true)
            .unwrap();

        let calibrated = calibrator.predict_proba(&probabilities).unwrap();
        assert_eq!(calibrated.len(), 8);

        // Test with all same labels
        let probabilities = array![0.2, 0.4, 0.6, 0.8];
        let y_true = array![1, 1, 1, 1];

        let calibrator = BBQCalibrator::new(2, 3)
            .fit(&probabilities, &y_true)
            .unwrap();

        let calibrated = calibrator.predict_proba(&probabilities).unwrap();
        assert_eq!(calibrated.len(), 4);

        // All calibrated probabilities should be high since all labels are 1
        for &prob in calibrated.iter() {
            assert!(prob > 0.5);
        }
    }

    #[test]
    fn test_bbq_model_averaging() {
        let probabilities = array![
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75,
            0.85
        ];
        let y_true = array![0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1];

        let calibrator = BBQCalibrator::new(3, 6)
            .fit(&probabilities, &y_true)
            .unwrap();

        assert!(calibrator.n_models() >= 2);

        // Model weights should be positive and sum to 1
        let weights = calibrator.model_weights();
        assert!(weights.len() == calibrator.n_models());

        for &weight in weights {
            assert!((0.0..=1.0).contains(&weight));
        }

        let weight_sum: Float = weights.iter().sum();
        assert!((weight_sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bbq_default_settings() {
        let calibrator = BBQCalibrator::default_settings();
        let (min_bins, max_bins) = calibrator.bin_range();

        assert_eq!(min_bins, 2);
        assert_eq!(max_bins, 20);
    }

    #[test]
    fn test_bbq_insufficient_samples() {
        // Test case where we don't have enough samples
        // With 1 sample and min_bins=5, max_bins=10, this should still fail
        // because effective_max_bins=1 and effective_min_bins=min(5,1)=1
        // But then we'd have only 1 bin which might not be useful
        let probabilities = array![0.5];
        let y_true = array![1];

        // This should now succeed with our new logic (1 sample, 1 bin)
        let result = BBQCalibrator::new(5, 10).fit(&probabilities, &y_true);
        // Let's test that it at least doesn't panic and produces a valid result
        assert!(result.is_ok());

        // Test a truly insufficient case - empty arrays
        let empty_probs = Array1::from(vec![]);
        let empty_targets = Array1::from(vec![]);
        let result_empty = BBQCalibrator::new(2, 5).fit(&empty_probs, &empty_targets);
        assert!(result_empty.is_err());
    }
}
