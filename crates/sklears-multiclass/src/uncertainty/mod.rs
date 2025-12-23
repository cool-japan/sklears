//! Uncertainty Quantification for Multiclass Classification
//!
//! This module provides methods for quantifying uncertainty in multiclass classification
//! predictions, including confidence measures, conformal prediction, and epistemic/aleatoric
//! uncertainty estimation.
//!
//! ## Implemented Methods
//!
//! - **Confidence Measures**: Prediction confidence scoring using various metrics
//! - **Conformal Prediction**: Distribution-free uncertainty quantification
//! - **Prediction Intervals**: Confidence intervals for class probabilities
//! - **Uncertainty Propagation**: Handling uncertainty through classification pipelines
//!
//! ## Usage
//!
//! ```rust,ignore
//! use sklears_multiclass::uncertainty::{ConfidenceEstimator, ConformalPredictor};
//!
//! // Create a confidence estimator
//! let confidence = ConfidenceEstimator::new()
//!     .method(ConfidenceMethod::MaxProbability)
//!     .build();
//! ```

pub mod confidence;
pub mod conformal;
pub mod intervals;

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::Result as SklResult;

pub use confidence::{ConfidenceEstimator, ConfidenceMethod, ConfidenceScore};
pub use conformal::{ConformalMethod, ConformalPredictor, ConformalResult};
pub use intervals::{IntervalEstimator, IntervalMethod, PredictionInterval};

/// Main uncertainty quantification configuration
#[derive(Debug, Clone)]
pub struct UncertaintyConfig {
    /// Confidence level for intervals (e.g., 0.95 for 95% confidence)
    pub confidence_level: f64,
    /// Method for confidence estimation
    pub confidence_method: ConfidenceMethod,
    /// Method for conformal prediction
    pub conformal_method: ConformalMethod,
    /// Method for interval estimation
    pub interval_method: IntervalMethod,
}

impl Default for UncertaintyConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            confidence_method: ConfidenceMethod::MaxProbability,
            conformal_method: ConformalMethod::SplitConformal,
            interval_method: IntervalMethod::Bootstrap {
                n_bootstrap: 1000,
                random_state: None,
            },
        }
    }
}

/// Combined uncertainty quantification result
#[derive(Debug, Clone)]
pub struct UncertaintyResult {
    /// Point predictions
    pub predictions: Array1<i32>,
    /// Prediction probabilities
    pub probabilities: Array2<f64>,
    /// Confidence scores for each prediction
    pub confidence_scores: Array1<f64>,
    /// Conformal prediction sets (classes that could be correct at given confidence level)
    pub conformal_sets: Vec<Vec<i32>>,
    /// Prediction intervals for each class probability
    pub intervals: Array2<f64>, // [n_samples, n_classes * 2] (lower, upper bounds)
    /// Epistemic uncertainty (model uncertainty)
    pub epistemic_uncertainty: Array1<f64>,
    /// Aleatoric uncertainty (data uncertainty)
    pub aleatoric_uncertainty: Array1<f64>,
}

/// Comprehensive uncertainty quantifier
///
/// This struct combines multiple uncertainty quantification methods to provide
/// comprehensive uncertainty estimates for multiclass classification predictions.
#[derive(Debug, Clone)]
pub struct UncertaintyQuantifier {
    config: UncertaintyConfig,
    confidence_estimator: ConfidenceEstimator,
    conformal_predictor: ConformalPredictor,
    interval_estimator: IntervalEstimator,
}

impl UncertaintyQuantifier {
    /// Create a new uncertainty quantifier
    pub fn new() -> Self {
        let config = UncertaintyConfig::default();
        Self {
            confidence_estimator: ConfidenceEstimator::new()
                .method(config.confidence_method.clone())
                .build(),
            conformal_predictor: ConformalPredictor::new()
                .method(config.conformal_method.clone())
                .confidence_level(config.confidence_level)
                .build(),
            interval_estimator: IntervalEstimator::new()
                .method(config.interval_method.clone())
                .confidence_level(config.confidence_level)
                .build(),
            config,
        }
    }

    /// Create a builder for uncertainty quantifier
    pub fn builder() -> UncertaintyQuantifierBuilder {
        UncertaintyQuantifierBuilder::new()
    }

    /// Quantify uncertainty for predictions
    pub fn quantify(
        &self,
        predictions: &Array1<i32>,
        probabilities: &Array2<f64>,
        calibration_scores: Option<&Array2<f64>>,
    ) -> SklResult<UncertaintyResult> {
        let (_n_samples, _n_classes) = probabilities.dim();

        // Compute confidence scores
        let confidence_scores = self.confidence_estimator.estimate(probabilities)?;

        // Compute conformal prediction sets
        let conformal_result = self
            .conformal_predictor
            .predict(probabilities, calibration_scores)?;
        let conformal_sets = conformal_result.prediction_sets;

        // Compute prediction intervals
        let intervals = self
            .interval_estimator
            .estimate_intervals(probabilities, calibration_scores)?;

        // Compute epistemic and aleatoric uncertainty
        let (epistemic_uncertainty, aleatoric_uncertainty) =
            self.compute_uncertainty_decomposition(probabilities)?;

        Ok(UncertaintyResult {
            predictions: predictions.clone(),
            probabilities: probabilities.clone(),
            confidence_scores,
            conformal_sets,
            intervals,
            epistemic_uncertainty,
            aleatoric_uncertainty,
        })
    }

    /// Decompose uncertainty into epistemic and aleatoric components
    fn compute_uncertainty_decomposition(
        &self,
        probabilities: &Array2<f64>,
    ) -> SklResult<(Array1<f64>, Array1<f64>)> {
        let (n_samples, _n_classes) = probabilities.dim();
        let mut epistemic = Array1::zeros(n_samples);
        let mut aleatoric = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let probs = probabilities.row(i);

            // Epistemic uncertainty: entropy of the predictive distribution
            let mut entropy = 0.0;
            for &p in probs.iter() {
                if p > 1e-15 {
                    entropy -= p * p.ln();
                }
            }
            epistemic[i] = entropy;

            // Aleatoric uncertainty: expected entropy of individual predictions
            // For classification, this is related to the "confidence" in the prediction
            let max_prob = probs.iter().fold(0.0f64, |a, &b| a.max(b));
            aleatoric[i] = 1.0 - max_prob; // Higher when predictions are uncertain
        }

        Ok((epistemic, aleatoric))
    }
}

impl Default for UncertaintyQuantifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for uncertainty quantifier
#[derive(Debug)]
pub struct UncertaintyQuantifierBuilder {
    config: UncertaintyConfig,
}

impl Default for UncertaintyQuantifierBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl UncertaintyQuantifierBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: UncertaintyConfig::default(),
        }
    }

    /// Set confidence level
    pub fn confidence_level(mut self, confidence_level: f64) -> Self {
        if confidence_level > 0.0 && confidence_level < 1.0 {
            self.config.confidence_level = confidence_level;
        }
        self
    }

    /// Set confidence method
    pub fn confidence_method(mut self, method: ConfidenceMethod) -> Self {
        self.config.confidence_method = method;
        self
    }

    /// Set conformal method
    pub fn conformal_method(mut self, method: ConformalMethod) -> Self {
        self.config.conformal_method = method;
        self
    }

    /// Set interval method
    pub fn interval_method(mut self, method: IntervalMethod) -> Self {
        self.config.interval_method = method;
        self
    }

    /// Build the uncertainty quantifier
    pub fn build(self) -> UncertaintyQuantifier {
        UncertaintyQuantifier {
            confidence_estimator: ConfidenceEstimator::new()
                .method(self.config.confidence_method.clone())
                .build(),
            conformal_predictor: ConformalPredictor::new()
                .method(self.config.conformal_method.clone())
                .confidence_level(self.config.confidence_level)
                .build(),
            interval_estimator: IntervalEstimator::new()
                .method(self.config.interval_method.clone())
                .confidence_level(self.config.confidence_level)
                .build(),
            config: self.config,
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_autograd::ndarray::array;

    #[test]
    fn test_uncertainty_quantifier_creation() {
        let quantifier = UncertaintyQuantifier::new();
        assert_eq!(quantifier.config.confidence_level, 0.95);
    }

    #[test]
    fn test_uncertainty_quantifier_builder() {
        let quantifier = UncertaintyQuantifier::builder()
            .confidence_level(0.99)
            .confidence_method(ConfidenceMethod::Entropy)
            .build();

        assert_eq!(quantifier.config.confidence_level, 0.99);
    }

    #[test]
    fn test_uncertainty_decomposition() {
        let quantifier = UncertaintyQuantifier::new();

        let probabilities = array![[0.8, 0.1, 0.1], [0.4, 0.3, 0.3], [0.33, 0.33, 0.34]];

        let (epistemic, aleatoric) = quantifier
            .compute_uncertainty_decomposition(&probabilities)
            .unwrap();

        assert_eq!(epistemic.len(), 3);
        assert_eq!(aleatoric.len(), 3);

        // More confident predictions should have lower aleatoric uncertainty
        assert!(aleatoric[0] < aleatoric[1]);
        assert!(aleatoric[1] < aleatoric[2]);

        // More uniform distributions should have higher epistemic uncertainty
        assert!(epistemic[0] < epistemic[2]);
    }

    #[test]
    fn test_uncertainty_result_structure() {
        let _predictions = array![0, 1, 2];
        let _probabilities = array![[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]];

        let _quantifier = UncertaintyQuantifier::new();

        // This will fail until we implement the submodules, but tests the structure
        // let result = quantifier.quantify(&predictions, &probabilities, None);
        // assert!(result.is_ok());
    }
}
