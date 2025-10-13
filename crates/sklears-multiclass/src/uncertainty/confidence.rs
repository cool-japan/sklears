//! Confidence Estimation for Multiclass Classification
//!
//! This module provides various methods for estimating prediction confidence
//! in multiclass classification problems.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::{Result as SklResult, SklearsError};

/// Methods for confidence estimation
#[derive(Debug, Clone, PartialEq)]
pub enum ConfidenceMethod {
    /// Maximum probability across all classes
    MaxProbability,
    /// Entropy-based confidence (negative entropy)
    Entropy,
    /// Margin between top two classes
    Margin,
    /// Ratio between top two classes
    Ratio,
    /// Least confident prediction
    LeastConfident,
    /// Margin of confidence (difference between most and least confident)
    MarginOfConfidence,
    /// Combined method using multiple metrics
    Combined(Vec<ConfidenceMethod>),
}

impl Default for ConfidenceMethod {
    fn default() -> Self {
        Self::MaxProbability
    }
}

/// Confidence score with metadata
#[derive(Debug, Clone)]
pub struct ConfidenceScore {
    /// The confidence value (0.0 to 1.0, higher is more confident)
    pub score: f64,
    /// The method used to compute the confidence
    pub method: ConfidenceMethod,
    /// Additional metadata about the computation
    pub metadata: ConfidenceMetadata,
}

/// Additional metadata for confidence scores
#[derive(Debug, Clone)]
pub struct ConfidenceMetadata {
    /// Maximum probability value
    pub max_probability: f64,
    /// Second highest probability value
    pub second_max_probability: f64,
    /// Entropy of the probability distribution
    pub entropy: f64,
    /// Index of the predicted class
    pub predicted_class_idx: usize,
}

/// Confidence estimator for multiclass classification
#[derive(Debug, Clone)]
pub struct ConfidenceEstimator {
    method: ConfidenceMethod,
    temperature: f64, // For temperature scaling
}

impl ConfidenceEstimator {
    /// Create a new confidence estimator
    pub fn new() -> Self {
        Self {
            method: ConfidenceMethod::default(),
            temperature: 1.0,
        }
    }

    /// Create a builder for the confidence estimator
    pub fn builder() -> ConfidenceEstimatorBuilder {
        ConfidenceEstimatorBuilder::new()
    }

    /// Set the confidence method
    pub fn method(mut self, method: ConfidenceMethod) -> Self {
        self.method = method;
        self
    }

    /// Set temperature for calibration
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature.max(1e-6);
        self
    }

    /// Build the estimator
    pub fn build(self) -> Self {
        self
    }

    /// Estimate confidence for probability predictions
    pub fn estimate(&self, probabilities: &Array2<f64>) -> SklResult<Array1<f64>> {
        let (n_samples, _) = probabilities.dim();
        let mut confidence_scores = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let probs = probabilities.row(i);
            confidence_scores[i] = self.compute_confidence(&probs.to_owned())?;
        }

        Ok(confidence_scores)
    }

    /// Estimate confidence with detailed metadata
    pub fn estimate_detailed(
        &self,
        probabilities: &Array2<f64>,
    ) -> SklResult<Vec<ConfidenceScore>> {
        let (n_samples, _) = probabilities.dim();
        let mut scores = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let probs = probabilities.row(i);
            let score = self.compute_confidence_detailed(&probs.to_owned())?;
            scores.push(score);
        }

        Ok(scores)
    }

    /// Compute confidence for a single probability distribution
    fn compute_confidence(&self, probabilities: &Array1<f64>) -> SklResult<f64> {
        match &self.method {
            ConfidenceMethod::MaxProbability => self.max_probability_confidence(probabilities),
            ConfidenceMethod::Entropy => self.entropy_confidence(probabilities),
            ConfidenceMethod::Margin => self.margin_confidence(probabilities),
            ConfidenceMethod::Ratio => self.ratio_confidence(probabilities),
            ConfidenceMethod::LeastConfident => self.least_confident(probabilities),
            ConfidenceMethod::MarginOfConfidence => self.margin_of_confidence(probabilities),
            ConfidenceMethod::Combined(methods) => self.combined_confidence(probabilities, methods),
        }
    }

    /// Compute confidence with detailed metadata
    fn compute_confidence_detailed(
        &self,
        probabilities: &Array1<f64>,
    ) -> SklResult<ConfidenceScore> {
        let metadata = self.compute_metadata(probabilities);
        let score = self.compute_confidence(probabilities)?;

        Ok(ConfidenceScore {
            score,
            method: self.method.clone(),
            metadata,
        })
    }

    /// Compute metadata for a probability distribution
    fn compute_metadata(&self, probabilities: &Array1<f64>) -> ConfidenceMetadata {
        let mut sorted_probs: Vec<(usize, f64)> = probabilities
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        sorted_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let max_probability = sorted_probs[0].1;
        let second_max_probability = if sorted_probs.len() > 1 {
            sorted_probs[1].1
        } else {
            0.0
        };
        let predicted_class_idx = sorted_probs[0].0;

        // Compute entropy
        let mut entropy = 0.0;
        for &p in probabilities.iter() {
            if p > 1e-15 {
                entropy -= p * p.ln();
            }
        }

        ConfidenceMetadata {
            max_probability,
            second_max_probability,
            entropy,
            predicted_class_idx,
        }
    }

    /// Maximum probability confidence
    fn max_probability_confidence(&self, probabilities: &Array1<f64>) -> SklResult<f64> {
        let max_prob = probabilities.iter().fold(0.0f64, |a, &b| a.max(b));
        Ok(max_prob)
    }

    /// Entropy-based confidence (negative normalized entropy)
    fn entropy_confidence(&self, probabilities: &Array1<f64>) -> SklResult<f64> {
        let mut entropy = 0.0;
        for &p in probabilities.iter() {
            if p > 1e-15 {
                entropy -= p * p.ln();
            }
        }

        // Normalize entropy by maximum possible entropy (uniform distribution)
        let max_entropy = (probabilities.len() as f64).ln();
        let normalized_entropy = if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        };

        // Confidence is 1 - normalized_entropy
        Ok(1.0 - normalized_entropy)
    }

    /// Margin confidence (difference between top two probabilities)
    fn margin_confidence(&self, probabilities: &Array1<f64>) -> SklResult<f64> {
        let mut sorted_probs: Vec<f64> = probabilities.to_vec();
        sorted_probs.sort_by(|a, b| b.partial_cmp(a).unwrap());

        if sorted_probs.len() < 2 {
            return Ok(sorted_probs.first().copied().unwrap_or(0.0));
        }

        let margin = sorted_probs[0] - sorted_probs[1];
        Ok(margin)
    }

    /// Ratio confidence (ratio between top two probabilities)
    fn ratio_confidence(&self, probabilities: &Array1<f64>) -> SklResult<f64> {
        let mut sorted_probs: Vec<f64> = probabilities.to_vec();
        sorted_probs.sort_by(|a, b| b.partial_cmp(a).unwrap());

        if sorted_probs.len() < 2 || sorted_probs[1] <= 1e-15 {
            return Ok(1.0);
        }

        let ratio = sorted_probs[0] / sorted_probs[1];
        // Normalize to [0, 1] range - ratio of 1 means equal probabilities (low confidence)
        // Higher ratios indicate higher confidence
        Ok((ratio - 1.0) / ratio)
    }

    /// Least confident prediction
    fn least_confident(&self, probabilities: &Array1<f64>) -> SklResult<f64> {
        let max_prob = probabilities.iter().fold(0.0f64, |a, &b| a.max(b));

        // Least confident is just 1 - max_probability in multiclass setting
        Ok(1.0 - max_prob)
    }

    /// Margin of confidence (difference between most and least confident)
    fn margin_of_confidence(&self, probabilities: &Array1<f64>) -> SklResult<f64> {
        let max_prob = probabilities.iter().fold(0.0f64, |a, &b| a.max(b));
        let min_prob = probabilities.iter().fold(1.0f64, |a, &b| a.min(b));

        Ok(max_prob - min_prob)
    }

    /// Combined confidence using multiple methods
    fn combined_confidence(
        &self,
        probabilities: &Array1<f64>,
        methods: &[ConfidenceMethod],
    ) -> SklResult<f64> {
        if methods.is_empty() {
            return Ok(0.0);
        }

        let mut total_confidence = 0.0;
        let mut valid_methods = 0;

        for method in methods {
            // Temporarily change method to compute confidence
            let temp_estimator = ConfidenceEstimator {
                method: method.clone(),
                temperature: self.temperature,
            };

            match temp_estimator.compute_confidence(probabilities) {
                Ok(confidence) => {
                    total_confidence += confidence;
                    valid_methods += 1;
                }
                Err(_) => continue, // Skip invalid methods
            }
        }

        if valid_methods == 0 {
            return Err(SklearsError::InvalidInput(
                "No valid confidence methods provided".to_string(),
            ));
        }

        Ok(total_confidence / valid_methods as f64)
    }

    /// Apply temperature scaling to probabilities
    pub fn apply_temperature_scaling(&self, probabilities: &Array1<f64>) -> Array1<f64> {
        if (self.temperature - 1.0).abs() < 1e-10 {
            return probabilities.clone();
        }

        // Convert probabilities to logits
        let mut logits = Array1::zeros(probabilities.len());
        for (i, &p) in probabilities.iter().enumerate() {
            let p_clipped = p.max(1e-15).min(1.0 - 1e-15);
            logits[i] = p_clipped.ln();
        }

        // Apply temperature scaling
        for logit in logits.iter_mut() {
            *logit /= self.temperature;
        }

        // Convert back to probabilities (softmax)
        let max_logit = logits.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mut exp_logits = Array1::zeros(logits.len());
        for (i, &logit) in logits.iter().enumerate() {
            exp_logits[i] = (logit - max_logit).exp();
        }

        let sum_exp: f64 = exp_logits.sum();
        for exp_logit in exp_logits.iter_mut() {
            *exp_logit /= sum_exp;
        }

        exp_logits
    }
}

impl Default for ConfidenceEstimator {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for confidence estimator
#[derive(Debug)]
pub struct ConfidenceEstimatorBuilder {
    method: ConfidenceMethod,
    temperature: f64,
}

impl Default for ConfidenceEstimatorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfidenceEstimatorBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            method: ConfidenceMethod::default(),
            temperature: 1.0,
        }
    }

    /// Set the confidence method
    pub fn method(mut self, method: ConfidenceMethod) -> Self {
        self.method = method;
        self
    }

    /// Set temperature for calibration
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature.max(1e-6);
        self
    }

    /// Build the confidence estimator
    pub fn build(self) -> ConfidenceEstimator {
        ConfidenceEstimator {
            method: self.method,
            temperature: self.temperature,
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_autograd::ndarray::array;

    #[test]
    fn test_max_probability_confidence() {
        let estimator = ConfidenceEstimator::new()
            .method(ConfidenceMethod::MaxProbability)
            .build();

        let probabilities = array![[0.8, 0.1, 0.1], [0.4, 0.3, 0.3]];

        let confidence = estimator.estimate(&probabilities).unwrap();

        assert_eq!(confidence.len(), 2);
        assert!((confidence[0] - 0.8).abs() < 1e-10);
        assert!((confidence[1] - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_confidence() {
        let estimator = ConfidenceEstimator::new()
            .method(ConfidenceMethod::Entropy)
            .build();

        let probabilities = array![[1.0, 0.0, 0.0], [0.33, 0.33, 0.34]];

        let confidence = estimator.estimate(&probabilities).unwrap();

        assert_eq!(confidence.len(), 2);
        // Perfect prediction (1.0, 0.0, 0.0) should have high confidence
        assert!(confidence[0] > 0.9);
        // Uniform prediction should have low confidence
        assert!(confidence[1] < 0.1);
    }

    #[test]
    fn test_margin_confidence() {
        let estimator = ConfidenceEstimator::new()
            .method(ConfidenceMethod::Margin)
            .build();

        let probabilities = array![[0.8, 0.1, 0.1], [0.4, 0.35, 0.25]];

        let confidence = estimator.estimate(&probabilities).unwrap();

        assert_eq!(confidence.len(), 2);
        // First prediction has larger margin (0.8 - 0.1 = 0.7)
        // Second prediction has smaller margin (0.4 - 0.35 = 0.05)
        assert!(confidence[0] > confidence[1]);
        assert!((confidence[0] - 0.7).abs() < 1e-10);
        assert!((confidence[1] - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_combined_confidence() {
        let methods = vec![ConfidenceMethod::MaxProbability, ConfidenceMethod::Margin];

        let estimator = ConfidenceEstimator::new()
            .method(ConfidenceMethod::Combined(methods))
            .build();

        let probabilities = array![[0.8, 0.1, 0.1]];

        let confidence = estimator.estimate(&probabilities).unwrap();

        assert_eq!(confidence.len(), 1);
        // Should be average of max_prob (0.8) and margin (0.7) = 0.75
        assert!((confidence[0] - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_confidence_detailed() {
        let estimator = ConfidenceEstimator::new()
            .method(ConfidenceMethod::MaxProbability)
            .build();

        let probabilities = array![[0.8, 0.1, 0.1]];

        let detailed = estimator.estimate_detailed(&probabilities).unwrap();

        assert_eq!(detailed.len(), 1);
        let score = &detailed[0];
        assert!((score.score - 0.8).abs() < 1e-10);
        assert!((score.metadata.max_probability - 0.8).abs() < 1e-10);
        assert!((score.metadata.second_max_probability - 0.1).abs() < 1e-10);
        assert_eq!(score.metadata.predicted_class_idx, 0);
    }

    #[test]
    fn test_temperature_scaling() {
        let estimator = ConfidenceEstimator::new().temperature(2.0).build();

        let probabilities = array![0.9, 0.05, 0.05];

        let scaled = estimator.apply_temperature_scaling(&probabilities);

        // Temperature scaling should make the distribution less extreme
        assert!(scaled[0] < probabilities[0]); // Max probability should decrease
        assert!(scaled[1] > probabilities[1]); // Other probabilities should increase
        assert!(scaled[2] > probabilities[2]);

        // Probabilities should still sum to 1
        let sum: f64 = scaled.sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_confidence_builder() {
        let estimator = ConfidenceEstimator::builder()
            .method(ConfidenceMethod::Entropy)
            .temperature(1.5)
            .build();

        assert_eq!(estimator.method, ConfidenceMethod::Entropy);
        assert!((estimator.temperature - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_ratio_confidence() {
        let estimator = ConfidenceEstimator::new()
            .method(ConfidenceMethod::Ratio)
            .build();

        let probabilities = array![[0.8, 0.2], [0.5, 0.5]];

        let confidence = estimator.estimate(&probabilities).unwrap();

        assert_eq!(confidence.len(), 2);
        // First has ratio 0.8/0.2 = 4, confidence = (4-1)/4 = 0.75
        assert!((confidence[0] - 0.75).abs() < 1e-10);
        // Second has ratio 0.5/0.5 = 1, confidence = (1-1)/1 = 0
        assert!((confidence[1] - 0.0).abs() < 1e-10);
    }
}
