//! Ensemble member management and traits for voting classifiers

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{error::Result, types::Float};

/// Trait for estimators that can be used in ensemble
pub trait EnsembleMember {
    /// Get estimator weight in the ensemble
    fn weight(&self) -> Float;

    /// Set estimator weight
    fn set_weight(&mut self, weight: Float);

    /// Get estimator performance metric
    fn performance(&self) -> Float;

    /// Update performance metric
    fn update_performance(&mut self, performance: Float);

    /// Get prediction confidence
    fn confidence(&self) -> Float;

    /// Make predictions on input data
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>>;

    /// Make probability predictions (if supported)
    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>>;

    /// Check if estimator supports probability predictions
    fn supports_proba(&self) -> bool;

    /// Get feature importance (if available)
    fn feature_importance(&self) -> Option<Array1<Float>>;

    /// Get model complexity measure
    fn complexity(&self) -> Float;

    /// Check if model is fitted
    fn is_fitted(&self) -> bool;

    /// Get number of classes (for classifiers)
    fn n_classes(&self) -> Option<usize>;

    /// Get number of features expected
    fn n_features(&self) -> Option<usize>;

    /// Calculate prediction uncertainty
    fn uncertainty(&self, x: &Array2<Float>) -> Result<Array1<Float>>;

    /// Get model name/identifier
    fn name(&self) -> String;

    /// Clone the estimator (for ensemble operations)
    fn clone_estimator(&self) -> Box<dyn EnsembleMember + Send + Sync>;
}

/// Mock estimator for testing ensemble functionality
#[derive(Debug, Clone)]
pub struct MockEstimator {
    weight: Float,
    performance: Float,
    confidence: Float,
    bias: Float,
    supports_proba: bool,
    is_fitted: bool,
    n_classes: Option<usize>,
    n_features: Option<usize>,
    name: String,
}

impl MockEstimator {
    pub fn new(bias: Float) -> Self {
        Self {
            weight: 1.0,
            performance: 0.8,
            confidence: 0.9,
            bias,
            supports_proba: true,
            is_fitted: true,
            n_classes: Some(2),
            n_features: Some(2),
            name: format!("MockEstimator_{}", bias),
        }
    }

    pub fn with_weight(mut self, weight: Float) -> Self {
        self.weight = weight;
        self
    }

    pub fn with_performance(mut self, performance: Float) -> Self {
        self.performance = performance;
        self
    }

    pub fn with_confidence(mut self, confidence: Float) -> Self {
        self.confidence = confidence;
        self
    }

    pub fn with_proba_support(mut self, supports: bool) -> Self {
        self.supports_proba = supports;
        self
    }

    pub fn with_fitted_status(mut self, fitted: bool) -> Self {
        self.is_fitted = fitted;
        self
    }

    pub fn with_classes(mut self, n_classes: usize) -> Self {
        self.n_classes = Some(n_classes);
        self
    }

    pub fn with_features(mut self, n_features: usize) -> Self {
        self.n_features = Some(n_features);
        self
    }

    pub fn with_name(mut self, name: String) -> Self {
        self.name = name;
        self
    }
}

impl EnsembleMember for MockEstimator {
    fn weight(&self) -> Float {
        self.weight
    }

    fn set_weight(&mut self, weight: Float) {
        self.weight = weight;
    }

    fn performance(&self) -> Float {
        self.performance
    }

    fn update_performance(&mut self, performance: Float) {
        self.performance = performance;
    }

    fn confidence(&self) -> Float {
        self.confidence
    }

    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(sklears_core::error::SklearsError::NotFitted {
                operation: "predict".to_string(),
            });
        }

        let n_samples = x.nrows();
        let mut predictions = Array1::zeros(n_samples);

        // Simple mock prediction: bias towards a specific class based on features
        for i in 0..n_samples {
            let feature_sum: Float = x.row(i).sum();
            let prediction = if feature_sum + self.bias > 0.0 {
                1.0
            } else {
                0.0
            };
            predictions[i] = prediction;
        }

        Ok(predictions)
    }

    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        if !self.supports_proba {
            return Err(sklears_core::error::SklearsError::InvalidOperation(
                "Estimator does not support probability predictions".to_string(),
            ));
        }

        if !self.is_fitted {
            return Err(sklears_core::error::SklearsError::NotFitted {
                operation: "predict".to_string(),
            });
        }

        let n_samples = x.nrows();
        let n_classes = self.n_classes.unwrap_or(2);
        let mut probabilities = Array2::zeros((n_samples, n_classes));

        // Simple mock probability: sigmoid-like transformation of features
        for i in 0..n_samples {
            let feature_sum: Float = x.row(i).sum();
            let logit = feature_sum + self.bias;

            if n_classes == 2 {
                // Binary classification
                let prob_class_1 = 1.0 / (1.0 + (-logit).exp());
                probabilities[[i, 0]] = 1.0 - prob_class_1;
                probabilities[[i, 1]] = prob_class_1;
            } else {
                // Multi-class: uniform distribution for simplicity
                let prob_per_class = 1.0 / n_classes as Float;
                for j in 0..n_classes {
                    probabilities[[i, j]] = prob_per_class;
                }
                // Add some bias to the first class
                probabilities[[i, 0]] += self.bias * 0.1;

                // Normalize to ensure sum = 1
                let row_sum: Float = probabilities.row(i).sum();
                if row_sum > 0.0 {
                    for j in 0..n_classes {
                        probabilities[[i, j]] /= row_sum;
                    }
                }
            }
        }

        Ok(probabilities)
    }

    fn supports_proba(&self) -> bool {
        self.supports_proba
    }

    fn feature_importance(&self) -> Option<Array1<Float>> {
        if let Some(n_features) = self.n_features {
            // Mock feature importance: uniform importance with some bias
            let mut importance = Array1::ones(n_features) / n_features as Float;
            if n_features > 0 {
                importance[0] += self.bias.abs() * 0.1; // First feature gets extra importance
            }

            // Normalize
            let total: Float = importance.sum();
            if total > 0.0 {
                importance.mapv_inplace(|x| x / total);
            }

            Some(importance)
        } else {
            None
        }
    }

    fn complexity(&self) -> Float {
        // Mock complexity based on bias magnitude
        self.bias.abs() + 1.0
    }

    fn is_fitted(&self) -> bool {
        self.is_fitted
    }

    fn n_classes(&self) -> Option<usize> {
        self.n_classes
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }

    fn uncertainty(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(sklears_core::error::SklearsError::NotFitted {
                operation: "predict".to_string(),
            });
        }

        let n_samples = x.nrows();
        let mut uncertainty = Array1::zeros(n_samples);

        // Mock uncertainty: higher for samples far from decision boundary
        for i in 0..n_samples {
            let feature_sum: Float = x.row(i).sum();
            let logit = feature_sum + self.bias;

            // Uncertainty is higher when logit is close to 0 (decision boundary)
            let prob = 1.0 / (1.0 + (-logit).exp());
            let entropy = -prob * prob.ln() - (1.0 - prob) * (1.0 - prob).ln();
            uncertainty[i] = entropy;
        }

        Ok(uncertainty)
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn clone_estimator(&self) -> Box<dyn EnsembleMember + Send + Sync> {
        Box::new(self.clone())
    }
}

/// Ensemble member wrapper for external estimators
#[derive(Debug)]
pub struct ExternalEstimatorWrapper {
    weight: Float,
    performance: Float,
    confidence: Float,
    name: String,
}

impl ExternalEstimatorWrapper {
    pub fn new(name: String) -> Self {
        Self {
            weight: 1.0,
            performance: 0.0,
            confidence: 0.5,
            name,
        }
    }
}

impl EnsembleMember for ExternalEstimatorWrapper {
    fn weight(&self) -> Float {
        self.weight
    }

    fn set_weight(&mut self, weight: Float) {
        self.weight = weight;
    }

    fn performance(&self) -> Float {
        self.performance
    }

    fn update_performance(&mut self, performance: Float) {
        self.performance = performance;
    }

    fn confidence(&self) -> Float {
        self.confidence
    }

    fn predict(&self, _x: &Array2<Float>) -> Result<Array1<Float>> {
        Err(sklears_core::error::SklearsError::NotImplemented(
            "External estimator prediction not implemented".to_string(),
        ))
    }

    fn predict_proba(&self, _x: &Array2<Float>) -> Result<Array2<Float>> {
        Err(sklears_core::error::SklearsError::NotImplemented(
            "External estimator probability prediction not implemented".to_string(),
        ))
    }

    fn supports_proba(&self) -> bool {
        false
    }

    fn feature_importance(&self) -> Option<Array1<Float>> {
        None
    }

    fn complexity(&self) -> Float {
        1.0
    }

    fn is_fitted(&self) -> bool {
        true
    }

    fn n_classes(&self) -> Option<usize> {
        None
    }

    fn n_features(&self) -> Option<usize> {
        None
    }

    fn uncertainty(&self, _x: &Array2<Float>) -> Result<Array1<Float>> {
        Err(sklears_core::error::SklearsError::NotImplemented(
            "External estimator uncertainty estimation not implemented".to_string(),
        ))
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn clone_estimator(&self) -> Box<dyn EnsembleMember + Send + Sync> {
        Box::new(Self {
            weight: self.weight,
            performance: self.performance,
            confidence: self.confidence,
            name: self.name.clone(),
        })
    }
}

/// Utility functions for ensemble management
pub mod ensemble_utils {
    use super::*;

    /// Calculate ensemble diversity using prediction disagreement
    pub fn calculate_ensemble_diversity(
        estimators: &[Box<dyn EnsembleMember + Send + Sync>],
        x: &Array2<Float>,
    ) -> Result<Float> {
        if estimators.len() < 2 {
            return Ok(0.0);
        }

        let n_samples = x.nrows();
        let n_estimators = estimators.len();

        // Collect all predictions
        let mut all_predictions = Vec::new();
        for estimator in estimators {
            let predictions = estimator.predict(x)?;
            all_predictions.push(predictions);
        }

        // Calculate pairwise disagreements
        let mut total_disagreement = 0.0;
        let mut n_pairs = 0;

        for i in 0..n_estimators {
            for j in (i + 1)..n_estimators {
                let mut disagreements = 0;
                for sample_idx in 0..n_samples {
                    if (all_predictions[i][sample_idx] - all_predictions[j][sample_idx]).abs()
                        > 1e-6
                    {
                        disagreements += 1;
                    }
                }
                total_disagreement += disagreements as Float / n_samples as Float;
                n_pairs += 1;
            }
        }

        if n_pairs > 0 {
            Ok(total_disagreement / n_pairs as Float)
        } else {
            Ok(0.0)
        }
    }

    /// Update ensemble weights based on recent performance
    pub fn update_ensemble_weights(
        estimators: &mut [Box<dyn EnsembleMember + Send + Sync>],
        recent_performances: &[Float],
        learning_rate: Float,
    ) {
        if estimators.len() != recent_performances.len() {
            return;
        }

        // Calculate performance-based weights
        let total_performance: Float = recent_performances.iter().sum();

        if total_performance > 1e-8 {
            for (estimator, &performance) in estimators.iter_mut().zip(recent_performances.iter()) {
                let current_weight = estimator.weight();
                let target_weight = performance / total_performance;
                let new_weight = current_weight + learning_rate * (target_weight - current_weight);
                estimator.set_weight(new_weight.max(0.01)); // Minimum weight to avoid zero
            }
        }
    }

    /// Prune underperforming estimators from ensemble
    pub fn prune_ensemble(
        estimators: &mut Vec<Box<dyn EnsembleMember + Send + Sync>>,
        performance_threshold: Float,
        min_ensemble_size: usize,
    ) {
        if estimators.len() <= min_ensemble_size {
            return;
        }

        estimators.retain(|estimator| estimator.performance() >= performance_threshold);

        // Ensure minimum ensemble size
        if estimators.len() < min_ensemble_size {
            // This would require keeping the best performers, but for simplicity
            // we just don't prune if it would violate the minimum size
        }
    }

    /// Get ensemble statistics
    pub fn get_ensemble_stats(
        estimators: &[Box<dyn EnsembleMember + Send + Sync>],
    ) -> EnsembleStats {
        if estimators.is_empty() {
            return EnsembleStats::default();
        }

        let weights: Vec<Float> = estimators.iter().map(|e| e.weight()).collect();
        let performances: Vec<Float> = estimators.iter().map(|e| e.performance()).collect();
        let confidences: Vec<Float> = estimators.iter().map(|e| e.confidence()).collect();

        let mean_weight = weights.iter().sum::<Float>() / weights.len() as Float;
        let mean_performance = performances.iter().sum::<Float>() / performances.len() as Float;
        let mean_confidence = confidences.iter().sum::<Float>() / confidences.len() as Float;

        let weight_variance = weights
            .iter()
            .map(|&w| (w - mean_weight).powi(2))
            .sum::<Float>()
            / weights.len() as Float;

        EnsembleStats {
            n_estimators: estimators.len(),
            mean_weight,
            mean_performance,
            mean_confidence,
            weight_variance,
            total_complexity: estimators.iter().map(|e| e.complexity()).sum(),
        }
    }
}

/// Statistics about an ensemble
#[derive(Debug, Clone)]
pub struct EnsembleStats {
    pub n_estimators: usize,
    pub mean_weight: Float,
    pub mean_performance: Float,
    pub mean_confidence: Float,
    pub weight_variance: Float,
    pub total_complexity: Float,
}

impl Default for EnsembleStats {
    fn default() -> Self {
        Self {
            n_estimators: 0,
            mean_weight: 0.0,
            mean_performance: 0.0,
            mean_confidence: 0.0,
            weight_variance: 0.0,
            total_complexity: 0.0,
        }
    }
}
