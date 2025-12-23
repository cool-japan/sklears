//! Model complexity analysis and overfitting detection
//!
//! This module provides tools for analyzing model complexity and detecting overfitting.
//! It includes various complexity measures, overfitting detection strategies, and
//! methods for optimal model selection based on complexity-performance trade-offs.

use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict},
};
use std::collections::HashMap;
use std::fmt::{self, Display, Formatter};

/// Result of model complexity analysis
#[derive(Debug, Clone)]
pub struct ComplexityAnalysisResult {
    /// Training error
    pub train_error: f64,
    /// Validation error
    pub validation_error: f64,
    /// Estimated model complexity
    pub complexity_score: f64,
    /// Overfitting indicator (0 = no overfitting, 1 = severe overfitting)
    pub overfitting_score: f64,
    /// Generalization gap (validation_error - train_error)
    pub generalization_gap: f64,
    /// Complexity measures used
    pub complexity_measures: HashMap<String, f64>,
    /// Whether overfitting is detected
    pub overfitting_detected: bool,
    /// Recommended action
    pub recommendation: ComplexityRecommendation,
}

/// Recommendations based on complexity analysis
#[derive(Debug, Clone, PartialEq)]
pub enum ComplexityRecommendation {
    /// Model is appropriate
    Appropriate,
    /// Model is too simple (underfitting)
    IncreaseComplexity,
    /// Model is too complex (overfitting)
    ReduceComplexity,
    /// Use regularization
    UseRegularization,
    /// Collect more data
    CollectMoreData,
    /// Try ensemble methods
    TryEnsembles,
}

impl Display for ComplexityRecommendation {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let msg = match self {
            ComplexityRecommendation::Appropriate => "Model complexity is appropriate",
            ComplexityRecommendation::IncreaseComplexity => "Consider increasing model complexity",
            ComplexityRecommendation::ReduceComplexity => "Consider reducing model complexity",
            ComplexityRecommendation::UseRegularization => "Consider using regularization",
            ComplexityRecommendation::CollectMoreData => "Consider collecting more training data",
            ComplexityRecommendation::TryEnsembles => "Consider using ensemble methods",
        };
        write!(f, "{}", msg)
    }
}

impl Display for ComplexityAnalysisResult {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Model Complexity Analysis:\n\
             Train Error: {:.6}\n\
             Validation Error: {:.6}\n\
             Generalization Gap: {:.6}\n\
             Complexity Score: {:.6}\n\
             Overfitting Score: {:.6}\n\
             Overfitting Detected: {}\n\
             Recommendation: {}",
            self.train_error,
            self.validation_error,
            self.generalization_gap,
            self.complexity_score,
            self.overfitting_score,
            self.overfitting_detected,
            self.recommendation
        )
    }
}

/// Configuration for complexity analysis
#[derive(Debug, Clone)]
pub struct ComplexityAnalysisConfig {
    /// Threshold for overfitting detection (generalization gap)
    pub overfitting_threshold: f64,
    /// Threshold for underfitting detection (high training error)
    pub underfitting_threshold: f64,
    /// Weight for training set size in complexity calculation
    pub data_size_weight: f64,
    /// Whether to include information-theoretic measures
    pub include_information_measures: bool,
    /// Whether to perform cross-validation for robustness
    pub use_cross_validation: bool,
    /// Number of CV folds if using cross-validation
    pub cv_folds: usize,
}

impl Default for ComplexityAnalysisConfig {
    fn default() -> Self {
        Self {
            overfitting_threshold: 0.1,
            underfitting_threshold: 0.3,
            data_size_weight: 0.1,
            include_information_measures: true,
            use_cross_validation: false,
            cv_folds: 5,
        }
    }
}

/// Complexity measures for different types of models
#[derive(Debug, Clone)]
pub enum ComplexityMeasure {
    /// Number of parameters
    ParameterCount,
    /// Effective degrees of freedom
    DegreesOfFreedom,
    /// VC dimension estimate
    VCDimension,
    /// Rademacher complexity
    RademacherComplexity,
    /// Path length (for tree models)
    PathLength,
    /// Number of support vectors (for SVM)
    SupportVectorCount,
    /// Spectral complexity (for neural networks)
    SpectralComplexity,
}

/// Model complexity analyzer
pub struct ModelComplexityAnalyzer {
    config: ComplexityAnalysisConfig,
}

impl ModelComplexityAnalyzer {
    /// Create a new complexity analyzer with default configuration
    pub fn new() -> Self {
        Self {
            config: ComplexityAnalysisConfig::default(),
        }
    }

    /// Create a new complexity analyzer with custom configuration
    pub fn with_config(config: ComplexityAnalysisConfig) -> Self {
        Self { config }
    }

    /// Set overfitting threshold
    pub fn overfitting_threshold(mut self, threshold: f64) -> Self {
        self.config.overfitting_threshold = threshold;
        self
    }

    /// Set underfitting threshold
    pub fn underfitting_threshold(mut self, threshold: f64) -> Self {
        self.config.underfitting_threshold = threshold;
        self
    }

    /// Enable or disable cross-validation
    pub fn use_cross_validation(mut self, use_cv: bool) -> Self {
        self.config.use_cross_validation = use_cv;
        self
    }

    /// Set number of CV folds
    pub fn cv_folds(mut self, folds: usize) -> Self {
        self.config.cv_folds = folds;
        self
    }

    /// Analyze model complexity
    pub fn analyze<E, X, Y>(
        &self,
        estimator: &E,
        x_train: &[X],
        y_train: &[Y],
        x_val: &[X],
        y_val: &[Y],
    ) -> Result<ComplexityAnalysisResult>
    where
        E: Estimator + Fit<Vec<X>, Vec<Y>> + Clone,
        E::Fitted: Predict<Vec<X>, Vec<f64>>,
        X: Clone,
        Y: Clone + Into<f64>,
    {
        // Train the model
        let x_train_vec = x_train.to_vec();
        let y_train_vec = y_train.to_vec();
        let trained_model = estimator.clone().fit(&x_train_vec, &y_train_vec)?;

        // Calculate training error
        let train_predictions = trained_model.predict(&x_train_vec)?;
        let train_targets: Vec<f64> = y_train.iter().map(|y| y.clone().into()).collect();
        let train_error = self.calculate_error(&train_predictions, &train_targets);

        // Calculate validation error
        let x_val_vec = x_val.to_vec();
        let val_predictions = trained_model.predict(&x_val_vec)?;
        let val_targets: Vec<f64> = y_val.iter().map(|y| y.clone().into()).collect();
        let validation_error = self.calculate_error(&val_predictions, &val_targets);

        // Calculate generalization gap
        let generalization_gap = validation_error - train_error;

        // Estimate model complexity
        let complexity_measures = self.estimate_complexity(x_train, y_train, &trained_model)?;
        let complexity_score = self.aggregate_complexity(&complexity_measures);

        // Calculate overfitting score
        let overfitting_score = self.calculate_overfitting_score(
            train_error,
            validation_error,
            complexity_score,
            x_train.len(),
        );

        // Detect overfitting
        let overfitting_detected = generalization_gap > self.config.overfitting_threshold;

        // Generate recommendation
        let recommendation = self.generate_recommendation(
            train_error,
            validation_error,
            generalization_gap,
            complexity_score,
            overfitting_detected,
        );

        Ok(ComplexityAnalysisResult {
            train_error,
            validation_error,
            complexity_score,
            overfitting_score,
            generalization_gap,
            complexity_measures,
            overfitting_detected,
            recommendation,
        })
    }

    /// Calculate prediction error (MSE for regression)
    fn calculate_error(&self, predictions: &[f64], targets: &[f64]) -> f64 {
        if predictions.len() != targets.len() {
            return f64::INFINITY;
        }

        let mse = predictions
            .iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| (pred - target).powi(2))
            .sum::<f64>()
            / predictions.len() as f64;

        mse
    }

    /// Estimate model complexity using various measures
    fn estimate_complexity<X, Y>(
        &self,
        x_train: &[X],
        y_train: &[Y],
        _trained_model: &impl Predict<Vec<X>, Vec<f64>>,
    ) -> Result<HashMap<String, f64>>
    where
        X: Clone,
        Y: Clone + Into<f64>,
    {
        let mut measures = HashMap::new();

        // Basic complexity based on training set size
        let n_samples = x_train.len() as f64;
        let n_features = self.estimate_feature_count(x_train);

        measures.insert("training_set_size".to_string(), n_samples);
        measures.insert("feature_count".to_string(), n_features);

        // Parameter count estimation (approximate)
        let param_count = self.estimate_parameter_count(n_features);
        measures.insert("estimated_parameters".to_string(), param_count);

        // Data-dependent complexity measures
        let data_complexity = self.calculate_data_complexity(x_train, y_train);
        measures.insert("data_complexity".to_string(), data_complexity);

        // Effective degrees of freedom estimation
        let eff_dof = self.estimate_effective_dof(n_samples, param_count);
        measures.insert("effective_dof".to_string(), eff_dof);

        Ok(measures)
    }

    /// Estimate number of features (simplified)
    fn estimate_feature_count<X>(&self, _x_train: &[X]) -> f64 {
        // This is a simplified estimation - in practice, this would depend on the actual data type
        // For now, we'll use a default estimate
        10.0 // Placeholder
    }

    /// Estimate parameter count based on features
    fn estimate_parameter_count(&self, n_features: f64) -> f64 {
        // Simple linear model assumption: one parameter per feature plus intercept
        n_features + 1.0
    }

    /// Calculate data complexity (variance in targets)
    fn calculate_data_complexity<X, Y>(&self, _x_train: &[X], y_train: &[Y]) -> f64
    where
        Y: Clone + Into<f64>,
    {
        let targets: Vec<f64> = y_train.iter().map(|y| y.clone().into()).collect();
        if targets.is_empty() {
            return 0.0;
        }

        let mean = targets.iter().sum::<f64>() / targets.len() as f64;
        let variance =
            targets.iter().map(|&y| (y - mean).powi(2)).sum::<f64>() / targets.len() as f64;

        variance.sqrt()
    }

    /// Estimate effective degrees of freedom
    fn estimate_effective_dof(&self, n_samples: f64, param_count: f64) -> f64 {
        // Simple heuristic: effective DOF is limited by sample size
        param_count.min(n_samples * 0.1)
    }

    /// Aggregate complexity measures into a single score
    fn aggregate_complexity(&self, measures: &HashMap<String, f64>) -> f64 {
        let mut score = 0.0;
        let mut weight_sum = 0.0;

        // Weight different complexity measures
        if let Some(&param_count) = measures.get("estimated_parameters") {
            score += param_count * 0.4;
            weight_sum += 0.4;
        }

        if let Some(&eff_dof) = measures.get("effective_dof") {
            score += eff_dof * 0.3;
            weight_sum += 0.3;
        }

        if let Some(&data_complexity) = measures.get("data_complexity") {
            score += data_complexity * 0.2;
            weight_sum += 0.2;
        }

        if let Some(&n_samples) = measures.get("training_set_size") {
            // Complexity decreases with more data
            score += (1.0 / (n_samples + 1.0)) * 100.0 * 0.1;
            weight_sum += 0.1;
        }

        if weight_sum > 0.0 {
            score / weight_sum
        } else {
            0.0
        }
    }

    /// Calculate overfitting score
    fn calculate_overfitting_score(
        &self,
        train_error: f64,
        validation_error: f64,
        complexity_score: f64,
        n_samples: usize,
    ) -> f64 {
        // Overfitting score combines generalization gap with complexity
        let generalization_gap = validation_error - train_error;
        let relative_gap = if train_error > 0.0 {
            generalization_gap / train_error
        } else {
            generalization_gap
        };

        // Adjust for sample size (small datasets are more prone to overfitting)
        let size_factor = 1.0 / (n_samples as f64).sqrt();

        // Combine factors
        let overfitting_score = relative_gap * (1.0 + complexity_score * 0.1) * (1.0 + size_factor);

        // Normalize to [0, 1]
        overfitting_score.clamp(0.0, 1.0)
    }

    /// Generate recommendation based on analysis
    fn generate_recommendation(
        &self,
        train_error: f64,
        validation_error: f64,
        generalization_gap: f64,
        complexity_score: f64,
        overfitting_detected: bool,
    ) -> ComplexityRecommendation {
        // High training error suggests underfitting
        if train_error > self.config.underfitting_threshold {
            return ComplexityRecommendation::IncreaseComplexity;
        }

        // Overfitting detected
        if overfitting_detected {
            if complexity_score > 10.0 {
                ComplexityRecommendation::ReduceComplexity
            } else {
                ComplexityRecommendation::UseRegularization
            }
        } else if generalization_gap > 0.05
            && generalization_gap <= self.config.overfitting_threshold
        {
            // Mild overfitting
            if validation_error > train_error * 1.5 {
                ComplexityRecommendation::CollectMoreData
            } else {
                ComplexityRecommendation::UseRegularization
            }
        } else if train_error > 0.1 && validation_error > 0.1 {
            // Both errors are high
            ComplexityRecommendation::TryEnsembles
        } else {
            ComplexityRecommendation::Appropriate
        }
    }
}

impl Default for ModelComplexityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Overfitting detector for time series data
pub struct OverfittingDetector {
    config: ComplexityAnalysisConfig,
}

impl OverfittingDetector {
    /// Create a new overfitting detector
    pub fn new() -> Self {
        Self {
            config: ComplexityAnalysisConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: ComplexityAnalysisConfig) -> Self {
        Self { config }
    }

    /// Detect overfitting using learning curves
    pub fn detect_from_learning_curve(
        &self,
        train_sizes: &[usize],
        train_scores: &[f64],
        val_scores: &[f64],
    ) -> Result<bool> {
        if train_sizes.len() != train_scores.len() || train_scores.len() != val_scores.len() {
            return Err(SklearsError::InvalidParameter {
                name: "arrays".to_string(),
                reason: "array lengths must match".to_string(),
            });
        }

        if train_sizes.is_empty() {
            return Err(SklearsError::InvalidParameter {
                name: "arrays".to_string(),
                reason: "arrays cannot be empty".to_string(),
            });
        }

        // Look for diverging learning curves
        let mut divergence_count = 0;
        for i in 1..train_scores.len() {
            let train_improvement = train_scores[i - 1] - train_scores[i];
            let val_improvement = val_scores[i - 1] - val_scores[i];

            // If training improves but validation doesn't (or gets worse)
            if train_improvement > 0.01 && val_improvement < 0.01 {
                divergence_count += 1;
            }
        }

        // Overfitting if curves diverge in more than half the steps
        Ok(divergence_count > train_scores.len() / 2)
    }

    /// Detect overfitting using validation curves
    pub fn detect_from_validation_curve(
        &self,
        param_values: &[f64],
        train_scores: &[f64],
        val_scores: &[f64],
    ) -> Result<(bool, Option<f64>)> {
        if param_values.len() != train_scores.len() || train_scores.len() != val_scores.len() {
            return Err(SklearsError::InvalidParameter {
                name: "arrays".to_string(),
                reason: "array lengths must match".to_string(),
            });
        }

        if param_values.is_empty() {
            return Err(SklearsError::InvalidParameter {
                name: "arrays".to_string(),
                reason: "arrays cannot be empty".to_string(),
            });
        }

        // Find optimal parameter value (minimum validation error)
        let min_val_idx = val_scores
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        let optimal_param = param_values[min_val_idx];
        let min_val_score = val_scores[min_val_idx];

        // Check if validation score increases for higher complexity
        let mut overfitting_detected = false;
        for i in (min_val_idx + 1)..val_scores.len() {
            if val_scores[i] > min_val_score + self.config.overfitting_threshold {
                overfitting_detected = true;
                break;
            }
        }

        Ok((overfitting_detected, Some(optimal_param)))
    }
}

impl Default for OverfittingDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function for analyzing model complexity
pub fn analyze_model_complexity<E, X, Y>(
    estimator: &E,
    x_train: &[X],
    y_train: &[Y],
    x_val: &[X],
    y_val: &[Y],
) -> Result<ComplexityAnalysisResult>
where
    E: Estimator + Fit<Vec<X>, Vec<Y>> + Clone,
    E::Fitted: Predict<Vec<X>, Vec<f64>>,
    X: Clone,
    Y: Clone + Into<f64>,
{
    let analyzer = ModelComplexityAnalyzer::new();
    analyzer.analyze(estimator, x_train, y_train, x_val, y_val)
}

/// Convenience function for detecting overfitting from learning curves
pub fn detect_overfitting_learning_curve(
    train_sizes: &[usize],
    train_scores: &[f64],
    val_scores: &[f64],
) -> Result<bool> {
    let detector = OverfittingDetector::new();
    detector.detect_from_learning_curve(train_sizes, train_scores, val_scores)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    // Mock estimator for testing
    #[derive(Clone)]
    struct MockEstimator;

    struct MockTrained;

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
            Ok(MockTrained)
        }
    }

    impl Predict<Vec<f64>, Vec<f64>> for MockTrained {
        fn predict(&self, x: &Vec<f64>) -> Result<Vec<f64>> {
            // Simple linear prediction with some noise to simulate overfitting
            Ok(x.iter().map(|&xi| xi * 0.5 + 0.1).collect())
        }
    }

    #[test]
    fn test_complexity_analyzer_creation() {
        let analyzer = ModelComplexityAnalyzer::new();
        assert_eq!(analyzer.config.overfitting_threshold, 0.1);
        assert_eq!(analyzer.config.underfitting_threshold, 0.3);
    }

    #[test]
    fn test_complexity_analysis() {
        let estimator = MockEstimator;
        let x_train: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let y_train: Vec<f64> = x_train.iter().map(|&x| x * 0.5).collect();
        let x_val: Vec<f64> = (0..20).map(|i| i as f64 * 0.1 + 10.0).collect();
        let y_val: Vec<f64> = x_val.iter().map(|&x| x * 0.5).collect();

        let analyzer = ModelComplexityAnalyzer::new();
        let result = analyzer.analyze(&estimator, &x_train, &y_train, &x_val, &y_val);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.train_error >= 0.0);
        assert!(result.validation_error >= 0.0);
        assert!(result.complexity_score >= 0.0);
        assert!(result.overfitting_score >= 0.0 && result.overfitting_score <= 1.0);
    }

    #[test]
    fn test_overfitting_detector() {
        let detector = OverfittingDetector::new();

        // Test learning curve overfitting detection
        let train_sizes = vec![10, 20, 30, 40, 50];
        let train_scores = vec![0.5, 0.3, 0.2, 0.1, 0.05]; // Improving
        let val_scores = vec![0.6, 0.4, 0.4, 0.45, 0.5]; // Getting worse

        let result = detector.detect_from_learning_curve(&train_sizes, &train_scores, &val_scores);
        assert!(result.is_ok());

        // Test validation curve overfitting detection
        let param_values = vec![0.1, 0.5, 1.0, 2.0, 5.0];
        let train_scores = vec![0.5, 0.3, 0.2, 0.1, 0.05];
        let val_scores = vec![0.6, 0.4, 0.35, 0.4, 0.5];

        let result =
            detector.detect_from_validation_curve(&param_values, &train_scores, &val_scores);
        assert!(result.is_ok());
        let (_overfitting, optimal_param) = result.unwrap();
        assert!(optimal_param.is_some());
    }

    #[test]
    fn test_convenience_functions() {
        let estimator = MockEstimator;
        let x_train: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
        let y_train: Vec<f64> = x_train.iter().map(|&x| x * 0.3).collect();
        let x_val: Vec<f64> = (0..10).map(|i| i as f64 * 0.1 + 5.0).collect();
        let y_val: Vec<f64> = x_val.iter().map(|&x| x * 0.3).collect();

        let result = analyze_model_complexity(&estimator, &x_train, &y_train, &x_val, &y_val);
        assert!(result.is_ok());

        let train_sizes = vec![10, 20, 30];
        let train_scores = vec![0.5, 0.3, 0.2];
        let val_scores = vec![0.6, 0.4, 0.45];

        let result = detect_overfitting_learning_curve(&train_sizes, &train_scores, &val_scores);
        assert!(result.is_ok());
    }

    #[test]
    fn test_complexity_recommendations() {
        use ComplexityRecommendation::*;

        let recommendation = Appropriate;
        assert_eq!(
            format!("{}", recommendation),
            "Model complexity is appropriate"
        );

        let recommendation = IncreaseComplexity;
        assert_eq!(
            format!("{}", recommendation),
            "Consider increasing model complexity"
        );

        let recommendation = ReduceComplexity;
        assert_eq!(
            format!("{}", recommendation),
            "Consider reducing model complexity"
        );
    }
}
