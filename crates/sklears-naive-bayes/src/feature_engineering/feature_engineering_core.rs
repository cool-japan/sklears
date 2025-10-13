//! Core feature engineering types and base structures
//!
//! This module provides the foundational types and structures for feature engineering
//! operations, including processors, validators, analyzers, and automation engines.
//! All implementations follow SciRS2 Policy guidelines.

// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
// SciRS2 Policy Compliance - Use scirs2-core for random functionality
// SciRS2 Policy Compliance - Use scirs2-core for random functionality
use serde::{Deserialize, Serialize};
use sklears_core::error::Result;
use sklears_core::prelude::SklearsError;
use std::collections::HashMap;

/// Configuration for feature engineering operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureEngineeringConfig {
    /// Whether to enable automatic feature transformation
    pub auto_transform: bool,
    /// Number of features to select automatically (if 0, select all)
    pub n_features_to_select: usize,
    /// Threshold for feature importance filtering
    pub importance_threshold: f64,
    /// Whether to enable numerical stability checks
    pub numerical_stability: bool,
    /// Random state for reproducible operations
    pub random_state: Option<u64>,
}

impl Default for FeatureEngineeringConfig {
    fn default() -> Self {
        Self {
            auto_transform: true,
            n_features_to_select: 0,
            importance_threshold: 0.01,
            numerical_stability: true,
            random_state: Some(42),
        }
    }
}

/// Core feature engineering processor for data transformations
#[derive(Debug, Clone)]
pub struct FeatureEngineeringProcessor<T> {
    config: FeatureEngineeringConfig,
    feature_names: Option<Vec<String>>,
    feature_importances: Option<Array1<T>>,
    selected_features: Option<Vec<usize>>,
    transformation_stats: HashMap<String, T>,
}

impl<T> FeatureEngineeringProcessor<T>
where
    T: Clone + std::fmt::Debug + serde::Serialize + serde::de::DeserializeOwned,
{
    /// Create a new feature engineering processor
    pub fn new(config: FeatureEngineeringConfig) -> Self {
        Self {
            config,
            feature_names: None,
            feature_importances: None,
            selected_features: None,
            transformation_stats: HashMap::new(),
        }
    }

    /// Get the current configuration
    pub fn config(&self) -> &FeatureEngineeringConfig {
        &self.config
    }

    /// Set feature names for interpretability
    pub fn set_feature_names(&mut self, names: Vec<String>) {
        self.feature_names = Some(names);
    }

    /// Get feature names
    pub fn feature_names(&self) -> Option<&[String]> {
        self.feature_names.as_deref()
    }

    /// Set feature importances
    pub fn set_feature_importances(&mut self, importances: Array1<T>) {
        self.feature_importances = Some(importances);
    }

    /// Get feature importances
    pub fn feature_importances(&self) -> Option<&Array1<T>> {
        self.feature_importances.as_ref()
    }

    /// Set selected features
    pub fn set_selected_features(&mut self, features: Vec<usize>) {
        self.selected_features = Some(features);
    }

    /// Get selected features
    pub fn selected_features(&self) -> Option<&[usize]> {
        self.selected_features.as_deref()
    }

    /// Add transformation statistic
    pub fn add_transformation_stat(&mut self, key: String, value: T) {
        self.transformation_stats.insert(key, value);
    }

    /// Get transformation statistics
    pub fn transformation_stats(&self) -> &HashMap<String, T> {
        &self.transformation_stats
    }
}

/// Validator for feature engineering operations
#[derive(Debug, Clone)]
pub struct FeatureEngineeringValidator {
    min_samples: usize,
    min_features: usize,
    max_features: Option<usize>,
    check_finite: bool,
}

impl Default for FeatureEngineeringValidator {
    fn default() -> Self {
        Self {
            min_samples: 1,
            min_features: 1,
            max_features: None,
            check_finite: true,
        }
    }
}

impl FeatureEngineeringValidator {
    /// Create a new validator with custom parameters
    pub fn new(min_samples: usize, min_features: usize, max_features: Option<usize>) -> Self {
        Self {
            min_samples,
            min_features,
            max_features,
            check_finite: true,
        }
    }

    /// Validate input data dimensions and properties
    pub fn validate_data<T>(&self, data: &ArrayView2<T>) -> Result<()> {
        let (n_samples, n_features) = data.dim();

        if n_samples < self.min_samples {
            return Err(SklearsError::InvalidInput(format!(
                "Number of samples ({}) is less than minimum required ({})",
                n_samples, self.min_samples
            )));
        }

        if n_features < self.min_features {
            return Err(SklearsError::InvalidInput(format!(
                "Number of features ({}) is less than minimum required ({})",
                n_features, self.min_features
            )));
        }

        if let Some(max_features) = self.max_features {
            if n_features > max_features {
                return Err(SklearsError::InvalidInput(format!(
                    "Number of features ({}) exceeds maximum allowed ({})",
                    n_features, max_features
                )));
            }
        }

        Ok(())
    }

    /// Validate feature indices
    pub fn validate_feature_indices(&self, indices: &[usize], n_features: usize) -> Result<()> {
        for &idx in indices {
            if idx >= n_features {
                return Err(SklearsError::InvalidInput(format!(
                    "Feature index {} is out of bounds for {} features",
                    idx, n_features
                )));
            }
        }
        Ok(())
    }
}

/// Estimator for feature engineering parameters
pub trait FeatureEngineeringEstimator<T> {
    /// Fit the estimator to data
    fn fit(&mut self, x: &ArrayView2<T>) -> Result<()>;

    /// Transform data using the fitted estimator
    fn transform(&self, x: &ArrayView2<T>) -> Result<Array2<T>>;

    /// Fit and transform in one step
    fn fit_transform(&mut self, x: &ArrayView2<T>) -> Result<Array2<T>> {
        self.fit(x)?;
        self.transform(x)
    }
}

/// Transformer for feature engineering operations
pub trait FeatureEngineeringTransformer<T> {
    /// Transform features
    fn transform_features(&self, x: &ArrayView2<T>) -> Result<Array2<T>>;

    /// Inverse transform features (if applicable)
    fn inverse_transform(&self, x: &ArrayView2<T>) -> Result<Array2<T>> {
        Err(SklearsError::NotImplemented(
            "Inverse transform not implemented".to_string(),
        ))
    }

    /// Get the number of output features
    fn n_features_out(&self) -> Option<usize> {
        None
    }
}

/// Analyzer for feature engineering results
#[derive(Debug, Clone)]
pub struct FeatureEngineeringAnalyzer<T> {
    analysis_results: HashMap<String, T>,
    feature_rankings: Option<Vec<usize>>,
    quality_metrics: HashMap<String, T>,
}

impl<T> FeatureEngineeringAnalyzer<T>
where
    T: Clone + std::fmt::Debug,
{
    /// Create a new analyzer
    pub fn new() -> Self {
        Self {
            analysis_results: HashMap::new(),
            feature_rankings: None,
            quality_metrics: HashMap::new(),
        }
    }

    /// Add analysis result
    pub fn add_result(&mut self, key: String, value: T) {
        self.analysis_results.insert(key, value);
    }

    /// Get analysis results
    pub fn results(&self) -> &HashMap<String, T> {
        &self.analysis_results
    }

    /// Set feature rankings
    pub fn set_feature_rankings(&mut self, rankings: Vec<usize>) {
        self.feature_rankings = Some(rankings);
    }

    /// Get feature rankings
    pub fn feature_rankings(&self) -> Option<&[usize]> {
        self.feature_rankings.as_deref()
    }

    /// Add quality metric
    pub fn add_quality_metric(&mut self, key: String, value: T) {
        self.quality_metrics.insert(key, value);
    }

    /// Get quality metrics
    pub fn quality_metrics(&self) -> &HashMap<String, T> {
        &self.quality_metrics
    }
}

impl<T> Default for FeatureEngineeringAnalyzer<T>
where
    T: Clone + std::fmt::Debug,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Automation engine for feature engineering workflows
#[derive(Debug, Clone)]
pub struct AutomationEngine<T> {
    config: FeatureEngineeringConfig,
    pipeline_steps: Vec<String>,
    performance_metrics: HashMap<String, T>,
    best_configuration: Option<FeatureEngineeringConfig>,
}

impl<T> AutomationEngine<T>
where
    T: Clone + std::fmt::Debug,
{
    /// Create a new automation engine
    pub fn new(config: FeatureEngineeringConfig) -> Self {
        Self {
            config,
            pipeline_steps: Vec::new(),
            performance_metrics: HashMap::new(),
            best_configuration: None,
        }
    }

    /// Add a pipeline step
    pub fn add_step(&mut self, step: String) {
        self.pipeline_steps.push(step);
    }

    /// Get pipeline steps
    pub fn pipeline_steps(&self) -> &[String] {
        &self.pipeline_steps
    }

    /// Add performance metric
    pub fn add_performance_metric(&mut self, key: String, value: T) {
        self.performance_metrics.insert(key, value);
    }

    /// Get performance metrics
    pub fn performance_metrics(&self) -> &HashMap<String, T> {
        &self.performance_metrics
    }

    /// Set best configuration found
    pub fn set_best_configuration(&mut self, config: FeatureEngineeringConfig) {
        self.best_configuration = Some(config);
    }

    /// Get best configuration
    pub fn best_configuration(&self) -> Option<&FeatureEngineeringConfig> {
        self.best_configuration.as_ref()
    }
}

/// Preprocessing engine for data preparation
#[derive(Debug, Clone)]
pub struct PreprocessingEngine<T> {
    operations: Vec<String>,
    preprocessing_stats: HashMap<String, T>,
    is_fitted: bool,
}

impl<T> PreprocessingEngine<T>
where
    T: Clone + std::fmt::Debug,
{
    /// Create a new preprocessing engine
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
            preprocessing_stats: HashMap::new(),
            is_fitted: false,
        }
    }

    /// Add preprocessing operation
    pub fn add_operation(&mut self, operation: String) {
        self.operations.push(operation);
    }

    /// Get operations
    pub fn operations(&self) -> &[String] {
        &self.operations
    }

    /// Add preprocessing statistic
    pub fn add_stat(&mut self, key: String, value: T) {
        self.preprocessing_stats.insert(key, value);
    }

    /// Get preprocessing statistics
    pub fn stats(&self) -> &HashMap<String, T> {
        &self.preprocessing_stats
    }

    /// Mark as fitted
    pub fn set_fitted(&mut self, fitted: bool) {
        self.is_fitted = fitted;
    }

    /// Check if fitted
    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }
}

impl<T> Default for PreprocessingEngine<T>
where
    T: Clone + std::fmt::Debug,
{
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_engineering_config_default() {
        let config = FeatureEngineeringConfig::default();
        assert!(config.auto_transform);
        assert_eq!(config.n_features_to_select, 0);
        assert_eq!(config.importance_threshold, 0.01);
        assert!(config.numerical_stability);
        assert_eq!(config.random_state, Some(42));
    }

    #[test]
    fn test_feature_engineering_processor() {
        let config = FeatureEngineeringConfig::default();
        let mut processor = FeatureEngineeringProcessor::<f64>::new(config);

        processor.set_feature_names(vec!["feature1".to_string(), "feature2".to_string()]);
        assert_eq!(processor.feature_names().unwrap().len(), 2);

        processor.add_transformation_stat("mean".to_string(), 1.5);
        assert_eq!(processor.transformation_stats().get("mean"), Some(&1.5));
    }

    #[test]
    fn test_feature_engineering_validator() {
        let validator = FeatureEngineeringValidator::default();

        // Test valid data
        let data = Array2::<f64>::zeros((10, 5));
        assert!(validator.validate_data(&data.view()).is_ok());

        // Test feature indices validation
        let indices = vec![0, 1, 2];
        assert!(validator.validate_feature_indices(&indices, 5).is_ok());

        let invalid_indices = vec![0, 1, 5];
        assert!(validator
            .validate_feature_indices(&invalid_indices, 5)
            .is_err());
    }

    #[test]
    fn test_feature_engineering_analyzer() {
        let mut analyzer = FeatureEngineeringAnalyzer::<f64>::new();

        analyzer.add_result("accuracy".to_string(), 0.95);
        assert_eq!(analyzer.results().get("accuracy"), Some(&0.95));

        analyzer.set_feature_rankings(vec![2, 0, 1]);
        assert_eq!(analyzer.feature_rankings().unwrap(), &[2, 0, 1]);

        analyzer.add_quality_metric("completeness".to_string(), 0.98);
        assert_eq!(analyzer.quality_metrics().get("completeness"), Some(&0.98));
    }

    #[test]
    fn test_automation_engine() {
        let config = FeatureEngineeringConfig::default();
        let mut engine = AutomationEngine::<f64>::new(config.clone());

        engine.add_step("normalization".to_string());
        engine.add_step("feature_selection".to_string());
        assert_eq!(engine.pipeline_steps().len(), 2);

        engine.add_performance_metric("processing_time".to_string(), 0.5);
        assert_eq!(
            engine.performance_metrics().get("processing_time"),
            Some(&0.5)
        );

        engine.set_best_configuration(config.clone());
        assert!(engine.best_configuration().is_some());
    }

    #[test]
    fn test_preprocessing_engine() {
        let mut engine = PreprocessingEngine::<f64>::new();

        engine.add_operation("scaling".to_string());
        engine.add_operation("imputation".to_string());
        assert_eq!(engine.operations().len(), 2);

        engine.add_stat("missing_ratio".to_string(), 0.1);
        assert_eq!(engine.stats().get("missing_ratio"), Some(&0.1));

        assert!(!engine.is_fitted());
        engine.set_fitted(true);
        assert!(engine.is_fitted());
    }
}
