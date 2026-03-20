//! Fluent API for baseline estimator configuration
//!
//! This module provides a comprehensive fluent API for configuring dummy estimators
//! with method chaining, configuration presets, and streamlined parameter setting.

use crate::dummy_classifier::{DummyClassifier, Strategy as ClassifierStrategy};
use crate::dummy_regressor::{DummyRegressor, Strategy as RegressorStrategy};
use scirs2_core::ndarray::Array1;
use sklears_core::types::Float;

/// Configuration presets for common use cases
#[derive(Debug, Clone)]
pub struct ConfigPresets;

impl ConfigPresets {
    /// Configuration for highly imbalanced datasets
    pub fn imbalanced_classification() -> ClassifierConfig {
        ClassifierConfig::new()
            .strategy(ClassifierStrategy::MostFrequent)
            .with_description("Optimized for imbalanced datasets")
    }

    /// Configuration for balanced multiclass classification
    pub fn balanced_multiclass() -> ClassifierConfig {
        ClassifierConfig::new()
            .strategy(ClassifierStrategy::Stratified)
            .with_description("Balanced multiclass classification")
    }

    /// Configuration for uncertainty-aware classification
    pub fn uncertainty_aware_classification() -> ClassifierConfig {
        ClassifierConfig::new()
            .strategy(ClassifierStrategy::Bayesian)
            .with_description("Provides uncertainty estimates")
    }

    /// Configuration for time series forecasting baselines
    pub fn time_series_forecasting() -> RegressorConfig {
        RegressorConfig::new()
            .strategy(RegressorStrategy::SeasonalNaive(12))
            .with_description("Time series forecasting baseline")
    }

    /// Configuration for high-variance regression data
    pub fn high_variance_regression() -> RegressorConfig {
        RegressorConfig::new()
            .strategy(RegressorStrategy::Median)
            .with_description("Robust to high variance and outliers")
    }

    /// Configuration for probabilistic regression
    pub fn probabilistic_regression() -> RegressorConfig {
        RegressorConfig::new()
            .strategy(RegressorStrategy::Normal {
                mean: None,
                std: None,
            })
            .with_description("Provides probabilistic predictions")
    }

    /// Configuration for competition-grade baselines
    pub fn competition_baseline() -> RegressorConfig {
        RegressorConfig::new()
            .strategy(RegressorStrategy::Auto)
            .with_description("Adaptive baseline for competitions")
    }

    /// Configuration for streaming/online learning
    pub fn streaming_baseline() -> RegressorConfig {
        RegressorConfig::new()
            .strategy(RegressorStrategy::Mean)
            .with_description("Suitable for streaming scenarios")
    }
}

/// Fluent configuration builder for DummyClassifier
#[derive(Debug, Clone)]
pub struct ClassifierConfig {
    strategy: ClassifierStrategy,
    random_state: Option<u64>,
    constant: Option<i32>,
    bayesian_alpha: Option<Array1<Float>>,
    description: Option<String>,
}

impl ClassifierConfig {
    /// Create a new configuration builder
    pub fn new() -> Self {
        Self {
            strategy: ClassifierStrategy::Auto,
            random_state: None,
            constant: None,
            bayesian_alpha: None,
            description: None,
        }
    }

    /// Set the prediction strategy
    pub fn strategy(mut self, strategy: ClassifierStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set random state for reproducible results
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set constant value for constant strategy
    pub fn constant(mut self, value: i32) -> Self {
        self.constant = Some(value);
        self
    }

    /// Set Bayesian prior parameters
    pub fn bayesian_prior(mut self, alpha: Array1<Float>) -> Self {
        self.bayesian_alpha = Some(alpha);
        self
    }

    /// Add description for documentation
    pub fn with_description<S: Into<String>>(mut self, description: S) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Enable reproducible mode with fixed seed
    pub fn reproducible(self) -> Self {
        self.random_state(42)
    }

    /// Configure for fast predictions (minimal computation)
    pub fn fast_mode(self) -> Self {
        self.strategy(ClassifierStrategy::MostFrequent)
    }

    /// Configure for balanced predictions
    pub fn balanced_mode(self) -> Self {
        self.strategy(ClassifierStrategy::Stratified)
    }

    /// Configure for uncertainty quantification
    pub fn uncertainty_mode(self) -> Self {
        self.strategy(ClassifierStrategy::Bayesian)
    }

    /// Build the configured DummyClassifier
    pub fn build(self) -> DummyClassifier {
        let mut classifier = DummyClassifier::new(self.strategy);

        if let Some(seed) = self.random_state {
            classifier = classifier.with_random_state(seed);
        }

        if let Some(constant) = self.constant {
            classifier = classifier.with_constant(constant);
        }

        if let Some(alpha) = self.bayesian_alpha {
            classifier = classifier.with_bayesian_prior(alpha);
        }

        classifier
    }

    /// Get the configuration description
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Fluent configuration builder for DummyRegressor
#[derive(Debug, Clone)]
pub struct RegressorConfig {
    strategy: RegressorStrategy,
    random_state: Option<u64>,
    constant: Option<Float>,
    description: Option<String>,
}

impl RegressorConfig {
    /// Create a new configuration builder
    pub fn new() -> Self {
        Self {
            strategy: RegressorStrategy::Auto,
            random_state: None,
            constant: None,
            description: None,
        }
    }

    /// Set the prediction strategy
    pub fn strategy(mut self, strategy: RegressorStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set random state for reproducible results
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set constant value for constant strategy
    pub fn constant(mut self, value: Float) -> Self {
        self.constant = Some(value);
        self
    }

    /// Add description for documentation
    pub fn with_description<S: Into<String>>(mut self, description: S) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Enable reproducible mode with fixed seed
    pub fn reproducible(self) -> Self {
        self.random_state(42)
    }

    /// Configure for fast predictions (minimal computation)
    pub fn fast_mode(self) -> Self {
        self.strategy(RegressorStrategy::Mean)
    }

    /// Configure for robust predictions (outlier resistant)
    pub fn robust_mode(self) -> Self {
        self.strategy(RegressorStrategy::Median)
    }

    /// Configure for probabilistic predictions
    pub fn probabilistic_mode(self) -> Self {
        self.strategy(RegressorStrategy::Normal {
            mean: None,
            std: None,
        })
    }

    /// Configure for time series forecasting
    pub fn time_series_mode(self) -> Self {
        self.strategy(RegressorStrategy::SeasonalNaive(12))
    }

    /// Build the configured DummyRegressor
    pub fn build(self) -> DummyRegressor {
        let mut regressor = DummyRegressor::new(self.strategy);

        if let Some(seed) = self.random_state {
            regressor = regressor.with_random_state(seed);
        }

        if let Some(constant) = self.constant {
            regressor = regressor.with_constant(constant);
        }

        regressor
    }

    /// Get the configuration description
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }
}

impl Default for RegressorConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for method chaining with preprocessing
pub trait PreprocessingChain<T> {
    /// Apply preprocessing step and return self for chaining
    fn with_preprocessing<F>(self, preprocessor: F) -> Self
    where
        F: Fn(T) -> T;
}

/// Enhanced fluent API extensions for DummyClassifier
pub trait ClassifierFluentExt {
    /// Create a fluent configuration builder
    fn configure() -> ClassifierConfig;

    /// Quick setup for common scenarios
    fn for_imbalanced_data() -> DummyClassifier;
    fn for_balanced_multiclass() -> DummyClassifier;
    fn for_uncertainty_estimation() -> DummyClassifier;
    fn for_fast_baseline() -> DummyClassifier;
}

impl ClassifierFluentExt for DummyClassifier {
    fn configure() -> ClassifierConfig {
        ClassifierConfig::new()
    }

    fn for_imbalanced_data() -> DummyClassifier {
        ConfigPresets::imbalanced_classification().build()
    }

    fn for_balanced_multiclass() -> DummyClassifier {
        ConfigPresets::balanced_multiclass().build()
    }

    fn for_uncertainty_estimation() -> DummyClassifier {
        ConfigPresets::uncertainty_aware_classification().build()
    }

    fn for_fast_baseline() -> DummyClassifier {
        ClassifierConfig::new().fast_mode().build()
    }
}

/// Enhanced fluent API extensions for DummyRegressor
pub trait RegressorFluentExt {
    /// Create a fluent configuration builder
    fn configure() -> RegressorConfig;

    /// Quick setup for common scenarios
    fn for_time_series() -> DummyRegressor;
    fn for_high_variance() -> DummyRegressor;
    fn for_probabilistic() -> DummyRegressor;
    fn for_competition() -> DummyRegressor;
    fn for_streaming() -> DummyRegressor;
}

impl RegressorFluentExt for DummyRegressor {
    fn configure() -> RegressorConfig {
        RegressorConfig::new()
    }

    fn for_time_series() -> DummyRegressor {
        ConfigPresets::time_series_forecasting().build()
    }

    fn for_high_variance() -> DummyRegressor {
        ConfigPresets::high_variance_regression().build()
    }

    fn for_probabilistic() -> DummyRegressor {
        ConfigPresets::probabilistic_regression().build()
    }

    fn for_competition() -> DummyRegressor {
        ConfigPresets::competition_baseline().build()
    }

    fn for_streaming() -> DummyRegressor {
        ConfigPresets::streaming_baseline().build()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr1;

    #[test]
    fn test_classifier_config_builder() {
        let config = ClassifierConfig::new()
            .strategy(ClassifierStrategy::MostFrequent)
            .random_state(42)
            .with_description("Test configuration");

        assert_eq!(config.description(), Some("Test configuration"));
        let classifier = config.build();
        assert_eq!(classifier.strategy, ClassifierStrategy::MostFrequent);
        assert_eq!(classifier.random_state, Some(42));
    }

    #[test]
    fn test_regressor_config_builder() {
        let config = RegressorConfig::new()
            .strategy(RegressorStrategy::Mean)
            .random_state(123)
            .constant(5.0)
            .with_description("Test regressor");

        assert_eq!(config.description(), Some("Test regressor"));
        let regressor = config.build();
        assert_eq!(regressor.strategy, RegressorStrategy::Constant(5.0));
        assert_eq!(regressor.random_state, Some(123));
    }

    #[test]
    fn test_fluent_extensions() {
        let classifier = DummyClassifier::for_imbalanced_data();
        assert_eq!(classifier.strategy, ClassifierStrategy::MostFrequent);

        let regressor = DummyRegressor::for_time_series();
        assert!(matches!(
            regressor.strategy,
            RegressorStrategy::SeasonalNaive(_)
        ));
    }

    #[test]
    fn test_config_presets() {
        let config = ConfigPresets::imbalanced_classification();
        assert_eq!(
            config.description(),
            Some("Optimized for imbalanced datasets")
        );

        let config = ConfigPresets::probabilistic_regression();
        assert_eq!(
            config.description(),
            Some("Provides probabilistic predictions")
        );
    }

    #[test]
    fn test_method_chaining() {
        let classifier = ClassifierConfig::new()
            .strategy(ClassifierStrategy::Bayesian)
            .reproducible()
            .bayesian_prior(arr1(&[1.0, 1.0, 1.0]))
            .with_description("Chained configuration")
            .build();

        assert_eq!(classifier.strategy, ClassifierStrategy::Bayesian);
        assert_eq!(classifier.random_state, Some(42));
        assert!(classifier.bayesian_alpha_.is_some());
    }

    #[test]
    fn test_mode_configurations() {
        let fast_config = ClassifierConfig::new().fast_mode();
        assert_eq!(fast_config.strategy, ClassifierStrategy::MostFrequent);

        let balanced_config = ClassifierConfig::new().balanced_mode();
        assert_eq!(balanced_config.strategy, ClassifierStrategy::Stratified);

        let uncertainty_config = ClassifierConfig::new().uncertainty_mode();
        assert_eq!(uncertainty_config.strategy, ClassifierStrategy::Bayesian);
    }
}
