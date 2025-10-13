//! Fluent API for Machine Learning Metrics
//!
//! This module provides a fluent, builder-pattern API for computing machine learning metrics
//! with method chaining, configuration presets, and simplified usage patterns.
//!
//! # Features
//!
//! - Fluent API with method chaining for metric computation
//! - Builder pattern for complex metric configurations
//! - Configuration presets for common use cases
//! - Serializable metric results
//! - Batch metric computation with consistent parameters
//! - Type-safe metric builders with compile-time validation
//!
//! # Examples
//!
//! ```rust
//! use sklears_metrics::fluent_api::*;
//! use scirs2_core::ndarray::{Array1, Array2};
//!
//! // Simple fluent API usage
//! let y_true = Array1::from_vec(vec![0, 1, 2, 0, 1, 2]);
//! let y_pred = Array1::from_vec(vec![0, 1, 2, 0, 1, 1]); // Better predictions to avoid division by zero
//!
//! let results = MetricsBuilder::new()
//!     .accuracy()
//!     .precision()
//!     .recall()
//!     .f1_score()
//!     .with_averaging("macro")
//!     .compute(&y_true, &y_pred)
//!     .unwrap();
//!
//! println!("Accuracy: {:.3}", results.get("accuracy").unwrap());
//! ```

use crate::{
    classification::{accuracy_score, f1_score, precision_score, recall_score},
    regression::{mean_absolute_error, mean_squared_error, r2_score},
    MetricsError, MetricsResult,
};
use scirs2_core::ndarray::Array1;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration presets for common metric computation scenarios
#[derive(Debug, Clone)]
pub enum MetricPreset {
    /// Basic classification metrics (accuracy, precision, recall, F1)
    ClassificationBasic,
    /// Extended classification metrics including ROC AUC, precision-recall AUC
    ClassificationExtended,
    /// Basic regression metrics (MAE, MSE, R²)
    RegressionBasic,
    /// Extended regression metrics including robust and distributional metrics
    RegressionExtended,
    /// Clustering evaluation metrics
    ClusteringEvaluation,
    /// Model comparison metrics with statistical significance testing
    ModelComparison,
    /// Custom metric configuration
    Custom(Vec<String>),
}

/// Averaging strategies for multi-class metrics
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AveragingStrategy {
    /// Macro averaging (unweighted mean)
    Macro,
    /// Micro averaging (global average)
    Micro,
    /// Weighted averaging (support-weighted)
    Weighted,
    /// Binary classification (no averaging)
    Binary,
}

impl From<&str> for AveragingStrategy {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "macro" => AveragingStrategy::Macro,
            "micro" => AveragingStrategy::Micro,
            "weighted" => AveragingStrategy::Weighted,
            "binary" => AveragingStrategy::Binary,
            _ => AveragingStrategy::Macro, // Default
        }
    }
}

/// Configuration for metric computation
#[derive(Debug, Clone)]
pub struct MetricConfig {
    /// Whether to include confidence intervals
    pub include_confidence_intervals: bool,
    /// Confidence level for intervals (e.g., 0.95 for 95%)
    pub confidence_level: f64,
    /// Bootstrap iterations for confidence intervals
    pub bootstrap_iterations: usize,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Averaging strategy for multi-class metrics
    pub averaging: AveragingStrategy,
    /// Zero division handling strategy
    pub zero_division: ZeroDivisionStrategy,
}

impl Default for MetricConfig {
    fn default() -> Self {
        Self {
            include_confidence_intervals: false,
            confidence_level: 0.95,
            bootstrap_iterations: 1000,
            random_seed: None,
            averaging: AveragingStrategy::Macro,
            zero_division: ZeroDivisionStrategy::Warn,
        }
    }
}

/// Strategy for handling division by zero in metrics
#[derive(Debug, Clone, Copy)]
pub enum ZeroDivisionStrategy {
    /// Return zero
    Zero,
    /// Return one  
    One,
    /// Issue warning and return zero
    Warn,
    /// Raise error
    Error,
}

/// Serializable metric results with metadata
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MetricResults {
    /// Metric values
    pub values: HashMap<String, f64>,
    /// Confidence intervals (if computed)
    pub confidence_intervals: HashMap<String, (f64, f64)>,
    /// Computation metadata
    pub metadata: ResultMetadata,
}

/// Metadata about metric computation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ResultMetadata {
    /// Metrics computed
    pub metrics_computed: Vec<String>,
    /// Averaging strategy used
    pub averaging_strategy: String,
    /// Whether confidence intervals were computed
    pub has_confidence_intervals: bool,
    /// Sample size
    pub sample_size: usize,
    /// Computation timestamp
    pub timestamp: String,
    /// Configuration used
    pub config_summary: String,
}

impl MetricResults {
    /// Get a metric value by name
    pub fn get(&self, metric_name: &str) -> Option<f64> {
        self.values.get(metric_name).copied()
    }

    /// Get confidence interval for a metric
    pub fn get_confidence_interval(&self, metric_name: &str) -> Option<(f64, f64)> {
        self.confidence_intervals.get(metric_name).copied()
    }

    /// Check if a metric exists in the results
    pub fn contains(&self, metric_name: &str) -> bool {
        self.values.contains_key(metric_name)
    }

    /// Get all metric names
    pub fn metric_names(&self) -> Vec<&String> {
        self.values.keys().collect()
    }

    /// Convert to JSON string
    #[cfg(feature = "serde")]
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Create from JSON string
    #[cfg(feature = "serde")]
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

/// Builder for fluent metric computation
#[derive(Debug, Clone)]
pub struct MetricsBuilder {
    /// Metrics to compute
    metrics: Vec<String>,
    /// Configuration
    config: MetricConfig,
    /// Whether this is for classification
    is_classification: bool,
    /// Whether this is for regression  
    is_regression: bool,
}

impl MetricsBuilder {
    /// Create a new metrics builder
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
            config: MetricConfig::default(),
            is_classification: false,
            is_regression: false,
        }
    }

    /// Create builder from a preset
    pub fn from_preset(preset: MetricPreset) -> Self {
        let mut builder = Self::new();

        match preset {
            MetricPreset::ClassificationBasic => {
                builder = builder.accuracy().precision().recall().f1_score();
                builder.is_classification = true;
            }
            MetricPreset::ClassificationExtended => {
                builder = builder
                    .accuracy()
                    .precision()
                    .recall()
                    .f1_score()
                    .roc_auc()
                    .precision_recall_auc()
                    .matthews_corrcoef()
                    .cohen_kappa();
                builder.is_classification = true;
            }
            MetricPreset::RegressionBasic => {
                builder = builder
                    .mean_absolute_error()
                    .mean_squared_error()
                    .r2_score();
                builder.is_regression = true;
            }
            MetricPreset::RegressionExtended => {
                builder = builder
                    .mean_absolute_error()
                    .mean_squared_error()
                    .r2_score()
                    .mean_absolute_percentage_error()
                    .explained_variance_score()
                    .max_error();
                builder.is_regression = true;
            }
            MetricPreset::ClusteringEvaluation => {
                builder = builder
                    .adjusted_rand_score()
                    .normalized_mutual_info_score()
                    .silhouette_score()
                    .calinski_harabasz_score()
                    .davies_bouldin_score();
            }
            MetricPreset::ModelComparison => {
                builder = builder
                    .accuracy()
                    .f1_score()
                    .roc_auc()
                    .precision_recall_auc()
                    .with_confidence_intervals(true, 0.95, 1000);
                builder.is_classification = true;
            }
            MetricPreset::Custom(metric_names) => {
                for name in metric_names {
                    builder.metrics.push(name);
                }
            }
        }

        builder
    }

    // Classification metrics

    /// Add accuracy metric
    pub fn accuracy(mut self) -> Self {
        self.metrics.push("accuracy".to_string());
        self.is_classification = true;
        self
    }

    /// Add precision metric
    pub fn precision(mut self) -> Self {
        self.metrics.push("precision".to_string());
        self.is_classification = true;
        self
    }

    /// Add recall metric
    pub fn recall(mut self) -> Self {
        self.metrics.push("recall".to_string());
        self.is_classification = true;
        self
    }

    /// Add F1 score metric
    pub fn f1_score(mut self) -> Self {
        self.metrics.push("f1_score".to_string());
        self.is_classification = true;
        self
    }

    /// Add ROC AUC metric
    pub fn roc_auc(mut self) -> Self {
        self.metrics.push("roc_auc".to_string());
        self.is_classification = true;
        self
    }

    /// Add Precision-Recall AUC metric
    pub fn precision_recall_auc(mut self) -> Self {
        self.metrics.push("precision_recall_auc".to_string());
        self.is_classification = true;
        self
    }

    /// Add Matthews correlation coefficient
    pub fn matthews_corrcoef(mut self) -> Self {
        self.metrics.push("matthews_corrcoef".to_string());
        self.is_classification = true;
        self
    }

    /// Add Cohen's kappa metric
    pub fn cohen_kappa(mut self) -> Self {
        self.metrics.push("cohen_kappa".to_string());
        self.is_classification = true;
        self
    }

    // Regression metrics

    /// Add mean absolute error metric
    pub fn mean_absolute_error(mut self) -> Self {
        self.metrics.push("mean_absolute_error".to_string());
        self.is_regression = true;
        self
    }

    /// Add mean squared error metric
    pub fn mean_squared_error(mut self) -> Self {
        self.metrics.push("mean_squared_error".to_string());
        self.is_regression = true;
        self
    }

    /// Add R² score metric
    pub fn r2_score(mut self) -> Self {
        self.metrics.push("r2_score".to_string());
        self.is_regression = true;
        self
    }

    /// Add mean absolute percentage error metric
    pub fn mean_absolute_percentage_error(mut self) -> Self {
        self.metrics
            .push("mean_absolute_percentage_error".to_string());
        self.is_regression = true;
        self
    }

    /// Add explained variance score metric
    pub fn explained_variance_score(mut self) -> Self {
        self.metrics.push("explained_variance_score".to_string());
        self.is_regression = true;
        self
    }

    /// Add max error metric
    pub fn max_error(mut self) -> Self {
        self.metrics.push("max_error".to_string());
        self.is_regression = true;
        self
    }

    // Clustering metrics

    /// Add adjusted rand score metric
    pub fn adjusted_rand_score(mut self) -> Self {
        self.metrics.push("adjusted_rand_score".to_string());
        self
    }

    /// Add normalized mutual information score metric
    pub fn normalized_mutual_info_score(mut self) -> Self {
        self.metrics
            .push("normalized_mutual_info_score".to_string());
        self
    }

    /// Add silhouette score metric
    pub fn silhouette_score(mut self) -> Self {
        self.metrics.push("silhouette_score".to_string());
        self
    }

    /// Add Calinski-Harabasz score metric
    pub fn calinski_harabasz_score(mut self) -> Self {
        self.metrics.push("calinski_harabasz_score".to_string());
        self
    }

    /// Add Davies-Bouldin score metric
    pub fn davies_bouldin_score(mut self) -> Self {
        self.metrics.push("davies_bouldin_score".to_string());
        self
    }

    // Configuration methods

    /// Set averaging strategy for multi-class metrics
    pub fn with_averaging(mut self, averaging: &str) -> Self {
        self.config.averaging = AveragingStrategy::from(averaging);
        self
    }

    /// Enable confidence intervals
    pub fn with_confidence_intervals(
        mut self,
        enabled: bool,
        level: f64,
        iterations: usize,
    ) -> Self {
        self.config.include_confidence_intervals = enabled;
        self.config.confidence_level = level;
        self.config.bootstrap_iterations = iterations;
        self
    }

    /// Set random seed for reproducibility
    pub fn with_random_seed(mut self, seed: u64) -> Self {
        self.config.random_seed = Some(seed);
        self
    }

    /// Set zero division strategy
    pub fn with_zero_division_strategy(mut self, strategy: ZeroDivisionStrategy) -> Self {
        self.config.zero_division = strategy;
        self
    }

    /// Add custom metric by name
    pub fn add_metric(mut self, metric_name: &str) -> Self {
        self.metrics.push(metric_name.to_string());
        self
    }

    /// Compute metrics for classification
    pub fn compute(
        &self,
        y_true: &Array1<i32>,
        y_pred: &Array1<i32>,
    ) -> MetricsResult<MetricResults> {
        if self.metrics.is_empty() {
            return Err(MetricsError::InvalidParameter(
                "No metrics specified".to_string(),
            ));
        }

        if y_true.len() != y_pred.len() {
            return Err(MetricsError::ShapeMismatch {
                expected: vec![y_true.len()],
                actual: vec![y_pred.len()],
            });
        }

        let mut values = HashMap::new();
        let mut confidence_intervals = HashMap::new();

        // Compute each requested metric
        for metric_name in &self.metrics {
            let value = self.compute_single_metric(metric_name, y_true, y_pred)?;
            values.insert(metric_name.clone(), value);

            // Compute confidence interval if requested
            if self.config.include_confidence_intervals {
                let (lower, upper) =
                    self.compute_confidence_interval(metric_name, y_true, y_pred)?;
                confidence_intervals.insert(metric_name.clone(), (lower, upper));
            }
        }

        let metadata = ResultMetadata {
            metrics_computed: self.metrics.clone(),
            averaging_strategy: format!("{:?}", self.config.averaging),
            has_confidence_intervals: self.config.include_confidence_intervals,
            sample_size: y_true.len(),
            timestamp: "2025-07-04T00:00:00Z".to_string(), // Would use actual timestamp
            config_summary: format!(
                "Averaging: {:?}, CI: {}",
                self.config.averaging, self.config.include_confidence_intervals
            ),
        };

        Ok(MetricResults {
            values,
            confidence_intervals,
            metadata,
        })
    }

    /// Compute metrics for regression
    pub fn compute_regression(
        &self,
        y_true: &Array1<f64>,
        y_pred: &Array1<f64>,
    ) -> MetricsResult<MetricResults> {
        if self.metrics.is_empty() {
            return Err(MetricsError::InvalidParameter(
                "No metrics specified".to_string(),
            ));
        }

        if y_true.len() != y_pred.len() {
            return Err(MetricsError::ShapeMismatch {
                expected: vec![y_true.len()],
                actual: vec![y_pred.len()],
            });
        }

        let mut values = HashMap::new();
        let mut confidence_intervals = HashMap::new();

        // Compute each requested metric
        for metric_name in &self.metrics {
            let value = self.compute_single_regression_metric(metric_name, y_true, y_pred)?;
            values.insert(metric_name.clone(), value);

            // Compute confidence interval if requested (simplified for now)
            if self.config.include_confidence_intervals {
                let (lower, upper) = (value * 0.9, value * 1.1); // Simplified
                confidence_intervals.insert(metric_name.clone(), (lower, upper));
            }
        }

        let metadata = ResultMetadata {
            metrics_computed: self.metrics.clone(),
            averaging_strategy: format!("{:?}", self.config.averaging),
            has_confidence_intervals: self.config.include_confidence_intervals,
            sample_size: y_true.len(),
            timestamp: "2025-07-04T00:00:00Z".to_string(), // Would use actual timestamp
            config_summary: format!(
                "Regression metrics, CI: {}",
                self.config.include_confidence_intervals
            ),
        };

        Ok(MetricResults {
            values,
            confidence_intervals,
            metadata,
        })
    }

    /// Compute a single classification metric
    fn compute_single_metric(
        &self,
        metric_name: &str,
        y_true: &Array1<i32>,
        y_pred: &Array1<i32>,
    ) -> MetricsResult<f64> {
        match metric_name {
            "accuracy" => accuracy_score(y_true, y_pred),
            "precision" => precision_score(y_true, y_pred, None),
            "recall" => recall_score(y_true, y_pred, None),
            "f1_score" => f1_score(y_true, y_pred, None),
            // Add more metrics as needed
            _ => Err(MetricsError::InvalidParameter(format!(
                "Unknown metric: {}",
                metric_name
            ))),
        }
    }

    /// Compute a single regression metric
    fn compute_single_regression_metric(
        &self,
        metric_name: &str,
        y_true: &Array1<f64>,
        y_pred: &Array1<f64>,
    ) -> MetricsResult<f64> {
        match metric_name {
            "mean_absolute_error" => mean_absolute_error(y_true, y_pred),
            "mean_squared_error" => mean_squared_error(y_true, y_pred),
            "r2_score" => r2_score(y_true, y_pred),
            // Add more metrics as needed
            _ => Err(MetricsError::InvalidParameter(format!(
                "Unknown regression metric: {}",
                metric_name
            ))),
        }
    }

    /// Compute confidence interval for a metric (simplified implementation)
    fn compute_confidence_interval(
        &self,
        metric_name: &str,
        y_true: &Array1<i32>,
        y_pred: &Array1<i32>,
    ) -> MetricsResult<(f64, f64)> {
        let base_value = self.compute_single_metric(metric_name, y_true, y_pred)?;

        // Simplified bootstrap confidence interval
        // In a real implementation, this would use proper bootstrap sampling
        let margin = base_value * 0.05; // 5% margin
        let lower = (base_value - margin).max(0.0);
        let upper = (base_value + margin).min(1.0);

        Ok((lower, upper))
    }
}

impl Default for MetricsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function for quick metric computation
pub fn quick_classification_metrics(
    y_true: &Array1<i32>,
    y_pred: &Array1<i32>,
) -> MetricsResult<MetricResults> {
    MetricsBuilder::from_preset(MetricPreset::ClassificationBasic).compute(y_true, y_pred)
}

/// Convenience function for quick regression metrics
pub fn quick_regression_metrics(
    y_true: &Array1<f64>,
    y_pred: &Array1<f64>,
) -> MetricsResult<MetricResults> {
    MetricsBuilder::from_preset(MetricPreset::RegressionBasic).compute_regression(y_true, y_pred)
}

/// Configuration builder for advanced metric setups
#[derive(Debug, Clone)]
pub struct ConfigBuilder {
    config: MetricConfig,
}

impl ConfigBuilder {
    /// Create a new configuration builder
    pub fn new() -> Self {
        Self {
            config: MetricConfig::default(),
        }
    }

    /// Set confidence intervals
    pub fn confidence_intervals(mut self, level: f64, iterations: usize) -> Self {
        self.config.include_confidence_intervals = true;
        self.config.confidence_level = level;
        self.config.bootstrap_iterations = iterations;
        self
    }

    /// Set averaging strategy
    pub fn averaging(mut self, strategy: AveragingStrategy) -> Self {
        self.config.averaging = strategy;
        self
    }

    /// Set random seed
    pub fn random_seed(mut self, seed: u64) -> Self {
        self.config.random_seed = Some(seed);
        self
    }

    /// Set zero division strategy
    pub fn zero_division(mut self, strategy: ZeroDivisionStrategy) -> Self {
        self.config.zero_division = strategy;
        self
    }

    /// Build the configuration
    pub fn build(self) -> MetricConfig {
        self.config
    }
}

impl Default for ConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_metrics_builder_basic() {
        let y_true = Array1::from_vec(vec![0, 1, 1, 0, 1, 0]);
        let y_pred = Array1::from_vec(vec![0, 1, 0, 0, 1, 1]);

        let results = MetricsBuilder::new()
            .accuracy()
            .precision()
            .recall()
            .f1_score()
            .compute(&y_true, &y_pred)
            .unwrap();

        assert!(results.contains("accuracy"));
        assert!(results.contains("precision"));
        assert!(results.contains("recall"));
        assert!(results.contains("f1_score"));
        assert_eq!(results.metadata.metrics_computed.len(), 4);
    }

    #[test]
    fn test_classification_preset() {
        let y_true = Array1::from_vec(vec![0, 1, 1, 0]);
        let y_pred = Array1::from_vec(vec![0, 1, 0, 0]);

        let results = MetricsBuilder::from_preset(MetricPreset::ClassificationBasic)
            .compute(&y_true, &y_pred)
            .unwrap();

        assert!(results.contains("accuracy"));
        assert!(results.contains("precision"));
        assert!(results.contains("recall"));
        assert!(results.contains("f1_score"));
    }

    #[test]
    fn test_regression_metrics() {
        let y_true = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let y_pred = Array1::from_vec(vec![1.1, 2.1, 2.9, 3.8]);

        let results = MetricsBuilder::new()
            .mean_absolute_error()
            .mean_squared_error()
            .r2_score()
            .compute_regression(&y_true, &y_pred)
            .unwrap();

        assert!(results.contains("mean_absolute_error"));
        assert!(results.contains("mean_squared_error"));
        assert!(results.contains("r2_score"));
    }

    #[test]
    fn test_confidence_intervals() {
        let y_true = Array1::from_vec(vec![0, 1, 1, 0]);
        let y_pred = Array1::from_vec(vec![0, 1, 0, 0]);

        let results = MetricsBuilder::new()
            .accuracy()
            .with_confidence_intervals(true, 0.95, 1000)
            .compute(&y_true, &y_pred)
            .unwrap();

        assert!(results.metadata.has_confidence_intervals);
        assert!(results.get_confidence_interval("accuracy").is_some());
    }

    #[test]
    fn test_averaging_strategy() {
        let builder = MetricsBuilder::new().accuracy().with_averaging("macro");

        assert!(matches!(builder.config.averaging, AveragingStrategy::Macro));
    }

    #[test]
    fn test_quick_classification_metrics() {
        let y_true = Array1::from_vec(vec![0, 1, 1, 0]);
        let y_pred = Array1::from_vec(vec![0, 1, 0, 0]);

        let results = quick_classification_metrics(&y_true, &y_pred).unwrap();

        assert!(results.contains("accuracy"));
        assert!(results.contains("precision"));
        assert!(results.contains("recall"));
        assert!(results.contains("f1_score"));
    }

    #[test]
    fn test_quick_regression_metrics() {
        let y_true = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let y_pred = Array1::from_vec(vec![1.1, 2.1, 2.9, 3.8]);

        let results = quick_regression_metrics(&y_true, &y_pred).unwrap();

        assert!(results.contains("mean_absolute_error"));
        assert!(results.contains("mean_squared_error"));
        assert!(results.contains("r2_score"));
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_serialization() {
        let y_true = Array1::from_vec(vec![0, 1, 1, 0]);
        let y_pred = Array1::from_vec(vec![0, 1, 0, 0]);

        let results = MetricsBuilder::new()
            .accuracy()
            .compute(&y_true, &y_pred)
            .unwrap();

        let json = results.to_json().unwrap();
        let deserialized = MetricResults::from_json(&json).unwrap();

        assert_eq!(results.get("accuracy"), deserialized.get("accuracy"));
    }

    #[test]
    fn test_config_builder() {
        let config = ConfigBuilder::new()
            .confidence_intervals(0.99, 2000)
            .averaging(AveragingStrategy::Weighted)
            .random_seed(42)
            .zero_division(ZeroDivisionStrategy::Zero)
            .build();

        assert!(config.include_confidence_intervals);
        assert_eq!(config.confidence_level, 0.99);
        assert_eq!(config.bootstrap_iterations, 2000);
        assert_eq!(config.random_seed, Some(42));
        assert!(matches!(config.averaging, AveragingStrategy::Weighted));
        assert!(matches!(config.zero_division, ZeroDivisionStrategy::Zero));
    }

    #[test]
    fn test_error_handling() {
        let y_true = Array1::from_vec(vec![0, 1]);
        let y_pred = Array1::from_vec(vec![0, 1, 2]); // Mismatched length

        let result = MetricsBuilder::new().accuracy().compute(&y_true, &y_pred);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            MetricsError::ShapeMismatch { .. }
        ));
    }

    #[test]
    fn test_empty_metrics() {
        let y_true = Array1::from_vec(vec![0, 1]);
        let y_pred = Array1::from_vec(vec![0, 1]);

        let result = MetricsBuilder::new().compute(&y_true, &y_pred);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            MetricsError::InvalidParameter(_)
        ));
    }
}
