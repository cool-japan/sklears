//! Type Safety and Compile-Time Validation for Metrics
//!
//! This module provides a type-safe framework for machine learning metrics
//! that leverages Rust's type system to catch errors at compile time and
//! provide zero-cost abstractions for metric computation.
//!
//! # Features
//!
//! - Phantom types for metric categories (classification, regression, clustering)
//! - Compile-time validation of metric compatibility
//! - Zero-cost abstractions for metric computation
//! - Type-safe metric composition and chaining
//! - Const generics for fixed-size computations
//! - Trait-based metric definitions with associated types
//!
//! # Examples
//!
//! ```rust
//! use sklears_metrics::type_safety::*;
//! use sklears_metrics::classification::accuracy_score;
//! use sklears_metrics::regression::mean_squared_error;
//! use scirs2_core::ndarray::Array1;
//!
//! // Type-safe classification metrics
//! let y_true = Array1::from_vec(vec![0, 1, 0, 1]);
//! let y_pred = Array1::from_vec(vec![0, 1, 1, 1]);
//!
//! let accuracy: TypedMetric<Classification> =
//!     TypedMetric::new(accuracy_score(&y_true, &y_pred).unwrap());
//!
//! // Type-safe regression metrics
//! let y_true_reg = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
//! let y_pred_reg = Array1::from_vec(vec![1.1, 2.1, 2.9, 3.8]);
//!
//! let mse: TypedMetric<Regression> =
//!     TypedMetric::new(mean_squared_error(&y_true_reg, &y_pred_reg).unwrap());
//!
//! // This would fail at compile time:
//! // let invalid_combo = accuracy + mse; // Error: cannot add Classification and Regression metrics
//! ```

use crate::{MetricsError, MetricsResult};
use scirs2_core::ndarray::Array1;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul};

/// Phantom type for classification metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Classification;

/// Phantom type for regression metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Regression;

/// Phantom type for clustering metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Clustering;

/// Phantom type for ranking metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ranking;

/// Phantom type for information-theoretic metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InformationTheory;

/// Trait for metric categories
pub trait MetricCategory {
    /// Human-readable name of the metric category
    const NAME: &'static str;

    /// Whether higher values are better for this category
    const HIGHER_IS_BETTER: bool;

    /// Typical range of values for this category
    const TYPICAL_RANGE: (f64, f64);
}

impl MetricCategory for Classification {
    const NAME: &'static str = "Classification";
    const HIGHER_IS_BETTER: bool = true;
    const TYPICAL_RANGE: (f64, f64) = (0.0, 1.0);
}

impl MetricCategory for Regression {
    const NAME: &'static str = "Regression";
    const HIGHER_IS_BETTER: bool = false;
    const TYPICAL_RANGE: (f64, f64) = (0.0, f64::INFINITY);
}

impl MetricCategory for Clustering {
    const NAME: &'static str = "Clustering";
    const HIGHER_IS_BETTER: bool = true;
    const TYPICAL_RANGE: (f64, f64) = (-1.0, 1.0);
}

impl MetricCategory for Ranking {
    const NAME: &'static str = "Ranking";
    const HIGHER_IS_BETTER: bool = true;
    const TYPICAL_RANGE: (f64, f64) = (0.0, 1.0);
}

impl MetricCategory for InformationTheory {
    const NAME: &'static str = "InformationTheory";
    const HIGHER_IS_BETTER: bool = false;
    const TYPICAL_RANGE: (f64, f64) = (0.0, f64::INFINITY);
}

/// Type-safe metric wrapper with phantom type parameter
#[derive(Debug, Clone)]
pub struct TypedMetric<T: MetricCategory> {
    /// The metric value
    pub value: f64,
    /// Phantom type parameter for compile-time type safety
    _phantom: PhantomData<T>,
}

impl<T: MetricCategory> TypedMetric<T> {
    /// Create a new typed metric
    pub fn new(value: f64) -> Self {
        Self {
            value,
            _phantom: PhantomData,
        }
    }

    /// Get the metric value
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Get the metric category name
    pub fn category(&self) -> &'static str {
        T::NAME
    }

    /// Check if higher values are better for this metric category
    pub fn higher_is_better(&self) -> bool {
        T::HIGHER_IS_BETTER
    }

    /// Get the typical range for this metric category
    pub fn typical_range(&self) -> (f64, f64) {
        T::TYPICAL_RANGE
    }

    /// Validate that the metric value is within expected bounds
    pub fn validate(&self) -> MetricsResult<()> {
        let (min, max) = T::TYPICAL_RANGE;
        if self.value < min || self.value > max {
            return Err(MetricsError::InvalidParameter(format!(
                "Metric value {} is outside typical range [{}, {}] for category {}",
                self.value,
                min,
                max,
                T::NAME
            )));
        }
        Ok(())
    }

    /// Convert to a different metric category (unsafe)
    pub unsafe fn convert<U: MetricCategory>(self) -> TypedMetric<U> {
        TypedMetric::new(self.value)
    }
}

/// Addition is only allowed between metrics of the same category
impl<T: MetricCategory> Add for TypedMetric<T> {
    type Output = TypedMetric<T>;

    fn add(self, other: TypedMetric<T>) -> Self::Output {
        TypedMetric::new(self.value + other.value)
    }
}

/// Multiplication by scalar
impl<T: MetricCategory> Mul<f64> for TypedMetric<T> {
    type Output = TypedMetric<T>;

    fn mul(self, scalar: f64) -> Self::Output {
        TypedMetric::new(self.value * scalar)
    }
}

/// Division by scalar
impl<T: MetricCategory> Div<f64> for TypedMetric<T> {
    type Output = TypedMetric<T>;

    fn div(self, scalar: f64) -> Self::Output {
        TypedMetric::new(self.value / scalar)
    }
}

/// Trait for metrics that can be computed
pub trait Metric<T: MetricCategory> {
    /// Input type for the metric
    type Input;

    /// Compute the metric value
    fn compute(&self, input: &Self::Input) -> MetricsResult<TypedMetric<T>>;

    /// Get the metric name
    fn name(&self) -> &'static str;

    /// Get metric metadata
    fn metadata(&self) -> MetricMetadata {
        MetricMetadata {
            name: self.name(),
            category: T::NAME,
            higher_is_better: T::HIGHER_IS_BETTER,
            typical_range: T::TYPICAL_RANGE,
        }
    }
}

/// Metadata for a metric
#[derive(Debug, Clone)]
pub struct MetricMetadata {
    pub name: &'static str,
    pub category: &'static str,
    pub higher_is_better: bool,
    pub typical_range: (f64, f64),
}

/// Type-safe metric suite for multiple metrics of the same category
#[derive(Debug, Clone)]
pub struct MetricSuite<T: MetricCategory> {
    metrics: Vec<TypedMetric<T>>,
    names: Vec<String>,
}

impl<T: MetricCategory> MetricSuite<T> {
    /// Create a new metric suite
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
            names: Vec::new(),
        }
    }

    /// Add a metric to the suite
    pub fn add_metric(&mut self, name: String, metric: TypedMetric<T>) {
        self.metrics.push(metric);
        self.names.push(name);
    }

    /// Get the number of metrics
    pub fn len(&self) -> usize {
        self.metrics.len()
    }

    /// Check if the suite is empty
    pub fn is_empty(&self) -> bool {
        self.metrics.is_empty()
    }

    /// Get metric by index
    pub fn get(&self, index: usize) -> Option<&TypedMetric<T>> {
        self.metrics.get(index)
    }

    /// Get metric by name
    pub fn get_by_name(&self, name: &str) -> Option<&TypedMetric<T>> {
        self.names
            .iter()
            .position(|n| n == name)
            .and_then(|i| self.metrics.get(i))
    }

    /// Get all metric names
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Get all metric values
    pub fn values(&self) -> Vec<f64> {
        self.metrics.iter().map(|m| m.value).collect()
    }

    /// Compute summary statistics
    pub fn summary(&self) -> MetricSummary<T> {
        if self.metrics.is_empty() {
            return MetricSummary::empty();
        }

        let values: Vec<f64> = self.values();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        let mut sorted_values = values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min = sorted_values[0];
        let max = sorted_values[sorted_values.len() - 1];
        let median = if sorted_values.len() % 2 == 0 {
            (sorted_values[sorted_values.len() / 2 - 1] + sorted_values[sorted_values.len() / 2])
                / 2.0
        } else {
            sorted_values[sorted_values.len() / 2]
        };

        MetricSummary {
            count: values.len(),
            mean: TypedMetric::new(mean),
            std_dev: TypedMetric::new(std_dev),
            min: TypedMetric::new(min),
            max: TypedMetric::new(max),
            median: TypedMetric::new(median),
        }
    }

    /// Validate all metrics in the suite
    pub fn validate_all(&self) -> MetricsResult<()> {
        for metric in &self.metrics {
            metric.validate()?;
        }
        Ok(())
    }
}

impl<T: MetricCategory> Default for MetricSuite<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary statistics for a metric suite
#[derive(Debug, Clone)]
pub struct MetricSummary<T: MetricCategory> {
    pub count: usize,
    pub mean: TypedMetric<T>,
    pub std_dev: TypedMetric<T>,
    pub min: TypedMetric<T>,
    pub max: TypedMetric<T>,
    pub median: TypedMetric<T>,
}

impl<T: MetricCategory> MetricSummary<T> {
    fn empty() -> Self {
        Self {
            count: 0,
            mean: TypedMetric::new(0.0),
            std_dev: TypedMetric::new(0.0),
            min: TypedMetric::new(0.0),
            max: TypedMetric::new(0.0),
            median: TypedMetric::new(0.0),
        }
    }
}

/// Compile-time metric validation using const generics
pub trait ValidatedMetric<T: MetricCategory, const N: usize> {
    /// Metric bounds for validation
    const BOUNDS: [(f64, f64); N];

    /// Validate metric values at compile time
    fn validate_bounds(values: &[f64; N]) -> bool {
        for (i, &value) in values.iter().enumerate() {
            let (min, max) = Self::BOUNDS[i];
            if value < min || value > max {
                return false;
            }
        }
        true
    }
}

/// Accuracy metric with compile-time validation
pub struct AccuracyMetric;

impl ValidatedMetric<Classification, 1> for AccuracyMetric {
    const BOUNDS: [(f64, f64); 1] = [(0.0, 1.0)];
}

impl Metric<Classification> for AccuracyMetric {
    type Input = (Array1<i32>, Array1<i32>);

    fn compute(&self, input: &Self::Input) -> MetricsResult<TypedMetric<Classification>> {
        let (y_true, y_pred) = input;

        if y_true.len() != y_pred.len() {
            return Err(MetricsError::ShapeMismatch {
                expected: vec![y_true.len()],
                actual: vec![y_pred.len()],
            });
        }

        let correct = y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(a, b)| a == b)
            .count();

        let accuracy = correct as f64 / y_true.len() as f64;
        Ok(TypedMetric::new(accuracy))
    }

    fn name(&self) -> &'static str {
        "accuracy"
    }
}

/// Mean squared error metric with compile-time validation
pub struct MeanSquaredErrorMetric;

impl ValidatedMetric<Regression, 1> for MeanSquaredErrorMetric {
    const BOUNDS: [(f64, f64); 1] = [(0.0, f64::INFINITY)];
}

impl Metric<Regression> for MeanSquaredErrorMetric {
    type Input = (Array1<f64>, Array1<f64>);

    fn compute(&self, input: &Self::Input) -> MetricsResult<TypedMetric<Regression>> {
        let (y_true, y_pred) = input;

        if y_true.len() != y_pred.len() {
            return Err(MetricsError::ShapeMismatch {
                expected: vec![y_true.len()],
                actual: vec![y_pred.len()],
            });
        }

        let mse = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / y_true.len() as f64;

        Ok(TypedMetric::new(mse))
    }

    fn name(&self) -> &'static str {
        "mean_squared_error"
    }
}

/// Trait for composable metrics
pub trait ComposableMetric<T: MetricCategory> {
    /// Compose this metric with another metric
    fn compose<U: MetricCategory>(self, other: impl ComposableMetric<U>) -> CompositeMetric<T, U>;
}

/// Composite metric combining two different metric categories
#[derive(Debug, Clone)]
pub struct CompositeMetric<T: MetricCategory, U: MetricCategory> {
    primary: TypedMetric<T>,
    secondary: TypedMetric<U>,
}

impl<T: MetricCategory, U: MetricCategory> CompositeMetric<T, U> {
    pub fn new(primary: TypedMetric<T>, secondary: TypedMetric<U>) -> Self {
        Self { primary, secondary }
    }

    pub fn primary(&self) -> &TypedMetric<T> {
        &self.primary
    }

    pub fn secondary(&self) -> &TypedMetric<U> {
        &self.secondary
    }

    /// Compute a weighted combination of the metrics
    pub fn weighted_combination(&self, weight_primary: f64, weight_secondary: f64) -> f64 {
        let primary_normalized = if T::HIGHER_IS_BETTER {
            self.primary.value
        } else {
            1.0 / (1.0 + self.primary.value)
        };

        let secondary_normalized = if U::HIGHER_IS_BETTER {
            self.secondary.value
        } else {
            1.0 / (1.0 + self.secondary.value)
        };

        weight_primary * primary_normalized + weight_secondary * secondary_normalized
    }
}

/// Trait for metric transformations
pub trait MetricTransform<T: MetricCategory>: std::fmt::Debug + Send + Sync {
    /// Transform the metric value
    fn transform(&self, metric: &TypedMetric<T>) -> TypedMetric<T>;
}

/// Logarithmic transformation
#[derive(Debug)]
pub struct LogTransform;

impl<T: MetricCategory> MetricTransform<T> for LogTransform {
    fn transform(&self, metric: &TypedMetric<T>) -> TypedMetric<T> {
        TypedMetric::new(metric.value.ln())
    }
}

/// Normalization transformation
#[derive(Debug)]
pub struct NormalizeTransform {
    min: f64,
    max: f64,
}

impl NormalizeTransform {
    pub fn new(min: f64, max: f64) -> Self {
        Self { min, max }
    }
}

impl<T: MetricCategory> MetricTransform<T> for NormalizeTransform {
    fn transform(&self, metric: &TypedMetric<T>) -> TypedMetric<T> {
        let normalized = (metric.value - self.min) / (self.max - self.min);
        TypedMetric::new(normalized)
    }
}

/// Metric builder pattern for type-safe metric construction
#[derive(Debug)]
pub struct MetricBuilder<T: MetricCategory> {
    name: Option<String>,
    transform: Option<Box<dyn MetricTransform<T>>>,
    _phantom: PhantomData<T>,
}

impl<T: MetricCategory> MetricBuilder<T> {
    pub fn new() -> Self {
        Self {
            name: None,
            transform: None,
            _phantom: PhantomData,
        }
    }

    pub fn name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    pub fn transform<F: MetricTransform<T> + 'static>(mut self, transform: F) -> Self {
        self.transform = Some(Box::new(transform));
        self
    }

    pub fn build(self, value: f64) -> TypedMetric<T> {
        let mut metric = TypedMetric::new(value);

        if let Some(transform) = self.transform {
            metric = transform.transform(&metric);
        }

        metric
    }
}

impl<T: MetricCategory> Default for MetricBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Zero-cost metric computation trait
pub trait ZeroCostMetric<T: MetricCategory> {
    /// Compute metric at compile time if possible
    fn compute_const(&self) -> Option<f64>;

    /// Runtime computation fallback
    fn compute_runtime(&self) -> f64;

    /// Get the metric value (compile-time if possible, runtime otherwise)
    fn get_value(&self) -> f64 {
        self.compute_const()
            .unwrap_or_else(|| self.compute_runtime())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_typed_metric_creation() {
        let accuracy = TypedMetric::<Classification>::new(0.85);
        assert_eq!(accuracy.value(), 0.85);
        assert_eq!(accuracy.category(), "Classification");
        assert!(accuracy.higher_is_better());
    }

    #[test]
    fn test_typed_metric_validation() {
        let accuracy = TypedMetric::<Classification>::new(0.85);
        assert!(accuracy.validate().is_ok());

        let invalid_accuracy = TypedMetric::<Classification>::new(1.5);
        assert!(invalid_accuracy.validate().is_err());
    }

    #[test]
    fn test_typed_metric_arithmetic() {
        let accuracy1 = TypedMetric::<Classification>::new(0.8);
        let accuracy2 = TypedMetric::<Classification>::new(0.9);

        let sum = accuracy1 + accuracy2.clone();
        assert!((sum.value() - 1.7).abs() < 1e-10);

        let scaled = accuracy2 * 2.0;
        assert!((scaled.value() - 1.8).abs() < 1e-10);
    }

    #[test]
    fn test_accuracy_metric() {
        let y_true = Array1::from_vec(vec![0, 1, 1, 0, 1]);
        let y_pred = Array1::from_vec(vec![0, 1, 0, 0, 1]);

        let accuracy_metric = AccuracyMetric;
        let result = accuracy_metric.compute(&(y_true, y_pred)).unwrap();

        assert_eq!(result.value(), 0.8);
        assert_eq!(result.category(), "Classification");
    }

    #[test]
    fn test_mse_metric() {
        let y_true = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y_pred = Array1::from_vec(vec![1.1, 2.1, 2.9]);

        let mse_metric = MeanSquaredErrorMetric;
        let result = mse_metric.compute(&(y_true, y_pred)).unwrap();

        assert!((result.value() - 0.01).abs() < 1e-10);
        assert_eq!(result.category(), "Regression");
    }

    #[test]
    fn test_metric_suite() {
        let mut suite = MetricSuite::<Classification>::new();

        suite.add_metric("accuracy".to_string(), TypedMetric::new(0.85));
        suite.add_metric("precision".to_string(), TypedMetric::new(0.90));
        suite.add_metric("recall".to_string(), TypedMetric::new(0.80));

        assert_eq!(suite.len(), 3);
        assert_eq!(suite.get(0).unwrap().value(), 0.85);
        assert_eq!(suite.get_by_name("precision").unwrap().value(), 0.90);

        let summary = suite.summary();
        assert_eq!(summary.count, 3);
        assert!((summary.mean.value() - 0.85).abs() < 1e-10);
    }

    #[test]
    fn test_composite_metric() {
        let accuracy = TypedMetric::<Classification>::new(0.85);
        let mse = TypedMetric::<Regression>::new(0.1);

        let composite = CompositeMetric::new(accuracy, mse);

        assert_eq!(composite.primary().value(), 0.85);
        assert_eq!(composite.secondary().value(), 0.1);

        let combined = composite.weighted_combination(0.7, 0.3);
        assert!(combined > 0.0 && combined < 1.0);
    }

    #[test]
    fn test_metric_transforms() {
        let accuracy = TypedMetric::<Classification>::new(0.85);

        let normalizer = NormalizeTransform::new(0.0, 1.0);
        let normalized = normalizer.transform(&accuracy);
        assert_eq!(normalized.value(), 0.85);

        let log_transform = LogTransform;
        let log_transformed = log_transform.transform(&accuracy);
        assert!((log_transformed.value() - 0.85_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_metric_builder() {
        let builder = MetricBuilder::<Classification>::new()
            .name("test_accuracy".to_string())
            .transform(NormalizeTransform::new(0.0, 1.0));

        let metric = builder.build(0.85);
        assert_eq!(metric.value(), 0.85);
        assert_eq!(metric.category(), "Classification");
    }

    #[test]
    fn test_validated_metric() {
        let values = [0.85];
        assert!(AccuracyMetric::validate_bounds(&values));

        let invalid_values = [1.5];
        assert!(!AccuracyMetric::validate_bounds(&invalid_values));
    }

    #[test]
    fn test_metric_metadata() {
        let accuracy_metric = AccuracyMetric;
        let metadata = accuracy_metric.metadata();

        assert_eq!(metadata.name, "accuracy");
        assert_eq!(metadata.category, "Classification");
        assert!(metadata.higher_is_better);
        assert_eq!(metadata.typical_range, (0.0, 1.0));
    }
}
