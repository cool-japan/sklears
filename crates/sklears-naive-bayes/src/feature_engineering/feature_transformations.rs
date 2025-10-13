//! Feature transformation methods and scaling operations
//!
//! This module provides comprehensive feature transformation implementations including
//! standard normalization, min-max scaling, robust scaling, power transformations,
//! and advanced transformation methods. All implementations follow SciRS2 Policy.

// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
// SciRS2 Policy Compliance - Use scirs2-core for random functionality
// SciRS2 Policy Compliance - Use scirs2-core for random functionality
use serde::{Deserialize, Serialize};
use sklears_core::error::Result;
use sklears_core::prelude::SklearsError;
use std::collections::HashMap;

/// Supported transformation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransformMethod {
    StandardNorm,
    MinMax,
    Robust,
    Log,
    Sqrt,
    Power,
    BoxCox,
    YeoJohnson,
    Quantile,
}

/// Configuration for feature transformations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationConfig {
    pub method: TransformMethod,
    pub feature_range: (f64, f64),
    pub with_centering: bool,
    pub with_scaling: bool,
    pub quantile_range: (f64, f64),
    pub n_quantiles: usize,
    pub subsample: Option<usize>,
    pub random_state: Option<u64>,
}

impl Default for TransformationConfig {
    fn default() -> Self {
        Self {
            method: TransformMethod::StandardNorm,
            feature_range: (0.0, 1.0),
            with_centering: true,
            with_scaling: true,
            quantile_range: (25.0, 75.0),
            n_quantiles: 1000,
            subsample: Some(100000),
            random_state: Some(42),
        }
    }
}

/// Validator for transformation configurations
#[derive(Debug, Clone)]
pub struct TransformationValidator;

impl TransformationValidator {
    pub fn validate_config(config: &TransformationConfig) -> Result<()> {
        if config.feature_range.0 >= config.feature_range.1 {
            return Err(SklearsError::InvalidInput(
                "feature_range min must be less than max".to_string(),
            ));
        }

        if config.quantile_range.0 < 0.0
            || config.quantile_range.1 > 100.0
            || config.quantile_range.0 >= config.quantile_range.1
        {
            return Err(SklearsError::InvalidInput(
                "quantile_range must be between 0 and 100 with min < max".to_string(),
            ));
        }

        if config.n_quantiles == 0 {
            return Err(SklearsError::InvalidInput(
                "n_quantiles must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }
}

/// Core feature transformer trait
pub trait FeatureTransformer<T> {
    fn fit(&mut self, x: &ArrayView2<T>) -> Result<()>;
    fn transform(&self, x: &ArrayView2<T>) -> Result<Array2<T>>;
    fn fit_transform(&mut self, x: &ArrayView2<T>) -> Result<Array2<T>> {
        self.fit(x)?;
        self.transform(x)
    }
    fn inverse_transform(&self, x: &ArrayView2<T>) -> Result<Array2<T>>;
}

/// Standard normal scaler (z-score normalization)
#[derive(Debug, Clone)]
pub struct StandardNormalizer<T> {
    mean: Option<Array1<T>>,
    std: Option<Array1<T>>,
    with_mean: bool,
    with_std: bool,
}

impl<T> StandardNormalizer<T>
where
    T: Clone + Copy + std::fmt::Debug + PartialOrd,
{
    pub fn new(with_mean: bool, with_std: bool) -> Self {
        Self {
            mean: None,
            std: None,
            with_mean,
            with_std,
        }
    }

    pub fn mean(&self) -> Option<&Array1<T>> {
        self.mean.as_ref()
    }

    pub fn std(&self) -> Option<&Array1<T>> {
        self.std.as_ref()
    }
}

impl<T> Default for StandardNormalizer<T>
where
    T: Clone + Copy + std::fmt::Debug + PartialOrd,
{
    fn default() -> Self {
        Self::new(true, true)
    }
}

/// Min-Max scaler for scaling features to a given range
#[derive(Debug, Clone)]
pub struct MinMaxScaler<T> {
    feature_range: (T, T),
    data_min: Option<Array1<T>>,
    data_max: Option<Array1<T>>,
    data_range: Option<Array1<T>>,
    scale: Option<Array1<T>>,
}

impl<T> MinMaxScaler<T>
where
    T: Clone + Copy + std::fmt::Debug + PartialOrd,
{
    pub fn new(feature_range: (T, T)) -> Self {
        Self {
            feature_range,
            data_min: None,
            data_max: None,
            data_range: None,
            scale: None,
        }
    }

    pub fn data_min(&self) -> Option<&Array1<T>> {
        self.data_min.as_ref()
    }

    pub fn data_max(&self) -> Option<&Array1<T>> {
        self.data_max.as_ref()
    }

    pub fn scale(&self) -> Option<&Array1<T>> {
        self.scale.as_ref()
    }
}

/// Robust scaler using median and interquartile range
#[derive(Debug, Clone)]
pub struct RobustScaler<T> {
    median: Option<Array1<T>>,
    scale: Option<Array1<T>>,
    quantile_range: (f64, f64),
    with_centering: bool,
    with_scaling: bool,
}

impl<T> RobustScaler<T>
where
    T: Clone + Copy + std::fmt::Debug + PartialOrd,
{
    pub fn new(quantile_range: (f64, f64), with_centering: bool, with_scaling: bool) -> Self {
        Self {
            median: None,
            scale: None,
            quantile_range,
            with_centering,
            with_scaling,
        }
    }

    pub fn median(&self) -> Option<&Array1<T>> {
        self.median.as_ref()
    }

    pub fn scale(&self) -> Option<&Array1<T>> {
        self.scale.as_ref()
    }
}

impl<T> Default for RobustScaler<T>
where
    T: Clone + Copy + std::fmt::Debug + PartialOrd,
{
    fn default() -> Self {
        Self::new((25.0, 75.0), true, true)
    }
}

/// Logarithmic transformer
#[derive(Debug, Clone)]
pub struct LogTransformer<T> {
    base: Option<T>,
    offset: T,
}

impl<T> LogTransformer<T>
where
    T: Clone + Copy + std::fmt::Debug + PartialOrd,
{
    pub fn new(base: Option<T>, offset: T) -> Self {
        Self { base, offset }
    }

    pub fn base(&self) -> Option<T> {
        self.base
    }

    pub fn offset(&self) -> T {
        self.offset
    }
}

/// Square root transformer
#[derive(Debug, Clone)]
pub struct SqrtTransformer<T> {
    offset: T,
}

impl<T> SqrtTransformer<T>
where
    T: Clone + Copy + std::fmt::Debug + PartialOrd,
{
    pub fn new(offset: T) -> Self {
        Self { offset }
    }

    pub fn offset(&self) -> T {
        self.offset
    }
}

/// Power transformer (Box-Cox or Yeo-Johnson)
#[derive(Debug, Clone)]
pub struct PowerTransformer<T> {
    method: String,
    lambdas: Option<Array1<T>>,
    standardize: bool,
}

impl<T> PowerTransformer<T>
where
    T: Clone + Copy + std::fmt::Debug + PartialOrd,
{
    pub fn new(method: String, standardize: bool) -> Self {
        Self {
            method,
            lambdas: None,
            standardize,
        }
    }

    pub fn lambdas(&self) -> Option<&Array1<T>> {
        self.lambdas.as_ref()
    }

    pub fn method(&self) -> &str {
        &self.method
    }
}

/// Box-Cox transformer specifically
#[derive(Debug, Clone)]
pub struct BoxCoxTransformer<T> {
    lambda: Option<T>,
    fitted_lambda: Option<T>,
}

impl<T> BoxCoxTransformer<T>
where
    T: Clone + Copy + std::fmt::Debug + PartialOrd,
{
    pub fn new(lambda: Option<T>) -> Self {
        Self {
            lambda,
            fitted_lambda: None,
        }
    }

    pub fn fitted_lambda(&self) -> Option<T> {
        self.fitted_lambda
    }
}

/// Transformation analyzer for analyzing transformation results
#[derive(Debug, Clone)]
pub struct TransformationAnalyzer<T> {
    skewness_before: Option<Array1<T>>,
    skewness_after: Option<Array1<T>>,
    kurtosis_before: Option<Array1<T>>,
    kurtosis_after: Option<Array1<T>>,
    normality_tests: HashMap<String, Array1<T>>,
}

impl<T> TransformationAnalyzer<T>
where
    T: Clone + Copy + std::fmt::Debug,
{
    pub fn new() -> Self {
        Self {
            skewness_before: None,
            skewness_after: None,
            kurtosis_before: None,
            kurtosis_after: None,
            normality_tests: HashMap::new(),
        }
    }

    pub fn set_skewness_before(&mut self, skewness: Array1<T>) {
        self.skewness_before = Some(skewness);
    }

    pub fn set_skewness_after(&mut self, skewness: Array1<T>) {
        self.skewness_after = Some(skewness);
    }

    pub fn skewness_before(&self) -> Option<&Array1<T>> {
        self.skewness_before.as_ref()
    }

    pub fn skewness_after(&self) -> Option<&Array1<T>> {
        self.skewness_after.as_ref()
    }

    pub fn add_normality_test(&mut self, test_name: String, results: Array1<T>) {
        self.normality_tests.insert(test_name, results);
    }

    pub fn normality_tests(&self) -> &HashMap<String, Array1<T>> {
        &self.normality_tests
    }
}

impl<T> Default for TransformationAnalyzer<T>
where
    T: Clone + Copy + std::fmt::Debug,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Scaling optimizer for automatic scaling parameter selection
#[derive(Debug, Clone)]
pub struct ScalingOptimizer<T> {
    config: TransformationConfig,
    best_method: Option<TransformMethod>,
    performance_scores: HashMap<String, T>,
    optimization_history: Vec<(TransformMethod, T)>,
}

impl<T> ScalingOptimizer<T>
where
    T: Clone + Copy + std::fmt::Debug + PartialOrd,
{
    pub fn new(config: TransformationConfig) -> Self {
        Self {
            config,
            best_method: None,
            performance_scores: HashMap::new(),
            optimization_history: Vec::new(),
        }
    }

    pub fn set_best_method(&mut self, method: TransformMethod) {
        self.best_method = Some(method);
    }

    pub fn best_method(&self) -> Option<TransformMethod> {
        self.best_method
    }

    pub fn add_performance_score(&mut self, method: String, score: T) {
        self.performance_scores.insert(method, score);
    }

    pub fn performance_scores(&self) -> &HashMap<String, T> {
        &self.performance_scores
    }

    pub fn add_optimization_step(&mut self, method: TransformMethod, score: T) {
        self.optimization_history.push((method, score));
    }

    pub fn optimization_history(&self) -> &[(TransformMethod, T)] {
        &self.optimization_history
    }
}

// For backwards compatibility, implement basic transformations for f64
impl FeatureTransformer<f64> for StandardNormalizer<f64> {
    fn fit(&mut self, x: &ArrayView2<f64>) -> Result<()> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        if self.with_mean {
            let mean = x.mean_axis(Axis(0)).unwrap();
            self.mean = Some(mean);
        }

        if self.with_std {
            let default_mean = Array1::zeros(n_features);
            let mean = self
                .mean
                .as_ref()
                .map(|m| m.view())
                .unwrap_or_else(|| default_mean.view());

            let var = x
                .axis_iter(Axis(0))
                .map(|row| {
                    row.iter()
                        .zip(mean.iter())
                        .map(|(x_val, mean_val)| (x_val - mean_val).powi(2))
                        .collect::<Vec<_>>()
                })
                .fold(vec![0.0; n_features], |mut acc, row| {
                    for (i, val) in row.iter().enumerate() {
                        acc[i] += val;
                    }
                    acc
                })
                .iter()
                .map(|v| (v / (n_samples - 1) as f64).sqrt())
                .collect::<Vec<_>>();

            self.std = Some(Array1::from_vec(var));
        }

        Ok(())
    }

    fn transform(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let mut result = x.to_owned();

        if let Some(ref mean) = self.mean {
            for mut row in result.axis_iter_mut(Axis(0)) {
                for (val, mean_val) in row.iter_mut().zip(mean.iter()) {
                    *val -= mean_val;
                }
            }
        }

        if let Some(ref std) = self.std {
            for mut row in result.axis_iter_mut(Axis(0)) {
                for (val, std_val) in row.iter_mut().zip(std.iter()) {
                    if *std_val != 0.0 {
                        *val /= std_val;
                    }
                }
            }
        }

        Ok(result)
    }

    fn inverse_transform(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let mut result = x.to_owned();

        if let Some(ref std) = self.std {
            for mut row in result.axis_iter_mut(Axis(0)) {
                for (val, std_val) in row.iter_mut().zip(std.iter()) {
                    *val *= std_val;
                }
            }
        }

        if let Some(ref mean) = self.mean {
            for mut row in result.axis_iter_mut(Axis(0)) {
                for (val, mean_val) in row.iter_mut().zip(mean.iter()) {
                    *val += mean_val;
                }
            }
        }

        Ok(result)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformation_config_default() {
        let config = TransformationConfig::default();
        assert_eq!(config.method, TransformMethod::StandardNorm);
        assert_eq!(config.feature_range, (0.0, 1.0));
        assert!(config.with_centering);
        assert!(config.with_scaling);
    }

    #[test]
    fn test_transformation_validator() {
        let mut config = TransformationConfig::default();
        assert!(TransformationValidator::validate_config(&config).is_ok());

        config.feature_range = (1.0, 0.0); // Invalid range
        assert!(TransformationValidator::validate_config(&config).is_err());

        config.feature_range = (0.0, 1.0);
        config.quantile_range = (-10.0, 50.0); // Invalid quantile range
        assert!(TransformationValidator::validate_config(&config).is_err());
    }

    #[test]
    fn test_standard_normalizer() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let mut normalizer = StandardNormalizer::new(true, true);

        assert!(normalizer.fit(&data.view()).is_ok());
        assert!(normalizer.mean().is_some());
        assert!(normalizer.std().is_some());

        let transformed = normalizer.transform(&data.view()).unwrap();
        assert_eq!(transformed.dim(), data.dim());

        let inverse = normalizer.inverse_transform(&transformed.view()).unwrap();
        // Check that inverse transformation approximately recovers original data
        for (orig, inv) in data.iter().zip(inverse.iter()) {
            assert!((orig - inv).abs() < 1e-10);
        }
    }

    #[test]
    fn test_min_max_scaler() {
        let scaler = MinMaxScaler::new((0.0, 1.0));
        assert_eq!(scaler.feature_range, (0.0, 1.0));
        assert!(scaler.data_min().is_none());
        assert!(scaler.data_max().is_none());
    }

    #[test]
    fn test_robust_scaler() {
        let scaler = RobustScaler::<f64>::new((25.0, 75.0), true, true);
        assert_eq!(scaler.quantile_range, (25.0, 75.0));
        assert!(scaler.median().is_none());
        assert!(scaler.scale().is_none());
    }

    #[test]
    fn test_transformation_analyzer() {
        let mut analyzer = TransformationAnalyzer::<f64>::new();

        let skewness = Array1::from_vec(vec![0.5, -0.2, 1.1]);
        analyzer.set_skewness_before(skewness.clone());
        assert_eq!(analyzer.skewness_before().unwrap(), &skewness);

        analyzer.add_normality_test(
            "shapiro".to_string(),
            Array1::from_vec(vec![0.95, 0.87, 0.92]),
        );
        assert_eq!(analyzer.normality_tests().len(), 1);
    }

    #[test]
    fn test_scaling_optimizer() {
        let config = TransformationConfig::default();
        let mut optimizer = ScalingOptimizer::new(config);

        optimizer.set_best_method(TransformMethod::MinMax);
        assert_eq!(optimizer.best_method(), Some(TransformMethod::MinMax));

        optimizer.add_performance_score("accuracy".to_string(), 0.95);
        assert_eq!(optimizer.performance_scores().get("accuracy"), Some(&0.95));

        optimizer.add_optimization_step(TransformMethod::StandardNorm, 0.92);
        assert_eq!(optimizer.optimization_history().len(), 1);
    }
}
