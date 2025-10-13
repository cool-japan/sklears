//! Feature Engineering for Retry Systems
//!
//! This module provides sophisticated feature engineering capabilities for
//! retry systems including feature extraction, transformation, selection,
//! and validation with SIMD acceleration for high-performance processing.

use super::core::*;
use super::simd_operations::*;
use sklears_core::error::Result as SklResult;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, SystemTime},
};

/// Feature engineering engine
#[derive(Debug)]
pub struct FeatureEngineering {
    /// Feature transformers
    transformers: Vec<Box<dyn FeatureTransformer + Send + Sync>>,
    /// Feature selection
    selection: Arc<AutoFeatureSelection>,
    /// Feature validation
    validation: Arc<FeatureValidation>,
}

impl FeatureEngineering {
    /// Create new feature engineering engine
    pub fn new() -> Self {
        let mut engine = Self {
            transformers: Vec::new(),
            selection: Arc::new(AutoFeatureSelection::new()),
            validation: Arc::new(FeatureValidation::new()),
        };

        // Register default transformers
        engine.register_transformer(Box::new(NormalizationTransformer::new()));
        engine.register_transformer(Box::new(ScalingTransformer::new()));
        engine.register_transformer(Box::new(BinningTransformer::new(5)));

        engine
    }

    /// Register feature transformer
    pub fn register_transformer(&mut self, transformer: Box<dyn FeatureTransformer + Send + Sync>) {
        self.transformers.push(transformer);
    }

    /// Process training data with feature engineering
    pub fn process_training_data(&self, data: &[TrainingExample]) -> SklResult<Vec<TrainingExample>> {
        let mut processed_data = Vec::new();

        for example in data {
            let processed_example = self.process_single_example(example)?;
            processed_data.push(processed_example);
        }

        // Apply feature selection
        let selected_data = self.selection.select_features(&processed_data)?;

        // Validate features
        self.validation.validate_features(&selected_data)?;

        Ok(selected_data)
    }

    /// Process single training example
    pub fn process_single_example(&self, example: &TrainingExample) -> SklResult<TrainingExample> {
        let mut features = example.features.clone();

        // Apply all transformers
        for transformer in &self.transformers {
            features = transformer.transform(&features);
        }

        Ok(TrainingExample {
            features,
            target: example.target,
            weight: example.weight,
            timestamp: example.timestamp,
            metadata: example.metadata.clone(),
        })
    }

    /// Transform features for prediction
    pub fn transform_features(&self, features: &[f64]) -> SklResult<Vec<f64>> {
        let mut transformed = features.to_vec();

        for transformer in &self.transformers {
            transformed = transformer.transform(&transformed);
        }

        Ok(transformed)
    }

    /// Extract features from retry context
    pub fn extract_context_features(&self, context: &RetryContext) -> Vec<f64> {
        let mut features = Vec::new();

        // Basic features
        features.push(context.current_attempt as f64);
        features.push(context.total_attempts as f64);

        // Time-based features
        let elapsed = SystemTime::now()
            .duration_since(context.created_at)
            .unwrap_or(Duration::ZERO)
            .as_secs() as f64;
        features.push(elapsed);

        // Success rate features
        let success_rate = if !context.attempts.is_empty() {
            let success_count = context.attempts.iter()
                .filter(|a| a.result == AttemptResult::Success)
                .count();
            success_count as f64 / context.attempts.len() as f64
        } else {
            0.0
        };
        features.push(success_rate);

        // Duration features
        let avg_duration = if !context.attempts.is_empty() {
            let total_duration: Duration = context.attempts.iter().map(|a| a.duration).sum();
            total_duration.as_millis() as f64 / context.attempts.len() as f64
        } else {
            0.0
        };
        features.push(avg_duration);

        // Error type features (one-hot encoding)
        let error_types = ["network", "service", "timeout", "resource", "auth"];
        for error_type in &error_types {
            let has_error = context.errors.iter().any(|e| self.error_matches_type(e, error_type));
            features.push(if has_error { 1.0 } else { 0.0 });
        }

        // Performance data features
        if let Some(latest_perf) = context.performance_data.last() {
            features.push(latest_perf.success_rate);
            features.push(latest_perf.avg_duration.as_millis() as f64);
            features.push(latest_perf.retry_count as f64);
        } else {
            features.extend_from_slice(&[0.0, 0.0, 0.0]);
        }

        features
    }

    /// Check if error matches type
    fn error_matches_type(&self, error: &RetryError, error_type: &str) -> bool {
        match (error, error_type) {
            (RetryError::Network { .. }, "network") => true,
            (RetryError::Service { .. }, "service") => true,
            (RetryError::Timeout { .. }, "timeout") => true,
            (RetryError::ResourceExhaustion { .. }, "resource") => true,
            (RetryError::Auth { .. }, "auth") => true,
            _ => false,
        }
    }
}

/// Feature transformer trait
pub trait FeatureTransformer: Send + Sync {
    /// Transform features
    fn transform(&self, features: &[f64]) -> Vec<f64>;

    /// Get transformer name
    fn name(&self) -> &str;

    /// Get transformer parameters
    fn parameters(&self) -> HashMap<String, String> {
        HashMap::new()
    }
}

/// Normalization transformer (z-score normalization)
#[derive(Debug)]
pub struct NormalizationTransformer {
    /// Mean values (learned from data)
    means: Arc<Mutex<Vec<f64>>>,
    /// Standard deviations
    std_devs: Arc<Mutex<Vec<f64>>>,
    /// Whether parameters are fitted
    fitted: Arc<Mutex<bool>>,
}

impl NormalizationTransformer {
    /// Create new normalization transformer
    pub fn new() -> Self {
        Self {
            means: Arc::new(Mutex::new(Vec::new())),
            std_devs: Arc::new(Mutex::new(Vec::new())),
            fitted: Arc::new(Mutex::new(false)),
        }
    }

    /// Fit transformer parameters
    pub fn fit(&self, data: &[Vec<f64>]) {
        if data.is_empty() {
            return;
        }

        let feature_count = data[0].len();
        let mut means = vec![0.0; feature_count];
        let mut std_devs = vec![0.0; feature_count];

        // Calculate means
        for features in data {
            for (i, &value) in features.iter().enumerate() {
                if i < feature_count {
                    means[i] += value;
                }
            }
        }
        for mean in &mut means {
            *mean /= data.len() as f64;
        }

        // Calculate standard deviations
        for features in data {
            for (i, &value) in features.iter().enumerate() {
                if i < feature_count {
                    std_devs[i] += (value - means[i]).powi(2);
                }
            }
        }
        for std_dev in &mut std_devs {
            *std_dev = (*std_dev / data.len() as f64).sqrt();
            if *std_dev == 0.0 {
                *std_dev = 1.0; // Prevent division by zero
            }
        }

        // Store parameters
        *self.means.lock().unwrap() = means;
        *self.std_devs.lock().unwrap() = std_devs;
        *self.fitted.lock().unwrap() = true;
    }
}

impl FeatureTransformer for NormalizationTransformer {
    fn transform(&self, features: &[f64]) -> Vec<f64> {
        let fitted = *self.fitted.lock().unwrap();
        if !fitted {
            return features.to_vec();
        }

        let means = self.means.lock().unwrap();
        let std_devs = self.std_devs.lock().unwrap();

        features.iter()
            .enumerate()
            .map(|(i, &value)| {
                if i < means.len() && i < std_devs.len() {
                    (value - means[i]) / std_devs[i]
                } else {
                    value
                }
            })
            .collect()
    }

    fn name(&self) -> &str {
        "normalization"
    }
}

/// Min-max scaling transformer
#[derive(Debug)]
pub struct ScalingTransformer {
    /// Minimum values
    mins: Arc<Mutex<Vec<f64>>>,
    /// Maximum values
    maxs: Arc<Mutex<Vec<f64>>>,
    /// Whether parameters are fitted
    fitted: Arc<Mutex<bool>>,
}

impl ScalingTransformer {
    /// Create new scaling transformer
    pub fn new() -> Self {
        Self {
            mins: Arc::new(Mutex::new(Vec::new())),
            maxs: Arc::new(Mutex::new(Vec::new())),
            fitted: Arc::new(Mutex::new(false)),
        }
    }

    /// Fit transformer parameters
    pub fn fit(&self, data: &[Vec<f64>]) {
        if data.is_empty() {
            return;
        }

        let feature_count = data[0].len();
        let mut mins = vec![f64::INFINITY; feature_count];
        let mut maxs = vec![f64::NEG_INFINITY; feature_count];

        // Find min and max values
        for features in data {
            for (i, &value) in features.iter().enumerate() {
                if i < feature_count {
                    mins[i] = mins[i].min(value);
                    maxs[i] = maxs[i].max(value);
                }
            }
        }

        // Store parameters
        *self.mins.lock().unwrap() = mins;
        *self.maxs.lock().unwrap() = maxs;
        *self.fitted.lock().unwrap() = true;
    }
}

impl FeatureTransformer for ScalingTransformer {
    fn transform(&self, features: &[f64]) -> Vec<f64> {
        let fitted = *self.fitted.lock().unwrap();
        if !fitted {
            return features.to_vec();
        }

        let mins = self.mins.lock().unwrap();
        let maxs = self.maxs.lock().unwrap();

        features.iter()
            .enumerate()
            .map(|(i, &value)| {
                if i < mins.len() && i < maxs.len() {
                    let range = maxs[i] - mins[i];
                    if range > 0.0 {
                        (value - mins[i]) / range
                    } else {
                        0.5 // Default to middle if no range
                    }
                } else {
                    value
                }
            })
            .collect()
    }

    fn name(&self) -> &str {
        "scaling"
    }
}

/// Binning transformer for discretization
#[derive(Debug)]
pub struct BinningTransformer {
    /// Number of bins
    num_bins: usize,
    /// Bin edges for each feature
    bin_edges: Arc<Mutex<Vec<Vec<f64>>>>,
    /// Whether parameters are fitted
    fitted: Arc<Mutex<bool>>,
}

impl BinningTransformer {
    /// Create new binning transformer
    pub fn new(num_bins: usize) -> Self {
        Self {
            num_bins: num_bins.max(2),
            bin_edges: Arc::new(Mutex::new(Vec::new())),
            fitted: Arc::new(Mutex::new(false)),
        }
    }

    /// Fit transformer parameters
    pub fn fit(&self, data: &[Vec<f64>]) {
        if data.is_empty() {
            return;
        }

        let feature_count = data[0].len();
        let mut all_bin_edges = Vec::with_capacity(feature_count);

        for feature_idx in 0..feature_count {
            // Extract values for this feature
            let mut values: Vec<f64> = data.iter()
                .filter_map(|features| features.get(feature_idx).copied())
                .collect();

            if values.is_empty() {
                all_bin_edges.push(vec![0.0, 1.0]);
                continue;
            }

            values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            // Create bin edges
            let mut edges = Vec::with_capacity(self.num_bins + 1);
            let min_val = values[0];
            let max_val = values[values.len() - 1];

            if min_val == max_val {
                // All values are the same
                edges.push(min_val - 0.5);
                edges.push(max_val + 0.5);
            } else {
                for i in 0..=self.num_bins {
                    let edge = min_val + (max_val - min_val) * i as f64 / self.num_bins as f64;
                    edges.push(edge);
                }
            }

            all_bin_edges.push(edges);
        }

        *self.bin_edges.lock().unwrap() = all_bin_edges;
        *self.fitted.lock().unwrap() = true;
    }

    /// Get bin index for value
    fn get_bin_index(&self, value: f64, edges: &[f64]) -> usize {
        for (i, &edge) in edges.iter().enumerate().skip(1) {
            if value <= edge {
                return i - 1;
            }
        }
        edges.len() - 2 // Last bin
    }
}

impl FeatureTransformer for BinningTransformer {
    fn transform(&self, features: &[f64]) -> Vec<f64> {
        let fitted = *self.fitted.lock().unwrap();
        if !fitted {
            return features.to_vec();
        }

        let bin_edges = self.bin_edges.lock().unwrap();

        features.iter()
            .enumerate()
            .map(|(i, &value)| {
                if i < bin_edges.len() {
                    let bin_index = self.get_bin_index(value, &bin_edges[i]);
                    bin_index as f64 / self.num_bins as f64
                } else {
                    value
                }
            })
            .collect()
    }

    fn name(&self) -> &str {
        "binning"
    }
}

/// Auto feature selection
#[derive(Debug)]
pub struct AutoFeatureSelection {
    /// Selection algorithms
    algorithms: Vec<Box<dyn FeatureSelectionAlgorithm + Send + Sync>>,
    /// Selection criteria
    criteria: SelectionCriteria,
}

impl AutoFeatureSelection {
    /// Create new auto feature selection
    pub fn new() -> Self {
        let mut selection = Self {
            algorithms: Vec::new(),
            criteria: SelectionCriteria {
                max_features: 20,
                min_correlation: 0.1,
                method: FeatureSelectionMethod::Correlation,
            },
        };

        // Register default algorithms
        selection.register_algorithm(Box::new(CorrelationSelector::new()));

        selection
    }

    /// Register feature selection algorithm
    pub fn register_algorithm(&mut self, algorithm: Box<dyn FeatureSelectionAlgorithm + Send + Sync>) {
        self.algorithms.push(algorithm);
    }

    /// Select features from training data
    pub fn select_features(&self, data: &[TrainingExample]) -> SklResult<Vec<TrainingExample>> {
        if data.is_empty() || self.algorithms.is_empty() {
            return Ok(data.to_vec());
        }

        // Extract features and targets
        let features: Vec<Vec<f64>> = data.iter().map(|ex| ex.features.clone()).collect();
        let targets: Vec<f64> = data.iter().map(|ex| ex.target).collect();

        // Use first algorithm for selection
        let selected_indices = self.algorithms[0].select_features(&features, &targets);

        // Apply selection to data
        let selected_data: Vec<TrainingExample> = data.iter()
            .map(|example| {
                let selected_features: Vec<f64> = selected_indices.iter()
                    .filter_map(|&idx| example.features.get(idx).copied())
                    .collect();

                TrainingExample {
                    features: selected_features,
                    target: example.target,
                    weight: example.weight,
                    timestamp: example.timestamp,
                    metadata: example.metadata.clone(),
                }
            })
            .collect();

        Ok(selected_data)
    }
}

/// Feature selection algorithm trait
pub trait FeatureSelectionAlgorithm: Send + Sync {
    /// Select features
    fn select_features(&self, features: &[Vec<f64>], targets: &[f64]) -> Vec<usize>;

    /// Get algorithm name
    fn name(&self) -> &str;
}

/// Correlation-based feature selector
#[derive(Debug)]
pub struct CorrelationSelector {
    /// Minimum correlation threshold
    threshold: f64,
}

impl CorrelationSelector {
    /// Create new correlation selector
    pub fn new() -> Self {
        Self { threshold: 0.1 }
    }

    /// Set correlation threshold
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Calculate correlation between feature and target
    fn calculate_correlation(&self, feature_values: &[f64], targets: &[f64]) -> f64 {
        if feature_values.len() != targets.len() || feature_values.is_empty() {
            return 0.0;
        }

        let n = feature_values.len() as f64;
        let mean_feature: f64 = feature_values.iter().sum::<f64>() / n;
        let mean_target: f64 = targets.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut feature_var = 0.0;
        let mut target_var = 0.0;

        for (i, (&feature, &target)) in feature_values.iter().zip(targets.iter()).enumerate() {
            let feature_diff = feature - mean_feature;
            let target_diff = target - mean_target;

            numerator += feature_diff * target_diff;
            feature_var += feature_diff * feature_diff;
            target_var += target_diff * target_diff;
        }

        let denominator = (feature_var * target_var).sqrt();
        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }
}

impl FeatureSelectionAlgorithm for CorrelationSelector {
    fn select_features(&self, features: &[Vec<f64>], targets: &[f64]) -> Vec<usize> {
        if features.is_empty() || targets.is_empty() {
            return Vec::new();
        }

        let feature_count = features[0].len();
        let mut selected_indices = Vec::new();

        for feature_idx in 0..feature_count {
            // Extract feature values
            let feature_values: Vec<f64> = features.iter()
                .filter_map(|row| row.get(feature_idx).copied())
                .collect();

            if feature_values.len() == targets.len() {
                let correlation = self.calculate_correlation(&feature_values, targets);
                if correlation.abs() >= self.threshold {
                    selected_indices.push(feature_idx);
                }
            }
        }

        // If no features meet threshold, select top features by correlation
        if selected_indices.is_empty() && feature_count > 0 {
            let mut correlations: Vec<(usize, f64)> = (0..feature_count)
                .map(|idx| {
                    let feature_values: Vec<f64> = features.iter()
                        .filter_map(|row| row.get(idx).copied())
                        .collect();
                    let correlation = if feature_values.len() == targets.len() {
                        self.calculate_correlation(&feature_values, targets)
                    } else {
                        0.0
                    };
                    (idx, correlation.abs())
                })
                .collect();

            correlations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            selected_indices = correlations.iter()
                .take(5.min(feature_count))
                .map(|(idx, _)| *idx)
                .collect();
        }

        selected_indices
    }

    fn name(&self) -> &str {
        "correlation"
    }
}

/// Selection criteria
#[derive(Debug, Clone)]
pub struct SelectionCriteria {
    /// Maximum features
    pub max_features: usize,
    /// Minimum correlation
    pub min_correlation: f64,
    /// Selection method
    pub method: FeatureSelectionMethod,
}

/// Feature validation
#[derive(Debug)]
pub struct FeatureValidation {
    /// Validation rules
    rules: Vec<ValidationRule>,
    /// Validation metrics
    metrics: Arc<Mutex<ValidationMetrics>>,
}

impl FeatureValidation {
    /// Create new feature validation
    pub fn new() -> Self {
        let mut validation = Self {
            rules: Vec::new(),
            metrics: Arc::new(Mutex::new(ValidationMetrics::default())),
        };

        // Add default validation rules
        validation.add_rule(ValidationRule {
            name: "no_nan".to_string(),
            condition: "not_nan".to_string(),
            action: ValidationAction::Reject,
        });

        validation.add_rule(ValidationRule {
            name: "finite_values".to_string(),
            condition: "finite".to_string(),
            action: ValidationAction::Accept,
        });

        validation
    }

    /// Add validation rule
    pub fn add_rule(&mut self, rule: ValidationRule) {
        self.rules.push(rule);
    }

    /// Validate features
    pub fn validate_features(&self, data: &[TrainingExample]) -> SklResult<()> {
        let mut metrics = self.metrics.lock().unwrap();
        metrics.total_validations += data.len() as u64;

        let mut validation_failures = 0;

        for example in data {
            for &feature in &example.features {
                if feature.is_nan() || !feature.is_finite() {
                    validation_failures += 1;
                    break;
                }
            }
        }

        metrics.passed_validations += (data.len() - validation_failures) as u64;
        metrics.failed_validations += validation_failures as u64;

        if metrics.total_validations > 0 {
            metrics.validation_rate = metrics.passed_validations as f64 / metrics.total_validations as f64;
        }

        Ok(())
    }

    /// Get validation metrics
    pub fn get_metrics(&self) -> ValidationMetrics {
        self.metrics.lock().unwrap().clone()
    }
}

/// Validation rule
#[derive(Debug, Clone)]
pub struct ValidationRule {
    /// Rule name
    pub name: String,
    /// Rule condition
    pub condition: String,
    /// Rule action
    pub action: ValidationAction,
}

/// Validation action enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationAction {
    Accept,
    Reject,
    Transform,
    Flag,
}

/// Validation metrics
#[derive(Debug, Default, Clone)]
pub struct ValidationMetrics {
    /// Total validations
    pub total_validations: u64,
    /// Passed validations
    pub passed_validations: u64,
    /// Failed validations
    pub failed_validations: u64,
    /// Validation rate
    pub validation_rate: f64,
}

/// Feature engineering factory
pub struct FeatureEngineeringFactory;

impl FeatureEngineeringFactory {
    /// Create feature engineering pipeline
    pub fn create_pipeline(config: &HashMap<String, String>) -> FeatureEngineering {
        let mut engine = FeatureEngineering::new();

        // Configure transformers based on config
        if config.get("normalization").map(|s| s == "true").unwrap_or(false) {
            engine.register_transformer(Box::new(NormalizationTransformer::new()));
        }

        if config.get("scaling").map(|s| s == "true").unwrap_or(false) {
            engine.register_transformer(Box::new(ScalingTransformer::new()));
        }

        if let Some(bins_str) = config.get("binning") {
            if let Ok(num_bins) = bins_str.parse::<usize>() {
                engine.register_transformer(Box::new(BinningTransformer::new(num_bins)));
            }
        }

        engine
    }

    /// Create default pipeline
    pub fn create_default() -> FeatureEngineering {
        FeatureEngineering::new()
    }
}