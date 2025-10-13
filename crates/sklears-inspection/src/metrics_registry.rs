//! Custom Explanation Metric Registration System
//!
//! This module provides a comprehensive system for registering and managing
//! custom metrics for evaluating explanation quality, fidelity, and consistency.

use crate::{Float, SklResult};
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

/// Trait for custom explanation metrics
pub trait ExplanationMetric: Debug + Send + Sync {
    /// Metric identifier
    fn metric_id(&self) -> &str;

    /// Metric name
    fn metric_name(&self) -> &str;

    /// Metric description
    fn metric_description(&self) -> &str;

    /// Metric version
    fn metric_version(&self) -> &str;

    /// Metric author
    fn metric_author(&self) -> &str;

    /// Metric category
    fn metric_category(&self) -> MetricCategory;

    /// Supported explanation types
    fn supported_explanation_types(&self) -> Vec<ExplanationType>;

    /// Output range (min, max)
    fn output_range(&self) -> (Float, Float);

    /// Whether higher values are better
    fn higher_is_better(&self) -> bool;

    /// Metric properties
    fn properties(&self) -> MetricProperties;

    /// Compute the metric
    fn compute(&self, input: &MetricInput) -> SklResult<MetricOutput>;

    /// Validate input before computation
    fn validate_input(&self, input: &MetricInput) -> SklResult<()>;

    /// Get metric metadata
    fn metadata(&self) -> MetricMetadata {
        MetricMetadata {
            id: self.metric_id().to_string(),
            name: self.metric_name().to_string(),
            description: self.metric_description().to_string(),
            version: self.metric_version().to_string(),
            author: self.metric_author().to_string(),
            category: self.metric_category(),
            supported_types: self.supported_explanation_types(),
            output_range: self.output_range(),
            higher_is_better: self.higher_is_better(),
            properties: self.properties(),
            created_at: chrono::Utc::now(),
        }
    }
}

/// Metric categories
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetricCategory {
    /// Fidelity metrics (how well explanations match model behavior)
    Fidelity,
    /// Stability metrics (consistency across perturbations)
    Stability,
    /// Completeness metrics (coverage of feature space)
    Completeness,
    /// Efficiency metrics (computational cost)
    Efficiency,
    /// Interpretability metrics (human understanding)
    Interpretability,
    /// Robustness metrics (resilience to noise)
    Robustness,
    /// Fairness metrics (bias and discrimination)
    Fairness,
    /// Consistency metrics (agreement between methods)
    Consistency,
    /// Custom category
    Custom(String),
}

/// Explanation types supported by metrics
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExplanationType {
    /// Feature importance explanations
    FeatureImportance,
    /// Local explanations (instance-level)
    LocalExplanation,
    /// Global explanations (model-level)
    GlobalExplanation,
    /// Counterfactual explanations
    CounterfactualExplanation,
    /// SHAP value explanations
    ShapExplanation,
    /// LIME explanations
    LimeExplanation,
    /// Anchor explanations
    AnchorExplanation,
    /// Attention explanations
    AttentionExplanation,
    /// Gradient explanations
    GradientExplanation,
    /// Custom explanation type
    Custom(String),
}

/// Metric properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricProperties {
    /// Metric is deterministic (same input always produces same output)
    pub deterministic: bool,
    /// Metric requires model predictions
    pub requires_predictions: bool,
    /// Metric requires ground truth labels
    pub requires_labels: bool,
    /// Metric requires feature values
    pub requires_features: bool,
    /// Metric supports batch computation
    pub supports_batch: bool,
    /// Metric can be parallelized
    pub parallelizable: bool,
    /// Metric supports weighted samples
    pub supports_weights: bool,
    /// Minimum number of samples required
    pub min_samples: Option<usize>,
    /// Maximum number of samples efficiently handled
    pub max_samples: Option<usize>,
    /// Computational complexity (rough estimate)
    pub complexity: ComputationalComplexity,
}

impl Default for MetricProperties {
    fn default() -> Self {
        Self {
            deterministic: true,
            requires_predictions: false,
            requires_labels: false,
            requires_features: true,
            supports_batch: true,
            parallelizable: false,
            supports_weights: false,
            min_samples: None,
            max_samples: None,
            complexity: ComputationalComplexity::Linear,
        }
    }
}

/// Computational complexity estimates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputationalComplexity {
    /// O(1) - constant time
    Constant,
    /// O(log n) - logarithmic
    Logarithmic,
    /// O(n) - linear
    Linear,
    /// O(n log n) - linearithmic
    Linearithmic,
    /// O(n²) - quadratic
    Quadratic,
    /// O(n³) - cubic
    Cubic,
    /// O(2ⁿ) - exponential
    Exponential,
}

/// Metric input data
#[derive(Debug, Clone)]
pub struct MetricInput {
    /// Original explanations
    pub explanations: ExplanationData,
    /// Reference explanations (for comparison metrics)
    pub reference_explanations: Option<ExplanationData>,
    /// Model predictions
    pub predictions: Option<Array1<Float>>,
    /// Ground truth labels
    pub labels: Option<Array1<Float>>,
    /// Feature values
    pub features: Option<Array2<Float>>,
    /// Feature names
    pub feature_names: Option<Vec<String>>,
    /// Sample weights
    pub sample_weights: Option<Array1<Float>>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Explanation data wrapper
#[derive(Debug, Clone)]
pub enum ExplanationData {
    /// Feature importance values
    FeatureImportance(Array1<Float>),
    /// Local explanation matrix (samples x features)
    LocalExplanations(Array2<Float>),
    /// Global explanation values
    GlobalExplanation(Array1<Float>),
    /// Counterfactual instances
    CounterfactualExplanations(Array2<Float>),
    /// SHAP values
    ShapValues(Array2<Float>),
    /// LIME coefficients
    LimeCoefficients(Array2<Float>),
    /// Anchor rules (as binary indicators)
    AnchorRules(Array2<bool>),
    /// Attention weights
    AttentionWeights(Array2<Float>),
    /// Gradient attributions
    GradientAttributions(Array2<Float>),
    /// Custom explanation format
    Custom(serde_json::Value),
}

/// Metric output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricOutput {
    /// Metric value
    pub value: Float,
    /// Per-sample values (if applicable)
    pub per_sample_values: Option<Array1<Float>>,
    /// Per-feature values (if applicable)
    pub per_feature_values: Option<Array1<Float>>,
    /// Confidence interval (if applicable)
    pub confidence_interval: Option<(Float, Float)>,
    /// Statistical significance (p-value)
    pub p_value: Option<Float>,
    /// Computation metadata
    pub metadata: ComputationMetadata,
    /// Additional statistics
    pub statistics: HashMap<String, Float>,
}

/// Computation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationMetadata {
    /// Computation time in milliseconds
    pub computation_time_ms: u64,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Number of samples processed
    pub samples_processed: usize,
    /// Number of features processed
    pub features_processed: usize,
    /// Warning messages
    pub warnings: Vec<String>,
    /// Computation timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Metric metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricMetadata {
    /// Metric ID
    pub id: String,
    /// Metric name
    pub name: String,
    /// Metric description
    pub description: String,
    /// Metric version
    pub version: String,
    /// Metric author
    pub author: String,
    /// Metric category
    pub category: MetricCategory,
    /// Supported explanation types
    pub supported_types: Vec<ExplanationType>,
    /// Output range
    pub output_range: (Float, Float),
    /// Whether higher is better
    pub higher_is_better: bool,
    /// Metric properties
    pub properties: MetricProperties,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Metric registry for managing custom metrics
#[derive(Debug, Default)]
pub struct MetricRegistry {
    metrics: Arc<RwLock<HashMap<String, Arc<dyn ExplanationMetric>>>>,
    metadata: Arc<RwLock<HashMap<String, MetricMetadata>>>,
    categories: Arc<RwLock<HashMap<MetricCategory, Vec<String>>>>,
}

impl MetricRegistry {
    /// Create a new metric registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Create registry with default metrics
    pub fn with_default_metrics() -> Self {
        let registry = Self::new();
        registry.register_default_metrics();
        registry
    }

    /// Register default metrics
    pub fn register_default_metrics(&self) {
        // Register built-in metrics
        if let Ok(fidelity_metric) = self.register_metric(FidelityMetric::new()) {
            // Metric registered successfully
        }

        if let Ok(stability_metric) = self.register_metric(StabilityMetric::new()) {
            // Metric registered successfully
        }

        if let Ok(completeness_metric) = self.register_metric(CompletenessMetric::new()) {
            // Metric registered successfully
        }
    }

    /// Register a new metric
    pub fn register_metric<M: ExplanationMetric + 'static>(&self, metric: M) -> SklResult<()> {
        let metric_id = metric.metric_id().to_string();
        let metadata = metric.metadata();
        let category = metadata.category.clone();

        // Store metric and metadata
        {
            let mut metrics = self.metrics.write().map_err(|_| {
                crate::SklearsError::InvalidInput("Failed to acquire metrics lock".to_string())
            })?;
            metrics.insert(metric_id.clone(), Arc::new(metric));
        }

        {
            let mut metadata_store = self.metadata.write().map_err(|_| {
                crate::SklearsError::InvalidInput("Failed to acquire metadata lock".to_string())
            })?;
            metadata_store.insert(metric_id.clone(), metadata);
        }

        // Update category index
        {
            let mut categories = self.categories.write().map_err(|_| {
                crate::SklearsError::InvalidInput("Failed to acquire categories lock".to_string())
            })?;
            categories
                .entry(category)
                .or_insert_with(Vec::new)
                .push(metric_id);
        }

        Ok(())
    }

    /// Get a metric by ID
    pub fn get_metric(&self, metric_id: &str) -> Option<Arc<dyn ExplanationMetric>> {
        self.metrics.read().ok()?.get(metric_id).cloned()
    }

    /// Get metric metadata
    pub fn get_metric_metadata(&self, metric_id: &str) -> Option<MetricMetadata> {
        self.metadata.read().ok()?.get(metric_id).cloned()
    }

    /// List all registered metrics
    pub fn list_metrics(&self) -> Vec<String> {
        self.metrics
            .read()
            .ok()
            .map(|metrics| metrics.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// List metrics by category
    pub fn list_metrics_by_category(&self, category: MetricCategory) -> Vec<String> {
        self.categories
            .read()
            .ok()
            .and_then(|cats| cats.get(&category).cloned())
            .unwrap_or_default()
    }

    /// List metrics by explanation type
    pub fn list_metrics_by_explanation_type(
        &self,
        explanation_type: ExplanationType,
    ) -> Vec<String> {
        self.metadata
            .read()
            .ok()
            .map(|metadata| {
                metadata
                    .iter()
                    .filter(|(_, meta)| meta.supported_types.contains(&explanation_type))
                    .map(|(id, _)| id.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Compute a metric
    pub fn compute_metric(&self, metric_id: &str, input: &MetricInput) -> SklResult<MetricOutput> {
        let metric = self.get_metric(metric_id).ok_or_else(|| {
            crate::SklearsError::InvalidInput(format!("Metric '{}' not found", metric_id))
        })?;

        // Validate input
        metric.validate_input(input)?;

        // Compute metric
        let start_time = std::time::Instant::now();
        let mut result = metric.compute(input)?;
        let computation_time = start_time.elapsed().as_millis() as u64;

        // Update timing information
        result.metadata.computation_time_ms = computation_time;

        Ok(result)
    }

    /// Compute multiple metrics
    pub fn compute_multiple_metrics(
        &self,
        metric_ids: &[String],
        input: &MetricInput,
    ) -> HashMap<String, SklResult<MetricOutput>> {
        let mut results = HashMap::new();

        for metric_id in metric_ids {
            let result = self.compute_metric(metric_id, input);
            results.insert(metric_id.clone(), result);
        }

        results
    }

    /// Unregister a metric
    pub fn unregister_metric(&self, metric_id: &str) -> SklResult<()> {
        let category = self
            .get_metric_metadata(metric_id)
            .map(|meta| meta.category);

        // Remove from main registry
        {
            let mut metrics = self.metrics.write().map_err(|_| {
                crate::SklearsError::InvalidInput("Failed to acquire metrics lock".to_string())
            })?;
            metrics.remove(metric_id);
        }

        {
            let mut metadata_store = self.metadata.write().map_err(|_| {
                crate::SklearsError::InvalidInput("Failed to acquire metadata lock".to_string())
            })?;
            metadata_store.remove(metric_id);
        }

        // Remove from category index
        if let Some(cat) = category {
            let mut categories = self.categories.write().map_err(|_| {
                crate::SklearsError::InvalidInput("Failed to acquire categories lock".to_string())
            })?;
            if let Some(cat_metrics) = categories.get_mut(&cat) {
                cat_metrics.retain(|id| id != metric_id);
            }
        }

        Ok(())
    }

    /// Get registry statistics
    pub fn get_statistics(&self) -> RegistryStatistics {
        let metrics = self.metrics.read().ok();
        let categories = self.categories.read().ok();
        let metadata = self.metadata.read().ok();

        let total_metrics = metrics.as_ref().map(|m| m.len()).unwrap_or(0);

        let metrics_by_category = categories
            .as_ref()
            .map(|cats| {
                cats.iter()
                    .map(|(cat, metrics)| (cat.clone(), metrics.len()))
                    .collect()
            })
            .unwrap_or_default();

        let metrics_by_type = metadata
            .as_ref()
            .map(|meta| {
                let mut type_counts = HashMap::new();
                for (_, metadata) in meta.iter() {
                    for exp_type in &metadata.supported_types {
                        *type_counts.entry(exp_type.clone()).or_insert(0) += 1;
                    }
                }
                type_counts
            })
            .unwrap_or_default();

        RegistryStatistics {
            total_metrics,
            metrics_by_category,
            metrics_by_explanation_type: metrics_by_type,
            registry_created_at: chrono::Utc::now(),
        }
    }
}

/// Registry statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryStatistics {
    /// Total number of metrics
    pub total_metrics: usize,
    /// Number of metrics by category
    pub metrics_by_category: HashMap<MetricCategory, usize>,
    /// Number of metrics by explanation type
    pub metrics_by_explanation_type: HashMap<ExplanationType, usize>,
    /// Registry creation timestamp
    pub registry_created_at: chrono::DateTime<chrono::Utc>,
}

/// Built-in fidelity metric
#[derive(Debug)]
pub struct FidelityMetric {
    id: String,
    name: String,
    description: String,
    version: String,
    author: String,
}

impl FidelityMetric {
    /// Create a new fidelity metric
    pub fn new() -> Self {
        Self {
            id: "fidelity_correlation".to_string(),
            name: "Explanation Fidelity (Correlation)".to_string(),
            description:
                "Measures correlation between explanation importance and actual feature influence"
                    .to_string(),
            version: "1.0.0".to_string(),
            author: "Sklears Team".to_string(),
        }
    }
}

impl ExplanationMetric for FidelityMetric {
    fn metric_id(&self) -> &str {
        &self.id
    }

    fn metric_name(&self) -> &str {
        &self.name
    }

    fn metric_description(&self) -> &str {
        &self.description
    }

    fn metric_version(&self) -> &str {
        &self.version
    }

    fn metric_author(&self) -> &str {
        &self.author
    }

    fn metric_category(&self) -> MetricCategory {
        MetricCategory::Fidelity
    }

    fn supported_explanation_types(&self) -> Vec<ExplanationType> {
        vec![
            ExplanationType::FeatureImportance,
            ExplanationType::LocalExplanation,
            ExplanationType::ShapExplanation,
        ]
    }

    fn output_range(&self) -> (Float, Float) {
        (-1.0, 1.0)
    }

    fn higher_is_better(&self) -> bool {
        true
    }

    fn properties(&self) -> MetricProperties {
        MetricProperties {
            deterministic: true,
            requires_predictions: true,
            requires_labels: false,
            requires_features: true,
            supports_batch: true,
            parallelizable: true,
            supports_weights: false,
            min_samples: Some(2),
            max_samples: None,
            complexity: ComputationalComplexity::Linear,
        }
    }

    fn compute(&self, input: &MetricInput) -> SklResult<MetricOutput> {
        let start_time = std::time::Instant::now();

        // Extract explanation values
        let explanation_values = match &input.explanations {
            ExplanationData::FeatureImportance(values) => values.clone(),
            ExplanationData::LocalExplanations(matrix) => {
                // Use mean across instances
                matrix.mean_axis(Axis(0)).unwrap()
            }
            ExplanationData::ShapValues(matrix) => {
                // Use mean absolute SHAP values across instances
                matrix.mapv(|x| x.abs()).mean_axis(Axis(0)).unwrap()
            }
            _ => {
                return Err(crate::SklearsError::InvalidInput(
                    "Unsupported explanation type for fidelity metric".to_string(),
                ));
            }
        };

        // For this example, we'll use feature variance as a proxy for true importance
        let true_importance = if let Some(features) = &input.features {
            features.var_axis(Axis(0), 0.0)
        } else {
            return Err(crate::SklearsError::InvalidInput(
                "Features required for fidelity computation".to_string(),
            ));
        };

        // Compute correlation
        let correlation = compute_correlation(&explanation_values.view(), &true_importance.view())?;

        let computation_time = start_time.elapsed().as_millis() as u64;

        Ok(MetricOutput {
            value: correlation,
            per_sample_values: None,
            per_feature_values: None,
            confidence_interval: None,
            p_value: None,
            metadata: ComputationMetadata {
                computation_time_ms: computation_time,
                memory_usage_bytes: 0,
                samples_processed: input.features.as_ref().map(|f| f.nrows()).unwrap_or(0),
                features_processed: explanation_values.len(),
                warnings: Vec::new(),
                timestamp: chrono::Utc::now(),
            },
            statistics: HashMap::new(),
        })
    }

    fn validate_input(&self, input: &MetricInput) -> SklResult<()> {
        if input.features.is_none() {
            return Err(crate::SklearsError::InvalidInput(
                "Features required for fidelity metric".to_string(),
            ));
        }

        match &input.explanations {
            ExplanationData::FeatureImportance(_)
            | ExplanationData::LocalExplanations(_)
            | ExplanationData::ShapValues(_) => Ok(()),
            _ => Err(crate::SklearsError::InvalidInput(
                "Unsupported explanation type".to_string(),
            )),
        }
    }
}

impl Default for FidelityMetric {
    fn default() -> Self {
        Self::new()
    }
}

/// Built-in stability metric
#[derive(Debug)]
pub struct StabilityMetric {
    id: String,
    name: String,
    description: String,
    version: String,
    author: String,
}

impl StabilityMetric {
    /// Create a new stability metric
    pub fn new() -> Self {
        Self {
            id: "explanation_stability".to_string(),
            name: "Explanation Stability".to_string(),
            description: "Measures consistency of explanations across similar instances"
                .to_string(),
            version: "1.0.0".to_string(),
            author: "Sklears Team".to_string(),
        }
    }
}

impl ExplanationMetric for StabilityMetric {
    fn metric_id(&self) -> &str {
        &self.id
    }

    fn metric_name(&self) -> &str {
        &self.name
    }

    fn metric_description(&self) -> &str {
        &self.description
    }

    fn metric_version(&self) -> &str {
        &self.version
    }

    fn metric_author(&self) -> &str {
        &self.author
    }

    fn metric_category(&self) -> MetricCategory {
        MetricCategory::Stability
    }

    fn supported_explanation_types(&self) -> Vec<ExplanationType> {
        vec![
            ExplanationType::LocalExplanation,
            ExplanationType::ShapExplanation,
            ExplanationType::LimeExplanation,
        ]
    }

    fn output_range(&self) -> (Float, Float) {
        (0.0, 1.0)
    }

    fn higher_is_better(&self) -> bool {
        true
    }

    fn properties(&self) -> MetricProperties {
        MetricProperties {
            deterministic: true,
            requires_predictions: false,
            requires_labels: false,
            requires_features: false,
            supports_batch: true,
            parallelizable: true,
            supports_weights: false,
            min_samples: Some(2),
            max_samples: None,
            complexity: ComputationalComplexity::Quadratic,
        }
    }

    fn compute(&self, input: &MetricInput) -> SklResult<MetricOutput> {
        let start_time = std::time::Instant::now();

        // Extract explanation matrix
        let explanation_matrix = match &input.explanations {
            ExplanationData::LocalExplanations(matrix) => matrix.clone(),
            ExplanationData::ShapValues(matrix) => matrix.clone(),
            ExplanationData::LimeCoefficients(matrix) => matrix.clone(),
            _ => {
                return Err(crate::SklearsError::InvalidInput(
                    "Unsupported explanation type for stability metric".to_string(),
                ));
            }
        };

        if explanation_matrix.nrows() < 2 {
            return Err(crate::SklearsError::InvalidInput(
                "At least 2 instances required for stability computation".to_string(),
            ));
        }

        // Compute pairwise correlations
        let mut correlations = Vec::new();
        for i in 0..explanation_matrix.nrows() {
            for j in (i + 1)..explanation_matrix.nrows() {
                let exp_i = explanation_matrix.row(i);
                let exp_j = explanation_matrix.row(j);
                let corr = compute_correlation(&exp_i, &exp_j)?;
                if !corr.is_nan() {
                    correlations.push(corr);
                }
            }
        }

        let stability = if !correlations.is_empty() {
            correlations.iter().sum::<Float>() / correlations.len() as Float
        } else {
            0.0
        };

        let computation_time = start_time.elapsed().as_millis() as u64;

        Ok(MetricOutput {
            value: stability,
            per_sample_values: None,
            per_feature_values: None,
            confidence_interval: None,
            p_value: None,
            metadata: ComputationMetadata {
                computation_time_ms: computation_time,
                memory_usage_bytes: 0,
                samples_processed: explanation_matrix.nrows(),
                features_processed: explanation_matrix.ncols(),
                warnings: Vec::new(),
                timestamp: chrono::Utc::now(),
            },
            statistics: HashMap::new(),
        })
    }

    fn validate_input(&self, input: &MetricInput) -> SklResult<()> {
        match &input.explanations {
            ExplanationData::LocalExplanations(_)
            | ExplanationData::ShapValues(_)
            | ExplanationData::LimeCoefficients(_) => Ok(()),
            _ => Err(crate::SklearsError::InvalidInput(
                "Unsupported explanation type".to_string(),
            )),
        }
    }
}

impl Default for StabilityMetric {
    fn default() -> Self {
        Self::new()
    }
}

/// Built-in completeness metric
#[derive(Debug)]
pub struct CompletenessMetric {
    id: String,
    name: String,
    description: String,
    version: String,
    author: String,
}

impl CompletenessMetric {
    /// Create a new completeness metric
    pub fn new() -> Self {
        Self {
            id: "feature_completeness".to_string(),
            name: "Feature Completeness".to_string(),
            description: "Measures what fraction of features receive non-zero importance"
                .to_string(),
            version: "1.0.0".to_string(),
            author: "Sklears Team".to_string(),
        }
    }
}

impl ExplanationMetric for CompletenessMetric {
    fn metric_id(&self) -> &str {
        &self.id
    }

    fn metric_name(&self) -> &str {
        &self.name
    }

    fn metric_description(&self) -> &str {
        &self.description
    }

    fn metric_version(&self) -> &str {
        &self.version
    }

    fn metric_author(&self) -> &str {
        &self.author
    }

    fn metric_category(&self) -> MetricCategory {
        MetricCategory::Completeness
    }

    fn supported_explanation_types(&self) -> Vec<ExplanationType> {
        vec![
            ExplanationType::FeatureImportance,
            ExplanationType::LocalExplanation,
            ExplanationType::GlobalExplanation,
        ]
    }

    fn output_range(&self) -> (Float, Float) {
        (0.0, 1.0)
    }

    fn higher_is_better(&self) -> bool {
        true
    }

    fn properties(&self) -> MetricProperties {
        MetricProperties {
            deterministic: true,
            requires_predictions: false,
            requires_labels: false,
            requires_features: false,
            supports_batch: true,
            parallelizable: true,
            supports_weights: false,
            min_samples: Some(1),
            max_samples: None,
            complexity: ComputationalComplexity::Linear,
        }
    }

    fn compute(&self, input: &MetricInput) -> SklResult<MetricOutput> {
        let start_time = std::time::Instant::now();

        // Extract explanation values
        let explanation_values = match &input.explanations {
            ExplanationData::FeatureImportance(values) => values.clone(),
            ExplanationData::LocalExplanations(matrix) => {
                // Use mean across instances
                matrix.mean_axis(Axis(0)).unwrap()
            }
            ExplanationData::GlobalExplanation(values) => values.clone(),
            _ => {
                return Err(crate::SklearsError::InvalidInput(
                    "Unsupported explanation type for completeness metric".to_string(),
                ));
            }
        };

        // Count non-zero features
        let non_zero_count = explanation_values
            .iter()
            .filter(|&&x| x.abs() > 1e-10)
            .count();

        let completeness = non_zero_count as Float / explanation_values.len() as Float;

        let computation_time = start_time.elapsed().as_millis() as u64;

        Ok(MetricOutput {
            value: completeness,
            per_sample_values: None,
            per_feature_values: None,
            confidence_interval: None,
            p_value: None,
            metadata: ComputationMetadata {
                computation_time_ms: computation_time,
                memory_usage_bytes: 0,
                samples_processed: 1,
                features_processed: explanation_values.len(),
                warnings: Vec::new(),
                timestamp: chrono::Utc::now(),
            },
            statistics: HashMap::new(),
        })
    }

    fn validate_input(&self, input: &MetricInput) -> SklResult<()> {
        match &input.explanations {
            ExplanationData::FeatureImportance(_)
            | ExplanationData::LocalExplanations(_)
            | ExplanationData::GlobalExplanation(_) => Ok(()),
            _ => Err(crate::SklearsError::InvalidInput(
                "Unsupported explanation type".to_string(),
            )),
        }
    }
}

impl Default for CompletenessMetric {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility function to compute correlation between two arrays
fn compute_correlation(x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> SklResult<Float> {
    if x.len() != y.len() {
        return Err(crate::SklearsError::InvalidInput(
            "Arrays must have the same length".to_string(),
        ));
    }

    if x.len() < 2 {
        return Err(crate::SklearsError::InvalidInput(
            "At least 2 values required for correlation".to_string(),
        ));
    }

    let n = x.len() as Float;
    let mean_x = x.sum() / n;
    let mean_y = y.sum() / n;

    let mut numerator = 0.0;
    let mut sum_sq_x = 0.0;
    let mut sum_sq_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        numerator += dx * dy;
        sum_sq_x += dx * dx;
        sum_sq_y += dy * dy;
    }

    let denominator = (sum_sq_x * sum_sq_y).sqrt();

    if denominator < 1e-10 {
        Ok(0.0) // No variance in one or both variables
    } else {
        Ok(numerator / denominator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::{array, Array2, ArrayView1};

    #[test]
    fn test_metric_registry() {
        let registry = MetricRegistry::new();

        // Register metrics
        let fidelity_metric = FidelityMetric::new();
        let result = registry.register_metric(fidelity_metric);
        assert!(result.is_ok());

        let stability_metric = StabilityMetric::new();
        let result = registry.register_metric(stability_metric);
        assert!(result.is_ok());

        // Check metrics are registered
        let metrics = registry.list_metrics();
        assert!(metrics.contains(&"fidelity_correlation".to_string()));
        assert!(metrics.contains(&"explanation_stability".to_string()));

        // Get metric metadata
        let metadata = registry.get_metric_metadata("fidelity_correlation");
        assert!(metadata.is_some());
        let metadata = metadata.unwrap();
        assert_eq!(metadata.category, MetricCategory::Fidelity);
    }

    #[test]
    fn test_fidelity_metric() {
        let metric = FidelityMetric::new();

        // Create test data
        let features =
            Array2::from_shape_vec((10, 3), (0..30).map(|x| x as Float).collect()).unwrap();
        let explanations = array![0.5, 0.3, 0.2];

        let input = MetricInput {
            explanations: ExplanationData::FeatureImportance(explanations),
            reference_explanations: None,
            predictions: None,
            labels: None,
            features: Some(features),
            feature_names: None,
            sample_weights: None,
            metadata: HashMap::new(),
        };

        // Compute metric
        let result = metric.compute(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert!(output.value >= -1.0 && output.value <= 1.0);
        assert!(output.metadata.features_processed == 3);
    }

    #[test]
    fn test_stability_metric() {
        let metric = StabilityMetric::new();

        // Create test data (local explanations)
        let explanations = Array2::from_shape_vec(
            (5, 3),
            vec![
                0.1, 0.2, 0.3, 0.15, 0.25, 0.35, 0.12, 0.22, 0.32, 0.11, 0.21, 0.31, 0.13, 0.23,
                0.33,
            ],
        )
        .unwrap();

        let input = MetricInput {
            explanations: ExplanationData::LocalExplanations(explanations),
            reference_explanations: None,
            predictions: None,
            labels: None,
            features: None,
            feature_names: None,
            sample_weights: None,
            metadata: HashMap::new(),
        };

        // Compute metric
        let result = metric.compute(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert!(output.value >= 0.0 && output.value <= 1.0);
    }

    #[test]
    fn test_completeness_metric() {
        let metric = CompletenessMetric::new();

        // Create test data with some zero values
        let explanations = array![0.5, 0.0, 0.3, 0.0, 0.2];

        let input = MetricInput {
            explanations: ExplanationData::FeatureImportance(explanations),
            reference_explanations: None,
            predictions: None,
            labels: None,
            features: None,
            feature_names: None,
            sample_weights: None,
            metadata: HashMap::new(),
        };

        // Compute metric
        let result = metric.compute(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.value, 0.6); // 3 out of 5 features are non-zero
    }

    #[test]
    fn test_metric_registry_by_category() {
        let registry = MetricRegistry::with_default_metrics();

        // List by category
        let fidelity_metrics = registry.list_metrics_by_category(MetricCategory::Fidelity);
        assert!(fidelity_metrics.contains(&"fidelity_correlation".to_string()));

        let stability_metrics = registry.list_metrics_by_category(MetricCategory::Stability);
        assert!(stability_metrics.contains(&"explanation_stability".to_string()));
    }

    #[test]
    fn test_metric_registry_by_explanation_type() {
        let registry = MetricRegistry::with_default_metrics();

        // List by explanation type
        let feature_importance_metrics =
            registry.list_metrics_by_explanation_type(ExplanationType::FeatureImportance);
        assert!(feature_importance_metrics.contains(&"fidelity_correlation".to_string()));
        assert!(feature_importance_metrics.contains(&"feature_completeness".to_string()));

        let local_explanation_metrics =
            registry.list_metrics_by_explanation_type(ExplanationType::LocalExplanation);
        assert!(local_explanation_metrics.contains(&"explanation_stability".to_string()));
    }

    #[test]
    fn test_compute_correlation() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let corr = compute_correlation(&x.view(), &y.view()).unwrap();
        assert!((corr - 1.0).abs() < 1e-10); // Perfect positive correlation

        let z = array![5.0, 4.0, 3.0, 2.0, 1.0];
        let corr = compute_correlation(&x.view(), &z.view()).unwrap();
        assert!((corr - (-1.0)).abs() < 1e-10); // Perfect negative correlation
    }

    #[test]
    fn test_metric_registry_compute_multiple() {
        let registry = MetricRegistry::with_default_metrics();

        // Create test data
        let features =
            Array2::from_shape_vec((10, 3), (0..30).map(|x| x as Float).collect()).unwrap();
        let explanations = array![0.5, 0.3, 0.2];

        let input = MetricInput {
            explanations: ExplanationData::FeatureImportance(explanations),
            reference_explanations: None,
            predictions: None,
            labels: None,
            features: Some(features),
            feature_names: None,
            sample_weights: None,
            metadata: HashMap::new(),
        };

        // Compute multiple metrics
        let metric_ids = vec![
            "fidelity_correlation".to_string(),
            "feature_completeness".to_string(),
        ];

        let results = registry.compute_multiple_metrics(&metric_ids, &input);
        assert_eq!(results.len(), 2);
        assert!(results.get("fidelity_correlation").unwrap().is_ok());
        assert!(results.get("feature_completeness").unwrap().is_ok());
    }

    #[test]
    fn test_metric_properties() {
        let metric = FidelityMetric::new();
        let properties = metric.properties();

        assert!(properties.deterministic);
        assert!(properties.requires_predictions);
        assert!(properties.requires_features);
        assert!(properties.supports_batch);
        assert_eq!(properties.complexity, ComputationalComplexity::Linear);
    }
}
