//! Integration Utilities for Baseline Estimators
//!
//! This module provides utilities for automatic baseline generation,
//! pipeline integration, and smart default selection.
//!
//! The module includes:
//! - [`AutoBaselineGenerator`] - Automatic baseline generation based on data characteristics
//! - [`BaselinePipeline`] - Integration with preprocessing and evaluation pipelines
//! - [`SmartDefaultSelector`] - Intelligent default strategy selection
//! - [`ConfigurationHelper`] - Configuration assistance for baseline methods
//! - [`BaselineRecommendationEngine`] - Advanced recommendation system for baseline selection

use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use sklears_core::error::SklearsError;
use std::collections::HashMap;

use crate::{
    dummy_classifier::Strategy as ClassifierStrategy,
    dummy_regressor::Strategy as RegressorStrategy, CausalDiscoveryStrategy, ContextAwareStrategy,
    EnsembleStrategy, FairnessStrategy, FewShotStrategy, RobustStrategy,
};

/// Data characteristics for automatic baseline selection
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DataCharacteristics {
    /// n_samples
    pub n_samples: usize,
    /// n_features
    pub n_features: usize,
    /// n_classes
    pub n_classes: Option<usize>,
    /// class_balance
    pub class_balance: Option<f64>,
    /// feature_sparsity
    pub feature_sparsity: f64,
    /// missing_data_ratio
    pub missing_data_ratio: f64,
    /// outlier_ratio
    pub outlier_ratio: f64,
    /// noise_level
    pub noise_level: f64,
    /// correlation_strength
    pub correlation_strength: f64,
    /// temporal_dependency
    pub temporal_dependency: bool,
    /// categorical_features_ratio
    pub categorical_features_ratio: f64,
    /// high_dimensional
    pub high_dimensional: bool,
    /// imbalanced
    pub imbalanced: bool,
    /// has_protected_attributes
    pub has_protected_attributes: bool,
}

/// Recommended baseline configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BaselineRecommendation {
    /// primary_strategy
    pub primary_strategy: BaselineType,
    /// fallback_strategies
    pub fallback_strategies: Vec<BaselineType>,
    /// ensemble_recommended
    pub ensemble_recommended: bool,
    /// preprocessing_needed
    pub preprocessing_needed: bool,
    /// robustness_needed
    pub robustness_needed: bool,
    /// fairness_considerations
    pub fairness_considerations: bool,
    /// confidence_score
    pub confidence_score: f64,
    /// reasoning
    pub reasoning: String,
}

/// Types of baseline estimators
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum BaselineType {
    /// Standard dummy classifier
    DummyClassifier(ClassifierStrategy),
    /// Standard dummy regressor
    DummyRegressor(RegressorStrategy),
    /// Ensemble baseline
    EnsembleBaseline(EnsembleStrategy),
    /// Robust baseline
    RobustBaseline(RobustStrategy),
    /// Context-aware baseline
    ContextAwareBaseline(ContextAwareStrategy),
    /// Fairness-aware baseline
    FairnessBaseline(FairnessStrategy),
    /// Few-shot baseline
    FewShotBaseline(FewShotStrategy),
    /// Causal baseline
    CausalBaseline(CausalDiscoveryStrategy),
}

/// Pipeline configuration for baseline integration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PipelineConfig {
    /// preprocessing_steps
    pub preprocessing_steps: Vec<PreprocessingStep>,
    /// baseline_config
    pub baseline_config: BaselineType,
    /// evaluation_metrics
    pub evaluation_metrics: Vec<String>,
    /// validation_strategy
    pub validation_strategy: ValidationStrategy,
    /// output_format
    pub output_format: OutputFormat,
}

/// Preprocessing steps for baseline pipelines
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum PreprocessingStep {
    /// Feature scaling/normalization
    FeatureScaling { method: String },
    /// Missing data imputation
    MissingDataImputation { strategy: String },
    /// Outlier detection and removal
    OutlierHandling { method: String, threshold: f64 },
    /// Feature selection
    FeatureSelection { method: String, n_features: usize },
    /// Dimensionality reduction
    DimensionalityReduction { method: String, n_components: usize },
}

/// Validation strategies for baseline evaluation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ValidationStrategy {
    /// Hold-out validation
    HoldOut { test_size: f64 },
    /// K-fold cross-validation
    KFold { k: usize },
    /// Time series split
    TimeSeriesSplit { n_splits: usize },
    /// Stratified validation
    Stratified { n_splits: usize },
    /// Bootstrap validation
    Bootstrap { n_samples: usize },
}

/// Output formats for baseline results
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum OutputFormat {
    /// Simple predictions array
    Predictions,
    /// Predictions with confidence intervals
    PredictionsWithConfidence,
    /// Full performance report
    PerformanceReport,
    /// Comparative analysis
    ComparativeAnalysis,
}

/// Automatic baseline generator
#[derive(Debug, Clone)]
pub struct AutoBaselineGenerator {
    recommendation_engine: BaselineRecommendationEngine,
    configuration_helper: ConfigurationHelper,
    random_state: Option<u64>,
}

/// Baseline pipeline for integration
#[derive(Debug)]
pub struct BaselinePipeline {
    config: PipelineConfig,
    fitted_baseline: Option<Box<dyn BaselineEstimator>>,
    preprocessing_fitted: bool,
    random_state: Option<u64>,
}

/// Smart default selector
#[derive(Debug, Clone)]
pub struct SmartDefaultSelector {
    selection_criteria: Vec<SelectionCriterion>,
    fallback_strategy: BaselineType,
    random_state: Option<u64>,
}

/// Configuration helper for baselines
#[derive(Debug, Clone)]
pub struct ConfigurationHelper {
    parameter_defaults: HashMap<String, ParameterDefault>,
    optimization_hints: Vec<OptimizationHint>,
}

/// Baseline recommendation engine
#[derive(Debug, Clone)]
pub struct BaselineRecommendationEngine {
    recommendation_rules: Vec<RecommendationRule>,
    performance_history: HashMap<String, PerformanceMetrics>,
    adaptation_enabled: bool,
}

/// Selection criteria for baseline choice
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SelectionCriterion {
    /// Data size criterion
    DataSize {
        min_samples: usize,
        max_samples: usize,
    },
    /// Feature dimensionality criterion
    FeatureDimensionality {
        min_features: usize,
        max_features: usize,
    },
    /// Task type criterion
    TaskType {
        classification: bool,
        regression: bool,
    },
    /// Performance requirement criterion
    PerformanceRequirement { min_accuracy: f64, max_time: f64 },
    /// Robustness requirement criterion
    RobustnessRequirement { outlier_tolerance: f64 },
}

/// Parameter defaults for baseline configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ParameterDefault {
    /// parameter_name
    pub parameter_name: String,
    /// default_value
    pub default_value: f64,
    /// valid_range
    pub valid_range: (f64, f64),
    /// description
    pub description: String,
}

/// Optimization hints for baseline tuning
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OptimizationHint {
    /// context
    pub context: String,
    /// suggestion
    pub suggestion: String,
    /// impact
    pub impact: String,
    /// priority
    pub priority: u8,
}

/// Recommendation rules for baseline selection
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RecommendationRule {
    /// condition
    pub condition: String,
    /// recommended_baseline
    pub recommended_baseline: BaselineType,
    /// confidence
    pub confidence: f64,
    /// reasoning
    pub reasoning: String,
}

/// Performance metrics for recommendation engine
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PerformanceMetrics {
    /// accuracy
    pub accuracy: f64,
    /// precision
    pub precision: f64,
    /// recall
    pub recall: f64,
    /// f1_score
    pub f1_score: f64,
    /// execution_time
    pub execution_time: f64,
    /// memory_usage
    pub memory_usage: f64,
}

/// Trait for baseline estimators in pipeline
pub trait BaselineEstimator: std::fmt::Debug {
    fn fit_baseline(&mut self, x: &Array2<f64>, y: &Array1<i32>) -> Result<(), SklearsError>;
    fn predict_baseline(&self, x: &Array2<f64>) -> Result<Array1<i32>, SklearsError>;
    fn get_type(&self) -> BaselineType;
}

impl AutoBaselineGenerator {
    /// Create a new automatic baseline generator
    pub fn new() -> Self {
        Self {
            recommendation_engine: BaselineRecommendationEngine::new(),
            configuration_helper: ConfigurationHelper::new(),
            random_state: None,
        }
    }

    /// Set the random state for reproducible results
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Analyze data and generate baseline recommendations
    pub fn analyze_and_recommend(
        &self,
        x: &Array2<f64>,
        y: &Array1<i32>,
    ) -> Result<BaselineRecommendation, SklearsError> {
        let characteristics = self.analyze_data_characteristics(x, y);
        let recommendation = self
            .recommendation_engine
            .recommend_baseline(&characteristics);
        Ok(recommendation)
    }

    /// Generate automatic baseline configuration
    pub fn generate_baseline(
        &self,
        x: &Array2<f64>,
        y: &Array1<i32>,
    ) -> Result<BaselineType, SklearsError> {
        let recommendation = self.analyze_and_recommend(x, y)?;
        Ok(recommendation.primary_strategy)
    }

    fn analyze_data_characteristics(
        &self,
        x: &Array2<f64>,
        y: &Array1<i32>,
    ) -> DataCharacteristics {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Analyze class distribution
        let mut class_counts = HashMap::new();
        for &class in y.iter() {
            *class_counts.entry(class).or_insert(0) += 1;
        }
        let n_classes = Some(class_counts.len());

        // Calculate class balance (ratio of smallest to largest class)
        let class_balance = if class_counts.len() > 1 {
            let min_count = *class_counts.values().min().unwrap() as f64;
            let max_count = *class_counts.values().max().unwrap() as f64;
            Some(min_count / max_count)
        } else {
            None
        };

        // Calculate feature sparsity
        let total_elements = (n_samples * n_features) as f64;
        let zero_elements = x.iter().filter(|&&val| val.abs() < 1e-10).count() as f64;
        let feature_sparsity = zero_elements / total_elements;

        // Estimate outlier ratio using IQR method
        let mut outlier_count = 0;
        for col in 0..n_features {
            let column = x.column(col);
            let mut sorted_col = column.to_vec();
            sorted_col.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let q1_idx = sorted_col.len() / 4;
            let q3_idx = 3 * sorted_col.len() / 4;

            if q1_idx < sorted_col.len() && q3_idx < sorted_col.len() {
                let q1 = sorted_col[q1_idx];
                let q3 = sorted_col[q3_idx];
                let iqr = q3 - q1;
                let lower_bound = q1 - 1.5 * iqr;
                let upper_bound = q3 + 1.5 * iqr;

                outlier_count += column
                    .iter()
                    .filter(|&&val| val < lower_bound || val > upper_bound)
                    .count();
            }
        }
        let outlier_ratio = outlier_count as f64 / total_elements;

        // Estimate noise level using standard deviation
        let feature_stds: Vec<f64> = (0..n_features).map(|col| x.column(col).std(0.0)).collect();
        let noise_level = feature_stds.iter().sum::<f64>() / feature_stds.len() as f64;

        // Calculate correlation strength (average absolute correlation)
        let mut correlation_sum = 0.0;
        let mut correlation_count = 0;
        for i in 0..n_features {
            for j in i + 1..n_features {
                let col_i = x.column(i);
                let col_j = x.column(j);
                let correlation = self.compute_correlation(&col_i, &col_j);
                correlation_sum += correlation.abs();
                correlation_count += 1;
            }
        }
        let correlation_strength = if correlation_count > 0 {
            correlation_sum / correlation_count as f64
        } else {
            0.0
        };

        DataCharacteristics {
            n_samples,
            n_features,
            n_classes,
            class_balance,
            feature_sparsity,
            missing_data_ratio: 0.0, // Simplified: assume no missing data in ndarray
            outlier_ratio,
            noise_level,
            correlation_strength,
            temporal_dependency: false, // Simplified: would need domain knowledge
            categorical_features_ratio: 0.0, // Simplified: assume continuous features
            high_dimensional: n_features > 100,
            imbalanced: class_balance.is_some_and(|balance| balance < 0.1),
            has_protected_attributes: false, // Simplified: would need domain knowledge
        }
    }

    fn compute_correlation(&self, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> f64 {
        let n = x.len() as f64;
        let mean_x = x.mean().unwrap();
        let mean_y = y.mean().unwrap();

        let numerator: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
            .sum();

        let var_x: f64 = x.iter().map(|xi| (xi - mean_x).powi(2)).sum();
        let var_y: f64 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum();

        if var_x == 0.0 || var_y == 0.0 {
            0.0
        } else {
            numerator / (var_x * var_y).sqrt()
        }
    }
}

impl Default for AutoBaselineGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl BaselineRecommendationEngine {
    /// Create a new recommendation engine
    pub fn new() -> Self {
        let mut recommendation_rules = Vec::new();

        // Add default recommendation rules
        recommendation_rules.push(RecommendationRule {
            condition: "high_dimensional".to_string(),
            recommended_baseline: BaselineType::DummyClassifier(ClassifierStrategy::MostFrequent),
            confidence: 0.8,
            reasoning: "High-dimensional data benefits from simple baselines".to_string(),
        });

        recommendation_rules.push(RecommendationRule {
            condition: "imbalanced".to_string(),
            recommended_baseline: BaselineType::DummyClassifier(ClassifierStrategy::Stratified),
            confidence: 0.7,
            reasoning: "Imbalanced data requires stratified sampling".to_string(),
        });

        recommendation_rules.push(RecommendationRule {
            condition: "high_outlier_ratio".to_string(),
            recommended_baseline: BaselineType::RobustBaseline(RobustStrategy::TrimmedMean {
                trim_proportion: 0.1,
            }),
            confidence: 0.9,
            reasoning: "High outlier ratio requires robust methods".to_string(),
        });

        Self {
            recommendation_rules,
            performance_history: HashMap::new(),
            adaptation_enabled: true,
        }
    }

    /// Recommend baseline based on data characteristics
    pub fn recommend_baseline(
        &self,
        characteristics: &DataCharacteristics,
    ) -> BaselineRecommendation {
        let mut candidate_recommendations = Vec::new();

        // Apply recommendation rules
        for rule in &self.recommendation_rules {
            let matches = match rule.condition.as_str() {
                "high_dimensional" => characteristics.high_dimensional,
                "imbalanced" => characteristics.imbalanced,
                "high_outlier_ratio" => characteristics.outlier_ratio > 0.1,
                "has_protected_attributes" => characteristics.has_protected_attributes,
                "small_dataset" => characteristics.n_samples < 1000,
                "large_dataset" => characteristics.n_samples > 10000,
                "high_correlation" => characteristics.correlation_strength > 0.7,
                "sparse_features" => characteristics.feature_sparsity > 0.5,
                _ => false,
            };

            if matches {
                candidate_recommendations.push((rule.clone(), rule.confidence));
            }
        }

        // Select best recommendation
        let (primary_rule, confidence_score) = candidate_recommendations
            .into_iter()
            .max_by(|(_, conf_a), (_, conf_b)| conf_a.partial_cmp(conf_b).unwrap())
            .unwrap_or((
                RecommendationRule {
                    condition: "default".to_string(),
                    recommended_baseline: BaselineType::DummyClassifier(
                        ClassifierStrategy::MostFrequent,
                    ),
                    confidence: 0.5,
                    reasoning: "Default baseline when no specific conditions are met".to_string(),
                },
                0.5,
            ));

        // Generate fallback strategies
        let fallback_strategies = vec![
            BaselineType::DummyClassifier(ClassifierStrategy::Uniform),
            BaselineType::EnsembleBaseline(EnsembleStrategy::Average),
        ];

        BaselineRecommendation {
            primary_strategy: primary_rule.recommended_baseline,
            fallback_strategies,
            ensemble_recommended: characteristics.n_samples > 1000,
            preprocessing_needed: characteristics.outlier_ratio > 0.05
                || characteristics.feature_sparsity > 0.3,
            robustness_needed: characteristics.outlier_ratio > 0.1,
            fairness_considerations: characteristics.has_protected_attributes,
            confidence_score,
            reasoning: primary_rule.reasoning,
        }
    }
}

impl Default for BaselineRecommendationEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfigurationHelper {
    /// Create a new configuration helper
    pub fn new() -> Self {
        let mut parameter_defaults = HashMap::new();

        // Add default parameters for different baseline types
        parameter_defaults.insert(
            "trim_proportion".to_string(),
            ParameterDefault {
                parameter_name: "trim_proportion".to_string(),
                default_value: 0.1,
                valid_range: (0.0, 0.5),
                description: "Proportion of extreme values to trim for robust estimation"
                    .to_string(),
            },
        );

        parameter_defaults.insert(
            "ensemble_size".to_string(),
            ParameterDefault {
                parameter_name: "ensemble_size".to_string(),
                default_value: 5.0,
                valid_range: (3.0, 50.0),
                description: "Number of base estimators in ensemble".to_string(),
            },
        );

        let optimization_hints = vec![
            OptimizationHint {
                context: "high_dimensional".to_string(),
                suggestion: "Use feature selection or dimensionality reduction".to_string(),
                impact: "Reduces overfitting and improves computational efficiency".to_string(),
                priority: 8,
            },
            OptimizationHint {
                context: "imbalanced".to_string(),
                suggestion: "Use stratified sampling or class weighting".to_string(),
                impact: "Improves performance on minority classes".to_string(),
                priority: 9,
            },
        ];

        Self {
            parameter_defaults,
            optimization_hints,
        }
    }

    /// Get default parameter configuration
    pub fn get_default_config(&self, baseline_type: &BaselineType) -> HashMap<String, f64> {
        let mut config = HashMap::new();

        match baseline_type {
            BaselineType::RobustBaseline(_) => {
                if let Some(default) = self.parameter_defaults.get("trim_proportion") {
                    config.insert("trim_proportion".to_string(), default.default_value);
                }
            }
            BaselineType::EnsembleBaseline(_) => {
                if let Some(default) = self.parameter_defaults.get("ensemble_size") {
                    config.insert("ensemble_size".to_string(), default.default_value);
                }
            }
            _ => {}
        }

        config
    }

    /// Get optimization hints for given data characteristics
    pub fn get_optimization_hints(
        &self,
        characteristics: &DataCharacteristics,
    ) -> Vec<OptimizationHint> {
        let mut relevant_hints = Vec::new();

        for hint in &self.optimization_hints {
            let relevant = match hint.context.as_str() {
                "high_dimensional" => characteristics.high_dimensional,
                "imbalanced" => characteristics.imbalanced,
                "sparse" => characteristics.feature_sparsity > 0.5,
                "noisy" => characteristics.noise_level > 1.0,
                _ => false,
            };

            if relevant {
                relevant_hints.push(hint.clone());
            }
        }

        // Sort by priority (higher priority first)
        relevant_hints.sort_by(|a, b| b.priority.cmp(&a.priority));

        relevant_hints
    }
}

impl Default for ConfigurationHelper {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_auto_baseline_generator() {
        let x = Array2::from_shape_vec((100, 5), (0..500).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_vec((0..100).map(|i| i % 3).collect());

        let generator = AutoBaselineGenerator::new();
        let recommendation = generator.analyze_and_recommend(&x, &y).unwrap();

        assert!(recommendation.confidence_score > 0.0);
        assert!(!recommendation.reasoning.is_empty());
    }

    #[test]
    fn test_data_characteristics_analysis() {
        let x = Array2::from_shape_vec(
            (50, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
                30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0,
                44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0,
                58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0,
                72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0,
                86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0,
                100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
                112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0,
                124.0, 125.0, 126.0, 127.0, 128.0, 129.0, 130.0, 131.0, 132.0, 133.0, 134.0, 135.0,
                136.0, 137.0, 138.0, 139.0, 140.0, 141.0, 142.0, 143.0, 144.0, 145.0, 146.0, 147.0,
                148.0, 149.0, 150.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec((0..50).map(|i| i % 2).collect());

        let generator = AutoBaselineGenerator::new();
        let characteristics = generator.analyze_data_characteristics(&x, &y);

        assert_eq!(characteristics.n_samples, 50);
        assert_eq!(characteristics.n_features, 3);
        assert_eq!(characteristics.n_classes, Some(2));
        assert!(characteristics.class_balance.is_some());
    }

    #[test]
    fn test_recommendation_engine() {
        let characteristics = DataCharacteristics {
            n_samples: 1000,
            n_features: 50,
            n_classes: Some(3),
            class_balance: Some(0.8),
            feature_sparsity: 0.1,
            missing_data_ratio: 0.0,
            outlier_ratio: 0.15,
            noise_level: 0.5,
            correlation_strength: 0.3,
            temporal_dependency: false,
            categorical_features_ratio: 0.0,
            high_dimensional: false,
            imbalanced: false,
            has_protected_attributes: false,
        };

        let engine = BaselineRecommendationEngine::new();
        let recommendation = engine.recommend_baseline(&characteristics);

        assert!(recommendation.confidence_score > 0.0);
        assert!(recommendation.robustness_needed); // Due to high outlier ratio
    }

    #[test]
    fn test_configuration_helper() {
        let helper = ConfigurationHelper::new();
        let baseline_type = BaselineType::RobustBaseline(RobustStrategy::TrimmedMean {
            trim_proportion: 0.1,
        });

        let config = helper.get_default_config(&baseline_type);
        assert!(config.contains_key("trim_proportion"));

        let characteristics = DataCharacteristics {
            n_samples: 1000,
            n_features: 200, // High dimensional
            n_classes: Some(2),
            class_balance: Some(0.1), // Imbalanced
            feature_sparsity: 0.0,
            missing_data_ratio: 0.0,
            outlier_ratio: 0.05,
            noise_level: 0.5,
            correlation_strength: 0.3,
            temporal_dependency: false,
            categorical_features_ratio: 0.0,
            high_dimensional: true,
            imbalanced: true,
            has_protected_attributes: false,
        };

        let hints = helper.get_optimization_hints(&characteristics);
        assert!(!hints.is_empty());
        assert!(hints.iter().any(|hint| hint.context == "high_dimensional"));
        assert!(hints.iter().any(|hint| hint.context == "imbalanced"));
    }
}
