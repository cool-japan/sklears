//! Common Types and Enumerations for Ensemble Learning
//!
//! This module provides the foundational types, traits, and enumerations used
//! throughout the ensemble learning framework. These types define the common
//! interfaces and data structures that enable interoperability between different
//! ensemble methods and strategies.
//!
//! # Core Types
//!
//! ## EnsembleMethod
//! Defines the different ensemble learning approaches available in the framework.
//! Each method has its own characteristics and use cases.
//!
//! ## CombiningRule
//! Specifies how individual model predictions are combined to produce the final
//! ensemble prediction. Different combining rules are suitable for different
//! scenarios and data distributions.
//!
//! ## WeightingScheme
//! Determines how weights are assigned to individual models within the ensemble.
//! Proper weighting is crucial for optimal ensemble performance.
//!
//! ## ValidationStrategy
//! Defines the validation approaches used for ensemble evaluation and
//! hyperparameter tuning.
//!
//! # Usage Examples
//!
//! ```rust,ignore
//! use sklears_compose::ensemble_types::{
//!     EnsembleMethod, CombiningRule, WeightingScheme, ValidationStrategy
//! };
//!
//! // Define ensemble configuration
//! let ensemble_config = EnsembleConfiguration {
//!     method: EnsembleMethod::Voting,
//!     combining_rule: CombiningRule::Majority,
//!     weighting_scheme: WeightingScheme::Uniform,
//!     validation: ValidationStrategy::CrossValidation { folds: 5 },
//! };
//!
//! // Create ensemble metrics
//! let metrics = EnsembleMetrics::new()
//!     .with_accuracy_metric()
//!     .with_diversity_metric()
//!     .with_efficiency_metric();
//! ```

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::{
    error::Result as SklResult,
    types::{Float, FloatBounds},
};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Ensemble learning methods available in the framework
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EnsembleMethod {
    /// Voting-based ensemble (hard/soft voting)
    Voting,
    /// Bagging ensemble (Bootstrap Aggregating)
    Bagging,
    /// Boosting ensemble (AdaBoost, Gradient Boosting, etc.)
    Boosting,
    /// Stacking ensemble (Stacked Generalization)
    Stacking,
    /// Random Forest ensemble
    RandomForest,
    /// Dynamic selection ensemble
    DynamicSelection,
    /// Model fusion ensemble
    ModelFusion,
    /// Hierarchical composition
    HierarchicalComposition,
    /// Bayesian ensemble
    BayesianEnsemble,
    /// Neural ensemble
    NeuralEnsemble,
    /// Evolutionary ensemble
    EvolutionaryEnsemble,
    /// Custom ensemble method
    Custom { name: String, description: String },
}

impl EnsembleMethod {
    /// Get the method name as string
    pub fn name(&self) -> &str {
        match self {
            EnsembleMethod::Voting => "voting",
            EnsembleMethod::Bagging => "bagging",
            EnsembleMethod::Boosting => "boosting",
            EnsembleMethod::Stacking => "stacking",
            EnsembleMethod::RandomForest => "random_forest",
            EnsembleMethod::DynamicSelection => "dynamic_selection",
            EnsembleMethod::ModelFusion => "model_fusion",
            EnsembleMethod::HierarchicalComposition => "hierarchical_composition",
            EnsembleMethod::BayesianEnsemble => "bayesian_ensemble",
            EnsembleMethod::NeuralEnsemble => "neural_ensemble",
            EnsembleMethod::EvolutionaryEnsemble => "evolutionary_ensemble",
            EnsembleMethod::Custom { name, .. } => name,
        }
    }

    /// Get method description
    pub fn description(&self) -> &str {
        match self {
            EnsembleMethod::Voting => "Democratic voting between multiple models",
            EnsembleMethod::Bagging => "Bootstrap aggregating with random sampling",
            EnsembleMethod::Boosting => "Sequential learning with adaptive weighting",
            EnsembleMethod::Stacking => "Multi-level learning with meta-models",
            EnsembleMethod::RandomForest => "Random decision tree ensemble",
            EnsembleMethod::DynamicSelection => "Adaptive model selection based on context",
            EnsembleMethod::ModelFusion => "Advanced fusion of model predictions",
            EnsembleMethod::HierarchicalComposition => "Multi-level hierarchical ensemble",
            EnsembleMethod::BayesianEnsemble => "Bayesian model averaging",
            EnsembleMethod::NeuralEnsemble => "Neural network-based ensemble",
            EnsembleMethod::EvolutionaryEnsemble => "Evolutionary optimization of ensemble",
            EnsembleMethod::Custom { description, .. } => description,
        }
    }

    /// Check if method supports probability predictions
    pub fn supports_probabilities(&self) -> bool {
        match self {
            EnsembleMethod::Voting |
            EnsembleMethod::Bagging |
            EnsembleMethod::RandomForest |
            EnsembleMethod::BayesianEnsemble => true,
            EnsembleMethod::Boosting |
            EnsembleMethod::Stacking |
            EnsembleMethod::DynamicSelection |
            EnsembleMethod::ModelFusion |
            EnsembleMethod::HierarchicalComposition |
            EnsembleMethod::NeuralEnsemble |
            EnsembleMethod::EvolutionaryEnsemble => true,
            EnsembleMethod::Custom { .. } => false, // Conservative default
        }
    }
}

/// Rules for combining individual model predictions
#[derive(Debug, Clone, PartialEq)]
pub enum CombiningRule {
    /// Majority vote (hard voting)
    Majority,
    /// Plurality vote (most votes wins)
    Plurality,
    /// Weighted majority vote
    WeightedMajority { weights: Vec<Float> },
    /// Average of predictions
    Average,
    /// Weighted average of predictions
    WeightedAverage { weights: Vec<Float> },
    /// Median of predictions
    Median,
    /// Trimmed mean (exclude outliers)
    TrimmedMean { trim_ratio: Float },
    /// Maximum prediction
    Maximum,
    /// Minimum prediction
    Minimum,
    /// Geometric mean
    GeometricMean,
    /// Harmonic mean
    HarmonicMean,
    /// Linear combination with learned coefficients
    LinearCombination { coefficients: Array1<Float> },
    /// Rank-based combination
    RankBased,
    /// Bayesian model averaging
    BayesianAveraging,
    /// Neural network combination
    NeuralCombination,
    /// Custom combining rule
    Custom {
        name: String,
        combiner: fn(&[Array1<Float>]) -> SklResult<Array1<Float>>,
    },
}

impl CombiningRule {
    /// Apply the combining rule to a set of predictions
    pub fn combine(&self, predictions: &[Array1<Float>]) -> SklResult<Array1<Float>> {
        if predictions.is_empty() {
            return Err(sklears_core::prelude::SklearsError::InvalidInput(
                "No predictions to combine".to_string()
            ));
        }

        let n_samples = predictions[0].len();
        let mut result = Array1::zeros(n_samples);

        match self {
            CombiningRule::Average => {
                for i in 0..n_samples {
                    let sum: Float = predictions.iter().map(|pred| pred[i]).sum();
                    result[i] = sum / predictions.len() as Float;
                }
            },
            CombiningRule::WeightedAverage { weights } => {
                if weights.len() != predictions.len() {
                    return Err(sklears_core::prelude::SklearsError::InvalidInput(
                        "Number of weights must match number of predictions".to_string()
                    ));
                }
                let weight_sum: Float = weights.iter().sum();
                for i in 0..n_samples {
                    let weighted_sum: Float = predictions.iter()
                        .zip(weights.iter())
                        .map(|(pred, &weight)| pred[i] * weight)
                        .sum();
                    result[i] = weighted_sum / weight_sum;
                }
            },
            CombiningRule::Median => {
                for i in 0..n_samples {
                    let mut values: Vec<Float> = predictions.iter().map(|pred| pred[i]).collect();
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let mid = values.len() / 2;
                    result[i] = if values.len() % 2 == 0 {
                        (values[mid - 1] + values[mid]) / 2.0
                    } else {
                        values[mid]
                    };
                }
            },
            CombiningRule::Maximum => {
                for i in 0..n_samples {
                    result[i] = predictions.iter()
                        .map(|pred| pred[i])
                        .fold(Float::NEG_INFINITY, |a, b| a.max(b));
                }
            },
            CombiningRule::Minimum => {
                for i in 0..n_samples {
                    result[i] = predictions.iter()
                        .map(|pred| pred[i])
                        .fold(Float::INFINITY, |a, b| a.min(b));
                }
            },
            CombiningRule::GeometricMean => {
                for i in 0..n_samples {
                    let product: Float = predictions.iter()
                        .map(|pred| pred[i].abs())
                        .product();
                    result[i] = product.powf(1.0 / predictions.len() as Float);
                }
            },
            CombiningRule::TrimmedMean { trim_ratio } => {
                let trim_count = (predictions.len() as Float * trim_ratio) as usize;
                for i in 0..n_samples {
                    let mut values: Vec<Float> = predictions.iter().map(|pred| pred[i]).collect();
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let start = trim_count;
                    let end = values.len() - trim_count;
                    if end > start {
                        let sum: Float = values[start..end].iter().sum();
                        result[i] = sum / (end - start) as Float;
                    } else {
                        result[i] = values[values.len() / 2]; // Fallback to median
                    }
                }
            },
            CombiningRule::Custom { combiner, .. } => {
                return combiner(predictions);
            },
            _ => {
                // Fallback to average for unimplemented rules
                for i in 0..n_samples {
                    let sum: Float = predictions.iter().map(|pred| pred[i]).sum();
                    result[i] = sum / predictions.len() as Float;
                }
            }
        }

        Ok(result)
    }

    /// Get the name of the combining rule
    pub fn name(&self) -> &str {
        match self {
            CombiningRule::Majority => "majority",
            CombiningRule::Plurality => "plurality",
            CombiningRule::WeightedMajority { .. } => "weighted_majority",
            CombiningRule::Average => "average",
            CombiningRule::WeightedAverage { .. } => "weighted_average",
            CombiningRule::Median => "median",
            CombiningRule::TrimmedMean { .. } => "trimmed_mean",
            CombiningRule::Maximum => "maximum",
            CombiningRule::Minimum => "minimum",
            CombiningRule::GeometricMean => "geometric_mean",
            CombiningRule::HarmonicMean => "harmonic_mean",
            CombiningRule::LinearCombination { .. } => "linear_combination",
            CombiningRule::RankBased => "rank_based",
            CombiningRule::BayesianAveraging => "bayesian_averaging",
            CombiningRule::NeuralCombination => "neural_combination",
            CombiningRule::Custom { name, .. } => name,
        }
    }
}

/// Weighting schemes for ensemble members
#[derive(Debug, Clone, PartialEq)]
pub enum WeightingScheme {
    /// Uniform weights (all models equal)
    Uniform,
    /// Performance-based weights
    PerformanceBased { metric: PerformanceMetric },
    /// Inverse error weights
    InverseError,
    /// Diversity-based weights
    DiversityBased { diversity_measure: DiversityMeasure },
    /// Age-based weights (for temporal ensembles)
    AgeBased { decay_factor: Float },
    /// Confidence-based weights
    ConfidenceBased,
    /// Cross-validation based weights
    CrossValidationBased { folds: usize },
    /// Bayesian weights with prior
    BayesianWeights { prior_alpha: Float, prior_beta: Float },
    /// Learned weights through optimization
    LearnedWeights { optimization_method: OptimizationMethod },
    /// Dynamic weights that change per prediction
    DynamicWeights,
    /// Custom weighting scheme
    Custom { name: String },
}

/// Performance metrics for weight calculation
#[derive(Debug, Clone, PartialEq)]
pub enum PerformanceMetric {
    /// Accuracy
    Accuracy,
    /// Mean Squared Error
    MeanSquaredError,
    /// Mean Absolute Error
    MeanAbsoluteError,
    /// F1 Score
    F1Score,
    /// Area Under ROC Curve
    AucRoc,
    /// Area Under Precision-Recall Curve
    AucPr,
    /// Log Loss
    LogLoss,
    /// R² Score
    R2Score,
    /// Custom metric
    Custom { name: String },
}

/// Diversity measures for ensemble evaluation
#[derive(Debug, Clone, PartialEq)]
pub enum DiversityMeasure {
    /// Disagreement measure
    Disagreement,
    /// Q-statistic
    QStatistic,
    /// Correlation coefficient
    CorrelationCoefficient,
    /// Yule's Q statistic
    YuleQ,
    /// Double-fault measure
    DoubleFault,
    /// Kohavi-Wolpert variance
    KohaviWolpertVariance,
    /// Interrater agreement
    InterraterAgreement,
    /// Entropy-based diversity
    EntropyDiversity,
    /// Custom diversity measure
    Custom { name: String },
}

/// Optimization methods for learning weights
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationMethod {
    /// Gradient descent
    GradientDescent { learning_rate: Float },
    /// Quasi-Newton methods
    QuasiNewton,
    /// Genetic algorithm
    GeneticAlgorithm { population_size: usize },
    /// Simulated annealing
    SimulatedAnnealing { initial_temperature: Float },
    /// Particle swarm optimization
    ParticleSwarm { swarm_size: usize },
    /// Bayesian optimization
    BayesianOptimization,
    /// Grid search
    GridSearch,
    /// Random search
    RandomSearch { n_iterations: usize },
    /// Custom optimization method
    Custom { name: String },
}

/// Prediction modes for ensemble systems
#[derive(Debug, Clone, PartialEq)]
pub enum PredictionMode {
    /// Standard prediction
    Standard,
    /// Prediction with confidence intervals
    WithConfidence,
    /// Prediction with full distribution
    WithDistribution,
    /// Prediction with explanation
    WithExplanation,
    /// Batch prediction for efficiency
    Batch,
    /// Streaming prediction for online learning
    Streaming,
    /// Probabilistic prediction
    Probabilistic,
    /// Multi-output prediction
    MultiOutput,
    /// Hierarchical prediction
    Hierarchical,
    /// Custom prediction mode
    Custom { name: String },
}

/// Validation strategies for ensemble evaluation
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationStrategy {
    /// Hold-out validation
    HoldOut { test_size: Float },
    /// K-fold cross-validation
    CrossValidation { folds: usize },
    /// Stratified K-fold cross-validation
    StratifiedCrossValidation { folds: usize },
    /// Leave-one-out cross-validation
    LeaveOneOut,
    /// Time series cross-validation
    TimeSeriesCV { n_splits: usize, test_size: usize },
    /// Bootstrap validation
    Bootstrap { n_iterations: usize, sample_ratio: Float },
    /// Monte Carlo cross-validation
    MonteCarlo { n_iterations: usize, test_size: Float },
    /// Nested cross-validation
    NestedCV { outer_folds: usize, inner_folds: usize },
    /// Custom validation strategy
    Custom { name: String },
}

/// Comprehensive metrics for ensemble evaluation
#[derive(Debug, Clone)]
pub struct EnsembleMetrics {
    /// Performance metrics
    pub performance: HashMap<String, Float>,
    /// Diversity metrics
    pub diversity: HashMap<String, Float>,
    /// Efficiency metrics
    pub efficiency: HashMap<String, Float>,
    /// Robustness metrics
    pub robustness: HashMap<String, Float>,
    /// Individual model contributions
    pub model_contributions: HashMap<String, Float>,
    /// Timestamp of evaluation
    pub timestamp: SystemTime,
}

impl EnsembleMetrics {
    /// Create new empty metrics
    pub fn new() -> Self {
        Self {
            performance: HashMap::new(),
            diversity: HashMap::new(),
            efficiency: HashMap::new(),
            robustness: HashMap::new(),
            model_contributions: HashMap::new(),
            timestamp: SystemTime::now(),
        }
    }

    /// Add performance metric
    pub fn add_performance_metric(mut self, name: &str, value: Float) -> Self {
        self.performance.insert(name.to_string(), value);
        self
    }

    /// Add diversity metric
    pub fn add_diversity_metric(mut self, name: &str, value: Float) -> Self {
        self.diversity.insert(name.to_string(), value);
        self
    }

    /// Add efficiency metric
    pub fn add_efficiency_metric(mut self, name: &str, value: Float) -> Self {
        self.efficiency.insert(name.to_string(), value);
        self
    }

    /// Add robustness metric
    pub fn add_robustness_metric(mut self, name: &str, value: Float) -> Self {
        self.robustness.insert(name.to_string(), value);
        self
    }

    /// Add model contribution
    pub fn add_model_contribution(mut self, model_name: &str, contribution: Float) -> Self {
        self.model_contributions.insert(model_name.to_string(), contribution);
        self
    }

    /// Get overall ensemble score (weighted combination of metrics)
    pub fn overall_score(&self, weights: Option<&HashMap<String, Float>>) -> Float {
        let default_weights = HashMap::from([
            ("performance".to_string(), 0.5),
            ("diversity".to_string(), 0.2),
            ("efficiency".to_string(), 0.2),
            ("robustness".to_string(), 0.1),
        ]);

        let weights = weights.unwrap_or(&default_weights);

        let perf_score = self.performance.values().sum::<Float>() / self.performance.len().max(1) as Float;
        let div_score = self.diversity.values().sum::<Float>() / self.diversity.len().max(1) as Float;
        let eff_score = self.efficiency.values().sum::<Float>() / self.efficiency.len().max(1) as Float;
        let rob_score = self.robustness.values().sum::<Float>() / self.robustness.len().max(1) as Float;

        let weighted_score =
            perf_score * weights.get("performance").unwrap_or(&0.5) +
            div_score * weights.get("diversity").unwrap_or(&0.2) +
            eff_score * weights.get("efficiency").unwrap_or(&0.2) +
            rob_score * weights.get("robustness").unwrap_or(&0.1);

        weighted_score
    }

    /// Convert metrics to JSON-compatible format
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "performance": self.performance,
            "diversity": self.diversity,
            "efficiency": self.efficiency,
            "robustness": self.robustness,
            "model_contributions": self.model_contributions,
            "timestamp": self.timestamp.duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or(Duration::ZERO).as_secs(),
            "overall_score": self.overall_score(None)
        })
    }
}

impl Default for EnsembleMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for ensemble systems
#[derive(Debug, Clone)]
pub struct EnsembleConfiguration {
    /// Ensemble method to use
    pub method: EnsembleMethod,
    /// Rule for combining predictions
    pub combining_rule: CombiningRule,
    /// Weighting scheme for models
    pub weighting_scheme: WeightingScheme,
    /// Validation strategy
    pub validation_strategy: ValidationStrategy,
    /// Prediction mode
    pub prediction_mode: PredictionMode,
    /// Enable parallel processing
    pub parallel_processing: bool,
    /// Memory optimization level
    pub memory_optimization: MemoryOptimization,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Verbose output level
    pub verbose: VerbosityLevel,
}

/// Memory optimization levels
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryOptimization {
    None,
    Basic,
    Aggressive,
    Custom {
        cache_predictions: bool,
        stream_processing: bool,
        lazy_evaluation: bool,
    },
}

/// Verbosity levels for ensemble output
#[derive(Debug, Clone, PartialEq)]
pub enum VerbosityLevel {
    /// Silent (no output)
    Silent,
    /// Minimal output (errors only)
    Minimal,
    /// Standard output (progress and results)
    Standard,
    /// Detailed output (including intermediate results)
    Detailed,
    /// Debug output (all information)
    Debug,
}

impl Default for EnsembleConfiguration {
    fn default() -> Self {
        Self {
            method: EnsembleMethod::Voting,
            combining_rule: CombiningRule::Average,
            weighting_scheme: WeightingScheme::Uniform,
            validation_strategy: ValidationStrategy::CrossValidation { folds: 5 },
            prediction_mode: PredictionMode::Standard,
            parallel_processing: false,
            memory_optimization: MemoryOptimization::Basic,
            random_state: None,
            verbose: VerbosityLevel::Standard,
        }
    }
}

/// Traits for ensemble components
pub trait EnsembleComponent {
    /// Get component name
    fn name(&self) -> &str;

    /// Get component type
    fn component_type(&self) -> &str;

    /// Check if component is ready for use
    fn is_ready(&self) -> bool;

    /// Get component configuration
    fn configuration(&self) -> HashMap<String, String>;
}

/// Trait for ensemble predictors
pub trait EnsemblePredictor {
    /// Get ensemble member names
    fn member_names(&self) -> Vec<String>;

    /// Get ensemble size
    fn ensemble_size(&self) -> usize;

    /// Get individual member predictions
    fn member_predictions(&self, x: &ArrayView2<'_, Float>) -> SklResult<Vec<Array1<Float>>>;

    /// Get prediction confidence/uncertainty
    fn prediction_confidence(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<Float>>;

    /// Get ensemble metrics
    fn ensemble_metrics(&self) -> SklResult<EnsembleMetrics>;
}

/// Ensemble factory for creating different ensemble types
pub struct EnsembleFactory;

impl EnsembleFactory {
    /// Create ensemble configuration from method type
    pub fn create_config(method: EnsembleMethod) -> EnsembleConfiguration {
        match method {
            EnsembleMethod::Voting => EnsembleConfiguration {
                method,
                combining_rule: CombiningRule::Majority,
                weighting_scheme: WeightingScheme::Uniform,
                ..Default::default()
            },
            EnsembleMethod::Bagging => EnsembleConfiguration {
                method,
                combining_rule: CombiningRule::Average,
                weighting_scheme: WeightingScheme::Uniform,
                ..Default::default()
            },
            EnsembleMethod::Boosting => EnsembleConfiguration {
                method,
                combining_rule: CombiningRule::WeightedAverage { weights: vec![] },
                weighting_scheme: WeightingScheme::PerformanceBased {
                    metric: PerformanceMetric::Accuracy
                },
                ..Default::default()
            },
            EnsembleMethod::Stacking => EnsembleConfiguration {
                method,
                combining_rule: CombiningRule::LinearCombination {
                    coefficients: Array1::zeros(0)
                },
                weighting_scheme: WeightingScheme::LearnedWeights {
                    optimization_method: OptimizationMethod::GradientDescent { learning_rate: 0.01 }
                },
                ..Default::default()
            },
            _ => EnsembleConfiguration {
                method,
                ..Default::default()
            },
        }
    }

    /// Get recommended validation strategy for method
    pub fn recommended_validation(method: &EnsembleMethod) -> ValidationStrategy {
        match method {
            EnsembleMethod::Boosting | EnsembleMethod::Stacking => {
                ValidationStrategy::StratifiedCrossValidation { folds: 5 }
            },
            EnsembleMethod::BayesianEnsemble => {
                ValidationStrategy::Bootstrap { n_iterations: 1000, sample_ratio: 0.8 }
            },
            _ => ValidationStrategy::CrossValidation { folds: 5 },
        }
    }
}

/// Utility functions for ensemble types
pub mod utils {
    use super::*;

    /// Calculate diversity between two prediction arrays
    pub fn calculate_diversity(
        pred1: &Array1<Float>,
        pred2: &Array1<Float>,
        measure: &DiversityMeasure,
    ) -> SklResult<Float> {
        if pred1.len() != pred2.len() {
            return Err(sklears_core::prelude::SklearsError::InvalidInput(
                "Prediction arrays must have same length".to_string()
            ));
        }

        match measure {
            DiversityMeasure::Disagreement => {
                let disagreements = pred1.iter()
                    .zip(pred2.iter())
                    .filter(|(a, b)| (a - b).abs() > Float::EPSILON)
                    .count();
                Ok(disagreements as Float / pred1.len() as Float)
            },
            DiversityMeasure::CorrelationCoefficient => {
                let mean1 = pred1.mean().unwrap_or(0.0);
                let mean2 = pred2.mean().unwrap_or(0.0);

                let covariance: Float = pred1.iter()
                    .zip(pred2.iter())
                    .map(|(a, b)| (a - mean1) * (b - mean2))
                    .sum::<Float>() / pred1.len() as Float;

                let var1: Float = pred1.iter()
                    .map(|a| (a - mean1).powi(2))
                    .sum::<Float>() / pred1.len() as Float;

                let var2: Float = pred2.iter()
                    .map(|b| (b - mean2).powi(2))
                    .sum::<Float>() / pred2.len() as Float;

                let correlation = covariance / (var1.sqrt() * var2.sqrt());
                Ok(correlation)
            },
            _ => Ok(0.0), // Simplified for other measures
        }
    }

    /// Convert performance metric to optimization direction
    pub fn optimization_direction(metric: &PerformanceMetric) -> OptimizationDirection {
        match metric {
            PerformanceMetric::Accuracy |
            PerformanceMetric::F1Score |
            PerformanceMetric::AucRoc |
            PerformanceMetric::AucPr |
            PerformanceMetric::R2Score => OptimizationDirection::Maximize,

            PerformanceMetric::MeanSquaredError |
            PerformanceMetric::MeanAbsoluteError |
            PerformanceMetric::LogLoss => OptimizationDirection::Minimize,

            PerformanceMetric::Custom { .. } => OptimizationDirection::Maximize, // Default
        }
    }

    /// Normalize weights to sum to 1
    pub fn normalize_weights(weights: &mut [Float]) {
        let sum: Float = weights.iter().sum();
        if sum > 0.0 {
            for weight in weights.iter_mut() {
                *weight /= sum;
            }
        } else {
            let uniform_weight = 1.0 / weights.len() as Float;
            for weight in weights.iter_mut() {
                *weight = uniform_weight;
            }
        }
    }
}

/// Optimization directions for hyperparameter tuning
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationDirection {
    /// Maximize the objective
    Maximize,
    /// Minimize the objective
    Minimize,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ensemble_method_properties() {
        let voting = EnsembleMethod::Voting;
        assert_eq!(voting.name(), "voting");
        assert!(voting.supports_probabilities());

        let boosting = EnsembleMethod::Boosting;
        assert_eq!(boosting.name(), "boosting");
        assert!(boosting.supports_probabilities());

        let custom = EnsembleMethod::Custom {
            name: "test_ensemble".to_string(),
            description: "Test ensemble method".to_string(),
        };
        assert_eq!(custom.name(), "test_ensemble");
        assert_eq!(custom.description(), "Test ensemble method");
    }

    #[test]
    fn test_combining_rule_average() {
        let rule = CombiningRule::Average;
        let pred1 = Array1::from(vec![1.0, 2.0, 3.0]);
        let pred2 = Array1::from(vec![2.0, 4.0, 6.0]);
        let predictions = vec![pred1, pred2];

        let result = rule.combine(&predictions).unwrap();
        let expected = Array1::from(vec![1.5, 3.0, 4.5]);

        for (a, b) in result.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_combining_rule_weighted_average() {
        let weights = vec![0.3, 0.7];
        let rule = CombiningRule::WeightedAverage { weights };
        let pred1 = Array1::from(vec![1.0, 2.0]);
        let pred2 = Array1::from(vec![3.0, 4.0]);
        let predictions = vec![pred1, pred2];

        let result = rule.combine(&predictions).unwrap();
        let expected = Array1::from(vec![2.4, 3.4]); // 0.3*1 + 0.7*3 = 2.4, etc.

        for (a, b) in result.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_combining_rule_median() {
        let rule = CombiningRule::Median;
        let pred1 = Array1::from(vec![1.0, 5.0]);
        let pred2 = Array1::from(vec![2.0, 3.0]);
        let pred3 = Array1::from(vec![3.0, 4.0]);
        let predictions = vec![pred1, pred2, pred3];

        let result = rule.combine(&predictions).unwrap();
        let expected = Array1::from(vec![2.0, 4.0]);

        for (a, b) in result.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_ensemble_metrics() {
        let metrics = EnsembleMetrics::new()
            .add_performance_metric("accuracy", 0.95)
            .add_diversity_metric("disagreement", 0.3)
            .add_efficiency_metric("training_time", 120.0)
            .add_robustness_metric("noise_resistance", 0.8);

        assert_eq!(metrics.performance.get("accuracy"), Some(&0.95));
        assert_eq!(metrics.diversity.get("disagreement"), Some(&0.3));
        assert_eq!(metrics.efficiency.get("training_time"), Some(&120.0));
        assert_eq!(metrics.robustness.get("noise_resistance"), Some(&0.8));

        let overall = metrics.overall_score(None);
        assert!(overall > 0.0);
    }

    #[test]
    fn test_ensemble_configuration_defaults() {
        let config = EnsembleConfiguration::default();
        assert_eq!(config.method, EnsembleMethod::Voting);
        assert_eq!(config.combining_rule, CombiningRule::Average);
        assert_eq!(config.weighting_scheme, WeightingScheme::Uniform);
        assert!(!config.parallel_processing);
    }

    #[test]
    fn test_ensemble_factory() {
        let voting_config = EnsembleFactory::create_config(EnsembleMethod::Voting);
        assert_eq!(voting_config.combining_rule, CombiningRule::Majority);

        let boosting_config = EnsembleFactory::create_config(EnsembleMethod::Boosting);
        match boosting_config.combining_rule {
            CombiningRule::WeightedAverage { .. } => {},
            _ => panic!("Expected WeightedAverage for boosting"),
        }

        let validation = EnsembleFactory::recommended_validation(&EnsembleMethod::Stacking);
        match validation {
            ValidationStrategy::StratifiedCrossValidation { folds: 5 } => {},
            _ => panic!("Expected StratifiedCrossValidation for stacking"),
        }
    }

    #[test]
    fn test_diversity_calculation() {
        let pred1 = Array1::from(vec![1.0, 2.0, 3.0]);
        let pred2 = Array1::from(vec![1.0, 3.0, 3.0]);

        let disagreement = utils::calculate_diversity(
            &pred1, &pred2, &DiversityMeasure::Disagreement
        ).unwrap();
        assert!((disagreement - 1.0/3.0).abs() < 1e-6);

        let correlation = utils::calculate_diversity(
            &pred1, &pred2, &DiversityMeasure::CorrelationCoefficient
        ).unwrap();
        assert!(correlation >= -1.0 && correlation <= 1.0);
    }

    #[test]
    fn test_weight_normalization() {
        let mut weights = vec![1.0, 2.0, 3.0];
        utils::normalize_weights(&mut weights);
        let sum: Float = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        let mut zero_weights = vec![0.0, 0.0, 0.0];
        utils::normalize_weights(&mut zero_weights);
        let uniform_sum: Float = zero_weights.iter().sum();
        assert!((uniform_sum - 1.0).abs() < 1e-6);
    }
}