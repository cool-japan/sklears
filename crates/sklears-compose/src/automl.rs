//! `AutoML` integration with neural architecture search
//!
//! This module provides automated machine learning capabilities including
//! neural architecture search, hyperparameter optimization, and automated
//! feature engineering for pipeline construction.

use scirs2_core::ndarray::{ArrayView1, ArrayView2};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{thread_rng, Rng, SeedableRng};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::Estimator,
    types::{Float, FloatBounds},
};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use crate::{FluentPipelineBuilder, PipelineConfiguration};

/// `AutoML` pipeline optimizer for automated model selection and hyperparameter tuning
#[derive(Debug)]
pub struct AutoMLOptimizer {
    /// Search configuration
    config: AutoMLConfig,
    /// Search space for architectures
    search_space: SearchSpace,
    /// Optimization history
    history: OptimizationHistory,
    /// Random number generator
    rng: StdRng,
}

/// `AutoML` configuration
#[derive(Debug, Clone)]
pub struct AutoMLConfig {
    /// Maximum optimization time
    pub max_time: Duration,
    /// Maximum number of trials
    pub max_trials: usize,
    /// Cross-validation folds
    pub cv_folds: usize,
    /// Optimization metric
    pub metric: OptimizationMetric,
    /// Search strategy
    pub strategy: SearchStrategy,
    /// Population size for genetic algorithms
    pub population_size: usize,
    /// Early stopping patience
    pub early_stopping_patience: Option<usize>,
    /// Random seed
    pub random_seed: Option<u64>,
}

/// Optimization metric
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizationMetric {
    /// Accuracy for classification
    Accuracy,
    /// F1-score for classification
    F1Score,
    /// AUC-ROC for classification
    AUCROC,
    /// Mean Squared Error for regression
    MSE,
    /// Root Mean Squared Error for regression
    RMSE,
    /// Mean Absolute Error for regression
    MAE,
    /// R-squared for regression
    R2,
    /// Custom metric with function
    Custom(String),
}

/// Search strategy for optimization
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchStrategy {
    /// Random search
    Random,
    /// Grid search
    Grid,
    /// Bayesian optimization
    Bayesian,
    /// Genetic algorithm
    Genetic,
    /// Particle swarm optimization
    ParticleSwarm,
    /// Differential evolution
    DifferentialEvolution,
    /// Tree-structured Parzen Estimator
    TPE,
    /// Hyperband
    Hyperband,
}

/// Search space definition for `AutoML`
#[derive(Debug, Clone)]
pub struct SearchSpace {
    /// Algorithm choices
    pub algorithms: Vec<AlgorithmChoice>,
    /// Preprocessing options
    pub preprocessing: Vec<PreprocessingChoice>,
    /// Feature engineering options
    pub feature_engineering: Vec<FeatureEngineeringChoice>,
    /// Hyperparameter ranges
    pub hyperparameters: BTreeMap<String, ParameterRange>,
    /// Architecture constraints
    pub constraints: Vec<ArchitectureConstraint>,
}

/// Algorithm choice in search space
#[derive(Debug, Clone)]
pub struct AlgorithmChoice {
    /// Algorithm name
    pub name: String,
    /// Algorithm type
    pub algorithm_type: AlgorithmType,
    /// Hyperparameter ranges specific to this algorithm
    pub hyperparameters: BTreeMap<String, ParameterRange>,
    /// Resource requirements
    pub resource_requirements: ResourceRequirements,
}

/// Algorithm type enumeration
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlgorithmType {
    /// Linear models
    Linear,
    /// Tree-based models
    Tree,
    /// Ensemble methods
    Ensemble,
    /// Neural networks
    NeuralNetwork,
    /// Support Vector Machines
    SVM,
    /// Nearest neighbors
    KNN,
    /// Naive Bayes
    NaiveBayes,
    /// Custom algorithm
    Custom(String),
}

/// Preprocessing choice
#[derive(Debug, Clone)]
pub struct PreprocessingChoice {
    /// Preprocessing step name
    pub name: String,
    /// Step parameters
    pub parameters: BTreeMap<String, ParameterRange>,
    /// Optional flag (can be skipped)
    pub optional: bool,
}

/// Feature engineering choice
#[derive(Debug, Clone)]
pub struct FeatureEngineeringChoice {
    /// Feature engineering method name
    pub name: String,
    /// Method parameters
    pub parameters: BTreeMap<String, ParameterRange>,
    /// Optional flag
    pub optional: bool,
    /// Computational cost estimate
    pub cost_estimate: f64,
}

/// Parameter range for hyperparameter optimization
#[derive(Debug, Clone)]
pub enum ParameterRange {
    /// Continuous range
    Continuous { min: f64, max: f64, log_scale: bool },
    /// Discrete integer range
    Integer { min: i64, max: i64 },
    /// Categorical choices
    Categorical(Vec<String>),
    /// Boolean choice
    Boolean,
    /// Fixed value
    Fixed(ParameterValue),
}

/// Parameter value
#[derive(Debug, Clone)]
pub enum ParameterValue {
    /// Float
    Float(f64),
    /// Int
    Int(i64),
    /// String
    String(String),
    /// Bool
    Bool(bool),
    /// Array
    Array(Vec<ParameterValue>),
}

/// Architecture constraint
#[derive(Debug, Clone)]
pub enum ArchitectureConstraint {
    /// Maximum number of layers
    MaxLayers(usize),
    /// Maximum number of parameters
    MaxParameters(usize),
    /// Maximum memory usage (MB)
    MaxMemoryMB(usize),
    /// Maximum training time
    MaxTrainingTime(Duration),
    /// Required accuracy threshold
    MinAccuracy(f64),
}

/// Resource requirements for an algorithm
#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    /// Memory requirement (MB)
    pub memory_mb: usize,
    /// CPU cores needed
    pub cpu_cores: usize,
    /// Training time complexity
    pub time_complexity: TimeComplexity,
    /// GPU requirement
    pub requires_gpu: bool,
}

/// Time complexity classification
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimeComplexity {
    /// Constant
    Constant,
    /// Logarithmic
    Logarithmic,
    /// Linear
    Linear,
    /// Linearithmic
    Linearithmic,
    /// Quadratic
    Quadratic,
    /// Cubic
    Cubic,
    /// Exponential
    Exponential,
}

/// Optimization history tracking
#[derive(Debug, Clone)]
pub struct OptimizationHistory {
    /// Trial results
    pub trials: Vec<TrialResult>,
    /// Best score achieved
    pub best_score: Option<f64>,
    /// Best configuration
    pub best_config: Option<PipelineConfiguration>,
    /// Optimization start time
    pub start_time: Option<Instant>,
    /// Total optimization time
    pub total_time: Duration,
}

/// Individual trial result
#[derive(Debug, Clone)]
pub struct TrialResult {
    /// Trial identifier
    pub trial_id: usize,
    /// Pipeline configuration tried
    pub config: PipelineConfiguration,
    /// Achieved score
    pub score: f64,
    /// Training time
    pub training_time: Duration,
    /// Validation scores (for CV)
    pub cv_scores: Vec<f64>,
    /// Trial timestamp
    pub timestamp: Instant,
    /// Trial status
    pub status: TrialStatus,
    /// Error message if failed
    pub error: Option<String>,
}

/// Trial status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrialStatus {
    /// Trial completed successfully
    Success,
    /// Trial failed
    Failed,
    /// Trial was stopped early
    Stopped,
    /// Trial is running
    Running,
    /// Trial is queued
    Queued,
}

/// Neural Architecture Search (NAS) component
#[derive(Debug)]
pub struct NeuralArchitectureSearch {
    /// Search space for neural architectures
    search_space: NeuralSearchSpace,
    /// Search strategy
    strategy: NASStrategy,
    /// Evaluation method
    evaluator: ArchitectureEvaluator,
}

/// Neural architecture search space
#[derive(Debug, Clone)]
pub struct NeuralSearchSpace {
    /// Available layer types
    pub layer_types: Vec<LayerType>,
    /// Number of layers range
    pub num_layers: ParameterRange,
    /// Hidden units range
    pub hidden_units: ParameterRange,
    /// Activation functions
    pub activations: Vec<ActivationFunction>,
    /// Regularization options
    pub regularization: Vec<RegularizationOption>,
    /// Connection patterns
    pub connections: Vec<ConnectionPattern>,
}

/// Layer type for neural networks
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerType {
    /// Dense
    Dense,
    /// Dropout
    Dropout,
    /// BatchNorm
    BatchNorm,
    /// Convolution
    Convolution,
    /// Pooling
    Pooling,
    /// LSTM
    LSTM,
    /// GRU
    GRU,
    /// Attention
    Attention,
    /// Embedding
    Embedding,
    /// Custom
    Custom(String),
}

/// Activation function options
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActivationFunction {
    /// ReLU
    ReLU,
    /// LeakyReLU
    LeakyReLU,
    /// ELU
    ELU,
    /// Swish
    Swish,
    /// GELU
    GELU,
    /// Tanh
    Tanh,
    /// Sigmoid
    Sigmoid,
    /// Identity
    Identity,
    /// Custom
    Custom(String),
}

/// Regularization options
#[derive(Debug, Clone)]
pub struct RegularizationOption {
    /// Regularization type
    pub reg_type: RegularizationType,
    /// Strength parameter range
    pub strength: ParameterRange,
}

/// Regularization type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegularizationType {
    /// L1
    L1,
    /// L2
    L2,
    /// Dropout
    Dropout,
    /// BatchNorm
    BatchNorm,
    /// LayerNorm
    LayerNorm,
    /// EarlyStopping
    EarlyStopping,
    /// Custom
    Custom(String),
}

/// Connection pattern for neural architectures
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnectionPattern {
    /// Sequential
    Sequential,
    /// Residual
    Residual,
    /// DenseNet
    DenseNet,
    /// Highway
    Highway,
    /// Custom
    Custom(String),
}

/// Neural Architecture Search strategy
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NASStrategy {
    /// Random search over architectures
    Random,
    /// Evolutionary algorithm
    Evolutionary,
    /// Reinforcement learning based
    ReinforcementLearning,
    /// Differentiable architecture search
    DARTS,
    /// Progressive search
    Progressive,
    /// One-shot architecture search
    OneShot,
}

/// Architecture evaluator for NAS
#[derive(Debug, Clone)]
pub struct ArchitectureEvaluator {
    /// Evaluation strategy
    pub strategy: EvaluationStrategy,
    /// Maximum evaluation time per architecture
    pub max_eval_time: Duration,
    /// Early stopping criteria
    pub early_stopping: Option<EarlyStoppingCriteria>,
}

/// Evaluation strategy for architectures
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvaluationStrategy {
    /// Full training
    FullTraining,
    /// Early stopping
    EarlyStopping,
    /// Weight sharing
    WeightSharing,
    /// Performance prediction
    PerformancePrediction,
    /// Progressive evaluation
    Progressive,
}

/// Early stopping criteria
#[derive(Debug, Clone)]
pub struct EarlyStoppingCriteria {
    /// Metric to monitor
    pub metric: String,
    /// Patience (epochs)
    pub patience: usize,
    /// Minimum improvement threshold
    pub min_delta: f64,
}

impl Default for AutoMLConfig {
    fn default() -> Self {
        Self {
            max_time: Duration::from_secs(3600), // 1 hour
            max_trials: 100,
            cv_folds: 5,
            metric: OptimizationMetric::Accuracy,
            strategy: SearchStrategy::Random,
            population_size: 20,
            early_stopping_patience: Some(10),
            random_seed: None,
        }
    }
}

impl Default for OptimizationHistory {
    fn default() -> Self {
        Self {
            trials: Vec::new(),
            best_score: None,
            best_config: None,
            start_time: None,
            total_time: Duration::ZERO,
        }
    }
}

impl AutoMLOptimizer {
    /// Create a new `AutoML` optimizer
    pub fn new(config: AutoMLConfig) -> SklResult<Self> {
        let rng = if let Some(seed) = config.random_seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_rng(&mut thread_rng())
        };

        Ok(Self {
            config,
            search_space: SearchSpace::default(),
            history: OptimizationHistory::default(),
            rng,
        })
    }

    /// Set the search space for optimization
    #[must_use]
    pub fn search_space(mut self, search_space: SearchSpace) -> Self {
        self.search_space = search_space;
        self
    }

    /// Run automated optimization
    pub fn optimize(
        &mut self,
        x_train: &ArrayView2<Float>,
        y_train: &ArrayView1<Float>,
        x_val: Option<&ArrayView2<Float>>,
        y_val: Option<&ArrayView1<Float>>,
    ) -> SklResult<FluentPipelineBuilder> {
        self.history.start_time = Some(Instant::now());
        let start_time = Instant::now();

        let mut best_score = f64::NEG_INFINITY;
        let mut best_config = None;
        let mut trials_without_improvement = 0;

        for trial_id in 0..self.config.max_trials {
            // Check time limit
            if start_time.elapsed() > self.config.max_time {
                break;
            }

            // Generate candidate configuration
            let config = self.generate_candidate_config()?;

            // Evaluate configuration
            let trial_result =
                self.evaluate_config(&config, x_train, y_train, x_val, y_val, trial_id)?;

            // Update history
            self.history.trials.push(trial_result.clone());

            // Check if this is the best configuration
            if trial_result.score > best_score {
                best_score = trial_result.score;
                best_config = Some(config);
                trials_without_improvement = 0;

                self.history.best_score = Some(best_score);
                self.history.best_config = best_config.clone();
            } else {
                trials_without_improvement += 1;
            }

            // Early stopping
            if let Some(patience) = self.config.early_stopping_patience {
                if trials_without_improvement >= patience {
                    break;
                }
            }
        }

        self.history.total_time = start_time.elapsed();

        // Return best configuration as FluentPipelineBuilder
        if let Some(best_config) = best_config {
            Ok(self.config_to_builder(best_config))
        } else {
            Err(SklearsError::InvalidData {
                reason: "No valid configuration found during optimization".to_string(),
            })
        }
    }

    /// Generate a candidate configuration based on search strategy
    fn generate_candidate_config(&mut self) -> SklResult<PipelineConfiguration> {
        match self.config.strategy {
            SearchStrategy::Random => self.generate_random_config(),
            SearchStrategy::Genetic => self.generate_genetic_config(),
            SearchStrategy::Bayesian => self.generate_bayesian_config(),
            _ => self.generate_random_config(), // Fallback to random
        }
    }

    /// Generate random configuration
    fn generate_random_config(&mut self) -> SklResult<PipelineConfiguration> {
        // Sample random algorithm
        let algorithm = &self.search_space.algorithms
            [self.rng.gen_range(0..self.search_space.algorithms.len())];

        // Sample random preprocessing steps
        let preprocessing_steps: Vec<_> = self
            .search_space
            .preprocessing
            .iter()
            .filter(|step| !step.optional || self.rng.gen_bool(0.5))
            .collect();

        // Sample random feature engineering steps
        let feature_steps: Vec<_> = self
            .search_space
            .feature_engineering
            .iter()
            .filter(|step| !step.optional || self.rng.gen_bool(0.3))
            .collect();

        // Create configuration
        Ok(PipelineConfiguration::default())
    }

    /// Generate configuration using genetic algorithm
    fn generate_genetic_config(&mut self) -> SklResult<PipelineConfiguration> {
        // Simplified genetic algorithm implementation
        // In practice, this would maintain a population and perform crossover/mutation
        self.generate_random_config()
    }

    /// Generate configuration using Bayesian optimization
    fn generate_bayesian_config(&mut self) -> SklResult<PipelineConfiguration> {
        // Simplified Bayesian optimization
        // In practice, this would use Gaussian processes or other surrogate models
        self.generate_random_config()
    }

    /// Evaluate a configuration
    fn evaluate_config(
        &mut self,
        config: &PipelineConfiguration,
        x_train: &ArrayView2<Float>,
        y_train: &ArrayView1<Float>,
        x_val: Option<&ArrayView2<Float>>,
        y_val: Option<&ArrayView1<Float>>,
        trial_id: usize,
    ) -> SklResult<TrialResult> {
        let start_time = Instant::now();

        // Create pipeline from configuration
        let pipeline_builder = self.config_to_builder(config.clone());

        // For now, return a mock result
        // In a real implementation, this would:
        // 1. Build the pipeline
        // 2. Perform cross-validation or train/validation split
        // 3. Calculate the specified metric

        let mock_score = self.rng.gen_range(0.5..1.0);
        let cv_scores = (0..self.config.cv_folds)
            .map(|_| self.rng.gen_range(0.4..1.0))
            .collect();

        Ok(TrialResult {
            trial_id,
            config: config.clone(),
            score: mock_score,
            training_time: start_time.elapsed(),
            cv_scores,
            timestamp: start_time,
            status: TrialStatus::Success,
            error: None,
        })
    }

    /// Convert configuration to `FluentPipelineBuilder`
    fn config_to_builder(&self, config: PipelineConfiguration) -> FluentPipelineBuilder {
        // Create builder with the configuration
        FluentPipelineBuilder::data_science_preset()
            .memory(config.memory_config)
            .caching(config.caching)
            .validation(config.validation)
            .debug(config.debug)
    }

    /// Get optimization results
    #[must_use]
    pub fn get_results(&self) -> &OptimizationHistory {
        &self.history
    }

    /// Get best trial
    #[must_use]
    pub fn get_best_trial(&self) -> Option<&TrialResult> {
        self.history.trials.iter().max_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Generate optimization report
    #[must_use]
    pub fn generate_report(&self) -> OptimizationReport {
        /// OptimizationReport
        OptimizationReport {
            total_trials: self.history.trials.len(),
            successful_trials: self
                .history
                .trials
                .iter()
                .filter(|t| t.status == TrialStatus::Success)
                .count(),
            best_score: self.history.best_score,
            total_time: self.history.total_time,
            average_trial_time: if self.history.trials.is_empty() {
                None
            } else {
                Some(Duration::from_secs_f64(
                    self.history
                        .trials
                        .iter()
                        .map(|t| t.training_time.as_secs_f64())
                        .sum::<f64>()
                        / self.history.trials.len() as f64,
                ))
            },
            trials_summary: self.history.trials.clone(),
        }
    }
}

/// Optimization report
#[derive(Debug, Clone)]
pub struct OptimizationReport {
    /// Total number of trials
    pub total_trials: usize,
    /// Number of successful trials
    pub successful_trials: usize,
    /// Best score achieved
    pub best_score: Option<f64>,
    /// Total optimization time
    pub total_time: Duration,
    /// Average time per trial
    pub average_trial_time: Option<Duration>,
    /// Summary of all trials
    pub trials_summary: Vec<TrialResult>,
}

impl Default for SearchSpace {
    fn default() -> Self {
        Self {
            algorithms: vec![
                /// AlgorithmChoice
                AlgorithmChoice {
                    name: "LinearRegression".to_string(),
                    algorithm_type: AlgorithmType::Linear,
                    hyperparameters: BTreeMap::new(),
                    resource_requirements: ResourceRequirements {
                        memory_mb: 100,
                        cpu_cores: 1,
                        time_complexity: TimeComplexity::Linear,
                        requires_gpu: false,
                    },
                },
                /// AlgorithmChoice
                AlgorithmChoice {
                    name: "RandomForest".to_string(),
                    algorithm_type: AlgorithmType::Ensemble,
                    hyperparameters: BTreeMap::from([
                        (
                            "n_estimators".to_string(),
                            ParameterRange::Integer { min: 10, max: 1000 },
                        ),
                        (
                            "max_depth".to_string(),
                            ParameterRange::Integer { min: 1, max: 50 },
                        ),
                    ]),
                    resource_requirements: ResourceRequirements {
                        memory_mb: 500,
                        cpu_cores: 4,
                        time_complexity: TimeComplexity::Linearithmic,
                        requires_gpu: false,
                    },
                },
            ],
            preprocessing: vec![
                /// PreprocessingChoice
                PreprocessingChoice {
                    name: "StandardScaler".to_string(),
                    parameters: BTreeMap::new(),
                    optional: false,
                },
                /// PreprocessingChoice
                PreprocessingChoice {
                    name: "MinMaxScaler".to_string(),
                    parameters: BTreeMap::new(),
                    optional: true,
                },
            ],
            feature_engineering: vec![FeatureEngineeringChoice {
                name: "PolynomialFeatures".to_string(),
                parameters: BTreeMap::from([(
                    "degree".to_string(),
                    ParameterRange::Integer { min: 2, max: 4 },
                )]),
                optional: true,
                cost_estimate: 0.5,
            }],
            hyperparameters: BTreeMap::new(),
            constraints: vec![
                ArchitectureConstraint::MaxTrainingTime(Duration::from_secs(300)),
                ArchitectureConstraint::MaxMemoryMB(2048),
            ],
        }
    }
}

impl NeuralArchitectureSearch {
    /// Create a new NAS instance
    #[must_use]
    pub fn new(search_space: NeuralSearchSpace, strategy: NASStrategy) -> Self {
        Self {
            search_space,
            strategy,
            evaluator: ArchitectureEvaluator {
                strategy: EvaluationStrategy::EarlyStopping,
                max_eval_time: Duration::from_secs(300),
                early_stopping: Some(EarlyStoppingCriteria {
                    metric: "val_accuracy".to_string(),
                    patience: 10,
                    min_delta: 0.001,
                }),
            },
        }
    }

    /// Search for optimal neural architecture
    pub fn search(&mut self, max_architectures: usize) -> SklResult<Vec<NeuralArchitecture>> {
        let mut architectures = Vec::new();
        let mut rng = StdRng::from_rng(&mut thread_rng());

        for _ in 0..max_architectures {
            let architecture = self.generate_architecture(&mut rng)?;
            architectures.push(architecture);
        }

        // Sort by estimated performance (mock implementation)
        architectures.sort_by(|a, b| {
            b.estimated_performance
                .partial_cmp(&a.estimated_performance)
                .unwrap()
        });

        Ok(architectures)
    }

    /// Generate a neural architecture
    fn generate_architecture(&self, rng: &mut StdRng) -> SklResult<NeuralArchitecture> {
        let num_layers = match &self.search_space.num_layers {
            ParameterRange::Integer { min, max } => rng.gen_range(*min..*max + 1) as usize,
            _ => 3, // Default
        };

        let mut layers = Vec::new();
        for i in 0..num_layers {
            let layer_type = &self.search_space.layer_types
                [rng.gen_range(0..self.search_space.layer_types.len())];

            let activation = &self.search_space.activations
                [rng.gen_range(0..self.search_space.activations.len())];

            layers.push(NeuralLayer {
                layer_type: layer_type.clone(),
                units: Some(rng.gen_range(32..512)),
                activation: activation.clone(),
                layer_id: i,
            });
        }

        Ok(NeuralArchitecture {
            layers,
            connection_pattern: ConnectionPattern::Sequential,
            estimated_performance: rng.gen_range(0.5..1.0),
            parameter_count: rng.gen_range(1000..1_000_000),
            memory_usage_mb: rng.gen_range(10..500),
        })
    }
}

/// Neural architecture representation
#[derive(Debug, Clone)]
pub struct NeuralArchitecture {
    /// Network layers
    pub layers: Vec<NeuralLayer>,
    /// Connection pattern
    pub connection_pattern: ConnectionPattern,
    /// Estimated performance
    pub estimated_performance: f64,
    /// Total parameter count
    pub parameter_count: usize,
    /// Memory usage estimate (MB)
    pub memory_usage_mb: usize,
}

/// Neural layer representation
#[derive(Debug, Clone)]
pub struct NeuralLayer {
    /// Layer type
    pub layer_type: LayerType,
    /// Number of units (for dense layers)
    pub units: Option<usize>,
    /// Activation function
    pub activation: ActivationFunction,
    /// Layer identifier
    pub layer_id: usize,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_automl_config() {
        let config = AutoMLConfig::default();
        assert_eq!(config.max_trials, 100);
        assert_eq!(config.cv_folds, 5);
        assert_eq!(config.metric, OptimizationMetric::Accuracy);
    }

    #[test]
    fn test_automl_optimizer() {
        let config = AutoMLConfig::default();
        let optimizer = AutoMLOptimizer::new(config).unwrap();
        assert_eq!(optimizer.history.trials.len(), 0);
        assert!(optimizer.history.best_score.is_none());
    }

    #[test]
    fn test_search_space() {
        let search_space = SearchSpace::default();
        assert!(!search_space.algorithms.is_empty());
        assert!(!search_space.preprocessing.is_empty());
    }

    #[test]
    fn test_neural_architecture_search() {
        let search_space = NeuralSearchSpace {
            layer_types: vec![LayerType::Dense, LayerType::Dropout],
            num_layers: ParameterRange::Integer { min: 2, max: 5 },
            hidden_units: ParameterRange::Integer { min: 32, max: 512 },
            activations: vec![ActivationFunction::ReLU, ActivationFunction::Tanh],
            regularization: vec![],
            connections: vec![ConnectionPattern::Sequential],
        };

        let mut nas = NeuralArchitectureSearch::new(search_space, NASStrategy::Random);
        let architectures = nas.search(5).unwrap();

        assert_eq!(architectures.len(), 5);
        assert!(architectures[0].estimated_performance >= architectures[1].estimated_performance);
    }

    #[test]
    fn test_parameter_ranges() {
        let float_range = ParameterRange::Continuous {
            min: 0.1,
            max: 1.0,
            log_scale: false,
        };
        let int_range = ParameterRange::Integer { min: 1, max: 100 };
        let categorical = ParameterRange::Categorical(vec!["a".to_string(), "b".to_string()]);

        match float_range {
            ParameterRange::Continuous { min, max, .. } => {
                assert_eq!(min, 0.1);
                assert_eq!(max, 1.0);
            }
            _ => panic!("Wrong parameter range type"),
        }
    }

    #[test]
    fn test_optimization_history() {
        let mut history = OptimizationHistory::default();
        assert!(history.trials.is_empty());
        assert!(history.best_score.is_none());

        let trial = TrialResult {
            trial_id: 0,
            config: PipelineConfiguration::default(),
            score: 0.85,
            training_time: Duration::from_secs(30),
            cv_scores: vec![0.8, 0.9, 0.85],
            timestamp: Instant::now(),
            status: TrialStatus::Success,
            error: None,
        };

        history.trials.push(trial);
        assert_eq!(history.trials.len(), 1);
    }

    #[test]
    fn test_trial_status() {
        assert_eq!(TrialStatus::Success, TrialStatus::Success);
        assert_ne!(TrialStatus::Success, TrialStatus::Failed);
    }

    #[test]
    fn test_algorithm_types() {
        let linear = AlgorithmType::Linear;
        let tree = AlgorithmType::Tree;
        let neural = AlgorithmType::NeuralNetwork;

        assert_eq!(linear, AlgorithmType::Linear);
        assert_ne!(linear, tree);
        assert_ne!(tree, neural);
    }

    #[test]
    fn test_resource_requirements() {
        let requirements = ResourceRequirements {
            memory_mb: 1024,
            cpu_cores: 4,
            time_complexity: TimeComplexity::Linear,
            requires_gpu: false,
        };

        assert_eq!(requirements.memory_mb, 1024);
        assert_eq!(requirements.cpu_cores, 4);
        assert!(!requirements.requires_gpu);
    }
}
