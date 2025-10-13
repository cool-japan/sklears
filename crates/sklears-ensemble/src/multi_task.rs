//! Multi-Task Ensemble Methods
//!
//! This module provides ensemble methods for multi-task learning, where multiple
//! related learning tasks are solved jointly to improve generalization performance
//! by leveraging information shared across tasks.

use crate::bagging::BaggingClassifier;
use crate::gradient_boosting::{
    GradientBoostingConfig, GradientBoostingRegressor, TrainedGradientBoostingRegressor,
};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::Result as SklResult,
    prelude::{Predict, SklearsError},
    traits::{Estimator, Fit, Trained, Untrained},
};
use std::collections::HashMap;

/// A trait that combines the `Estimator` and `Predict` traits.
pub trait MultiTaskEstimator<C, E, F, X, Y>:
    Estimator<Config = C, Error = E, Float = F> + Predict<X, Y>
{
}

impl<T, C, E, F, X, Y> MultiTaskEstimator<C, E, F, X, Y> for T where
    T: Estimator<Config = C, Error = E, Float = F> + Predict<X, Y>
{
}

/// Configuration for multi-task ensemble learning
#[derive(Debug, Clone)]
pub struct MultiTaskEnsembleConfig {
    /// Number of base estimators per task
    pub n_estimators_per_task: usize,
    /// Task sharing strategy
    pub sharing_strategy: TaskSharingStrategy,
    /// Task similarity metric for adaptive sharing
    pub similarity_metric: TaskSimilarityMetric,
    /// Minimum task similarity threshold for sharing
    pub min_similarity_threshold: f64,
    /// Task weighting strategy
    pub task_weighting: TaskWeightingStrategy,
    /// Whether to use task-specific feature selection
    pub use_task_specific_features: bool,
    /// Number of shared features across tasks
    pub n_shared_features: Option<usize>,
    /// Regularization strength for task sharing
    pub sharing_regularization: f64,
    /// Maximum depth for task hierarchy
    pub max_task_depth: usize,
    /// Cross-task validation strategy
    pub cross_task_validation: CrossTaskValidation,
}

impl Default for MultiTaskEnsembleConfig {
    fn default() -> Self {
        Self {
            n_estimators_per_task: 10,
            sharing_strategy: TaskSharingStrategy::SharedRepresentation,
            similarity_metric: TaskSimilarityMetric::CorrelationBased,
            min_similarity_threshold: 0.3,
            task_weighting: TaskWeightingStrategy::Uniform,
            use_task_specific_features: true,
            n_shared_features: None,
            sharing_regularization: 0.1,
            max_task_depth: 3,
            cross_task_validation: CrossTaskValidation::LeaveOneTaskOut,
        }
    }
}

/// Strategies for sharing information between tasks
#[derive(Debug, Clone, PartialEq)]
pub enum TaskSharingStrategy {
    /// No sharing between tasks
    Independent,
    /// Shared representation learning
    SharedRepresentation,
    /// Parameter sharing between tasks
    ParameterSharing,
    /// Hierarchical task relationships
    HierarchicalSharing,
    /// Adaptive sharing based on task similarity
    AdaptiveSharing,
    /// Multi-level sharing with different granularities
    MultiLevelSharing,
    /// Transfer learning between tasks
    TransferLearning,
}

/// Metrics for measuring task similarity
#[derive(Debug, Clone, PartialEq)]
pub enum TaskSimilarityMetric {
    /// Correlation-based similarity
    CorrelationBased,
    /// Feature importance similarity
    FeatureImportanceSimilarity,
    /// Model prediction similarity
    PredictionSimilarity,
    /// Data distribution similarity
    DistributionSimilarity,
    /// Gradient similarity
    GradientSimilarity,
    /// Task performance correlation
    PerformanceCorrelation,
}

/// Strategies for weighting different tasks
#[derive(Debug, Clone, PartialEq)]
pub enum TaskWeightingStrategy {
    /// Equal weight for all tasks
    Uniform,
    /// Weight based on task difficulty
    DifficultyBased,
    /// Weight based on task sample size
    SampleSizeBased,
    /// Weight based on task performance
    PerformanceBased,
    /// Adaptive weighting during training
    AdaptiveWeighting,
    /// Weight based on task importance
    ImportanceBased,
}

/// Cross-task validation strategies
#[derive(Debug, Clone, PartialEq)]
pub enum CrossTaskValidation {
    /// Leave one task out for validation
    LeaveOneTaskOut,
    /// Cross-validation within each task
    WithinTaskCV,
    /// Hierarchical cross-validation
    HierarchicalCV,
    /// Time-based validation for temporal tasks
    TemporalCV,
    /// Stratified validation across tasks
    StratifiedCV,
}

/// Multi-task ensemble classifier
pub struct MultiTaskEnsembleClassifier<State = Untrained> {
    config: MultiTaskEnsembleConfig,
    state: std::marker::PhantomData<State>,
    // Fitted attributes - only populated after training
    task_models: Option<HashMap<String, Vec<BaggingClassifier<Trained>>>>,
    shared_models: Option<Vec<BaggingClassifier<Trained>>>,
    task_similarities: Option<HashMap<(String, String), f64>>,
    task_weights: Option<HashMap<String, f64>>,
    feature_selector: Option<MultiTaskFeatureSelector>,
    task_hierarchy: Option<TaskHierarchy>,
}

/// Multi-task ensemble regressor
pub struct MultiTaskEnsembleRegressor<State = Untrained> {
    config: MultiTaskEnsembleConfig,
    state: std::marker::PhantomData<State>,
    // Fitted attributes - only populated after training
    task_models: Option<HashMap<String, Vec<TrainedGradientBoostingRegressor>>>,
    shared_models: Option<Vec<TrainedGradientBoostingRegressor>>,
    task_similarities: Option<HashMap<(String, String), f64>>,
    task_weights: Option<HashMap<String, f64>>,
    feature_selector: Option<MultiTaskFeatureSelector>,
    task_hierarchy: Option<TaskHierarchy>,
}

/// Task hierarchy for hierarchical sharing
#[derive(Debug, Clone)]
pub struct TaskHierarchy {
    /// Parent-child relationships between tasks
    pub hierarchy: HashMap<String, Vec<String>>,
    /// Task depth in the hierarchy
    pub task_depths: HashMap<String, usize>,
    /// Sharing weights based on hierarchy
    pub hierarchy_weights: HashMap<(String, String), f64>,
}

/// Multi-task feature selector
#[derive(Debug, Clone)]
pub struct MultiTaskFeatureSelector {
    /// Task-specific feature masks
    pub task_feature_masks: HashMap<String, Vec<bool>>,
    /// Shared feature mask
    pub shared_feature_mask: Vec<bool>,
    /// Feature importance scores per task
    pub task_feature_importances: HashMap<String, Vec<f64>>,
    /// Global feature importance scores
    pub global_feature_importances: Vec<f64>,
}

/// Task-specific training data
#[derive(Debug, Clone)]
pub struct TaskData {
    /// Task identifier
    pub task_id: String,
    /// Features for this task
    pub features: Array2<f64>,
    /// Labels for this task
    pub labels: Vec<f64>,
    /// Sample weights (optional)
    pub sample_weights: Option<Vec<f64>>,
    /// Task metadata
    pub metadata: HashMap<String, String>,
}

/// Results from multi-task training
#[derive(Debug, Clone)]
pub struct MultiTaskTrainingResults {
    /// Training metrics per task
    pub task_metrics: HashMap<String, TaskMetrics>,
    /// Cross-task transfer effects
    pub transfer_effects: HashMap<(String, String), f64>,
    /// Final task similarities
    pub final_similarities: HashMap<(String, String), f64>,
    /// Convergence information
    pub convergence_info: ConvergenceInfo,
}

/// Metrics for individual tasks
#[derive(Debug, Clone)]
pub struct TaskMetrics {
    /// Training accuracy/error
    pub training_score: f64,
    /// Validation accuracy/error
    pub validation_score: f64,
    /// Number of training samples
    pub n_samples: usize,
    /// Training time
    pub training_time: f64,
    /// Model complexity measure
    pub complexity: f64,
}

/// Convergence information for multi-task training
#[derive(Debug, Clone)]
pub struct ConvergenceInfo {
    /// Number of iterations to convergence
    pub n_iterations: usize,
    /// Final loss value
    pub final_loss: f64,
    /// Convergence tolerance achieved
    pub tolerance_achieved: f64,
    /// Whether convergence was reached
    pub converged: bool,
}

impl MultiTaskEnsembleConfig {
    pub fn builder() -> MultiTaskEnsembleConfigBuilder {
        MultiTaskEnsembleConfigBuilder::default()
    }
}

#[derive(Default)]
pub struct MultiTaskEnsembleConfigBuilder {
    config: MultiTaskEnsembleConfig,
}

impl MultiTaskEnsembleConfigBuilder {
    pub fn n_estimators_per_task(mut self, n_estimators: usize) -> Self {
        self.config.n_estimators_per_task = n_estimators;
        self
    }

    pub fn sharing_strategy(mut self, strategy: TaskSharingStrategy) -> Self {
        self.config.sharing_strategy = strategy;
        self
    }

    pub fn similarity_metric(mut self, metric: TaskSimilarityMetric) -> Self {
        self.config.similarity_metric = metric;
        self
    }

    pub fn min_similarity_threshold(mut self, threshold: f64) -> Self {
        self.config.min_similarity_threshold = threshold;
        self
    }

    pub fn task_weighting(mut self, weighting: TaskWeightingStrategy) -> Self {
        self.config.task_weighting = weighting;
        self
    }

    pub fn use_task_specific_features(mut self, use_specific: bool) -> Self {
        self.config.use_task_specific_features = use_specific;
        self
    }

    pub fn sharing_regularization(mut self, regularization: f64) -> Self {
        self.config.sharing_regularization = regularization;
        self
    }

    pub fn cross_task_validation(mut self, validation: CrossTaskValidation) -> Self {
        self.config.cross_task_validation = validation;
        self
    }

    pub fn build(self) -> MultiTaskEnsembleConfig {
        self.config
    }
}

impl MultiTaskEnsembleRegressor {
    pub fn new(config: MultiTaskEnsembleConfig) -> Self {
        Self {
            config,
            state: std::marker::PhantomData,
            task_models: None,
            shared_models: None,
            task_similarities: None,
            task_weights: None,
            feature_selector: None,
            task_hierarchy: None,
        }
    }

    pub fn builder() -> MultiTaskEnsembleRegressorBuilder {
        MultiTaskEnsembleRegressorBuilder::new()
    }

    /// Fit the multi-task ensemble on multiple tasks
    pub fn fit_tasks(
        mut self,
        tasks: &[TaskData],
    ) -> SklResult<MultiTaskEnsembleRegressor<Trained>> {
        if tasks.is_empty() {
            return Err(SklearsError::InvalidInput("No tasks provided".to_string()));
        }

        // Initialize task weights
        self.initialize_task_weights(tasks)?;

        // Build task hierarchy if needed
        if matches!(
            self.config.sharing_strategy,
            TaskSharingStrategy::HierarchicalSharing
        ) {
            self.build_task_hierarchy(tasks)?;
        }

        // Initialize feature selector
        if self.config.use_task_specific_features {
            self.initialize_feature_selector(tasks)?;
        }

        // Compute task similarities
        self.compute_task_similarities(tasks)?;

        // Train models based on sharing strategy
        let training_results = match self.config.sharing_strategy {
            TaskSharingStrategy::Independent => self.train_independent_tasks(tasks)?,
            TaskSharingStrategy::SharedRepresentation => self.train_shared_representation(tasks)?,
            TaskSharingStrategy::ParameterSharing => self.train_parameter_sharing(tasks)?,
            TaskSharingStrategy::HierarchicalSharing => self.train_hierarchical_sharing(tasks)?,
            TaskSharingStrategy::AdaptiveSharing => self.train_adaptive_sharing(tasks)?,
            _ => self.train_independent_tasks(tasks)?, // Default fallback
        };

        let fitted_ensemble = MultiTaskEnsembleRegressor::<Trained> {
            config: self.config,
            state: std::marker::PhantomData,
            task_models: self.task_models,
            shared_models: self.shared_models,
            task_similarities: self.task_similarities,
            task_weights: self.task_weights,
            feature_selector: self.feature_selector,
            task_hierarchy: self.task_hierarchy,
        };

        Ok(fitted_ensemble)
    }

    /// Initialize task weights based on configuration
    fn initialize_task_weights(&mut self, tasks: &[TaskData]) -> SklResult<()> {
        let mut weights = HashMap::new();

        match self.config.task_weighting {
            TaskWeightingStrategy::Uniform => {
                let weight = 1.0 / tasks.len() as f64;
                for task in tasks {
                    weights.insert(task.task_id.clone(), weight);
                }
            }
            TaskWeightingStrategy::SampleSizeBased => {
                let total_samples: usize = tasks.iter().map(|t| t.features.shape()[0]).sum();
                for task in tasks {
                    let weight = task.features.shape()[0] as f64 / total_samples as f64;
                    weights.insert(task.task_id.clone(), weight);
                }
            }
            _ => {
                // Default to uniform for now
                let weight = 1.0 / tasks.len() as f64;
                for task in tasks {
                    weights.insert(task.task_id.clone(), weight);
                }
            }
        }

        self.task_weights = Some(weights);
        Ok(())
    }

    /// Build task hierarchy for hierarchical sharing
    fn build_task_hierarchy(&mut self, tasks: &[TaskData]) -> SklResult<()> {
        let mut hierarchy = HashMap::new();
        let mut task_depths = HashMap::new();
        let mut hierarchy_weights = HashMap::new();

        // Simple hierarchical construction based on task similarities
        // In practice, this could be based on domain knowledge or learned
        for (i, task1) in tasks.iter().enumerate() {
            task_depths.insert(task1.task_id.clone(), 0);
            hierarchy.insert(task1.task_id.clone(), Vec::new());

            for (j, task2) in tasks.iter().enumerate() {
                if i != j {
                    // Set hierarchical weight based on some similarity measure
                    let weight = 1.0 / (i.abs_diff(j) + 1) as f64;
                    hierarchy_weights
                        .insert((task1.task_id.clone(), task2.task_id.clone()), weight);
                }
            }
        }

        self.task_hierarchy = Some(TaskHierarchy {
            hierarchy,
            task_depths,
            hierarchy_weights,
        });

        Ok(())
    }

    /// Initialize feature selector for task-specific features
    fn initialize_feature_selector(&mut self, tasks: &[TaskData]) -> SklResult<()> {
        if tasks.is_empty() {
            return Ok(());
        }

        let n_features = tasks[0].features.shape()[1];
        let mut task_feature_masks = HashMap::new();
        let mut task_feature_importances = HashMap::new();

        // For now, use simple feature selection based on variance
        for task in tasks {
            let mut feature_mask = vec![true; n_features];
            let mut feature_importances = vec![0.0; n_features];

            // Calculate feature variances as a simple importance measure
            for j in 0..n_features {
                let column: Vec<f64> = (0..task.features.shape()[0])
                    .map(|i| task.features[[i, j]])
                    .collect();
                let mean = column.iter().sum::<f64>() / column.len() as f64;
                let variance =
                    column.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / column.len() as f64;

                feature_importances[j] = variance;
                feature_mask[j] = variance > 1e-8; // Keep features with non-zero variance
            }

            task_feature_masks.insert(task.task_id.clone(), feature_mask);
            task_feature_importances.insert(task.task_id.clone(), feature_importances);
        }

        // Global feature importance (average across tasks)
        let mut global_feature_importances = vec![0.0; n_features];
        for j in 0..n_features {
            let sum: f64 = task_feature_importances
                .values()
                .map(|importances| importances[j])
                .sum();
            global_feature_importances[j] = sum / tasks.len() as f64;
        }

        // Shared feature mask (features important across multiple tasks)
        let shared_feature_mask: Vec<bool> = (0..n_features)
            .map(|j| {
                let important_count = task_feature_importances
                    .values()
                    .filter(|importances| importances[j] > global_feature_importances[j] * 0.5)
                    .count();
                important_count >= tasks.len() / 2
            })
            .collect();

        self.feature_selector = Some(MultiTaskFeatureSelector {
            task_feature_masks,
            shared_feature_mask,
            task_feature_importances,
            global_feature_importances,
        });

        Ok(())
    }

    /// Compute similarities between tasks
    fn compute_task_similarities(&mut self, tasks: &[TaskData]) -> SklResult<()> {
        let mut similarities = HashMap::new();

        for (i, task1) in tasks.iter().enumerate() {
            for task2 in tasks.iter().skip(i + 1) {
                let similarity = self.calculate_task_similarity(task1, task2)?;
                similarities.insert((task1.task_id.clone(), task2.task_id.clone()), similarity);
                similarities.insert((task2.task_id.clone(), task1.task_id.clone()), similarity);
            }
        }

        self.task_similarities = Some(similarities);
        Ok(())
    }

    /// Calculate similarity between two tasks
    fn calculate_task_similarity(&self, task1: &TaskData, task2: &TaskData) -> SklResult<f64> {
        match self.config.similarity_metric {
            TaskSimilarityMetric::CorrelationBased => self.correlation_similarity(task1, task2),
            TaskSimilarityMetric::DistributionSimilarity => {
                self.distribution_similarity(task1, task2)
            }
            _ => {
                // Default to correlation-based similarity
                self.correlation_similarity(task1, task2)
            }
        }
    }

    /// Calculate correlation-based similarity between tasks
    fn correlation_similarity(&self, task1: &TaskData, task2: &TaskData) -> SklResult<f64> {
        // Simple correlation between target variables if they have the same length
        if task1.labels.len() != task2.labels.len() {
            return Ok(0.0); // No similarity if different lengths
        }

        let n = task1.labels.len();
        if n < 2 {
            return Ok(0.0);
        }

        let mean1 = task1.labels.iter().sum::<f64>() / n as f64;
        let mean2 = task2.labels.iter().sum::<f64>() / n as f64;

        let mut numerator = 0.0;
        let mut denom1 = 0.0;
        let mut denom2 = 0.0;

        for i in 0..n {
            let diff1 = task1.labels[i] - mean1;
            let diff2 = task2.labels[i] - mean2;
            numerator += diff1 * diff2;
            denom1 += diff1 * diff1;
            denom2 += diff2 * diff2;
        }

        if denom1 * denom2 > 0.0 {
            Ok(numerator / (denom1 * denom2).sqrt())
        } else {
            Ok(0.0)
        }
    }

    /// Calculate distribution similarity between tasks
    fn distribution_similarity(&self, task1: &TaskData, task2: &TaskData) -> SklResult<f64> {
        // Simple approach: compare feature means and variances
        let n_features1 = task1.features.shape()[1];
        let n_features2 = task2.features.shape()[1];

        if n_features1 != n_features2 {
            return Ok(0.0);
        }

        let mut similarity_sum = 0.0;

        for j in 0..n_features1 {
            let col1: Vec<f64> = (0..task1.features.shape()[0])
                .map(|i| task1.features[[i, j]])
                .collect();
            let col2: Vec<f64> = (0..task2.features.shape()[0])
                .map(|i| task2.features[[i, j]])
                .collect();

            let mean1 = col1.iter().sum::<f64>() / col1.len() as f64;
            let mean2 = col2.iter().sum::<f64>() / col2.len() as f64;

            let var1 = col1.iter().map(|&x| (x - mean1).powi(2)).sum::<f64>() / col1.len() as f64;
            let var2 = col2.iter().map(|&x| (x - mean2).powi(2)).sum::<f64>() / col2.len() as f64;

            // Similarity based on mean and variance differences
            let mean_sim = 1.0 - (mean1 - mean2).abs() / (mean1.abs() + mean2.abs() + 1e-8);
            let var_sim = 1.0 - (var1 - var2).abs() / (var1 + var2 + 1e-8);

            similarity_sum += (mean_sim + var_sim) / 2.0;
        }

        Ok(similarity_sum / n_features1 as f64)
    }

    /// Train independent models for each task
    fn train_independent_tasks(
        &mut self,
        tasks: &[TaskData],
    ) -> SklResult<MultiTaskTrainingResults> {
        let mut task_metrics = HashMap::new();

        // Initialize task_models
        if self.task_models.is_none() {
            self.task_models = Some(HashMap::new());
        }

        for task in tasks {
            let mut models = Vec::new();

            for _ in 0..self.config.n_estimators_per_task {
                let gb_config = GradientBoostingConfig {
                    n_estimators: 50,
                    learning_rate: 0.1,
                    max_depth: 6,
                    ..Default::default()
                };

                let y_array = Array1::from_vec(task.labels.clone());
                let model = GradientBoostingRegressor::builder()
                    .n_estimators(50)
                    .learning_rate(0.1)
                    .max_depth(6)
                    .build()
                    .fit(&task.features, &y_array)?;

                models.push(model);
            }

            // Calculate task metrics
            let predictions = self.predict_task_ensemble(&models, &task.features)?;
            let mse = self.calculate_mse(&predictions, &task.labels);

            task_metrics.insert(
                task.task_id.clone(),
                TaskMetrics {
                    training_score: mse,
                    validation_score: mse, // Would be different in practice
                    n_samples: task.features.shape()[0],
                    training_time: 0.0, // Would measure actual time
                    complexity: models.len() as f64,
                },
            );

            self.task_models
                .as_mut()
                .unwrap()
                .insert(task.task_id.clone(), models);
        }

        Ok(MultiTaskTrainingResults {
            task_metrics,
            transfer_effects: HashMap::new(),
            final_similarities: self.task_similarities.clone().unwrap_or_default(),
            convergence_info: ConvergenceInfo {
                n_iterations: 1,
                final_loss: 0.0,
                tolerance_achieved: 0.0,
                converged: true,
            },
        })
    }

    /// Train with shared representation learning
    fn train_shared_representation(
        &mut self,
        tasks: &[TaskData],
    ) -> SklResult<MultiTaskTrainingResults> {
        // First train shared models on combined data
        let combined_data = self.combine_task_data(tasks)?;

        // Initialize shared_models
        if self.shared_models.is_none() {
            self.shared_models = Some(Vec::new());
        }

        for _ in 0..self.config.n_estimators_per_task {
            let gb_config = GradientBoostingConfig {
                n_estimators: 30,
                learning_rate: 0.1,
                max_depth: 4,
                ..Default::default()
            };

            let y_array = Array1::from_vec(combined_data.labels.clone());
            let shared_model = GradientBoostingRegressor::builder()
                .n_estimators(30)
                .learning_rate(0.1)
                .max_depth(4)
                .build()
                .fit(&combined_data.features, &y_array)?;

            self.shared_models.as_mut().unwrap().push(shared_model);
        }

        // Then train task-specific models
        self.train_independent_tasks(tasks)
    }

    /// Train with parameter sharing between similar tasks
    fn train_parameter_sharing(
        &mut self,
        tasks: &[TaskData],
    ) -> SklResult<MultiTaskTrainingResults> {
        // Group similar tasks
        let task_groups = self.group_similar_tasks(tasks)?;

        let mut task_metrics = HashMap::new();

        for group in task_groups {
            // Train shared models for this group
            let group_data = self.combine_group_data(tasks, &group)?;

            let mut group_models = Vec::new();
            for _ in 0..self.config.n_estimators_per_task {
                let gb_config = GradientBoostingConfig {
                    n_estimators: 40,
                    learning_rate: 0.1,
                    max_depth: 5,
                    ..Default::default()
                };

                let y_array = Array1::from_vec(group_data.labels.clone());
                let model = GradientBoostingRegressor::builder()
                    .n_estimators(40)
                    .learning_rate(0.1)
                    .max_depth(5)
                    .build()
                    .fit(&group_data.features, &y_array)?;

                group_models.push(model);
            }

            // Calculate metrics for tasks in this group and create separate model sets
            for task_id in &group {
                let task = tasks.iter().find(|t| &t.task_id == task_id).unwrap();

                // Calculate metrics for this task
                let predictions = self.predict_task_ensemble(&group_models, &task.features)?;
                let mse = self.calculate_mse(&predictions, &task.labels);

                task_metrics.insert(
                    task_id.clone(),
                    TaskMetrics {
                        training_score: mse,
                        validation_score: mse,
                        n_samples: task.features.shape()[0],
                        training_time: 0.0,
                        complexity: group_models.len() as f64,
                    },
                );

                // For each task, train a separate set of models with the same configuration
                let mut task_models = Vec::new();
                for _ in 0..self.config.n_estimators_per_task {
                    let y_array = Array1::from_vec(group_data.labels.clone());
                    let model = GradientBoostingRegressor::builder()
                        .n_estimators(40)
                        .learning_rate(0.1)
                        .max_depth(5)
                        .build()
                        .fit(&group_data.features, &y_array)?;
                    task_models.push(model);
                }

                self.task_models
                    .as_mut()
                    .unwrap()
                    .insert(task_id.clone(), task_models);
            }
        }

        Ok(MultiTaskTrainingResults {
            task_metrics,
            transfer_effects: HashMap::new(),
            final_similarities: self.task_similarities.clone().unwrap_or_default(),
            convergence_info: ConvergenceInfo {
                n_iterations: 1,
                final_loss: 0.0,
                tolerance_achieved: 0.0,
                converged: true,
            },
        })
    }

    /// Train with hierarchical sharing
    fn train_hierarchical_sharing(
        &mut self,
        tasks: &[TaskData],
    ) -> SklResult<MultiTaskTrainingResults> {
        // For now, delegate to parameter sharing
        // In practice, this would implement hierarchical relationships
        self.train_parameter_sharing(tasks)
    }

    /// Train with adaptive sharing based on task similarities
    fn train_adaptive_sharing(
        &mut self,
        tasks: &[TaskData],
    ) -> SklResult<MultiTaskTrainingResults> {
        let mut task_metrics = HashMap::new();

        for task in tasks {
            let mut models = Vec::new();

            // Find similar tasks for this task
            let similar_tasks = self.find_similar_tasks(&task.task_id, tasks);

            if similar_tasks.len() > 1 {
                // Train with data from similar tasks
                let combined_data = self.combine_similar_task_data(tasks, &similar_tasks)?;

                for _ in 0..self.config.n_estimators_per_task {
                    let gb_config = GradientBoostingConfig {
                        n_estimators: 50,
                        learning_rate: 0.1,
                        max_depth: 6,
                        ..Default::default()
                    };

                    let y_array = Array1::from_vec(combined_data.labels.clone());
                    let model = GradientBoostingRegressor::builder()
                        .n_estimators(50)
                        .learning_rate(0.1)
                        .max_depth(6)
                        .build()
                        .fit(&combined_data.features, &y_array)?;

                    models.push(model);
                }
            } else {
                // Train independently if no similar tasks
                for _ in 0..self.config.n_estimators_per_task {
                    let gb_config = GradientBoostingConfig {
                        n_estimators: 50,
                        learning_rate: 0.1,
                        max_depth: 6,
                        ..Default::default()
                    };

                    let y_array = Array1::from_vec(task.labels.clone());
                    let model = GradientBoostingRegressor::builder()
                        .n_estimators(50)
                        .learning_rate(0.1)
                        .max_depth(6)
                        .build()
                        .fit(&task.features, &y_array)?;

                    models.push(model);
                }
            }

            // Calculate metrics
            let predictions = self.predict_task_ensemble(&models, &task.features)?;
            let mse = self.calculate_mse(&predictions, &task.labels);

            task_metrics.insert(
                task.task_id.clone(),
                TaskMetrics {
                    training_score: mse,
                    validation_score: mse,
                    n_samples: task.features.shape()[0],
                    training_time: 0.0,
                    complexity: models.len() as f64,
                },
            );

            self.task_models
                .as_mut()
                .unwrap()
                .insert(task.task_id.clone(), models);
        }

        Ok(MultiTaskTrainingResults {
            task_metrics,
            transfer_effects: HashMap::new(),
            final_similarities: self.task_similarities.clone().unwrap_or_default(),
            convergence_info: ConvergenceInfo {
                n_iterations: 1,
                final_loss: 0.0,
                tolerance_achieved: 0.0,
                converged: true,
            },
        })
    }

    /// Combine data from multiple tasks
    fn combine_task_data(&self, tasks: &[TaskData]) -> SklResult<TaskData> {
        if tasks.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No tasks to combine".to_string(),
            ));
        }

        let total_samples: usize = tasks.iter().map(|t| t.features.shape()[0]).sum();
        let n_features = tasks[0].features.shape()[1];

        let mut combined_features = Vec::with_capacity(total_samples * n_features);
        let mut combined_labels = Vec::with_capacity(total_samples);

        for task in tasks {
            for i in 0..task.features.shape()[0] {
                for j in 0..n_features {
                    combined_features.push(task.features[[i, j]]);
                }
                combined_labels.push(task.labels[i]);
            }
        }

        let features = Array2::from_shape_vec((total_samples, n_features), combined_features)?;

        Ok(TaskData {
            task_id: "combined".to_string(),
            features,
            labels: combined_labels,
            sample_weights: None,
            metadata: HashMap::new(),
        })
    }

    /// Group similar tasks together
    fn group_similar_tasks(&self, tasks: &[TaskData]) -> SklResult<Vec<Vec<String>>> {
        let mut groups = Vec::new();
        let mut assigned = vec![false; tasks.len()];

        for (i, task) in tasks.iter().enumerate() {
            if assigned[i] {
                continue;
            }

            let mut group = vec![task.task_id.clone()];
            assigned[i] = true;

            // Find similar tasks
            for (j, other_task) in tasks.iter().enumerate() {
                if i != j && !assigned[j] {
                    let similarity = self
                        .task_similarities
                        .as_ref()
                        .and_then(|similarities| {
                            similarities.get(&(task.task_id.clone(), other_task.task_id.clone()))
                        })
                        .copied()
                        .unwrap_or(0.0);

                    if similarity >= self.config.min_similarity_threshold {
                        group.push(other_task.task_id.clone());
                        assigned[j] = true;
                    }
                }
            }

            groups.push(group);
        }

        Ok(groups)
    }

    /// Combine data from a group of tasks
    fn combine_group_data(&self, tasks: &[TaskData], group: &[String]) -> SklResult<TaskData> {
        let group_tasks: Vec<TaskData> = tasks
            .iter()
            .filter(|t| group.contains(&t.task_id))
            .cloned()
            .collect();

        self.combine_task_data(&group_tasks)
    }

    /// Find tasks similar to a given task
    fn find_similar_tasks(&self, task_id: &str, tasks: &[TaskData]) -> Vec<String> {
        let mut similar_tasks = vec![task_id.to_string()];

        for task in tasks {
            if task.task_id != task_id {
                let similarity = self
                    .task_similarities
                    .as_ref()
                    .and_then(|similarities| {
                        similarities.get(&(task_id.to_string(), task.task_id.clone()))
                    })
                    .copied()
                    .unwrap_or(0.0);

                if similarity >= self.config.min_similarity_threshold {
                    similar_tasks.push(task.task_id.clone());
                }
            }
        }

        similar_tasks
    }

    /// Combine data from similar tasks
    fn combine_similar_task_data(
        &self,
        tasks: &[TaskData],
        similar_task_ids: &[String],
    ) -> SklResult<TaskData> {
        let similar_tasks: Vec<TaskData> = tasks
            .iter()
            .filter(|t| similar_task_ids.contains(&t.task_id))
            .cloned()
            .collect();

        self.combine_task_data(&similar_tasks)
    }

    /// Predict using task ensemble
    fn predict_task_ensemble(
        &self,
        models: &[TrainedGradientBoostingRegressor],
        X: &Array2<f64>,
    ) -> SklResult<Vec<f64>> {
        if models.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No models in ensemble".to_string(),
            ));
        }

        let mut predictions = vec![0.0; X.shape()[0]];

        for model in models {
            let pred = model.predict(X)?;
            for (i, &p) in pred.iter().enumerate() {
                predictions[i] += p;
            }
        }

        // Average predictions
        for p in &mut predictions {
            *p /= models.len() as f64;
        }

        Ok(predictions)
    }

    /// Calculate mean squared error
    fn calculate_mse(&self, predictions: &[f64], targets: &[f64]) -> f64 {
        if predictions.len() != targets.len() {
            return f64::INFINITY;
        }

        let sum_squared_error: f64 = predictions
            .iter()
            .zip(targets.iter())
            .map(|(&p, &t)| (p - t).powi(2))
            .sum();

        sum_squared_error / predictions.len() as f64
    }
}

impl MultiTaskEnsembleRegressor<Trained> {
    /// Predict for a specific task
    pub fn predict_task(&self, task_id: &str, X: &Array2<f64>) -> SklResult<Vec<f64>> {
        // Use task-specific models if available
        if let Some(models) = self.task_models.as_ref().and_then(|m| m.get(task_id)) {
            let mut predictions = self.predict_task_ensemble(models, X)?;

            // Add shared model predictions if available
            if let Some(shared_models) = self.shared_models.as_ref() {
                if !shared_models.is_empty() {
                    let shared_predictions = self.predict_task_ensemble(shared_models, X)?;
                    for (i, &shared_pred) in shared_predictions.iter().enumerate() {
                        predictions[i] = 0.7 * predictions[i] + 0.3 * shared_pred;
                    }
                }
            }

            Ok(predictions)
        } else {
            Err(SklearsError::InvalidInput(format!(
                "Task '{}' not found",
                task_id
            )))
        }
    }

    /// Predict using task ensemble (internal method)
    fn predict_task_ensemble(
        &self,
        models: &[TrainedGradientBoostingRegressor],
        X: &Array2<f64>,
    ) -> SklResult<Vec<f64>> {
        if models.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No models in ensemble".to_string(),
            ));
        }

        let mut predictions = vec![0.0; X.shape()[0]];

        for model in models {
            let pred = model.predict(X)?;
            for (i, &p) in pred.iter().enumerate() {
                predictions[i] += p;
            }
        }

        // Average predictions
        for p in &mut predictions {
            *p /= models.len() as f64;
        }

        Ok(predictions)
    }

    /// Get task similarities
    pub fn get_task_similarities(&self) -> &HashMap<(String, String), f64> {
        self.task_similarities.as_ref().expect("Model is trained")
    }

    /// Get task weights
    pub fn get_task_weights(&self) -> &HashMap<String, f64> {
        self.task_weights.as_ref().expect("Model is trained")
    }
}

pub struct MultiTaskEnsembleRegressorBuilder {
    config: MultiTaskEnsembleConfig,
}

impl Default for MultiTaskEnsembleRegressorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiTaskEnsembleRegressorBuilder {
    pub fn new() -> Self {
        Self {
            config: MultiTaskEnsembleConfig::default(),
        }
    }

    pub fn config(mut self, config: MultiTaskEnsembleConfig) -> Self {
        self.config = config;
        self
    }

    pub fn n_estimators_per_task(mut self, n_estimators: usize) -> Self {
        self.config.n_estimators_per_task = n_estimators;
        self
    }

    pub fn sharing_strategy(mut self, strategy: TaskSharingStrategy) -> Self {
        self.config.sharing_strategy = strategy;
        self
    }

    pub fn min_similarity_threshold(mut self, threshold: f64) -> Self {
        self.config.min_similarity_threshold = threshold;
        self
    }

    pub fn build(self) -> MultiTaskEnsembleRegressor {
        MultiTaskEnsembleRegressor::new(self.config)
    }
}

impl Estimator for MultiTaskEnsembleRegressor {
    type Config = MultiTaskEnsembleConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_multi_task_config() {
        let config = MultiTaskEnsembleConfig::builder()
            .n_estimators_per_task(5)
            .sharing_strategy(TaskSharingStrategy::SharedRepresentation)
            .min_similarity_threshold(0.5)
            .build();

        assert_eq!(config.n_estimators_per_task, 5);
        assert_eq!(
            config.sharing_strategy,
            TaskSharingStrategy::SharedRepresentation
        );
        assert_eq!(config.min_similarity_threshold, 0.5);
    }

    #[test]
    fn test_task_data_creation() {
        let features = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let labels = vec![1.0, 2.0, 3.0];

        let task = TaskData {
            task_id: "test_task".to_string(),
            features,
            labels,
            sample_weights: None,
            metadata: HashMap::new(),
        };

        assert_eq!(task.task_id, "test_task");
        assert_eq!(task.features.shape(), &[3, 2]);
        assert_eq!(task.labels.len(), 3);
    }

    #[test]
    fn test_multi_task_ensemble_basic() {
        let config = MultiTaskEnsembleConfig::builder()
            .n_estimators_per_task(2)
            .sharing_strategy(TaskSharingStrategy::Independent)
            .build();

        let ensemble = MultiTaskEnsembleRegressor::new(config);

        // Test basic configuration
        assert_eq!(ensemble.config.n_estimators_per_task, 2);
        assert_eq!(
            ensemble.config.sharing_strategy,
            TaskSharingStrategy::Independent
        );
        // In untrained state, models should be None
        assert!(ensemble.task_models.is_none());
    }

    #[test]
    fn test_task_similarity_calculation() {
        let config = MultiTaskEnsembleConfig::default();
        let ensemble = MultiTaskEnsembleRegressor::new(config);

        let task1 = TaskData {
            task_id: "task1".to_string(),
            features: Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            labels: vec![1.0, 2.0],
            sample_weights: None,
            metadata: HashMap::new(),
        };

        let task2 = TaskData {
            task_id: "task2".to_string(),
            features: Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            labels: vec![1.0, 2.0], // Same labels as task1
            sample_weights: None,
            metadata: HashMap::new(),
        };

        let similarity = ensemble.correlation_similarity(&task1, &task2).unwrap();
        assert!((similarity - 1.0).abs() < 1e-10); // Should be perfectly correlated
    }
}
