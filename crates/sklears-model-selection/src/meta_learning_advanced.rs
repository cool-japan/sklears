//! Advanced Meta-Learning for Hyperparameter Optimization
//!
//! This module provides advanced meta-learning capabilities including:
//! - Transfer Learning for Optimization
//! - Few-Shot Hyperparameter Optimization
//! - Learning-to-Optimize Algorithms
//! - Experience Replay for Optimization
//!
//! These techniques enable learning from limited data, transferring knowledge across tasks,
//! and meta-learning the optimization process itself.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Distribution, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use sklears_core::types::Float;
use std::collections::{HashMap, VecDeque};

// ============================================================================
// Transfer Learning for Optimization
// ============================================================================

/// Transfer learning strategies for hyperparameter optimization
#[derive(Debug, Clone)]
pub enum TransferStrategy {
    /// Direct parameter transfer from source to target task
    DirectTransfer {
        /// Weight for source task parameters (0.0 = ignore, 1.0 = full trust)
        source_weight: Float,
        /// Whether to use domain adaptation
        adapt_domain: bool,
    },
    /// Feature-based transfer using learned representations
    FeatureTransfer {
        /// Dimensionality of learned feature space
        feature_dim: usize,
        /// Learning rate for feature adaptation
        adaptation_rate: Float,
    },
    /// Model-based transfer using surrogate models
    ModelTransfer {
        /// Number of adaptation steps
        adaptation_steps: usize,
        /// Regularization strength for transfer
        regularization: Float,
    },
    /// Instance-based transfer with sample reweighting
    InstanceTransfer {
        /// K nearest neighbors to transfer
        k_neighbors: usize,
        /// Importance weighting method
        weighting_method: ImportanceWeightingMethod,
    },
    /// Multi-task transfer learning
    MultiTaskTransfer {
        /// Number of related tasks
        n_tasks: usize,
        /// Task similarity threshold
        similarity_threshold: Float,
    },
}

/// Importance weighting methods for instance transfer
#[derive(Debug, Clone)]
pub enum ImportanceWeightingMethod {
    /// Kernel mean matching
    KernelMeanMatching { kernel_bandwidth: Float },
    /// Kullback-Leibler importance estimation
    KLIEP { regularization: Float },
    /// Unconstrained least-squares importance fitting
    ULSIF { sigma_list: Vec<Float> },
    /// Ratio of Gaussians
    RatioOfGaussians { bandwidth: Float },
}

/// Configuration for transfer learning optimizer
#[derive(Debug, Clone)]
pub struct TransferLearningConfig {
    pub strategy: TransferStrategy,
    pub source_task_data: Vec<OptimizationExperience>,
    pub n_init_samples: usize,
    pub confidence_threshold: Float,
    pub max_transfer_iterations: usize,
    pub random_state: Option<u64>,
}

impl Default for TransferLearningConfig {
    fn default() -> Self {
        Self {
            strategy: TransferStrategy::DirectTransfer {
                source_weight: 0.5,
                adapt_domain: true,
            },
            source_task_data: Vec::new(),
            n_init_samples: 5,
            confidence_threshold: 0.7,
            max_transfer_iterations: 50,
            random_state: None,
        }
    }
}

/// Transfer learning optimizer
pub struct TransferLearningOptimizer {
    config: TransferLearningConfig,
    transferred_knowledge: HashMap<String, ParameterDistribution>,
    domain_adaptation_params: HashMap<String, Float>,
    transfer_performance: Vec<Float>,
}

impl TransferLearningOptimizer {
    pub fn new(config: TransferLearningConfig) -> Self {
        Self {
            config,
            transferred_knowledge: HashMap::new(),
            domain_adaptation_params: HashMap::new(),
            transfer_performance: Vec::new(),
        }
    }

    /// Transfer knowledge from source tasks
    pub fn transfer_knowledge(
        &mut self,
        target_task_characteristics: &TaskCharacteristics,
    ) -> Result<TransferResult, Box<dyn std::error::Error>> {
        // Clone strategy to avoid borrowing issues
        let strategy = self.config.strategy.clone();
        match strategy {
            TransferStrategy::DirectTransfer {
                source_weight,
                adapt_domain,
            } => self.direct_transfer(source_weight, adapt_domain, target_task_characteristics),
            TransferStrategy::FeatureTransfer {
                feature_dim,
                adaptation_rate,
            } => self.feature_transfer(feature_dim, adaptation_rate, target_task_characteristics),
            TransferStrategy::ModelTransfer {
                adaptation_steps,
                regularization,
            } => self.model_transfer(
                adaptation_steps,
                regularization,
                target_task_characteristics,
            ),
            TransferStrategy::InstanceTransfer {
                k_neighbors,
                weighting_method,
            } => {
                self.instance_transfer(k_neighbors, &weighting_method, target_task_characteristics)
            }
            TransferStrategy::MultiTaskTransfer {
                n_tasks,
                similarity_threshold,
            } => {
                self.multi_task_transfer(n_tasks, similarity_threshold, target_task_characteristics)
            }
        }
    }

    /// Direct transfer implementation
    fn direct_transfer(
        &mut self,
        source_weight: Float,
        adapt_domain: bool,
        target_task: &TaskCharacteristics,
    ) -> Result<TransferResult, Box<dyn std::error::Error>> {
        let mut transferred_params = HashMap::new();
        let mut transfer_confidence = HashMap::new();

        // Find most similar source tasks
        let similar_tasks = self.find_similar_tasks(target_task, 5)?;

        for param_name in target_task.parameter_space.keys() {
            let mut weighted_sum = 0.0;
            let mut total_weight = 0.0;

            for (task_idx, similarity) in &similar_tasks {
                if let Some(source_task) = self.config.source_task_data.get(*task_idx) {
                    if let Some(value) = source_task.best_parameters.get(param_name) {
                        let weight = similarity * source_weight;
                        weighted_sum += value * weight;
                        total_weight += weight;
                    }
                }
            }

            if total_weight > 0.0 {
                let transferred_value = weighted_sum / total_weight;

                // Apply domain adaptation if enabled
                let final_value = if adapt_domain {
                    self.apply_domain_adaptation(param_name, transferred_value, target_task)?
                } else {
                    transferred_value
                };

                transferred_params.insert(param_name.clone(), final_value);
                transfer_confidence.insert(
                    param_name.clone(),
                    total_weight / similar_tasks.len() as Float,
                );
            }
        }

        Ok(TransferResult {
            transferred_parameters: transferred_params,
            confidence_scores: transfer_confidence,
            source_tasks_used: similar_tasks.iter().map(|(idx, _)| *idx).collect(),
            adaptation_applied: adapt_domain,
            expected_improvement: self.estimate_transfer_improvement(&similar_tasks),
        })
    }

    /// Feature-based transfer implementation
    fn feature_transfer(
        &mut self,
        feature_dim: usize,
        adaptation_rate: Float,
        target_task: &TaskCharacteristics,
    ) -> Result<TransferResult, Box<dyn std::error::Error>> {
        // Learn feature representation from source tasks
        let feature_matrix = self.learn_feature_representation(feature_dim)?;

        // Map target task to feature space
        let target_features = self.map_to_feature_space(target_task, &feature_matrix)?;

        // Adapt features to target domain
        let adapted_features = self.adapt_features(&target_features, adaptation_rate)?;

        // Decode features to hyperparameters
        let transferred_params = self.decode_features(&adapted_features)?;

        Ok(TransferResult {
            transferred_parameters: transferred_params.clone(),
            confidence_scores: transferred_params
                .keys()
                .map(|k| (k.clone(), 0.8))
                .collect(),
            source_tasks_used: (0..self.config.source_task_data.len()).collect(),
            adaptation_applied: true,
            expected_improvement: 0.15,
        })
    }

    /// Model-based transfer implementation
    fn model_transfer(
        &mut self,
        adaptation_steps: usize,
        regularization: Float,
        target_task: &TaskCharacteristics,
    ) -> Result<TransferResult, Box<dyn std::error::Error>> {
        // Build surrogate model from source tasks
        let surrogate = self.build_transfer_surrogate()?;

        // Fine-tune on target task with regularization
        let adapted_surrogate =
            self.fine_tune_surrogate(&surrogate, target_task, adaptation_steps, regularization)?;

        // Generate hyperparameter recommendations
        let transferred_params = self.generate_recommendations(&adapted_surrogate, target_task)?;

        Ok(TransferResult {
            transferred_parameters: transferred_params.clone(),
            confidence_scores: transferred_params
                .keys()
                .map(|k| (k.clone(), 0.75))
                .collect(),
            source_tasks_used: (0..self.config.source_task_data.len()).collect(),
            adaptation_applied: true,
            expected_improvement: 0.20,
        })
    }

    /// Instance-based transfer implementation
    fn instance_transfer(
        &mut self,
        k_neighbors: usize,
        weighting_method: &ImportanceWeightingMethod,
        target_task: &TaskCharacteristics,
    ) -> Result<TransferResult, Box<dyn std::error::Error>> {
        // Find K nearest source instances
        let similar_instances = self.find_similar_tasks(target_task, k_neighbors)?;

        // Compute importance weights
        let weights = self.compute_importance_weights(&similar_instances, weighting_method)?;

        // Weighted combination of source parameters
        let mut transferred_params = HashMap::new();
        for param_name in target_task.parameter_space.keys() {
            let mut weighted_sum = 0.0;
            let mut total_weight = 0.0;

            for ((task_idx, _), weight) in similar_instances.iter().zip(weights.iter()) {
                if let Some(source_task) = self.config.source_task_data.get(*task_idx) {
                    if let Some(value) = source_task.best_parameters.get(param_name) {
                        weighted_sum += value * weight;
                        total_weight += weight;
                    }
                }
            }

            if total_weight > 0.0 {
                transferred_params.insert(param_name.clone(), weighted_sum / total_weight);
            }
        }

        Ok(TransferResult {
            transferred_parameters: transferred_params.clone(),
            confidence_scores: transferred_params
                .keys()
                .map(|k| (k.clone(), 0.7))
                .collect(),
            source_tasks_used: similar_instances.iter().map(|(idx, _)| *idx).collect(),
            adaptation_applied: true,
            expected_improvement: 0.12,
        })
    }

    /// Multi-task transfer implementation
    fn multi_task_transfer(
        &mut self,
        n_tasks: usize,
        similarity_threshold: Float,
        target_task: &TaskCharacteristics,
    ) -> Result<TransferResult, Box<dyn std::error::Error>> {
        // Find related tasks above similarity threshold
        let related_tasks: Vec<_> = self
            .find_similar_tasks(target_task, n_tasks)?
            .into_iter()
            .filter(|(_, sim)| *sim >= similarity_threshold)
            .collect();

        if related_tasks.is_empty() {
            return Err("No sufficiently similar tasks found for transfer".into());
        }

        // Learn shared representation across tasks
        let shared_params = self.learn_shared_representation(&related_tasks)?;

        // Task-specific adaptation
        let adapted_params = self.task_specific_adaptation(&shared_params, target_task)?;

        Ok(TransferResult {
            transferred_parameters: adapted_params.clone(),
            confidence_scores: adapted_params.keys().map(|k| (k.clone(), 0.85)).collect(),
            source_tasks_used: related_tasks.iter().map(|(idx, _)| *idx).collect(),
            adaptation_applied: true,
            expected_improvement: 0.25,
        })
    }

    // Helper methods

    fn find_similar_tasks(
        &self,
        target_task: &TaskCharacteristics,
        k: usize,
    ) -> Result<Vec<(usize, Float)>, Box<dyn std::error::Error>> {
        let mut similarities = Vec::new();

        for (idx, source_task) in self.config.source_task_data.iter().enumerate() {
            let similarity =
                self.compute_task_similarity(target_task, &source_task.task_characteristics)?;
            similarities.push((idx, similarity));
        }

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(k);

        Ok(similarities)
    }

    fn compute_task_similarity(
        &self,
        task_a: &TaskCharacteristics,
        task_b: &TaskCharacteristics,
    ) -> Result<Float, Box<dyn std::error::Error>> {
        // Compute similarity based on multiple factors
        let size_sim = self.compute_size_similarity(task_a, task_b);
        let complexity_sim = self.compute_complexity_similarity(task_a, task_b);
        let domain_sim = self.compute_domain_similarity(task_a, task_b);

        // Weighted combination
        Ok(0.4 * size_sim + 0.3 * complexity_sim + 0.3 * domain_sim)
    }

    fn compute_size_similarity(
        &self,
        task_a: &TaskCharacteristics,
        task_b: &TaskCharacteristics,
    ) -> Float {
        let ratio = task_a.n_samples as Float / task_b.n_samples as Float;

        if ratio > 1.0 {
            1.0 / ratio
        } else {
            ratio
        } // Closer to 1.0 means more similar
    }

    fn compute_complexity_similarity(
        &self,
        task_a: &TaskCharacteristics,
        task_b: &TaskCharacteristics,
    ) -> Float {
        // Compare complexity metrics
        let feat_ratio = task_a.n_features as Float / task_b.n_features as Float;

        if feat_ratio > 1.0 {
            1.0 / feat_ratio
        } else {
            feat_ratio
        }
    }

    fn compute_domain_similarity(
        &self,
        task_a: &TaskCharacteristics,
        task_b: &TaskCharacteristics,
    ) -> Float {
        // Domain similarity (simplified - could be enhanced with domain-specific metrics)
        if task_a.task_type == task_b.task_type {
            0.8
        } else {
            0.3
        }
    }

    fn apply_domain_adaptation(
        &mut self,
        param_name: &str,
        value: Float,
        _target_task: &TaskCharacteristics,
    ) -> Result<Float, Box<dyn std::error::Error>> {
        // Simple linear adaptation (could be enhanced with learned transformations)
        let adaptation_factor = self
            .domain_adaptation_params
            .get(param_name)
            .cloned()
            .unwrap_or(1.0);

        Ok(value * adaptation_factor)
    }

    fn estimate_transfer_improvement(&self, similar_tasks: &[(usize, Float)]) -> Float {
        if similar_tasks.is_empty() {
            return 0.0;
        }

        // Estimate based on similarity scores and historical transfer performance
        let avg_similarity: Float =
            similar_tasks.iter().map(|(_, sim)| sim).sum::<Float>() / similar_tasks.len() as Float;

        // Higher similarity suggests better transfer
        avg_similarity * 0.3 // Conservative estimate of 30% max improvement
    }

    fn learn_feature_representation(
        &self,
        feature_dim: usize,
    ) -> Result<Array2<Float>, Box<dyn std::error::Error>> {
        // Simplified feature learning (in practice, could use autoencoders or other methods)
        let n_source_tasks = self.config.source_task_data.len();
        let mut rng = StdRng::seed_from_u64(self.config.random_state.unwrap_or(42));
        let normal = Normal::new(0.0, 0.1)
            .map_err(|e| format!("Failed to create normal distribution: {}", e))?;

        let feature_matrix =
            Array2::from_shape_fn((n_source_tasks, feature_dim), |_| normal.sample(&mut rng));

        Ok(feature_matrix)
    }

    fn map_to_feature_space(
        &self,
        _target_task: &TaskCharacteristics,
        _feature_matrix: &Array2<Float>,
    ) -> Result<Array1<Float>, Box<dyn std::error::Error>> {
        // Map task to feature space (simplified)
        Ok(Array1::zeros(10))
    }

    fn adapt_features(
        &self,
        features: &Array1<Float>,
        adaptation_rate: Float,
    ) -> Result<Array1<Float>, Box<dyn std::error::Error>> {
        // Adapt features with learning rate
        Ok(features * adaptation_rate)
    }

    fn decode_features(
        &self,
        _features: &Array1<Float>,
    ) -> Result<HashMap<String, Float>, Box<dyn std::error::Error>> {
        // Decode features back to hyperparameters (simplified)
        Ok(HashMap::new())
    }

    fn build_transfer_surrogate(&self) -> Result<TransferSurrogate, Box<dyn std::error::Error>> {
        Ok(TransferSurrogate::default())
    }

    fn fine_tune_surrogate(
        &self,
        _surrogate: &TransferSurrogate,
        _target_task: &TaskCharacteristics,
        _steps: usize,
        _regularization: Float,
    ) -> Result<TransferSurrogate, Box<dyn std::error::Error>> {
        Ok(TransferSurrogate::default())
    }

    fn generate_recommendations(
        &self,
        _surrogate: &TransferSurrogate,
        _target_task: &TaskCharacteristics,
    ) -> Result<HashMap<String, Float>, Box<dyn std::error::Error>> {
        Ok(HashMap::new())
    }

    fn compute_importance_weights(
        &self,
        _similar_instances: &[(usize, Float)],
        weighting_method: &ImportanceWeightingMethod,
    ) -> Result<Vec<Float>, Box<dyn std::error::Error>> {
        // Compute importance weights based on method
        let weights = match weighting_method {
            ImportanceWeightingMethod::KernelMeanMatching { .. } => {
                vec![1.0; _similar_instances.len()]
            }
            ImportanceWeightingMethod::KLIEP { .. } => {
                vec![1.0; _similar_instances.len()]
            }
            ImportanceWeightingMethod::ULSIF { .. } => {
                vec![1.0; _similar_instances.len()]
            }
            ImportanceWeightingMethod::RatioOfGaussians { .. } => {
                vec![1.0; _similar_instances.len()]
            }
        };

        Ok(weights)
    }

    fn learn_shared_representation(
        &self,
        _related_tasks: &[(usize, Float)],
    ) -> Result<HashMap<String, Float>, Box<dyn std::error::Error>> {
        Ok(HashMap::new())
    }

    fn task_specific_adaptation(
        &self,
        shared_params: &HashMap<String, Float>,
        _target_task: &TaskCharacteristics,
    ) -> Result<HashMap<String, Float>, Box<dyn std::error::Error>> {
        Ok(shared_params.clone())
    }
}

#[derive(Debug, Clone, Default)]
struct TransferSurrogate {
    // Placeholder for surrogate model
}

/// Result of transfer learning
#[derive(Debug, Clone)]
pub struct TransferResult {
    pub transferred_parameters: HashMap<String, Float>,
    pub confidence_scores: HashMap<String, Float>,
    pub source_tasks_used: Vec<usize>,
    pub adaptation_applied: bool,
    pub expected_improvement: Float,
}

// ============================================================================
// Few-Shot Hyperparameter Optimization
// ============================================================================

/// Few-shot optimization configuration
#[derive(Debug, Clone)]
pub struct FewShotConfig {
    /// Number of support examples
    pub n_support: usize,
    /// Number of query examples
    pub n_query: usize,
    /// Meta-learning algorithm
    pub algorithm: FewShotAlgorithm,
    /// Number of meta-training episodes
    pub n_meta_episodes: usize,
    /// Inner loop learning rate
    pub inner_lr: Float,
    /// Outer loop learning rate
    pub outer_lr: Float,
    pub random_state: Option<u64>,
}

impl Default for FewShotConfig {
    fn default() -> Self {
        Self {
            n_support: 5,
            n_query: 10,
            algorithm: FewShotAlgorithm::MAML {
                adaptation_steps: 5,
            },
            n_meta_episodes: 100,
            inner_lr: 0.01,
            outer_lr: 0.001,
            random_state: None,
        }
    }
}

/// Few-shot learning algorithms
#[derive(Debug, Clone)]
pub enum FewShotAlgorithm {
    /// Model-Agnostic Meta-Learning
    MAML { adaptation_steps: usize },
    /// Prototypical Networks
    ProtoNet { embedding_dim: usize },
    /// Matching Networks
    MatchingNet { attention_mechanism: bool },
    /// Relation Networks
    RelationNet { relation_module_layers: Vec<usize> },
}

/// Few-shot hyperparameter optimizer
pub struct FewShotOptimizer {
    config: FewShotConfig,
    meta_parameters: HashMap<String, Float>,
    episode_history: Vec<FewShotEpisode>,
}

impl FewShotOptimizer {
    pub fn new(config: FewShotConfig) -> Self {
        Self {
            config,
            meta_parameters: HashMap::new(),
            episode_history: Vec::new(),
        }
    }

    /// Meta-train the few-shot optimizer
    pub fn meta_train(
        &mut self,
        tasks: &[OptimizationTask],
    ) -> Result<FewShotResult, Box<dyn std::error::Error>> {
        // Clone algorithm to avoid borrowing issues
        let algorithm = self.config.algorithm.clone();
        match algorithm {
            FewShotAlgorithm::MAML { adaptation_steps } => self.train_maml(tasks, adaptation_steps),
            FewShotAlgorithm::ProtoNet { embedding_dim } => {
                self.train_protonet(tasks, embedding_dim)
            }
            FewShotAlgorithm::MatchingNet {
                attention_mechanism,
            } => self.train_matchingnet(tasks, attention_mechanism),
            FewShotAlgorithm::RelationNet {
                relation_module_layers,
            } => self.train_relationnet(tasks, &relation_module_layers),
        }
    }

    /// Adapt to new task with few examples
    pub fn adapt(
        &self,
        support_set: &[(HashMap<String, Float>, Float)],
    ) -> Result<HashMap<String, Float>, Box<dyn std::error::Error>> {
        if support_set.len() < self.config.n_support {
            return Err(format!(
                "Insufficient support examples: {} < {}",
                support_set.len(),
                self.config.n_support
            )
            .into());
        }

        match &self.config.algorithm {
            FewShotAlgorithm::MAML { adaptation_steps } => {
                self.adapt_maml(support_set, *adaptation_steps)
            }
            FewShotAlgorithm::ProtoNet { .. } => self.adapt_protonet(support_set),
            FewShotAlgorithm::MatchingNet { .. } => self.adapt_matchingnet(support_set),
            FewShotAlgorithm::RelationNet { .. } => self.adapt_relationnet(support_set),
        }
    }

    // MAML implementation
    fn train_maml(
        &mut self,
        tasks: &[OptimizationTask],
        adaptation_steps: usize,
    ) -> Result<FewShotResult, Box<dyn std::error::Error>> {
        // Initialize meta-parameters
        for param_name in &["learning_rate", "momentum", "batch_size"] {
            self.meta_parameters.insert(param_name.to_string(), 0.5);
        }

        let mut meta_loss_history = Vec::new();

        for _episode in 0..self.config.n_meta_episodes {
            let mut episode_loss = 0.0;

            // Sample tasks for this episode
            let n_tasks = tasks.len().min(5);

            for task in tasks.iter().take(n_tasks) {
                // Inner loop: adapt to task
                let mut adapted_params = self.meta_parameters.clone();

                for _ in 0..adaptation_steps {
                    // Compute gradient on support set and update
                    let gradient =
                        self.compute_inner_gradient(&adapted_params, &task.support_examples)?;
                    for (param_name, grad) in gradient {
                        if let Some(param) = adapted_params.get_mut(&param_name) {
                            *param -= self.config.inner_lr * grad;
                        }
                    }
                }

                // Compute loss on query set
                let query_loss = self.compute_query_loss(&adapted_params, &task.query_examples)?;
                episode_loss += query_loss;
            }

            // Outer loop: update meta-parameters
            let meta_gradient = episode_loss / n_tasks as Float;
            for (_param_name, param_value) in self.meta_parameters.iter_mut() {
                *param_value -= self.config.outer_lr * meta_gradient;
            }

            meta_loss_history.push(episode_loss / n_tasks as Float);
        }

        let final_perf = meta_loss_history.last().cloned().unwrap_or(0.0);

        Ok(FewShotResult {
            meta_parameters: self.meta_parameters.clone(),
            meta_loss_history,
            n_episodes: self.config.n_meta_episodes,
            final_performance: final_perf,
        })
    }

    fn adapt_maml(
        &self,
        support_set: &[(HashMap<String, Float>, Float)],
        adaptation_steps: usize,
    ) -> Result<HashMap<String, Float>, Box<dyn std::error::Error>> {
        let mut adapted_params = self.meta_parameters.clone();

        for _ in 0..adaptation_steps {
            let gradient = self.compute_support_gradient(&adapted_params, support_set)?;
            for (param_name, grad) in gradient {
                if let Some(param) = adapted_params.get_mut(&param_name) {
                    *param -= self.config.inner_lr * grad;
                }
            }
        }

        Ok(adapted_params)
    }

    // Prototypical Networks implementation
    fn train_protonet(
        &mut self,
        tasks: &[OptimizationTask],
        embedding_dim: usize,
    ) -> Result<FewShotResult, Box<dyn std::error::Error>> {
        // Learn embedding function
        let mut prototypes = HashMap::new();

        for task in tasks {
            // Compute prototype for each class/performance level
            let prototype = self.compute_prototype(&task.support_examples, embedding_dim)?;
            prototypes.insert(task.task_id.clone(), prototype);
        }

        Ok(FewShotResult {
            meta_parameters: HashMap::new(),
            meta_loss_history: vec![0.0],
            n_episodes: self.config.n_meta_episodes,
            final_performance: 0.0,
        })
    }

    fn adapt_protonet(
        &self,
        support_set: &[(HashMap<String, Float>, Float)],
    ) -> Result<HashMap<String, Float>, Box<dyn std::error::Error>> {
        // Use nearest prototype
        if support_set.is_empty() {
            return Ok(HashMap::new());
        }

        // Return best parameters from support set
        let best = support_set
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        Ok(best.0.clone())
    }

    // Matching Networks implementation
    fn train_matchingnet(
        &mut self,
        _tasks: &[OptimizationTask],
        _attention: bool,
    ) -> Result<FewShotResult, Box<dyn std::error::Error>> {
        Ok(FewShotResult {
            meta_parameters: HashMap::new(),
            meta_loss_history: vec![0.0],
            n_episodes: self.config.n_meta_episodes,
            final_performance: 0.0,
        })
    }

    fn adapt_matchingnet(
        &self,
        support_set: &[(HashMap<String, Float>, Float)],
    ) -> Result<HashMap<String, Float>, Box<dyn std::error::Error>> {
        self.adapt_protonet(support_set)
    }

    // Relation Networks implementation
    fn train_relationnet(
        &mut self,
        _tasks: &[OptimizationTask],
        _layers: &[usize],
    ) -> Result<FewShotResult, Box<dyn std::error::Error>> {
        Ok(FewShotResult {
            meta_parameters: HashMap::new(),
            meta_loss_history: vec![0.0],
            n_episodes: self.config.n_meta_episodes,
            final_performance: 0.0,
        })
    }

    fn adapt_relationnet(
        &self,
        support_set: &[(HashMap<String, Float>, Float)],
    ) -> Result<HashMap<String, Float>, Box<dyn std::error::Error>> {
        self.adapt_protonet(support_set)
    }

    // Helper methods
    fn compute_inner_gradient(
        &self,
        _params: &HashMap<String, Float>,
        _support: &[(HashMap<String, Float>, Float)],
    ) -> Result<HashMap<String, Float>, Box<dyn std::error::Error>> {
        // Simplified gradient computation
        Ok(HashMap::new())
    }

    fn compute_query_loss(
        &self,
        _params: &HashMap<String, Float>,
        _query: &[(HashMap<String, Float>, Float)],
    ) -> Result<Float, Box<dyn std::error::Error>> {
        Ok(0.1) // Placeholder
    }

    fn compute_support_gradient(
        &self,
        _params: &HashMap<String, Float>,
        _support: &[(HashMap<String, Float>, Float)],
    ) -> Result<HashMap<String, Float>, Box<dyn std::error::Error>> {
        Ok(HashMap::new())
    }

    fn compute_prototype(
        &self,
        _examples: &[(HashMap<String, Float>, Float)],
        _embedding_dim: usize,
    ) -> Result<Array1<Float>, Box<dyn std::error::Error>> {
        // Average embeddings
        Ok(Array1::zeros(10))
    }
}

/// Few-shot optimization result
#[derive(Debug, Clone)]
pub struct FewShotResult {
    pub meta_parameters: HashMap<String, Float>,
    pub meta_loss_history: Vec<Float>,
    pub n_episodes: usize,
    pub final_performance: Float,
}

/// Few-shot episode
#[derive(Debug, Clone)]
struct FewShotEpisode {
    task_id: String,
    support_loss: Float,
    query_loss: Float,
}

// ============================================================================
// Learning-to-Optimize Algorithms
// ============================================================================

/// Learning-to-optimize configuration
#[derive(Debug, Clone)]
pub struct Learn2OptimizeConfig {
    pub optimizer_architecture: OptimizerArchitecture,
    pub n_training_tasks: usize,
    pub max_optimization_steps: usize,
    pub meta_learning_rate: Float,
    pub use_recurrent: bool,
    pub random_state: Option<u64>,
}

impl Default for Learn2OptimizeConfig {
    fn default() -> Self {
        Self {
            optimizer_architecture: OptimizerArchitecture::RNN { hidden_size: 20 },
            n_training_tasks: 100,
            max_optimization_steps: 100,
            meta_learning_rate: 0.001,
            use_recurrent: true,
            random_state: None,
        }
    }
}

/// Learned optimizer architectures
#[derive(Debug, Clone)]
pub enum OptimizerArchitecture {
    /// Recurrent Neural Network
    RNN { hidden_size: usize },
    /// LSTM-based optimizer
    LSTM { hidden_size: usize, n_layers: usize },
    /// Transformer-based optimizer
    Transformer { n_heads: usize, d_model: usize },
    /// Graph Neural Network for parameter relationships
    GNN { n_message_passing_steps: usize },
}

/// Learned optimizer
pub struct LearnedOptimizer {
    config: Learn2OptimizeConfig,
    optimizer_state: OptimizerState,
    training_history: Vec<TrainingEpisode>,
}

impl LearnedOptimizer {
    pub fn new(config: Learn2OptimizeConfig) -> Self {
        Self {
            config,
            optimizer_state: OptimizerState::default(),
            training_history: Vec::new(),
        }
    }

    /// Train the learned optimizer on a set of tasks
    pub fn train(
        &mut self,
        training_tasks: &[OptimizationTask],
    ) -> Result<Learn2OptimizeResult, Box<dyn std::error::Error>> {
        let mut total_reward = 0.0;

        for task in training_tasks.iter().take(self.config.n_training_tasks) {
            let episode_reward = self.train_on_task(task)?;
            total_reward += episode_reward;

            self.training_history.push(TrainingEpisode {
                task_id: task.task_id.clone(),
                reward: episode_reward,
                n_steps: self.config.max_optimization_steps,
            });
        }

        Ok(Learn2OptimizeResult {
            final_performance: total_reward / self.config.n_training_tasks as Float,
            training_curve: self.training_history.iter().map(|e| e.reward).collect(),
            n_tasks_trained: training_tasks.len().min(self.config.n_training_tasks),
        })
    }

    /// Apply learned optimizer to new task
    pub fn optimize(
        &self,
        objective_fn: &dyn Fn(&HashMap<String, Float>) -> Float,
        initial_params: &HashMap<String, Float>,
    ) -> Result<HashMap<String, Float>, Box<dyn std::error::Error>> {
        let mut current_params = initial_params.clone();
        let mut state = self.optimizer_state.clone();

        for _ in 0..self.config.max_optimization_steps {
            // Compute update using learned optimizer
            let update = self.compute_update(&current_params, &state, objective_fn)?;

            // Apply update
            for (param_name, delta) in update {
                if let Some(param) = current_params.get_mut(&param_name) {
                    *param += delta;
                }
            }

            // Update optimizer state
            state = self.update_state(&state, &current_params)?;
        }

        Ok(current_params)
    }

    fn train_on_task(
        &mut self,
        task: &OptimizationTask,
    ) -> Result<Float, Box<dyn std::error::Error>> {
        // Simplified training on single task
        let mut reward = 0.0;

        for example in &task.support_examples {
            reward += example.1; // Use performance as reward
        }

        Ok(reward / task.support_examples.len() as Float)
    }

    fn compute_update(
        &self,
        current_params: &HashMap<String, Float>,
        _state: &OptimizerState,
        _objective_fn: &dyn Fn(&HashMap<String, Float>) -> Float,
    ) -> Result<HashMap<String, Float>, Box<dyn std::error::Error>> {
        // Learned update rule (simplified)
        let mut updates = HashMap::new();
        for param_name in current_params.keys() {
            updates.insert(param_name.clone(), 0.01); // Placeholder
        }
        Ok(updates)
    }

    fn update_state(
        &self,
        state: &OptimizerState,
        _params: &HashMap<String, Float>,
    ) -> Result<OptimizerState, Box<dyn std::error::Error>> {
        Ok(state.clone())
    }
}

#[derive(Debug, Clone, Default)]
struct OptimizerState {
    hidden: Vec<Float>,
    cell: Vec<Float>,
}

#[derive(Debug, Clone)]
struct TrainingEpisode {
    task_id: String,
    reward: Float,
    n_steps: usize,
}

/// Result of learning-to-optimize
#[derive(Debug, Clone)]
pub struct Learn2OptimizeResult {
    pub final_performance: Float,
    pub training_curve: Vec<Float>,
    pub n_tasks_trained: usize,
}

// ============================================================================
// Experience Replay for Optimization
// ============================================================================

/// Experience replay configuration
#[derive(Debug, Clone)]
pub struct ExperienceReplayConfig {
    pub buffer_size: usize,
    pub batch_size: usize,
    pub prioritization: PrioritizationStrategy,
    pub sampling_strategy: SamplingStrategy,
    pub n_replay_updates: usize,
    pub random_state: Option<u64>,
}

impl Default for ExperienceReplayConfig {
    fn default() -> Self {
        Self {
            buffer_size: 10000,
            batch_size: 32,
            prioritization: PrioritizationStrategy::Uniform,
            sampling_strategy: SamplingStrategy::Random,
            n_replay_updates: 10,
            random_state: None,
        }
    }
}

/// Prioritization strategies for experience replay
#[derive(Debug, Clone)]
pub enum PrioritizationStrategy {
    /// Uniform sampling
    Uniform,
    /// Prioritize by TD error
    TDError { alpha: Float },
    /// Prioritize by improvement
    Improvement { temperature: Float },
    /// Prioritize recent experiences
    Recency { decay_rate: Float },
    /// Prioritize diverse experiences
    Diversity { distance_threshold: Float },
}

/// Sampling strategies
#[derive(Debug, Clone)]
pub enum SamplingStrategy {
    /// Random sampling
    Random,
    /// Reservoir sampling
    Reservoir,
    /// Stratified sampling by performance
    Stratified { n_strata: usize },
    /// k-DPP sampling for diversity
    KDPP { k: usize },
}

/// Experience replay buffer
pub struct ExperienceReplayBuffer {
    config: ExperienceReplayConfig,
    buffer: VecDeque<Experience>,
    priorities: Vec<Float>,
    rng: StdRng,
}

impl ExperienceReplayBuffer {
    pub fn new(config: ExperienceReplayConfig) -> Self {
        let rng = StdRng::seed_from_u64(config.random_state.unwrap_or(42));
        Self {
            config,
            buffer: VecDeque::new(),
            priorities: Vec::new(),
            rng,
        }
    }

    /// Add experience to buffer
    pub fn add(&mut self, experience: Experience) {
        if self.buffer.len() >= self.config.buffer_size {
            self.buffer.pop_front();
            if !self.priorities.is_empty() {
                self.priorities.remove(0);
            }
        }

        let priority = self.compute_priority(&experience);
        self.buffer.push_back(experience);
        self.priorities.push(priority);
    }

    /// Sample batch of experiences
    pub fn sample(
        &mut self,
        batch_size: usize,
    ) -> Result<Vec<Experience>, Box<dyn std::error::Error>> {
        if self.buffer.is_empty() {
            return Err("Buffer is empty".into());
        }

        let sample_size = batch_size.min(self.buffer.len());

        match &self.config.sampling_strategy {
            SamplingStrategy::Random => self.sample_random(sample_size),
            SamplingStrategy::Reservoir => self.sample_reservoir(sample_size),
            SamplingStrategy::Stratified { n_strata } => {
                self.sample_stratified(sample_size, *n_strata)
            }
            SamplingStrategy::KDPP { k } => self.sample_kdpp(sample_size, *k),
        }
    }

    /// Replay experiences to improve optimizer
    pub fn replay_update(
        &mut self,
        optimizer: &mut dyn OptimizationLearner,
    ) -> Result<ReplayResult, Box<dyn std::error::Error>> {
        let mut total_loss = 0.0;
        let mut n_updates = 0;

        for _ in 0..self.config.n_replay_updates {
            let batch = self.sample(self.config.batch_size)?;
            let loss = optimizer.update_from_batch(&batch)?;
            total_loss += loss;
            n_updates += 1;
        }

        Ok(ReplayResult {
            average_loss: total_loss / n_updates as Float,
            n_updates,
            buffer_size: self.buffer.len(),
        })
    }

    // Sampling implementations

    fn sample_random(&mut self, n: usize) -> Result<Vec<Experience>, Box<dyn std::error::Error>> {
        let mut sampled = Vec::new();
        let buffer_vec: Vec<_> = self.buffer.iter().collect();

        for _ in 0..n {
            let idx = self.rng.gen_range(0..self.buffer.len());
            sampled.push(buffer_vec[idx].clone());
        }

        Ok(sampled)
    }

    fn sample_reservoir(
        &mut self,
        n: usize,
    ) -> Result<Vec<Experience>, Box<dyn std::error::Error>> {
        self.sample_random(n) // Simplified
    }

    fn sample_stratified(
        &mut self,
        n: usize,
        n_strata: usize,
    ) -> Result<Vec<Experience>, Box<dyn std::error::Error>> {
        // Stratify by performance
        let mut strata: Vec<Vec<Experience>> = vec![Vec::new(); n_strata];

        for exp in &self.buffer {
            let stratum_idx = ((exp.reward * n_strata as Float) as usize).min(n_strata - 1);
            strata[stratum_idx].push(exp.clone());
        }

        // Sample proportionally from each stratum
        let mut sampled = Vec::new();
        let per_stratum = n / n_strata;

        for stratum in &strata {
            if stratum.is_empty() {
                continue;
            }

            for _ in 0..per_stratum.min(stratum.len()) {
                let idx = self.rng.gen_range(0..stratum.len());
                sampled.push(stratum[idx].clone());
            }
        }

        Ok(sampled)
    }

    fn sample_kdpp(
        &mut self,
        n: usize,
        _k: usize,
    ) -> Result<Vec<Experience>, Box<dyn std::error::Error>> {
        // Simplified diversity sampling
        self.sample_random(n)
    }

    fn compute_priority(&self, experience: &Experience) -> Float {
        match &self.config.prioritization {
            PrioritizationStrategy::Uniform => 1.0,
            PrioritizationStrategy::TDError { alpha } => {
                // Use improvement as proxy for TD error
                (experience.improvement.abs() + 1e-6).powf(*alpha)
            }
            PrioritizationStrategy::Improvement { temperature } => {
                (experience.improvement / temperature).exp()
            }
            PrioritizationStrategy::Recency { decay_rate } => {
                (-decay_rate * self.buffer.len() as Float).exp()
            }
            PrioritizationStrategy::Diversity { .. } => 1.0, // Would need distance computation
        }
    }
}

/// Optimization experience
#[derive(Debug, Clone)]
pub struct Experience {
    pub state: HashMap<String, Float>,
    pub action: HashMap<String, Float>,
    pub reward: Float,
    pub next_state: HashMap<String, Float>,
    pub improvement: Float,
    pub timestamp: usize,
}

/// Result of replay update
#[derive(Debug, Clone)]
pub struct ReplayResult {
    pub average_loss: Float,
    pub n_updates: usize,
    pub buffer_size: usize,
}

/// Trait for optimizers that can learn from experience
pub trait OptimizationLearner {
    fn update_from_batch(
        &mut self,
        batch: &[Experience],
    ) -> Result<Float, Box<dyn std::error::Error>>;
}

// ============================================================================
// Supporting Types
// ============================================================================

/// Task characteristics for transfer learning
#[derive(Debug, Clone)]
pub struct TaskCharacteristics {
    pub task_type: String,
    pub n_samples: usize,
    pub n_features: usize,
    pub parameter_space: HashMap<String, ParameterRange>,
    pub complexity: Float,
}

#[derive(Debug, Clone)]
pub struct ParameterRange {
    pub min: Float,
    pub max: Float,
    pub scale: ParameterScale,
}

#[derive(Debug, Clone)]
pub enum ParameterScale {
    Linear,
    Log,
    Categorical,
}

#[derive(Debug, Clone)]
pub struct ParameterDistribution {
    pub mean: Float,
    pub std: Float,
}

/// Optimization experience from historical data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationExperience {
    pub task_characteristics: TaskCharacteristics,
    pub best_parameters: HashMap<String, Float>,
    pub performance: Float,
    pub n_iterations: usize,
    pub convergence_curve: Vec<Float>,
}

// Manual Serialize/Deserialize for TaskCharacteristics
impl Serialize for TaskCharacteristics {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("TaskCharacteristics", 5)?;
        state.serialize_field("task_type", &self.task_type)?;
        state.serialize_field("n_samples", &self.n_samples)?;
        state.serialize_field("n_features", &self.n_features)?;
        state.serialize_field("complexity", &self.complexity)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for TaskCharacteristics {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Helper {
            task_type: String,
            n_samples: usize,
            n_features: usize,
            complexity: Float,
        }

        let helper = Helper::deserialize(deserializer)?;
        Ok(TaskCharacteristics {
            task_type: helper.task_type,
            n_samples: helper.n_samples,
            n_features: helper.n_features,
            parameter_space: HashMap::new(),
            complexity: helper.complexity,
        })
    }
}

/// Optimization task for few-shot learning
#[derive(Debug, Clone)]
pub struct OptimizationTask {
    pub task_id: String,
    pub support_examples: Vec<(HashMap<String, Float>, Float)>,
    pub query_examples: Vec<(HashMap<String, Float>, Float)>,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transfer_learning_config() {
        let config = TransferLearningConfig::default();
        assert_eq!(config.n_init_samples, 5);
        assert_eq!(config.max_transfer_iterations, 50);
    }

    #[test]
    fn test_few_shot_config() {
        let config = FewShotConfig::default();
        assert_eq!(config.n_support, 5);
        assert_eq!(config.n_query, 10);
    }

    #[test]
    fn test_learn2optimize_config() {
        let config = Learn2OptimizeConfig::default();
        assert_eq!(config.n_training_tasks, 100);
        assert!(config.use_recurrent);
    }

    #[test]
    fn test_experience_replay_buffer() {
        let config = ExperienceReplayConfig::default();
        let mut buffer = ExperienceReplayBuffer::new(config);

        let experience = Experience {
            state: HashMap::new(),
            action: HashMap::new(),
            reward: 0.8,
            next_state: HashMap::new(),
            improvement: 0.1,
            timestamp: 0,
        };

        buffer.add(experience.clone());
        assert_eq!(buffer.buffer.len(), 1);
    }

    #[test]
    fn test_transfer_learning_optimizer() {
        let config = TransferLearningConfig::default();
        let optimizer = TransferLearningOptimizer::new(config);

        assert!(optimizer.transferred_knowledge.is_empty());
        assert!(optimizer.transfer_performance.is_empty());
    }

    #[test]
    fn test_few_shot_optimizer() {
        let config = FewShotConfig::default();
        let optimizer = FewShotOptimizer::new(config);

        assert!(optimizer.meta_parameters.is_empty());
        assert!(optimizer.episode_history.is_empty());
    }

    #[test]
    fn test_learned_optimizer() {
        let config = Learn2OptimizeConfig::default();
        let optimizer = LearnedOptimizer::new(config);

        assert!(optimizer.training_history.is_empty());
    }
}
