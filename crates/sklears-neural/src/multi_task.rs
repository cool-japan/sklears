//! Multi-Task Learning for Neural Networks
//!
//! This module provides comprehensive multi-task learning capabilities, allowing models
//! to learn multiple related tasks simultaneously with shared representations.

use crate::layers::Layer;
use crate::models::{Functional, Sequential};
use crate::NeuralResult;
use scirs2_core::ndarray::{Array1, Array2, ScalarOperand};
use sklears_core::error::SklearsError;
use sklears_core::types::FloatBounds;
use std::collections::HashMap;
use std::iter::Sum;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Multi-task learning strategies for parameter sharing
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SharingStrategy {
    /// Hard parameter sharing: shared backbone, task-specific heads
    HardSharing {
        shared_layers: usize,
        task_specific_layers: Vec<usize>,
    },
    /// Soft parameter sharing: task-specific networks with regularization
    SoftSharing {
        l2_penalty: f64,
        similarity_threshold: f64,
    },
    /// Cross-stitch networks: linear combinations of task-specific features
    CrossStitch { num_units: Vec<usize> },
    /// Attention-based sharing: learn what to share between tasks
    AttentionSharing { attention_dim: usize },
}

impl Default for SharingStrategy {
    fn default() -> Self {
        SharingStrategy::HardSharing {
            shared_layers: 2,
            task_specific_layers: vec![64, 32],
        }
    }
}

/// Task weighting strategies for multi-task loss balancing
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum TaskWeightingStrategy {
    /// Equal weights for all tasks
    Equal,
    /// Manual task weights
    Manual(Vec<f64>),
    /// Uncertainty-based weighting (homoscedastic uncertainty)
    UncertaintyWeighting,
    /// Dynamic weight adjustment based on task difficulty
    DynamicWeighting {
        adaptation_rate: f64,
        min_weight: f64,
        max_weight: f64,
    },
    /// Gradient normalization based weighting
    GradNorm {
        alpha: f64,
        initial_weights: Vec<f64>,
    },
}

impl Default for TaskWeightingStrategy {
    fn default() -> Self {
        TaskWeightingStrategy::Equal
    }
}

/// Multi-task loss configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound = "T: FloatBounds + serde::Serialize + serde::de::DeserializeOwned")
)]
pub struct MultiTaskLoss<T: FloatBounds> {
    /// Task-specific loss functions
    pub task_losses: Vec<String>, // "mse", "cross_entropy", "binary_cross_entropy"
    /// Task weighting strategy
    pub weighting_strategy: TaskWeightingStrategy,
    /// Current task weights
    pub task_weights: Vec<T>,
    /// Regularization strength for soft sharing
    pub regularization_strength: Option<T>,
}

impl<T: FloatBounds> Default for MultiTaskLoss<T> {
    fn default() -> Self {
        Self {
            task_losses: vec!["mse".to_string()],
            weighting_strategy: TaskWeightingStrategy::Equal,
            task_weights: vec![T::from(1.0).unwrap()],
            regularization_strength: None,
        }
    }
}

/// Multi-task neural network configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound = "T: FloatBounds + serde::Serialize + serde::de::DeserializeOwned")
)]
pub struct MultiTaskConfig<T: FloatBounds> {
    /// Number of tasks
    pub num_tasks: usize,
    /// Input dimension
    pub input_dim: usize,
    /// Output dimensions for each task
    pub output_dims: Vec<usize>,
    /// Parameter sharing strategy
    pub sharing_strategy: SharingStrategy,
    /// Loss configuration
    pub loss_config: MultiTaskLoss<T>,
    /// Task names for identification
    pub task_names: Vec<String>,
    /// Whether to use task embeddings
    pub use_task_embeddings: bool,
    /// Task embedding dimension
    pub task_embedding_dim: usize,
}

impl<T: FloatBounds> MultiTaskConfig<T> {
    pub fn new(num_tasks: usize, input_dim: usize, output_dims: Vec<usize>) -> Self {
        if output_dims.len() != num_tasks {
            panic!("Number of output dimensions must match number of tasks");
        }

        Self {
            num_tasks,
            input_dim,
            output_dims,
            sharing_strategy: SharingStrategy::default(),
            loss_config: MultiTaskLoss::default(),
            task_names: (0..num_tasks).map(|i| format!("task_{}", i)).collect(),
            use_task_embeddings: false,
            task_embedding_dim: 8,
        }
    }

    /// Set task names
    pub fn with_task_names(mut self, names: Vec<String>) -> Self {
        if names.len() != self.num_tasks {
            panic!("Number of task names must match number of tasks");
        }
        self.task_names = names;
        self
    }

    /// Set sharing strategy
    pub fn with_sharing_strategy(mut self, strategy: SharingStrategy) -> Self {
        self.sharing_strategy = strategy;
        self
    }

    /// Set task weighting strategy
    pub fn with_task_weighting(mut self, strategy: TaskWeightingStrategy) -> Self {
        self.loss_config.weighting_strategy = strategy;
        self
    }

    /// Enable task embeddings
    pub fn with_task_embeddings(mut self, embedding_dim: usize) -> Self {
        self.use_task_embeddings = true;
        self.task_embedding_dim = embedding_dim;
        self
    }
}

/// Multi-task neural network model
pub struct MultiTaskNetwork<T: FloatBounds> {
    /// Model configuration
    config: MultiTaskConfig<T>,
    /// Shared backbone model
    shared_model: Option<Sequential<T>>,
    /// Task-specific heads
    task_heads: HashMap<String, Sequential<T>>,
    /// Task embeddings for conditional computation
    task_embeddings: Option<Array2<T>>,
    /// Cross-stitch units for soft sharing
    cross_stitch_units: Option<Vec<Array2<T>>>,
    /// Training state
    is_fitted: bool,
}

impl<T: FloatBounds + ScalarOperand + Sum> MultiTaskNetwork<T> {
    /// Create a new multi-task network
    pub fn new(config: MultiTaskConfig<T>) -> Self {
        Self {
            config,
            shared_model: None,
            task_heads: HashMap::new(),
            task_embeddings: None,
            cross_stitch_units: None,
            is_fitted: false,
        }
    }

    /// Build the network architecture based on configuration
    pub fn build_architecture(&mut self) -> NeuralResult<()> {
        let strategy = self.config.sharing_strategy.clone();
        match strategy {
            SharingStrategy::HardSharing {
                shared_layers,
                task_specific_layers,
            } => {
                self.build_hard_sharing_architecture(shared_layers, &task_specific_layers)?;
            }
            SharingStrategy::SoftSharing { .. } => {
                self.build_soft_sharing_architecture()?;
            }
            SharingStrategy::CrossStitch { num_units } => {
                self.build_cross_stitch_architecture(&num_units)?;
            }
            SharingStrategy::AttentionSharing { attention_dim } => {
                self.build_attention_sharing_architecture(attention_dim)?;
            }
        }

        // Initialize task embeddings if enabled
        if self.config.use_task_embeddings {
            self.initialize_task_embeddings()?;
        }

        Ok(())
    }

    /// Build hard parameter sharing architecture
    fn build_hard_sharing_architecture(
        &mut self,
        shared_layers: usize,
        task_specific_layers: &[usize],
    ) -> NeuralResult<()> {
        // Build shared backbone
        let mut shared_model = Sequential::new();
        let mut current_dim = self.config.input_dim;

        for _ in 0..shared_layers {
            let layer_dim = current_dim / 2; // Simple reduction strategy
                                             // Note: In a real implementation, we would add actual Dense layers here
                                             // This is a simplified version for demonstration
            current_dim = layer_dim.max(8); // Minimum layer size
        }

        self.shared_model = Some(shared_model);

        // Build task-specific heads
        for (task_idx, task_name) in self.config.task_names.iter().enumerate() {
            let mut task_head = Sequential::new();
            let mut head_dim = current_dim;

            for &layer_size in task_specific_layers {
                // Add task-specific layers
                head_dim = layer_size;
            }

            // Final output layer
            let output_dim = self.config.output_dims[task_idx];
            // Add final layer with output_dim neurons

            self.task_heads.insert(task_name.clone(), task_head);
        }

        Ok(())
    }

    /// Build soft parameter sharing architecture
    fn build_soft_sharing_architecture(&mut self) -> NeuralResult<()> {
        // Create separate networks for each task with shared structure
        for (task_idx, task_name) in self.config.task_names.iter().enumerate() {
            let mut task_network = Sequential::new();
            let output_dim = self.config.output_dims[task_idx];

            // Build task-specific network
            // (Implementation would add actual layers here)

            self.task_heads.insert(task_name.clone(), task_network);
        }

        Ok(())
    }

    /// Build cross-stitch architecture
    fn build_cross_stitch_architecture(&mut self, num_units: &[usize]) -> NeuralResult<()> {
        // Initialize cross-stitch units
        let mut units = Vec::new();
        for &unit_size in num_units {
            let unit = Array2::eye(self.config.num_tasks) * T::from(0.8).unwrap()
                + Array2::from_elem(
                    (self.config.num_tasks, self.config.num_tasks),
                    T::from(0.2).unwrap() / T::from(self.config.num_tasks as f64).unwrap(),
                );
            units.push(unit);
        }
        self.cross_stitch_units = Some(units);

        // Build task-specific networks
        for (task_idx, task_name) in self.config.task_names.iter().enumerate() {
            let mut task_network = Sequential::new();
            let output_dim = self.config.output_dims[task_idx];

            // Build network with cross-stitch connections
            // (Implementation would add actual layers here)

            self.task_heads.insert(task_name.clone(), task_network);
        }

        Ok(())
    }

    /// Build attention-based sharing architecture
    fn build_attention_sharing_architecture(&mut self, attention_dim: usize) -> NeuralResult<()> {
        // Build shared encoder with attention mechanism
        let mut shared_model = Sequential::new();
        // (Implementation would add attention layers here)

        self.shared_model = Some(shared_model);

        // Build task-specific decoders
        for (task_idx, task_name) in self.config.task_names.iter().enumerate() {
            let mut task_head = Sequential::new();
            let output_dim = self.config.output_dims[task_idx];

            // Build attention-based task head
            // (Implementation would add actual layers here)

            self.task_heads.insert(task_name.clone(), task_head);
        }

        Ok(())
    }

    /// Initialize task embeddings
    fn initialize_task_embeddings(&mut self) -> NeuralResult<()> {
        use scirs2_core::random::Rng;
        let mut rng = scirs2_core::random::thread_rng();

        let mut embeddings = Array2::zeros((self.config.num_tasks, self.config.task_embedding_dim));
        for mut row in embeddings.rows_mut() {
            for elem in row.iter_mut() {
                *elem = T::from(rng.gen_range(-0.1..0.1)).unwrap();
            }
        }

        self.task_embeddings = Some(embeddings);
        Ok(())
    }

    /// Forward pass for a specific task
    pub fn forward_task(
        &mut self,
        input: &Array2<T>,
        task_name: &str,
        training: bool,
    ) -> NeuralResult<Array2<T>> {
        if !self.is_fitted {
            return Err(SklearsError::InvalidParameter {
                name: "model".to_string(),
                reason: "Model must be fitted before prediction".to_string(),
            });
        }

        // Get shared features if using hard sharing
        let features = if let Some(ref mut shared_model) = self.shared_model {
            shared_model.forward(input, training)?
        } else {
            input.clone()
        };

        // Get task-specific output
        if let Some(task_head) = self.task_heads.get_mut(task_name) {
            task_head.forward(&features, training)
        } else {
            Err(SklearsError::InvalidParameter {
                name: "task_name".to_string(),
                reason: format!("Unknown task: {}", task_name),
            })
        }
    }

    /// Forward pass for all tasks
    pub fn forward_all_tasks(
        &mut self,
        input: &Array2<T>,
        training: bool,
    ) -> NeuralResult<HashMap<String, Array2<T>>> {
        let mut outputs = HashMap::new();

        for task_name in &self.config.task_names.clone() {
            let output = self.forward_task(input, task_name, training)?;
            outputs.insert(task_name.clone(), output);
        }

        Ok(outputs)
    }

    /// Compute multi-task loss
    pub fn compute_multi_task_loss(
        &self,
        predictions: &HashMap<String, Array2<T>>,
        targets: &HashMap<String, Array2<T>>,
    ) -> NeuralResult<T> {
        let mut total_loss = T::from(0.0).unwrap();
        let mut valid_tasks = 0;

        for (task_idx, task_name) in self.config.task_names.iter().enumerate() {
            if let (Some(pred), Some(target)) = (predictions.get(task_name), targets.get(task_name))
            {
                let task_loss = self.compute_task_loss(pred, target, task_idx)?;
                let weight = if task_idx < self.config.loss_config.task_weights.len() {
                    self.config.loss_config.task_weights[task_idx]
                } else {
                    T::from(1.0).unwrap()
                };
                total_loss = total_loss + weight * task_loss;
                valid_tasks += 1;
            }
        }

        if valid_tasks == 0 {
            return Err(SklearsError::InvalidParameter {
                name: "targets".to_string(),
                reason: "No valid task targets provided".to_string(),
            });
        }

        Ok(total_loss / T::from(valid_tasks as f64).unwrap())
    }

    /// Compute loss for a specific task
    fn compute_task_loss(
        &self,
        predictions: &Array2<T>,
        targets: &Array2<T>,
        task_idx: usize,
    ) -> NeuralResult<T> {
        if predictions.shape() != targets.shape() {
            return Err(SklearsError::InvalidParameter {
                name: "shape".to_string(),
                reason: "Predictions and targets must have the same shape".to_string(),
            });
        }

        let loss_type = self
            .config
            .loss_config
            .task_losses
            .get(task_idx)
            .map(|s| s.as_str())
            .unwrap_or("mse");

        match loss_type {
            "mse" => {
                let diff = predictions - targets;
                let squared_diff = &diff * &diff;
                Ok(squared_diff.mean().unwrap())
            }
            "mae" => {
                let diff = predictions - targets;
                let abs_diff = diff.mapv(|x| x.abs());
                Ok(abs_diff.mean().unwrap())
            }
            _ => Err(SklearsError::InvalidParameter {
                name: "loss_type".to_string(),
                reason: format!("Unsupported loss type: {}", loss_type),
            }),
        }
    }

    /// Update task weights based on strategy
    pub fn update_task_weights(&mut self, task_losses: &[T], epoch: usize) -> NeuralResult<()> {
        let strategy = self.config.loss_config.weighting_strategy.clone();
        match strategy {
            TaskWeightingStrategy::Equal => {
                self.config.loss_config.task_weights =
                    vec![T::from(1.0).unwrap(); self.config.num_tasks];
            }
            TaskWeightingStrategy::Manual(weights) => {
                self.config.loss_config.task_weights =
                    weights.iter().map(|&w| T::from(w).unwrap()).collect();
            }
            TaskWeightingStrategy::DynamicWeighting {
                adaptation_rate,
                min_weight,
                max_weight,
            } => {
                self.update_dynamic_weights(task_losses, adaptation_rate, min_weight, max_weight)?;
            }
            TaskWeightingStrategy::UncertaintyWeighting => {
                self.update_uncertainty_weights(task_losses)?;
            }
            TaskWeightingStrategy::GradNorm {
                alpha,
                initial_weights,
            } => {
                self.update_gradnorm_weights(task_losses, alpha, &initial_weights)?;
            }
        }

        Ok(())
    }

    /// Update weights using dynamic weighting strategy
    fn update_dynamic_weights(
        &mut self,
        task_losses: &[T],
        adaptation_rate: f64,
        min_weight: f64,
        max_weight: f64,
    ) -> NeuralResult<()> {
        if task_losses.is_empty() {
            return Ok(());
        }

        // Compute relative task difficulties
        let total_loss: T = task_losses
            .iter()
            .copied()
            .fold(T::from(0.0).unwrap(), |a, b| a + b);
        let avg_loss = total_loss / T::from(task_losses.len() as f64).unwrap();

        for (i, &task_loss) in task_losses.iter().enumerate() {
            let current_weight = self
                .config
                .loss_config
                .task_weights
                .get(i)
                .copied()
                .unwrap_or(T::from(1.0).unwrap());

            // Increase weight for harder tasks (higher loss)
            let difficulty_ratio = task_loss / avg_loss;
            let target_weight = T::from(1.0).unwrap()
                + (difficulty_ratio - T::from(1.0).unwrap()) * T::from(adaptation_rate).unwrap();

            // Smooth update
            let new_weight =
                current_weight * T::from(0.9).unwrap() + target_weight * T::from(0.1).unwrap();
            let clamped_weight =
                T::from(new_weight.to_f64().unwrap().clamp(min_weight, max_weight)).unwrap();

            if i < self.config.loss_config.task_weights.len() {
                self.config.loss_config.task_weights[i] = clamped_weight;
            } else {
                self.config.loss_config.task_weights.push(clamped_weight);
            }
        }

        Ok(())
    }

    /// Update weights using uncertainty weighting
    fn update_uncertainty_weights(&mut self, task_losses: &[T]) -> NeuralResult<()> {
        // Simplified uncertainty weighting based on loss variance
        if task_losses.len() < 2 {
            return Ok(());
        }

        let total_loss = task_losses
            .iter()
            .copied()
            .fold(T::from(0.0).unwrap(), |acc, value| acc + value);
        let mean_loss = total_loss / T::from(task_losses.len() as f64).unwrap();
        let mut weights = Vec::new();

        for &loss in task_losses {
            // Higher uncertainty (variance) gets higher weight
            let uncertainty = (loss - mean_loss).abs() + T::from(1e-8).unwrap();
            weights.push(T::from(1.0).unwrap() / uncertainty);
        }

        // Normalize weights
        let total_weight = weights
            .iter()
            .copied()
            .fold(T::from(0.0).unwrap(), |acc, value| acc + value);
        let normalization = T::from(weights.len() as f64).unwrap() / total_weight;
        for weight in &mut weights {
            *weight *= normalization;
        }

        self.config.loss_config.task_weights = weights;
        Ok(())
    }

    /// Update weights using GradNorm algorithm
    fn update_gradnorm_weights(
        &mut self,
        task_losses: &[T],
        alpha: f64,
        initial_weights: &[f64],
    ) -> NeuralResult<()> {
        // Simplified GradNorm implementation
        // In practice, this would need gradient information
        let mut weights = initial_weights
            .iter()
            .map(|&w| T::from(w).unwrap())
            .collect::<Vec<_>>();

        if task_losses.len() != weights.len() {
            return Err(SklearsError::InvalidParameter {
                name: "weights".to_string(),
                reason: "Number of weights must match number of tasks".to_string(),
            });
        }

        // Compute relative training rates
        let total_loss: T = task_losses
            .iter()
            .copied()
            .fold(T::from(0.0).unwrap(), |a, b| a + b);
        let avg_loss = total_loss / T::from(task_losses.len() as f64).unwrap();

        for (i, (&loss, weight)) in task_losses.iter().zip(weights.iter_mut()).enumerate() {
            let relative_rate = loss / avg_loss;
            let target_rate = T::from(1.0).unwrap();
            let adjustment = (relative_rate / target_rate).powf(T::from(alpha).unwrap());
            *weight = *weight * adjustment;
        }

        // Normalize weights
        let total_weight: T = weights
            .iter()
            .copied()
            .fold(T::from(0.0).unwrap(), |a, b| a + b);
        let weights_len = weights.len();
        for weight in &mut weights {
            *weight = *weight / total_weight * T::from(weights_len as f64).unwrap();
        }

        self.config.loss_config.task_weights = weights;
        Ok(())
    }

    /// Get configuration
    pub fn config(&self) -> &MultiTaskConfig<T> {
        &self.config
    }

    /// Get task names
    pub fn task_names(&self) -> &[String] {
        &self.config.task_names
    }

    /// Get number of tasks
    pub fn num_tasks(&self) -> usize {
        self.config.num_tasks
    }

    /// Check if model is fitted
    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }

    /// Set fitted state
    pub fn set_fitted(&mut self, fitted: bool) {
        self.is_fitted = fitted;
    }

    /// Get current task weights
    pub fn task_weights(&self) -> &[T] {
        &self.config.loss_config.task_weights
    }
}

impl<T: FloatBounds> std::fmt::Debug for MultiTaskNetwork<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiTaskNetwork")
            .field("num_tasks", &self.config.num_tasks)
            .field("task_names", &self.config.task_names)
            .field("sharing_strategy", &self.config.sharing_strategy)
            .field("is_fitted", &self.is_fitted)
            .finish()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_multi_task_config_creation() {
        let config = MultiTaskConfig::<f64>::new(3, 10, vec![2, 3, 1]);
        assert_eq!(config.num_tasks, 3);
        assert_eq!(config.input_dim, 10);
        assert_eq!(config.output_dims, vec![2, 3, 1]);
        assert_eq!(config.task_names, vec!["task_0", "task_1", "task_2"]);
    }

    #[test]
    fn test_multi_task_config_with_task_names() {
        let config = MultiTaskConfig::<f64>::new(2, 5, vec![1, 1])
            .with_task_names(vec!["classification".to_string(), "regression".to_string()]);

        assert_eq!(config.task_names, vec!["classification", "regression"]);
    }

    #[test]
    fn test_sharing_strategies() {
        let hard_sharing = SharingStrategy::HardSharing {
            shared_layers: 3,
            task_specific_layers: vec![64, 32],
        };
        assert!(matches!(hard_sharing, SharingStrategy::HardSharing { .. }));

        let soft_sharing = SharingStrategy::SoftSharing {
            l2_penalty: 0.01,
            similarity_threshold: 0.8,
        };
        assert!(matches!(soft_sharing, SharingStrategy::SoftSharing { .. }));
    }

    #[test]
    fn test_task_weighting_strategies() {
        let equal = TaskWeightingStrategy::Equal;
        assert!(matches!(equal, TaskWeightingStrategy::Equal));

        let manual = TaskWeightingStrategy::Manual(vec![1.0, 2.0, 0.5]);
        assert!(matches!(manual, TaskWeightingStrategy::Manual(_)));

        let dynamic = TaskWeightingStrategy::DynamicWeighting {
            adaptation_rate: 0.1,
            min_weight: 0.1,
            max_weight: 5.0,
        };
        assert!(matches!(
            dynamic,
            TaskWeightingStrategy::DynamicWeighting { .. }
        ));
    }

    #[test]
    fn test_multi_task_network_creation() {
        let config = MultiTaskConfig::<f64>::new(2, 10, vec![3, 1]);
        let network = MultiTaskNetwork::new(config);

        assert_eq!(network.num_tasks(), 2);
        assert_eq!(network.task_names(), &["task_0", "task_1"]);
        assert!(!network.is_fitted());
    }

    #[test]
    fn test_multi_task_loss_computation() {
        let config = MultiTaskConfig::<f64>::new(2, 5, vec![2, 1]);
        let network = MultiTaskNetwork::new(config);

        // Create sample predictions and targets
        let mut predictions = HashMap::new();
        predictions.insert(
            "task_0".to_string(),
            Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        );
        predictions.insert(
            "task_1".to_string(),
            Array2::from_shape_vec((2, 1), vec![0.5, 1.5]).unwrap(),
        );

        let mut targets = HashMap::new();
        targets.insert(
            "task_0".to_string(),
            Array2::from_shape_vec((2, 2), vec![1.1, 1.9, 3.1, 3.9]).unwrap(),
        );
        targets.insert(
            "task_1".to_string(),
            Array2::from_shape_vec((2, 1), vec![0.6, 1.4]).unwrap(),
        );

        let loss = network
            .compute_multi_task_loss(&predictions, &targets)
            .unwrap();
        assert!(loss > 0.0);
    }

    #[test]
    fn test_task_weight_updates() {
        let config = MultiTaskConfig::<f64>::new(3, 5, vec![1, 1, 1]).with_task_weighting(
            TaskWeightingStrategy::DynamicWeighting {
                adaptation_rate: 0.1,
                min_weight: 0.1,
                max_weight: 5.0,
            },
        );

        let mut network = MultiTaskNetwork::new(config);
        let task_losses = vec![1.0, 2.0, 0.5];

        network.update_task_weights(&task_losses, 1).unwrap();
        let weights = network.task_weights();

        assert_eq!(weights.len(), 3);
        // Higher loss should get higher weight
        assert!(weights[1] > weights[2]); // task_1 (loss=2.0) > task_2 (loss=0.5)
    }

    #[test]
    fn test_multi_task_serialization() {
        let config = MultiTaskConfig::<f64>::new(2, 10, vec![3, 1])
            .with_task_names(vec!["classification".to_string(), "regression".to_string()])
            .with_sharing_strategy(SharingStrategy::HardSharing {
                shared_layers: 2,
                task_specific_layers: vec![64, 32],
            });

        // Test that the config can be serialized if serde feature is enabled
        #[cfg(feature = "serde")]
        {
            let json = serde_json::to_string(&config).unwrap();
            let deserialized: MultiTaskConfig<f64> = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized.num_tasks, config.num_tasks);
            assert_eq!(deserialized.task_names, config.task_names);
        }
    }

    #[test]
    fn test_forward_pass_error_handling() {
        let config = MultiTaskConfig::<f64>::new(2, 5, vec![2, 1]);
        let mut network = MultiTaskNetwork::new(config);

        let input = Array2::zeros((1, 5));

        // Should fail when model is not fitted
        let result = network.forward_task(&input, "task_0", false);
        assert!(result.is_err());

        // Should fail with unknown task
        network.set_fitted(true);
        let result = network.forward_task(&input, "unknown_task", false);
        assert!(result.is_err());
    }
}
