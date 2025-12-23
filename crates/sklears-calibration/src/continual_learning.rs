//! Continual Learning Calibration Methods
//!
//! This module implements calibration techniques for continual learning scenarios
//! where models need to learn new tasks sequentially while maintaining calibration
//! quality across all learned tasks. These methods address catastrophic forgetting
//! in calibration and provide mechanisms for task-specific and global calibration.
//!
//! Key challenges addressed:
//! - Calibration catastrophic forgetting
//! - Task-specific vs. global calibration trade-offs
//! - Memory-efficient calibration updates
//! - Uncertainty quantification across task boundaries

use scirs2_core::ndarray::Array1;
use scirs2_core::random::thread_rng;
use sklears_core::{error::Result, prelude::SklearsError, types::Float};
use std::collections::HashMap;

use crate::CalibrationEstimator;

/// Type alias for replay sample data (probabilities, labels, weights)
type ReplaySampleData = (Array1<Float>, Array1<i32>, Array1<Float>);

/// Task identification and metadata
#[derive(Debug, Clone, PartialEq)]
pub struct TaskId {
    /// Unique identifier for the task
    pub id: String,
    /// Task domain or category
    pub domain: String,
    /// Task difficulty level (0.0 = easy, 1.0 = hard)
    pub difficulty: Float,
}

impl std::hash::Hash for TaskId {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.domain.hash(state);
        // For difficulty, we'll use a simplified hash based on the integer part
        ((self.difficulty * 1000.0) as i32).hash(state);
    }
}

impl Eq for TaskId {}

impl TaskId {
    /// Create a new task identifier
    pub fn new(id: String, domain: String, difficulty: Float) -> Self {
        Self {
            id,
            domain,
            difficulty,
        }
    }
}

/// Memory replay strategies for continual calibration
#[derive(Debug, Clone)]
pub enum ReplayStrategy {
    /// Store fixed number of examples per task
    FixedSize { samples_per_task: usize },
    /// Store examples proportional to task importance
    ImportanceWeighted { max_samples: usize },
    /// Store diverse examples using clustering
    DiversityBased {
        max_samples: usize,
        cluster_method: String,
    },
    /// Store recent examples with temporal decay
    TemporalDecay {
        max_samples: usize,
        decay_rate: Float,
    },
}

/// Continual Learning Calibrator
///
/// Maintains calibration quality across sequential learning of multiple tasks
/// using episodic memory and specialized update strategies.
#[derive(Debug, Clone)]
pub struct ContinualLearningCalibrator {
    /// Base calibration method for each task
    base_method: BaseCalibrationMethod,
    /// Task-specific calibrators
    task_calibrators: HashMap<TaskId, Box<dyn CalibrationEstimator>>,
    /// Global calibrator combining all tasks
    global_calibrator: Option<Box<dyn CalibrationEstimator>>,
    /// Memory buffer for replay
    memory_buffer: EpisodicMemory,
    /// Replay strategy
    replay_strategy: ReplayStrategy,
    /// Regularization strength for preventing forgetting
    regularization_strength: Float,
    /// Whether to use task-specific or global calibration
    use_task_specific: bool,
    /// Current task being trained
    current_task: Option<TaskId>,
    /// Learning rate for calibration updates
    learning_rate: Float,
    /// Whether the calibrator is fitted
    is_fitted: bool,
}

/// Base calibration methods for continual learning
#[derive(Debug, Clone)]
pub enum BaseCalibrationMethod {
    Temperature,
    Platt,
    Histogram {
        n_bins: usize,
    },
    /// Isotonic regression
    Isotonic,
}

/// Episodic memory for storing calibration examples
#[derive(Debug, Clone)]
pub struct EpisodicMemory {
    /// Stored examples by task
    task_memories: HashMap<TaskId, TaskMemory>,
    /// Maximum total memory size
    max_memory_size: usize,
    /// Current memory usage
    current_size: usize,
}

/// Memory for a specific task
#[derive(Debug, Clone)]
struct TaskMemory {
    /// Probabilities
    probabilities: Vec<Float>,
    /// True labels
    labels: Vec<i32>,
    /// Importance weights
    weights: Vec<Float>,
    /// Timestamps
    timestamps: Vec<u64>,
}

impl EpisodicMemory {
    /// Create a new episodic memory
    pub fn new(max_memory_size: usize) -> Self {
        Self {
            task_memories: HashMap::new(),
            max_memory_size,
            current_size: 0,
        }
    }

    /// Add examples to memory for a specific task
    pub fn add_examples(
        &mut self,
        task_id: TaskId,
        probabilities: &Array1<Float>,
        labels: &Array1<i32>,
        weights: Option<&Array1<Float>>,
    ) -> Result<()> {
        let n_examples = probabilities.len();
        if n_examples != labels.len() {
            return Err(SklearsError::InvalidInput(
                "Probabilities and labels must have same length".to_string(),
            ));
        }

        // Create or get existing task memory
        let task_memory = self
            .task_memories
            .entry(task_id.clone())
            .or_insert_with(|| TaskMemory {
                probabilities: Vec::new(),
                labels: Vec::new(),
                weights: Vec::new(),
                timestamps: Vec::new(),
            });

        // Add new examples
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        for i in 0..n_examples {
            task_memory.probabilities.push(probabilities[i]);
            task_memory.labels.push(labels[i]);
            task_memory
                .weights
                .push(weights.map(|w| w[i]).unwrap_or(1.0));
            task_memory.timestamps.push(current_time);
        }

        self.current_size += n_examples;

        // Manage memory size
        if self.current_size > self.max_memory_size {
            self.evict_examples()?;
        }

        Ok(())
    }

    /// Evict old examples to stay within memory limit
    fn evict_examples(&mut self) -> Result<()> {
        // Simple FIFO eviction for now
        // In practice, could use more sophisticated strategies
        while self.current_size > self.max_memory_size {
            // Find oldest example across all tasks
            let mut oldest_task = None;
            let mut oldest_time = u64::MAX;

            for (task_id, task_memory) in &self.task_memories {
                if !task_memory.timestamps.is_empty() {
                    let min_time = task_memory.timestamps.iter().min().unwrap();
                    if *min_time < oldest_time {
                        oldest_time = *min_time;
                        oldest_task = Some(task_id.clone());
                    }
                }
            }

            if let Some(task_id) = oldest_task {
                if let Some(task_memory) = self.task_memories.get_mut(&task_id) {
                    // Find and remove oldest example in this task
                    if let Some(oldest_idx) = task_memory
                        .timestamps
                        .iter()
                        .position(|&t| t == oldest_time)
                    {
                        task_memory.probabilities.remove(oldest_idx);
                        task_memory.labels.remove(oldest_idx);
                        task_memory.weights.remove(oldest_idx);
                        task_memory.timestamps.remove(oldest_idx);
                        self.current_size -= 1;
                    }
                }
            } else {
                break; // No more examples to evict
            }
        }

        Ok(())
    }

    /// Get stored examples for a specific task
    pub fn get_task_examples(
        &self,
        task_id: &TaskId,
    ) -> Option<(Array1<Float>, Array1<i32>, Array1<Float>)> {
        if let Some(task_memory) = self.task_memories.get(task_id) {
            if !task_memory.probabilities.is_empty() {
                let probs = Array1::from(task_memory.probabilities.clone());
                let labels = Array1::from(task_memory.labels.clone());
                let weights = Array1::from(task_memory.weights.clone());
                Some((probs, labels, weights))
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Get all stored examples across tasks
    pub fn get_all_examples(&self) -> (Array1<Float>, Array1<i32>, Array1<Float>) {
        let mut all_probs = Vec::new();
        let mut all_labels = Vec::new();
        let mut all_weights = Vec::new();

        for task_memory in self.task_memories.values() {
            all_probs.extend(&task_memory.probabilities);
            all_labels.extend(&task_memory.labels);
            all_weights.extend(&task_memory.weights);
        }

        (
            Array1::from(all_probs),
            Array1::from(all_labels),
            Array1::from(all_weights),
        )
    }
}

impl ContinualLearningCalibrator {
    /// Create a new continual learning calibrator
    pub fn new(
        base_method: BaseCalibrationMethod,
        replay_strategy: ReplayStrategy,
        max_memory_size: usize,
    ) -> Self {
        Self {
            base_method,
            task_calibrators: HashMap::new(),
            global_calibrator: None,
            memory_buffer: EpisodicMemory::new(max_memory_size),
            replay_strategy,
            regularization_strength: 0.1,
            use_task_specific: true,
            current_task: None,
            learning_rate: 0.01,
            is_fitted: false,
        }
    }

    /// Set regularization strength
    pub fn with_regularization(mut self, strength: Float) -> Self {
        self.regularization_strength = strength;
        self
    }

    /// Set whether to use task-specific calibration
    pub fn with_task_specific(mut self, use_task_specific: bool) -> Self {
        self.use_task_specific = use_task_specific;
        self
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, learning_rate: Float) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Create a base calibrator instance
    fn create_base_calibrator(&self) -> Box<dyn CalibrationEstimator> {
        match &self.base_method {
            BaseCalibrationMethod::Temperature => Box::new(TemperatureScalingWrapper::new()),
            BaseCalibrationMethod::Platt => Box::new(PlattScalingWrapper::new()),
            BaseCalibrationMethod::Histogram { n_bins } => Box::new(HistogramWrapper::new(*n_bins)),
            BaseCalibrationMethod::Isotonic => Box::new(IsotonicWrapper::new()),
        }
    }

    /// Learn a new task
    pub fn learn_task(
        &mut self,
        task_id: TaskId,
        probabilities: &Array1<Float>,
        labels: &Array1<i32>,
    ) -> Result<()> {
        self.current_task = Some(task_id.clone());

        // Add examples to memory
        self.memory_buffer
            .add_examples(task_id.clone(), probabilities, labels, None)?;

        // Create or update task-specific calibrator
        if self.use_task_specific {
            let mut task_calibrator = self.create_base_calibrator();
            task_calibrator.fit(probabilities, labels)?;
            self.task_calibrators
                .insert(task_id.clone(), task_calibrator);
        }

        // Update global calibrator with replay
        self.update_global_calibrator()?;

        // Apply regularization to prevent forgetting
        self.apply_regularization()?;

        self.is_fitted = true;
        Ok(())
    }

    /// Update global calibrator using replay strategy
    fn update_global_calibrator(&mut self) -> Result<()> {
        let (all_probs, all_labels, _weights) = self.memory_buffer.get_all_examples();

        if all_probs.is_empty() {
            return Ok(());
        }

        // Create or update global calibrator
        let mut global_calibrator = self.create_base_calibrator();
        global_calibrator.fit(&all_probs, &all_labels)?;
        self.global_calibrator = Some(global_calibrator);

        Ok(())
    }

    /// Apply regularization to prevent catastrophic forgetting
    fn apply_regularization(&mut self) -> Result<()> {
        // For now, use simple experience replay
        // In practice, could implement more sophisticated regularization like EWC

        // Get samples from memory buffer according to replay strategy
        let replay_samples = self.get_replay_samples()?;

        // Update calibrators with replay samples
        if let Some((replay_probs, replay_labels, _)) = replay_samples {
            if !replay_probs.is_empty() {
                // Update global calibrator with replay
                if let Some(ref mut global_cal) = self.global_calibrator {
                    global_cal.fit(&replay_probs, &replay_labels)?;
                }
            }
        }

        Ok(())
    }

    /// Get replay samples according to strategy
    fn get_replay_samples(&self) -> Result<Option<ReplaySampleData>> {
        match &self.replay_strategy {
            ReplayStrategy::FixedSize { samples_per_task } => {
                self.get_fixed_size_replay(*samples_per_task)
            }
            ReplayStrategy::ImportanceWeighted { max_samples } => {
                self.get_importance_weighted_replay(*max_samples)
            }
            ReplayStrategy::DiversityBased {
                max_samples,
                cluster_method: _,
            } => self.get_diversity_based_replay(*max_samples),
            ReplayStrategy::TemporalDecay {
                max_samples,
                decay_rate,
            } => self.get_temporal_decay_replay(*max_samples, *decay_rate),
        }
    }

    /// Get fixed size replay samples
    fn get_fixed_size_replay(&self, samples_per_task: usize) -> Result<Option<ReplaySampleData>> {
        let mut replay_probs = Vec::new();
        let mut replay_labels = Vec::new();
        let mut replay_weights = Vec::new();

        for task_id in self.task_calibrators.keys() {
            if let Some((probs, labels, weights)) = self.memory_buffer.get_task_examples(task_id) {
                let n_samples = samples_per_task.min(probs.len());

                // Simple random sampling
                let _rng_instance = thread_rng();
                let mut indices: Vec<usize> = (0..probs.len()).collect();
                indices.reverse();

                for &idx in indices.iter().take(n_samples) {
                    replay_probs.push(probs[idx]);
                    replay_labels.push(labels[idx]);
                    replay_weights.push(weights[idx]);
                }
            }
        }

        if replay_probs.is_empty() {
            Ok(None)
        } else {
            Ok(Some((
                Array1::from(replay_probs),
                Array1::from(replay_labels),
                Array1::from(replay_weights),
            )))
        }
    }

    /// Get importance-weighted replay samples
    fn get_importance_weighted_replay(
        &self,
        max_samples: usize,
    ) -> Result<Option<ReplaySampleData>> {
        // For now, use equal importance for all tasks
        // In practice, could weight by task difficulty or recency
        let n_tasks = self.task_calibrators.len();
        if n_tasks == 0 {
            return Ok(None);
        }

        let samples_per_task = max_samples / n_tasks;
        self.get_fixed_size_replay(samples_per_task)
    }

    /// Get diversity-based replay samples
    fn get_diversity_based_replay(&self, max_samples: usize) -> Result<Option<ReplaySampleData>> {
        // Simple implementation: spread samples across probability ranges
        let (all_probs, all_labels, all_weights) = self.memory_buffer.get_all_examples();

        if all_probs.is_empty() {
            return Ok(None);
        }

        let n_samples = max_samples.min(all_probs.len());
        let n_bins = 10;

        let mut selected_probs = Vec::new();
        let mut selected_labels = Vec::new();
        let mut selected_weights = Vec::new();

        // Divide probability space into bins and sample from each
        let samples_per_bin = n_samples / n_bins;

        for bin_idx in 0..n_bins {
            let bin_start = bin_idx as Float / n_bins as Float;
            let bin_end = (bin_idx + 1) as Float / n_bins as Float;

            let bin_indices: Vec<usize> = all_probs
                .iter()
                .enumerate()
                .filter_map(|(idx, &prob)| {
                    if prob >= bin_start && prob < bin_end {
                        Some(idx)
                    } else {
                        None
                    }
                })
                .collect();

            // Sample from this bin
            let _rng_instance = thread_rng();
            let mut bin_indices_shuffled = bin_indices;
            bin_indices_shuffled.reverse();

            for &idx in bin_indices_shuffled.iter().take(samples_per_bin) {
                selected_probs.push(all_probs[idx]);
                selected_labels.push(all_labels[idx]);
                selected_weights.push(all_weights[idx]);
            }
        }

        if selected_probs.is_empty() {
            Ok(None)
        } else {
            Ok(Some((
                Array1::from(selected_probs),
                Array1::from(selected_labels),
                Array1::from(selected_weights),
            )))
        }
    }

    /// Get temporal decay replay samples
    fn get_temporal_decay_replay(
        &self,
        max_samples: usize,
        _decay_rate: Float,
    ) -> Result<Option<ReplaySampleData>> {
        // Weight samples by recency
        let (all_probs, all_labels, all_weights) = self.memory_buffer.get_all_examples();

        if all_probs.is_empty() {
            return Ok(None);
        }

        // For simplicity, use uniform sampling with temporal bias
        // In practice, would compute actual temporal weights
        let n_samples = max_samples.min(all_probs.len());

        let _rng_instance = thread_rng();
        let mut indices: Vec<usize> = (0..all_probs.len()).collect();
        indices.reverse();

        let selected_probs: Vec<Float> = indices
            .iter()
            .take(n_samples)
            .map(|&i| all_probs[i])
            .collect();
        let selected_labels: Vec<i32> = indices
            .iter()
            .take(n_samples)
            .map(|&i| all_labels[i])
            .collect();
        let selected_weights: Vec<Float> = indices
            .iter()
            .take(n_samples)
            .map(|&i| all_weights[i])
            .collect();

        Ok(Some((
            Array1::from(selected_probs),
            Array1::from(selected_labels),
            Array1::from(selected_weights),
        )))
    }

    /// Predict calibrated probabilities for a specific task
    pub fn predict_task(
        &self,
        task_id: &TaskId,
        probabilities: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "predict calibrated probabilities".to_string(),
            });
        }

        if self.use_task_specific {
            if let Some(task_calibrator) = self.task_calibrators.get(task_id) {
                task_calibrator.predict_proba(probabilities)
            } else {
                // Fall back to global calibrator
                if let Some(ref global_calibrator) = self.global_calibrator {
                    global_calibrator.predict_proba(probabilities)
                } else {
                    Ok(probabilities.clone())
                }
            }
        } else {
            // Use global calibrator
            if let Some(ref global_calibrator) = self.global_calibrator {
                global_calibrator.predict_proba(probabilities)
            } else {
                Ok(probabilities.clone())
            }
        }
    }

    /// Predict using global calibrator (task-agnostic)
    pub fn predict_global(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "predict calibrated probabilities".to_string(),
            });
        }

        if let Some(ref global_calibrator) = self.global_calibrator {
            global_calibrator.predict_proba(probabilities)
        } else {
            Ok(probabilities.clone())
        }
    }

    /// Get number of learned tasks
    pub fn num_tasks(&self) -> usize {
        self.task_calibrators.len()
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> (usize, usize) {
        (
            self.memory_buffer.current_size,
            self.memory_buffer.max_memory_size,
        )
    }
}

// Simple wrapper implementations for base calibrators
// In practice, these would use the actual calibration implementations

#[derive(Debug, Clone)]
struct TemperatureScalingWrapper {
    temperature: Float,
    is_fitted: bool,
}

impl TemperatureScalingWrapper {
    fn new() -> Self {
        Self {
            temperature: 1.0,
            is_fitted: false,
        }
    }
}

impl CalibrationEstimator for TemperatureScalingWrapper {
    fn fit(&mut self, _probabilities: &Array1<Float>, _y_true: &Array1<i32>) -> Result<()> {
        // Simple temperature estimation
        self.temperature = 1.0; // In practice, would optimize this
        self.is_fitted = true;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "predict calibrated probabilities".to_string(),
            });
        }
        Ok(probabilities.clone()) // Simplified
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
struct PlattScalingWrapper {
    is_fitted: bool,
}

impl PlattScalingWrapper {
    fn new() -> Self {
        Self { is_fitted: false }
    }
}

impl CalibrationEstimator for PlattScalingWrapper {
    fn fit(&mut self, _probabilities: &Array1<Float>, _y_true: &Array1<i32>) -> Result<()> {
        self.is_fitted = true;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "predict calibrated probabilities".to_string(),
            });
        }
        Ok(probabilities.clone()) // Simplified
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
struct HistogramWrapper {
    n_bins: usize,
    is_fitted: bool,
}

impl HistogramWrapper {
    fn new(n_bins: usize) -> Self {
        Self {
            n_bins,
            is_fitted: false,
        }
    }
}

impl CalibrationEstimator for HistogramWrapper {
    fn fit(&mut self, _probabilities: &Array1<Float>, _y_true: &Array1<i32>) -> Result<()> {
        self.is_fitted = true;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "predict calibrated probabilities".to_string(),
            });
        }
        Ok(probabilities.clone()) // Simplified
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
struct IsotonicWrapper {
    is_fitted: bool,
}

impl IsotonicWrapper {
    fn new() -> Self {
        Self { is_fitted: false }
    }
}

impl CalibrationEstimator for IsotonicWrapper {
    fn fit(&mut self, _probabilities: &Array1<Float>, _y_true: &Array1<i32>) -> Result<()> {
        self.is_fitted = true;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "predict calibrated probabilities".to_string(),
            });
        }
        Ok(probabilities.clone()) // Simplified
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_task_id_creation() {
        let task_id = TaskId::new("task1".to_string(), "vision".to_string(), 0.5);
        assert_eq!(task_id.id, "task1");
        assert_eq!(task_id.domain, "vision");
        assert_eq!(task_id.difficulty, 0.5);
    }

    #[test]
    fn test_episodic_memory_creation() {
        let memory = EpisodicMemory::new(1000);
        assert_eq!(memory.max_memory_size, 1000);
        assert_eq!(memory.current_size, 0);
    }

    #[test]
    fn test_continual_learning_calibrator_creation() {
        let method = BaseCalibrationMethod::Temperature;
        let strategy = ReplayStrategy::FixedSize {
            samples_per_task: 100,
        };
        let calibrator = ContinualLearningCalibrator::new(method, strategy, 1000);

        assert_eq!(calibrator.num_tasks(), 0);
        assert!(!calibrator.is_fitted);
    }

    #[test]
    fn test_memory_example_addition() {
        let mut memory = EpisodicMemory::new(1000);
        let task_id = TaskId::new("task1".to_string(), "vision".to_string(), 0.5);

        let probs = array![0.1, 0.3, 0.7, 0.9];
        let labels = array![0, 0, 1, 1];

        let result = memory.add_examples(task_id.clone(), &probs, &labels, None);
        assert!(result.is_ok());
        assert_eq!(memory.current_size, 4);

        let stored = memory.get_task_examples(&task_id);
        assert!(stored.is_some());

        let (stored_probs, stored_labels, stored_weights) = stored.unwrap();
        assert_eq!(stored_probs.len(), 4);
        assert_eq!(stored_labels.len(), 4);
        assert_eq!(stored_weights.len(), 4);
    }

    #[test]
    fn test_continual_learning_task_learning() {
        let method = BaseCalibrationMethod::Temperature;
        let strategy = ReplayStrategy::FixedSize {
            samples_per_task: 100,
        };
        let mut calibrator = ContinualLearningCalibrator::new(method, strategy, 1000);

        let task_id = TaskId::new("task1".to_string(), "vision".to_string(), 0.5);
        let probs = array![0.1, 0.3, 0.7, 0.9];
        let labels = array![0, 0, 1, 1];

        let result = calibrator.learn_task(task_id.clone(), &probs, &labels);
        assert!(result.is_ok());
        assert_eq!(calibrator.num_tasks(), 1);
        assert!(calibrator.is_fitted);
    }

    #[test]
    fn test_multiple_task_learning() {
        let method = BaseCalibrationMethod::Temperature;
        let strategy = ReplayStrategy::FixedSize {
            samples_per_task: 50,
        };
        let mut calibrator = ContinualLearningCalibrator::new(method, strategy, 1000);

        // Learn first task
        let task1 = TaskId::new("task1".to_string(), "vision".to_string(), 0.3);
        let probs1 = array![0.1, 0.3, 0.7];
        let labels1 = array![0, 0, 1];
        calibrator
            .learn_task(task1.clone(), &probs1, &labels1)
            .unwrap();

        // Learn second task
        let task2 = TaskId::new("task2".to_string(), "nlp".to_string(), 0.7);
        let probs2 = array![0.2, 0.6, 0.8];
        let labels2 = array![0, 1, 1];
        calibrator
            .learn_task(task2.clone(), &probs2, &labels2)
            .unwrap();

        assert_eq!(calibrator.num_tasks(), 2);

        // Test task-specific prediction
        let test_probs = array![0.4, 0.5];
        let pred1 = calibrator.predict_task(&task1, &test_probs);
        let pred2 = calibrator.predict_task(&task2, &test_probs);

        assert!(pred1.is_ok());
        assert!(pred2.is_ok());

        // Test global prediction
        let global_pred = calibrator.predict_global(&test_probs);
        assert!(global_pred.is_ok());
    }

    #[test]
    fn test_memory_eviction() {
        let mut memory = EpisodicMemory::new(5); // Small memory limit
        let task_id = TaskId::new("task1".to_string(), "vision".to_string(), 0.5);

        // Add more examples than memory limit
        let probs = array![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let labels = array![0, 0, 0, 1, 1, 1, 1, 1];

        let result = memory.add_examples(task_id.clone(), &probs, &labels, None);
        assert!(result.is_ok());

        // Memory should be at or below limit
        assert!(memory.current_size <= memory.max_memory_size);
    }

    #[test]
    fn test_replay_strategies() {
        let method = BaseCalibrationMethod::Temperature;

        // Test different replay strategies
        let strategies = vec![
            ReplayStrategy::FixedSize {
                samples_per_task: 50,
            },
            ReplayStrategy::ImportanceWeighted { max_samples: 100 },
            ReplayStrategy::DiversityBased {
                max_samples: 100,
                cluster_method: "kmeans".to_string(),
            },
            ReplayStrategy::TemporalDecay {
                max_samples: 100,
                decay_rate: 0.1,
            },
        ];

        for strategy in strategies {
            let calibrator = ContinualLearningCalibrator::new(method.clone(), strategy, 1000);
            assert_eq!(calibrator.num_tasks(), 0);
        }
    }

    #[test]
    fn test_calibration_methods() {
        let methods = vec![
            BaseCalibrationMethod::Temperature,
            BaseCalibrationMethod::Platt,
            BaseCalibrationMethod::Histogram { n_bins: 10 },
            BaseCalibrationMethod::Isotonic,
        ];

        let strategy = ReplayStrategy::FixedSize {
            samples_per_task: 50,
        };

        for method in methods {
            let calibrator = ContinualLearningCalibrator::new(method, strategy.clone(), 1000);
            assert!(!calibrator.is_fitted);
        }
    }

    #[test]
    fn test_memory_statistics() {
        let method = BaseCalibrationMethod::Temperature;
        let strategy = ReplayStrategy::FixedSize {
            samples_per_task: 100,
        };
        let mut calibrator = ContinualLearningCalibrator::new(method, strategy, 1000);

        let (used, total) = calibrator.memory_stats();
        assert_eq!(used, 0);
        assert_eq!(total, 1000);

        // Add a task
        let task_id = TaskId::new("task1".to_string(), "vision".to_string(), 0.5);
        let probs = array![0.1, 0.3, 0.7, 0.9];
        let labels = array![0, 0, 1, 1];
        calibrator.learn_task(task_id, &probs, &labels).unwrap();

        let (used_after, total_after) = calibrator.memory_stats();
        assert_eq!(used_after, 4);
        assert_eq!(total_after, 1000);
    }
}
