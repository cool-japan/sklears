//! Continual learning pipeline components
//!
//! This module provides continual learning capabilities including catastrophic forgetting
//! prevention, memory-based approaches, and progressive learning strategies.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::thread_rng;
use scirs2_core::random::Rng;
use sklears_core::{
    error::Result as SklResult,
    prelude::{Predict, SklearsError},
    traits::{Estimator, Fit, Untrained},
    types::Float,
};
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;

use crate::{PipelinePredictor, PipelineStep};

/// Task representation for continual learning
#[derive(Debug, Clone)]
pub struct Task {
    /// Unique task identifier
    pub id: String,
    /// Task features
    pub features: Array2<f64>,
    /// Task targets
    pub targets: Array1<f64>,
    /// Task metadata
    pub metadata: HashMap<String, String>,
    /// Task importance weights
    pub importance_weights: Option<HashMap<String, f64>>,
    /// Task learning statistics
    pub statistics: TaskStatistics,
}

/// Statistics for a learning task
#[derive(Debug, Clone)]
pub struct TaskStatistics {
    /// Number of training samples
    pub n_samples: usize,
    /// Number of features
    pub n_features: usize,
    /// Task difficulty (estimated)
    pub difficulty: f64,
    /// Performance metrics
    pub performance: HashMap<String, f64>,
    /// Learning time
    pub learning_time: f64,
}

impl Task {
    /// Create a new task
    #[must_use]
    pub fn new(id: String, features: Array2<f64>, targets: Array1<f64>) -> Self {
        let n_samples = features.nrows();
        let n_features = features.ncols();

        Self {
            id,
            features,
            targets,
            metadata: HashMap::new(),
            importance_weights: None,
            statistics: TaskStatistics {
                n_samples,
                n_features,
                difficulty: 1.0, // Default difficulty
                performance: HashMap::new(),
                learning_time: 0.0,
            },
        }
    }

    /// Set task metadata
    #[must_use]
    pub fn with_metadata(mut self, metadata: HashMap<String, String>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Set importance weights
    #[must_use]
    pub fn with_importance_weights(mut self, weights: HashMap<String, f64>) -> Self {
        self.importance_weights = Some(weights);
        self
    }

    /// Estimate task difficulty based on data characteristics
    pub fn estimate_difficulty(&mut self) {
        let feature_variance = self.features.var_axis(Axis(0), 1.0).mean().unwrap_or(1.0);
        let target_variance = self.targets.var(1.0);

        // Simple difficulty estimation based on variance
        self.statistics.difficulty = (feature_variance + target_variance).max(0.1);
    }
}

/// Continual learning strategy
#[derive(Debug, Clone)]
pub enum ContinualLearningStrategy {
    /// Elastic Weight Consolidation (EWC)
    ElasticWeightConsolidation {
        /// Regularization strength
        lambda: f64,
        /// Fisher information estimation samples
        fisher_samples: usize,
    },
    /// Progressive Neural Networks
    ProgressiveNetworks {
        /// Maximum number of parallel columns
        max_columns: usize,
        /// Lateral connection strength
        lateral_strength: f64,
    },
    /// Experience Replay
    ExperienceReplay {
        /// Memory buffer size
        buffer_size: usize,
        /// Replay batch size
        replay_batch_size: usize,
        /// Replay frequency
        replay_frequency: usize,
    },
    /// Learning without Forgetting (`LwF`)
    LearningWithoutForgetting {
        /// Distillation temperature
        temperature: f64,
        /// Distillation weight
        distillation_weight: f64,
    },
    /// Memory-Augmented Networks
    MemoryAugmented {
        /// Memory size
        memory_size: usize,
        /// Memory read heads
        read_heads: usize,
        /// Memory write strength
        write_strength: f64,
    },
    /// Gradient Episodic Memory (GEM)
    GradientEpisodicMemory {
        /// Memory buffer size
        memory_size: usize,
        /// Inequality constraint tolerance
        tolerance: f64,
    },
}

/// Memory buffer for continual learning
#[derive(Debug, Clone)]
pub struct MemoryBuffer {
    /// Maximum buffer size
    max_size: usize,
    /// Stored samples
    samples: VecDeque<MemorySample>,
    /// Task distributions
    task_distributions: HashMap<String, usize>,
    /// Sampling strategy
    sampling_strategy: SamplingStrategy,
}

/// Memory sample with task information
#[derive(Debug, Clone)]
pub struct MemorySample {
    /// Sample features
    pub features: Array1<f64>,
    /// Sample target
    pub target: f64,
    /// Source task ID
    pub task_id: String,
    /// Sample importance
    pub importance: f64,
    /// Gradient information (for GEM)
    pub gradient_info: Option<HashMap<String, f64>>,
}

/// Sampling strategy for memory buffer
#[derive(Debug, Clone)]
pub enum SamplingStrategy {
    /// Random sampling
    Random,
    /// Reservoir sampling
    Reservoir,
    /// Importance-based sampling
    ImportanceBased,
    /// Task-balanced sampling
    TaskBalanced,
    /// Gradient-based sampling (for GEM)
    GradientBased,
}

impl MemoryBuffer {
    /// Create a new memory buffer
    #[must_use]
    pub fn new(max_size: usize, sampling_strategy: SamplingStrategy) -> Self {
        Self {
            max_size,
            samples: VecDeque::new(),
            task_distributions: HashMap::new(),
            sampling_strategy,
        }
    }

    /// Add a sample to the buffer
    pub fn add_sample(&mut self, sample: MemorySample) {
        // Update task distribution
        *self
            .task_distributions
            .entry(sample.task_id.clone())
            .or_insert(0) += 1;

        if self.samples.len() >= self.max_size {
            match self.sampling_strategy {
                SamplingStrategy::Random => {
                    let replace_idx = thread_rng().gen_range(0..self.samples.len());
                    if let Some(old_sample) = self.samples.get(replace_idx) {
                        let count = self
                            .task_distributions
                            .get_mut(&old_sample.task_id)
                            .unwrap();
                        *count -= 1;
                        if *count == 0 {
                            self.task_distributions.remove(&old_sample.task_id);
                        }
                    }
                    self.samples[replace_idx] = sample;
                }
                SamplingStrategy::Reservoir => {
                    // Standard reservoir sampling
                    let replace_idx = thread_rng().gen_range(0..(self.samples.len() + 1));
                    if replace_idx < self.samples.len() {
                        if let Some(old_sample) = self.samples.get(replace_idx) {
                            let count = self
                                .task_distributions
                                .get_mut(&old_sample.task_id)
                                .unwrap();
                            *count -= 1;
                            if *count == 0 {
                                self.task_distributions.remove(&old_sample.task_id);
                            }
                        }
                        self.samples[replace_idx] = sample;
                    }
                }
                SamplingStrategy::ImportanceBased => {
                    // Replace sample with lowest importance
                    let min_importance_idx = self
                        .samples
                        .iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| a.importance.partial_cmp(&b.importance).unwrap())
                        .map_or(0, |(idx, _)| idx);

                    if sample.importance > self.samples[min_importance_idx].importance {
                        if let Some(old_sample) = self.samples.get(min_importance_idx) {
                            let count = self
                                .task_distributions
                                .get_mut(&old_sample.task_id)
                                .unwrap();
                            *count -= 1;
                            if *count == 0 {
                                self.task_distributions.remove(&old_sample.task_id);
                            }
                        }
                        self.samples[min_importance_idx] = sample;
                    }
                }
                SamplingStrategy::TaskBalanced => {
                    // Replace from overrepresented task
                    let max_task = self
                        .task_distributions
                        .iter()
                        .max_by_key(|(_, &count)| count)
                        .map(|(task_id, _)| task_id.clone());

                    if let Some(overrep_task) = max_task {
                        if let Some(idx) =
                            self.samples.iter().position(|s| s.task_id == overrep_task)
                        {
                            let count = self.task_distributions.get_mut(&overrep_task).unwrap();
                            *count -= 1;
                            if *count == 0 {
                                self.task_distributions.remove(&overrep_task);
                            }
                            self.samples[idx] = sample;
                        }
                    }
                }
                SamplingStrategy::GradientBased => {
                    // For GEM, replace based on gradient diversity
                    // Simplified: replace random sample
                    let replace_idx = thread_rng().gen_range(0..self.samples.len());
                    if let Some(old_sample) = self.samples.get(replace_idx) {
                        let count = self
                            .task_distributions
                            .get_mut(&old_sample.task_id)
                            .unwrap();
                        *count -= 1;
                        if *count == 0 {
                            self.task_distributions.remove(&old_sample.task_id);
                        }
                    }
                    self.samples[replace_idx] = sample;
                }
            }
        } else {
            self.samples.push_back(sample);
        }
    }

    /// Sample from the buffer
    #[must_use]
    pub fn sample(&self, n_samples: usize) -> Vec<&MemorySample> {
        if self.samples.is_empty() {
            return Vec::new();
        }

        let n_samples = n_samples.min(self.samples.len());
        let mut sampled = Vec::new();

        match self.sampling_strategy {
            SamplingStrategy::Random | SamplingStrategy::Reservoir => {
                for _ in 0..n_samples {
                    let idx = thread_rng().gen_range(0..self.samples.len());
                    sampled.push(&self.samples[idx]);
                }
            }
            SamplingStrategy::ImportanceBased => {
                // Sample based on importance weights
                let total_importance: f64 = self.samples.iter().map(|s| s.importance).sum();
                for _ in 0..n_samples {
                    let target = thread_rng().gen::<f64>() * total_importance;
                    let mut cumulative = 0.0;
                    for sample in &self.samples {
                        cumulative += sample.importance;
                        if cumulative >= target {
                            sampled.push(sample);
                            break;
                        }
                    }
                }
            }
            SamplingStrategy::TaskBalanced => {
                // Ensure balanced representation across tasks
                let unique_tasks: Vec<String> = self.task_distributions.keys().cloned().collect();
                if !unique_tasks.is_empty() {
                    let samples_per_task = n_samples / unique_tasks.len();
                    let extra_samples = n_samples % unique_tasks.len();

                    for (i, task_id) in unique_tasks.iter().enumerate() {
                        let task_samples: Vec<&MemorySample> = self
                            .samples
                            .iter()
                            .filter(|s| &s.task_id == task_id)
                            .collect();

                        let task_sample_count = samples_per_task + usize::from(i < extra_samples);
                        for _ in 0..task_sample_count.min(task_samples.len()) {
                            let idx = thread_rng().gen_range(0..task_samples.len());
                            sampled.push(task_samples[idx]);
                        }
                    }
                }
            }
            SamplingStrategy::GradientBased => {
                // For GEM, sample based on gradient information
                // Simplified: random sampling for now
                for _ in 0..n_samples {
                    let idx = thread_rng().gen_range(0..self.samples.len());
                    sampled.push(&self.samples[idx]);
                }
            }
        }

        sampled
    }

    /// Get buffer statistics
    #[must_use]
    pub fn statistics(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        stats.insert("total_samples".to_string(), self.samples.len() as f64);
        stats.insert(
            "unique_tasks".to_string(),
            self.task_distributions.len() as f64,
        );

        if !self.samples.is_empty() {
            let avg_importance =
                self.samples.iter().map(|s| s.importance).sum::<f64>() / self.samples.len() as f64;
            stats.insert("average_importance".to_string(), avg_importance);
        }

        stats
    }
}

/// Continual learning pipeline
#[derive(Debug)]
pub struct ContinualLearningPipeline<S = Untrained> {
    state: S,
    base_estimator: Option<Box<dyn PipelinePredictor>>,
    strategy: ContinualLearningStrategy,
    memory_buffer: MemoryBuffer,
    learned_tasks: Vec<String>,
    current_task_id: Option<String>,
}

/// Trained state for `ContinualLearningPipeline`
#[derive(Debug)]
pub struct ContinualLearningPipelineTrained {
    fitted_estimator: Box<dyn PipelinePredictor>,
    strategy: ContinualLearningStrategy,
    memory_buffer: MemoryBuffer,
    learned_tasks: Vec<String>,
    task_performance: HashMap<String, HashMap<String, f64>>,
    importance_weights: HashMap<String, f64>,
    n_features_in: usize,
    feature_names_in: Option<Vec<String>>,
}

impl ContinualLearningPipeline<Untrained> {
    /// Create a new continual learning pipeline
    #[must_use]
    pub fn new(
        base_estimator: Box<dyn PipelinePredictor>,
        strategy: ContinualLearningStrategy,
    ) -> Self {
        let memory_buffer = match &strategy {
            ContinualLearningStrategy::ExperienceReplay { buffer_size, .. } => {
                MemoryBuffer::new(*buffer_size, SamplingStrategy::Random)
            }
            ContinualLearningStrategy::GradientEpisodicMemory { memory_size, .. } => {
                MemoryBuffer::new(*memory_size, SamplingStrategy::GradientBased)
            }
            ContinualLearningStrategy::MemoryAugmented { memory_size, .. } => {
                MemoryBuffer::new(*memory_size, SamplingStrategy::ImportanceBased)
            }
            _ => MemoryBuffer::new(1000, SamplingStrategy::Random), // Default
        };

        Self {
            state: Untrained,
            base_estimator: Some(base_estimator),
            strategy,
            memory_buffer,
            learned_tasks: Vec::new(),
            current_task_id: None,
        }
    }

    /// Create EWC pipeline
    #[must_use]
    pub fn elastic_weight_consolidation(
        base_estimator: Box<dyn PipelinePredictor>,
        lambda: f64,
        fisher_samples: usize,
    ) -> Self {
        Self::new(
            base_estimator,
            ContinualLearningStrategy::ElasticWeightConsolidation {
                lambda,
                fisher_samples,
            },
        )
    }

    /// Create experience replay pipeline
    #[must_use]
    pub fn experience_replay(
        base_estimator: Box<dyn PipelinePredictor>,
        buffer_size: usize,
        replay_batch_size: usize,
        replay_frequency: usize,
    ) -> Self {
        Self::new(
            base_estimator,
            ContinualLearningStrategy::ExperienceReplay {
                buffer_size,
                replay_batch_size,
                replay_frequency,
            },
        )
    }

    /// Create Learning without Forgetting pipeline
    #[must_use]
    pub fn learning_without_forgetting(
        base_estimator: Box<dyn PipelinePredictor>,
        temperature: f64,
        distillation_weight: f64,
    ) -> Self {
        Self::new(
            base_estimator,
            ContinualLearningStrategy::LearningWithoutForgetting {
                temperature,
                distillation_weight,
            },
        )
    }

    /// Set current task ID
    #[must_use]
    pub fn set_current_task(mut self, task_id: String) -> Self {
        self.current_task_id = Some(task_id);
        self
    }
}

impl Estimator for ContinualLearningPipeline<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, Option<&ArrayView1<'_, Float>>>
    for ContinualLearningPipeline<Untrained>
{
    type Fitted = ContinualLearningPipeline<ContinualLearningPipelineTrained>;

    fn fit(
        mut self,
        x: &ArrayView2<'_, Float>,
        y: &Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<Self::Fitted> {
        if let Some(y_values) = y.as_ref() {
            let mut base_estimator = self.base_estimator.take().ok_or_else(|| {
                SklearsError::InvalidInput("No base estimator provided".to_string())
            })?;

            // Apply continual learning strategy
            let importance_weights =
                self.apply_continual_learning_strategy(&mut base_estimator, x, y_values)?;

            let task_id = self
                .current_task_id
                .clone()
                .unwrap_or_else(|| "default_task".to_string());
            self.learned_tasks.push(task_id.clone());

            let mut task_performance = HashMap::new();
            let mut perf_metrics = HashMap::new();
            perf_metrics.insert("training_completed".to_string(), 1.0);
            task_performance.insert(task_id, perf_metrics);

            Ok(ContinualLearningPipeline {
                state: ContinualLearningPipelineTrained {
                    fitted_estimator: base_estimator,
                    strategy: self.strategy,
                    memory_buffer: self.memory_buffer,
                    learned_tasks: self.learned_tasks,
                    task_performance,
                    importance_weights,
                    n_features_in: x.ncols(),
                    feature_names_in: None,
                },
                base_estimator: None,
                strategy: ContinualLearningStrategy::ExperienceReplay {
                    buffer_size: 1000,
                    replay_batch_size: 32,
                    replay_frequency: 10,
                },
                memory_buffer: MemoryBuffer::new(1000, SamplingStrategy::Random),
                learned_tasks: Vec::new(),
                current_task_id: None,
            })
        } else {
            Err(SklearsError::InvalidInput(
                "Target values required for continual learning".to_string(),
            ))
        }
    }
}

impl ContinualLearningPipeline<Untrained> {
    /// Apply continual learning strategy
    fn apply_continual_learning_strategy(
        &mut self,
        estimator: &mut Box<dyn PipelinePredictor>,
        x: &ArrayView2<'_, Float>,
        y: &ArrayView1<'_, Float>,
    ) -> SklResult<HashMap<String, f64>> {
        let mut importance_weights = HashMap::new();

        match &self.strategy {
            ContinualLearningStrategy::ElasticWeightConsolidation {
                lambda,
                fisher_samples,
            } => {
                // Simulate EWC by computing importance weights
                for i in 0..*fisher_samples.min(&x.nrows()) {
                    let param_name = format!("param_{i}");
                    let importance = self.compute_fisher_information(x, y, i);
                    importance_weights.insert(param_name, importance * lambda);
                }

                // Fit with regularization (simulated)
                estimator.fit(x, y)?;
            }
            ContinualLearningStrategy::ExperienceReplay {
                replay_batch_size,
                replay_frequency,
                ..
            } => {
                // Store current samples in memory
                for i in 0..x.nrows() {
                    let sample = MemorySample {
                        features: x.row(i).mapv(|v| v),
                        target: y[i],
                        task_id: self
                            .current_task_id
                            .clone()
                            .unwrap_or_else(|| "default".to_string()),
                        importance: 1.0,
                        gradient_info: None,
                    };
                    self.memory_buffer.add_sample(sample);
                }

                // Train with replay
                for epoch in 0..*replay_frequency {
                    // Train on current data
                    estimator.fit(x, y)?;

                    // Replay from memory
                    let replay_samples = self.memory_buffer.sample(*replay_batch_size);
                    if !replay_samples.is_empty() {
                        // Create replay batch (simplified)
                        let replay_x = Array2::from_shape_vec(
                            (replay_samples.len(), x.ncols()),
                            replay_samples
                                .iter()
                                .flat_map(|s| s.features.iter().copied())
                                .collect(),
                        )
                        .map_err(|e| SklearsError::InvalidData {
                            reason: format!("Replay batch creation failed: {e}"),
                        })?;

                        let replay_y = Array1::from_vec(
                            replay_samples.iter().map(|s| s.target as Float).collect(),
                        );

                        estimator.fit(&replay_x.view(), &replay_y.view())?;
                    }
                }
            }
            ContinualLearningStrategy::LearningWithoutForgetting {
                temperature,
                distillation_weight,
            } => {
                // Simulate LwF by storing distillation info
                importance_weights.insert("temperature".to_string(), *temperature);
                importance_weights.insert("distillation_weight".to_string(), *distillation_weight);

                estimator.fit(x, y)?;
            }
            ContinualLearningStrategy::ProgressiveNetworks {
                max_columns,
                lateral_strength,
            } => {
                // Simulate progressive networks
                importance_weights.insert("columns".to_string(), self.learned_tasks.len() as f64);
                importance_weights.insert("lateral_strength".to_string(), *lateral_strength);

                estimator.fit(x, y)?;
            }
            ContinualLearningStrategy::MemoryAugmented {
                memory_size,
                read_heads,
                write_strength,
            } => {
                // Simulate memory-augmented networks
                importance_weights.insert(
                    "memory_usage".to_string(),
                    self.memory_buffer.samples.len() as f64 / *memory_size as f64,
                );
                importance_weights.insert("read_heads".to_string(), *read_heads as f64);
                importance_weights.insert("write_strength".to_string(), *write_strength);

                estimator.fit(x, y)?;
            }
            ContinualLearningStrategy::GradientEpisodicMemory {
                memory_size,
                tolerance,
            } => {
                // Store samples with gradient information
                for i in 0..x.nrows() {
                    let mut gradient_info = HashMap::new();
                    gradient_info.insert("grad_norm".to_string(), thread_rng().gen::<f64>()); // Placeholder

                    let sample = MemorySample {
                        features: x.row(i).mapv(|v| v),
                        target: y[i],
                        task_id: self
                            .current_task_id
                            .clone()
                            .unwrap_or_else(|| "default".to_string()),
                        importance: 1.0,
                        gradient_info: Some(gradient_info),
                    };
                    self.memory_buffer.add_sample(sample);
                }

                importance_weights.insert(
                    "memory_utilization".to_string(),
                    self.memory_buffer.samples.len() as f64 / *memory_size as f64,
                );
                importance_weights.insert("tolerance".to_string(), *tolerance);

                estimator.fit(x, y)?;
            }
        }

        Ok(importance_weights)
    }

    /// Compute Fisher information approximation
    fn compute_fisher_information(
        &self,
        x: &ArrayView2<'_, Float>,
        y: &ArrayView1<'_, Float>,
        param_idx: usize,
    ) -> f64 {
        // Simplified Fisher information computation
        if param_idx < x.ncols() {
            let feature_variance = x.column(param_idx).var(1.0);
            feature_variance.max(1e-8) // Avoid zero importance
        } else {
            1e-4 // Default small importance
        }
    }
}

impl ContinualLearningPipeline<ContinualLearningPipelineTrained> {
    /// Predict using the fitted continual learning model
    pub fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<f64>> {
        self.state.fitted_estimator.predict(x)
    }

    /// Learn a new task
    pub fn learn_task(&mut self, task: Task) -> SklResult<()> {
        // Update current task performance
        let mut task_perf = HashMap::new();
        task_perf.insert("samples".to_string(), task.statistics.n_samples as f64);
        task_perf.insert("difficulty".to_string(), task.statistics.difficulty);

        self.state
            .task_performance
            .insert(task.id.clone(), task_perf);

        // Apply continual learning for new task
        let x_view = task.features.view().mapv(|v| v as Float);
        let y_view = task.targets.view().mapv(|v| v as Float);

        match &self.state.strategy {
            ContinualLearningStrategy::ExperienceReplay {
                replay_batch_size, ..
            } => {
                // Store new task samples
                for i in 0..task.features.nrows() {
                    let sample = MemorySample {
                        features: task.features.row(i).to_owned(),
                        target: task.targets[i],
                        task_id: task.id.clone(),
                        importance: 1.0,
                        gradient_info: None,
                    };
                    self.state.memory_buffer.add_sample(sample);
                }

                // Train with replay
                self.state
                    .fitted_estimator
                    .fit(&x_view.view(), &y_view.view())?;

                // Replay from memory
                let replay_samples = self.state.memory_buffer.sample(*replay_batch_size);
                if !replay_samples.is_empty() {
                    // Create replay batch
                    let n_features = task.features.ncols();
                    let replay_x = Array2::from_shape_vec(
                        (replay_samples.len(), n_features),
                        replay_samples
                            .iter()
                            .flat_map(|s| s.features.iter().copied().map(|v| v as Float))
                            .collect(),
                    )
                    .map_err(|e| SklearsError::InvalidData {
                        reason: format!("Replay batch creation failed: {e}"),
                    })?;

                    let replay_y = Array1::from_vec(
                        replay_samples.iter().map(|s| s.target as Float).collect(),
                    );

                    self.state
                        .fitted_estimator
                        .fit(&replay_x.view(), &replay_y.view())?;
                }
            }
            _ => {
                // Default: just train on new task
                self.state
                    .fitted_estimator
                    .fit(&x_view.view(), &y_view.view())?;
            }
        }

        if !self.state.learned_tasks.contains(&task.id) {
            self.state.learned_tasks.push(task.id);
        }

        Ok(())
    }

    /// Evaluate catastrophic forgetting
    pub fn evaluate_forgetting(&self, previous_tasks: &[Task]) -> SklResult<HashMap<String, f64>> {
        let mut forgetting_metrics = HashMap::new();

        for task in previous_tasks {
            let x_view = task.features.view().mapv(|v| v as Float);
            let predictions = self.predict(&x_view.view())?;

            // Simple accuracy computation
            let correct = predictions
                .iter()
                .zip(task.targets.iter())
                .filter(|(&pred, &actual)| (pred - actual).abs() < 0.5)
                .count();

            let accuracy = correct as f64 / task.targets.len() as f64;
            forgetting_metrics.insert(format!("task_{}_accuracy", task.id), accuracy);
        }

        // Compute average forgetting
        if !forgetting_metrics.is_empty() {
            let avg_accuracy =
                forgetting_metrics.values().sum::<f64>() / forgetting_metrics.len() as f64;
            forgetting_metrics.insert("average_accuracy".to_string(), avg_accuracy);
        }

        Ok(forgetting_metrics)
    }

    /// Get memory buffer statistics
    #[must_use]
    pub fn memory_statistics(&self) -> HashMap<String, f64> {
        self.state.memory_buffer.statistics()
    }

    /// Get learned tasks
    #[must_use]
    pub fn learned_tasks(&self) -> &[String] {
        &self.state.learned_tasks
    }

    /// Get task performance
    #[must_use]
    pub fn task_performance(&self) -> &HashMap<String, HashMap<String, f64>> {
        &self.state.task_performance
    }

    /// Get importance weights
    #[must_use]
    pub fn importance_weights(&self) -> &HashMap<String, f64> {
        &self.state.importance_weights
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::MockPredictor;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_task_creation() {
        let features = array![[1.0, 2.0], [3.0, 4.0]];
        let targets = array![1.0, 0.0];

        let mut task = Task::new("task1".to_string(), features, targets);
        task.estimate_difficulty();

        assert_eq!(task.id, "task1");
        assert_eq!(task.statistics.n_samples, 2);
        assert_eq!(task.statistics.n_features, 2);
        assert!(task.statistics.difficulty > 0.0);
    }

    #[test]
    fn test_memory_buffer() {
        let mut buffer = MemoryBuffer::new(3, SamplingStrategy::Random);

        let sample1 = MemorySample {
            features: array![1.0, 2.0],
            target: 1.0,
            task_id: "task1".to_string(),
            importance: 1.0,
            gradient_info: None,
        };

        buffer.add_sample(sample1);
        assert_eq!(buffer.samples.len(), 1);

        let sampled = buffer.sample(1);
        assert_eq!(sampled.len(), 1);
    }

    #[test]
    fn test_continual_learning_pipeline() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 0.0];

        let base_estimator = Box::new(MockPredictor::new());
        let pipeline = ContinualLearningPipeline::experience_replay(base_estimator, 100, 10, 5)
            .set_current_task("task1".to_string());

        let fitted_pipeline = pipeline.fit(&x.view(), &Some(&y.view())).unwrap();
        let predictions = fitted_pipeline.predict(&x.view()).unwrap();

        assert_eq!(predictions.len(), x.nrows());
        assert!(fitted_pipeline
            .learned_tasks()
            .contains(&"task1".to_string()));
    }

    #[test]
    fn test_ewc_pipeline() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 0.0];

        let base_estimator = Box::new(MockPredictor::new());
        let pipeline =
            ContinualLearningPipeline::elastic_weight_consolidation(base_estimator, 0.1, 10);

        let fitted_pipeline = pipeline.fit(&x.view(), &Some(&y.view())).unwrap();

        assert!(!fitted_pipeline.importance_weights().is_empty());

        let predictions = fitted_pipeline.predict(&x.view()).unwrap();
        assert_eq!(predictions.len(), x.nrows());
    }

    #[test]
    fn test_new_task_learning() {
        let x1 = array![[1.0, 2.0], [3.0, 4.0]];
        let y1 = array![1.0, 0.0];

        let base_estimator = Box::new(MockPredictor::new());
        let pipeline = ContinualLearningPipeline::experience_replay(base_estimator, 100, 10, 5);

        let mut fitted_pipeline = pipeline.fit(&x1.view(), &Some(&y1.view())).unwrap();

        // Learn new task
        let x2 = array![[5.0, 6.0], [7.0, 8.0]];
        let y2 = array![0.0, 1.0];
        let task2 = Task::new("task2".to_string(), x2, y2);

        fitted_pipeline.learn_task(task2).unwrap();

        assert_eq!(fitted_pipeline.learned_tasks().len(), 2);
        assert!(fitted_pipeline
            .learned_tasks()
            .contains(&"task2".to_string()));
    }
}
