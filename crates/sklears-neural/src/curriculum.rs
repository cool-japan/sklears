//! Curriculum learning implementation for neural networks.
//!
//! Curriculum learning is a training strategy where models are trained on easier
//! examples first, then gradually introduced to more difficult examples. This can
//! lead to better convergence and improved generalization performance.

use crate::NeuralResult;
use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use scirs2_core::random::thread_rng;
use scirs2_core::SliceRandomExt;
use sklears_core::error::SklearsError;

/// Strategies for determining the difficulty of training examples
#[derive(Debug, Clone, PartialEq)]
pub enum DifficultyStrategy {
    /// Sort by prediction confidence (low confidence = high difficulty)
    PredictionConfidence,
    /// Sort by loss value (high loss = high difficulty)
    LossValue,
    /// Sort by gradient magnitude (high gradient = high difficulty)
    GradientMagnitude,
    /// Custom difficulty scores provided by user
    Custom(Array1<f64>),
    /// Random ordering (baseline for comparison)
    Random,
    /// Self-paced learning based on adaptive threshold
    SelfPaced {
        initial_threshold: f64,
        growth_rate: f64,
        max_threshold: f64,
    },
}

/// Pacing strategies for curriculum learning
#[derive(Debug, Clone, PartialEq)]
pub enum PacingStrategy {
    /// Linear increase in the number of examples
    Linear {
        initial_size: usize,
        growth_rate: usize,
    },
    /// Exponential increase in the number of examples
    Exponential {
        initial_size: usize,
        growth_factor: f64,
    },
    /// Step-wise increase at fixed epochs
    Stepwise {
        step_size: usize,
        step_epochs: Vec<usize>,
    },
    /// Polynomial pacing
    Polynomial { initial_size: usize, power: f64 },
    /// Sigmoid-based pacing for smooth transitions
    Sigmoid {
        max_size: usize,
        steepness: f64,
        midpoint: f64,
    },
    /// Custom pacing function
    Custom(Vec<usize>),
}

/// Configuration for curriculum learning
#[derive(Debug, Clone)]
pub struct CurriculumConfig {
    /// Strategy for determining example difficulty
    pub difficulty_strategy: DifficultyStrategy,
    /// Strategy for pacing the curriculum
    pub pacing_strategy: PacingStrategy,
    /// How often to re-evaluate difficulty scores (in epochs)
    pub update_frequency: usize,
    /// Whether to shuffle examples within each difficulty bucket
    pub shuffle_within_bucket: bool,
    /// Minimum number of examples per batch
    pub min_batch_size: usize,
    /// Maximum number of examples to use
    pub max_examples: Option<usize>,
    /// Whether to use adaptive thresholding for self-paced learning
    pub adaptive_threshold: bool,
}

impl Default for CurriculumConfig {
    fn default() -> Self {
        Self {
            difficulty_strategy: DifficultyStrategy::PredictionConfidence,
            pacing_strategy: PacingStrategy::Linear {
                initial_size: 100,
                growth_rate: 50,
            },
            update_frequency: 5,
            shuffle_within_bucket: true,
            min_batch_size: 32,
            max_examples: None,
            adaptive_threshold: false,
        }
    }
}

/// Curriculum learning scheduler
pub struct CurriculumScheduler {
    config: CurriculumConfig,
    current_epoch: usize,
    difficulty_scores: Option<Array1<f64>>,
    sorted_indices: Vec<usize>,
    current_threshold: f64,
    training_history: Vec<f64>,
}

impl CurriculumScheduler {
    /// Create a new curriculum scheduler
    pub fn new(config: CurriculumConfig) -> Self {
        let current_threshold = match &config.difficulty_strategy {
            DifficultyStrategy::SelfPaced {
                initial_threshold, ..
            } => *initial_threshold,
            _ => 0.0,
        };

        Self {
            config,
            current_epoch: 0,
            difficulty_scores: None,
            sorted_indices: Vec::new(),
            current_threshold,
            training_history: Vec::new(),
        }
    }

    /// Update difficulty scores based on model predictions or losses
    pub fn update_difficulty_scores<M>(
        &mut self,
        model: &M,
        inputs: &Array2<f64>,
        targets: &Array1<usize>,
        losses: Option<&Array1<f64>>,
    ) -> NeuralResult<()>
    where
        M: crate::interpretation::InterpretableModel,
    {
        let n_samples = inputs.nrows();

        let scores = match &self.config.difficulty_strategy {
            DifficultyStrategy::PredictionConfidence => {
                let (predictions, _) = model.forward_with_activations(inputs)?;
                self.compute_confidence_scores(&predictions, targets)?
            }
            DifficultyStrategy::LossValue => losses
                .ok_or_else(|| SklearsError::InvalidParameter {
                    name: "losses".to_string(),
                    reason: "Loss values required for LossValue difficulty strategy".to_string(),
                })?
                .to_owned(),
            DifficultyStrategy::GradientMagnitude => {
                let mut total_gradient_magnitude = Array1::zeros(n_samples);
                for i in 0..n_samples {
                    let sample = inputs.slice(s![i..i + 1, ..]).to_owned();
                    let gradients = model.compute_gradients(&sample, Some(targets[i]))?;
                    total_gradient_magnitude[i] = gradients.iter().map(|&x| x.abs()).sum();
                }
                total_gradient_magnitude
            }
            DifficultyStrategy::Custom(scores) => {
                if scores.len() != n_samples {
                    return Err(SklearsError::ShapeMismatch {
                        expected: format!("n_samples={}", n_samples),
                        actual: format!("custom_scores.len()={}", scores.len()),
                    });
                }
                scores.clone()
            }
            DifficultyStrategy::Random => {
                use scirs2_core::random::prelude::*;
                let mut rng = thread_rng();
                Array1::from_shape_fn(n_samples, |_| rng.gen::<f64>())
            }
            DifficultyStrategy::SelfPaced { .. } => losses
                .ok_or_else(|| SklearsError::InvalidParameter {
                    name: "losses".to_string(),
                    reason: "Loss values required for SelfPaced difficulty strategy".to_string(),
                })?
                .to_owned(),
        };

        // Sort indices by difficulty (easy to hard)
        let mut indexed_scores: Vec<(usize, f64)> = scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();

        indexed_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        self.sorted_indices = indexed_scores.into_iter().map(|(i, _)| i).collect();
        self.difficulty_scores = Some(scores);

        Ok(())
    }

    /// Get the current subset of training examples based on curriculum pacing
    pub fn get_current_examples(&mut self) -> NeuralResult<Vec<usize>> {
        let total_examples = self.sorted_indices.len();
        if total_examples == 0 {
            return Err(SklearsError::InvalidParameter {
                name: "sorted_indices".to_string(),
                reason: "No examples available. Call update_difficulty_scores first.".to_string(),
            });
        }

        let max_examples = self.config.max_examples.unwrap_or(total_examples);
        let available_examples = total_examples.min(max_examples);

        let current_size = match &self.config.pacing_strategy {
            PacingStrategy::Linear {
                initial_size,
                growth_rate,
            } => {
                let size = initial_size + growth_rate * self.current_epoch;
                size.min(available_examples).max(self.config.min_batch_size)
            }
            PacingStrategy::Exponential {
                initial_size,
                growth_factor,
            } => {
                let size =
                    (*initial_size as f64 * growth_factor.powf(self.current_epoch as f64)) as usize;
                size.min(available_examples).max(self.config.min_batch_size)
            }
            PacingStrategy::Stepwise {
                step_size,
                step_epochs,
            } => {
                let steps_completed = step_epochs
                    .iter()
                    .filter(|&&epoch| epoch <= self.current_epoch)
                    .count();
                let size = self.config.min_batch_size + step_size * steps_completed;
                size.min(available_examples)
            }
            PacingStrategy::Polynomial {
                initial_size,
                power,
            } => {
                let progress = (self.current_epoch as f64 + 1.0) / 100.0; // Normalize to [0, 1]
                let size = *initial_size
                    + ((available_examples - initial_size) as f64 * progress.powf(*power)) as usize;
                size.min(available_examples).max(self.config.min_batch_size)
            }
            PacingStrategy::Sigmoid {
                max_size,
                steepness,
                midpoint,
            } => {
                let x = self.current_epoch as f64;
                let sigmoid = 1.0 / (1.0 + (-steepness * (x - midpoint)).exp());
                let size = self.config.min_batch_size
                    + ((max_size - self.config.min_batch_size) as f64 * sigmoid) as usize;
                size.min(available_examples)
            }
            PacingStrategy::Custom(sizes) => {
                if self.current_epoch < sizes.len() {
                    sizes[self.current_epoch]
                        .min(available_examples)
                        .max(self.config.min_batch_size)
                } else {
                    available_examples
                }
            }
        };

        // For self-paced learning, filter by threshold
        let mut selected_indices: Vec<usize> =
            if let DifficultyStrategy::SelfPaced { .. } = &self.config.difficulty_strategy {
                if let Some(ref scores) = self.difficulty_scores {
                    self.sorted_indices
                        .iter()
                        .filter(|&&idx| scores[idx] <= self.current_threshold)
                        .take(current_size)
                        .cloned()
                        .collect()
                } else {
                    self.sorted_indices
                        .iter()
                        .take(current_size)
                        .cloned()
                        .collect()
                }
            } else {
                self.sorted_indices
                    .iter()
                    .take(current_size)
                    .cloned()
                    .collect()
            };

        // Shuffle within bucket if requested
        if self.config.shuffle_within_bucket {
            use scirs2_core::random::prelude::*;
            selected_indices.shuffle(&mut thread_rng());
        }

        Ok(selected_indices)
    }

    /// Update the curriculum for the next epoch
    pub fn step_epoch(&mut self, average_loss: Option<f64>) {
        self.current_epoch += 1;

        if let Some(loss) = average_loss {
            self.training_history.push(loss);
        }

        // Update threshold for self-paced learning
        if let DifficultyStrategy::SelfPaced {
            growth_rate,
            max_threshold,
            ..
        } = &self.config.difficulty_strategy
        {
            if self.config.adaptive_threshold && !self.training_history.is_empty() {
                // Adaptive threshold based on training progress
                let recent_losses =
                    &self.training_history[self.training_history.len().saturating_sub(5)..];
                let avg_recent_loss =
                    recent_losses.iter().sum::<f64>() / recent_losses.len() as f64;

                if self.training_history.len() > 1 {
                    let prev_loss = self.training_history[self.training_history.len() - 2];
                    if avg_recent_loss < prev_loss * 0.95 {
                        // Good progress
                        self.current_threshold += growth_rate;
                    }
                }
            } else {
                // Fixed growth rate
                self.current_threshold += growth_rate;
            }

            self.current_threshold = self.current_threshold.min(*max_threshold);
        }
    }

    /// Check if difficulty scores should be updated
    pub fn should_update_difficulty(&self) -> bool {
        self.current_epoch % self.config.update_frequency == 0
    }

    /// Get curriculum statistics
    pub fn get_statistics(&mut self) -> CurriculumStatistics {
        let current_size = if let Ok(indices) = self.get_current_examples() {
            indices.len()
        } else {
            0
        };

        CurriculumStatistics {
            current_epoch: self.current_epoch,
            current_subset_size: current_size,
            total_examples: self.sorted_indices.len(),
            current_threshold: self.current_threshold,
            average_difficulty: self
                .difficulty_scores
                .as_ref()
                .map(|scores| scores.mean().unwrap_or(0.0)),
            training_loss_trend: if self.training_history.len() >= 2 {
                let recent = self.training_history[self.training_history.len() - 1];
                let previous = self.training_history[self.training_history.len() - 2];
                Some(recent - previous)
            } else {
                None
            },
        }
    }

    /// Reset the scheduler
    pub fn reset(&mut self) {
        self.current_epoch = 0;
        self.difficulty_scores = None;
        self.sorted_indices.clear();
        self.training_history.clear();
        self.current_threshold = match &self.config.difficulty_strategy {
            DifficultyStrategy::SelfPaced {
                initial_threshold, ..
            } => *initial_threshold,
            _ => 0.0,
        };
    }

    /// Compute confidence scores from predictions
    fn compute_confidence_scores(
        &self,
        predictions: &Array2<f64>,
        targets: &Array1<usize>,
    ) -> NeuralResult<Array1<f64>> {
        let mut confidence_scores = Array1::zeros(predictions.nrows());

        for (i, (&target, pred_row)) in targets
            .iter()
            .zip(predictions.axis_iter(Axis(0)))
            .enumerate()
        {
            if target >= pred_row.len() {
                return Err(SklearsError::InvalidParameter {
                    name: "target".to_string(),
                    reason: format!(
                        "Target class {} is out of bounds for {} classes",
                        target,
                        pred_row.len()
                    ),
                });
            }

            // Confidence is the predicted probability for the true class
            // Lower confidence = higher difficulty
            confidence_scores[i] = pred_row[target];
        }

        // Invert scores so that low confidence = high difficulty score
        Ok(confidence_scores.mapv(|x| 1.0 - x))
    }
}

/// Statistics about the current curriculum state
#[derive(Debug, Clone)]
pub struct CurriculumStatistics {
    pub current_epoch: usize,
    pub current_subset_size: usize,
    pub total_examples: usize,
    pub current_threshold: f64,
    pub average_difficulty: Option<f64>,
    pub training_loss_trend: Option<f64>,
}

impl CurriculumStatistics {
    /// Get the percentage of examples currently being used
    pub fn usage_percentage(&self) -> f64 {
        if self.total_examples == 0 {
            0.0
        } else {
            (self.current_subset_size as f64 / self.total_examples as f64) * 100.0
        }
    }

    /// Check if curriculum is complete (using all examples)
    pub fn is_complete(&self) -> bool {
        self.current_subset_size >= self.total_examples
    }
}

/// Anti-curriculum learning (hard examples first)
pub struct AntiCurriculumScheduler {
    base_scheduler: CurriculumScheduler,
}

impl AntiCurriculumScheduler {
    /// Create a new anti-curriculum scheduler
    pub fn new(config: CurriculumConfig) -> Self {
        Self {
            base_scheduler: CurriculumScheduler::new(config),
        }
    }

    /// Update difficulty scores (same as regular curriculum)
    pub fn update_difficulty_scores<M>(
        &mut self,
        model: &M,
        inputs: &Array2<f64>,
        targets: &Array1<usize>,
        losses: Option<&Array1<f64>>,
    ) -> NeuralResult<()>
    where
        M: crate::interpretation::InterpretableModel,
    {
        self.base_scheduler
            .update_difficulty_scores(model, inputs, targets, losses)?;
        // Reverse the sorted indices to start with hardest examples
        self.base_scheduler.sorted_indices.reverse();
        Ok(())
    }

    /// Get current examples (delegates to base scheduler)
    pub fn get_current_examples(&mut self) -> NeuralResult<Vec<usize>> {
        self.base_scheduler.get_current_examples()
    }

    /// Step to next epoch
    pub fn step_epoch(&mut self, average_loss: Option<f64>) {
        self.base_scheduler.step_epoch(average_loss);
    }

    /// Check if difficulty scores should be updated
    pub fn should_update_difficulty(&self) -> bool {
        self.base_scheduler.should_update_difficulty()
    }

    /// Get statistics
    pub fn get_statistics(&mut self) -> CurriculumStatistics {
        self.base_scheduler.get_statistics()
    }

    /// Reset the scheduler
    pub fn reset(&mut self) {
        self.base_scheduler.reset();
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    struct MockModel;

    impl crate::interpretation::InterpretableModel for MockModel {
        fn compute_gradients(
            &self,
            input: &Array2<f64>,
            _target_class: Option<usize>,
        ) -> NeuralResult<Array2<f64>> {
            Ok(input.mapv(|x| x * 0.1))
        }

        fn forward_with_activations(
            &self,
            input: &Array2<f64>,
        ) -> NeuralResult<(Array2<f64>, Vec<Array2<f64>>)> {
            // Mock softmax predictions
            let predictions = Array2::from_shape_fn((input.nrows(), 3), |(i, j)| match j {
                0 => 0.7 - (i as f64 * 0.1),
                1 => 0.2 + (i as f64 * 0.05),
                2 => 0.1 + (i as f64 * 0.05),
                _ => 0.0,
            });
            Ok((predictions, vec![input.clone()]))
        }

        fn num_classes(&self) -> usize {
            3
        }
    }

    #[test]
    fn test_curriculum_scheduler_creation() {
        let config = CurriculumConfig::default();
        let scheduler = CurriculumScheduler::new(config);

        assert_eq!(scheduler.current_epoch, 0);
        assert!(scheduler.difficulty_scores.is_none());
        assert!(scheduler.sorted_indices.is_empty());
    }

    #[test]
    fn test_difficulty_score_update() {
        let config = CurriculumConfig {
            difficulty_strategy: DifficultyStrategy::PredictionConfidence,
            ..Default::default()
        };
        let mut scheduler = CurriculumScheduler::new(config);
        let model = MockModel;

        let inputs =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let targets = Array1::from_vec(vec![0, 0, 1, 2]);

        scheduler
            .update_difficulty_scores(&model, &inputs, &targets, None)
            .unwrap();

        assert!(scheduler.difficulty_scores.is_some());
        assert_eq!(scheduler.sorted_indices.len(), 4);
    }

    #[test]
    fn test_linear_pacing() {
        let config = CurriculumConfig {
            pacing_strategy: PacingStrategy::Linear {
                initial_size: 2,
                growth_rate: 1,
            },
            min_batch_size: 1,
            ..Default::default()
        };
        let mut scheduler = CurriculumScheduler::new(config);

        // Mock some sorted indices
        scheduler.sorted_indices = vec![0, 1, 2, 3, 4];

        // Epoch 0: should get 2 examples
        let indices = scheduler.get_current_examples().unwrap();
        assert_eq!(indices.len(), 2);

        scheduler.step_epoch(None);

        // Epoch 1: should get 3 examples
        let indices = scheduler.get_current_examples().unwrap();
        assert_eq!(indices.len(), 3);
    }

    #[test]
    fn test_exponential_pacing() {
        let config = CurriculumConfig {
            pacing_strategy: PacingStrategy::Exponential {
                initial_size: 2,
                growth_factor: 1.5,
            },
            min_batch_size: 1,
            ..Default::default()
        };
        let mut scheduler = CurriculumScheduler::new(config);

        scheduler.sorted_indices = vec![0, 1, 2, 3, 4, 5, 6, 7];

        // Epoch 0: should get 2 examples
        let indices = scheduler.get_current_examples().unwrap();
        assert_eq!(indices.len(), 2);

        scheduler.step_epoch(None);

        // Epoch 1: should get floor(2 * 1.5) = 3 examples
        let indices = scheduler.get_current_examples().unwrap();
        assert_eq!(indices.len(), 3);
    }

    #[test]
    fn test_curriculum_statistics() {
        let config = CurriculumConfig::default();
        let mut scheduler = CurriculumScheduler::new(config);

        scheduler.sorted_indices = vec![0, 1, 2, 3, 4];
        scheduler.training_history = vec![1.0, 0.8, 0.6];

        let stats = scheduler.get_statistics();
        assert_eq!(stats.total_examples, 5);
        assert_eq!(stats.current_epoch, 0);
        assert!(stats.training_loss_trend.is_some());
        assert_abs_diff_eq!(stats.training_loss_trend.unwrap(), -0.2, epsilon = 1e-10);
    }

    #[test]
    fn test_anti_curriculum() {
        let config = CurriculumConfig {
            difficulty_strategy: DifficultyStrategy::PredictionConfidence,
            ..Default::default()
        };
        let mut anti_scheduler = AntiCurriculumScheduler::new(config);
        let model = MockModel;

        let inputs =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let targets = Array1::from_vec(vec![0, 0, 1, 2]);

        anti_scheduler
            .update_difficulty_scores(&model, &inputs, &targets, None)
            .unwrap();

        // Anti-curriculum should start with the hardest examples
        // (Note: exact testing would require knowing the mock model's output pattern)
        assert_eq!(anti_scheduler.base_scheduler.sorted_indices.len(), 4);
    }

    #[test]
    fn test_self_paced_learning() {
        let config = CurriculumConfig {
            difficulty_strategy: DifficultyStrategy::SelfPaced {
                initial_threshold: 0.5,
                growth_rate: 0.1,
                max_threshold: 2.0,
            },
            ..Default::default()
        };
        let mut scheduler = CurriculumScheduler::new(config);

        assert_abs_diff_eq!(scheduler.current_threshold, 0.5, epsilon = 1e-10);

        scheduler.step_epoch(Some(1.0));
        assert_abs_diff_eq!(scheduler.current_threshold, 0.6, epsilon = 1e-10);

        // Test max threshold
        for _ in 0..20 {
            scheduler.step_epoch(Some(0.5));
        }
        assert_abs_diff_eq!(scheduler.current_threshold, 2.0, epsilon = 1e-10);
    }
}
