//! Knowledge Distillation implementation for model compression
//!
//! This module provides comprehensive knowledge distillation techniques that allow
//! transferring knowledge from large, complex models (teachers) to smaller, more
//! efficient models (students). This is crucial for model deployment in resource-
//! constrained environments while maintaining performance.
//!
//! # Theory
//!
//! Knowledge distillation works by training a student model to mimic the behavior
//! of a teacher model. The key insight is that the soft predictions (output probabilities)
//! of the teacher contain more information than hard labels.
//!
//! The distillation loss combines:
//! - Distillation loss: KL divergence between teacher and student soft predictions
//! - Student loss: Standard loss on ground truth labels
//! - Feature matching loss: Match intermediate representations
//!
//! Total loss = α * distillation_loss + β * student_loss + γ * feature_loss
//!
//! # Example
//!
//! ```rust,ignore
//! use sklears_neural::knowledge_distillation::{KnowledgeDistiller, DistillationConfig};
//!
//! let config = DistillationConfig::default()
//!     .temperature(4.0)
//!     .alpha(0.7)
//!     .beta(0.3);
//!
//! let distiller = KnowledgeDistiller::new(config);
//! ```

// Note: Simplified implementation using basic structures
use crate::{NeuralResult, SklearsError};
use scirs2_core::ndarray::{s, Array1, Array2, Axis, ScalarOperand};
use sklears_core::types::{Float, FloatBounds};
use std::collections::HashMap;
use std::iter::Sum;
use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Knowledge distillation strategy
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DistillationStrategy {
    /// Standard soft target distillation
    SoftTargets,
    /// Feature-based distillation matching intermediate representations
    FeatureMatching,
    /// Attention-based distillation
    AttentionTransfer,
    /// Progressive knowledge distillation
    Progressive,
    /// Multi-task distillation
    MultiTask,
}

impl Default for DistillationStrategy {
    fn default() -> Self {
        DistillationStrategy::SoftTargets
    }
}

/// Temperature scaling method for soft targets
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum TemperatureSchedule {
    /// Fixed temperature throughout training
    Fixed(Float),
    /// Linearly decreasing temperature
    Linear { start: Float, end: Float },
    /// Exponentially decreasing temperature
    Exponential { start: Float, decay: Float },
    /// Adaptive temperature based on training progress
    Adaptive,
}

impl Default for TemperatureSchedule {
    fn default() -> Self {
        TemperatureSchedule::Fixed(3.0)
    }
}

/// Configuration for knowledge distillation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DistillationConfig {
    /// Distillation strategy
    pub strategy: DistillationStrategy,
    /// Temperature for softmax in distillation
    pub temperature_schedule: TemperatureSchedule,
    /// Weight for distillation loss (α)
    pub alpha: Float,
    /// Weight for student loss (β)
    pub beta: Float,
    /// Weight for feature matching loss (γ)
    pub gamma: Float,
    /// Learning rate for student model
    pub learning_rate: Float,
    /// Number of training epochs
    pub n_epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Layers to use for feature matching (if applicable)
    pub feature_layers: Vec<String>,
    /// Whether to use progressive distillation
    pub progressive: bool,
    /// Number of progressive stages
    pub progressive_stages: usize,
    /// Early stopping patience
    pub patience: usize,
    /// Random seed
    pub random_state: Option<u64>,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            strategy: DistillationStrategy::SoftTargets,
            temperature_schedule: TemperatureSchedule::Fixed(3.0),
            alpha: 0.7,
            beta: 0.3,
            gamma: 0.1,
            learning_rate: 0.001,
            n_epochs: 100,
            batch_size: 32,
            feature_layers: vec!["layer_3".to_string(), "layer_6".to_string()],
            progressive: false,
            progressive_stages: 3,
            patience: 10,
            random_state: None,
        }
    }
}

impl DistillationConfig {
    /// Set distillation strategy
    pub fn strategy(mut self, strategy: DistillationStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set fixed temperature
    pub fn temperature(mut self, temp: Float) -> Self {
        self.temperature_schedule = TemperatureSchedule::Fixed(temp);
        self
    }

    /// Set temperature schedule
    pub fn temperature_schedule(mut self, schedule: TemperatureSchedule) -> Self {
        self.temperature_schedule = schedule;
        self
    }

    /// Set distillation weight (α)
    pub fn alpha(mut self, alpha: Float) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set student loss weight (β)
    pub fn beta(mut self, beta: Float) -> Self {
        self.beta = beta;
        self
    }

    /// Set feature matching weight (γ)
    pub fn gamma(mut self, gamma: Float) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, lr: Float) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Enable progressive distillation
    pub fn progressive(mut self, enabled: bool, stages: usize) -> Self {
        self.progressive = enabled;
        self.progressive_stages = stages;
        self
    }

    /// Set feature layers for matching
    pub fn feature_layers(mut self, layers: Vec<String>) -> Self {
        self.feature_layers = layers;
        self
    }
}

/// Teacher model interface - simplified version
pub trait TeacherModel<T: FloatBounds> {
    /// Forward pass returning outputs and intermediate features
    fn forward_with_features(
        &mut self,
        input: &Array2<T>,
    ) -> NeuralResult<(Array2<T>, HashMap<String, Array2<T>>)>;

    /// Get soft predictions with temperature scaling
    fn soft_predictions(&mut self, input: &Array2<T>, temperature: T) -> NeuralResult<Array2<T>>;

    /// Get attention maps (if applicable)
    fn get_attention_maps(&mut self, input: &Array2<T>)
        -> NeuralResult<HashMap<String, Array2<T>>>;
}

/// Student model interface - simplified version
pub trait StudentModel<T: FloatBounds> {
    /// Forward pass returning outputs and intermediate features
    fn forward_with_features(
        &mut self,
        input: &Array2<T>,
    ) -> NeuralResult<(Array2<T>, HashMap<String, Array2<T>>)>;

    /// Backward pass for training
    fn backward(&mut self, grad_output: &Array2<T>) -> NeuralResult<()>;

    /// Update model parameters (simplified - no optimizer dependency)
    fn update_parameters(&mut self, learning_rate: T) -> NeuralResult<()>;

    /// Get attention maps (if applicable)
    fn get_attention_maps(&mut self, input: &Array2<T>)
        -> NeuralResult<HashMap<String, Array2<T>>>;

    /// Get number of parameters
    fn parameter_count(&self) -> usize;
}

/// Knowledge distillation trainer
pub struct KnowledgeDistiller<T: FloatBounds> {
    config: DistillationConfig,
    current_temperature: T,
    current_epoch: usize,
    best_loss: Option<T>,
    patience_counter: usize,
    _phantom: PhantomData<T>,
}

impl<T: FloatBounds + ScalarOperand + Sum> KnowledgeDistiller<T> {
    /// Create new knowledge distiller
    pub fn new(config: DistillationConfig) -> Self {
        let initial_temperature = match config.temperature_schedule {
            TemperatureSchedule::Fixed(temp) => T::from(temp).unwrap(),
            TemperatureSchedule::Linear { start, .. } => T::from(start).unwrap(),
            TemperatureSchedule::Exponential { start, .. } => T::from(start).unwrap(),
            TemperatureSchedule::Adaptive => T::from(3.0).unwrap(),
        };

        Self {
            config,
            current_temperature: initial_temperature,
            current_epoch: 0,
            best_loss: None,
            patience_counter: 0,
            _phantom: PhantomData,
        }
    }

    /// Distill knowledge from teacher to student
    pub fn distill<Teacher, Student>(
        &mut self,
        teacher: &mut Teacher,
        student: &mut Student,
        train_data: &Array2<T>,
        train_labels: &Array2<T>,
        val_data: Option<&Array2<T>>,
        val_labels: Option<&Array2<T>>,
    ) -> NeuralResult<DistillationHistory>
    where
        Teacher: TeacherModel<T>,
        Student: StudentModel<T>,
    {
        let mut history = DistillationHistory::new();
        let batch_size = self.config.batch_size;
        let n_batches = train_data.nrows().div_ceil(batch_size);

        for epoch in 0..self.config.n_epochs {
            self.current_epoch = epoch;
            self.update_temperature(epoch);

            let mut epoch_losses = DistillationLoss::new();

            // Training loop
            for batch_idx in 0..n_batches {
                let start_idx = batch_idx * batch_size;
                let end_idx = (start_idx + batch_size).min(train_data.nrows());

                let batch_data = train_data.slice(s![start_idx..end_idx, ..]).to_owned();
                let batch_labels = train_labels.slice(s![start_idx..end_idx, ..]).to_owned();

                let batch_loss = self.train_batch(teacher, student, &batch_data, &batch_labels)?;
                epoch_losses = epoch_losses + batch_loss;
            }

            // Average the losses
            let avg_loss = epoch_losses / n_batches as f64;
            history.train_losses.push(avg_loss.clone());

            // Validation
            if let (Some(val_data), Some(val_labels)) = (val_data, val_labels) {
                let val_loss = self.evaluate(teacher, student, val_data, val_labels)?;
                history.val_losses.push(val_loss.clone());

                // Early stopping
                let total_val_loss = val_loss.total();
                if self.should_stop_early(total_val_loss) {
                    println!(
                        "Early stopping at epoch {} with validation loss {:.6}",
                        epoch, total_val_loss
                    );
                    break;
                }
            }

            // Progressive distillation stage switching
            if self.config.progressive && self.should_switch_stage(epoch) {
                self.switch_progressive_stage(student)?;
            }

            history.epochs_completed = epoch + 1;

            // Print progress
            if epoch % 10 == 0 {
                println!(
                    "Epoch {}: Loss = {:.6}, Temp = {:.3}",
                    epoch,
                    avg_loss.total(),
                    self.current_temperature.to_f64().unwrap_or(0.0)
                );
            }
        }

        Ok(history)
    }

    /// Train a single batch
    fn train_batch<Teacher, Student>(
        &mut self,
        teacher: &mut Teacher,
        student: &mut Student,
        batch_data: &Array2<T>,
        batch_labels: &Array2<T>,
    ) -> NeuralResult<DistillationLoss>
    where
        Teacher: TeacherModel<T>,
        Student: StudentModel<T>,
    {
        // Get teacher predictions and features
        let teacher_soft = teacher.soft_predictions(batch_data, self.current_temperature)?;
        let (teacher_outputs, teacher_features) = teacher.forward_with_features(batch_data)?;

        // Get student predictions and features
        let (student_outputs, student_features) = student.forward_with_features(batch_data)?;

        // Compute losses based on strategy
        let loss = match self.config.strategy {
            DistillationStrategy::SoftTargets => {
                let distill_loss = self.kl_divergence_loss(&student_outputs, &teacher_soft)?;
                let student_loss = self.cross_entropy_loss(&student_outputs, batch_labels)?;

                DistillationLoss {
                    distillation: distill_loss,
                    student: student_loss,
                    feature_matching: 0.0,
                    attention: 0.0,
                }
            }

            DistillationStrategy::FeatureMatching => {
                let distill_loss = self.kl_divergence_loss(&student_outputs, &teacher_soft)?;
                let student_loss = self.cross_entropy_loss(&student_outputs, batch_labels)?;
                let feature_loss =
                    self.feature_matching_loss(&student_features, &teacher_features)?;

                DistillationLoss {
                    distillation: distill_loss,
                    student: student_loss,
                    feature_matching: feature_loss,
                    attention: 0.0,
                }
            }

            DistillationStrategy::AttentionTransfer => {
                let distill_loss = self.kl_divergence_loss(&student_outputs, &teacher_soft)?;
                let student_loss = self.cross_entropy_loss(&student_outputs, batch_labels)?;

                let teacher_attention = teacher.get_attention_maps(batch_data)?;
                let student_attention = student.get_attention_maps(batch_data)?;
                let attention_loss =
                    self.attention_transfer_loss(&student_attention, &teacher_attention)?;

                DistillationLoss {
                    distillation: distill_loss,
                    student: student_loss,
                    feature_matching: 0.0,
                    attention: attention_loss,
                }
            }

            _ => {
                return Err(SklearsError::InvalidParameter {
                    name: "strategy".to_string(),
                    reason: "Strategy not yet implemented".to_string(),
                });
            }
        };

        // Compute total loss and backward pass
        let total_loss = self.config.alpha * loss.distillation
            + self.config.beta * loss.student
            + self.config.gamma * loss.feature_matching
            + 0.1 * loss.attention; // Fixed weight for attention

        // Convert total loss to gradients (would need automatic differentiation)
        let grad_output = Array2::from_elem(student_outputs.dim(), T::from(total_loss).unwrap());
        student.backward(&grad_output)?;
        student.update_parameters(T::from(self.config.learning_rate).unwrap())?;

        Ok(loss)
    }

    /// Evaluate on validation data
    fn evaluate<Teacher, Student>(
        &mut self,
        teacher: &mut Teacher,
        student: &mut Student,
        val_data: &Array2<T>,
        val_labels: &Array2<T>,
    ) -> NeuralResult<DistillationLoss>
    where
        Teacher: TeacherModel<T>,
        Student: StudentModel<T>,
    {
        let batch_size = self.config.batch_size;
        let n_batches = val_data.nrows().div_ceil(batch_size);
        let mut total_loss = DistillationLoss::new();

        for batch_idx in 0..n_batches {
            let start_idx = batch_idx * batch_size;
            let end_idx = (start_idx + batch_size).min(val_data.nrows());

            let batch_data = val_data.slice(s![start_idx..end_idx, ..]).to_owned();
            let batch_labels = val_labels.slice(s![start_idx..end_idx, ..]).to_owned();

            // Forward pass without training
            let teacher_soft = teacher.soft_predictions(&batch_data, self.current_temperature)?;
            let (student_outputs, _) = student.forward_with_features(&batch_data)?;

            let distill_loss = self.kl_divergence_loss(&student_outputs, &teacher_soft)?;
            let student_loss = self.cross_entropy_loss(&student_outputs, &batch_labels)?;

            total_loss.distillation += distill_loss;
            total_loss.student += student_loss;
        }

        Ok(total_loss / n_batches as f64)
    }

    /// Update temperature based on schedule
    fn update_temperature(&mut self, epoch: usize) {
        match self.config.temperature_schedule {
            TemperatureSchedule::Fixed(_) => {
                // Temperature stays the same
            }
            TemperatureSchedule::Linear { start, end } => {
                let progress = epoch as f64 / self.config.n_epochs as f64;
                let temp = start + (end - start) * progress;
                self.current_temperature = T::from(temp).unwrap();
            }
            TemperatureSchedule::Exponential { start, decay } => {
                let temp = start * decay.powf(epoch as f64);
                self.current_temperature = T::from(temp).unwrap();
            }
            TemperatureSchedule::Adaptive => {
                // Adapt temperature based on training progress
                // This would be more sophisticated in practice
                let base_temp = 3.0;
                let temp = base_temp * (1.0 + epoch as f64 / 100.0).recip();
                self.current_temperature = T::from(temp).unwrap();
            }
        }
    }

    /// Check if should stop early
    fn should_stop_early(&mut self, val_loss: f64) -> bool {
        match self.best_loss {
            Some(best) => {
                if val_loss < best.to_f64().unwrap_or(f64::INFINITY) {
                    self.best_loss = Some(T::from(val_loss).unwrap());
                    self.patience_counter = 0;
                    false
                } else {
                    self.patience_counter += 1;
                    self.patience_counter >= self.config.patience
                }
            }
            None => {
                self.best_loss = Some(T::from(val_loss).unwrap());
                self.patience_counter = 0;
                false
            }
        }
    }

    /// Check if should switch progressive stage
    fn should_switch_stage(&self, epoch: usize) -> bool {
        if !self.config.progressive {
            return false;
        }
        let stage_length = self.config.n_epochs / self.config.progressive_stages;
        epoch > 0 && epoch % stage_length == 0
    }

    /// Switch to next progressive stage
    fn switch_progressive_stage<Student>(&mut self, student: &mut Student) -> NeuralResult<()>
    where
        Student: StudentModel<T>,
    {
        // In progressive distillation, we would gradually increase model complexity
        // This is a placeholder for the actual implementation
        println!("Switching to next progressive distillation stage");
        Ok(())
    }

    /// KL divergence loss between soft predictions
    fn kl_divergence_loss(
        &self,
        student_logits: &Array2<T>,
        teacher_soft: &Array2<T>,
    ) -> NeuralResult<f64> {
        let student_soft =
            self.softmax_with_temperature(student_logits, self.current_temperature)?;

        let eps = T::from(1e-8).unwrap();
        let kl_div = teacher_soft
            .iter()
            .zip(student_soft.iter())
            .map(|(&t, &s)| {
                let t_f64 = t.to_f64().unwrap_or(0.0);
                let s_f64 = s.max(eps).to_f64().unwrap_or(1e-8);
                if t_f64 > 0.0 {
                    t_f64 * (t_f64 / s_f64).ln()
                } else {
                    0.0
                }
            })
            .sum::<f64>();

        Ok(kl_div / student_logits.nrows() as f64)
    }

    /// Cross entropy loss with ground truth
    fn cross_entropy_loss(
        &self,
        predictions: &Array2<T>,
        targets: &Array2<T>,
    ) -> NeuralResult<f64> {
        let eps = T::from(1e-8).unwrap();
        let loss = targets
            .iter()
            .zip(predictions.iter())
            .map(|(&target, &pred)| {
                let t_f64 = target.to_f64().unwrap_or(0.0);
                let p_f64 = pred.max(eps).to_f64().unwrap_or(1e-8);
                -t_f64 * p_f64.ln()
            })
            .sum::<f64>();

        Ok(loss / predictions.nrows() as f64)
    }

    /// Feature matching loss between intermediate representations
    fn feature_matching_loss(
        &self,
        student_features: &HashMap<String, Array2<T>>,
        teacher_features: &HashMap<String, Array2<T>>,
    ) -> NeuralResult<f64> {
        let mut total_loss = 0.0;
        let mut count = 0;

        for layer_name in &self.config.feature_layers {
            if let (Some(student_feat), Some(teacher_feat)) = (
                student_features.get(layer_name),
                teacher_features.get(layer_name),
            ) {
                // L2 loss between features
                let diff = student_feat - teacher_feat;
                let layer_loss = diff.mapv(|x| x * x).sum().to_f64().unwrap_or(0.0);
                total_loss += layer_loss;
                count += 1;
            }
        }

        if count > 0 {
            Ok(total_loss / count as f64)
        } else {
            Ok(0.0)
        }
    }

    /// Attention transfer loss
    fn attention_transfer_loss(
        &self,
        student_attention: &HashMap<String, Array2<T>>,
        teacher_attention: &HashMap<String, Array2<T>>,
    ) -> NeuralResult<f64> {
        let mut total_loss = 0.0;
        let mut count = 0;

        for (layer_name, teacher_attn) in teacher_attention {
            if let Some(student_attn) = student_attention.get(layer_name) {
                // Normalize attention maps
                let teacher_norm = self.normalize_attention(teacher_attn)?;
                let student_norm = self.normalize_attention(student_attn)?;

                // L2 loss between normalized attention maps
                let diff = &student_norm - &teacher_norm;
                let layer_loss = diff.mapv(|x| x * x).sum().to_f64().unwrap_or(0.0);
                total_loss += layer_loss;
                count += 1;
            }
        }

        if count > 0 {
            Ok(total_loss / count as f64)
        } else {
            Ok(0.0)
        }
    }

    /// Normalize attention maps
    fn normalize_attention(&self, attention: &Array2<T>) -> NeuralResult<Array2<T>> {
        let sum = attention.sum();
        if sum > T::from(0.0).unwrap() {
            Ok(attention / sum)
        } else {
            Ok(attention.clone())
        }
    }

    /// Apply softmax with temperature
    fn softmax_with_temperature(
        &self,
        logits: &Array2<T>,
        temperature: T,
    ) -> NeuralResult<Array2<T>> {
        let scaled_logits = logits / temperature;
        let mut result = Array2::zeros(logits.dim());

        for (i, mut row) in result.rows_mut().into_iter().enumerate() {
            let logit_row = scaled_logits.row(i);
            let max_logit = logit_row
                .iter()
                .fold(T::from(f64::NEG_INFINITY).unwrap(), |a, &b| a.max(b));

            let exp_sum = logit_row.iter().map(|&x| (x - max_logit).exp()).sum::<T>();

            for (j, elem) in row.iter_mut().enumerate() {
                *elem = (logit_row[j] - max_logit).exp() / exp_sum;
            }
        }

        Ok(result)
    }
}

/// Distillation loss components
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DistillationLoss {
    pub distillation: f64,
    pub student: f64,
    pub feature_matching: f64,
    pub attention: f64,
}

impl DistillationLoss {
    pub fn new() -> Self {
        Self {
            distillation: 0.0,
            student: 0.0,
            feature_matching: 0.0,
            attention: 0.0,
        }
    }

    pub fn total(&self) -> f64 {
        self.distillation + self.student + self.feature_matching + self.attention
    }
}

impl std::ops::Add for DistillationLoss {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            distillation: self.distillation + other.distillation,
            student: self.student + other.student,
            feature_matching: self.feature_matching + other.feature_matching,
            attention: self.attention + other.attention,
        }
    }
}

impl std::ops::Div<f64> for DistillationLoss {
    type Output = Self;

    fn div(self, rhs: f64) -> Self {
        Self {
            distillation: self.distillation / rhs,
            student: self.student / rhs,
            feature_matching: self.feature_matching / rhs,
            attention: self.attention / rhs,
        }
    }
}

impl Default for DistillationLoss {
    fn default() -> Self {
        Self::new()
    }
}

/// Training history for knowledge distillation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DistillationHistory {
    pub train_losses: Vec<DistillationLoss>,
    pub val_losses: Vec<DistillationLoss>,
    pub epochs_completed: usize,
}

impl DistillationHistory {
    pub fn new() -> Self {
        Self {
            train_losses: Vec::new(),
            val_losses: Vec::new(),
            epochs_completed: 0,
        }
    }

    /// Get final training loss
    pub fn final_train_loss(&self) -> Option<&DistillationLoss> {
        self.train_losses.last()
    }

    /// Get final validation loss
    pub fn final_val_loss(&self) -> Option<&DistillationLoss> {
        self.val_losses.last()
    }

    /// Check if training converged
    pub fn has_converged(&self, window_size: usize, threshold: f64) -> bool {
        if self.train_losses.len() < window_size * 2 {
            return false;
        }

        let recent_losses: Vec<f64> = self
            .train_losses
            .iter()
            .rev()
            .take(window_size)
            .map(|loss| loss.total())
            .collect();

        let variance = Self::variance(&recent_losses);
        variance < threshold
    }

    fn variance(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64
    }
}

impl Default for DistillationHistory {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for knowledge distillation
pub mod utils {
    use super::*;

    /// Create configuration for standard knowledge distillation
    pub fn standard_distillation_config() -> DistillationConfig {
        DistillationConfig::default()
            .strategy(DistillationStrategy::SoftTargets)
            .temperature(4.0)
            .alpha(0.7)
            .beta(0.3)
    }

    /// Create configuration for feature-based distillation
    pub fn feature_distillation_config(feature_layers: Vec<String>) -> DistillationConfig {
        DistillationConfig::default()
            .strategy(DistillationStrategy::FeatureMatching)
            .temperature(3.0)
            .alpha(0.5)
            .beta(0.3)
            .gamma(0.2)
            .feature_layers(feature_layers)
    }

    /// Create configuration for attention transfer
    pub fn attention_distillation_config() -> DistillationConfig {
        DistillationConfig::default()
            .strategy(DistillationStrategy::AttentionTransfer)
            .temperature(2.0)
            .alpha(0.6)
            .beta(0.3)
    }

    /// Create configuration for progressive distillation
    pub fn progressive_distillation_config(stages: usize) -> DistillationConfig {
        DistillationConfig::default()
            .strategy(DistillationStrategy::Progressive)
            .progressive(true, stages)
            .temperature_schedule(TemperatureSchedule::Linear {
                start: 5.0,
                end: 2.0,
            })
    }

    /// Evaluate compression ratio
    pub fn compute_compression_ratio<Teacher, Student>(teacher: &Teacher, student: &Student) -> f64
    where
        Teacher: StudentModel<f64>, // Reusing interface for simplicity
        Student: StudentModel<f64>,
    {
        let teacher_params = teacher.parameter_count() as f64;
        let student_params = student.parameter_count() as f64;
        teacher_params / student_params
    }

    /// Evaluate knowledge transfer efficiency
    pub fn compute_transfer_efficiency(
        teacher_accuracy: f64,
        student_accuracy: f64,
        compression_ratio: f64,
    ) -> f64 {
        let accuracy_retention = student_accuracy / teacher_accuracy;
        accuracy_retention * compression_ratio.ln()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx;

    #[test]
    fn test_distillation_config() {
        let config = DistillationConfig::default()
            .strategy(DistillationStrategy::FeatureMatching)
            .temperature(5.0)
            .alpha(0.8)
            .beta(0.2);

        assert_eq!(config.strategy, DistillationStrategy::FeatureMatching);
        matches!(config.temperature_schedule, TemperatureSchedule::Fixed(5.0));
        assert_eq!(config.alpha, 0.8);
        assert_eq!(config.beta, 0.2);
    }

    #[test]
    fn test_distillation_loss() {
        let loss1 = DistillationLoss {
            distillation: 1.0,
            student: 0.5,
            feature_matching: 0.2,
            attention: 0.1,
        };

        let loss2 = DistillationLoss {
            distillation: 0.8,
            student: 0.3,
            feature_matching: 0.1,
            attention: 0.05,
        };

        let total = loss1 + loss2;
        assert_eq!(total.distillation, 1.8);
        assert_eq!(total.student, 0.8);
        assert!((total.total() - 3.05).abs() < 1e-10);

        let avg = total / 2.0;
        assert_eq!(avg.distillation, 0.9);
    }

    #[test]
    fn test_temperature_schedule() {
        let mut distiller = KnowledgeDistiller::<f64>::new(
            DistillationConfig::default().temperature_schedule(TemperatureSchedule::Linear {
                start: 5.0,
                end: 1.0,
            }),
        );

        // Initial temperature
        assert_eq!(distiller.current_temperature, 5.0);

        // Update temperature
        distiller.config.n_epochs = 10;
        distiller.update_temperature(5); // Middle of training
        assert!(distiller.current_temperature < 5.0);
        assert!(distiller.current_temperature > 1.0);
    }

    #[test]
    fn test_distillation_history() {
        let mut history = DistillationHistory::new();

        history.train_losses.push(DistillationLoss {
            distillation: 1.0,
            student: 0.8,
            feature_matching: 0.2,
            attention: 0.0,
        });

        history.epochs_completed = 1;

        assert_eq!(history.final_train_loss().unwrap().total(), 2.0);
        assert!(!history.has_converged(5, 0.1)); // Not enough data for convergence check
    }

    #[test]
    fn test_utility_functions() {
        let config = utils::standard_distillation_config();
        assert_eq!(config.strategy, DistillationStrategy::SoftTargets);
        assert_eq!(config.alpha, 0.7);
        assert_eq!(config.beta, 0.3);

        let feature_config = utils::feature_distillation_config(vec!["layer_1".to_string()]);
        assert_eq!(
            feature_config.strategy,
            DistillationStrategy::FeatureMatching
        );
        assert_eq!(feature_config.gamma, 0.2);

        let compression_ratio = utils::compute_compression_ratio::<MockModel, MockModel>(
            &MockModel { params: 1000 },
            &MockModel { params: 100 },
        );
        assert_eq!(compression_ratio, 10.0);

        let efficiency = utils::compute_transfer_efficiency(0.95, 0.92, 10.0);
        assert!(efficiency > 0.0);
    }

    // Mock model for testing
    struct MockModel {
        params: usize,
    }

    impl StudentModel<f64> for MockModel {
        fn forward_with_features(
            &mut self,
            _input: &Array2<f64>,
        ) -> NeuralResult<(Array2<f64>, HashMap<String, Array2<f64>>)> {
            Ok((Array2::zeros((1, 1)), HashMap::new()))
        }

        fn backward(&mut self, _grad_output: &Array2<f64>) -> NeuralResult<()> {
            Ok(())
        }

        fn update_parameters(&mut self, _learning_rate: f64) -> NeuralResult<()> {
            Ok(())
        }

        fn get_attention_maps(
            &mut self,
            _input: &Array2<f64>,
        ) -> NeuralResult<HashMap<String, Array2<f64>>> {
            Ok(HashMap::new())
        }

        fn parameter_count(&self) -> usize {
            self.params
        }
    }
}
