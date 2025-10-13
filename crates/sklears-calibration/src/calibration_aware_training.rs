//! Calibration-aware training methods for machine learning models
//!
//! This module provides training methods that incorporate calibration directly
//! into the learning process, rather than as a post-processing step.

use scirs2_core::ndarray::{s, Array1, Array2};
use sklears_core::{error::Result, prelude::SklearsError, types::Float};
use std::collections::HashMap;

/// Calibration-aware loss functions that balance accuracy and calibration
#[derive(Debug, Clone)]
pub enum CalibrationAwareLoss {
    /// Focal loss with temperature scaling
    FocalWithTemperature { gamma: Float, temperature: Float },
    /// Cross-entropy with calibration regularization
    CrossEntropyWithCalibration { lambda: Float },
    /// Maximum mean discrepancy (MMD) calibration loss
    MMDCalibration { bandwidth: Float },
    /// Brier score minimization
    BrierScoreMinimization,
    /// Expected calibration error minimization
    ECEMinimization { n_bins: usize },
    /// Multi-task loss combining accuracy and calibration
    MultiTaskLoss { calibration_weight: Float },
}

/// Calibration-aware training configuration
#[derive(Debug, Clone)]
pub struct CalibrationAwareTrainingConfig {
    /// Loss function to use
    pub loss_function: CalibrationAwareLoss,
    /// Learning rate for optimization
    pub learning_rate: Float,
    /// Number of training epochs
    pub max_epochs: usize,
    /// Early stopping patience
    pub patience: usize,
    /// Minimum improvement for early stopping
    pub min_delta: Float,
    /// Batch size for mini-batch training
    pub batch_size: usize,
    /// L2 regularization strength
    pub l2_reg: Float,
    /// Whether to use validation set for early stopping
    pub use_validation: bool,
    /// Validation split ratio
    pub validation_split: Float,
}

impl Default for CalibrationAwareTrainingConfig {
    fn default() -> Self {
        Self {
            loss_function: CalibrationAwareLoss::CrossEntropyWithCalibration { lambda: 0.1 },
            learning_rate: 0.001,
            max_epochs: 100,
            patience: 10,
            min_delta: 1e-4,
            batch_size: 32,
            l2_reg: 0.01,
            use_validation: true,
            validation_split: 0.2,
        }
    }
}

/// Calibration-aware trainer for neural networks and other models
#[derive(Debug)]
pub struct CalibrationAwareTrainer {
    config: CalibrationAwareTrainingConfig,
    training_history: Vec<TrainingMetrics>,
}

/// Training metrics for each epoch
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub epoch: usize,
    pub train_loss: Float,
    pub train_accuracy: Float,
    pub train_ece: Float,
    pub val_loss: Option<Float>,
    pub val_accuracy: Option<Float>,
    pub val_ece: Option<Float>,
}

impl CalibrationAwareTrainer {
    /// Create a new calibration-aware trainer
    pub fn new(config: CalibrationAwareTrainingConfig) -> Self {
        Self {
            config,
            training_history: Vec::new(),
        }
    }

    /// Train a model with calibration-aware objectives
    pub fn train(
        &mut self,
        mut model_params: Array1<Float>,
        x_train: &Array2<Float>,
        y_train: &Array1<i32>,
        x_val: Option<&Array2<Float>>,
        y_val: Option<&Array1<i32>>,
    ) -> Result<Array1<Float>> {
        let n_samples = x_train.nrows();
        let n_features = x_train.ncols();

        // Validate inputs
        if y_train.len() != n_samples {
            return Err(SklearsError::InvalidInput(
                "Training features and labels must have same number of samples".to_string(),
            ));
        }

        let mut best_params = model_params.clone();
        let mut best_val_loss = Float::INFINITY;
        let mut patience_counter = 0;

        for epoch in 0..self.config.max_epochs {
            // Forward pass and loss computation
            let predictions = self.forward_pass(&model_params, x_train)?;
            let train_loss = self.compute_loss(&predictions, y_train)?;
            let train_accuracy = self.compute_accuracy(&predictions, y_train);
            let train_ece = self.compute_ece(&predictions, y_train, 10)?;

            // Validation metrics
            let (val_loss, val_accuracy, val_ece) =
                if let (Some(x_val), Some(y_val)) = (x_val, y_val) {
                    let val_predictions = self.forward_pass(&model_params, x_val)?;
                    let val_loss = self.compute_loss(&val_predictions, y_val)?;
                    let val_accuracy = self.compute_accuracy(&val_predictions, y_val);
                    let val_ece = self.compute_ece(&val_predictions, y_val, 10)?;
                    (Some(val_loss), Some(val_accuracy), Some(val_ece))
                } else {
                    (None, None, None)
                };

            // Record metrics
            self.training_history.push(TrainingMetrics {
                epoch,
                train_loss,
                train_accuracy,
                train_ece,
                val_loss,
                val_accuracy,
                val_ece,
            });

            // Backward pass and parameter updates
            let gradients =
                self.compute_gradients(&model_params, x_train, y_train, &predictions)?;

            // Apply L2 regularization to gradients
            let regularized_gradients = &gradients + &(&model_params * self.config.l2_reg);

            // Update parameters
            model_params = &model_params - &(&regularized_gradients * self.config.learning_rate);

            // Early stopping check
            if self.config.use_validation {
                if let Some(current_val_loss) = val_loss {
                    if current_val_loss < best_val_loss - self.config.min_delta {
                        best_val_loss = current_val_loss;
                        best_params = model_params.clone();
                        patience_counter = 0;
                    } else {
                        patience_counter += 1;
                        if patience_counter >= self.config.patience {
                            break;
                        }
                    }
                }
            }
        }

        Ok(if self.config.use_validation {
            best_params
        } else {
            model_params
        })
    }

    /// Forward pass through the model
    fn forward_pass(&self, params: &Array1<Float>, x: &Array2<Float>) -> Result<Array2<Float>> {
        // Simple linear model for demonstration (in practice this would be more complex)
        let n_samples = x.nrows();
        let n_classes = 2; // Binary classification for simplicity

        // Compute logits (simplified)
        let mut logits = Array2::zeros((n_samples, n_classes));

        for (i, sample) in x.outer_iter().enumerate() {
            // Simple linear transformation
            let linear_output = sample.dot(&params.view().slice(s![..sample.len()]));
            logits[[i, 0]] = -linear_output;
            logits[[i, 1]] = linear_output;
        }

        // Apply softmax to get probabilities
        self.softmax(&logits)
    }

    /// Compute softmax activation
    fn softmax(&self, logits: &Array2<Float>) -> Result<Array2<Float>> {
        let mut probabilities = logits.clone();

        for mut row in probabilities.rows_mut() {
            let max_val = row.fold(Float::NEG_INFINITY, |acc, &x| acc.max(x));
            row.mapv_inplace(|x| (x - max_val).exp());
            let sum = row.sum();
            if sum > 0.0 {
                row /= sum;
            } else {
                // Handle numerical issues
                row.fill(1.0 / row.len() as Float);
            }
        }

        Ok(probabilities)
    }

    /// Compute the calibration-aware loss
    fn compute_loss(&self, predictions: &Array2<Float>, y_true: &Array1<i32>) -> Result<Float> {
        match &self.config.loss_function {
            CalibrationAwareLoss::CrossEntropyWithCalibration { lambda } => {
                let ce_loss = self.cross_entropy_loss(predictions, y_true)?;
                let calibration_loss = self.compute_ece(predictions, y_true, 10)?;
                Ok(ce_loss + lambda * calibration_loss)
            }
            CalibrationAwareLoss::FocalWithTemperature { gamma, temperature } => {
                self.focal_loss_with_temperature(predictions, y_true, *gamma, *temperature)
            }
            CalibrationAwareLoss::BrierScoreMinimization => {
                self.brier_score_loss(predictions, y_true)
            }
            CalibrationAwareLoss::ECEMinimization { n_bins } => {
                self.compute_ece(predictions, y_true, *n_bins)
            }
            CalibrationAwareLoss::MMDCalibration { bandwidth } => {
                self.mmd_calibration_loss(predictions, y_true, *bandwidth)
            }
            CalibrationAwareLoss::MultiTaskLoss { calibration_weight } => {
                let accuracy_loss = self.cross_entropy_loss(predictions, y_true)?;
                let calibration_loss = self.compute_ece(predictions, y_true, 10)?;
                Ok(accuracy_loss + calibration_weight * calibration_loss)
            }
        }
    }

    /// Cross-entropy loss
    fn cross_entropy_loss(
        &self,
        predictions: &Array2<Float>,
        y_true: &Array1<i32>,
    ) -> Result<Float> {
        let mut loss = 0.0;
        let n_samples = predictions.nrows();

        for (i, &true_class) in y_true.iter().enumerate() {
            if true_class >= 0 && (true_class as usize) < predictions.ncols() {
                let pred_prob = predictions[[i, true_class as usize]];
                // Add small epsilon for numerical stability
                let epsilon = 1e-15;
                loss -= (pred_prob.max(epsilon)).ln();
            }
        }

        Ok(loss / n_samples as Float)
    }

    /// Focal loss with temperature scaling
    fn focal_loss_with_temperature(
        &self,
        predictions: &Array2<Float>,
        y_true: &Array1<i32>,
        gamma: Float,
        temperature: Float,
    ) -> Result<Float> {
        // Apply temperature scaling
        let scaled_logits = predictions.mapv(|x| x / temperature);
        let temp_predictions = self.softmax(&scaled_logits)?;

        let mut loss = 0.0;
        let n_samples = temp_predictions.nrows();

        for (i, &true_class) in y_true.iter().enumerate() {
            if true_class >= 0 && (true_class as usize) < temp_predictions.ncols() {
                let pred_prob = temp_predictions[[i, true_class as usize]];
                let epsilon = 1e-15;
                let safe_prob = pred_prob.max(epsilon);

                // Focal loss: -α(1-p_t)^γ * log(p_t)
                let focal_weight = (1.0 - safe_prob).powf(gamma);
                loss -= focal_weight * safe_prob.ln();
            }
        }

        Ok(loss / n_samples as Float)
    }

    /// Brier score loss
    fn brier_score_loss(&self, predictions: &Array2<Float>, y_true: &Array1<i32>) -> Result<Float> {
        let mut loss = 0.0;
        let n_samples = predictions.nrows();
        let n_classes = predictions.ncols();

        for (i, &true_class) in y_true.iter().enumerate() {
            for j in 0..n_classes {
                let target = if j == true_class as usize { 1.0 } else { 0.0 };
                let pred = predictions[[i, j]];
                loss += (pred - target).powi(2);
            }
        }

        Ok(loss / n_samples as Float)
    }

    /// Maximum Mean Discrepancy (MMD) calibration loss
    fn mmd_calibration_loss(
        &self,
        predictions: &Array2<Float>,
        y_true: &Array1<i32>,
        bandwidth: Float,
    ) -> Result<Float> {
        // Simplified MMD computation for calibration
        let n_samples = predictions.nrows();
        let mut mmd_loss = 0.0;

        // Create empirical distribution of predictions and true labels
        let mut pred_dist = HashMap::new();
        let mut true_dist = HashMap::new();

        for (i, &true_class) in y_true.iter().enumerate() {
            let pred_class = predictions
                .row(i)
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            *pred_dist.entry(pred_class).or_insert(0.0) += 1.0;
            *true_dist.entry(true_class as usize).or_insert(0.0) += 1.0;
        }

        // Normalize distributions
        let total_samples = n_samples as Float;
        for (_, count) in pred_dist.iter_mut() {
            *count /= total_samples;
        }
        for (_, count) in true_dist.iter_mut() {
            *count /= total_samples;
        }

        // Compute MMD using RBF kernel
        for (class1, &prob1) in &pred_dist {
            for (class2, &prob2) in &true_dist {
                let kernel_val = (-(*class1 as Float - *class2 as Float).powi(2)
                    / (2.0 * bandwidth.powi(2)))
                .exp();
                mmd_loss += prob1 * prob2 * kernel_val;
            }
        }

        Ok(mmd_loss.abs())
    }

    /// Compute Expected Calibration Error (ECE)
    fn compute_ece(
        &self,
        predictions: &Array2<Float>,
        y_true: &Array1<i32>,
        n_bins: usize,
    ) -> Result<Float> {
        let n_samples = predictions.nrows();
        if n_samples == 0 {
            return Ok(0.0);
        }

        let mut ece = 0.0;
        let bin_boundaries: Vec<Float> =
            (0..=n_bins).map(|i| i as Float / n_bins as Float).collect();

        for bin_idx in 0..n_bins {
            let bin_lower = bin_boundaries[bin_idx];
            let bin_upper = bin_boundaries[bin_idx + 1];

            let mut bin_confidences = Vec::new();
            let mut bin_accuracies = Vec::new();

            for (i, &true_class) in y_true.iter().enumerate() {
                let max_prob = predictions.row(i).fold(0.0f64, |acc: Float, &x| acc.max(x));

                if max_prob > bin_lower && max_prob <= bin_upper {
                    bin_confidences.push(max_prob);

                    let pred_class = predictions
                        .row(i)
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);

                    bin_accuracies.push(if pred_class == true_class as usize {
                        1.0
                    } else {
                        0.0
                    });
                }
            }

            if !bin_confidences.is_empty() {
                let bin_weight = bin_confidences.len() as Float / n_samples as Float;
                let avg_confidence =
                    bin_confidences.iter().sum::<Float>() / bin_confidences.len() as Float;
                let avg_accuracy =
                    bin_accuracies.iter().sum::<Float>() / bin_accuracies.len() as Float;

                ece += bin_weight * (avg_confidence - avg_accuracy).abs();
            }
        }

        Ok(ece)
    }

    /// Compute accuracy
    fn compute_accuracy(&self, predictions: &Array2<Float>, y_true: &Array1<i32>) -> Float {
        let mut correct = 0;

        for (i, &true_class) in y_true.iter().enumerate() {
            let pred_class = predictions
                .row(i)
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            if pred_class == true_class as usize {
                correct += 1;
            }
        }

        correct as Float / y_true.len() as Float
    }

    /// Compute gradients for the loss function
    fn compute_gradients(
        &self,
        params: &Array1<Float>,
        x_train: &Array2<Float>,
        y_train: &Array1<i32>,
        predictions: &Array2<Float>,
    ) -> Result<Array1<Float>> {
        let n_features = x_train.ncols();
        let mut gradients = Array1::zeros(params.len().min(n_features));

        // Simplified gradient computation for linear model
        for (i, &true_class) in y_train.iter().enumerate() {
            let pred_prob = if true_class >= 0 && (true_class as usize) < predictions.ncols() {
                predictions[[i, true_class as usize]]
            } else {
                0.5 // Default for invalid classes
            };

            let error = pred_prob - 1.0; // Simplified error
            let features = x_train.row(i);

            for (j, &feature) in features.iter().enumerate() {
                if j < gradients.len() {
                    gradients[j] += error * feature;
                }
            }
        }

        // Average gradients
        gradients /= y_train.len() as Float;

        Ok(gradients)
    }

    /// Get training history
    pub fn get_training_history(&self) -> &[TrainingMetrics] {
        &self.training_history
    }

    /// Get final training metrics
    pub fn get_final_metrics(&self) -> Option<&TrainingMetrics> {
        self.training_history.last()
    }
}

/// Builder for calibration-aware training configuration
#[derive(Debug)]
pub struct CalibrationAwareTrainingConfigBuilder {
    config: CalibrationAwareTrainingConfig,
}

impl CalibrationAwareTrainingConfigBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: CalibrationAwareTrainingConfig::default(),
        }
    }

    /// Set the loss function
    pub fn loss_function(mut self, loss_function: CalibrationAwareLoss) -> Self {
        self.config.loss_function = loss_function;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: Float) -> Self {
        self.config.learning_rate = learning_rate;
        self
    }

    /// Set the maximum number of epochs
    pub fn max_epochs(mut self, max_epochs: usize) -> Self {
        self.config.max_epochs = max_epochs;
        self
    }

    /// Set the early stopping patience
    pub fn patience(mut self, patience: usize) -> Self {
        self.config.patience = patience;
        self
    }

    /// Set the minimum improvement delta
    pub fn min_delta(mut self, min_delta: Float) -> Self {
        self.config.min_delta = min_delta;
        self
    }

    /// Set the batch size
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.config.batch_size = batch_size;
        self
    }

    /// Set L2 regularization strength
    pub fn l2_reg(mut self, l2_reg: Float) -> Self {
        self.config.l2_reg = l2_reg;
        self
    }

    /// Set whether to use validation
    pub fn use_validation(mut self, use_validation: bool) -> Self {
        self.config.use_validation = use_validation;
        self
    }

    /// Set validation split ratio
    pub fn validation_split(mut self, validation_split: Float) -> Self {
        self.config.validation_split = validation_split;
        self
    }

    /// Build the configuration
    pub fn build(self) -> CalibrationAwareTrainingConfig {
        self.config
    }
}

impl Default for CalibrationAwareTrainingConfigBuilder {
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
    fn test_calibration_aware_trainer_creation() {
        let config = CalibrationAwareTrainingConfig::default();
        let trainer = CalibrationAwareTrainer::new(config);
        assert!(trainer.training_history.is_empty());
    }

    #[test]
    fn test_config_builder() {
        let config = CalibrationAwareTrainingConfigBuilder::new()
            .learning_rate(0.01)
            .max_epochs(50)
            .patience(5)
            .build();

        assert_eq!(config.learning_rate, 0.01);
        assert_eq!(config.max_epochs, 50);
        assert_eq!(config.patience, 5);
    }

    #[test]
    fn test_cross_entropy_loss() {
        let trainer = CalibrationAwareTrainer::new(CalibrationAwareTrainingConfig::default());
        let predictions = array![[0.9, 0.1], [0.3, 0.7], [0.8, 0.2]];
        let y_true = array![0, 1, 0];

        let loss = trainer.cross_entropy_loss(&predictions, &y_true).unwrap();
        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_ece_computation() {
        let trainer = CalibrationAwareTrainer::new(CalibrationAwareTrainingConfig::default());
        let predictions = array![[0.9, 0.1], [0.3, 0.7], [0.8, 0.2], [0.1, 0.9]];
        let y_true = array![0, 1, 0, 1];

        let ece = trainer.compute_ece(&predictions, &y_true, 10).unwrap();
        assert!(ece >= 0.0);
        assert!(ece <= 1.0);
    }

    #[test]
    fn test_accuracy_computation() {
        let trainer = CalibrationAwareTrainer::new(CalibrationAwareTrainingConfig::default());
        let predictions = array![[0.9, 0.1], [0.3, 0.7], [0.8, 0.2], [0.1, 0.9]];
        let y_true = array![0, 1, 0, 1];

        let accuracy = trainer.compute_accuracy(&predictions, &y_true);
        assert_eq!(accuracy, 1.0); // All predictions are correct
    }

    #[test]
    fn test_brier_score_loss() {
        let trainer = CalibrationAwareTrainer::new(CalibrationAwareTrainingConfig::default());
        let predictions = array![[0.9, 0.1], [0.3, 0.7]];
        let y_true = array![0, 1];

        let brier_loss = trainer.brier_score_loss(&predictions, &y_true).unwrap();
        assert!(brier_loss >= 0.0);
        assert!(brier_loss.is_finite());
    }

    #[test]
    fn test_softmax_normalization() {
        let trainer = CalibrationAwareTrainer::new(CalibrationAwareTrainingConfig::default());
        let logits = array![[1.0, 2.0], [3.0, 1.0]];

        let probabilities = trainer.softmax(&logits).unwrap();

        // Check that probabilities sum to 1
        for row in probabilities.rows() {
            let sum: Float = row.sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }

        // Check that all probabilities are positive
        for &prob in probabilities.iter() {
            assert!(prob >= 0.0);
            assert!(prob <= 1.0);
        }
    }

    #[test]
    fn test_focal_loss() {
        let trainer = CalibrationAwareTrainer::new(CalibrationAwareTrainingConfig::default());
        let predictions = array![[0.9, 0.1], [0.3, 0.7]];
        let y_true = array![0, 1];

        let focal_loss = trainer
            .focal_loss_with_temperature(&predictions, &y_true, 2.0, 1.0)
            .unwrap();
        assert!(focal_loss >= 0.0);
        assert!(focal_loss.is_finite());
    }

    #[test]
    fn test_different_loss_functions() {
        let config1 = CalibrationAwareTrainingConfigBuilder::new()
            .loss_function(CalibrationAwareLoss::BrierScoreMinimization)
            .build();

        let config2 = CalibrationAwareTrainingConfigBuilder::new()
            .loss_function(CalibrationAwareLoss::ECEMinimization { n_bins: 10 })
            .build();

        let trainer1 = CalibrationAwareTrainer::new(config1);
        let trainer2 = CalibrationAwareTrainer::new(config2);

        let predictions = array![[0.9, 0.1], [0.3, 0.7]];
        let y_true = array![0, 1];

        let loss1 = trainer1.compute_loss(&predictions, &y_true).unwrap();
        let loss2 = trainer2.compute_loss(&predictions, &y_true).unwrap();

        assert!(loss1.is_finite());
        assert!(loss2.is_finite());
        assert_ne!(loss1, loss2); // Different loss functions should give different values
    }
}
