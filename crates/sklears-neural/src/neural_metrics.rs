//! Neural Network Specific Metrics
//!
//! This module provides metrics specifically designed for neural networks,
//! including gradient-based metrics, model complexity metrics, training
//! dynamics, and attention mechanism analysis.

use crate::NeuralResult;
use scirs2_core::ndarray::{s, Array1, Array2, Array3, ArrayView2, Axis};
use scirs2_core::numeric::{Float, FromPrimitive, Signed};
use sklears_core::error::SklearsError;
use std::collections::HashMap;

/// Configuration for neural network metrics
#[derive(Debug, Clone)]
pub struct NeuralMetricsConfig {
    /// Whether to track gradient statistics
    pub track_gradients: bool,
    /// Whether to compute attention statistics
    pub track_attention: bool,
    /// Whether to estimate model complexity
    pub estimate_complexity: bool,
    /// Smoothing factor for exponential moving averages
    pub smoothing_factor: f64,
}

impl Default for NeuralMetricsConfig {
    fn default() -> Self {
        Self {
            track_gradients: true,
            track_attention: true,
            estimate_complexity: true,
            smoothing_factor: 0.9,
        }
    }
}

/// Gradient statistics for monitoring training dynamics
#[derive(Debug, Clone)]
pub struct GradientStatistics {
    /// Global gradient norm (L2 norm of all gradients)
    pub global_norm: f64,
    /// Maximum gradient magnitude across all parameters
    pub max_gradient: f64,
    /// Minimum gradient magnitude across all parameters
    pub min_gradient: f64,
    /// Mean gradient magnitude
    pub mean_gradient: f64,
    /// Standard deviation of gradients
    pub std_gradient: f64,
    /// Percentage of gradients that are zero or near zero
    pub sparsity: f64,
    /// Gradient clipping statistics if applicable
    pub clipping_stats: Option<ClippingStatistics>,
}

/// Gradient clipping statistics
#[derive(Debug, Clone)]
pub struct ClippingStatistics {
    /// Percentage of updates that were clipped
    pub clip_percentage: f64,
    /// Average clipping ratio when clipping occurred
    pub avg_clip_ratio: f64,
    /// Maximum clipping ratio observed
    pub max_clip_ratio: f64,
}

/// Model complexity metrics
#[derive(Debug, Clone)]
pub struct ModelComplexity {
    /// Total number of trainable parameters
    pub trainable_parameters: usize,
    /// Total number of parameters (including non-trainable)
    pub total_parameters: usize,
    /// Estimated FLOPs for forward pass
    pub forward_flops: u64,
    /// Estimated FLOPs for backward pass
    pub backward_flops: u64,
    /// Memory usage estimate in bytes
    pub memory_usage: usize,
    /// Model depth (number of layers)
    pub depth: usize,
    /// Average layer width
    pub avg_width: f64,
}

/// Attention mechanism statistics
#[derive(Debug, Clone)]
pub struct AttentionStatistics {
    /// Entropy of attention weights (higher = more uniform attention)
    pub attention_entropy: f64,
    /// Maximum attention weight
    pub max_attention: f64,
    /// Percentage of attention weights above threshold
    pub attention_concentration: f64,
    /// Head importance scores (for multi-head attention)
    pub head_importance: Option<Vec<f64>>,
    /// Token importance scores
    pub token_importance: Option<Vec<f64>>,
}

/// Training dynamics metrics
#[derive(Debug, Clone)]
pub struct TrainingDynamics {
    /// Loss smoothed with exponential moving average
    pub smoothed_loss: f64,
    /// Loss variance over recent iterations
    pub loss_variance: f64,
    /// Learning rate at current step
    pub current_lr: f64,
    /// Effective batch size
    pub effective_batch_size: usize,
    /// Training stability score (lower is more stable)
    pub stability_score: f64,
}

/// Calculate gradient statistics from a collection of parameter gradients
pub fn calculate_gradient_statistics<T>(
    gradients: &[ArrayView2<T>],
    clip_threshold: Option<T>,
) -> NeuralResult<GradientStatistics>
where
    T: Float + Signed + FromPrimitive + std::fmt::Debug + std::iter::Sum,
{
    if gradients.is_empty() {
        return Err(SklearsError::InvalidInput("Empty gradients".to_string()));
    }

    let mut all_gradients = Vec::new();
    let mut global_norm_squared = T::zero();

    // Flatten all gradients and calculate global norm
    for grad in gradients {
        for &g in grad.iter() {
            all_gradients.push(g);
            global_norm_squared = global_norm_squared + g * g;
        }
    }

    if all_gradients.is_empty() {
        return Err(SklearsError::InvalidInput("Empty gradients".to_string()));
    }

    let global_norm = global_norm_squared.sqrt();
    let max_gradient = all_gradients
        .iter()
        .map(|g| g.abs())
        .fold(T::zero(), T::max);
    let min_gradient = all_gradients
        .iter()
        .map(|g| g.abs())
        .fold(max_gradient, T::min);

    // Calculate mean and standard deviation
    let n = T::from(all_gradients.len()).unwrap();
    let sum: T = all_gradients.iter().map(|g| g.abs()).sum();
    let mean_gradient = sum / n;

    let variance: T = all_gradients
        .iter()
        .map(|g| {
            let diff = g.abs() - mean_gradient;
            diff * diff
        })
        .sum::<T>()
        / n;
    let std_gradient = variance.sqrt();

    // Calculate sparsity (percentage of near-zero gradients)
    let threshold = T::from(1e-6).unwrap();
    let near_zero_count = all_gradients
        .iter()
        .filter(|&&g| g.abs() < threshold)
        .count();
    let sparsity = (near_zero_count as f64) / (all_gradients.len() as f64);

    // Calculate clipping statistics if threshold provided
    let clipping_stats = if let Some(clip_thresh) = clip_threshold {
        let clipped_count = all_gradients
            .iter()
            .filter(|&&g| g.abs() > clip_thresh)
            .count();
        let clip_percentage = (clipped_count as f64) / (all_gradients.len() as f64);

        let clip_ratios: Vec<T> = all_gradients
            .iter()
            .filter_map(|&g| {
                if g.abs() > clip_thresh {
                    Some(g.abs() / clip_thresh)
                } else {
                    None
                }
            })
            .collect();

        let avg_clip_ratio = if !clip_ratios.is_empty() {
            clip_ratios.iter().copied().sum::<T>() / T::from(clip_ratios.len()).unwrap()
        } else {
            T::one()
        };

        let max_clip_ratio = clip_ratios
            .iter()
            .fold(T::one(), |max, &ratio| max.max(ratio));

        Some(ClippingStatistics {
            clip_percentage,
            avg_clip_ratio: avg_clip_ratio.to_f64().unwrap_or(1.0),
            max_clip_ratio: max_clip_ratio.to_f64().unwrap_or(1.0),
        })
    } else {
        None
    };

    Ok(GradientStatistics {
        global_norm: global_norm.to_f64().unwrap_or(0.0),
        max_gradient: max_gradient.to_f64().unwrap_or(0.0),
        min_gradient: min_gradient.to_f64().unwrap_or(0.0),
        mean_gradient: mean_gradient.to_f64().unwrap_or(0.0),
        std_gradient: std_gradient.to_f64().unwrap_or(0.0),
        sparsity,
        clipping_stats,
    })
}

/// Calculate attention entropy for attention weights
pub fn attention_entropy<T>(attention_weights: &Array2<T>) -> NeuralResult<f64>
where
    T: Float + FromPrimitive + std::fmt::Debug,
{
    if attention_weights.is_empty() {
        return Err(SklearsError::InvalidInput("Empty gradients".to_string()));
    }

    let mut total_entropy = 0.0;
    let num_heads = attention_weights.nrows();

    for head_idx in 0..num_heads {
        let head_weights = attention_weights.row(head_idx);
        let mut entropy = 0.0;

        for &weight in head_weights.iter() {
            if weight > T::zero() {
                let p = weight.to_f64().unwrap_or(0.0);
                entropy -= p * p.ln();
            }
        }

        total_entropy += entropy;
    }

    Ok(total_entropy / (num_heads as f64))
}

/// Calculate head importance scores for multi-head attention
pub fn head_importance<T>(
    attention_weights: &Array3<T>,
    gradients: Option<&Array3<T>>,
) -> NeuralResult<Vec<f64>>
where
    T: Float + FromPrimitive + std::fmt::Debug + std::iter::Sum,
{
    if attention_weights.is_empty() {
        return Err(SklearsError::InvalidInput("Empty gradients".to_string()));
    }

    let (num_layers, num_heads, _) = attention_weights.dim();
    let mut importance_scores = Vec::with_capacity(num_heads);

    for head_idx in 0..num_heads {
        let mut head_score = 0.0;

        for layer_idx in 0..num_layers {
            // Calculate attention variance for this head
            let head_attention = attention_weights.slice(s![layer_idx, head_idx, ..]);
            let mean = head_attention.mean().unwrap_or(T::zero());
            let variance: T = head_attention
                .iter()
                .map(|&x| {
                    let diff = x - mean;
                    diff * diff
                })
                .sum::<T>()
                / T::from(head_attention.len()).unwrap();

            head_score += variance.to_f64().unwrap_or(0.0);

            // If gradients provided, weight by gradient magnitude
            if let Some(grads) = gradients {
                let head_grads = grads.slice(s![layer_idx, head_idx, ..]);
                let grad_magnitude: T = head_grads.iter().map(|&g| g.abs()).sum();
                head_score *= grad_magnitude.to_f64().unwrap_or(1.0);
            }
        }

        importance_scores.push(head_score);
    }

    Ok(importance_scores)
}

/// Estimate model complexity metrics
pub fn estimate_model_complexity(
    layer_sizes: &[usize],
    input_size: usize,
    use_bias: bool,
) -> ModelComplexity {
    let mut trainable_params = 0;
    let mut forward_flops = 0u64;

    let mut prev_size = input_size;

    for &layer_size in layer_sizes {
        // Weight parameters: prev_size * layer_size
        trainable_params += prev_size * layer_size;

        // Bias parameters if used
        if use_bias {
            trainable_params += layer_size;
        }

        // Forward pass FLOPs: matrix multiplication + bias addition + activation
        forward_flops += (prev_size * layer_size * 2) as u64; // Matrix multiply
        if use_bias {
            forward_flops += layer_size as u64; // Bias addition
        }
        forward_flops += layer_size as u64; // Activation function

        prev_size = layer_size;
    }

    // Backward pass typically ~2x forward pass FLOPs
    let backward_flops = forward_flops * 2;

    // Rough memory usage estimate (parameters + activations + gradients)
    let memory_usage = trainable_params * 4 * 3; // 4 bytes per float, 3 copies

    let avg_width = if layer_sizes.is_empty() {
        0.0
    } else {
        layer_sizes.iter().sum::<usize>() as f64 / layer_sizes.len() as f64
    };

    ModelComplexity {
        trainable_parameters: trainable_params,
        total_parameters: trainable_params,
        forward_flops,
        backward_flops,
        memory_usage,
        depth: layer_sizes.len(),
        avg_width,
    }
}

/// Calculate training stability score based on loss history
pub fn training_stability_score(loss_history: &Array1<f64>, window_size: usize) -> f64 {
    if loss_history.len() < window_size {
        return f64::INFINITY; // Not enough data
    }

    let mut stability_scores = Vec::new();

    for i in window_size..loss_history.len() {
        let window = loss_history.slice(s![i.saturating_sub(window_size)..i]);
        let mean = window.mean().unwrap_or(0.0);
        let variance = window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / window_size as f64;

        // Stability score is coefficient of variation (std / mean)
        let stability = if mean.abs() > 1e-8 {
            variance.sqrt() / mean.abs()
        } else {
            f64::INFINITY
        };

        stability_scores.push(stability);
    }

    // Return average stability score
    stability_scores.iter().sum::<f64>() / stability_scores.len() as f64
}

/// Comprehensive neural network metrics collector
pub struct NeuralMetricsCollector {
    config: NeuralMetricsConfig,
    gradient_history: Vec<GradientStatistics>,
    loss_history: Vec<f64>,
    learning_rate_history: Vec<f64>,
}

impl NeuralMetricsCollector {
    /// Create new metrics collector
    pub fn new(config: NeuralMetricsConfig) -> Self {
        Self {
            config,
            gradient_history: Vec::new(),
            loss_history: Vec::new(),
            learning_rate_history: Vec::new(),
        }
    }

    /// Add gradient statistics to history
    pub fn add_gradient_stats(&mut self, stats: GradientStatistics) {
        self.gradient_history.push(stats);
    }

    /// Add loss value to history
    pub fn add_loss(&mut self, loss: f64) {
        self.loss_history.push(loss);
    }

    /// Add learning rate to history
    pub fn add_learning_rate(&mut self, lr: f64) {
        self.learning_rate_history.push(lr);
    }

    /// Get comprehensive training dynamics
    pub fn get_training_dynamics(&self) -> Option<TrainingDynamics> {
        if self.loss_history.is_empty() {
            return None;
        }

        let current_loss = *self.loss_history.last()?;
        let smoothed_loss = if self.loss_history.len() > 1 {
            let mut smoothed = self.loss_history[0];
            for &loss in &self.loss_history[1..] {
                smoothed = self.config.smoothing_factor * smoothed
                    + (1.0 - self.config.smoothing_factor) * loss;
            }
            smoothed
        } else {
            current_loss
        };

        let loss_variance = if self.loss_history.len() > 10 {
            let recent_losses = &self.loss_history[self.loss_history.len().saturating_sub(10)..];
            let mean = recent_losses.iter().sum::<f64>() / recent_losses.len() as f64;
            recent_losses
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>()
                / recent_losses.len() as f64
        } else {
            0.0
        };

        let current_lr = self.learning_rate_history.last().copied().unwrap_or(0.0);

        let stability_score = if self.loss_history.len() > 20 {
            let loss_array = Array1::from_vec(self.loss_history.clone());
            training_stability_score(&loss_array, 10)
        } else {
            f64::INFINITY
        };

        Some(TrainingDynamics {
            smoothed_loss,
            loss_variance,
            current_lr,
            effective_batch_size: 32, // This would need to be passed in
            stability_score,
        })
    }

    /// Generate comprehensive metrics report
    pub fn generate_report(&self) -> HashMap<String, f64> {
        let mut report = HashMap::new();

        // Add gradient statistics
        if let Some(latest_grad_stats) = self.gradient_history.last() {
            report.insert("gradient_norm".to_string(), latest_grad_stats.global_norm);
            report.insert("gradient_max".to_string(), latest_grad_stats.max_gradient);
            report.insert("gradient_mean".to_string(), latest_grad_stats.mean_gradient);
            report.insert("gradient_sparsity".to_string(), latest_grad_stats.sparsity);
        }

        // Add training dynamics
        if let Some(dynamics) = self.get_training_dynamics() {
            report.insert("smoothed_loss".to_string(), dynamics.smoothed_loss);
            report.insert("loss_variance".to_string(), dynamics.loss_variance);
            report.insert("current_lr".to_string(), dynamics.current_lr);
            report.insert("stability_score".to_string(), dynamics.stability_score);
        }

        report
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_gradient_statistics() {
        let grad1 = array![[1.0, 2.0], [3.0, 4.0]];
        let grad2 = array![[0.5, -1.5], [2.5, -0.5]];
        let gradients = vec![grad1.view(), grad2.view()];

        let stats = calculate_gradient_statistics(&gradients, Some(2.0)).unwrap();

        assert!(stats.global_norm > 0.0);
        assert!(stats.max_gradient >= stats.mean_gradient);
        assert!(stats.mean_gradient >= stats.min_gradient);
        assert!(stats.sparsity >= 0.0 && stats.sparsity <= 1.0);
    }

    #[test]
    fn test_attention_entropy() {
        let attention = array![[0.5, 0.3, 0.2], [0.1, 0.8, 0.1]];
        let entropy = attention_entropy(&attention).unwrap();
        assert!(entropy > 0.0);
    }

    #[test]
    fn test_model_complexity() {
        let layer_sizes = vec![100, 50, 10];
        let complexity = estimate_model_complexity(&layer_sizes, 784, true);

        assert_eq!(complexity.depth, 3);
        assert!(complexity.trainable_parameters > 0);
        assert!(complexity.forward_flops > 0);
        assert_relative_eq!(complexity.avg_width, 53.33333333333333, epsilon = 1e-6);
    }

    #[test]
    fn test_training_stability() {
        let loss_history = Array1::from_vec(vec![
            1.0, 0.9, 0.8, 0.85, 0.75, 0.7, 0.72, 0.68, 0.65, 0.67, 0.6, 0.58, 0.55,
        ]);
        let stability = training_stability_score(&loss_history, 5);
        assert!(stability >= 0.0);
        assert!(stability.is_finite());
    }

    #[test]
    fn test_metrics_collector() {
        let config = NeuralMetricsConfig::default();
        let mut collector = NeuralMetricsCollector::new(config);

        // Add some sample data
        collector.add_loss(1.0);
        collector.add_loss(0.8);
        collector.add_loss(0.6);
        collector.add_learning_rate(0.001);

        let dynamics = collector.get_training_dynamics().unwrap();
        assert!(dynamics.smoothed_loss > 0.0);
        assert_relative_eq!(dynamics.current_lr, 0.001, epsilon = 1e-9);

        let report = collector.generate_report();
        assert!(report.contains_key("smoothed_loss"));
        assert!(report.contains_key("current_lr"));
    }
}
