//! Common utilities shared across ensemble implementations

use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::types::Float;

/// Activation functions for ensemble operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationFunction {
    /// ReLU
    ReLU,
    /// Sigmoid
    Sigmoid,
    /// Tanh
    Tanh,
}

/// Fallback SIMD implementations for ensemble operations
pub mod simd_fallback {
    /// Add two vectors element-wise
    pub fn add_vec(a: &[f32], b: &[f32], result: &mut [f32]) {
        for i in 0..a.len() {
            result[i] = a[i] + b[i];
        }
    }

    /// Multiply two vectors element-wise
    pub fn multiply_vec(a: &[f32], b: &[f32], result: &mut [f32]) {
        for i in 0..a.len() {
            result[i] = a[i] * b[i];
        }
    }

    /// Divide two vectors element-wise
    pub fn divide_vec(a: &[f32], b: &[f32], result: &mut [f32]) {
        for i in 0..a.len() {
            result[i] = a[i] / b[i];
        }
    }

    /// Scale a vector by a scalar value
    pub fn scale_vec(vec: &[f32], scale: f32, result: &mut [f32]) {
        for i in 0..vec.len() {
            result[i] = vec[i] * scale;
        }
    }
}

/// Ensemble prediction statistics computed using SIMD
#[derive(Debug, Clone)]
pub struct EnsembleStatistics {
    /// The mean.
    pub mean: Float,
    /// The variance.
    pub variance: Float,
    /// The confidence.
    pub confidence: Float,
    /// The diversity.
    pub diversity: Float,
    /// The bias.
    pub bias: Float,
    /// The prediction entropy.
    pub prediction_entropy: Float,
    /// The disagreement.
    pub disagreement: Float,
    /// The average confidence.
    pub average_confidence: Float,
    /// The min confidence.
    pub min_confidence: Float,
    /// The max confidence.
    pub max_confidence: Float,
    /// The std confidence.
    pub std_confidence: Float,
    /// The skew confidence.
    pub skew_confidence: Float,
    /// The kurtosis confidence.
    pub kurtosis_confidence: Float,
    /// The median confidence.
    pub median_confidence: Float,
    /// The iqr confidence.
    pub iqr_confidence: Float,
    /// The prediction stability.
    pub prediction_stability: Float,
    /// The convergence rate.
    pub convergence_rate: Float,
    /// The ensemble complexity.
    pub ensemble_complexity: Float,
    /// The overfitting risk.
    pub overfitting_risk: Float,
    /// The generalization error.
    pub generalization_error: Float,
    /// The calibration score.
    pub calibration_score: Float,
}

impl Default for EnsembleStatistics {
    fn default() -> Self {
        Self {
            mean: 0.0,
            variance: 0.0,
            confidence: 0.0,
            diversity: 0.0,
            bias: 0.0,
            prediction_entropy: 0.0,
            disagreement: 0.0,
            average_confidence: 0.0,
            min_confidence: 0.0,
            max_confidence: 0.0,
            std_confidence: 0.0,
            skew_confidence: 0.0,
            kurtosis_confidence: 0.0,
            median_confidence: 0.0,
            iqr_confidence: 0.0,
            prediction_stability: 0.0,
            convergence_rate: 0.0,
            ensemble_complexity: 0.0,
            overfitting_risk: 0.0,
            generalization_error: 0.0,
            calibration_score: 0.0,
        }
    }
}

impl EnsembleStatistics {
    /// Compute statistics from predictions array
    #[must_use]
    pub fn from_predictions(predictions: &Array2<Float>) -> Self {
        let mut stats = EnsembleStatistics::default();

        if predictions.is_empty() {
            return stats;
        }

        // Calculate mean and variance across ensemble predictions
        let mean = predictions.mean().unwrap_or(0.0);
        let variance = predictions.var(0.0);

        stats.mean = mean;
        stats.variance = variance;
        stats.confidence = 1.0 - variance; // Simple confidence measure

        // Calculate diversity as variance across predictors
        let row_means: Array1<Float> = predictions.mean_axis(Axis(1)).unwrap_or_default();
        stats.diversity = row_means.var(0.0);

        stats
    }
}

/// SIMD ensemble operations module
pub mod simd_ensemble {
    use super::ActivationFunction;

    /// Apply activation function using SIMD operations
    pub fn apply_activation_simd(values: &mut Vec<f32>, activation: ActivationFunction) {
        match activation {
            ActivationFunction::ReLU => {
                for value in values {
                    *value = value.max(0.0);
                }
            }
            ActivationFunction::Sigmoid => {
                for value in values {
                    *value = 1.0 / (1.0 + (-*value).exp());
                }
            }
            ActivationFunction::Tanh => {
                for value in values {
                    *value = value.tanh();
                }
            }
        }
    }
}
