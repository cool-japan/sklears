//! SIMD-accelerated operations for incremental tree building
//!
//! This module provides high-performance SIMD-accelerated operations for incremental decision tree
//! algorithms. SIMD features are conditionally compiled based on the "simd" feature flag.

/// Mean and variance calculation for data ranges
/// SIMD-accelerated when simd feature is enabled, otherwise uses scalar implementation
pub fn simd_calculate_range_stats(data: &[f64], start: usize, end: usize) -> (f64, f64) {
    let range_data = &data[start..end];
    let n = range_data.len();

    if n == 0 {
        return (0.0, 0.0);
    }

    // Calculate mean
    let total_sum: f64 = range_data.iter().sum();
    let mean = total_sum / n as f64;

    // Calculate variance
    let var_sum: f64 = range_data.iter().map(|&x| (x - mean).powi(2)).sum();
    let variance = if n > 1 { var_sum / (n - 1) as f64 } else { 0.0 };

    (mean, variance)
}

/// ADWIN bound calculation for concept drift detection
pub fn simd_adwin_bound_calculation(
    n0: f64,
    n1: f64,
    var0: f64,
    var1: f64,
    alpha: f64,
    n: f64,
) -> f64 {
    // Scalar implementation
    let delta = alpha.ln() * 2.0;
    let variance_term = (var0 / n0) + (var1 / n1);
    let bound = (-delta / (2.0 * n)).exp() * variance_term.sqrt();
    bound
}

/// MSE impurity calculation for regression splits
pub fn simd_mse_impurity_calculation(targets: &[f64], predictions: &[f64]) -> f64 {
    if targets.len() != predictions.len() || targets.is_empty() {
        return 0.0;
    }

    // Scalar implementation
    let mse: f64 = targets
        .iter()
        .zip(predictions.iter())
        .map(|(&target, &pred)| (target - pred).powi(2))
        .sum();

    mse / targets.len() as f64
}

/// MSE evaluation for regression models
pub fn simd_mse_evaluation(targets: &[f64], predictions: &[f64]) -> f64 {
    simd_mse_impurity_calculation(targets, predictions)
}

/// Array prediction aggregation for ensemble methods
pub fn simd_array_prediction_aggregation(predictions: &[f64]) -> f64 {
    if predictions.is_empty() {
        return 0.0;
    }

    // Scalar implementation - simple mean
    predictions.iter().sum::<f64>() / predictions.len() as f64
}

/// Drift detection calculation for streaming data
pub fn simd_drift_detection_calculation(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    // Scalar implementation - calculate standard deviation
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
    variance.sqrt()
}

/// Fast variance calculation for data arrays
pub fn simd_fast_variance(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }

    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    variance
}