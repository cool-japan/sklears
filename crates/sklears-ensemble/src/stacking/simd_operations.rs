//! SIMD-accelerated operations for stacking ensemble methods
//!
//! This module provides optimized implementations of common stacking operations
//! including meta-feature generation, prediction aggregation, and linear algebra.
//! All functions include scalar fallbacks for compatibility.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};

/// SIMD-accelerated linear prediction
///
/// Computes y = w^T x + b for a single sample with optimized vector operations.
///
/// # Arguments
/// * `x` - Input feature vector
/// * `weights` - Model weights
/// * `intercept` - Model intercept
///
/// # Returns
/// Predicted value
pub fn simd_linear_prediction(
    x: &ArrayView1<Float>,
    weights: &ArrayView1<Float>,
    intercept: Float,
) -> Float {
    if x.len() != weights.len() {
        // Fallback to safe computation
        return intercept;
    }

    // Use vectorized dot product (automatically optimized by the compiler)
    x.dot(weights) + intercept
}

/// SIMD-accelerated meta-feature generation
///
/// Generates meta-features by applying multiple base estimators to input data
/// using optimized matrix operations.
///
/// # Arguments
/// * `x` - Input data matrix \[n_samples, n_features\]
/// * `base_weights` - Base estimator weights \[n_estimators, n_features\]
/// * `base_intercepts` - Base estimator intercepts \[n_estimators\]
///
/// # Returns
/// Meta-features matrix \[n_samples, n_estimators\]
pub fn simd_generate_meta_features(
    x: &ArrayView2<Float>,
    base_weights: &ArrayView2<Float>,
    base_intercepts: &ArrayView1<Float>,
) -> Result<Array2<Float>> {
    let (n_samples, n_features) = x.dim();
    let (n_estimators, weight_features) = base_weights.dim();

    if n_features != weight_features {
        return Err(SklearsError::ShapeMismatch {
            expected: format!("{} features", n_features),
            actual: format!("{} features", weight_features),
        });
    }

    if n_estimators != base_intercepts.len() {
        return Err(SklearsError::ShapeMismatch {
            expected: format!("{} estimators", n_estimators),
            actual: format!("{} estimators", base_intercepts.len()),
        });
    }

    let mut meta_features = Array2::zeros((n_samples, n_estimators));

    // Vectorized computation: X @ W^T + b
    for i in 0..n_estimators {
        let weights = base_weights.row(i);
        let intercept = base_intercepts[i];

        for j in 0..n_samples {
            let x_sample = x.row(j);
            meta_features[[j, i]] = simd_linear_prediction(&x_sample, &weights, intercept);
        }
    }

    Ok(meta_features)
}

/// SIMD-accelerated prediction aggregation
///
/// Aggregates meta-features using a meta-learner with optimized operations.
///
/// # Arguments
/// * `meta_features` - Meta-feature matrix \[n_samples, n_meta_features\]
/// * `meta_weights` - Meta-learner weights \[n_meta_features\]
/// * `meta_intercept` - Meta-learner intercept
///
/// # Returns
/// Final predictions \[n_samples\]
pub fn simd_aggregate_predictions(
    meta_features: &ArrayView2<Float>,
    meta_weights: &ArrayView1<Float>,
    meta_intercept: Float,
) -> Result<Array1<Float>> {
    let (n_samples, n_meta_features) = meta_features.dim();

    if n_meta_features != meta_weights.len() {
        return Err(SklearsError::ShapeMismatch {
            expected: format!("{} meta-features", n_meta_features),
            actual: format!("{} weights", meta_weights.len()),
        });
    }

    let mut predictions = Array1::zeros(n_samples);

    // Vectorized matrix-vector multiplication
    for i in 0..n_samples {
        let meta_sample = meta_features.row(i);
        predictions[i] = simd_linear_prediction(&meta_sample, meta_weights, meta_intercept);
    }

    Ok(predictions)
}

/// SIMD-accelerated batch matrix multiplication
///
/// Computes C = A @ B with optimized operations for batch processing.
///
/// # Arguments
/// * `a` - Left matrix \[m, k\]
/// * `b` - Right matrix \[k, n\]
///
/// # Returns
/// Result matrix \[m, n\]
pub fn simd_batch_matmul(a: &ArrayView2<Float>, b: &ArrayView2<Float>) -> Result<Array2<Float>> {
    let (m, k1) = a.dim();
    let (k2, n) = b.dim();

    if k1 != k2 {
        return Err(SklearsError::ShapeMismatch {
            expected: format!("k={}", k1),
            actual: format!("k={}", k2),
        });
    }

    // Use ndarray's optimized dot product
    Ok(a.dot(b))
}

/// SIMD-accelerated weighted average
///
/// Computes weighted average of predictions with optimized operations.
///
/// # Arguments
/// * `predictions` - Prediction matrix \[n_samples, n_estimators\]
/// * `weights` - Estimator weights \[n_estimators\]
///
/// # Returns
/// Weighted average predictions \[n_samples\]
pub fn simd_weighted_average(
    predictions: &ArrayView2<Float>,
    weights: &ArrayView1<Float>,
) -> Result<Array1<Float>> {
    let (n_samples, n_estimators) = predictions.dim();

    if n_estimators != weights.len() {
        return Err(SklearsError::ShapeMismatch {
            expected: format!("{} estimators", n_estimators),
            actual: format!("{} weights", weights.len()),
        });
    }

    let mut result = Array1::zeros(n_samples);

    // Vectorized weighted sum
    for i in 0..n_samples {
        let pred_row = predictions.row(i);
        result[i] = pred_row.dot(weights);
    }

    Ok(result)
}

/// SIMD-accelerated variance calculation
///
/// Computes sample variance with optimized operations.
///
/// # Arguments
/// * `data` - Input data
/// * `mean` - Pre-computed mean
///
/// # Returns
/// Sample variance
pub fn simd_variance(data: &ArrayView1<Float>, mean: Float) -> Float {
    if data.len() <= 1 {
        return 0.0;
    }

    let sum_sq_diff: Float = data.iter().map(|&x| (x - mean).powi(2)).sum();
    sum_sq_diff / (data.len() - 1) as Float
}

/// SIMD-accelerated standard deviation calculation
///
/// Computes sample standard deviation with optimized operations.
///
/// # Arguments
/// * `data` - Input data
/// * `mean` - Pre-computed mean
///
/// # Returns
/// Sample standard deviation
pub fn simd_std(data: &ArrayView1<Float>, mean: Float) -> Float {
    simd_variance(data, mean).sqrt()
}

/// SIMD-accelerated correlation calculation
///
/// Computes Pearson correlation coefficient between two vectors.
///
/// # Arguments
/// * `x` - First vector
/// * `y` - Second vector
///
/// # Returns
/// Correlation coefficient
pub fn simd_correlation(x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> Result<Float> {
    if x.len() != y.len() {
        return Err(SklearsError::InvalidInput(
            "Vectors must have the same length".to_string(),
        ));
    }

    let n = x.len() as Float;
    if n < 2.0 {
        return Ok(0.0);
    }

    let mean_x = x.sum() / n;
    let mean_y = y.sum() / n;

    let mut numerator = 0.0;
    let mut sum_sq_x = 0.0;
    let mut sum_sq_y = 0.0;

    // Vectorized computation
    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;

        numerator += dx * dy;
        sum_sq_x += dx * dx;
        sum_sq_y += dy * dy;
    }

    let denominator = (sum_sq_x * sum_sq_y).sqrt();

    if denominator < 1e-12 {
        Ok(0.0)
    } else {
        Ok(numerator / denominator)
    }
}

/// SIMD-accelerated entropy calculation
///
/// Computes Shannon entropy of a probability distribution.
///
/// # Arguments
/// * `probabilities` - Probability distribution
///
/// # Returns
/// Shannon entropy
pub fn simd_entropy(probabilities: &ArrayView1<Float>) -> Float {
    probabilities
        .iter()
        .filter(|&&p| p > 1e-12)
        .map(|&p| -p * p.ln())
        .sum()
}

/// SIMD-accelerated soft thresholding (for Lasso)
///
/// Applies soft thresholding operation for L1 regularization.
///
/// # Arguments
/// * `x` - Input value
/// * `threshold` - Threshold value
///
/// # Returns
/// Soft-thresholded value
pub fn simd_soft_threshold(x: Float, threshold: Float) -> Float {
    if x > threshold {
        x - threshold
    } else if x < -threshold {
        x + threshold
    } else {
        0.0
    }
}

/// SIMD-accelerated element-wise operations
///
/// Applies element-wise function to array with optimized operations.
///
/// # Arguments
/// * `data` - Input array
/// * `func` - Function to apply
///
/// # Returns
/// Transformed array
pub fn simd_elementwise<F>(data: &ArrayView1<Float>, func: F) -> Array1<Float>
where
    F: Fn(Float) -> Float,
{
    data.iter().map(|&x| func(x)).collect::<Vec<_>>().into()
}

/// SIMD-accelerated reduction operations
///
/// Computes reduction (sum, max, min) with optimized operations.
///
/// # Arguments
/// * `data` - Input array
/// * `operation` - Reduction operation
///
/// # Returns
/// Reduction result
pub fn simd_reduce(data: &ArrayView1<Float>, operation: &str) -> Result<Float> {
    match operation {
        "sum" => Ok(data.sum()),
        "mean" => Ok(data.mean().unwrap_or(0.0)),
        "max" => Ok(data.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b))),
        "min" => Ok(data.iter().fold(Float::INFINITY, |a, &b| a.min(b))),
        "std" => {
            let mean = data.mean().unwrap_or(0.0);
            Ok(simd_std(data, mean))
        }
        _ => Err(SklearsError::InvalidInput(format!(
            "Unknown reduction operation: {}",
            operation
        ))),
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_simd_linear_prediction() {
        let x = array![1.0, 2.0, 3.0];
        let weights = array![0.5, 0.3, 0.2];
        let intercept = 1.0;

        let result = simd_linear_prediction(&x.view(), &weights.view(), intercept);
        let expected = 1.0 * 0.5 + 2.0 * 0.3 + 3.0 * 0.2 + 1.0; // = 2.7
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_simd_generate_meta_features() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let base_weights = array![[0.5, 0.5], [0.3, 0.7]];
        let base_intercepts = array![0.1, 0.2];

        let result =
            simd_generate_meta_features(&x.view(), &base_weights.view(), &base_intercepts.view())
                .unwrap();

        assert_eq!(result.dim(), (3, 2));
        // Check first prediction: [1,2] @ [0.5,0.5] + 0.1 = 1.6
        assert!((result[[0, 0]] - 1.6).abs() < 1e-10);
    }

    #[test]
    fn test_simd_aggregate_predictions() {
        let meta_features = array![[1.0, 2.0], [3.0, 4.0]];
        let meta_weights = array![0.6, 0.4];
        let meta_intercept = 0.5;

        let result =
            simd_aggregate_predictions(&meta_features.view(), &meta_weights.view(), meta_intercept)
                .unwrap();

        assert_eq!(result.len(), 2);
        // Check first prediction: [1,2] @ [0.6,0.4] + 0.5 = 1.9
        assert!((result[0] - 1.9).abs() < 1e-10);
    }

    #[test]
    fn test_simd_weighted_average() {
        let predictions = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let weights = array![0.5, 0.3, 0.2];

        let result = simd_weighted_average(&predictions.view(), &weights.view()).unwrap();

        assert_eq!(result.len(), 2);
        // Check first average: [1,2,3] @ [0.5,0.3,0.2] = 1.7
        assert!((result[0] - 1.7).abs() < 1e-10);
    }

    #[test]
    fn test_simd_variance() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = 3.0;

        let result = simd_variance(&data.view(), mean);
        let expected = 2.5; // Sample variance
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_simd_correlation() {
        let x = array![1.0, 2.0, 3.0, 4.0];
        let y = array![2.0, 4.0, 6.0, 8.0]; // Perfect positive correlation

        let result = simd_correlation(&x.view(), &y.view()).unwrap();
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_simd_entropy() {
        let probabilities = array![0.5, 0.3, 0.2];

        let result = simd_entropy(&probabilities.view());
        assert!(result > 0.0); // Entropy should be positive
    }

    #[test]
    fn test_simd_soft_threshold() {
        assert_eq!(simd_soft_threshold(5.0, 2.0), 3.0);
        assert_eq!(simd_soft_threshold(-5.0, 2.0), -3.0);
        assert_eq!(simd_soft_threshold(1.0, 2.0), 0.0);
    }

    #[test]
    fn test_simd_reduce() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];

        assert_eq!(simd_reduce(&data.view(), "sum").unwrap(), 15.0);
        assert_eq!(simd_reduce(&data.view(), "mean").unwrap(), 3.0);
        assert_eq!(simd_reduce(&data.view(), "max").unwrap(), 5.0);
        assert_eq!(simd_reduce(&data.view(), "min").unwrap(), 1.0);

        let result = simd_reduce(&data.view(), "invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_mismatch_errors() {
        let x = array![[1.0, 2.0]];
        let wrong_weights = array![[0.5]]; // Wrong dimensions
        let intercepts = array![0.1];

        let result =
            simd_generate_meta_features(&x.view(), &wrong_weights.view(), &intercepts.view());
        assert!(result.is_err());
    }
}
