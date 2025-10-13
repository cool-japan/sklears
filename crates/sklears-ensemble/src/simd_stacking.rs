//! SIMD-accelerated stacking ensemble operations (scalar implementations)
//!
//! This module provides high-performance implementations of stacking ensemble
//! algorithms. Currently uses scalar operations with plans for future SIMD optimization.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};

/// Dot product computation for stacking ensembles
pub fn simd_dot_product(x: &ArrayView1<Float>, weights: &ArrayView1<Float>) -> Float {
    if x.len() != weights.len() {
        return 0.0;
    }
    x.iter().zip(weights.iter()).map(|(&xi, &wi)| xi * wi).sum()
}

/// Linear prediction for base estimators
pub fn simd_linear_prediction(
    x: &ArrayView1<Float>,
    weights: &ArrayView1<Float>,
    intercept: Float,
) -> Float {
    simd_dot_product(x, weights) + intercept
}

/// Batch linear predictions
pub fn simd_batch_linear_predictions(
    X: &ArrayView2<Float>,
    weights: &ArrayView1<Float>,
    intercept: Float,
) -> Result<Array1<Float>> {
    let (n_samples, n_features) = X.dim();

    if weights.len() != n_features {
        return Err(SklearsError::FeatureMismatch {
            expected: n_features,
            actual: weights.len(),
        });
    }

    let mut predictions = Array1::<Float>::zeros(n_samples);

    for i in 0..n_samples {
        let x_sample = X.row(i);
        predictions[i] = simd_linear_prediction(&x_sample, weights, intercept);
    }

    Ok(predictions)
}

/// Meta-feature generation
pub fn simd_generate_meta_features(
    X: &ArrayView2<Float>,
    base_weights: &ArrayView2<Float>,
    base_intercepts: &ArrayView1<Float>,
) -> Result<Array2<Float>> {
    let (n_samples, n_features) = X.dim();
    let (n_estimators, weight_features) = base_weights.dim();

    if weight_features != n_features {
        return Err(SklearsError::FeatureMismatch {
            expected: n_features,
            actual: weight_features,
        });
    }

    if base_intercepts.len() != n_estimators {
        return Err(SklearsError::InvalidInput(
            "Number of intercepts must match number of estimators".to_string(),
        ));
    }

    let mut meta_features = Array2::<Float>::zeros((n_samples, n_estimators));

    for est_idx in 0..n_estimators {
        let weights = base_weights.row(est_idx);
        let intercept = base_intercepts[est_idx];

        for sample_idx in 0..n_samples {
            let x_sample = X.row(sample_idx);
            meta_features[[sample_idx, est_idx]] =
                simd_linear_prediction(&x_sample, &weights, intercept);
        }
    }

    Ok(meta_features)
}

/// Gradient computation for meta-learner
pub fn simd_compute_gradients(
    X: &ArrayView2<Float>,
    y: &ArrayView1<Float>,
    weights: &ArrayView1<Float>,
    intercept: Float,
    l2_reg: Float,
) -> Result<(Array1<Float>, Float)> {
    let (n_samples, n_features) = X.dim();

    if y.len() != n_samples {
        return Err(SklearsError::ShapeMismatch {
            expected: format!("{} samples", n_samples),
            actual: format!("{} samples", y.len()),
        });
    }

    if weights.len() != n_features {
        return Err(SklearsError::FeatureMismatch {
            expected: n_features,
            actual: weights.len(),
        });
    }

    let mut grad_weights = Array1::<Float>::zeros(n_features);
    let mut grad_intercept = 0.0;

    for i in 0..n_samples {
        let x_i = X.row(i);
        let y_i = y[i];

        let pred = simd_linear_prediction(&x_i, weights, intercept);
        let error = pred - y_i;

        grad_intercept += error;

        for j in 0..n_features {
            grad_weights[j] += error * x_i[j];
        }
    }

    // Normalize gradients and add L2 regularization
    let n_samples_f = n_samples as Float;
    grad_intercept /= n_samples_f;

    for i in 0..n_features {
        grad_weights[i] = grad_weights[i] / n_samples_f + l2_reg * weights[i];
    }

    Ok((grad_weights, grad_intercept))
}

/// Ensemble prediction aggregation
pub fn simd_aggregate_predictions(
    base_predictions: &ArrayView2<Float>,
    meta_weights: &ArrayView1<Float>,
    meta_intercept: Float,
) -> Result<Array1<Float>> {
    let (n_samples, n_estimators) = base_predictions.dim();

    if meta_weights.len() != n_estimators {
        return Err(SklearsError::FeatureMismatch {
            expected: n_estimators,
            actual: meta_weights.len(),
        });
    }

    let mut final_predictions = Array1::<Float>::zeros(n_samples);

    for i in 0..n_samples {
        let base_preds = base_predictions.row(i);
        final_predictions[i] = simd_dot_product(&base_preds, meta_weights) + meta_intercept;
    }

    Ok(final_predictions)
}

/// Trained stacking ensemble model structure
#[derive(Debug, Clone)]
pub struct StackingEnsembleModel {
    pub base_weights: Array2<Float>,
    pub base_intercepts: Array1<Float>,
    pub meta_weights: Array1<Float>,
    pub meta_intercept: Float,
    pub n_features: usize,
    pub n_estimators: usize,
}

impl StackingEnsembleModel {
    /// Prediction using trained stacking ensemble
    pub fn predict(&self, X: &ArrayView2<Float>) -> Result<Array1<Float>> {
        let meta_features = simd_generate_meta_features(
            X,
            &self.base_weights.view(),
            &self.base_intercepts.view(),
        )?;

        simd_aggregate_predictions(
            &meta_features.view(),
            &self.meta_weights.view(),
            self.meta_intercept,
        )
    }
}

/// Simplified stacking ensemble training
pub fn simd_train_stacking_ensemble(
    X: &ArrayView2<Float>,
    y: &ArrayView1<Float>,
    n_base_estimators: usize,
    learning_rate: Float,
    l2_reg: Float,
    n_iterations: usize,
) -> Result<StackingEnsembleModel> {
    let (n_samples, n_features) = X.dim();

    if y.len() != n_samples {
        return Err(SklearsError::ShapeMismatch {
            expected: format!("{} samples", n_samples),
            actual: format!("{} samples", y.len()),
        });
    }

    // Initialize parameters
    let base_weights = Array2::<Float>::zeros((n_base_estimators, n_features));
    let base_intercepts = Array1::<Float>::zeros(n_base_estimators);
    let mut meta_weights = Array1::<Float>::zeros(n_base_estimators);
    let mut meta_intercept = 0.0;

    // Simple training loop (placeholder implementation)
    for _iter in 0..n_iterations {
        // Generate meta-features
        let meta_features =
            simd_generate_meta_features(X, &base_weights.view(), &base_intercepts.view())?;

        // Compute gradients for meta-learner
        let (grad_weights, grad_intercept) = simd_compute_gradients(
            &meta_features.view(),
            y,
            &meta_weights.view(),
            meta_intercept,
            l2_reg,
        )?;

        // Update meta-learner parameters
        for i in 0..n_base_estimators {
            meta_weights[i] -= learning_rate * grad_weights[i];
        }
        meta_intercept -= learning_rate * grad_intercept;
    }

    Ok(StackingEnsembleModel {
        base_weights,
        base_intercepts,
        meta_weights,
        meta_intercept,
        n_features,
        n_estimators: n_base_estimators,
    })
}

// Additional utility functions for completeness

/// Calculate mean of array
fn simd_mean(arr: &ArrayView1<Float>) -> Float {
    if arr.is_empty() {
        return 0.0;
    }
    arr.sum() / arr.len() as Float
}

/// Calculate variance of array
fn simd_variance(arr: &ArrayView1<Float>, mean: Float) -> Float {
    if arr.len() < 2 {
        return 0.0;
    }
    let sum_sq_diff: Float = arr.iter().map(|&x| (x - mean).powi(2)).sum();
    sum_sq_diff / (arr.len() - 1) as Float
}

/// Ensemble diversity measurement
pub fn simd_compute_ensemble_diversity(predictions: &ArrayView2<Float>) -> Result<Float> {
    let (n_samples, n_estimators) = predictions.dim();

    if n_estimators < 2 {
        return Ok(0.0);
    }

    let mut total_diversity = 0.0;
    let mut pair_count = 0;

    // Compute pairwise diversity
    for i in 0..n_estimators {
        for j in i + 1..n_estimators {
            let pred_i = predictions.column(i);
            let pred_j = predictions.column(j);

            let correlation = simd_correlation_coefficient(&pred_i, &pred_j);
            let diversity = 1.0 - correlation.abs();
            total_diversity += diversity;
            pair_count += 1;
        }
    }

    Ok(total_diversity / pair_count as Float)
}

/// Correlation coefficient computation
fn simd_correlation_coefficient(x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> Float {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }

    let mean_x = simd_mean(x);
    let mean_y = simd_mean(y);

    let mut sum_xy = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_yy = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        sum_xy += dx * dy;
        sum_xx += dx * dx;
        sum_yy += dy * dy;
    }

    let denominator = (sum_xx * sum_yy).sqrt();
    if denominator > 1e-12 {
        sum_xy / denominator
    } else {
        0.0
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_simd_dot_product() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let w = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);

        let result = simd_dot_product(&x.view(), &w.view());
        let expected = 1.0 * 0.1 + 2.0 * 0.2 + 3.0 * 0.3 + 4.0 * 0.4;

        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_simd_linear_prediction() {
        let x = Array1::from_vec(vec![2.0, 3.0]);
        let w = Array1::from_vec(vec![0.5, 0.3]);
        let intercept = 1.5;

        let result = simd_linear_prediction(&x.view(), &w.view(), intercept);
        let expected = 2.0 * 0.5 + 3.0 * 0.3 + 1.5;

        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_simd_mean() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = simd_mean(&data.view());
        assert!((result - 3.0).abs() < 1e-10);
    }
}
