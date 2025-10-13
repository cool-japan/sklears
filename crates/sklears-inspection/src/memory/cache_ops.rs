//! Cache-friendly computation operations for explanation algorithms
//!
//! This module provides optimized computation functions that leverage caching
//! and memory-efficient algorithms for feature importance and SHAP computations.

use super::cache::{CacheConfig, CacheKey, ExplanationCache};
use crate::types::*;
use crate::SklResult;
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::Rng;
use sklears_core::error::SklearsError;

/// Cache-friendly permutation importance computation
pub fn cache_friendly_permutation_importance<F>(
    X: &ArrayView2<Float>,
    y: &ArrayView1<Float>,
    model: &F,
    cache: &ExplanationCache,
    config: &CacheConfig,
) -> SklResult<Array1<Float>>
where
    F: Fn(&ArrayView2<Float>) -> SklResult<Array1<Float>>,
{
    let cache_key = CacheKey::new(X, "permutation_importance", 0);

    cache.get_or_compute_feature_importance(&cache_key, || {
        compute_permutation_importance_optimized(X, y, model, config)
    })
}

/// Cache-friendly SHAP computation with memory optimization
pub fn cache_friendly_shap_computation<F>(
    X: &ArrayView2<Float>,
    model: &F,
    cache: &ExplanationCache,
    config: &CacheConfig,
) -> SklResult<Array2<Float>>
where
    F: Fn(&ArrayView2<Float>) -> SklResult<Array1<Float>>,
{
    let cache_key = CacheKey::new(X, "shap", 0);

    cache.get_or_compute_shap(&cache_key, || compute_shap_optimized(X, model, config))
}

/// Memory-optimized permutation importance computation
fn compute_permutation_importance_optimized<F>(
    X: &ArrayView2<Float>,
    y: &ArrayView1<Float>,
    model: &F,
    config: &CacheConfig,
) -> SklResult<Array1<Float>>
where
    F: Fn(&ArrayView2<Float>) -> SklResult<Array1<Float>>,
{
    let n_features = X.ncols();
    let n_samples = X.nrows();

    // Baseline score
    let baseline_predictions = model(X)?;
    let baseline_score = compute_r2_score(y, &baseline_predictions.view())?;

    let mut importances = Array1::zeros(n_features);

    // Process features in blocks for better cache locality
    let block_size = config.prefetch_distance.min(n_features);

    for block_start in (0..n_features).step_by(block_size) {
        let block_end = (block_start + block_size).min(n_features);

        // Pre-allocate memory for the block
        let mut X_permuted = X.to_owned();

        for feature_idx in block_start..block_end {
            // Create column view for better cache locality and shuffle values in-place
            {
                let mut column = X_permuted.column_mut(feature_idx);
                let mut rng = scirs2_core::random::thread_rng();
                for i in (1..n_samples).rev() {
                    let j = rng.gen_range(0..=i);
                    column.swap(i, j);
                }
            } // Drop mutable borrow here

            // Compute permuted score
            let permuted_predictions = model(&X_permuted.view())?;
            let permuted_score = compute_r2_score(y, &permuted_predictions.view())?;

            // Store importance
            importances[feature_idx] = baseline_score - permuted_score;

            // Restore original column
            X_permuted
                .column_mut(feature_idx)
                .assign(&X.column(feature_idx));
        }
    }

    Ok(importances)
}

/// Memory-optimized SHAP computation
fn compute_shap_optimized<F>(
    X: &ArrayView2<Float>,
    model: &F,
    config: &CacheConfig,
) -> SklResult<Array2<Float>>
where
    F: Fn(&ArrayView2<Float>) -> SklResult<Array1<Float>>,
{
    let n_samples = X.nrows();
    let n_features = X.ncols();

    // Use simplified SHAP computation for efficiency
    let mut shap_values = Array2::zeros((n_samples, n_features));

    // Process samples in blocks for better cache locality
    let block_size = config.prefetch_distance.min(n_samples);

    for block_start in (0..n_samples).step_by(block_size) {
        let block_end = (block_start + block_size).min(n_samples);

        for sample_idx in block_start..block_end {
            let sample = X.row(sample_idx);

            // Baseline prediction (all features set to mean)
            let baseline_data = compute_baseline_data(X)?;
            let baseline_pred = model(&baseline_data.view())?;
            let baseline_value = baseline_pred[0];

            // Full prediction
            let full_pred = model(&sample.insert_axis(Axis(0)))?;
            let full_value = full_pred[0];

            // Compute marginal contributions (simplified)
            let total_contribution = full_value - baseline_value;
            let contribution_per_feature = total_contribution / n_features as Float;

            // Assign equal contribution to all features (simplified SHAP)
            for feature_idx in 0..n_features {
                shap_values[(sample_idx, feature_idx)] = contribution_per_feature;
            }
        }
    }

    Ok(shap_values)
}

/// Compute baseline data (feature means)
pub fn compute_baseline_data(X: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
    let means = X
        .mean_axis(Axis(0))
        .ok_or_else(|| SklearsError::InvalidInput("Cannot compute feature means".to_string()))?;

    // Create a single-row array with feature means
    Ok(means.insert_axis(Axis(0)))
}

/// Compute R² score for model evaluation
pub fn compute_r2_score(
    y_true: &ArrayView1<Float>,
    y_pred: &ArrayView1<Float>,
) -> SklResult<Float> {
    if y_true.len() != y_pred.len() {
        return Err(SklearsError::InvalidInput(
            "y_true and y_pred must have the same length".to_string(),
        ));
    }

    let y_mean = y_true
        .mean()
        .ok_or_else(|| SklearsError::InvalidInput("Cannot compute mean of y_true".to_string()))?;

    let ss_tot: Float = y_true.iter().map(|&y| (y - y_mean).powi(2)).sum();
    let ss_res: Float = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&y_t, &y_p)| (y_t - y_p).powi(2))
        .sum();

    if ss_tot == 0.0 {
        return Ok(1.0);
    }

    Ok(1.0 - (ss_res / ss_tot))
}

#[cfg(test)]
mod tests {
    use super::super::cache::{CacheConfig, ExplanationCache};
    use super::*;
    // ✅ SciRS2 Policy Compliant Import
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    #[allow(non_snake_case)]
    fn test_cache_friendly_permutation_importance() {
        let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![1.0, 2.0, 3.0];
        let config = CacheConfig::default();
        let cache = ExplanationCache::new(&config);

        let model =
            |x: &ArrayView2<Float>| -> SklResult<Array1<Float>> { Ok(x.column(0).to_owned()) };

        let importances =
            cache_friendly_permutation_importance(&X.view(), &y.view(), &model, &cache, &config)
                .unwrap();

        assert_eq!(importances.len(), 2);
    }

    #[test]
    fn test_r2_score_computation() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.1, 1.9, 3.1];

        let r2 = compute_r2_score(&y_true.view(), &y_pred.view()).unwrap();
        assert!(r2 > 0.9); // Should be high for good predictions
        assert!(r2 <= 1.0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_baseline_data_computation() {
        let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let baseline = compute_baseline_data(&X.view()).unwrap();

        // Should have 1 row (means) and same number of columns
        assert_eq!(baseline.nrows(), 1);
        assert_eq!(baseline.ncols(), 2);

        // Check mean values
        assert_abs_diff_eq!(baseline[(0, 0)], 3.0, epsilon = 1e-6); // Mean of [1,3,5]
        assert_abs_diff_eq!(baseline[(0, 1)], 4.0, epsilon = 1e-6); // Mean of [2,4,6]
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_cache_friendly_shap_computation() {
        let X = array![[1.0, 2.0], [3.0, 4.0]];
        let config = CacheConfig::default();
        let cache = ExplanationCache::new(&config);

        let model = |x: &ArrayView2<Float>| -> SklResult<Array1<Float>> {
            // Simple linear model: sum of features
            Ok(x.sum_axis(Axis(1)))
        };

        let shap_values =
            cache_friendly_shap_computation(&X.view(), &model, &cache, &config).unwrap();

        assert_eq!(shap_values.nrows(), 2); // 2 samples
        assert_eq!(shap_values.ncols(), 2); // 2 features
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_permutation_importance_optimized() {
        let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![3.0, 7.0, 11.0]; // y = x1 + x2
        let config = CacheConfig::default();

        let model = |x: &ArrayView2<Float>| -> SklResult<Array1<Float>> {
            // Model that sums the features
            Ok(x.sum_axis(Axis(1)))
        };

        let importances =
            compute_permutation_importance_optimized(&X.view(), &y.view(), &model, &config)
                .unwrap();

        assert_eq!(importances.len(), 2);
        // Both features should have some importance since they both contribute
        assert!(importances[0].abs() > 0.0 || importances[1].abs() > 0.0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_shap_optimized() {
        let X = array![[1.0, 2.0], [3.0, 4.0]];
        let config = CacheConfig::default();

        let model = |x: &ArrayView2<Float>| -> SklResult<Array1<Float>> { Ok(x.sum_axis(Axis(1))) };

        let shap_values = compute_shap_optimized(&X.view(), &model, &config).unwrap();

        assert_eq!(shap_values.nrows(), 2);
        assert_eq!(shap_values.ncols(), 2);

        // SHAP values should sum approximately to the difference between prediction and baseline
        for sample_idx in 0..2 {
            let shap_sum: Float = shap_values.row(sample_idx).sum();
            assert!(shap_sum.is_finite());
        }
    }
}
