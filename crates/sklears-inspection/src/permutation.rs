//! Permutation importance calculation

// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{Array1, ArrayView1, ArrayView2};
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::random::seq::SliceRandom;
use scirs2_core::random::SeedableRng;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    types::Float,
};

use crate::types::{PermutationImportanceResult, ScoreFunction};

/// Permutation Importance
///
/// Permutation feature importance is a model inspection technique that calculates
/// the importance of a feature by measuring the decrease in a model score when a
/// single feature value is randomly shuffled.
///
/// # Parameters
///
/// * `scoring` - Scoring function to use (e.g., accuracy for classification, R² for regression)
/// * `n_repeats` - Number of times to permute each feature
/// * `random_state` - Random state for reproducibility
/// * `n_jobs` - Number of parallel jobs (not implemented in this basic version)
///
/// # Examples
///
/// ```
/// use sklears_inspection::{permutation_importance, ScoreFunction};
/// use scirs2_core::ndarray::array;
///
/// // Mock predictor function (would normally be a fitted model)
/// let predict_fn = |x: &scirs2_core::ndarray::ArrayView2<f64>| -> Vec<f64> {
///     x.rows().into_iter()
///         .map(|row| row.iter().sum())
///         .collect()
/// };
///
/// let X = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
/// let y = array![6.0, 15.0, 24.0];
///
/// let result = permutation_importance(
///     &predict_fn,
///     &X.view(),
///     &y.view(),
///     ScoreFunction::R2,
///     5,
///     Some(42),
/// ).unwrap();
///
/// assert_eq!(result.importances_mean.len(), 3);
/// ```
pub fn permutation_importance<F>(
    predict_fn: &F,
    X: &ArrayView2<Float>,
    y: &ArrayView1<Float>,
    scoring: ScoreFunction,
    n_repeats: usize,
    random_state: Option<u64>,
) -> SklResult<PermutationImportanceResult>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    let (n_samples, n_features) = X.dim();

    if n_samples != y.len() {
        return Err(SklearsError::InvalidInput(
            "X and y must have the same number of samples".to_string(),
        ));
    }

    let mut rng = match random_state {
        Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed),
        None => scirs2_core::random::rngs::StdRng::seed_from_u64(42),
    };

    // Compute baseline score
    let baseline_predictions = predict_fn(X);
    let baseline_score = match scoring {
        ScoreFunction::Accuracy => accuracy_score(y, &baseline_predictions),
        ScoreFunction::R2 => r2_score(y, &baseline_predictions),
        ScoreFunction::MeanSquaredError => -mean_squared_error(y, &baseline_predictions),
    };

    let mut feature_importances: Vec<Vec<Float>> = vec![Vec::new(); n_features];

    // For each feature
    for feature_idx in 0..n_features {
        // Repeat permutation n_repeats times
        for _ in 0..n_repeats {
            // Create a copy of X with the feature shuffled
            let mut X_permuted = X.to_owned();
            let mut feature_values: Vec<Float> = X_permuted.column(feature_idx).to_vec();
            feature_values.shuffle(&mut rng);

            for (i, &val) in feature_values.iter().enumerate() {
                X_permuted[[i, feature_idx]] = val;
            }

            // Compute score with permuted feature
            let permuted_predictions = predict_fn(&X_permuted.view());
            let permuted_score = match scoring {
                ScoreFunction::Accuracy => accuracy_score(y, &permuted_predictions),
                ScoreFunction::R2 => r2_score(y, &permuted_predictions),
                ScoreFunction::MeanSquaredError => -mean_squared_error(y, &permuted_predictions),
            };

            // Importance is the decrease in score
            let importance = baseline_score - permuted_score;
            feature_importances[feature_idx].push(importance);
        }
    }

    // Compute statistics
    let mut importances_mean = Array1::zeros(n_features);
    let mut importances_std = Array1::zeros(n_features);

    for (feature_idx, importances) in feature_importances.iter().enumerate() {
        let mean = importances.iter().sum::<Float>() / importances.len() as Float;
        let variance = importances
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<Float>()
            / importances.len() as Float;
        let std = variance.sqrt();

        importances_mean[feature_idx] = mean;
        importances_std[feature_idx] = std;
    }

    Ok(PermutationImportanceResult {
        importances: feature_importances,
        importances_mean,
        importances_std,
    })
}

/// Accuracy score (classification)
fn accuracy_score(y_true: &ArrayView1<Float>, y_pred: &[Float]) -> Float {
    if y_true.len() != y_pred.len() {
        return 0.0;
    }

    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(&true_val, &pred_val)| (true_val - pred_val).abs() < 1e-10)
        .count();

    correct as Float / y_true.len() as Float
}

/// R² score (regression)  
fn r2_score(y_true: &ArrayView1<Float>, y_pred: &[Float]) -> Float {
    if y_true.len() != y_pred.len() {
        return 0.0;
    }

    let y_mean = y_true.mean().unwrap_or(0.0);

    let ss_res: Float = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&true_val, &pred_val)| (true_val - pred_val).powi(2))
        .sum();

    let ss_tot: Float = y_true
        .iter()
        .map(|&true_val| (true_val - y_mean).powi(2))
        .sum();

    if ss_tot == 0.0 {
        return 1.0; // Perfect prediction when no variance
    }

    1.0 - (ss_res / ss_tot)
}

/// Mean squared error (regression)
fn mean_squared_error(y_true: &ArrayView1<Float>, y_pred: &[Float]) -> Float {
    if y_true.len() != y_pred.len() {
        return Float::INFINITY;
    }

    let mse: Float = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&true_val, &pred_val)| (true_val - pred_val).powi(2))
        .sum();

    mse / y_true.len() as Float
}

/// Helper function to compute score based on scoring function
pub fn compute_score(
    y_true: &ArrayView1<Float>,
    y_pred: &[Float],
    scoring: ScoreFunction,
) -> Float {
    match scoring {
        ScoreFunction::Accuracy => accuracy_score(y_true, y_pred),
        ScoreFunction::R2 => r2_score(y_true, y_pred),
        ScoreFunction::MeanSquaredError => -mean_squared_error(y_true, y_pred),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::{array, ArrayView1, ArrayView2};

    #[test]
    fn test_accuracy_score() {
        let y_true = array![1.0, 0.0, 1.0, 1.0];
        let y_pred = vec![1.0, 0.0, 0.0, 1.0];

        let accuracy = accuracy_score(&y_true.view(), &y_pred);
        assert!((accuracy - 0.75).abs() < 1e-10); // 3 out of 4 correct
    }

    #[test]
    fn test_r2_score() {
        let y_true = array![3.0, -0.5, 2.0, 7.0];
        let y_pred = vec![2.5, 0.0, 2.0, 8.0];

        let r2 = r2_score(&y_true.view(), &y_pred);
        assert!(r2 > 0.0 && r2 <= 1.0); // R² should be reasonable
    }

    #[test]
    fn test_perfect_predictions() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = vec![1.0, 2.0, 3.0];

        let accuracy = accuracy_score(&y_true.view(), &y_pred);
        assert_eq!(accuracy, 1.0);

        let r2 = r2_score(&y_true.view(), &y_pred);
        assert_eq!(r2, 1.0);

        let mse = mean_squared_error(&y_true.view(), &y_pred);
        assert_eq!(mse, 0.0);
    }

    #[test]
    fn test_mismatched_lengths() {
        let y_true = array![1.0, 2.0];
        let y_pred = vec![1.0]; // Wrong length

        let accuracy = accuracy_score(&y_true.view(), &y_pred);
        assert_eq!(accuracy, 0.0);

        let r2 = r2_score(&y_true.view(), &y_pred);
        assert_eq!(r2, 0.0);

        let mse = mean_squared_error(&y_true.view(), &y_pred);
        assert_eq!(mse, Float::INFINITY);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_permutation_importance() {
        // Simple additive model: prediction = sum of features
        let predict_fn = |x: &ArrayView2<Float>| -> Vec<Float> {
            x.rows().into_iter().map(|row| row.iter().sum()).collect()
        };

        let X = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let y = array![6.0, 15.0, 24.0]; // Perfect predictions

        let result = permutation_importance(
            &predict_fn,
            &X.view(),
            &y.view(),
            ScoreFunction::R2,
            5,
            Some(42),
        )
        .unwrap();

        assert_eq!(result.importances_mean.len(), 3);
        assert_eq!(result.importances_std.len(), 3);
        assert_eq!(result.importances.len(), 3);

        // All features should have some importance since they all contribute equally
        for importance in result.importances_mean.iter() {
            assert!(*importance > 0.0);
        }
    }
}
