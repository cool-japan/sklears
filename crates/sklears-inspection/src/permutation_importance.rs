//! Permutation Feature Importance Analysis
//!
//! This module provides permutation feature importance calculation, a model-agnostic
//! technique for measuring feature importance by observing the decrease in model
//! performance when feature values are randomly shuffled.

// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{Array1, ArrayView1, ArrayView2};
use scirs2_core::random::{seq::SliceRandom, SeedableRng};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    types::Float,
};

use crate::inspection_types::{ScoreFunction, PermutationImportanceResult};
use crate::inspection_utils::{accuracy_score, r2_score, mean_squared_error};

/// Permutation Feature Importance
///
/// Permutation feature importance is a model inspection technique that calculates
/// the importance of a feature by measuring the decrease in a model score when a
/// single feature value is randomly shuffled.
///
/// This approach is model-agnostic and can be applied to any fitted model that
/// provides predictions. The importance is calculated as the difference between
/// the baseline model performance and the performance when a feature is shuffled.
///
/// # Parameters
///
/// * `predict_fn` - Function that takes feature matrix and returns predictions
/// * `X` - Feature matrix of shape (n_samples, n_features)
/// * `y` - Target values of shape (n_samples,)
/// * `scoring` - Scoring function to use for evaluation
/// * `n_repeats` - Number of times to shuffle each feature (more repeats = more stable results)
/// * `random_state` - Random seed for reproducible results
///
/// # Returns
///
/// Returns a `PermutationImportanceResult` containing:
/// - `importances`: Raw importance scores for each repetition
/// - `importances_mean`: Mean importance score for each feature
/// - `importances_std`: Standard deviation of importance scores for each feature
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_inspection::permutation_importance;
/// // ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::array;
///
/// // Mock predictor function (would normally be a fitted model)
/// let predict_fn = |x: &ArrayView2<f64>| -> Vec<f64> {
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
///
/// # Algorithm
///
/// 1. Calculate baseline model performance on original data
/// 2. For each feature:
///    - Repeat n_repeats times:
///      - Create copy of data with target feature shuffled
///      - Calculate model performance on shuffled data
///      - Record importance as (baseline_score - shuffled_score)
/// 3. Calculate mean and standard deviation of importance scores
///
/// # Notes
///
/// - Higher importance scores indicate more important features
/// - Negative importance scores may indicate features that help the model generalize
/// - The method is computationally expensive as it requires n_features × n_repeats predictions
/// - Results can vary based on the random shuffle, hence the importance of n_repeats
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

    if n_features == 0 {
        return Err(SklearsError::InvalidInput(
            "X must have at least one feature".to_string(),
        ));
    }

    if n_repeats == 0 {
        return Err(SklearsError::InvalidInput(
            "n_repeats must be greater than 0".to_string(),
        ));
    }

    let mut rng = match random_state {
        Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed),
        None => scirs2_core::random::rngs::StdRng::from_rng(scirs2_core::random::thread_rng())
            .map_err(|e| SklearsError::InvalidInput(format!("RNG initialization failed: {}", e)))?,
    };

    // Compute baseline score
    let baseline_predictions = predict_fn(X);
    if baseline_predictions.len() != n_samples {
        return Err(SklearsError::InvalidInput(
            "predict_fn must return predictions for all samples".to_string(),
        ));
    }

    let baseline_score = match scoring {
        ScoreFunction::Accuracy => accuracy_score(y, &baseline_predictions)?,
        ScoreFunction::R2 => r2_score(y, &baseline_predictions)?,
        ScoreFunction::MeanSquaredError => -mean_squared_error(y, &baseline_predictions)?,
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
            if permuted_predictions.len() != n_samples {
                return Err(SklearsError::InvalidInput(
                    "predict_fn must return predictions for all samples".to_string(),
                ));
            }

            let permuted_score = match scoring {
                ScoreFunction::Accuracy => accuracy_score(y, &permuted_predictions)?,
                ScoreFunction::R2 => r2_score(y, &permuted_predictions)?,
                ScoreFunction::MeanSquaredError => -mean_squared_error(y, &permuted_predictions)?,
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
        if importances.is_empty() {
            continue;
        }

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

/// Advanced permutation importance with additional options
///
/// This function provides additional options for permutation importance calculation,
/// including the ability to compute importance for groups of features and to use
/// custom scoring functions.
pub fn permutation_importance_advanced<F, S>(
    predict_fn: &F,
    X: &ArrayView2<Float>,
    y: &ArrayView1<Float>,
    scoring_fn: &S,
    feature_groups: Option<&[Vec<usize>]>,
    n_repeats: usize,
    random_state: Option<u64>,
) -> SklResult<PermutationImportanceResult>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
    S: Fn(&ArrayView1<Float>, &[Float]) -> Float,
{
    let (n_samples, n_features) = X.dim();

    if n_samples != y.len() {
        return Err(SklearsError::InvalidInput(
            "X and y must have the same number of samples".to_string(),
        ));
    }

    let groups = match feature_groups {
        Some(groups) => groups.to_vec(),
        None => (0..n_features).map(|i| vec![i]).collect(),
    };

    let mut rng = match random_state {
        Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed),
        None => scirs2_core::random::rngs::StdRng::from_rng(scirs2_core::random::thread_rng())
            .map_err(|e| SklearsError::InvalidInput(format!("RNG initialization failed: {}", e)))?,
    };

    // Compute baseline score
    let baseline_predictions = predict_fn(X);
    if baseline_predictions.len() != n_samples {
        return Err(SklearsError::InvalidInput(
            "predict_fn must return predictions for all samples".to_string(),
        ));
    }

    let baseline_score = scoring_fn(y, &baseline_predictions);

    let mut feature_importances: Vec<Vec<Float>> = vec![Vec::new(); groups.len()];

    // For each feature group
    for (group_idx, feature_group) in groups.iter().enumerate() {
        // Repeat permutation n_repeats times
        for _ in 0..n_repeats {
            // Create a copy of X with the feature group shuffled
            let mut X_permuted = X.to_owned();

            for &feature_idx in feature_group {
                if feature_idx >= n_features {
                    return Err(SklearsError::InvalidInput(format!(
                        "Feature index {} exceeds number of features {}",
                        feature_idx, n_features
                    )));
                }

                let mut feature_values: Vec<Float> = X_permuted.column(feature_idx).to_vec();
                feature_values.shuffle(&mut rng);

                for (i, &val) in feature_values.iter().enumerate() {
                    X_permuted[[i, feature_idx]] = val;
                }
            }

            // Compute score with permuted features
            let permuted_predictions = predict_fn(&X_permuted.view());
            if permuted_predictions.len() != n_samples {
                return Err(SklearsError::InvalidInput(
                    "predict_fn must return predictions for all samples".to_string(),
                ));
            }

            let permuted_score = scoring_fn(y, &permuted_predictions);

            // Importance is the decrease in score
            let importance = baseline_score - permuted_score;
            feature_importances[group_idx].push(importance);
        }
    }

    // Compute statistics
    let mut importances_mean = Array1::zeros(groups.len());
    let mut importances_std = Array1::zeros(groups.len());

    for (group_idx, importances) in feature_importances.iter().enumerate() {
        if importances.is_empty() {
            continue;
        }

        let mean = importances.iter().sum::<Float>() / importances.len() as Float;
        let variance = importances
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<Float>()
            / importances.len() as Float;
        let std = variance.sqrt();

        importances_mean[group_idx] = mean;
        importances_std[group_idx] = std;
    }

    Ok(PermutationImportanceResult {
        importances: feature_importances,
        importances_mean,
        importances_std,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    // ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::array;
    use crate::inspection_types::ScoreFunction;

    #[test]
    fn test_permutation_importance_basic() {
        // Simple linear predictor: prediction = sum of features
        let predict_fn = |x: &ArrayView2<f64>| -> Vec<f64> {
            x.rows().into_iter()
                .map(|row| row.iter().sum())
                .collect()
        };

        let X = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let y = array![6.0, 15.0, 24.0];

        let result = permutation_importance(
            &predict_fn,
            &X.view(),
            &y.view(),
            ScoreFunction::R2,
            3,
            Some(42),
        ).unwrap();

        assert_eq!(result.importances_mean.len(), 3);
        assert_eq!(result.importances_std.len(), 3);
        assert_eq!(result.importances.len(), 3);

        // Each feature should have 3 importance scores (n_repeats = 3)
        for feature_importances in &result.importances {
            assert_eq!(feature_importances.len(), 3);
        }
    }

    #[test]
    fn test_permutation_importance_validation() {
        let predict_fn = |x: &ArrayView2<f64>| -> Vec<f64> {
            vec![1.0; x.nrows()]
        };

        let X = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 2.0, 3.0]; // Wrong size

        let result = permutation_importance(
            &predict_fn,
            &X.view(),
            &y.view(),
            ScoreFunction::R2,
            1,
            Some(42),
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_permutation_importance_feature_effectiveness() {
        // Model where only the first feature matters
        let predict_fn = |x: &ArrayView2<f64>| -> Vec<f64> {
            x.rows().into_iter()
                .map(|row| row[0]) // Only first feature
                .collect()
        };

        let X = array![
            [1.0, 100.0, 200.0],
            [2.0, 100.0, 200.0],
            [3.0, 100.0, 200.0]
        ];
        let y = array![1.0, 2.0, 3.0];

        let result = permutation_importance(
            &predict_fn,
            &X.view(),
            &y.view(),
            ScoreFunction::R2,
            5,
            Some(42),
        ).unwrap();

        // First feature should have higher importance than others
        assert!(result.importances_mean[0] > result.importances_mean[1]);
        assert!(result.importances_mean[0] > result.importances_mean[2]);
    }

    #[test]
    fn test_permutation_importance_advanced() {
        let predict_fn = |x: &ArrayView2<f64>| -> Vec<f64> {
            x.rows().into_iter()
                .map(|row| row.iter().sum())
                .collect()
        };

        let scoring_fn = |y_true: &ArrayView1<f64>, y_pred: &[f64]| -> f64 {
            // Simple MSE
            y_true.iter()
                .zip(y_pred.iter())
                .map(|(true_val, pred_val)| (true_val - pred_val).powi(2))
                .sum::<f64>() / y_true.len() as f64
        };

        let X = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let y = array![6.0, 15.0];

        let feature_groups = vec![vec![0], vec![1, 2]];

        let result = permutation_importance_advanced(
            &predict_fn,
            &X.view(),
            &y.view(),
            &scoring_fn,
            Some(&feature_groups),
            3,
            Some(42),
        ).unwrap();

        assert_eq!(result.importances_mean.len(), 2); // Two groups
    }
}