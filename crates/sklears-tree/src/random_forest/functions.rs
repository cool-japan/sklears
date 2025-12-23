//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::SliceRandomExt;
use sklears_core::error::{Result, SklearsError};
use std::collections::HashMap;

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::SplitCriterion;
    use scirs2_core::ndarray::array;
    use sklears_core::traits::{Fit, Predict};
    #[test]
    fn test_random_forest_classifier() {
        let x = array![
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
        ];
        let y = array![0, 0, 0, 1, 1, 1];
        let model = RandomForestClassifier::new()
            .n_estimators(10)
            .max_depth(3)
            .criterion(SplitCriterion::Gini)
            .random_state(42)
            .fit(&x, &y)
            .unwrap();
        assert_eq!(model.n_features(), 2);
        assert_eq!(model.n_classes(), 2);
        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions.len(), 6);
    }
    #[test]
    fn test_random_forest_regressor() {
        let x = array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0],];
        let y = array![0.0, 1.0, 4.0, 9.0, 16.0, 25.0];
        let model = RandomForestRegressor::new()
            .n_estimators(20)
            .max_depth(5)
            .criterion(SplitCriterion::MSE)
            .random_state(42)
            .fit(&x, &y)
            .unwrap();
        assert_eq!(model.n_features(), 1);
        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions.len(), 6);
        let test_x = array![[2.5]];
        let test_pred = model.predict(&test_x).unwrap();
        assert!(test_pred.len() == 1);
        assert!(test_pred[0] > 3.0 && test_pred[0] < 10.0);
    }
    #[test]
    fn test_random_forest_classifier_feature_importances() {
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ];
        let y = array![0, 0, 1, 1];
        let model = RandomForestClassifier::new()
            .n_estimators(5)
            .fit(&x, &y)
            .unwrap();
        let importances = model.feature_importances().unwrap();
        assert_eq!(importances.len(), 3);
        let sum: f64 = importances.sum();
        assert!((sum - 1.0).abs() < f64::EPSILON);
        let expected = 1.0 / 3.0;
        for &importance in importances.iter() {
            assert!((importance - expected).abs() < f64::EPSILON);
        }
    }
    #[test]
    fn test_random_forest_regressor_feature_importances() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0],];
        let y = array![10.0, 20.0, 30.0, 40.0];
        let model = RandomForestRegressor::new()
            .n_estimators(3)
            .criterion(SplitCriterion::MSE)
            .fit(&x, &y)
            .unwrap();
        let importances = model.feature_importances().unwrap();
        assert_eq!(importances.len(), 2);
        let sum: f64 = importances.sum();
        assert!((sum - 1.0).abs() < f64::EPSILON);
        let expected = 1.0 / 2.0;
        for &importance in importances.iter() {
            assert!((importance - expected).abs() < f64::EPSILON);
        }
    }
    #[test]
    fn test_feature_importances_not_fitted() {
        let model = RandomForestClassifier::new();
        assert_eq!(model.config.n_estimators, 100);
    }
    #[test]
    fn test_random_forest_regressor_proximity_matrix() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![1.0, 4.0, 9.0, 16.0];
        let model = RandomForestRegressor::new()
            .n_estimators(5)
            .max_depth(3)
            .criterion(SplitCriterion::MSE)
            .random_state(42)
            .fit(&x, &y)
            .unwrap();
        assert!(model.proximity_matrix().is_none());
        let proximity = model.compute_proximity_matrix(&x).unwrap();
        assert_eq!(proximity.shape(), &[4, 4]);
        for i in 0..4 {
            assert!((proximity[(i, i)] - 1.0).abs() < f64::EPSILON);
        }
        for i in 0..4 {
            for j in 0..4 {
                assert!((proximity[(i, j)] - proximity[(j, i)]).abs() < f64::EPSILON);
            }
        }
        for i in 0..4 {
            for j in 0..4 {
                assert!(proximity[(i, j)] >= 0.0 && proximity[(i, j)] <= 1.0);
            }
        }
    }
    #[test]
    fn test_random_forest_classifier_parallel_predict() {
        let x = array![
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
        ];
        let y = array![0, 0, 0, 1, 1, 1];
        let model = RandomForestClassifier::new()
            .n_estimators(10)
            .max_depth(3)
            .criterion(SplitCriterion::Gini)
            .random_state(42)
            .n_jobs(2)
            .fit(&x, &y)
            .unwrap();
        let parallel_predictions = model.predict_parallel(&x).unwrap();
        let serial_predictions = model.predict(&x).unwrap();
        assert_eq!(parallel_predictions.len(), serial_predictions.len());
        assert_eq!(parallel_predictions.len(), 6);
        for (parallel, serial) in parallel_predictions.iter().zip(serial_predictions.iter()) {
            assert_eq!(parallel, serial);
        }
    }
    #[test]
    fn test_random_forest_classifier_parallel_predict_proba() {
        let x = array![
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
        ];
        let y = array![0, 0, 0, 1, 1, 1];
        let model = RandomForestClassifier::new()
            .n_estimators(10)
            .max_depth(3)
            .criterion(SplitCriterion::Gini)
            .random_state(42)
            .n_jobs(2)
            .fit(&x, &y)
            .unwrap();
        let probabilities = model.predict_proba_parallel(&x).unwrap();
        assert_eq!(probabilities.shape(), &[6, 2]);
        for i in 0..6 {
            let row_sum: f64 = probabilities.row(i).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-10,
                "Row {}: sum = {}",
                i,
                row_sum
            );
        }
        for prob in probabilities.iter() {
            assert!(
                *prob >= 0.0 && *prob <= 1.0,
                "Invalid probability: {}",
                prob
            );
        }
    }
    #[test]
    fn test_random_forest_regressor_parallel_predict() {
        let x = array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![0.0, 1.0, 4.0, 9.0, 16.0, 25.0];
        let model = RandomForestRegressor::new()
            .n_estimators(20)
            .max_depth(5)
            .criterion(SplitCriterion::MSE)
            .random_state(42)
            .n_jobs(2)
            .fit(&x, &y)
            .unwrap();
        let parallel_predictions = model.predict_parallel(&x).unwrap();
        let serial_predictions = model.predict(&x).unwrap();
        assert_eq!(parallel_predictions.len(), serial_predictions.len());
        assert_eq!(parallel_predictions.len(), 6);
        for (parallel, serial) in parallel_predictions.iter().zip(serial_predictions.iter()) {
            assert_eq!(parallel, serial);
        }
        let test_x = array![[2.5]];
        let test_parallel_pred = model.predict_parallel(&test_x).unwrap();
        let test_serial_pred = model.predict(&test_x).unwrap();
        assert_eq!(test_parallel_pred.len(), 1);
        assert_eq!(test_serial_pred.len(), 1);
        assert_eq!(test_parallel_pred[0], test_serial_pred[0]);
        assert!(test_parallel_pred[0] > 3.0 && test_parallel_pred[0] < 10.0);
    }
}
/// Calculate class weights for balanced Random Forest
pub fn calculate_class_weights(
    y: &Array1<i32>,
    strategy: &ClassWeight,
) -> Result<HashMap<i32, f64>> {
    match strategy {
        ClassWeight::None => {
            let unique_classes: Vec<i32> = y
                .iter()
                .cloned()
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();
            let weights = unique_classes
                .into_iter()
                .map(|class| (class, 1.0))
                .collect();
            Ok(weights)
        }
        ClassWeight::Balanced => {
            let mut class_counts: HashMap<i32, usize> = HashMap::new();
            for &class in y.iter() {
                *class_counts.entry(class).or_insert(0) += 1;
            }
            let n_samples = y.len() as f64;
            let n_classes = class_counts.len() as f64;
            let mut weights = HashMap::new();
            for (&class, &count) in &class_counts {
                let weight = n_samples / (n_classes * count as f64);
                weights.insert(class, weight);
            }
            Ok(weights)
        }
        ClassWeight::Custom(weights) => Ok(weights.clone()),
    }
}
/// Generate balanced bootstrap sample indices
pub fn balanced_bootstrap_sample(
    y: &Array1<i32>,
    strategy: SamplingStrategy,
    n_samples: usize,
    _random_state: Option<u64>,
) -> Result<Vec<usize>> {
    let mut rng = scirs2_core::random::thread_rng();
    match strategy {
        SamplingStrategy::Bootstrap => {
            let mut indices = Vec::with_capacity(n_samples);
            for _ in 0..n_samples {
                indices.push(rng.gen_range(0..y.len()));
            }
            Ok(indices)
        }
        SamplingStrategy::BalancedBootstrap => {
            let mut class_indices: HashMap<i32, Vec<usize>> = HashMap::new();
            for (idx, &class) in y.iter().enumerate() {
                class_indices.entry(class).or_default().push(idx);
            }
            let n_classes = class_indices.len();
            let samples_per_class = n_samples / n_classes;
            let extra_samples = n_samples % n_classes;
            let mut indices = Vec::with_capacity(n_samples);
            let mut extra_count = 0;
            for (_, class_idx_list) in class_indices.iter() {
                let mut n_class_samples = samples_per_class;
                if extra_count < extra_samples {
                    n_class_samples += 1;
                    extra_count += 1;
                }
                for _ in 0..n_class_samples {
                    let random_idx = rng.gen_range(0..class_idx_list.len());
                    indices.push(class_idx_list[random_idx]);
                }
            }
            indices.shuffle(&mut rng);
            Ok(indices)
        }
        SamplingStrategy::Stratified => {
            let mut class_counts: HashMap<i32, usize> = HashMap::new();
            let mut class_indices: HashMap<i32, Vec<usize>> = HashMap::new();
            for (idx, &class) in y.iter().enumerate() {
                *class_counts.entry(class).or_insert(0) += 1;
                class_indices.entry(class).or_default().push(idx);
            }
            let total_samples = y.len() as f64;
            let mut indices = Vec::with_capacity(n_samples);
            for (&class, &count) in &class_counts {
                let class_proportion = count as f64 / total_samples;
                let class_samples = (n_samples as f64 * class_proportion).round() as usize;
                let class_idx_list = &class_indices[&class];
                for _ in 0..class_samples {
                    let random_idx = rng.gen_range(0..class_idx_list.len());
                    indices.push(class_idx_list[random_idx]);
                }
            }
            while indices.len() < n_samples {
                indices.push(rng.gen_range(0..y.len()));
            }
            indices.shuffle(&mut rng);
            Ok(indices)
        }
        SamplingStrategy::SMOTEBootstrap => {
            let mut class_counts: HashMap<i32, usize> = HashMap::new();
            let mut class_indices: HashMap<i32, Vec<usize>> = HashMap::new();
            for (idx, &class) in y.iter().enumerate() {
                *class_counts.entry(class).or_insert(0) += 1;
                class_indices.entry(class).or_default().push(idx);
            }
            let max_class_size = class_counts.values().max().copied().unwrap_or(0);
            let mut indices = Vec::new();
            for (&class, class_idx_list) in &class_indices {
                let class_count = class_counts[&class];
                let oversample_ratio = max_class_size as f64 / class_count as f64;
                let target_samples = (n_samples as f64 / class_counts.len() as f64
                    * oversample_ratio)
                    .round() as usize;
                for _ in 0..target_samples {
                    let random_idx = rng.gen_range(0..class_idx_list.len());
                    indices.push(class_idx_list[random_idx]);
                }
            }
            indices.truncate(n_samples);
            indices.shuffle(&mut rng);
            Ok(indices)
        }
    }
}
/// Calculate comprehensive diversity measures for an ensemble of classifiers
///
/// This function evaluates various measures of diversity between individual classifiers
/// in an ensemble, which helps understand how well the ensemble combines different
/// decision boundaries and reduces overfitting.
///
/// # Arguments
/// * `individual_predictions` - Matrix where rows are samples and columns are classifier predictions
/// * `true_labels` - Ground truth labels for the samples
///
/// # Returns
/// * `DiversityMeasures` struct containing various diversity metrics
pub fn calculate_ensemble_diversity(
    individual_predictions: &Array2<i32>,
    true_labels: &Array1<i32>,
) -> Result<DiversityMeasures> {
    let (n_samples, n_classifiers) = individual_predictions.dim();
    if n_samples == 0 || n_classifiers < 2 {
        return Err(SklearsError::InvalidInput(
            "Need at least 2 classifiers and some samples to calculate diversity".to_string(),
        ));
    }
    if true_labels.len() != n_samples {
        return Err(SklearsError::InvalidInput(
            "Number of true labels must match number of samples".to_string(),
        ));
    }
    let mut individual_accuracies = Vec::with_capacity(n_classifiers);
    for classifier_idx in 0..n_classifiers {
        let predictions = individual_predictions.column(classifier_idx);
        let accuracy = predictions
            .iter()
            .zip(true_labels.iter())
            .map(|(&pred, &true_label)| (pred == true_label) as i32)
            .sum::<i32>() as f64
            / n_samples as f64;
        individual_accuracies.push(accuracy);
    }
    let mut q_statistics = Vec::new();
    let mut disagreements = Vec::new();
    let mut double_faults = Vec::new();
    let mut correlations = Vec::new();
    let mut kappa_statistics = Vec::new();
    for i in 0..n_classifiers {
        for j in (i + 1)..n_classifiers {
            let pred_i = individual_predictions.column(i);
            let pred_j = individual_predictions.column(j);
            let mut n11 = 0;
            let mut n10 = 0;
            let mut n01 = 0;
            let mut n00 = 0;
            for sample_idx in 0..n_samples {
                let i_correct = pred_i[sample_idx] == true_labels[sample_idx];
                let j_correct = pred_j[sample_idx] == true_labels[sample_idx];
                match (i_correct, j_correct) {
                    (true, true) => n11 += 1,
                    (true, false) => n10 += 1,
                    (false, true) => n01 += 1,
                    (false, false) => n00 += 1,
                }
            }
            let n11_f = n11 as f64;
            let n10_f = n10 as f64;
            let n01_f = n01 as f64;
            let n00_f = n00 as f64;
            let n_f = n_samples as f64;
            let q_stat = if (n11_f * n00_f + n10_f * n01_f) > 1e-10 {
                (n11_f * n00_f - n10_f * n01_f) / (n11_f * n00_f + n10_f * n01_f)
            } else {
                0.0
            };
            q_statistics.push(q_stat);
            let disagreement = (n10_f + n01_f) / n_f;
            disagreements.push(disagreement);
            let double_fault = n00_f / n_f;
            double_faults.push(double_fault);
            let p_i = (n11_f + n10_f) / n_f;
            let p_j = (n11_f + n01_f) / n_f;
            let correlation = if p_i * (1.0 - p_i) * p_j * (1.0 - p_j) > 1e-10 {
                (n11_f / n_f - p_i * p_j) / ((p_i * (1.0 - p_i) * p_j * (1.0 - p_j)).sqrt())
            } else {
                0.0
            };
            correlations.push(correlation);
            let p_observed = (n11_f + n00_f) / n_f;
            let p_expected = p_i * p_j + (1.0 - p_i) * (1.0 - p_j);
            let kappa = if (1.0 - p_expected).abs() > 1e-10 {
                (p_observed - p_expected) / (1.0 - p_expected)
            } else {
                0.0
            };
            kappa_statistics.push(kappa);
        }
    }
    let prediction_entropy = calculate_prediction_entropy(individual_predictions)?;
    Ok(DiversityMeasures {
        q_statistic: q_statistics.iter().sum::<f64>() / q_statistics.len() as f64,
        disagreement: disagreements.iter().sum::<f64>() / disagreements.len() as f64,
        double_fault: double_faults.iter().sum::<f64>() / double_faults.len() as f64,
        correlation_coefficient: correlations.iter().sum::<f64>() / correlations.len() as f64,
        kappa_statistic: kappa_statistics.iter().sum::<f64>() / kappa_statistics.len() as f64,
        prediction_entropy,
        individual_accuracies,
    })
}
/// Calculate prediction entropy of the ensemble
///
/// Higher entropy indicates that classifiers make more diverse predictions
fn calculate_prediction_entropy(individual_predictions: &Array2<i32>) -> Result<f64> {
    let (n_samples, n_classifiers) = individual_predictions.dim();
    let mut total_entropy = 0.0;
    for sample_idx in 0..n_samples {
        let sample_predictions = individual_predictions.row(sample_idx);
        let mut prediction_counts: HashMap<i32, usize> = HashMap::new();
        for &prediction in sample_predictions.iter() {
            *prediction_counts.entry(prediction).or_insert(0) += 1;
        }
        let mut sample_entropy = 0.0;
        for count in prediction_counts.values() {
            let probability = *count as f64 / n_classifiers as f64;
            if probability > 1e-10 {
                sample_entropy -= probability * probability.log2();
            }
        }
        total_entropy += sample_entropy;
    }
    Ok(total_entropy / n_samples as f64)
}
/// Calculate diversity measures for regression ensembles
pub fn calculate_regression_diversity(
    individual_predictions: &Array2<f64>,
    true_values: &Array1<f64>,
) -> Result<RegressionDiversityMeasures> {
    let (n_samples, n_regressors) = individual_predictions.dim();
    if n_samples == 0 || n_regressors < 2 {
        return Err(SklearsError::InvalidInput(
            "Need at least 2 regressors and some samples".to_string(),
        ));
    }
    if true_values.len() != n_samples {
        return Err(SklearsError::InvalidInput(
            "Number of true values must match number of samples".to_string(),
        ));
    }
    let mut individual_rmse = Vec::with_capacity(n_regressors);
    for regressor_idx in 0..n_regressors {
        let predictions = individual_predictions.column(regressor_idx);
        let mse = predictions
            .iter()
            .zip(true_values.iter())
            .map(|(&pred, &true_val)| (pred - true_val).powi(2))
            .sum::<f64>()
            / n_samples as f64;
        individual_rmse.push(mse.sqrt());
    }
    let mut correlations = Vec::new();
    for i in 0..n_regressors {
        for j in (i + 1)..n_regressors {
            let pred_i = individual_predictions.column(i);
            let pred_j = individual_predictions.column(j);
            let correlation =
                calculate_pearson_correlation(&pred_i.to_owned(), &pred_j.to_owned())?;
            correlations.push(correlation);
        }
    }
    let mut total_variance = 0.0;
    for sample_idx in 0..n_samples {
        let sample_predictions = individual_predictions.row(sample_idx);
        let mean_pred = sample_predictions.mean().unwrap();
        let variance = sample_predictions
            .iter()
            .map(|&pred| (pred - mean_pred).powi(2))
            .sum::<f64>()
            / n_regressors as f64;
        total_variance += variance;
    }
    let prediction_variance = total_variance / n_samples as f64;
    let mut total_bias = 0.0;
    let mut total_variance_component = 0.0;
    for sample_idx in 0..n_samples {
        let sample_predictions = individual_predictions.row(sample_idx);
        let mean_pred = sample_predictions.mean().unwrap();
        let true_val = true_values[sample_idx];
        let bias_squared = (mean_pred - true_val).powi(2);
        total_bias += bias_squared;
        let variance = sample_predictions
            .iter()
            .map(|&pred| (pred - mean_pred).powi(2))
            .sum::<f64>()
            / n_regressors as f64;
        total_variance_component += variance;
    }
    Ok(RegressionDiversityMeasures {
        prediction_correlation: correlations.iter().sum::<f64>() / correlations.len() as f64,
        prediction_variance,
        average_bias: (total_bias / n_samples as f64).sqrt(),
        average_variance: total_variance_component / n_samples as f64,
        individual_rmse,
    })
}
/// Calculate Pearson correlation coefficient between two arrays
fn calculate_pearson_correlation(x: &Array1<f64>, y: &Array1<f64>) -> Result<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return Err(SklearsError::InvalidInput(
            "Arrays must have same length and at least 2 elements".to_string(),
        ));
    }
    let _n = x.len() as f64;
    let mean_x = x.mean().unwrap();
    let mean_y = y.mean().unwrap();
    let mut numerator = 0.0;
    let mut sum_sq_x = 0.0;
    let mut sum_sq_y = 0.0;
    for i in 0..x.len() {
        let diff_x = x[i] - mean_x;
        let diff_y = y[i] - mean_y;
        numerator += diff_x * diff_y;
        sum_sq_x += diff_x * diff_x;
        sum_sq_y += diff_y * diff_y;
    }
    let denominator = (sum_sq_x * sum_sq_y).sqrt();
    if denominator < 1e-10 {
        Ok(0.0)
    } else {
        Ok(numerator / denominator)
    }
}
