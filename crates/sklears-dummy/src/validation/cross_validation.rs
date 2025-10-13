use super::data_splitting::{create_cv_folds, create_shuffled_indices, create_stratified_folds};
use super::validation_core::{
    ComprehensiveValidationResult, DummyValidationResult, FoldResult, StatisticalSummary,
    ValidationConfig,
};
use super::validation_utils::{
    calculate_classification_score, calculate_regression_score, validate_cv_params,
};

use scirs2_core::ndarray::Array1;
use sklears_core::error::{Result, SklearsError};
use sklears_core::traits::{Fit, Predict};
use sklears_core::types::{Features, Float, Int};
use std::time::Instant;

use crate::{DummyClassifier, DummyRegressor};

/// Perform cross-validation for a dummy classifier
pub fn cross_validate_dummy_classifier(
    classifier: DummyClassifier,
    x: &Features,
    y: &Array1<Int>,
    cv: usize,
) -> Result<DummyValidationResult> {
    validate_cv_params(x.nrows(), cv)?;

    let fold_size = x.nrows() / cv;
    let mut fold_scores = Vec::with_capacity(cv);

    for fold in 0..cv {
        let start_idx = fold * fold_size;
        let end_idx = if fold == cv - 1 {
            x.nrows()
        } else {
            (fold + 1) * fold_size
        };

        // Create train/test splits
        let mut train_indices = Vec::new();
        let mut test_indices = Vec::new();

        for i in 0..x.nrows() {
            if i >= start_idx && i < end_idx {
                test_indices.push(i);
            } else {
                train_indices.push(i);
            }
        }

        // Extract train/test data
        let x_train = x.select(scirs2_core::ndarray::Axis(0), &train_indices);
        let y_train = y.select(scirs2_core::ndarray::Axis(0), &train_indices);
        let x_test = x.select(scirs2_core::ndarray::Axis(0), &test_indices);
        let y_test = y.select(scirs2_core::ndarray::Axis(0), &test_indices);

        // Fit and predict
        let fitted = classifier.clone().fit(&x_train, &y_train)?;
        let predictions = fitted.predict(&x_test)?;

        // Calculate accuracy
        let correct = predictions
            .iter()
            .zip(y_test.iter())
            .filter(|(&pred, &actual)| pred == actual)
            .count();
        let accuracy = correct as Float / test_indices.len() as Float;
        fold_scores.push(accuracy);
    }

    let mean_score = fold_scores.iter().sum::<Float>() / fold_scores.len() as Float;
    let variance = fold_scores
        .iter()
        .map(|&score| (score - mean_score).powi(2))
        .sum::<Float>()
        / fold_scores.len() as Float;
    let std_score = variance.sqrt();

    Ok(DummyValidationResult {
        mean_score,
        std_score,
        fold_scores,
        strategy: format!("{:?}", classifier.strategy),
    })
}

/// Perform cross-validation for a dummy regressor
pub fn cross_validate_dummy_regressor(
    regressor: DummyRegressor,
    x: &Features,
    y: &Array1<Float>,
    cv: usize,
) -> Result<DummyValidationResult> {
    validate_cv_params(x.nrows(), cv)?;

    let fold_size = x.nrows() / cv;
    let mut fold_scores = Vec::with_capacity(cv);

    for fold in 0..cv {
        let start_idx = fold * fold_size;
        let end_idx = if fold == cv - 1 {
            x.nrows()
        } else {
            (fold + 1) * fold_size
        };

        // Create train/test splits
        let mut train_indices = Vec::new();
        let mut test_indices = Vec::new();

        for i in 0..x.nrows() {
            if i >= start_idx && i < end_idx {
                test_indices.push(i);
            } else {
                train_indices.push(i);
            }
        }

        // Extract train/test data
        let x_train = x.select(scirs2_core::ndarray::Axis(0), &train_indices);
        let y_train = y.select(scirs2_core::ndarray::Axis(0), &train_indices);
        let x_test = x.select(scirs2_core::ndarray::Axis(0), &test_indices);
        let y_test = y.select(scirs2_core::ndarray::Axis(0), &test_indices);

        // Fit and predict
        let fitted = regressor.clone().fit(&x_train, &y_train)?;
        let predictions = fitted.predict(&x_test)?;

        // Calculate negative MSE (higher is better)
        let mse = predictions
            .iter()
            .zip(y_test.iter())
            .map(|(&pred, &actual)| (pred - actual).powi(2))
            .sum::<Float>()
            / test_indices.len() as Float;
        fold_scores.push(-mse);
    }

    let mean_score = fold_scores.iter().sum::<Float>() / fold_scores.len() as Float;
    let variance = fold_scores
        .iter()
        .map(|&score| (score - mean_score).powi(2))
        .sum::<Float>()
        / fold_scores.len() as Float;
    let std_score = variance.sqrt();

    Ok(DummyValidationResult {
        mean_score,
        std_score,
        fold_scores,
        strategy: format!("{:?}", regressor.strategy),
    })
}

/// Perform comprehensive cross-validation with detailed results
pub fn comprehensive_cross_validate_classifier(
    classifier: DummyClassifier,
    x: &Features,
    y: &Array1<Int>,
    config: &ValidationConfig,
) -> Result<ComprehensiveValidationResult> {
    validate_cv_params(x.nrows(), config.cv_folds)?;

    let indices = if config.shuffle {
        create_shuffled_indices(x.nrows(), config.random_state)
    } else {
        (0..x.nrows()).collect()
    };

    let folds = create_cv_folds(&indices, config.cv_folds);
    let mut fold_results = Vec::new();
    let mut fold_scores = Vec::new();

    for (fold_idx, (train_indices, test_indices)) in folds.iter().enumerate() {
        let start_time = Instant::now();

        // Extract train/test data
        let x_train = x.select(scirs2_core::ndarray::Axis(0), train_indices);
        let y_train = y.select(scirs2_core::ndarray::Axis(0), train_indices);
        let x_test = x.select(scirs2_core::ndarray::Axis(0), test_indices);
        let y_test = y.select(scirs2_core::ndarray::Axis(0), test_indices);

        // Fit
        let fit_start = Instant::now();
        let fitted = classifier.clone().fit(&x_train, &y_train)?;
        let fit_time = fit_start.elapsed().as_secs_f64();

        // Predict
        let predict_start = Instant::now();
        let predictions = fitted.predict(&x_test)?;
        let predict_time = predict_start.elapsed().as_secs_f64();

        // Calculate score
        let score = calculate_classification_score(&predictions, &y_test);

        fold_results.push(FoldResult {
            fold_index: fold_idx,
            train_size: train_indices.len(),
            test_size: test_indices.len(),
            score,
            fit_time,
            predict_time,
        });

        fold_scores.push(score);
    }

    let statistical_summary = StatisticalSummary::from_scores(&fold_scores);
    let mean_score = statistical_summary.mean;
    let std_score = statistical_summary.std;

    let validation_result = DummyValidationResult {
        mean_score,
        std_score,
        fold_scores,
        strategy: format!("{:?}", classifier.strategy),
    };

    Ok(ComprehensiveValidationResult {
        validation_result,
        fold_details: fold_results,
        statistical_summary,
        config: config.clone(),
    })
}

/// Perform comprehensive cross-validation for regressor with detailed results
pub fn comprehensive_cross_validate_regressor(
    regressor: DummyRegressor,
    x: &Features,
    y: &Array1<Float>,
    config: &ValidationConfig,
) -> Result<ComprehensiveValidationResult> {
    validate_cv_params(x.nrows(), config.cv_folds)?;

    let indices = if config.shuffle {
        create_shuffled_indices(x.nrows(), config.random_state)
    } else {
        (0..x.nrows()).collect()
    };

    let folds = create_cv_folds(&indices, config.cv_folds);
    let mut fold_results = Vec::new();
    let mut fold_scores = Vec::new();

    for (fold_idx, (train_indices, test_indices)) in folds.iter().enumerate() {
        // Extract train/test data
        let x_train = x.select(scirs2_core::ndarray::Axis(0), train_indices);
        let y_train = y.select(scirs2_core::ndarray::Axis(0), train_indices);
        let x_test = x.select(scirs2_core::ndarray::Axis(0), test_indices);
        let y_test = y.select(scirs2_core::ndarray::Axis(0), test_indices);

        // Fit
        let fit_start = Instant::now();
        let fitted = regressor.clone().fit(&x_train, &y_train)?;
        let fit_time = fit_start.elapsed().as_secs_f64();

        // Predict
        let predict_start = Instant::now();
        let predictions = fitted.predict(&x_test)?;
        let predict_time = predict_start.elapsed().as_secs_f64();

        // Calculate score
        let score = calculate_regression_score(&predictions, &y_test);

        fold_results.push(FoldResult {
            fold_index: fold_idx,
            train_size: train_indices.len(),
            test_size: test_indices.len(),
            score,
            fit_time,
            predict_time,
        });

        fold_scores.push(score);
    }

    let statistical_summary = StatisticalSummary::from_scores(&fold_scores);
    let mean_score = statistical_summary.mean;
    let std_score = statistical_summary.std;

    let validation_result = DummyValidationResult {
        mean_score,
        std_score,
        fold_scores,
        strategy: format!("{:?}", regressor.strategy),
    };

    Ok(ComprehensiveValidationResult {
        validation_result,
        fold_details: fold_results,
        statistical_summary,
        config: config.clone(),
    })
}

/// Perform stratified cross-validation for classification
pub fn stratified_cross_validate_classifier(
    classifier: DummyClassifier,
    x: &Features,
    y: &Array1<Int>,
    cv: usize,
    random_state: Option<u64>,
) -> Result<DummyValidationResult> {
    validate_cv_params(x.nrows(), cv)?;

    let folds = create_stratified_folds(y, cv, random_state)?;
    let mut fold_scores = Vec::new();

    for (train_indices, test_indices) in folds {
        // Extract train/test data
        let x_train = x.select(scirs2_core::ndarray::Axis(0), &train_indices);
        let y_train = y.select(scirs2_core::ndarray::Axis(0), &train_indices);
        let x_test = x.select(scirs2_core::ndarray::Axis(0), &test_indices);
        let y_test = y.select(scirs2_core::ndarray::Axis(0), &test_indices);

        // Fit and predict
        let fitted = classifier.clone().fit(&x_train, &y_train)?;
        let predictions = fitted.predict(&x_test)?;

        // Calculate accuracy
        let correct = predictions
            .iter()
            .zip(y_test.iter())
            .filter(|(&pred, &actual)| pred == actual)
            .count();
        let accuracy = correct as Float / test_indices.len() as Float;
        fold_scores.push(accuracy);
    }

    let mean_score = fold_scores.iter().sum::<Float>() / fold_scores.len() as Float;
    let variance = fold_scores
        .iter()
        .map(|&score| (score - mean_score).powi(2))
        .sum::<Float>()
        / fold_scores.len() as Float;
    let std_score = variance.sqrt();

    Ok(DummyValidationResult {
        mean_score,
        std_score,
        fold_scores,
        strategy: format!("{:?}", classifier.strategy),
    })
}

/// Perform time series cross-validation (for temporal data)
pub fn time_series_cross_validate_regressor(
    regressor: DummyRegressor,
    x: &Features,
    y: &Array1<Float>,
    n_splits: usize,
    test_size: usize,
) -> Result<DummyValidationResult> {
    let n_samples = x.nrows();

    if n_samples < n_splits + test_size {
        return Err(SklearsError::InvalidInput(
            "Insufficient samples for time series cross-validation".to_string(),
        ));
    }

    let mut fold_scores = Vec::new();

    for i in 0..n_splits {
        let train_end = n_samples - (n_splits - i) * test_size;
        let test_start = train_end;
        let test_end = test_start + test_size;

        if train_end < 1 || test_end > n_samples {
            continue;
        }

        // Create train/test indices
        let train_indices: Vec<usize> = (0..train_end).collect();
        let test_indices: Vec<usize> = (test_start..test_end).collect();

        // Extract train/test data
        let x_train = x.select(scirs2_core::ndarray::Axis(0), &train_indices);
        let y_train = y.select(scirs2_core::ndarray::Axis(0), &train_indices);
        let x_test = x.select(scirs2_core::ndarray::Axis(0), &test_indices);
        let y_test = y.select(scirs2_core::ndarray::Axis(0), &test_indices);

        // Fit and predict
        let fitted = regressor.clone().fit(&x_train, &y_train)?;
        let predictions = fitted.predict(&x_test)?;

        // Calculate negative MSE
        let mse = predictions
            .iter()
            .zip(y_test.iter())
            .map(|(&pred, &actual)| (pred - actual).powi(2))
            .sum::<Float>()
            / test_indices.len() as Float;
        fold_scores.push(-mse);
    }

    if fold_scores.is_empty() {
        return Err(SklearsError::InvalidInput(
            "No valid folds created for time series cross-validation".to_string(),
        ));
    }

    let mean_score = fold_scores.iter().sum::<Float>() / fold_scores.len() as Float;
    let variance = fold_scores
        .iter()
        .map(|&score| (score - mean_score).powi(2))
        .sum::<Float>()
        / fold_scores.len() as Float;
    let std_score = variance.sqrt();

    Ok(DummyValidationResult {
        mean_score,
        std_score,
        fold_scores,
        strategy: format!("{:?}", regressor.strategy),
    })
}
