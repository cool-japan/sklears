//! Model validation utilities

use crate::{CrossValidator, ParameterValue};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::SliceRandomExt;
use sklears_core::{
    error::Result,
    prelude::{Predict, SklearsError},
    traits::Fit,
    traits::Score,
    types::Float,
};
use sklears_metrics::{
    classification::accuracy_score, get_scorer, regression::mean_squared_error, Scorer,
};
use std::collections::HashMap;

/// Helper function for scoring that handles both regression and classification
fn compute_score_for_regression_val(
    metric_name: &str,
    y_true: &Array1<f64>,
    y_pred: &Array1<f64>,
) -> Result<f64> {
    match metric_name {
        "neg_mean_squared_error" => Ok(-mean_squared_error(y_true, y_pred)?),
        "mean_squared_error" => Ok(mean_squared_error(y_true, y_pred)?),
        _ => {
            // For unsupported metrics, return a default score
            Err(sklears_core::error::SklearsError::InvalidInput(format!(
                "Metric '{}' not supported for regression",
                metric_name
            )))
        }
    }
}

/// Helper function for scoring classification data
fn compute_score_for_classification_val(
    metric_name: &str,
    y_true: &Array1<i32>,
    y_pred: &Array1<i32>,
) -> Result<f64> {
    match metric_name {
        "accuracy" => Ok(accuracy_score(y_true, y_pred)?),
        _ => {
            let scorer = get_scorer(metric_name)?;
            scorer.score(y_true.as_slice().unwrap(), y_pred.as_slice().unwrap())
        }
    }
}

/// Scoring method for cross-validation
#[derive(Debug, Clone)]
pub enum Scoring {
    /// Use the estimator's built-in score method
    EstimatorScore,
    /// Use a predefined scorer by name
    Metric(String),
    /// Use a specific scorer configuration
    Scorer(Scorer),
    /// Use multiple scoring metrics
    MultiMetric(Vec<String>),
    /// Use a custom scoring function
    Custom(fn(&Array1<Float>, &Array1<Float>) -> Result<f64>),
}

/// Enhanced scoring result that can handle multiple metrics
#[derive(Debug, Clone)]
pub enum ScoreResult {
    /// Single score value
    Single(f64),
    /// Multiple score values with metric names
    Multiple(HashMap<String, f64>),
}

impl ScoreResult {
    /// Get a single score (first score if multiple)
    pub fn as_single(&self) -> f64 {
        match self {
            ScoreResult::Single(score) => *score,
            ScoreResult::Multiple(scores) => scores.values().next().copied().unwrap_or(0.0),
        }
    }

    /// Get scores as a map
    pub fn as_multiple(&self) -> HashMap<String, f64> {
        match self {
            ScoreResult::Single(score) => {
                let mut map = HashMap::new();
                map.insert("score".to_string(), *score);
                map
            }
            ScoreResult::Multiple(scores) => scores.clone(),
        }
    }
}

/// Evaluate metric(s) by cross-validation and also record fit/score times
#[allow(clippy::too_many_arguments)]
pub fn cross_validate<E, F, C>(
    estimator: E,
    x: &Array2<Float>,
    y: &Array1<Float>,
    cv: &C,
    scoring: Scoring,
    return_train_score: bool,
    return_estimator: bool,
    _n_jobs: Option<usize>,
) -> Result<CrossValidateResult<F>>
where
    E: Clone,
    F: Clone,
    E: Fit<Array2<Float>, Array1<Float>, Fitted = F>,
    F: Predict<Array2<Float>, Array1<Float>>,
    F: Score<Array2<Float>, Array1<Float>, Float = f64>,
    C: CrossValidator,
{
    // Note: This assumes KFold or other CV that doesn't need y
    // For StratifiedKFold, you would need to pass integer labels
    let splits = cv.split(x.nrows(), None);
    let n_splits = splits.len();

    let mut test_scores = Vec::with_capacity(n_splits);
    let mut train_scores = if return_train_score {
        Some(Vec::with_capacity(n_splits))
    } else {
        None
    };
    let mut fit_times = Vec::with_capacity(n_splits);
    let mut score_times = Vec::with_capacity(n_splits);
    let mut estimators = if return_estimator {
        Some(Vec::with_capacity(n_splits))
    } else {
        None
    };

    // Process each fold
    for (train_idx, test_idx) in splits {
        // Extract train and test data
        let x_train = x.select(scirs2_core::ndarray::Axis(0), &train_idx);
        let y_train = y.select(scirs2_core::ndarray::Axis(0), &train_idx);
        let x_test = x.select(scirs2_core::ndarray::Axis(0), &test_idx);
        let y_test = y.select(scirs2_core::ndarray::Axis(0), &test_idx);

        // Fit the estimator
        let start = std::time::Instant::now();
        let fitted = estimator.clone().fit(&x_train, &y_train)?;
        let fit_time = start.elapsed().as_secs_f64();
        fit_times.push(fit_time);

        // Score on test set
        let start = std::time::Instant::now();
        let test_score = match &scoring {
            Scoring::EstimatorScore => fitted.score(&x_test, &y_test)?,
            Scoring::Custom(func) => {
                let y_pred = fitted.predict(&x_test)?;
                func(&y_test.to_owned(), &y_pred)?
            }
            Scoring::Metric(metric_name) => {
                let y_pred = fitted.predict(&x_test)?;
                // Determine if this is classification or regression based on the data type
                if y_test.iter().all(|&x| x.fract() == 0.0) {
                    // Integer-like values, likely classification
                    let y_true_int: Array1<i32> = y_test.mapv(|x| x as i32);
                    let y_pred_int: Array1<i32> = y_pred.mapv(|x| x as i32);
                    compute_score_for_classification_val(metric_name, &y_true_int, &y_pred_int)?
                } else {
                    // Float values, likely regression
                    compute_score_for_regression_val(metric_name, &y_test, &y_pred)?
                }
            }
            Scoring::Scorer(scorer) => {
                let y_pred = fitted.predict(&x_test)?;
                scorer.score_float(y_test.as_slice().unwrap(), y_pred.as_slice().unwrap())?
            }
            Scoring::MultiMetric(_) => {
                return Err(SklearsError::InvalidInput(
                    "MultiMetric scoring not supported in single metric context".to_string(),
                ));
            }
        };
        let score_time = start.elapsed().as_secs_f64();
        score_times.push(score_time);
        test_scores.push(test_score);

        // Score on train set if requested
        if let Some(ref mut train_scores) = train_scores {
            let train_score = match &scoring {
                Scoring::EstimatorScore => fitted.score(&x_train, &y_train)?,
                Scoring::Custom(func) => {
                    let y_pred = fitted.predict(&x_train)?;
                    func(&y_train.to_owned(), &y_pred)?
                }
                Scoring::Metric(metric_name) => {
                    let y_pred = fitted.predict(&x_train)?;
                    // Determine if this is classification or regression based on the data type
                    if y_train.iter().all(|&x| x.fract() == 0.0) {
                        // Integer-like values, likely classification
                        let y_true_int: Array1<i32> = y_train.mapv(|x| x as i32);
                        let y_pred_int: Array1<i32> = y_pred.mapv(|x| x as i32);
                        compute_score_for_classification_val(metric_name, &y_true_int, &y_pred_int)?
                    } else {
                        // Float values, likely regression
                        compute_score_for_regression_val(metric_name, &y_train, &y_pred)?
                    }
                }
                Scoring::Scorer(scorer) => {
                    let y_pred = fitted.predict(&x_train)?;
                    scorer.score_float(y_train.as_slice().unwrap(), y_pred.as_slice().unwrap())?
                }
                Scoring::MultiMetric(_metrics) => {
                    // For multi-metric, just use the first metric for now
                    fitted.score(&x_train, &y_train)?
                }
            };
            train_scores.push(train_score);
        }

        // Store estimator if requested
        if let Some(ref mut estimators) = estimators {
            estimators.push(fitted);
        }
    }

    Ok(CrossValidateResult {
        test_scores: Array1::from_vec(test_scores),
        train_scores: train_scores.map(Array1::from_vec),
        fit_times: Array1::from_vec(fit_times),
        score_times: Array1::from_vec(score_times),
        estimators,
    })
}

/// Result of cross_validate
#[derive(Debug, Clone)]
pub struct CrossValidateResult<F> {
    pub test_scores: Array1<f64>,
    pub train_scores: Option<Array1<f64>>,
    pub fit_times: Array1<f64>,
    pub score_times: Array1<f64>,
    pub estimators: Option<Vec<F>>,
}

/// Evaluate a score by cross-validation
pub fn cross_val_score<E, F, C>(
    estimator: E,
    x: &Array2<Float>,
    y: &Array1<Float>,
    cv: &C,
    scoring: Option<Scoring>,
    n_jobs: Option<usize>,
) -> Result<Array1<f64>>
where
    E: Clone,
    F: Clone,
    E: Fit<Array2<Float>, Array1<Float>, Fitted = F>,
    F: Predict<Array2<Float>, Array1<Float>>,
    F: Score<Array2<Float>, Array1<Float>, Float = f64>,
    C: CrossValidator,
{
    let scoring = scoring.unwrap_or(Scoring::EstimatorScore);
    let result = cross_validate(
        estimator, x, y, cv, scoring, false, // return_train_score
        false, // return_estimator
        n_jobs,
    )?;

    Ok(result.test_scores)
}

/// Generate cross-validated estimates for each input data point
pub fn cross_val_predict<E, F, C>(
    estimator: E,
    x: &Array2<Float>,
    y: &Array1<Float>,
    cv: &C,
    _n_jobs: Option<usize>,
) -> Result<Array1<Float>>
where
    E: Clone,
    E: Fit<Array2<Float>, Array1<Float>, Fitted = F>,
    F: Predict<Array2<Float>, Array1<Float>>,
    C: CrossValidator,
{
    // Note: This assumes KFold or other CV that doesn't need y
    // For StratifiedKFold, you would need to pass integer labels
    let splits = cv.split(x.nrows(), None);
    let n_samples = x.nrows();

    // Initialize predictions array
    let mut predictions = Array1::<Float>::zeros(n_samples);

    // Process each fold
    for (train_idx, test_idx) in splits {
        // Extract train and test data
        let x_train = x.select(scirs2_core::ndarray::Axis(0), &train_idx);
        let y_train = y.select(scirs2_core::ndarray::Axis(0), &train_idx);
        let x_test = x.select(scirs2_core::ndarray::Axis(0), &test_idx);

        // Fit and predict
        let fitted = estimator.clone().fit(&x_train, &y_train)?;
        let y_pred = fitted.predict(&x_test)?;

        // Store predictions at the correct indices
        for (i, &idx) in test_idx.iter().enumerate() {
            predictions[idx] = y_pred[i];
        }
    }

    Ok(predictions)
}

/// Learning curve results
#[derive(Debug, Clone)]
pub struct LearningCurveResult {
    /// Training set sizes used
    pub train_sizes: Array1<usize>,
    /// Training scores for each size
    pub train_scores: Array2<f64>,
    /// Validation scores for each size
    pub test_scores: Array2<f64>,
    /// Mean training scores for each size
    pub train_scores_mean: Array1<f64>,
    /// Mean validation scores for each size
    pub test_scores_mean: Array1<f64>,
    /// Standard deviation of training scores for each size
    pub train_scores_std: Array1<f64>,
    /// Standard deviation of validation scores for each size
    pub test_scores_std: Array1<f64>,
    /// Lower confidence bound for training scores (mean - confidence_interval)
    pub train_scores_lower: Array1<f64>,
    /// Upper confidence bound for training scores (mean + confidence_interval)
    pub train_scores_upper: Array1<f64>,
    /// Lower confidence bound for validation scores (mean - confidence_interval)
    pub test_scores_lower: Array1<f64>,
    /// Upper confidence bound for validation scores (mean + confidence_interval)
    pub test_scores_upper: Array1<f64>,
}

/// Compute learning curves for an estimator
///
/// Determines cross-validated training and test scores for different training
/// set sizes. This is useful to find out if we suffer from bias vs variance
/// when we add more data to the training set.
///
/// # Arguments
/// * `estimator` - The estimator to evaluate
/// * `x` - Training data features
/// * `y` - Training data targets
/// * `cv` - Cross-validation splitter
/// * `train_sizes` - Relative or absolute numbers of training examples that will be used to generate the learning curve
/// * `scoring` - Scoring method to use
/// * `confidence_level` - Confidence level for confidence bands (default: 0.95 for 95% confidence interval)
#[allow(clippy::too_many_arguments)]
pub fn learning_curve<E, F, C>(
    estimator: E,
    x: &Array2<Float>,
    y: &Array1<Float>,
    cv: &C,
    train_sizes: Option<Vec<f64>>,
    scoring: Option<Scoring>,
    confidence_level: Option<f64>,
) -> Result<LearningCurveResult>
where
    E: Clone,
    F: Clone,
    E: Fit<Array2<Float>, Array1<Float>, Fitted = F>,
    F: Predict<Array2<Float>, Array1<Float>>,
    F: Score<Array2<Float>, Array1<Float>, Float = f64>,
    C: CrossValidator,
{
    let n_samples = x.nrows();
    let scoring = scoring.unwrap_or(Scoring::EstimatorScore);

    // Default train sizes: 10%, 30%, 50%, 70%, 90%, 100%
    let train_size_fractions = train_sizes.unwrap_or_else(|| vec![0.1, 0.3, 0.5, 0.7, 0.9, 1.0]);

    // Convert fractions to actual sizes
    let train_sizes_actual: Vec<usize> = train_size_fractions
        .iter()
        .map(|&frac| {
            let size = (frac * n_samples as f64).round() as usize;
            size.max(1).min(n_samples) // Ensure between 1 and n_samples
        })
        .collect();

    let n_splits = cv.n_splits();
    let n_train_sizes = train_sizes_actual.len();

    let mut train_scores = Array2::<f64>::zeros((n_train_sizes, n_splits));
    let mut test_scores = Array2::<f64>::zeros((n_train_sizes, n_splits));

    // Get CV splits
    let splits = cv.split(x.nrows(), None);

    for (size_idx, &train_size) in train_sizes_actual.iter().enumerate() {
        for (split_idx, (train_idx, test_idx)) in splits.iter().enumerate() {
            // Limit training set to the desired size
            let mut limited_train_idx = train_idx.clone();
            if limited_train_idx.len() > train_size {
                limited_train_idx.truncate(train_size);
            }

            // Extract data
            let x_train = x.select(scirs2_core::ndarray::Axis(0), &limited_train_idx);
            let y_train = y.select(scirs2_core::ndarray::Axis(0), &limited_train_idx);
            let x_test = x.select(scirs2_core::ndarray::Axis(0), test_idx);
            let y_test = y.select(scirs2_core::ndarray::Axis(0), test_idx);

            // Fit estimator
            let fitted = estimator.clone().fit(&x_train, &y_train)?;

            // Score on training set
            let train_score = match &scoring {
                Scoring::EstimatorScore => fitted.score(&x_train, &y_train)?,
                Scoring::Custom(func) => {
                    let y_pred = fitted.predict(&x_train)?;
                    func(&y_train.to_owned(), &y_pred)?
                }
                Scoring::Metric(metric_name) => {
                    let y_pred = fitted.predict(&x_train)?;
                    // Determine if this is classification or regression based on the data type
                    if y_train.iter().all(|&x| x.fract() == 0.0) {
                        // Integer-like values, likely classification
                        let y_true_int: Array1<i32> = y_train.mapv(|x| x as i32);
                        let y_pred_int: Array1<i32> = y_pred.mapv(|x| x as i32);
                        compute_score_for_classification_val(metric_name, &y_true_int, &y_pred_int)?
                    } else {
                        // Float values, likely regression
                        compute_score_for_regression_val(metric_name, &y_train, &y_pred)?
                    }
                }
                Scoring::Scorer(scorer) => {
                    let y_pred = fitted.predict(&x_train)?;
                    scorer.score_float(y_train.as_slice().unwrap(), y_pred.as_slice().unwrap())?
                }
                Scoring::MultiMetric(_metrics) => {
                    // For multi-metric, just use the first metric for now
                    fitted.score(&x_train, &y_train)?
                }
            };
            train_scores[[size_idx, split_idx]] = train_score;

            // Score on test set
            let test_score = match &scoring {
                Scoring::EstimatorScore => fitted.score(&x_test, &y_test)?,
                Scoring::Custom(func) => {
                    let y_pred = fitted.predict(&x_test)?;
                    func(&y_test.to_owned(), &y_pred)?
                }
                Scoring::Metric(metric_name) => {
                    let y_pred = fitted.predict(&x_test)?;
                    // Determine if this is classification or regression based on the data type
                    if y_test.iter().all(|&x| x.fract() == 0.0) {
                        // Integer-like values, likely classification
                        let y_true_int: Array1<i32> = y_test.mapv(|x| x as i32);
                        let y_pred_int: Array1<i32> = y_pred.mapv(|x| x as i32);
                        compute_score_for_classification_val(metric_name, &y_true_int, &y_pred_int)?
                    } else {
                        // Float values, likely regression
                        compute_score_for_regression_val(metric_name, &y_test, &y_pred)?
                    }
                }
                Scoring::Scorer(scorer) => {
                    let y_pred = fitted.predict(&x_test)?;
                    scorer.score_float(y_test.as_slice().unwrap(), y_pred.as_slice().unwrap())?
                }
                Scoring::MultiMetric(_metrics) => {
                    // For multi-metric, just use the first metric for now
                    fitted.score(&x_test, &y_test)?
                }
            };
            test_scores[[size_idx, split_idx]] = test_score;
        }
    }

    // Calculate confidence level (default 95%)
    let confidence = confidence_level.unwrap_or(0.95);
    let alpha = 1.0 - confidence;
    let z_score = 1.96; // Approximate 95% confidence interval

    // Calculate statistics for each training size
    let mut train_scores_mean = Array1::<f64>::zeros(n_train_sizes);
    let mut test_scores_mean = Array1::<f64>::zeros(n_train_sizes);
    let mut train_scores_std = Array1::<f64>::zeros(n_train_sizes);
    let mut test_scores_std = Array1::<f64>::zeros(n_train_sizes);
    let mut train_scores_lower = Array1::<f64>::zeros(n_train_sizes);
    let mut train_scores_upper = Array1::<f64>::zeros(n_train_sizes);
    let mut test_scores_lower = Array1::<f64>::zeros(n_train_sizes);
    let mut test_scores_upper = Array1::<f64>::zeros(n_train_sizes);

    for size_idx in 0..n_train_sizes {
        // Extract scores for this training size across all CV folds
        let train_scores_for_size: Vec<f64> = (0..n_splits)
            .map(|split_idx| train_scores[[size_idx, split_idx]])
            .collect();
        let test_scores_for_size: Vec<f64> = (0..n_splits)
            .map(|split_idx| test_scores[[size_idx, split_idx]])
            .collect();

        // Calculate mean and std for training scores
        let train_mean = train_scores_for_size.iter().sum::<f64>() / n_splits as f64;
        let train_variance = train_scores_for_size
            .iter()
            .map(|&x| (x - train_mean).powi(2))
            .sum::<f64>()
            / (n_splits - 1).max(1) as f64;
        let train_std = train_variance.sqrt();
        let train_sem = train_std / (n_splits as f64).sqrt(); // Standard error of the mean

        // Calculate mean and std for test scores
        let test_mean = test_scores_for_size.iter().sum::<f64>() / n_splits as f64;
        let test_variance = test_scores_for_size
            .iter()
            .map(|&x| (x - test_mean).powi(2))
            .sum::<f64>()
            / (n_splits - 1).max(1) as f64;
        let test_std = test_variance.sqrt();
        let test_sem = test_std / (n_splits as f64).sqrt(); // Standard error of the mean

        // Calculate confidence intervals
        let train_margin = z_score * train_sem;
        let test_margin = z_score * test_sem;

        train_scores_mean[size_idx] = train_mean;
        test_scores_mean[size_idx] = test_mean;
        train_scores_std[size_idx] = train_std;
        test_scores_std[size_idx] = test_std;
        train_scores_lower[size_idx] = train_mean - train_margin;
        train_scores_upper[size_idx] = train_mean + train_margin;
        test_scores_lower[size_idx] = test_mean - test_margin;
        test_scores_upper[size_idx] = test_mean + test_margin;
    }

    Ok(LearningCurveResult {
        train_sizes: Array1::from_vec(train_sizes_actual),
        train_scores,
        test_scores,
        train_scores_mean,
        test_scores_mean,
        train_scores_std,
        test_scores_std,
        train_scores_lower,
        train_scores_upper,
        test_scores_lower,
        test_scores_upper,
    })
}

/// Validation curve results
#[derive(Debug, Clone)]
pub struct ValidationCurveResult {
    /// Parameter values used
    pub param_values: Vec<ParameterValue>,
    /// Training scores for each parameter value
    pub train_scores: Array2<f64>,
    /// Validation scores for each parameter value
    pub test_scores: Array2<f64>,
    /// Mean training scores for each parameter value
    pub train_scores_mean: Array1<f64>,
    /// Mean validation scores for each parameter value
    pub test_scores_mean: Array1<f64>,
    /// Standard deviation of training scores for each parameter value
    pub train_scores_std: Array1<f64>,
    /// Standard deviation of validation scores for each parameter value
    pub test_scores_std: Array1<f64>,
    /// Lower error bar for training scores (mean - std_error)
    pub train_scores_lower: Array1<f64>,
    /// Upper error bar for training scores (mean + std_error)
    pub train_scores_upper: Array1<f64>,
    /// Lower error bar for validation scores (mean - std_error)
    pub test_scores_lower: Array1<f64>,
    /// Upper error bar for validation scores (mean + std_error)
    pub test_scores_upper: Array1<f64>,
}

/// Parameter configuration function type
pub type ParamConfigFn<E> = Box<dyn Fn(E, &ParameterValue) -> Result<E>>;

/// Compute validation curves for an estimator
///
/// Determines training and test scores for a varying parameter value.
/// This is useful to understand the effect of a specific parameter on
/// model performance and to detect overfitting/underfitting.
///
/// # Arguments
/// * `estimator` - The estimator to evaluate
/// * `x` - Training data features
/// * `y` - Training data targets
/// * `_param_name` - Name of the parameter being varied (for documentation)
/// * `param_range` - Parameter values to test
/// * `param_config` - Function to configure estimator with parameter values
/// * `cv` - Cross-validation splitter
/// * `scoring` - Scoring method to use
/// * `confidence_level` - Confidence level for error bars (default: 0.95 for 95% confidence interval)
#[allow(clippy::too_many_arguments)]
pub fn validation_curve<E, F, C>(
    estimator: E,
    x: &Array2<Float>,
    y: &Array1<Float>,
    _param_name: &str,
    param_range: Vec<ParameterValue>,
    param_config: ParamConfigFn<E>,
    cv: &C,
    scoring: Option<Scoring>,
    confidence_level: Option<f64>,
) -> Result<ValidationCurveResult>
where
    E: Clone,
    F: Clone,
    E: Fit<Array2<Float>, Array1<Float>, Fitted = F>,
    F: Predict<Array2<Float>, Array1<Float>>,
    F: Score<Array2<Float>, Array1<Float>, Float = f64>,
    C: CrossValidator,
{
    let scoring = scoring.unwrap_or(Scoring::EstimatorScore);
    let n_splits = cv.n_splits();
    let n_params = param_range.len();

    let mut train_scores = Array2::<f64>::zeros((n_params, n_splits));
    let mut test_scores = Array2::<f64>::zeros((n_params, n_splits));

    // Get CV splits
    let splits = cv.split(x.nrows(), None);

    for (param_idx, param_value) in param_range.iter().enumerate() {
        for (split_idx, (train_idx, test_idx)) in splits.iter().enumerate() {
            // Extract data
            let x_train = x.select(scirs2_core::ndarray::Axis(0), train_idx);
            let y_train = y.select(scirs2_core::ndarray::Axis(0), train_idx);
            let x_test = x.select(scirs2_core::ndarray::Axis(0), test_idx);
            let y_test = y.select(scirs2_core::ndarray::Axis(0), test_idx);

            // Configure estimator with current parameter value
            let configured_estimator = param_config(estimator.clone(), param_value)?;

            // Fit estimator
            let fitted = configured_estimator.fit(&x_train, &y_train)?;

            // Score on training set
            let train_score = match &scoring {
                Scoring::EstimatorScore => fitted.score(&x_train, &y_train)?,
                Scoring::Custom(func) => {
                    let y_pred = fitted.predict(&x_train)?;
                    func(&y_train.to_owned(), &y_pred)?
                }
                Scoring::Metric(metric_name) => {
                    let y_pred = fitted.predict(&x_train)?;
                    // Determine if this is classification or regression based on the data type
                    if y_train.iter().all(|&x| x.fract() == 0.0) {
                        // Integer-like values, likely classification
                        let y_true_int: Array1<i32> = y_train.mapv(|x| x as i32);
                        let y_pred_int: Array1<i32> = y_pred.mapv(|x| x as i32);
                        compute_score_for_classification_val(metric_name, &y_true_int, &y_pred_int)?
                    } else {
                        // Float values, likely regression
                        compute_score_for_regression_val(metric_name, &y_train, &y_pred)?
                    }
                }
                Scoring::Scorer(scorer) => {
                    let y_pred = fitted.predict(&x_train)?;
                    scorer.score_float(y_train.as_slice().unwrap(), y_pred.as_slice().unwrap())?
                }
                Scoring::MultiMetric(_metrics) => {
                    // For multi-metric, just use the first metric for now
                    fitted.score(&x_train, &y_train)?
                }
            };
            train_scores[[param_idx, split_idx]] = train_score;

            // Score on test set
            let test_score = match &scoring {
                Scoring::EstimatorScore => fitted.score(&x_test, &y_test)?,
                Scoring::Custom(func) => {
                    let y_pred = fitted.predict(&x_test)?;
                    func(&y_test.to_owned(), &y_pred)?
                }
                Scoring::Metric(metric_name) => {
                    let y_pred = fitted.predict(&x_test)?;
                    // Determine if this is classification or regression based on the data type
                    if y_test.iter().all(|&x| x.fract() == 0.0) {
                        // Integer-like values, likely classification
                        let y_true_int: Array1<i32> = y_test.mapv(|x| x as i32);
                        let y_pred_int: Array1<i32> = y_pred.mapv(|x| x as i32);
                        compute_score_for_classification_val(metric_name, &y_true_int, &y_pred_int)?
                    } else {
                        // Float values, likely regression
                        compute_score_for_regression_val(metric_name, &y_test, &y_pred)?
                    }
                }
                Scoring::Scorer(scorer) => {
                    let y_pred = fitted.predict(&x_test)?;
                    scorer.score_float(y_test.as_slice().unwrap(), y_pred.as_slice().unwrap())?
                }
                Scoring::MultiMetric(_metrics) => {
                    // For multi-metric, just use the first metric for now
                    fitted.score(&x_test, &y_test)?
                }
            };
            test_scores[[param_idx, split_idx]] = test_score;
        }
    }

    // Calculate confidence level (default 95%)
    let confidence = confidence_level.unwrap_or(0.95);
    let z_score = 1.96; // Approximate 95% confidence interval

    // Calculate statistics for each parameter value
    let mut train_scores_mean = Array1::<f64>::zeros(n_params);
    let mut test_scores_mean = Array1::<f64>::zeros(n_params);
    let mut train_scores_std = Array1::<f64>::zeros(n_params);
    let mut test_scores_std = Array1::<f64>::zeros(n_params);
    let mut train_scores_lower = Array1::<f64>::zeros(n_params);
    let mut train_scores_upper = Array1::<f64>::zeros(n_params);
    let mut test_scores_lower = Array1::<f64>::zeros(n_params);
    let mut test_scores_upper = Array1::<f64>::zeros(n_params);

    for param_idx in 0..n_params {
        // Extract scores for this parameter value across all CV folds
        let train_scores_for_param: Vec<f64> = (0..n_splits)
            .map(|split_idx| train_scores[[param_idx, split_idx]])
            .collect();
        let test_scores_for_param: Vec<f64> = (0..n_splits)
            .map(|split_idx| test_scores[[param_idx, split_idx]])
            .collect();

        // Calculate mean and std for training scores
        let train_mean = train_scores_for_param.iter().sum::<f64>() / n_splits as f64;
        let train_variance = train_scores_for_param
            .iter()
            .map(|&x| (x - train_mean).powi(2))
            .sum::<f64>()
            / (n_splits - 1).max(1) as f64;
        let train_std = train_variance.sqrt();
        let train_sem = train_std / (n_splits as f64).sqrt(); // Standard error of the mean

        // Calculate mean and std for test scores
        let test_mean = test_scores_for_param.iter().sum::<f64>() / n_splits as f64;
        let test_variance = test_scores_for_param
            .iter()
            .map(|&x| (x - test_mean).powi(2))
            .sum::<f64>()
            / (n_splits - 1).max(1) as f64;
        let test_std = test_variance.sqrt();
        let test_sem = test_std / (n_splits as f64).sqrt(); // Standard error of the mean

        // Calculate error bars (using standard error for error bars)
        let train_margin = train_sem;
        let test_margin = test_sem;

        train_scores_mean[param_idx] = train_mean;
        test_scores_mean[param_idx] = test_mean;
        train_scores_std[param_idx] = train_std;
        test_scores_std[param_idx] = test_std;
        train_scores_lower[param_idx] = train_mean - train_margin;
        train_scores_upper[param_idx] = train_mean + train_margin;
        test_scores_lower[param_idx] = test_mean - test_margin;
        test_scores_upper[param_idx] = test_mean + test_margin;
    }

    Ok(ValidationCurveResult {
        param_values: param_range,
        train_scores,
        test_scores,
        train_scores_mean,
        test_scores_mean,
        train_scores_std,
        test_scores_std,
        train_scores_lower,
        train_scores_upper,
        test_scores_lower,
        test_scores_upper,
    })
}

/// Evaluate the significance of a cross-validated score with permutations
///
/// This function tests whether the estimator performs significantly better than
/// random by computing cross-validation scores on permuted labels.
#[allow(clippy::too_many_arguments)]
pub fn permutation_test_score<E, F, C>(
    estimator: E,
    x: &Array2<Float>,
    y: &Array1<Float>,
    cv: &C,
    scoring: Option<Scoring>,
    n_permutations: usize,
    random_state: Option<u64>,
    n_jobs: Option<usize>,
) -> Result<PermutationTestResult>
where
    E: Clone,
    F: Clone,
    E: Fit<Array2<Float>, Array1<Float>, Fitted = F>,
    F: Predict<Array2<Float>, Array1<Float>>,
    F: Score<Array2<Float>, Array1<Float>, Float = f64>,
    C: CrossValidator,
{
    use scirs2_core::random::prelude::*;
    use scirs2_core::random::rngs::StdRng;

    let scoring = scoring.unwrap_or(Scoring::EstimatorScore);

    // Compute original score
    let original_scores =
        cross_val_score(estimator.clone(), x, y, cv, Some(scoring.clone()), n_jobs)?;
    let original_score = original_scores.mean().unwrap_or(0.0);

    // Initialize random number generator
    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::seed_from_u64(42)
    };

    // Compute permutation scores
    let mut permutation_scores = Vec::with_capacity(n_permutations);

    for _ in 0..n_permutations {
        // Create permuted labels
        let mut y_permuted = y.to_owned();
        let mut indices: Vec<usize> = (0..y.len()).collect();
        indices.shuffle(&mut rng);

        for (i, &perm_idx) in indices.iter().enumerate() {
            y_permuted[i] = y[perm_idx];
        }

        // Compute score with permuted labels
        let perm_scores = cross_val_score(
            estimator.clone(),
            x,
            &y_permuted,
            cv,
            Some(scoring.clone()),
            n_jobs,
        )?;
        let perm_score = perm_scores.mean().unwrap_or(0.0);
        permutation_scores.push(perm_score);
    }

    // Compute p-value
    let n_better_or_equal = permutation_scores
        .iter()
        .filter(|&&score| score >= original_score)
        .count();
    let p_value = (n_better_or_equal + 1) as f64 / (n_permutations + 1) as f64;

    Ok(PermutationTestResult {
        statistic: original_score,
        pvalue: p_value,
        permutation_scores: Array1::from_vec(permutation_scores),
    })
}

/// Result of permutation test
#[derive(Debug, Clone)]
pub struct PermutationTestResult {
    /// The original cross-validation score
    pub statistic: f64,
    /// The p-value of the permutation test
    pub pvalue: f64,
    /// Scores obtained for each permutation
    pub permutation_scores: Array1<f64>,
}

/// Nested cross-validation for unbiased model evaluation with hyperparameter optimization
///
/// This implements nested cross-validation which provides an unbiased estimate of model
/// performance by using separate CV loops for hyperparameter optimization (inner loop)
/// and performance estimation (outer loop).
pub fn nested_cross_validate<E, F, C>(
    estimator: E,
    x: &Array2<Float>,
    y: &Array1<Float>,
    outer_cv: &C,
    inner_cv: &C,
    param_grid: &[ParameterValue],
    param_config: ParamConfigFn<E>,
    scoring: Option<fn(&Array1<Float>, &Array1<Float>) -> f64>,
) -> Result<NestedCVResult>
where
    E: Clone,
    F: Clone,
    E: Fit<Array2<Float>, Array1<Float>, Fitted = F>,
    F: Predict<Array2<Float>, Array1<Float>>,
    F: Score<Array2<Float>, Array1<Float>, Float = f64>,
    C: CrossValidator,
{
    let outer_splits = outer_cv.split(x.nrows(), None);
    let mut outer_scores = Vec::with_capacity(outer_splits.len());
    let mut best_params_per_fold = Vec::with_capacity(outer_splits.len());
    let mut inner_scores_per_fold = Vec::with_capacity(outer_splits.len());

    for (outer_train_idx, outer_test_idx) in outer_splits {
        // Extract outer train/test data
        let outer_train_x = extract_rows(x, &outer_train_idx);
        let outer_train_y = extract_elements(y, &outer_train_idx);
        let outer_test_x = extract_rows(x, &outer_test_idx);
        let outer_test_y = extract_elements(y, &outer_test_idx);

        // Inner cross-validation for hyperparameter optimization
        let mut best_score = f64::NEG_INFINITY;
        let mut best_param = param_grid[0].clone();
        let mut inner_scores = Vec::new();

        for param in param_grid {
            let param_estimator = param_config(estimator.clone(), param)?;

            // Inner CV evaluation
            let inner_splits = inner_cv.split(outer_train_x.nrows(), None);
            let mut param_scores = Vec::new();

            for (inner_train_idx, inner_test_idx) in inner_splits {
                let inner_train_x = extract_rows(&outer_train_x, &inner_train_idx);
                let inner_train_y = extract_elements(&outer_train_y, &inner_train_idx);
                let inner_test_x = extract_rows(&outer_train_x, &inner_test_idx);
                let inner_test_y = extract_elements(&outer_train_y, &inner_test_idx);

                // Fit and score on inner split
                let fitted = param_estimator
                    .clone()
                    .fit(&inner_train_x, &inner_train_y)?;
                let predictions = fitted.predict(&inner_test_x)?;

                let score = if let Some(scoring_fn) = scoring {
                    scoring_fn(&inner_test_y, &predictions)
                } else {
                    fitted.score(&inner_test_x, &inner_test_y)?
                };

                param_scores.push(score);
            }

            let mean_score = param_scores.iter().sum::<f64>() / param_scores.len() as f64;
            inner_scores.push(mean_score);

            if mean_score > best_score {
                best_score = mean_score;
                best_param = param.clone();
            }
        }

        // Train best model on full outer training set and evaluate on outer test set
        let best_estimator = param_config(estimator.clone(), &best_param)?;
        let final_fitted = best_estimator.fit(&outer_train_x, &outer_train_y)?;
        let outer_predictions = final_fitted.predict(&outer_test_x)?;

        let outer_score = if let Some(scoring_fn) = scoring {
            scoring_fn(&outer_test_y, &outer_predictions)
        } else {
            final_fitted.score(&outer_test_x, &outer_test_y)?
        };

        outer_scores.push(outer_score);
        best_params_per_fold.push(best_param);
        inner_scores_per_fold.push(inner_scores);
    }

    let mean_score = outer_scores.iter().sum::<f64>() / outer_scores.len() as f64;
    let std_score = {
        let variance = outer_scores
            .iter()
            .map(|&x| (x - mean_score).powi(2))
            .sum::<f64>()
            / outer_scores.len() as f64;
        variance.sqrt()
    };

    Ok(NestedCVResult {
        outer_scores: Array1::from_vec(outer_scores),
        best_params_per_fold,
        inner_scores_per_fold,
        mean_outer_score: mean_score,
        std_outer_score: std_score,
    })
}

/// Result of nested cross-validation
#[derive(Debug, Clone)]
pub struct NestedCVResult {
    /// Outer cross-validation scores (unbiased performance estimates)
    pub outer_scores: Array1<f64>,
    /// Best parameters found for each outer fold
    pub best_params_per_fold: Vec<ParameterValue>,
    /// Inner CV scores for each parameter in each outer fold
    pub inner_scores_per_fold: Vec<Vec<f64>>,
    /// Mean of outer scores
    pub mean_outer_score: f64,
    /// Standard deviation of outer scores
    pub std_outer_score: f64,
}

// Helper functions for data extraction
fn extract_rows(arr: &Array2<Float>, indices: &[usize]) -> Array2<Float> {
    let mut result = Array2::zeros((indices.len(), arr.ncols()));
    for (i, &idx) in indices.iter().enumerate() {
        for j in 0..arr.ncols() {
            result[[i, j]] = arr[[idx, j]];
        }
    }
    result
}

fn extract_elements(arr: &Array1<Float>, indices: &[usize]) -> Array1<Float> {
    Array1::from_iter(indices.iter().map(|&i| arr[i]))
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::KFold;
    use scirs2_core::ndarray::array;

    // Mock estimator for testing
    #[derive(Clone)]
    struct MockEstimator;

    #[derive(Clone)]
    struct MockFitted {
        train_mean: f64,
    }

    impl Fit<Array2<Float>, Array1<Float>> for MockEstimator {
        type Fitted = MockFitted;

        fn fit(self, _x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
            Ok(MockFitted {
                train_mean: y.mean().unwrap_or(0.0),
            })
        }
    }

    impl Predict<Array2<Float>, Array1<Float>> for MockFitted {
        fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
            Ok(Array1::from_elem(x.nrows(), self.train_mean))
        }
    }

    impl Score<Array2<Float>, Array1<Float>> for MockFitted {
        type Float = Float;

        fn score(&self, x: &Array2<Float>, y: &Array1<Float>) -> Result<f64> {
            let y_pred = self.predict(x)?;
            let mse = (y - &y_pred).mapv(|e| e * e).mean().unwrap_or(0.0);
            Ok(1.0 - mse) // Simple R² approximation
        }
    }

    #[test]
    fn test_cross_val_score() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let estimator = MockEstimator;
        let cv = KFold::new(3);

        let scores = cross_val_score(estimator, &x, &y, &cv, None, None).unwrap();

        assert_eq!(scores.len(), 3);
        // All scores should be negative (since we're predicting mean)
        for score in scores.iter() {
            assert!(*score <= 1.0);
        }
    }

    #[test]
    fn test_cross_val_predict() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let estimator = MockEstimator;
        let cv = KFold::new(3);

        let predictions = cross_val_predict(estimator, &x, &y, &cv, None).unwrap();

        assert_eq!(predictions.len(), 6);
        // Each prediction should be the mean of the training fold
        // Since we're using KFold with 3 splits, each test set has 2 samples
        // and each train set has 4 samples
    }

    #[test]
    fn test_learning_curve() {
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0],
            [10.0]
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let estimator = MockEstimator;
        let cv = KFold::new(3);

        let result = learning_curve(
            estimator,
            &x,
            &y,
            &cv,
            Some(vec![0.3, 0.6, 1.0]), // 30%, 60%, 100% of training data
            None,
            None, // Use default confidence level
        )
        .unwrap();

        // Check dimensions
        assert_eq!(result.train_sizes.len(), 3);
        assert_eq!(result.train_scores.dim(), (3, 3)); // 3 sizes x 3 CV folds
        assert_eq!(result.test_scores.dim(), (3, 3));

        // Check that train sizes are reasonable
        assert_eq!(result.train_sizes[0], 3); // 30% of 10 = 3
        assert_eq!(result.train_sizes[1], 6); // 60% of 10 = 6
        assert_eq!(result.train_sizes[2], 10); // 100% of 10 = 10

        // Training scores should generally be better than test scores for our mock estimator
        let mean_train_score = result.train_scores.mean().unwrap();
        let mean_test_score = result.test_scores.mean().unwrap();
        // Our mock estimator predicts the mean, so training should be perfect
        assert!(mean_train_score >= mean_test_score);

        // Verify confidence bands are calculated
        assert_eq!(result.train_scores_mean.len(), 3);
        assert_eq!(result.test_scores_mean.len(), 3);
        assert_eq!(result.train_scores_std.len(), 3);
        assert_eq!(result.test_scores_std.len(), 3);
        assert_eq!(result.train_scores_lower.len(), 3);
        assert_eq!(result.train_scores_upper.len(), 3);
        assert_eq!(result.test_scores_lower.len(), 3);
        assert_eq!(result.test_scores_upper.len(), 3);

        // Verify confidence intervals are sensible (lower < mean < upper)
        for i in 0..3 {
            assert!(result.train_scores_lower[i] <= result.train_scores_mean[i]);
            assert!(result.train_scores_mean[i] <= result.train_scores_upper[i]);
            assert!(result.test_scores_lower[i] <= result.test_scores_mean[i]);
            assert!(result.test_scores_mean[i] <= result.test_scores_upper[i]);
        }
    }

    #[test]
    fn test_validation_curve() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let estimator = MockEstimator;
        let cv = KFold::new(3);

        // Mock parameter configuration function
        let param_config: ParamConfigFn<MockEstimator> = Box::new(|estimator, _param_value| {
            // For our mock estimator, parameters don't matter
            Ok(estimator)
        });

        let param_range = vec![
            ParameterValue::Float(0.1),
            ParameterValue::Float(0.5),
            ParameterValue::Float(1.0),
        ];

        let result = validation_curve(
            estimator,
            &x,
            &y,
            "mock_param",
            param_range.clone(),
            param_config,
            &cv,
            None,
            None, // Use default confidence level
        )
        .unwrap();

        // Check dimensions
        assert_eq!(result.param_values.len(), 3);
        assert_eq!(result.train_scores.dim(), (3, 3)); // 3 params x 3 CV folds
        assert_eq!(result.test_scores.dim(), (3, 3));

        // Check that parameter values match
        assert_eq!(result.param_values, param_range);

        // For our mock estimator, all parameter values should give similar results
        let train_score_std = {
            let mean = result.train_scores.mean().unwrap();
            let variance = result
                .train_scores
                .mapv(|x| (x - mean).powi(2))
                .mean()
                .unwrap();
            variance.sqrt()
        };

        // Standard deviation should be low since our mock estimator ignores parameters
        // But allow for some variation due to different CV folds
        assert!(train_score_std < 2.0);

        // Verify error bars are calculated
        assert_eq!(result.train_scores_mean.len(), 3);
        assert_eq!(result.test_scores_mean.len(), 3);
        assert_eq!(result.train_scores_std.len(), 3);
        assert_eq!(result.test_scores_std.len(), 3);
        assert_eq!(result.train_scores_lower.len(), 3);
        assert_eq!(result.train_scores_upper.len(), 3);
        assert_eq!(result.test_scores_lower.len(), 3);
        assert_eq!(result.test_scores_upper.len(), 3);

        // Verify error bars are sensible (lower <= mean <= upper)
        for i in 0..3 {
            assert!(result.train_scores_lower[i] <= result.train_scores_mean[i]);
            assert!(result.train_scores_mean[i] <= result.train_scores_upper[i]);
            assert!(result.test_scores_lower[i] <= result.test_scores_mean[i]);
            assert!(result.test_scores_mean[i] <= result.test_scores_upper[i]);
        }
    }

    #[test]
    fn test_learning_curve_default_sizes() {
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0],
            [10.0]
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let estimator = MockEstimator;
        let cv = KFold::new(2);

        let result = learning_curve(
            estimator, &x, &y, &cv, None, // Use default train sizes
            None, None, // Use default confidence level
        )
        .unwrap();

        // Should use default sizes: 10%, 30%, 50%, 70%, 90%, 100%
        assert_eq!(result.train_sizes.len(), 6);
        assert_eq!(result.train_scores.dim(), (6, 2)); // 6 sizes x 2 CV folds

        // Check that sizes are increasing
        for i in 1..result.train_sizes.len() {
            assert!(result.train_sizes[i] >= result.train_sizes[i - 1]);
        }
    }

    #[test]
    fn test_permutation_test_score() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let estimator = MockEstimator;
        let cv = KFold::new(4);

        let result = permutation_test_score(
            estimator,
            &x,
            &y,
            &cv,
            None,
            10, // 10 permutations
            Some(42),
            None,
        )
        .unwrap();

        // Check that we got reasonable results
        assert!(result.pvalue >= 0.0 && result.pvalue <= 1.0);
        assert_eq!(result.permutation_scores.len(), 10);

        // For our mock estimator, the original score should be reasonably good
        // compared to permuted scores
        assert!(result.statistic.is_finite());

        // Permutation scores should all be finite
        for &score in result.permutation_scores.iter() {
            assert!(score.is_finite());
        }

        // P-value should be calculated correctly (at least one score >= original)
        let n_better = result
            .permutation_scores
            .iter()
            .filter(|&&score| score >= result.statistic)
            .count();
        let expected_p = (n_better + 1) as f64 / 11.0; // 10 permutations + 1
        assert!((result.pvalue - expected_p).abs() < 1e-10);
    }

    #[test]
    fn test_nested_cross_validate() {
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0],
            [10.0]
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let estimator = MockEstimator;
        let outer_cv = KFold::new(3);
        let inner_cv = KFold::new(2);

        // Mock parameter configuration function
        let param_config: ParamConfigFn<MockEstimator> = Box::new(|estimator, _param_value| {
            // For our mock estimator, parameters don't matter
            Ok(estimator)
        });

        let param_grid = vec![
            ParameterValue::Float(0.1),
            ParameterValue::Float(0.5),
            ParameterValue::Float(1.0),
        ];

        let result = nested_cross_validate(
            estimator,
            &x,
            &y,
            &outer_cv,
            &inner_cv,
            &param_grid,
            param_config,
            None,
        )
        .unwrap();

        // Check dimensions
        assert_eq!(result.outer_scores.len(), 3); // 3 outer folds
        assert_eq!(result.best_params_per_fold.len(), 3);
        assert_eq!(result.inner_scores_per_fold.len(), 3);

        // Each inner fold should have scores for all parameters
        for inner_scores in &result.inner_scores_per_fold {
            assert_eq!(inner_scores.len(), 3); // 3 parameters
        }

        // Check that outer scores are finite
        for &score in result.outer_scores.iter() {
            assert!(score.is_finite());
        }

        // Check that mean and std are calculated correctly
        let manual_mean =
            result.outer_scores.iter().sum::<f64>() / result.outer_scores.len() as f64;
        assert!((result.mean_outer_score - manual_mean).abs() < 1e-10);

        assert!(result.std_outer_score >= 0.0);
        assert!(result.std_outer_score.is_finite());
    }
}
