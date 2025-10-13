//! Python bindings for regression metrics
//!
//! This module provides Python bindings for regression evaluation metrics,
//! offering scikit-learn compatible interfaces.

use super::common::*;
use sklears::metrics;

/// Calculate mean squared error
#[pyfunction]
#[pyo3(signature = (y_true, y_pred, sample_weight=None, multioutput="uniform_average", squared=true))]
pub fn mean_squared_error(
    y_true: PyReadonlyArray1<f64>,
    y_pred: PyReadonlyArray1<f64>,
    sample_weight: Option<PyReadonlyArray1<f64>>,
    multioutput: &str,
    squared: bool,
) -> PyResult<f64> {
    let y_true_array = y_true.as_array().to_owned();
    let y_pred_array = y_pred.as_array().to_owned();
    let weights = sample_weight.map(|w| w.as_array().to_owned());

    validate_arrays_same_length(&y_true_array, &y_pred_array)?;
    validate_sample_weight(&weights, y_true_array.len())?;

    match metrics::mean_squared_error(&y_true_array, &y_pred_array, weights.as_ref()) {
        Ok(mse) => {
            let result = if squared { mse } else { mse.sqrt() };
            Ok(result)
        }
        Err(e) => Err(PyValueError::new_err(format!(
            "Failed to calculate MSE: {}",
            e
        ))),
    }
}

/// Calculate mean absolute error
#[pyfunction]
#[pyo3(signature = (y_true, y_pred, sample_weight=None, multioutput="uniform_average"))]
pub fn mean_absolute_error(
    y_true: PyReadonlyArray1<f64>,
    y_pred: PyReadonlyArray1<f64>,
    sample_weight: Option<PyReadonlyArray1<f64>>,
    multioutput: &str,
) -> PyResult<f64> {
    let y_true_array = y_true.as_array().to_owned();
    let y_pred_array = y_pred.as_array().to_owned();
    let weights = sample_weight.map(|w| w.as_array().to_owned());

    validate_arrays_same_length(&y_true_array, &y_pred_array)?;
    validate_sample_weight(&weights, y_true_array.len())?;

    match metrics::mean_absolute_error(&y_true_array, &y_pred_array, weights.as_ref()) {
        Ok(mae) => Ok(mae),
        Err(e) => Err(PyValueError::new_err(format!(
            "Failed to calculate MAE: {}",
            e
        ))),
    }
}

/// Calculate R² (coefficient of determination) score
#[pyfunction]
#[pyo3(signature = (y_true, y_pred, sample_weight=None, multioutput="uniform_average"))]
pub fn r2_score(
    y_true: PyReadonlyArray1<f64>,
    y_pred: PyReadonlyArray1<f64>,
    sample_weight: Option<PyReadonlyArray1<f64>>,
    multioutput: &str,
) -> PyResult<f64> {
    let y_true_array = y_true.as_array().to_owned();
    let y_pred_array = y_pred.as_array().to_owned();
    let weights = sample_weight.map(|w| w.as_array().to_owned());

    validate_arrays_same_length(&y_true_array, &y_pred_array)?;
    validate_sample_weight(&weights, y_true_array.len())?;

    match metrics::r2_score(&y_true_array, &y_pred_array, weights.as_ref()) {
        Ok(r2) => Ok(r2),
        Err(e) => Err(PyValueError::new_err(format!(
            "Failed to calculate R²: {}",
            e
        ))),
    }
}

/// Calculate mean squared logarithmic error
#[pyfunction]
#[pyo3(signature = (y_true, y_pred, sample_weight=None, multioutput="uniform_average", squared=true))]
pub fn mean_squared_log_error(
    y_true: PyReadonlyArray1<f64>,
    y_pred: PyReadonlyArray1<f64>,
    sample_weight: Option<PyReadonlyArray1<f64>>,
    multioutput: &str,
    squared: bool,
) -> PyResult<f64> {
    let y_true_array = y_true.as_array().to_owned();
    let y_pred_array = y_pred.as_array().to_owned();
    let weights = sample_weight.map(|w| w.as_array().to_owned());

    validate_arrays_same_length(&y_true_array, &y_pred_array)?;
    validate_sample_weight(&weights, y_true_array.len())?;

    // Validate that all values are non-negative
    if y_true_array.iter().any(|&x| x < 0.0) || y_pred_array.iter().any(|&x| x < 0.0) {
        return Err(PyValueError::new_err(
            "Mean Squared Logarithmic Error cannot be used when targets contain negative values.",
        ));
    }

    // Calculate MSLE manually since it might not be in sklears::metrics
    let log_true = y_true_array.mapv(|x| (x + 1.0).ln());
    let log_pred = y_pred_array.mapv(|x| (x + 1.0).ln());

    let squared_log_errors = (&log_true - &log_pred).mapv(|x| x * x);

    let msle = match weights {
        Some(ref w) => {
            let weighted_errors = apply_sample_weight(&squared_log_errors, &Some(w.clone()));
            weighted_errors.sum() / w.sum()
        }
        None => squared_log_errors.mean().unwrap_or(0.0),
    };

    let result = if squared { msle } else { msle.sqrt() };
    Ok(result)
}

/// Calculate median absolute error
#[pyfunction]
#[pyo3(signature = (y_true, y_pred, multioutput="uniform_average", sample_weight=None))]
pub fn median_absolute_error(
    y_true: PyReadonlyArray1<f64>,
    y_pred: PyReadonlyArray1<f64>,
    multioutput: &str,
    sample_weight: Option<PyReadonlyArray1<f64>>,
) -> PyResult<f64> {
    let y_true_array = y_true.as_array().to_owned();
    let y_pred_array = y_pred.as_array().to_owned();

    validate_arrays_same_length(&y_true_array, &y_pred_array)?;

    if sample_weight.is_some() {
        return Err(PyValueError::new_err(
            "median_absolute_error does not support sample weights",
        ));
    }

    let absolute_errors = (&y_true_array - &y_pred_array).mapv(|x| x.abs());
    let mut errors_vec: Vec<f64> = absolute_errors.to_vec();
    errors_vec.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = errors_vec.len();
    let median = if n % 2 == 0 {
        (errors_vec[n / 2 - 1] + errors_vec[n / 2]) / 2.0
    } else {
        errors_vec[n / 2]
    };

    Ok(median)
}
