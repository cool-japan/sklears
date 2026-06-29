//! Python bindings for regression metrics

use super::common::*;
use sklears_metrics::regression::{
    mean_absolute_error as skl_mae, mean_squared_error as skl_mse, r2_score as skl_r2,
};

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
    let _ = (sample_weight, multioutput);
    let yt = y_true.as_array().to_owned();
    let yp = y_pred.as_array().to_owned();

    validate_arrays_same_length(&yt, &yp)?;

    match skl_mse(&yt, &yp) {
        Ok(mse) => Ok(if squared { mse } else { mse.sqrt() }),
        Err(e) => Err(PyValueError::new_err(format!("mean_squared_error: {}", e))),
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
    let _ = (sample_weight, multioutput);
    let yt = y_true.as_array().to_owned();
    let yp = y_pred.as_array().to_owned();

    validate_arrays_same_length(&yt, &yp)?;

    match skl_mae(&yt, &yp) {
        Ok(mae) => Ok(mae),
        Err(e) => Err(PyValueError::new_err(format!("mean_absolute_error: {}", e))),
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
    let _ = (sample_weight, multioutput);
    let yt = y_true.as_array().to_owned();
    let yp = y_pred.as_array().to_owned();

    validate_arrays_same_length(&yt, &yp)?;

    match skl_r2(&yt, &yp) {
        Ok(r2) => Ok(r2),
        Err(e) => Err(PyValueError::new_err(format!("r2_score: {}", e))),
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
    let _ = (sample_weight, multioutput);
    let yt = y_true.as_array().to_owned();
    let yp = y_pred.as_array().to_owned();

    validate_arrays_same_length(&yt, &yp)?;

    if yt.iter().any(|&x| x < 0.0) || yp.iter().any(|&x| x < 0.0) {
        return Err(PyValueError::new_err(
            "Mean Squared Logarithmic Error cannot be used when targets contain negative values.",
        ));
    }

    let log_true = yt.mapv(|x| (x + 1.0).ln());
    let log_pred = yp.mapv(|x| (x + 1.0).ln());
    let sq_errors = (&log_true - &log_pred).mapv(|x| x * x);
    let msle = sq_errors.mean().unwrap_or(0.0);

    Ok(if squared { msle } else { msle.sqrt() })
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
    let _ = multioutput;
    let yt = y_true.as_array().to_owned();
    let yp = y_pred.as_array().to_owned();

    validate_arrays_same_length(&yt, &yp)?;

    if sample_weight.is_some() {
        return Err(PyValueError::new_err(
            "median_absolute_error does not support sample weights",
        ));
    }

    let mut errors: Vec<f64> = (&yt - &yp).mapv(|x| x.abs()).to_vec();
    errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = errors.len();
    let median = if n.is_multiple_of(2) {
        (errors[n / 2 - 1] + errors[n / 2]) / 2.0
    } else {
        errors[n / 2]
    };

    Ok(median)
}
