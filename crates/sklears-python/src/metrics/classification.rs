//! Python bindings for classification metrics
//!
//! This module provides Python bindings for classification evaluation metrics,
//! offering scikit-learn compatible interfaces.

use super::common::*;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use sklears::metrics;

/// Calculate accuracy score for classification
#[pyfunction]
#[pyo3(signature = (y_true, y_pred, normalize=true, sample_weight=None))]
pub fn accuracy_score(
    y_true: PyReadonlyArray1<i32>,
    y_pred: PyReadonlyArray1<i32>,
    normalize: bool,
    sample_weight: Option<PyReadonlyArray1<f64>>,
) -> PyResult<f64> {
    let y_true_array = y_true.as_array().to_vec();
    let y_pred_array = y_pred.as_array().to_vec();
    let weights = sample_weight.map(|w| w.as_array().to_owned());

    validate_int_arrays_same_length(&y_true_array, &y_pred_array)?;
    validate_sample_weight(&weights, y_true_array.len())?;

    match metrics::accuracy_score(&y_true_array, &y_pred_array, weights.as_ref()) {
        Ok(accuracy) => Ok(accuracy),
        Err(e) => Err(PyValueError::new_err(format!(
            "Failed to calculate accuracy: {}",
            e
        ))),
    }
}

/// Calculate precision score for classification
#[pyfunction]
#[pyo3(signature = (y_true, y_pred, labels=None, pos_label=1, average="binary", sample_weight=None, zero_division="warn"))]
pub fn precision_score(
    y_true: PyReadonlyArray1<i32>,
    y_pred: PyReadonlyArray1<i32>,
    labels: Option<PyReadonlyArray1<i32>>,
    pos_label: i32,
    average: &str,
    sample_weight: Option<PyReadonlyArray1<f64>>,
    zero_division: &str,
) -> PyResult<f64> {
    let y_true_array = y_true.as_array().to_vec();
    let y_pred_array = y_pred.as_array().to_vec();
    let labels_array = labels.map(|l| l.as_array().to_owned());
    let weights = sample_weight.map(|w| w.as_array().to_owned());

    validate_int_arrays_same_length(&y_true_array, &y_pred_array)?;
    validate_sample_weight(&weights, y_true_array.len())?;

    match metrics::precision_score(
        &y_true_array,
        &y_pred_array,
        labels_array.as_ref(),
        Some(pos_label),
        average,
        weights.as_ref(),
    ) {
        Ok(precision) => Ok(precision),
        Err(e) => Err(PyValueError::new_err(format!(
            "Failed to calculate precision: {}",
            e
        ))),
    }
}

/// Calculate recall score for classification
#[pyfunction]
#[pyo3(signature = (y_true, y_pred, labels=None, pos_label=1, average="binary", sample_weight=None, zero_division="warn"))]
pub fn recall_score(
    y_true: PyReadonlyArray1<i32>,
    y_pred: PyReadonlyArray1<i32>,
    labels: Option<PyReadonlyArray1<i32>>,
    pos_label: i32,
    average: &str,
    sample_weight: Option<PyReadonlyArray1<f64>>,
    zero_division: &str,
) -> PyResult<f64> {
    let y_true_array = y_true.as_array().to_vec();
    let y_pred_array = y_pred.as_array().to_vec();
    let labels_array = labels.map(|l| l.as_array().to_owned());
    let weights = sample_weight.map(|w| w.as_array().to_owned());

    validate_int_arrays_same_length(&y_true_array, &y_pred_array)?;
    validate_sample_weight(&weights, y_true_array.len())?;

    match metrics::recall_score(
        &y_true_array,
        &y_pred_array,
        labels_array.as_ref(),
        Some(pos_label),
        average,
        weights.as_ref(),
    ) {
        Ok(recall) => Ok(recall),
        Err(e) => Err(PyValueError::new_err(format!(
            "Failed to calculate recall: {}",
            e
        ))),
    }
}

/// Calculate F1 score for classification
#[pyfunction]
#[pyo3(signature = (y_true, y_pred, labels=None, pos_label=1, average="binary", sample_weight=None, zero_division="warn"))]
pub fn f1_score(
    y_true: PyReadonlyArray1<i32>,
    y_pred: PyReadonlyArray1<i32>,
    labels: Option<PyReadonlyArray1<i32>>,
    pos_label: i32,
    average: &str,
    sample_weight: Option<PyReadonlyArray1<f64>>,
    zero_division: &str,
) -> PyResult<f64> {
    let y_true_array = y_true.as_array().to_vec();
    let y_pred_array = y_pred.as_array().to_vec();
    let labels_array = labels.map(|l| l.as_array().to_owned());
    let weights = sample_weight.map(|w| w.as_array().to_owned());

    validate_int_arrays_same_length(&y_true_array, &y_pred_array)?;
    validate_sample_weight(&weights, y_true_array.len())?;

    match metrics::f1_score(
        &y_true_array,
        &y_pred_array,
        labels_array.as_ref(),
        Some(pos_label),
        average,
        weights.as_ref(),
    ) {
        Ok(f1) => Ok(f1),
        Err(e) => Err(PyValueError::new_err(format!(
            "Failed to calculate F1 score: {}",
            e
        ))),
    }
}

/// Calculate confusion matrix
#[pyfunction]
#[pyo3(signature = (y_true, y_pred, labels=None, sample_weight=None, normalize=None))]
pub fn confusion_matrix(
    py: Python,
    y_true: PyReadonlyArray1<i32>,
    y_pred: PyReadonlyArray1<i32>,
    labels: Option<PyReadonlyArray1<i32>>,
    sample_weight: Option<PyReadonlyArray1<f64>>,
    normalize: Option<&str>,
) -> PyResult<Py<PyArray2<i32>>> {
    let y_true_array = y_true.as_array().to_vec();
    let y_pred_array = y_pred.as_array().to_vec();
    let labels_array = labels.map(|l| l.as_array().to_owned());
    let weights = sample_weight.map(|w| w.as_array().to_owned());

    validate_int_arrays_same_length(&y_true_array, &y_pred_array)?;
    validate_sample_weight(&weights, y_true_array.len())?;

    match metrics::confusion_matrix(
        &y_true_array,
        &y_pred_array,
        labels_array.as_ref(),
        weights.as_ref(),
    ) {
        Ok(cm) => Ok(PyArray2::from_array(py, &cm).to_owned()),
        Err(e) => Err(PyValueError::new_err(format!(
            "Failed to calculate confusion matrix: {}",
            e
        ))),
    }
}

/// Calculate classification report (returns a dictionary-like structure)
#[pyfunction]
#[pyo3(signature = (y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=true, zero_division="warn"))]
pub fn classification_report(
    py: Python,
    y_true: PyReadonlyArray1<i32>,
    y_pred: PyReadonlyArray1<i32>,
    labels: Option<PyReadonlyArray1<i32>>,
    target_names: Option<Vec<String>>,
    sample_weight: Option<PyReadonlyArray1<f64>>,
    digits: usize,
    output_dict: bool,
    zero_division: &str,
) -> PyResult<PyObject> {
    let y_true_array = y_true.as_array().to_vec();
    let y_pred_array = y_pred.as_array().to_vec();

    validate_int_arrays_same_length(&y_true_array, &y_pred_array)?;

    if !output_dict {
        return Err(PyValueError::new_err(
            "String output format not supported in this implementation. Use output_dict=True.",
        ));
    }

    // Create a simple classification report as a Python dictionary
    let mut report = std::collections::HashMap::new();

    // Calculate basic metrics for binary classification case
    if let (Ok(precision), Ok(recall), Ok(f1)) = (
        precision_score(
            y_true,
            y_pred,
            labels,
            1,
            "binary",
            sample_weight,
            zero_division,
        ),
        recall_score(
            y_true,
            y_pred,
            labels,
            1,
            "binary",
            sample_weight,
            zero_division,
        ),
        f1_score(
            y_true,
            y_pred,
            labels,
            1,
            "binary",
            sample_weight,
            zero_division,
        ),
    ) {
        let mut class_metrics = std::collections::HashMap::new();
        class_metrics.insert("precision".to_string(), precision.to_object(py));
        class_metrics.insert("recall".to_string(), recall.to_object(py));
        class_metrics.insert("f1-score".to_string(), f1.to_object(py));
        class_metrics.insert("support".to_string(), y_true_array.len().to_object(py));

        report.insert("1".to_string(), class_metrics.to_object(py));
    }

    Ok(report.to_object(py))
}
