//! Python bindings for classification metrics

use super::common::*;
use numpy::{PyArray2, PyReadonlyArray1};
use scirs2_core::ndarray::Array1;
use sklears_metrics::basic_metrics::{
    accuracy_score as skl_accuracy, confusion_matrix as skl_confusion_matrix, f1_score as skl_f1,
    precision_score as skl_precision, recall_score as skl_recall,
};
use std::collections::HashMap;

/// Calculate accuracy score for classification
#[pyfunction]
#[pyo3(signature = (y_true, y_pred, normalize=true, sample_weight=None))]
pub fn accuracy_score(
    y_true: PyReadonlyArray1<i32>,
    y_pred: PyReadonlyArray1<i32>,
    normalize: bool,
    sample_weight: Option<PyReadonlyArray1<f64>>,
) -> PyResult<f64> {
    let _ = sample_weight;
    let yt = Array1::from_vec(y_true.as_array().to_vec());
    let yp = Array1::from_vec(y_pred.as_array().to_vec());

    validate_int_arrays_same_length(yt.as_slice().unwrap(), yp.as_slice().unwrap())?;

    match skl_accuracy(&yt, &yp) {
        Ok(acc) => Ok(if normalize {
            acc
        } else {
            acc * yt.len() as f64
        }),
        Err(e) => Err(PyValueError::new_err(format!("accuracy_score: {}", e))),
    }
}

/// Calculate precision score for binary classification
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
    let _ = (labels, average, sample_weight, zero_division);
    let yt = Array1::from_vec(y_true.as_array().to_vec());
    let yp = Array1::from_vec(y_pred.as_array().to_vec());

    validate_int_arrays_same_length(yt.as_slice().unwrap(), yp.as_slice().unwrap())?;

    match skl_precision(&yt, &yp, Some(pos_label)) {
        Ok(v) => Ok(v),
        Err(e) => Err(PyValueError::new_err(format!("precision_score: {}", e))),
    }
}

/// Calculate recall score for binary classification
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
    let _ = (labels, average, sample_weight, zero_division);
    let yt = Array1::from_vec(y_true.as_array().to_vec());
    let yp = Array1::from_vec(y_pred.as_array().to_vec());

    validate_int_arrays_same_length(yt.as_slice().unwrap(), yp.as_slice().unwrap())?;

    match skl_recall(&yt, &yp, Some(pos_label)) {
        Ok(v) => Ok(v),
        Err(e) => Err(PyValueError::new_err(format!("recall_score: {}", e))),
    }
}

/// Calculate F1 score for binary classification
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
    let _ = (labels, average, sample_weight, zero_division);
    let yt = Array1::from_vec(y_true.as_array().to_vec());
    let yp = Array1::from_vec(y_pred.as_array().to_vec());

    validate_int_arrays_same_length(yt.as_slice().unwrap(), yp.as_slice().unwrap())?;

    match skl_f1(&yt, &yp, Some(pos_label)) {
        Ok(v) => Ok(v),
        Err(e) => Err(PyValueError::new_err(format!("f1_score: {}", e))),
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
) -> PyResult<Py<PyArray2<i64>>> {
    let _ = (labels, sample_weight, normalize);
    let yt = Array1::from_vec(y_true.as_array().to_vec());
    let yp = Array1::from_vec(y_pred.as_array().to_vec());

    validate_int_arrays_same_length(yt.as_slice().unwrap(), yp.as_slice().unwrap())?;

    match skl_confusion_matrix(&yt, &yp) {
        Ok(cm) => {
            let cm_i64 = cm.mapv(|v| v as i64);
            Ok(PyArray2::from_array(py, &cm_i64).unbind())
        }
        Err(e) => Err(PyValueError::new_err(format!("confusion_matrix: {}", e))),
    }
}

/// Calculate classification report (returns a nested dict when output_dict=True).
/// Note: `digits` and `zero_division` are accepted for API compatibility but ignored.
#[pyfunction]
#[pyo3(signature = (y_true, y_pred, labels=None, target_names=None, sample_weight=None, output_dict=true))]
pub fn classification_report(
    y_true: PyReadonlyArray1<i32>,
    y_pred: PyReadonlyArray1<i32>,
    labels: Option<PyReadonlyArray1<i32>>,
    target_names: Option<Vec<String>>,
    sample_weight: Option<PyReadonlyArray1<f64>>,
    output_dict: bool,
) -> PyResult<HashMap<String, HashMap<String, f64>>> {
    let _ = (labels, target_names, sample_weight);
    if !output_dict {
        return Err(PyValueError::new_err(
            "String output not supported; use output_dict=True.",
        ));
    }

    let yt = Array1::from_vec(y_true.as_array().to_vec());
    let yp = Array1::from_vec(y_pred.as_array().to_vec());

    validate_int_arrays_same_length(yt.as_slice().unwrap(), yp.as_slice().unwrap())?;

    let pos_label = *yt.iter().max().unwrap_or(&1);
    let support = yt.len() as f64;

    let precision = skl_precision(&yt, &yp, Some(pos_label)).unwrap_or(0.0);
    let recall = skl_recall(&yt, &yp, Some(pos_label)).unwrap_or(0.0);
    let f1 = skl_f1(&yt, &yp, Some(pos_label)).unwrap_or(0.0);

    let mut class_entry = HashMap::new();
    class_entry.insert("precision".to_string(), precision);
    class_entry.insert("recall".to_string(), recall);
    class_entry.insert("f1-score".to_string(), f1);
    class_entry.insert("support".to_string(), support);

    let mut report = HashMap::new();
    report.insert(pos_label.to_string(), class_entry);
    Ok(report)
}
