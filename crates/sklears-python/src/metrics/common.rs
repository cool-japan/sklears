//! Common functionality for metrics Python bindings
//!
//! This module contains shared imports, types, and utilities used
//! across all metric implementations.

// Re-export commonly used types and traits
pub use scirs2_core::ndarray::Array1;
pub use numpy::{PyReadonlyArray1, PyReadonlyArray2};
pub use pyo3::exceptions::PyValueError;
pub use pyo3::prelude::*;

/// Common error type for metric operations
pub type MetricResult<T> = Result<T, PyValueError>;

/// Validate that prediction and true value arrays have the same length
pub fn validate_arrays_same_length(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> PyResult<()> {
    if y_true.len() != y_pred.len() {
        return Err(PyValueError::new_err(format!(
            "y_true and y_pred must have the same length: {} vs {}",
            y_true.len(),
            y_pred.len()
        )));
    }

    if y_true.is_empty() {
        return Err(PyValueError::new_err("y_true and y_pred must not be empty"));
    }

    Ok(())
}

/// Validate that prediction and true value arrays have the same length for integer arrays
pub fn validate_int_arrays_same_length(y_true: &[i32], y_pred: &[i32]) -> PyResult<()> {
    if y_true.len() != y_pred.len() {
        return Err(PyValueError::new_err(format!(
            "y_true and y_pred must have the same length: {} vs {}",
            y_true.len(),
            y_pred.len()
        )));
    }

    if y_true.is_empty() {
        return Err(PyValueError::new_err("y_true and y_pred must not be empty"));
    }

    Ok(())
}

/// Validate sample weights if provided
pub fn validate_sample_weight(
    sample_weight: &Option<Array1<f64>>,
    n_samples: usize,
) -> PyResult<()> {
    if let Some(weights) = sample_weight {
        if weights.len() != n_samples {
            return Err(PyValueError::new_err(format!(
                "sample_weight must have the same length as y_true: {} vs {}",
                weights.len(),
                n_samples
            )));
        }

        if weights.iter().any(|&w| w < 0.0 || !w.is_finite()) {
            return Err(PyValueError::new_err(
                "sample_weight must contain non-negative finite values",
            ));
        }
    }

    Ok(())
}

/// Apply sample weights to values if provided
pub fn apply_sample_weight(
    values: &Array1<f64>,
    sample_weight: &Option<Array1<f64>>,
) -> Array1<f64> {
    match sample_weight {
        Some(weights) => values * weights,
        None => values.clone(),
    }
}
