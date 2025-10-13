//! Common functionality for preprocessing Python bindings
//!
//! This module contains shared imports, types, and utilities used
//! across all preprocessing implementations.

// Re-export commonly used types and traits
pub use scirs2_core::ndarray::{Array1, Array2};
pub use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
pub use pyo3::exceptions::PyValueError;
pub use pyo3::prelude::*;

// Performance optimization imports
#[cfg(feature = "parallel")]
pub use rayon::prelude::*;

/// Common error type for preprocessing operations
pub type PreprocessingResult<T> = Result<T, PyValueError>;

/// Validate input array for fitting transformers
pub fn validate_fit_array(x: &Array2<f64>) -> PyResult<()> {
    if x.nrows() == 0 {
        return Err(PyValueError::new_err("Input array must not be empty"));
    }

    if x.ncols() == 0 {
        return Err(PyValueError::new_err(
            "Input array must have at least one feature",
        ));
    }

    // Check for non-finite values
    if !x.iter().all(|&val| val.is_finite()) {
        return Err(PyValueError::new_err(
            "Input array contains non-finite values",
        ));
    }

    Ok(())
}

/// Validate input array for transformation
pub fn validate_transform_array(x: &Array2<f64>) -> PyResult<()> {
    if x.nrows() == 0 {
        return Err(PyValueError::new_err("Input array must not be empty"));
    }

    if x.ncols() == 0 {
        return Err(PyValueError::new_err(
            "Input array must have at least one feature",
        ));
    }

    // Check for non-finite values
    if !x.iter().all(|&val| val.is_finite()) {
        return Err(PyValueError::new_err(
            "Input array contains non-finite values",
        ));
    }

    Ok(())
}

/// Validate input array dimensions match expected features
pub fn validate_feature_dimensions(x: &Array2<f64>, expected_features: usize) -> PyResult<()> {
    if x.ncols() != expected_features {
        return Err(PyValueError::new_err(format!(
            "Input has {} features, but transformer was fitted with {} features",
            x.ncols(),
            expected_features
        )));
    }

    Ok(())
}
