//! Common functionality for preprocessing Python bindings
//!
//! This module contains shared imports, types, and utilities used
//! across all preprocessing implementations.

// Re-export commonly used types and traits
pub use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
pub use pyo3::exceptions::PyValueError;
pub use pyo3::prelude::*;
pub use scirs2_core::ndarray::{Array1, Array2};

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

/// Validate input array for transformation with feature count check
pub fn validate_transform_array(x: &Array2<f64>, expected_features: usize) -> PyResult<()> {
    if x.nrows() == 0 {
        return Err(PyValueError::new_err("Input array must not be empty"));
    }

    if x.ncols() == 0 {
        return Err(PyValueError::new_err(
            "Input array must have at least one feature",
        ));
    }

    // Check feature dimensions
    if x.ncols() != expected_features {
        return Err(PyValueError::new_err(format!(
            "Input has {} features, but transformer was fitted with {} features",
            x.ncols(),
            expected_features
        )));
    }

    // Check for non-finite values
    if !x.iter().all(|&val| val.is_finite()) {
        return Err(PyValueError::new_err(
            "Input array contains non-finite values",
        ));
    }

    Ok(())
}

/// Convert a read-only NumPy array view into an owned SciRS2 ndarray Array2
pub fn pyarray_to_core_array2(py_array: &PyReadonlyArray2<f64>) -> PyResult<Array2<f64>> {
    let array_view = py_array.as_array();
    let shape = array_view.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("Expected a 2D array"));
    }
    let rows = shape[0];
    let cols = shape[1];
    Array2::from_shape_vec((rows, cols), array_view.iter().cloned().collect())
        .map_err(|_| PyValueError::new_err("Failed to convert NumPy array to ndarray"))
}

/// Convert an ndarray Array1 into a Python-owned NumPy array object
pub fn core_array1_to_py(py: Python<'_>, array: &Array1<f64>) -> Py<PyArray1<f64>> {
    let numpy_array = numpy::ndarray::Array1::from_vec(array.to_vec());
    PyArray1::from_owned_array(py, numpy_array).into()
}

/// Convert an ndarray Array2 into a Python-owned NumPy array object
pub fn core_array2_to_py(py: Python<'_>, array: &Array2<f64>) -> PyResult<Py<PyArray2<f64>>> {
    let (rows, cols) = array.dim();
    let data: Vec<f64> = array.iter().cloned().collect();
    let numpy_array = numpy::ndarray::Array2::from_shape_vec((rows, cols), data)
        .map_err(|_| PyValueError::new_err("Failed to convert ndarray to NumPy array"))?;
    Ok(PyArray2::from_owned_array(py, numpy_array).into())
}
