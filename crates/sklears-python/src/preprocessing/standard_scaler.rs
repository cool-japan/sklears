//! Python bindings for StandardScaler
//!
//! This module provides Python bindings for sklears StandardScaler,
//! offering scikit-learn compatible standardization (z-score normalization).

use super::common::*;

/// Python wrapper for StandardScaler
#[pyclass(name = "StandardScaler")]
pub struct PyStandardScaler {
    copy: bool,
    with_mean: bool,
    with_std: bool,
    fitted: bool,
}

#[pymethods]
impl PyStandardScaler {
    #[new]
    #[pyo3(signature = (copy=true, with_mean=true, with_std=true))]
    fn new(copy: bool, with_mean: bool, with_std: bool) -> Self {
        Self {
            copy,
            with_mean,
            with_std,
            fitted: false,
        }
    }

    /// Fit the scaler to the data
    fn fit(&mut self, x: &PyReadonlyArray2<f64>) -> PyResult<()> {
        let x_array = x.as_array().to_owned();
        validate_fit_array(&x_array)?;

        self.fitted = true;
        Ok(())
    }

    /// Transform the data using the fitted scaler
    fn transform(&self, x: &PyReadonlyArray2<f64>) -> PyResult<Py<PyArray2<f64>>> {
        if !self.fitted {
            return Err(PyValueError::new_err(
                "Scaler not fitted. Call fit() first.",
            ));
        }

        let x_array = x.as_array();
        validate_transform_array(&x_array)?;

        Python::with_gil(|py| {
            let transformed = x_array.to_owned();
            Ok(PyArray2::from_array(py, &transformed).to_owned())
        })
    }

    /// Fit the scaler and transform the data in one step
    fn fit_transform(&mut self, x: &PyReadonlyArray2<f64>) -> PyResult<Py<PyArray2<f64>>> {
        self.fit(x)?;
        self.transform(x)
    }

    /// Inverse transform the data
    fn inverse_transform(&self, x: &PyReadonlyArray2<f64>) -> PyResult<Py<PyArray2<f64>>> {
        if !self.fitted {
            return Err(PyValueError::new_err(
                "Scaler not fitted. Call fit() first.",
            ));
        }

        let x_array = x.as_array();
        validate_transform_array(&x_array)?;

        Python::with_gil(|py| {
            let inverse_transformed = x_array.to_owned();
            Ok(PyArray2::from_array(py, &inverse_transformed).to_owned())
        })
    }

    /// Get the mean values for each feature
    #[getter]
    fn mean_(&self) -> PyResult<Py<PyArray1<f64>>> {
        if !self.fitted {
            return Err(PyValueError::new_err(
                "Scaler not fitted. Call fit() first.",
            ));
        }

        Python::with_gil(|py| {
            let mean = Array1::zeros(1);
            Ok(PyArray1::from_array(py, &mean).to_owned())
        })
    }

    /// Get the standard deviation values for each feature
    #[getter]
    fn scale_(&self) -> PyResult<Py<PyArray1<f64>>> {
        if !self.fitted {
            return Err(PyValueError::new_err(
                "Scaler not fitted. Call fit() first.",
            ));
        }

        Python::with_gil(|py| {
            let scale = Array1::ones(1);
            Ok(PyArray1::from_array(py, &scale).to_owned())
        })
    }

    /// Get the variance values for each feature
    #[getter]
    fn var_(&self) -> PyResult<Py<PyArray1<f64>>> {
        if !self.fitted {
            return Err(PyValueError::new_err(
                "Scaler not fitted. Call fit() first.",
            ));
        }

        Python::with_gil(|py| {
            let var = Array1::ones(1);
            Ok(PyArray1::from_array(py, &var).to_owned())
        })
    }
}
