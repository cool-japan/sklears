//! Python bindings for MinMaxScaler
//!
//! This module provides Python bindings for sklears MinMaxScaler,
//! offering scikit-learn compatible min-max normalization.

use super::common::*;
use sklears::preprocessing::MinMaxScaler;

/// Python wrapper for MinMaxScaler
#[pyclass(name = "MinMaxScaler")]
pub struct PyMinMaxScaler {
    scaler: Option<MinMaxScaler<f64>>,
    fitted_scaler: Option<MinMaxScaler<f64>>,
}

#[pymethods]
impl PyMinMaxScaler {
    #[new]
    #[pyo3(signature = (feature_range=(0.0, 1.0), copy=true, clip=false))]
    fn new(feature_range: (f64, f64), copy: bool, clip: bool) -> Self {
        let scaler = MinMaxScaler::builder()
            .feature_range(feature_range)
            .copy(copy)
            .clip(clip)
            .build();

        Self {
            scaler: Some(scaler),
            fitted_scaler: None,
        }
    }

    /// Fit the scaler to the data
    fn fit(&mut self, x: PyReadonlyArray2<f64>) -> PyResult<()> {
        let x_array = x.as_array().to_owned();
        validate_fit_array(&x_array)?;

        let scaler = self
            .scaler
            .take()
            .ok_or_else(|| PyValueError::new_err("Scaler already fitted or invalid state"))?;

        match scaler.fit(&x_array) {
            Ok(fitted) => {
                self.fitted_scaler = Some(fitted);
                Ok(())
            }
            Err(e) => Err(PyValueError::new_err(format!(
                "Failed to fit scaler: {}",
                e
            ))),
        }
    }

    /// Transform the data using the fitted scaler
    fn transform(&self, x: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray2<f64>>> {
        let fitted = self
            .fitted_scaler
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Scaler not fitted. Call fit() first."))?;

        let x_array = x.as_array();
        validate_transform_array(&x_array)?;

        match fitted.transform(&x_array) {
            Ok(transformed) => {
                Python::with_gil(|py| Ok(PyArray2::from_array(py, &transformed).to_owned()))
            }
            Err(e) => Err(PyValueError::new_err(format!(
                "Transformation failed: {}",
                e
            ))),
        }
    }

    /// Fit the scaler and transform the data in one step
    fn fit_transform(&mut self, x: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray2<f64>>> {
        self.fit(x)?;
        self.transform(x)
    }

    /// Inverse transform the data
    fn inverse_transform(&self, x: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray2<f64>>> {
        let fitted = self
            .fitted_scaler
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Scaler not fitted. Call fit() first."))?;

        let x_array = x.as_array();
        validate_transform_array(&x_array)?;

        match fitted.inverse_transform(&x_array) {
            Ok(inverse_transformed) => {
                Python::with_gil(|py| Ok(PyArray2::from_array(py, &inverse_transformed).to_owned()))
            }
            Err(e) => Err(PyValueError::new_err(format!(
                "Inverse transformation failed: {}",
                e
            ))),
        }
    }

    /// Get the minimum values for each feature
    #[getter]
    fn data_min_(&self) -> PyResult<Py<PyArray1<f64>>> {
        let fitted = self
            .fitted_scaler
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Scaler not fitted. Call fit() first."))?;

        Python::with_gil(|py| {
            let data_min = fitted.data_min().unwrap_or_else(|| Array1::zeros(0));
            Ok(PyArray1::from_array(py, &data_min).to_owned())
        })
    }

    /// Get the maximum values for each feature
    #[getter]
    fn data_max_(&self) -> PyResult<Py<PyArray1<f64>>> {
        let fitted = self
            .fitted_scaler
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Scaler not fitted. Call fit() first."))?;

        Python::with_gil(|py| {
            let data_max = fitted.data_max().unwrap_or_else(|| Array1::zeros(0));
            Ok(PyArray1::from_array(py, &data_max).to_owned())
        })
    }

    /// Get the range (max - min) for each feature
    #[getter]
    fn data_range_(&self) -> PyResult<Py<PyArray1<f64>>> {
        let fitted = self
            .fitted_scaler
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Scaler not fitted. Call fit() first."))?;

        Python::with_gil(|py| {
            let data_range = fitted.data_range().unwrap_or_else(|| Array1::zeros(0));
            Ok(PyArray1::from_array(py, &data_range).to_owned())
        })
    }

    /// Get the scaling factor for each feature
    #[getter]
    fn scale_(&self) -> PyResult<Py<PyArray1<f64>>> {
        let fitted = self
            .fitted_scaler
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Scaler not fitted. Call fit() first."))?;

        Python::with_gil(|py| {
            let scale = fitted.scale().unwrap_or_else(|| Array1::zeros(0));
            Ok(PyArray1::from_array(py, &scale).to_owned())
        })
    }
}
