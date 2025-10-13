//! Python bindings for LabelEncoder
//!
//! This module provides Python bindings for sklears LabelEncoder,
//! offering scikit-learn compatible label encoding for categorical data.

use super::common::*;
use sklears::preprocessing::LabelEncoder;

/// Python wrapper for LabelEncoder
#[pyclass(name = "LabelEncoder")]
pub struct PyLabelEncoder {
    encoder: Option<LabelEncoder>,
    fitted_encoder: Option<LabelEncoder>,
}

#[pymethods]
impl PyLabelEncoder {
    #[new]
    fn new() -> Self {
        Self {
            encoder: Some(LabelEncoder::new()),
            fitted_encoder: None,
        }
    }

    /// Fit the encoder to the labels
    fn fit(&mut self, y: PyReadonlyArray1<&str>) -> PyResult<()> {
        let y_vec: Vec<String> = y.as_array().iter().map(|&s| s.to_string()).collect();

        // Validate input
        if y_vec.is_empty() {
            return Err(PyValueError::new_err("Input labels must not be empty"));
        }

        let encoder = self
            .encoder
            .take()
            .ok_or_else(|| PyValueError::new_err("Encoder already fitted or invalid state"))?;

        match encoder.fit(&y_vec) {
            Ok(fitted) => {
                self.fitted_encoder = Some(fitted);
                Ok(())
            }
            Err(e) => Err(PyValueError::new_err(format!(
                "Failed to fit encoder: {}",
                e
            ))),
        }
    }

    /// Transform string labels to integer labels
    fn transform(&self, y: PyReadonlyArray1<&str>) -> PyResult<Py<PyArray1<i32>>> {
        let fitted = self
            .fitted_encoder
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Encoder not fitted. Call fit() first."))?;

        let y_vec: Vec<String> = y.as_array().iter().map(|&s| s.to_string()).collect();

        if y_vec.is_empty() {
            return Err(PyValueError::new_err("Input labels must not be empty"));
        }

        match fitted.transform(&y_vec) {
            Ok(encoded) => Python::with_gil(|py| {
                let encoded_array = Array1::from_vec(encoded);
                Ok(PyArray1::from_array(py, &encoded_array).to_owned())
            }),
            Err(e) => Err(PyValueError::new_err(format!(
                "Transformation failed: {}",
                e
            ))),
        }
    }

    /// Fit and transform in one step
    fn fit_transform(&mut self, y: PyReadonlyArray1<&str>) -> PyResult<Py<PyArray1<i32>>> {
        self.fit(y)?;
        self.transform(y)
    }

    /// Transform integer labels back to string labels
    fn inverse_transform(&self, y: PyReadonlyArray1<i32>) -> PyResult<Vec<String>> {
        let fitted = self
            .fitted_encoder
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Encoder not fitted. Call fit() first."))?;

        let y_vec: Vec<i32> = y.as_array().to_vec();

        if y_vec.is_empty() {
            return Err(PyValueError::new_err("Input labels must not be empty"));
        }

        match fitted.inverse_transform(&y_vec) {
            Ok(decoded) => Ok(decoded),
            Err(e) => Err(PyValueError::new_err(format!(
                "Inverse transformation failed: {}",
                e
            ))),
        }
    }

    /// Get the unique classes seen during fit
    #[getter]
    fn classes_(&self) -> PyResult<Vec<String>> {
        let fitted = self
            .fitted_encoder
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Encoder not fitted. Call fit() first."))?;

        Ok(fitted.classes().clone())
    }

    /// Get the number of unique classes
    fn n_classes_(&self) -> PyResult<usize> {
        let fitted = self
            .fitted_encoder
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Encoder not fitted. Call fit() first."))?;

        Ok(fitted.classes().len())
    }
}
