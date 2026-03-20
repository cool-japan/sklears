//! Python bindings for LabelEncoder
//!
//! This module provides Python bindings for LabelEncoder,
//! offering scikit-learn compatible label encoding for categorical features.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

/// LabelEncoder state after fitting
#[derive(Debug, Clone)]
struct LabelEncoderState {
    classes: Vec<String>,
    class_to_index: HashMap<String, usize>,
}

/// Encode target labels with value between 0 and n_classes-1.
///
/// This transformer should be used to encode target values, *i.e.* `y`, and
/// not the input `X`.
///
/// Attributes
/// ----------
/// classes_ : list of shape (n_classes,)
///     Holds the label for each class.
///
/// Examples
/// --------
/// >>> from sklears_python import LabelEncoder
/// >>> le = LabelEncoder()
/// >>> le.fit(["paris", "paris", "tokyo", "amsterdam"])
/// LabelEncoder()
/// >>> list(le.classes_)
/// ['amsterdam', 'paris', 'tokyo']
/// >>> le.transform(["tokyo", "tokyo", "paris"])
/// [2, 2, 1]
/// >>> list(le.inverse_transform([2, 2, 1]))
/// ['tokyo', 'tokyo', 'paris']
#[pyclass(name = "LabelEncoder")]
pub struct PyLabelEncoder {
    state: Option<LabelEncoderState>,
}

#[pymethods]
impl PyLabelEncoder {
    #[new]
    fn new() -> Self {
        Self { state: None }
    }

    /// Fit label encoder.
    ///
    /// Parameters
    /// ----------
    /// y : array-like of shape (n_samples,)
    ///     Target values.
    ///
    /// Returns
    /// -------
    /// self : returns an instance of self
    ///     Fitted label encoder.
    fn fit(&mut self, y: Vec<String>) -> PyResult<()> {
        if y.is_empty() {
            return Err(PyValueError::new_err("y cannot be empty"));
        }

        // Get unique classes and sort them
        let mut classes: Vec<String> = y
            .into_iter()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        classes.sort();

        // Create mapping from class to index
        let class_to_index: HashMap<String, usize> = classes
            .iter()
            .enumerate()
            .map(|(i, c)| (c.clone(), i))
            .collect();

        self.state = Some(LabelEncoderState {
            classes,
            class_to_index,
        });

        Ok(())
    }

    /// Fit label encoder and return encoded labels.
    ///
    /// Parameters
    /// ----------
    /// y : array-like of shape (n_samples,)
    ///     Target values.
    ///
    /// Returns
    /// -------
    /// y : array-like of shape (n_samples,)
    ///     Encoded labels.
    fn fit_transform(&mut self, y: Vec<String>) -> PyResult<Vec<i64>> {
        self.fit(y.clone())?;
        self.transform(y)
    }

    /// Transform labels to normalized encoding.
    ///
    /// Parameters
    /// ----------
    /// y : array-like of shape (n_samples,)
    ///     Target values.
    ///
    /// Returns
    /// -------
    /// y : array-like of shape (n_samples,)
    ///     Labels as normalized encodings.
    fn transform(&self, y: Vec<String>) -> PyResult<Vec<i64>> {
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("LabelEncoder not fitted. Call fit() first."))?;

        let mut encoded = Vec::with_capacity(y.len());

        for label in y.iter() {
            match state.class_to_index.get(label) {
                Some(&index) => encoded.push(index as i64),
                None => {
                    return Err(PyValueError::new_err(format!(
                        "Unknown label '{}'. Label encoder has only seen: {:?}",
                        label, state.classes
                    )));
                }
            }
        }

        Ok(encoded)
    }

    /// Transform labels back to original encoding.
    ///
    /// Parameters
    /// ----------
    /// y : array-like of shape (n_samples,)
    ///     Target values.
    ///
    /// Returns
    /// -------
    /// y : array-like of shape (n_samples,)
    ///     Original encoding.
    fn inverse_transform(&self, y: Vec<i64>) -> PyResult<Vec<String>> {
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("LabelEncoder not fitted. Call fit() first."))?;

        let mut decoded = Vec::with_capacity(y.len());

        for &index in y.iter() {
            if index < 0 || index >= state.classes.len() as i64 {
                return Err(PyValueError::new_err(format!(
                    "Index {} is out of bounds for {} classes",
                    index,
                    state.classes.len()
                )));
            }
            decoded.push(state.classes[index as usize].clone());
        }

        Ok(decoded)
    }

    /// Get the classes seen during fit
    #[getter]
    fn classes_(&self) -> PyResult<Vec<String>> {
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("LabelEncoder not fitted. Call fit() first."))?;

        Ok(state.classes.clone())
    }

    /// String representation
    fn __repr__(&self) -> String {
        "LabelEncoder()".to_string()
    }
}
