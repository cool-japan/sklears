//! Python bindings for MinMaxScaler
//!
//! This module provides Python bindings for MinMaxScaler,
//! offering scikit-learn compatible min-max normalization.

use super::common::*;
use scirs2_core::ndarray::Array1;

/// MinMaxScaler state after fitting
#[derive(Debug, Clone)]
struct MinMaxScalerState {
    data_min: Array1<f64>,
    data_max: Array1<f64>,
    data_range: Array1<f64>,
    scale: Array1<f64>,
    min_: Array1<f64>,
    n_features: usize,
    n_samples_seen: usize,
    feature_range: (f64, f64),
}

/// Transform features by scaling each feature to a given range.
///
/// This estimator scales and translates each feature individually such
/// that it is in the given range on the training set, e.g. between
/// zero and one.
///
/// The transformation is given by:
///
///     X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
///     X_scaled = X_std * (max - min) + min
///
/// where min, max = feature_range.
///
/// This transformation is often used as an alternative to zero mean,
/// unit variance scaling.
///
/// Parameters
/// ----------
/// feature_range : tuple (min, max), default=(0, 1)
///     Desired range of transformed data.
///
/// copy : bool, default=True
///     Set to False to perform inplace row normalization and avoid a
///     copy (if the input is already a numpy array).
///
/// clip : bool, default=False
///     Set to True to clip transformed values of held-out data to
///     provided `feature range`.
///
/// Attributes
/// ----------
/// min_ : ndarray of shape (n_features,)
///     Per feature adjustment for minimum. Equivalent to
///     ``min - X.min(axis=0) * self.scale_``
///
/// scale_ : ndarray of shape (n_features,)
///     Per feature relative scaling of the data. Equivalent to
///     ``(max - min) / (X.max(axis=0) - X.min(axis=0))``
///
/// data_min_ : ndarray of shape (n_features,)
///     Per feature minimum seen in the data
///
/// data_max_ : ndarray of shape (n_features,)
///     Per feature maximum seen in the data
///
/// data_range_ : ndarray of shape (n_features,)
///     Per feature range ``(data_max_ - data_min_)`` seen in the data
///
/// n_features_in_ : int
///     Number of features seen during :term:`fit`.
///
/// n_samples_seen_ : int
///     The number of samples processed by the estimator.
///     It will be reset on new calls to fit, but increments across
///     ``partial_fit`` calls.
///
/// Examples
/// --------
/// >>> from sklears_python import MinMaxScaler
/// >>> import numpy as np
/// >>> data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
/// >>> scaler = MinMaxScaler()
/// >>> scaler.fit(data)
/// MinMaxScaler()
/// >>> print(scaler.data_max_)
/// [ 1. 18.]
/// >>> print(scaler.transform(data))
/// [[0.   0.  ]
///  [0.25 0.25]
///  [0.5  0.5 ]
///  [1.   1.  ]]
/// >>> print(scaler.transform([[2, 2]]))
/// [[1.5 0. ]]
#[pyclass(name = "MinMaxScaler")]
pub struct PyMinMaxScaler {
    feature_range: (f64, f64),
    copy: bool,
    clip: bool,
    state: Option<MinMaxScalerState>,
}

#[pymethods]
impl PyMinMaxScaler {
    #[new]
    #[pyo3(signature = (feature_range=(0.0, 1.0), copy=true, clip=false))]
    fn new(feature_range: (f64, f64), copy: bool, clip: bool) -> Self {
        Self {
            feature_range,
            copy,
            clip,
            state: None,
        }
    }

    /// Compute the minimum and maximum to be used for later scaling.
    ///
    /// Parameters
    /// ----------
    /// X : {array-like, sparse matrix} of shape (n_samples, n_features)
    ///     The data used to compute the per-feature minimum and maximum
    ///     used for later scaling along the features axis.
    ///
    /// y : None
    ///     Ignored.
    ///
    /// Returns
    /// -------
    /// self : object
    ///     Fitted scaler.
    fn fit(&mut self, x: PyReadonlyArray2<f64>) -> PyResult<()> {
        let x_array = pyarray_to_core_array2(&x)?;
        validate_fit_array(&x_array)?;

        let n_samples = x_array.nrows();
        let n_features = x_array.ncols();

        // Compute min and max for each feature
        let mut data_min = Array1::zeros(n_features);
        let mut data_max = Array1::zeros(n_features);

        for j in 0..n_features {
            let col = x_array.column(j);
            data_min[j] = col.iter().cloned().fold(f64::INFINITY, |a, b| a.min(b));
            data_max[j] = col.iter().cloned().fold(f64::NEG_INFINITY, |a, b| a.max(b));
        }

        // Compute data range
        let data_range = &data_max - &data_min;

        // Compute scale and min_
        let (feature_min, feature_max) = self.feature_range;
        let feature_range = feature_max - feature_min;

        let mut scale = Array1::zeros(n_features);
        let mut min_ = Array1::zeros(n_features);

        for j in 0..n_features {
            if data_range[j].abs() < 1e-10 {
                // Handle constant features
                scale[j] = 1.0;
                min_[j] = feature_min - data_min[j];
            } else {
                scale[j] = feature_range / data_range[j];
                min_[j] = feature_min - data_min[j] * scale[j];
            }
        }

        self.state = Some(MinMaxScalerState {
            data_min,
            data_max,
            data_range,
            scale,
            min_,
            n_features,
            n_samples_seen: n_samples,
            feature_range: self.feature_range,
        });

        Ok(())
    }

    /// Scale features of X according to feature_range.
    ///
    /// Parameters
    /// ----------
    /// X : {array-like, sparse matrix} of shape (n_samples, n_features)
    ///     Input data that will be transformed.
    ///
    /// Returns
    /// -------
    /// Xt : ndarray of shape (n_samples, n_features)
    ///     Transformed data.
    fn transform(&self, py: Python<'_>, x: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray2<f64>>> {
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Scaler not fitted. Call fit() first."))?;

        let x_array = pyarray_to_core_array2(&x)?;
        validate_transform_array(&x_array, state.n_features)?;

        let mut transformed = x_array.clone();

        // Apply scaling: X_scaled = X * scale + min_
        for j in 0..state.n_features {
            for i in 0..transformed.nrows() {
                transformed[[i, j]] = transformed[[i, j]] * state.scale[j] + state.min_[j];

                // Clip values if requested
                if self.clip {
                    let (min_val, max_val) = state.feature_range;
                    transformed[[i, j]] = transformed[[i, j]].clamp(min_val, max_val);
                }
            }
        }

        core_array2_to_py(py, &transformed)
    }

    /// Fit to data, then transform it.
    ///
    /// Fits transformer to `X` and returns a transformed version of `X`.
    ///
    /// Parameters
    /// ----------
    /// X : {array-like, sparse matrix} of shape (n_samples, n_features)
    ///     Input samples.
    ///
    /// y :  array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
    ///     Target values (None for unsupervised transformations).
    ///
    /// Returns
    /// -------
    /// X_new : ndarray array of shape (n_samples, n_features_new)
    ///     Transformed array.
    fn fit_transform(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let x_array = pyarray_to_core_array2(&x)?;
        self.fit(x)?;

        // Transform using the saved x_array
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Scaler not fitted. Call fit() first."))?;

        let mut transformed = x_array.clone();

        // Apply scaling: X_scaled = X * scale + min_
        for j in 0..state.n_features {
            for i in 0..transformed.nrows() {
                transformed[[i, j]] = transformed[[i, j]] * state.scale[j] + state.min_[j];

                // Clip values if requested
                if self.clip {
                    let (min_val, max_val) = state.feature_range;
                    transformed[[i, j]] = transformed[[i, j]].clamp(min_val, max_val);
                }
            }
        }

        core_array2_to_py(py, &transformed)
    }

    /// Undo the scaling of X according to feature_range.
    ///
    /// Parameters
    /// ----------
    /// X : {array-like, sparse matrix} of shape (n_samples, n_features)
    ///     Input data that will be transformed. It cannot be sparse.
    ///
    /// Returns
    /// -------
    /// Xt : ndarray of shape (n_samples, n_features)
    ///     Transformed data.
    fn inverse_transform(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Scaler not fitted. Call fit() first."))?;

        let x_array = pyarray_to_core_array2(&x)?;
        validate_transform_array(&x_array, state.n_features)?;

        let mut inverse = x_array.clone();

        // Reverse scaling: X = (X_scaled - min_) / scale
        for j in 0..state.n_features {
            for i in 0..inverse.nrows() {
                inverse[[i, j]] = (inverse[[i, j]] - state.min_[j]) / state.scale[j];
            }
        }

        core_array2_to_py(py, &inverse)
    }

    /// Per feature minimum seen in the data
    #[getter]
    fn data_min_(&self, py: Python<'_>) -> PyResult<Py<PyArray1<f64>>> {
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Scaler not fitted. Call fit() first."))?;

        Ok(core_array1_to_py(py, &state.data_min))
    }

    /// Per feature maximum seen in the data
    #[getter]
    fn data_max_(&self, py: Python<'_>) -> PyResult<Py<PyArray1<f64>>> {
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Scaler not fitted. Call fit() first."))?;

        Ok(core_array1_to_py(py, &state.data_max))
    }

    /// Per feature range (data_max_ - data_min_) seen in the data
    #[getter]
    fn data_range_(&self, py: Python<'_>) -> PyResult<Py<PyArray1<f64>>> {
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Scaler not fitted. Call fit() first."))?;

        Ok(core_array1_to_py(py, &state.data_range))
    }

    /// Per feature relative scaling of the data
    #[getter]
    fn scale_(&self, py: Python<'_>) -> PyResult<Py<PyArray1<f64>>> {
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Scaler not fitted. Call fit() first."))?;

        Ok(core_array1_to_py(py, &state.scale))
    }

    /// Per feature adjustment for minimum
    #[getter]
    fn min_(&self, py: Python<'_>) -> PyResult<Py<PyArray1<f64>>> {
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Scaler not fitted. Call fit() first."))?;

        Ok(core_array1_to_py(py, &state.min_))
    }

    /// Number of features seen during fit.
    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Scaler not fitted. Call fit() first."))?;

        Ok(state.n_features)
    }

    /// The number of samples processed by the estimator.
    #[getter]
    fn n_samples_seen_(&self) -> PyResult<usize> {
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Scaler not fitted. Call fit() first."))?;

        Ok(state.n_samples_seen)
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "MinMaxScaler(feature_range=({}, {}), copy={}, clip={})",
            self.feature_range.0, self.feature_range.1, self.copy, self.clip
        )
    }
}
