//! Python bindings for StandardScaler
//!
//! This module provides Python bindings for StandardScaler,
//! offering scikit-learn compatible standardization (z-score normalization).

use super::common::*;
use scirs2_core::ndarray::{Array1, Axis};

/// StandardScaler state after fitting
#[derive(Debug, Clone)]
struct StandardScalerState {
    mean: Array1<f64>,
    scale: Array1<f64>,
    var: Array1<f64>,
    n_features: usize,
    n_samples_seen: usize,
}

/// Standardize features by removing the mean and scaling to unit variance.
///
/// The standard score of a sample `x` is calculated as:
///
///     z = (x - u) / s
///
/// where `u` is the mean of the training samples or zero if `with_mean=False`,
/// and `s` is the standard deviation of the training samples or one if
/// `with_std=False`.
///
/// Centering and scaling happen independently on each feature by computing
/// the relevant statistics on the samples in the training set. Mean and
/// standard deviation are then stored to be used on later data using
/// :meth:`transform`.
///
/// Standardization of a dataset is a common requirement for many
/// machine learning estimators: they might behave badly if the
/// individual features do not more or less look like standard normally
/// distributed data (e.g. Gaussian with 0 mean and unit variance).
///
/// Parameters
/// ----------
/// copy : bool, default=True
///     If False, try to avoid a copy and do inplace scaling instead.
///     This is not guaranteed to always work inplace; e.g. if the data is
///     not a NumPy array or scipy.sparse CSR matrix, a copy may still be
///     returned.
///
/// with_mean : bool, default=True
///     If True, center the data before scaling.
///     This does not work (and will raise an exception) when attempted on
///     sparse matrices, because centering them entails building a dense
///     matrix which in common use cases is likely to be too large to fit in
///     memory.
///
/// with_std : bool, default=True
///     If True, scale the data to unit variance (or equivalently,
///     unit standard deviation).
///
/// Attributes
/// ----------
/// scale_ : ndarray of shape (n_features,) or None
///     Per feature relative scaling of the data to achieve zero mean and unit
///     variance. Generally this is calculated using `np.sqrt(var_)`. If a
///     variance is zero, we can't achieve unit variance, and the data is left
///     as-is, giving a scaling factor of 1. `scale_` is equal to `None`
///     when `with_std=False`.
///
/// mean_ : ndarray of shape (n_features,) or None
///     The mean value for each feature in the training set.
///     Equal to ``None`` when ``with_mean=False``.
///
/// var_ : ndarray of shape (n_features,) or None
///     The variance for each feature in the training set. Used to compute
///     `scale_`. Equal to ``None`` when ``with_std=False``.
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
/// >>> from sklears_python import StandardScaler
/// >>> import numpy as np
/// >>> data = [[0, 0], [0, 0], [1, 1], [1, 1]]
/// >>> scaler = StandardScaler()
/// >>> scaler.fit(data)
/// StandardScaler()
/// >>> print(scaler.mean_)
/// [0.5 0.5]
/// >>> print(scaler.transform(data))
/// [[-1. -1.]
///  [-1. -1.]
///  [ 1.  1.]
///  [ 1.  1.]]
/// >>> print(scaler.transform([[2, 2]]))
/// [[3. 3.]]
#[pyclass(name = "StandardScaler")]
pub struct PyStandardScaler {
    copy: bool,
    with_mean: bool,
    with_std: bool,
    state: Option<StandardScalerState>,
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
            state: None,
        }
    }

    /// Compute the mean and std to be used for later scaling.
    ///
    /// Parameters
    /// ----------
    /// X : {array-like, sparse matrix} of shape (n_samples, n_features)
    ///     The data used to compute the mean and standard deviation
    ///     used for later scaling along the features axis.
    ///
    /// y : None
    ///     Ignored.
    ///
    /// sample_weight : array-like of shape (n_samples,), default=None
    ///     Individual weights for each sample.
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

        // Compute mean
        let mean = if self.with_mean {
            x_array.mean_axis(Axis(0)).unwrap()
        } else {
            Array1::zeros(n_features)
        };

        // Compute variance and scale
        let (var, scale) = if self.with_std {
            // Calculate variance: E[(X - mean)^2]
            let mut var = Array1::zeros(n_features);
            for j in 0..n_features {
                let col = x_array.column(j);
                let mean_j = mean[j];
                let sum_sq_diff: f64 = col.iter().map(|&x| (x - mean_j).powi(2)).sum();
                var[j] = sum_sq_diff / n_samples as f64;
            }

            // Calculate scale (std dev), but avoid division by zero
            let scale = var.mapv(|v| {
                let std = v.sqrt();
                if std < 1e-10 {
                    1.0 // Avoid division by zero
                } else {
                    std
                }
            });

            (var, scale)
        } else {
            (Array1::ones(n_features), Array1::ones(n_features))
        };

        self.state = Some(StandardScalerState {
            mean,
            scale,
            var,
            n_features,
            n_samples_seen: n_samples,
        });

        Ok(())
    }

    /// Perform standardization by centering and scaling.
    ///
    /// Parameters
    /// ----------
    /// X : {array-like, sparse matrix} of shape (n_samples, n_features)
    ///     The data used to scale along the features axis.
    ///
    /// copy : bool, default=None
    ///     Copy the input X or not.
    ///
    /// Returns
    /// -------
    /// X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
    ///     Transformed array.
    fn transform(&self, py: Python<'_>, x: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray2<f64>>> {
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Scaler not fitted. Call fit() first."))?;

        let x_array = pyarray_to_core_array2(&x)?;
        validate_transform_array(&x_array, state.n_features)?;

        let mut transformed = x_array.clone();

        // Center the data
        if self.with_mean {
            for j in 0..state.n_features {
                for i in 0..transformed.nrows() {
                    transformed[[i, j]] -= state.mean[j];
                }
            }
        }

        // Scale the data
        if self.with_std {
            for j in 0..state.n_features {
                for i in 0..transformed.nrows() {
                    transformed[[i, j]] /= state.scale[j];
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
        // Create copy of x for transform since fit consumes x
        let x_array = pyarray_to_core_array2(&x)?;
        self.fit(x)?;

        // Transform using the saved x_array
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Scaler not fitted. Call fit() first."))?;

        let mut transformed = x_array.clone();

        // Center the data
        if self.with_mean {
            for j in 0..state.n_features {
                for i in 0..transformed.nrows() {
                    transformed[[i, j]] -= state.mean[j];
                }
            }
        }

        // Scale the data
        if self.with_std {
            for j in 0..state.n_features {
                for i in 0..transformed.nrows() {
                    transformed[[i, j]] /= state.scale[j];
                }
            }
        }

        core_array2_to_py(py, &transformed)
    }

    /// Scale back the data to the original representation.
    ///
    /// Parameters
    /// ----------
    /// X : {array-like, sparse matrix} of shape (n_samples, n_features)
    ///     The data used to scale along the features axis.
    ///
    /// copy : bool, default=None
    ///     Copy the input X or not.
    ///
    /// Returns
    /// -------
    /// X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
    ///     Transformed array.
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

        // Reverse scaling
        if self.with_std {
            for j in 0..state.n_features {
                for i in 0..inverse.nrows() {
                    inverse[[i, j]] *= state.scale[j];
                }
            }
        }

        // Reverse centering
        if self.with_mean {
            for j in 0..state.n_features {
                for i in 0..inverse.nrows() {
                    inverse[[i, j]] += state.mean[j];
                }
            }
        }

        core_array2_to_py(py, &inverse)
    }

    /// The mean value for each feature in the training set.
    #[getter]
    fn mean_(&self, py: Python<'_>) -> PyResult<Py<PyArray1<f64>>> {
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Scaler not fitted. Call fit() first."))?;

        Ok(core_array1_to_py(py, &state.mean))
    }

    /// Per feature relative scaling of the data.
    #[getter]
    fn scale_(&self, py: Python<'_>) -> PyResult<Py<PyArray1<f64>>> {
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Scaler not fitted. Call fit() first."))?;

        Ok(core_array1_to_py(py, &state.scale))
    }

    /// The variance for each feature in the training set.
    #[getter]
    fn var_(&self, py: Python<'_>) -> PyResult<Py<PyArray1<f64>>> {
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Scaler not fitted. Call fit() first."))?;

        Ok(core_array1_to_py(py, &state.var))
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
            "StandardScaler(copy={}, with_mean={}, with_std={})",
            self.copy, self.with_mean, self.with_std
        )
    }
}
