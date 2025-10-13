//! Python bindings for Linear Regression
//!
//! This module provides Python bindings for Linear Regression,
//! offering scikit-learn compatible interfaces with high-performance OLS implementation
//! using the sklears-linear crate.

use super::common::*;
use numpy::IntoPyArray;
use pyo3::types::PyDict;
use pyo3::Bound;
use sklears_core::traits::{Fit, Predict, Score, Trained};
use sklears_linear::{LinearRegression, LinearRegressionConfig};

/// Python-specific configuration wrapper
#[derive(Debug, Clone)]
pub struct PyLinearRegressionConfig {
    pub fit_intercept: bool,
    pub copy_x: bool,
    pub n_jobs: Option<i32>,
    pub positive: bool,
}

impl Default for PyLinearRegressionConfig {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            copy_x: true,
            n_jobs: None,
            positive: false,
        }
    }
}

impl From<PyLinearRegressionConfig> for LinearRegressionConfig {
    fn from(py_config: PyLinearRegressionConfig) -> Self {
        let mut config = LinearRegressionConfig::default();
        config.fit_intercept = py_config.fit_intercept;
        // Note: copy_x, n_jobs, and positive are Python-specific and handled at the Python level
        config
    }
}

/// Ordinary least squares Linear Regression.
///
/// LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
/// to minimize the residual sum of squares between the observed targets in
/// the dataset, and the targets predicted by the linear approximation.
///
/// Parameters
/// ----------
/// fit_intercept : bool, default=True
///     Whether to calculate the intercept for this model. If set
///     to False, no intercept will be used in calculations
///     (i.e. data is expected to be centered).
///
/// copy_X : bool, default=True
///     If True, X will be copied; else, it may be overwritten.
///
/// n_jobs : int, default=None
///     The number of jobs to use for the computation. This will only provide
///     speedup in case of sufficiently large problems, that is if firstly
///     `n_targets > 1` and secondly `X` is sparse or if `positive` is set
///     to `True`. ``None`` means 1 unless in a
///     :obj:`joblib.parallel_backend` context. ``-1`` means using all
///     processors.
///
/// positive : bool, default=False
///     When set to ``True``, forces the coefficients to be positive. This
///     option is only supported for dense arrays.
///
/// Attributes
/// ----------
/// coef_ : array of shape (n_features,) or (n_targets, n_features)
///     Estimated coefficients for the linear regression problem.
///     If multiple targets are passed during the fit (y 2D), this
///     is a 2D array of shape (n_targets, n_features), while if only
///     one target is passed, this is a 1D array of length n_features.
///
/// intercept_ : float or array of shape (n_targets,)
///     Independent term in the linear model. Set to 0.0 if
///     `fit_intercept = False`.
///
/// n_features_in_ : int
///     Number of features seen during :term:`fit`.
///
/// Examples
/// --------
/// >>> import numpy as np
/// >>> from sklears_python import LinearRegression
/// >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
/// >>> # y = 1 * x_0 + 2 * x_1 + 3
/// >>> y = np.dot(X, [1, 2]) + 3
/// >>> reg = LinearRegression().fit(X, y)
/// >>> reg.score(X, y)
/// 1.0
/// >>> reg.coef_
/// array([1., 2.])
/// >>> reg.intercept_
/// 3.0...
/// >>> reg.predict(np.array([[3, 5]]))
/// array([16.])
///
/// Notes
/// -----
/// From the implementation point of view, this is just plain Ordinary
/// Least Squares (scipy.linalg.lstsq) or Non Negative Least Squares
/// (scipy.optimize.nnls) wrapped as a predictor object.
#[pyclass(name = "LinearRegression")]
pub struct PyLinearRegression {
    py_config: PyLinearRegressionConfig,
    fitted_model: Option<LinearRegression<Trained>>,
}

#[pymethods]
impl PyLinearRegression {
    #[new]
    #[pyo3(signature = (fit_intercept=true, copy_x=true, n_jobs=None, positive=false))]
    fn new(fit_intercept: bool, copy_x: bool, n_jobs: Option<i32>, positive: bool) -> Self {
        let py_config = PyLinearRegressionConfig {
            fit_intercept,
            copy_x,
            n_jobs,
            positive,
        };

        Self {
            py_config,
            fitted_model: None,
        }
    }

    /// Fit linear model.
    ///
    /// Parameters
    /// ----------
    /// X : {array-like, sparse matrix} of shape (n_samples, n_features)
    ///     Training data.
    ///
    /// y : array-like of shape (n_samples,) or (n_samples, n_targets)
    ///     Target values. Will be cast to X's dtype if necessary.
    ///
    /// sample_weight : array-like of shape (n_samples,), default=None
    ///     Individual weights for each sample
    ///
    /// Returns
    /// -------
    /// self : object
    ///     Fitted Estimator.
    fn fit(&mut self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        let x_array = x.as_array().to_owned();
        let y_array = y.as_array().to_owned();

        // Validate input arrays
        validate_fit_arrays(&x_array, &y_array)?;

        // Create sklears-linear model with configuration
        let config = LinearRegressionConfig::from(self.py_config.clone());
        let model = LinearRegression::new().fit_intercept(config.fit_intercept);

        // Fit the model using sklears-linear's implementation
        match model.fit(&x_array, &y_array) {
            Ok(fitted_model) => {
                self.fitted_model = Some(fitted_model);
                Ok(())
            }
            Err(e) => Err(PyValueError::new_err(format!(
                "Failed to fit model: {:?}",
                e
            ))),
        }
    }

    /// Predict using the linear model.
    ///
    /// Parameters
    /// ----------
    /// X : array-like or sparse matrix, shape (n_samples, n_features)
    ///     Samples.
    ///
    /// Returns
    /// -------
    /// C : array, shape (n_samples,)
    ///     Returns predicted values.
    fn predict(&self, x: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray1<f64>>> {
        let fitted = self
            .fitted_model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted. Call fit() first."))?;

        let x_array = x.as_array().to_owned();
        validate_predict_array(&x_array)?;

        match fitted.predict(&x_array) {
            Ok(predictions) => {
                let py = unsafe { Python::assume_attached() };
                Ok(predictions.into_pyarray(py).into())
            }
            Err(e) => Err(PyValueError::new_err(format!("Prediction failed: {:?}", e))),
        }
    }

    /// Get model coefficients
    #[getter]
    fn coef_(&self) -> PyResult<Py<PyArray1<f64>>> {
        let fitted = self
            .fitted_model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted. Call fit() first."))?;

        let py = unsafe { Python::assume_attached() };
        Ok(fitted.coef().clone().into_pyarray(py).into())
    }

    /// Get model intercept
    #[getter]
    fn intercept_(&self) -> PyResult<f64> {
        let fitted = self
            .fitted_model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted. Call fit() first."))?;

        Ok(fitted.intercept().unwrap_or(0.0))
    }

    /// Return the coefficient of determination of the prediction.
    ///
    /// The coefficient of determination :math:`R^2` is defined as
    /// :math:`(1 - \\frac{SS_{res}}{SS_{tot}})` where :math:`SS_{res} = \\sum_i (y_i - y(x_i))^2`
    /// is the residual sum of squares, and :math:`SS_{tot} = \\sum_i (y_i - \\bar{y})^2`
    /// is the total sum of squares.
    ///
    /// The best possible score is 1.0 and it can be negative (because the
    /// model can be arbitrarily worse). A constant model that always predicts
    /// the expected value of `y`, disregarding the input features, would get
    /// a :math:`R^2` score of 0.0.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Test samples. For some estimators this may be a precomputed
    ///     kernel matrix or a list of generic objects instead with shape
    ///     ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted``
    ///     is the number of samples used in the fitting for the estimator.
    ///
    /// y : array-like of shape (n_samples,) or (n_samples, n_outputs)
    ///     True values for `X`.
    ///
    /// sample_weight : array-like of shape (n_samples,), default=None
    ///     Sample weights.
    ///
    /// Returns
    /// -------
    /// score : float
    ///     :math:`R^2` of ``self.predict(X)`` w.r.t. `y`.
    ///
    /// Notes
    /// -----
    /// The :math:`R^2` score used when calling ``score`` on a regressor uses
    /// ``multioutput='uniform_average'`` from version 0.23 to keep consistent
    /// with default value of :func:`~sklearn.metrics.r2_score`.
    /// This influences the ``score`` method of all the multioutput
    /// regressors (except for
    /// :class:`~sklearn.multioutput.MultiOutputRegressor`).
    fn score(&self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<f64> {
        let fitted = self
            .fitted_model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted. Call fit() first."))?;

        let x_array = x.as_array().to_owned();
        let y_array = y.as_array().to_owned();

        match fitted.score(&x_array, &y_array) {
            Ok(score) => Ok(score),
            Err(e) => Err(PyValueError::new_err(format!(
                "Score calculation failed: {:?}",
                e
            ))),
        }
    }

    /// Get number of features
    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        let fitted = self
            .fitted_model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted. Call fit() first."))?;

        // Infer number of features from coefficient array length
        Ok(fitted.coef().len())
    }

    /// Return parameters for this estimator (sklearn compatibility)
    fn get_params(&self, deep: Option<bool>) -> PyResult<Py<PyDict>> {
        let _deep = deep.unwrap_or(true);

        let py = unsafe { Python::assume_attached() };
        let dict = PyDict::new(py);

        dict.set_item("fit_intercept", self.py_config.fit_intercept)?;
        dict.set_item("copy_X", self.py_config.copy_x)?;
        dict.set_item("n_jobs", self.py_config.n_jobs)?;
        dict.set_item("positive", self.py_config.positive)?;

        Ok(dict.into())
    }

    /// Set parameters for this estimator (sklearn compatibility)
    fn set_params(&mut self, kwargs: &Bound<'_, PyDict>) -> PyResult<()> {
        // Update configuration parameters
        if let Some(fit_intercept) = kwargs.get_item("fit_intercept")? {
            self.py_config.fit_intercept = fit_intercept.extract()?;
        }
        if let Some(copy_x) = kwargs.get_item("copy_X")? {
            self.py_config.copy_x = copy_x.extract()?;
        }
        if let Some(n_jobs) = kwargs.get_item("n_jobs")? {
            self.py_config.n_jobs = n_jobs.extract()?;
        }
        if let Some(positive) = kwargs.get_item("positive")? {
            self.py_config.positive = positive.extract()?;
        }

        // Clear fitted model since config changed
        self.fitted_model = None;

        Ok(())
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "LinearRegression(fit_intercept={}, copy_X={}, n_jobs={:?}, positive={})",
            self.py_config.fit_intercept,
            self.py_config.copy_x,
            self.py_config.n_jobs,
            self.py_config.positive
        )
    }
}
