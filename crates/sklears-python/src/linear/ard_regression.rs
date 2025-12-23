//! Python bindings for ARD Regression
//!
//! This module provides Python bindings for Automatic Relevance Determination (ARD) Regression,
//! offering scikit-learn compatible interfaces with automatic feature selection
//! and uncertainty quantification using the sklears-linear crate.

use super::common::*;
use pyo3::types::PyDict;
use pyo3::Bound;
use sklears_core::traits::{Fit, Predict, Score, Trained};
use sklears_linear::{ARDRegression, ARDRegressionConfig};

/// Python-specific configuration wrapper for ARDRegression
#[derive(Debug, Clone)]
pub struct PyARDRegressionConfig {
    pub max_iter: usize,
    pub tol: f64,
    pub alpha_init: Option<f64>,
    pub lambda_init: Option<f64>,
    pub threshold_alpha: f64,
    pub fit_intercept: bool,
    pub compute_score: bool,
    pub copy_x: bool,
}

impl Default for PyARDRegressionConfig {
    fn default() -> Self {
        Self {
            max_iter: 300,
            tol: 1e-3,
            alpha_init: Some(1.0),
            lambda_init: Some(1.0),
            threshold_alpha: 1e10,
            fit_intercept: true,
            compute_score: false,
            copy_x: true,
        }
    }
}

impl From<PyARDRegressionConfig> for ARDRegressionConfig {
    fn from(py_config: PyARDRegressionConfig) -> Self {
        ARDRegressionConfig {
            max_iter: py_config.max_iter,
            tol: py_config.tol,
            alpha_init: py_config
                .alpha_init
                .unwrap_or_else(|| ARDRegressionConfig::default().alpha_init),
            lambda_init: py_config
                .lambda_init
                .unwrap_or_else(|| ARDRegressionConfig::default().lambda_init),
            threshold_alpha: py_config.threshold_alpha,
            fit_intercept: py_config.fit_intercept,
            compute_score: py_config.compute_score,
        }
    }
}

/// Bayesian ARD regression.
///
/// Fit the weights of a regression model, using an ARD prior. The weights of
/// the regression model are assumed to be drawn from an isotropic Gaussian
/// distribution with precision lambda. The shrinkage is data-dependent,
/// and the parameters of the prior are estimated from the data using empirical
/// Bayes approach.
///
/// Parameters
/// ----------
/// max_iter : int, default=300
///     Maximum number of iterations.
///
/// tol : float, default=1e-3
///     Stop the algorithm if w has converged.
///
/// alpha_init : float, default=1.0
///     Initial value for alpha (per-feature precisions).
///     If not provided, alpha_init is 1.0.
///
/// lambda_init : float, default=1.0
///     Initial value for lambda (precision of the noise).
///     If not provided, lambda_init is 1.0.
///
/// threshold_alpha : float, default=1e10
///     Threshold for removing (pruning) weights with high precision from
///     the computation: features with precision higher than this threshold
///     are considered to have zero weight.
///
/// fit_intercept : bool, default=True
///     Whether to calculate the intercept for this model. If set
///     to False, no intercept will be used in calculations
///     (i.e. data is expected to be centered).
///
/// compute_score : bool, default=False
///     If True, compute the objective function at each step of the model.
///
/// copy_X : bool, default=True
///     If True, X will be copied; else, it may be overwritten.
///
/// Attributes
/// ----------
/// coef_ : array-like of shape (n_features,)
///     Coefficients of the regression model (mean of distribution)
///
/// alpha_ : array-like of shape (n_features,)
///     estimated precision of the weights.
///
/// lambda_ : float
///     estimated precision of the noise.
///
/// sigma_ : array-like of shape (n_features, n_features)
///     estimated variance-covariance matrix of the weights
///
/// scores_ : array-like of shape (n_iter_+1,)
///     if computed, value of the objective function (to be maximized)
///     at each iteration of the optimization.
///
/// intercept_ : float
///     Independent term in decision function. Set to 0.0 if
///     ``fit_intercept = False``.
///
/// n_features_in_ : int
///     Number of features seen during fit.
///
/// Examples
/// --------
/// >>> from sklears_python import ARDRegression
/// >>> import numpy as np
/// >>> X = np.array([[1], [2], [3], [4], [5]])
/// >>> y = np.array([1, 2, 3, 4, 5])
/// >>> reg = ARDRegression()
/// >>> reg.fit(X, y)
/// ARDRegression()
/// >>> reg.predict([[3]])
/// array([3.])
///
/// Notes
/// -----
/// ARD performs feature selection by setting the weights of many features
/// to zero, as they are deemed irrelevant. This is particularly useful when
/// the number of features is much larger than the number of samples.
///
/// For polynomial regression, it is recommended to "center" the data by
/// subtracting its mean before fitting the ARD model.
///
/// References
/// ----------
/// D. J. C. MacKay, Bayesian nonlinear modeling for the prediction
/// competition, ASHRAE Transactions, 1994.
///
/// R. Salakhutdinov, Lecture notes on Statistical Machine Learning,
/// http://www.cs.toronto.edu/~rsalakhu/sta4273/notes/Lecture2.pdf
/// Their beta is our ``lambda_``, and their alpha is our ``alpha_``
/// ARD is a little different: only ``lambda_`` is inferred; ``alpha_``
/// is fixed by the user.
#[pyclass(name = "ARDRegression")]
pub struct PyARDRegression {
    /// Python-specific configuration
    py_config: PyARDRegressionConfig,
    /// Trained model instance using the actual sklears-linear implementation
    fitted_model: Option<ARDRegression<Trained>>,
}

#[pymethods]
impl PyARDRegression {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (max_iter=300, tol=1e-3, alpha_init=1.0, lambda_init=1.0, threshold_alpha=1e10, fit_intercept=true, compute_score=false, copy_x=true))]
    fn new(
        max_iter: usize,
        tol: f64,
        alpha_init: f64,
        lambda_init: f64,
        threshold_alpha: f64,
        fit_intercept: bool,
        compute_score: bool,
        copy_x: bool,
    ) -> PyResult<Self> {
        // Validate parameters
        if max_iter == 0 {
            return Err(PyValueError::new_err("max_iter must be greater than 0"));
        }
        if tol <= 0.0 {
            return Err(PyValueError::new_err("tol must be positive"));
        }
        if alpha_init <= 0.0 {
            return Err(PyValueError::new_err("alpha_init must be positive"));
        }
        if lambda_init <= 0.0 {
            return Err(PyValueError::new_err("lambda_init must be positive"));
        }
        if threshold_alpha <= 0.0 {
            return Err(PyValueError::new_err("threshold_alpha must be positive"));
        }

        let py_config = PyARDRegressionConfig {
            max_iter,
            tol,
            alpha_init: Some(alpha_init),
            lambda_init: Some(lambda_init),
            threshold_alpha,
            fit_intercept,
            compute_score,
            copy_x,
        };

        Ok(Self {
            py_config,
            fitted_model: None,
        })
    }

    /// Fit the ARD regression model
    fn fit(&mut self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        let x_array = pyarray_to_core_array2(x)?;
        let y_array = pyarray_to_core_array1(y)?;

        // Validate input arrays
        validate_fit_arrays(&x_array, &y_array)?;

        // Create sklears-linear model with ARD configuration
        let model = ARDRegression::new()
            .max_iter(self.py_config.max_iter)
            .tol(self.py_config.tol)
            .threshold_alpha(self.py_config.threshold_alpha)
            .fit_intercept(self.py_config.fit_intercept);

        // Fit the model using sklears-linear's implementation
        match model.fit(&x_array, &y_array) {
            Ok(fitted_model) => {
                self.fitted_model = Some(fitted_model);
                Ok(())
            }
            Err(e) => Err(PyValueError::new_err(format!(
                "Failed to fit ARD regression model: {:?}",
                e
            ))),
        }
    }

    /// Predict using the fitted model
    fn predict(&self, py: Python<'_>, x: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray1<f64>>> {
        let fitted = self
            .fitted_model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted. Call fit() first."))?;

        let x_array = pyarray_to_core_array2(x)?;
        validate_predict_array(&x_array)?;

        match fitted.predict(&x_array) {
            Ok(predictions) => Ok(core_array1_to_py(py, &predictions)),
            Err(e) => Err(PyValueError::new_err(format!("Prediction failed: {:?}", e))),
        }
    }

    /// Get model coefficients
    #[getter]
    fn coef_(&self, py: Python<'_>) -> PyResult<Py<PyArray1<f64>>> {
        let fitted = self
            .fitted_model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted. Call fit() first."))?;

        Ok(core_array1_to_py(py, fitted.coef()))
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

    /// Get estimated per-feature precisions (alpha)
    #[getter]
    fn alpha_(&self, py: Python<'_>) -> PyResult<Py<PyArray1<f64>>> {
        let fitted = self
            .fitted_model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted. Call fit() first."))?;

        Ok(core_array1_to_py(py, fitted.alpha()))
    }

    /// Get estimated precision of noise (lambda)
    #[getter]
    fn lambda_(&self) -> PyResult<f64> {
        let fitted = self
            .fitted_model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted. Call fit() first."))?;

        Ok(fitted.lambda())
    }

    /// Calculate R² score
    fn score(&self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<f64> {
        let fitted = self
            .fitted_model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted. Call fit() first."))?;

        let x_array = pyarray_to_core_array2(x)?;
        let y_array = pyarray_to_core_array1(y)?;

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
    fn get_params(&self, py: Python<'_>, deep: Option<bool>) -> PyResult<Py<PyDict>> {
        let _deep = deep.unwrap_or(true);

        let dict = PyDict::new(py);

        dict.set_item("max_iter", self.py_config.max_iter)?;
        dict.set_item("tol", self.py_config.tol)?;
        dict.set_item("alpha_init", self.py_config.alpha_init)?;
        dict.set_item("lambda_init", self.py_config.lambda_init)?;
        dict.set_item("threshold_alpha", self.py_config.threshold_alpha)?;
        dict.set_item("fit_intercept", self.py_config.fit_intercept)?;
        dict.set_item("compute_score", self.py_config.compute_score)?;
        dict.set_item("copy_X", self.py_config.copy_x)?;

        Ok(dict.into())
    }

    /// Set parameters for this estimator (sklearn compatibility)
    fn set_params(&mut self, kwargs: &Bound<'_, PyDict>) -> PyResult<()> {
        // Update configuration parameters
        if let Some(max_iter) = kwargs.get_item("max_iter")? {
            let max_iter_val: usize = max_iter.extract()?;
            if max_iter_val == 0 {
                return Err(PyValueError::new_err("max_iter must be greater than 0"));
            }
            self.py_config.max_iter = max_iter_val;
        }
        if let Some(tol) = kwargs.get_item("tol")? {
            let tol_val: f64 = tol.extract()?;
            if tol_val <= 0.0 {
                return Err(PyValueError::new_err("tol must be positive"));
            }
            self.py_config.tol = tol_val;
        }
        if let Some(alpha_init) = kwargs.get_item("alpha_init")? {
            let alpha_init_val: f64 = alpha_init.extract()?;
            if alpha_init_val <= 0.0 {
                return Err(PyValueError::new_err("alpha_init must be positive"));
            }
            self.py_config.alpha_init = Some(alpha_init_val);
        }
        if let Some(lambda_init) = kwargs.get_item("lambda_init")? {
            let lambda_init_val: f64 = lambda_init.extract()?;
            if lambda_init_val <= 0.0 {
                return Err(PyValueError::new_err("lambda_init must be positive"));
            }
            self.py_config.lambda_init = Some(lambda_init_val);
        }
        if let Some(threshold_alpha) = kwargs.get_item("threshold_alpha")? {
            let threshold_alpha_val: f64 = threshold_alpha.extract()?;
            if threshold_alpha_val <= 0.0 {
                return Err(PyValueError::new_err("threshold_alpha must be positive"));
            }
            self.py_config.threshold_alpha = threshold_alpha_val;
        }
        if let Some(fit_intercept) = kwargs.get_item("fit_intercept")? {
            self.py_config.fit_intercept = fit_intercept.extract()?;
        }
        if let Some(compute_score) = kwargs.get_item("compute_score")? {
            self.py_config.compute_score = compute_score.extract()?;
        }
        if let Some(copy_x) = kwargs.get_item("copy_X")? {
            self.py_config.copy_x = copy_x.extract()?;
        }

        // Clear fitted model since config changed
        self.fitted_model = None;

        Ok(())
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "ARDRegression(max_iter={}, tol={}, alpha_init={:?}, lambda_init={:?}, threshold_alpha={}, fit_intercept={}, compute_score={}, copy_X={})",
            self.py_config.max_iter,
            self.py_config.tol,
            self.py_config.alpha_init,
            self.py_config.lambda_init,
            self.py_config.threshold_alpha,
            self.py_config.fit_intercept,
            self.py_config.compute_score,
            self.py_config.copy_x
        )
    }
}
