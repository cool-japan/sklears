//! Python bindings for Bayesian Ridge Regression
//!
//! This module provides Python bindings for Bayesian Ridge Regression,
//! offering scikit-learn compatible interfaces with automatic relevance determination
//! and uncertainty quantification using the sklears-linear crate.

use super::common::*;
use pyo3::types::PyDict;
use pyo3::Bound;
use sklears_core::traits::{Fit, Predict, Score, Trained};
use sklears_linear::{BayesianRidge, BayesianRidgeConfig};

/// Python-specific configuration wrapper for BayesianRidge
#[derive(Debug, Clone)]
pub struct PyBayesianRidgeConfig {
    pub max_iter: usize,
    pub tol: f64,
    pub alpha_init: Option<f64>,
    pub lambda_init: Option<f64>,
    pub fit_intercept: bool,
    pub compute_score: bool,
    pub copy_x: bool,
}

impl Default for PyBayesianRidgeConfig {
    fn default() -> Self {
        Self {
            max_iter: 300,
            tol: 1e-3,
            alpha_init: Some(1.0),
            lambda_init: Some(1.0),
            fit_intercept: true,
            compute_score: false,
            copy_x: true,
        }
    }
}

impl From<PyBayesianRidgeConfig> for BayesianRidgeConfig {
    fn from(py_config: PyBayesianRidgeConfig) -> Self {
        BayesianRidgeConfig {
            max_iter: py_config.max_iter,
            tol: py_config.tol,
            alpha_init: py_config
                .alpha_init
                .unwrap_or_else(|| BayesianRidgeConfig::default().alpha_init),
            lambda_init: py_config
                .lambda_init
                .unwrap_or_else(|| BayesianRidgeConfig::default().lambda_init),
            fit_intercept: py_config.fit_intercept,
            compute_score: py_config.compute_score,
        }
    }
}

/// Bayesian ridge regression.
///
/// Fit a Bayesian ridge model. See the Notes section for details on this
/// implementation and the optimization of the regularization parameters
/// lambda (precision of the weights) and alpha (precision of the noise).
///
/// Parameters
/// ----------
/// max_iter : int, default=300
///     Maximum number of iterations. Should be greater than or equal to 1.
///
/// tol : float, default=1e-3
///     Stop the algorithm if w has converged.
///
/// alpha_init : float, default=1.0
///     Initial value for alpha (precision of the weights).
///     If not provided, alpha_init is set to 1.0.
///
/// lambda_init : float, default=1.0
///     Initial value for lambda (precision of the noise).
///     If not provided, lambda_init is set to 1.0.
///
/// fit_intercept : bool, default=True
///     Whether to calculate the intercept for this model.
///     The intercept is not treated as a probabilistic parameter
///     and thus has no associated variance. If set
///     to False, no intercept will be used in calculations
///     (i.e. data is expected to be centered).
///
/// compute_score : bool, default=False
///     If True, compute the log marginal likelihood at each iteration of the
///     optimization.
///
/// copy_X : bool, default=True
///     If True, X will be copied; else, it may be overwritten.
///
/// Attributes
/// ----------
/// coef_ : array-like of shape (n_features,)
///     Coefficients of the regression model (mean of distribution)
///
/// intercept_ : float
///     Independent term in decision function. Set to 0.0 if
///     ``fit_intercept = False``.
///
/// alpha_ : float
///     Estimated precision of the weights.
///
/// lambda_ : float
///     Estimated precision of the noise.
///
/// sigma_ : array-like of shape (n_features, n_features)
///     Estimated variance-covariance matrix of the weights
///
/// scores_ : array-like of shape (n_iter_ + 1,)
///     If computed_score is True, value of the log marginal likelihood (to be
///     maximized) at each iteration of the optimization. The array starts with
///     the value of the log marginal likelihood obtained for the initial values
///     of alpha and lambda and ends with the value obtained for the estimated
///     alpha and lambda.
///
/// n_iter_ : int
///     The actual number of iterations to reach the stopping criterion.
///
/// n_features_in_ : int
///     Number of features seen during fit.
///
/// Examples
/// --------
/// >>> from sklears_python import BayesianRidge
/// >>> import numpy as np
/// >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
/// >>> # y = 1 * x_0 + 2 * x_1 + 3
/// >>> y = np.dot(X, [1, 2]) + 3
/// >>> reg = BayesianRidge()
/// >>> reg.fit(X, y)
/// BayesianRidge()
/// >>> reg.predict([[1, 0]])
/// array([4.])
/// >>> reg.coef_
/// array([1., 2.])
///
/// Notes
/// -----
/// There exist several strategies to perform Bayesian ridge regression. This
/// implementation is based on the algorithm described in Appendix A of
/// (Tipping, 2001) where updates of the regularization parameters are done as
/// suggested in (MacKay, 1992). Note that according to A New
/// View of Automatic Relevance Determination (Wipf and Nagarajan, 2008) these
/// update rules do not guarantee that the marginal likelihood is increasing
/// between two consecutive iterations of the optimization.
///
/// References
/// ----------
/// D. J. C. MacKay, Bayesian Interpolation, Computation and Neural Systems,
/// Vol. 4, No. 3, 1992.
///
/// M. E. Tipping, Sparse Bayesian Learning and the Relevance Vector Machine,
/// Journal of Machine Learning Research, Vol. 1, 2001.
#[pyclass(name = "BayesianRidge")]
pub struct PyBayesianRidge {
    /// Python-specific configuration
    py_config: PyBayesianRidgeConfig,
    /// Trained model instance using the actual sklears-linear implementation
    fitted_model: Option<BayesianRidge<Trained>>,
}

#[pymethods]
impl PyBayesianRidge {
    #[new]
    #[pyo3(signature = (max_iter=300, tol=1e-3, alpha_init=1.0, lambda_init=1.0, fit_intercept=true, compute_score=false, copy_x=true))]
    fn new(
        max_iter: usize,
        tol: f64,
        alpha_init: f64,
        lambda_init: f64,
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

        let py_config = PyBayesianRidgeConfig {
            max_iter,
            tol,
            alpha_init: Some(alpha_init),
            lambda_init: Some(lambda_init),
            fit_intercept,
            compute_score,
            copy_x,
        };

        Ok(Self {
            py_config,
            fitted_model: None,
        })
    }

    /// Fit the Bayesian Ridge regression model
    fn fit(&mut self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        let x_array = pyarray_to_core_array2(x)?;
        let y_array = pyarray_to_core_array1(y)?;

        // Validate input arrays
        validate_fit_arrays(&x_array, &y_array)?;

        // Create sklears-linear model with Bayesian Ridge configuration
        let model = BayesianRidge::new()
            .max_iter(self.py_config.max_iter)
            .tol(self.py_config.tol)
            .fit_intercept(self.py_config.fit_intercept)
            .compute_score(self.py_config.compute_score);

        // Fit the model using sklears-linear's implementation
        match model.fit(&x_array, &y_array) {
            Ok(fitted_model) => {
                self.fitted_model = Some(fitted_model);
                Ok(())
            }
            Err(e) => Err(PyValueError::new_err(format!(
                "Failed to fit Bayesian Ridge model: {:?}",
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

    /// Get estimated precision of weights (alpha)
    #[getter]
    fn alpha_(&self) -> PyResult<f64> {
        let fitted = self
            .fitted_model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted. Call fit() first."))?;

        Ok(fitted.alpha())
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
            "BayesianRidge(max_iter={}, tol={}, alpha_init={:?}, lambda_init={:?}, fit_intercept={}, compute_score={}, copy_X={})",
            self.py_config.max_iter,
            self.py_config.tol,
            self.py_config.alpha_init,
            self.py_config.lambda_init,
            self.py_config.fit_intercept,
            self.py_config.compute_score,
            self.py_config.copy_x
        )
    }
}
