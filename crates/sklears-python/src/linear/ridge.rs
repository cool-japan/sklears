//! Python bindings for Ridge Regression
//!
//! This module provides Python bindings for Ridge Regression,
//! offering scikit-learn compatible interfaces with L2 regularization
//! using the sklears-linear crate.

use super::common::*;
use numpy::IntoPyArray;
use pyo3::types::PyDict;
use pyo3::Bound;
use sklears_core::traits::{Fit, Predict, Score, Trained};
use sklears_linear::{LinearRegression, LinearRegressionConfig, Penalty};

/// Python-specific configuration wrapper for Ridge
#[derive(Debug, Clone)]
pub struct PyRidgeConfig {
    pub alpha: f64,
    pub fit_intercept: bool,
    pub copy_x: bool,
    pub max_iter: Option<usize>,
    pub tol: Option<f64>,
    pub solver: Option<String>,
    pub positive: bool,
    pub random_state: Option<i32>,
}

impl Default for PyRidgeConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            fit_intercept: true,
            copy_x: true,
            max_iter: None,
            tol: Some(1e-4),
            solver: Some("auto".to_string()),
            positive: false,
            random_state: None,
        }
    }
}

impl From<PyRidgeConfig> for LinearRegressionConfig {
    fn from(py_config: PyRidgeConfig) -> Self {
        let mut config = LinearRegressionConfig::default();
        config.fit_intercept = py_config.fit_intercept;
        config.penalty = Penalty::L2(py_config.alpha);
        if let Some(max_iter) = py_config.max_iter {
            config.max_iter = max_iter;
        }
        if let Some(tol) = py_config.tol {
            config.tol = tol;
        }
        config
    }
}

/// Python wrapper for Ridge regression
#[pyclass(name = "Ridge")]
pub struct PyRidge {
    /// Python-specific configuration
    py_config: PyRidgeConfig,
    /// Trained model instance using the actual sklears-linear implementation
    fitted_model: Option<LinearRegression<Trained>>,
}

#[pymethods]
impl PyRidge {
    #[new]
    #[pyo3(signature = (alpha=1.0, fit_intercept=true, copy_x=true, max_iter=None, tol=1e-4, solver="auto", positive=false, random_state=None))]
    fn new(
        alpha: f64,
        fit_intercept: bool,
        copy_x: bool,
        max_iter: Option<usize>,
        tol: f64,
        solver: &str,
        positive: bool,
        random_state: Option<i32>,
    ) -> Self {
        let py_config = PyRidgeConfig {
            alpha,
            fit_intercept,
            copy_x,
            max_iter,
            tol: Some(tol),
            solver: Some(solver.to_string()),
            positive,
            random_state,
        };

        Self {
            py_config,
            fitted_model: None,
        }
    }

    /// Fit the Ridge regression model
    fn fit(&mut self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        let x_array = x.as_array().to_owned();
        let y_array = y.as_array().to_owned();

        // Validate input arrays
        validate_fit_arrays(&x_array, &y_array)?;

        // Create sklears-linear model with Ridge configuration
        let config = LinearRegressionConfig::from(self.py_config.clone());
        let model = LinearRegression::new()
            .fit_intercept(config.fit_intercept)
            .regularization(self.py_config.alpha);

        // Fit the model using sklears-linear's implementation
        match model.fit(&x_array, &y_array) {
            Ok(fitted_model) => {
                self.fitted_model = Some(fitted_model);
                Ok(())
            }
            Err(e) => Err(PyValueError::new_err(format!(
                "Failed to fit Ridge model: {:?}",
                e
            ))),
        }
    }

    /// Predict using the fitted model
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

    /// Calculate R² score
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

        dict.set_item("alpha", self.py_config.alpha)?;
        dict.set_item("fit_intercept", self.py_config.fit_intercept)?;
        dict.set_item("copy_X", self.py_config.copy_x)?;
        dict.set_item("max_iter", self.py_config.max_iter)?;
        dict.set_item("tol", self.py_config.tol)?;
        dict.set_item("solver", &self.py_config.solver)?;
        dict.set_item("positive", self.py_config.positive)?;
        dict.set_item("random_state", self.py_config.random_state)?;

        Ok(dict.into())
    }

    /// Set parameters for this estimator (sklearn compatibility)
    fn set_params(&mut self, kwargs: &Bound<'_, PyDict>) -> PyResult<()> {
        // Update configuration parameters
        if let Some(alpha) = kwargs.get_item("alpha")? {
            self.py_config.alpha = alpha.extract()?;
        }
        if let Some(fit_intercept) = kwargs.get_item("fit_intercept")? {
            self.py_config.fit_intercept = fit_intercept.extract()?;
        }
        if let Some(copy_x) = kwargs.get_item("copy_X")? {
            self.py_config.copy_x = copy_x.extract()?;
        }
        if let Some(max_iter) = kwargs.get_item("max_iter")? {
            self.py_config.max_iter = max_iter.extract()?;
        }
        if let Some(tol) = kwargs.get_item("tol")? {
            self.py_config.tol = tol.extract()?;
        }
        if let Some(solver) = kwargs.get_item("solver")? {
            let solver_str: String = solver.extract()?;
            self.py_config.solver = Some(solver_str);
        }
        if let Some(positive) = kwargs.get_item("positive")? {
            self.py_config.positive = positive.extract()?;
        }
        if let Some(random_state) = kwargs.get_item("random_state")? {
            self.py_config.random_state = random_state.extract()?;
        }

        // Clear fitted model since config changed
        self.fitted_model = None;

        Ok(())
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "Ridge(alpha={}, fit_intercept={}, copy_X={}, max_iter={:?}, tol={:?}, solver={:?}, positive={}, random_state={:?})",
            self.py_config.alpha,
            self.py_config.fit_intercept,
            self.py_config.copy_x,
            self.py_config.max_iter,
            self.py_config.tol,
            self.py_config.solver,
            self.py_config.positive,
            self.py_config.random_state
        )
    }
}
