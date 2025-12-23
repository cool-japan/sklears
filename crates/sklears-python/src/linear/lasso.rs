//! Python bindings for Lasso Regression
//!
//! This module provides Python bindings for Lasso Regression,
//! offering scikit-learn compatible interfaces with L1 regularization
//! using the sklears-linear crate.

use super::common::*;
use pyo3::types::PyDict;
use pyo3::Bound;
use sklears_core::traits::{Fit, Predict, Score, Trained};
use sklears_linear::{LinearRegression, LinearRegressionConfig, Penalty};

/// Python-specific configuration wrapper for Lasso
#[derive(Debug, Clone)]
pub struct PyLassoConfig {
    pub alpha: f64,
    pub fit_intercept: bool,
    pub copy_x: bool,
    pub max_iter: usize,
    pub tol: f64,
    pub warm_start: bool,
    pub positive: bool,
    pub random_state: Option<i32>,
    pub selection: String,
}

impl Default for PyLassoConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            fit_intercept: true,
            copy_x: true,
            max_iter: 1000,
            tol: 1e-4,
            warm_start: false,
            positive: false,
            random_state: None,
            selection: "cyclic".to_string(),
        }
    }
}

impl From<PyLassoConfig> for LinearRegressionConfig {
    fn from(py_config: PyLassoConfig) -> Self {
        LinearRegressionConfig {
            fit_intercept: py_config.fit_intercept,
            penalty: Penalty::L1(py_config.alpha),
            max_iter: py_config.max_iter,
            tol: py_config.tol,
            warm_start: py_config.warm_start,
            ..Default::default()
        }
    }
}

/// Python wrapper for Lasso regression
#[pyclass(name = "Lasso")]
pub struct PyLasso {
    /// Python-specific configuration
    py_config: PyLassoConfig,
    /// Trained model instance using the actual sklears-linear implementation
    fitted_model: Option<LinearRegression<Trained>>,
}

#[pymethods]
impl PyLasso {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (alpha=1.0, fit_intercept=true, copy_x=true, max_iter=1000, tol=1e-4, warm_start=false, positive=false, random_state=None, selection="cyclic"))]
    fn new(
        alpha: f64,
        fit_intercept: bool,
        copy_x: bool,
        max_iter: usize,
        tol: f64,
        warm_start: bool,
        positive: bool,
        random_state: Option<i32>,
        selection: &str,
    ) -> Self {
        let py_config = PyLassoConfig {
            alpha,
            fit_intercept,
            copy_x,
            max_iter,
            tol,
            warm_start,
            positive,
            random_state,
            selection: selection.to_string(),
        };

        Self {
            py_config,
            fitted_model: None,
        }
    }

    /// Fit the Lasso regression model
    fn fit(&mut self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        let x_array = pyarray_to_core_array2(x)?;
        let y_array = pyarray_to_core_array1(y)?;

        // Validate input arrays
        validate_fit_arrays(&x_array, &y_array)?;

        // Create sklears-linear model with Lasso configuration
        let model = LinearRegression::lasso(self.py_config.alpha)
            .fit_intercept(self.py_config.fit_intercept);

        // Fit the model using sklears-linear's implementation
        match model.fit(&x_array, &y_array) {
            Ok(fitted_model) => {
                self.fitted_model = Some(fitted_model);
                Ok(())
            }
            Err(e) => Err(PyValueError::new_err(format!(
                "Failed to fit Lasso model: {:?}",
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

        dict.set_item("alpha", self.py_config.alpha)?;
        dict.set_item("fit_intercept", self.py_config.fit_intercept)?;
        dict.set_item("copy_X", self.py_config.copy_x)?;
        dict.set_item("max_iter", self.py_config.max_iter)?;
        dict.set_item("tol", self.py_config.tol)?;
        dict.set_item("warm_start", self.py_config.warm_start)?;
        dict.set_item("positive", self.py_config.positive)?;
        dict.set_item("random_state", self.py_config.random_state)?;
        dict.set_item("selection", &self.py_config.selection)?;

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
        if let Some(warm_start) = kwargs.get_item("warm_start")? {
            self.py_config.warm_start = warm_start.extract()?;
        }
        if let Some(positive) = kwargs.get_item("positive")? {
            self.py_config.positive = positive.extract()?;
        }
        if let Some(random_state) = kwargs.get_item("random_state")? {
            self.py_config.random_state = random_state.extract()?;
        }
        if let Some(selection) = kwargs.get_item("selection")? {
            let selection_str: String = selection.extract()?;
            self.py_config.selection = selection_str;
        }

        // Clear fitted model since config changed
        self.fitted_model = None;

        Ok(())
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "Lasso(alpha={}, fit_intercept={}, copy_X={}, max_iter={}, tol={}, warm_start={}, positive={}, random_state={:?}, selection='{}')",
            self.py_config.alpha,
            self.py_config.fit_intercept,
            self.py_config.copy_x,
            self.py_config.max_iter,
            self.py_config.tol,
            self.py_config.warm_start,
            self.py_config.positive,
            self.py_config.random_state,
            self.py_config.selection
        )
    }
}
