//! Python bindings for Logistic Regression
//!
//! This module provides Python bindings for Logistic Regression,
//! offering scikit-learn compatible interfaces for binary classification
//! using the sklears-linear crate.

use super::common::*;
use pyo3::types::PyDict;
use pyo3::Bound;
use sklears_core::traits::{Fit, Predict, PredictProba, Score, Trained};
use sklears_linear::{LogisticRegression, LogisticRegressionConfig, Penalty, Solver};

/// Python-specific configuration wrapper for LogisticRegression
#[derive(Debug, Clone)]
pub struct PyLogisticRegressionConfig {
    pub penalty: String,
    pub c: f64,
    pub fit_intercept: bool,
    pub max_iter: usize,
    pub tol: f64,
    pub solver: String,
    pub random_state: Option<i32>,
    pub class_weight: Option<String>,
    pub multi_class: String,
    pub warm_start: bool,
    pub n_jobs: Option<i32>,
    pub l1_ratio: Option<f64>,
}

impl Default for PyLogisticRegressionConfig {
    fn default() -> Self {
        Self {
            penalty: "l2".to_string(),
            c: 1.0,
            fit_intercept: true,
            max_iter: 100,
            tol: 1e-4,
            solver: "lbfgs".to_string(),
            random_state: None,
            class_weight: None,
            multi_class: "auto".to_string(),
            warm_start: false,
            n_jobs: None,
            l1_ratio: None,
        }
    }
}

impl From<PyLogisticRegressionConfig> for LogisticRegressionConfig {
    fn from(py_config: PyLogisticRegressionConfig) -> Self {
        // Convert penalty string to Penalty enum
        let penalty = match py_config.penalty.as_str() {
            "l1" => Penalty::L1(1.0 / py_config.c),
            "l2" => Penalty::L2(1.0 / py_config.c),
            "elasticnet" => Penalty::ElasticNet {
                alpha: 1.0 / py_config.c,
                l1_ratio: py_config.l1_ratio.unwrap_or(0.5),
            },
            _ => Penalty::L2(1.0 / py_config.c), // Default to L2
        };

        // Convert solver string to Solver enum
        let solver = match py_config.solver.as_str() {
            "lbfgs" => Solver::Lbfgs,
            "sag" => Solver::Sag,
            "saga" => Solver::Saga,
            "newton-cg" => Solver::Newton,
            _ => Solver::Auto, // Default to Auto
        };

        LogisticRegressionConfig {
            penalty,
            solver,
            max_iter: py_config.max_iter,
            tol: py_config.tol,
            fit_intercept: py_config.fit_intercept,
            random_state: py_config.random_state.map(|s| s as u64),
        }
    }
}

/// Logistic Regression (aka logit, MaxEnt) classifier.
///
/// In the multiclass case, the training algorithm uses the one-vs-rest (OvR)
/// scheme if the 'multi_class' option is set to 'ovr', and uses the
/// cross-entropy loss if the 'multi_class' option is set to 'multinomial'.
///
/// This class implements regularized logistic regression using various solvers.
/// **Note that regularization is applied by default**.
///
/// # Parameters
///
/// - `penalty` - One of "l1", "l2", "elasticnet". Default: "l2"
/// - `tol` - Tolerance for stopping criteria. Default: 1e-4
/// - `c` - Inverse of regularization strength. Default: 1.0
/// - `fit_intercept` - Whether to add bias term. Default: true
/// - `solver` - One of "lbfgs", "newton-cg", "sag", "saga". Default: "lbfgs"
/// - `max_iter` - Maximum iterations. Default: 100
/// - `multi_class` - One of "auto", "ovr", "multinomial". Default: "auto"
/// - `random_state` - Random seed for reproducibility
/// - `l1_ratio` - Elastic-Net mixing parameter (0 to 1)
///
/// # References
///
/// - L-BFGS-B: <http://users.iems.northwestern.edu/~nocedal/lbfgsb.html>
/// - SAG: <https://hal.inria.fr/hal-00860051/document>
/// - SAGA: <https://arxiv.org/abs/1407.0202>
#[pyclass(name = "LogisticRegression")]
pub struct PyLogisticRegression {
    py_config: PyLogisticRegressionConfig,
    fitted_model: Option<LogisticRegression<Trained>>,
}

#[pymethods]
impl PyLogisticRegression {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (penalty="l2", dual=false, tol=1e-4, c=1.0, fit_intercept=true, intercept_scaling=1.0, class_weight=None, random_state=None, solver="lbfgs", max_iter=100, multi_class="auto", verbose=0, warm_start=false, n_jobs=None, l1_ratio=None))]
    fn new(
        penalty: &str,
        dual: bool,
        tol: f64,
        c: f64,
        fit_intercept: bool,
        intercept_scaling: f64,
        class_weight: Option<&str>,
        random_state: Option<i32>,
        solver: &str,
        max_iter: usize,
        multi_class: &str,
        verbose: i32,
        warm_start: bool,
        n_jobs: Option<i32>,
        l1_ratio: Option<f64>,
    ) -> Self {
        // Note: Some parameters are sklearn-specific and don't directly map to our implementation
        let _dual = dual;
        let _intercept_scaling = intercept_scaling;
        let _verbose = verbose;

        let py_config = PyLogisticRegressionConfig {
            penalty: penalty.to_string(),
            c,
            fit_intercept,
            max_iter,
            tol,
            solver: solver.to_string(),
            random_state,
            class_weight: class_weight.map(|s| s.to_string()),
            multi_class: multi_class.to_string(),
            warm_start,
            n_jobs,
            l1_ratio,
        };

        Self {
            py_config,
            fitted_model: None,
        }
    }

    /// Fit the logistic regression model
    fn fit(&mut self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        let x_array = pyarray_to_core_array2(x)?;
        let y_array = pyarray_to_core_array1(y)?;

        // Validate input arrays
        validate_fit_arrays(&x_array, &y_array)?;

        // Create sklears-linear model with Logistic Regression configuration
        let model = LogisticRegression::new()
            .max_iter(self.py_config.max_iter)
            .fit_intercept(self.py_config.fit_intercept);

        // Apply penalty if specified
        let model = match self.py_config.penalty.as_str() {
            "l1" => model.penalty(Penalty::L1(1.0 / self.py_config.c)),
            "l2" => model.penalty(Penalty::L2(1.0 / self.py_config.c)),
            "elasticnet" => model.penalty(Penalty::ElasticNet {
                alpha: 1.0 / self.py_config.c,
                l1_ratio: self.py_config.l1_ratio.unwrap_or(0.5),
            }),
            _ => model, // Default (no additional penalty)
        };

        // Apply solver if specified
        let model = match self.py_config.solver.as_str() {
            "lbfgs" => model.solver(Solver::Lbfgs),
            "sag" => model.solver(Solver::Sag),
            "saga" => model.solver(Solver::Saga),
            "newton-cg" => model.solver(Solver::Newton),
            _ => model.solver(Solver::Auto),
        };

        // Apply random state if specified
        let model = if let Some(rs) = self.py_config.random_state {
            model.random_state(rs as u64)
        } else {
            model
        };

        // Fit the model using sklears-linear's implementation
        match model.fit(&x_array, &y_array) {
            Ok(fitted_model) => {
                self.fitted_model = Some(fitted_model);
                Ok(())
            }
            Err(e) => Err(PyValueError::new_err(format!(
                "Failed to fit Logistic Regression model: {:?}",
                e
            ))),
        }
    }

    /// Predict class labels for samples
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

    /// Predict class probabilities for samples
    fn predict_proba(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let fitted = self
            .fitted_model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted. Call fit() first."))?;

        let x_array = pyarray_to_core_array2(x)?;
        validate_predict_array(&x_array)?;

        match fitted.predict_proba(&x_array) {
            Ok(probabilities) => core_array2_to_py(py, &probabilities),
            Err(e) => Err(PyValueError::new_err(format!(
                "Probability prediction failed: {:?}",
                e
            ))),
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

    /// Get unique class labels
    #[getter]
    fn classes_(&self, py: Python<'_>) -> PyResult<Py<PyArray1<f64>>> {
        let fitted = self
            .fitted_model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted. Call fit() first."))?;

        Ok(core_array1_to_py(py, fitted.classes()))
    }

    /// Calculate accuracy score
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

        dict.set_item("penalty", &self.py_config.penalty)?;
        dict.set_item("C", self.py_config.c)?;
        dict.set_item("fit_intercept", self.py_config.fit_intercept)?;
        dict.set_item("max_iter", self.py_config.max_iter)?;
        dict.set_item("tol", self.py_config.tol)?;
        dict.set_item("solver", &self.py_config.solver)?;
        dict.set_item("random_state", self.py_config.random_state)?;
        dict.set_item("class_weight", &self.py_config.class_weight)?;
        dict.set_item("multi_class", &self.py_config.multi_class)?;
        dict.set_item("warm_start", self.py_config.warm_start)?;
        dict.set_item("n_jobs", self.py_config.n_jobs)?;
        dict.set_item("l1_ratio", self.py_config.l1_ratio)?;

        Ok(dict.into())
    }

    /// Set parameters for this estimator (sklearn compatibility)
    fn set_params(&mut self, kwargs: &Bound<'_, PyDict>) -> PyResult<()> {
        // Update configuration parameters
        if let Some(penalty) = kwargs.get_item("penalty")? {
            let penalty_str: String = penalty.extract()?;
            self.py_config.penalty = penalty_str;
        }
        if let Some(c) = kwargs.get_item("C")? {
            self.py_config.c = c.extract()?;
        }
        if let Some(fit_intercept) = kwargs.get_item("fit_intercept")? {
            self.py_config.fit_intercept = fit_intercept.extract()?;
        }
        if let Some(max_iter) = kwargs.get_item("max_iter")? {
            self.py_config.max_iter = max_iter.extract()?;
        }
        if let Some(tol) = kwargs.get_item("tol")? {
            self.py_config.tol = tol.extract()?;
        }
        if let Some(solver) = kwargs.get_item("solver")? {
            let solver_str: String = solver.extract()?;
            self.py_config.solver = solver_str;
        }
        if let Some(random_state) = kwargs.get_item("random_state")? {
            self.py_config.random_state = random_state.extract()?;
        }
        if let Some(class_weight) = kwargs.get_item("class_weight")? {
            let weight_str: Option<String> = class_weight.extract()?;
            self.py_config.class_weight = weight_str;
        }
        if let Some(multi_class) = kwargs.get_item("multi_class")? {
            let multi_class_str: String = multi_class.extract()?;
            self.py_config.multi_class = multi_class_str;
        }
        if let Some(warm_start) = kwargs.get_item("warm_start")? {
            self.py_config.warm_start = warm_start.extract()?;
        }
        if let Some(n_jobs) = kwargs.get_item("n_jobs")? {
            self.py_config.n_jobs = n_jobs.extract()?;
        }
        if let Some(l1_ratio) = kwargs.get_item("l1_ratio")? {
            self.py_config.l1_ratio = l1_ratio.extract()?;
        }

        // Clear fitted model since config changed
        self.fitted_model = None;

        Ok(())
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "LogisticRegression(penalty='{}', C={}, fit_intercept={}, max_iter={}, tol={}, solver='{}', random_state={:?}, multi_class='{}')",
            self.py_config.penalty,
            self.py_config.c,
            self.py_config.fit_intercept,
            self.py_config.max_iter,
            self.py_config.tol,
            self.py_config.solver,
            self.py_config.random_state,
            self.py_config.multi_class
        )
    }
}
