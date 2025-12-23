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
/// (Currently the 'multinomial' option is supported only by the 'lbfgs',
/// 'sag', 'saga' and 'newton-cg' solvers.)
///
/// This class implements regularized logistic regression using various solvers.
/// **Note that regularization is applied by default**. It can handle both
/// dense and sparse input. Use C-ordered arrays containing
/// 64-bit floats for optimal performance; any other input format will be
/// converted (and copied).
///
/// The 'newton-cg', 'sag', and 'lbfgs' solvers support only L2 regularization
/// with primal formulation, or no regularization. The Elastic-Net regularization
/// is only supported by the 'saga' solver.
///
/// Parameters
/// ----------
/// penalty : {'l1', 'l2', 'elasticnet'}, default='l2'
///     Specify the norm of the penalty:
///
///     - 'l2': add a L2 penalty term and it is the default choice;
///     - 'l1': add a L1 penalty term;
///     - 'elasticnet': both L1 and L2 penalty terms are added.
///
/// tol : float, default=1e-4
///     Tolerance for stopping criteria.
///
/// C : float, default=1.0
///     Inverse of regularization strength; must be a positive float.
///     Like in support vector machines, smaller values specify stronger
///     regularization.
///
/// fit_intercept : bool, default=True
///     Specifies if a constant (a.k.a. bias or intercept) should be
///     added to the decision function.
///
/// class_weight : dict or 'balanced', default=None
///     Weights associated with classes in the form ``{class_label: weight}``.
///     If not given, all classes are supposed to have weight one.
///
///     The "balanced" mode uses the values of y to automatically adjust
///     weights inversely proportional to class frequencies in the input data
///     as ``n_samples / (n_classes * np.bincount(y))``.
///
/// random_state : int, default=None
///     Used when ``solver`` == 'sag', 'saga' to shuffle the
///     data. See :term:`Glossary <random_state>` for details.
///
/// solver : {'lbfgs', 'newton-cg', 'sag', 'saga'}, default='lbfgs'
///
///     Algorithm to use in the optimization problem. Default is 'lbfgs'.
///     To choose a solver, you might want to consider the following aspects:
///
///         - For small datasets, 'lbfgs' is a good choice, whereas 'sag'
///           and 'saga' are faster for large ones;
///         - For multiclass problems, only 'newton-cg', 'sag', 'saga' and
///           'lbfgs' handle multinomial loss.
///
/// max_iter : int, default=100
///     Maximum number of iterations taken for the solvers to converge.
///
/// multi_class : {'auto', 'ovr', 'multinomial'}, default='auto'
///     If the option chosen is 'ovr', then a binary problem is fit for each
///     label. For 'multinomial' the loss minimised is the multinomial loss fit
///     across the entire probability distribution, *even when the data is
///     binary*. 'auto' selects 'ovr' if the data is binary,
///     and otherwise selects 'multinomial'.
///
/// warm_start : bool, default=False
///     When set to True, reuse the solution of the previous call to fit as
///     initialization, otherwise, just erase the previous solution.
///     See :term:`the Glossary <warm_start>`.
///
/// n_jobs : int, default=None
///     Number of CPU cores used when parallelizing over classes if
///     multi_class='ovr'". ``None`` means 1 unless in a
///     :obj:`joblib.parallel_backend` context. ``-1`` means using all
///     processors. See :term:`Glossary <n_jobs>` for more details.
///
/// l1_ratio : float, default=None
///     The Elastic-Net mixing parameter, with ``0 <= l1_ratio <= 1``. Only
///     used if ``penalty='elasticnet'``. Setting ``l1_ratio=0`` is equivalent
///     to using ``penalty='l2'``, while setting ``l1_ratio=1`` is equivalent
///     to using ``penalty='l1'``. For ``0 < l1_ratio <1``, the penalty is a
///     combination of L1 and L2.
///
/// Attributes
/// ----------
/// classes_ : ndarray of shape (n_classes, )
///     A list of class labels known to the classifier.
///
/// coef_ : ndarray of shape (1, n_features) or (n_classes, n_features)
///     Coefficient of the features in the decision function.
///
///     `coef_` is of shape (1, n_features) when the given problem is binary.
///
/// intercept_ : float or ndarray of shape (n_classes,)
///     Intercept (a.k.a. bias) added to the decision function.
///
///     If `fit_intercept` is set to False, the intercept is set to zero.
///     `intercept_` is of shape (1,) when the given problem is binary.
///
/// n_features_in_ : int
///     Number of features seen during :term:`fit`.
///
/// Examples
/// --------
/// >>> from sklears_python import LogisticRegression
/// >>> import numpy as np
/// >>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
/// >>> y = np.array([0, 0, 1, 1])
/// >>> clf = LogisticRegression(random_state=0).fit(X, y)
/// >>> clf.predict(X[:2, :])
/// array([0, 0])
/// >>> clf.predict_proba(X[:2, :])
/// array([[...]])
/// >>> clf.score(X, y)
/// 1.0
///
/// Notes
/// -----
/// The underlying implementation uses optimized solvers from sklears-linear.
///
/// References
/// ----------
/// L-BFGS-B -- Software for Large-scale Bound-constrained Optimization
/// Ciyou Zhu, Richard Byrd, Jorge Nocedal and Jose Luis Morales.
/// http://users.iems.northwestern.edu/~nocedal/lbfgsb.html
///
/// SAG -- Mark Schmidt, Nicolas Le Roux, and Francis Bach
/// Minimizing Finite Sums with the Stochastic Average Gradient
/// https://hal.inria.fr/hal-00860051/document
///
/// SAGA -- Defazio, A., Bach F. & Lacoste-Julien S. (2014).
/// SAGA: A Fast Incremental Gradient Method With Support
/// for Non-Strongly Convex Composite Objectives
/// https://arxiv.org/abs/1407.0202
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
